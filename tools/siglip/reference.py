#!/usr/bin/env python3
"""Transformers reference dump for `flint.ml.siglip` parity (`tests/test_flint_v5_siglip.mojo`).

    python tools/siglip/reference.py MODEL_DIR OUT_DIR [--images DIR] [--corpus FILE]

MODEL_DIR is a Hugging Face SigLIP / SigLIP 2 folder (tools/siglip/fetch.py). OUT_DIR receives

    images/*.png          deterministic synthetic test images (RGB and RGBA, up- and downscaled)
    pixels.f32            the processor's pixel_values, channels-last [N, S, S, 3]
    img_pooled.f32        vision pooler_output (before L2 normalisation) [N, D]
    vis_hidden0.f32       post-LayerNorm vision hidden states of image 0 [T, D]
    ids.i32               processor input_ids (padding="max_length", max_length=64) [M, 64]
    txt_pooled.f32        text pooler_output [M, D]
    txt_hidden0.f32       final-LayerNorm text hidden states of text 0 [64, D]
    logits.f32            logits_per_text [M, N]
    tok_ids.i32 / tok_len.i32   tokenizer ids (no specials, no padding) for every corpus string
    index.json            images, texts, corpus, shapes, logit_scale / logit_bias, processor class

Everything runs in float32 on the CPU. Needs torch, transformers (>= 4.49; tested with 5.24),
pillow, numpy (and sentencepiece / tokenizers, which transformers pulls in).
Run the test with FLINT_SIGLIP_DIR=MODEL_DIR FLINT_SIGLIP_REF=OUT_DIR.
"""
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor, AutoTokenizer

torch.set_grad_enabled(False)

TEXTS = [
    "a photo of a red circle on a blue background",
    "a green square",
    "a rainbow gradient",
    "noise",
    "ein rotes Auto",                       # German
    "un coffre au trésor en bois",          # French
    "猫の写真",                              # Japanese
    "一把生锈的中世纪剑",                     # Chinese
    "деревянный ящик",                      # Russian
    "the quick brown fox jumps over the lazy dog, repeatedly and with great enthusiasm, until "
    "the sun sets over the distant mountains and the stars begin to appear one by one in the "
    "darkening sky above the quiet village far away from the city",   # > 64 tokens: truncation
]

CORPUS = [
    "a photo of a cat", "A photo of a Cat.", "  two  spaces ", "tab\tseparated", "line\nbreak",
    "émoji 🐱 1234", "NaÏve façade ﬁ ① ｆｕｌｌｗｉｄｔｈ", "<start_of_turn>user hi<end_of_turn>",
    "3.14159 + 2.71828", "x", " ", "", "Ελληνικά", "हिन्दी में बिल्ली", "ภาษาไทย", "Tiếng Việt",
    "صورة قطة", "고양이 사진", "sci-fi crate with glowing panels", "medieval sword (iron) — rusty",
]


def synth_images(dst):
    os.makedirs(dst, exist_ok=True)
    rng = np.random.default_rng(1234)
    out = []
    im = Image.new("RGB", (320, 240), (30, 60, 200))
    ImageDraw.Draw(im).ellipse((90, 50, 230, 190), fill=(220, 30, 30))
    out.append(("circle", im))
    im = Image.new("RGB", (500, 380), (250, 250, 250))
    ImageDraw.Draw(im).rectangle((120, 80, 380, 300), fill=(20, 170, 60))
    out.append(("square", im))
    x = np.linspace(0, 1, 160)[None, :, None]
    y = np.linspace(0, 1, 120)[:, None, None]
    g = np.concatenate([np.broadcast_to(x, (120, 160, 1)), np.broadcast_to(y, (120, 160, 1)),
                        np.broadcast_to(1 - x, (120, 160, 1))], 2)
    out.append(("gradient", Image.fromarray((g * 255).astype(np.uint8))))   # upscaled
    out.append(("noise", Image.fromarray(rng.integers(0, 256, (231, 263, 3), dtype=np.uint8))))
    a = rng.integers(0, 256, (200, 200, 4), dtype=np.uint8)
    a[..., 3] = np.linspace(0, 255, 200).astype(np.uint8)[None, :]
    out.append(("rgba", Image.fromarray(a, "RGBA")))
    paths = []
    for name, im in out:
        p = os.path.join(dst, name + ".png")
        im.save(p)
        paths.append(p)
    return paths


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    model_dir, out = args[0], args[1]
    extra = None
    corpus = CORPUS
    if "--images" in args:
        extra = args[args.index("--images") + 1]
    if "--corpus" in args:
        corpus = open(args[args.index("--corpus") + 1], encoding="utf-8").read().split("\n")
    os.makedirs(out, exist_ok=True)
    paths = synth_images(os.path.join(out, "images"))
    if extra:
        paths += sorted(os.path.join(extra, f) for f in os.listdir(extra) if f.lower().endswith(".png"))

    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    images = [Image.open(p) for p in paths]
    # The SigLIP 2 processors leave do_convert_rgb unset, so RGBA is composited over white here
    # exactly as flint.ml.siglip.image does (and transformers' convert_to_rgb would).
    images = [Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im.convert("RGBA")).convert("RGB")
              if im.mode in ("RGBA", "LA", "P") else im.convert("RGB") for im in images]
    inputs = proc(text=TEXTS, images=images, padding="max_length", max_length=64, truncation=True,
                  return_tensors="pt")
    pv = inputs["pixel_values"].float()
    ids = inputs["input_ids"]
    vo = model.vision_model(pixel_values=pv)
    to = model.text_model(input_ids=ids)
    o = model(input_ids=ids, pixel_values=pv)

    def w(name, arr, dt=np.float32):
        np.ascontiguousarray(arr, dtype=dt).tofile(os.path.join(out, name))

    w("pixels.f32", pv.permute(0, 2, 3, 1).numpy())
    w("img_pooled.f32", vo.pooler_output.numpy())
    w("vis_hidden0.f32", vo.last_hidden_state[0].numpy())
    w("ids.i32", ids.numpy(), np.int32)
    w("txt_pooled.f32", to.pooler_output.numpy())
    w("txt_hidden0.f32", to.last_hidden_state[0].numpy())
    w("logits.f32", o.logits_per_text.numpy())
    t_ids, t_len = [], []
    for line in corpus:
        e = tok(line, add_special_tokens=False)["input_ids"]
        t_ids += e
        t_len.append(len(e))
    w("tok_ids.i32", np.array(t_ids), np.int32)
    w("tok_len.i32", np.array(t_len), np.int32)
    index = {
        "images": [os.path.relpath(p, out) if p.startswith(out) else p for p in paths],
        "texts": TEXTS,
        "image_size": int(pv.shape[-1]),
        "dim": int(vo.pooler_output.shape[-1]),
        "tokens": int(vo.last_hidden_state.shape[1]),
        "logit_scale": float(model.logit_scale.item()),
        "logit_bias": float(model.logit_bias.item()),
        "image_processor": type(proc.image_processor).__name__,
        "tokenizer": type(tok).__name__,
        "corpus": corpus,
    }
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    probs = torch.sigmoid(o.logits_per_text)
    print(json.dumps({k: v for k, v in index.items() if k not in ("texts", "corpus")}, ensure_ascii=False))
    for t in range(len(TEXTS)):
        best = int(probs[t].argmax())
        print("%-50s -> %-10s p=%.4f" % (TEXTS[t][:50], os.path.basename(paths[best]), probs[t, best]))


if __name__ == "__main__":
    main()
