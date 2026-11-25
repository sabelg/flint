#!/usr/bin/env python3
"""Fetch the diffusion checkpoints used by flint.ml.diffusion into DIR.

    python tools/diffusion/fetch.py DIR [stablematerials] [matfuse] [stablematerials-ddim] [code] [--fp32]

`code` fetches the models' Hugging Face pipeline code (StableMaterials `pipeline.py`, MatFuse
`pipeline_matfuse.py`, condition encoder and VQ model) into DIR/code/, which only
tools/diffusion/reference.py needs to produce the diffusers parity dumps.

Weights never enter the repository. Each `.safetensors` is streamed tensor by
tensor with HTTP range requests and (by default) stored as float16, which
halves the disk/RAM footprint; Flint and tools/diffusion/reference.py both
upcast to float32, so parity is unaffected. Configs and tokenizer files are
copied verbatim.

Licences (read before use, see licenses/):
  gvecchio/StableMaterials       openrail (OpenRAIL-M use restrictions)
  gvecchio/MatFuse               MIT
  sentence-transformers/clip-ViT-B-16 (OpenAI CLIP ViT-B/16)  MIT
"""
import json
import os
import struct
import sys
import urllib.request

HF = "https://huggingface.co"

SETS = {
    "stablematerials": [
        ("gvecchio/StableMaterials", "stablematerials", [
            "model_index.json", "scheduler/scheduler_config.json",
            "tokenizer/vocab.json", "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json",
            "processor/preprocessor_config.json",
            "text_encoder/config.json", "text_encoder/model.safetensors",
            "vision_encoder/config.json", "vision_encoder/model.safetensors",
            "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
            "unet_lcm/config.json", "unet_lcm/diffusion_pytorch_model.safetensors",
        ]),
    ],
    "stablematerials-ddim": [
        ("gvecchio/StableMaterials", "stablematerials", [
            "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
        ]),
    ],
    "matfuse": [
        ("gvecchio/MatFuse", "matfuse", [
            "model_index.json", "scheduler/scheduler_config.json",
            "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
            "condition_encoder/config.json",
            "condition_encoder/diffusion_pytorch_model.safetensors",
        ]),
        ("sentence-transformers/clip-ViT-B-16", "matfuse/clip", [
            "0_CLIPModel/config.json", "0_CLIPModel/model.safetensors",
            "0_CLIPModel/vocab.json", "0_CLIPModel/merges.txt",
            "0_CLIPModel/tokenizer_config.json",
        ]),
    ],
    "code": [
        ("gvecchio/StableMaterials", "code/StableMaterials", ["pipeline.py"]),
        ("gvecchio/MatFuse", "code/MatFuse", [
            "pipeline_matfuse.py", "condition_encoder/condition_encoders.py", "vae/vae_matfuse.py",
        ]),
    ],
}


def _get(url, rng=None):
    req = urllib.request.Request(url, headers={"User-Agent": "flint-fetch"})
    if rng is not None:
        req.add_header("Range", "bytes=%d-%d" % rng)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _f32_to_f16(raw):
    import numpy as np
    return np.frombuffer(raw, dtype=np.float32).astype(np.float16).tobytes()


def fetch_safetensors(url, dst, fp16):
    n = struct.unpack("<Q", _get(url, (0, 7)))[0]
    header = json.loads(_get(url, (8, 8 + n - 1)))
    meta = header.pop("__metadata__", None)
    names = sorted(header, key=lambda k: header[k]["data_offsets"][0])
    out_hdr = {}
    off = 0
    for k in names:
        t = header[k]
        a, b = t["data_offsets"]
        size = b - a
        dt = t["dtype"]
        if fp16 and dt == "F32":
            size //= 2
            dt = "F16"
        out_hdr[k] = {"dtype": dt, "shape": t["shape"], "data_offsets": [off, off + size]}
        off += size
    if meta is not None:
        out_hdr["__metadata__"] = meta
    hb = json.dumps(out_hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    base = 8 + n
    tmp = dst + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        # Coalesce consecutive tensors into ~64 MB range requests.
        i = 0
        while i < len(names):
            j = i
            a0 = header[names[i]]["data_offsets"][0]
            while j + 1 < len(names) and header[names[j + 1]]["data_offsets"][1] - a0 < (64 << 20):
                j += 1
            b0 = header[names[j]]["data_offsets"][1]
            blob = _get(url, (base + a0, base + b0 - 1)) if b0 > a0 else b""
            for k in names[i:j + 1]:
                a, b = header[k]["data_offsets"]
                raw = blob[a - a0:b - a0]
                if fp16 and header[k]["dtype"] == "F32":
                    raw = _f32_to_f16(raw)
                f.write(raw)
            i = j + 1
            sys.stdout.write("\r  %s %5.1f%%" % (os.path.basename(dst), 100.0 * b0 / max(1, off * (2 if fp16 else 1))))
            sys.stdout.flush()
    os.replace(tmp, dst)
    sys.stdout.write("\n")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fp16 = "--fp32" not in sys.argv
    if not args:
        print(__doc__)
        sys.exit(1)
    root = args[0]
    sets = args[1:] or ["stablematerials", "matfuse"]
    for s in sets:
        for repo, sub, files in SETS[s]:
            for fn in files:
                dst = os.path.join(root, sub, fn.replace("0_CLIPModel/", ""))
                if os.path.exists(dst):
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                url = "%s/%s/resolve/main/%s" % (HF, repo, fn)
                print("fetch", repo, fn)
                if fn.endswith(".safetensors"):
                    fetch_safetensors(url, dst, fp16)
                else:
                    data = _get(url)
                    with open(dst, "wb") as f:
                        f.write(data)


if __name__ == "__main__":
    main()
