"""Pack the HY-Motion Lite text encoders and the fixed system prompt for native inference."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import onnx
import torch
from transformers import AutoModel, AutoTokenizer, CLIPTextModel
from torch import nn


def prepare(checkout, qwen, clip, out):
    sys.path.insert(0, str(checkout.resolve()))
    from hymotion.network.text_encoders.text_encoder import LLM_ENCODER_LAYOUT

    torch.set_num_threads(8)
    out.mkdir(exist_ok=True, parents=True)
    for name, source in [("qwen", qwen), ("clip", clip)]:
        tokenizer = json.loads((source / "tokenizer.json").read_text())["model"]
        merges = [
            " ".join(pair) if isinstance(pair, list) else pair
            for pair in tokenizer["merges"]
        ]
        payload = {"model": {"vocab": tokenizer["vocab"], "merges": merges}}
        with (out / (name + "-tokens.json")).open("x") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    tokenizer = AutoTokenizer.from_pretrained(qwen)
    template = LLM_ENCODER_LAYOUT["qwen3"]["template"]
    text = tokenizer.apply_chat_template(
        [template[0], {"role": "user", "content": "<BOC>"}],
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=False,
    )
    prefix, suffix = text.split("<BOC>")
    prefix = tokenizer.encode(prefix)
    if len(prefix) != 101 or tokenizer.encode(suffix) != [151645, 198]:
        raise ValueError("Unexpected Qwen3 human-motion system template")
    model = AutoModel.from_pretrained(qwen, dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        cache = model(input_ids=torch.tensor([prefix]), use_cache=True).past_key_values

    def write_bf(f, t):
        f.write(t.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes())

    with (out / "qwen.bf16").open("xb") as f:
        f.write(b"HYQ1")
        write_bf(f, model.embed_tokens.weight)
        for i, layer in enumerate(model.layers):
            a = layer.self_attn
            for norm in [
                layer.input_layernorm,
                layer.post_attention_layernorm,
                a.q_norm,
                a.k_norm,
            ]:
                f.write(norm.weight.detach().float().numpy().astype("<f4").tobytes())
            kv = cache.layers[i]
            for t in (kv.keys, kv.values):
                f.write(
                    t[0]
                    .transpose(0, 1)
                    .contiguous()
                    .float()
                    .numpy()
                    .astype("<f4")
                    .tobytes()
                )
            for linear in (
                a.q_proj,
                a.k_proj,
                a.v_proj,
                a.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            ):
                write_bf(f, linear.weight)
        f.write(model.norm.weight.detach().float().numpy().astype("<f4").tobytes())
    del model, cache
    prepare_clip(clip, out)


def prepare_clip(clip, out):
    model = CLIPTextModel.from_pretrained(clip).eval()
    model = getattr(model, "text_model", model)

    class ClipText(nn.Module):
        def __init__(self):
            super().__init__()
            self.text = model
            self.register_buffer(
                "mask", torch.triu(torch.full((77, 77), float("-inf")), diagonal=1)
            )

        def forward(self, ids):
            x = (
                self.text.embeddings.token_embedding(ids)
                + self.text.embeddings.position_embedding.weight[None]
            )
            for layer in self.text.encoder.layers:
                a = layer.self_attn
                norm = layer.layer_norm1(x)
                q, k, v = [
                    proj(norm).reshape(1, 77, 12, 64).transpose(1, 2)
                    for proj in (a.q_proj, a.k_proj, a.v_proj)
                ]
                weights = (q @ k.transpose(-1, -2) * 0.125 + self.mask).softmax(-1)
                x = x + a.out_proj((weights @ v).transpose(1, 2).reshape(1, 77, 768))
                y = layer.mlp.fc1(layer.layer_norm2(x))
                x = x + layer.mlp.fc2(y * torch.sigmoid(1.702 * y))
            return self.text.final_layer_norm(x)

    path = out / "clip-text.onnx"
    if path.exists():
        raise FileExistsError(path)
    torch.onnx.export(
        ClipText().eval(),
        (torch.zeros(1, 77, dtype=torch.int64),),
        str(path),
        input_names=["ids"],
        output_names=["states"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(path))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["checkout", "qwen", "clip", "output"]:
        p.add_argument(name, type=Path)
    args = p.parse_args()
    from hymotion_assets import staged_assets

    with staged_assets(
        args.output,
        ["qwen-tokens.json", "clip-tokens.json", "qwen.bf16", "clip-text.onnx"],
    ) as stage:
        prepare(args.checkout, args.qwen, args.clip, stage)
