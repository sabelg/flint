"""Export a compact HY-Motion Lite flow network in separately loadable stages."""

import argparse, sys
from pathlib import Path
import numpy as np
import torch
import onnx
import yaml
from torch import nn
from onnx import helper as H, numpy_helper as NH


def export(name, module, args, inputs, outputs, out):
    path = out / (name + ".onnx")
    if path.exists():
        raise FileExistsError(path)
    torch.onnx.export(
        module.eval(),
        args,
        str(path),
        input_names=inputs,
        output_names=outputs,
        opset_version=17,
        dynamo=False,
    )
    model = onnx.load(path)
    # The text refiner's diagonal is constant for the exported token window.
    for node in list(model.graph.node):
        if node.op_type == "EyeLike":
            model.graph.initializer.append(
                NH.from_array(np.eye(32, dtype=bool), node.output[0])
            )
            model.graph.node.remove(node)
    onnx.save(model, path)
    onnx.checker.check_model(model)


def prepare(checkout, weights, out):
    sys.path.insert(0, str(checkout.resolve()))
    from hymotion.network.hymotion_mmdit import HunyuanMotionMMDiT
    from hymotion.network.bricks import RMSNorm

    torch.set_num_threads(5)

    def rms(self, x):
        x = x.float()
        return (
            x
            * torch.rsqrt(
                (x * x).sum(-1, keepdim=True) / self.weight.shape[0] + self.eps
            )
            * self.weight
        )

    def gelu(self, x):
        if self.approximate != "tanh":
            return x * (1 + torch.erf(x * 0.7071067811865476)) * 0.5
        return x * torch.sigmoid(1.5957691216057308 * (x + 0.044715 * x * x * x))

    RMSNorm.forward = rms
    nn.GELU.forward = gelu
    opts = yaml.safe_load((weights / "config.yml").read_text())["network_module_args"]
    model = HunyuanMotionMMDiT(**opts).eval()
    state = torch.load(
        weights / "latest.ckpt", weights_only=True, map_location="cpu", mmap=True
    )["model_state_dict"]
    model.load_state_dict(
        {
            k.removeprefix("motion_transformer."): v
            for k, v in state.items()
            if k.startswith("motion_transformer.")
        }
    )
    M, T = 60, 32
    B = 2
    x = torch.zeros(B, M, 201)
    c = torch.zeros(B, T, 4096)
    v = torch.zeros(B, 1, 768)
    t = torch.zeros(B)
    a = torch.zeros(B, 1, 1024)
    m = torch.zeros(B, M, 1024)
    z = torch.zeros(B, T, 1024)
    mask = torch.ones(B, T, dtype=torch.bool)
    penalties = torch.zeros(B, 1, 1, M + T)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            for n in [
                "timestep_encoder",
                "vtxt_encoder",
                "ctxt_encoder",
                "text_refiner",
                "input_encoder",
            ]:
                setattr(self, n, getattr(model, n))

        def forward(self, x, c, v, t, mask):
            t = t.float()
            return (
                self.input_encoder(x),
                self.text_refiner(x=self.ctxt_encoder(c), t=t, mask=mask),
                self.timestep_encoder(t) + self.vtxt_encoder(v),
            )

    class Block(nn.Module):
        def __init__(self, block, single):
            super().__init__()
            self.block = block
            self.single = single
            # Omit padded motion rows while keeping the text's original rotary positions.
            cos, sin = block.rotary_emb.freqs_cis
            block.rotary_emb.freqs_cis = tuple(
                torch.cat([w[:M], w[360 : 360 + T]]) for w in (cos, sin)
            )
            bias = torch.zeros(1, 1, M + T, M + T)
            bias[:, :, M:, :M] = float("-inf")
            self.register_buffer("bias", bias)

        def forward(self, x, z, a, penalty):
            if self.single:
                y = self.block(torch.cat([x, z], 1), M, a, self.bias + penalty)
                return y[:, :M], y[:, M:]
            return self.block(x, z, a, self.bias + penalty)

    out.mkdir(exist_ok=True, parents=True)
    export(
        "head",
        Head(),
        (x, c, v, t, mask),
        ["x", "context", "pooled", "time", "valid"],
        ["motion", "text", "adapter"],
        out,
    )
    for idx, block in enumerate([*model.double_blocks, *model.single_blocks]):
        export(
            "block-" + str(idx),
            Block(block, idx >= len(model.double_blocks)),
            (m, z, a, penalties),
            ["x", "z", "a", "penalty"],
            ["motion", "text"],
            out,
        )
    export("tail", model.final_layer, (m, a), ["x", "a"], ["flow"], out)
    for n in ["mean", "std", "null_vtxt_feat", "null_ctxt_input"]:
        state[n].float().contiguous().numpy().astype("<f4").tofile(out / (n + ".bin"))


def prepare_cue(checkout, qwen, clip, prompt, path):
    if path.exists():
        raise FileExistsError(path)
    sys.path.insert(0, str(checkout.resolve()))
    from hymotion.network.text_encoders import text_encoder as te

    te.LLM_ENCODER_LAYOUT["qwen3"]["module_path"] = str(qwen)
    te.SENTENCE_EMB_LAYOUT["clipl"]["module_path"] = str(clip)
    encoder = te.HYTextModel(max_length_llm=128).eval()
    with torch.inference_mode():
        pooled, context, length = encoder.encode([prompt])
    count = int(length.item())
    if not 1 <= count <= 32:
        raise ValueError(
            "This runtime supports 1–32 cue tokens after the system prompt"
        )
    with path.open("xb") as f:
        f.write(b"HYC1")
        f.write(count.to_bytes(4, "little"))
        f.write(pooled.float().contiguous().numpy().astype("<f4").tobytes())
        f.write(context[0, :count].float().contiguous().numpy().astype("<f4").tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model")
    for name in ["checkout", "weights", "output"]:
        model.add_argument(name, type=Path)
    cue = commands.add_parser("cue")
    for name in ["checkout", "qwen", "clip"]:
        cue.add_argument(name, type=Path)
    cue.add_argument("prompt")
    cue.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "model":
        from hymotion_assets import staged_assets

        files = [
            "head.onnx",
            "tail.onnx",
            "mean.bin",
            "std.bin",
            "null_ctxt_input.bin",
            "null_vtxt_feat.bin",
        ]
        files.extend(f"block-{index}.onnx" for index in range(18))
        with staged_assets(args.output, files) as stage:
            prepare(args.checkout, args.weights, stage)
    else:
        prepare_cue(args.checkout, args.qwen, args.clip, args.prompt, args.output)
