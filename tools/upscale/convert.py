#!/usr/bin/env python3
"""Convert Real-ESRGAN `.pth` checkpoints to `.safetensors` for `flint.ml.realesrgan`.

    python tools/upscale/convert.py DIR

Every `DIR/*.pth` (RealESRGAN_x4plus, RealESRGAN_x4plus_anime_6B, realesr-general-x4v3,
realesr-general-wdn-x4v3, realesr-animevideov3) is written next to it as `NAME.safetensors`
with the original parameter names, float32, plus a `__metadata__` entry naming the
architecture (`arch`: `rrdb` or `compact`) and its hyper-parameters, which the Mojo loader reads.
Needs torch (CPU) and numpy; the safetensors file is written by hand, no extra package.
The weights stay outside the repository.
"""
import json
import os
import struct
import sys

import numpy as np
import torch


def _state(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params"):
        if isinstance(ck, dict) and key in ck:
            return ck[key]
    return ck


def describe(sd):
    """Architecture hyper-parameters read back from the parameter shapes."""
    if "conv_first.weight" in sd:
        blocks = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("body."))
        num_feat = sd["conv_first.weight"].shape[0]
        num_in = sd["conv_first.weight"].shape[1]
        grow = sd["body.0.rdb1.conv1.weight"].shape[0]
        # RRDBNet pixel-unshuffles for scale 2 (x4 channels) and 1 (x16): the input conv sees them.
        scale = {3: 4, 12: 2, 48: 1}.get(num_in, 4)
        return {"arch": "rrdb", "num_feat": num_feat, "num_block": blocks, "num_grow_ch": grow,
                "num_in_ch": 3, "num_out_ch": sd["conv_last.weight"].shape[0], "scale": scale}
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("body.")})
    last = idx[-1]
    num_feat = sd["body.0.weight"].shape[0]
    out_ch = sd["body.%d.weight" % last].shape[0]
    num_out = 3
    scale = int(round((out_ch / num_out) ** 0.5))
    convs = sum(1 for i in idx if sd["body.%d.weight" % i].dim() == 4) - 2
    act = "prelu" if sd["body.1.weight"].dim() == 1 else "relu"
    return {"arch": "compact", "num_feat": num_feat, "num_conv": convs, "upscale": scale,
            "num_in_ch": sd["body.0.weight"].shape[1], "num_out_ch": num_out, "act_type": act}


def write_safetensors(path, tensors, meta):
    header = {"__metadata__": {k: str(v) for k, v in meta.items()}}
    blobs = []
    off = 0
    for name, t in tensors.items():
        a = np.ascontiguousarray(t.detach().cpu().float().numpy())
        b = a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def main():
    d = sys.argv[1]
    for name in sorted(os.listdir(d)):
        if not name.endswith(".pth"):
            continue
        sd = _state(os.path.join(d, name))
        meta = describe(sd)
        out = os.path.join(d, name[:-4] + ".safetensors")
        write_safetensors(out, sd, meta)
        print("[upscale] %s -> %s %s" % (name, os.path.basename(out), meta))


if __name__ == "__main__":
    main()
