#!/usr/bin/env python3
"""PyTorch reference dumps for `flint.ml.realesrgan` parity checks.

    python tools/upscale/reference.py WEIGHTS OUT [--size W H] [--tile T] [--image IMG.png]

WEIGHTS holds the `.pth` files from tools/upscale/fetch.sh. For each checkpoint found, the
network is run in float32 on the CPU on the same input and OUT receives raw little-endian
float32 arrays plus `index.json` (name -> shape):

  input.f32                  [H, W, 3] RGB in 0..1 (from --image, else a fixed synthetic pattern)
  <model>.f32                [4H, 4W, 3] network output, channels-last, before clamping
  <model>_tile.f32           the same through Real-ESRGAN's `tile_process` (tile T, tile_pad 10)
With --fixtures DIR: tiny RRDBNet / SRVGGNetCompact nets with closed-form weights (`tiny_weights`,
not trained, no upstream data) as safetensors plus their outputs (whole and tiled) on a 13 x 10
input, which tests/test_flint_v5_upscale.mojo checks without any download.

The two architectures are written out here from basicsr's RRDBNet (Apache-2.0) and Real-ESRGAN's
SRVGGNetCompact (BSD-3-Clause) so that no basicsr install is needed; they load the released state
dicts unchanged (strict=True), which pins the structure. Verification only, never shipped.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, os.cpu_count() or 1))


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


MODELS = {
    "RealESRGAN_x4plus": lambda: RRDBNet(num_block=23),
    "RealESRGAN_x4plus_anime_6B": lambda: RRDBNet(num_block=6),
    "realesr-general-x4v3": lambda: SRVGGNetCompact(num_conv=32),
    "realesr-general-wdn-x4v3": lambda: SRVGGNetCompact(num_conv=32),
    "realesr-animevideov3": lambda: SRVGGNetCompact(num_conv=16),
}


def load(path, net):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params"):
        if key in ck:
            ck = ck[key]
            break
    net.load_state_dict(ck, strict=True)
    return net.eval()


def tile_process(model, img, scale, tile, tile_pad):
    """Real-ESRGAN `RealESRGANer.tile_process`, verbatim in structure."""
    b, c, h, w = img.shape
    out = img.new_zeros((b, c, h * scale, w * scale))
    tiles_x = math.ceil(w / tile)
    tiles_y = math.ceil(h / tile)
    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x, ofs_y = x * tile, y * tile
            sx, ex = ofs_x, min(ofs_x + tile, w)
            sy, ey = ofs_y, min(ofs_y + tile, h)
            sxp, exp_ = max(sx - tile_pad, 0), min(ex + tile_pad, w)
            syp, eyp = max(sy - tile_pad, 0), min(ey + tile_pad, h)
            t = model(img[:, :, syp:eyp, sxp:exp_])
            otx, oty = (sx - sxp) * scale, (sy - syp) * scale
            out[:, :, sy * scale:ey * scale, sx * scale:ex * scale] = \
                t[:, :, oty:oty + (ey - sy) * scale, otx:otx + (ex - sx) * scale]
    return out


def synthetic(w, h):
    """Smooth gradients, a hard diagonal edge and a checker: the same formula as the Mojo test."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 0.5 + 0.5 * np.sin(x * 0.21 + y * 0.07)
    g = (x + 2 * y) / (w + 2 * h)
    b = np.where((x - y) > (w - h) / 2, 0.9, 0.15) * np.where(((x // 4 + y // 4) % 2) == 0, 1.0, 0.6)
    return np.stack([r, g, b], -1).astype(np.float32)


def tiny_weights(sd_like, seed):
    """Closed-form weights: w[i] = 0.25 * sin(0.37 * (i + 1) + seed) / sqrt(fan_in), bias /4."""
    out = {}
    for k, (name, t) in enumerate(sd_like.items()):
        n = t.numel()
        i = torch.arange(n, dtype=torch.float64)
        fan = t[0].numel() if t.dim() > 1 else 1
        v = 0.25 * torch.sin(0.37 * (i + 1) + seed + k) / math.sqrt(fan)
        if name.endswith("bias") or t.dim() == 1:
            v = v * 0.25
        if t.dim() == 1 and name.endswith("weight"):
            v = 0.1 + 0.1 * torch.sin(i + k)    # PReLU slopes in (0, 0.2)
        out[name] = v.reshape(t.shape).float()
    return out


def save(out_dir, index, name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(os.path.join(out_dir, name + ".f32"))
    index[name] = list(arr.shape)


def run(net, x):
    t = torch.from_numpy(x).permute(2, 0, 1)[None]
    return net(t)[0].permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--size", nargs=2, type=int, default=[64, 48])
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--image", default="")
    ap.add_argument("--fixtures", default="", help="also write the tiny-net test fixtures here")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index = {}
    if a.image:
        from PIL import Image
        x = np.asarray(Image.open(a.image).convert("RGB"), dtype=np.float32) / 255.0
    else:
        x = synthetic(*a.size)
    save(a.out, index, "input", x)

    if a.fixtures:
        # Download-free nets for the always-on Mojo test (tests/fixtures/upscale).
        import importlib.util
        spec = importlib.util.spec_from_file_location("convert", os.path.join(os.path.dirname(__file__), "convert.py"))
        conv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conv)
        os.makedirs(a.fixtures, exist_ok=True)
        tx = synthetic(13, 10)
        tx.tofile(os.path.join(a.fixtures, "tiny_input.f32"))
        rr = RRDBNet(num_feat=8, num_block=2, num_grow_ch=4)
        rr.load_state_dict(tiny_weights(rr.state_dict(), 0.5))
        conv.write_safetensors(os.path.join(a.fixtures, "tiny_rrdb.safetensors"), rr.state_dict(), {"arch": "rrdb"})
        run(rr, tx).tofile(os.path.join(a.fixtures, "tiny_rrdb_out.f32"))
        t = torch.from_numpy(tx).permute(2, 0, 1)[None]
        tile_process(rr, t, 4, 5, 2)[0].permute(1, 2, 0).numpy().tofile(os.path.join(a.fixtures, "tiny_rrdb_tile.f32"))
        cp = SRVGGNetCompact(num_feat=8, num_conv=3, upscale=4)
        cp.load_state_dict(tiny_weights(cp.state_dict(), 1.5))
        conv.write_safetensors(os.path.join(a.fixtures, "tiny_compact.safetensors"), cp.state_dict(), {"arch": "compact"})
        run(cp, tx).tofile(os.path.join(a.fixtures, "tiny_compact_out.f32"))

    for name, make in MODELS.items():
        p = os.path.join(a.weights, name + ".pth")
        if not os.path.exists(p):
            continue
        net = load(p, make())
        save(a.out, index, name, run(net, x))
        t = torch.from_numpy(x).permute(2, 0, 1)[None]
        save(a.out, index, name + "_tile", tile_process(net, t, 4, a.tile, 10)[0].permute(1, 2, 0).numpy())
        print("[reference]", name)
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
