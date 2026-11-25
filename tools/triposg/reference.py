"""PyTorch reference dumps for `flint.ml.triposg` parity checks.

Runs the original TripoSG modules (VAST-AI-Research/TripoSG, MIT) in float32 on the CPU and
writes raw little-endian float32 files that `tests/test_flint_v5_triposg.mojo` compares against:

  prepared.bin   uint8 [H, W, 3]   image after background keying, crop and padding (shape in meta)
  pixels.bin     [3, 224, 224]     DINOv2 input (BitImageProcessor: PIL bicubic 256, crop 224)
  cond.bin       [257, 1024]       DINOv2-L last_hidden_state
  noise.bin      [2048, 64]        initial latents (torch.Generator seed)
  v_XX.bin       [2048, 64]        guided velocity at step XX
  x_XX.bin       [2048, 64]        latents after step XX
  sdf.bin        [(R+1)^3]         VAE SDF logits on the dense grid linspace(-1.005, 1.005, R+1)^3, ij order
  mesh_v.bin / mesh_f.bin          skimage marching cubes (level 0) of that grid, in world units
  meta.json                        shapes, seed, steps, guidance, timings

RMBG-1.4 (the upstream background remover) is non-commercial and is NOT used: the image is
keyed the same way Flint does it (valid alpha channel, else a border flood fill against the
median border colour), so both sides start from the same foreground mask.

Usage (weights from HF VAST-AI/TripoSG; source = a TripoSG checkout):
  python -I tools/triposg/reference.py SRC WEIGHTS IMAGE OUT [--seed 42] [--steps 50]
         [--guidance 7.0] [--grid 128] [--save-every 1] [--threads 16] [--latents FILE]
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def is_valid_alpha(alpha, min_ratio=0.01):
    hist, _ = np.histogram(alpha, bins=20, range=(0, 256))
    m = alpha.shape[0] * alpha.shape[1] * min_ratio
    return hist[0] >= m and hist[-1] >= m


def key_background(rgb, tol=24):
    """Foreground mask (bool [H, W]) of an opaque image: pixels connected to the border whose
    colour stays within `tol` (max channel difference) of the median border colour are background."""
    h, w, _ = rgb.shape
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0).astype(np.int32)
    bg = np.array([int(np.median(border[:, c])) for c in range(3)], dtype=np.int32)
    near = np.abs(rgb.astype(np.int32) - bg[None, None, :]).max(axis=2) <= tol
    seen = np.zeros((h, w), dtype=bool)
    stack = []
    for x in range(w):
        for y in (0, h - 1):
            if near[y, x] and not seen[y, x]:
                seen[y, x] = True
                stack.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if near[y, x] and not seen[y, x]:
                seen[y, x] = True
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and near[yy, xx] and not seen[yy, xx]:
                seen[yy, xx] = True
                stack.append((yy, xx))
    return ~seen


def remove_small(mask, min_size=200):
    """Drops 4-connected foreground components smaller than `min_size` pixels."""
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    out = mask.copy()
    cur = 0
    for y0 in range(h):
        for x0 in range(w):
            if mask[y0, x0] and lab[y0, x0] == 0:
                cur += 1
                lab[y0, x0] = cur
                comp = [(y0, x0)]
                i = 0
                while i < len(comp):
                    y, x = comp[i]
                    i += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and lab[yy, xx] == 0:
                            lab[yy, xx] = cur
                            comp.append((yy, xx))
                if len(comp) < min_size:
                    for y, x in comp:
                        out[y, x] = False
    return out


def prepare(path, padding_ratio=0.1):
    """TripoSG `load_image` (alpha path) with Flint's keying in place of RMBG; white background."""
    from PIL import Image
    im = Image.open(path)
    rgba = np.array(im.convert("RGBA"))
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    if im.mode in ("RGBA", "LA") and is_valid_alpha(alpha):
        a = alpha.astype(np.float32) / 255.0
    else:
        a = remove_small(key_background(rgb)).astype(np.float32)
    rows, cols = np.where(a > 0)
    if rows.size == 0:
        raise ValueError("empty foreground")
    x, y = cols.min(), rows.min()
    w, h = cols.max() - x + 1, rows.max() - y + 1
    img = rgb.astype(np.float32) / 255.0
    img = img * a[..., None] + 1.0 * (1.0 - a[..., None])
    if w > h:
        px = int(w * padding_ratio)
        py = int(px + (w - h) / 2)
    else:
        py = int(h * padding_ratio)
        px = int(py + (h - w) / 2)
    crop = img[y:y + h, x:x + w]
    out = np.ones((h + 2 * py, w + 2 * px, 3), dtype=np.float32)
    out[py:py + h, px:px + w] = crop
    return (out * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("weights")
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--run-steps", type=int, default=-1, help="stop after this many steps (parity of a prefix)")
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--latents", default="", help="decode these latents (.bin [2048, 64]) instead of sampling")
    args = ap.parse_args()

    sys.path.insert(0, args.src)
    import torch
    from PIL import Image
    from transformers import BitImageProcessor, Dinov2Model
    from triposg.models.autoencoders import TripoSGVAEModel
    from triposg.models.transformers import TripoSGDiTModel
    from triposg.schedulers import RectifiedFlowScheduler

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    meta = {"seed": args.seed, "steps": args.steps, "guidance": args.guidance, "grid": args.grid}

    def dump(name, t):
        a = t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        a.astype(a.dtype if a.dtype == np.uint8 else np.float32).tofile(os.path.join(args.out, name))

    W = args.weights
    with torch.no_grad():
        if args.latents:
            latents = torch.from_numpy(np.fromfile(args.latents, dtype=np.float32).reshape(1, 2048, 64))
        else:
            prepared = prepare(args.image)
            meta["prepared_shape"] = list(prepared.shape)
            dump("prepared.bin", prepared)
            proc = BitImageProcessor.from_pretrained(os.path.join(W, "feature_extractor_dinov2"))
            pixels = proc(Image.fromarray(prepared), return_tensors="pt").pixel_values
            dump("pixels.bin", pixels[0])
            t0 = time.time()
            dino = Dinov2Model.from_pretrained(os.path.join(W, "image_encoder_dinov2")).float().eval()
            cond = dino(pixels).last_hidden_state
            meta["dino_s"] = time.time() - t0
            dump("cond.bin", cond[0])
            del dino

            dit = TripoSGDiTModel.from_pretrained(os.path.join(W, "transformer")).float().eval()
            sched = RectifiedFlowScheduler.from_pretrained(os.path.join(W, "scheduler"))
            sched.set_timesteps(args.steps)
            g = torch.Generator().manual_seed(args.seed)
            latents = torch.randn((1, 2048, 64), generator=g, dtype=torch.float32)
            dump("noise.bin", latents[0])
            embeds = torch.cat([torch.zeros_like(cond), cond], dim=0)
            n_run = len(sched.timesteps) if args.run_steps < 0 else args.run_steps
            t0 = time.time()
            for i, t in enumerate(sched.timesteps[:n_run]):
                inp = torch.cat([latents] * 2)
                v = dit(inp, t.expand(2), encoder_hidden_states=embeds, return_dict=False)[0]
                vu, vc = v.chunk(2)
                v = vu + args.guidance * (vc - vu)
                latents = sched.step(v, t, latents, return_dict=False)[0]
                dump("v_%02d.bin" % i, v[0])
                dump("x_%02d.bin" % i, latents[0])
                print("step %d t=%.1f %.1fs" % (i, float(t), time.time() - t0), flush=True)
            meta["dit_s"] = time.time() - t0
            meta["run_steps"] = n_run
            del dit
            if n_run < len(sched.timesteps):
                json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
                return

        vae = TripoSGVAEModel.from_pretrained(os.path.join(W, "vae")).float().eval()
        R = args.grid
        ax = np.linspace(-1.005, 1.005, R + 1, dtype=np.float32)
        xs, ys, zs = np.meshgrid(ax, ax, ax, indexing="ij")
        pts = torch.from_numpy(np.stack([xs, ys, zs], -1).reshape(1, -1, 3))
        t0 = time.time()
        sdf = vae.decode(latents, sampled_points=pts, num_chunks=20000).sample[0, :, 0]
        meta["vae_s"] = time.time() - t0
        dump("sdf.bin", sdf)
        grid = sdf.numpy().reshape(R + 1, R + 1, R + 1)
        from skimage import measure
        v, f, _, _ = measure.marching_cubes(grid, 0.0, method="lewiner")
        v = v / R * 2.01 - 1.005
        dump("mesh_v.bin", v.astype(np.float32))
        np.ascontiguousarray(f.astype(np.int32)).tofile(os.path.join(args.out, "mesh_f.bin"))
        meta["mesh_vertices"] = int(v.shape[0])
        meta["mesh_faces"] = int(f.shape[0])
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
