"""PyTorch reference dumps for `flint.ml.depth` parity checks.

Runs the original networks in float32 on the CPU and writes raw little-endian float32 files that
`tests/test_flint_v5_depth.mojo` (FLINT_DEPTH_PARITY=OUT) and `examples/depth --parity OUT`
compare against. Each model gets the network input its own pipeline builds, so the Mojo side can
be checked on identical inputs (network parity) and, separately, on its own preprocessing.

  meta.json      kind, shapes (S, H, W network input; orig_h, orig_w), timings, scalars
  input.f32      [S, 3, H, W] normalised network input
  --- Depth Anything V2 (transformers DepthAnythingForDepthEstimation, DA2 repo transform)
  raw.f32        [H, W]           predicted_depth (relative disparity) at the network size
  depth.f32      [orig_h, orig_w] bilinear (align_corners) to the image size
  --- Depth Anything 3 (depth_anything_3 DepthAnything3Net, InputProcessor upper_bound_resize 504)
  depth_head.f32 [S, H, W]        head depth before the sky handling
  depth.f32      [S, H, W]        output depth (mono: after sky handling)
  conf.f32 / sky.f32              [S, H, W] where the head has them
  extrinsics.f32 [S, 3, 4] / intrinsics.f32 [S, 3, 3]   any-view camera decoder
  --- MoGe-2 (moge.model.v2.MoGeModel, num_tokens from resolution_level 9, use_fp16=False)
  points_raw.f32 / normal.f32 [orig_h, orig_w, 3], mask.f32 [orig_h, orig_w]   forward()
  points.f32 [orig_h, orig_w, 3], depth.f32 [orig_h, orig_w], intrinsics.f32 [3, 3]   infer()

Usage (sources: the Depth-Anything-V2, depth-anything-3 and MoGe checkouts; weights: the HF
folders; images: PNG; several images only for DA3 any-view):
  python -I tools/depth/reference.py --src SRC_ROOT --weights DIR --out OUT IMAGE [IMAGE ...]
         [--threads 16]
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def f32(path, a):
    np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tofile(path)


def kind_of(weights):
    cfg = os.path.join(weights, "config.json")
    if os.path.exists(cfg):
        c = json.load(open(cfg))
        if "model_name" in c:
            return "da3", c
        return "da2", c
    return "moge", None


def run_da2(args, cfg, meta):
    import cv2
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(args.src, "Depth-Anything-V2"))
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
    from transformers import DepthAnythingForDepthEstimation

    raw = cv2.imread(args.images[0])
    h, w = raw.shape[:2]
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) / 255.0
    tf = [
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
    sample = {"image": image}
    for t in tf:
        sample = t(sample)
    x = torch.from_numpy(sample["image"]).unsqueeze(0).float()
    model = DepthAnythingForDepthEstimation.from_pretrained(args.weights).eval().float()
    t0 = time.time()
    with torch.no_grad():
        d = model(pixel_values=x).predicted_depth
    meta["forward_s"] = time.time() - t0
    up = F.interpolate(d[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
    f32(os.path.join(args.out, "input.f32"), x[0].numpy())
    f32(os.path.join(args.out, "raw.f32"), d[0].numpy())
    f32(os.path.join(args.out, "depth.f32"), up.numpy())
    meta.update(S=1, H=x.shape[2], W=x.shape[3], orig_h=h, orig_w=w)


def run_da3(args, cfg, meta):
    import torch
    sys.path.insert(0, os.path.join(args.src, "depth-anything-3", "src"))
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from depth_anything_3.cfg import create_object
    from depth_anything_3.utils.io.input_processor import InputProcessor

    # built from the checkpoint's own config.json (depth_anything_3.api pulls in the Gaussian
    # splatting exporters and their video dependencies)
    net = create_object(OmegaConf.create(cfg["config"]))
    sd = load_file(os.path.join(args.weights, "model.safetensors"))
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    meta["missing"] = list(missing)
    meta["unexpected"] = len(unexpected)
    net = net.eval().float()
    from PIL import Image
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in args.images]
    x, _, _ = InputProcessor()(imgs, None, None, 504, "upper_bound_resize", num_workers=1, sequential=True)
    x = x.float()
    if x.dim() == 4:
        x = x[None]
    cap = {}

    def hook(mod, inp, out):
        cap["head"] = {k: v.detach().clone() for k, v in out.items() if torch.is_tensor(v)}

    # head outputs before the sky / camera post-processing
    net.head.register_forward_hook(hook)
    t0 = time.time()
    with torch.no_grad():
        out = net(x, None, None, [], False, False, "saddle_balanced")
    meta["forward_s"] = time.time() - t0
    S, H, W = x.shape[1], x.shape[3], x.shape[4]
    meta.update(S=S, H=H, W=W, orig_h=imgs[0].shape[0], orig_w=imgs[0].shape[1], model_name=cfg["model_name"])
    f32(os.path.join(args.out, "input.f32"), x[0].numpy())
    f32(os.path.join(args.out, "depth.f32"), out["depth"][0].numpy())
    if "depth_conf" in out:
        f32(os.path.join(args.out, "conf.f32"), out["depth_conf"][0].numpy())
    if "sky" in out:
        f32(os.path.join(args.out, "sky.f32"), out["sky"][0].numpy())
    if "head" in cap and "depth" in cap["head"]:
        f32(os.path.join(args.out, "depth_head.f32"), cap["head"]["depth"][0].numpy())
    if "extrinsics" in out:
        f32(os.path.join(args.out, "extrinsics.f32"), out["extrinsics"][0, :, :3, :4].numpy())
        f32(os.path.join(args.out, "intrinsics.f32"), out["intrinsics"][0].numpy())



def run_moge(args, meta):
    import torch
    sys.path.insert(0, os.path.join(args.src, "MoGe"))
    from moge.model.v2 import MoGeModel
    from PIL import Image

    model = MoGeModel.from_pretrained(os.path.join(args.weights, "model.pt")).eval().float()
    img = np.asarray(Image.open(args.images[0]).convert("RGB"))
    h, w = img.shape[:2]
    image = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    cap = {}

    bb = model.encoder.backbone
    gil = bb.get_intermediate_layers

    def capture(x_, *a, **k):
        cap.setdefault("x", x_.detach().clone())
        return gil(x_, *a, **k)

    bb.get_intermediate_layers = capture
    nt = model.num_tokens_range
    num_tokens = int(nt[0] + (9 / 9) * (nt[1] - nt[0]))
    t0 = time.time()
    with torch.no_grad():
        fo = model.forward(image[None], num_tokens=num_tokens)
    meta["forward_s"] = time.time() - t0
    x = cap["x"]
    with torch.no_grad():
        inf = model.infer(image, num_tokens=num_tokens, use_fp16=False)
    meta.update(S=1, H=x.shape[2], W=x.shape[3], orig_h=h, orig_w=w, num_tokens=num_tokens)
    f32(os.path.join(args.out, "input.f32"), x[0].numpy())
    f32(os.path.join(args.out, "points_raw.f32"), fo["points"][0].numpy())
    if "normal" in fo:
        f32(os.path.join(args.out, "normal.f32"), fo["normal"][0].numpy())
    if "mask" in fo:
        f32(os.path.join(args.out, "mask.f32"), fo["mask"][0].numpy())
    meta["metric_scale"] = float(fo["metric_scale"][0]) if "metric_scale" in fo else 1.0
    pts = inf["points"].numpy()
    dep = inf["depth"].numpy()
    f32(os.path.join(args.out, "points.f32"), pts)
    f32(os.path.join(args.out, "depth.f32"), dep)
    f32(os.path.join(args.out, "intrinsics.f32"), inf["intrinsics"].numpy())
    k = inf["intrinsics"].numpy()
    meta["fx"] = float(k[0, 0])
    meta["fy"] = float(k[1, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    kind, cfg = kind_of(args.weights)
    meta = {"kind": kind, "weights": os.path.basename(os.path.normpath(args.weights)),
            "images": [os.path.basename(p) for p in args.images]}
    if kind == "da2":
        run_da2(args, cfg, meta)
    elif kind == "da3":
        run_da3(args, cfg, meta)
    else:
        run_moge(args, meta)
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
