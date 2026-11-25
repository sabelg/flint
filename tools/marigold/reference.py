#!/usr/bin/env python3
"""diffusers parity dumps for flint.ml.marigold (Marigold-Normals v1.1).

    python tools/marigold/reference.py MODEL_DIR IMAGE.png OUT_DIR [--res R] [--steps N]
           [--ensemble E] [--seed S] [--name NAME] [--sched-only]

MODEL_DIR is the folder written by tools/marigold/fetch.py (prs-eth/marigold-normals-v1-1 in the
diffusers layout). Runs `MarigoldNormalsPipeline` in float32 on the CPU and writes, for case NAME:

  NAME_rgb.f32        the input [H, W, 3] in [0, 1]
  NAME_image.f32      preprocessed image [3, PH, PW] in [-1, 1] (resized + padded)
  NAME_ilat.f32       image latent * scaling_factor [4, h, w]
  NAME_noise.f32      initial prediction latent [E, 4, h, w]
  NAME_plat.f32       final prediction latent [E, 4, h, w]
  NAME_normals.f32    output normals [H, W, 3] (ensembled, input resolution)
  NAME_unc.f32        uncertainty [H, W] (ensemble >= 3)
  NAME.txt            sizes, timings
  empty_text.f32      empty-prompt CLIP embedding [2, 1024]
  sched.txt           DDIM timesteps for 1 / 4 / 10 / 50 steps and a few alphas_cumprod values

The scheduler / text dumps only need the config and text encoder. Reads the weights (fp16 or fp32
safetensors, upcast to fp32 like Flint) at verification time; nothing here is copied into Flint.
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="case")
    ap.add_argument("--sched-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_grad_enabled(False)
    from diffusers import DDIMScheduler, MarigoldNormalsPipeline

    sch = DDIMScheduler.from_pretrained(a.model, subfolder="scheduler")
    lines = []
    for n in (1, 2, 4, 10, 50):
        sch.set_timesteps(n)
        lines.append("steps %d %s" % (n, " ".join(str(int(t)) for t in sch.timesteps)))
    ac = sch.alphas_cumprod
    lines.append("alphas_cumprod " + " ".join("%d:%.9g" % (i, float(ac[i])) for i in (0, 1, 250, 500, 998, 999)))
    lines.append("final_alpha_cumprod %.9g" % float(sch.final_alpha_cumprod))
    open(os.path.join(a.out, "sched.txt"), "w").write("\n".join(lines) + "\n")
    if a.sched_only:
        return

    pipe = MarigoldNormalsPipeline.from_pretrained(a.model, torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    if pipe.empty_text_embedding is None:
        ids = pipe.tokenizer("", padding="do_not_pad", max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids
        pipe.empty_text_embedding = pipe.text_encoder(ids)[0]
    pipe.empty_text_embedding[0].float().numpy().tofile(os.path.join(a.out, "empty_text.f32"))

    img = np.asarray(Image.open(a.image).convert("RGB"))
    H, W = img.shape[:2]
    (img.astype(np.float32) / 255.0).tofile(os.path.join(a.out, a.name + "_rgb.f32"))
    # capture the intermediates the pipeline does not return
    got = {}
    orig_prep = pipe.prepare_latents

    def prep(image, latents, generator, ensemble_size, batch_size):
        il, pl = orig_prep(image, latents, generator, ensemble_size, batch_size)
        got["image"] = image[0].clone()
        got["ilat"] = il[0].clone()
        got["noise"] = pl.clone()
        return il, pl

    pipe.prepare_latents = prep
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    out = pipe(img, num_inference_steps=a.steps, ensemble_size=a.ensemble, processing_resolution=a.res,
               generator=gen, batch_size=a.ensemble, output_uncertainty=a.ensemble >= 3, output_latent=True)
    dt = time.time() - t0
    got["image"].numpy().astype(np.float32).tofile(os.path.join(a.out, a.name + "_image.f32"))
    got["ilat"].numpy().astype(np.float32).tofile(os.path.join(a.out, a.name + "_ilat.f32"))
    got["noise"].numpy().astype(np.float32).tofile(os.path.join(a.out, a.name + "_noise.f32"))
    out.latent.numpy().astype(np.float32).tofile(os.path.join(a.out, a.name + "_plat.f32"))
    out.prediction[0].astype(np.float32).tofile(os.path.join(a.out, a.name + "_normals.f32"))
    if out.uncertainty is not None:
        out.uncertainty[0, ..., 0].astype(np.float32).tofile(os.path.join(a.out, a.name + "_unc.f32"))
    ph, pw = got["image"].shape[1:]
    with open(os.path.join(a.out, a.name + ".txt"), "w") as f:
        f.write("H %d W %d PH %d PW %d res %d steps %d ensemble %d seed %d seconds %.3f threads %d\n"
                % (H, W, ph, pw, a.res, a.steps, a.ensemble, a.seed, dt, torch.get_num_threads()))
    vis = ((out.prediction[0] * 0.5 + 0.5).clip(0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(vis).save(os.path.join(a.out, a.name + "_normals.png"))
    print("%s: %dx%d res %d steps %d ensemble %d: %.2fs" % (a.name, W, H, a.res, a.steps, a.ensemble, dt))


if __name__ == "__main__":
    main()
