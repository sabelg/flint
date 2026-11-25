#!/usr/bin/env python3
"""diffusers / transformers reference dumps for flint.ml.diffusion parity tests.

    python tools/diffusion/reference.py WEIGHTS OUT [stablematerials] [matfuse] [sched] [--photo IMG]

WEIGHTS is the folder written by tools/diffusion/fetch.py (include its `code` set: the
models' HF pipeline code is read from WEIGHTS/code, or $DIFFUSION_HF_CODE). Needs torch,
transformers and diffusers (torchvision is optional). Everything is run in
float32 on the CPU with fixed seeds. OUT receives raw little-endian float32
(`*.f32`) / int32 (`*.i32`) arrays plus `index.json` (name -> shape, scalars).

* stablematerials: the HF custom pipeline (gvecchio/StableMaterials pipeline.py,
  LCM UNet, LCMScheduler, tileable=True) run unmodified; the UNet and the
  scheduler are wrapped to record every step's model input, timestep, noise
  prediction and updated latents. Also CLIP text/vision embeddings and a VAE
  encode of the decoded maps.
* matfuse: the HF MatFusePipeline (DDIM). Its condition encoder lazily loads
  OpenAI `clip` and `sentence_transformers`; both are swapped for the same
  weights through `transformers` (sentence-transformers/clip-ViT-B-16).
* sched: DDIM / Euler / LCM timesteps, sigmas and single steps on fixed inputs,
  and torch.randn streams for the RNG test.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

torch.set_grad_enabled(False)


def _torchvision_shim():
    """The StableMaterials pipeline imports torchvision.transforms.functional for center_crop /
    resize / to_pil_image only; provide those when torchvision is not installed."""
    try:
        import torchvision  # noqa: F401
        return
    except ImportError:
        pass
    import types
    import torch.nn.functional as F
    tv = types.ModuleType("torchvision")
    tr = types.ModuleType("torchvision.transforms")
    fn = types.ModuleType("torchvision.transforms.functional")

    def center_crop(img, size):
        if isinstance(size, int):
            size = (size, size)
        h, w = img.shape[-2:]
        th, tw = size
        top = int(round((h - th) / 2.0))
        left = int(round((w - tw) / 2.0))
        return img[..., top:top + th, left:left + tw]

    def resize(img, size, antialias=True):
        x = img[None] if img.dim() == 3 else img
        y = F.interpolate(x, size=size, mode="bilinear", antialias=antialias, align_corners=False)
        return y[0] if img.dim() == 3 else y

    def to_pil_image(t):
        from PIL import Image
        a = (t.detach().cpu().float().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        if a.shape[0] == 1:
            return Image.fromarray(a[0], "L")
        return Image.fromarray(a.transpose(1, 2, 0), "RGB")

    def to_tensor(img):
        a = np.asarray(img).astype(np.float32) / 255.0
        if a.ndim == 2:
            a = a[None]
        else:
            a = a.transpose(2, 0, 1)
        return torch.from_numpy(a.copy())

    fn.center_crop, fn.resize, fn.to_pil_image, fn.to_tensor = center_crop, resize, to_pil_image, to_tensor
    tr.functional = fn
    tv.transforms = tr
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = tr
    sys.modules["torchvision.transforms.functional"] = fn

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

IDX = {}
OUT = "."


def dump(name, x, scalars=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        a = x.numpy()
    else:
        a = np.asarray(x)
    if a.dtype.kind in "iub":
        a.astype("<i4").tofile(os.path.join(OUT, name + ".i32"))
    else:
        a.astype("<f4").tofile(os.path.join(OUT, name + ".f32"))
    IDX[name] = {"shape": list(a.shape)}
    if scalars:
        IDX[name].update(scalars)


def note(name, value):
    IDX[name] = value


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Recorder:
    """Wraps a UNet: records (input, t, cond, output) per call."""

    def __init__(self, unet, prefix):
        self.unet = unet
        self.prefix = prefix
        self.calls = 0
        self.orig = unet.forward

    def __call__(self, sample, timestep, encoder_hidden_states=None, timestep_cond=None, **kw):
        out = self.orig(sample, timestep, encoder_hidden_states=encoder_hidden_states,
                        timestep_cond=timestep_cond, **kw)
        k = self.calls
        p = "%s_step%d" % (self.prefix, k)
        dump(p + "_in", sample, {"t": float(timestep)})
        dump(p + "_ctx", encoder_hidden_states)
        if timestep_cond is not None:
            dump(p + "_tcond", timestep_cond)
        o = out[0] if isinstance(out, tuple) else out.sample
        dump(p + "_eps", o)
        self.calls += 1
        return out


def stablematerials(weights, hf_code, photo):
    from diffusers import UNet2DConditionModel, LCMScheduler, AutoencoderKL
    from transformers import (CLIPTextModelWithProjection, CLIPVisionModelWithProjection,
                              CLIPTokenizer, CLIPImageProcessor)
    root = os.path.join(weights, "stablematerials")
    _torchvision_shim()  # after diffusers has probed its optional packages
    mod = load_module(os.path.join(hf_code, "pipeline.py"), "sm_pipeline")
    f32 = torch.float32
    unet = UNet2DConditionModel.from_pretrained(root, subfolder="unet_lcm", torch_dtype=f32)
    vae = AutoencoderKL.from_pretrained(root, subfolder="vae", torch_dtype=f32)
    te = CLIPTextModelWithProjection.from_pretrained(root, subfolder="text_encoder", torch_dtype=f32)
    ve = CLIPVisionModelWithProjection.from_pretrained(root, subfolder="vision_encoder", torch_dtype=f32)
    tok = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer")
    proc = CLIPImageProcessor.from_pretrained(root, subfolder="processor")
    from diffusers import DDIMScheduler
    ddim = DDIMScheduler.from_pretrained(root, subfolder="scheduler")
    sched = LCMScheduler.from_config(ddim.config)
    pipe = mod.StableMaterialsPipeline(vae=vae, unet=unet, scheduler=sched, text_encoder=te,
                                       tokenizer=tok, vision_encoder=ve, processor=proc)
    prompt = "Old rusty metal bars with peeling paint"
    ids = tok(prompt, padding=True, return_tensors="pt")
    dump("sm_tokens", ids["input_ids"][0].to(torch.int32))
    for extra in ["red brick wall, white mortar", "Polished marble tiles with gold veins!", "a 1920's café floor"]:
        dump("tok_" + "".join(c if c.isalnum() else "_" for c in extra)[:24], tok(extra)["input_ids"])
    note("tok_texts", ["red brick wall, white mortar", "Polished marble tiles with gold veins!", "a 1920's café floor"])
    dump("sm_text_embeds", te(**ids).text_embeds)
    from PIL import Image
    black = Image.new("RGB", (512, 512), (0, 0, 0))
    dump("sm_black_pixels", proc(images=black, return_tensors="pt")["pixel_values"])
    dump("sm_black_embeds", ve(**proc(images=black, return_tensors="pt")).image_embeds)
    if photo:
        img = Image.open(photo).convert("RGB")
        pv = proc(images=img, return_tensors="pt")["pixel_values"]
        dump("sm_photo_pixels", pv)
        dump("sm_photo_embeds", ve(pixel_values=pv).image_embeds)

    rec = Recorder(unet, "sm")
    unet.forward = rec
    orig_step = sched.step
    steps = []

    def step(model_output, timestep, sample, generator=None, return_dict=True, **kw):
        r = orig_step(model_output, timestep, sample, generator=generator, return_dict=return_dict, **kw)
        prev = r[0] if isinstance(r, tuple) else r.prev_sample
        dump("sm_step%d_out" % len(steps), prev)
        steps.append(int(timestep))
        return r

    sched.step = step
    g = torch.Generator().manual_seed(42)
    note("sm_seed", 42)
    note("sm_prompt", prompt)
    # The initial noise the pipeline draws (prepare_latents uses the same generator first).
    dump("sm_noise0", torch.randn((1, 18, 64, 64), generator=torch.Generator().manual_seed(42)))
    out = pipe(prompt=prompt, guidance_scale=10.0, tileable=True, num_inference_steps=4,
               generator=g, output_type="pt")
    note("sm_timesteps", steps)
    m = out.images[0]
    dump("sm_maps", m.as_tensor())          # basecolor(3) normal xy(2) height roughness metallic
    dump("sm_normal", m._normal)
    # The raw decoded 9-channel image: re-run the decode on the final latents.
    lat = None
    # Recompute final latents from the recorded last step for an exact VAE decode dump.
    lat = torch.from_numpy(np.fromfile(os.path.join(OUT, "sm_step%d_out.f32" % (len(steps) - 1)),
                                       dtype="<f4").reshape(1, 18, 64, 64))
    import torchvision.transforms.functional as TF
    big = TF.center_crop(lat.repeat(1, 1, 3, 3), (80, 80))
    img = vae.decode(big / vae.config.scaling_factor, return_dict=False)[0]
    img = TF.center_crop(img, (512, 512))
    dump("sm_image", img)
    # VAE encoder parity on a 256 x 256 crop of the decoded maps (zero padding, mean of the posterior).
    crop = img[:, :, :256, :256].clamp(-1, 1)
    dump("sm_enc_in", crop)
    dump("sm_enc_mean", vae.encode(crop).latent_dist.mean)
    for name, t in [("basecolor", m._basecolor), ("normal", m._normal), ("height", m._height),
                    ("roughness", m._roughness), ("metallic", m._metallic)]:
        TF.to_pil_image(t).save(os.path.join(OUT, "sm_%s.png" % name))


def matfuse(weights, hf_code, photo):
    root = os.path.join(weights, "matfuse")
    sys.path.insert(0, os.path.join(hf_code, "condition_encoder"))
    sys.path.insert(0, os.path.join(hf_code, "vae"))
    ce_mod = load_module(os.path.join(hf_code, "condition_encoder", "condition_encoders.py"), "condition_encoders")
    vae_mod = load_module(os.path.join(hf_code, "vae", "vae_matfuse.py"), "vae_matfuse")
    from transformers import CLIPModel, CLIPTokenizer
    clip = CLIPModel.from_pretrained(os.path.join(root, "clip"), torch_dtype=torch.float32).eval()
    ctok = CLIPTokenizer.from_pretrained(os.path.join(root, "clip"))

    class HFVisual(torch.nn.Module):
        def forward(self, x):
            return clip.visual_projection(clip.vision_model(pixel_values=x).pooler_output)

    def img_load(self):
        if self.model is None:
            self.model = HFVisual()

    class HFText:
        def __init__(self):
            self.calls = 0

        def __call__(self, text):
            if isinstance(text, str):
                text = [text]
            t = ctok(text, padding=True, truncation=True, max_length=77, return_tensors="pt")
            return clip.text_projection(clip.text_model(**t).pooler_output)

    def txt_load(self):
        if getattr(self, "model", None) is None:
            self.model = HFText()

    ce_mod.CLIPImageEncoder._load_model = img_load
    ce_mod.CLIPTextEncoder._load_model = txt_load
    ce_mod.CLIPTextEncoder.forward = lambda self, text: (txt_load(self), self.model(text))[1]

    from diffusers import UNet2DConditionModel, DDIMScheduler
    unet = UNet2DConditionModel.from_pretrained(root, subfolder="unet", torch_dtype=torch.float32)
    vae = vae_mod.MatFuseVQModel.from_pretrained(root, subfolder="vae", torch_dtype=torch.float32)
    ce = ce_mod.MultiConditionEncoder.from_pretrained(root, subfolder="condition_encoder", torch_dtype=torch.float32)
    sched = DDIMScheduler.from_pretrained(root, subfolder="scheduler")
    pmod = load_module(os.path.join(hf_code, "pipeline_matfuse.py"), "pipeline_matfuse")
    pipe = pmod.MatFusePipeline(unet=unet, vae=vae, condition_encoder=ce, scheduler=sched)
    ce.eval()
    text = "red brick wall"
    dump("mf_tokens", ctok(text)["input_ids"])
    cond = ce(text=text, batch_size=1, image_size=256)
    unc = ce.get_unconditional_conditioning(batch_size=1, image_size=256)
    dump("mf_c_crossattn", cond["c_crossattn"])
    dump("mf_c_concat", cond["c_concat"])
    dump("mf_uc_crossattn", unc["c_crossattn"])
    dump("mf_uc_concat", unc["c_concat"])
    pal = torch.tensor([[[0.6, 0.2, 0.1], [0.5, 0.45, 0.4], [0.9, 0.9, 0.85], [0.3, 0.1, 0.05], [0.7, 0.3, 0.2]]])
    dump("mf_palette_in", pal)
    dump("mf_palette_emb", ce.encode_palette(pal))
    if photo:
        from PIL import Image
        im = pipe._preprocess_image(Image.open(photo), "cpu", torch.float32)
        dump("mf_photo_in", im)
        dump("mf_photo_emb", ce.encode_image(im))
    rec = Recorder(unet, "mf")
    unet.forward = rec
    orig_step = sched.step
    steps = []

    def step(model_output, timestep, sample, *a, **kw):
        r = orig_step(model_output, timestep, sample, *a, **kw)
        prev = r[0] if isinstance(r, tuple) else r.prev_sample
        dump("mf_step%d_out" % len(steps), prev)
        steps.append(int(timestep))
        return r

    sched.step = step
    nsteps = int(os.environ.get("MF_STEPS", "20"))
    note("mf_steps", nsteps)
    dump("mf_noise0", torch.randn((1, 12, 32, 32), generator=torch.Generator().manual_seed(7)))
    res = pipe(text=text, num_inference_steps=nsteps, guidance_scale=4.0, generator=torch.Generator().manual_seed(7),
               output_type="tensor")
    note("mf_timesteps", steps)
    lat = torch.from_numpy(np.fromfile(os.path.join(OUT, "mf_step%d_out.f32" % (len(steps) - 1)),
                                       dtype="<f4").reshape(1, 12, 32, 32))
    dump("mf_image", pipe.decode_latents(lat))
    _torchvision_shim()
    import torchvision.transforms.functional as TF
    for name in ["diffuse", "normal", "roughness", "specular"]:
        t = res[name][0] if isinstance(res[name], (list, tuple)) else res[name][0]
        if isinstance(t, torch.Tensor):
            TF.to_pil_image(((t.float() + 1) / 2).clamp(0, 1) if t.min() < 0 else t.float().clamp(0, 1)).save(
                os.path.join(OUT, "mf_%s.png" % name))


def schedulers():
    from diffusers import DDIMScheduler, EulerDiscreteScheduler, LCMScheduler
    cfg = dict(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
               clip_sample=False, set_alpha_to_one=False, steps_offset=1, prediction_type="epsilon",
               timestep_spacing="leading")
    torch.manual_seed(0)
    x = torch.randn(64)
    e = torch.randn(64)
    dump("sch_x", x)
    dump("sch_e", e)
    d = DDIMScheduler(**cfg)
    dump("sch_ddim_ac", d.alphas_cumprod)
    d.set_timesteps(25)
    dump("sch_ddim_ts", d.timesteps.to(torch.int32))
    dump("sch_ddim_step", d.step(e, d.timesteps[3], x).prev_sample)
    mf = DDIMScheduler(beta_start=0.0015, beta_end=0.0195, beta_schedule="scaled_linear", clip_sample=False,
                       set_alpha_to_one=False, steps_offset=0, timestep_spacing="leading")
    mf.set_timesteps(20)
    dump("sch_ddim_mf_ts", mf.timesteps.to(torch.int32))
    dump("sch_ddim_mf_step", mf.step(e, mf.timesteps[0], x).prev_sample)
    eu = EulerDiscreteScheduler(**{k: v for k, v in cfg.items() if k not in ("clip_sample", "set_alpha_to_one")})
    eu.set_timesteps(30)
    dump("sch_euler_ts", eu.timesteps.to(torch.float32))
    dump("sch_euler_sigmas", eu.sigmas)
    note("sch_euler_init_sigma", float(eu.init_noise_sigma))
    dump("sch_euler_scaled", eu.scale_model_input(x, eu.timesteps[5]))
    eu._step_index = 5
    dump("sch_euler_step", eu.step(e, eu.timesteps[5], x).prev_sample)
    lc = LCMScheduler.from_config(DDIMScheduler(**cfg).config)
    for n in (1, 2, 4, 8):
        lc.set_timesteps(n)
        dump("sch_lcm_ts%d" % n, lc.timesteps.to(torch.int32))
    lc.set_timesteps(4)
    g = torch.Generator().manual_seed(5)
    dump("sch_lcm_step", lc.step(e, lc.timesteps[1], x, generator=g).prev_sample)
    lc._step_index = 3
    dump("sch_lcm_last", lc.step(e, lc.timesteps[3], x).prev_sample)
    # RNG streams: torch.Generator().manual_seed(s); randn(n)
    for s, n in ((0, 40), (42, 73728), (123456789, 1000), (2**40 + 3, 17)):
        dump("rng_%d_%d" % (s % 1000003, n), torch.randn(n, generator=torch.Generator().manual_seed(s)),
             {"seed": s})
    # guidance-scale embedding (LCM w-conditioning)
    w = torch.tensor([9.0]) * 1000.0
    half = 128
    emb = torch.log(torch.tensor(10000.0)) / (half - 1)
    emb = torch.exp(torch.arange(half, dtype=torch.float32) * -emb)
    emb = w[:, None] * emb[None, :]
    dump("sch_wemb", torch.cat([torch.sin(emb), torch.cos(emb)], dim=1))


def main():
    global OUT
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    photo = None
    if "--photo" in sys.argv:
        photo = sys.argv[sys.argv.index("--photo") + 1]
        args = [a for a in args if a != photo]
    weights, OUT = args[0], args[1]
    which = args[2:] or ["sched", "stablematerials", "matfuse"]
    hf_code = os.environ.get("DIFFUSION_HF_CODE", os.path.join(weights, "code"))
    os.makedirs(OUT, exist_ok=True)
    ip = os.path.join(OUT, "index.json")
    if os.path.exists(ip):
        IDX.update(json.load(open(ip)))
    try:
        if "sched" in which:
            schedulers()
        if "stablematerials" in which:
            stablematerials(weights, os.path.join(hf_code, "StableMaterials"), photo)
        if "matfuse" in which:
            matfuse(weights, os.path.join(hf_code, "MatFuse"), photo)
    finally:
        json.dump(IDX, open(ip, "w"), indent=1)


if __name__ == "__main__":
    main()
