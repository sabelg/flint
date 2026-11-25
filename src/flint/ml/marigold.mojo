"""Marigold dense prediction on `flint.ml.diffusion` (`flint.ml.marigold`): image -> normal map.

Marigold (Ke et al., CVPR 2024; journal extension 2025) fine-tunes Stable Diffusion 2 into a
dense predictor. The UNet's input convolution takes 8 channels: the VAE latent of the RGB image
(the condition) followed by the latent being denoised; the cross-attention context is the fixed
CLIP embedding of the empty prompt. A v-prediction DDIM schedule with zero terminal SNR and
"trailing" spacing makes 1..50 steps usable (1-4 is the intended range). The final latent is
decoded by the SD2 VAE into the target image, here 3-D surface normals in [-1, 1].

Layout of this module (shared core + task heads, so Marigold depth / intrinsics can follow):

  * `MarigoldCore`: UNet / VAE / scheduler / empty-text embedding from a diffusers folder;
    `preprocess` (resize the longer side to the processing resolution with antialiased bilinear,
    replicate-pad to a multiple of 8, exactly `MarigoldImageProcessor`), `encode_image`,
    `denoise` (seeded `torch.randn`-exact noise for the whole ensemble, batched UNet calls) and
    `decode` (VAE decode of one member, clipped to [-1, 1], padding removed). `tile = True` swaps
    every convolution to wrap-around padding and resizes to multiples of 8 instead of padding, for
    seamless predictions on tileable textures.
  * `MarigoldNormals`: the normals head of `MarigoldNormalsPipeline` — unit-normalise each member,
    ensemble (`closest` = the member nearest the normalised mean per pixel, or `mean`),
    uncertainty = mean angle to the mean / pi (ensemble >= 3), bilinear resize to the input size
    and normalise again. Normals are in the camera frame: +X right, +Y up, +Z towards the viewer
    (the OpenGL / glTF tangent-space convention; `normals_to_pixels(.., directx=True)` flips Y).

A depth head would decode the mean of the three channels and align members with a scale / shift
least-squares fit; an intrinsics head would set `targets > 1` (each target is 4 latent channels,
`decode` takes the target index). Weights are read from Hugging Face `prs-eth/marigold-normals-v1-1`
(CreativeML Open RAIL++-M, see `licenses/marigold-LICENSE-MODEL.txt`): no commercial ban, but the
use restrictions of its Attachment A apply to users of the model and its outputs.
"""

from std.math import sqrt, acos, floor
from flint.platform.os import read_bytes
from flint.assets.json import parse_json
from flint.ml.diffusion.nn import Runtime, Act
from flint.ml.diffusion.unet import UNet
from flint.ml.diffusion.vae import VaeEncoder, VaeDecoder
from flint.ml.diffusion.clip import ClipText
from flint.ml.diffusion.schedulers import Scheduler, SCHED_DDIM
from flint.ml.diffusion.rng import TorchGenerator
from flint.ml.diffusion.pipeline import nchw_to_nhwc

comptime CLIP_BOS: Int = 49406
comptime CLIP_EOS: Int = 49407

comptime ENSEMBLE_CLOSEST: Int = 0
comptime ENSEMBLE_MEAN: Int = 1


# ---------------------------------------------------------------------------------------------
# resampling (torch.nn.functional.interpolate, mode="bilinear", align_corners=False)


struct _Taps(Movable):
    var start: List[Int]
    var count: List[Int]
    var w: List[Float32]
    var stride: Int

    def __init__(out self):
        self.start = List[Int]()
        self.count = List[Int]()
        self.w = List[Float32]()
        self.stride = 0


def _bilinear_taps(inn: Int, outn: Int, antialias: Bool) -> _Taps:
    """Per-output taps in float32 like ATen. `antialias`: `_upsample_bilinear2d_aa` (triangle
    filter widened by the scale when shrinking, normalised weights); otherwise plain
    `upsample_bilinear2d` (source coordinate clamped at 0)."""
    var t = _Taps()
    var scale = Float32(inn) / Float32(outn)
    if antialias:
        var support = scale if scale >= 1.0 else Float32(1.0)
        var inv = Float32(1.0) / scale if scale >= 1.0 else Float32(1.0)
        t.stride = Int(Float64(support) * 2.0) + 3
        t.w = List[Float32](length=outn * t.stride, fill=0.0)
        for i in range(outn):
            var c = scale * (Float32(i) + Float32(0.5))
            var lo = max(Int(c - support + Float32(0.5)), 0)
            var hi = min(Int(c + support + Float32(0.5)), inn)
            var n = min(hi - lo, t.stride)
            var total: Float32 = 0
            for j in range(n):
                var x = abs((Float32(j + lo) - c + Float32(0.5)) * inv)
                var v = Float32(1.0) - x if x < 1.0 else Float32(0.0)
                t.w[i * t.stride + j] = v
                total += v
            if total != 0.0:
                for j in range(n):
                    t.w[i * t.stride + j] /= total
            t.start.append(lo)
            t.count.append(n)
        return t^
    t.stride = 2
    t.w = List[Float32](length=outn * 2, fill=0.0)
    for i in range(outn):
        var src = scale * (Float32(i) + Float32(0.5)) - Float32(0.5)
        if src < 0:
            src = 0
        var i0 = min(Int(src), inn - 1)
        var l1 = min(max(src - Float32(i0), Float32(0.0)), Float32(1.0))
        t.start.append(i0)
        if i0 < inn - 1:
            t.count.append(2)
            t.w[i * 2] = Float32(1.0) - l1
            t.w[i * 2 + 1] = l1
        else:
            # the second tap is the same pixel: its weights add up to (1 - l1) + l1
            t.count.append(1)
            t.w[i * 2] = (Float32(1.0) - l1) + l1
    return t^


def _resize_axis(x: List[Float32], outer: Int, n: Int, inner: Int, t: _Taps, on: Int) -> List[Float32]:
    """Resample the middle axis of an [outer, n, inner] array to `on` entries."""
    var out = List[Float32](length=outer * on * inner, fill=0.0)
    for a in range(outer):
        for i in range(on):
            var o = (a * on + i) * inner
            for j in range(t.count[i]):
                var wt = t.w[i * t.stride + j]
                var s = (a * n + t.start[i] + j) * inner
                for e in range(inner):
                    out[o + e] += wt * x[s + e]
    return out^


def resize_bilinear(x: List[Float32], h: Int, w: Int, c: Int, oh: Int, ow: Int,
                    antialias: Bool) -> List[Float32]:
    """`F.interpolate(mode="bilinear", align_corners=False, antialias=..)` on a channels-last
    [h, w, c] image -> [oh, ow, c]: height pass then width pass, unchanged axes skipped."""
    var cur = x.copy()
    if oh != h:
        cur = _resize_axis(cur, 1, h, w * c, _bilinear_taps(h, oh, antialias), oh)
    if ow != w:
        cur = _resize_axis(cur, oh, w, c, _bilinear_taps(w, ow, antialias), ow)
    return cur^


# ---------------------------------------------------------------------------------------------
# core


struct MarigoldImage(Movable):
    """A preprocessed input: channels-last [ph, pw, 3] in [-1, 1] (resized + padded)."""
    var data: List[Float32]
    var h: Int          # original size
    var w: Int
    var rh: Int         # resized size (before padding)
    var rw: Int
    var ph: Int         # padded size (multiple of 8)
    var pw: Int

    def __init__(out self):
        self.data = List[Float32]()
        self.h = 0
        self.w = 0
        self.rh = 0
        self.rw = 0
        self.ph = 0
        self.pw = 0


struct MarigoldCore(Movable):
    var unet: UNet
    var enc: VaeEncoder
    var dec: VaeDecoder
    var sched: Scheduler
    var text: List[Float32]      # empty-prompt embedding [tokens, width]
    var tokens: Int
    var width: Int
    var default_steps: Int
    var default_resolution: Int
    var targets: Int             # predicted latents (4 channels each): 1 for normals / depth
    var tile: Bool

    def __init__(out self, var unet: UNet, var enc: VaeEncoder, var dec: VaeDecoder, var sched: Scheduler):
        self.unet = unet^
        self.enc = enc^
        self.dec = dec^
        self.sched = sched^
        self.text = List[Float32]()
        self.tokens = 0
        self.width = 0
        self.default_steps = 4
        self.default_resolution = 768
        self.targets = 1
        self.tile = False

    @staticmethod
    def load(mut rt: Runtime, dir: String) raises -> MarigoldCore:
        """A diffusers Marigold folder (unet/, vae/, scheduler/, text_encoder/, model_index.json).
        The empty-prompt embedding is computed once with the CLIP text encoder."""
        var sched = Scheduler.from_config(read_bytes(dir + "/scheduler/scheduler_config.json"), SCHED_DDIM)
        var unet = UNet.load(rt, dir + "/unet")
        var enc = VaeEncoder.load(rt, dir + "/vae")
        var dec = VaeDecoder.load(rt, dir + "/vae")
        var core = MarigoldCore(unet^, enc^, dec^, sched^)
        if core.unet.in_channels % 4 != 0 or core.unet.in_channels < 8:
            raise Error("marigold: expected an image-latent-concat UNet (8+ input channels)")
        core.targets = core.unet.out_channels // 4
        var idx = read_bytes(dir + "/model_index.json")
        if len(idx) > 0:
            var doc = parse_json(idx)
            core.default_steps = doc.integer(doc.object_get(doc.root, "default_denoising_steps"), 4)
            core.default_resolution = doc.integer(doc.object_get(doc.root, "default_processing_resolution"), 768)
        var te = ClipText.load(rt, dir + "/text_encoder/model.safetensors", dir + "/text_encoder/config.json")
        var ids = List[Int]()
        ids.append(CLIP_BOS)
        ids.append(CLIP_EOS)
        var emb = te.encode(rt, ids)
        core.text = emb.hidden.copy()
        core.tokens = emb.tokens
        core.width = emb.width
        return core^

    def preprocess(self, rgb01: List[Float32], h: Int, w: Int, resolution: Int) raises -> MarigoldImage:
        """[h, w, 3] in [0, 1] -> normalised, resized (longer side = `resolution`, 0 keeps the
        size) and replicate-padded to a multiple of 8. With `tile`, both sides are resized to
        multiples of 8 instead (padding would break the wrap-around)."""
        if len(rgb01) != h * w * 3:
            raise Error("marigold: image size mismatch")
        if resolution % 8 != 0:
            raise Error("marigold: processing resolution must be a multiple of 8")
        var m = MarigoldImage()
        m.h = h
        m.w = w
        var x = List[Float32](length=len(rgb01), fill=0.0)
        for i in range(len(rgb01)):
            x[i] = rgb01[i] * Float32(2.0) - Float32(1.0)
        var rh = h
        var rw = w
        if resolution > 0:
            var mx = max(h, w)
            rh = h * resolution // mx
            rw = w * resolution // mx
            if rh == 0 or rw == 0:
                raise Error("marigold: extreme aspect ratio")
        if self.tile:
            rh = max(8, (rh + 4) // 8 * 8)
            rw = max(8, (rw + 4) // 8 * 8)
        var r = resize_bilinear(x, h, w, 3, rh, rw, True)
        m.rh = rh
        m.rw = rw
        m.ph = (rh + 7) // 8 * 8
        m.pw = (rw + 7) // 8 * 8
        if m.ph == rh and m.pw == rw:
            m.data = r^
            return m^
        m.data = List[Float32](length=m.ph * m.pw * 3, fill=0.0)
        for y in range(m.ph):
            var sy = min(y, rh - 1)
            for xx in range(m.pw):
                var sx = min(xx, rw - 1)
                for k in range(3):
                    m.data[(y * m.pw + xx) * 3 + k] = r[(sy * rw + sx) * 3 + k]
        return m^

    def encode_image(mut self, mut rt: Runtime, img: MarigoldImage) raises -> List[Float32]:
        """Scaled VAE posterior mean, channels-last [ph/8, pw/8, 4]."""
        var xa = rt.upload(1, img.ph, img.pw, 3, img.data)
        var lat = self.enc.encode_mean(rt, xa, self.tile)
        rt.free(xa^)
        var z = rt.download(lat)
        rt.free(lat^)
        var s = self.enc.scaling_factor
        for i in range(len(z)):
            z[i] *= s
        return z^

    def initial_noise(self, ensemble: Int, h8: Int, w8: Int, seed: Int) -> List[Float32]:
        """`torch.randn([ensemble, 4 * targets, h8, w8], generator=manual_seed(seed))`,
        returned channels-last per member: [ensemble, h8, w8, 4 * targets]."""
        var c = 4 * self.targets
        var gen = TorchGenerator(seed)
        var n = gen.randn(ensemble * c * h8 * w8)
        return nchw_to_nhwc(n, ensemble, c, h8, w8)

    def denoise(mut self, mut rt: Runtime, ilat: List[Float32], h8: Int, w8: Int, var latents: List[Float32],
                ensemble: Int, steps: Int, batch: Int = 0) raises -> List[Float32]:
        """Run the DDIM loop for every ensemble member (channels-last latents, `batch` members per
        UNet call, 0 = all). Returns the final latents [ensemble, h8, w8, 4 * targets]."""
        if steps < 1:
            raise Error("marigold: need at least one denoising step")
        var c = 4 * self.targets
        var ci = 4
        var hw = h8 * w8
        if len(ilat) != hw * ci or len(latents) != ensemble * hw * c:
            raise Error("marigold: latent size mismatch")
        var bs = ensemble if batch <= 0 else min(batch, ensemble)
        self.sched.set_timesteps(steps)
        var b0 = 0
        while b0 < ensemble:
            var B = min(bs, ensemble - b0)
            var x = List[Float32](length=B * hw * c, fill=0.0)
            for k in range(B * hw * c):
                x[k] = latents[b0 * hw * c + k]
            var ctx = List[Float32]()
            for _ in range(B):
                for v in self.text:
                    ctx.append(v)
            var ca = rt.upload(B, self.tokens, 1, self.width, ctx)
            for i in range(self.sched.num_steps()):
                var C = ci + c
                var inp = List[Float32](length=B * hw * C, fill=0.0)
                for b in range(B):
                    for p in range(hw):
                        var o = (b * hw + p) * C
                        for k in range(ci):
                            inp[o + k] = ilat[p * ci + k]
                        for k in range(c):
                            inp[o + ci + k] = x[(b * hw + p) * c + k]
                var xa = rt.upload(B, h8, w8, C, inp)
                var oa = self.unet.forward(rt, xa, self.sched.timestep(i), ca, List[Float32](), self.tile)
                rt.free(xa^)
                var v = rt.download(oa)
                rt.free(oa^)
                var gen = TorchGenerator(0)     # DDIM with eta = 0 draws no noise
                x = self.sched.step(v, i, x, gen)
            rt.free(ca^)
            for k in range(B * hw * c):
                latents[b0 * hw * c + k] = x[k]
            b0 += B
        return latents^

    def decode(mut self, mut rt: Runtime, latents: List[Float32], member: Int, target: Int, h8: Int, w8: Int,
               img: MarigoldImage) raises -> List[Float32]:
        """VAE decode of one member / target, clipped to [-1, 1] and un-padded: [rh, rw, 3]."""
        var c = 4 * self.targets
        var hw = h8 * w8
        var z = List[Float32](length=hw * 4, fill=0.0)
        var inv = Float32(1.0) / self.dec.scaling_factor
        for p in range(hw):
            for k in range(4):
                z[p * 4 + k] = latents[(member * hw + p) * c + target * 4 + k] * inv
        var za = rt.upload(1, h8, w8, 4, z)
        var ia = self.dec.decode(rt, za, self.tile)
        rt.free(za^)
        var full = rt.download(ia)
        var W = ia.w
        var oc = ia.c
        rt.free(ia^)
        var out = List[Float32](length=img.rh * img.rw * 3, fill=0.0)
        for y in range(img.rh):
            for x in range(img.rw):
                for k in range(3):
                    out[(y * img.rw + x) * 3 + k] = min(max(full[(y * W + x) * oc + k], Float32(-1.0)), Float32(1.0))
        return out^


# ---------------------------------------------------------------------------------------------
# normals head


def normalize_normals(mut n: List[Float32], eps: Float32 = 1e-6):
    """In place: n /= max(|n|, eps) per pixel ([*, 3])."""
    for p in range(len(n) // 3):
        var x = n[p * 3]
        var y = n[p * 3 + 1]
        var z = n[p * 3 + 2]
        var l = max(sqrt(x * x + y * y + z * z), eps)
        n[p * 3] = x / l
        n[p * 3 + 1] = y / l
        n[p * 3 + 2] = z / l


def ensemble_normals(members: List[List[Float32]], reduction: Int, want_uncertainty: Bool) raises -> Tuple[
        List[Float32], List[Float32]]:
    """`MarigoldNormalsPipeline.ensemble_normals`: per pixel, the normalised mean, the cosine of
    each member to it (clamped to [-1, 1]), the member with the largest cosine (`closest`, first
    on ties) or the mean itself, and the mean angle / pi as uncertainty."""
    var E = len(members)
    if E == 0:
        raise Error("marigold: empty ensemble")
    var n = len(members[0]) // 3
    var mean = List[Float32](length=n * 3, fill=0.0)
    for e in range(E):
        for k in range(n * 3):
            mean[k] += members[e][k]
    for k in range(n * 3):
        mean[k] /= Float32(E)
    normalize_normals(mean)
    var out = List[Float32](length=n * 3, fill=0.0)
    var unc = List[Float32]()
    if want_uncertainty:
        unc = List[Float32](length=n, fill=0.0)
    for p in range(n):
        var best = -1
        var bs: Float32 = -2.0
        var acc: Float32 = 0.0
        for e in range(E):
            var s = mean[p * 3] * members[e][p * 3] + mean[p * 3 + 1] * members[e][p * 3 + 1] \
                + mean[p * 3 + 2] * members[e][p * 3 + 2]
            s = min(max(s, Float32(-1.0)), Float32(1.0))
            if s > bs:
                bs = s
                best = e
            if want_uncertainty:
                acc += acos(s)
        if want_uncertainty:
            unc[p] = Float32(Float64(acc / Float32(E)) / 3.141592653589793)
        for k in range(3):
            out[p * 3 + k] = mean[p * 3 + k] if reduction == ENSEMBLE_MEAN else members[best][p * 3 + k]
    return (out^, unc^)


struct NormalsResult(Movable):
    var normals: List[Float32]       # [h, w, 3] unit vectors, +X right, +Y up, +Z to the viewer
    var uncertainty: List[Float32]   # [h, w] in [0, 1] (empty unless ensemble >= 3)
    var h: Int
    var w: Int
    var latents: List[Float32]       # final latents [ensemble, h8, w8, 4] (reusable)

    def __init__(out self):
        self.normals = List[Float32]()
        self.uncertainty = List[Float32]()
        self.h = 0
        self.w = 0
        self.latents = List[Float32]()


struct MarigoldNormals(Movable):
    var core: MarigoldCore

    def __init__(out self, var core: MarigoldCore):
        self.core = core^

    @staticmethod
    def load(mut rt: Runtime, dir: String) raises -> MarigoldNormals:
        var core = MarigoldCore.load(rt, dir)
        if core.targets != 1:
            raise Error("marigold: normals expects a 4-channel prediction UNet")
        return MarigoldNormals(core^)

    def predict(mut self, mut rt: Runtime, rgb01: List[Float32], h: Int, w: Int, steps: Int = 0,
                ensemble: Int = 1, resolution: Int = -1, seed: Int = 0, reduction: Int = ENSEMBLE_CLOSEST,
                match_input: Bool = True, batch: Int = 0) raises -> NormalsResult:
        """Normals of an [h, w, 3] image in [0, 1]. `steps` 0 / `resolution` -1 = the model
        defaults (4 steps, 768 px); `resolution` 0 processes at the input size."""
        if ensemble < 1:
            raise Error("marigold: ensemble must be >= 1")
        var st = steps if steps > 0 else self.core.default_steps
        if st > 50:
            raise Error("marigold: the model supports 1 to 50 steps")
        var res = resolution if resolution >= 0 else self.core.default_resolution
        var img = self.core.preprocess(rgb01, h, w, res)
        var ilat = self.core.encode_image(rt, img)
        var h8 = img.ph // 8
        var w8 = img.pw // 8
        var noise = self.core.initial_noise(ensemble, h8, w8, seed)
        var lat = self.core.denoise(rt, ilat, h8, w8, noise^, ensemble, st, batch)
        var members = List[List[Float32]]()
        for e in range(ensemble):
            var m = self.core.decode(rt, lat, e, 0, h8, w8, img)
            normalize_normals(m)
            members.append(m^)
        var r = NormalsResult()
        var n: List[Float32]
        var unc = List[Float32]()
        if ensemble > 1:
            var t = ensemble_normals(members, reduction, ensemble >= 3)
            n = t[0].copy()
            unc = t[1].copy()
        else:
            n = members[0].copy()
        var oh = img.rh
        var ow = img.rw
        if match_input and (img.rh != h or img.rw != w):
            n = resize_bilinear(n, img.rh, img.rw, 3, h, w, False)
            normalize_normals(n)
            if len(unc) > 0:
                unc = resize_bilinear(unc, img.rh, img.rw, 1, h, w, False)
            oh = h
            ow = w
        r.normals = n^
        r.uncertainty = unc^
        r.h = oh
        r.w = ow
        r.latents = lat^
        return r^


def normals_to_pixels(n: List[Float32], count: Int, directx: Bool = False) -> List[UInt32]:
    """0xAARRGGBB normal-map pixels: rgb = 0.5 + 0.5 * n (DirectX convention flips green)."""
    var out = List[UInt32](length=count, fill=0)
    for i in range(count):
        var c = List[UInt32](length=3, fill=0)
        for k in range(3):
            var v = n[i * 3 + k]
            if k == 1 and directx:
                v = -v
            var u = min(max(v * Float32(0.5) + Float32(0.5), Float32(0.0)), Float32(1.0))
            c[k] = UInt32(Int(u * Float32(255.0) + Float32(0.5)))
        out[i] = UInt32(0xFF000000) | (c[0] << 16) | (c[1] << 8) | c[2]
    return out^


def uncertainty_to_pixels(u: List[Float32], count: Int) -> List[UInt32]:
    """Grey pixels, white = most uncertain (values scaled by 2 and clamped: angles near 90 degrees
    are rare)."""
    var out = List[UInt32](length=count, fill=0)
    for i in range(count):
        var g = UInt32(Int(min(max(u[i] * Float32(2.0), Float32(0.0)), Float32(1.0)) * Float32(255.0) + Float32(0.5)))
        out[i] = UInt32(0xFF000000) | (g << 16) | (g << 8) | g
    return out^
