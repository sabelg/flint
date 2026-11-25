"""TripoSG image-to-3D in pure Mojo (`flint.ml.triposg`).

One image in, a watertight triangle mesh out. Runs the released VAST-AI/TripoSG checkpoint
(MIT, ~1.5 B parameters) from its Hugging Face folder as is (`image_encoder_dinov2/`,
`transformer/`, `vae/` safetensors are mmapped; no conversion step), on the MAX GPU.

Pipeline (matching `TripoSGPipeline` with `guidance_scale` 7 and 50 steps):

  1. `prepare_image`: a valid alpha channel is used as is; otherwise the background is keyed
     (border-connected pixels close to the median border colour; the upstream RMBG-1.4 model is
     non-commercial and is not used), small specks are dropped, the object is composited on
     white, cropped to its bounding box and padded by 10 % to a square.
  2. `preprocess_dinov2`: BitImageProcessor, bit-exact: PIL's 8-bit bicubic resample (fixed-point
     separable filter) to a 256 short side, 224 centre crop, ImageNet normalisation.
  3. DINOv2-L/14 (24 layers, 257 tokens; position embeddings bicubic-resized 37 -> 16 like
     `transformers`) gives the condition `[257, 1024]`.
  4. Rectified-flow sampling: 2048 latent tokens x 64 channels from N(0, 1) (`torch_randn`
     reproduces `torch.randn` with `torch.manual_seed`), sigmas 1 - i / N, Euler steps
     x += (sigma_i - sigma_{i+1}) v, classifier-free guidance against an all-zero condition.
     The DiT: 21 blocks of width 2048 (16 heads x 128, RMS QK-norm, cross-attention to the
     image tokens, GELU MLP), U-Net style long skips into blocks 11..20, the timestep as an
     extra leading token. Flash-attention is replaced by plain softmax attention.
  5. The VAE decoder turns the latents into an SDF: 16 self-attention blocks over the latents
     (once), then per query point a Fourier embedding (3 -> 51), one cross-attention block and a
     linear head. `extract_sdf` evaluates it on a coarse 64^3 grid and refines only the cells
     next to the surface, doubling the resolution up to `res` (512 by default, like FlashVDM's
     octree depth 9); `marching_cubes` (in `flint.ml.triposg_mesh`) meshes the final grid.

Tripo's attention layers split heads before q / k / v (`view(..., heads, 3 * hd)`); the fused
projection weights are permuted at load so the kernels see the usual [q | k | v] layout.
"""

from std.math import sqrt, exp, log, sin, cos, floor
from std.collections import Dict
from std.time import perf_counter_ns
from std.ffi import external_call
from flint.platform.os import read_bytes
from flint.assets.image import decode_png
from flint.assets.json import parse_json
from flint.ml.tensor import fptr_at
from flint.ml.safetensors import SafeTensors
from flint.ml.gpu import FP
from flint.ml.triposg_gpu import TgGpu

comptime TG_TOKENS: Int = 2048
comptime TG_LATENT: Int = 64
comptime TG_WIDTH: Int = 2048
comptime TG_HEADS: Int = 16
comptime TG_LAYERS: Int = 21
comptime TG_COND: Int = 1024
comptime TG_COND_TOKENS: Int = 257
comptime TG_VAE_W: Int = 1024
comptime TG_VAE_HEADS: Int = 8
comptime TG_VAE_LAYERS: Int = 16
comptime TG_DINO_W: Int = 1024
comptime TG_DINO_LAYERS: Int = 24
comptime TG_BOUND: Float32 = 1.005


# ---------------------------------------------------------------------------------------------
# image preparation

struct TgImage(Copyable, Movable):
    """8-bit RGB image, rows top to bottom."""
    var width: Int
    var height: Int
    var rgb: List[UInt8]

    def __init__(out self, width: Int, height: Int):
        self.width = width
        self.height = height
        self.rgb = List[UInt8](length=width * height * 3, fill=0)


struct TgRgba(Copyable, Movable):
    var width: Int
    var height: Int
    var rgb: List[UInt8]
    var alpha: List[UInt8]
    var has_alpha: Bool

    def __init__(out self, width: Int, height: Int):
        self.width = width
        self.height = height
        self.rgb = List[UInt8](length=width * height * 3, fill=0)
        self.alpha = List[UInt8](length=width * height, fill=255)
        self.has_alpha = False


def load_rgba_png(path: String) raises -> TgRgba:
    var data = read_bytes(path)
    if len(data) < 32:
        raise Error("triposg: cannot read " + path)
    var color_type = Int(data[25])
    var img = decode_png(data, 1 << 15)
    if not img.valid:
        raise Error("triposg: cannot decode PNG " + path)
    var out = TgRgba(img.width, img.height)
    out.has_alpha = color_type == 4 or color_type == 6
    for k in range(img.width * img.height):
        var p = img.pixels[k]
        out.rgb[3 * k] = UInt8((p >> 16) & 255)
        out.rgb[3 * k + 1] = UInt8((p >> 8) & 255)
        out.rgb[3 * k + 2] = UInt8(p & 255)
        out.alpha[k] = UInt8((p >> 24) & 255)
    return out^


def alpha_is_valid(alpha: List[UInt8], count: Int) -> Bool:
    """TripoSG's test: >= 1 % of pixels in the lowest and in the highest of 20 alpha bins."""
    var lo = 0
    var hi = 0
    for i in range(count):
        var a = Int(alpha[i])
        if a * 20 < 256:
            lo += 1
        if a * 20 >= 256 * 19:
            hi += 1
    var m = Float64(count) * 0.01
    return Float64(lo) >= m and Float64(hi) >= m


def _median_int(mut v: List[Int]) -> Int:
    sort(v)
    var n = len(v)
    if n % 2 == 1:
        return v[n // 2]
    return Int((Float64(v[n // 2 - 1]) + Float64(v[n // 2])) / 2.0)


def key_background(img: TgRgba, tol: Int = 24) -> List[Bool]:
    """Foreground mask of an opaque image: background = pixels connected (4-neighbourhood) to the
    border whose colour is within `tol` (max channel difference) of the median border colour."""
    var w = img.width
    var h = img.height
    var bg = List[Int]()
    for c in range(3):
        var vals = List[Int]()
        for x in range(w):
            vals.append(Int(img.rgb[3 * x + c]))
        for x in range(w):
            vals.append(Int(img.rgb[3 * ((h - 1) * w + x) + c]))
        for y in range(h):
            vals.append(Int(img.rgb[3 * (y * w) + c]))
        for y in range(h):
            vals.append(Int(img.rgb[3 * (y * w + w - 1) + c]))
        bg.append(_median_int(vals))
    var near = List[Bool](length=w * h, fill=False)
    for k in range(w * h):
        var d = 0
        for c in range(3):
            d = max(d, abs(Int(img.rgb[3 * k + c]) - bg[c]))
        near[k] = d <= tol
    var seen = List[Bool](length=w * h, fill=False)
    var stack = List[Int]()
    for x in range(w):
        for y in [0, h - 1]:
            var k = y * w + x
            if near[k] and not seen[k]:
                seen[k] = True
                stack.append(k)
    for y in range(h):
        for x in [0, w - 1]:
            var k = y * w + x
            if near[k] and not seen[k]:
                seen[k] = True
                stack.append(k)
    while len(stack) > 0:
        var k = stack.pop()
        var y = k // w
        var x = k % w
        if y + 1 < h and near[k + w] and not seen[k + w]:
            seen[k + w] = True
            stack.append(k + w)
        if y > 0 and near[k - w] and not seen[k - w]:
            seen[k - w] = True
            stack.append(k - w)
        if x + 1 < w and near[k + 1] and not seen[k + 1]:
            seen[k + 1] = True
            stack.append(k + 1)
        if x > 0 and near[k - 1] and not seen[k - 1]:
            seen[k - 1] = True
            stack.append(k - 1)
    var fg = List[Bool](length=w * h, fill=False)
    for k in range(w * h):
        fg[k] = not seen[k]
    return fg^


def remove_small(mask: List[Bool], w: Int, h: Int, min_size: Int = 200) -> List[Bool]:
    """Drops 4-connected foreground components with fewer than `min_size` pixels."""
    var out = mask.copy()
    var lab = List[Bool](length=w * h, fill=False)
    for k0 in range(w * h):
        if not mask[k0] or lab[k0]:
            continue
        lab[k0] = True
        var comp = List[Int]()
        comp.append(k0)
        var i = 0
        while i < len(comp):
            var k = comp[i]
            i += 1
            var y = k // w
            var x = k % w
            if y + 1 < h and mask[k + w] and not lab[k + w]:
                lab[k + w] = True
                comp.append(k + w)
            if y > 0 and mask[k - w] and not lab[k - w]:
                lab[k - w] = True
                comp.append(k - w)
            if x + 1 < w and mask[k + 1] and not lab[k + 1]:
                lab[k + 1] = True
                comp.append(k + 1)
            if x > 0 and mask[k - 1] and not lab[k - 1]:
                lab[k - 1] = True
                comp.append(k - 1)
        if len(comp) < min_size:
            for c in comp:
                out[c] = False
    return out^


def prepare_image(img: TgRgba, padding_ratio: Float64 = 0.1) raises -> TgImage:
    """TripoSG `load_image` (alpha path, white background) with background keying instead of
    RMBG: composite, crop to the foreground box, pad to a square with a 10 % margin."""
    var w = img.width
    var h = img.height
    var a = List[Float32](length=w * h, fill=0.0)
    if img.has_alpha and alpha_is_valid(img.alpha, w * h):
        for k in range(w * h):
            a[k] = Float32(img.alpha[k]) / 255.0
    else:
        var fg = remove_small(key_background(img), w, h)
        for k in range(w * h):
            a[k] = 1.0 if fg[k] else 0.0
    var x0 = w
    var y0 = h
    var x1 = -1
    var y1 = -1
    for y in range(h):
        for x in range(w):
            if a[y * w + x] > 0:
                x0 = min(x0, x)
                x1 = max(x1, x)
                y0 = min(y0, y)
                y1 = max(y1, y)
    if x1 < 0:
        raise Error("triposg: empty foreground")
    var bw = x1 - x0 + 1
    var bh = y1 - y0 + 1
    var px: Int
    var py: Int
    if bw > bh:
        px = Int(Float64(bw) * padding_ratio)
        py = Int(Float64(px) + Float64(bw - bh) / 2.0)
    else:
        py = Int(Float64(bh) * padding_ratio)
        px = Int(Float64(py) + Float64(bh - bw) / 2.0)
    var ow = bw + 2 * px
    var oh = bh + 2 * py
    var out = TgImage(ow, oh)
    for k in range(ow * oh * 3):
        out.rgb[k] = 255
    for y in range(bh):
        for x in range(bw):
            var sk = (y0 + y) * w + x0 + x
            var al = a[sk]
            var dk = (py + y) * ow + px + x
            for c in range(3):
                var v = Float32(img.rgb[3 * sk + c]) / 255.0
                v = v * al + 1.0 * (1.0 - al)
                out.rgb[3 * dk + c] = UInt8(Int(v * 255.0))
    return out^


def _pil_bicubic(x0: Float64) -> Float64:
    var x = x0 if x0 >= 0 else -x0
    comptime A = -0.5
    if x < 1.0:
        return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * A
    return 0.0


struct _PilCoeffs(Movable):
    var ksize: Int
    var bounds: List[Int]   # [xmin, count] per output
    var k: List[Int]        # fixed-point coefficients, ksize per output

    def __init__(out self, in_size: Int, out_size: Int):
        var scale = Float64(in_size) / Float64(out_size)
        var fscale = scale if scale > 1.0 else 1.0
        var support = 2.0 * fscale
        self.ksize = Int(support + 0.999999999999) * 2 + 1
        self.bounds = List[Int](length=out_size * 2, fill=0)
        self.k = List[Int](length=out_size * self.ksize, fill=0)
        var tmp = List[Float64](length=self.ksize, fill=0.0)
        for xx in range(out_size):
            var center = (Float64(xx) + 0.5) * scale
            var ss = 1.0 / fscale
            var xmin = Int(center - support + 0.5)
            if xmin < 0:
                xmin = 0
            var xmax = Int(center + support + 0.5)
            if xmax > in_size:
                xmax = in_size
            xmax -= xmin
            var ww = 0.0
            for x in range(xmax):
                var wv = _pil_bicubic((Float64(x + xmin) - center + 0.5) * ss)
                tmp[x] = wv
                ww += wv
            for x in range(xmax):
                var kv = tmp[x] / ww if ww != 0.0 else tmp[x]
                var f = kv * Float64(1 << 22)
                self.k[xx * self.ksize + x] = Int(-0.5 + f) if kv < 0 else Int(0.5 + f)
            self.bounds[2 * xx] = xmin
            self.bounds[2 * xx + 1] = xmax


def _clip8(ss: Int) -> UInt8:
    var v = ss >> 22
    if ss < 0:
        return 0
    if v > 255:
        return 255
    return UInt8(v)


def pil_resize_bicubic(img: TgImage, ow: Int, oh: Int) -> TgImage:
    """Pillow's `Image.resize(..., BICUBIC)` for 8-bit RGB, bit-exact (horizontal pass into an
    8-bit image, then vertical; 22-bit fixed-point coefficients)."""
    var hc = _PilCoeffs(img.width, ow)
    var vc = _PilCoeffs(img.height, oh)
    var yfirst = vc.bounds[0]
    var ylast = vc.bounds[2 * oh - 2] + vc.bounds[2 * oh - 1]
    var th = ylast - yfirst
    var tmp = TgImage(ow, th)
    var need_h = ow != img.width
    if need_h:
        for yy in range(th):
            for xx in range(ow):
                var xmin = hc.bounds[2 * xx]
                var xn = hc.bounds[2 * xx + 1]
                for c in range(3):
                    var ss = 1 << 21
                    for x in range(xn):
                        ss += Int(img.rgb[3 * ((yy + yfirst) * img.width + xmin + x) + c]) * hc.k[xx * hc.ksize + x]
                    tmp.rgb[3 * (yy * ow + xx) + c] = _clip8(ss)
    var out = TgImage(ow, oh)
    for yy in range(oh):
        var ymin = vc.bounds[2 * yy] - (yfirst if need_h else 0)
        var yn = vc.bounds[2 * yy + 1]
        for xx in range(ow):
            for c in range(3):
                var ss = 1 << 21
                for y in range(yn):
                    var v: Int
                    if need_h:
                        v = Int(tmp.rgb[3 * ((ymin + y) * ow + xx) + c])
                    else:
                        v = Int(img.rgb[3 * ((ymin + y) * img.width + xx) + c])
                    ss += v * vc.k[yy * vc.ksize + y]
                out.rgb[3 * (yy * ow + xx) + c] = _clip8(ss)
    return out^


def preprocess_dinov2(img: TgImage) -> List[Float32]:
    """BitImageProcessor (shortest edge 256, bicubic; centre crop 224; ImageNet mean / std):
    float32 [3, 224, 224]."""
    var w = img.width
    var h = img.height
    var nw: Int
    var nh: Int
    if w <= h:
        nw = 256
        nh = Int(256.0 * Float64(h) / Float64(w))
    else:
        nh = 256
        nw = Int(256.0 * Float64(w) / Float64(h))
    var r = img.copy() if (nw == w and nh == h) else pil_resize_bicubic(img, nw, nh)
    var top = (nh - 224) // 2
    var left = (nw - 224) // 2
    var mean: List[Float32] = [0.485, 0.456, 0.406]
    var std: List[Float32] = [0.229, 0.224, 0.225]
    var out = List[Float32](length=3 * 224 * 224, fill=0.0)
    for c in range(3):
        for y in range(224):
            for x in range(224):
                var v = Float32(Float64(r.rgb[3 * ((top + y) * nw + left + x) + c]) * 0.00392156862745098)
                out[(c * 224 + y) * 224 + x] = (v - mean[c]) / std[c]
    return out^


# ---------------------------------------------------------------------------------------------
# noise

struct TorchMt19937(Movable):
    """`torch.manual_seed(seed)` CPU generator (mt19937)."""
    var mt: List[UInt32]
    var idx: Int

    def __init__(out self, seed: UInt64):
        self.mt = List[UInt32](length=624, fill=0)
        self.mt[0] = UInt32(seed & 0xFFFFFFFF)
        for i in range(1, 624):
            var prev = self.mt[i - 1]
            self.mt[i] = UInt32(1812433253) * (prev ^ (prev >> 30)) + UInt32(i)
        self.idx = 624

    def next(mut self) -> UInt32:
        if self.idx >= 624:
            for i in range(624):
                var y = (self.mt[i] & 0x80000000) | (self.mt[(i + 1) % 624] & 0x7FFFFFFF)
                var v = self.mt[(i + 397) % 624] ^ (y >> 1)
                if (y & 1) != 0:
                    v ^= 0x9908B0DF
                self.mt[i] = v
            self.idx = 0
        var y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y


def torch_randn(n: Int, seed: UInt64) -> List[Float32]:
    """`torch.randn(n, generator=torch.Generator().manual_seed(seed))` on the CPU for n a multiple
    of 16 (uniforms from 24 random bits, then Box-Muller over blocks of 16)."""
    var g = TorchMt19937(seed)
    var d = List[Float32](length=n, fill=0.0)
    for i in range(n):
        d[i] = Float32(Float64(Int(g.next() & 0xFFFFFF)) * (1.0 / 16777216.0))
    var i = 0
    while i + 16 <= n:
        for j in range(8):
            var u1 = 1.0 - d[i + j]
            var u2 = d[i + j + 8]
            var radius = sqrt(-2.0 * log(u1))
            var theta = Float32(2.0 * 3.14159265358979323846) * u2
            d[i + j] = radius * cos(theta)
            d[i + j + 8] = radius * sin(theta)
        i += 16
    return d^


def rf_sigmas(steps: Int) -> List[Float32]:
    """RectifiedFlowScheduler (shift 1): sigma_i = 1 - i / steps (float64 -> float32), then 0."""
    var s = List[Float32]()
    for i in range(steps):
        s.append(Float32((1.0 - Float64(i) / Float64(steps)) * 1000.0 / 1000.0))
    s.append(0.0)
    return s^


def euler_step(mut x: List[Float32], v: List[Float32], sigma: Float32, sigma_next: Float32):
    for i in range(len(x)):
        x[i] = x[i] + (sigma - sigma_next) * v[i]


def cfg_combine(vu: List[Float32], vc: List[Float32], scale: Float32) -> List[Float32]:
    var o = List[Float32](length=len(vu), fill=0.0)
    for i in range(len(vu)):
        o[i] = vu[i] + scale * (vc[i] - vu[i])
    return o^


def timestep_embedding(t: Float32, dim: Int) -> List[Float32]:
    """diffusers `get_timestep_embedding` (flip_sin_to_cos False, shift 0): [sin, cos]."""
    var half = dim // 2
    var o = List[Float32](length=dim, fill=0.0)
    for i in range(half):
        var e = exp(Float32(-log(10000.0)) * Float32(i) / Float32(half))
        var a = t * e
        o[i] = sin(a)
        o[half + i] = cos(a)
    return o^


def _cubic1(x: Float32, a: Float32) -> Float32:
    return ((a + 2) * x - (a + 3)) * x * x + 1


def _cubic2(x: Float32, a: Float32) -> Float32:
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a


def bicubic_resize_grid(src: List[Float32], n_in: Int, n_out: Int, ch: Int) -> List[Float32]:
    """torch `interpolate(mode="bicubic", align_corners=False)` of a [n_in, n_in, ch] grid
    (channels last) to [n_out, n_out, ch]."""
    var scale = Float32(n_in) / Float32(n_out)
    var idx = List[Int](length=n_out * 4, fill=0)
    var wts = List[Float32](length=n_out * 4, fill=0.0)
    for o in range(n_out):
        var s = scale * (Float32(o) + 0.5) - 0.5
        var i0 = Int(floor(s))
        var t = s - Float32(i0)
        var ws: List[Float32] = [_cubic2(t + 1, -0.75), _cubic1(t, -0.75), _cubic1(1 - t, -0.75), _cubic2(2 - t, -0.75)]
        for k in range(4):
            idx[o * 4 + k] = min(max(i0 - 1 + k, 0), n_in - 1)
            wts[o * 4 + k] = ws[k]
    var out = List[Float32](length=n_out * n_out * ch, fill=0.0)
    for oy in range(n_out):
        for ox in range(n_out):
            for c in range(ch):
                var acc: Float32 = 0
                for ky in range(4):
                    var row: Float32 = 0
                    for kx in range(4):
                        row += src[(idx[oy * 4 + ky] * n_in + idx[ox * 4 + kx]) * ch + c] * wts[ox * 4 + kx]
                    acc += row * wts[oy * 4 + ky]
                out[(oy * n_out + ox) * ch + c] = acc
    return out^


# ---------------------------------------------------------------------------------------------
# model

struct TripoSG(Movable):
    var g: TgGpu
    var w: Dict[String, Int]
    var key: Int
    var dino: SafeTensors
    var dit: SafeTensors
    var vae: SafeTensors
    var slots: Dict[String, Int]
    var dit_batch: Int
    var vae_ready: Bool
    var verbose: Bool
    var chunk: Int

    def __init__(out self, var dino: SafeTensors, var dit: SafeTensors, var vae: SafeTensors) raises:
        self.g = TgGpu()
        self.w = Dict[String, Int]()
        self.key = 1 << 40
        self.dino = dino^
        self.dit = dit^
        self.vae = vae^
        self.slots = Dict[String, Int]()
        self.dit_batch = 0
        self.vae_ready = False
        self.verbose = False
        self.chunk = 8192

    @staticmethod
    def load(dir: String) raises -> TripoSG:
        """Maps the three safetensors files of a VAST-AI/TripoSG folder and uploads the weights."""
        var m = TripoSG(SafeTensors.open(dir + "/image_encoder_dinov2/model.safetensors"),
                        SafeTensors.open(dir + "/transformer/diffusion_pytorch_model.safetensors"),
                        SafeTensors.open(dir + "/vae/diffusion_pytorch_model.safetensors"))
        m._upload_dino()
        m._upload_dit()
        m._upload_vae()
        return m^

    def device(self) raises -> String:
        return self.g.name()

    # ---- weights

    def _addr(self, st: SafeTensors, name: String) raises -> Int:
        var e = st.entries.get(name)
        if not e:
            raise Error("triposg: missing tensor " + name)
        if e.value().dtype != "F32":
            raise Error("triposg: expected F32 for " + name + " (got " + e.value().dtype + ")")
        return st.data_start + e.value().begin

    def _numel(self, st: SafeTensors, name: String) raises -> Int:
        var n = 1
        for d in st.shape(name):
            n *= d
        return n

    def _put(mut self, key: String, addr: Int, n: Int) raises:
        self.key += 1
        self.w[key] = self.g.mm.weight(self.key, addr, 1, n, False)

    def _put_list(mut self, key: String, data: List[Float32]) raises:
        self._put(key, Int(data.unsafe_ptr()), len(data))

    def _up(mut self, which: Int, name: String, prefix: String) raises:
        if which == 0:
            self._put(prefix + name, self._addr(self.dino, name), self._numel(self.dino, name))
        elif which == 1:
            self._put(prefix + name, self._addr(self.dit, name), self._numel(self.dit, name))
        else:
            self._put(prefix + name, self._addr(self.vae, name), self._numel(self.vae, name))

    def _st_floats(self, which: Int, name: String) raises -> List[Float32]:
        var addr: Int
        var n: Int
        if which == 0:
            addr = self._addr(self.dino, name)
            n = self._numel(self.dino, name)
        elif which == 1:
            addr = self._addr(self.dit, name)
            n = self._numel(self.dit, name)
        else:
            addr = self._addr(self.vae, name)
            n = self._numel(self.vae, name)
        var out = List[Float32](length=n, fill=0.0)
        _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), addr, n * 4)
        return out^

    def _tripo_qkv(mut self, which: Int, p: String, D: Int, heads: Int, prefix: String) raises:
        """Fused [3D, D] q/k/v weight with rows reordered from Tripo's per-head split."""
        var q = self._st_floats(which, p + "to_q.weight")
        var k = self._st_floats(which, p + "to_k.weight")
        var v = self._st_floats(which, p + "to_v.weight")
        var hd = D // heads
        var out = List[Float32](length=3 * D * D, fill=0.0)
        for part in range(3):
            for h in range(heads):
                for i in range(hd):
                    var r = h * 3 * hd + part * hd + i      # row of cat(q, k, v)
                    var dst = part * D + h * hd + i
                    var srow = r % D
                    var sp = r // D
                    var src = Int(q.unsafe_ptr()) if sp == 0 else (Int(k.unsafe_ptr()) if sp == 1 else Int(v.unsafe_ptr()))
                    _ = external_call["memcpy", Int](Int(out.unsafe_ptr()) + dst * D * 4, src + srow * D * 4, D * 4)
        self._put_list(prefix + p + "qkv", out)

    def _tripo_kv(mut self, which: Int, p: String, D: Int, Dc: Int, heads: Int, prefix: String) raises:
        """Fused [2D, Dc] k/v weight with rows reordered from Tripo's per-head split."""
        var k = self._st_floats(which, p + "to_k.weight")
        var v = self._st_floats(which, p + "to_v.weight")
        var hd = D // heads
        var out = List[Float32](length=2 * D * Dc, fill=0.0)
        for part in range(2):
            for h in range(heads):
                for i in range(hd):
                    var r = h * 2 * hd + part * hd + i
                    var dst = part * D + h * hd + i
                    var src = Int(k.unsafe_ptr()) if r < D else Int(v.unsafe_ptr())
                    _ = external_call["memcpy", Int](Int(out.unsafe_ptr()) + dst * Dc * 4, src + (r % D) * Dc * 4, Dc * 4)
        self._put_list(prefix + p + "kv", out)

    def _upload_dino(mut self) raises:
        var names: List[String] = ["embeddings.patch_embeddings.projection.weight",
                                   "embeddings.patch_embeddings.projection.bias", "layernorm.weight", "layernorm.bias"]
        for n in names:
            self._up(0, n, "dino.")
        # position embeddings: class token + 37 x 37 grid resized to 16 x 16 (224 / 14)
        var pos = self._st_floats(0, "embeddings.position_embeddings")
        var cls = self._st_floats(0, "embeddings.cls_token")
        var grid = List[Float32](length=37 * 37 * TG_DINO_W, fill=0.0)
        for i in range(37 * 37 * TG_DINO_W):
            grid[i] = pos[TG_DINO_W + i]
        var g16 = bicubic_resize_grid(grid, 37, 16, TG_DINO_W)
        var p = List[Float32](length=257 * TG_DINO_W, fill=0.0)
        for i in range(TG_DINO_W):
            p[i] = pos[i] + cls[i]          # row 0 holds cls_token + its position
        for i in range(256 * TG_DINO_W):
            p[TG_DINO_W + i] = g16[i]
        self._put_list("dino.pos", p)
        for l in range(TG_DINO_LAYERS):
            var b = "encoder.layer." + String(l) + "."
            var qkv = List[Float32]()
            var qkvb = List[Float32]()
            var parts: List[String] = ["query", "key", "value"]
            for part in parts:
                qkv.extend(self._st_floats(0, b + "attention.attention." + part + ".weight"))
                qkvb.extend(self._st_floats(0, b + "attention.attention." + part + ".bias"))
            self._put_list("dino." + b + "qkv", qkv)
            self._put_list("dino." + b + "qkv_b", qkvb)
            var rest: List[String] = ["attention.output.dense.weight", "attention.output.dense.bias",
                                      "layer_scale1.lambda1", "layer_scale2.lambda1", "mlp.fc1.weight", "mlp.fc1.bias",
                                      "mlp.fc2.weight", "mlp.fc2.bias", "norm1.weight", "norm1.bias", "norm2.weight",
                                      "norm2.bias"]
            for n in rest:
                self._up(0, b + n, "dino.")

    def _upload_dit(mut self) raises:
        var top: List[String] = ["proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias", "norm_out.weight",
                                 "norm_out.bias", "time_proj.linear_1.weight", "time_proj.linear_1.bias",
                                 "time_proj.linear_2.weight", "time_proj.linear_2.bias"]
        for n in top:
            self._up(1, n, "dit.")
        for l in range(TG_LAYERS):
            var b = "blocks." + String(l) + "."
            self._tripo_qkv(1, b + "attn1.", TG_WIDTH, TG_HEADS, "dit.")
            self._tripo_kv(1, b + "attn2.", TG_WIDTH, TG_COND, TG_HEADS, "dit.")
            var rest: List[String] = ["attn1.norm_q.weight", "attn1.norm_k.weight", "attn1.to_out.0.weight",
                                      "attn1.to_out.0.bias", "attn2.to_q.weight", "attn2.norm_q.weight",
                                      "attn2.norm_k.weight", "attn2.to_out.0.weight", "attn2.to_out.0.bias",
                                      "ff.net.0.proj.weight", "ff.net.0.proj.bias", "ff.net.2.weight", "ff.net.2.bias",
                                      "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "norm3.weight",
                                      "norm3.bias"]
            for n in rest:
                self._up(1, b + n, "dit.")
            if l > TG_LAYERS // 2:
                var sk: List[String] = ["skip_linear.weight", "skip_linear.bias", "skip_norm.weight", "skip_norm.bias"]
                for n in sk:
                    self._up(1, b + n, "dit.")

    def _upload_vae(mut self) raises:
        var top: List[String] = ["post_quant.weight", "post_quant.bias", "decoder.proj_query.weight",
                                 "decoder.proj_query.bias", "decoder.norm_out.weight", "decoder.norm_out.bias",
                                 "decoder.proj_out.weight", "decoder.proj_out.bias"]
        for n in top:
            self._up(2, n, "vae.")
        for l in range(TG_VAE_LAYERS):
            var b = "decoder.blocks." + String(l) + "."
            self._tripo_qkv(2, b + "attn1.", TG_VAE_W, TG_VAE_HEADS, "vae.")
            var rest: List[String] = ["attn1.to_out.0.weight", "attn1.to_out.0.bias", "ff.net.0.proj.weight",
                                      "ff.net.0.proj.bias", "ff.net.2.weight", "ff.net.2.bias", "norm1.weight",
                                      "norm1.bias", "norm3.weight", "norm3.bias"]
            for n in rest:
                self._up(2, b + n, "vae.")
        var c = "decoder.blocks." + String(TG_VAE_LAYERS) + "."
        self._tripo_kv(2, c + "attn2.", TG_VAE_W, TG_VAE_W, TG_VAE_HEADS, "vae.")
        var rest: List[String] = ["attn2.to_q.weight", "attn2.to_out.0.weight", "attn2.to_out.0.bias",
                                  "attn2.norm_cross.weight", "attn2.norm_cross.bias", "ff.net.0.proj.weight",
                                  "ff.net.0.proj.bias", "ff.net.2.weight", "ff.net.2.bias", "norm2.weight",
                                  "norm2.bias", "norm3.weight", "norm3.bias"]
        for n in rest:
            self._up(2, c + n, "vae.")

    def P(self, key: String) raises -> FP:
        return self.g.mm.wptr(self.w[key])

    def S(mut self, name: String, n: Int) raises -> FP:
        """A named device workspace of at least n floats (allocated once)."""
        var found = self.slots.get(name)
        if found:
            var s = found.value()
            if self.g.mm.pool_cap[s] >= n:
                return self.g.mm.ptr(s)
            self.g.mm.release(s)
        var s = self.g.mm.alloc(n)
        self.slots[name] = s
        return self.g.mm.ptr(s)

    # ---- building blocks

    def lin(mut self, x: FP, wkey: String, bkey: String, dst: FP, rows: Int, k: Int, n: Int,
            accum: Bool = False) raises:
        var hb = bkey != ""
        var pb = self.P(bkey) if hb else dst
        self.g.gemm(x, self.P(wkey), dst, rows, k, n, k, k, n, True, pb, hb, accum)

    def ln(mut self, x: FP, key: String, dst: FP, rows: Int, n: Int, eps: Float32) raises:
        self.g.layer_norm(x, self.P(key + ".weight"), self.P(key + ".bias"), dst, rows, n, eps)

    def attention(mut self, q: FP, ldq: Int, q_bs: Int, k: FP, ldk: Int, v: FP, ldv: Int, kv_bs: Int, o: FP,
                  ldo: Int, o_bs: Int, B: Int, L: Int, S: Int, H: Int, hd: Int, scores: FP) raises:
        """o[b, l, h] = softmax(q k^T / sqrt(hd)) v per (batch, head); row strides ld*, batch
        strides *_bs (floats); head h lives at column h * hd of q / k / v / o."""
        var scale = Float32(1.0 / sqrt(Float64(hd)))
        var sp = (S + 3) // 4 * 4    # padded score rows keep 16-byte alignment for the P V product
        self.g.gemm(q, k, scores, L, hd, S, ldq, ldk, sp, True, scores, False, False, scale, B, H, q_bs, hd, kv_bs, hd,
                    H * L * sp, L * sp)
        self.g.softmax(scores, B * H * L, S, sp)
        self.g.gemm(scores, v, o, L, S, hd, sp, ldv, ldo, False, o, False, False, 1.0, B, H, H * L * sp, L * sp, kv_bs,
                    hd, o_bs, hd)

    # ---- DINOv2

    def encode_image(mut self, pixels: List[Float32]) raises -> List[Float32]:
        """DINOv2-L last_hidden_state for one [3, 224, 224] input: [257, 1024]."""
        comptime D = TG_DINO_W
        comptime T = 257
        var patches = List[Float32](length=256 * 588, fill=0.0)
        for py in range(16):
            for px in range(16):
                var r = py * 16 + px
                for c in range(3):
                    for ky in range(14):
                        for kx in range(14):
                            patches[r * 588 + (c * 14 + ky) * 14 + kx] = pixels[(c * 224 + py * 14 + ky) * 224 + px * 14 + kx]
        var pin = self.S("d.in", 256 * 588)
        var s_in = self.slots["d.in"]
        self.g.mm.upload(s_in, Int(patches.unsafe_ptr()), 256 * 588)
        var h = self.S("d.h", T * D)
        var n = self.S("d.n", T * D)
        var qkv = self.S("d.qkv", T * 3 * D)
        var att = self.S("d.att", T * D)
        var tmp = self.S("d.tmp", T * D)
        var ff = self.S("d.ff", T * 4 * D)
        var sc = self.S("d.sc", 16 * T * (T + 3))
        # row 0 = 0 (cls comes with the position table), rows 1.. = patch embeddings
        self.g.gemm(pin, self.P("dino.embeddings.patch_embeddings.projection.weight"), h.unsafe_offset(D), 256, 588, D, 588, 588, D,
                    True, self.P("dino.embeddings.patch_embeddings.projection.bias"), True)
        self.g.copy2d(self.P("dino.pos"), h, 1, D, D, D)
        self.g.add_scaled(h.unsafe_offset(D), self.P("dino.pos").unsafe_offset(D), h, 256 * D, 0)
        for l in range(TG_DINO_LAYERS):
            var b = "dino.encoder.layer." + String(l) + "."
            self.ln(h, b + "norm1", n, T, D, 1e-6)
            self.lin(n, b + "qkv", b + "qkv_b", qkv, T, D, 3 * D)
            self.attention(qkv, 3 * D, 0, qkv.unsafe_offset(D), 3 * D, qkv.unsafe_offset(2 * D), 3 * D, 0, att, D, 0, 1, T, T, 16, 64, sc)
            self.lin(att, b + "attention.output.dense.weight", b + "attention.output.dense.bias", tmp, T, D, D)
            self.g.add_scaled(h, tmp, self.P(b + "layer_scale1.lambda1"), T * D, D)
            self.ln(h, b + "norm2", n, T, D, 1e-6)
            self.lin(n, b + "mlp.fc1.weight", b + "mlp.fc1.bias", ff, T, D, 4 * D)
            self.g.gelu(ff, T * 4 * D)
            self.lin(ff, b + "mlp.fc2.weight", b + "mlp.fc2.bias", tmp, T, 4 * D, D)
            self.g.add_scaled(h, tmp, self.P(b + "layer_scale2.lambda1"), T * D, D)
        self.ln(h, "dino.layernorm", n, T, D, 1e-6)
        var out = List[Float32](length=T * D, fill=0.0)
        self.g.mm.download(self.slots["d.n"], Int(out.unsafe_ptr()), T * D)
        return out^

    # ---- DiT

    def set_condition(mut self, cond: List[Float32], cfg: Bool) raises:
        """Caches every block's cross-attention K / V for [zeros, cond] (cfg) or [cond]."""
        comptime D = TG_WIDTH
        comptime C = TG_COND_TOKENS
        var B = 2 if cfg else 1
        self.dit_batch = B
        var enc = List[Float32](length=B * C * TG_COND, fill=0.0)
        var off = (B - 1) * C * TG_COND
        for i in range(C * TG_COND):
            enc[off + i] = cond[i]
        _ = self.S("t.enc", B * C * TG_COND)
        self.g.mm.upload(self.slots["t.enc"], Int(enc.unsafe_ptr()), B * C * TG_COND)
        var pe = self.g.mm.ptr(self.slots["t.enc"])
        for l in range(TG_LAYERS):
            var b = "dit.blocks." + String(l) + "."
            var kv = self.S("t.kv" + String(l), B * C * 2 * D)
            self.lin(pe, b + "attn2.kv", "", kv, B * C, TG_COND, 2 * D)
            self.g.head_rms(kv, self.P(b + "attn2.norm_k.weight"), B * C, TG_HEADS, D // TG_HEADS, 2 * D, 1e-6)

    def velocity(mut self, latents: List[Float32], t: Float32) raises -> List[Float32]:
        """DiT output for every batch entry ([B, 2048, 64]); `set_condition` first."""
        comptime D = TG_WIDTH
        comptime N = TG_TOKENS
        comptime L = N + 1
        comptime H = TG_HEADS
        comptime HD = D // H
        comptime C = TG_COND_TOKENS
        var B = self.dit_batch
        var R = B * L
        var h = self.S("t.h", R * D)
        var n = self.S("t.n", R * D)
        var qkv = self.S("t.qkv", R * 3 * D)
        var att = self.S("t.att", R * D)
        var ff = self.S("t.ff", R * 4 * D)
        var sc = self.S("t.sc", B * H * L * (L + 3))
        var lat = self.S("t.lat", N * TG_LATENT)
        self.g.mm.upload(self.slots["t.lat"], Int(latents.unsafe_ptr()), N * TG_LATENT)
        var te = timestep_embedding(t, D)
        var tin = self.S("t.te", D)
        self.g.mm.upload(self.slots["t.te"], Int(te.unsafe_ptr()), D)
        var t1 = self.S("t.t1", 4 * D)
        self.lin(tin, "dit.time_proj.linear_1.weight", "dit.time_proj.linear_1.bias", t1, 1, D, 4 * D)
        self.g.gelu(t1, 4 * D)
        self.lin(t1, "dit.time_proj.linear_2.weight", "dit.time_proj.linear_2.bias", h, 1, 4 * D, D)
        for bb in range(1, B):
            self.g.copy2d(h, h.unsafe_offset(bb * L * D), 1, D, D, D)
        self.g.gemm(lat, self.P("dit.proj_in.weight"), h.unsafe_offset(D), N, TG_LATENT, D, TG_LATENT, TG_LATENT, D, True,
                    self.P("dit.proj_in.bias"), True, False, 1.0, B, 1, 0, 0, 0, 0, L * D, 0)
        for l in range(TG_LAYERS):
            var b = "dit.blocks." + String(l) + "."
            if l > TG_LAYERS // 2:
                var skip = self.S("t.skip" + String(TG_LAYERS - 1 - l), R * D)
                self.g.copy2d(skip, ff, R, D, D, 2 * D)
                self.g.copy2d(h, ff.unsafe_offset(D), R, D, D, 2 * D)
                self.lin(ff, b + "skip_linear.weight", b + "skip_linear.bias", h, R, 2 * D, D)
                self.ln(h, b + "skip_norm", h, R, D, 1e-5)
            # self-attention
            self.ln(h, b + "norm1", n, R, D, 1e-5)
            self.lin(n, b + "attn1.qkv", "", qkv, R, D, 3 * D)
            self.g.head_rms(qkv, self.P(b + "attn1.norm_q.weight"), R, H, HD, 3 * D, 1e-6)
            self.g.head_rms(qkv.unsafe_offset(D), self.P(b + "attn1.norm_k.weight"), R, H, HD, 3 * D, 1e-6)
            self.attention(qkv, 3 * D, L * 3 * D, qkv.unsafe_offset(D), 3 * D, qkv.unsafe_offset(2 * D), 3 * D, L * 3 * D, att, D, L * D, B, L, L,
                           H, HD, sc)
            self.lin(att, b + "attn1.to_out.0.weight", b + "attn1.to_out.0.bias", h, R, D, D, True)
            # cross-attention to the image tokens
            self.ln(h, b + "norm2", n, R, D, 1e-5)
            self.lin(n, b + "attn2.to_q.weight", "", qkv, R, D, D)
            self.g.head_rms(qkv, self.P(b + "attn2.norm_q.weight"), R, H, HD, D, 1e-6)
            var kv = self.S("t.kv" + String(l), B * C * 2 * D)
            self.attention(qkv, D, L * D, kv, 2 * D, kv.unsafe_offset(D), 2 * D, C * 2 * D, att, D, L * D, B, L, C, H, HD, sc)
            self.lin(att, b + "attn2.to_out.0.weight", b + "attn2.to_out.0.bias", h, R, D, D, True)
            # MLP
            self.ln(h, b + "norm3", n, R, D, 1e-5)
            self.lin(n, b + "ff.net.0.proj.weight", b + "ff.net.0.proj.bias", ff, R, D, 4 * D)
            self.g.gelu(ff, R * 4 * D)
            self.lin(ff, b + "ff.net.2.weight", b + "ff.net.2.bias", h, R, 4 * D, D, True)
            if l < TG_LAYERS // 2:
                var skip = self.S("t.skip" + String(l), R * D)
                self.g.copy2d(h, skip, R, D, D, D)
        self.ln(h, "dit.norm_out", n, R, D, 1e-5)
        var vo = self.S("t.v", B * N * TG_LATENT)
        self.g.gemm(n.unsafe_offset(D), self.P("dit.proj_out.weight"), vo, N, D, TG_LATENT, D, D, TG_LATENT, True,
                    self.P("dit.proj_out.bias"), True, False, 1.0, B, 1, L * D, 0, 0, 0, N * TG_LATENT, 0)
        var out = List[Float32](length=B * N * TG_LATENT, fill=0.0)
        self.g.mm.download(self.slots["t.v"], Int(out.unsafe_ptr()), B * N * TG_LATENT)
        return out^

    def sample(mut self, cond: List[Float32], var latents: List[Float32], steps: Int = 50, guidance: Float32 = 7.0,
               dump_dir: String = "") raises -> List[Float32]:
        """Rectified-flow Euler sampling from `latents` (noise, [2048, 64]); returns clean latents."""
        var cfg = guidance > 1.0
        self.set_condition(cond, cfg)
        var sig = rf_sigmas(steps)
        comptime NL = TG_TOKENS * TG_LATENT
        for i in range(steps):
            var t0 = perf_counter_ns()
            var vb = self.velocity(latents, sig[i] * 1000.0)
            var v: List[Float32]
            if cfg:
                var vu = List[Float32](length=NL, fill=0.0)
                var vc = List[Float32](length=NL, fill=0.0)
                for k in range(NL):
                    vu[k] = vb[k]
                    vc[k] = vb[NL + k]
                v = cfg_combine(vu, vc, guidance)
            else:
                v = vb^
            euler_step(latents, v, sig[i], sig[i + 1])
            if dump_dir != "":
                write_f32(dump_dir + "/mojo_v_" + _two(i) + ".bin", v)
            if self.verbose:
                print("[triposg] step", i, "t =", sig[i] * 1000.0, "in", Float64(perf_counter_ns() - t0) / 1e6, "ms")
        return latents^

    # ---- VAE decoder

    def set_latents(mut self, latents: List[Float32]) raises:
        """Runs the decoder's self-attention trunk once and caches the cross-attention K / V."""
        comptime D = TG_VAE_W
        comptime N = TG_TOKENS
        comptime H = TG_VAE_HEADS
        var lat = self.S("v.lat", N * TG_LATENT)
        self.g.mm.upload(self.slots["v.lat"], Int(latents.unsafe_ptr()), N * TG_LATENT)
        var h = self.S("v.h", N * D)
        var n = self.S("v.n", N * D)
        var qkv = self.S("v.qkv", N * 3 * D)
        var att = self.S("v.att", N * D)
        var ff = self.S("v.ff", N * 4 * D)
        var sc = self.S("v.sc", H * N * N)
        self.lin(lat, "vae.post_quant.weight", "vae.post_quant.bias", h, N, TG_LATENT, D)
        for l in range(TG_VAE_LAYERS):
            var b = "vae.decoder.blocks." + String(l) + "."
            self.ln(h, b + "norm1", n, N, D, 1e-5)
            self.lin(n, b + "attn1.qkv", "", qkv, N, D, 3 * D)
            self.attention(qkv, 3 * D, 0, qkv.unsafe_offset(D), 3 * D, qkv.unsafe_offset(2 * D), 3 * D, 0, att, D, 0, 1, N, N, H, D // H, sc)
            self.lin(att, b + "attn1.to_out.0.weight", b + "attn1.to_out.0.bias", h, N, D, D, True)
            self.ln(h, b + "norm3", n, N, D, 1e-5)
            self.lin(n, b + "ff.net.0.proj.weight", b + "ff.net.0.proj.bias", ff, N, D, 4 * D)
            self.g.gelu(ff, N * 4 * D)
            self.lin(ff, b + "ff.net.2.weight", b + "ff.net.2.bias", h, N, 4 * D, D, True)
        var c = "vae.decoder.blocks." + String(TG_VAE_LAYERS) + "."
        self.ln(h, c + "attn2.norm_cross", n, N, D, 1e-5)
        var kv = self.S("v.kv", N * 2 * D)
        self.lin(n, c + "attn2.kv", "", kv, N, D, 2 * D)
        self.vae_ready = True

    def query_sdf(mut self, points: List[Float32]) raises -> List[Float32]:
        """SDF logits (positive inside) at points [n, 3]; `set_latents` first."""
        if not self.vae_ready:
            raise Error("triposg: set_latents before query_sdf")
        comptime D = TG_VAE_W
        comptime N = TG_TOKENS
        comptime H = TG_VAE_HEADS
        var total = len(points) // 3
        var out = List[Float32](length=total, fill=0.0)
        var Q = self.chunk
        var pts = self.S("q.pts", Q * 3)
        var emb = self.S("q.emb", Q * 51)
        var x = self.S("q.x", Q * D)
        var n = self.S("q.n", Q * D)
        var q = self.S("q.q", Q * D)
        var att = self.S("q.att", Q * D)
        var ff = self.S("q.ff", Q * 4 * D)
        var sc = self.S("q.sc", H * Q * N)
        var o1 = self.S("q.o", Q)
        var kv = self.S("v.kv", N * 2 * D)
        var c = "vae.decoder.blocks." + String(TG_VAE_LAYERS) + "."
        var start = 0
        while start < total:
            var m = min(Q, total - start)
            self.g.mm.upload(self.slots["q.pts"], Int(points.unsafe_ptr()) + start * 12, m * 3)
            self.g.freq_embed(pts, emb, m)
            self.lin(emb, "vae.decoder.proj_query.weight", "vae.decoder.proj_query.bias", x, m, 51, D)
            self.ln(x, c + "norm2", n, m, D, 1e-5)
            self.lin(n, c + "attn2.to_q.weight", "", q, m, D, D)
            self.attention(q, D, 0, kv, 2 * D, kv.unsafe_offset(D), 2 * D, 0, att, D, 0, 1, m, N, H, D // H, sc)
            self.lin(att, c + "attn2.to_out.0.weight", c + "attn2.to_out.0.bias", x, m, D, D, True)
            self.ln(x, c + "norm3", n, m, D, 1e-5)
            self.lin(n, c + "ff.net.0.proj.weight", c + "ff.net.0.proj.bias", ff, m, D, 4 * D)
            self.g.gelu(ff, m * 4 * D)
            self.lin(ff, c + "ff.net.2.weight", c + "ff.net.2.bias", x, m, 4 * D, D, True)
            self.ln(x, "vae.decoder.norm_out", n, m, D, 1e-5)
            self.lin(n, "vae.decoder.proj_out.weight", "vae.decoder.proj_out.bias", o1, m, D, 1)
            self.g.mm.download(self.slots["q.o"], Int(out.unsafe_ptr()) + start * 4, m)
            start += m
        for i in range(total):
            out[i] = -out[i]
        return out^


# ---------------------------------------------------------------------------------------------
# SDF grid extraction

def grid_points(res: Int, bound: Float32 = TG_BOUND) -> List[Float32]:
    """(res + 1)^3 points of linspace(-bound, bound, res + 1)^3 in ij order (x slowest)."""
    var n = res + 1
    var ax = grid_axis(res, bound)
    var p = List[Float32](length=n * n * n * 3, fill=0.0)
    var k = 0
    for i in range(n):
        for j in range(n):
            for l in range(n):
                p[k] = ax[i]
                p[k + 1] = ax[j]
                p[k + 2] = ax[l]
                k += 3
    return p^


def grid_axis(res: Int, bound: Float32 = TG_BOUND) -> List[Float32]:
    """numpy linspace(-bound, bound, res + 1) in float32 (step computed in float64)."""
    var ax = List[Float32](length=res + 1, fill=0.0)
    var lo = Float64(-bound)
    var step = (Float64(bound) - lo) / Float64(res)
    for i in range(res + 1):
        ax[i] = Float32(lo + Float64(i) * step)
    ax[res] = bound
    return ax^


struct SdfGrid(Movable):
    var res: Int          # cells per axis; values has (res + 1)^3 entries, ij order
    var values: List[Float32]
    var queried: Int

    def __init__(out self, res: Int):
        self.res = res
        var n = res + 1
        self.values = List[Float32](length=n * n * n, fill=0.0)
        self.queried = 0


def extract_sdf(mut model: TripoSG, res: Int = 512, base: Int = 64, band: Float32 = 0.95) raises -> SdfGrid:
    """Coarse-to-fine SDF grid: dense at `base`, then each doubling evaluates only the points of
    cells next to the surface (a sign change or |logit| < band, dilated by one cell) and fills
    the rest by trilinear interpolation (their sign cannot change). `set_latents` first."""
    var r = min(base, res)
    var g = SdfGrid(r)
    g.values = model.query_sdf(grid_points(r))
    g.queried = len(g.values)
    while r < res:
        var nr = r * 2
        var n0 = r + 1
        var n1 = nr + 1
        # active coarse cells
        var act = List[Bool](length=r * r * r, fill=False)
        for i in range(r):
            for j in range(r):
                for k in range(r):
                    var pos = False
                    var neg = False
                    var near = False
                    for c in range(8):
                        var v = g.values[((i + (c & 1)) * n0 + j + ((c >> 1) & 1)) * n0 + k + ((c >> 2) & 1)]
                        if v > 0:
                            pos = True
                        else:
                            neg = True
                        if abs(v) < band:
                            near = True
                    if (pos and neg) or near:
                        act[(i * r + j) * r + k] = True
        var dil = List[Bool](length=r * r * r, fill=False)
        for i in range(r):
            for j in range(r):
                for k in range(r):
                    if not act[(i * r + j) * r + k]:
                        continue
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            for dk in range(-1, 2):
                                var a = i + di
                                var b = j + dj
                                var c = k + dk
                                if a >= 0 and b >= 0 and c >= 0 and a < r and b < r and c < r:
                                    dil[(a * r + b) * r + c] = True
        # fine grid: copy coarse points, trilinear elsewhere, mark points to query
        var fine = List[Float32](length=n1 * n1 * n1, fill=0.0)
        var want = List[Bool](length=n1 * n1 * n1, fill=False)
        for i in range(r):
            for j in range(r):
                for k in range(r):
                    if dil[(i * r + j) * r + k]:
                        for a in range(3):
                            for b in range(3):
                                for c in range(3):
                                    want[((2 * i + a) * n1 + 2 * j + b) * n1 + 2 * k + c] = True
        var qpts = List[Float32]()
        var qidx = List[Int]()
        var ax = grid_axis(nr)
        for i in range(n1):
            for j in range(n1):
                for k in range(n1):
                    var fi = (i * n1 + j) * n1 + k
                    if i % 2 == 0 and j % 2 == 0 and k % 2 == 0:
                        fine[fi] = g.values[((i // 2) * n0 + j // 2) * n0 + k // 2]
                        continue
                    if want[fi]:
                        qidx.append(fi)
                        qpts.append(ax[i])
                        qpts.append(ax[j])
                        qpts.append(ax[k])
                        continue
                    var i0 = i // 2
                    var j0 = j // 2
                    var k0 = k // 2
                    var i1 = min(i0 + (i % 2), r)
                    var j1 = min(j0 + (j % 2), r)
                    var k1 = min(k0 + (k % 2), r)
                    var s: Float32 = 0
                    for c in range(8):
                        var ii = i1 if (c & 1) != 0 else i0
                        var jj = j1 if (c & 2) != 0 else j0
                        var kk = k1 if (c & 4) != 0 else k0
                        s += g.values[(ii * n0 + jj) * n0 + kk]
                    fine[fi] = s * 0.125
        var vals = model.query_sdf(qpts)
        for q in range(len(qidx)):
            fine[qidx[q]] = vals[q]
        g.queried += len(qidx)
        g.res = nr
        g.values = fine^
        r = nr
    return g^


# ---------------------------------------------------------------------------------------------
# raw float files (parity dumps)

def _two(i: Int) -> String:
    return ("0" + String(i)) if i < 10 else String(i)


def read_f32(path: String) raises -> List[Float32]:
    var b = read_bytes(path)
    if len(b) == 0 or len(b) % 4 != 0:
        raise Error("triposg: cannot read float32 file " + path)
    var n = len(b) // 4
    var out = List[Float32](length=n, fill=0.0)
    _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), Int(b.unsafe_ptr()), n * 4)
    return out^


def write_f32(path: String, data: List[Float32]) raises:
    from flint.platform.os import write_bytes
    var b = List[UInt8](length=len(data) * 4, fill=0)
    _ = external_call["memcpy", Int](Int(b.unsafe_ptr()), Int(data.unsafe_ptr()), len(data) * 4)
    if not write_bytes(path, b):
        raise Error("triposg: cannot write " + path)
