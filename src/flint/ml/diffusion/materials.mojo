"""Text / photo -> PBR material sets on `flint.ml.diffusion` (`flint.ml.diffusion.materials`).

Two latent-diffusion material models, run from their Hugging Face safetensors:

- **StableMaterials** (`gvecchio/StableMaterials`, CreativeML OpenRAIL-M, see
  `licenses/stablematerials-LICENSE.txt`): a text prompt (CLIP-L text, projected `text_embeds`) or a photo
  (CLIP-L vision `image_embeds`) as a single cross-attention token; the LCM UNet (4 steps,
  guidance embedding w = guidance - 1, no CFG) or the base UNet (DDIM + CFG against the embedding of
  a black image); an 18-channel latent decoded by a KL VAE into basecolor (3), normal xy (2),
  height, roughness and metallic. Tileable output switches every convolution to circular padding
  from step n / 4 on (the pipeline's `rolled_conv`) and decodes a circularly padded latent.
- **MatFuse** (`gvecchio/MatFuse`, MIT, see `licenses/matfuse-LICENSE.txt`): three cross-attention tokens
  (CLIP ViT-B/16 image embedding, sentence-transformers CLIP text embedding, colour-palette MLP) plus
  a sketch CNN concatenated to the 12-channel latent; 20 DDIM steps with CFG 4; a VQ decoder
  (no quantisation at decode) into diffuse, normal, roughness and specular. MatFuse has no height
  or metallic map: `PbrMaps.from_matfuse` derives metallic from the specular brightness (MatFuse
  dielectrics carry a flat ~0.3 sRGB specular, metals a bright tinted one), blends basecolor
  between diffuse and specular by it, and integrates height from the normal map (periodic Poisson).

`write_material` saves the maps as PNGs, a packed glTF ORM texture (R = 1 occlusion, G roughness,
B metallic), a `<name>.glb` textured quad (baseColor / normal / metallicRoughness textures, loadable
by Flint's GLB loader and any glTF viewer) and a `<name>.material.json` descriptor.
"""

from std.math import sqrt, cos, sin, pi
from flint.platform.os import read_bytes, write_bytes, make_dirs
from flint.assets.image import write_png
from flint.assets.model import NativeModel, NativeMesh, NativeTexture
from flint.assets.glb_writer import write_native_glb
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.safetensors import SafeTensors
from flint.ml.diffusion.nn import Runtime, Act, Conv
from flint.ml.diffusion.kernels import ACT_GELU
from flint.ml.diffusion.unet import UNet
from flint.ml.diffusion.vae import VaeDecoder, decode_padded
from flint.ml.diffusion.clip import ClipText, ClipVision
from flint.ml.diffusion.tokenizer import ClipTokenizer
from flint.ml.diffusion.schedulers import Scheduler, SCHED_DDIM, SCHED_LCM, guidance_embedding
from flint.ml.diffusion.rng import TorchGenerator
from flint.ml.diffusion.pipeline import Denoiser, nchw_to_nhwc
from flint.ml.diffusion.imageproc import (
    resize_bicubic_aa, clip_preprocess, rgb_to_pixels, gray_to_pixels,
)


# ---------------------------------------------------------------------------------------------
# material maps


@always_inline
def _clamp01(v: Float32) -> Float32:
    return min(max(v, Float32(0.0)), Float32(1.0))


@always_inline
def _to_linear(c: Float32) -> Float32:
    if c <= 0.04045:
        return c / 12.92
    return Float32(((Float64(c) + 0.055) / 1.055) ** 2.4)


@always_inline
def _to_srgb(c: Float32) -> Float32:
    if c <= 0.0031308:
        return c * 12.92
    return Float32(1.055 * Float64(c) ** (1.0 / 2.4) - 0.055)


struct PbrMaps(Copyable, Movable):
    """A square material tile; all maps channels-last in [0, 1]. `normal` is the usual
    0.5 + 0.5 * n encoding (OpenGL / glTF convention, +Y up)."""
    var size: Int
    var basecolor: List[Float32]   # [n, 3]
    var normal: List[Float32]      # [n, 3]
    var height: List[Float32]      # [n]
    var roughness: List[Float32]   # [n]
    var metallic: List[Float32]    # [n]
    var raw: List[Float32]         # decoder output, NHWC [size, size, raw_c], [-1, 1]
    var raw_c: Int

    def __init__(out self):
        self.size = 0
        self.basecolor = List[Float32]()
        self.normal = List[Float32]()
        self.height = List[Float32]()
        self.roughness = List[Float32]()
        self.metallic = List[Float32]()
        self.raw = List[Float32]()
        self.raw_c = 0

    @staticmethod
    def from_stablematerials(var raw: List[Float32], size: Int) -> PbrMaps:
        """9-channel StableMaterials decode -> maps (`MaterialImage.from_tensor`)."""
        var m = PbrMaps()
        var n = size * size
        m.size = size
        m.raw_c = 9
        m.basecolor = List[Float32](length=n * 3, fill=0.0)
        m.normal = List[Float32](length=n * 3, fill=0.0)
        m.height = List[Float32](length=n, fill=0.0)
        m.roughness = List[Float32](length=n, fill=0.0)
        m.metallic = List[Float32](length=n, fill=0.0)
        for p in range(n):
            var o = p * 9
            for k in range(3):
                m.basecolor[p * 3 + k] = _clamp01(raw[o + k] * 0.5 + 0.5)
            # normal xy: (v * 0.5 + 0.5) * 2 - 1, z = sqrt(1 - x^2 - y^2) (rescaled when outside the disc)
            var nx = (raw[o + 3] * 0.5 + 0.5) * 2.0 - 1.0
            var ny = (raw[o + 4] * 0.5 + 0.5) * 2.0 - 1.0
            var s = nx * nx + ny * ny
            var nz: Float32 = 0.0
            if s <= 1.0:
                nz = sqrt(1.0 - s)
            else:
                var r = sqrt(s)
                nx /= r
                ny /= r
            m.normal[p * 3] = _clamp01(nx * 0.5 + 0.5)
            m.normal[p * 3 + 1] = _clamp01(ny * 0.5 + 0.5)
            m.normal[p * 3 + 2] = _clamp01(nz * 0.5 + 0.5)
            m.height[p] = _clamp01(raw[o + 5] * 0.5 + 0.5)
            m.roughness[p] = _clamp01(raw[o + 6] * 0.5 + 0.5)
            m.metallic[p] = _clamp01(raw[o + 7] * 0.5 + 0.5)
        m.raw = raw^
        return m^

    @staticmethod
    def from_matfuse(var raw: List[Float32], size: Int) raises -> PbrMaps:
        """12-channel MatFuse decode (diffuse, normal, roughness, specular) -> metallic-roughness maps."""
        var m = PbrMaps()
        var n = size * size
        m.size = size
        m.raw_c = 12
        m.basecolor = List[Float32](length=n * 3, fill=0.0)
        m.normal = List[Float32](length=n * 3, fill=0.0)
        m.roughness = List[Float32](length=n, fill=0.0)
        m.metallic = List[Float32](length=n, fill=0.0)
        for p in range(n):
            var o = p * 12
            var d = List[Float32](length=3, fill=0.0)
            var sp = List[Float32](length=3, fill=0.0)
            var nv = List[Float32](length=3, fill=0.0)
            var r: Float32 = 0
            for k in range(3):
                d[k] = _to_linear(_clamp01((raw[o + k] + 1.0) * 0.5))
                nv[k] = _clamp01((raw[o + 3 + k] + 1.0) * 0.5) * 2.0 - 1.0
                r += _clamp01((raw[o + 6 + k] + 1.0) * 0.5)
                sp[k] = _to_linear(_clamp01((raw[o + 9 + k] + 1.0) * 0.5))
            m.roughness[p] = r / 3.0
            var ln = sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2])
            if ln < 1e-6:
                nv[0] = 0
                nv[1] = 0
                nv[2] = 1
                ln = 1
            for k in range(3):
                m.normal[p * 3 + k] = nv[k] / ln * 0.5 + 0.5
            # Metallic from the specular map. MatFuse's training data gives dielectrics a flat grey
            # specular of ~0.3 sRGB (F0 ~0.07 linear), so the Khronos spec-gloss solver (metal above
            # F0 0.04) would turn brick into metal. Instead metallic ramps (smoothstep) with the
            # perceived linear specular brightness from 0.10 to 0.45, and the base colour blends from
            # the diffuse colour (dielectric) to the specular colour (metal F0).
            var ps = sqrt(0.299 * sp[0] * sp[0] + 0.587 * sp[1] * sp[1] + 0.114 * sp[2] * sp[2])
            var t = _clamp01((ps - 0.10) / 0.35)
            var met = t * t * (3.0 - 2.0 * t)
            m.metallic[p] = met
            for k in range(3):
                m.basecolor[p * 3 + k] = _to_srgb(_clamp01(d[k] * (1.0 - met) + sp[k] * met))
        m.height = height_from_normal(m.normal, size)
        m.raw = raw^
        return m^


def _dft_rows(mut re: List[Float64], mut im: List[Float64], n: Int, inverse: Bool, transpose: Bool):
    """In-place 1-D DFT of every row (or column) of an n x n complex array (direct O(n^2) per line)."""
    var cs = List[Float64](length=n, fill=0.0)
    var sn = List[Float64](length=n, fill=0.0)
    var sign: Float64 = 1.0 if inverse else -1.0
    for k in range(n):
        var a = 2.0 * pi * Float64(k) / Float64(n)
        cs[k] = cos(a)
        sn[k] = sign * sin(a)
    var lr = List[Float64](length=n, fill=0.0)
    var li = List[Float64](length=n, fill=0.0)
    for line in range(n):
        for j in range(n):
            var idx = j * n + line if transpose else line * n + j
            lr[j] = re[idx]
            li[j] = im[idx]
        for k in range(n):
            var sr: Float64 = 0
            var si: Float64 = 0
            var step = 0
            for j in range(n):
                sr += lr[j] * cs[step] - li[j] * sn[step]
                si += lr[j] * sn[step] + li[j] * cs[step]
                step += k
                if step >= n:
                    step -= n
            var idx = k * n + line if transpose else line * n + k
            re[idx] = sr
            im[idx] = si


def height_from_normal(normal01: List[Float32], size: Int) raises -> List[Float32]:
    """Periodic least-squares height field whose slopes match a tangent-space normal map
    (discrete Poisson solve in the Fourier domain), normalised to [0, 1]."""
    var n = size
    var gx = List[Float64](length=n * n, fill=0.0)
    var gy = List[Float64](length=n * n, fill=0.0)
    for p in range(n * n):
        var nx = Float64(normal01[p * 3]) * 2.0 - 1.0
        var ny = Float64(normal01[p * 3 + 1]) * 2.0 - 1.0
        var nz = max(Float64(normal01[p * 3 + 2]) * 2.0 - 1.0, 0.2)
        gx[p] = -nx / nz          # dh / dx (x right)
        gy[p] = ny / nz           # dh / drow (+Y up = towards row 0)
    var re = List[Float64](length=n * n, fill=0.0)
    var im = List[Float64](length=n * n, fill=0.0)
    for r in range(n):
        for c in range(n):
            var cl = (c - 1 + n) % n
            var ru = (r - 1 + n) % n
            # divergence of the slopes averaged onto the half-pixel edges
            var ex = 0.5 * (gx[r * n + c] + gx[r * n + (c + 1) % n]) - 0.5 * (gx[r * n + cl] + gx[r * n + c])
            var ey = 0.5 * (gy[r * n + c] + gy[((r + 1) % n) * n + c]) - 0.5 * (gy[ru * n + c] + gy[r * n + c])
            re[r * n + c] = ex + ey
    _dft_rows(re, im, n, False, False)
    _dft_rows(re, im, n, False, True)
    for v in range(n):
        for u in range(n):
            var lam = 2.0 * cos(2.0 * pi * Float64(u) / Float64(n)) + 2.0 * cos(2.0 * pi * Float64(v) / Float64(n)) - 4.0
            var i = v * n + u
            if u == 0 and v == 0:
                re[i] = 0
                im[i] = 0
            else:
                re[i] /= lam
                im[i] /= lam
    _dft_rows(re, im, n, True, False)
    _dft_rows(re, im, n, True, True)
    var lo: Float64 = 1e300
    var hi: Float64 = -1e300
    for i in range(n * n):
        lo = min(lo, re[i])
        hi = max(hi, re[i])
    var out = List[Float32](length=n * n, fill=0.5)
    if hi - lo > 1e-12:
        for i in range(n * n):
            out[i] = Float32((re[i] - lo) / (hi - lo))
    return out^


def _quad_model(size: Int, var base: List[UInt32], var normal: List[UInt32], var orm: List[UInt32],
                tiles: Int) -> NativeModel:
    var model = NativeModel()
    for k in range(3):
        var t = NativeTexture()
        t.width = size
        t.height = size
        if k == 0:
            t.pixels = base.copy()
        elif k == 1:
            t.pixels = normal.copy()
        else:
            t.pixels = orm.copy()
        model.textures.append(t^)
    var mesh = NativeMesh()
    var uv = Float32(tiles)
    var xs: List[Float32] = [-0.5, 0.5, 0.5, -0.5]
    var zs: List[Float32] = [0.5, 0.5, -0.5, -0.5]
    var us: List[Float32] = [0.0, uv, uv, 0.0]
    var vs: List[Float32] = [uv, uv, 0.0, 0.0]
    for i in range(4):
        mesh.positions.append(xs[i] * Float32(tiles))
        mesh.positions.append(0.0)
        mesh.positions.append(zs[i] * Float32(tiles))
        mesh.normals.append(0.0)
        mesh.normals.append(1.0)
        mesh.normals.append(0.0)
        mesh.texcoords.append(us[i])
        mesh.texcoords.append(vs[i])
    var idx: List[Int] = [0, 1, 2, 0, 2, 3]
    mesh.indices = idx^
    mesh.texture = 0
    mesh.normal_orm_texture = 1
    mesh.orm_texture = 2
    mesh.base_color = 0xFFFFFFFF
    mesh.roughness = 1.0
    mesh.metallic = 1.0
    model.meshes.append(mesh^)
    return model^


def write_material(m: PbrMaps, dir: String, name: String, model_name: String, prompt: String,
                   tiles: Int = 1) raises -> String:
    """Write `<dir>/<name>_{basecolor,normal,height,roughness,metallic,orm}.png`, `<name>.glb` and
    `<name>.material.json`; returns the GLB path."""
    _ = make_dirs(dir)
    var n = m.size * m.size
    var base = dir + "/" + name
    var bc = rgb_to_pixels(m.basecolor, n)
    var nm = rgb_to_pixels(m.normal, n)
    var hp = gray_to_pixels(m.height, n)
    var rp = gray_to_pixels(m.roughness, n)
    var mp = gray_to_pixels(m.metallic, n)
    var orm = List[UInt32](length=n, fill=0)
    for i in range(n):
        orm[i] = UInt32(0xFFFF0000) | (rp[i] & 0xFF00) | (mp[i] & 0xFF)
    if not (write_png(base + "_basecolor.png", bc, m.size, m.size) and write_png(base + "_normal.png", nm, m.size, m.size)
            and write_png(base + "_height.png", hp, m.size, m.size)
            and write_png(base + "_roughness.png", rp, m.size, m.size)
            and write_png(base + "_metallic.png", mp, m.size, m.size)
            and write_png(base + "_orm.png", orm, m.size, m.size)):
        raise Error("materials: cannot write PNGs under " + dir)
    var model = _quad_model(m.size, bc^, nm^, orm^, tiles)
    if not write_native_glb(base + ".glb", model):
        raise Error("materials: cannot write " + base + ".glb")
    var esc = String()
    for ch in prompt.codepoint_slices():
        if ch == '"' or ch == "\\":
            esc += "\\"
        esc += ch
    var js = String('{\n  "flint_material": 1,\n  "generator": "flint.ml.diffusion.materials / ') + model_name + '",\n'
    js += '  "prompt": "' + esc + '",\n  "size": ' + String(m.size) + ',\n  "tileable": true,\n'
    js += '  "workflow": "metallic_roughness",\n  "normal_convention": "opengl_y_up",\n'
    js += '  "maps": {"basecolor": "' + name + '_basecolor.png", "normal": "' + name + '_normal.png", "height": "'
    js += name + '_height.png", "roughness": "' + name + '_roughness.png", "metallic": "' + name + '_metallic.png", "orm": "'
    js += name + '_orm.png"},\n  "glb": "' + name + '.glb"\n}\n'
    var jb = List[UInt8](js.as_bytes())
    _ = write_bytes(base + ".material.json", jb)
    return base + ".glb"


# ---------------------------------------------------------------------------------------------
# StableMaterials


struct StableMaterials(Movable):
    var tok: ClipTokenizer
    var text: ClipText
    var vision: ClipVision
    var den: Denoiser
    var vae: VaeDecoder
    var lcm: Bool
    var black: List[Float32]        # vision embedding of a black image (CFG negative)

    def __init__(out self, var tok: ClipTokenizer, var text: ClipText, var vision: ClipVision, var den: Denoiser,
                 var vae: VaeDecoder, lcm: Bool):
        self.tok = tok^
        self.text = text^
        self.vision = vision^
        self.den = den^
        self.vae = vae^
        self.lcm = lcm
        self.black = List[Float32]()

    @staticmethod
    def load(mut rt: Runtime, dir: String, lcm: Bool = True) raises -> StableMaterials:
        """`dir`: the StableMaterials repo layout (tokenizer/, text_encoder/, vision_encoder/, vae/,
        unet_lcm/ or unet/, scheduler/)."""
        var tok = ClipTokenizer.load(dir + "/tokenizer")
        var text = ClipText.load(rt, dir + "/text_encoder/model.safetensors", dir + "/text_encoder/config.json")
        var vision = ClipVision.load(rt, dir + "/vision_encoder/model.safetensors", dir + "/vision_encoder/config.json")
        var unet = UNet.load(rt, dir + ("/unet_lcm" if lcm else "/unet"))
        var sched = Scheduler.from_config(read_bytes(dir + "/scheduler/scheduler_config.json"),
                                          SCHED_LCM if lcm else SCHED_DDIM)
        var vae = VaeDecoder.load(rt, dir + "/vae")
        var sm = StableMaterials(tok^, text^, vision^, Denoiser(unet^, sched^), vae^, lcm)
        return sm^

    def text_condition(mut self, mut rt: Runtime, prompt: String) raises -> List[Float32]:
        var ids = self.tok.encode(prompt)
        return self.text.encode(rt, ids).pooled.copy()

    def image_condition(mut self, mut rt: Runtime, rgb01: List[Float32], h: Int, w: Int) raises -> List[Float32]:
        var px = self.vision.normalize(clip_preprocess(rgb01, h, w, self.vision.image_size))
        return self.vision.encode_pixels(rt, px).pooled.copy()

    def latent(mut self, mut rt: Runtime, var cond: List[Float32], steps: Int, seed: Int, guidance: Float32,
               tile: Bool, size: Int = 512) raises -> List[Float32]:
        """The final latent (NCHW [1, 18, size/8, size/8])."""
        var D = len(cond)
        if self.lcm:
            self.den.tcond = guidance_embedding(guidance - 1.0, self.den.unet.cond_dim)
            self.den.set_context(cond^, List[Float32](), 1, D, 1.0)
        else:
            if len(self.black) == 0:
                var S = self.vision.image_size
                self.black = self.vision.encode_pixels(rt, self.vision.normalize(List[Float32](length=S * S * 3, fill=0.0))).pooled.copy()
            self.den.set_context(cond^, self.black.copy(), 1, D, guidance)
        self.den.sched.set_timesteps(steps)
        self.den.tile_from = self.den.sched.num_steps() // 4 if tile else -1
        var gen = TorchGenerator(seed)
        var l = size // 8
        return self.den.sample(rt, gen, self.den.unet.in_channels, l, l, steps)

    def decode(mut self, mut rt: Runtime, lat_nchw: List[Float32], tile: Bool, size: Int = 512) raises -> PbrMaps:
        var l = size // 8
        var c = self.vae.latent_channels
        var z = nchw_to_nhwc(lat_nchw, 1, c, l, l)
        for i in range(len(z)):
            z[i] /= self.vae.scaling_factor
        var raw: List[Float32]
        if tile:
            raw = decode_padded(rt, self.vae, z, 1, l, l, l // 8)
        else:
            var za = rt.upload(1, l, l, c, z)
            var img = self.vae.decode(rt, za, False)
            rt.free(za^)
            raw = rt.download(img)
            rt.free(img^)
        return PbrMaps.from_stablematerials(raw^, size)

    def generate(mut self, mut rt: Runtime, var cond: List[Float32], steps: Int = 4, seed: Int = 42,
                 guidance: Float32 = 10.0, tile: Bool = True, size: Int = 512) raises -> PbrMaps:
        var lat = self.latent(rt, cond^, steps, seed, guidance, tile, size)
        return self.decode(rt, lat, tile, size)


# ---------------------------------------------------------------------------------------------
# MatFuse


def _bn_fold_conv(mut rt: Runtime, st: SafeTensors, conv: String, bn: String, stride: Int, pad: Int) raises -> Conv:
    """Conv2d followed by an eval-mode BatchNorm2d (eps 1e-5), folded into one convolution."""
    var w = st.weight(conv + ".weight")
    var b = st.weight(conv + ".bias").owned_floats()
    var g = st.weight(bn + ".weight").owned_floats()
    var be = st.weight(bn + ".bias").owned_floats()
    var mu = st.weight(bn + ".running_mean").owned_floats()
    var var_ = st.weight(bn + ".running_var").owned_floats()
    var co = w.shape[0]
    var per = w.numel() // co
    var wf = w.owned_floats()
    var bf = List[Float32](length=co, fill=0.0)
    for o in range(co):
        var s = g[o] / sqrt(var_[o] + 1e-5)
        for i in range(per):
            wf[o * per + i] *= s
        bf[o] = (b[o] - mu[o]) * s + be[o]
    var wt = Tensor.floats(w.shape.copy(), wf^)
    var bt = Tensor.floats([co], bf^)
    return rt.conv_from(wt, bt, True, stride, pad, pad, pad, pad)


@always_inline
def _gelu(x: Float32) -> Float32:
    from std.math import erf
    return 0.5 * x * (1.0 + erf(x * 0.7071067811865476))


struct MatFuseConditions(Movable):
    var crossattn: List[Float32]    # [3, 512]: image, text, palette tokens
    var concat: List[Float32]       # NHWC [h/8, w/8, 4] sketch features

    def __init__(out self):
        self.crossattn = List[Float32]()
        self.concat = List[Float32]()


struct MatFuse(Movable):
    var tok: ClipTokenizer
    var text: ClipText
    var vision: ClipVision
    var den: Denoiser
    var vae: VaeDecoder
    var sketch: List[Conv]
    var p1w: List[Float32]   # palette Linear(3, 64): [64, 3]
    var p1b: List[Float32]
    var p2w: List[Float32]   # Linear(320, 512): [512, 320]
    var p2b: List[Float32]
    var n_colors: Int

    def __init__(out self, var tok: ClipTokenizer, var text: ClipText, var vision: ClipVision, var den: Denoiser,
                 var vae: VaeDecoder):
        self.tok = tok^
        self.text = text^
        self.vision = vision^
        self.den = den^
        self.vae = vae^
        self.sketch = List[Conv]()
        self.p1w = List[Float32]()
        self.p1b = List[Float32]()
        self.p2w = List[Float32]()
        self.p2b = List[Float32]()
        self.n_colors = 5

    @staticmethod
    def load(mut rt: Runtime, dir: String) raises -> MatFuse:
        """`dir`: the MatFuse repo layout (unet/, vae/, condition_encoder/, scheduler/) plus `clip/`
        = `sentence-transformers/clip-ViT-B-16` `0_CLIPModel` (CLIP ViT-B/16, both towers)."""
        var tok = ClipTokenizer.load(dir + "/clip")
        var text = ClipText.load(rt, dir + "/clip/model.safetensors", dir + "/clip/config.json")
        var vision = ClipVision.load(rt, dir + "/clip/model.safetensors", dir + "/clip/config.json")
        var unet = UNet.load(rt, dir + "/unet")
        var sched = Scheduler.from_config(read_bytes(dir + "/scheduler/scheduler_config.json"), SCHED_DDIM)
        var vae = VaeDecoder.load(rt, dir + "/vae")
        var mf = MatFuse(tok^, text^, vision^, Denoiser(unet^, sched^), vae^)
        var st = SafeTensors.open(dir + "/condition_encoder/diffusion_pytorch_model.safetensors")
        var p = "sketch_encoder.net."
        mf.sketch.append(_bn_fold_conv(rt, st, p + "0", p + "1", 1, 1))
        mf.sketch.append(_bn_fold_conv(rt, st, p + "3", p + "4", 2, 1))
        mf.sketch.append(_bn_fold_conv(rt, st, p + "6", p + "7", 2, 1))
        mf.sketch.append(_bn_fold_conv(rt, st, p + "9", p + "10", 2, 1))
        mf.sketch.append(_bn_fold_conv(rt, st, p + "12", p + "13", 1, 0))
        mf.p1w = st.weight("palette_encoder.net.0.weight").owned_floats()
        mf.p1b = st.weight("palette_encoder.net.0.bias").owned_floats()
        mf.p2w = st.weight("palette_encoder.net.3.weight").owned_floats()
        mf.p2b = st.weight("palette_encoder.net.3.bias").owned_floats()
        mf.n_colors = st.shape("palette_encoder.net.3.weight")[1] // st.shape("palette_encoder.net.0.weight")[0]
        return mf^

    def palette_embedding(self, palette01: List[Float32]) -> List[Float32]:
        """[n_colors, 3] RGB in [0, 1] (empty = the all-zero placeholder) -> [512]."""
        var hdim = len(self.p1b)
        var od = len(self.p2b)
        var hid = List[Float32](length=self.n_colors * hdim, fill=0.0)
        for c in range(self.n_colors):
            for j in range(hdim):
                var acc = self.p1b[j]
                if len(palette01) >= self.n_colors * 3:
                    for k in range(3):
                        acc += self.p1w[j * 3 + k] * palette01[c * 3 + k]
                hid[c * hdim + j] = _gelu(acc)
        var out = List[Float32](length=od, fill=0.0)
        var K = self.n_colors * hdim
        for o in range(od):
            var acc = Float64(self.p2b[o])
            for k in range(K):
                acc += Float64(self.p2w[o * K + k]) * Float64(hid[k])
            out[o] = _gelu(Float32(acc))
        return out^

    def image_embedding(mut self, mut rt: Runtime, rgb01: List[Float32], h: Int, w: Int) raises -> List[Float32]:
        """CLIP ViT-B/16 embedding (L2-normalised) of a photo; an empty image = the grey placeholder.
        Resize: `F.interpolate(bicubic, antialias=True, align_corners=True)` to 224 x 224."""
        var S = self.vision.image_size
        var px: List[Float32]
        if len(rgb01) == 0:
            px = List[Float32](length=S * S * 3, fill=0.5)
        else:
            px = resize_bicubic_aa(rgb01, h, w, 3, S, S, True)
        var e = self.vision.encode_pixels(rt, self.vision.normalize(px)).pooled.copy()
        var s: Float64 = 0
        for v in e:
            s += Float64(v) * Float64(v)
        var inv = Float32(1.0 / sqrt(s))
        for i in range(len(e)):
            e[i] *= inv
        return e^

    def text_embedding(mut self, mut rt: Runtime, text: String) raises -> List[Float32]:
        var ids = self.tok.encode(text)
        return self.text.encode(rt, ids).pooled.copy()

    def sketch_features(mut self, mut rt: Runtime, sketch01: List[Float32], size: Int) raises -> List[Float32]:
        """Sketch CNN on a [size, size] edge map in [0, 1] (empty = zeros) -> NHWC [size/8, size/8, 4]."""
        var s = sketch01.copy()
        if len(s) == 0:
            s = List[Float32](length=size * size, fill=0.0)
        var x = rt.upload(1, size, size, 1, s)
        for i in range(len(self.sketch)):
            var y = rt.conv2d(x, self.sketch[i])
            rt.free(x^)
            rt.activate(y, ACT_GELU)
            x = y^
        var out = rt.download(x)
        rt.free(x^)
        return out^

    def conditions(mut self, mut rt: Runtime, text: String, rgb01: List[Float32], h: Int, w: Int,
                   palette01: List[Float32], sketch01: List[Float32], size: Int = 256) raises -> MatFuseConditions:
        """Every condition is optional (empty string / list = the training-time placeholder)."""
        var c = MatFuseConditions()
        c.crossattn = self.image_embedding(rt, rgb01, h, w)
        for v in self.text_embedding(rt, text):
            c.crossattn.append(v)
        for v in self.palette_embedding(palette01):
            c.crossattn.append(v)
        c.concat = self.sketch_features(rt, sketch01, size)
        return c^

    def latent(mut self, mut rt: Runtime, cond: MatFuseConditions, uncond: MatFuseConditions, steps: Int,
               seed: Int, guidance: Float32, tile: Bool, size: Int = 256) raises -> List[Float32]:
        var D = len(cond.crossattn) // 3
        self.den.set_context(cond.crossattn.copy(), uncond.crossattn.copy(), 3, D, guidance)
        self.den.set_concat(cond.concat.copy(), uncond.concat.copy(), 4)
        self.den.sched.set_timesteps(steps)
        self.den.tile_from = self.den.sched.num_steps() // 4 if tile else -1
        var gen = TorchGenerator(seed)
        var l = size // 8
        return self.den.sample(rt, gen, self.vae.latent_channels, l, l, steps)

    def decode(mut self, mut rt: Runtime, lat_nchw: List[Float32], size: Int = 256) raises -> PbrMaps:
        """`decode_latents`: circular pad 7, VQ decode, centre crop."""
        var l = size // 8
        var z = nchw_to_nhwc(lat_nchw, 1, self.vae.latent_channels, l, l)
        var raw = decode_padded(rt, self.vae, z, 1, l, l, 7)
        return PbrMaps.from_matfuse(raw^, size)

    def generate(mut self, mut rt: Runtime, text: String, rgb01: List[Float32], h: Int, w: Int,
                 palette01: List[Float32], sketch01: List[Float32], steps: Int = 20, seed: Int = 7,
                 guidance: Float32 = 4.0, tile: Bool = False, size: Int = 256) raises -> PbrMaps:
        var cond = self.conditions(rt, text, rgb01, h, w, palette01, sketch01, size)
        var uncond = self.conditions(rt, "", List[Float32](), 0, 0, List[Float32](), List[Float32](), size)
        var lat = self.latent(rt, cond, uncond, steps, seed, guidance, tile, size)
        return self.decode(rt, lat, size)
