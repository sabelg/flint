"""Latent autoencoders for `flint.ml.diffusion`.

`VaeDecoder` loads the decoder of a diffusers `AutoencoderKL` (`decoder.up_blocks.*`,
`post_quant_conv`) or of a latent-diffusion (LDM / taming) VQ / KL model (`decoder.up.*`, e.g.
MatFuse's `MatFuseVQModel`): conv_in -> mid (ResNet, single-head self-attention, ResNet) ->
upsampling stages -> GroupNorm / SiLU / conv_out. `VaeEncoder` is the KL encoder (with
`quant_conv`) and returns the posterior mean.

Both take channels-last latents / images on a `Runtime`. `tile = True` wraps every convolution
(seamless decoding); `decode_padded` instead reproduces the pipelines that circularly pad the
latent, decode with ordinary zero padding and crop the centre (StableMaterials, MatFuse).
"""

from flint.platform.os import read_bytes
from flint.assets.json import parse_json
from flint.ml.safetensors import SafeTensors
from flint.ml.diffusion.nn import Runtime, Act, Linear, Conv, Norm, PAD_ZERO, PAD_WRAP
from flint.ml.diffusion.unet import ResBlock, load_resblock, resblock_forward


struct MidAttn(Copyable, Movable):
    var norm: Norm
    var q: Linear
    var k: Linear
    var v: Linear
    var o: Linear

    def __init__(out self):
        self.norm = Norm()
        self.q = Linear()
        self.k = Linear()
        self.v = Linear()
        self.o = Linear()


def _load_mid_attn(mut rt: Runtime, st: SafeTensors, p: String) raises -> MidAttn:
    var a = MidAttn()
    if st.has(p + ".group_norm.weight"):        # diffusers
        a.norm = rt.norm(st, p + ".group_norm", 32, 1e-6)
        a.q = rt.linear(st, p + ".to_q")
        a.k = rt.linear(st, p + ".to_k")
        a.v = rt.linear(st, p + ".to_v")
        a.o = rt.linear(st, p + ".to_out.0")
    else:                                         # LDM: 1x1 convs q / k / v / proj_out
        a.norm = rt.norm(st, p + ".norm", 32, 1e-6)
        a.q = rt.linear(st, p + ".q")
        a.k = rt.linear(st, p + ".k")
        a.v = rt.linear(st, p + ".v")
        a.o = rt.linear(st, p + ".proj_out")
    return a^


def _mid_attn_forward(mut rt: Runtime, a: MidAttn, x: Act) raises -> Act:
    var h = rt.group_norm(x, a.norm, False)
    var q = rt.linear(h, a.q)
    var k = rt.linear(h, a.k)
    var v = rt.linear(h, a.v)
    rt.free(h^)
    var o = rt.attention(q, k, v, 1)
    rt.free(q^)
    rt.free(k^)
    rt.free(v^)
    var y = rt.linear(o, a.o)
    rt.free(o^)
    rt.add_(y, x)
    return y^


struct Stage(Copyable, Movable):
    var resnets: List[ResBlock]
    var has_sampler: Bool
    var sampler: Conv

    def __init__(out self):
        self.resnets = List[ResBlock]()
        self.has_sampler = False
        self.sampler = Conv()


def _scaling_factor(dir: String) -> Float32:
    var cfg = read_bytes(dir + "/config.json")
    if len(cfg) == 0:
        return 1.0
    var doc = parse_json(cfg)
    return Float32(doc.number(doc.object_get(doc.root, "scaling_factor"), 1.0))


struct VaeDecoder(Movable):
    var has_pq: Bool
    var post_quant: Conv
    var conv_in: Conv
    var mid1: ResBlock
    var attn: MidAttn
    var mid2: ResBlock
    var stages: List[Stage]
    var norm_out: Norm
    var conv_out: Conv
    var latent_channels: Int
    var out_channels: Int
    var scaling_factor: Float32

    def __init__(out self):
        self.has_pq = False
        self.post_quant = Conv()
        self.conv_in = Conv()
        self.mid1 = ResBlock()
        self.attn = MidAttn()
        self.mid2 = ResBlock()
        self.stages = List[Stage]()
        self.norm_out = Norm()
        self.conv_out = Conv()
        self.latent_channels = 0
        self.out_channels = 0
        self.scaling_factor = 1.0

    @staticmethod
    def load(mut rt: Runtime, dir: String) raises -> VaeDecoder:
        """From a diffusers `vae/` folder (diffusers or LDM tensor names)."""
        var st = SafeTensors.open(dir + "/diffusion_pytorch_model.safetensors")
        var d = VaeDecoder()
        d.scaling_factor = _scaling_factor(dir)
        if st.has("post_quant_conv.weight"):
            d.post_quant = rt.conv(st, "post_quant_conv")
            d.has_pq = True
        d.conv_in = rt.conv(st, "decoder.conv_in")
        d.latent_channels = d.post_quant.cin if d.has_pq else d.conv_in.cin
        var g = 32
        var eps = Float32(1e-6)
        if st.has("decoder.mid_block.resnets.0.conv1.weight"):
            d.mid1 = load_resblock(rt, st, "decoder.mid_block.resnets.0", g, eps)
            d.attn = _load_mid_attn(rt, st, "decoder.mid_block.attentions.0")
            d.mid2 = load_resblock(rt, st, "decoder.mid_block.resnets.1", g, eps)
            var i = 0
            while st.has("decoder.up_blocks." + String(i) + ".resnets.0.conv1.weight"):
                var p = "decoder.up_blocks." + String(i)
                var s = Stage()
                var j = 0
                while st.has(p + ".resnets." + String(j) + ".conv1.weight"):
                    s.resnets.append(load_resblock(rt, st, p + ".resnets." + String(j), g, eps))
                    j += 1
                if st.has(p + ".upsamplers.0.conv.weight"):
                    s.sampler = rt.conv(st, p + ".upsamplers.0.conv")
                    s.has_sampler = True
                d.stages.append(s^)
                i += 1
            d.norm_out = rt.norm(st, "decoder.conv_norm_out", g, eps)
        else:
            d.mid1 = load_resblock(rt, st, "decoder.mid.block_1", g, eps)
            d.attn = _load_mid_attn(rt, st, "decoder.mid.attn_1")
            d.mid2 = load_resblock(rt, st, "decoder.mid.block_2", g, eps)
            var levels = 0
            while st.has("decoder.up." + String(levels) + ".block.0.conv1.weight"):
                levels += 1
            for lv in range(levels - 1, -1, -1):
                var p = "decoder.up." + String(lv)
                var s = Stage()
                var j = 0
                while st.has(p + ".block." + String(j) + ".conv1.weight"):
                    s.resnets.append(load_resblock(rt, st, p + ".block." + String(j), g, eps))
                    j += 1
                if st.has(p + ".upsample.conv.weight"):
                    s.sampler = rt.conv(st, p + ".upsample.conv")
                    s.has_sampler = True
                d.stages.append(s^)
            d.norm_out = rt.norm(st, "decoder.norm_out", g, eps)
        d.conv_out = rt.conv(st, "decoder.conv_out")
        d.out_channels = d.conv_out.cout
        return d^

    def decode(mut self, mut rt: Runtime, z: Act, tile: Bool = False) raises -> Act:
        """Image [b, 8h, 8w, out] from latents [b, h, w, latent] (already divided by the scaling factor)."""
        var mode = PAD_WRAP if tile else PAD_ZERO
        var empty = Act()
        var h: Act
        if self.has_pq:
            var t = rt.conv2d(z, self.post_quant, mode)
            h = rt.conv2d(t, self.conv_in, mode)
            rt.free(t^)
        else:
            h = rt.conv2d(z, self.conv_in, mode)
        var r = resblock_forward(rt, self.mid1, h, empty, False, mode)
        rt.free(h^)
        var a = _mid_attn_forward(rt, self.attn, r)
        rt.free(r^)
        h = resblock_forward(rt, self.mid2, a, empty, False, mode)
        rt.free(a^)
        for i in range(len(self.stages)):
            for j in range(len(self.stages[i].resnets)):
                var n = resblock_forward(rt, self.stages[i].resnets[j], h, empty, False, mode)
                rt.free(h^)
                h = n^
            if self.stages[i].has_sampler:
                var u = rt.conv2d(h, self.stages[i].sampler, mode, 2)
                rt.free(h^)
                h = u^
        var n = rt.group_norm(h, self.norm_out, True)
        rt.free(h^)
        var o = rt.conv2d(n, self.conv_out, mode)
        rt.free(n^)
        return o^


def wrap_pad(x: List[Float32], b: Int, h: Int, w: Int, c: Int, pad: Int) -> List[Float32]:
    """Circular padding of a channels-last [b, h, w, c] array by `pad` on every side."""
    var H = h + 2 * pad
    var W = w + 2 * pad
    var out = List[Float32](length=b * H * W * c, fill=0.0)
    for bi in range(b):
        for y in range(H):
            var sy = ((y - pad) % h + h) % h
            for xx in range(W):
                var sx = ((xx - pad) % w + w) % w
                var so = ((bi * h + sy) * w + sx) * c
                var do = ((bi * H + y) * W + xx) * c
                for k in range(c):
                    out[do + k] = x[so + k]
    return out^


def crop(x: List[Float32], b: Int, h: Int, w: Int, c: Int, top: Int, left: Int, oh: Int, ow: Int) -> List[Float32]:
    var out = List[Float32](length=b * oh * ow * c, fill=0.0)
    for bi in range(b):
        for y in range(oh):
            for xx in range(ow):
                var so = ((bi * h + top + y) * w + left + xx) * c
                var do = ((bi * oh + y) * ow + xx) * c
                for k in range(c):
                    out[do + k] = x[so + k]
    return out^


def decode_padded(mut rt: Runtime, mut dec: VaeDecoder, z: List[Float32], b: Int, h: Int, w: Int,
                  pad: Int) raises -> List[Float32]:
    """Seam-free decode as in StableMaterials / MatFuse: circularly pad the latent by `pad`,
    decode with zero padding and keep the centre [b, 8h, 8w, out] (host array)."""
    var zc = dec.latent_channels
    var zp = wrap_pad(z, b, h, w, zc, pad)
    var za = rt.upload(b, h + 2 * pad, w + 2 * pad, zc, zp)
    var img = dec.decode(rt, za, False)
    rt.free(za^)
    var full = rt.download(img)
    var f = img.h // (h + 2 * pad)
    var out = crop(full, b, img.h, img.w, img.c, pad * f, pad * f, h * f, w * f)
    rt.free(img^)
    return out^


struct VaeEncoder(Movable):
    var conv_in: Conv
    var stages: List[Stage]
    var mid1: ResBlock
    var attn: MidAttn
    var mid2: ResBlock
    var norm_out: Norm
    var conv_out: Conv
    var has_q: Bool
    var quant: Conv
    var latent_channels: Int
    var scaling_factor: Float32

    def __init__(out self):
        self.conv_in = Conv()
        self.stages = List[Stage]()
        self.mid1 = ResBlock()
        self.attn = MidAttn()
        self.mid2 = ResBlock()
        self.norm_out = Norm()
        self.conv_out = Conv()
        self.has_q = False
        self.quant = Conv()
        self.latent_channels = 0
        self.scaling_factor = 1.0

    @staticmethod
    def load(mut rt: Runtime, dir: String) raises -> VaeEncoder:
        """The encoder of a diffusers `AutoencoderKL` folder."""
        var st = SafeTensors.open(dir + "/diffusion_pytorch_model.safetensors")
        var e = VaeEncoder()
        e.scaling_factor = _scaling_factor(dir)
        var g = 32
        var eps = Float32(1e-6)
        e.conv_in = rt.conv(st, "encoder.conv_in")
        var i = 0
        while st.has("encoder.down_blocks." + String(i) + ".resnets.0.conv1.weight"):
            var p = "encoder.down_blocks." + String(i)
            var s = Stage()
            var j = 0
            while st.has(p + ".resnets." + String(j) + ".conv1.weight"):
                s.resnets.append(load_resblock(rt, st, p + ".resnets." + String(j), g, eps))
                j += 1
            if st.has(p + ".downsamplers.0.conv.weight"):
                # diffusers Downsample2D(padding=0): F.pad(x, (0, 1, 0, 1)) then a stride-2 conv
                var w = st.weight(p + ".downsamplers.0.conv.weight")
                var b = st.weight(p + ".downsamplers.0.conv.bias")
                s.sampler = rt.conv_from(w, b, True, 2, 0, 1, 0, 1)
                s.has_sampler = True
            e.stages.append(s^)
            i += 1
        e.mid1 = load_resblock(rt, st, "encoder.mid_block.resnets.0", g, eps)
        e.attn = _load_mid_attn(rt, st, "encoder.mid_block.attentions.0")
        e.mid2 = load_resblock(rt, st, "encoder.mid_block.resnets.1", g, eps)
        e.norm_out = rt.norm(st, "encoder.conv_norm_out", g, eps)
        e.conv_out = rt.conv(st, "encoder.conv_out")
        if st.has("quant_conv.weight"):
            e.quant = rt.conv(st, "quant_conv")
            e.has_q = True
        e.latent_channels = e.conv_out.cout // 2
        return e^

    def encode_mean(mut self, mut rt: Runtime, x: Act, tile: Bool = False) raises -> Act:
        """Posterior mean [b, h/8, w/8, latent] of an image in [-1, 1] (not yet scaled)."""
        var mode = PAD_WRAP if tile else PAD_ZERO
        var empty = Act()
        var h = rt.conv2d(x, self.conv_in, mode)
        for i in range(len(self.stages)):
            for j in range(len(self.stages[i].resnets)):
                var n = resblock_forward(rt, self.stages[i].resnets[j], h, empty, False, mode)
                rt.free(h^)
                h = n^
            if self.stages[i].has_sampler:
                var d = rt.conv2d(h, self.stages[i].sampler, mode)
                rt.free(h^)
                h = d^
        var r = resblock_forward(rt, self.mid1, h, empty, False, mode)
        rt.free(h^)
        var a = _mid_attn_forward(rt, self.attn, r)
        rt.free(r^)
        h = resblock_forward(rt, self.mid2, a, empty, False, mode)
        rt.free(a^)
        var n = rt.group_norm(h, self.norm_out, True)
        rt.free(h^)
        var m = rt.conv2d(n, self.conv_out, mode)
        rt.free(n^)
        if self.has_q:
            var q = rt.conv2d(m, self.quant, mode)
            rt.free(m^)
            m = q^
        var mean = rt.slice_c(m, 0, self.latent_channels)
        rt.free(m^)
        return mean^
