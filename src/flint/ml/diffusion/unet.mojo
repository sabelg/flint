"""`UNet2DConditionModel` (diffusers layout) for `flint.ml.diffusion`.

Covers the Stable Diffusion 1.x / 2.x family and its relatives (StableMaterials, MatFuse, Marigold,
SD LoRA-merged checkpoints): ResNet blocks with time-embedding injection, `Transformer2DModel`
blocks (GroupNorm -> proj_in -> [self-attention, cross-attention, GEGLU feed-forward] ->
proj_out, residual), stride-2 downsamplers, nearest + conv upsamplers, skip concatenation, the
sinusoidal time projection and, for LCM-distilled models, the guidance-scale `cond_proj`.

The block structure (number of resnets, which blocks carry attention / samplers, channels) is
read from the safetensors names and shapes; heads, norm groups / eps and the time projection
come from `config.json`. `forward(rt, x, t, ctx, tcond, tile)` runs one denoiser evaluation on a
batch: x [b, h, w, in], ctx [b, tokens, cross_dim] (already on the runtime); `tile = True` makes
every convolution wrap around (seamless / tileable output, StableMaterials' "feature rolling").
"""

from std.collections import Dict
from flint.assets.json import parse_json
from flint.platform.os import read_bytes
from flint.ml.tensor import Tensor
from flint.ml.safetensors import SafeTensors
from flint.ml.diffusion.nn import Runtime, Act, Linear, Conv, Norm, PAD_ZERO, PAD_WRAP
from flint.ml.diffusion.kernels import ACT_SILU
from flint.ml.diffusion.schedulers import timestep_embedding


struct ResBlock(Copyable, Movable):
    var norm1: Norm
    var conv1: Conv
    var temb: Linear
    var has_temb: Bool
    var norm2: Norm
    var conv2: Conv
    var has_short: Bool
    var short: Conv

    def __init__(out self):
        self.norm1 = Norm()
        self.conv1 = Conv()
        self.temb = Linear()
        self.has_temb = False
        self.norm2 = Norm()
        self.conv2 = Conv()
        self.has_short = False
        self.short = Conv()


def load_resblock(mut rt: Runtime, st: SafeTensors, p: String, groups: Int, eps: Float32) raises -> ResBlock:
    """diffusers `ResnetBlock2D` (norm1/conv1/time_emb_proj/norm2/conv2/conv_shortcut) or the LDM
    names (`nin_shortcut`, `temb_proj`)."""
    var r = ResBlock()
    r.norm1 = rt.norm(st, p + ".norm1", groups, eps)
    r.conv1 = rt.conv(st, p + ".conv1")
    if st.has(p + ".time_emb_proj.weight"):
        r.temb = rt.linear(st, p + ".time_emb_proj")
        r.has_temb = True
    elif st.has(p + ".temb_proj.weight"):
        r.temb = rt.linear(st, p + ".temb_proj")
        r.has_temb = True
    r.norm2 = rt.norm(st, p + ".norm2", groups, eps)
    r.conv2 = rt.conv(st, p + ".conv2")
    if st.has(p + ".conv_shortcut.weight"):
        r.short = rt.conv(st, p + ".conv_shortcut")
        r.has_short = True
    elif st.has(p + ".nin_shortcut.weight"):
        r.short = rt.conv(st, p + ".nin_shortcut")
        r.has_short = True
    return r^


def resblock_forward(mut rt: Runtime, r: ResBlock, x: Act, temb: Act, has_temb: Bool, mode: Int) raises -> Act:
    """x + conv2(silu(gn2(conv1(silu(gn1(x))) + temb_proj(silu(temb))))); `temb` is already SiLU'd."""
    var h = rt.group_norm(x, r.norm1, True)
    var h1 = rt.conv2d(h, r.conv1, mode)
    rt.free(h^)
    if has_temb and r.has_temb:
        var t = rt.linear(temb, r.temb)
        rt.bcast_add_(h1, t)
        rt.free(t^)
    var h2 = rt.group_norm(h1, r.norm2, True)
    rt.free(h1^)
    var h3 = rt.conv2d(h2, r.conv2, mode)
    rt.free(h2^)
    if r.has_short:
        var s = rt.conv2d(x, r.short, mode)
        rt.add_(h3, s)
        rt.free(s^)
    else:
        rt.add_(h3, x)
    return h3^


struct AttnLayer(Copyable, Movable):
    var q: Linear
    var k: Linear
    var v: Linear
    var o: Linear

    def __init__(out self):
        self.q = Linear()
        self.k = Linear()
        self.v = Linear()
        self.o = Linear()


def load_attn(mut rt: Runtime, st: SafeTensors, p: String) raises -> AttnLayer:
    var a = AttnLayer()
    a.q = rt.linear(st, p + ".to_q", st.has(p + ".to_q.bias"))
    a.k = rt.linear(st, p + ".to_k", st.has(p + ".to_k.bias"))
    a.v = rt.linear(st, p + ".to_v", st.has(p + ".to_v.bias"))
    a.o = rt.linear(st, p + ".to_out.0", True)
    return a^


def attn_forward(mut rt: Runtime, a: AttnLayer, x: Act, ctx: Act, heads: Int, self_attn: Bool) raises -> Act:
    var q = rt.linear(x, a.q)
    var k: Act
    var v: Act
    if self_attn:
        k = rt.linear(x, a.k)
        v = rt.linear(x, a.v)
    else:
        k = rt.linear(ctx, a.k)
        v = rt.linear(ctx, a.v)
    var o = rt.attention(q, k, v, heads)
    rt.free(q^)
    rt.free(k^)
    rt.free(v^)
    var y = rt.linear(o, a.o)
    rt.free(o^)
    return y^


struct TBlock(Copyable, Movable):
    var ln1: Norm
    var attn1: AttnLayer
    var ln2: Norm
    var attn2: AttnLayer
    var ln3: Norm
    var ff1: Linear
    var ff2: Linear

    def __init__(out self):
        self.ln1 = Norm()
        self.attn1 = AttnLayer()
        self.ln2 = Norm()
        self.attn2 = AttnLayer()
        self.ln3 = Norm()
        self.ff1 = Linear()
        self.ff2 = Linear()


struct Transformer2D(Copyable, Movable):
    var norm: Norm
    var proj_in: Conv
    var blocks: List[TBlock]
    var proj_out: Conv
    var heads: Int

    def __init__(out self):
        self.norm = Norm()
        self.proj_in = Conv()
        self.blocks = List[TBlock]()
        self.proj_out = Conv()
        self.heads = 8


def _conv_or_linear(mut rt: Runtime, st: SafeTensors, name: String) raises -> Conv:
    """A 1x1 conv (`use_linear_projection = False`) or a Linear (True), both as a 1x1 Conv."""
    var w = st.weight(name + ".weight")
    var b = st.weight(name + ".bias")
    if w.rank() == 2:
        var w4 = w.reshaped([w.shape[0], w.shape[1], 1, 1])
        return rt.conv_from(w4, b, True, 1, 0, 0, 0, 0)
    return rt.conv_from(w, b, True, 1, 0, 0, 0, 0)


def load_transformer(mut rt: Runtime, st: SafeTensors, p: String, heads: Int, groups: Int) raises -> Transformer2D:
    var t = Transformer2D()
    t.heads = heads
    t.norm = rt.norm(st, p + ".norm", groups, 1e-6)
    t.proj_in = _conv_or_linear(rt, st, p + ".proj_in")
    var i = 0
    while st.has(p + ".transformer_blocks." + String(i) + ".norm1.weight"):
        var q = p + ".transformer_blocks." + String(i)
        var b = TBlock()
        b.ln1 = rt.norm(st, q + ".norm1", 1, 1e-5)
        b.attn1 = load_attn(rt, st, q + ".attn1")
        b.ln2 = rt.norm(st, q + ".norm2", 1, 1e-5)
        b.attn2 = load_attn(rt, st, q + ".attn2")
        b.ln3 = rt.norm(st, q + ".norm3", 1, 1e-5)
        b.ff1 = rt.linear(st, q + ".ff.net.0.proj")
        b.ff2 = rt.linear(st, q + ".ff.net.2")
        t.blocks.append(b^)
        i += 1
    t.proj_out = _conv_or_linear(rt, st, p + ".proj_out")
    return t^


def transformer_forward(mut rt: Runtime, t: Transformer2D, x: Act, ctx: Act, mode: Int) raises -> Act:
    var h0 = rt.group_norm(x, t.norm, False)
    var h = rt.conv2d(h0, t.proj_in, mode)
    rt.free(h0^)
    for i in range(len(t.blocks)):
        var n1 = rt.layer_norm(h, t.blocks[i].ln1)
        var a1 = attn_forward(rt, t.blocks[i].attn1, n1, n1, t.heads, True)
        rt.free(n1^)
        rt.add_(h, a1)
        rt.free(a1^)
        var n2 = rt.layer_norm(h, t.blocks[i].ln2)
        var a2 = attn_forward(rt, t.blocks[i].attn2, n2, ctx, t.heads, False)
        rt.free(n2^)
        rt.add_(h, a2)
        rt.free(a2^)
        var n3 = rt.layer_norm(h, t.blocks[i].ln3)
        var f1 = rt.linear(n3, t.blocks[i].ff1)
        rt.free(n3^)
        var g = rt.geglu(f1)
        rt.free(f1^)
        var f2 = rt.linear(g, t.blocks[i].ff2)
        rt.free(g^)
        rt.add_(h, f2)
        rt.free(f2^)
    var o = rt.conv2d(h, t.proj_out, mode)
    rt.free(h^)
    rt.add_(o, x)
    return o^


struct UBlock(Copyable, Movable):
    var resnets: List[ResBlock]
    var attns: List[Transformer2D]
    var has_attn: Bool
    var has_sampler: Bool
    var sampler: Conv

    def __init__(out self):
        self.resnets = List[ResBlock]()
        self.attns = List[Transformer2D]()
        self.has_attn = False
        self.has_sampler = False
        self.sampler = Conv()


def _json_heads(doc_text: List[UInt8], n_blocks: Int) raises -> List[Int]:
    """Heads per down block: `num_attention_heads` or (diffusers' legacy naming) `attention_head_dim`."""
    var doc = parse_json(doc_text)
    var r = doc.root
    var node = doc.object_get(r, "num_attention_heads")
    if node < 0 or doc.nodes[node].kind == 0:
        node = doc.object_get(r, "attention_head_dim")
    var out = List[Int]()
    if node >= 0 and doc.array_size(node) > 0:
        for i in range(doc.array_size(node)):
            out.append(doc.integer(doc.at(node, i), 8))
    else:
        var v = doc.integer(node, 8) if node >= 0 else 8
        for _ in range(n_blocks):
            out.append(v)
    return out^


def resize_nearest_act(mut rt: Runtime, x: Act, oh: Int, ow: Int) raises -> Act:
    """`F.interpolate(x, size=(oh, ow), mode="nearest")` of a channels-last value (src = floor(dst
    * in / out) in float32, as ATen). Round-trips through the host on the GPU: only used for the
    odd-sized up-sampling of latents that are not a multiple of 8."""
    var src = rt.download(x)
    var c = x.c
    var out = List[Float32](length=x.b * oh * ow * c, fill=0.0)
    var sy = Float32(x.h) / Float32(oh)
    var sx = Float32(x.w) / Float32(ow)
    for b in range(x.b):
        for y in range(oh):
            var iy = min(Int(Float32(y) * sy), x.h - 1)
            for xx in range(ow):
                var ix = min(Int(Float32(xx) * sx), x.w - 1)
                var so = ((b * x.h + iy) * x.w + ix) * c
                var do = ((b * oh + y) * ow + xx) * c
                for k in range(c):
                    out[do + k] = src[so + k]
    return rt.upload(x.b, oh, ow, c, out)


struct UNet(Movable):
    var conv_in: Conv
    var t1: Linear
    var t2: Linear
    var has_cond: Bool
    var cond_proj: Linear
    var cond_dim: Int
    var down: List[UBlock]
    var mid_res: List[ResBlock]
    var mid_attn: List[Transformer2D]
    var up: List[UBlock]
    var norm_out: Norm
    var conv_out: Conv
    var in_channels: Int
    var out_channels: Int
    var time_dim: Int
    var flip_sin_to_cos: Bool
    var freq_shift: Float64
    var cross_dim: Int

    def __init__(out self):
        self.conv_in = Conv()
        self.t1 = Linear()
        self.t2 = Linear()
        self.has_cond = False
        self.cond_proj = Linear()
        self.cond_dim = 0
        self.down = List[UBlock]()
        self.mid_res = List[ResBlock]()
        self.mid_attn = List[Transformer2D]()
        self.up = List[UBlock]()
        self.norm_out = Norm()
        self.conv_out = Conv()
        self.in_channels = 0
        self.out_channels = 0
        self.time_dim = 0
        self.flip_sin_to_cos = True
        self.freq_shift = 0.0
        self.cross_dim = 0

    @staticmethod
    def load(mut rt: Runtime, dir: String) raises -> UNet:
        """From a diffusers `unet/` folder (config.json + diffusion_pytorch_model.safetensors)."""
        var cfg = read_bytes(dir + "/config.json")
        if len(cfg) == 0:
            raise Error("unet: cannot read " + dir + "/config.json")
        var doc = parse_json(cfg)
        var root = doc.root
        var groups = doc.integer(doc.object_get(root, "norm_num_groups"), 32)
        var eps = Float32(doc.number(doc.object_get(root, "norm_eps"), 1e-5))
        var u = UNet()
        u.flip_sin_to_cos = doc.boolean(doc.object_get(root, "flip_sin_to_cos"), True)
        u.freq_shift = doc.number(doc.object_get(root, "freq_shift"), 0.0)
        var st = SafeTensors.open(dir + "/diffusion_pytorch_model.safetensors")
        var nd = 0
        while st.has("down_blocks." + String(nd) + ".resnets.0.conv1.weight"):
            nd += 1
        var heads = _json_heads(cfg, nd)
        u.conv_in = rt.conv(st, "conv_in")
        u.in_channels = u.conv_in.cin
        u.time_dim = u.conv_in.cout
        u.t1 = rt.linear(st, "time_embedding.linear_1")
        u.t2 = rt.linear(st, "time_embedding.linear_2")
        if st.has("time_embedding.cond_proj.weight"):
            u.cond_proj = rt.linear(st, "time_embedding.cond_proj", False)
            u.has_cond = True
            u.cond_dim = u.cond_proj.k
        for i in range(nd):
            var p = "down_blocks." + String(i)
            var blk = UBlock()
            var j = 0
            while st.has(p + ".resnets." + String(j) + ".conv1.weight"):
                blk.resnets.append(load_resblock(rt, st, p + ".resnets." + String(j), groups, eps))
                if st.has(p + ".attentions." + String(j) + ".norm.weight"):
                    blk.attns.append(load_transformer(rt, st, p + ".attentions." + String(j), heads[i], groups))
                    blk.has_attn = True
                j += 1
            if st.has(p + ".downsamplers.0.conv.weight"):
                blk.sampler = rt.conv(st, p + ".downsamplers.0.conv", 2, 1)
                blk.has_sampler = True
            u.down.append(blk^)
        var j = 0
        while st.has("mid_block.resnets." + String(j) + ".conv1.weight"):
            u.mid_res.append(load_resblock(rt, st, "mid_block.resnets." + String(j), groups, eps))
            j += 1
        j = 0
        while st.has("mid_block.attentions." + String(j) + ".norm.weight"):
            u.mid_attn.append(load_transformer(rt, st, "mid_block.attentions." + String(j), heads[len(heads) - 1],
                                               groups))
            j += 1
        var nu = 0
        while st.has("up_blocks." + String(nu) + ".resnets.0.conv1.weight"):
            nu += 1
        for i in range(nu):
            var p = "up_blocks." + String(i)
            var blk = UBlock()
            var k = 0
            var hi = heads[len(heads) - 1 - i] if len(heads) - 1 - i >= 0 else heads[0]
            while st.has(p + ".resnets." + String(k) + ".conv1.weight"):
                blk.resnets.append(load_resblock(rt, st, p + ".resnets." + String(k), groups, eps))
                if st.has(p + ".attentions." + String(k) + ".norm.weight"):
                    blk.attns.append(load_transformer(rt, st, p + ".attentions." + String(k), hi, groups))
                    blk.has_attn = True
                k += 1
            if st.has(p + ".upsamplers.0.conv.weight"):
                blk.sampler = rt.conv(st, p + ".upsamplers.0.conv")
                blk.has_sampler = True
            u.up.append(blk^)
        u.norm_out = rt.norm(st, "conv_norm_out", groups, eps)
        u.conv_out = rt.conv(st, "conv_out")
        u.out_channels = u.conv_out.cout
        if len(u.down) > 0 and u.down[0].has_attn:
            u.cross_dim = u.down[0].attns[0].blocks[0].attn2.k.k
        elif len(u.mid_attn) > 0:
            u.cross_dim = u.mid_attn[0].blocks[0].attn2.k.k
        return u^

    def time_embed(mut self, mut rt: Runtime, t: Float64, batch: Int, tcond: List[Float32]) raises -> Act:
        """SiLU(time_embedding(time_proj(t) [+ cond_proj(tcond)])) for every image: [b, 1, 1, 4 * C]."""
        var e = timestep_embedding(t, self.time_dim, self.flip_sin_to_cos, self.freq_shift)
        var rows = List[Float32]()
        for _ in range(batch):
            for v in e:
                rows.append(v)
        var x = rt.upload(batch, 1, 1, self.time_dim, rows)
        if self.has_cond and len(tcond) > 0:
            var tc = List[Float32]()
            for b in range(batch):
                for k in range(self.cond_dim):
                    tc.append(tcond[(b * self.cond_dim + k) % len(tcond)])
            var c = rt.upload(batch, 1, 1, self.cond_dim, tc)
            var cp = rt.linear(c, self.cond_proj)
            rt.free(c^)
            rt.add_(x, cp)
            rt.free(cp^)
        var h = rt.linear(x, self.t1)
        rt.free(x^)
        rt.activate(h, ACT_SILU)
        var o = rt.linear(h, self.t2)
        rt.free(h^)
        rt.activate(o, ACT_SILU)
        return o^

    def forward(mut self, mut rt: Runtime, x: Act, t: Float64, ctx: Act, tcond: List[Float32], tile: Bool) raises -> Act:
        """Noise / velocity prediction for x [b, h, w, in_channels] at timestep t."""
        var mode = PAD_WRAP if tile else PAD_ZERO
        var temb = self.time_embed(rt, t, x.b, tcond)
        var h = rt.conv2d(x, self.conv_in, mode)
        var skips = List[Act]()
        skips.append(rt.copy(h))
        for i in range(len(self.down)):
            for j in range(len(self.down[i].resnets)):
                var r = resblock_forward(rt, self.down[i].resnets[j], h, temb, True, mode)
                rt.free(h^)
                h = r^
                if self.down[i].has_attn:
                    var a = transformer_forward(rt, self.down[i].attns[j], h, ctx, mode)
                    rt.free(h^)
                    h = a^
                skips.append(rt.copy(h))
            if self.down[i].has_sampler:
                var d = rt.conv2d(h, self.down[i].sampler, mode)
                rt.free(h^)
                h = d^
                skips.append(rt.copy(h))
        for j in range(len(self.mid_res)):
            var r = resblock_forward(rt, self.mid_res[j], h, temb, True, mode)
            rt.free(h^)
            h = r^
            if j < len(self.mid_attn):
                var a = transformer_forward(rt, self.mid_attn[j], h, ctx, mode)
                rt.free(h^)
                h = a^
        for i in range(len(self.up)):
            for j in range(len(self.up[i].resnets)):
                var s = skips.pop()
                var cat = rt.concat_c(h, s)
                rt.free(s^)
                rt.free(h^)
                var r = resblock_forward(rt, self.up[i].resnets[j], cat, temb, True, mode)
                rt.free(cat^)
                h = r^
                if self.up[i].has_attn:
                    var a = transformer_forward(rt, self.up[i].attns[j], h, ctx, mode)
                    rt.free(h^)
                    h = a^
            if self.up[i].has_sampler:
                # latents that are not a multiple of 2^downsamplers: diffusers upsamples to the
                # size of the next skip (`forward_upsample_size`) instead of exactly 2x
                var sh = skips[len(skips) - 1].h if len(skips) > 0 else 2 * h.h
                var sw = skips[len(skips) - 1].w if len(skips) > 0 else 2 * h.w
                if sh != 2 * h.h or sw != 2 * h.w:
                    var r = resize_nearest_act(rt, h, sh, sw)
                    rt.free(h^)
                    h = rt.conv2d(r, self.up[i].sampler, mode)
                    rt.free(r^)
                    continue
                var u = rt.conv2d(h, self.up[i].sampler, mode, 2)
                rt.free(h^)
                h = u^
        while len(skips) > 0:
            rt.free(skips.pop())
        rt.free(temb^)
        var n = rt.group_norm(h, self.norm_out, True)
        rt.free(h^)
        var o = rt.conv2d(n, self.conv_out, mode)
        rt.free(n^)
        return o^
