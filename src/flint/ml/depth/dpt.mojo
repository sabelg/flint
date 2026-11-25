"""The shared DPT reassemble-and-fuse head of `flint.ml.depth`.

Depth Anything V2 (`transformers` `DepthAnythingForDepthEstimation` names), Depth Anything 3
mono / metric (`DPT`, with the sky branch) and Depth Anything 3 any-view (`DualDPT`'s main
branch: an input LayerNorm and UV sine position embeddings) are the same network:

  reassemble  4 ViT layers -> 1x1 projection (+ UV embedding) -> resize (4x / 2x transposed
              conv, identity, 3x3 stride-2 conv) -> 3x3 `layerN_rn` to `features` channels
  fuse        refinenet4 .. refinenet1: [+ ResidualConvUnit(lateral)] -> ResidualConvUnit ->
              bilinear (align_corners) to the next level (2x for the last) -> 1x1 out conv
  output      3x3 conv to features / 2 -> bilinear to (14 ph, 14 pw) [+ UV embedding] ->
              3x3 conv to 32 -> ReLU -> 1x1 conv to `out_dim` (raw logits; the caller applies
              exp / ReLU / expp1); an optional sky head shares everything up to the 3x3 / 2 conv.

DualDPT's auxiliary ray branch is not needed for depth / pose (the camera decoder gives the
pose) and is not run.
"""

from flint.ml.diffusion.nn import Runtime, Act, Linear, Conv, Norm, PAD_ZERO
from flint.ml.depth.weights import Weights
from flint.ml.depth.layers import (
    ConvT, load_linear, load_conv, load_conv_t, load_norm, relu_, relu, resize_bilinear, conv_t, uv_embed,
    upload_rep,
)

comptime DPT_HF: Int = 0     # transformers DepthAnything names
comptime DPT_DA3: Int = 1    # depth_anything_3 DPT / DualDPT names


struct DptConfig(Copyable, Movable):
    var style: Int
    var prefix: String       # "model.head." / "" (transformers uses neck. + head.)
    var dim_in: Int
    var features: Int
    var out_channels: List[Int]
    var out_dim: Int
    var in_norm: Bool
    var pos_embed: Bool
    var sky: Bool
    var patch: Int

    def __init__(out self):
        self.style = DPT_DA3
        self.prefix = String()
        self.dim_in = 384
        self.features = 64
        self.out_channels = [48, 96, 192, 384]
        self.out_dim = 1
        self.in_norm = False
        self.pos_embed = False
        self.sky = False
        self.patch = 14


struct Rcu(Movable):
    var c1: Conv
    var c2: Conv

    def __init__(out self):
        self.c1 = Conv()
        self.c2 = Conv()


struct Fusion(Movable):
    var has_res: Bool
    var rcu1: Rcu
    var rcu2: Rcu
    var out: Linear

    def __init__(out self):
        self.has_res = False
        self.rcu1 = Rcu()
        self.rcu2 = Rcu()
        self.out = Linear()


struct DptOutput(Movable):
    var main: List[Float32]   # [S, H, W, out_dim] raw logits
    var sky: List[Float32]    # [S, H, W] raw logits (empty without a sky head)
    var h: Int
    var w: Int

    def __init__(out self):
        self.main = List[Float32]()
        self.sky = List[Float32]()
        self.h = 0
        self.w = 0


struct DptHead(Movable):
    var cfg: DptConfig
    var norm: Norm
    var proj: List[Linear]
    var up0: ConvT
    var up1: ConvT
    var down3: Conv
    var rn: List[Conv]
    var fuse: List[Fusion]      # refinenet4, 3, 2, 1
    var oc1: Conv
    var oc2a: Conv
    var oc2b: Linear
    var sky_a: Conv
    var sky_b: Linear

    def __init__(out self, var cfg: DptConfig):
        self.cfg = cfg^
        self.norm = Norm()
        self.proj = List[Linear]()
        self.up0 = ConvT()
        self.up1 = ConvT()
        self.down3 = Conv()
        self.rn = List[Conv]()
        self.fuse = List[Fusion]()
        self.oc1 = Conv()
        self.oc2a = Conv()
        self.oc2b = Linear()
        self.sky_a = Conv()
        self.sky_b = Linear()

    @staticmethod
    def load(mut rt: Runtime, w: Weights, var cfg: DptConfig) raises -> DptHead:
        var m = DptHead(cfg^)
        var p = m.cfg.prefix
        var hf = m.cfg.style == DPT_HF
        if m.cfg.in_norm:
            m.norm = load_norm(rt, w, p + "norm", 1, 1e-5)
        for i in range(4):
            if hf:
                m.proj.append(load_linear(rt, w, "neck.reassemble_stage.layers." + String(i) + ".projection"))
            else:
                m.proj.append(load_linear(rt, w, p + "projects." + String(i)))
        var rs0 = "neck.reassemble_stage.layers.0.resize" if hf else p + "resize_layers.0"
        var rs1 = "neck.reassemble_stage.layers.1.resize" if hf else p + "resize_layers.1"
        var rs3 = "neck.reassemble_stage.layers.3.resize" if hf else p + "resize_layers.3"
        m.up0 = load_conv_t(rt, w, rs0)
        m.up1 = load_conv_t(rt, w, rs1)
        m.down3 = load_conv(rt, w, rs3, 2, 1)
        for i in range(4):
            var name = "neck.convs." + String(i) if hf else p + "scratch.layer" + String(i + 1) + "_rn"
            m.rn.append(load_conv(rt, w, name, 1, 1, False))
        for j in range(4):
            var f = Fusion()
            f.has_res = j > 0
            var b: String
            if hf:
                b = "neck.fusion_stage.layers." + String(j) + "."
                if f.has_res:
                    f.rcu1.c1 = load_conv(rt, w, b + "residual_layer1.convolution1")
                    f.rcu1.c2 = load_conv(rt, w, b + "residual_layer1.convolution2")
                f.rcu2.c1 = load_conv(rt, w, b + "residual_layer2.convolution1")
                f.rcu2.c2 = load_conv(rt, w, b + "residual_layer2.convolution2")
                f.out = load_linear(rt, w, b + "projection")
            else:
                b = p + "scratch.refinenet" + String(4 - j) + "."
                if f.has_res:
                    f.rcu1.c1 = load_conv(rt, w, b + "resConfUnit1.conv1")
                    f.rcu1.c2 = load_conv(rt, w, b + "resConfUnit1.conv2")
                f.rcu2.c1 = load_conv(rt, w, b + "resConfUnit2.conv1")
                f.rcu2.c2 = load_conv(rt, w, b + "resConfUnit2.conv2")
                f.out = load_linear(rt, w, b + "out_conv")
            m.fuse.append(f^)
        if hf:
            m.oc1 = load_conv(rt, w, "head.conv1")
            m.oc2a = load_conv(rt, w, "head.conv2")
            m.oc2b = load_linear(rt, w, "head.conv3")
        else:
            m.oc1 = load_conv(rt, w, p + "scratch.output_conv1")
            m.oc2a = load_conv(rt, w, p + "scratch.output_conv2.0")
            m.oc2b = load_linear(rt, w, p + "scratch.output_conv2.2")
            if m.cfg.sky and w.has(p + "scratch.sky_output_conv2.0.weight"):
                m.sky_a = load_conv(rt, w, p + "scratch.sky_output_conv2.0")
                m.sky_b = load_linear(rt, w, p + "scratch.sky_output_conv2.2")
            else:
                m.cfg.sky = False
        return m^

    def _rcu(mut self, mut rt: Runtime, x: Act, j: Int, second: Bool) raises -> Act:
        var y = relu(rt, x)
        var a: Act
        if second:
            a = rt.conv2d(y, self.fuse[j].rcu2.c1, PAD_ZERO)
        else:
            a = rt.conv2d(y, self.fuse[j].rcu1.c1, PAD_ZERO)
        rt.free(y^)
        relu_(rt, a)
        var b: Act
        if second:
            b = rt.conv2d(a, self.fuse[j].rcu2.c2, PAD_ZERO)
        else:
            b = rt.conv2d(a, self.fuse[j].rcu1.c2, PAD_ZERO)
        rt.free(a^)
        rt.add_(b, x)
        return b^

    def _fusion(mut self, mut rt: Runtime, var x: Act, lateral: Act, j: Int, ho: Int, wo: Int) raises -> Act:
        if self.fuse[j].has_res:
            var r = self._rcu(rt, lateral, j, False)
            rt.add_(x, r)
            rt.free(r^)
        var y = self._rcu(rt, x, j, True)
        rt.free(x^)
        var u = resize_bilinear(rt, y, ho, wo, True)
        rt.free(y^)
        var o = rt.linear(u, self.fuse[j].out)
        rt.free(u^)
        return o^

    def _add_uv(mut self, mut rt: Runtime, mut x: Act, W: Int, H: Int) raises:
        var t = uv_embed(x.w, x.h, x.c, Float64(W) / Float64(H))
        var e = upload_rep(rt, t, x.b, x.h, x.w, x.c)
        rt.add_(x, e)
        rt.free(e^)

    def forward(mut self, mut rt: Runtime, feats: List[Act], H: Int, W: Int) raises -> DptOutput:
        """`feats`: 4 ViT layers as patch tokens [S, ph * pw, 1, dim_in] (token 0 removed);
        H, W: the network input size."""
        var ph = H // self.cfg.patch
        var pw = W // self.cfg.patch
        var S = feats[0].b
        var lv = List[Act]()
        for i in range(4):
            var f: Act
            if self.cfg.in_norm:
                f = rt.layer_norm(feats[i], self.norm)
            else:
                f = rt.copy(feats[i])
            f = rt.reshape(f^, S, ph, pw, feats[i].c)
            var x = rt.linear(f, self.proj[i])
            rt.free(f^)
            if self.cfg.pos_embed:
                self._add_uv(rt, x, W, H)
            var r: Act
            if i == 0:
                r = conv_t(rt, x, self.up0)
                rt.free(x^)
            elif i == 1:
                r = conv_t(rt, x, self.up1)
                rt.free(x^)
            elif i == 2:
                r = x^
            else:
                r = rt.conv2d(x, self.down3, PAD_ZERO)
                rt.free(x^)
            var n = rt.conv2d(r, self.rn[i], PAD_ZERO)
            rt.free(r^)
            lv.append(n^)
        # top-down fusion
        var none = Act()
        var out = self._fusion(rt, rt.copy(lv[3]), none, 0, lv[2].h, lv[2].w)
        out = self._fusion(rt, out^, lv[2], 1, lv[1].h, lv[1].w)
        out = self._fusion(rt, out^, lv[1], 2, lv[0].h, lv[0].w)
        out = self._fusion(rt, out^, lv[0], 3, 2 * lv[0].h, 2 * lv[0].w)
        while len(lv) > 0:
            var t = lv.pop()
            rt.free(t^)
        var o1 = rt.conv2d(out, self.oc1, PAD_ZERO)
        rt.free(out^)
        var ho = ph * self.cfg.patch
        var wo = pw * self.cfg.patch
        var up = resize_bilinear(rt, o1, ho, wo, True)
        rt.free(o1^)
        if self.cfg.pos_embed:
            self._add_uv(rt, up, W, H)
        var res = DptOutput()
        res.h = ho
        res.w = wo
        var a = rt.conv2d(up, self.oc2a, PAD_ZERO)
        relu_(rt, a)
        var m = rt.linear(a, self.oc2b)
        rt.free(a^)
        res.main = rt.download(m)
        rt.free(m^)
        if self.cfg.sky:
            var sa = rt.conv2d(up, self.sky_a, PAD_ZERO)
            relu_(rt, sa)
            var sm = rt.linear(sa, self.sky_b)
            rt.free(sa^)
            res.sky = rt.download(sm)
            rt.free(sm^)
        rt.free(up^)
        return res^
