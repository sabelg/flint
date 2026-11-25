"""MoGe-2 (`Ruicheng/moge-2-vit{s,b,l}-normal`) heads and geometry recovery for `flint.ml.depth`.

MoGe-2 puts a ConvStack pyramid on the shared DINOv2 trunk instead of a DPT fuse:

  encoder     the image is resized (bilinear, antialiased) to a base grid of ~`num_tokens`
              patches with the input aspect, two (vits) / four ViT layers are projected by 1x1
              convs and summed; the class token feeds the metric-scale MLP
  neck        5 levels (base, 2x .. 16x): [features | view-plane UV] -> 1x1 conv, residual
              3x3 blocks (replicate padding), 2x transposed-conv or bilinear resamplers
  heads       points (3), normal (3), mask (1): the same ConvStack on the neck outputs, the
              last level's 1x1 projection, bilinear (align_corners False) to the input size
  geometry    points z = exp(z), xy *= z; normals L2-normalised; mask sigmoid > 0.5; the focal
              and z-shift that best project the point map onto the view plane (the 1-D
              Levenberg-Marquardt fit of `recover_focal_shift` on a 64 x 64 nearest
              subsample), depth = z + shift, points re-projected from depth (OpenCV pixel
              centres), times exp(metric scale).

The checkpoint is the original `model.pt` pickle, read in place by `flint.ml.torch_pt`.
"""

from std.math import sqrt, exp, tan
from flint.ml.diffusion.nn import Runtime, Act, Linear, Conv, Norm
from flint.ml.depth.weights import Weights
from flint.ml.depth.layers import (
    ConvT, load_linear, load_conv, load_conv_t, load_norm, relu_, resize_bilinear, conv_t, conv_rep,
    torch_linspace, take_tokens,
)
from flint.ml.depth.vit import VitConfig, DinoVit, vit_size, VIT_DINO, POS_OFFSET


struct ResBlock(Copyable, Movable):
    var in_norm: Int          # 0 none, else GroupNorm groups
    var n1: Norm
    var c1: Conv
    var hid_norm: Int
    var n2: Norm
    var c2: Conv

    def __init__(out self):
        self.in_norm = 0
        self.n1 = Norm()
        self.c1 = Conv()
        self.hid_norm = 0
        self.n2 = Norm()
        self.c2 = Conv()


struct ConvStack(Copyable, Movable):
    var levels: Int
    var has_in: List[Bool]
    var inp: List[Linear]
    var res_start: List[Int]      # index into `res` of level i's first block
    var res_count: List[Int]
    var res: List[ResBlock]
    var rs_kind: List[Int]        # 0 conv_transpose, 1 bilinear
    var rs_t: List[ConvT]
    var rs_c: List[Conv]
    var has_out: List[Bool]
    var outp: List[Linear]

    def __init__(out self):
        self.levels = 0
        self.has_in = List[Bool]()
        self.inp = List[Linear]()
        self.res_start = List[Int]()
        self.res_count = List[Int]()
        self.res = List[ResBlock]()
        self.rs_kind = List[Int]()
        self.rs_t = List[ConvT]()
        self.rs_c = List[Conv]()
        self.has_out = List[Bool]()
        self.outp = List[Linear]()


def _norm_groups(kind: String, ch: Int) -> Int:
    if kind == "group_norm":
        return ch // 32
    if kind == "layer_norm":
        return 1
    if kind == "none" or kind == "":
        return 0
    return -1


def load_convstack(mut rt: Runtime, w: Weights, name: String, only_last_output: Bool) raises -> ConvStack:
    var cs = ConvStack()
    var dims = w.cfg_ints(name + ".dim_res_blocks")
    var dim_in = w.cfg_ints(name + ".dim_in")
    var dim_out = w.cfg_ints(name + ".dim_out")
    var nres = w.cfg_ints(name + ".num_res_blocks")
    var rsk = w.cfg_strs(name + ".resamplers")
    var in_norm = w.cfg_str(name + ".res_block_in_norm")
    var hid_norm = w.cfg_str(name + ".res_block_hidden_norm")
    if in_norm == "":
        in_norm = "layer_norm"
    if hid_norm == "":
        hid_norm = "group_norm"
    var L = len(dims)
    cs.levels = L
    for i in range(L):
        var ii = name + ".input_blocks." + String(i)
        var hi = w.has(ii + ".weight")
        cs.has_in.append(hi)
        cs.inp.append(load_linear(rt, w, ii) if hi else Linear())
        var n = nres[i] if len(nres) == L else (nres[0] if len(nres) > 0 else 1)
        cs.res_start.append(len(cs.res))
        cs.res_count.append(n)
        for j in range(n):
            var b = name + ".res_blocks." + String(i) + "." + String(j) + ".layers."
            var rb = ResBlock()
            rb.in_norm = _norm_groups(in_norm, dims[i])
            rb.hid_norm = _norm_groups(hid_norm, dims[i])
            if rb.in_norm < 0 or rb.hid_norm < 0:
                raise Error("depth: unsupported MoGe res-block norm " + in_norm + " / " + hid_norm)
            if rb.in_norm > 0:
                rb.n1 = load_norm(rt, w, b + "0", rb.in_norm, 1e-5)
            rb.c1 = load_conv(rt, w, b + "2", 1, 0)
            if rb.hid_norm > 0:
                rb.n2 = load_norm(rt, w, b + "3", rb.hid_norm, 1e-5)
            rb.c2 = load_conv(rt, w, b + "5", 1, 0)
            if w.has(name + ".res_blocks." + String(i) + "." + String(j) + ".skip_connection.weight"):
                raise Error("depth: MoGe res-block channel change is not supported")
            cs.res.append(rb^)
        var oi = name + ".output_blocks." + String(i)
        var ho = w.has(oi + ".weight") and (not only_last_output or i == L - 1)
        cs.has_out.append(ho)
        cs.outp.append(load_linear(rt, w, oi) if ho else Linear())
    for i in range(L - 1):
        var kind = rsk[i] if len(rsk) == L - 1 else (rsk[0] if len(rsk) > 0 else "conv_transpose")
        var r = name + ".resamplers." + String(i)
        if kind == "conv_transpose":
            cs.rs_kind.append(0)
            cs.rs_t.append(load_conv_t(rt, w, r + ".0"))
        elif kind == "bilinear":
            cs.rs_kind.append(1)
            cs.rs_t.append(ConvT())
        else:
            raise Error("depth: unsupported MoGe resampler " + kind)
        cs.rs_c.append(load_conv(rt, w, r + ".1", 1, 0))
    _ = dim_in
    _ = dim_out
    return cs^


def _resblock(mut rt: Runtime, x: Act, rb: ResBlock) raises -> Act:
    var y: Act
    if rb.in_norm > 0:
        y = rt.group_norm(x, rb.n1)
    else:
        y = rt.copy(x)
    relu_(rt, y)
    var a = conv_rep(rt, y, rb.c1)
    rt.free(y^)
    if rb.hid_norm > 0:
        var t = rt.group_norm(a, rb.n2)
        rt.free(a^)
        a = t^
    relu_(rt, a)
    var b = conv_rep(rt, a, rb.c2)
    rt.free(a^)
    rt.add_(b, x)
    return b^


def run_convstack(mut rt: Runtime, cs: ConvStack, inputs: List[Act], all_levels: Bool) raises -> List[Act]:
    """ConvStack.forward: one output per level (identity outputs are copies), or only the last
    level's output when `all_levels` is off."""
    var outs = List[Act]()
    var x = Act()
    for i in range(cs.levels):
        var f: Act
        if cs.has_in[i]:
            f = rt.linear(inputs[i], cs.inp[i])
        else:
            f = rt.copy(inputs[i])
        if i == 0:
            x = f^
        else:
            rt.add_(x, f)
            rt.free(f^)
        for j in range(cs.res_count[i]):
            var nx = _resblock(rt, x, cs.res[cs.res_start[i] + j])
            rt.free(x^)
            x = nx^
        if all_levels or i == cs.levels - 1:
            if cs.has_out[i]:
                outs.append(rt.linear(x, cs.outp[i]))
            else:
                outs.append(rt.copy(x))
        if i < cs.levels - 1:
            var up: Act
            if cs.rs_kind[i] == 0:
                up = conv_t(rt, x, cs.rs_t[i])
            else:
                up = resize_bilinear(rt, x, 2 * x.h, 2 * x.w, False)
            rt.free(x^)
            x = conv_rep(rt, up, cs.rs_c[i])
            rt.free(up^)
    rt.free(x^)
    return outs^


def view_plane_uv(w: Int, h: Int, aspect: Float64) -> List[Float32]:
    """MoGe `normalized_view_plane_uv` as channels-last [h, w, 2]."""
    var sx = aspect / sqrt(1.0 + aspect * aspect)
    var sy = 1.0 / sqrt(1.0 + aspect * aspect)
    var u = torch_linspace(Float32(-sx * Float64(w - 1) / Float64(w)), Float32(sx * Float64(w - 1) / Float64(w)), w)
    var v = torch_linspace(Float32(-sy * Float64(h - 1) / Float64(h)), Float32(sy * Float64(h - 1) / Float64(h)), h)
    var out = List[Float32](length=h * w * 2, fill=0.0)
    for y in range(h):
        for x in range(w):
            out[2 * (y * w + x)] = u[x]
            out[2 * (y * w + x) + 1] = v[y]
    return out^


def py_round(x: Float64) -> Int:
    """Python 3 `round` (half to even)."""
    var f = Int(x)
    if Float64(f) > x:
        f -= 1
    var d = x - Float64(f)
    if d > 0.5:
        return f + 1
    if d < 0.5:
        return f
    return f if f % 2 == 0 else f + 1


def base_grid(num_tokens: Int, H: Int, W: Int) -> Tuple[Int, Int]:
    var ar = Float64(W) / Float64(H)
    return (py_round(sqrt(Float64(num_tokens) / ar)), py_round(sqrt(Float64(num_tokens) * ar)))


# ----------------------------------------------------------------------------- focal / shift

def _focal_shift_cost(uv: List[Float64], xyz: List[Float64], n: Int, shift: Float64,
                      fixed_focal: Float64) -> Tuple[Float64, Float64]:
    """(cost, focal) of `focal * xy / (z + shift) - uv`; focal is the closed-form optimum unless
    `fixed_focal` > 0."""
    var num = 0.0
    var den = 0.0
    for i in range(n):
        var zz = xyz[3 * i + 2] + shift
        var px = xyz[3 * i] / zz
        var py = xyz[3 * i + 1] / zz
        num += px * uv[2 * i] + py * uv[2 * i + 1]
        den += px * px + py * py
    var f = fixed_focal if fixed_focal > 0 else (num / den if den > 0 else 1.0)
    var c = 0.0
    for i in range(n):
        var zz = xyz[3 * i + 2] + shift
        var ex = f * xyz[3 * i] / zz - uv[2 * i]
        var ey = f * xyz[3 * i + 1] / zz - uv[2 * i + 1]
        c += ex * ex + ey * ey
    return (c, f)


def solve_focal_shift(uv: List[Float64], xyz: List[Float64], n: Int, fixed_focal: Float64 = 0.0) -> Tuple[Float64, Float64]:
    """(shift, focal) minimising the view-plane reprojection error (1-D Levenberg-Marquardt on the
    shift from 0, like `scipy.optimize.least_squares(method="lm")`)."""
    var s = 0.0
    var cur = _focal_shift_cost(uv, xyz, n, s, fixed_focal)
    var lam = 1e-3
    for _ in range(200):
        var h = 1.4901161193847656e-08 * max(abs(s), 1.0)
        var cp = _focal_shift_cost(uv, xyz, n, s + h, fixed_focal)[0]
        var cm = _focal_shift_cost(uv, xyz, n, s - h, fixed_focal)[0]
        var g = (cp - cm) / (2 * h)
        var hh = (cp - 2 * cur[0] + cm) / (h * h)
        if hh <= 0:
            hh = abs(g) + 1e-12
        var step = -g / (hh * (1 + lam))
        var t = s + step
        var nxt = _focal_shift_cost(uv, xyz, n, t, fixed_focal)
        if nxt[0] < cur[0]:
            var rel = (cur[0] - nxt[0]) / max(cur[0], 1e-30)
            s = t
            cur = nxt
            lam = max(lam * 0.1, 1e-9)
            if rel < 1e-12:
                break
        else:
            lam *= 10
            if lam > 1e12:
                break
    return (s, cur[1])


def nearest_idx(o: Int, n_in: Int, n_out: Int) -> Int:
    """PyTorch `interpolate(mode="nearest")` source index."""
    var sc = Float32(n_in) / Float32(n_out)
    var i = Int(Float32(o) * sc)
    return min(i, n_in - 1)


def recover_focal_shift(points: List[Float32], mask: List[Bool], H: Int, W: Int, use_mask: Bool,
                        fixed_focal: Float64 = 0.0) -> Tuple[Float64, Float64]:
    """MoGe `recover_focal_shift` with a 64 x 64 nearest-neighbour subsample: (focal relative to
    the half diagonal, z shift)."""
    var uv = view_plane_uv(W, H, Float64(W) / Float64(H))
    var us = List[Float64]()
    var ps = List[Float64]()
    var n = 0
    for oy in range(64):
        var y = nearest_idx(oy, H, 64)
        for ox in range(64):
            var x = nearest_idx(ox, W, 64)
            var k = y * W + x
            if use_mask and not mask[k]:
                continue
            us.append(Float64(uv[2 * k]))
            us.append(Float64(uv[2 * k + 1]))
            ps.append(Float64(points[3 * k]))
            ps.append(Float64(points[3 * k + 1]))
            ps.append(Float64(points[3 * k + 2]))
            n += 1
    if n < 2:
        return (1.0, 0.0)
    var r = solve_focal_shift(us, ps, n, fixed_focal)
    return (Float64(Float32(r[1])), Float64(Float32(r[0])))


# ----------------------------------------------------------------------------- model

struct MogeRaw(Movable):
    """MoGe forward outputs at the input resolution (H x W, row-major)."""
    var points: List[Float32]     # [H, W, 3] affine point map (after the exp remap)
    var normal: List[Float32]     # [H, W, 3] unit normals (empty without a normal head)
    var mask: List[Float32]       # [H, W] sigmoid probabilities (empty without a mask head)
    var metric_scale: Float64     # exp(scale head) or 1 without one
    var has_scale: Bool
    var h: Int
    var w: Int

    def __init__(out self):
        self.points = List[Float32]()
        self.normal = List[Float32]()
        self.mask = List[Float32]()
        self.metric_scale = 1.0
        self.has_scale = False
        self.h = 0
        self.w = 0


struct MogeModel(Movable):
    var vit: DinoVit
    var proj: List[Linear]
    var neck: ConvStack
    var points: ConvStack
    var normal: ConvStack
    var mask: ConvStack
    var has_normal: Bool
    var has_mask: Bool
    var scale: List[Linear]
    var mean: List[Float32]
    var std: List[Float32]
    var tokens_min: Int
    var tokens_max: Int
    var remap: String

    def __init__(out self, var vit: DinoVit):
        self.vit = vit^
        self.proj = List[Linear]()
        self.neck = ConvStack()
        self.points = ConvStack()
        self.normal = ConvStack()
        self.mask = ConvStack()
        self.has_normal = False
        self.has_mask = False
        self.scale = List[Linear]()
        self.mean = List[Float32]()
        self.std = List[Float32]()
        self.tokens_min = 1200
        self.tokens_max = 3600
        self.remap = "exp"

    @staticmethod
    def load(mut rt: Runtime, w: Weights) raises -> MogeModel:
        var cfg = VitConfig()
        vit_size(w.cfg_str("encoder.backbone"), cfg)
        cfg.style = VIT_DINO
        cfg.prefix = "encoder.backbone."
        cfg.pos_mode = POS_OFFSET
        cfg.eps = 1e-6
        cfg.final_eps = 1e-6
        cfg.out_layers = w.cfg_ints("encoder.intermediate_layers")
        if len(cfg.out_layers) == 0:
            var n = w.cfg_int("encoder.intermediate_layers", 4)
            for i in range(n):
                cfg.out_layers.append(cfg.depth - n + i)
        var nl = len(cfg.out_layers)
        var vit = DinoVit.load(rt, w, cfg^)
        var m = MogeModel(vit^)
        for i in range(nl):
            m.proj.append(load_linear(rt, w, "encoder.output_projections." + String(i)))
        m.mean = w.floats("encoder.image_mean")
        m.std = w.floats("encoder.image_std")
        m.neck = load_convstack(rt, w, "neck", False)
        m.points = load_convstack(rt, w, "points_head", True)
        m.has_normal = w.cfg_has("normal_head")
        if m.has_normal:
            m.normal = load_convstack(rt, w, "normal_head", True)
        m.has_mask = w.cfg_has("mask_head")
        if m.has_mask:
            m.mask = load_convstack(rt, w, "mask_head", True)
        if w.cfg_has("scale_head"):
            var dims = w.cfg_ints("scale_head.dims")
            for i in range(len(dims) - 1):
                m.scale.append(load_linear(rt, w, "scale_head." + String(2 * i)))
        var tr = w.cfg_ints("num_tokens_range")
        if len(tr) == 2:
            m.tokens_min = tr[0]
            m.tokens_max = tr[1]
        var rm = w.cfg_str("remap_output")
        if rm != "":
            m.remap = rm
        if m.remap != "exp" and m.remap != "linear":
            raise Error("depth: unsupported MoGe remap_output " + m.remap)
        return m^

    def num_tokens(self, resolution_level: Int = 9) -> Int:
        return Int(Float64(self.tokens_min) + (Float64(resolution_level) / 9.0) * Float64(self.tokens_max - self.tokens_min))

    def forward(mut self, mut rt: Runtime, pixels: List[Float32], bh: Int, bw: Int, H: Int, W: Int) raises -> MogeRaw:
        """`pixels`: one normalised image [1, 3, 14 bh, 14 bw]; outputs are resized to H x W."""
        var P = self.vit.cfg.patch
        var feats = self.vit.forward(rt, pixels, 1, bh * P, bw * P)
        var C = self.vit.cfg.dim
        var N = bh * bw
        var cls = self.vit.raw_token0(rt, feats[len(feats) - 1])
        var x = Act()
        for i in range(len(feats)):
            var t = take_tokens(rt, feats[i], 1, N)
            t = rt.reshape(t^, 1, bh, bw, C)
            var p = rt.linear(t, self.proj[i])
            rt.free(t^)
            if i == 0:
                x = p^
            else:
                rt.add_(x, p)
                rt.free(p^)
        while len(feats) > 0:
            var f = feats.pop()
            rt.free(f^)
        var aspect = Float64(W) / Float64(H)
        var ins = List[Act]()
        for level in range(5):
            var lw = bw * (1 << level)
            var lh = bh * (1 << level)
            var uv = rt.upload(1, lh, lw, 2, view_plane_uv(lw, lh, aspect))
            if level == 0:
                ins.append(rt.concat_c(x, uv))
                rt.free(uv^)
            else:
                ins.append(uv^)
        rt.free(x^)
        var neck = run_convstack(rt, self.neck, ins, True)
        while len(ins) > 0:
            var t = ins.pop()
            rt.free(t^)
        var res = MogeRaw()
        res.h = H
        res.w = W
        res.points = _moge_head(rt, self.points, neck, H, W)
        if self.has_normal:
            res.normal = _moge_head(rt, self.normal, neck, H, W)
        if self.has_mask:
            res.mask = _moge_head(rt, self.mask, neck, H, W)
        while len(neck) > 0:
            var t = neck.pop()
            rt.free(t^)
        # remap / normalise / sigmoid
        var n = H * W
        if self.remap == "exp":
            for i in range(n):
                var z = exp(res.points[3 * i + 2])
                res.points[3 * i] *= z
                res.points[3 * i + 1] *= z
                res.points[3 * i + 2] = z
        if self.has_normal:
            for i in range(n):
                var a = res.normal[3 * i]
                var b = res.normal[3 * i + 1]
                var c = res.normal[3 * i + 2]
                var l = max(sqrt(a * a + b * b + c * c), Float32(1e-12))
                res.normal[3 * i] = a / l
                res.normal[3 * i + 1] = b / l
                res.normal[3 * i + 2] = c / l
        if self.has_mask:
            for i in range(n):
                res.mask[i] = Float32(1) / (Float32(1) + exp(-res.mask[i]))
        if len(self.scale) > 0:
            var s = rt.upload(1, 1, 1, C, cls)
            for i in range(len(self.scale)):
                var t = rt.linear(s, self.scale[i])
                rt.free(s^)
                if i < len(self.scale) - 1:
                    relu_(rt, t)
                s = t^
            var hv = rt.download(s)
            rt.free(s^)
            res.metric_scale = Float64(exp(hv[0]))
            res.has_scale = True
        return res^

def _moge_head(mut rt: Runtime, cs: ConvStack, neck: List[Act], H: Int, W: Int) raises -> List[Float32]:
        var outs = run_convstack(rt, cs, neck, False)
        var o = outs.pop()
        var r = resize_bilinear(rt, o, H, W, False)
        rt.free(o^)
        var h = rt.download(r)
        rt.free(r^)
        return h^


struct MogeGeometry(Movable):
    var points: List[Float32]   # [H, W, 3] metric (or relative) camera-space points; masked -> inf
    var depth: List[Float32]    # [H, W]; masked -> inf
    var normal: List[Float32]   # [H, W, 3]; masked -> 0
    var mask: List[Bool]
    var fx: Float64             # normalised intrinsics (cx = cy = 0.5)
    var fy: Float64
    var focal: Float64
    var shift: Float64
    var metric_scale: Float64

    def __init__(out self):
        self.points = List[Float32]()
        self.depth = List[Float32]()
        self.normal = List[Float32]()
        self.mask = List[Bool]()
        self.fx = 1.0
        self.fy = 1.0
        self.focal = 1.0
        self.shift = 0.0
        self.metric_scale = 1.0


def moge_geometry(raw: MogeRaw, apply_mask: Bool = True, force_projection: Bool = True,
                  fov_x_deg: Float64 = 0.0) raises -> MogeGeometry:
    """MoGe `infer` post-processing: focal / shift recovery, intrinsics, depth, re-projection,
    metric scale and masking."""
    var H = raw.h
    var W = raw.w
    var n = H * W
    var g = MogeGeometry()
    var has_mask = len(raw.mask) == n
    g.mask = List[Bool](length=n, fill=True)
    if has_mask:
        for i in range(n):
            g.mask[i] = raw.mask[i] > 0.5
    var ar = Float64(W) / Float64(H)
    var fixed = 0.0
    if fov_x_deg > 0:
        fixed = ar / sqrt(1 + ar * ar) / tan(fov_x_deg * 3.141592653589793 / 360.0)
    var fs = recover_focal_shift(raw.points, g.mask, H, W, has_mask, fixed)
    g.focal = fs[0] if fixed <= 0 else fixed
    g.shift = fs[1]
    g.fx = g.focal / 2 * sqrt(1 + ar * ar) / ar
    g.fy = g.focal / 2 * sqrt(1 + ar * ar)
    g.points = raw.points.copy()
    g.depth = List[Float32](length=n, fill=0.0)
    var sh = Float32(g.shift)
    for i in range(n):
        g.points[3 * i + 2] += sh
        if has_mask and g.points[3 * i + 2] <= 0:
            g.mask[i] = False
        g.depth[i] = g.points[3 * i + 2]
    if force_projection:
        var fx = Float32(g.fx)
        var fy = Float32(g.fy)
        var us = torch_linspace(Float32(0.5) / Float32(W), Float32(1) - Float32(0.5) / Float32(W), W)
        var vs = torch_linspace(Float32(0.5) / Float32(H), Float32(1) - Float32(0.5) / Float32(H), H)
        for y in range(H):
            for x in range(W):
                var i = y * W + x
                var d = g.depth[i]
                g.points[3 * i] = (us[x] - 0.5) / fx * d
                g.points[3 * i + 1] = (vs[y] - 0.5) / fy * d
                g.points[3 * i + 2] = d
    g.metric_scale = raw.metric_scale
    if raw.has_scale:
        var s = Float32(raw.metric_scale)
        for i in range(n):
            g.points[3 * i] *= s
            g.points[3 * i + 1] *= s
            g.points[3 * i + 2] *= s
            g.depth[i] *= s
    g.normal = raw.normal.copy()
    if apply_mask and has_mask:
        var inf = Float32.MAX * 2
        for i in range(n):
            if not g.mask[i]:
                g.points[3 * i] = inf
                g.points[3 * i + 1] = inf
                g.points[3 * i + 2] = inf
                g.depth[i] = inf
                if len(g.normal) == 3 * n:
                    g.normal[3 * i] = 0
                    g.normal[3 * i + 1] = 0
                    g.normal[3 * i + 2] = 0
    return g^
