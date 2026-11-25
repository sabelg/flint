"""Layer helpers for `flint.ml.depth` on top of `flint.ml.diffusion.nn.Runtime` (CPU or MAX GPU).

Weights come from a `Weights` checkpoint by name; DINOv2 LayerScale factors are folded into
the projection that precedes them at load time, and a stride == kernel `ConvTranspose2d`
becomes one GEMM ([Cin] -> [s * s * Cout]) plus a depth-to-space scatter. Every op below has a
CPU loop and a GPU kernel, so the same model code runs on either runtime.
"""

from std.math import sqrt, sin, cos, exp
from std.ffi import external_call
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.diffusion.nn import Runtime, Act, Linear, Conv, Norm, PAD_ZERO
from flint.ml.depth.weights import Weights
from flint.ml.depth.kernels import (
    _dp_relu_kernel, _dp_bilinear_kernel, _dp_d2s_kernel, _dp_pad_rep_kernel, _dp_rope_kernel, _dp_add_rows_kernel,
    bilinear_src, DP_EW,
)


struct ConvT(Copyable, Movable):
    """ConvTranspose2d with kernel == stride (no overlap): a GEMM into s * s * cout columns."""
    var lin: Linear
    var cout: Int
    var s: Int

    def __init__(out self):
        self.lin = Linear()
        self.cout = 0
        self.s = 1


# ----------------------------------------------------------------------------- loading

def load_linear(mut rt: Runtime, w: Weights, name: String, has_bias: Bool = True) raises -> Linear:
    var b = Tensor()
    if has_bias:
        b = w.tensor(name + ".bias")
    return rt.linear_from(w.tensor(name + ".weight"), b, has_bias)


def linear_rows(mut rt: Runtime, w: Weights, name: String, r0: Int, r1: Int, has_bias: Bool = True,
                gamma: List[Float32] = List[Float32]()) raises -> Linear:
    """Rows [r0, r1) of a torch Linear weight [n, k] (a q / k / v slice of a fused qkv), with
    the outputs optionally scaled by `gamma` (LayerScale folded in)."""
    var wt = w.tensor(name + ".weight")
    var k = wt.numel() // wt.shape[0]
    var n = r1 - r0
    var m = List[Float32](length=n * k, fill=0.0)
    _ = external_call["memcpy", Int](Int(m.unsafe_ptr()), wt.fptr() + r0 * k * 4, n * k * 4)
    _ = wt.numel()
    var bv = List[Float32](length=n, fill=0.0)
    if has_bias:
        var bt = w.tensor(name + ".bias")
        _ = external_call["memcpy", Int](Int(bv.unsafe_ptr()), bt.fptr() + r0 * 4, n * 4)
        _ = bt.numel()
    if len(gamma) == n:
        for r in range(n):
            for c in range(k):
                m[r * k + c] *= gamma[r]
            bv[r] *= gamma[r]
    var tw = Tensor.floats([n, k], m^)
    var tb = Tensor.floats([n], bv^)
    return rt.linear_from(tw, tb, has_bias)


def load_conv(mut rt: Runtime, w: Weights, name: String, stride: Int = 1, pad: Int = -1,
              has_bias: Bool = True) raises -> Conv:
    var wt = w.tensor(name + ".weight")
    var b = Tensor()
    if has_bias:
        b = w.tensor(name + ".bias")
    var p = pad
    if p < 0:
        p = wt.shape[2] // 2
    return rt.conv_from(wt, b, has_bias, stride, p, p, p, p)


def load_conv_t(mut rt: Runtime, w: Weights, name: String) raises -> ConvT:
    """torch ConvTranspose2d weight [cin, cout, s, s] (kernel == stride, padding 0)."""
    var wt = w.tensor(name + ".weight")
    var cin = wt.shape[0]
    var cout = wt.shape[1]
    var s = wt.shape[2]
    var bt = w.tensor(name + ".bias")
    var n = s * s * cout
    # torch-style [n, k] weight for linear_from: row (dy * s + dx) * cout + co, column ci
    var m = List[Float32](length=n * cin, fill=0.0)
    var p = fptr_at(wt.fptr())
    for ci in range(cin):
        for co in range(cout):
            for t in range(s * s):
                m[(t * cout + co) * cin + ci] = p.unsafe_offset((ci * cout + co) * s * s + t).unsafe_load[alignment=1]()
    _ = wt.numel()  # keep a converted (F16 / BF16) tensor alive past the raw reads
    var bv = List[Float32](length=n, fill=0.0)
    var pb = fptr_at(bt.fptr())
    for t in range(s * s):
        for co in range(cout):
            bv[t * cout + co] = pb.unsafe_offset(co).unsafe_load[alignment=1]()
    _ = bt.numel()
    var ct = ConvT()
    ct.lin = rt.linear_from(Tensor.floats([n, cin], m^), Tensor.floats([n], bv^), True)
    ct.cout = cout
    ct.s = s
    return ct^


def load_norm(mut rt: Runtime, w: Weights, name: String, groups: Int = 1, eps: Float32 = 1e-5) raises -> Norm:
    return rt.norm_from(w.floats(name + ".weight"), w.floats(name + ".bias"), groups, eps)


# ----------------------------------------------------------------------------- ops

def _grid(n: Int) -> Int:
    return (n + DP_EW - 1) // DP_EW


def relu_(mut rt: Runtime, mut x: Act) raises:
    var n = x.numel()
    if rt.on_gpu():
        rt.gpu.value().ctx.enqueue_function[_dp_relu_kernel](rt.ptr(x), Int32(n), grid_dim=_grid(n), block_dim=DP_EW)
        return
    for i in range(n):
        if x.host[i] < 0:
            x.host[i] = 0


def relu(mut rt: Runtime, x: Act) raises -> Act:
    var o = rt.copy(x)
    relu_(rt, o)
    return o^


def resize_bilinear(mut rt: Runtime, x: Act, ho: Int, wo: Int, align: Bool) raises -> Act:
    """PyTorch `interpolate(mode="bilinear", antialias=False)` of [b, h, w, c] to [b, ho, wo, c]."""
    var o = rt.alloc(x.b, ho, wo, x.c)
    var n = o.numel()
    if rt.on_gpu():
        rt.gpu.value().ctx.enqueue_function[_dp_bilinear_kernel](
            rt.ptr(x), rt.ptr(o), Int32(x.b), Int32(x.h), Int32(x.w), Int32(x.c), Int32(ho), Int32(wo),
            Int32(1 if align else 0), grid_dim=_grid(n), block_dim=DP_EW)
        return o^
    var C = x.c
    for b in range(x.b):
        for oy in range(ho):
            var yv = bilinear_src(oy, x.h, ho, align)
            for ox in range(wo):
                var xv = bilinear_src(ox, x.w, wo, align)
                var base = b * x.h * x.w
                var d = ((b * ho + oy) * wo + ox) * C
                var ly = yv[2]
                var lx = xv[2]
                for c in range(C):
                    var v00 = x.host[(base + yv[0] * x.w + xv[0]) * C + c]
                    var v01 = x.host[(base + yv[0] * x.w + xv[1]) * C + c]
                    var v10 = x.host[(base + yv[1] * x.w + xv[0]) * C + c]
                    var v11 = x.host[(base + yv[1] * x.w + xv[1]) * C + c]
                    o.host[d + c] = (1 - ly) * ((1 - lx) * v00 + lx * v01) + ly * ((1 - lx) * v10 + lx * v11)
    return o^


def conv_t(mut rt: Runtime, x: Act, ct: ConvT) raises -> Act:
    var y = rt.linear(x, ct.lin)
    var s = ct.s
    var C = ct.cout
    var o = rt.alloc(x.b, x.h * s, x.w * s, C)
    var n = o.numel()
    if rt.on_gpu():
        rt.gpu.value().ctx.enqueue_function[_dp_d2s_kernel](
            rt.ptr(y), rt.ptr(o), Int32(x.b), Int32(x.h), Int32(x.w), Int32(C), Int32(s), grid_dim=_grid(n),
            block_dim=DP_EW)
    else:
        var wo = x.w * s
        var ho = x.h * s
        for b in range(x.b):
            for oy in range(ho):
                for ox in range(wo):
                    var t = (oy % s) * s + ox % s
                    var src = ((b * x.h + oy // s) * x.w + ox // s) * s * s * C + t * C
                    var dst = ((b * ho + oy) * wo + ox) * C
                    for c in range(C):
                        o.host[dst + c] = y.host[src + c]
    rt.free(y^)
    return o^


def pad_replicate(mut rt: Runtime, x: Act, p: Int) raises -> Act:
    var o = rt.alloc(x.b, x.h + 2 * p, x.w + 2 * p, x.c)
    var n = o.numel()
    if rt.on_gpu():
        rt.gpu.value().ctx.enqueue_function[_dp_pad_rep_kernel](
            rt.ptr(x), rt.ptr(o), Int32(x.b), Int32(x.h), Int32(x.w), Int32(x.c), Int32(p), grid_dim=_grid(n),
            block_dim=DP_EW)
        return o^
    var C = x.c
    var ho = x.h + 2 * p
    var wo = x.w + 2 * p
    for b in range(x.b):
        for oy in range(ho):
            var y = min(max(oy - p, 0), x.h - 1)
            for ox in range(wo):
                var xx = min(max(ox - p, 0), x.w - 1)
                var s = ((b * x.h + y) * x.w + xx) * C
                var d = ((b * ho + oy) * wo + ox) * C
                for c in range(C):
                    o.host[d + c] = x.host[s + c]
    return o^


def conv_rep(mut rt: Runtime, x: Act, cv: Conv) raises -> Act:
    """A 3x3 (or k x k, stride 1) convolution with replicate padding (`cv` loaded with pad 0)."""
    var p = cv.kh // 2
    var xp = pad_replicate(rt, x, p)
    var o = rt.conv2d(xp, cv, PAD_ZERO)
    rt.free(xp^)
    return o^


def rope_(mut rt: Runtime, mut x: Act, pos: Act, freq: Act, heads: Int) raises:
    """2-D rotary embedding in place on token rows x [rows, heads * hd]; pos [rows, 2] (row, col)
    positions as floats, freq [hd / 4] inverse frequencies."""
    var rows = x.rows()
    var C = x.c
    var hd = C // heads
    var Q = hd // 4
    var n = rows * heads * 2 * Q
    if rt.on_gpu():
        rt.gpu.value().ctx.enqueue_function[_dp_rope_kernel](
            rt.ptr(x), rt.ptr(pos), rt.ptr(freq), Int32(rows), Int32(heads), Int32(hd), Int32(C), grid_dim=_grid(n),
            block_dim=DP_EW)
        return
    var F = hd // 2
    for r in range(rows):
        for h in range(heads):
            for axis in range(2):
                var p = pos.host[2 * r + axis]
                var base = r * C + h * hd + axis * F
                for i in range(Q):
                    var a = p * freq.host[i]
                    var cs = cos(a)
                    var sn = sin(a)
                    var x1 = x.host[base + i]
                    var x2 = x.host[base + Q + i]
                    x.host[base + i] = x1 * cs - x2 * sn
                    x.host[base + Q + i] = x2 * cs + x1 * sn


def copy_rows(mut rt: Runtime, src: Act, mut dst: Act, outer: Int, chunk: Int, s_stride: Int, s_off: Int,
              d_stride: Int, d_off: Int) raises:
    """dst[o * d_stride + d_off + c] = src[o * s_stride + s_off + c] (all in floats)."""
    if rt.on_gpu():
        rt.gpu.value().copy_dev(rt.ptr(src), rt.ptr(dst), outer, chunk, s_stride, s_off, d_stride, d_off)
        return
    for o in range(outer):
        for c in range(chunk):
            dst.host[o * d_stride + d_off + c] = src.host[o * s_stride + s_off + c]


def add_rows(mut rt: Runtime, mut dst: Act, src: Act, outer: Int, chunk: Int, d_stride: Int, d_off: Int,
             s_stride: Int) raises:
    """dst[o * d_stride + d_off + c] += src[o * s_stride + c]."""
    var total = outer * chunk
    if total == 0:
        return
    if rt.on_gpu():
        rt.gpu.value().ctx.enqueue_function[_dp_add_rows_kernel](
            rt.ptr(dst), rt.ptr(src), Int32(total), Int32(chunk), Int32(d_stride), Int32(d_off), Int32(s_stride),
            grid_dim=_grid(total), block_dim=DP_EW)
        return
    for o in range(outer):
        for c in range(chunk):
            dst.host[o * d_stride + d_off + c] += src.host[o * s_stride + c]


def take_tokens(mut rt: Runtime, x: Act, start: Int, count: Int) raises -> Act:
    """Rows [start, start + count) of every image of a token value [b, T, 1, c]."""
    var T = x.h * x.w
    var o = rt.alloc(x.b, count, 1, x.c)
    copy_rows(rt, x, o, x.b, count * x.c, T * x.c, start * x.c, count * x.c, 0)
    return o^


def as_grid(mut rt: Runtime, var x: Act, h: Int, w: Int) raises -> Act:
    return rt.reshape(x^, x.b, h, w, x.c)


def conv_out(mut rt: Runtime, x: Act, cv: Conv) raises -> Act:
    return rt.conv2d(x, cv, PAD_ZERO)


# ----------------------------------------------------------------------------- positional grids

def uv_embed(w: Int, h: Int, C: Int, aspect: Float64, ratio: Float32 = 0.1) -> List[Float32]:
    """Depth Anything 3 `create_uv_grid` + `position_grid_to_embed` (omega_0 = 100) * ratio as a
    channels-last [h, w, C] table: [sin(u w), cos(u w), sin(v w), cos(v w)] with C / 4 each."""
    var diag = sqrt(aspect * aspect + 1.0)
    var span_x = aspect / diag
    var span_y = 1.0 / diag
    var lx = Float32(-span_x * Float64(w - 1) / Float64(w))
    var rx = Float32(span_x * Float64(w - 1) / Float64(w))
    var ty = Float32(-span_y * Float64(h - 1) / Float64(h))
    var by = Float32(span_y * Float64(h - 1) / Float64(h))
    var us = torch_linspace(lx, rx, w)
    var vs = torch_linspace(ty, by, h)
    var D = C // 2
    var Q = D // 2
    var om = List[Float32](length=Q, fill=0.0)
    for i in range(Q):
        var e = Float32(i) / Float32(Q)
        om[i] = Float32(1.0) / Float32(Float64(100.0) ** Float64(e))
    var out = List[Float32](length=h * w * C, fill=0.0)
    for y in range(h):
        for x in range(w):
            var b = (y * w + x) * C
            for i in range(Q):
                var au = us[x] * om[i]
                var av = vs[y] * om[i]
                out[b + i] = sin(au) * ratio
                out[b + Q + i] = cos(au) * ratio
                out[b + D + i] = sin(av) * ratio
                out[b + D + Q + i] = cos(av) * ratio
    return out^


def torch_linspace(a: Float32, b: Float32, n: Int) -> List[Float32]:
    """`torch.linspace` in float32 (CPU kernel: start + i * step for the first half, end - (n - 1 - i)
    * step for the second)."""
    var out = List[Float32](length=n, fill=0.0)
    if n == 1:
        out[0] = a
        return out^
    var step = (b - a) / Float32(n - 1)
    var half = n // 2
    for i in range(n):
        if i < half:
            out[i] = a + step * Float32(i)
        else:
            out[i] = b - step * Float32(n - 1 - i)
    return out^


def upload_rep(mut rt: Runtime, data: List[Float32], times: Int, h: Int, w: Int, c: Int) raises -> Act:
    """`times` copies of a [h, w, c] host table as one [times, h, w, c] value."""
    var all = List[Float32](capacity=len(data) * times)
    for _ in range(times):
        all.extend(data.copy())
    return rt.upload(times, h, w, c, all)
