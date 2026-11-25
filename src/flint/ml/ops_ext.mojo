"""More ONNX operator kernels: recurrent, transposed-convolution and signal ops.

  - `reduce_mean`        ReduceMean (contiguous axis blocks on a fast path)
  - `lstm`               LSTM, forward / reverse / bidirectional, gates i-o-f-c, optional h0/c0
  - `conv1d_ext`         1-D Conv with any stride and group count
  - `conv_transpose1d`   1-D ConvTranspose with stride, pads, dilation, output_padding, groups
  - `pad`                Pad in constant / reflect / edge mode (negative pads crop)
  - `round_even`         Round, halves to even
  - `stft`               STFT of a real signal (onesided or full), with a window
  - `nonzero`            NonZero
  - `resize_linear`      Resize mode=linear (n-linear, separable) for every coordinate mode
  - `conv1d_par`, `par_row_binary`, `par_sin_cos_square`: pool-parallel Conv (im2col and output
    transpose spread over the workers), per-row-scalar broadcasts ([.., C, 1] against [.., C, T]),
    Sin / Cos / Pow(x, 2)

Products go through `ops.matmul_2d` on the `JobPool`; the rest are plain loops.
"""

from std.math import floor, sqrt, exp, tanh, cos, sin, pi
from flint.core.jobs import JobPool
from flint.ml.tensor import Tensor, DT_F32, DT_I64, numel_of, strides_of, shape_str, norm_axis, fptr_at
from flint.core.ecs import run_parallel_ctx
from flint.ml.gpu import GpuMatmul
from flint.ml.ops import matmul_2d, reduce, U_SIN, U_COS, B_ADD, B_SUB, B_MUL, B_DIV, B_MAX, B_MIN


# ------------------------------------------------------------------------------- ReduceMean

def reduce_mean(x: Tensor, axes_in: List[Int], keepdims: Bool) raises -> Tensor:
    var r = x.rank()
    var red = List[Bool](length=r, fill=False)
    if len(axes_in) == 0:
        for d in range(r):
            red[d] = True
    for a in axes_in:
        red[norm_axis(a, r)] = True
    var out_shape = List[Int]()
    var count = 1
    for d in range(r):
        if red[d]:
            count *= x.shape[d]
            if keepdims:
                out_shape.append(1)
        else:
            out_shape.append(x.shape[d])
    # fast path: the reduced axes form one contiguous block [a0, a1)
    var a0 = -1
    var a1 = -1
    var contiguous = True
    for d in range(r):
        if red[d]:
            if a0 < 0:
                a0 = d
            elif a1 >= 0:
                contiguous = False
        elif a0 >= 0 and a1 < 0:
            a1 = d
    if a1 < 0:
        a1 = r
    if x.is_float() and contiguous and a0 >= 0:
        var outer = 1
        for d in range(a0):
            outer *= x.shape[d]
        var inner = 1
        for d in range(a1, r):
            inner *= x.shape[d]
        var o = List[Float32](length=outer * inner, fill=0.0)
        var p = fptr_at(x.fptr())
        var inv = 1.0 / Float64(count) if count > 0 else 0.0
        if inner == 1:
            for a in range(outer):
                var s = 0.0
                var base = a * count
                for j in range(count):
                    s += Float64(p.unsafe_offset(base + j).unsafe_load[alignment=1]())
                o[a] = Float32(s * inv)
        else:
            var acc = List[Float64](length=inner, fill=0.0)
            for a in range(outer):
                for c in range(inner):
                    acc[c] = 0.0
                for j in range(count):
                    var base = (a * count + j) * inner
                    for c in range(inner):
                        acc[c] += Float64(p.unsafe_offset(base + c).unsafe_load[alignment=1]())
                for c in range(inner):
                    o[a * inner + c] = Float32(acc[c] * inv)
        return Tensor.floats(out_shape^, o^)
    var s = reduce(x, axes_in, keepdims, False)
    if s.is_float():
        for e in range(len(s.f)):
            s.f[e] = Float32(Float64(s.f[e]) / Float64(count))
    else:
        for e in range(len(s.i)):
            s.i[e] = s.i[e] // count
    return s^


# ------------------------------------------------------------------------------------- LSTM

@always_inline
def _sigm(v: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-v))


def lstm(mut pool: JobPool, x: Tensor, w: Tensor, r: Tensor, b: Tensor, has_b: Bool, h0: Tensor, has_h0: Bool,
         c0: Tensor, has_c0: Bool, hidden: Int, direction: String, layout: Int) raises -> List[Tensor]:
    """ONNX LSTM (no peepholes, default activations). x [T, B, I] (layout 0) or [B, T, I]
    (layout 1); returns Y [T, D, B, H] (or [B, T, D, H]), Y_h and Y_c [D, B, H]."""
    if x.rank() != 3:
        raise Error("LSTM: input must be rank 3")
    var T = x.shape[0] if layout == 0 else x.shape[1]
    var B = x.shape[1] if layout == 0 else x.shape[0]
    var I = x.shape[2]
    var H = hidden
    var D = w.shape[0]
    if w.shape[1] != 4 * H or w.shape[2] != I or r.shape[2] != H:
        raise Error("LSTM: weight shapes " + shape_str(w.shape) + " " + shape_str(r.shape))
    var H4 = 4 * H
    # inputs as [T*B, I] rows in time-major order
    var xs = List[Float32](length=T * B * I, fill=0.0)
    for t in range(T):
        for bb in range(B):
            var src = (t * B + bb) * I if layout == 0 else (bb * T + t) * I
            for c in range(I):
                xs[(t * B + bb) * I + c] = x.fget(src + c)
    var y = List[Float32](length=T * D * B * H, fill=0.0)
    var yh = List[Float32](length=D * B * H, fill=0.0)
    var yc = List[Float32](length=D * B * H, fill=0.0)
    var g = List[Float32](length=T * B * H4, fill=0.0)
    var bias = List[Float32](length=H4, fill=0.0)
    var h = List[Float32](length=B * H, fill=0.0)
    var cc = List[Float32](length=B * H, fill=0.0)
    var gt = List[Float32](length=H4, fill=0.0)
    var pr = fptr_at(r.fptr())
    for d in range(D):
        var reverse = direction == "reverse" or (direction == "bidirectional" and d == 1)
        for j in range(H4):
            bias[j] = 0
            if has_b:
                bias[j] = b.fget(d * 8 * H + j) + b.fget(d * 8 * H + H4 + j)
        # G = X W_d^T + (Wb + Rb)
        matmul_2d(pool, Int(xs.unsafe_ptr()), w.fptr() + d * H4 * I * 4, Int(g.unsafe_ptr()), Int(bias.unsafe_ptr()),
                  T * B, I, H4, True)
        for e in range(B * H):
            h[e] = h0.fget(d * B * H + e) if has_h0 else 0
            cc[e] = c0.fget(d * B * H + e) if has_c0 else 0
        var rbase = d * H4 * H
        for step in range(T):
            var t = T - 1 - step if reverse else step
            for bb in range(B):
                var hb = bb * H
                for j in range(H4):
                    var row = pr.unsafe_offset(rbase + j * H)
                    var acc = SIMD[DType.float32, 8](0)
                    var q = 0
                    while q + 8 <= H:
                        acc += row.unsafe_offset(q).unsafe_load[width=8, alignment=1]() * \
                            h.unsafe_ptr().unsafe_offset(hb + q).unsafe_load[width=8, alignment=1]()
                        q += 8
                    var s = acc.reduce_add()
                    while q < H:
                        s += row.unsafe_offset(q).unsafe_load[alignment=1]() * h[hb + q]
                        q += 1
                    gt[j] = g[(t * B + bb) * H4 + j] + s
                for j in range(H):
                    var ig = _sigm(gt[j])
                    var og = _sigm(gt[H + j])
                    var fg = _sigm(gt[2 * H + j])
                    var cg = tanh(gt[3 * H + j])
                    var cn = fg * cc[hb + j] + ig * cg
                    cc[hb + j] = cn
                    h[hb + j] = og * tanh(cn)
                for j in range(H):
                    var at = ((t * D + d) * B + bb) * H + j if layout == 0 else ((bb * T + t) * D + d) * H + j
                    y[at] = h[hb + j]
        for e in range(B * H):
            yh[d * B * H + e] = h[e]
            yc[d * B * H + e] = cc[e]
    var out = List[Tensor]()
    if layout == 0:
        out.append(Tensor.floats([T, D, B, H], y^))
    else:
        out.append(Tensor.floats([B, T, D, H], y^))
    out.append(Tensor.floats([D, B, H], yh^))
    out.append(Tensor.floats([D, B, H], yc^))
    return out^


# ------------------------------------------------------------------------------ convolutions

def conv1d_ext(mut pool: JobPool, x: Tensor, w: Tensor, b: Tensor, has_b: Bool, pad0: Int, pad1: Int, dil: Int,
               stride: Int, group: Int) raises -> Tensor:
    """1-D Conv, any stride / group count: per group, im2col^T [t, cin_g*kw] @ W_g^T."""
    if x.rank() != 3 or w.rank() != 3:
        raise Error("Conv: only 1-D convolution is supported")
    var batch = x.shape[0]
    var cin = x.shape[1]
    var t_in = x.shape[2]
    var cout = w.shape[0]
    var cin_g = w.shape[1]
    var kw = w.shape[2]
    if cin_g * group != cin or cout % group != 0:
        raise Error("Conv: channels do not match the group count")
    var cout_g = cout // group
    var t_out = (t_in + pad0 + pad1 - dil * (kw - 1) - 1) // stride + 1
    var j = cin_g * kw
    var out = Tensor.zeros([batch, cout, t_out])
    var col = List[Float32](length=t_out * j, fill=0.0)
    var tmp = List[Float32](length=t_out * cout_g, fill=0.0)
    for bi in range(batch):
        for gi in range(group):
            for t in range(t_out):
                for ci in range(cin_g):
                    var xb = (bi * cin + gi * cin_g + ci) * t_in
                    for kk in range(kw):
                        var src = t * stride + kk * dil - pad0
                        col[t * j + ci * kw + kk] = x.fget(xb + src) if src >= 0 and src < t_in else 0
            matmul_2d(pool, Int(col.unsafe_ptr()), w.fptr() + gi * cout_g * j * 4, Int(tmp.unsafe_ptr()),
                      b.fptr() + gi * cout_g * 4 if has_b else 0, t_out, j, cout_g, True)
            for co in range(cout_g):
                var ob = (bi * cout + gi * cout_g + co) * t_out
                for t in range(t_out):
                    out.f[ob + t] = tmp[t * cout_g + co]
    return out^


def conv_transpose1d(mut pool: JobPool, x: Tensor, w: Tensor, b: Tensor, has_b: Bool, pad0: Int, pad1: Int,
                     dil: Int, stride: Int, out_pad: Int, group: Int) raises -> Tensor:
    """1-D ConvTranspose. x [B, Cin, T], w [Cin, Cout/group, K]."""
    if x.rank() != 3 or w.rank() != 3:
        raise Error("ConvTranspose: only 1-D is supported")
    var batch = x.shape[0]
    var cin = x.shape[1]
    var t_in = x.shape[2]
    var cout_g = w.shape[1]
    var kw = w.shape[2]
    if w.shape[0] != cin or cin % group != 0:
        raise Error("ConvTranspose: weight " + shape_str(w.shape) + " vs input " + shape_str(x.shape))
    var cin_g = cin // group
    var cout = cout_g * group
    var t_out = (t_in - 1) * stride + dil * (kw - 1) + 1 + out_pad - pad0 - pad1
    var out = Tensor.zeros([batch, cout, t_out])
    var pw = fptr_at(w.fptr())
    var n = cout_g * kw
    var xt = List[Float32]()
    var col = List[Float32]()
    if cin_g >= 8:
        xt = List[Float32](length=t_in * cin_g, fill=0.0)
        col = List[Float32](length=t_in * n, fill=0.0)
    for bi in range(batch):
        for gi in range(group):
            if cin_g >= 8:
                # col[t, co*K + k] = sum_ci x[ci, t] w[ci, co, k]
                for ci in range(cin_g):
                    var xb = (bi * cin + gi * cin_g + ci) * t_in
                    for t in range(t_in):
                        xt[t * cin_g + ci] = x.fget(xb + t)
                matmul_2d(pool, Int(xt.unsafe_ptr()), w.fptr() + gi * cin_g * n * 4, Int(col.unsafe_ptr()), 0,
                          t_in, cin_g, n, False)
                for co in range(cout_g):
                    var ob = (bi * cout + gi * cout_g + co) * t_out
                    for t in range(t_in):
                        for kk in range(kw):
                            var o = t * stride + kk * dil - pad0
                            if o >= 0 and o < t_out:
                                out.f[ob + o] += col[t * n + co * kw + kk]
            else:
                for ci in range(cin_g):
                    var c_abs = gi * cin_g + ci
                    var xb = (bi * cin + c_abs) * t_in
                    for co in range(cout_g):
                        var ob = (bi * cout + gi * cout_g + co) * t_out
                        var wb = (c_abs * cout_g + co) * kw
                        for t in range(t_in):
                            var xv = x.fget(xb + t)
                            for kk in range(kw):
                                var o = t * stride + kk * dil - pad0
                                if o >= 0 and o < t_out:
                                    out.f[ob + o] += xv * pw.unsafe_offset(wb + kk).unsafe_load[alignment=1]()
    if has_b:
        for bi in range(batch):
            for co in range(cout):
                var bv = b.fget(co)
                var ob = (bi * cout + co) * t_out
                for t in range(t_out):
                    out.f[ob + t] += bv
    return out^


# ------------------------------------------------------------------------------------- Pad

def pad(x: Tensor, pads_in: List[Int], axes_in: List[Int], mode: String, value: Float64) raises -> Tensor:
    """Pad with ONNX `pads` = [x1_begin, x2_begin, ..., x1_end, x2_end, ...] over `axes`
    (all axes when empty); negative pads crop."""
    var r = x.rank()
    var lo = List[Int](length=r, fill=0)
    var hi = List[Int](length=r, fill=0)
    var na = len(axes_in) if len(axes_in) > 0 else r
    if len(pads_in) != 2 * na:
        raise Error("Pad: " + String(len(pads_in)) + " pads for " + String(na) + " axes")
    for j in range(na):
        var ax = norm_axis(axes_in[j], r) if len(axes_in) > 0 else j
        lo[ax] = pads_in[j]
        hi[ax] = pads_in[na + j]
    var out_shape = List[Int](length=r, fill=0)
    for d in range(r):
        out_shape[d] = x.shape[d] + lo[d] + hi[d]
        if out_shape[d] < 0:
            raise Error("Pad: negative output size")
    var out = Tensor.like(x, out_shape.copy())
    var sst = strides_of(x.shape)
    var ctr = List[Int](length=r, fill=0)
    var reflect = mode == "reflect"
    var edge = mode == "edge"
    if not reflect and not edge and mode != "constant" and mode.byte_length() > 0:
        raise Error("Pad: mode " + mode + " is not supported")
    for e in range(out.numel()):
        var off = 0
        var inside = True
        for d in range(r):
            var n = x.shape[d]
            var i = ctr[d] - lo[d]
            if i < 0 or i >= n:
                if reflect and n > 1:
                    var per = 2 * (n - 1)
                    i = i % per
                    if i < 0:
                        i += per
                    if i >= n:
                        i = per - i
                elif edge or (reflect and n == 1):
                    i = 0 if i < 0 else n - 1
                else:
                    inside = False
            off += i * sst[d]
        if out.is_float():
            out.f[e] = x.fget(off) if inside else Float32(value)
        else:
            out.i[e] = x.i[off] if inside else Int(value)
        var d = r - 1
        while d >= 0:
            ctr[d] += 1
            if ctr[d] < out_shape[d]:
                break
            ctr[d] = 0
            d -= 1
    return out^


# ----------------------------------------------------------------------------------- Round

def round_even(x: Tensor) -> Tensor:
    var out = Tensor.like(x, x.shape.copy())
    if not x.is_float():
        for e in range(x.numel()):
            out.i[e] = x.i[e]
        return out^
    for e in range(x.numel()):
        var v = x.fget(e)
        var f = floor(v)
        var d = v - f
        var rr = f
        if d > 0.5:
            rr = f + 1
        elif d == 0.5:
            var half = f * 0.5
            rr = f if floor(half) == half else f + 1
        out.f[e] = rr
    return out^


# ------------------------------------------------------------------------------------ STFT

def stft(signal: Tensor, step: Int, window: Tensor, has_window: Bool, frame_len_in: Int, onesided: Bool) raises -> Tensor:
    """STFT of a real signal [B, N] or [B, N, 1] -> [B, frames, bins, 2] (re, im)."""
    if signal.rank() == 3 and signal.shape[2] != 1:
        raise Error("STFT: only real signals are supported")
    var batch = signal.shape[0]
    var n = signal.shape[1]
    var L = frame_len_in if frame_len_in > 0 else window.numel()
    if L <= 0:
        raise Error("STFT: needs a window or a frame length")
    var frames = (n - L) // step + 1
    var bins = L // 2 + 1 if onesided else L
    var out = Tensor.zeros([batch, frames, bins, 2])
    var cw = List[Float64](length=bins * L, fill=0.0)
    var sw = List[Float64](length=bins * L, fill=0.0)
    for kk in range(bins):
        for t in range(L):
            var m = kk * t % L
            if (4 * m) % L == 0:
                # quarter turns exactly, so the DC / Nyquist imaginary parts are exactly 0
                var qd = (4 * m) // L
                cw[kk * L + t] = 1.0 if qd == 0 else (-1.0 if qd == 2 else 0.0)
                sw[kk * L + t] = -1.0 if qd == 1 else (1.0 if qd == 3 else 0.0)
            else:
                var ang = -2.0 * pi * Float64(m) / Float64(L)
                cw[kk * L + t] = cos(ang)
                sw[kk * L + t] = sin(ang)
    var fr = List[Float64](length=L, fill=0.0)
    for bi in range(batch):
        for f in range(frames):
            for t in range(L):
                var wv = Float64(window.fget(t)) if has_window else 1.0
                fr[t] = Float64(signal.fget(bi * n + f * step + t)) * wv
            for kk in range(bins):
                var re = 0.0
                var im = 0.0
                for t in range(L):
                    re += fr[t] * cw[kk * L + t]
                    im += fr[t] * sw[kk * L + t]
                var o = ((bi * frames + f) * bins + kk) * 2
                out.f[o] = Float32(re)
                out.f[o + 1] = Float32(im)
    return out^


# --------------------------------------------------------------------------------- NonZero

def nonzero(x: Tensor) -> Tensor:
    var r = x.rank()
    var hits = List[Int]()
    for e in range(x.numel()):
        if (x.fget(e) != 0) if x.is_float() else (x.i[e] != 0):
            hits.append(e)
    var nz = len(hits)
    var o = List[Int](length=max(r, 1) * nz, fill=0)
    var st = strides_of(x.shape)
    for j in range(nz):
        var rem = hits[j]
        for d in range(r):
            o[d * nz + j] = rem // st[d]
            rem = rem % st[d]
    return Tensor.ints(DT_I64, [max(r, 1) if r > 0 else 1, nz], o^)


# ----------------------------------------------------------------------------- Resize linear

def _src_coord(mode: String, xo: Int, scale: Float64, n_in: Int, n_out: Int) -> Float64:
    var x = Float64(xo)
    if mode == "align_corners":
        return 0.0 if n_out <= 1 else x * Float64(n_in - 1) / Float64(n_out - 1)
    if mode == "asymmetric":
        return x / scale
    if mode == "pytorch_half_pixel":
        return (x + 0.5) / scale - 0.5 if n_out > 1 else 0.0
    if mode == "half_pixel_symmetric":
        var adj = Float64(n_out) / (scale * Float64(n_in))
        var center = Float64(n_in) * 0.5
        var off = center * (1.0 - adj)
        return off + (x + 0.5) / scale - 0.5
    # half_pixel (default)
    return (x + 0.5) / scale - 0.5


def resize_linear(x: Tensor, out_shape: List[Int], scales: List[Float64], mode: String) raises -> Tensor:
    """Separable n-linear Resize; `scales[d]` (0 = derive from the sizes) per axis."""
    if not x.is_float():
        raise Error("Resize linear: float input only")
    var cur = Tensor.floats(x.shape.copy(), x.owned_floats())
    for d in range(x.rank()):
        var n_in = cur.shape[d]
        var n_out = out_shape[d]
        if n_in == n_out and (scales[d] == 0 or scales[d] == 1):
            continue
        var sc = scales[d] if scales[d] > 0 else Float64(n_out) / Float64(n_in)
        var ns = cur.shape.copy()
        ns[d] = n_out
        var nxt = Tensor.zeros(ns.copy())
        var outer = 1
        for q in range(d):
            outer *= cur.shape[q]
        var inner = 1
        for q in range(d + 1, cur.rank()):
            inner *= cur.shape[q]
        for i in range(n_out):
            var s = _src_coord(mode, i, sc, n_in, n_out)
            s = min(max(s, 0.0), Float64(n_in - 1))
            var i0 = Int(floor(s))
            var i1 = min(i0 + 1, n_in - 1)
            var lam = Float32(s - Float64(i0))
            for a in range(outer):
                var b0 = (a * n_in + i0) * inner
                var b1 = (a * n_in + i1) * inner
                var bo = (a * n_out + i) * inner
                for c in range(inner):
                    nxt.f[bo + c] = cur.f[b0 + c] * (1 - lam) + cur.f[b1 + c] * lam
        cur = nxt^
    return cur^


# ------------------------------------------------------------------ parallel element-wise helpers

comptime _CHUNK: Int = 1 << 14
comptime PAR_MIN: Int = 1 << 14


struct _PwCtx(Movable):
    var a: Int
    var b: Int
    var o: Int
    var n: Int
    var op: Int
    var inner: Int
    var mode: Int     # 0: op(a[e], b[e / inner]); 1: op(b[e / inner], a[e]); 2: unary on a

    def __init__(out self):
        self.a = 0
        self.b = 0
        self.o = 0
        self.n = 0
        self.op = 0
        self.inner = 1
        self.mode = 0


@always_inline
def _bop(op: Int, x: SIMD[DType.float32, 8], y: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    if op == B_ADD:
        return x + y
    if op == B_SUB:
        return x - y
    if op == B_MUL:
        return x * y
    if op == B_DIV:
        return x / y
    if op == B_MAX:
        return max(x, y)
    return min(x, y)


def _pw_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_PwCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var e0 = job * _CHUNK
    var e1 = min(c[].n, e0 + _CHUNK)
    var pa = fptr_at(c[].a)
    var pb = fptr_at(c[].b)
    var po = fptr_at(c[].o)
    var op = c[].op
    if c[].mode == 2:
        for e in range(e0, e1):
            var v = pa.unsafe_offset(e).unsafe_load[alignment=1]()
            var r: Float32
            if op == U_SIN:
                r = sin(v)
            elif op == U_COS:
                r = cos(v)
            else:
                r = v * v
            po.unsafe_offset(e).unsafe_store[alignment=1](r)
        return
    var inner = c[].inner
    var swapped = c[].mode == 1
    var e = e0
    while e < e1:
        var row = e // inner
        var run = min(inner - e % inner, e1 - e)
        var s = SIMD[DType.float32, 8](pb.unsafe_offset(row).unsafe_load[alignment=1]())
        var k = 0
        while k + 8 <= run:
            var x = pa.unsafe_offset(e + k).unsafe_load[width=8, alignment=1]()
            po.unsafe_offset(e + k).unsafe_store[alignment=1](_bop(op, s, x) if swapped else _bop(op, x, s))
            k += 8
        while k < run:
            var x = SIMD[DType.float32, 8](pa.unsafe_offset(e + k).unsafe_load[alignment=1]())
            var r = _bop(op, s, x) if swapped else _bop(op, x, s)
            po.unsafe_offset(e + k).unsafe_store[alignment=1](r[0])
            k += 1
        e += run


def _row_broadcast(big: Tensor, small: Tensor) -> Bool:
    """small = big's shape with the last dim 1 (a per-row scalar, e.g. [B, C, 1] vs [B, C, T])."""
    var r = big.rank()
    if r < 2 or small.rank() != r or big.shape[r - 1] <= 1 or small.shape[r - 1] != 1:
        return False
    for d in range(r - 1):
        if small.shape[d] != big.shape[d]:
            return False
    return True


def row_binary_applies(op: Int, a: Tensor, b: Tensor) -> Bool:
    if not a.is_float() or not b.is_float():
        return False
    if not (op == B_ADD or op == B_SUB or op == B_MUL or op == B_DIV or op == B_MAX or op == B_MIN):
        return False
    if a.numel() < PAR_MIN and b.numel() < PAR_MIN:
        return False
    return _row_broadcast(a, b) or _row_broadcast(b, a)


def par_row_binary(mut pool: JobPool, op: Int, a: Tensor, b: Tensor) raises -> Tensor:
    """Binary op where one operand is a per-row scalar ([.., C, 1] against [.., C, T])."""
    var swapped = _row_broadcast(b, a)
    var out = Tensor.zeros((b if swapped else a).shape.copy())
    var c = _PwCtx()
    c.a = b.fptr() if swapped else a.fptr()
    c.b = a.fptr() if swapped else b.fptr()
    c.o = out.fptr()
    c.n = out.numel()
    c.op = op
    c.inner = out.shape[out.rank() - 1]
    c.mode = 1 if swapped else 0
    run_parallel_ctx(pool, (c.n + _CHUNK - 1) // _CHUNK, c, _pw_worker)
    return out^


def par_sin_cos_square(mut pool: JobPool, op: Int, x: Tensor) raises -> Tensor:
    """Sin (U_SIN), Cos (U_COS) or x*x (any other code) over the pool, float32 libm per element."""
    var out = Tensor.zeros(x.shape.copy())
    var c = _PwCtx()
    c.a = x.fptr()
    c.o = out.fptr()
    c.n = x.numel()
    c.op = op
    c.mode = 2
    run_parallel_ctx(pool, (c.n + _CHUNK - 1) // _CHUNK, c, _pw_worker)
    return out^


# ---------------------------------------------------------------- parallel conv im2col / col2out

struct _ColCtx(Movable):
    var x: Int
    var col: Int
    var tmp: Int
    var out: Int
    var cin_g: Int
    var kw: Int
    var t_in: Int
    var t_out: Int
    var stride: Int
    var dil: Int
    var pad0: Int
    var cout_g: Int
    var per: Int

    def __init__(out self):
        self.x = 0
        self.col = 0
        self.tmp = 0
        self.out = 0
        self.cin_g = 0
        self.kw = 0
        self.t_in = 0
        self.t_out = 0
        self.stride = 1
        self.dil = 1
        self.pad0 = 0
        self.cout_g = 0
        self.per = 1


def _im2col_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_ColCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var t0 = job * c[].per
    var t1 = min(c[].t_out, t0 + c[].per)
    var px = fptr_at(c[].x)
    var pc = fptr_at(c[].col)
    var kw = c[].kw
    var j = c[].cin_g * kw
    var t_in = c[].t_in
    for t in range(t0, t1):
        var base = t * c[].stride - c[].pad0
        for ci in range(c[].cin_g):
            var xb = ci * t_in
            var cb = t * j + ci * kw
            for kk in range(kw):
                var src = base + kk * c[].dil
                var v: Float32 = 0
                if src >= 0 and src < t_in:
                    v = px.unsafe_offset(xb + src).unsafe_load[alignment=1]()
                pc.unsafe_offset(cb + kk).unsafe_store[alignment=1](v)


def _col2out_worker(ctx_addr: Int, job: Int):
    """out[co, t] = tmp[t, co] for a block of output channels."""
    var c = Pointer[_ColCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var c0 = job * c[].per
    var c1 = min(c[].cout_g, c0 + c[].per)
    var pt = fptr_at(c[].tmp)
    var po = fptr_at(c[].out)
    var n = c[].cout_g
    var T = c[].t_out
    for co in range(c0, c1):
        for t in range(T):
            po.unsafe_offset(co * T + t).unsafe_store[alignment=1](pt.unsafe_offset(t * n + co).unsafe_load[alignment=1]())


def conv1d_par(mut pool: JobPool, x: Tensor, w: Tensor, b: Tensor, has_b: Bool, pad0: Int, pad1: Int, dil: Int,
               stride: Int, group: Int) raises -> Tensor:
    """1-D Conv like `conv1d_ext`, with the im2col and the output transpose spread over the pool."""
    if x.rank() != 3 or w.rank() != 3:
        raise Error("Conv: only 1-D convolution is supported")
    var batch = x.shape[0]
    var cin = x.shape[1]
    var t_in = x.shape[2]
    var cout = w.shape[0]
    var cin_g = w.shape[1]
    var kw = w.shape[2]
    if cin_g * group != cin or cout % group != 0:
        raise Error("Conv: channels do not match the group count")
    var cout_g = cout // group
    var t_out = (t_in + pad0 + pad1 - dil * (kw - 1) - 1) // stride + 1
    var j = cin_g * kw
    var out = Tensor.zeros([batch, cout, t_out])
    var col = List[Float32](length=t_out * j, fill=0.0)
    var tmp = List[Float32](length=t_out * cout_g, fill=0.0)
    var wt = List[Float32](length=j * cout_g, fill=0.0)
    var c = _ColCtx()
    c.cin_g = cin_g
    c.kw = kw
    c.t_in = t_in
    c.t_out = t_out
    c.stride = stride
    c.dil = dil
    c.pad0 = pad0
    c.cout_g = cout_g
    c.col = Int(col.unsafe_ptr())
    c.tmp = Int(tmp.unsafe_ptr())
    for bi in range(batch):
        for gi in range(group):
            c.x = x.fptr() + (bi * cin + gi * cin_g) * t_in * 4
            c.per = max(16, (t_out + 63) // 64)
            run_parallel_ctx(pool, (t_out + c.per - 1) // c.per, c, _im2col_worker)
            # W_g^T [j, cout_g] so the product takes the blocked (non-transposed) kernel
            var pw = fptr_at(w.fptr() + gi * cout_g * j * 4)
            for co in range(cout_g):
                for q in range(j):
                    wt[q * cout_g + co] = pw.unsafe_offset(co * j + q).unsafe_load[alignment=1]()
            matmul_2d(pool, Int(col.unsafe_ptr()), Int(wt.unsafe_ptr()), Int(tmp.unsafe_ptr()),
                      b.fptr() + gi * cout_g * 4 if has_b else 0, t_out, j, cout_g, False)
            c.out = out.fptr() + (bi * cout + gi * cout_g) * t_out * 4
            c.per = max(1, (cout_g + 63) // 64)
            run_parallel_ctx(pool, (cout_g + c.per - 1) // c.per, c, _col2out_worker)
    _ = col^
    _ = tmp^
    _ = wt^
    return out^


def conv1d_gpu(mut gpu: GpuMatmul, mut pool: JobPool, x: Tensor, w: Tensor, b: Tensor, has_b: Bool, pad0: Int,
               pad1: Int, dil: Int, stride: Int, group: Int, key: Int) raises -> Tensor:
    """1-D Conv on the GPU as a gather-GEMM: rows of x^T [t_in + 1, cin_g] (the extra row is zeros
    and stands in for padding taps) are gathered per output step by the tiled GEMM, so there is no
    host im2col. The [K * cin_g, cout_g] weight is cached on the device under `key` (+ group)."""
    if x.rank() != 3 or w.rank() != 3:
        raise Error("Conv: only 1-D convolution is supported")
    var batch = x.shape[0]
    var cin = x.shape[1]
    var t_in = x.shape[2]
    var cout = w.shape[0]
    var cin_g = w.shape[1]
    var kw = w.shape[2]
    if cin_g * group != cin or cout % group != 0:
        raise Error("Conv: channels do not match the group count")
    var cout_g = cout // group
    var t_out = (t_in + pad0 + pad1 - dil * (kw - 1) - 1) // stride + 1
    var out = Tensor.zeros([batch, cout, t_out])
    var idx = List[Float32](length=t_out * kw, fill=0.0)
    for t in range(t_out):
        for kk in range(kw):
            var src = t * stride + kk * dil - pad0
            idx[t * kw + kk] = Float32(src if src >= 0 and src < t_in else t_in)
    var xt = List[Float32](length=(t_in + 1) * cin_g, fill=0.0)
    var tmp = List[Float32](length=t_out * cout_g, fill=0.0)
    var c = _ColCtx()
    for gi in range(group):
        var wkey = key * 64 + gi
        var widx: Int
        var found = gpu.weights.get(wkey)
        if found:
            widx = found.value()
        else:
            var wg = List[Float32](length=kw * cin_g * cout_g, fill=0.0)
            var pw = fptr_at(w.fptr())
            for co in range(cout_g):
                for ci in range(cin_g):
                    for kk in range(kw):
                        wg[(kk * cin_g + ci) * cout_g + co] = \
                            pw.unsafe_offset(((gi * cout_g + co) * cin_g + ci) * kw + kk).unsafe_load[alignment=1]()
            widx = gpu.weight(wkey, Int(wg.unsafe_ptr()), kw * cin_g, cout_g, False)
            _ = wg^
        for bi in range(batch):
            # x[b, group channels] [cin_g, t_in] -> xt [t_in, cin_g]
            c.tmp = x.fptr() + (bi * cin + gi * cin_g) * t_in * 4
            c.out = Int(xt.unsafe_ptr())
            c.cout_g = t_in
            c.t_out = cin_g
            c.per = max(1, (t_in + 63) // 64)
            run_parallel_ctx(pool, (t_in + c.per - 1) // c.per, c, _col2out_worker)
            gpu.gather_matmul(Int(xt.unsafe_ptr()), t_in + 1, cin_g, idx, t_out, kw, widx, Int(tmp.unsafe_ptr()), cout_g)
            if has_b:
                for t in range(t_out):
                    for co in range(cout_g):
                        tmp[t * cout_g + co] += b.fget(gi * cout_g + co)
            c.tmp = Int(tmp.unsafe_ptr())
            c.out = out.fptr() + (bi * cout + gi * cout_g) * t_out * 4
            c.cout_g = cout_g
            c.t_out = t_out
            c.per = max(1, (cout_g + 63) // 64)
            run_parallel_ctx(pool, (cout_g + c.per - 1) // c.per, c, _col2out_worker)
    _ = xt^
    _ = tmp^
    _ = idx^
    return out^


# ----------------------------------------------------------------------------- GatherND / ReduceMin / ReduceMax

def gather_nd(x: Tensor, idx: Tensor, batch_dims: Int) raises -> Tensor:
    """GatherND with batch_dims = 0: `idx` [..., k] picks slices x[i0, .., ik-1, :]."""
    if batch_dims != 0:
        raise Error("GatherND: only batch_dims = 0 is supported")
    var r = x.rank()
    var q = idx.rank()
    var k = idx.shape[q - 1] if q > 0 else 0
    if k > r:
        raise Error("GatherND: index depth " + String(k) + " exceeds rank " + String(r))
    var st = strides_of(x.shape)
    var inner = 1
    var out_shape = List[Int]()
    for d in range(q - 1):
        out_shape.append(idx.shape[d])
    for d in range(k, r):
        out_shape.append(x.shape[d])
        inner *= x.shape[d]
    var n = numel_of(idx.shape) // k if k > 0 else 1
    var out = Tensor.like(x, out_shape^)
    for j in range(n):
        var base = 0
        for d in range(k):
            var v = idx.int_at(j * k + d)
            if v < 0:
                v += x.shape[d]
            if v < 0 or v >= x.shape[d]:
                raise Error("GatherND: index out of range")
            base += v * st[d]
        for e in range(inner):
            if x.is_float():
                out.f[j * inner + e] = x.fget(base + e)
            else:
                out.i[j * inner + e] = x.i[base + e]
    return out^


def reduce_min_max(x: Tensor, axes_in: List[Int], keepdims: Bool, is_max: Bool) raises -> Tensor:
    """ReduceMin / ReduceMax over `axes_in` (all axes when empty)."""
    var r = x.rank()
    var red = List[Bool](length=r, fill=False)
    if len(axes_in) == 0:
        for d in range(r):
            red[d] = True
    for a in axes_in:
        red[norm_axis(a, r)] = True
    var out_shape = List[Int]()
    var kept = List[Int]()
    for d in range(r):
        kept.append(1 if red[d] else x.shape[d])
        if not red[d]:
            out_shape.append(x.shape[d])
        elif keepdims:
            out_shape.append(1)
    var kst = strides_of(kept)
    var nk = numel_of(kept)
    var acc = List[Float64](length=nk, fill=-1.0e300 if is_max else 1.0e300)
    var ctr = List[Int](length=r, fill=0)
    for e in range(x.numel()):
        var o = 0
        for d in range(r):
            if not red[d]:
                o += ctr[d] * kst[d]
        var v = x.get(e)
        if (v > acc[o]) if is_max else (v < acc[o]):
            acc[o] = v
        var d = r - 1
        while d >= 0:
            ctr[d] += 1
            if ctr[d] < x.shape[d]:
                break
            ctr[d] = 0
            d -= 1
    var out = Tensor.like(x, out_shape^)
    for k in range(nk):
        if x.is_float():
            out.f[k] = Float32(acc[k])
        else:
            out.i[k] = Int(acc[k])
    return out^
