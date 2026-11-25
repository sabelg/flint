"""ONNX operator kernels for normalisation and recurrent audio models (`flint.ml.ops_more`).

  - `instance_norm`  InstanceNormalization: per (batch, channel) mean / variance over the rest
  - `reduce_max`     ReduceMax over any axes
  - `gru`            GRU, forward / reverse / bidirectional, gates z-r-h, `linear_before_reset`
  - `softplus`, `ceil_op`
  - `conv_rows_in` / `conv_rows_out`: 2-D Conv with a 1 x k or k x 1 kernel as a 1-D Conv
  - `chan_binary_applies` / `par_chan_binary`: Add/Sub/Mul/Div of [C, 1] against [B, C, T]
  - `par_instance_norm`: InstanceNormalization, one (batch, channel) plane per pool job
  - `conv1d_gpu_batched`: the GPU gather-GEMM Conv with the batch folded into one product

Op codes 80..89 in `flint.ml.onnx` (60..79 belong to `ops_ext`). Used by the Audio2Face-3D
graphs (`flint.ml.audio2face`).
"""

from std.math import sqrt, exp, log, tanh, ceil
from flint.core.jobs import JobPool
from flint.ml.tensor import Tensor, numel_of, norm_axis, fptr_at
from flint.ml.ops import matmul_2d, B_ADD, B_SUB, B_MUL, B_DIV
from flint.ml.ops_ext import _ColCtx, _col2out_worker
from flint.ml.gpu import GpuMatmul
from flint.core.ecs import run_parallel_ctx


def instance_norm(x: Tensor, scale: Tensor, bias: Tensor, eps: Float32) raises -> Tensor:
    """y = scale[c] * (x - mean) / sqrt(var + eps) + bias[c] over the spatial axes of [N, C, ...]."""
    if x.rank() < 2:
        raise Error("InstanceNormalization needs rank >= 2")
    var n = x.shape[0]
    var c = x.shape[1]
    var inner = 1
    for d in range(2, x.rank()):
        inner *= x.shape[d]
    var o = List[Float32](length=x.numel(), fill=0.0)
    var p = fptr_at(x.fptr())
    for b in range(n):
        for ch in range(c):
            var base = (b * c + ch) * inner
            var s = 0.0
            for i in range(inner):
                s += Float64(p.unsafe_offset(base + i).unsafe_load())
            var mean = s / Float64(inner)
            var v = 0.0
            for i in range(inner):
                var d = Float64(p.unsafe_offset(base + i).unsafe_load()) - mean
                v += d * d
            v /= Float64(inner)
            var inv = 1.0 / sqrt(v + Float64(eps))
            var g = Float64(scale.fget(ch))
            var bb = Float64(bias.fget(ch))
            for i in range(inner):
                o[base + i] = Float32((Float64(p.unsafe_offset(base + i).unsafe_load()) - mean) * inv * g + bb)
    return Tensor.floats(x.shape.copy(), o^)


def reduce_max(x: Tensor, axes_in: List[Int], keepdims: Bool) raises -> Tensor:
    var r = x.rank()
    var red = List[Bool](length=r, fill=False)
    if len(axes_in) == 0:
        for d in range(r):
            red[d] = True
    for a in axes_in:
        red[norm_axis(a, r)] = True
    var out_shape = List[Int]()
    var full = List[Int]()  # output shape with reduced axes kept as 1
    for d in range(r):
        full.append(1 if red[d] else x.shape[d])
        if not red[d]:
            out_shape.append(x.shape[d])
        elif keepdims:
            out_shape.append(1)
    var on = numel_of(full)
    var o = List[Float32](length=on, fill=-3.4028234663852886e38)
    # walk the input once; map each element to its output slot
    var n = x.numel()
    var idx = List[Int](length=r, fill=0)
    var ostr = List[Int](length=r, fill=0)
    var acc = 1
    var d = r - 1
    while d >= 0:
        ostr[d] = 0 if red[d] else acc
        acc *= full[d]
        d -= 1
    for k in range(n):
        var off = 0
        for a in range(r):
            off += idx[a] * ostr[a]
        var v = Float32(x.get(k))
        if v > o[off]:
            o[off] = v
        var a = r - 1
        while a >= 0:
            idx[a] += 1
            if idx[a] < x.shape[a]:
                break
            idx[a] = 0
            a -= 1
    return Tensor.floats(out_shape^, o^)


def _sigmoid(v: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-v))


def gru(mut pool: JobPool, x: Tensor, w: Tensor, r: Tensor, b: Tensor, has_b: Bool, h0: Tensor, has_h0: Bool,
        hidden: Int, direction: String, linear_before_reset: Bool, layout: Int) raises -> List[Tensor]:
    """ONNX GRU (default activations). x [T, B, I] (layout 1: [B, T, I]); W [D, 3H, I];
    R [D, 3H, H]; B [D, 6H]. Returns Y [T, D, B, H] (layout 1: [B, T, D, H]) and Y_h [D, B, H]."""
    var T = x.shape[1] if layout == 1 else x.shape[0]
    var B = x.shape[0] if layout == 1 else x.shape[1]
    var I = x.shape[2]
    var H = hidden
    var ndir = 2 if direction == "bidirectional" else 1
    var y = List[Float32](length=T * ndir * B * H, fill=0.0)
    var yh = List[Float32](length=ndir * B * H, fill=0.0)
    var px = fptr_at(x.fptr())
    # x in [T*B, I] order (time-major)
    var xt = List[Float32](length=T * B * I, fill=0.0)
    for t in range(T):
        for bi in range(B):
            var src = (bi * T + t) * I if layout == 1 else (t * B + bi) * I
            var dst = (t * B + bi) * I
            for i in range(I):
                xt[dst + i] = px.unsafe_offset(src + i).unsafe_load()
    for dd in range(ndir):
        var reverse = direction == "reverse" or dd == 1
        # input projections for every step: [T*B, 3H] = x W_d^T
        var xw = List[Float32](length=T * B * 3 * H, fill=0.0)
        matmul_2d(pool, Int(xt.unsafe_ptr()), w.fptr() + dd * 3 * H * I * 4, Int(xw.unsafe_ptr()), 0, T * B, I, 3 * H, True)
        var pxw = fptr_at(Int(xw.unsafe_ptr()))
        var pr = fptr_at(r.fptr())
        var h = List[Float32](length=B * H, fill=0.0)
        if has_h0:
            for k in range(B * H):
                h[k] = h0.fget(dd * B * H + k)
        var hn = List[Float32](length=B * H, fill=0.0)
        for step in range(T):
            var t = T - 1 - step if reverse else step
            for bi in range(B):
                for j in range(H):
                    # gates z (0), r (1), h (2); W/R/B rows ordered z, r, h
                    var xz = pxw.unsafe_offset((t * B + bi) * 3 * H + j).unsafe_load()
                    var xr = pxw.unsafe_offset((t * B + bi) * 3 * H + H + j).unsafe_load()
                    var xh = pxw.unsafe_offset((t * B + bi) * 3 * H + 2 * H + j).unsafe_load()
                    var hz: Float32 = 0
                    var hr: Float32 = 0
                    var hh: Float32 = 0
                    for q in range(H):
                        var hv = h[bi * H + q]
                        hz += pr.unsafe_offset((dd * 3 * H + j) * H + q).unsafe_load() * hv
                        hr += pr.unsafe_offset((dd * 3 * H + H + j) * H + q).unsafe_load() * hv
                    var wbz: Float32 = 0
                    var wbr: Float32 = 0
                    var wbh: Float32 = 0
                    var rbz: Float32 = 0
                    var rbr: Float32 = 0
                    var rbh: Float32 = 0
                    if has_b:
                        wbz = b.fget(dd * 6 * H + j)
                        wbr = b.fget(dd * 6 * H + H + j)
                        wbh = b.fget(dd * 6 * H + 2 * H + j)
                        rbz = b.fget(dd * 6 * H + 3 * H + j)
                        rbr = b.fget(dd * 6 * H + 4 * H + j)
                        rbh = b.fget(dd * 6 * H + 5 * H + j)
                    var z = _sigmoid(xz + hz + wbz + rbz)
                    var rr = _sigmoid(xr + hr + wbr + rbr)
                    if linear_before_reset:
                        for q in range(H):
                            hh += pr.unsafe_offset((dd * 3 * H + 2 * H + j) * H + q).unsafe_load() * h[bi * H + q]
                        hh = rr * (hh + rbh)
                    else:
                        for q in range(H):
                            hh += pr.unsafe_offset((dd * 3 * H + 2 * H + j) * H + q).unsafe_load() * (rr * h[bi * H + q])
                        hh += rbh
                    var cand = tanh(xh + hh + wbh)
                    hn[bi * H + j] = (1.0 - z) * cand + z * h[bi * H + j]
            for k in range(B * H):
                h[k] = hn[k]
            for bi in range(B):
                for j in range(H):
                    if layout == 1:
                        y[((bi * T + t) * ndir + dd) * H + j] = h[bi * H + j]
                    else:
                        y[(t * ndir + dd) * B * H + bi * H + j] = h[bi * H + j]
        for k in range(B * H):
            yh[dd * B * H + k] = h[k]
    var out = List[Tensor]()
    if layout == 1:
        out.append(Tensor.floats([B, T, ndir, H], y^))
        out.append(Tensor.floats([B, ndir, H], yh^))
    else:
        out.append(Tensor.floats([T, ndir, B, H], y^))
        out.append(Tensor.floats([ndir, B, H], yh^))
    return out^


def softplus(x: Tensor) raises -> Tensor:
    var o = List[Float32](length=x.numel(), fill=0.0)
    for i in range(x.numel()):
        var v = Float32(x.get(i))
        # log(1 + e^v) without overflow
        o[i] = v + log(1.0 + exp(-v)) if v > 0 else log(1.0 + exp(v))
    return Tensor.floats(x.shape.copy(), o^)


def ceil_op(x: Tensor) raises -> Tensor:
    var o = List[Float32](length=x.numel(), fill=0.0)
    for i in range(x.numel()):
        o[i] = ceil(Float32(x.get(i)))
    return Tensor.floats(x.shape.copy(), o^)


# ------------------------------------------------------------- 2-D Conv with a 1 x k / k x 1 kernel
# A [N, C, H, W] convolution whose kernel is 1 wide (or 1 tall) is a 1-D convolution along the
# other axis with the remaining spatial axis folded into the batch: `conv_rows_in` makes the
# [N*H, C, W] (along W) or [N*W, C, H] (along H) input, `conv_rows_out` puts the result back.

def conv_rows_in(x: Tensor, along_w: Bool) raises -> Tensor:
    var n = x.shape[0]
    var c = x.shape[1]
    var h = x.shape[2]
    var w = x.shape[3]
    var o = List[Float32](length=x.numel(), fill=0.0)
    var p = fptr_at(x.fptr())
    if along_w:
        for b in range(n):
            for ch in range(c):
                for i in range(h):
                    var src = ((b * c + ch) * h + i) * w
                    var dst = ((b * h + i) * c + ch) * w
                    for j in range(w):
                        o[dst + j] = p.unsafe_offset(src + j).unsafe_load()
        return Tensor.floats([n * h, c, w], o^)
    for b in range(n):
        for ch in range(c):
            for i in range(h):
                var src = ((b * c + ch) * h + i) * w
                for j in range(w):
                    o[((b * w + j) * c + ch) * h + i] = p.unsafe_offset(src + j).unsafe_load()
    return Tensor.floats([n * w, c, h], o^)


def conv_rows_out(y: Tensor, n: Int, along_w: Bool) raises -> Tensor:
    var rows = y.shape[0] // n
    var co = y.shape[1]
    var l = y.shape[2]
    var o = List[Float32](length=y.numel(), fill=0.0)
    var p = fptr_at(y.fptr())
    if along_w:
        # y [N*H, Co, W'] -> [N, Co, H, W']
        for b in range(n):
            for i in range(rows):
                for ch in range(co):
                    var src = ((b * rows + i) * co + ch) * l
                    var dst = ((b * co + ch) * rows + i) * l
                    for j in range(l):
                        o[dst + j] = p.unsafe_offset(src + j).unsafe_load()
        return Tensor.floats([n, co, rows, l], o^)
    # y [N*W, Co, H'] -> [N, Co, H', W]
    for b in range(n):
        for j in range(rows):
            for ch in range(co):
                var src = ((b * rows + j) * co + ch) * l
                for i in range(l):
                    o[((b * co + ch) * l + i) * rows + j] = p.unsafe_offset(src + i).unsafe_load()
    return Tensor.floats([n, co, l, rows], o^)


def conv1d_gpu_batched(mut gpu: GpuMatmul, mut pool: JobPool, x: Tensor, w: Tensor, b: Tensor, has_b: Bool,
                       pad0: Int, pad1: Int, dil: Int, stride: Int, group: Int, key: Int) raises -> Tensor:
    """`ops_ext.conv1d_gpu` with the whole batch in one gather-GEMM per group: the transposed
    inputs of every batch item are stacked (each followed by its zero padding row), so a
    [B, C, T] conv costs one device product instead of B. Shares the weight cache layout and
    keys of `conv1d_gpu`."""
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
    var rows_b = t_in + 1
    var out = Tensor.zeros([batch, cout, t_out])
    var idx = List[Float32](length=batch * t_out * kw, fill=0.0)
    for bi in range(batch):
        for t in range(t_out):
            for kk in range(kw):
                var src = t * stride + kk * dil - pad0
                idx[(bi * t_out + t) * kw + kk] = Float32(bi * rows_b + (src if src >= 0 and src < t_in else t_in))
    var xt = List[Float32](length=batch * rows_b * cin_g, fill=0.0)
    var tmp = List[Float32](length=batch * t_out * cout_g, fill=0.0)
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
            # x[b, group channels] [cin_g, t_in] -> xt rows [b * (t_in + 1), +t_in)
            c.tmp = x.fptr() + (bi * cin + gi * cin_g) * t_in * 4
            c.out = Int(xt.unsafe_ptr()) + bi * rows_b * cin_g * 4
            c.cout_g = t_in
            c.t_out = cin_g
            c.per = max(1, (t_in + 63) // 64)
            run_parallel_ctx(pool, (t_in + c.per - 1) // c.per, c, _col2out_worker)
        gpu.gather_matmul(Int(xt.unsafe_ptr()), batch * rows_b, cin_g, idx, batch * t_out, kw, widx,
                          Int(tmp.unsafe_ptr()), cout_g)
        if has_b:
            for r in range(batch * t_out):
                for co in range(cout_g):
                    tmp[r * cout_g + co] += b.fget(gi * cout_g + co)
        for bi in range(batch):
            c.tmp = Int(tmp.unsafe_ptr()) + bi * t_out * cout_g * 4
            c.out = out.fptr() + (bi * cout + gi * cout_g) * t_out * 4
            c.cout_g = cout_g
            c.t_out = t_out
            c.per = max(1, (cout_g + 63) // 64)
            run_parallel_ctx(pool, (cout_g + c.per - 1) // c.per, c, _col2out_worker)
    _ = xt^
    _ = tmp^
    _ = idx^
    return out^


# ------------------------------------------------- per-channel broadcasts ([C, 1] against [B, C, T])

def chan_binary_applies(op: Int, a: Tensor, b: Tensor) -> Bool:
    """One operand is a per-channel scalar that broadcasts over a leading batch: small has the big
    tensor's shape from some axis on, with the last dim 1 and every dim before it missing or 1
    (e.g. [C, 1] or [1, C, 1] against [B, C, T]). `ops_ext.row_binary_applies` covers equal ranks."""
    if not a.is_float() or not b.is_float():
        return False
    if not (op == B_ADD or op == B_SUB or op == B_MUL or op == B_DIV):
        return False
    if a.numel() < PAR_MIN_MORE and b.numel() < PAR_MIN_MORE:
        return False
    return _chan_broadcast(a, b) or _chan_broadcast(b, a)


def _chan_broadcast(big: Tensor, small: Tensor) -> Bool:
    var r = big.rank()
    var s = small.rank()
    if r < 2 or s < 2 or s > r or big.shape[r - 1] <= 1 or small.shape[s - 1] != 1:
        return False
    # the trailing run of small (before its last 1) that matches big; everything before must be 1
    var i = s - 2
    var j = r - 2
    var matched = 0
    while i >= 0 and small.shape[i] == big.shape[j] and small.shape[i] != 1:
        i -= 1
        j -= 1
        matched += 1
    if matched == 0:
        return False
    while i >= 0:
        if small.shape[i] != 1:
            return False
        i -= 1
    return True


comptime PAR_MIN_MORE: Int = 1 << 14
comptime _CH_CHUNK: Int = 1 << 14


struct _ChanCtx(Movable):
    var a: Int
    var b: Int
    var o: Int
    var n: Int
    var inner: Int
    var chans: Int
    var op: Int
    var swapped: Bool

    def __init__(out self):
        self.a = 0
        self.b = 0
        self.o = 0
        self.n = 0
        self.inner = 0
        self.chans = 0
        self.op = 0
        self.swapped = False


def _chan_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_ChanCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var e0 = job * _CH_CHUNK
    var e1 = min(c[].n, e0 + _CH_CHUNK)
    var pa = fptr_at(c[].a)
    var pb = fptr_at(c[].b)
    var po = fptr_at(c[].o)
    var op = c[].op
    var inner = c[].inner
    var e = e0
    while e < e1:
        var row = (e // inner) % c[].chans
        var run = min(inner - e % inner, e1 - e)
        var s = pb.unsafe_offset(row).unsafe_load[alignment=1]()
        for k in range(run):
            var x = pa.unsafe_offset(e + k).unsafe_load[alignment=1]()
            var u = x
            var v = s
            if c[].swapped:
                u = s
                v = x
            var r: Float32
            if op == B_ADD:
                r = u + v
            elif op == B_SUB:
                r = u - v
            elif op == B_MUL:
                r = u * v
            else:
                r = u / v
            po.unsafe_offset(e + k).unsafe_store[alignment=1](r)
        e += run


def par_chan_binary(mut pool: JobPool, op: Int, a: Tensor, b: Tensor) raises -> Tensor:
    var swapped = _chan_broadcast(b, a)
    ref big = b if swapped else a
    ref small = a if swapped else b
    var out = Tensor.zeros(big.shape.copy())
    var c = _ChanCtx()
    c.a = big.fptr()
    c.b = small.fptr()
    c.o = out.fptr()
    c.n = out.numel()
    c.inner = big.shape[big.rank() - 1]
    c.chans = small.numel()
    c.op = op
    c.swapped = swapped
    run_parallel_ctx(pool, (c.n + _CH_CHUNK - 1) // _CH_CHUNK, c, _chan_worker)
    return out^


# ------------------------------------------------------------------ parallel InstanceNormalization

struct _InCtx(Movable):
    var x: Int
    var o: Int
    var s: Int
    var b: Int
    var chans: Int
    var inner: Int
    var eps: Float64

    def __init__(out self):
        self.x = 0
        self.o = 0
        self.s = 0
        self.b = 0
        self.chans = 0
        self.inner = 0
        self.eps = 0


def _in_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_InCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var inner = c[].inner
    var base = job * inner
    var p = fptr_at(c[].x)
    var po = fptr_at(c[].o)
    var ch = job % c[].chans
    var s = 0.0
    for i in range(inner):
        s += Float64(p.unsafe_offset(base + i).unsafe_load())
    var mean = s / Float64(inner)
    var v = 0.0
    for i in range(inner):
        var d = Float64(p.unsafe_offset(base + i).unsafe_load()) - mean
        v += d * d
    v /= Float64(inner)
    var inv = 1.0 / sqrt(v + c[].eps)
    var g = Float64(fptr_at(c[].s).unsafe_offset(ch).unsafe_load())
    var bb = Float64(fptr_at(c[].b).unsafe_offset(ch).unsafe_load())
    var mul = Float32(inv * g)
    var add = Float32(bb - mean * inv * g)
    for i in range(inner):
        po.unsafe_offset(base + i).unsafe_store(p.unsafe_offset(base + i).unsafe_load() * mul + add)


def par_instance_norm(mut pool: JobPool, x: Tensor, scale: Tensor, bias: Tensor, eps: Float32) raises -> Tensor:
    """`instance_norm` with one (batch, channel) plane per job."""
    if x.rank() < 2:
        raise Error("InstanceNormalization needs rank >= 2")
    var inner = 1
    for d in range(2, x.rank()):
        inner *= x.shape[d]
    var out = Tensor.zeros(x.shape.copy())
    var c = _InCtx()
    c.x = x.fptr()
    c.o = out.fptr()
    c.s = scale.fptr()
    c.b = bias.fptr()
    c.chans = x.shape[1]
    c.inner = inner
    c.eps = Float64(eps)
    run_parallel_ctx(pool, x.shape[0] * x.shape[1], c, _in_worker)
    return out^


# ------------------------------------------------------------------ ReduceSum over the last axis

def is_last_axis(x: Tensor, axes: List[Int]) -> Bool:
    return x.is_float() and x.rank() >= 1 and len(axes) == 1 and norm_axis(axes[0], x.rank()) == x.rank() - 1


struct _RsCtx(Movable):
    var x: Int
    var o: Int
    var inner: Int
    var rows: Int
    var per: Int

    def __init__(out self):
        self.x = 0
        self.o = 0
        self.inner = 0
        self.rows = 0
        self.per = 0


def _rs_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_RsCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var r0 = job * c[].per
    var r1 = min(c[].rows, r0 + c[].per)
    var p = fptr_at(c[].x)
    var po = fptr_at(c[].o)
    var n = c[].inner
    for r in range(r0, r1):
        var s: Float32 = 0
        for i in range(n):
            s += p.unsafe_offset(r * n + i).unsafe_load()
        po.unsafe_offset(r).unsafe_store(s)


def par_reduce_sum_last(mut pool: JobPool, x: Tensor, keepdims: Bool) raises -> Tensor:
    """ReduceSum over the last axis, rows spread over the pool (sequential float32 sum per row)."""
    var r = x.rank()
    var shape = List[Int]()
    for d in range(r - 1):
        shape.append(x.shape[d])
    if keepdims:
        shape.append(1)
    var inner = x.shape[r - 1]
    var rows = x.numel() // inner if inner > 0 else 0
    var out = Tensor.zeros(shape^)
    var c = _RsCtx()
    c.x = x.fptr()
    c.o = out.fptr()
    c.inner = inner
    c.rows = rows
    c.per = max(1, (rows + 63) // 64)
    run_parallel_ctx(pool, (rows + c.per - 1) // c.per, c, _rs_worker)
    return out^
