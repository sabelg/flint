"""Parallel SIMD kernels for the ONNX interpreter's hot paths (`flint.ml.fast`).

  * `par_unary` / `par_binary`: element-wise float ops over the job pool (same-shape operands, a
    scalar, or a trailing-dims broadcast such as a bias row).
  * `par_gather_rows`: Gather on axis 0 of a float tensor, one row copy per index.
  * `par_layer_norm`: last-axis LayerNormalization, rows in parallel.
  * `attention`: fused softmax(scale * Q K^T) V for [X.., L, H, D] / [X.., S, H, D] operands (the
    Einsum -> Mul -> Softmax -> Einsum chain exported for multi-head attention), blocked over query
    rows so the [H, L, S] score tensor is never stored.

Each returns an empty tensor (rank 0) when it does not apply, so callers fall back.
"""

from std.math import sqrt, exp, erf, tanh, floor
from std.ffi import external_call
from flint.core.jobs import JobPool
from flint.core.ecs import run_parallel_ctx
from flint.ml.tensor import Tensor, fptr_at, numel_of
from flint.ml.ops import (
    U_SQRT, U_RELU, U_COS, U_SIN, U_ABS, U_NEG, U_ERF, U_EXP, U_TANH, U_SIGMOID, U_RECIP, U_FLOOR,
    B_ADD, B_SUB, B_MUL, B_DIV, B_MAX, B_MIN, broadcast_shape,
)

comptime W: Int = 8
comptime V8 = SIMD[DType.float32, W]
comptime CHUNK: Int = 1 << 15
comptime MIN_PAR: Int = 1 << 14


# --------------------------------------------------------------------------- element-wise

struct _EwCtx(Movable):
    var a: Int
    var b: Int
    var o: Int
    var n: Int
    var op: Int
    var mode: Int       # 0 same shape, 1 b scalar, 2 a scalar, 3 b periodic, 4 a periodic, 9 unary
    var period: Int

    def __init__(out self):
        self.a = 0
        self.b = 0
        self.o = 0
        self.n = 0
        self.op = 0
        self.mode = 0
        self.period = 1


@always_inline
def _vop(op: Int, x: V8, y: V8) -> V8:
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


@always_inline
def _uop(op: Int, x: V8) -> V8:
    if op == U_ERF:
        return erf(x)
    if op == U_EXP:
        return exp(x)
    if op == U_TANH:
        return tanh(x)
    if op == U_SIGMOID:
        return 1.0 / (1.0 + exp(-x))
    if op == U_RELU:
        return max(x, V8(0))
    if op == U_SQRT:
        return sqrt(x)
    if op == U_ABS:
        return abs(x)
    if op == U_NEG:
        return -x
    if op == U_RECIP:
        return 1.0 / x
    return floor(x)


def _ew_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_EwCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var e0 = job * CHUNK
    var e1 = min(c[].n, e0 + CHUNK)
    var pa = fptr_at(c[].a)
    var pb = fptr_at(c[].b)
    var po = fptr_at(c[].o)
    var op = c[].op
    var mode = c[].mode
    var per = c[].period
    var e = e0
    if mode == 9:
        while e + W <= e1:
            po.unsafe_offset(e).unsafe_store[alignment=1](_uop(op, pa.unsafe_offset(e).unsafe_load[width=W, alignment=1]()))
            e += W
        while e < e1:
            var r = _uop(op, V8(pa.unsafe_offset(e).unsafe_load[alignment=1]()))
            po.unsafe_offset(e).unsafe_store[alignment=1](r[0])
            e += 1
        return
    if mode == 0 or mode == 1 or mode == 2:
        var sa = V8(pa.unsafe_load[alignment=1]()) if mode == 2 else V8(0)
        var sb = V8(pb.unsafe_load[alignment=1]()) if mode == 1 else V8(0)
        while e + W <= e1:
            var x = sa if mode == 2 else pa.unsafe_offset(e).unsafe_load[width=W, alignment=1]()
            var y = sb if mode == 1 else pb.unsafe_offset(e).unsafe_load[width=W, alignment=1]()
            po.unsafe_offset(e).unsafe_store[alignment=1](_vop(op, x, y))
            e += W
        while e < e1:
            var x = V8(pa.unsafe_load[alignment=1]()) if mode == 2 else V8(pa.unsafe_offset(e).unsafe_load[alignment=1]())
            var y = V8(pb.unsafe_load[alignment=1]()) if mode == 1 else V8(pb.unsafe_offset(e).unsafe_load[alignment=1]())
            po.unsafe_offset(e).unsafe_store[alignment=1](_vop(op, x, y)[0])
            e += 1
        return
    # periodic operand (mode 3: b repeats, mode 4: a repeats)
    while e < e1:
        var q = e % per
        var run = min(per - q, e1 - e)
        var k = 0
        while k + W <= run:
            var big = (pa if mode == 3 else pb).unsafe_offset(e + k).unsafe_load[width=W, alignment=1]()
            var sm = (pb if mode == 3 else pa).unsafe_offset(q + k).unsafe_load[width=W, alignment=1]()
            var r = _vop(op, big, sm) if mode == 3 else _vop(op, sm, big)
            po.unsafe_offset(e + k).unsafe_store[alignment=1](r)
            k += W
        while k < run:
            var big = V8((pa if mode == 3 else pb).unsafe_offset(e + k).unsafe_load[alignment=1]())
            var sm = V8((pb if mode == 3 else pa).unsafe_offset(q + k).unsafe_load[alignment=1]())
            var r = _vop(op, big, sm) if mode == 3 else _vop(op, sm, big)
            po.unsafe_offset(e + k).unsafe_store[alignment=1](r[0])
            k += 1
        e += run


def _period(small: List[Int], out_shape: List[Int]) -> Int:
    var lead = 0
    while lead < len(small) and small[lead] == 1:
        lead += 1
    var t = len(small) - lead
    if t > len(out_shape):
        return 0
    for d in range(t):
        if small[lead + d] != out_shape[len(out_shape) - t + d]:
            return 0
    return numel_of(small)


def par_unary(mut pool: JobPool, op: Int, x: Tensor) raises -> Tensor:
    if not x.is_float() or x.numel() < MIN_PAR:
        return Tensor()
    if not (op == U_ERF or op == U_EXP or op == U_TANH or op == U_SIGMOID or op == U_RELU or op == U_SQRT
            or op == U_ABS or op == U_NEG or op == U_RECIP or op == U_FLOOR):
        return Tensor()
    var n = x.numel()
    var out = Tensor.zeros(x.shape.copy())
    var c = _EwCtx()
    c.a = x.fptr()
    c.o = out.fptr()
    c.n = n
    c.op = op
    c.mode = 9
    run_parallel_ctx(pool, (n + CHUNK - 1) // CHUNK, c, _ew_worker)
    return out^


def par_binary(mut pool: JobPool, op: Int, a: Tensor, b: Tensor) raises -> Tensor:
    if not a.is_float() or not b.is_float():
        return Tensor()
    if not (op == B_ADD or op == B_SUB or op == B_MUL or op == B_DIV or op == B_MAX or op == B_MIN):
        return Tensor()
    var shape = broadcast_shape(a.shape, b.shape)
    var n = numel_of(shape)
    if n < MIN_PAR:
        return Tensor()
    var c = _EwCtx()
    var na = a.numel()
    var nb = b.numel()
    if na == n and nb == n and a.rank() == len(shape) and b.rank() == len(shape):
        c.mode = 0
    elif na == n and nb == 1:
        c.mode = 1
    elif nb == n and na == 1:
        c.mode = 2
    elif na == n and a.rank() == len(shape) and _period(b.shape, shape) > 0:
        c.mode = 3
        c.period = nb
    elif nb == n and b.rank() == len(shape) and _period(a.shape, shape) > 0:
        c.mode = 4
        c.period = na
    else:
        return Tensor()
    var out = Tensor.zeros(shape^)
    c.a = a.fptr()
    c.b = b.fptr()
    c.o = out.fptr()
    c.n = n
    c.op = op
    run_parallel_ctx(pool, (n + CHUNK - 1) // CHUNK, c, _ew_worker)
    return out^


# --------------------------------------------------------------------------- gather / layer norm

struct _RowCtx(Movable):
    var src: Int
    var dst: Int
    var idx: Int        # Int pointer (gather)
    var rows: Int
    var width: Int
    var dim: Int
    var gamma: Int
    var beta: Int
    var eps: Float32
    var per_job: Int

    def __init__(out self):
        self.src = 0
        self.dst = 0
        self.idx = 0
        self.rows = 0
        self.width = 0
        self.dim = 0
        self.gamma = 0
        self.beta = 0
        self.eps = 1e-5
        self.per_job = 1


def _gather_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_RowCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var ip = Pointer[Int, MutAnyOrigin](unsafe_from_address=c[].idx)
    var r0 = job * c[].per_job
    var r1 = min(c[].rows, r0 + c[].per_job)
    var wb = c[].width * 4
    for r in range(r0, r1):
        var g = ip.unsafe_offset(r).unsafe_load()
        if g < 0:
            g += c[].dim
        _ = external_call["memcpy", Int](c[].dst + r * wb, c[].src + g * wb, wb)


def par_gather_rows(mut pool: JobPool, data: Tensor, idx: Tensor, axis: Int) raises -> Tensor:
    """Gather(data, idx, axis=0) for float data and integer indices (bounds checked)."""
    if axis != 0 or not data.is_float() or idx.is_float() or data.rank() < 1:
        return Tensor()
    var dim = data.shape[0]
    var width = data.numel() // max(dim, 1)
    var ni = idx.numel()
    if ni * width < MIN_PAR:
        return Tensor()
    for k in range(ni):
        var g = idx.i[k]
        if g < -dim or g >= dim:
            raise Error("Gather: index out of range")
    var shape = idx.shape.copy()
    for d in range(1, data.rank()):
        shape.append(data.shape[d])
    var out = Tensor.zeros(shape^)
    var c = _RowCtx()
    c.src = data.fptr()
    c.dst = out.fptr()
    c.idx = Int(idx.i.unsafe_ptr())
    c.rows = ni
    c.width = width
    c.dim = dim
    c.per_job = max(1, CHUNK // max(width, 1))
    run_parallel_ctx(pool, (ni + c.per_job - 1) // c.per_job, c, _gather_worker)
    return out^


def _ln_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_RowCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var n = c[].width
    var ps = fptr_at(c[].src)
    var pd = fptr_at(c[].dst)
    var pg = fptr_at(c[].gamma)
    var pb = fptr_at(c[].beta)
    var r0 = job * c[].per_job
    var r1 = min(c[].rows, r0 + c[].per_job)
    for r in range(r0, r1):
        var base = r * n
        var acc = V8(0)
        var i = 0
        while i + W <= n:
            acc += ps.unsafe_offset(base + i).unsafe_load[width=W, alignment=1]()
            i += W
        var s = acc.reduce_add()
        while i < n:
            s += ps.unsafe_offset(base + i).unsafe_load[alignment=1]()
            i += 1
        var mean = s / Float32(n)
        var acc2 = V8(0)
        i = 0
        while i + W <= n:
            var d = ps.unsafe_offset(base + i).unsafe_load[width=W, alignment=1]() - mean
            acc2 += d * d
            i += W
        var v = acc2.reduce_add()
        while i < n:
            var d = ps.unsafe_offset(base + i).unsafe_load[alignment=1]() - mean
            v += d * d
            i += 1
        var inv = 1.0 / sqrt(v / Float32(n) + c[].eps)
        i = 0
        while i + W <= n:
            var x = (ps.unsafe_offset(base + i).unsafe_load[width=W, alignment=1]() - mean) * inv
            var y = x * pg.unsafe_offset(i).unsafe_load[width=W, alignment=1]()
            if c[].beta != 0:
                y += pb.unsafe_offset(i).unsafe_load[width=W, alignment=1]()
            pd.unsafe_offset(base + i).unsafe_store[alignment=1](y)
            i += W
        while i < n:
            var y = (ps.unsafe_offset(base + i).unsafe_load[alignment=1]() - mean) * inv * pg.unsafe_offset(i).unsafe_load[alignment=1]()
            if c[].beta != 0:
                y += pb.unsafe_offset(i).unsafe_load[alignment=1]()
            pd.unsafe_offset(base + i).unsafe_store[alignment=1](y)
            i += 1


def par_layer_norm(mut pool: JobPool, x: Tensor, scale: Tensor, bias: Tensor, has_bias: Bool, eps: Float32) raises -> Tensor:
    if not x.is_float() or x.numel() < MIN_PAR or not scale.is_float():
        return Tensor()
    var n = x.shape[x.rank() - 1]
    if scale.numel() != n or (has_bias and bias.numel() != n):
        return Tensor()
    var out = Tensor.zeros(x.shape.copy())
    var c = _RowCtx()
    c.src = x.fptr()
    c.dst = out.fptr()
    c.rows = x.numel() // n
    c.width = n
    c.gamma = scale.fptr()
    c.beta = bias.fptr() if has_bias else 0
    c.eps = eps
    c.per_job = max(1, CHUNK // n)
    run_parallel_ctx(pool, (c.rows + c.per_job - 1) // c.per_job, c, _ln_worker)
    return out^


# --------------------------------------------------------------------------- fused attention

comptime RB: Int = 8   # query rows per block


struct _AttCtx(Movable):
    var q: Int          # Q [X, L, H, D]
    var kt: Int         # K^T  [X, H, D, S] (contiguous copy)
    var vt: Int         # V    [X, H, S, D] (contiguous copy)
    var o: Int          # out  [X, L, H, D]
    var X: Int
    var L: Int
    var S: Int
    var H: Int
    var D: Int
    var scale: Float32
    var blocks: Int     # ceil(L / RB)

    def __init__(out self):
        self.q = 0
        self.kt = 0
        self.vt = 0
        self.o = 0
        self.X = 1
        self.L = 0
        self.S = 0
        self.H = 1
        self.D = 0
        self.scale = 1
        self.blocks = 0


struct _TrCtx(Movable):
    var src: Int
    var dst: Int
    var S: Int
    var H: Int
    var D: Int
    var to_dsx: Bool    # True: [S, H, D] -> [H, D, S]; False: -> [H, S, D]

    def __init__(out self):
        self.src = 0
        self.dst = 0
        self.S = 0
        self.H = 0
        self.D = 0
        self.to_dsx = False


def _tr_worker(ctx_addr: Int, job: Int):
    """Job = (x, h): copies one head of one batch."""
    var c = Pointer[_TrCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var S = c[].S
    var H = c[].H
    var D = c[].D
    var x = job // H
    var h = job % H
    var ps = fptr_at(c[].src + x * S * H * D * 4)
    var pd = fptr_at(c[].dst + (x * H + h) * S * D * 4)
    for s in range(S):
        for d in range(D):
            var v = ps.unsafe_offset((s * H + h) * D + d).unsafe_load[alignment=1]()
            if c[].to_dsx:
                pd.unsafe_offset(d * S + s).unsafe_store[alignment=1](v)
            else:
                pd.unsafe_offset(s * D + d).unsafe_store[alignment=1](v)


def _att_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_AttCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var S = c[].S
    var H = c[].H
    var D = c[].D
    var L = c[].L
    var blk = job % c[].blocks
    var xh = job // c[].blocks
    var x = xh // H
    var h = xh % H
    var l0 = blk * RB
    var nr = min(RB, L - l0)
    var sc = external_call["malloc", Int](RB * S * 4)
    var ps = fptr_at(sc)
    var kt = fptr_at(c[].kt + (x * H + h) * D * S * 4)
    var vt = fptr_at(c[].vt + (x * H + h) * S * D * 4)
    var pq = fptr_at(c[].q)
    var po = fptr_at(c[].o)
    # scores[r, s] = scale * sum_d q[r, d] k[d, s]
    for r in range(nr):
        for s in range(S):
            ps.unsafe_offset(r * S + s).unsafe_store(0)
    for d in range(D):
        for r in range(nr):
            var qv = pq.unsafe_offset(((x * L + l0 + r) * H + h) * D + d).unsafe_load[alignment=1]() * c[].scale
            var qq = V8(qv)
            var s = 0
            while s + W <= S:
                var acc = ps.unsafe_offset(r * S + s).unsafe_load[width=W, alignment=1]()
                acc += qq * kt.unsafe_offset(d * S + s).unsafe_load[width=W, alignment=1]()
                ps.unsafe_offset(r * S + s).unsafe_store[alignment=1](acc)
                s += W
            while s < S:
                ps.unsafe_offset(r * S + s).unsafe_store(ps.unsafe_offset(r * S + s).unsafe_load() + qv * kt.unsafe_offset(d * S + s).unsafe_load())
                s += 1
    # softmax rows
    var inv = List[Float32](length=RB, fill=0.0)
    for r in range(nr):
        var m: Float32 = -3.4e38
        for s in range(S):
            m = max(m, ps.unsafe_offset(r * S + s).unsafe_load())
        var acc = V8(0)
        var s = 0
        var mm = V8(m)
        while s + W <= S:
            var e = exp(ps.unsafe_offset(r * S + s).unsafe_load[width=W, alignment=1]() - mm)
            ps.unsafe_offset(r * S + s).unsafe_store[alignment=1](e)
            acc += e
            s += W
        var tot = acc.reduce_add()
        while s < S:
            var e = exp(ps.unsafe_offset(r * S + s).unsafe_load() - m)
            ps.unsafe_offset(r * S + s).unsafe_store(e)
            tot += e
            s += 1
        inv[r] = 1.0 / tot
    # out[r, :] = sum_s p[r, s] v[s, :]
    for r in range(nr):
        var ob = ((x * L + l0 + r) * H + h) * D
        var d0 = 0
        while d0 + W <= D:
            var acc = V8(0)
            for s in range(S):
                acc += ps.unsafe_offset(r * S + s).unsafe_load() * vt.unsafe_offset(s * D + d0).unsafe_load[width=W, alignment=1]()
            po.unsafe_offset(ob + d0).unsafe_store[alignment=1](acc * inv[r])
            d0 += W
        while d0 < D:
            var a: Float32 = 0
            for s in range(S):
                a += ps.unsafe_offset(r * S + s).unsafe_load() * vt.unsafe_offset(s * D + d0).unsafe_load()
            po.unsafe_offset(ob + d0).unsafe_store(a * inv[r])
            d0 += 1
    external_call["free", NoneType](sc)


def attention(mut pool: JobPool, q: Tensor, k: Tensor, v: Tensor, scale: Float32) raises -> Tensor:
    """softmax(scale * Q K^T) V per head: Q [X.., L, H, D], K and V [X.., S, H, D] -> [X.., L, H, D]."""
    var r = q.rank()
    if r < 3 or k.rank() != r or v.rank() != r or not q.is_float() or not k.is_float() or not v.is_float():
        return Tensor()
    var D = q.shape[r - 1]
    var H = q.shape[r - 2]
    var L = q.shape[r - 3]
    var S = k.shape[r - 3]
    if k.shape[r - 1] != D or k.shape[r - 2] != H or v.shape[r - 1] != D or v.shape[r - 2] != H or v.shape[r - 3] != S:
        return Tensor()
    var X = 1
    for d in range(r - 3):
        if k.shape[d] != q.shape[d] or v.shape[d] != q.shape[d]:
            return Tensor()
        X *= q.shape[d]
    var kt = Tensor.zeros([X * H * D * S])
    var vt = Tensor.zeros([X * H * S * D])
    var tc = _TrCtx()
    tc.src = k.fptr()
    tc.dst = kt.fptr()
    tc.S = S
    tc.H = H
    tc.D = D
    tc.to_dsx = True
    run_parallel_ctx(pool, X * H, tc, _tr_worker)
    var tv = _TrCtx()
    tv.src = v.fptr()
    tv.dst = vt.fptr()
    tv.S = S
    tv.H = H
    tv.D = D
    tv.to_dsx = False
    run_parallel_ctx(pool, X * H, tv, _tr_worker)
    var out = Tensor.zeros(q.shape.copy())
    var c = _AttCtx()
    c.q = q.fptr()
    c.kt = kt.fptr()
    c.vt = vt.fptr()
    c.o = out.fptr()
    c.X = X
    c.L = L
    c.S = S
    c.H = H
    c.D = D
    c.scale = scale
    c.blocks = (L + RB - 1) // RB
    run_parallel_ctx(pool, X * H * c.blocks, c, _att_worker)
    _ = kt^
    _ = vt^
    return out^
