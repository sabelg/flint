"""Weight-only quantised matrices and their CPU kernels (`flint.ml.llm.quant`).

A `QMat` holds a linear layer's weight `W[n, k]` (torch layout, `y = x W^T`) in one of:

  - `Q_F32`: float32 rows.
  - `Q_INT8`: symmetric int8 in groups of 128 along k with one float32 scale per group
    (`absmax / 127`).
  - `Q_INT4`: symmetric 4-bit in groups of 32 along k with one float32 scale per group, codes
    -8..7 stored biased by 8. Each group's scale is the one of `absmax / f`, f in 6..8.5, with
    the least squared error (a cheap stand-in for calibration). In each group's 16 bytes, byte
    j holds element j in its low nibble and element j + 16 in its high nibble, so both halves
    unpack into contiguous SIMD vectors that line up with contiguous activations.

Quantisation needs k to be a multiple of 128 (int8) or 256 (int4); other matrices fall back to
fp32 / int8.

Several source tensors with the same k can be stacked into one matrix (q/k/v, gate/up), so a
block needs fewer, larger products. Conversion and quantisation run on the job pool.

`qmatmul` computes `y[m, n] (+)= x[m, k] W^T` on the job pool, dequantising inside the dot
product (rows of x are processed four at a time so each weight vector loads once).
"""

from std.ffi import external_call
from std.math import abs
from flint.core.jobs import JobPool
from flint.core.ecs import run_parallel_ctx
from flint.ml.llm.safetensors import SafeTensor, widen_to_f32

comptime Q_F32: Int = 0
comptime Q_INT8: Int = 1
comptime Q_INT4: Int = 2
comptime G4: Int = 32
comptime G8: Int = 128

comptime F16V = SIMD[DType.float32, 16]


def fmt_name(fmt: Int) -> String:
    if fmt == Q_INT8:
        return "int8"
    if fmt == Q_INT4:
        return "int4"
    return "fp32"


@always_inline
def _f(addr: Int) -> Pointer[Float32, MutAnyOrigin]:
    return Pointer[Float32, MutAnyOrigin](unsafe_from_address=addr)


@always_inline
def _i8(addr: Int) -> Pointer[Int8, MutAnyOrigin]:
    return Pointer[Int8, MutAnyOrigin](unsafe_from_address=addr)


@always_inline
def _u8(addr: Int) -> Pointer[UInt8, MutAnyOrigin]:
    return Pointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)


def qmat_bytes(fmt: Int, n: Int, k: Int) -> Int:
    if fmt == Q_INT8:
        return n * k
    if fmt == Q_INT4:
        return n * k // 2
    return n * k * 4


def int8_group(k: Int) -> Int:
    return G8 if k % G8 == 0 else k


def qmat_scales(fmt: Int, n: Int, k: Int) -> Int:
    if fmt == Q_INT8:
        return n * (k // int8_group(k))
    if fmt == Q_INT4:
        return n * (k // G4)
    return 0


struct _BuildCtx(Movable):
    var srcs: List[SafeTensor]
    var row0: List[Int]        # first stacked row of each source
    var k: Int
    var fmt: Int
    var data: Int
    var scales: Int
    var rows_per_job: Int
    var n: Int
    var scratch: Int           # [jobs, k] float32

    def __init__(out self, var srcs: List[SafeTensor], var row0: List[Int], k: Int, fmt: Int, data: Int,
                 scales: Int, rows_per_job: Int, n: Int, scratch: Int):
        self.srcs = srcs^
        self.row0 = row0^
        self.k = k
        self.fmt = fmt
        self.data = data
        self.scales = scales
        self.rows_per_job = rows_per_job
        self.n = n
        self.scratch = scratch


def _build_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_BuildCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var k = c[].k
    var r0 = job * c[].rows_per_job
    var r1 = min(c[].n, r0 + c[].rows_per_job)
    var tmp = c[].scratch + job * k * 4
    var s = 0
    for r in range(r0, r1):
        while s + 1 < len(c[].srcs) and r >= c[].row0[s + 1]:
            s += 1
        var local = r - c[].row0[s]
        var fmt = c[].fmt
        var dst = tmp if fmt != Q_F32 else c[].data + r * k * 4
        try:
            widen_to_f32(c[].srcs[s], dst, local * k, k)
        except:
            pass
        if fmt == Q_F32:
            continue
        var x = _f(dst)
        if fmt == Q_INT8:
            var g8 = int8_group(k)
            var q = _i8(c[].data + r * k)
            for g in range(k // g8):
                var amax: Float32 = 0
                for i in range(g8):
                    amax = max(amax, abs(x.unsafe_offset(g * g8 + i).unsafe_load()))
                var scale = amax / 127.0
                var inv: Float32 = 0 if amax == 0 else 1.0 / scale
                for i in range(g * g8, (g + 1) * g8):
                    var iv = Int(round(x.unsafe_offset(i).unsafe_load() * inv))
                    q.unsafe_offset(i).unsafe_store(Int8(max(-127, min(127, iv))))
                _f(c[].scales).unsafe_offset(r * (k // g8) + g).unsafe_store(scale)
        else:
            var groups = k // G4
            var q = _u8(c[].data + r * (k // 2))
            for g in range(groups):
                var xg = x.unsafe_offset(g * G4)
                var amax: Float32 = 0
                for i in range(G4):
                    amax = max(amax, abs(xg.unsafe_offset(i).unsafe_load()))
                var best_scale: Float32 = 0
                var best_err: Float32 = 3.4e38
                if amax > 0:
                    for step in range(11):
                        var scale = amax / (Float32(6.0) + Float32(step) * 0.25)
                        var inv = 1.0 / scale
                        var err: Float32 = 0
                        for i in range(G4):
                            var v = xg.unsafe_offset(i).unsafe_load()
                            var qv = Float32(max(-8, min(7, Int(round(v * inv)))))
                            err += (qv * scale - v) * (qv * scale - v)
                        if err < best_err:
                            best_err = err
                            best_scale = scale
                var inv: Float32 = 0 if best_scale == 0 else 1.0 / best_scale
                for j in range(G4 // 2):
                    var lo = Int(round(xg.unsafe_offset(j).unsafe_load() * inv))
                    var hi = Int(round(xg.unsafe_offset(j + G4 // 2).unsafe_load() * inv))
                    lo = max(-8, min(7, lo)) + 8
                    hi = max(-8, min(7, hi)) + 8
                    q.unsafe_offset(g * (G4 // 2) + j).unsafe_store(UInt8(lo | (hi << 4)))
                _f(c[].scales).unsafe_offset(r * groups + g).unsafe_store(best_scale)


struct QMat(Copyable, Movable):
    var n: Int
    var k: Int
    var fmt: Int
    var data: Int       # malloc'd host data (0 once released after a GPU upload)
    var scales: Int
    var dev: Int        # device buffer index of the data (-1: none)
    var dev_scales: Int

    def __init__(out self):
        self.n = 0
        self.k = 0
        self.fmt = Q_F32
        self.data = 0
        self.scales = 0
        self.dev = -1
        self.dev_scales = -1

    def release_host(mut self):
        """Frees the host copy (after a GPU upload, or when the owning model goes away)."""
        if self.data != 0:
            external_call["free", NoneType](self.data)
        if self.scales != 0:
            external_call["free", NoneType](self.scales)
        self.data = 0
        self.scales = 0

    def bytes(self) -> Int:
        return qmat_bytes(self.fmt, self.n, self.k) + qmat_scales(self.fmt, self.n, self.k) * 4

    @staticmethod
    def build(mut pool: JobPool, var srcs: List[SafeTensor], fmt_in: Int) raises -> QMat:
        """Stacks the rows of `srcs` (all [*, k]) into one matrix in format `fmt_in` (int4 falls
        back to int8 when k is not a multiple of 32)."""
        var m = QMat()
        m.k = srcs[0].cols()
        var fmt = fmt_in
        if fmt == Q_INT4 and m.k % 256 != 0:
            fmt = Q_INT8
        if fmt == Q_INT8 and m.k % G8 != 0:
            fmt = Q_F32
        m.fmt = fmt
        var row0 = List[Int]()
        for s in srcs:
            if s.cols() != m.k:
                raise Error("QMat: stacked tensors need the same k (" + s.name + ")")
            row0.append(m.n)
            m.n += s.rows()
        m.data = external_call["malloc", Int](qmat_bytes(fmt, m.n, m.k))
        var ns = qmat_scales(fmt, m.n, m.k)
        if ns > 0:
            m.scales = external_call["malloc", Int](ns * 4)
        var rows_per_job = 64
        var jobs = (m.n + rows_per_job - 1) // rows_per_job
        var scratch = external_call["malloc", Int](jobs * m.k * 4)
        var ctx = _BuildCtx(srcs^, row0^, m.k, fmt, m.data, m.scales, rows_per_job, m.n, scratch)
        run_parallel_ctx(pool, jobs, ctx, _build_worker)
        external_call["free", NoneType](scratch)
        return m^


# ---------------------------------------------------------------- CPU products

@always_inline
def _ld16(addr: Int) -> F16V:
    return _f(addr).unsafe_load[width=16, alignment=1]()


@always_inline
def _w16(fmt: Int, w: Int, kk: Int) -> F16V:
    """16 dequantised-but-unscaled weights at column kk (int4: one half of a group)."""
    if fmt == Q_INT8:
        return _i8(w + kk).unsafe_load[width=16, alignment=1]().cast[DType.float32]()
    return _ld16(w + kk * 4)


struct _MmCtx(Movable):
    var x: Int
    var y: Int
    var w: Int
    var s: Int
    var m: Int
    var k: Int
    var n: Int
    var fmt: Int
    var acc: Bool
    var cols: Int

    def __init__(out self, x: Int, y: Int, w: Int, s: Int, m: Int, k: Int, n: Int, fmt: Int, acc: Bool, cols: Int):
        self.x = x
        self.y = y
        self.w = w
        self.s = s
        self.m = m
        self.k = k
        self.n = n
        self.fmt = fmt
        self.acc = acc
        self.cols = cols


@always_inline
def _row_dot4(fmt: Int, w: Int, s: Int, j: Int, x: Int, k: Int, rows: Int) -> SIMD[DType.float32, 4]:
    """Dots of weight row j with up to four activation rows starting at x."""
    var a0 = F16V(0)
    var a1 = F16V(0)
    var a2 = F16V(0)
    var a3 = F16V(0)
    var x1 = x + k * 4
    var x2 = x + 2 * k * 4
    var x3 = x + 3 * k * 4
    if fmt == Q_INT4:
        var groups = k // G4
        var wr = w + j * (k // 2)
        var sr = _f(s + j * groups * 4)
        for g in range(groups):
            var b = _u8(wr + g * 16).unsafe_load[width=16, alignment=1]()
            var lo = ((b & 15).cast[DType.int8]() - 8).cast[DType.float32]()
            var hi = ((b >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
            var sc = sr.unsafe_offset(g).unsafe_load()
            var o = g * G4 * 4
            a0 += (lo * _ld16(x + o) + hi * _ld16(x + o + 64)) * sc
            if rows > 1:
                a1 += (lo * _ld16(x1 + o) + hi * _ld16(x1 + o + 64)) * sc
            if rows > 2:
                a2 += (lo * _ld16(x2 + o) + hi * _ld16(x2 + o + 64)) * sc
            if rows > 3:
                a3 += (lo * _ld16(x3 + o) + hi * _ld16(x3 + o + 64)) * sc
        return SIMD[DType.float32, 4](a0.reduce_add(), a1.reduce_add(), a2.reduce_add(), a3.reduce_add())
    if fmt == Q_INT8:
        var g8 = int8_group(k)
        var groups = k // g8
        var wr = w + j * k
        var sr = _f(s + j * groups * 4)
        var r8 = SIMD[DType.float32, 4](0)
        for g in range(groups):
            var b0 = F16V(0)
            var b1 = F16V(0)
            var b2 = F16V(0)
            var b3 = F16V(0)
            var kk = g * g8
            while kk < (g + 1) * g8:
                var wv = _w16(fmt, wr, kk)
                b0 += wv * _ld16(x + kk * 4)
                if rows > 1:
                    b1 += wv * _ld16(x1 + kk * 4)
                if rows > 2:
                    b2 += wv * _ld16(x2 + kk * 4)
                if rows > 3:
                    b3 += wv * _ld16(x3 + kk * 4)
                kk += 16
            r8 += SIMD[DType.float32, 4](b0.reduce_add(), b1.reduce_add(), b2.reduce_add(), b3.reduce_add()) \
                * sr.unsafe_offset(g).unsafe_load()
        return r8
    var wr = w + j * k * 4
    var kk = 0
    while kk + 16 <= k:
        var wv = _w16(fmt, wr, kk)
        a0 += wv * _ld16(x + kk * 4)
        if rows > 1:
            a1 += wv * _ld16(x1 + kk * 4)
        if rows > 2:
            a2 += wv * _ld16(x2 + kk * 4)
        if rows > 3:
            a3 += wv * _ld16(x3 + kk * 4)
        kk += 16
    var r = SIMD[DType.float32, 4](a0.reduce_add(), a1.reduce_add(), a2.reduce_add(), a3.reduce_add())
    while kk < k:
        var wv = _f(wr + kk * 4).unsafe_load()
        for q in range(rows):
            r[q] += wv * _f(x + (q * k + kk) * 4).unsafe_load()
        kk += 1
    return r


def _mm_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_MmCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var j0 = job * c[].cols
    var j1 = min(c[].n, j0 + c[].cols)
    var k = c[].k
    var n = c[].n
    var r0 = 0
    while r0 < c[].m:
        var rows = min(4, c[].m - r0)
        for j in range(j0, j1):
            var d = _row_dot4(c[].fmt, c[].w, c[].s, j, c[].x + r0 * k * 4, k, rows)
            for q in range(rows):
                var py = _f(c[].y + ((r0 + q) * n + j) * 4)
                if c[].acc:
                    py.unsafe_store(py.unsafe_load() + d[q])
                else:
                    py.unsafe_store(d[q])
        r0 += 4


def qmatmul(mut pool: JobPool, w: QMat, x: Int, y: Int, m: Int, acc: Bool):
    """y[m, n] = x[m, k] W^T, or y += x W^T when `acc` (fused residual)."""
    var cols = 16
    var jobs = (w.n + cols - 1) // cols
    if jobs > 512:
        cols = (w.n + 511) // 512
        jobs = (w.n + cols - 1) // cols
    var ctx = _MmCtx(x, y, w.data, w.scales, m, w.k, w.n, w.fmt, acc, cols)
    run_parallel_ctx(pool, jobs, ctx, _mm_worker)
