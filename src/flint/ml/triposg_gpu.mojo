"""MAX GPU kernels for `flint.ml.triposg`.

`TgGpu` owns a `GpuMatmul` (device context, resident weights, activation pool, LayerNorm) and
adds what the TripoSG networks need on top of it:

  - `gemm`: strided, batched SGEMM, C = alpha A B (+ bias) (+= C), B either [k, n] or [n, k]
    (PyTorch `Linear` weights are used as stored, no transpose on upload); batches index an
    (outer, inner) pair so one launch covers every (sample, head) of an attention product.
  - `softmax`: row softmax in place (attention scores).
  - `head_rms`: per-head RMSNorm with a learned scale (the QK-norm of the DiT).
  - `gelu`, `add`, `add_scaled` (DINOv2 LayerScale), `copy2d` (strided row copies for concat /
    split), `freq_embed` (VAE query positional embedding, 3 -> 51 channels).

Attention is two batched GEMMs around a softmax: a [L, S] score tile per (sample, head) fits
easily in memory for the sizes TripoSG uses (2049 x 2049 DiT tokens, query chunks against the
2048 VAE latents), and the GEMM is the fastest kernel Flint has.
"""

from std.math import exp, erf, sqrt, sin, cos
from max.gpu import global_idx, thread_idx, block_idx, barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation
from flint.ml.gpu import GpuMatmul, FP

comptime TG_BM: Int = 128
comptime TG_BN: Int = 128
comptime TG_BK: Int = 8
comptime TG_TT: Int = 8
comptime TG_V8 = SIMD[DType.float32, 8]
comptime TG_EW: Int = 256
comptime TG_RT: Int = 256


def _tg_gemm[TB: Bool](a: FP, b: FP, c: FP, bias: FP, m: Int32, k: Int32, n: Int32, lda: Int32, ldb: Int32,
                       ldc: Int32, alpha: Float32, accum: Int32, has_bias: Int32, tiles_n: Int32, tiles_mn: Int32,
                       inner: Int32, a_so: Int32, a_si: Int32, b_so: Int32, b_si: Int32, c_so: Int32, c_si: Int32):
    """One 128 x 128 tile of C[batch] per block (16 x 16 threads, 8 x 8 outputs each).
    batch = (outer, inner); operand offsets are outer * s_o + inner * s_i floats.
    TB: B is stored [n, k] (row j holds column j of the product)."""
    var As = stack_allocation[TG_BK * TG_BM, Float32, address_space=AddressSpace.SHARED]()
    var Bs = stack_allocation[TG_BK * TG_BN, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var tx = t % 16
    var ty = t // 16
    var blk = Int(block_idx.x)
    var bt = blk // Int(tiles_mn)
    var rem = blk % Int(tiles_mn)
    var bo = bt // Int(inner)
    var bi = bt % Int(inner)
    var ao = bo * Int(a_so) + bi * Int(a_si)
    var bof = bo * Int(b_so) + bi * Int(b_si)
    var co = bo * Int(c_so) + bi * Int(c_si)
    var bn = rem % Int(tiles_n)
    var bm = rem // Int(tiles_n)
    var mm = Int(m)
    var kn = Int(k)
    var nn = Int(n)
    var la = Int(lda)
    var lb = Int(ldb)
    var lc = Int(ldc)
    var row0 = bm * TG_BM
    var col0 = bn * TG_BN
    var acc0 = TG_V8(0)
    var acc1 = TG_V8(0)
    var acc2 = TG_V8(0)
    var acc3 = TG_V8(0)
    var acc4 = TG_V8(0)
    var acc5 = TG_V8(0)
    var acc6 = TG_V8(0)
    var acc7 = TG_V8(0)
    var k0 = 0
    while k0 < kn:
        comptime for q in range(4):
            var e = t * 4 + q
            var r = e // TG_BK
            var kk = e % TG_BK
            var gr = row0 + r
            var gk = k0 + kk
            var v: Float32 = 0
            if gr < mm and gk < kn:
                v = a.unsafe_offset(ao + gr * la + gk).unsafe_load()
            As[unsafe_offset=kk * TG_BM + r] = v
        comptime for q in range(4):
            var e = t * 4 + q
            var kk: Int
            var cc: Int
            comptime if TB:
                kk = e % TG_BK
                cc = e // TG_BK
            else:
                kk = e // TG_BN
                cc = e % TG_BN
            var gk = k0 + kk
            var gc = col0 + cc
            var v: Float32 = 0
            if gk < kn and gc < nn:
                comptime if TB:
                    v = b.unsafe_offset(bof + gc * lb + gk).unsafe_load()
                else:
                    v = b.unsafe_offset(bof + gk * lb + gc).unsafe_load()
            Bs[unsafe_offset=kk * TG_BN + cc] = v
        barrier()
        comptime for kk in range(TG_BK):
            var bv = TG_V8(0)
            comptime for j in range(8):
                bv[j] = Bs[unsafe_offset=kk * TG_BN + tx * TG_TT + j]
            acc0 += As[unsafe_offset=kk * TG_BM + ty * TG_TT + 0] * bv
            acc1 += As[unsafe_offset=kk * TG_BM + ty * TG_TT + 1] * bv
            acc2 += As[unsafe_offset=kk * TG_BM + ty * TG_TT + 2] * bv
            acc3 += As[unsafe_offset=kk * TG_BM + ty * TG_TT + 3] * bv
            acc4 += As[unsafe_offset=kk * TG_BM + ty * TG_TT + 4] * bv
            acc5 += As[unsafe_offset=kk * TG_BM + ty * TG_TT + 5] * bv
            acc6 += As[unsafe_offset=kk * TG_BM + ty * TG_TT + 6] * bv
            acc7 += As[unsafe_offset=kk * TG_BM + ty * TG_TT + 7] * bv
        barrier()
        k0 += TG_BK
    var bvec = TG_V8(0)
    if has_bias != 0:
        comptime for j in range(8):
            var gc = col0 + tx * TG_TT + j
            if gc < nn:
                bvec[j] = bias.unsafe_offset(gc).unsafe_load()
    comptime for i in range(8):
        var gr = row0 + ty * TG_TT + i
        if gr < mm:
            var accv: TG_V8
            comptime if i == 0:
                accv = acc0
            elif i == 1:
                accv = acc1
            elif i == 2:
                accv = acc2
            elif i == 3:
                accv = acc3
            elif i == 4:
                accv = acc4
            elif i == 5:
                accv = acc5
            elif i == 6:
                accv = acc6
            else:
                accv = acc7
            accv = accv * alpha + bvec
            comptime for j in range(8):
                var gc = col0 + tx * TG_TT + j
                if gc < nn:
                    var p = c.unsafe_offset(co + gr * lc + gc)
                    if accum != 0:
                        p.unsafe_store(p.unsafe_load() + accv[j])
                    else:
                        p.unsafe_store(accv[j])


comptime TG_V4 = SIMD[DType.float32, 4]


@always_inline
def _ld4(p: FP, off: Int, valid: Int) -> TG_V4:
    """Four floats at p[off..]; `valid` (0..4) of them in bounds (the rest read as 0)."""
    if valid >= 4:
        return p.unsafe_offset(off).unsafe_load[width=4, alignment=16]()
    var v = TG_V4(0)
    for i in range(valid):
        v[i] = p.unsafe_offset(off + i).unsafe_load()
    return v


def _tg_gemm4[TB: Bool](a: FP, b: FP, c: FP, bias: FP, m: Int32, k: Int32, n: Int32, lda: Int32, ldb: Int32,
                        ldc: Int32, alpha: Float32, accum: Int32, has_bias: Int32, tiles_n: Int32, tiles_mn: Int32,
                        inner: Int32, a_so: Int32, a_si: Int32, b_so: Int32, b_si: Int32, c_so: Int32, c_si: Int32):
    """`_tg_gemm` with 16-byte global loads, a register-prefetched double-buffered K loop and
    bank-conflict-free fragments (thread rows ty*4 + {0..3, 64..67}, columns tx*4 + {0..3, 64..67}).
    Needs lda / ldb and every operand offset to be multiples of 4 floats."""
    var As = stack_allocation[2 * TG_BK * TG_BM, Float32, address_space=AddressSpace.SHARED]()
    var Bs = stack_allocation[2 * TG_BK * TG_BN, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var tx = t % 16
    var ty = t // 16
    var blk = Int(block_idx.x)
    var bt = blk // Int(tiles_mn)
    var rem = blk % Int(tiles_mn)
    var bo = bt // Int(inner)
    var bi = bt % Int(inner)
    var ao = bo * Int(a_so) + bi * Int(a_si)
    var bof = bo * Int(b_so) + bi * Int(b_si)
    var co = bo * Int(c_so) + bi * Int(c_si)
    var bn = rem % Int(tiles_n)
    var bm = rem // Int(tiles_n)
    var mm = Int(m)
    var kn = Int(k)
    var nn = Int(n)
    var la = Int(lda)
    var lb = Int(ldb)
    var lc = Int(ldc)
    var row0 = bm * TG_BM
    var col0 = bn * TG_BN
    # this thread's global-load slots
    var ar = t // 2
    var akq = (t % 2) * 4
    var bkk = t // 32
    var bcq = (t % 32) * 4
    var acc0 = TG_V8(0)
    var acc1 = TG_V8(0)
    var acc2 = TG_V8(0)
    var acc3 = TG_V8(0)
    var acc4 = TG_V8(0)
    var acc5 = TG_V8(0)
    var acc6 = TG_V8(0)
    var acc7 = TG_V8(0)
    var nt = (kn + TG_BK - 1) // TG_BK
    # prologue: tile 0
    var ra: TG_V4
    var rb: TG_V4
    var gk = akq
    ra = _ld4(a, ao + (row0 + ar) * la + gk, min(4, kn - gk)) if row0 + ar < mm else TG_V4(0)
    comptime if TB:
        rb = _ld4(b, bof + (col0 + ar) * lb + gk, min(4, kn - gk)) if col0 + ar < nn else TG_V4(0)
    else:
        rb = _ld4(b, bof + bkk * lb + col0 + bcq, min(4, nn - col0 - bcq)) if bkk < kn else TG_V4(0)
    comptime for i in range(4):
        As[unsafe_offset=(akq + i) * TG_BM + ar] = ra[i]
    comptime if TB:
        comptime for i in range(4):
            Bs[unsafe_offset=(akq + i) * TG_BN + ar] = rb[i]
    else:
        comptime for i in range(4):
            Bs[unsafe_offset=bkk * TG_BN + bcq + i] = rb[i]
    barrier()
    var cur = 0
    for tile in range(nt):
        var more = tile + 1 < nt
        if more:
            var k1 = (tile + 1) * TG_BK
            gk = k1 + akq
            ra = _ld4(a, ao + (row0 + ar) * la + gk, min(4, kn - gk)) if row0 + ar < mm else TG_V4(0)
            comptime if TB:
                rb = _ld4(b, bof + (col0 + ar) * lb + gk, min(4, kn - gk)) if col0 + ar < nn else TG_V4(0)
            else:
                rb = _ld4(b, bof + (k1 + bkk) * lb + col0 + bcq, min(4, nn - col0 - bcq)) if k1 + bkk < kn else TG_V4(0)
        var abase = cur * TG_BK * TG_BM
        var bbase = cur * TG_BK * TG_BN
        comptime for kk in range(TG_BK):
            var a0 = As.unsafe_offset(abase + kk * TG_BM + ty * 4).unsafe_load[width=4, alignment=16]()
            var a1 = As.unsafe_offset(abase + kk * TG_BM + 64 + ty * 4).unsafe_load[width=4, alignment=16]()
            var b0 = Bs.unsafe_offset(bbase + kk * TG_BN + tx * 4).unsafe_load[width=4, alignment=16]()
            var b1 = Bs.unsafe_offset(bbase + kk * TG_BN + 64 + tx * 4).unsafe_load[width=4, alignment=16]()
            var bv = b0.join(b1)
            acc0 += a0[0] * bv
            acc1 += a0[1] * bv
            acc2 += a0[2] * bv
            acc3 += a0[3] * bv
            acc4 += a1[0] * bv
            acc5 += a1[1] * bv
            acc6 += a1[2] * bv
            acc7 += a1[3] * bv
        if more:
            var nb = 1 - cur
            comptime for i in range(4):
                As[unsafe_offset=nb * TG_BK * TG_BM + (akq + i) * TG_BM + ar] = ra[i]
            comptime if TB:
                comptime for i in range(4):
                    Bs[unsafe_offset=nb * TG_BK * TG_BN + (akq + i) * TG_BN + ar] = rb[i]
            else:
                comptime for i in range(4):
                    Bs[unsafe_offset=nb * TG_BK * TG_BN + bkk * TG_BN + bcq + i] = rb[i]
            barrier()
            cur = nb
    var bvec = TG_V8(0)
    if has_bias != 0:
        comptime for j in range(8):
            var gc = col0 + (tx * 4 + j if j < 4 else 64 + tx * 4 + j - 4)
            if gc < nn:
                bvec[j] = bias.unsafe_offset(gc).unsafe_load()
    comptime for i in range(8):
        var gr = row0 + (ty * 4 + i if i < 4 else 64 + ty * 4 + i - 4)
        if gr < mm:
            var accv: TG_V8
            comptime if i == 0:
                accv = acc0
            elif i == 1:
                accv = acc1
            elif i == 2:
                accv = acc2
            elif i == 3:
                accv = acc3
            elif i == 4:
                accv = acc4
            elif i == 5:
                accv = acc5
            elif i == 6:
                accv = acc6
            else:
                accv = acc7
            accv = accv * alpha + bvec
            comptime for j in range(8):
                var gc = col0 + (tx * 4 + j if j < 4 else 64 + tx * 4 + j - 4)
                if gc < nn:
                    var p = c.unsafe_offset(co + gr * lc + gc)
                    if accum != 0:
                        p.unsafe_store(p.unsafe_load() + accv[j])
                    else:
                        p.unsafe_store(accv[j])


def _tg_softmax_kernel(x: FP, cols: Int32, ld: Int32):
    var red = stack_allocation[TG_RT, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var n = Int(cols)
    var base = Int(block_idx.x) * Int(ld)
    var mx: Float32 = -3.0e38
    var j = t
    while j < n:
        mx = max(mx, x.unsafe_offset(base + j).unsafe_load())
        j += TG_RT
    red[unsafe_offset=t] = mx
    barrier()
    var w = TG_RT // 2
    while w > 0:
        if t < w:
            red[unsafe_offset=t] = max(red[unsafe_offset=t], red[unsafe_offset=t + w])
        barrier()
        w //= 2
    var m = red[unsafe_offset=0]
    barrier()
    var tot: Float32 = 0
    j = t
    while j < n:
        var e = exp(x.unsafe_offset(base + j).unsafe_load() - m)
        x.unsafe_offset(base + j).unsafe_store(e)
        tot += e
        j += TG_RT
    red[unsafe_offset=t] = tot
    barrier()
    w = TG_RT // 2
    while w > 0:
        if t < w:
            red[unsafe_offset=t] = red[unsafe_offset=t] + red[unsafe_offset=t + w]
        barrier()
        w //= 2
    var inv = 1.0 / red[unsafe_offset=0]
    j = t
    while j < n:
        x.unsafe_offset(base + j).unsafe_store(x.unsafe_offset(base + j).unsafe_load() * inv)
        j += TG_RT


def _tg_head_rms_kernel(x: FP, g: FP, heads: Int32, hd: Int32, ld: Int32, eps: Float32):
    """Block per (row, head), 128 threads: x[row, h*hd : (h+1)*hd] *= rsqrt(mean(x^2) + eps) * g."""
    var red = stack_allocation[128, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var H = Int(heads)
    var r = Int(block_idx.x) // H
    var h = Int(block_idx.x) % H
    var D = Int(hd)
    var base = r * Int(ld) + h * D
    var s: Float32 = 0
    var d = t
    while d < D:
        var v = x.unsafe_offset(base + d).unsafe_load()
        s += v * v
        d += 128
    red[unsafe_offset=t] = s
    barrier()
    var w = 64
    while w > 0:
        if t < w:
            red[unsafe_offset=t] = red[unsafe_offset=t] + red[unsafe_offset=t + w]
        barrier()
        w //= 2
    var inv = 1.0 / sqrt(red[unsafe_offset=0] / Float32(D) + eps)
    d = t
    while d < D:
        var p = x.unsafe_offset(base + d)
        p.unsafe_store(p.unsafe_load() * inv * g.unsafe_offset(d).unsafe_load())
        d += 128


def _tg_gelu_kernel(x: FP, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        var v = x.unsafe_offset(i).unsafe_load()
        x.unsafe_offset(i).unsafe_store(0.5 * v * (1.0 + erf(v * 0.7071067811865476)))


def _tg_add_scaled_kernel(dst: FP, src: FP, scale: FP, n: Int32, cols: Int32):
    """dst += src * scale[col] (scale may be null: plain add)."""
    var i = Int(global_idx.x)
    if i < Int(n):
        var v = src.unsafe_offset(i).unsafe_load()
        if Int(cols) > 0:
            v *= scale.unsafe_offset(i % Int(cols)).unsafe_load()
        dst.unsafe_offset(i).unsafe_store(dst.unsafe_offset(i).unsafe_load() + v)


def _tg_copy2d_kernel(src: FP, dst: FP, rows: Int32, cols: Int32, lds: Int32, ldd: Int32):
    var i = Int(global_idx.x)
    if i < Int(rows) * Int(cols):
        var r = i // Int(cols)
        var c = i % Int(cols)
        dst.unsafe_offset(r * Int(ldd) + c).unsafe_store(src.unsafe_offset(r * Int(lds) + c).unsafe_load())


def _tg_freq_embed_kernel(p: FP, o: FP, n: Int32):
    """FrequencyPositionalEmbedding(num_freqs=8, logspace, include_input, no pi): 3 -> 51."""
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    var ob = i * 51
    for d in range(3):
        var x = p.unsafe_offset(i * 3 + d).unsafe_load()
        o.unsafe_offset(ob + d).unsafe_store(x)
        var f: Float32 = 1.0
        for k in range(8):
            var e = x * f
            o.unsafe_offset(ob + 3 + d * 8 + k).unsafe_store(sin(e))
            o.unsafe_offset(ob + 27 + d * 8 + k).unsafe_store(cos(e))
            f *= 2.0


struct TgGpu(Movable):
    var mm: GpuMatmul
    var fast: Bool

    def __init__(out self) raises:
        self.mm = GpuMatmul()
        self.fast = True

    def name(self) raises -> String:
        return self.mm.name()

    def gemm(mut self, a: FP, b: FP, c: FP, m: Int, k: Int, n: Int, lda: Int, ldb: Int, ldc: Int,
             trans_b: Bool, bias: FP, has_bias: Bool = False, accum: Bool = False, alpha: Float32 = 1.0,
             outer: Int = 1, inner: Int = 1, a_so: Int = 0, a_si: Int = 0, b_so: Int = 0, b_si: Int = 0,
             c_so: Int = 0, c_si: Int = 0) raises:
        if m <= 0 or n <= 0:
            return
        var tn = (n + TG_BN - 1) // TG_BN
        var tm = (m + TG_BM - 1) // TG_BM
        var grid = tn * tm * outer * inner
        var ok4 = lda % 4 == 0 and ldb % 4 == 0 and a_so % 4 == 0 and a_si % 4 == 0 and b_so % 4 == 0 and b_si % 4 == 0
        if ok4 and self.fast:
            if trans_b:
                self.mm.ctx.enqueue_function[_tg_gemm4[True]](
                    a, b, c, bias, Int32(m), Int32(k), Int32(n), Int32(lda), Int32(ldb), Int32(ldc), alpha,
                    Int32(1 if accum else 0), Int32(1 if has_bias else 0), Int32(tn), Int32(tn * tm), Int32(inner),
                    Int32(a_so), Int32(a_si), Int32(b_so), Int32(b_si), Int32(c_so), Int32(c_si),
                    grid_dim=grid, block_dim=256)
            else:
                self.mm.ctx.enqueue_function[_tg_gemm4[False]](
                    a, b, c, bias, Int32(m), Int32(k), Int32(n), Int32(lda), Int32(ldb), Int32(ldc), alpha,
                    Int32(1 if accum else 0), Int32(1 if has_bias else 0), Int32(tn), Int32(tn * tm), Int32(inner),
                    Int32(a_so), Int32(a_si), Int32(b_so), Int32(b_si), Int32(c_so), Int32(c_si),
                    grid_dim=grid, block_dim=256)
            return
        if trans_b:
            self.mm.ctx.enqueue_function[_tg_gemm[True]](
                a, b, c, bias, Int32(m), Int32(k), Int32(n), Int32(lda), Int32(ldb), Int32(ldc), alpha,
                Int32(1 if accum else 0), Int32(1 if has_bias else 0), Int32(tn), Int32(tn * tm), Int32(inner),
                Int32(a_so), Int32(a_si), Int32(b_so), Int32(b_si), Int32(c_so), Int32(c_si),
                grid_dim=grid, block_dim=256)
        else:
            self.mm.ctx.enqueue_function[_tg_gemm[False]](
                a, b, c, bias, Int32(m), Int32(k), Int32(n), Int32(lda), Int32(ldb), Int32(ldc), alpha,
                Int32(1 if accum else 0), Int32(1 if has_bias else 0), Int32(tn), Int32(tn * tm), Int32(inner),
                Int32(a_so), Int32(a_si), Int32(b_so), Int32(b_si), Int32(c_so), Int32(c_si),
                grid_dim=grid, block_dim=256)

    def softmax(mut self, x: FP, rows: Int, cols: Int, ld: Int) raises:
        """Softmax over the first `cols` entries of each row (row stride `ld`)."""
        self.mm.ctx.enqueue_function[_tg_softmax_kernel](x, Int32(cols), Int32(ld), grid_dim=rows, block_dim=TG_RT)

    def head_rms(mut self, x: FP, g: FP, rows: Int, heads: Int, hd: Int, ld: Int, eps: Float32) raises:
        self.mm.ctx.enqueue_function[_tg_head_rms_kernel](x, g, Int32(heads), Int32(hd), Int32(ld), eps,
                                                          grid_dim=rows * heads, block_dim=128)

    def gelu(mut self, x: FP, n: Int) raises:
        self.mm.ctx.enqueue_function[_tg_gelu_kernel](x, Int32(n), grid_dim=(n + TG_EW - 1) // TG_EW,
                                                      block_dim=TG_EW)

    def add_scaled(mut self, dst: FP, src: FP, scale: FP, n: Int, cols: Int) raises:
        """dst += src * scale[col]; cols = 0 for a plain add."""
        self.mm.ctx.enqueue_function[_tg_add_scaled_kernel](dst, src, scale, Int32(n), Int32(cols),
                                                            grid_dim=(n + TG_EW - 1) // TG_EW, block_dim=TG_EW)

    def copy2d(mut self, src: FP, dst: FP, rows: Int, cols: Int, lds: Int, ldd: Int) raises:
        var n = rows * cols
        if n <= 0:
            return
        self.mm.ctx.enqueue_function[_tg_copy2d_kernel](src, dst, Int32(rows), Int32(cols), Int32(lds), Int32(ldd),
                                                        grid_dim=(n + TG_EW - 1) // TG_EW, block_dim=TG_EW)

    def freq_embed(mut self, p: FP, o: FP, n: Int) raises:
        self.mm.ctx.enqueue_function[_tg_freq_embed_kernel](p, o, Int32(n), grid_dim=(n + TG_EW - 1) // TG_EW,
                                                            block_dim=TG_EW)

    def layer_norm(mut self, x: FP, g: FP, b: FP, o: FP, rows: Int, n: Int, eps: Float32) raises:
        self.mm.layer_norm_dev(x, g, b, o, rows, n, eps, True)
