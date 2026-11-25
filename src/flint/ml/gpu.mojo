"""MAX GPU backend for the ONNX interpreter.

Weights upload once, as [k, n] row-major device buffers (Gemm's transB and Conv weights are
transposed on upload), and stay resident. Matrix products use a 128 x 128 tiled SGEMM (split-K
for a handful of rows); a sparse-conv gather-GEMM, fused attention, element-wise, LayerNorm,
row-gather and strided-copy (Split / Concat) kernels complete the set. Activations live in a
pool of device buffers (`alloc` / `release`), so a chain of GPU ops never touches the host:
`OnnxModel` downloads a value only when a CPU op reads it (`download`, one synchronisation).
"""

from std.collections import Dict
from std.ffi import external_call
from std.math import exp, erf, tanh, sqrt
from max.gpu import global_idx, thread_idx, block_idx, barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.memory import stack_allocation
from flint.ml.tensor import fptr_at

comptime BLOCK: Int = 128
comptime ROWS: Int = 8
comptime SLICE_K: Int = 64


def _mm_partial(a: Pointer[Float32, MutAnyOrigin], b: Pointer[Float32, MutAnyOrigin],
                part: Pointer[Float32, MutAnyOrigin], m: Int32, k: Int32, n: Int32, slices: Int32):
    """part[s, i, j] = sum over k in slice s of A[i, k] * B[k, j]."""
    var tid = Int(global_idx.x)
    var nn = Int(n)
    var j = tid % nn
    var s = tid // nn
    if s >= Int(slices):
        return
    var mm = Int(m)
    var kn = Int(k)
    var k0 = s * SLICE_K
    var k1 = min(kn, k0 + SLICE_K)
    var i0 = 0
    while i0 < mm:
        var acc = SIMD[DType.float32, ROWS](0)
        var rows = min(ROWS, mm - i0)
        for kk in range(k0, k1):
            var w = b.unsafe_offset(kk * nn + j).unsafe_load()
            var x = SIMD[DType.float32, ROWS](0)
            comptime for r in range(ROWS):
                if r < rows:
                    x[r] = a.unsafe_offset((i0 + r) * kn + kk).unsafe_load()
            acc += x * w
        comptime for r in range(ROWS):
            if r < rows:
                part.unsafe_offset((s * mm + i0 + r) * nn + j).unsafe_store(acc[r])
        i0 += ROWS


def _mm_reduce(part: Pointer[Float32, MutAnyOrigin], c: Pointer[Float32, MutAnyOrigin],
               bias: Pointer[Float32, MutAnyOrigin], m: Int32, n: Int32, slices: Int32, has_bias: Int32):
    var e = Int(global_idx.x)
    var total = Int(m) * Int(n)
    if e >= total:
        return
    var acc: Float32 = 0
    if has_bias != 0:
        acc = bias.unsafe_offset(e % Int(n)).unsafe_load()
    for s in range(Int(slices)):
        acc += part.unsafe_offset(s * total + e).unsafe_load()
    c.unsafe_offset(e).unsafe_store(acc)


comptime TBM: Int = 128
comptime TBN: Int = 128
comptime TBK: Int = 8
comptime TT: Int = 8          # outputs per thread along each axis (16 x 16 threads)
comptime V8G = SIMD[DType.float32, 8]


def _mm_tiled[GATHER: Bool](a: Pointer[Float32, MutAnyOrigin], b: Pointer[Float32, MutAnyOrigin],
              c: Pointer[Float32, MutAnyOrigin], bias: Pointer[Float32, MutAnyOrigin],
              m: Int32, k: Int32, n: Int32, has_bias: Int32, tiles_n: Int32,
              idx: Pointer[Float32, MutAnyOrigin], cin: Int32, slots: Int32):
    """C = A[m,k] @ B[k,n] (+ bias): 128 x 128 output tile per block, 8 x 8 per thread, K in
    steps of 8 through shared memory (A stored transposed so both operands read as rows).
    GATHER: A is implicit, A[r, j * cin + ch] = x[idx[r, j], ch] with x = `a` ([rows, cin]) and
    idx [m, slots] row numbers stored as floats (sparse-conv neighbour gather)."""
    var As = stack_allocation[TBK * TBM, Float32, address_space=AddressSpace.SHARED]()
    var Bs = stack_allocation[TBK * TBN, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var tx = t % 16
    var ty = t // 16
    var blk = Int(block_idx.x)
    var bn = blk % Int(tiles_n)
    var bm = blk // Int(tiles_n)
    var mm = Int(m)
    var kn = Int(k)
    var nn = Int(n)
    var row0 = bm * TBM
    var col0 = bn * TBN
    var acc0 = V8G(0)
    var acc1 = V8G(0)
    var acc2 = V8G(0)
    var acc3 = V8G(0)
    var acc4 = V8G(0)
    var acc5 = V8G(0)
    var acc6 = V8G(0)
    var acc7 = V8G(0)
    var k0 = 0
    while k0 < kn:
        # A tile [128 rows, 8 k] -> As[unsafe_offset=k][row]; 4 elements per thread
        comptime for q in range(4):
            var e = t * 4 + q
            var r = e // TBK
            var kk = e % TBK
            var gr = row0 + r
            var gk = k0 + kk
            var v: Float32 = 0
            if gr < mm and gk < kn:
                comptime if GATHER:
                    var ci = Int(cin)
                    var row = Int(idx.unsafe_offset(gr * Int(slots) + gk // ci).unsafe_load())
                    v = a.unsafe_offset(row * ci + gk % ci).unsafe_load()
                else:
                    v = a.unsafe_offset(gr * kn + gk).unsafe_load()
            As[unsafe_offset=kk * TBM + r] = v
        # B tile [8 k, 128 cols] -> Bs[unsafe_offset=k][col]
        comptime for q in range(4):
            var e = t * 4 + q
            var kk = e // TBN
            var cc = e % TBN
            var gk = k0 + kk
            var gc = col0 + cc
            var v: Float32 = 0
            if gk < kn and gc < nn:
                v = b.unsafe_offset(gk * nn + gc).unsafe_load()
            Bs[unsafe_offset=kk * TBN + cc] = v
        barrier()
        comptime for kk in range(TBK):
            var bv = V8G(0)
            comptime for j in range(8):
                bv[j] = Bs[unsafe_offset=kk * TBN + tx * TT + j]
            acc0 += As[unsafe_offset=kk * TBM + ty * TT + 0] * bv
            acc1 += As[unsafe_offset=kk * TBM + ty * TT + 1] * bv
            acc2 += As[unsafe_offset=kk * TBM + ty * TT + 2] * bv
            acc3 += As[unsafe_offset=kk * TBM + ty * TT + 3] * bv
            acc4 += As[unsafe_offset=kk * TBM + ty * TT + 4] * bv
            acc5 += As[unsafe_offset=kk * TBM + ty * TT + 5] * bv
            acc6 += As[unsafe_offset=kk * TBM + ty * TT + 6] * bv
            acc7 += As[unsafe_offset=kk * TBM + ty * TT + 7] * bv
        barrier()
        k0 += TBK
    var bvec = V8G(0)
    if has_bias != 0:
        comptime for j in range(8):
            var gc = col0 + tx * TT + j
            if gc < nn:
                bvec[j] = bias.unsafe_offset(gc).unsafe_load()
    comptime for i in range(8):
        var gr = row0 + ty * TT + i
        if gr < mm:
            var accv: V8G
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
            accv += bvec
            comptime for j in range(8):
                var gc = col0 + tx * TT + j
                if gc < nn:
                    c.unsafe_offset(gr * nn + gc).unsafe_store(accv[j])


comptime AT_ROWS: Int = 128     # query rows (threads) per block
comptime AT_KEYS: Int = 32      # keys per shared-memory tile


def _attn_kernel[DP: Int](q: Pointer[Float32, MutAnyOrigin], k: Pointer[Float32, MutAnyOrigin],
                          v: Pointer[Float32, MutAnyOrigin], o: Pointer[Float32, MutAnyOrigin],
                          L: Int32, S: Int32, H: Int32, D: Int32, scale: Float32, row_blocks: Int32):
    """softmax(scale q k^T) v for one query row per thread (online softmax), K / V tiles of 32
    keys staged in shared memory; layouts Q [X, L, H, D], K / V [X, S, H, D]; head dim D <= DP."""
    var Ks = stack_allocation[AT_KEYS * DP, Float32, address_space=AddressSpace.SHARED]()
    var Vs = stack_allocation[AT_KEYS * DP, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var rb = blk % Int(row_blocks)
    var xh = blk // Int(row_blocks)
    var hh = Int(H)
    var x = xh // hh
    var h = xh % hh
    var ll = Int(L)
    var ss = Int(S)
    var dd = Int(D)
    var l = rb * AT_ROWS + t
    var live = l < ll
    var qv = SIMD[DType.float32, DP](0)
    if live:
        var qb = ((x * ll + l) * hh + h) * dd
        for d in range(dd):
            qv[d] = q.unsafe_offset(qb + d).unsafe_load() * scale
    var acc = SIMD[DType.float32, DP](0)
    var mx: Float32 = -3.0e38
    var den: Float32 = 0
    var s0 = 0
    while s0 < ss:
        var e = t
        while e < AT_KEYS * DP:
            var j = e // DP
            var d = e % DP
            var kv: Float32 = 0
            var vv: Float32 = 0
            if s0 + j < ss and d < dd:
                var kb = ((x * ss + s0 + j) * hh + h) * dd + d
                kv = k.unsafe_offset(kb).unsafe_load()
                vv = v.unsafe_offset(kb).unsafe_load()
            Ks[unsafe_offset=e] = kv
            Vs[unsafe_offset=e] = vv
            e += AT_ROWS
        barrier()
        if live:
            var nk = min(AT_KEYS, ss - s0)
            for j in range(nk):
                var kr = SIMD[DType.float32, DP](0)
                var vr = SIMD[DType.float32, DP](0)
                comptime for d in range(DP):
                    kr[d] = Ks[unsafe_offset=j * DP + d]
                    vr[d] = Vs[unsafe_offset=j * DP + d]
                var sc = (qv * kr).reduce_add()
                if sc > mx:
                    var corr = exp(mx - sc)
                    acc = acc * corr
                    den = den * corr
                    mx = sc
                var p = exp(sc - mx)
                den += p
                acc += vr * p
        barrier()
        s0 += AT_KEYS
    if live:
        var ob = ((x * ll + l) * hh + h) * dd
        var inv = 1.0 / den
        for d in range(dd):
            o.unsafe_offset(ob + d).unsafe_store(acc[d] * inv)


comptime FP = Pointer[Float32, MutAnyOrigin]
comptime EW_BLOCK: Int = 256
comptime LN_THREADS: Int = 256

# element-wise codes (match flint.ml.ops B_* / U_*)
comptime GB_ADD: Int = 0
comptime GB_SUB: Int = 1
comptime GB_MUL: Int = 2
comptime GB_DIV: Int = 3
comptime GB_MAX: Int = 5
comptime GB_MIN: Int = 11
comptime GU_SQRT: Int = 0
comptime GU_RELU: Int = 1
comptime GU_ABS: Int = 5
comptime GU_NEG: Int = 6
comptime GU_ERF: Int = 9
comptime GU_EXP: Int = 10
comptime GU_TANH: Int = 11
comptime GU_SIGMOID: Int = 12
comptime GU_RECIP: Int = 13


def gpu_binary_supported(code: Int) -> Bool:
    return code == GB_ADD or code == GB_SUB or code == GB_MUL or code == GB_DIV or code == GB_MAX or code == GB_MIN


def gpu_unary_supported(code: Int) -> Bool:
    return code == GU_SQRT or code == GU_RELU or code == GU_ABS or code == GU_NEG or code == GU_ERF \
        or code == GU_EXP or code == GU_TANH or code == GU_SIGMOID or code == GU_RECIP


def _ew_binary_kernel(a: FP, b: FP, o: FP, n: Int32, period: Int32, mode: Int32, code: Int32,
                      scalar: Float32, swap: Int32):
    """o[i] = a[i] op b[...]: mode 0 b[i], 1 the scalar, 2 b[i % period]; swap: b op a."""
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    var x = a.unsafe_offset(i).unsafe_load()
    var y = scalar
    if mode == 0:
        y = b.unsafe_offset(i).unsafe_load()
    elif mode == 2:
        y = b.unsafe_offset(i % Int(period)).unsafe_load()
    if swap != 0:
        var t = x
        x = y
        y = t
    var r: Float32
    var c = Int(code)
    if c == GB_ADD:
        r = x + y
    elif c == GB_SUB:
        r = x - y
    elif c == GB_MUL:
        r = x * y
    elif c == GB_DIV:
        r = x / y
    elif c == GB_MAX:
        r = max(x, y)
    else:
        r = min(x, y)
    o.unsafe_offset(i).unsafe_store(r)


def _ew_unary_kernel(a: FP, o: FP, n: Int32, code: Int32):
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    var x = a.unsafe_offset(i).unsafe_load()
    var r: Float32
    var c = Int(code)
    if c == GU_ERF:
        r = erf(x)
    elif c == GU_RELU:
        r = max(x, Float32(0))
    elif c == GU_SIGMOID:
        r = 1.0 / (1.0 + exp(-x))
    elif c == GU_TANH:
        r = tanh(x)
    elif c == GU_EXP:
        r = exp(x)
    elif c == GU_SQRT:
        r = sqrt(x)
    elif c == GU_NEG:
        r = -x
    elif c == GU_ABS:
        r = abs(x)
    else:
        r = 1.0 / x
    o.unsafe_offset(i).unsafe_store(r)


def _layer_norm_kernel(x: FP, g: FP, b: FP, o: FP, n: Int32, eps: Float32, has_b: Int32):
    """One block per row: (x - mean) / sqrt(var + eps) * g (+ b) over the last axis."""
    var sh = stack_allocation[LN_THREADS, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var nn = Int(n)
    var base = Int(block_idx.x) * nn
    var s: Float32 = 0
    var i = t
    while i < nn:
        s += x.unsafe_offset(base + i).unsafe_load()
        i += LN_THREADS
    sh[unsafe_offset=t] = s
    barrier()
    var w = LN_THREADS // 2
    while w > 0:
        if t < w:
            sh[unsafe_offset=t] = sh[unsafe_offset=t] + sh[unsafe_offset=t + w]
        barrier()
        w //= 2
    var mean = sh[unsafe_offset=0] / Float32(nn)
    barrier()
    var q: Float32 = 0
    i = t
    while i < nn:
        var d = x.unsafe_offset(base + i).unsafe_load() - mean
        q += d * d
        i += LN_THREADS
    sh[unsafe_offset=t] = q
    barrier()
    w = LN_THREADS // 2
    while w > 0:
        if t < w:
            sh[unsafe_offset=t] = sh[unsafe_offset=t] + sh[unsafe_offset=t + w]
        barrier()
        w //= 2
    var inv = 1.0 / sqrt(sh[unsafe_offset=0] / Float32(nn) + eps)
    i = t
    while i < nn:
        var y = (x.unsafe_offset(base + i).unsafe_load() - mean) * inv * g.unsafe_offset(i).unsafe_load()
        if has_b != 0:
            y += b.unsafe_offset(i).unsafe_load()
        o.unsafe_offset(base + i).unsafe_store(y)
        i += LN_THREADS


def _copy_kernel(src: FP, dst: FP, total: Int32, chunk: Int32, s_stride: Int32, s_off: Int32, d_stride: Int32,
                 d_off: Int32):
    """Strided block copy: element (o, c) of `total` = outer x chunk goes from src[o * s_stride +
    s_off + c] to dst[o * d_stride + d_off + c] (Split / Concat / Slice along one axis)."""
    var e = Int(global_idx.x)
    if e >= Int(total):
        return
    var ch = Int(chunk)
    var o = e // ch
    var c = e % ch
    dst.unsafe_offset(o * Int(d_stride) + Int(d_off) + c).unsafe_store(
        src.unsafe_offset(o * Int(s_stride) + Int(s_off) + c).unsafe_load())


def _gather_rows_kernel(src: FP, idx: FP, dst: FP, total: Int32, row: Int32):
    var e = Int(global_idx.x)
    if e >= Int(total):
        return
    var rw = Int(row)
    var r = e // rw
    var src_row = Int(idx.unsafe_offset(r).unsafe_load())
    dst.unsafe_offset(e).unsafe_store(src.unsafe_offset(src_row * rw + e % rw).unsafe_load())


comptime DA_THREADS: Int = 128
comptime DA_HD: Int = 64          # OPT head dim
comptime DA_MAXT: Int = 2048      # positions (prefix + generated) a query may see


def _dec_attn_kernel(q: FP, o: FP, pk: FP, pv: FP, gk: FP, gv: FP, anc: FP, prefix_len: Int32, causal: Int32,
                     steps: Int32, beams_cap: Int32, hid: Int32, heads: Int32, scale: Float32, anc_stride: Int32):
    """OPT decoder attention, one block per (row, head): the query attends to the shared prefix
    (causally in the prefix pass) and to its beam's generated history, read through the ancestry
    table anc[row, g] (slot beam of generated step g, stored as floats)."""
    var qs = stack_allocation[DA_HD, Float32, address_space=AddressSpace.SHARED]()
    var sc = stack_allocation[DA_MAXT, Float32, address_space=AddressSpace.SHARED]()
    var red = stack_allocation[DA_THREADS, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var hh = Int(heads)
    var row = Int(block_idx.x) // hh
    var h = Int(block_idx.x) % hh
    var hd = Int(hid)
    var col = h * DA_HD
    if t < DA_HD:
        qs[unsafe_offset=t] = q.unsafe_offset(row * hd + col + t).unsafe_load() * scale
    barrier()
    var npre = Int(prefix_len)
    if causal != 0:
        npre = row + 1
    var total = npre + Int(steps)
    var bc = Int(beams_cap)
    var ast = Int(anc_stride)
    var mx: Float32 = -3.4e38
    var i = t
    while i < total:
        var kp: FP
        if i < npre:
            kp = pk.unsafe_offset(i * hd + col)
        else:
            var g = i - npre
            var b = Int(anc.unsafe_offset(row * ast + g).unsafe_load())
            kp = gk.unsafe_offset((g * bc + b) * hd + col)
        var acc = SIMD[DType.float32, 8](0)
        comptime for d8 in range(DA_HD // 8):
            var kv = kp.unsafe_offset(d8 * 8).unsafe_load[width=8]()
            var qv = SIMD[DType.float32, 8](0)
            comptime for e in range(8):
                qv[e] = qs[unsafe_offset=d8 * 8 + e]
            acc += kv * qv
        var s = acc.reduce_add()
        sc[unsafe_offset=i] = s
        mx = max(mx, s)
        i += DA_THREADS
    red[unsafe_offset=t] = mx
    barrier()
    var w = DA_THREADS // 2
    while w > 0:
        if t < w:
            red[unsafe_offset=t] = max(red[unsafe_offset=t], red[unsafe_offset=t + w])
        barrier()
        w //= 2
    var m = red[unsafe_offset=0]
    barrier()
    var sum: Float32 = 0
    i = t
    while i < total:
        var e = exp(sc[unsafe_offset=i] - m)
        sc[unsafe_offset=i] = e
        sum += e
        i += DA_THREADS
    red[unsafe_offset=t] = sum
    barrier()
    w = DA_THREADS // 2
    while w > 0:
        if t < w:
            red[unsafe_offset=t] = red[unsafe_offset=t] + red[unsafe_offset=t + w]
        barrier()
        w //= 2
    var inv = 1.0 / red[unsafe_offset=0]
    barrier()
    # output: thread (half, d) sums every other position for dim d
    var d = t % DA_HD
    var half = t // DA_HD
    var acc_o: Float32 = 0
    i = half
    while i < total:
        var vp: FP
        if i < npre:
            vp = pv.unsafe_offset(i * hd + col)
        else:
            var g = i - npre
            var b = Int(anc.unsafe_offset(row * ast + g).unsafe_load())
            vp = gv.unsafe_offset((g * bc + b) * hd + col)
        acc_o += sc[unsafe_offset=i] * vp.unsafe_offset(d).unsafe_load()
        i += DA_THREADS // DA_HD
    red[unsafe_offset=t] = acc_o
    barrier()
    if t < DA_HD:
        var r = red[unsafe_offset=t]
        comptime for k in range(1, DA_THREADS // DA_HD):
            r += red[unsafe_offset=t + k * DA_HD]
        o.unsafe_offset(row * hd + col + t).unsafe_store(r * inv)


@always_inline
def _dptr(buf: DeviceBuffer[DType.float32]) -> Pointer[Float32, MutAnyOrigin]:
    return Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(buf.unsafe_ptr()))


struct GpuMatmul(Movable):
    """Resident weights on one MAX `DeviceContext`, keyed by (value id, transposed)."""
    var ctx: DeviceContext
    var weights: Dict[Int, Int]
    var buffers: List[DeviceBuffer[DType.float32]]
    var uploaded_bytes: Int
    var cap_a: Int
    var cap_c: Int
    var cap_p: Int
    var host_a: HostBuffer[DType.float32]
    var host_c: HostBuffer[DType.float32]
    var dev_a: DeviceBuffer[DType.float32]
    var dev_c: DeviceBuffer[DType.float32]
    var dev_p: DeviceBuffer[DType.float32]
    var products: Int
    var att_cap: Int
    var cap_i: Int
    var host_i: HostBuffer[DType.float32]
    var dev_i: DeviceBuffer[DType.float32]
    var att_dev: List[DeviceBuffer[DType.float32]]
    var att_host: List[HostBuffer[DType.float32]]
    var pool: List[DeviceBuffer[DType.float32]]   # device-resident activations (`alloc` / `release`)
    var pool_cap: List[Int]
    var pool_used: List[Bool]
    var downloads: Int
    var uploads: Int

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.weights = Dict[Int, Int]()
        self.buffers = List[DeviceBuffer[DType.float32]]()
        self.uploaded_bytes = 0
        self.cap_a = 1 << 16
        self.cap_c = 1 << 16
        self.cap_p = 1 << 20
        self.host_a = self.ctx.enqueue_create_host_buffer[DType.float32](self.cap_a)
        self.host_c = self.ctx.enqueue_create_host_buffer[DType.float32](self.cap_c)
        self.dev_a = self.ctx.enqueue_create_buffer[DType.float32](self.cap_a)
        self.dev_c = self.ctx.enqueue_create_buffer[DType.float32](self.cap_c)
        self.dev_p = self.ctx.enqueue_create_buffer[DType.float32](self.cap_p)
        self.products = 0
        self.att_cap = 0
        self.cap_i = 1 << 16
        self.host_i = self.ctx.enqueue_create_host_buffer[DType.float32](self.cap_i)
        self.dev_i = self.ctx.enqueue_create_buffer[DType.float32](self.cap_i)
        self.att_dev = List[DeviceBuffer[DType.float32]]()
        self.att_host = List[HostBuffer[DType.float32]]()
        self.pool = List[DeviceBuffer[DType.float32]]()
        self.pool_cap = List[Int]()
        self.pool_used = List[Bool]()
        self.downloads = 0
        self.uploads = 0

    def attention(mut self, q: Int, k: Int, v: Int, o: Int, X: Int, L: Int, S: Int, H: Int, D: Int,
                  scale: Float32) raises -> Bool:
        """Fused softmax(scale Q K^T) V on the GPU (host addresses in and out); False when the
        head dim is over 64."""
        if D > 64:
            return False
        var nq = X * L * H * D
        var nk = X * S * H * D
        var need = max(nq, nk)
        if need > self.att_cap:
            self.ctx.synchronize()
            self.att_cap = need
            self.att_dev = List[DeviceBuffer[DType.float32]]()
            self.att_host = List[HostBuffer[DType.float32]]()
            for _ in range(4):
                self.att_dev.append(self.ctx.enqueue_create_buffer[DType.float32](need))
                self.att_host.append(self.ctx.enqueue_create_host_buffer[DType.float32](need))
            self.ctx.synchronize()
        _ = external_call["memcpy", Int](Int(self.att_host[0].unsafe_ptr()), q, nq * 4)
        _ = external_call["memcpy", Int](Int(self.att_host[1].unsafe_ptr()), k, nk * 4)
        _ = external_call["memcpy", Int](Int(self.att_host[2].unsafe_ptr()), v, nk * 4)
        self.ctx.enqueue_copy(self.att_dev[0].create_sub_buffer[DType.float32](0, nq), self.att_host[0].create_sub_buffer[DType.float32](0, nq))
        self.ctx.enqueue_copy(self.att_dev[1].create_sub_buffer[DType.float32](0, nk), self.att_host[1].create_sub_buffer[DType.float32](0, nk))
        self.ctx.enqueue_copy(self.att_dev[2].create_sub_buffer[DType.float32](0, nk), self.att_host[2].create_sub_buffer[DType.float32](0, nk))
        var rbk = (L + AT_ROWS - 1) // AT_ROWS
        var grid = X * H * rbk
        if D <= 16:
            self.ctx.enqueue_function[_attn_kernel[16]](
                _dptr(self.att_dev[0]), _dptr(self.att_dev[1]), _dptr(self.att_dev[2]), _dptr(self.att_dev[3]),
                Int32(L), Int32(S), Int32(H), Int32(D), scale, Int32(rbk), grid_dim=grid, block_dim=AT_ROWS)
        elif D <= 32:
            self.ctx.enqueue_function[_attn_kernel[32]](
                _dptr(self.att_dev[0]), _dptr(self.att_dev[1]), _dptr(self.att_dev[2]), _dptr(self.att_dev[3]),
                Int32(L), Int32(S), Int32(H), Int32(D), scale, Int32(rbk), grid_dim=grid, block_dim=AT_ROWS)
        else:
            self.ctx.enqueue_function[_attn_kernel[64]](
                _dptr(self.att_dev[0]), _dptr(self.att_dev[1]), _dptr(self.att_dev[2]), _dptr(self.att_dev[3]),
                Int32(L), Int32(S), Int32(H), Int32(D), scale, Int32(rbk), grid_dim=grid, block_dim=AT_ROWS)
        self.ctx.enqueue_copy(self.att_host[3].create_sub_buffer[DType.float32](0, nq), self.att_dev[3].create_sub_buffer[DType.float32](0, nq))
        self.ctx.synchronize()
        _ = external_call["memcpy", Int](o, Int(self.att_host[3].unsafe_ptr()), nq * 4)
        return True

    def name(self) raises -> String:
        return self.ctx.name()

    def weight(mut self, key: Int, src: Int, rows: Int, cols: Int, transpose: Bool) raises -> Int:
        """Index of the device buffer holding the float32 matrix at `src` ([rows, cols]), uploaded
        as [cols, rows] when `transpose`."""
        var found = self.weights.get(key)
        if found:
            return found.value()
        var n = rows * cols
        var buf = self.ctx.enqueue_create_buffer[DType.float32](n)
        with buf.map_to_host() as h:
            var p = fptr_at(src)
            if transpose:
                for r in range(rows):
                    for c in range(cols):
                        h[c * rows + r] = p.unsafe_offset(r * cols + c).unsafe_load[alignment=1]()
            else:
                var dst = Int(h.unsafe_ptr())
                _ = external_call["memcpy", Int](dst, src, n * 4)
        var idx = len(self.buffers)
        self.buffers.append(buf^)
        self.weights[key] = idx
        self.uploaded_bytes += n * 4
        return idx

    def _reserve(mut self, a: Int, c: Int, p: Int) raises:
        if a > self.cap_a:
            self.ctx.synchronize()
            self.cap_a = a
            self.host_a = self.ctx.enqueue_create_host_buffer[DType.float32](a)
            self.dev_a = self.ctx.enqueue_create_buffer[DType.float32](a)
        if c > self.cap_c:
            self.ctx.synchronize()
            self.cap_c = c
            self.host_c = self.ctx.enqueue_create_host_buffer[DType.float32](c)
            self.dev_c = self.ctx.enqueue_create_buffer[DType.float32](c)
        if p > self.cap_p:
            self.ctx.synchronize()
            self.cap_p = p
            self.dev_p = self.ctx.enqueue_create_buffer[DType.float32](p)

    def gather_matmul(mut self, x: Int, rows: Int, cin: Int, idx: List[Float32], m: Int, slots: Int,
                      w: Int, c: Int, n: Int) raises:
        """C[m,n] = G @ W[slots * cin, n] with G[r, j * cin + ch] = X[idx[r, j], ch]: the neighbour
        gather of a submanifold sparse conv done inside the tiled GEMM (X [rows, cin] host)."""
        self._reserve(rows * cin, m * n, 1)
        if m * slots > self.cap_i:
            self.ctx.synchronize()
            self.cap_i = m * slots
            self.host_i = self.ctx.enqueue_create_host_buffer[DType.float32](self.cap_i)
            self.dev_i = self.ctx.enqueue_create_buffer[DType.float32](self.cap_i)
        _ = external_call["memcpy", Int](Int(self.host_a.unsafe_ptr()), x, rows * cin * 4)
        _ = external_call["memcpy", Int](Int(self.host_i.unsafe_ptr()), Int(idx.unsafe_ptr()), m * slots * 4)
        self.ctx.enqueue_copy(self.dev_a.create_sub_buffer[DType.float32](0, rows * cin),
                              self.host_a.create_sub_buffer[DType.float32](0, rows * cin))
        self.ctx.enqueue_copy(self.dev_i.create_sub_buffer[DType.float32](0, m * slots),
                              self.host_i.create_sub_buffer[DType.float32](0, m * slots))
        var tn = (n + TBN - 1) // TBN
        var tm = (m + TBM - 1) // TBM
        var k = slots * cin
        self.ctx.enqueue_function[_mm_tiled[True]](
            _dptr(self.dev_a), _dptr(self.buffers[w]), _dptr(self.dev_c), _dptr(self.dev_c), Int32(m), Int32(k),
            Int32(n), Int32(0), Int32(tn), _dptr(self.dev_i), Int32(cin), Int32(slots),
            grid_dim=tn * tm, block_dim=256)
        self.ctx.enqueue_copy(self.host_c.create_sub_buffer[DType.float32](0, m * n),
                              self.dev_c.create_sub_buffer[DType.float32](0, m * n))
        self.ctx.synchronize()
        _ = external_call["memcpy", Int](c, Int(self.host_c.unsafe_ptr()), m * n * 4)
        self.products += 1

    def matmul(mut self, a: Int, w: Int, c: Int, bias: Int, bias_key: Int, m: Int, k: Int, n: Int) raises:
        """C[m,n] = A[m,k] @ W[k,n] (+ bias[n]); A and C are host addresses, W a resident buffer."""
        var slices = (k + SLICE_K - 1) // SLICE_K
        self._reserve(m * k, m * n, slices * m * n if m < 64 else 1)
        var bidx = -1
        if bias != 0:
            bidx = self.weight(bias_key, bias, 1, n, False)
        _ = external_call["memcpy", Int](Int(self.host_a.unsafe_ptr()), a, m * k * 4)
        self.ctx.enqueue_copy(self.dev_a.create_sub_buffer[DType.float32](0, m * k),
                              self.host_a.create_sub_buffer[DType.float32](0, m * k))
        var pw = _dptr(self.buffers[w])
        var pb = _dptr(self.dev_c)
        if bidx >= 0:
            pb = _dptr(self.buffers[bidx])
        if m >= 64:
            var tn = (n + TBN - 1) // TBN
            var tm = (m + TBM - 1) // TBM
            self.ctx.enqueue_function[_mm_tiled[False]](
                _dptr(self.dev_a), pw, _dptr(self.dev_c), pb, Int32(m), Int32(k), Int32(n),
                Int32(1 if bidx >= 0 else 0), Int32(tn), _dptr(self.dev_a), Int32(1), Int32(1),
                grid_dim=tn * tm, block_dim=256)
            self.ctx.enqueue_copy(self.host_c.create_sub_buffer[DType.float32](0, m * n),
                                  self.dev_c.create_sub_buffer[DType.float32](0, m * n))
            self.ctx.synchronize()
            _ = external_call["memcpy", Int](c, Int(self.host_c.unsafe_ptr()), m * n * 4)
            self.products += 1
            return
        var threads = slices * n
        self.ctx.enqueue_function[_mm_partial](
            _dptr(self.dev_a), pw, _dptr(self.dev_p), Int32(m), Int32(k), Int32(n), Int32(slices),
            grid_dim=(threads + BLOCK - 1) // BLOCK, block_dim=BLOCK)
        self.ctx.enqueue_function[_mm_reduce](
            _dptr(self.dev_p), _dptr(self.dev_c), pb, Int32(m), Int32(n), Int32(slices),
            Int32(1 if bidx >= 0 else 0),
            grid_dim=(m * n + BLOCK - 1) // BLOCK, block_dim=BLOCK)
        self.ctx.enqueue_copy(self.host_c.create_sub_buffer[DType.float32](0, m * n),
                              self.dev_c.create_sub_buffer[DType.float32](0, m * n))
        self.ctx.synchronize()
        _ = external_call["memcpy", Int](c, Int(self.host_c.unsafe_ptr()), m * n * 4)
        self.products += 1

    # ------------------------------------------------------------------ device-resident values

    def alloc(mut self, n: Int) raises -> Int:
        """A free pooled device buffer of at least `n` floats (best fit, else a new one)."""
        var best = -1
        for s in range(len(self.pool)):
            if not self.pool_used[s] and self.pool_cap[s] >= n and (best < 0 or self.pool_cap[s] < self.pool_cap[best]):
                best = s
        if best < 0:
            best = len(self.pool)
            var cap = max(n, 1)
            self.pool.append(self.ctx.enqueue_create_buffer[DType.float32](cap))
            self.pool_cap.append(cap)
            self.pool_used.append(False)
        self.pool_used[best] = True
        return best

    def release(mut self, slot: Int):
        """Returns a pooled buffer; later kernels on the in-order stream may reuse it safely."""
        if slot >= 0 and slot < len(self.pool_used):
            self.pool_used[slot] = False

    def sync(mut self) raises:
        self.ctx.synchronize()

    def ptr(self, slot: Int) -> FP:
        return _dptr(self.pool[slot])

    def wptr(self, w: Int) -> FP:
        return _dptr(self.buffers[w])

    def upload(mut self, slot: Int, src: Int, n: Int) raises:
        self._reserve(n, 1, 1)
        self.ctx.synchronize()
        _ = external_call["memcpy", Int](Int(self.host_a.unsafe_ptr()), src, n * 4)
        self.ctx.enqueue_copy(self.pool[slot].create_sub_buffer[DType.float32](0, n),
                              self.host_a.create_sub_buffer[DType.float32](0, n))
        self.ctx.synchronize()
        self.uploads += 1

    def download(mut self, slot: Int, dst: Int, n: Int) raises:
        self._reserve(1, n, 1)
        self.ctx.enqueue_copy(self.host_c.create_sub_buffer[DType.float32](0, n),
                              self.pool[slot].create_sub_buffer[DType.float32](0, n))
        self.ctx.synchronize()
        _ = external_call["memcpy", Int](dst, Int(self.host_c.unsafe_ptr()), n * 4)
        self.downloads += 1

    def _launch_mm(mut self, pa: FP, pw: FP, pc: FP, pb: FP, has_bias: Bool, m: Int, k: Int, n: Int) raises:
        if m >= 64:
            var tn = (n + TBN - 1) // TBN
            var tm = (m + TBM - 1) // TBM
            self.ctx.enqueue_function[_mm_tiled[False]](
                pa, pw, pc, pb, Int32(m), Int32(k), Int32(n), Int32(1 if has_bias else 0), Int32(tn), pa,
                Int32(1), Int32(1), grid_dim=tn * tm, block_dim=256)
            return
        var slices = (k + SLICE_K - 1) // SLICE_K
        self._reserve(1, 1, slices * m * n)
        var threads = slices * n
        self.ctx.enqueue_function[_mm_partial](
            pa, pw, _dptr(self.dev_p), Int32(m), Int32(k), Int32(n), Int32(slices),
            grid_dim=(threads + BLOCK - 1) // BLOCK, block_dim=BLOCK)
        self.ctx.enqueue_function[_mm_reduce](
            _dptr(self.dev_p), pc, pb, Int32(m), Int32(n), Int32(slices), Int32(1 if has_bias else 0),
            grid_dim=(m * n + BLOCK - 1) // BLOCK, block_dim=BLOCK)

    def matmul_dev(mut self, pa: FP, w: Int, pc: FP, bias: Int, bias_key: Int, m: Int, k: Int, n: Int) raises:
        """C[m,n] = A[m,k] @ W (+ bias) with A and C device buffers; no synchronisation."""
        var pb = pc
        var hb = False
        if bias != 0:
            pb = _dptr(self.buffers[self.weight(bias_key, bias, 1, n, False)])
            hb = True
        self._launch_mm(pa, _dptr(self.buffers[w]), pc, pb, hb, m, k, n)
        self.products += 1

    def gather_matmul_dev(mut self, px: FP, cin: Int, idx: List[Float32], m: Int, slots: Int, w: Int,
                          pc: FP, n: Int) raises:
        """`gather_matmul` with X and C on the device (the index table is uploaded)."""
        if m * slots > self.cap_i:
            self.ctx.synchronize()
            self.cap_i = m * slots
            self.host_i = self.ctx.enqueue_create_host_buffer[DType.float32](self.cap_i)
            self.dev_i = self.ctx.enqueue_create_buffer[DType.float32](self.cap_i)
        self.ctx.synchronize()
        _ = external_call["memcpy", Int](Int(self.host_i.unsafe_ptr()), Int(idx.unsafe_ptr()), m * slots * 4)
        self.ctx.enqueue_copy(self.dev_i.create_sub_buffer[DType.float32](0, m * slots),
                              self.host_i.create_sub_buffer[DType.float32](0, m * slots))
        var tn = (n + TBN - 1) // TBN
        var tm = (m + TBM - 1) // TBM
        self.ctx.enqueue_function[_mm_tiled[True]](
            px, _dptr(self.buffers[w]), pc, pc, Int32(m), Int32(slots * cin), Int32(n), Int32(0), Int32(tn),
            _dptr(self.dev_i), Int32(cin), Int32(slots), grid_dim=tn * tm, block_dim=256)
        self.products += 1

    def attention_dev(mut self, pq: FP, pk: FP, pv: FP, po: FP, X: Int, L: Int, S: Int, H: Int, D: Int,
                      scale: Float32) raises -> Bool:
        if D > 64:
            return False
        var rbk = (L + AT_ROWS - 1) // AT_ROWS
        var grid = X * H * rbk
        if D <= 16:
            self.ctx.enqueue_function[_attn_kernel[16]](pq, pk, pv, po, Int32(L), Int32(S), Int32(H), Int32(D), scale,
                                                        Int32(rbk), grid_dim=grid, block_dim=AT_ROWS)
        elif D <= 32:
            self.ctx.enqueue_function[_attn_kernel[32]](pq, pk, pv, po, Int32(L), Int32(S), Int32(H), Int32(D), scale,
                                                        Int32(rbk), grid_dim=grid, block_dim=AT_ROWS)
        else:
            self.ctx.enqueue_function[_attn_kernel[64]](pq, pk, pv, po, Int32(L), Int32(S), Int32(H), Int32(D), scale,
                                                        Int32(rbk), grid_dim=grid, block_dim=AT_ROWS)
        return True

    def binary_dev(mut self, code: Int, pa: FP, pb: FP, po: FP, n: Int, period: Int, mode: Int, scalar: Float32,
                   swap: Bool) raises:
        self.ctx.enqueue_function[_ew_binary_kernel](
            pa, pb, po, Int32(n), Int32(max(period, 1)), Int32(mode), Int32(code), scalar, Int32(1 if swap else 0),
            grid_dim=(n + EW_BLOCK - 1) // EW_BLOCK, block_dim=EW_BLOCK)

    def unary_dev(mut self, code: Int, pa: FP, po: FP, n: Int) raises:
        self.ctx.enqueue_function[_ew_unary_kernel](pa, po, Int32(n), Int32(code),
                                                    grid_dim=(n + EW_BLOCK - 1) // EW_BLOCK, block_dim=EW_BLOCK)

    def layer_norm_dev(mut self, px: FP, pg: FP, pb: FP, po: FP, rows: Int, n: Int, eps: Float32, has_b: Bool) raises:
        self.ctx.enqueue_function[_layer_norm_kernel](px, pg, pb, po, Int32(n), eps, Int32(1 if has_b else 0),
                                                      grid_dim=rows, block_dim=LN_THREADS)

    def copy_dev(mut self, src: FP, dst: FP, outer: Int, chunk: Int, s_stride: Int, s_off: Int, d_stride: Int,
                 d_off: Int) raises:
        var total = outer * chunk
        if total == 0:
            return
        self.ctx.enqueue_function[_copy_kernel](src, dst, Int32(total), Int32(chunk), Int32(s_stride), Int32(s_off),
                                                Int32(d_stride), Int32(d_off),
                                                grid_dim=(total + EW_BLOCK - 1) // EW_BLOCK, block_dim=EW_BLOCK)

    def gather_rows_dev(mut self, src: FP, idx: List[Float32], dst: FP, row: Int) raises:
        """dst[i, :] = src[idx[i], :] (row length `row`); the index list is uploaded."""
        var n = len(idx)
        if n == 0:
            return
        var slot = self.alloc(n)
        self.upload(slot, Int(idx.unsafe_ptr()), n)
        var total = n * row
        self.ctx.enqueue_function[_gather_rows_kernel](src, _dptr(self.pool[slot]), dst, Int32(total), Int32(row),
                                                       grid_dim=(total + EW_BLOCK - 1) // EW_BLOCK, block_dim=EW_BLOCK)
        self.release(slot)

    def decoder_attention(mut self, q: FP, o: FP, pk: FP, pv: FP, gk: FP, gv: FP, anc: FP, rows: Int,
                          prefix_len: Int, causal: Bool, steps: Int, beams_cap: Int, hid: Int, heads: Int,
                          scale: Float32, anc_stride: Int) raises -> Bool:
        """`_dec_attn_kernel` launch; False when the head dim is not 64 or the context is too long."""
        if hid // heads != DA_HD or (prefix_len if not causal else rows) + steps > DA_MAXT:
            return False
        self.ctx.enqueue_function[_dec_attn_kernel](
            q, o, pk, pv, gk, gv, anc, Int32(prefix_len), Int32(1 if causal else 0), Int32(steps), Int32(beams_cap),
            Int32(hid), Int32(heads), scale, Int32(max(anc_stride, 1)), grid_dim=rows * heads, block_dim=DA_THREADS)
        return True
