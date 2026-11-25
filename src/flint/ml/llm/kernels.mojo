"""MAX GPU kernels for the small-LLM runtime (`flint.ml.llm.kernels`).

Everything a decoder block needs stays on the device: quantised weights upload once into byte
buffers, activations and the KV cache live in float32 buffers, and a token step uploads one
embedding row and downloads the logits.

  - `_qmv_kernel[FMT, MR]`: y[m, n] (+)= x[m, k] W^T for fp32 / int8-per-row / int4-group-32
    weights. One warp per output row, `MR` activation rows per block (1 for decoding, 4 for
    short batches such as speculative verification, 8 for prefill; a row's sum never depends
    on `MR`, so batched rows are bit-identical to single-row steps), 16-byte vector loads,
    dequantisation in registers, warp-shuffle reduction. The `acc` flag adds into y (fused
    residual).
  - `_rmsnorm_kernel`: one block per row.
  - `_qk_rope_kernel`: per (row, head) block: optional per-head RMSNorm (Qwen3 QK-norm), rotary
    embedding from a precomputed cos/sin table (rotate-half layout), and the K / V rows written
    straight into the layer's cache at their positions.
  - `_attn_kernel[HD]`: causal grouped-query attention, one block per (row, query head), keys
    in tiles of 1024 with an online softmax (no context-length limit), four warps splitting the
    keys with four rows in flight each.
  - `_silu_mul_kernel`: SwiGLU, silu(gate) * up from the stacked gate/up product.
"""

from std.ffi import external_call
from std.math import exp, sqrt
from std.memory import stack_allocation
from max.gpu import thread_idx, block_idx, barrier, global_idx, lane_id, warp_id
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from max.gpu.primitives.warp import sum as warp_sum, max as warp_max
from flint.ml.llm.quant import Q_F32, Q_INT8, Q_INT4, G4

comptime FP = Pointer[Float32, MutAnyOrigin]
comptime BP = Pointer[UInt8, MutAnyOrigin]
comptime WARPS: Int = 8
comptime ATT_THREADS: Int = 256
comptime ATT_TILE: Int = 1024
comptime MAX_HD: Int = 256


def _qmv_kernel[FMT: Int, MR: Int](x: FP, w: BP, sc: FP, y: FP, m: Int32, k: Int32, n: Int32, nb: Int32,
                                   acc: Int32, g8: Int32):
    var b = Int(block_idx.x)
    var nbk = Int(nb)
    var row = (b % nbk) * WARPS + Int(warp_id())
    var r0 = (b // nbk) * MR
    var nn = Int(n)
    if row >= nn:
        return
    var kk = Int(k)
    var mm = Int(m)
    var l = Int(lane_id())
    var a = SIMD[DType.float32, MR](0)
    comptime if FMT == Q_F32:
        var wf = FP(unsafe_from_address=Int(w)).unsafe_offset(row * kk)
        var c = l * 4
        while c < kk:
            var wv = wf.unsafe_offset(c).unsafe_load[width=4, alignment=16]()
            comptime for r in range(MR):
                if r0 + r < mm:
                    a[r] += (wv * x.unsafe_offset((r0 + r) * kk + c).unsafe_load[width=4, alignment=16]()).reduce_add()
            c += 128
    elif FMT == Q_INT8:
        # lane-interleaved: in each 128-element block, lane l's 4 bytes hold elements l + 32 i
        var wq = w.unsafe_offset(row * kk + l * 4)
        var srow = sc.unsafe_offset(row * (kk // 128))
        var nsb = kk // 128
        for sb in range(nsb):
            var wv = wq.unsafe_offset(sb * 128).unsafe_load[width=4, alignment=4]().cast[DType.int8]().cast[DType.float32]()
            var s = srow.unsafe_offset(sb).unsafe_load()
            comptime for r in range(MR):
                if r0 + r < mm:
                    var xp = x.unsafe_offset((r0 + r) * kk + sb * 128 + l)
                    var xv = SIMD[DType.float32, 4](xp.unsafe_load(), xp.unsafe_offset(32).unsafe_load(),
                                                    xp.unsafe_offset(64).unsafe_load(), xp.unsafe_offset(96).unsafe_load())
                    a[r] += (wv * xv).reduce_add() * s
    else:
        # lane-interleaved: in each 256-element block, lane l's byte b holds elements
        # l + 32 (2 b) (low nibble) and l + 32 (2 b + 1) (high nibble); groups of 32 = one i
        var wq = w.unsafe_offset(row * (kk // 2) + l * 4)
        var srow = sc.unsafe_offset(row * (kk // G4))
        var nsb = kk // 256
        for sb in range(nsb):
            var bv = wq.unsafe_offset(sb * 128).unsafe_load[width=4, alignment=4]()
            var lo = ((bv & 15).cast[DType.int8]() - 8).cast[DType.float32]()
            var hi = ((bv >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
            var s8 = srow.unsafe_offset(sb * 8).unsafe_load[width=8, alignment=4]()
            var se = SIMD[DType.float32, 4](s8[0], s8[2], s8[4], s8[6])
            var so = SIMD[DType.float32, 4](s8[1], s8[3], s8[5], s8[7])
            var wl = lo * se
            var wh = hi * so
            comptime for r in range(MR):
                if r0 + r < mm:
                    var xp = x.unsafe_offset((r0 + r) * kk + sb * 256 + l)
                    var xe = SIMD[DType.float32, 4](xp.unsafe_load(), xp.unsafe_offset(64).unsafe_load(),
                                                    xp.unsafe_offset(128).unsafe_load(), xp.unsafe_offset(192).unsafe_load())
                    var xo = SIMD[DType.float32, 4](xp.unsafe_offset(32).unsafe_load(), xp.unsafe_offset(96).unsafe_load(),
                                                    xp.unsafe_offset(160).unsafe_load(), xp.unsafe_offset(224).unsafe_load())
                    a[r] += (wl * xe + wh * xo).reduce_add()
    comptime for r in range(MR):
        var t = warp_sum(a[r])
        if l == 0 and r0 + r < mm:
            var py = y.unsafe_offset((r0 + r) * nn + row)
            if acc != 0:
                py.unsafe_store(py.unsafe_load() + t)
            else:
                py.unsafe_store(t)


@always_inline
def _block_sum[T: Int](v: Float32, sh: Pointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]) -> Float32:
    """Sum over a block of T threads (T a multiple of 32, at most 1024)."""
    var s = warp_sum(v)
    var w = Int(warp_id())
    if Int(lane_id()) == 0:
        sh[unsafe_offset=w] = s
    barrier()
    var tot: Float32 = 0
    comptime for i in range(T // 32):
        tot += sh[unsafe_offset=i]
    barrier()
    return tot


@always_inline
def _block_max[T: Int](v: Float32, sh: Pointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]) -> Float32:
    var s = warp_max(v)
    var w = Int(warp_id())
    if Int(lane_id()) == 0:
        sh[unsafe_offset=w] = s
    barrier()
    var tot: Float32 = -3.4e38
    comptime for i in range(T // 32):
        tot = max(tot, sh[unsafe_offset=i])
    barrier()
    return tot


comptime NORM_THREADS: Int = 256


def _rmsnorm_kernel(x: FP, g: FP, o: FP, n: Int32, eps: Float32):
    var sh = stack_allocation[32, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var nn = Int(n)
    var base = Int(block_idx.x) * nn
    var s: Float32 = 0
    var i = t
    while i < nn:
        var v = x.unsafe_offset(base + i).unsafe_load()
        s += v * v
        i += NORM_THREADS
    var tot = _block_sum[NORM_THREADS](s, sh)
    var inv = 1.0 / sqrt(tot / Float32(nn) + eps)
    i = t
    while i < nn:
        o.unsafe_offset(base + i).unsafe_store(x.unsafe_offset(base + i).unsafe_load() * inv * g.unsafe_offset(i).unsafe_load())
        i += NORM_THREADS


comptime ROPE_THREADS: Int = 128


def _qk_rope_kernel(qkv: FP, qout: FP, kc: FP, vc: FP, qn: FP, kn: FP, cs: FP, sn: FP, heads: Int32, kvh: Int32,
                    hd: Int32, pos0: Int32, cap: Int32, use_norm: Int32, use_rope: Int32, eps: Float32):
    """Block (row r, head h) of [m, heads + kvh]. qkv rows are [q heads | k heads | v heads]."""
    var sh = stack_allocation[32, Float32, address_space=AddressSpace.SHARED]()
    var vals = stack_allocation[MAX_HD, Float32, address_space=AddressSpace.SHARED]()
    var H = Int(heads)
    var KV = Int(kvh)
    var D = Int(hd)
    var nh = H + KV
    var b = Int(block_idx.x)
    var r = b // nh
    var h = b % nh
    var width = (H + 2 * KV) * D
    var src = qkv.unsafe_offset(r * width + h * D)
    var t = Int(thread_idx.x)
    var half = D // 2
    var pos = Int(pos0) + r
    var ss: Float32 = 0
    var d = t
    while d < D:
        var v = src.unsafe_offset(d).unsafe_load()
        vals[unsafe_offset=d] = v
        ss += v * v
        d += ROPE_THREADS
    var tot = _block_sum[ROPE_THREADS](ss, sh)
    var inv: Float32 = 1.0
    if use_norm != 0:
        inv = 1.0 / sqrt(tot / Float32(D) + eps)
    var gw = qn if h < H else kn
    d = t
    while d < D:
        var v = vals[unsafe_offset=d] * inv
        if use_norm != 0:
            v *= gw.unsafe_offset(d).unsafe_load()
        vals[unsafe_offset=d] = v
        d += ROPE_THREADS
    barrier()
    var dst: FP
    if h < H:
        dst = qout.unsafe_offset(r * H * D + h * D)
    else:
        var j = h - H
        dst = kc.unsafe_offset((j * Int(cap) + pos) * D)
        var vsrc = qkv.unsafe_offset(r * width + (H + KV + j) * D)
        var vdst = vc.unsafe_offset((j * Int(cap) + pos) * D)
        d = t
        while d < D:
            vdst.unsafe_offset(d).unsafe_store(vsrc.unsafe_offset(d).unsafe_load())
            d += ROPE_THREADS
    d = t
    while d < D:
        var v = vals[unsafe_offset=d]
        if use_rope != 0:
            var f = d % half
            var c = cs.unsafe_offset(pos * half + f).unsafe_load()
            var s = sn.unsafe_offset(pos * half + f).unsafe_load()
            var partner = vals[unsafe_offset=d + half] if d < half else vals[unsafe_offset=d - half]
            if d < half:
                v = v * c - partner * s
            else:
                v = v * c + partner * s
        dst.unsafe_offset(d).unsafe_store(v)
        d += ROPE_THREADS


comptime ATT_WARPS: Int = ATT_THREADS // 32


def _attn_kernel[HD: Int](q: FP, o: FP, kc: FP, vc: FP, heads: Int32, kvh: Int32, pos0: Int32, cap: Int32,
                          scale: Float32):
    """Block (row r, query head h): o = softmax(scale q K^T) V over positions 0..pos0 + r.

    Lane l of every warp owns dims [l * DPL, (l + 1) * DPL). Warp w scores keys w, w + 4, ...
    of a tile (one coalesced row load and a warp reduction per key, four keys in flight) and
    accumulates p V for the same keys; the four partial outputs are summed at the end."""
    comptime DPL = HD // 32
    var sh = stack_allocation[32, Float32, address_space=AddressSpace.SHARED]()
    var sc = stack_allocation[ATT_TILE, Float32, address_space=AddressSpace.SHARED]()
    var red = stack_allocation[ATT_WARPS * HD, Float32, address_space=AddressSpace.SHARED]()
    var H = Int(heads)
    var b = Int(block_idx.x)
    var r = b // H
    var h = b % H
    var kh = h // (H // Int(kvh))
    var t = Int(thread_idx.x)
    var w = Int(warp_id())
    var l = Int(lane_id())
    var qv = q.unsafe_offset((r * H + h) * HD + l * DPL).unsafe_load[width=DPL, alignment=4 * DPL]() * scale
    var n_keys = Int(pos0) + r + 1
    var kbase = kc.unsafe_offset(kh * Int(cap) * HD + l * DPL)
    var vbase = vc.unsafe_offset(kh * Int(cap) * HD + l * DPL)
    var m_run: Float32 = -3.4e38
    var l_run: Float32 = 0
    var acc = SIMD[DType.float32, DPL](0)
    var t0 = 0
    while t0 < n_keys:
        var tn = min(ATT_TILE, n_keys - t0)
        var lmax: Float32 = -3.4e38
        var j = w
        while j + 3 * ATT_WARPS < tn:
            var k0 = kbase.unsafe_offset((t0 + j) * HD).unsafe_load[width=DPL, alignment=4 * DPL]()
            var k1 = kbase.unsafe_offset((t0 + j + ATT_WARPS) * HD).unsafe_load[width=DPL, alignment=4 * DPL]()
            var k2 = kbase.unsafe_offset((t0 + j + 2 * ATT_WARPS) * HD).unsafe_load[width=DPL, alignment=4 * DPL]()
            var k3 = kbase.unsafe_offset((t0 + j + 3 * ATT_WARPS) * HD).unsafe_load[width=DPL, alignment=4 * DPL]()
            var s0 = warp_sum((qv * k0).reduce_add())
            var s1 = warp_sum((qv * k1).reduce_add())
            var s2 = warp_sum((qv * k2).reduce_add())
            var s3 = warp_sum((qv * k3).reduce_add())
            if l == 0:
                sc[unsafe_offset=j] = s0
                sc[unsafe_offset=j + ATT_WARPS] = s1
                sc[unsafe_offset=j + 2 * ATT_WARPS] = s2
                sc[unsafe_offset=j + 3 * ATT_WARPS] = s3
            lmax = max(lmax, max(max(s0, s1), max(s2, s3)))
            j += 4 * ATT_WARPS
        while j < tn:
            var k0 = kbase.unsafe_offset((t0 + j) * HD).unsafe_load[width=DPL, alignment=4 * DPL]()
            var s0 = warp_sum((qv * k0).reduce_add())
            if l == 0:
                sc[unsafe_offset=j] = s0
            lmax = max(lmax, s0)
            j += ATT_WARPS
        var tmax = _block_max[ATT_THREADS](lmax, sh)
        var new_m = max(m_run, tmax)
        var lsum: Float32 = 0
        j = t
        while j < tn:
            var p = exp(sc[unsafe_offset=j] - new_m)
            sc[unsafe_offset=j] = p
            lsum += p
            j += ATT_THREADS
        var tsum = _block_sum[ATT_THREADS](lsum, sh)
        var corr = exp(m_run - new_m)
        l_run = l_run * corr + tsum
        acc *= corr
        j = w
        while j + 3 * ATT_WARPS < tn:
            var v0 = vbase.unsafe_offset((t0 + j) * HD).unsafe_load[width=DPL, alignment=4 * DPL]()
            var v1 = vbase.unsafe_offset((t0 + j + ATT_WARPS) * HD).unsafe_load[width=DPL, alignment=4 * DPL]()
            var v2 = vbase.unsafe_offset((t0 + j + 2 * ATT_WARPS) * HD).unsafe_load[width=DPL, alignment=4 * DPL]()
            var v3 = vbase.unsafe_offset((t0 + j + 3 * ATT_WARPS) * HD).unsafe_load[width=DPL, alignment=4 * DPL]()
            acc += v0 * sc[unsafe_offset=j] + v1 * sc[unsafe_offset=j + ATT_WARPS] \
                + v2 * sc[unsafe_offset=j + 2 * ATT_WARPS] + v3 * sc[unsafe_offset=j + 3 * ATT_WARPS]
            j += 4 * ATT_WARPS
        while j < tn:
            acc += vbase.unsafe_offset((t0 + j) * HD).unsafe_load[width=DPL, alignment=4 * DPL]() * sc[unsafe_offset=j]
            j += ATT_WARPS
        m_run = new_m
        barrier()
        t0 += ATT_TILE
    comptime for e in range(DPL):
        red[unsafe_offset=w * HD + l * DPL + e] = acc[e]
    barrier()
    var inv = 1.0 / l_run
    var op = o.unsafe_offset((r * H + h) * HD)
    var d = t
    while d < HD:
        var tot: Float32 = 0
        comptime for ww in range(ATT_WARPS):
            tot += red[unsafe_offset=ww * HD + d]
        op.unsafe_offset(d).unsafe_store(tot * inv)
        d += ATT_THREADS


def _silu_mul_kernel(gu: FP, a: FP, ffn: Int32, total: Int32):
    var e = Int(global_idx.x)
    if e >= Int(total):
        return
    var f = Int(ffn)
    var r = e // f
    var i = e % f
    var g = gu.unsafe_offset(r * 2 * f + i).unsafe_load()
    var u = gu.unsafe_offset(r * 2 * f + f + i).unsafe_load()
    a.unsafe_offset(e).unsafe_store(g / (1.0 + exp(-g)) * u)


def gpu_layout(fmt: Int, src: Int, n: Int, k: Int) -> Int:
    """A malloc'd copy of a canonical `QMat` payload in the kernels' lane-interleaved layout."""
    var bytes = n * k if fmt == Q_INT8 else n * k // 2
    var dst = external_call["malloc", Int](bytes)
    var ps = BP(unsafe_from_address=src)
    var pd = BP(unsafe_from_address=dst)
    if fmt == Q_INT8:
        for r in range(n):
            for sb in range(k // 128):
                var base = r * k + sb * 128
                for l in range(32):
                    for i in range(4):
                        pd.unsafe_offset(base + l * 4 + i).unsafe_store(ps.unsafe_offset(base + i * 32 + l).unsafe_load())
        return dst
    var half = G4 // 2
    for r in range(n):
        var rs = r * (k // 2)
        for sb in range(k // 256):
            for l in range(32):
                for b in range(4):
                    var v = 0
                    for e in range(2):
                        var elem = sb * 256 + (2 * b + e) * 32 + l
                        var g = elem // G4
                        var j = elem % G4
                        var byte = Int(ps.unsafe_offset(rs + g * half + (j % half)).unsafe_load())
                        var nib = (byte & 15) if j < half else (byte >> 4)
                        v |= nib << (4 * e)
                    pd.unsafe_offset(rs + sb * 128 + l * 4 + b).unsafe_store(UInt8(v))
    return dst


@always_inline
def _fp(buf: DeviceBuffer[DType.float32]) -> FP:
    return FP(unsafe_from_address=Int(buf.unsafe_ptr()))


struct LlmGpu(Movable):
    """One MAX device context with resident weight bytes and float32 working buffers."""
    var ctx: DeviceContext
    var wbufs: List[DeviceBuffer[DType.uint8]]
    var fbufs: List[DeviceBuffer[DType.float32]]
    var host: HostBuffer[DType.float32]
    var host_cap: Int
    var weight_bytes: Int

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.wbufs = List[DeviceBuffer[DType.uint8]]()
        self.fbufs = List[DeviceBuffer[DType.float32]]()
        self.host_cap = 1 << 20
        self.host = self.ctx.enqueue_create_host_buffer[DType.float32](self.host_cap)
        self.weight_bytes = 0

    def name(self) raises -> String:
        return self.ctx.name()

    def upload_bytes(mut self, src: Int, nbytes: Int) raises -> Int:
        var buf = self.ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 16))
        with buf.map_to_host() as h:
            _ = external_call["memcpy", Int](Int(h.unsafe_ptr()), src, nbytes)
        self.wbufs.append(buf^)
        self.weight_bytes += nbytes
        return len(self.wbufs) - 1

    def wptr(self, idx: Int) -> BP:
        return BP(unsafe_from_address=Int(self.wbufs[idx].unsafe_ptr()))

    def wfptr(self, idx: Int) -> FP:
        return FP(unsafe_from_address=Int(self.wbufs[idx].unsafe_ptr()))

    def alloc(mut self, n: Int) raises -> Int:
        var buf = self.ctx.enqueue_create_buffer[DType.float32](max(n, 4))
        buf.enqueue_fill(0.0)
        self.fbufs.append(buf^)
        return len(self.fbufs) - 1

    def ptr(self, idx: Int) -> FP:
        return _fp(self.fbufs[idx])

    def _stage(mut self, n: Int) raises:
        if n > self.host_cap:
            self.ctx.synchronize()
            self.host_cap = n
            self.host = self.ctx.enqueue_create_host_buffer[DType.float32](n)

    def upload(mut self, idx: Int, src: Int, n: Int, offset: Int = 0) raises:
        self._stage(n)
        self.ctx.synchronize()
        _ = external_call["memcpy", Int](Int(self.host.unsafe_ptr()), src, n * 4)
        self.ctx.enqueue_copy(self.fbufs[idx].create_sub_buffer[DType.float32](offset, n),
                              self.host.create_sub_buffer[DType.float32](0, n))
        self.ctx.synchronize()

    def download(mut self, idx: Int, dst: Int, n: Int, offset: Int = 0) raises:
        self._stage(n)
        self.ctx.enqueue_copy(self.host.create_sub_buffer[DType.float32](0, n),
                              self.fbufs[idx].create_sub_buffer[DType.float32](offset, n))
        self.ctx.synchronize()
        _ = external_call["memcpy", Int](dst, Int(self.host.unsafe_ptr()), n * 4)

    def sync(mut self) raises:
        self.ctx.synchronize()

    def matvec(mut self, fmt: Int, x: FP, w: BP, sc: FP, y: FP, m: Int, k: Int, n: Int, acc: Bool) raises:
        var nb = (n + WARPS - 1) // WARPS
        var a = Int32(1 if acc else 0)
        var gs = Int32(128)
        if m == 1:
            if fmt == Q_INT8:
                self.ctx.enqueue_function[_qmv_kernel[Q_INT8, 1]](x, w, sc, y, Int32(m), Int32(k), Int32(n), Int32(nb), a, gs,
                                                                 grid_dim=nb, block_dim=WARPS * 32)
            elif fmt == Q_INT4:
                self.ctx.enqueue_function[_qmv_kernel[Q_INT4, 1]](x, w, sc, y, Int32(m), Int32(k), Int32(n), Int32(nb), a, gs,
                                                                 grid_dim=nb, block_dim=WARPS * 32)
            else:
                self.ctx.enqueue_function[_qmv_kernel[Q_F32, 1]](x, w, sc, y, Int32(m), Int32(k), Int32(n), Int32(nb), a, gs,
                                                                grid_dim=nb, block_dim=WARPS * 32)
            return
        if m <= 4:
            # short batches (speculative verification): four rows per block, fewer idle registers
            if fmt == Q_INT8:
                self.ctx.enqueue_function[_qmv_kernel[Q_INT8, 4]](x, w, sc, y, Int32(m), Int32(k), Int32(n), Int32(nb), a,
                                                                 gs, grid_dim=nb, block_dim=WARPS * 32)
            elif fmt == Q_INT4:
                self.ctx.enqueue_function[_qmv_kernel[Q_INT4, 4]](x, w, sc, y, Int32(m), Int32(k), Int32(n), Int32(nb), a,
                                                                 gs, grid_dim=nb, block_dim=WARPS * 32)
            else:
                self.ctx.enqueue_function[_qmv_kernel[Q_F32, 4]](x, w, sc, y, Int32(m), Int32(k), Int32(n), Int32(nb), a,
                                                                gs, grid_dim=nb, block_dim=WARPS * 32)
            return
        var mb = (m + 7) // 8
        if fmt == Q_INT8:
            self.ctx.enqueue_function[_qmv_kernel[Q_INT8, 8]](x, w, sc, y, Int32(m), Int32(k), Int32(n), Int32(nb), a, gs,
                                                             grid_dim=nb * mb, block_dim=WARPS * 32)
        elif fmt == Q_INT4:
            self.ctx.enqueue_function[_qmv_kernel[Q_INT4, 8]](x, w, sc, y, Int32(m), Int32(k), Int32(n), Int32(nb), a, gs,
                                                             grid_dim=nb * mb, block_dim=WARPS * 32)
        else:
            self.ctx.enqueue_function[_qmv_kernel[Q_F32, 8]](x, w, sc, y, Int32(m), Int32(k), Int32(n), Int32(nb), a, gs,
                                                            grid_dim=nb * mb, block_dim=WARPS * 32)

    def rmsnorm(mut self, x: FP, g: FP, o: FP, rows: Int, n: Int, eps: Float32) raises:
        self.ctx.enqueue_function[_rmsnorm_kernel](x, g, o, Int32(n), eps, grid_dim=rows, block_dim=NORM_THREADS)

    def qk_rope(mut self, qkv: FP, qout: FP, kc: FP, vc: FP, qn: FP, kn: FP, cs: FP, sn: FP, rows: Int, heads: Int,
                kvh: Int, hd: Int, pos0: Int, cap: Int, use_norm: Bool, use_rope: Bool, eps: Float32) raises:
        self.ctx.enqueue_function[_qk_rope_kernel](
            qkv, qout, kc, vc, qn, kn, cs, sn, Int32(heads), Int32(kvh), Int32(hd), Int32(pos0), Int32(cap),
            Int32(1 if use_norm else 0), Int32(1 if use_rope else 0), eps,
            grid_dim=rows * (heads + kvh), block_dim=ROPE_THREADS)

    def attention(mut self, q: FP, o: FP, kc: FP, vc: FP, rows: Int, heads: Int, kvh: Int, hd: Int, pos0: Int,
                  cap: Int, scale: Float32) raises:
        if hd == 64:
            self.ctx.enqueue_function[_attn_kernel[64]](q, o, kc, vc, Int32(heads), Int32(kvh), Int32(pos0),
                                                        Int32(cap), scale, grid_dim=rows * heads, block_dim=ATT_THREADS)
        elif hd == 128:
            self.ctx.enqueue_function[_attn_kernel[128]](q, o, kc, vc, Int32(heads), Int32(kvh), Int32(pos0),
                                                         Int32(cap), scale, grid_dim=rows * heads, block_dim=ATT_THREADS)
        elif hd == 256:
            self.ctx.enqueue_function[_attn_kernel[256]](q, o, kc, vc, Int32(heads), Int32(kvh), Int32(pos0),
                                                         Int32(cap), scale, grid_dim=rows * heads, block_dim=ATT_THREADS)
        else:
            raise Error("llm: GPU attention supports head_dim 64, 128 or 256")

    def silu_mul(mut self, gu: FP, a: FP, rows: Int, ffn: Int) raises:
        var total = rows * ffn
        self.ctx.enqueue_function[_silu_mul_kernel](gu, a, Int32(ffn), Int32(total),
                                                    grid_dim=(total + 255) // 256, block_dim=256)
