"""MAX GPU path for the streaming SpectroStream decoder (`flint.ml.audio_codec_gpu`).

`SpectroStreamGpu.build(dec)` uploads the weights of a loaded `SpectroStreamDecoder` once and
runs its convolution stack on the GPU with device-resident activations and streaming state:

  - every conv phase (transposed convs split into stride phases, as on the CPU) is one gather
    GEMM (64 x 64 tiles, split-K when the tile count is low, a row-dot kernel for the 2-channel
    output layer) whose im2col index table is built once, because the
    per-step geometry is fixed (one codec frame in, 4 STFT frames out); taps that fall into the
    padding point at an all-zero row of the input buffer;
  - the GEMM epilogue adds the bias and writes each phase's rows to their interleaved output
    positions; ELU / residual / nearest-upsampling
    shortcuts are small element-wise kernels, and the time history of each conv lives in a
    ping-pong pair of device buffers;
  - one download per frame returns both channel groups' output layer, which the CPU decoder's
    `finish_frame` turns into 1920 stereo samples (lookahead mask + inverse STFT).

Results match the CPU decoder up to float summation order.
"""

from std.ffi import external_call
from std.math import exp
from max.gpu import global_idx, thread_idx, block_idx, barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation
from flint.ml.gpu import GpuMatmul, FP
from flint.ml.audio_codec import (FBuf, StreamConv2D, SpectroStreamDecoder, floor_div, SS_STFT_FRAMES,
                                  SS_BINS)

comptime _EW: Int = 256


def _elu_kernel(src: FP, dst: FP, n: Int32, act: Int32):
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    var x = src.unsafe_offset(i).unsafe_load()
    if act != 0 and x <= 0:
        x = exp(x) - 1
    dst.unsafe_offset(i).unsafe_store(x)


def _add_rows_kernel(dst: FP, src: FP, rmap: FP, total: Int32, row: Int32, mapped: Int32):
    """dst[r, :] += src[map[r], :] (src[r, :] when not mapped)."""
    var e = Int(global_idx.x)
    if e >= Int(total):
        return
    var rw = Int(row)
    var r = e // rw
    var s = r
    if mapped != 0:
        s = Int(rmap.unsafe_offset(r).unsafe_load())
    var p = dst.unsafe_offset(e)
    p.unsafe_store(p.unsafe_load() + src.unsafe_offset(s * rw + e % rw).unsafe_load())


comptime _BM: Int = 64
comptime _BN: Int = 64
comptime _BK: Int = 16
comptime _TM: Int = 4           # 16 x 16 threads, 4 x 4 outputs each
comptime _K_PER_SLICE: Int = 256


def _gather_gemm_kernel(a: FP, b: FP, c: FP, bias: FP, idx: FP, rmap: FP, m: Int32, k: Int32, n: Int32,
                        cin: Int32, slots: Int32, tiles_n: Int32, tiles: Int32, kslice: Int32, mode: Int32):
    """64 x 64 tile of G[m, k] @ B[k, n] with G[r, j * cin + ch] = a[idx[r, j], ch].

    Block b covers tile b % tiles and K range [s * kslice, (s + 1) * kslice) with s = b / tiles.
    mode 0 / 2: write c[row(r), col] = acc + bias with row(r) = r / rmap[r];
    mode 1: write the split-K partial c[s, r, col] = acc (see `_splitk_reduce_kernel`)."""
    var As = stack_allocation[_BK * _BM, Float32, address_space=AddressSpace.SHARED]()
    var Bs = stack_allocation[_BK * _BN, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var tx = t % 16
    var ty = t // 16
    var blk = Int(block_idx.x)
    var tile = blk % Int(tiles)
    var sl = blk // Int(tiles)
    var bn = tile % Int(tiles_n)
    var bm = tile // Int(tiles_n)
    var mm = Int(m)
    var kn = Int(k)
    var nn = Int(n)
    var ci = Int(cin)
    var sk = Int(slots)
    var row0 = bm * _BM
    var col0 = bn * _BN
    var kbeg = sl * Int(kslice)
    var kend = min(kn, kbeg + Int(kslice))
    var acc0 = SIMD[DType.float32, 4](0)
    var acc1 = SIMD[DType.float32, 4](0)
    var acc2 = SIMD[DType.float32, 4](0)
    var acc3 = SIMD[DType.float32, 4](0)
    var k0 = kbeg
    while k0 < kend:
        # A tile [64 rows, 16 k]: 4 per thread
        comptime for q in range(4):
            var e = t * 4 + q
            var r = e // _BK
            var kk = e % _BK
            var gr = row0 + r
            var gk = k0 + kk
            var v: Float32 = 0
            if gr < mm and gk < kend:
                var row = Int(idx.unsafe_offset(gr * sk + gk // ci).unsafe_load())
                v = a.unsafe_offset(row * ci + gk % ci).unsafe_load()
            As[unsafe_offset=kk * _BM + r] = v
        # B tile [16 k, 64 cols]: 4 per thread
        comptime for q in range(4):
            var e = t * 4 + q
            var kk = e // _BN
            var cc = e % _BN
            var gk = k0 + kk
            var gc = col0 + cc
            var v: Float32 = 0
            if gk < kend and gc < nn:
                v = b.unsafe_offset(gk * nn + gc).unsafe_load()
            Bs[unsafe_offset=kk * _BN + cc] = v
        barrier()
        comptime for kk in range(_BK):
            var bv = SIMD[DType.float32, 4](0)
            comptime for j in range(4):
                bv[j] = Bs[unsafe_offset=kk * _BN + tx * _TM + j]
            acc0 += As[unsafe_offset=kk * _BM + ty * _TM + 0] * bv
            acc1 += As[unsafe_offset=kk * _BM + ty * _TM + 1] * bv
            acc2 += As[unsafe_offset=kk * _BM + ty * _TM + 2] * bv
            acc3 += As[unsafe_offset=kk * _BM + ty * _TM + 3] * bv
        barrier()
        k0 += _BK
    comptime for i in range(4):
        var gr = row0 + ty * _TM + i
        if gr < mm:
            var accv: SIMD[DType.float32, 4]
            comptime if i == 0:
                accv = acc0
            elif i == 1:
                accv = acc1
            elif i == 2:
                accv = acc2
            else:
                accv = acc3
            comptime for j in range(4):
                var gc = col0 + tx * _TM + j
                if gc < nn:
                    if mode != 1:
                        var d = gr
                        if mode == 2:
                            d = Int(rmap.unsafe_offset(gr).unsafe_load())
                        c.unsafe_offset(d * nn + gc).unsafe_store(accv[j] + bias.unsafe_offset(gc).unsafe_load())
                    else:
                        c.unsafe_offset((sl * mm + gr) * nn + gc).unsafe_store(accv[j])


def _splitk_reduce_kernel(part: FP, c: FP, bias: FP, rmap: FP, m: Int32, n: Int32, slices: Int32, mapped: Int32):
    """c[map[r], col] = bias[col] + sum_s part[s, r, col]."""
    var e = Int(global_idx.x)
    var mm = Int(m)
    var nn = Int(n)
    if e >= mm * nn:
        return
    var r = e // nn
    var col = e % nn
    var v = bias.unsafe_offset(col).unsafe_load()
    for s in range(Int(slices)):
        v += part.unsafe_offset((s * mm + r) * nn + col).unsafe_load()
    var d = r
    if mapped != 0:
        d = Int(rmap.unsafe_offset(r).unsafe_load())
    c.unsafe_offset(d * nn + col).unsafe_store(v)


comptime _NARROW: Int = 8   # output channels handled per thread by the row-dot kernel


def _gather_rowdot_kernel(a: FP, b: FP, c: FP, bias: FP, idx: FP, rmap: FP, m: Int32, n: Int32, cin: Int32,
                          slots: Int32, mapped: Int32):
    """Few output channels (n <= 8): one thread per GEMM row over the whole K."""
    var r = Int(global_idx.x)
    var mm = Int(m)
    if r >= mm:
        return
    var nn = Int(n)
    var ci = Int(cin)
    var acc = SIMD[DType.float32, _NARROW](0)
    for j in range(Int(slots)):
        var row = Int(idx.unsafe_offset(r * Int(slots) + j).unsafe_load())
        var ab = row * ci
        var kb = j * ci
        for ch in range(ci):
            var x = a.unsafe_offset(ab + ch).unsafe_load()
            comptime for o in range(_NARROW):
                if o < nn:
                    acc[o] += x * b.unsafe_offset((kb + ch) * nn + o).unsafe_load()
    var d = r
    if mapped != 0:
        d = Int(rmap.unsafe_offset(r).unsafe_load())
    comptime for o in range(_NARROW):
        if o < nn:
            c.unsafe_offset(d * nn + o).unsafe_store(acc[o] + bias.unsafe_offset(o).unsafe_load())


def _grid(n: Int) -> Int:
    return (n + _EW - 1) // _EW


def _upload_floats(mut gm: GpuMatmul, values: List[Float32]) raises -> Int:
    var slot = gm.alloc(len(values))
    gm.upload(slot, Int(values.unsafe_ptr()), len(values))
    return slot


def _zeros(mut gm: GpuMatmul, n: Int) raises -> Int:
    var z = FBuf(n)
    var slot = gm.alloc(n)
    gm.upload(slot, z.addr, n)
    _ = z.n
    return slot




comptime _TARGET_BLOCKS: Int = 192   # aim for a few blocks per SM


@fieldwise_init
struct _GpuPhase(Copyable, Movable):
    var w: Int          # resident weight index ([ntap * cin, cout])
    var idx: Int        # pooled slot: [m, ntap] input row numbers (floats)
    var rmap: Int       # pooled slot: output row per GEMM row, -1 when rows are already in order
    var m: Int
    var ntap: Int
    var slices: Int      # split-K slices (1: the GEMM writes the output directly)


struct GpuConv(Movable):
    """One `StreamConv2D` on the GPU for a fixed `t` new input frames per step."""
    var t: Int
    var fin: Int
    var cin: Int
    var cout: Int
    var fout: Int
    var tout: Int
    var hist: Int
    var phases: List[_GpuPhase]
    var bias: Int
    var bufs: List[Int]       # ping-pong input buffers [(hist + t) * fin + 1 zero row, cin]
    var cur: Int
    var out: Int
    var in_rows: Int
    var part: Int        # split-K partial sums (-1 when unused)

    def __init__(out self, mut gm: GpuMatmul, c: StreamConv2D, t: Int) raises:
        self.t = t
        self.fin = c.fin
        self.cin = c.cin
        self.cout = c.cout
        self.fout = c.fout
        self.hist = c.hist
        var nph_t = c.st if c.transposed else 1
        var nph_f = c.sf if c.transposed else 1
        self.tout = t * nph_t
        self.in_rows = (self.hist + t) * self.fin
        var zero_row = self.in_rows
        self.bias = gm.weight(c.bias.addr, c.bias.addr, 1, self.cout, False)
        self.phases = List[_GpuPhase]()
        var max_part = 0
        for pi in range(len(c.phases)):
            ref ph = c.phases[pi]
            var ntap = len(ph.taps)
            if ntap == 0:
                continue
            var rt = (self.tout - ph.pt + nph_t - 1) // nph_t
            var rf = (self.fout - ph.pf + nph_f - 1) // nph_f
            var m = rt * rf
            if m == 0:
                continue
            var idx = List[Float32](capacity=m * ntap)
            var rmap = List[Float32](capacity=m)
            var ordered = True
            for i in range(rt):
                var ot = ph.pt + nph_t * i
                for j in range(rf):
                    var of_ = ph.pf + nph_f * j
                    var orow = ot * self.fout + of_
                    if orow != len(rmap):
                        ordered = False
                    rmap.append(Float32(orow))
                    for ti in range(ntap):
                        var it = floor_div(ot + ph.taps[ti].a - c.plt, nph_t)
                        var if_ = floor_div(of_ + ph.taps[ti].b - c.plf, nph_f)
                        if if_ < 0 or if_ >= self.fin or it < -self.hist or it >= t:
                            idx.append(Float32(zero_row))
                        else:
                            idx.append(Float32((self.hist + it) * self.fin + if_))
            var w = gm.weight(ph.w.addr, ph.w.addr, ntap * self.cin, self.cout, False)
            var islot = _upload_floats(gm, idx)
            var mslot = -1 if ordered else _upload_floats(gm, rmap)
            var k = ntap * self.cin
            var tiles = ((m + _BM - 1) // _BM) * ((self.cout + _BN - 1) // _BN)
            var slices = 1
            if self.cout > _NARROW and tiles < _TARGET_BLOCKS:
                slices = max(1, min((_TARGET_BLOCKS + tiles - 1) // tiles, k // _K_PER_SLICE))
            self.phases.append(_GpuPhase(w, islot, mslot, m, ntap, slices))
            if slices > 1:
                max_part = max(max_part, slices * m * self.cout)
        self.bufs = List[Int]()
        for _ in range(2):
            self.bufs.append(_zeros(gm, (self.in_rows + 1) * self.cin))
        self.cur = 0
        self.part = gm.alloc(max_part) if max_part > 0 else -1
        self.out = gm.alloc(self.tout * self.fout * self.cout)

    def reset(mut self, mut gm: GpuMatmul) raises:
        for b in range(2):
            var z = FBuf((self.in_rows + 1) * self.cin)
            gm.upload(self.bufs[b], z.addr, (self.in_rows + 1) * self.cin)
            _ = z.n
        self.cur = 0

    def slot(self, gm: GpuMatmul) -> FP:
        """Device address of the t new input frames ([t * fin, cin])."""
        return gm.ptr(self.bufs[self.cur]).unsafe_offset(self.hist * self.fin * self.cin)

    def load_input(mut self, mut gm: GpuMatmul, x: FP, elu: Bool) raises:
        var n = self.t * self.fin * self.cin
        gm.ctx.enqueue_function[_elu_kernel](x, self.slot(gm), Int32(n), Int32(1 if elu else 0),
                                             grid_dim=_grid(n), block_dim=_EW)

    def out_ptr(self, gm: GpuMatmul) -> FP:
        return gm.ptr(self.out)

    def run(mut self, mut gm: GpuMatmul) raises:
        """Runs the step on the frames already written to `slot()`."""
        var a = gm.ptr(self.bufs[self.cur])
        var pb = gm.wptr(self.bias)
        var o = gm.ptr(self.out)
        for pi in range(len(self.phases)):
            var ph = self.phases[pi].copy()
            var mp = gm.ptr(ph.rmap) if ph.rmap >= 0 else a
            var mapped = Int32(1 if ph.rmap >= 0 else 0)
            var k = ph.ntap * self.cin
            if self.cout <= _NARROW:
                gm.ctx.enqueue_function[_gather_rowdot_kernel](
                    a, gm.wptr(ph.w), o, pb, gm.ptr(ph.idx), mp, Int32(ph.m), Int32(self.cout), Int32(self.cin),
                    Int32(ph.ntap), mapped, grid_dim=_grid(ph.m), block_dim=_EW)
                continue
            var tn = (self.cout + _BN - 1) // _BN
            var tiles = ((ph.m + _BM - 1) // _BM) * tn
            if ph.slices == 1:
                gm.ctx.enqueue_function[_gather_gemm_kernel](
                    a, gm.wptr(ph.w), o, pb, gm.ptr(ph.idx), mp, Int32(ph.m), Int32(k), Int32(self.cout),
                    Int32(self.cin), Int32(ph.ntap), Int32(tn), Int32(tiles), Int32(k), Int32(2 if ph.rmap >= 0 else 0),
                    grid_dim=tiles, block_dim=256)
                continue
            var ks = (k + ph.slices - 1) // ph.slices
            ks = (ks + _BK - 1) // _BK * _BK
            var slices = (k + ks - 1) // ks
            gm.ctx.enqueue_function[_gather_gemm_kernel](
                a, gm.wptr(ph.w), gm.ptr(self.part), pb, gm.ptr(ph.idx), mp, Int32(ph.m), Int32(k), Int32(self.cout),
                Int32(self.cin), Int32(ph.ntap), Int32(tn), Int32(tiles), Int32(ks), Int32(1),
                grid_dim=tiles * slices, block_dim=256)
            var total = ph.m * self.cout
            gm.ctx.enqueue_function[_splitk_reduce_kernel](
                gm.ptr(self.part), o, pb, mp, Int32(ph.m), Int32(self.cout), Int32(slices), mapped,
                grid_dim=_grid(total), block_dim=_EW)
        if self.hist > 0:
            var row = self.fin * self.cin
            var nxt = 1 - self.cur
            var n = self.hist * row
            gm.ctx.enqueue_function[_elu_kernel](a.unsafe_offset(self.t * row), gm.ptr(self.bufs[nxt]), Int32(n), Int32(0),
                                                 grid_dim=_grid(n), block_dim=_EW)
            self.cur = nxt


struct GpuResUnit(Movable):
    var a: GpuConv
    var b: GpuConv
    var has_short: Bool
    var short: List[GpuConv]
    var upmap: Int     # slot: shortcut row per output row (-1: identity)

    def __init__(out self, mut gm: GpuMatmul, var a: GpuConv, var b: GpuConv, var short: List[GpuConv],
                 up_t: Int, up_f: Int) raises:
        self.has_short = len(short) > 0
        var fin = a.fin
        var fout = b.fout
        var tout = b.tout
        self.upmap = -1
        if up_t != 1 or up_f != 1:
            var m = List[Float32](capacity=tout * fout)
            for ot in range(tout):
                for of_ in range(fout):
                    m.append(Float32((ot // up_t) * fin + of_ // up_f))
            self.upmap = _upload_floats(gm, m)
        self.a = a^
        self.b = b^
        self.short = short^

    def reset(mut self, mut gm: GpuMatmul) raises:
        self.a.reset(gm)
        self.b.reset(gm)
        for i in range(len(self.short)):
            self.short[i].reset(gm)

    def step(mut self, mut gm: GpuMatmul, x: FP) raises:
        """x: [t * fin, cin] device rows; the result is `self.b.out`."""
        self.a.load_input(gm, x, True)
        self.a.run(gm)
        self.b.load_input(gm, self.a.out_ptr(gm), True)
        self.b.run(gm)
        var src = x
        var row = self.b.cout
        if self.has_short:
            self.short[0].load_input(gm, x, False)
            self.short[0].run(gm)
            src = self.short[0].out_ptr(gm)
        var total = self.b.tout * self.b.fout * row
        var mp = gm.ptr(self.upmap) if self.upmap >= 0 else src
        gm.ctx.enqueue_function[_add_rows_kernel](self.b.out_ptr(gm), src, mp, Int32(total), Int32(row),
                                                  Int32(1 if self.upmap >= 0 else 0),
                                                  grid_dim=_grid(total), block_dim=_EW)


def _gpu_unit(mut gm: GpuMatmul, ref u: SpectroStreamDecoder, g: Int, i: Int, t: Int) raises -> GpuResUnit:
    ref cu = u.group_state[g][i]
    var a = GpuConv(gm, cu.conv_a, t)
    var b = GpuConv(gm, cu.conv_b, a.tout)
    var sh = List[GpuConv]()
    if cu.has_short:
        sh.append(GpuConv(gm, cu.short, t))
    return GpuResUnit(gm, a^, b^, sh^, cu.up_t, cu.up_f)


struct SpectroStreamGpu(Movable):
    """GPU convolution stack of a `SpectroStreamDecoder` (see the module docstring)."""
    var gm: GpuMatmul
    var first: GpuConv
    var b1: GpuConv
    var b2: GpuConv
    var iru: GpuResUnit
    var d0: GpuResUnit
    var groups: List[List[GpuResUnit]]
    var last: List[GpuConv]
    var emb: Int
    var x0: Int
    var gin: List[Int]
    var host_out: FBuf     # [2 groups, 4 * 480 * 2]

    def __init__(out self, ref dec: SpectroStreamDecoder) raises:
        self.gm = GpuMatmul()
        self.first = GpuConv(self.gm, dec.first, 1)
        self.b1 = GpuConv(self.gm, dec.b1, 1)
        self.b2 = GpuConv(self.gm, dec.b2, 1)
        var ia = GpuConv(self.gm, dec.iru.conv_a, 1)
        var ib = GpuConv(self.gm, dec.iru.conv_b, 1)
        self.iru = GpuResUnit(self.gm, ia^, ib^, List[GpuConv](), 1, 1)
        var da = GpuConv(self.gm, dec.d0.conv_a, 1)
        var db = GpuConv(self.gm, dec.d0.conv_b, da.tout)
        var ds = List[GpuConv]()
        ds.append(GpuConv(self.gm, dec.d0.short, 1))
        self.d0 = GpuResUnit(self.gm, da^, db^, ds^, dec.d0.up_t, dec.d0.up_f)
        self.groups = List[List[GpuResUnit]]()
        self.last = List[GpuConv]()
        self.gin = List[Int]()
        for g in range(len(dec.group_state)):
            var units = List[GpuResUnit]()
            var t = 2
            for i in range(len(dec.group_state[g])):
                var u = _gpu_unit(self.gm, dec, g, i, t)
                t = u.b.tout
                units.append(u^)
            self.groups.append(units^)
            self.last.append(GpuConv(self.gm, dec.last_state[g], t))
            self.gin.append(self.gm.alloc(2 * 5 * 512))
        self.emb = self.gm.alloc(256)
        self.x0 = self.gm.alloc(2560)
        self.host_out = FBuf(2 * SS_STFT_FRAMES * SS_BINS * 2)
        self.gm.sync()

    def name(self) raises -> String:
        return self.gm.name()

    def reset(mut self) raises:
        self.first.reset(self.gm)
        self.b1.reset(self.gm)
        self.b2.reset(self.gm)
        self.iru.reset(self.gm)
        self.d0.reset(self.gm)
        for g in range(len(self.groups)):
            for i in range(len(self.groups[g])):
                self.groups[g][i].reset(self.gm)
            self.last[g].reset(self.gm)

    def run_embedding(mut self, emb: Int) raises -> Int:
        """256-d RVQ embedding (host) -> host address of both groups' output layers
        ([2][4 * 480 * 2]); pass the halves to `SpectroStreamDecoder.finish_frame`."""
        self.gm.upload(self.emb, emb, 256)
        var e = self.gm.ptr(self.emb)
        self.first.load_input(self.gm, e, False)
        self.first.run(self.gm)
        self.b1.load_input(self.gm, e, False)
        self.b1.run(self.gm)
        self.b2.load_input(self.gm, self.b1.out_ptr(self.gm), True)
        self.b2.run(self.gm)
        var x0 = self.gm.ptr(self.x0)
        self.gm.ctx.enqueue_function[_elu_kernel](self.first.out_ptr(self.gm), x0, Int32(2560), Int32(0),
                                                  grid_dim=_grid(2560), block_dim=_EW)
        self.gm.ctx.enqueue_function[_add_rows_kernel](x0, self.b2.out_ptr(self.gm), x0, Int32(2560), Int32(2560),
                                                       Int32(0), grid_dim=_grid(2560), block_dim=_EW)
        self.iru.step(self.gm, x0)
        self.d0.step(self.gm, self.iru.b.out_ptr(self.gm))
        var d0o = self.d0.b.out_ptr(self.gm)
        var per = SS_STFT_FRAMES * SS_BINS * 2
        for g in range(len(self.groups)):
            var gin = self.gm.ptr(self.gin[g])
            self.gm.copy_dev(d0o, gin, 2 * 5, 512, 1024, g * 512, 512, 0)
            var x = gin
            for i in range(len(self.groups[g])):
                self.groups[g][i].step(self.gm, x)
                x = self.groups[g][i].b.out_ptr(self.gm)
            self.last[g].load_input(self.gm, x, True)
            self.last[g].run(self.gm)
        for g in range(len(self.groups)):
            self.gm.download(self.last[g].out, self.host_out.addr + g * per * 4, per)
        return self.host_out.addr
