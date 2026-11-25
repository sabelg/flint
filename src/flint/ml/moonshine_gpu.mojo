"""MAX GPU kernels for `flint.ml.moonshine`.

`MoonshineGpu` owns a `GpuMatmul` (resident weights, pooled device activations, tiled / split-K
SGEMM, LayerNorm) and adds the few kernels Moonshine needs beyond it:

  - `attention`: one block per (query row, head) over a key range: a sliding window with
    look-ahead for the encoder, the KV cache for decoder self-attention, all valid encoder
    frames for cross-attention; scores staged in shared memory, block-reduced softmax.
  - `rope`: interleaved partial rotary embedding at one position.
  - `im2col`: causal stride-2, kernel-5 Conv1d patches (time-major), so the convs are GEMMs.
  - `act`: GELU / SiLU with a frame mask, `gate` (SiLU-gated MLP), `add`, `embed`.
"""

from std.math import exp, erf, sqrt, sin, cos
from max.gpu import global_idx, thread_idx, block_idx, barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation
from flint.ml.gpu import GpuMatmul, FP

comptime MS_THREADS: Int = 128
comptime MS_MAXK: Int = 4096     # keys one query row may see (the model's position limit)
comptime MS_EW: Int = 256

comptime ACT_GELU: Int = 0
comptime ACT_SILU: Int = 1
comptime ACT_MASK: Int = 2


def _ms_attn_kernel(q: FP, k: FP, v: FP, o: FP, heads: Int32, hd: Int32, qstride: Int32, kvstride: Int32,
                    row0: Int32, left: Int32, right: Int32, valid: Int32, scale: Float32):
    """Block (r, h): query row i = row0 + r attends keys [lo, hi): lo = i - left + 1 (0 when
    left <= 0), hi = min(valid, i + max(right, 1)) (valid when right < 0)."""
    var sc = stack_allocation[MS_MAXK, Float32, address_space=AddressSpace.SHARED]()
    var red = stack_allocation[MS_THREADS, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var H = Int(heads)
    var r = Int(block_idx.x) // H
    var h = Int(block_idx.x) % H
    var i = Int(row0) + r
    var D = Int(hd)
    var lo = 0
    if Int(left) > 0:
        lo = max(0, i - Int(left) + 1)
    var hi = Int(valid)
    if Int(right) >= 0:
        hi = min(hi, i + max(Int(right), 1))
    var n = hi - lo
    var qb = r * Int(qstride) + h * D
    var mx: Float32 = -3.0e38
    var j = t
    while j < n:
        var kb = (lo + j) * Int(kvstride) + h * D
        var s: Float32 = 0
        for d in range(D):
            s += q.unsafe_offset(qb + d).unsafe_load() * k.unsafe_offset(kb + d).unsafe_load()
        s *= scale
        sc[unsafe_offset=j] = s
        mx = max(mx, s)
        j += MS_THREADS
    red[unsafe_offset=t] = mx
    barrier()
    var w = MS_THREADS // 2
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
        var e = exp(sc[unsafe_offset=j] - m)
        sc[unsafe_offset=j] = e
        tot += e
        j += MS_THREADS
    red[unsafe_offset=t] = tot
    barrier()
    w = MS_THREADS // 2
    while w > 0:
        if t < w:
            red[unsafe_offset=t] = red[unsafe_offset=t] + red[unsafe_offset=t + w]
        barrier()
        w //= 2
    var den = red[unsafe_offset=0]
    var d = t
    while d < D:
        var acc: Float32 = 0
        for jj in range(n):
            acc += sc[unsafe_offset=jj] * v.unsafe_offset((lo + jj) * Int(kvstride) + h * D + d).unsafe_load()
        o.unsafe_offset(qb + d).unsafe_store(acc / den if n > 0 else Float32(0))
        d += MS_THREADS


def _ms_add_kernel(dst: FP, src: FP, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        dst.unsafe_offset(i).unsafe_store(dst.unsafe_offset(i).unsafe_load() + src.unsafe_offset(i).unsafe_load())


def _ms_act_kernel(x: FP, n: Int32, cols: Int32, valid_rows: Int32, mode: Int32):
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    var y: Float32 = 0
    if i // Int(cols) < Int(valid_rows):
        var v = x.unsafe_offset(i).unsafe_load()
        if Int(mode) == ACT_GELU:
            y = 0.5 * v * (1.0 + erf(v * 0.7071067811865476))
        elif Int(mode) == ACT_SILU:
            y = v / (1.0 + exp(-v))
        else:
            y = v
    x.unsafe_offset(i).unsafe_store(y)


def _ms_gate_kernel(f: FP, g: FP, inter: Int32):
    var i = Int(global_idx.x)
    if i < Int(inter):
        var gate = f.unsafe_offset(Int(inter) + i).unsafe_load()
        g.unsafe_offset(i).unsafe_store(gate / (1.0 + exp(-gate)) * f.unsafe_offset(i).unsafe_load())


def _ms_im2col_kernel(x: FP, col: FP, T: Int32, cin: Int32, To: Int32):
    """col[t, c * 5 + j] = x[2 t + j - 4, c] (zero outside [0, T))."""
    var e = Int(global_idx.x)
    var C = Int(cin)
    var kc = C * 5
    if e >= Int(To) * kc:
        return
    var t = e // kc
    var rem = e % kc
    var c = rem // 5
    var j = rem % 5
    var s = 2 * t + j - 4
    var v: Float32 = 0
    if s >= 0 and s < Int(T):
        v = x.unsafe_offset(s * C + c).unsafe_load()
    col.unsafe_offset(e).unsafe_store(v)


def _ms_rope_kernel(x: FP, heads: Int32, hd: Int32, rot: Int32, pos: Int32, invf: FP):
    var e = Int(global_idx.x)
    var half = Int(rot) // 2
    if e >= Int(heads) * half:
        return
    var h = e // half
    var i = e % half
    var ang = Float32(Int(pos)) * invf.unsafe_offset(i).unsafe_load()
    var c = cos(ang)
    var s = sin(ang)
    var b = h * Int(hd) + 2 * i
    var x0 = x.unsafe_offset(b).unsafe_load()
    var x1 = x.unsafe_offset(b + 1).unsafe_load()
    x.unsafe_offset(b).unsafe_store(x0 * c - x1 * s)
    x.unsafe_offset(b + 1).unsafe_store(x1 * c + x0 * s)


def _ms_embed_kernel(emb: FP, dst: FP, token: Int32, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        dst.unsafe_offset(i).unsafe_store(emb.unsafe_offset(Int(token) * Int(n) + i).unsafe_load())


struct MoonshineGpu(Movable):
    var mm: GpuMatmul

    def __init__(out self, enc_layers: Int, dec_layers: Int) raises:
        self.mm = GpuMatmul()

    def name(self) raises -> String:
        return self.mm.name()

    def attention(mut self, q: FP, k: FP, v: FP, o: FP, rows: Int, heads: Int, hd: Int, qstride: Int,
                  kvstride: Int, row0: Int, left: Int, right: Int, valid: Int, scale: Float32) raises:
        var span = valid if left <= 0 else min(valid, left + max(right, 1))
        if span > MS_MAXK:
            raise Error("moonshine GPU: " + String(span) + " keys exceed " + String(MS_MAXK))
        self.mm.ctx.enqueue_function[_ms_attn_kernel](
            q, k, v, o, Int32(heads), Int32(hd), Int32(qstride), Int32(kvstride), Int32(row0), Int32(left),
            Int32(right), Int32(valid), scale, grid_dim=rows * heads, block_dim=MS_THREADS)

    def add(mut self, dst: FP, src: FP, n: Int) raises:
        self.mm.ctx.enqueue_function[_ms_add_kernel](dst, src, Int32(n), grid_dim=(n + MS_EW - 1) // MS_EW,
                                                     block_dim=MS_EW)

    def act(mut self, x: FP, rows: Int, cols: Int, valid_rows: Int, mode: Int) raises:
        var n = rows * cols
        self.mm.ctx.enqueue_function[_ms_act_kernel](x, Int32(n), Int32(cols), Int32(valid_rows), Int32(mode),
                                                     grid_dim=(n + MS_EW - 1) // MS_EW, block_dim=MS_EW)

    def gate(mut self, f: FP, g: FP, inter: Int) raises:
        self.mm.ctx.enqueue_function[_ms_gate_kernel](f, g, Int32(inter), grid_dim=(inter + MS_EW - 1) // MS_EW,
                                                      block_dim=MS_EW)

    def im2col(mut self, x: FP, col: FP, T: Int, cin: Int, To: Int) raises:
        var n = To * cin * 5
        self.mm.ctx.enqueue_function[_ms_im2col_kernel](x, col, Int32(T), Int32(cin), Int32(To),
                                                        grid_dim=(n + MS_EW - 1) // MS_EW, block_dim=MS_EW)

    def rope(mut self, x: FP, heads: Int, hd: Int, rot: Int, pos: Int, invf: FP) raises:
        var n = heads * (rot // 2)
        self.mm.ctx.enqueue_function[_ms_rope_kernel](x, Int32(heads), Int32(hd), Int32(rot), Int32(pos), invf,
                                                      grid_dim=(n + MS_EW - 1) // MS_EW, block_dim=MS_EW)

    def embed(mut self, emb: FP, dst: FP, token: Int, n: Int) raises:
        self.mm.ctx.enqueue_function[_ms_embed_kernel](emb, dst, Int32(token), Int32(n),
                                                       grid_dim=(n + MS_EW - 1) // MS_EW, block_dim=MS_EW)
