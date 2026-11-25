"""MAX GPU kernels for `flint.ml.diffusion` (channels-last activations, one float32 per element).

Matrix products (linear layers, implicit-GEMM convolutions, attention scores) go through
`flint.ml.gpu`'s tiled SGEMM; these kernels cover the rest of a UNet / VAE / CLIP:
GroupNorm (+ SiLU), activations, residual and per-image broadcast adds, GEGLU, head
split / merge for attention, row softmax (optionally causal) and buffer fills.
"""

from std.math import exp, erf, sqrt
from max.gpu import global_idx, thread_idx, block_idx, barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation
from flint.ml.gpu import FP

comptime DK_EW: Int = 256
comptime DK_RED: Int = 256

comptime ACT_SILU: Int = 0
comptime ACT_GELU: Int = 1
comptime ACT_QUICK_GELU: Int = 2


@always_inline
def _act(x: Float32, mode: Int) -> Float32:
    if mode == ACT_SILU:
        return x / (1.0 + exp(-x))
    if mode == ACT_GELU:
        return 0.5 * x * (1.0 + erf(x * 0.7071067811865476))
    return x / (1.0 + exp(-1.702 * x))


def _dk_act_kernel(x: FP, o: FP, n: Int32, mode: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        o.unsafe_offset(i).unsafe_store(_act(x.unsafe_offset(i).unsafe_load(), Int(mode)))


def _dk_add_kernel(a: FP, b: FP, o: FP, n: Int32, alpha: Float32):
    """o = a + alpha * b."""
    var i = Int(global_idx.x)
    if i < Int(n):
        o.unsafe_offset(i).unsafe_store(a.unsafe_offset(i).unsafe_load() + alpha * b.unsafe_offset(i).unsafe_load())


def _dk_scale_kernel(a: FP, o: FP, n: Int32, s: Float32):
    var i = Int(global_idx.x)
    if i < Int(n):
        o.unsafe_offset(i).unsafe_store(a.unsafe_offset(i).unsafe_load() * s)


def _dk_fill_kernel(o: FP, n: Int32, v: Float32):
    var i = Int(global_idx.x)
    if i < Int(n):
        o.unsafe_offset(i).unsafe_store(v)


def _dk_bcast_add_kernel(x: FP, t: FP, n: Int32, per_image: Int32, C: Int32):
    """x[b, p, c] += t[b, c] (a per-image channel vector, e.g. the time embedding)."""
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    var b = i // Int(per_image)
    var c = i % Int(C)
    x.unsafe_offset(i).unsafe_store(x.unsafe_offset(i).unsafe_load() + t.unsafe_offset(b * Int(C) + c).unsafe_load())


def _dk_geglu_kernel(h: FP, o: FP, rows: Int32, inner: Int32):
    """o[r, j] = h[r, j] * gelu(h[r, inner + j]) for h [rows, 2 * inner]."""
    var i = Int(global_idx.x)
    var I = Int(inner)
    if i >= Int(rows) * I:
        return
    var r = i // I
    var j = i % I
    var a = h.unsafe_offset(r * 2 * I + j).unsafe_load()
    var g = h.unsafe_offset(r * 2 * I + I + j).unsafe_load()
    o.unsafe_offset(i).unsafe_store(a * _act(g, ACT_GELU))


def _dk_group_norm_kernel(x: FP, g: FP, b: FP, o: FP, HW: Int32, C: Int32, G: Int32, eps: Float32, silu: Int32):
    """One block per (image, group): two-pass mean / variance over HW x C/G, affine, optional SiLU."""
    var sh = stack_allocation[DK_RED, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var GG = Int(G)
    var img = blk // GG
    var grp = blk % GG
    var CC = Int(C)
    var cg = CC // GG
    var hw = Int(HW)
    var cnt = hw * cg
    var base = img * hw * CC + grp * cg
    var s: Float32 = 0
    var e = t
    while e < cnt:
        var p = e // cg
        var c = e % cg
        s += x.unsafe_offset(base + p * CC + c).unsafe_load()
        e += DK_RED
    sh[unsafe_offset=t] = s
    barrier()
    var w = DK_RED // 2
    while w > 0:
        if t < w:
            sh[unsafe_offset=t] = sh[unsafe_offset=t] + sh[unsafe_offset=t + w]
        barrier()
        w //= 2
    var mean = sh[unsafe_offset=0] / Float32(cnt)
    barrier()
    var q: Float32 = 0
    e = t
    while e < cnt:
        var p = e // cg
        var c = e % cg
        var d = x.unsafe_offset(base + p * CC + c).unsafe_load() - mean
        q += d * d
        e += DK_RED
    sh[unsafe_offset=t] = q
    barrier()
    w = DK_RED // 2
    while w > 0:
        if t < w:
            sh[unsafe_offset=t] = sh[unsafe_offset=t] + sh[unsafe_offset=t + w]
        barrier()
        w //= 2
    var inv = 1.0 / sqrt(sh[unsafe_offset=0] / Float32(cnt) + eps)
    e = t
    while e < cnt:
        var p = e // cg
        var c = e % cg
        var ch = grp * cg + c
        var off = base + p * CC + c
        var y = (x.unsafe_offset(off).unsafe_load() - mean) * inv * g.unsafe_offset(ch).unsafe_load() \
            + b.unsafe_offset(ch).unsafe_load()
        if silu != 0:
            y = y / (1.0 + exp(-y))
        o.unsafe_offset(off).unsafe_store(y)
        e += DK_RED


def _dk_heads_kernel(src: FP, dst: FP, B: Int32, L: Int32, H: Int32, D: Int32, row_stride: Int32, col_off: Int32,
                     transpose: Int32):
    """src rows [B * L, row_stride] (head h at columns col_off + h * D) -> dst [B, H, L, D], or
    [B, H, D, L] when `transpose`."""
    var i = Int(global_idx.x)
    var BB = Int(B)
    var LL = Int(L)
    var HH = Int(H)
    var DD = Int(D)
    if i >= BB * LL * HH * DD:
        return
    var d = i % DD
    var r = i // DD
    var h = r % HH
    var r2 = r // HH
    var l = r2 % LL
    var b = r2 // LL
    var v = src.unsafe_offset((b * LL + l) * Int(row_stride) + Int(col_off) + h * DD + d).unsafe_load()
    if transpose != 0:
        dst.unsafe_offset(((b * HH + h) * DD + d) * LL + l).unsafe_store(v)
    else:
        dst.unsafe_offset(((b * HH + h) * LL + l) * DD + d).unsafe_store(v)


def _dk_merge_kernel(src: FP, dst: FP, B: Int32, L: Int32, H: Int32, D: Int32):
    """[B, H, L, D] -> rows [B * L, H * D]."""
    var i = Int(global_idx.x)
    var BB = Int(B)
    var LL = Int(L)
    var HH = Int(H)
    var DD = Int(D)
    if i >= BB * LL * HH * DD:
        return
    var d = i % DD
    var r = i // DD
    var h = r % HH
    var r2 = r // HH
    var l = r2 % LL
    var b = r2 // LL
    dst.unsafe_offset(i).unsafe_store(src.unsafe_offset(((b * HH + h) * LL + l) * DD + d).unsafe_load())


def _dk_softmax_kernel(x: FP, cols: Int32, scale: Float32, causal: Int32, row_mod: Int32):
    """In-place softmax(scale * row) per block; causal: keys after (row % row_mod) are masked."""
    var sh = stack_allocation[DK_RED, Float32, address_space=AddressSpace.SHARED]()
    var t = Int(thread_idx.x)
    var r = Int(block_idx.x)
    var n = Int(cols)
    var lim = n
    if causal != 0:
        lim = r % Int(row_mod) + 1
    var base = r * n
    var m: Float32 = -3.0e38
    var j = t
    while j < lim:
        m = max(m, x.unsafe_offset(base + j).unsafe_load() * scale)
        j += DK_RED
    sh[unsafe_offset=t] = m
    barrier()
    var w = DK_RED // 2
    while w > 0:
        if t < w:
            sh[unsafe_offset=t] = max(sh[unsafe_offset=t], sh[unsafe_offset=t + w])
        barrier()
        w //= 2
    var mx = sh[unsafe_offset=0]
    barrier()
    var s: Float32 = 0
    j = t
    while j < n:
        var e: Float32 = 0
        if j < lim:
            e = exp(x.unsafe_offset(base + j).unsafe_load() * scale - mx)
        x.unsafe_offset(base + j).unsafe_store(e)
        s += e
        j += DK_RED
    sh[unsafe_offset=t] = s
    barrier()
    w = DK_RED // 2
    while w > 0:
        if t < w:
            sh[unsafe_offset=t] = sh[unsafe_offset=t] + sh[unsafe_offset=t + w]
        barrier()
        w //= 2
    var inv = 1.0 / sh[unsafe_offset=0]
    j = t
    while j < n:
        x.unsafe_offset(base + j).unsafe_store(x.unsafe_offset(base + j).unsafe_load() * inv)
        j += DK_RED
