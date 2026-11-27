"""UltraTex per-head top-k block attention, including truncated edge blocks.
Mojo adaptation of the SLA block selection. See licenses/ultratex.
"""
from std.math import exp, sqrt
from std.memory import bitcast
from flint.core.detsort import stable_order
from flint.core.jobs import JobPool
from flint.ml.ops import matmul_2d
from flint.ml.tensor import DT_F32, Tensor


@always_inline
def _precision(a: Float32, bf16: Bool) -> Float32:
    return Float32(BFloat16(a)) if bf16 else a


def _pool(x: Tensor, block: Int, center: Bool, bf16: Bool) -> Tensor:
    var n = x.shape[0]
    var d = x.shape[1]
    var out = Tensor.zeros([(n + block - 1) // block, d])
    var mean = Tensor.zeros([d])
    if center:
        for i in range(n):
            for j in range(d):
                mean.f[j] += x.f[i * d + j]
        for j in range(d):
            mean.f[j] = _precision(mean.f[j] / Float32(n), bf16)
    for i in range(n):
        for j in range(d):
            out.f[(i // block) * d + j] += _precision(x.f[i * d + j] - mean.f[j], bf16)
    for b in range(out.shape[0]):
        var inv = 1 / Float32(min(block, n - b * block))
        for j in range(d):
            out.f[b * d + j] = _precision(out.f[b * d + j] * inv, bf16)
    return out^


def block_attention(
    mut pool: JobPool,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    ratio: Float32 = 0.2,
    query_block: Int = 128,
    key_block: Int = 64,
    bf16: Bool = False,
) raises -> Tensor:
    """Single-head [tokens,channels] attention with an online softmax across selected blocks."""
    if (
        len(q.shape) != 2
        or q.shape != k.shape
        or q.shape != v.shape
        or q.dtype != DT_F32
        or k.dtype != DT_F32
        or v.dtype != DT_F32
        or len(q.f) != q.numel()
        or len(k.f) != len(q.f)
        or len(v.f) != len(q.f)
    ):
        raise Error("Incompatible Q/K/V tensors")
    var n = q.shape[0]
    var d = q.shape[1]
    if min(n, d) < 1 or min(query_block, key_block) < 1 or not (ratio > 0 and ratio <= 1):
        raise Error("Invalid attention block configuration")
    var bq = min(n, query_block)
    var bk = min(n, key_block)
    var keys = (n + bk - 1) // bk
    var keep = Int(ratio * Float32(keys))
    if keep == 0:
        raise Error("Sparse attention selected zero blocks; use a larger ratio for short sequences")
    return _selected_attention(pool, q, k, v, bq, bk, keep, bf16)


def _selected_attention(
    mut pool: JobPool, q: Tensor, k: Tensor, v: Tensor, bq: Int, bk: Int, keep: Int, bf16: Bool
) -> Tensor:
    var n = q.shape[0]
    var d = q.shape[1]
    var keys = (n + bk - 1) // bk
    var pq = _pool(q, bq, False, bf16)
    var pk = _pool(k, bk, True, bf16)
    var scores = Tensor.zeros([bq, bk])
    var weighted = Tensor.zeros([bq, d])
    var ranks = Tensor.zeros([keys])
    var order_keys = List[Int](length=keys, fill=0)
    var out = Tensor.zeros([n, d])
    var inv = 1 / sqrt(Float32(d))
    for start in range(0, n, bq):
        var rows = min(bq, n - start)
        matmul_2d(pool, pq.fptr() + (start // bq) * d * 4, pk.fptr(), ranks.fptr(), 0, 1, d, keys, True)
        for j in range(keys):
            # Sort descending IEEE float values with stable block-index ties.
            var bits = bitcast[DType.uint32](_precision(ranks.f[j], bf16))
            order_keys[j] = -Int(~bits if bits >> 31 else bits ^ 0x80000000)
        var order = stable_order(order_keys)
        # Upstream requests an unsorted top-k lookup. Visit the selected blocks
        # in stable score order on both backends for repeatable accumulation.
        var peaks = List[Float32](length=rows, fill=-Float32.MAX)
        var totals = List[Float32](length=rows, fill=0)
        for selected in range(keep):
            var offset = order[selected] * bk
            var cols = min(bk, n - offset)
            matmul_2d(
                pool,
                q.fptr() + start * d * 4,
                k.fptr() + offset * d * 4,
                scores.fptr(),
                0,
                rows,
                d,
                cols,
                True,
            )
            for r in range(rows):
                var new_peak = peaks[r]
                for c in range(cols):
                    new_peak = max(new_peak, scores.f[r * cols + c] * inv)
                var adjust = exp(peaks[r] - new_peak)
                totals[r] *= adjust
                for j in range(d):
                    out.f[(start + r) * d + j] *= adjust
                for c in range(cols):
                    var value = exp(scores.f[r * cols + c] * inv - new_peak)
                    scores.f[r * cols + c] = _precision(value, bf16)
                    totals[r] += value
                peaks[r] = new_peak
            matmul_2d(
                pool, scores.fptr(), v.fptr() + offset * d * 4, weighted.fptr(), 0, rows, cols, d, False
            )
            for i in range(rows * d):
                out.f[start * d + i] += weighted.f[i]
        for r in range(rows):
            for j in range(d):
                out.f[(start + r) * d + j] = _precision(out.f[(start + r) * d + j] / totals[r], bf16)
    _ = pq.fptr()
    _ = pk.fptr()
    return out^
