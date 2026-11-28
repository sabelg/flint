"""CUDA tensor-core implementation of UltraTex's 128-by-64 sparse attention."""
from max.gpu import block_dim, block_idx, global_idx, thread_idx
from std.memory import bitcast
from max.gpu.compute.mma import mma
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.warp import shuffle_xor
from std.math import exp2
from std.utils import StaticTuple
from flint.ml.tensor import Tensor


def _center(
    packed: Pointer[BFloat16, MutAnyOrigin],
    mean: Pointer[BFloat16, MutAnyOrigin],
    rows: Int32,
    width: Int32,
):
    var c = Int(global_idx.x)
    if c < Int(width):
        var total = Float32(0)
        for r in range(Int(rows)):
            total += Float32(packed.unsafe_load((r * 3 + 1) * Int(width) + c))
        mean.unsafe_store(c, BFloat16(total / Float32(rows)))


def _pool_blocks(
    packed: Pointer[BFloat16, MutAnyOrigin],
    mean: Pointer[BFloat16, MutAnyOrigin],
    pooled: Pointer[BFloat16, MutAnyOrigin],
    rows: Int32,
    width: Int32,
    queries: Int32,
    keys: Int32,
):
    var i = Int(global_idx.x)
    var block = i // Int(width)
    if block >= Int(queries + keys):
        return
    var c = i % Int(width)
    var key = block >= Int(queries)
    var first = (block - Int(queries)) * 64 if key else block * 128
    var size = min(Int(rows) - first, 64 if key else 128)
    var total = Float32(0)
    for r in range(first, first + size):
        var x = packed.unsafe_load((3 * r + Int(key)) * Int(width) + c)
        if key:
            x = BFloat16(Float32(x) - Float32(mean.unsafe_load(c)))
        total += Float32(x)
    pooled.unsafe_store(i, BFloat16(total / Float32(size)))


def _rank_blocks(
    pooled: Pointer[BFloat16, MutAnyOrigin],
    scores: Pointer[BFloat16, MutAnyOrigin],
    heads: Int32,
    queries: Int32,
    keys: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var lane = i % 32
    var t = i // 32
    var key = t % Int(keys)
    var query = t // Int(keys) % Int(queries)
    var head = t // Int(keys) // Int(queries)
    if head >= Int(heads):
        return
    var q = (query * Int(heads) + head) * 128
    var k = ((Int(queries) + key) * Int(heads) + head) * 128
    var v = Float32(0)
    comptime for j in range(4):
        v += Float32(pooled.unsafe_load(q + lane + 32 * j)) * Float32(
            pooled.unsafe_load(k + lane + 32 * j)
        )
    comptime for shift in range(5):
        v += shuffle_xor(v, UInt32(1 << shift))
    if lane == 0:
        scores.unsafe_store(t, BFloat16(v))


def _select_blocks(
    scores: Pointer[BFloat16, MutAnyOrigin],
    lut: Pointer[Int32, MutAnyOrigin],
    groups: Int32,
    keys: Int32,
    keep: Int32,
):
    # Find the BF16 cut-off by counting ordered bit keys, then emit the CUDA
    # top-k order: strictly greater keys first, then equal keys in input order.
    var group = Int(global_idx.x)
    if group >= Int(groups):
        return
    var begin = group * Int(keys)
    # Binary search over the sortable 16-bit representation, including negatives.
    var lo = 0
    var hi = 65535
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        var bits = UInt16(mid ^ (32768 if mid >= 32768 else 65535))
        var threshold = Float32(bitcast[DType.bfloat16](bits))
        # NaN bit patterns lie beyond both infinite endpoints.
        var larger = 0
        if mid <= 127:
            larger = Int(keys)
        elif mid <= 65408:
            for j in range(Int(keys)):
                larger += Int(
                    Float32(scores.unsafe_load(begin + j)) >= threshold
                )
        if larger >= Int(keep):
            lo = mid
        else:
            hi = mid - 1
    var bits = UInt16(lo ^ (32768 if lo >= 32768 else 65535))
    var cut = Float32(bitcast[DType.bfloat16](bits))
    var used = 0
    for pass_ in range(2):
        for j in range(Int(keys)):
            var val = Float32(scores.unsafe_load(begin + j))
            if val > cut if pass_ == 0 else val == cut:
                lut.unsafe_store(group * Int(keep) + used, Int32(j))
                used += 1
                if used == Int(keep):
                    return


def _sparse(
    packed: Pointer[BFloat16, MutAnyOrigin],
    lut: Pointer[Int32, MutAnyOrigin],
    dest: Pointer[BFloat16, MutAnyOrigin],
    rows: Int32,
    heads: Int32,
    queries: Int32,
    keep: Int32,
):
    var thread = Int(global_idx.x)
    var lane = thread % 32
    var group = lane // 4
    var pair = (lane % 4) * 2
    var head = thread // 32 % Int(heads)
    var row = (thread // 32 // Int(heads)) * 16
    if row >= Int(rows):
        return
    var width = Int(heads) * 128
    var base = head * 128
    var qa = StaticTuple[SIMD[DType.bfloat16, 8], 8](fill=0)
    comptime for inner in range(8):
        comptime for j in range(8):
            var r = row + group + 8 * ((j // 2) % 2)
            if r < Int(rows):
                qa[inner][j] = packed.unsafe_load(
                    3 * r * width
                    + base
                    + inner * 16
                    + pair
                    + j % 2
                    + 8 * (j // 4)
                )
    var sums = StaticTuple[SIMD[DType.float32, 4], 16](fill=0)
    var peak = SIMD[DType.float32, 2](-Float32.MAX)
    var mass = SIMD[DType.float32, 2](0)
    for block in range(Int(keep)):
        var first = 64 * Int(
            lut.unsafe_load(
                (head * Int(queries) + row // 128) * Int(keep) + block
            )
        )
        var prob = StaticTuple[SIMD[DType.float32, 4], 8](fill=0)
        var next_peak = peak
        comptime for col in range(8):
            var kr = first + col * 8 + group
            comptime for inner in range(8):
                var kb = SIMD[DType.bfloat16, 4](0)
                if kr < Int(rows):
                    comptime for j in range(4):
                        kb[j] = packed.unsafe_load(
                            (3 * kr + 1) * width
                            + base
                            + inner * 16
                            + pair
                            + j % 2
                            + 8 * (j // 2)
                        )
                var prev = prob[col]
                mma(prob[col], qa[inner], kb, prev)
            comptime for j in range(4):
                var score = prob[col][j] * Float32(0.12751743082459868)
                if first + col * 8 + pair + j % 2 >= Int(rows):
                    score = -Float32.MAX
                prob[col][j] = score
                next_peak[j // 2] = max(next_peak[j // 2], score)
        comptime for half in range(2):
            var v = next_peak[half]
            v = max(v, shuffle_xor(v, UInt32(1)))
            next_peak[half] = max(v, shuffle_xor(v, UInt32(2)))
        var alpha = exp2(peak - next_peak)
        peak = next_peak
        var local_mass = SIMD[DType.float32, 2](0)
        comptime for col in range(8):
            comptime for j in range(4):
                var p = exp2(prob[col][j] - peak[j // 2])
                prob[col][j] = p
                local_mass[j // 2] += p
        comptime for half in range(2):
            var v = local_mass[half]
            v += shuffle_xor(v, UInt32(1))
            v += shuffle_xor(v, UInt32(2))
            mass[half] = mass[half] * alpha[half] + v
        comptime for col in range(16):
            var tile = SIMD[DType.float32, 4](0)
            comptime for inner in range(4):
                var pa = SIMD[DType.bfloat16, 8](0)
                var vb = SIMD[DType.bfloat16, 4](0)
                comptime for j in range(8):
                    pa[j] = BFloat16(prob[inner * 2 + j // 4][j % 4])
                comptime for j in range(4):
                    var r = first + inner * 16 + pair + j % 2 + 8 * (j // 2)
                    if r < Int(rows):
                        vb[j] = packed.unsafe_load(
                            (3 * r + 2) * width + base + col * 8 + group
                        )
                var prev = tile
                mma(tile, pa, vb, prev)
            comptime for j in range(4):
                sums[col][j] = sums[col][j] * alpha[j // 2] + tile[j]
    comptime for col in range(16):
        comptime for j in range(4):
            var r = row + group + 8 * (j // 2)
            if r < Int(rows):
                dest.unsafe_store(
                    r * width + base + col * 8 + pair + j % 2,
                    BFloat16(sums[col][j] / mass[j // 2]),
                )


def gpu_sparse_buffer(
    ctx: DeviceContext,
    src: DeviceBuffer[DType.bfloat16],
    rows: Int,
    heads: Int,
    keep: Int,
) raises -> DeviceBuffer[DType.bfloat16]:
    var width = heads * 128
    var keys = (rows + 63) // 64
    var queries = (rows + 127) // 128
    var groups = heads * queries
    var mean = ctx.enqueue_create_buffer[DType.bfloat16](width)
    var pooled = ctx.enqueue_create_buffer[DType.bfloat16](
        (keys + queries) * width
    )
    var scores = ctx.enqueue_create_buffer[DType.bfloat16](groups * keys)
    var lut = ctx.enqueue_create_buffer[DType.int32](groups * keep)
    var output = ctx.enqueue_create_buffer[DType.bfloat16](rows * width)
    ctx.enqueue_function[_center](
        src.unsafe_ptr(),
        mean.unsafe_ptr(),
        Int32(rows),
        Int32(width),
        grid_dim=(width + 127) // 128,
        block_dim=128,
    )
    ctx.enqueue_function[_pool_blocks](
        src.unsafe_ptr(),
        mean.unsafe_ptr(),
        pooled.unsafe_ptr(),
        Int32(rows),
        Int32(width),
        Int32(queries),
        Int32(keys),
        grid_dim=((keys + queries) * width + 127) // 128,
        block_dim=128,
    )
    ctx.enqueue_function[_rank_blocks](
        pooled.unsafe_ptr(),
        scores.unsafe_ptr(),
        Int32(heads),
        Int32(queries),
        Int32(keys),
        grid_dim=(groups * keys + 3) // 4,
        block_dim=128,
    )
    ctx.enqueue_function[_select_blocks](
        scores.unsafe_ptr(),
        lut.unsafe_ptr(),
        Int32(groups),
        Int32(keys),
        Int32(keep),
        grid_dim=(groups + 31) // 32,
        block_dim=32,
    )
    ctx.enqueue_function[_sparse](
        src.unsafe_ptr(),
        lut.unsafe_ptr(),
        output.unsafe_ptr(),
        Int32(rows),
        Int32(heads),
        Int32(queries),
        Int32(keep),
        grid_dim=(((rows + 15) // 16) * heads + 3) // 4,
        block_dim=128,
    )
    return output^


def gpu_sparse_attention(
    ctx: DeviceContext, x: Tensor, heads: Int, keep: Int
) raises -> Tensor:
    var rows = x.shape[0]
    var width = heads * 128
    var src = ctx.enqueue_create_buffer[DType.bfloat16](len(x.f))
    var upload = ctx.enqueue_create_host_buffer[DType.bfloat16](len(x.f))
    for i in range(len(x.f)):
        upload[i] = BFloat16(x.f[i])
    ctx.enqueue_copy(src, upload)
    var output = gpu_sparse_buffer(ctx, src, rows, heads, keep)
    var result = Tensor.zeros([rows, width])
    with output.map_to_host() as host:
        for i in range(len(result.f)):
            result.f[i] = Float32(host[i])
    return result^
