"""Pixal3D single-view sparse shape and texture VAEs, with shared subdivision masks."""
from std.ffi import external_call
from std.math import exp, sqrt
from max.gpu import global_idx
from max.gpu.compute.mma import mma
from max.gpu.host import DeviceContext
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.hi3d_sparse import SparseGrid
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import DT_I32, Tensor
from flint.ml.vision_ops import weighted_linear


def _sparse_product(
    x: Pointer[Float16, MutAnyOrigin],
    w: Pointer[Float16, MutAnyOrigin],
    neighbors: Pointer[Int32, MutAnyOrigin],
    result: Pointer[Float32, MutAnyOrigin],
    rows: Int32,
    inputs: Int32,
    outputs: Int32,
    taps: Int32,
):
    var thread = Int(global_idx.x)
    var lane = thread % 32
    var groups = Int(outputs) // 8
    var row = 16 * (thread // 32 // groups)
    var col = 8 * (thread // 32 % groups)
    if row >= Int(rows):
        return
    var group = lane // 4
    var pair = 2 * (lane % 4)
    var sums = SIMD[DType.float32, 4](0)
    for tap in range(Int(taps)):
        var src = SIMD[DType.int32, 2](-1)
        comptime for half in range(2):
            var r = row + group + 8 * half
            if r < Int(rows):
                src[half] = neighbors.unsafe_load(r * 27 + tap) if taps == 27 else Int32(r)
        for first in range(0, Int(inputs), 16):
            var a = SIMD[DType.float16, 8](0)
            var b = SIMD[DType.float16, 4](0)
            comptime for j in range(8):
                var r = Int(src[(j // 2) % 2])
                if r >= 0:
                    a[j] = x.unsafe_load(r * Int(inputs) + first + pair + j % 2 + 8 * (j // 4))
            comptime for j in range(4):
                b[j] = w.unsafe_load(
                    ((col + group) * Int(taps) + tap) * Int(inputs) + first + pair + j % 2 + 8 * (j // 2)
                )
            var previous = sums
            mma(sums, a, b, previous)
    comptime for j in range(4):
        var r = row + group + 8 * (j // 2)
        if r < Int(rows):
            result.unsafe_store(r * Int(outputs) + col + pair + j % 2, sums[j])


def _neighbors(grid: SparseGrid) -> List[Int32]:
    var n = grid.xyz.shape[0]
    var indices = List[Int32](capacity=27 * n)
    for r in range(n):
        for tap in range(27):
            indices.append(
                Int32(
                    grid.get(
                        grid.xyz.i[3 * r] + tap // 9 - 1,
                        grid.xyz.i[3 * r + 1] + tap // 3 % 3 - 1,
                        grid.xyz.i[3 * r + 2] + tap % 3 - 1,
                    )
                )
            )
    return indices^


def _product(
    ctx: DeviceContext,
    weights: SafeTensorWeights,
    key: String,
    x: Tensor,
    indices: List[Int32],
    co: Int,
    taps: Int,
) raises -> Tensor:
    var rows = x.shape[0]
    var ci = x.shape[1]
    var shape: List[Int] = [co, 3, 3, 3, ci] if taps == 27 else [co, ci]
    var offset, dtype = weights.storage(key + ".weight", shape)
    var bias = weights.tensor(key + ".bias", [co])
    var kernel = ctx.enqueue_create_host_buffer[DType.float16](co * taps * ci)
    if dtype == "F16":
        _ = external_call["memcpy", Int](
            Int(kernel.unsafe_ptr()), weights.address() + offset, co * taps * ci * 2
        )
    else:
        var original = weights.tensor(key + ".weight", shape^)
        for i in range(original.numel()):
            kernel[i] = Float16(original.f[i])
    var dk = ctx.enqueue_create_buffer[DType.float16](co * taps * ci)
    ctx.enqueue_copy(dk, kernel)
    var host = ctx.enqueue_create_host_buffer[DType.float16](rows * ci)
    for i in range(rows * ci):
        host[i] = Float16(x.f[i])
    var dx = ctx.enqueue_create_buffer[DType.float16](rows * ci)
    ctx.enqueue_copy(dx, host)
    var count = len(indices) if taps == 27 else 1
    var map = ctx.enqueue_create_host_buffer[DType.int32](count)
    if taps == 27:
        _ = external_call["memcpy", Int](Int(map.unsafe_ptr()), Int(indices.unsafe_ptr()), 4 * count)
    var dm = ctx.enqueue_create_buffer[DType.int32](count)
    ctx.enqueue_copy(dm, map)
    var dy = ctx.enqueue_create_buffer[DType.float32](rows * co)
    var warps = ((rows + 15) // 16) * (co // 8)
    ctx.enqueue_function[_sparse_product](
        dx.unsafe_ptr(),
        dk.unsafe_ptr(),
        dm.unsafe_ptr(),
        dy.unsafe_ptr(),
        Int32(rows),
        Int32(ci),
        Int32(co),
        Int32(taps),
        grid_dim=(warps + 3) // 4,
        block_dim=128,
    )
    var h = Tensor.zeros([rows, co])
    with dy.map_to_host() as result:
        for i in range(rows * co):
            h.f[i] = Float32(Float16(result[i] + Float32(Float16(bias.f[i % co]))))
    return h^


def _norm(mut x: Tensor, gain: Tensor, bias: Tensor, eps: Float32, silu: Bool):
    var c = x.shape[1]
    for r in range(x.shape[0]):
        var avg: Float32 = 0
        for j in range(c):
            avg += x.f[r * c + j]
        avg /= Float32(c)
        var ss: Float32 = 0
        for j in range(c):
            var diff = x.f[r * c + j] - avg
            ss += diff * diff
        var inv = 1 / sqrt(ss / Float32(c) + eps)
        for j in range(c):
            var value = (x.f[r * c + j] - avg) * inv
            if gain.rank():
                value = value * gain.f[j] + bias.f[j]
            if silu:
                value = Float32(Float16(value))
                value = value / (1 + exp(-value))
            x.f[r * c + j] = value


def _grid(coords: Tensor, edge: Int) raises -> SparseGrid:
    if coords.rank() != 2 or coords.shape[1] != 4 or coords.dtype != DT_I32 or not coords.has_valid_storage():
        raise Error("Expected batch/XYZ sparse decoder coordinates")
    var xyz = List[Int](capacity=coords.shape[0] * 3)
    for r in range(coords.shape[0]):
        if coords.i[r * 4] != 0:
            raise Error("Pixal3D sparse decoder expects batch zero")
        for j in range(3):
            xyz.append(coords.i[r * 4 + 1 + j])
    return SparseGrid(Tensor.ints(DT_I32, [coords.shape[0], 3], xyz^), edge)


def decode_slat(
    weights: SafeTensorWeights,
    var coords: Tensor,
    latent: Tensor,
    edge: Int,
    mut masks: List[Tensor],
    texture: Bool = False,
    coords_only: Bool = False,
) raises -> Tuple[Tensor, Tensor]:
    """Decode denormalized SLat rows; texture follows the four masks emitted by shape."""
    if (
        edge < 1
        or edge > 96
        or coords.rank() != 2
        or latent.shape != [coords.shape[0], 32]
        or not latent.is_float()
        or not latent.has_valid_storage()
        or latent.ext != 0
        or len(masks) != (4 if texture else 0)
    ):
        raise Error("Invalid shape or texture SLat input")
    var gpu = DeviceContext()
    var pool = JobPool()
    _ = pool.start(default_worker_count())
    var grid = _grid(coords, edge)
    var x = weighted_linear(weights, pool, latent, "from_latent", 1024)
    for i in range(len(x.f)):
        x.f[i] = Float32(Float16(x.f[i]))
    var blocks: List[Int] = [4, 16, 8, 4]
    var indices = _neighbors(grid)
    for level in range(4):
        var c = 1024 >> level
        var prefix = "blocks." + String(level) + "."
        for j in range(blocks[level]):
            var key = prefix + String(j)
            var h = _product(gpu, weights, key + ".conv", x, indices, c, 27)
            _norm(
                h,
                weights.tensor(key + ".norm.weight", [c]),
                weights.tensor(key + ".norm.bias", [c]),
                1e-6,
                False,
            )
            h = _product(gpu, weights, key + ".mlp.0", h, indices, 4 * c, 1)
            for i in range(len(h.f)):
                h.f[i] /= 1 + exp(-h.f[i])
            h = _product(gpu, weights, key + ".mlp.2", h, indices, c, 1)
            for i in range(len(x.f)):
                x.f[i] = Float32(Float16(x.f[i] + h.f[i]))
        var key = prefix + String(blocks[level])
        if not texture:
            masks.append(_product(gpu, weights, key + ".to_subdiv", x, indices, 8, 1))
        if (
            masks[level].shape != [x.shape[0], 8]
            or not masks[level].is_float()
            or not masks[level].has_valid_storage()
        ):
            raise Error("Incompatible texture subdivision mask")
        var children = List[Int]()
        var sites = List[Int]()
        for r in range(x.shape[0]):
            for k in range(8):
                if masks[level].fget(r * 8 + k) > 0:
                    children.append(r * 8 + k)
                    sites.append(0)
                    for axis in range(3):
                        sites.append(2 * coords.i[r * 4 + 1 + axis] + ((k >> axis) & 1))
        if not len(children):
            raise Error("Pixal3D decoder predicted an empty subdivision")
        coords = Tensor.ints(DT_I32, [len(children), 4], sites^)
        if coords_only and level == 3:
            return (coords^, Tensor())
        var h = x.copy()
        _norm(
            h,
            weights.tensor(key + ".norm1.weight", [c]),
            weights.tensor(key + ".norm1.bias", [c]),
            1e-6,
            True,
        )
        h = _product(gpu, weights, key + ".conv1", h, indices, 4 * c, 27)
        var out_c = c // 2
        var expanded = Tensor.zeros([len(children), out_c])
        var skip = Tensor.zeros([len(children), out_c])
        for r in range(len(children)):
            for j in range(out_c):
                expanded.f[r * out_c + j] = h.f[children[r] * out_c + j]
                skip.f[r * out_c + j] = x.f[children[r] * (c // 8) + j // 4]
        _norm(expanded, Tensor(), Tensor(), 1e-6, True)
        grid = _grid(coords, edge << (level + 1))
        indices = _neighbors(grid)
        x = _product(gpu, weights, key + ".conv2", expanded, indices, out_c, 27)
        for i in range(len(x.f)):
            x.f[i] = Float32(Float16(x.f[i] + skip.f[i]))
    _norm(x, Tensor(), Tensor(), 1e-5, False)
    var output = weighted_linear(weights, pool, x, "output_layer", 6 if texture else 7)
    return (coords^, output^)


def denormalize_slat(var x: Tensor, texture: Bool) raises -> Tensor:
    if x.rank() != 2 or x.shape[1] != 32 or not x.is_float() or x.ext != 0 or not x.has_valid_storage():
        raise Error("Expected owned [N,32] SLat rows")
    var shape_mean: List[Float32] = [
        0.781296,
        0.018091,
        -0.495192,
        -0.558457,
        1.06053,
        0.093252,
        1.518149,
        -0.933218,
        -0.732996,
        2.604095,
        -0.118341,
        -2.143904,
        0.495076,
        -2.179512,
        -2.130751,
        -0.996944,
        0.261421,
        -2.217463,
        1.260067,
        -0.150213,
        3.790713,
        1.481266,
        -1.046058,
        -1.523667,
        -0.059621,
        2.22078,
        1.621212,
        0.87723,
        0.567247,
        -3.175944,
        -3.186688,
        1.578665,
    ]
    var tex_mean: List[Float32] = [
        3.501659,
        2.212398,
        2.226094,
        0.251093,
        -0.026248,
        -0.687364,
        0.439898,
        -0.928075,
        0.029398,
        -0.339596,
        -0.869527,
        1.038479,
        -0.972385,
        0.126042,
        -1.129303,
        0.455149,
        -1.209521,
        2.069067,
        0.544735,
        2.569128,
        -0.323407,
        2.293,
        -1.925608,
        -1.217717,
        1.213905,
        0.971588,
        -0.023631,
        0.10675,
        2.021786,
        0.250524,
        -0.662387,
        -0.768862,
    ]
    var shape_std: List[Float32] = [
        5.972266,
        4.706852,
        5.44501,
        5.209927,
        5.32022,
        4.547237,
        5.020802,
        5.444004,
        5.226681,
        5.683095,
        4.831436,
        5.286469,
        5.652043,
        5.367606,
        5.525084,
        4.730578,
        4.805265,
        5.124013,
        5.530808,
        5.619001,
        5.10393,
        5.41767,
        5.269677,
        5.547194,
        5.634698,
        5.235274,
        6.110351,
        5.511298,
        6.237273,
        4.879207,
        5.347008,
        5.405691,
    ]
    var tex_std: List[Float32] = [
        2.665652,
        2.743913,
        2.765121,
        2.595319,
        3.037293,
        2.291316,
        2.144656,
        2.911822,
        2.969419,
        2.501689,
        2.154811,
        3.163343,
        2.621215,
        2.381943,
        3.186697,
        3.021588,
        2.295916,
        3.234985,
        3.233086,
        2.26014,
        2.874801,
        2.810596,
        3.29272,
        2.674999,
        2.680878,
        2.372054,
        2.451546,
        2.353556,
        2.995195,
        2.379849,
        2.786195,
        2.77519,
    ]
    for i in range(len(x.f)):
        var j = i % 32
        x.f[i] = x.f[i] * tex_std[j] + tex_mean[j] if texture else x.f[i] * shape_std[j] + shape_mean[j]
    return x^


def cascade_coords(decoded: Tensor, resolution: Int, max_tokens: Int = 49152) raises -> Tuple[Tensor, Int]:
    """Quantize the decoded 512 surface and emit torch.unique's lexicographic order."""
    if (
        (resolution != 1024 and resolution != 1536)
        or max_tokens < 1
        or decoded.rank() != 2
        or decoded.shape[1] != 4
        or decoded.dtype != DT_I32
        or not decoded.has_valid_storage()
    ):
        raise Error("Invalid Pixal3D cascade coordinates or resolution")
    var edge = resolution // 16
    while edge >= 64:
        var occupied = List[Bool](length=edge**3, fill=False)
        var count = 0
        for r in range(decoded.shape[0]):
            var key = 0
            if decoded.i[r * 4] != 0:
                raise Error("Expected batch zero for cascade coordinates")
            for axis in range(3):
                var p = decoded.i[r * 4 + 1 + axis]
                if p < 0 or p >= 512:
                    raise Error("Decoded coordinate outside the 512 grid")
                # Round-to-nearest-even using integers at exact half-way cases.
                var numerator = (2 * p + 1) * (edge - 1)
                var q = numerator // 1024
                var remainder = numerator % 1024
                q += Int(remainder > 512 or (remainder == 512 and q % 2 == 1))
                key = key * edge + q
            if not occupied[key]:
                occupied[key] = True
                count += 1
        if count < max_tokens or edge == 64:
            var values = List[Int](capacity=count * 4)
            for k in range(len(occupied)):
                if occupied[k]:
                    values.extend([0, k // (edge * edge), k // edge % edge, k % edge])
            return (Tensor.ints(DT_I32, [count, 4], values^), edge)
        edge -= 8
    raise Error("Invalid Pixal3D cascade")
