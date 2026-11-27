"""Sparse Hi3D VAE decoder, ported from Hi3D v1.0. See licenses/hi3d."""
from std.math import exp, erf, sqrt
from flint.core.detsort import stable_order
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.hi3d_sparse import SparseGrid, sparse_conv3
from flint.ml.hi3d_transformer import _half, _norm
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import Tensor, DT_I32, DT_F32
from flint.ml.vision_ops import weighted_linear


def _group_swish(w: SafeTensorWeights, var x: Tensor, key: String) raises -> Tensor:
    var c = x.shape[1]
    var n = x.shape[0]
    var groups = min(32, c // 4)
    var span = c // groups
    var a = w.tensor(key + ".weights", [1, c])
    var b = w.tensor(key + ".bias", [1, c])
    var sums = List[Float64](length=groups, fill=0)
    var squares = List[Float64](length=groups, fill=0)
    for i in range(n):
        for j in range(c):
            var v = Float64(x.f[i * c + j])
            sums[j // span] += v
            squares[j // span] += v * v
    for g in range(groups):
        var count = Float64(n * span) + 1e-6
        sums[g] /= count
        squares[g] = 1 / sqrt(max(Float64(0), squares[g] / count - sums[g] * sums[g]) + 1e-6)
    for i in range(n):
        for j in range(c):
            var g = j // span
            var h = (x.f[i * c + j] - Float32(sums[g])) * Float32(squares[g]) * a.f[j] + b.f[j]
            x.f[i * c + j] = h / (1 + exp(-h))
    return x^


def _conv(
    w: SafeTensorWeights, mut p: JobPool, g: SparseGrid, x: Tensor, key: String, c: Int
) raises -> Tensor:
    return sparse_conv3(
        p, g, x, w.tensor(key + ".weight", [c, 3, 3, 3, x.shape[1]]), w.tensor(key + ".bias", [c])
    )


def _res(
    w: SafeTensorWeights, mut p: JobPool, g: SparseGrid, var x: Tensor, key: String, c: Int
) raises -> Tensor:
    var h = _group_swish(w, x.copy(), key + "block.0")
    h = _conv(w, p, g, h, key + "block.2", c)
    h = _group_swish(w, h^, key + "block.3")
    h = _conv(w, p, g, h, key + "block.5", c)
    if x.shape[1] != c:
        x = weighted_linear(w, p, x, key + "res_conv", c)
        _half(x)
    for i in range(len(x.f)):
        h.f[i] += x.f[i]
    return h^


def _window(qkv: Tensor, g: SparseGrid, transpose: Bool) -> Tensor:
    var n = g.xyz.shape[0]
    var codes = List[Int](capacity=n)
    for i in range(n):
        var x = g.xyz.i[i * 3 + Int(transpose)]
        var y = g.xyz.i[i * 3 + Int(not transpose)]
        var z = g.xyz.i[i * 3 + 2]
        var code = 0
        comptime for bit in range(6):
            code |= (((x >> bit) & 1) * 4 + ((y >> bit) & 1) * 2 + ((z >> bit) & 1)) << (3 * bit)
        codes.append(code)
    var order = stable_order(codes)
    var result = Tensor.zeros([n, 512])
    var scores = List[Float32](length=48, fill=0)
    for first in range(0, n, 48):
        var count = min(48, n - first)
        var patch = min(48, n)
        for j in range(count):
            var row = order[first + j]
            for head in range(32):
                var peak = Float32(-1e30)
                for k in range(patch):
                    var ix = first + k if k < count else first + k - 48
                    var base = order[ix] * 1536 + 512 + head * 16
                    var dot = Float32(0)
                    for t in range(16):
                        dot += qkv.f[row * 1536 + head * 16 + t] * qkv.f[base + t]
                    scores[k] = dot * 0.25
                    peak = max(peak, scores[k])
                var total = Float32(0)
                for k in range(patch):
                    scores[k] = exp(scores[k] - peak)
                    total += scores[k]
                for k in range(patch):
                    var ix = first + k if k < count else first + k - 48
                    var base = order[ix] * 1536 + 1024 + head * 16
                    for t in range(16):
                        result.f[row * 512 + head * 16 + t] += (scores[k] / total) * qkv.f[base + t]
    _half(result)
    return result^


def _mid(
    w: SafeTensorWeights, mut pool: JobPool, g: SparseGrid, var x: Tensor, key: String, other: Bool
) raises -> Tensor:
    var h = _conv(w, pool, g, x, key + "cpe.0", 512)
    h = weighted_linear(w, pool, h, key + "cpe.1", 512)
    _half(h)
    h = _norm(w, h, key + "cpe.2", 1e-5)
    for i in range(len(x.f)):
        x.f[i] += h.f[i]
    h = _norm(w, x, key + "norm1.0", 1e-5)
    h = weighted_linear(w, pool, h, key + "attn.qkv", 1536)
    _half(h)
    h = _window(h, g, other)
    h = weighted_linear(w, pool, h, key + "attn.proj", 512)
    _half(h)
    for i in range(len(x.f)):
        x.f[i] += h.f[i]
    h = weighted_linear(w, pool, _norm(w, x, key + "norm2.0", 1e-5), key + "mlp.0.fc1", 2048)
    _half(h)
    for i in range(len(h.f)):
        var a = h.f[i]
        h.f[i] = Float32(Float16(a * 0.5 * (1 + erf(Float32(0.70710678118) * a))))
    h = weighted_linear(w, pool, h, key + "mlp.0.fc2", 512)
    _half(h)
    for i in range(len(x.f)):
        x.f[i] += h.f[i]
    return x^


def _unpool(
    w: SafeTensorWeights, mut pool: JobPool, mut grid: SparseGrid, var x: Tensor, key: String
) raises -> Tensor:
    var c = x.shape[1]
    var pred = _conv(
        w, pool, grid, _group_swish(w, x.copy(), key + "cls_head.fc.0"), key + "cls_head.fc.2", 8
    )
    var values = weighted_linear(w, pool, x, key + "proj", c * 8, False)
    var coords = List[Int]()
    var feats = List[Float32]()
    for i in range(x.shape[0]):
        for child in range(8):
            if pred.f[i * 8 + child] <= 0:
                continue
            comptime for dim in range(3):
                coords.append(grid.xyz.i[i * 3 + dim] * 2 + ((child >> (2 - dim)) & 1))
            for j in range(c):
                feats.append(Float32(Float16(values.f[(i * 8 + child) * c + j])))
    var n = len(coords) // 3
    if n == 0:
        raise Error("Hi3D decoder predicted no occupied children")
    grid = SparseGrid(Tensor.ints(DT_I32, [n, 3], coords^), grid.edge * 2)
    return Tensor.floats([n, c], feats^)


struct Hi3DField(Movable):
    var grid: SparseGrid
    var attributes: Tensor

    def __init__(out self, var coords: Tensor, var latents: Tensor, weights_path: String) raises:
        var g = SparseGrid(coords^, 64)
        if (
            latents.dtype != DT_F32
            or len(latents.shape) != 2
            or latents.shape[0] != g.xyz.shape[0]
            or latents.shape[1] != 32
            or len(latents.f) != latents.numel()
        ):
            raise Error("Expected one 32-channel latent per input voxel")
        var w = SafeTensorWeights(weights_path, 1 << 30)
        var p = JobPool()
        _ = p.start(default_worker_count())
        var h = _conv(w, p, g, latents, "decoder.z_to_f.conv.0", 512)
        h = _res(w, p, g, h^, "decoder.mid.block_in.", 512)
        h = _mid(w, p, g, h^, "decoder.mid.block0.", False)
        h = _mid(w, p, g, h^, "decoder.mid.block1.", True)
        h = _res(w, p, g, h^, "decoder.mid.block_out.", 512)
        for stage in range(3, -1, -1):
            var key = "decoder.dec.dec" + String(stage) + "."
            var width = 512 if stage == 3 else (256 if stage == 2 else 128)
            h = _res(w, p, g, h^, key + "block0.", width)
            h = _res(w, p, g, h^, key + "block1.", width)
            if stage > 0:
                h = _unpool(w, p, g, h^, key + "up.upsample.")
        h = _group_swish(w, h^, "decoder.conv_out.conv.0")
        self.attributes = _conv(w, p, g, h, "decoder.conv_out.conv.2", 40)
        self.grid = g^
