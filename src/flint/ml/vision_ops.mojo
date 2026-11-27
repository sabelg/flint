"""Shared float32 ViT attention and checkpoint-backed dense projection."""
from std.math import exp
from flint.core.jobs import JobPool
from flint.ml.ops import matmul_2d
from flint.ml.tensor import Tensor
from flint.ml.safetensors import SafeTensorWeights

comptime WIDTH = 1024
comptime HEAD_WIDTH = 64


def _rotate(mut x: Tensor, rotations: Tensor):
    var prefix = x.shape[0] - rotations.shape[0]
    for row in range(prefix, x.shape[0]):
        var start = row * HEAD_WIDTH
        var phase = (row - prefix) * HEAD_WIDTH
        for j in range(HEAD_WIDTH // 2):
            var a = x.f[start + j]
            var b = x.f[start + 32 + j]
            var c = rotations.f[phase + j]
            var s = rotations.f[phase + 32 + j]
            x.f[start + j] = a * c - b * s
            x.f[start + 32 + j] = b * c + a * s


def vision_attention(mut pool: JobPool, q: Tensor, k: Tensor, v: Tensor, rotations: Tensor) -> Tensor:
    var rows = q.shape[0]
    var result = Tensor.zeros([rows, WIDTH])
    var qh = Tensor.zeros([rows, HEAD_WIDTH])
    var kh = Tensor.zeros([rows, HEAD_WIDTH])
    var vh = Tensor.zeros([rows, HEAD_WIDTH])
    # Bound the attention score buffer for a 1024px reference image.
    var tile = min(rows, 128)
    var scores = Tensor.zeros([tile, rows])
    var attended = Tensor.zeros([tile, HEAD_WIDTH])
    for head in range(WIDTH // HEAD_WIDTH):
        for row in range(rows):
            var src = row * WIDTH + head * HEAD_WIDTH
            for j in range(HEAD_WIDTH):
                var dst = row * HEAD_WIDTH + j
                qh.f[dst] = q.f[src + j]
                kh.f[dst] = k.f[src + j]
                vh.f[dst] = v.f[src + j]
        if len(rotations.f) != 0:
            _rotate(qh, rotations)
            _rotate(kh, rotations)
        for first in range(0, rows, tile):
            var n = min(tile, rows - first)
            matmul_2d(
                pool,
                qh.fptr() + first * HEAD_WIDTH * 4,
                kh.fptr(),
                scores.fptr(),
                0,
                n,
                HEAD_WIDTH,
                rows,
                True,
            )
            for r in range(n):
                var base = r * rows
                var peak = scores.f[base]
                for j in range(1, rows):
                    peak = max(peak, scores.f[base + j])
                var total = Float32(0)
                for j in range(rows):
                    var value = exp((scores.f[base + j] - peak) * Float32(0.125))
                    scores.f[base + j] = value
                    total += value
                for j in range(rows):
                    scores.f[base + j] /= total
            matmul_2d(pool, scores.fptr(), vh.fptr(), attended.fptr(), 0, n, rows, HEAD_WIDTH, False)
            for r in range(n):
                for j in range(HEAD_WIDTH):
                    result.f[(first + r) * WIDTH + head * HEAD_WIDTH + j] = attended.f[r * HEAD_WIDTH + j]
    return result^


def weighted_linear(
    weights: SafeTensorWeights, mut pool: JobPool, x: Tensor, key: String, outputs: Int, bias: Bool = True
) raises -> Tensor:
    var cols = x.shape[1]
    var w = weights.tensor(key + ".weight", [outputs, cols])
    var b = weights.tensor(key + ".bias", [outputs]) if bias else Tensor()
    var out = Tensor.zeros([x.shape[0], outputs])
    matmul_2d(pool, x.fptr(), w.fptr(), out.fptr(), b.fptr() if bias else 0, x.shape[0], cols, outputs, True)
    # Raw addresses passed to worker threads must remain live through their join.
    _ = w.fptr()
    _ = b.fptr()
    return out^
