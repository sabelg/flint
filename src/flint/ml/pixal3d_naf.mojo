"""Pixal3D's NAF-conditioned sparse shape and material features. See licenses/pixal3d."""
from std.utils import StaticTuple
from std.math import cos, exp, sin, sqrt
from flint.ml.pixal3d import PixalCamera, project_patch_features, projection_pixels
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import Tensor
from flint.ml.trellis2_dino import PREFIX
from flint.ml.ultratex_gpu_vae import UltraTexVaeGpu


def _group_silu(mut x: Tensor, weights: SafeTensorWeights, key: String) raises:
    var channels = x.shape[2]
    var span = channels // 8
    var count = Float64(x.numel() // 8)
    var mean = StaticTuple[Float64, 8](fill=0)
    var variance = StaticTuple[Float64, 8](fill=0)
    var gain = weights.tensor(key + ".weight", [channels])
    var bias = weights.tensor(key + ".bias", [channels])
    for i in range(len(x.f)):
        mean[i % channels // span] += Float64(x.f[i]) / count
    for i in range(len(x.f)):
        var g = i % channels // span
        var delta = Float64(x.f[i]) - mean[g]
        variance[g] += delta * delta / count
    for g in range(8):
        variance[g] = 1 / sqrt(variance[g] + 1e-5)
    for i in range(len(x.f)):
        var c = i % channels
        var g = c // span
        var value = Float32((Float64(x.f[i]) - mean[g]) * variance[g]) * gain.f[c] + bias.f[c]
        x.f[i] = value / (1 + exp(-value))


def _pool(x: Tensor, size: Int) -> Tensor:
    # All Pixal3D stage resolutions divide evenly.
    var stride = x.shape[0] // size
    var channels = x.shape[2]
    var out = Tensor.zeros([size, size, channels])
    for y in range(x.shape[0]):
        for col in range(x.shape[1]):
            var src = (y * x.shape[1] + col) * channels
            var dst = ((y // stride) * size + col // stride) * channels
            for c in range(channels):
                out.f[dst + c] += x.f[src + c]
    for i in range(len(out.f)):
        out.f[i] /= Float32(stride * stride)
    return out^


def _branch(
    mut gpu: UltraTexVaeGpu, weights: SafeTensorWeights, image: Tensor, key: String, kernel: Int, size: Int
) raises -> Tensor:
    var x = gpu.conv[True](weights, image, key + ".0", 128, kernel, 1)
    for i in range(1, 3):
        var block = key + "." + String(i)
        _group_silu(x, weights, block + ".norm1")
        x = gpu.conv[True](weights, x, block + ".conv1", 128, kernel, 1)
        _group_silu(x, weights, block + ".norm2")
        x = gpu.conv[True](weights, x, block + ".conv2", 128, kernel, 1)
    if x.shape[0] != size:
        return _pool(x, size)
    return x^


def _guide(normalized: Tensor, weights: SafeTensorWeights, target: Int) raises -> Tensor:
    var size = normalized.shape[2]
    var image = Tensor.zeros([size, size, 3])
    var mean: List[Float32] = [0.485, 0.456, 0.406]
    var std: List[Float32] = [0.229, 0.224, 0.225]
    for i in range(size * size):
        for c in range(3):
            image.f[3 * i + c] = normalized.fget(c * size * size + i) * std[c] + mean[c]
    var gpu = UltraTexVaeGpu()
    var detail = _branch(gpu, weights, image, "image_encoder.encoder", 1, target)
    var semantic = _branch(gpu, weights, image, "image_encoder.sem_encoder", 3, target)
    var periods = weights.tensor("image_encoder.rope.periods", [16])
    var guide = Tensor.zeros([target, target, 256])
    for i in range(target * target):
        for j in range(32):
            var axis = i // target if j < 16 else i % target
            var coord = 2 * (Float32(axis) + 0.5) / Float32(target) - 1
            var angle = Float32(6.283185307179586) * coord / periods.f[j % 16]
            var c = cos(angle)
            var s = sin(angle)
            for h in range(4):
                var k = (h % 2) * 64 + j
                var a = detail.f[128 * i + k] if h < 2 else semantic.f[128 * i + k]
                var b = detail.f[128 * i + k + 32] if h < 2 else semantic.f[128 * i + k + 32]
                guide.f[256 * i + 64 * h + j] = a * c - b * s
                guide.f[256 * i + 64 * h + j + 32] = a * s + b * c
    return guide^


def _sample_pixel(queries: Tensor, keys: Tensor, tokens: Tensor, pixel: Int, mut out: Tensor, row: Int):
    var target = queries.shape[0]
    var patches = keys.shape[0]
    var step = target // patches
    var start_y = min(max(pixel // target // step - 4, 0), patches - 9)
    var start_x = min(max(pixel % target // step - 4, 0), patches - 9)
    for h in range(4):
        var scores = StaticTuple[Float32, 81](fill=0)
        var peak = -Float32.MAX
        var q = pixel * 256 + h * 64
        for tap in range(81):
            var index = (start_y + tap // 9) * patches + start_x + tap % 9
            var key = index * 256 + h * 64
            var score: Float32 = 0
            for j in range(64):
                score += queries.f[q + j] * keys.f[key + j]
            scores[tap] = score / 8
            peak = max(peak, scores[tap])
        var mass: Float32 = 0
        for tap in range(81):
            scores[tap] = exp(scores[tap] - peak)
            mass += scores[tap]
        for tap in range(81):
            var index = (start_y + tap // 9) * patches + start_x + tap % 9
            var probability = scores[tap] / mass
            var value = (PREFIX + index) * 1024 + h * 256
            for j in range(256):
                out.f[row * 1024 + h * 256 + j] += probability * tokens.fget(value + j)


def sparse_condition(
    tokens: Tensor,
    normalized: Tensor,
    coords: Tensor,
    grid: Int,
    camera: PixalCamera,
    weights: SafeTensorWeights,
    target: Int,
) raises -> Tensor:
    """Return [N,2048] low/high projected features for Pixal3D's single-view SLat flow."""
    if normalized.rank() != 4 or coords.rank() != 2:
        raise Error("Expected normalized image and sparse XYZ coordinates")
    var size = normalized.shape[2]
    if (
        (size != 512 and size != 1024)
        or (target != 512 and target != size)
        or normalized.shape != [1, 3, size, size]
        or not normalized.is_float()
        or not normalized.has_valid_storage()
        or tokens.shape != [1, PREFIX + (size // 16) ** 2, 1024]
    ):
        raise Error("Invalid Pixal3D NAF image, token or target dimensions")
    var low = project_patch_features(tokens, size, grid, camera, coords)
    var pixels = projection_pixels(size, grid, camera, coords)
    var queries = _guide(normalized, weights, target)
    var keys = _pool(queries, size // 16)
    var lookup = List[Int](length=target * target, fill=-1)
    var indices = List[Int]()
    var neighbors = List[Int]()
    var blends = List[Float32]()
    for i in range(coords.shape[0]):
        var x = min(
            Float32(target - 1),
            max(Float32(0), (pixels.f[2 * i] + 0.5) * Float32(target) / Float32(size) - 0.5),
        )
        var y = min(
            Float32(target - 1),
            max(Float32(0), (pixels.f[2 * i + 1] + 0.5) * Float32(target) / Float32(size) - 0.5),
        )
        var dx = x - Float32(Int(x))
        var dy = y - Float32(Int(y))
        for corner in range(4):
            var col = min(target - 1, Int(x) + corner % 2)
            var line = min(target - 1, Int(y) + corner // 2)
            var p = line * target + col
            if lookup[p] == -1:
                lookup[p] = len(indices)
                indices.append(p)
            neighbors.append(lookup[p])
            blends.append((dx if corner % 2 else 1 - dx) * (dy if corner // 2 else 1 - dy))
    var high = Tensor.zeros([len(indices), 1024])
    for i in range(len(indices)):
        _sample_pixel(queries, keys, tokens, indices[i], high, i)
    var out = Tensor.zeros([coords.shape[0], 2048])
    for i in range(coords.shape[0]):
        for j in range(1024):
            out.f[i * 2048 + j] = low.f[i * 1024 + j]
            var value: Float32 = 0
            for corner in range(4):
                value += high.f[neighbors[i * 4 + corner] * 1024 + j] * blends[i * 4 + corner]
            out.f[i * 2048 + 1024 + j] = value
    return out^
