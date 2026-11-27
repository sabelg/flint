"""Hi3D image conditioning in Mojo. Port of Hi3D v1.0, see licenses/hi3d."""
from std.math import erf, exp, floor, sqrt
from flint.assets.image import NativeImage
from flint.core.jobs import JobPool
from flint.ml.ops import matmul_2d, layer_norm
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import Tensor
from flint.ml.trellis2_image import _resize
from flint.ml.vision_ops import vision_attention, weighted_linear

comptime WIDTH = 1024


def _frame(image: NativeImage) raises -> List[UInt32]:
    var w = image.width
    var h = image.height
    if not image.valid or w < 1 or h < 1 or w > 40000000 // h or len(image.pixels) != w * h:
        raise Error("Invalid reference image")
    var l = w
    var t = h
    var r = 0
    var b = 0
    for i in range(w * h):
        if image.pixels[i] >> 24 != 0:
            l = min(l, i % w)
            r = max(r, i % w + 1)
            t = min(t, i // w)
            b = max(b, i // w + 1)
    var side = max(r - l, b - t) if r > l else max(w, h)
    var size = Int(Float64(side) * 1.2)
    if size > 40000000 // size:
        raise Error("Padded reference image exceeds pixel budget")
    var out = List[UInt32](length=size * size, fill=0xFFFFFFFF)
    if r <= l:
        return out^
    var pad = (size - side) // 2
    var left = l - (side - (r - l)) // 2
    var top = t - (side - (b - t)) // 2
    for y in range(max(0, top), min(h, top + side)):
        for x in range(max(0, left), min(w, left + side)):
            var p = image.pixels[y * w + x]
            var a = p >> 24
            var v = UInt32(0xFF000000)
            comptime for c in range(3):
                var color = (p >> UInt32(c * 8)) & 255
                v |= ((color * a + 255 * (255 - a) + 127) // 255) << UInt32(c * 8)
            out[(y - top + pad) * size + x - left + pad] = v
    return out^


def _patches(frame: List[UInt32], clip: Bool) -> Tensor:
    var side = Int(sqrt(Float64(len(frame))))
    var grid = 16 if clip else 54
    var target = grid * 14
    var resized = List[UInt32]()
    if clip:
        resized = _resize(frame.copy(), side, side, target, target)
    var mean: List[Float32] = [0.48145466, 0.4578275, 0.40821073] if clip else [0.485, 0.456, 0.406]
    var std: List[Float32] = [0.26862954, 0.26130258, 0.27577711] if clip else [0.229, 0.224, 0.225]
    var out = Tensor.zeros([grid * grid, 588])
    for y in range(target):
        var fy = max(Float32(0), (Float32(y) + 0.5) * (Float32(side) / Float32(target)) - 0.5)
        var iy = Int(fy)
        var dy = fy - Float32(iy)
        for x in range(target):
            var fx = max(Float32(0), (Float32(x) + 0.5) * (Float32(side) / Float32(target)) - 0.5)
            var ix = Int(fx)
            var dx = fx - Float32(ix)
            var dst = ((y // 14) * grid + x // 14) * 588 + (y % 14) * 14 + x % 14
            comptime for c in range(3):
                comptime shift = (2 - c) * 8
                var val = Float32(0)
                if clip:
                    val = Float32((resized[y * target + x] >> UInt32(shift)) & 255) / 255
                else:
                    for j in range(2):
                        var src = min(side - 1, iy + j) * side
                        var a = Float32((frame[src + ix] >> UInt32(shift)) & 255) / 255
                        var b = Float32((frame[src + min(side - 1, ix + 1)] >> UInt32(shift)) & 255) / 255
                        val += ((1 - dx) * a + dx * b) * (dy if j else 1 - dy)
                out.f[dst + c * 196] = (val - mean[c]) / std[c]
    return out^


def _cubic(x: Float32) -> Float32:
    var a = abs(x)
    if a <= 1:
        return (1.25 * a - 2.25) * a * a + 1
    if a < 2:
        return ((-0.75 * a + 3.75) * a - 6) * a + 3
    return 0


def _positions(mut h: Tensor, pos: Tensor, grid: Int):
    var source = Int(sqrt(Float64(len(pos.f) // WIDTH - 1)))
    if source == grid:
        for i in range(len(h.f)):
            h.f[i] += pos.f[i]
        return
    for i in range(WIDTH):
        h.f[i] += pos.f[i]
    for y in range(grid):
        var fy = (Float32(y) + 0.5) * Float32(source) / Float32(grid) - 0.5
        var iy = Int(floor(fy))
        for x in range(grid):
            var fx = (Float32(x) + 0.5) * Float32(source) / Float32(grid) - 0.5
            var ix = Int(floor(fx))
            var dst = (1 + y * grid + x) * WIDTH
            for j in range(-1, 3):
                for k in range(-1, 3):
                    var w = _cubic(fy - Float32(iy + j)) * _cubic(fx - Float32(ix + k))
                    var src = (
                        1 + min(source - 1, max(0, iy + j)) * source + min(source - 1, max(0, ix + k))
                    ) * WIDTH
                    for c in range(WIDTH):
                        h.f[dst + c] += w * pos.f[src + c]


def _norm(weights: SafeTensorWeights, x: Tensor, name: String, eps: Float32) raises -> Tensor:
    return layer_norm(
        x, weights.tensor(name + ".weight", [WIDTH]), weights.tensor(name + ".bias", [WIDTH]), True, eps
    )


def _residual(mut x: Tensor, update: Tensor, weights: SafeTensorWeights, key: String, clip: Bool) raises:
    if clip:
        for i in range(len(x.f)):
            x.f[i] += update.f[i]
    else:
        var gain = weights.tensor(key, [WIDTH])
        for i in range(len(x.f)):
            x.f[i] += update.f[i] * gain.f[i % WIDTH]


def _encode(mut pool: JobPool, frame: List[UInt32], path: String, clip: Bool) raises -> Tensor:
    var weights = SafeTensorWeights(path, 2 << 30)
    var grid = 16 if clip else 54
    var patches = _patches(frame, clip)
    var h = Tensor.zeros([1 + grid * grid, WIDTH])
    var p = String("vision_model.embeddings.") if clip else String("embeddings.")
    var cls = weights.tensor(p + "class_embedding", [WIDTH]) if clip else weights.tensor(
        p + "cls_token", [1, 1, WIDTH]
    )
    for j in range(WIDTH):
        h.f[j] = cls.f[j]
    var patch = p + ("patch_embedding" if clip else "patch_embeddings.projection")
    var kernel = weights.tensor(patch + ".weight", [WIDTH, 3, 14, 14])
    var bias = Tensor() if clip else weights.tensor(patch + ".bias", [WIDTH])
    matmul_2d(
        pool,
        patches.fptr(),
        kernel.fptr(),
        h.fptr() + WIDTH * 4,
        0 if clip else bias.fptr(),
        grid * grid,
        588,
        WIDTH,
        True,
    )
    _ = kernel.fptr()
    _ = bias.fptr()
    _ = patches.fptr()
    var pos = weights.tensor(p + "position_embedding.weight", [257, WIDTH]) if clip else weights.tensor(
        p + "position_embeddings", [1, 1370, WIDTH]
    )
    _positions(h, pos, grid)
    var eps = Float32(1e-5) if clip else Float32(1e-6)
    if clip:
        h = _norm(weights, h, "vision_model.pre_layrnorm", eps)
    for i in range(24):
        var base = (
            (String("vision_model.encoder.layers.") if clip else String("encoder.layer.")) + String(i) + "."
        )
        var att = base + ("self_attn." if clip else "attention.attention.")
        var x = _norm(weights, h, base + ("layer_norm1" if clip else "norm1"), eps)
        var q = weighted_linear(weights, pool, x, att + ("q_proj" if clip else "query"), WIDTH)
        var k = weighted_linear(weights, pool, x, att + ("k_proj" if clip else "key"), WIDTH)
        var v = weighted_linear(weights, pool, x, att + ("v_proj" if clip else "value"), WIDTH)
        var ctx = vision_attention(pool, q, k, v, Tensor())
        var upd = weighted_linear(
            weights, pool, ctx, base + ("self_attn.out_proj" if clip else "attention.output.dense"), WIDTH
        )
        _residual(h, upd, weights, base + "layer_scale1.lambda1", clip)
        x = _norm(weights, h, base + ("layer_norm2" if clip else "norm2"), eps)
        var hidden = weighted_linear(weights, pool, x, base + "mlp.fc1", 4096)
        for j in range(len(hidden.f)):
            var a = hidden.f[j]
            hidden.f[j] = (
                a
                / (1 + exp(Float32(-1.702) * a)) if clip else a
                * 0.5
                * (1 + erf(a * Float32(0.7071067811865475)))
            )
        upd = weighted_linear(weights, pool, hidden, base + "mlp.fc2", WIDTH)
        _residual(h, upd, weights, base + "layer_scale2.lambda1", clip)
    if clip:
        var first = Tensor.zeros([1, WIDTH])
        for j in range(WIDTH):
            first.f[j] = h.f[j]
        return _norm(weights, first, "vision_model.post_layernorm", eps)
    return _norm(weights, h, "layernorm", eps)


def hi3d_condition(
    mut pool: JobPool, image: NativeImage, dino_path: String, clip_path: String
) raises -> Tensor:
    """Encode an oriented RGB(A) image as [1,2917,2048] normalized DINOv2 and CLIP features."""
    var frame = _frame(image)
    var dino = _encode(pool, frame, dino_path, False)
    var clip = _encode(pool, frame, clip_path, True)
    var out = Tensor.zeros([1, 2917, 2048])
    var clip_sum = Float32(0)
    for j in range(WIDTH):
        clip_sum += clip.f[j] * clip.f[j]
    var clip_norm = max(Float32(1e-12), sqrt(clip_sum))
    for r in range(2917):
        var sum = Float32(0)
        for j in range(WIDTH):
            var a = dino.f[r * WIDTH + j]
            sum += a * a
        var norm = max(Float32(1e-12), sqrt(sum))
        for j in range(WIDTH):
            out.f[r * 2048 + j] = dino.f[r * WIDTH + j] / norm
            out.f[r * 2048 + WIDTH + j] = clip.f[j] / clip_norm
    return out^
