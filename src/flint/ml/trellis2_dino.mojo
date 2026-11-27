"""DINOv3 ViT-L/16 image conditioning for TRELLIS.2, single-image float32 inference."""

from std.math import cos, erf, exp, pow, sin
from flint.assets.image import NativeImage
from flint.core.jobs import JobPool
from flint.ml.vision_ops import vision_attention, weighted_linear
from flint.ml.ops import layer_norm, matmul_2d
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import Tensor
from flint.ml.trellis2_image import prepare_trellis_image

comptime WIDTH = 1024
comptime HEAD_WIDTH = 64
comptime PREFIX = 5


def _apply_scale(mut residual: Tensor, update: Tensor, scale: Tensor):
    for row in range(residual.shape[0]):
        for col in range(WIDTH):
            var i = row * WIDTH + col
            residual.f[i] += update.f[i] * scale.f[col]


def _rope(grid: Int) -> Tensor:
    var rotations = Tensor.zeros([grid * grid, HEAD_WIDTH])
    for p in range(grid * grid):
        for j in range(HEAD_WIDTH // 2):
            var index = p // grid if j < 16 else p % grid
            var coord = 2 * ((Float32(index) + 0.5) / Float32(grid)) - 1
            var freq = 1 / pow(Float32(100), Float32(j % 16) / 16)
            var angle = Float32(6.283185307179586) * coord * freq
            rotations.f[p * HEAD_WIDTH + j] = cos(angle)
            rotations.f[p * HEAD_WIDTH + 32 + j] = sin(angle)
    return rotations^


struct TrellisImageEncoder(Movable):
    var weights: SafeTensorWeights

    def __init__(out self, path: String) raises:
        self.weights = SafeTensorWeights(path, 2 << 30)

    def _norm(self, x: Tensor, key: String) raises -> Tensor:
        var a = self.weights.tensor(key + ".weight", [WIDTH])
        var b = self.weights.tensor(key + ".bias", [WIDTH])
        return layer_norm(x, a, b, True, Float32(1e-5))

    def encode(self, mut pool: JobPool, image: NativeImage, size: Int = 512) raises -> Tensor:
        """Return [1,5+(size/16)^2,1024], preserving CLS and all four register tokens."""
        var pixels = prepare_trellis_image(image, size)
        var grid = size // 16
        var patches = Tensor.zeros([grid * grid, 768])
        for p in range(grid * grid):
            for ch in range(3):
                for dy in range(16):
                    var base = (ch * size + (p // grid) * 16 + dy) * size + (p % grid) * 16
                    for dx in range(16):
                        patches.f[p * 768 + ch * 256 + dy * 16 + dx] = pixels.f[base + dx]
        var count = grid * grid + PREFIX
        var h = Tensor.zeros([count, WIDTH])
        var cls = self.weights.tensor("embeddings.cls_token", [1, 1, WIDTH])
        var regs = self.weights.tensor("embeddings.register_tokens", [1, 4, WIDTH])
        for i in range(WIDTH):
            h.f[i] = cls.f[i]
        for i in range(4 * WIDTH):
            h.f[WIDTH + i] = regs.f[i]
        var kernel = self.weights.tensor("embeddings.patch_embeddings.weight", [WIDTH, 3, 16, 16])
        var bias = self.weights.tensor("embeddings.patch_embeddings.bias", [WIDTH])
        matmul_2d(pool, patches.fptr(), kernel.fptr(), h.fptr() + PREFIX * WIDTH * 4, bias.fptr(),
                  grid * grid, 768, WIDTH, True)
        _ = patches.fptr()
        _ = kernel.fptr()
        _ = bias.fptr()
        var rotations = _rope(grid)
        for i in range(24):
            var prefix = "layer." + String(i) + "."
            var att = prefix + "attention."
            var pre = self._norm(h, prefix + "norm1")
            var q = weighted_linear(self.weights, pool, pre, att + "q_proj", WIDTH)
            var k = weighted_linear(self.weights, pool, pre, att + "k_proj", WIDTH, False)
            var v = weighted_linear(self.weights, pool, pre, att + "v_proj", WIDTH)
            var context = vision_attention(pool, q, k, v, rotations)
            var update = weighted_linear(self.weights, pool, context, att + "o_proj", WIDTH)
            _apply_scale(h, update, self.weights.tensor(prefix + "layer_scale1.lambda1", [WIDTH]))
            var wide = weighted_linear(self.weights, pool, self._norm(h, prefix + "norm2"), prefix + "mlp.up_proj", 4096)
            for j in range(len(wide.f)):
                var x = wide.f[j]
                wide.f[j] = x * Float32(0.5) * (1 + erf(x * Float32(0.7071067811865475)))
            var mlp = weighted_linear(self.weights, pool, wide, prefix + "mlp.down_proj", WIDTH)
            _apply_scale(h, mlp, self.weights.tensor(prefix + "layer_scale2.lambda1", [WIDTH]))
        # TRELLIS applies a final parameter-free layer_norm to the encoder states.
        var scale = Tensor.floats([WIDTH], List[Float32](length=WIDTH, fill=1))
        var out = layer_norm(h, scale, Tensor(), False, Float32(1e-5))
        out.shape = [1, count, WIDTH]
        return out^
