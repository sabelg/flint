"""Pixal3D single-view structure, shape and texture flow transformers."""
from std.math import cos, exp, sin, sqrt, tanh
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.ops import layer_norm, MAX_RANGE_ELEMS
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import DT_I32, Tensor
from flint.ml.trellis2_flow import StructureVelocityModel
from flint.ml.ultratex_gpu import UltraTexGpu
from flint.ml.ultratex_gpu_attention import gpu_sparse_attention
from flint.ml.vision_ops import weighted_linear

comptime WIDTH = 1536
comptime HEADS = 12
comptime DIM = 128
comptime SITES = 4096


@always_inline
def _bf(x: Float32) -> Float32:
    return Float32(BFloat16(x))


def _silu(mut x: Tensor):
    for i in range(len(x.f)):
        x.f[i] /= 1 + exp(-x.f[i])


def _rope(coords: Tensor) -> Tensor:
    var angles = Tensor.zeros([coords.shape[0], DIM])
    for r in range(coords.shape[0]):
        for pair in range(DIM // 2):
            var axis = pair // 21
            var xyz = coords.i[r * 4 + 1 + min(axis, 2)]
            var theta = Float32(xyz) * exp(Float32(-9.210340372) * Float32(pair % 21) / 21)
            if axis == 3:
                theta = 0
            angles.f[r * DIM + pair * 2] = cos(theta)
            angles.f[r * DIM + pair * 2 + 1] = sin(theta)
    return angles^


def _normalize_heads(mut x: Tensor, part: Int, gamma: Tensor, rotations: Tensor):
    var stride = x.shape[1]
    for r in range(x.shape[0]):
        for h in range(HEADS):
            var start = r * stride + part * WIDTH + h * DIM
            var ss: Float32 = 0
            for j in range(DIM):
                ss += x.f[start + j] * x.f[start + j]
            var inv = sqrt(Float32(DIM)) / max(sqrt(ss), Float32(1e-12))
            for j in range(0, DIM, 2):
                var a = _bf(x.f[start + j] * inv * gamma.f[h * DIM + j])
                var b = _bf(x.f[start + j + 1] * inv * gamma.f[h * DIM + j + 1])
                if rotations.rank() != 0:
                    var c = rotations.f[r * DIM + j]
                    var s = rotations.f[r * DIM + j + 1]
                    x.f[start + j] = _bf(a * c - b * s)
                    x.f[start + j + 1] = _bf(a * s + b * c)
                else:
                    x.f[start + j] = a
                    x.f[start + j + 1] = b


def _global_attention(q: Tensor, kv: Tensor) -> Tensor:
    var output = Tensor.zeros(q.shape.copy())
    var count = kv.shape[0]
    var scores = List[Float32](length=count, fill=0)
    for r in range(q.shape[0]):
        for h in range(HEADS):
            var base = r * WIDTH + h * DIM
            var peak = -Float32.MAX
            for t in range(count):
                var dot: Float32 = 0
                for j in range(DIM):
                    dot += q.f[base + j] * kv.f[2 * t * WIDTH + h * DIM + j]
                scores[t] = dot / sqrt(Float32(DIM))
                peak = max(peak, scores[t])
            var total: Float32 = 0
            for t in range(count):
                scores[t] = exp(scores[t] - peak)
                total += scores[t]
            for j in range(DIM):
                var value: Float32 = 0
                for t in range(count):
                    value += (scores[t] / total) * kv.f[(2 * t + 1) * WIDTH + h * DIM + j]
                output.f[base + j] = _bf(value)
    return output^


def _norm_rows(x: Tensor, gain: Tensor, bias: Tensor, affine: Bool, eps: Float32) raises -> Tensor:
    if x.numel() <= MAX_RANGE_ELEMS:
        return layer_norm(x, gain, bias, affine, eps)
    var output = Tensor.zeros(x.shape.copy())
    var width = x.shape[1]
    var batch = MAX_RANGE_ELEMS // width
    for first in range(0, x.shape[0], batch):
        var n = min(batch, x.shape[0] - first)
        var view = Tensor.borrowed([n, width], x.fptr() + 4 * first * width)
        var chunk = layer_norm(view, gain, bias, affine, eps)
        for i in range(n * width):
            output.f[first * width + i] = chunk.f[i]
    _ = x.fptr()
    return output^


def _modulate(mut h: Tensor, mod: Tensor, offset: Int):
    for i in range(len(h.f)):
        var j = offset * WIDTH + i % WIDTH
        h.f[i] = _bf(_bf(_bf(h.f[i]) * _bf(1 + mod.f[j + WIDTH])) + mod.f[j])


def _gate(mut x: Tensor, h: Tensor, mod: Tensor, offset: Int):
    for i in range(len(x.f)):
        x.f[i] = _bf(x.f[i] + _bf(h.f[i] * mod.f[offset * WIDTH + i % WIDTH]))


def _linear(
    mut gpu: UltraTexGpu, weights: SafeTensorWeights, x: Tensor, key: String, cols: Int
) raises -> Tensor:
    var h = gpu.linear(weights, x, key, cols)
    var bias = weights.tensor(key + ".bias", [cols])
    for i in range(len(h.f)):
        h.f[i] = _bf(h.f[i] + _bf(bias.f[i % cols]))
    return h^


struct PixalFlow(Movable, StructureVelocityModel):
    var weights: SafeTensorWeights
    var pool: JobPool
    var gpu: UltraTexGpu
    var globals: Tensor
    var projected: Tensor
    var negative: Tensor
    var ones: Tensor
    var angles: Tensor
    var negative_projected: Tensor
    var shape_latent: Tensor

    def __init__(
        out self,
        path: String,
        var globals: Tensor,
        var projected: Tensor,
        coords: Tensor = Tensor(),
        var shape_latent: Tensor = Tensor(),
    ) raises:
        var structure = coords.rank() == 0
        var sites = SITES if structure else (coords.shape[0] if coords.rank() == 2 else 0)
        var channels = 1024 if structure else 2048
        if (
            globals.shape != [5, 1024]
            or projected.shape != [sites, channels]
            or sites < 1
            or not globals.is_float()
            or not projected.is_float()
            or not globals.has_valid_storage()
            or not projected.has_valid_storage()
        ):
            raise Error("Invalid Pixal3D global or projected conditioning")
        if shape_latent.rank() != 0 and (
            structure
            or shape_latent.shape != [sites, 32]
            or not shape_latent.is_float()
            or not shape_latent.has_valid_storage()
        ):
            raise Error("Expected normalized [N,32] shape latent for Pixal3D texture flow")
        self.shape_latent = shape_latent^
        self.negative_projected = Tensor.zeros([1, channels])
        if structure:
            var dense = List[Int](capacity=4 * SITES)
            for r in range(SITES):
                dense.extend([0, r // 256, r // 16 % 16, r % 16])
            self.angles = _rope(Tensor.ints(DT_I32, [SITES, 4], dense^))
        else:
            if coords.dtype != DT_I32 or coords.shape != [sites, 4] or not coords.has_valid_storage():
                raise Error("Expected integer batch/XYZ coordinates for Pixal3D")
            for i in range(sites):
                if (
                    coords.i[i * 4] != 0
                    or min(coords.i[i * 4 + 1], min(coords.i[i * 4 + 2], coords.i[i * 4 + 3])) < 0
                ):
                    raise Error("Invalid sparse coordinates for Pixal3D")
            self.angles = _rope(coords)
        if globals.ext != 0:
            globals = Tensor.floats(globals.shape.copy(), globals.owned_floats())
        if projected.ext != 0:
            projected = Tensor.floats(projected.shape.copy(), projected.owned_floats())
        self.globals = globals^
        self.projected = projected^
        self.weights = SafeTensorWeights(path, 6 << 30, private_mapping=True)
        self.pool = JobPool()
        _ = self.pool.start(default_worker_count())
        self.gpu = UltraTexGpu()
        self.negative = Tensor.zeros([1, 1024])
        self.ones = Tensor.floats([WIDTH], List[Float32](length=WIDTH, fill=1))

    def _block(mut self, mut x: Tensor, timestep: Tensor, block: Int, conditional: Bool) raises:
        var key = "blocks." + String(block) + "."
        var mod = self.weights.tensor(key + "modulation", [6 * WIDTH])
        for i in range(len(mod.f)):
            mod.f[i] = _bf(mod.f[i] + timestep.f[i])
        var h = _norm_rows(x, self.ones, Tensor(), False, 1e-6)
        _modulate(h, mod, 0)
        var att = key + "self_attn."
        var qkv = _linear(self.gpu, self.weights, h, att + "to_qkv", 3 * WIDTH)
        _normalize_heads(qkv, 0, self.weights.tensor(att + "q_rms_norm.gamma", [HEADS, DIM]), self.angles)
        _normalize_heads(qkv, 1, self.weights.tensor(att + "k_rms_norm.gamma", [HEADS, DIM]), self.angles)
        # Keep all 64-key blocks for exact dense attention connectivity.
        h = gpu_sparse_attention(self.gpu.ctx, qkv, HEADS, (x.shape[0] + 63) // 64)
        h = _linear(self.gpu, self.weights, h, att + "to_out", WIDTH)
        _gate(x, h, mod, 2)
        h = _norm_rows(
            x,
            self.weights.tensor(key + "norm2.weight", [WIDTH]),
            self.weights.tensor(key + "norm2.bias", [WIDTH]),
            True,
            1e-6,
        )
        att = key + "cross_attn.cross_attn_block."
        var q = _linear(self.gpu, self.weights, h, att + "to_q", WIDTH)
        var kv = _linear(
            self.gpu, self.weights, self.globals if conditional else self.negative, att + "to_kv", 2 * WIDTH
        )
        _normalize_heads(q, 0, self.weights.tensor(att + "q_rms_norm.gamma", [HEADS, DIM]), Tensor())
        _normalize_heads(kv, 0, self.weights.tensor(att + "k_rms_norm.gamma", [HEADS, DIM]), Tensor())
        h = _linear(self.gpu, self.weights, _global_attention(q, kv), att + "to_out", WIDTH)
        var proj = _linear(
            self.gpu,
            self.weights,
            self.projected if conditional else self.negative_projected,
            key + "cross_attn.proj_linear",
            WIDTH,
        )
        for i in range(len(x.f)):
            x.f[i] = _bf(x.f[i] + _bf(h.f[i] + proj.f[i if conditional else i % WIDTH]))
        h = _norm_rows(x, self.ones, Tensor(), False, 1e-6)
        _modulate(h, mod, 3)
        h = _linear(self.gpu, self.weights, h, key + "mlp.mlp.0", 8192)
        for i in range(len(h.f)):
            var a = h.f[i]
            h.f[i] = _bf(a * 0.5 * (1 + tanh(Float32(0.7978845608) * (a + Float32(0.044715) * a * a * a))))
        h = _linear(self.gpu, self.weights, h, key + "mlp.mlp.2", WIDTH)
        _gate(x, h, mod, 5)

    def _predict_rows(
        mut self, rows: Tensor, model_time: Float32, conditional: Bool, outputs: Int
    ) raises -> Tensor:
        var te = Tensor.zeros([1, 256])
        for j in range(128):
            var theta = model_time * exp(Float32(-9.210340372) * Float32(j) / 128)
            te.f[j] = cos(theta)
            te.f[j + 128] = sin(theta)
        te = weighted_linear(self.weights, self.pool, te, "t_embedder.mlp.0", WIDTH)
        _silu(te)
        te = weighted_linear(self.weights, self.pool, te, "t_embedder.mlp.2", WIDTH)
        _silu(te)
        te = weighted_linear(self.weights, self.pool, te, "adaLN_modulation.1", 6 * WIDTH)
        for i in range(len(te.f)):
            te.f[i] = _bf(te.f[i])
        var x = weighted_linear(self.weights, self.pool, rows, "input_layer", WIDTH)
        for i in range(len(x.f)):
            x.f[i] = _bf(x.f[i])
        for block in range(30):
            self._block(x, te, block, conditional)
        x = _norm_rows(x, self.ones, Tensor(), False, 1e-5)
        return weighted_linear(self.weights, self.pool, x, "out_layer", outputs)

    def predict(mut self, latent: Tensor, model_time: Float32, conditional: Bool) raises -> Tensor:
        if not latent.is_float() or not latent.has_valid_storage():
            raise Error("Expected float32 Pixal3D latent")
        if self.projected.shape[1] == 2048:
            var sites = self.angles.shape[0]
            if latent.shape != [sites, 32]:
                raise Error("Expected [N,32] Pixal3D sparse latent")
            if self.shape_latent.rank() == 0:
                return self._predict_rows(latent, model_time, conditional, 32)
            var combined = Tensor.zeros([sites, 64])
            for r in range(sites):
                for c in range(32):
                    combined.f[r * 64 + c] = latent.fget(r * 32 + c)
                    combined.f[r * 64 + 32 + c] = self.shape_latent.fget(r * 32 + c)
            return self._predict_rows(combined, model_time, conditional, 32)
        if latent.shape != [8, 16, 16, 16]:
            raise Error("Expected an 8-channel structure latent on a 16-cubed grid")
        var rows = Tensor.zeros([SITES, 8])
        for r in range(SITES):
            for ch in range(8):
                rows.f[r * 8 + ch] = latent.fget(ch * SITES + r)
        rows = self._predict_rows(rows, model_time, conditional, 8)
        var velocity = Tensor.zeros(latent.shape.copy())
        for ch in range(8):
            for r in range(SITES):
                velocity.f[ch * SITES + r] = rows.f[r * 8 + ch]
        return velocity^
