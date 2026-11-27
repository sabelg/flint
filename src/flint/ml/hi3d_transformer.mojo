"""Hi3D 512 checkpoint-backed sparse flow transformer. Mojo port, see licenses/hi3d."""
from std.math import cos, exp, sin, sqrt, tanh
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.hi3d_flow import Hi3DVelocityModel
from flint.ml.hi3d_sparse import SparseGrid, SparseParents, sparse_conv3
from flint.ml.ops import layer_norm, matmul_2d
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import Tensor, DT_F32
from flint.ml.vision_ops import weighted_linear

comptime W = 2112
comptime D = 96


def _half(mut x: Tensor):
    for i in range(len(x.f)):
        x.f[i] = Float32(Float16(x.f[i]))


def _silu(mut x: Tensor):
    for i in range(len(x.f)):
        var v = x.f[i]
        x.f[i] = v / (1 + exp(-v))


def _norm(w: SafeTensorWeights, x: Tensor, key: String, eps: Float32) raises -> Tensor:
    # Hi3D's 9k+ sparse sites can exceed the ONNX interpreter's single-op
    # limit. Layer normalization is independent per row, so process tiles.
    var c = x.shape[1]
    var affine = key.byte_length() > 0
    var gain = w.tensor(key + ".weight", [c]) if affine else Tensor.floats(
        [c], List[Float32](length=c, fill=1)
    )
    var bias = w.tensor(key + ".bias", [c]) if affine else Tensor()
    if len(x.f) <= 16777216:
        return layer_norm(x, gain, bias, affine, eps)
    var out = Tensor.zeros([x.shape[0], c])
    var chunk = max(1, 16777216 // c)
    for first in range(0, x.shape[0], chunk):
        var size = min(chunk, x.shape[0] - first)
        var tile = Tensor.zeros([size, c])
        for j in range(size * c):
            tile.f[j] = x.f[first * c + j]
        tile = layer_norm(tile, gain, bias, affine, eps)
        for j in range(size * c):
            out.f[first * c + j] = tile.f[j]
    return out^


def _join(a: Tensor, b: Tensor) -> Tensor:
    var n = a.shape[0]
    var ca = a.shape[1]
    var cb = b.shape[1]
    var out = Tensor.zeros([n, ca + cb])
    for i in range(n):
        for j in range(ca):
            out.f[i * (ca + cb) + j] = a.f[i * ca + j]
        for j in range(cb):
            out.f[i * (ca + cb) + ca + j] = b.f[i * cb + j]
    return out^


def _torso_linear(
    w: SafeTensorWeights, mut pool: JobPool, x: Tensor, key: String, cols: Int
) raises -> Tensor:
    var out = weighted_linear(w, pool, x, key, cols)
    _half(out)
    return out^


def _resblock(
    w: SafeTensorWeights, mut pool: JobPool, g: SparseGrid, x: Tensor, emb: Tensor, key: String, co: Int
) raises -> Tensor:
    var ci = x.shape[1]
    var mid = co // 2
    var modulation = _torso_linear(w, pool, emb, key + "emb_layers.1", 2 * mid)
    var h = _norm(w, x, key + "norm1", 1e-6)
    _half(h)
    _silu(h)
    _half(h)
    h = sparse_conv3(
        pool,
        g,
        h,
        w.tensor(key + "conv1.conv.weight", [mid, 3, 3, 3, ci]),
        w.tensor(key + "conv1.conv.bias", [mid]),
    )
    _half(h)
    h = _norm(w, h, "", 1e-6)
    for i in range(len(h.f)):
        var j = i % mid
        var scale = Float32(Float16(1 + modulation.f[j]))
        h.f[i] = Float32(Float16(Float32(Float16(Float32(Float16(h.f[i])) * scale)) + modulation.f[mid + j]))
    _silu(h)
    _half(h)
    h = sparse_conv3(
        pool,
        g,
        h,
        w.tensor(key + "conv2.conv.weight", [mid, 3, 3, 3, mid]),
        w.tensor(key + "conv2.conv.bias", [mid]),
    )
    _half(h)
    _silu(h)
    _half(h)
    h = _torso_linear(w, pool, h, key + "linear", co)
    var skip = _torso_linear(w, pool, x, key + "skip_connection", co) if ci != co else x.copy()
    for i in range(len(h.f)):
        h.f[i] = Float32(Float16(h.f[i] + skip.f[i]))
    return h^


def _prepare_head(mut q: Tensor, grid: SparseGrid, gain: Tensor, head: Int):
    # Three consecutive 32-channel rotary groups, with adjacent real/imag pairs.
    for row in range(q.shape[0]):
        var norm = Float32(0)
        for j in range(0, D, 2):
            var a = row * D + j
            var angle = Float32(grid.xyz.i[row * 3 + j // 32]) * exp(
                Float32(-9.210340372) * (Float32((j % 32) // 2) / 16)
            )
            var x = q.f[a]
            var y = q.f[a + 1]
            var u = Float32(Float16(x * cos(angle) - y * sin(angle)))
            var v = Float32(Float16(x * sin(angle) + y * cos(angle)))
            q.f[a] = u
            q.f[a + 1] = v
            norm += u * u + v * v
        var factor = sqrt(Float32(D)) / max(Float32(1e-12), sqrt(norm))
        for j in range(D):
            q.f[row * D + j] = Float32(Float16(q.f[row * D + j] * factor * gain.f[head * D + j]))


def _attend(
    w: SafeTensorWeights,
    mut pool: JobPool,
    g: SparseGrid,
    x: Tensor,
    context: Tensor,
    key: String,
    cross: Bool,
) raises -> Tensor:
    var qpack = _torso_linear(w, pool, x, key + ("to_q" if cross else "to_qkv"), W if cross else 3 * W)
    var rows = x.shape[0]
    var cols = context.shape[0] if cross else rows
    var q = Tensor.zeros([rows, D])
    var k = Tensor.zeros([cols, D])
    var v = Tensor.zeros([cols, D])
    var tile = min(rows, 64)
    var p = Tensor.zeros([tile, cols])
    var h = Tensor.zeros([tile, D])
    var out = Tensor.zeros([rows, W])
    var qgain = Tensor() if cross else w.tensor(key + "q_rms_norm.gamma", [22, D])
    var kgain = Tensor() if cross else w.tensor(key + "k_rms_norm.gamma", [22, D])
    for head in range(22):
        for i in range(rows):
            for j in range(D):
                q.f[i * D + j] = qpack.f[i * qpack.shape[1] + head * D + j]
        for i in range(cols):
            for j in range(D):
                if cross:
                    k.f[i * D + j] = context.f[i * 2 * W + head * D + j]
                    v.f[i * D + j] = context.f[i * 2 * W + W + head * D + j]
                else:
                    k.f[i * D + j] = qpack.f[i * 3 * W + W + head * D + j]
                    v.f[i * D + j] = qpack.f[i * 3 * W + 2 * W + head * D + j]
        if not cross:
            _prepare_head(q, g, qgain, head)
            _prepare_head(k, g, kgain, head)
        for first in range(0, rows, tile):
            var n = min(tile, rows - first)
            matmul_2d(pool, q.fptr() + first * D * 4, k.fptr(), p.fptr(), 0, n, D, cols, True)
            for r in range(n):
                var begin = r * cols
                var peak = p.f[begin]
                for j in range(1, cols):
                    peak = max(peak, p.f[begin + j])
                var total = Float32(0)
                for j in range(cols):
                    var prob = exp((p.f[begin + j] - peak) / sqrt(Float32(D)))
                    p.f[begin + j] = prob
                    total += prob
                for j in range(cols):
                    p.f[begin + j] /= total
            matmul_2d(pool, p.fptr(), v.fptr(), h.fptr(), 0, n, cols, D, False)
            for r in range(n):
                for j in range(D):
                    out.f[(first + r) * W + head * D + j] = Float32(Float16(h.f[r * D + j]))
    return _torso_linear(w, pool, out, key + "to_out", W)


def _affine(mut x: Tensor, m: Tensor, index: Int):
    for i in range(len(x.f)):
        var j = i % W
        var scale = Float32(Float16(1 + m.f[(index + 1) * W + j]))
        var v = Float32(Float16(Float32(Float16(x.f[i])) * scale))
        x.f[i] = Float32(Float16(v + m.f[index * W + j]))


def _add(mut x: Tensor, y: Tensor, m: Tensor, index: Int):
    for i in range(len(x.f)):
        var v = y.f[i]
        if index >= 0:
            v = Float32(Float16(v * m.f[index * W + i % W]))
        x.f[i] = Float32(Float16(x.f[i] + v))


def _block(
    w: SafeTensorWeights,
    mut pool: JobPool,
    g: SparseGrid,
    mut x: Tensor,
    mod: Tensor,
    ctx: Tensor,
    key: String,
) raises:
    var m = w.tensor(key + "modulation", [1, 6 * W])
    for i in range(len(m.f)):
        m.f[i] = Float32(Float16(m.f[i] + mod.f[i]))
    var h = _norm(w, x, "", 1e-6)
    _affine(h, m, 0)
    h = _attend(w, pool, g, h, ctx, key + "self_attn.", False)
    _add(x, h, m, 2)
    h = _norm(w, x, key + "norm2", 1e-6)
    _half(h)
    h = _attend(w, pool, g, h, ctx, key + "cross_attn.", True)
    _add(x, h, m, -1)
    h = _norm(w, x, "", 1e-6)
    _affine(h, m, 3)
    h = _torso_linear(w, pool, h, key + "mlp.mlp.0", 4 * W)
    for i in range(len(h.f)):
        var a = h.f[i]
        h.f[i] = Float32(
            Float16(a * 0.5 * (1 + tanh(Float32(0.7978845608) * (a + Float32(0.044715) * a * a * a))))
        )
    h = _torso_linear(w, pool, h, key + "mlp.mlp.2", W)
    _add(x, h, m, 5)


struct Hi3DTransformer(Hi3DVelocityModel, Movable):
    var w: SafeTensorWeights
    var fine: SparseGrid
    var coarse: SparseParents
    var ctx: Tensor
    var zero: Tensor
    var positive_kv: List[Tensor]
    var negative_kv: List[Tensor]
    var pool: JobPool

    def __init__(out self, path: String, var coords: Tensor, var condition: Tensor) raises:
        if (
            len(condition.shape) != 3
            or condition.shape[0] != 1
            or condition.shape[1] < 1
            or condition.shape[2] != 2048
            or condition.dtype != DT_F32
            or len(condition.f) != condition.numel()
        ):
            raise Error("Expected [1,N,2048] Hi3D condition")
        self.w = SafeTensorWeights(path, 4 << 30)
        self.fine = SparseGrid(coords^, 64)
        self.coarse = SparseParents(self.fine)
        condition.shape = [condition.shape[1], 2048]
        _half(condition)
        self.ctx = condition^
        # Zero-condition tokens are identical, so their attention reduces to one token.
        self.zero = Tensor.zeros([1, 2048])
        self.positive_kv = List[Tensor]()
        self.negative_kv = List[Tensor]()
        self.pool = JobPool()
        _ = self.pool.start(default_worker_count())

    def velocity(mut self, x: Tensor, time: Float32, conditional: Bool) raises -> Tensor:
        if (
            len(x.shape) != 2
            or x.shape[0] != self.fine.xyz.shape[0]
            or x.shape[1] != 32
            or x.dtype != DT_F32
            or len(x.f) != x.numel()
        ):
            raise Error("Expected [N,32] Hi3D latents")
        var te = Tensor.zeros([1, 256])
        for j in range(128):
            var phase = time * exp(Float32(-9.210340372) * Float32(j) / 128)
            te.f[j] = cos(phase)
            te.f[128 + j] = sin(phase)
        te = weighted_linear(self.w, self.pool, te, "t_embedder.mlp.0", W)
        _silu(te)
        te = weighted_linear(self.w, self.pool, te, "t_embedder.mlp.2", W)
        var emb = te.copy()
        _half(emb)
        _silu(emb)
        _half(emb)
        _silu(te)
        var modulation = weighted_linear(self.w, self.pool, te, "adaLN_modulation.1", 6 * W)
        _half(modulation)
        var h = _torso_linear(self.w, self.pool, x, "input_layer", 256)
        var skip = _resblock(self.w, self.pool, self.fine, h, emb, "input_blocks.0.", 256)
        h = self.coarse.down(skip)
        var root = _resblock(self.w, self.pool, self.coarse.grid, h, emb, "input_blocks.1.", W)
        h = root.copy()
        # Keys and values depend only on the fixed image (or zero) condition.
        # Keep them for the second Heun slope and for subsequent timesteps.
        for i in range(24):
            var key = "blocks." + String(i) + "."
            if conditional:
                if len(self.positive_kv) == i:
                    self.positive_kv.append(
                        _torso_linear(self.w, self.pool, self.ctx, key + "cross_attn.to_kv", 2 * W)
                    )
                _block(self.w, self.pool, self.coarse.grid, h, modulation, self.positive_kv[i], key)
            else:
                if len(self.negative_kv) == i:
                    self.negative_kv.append(
                        _torso_linear(self.w, self.pool, self.zero, key + "cross_attn.to_kv", 2 * W)
                    )
                _block(self.w, self.pool, self.coarse.grid, h, modulation, self.negative_kv[i], key)
        h = self.coarse.up(_join(h, root))
        h = _resblock(self.w, self.pool, self.fine, h, emb, "out_blocks.0.", 256)
        h = _join(h, skip)
        h = _resblock(self.w, self.pool, self.fine, h, emb, "out_blocks.1.", 256)
        h = _norm(self.w, h, "", 1e-5)
        _half(h)
        return weighted_linear(self.w, self.pool, h, "out_layer", 32)
