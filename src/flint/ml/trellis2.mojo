"""Single-sample sparse structure decoder for Microsoft's TRELLIS.2-4B.

The checkpoint uses NCDHW layout, channel LayerNorm and 3D pixel shuffle.
The batch dimension (always one) is omitted from Flint tensors.
"""

from std.memory import bitcast
from std.math import exp, sqrt
from flint.ml.safetensors import SafeTensorWeights
from flint.core.jobs import JobPool
from flint.ml.ops import matmul_2d
from flint.ml.tensor import DT_I32, Tensor
from flint.platform.os import file_size, read_bytes


struct TrellisStructureDecoder(Movable):
    var weights: SafeTensorWeights

    def __init__(out self, path: String) raises:
        self.weights = SafeTensorWeights(path, 256 << 20)

    def _conv(self, mut pool: JobPool, x: Tensor, name: String, channels: Int, half: Bool) raises -> Tensor:
        var c = x.shape[0]
        var edge = x.shape[1]
        var plane = edge * edge
        var sites = plane * edge
        var weights = self.weights.tensor(name + ".weight", [channels, c, 3, 3, 3])
        var bias = self.weights.tensor(name + ".bias", [channels])
        var y = Tensor.zeros([channels, edge, edge, edge])
        # Bound im2col memory independently of the number of voxels.
        var k = c * 27
        var tile = min(sites, 256)
        var patches = List[Float32](length=tile * k, fill=0)
        var product = List[Float32](length=tile * channels, fill=0)
        for first in range(0, sites, tile):
            var n = min(tile, sites - first)
            for tap in range(27):
                var dz = tap // 9 - 1
                var dy = (tap // 3) % 3 - 1
                var dx = tap % 3 - 1
                for j in range(n):
                    var p = first + j
                    var z = p // plane + dz
                    var h = (p // edge) % edge + dy
                    var w = p % edge + dx
                    var inside = (z >= 0 and z < edge and h >= 0 and h < edge and w >= 0 and w < edge)
                    var source = (z * edge + h) * edge + w
                    for ch in range(c):
                        patches[(ch * 27 + tap) * n + j] = x.fget(ch * sites + source) if inside else Float32(0)
            matmul_2d(pool, weights.fptr(), Int(patches.unsafe_ptr()), Int(product.unsafe_ptr()),
                      0, channels, k, n, False)
            for ch in range(channels):
                for j in range(n):
                    var v = product[ch * n + j] + bias.f[ch]
                    y.f[ch * sites + first + j] = Float32(Float16(v)) if half else v
        return y^

    def _norm_silu(self, x: Tensor, name: String, half: Bool) raises -> Tensor:
        var c = x.shape[0]
        var sites = x.numel() // c
        var scale = self.weights.tensor(name + ".weight", [c])
        var bias = self.weights.tensor(name + ".bias", [c])
        var y = Tensor.zeros(x.shape.copy())
        for p in range(sites):
            var mean = Float32(0)
            for ch in range(c):
                mean += x.f[ch * sites + p]
            mean /= Float32(c)
            var variance = Float32(0)
            for ch in range(c):
                var centered = x.f[ch * sites + p] - mean
                variance += centered * centered
            var inv = 1 / sqrt(variance / Float32(c) + Float32(1e-5))
            for ch in range(c):
                var idx = ch * sites + p
                var v = (x.f[idx] - mean) * inv * scale.f[ch] + bias.f[ch]
                if half:
                    v = Float32(Float16(v))
                v = v / (1 + exp(-v))
                y.f[idx] = Float32(Float16(v)) if half else v
        return y^

    def _residual(self, mut pool: JobPool, x: Tensor, name: String) raises -> Tensor:
        var h = self._conv(pool, self._norm_silu(x, name + ".norm1", True), name + ".conv1", x.shape[0], True)
        var y = self._conv(pool, self._norm_silu(h, name + ".norm2", True), name + ".conv2", x.shape[0], True)
        for i in range(len(y.f)):
            y.f[i] = Float32(Float16(y.f[i] + x.f[i]))
        return y^

    def decode(self, mut pool: JobPool, latent: Tensor) raises -> Tensor:
        """Decode [8,D,D,D] to [1,4D,4D,4D] occupancy logits, D=16 for the released flow model."""
        if (not latent.is_float() or latent.rank() != 4 or latent.shape[0] != 8 or
                latent.shape[1] < 1 or latent.shape[1] > 16 or latent.shape[1] != latent.shape[2] or
                latent.shape[1] != latent.shape[3] or not latent.has_valid_storage()):
            raise Error("Expected [8,D,D,D] float32 structure latent with 1 <= D <= 16")
        var x = self._conv(pool, latent, "input_layer", 512, True)
        for i in range(2):
            x = self._residual(pool, x, "middle_block." + String(i))
        for i in range(8):
            var name = "blocks." + String(i)
            if i == 2 or i == 5:
                var c = 128 if i == 2 else 32
                x = _shuffle_2x(self._conv(pool, x, name + ".conv", c * 8, True))
            else:
                x = self._residual(pool, x, name)
        return self._conv(pool, self._norm_silu(x, "out_layer.0", False), "out_layer.2", 1, False)


def _shuffle_2x(x: Tensor) -> Tensor:
    var c = x.shape[0] // 8
    var edge = x.shape[1]
    var out_edge = edge * 2
    var y = Tensor.zeros([c, out_edge, out_edge, out_edge])
    for ch in range(c):
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    var src = ((ch * 2 + dz) * 2 + dy) * 2 + dx
                    for z in range(edge):
                        for h in range(edge):
                            for w in range(edge):
                                var dst = ((ch * out_edge + z * 2 + dz) * out_edge + h * 2 + dy) * out_edge + w * 2 + dx
                                y.f[dst] = x.f[((src * edge + z) * edge + h) * edge + w]
    return y^


def structure_coordinates(logits: Tensor, resolution: Int) raises -> Tensor:
    """Max-pool occupancy and emit [batch,z,y,x] int32 rows in upstream argwhere order."""
    if not logits.is_float() or logits.rank() != 4 or logits.shape[0] != 1 or not logits.has_valid_storage():
        raise Error("Expected a single occupancy logit grid")
    var edge = logits.shape[1]
    if (edge <= 0 or resolution <= 0 or resolution > edge or edge % resolution != 0
            or edge != logits.shape[2] or edge != logits.shape[3]):
        raise Error("Output resolution must divide the cubic occupancy grid")
    var stride = edge // resolution
    var active = List[Bool](length=resolution ** 3, fill=False)
    for z in range(edge):
        for y in range(edge):
            for x in range(edge):
                if logits.fget((z * edge + y) * edge + x) > 0:
                    active[((z // stride) * resolution + y // stride) * resolution + x // stride] = True
    var rows = List[Int]()
    for p in range(len(active)):
        if active[p]:
            rows.append(0)
            rows.append(p // (resolution * resolution))
            rows.append((p // resolution) % resolution)
            rows.append(p % resolution)
    var count = len(rows) // 4
    return Tensor.ints(DT_I32, [count, 4], rows^)


def read_structure_latent(input_path: String, edge: Int) raises -> Tensor:
    """Read one little-endian float32 [8,D,D,D] latent in channel-major order."""
    if edge < 1 or edge > 16:
        raise Error("Latent edge must be between 1 and 16")
    var count = 8 * edge ** 3
    if file_size(input_path) != count * 4:
        raise Error("Expected 8 * EDGE^3 little-endian float32 latent values")
    var raw = read_bytes(input_path)
    if len(raw) != count * 4:
        raise Error("Could not read complete latent")
    var latent = Tensor.zeros([8, edge, edge, edge])
    for i in range(count):
        var bits = UInt32(0)
        for b in range(4):
            bits |= UInt32(raw[4 * i + b]) << UInt32(8 * b)
        latent.f[i] = bitcast[DType.float32](bits)
    return latent^
