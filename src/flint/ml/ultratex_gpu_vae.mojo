"""FP32 VAE convolutions and dense bottleneck attention on NVIDIA tensor cores."""
from std.ffi import external_call
from std.math import exp, sqrt
from std.memory import bitcast
from std.utils import StaticTuple
from max.gpu import block_idx, block_dim, global_idx, thread_idx
from max.gpu.sync import barrier
from std.memory import unsafe_stack_allocation
from max.gpu.compute.mma import mma
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.warp import shuffle_xor
from flint.ml.tensor import Tensor
from flint.ml.safetensors import SafeTensorWeights


@always_inline
def _tf32(x: Float32) -> Float32:
    var u = bitcast[DType.uint32](x)
    return bitcast[DType.float32]((u + 4095 + ((u >> 13) & 1)) & 0xFFFFE000)


def _multiply[
    convolution: Bool, transpose_b: Bool, reflect: Bool = False
](
    a: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    c: Pointer[Float32, MutAnyOrigin],
    bias: Pointer[Float32, MutAnyOrigin],
    m: Int32,
    n: Int32,
    channels: Int32,
    height: Int32,
    width: Int32,
    kernel: Int32,
    stride: Int32,
    start: Int32,
):
    var th = Int(global_idx.x)
    var lane = th % 32
    var group = lane // 4
    var sub = lane % 4
    var cols = (Int(n) + 31) // 32
    var row = 16 * (th // 32 // cols)
    var col = 32 * (th // 32 % cols)
    if row >= Int(m):
        return
    var sums = StaticTuple[SIMD[DType.float32, 4], 4](fill=0)
    var padded = (Int(channels) + 7) // 8 * 8 if convolution else Int(channels)
    var corrections = StaticTuple[SIMD[DType.float32, 4], 4](fill=0)
    for tap in range(Int(kernel) * Int(kernel)):
        var offsets = SIMD[DType.int64, 2](-1)
        comptime for h in range(2):
            var r = row + group + h * 8
            if r < Int(m):
                comptime if convolution:
                    var pad = Int(kernel) // 2 if stride == 1 else 0
                    var out_w = Int(width) // Int(stride)
                    var y = r // out_w * Int(stride) + tap // Int(kernel) - pad
                    var x = r % out_w * Int(stride) + tap % Int(kernel) - pad
                    comptime if reflect:
                        y = min(abs(y), 2 * Int(height) - 2 - y)
                        x = min(abs(x), 2 * Int(width) - 2 - x)
                    if y >= 0 and x >= 0 and y < Int(height) and x < Int(width):
                        offsets[h] = Int64((y * Int(width) + x) * Int(channels))
                else:
                    offsets[h] = Int64((Int(start) + r) * Int(channels))
        for first in range(0, padded, 8):
            var af = SIMD[DType.float32, 4](0)
            var residual_a = SIMD[DType.float32, 4](0)
            comptime for j in range(4):
                var t = first + sub + 4 * (j // 2)
                if offsets[j % 2] >= 0 and t < Int(channels):
                    var value = a.unsafe_load(Int(offsets[j % 2]) + t)
                    af[j] = _tf32(value)
                    residual_a[j] = _tf32(value - af[j])
            comptime for tile in range(4):
                var ccol = col + 8 * tile + group
                var bf = SIMD[DType.float32, 2](0)
                var residual_b = SIMD[DType.float32, 2](0)
                if ccol < Int(n):
                    comptime for j in range(2):
                        var t = first + sub + 4 * j
                        if t < Int(channels):
                            var value: Float32
                            comptime if transpose_b:
                                value = b.unsafe_load(
                                    (ccol * Int(kernel) * Int(kernel) + tap)
                                    * padded
                                    + t
                                )
                            else:
                                value = b.unsafe_load(t * Int(n) + ccol)
                            bf[j] = _tf32(value)
                            residual_b[j] = _tf32(value - bf[j])
                var prev = sums[tile]
                mma(sums[tile], af, bf, prev)
                var corr = corrections[tile]
                mma(corrections[tile], residual_a, bf, corr)
                corr = corrections[tile]
                mma(corrections[tile], af, residual_b, corr)
    comptime for tile in range(4):
        comptime for j in range(4):
            var r = row + group + 8 * (j // 2)
            var column = col + 8 * tile + 2 * sub + j % 2
            if r < Int(m) and column < Int(n):
                var v = sums[tile][j] + corrections[tile][j]
                comptime if convolution:
                    v += bias.unsafe_load(column)
                c.unsafe_store(r * Int(n) + column, v)


def _softmax(
    scores: Pointer[Float32, MutAnyOrigin], rows: Int32, columns: Int32
):
    var i = Int(global_idx.x)
    var row = i // 32
    var lane = i % 32
    if row >= Int(rows):
        return
    var base = row * Int(columns)
    var peak = -Float32.MAX
    for j in range(lane, Int(columns), 32):
        peak = max(peak, scores.unsafe_load(base + j))
    comptime for b in range(5):
        peak = max(peak, shuffle_xor(peak, UInt32(1 << b)))
    var total = Float32(0)
    for j in range(lane, Int(columns), 32):
        var p = exp(
            (scores.unsafe_load(base + j) - peak) * Float32(0.04419417382415922)
        )
        scores.unsafe_store(base + j, p)
        total += p
    comptime for b in range(5):
        total += shuffle_xor(total, UInt32(1 << b))
    for j in range(lane, Int(columns), 32):
        scores.unsafe_store(base + j, scores.unsafe_load(base + j) / total)


def _group_partials(
    x: Pointer[Float32, MutAnyOrigin],
    partials: Pointer[Float64, MutAnyOrigin],
    pixels: Int32,
    channels: Int32,
    pieces: Int32,
):
    var tid = Int(thread_idx.x)
    var piece = Int(block_idx.x)
    var group = Int(block_idx.y)
    var span = Int(channels) // 32
    var total = Float64(0)
    var square = Float64(0)
    for j in range(4):
        var i = piece * 512 + tid + 128 * j
        if i < Int(pixels) * span:
            var v = Float64(
                x.unsafe_load(
                    (i // span) * Int(channels) + group * span + i % span
                )
            )
            total += v
            square += v * v
    comptime for b in range(5):
        total += shuffle_xor(total, UInt32(1 << b))
        square += shuffle_xor(square, UInt32(1 << b))
    var sums = unsafe_stack_allocation[8, Float64, address_space=.SHARED]()
    if tid % 32 == 0:
        sums.unsafe_store(tid // 32, total)
        sums.unsafe_store(4 + tid // 32, square)
    barrier()
    if tid == 0:
        var off = (group * Int(pieces) + piece) * 2
        partials.unsafe_store(
            off,
            sums.unsafe_load(0)
            + sums.unsafe_load(1)
            + sums.unsafe_load(2)
            + sums.unsafe_load(3),
        )
        partials.unsafe_store(
            off + 1,
            sums.unsafe_load(4)
            + sums.unsafe_load(5)
            + sums.unsafe_load(6)
            + sums.unsafe_load(7),
        )


def _group_scales(
    partials: Pointer[Float64, MutAnyOrigin],
    stats: Pointer[Float32, MutAnyOrigin],
    pieces: Int32,
    elements: Int64,
):
    var tid = Int(thread_idx.x)
    var group = Int(block_idx.x)
    var total = Float64(0)
    var square = Float64(0)
    for j in range(tid, Int(pieces), 32):
        var i = (group * Int(pieces) + j) * 2
        total += partials.unsafe_load(i)
        square += partials.unsafe_load(i + 1)
    comptime for b in range(5):
        total += shuffle_xor(total, UInt32(1 << b))
        square += shuffle_xor(square, UInt32(1 << b))
    if tid == 0:
        var mean = total / Float64(elements)
        var variance = max(Float64(0), square / Float64(elements) - mean * mean)
        stats.unsafe_store(group, Float32(mean))
        stats.unsafe_store(group + 32, Float32(1 / sqrt(variance + 1e-6)))


def _normalize_image(
    x: Pointer[Float32, MutAnyOrigin],
    stats: Pointer[Float32, MutAnyOrigin],
    gain: Pointer[Float32, MutAnyOrigin],
    bias: Pointer[Float32, MutAnyOrigin],
    count: Int64,
    channels: Int32,
    activate: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        var c = i % Int(channels)
        var g = c // (Int(channels) // 32)
        var v = (x.unsafe_load(i) - stats.unsafe_load(g)) * stats.unsafe_load(
            g + 32
        ) * gain.unsafe_load(c) + bias.unsafe_load(c)
        x.unsafe_store(i, v / (1 + exp(-v)) if activate != 0 else v)


struct UltraTexVaeGpu(Movable):
    var ctx: DeviceContext
    var names: Dict[String, Int]
    var kernels: List[DeviceBuffer[DType.float32]]
    var biases: List[DeviceBuffer[DType.float32]]

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.names = Dict[String, Int]()
        self.kernels = List[DeviceBuffer[DType.float32]]()
        self.biases = List[DeviceBuffer[DType.float32]]()

    def _upload(self, x: Tensor) raises -> DeviceBuffer[DType.float32]:
        var buf = self.ctx.enqueue_create_buffer[DType.float32](len(x.f))
        with buf.map_to_host() as host:
            _ = external_call["memcpy", Int](
                Int(host.unsafe_ptr()), x.fptr(), len(x.f) * 4
            )
        return buf^

    def _download(self, buf: DeviceBuffer[DType.float32], mut x: Tensor) raises:
        with buf.map_to_host() as host:
            _ = external_call["memcpy", Int](
                x.fptr(), Int(host.unsafe_ptr()), len(x.f) * 4
            )

    def conv[reflect: Bool = False](
        mut self,
        w: SafeTensorWeights,
        x: Tensor,
        key: String,
        co: Int,
        k: Int,
        stride: Int,
    ) raises -> Tensor:
        var ci = x.shape[2]
        var id = self.names.get(key, -1)
        if id < 0:
            var orig = w.tensor(key + ".weight", [co, ci, k, k])
            var padded = (ci + 7) // 8 * 8
            var packed = Tensor.zeros([co, k * k, padded])
            for c in range(co):
                for tap in range(k * k):
                    for j in range(ci):
                        packed.f[(c * k * k + tap) * padded + j] = orig.f[
                            (c * ci + j) * k * k + tap
                        ]
            id = len(self.kernels)
            self.kernels.append(self._upload(packed))
            self.biases.append(self._upload(w.tensor(key + ".bias", [co])))
            self.names[key] = id
        var h = x.shape[0]
        var width = x.shape[1]
        var rows = (h // stride) * (width // stride)
        var src = self._upload(x)
        var dst = self.ctx.enqueue_create_buffer[DType.float32](rows * co)
        var tiles = ((rows + 15) // 16) * ((co + 31) // 32)
        self.ctx.enqueue_function[_multiply[True, True, reflect]](
            src.unsafe_ptr(),
            self.kernels[id].unsafe_ptr(),
            dst.unsafe_ptr(),
            self.biases[id].unsafe_ptr(),
            Int32(rows),
            Int32(co),
            Int32(ci),
            Int32(h),
            Int32(width),
            Int32(k),
            Int32(stride),
            Int32(0),
            grid_dim=(tiles + 3) // 4,
            block_dim=128,
        )
        var result = Tensor.zeros([h // stride, width // stride, co])
        self._download(dst, result)
        return result^

    def group(
        self, w: SafeTensorWeights, var x: Tensor, key: String, activate: Bool
    ) raises -> Tensor:
        var channels = x.shape[2]
        var pixels = x.shape[0] * x.shape[1]
        var elements = len(x.f) // 32
        var pieces = (elements + 511) // 512
        var src = self._upload(x)
        var gain = self._upload(w.tensor(key + ".weight", [channels]))
        var bias = self._upload(w.tensor(key + ".bias", [channels]))
        var partials = self.ctx.enqueue_create_buffer[DType.float64](
            2 * 32 * pieces
        )
        var stats = self.ctx.enqueue_create_buffer[DType.float32](64)
        self.ctx.enqueue_function[_group_partials](
            src.unsafe_ptr(),
            partials.unsafe_ptr(),
            Int32(pixels),
            Int32(channels),
            Int32(pieces),
            grid_dim=(pieces, 32),
            block_dim=128,
        )
        self.ctx.enqueue_function[_group_scales](
            partials.unsafe_ptr(),
            stats.unsafe_ptr(),
            Int32(pieces),
            Int64(elements),
            grid_dim=32,
            block_dim=32,
        )
        self.ctx.enqueue_function[_normalize_image](
            src.unsafe_ptr(),
            stats.unsafe_ptr(),
            gain.unsafe_ptr(),
            bias.unsafe_ptr(),
            Int64(len(x.f)),
            Int32(channels),
            Int32(activate),
            grid_dim=(len(x.f) + 255) // 256,
            block_dim=256,
        )
        self._download(src, x)
        return x^

    def attention(self, q: Tensor, k: Tensor, v: Tensor) raises -> Tensor:
        var n = len(q.f) // 512
        var dq = self._upload(q)
        var dk = self._upload(k)
        var dv = self._upload(v)
        var tile = min(256, n)
        var scores = self.ctx.enqueue_create_buffer[DType.float32](tile * n)
        var dst = self.ctx.enqueue_create_buffer[DType.float32](n * 512)
        var unused_bias = self.ctx.enqueue_create_buffer[DType.float32](1)
        for first in range(0, n, tile):
            var m = min(tile, n - first)
            var tiles = ((m + 15) // 16) * ((n + 31) // 32)
            self.ctx.enqueue_function[_multiply[False, True]](
                dq.unsafe_ptr(),
                dk.unsafe_ptr(),
                scores.unsafe_ptr(),
                unused_bias.unsafe_ptr(),
                Int32(m),
                Int32(n),
                Int32(512),
                Int32(0),
                Int32(0),
                Int32(1),
                Int32(1),
                Int32(first),
                grid_dim=(tiles + 3) // 4,
                block_dim=128,
            )
            self.ctx.enqueue_function[_softmax](
                scores.unsafe_ptr(),
                Int32(m),
                Int32(n),
                grid_dim=(m + 3) // 4,
                block_dim=128,
            )
            self.ctx.enqueue_function[_multiply[False, False]](
                scores.unsafe_ptr(),
                dv.unsafe_ptr(),
                dst.unsafe_ptr().unsafe_offset(first * 512),
                unused_bias.unsafe_ptr(),
                Int32(m),
                Int32(512),
                Int32(n),
                Int32(0),
                Int32(0),
                Int32(1),
                Int32(1),
                Int32(0),
                grid_dim=(((m + 15) // 16) * 16 + 3) // 4,
                block_dim=128,
            )
        var result = Tensor.zeros(q.shape.copy())
        self._download(dst, result)
        return result^
