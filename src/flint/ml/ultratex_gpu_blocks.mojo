"""BF16 device activations for the released Klein transformer blocks."""
from max.gpu import block_dim, block_idx, global_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.warp import shuffle_down, shuffle_xor
from max.gpu.sync import barrier
from std.memory import unsafe_stack_allocation
from std.sys._assembly import inlined_assembly
from std.ffi import external_call
from std.math import exp, fma, sqrt
from flint.ml.tensor import Tensor
from flint.ml.ultratex_gpu import UltraTexGpu, _project_tiled


@always_inline
def _sum(var value: Float32) -> Float32:
    comptime for step in range(5):
        value += shuffle_xor(value, UInt32(1 << step))
    return value


@always_inline
def _combine_moments(
    a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]
) -> SIMD[DType.float32, 4]:
    var count = a[2] + b[2]
    if count == 0:
        return 0
    var inv = 1 / count
    var wa = a[2] * inv
    var wb = b[2] * inv
    var delta = a[0] - b[0]
    var mean = fma(wb, b[0], wa * a[0])
    var m2 = fma(delta * delta * b[2], wa, b[1] + a[1])
    return SIMD[DType.float32, 4](mean, m2, count, 0)


def _norm(
    x: Pointer[BFloat16, MutAnyOrigin],
    mods: Pointer[BFloat16, MutAnyOrigin],
    dest: Pointer[BFloat16, MutAnyOrigin],
    rows: Int32,
    width: Int32,
    shift: Int32,
):
    # The CUDA layer norm processes consecutive groups of four channels with
    # 128 threads, combining Welford statistics within and then across warps.
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = row * Int(width)
    var moments = SIMD[DType.float32, 4](0)
    for first in range(tid * 4, Int(width), 512):
        comptime for j in range(4):
            var v = Float32(x.unsafe_load(base + first + j))
            var d = v - moments[0]
            moments[2] += 1
            moments[0] = fma(d, 1 / moments[2], moments[0])
            moments[1] = fma(d, v - moments[0], moments[1])
    comptime for b in range(5):
        var other = SIMD[DType.float32, 4](0)
        comptime for j in range(3):
            other[j] = shuffle_down(moments[j], UInt32(16 >> b))
        moments = _combine_moments(moments, other)
    var warp_stats = unsafe_stack_allocation[
        16, Float32, address_space=.SHARED
    ]()
    if tid % 32 == 0:
        warp_stats.unsafe_store(tid // 32 * 4, moments)
    barrier()
    comptime for step in range(2):
        comptime skip = 2 >> step
        if tid < skip:
            var a = warp_stats.unsafe_load[width=4](tid * 4)
            var b = warp_stats.unsafe_load[width=4]((tid + skip) * 4)
            warp_stats.unsafe_store(tid * 4, _combine_moments(a, b))
        barrier()
    var mean = warp_stats.unsafe_load(0)
    var variance = warp_stats.unsafe_load(1) / Float32(width)
    var inv = inlined_assembly[
        "rsqrt.approx.ftz.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](variance + 1e-6)
    for c in range(tid, Int(width), 128):
        var norm = BFloat16((Float32(x.unsafe_load(base + c)) - mean) * inv)
        var scale = BFloat16(
            1 + Float32(mods.unsafe_load(Int(shift) + Int(width) + c))
        )
        var scaled = BFloat16(Float32(norm) * Float32(scale))
        dest.unsafe_store(
            base + c,
            BFloat16(
                Float32(scaled) + Float32(mods.unsafe_load(Int(shift) + c))
            ),
        )


def _add(
    x: Pointer[BFloat16, MutAnyOrigin],
    y: Pointer[BFloat16, MutAnyOrigin],
    mods: Pointer[BFloat16, MutAnyOrigin],
    count: Int64,
    width: Int32,
    gate: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        var delta = BFloat16(
            Float32(y.unsafe_load(i))
            * Float32(mods.unsafe_load(Int(gate) + i % Int(width)))
        )
        x.unsafe_store(i, BFloat16(Float32(x.unsafe_load(i)) + Float32(delta)))


def _glu(
    x: Pointer[BFloat16, MutAnyOrigin],
    dest: Pointer[BFloat16, MutAnyOrigin],
    count: Int64,
    width: Int32,
    stride: Int32,
    offset: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        var pos = i // Int(width) * Int(stride) + Int(offset) + i % Int(width)
        var a = Float32(x.unsafe_load(pos))
        var silu = BFloat16(a / (1 + exp(-a)))
        dest.unsafe_store(
            i,
            BFloat16(Float32(silu) * Float32(x.unsafe_load(pos + Int(width)))),
        )


def _columns(
    src: Pointer[BFloat16, MutAnyOrigin],
    dst: Pointer[BFloat16, MutAnyOrigin],
    count: Int64,
    width: Int32,
    src_stride: Int32,
    dst_stride: Int32,
    dst_col: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        var r = i // Int(width)
        var c = i % Int(width)
        dst.unsafe_store(
            r * Int(dst_stride) + Int(dst_col) + c,
            src.unsafe_load(r * Int(src_stride) + c),
        )


def _rotary(
    packed: Pointer[BFloat16, MutAnyOrigin],
    rot: Pointer[Float32, MutAnyOrigin],
    gain: Pointer[BFloat16, MutAnyOrigin],
    rows: Int32,
    heads: Int32,
):
    # One warp normalizes a complete 128-channel query or key head.
    var i = Int(global_idx.x)
    var lane = i % 32
    var warp = i // 32
    var head = warp % Int(heads)
    var part = warp // Int(heads) % 2
    var row = warp // (2 * Int(heads))
    if row >= Int(rows):
        return
    var base = ((row * 3 + part) * Int(heads) + head) * 128
    var v = SIMD[DType.float32, 4](0)
    var ss = Float32(0)
    comptime for j in range(4):
        v[j] = Float32(packed.unsafe_load(base + lane * 4 + j))
        ss += v[j] * v[j]
    var scale = 1 / sqrt(_sum(ss) / 128 + 1e-6)
    comptime for j in range(4):
        var norm = BFloat16(v[j] * scale)
        v[j] = Float32(
            BFloat16(
                Float32(norm)
                * Float32(gain.unsafe_load(part * 128 + lane * 4 + j))
            )
        )
    comptime for j in range(0, 4, 2):
        var cs = rot.unsafe_load(row * 128 + lane * 4 + j)
        var sn = rot.unsafe_load(row * 128 + lane * 4 + j + 1)
        packed.unsafe_store(
            base + lane * 4 + j, BFloat16(cs * v[j] - sn * v[j + 1])
        )
        packed.unsafe_store(
            base + lane * 4 + j + 1, BFloat16(sn * v[j] + cs * v[j + 1])
        )


def device_floats(
    ctx: DeviceContext, x: Tensor
) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](len(x.f))
    with buf.map_to_host() as mapped:
        _ = external_call["memcpy", Int](
            Int(mapped.unsafe_ptr()), x.fptr(), len(x.f) * 4
        )
    return buf^


struct DeviceRows(Movable):
    var buf: DeviceBuffer[DType.bfloat16]
    var rows: Int
    var cols: Int

    def __init__(
        out self, var buf: DeviceBuffer[DType.bfloat16], rows: Int, cols: Int
    ):
        self.buf = buf^
        self.rows = rows
        self.cols = cols

    @staticmethod
    def empty(ctx: DeviceContext, rows: Int, cols: Int) raises -> Self:
        return Self(
            ctx.enqueue_create_buffer[DType.bfloat16](rows * cols), rows, cols
        )

    @staticmethod
    def upload(ctx: DeviceContext, x: Tensor) raises -> Self:
        var dst = Self.empty(ctx, x.shape[0], len(x.f) // x.shape[0])
        with dst.buf.map_to_host() as host:
            for i in range(len(x.f)):
                host[i] = BFloat16(x.f[i])
        return dst^

    def host(self) raises -> Tensor:
        var dst = Tensor.zeros([self.rows, self.cols])
        with self.buf.map_to_host() as host:
            for i in range(len(dst.f)):
                dst.f[i] = Float32(host[i])
        return dst^

    def project(self, gpu: UltraTexGpu, weight: Int, co: Int) raises -> Self:
        var dst = Self.empty(gpu.ctx, self.rows, co)
        gpu.ctx.enqueue_function[_project_tiled[DType.bfloat16]](
            self.buf.unsafe_ptr(),
            gpu.weights[weight].unsafe_ptr(),
            dst.buf.unsafe_ptr(),
            Int32(self.rows),
            Int32(self.cols),
            Int32(co),
            grid_dim=((co + 63) // 64, (self.rows + 63) // 64),
            block_dim=128,
        )
        return dst^

    def norm(self, ctx: DeviceContext, mods: Self, chunk: Int) raises -> Self:
        var dst = Self.empty(ctx, self.rows, self.cols)
        ctx.enqueue_function[_norm](
            self.buf.unsafe_ptr(),
            mods.buf.unsafe_ptr(),
            dst.buf.unsafe_ptr(),
            Int32(self.rows),
            Int32(self.cols),
            Int32(chunk * self.cols),
            grid_dim=self.rows,
            block_dim=128,
        )
        return dst^

    def residual(
        mut self, ctx: DeviceContext, y: Self, mods: Self, gate: Int
    ) raises:
        var n = self.rows * self.cols
        ctx.enqueue_function[_add](
            self.buf.unsafe_ptr(),
            y.buf.unsafe_ptr(),
            mods.buf.unsafe_ptr(),
            Int64(n),
            Int32(self.cols),
            Int32(gate * self.cols),
            grid_dim=(n + 255) // 256,
            block_dim=256,
        )

    def glu(self, ctx: DeviceContext, offset: Int) raises -> Self:
        var width = (self.cols - offset) // 2
        var dst = Self.empty(ctx, self.rows, width)
        var n = self.rows * width
        ctx.enqueue_function[_glu](
            self.buf.unsafe_ptr(),
            dst.buf.unsafe_ptr(),
            Int64(n),
            Int32(width),
            Int32(self.cols),
            Int32(offset),
            grid_dim=(n + 255) // 256,
            block_dim=256,
        )
        return dst^

    def copy_columns(
        self, ctx: DeviceContext, mut dst: Self, width: Int, offset: Int = 0
    ) raises:
        var n = self.rows * width
        ctx.enqueue_function[_columns](
            self.buf.unsafe_ptr(),
            dst.buf.unsafe_ptr(),
            Int64(n),
            Int32(width),
            Int32(self.cols),
            Int32(dst.cols),
            Int32(offset),
            grid_dim=(n + 255) // 256,
            block_dim=256,
        )

    def rotate(
        mut self,
        ctx: DeviceContext,
        rot: DeviceBuffer[DType.float32],
        gain: Self,
        heads: Int,
    ) raises:
        ctx.enqueue_function[_rotary](
            self.buf.unsafe_ptr(),
            rot.unsafe_ptr(),
            gain.buf.unsafe_ptr(),
            Int32(self.rows),
            Int32(heads),
            grid_dim=(self.rows * heads * 2 + 3) // 4,
            block_dim=128,
        )

    def slice(self, first: Int, n: Int) raises -> Self:
        return Self(
            self.buf.create_sub_buffer[DType.bfloat16](
                first * self.cols, n * self.cols
            ),
            n,
            self.cols,
        )

    @staticmethod
    def join(ctx: DeviceContext, a: Self, b: Self) raises -> Self:
        var dst = Self.empty(ctx, a.rows + b.rows, a.cols)
        ctx.enqueue_copy(
            dst.buf.create_sub_buffer[DType.bfloat16](0, a.rows * a.cols), a.buf
        )
        ctx.enqueue_copy(
            dst.buf.create_sub_buffer[DType.bfloat16](
                a.rows * a.cols, b.rows * b.cols
            ),
            b.buf,
        )
        return dst^
