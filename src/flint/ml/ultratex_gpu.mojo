"""Resident BF16 denoiser projections on a MAX device. NVIDIA uses warp MMA."""
from std.utils import StaticTuple
from std.sys import is_nvidia_gpu
from std.ffi import external_call
from max.gpu import block_idx, global_idx, thread_idx
from max.gpu.sync import barrier
from std.memory import bitcast, unsafe_stack_allocation
from max.gpu.compute.mma import ld_matrix, mma
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from flint.ml.tensor import Tensor
from flint.ml.safetensors import SafeTensorWeights


def _project(
    a: Pointer[BFloat16, MutAnyOrigin],
    w: Pointer[BFloat16, MutAnyOrigin],
    result: Pointer[Float32, MutAnyOrigin],
    num_rows: Int32,
    inner: Int32,
    num_cols: Int32,
):
    var m = Int(num_rows)
    var k = Int(inner)
    var n = Int(num_cols)
    var thread = Int(global_idx.x)
    var lane = thread % 32
    var tile = thread // 32
    var tiles_n = (n + 7) // 8
    var row = 16 * (tile // tiles_n)
    var col = 8 * (tile % tiles_n)
    if row >= m:
        return
    comptime if is_nvidia_gpu():
        var acc = SIMD[DType.float32, 4](0)
        var group = lane // 4
        var pair = 2 * (lane % 4)
        for first in range(0, k, 16):
            var af = SIMD[DType.bfloat16, 8](0)
            var bf = SIMD[DType.bfloat16, 4](0)
            comptime for j in range(8):
                var r = row + group + 8 * ((j // 2) % 2)
                var inner = first + pair + (j % 2) + 8 * (j // 4)
                if r < m and inner < k:
                    af[j] = a.unsafe_load(r * k + inner)
            comptime for j in range(4):
                var inner = first + pair + j % 2 + 8 * (j // 2)
                if col + group < n and inner < k:
                    bf[j] = w.unsafe_load((col + group) * k + inner)
            var previous = acc
            mma(acc, af, bf, previous)
        comptime for j in range(4):
            var r = row + group + 8 * (j // 2)
            var c = col + pair + j % 2
            if r < m and c < n:
                result.unsafe_store(r * n + c, acc[j])
    else:
        for j in range(lane, 128, 32):
            var r = row + j // 8
            var c = col + j % 8
            if r < m and c < n:
                var sum = Float32(0)
                for i in range(k):
                    sum += Float32(a.unsafe_load(r * k + i)) * Float32(
                        w.unsafe_load(c * k + i)
                    )
                result.unsafe_store(r * n + c, sum)


def _project_tiled[
    dtype: DType = DType.float32
](
    a: Pointer[BFloat16, MutAnyOrigin],
    w: Pointer[BFloat16, MutAnyOrigin],
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    num_rows: Int32,
    inner: Int32,
    num_cols: Int32,
):
    # The four warps share 64 rows of inputs and weights for each 32-channel slice.
    # Padding the shared stride keeps simultaneous fragment reads in different banks.
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var group = lane // 4
    var pair = lane % 4 * 2
    var k = Int(inner)
    var n = Int(num_cols)
    var base_row = Int(block_idx.y) * 64
    var base_col = Int(block_idx.x) * 64
    var wr = (tid // 64) * 32
    var wc = ((tid // 32) % 2) * 32
    var sm = unsafe_stack_allocation[
        128 * 40, BFloat16, address_space=.SHARED
    ]()
    var sums = StaticTuple[SIMD[DType.float32, 4], 8](fill=0)
    for first in range(0, k, 32):
        comptime for chunk in range(2):
            var r = tid // 4 + 32 * chunk
            var off = tid % 4 * 8
            var v = SIMD[DType.bfloat16, 8](0)
            if base_row + r < Int(num_rows):
                v = a.unsafe_load[width=8]((base_row + r) * k + first + off)
            sm.unsafe_store(r * 40 + off, v)
            var weight = SIMD[DType.bfloat16, 8](0)
            if base_col + r < n:
                weight = w.unsafe_load[width=8](
                    (base_col + r) * k + first + off
                )
            sm.unsafe_store((64 + r) * 40 + off, weight)
        barrier()
        comptime for part in range(2):
            var bs = StaticTuple[SIMD[DType.bfloat16, 4], 4](fill=0)
            comptime for target in range(4):
                var wp = (
                    (64 + wc + 8 * target + lane % 8) * 40
                    + 16 * part
                    + 8 * ((lane // 8) % 2)
                )
                bs[target] = ld_matrix[4](sm.unsafe_offset(wp))
            comptime for m in range(2):
                var ap = (
                    (wr + m * 16 + lane % 16) * 40
                    + 16 * part
                    + 8 * (lane // 16)
                )
                var af = ld_matrix[8](sm.unsafe_offset(ap))
                comptime for target in range(4):
                    var prev = sums[m * 4 + target]
                    mma(sums[m * 4 + target], af, bs[target], prev)
        barrier()
    comptime for m in range(2):
        comptime for target in range(4):
            comptime for j in range(4):
                var r = base_row + wr + 16 * m + group + 8 * (j // 2)
                var c = base_col + wc + target * 8 + pair + j % 2
                if r < Int(num_rows) and c < n:
                    result.unsafe_store(
                        r * n + c, sums[m * 4 + target][j].cast[dtype]()
                    )


def _adapter_merge(
    down: Pointer[Float32, MutAnyOrigin],
    up: Pointer[Float32, MutAnyOrigin],
    weight: Pointer[BFloat16, MutAnyOrigin],
    rows: Int32,
    cols: Int32,
    rank: Int32,
):
    # PyTorch's CUDA FP32 high-precision matmul uses TF32 tensor cores here.
    var thread = Int(global_idx.x)
    var lane = thread % 32
    var warp = thread // 32
    var tiles = (Int(cols) + 7) // 8
    var row = (warp // tiles) * 16
    var col = (warp % tiles) * 8
    if row >= Int(rows):
        return
    var group = lane // 4
    var sub = lane % 4
    var acc = SIMD[DType.float32, 4](0)
    for first in range(0, Int(rank), 8):
        var a = SIMD[DType.float32, 4](0)
        var b = SIMD[DType.float32, 2](0)
        comptime for j in range(4):
            var r = row + group + 8 * (j % 2)
            var k = first + sub + 4 * (j // 2)
            if r < Int(rows) and k < Int(rank):
                a[j] = up.unsafe_load(r * Int(rank) + k)
        comptime for j in range(2):
            var k = first + sub + 4 * j
            if col + group < Int(cols) and k < Int(rank):
                b[j] = down.unsafe_load(k * Int(cols) + col + group)
        var prev = acc
        mma(acc, a, b, prev)
    comptime for j in range(4):
        var r = row + group + 8 * (j // 2)
        var c = col + 2 * sub + j % 2
        if r < Int(rows) and c < Int(cols):
            var i = r * Int(cols) + c
            var delta = BFloat16(acc[j])
            weight.unsafe_store(
                i, BFloat16(Float32(weight.unsafe_load(i)) + Float32(delta))
            )


struct UltraTexGpu(Movable):
    var ctx: DeviceContext
    var weights: List[DeviceBuffer[DType.bfloat16]]
    var names: Dict[String, Int]

    var host_input: HostBuffer[DType.bfloat16]
    var host_output: HostBuffer[DType.float32]
    var input: DeviceBuffer[DType.bfloat16]
    var output: DeviceBuffer[DType.float32]
    var input_capacity: Int
    var output_capacity: Int

    def __init__(out self) raises:
        self = Self(DeviceContext())

    def __init__(out self, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.weights = List[DeviceBuffer[DType.bfloat16]]()
        self.names = Dict[String, Int]()
        self.input_capacity = 1
        self.output_capacity = 1
        self.host_input = self.ctx.enqueue_create_host_buffer[DType.bfloat16](1)
        self.host_output = self.ctx.enqueue_create_host_buffer[DType.float32](1)
        self.input = self.ctx.enqueue_create_buffer[DType.bfloat16](1)
        self.output = self.ctx.enqueue_create_buffer[DType.float32](1)

    def _reserve(mut self, input_size: Int, output_size: Int) raises:
        if input_size > self.input_capacity:
            self.input_capacity = input_size
            self.host_input = self.ctx.enqueue_create_host_buffer[
                DType.bfloat16
            ](input_size)
            self.input = self.ctx.enqueue_create_buffer[DType.bfloat16](
                input_size
            )
        if output_size > self.output_capacity:
            self.output_capacity = output_size
            self.host_output = self.ctx.enqueue_create_host_buffer[
                DType.float32
            ](output_size)
            self.output = self.ctx.enqueue_create_buffer[DType.float32](
                output_size
            )

    def weight_id(
        mut self, weights: SafeTensorWeights, key: String, co: Int, k: Int
    ) raises -> Int:
        var id = self.names.get(key, -1)
        if id < 0:
            var offset, dtype = weights.storage(key + ".weight", [co, k])
            var buffer = self.ctx.enqueue_create_buffer[DType.bfloat16](co * k)
            var host = self.ctx.enqueue_create_host_buffer[DType.bfloat16](
                co * k
            )
            if dtype == "BF16":
                _ = external_call["memcpy", Int](
                    Int(host.unsafe_ptr()),
                    weights.address() + offset,
                    2 * co * k,
                )
            else:
                var w = weights.tensor(key + ".weight", [co, k])
                for i in range(len(w.f)):
                    host[i] = BFloat16(w.f[i])
            self.ctx.enqueue_copy(buffer, host)
            self.ctx.synchronize()
            id = len(self.weights)
            self.weights.append(buffer^)
            self.names[key] = id
        return id

    def _adapter_matrix(self, x: Tensor) raises -> DeviceBuffer[DType.float32]:
        var buf = self.ctx.enqueue_create_buffer[DType.float32](len(x.f))
        with buf.map_to_host() as host:
            for i in range(len(x.f)):
                var bits = bitcast[DType.uint32](x.f[i])
                host[i] = bitcast[DType.float32](
                    (bits + 4095 + ((bits >> 13) & 1)) & 0xFFFFE000
                )
        return buf^

    def adapted_weight_id(
        mut self,
        base: SafeTensorWeights,
        adapter: SafeTensorWeights,
        key: String,
        co: Int,
        k: Int,
        rank: Int,
    ) raises -> Int:
        var existing = self.names.get(key, -1)
        if existing >= 0:
            return existing
        var prefix = "base_model.model." + key + ".lora_"
        var down = self._adapter_matrix(
            adapter.tensor(prefix + "A.default.weight", [rank, k])
        )
        var up = self._adapter_matrix(
            adapter.tensor(prefix + "B.default.weight", [co, rank])
        )
        var id = self.weight_id(base, key, co, k)
        var warps = ((co + 15) // 16) * ((k + 7) // 8)
        self.ctx.enqueue_function[_adapter_merge](
            down.unsafe_ptr(),
            up.unsafe_ptr(),
            self.weights[id].unsafe_ptr(),
            Int32(co),
            Int32(k),
            Int32(rank),
            grid_dim=(warps + 3) // 4,
            block_dim=128,
        )
        return id

    def linear(
        mut self, weights: SafeTensorWeights, x: Tensor, key: String, co: Int
    ) raises -> Tensor:
        var k = x.shape[1]
        var rows = x.shape[0]
        var id = self.weight_id(weights, key, co, k)
        var result = Tensor.zeros([rows, co])
        # Keep temporary device activation memory bounded even for 2K view grids.
        var batch = min(rows, 512)
        self._reserve(batch * k, batch * co)
        for first in range(0, rows, batch):
            var nrows = min(batch, rows - first)
            for i in range(nrows * k):
                self.host_input[i] = BFloat16(x.f[first * k + i])
            self.ctx.enqueue_copy(
                self.input.create_sub_buffer[DType.bfloat16](0, nrows * k),
                self.host_input.create_sub_buffer[DType.bfloat16](0, nrows * k),
            )
            if (
                co % 32 == 0
                and k % 32 == 0
                and nrows >= 32
                and self.ctx.target.is_nvidia_gpu()
            ):
                self.ctx.enqueue_function[_project_tiled[DType.float32]](
                    self.input.unsafe_ptr(),
                    self.weights[id].unsafe_ptr(),
                    self.output.unsafe_ptr(),
                    Int32(nrows),
                    Int32(k),
                    Int32(co),
                    grid_dim=((co + 63) // 64, (nrows + 63) // 64),
                    block_dim=128,
                )
            else:
                var threads = ((nrows + 15) // 16) * ((co + 7) // 8) * 32
                self.ctx.enqueue_function[_project](
                    self.input.unsafe_ptr(),
                    self.weights[id].unsafe_ptr(),
                    self.output.unsafe_ptr(),
                    Int32(nrows),
                    Int32(k),
                    Int32(co),
                    grid_dim=(threads + 127) // 128,
                    block_dim=128,
                )
            self.ctx.enqueue_copy(
                self.host_output.create_sub_buffer[DType.float32](
                    0, nrows * co
                ),
                self.output.create_sub_buffer[DType.float32](0, nrows * co),
            )
            self.ctx.synchronize()
            _ = external_call["memcpy", Int](
                result.fptr() + first * co * 4,
                Int(self.host_output.unsafe_ptr()),
                nrows * co * 4,
            )
        return result^
