"""MAX GPU backend for the ONNX interpreter's matrix products.

Weights upload once, as [k, n] row-major device buffers (Gemm's transB and Conv weights are
transposed on upload), and stay resident. A product stages its small activation matrix in a
pinned host buffer, copies it over, runs a split-K kernel (one thread per output column and
K slice, eight rows at a time, so neighbouring threads read neighbouring weights) and a
reduction kernel that adds the slices and the bias, then copies the result back: one
synchronisation per product. Staging and partial buffers grow on demand and are reused.
"""

from std.collections import Dict
from std.ffi import external_call
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from flint.ml.tensor import fptr_at

comptime BLOCK: Int = 128
comptime ROWS: Int = 8
comptime SLICE_K: Int = 64


def _mm_partial(a: Pointer[Float32, MutAnyOrigin], b: Pointer[Float32, MutAnyOrigin],
                part: Pointer[Float32, MutAnyOrigin], m: Int32, k: Int32, n: Int32, slices: Int32):
    """part[s, i, j] = sum over k in slice s of A[i, k] * B[k, j]."""
    var tid = Int(global_idx.x)
    var nn = Int(n)
    var j = tid % nn
    var s = tid // nn
    if s >= Int(slices):
        return
    var mm = Int(m)
    var kn = Int(k)
    var k0 = s * SLICE_K
    var k1 = min(kn, k0 + SLICE_K)
    var i0 = 0
    while i0 < mm:
        var acc = SIMD[DType.float32, ROWS](0)
        var rows = min(ROWS, mm - i0)
        for kk in range(k0, k1):
            var w = b.unsafe_offset(kk * nn + j).unsafe_load()
            var x = SIMD[DType.float32, ROWS](0)
            comptime for r in range(ROWS):
                if r < rows:
                    x[r] = a.unsafe_offset((i0 + r) * kn + kk).unsafe_load()
            acc += x * w
        comptime for r in range(ROWS):
            if r < rows:
                part.unsafe_offset((s * mm + i0 + r) * nn + j).unsafe_store(acc[r])
        i0 += ROWS


def _mm_reduce(part: Pointer[Float32, MutAnyOrigin], c: Pointer[Float32, MutAnyOrigin],
               bias: Pointer[Float32, MutAnyOrigin], m: Int32, n: Int32, slices: Int32, has_bias: Int32):
    var e = Int(global_idx.x)
    var total = Int(m) * Int(n)
    if e >= total:
        return
    var acc: Float32 = 0
    if has_bias != 0:
        acc = bias.unsafe_offset(e % Int(n)).unsafe_load()
    for s in range(Int(slices)):
        acc += part.unsafe_offset(s * total + e).unsafe_load()
    c.unsafe_offset(e).unsafe_store(acc)


@always_inline
def _dptr(buf: DeviceBuffer[DType.float32]) -> Pointer[Float32, MutAnyOrigin]:
    return Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(buf.unsafe_ptr()))


struct GpuMatmul(Movable):
    """Resident weights on one MAX `DeviceContext`, keyed by (value id, transposed)."""
    var ctx: DeviceContext
    var weights: Dict[Int, Int]
    var buffers: List[DeviceBuffer[DType.float32]]
    var uploaded_bytes: Int
    var cap_a: Int
    var cap_c: Int
    var cap_p: Int
    var host_a: HostBuffer[DType.float32]
    var host_c: HostBuffer[DType.float32]
    var dev_a: DeviceBuffer[DType.float32]
    var dev_c: DeviceBuffer[DType.float32]
    var dev_p: DeviceBuffer[DType.float32]
    var products: Int

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.weights = Dict[Int, Int]()
        self.buffers = List[DeviceBuffer[DType.float32]]()
        self.uploaded_bytes = 0
        self.cap_a = 1 << 16
        self.cap_c = 1 << 16
        self.cap_p = 1 << 20
        self.host_a = self.ctx.enqueue_create_host_buffer[DType.float32](self.cap_a)
        self.host_c = self.ctx.enqueue_create_host_buffer[DType.float32](self.cap_c)
        self.dev_a = self.ctx.enqueue_create_buffer[DType.float32](self.cap_a)
        self.dev_c = self.ctx.enqueue_create_buffer[DType.float32](self.cap_c)
        self.dev_p = self.ctx.enqueue_create_buffer[DType.float32](self.cap_p)
        self.products = 0

    def name(self) raises -> String:
        return self.ctx.name()

    def weight(mut self, key: Int, src: Int, rows: Int, cols: Int, transpose: Bool) raises -> Int:
        """Index of the device buffer holding the float32 matrix at `src` ([rows, cols]), uploaded
        as [cols, rows] when `transpose`."""
        var found = self.weights.get(key)
        if found:
            return found.value()
        if rows <= 0 or cols <= 0 or src == 0:
            raise Error("GpuMatmul.weight: invalid dimensions or null pointer")
        if rows > 0x7FFFFFFF // cols or rows * cols > 0x7FFFFFFF // 4:
            raise Error("GpuMatmul.weight: buffer size overflow")
        var n = rows * cols
        var buf = self.ctx.enqueue_create_buffer[DType.float32](n)
        with buf.map_to_host() as h:
            var p = fptr_at(src)
            if transpose:
                for r in range(rows):
                    for c in range(cols):
                        h[c * rows + r] = p.unsafe_offset(r * cols + c).unsafe_load[alignment=1]()
            else:
                var dst = Int(h.unsafe_ptr())
                _ = external_call["memcpy", Int](dst, src, n * 4)
        var idx = len(self.buffers)
        self.buffers.append(buf^)
        self.weights[key] = idx
        self.uploaded_bytes += n * 4
        return idx

    def _reserve(mut self, a: Int, c: Int, p: Int) raises:
        if a > self.cap_a:
            self.ctx.synchronize()
            self.cap_a = a
            self.host_a = self.ctx.enqueue_create_host_buffer[DType.float32](a)
            self.dev_a = self.ctx.enqueue_create_buffer[DType.float32](a)
        if c > self.cap_c:
            self.ctx.synchronize()
            self.cap_c = c
            self.host_c = self.ctx.enqueue_create_host_buffer[DType.float32](c)
            self.dev_c = self.ctx.enqueue_create_buffer[DType.float32](c)
        if p > self.cap_p:
            self.ctx.synchronize()
            self.cap_p = p
            self.dev_p = self.ctx.enqueue_create_buffer[DType.float32](p)

    def matmul(mut self, a: Int, w: Int, c: Int, bias: Int, bias_key: Int, m: Int, k: Int, n: Int) raises:
        """C[m,n] = A[m,k] @ W[k,n] (+ bias[n]); A and C are host addresses, W a resident buffer."""
        if m <= 0 or k <= 0 or n <= 0:
            raise Error("GpuMatmul.matmul: dimensions must be positive")
        if a == 0 or c == 0 or w < 0 or w >= len(self.buffers):
            raise Error("GpuMatmul.matmul: invalid buffer or pointer")
        var slices = (k + SLICE_K - 1) // SLICE_K
        if m > 0x7FFFFFFF // max(k, n) or (m * n) > 0x7FFFFFFF // max(4, slices) or (m * k) > 0x7FFFFFFF // 4:
            raise Error("GpuMatmul.matmul: tensor size overflow")
        if slices <= 0 or n > 0x7FFFFFFF // slices:
            raise Error("GpuMatmul.matmul: grid size overflow")
        self._reserve(m * k, m * n, slices * m * n)
        var bidx = -1
        if bias != 0:
            bidx = self.weight(bias_key, bias, 1, n, False)
        _ = external_call["memcpy", Int](Int(self.host_a.unsafe_ptr()), a, m * k * 4)
        self.ctx.enqueue_copy(self.dev_a.create_sub_buffer[DType.float32](0, m * k),
                              self.host_a.create_sub_buffer[DType.float32](0, m * k))
        var pw = _dptr(self.buffers[w])
        var pb = _dptr(self.dev_c)
        if bidx >= 0:
            pb = _dptr(self.buffers[bidx])
        var threads = slices * n
        self.ctx.enqueue_function[_mm_partial](
            _dptr(self.dev_a), pw, _dptr(self.dev_p), Int32(m), Int32(k), Int32(n), Int32(slices),
            grid_dim=(threads + BLOCK - 1) // BLOCK, block_dim=BLOCK)
        self.ctx.enqueue_function[_mm_reduce](
            _dptr(self.dev_p), _dptr(self.dev_c), pb, Int32(m), Int32(n), Int32(slices),
            Int32(1 if bidx >= 0 else 0),
            grid_dim=(m * n + BLOCK - 1) // BLOCK, block_dim=BLOCK)
        self.ctx.enqueue_copy(self.host_c.create_sub_buffer[DType.float32](0, m * n),
                              self.dev_c.create_sub_buffer[DType.float32](0, m * n))
        self.ctx.synchronize()
        _ = external_call["memcpy", Int](c, Int(self.host_c.unsafe_ptr()), m * n * 4)
        self.products += 1
