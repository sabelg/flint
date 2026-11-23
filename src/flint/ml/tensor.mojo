"""Dense CPU tensors for the ONNX interpreter.

A tensor is float32 (ONNX type 1) or integer: int32 (6), int64 (7) or bool (9). Integer and
bool tensors keep their values in `i` as `Int`; float tensors keep theirs in `f`, or borrow
them from a memory-mapped model file (`ext`, the address of the first float), so a model's
weights are never copied.
"""

from std.memory import bitcast

comptime DT_F32: Int = 1
comptime DT_I32: Int = 6
comptime DT_I64: Int = 7
comptime DT_BOOL: Int = 9


@always_inline
def fptr_at(address: Int) -> Pointer[Float32, MutAnyOrigin]:
    return Pointer[Float32, MutAnyOrigin](unsafe_from_address=address)


def numel_of(shape: List[Int]) -> Int:
    var n = 1
    for d in shape:
        if d < 0:
            return -1
        if d == 0:
            n = 0
        elif n > 0 and d > 0x7FFFFFFFFFFFFFFF // n:
            return -1
        else:
            n *= d
    return n


def shape_str(shape: List[Int]) -> String:
    var s = String("[")
    for k in range(len(shape)):
        if k > 0:
            s += ","
        s += String(shape[k])
    return s + "]"


def strides_of(shape: List[Int]) -> List[Int]:
    var st = List[Int](length=len(shape), fill=1)
    var acc = 1
    var k = len(shape) - 1
    while k >= 0:
        st[k] = acc
        acc *= shape[k]
        k -= 1
    return st^


def _invalid_tensor(dtype: Int = DT_F32) -> Tensor:
    var t = Tensor()
    t.dtype = dtype
    t.shape = [0]
    return t^


struct Tensor(Copyable, Movable):
    var dtype: Int
    var shape: List[Int]
    var f: List[Float32]
    var i: List[Int]
    var ext: Int

    def __init__(out self):
        self.dtype = DT_F32
        self.shape = List[Int]()
        self.f = List[Float32]()
        self.i = List[Int]()
        self.ext = 0

    @staticmethod
    def floats(var shape: List[Int], var data: List[Float32]) -> Tensor:
        var n = numel_of(shape)
        if n < 0 or len(data) < n:
            return _invalid_tensor(DT_F32)
        var t = Tensor()
        t.shape = shape^
        t.f = data^
        return t^

    @staticmethod
    def zeros(var shape: List[Int]) -> Tensor:
        var n = numel_of(shape)
        if n < 0:
            return _invalid_tensor(DT_F32)
        var t = Tensor()
        t.shape = shape^
        t.f = List[Float32](length=n, fill=0.0)
        return t^

    @staticmethod
    def ints(dtype: Int, var shape: List[Int], var data: List[Int]) -> Tensor:
        var n = numel_of(shape)
        if n < 0 or len(data) < n:
            return _invalid_tensor(dtype)
        var t = Tensor()
        t.dtype = dtype
        t.shape = shape^
        t.i = data^
        return t^

    @staticmethod
    def borrowed(var shape: List[Int], address: Int) -> Tensor:
        var n = numel_of(shape)
        if n < 0 or (n > 0 and address == 0):
            return _invalid_tensor(DT_F32)
        var t = Tensor()
        t.shape = shape^
        t.ext = address
        return t^

    @staticmethod
    def like(src: Tensor, var shape: List[Int]) -> Tensor:
        """An empty (zero) tensor of `src`'s type."""
        var n = numel_of(shape)
        if n < 0:
            return _invalid_tensor(src.dtype)
        var t = Tensor()
        t.dtype = src.dtype
        t.shape = shape^
        if src.dtype == DT_F32:
            t.f = List[Float32](length=n, fill=0.0)
        else:
            t.i = List[Int](length=n, fill=0)
        return t^

    def numel(self) -> Int:
        return numel_of(self.shape)

    def has_valid_storage(self) -> Bool:
        var n = self.numel()
        if n < 0:
            return False
        if n == 0:
            return True
        if self.is_float():
            return self.ext != 0 or len(self.f) >= n
        return len(self.i) >= n

    def rank(self) -> Int:
        return len(self.shape)

    def is_float(self) -> Bool:
        return self.dtype == DT_F32

    def fptr(self) -> Int:
        if self.ext != 0:
            return self.ext
        return Int(self.f.unsafe_ptr())

    @always_inline
    def fget(self, k: Int) -> Float32:
        if self.ext != 0:
            return fptr_at(self.ext).unsafe_offset(k).unsafe_load[alignment=1]()
        return self.f[k]

    def owned_floats(self) -> List[Float32]:
        """The float values as an owned list (copies borrowed weights)."""
        if self.ext == 0:
            return self.f.copy()
        var n = self.numel()
        if n <= 0:
            return List[Float32]()
        var out = List[Float32](length=n, fill=0.0)
        var p = fptr_at(self.ext)
        for k in range(n):
            out[k] = p.unsafe_offset(k).unsafe_load[alignment=1]()
        return out^

    def get(self, k: Int) -> Float64:
        """Element `k` as a double, whatever the type."""
        if k < 0:
            return Float64(0)
        if self.dtype == DT_F32:
            if self.ext != 0:
                if k >= self.numel():
                    return Float64(0)
                return Float64(self.fget(k))
            if k >= len(self.f):
                return Float64(0)
            return Float64(self.f[k])
        if k >= len(self.i):
            return Float64(0)
        return Float64(self.i[k])

    def int_at(self, k: Int) -> Int:
        if k < 0:
            return 0
        if self.dtype == DT_F32:
            var v: Float32
            if self.ext != 0:
                if k >= self.numel():
                    return 0
                v = self.fget(k)
            else:
                if k >= len(self.f):
                    return 0
                v = self.f[k]
            if v != v or v <= Float32(-9e18) or v >= Float32(9e18):
                return 0
            return Int(v)
        if k >= len(self.i):
            return 0
        return self.i[k]

    def int_list(self) -> List[Int]:
        var n = self.numel()
        if n <= 0 or not self.has_valid_storage():
            return List[Int]()
        var out = List[Int](capacity=n)
        for k in range(n):
            out.append(self.int_at(k))
        return out^

    def reshaped(self, var shape: List[Int]) -> Tensor:
        """The same values with another shape (borrowed weights stay borrowed)."""
        var n = numel_of(shape)
        var sn = self.numel()
        if n < 0 or sn < 0 or n != sn:
            return _invalid_tensor(self.dtype)
        var t = Tensor()
        t.dtype = self.dtype
        t.shape = shape^
        t.ext = self.ext
        if self.ext == 0:
            if self.dtype == DT_F32:
                t.f = self.f.copy()
            else:
                t.i = self.i.copy()
        return t^

    def clear(mut self):
        self.shape = List[Int]()
        self.f = List[Float32]()
        self.i = List[Int]()
        self.ext = 0


def broadcast_shape(a: List[Int], b: List[Int]) raises -> List[Int]:
    var r = max(len(a), len(b))
    var out = List[Int](length=r, fill=1)
    for k in range(r):
        var da = a[k - (r - len(a))] if k >= r - len(a) else 1
        var db = b[k - (r - len(b))] if k >= r - len(b) else 1
        if da < 0 or db < 0:
            raise Error("broadcast: negative dimension")
        if da == db or db == 1:
            out[k] = da
        elif da == 1:
            out[k] = db
        else:
            raise Error("broadcast: incompatible shapes " + shape_str(a) + " " + shape_str(b))
    return out^


def broadcast_offsets(shape: List[Int], out_shape: List[Int]) -> List[Int]:
    """For each element of `out_shape` (row major), the offset of the element of a tensor of
    `shape` that numpy broadcasting maps to it."""
    var r = len(out_shape)
    var n = numel_of(out_shape)
    if n < 0:
        return List[Int]()
    var st = List[Int](length=r, fill=0)
    var acc = 1
    var k = len(shape) - 1
    var o = r - 1
    while o >= 0:
        if k >= 0:
            st[o] = 0 if shape[k] == 1 and out_shape[o] != 1 else acc
            acc *= shape[k]
        k -= 1
        o -= 1
    var offs = List[Int](length=n, fill=0)
    if n == 0:
        return offs^
    var ctr = List[Int](length=r, fill=0)
    var cur = 0
    for e in range(n):
        offs[e] = cur
        var d = r - 1
        while d >= 0:
            ctr[d] += 1
            cur += st[d]
            if ctr[d] < out_shape[d]:
                break
            cur -= st[d] * ctr[d]
            ctr[d] = 0
            d -= 1
    return offs^


def norm_axis(axis: Int, rank: Int) -> Int:
    return axis + rank if axis < 0 else axis


def f32_from_bits(bits: UInt32) -> Float32:
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](bits))[0]
