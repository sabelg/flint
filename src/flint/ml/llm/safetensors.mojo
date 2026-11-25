"""Memory-mapped safetensors reader (`flint.ml.llm.safetensors`).

A safetensors file is an 8-byte little-endian header length, a JSON header naming every tensor
(`dtype`, `shape`, `data_offsets` relative to the end of the header) and the raw data.
`SafeTensors.open_dir(dir)` maps every `*.safetensors` file in a directory (sharded checkpoints
included) read-only; tensors are zero-copy views into the mappings. `to_f32` widens BF16 / F16 /
F32 data into a fresh float32 buffer.
"""

from std.collections import Dict
from std.ffi import external_call
from std.memory import bitcast
from flint.platform.os import open_read, close_fd, list_directory, read_at
from flint.assets.json import parse_json


@fieldwise_init
struct SafeTensor(Copyable, Movable):
    var name: String
    var dtype: String       # "BF16", "F16", "F32", ...
    var shape: List[Int]
    var addr: Int           # first byte inside the mapping

    def numel(self) -> Int:
        var n = 1
        for d in self.shape:
            n *= d
        return n

    def rows(self) -> Int:
        return self.shape[0] if len(self.shape) > 0 else 1

    def cols(self) -> Int:
        var n = 1
        for i in range(1, len(self.shape)):
            n *= self.shape[i]
        return n


@always_inline
def bf16_to_f32(bits: UInt16) -> Float32:
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](UInt32(bits) << 16))[0]


@always_inline
def f16_to_f32(h: UInt16) -> Float32:
    var sign = UInt32(h >> 15) << 31
    var exp = Int((h >> 10) & 31)
    var man = UInt32(h & 1023)
    if exp == 0:
        if man == 0:
            return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](sign))[0]
        var f = Float32(man) * Float32(5.960464477539063e-08)   # 2^-24
        return -f if sign != 0 else f
    if exp == 31:
        return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](sign | 0x7F800000 | (man << 13)))[0]
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](sign | (UInt32(exp + 112) << 23) | (man << 13)))[0]


def widen_to_f32(t: SafeTensor, dst: Int, start: Int, count: Int) raises:
    """Writes elements [start, start + count) of `t` as float32 to `dst`."""
    var out = Pointer[Float32, MutAnyOrigin](unsafe_from_address=dst)
    if t.dtype == "F32":
        _ = external_call["memcpy", Int](dst, t.addr + start * 4, count * 4)
        return
    var src = Pointer[UInt16, MutAnyOrigin](unsafe_from_address=t.addr + start * 2)
    if t.dtype == "BF16":
        var i = 0
        while i + 8 <= count:
            var v = src.unsafe_offset(i).unsafe_load[width=8, alignment=1]()
            var w = v.cast[DType.uint32]() << 16
            out.unsafe_offset(i).unsafe_store[alignment=1](bitcast[DType.float32, 8](w))
            i += 8
        while i < count:
            out.unsafe_offset(i).unsafe_store(bf16_to_f32(src.unsafe_offset(i).unsafe_load[alignment=1]()))
            i += 1
        return
    if t.dtype == "F16":
        for i in range(count):
            out.unsafe_offset(i).unsafe_store(f16_to_f32(src.unsafe_offset(i).unsafe_load[alignment=1]()))
        return
    raise Error("safetensors: unsupported dtype " + t.dtype + " for " + t.name)


struct SafeTensors(Movable):
    var tensors: List[SafeTensor]
    var index: Dict[String, Int]
    var maps: List[Int]
    var sizes: List[Int]

    def __init__(out self):
        self.tensors = List[SafeTensor]()
        self.index = Dict[String, Int]()
        self.maps = List[Int]()
        self.sizes = List[Int]()

    def __deinit__(deinit self):
        for i in range(len(self.maps)):
            _ = external_call["munmap", Int32](self.maps[i], Int64(self.sizes[i]))

    @staticmethod
    def open_dir(dir: String) raises -> SafeTensors:
        var st = SafeTensors()
        var names = list_directory(dir)
        for n in names:
            if n.endswith(".safetensors"):
                st.add_file(dir + "/" + n)
        if len(st.tensors) == 0:
            raise Error("safetensors: no *.safetensors in " + dir)
        return st^

    def add_file(mut self, path: String) raises:
        var head = read_at(path, 0, 8)
        if len(head) != 8:
            raise Error("safetensors: cannot read " + path)
        var hlen = 0
        for i in range(8):
            hlen |= Int(head[i]) << (8 * i)
        var hdr = read_at(path, 8, hlen)
        if len(hdr) != hlen:
            raise Error("safetensors: short header in " + path)
        while len(hdr) > 0 and hdr[len(hdr) - 1] == 32:
            _ = hdr.pop()
        var doc = parse_json(hdr)
        if not doc.valid:
            raise Error("safetensors: bad JSON header in " + path)
        var fd = open_read(path)
        if fd < 0:
            raise Error("safetensors: cannot open " + path)
        var size = Int(external_call["lseek", Int64](Int32(fd), Int64(0), Int32(2)))
        var addr = external_call["mmap", Int](Int(0), Int64(size), Int32(1), Int32(2), Int32(fd), Int64(0))
        close_fd(fd)
        if addr == 0 or addr == -1:
            raise Error("safetensors: cannot map " + path)
        self.maps.append(addr)
        self.sizes.append(size)
        var base = addr + 8 + hlen
        var root = doc.root
        for k in range(len(doc.nodes[root].keys)):
            var name = doc.nodes[root].keys[k]
            if name == "__metadata__":
                continue
            var node = doc.nodes[root].children[k]
            var dt = doc.text(doc.object_get(node, "dtype"))
            var sh = doc.object_get(node, "shape")
            var shape = List[Int]()
            for i in range(doc.array_size(sh)):
                shape.append(Int(doc.number(doc.at(sh, i))))
            var offs = doc.object_get(node, "data_offsets")
            var b0 = Int(doc.number(doc.at(offs, 0)))
            self.index[name] = len(self.tensors)
            self.tensors.append(SafeTensor(name, dt, shape^, base + b0))

    def has(self, name: String) -> Bool:
        return name in self.index

    def get(self, name: String) raises -> SafeTensor:
        var i = self.index.get(name)
        if not i:
            raise Error("safetensors: missing tensor " + name)
        return self.tensors[i.value()].copy()

    def to_f32(self, name: String) raises -> Int:
        """A new calloc'd float32 copy of tensor `name` (the caller frees it)."""
        var t = self.get(name)
        var n = t.numel()
        var p = external_call["malloc", Int](n * 4)
        widen_to_f32(t, p, 0, n)
        return p
