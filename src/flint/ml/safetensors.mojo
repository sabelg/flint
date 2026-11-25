"""Memory-mapped `.safetensors` weights (`flint.ml.safetensors`).

The format is an 8-byte little-endian header length, a JSON header mapping each tensor name to
`{"dtype", "shape", "data_offsets": [begin, end]}`, then the raw little-endian tensor bytes.
`SafeTensors.open(path)` maps the file once; `weight(name)` returns a float32 `Tensor` that
borrows F32 data straight from the mapping (zero copy) and converts F16 / BF16 to owned floats.
"""

from std.ffi import external_call
from std.collections import Dict
from flint.assets.json import parse_json
from flint.platform.os import open_read, close_fd
from flint.ml.tensor import Tensor, f32_from_bits


struct SafeEntry(Copyable, Movable):
    var dtype: String
    var shape: List[Int]
    var begin: Int
    var end: Int

    def __init__(out self, dtype: String, var shape: List[Int], begin: Int, end: Int):
        self.dtype = dtype
        self.shape = shape^
        self.begin = begin
        self.end = end


def _f16_to_f32(h: UInt32) -> Float32:
    var sign = (h >> 15) & 1
    var expo = (h >> 10) & 0x1F
    var mant = h & 0x3FF
    var bits: UInt32
    if expo == 0:
        if mant == 0:
            bits = sign << 31
        else:
            # subnormal: renormalise
            var e: UInt32 = 127 - 15 + 1
            while (mant & 0x400) == 0:
                mant <<= 1
                e -= 1
            mant &= 0x3FF
            bits = (sign << 31) | (e << 23) | (mant << 13)
    elif expo == 0x1F:
        bits = (sign << 31) | (UInt32(0xFF) << 23) | (mant << 13)
    else:
        bits = (sign << 31) | ((expo + 127 - 15) << 23) | (mant << 13)
    return f32_from_bits(bits)


struct SafeTensors(Movable):
    var path: String
    var map_addr: Int
    var map_size: Int
    var data_start: Int
    var entries: Dict[String, SafeEntry]
    var names: List[String]

    def __init__(out self):
        self.path = String()
        self.map_addr = 0
        self.map_size = 0
        self.data_start = 0
        self.entries = Dict[String, SafeEntry]()
        self.names = List[String]()

    def __deinit__(deinit self):
        if self.map_addr != 0:
            _ = external_call["munmap", Int32](self.map_addr, Int64(self.map_size))

    @staticmethod
    def open(path: String) raises -> SafeTensors:
        var st = SafeTensors()
        st.path = path
        var fd = open_read(path)
        if fd < 0:
            raise Error("safetensors: cannot open " + path)
        var size = Int(external_call["lseek", Int64](Int32(fd), Int64(0), Int32(2)))
        # PROT_READ = 1, MAP_PRIVATE = 2
        var addr = external_call["mmap", Int](Int(0), Int64(size), Int32(1), Int32(2), Int32(fd), Int64(0))
        close_fd(fd)
        if addr == 0 or addr == -1 or size < 8:
            raise Error("safetensors: cannot map " + path)
        st.map_addr = addr
        st.map_size = size
        var pb = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr)
        var hlen = 0
        for i in range(8):
            hlen |= Int(pb.unsafe_offset(i).unsafe_load()) << (8 * i)
        if hlen <= 0 or 8 + hlen > size:
            raise Error("safetensors: bad header length in " + path)
        var header = List[UInt8](length=hlen, fill=0)
        _ = external_call["memcpy", Int](Int(header.unsafe_ptr()), addr + 8, hlen)
        st.data_start = addr + 8 + hlen
        var doc = parse_json(header)
        if not doc.valid:
            raise Error("safetensors: header is not valid JSON in " + path)
        var root = doc.root
        var keys = doc.nodes[root].keys.copy()
        for k in range(len(keys)):
            var name = keys[k]
            if name == "__metadata__":
                continue
            var node = doc.nodes[root].children[k]
            var dtype = doc.text(doc.object_get(node, "dtype"))
            var shape_node = doc.object_get(node, "shape")
            var shape = List[Int]()
            for i in range(doc.array_size(shape_node)):
                shape.append(doc.integer(doc.at(shape_node, i), 0))
            var offs = doc.object_get(node, "data_offsets")
            var b = doc.integer(doc.at(offs, 0), 0)
            var e = doc.integer(doc.at(offs, 1), 0)
            if st.data_start + e > addr + size or b > e:
                raise Error("safetensors: tensor " + name + " lies outside " + path)
            st.entries[name] = SafeEntry(dtype, shape^, b, e)
            st.names.append(name)
        return st^

    def has(self, name: String) -> Bool:
        return name in self.entries

    def shape(self, name: String) raises -> List[Int]:
        var e = self.entries.get(name)
        if not e:
            raise Error("safetensors: no tensor " + name + " in " + self.path)
        return e.value().shape.copy()

    def weight(self, name: String) raises -> Tensor:
        """`name` as float32: borrowed from the mapping for F32, converted for F16 / BF16."""
        var found = self.entries.get(name)
        if not found:
            raise Error("safetensors: no tensor " + name + " in " + self.path)
        var e = found.value().copy()
        var addr = self.data_start + e.begin
        var n = 1
        for d in e.shape:
            n *= d
        if e.dtype == "F32":
            return Tensor.borrowed(e.shape.copy(), addr)
        var out = List[Float32](length=n, fill=0.0)
        var ph = Pointer[UInt16, ImmutAnyOrigin](unsafe_from_address=addr)
        if e.dtype == "BF16":
            for i in range(n):
                out[i] = f32_from_bits(UInt32(ph.unsafe_offset(i).unsafe_load[alignment=1]()) << 16)
        elif e.dtype == "F16":
            for i in range(n):
                out[i] = _f16_to_f32(UInt32(ph.unsafe_offset(i).unsafe_load[alignment=1]()))
        else:
            raise Error("safetensors: unsupported dtype " + e.dtype + " for " + name)
        return Tensor.floats(e.shape.copy(), out^)

    def scalar(self, name: String) raises -> Float32:
        var t = self.weight(name)
        return t.fget(0)
