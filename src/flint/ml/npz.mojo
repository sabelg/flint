"""Memory-mapped NumPy `.npz` archives (`flint.ml.npz`).

An `.npz` is a zip of `.npy` files. `Npz.open(path)` maps the archive once and indexes its
central directory (zip64 included); members must be stored uncompressed (`np.savez`, not
`np.savez_compressed`). Each `.npy` member is a magic + version, a Python-dict header with
`descr`, `fortran_order` and `shape`, then raw C-order data.

  - `floats(name)`  `<f4` as a `Tensor` borrowing the mapping (zero copy when 4-byte aligned);
                    `<f8` / `<f2` / integer arrays are converted to owned floats
  - `ints(name)`    integer (or bool) arrays as `List[Int]`
  - `strings(name)` fixed-width byte strings (`|S<n>`), NUL padding stripped
  - `shape(name)`, `has(name)`, `names`
"""

from std.ffi import external_call
from std.collections import Dict
from flint.platform.os import open_read, close_fd
from std.memory import bitcast
from flint.ml.tensor import Tensor, f32_from_bits


struct NpyEntry(Copyable, Movable):
    var descr: String
    var shape: List[Int]
    var data: Int  # absolute address of the first element
    var fortran: Bool

    def __init__(out self, descr: String, var shape: List[Int], data: Int, fortran: Bool):
        self.descr = descr
        self.shape = shape^
        self.data = data
        self.fortran = fortran


def _rd(addr: Int, n: Int) -> Int:
    """Little-endian unsigned integer of n bytes at addr."""
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr)
    var v = 0
    for i in range(n):
        v |= Int(p.unsafe_offset(i).unsafe_load()) << (8 * i)
    return v


def _bytes_str(addr: Int, n: Int) -> String:
    var b = List[UInt8](length=n, fill=0)
    _ = external_call["memcpy", Int](Int(b.unsafe_ptr()), addr, n)
    return String(unsafe_from_utf8=b)


def _dict_value(header: String, key: String) -> String:
    """The raw text of `'key': value` in a .npy header dict (up to the next top-level comma)."""
    var k = "'" + key + "'"
    var at = header.find(k)
    if at < 0:
        return String()
    var i = at + k.byte_length()
    var n = header.byte_length()
    while i < n and (header[byte=i] == ":" or header[byte=i] == " "):
        i += 1
    var depth = 0
    var out = String()
    while i < n:
        var c = header[byte=i]
        if c == "(" or c == "[":
            depth += 1
        elif c == ")" or c == "]":
            depth -= 1
            if depth < 0:
                break
        elif (c == "," or c == "}") and depth == 0:
            break
        out += c
        i += 1
    return String(out.strip())


def parse_npy_header(header: String) raises -> Tuple[String, List[Int], Bool]:
    """(descr, shape, fortran_order) of a .npy header dict."""
    var descr = _dict_value(header, "descr").replace("'", "").replace('"', "")
    var fortran = _dict_value(header, "fortran_order") == "True"
    var sh = _dict_value(header, "shape").replace("(", "").replace(")", "")
    var shape = List[Int]()
    for part in sh.split(","):
        var t = String(part.strip())
        if t.byte_length() > 0:
            shape.append(Int(t))
    return (descr, shape^, fortran)


struct Npz(Movable):
    var path: String
    var map_addr: Int
    var map_size: Int
    var entries: Dict[String, NpyEntry]
    var names: List[String]

    def __init__(out self):
        self.path = String()
        self.map_addr = 0
        self.map_size = 0
        self.entries = Dict[String, NpyEntry]()
        self.names = List[String]()

    def __deinit__(deinit self):
        if self.map_addr != 0:
            _ = external_call["munmap", Int32](self.map_addr, Int64(self.map_size))

    @staticmethod
    def open(path: String) raises -> Npz:
        var z = Npz()
        z.path = path
        var fd = open_read(path)
        if fd < 0:
            raise Error("npz: cannot open " + path)
        var size = Int(external_call["lseek", Int64](Int32(fd), Int64(0), Int32(2)))
        var addr = external_call["mmap", Int](Int(0), Int64(size), Int32(1), Int32(2), Int32(fd), Int64(0))
        close_fd(fd)
        if addr == 0 or addr == -1 or size < 22:
            raise Error("npz: cannot map " + path)
        z.map_addr = addr
        z.map_size = size
        # end of central directory: scan back over a possible comment
        var eocd = -1
        var i = size - 22
        var stop = max(0, size - 22 - 65535)
        while i >= stop:
            if _rd(addr + i, 4) == 0x06054B50:
                eocd = i
                break
            i -= 1
        if eocd < 0:
            raise Error("npz: no zip end record in " + path)
        var count = _rd(addr + eocd + 10, 2)
        var cd = _rd(addr + eocd + 16, 4)
        if count == 0xFFFF or cd == 0xFFFFFFFF:
            # zip64 end-of-central-directory locator sits right before the EOCD
            var loc = eocd - 20
            if loc < 0 or _rd(addr + loc, 4) != 0x07064B50:
                raise Error("npz: missing zip64 locator in " + path)
            var e64 = _rd(addr + loc + 8, 8)
            if _rd(addr + e64, 4) != 0x06064B50:
                raise Error("npz: bad zip64 end record in " + path)
            count = _rd(addr + e64 + 32, 8)
            cd = _rd(addr + e64 + 48, 8)
        var p = cd
        for _ in range(count):
            if p + 46 > size or _rd(addr + p, 4) != 0x02014B50:
                raise Error("npz: bad central directory in " + path)
            var method = _rd(addr + p + 10, 2)
            var csize = _rd(addr + p + 20, 4)
            var usize = _rd(addr + p + 24, 4)
            var nlen = _rd(addr + p + 28, 2)
            var elen = _rd(addr + p + 30, 2)
            var clen = _rd(addr + p + 32, 2)
            var lho = _rd(addr + p + 42, 4)
            var name = _bytes_str(addr + p + 46, nlen)
            # zip64 extra field (id 1): the 0xFFFFFFFF fields, in order usize, csize, offset
            var e = p + 46 + nlen
            var eend = e + elen
            while e + 4 <= eend:
                var id = _rd(addr + e, 2)
                var sz = _rd(addr + e + 2, 2)
                if id == 1:
                    var q = e + 4
                    if usize == 0xFFFFFFFF:
                        usize = _rd(addr + q, 8)
                        q += 8
                    if csize == 0xFFFFFFFF:
                        csize = _rd(addr + q, 8)
                        q += 8
                    if lho == 0xFFFFFFFF:
                        lho = _rd(addr + q, 8)
                e += 4 + sz
            p = eend + clen
            if method != 0:
                raise Error("npz: member " + name + " is compressed (use np.savez, not savez_compressed)")
            if _rd(addr + lho, 4) != 0x04034B50:
                raise Error("npz: bad local header for " + name)
            var start = lho + 30 + _rd(addr + lho + 26, 2) + _rd(addr + lho + 28, 2)
            if start + usize > size:
                raise Error("npz: member " + name + " lies outside " + path)
            # .npy member
            if usize < 10 or _rd(addr + start, 1) != 0x93 or _bytes_str(addr + start + 1, 5) != "NUMPY":
                raise Error("npz: member " + name + " is not a .npy file")
            var major = _rd(addr + start + 6, 1)
            var hlen: Int
            var hstart: Int
            if major == 1:
                hlen = _rd(addr + start + 8, 2)
                hstart = start + 10
            else:
                hlen = _rd(addr + start + 8, 4)
                hstart = start + 12
            var parsed = parse_npy_header(_bytes_str(addr + hstart, hlen))
            var key = name
            if key.endswith(".npy"):
                var stem = String(name[byte=0 : name.byte_length() - 4])
                key = stem
            z.entries[key] = NpyEntry(parsed[0], parsed[1].copy(), addr + hstart + hlen, parsed[2])
            z.names.append(key)
        return z^

    def has(self, name: String) -> Bool:
        return name in self.entries

    def _entry(self, name: String) raises -> NpyEntry:
        var e = self.entries.get(name)
        if not e:
            raise Error("npz: no array " + name + " in " + self.path)
        if e.value().fortran:
            raise Error("npz: " + name + " is Fortran-ordered")
        return e.value().copy()

    def shape(self, name: String) raises -> List[Int]:
        return self._entry(name).shape.copy()

    def floats(self, name: String) raises -> Tensor:
        """`name` as float32; borrowed from the mapping for aligned `<f4`, converted otherwise."""
        var e = self._entry(name)
        var n = 1
        for d in e.shape:
            n *= d
        if e.descr == "<f4" and e.data % 4 == 0:
            return Tensor.borrowed(e.shape.copy(), e.data)
        var out = List[Float32](length=n, fill=0.0)
        if e.descr == "<f4":
            _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), e.data, n * 4)
        elif e.descr == "<f8":
            for i in range(n):
                out[i] = Float32(bitcast[DType.float64, 1](SIMD[DType.uint64, 1](UInt64(_rd(e.data + 8 * i, 8))))[0])
        elif e.descr == "<f2":
            for i in range(n):
                out[i] = _half(UInt32(_rd(e.data + 2 * i, 2)))
        else:
            var v = self.ints(name)
            for i in range(n):
                out[i] = Float32(v[i])
        return Tensor.floats(e.shape.copy(), out^)

    def ints(self, name: String) raises -> List[Int]:
        var e = self._entry(name)
        var n = 1
        for d in e.shape:
            n *= d
        var w: Int
        var signed = True
        var d = e.descr
        if d == "<i4": w = 4
        elif d == "<i8": w = 8
        elif d == "<i2": w = 2
        elif d == "|i1": w = 1
        elif d == "<u4":
            w = 4; signed = False
        elif d == "<u8":
            w = 8; signed = False
        elif d == "<u2":
            w = 2; signed = False
        elif d == "|u1" or d == "|b1":
            w = 1; signed = False
        else:
            raise Error("npz: " + name + " has non-integer dtype " + d)
        var out = List[Int](length=n, fill=0)
        for i in range(n):
            var v = _rd(e.data + w * i, w)
            if signed and w < 8 and (v >> (8 * w - 1)) & 1 == 1:
                v -= 1 << (8 * w)
            out[i] = v
        return out^

    def strings(self, name: String) raises -> List[String]:
        var e = self._entry(name)
        if not e.descr.startswith("|S"):
            raise Error("npz: " + name + " is not a byte-string array (" + e.descr + ")")
        var w = Int(String(e.descr[byte=2 : e.descr.byte_length()]))
        var n = 1
        for d in e.shape:
            n *= d
        var out = List[String]()
        for i in range(n):
            var b = List[UInt8]()
            for j in range(w):
                var c = UInt8(_rd(e.data + i * w + j, 1))
                if c == 0:
                    break
                b.append(c)
            out.append(String(unsafe_from_utf8=b))
        return out^


def _half(h: UInt32) -> Float32:
    var sign = (h >> 15) & 1
    var expo = (h >> 10) & 0x1F
    var mant = h & 0x3FF
    var bits: UInt32
    if expo == 0:
        if mant == 0:
            bits = sign << 31
        else:
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
