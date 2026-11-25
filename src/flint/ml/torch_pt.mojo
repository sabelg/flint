"""PyTorch `torch.save` checkpoints read in place (`flint.ml.torch_pt`).

A `.pt` / `.pth` file written by `torch.save` (zip format, PyTorch >= 1.6) is an uncompressed
zip holding `<archive>/data.pkl` (a protocol-2 pickle of the saved object) and one member per
tensor storage, `<archive>/data/<key>`. `TorchPt.open(path)` maps the file once, runs a small
pickle VM over `data.pkl` and resolves every tensor to an address inside the mapping, so the
weights are read with no copy and no Python.

The VM understands what `torch.save` emits for nested dicts / lists / tuples of numbers, strings
and tensors: dicts, `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` /
`_rebuild_parameter`, persistent-id storages (`FloatStorage`, `HalfStorage`, `BFloat16Storage`,
...). Anything else becomes an opaque value. No code from the file is ever executed.

  - `root`: the unpickled object; `get(node, key)`, `at(node, i)`, `size(node)`, `text(node)`,
    `integer(node)`, `number(node)`, `kind(node)` walk it (`KIND_*`)
  - `tensors(node, prefix)`: every tensor below a dict node, flattened to dotted names
    (`model.encoder.weight`); `PtTensor` has the dtype, shape, strides and data address
"""

from std.ffi import external_call
from std.collections import Dict
from std.memory import bitcast
from flint.platform.os import open_read, close_fd

comptime KIND_NONE: Int = 0
comptime KIND_BOOL: Int = 1
comptime KIND_INT: Int = 2
comptime KIND_FLOAT: Int = 3
comptime KIND_STR: Int = 4
comptime KIND_LIST: Int = 5
comptime KIND_TUPLE: Int = 6
comptime KIND_DICT: Int = 7
comptime KIND_GLOBAL: Int = 8
comptime KIND_TENSOR: Int = 9
comptime KIND_STORAGE: Int = 10
comptime KIND_OBJECT: Int = 11


struct PkVal(Copyable, Movable):
    var kind: Int
    var i: Int
    var f: Float64
    var s: String          # str value, global "module.name", storage key / dtype
    var s2: String
    var items: List[Int]   # list / tuple / dict values; tensor: [storage, offset, size tuple, stride tuple]
    var keys: List[Int]

    def __init__(out self, kind: Int):
        self.kind = kind
        self.i = 0
        self.f = 0.0
        self.s = String()
        self.s2 = String()
        self.items = List[Int]()
        self.keys = List[Int]()


struct PtTensor(Copyable, Movable):
    var dtype: String       # "F32", "F16", "BF16", "F64", "I64", "I32", "U8", "BOOL"
    var shape: List[Int]
    var strides: List[Int]
    var addr: Int           # address of element 0 (storage start + offset * itemsize)
    var contiguous: Bool

    def __init__(out self):
        self.dtype = String()
        self.shape = List[Int]()
        self.strides = List[Int]()
        self.addr = 0
        self.contiguous = True


def _rd(addr: Int, n: Int) -> Int:
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr)
    var v = 0
    for i in range(n):
        v |= Int(p.unsafe_offset(i).unsafe_load()) << (8 * i)
    return v


def _str_at(addr: Int, n: Int) -> String:
    var b = List[UInt8](length=n, fill=0)
    _ = external_call["memcpy", Int](Int(b.unsafe_ptr()), addr, n)
    return String(unsafe_from_utf8=b)


def _storage_dtype(name: String) -> String:
    if name.endswith("FloatStorage"):
        return "F32"
    if name.endswith("HalfStorage"):
        return "F16"
    if name.endswith("BFloat16Storage"):
        return "BF16"
    if name.endswith("DoubleStorage"):
        return "F64"
    if name.endswith("LongStorage"):
        return "I64"
    if name.endswith("IntStorage"):
        return "I32"
    if name.endswith("ByteStorage"):
        return "U8"
    if name.endswith("BoolStorage"):
        return "BOOL"
    return "?"


def dtype_size(dtype: String) -> Int:
    if dtype == "F64" or dtype == "I64":
        return 8
    if dtype == "F32" or dtype == "I32":
        return 4
    if dtype == "F16" or dtype == "BF16":
        return 2
    return 1


struct TorchPt(Movable):
    var path: String
    var map_addr: Int
    var map_size: Int
    var members: Dict[String, Int]     # zip member name -> data address
    var member_size: Dict[String, Int]
    var vals: List[PkVal]
    var root: Int

    def __init__(out self):
        self.path = String()
        self.map_addr = 0
        self.map_size = 0
        self.members = Dict[String, Int]()
        self.member_size = Dict[String, Int]()
        self.vals = List[PkVal]()
        self.root = -1

    def __deinit__(deinit self):
        if self.map_addr != 0:
            _ = external_call["munmap", Int32](self.map_addr, Int64(self.map_size))

    @staticmethod
    def open(path: String) raises -> TorchPt:
        var t = TorchPt()
        t.path = path
        var fd = open_read(path)
        if fd < 0:
            raise Error("torch_pt: cannot open " + path)
        var size = Int(external_call["lseek", Int64](Int32(fd), Int64(0), Int32(2)))
        var addr = external_call["mmap", Int](Int(0), Int64(size), Int32(1), Int32(2), Int32(fd), Int64(0))
        close_fd(fd)
        if addr == 0 or addr == -1 or size < 22:
            raise Error("torch_pt: cannot map " + path)
        t.map_addr = addr
        t.map_size = size
        t._index_zip()
        var pkl = String()
        var prefix = String()
        for e in t.members.items():
            if e.key.endswith("/data.pkl") or e.key == "data.pkl":
                pkl = e.key
                prefix = String(e.key[byte=0 : e.key.byte_length() - 8])
        if pkl == "":
            raise Error("torch_pt: no data.pkl in " + path + " (legacy non-zip checkpoints are not supported)")
        t.root = t._unpickle(t.members[pkl], t.member_size[pkl], prefix)
        return t^

    def _index_zip(mut self) raises:
        var addr = self.map_addr
        var size = self.map_size
        var eocd = -1
        var i = size - 22
        var stop = max(0, size - 22 - 65535)
        while i >= stop:
            if _rd(addr + i, 4) == 0x06054B50:
                eocd = i
                break
            i -= 1
        if eocd < 0:
            raise Error("torch_pt: no zip end record in " + self.path)
        var count = _rd(addr + eocd + 10, 2)
        var cd = _rd(addr + eocd + 16, 4)
        if count == 0xFFFF or cd == 0xFFFFFFFF:
            var loc = eocd - 20
            if loc < 0 or _rd(addr + loc, 4) != 0x07064B50:
                raise Error("torch_pt: missing zip64 locator in " + self.path)
            var e64 = _rd(addr + loc + 8, 8)
            count = _rd(addr + e64 + 32, 8)
            cd = _rd(addr + e64 + 48, 8)
        var p = cd
        for _ in range(count):
            if p + 46 > size or _rd(addr + p, 4) != 0x02014B50:
                raise Error("torch_pt: bad central directory in " + self.path)
            var method = _rd(addr + p + 10, 2)
            var csize = _rd(addr + p + 20, 4)
            var usize = _rd(addr + p + 24, 4)
            var nlen = _rd(addr + p + 28, 2)
            var elen = _rd(addr + p + 30, 2)
            var clen = _rd(addr + p + 32, 2)
            var lho = _rd(addr + p + 42, 4)
            var name = _str_at(addr + p + 46, nlen)
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
                raise Error("torch_pt: member " + name + " is compressed")
            if _rd(addr + lho, 4) != 0x04034B50:
                raise Error("torch_pt: bad local header for " + name)
            var start = lho + 30 + _rd(addr + lho + 26, 2) + _rd(addr + lho + 28, 2)
            if start + usize > size:
                raise Error("torch_pt: member " + name + " lies outside " + self.path)
            self.members[name] = addr + start
            self.member_size[name] = usize

    # ------------------------------------------------------------------ pickle VM

    def _new(mut self, var v: PkVal) -> Int:
        self.vals.append(v^)
        return len(self.vals) - 1

    def _unpickle(mut self, base: Int, n: Int, prefix: String) raises -> Int:
        var stack = List[Int]()
        var marks = List[Int]()
        var memo = Dict[Int, Int]()
        var p = 0
        while p < n:
            var op = _rd(base + p, 1)
            p += 1
            if op == 0x80:      # PROTO
                p += 1
            elif op == 0x95:    # FRAME
                p += 8
            elif op == 0x7D:    # EMPTY_DICT
                stack.append(self._new(PkVal(KIND_DICT)))
            elif op == 0x5D:    # EMPTY_LIST
                stack.append(self._new(PkVal(KIND_LIST)))
            elif op == 0x29:    # EMPTY_TUPLE
                stack.append(self._new(PkVal(KIND_TUPLE)))
            elif op == 0x71:    # BINPUT
                memo[_rd(base + p, 1)] = stack[len(stack) - 1]
                p += 1
            elif op == 0x72:    # LONG_BINPUT
                memo[_rd(base + p, 4)] = stack[len(stack) - 1]
                p += 4
            elif op == 0x94:    # MEMOIZE
                memo[len(memo)] = stack[len(stack) - 1]
            elif op == 0x68:    # BINGET
                stack.append(memo[_rd(base + p, 1)])
                p += 1
            elif op == 0x6A:    # LONG_BINGET
                stack.append(memo[_rd(base + p, 4)])
                p += 4
            elif op == 0x28:    # MARK
                marks.append(len(stack))
            elif op == 0x58 or op == 0x8C or op == 0x8D:   # BINUNICODE / SHORT_BINUNICODE / BINUNICODE8
                var ln: Int
                if op == 0x58:
                    ln = _rd(base + p, 4)
                    p += 4
                elif op == 0x8C:
                    ln = _rd(base + p, 1)
                    p += 1
                else:
                    ln = _rd(base + p, 8)
                    p += 8
                var v = PkVal(KIND_STR)
                v.s = _str_at(base + p, ln)
                p += ln
                stack.append(self._new(v^))
            elif op == 0x4A:    # BININT (signed 4)
                var v = PkVal(KIND_INT)
                var x = _rd(base + p, 4)
                if x >= 0x80000000:
                    x -= 0x100000000
                v.i = x
                p += 4
                stack.append(self._new(v^))
            elif op == 0x4B:    # BININT1
                var v = PkVal(KIND_INT)
                v.i = _rd(base + p, 1)
                p += 1
                stack.append(self._new(v^))
            elif op == 0x4D:    # BININT2
                var v = PkVal(KIND_INT)
                v.i = _rd(base + p, 2)
                p += 2
                stack.append(self._new(v^))
            elif op == 0x8A:    # LONG1
                var ln = _rd(base + p, 1)
                p += 1
                var x = 0
                for k in range(min(ln, 8)):
                    x |= _rd(base + p + k, 1) << (8 * k)
                if ln > 0 and ln < 8 and (_rd(base + p + ln - 1, 1) & 0x80) != 0:
                    x -= 1 << (8 * ln)
                p += ln
                var v = PkVal(KIND_INT)
                v.i = x
                stack.append(self._new(v^))
            elif op == 0x47:    # BINFLOAT (big-endian double)
                var bits: UInt64 = 0
                for k in range(8):
                    bits = (bits << 8) | UInt64(_rd(base + p + k, 1))
                p += 8
                var v = PkVal(KIND_FLOAT)
                v.f = _f64_from_bits(bits)
                stack.append(self._new(v^))
            elif op == 0x4E:    # NONE
                stack.append(self._new(PkVal(KIND_NONE)))
            elif op == 0x88 or op == 0x89:   # NEWTRUE / NEWFALSE
                var v = PkVal(KIND_BOOL)
                v.i = 1 if op == 0x88 else 0
                stack.append(self._new(v^))
            elif op == 0x61:    # APPEND
                var x = stack.pop()
                self.vals[stack[len(stack) - 1]].items.append(x)
            elif op == 0x65:    # APPENDS
                var m = marks.pop()
                var lst = stack[m - 1]
                for k in range(m, len(stack)):
                    self.vals[lst].items.append(stack[k])
                while len(stack) > m:
                    _ = stack.pop()
            elif op == 0x73:    # SETITEM
                var v = stack.pop()
                var k = stack.pop()
                var d = stack[len(stack) - 1]
                self.vals[d].keys.append(k)
                self.vals[d].items.append(v)
            elif op == 0x75:    # SETITEMS
                var m = marks.pop()
                var d = stack[m - 1]
                var k = m
                while k + 1 < len(stack):
                    self.vals[d].keys.append(stack[k])
                    self.vals[d].items.append(stack[k + 1])
                    k += 2
                while len(stack) > m:
                    _ = stack.pop()
            elif op == 0x74:    # TUPLE
                var m = marks.pop()
                var v = PkVal(KIND_TUPLE)
                for k in range(m, len(stack)):
                    v.items.append(stack[k])
                while len(stack) > m:
                    _ = stack.pop()
                stack.append(self._new(v^))
            elif op == 0x85 or op == 0x86 or op == 0x87:   # TUPLE1..3
                var cnt = op - 0x84
                var v = PkVal(KIND_TUPLE)
                var s0 = len(stack) - cnt
                for k in range(s0, len(stack)):
                    v.items.append(stack[k])
                while len(stack) > s0:
                    _ = stack.pop()
                stack.append(self._new(v^))
            elif op == 0x63:    # GLOBAL "module\nname\n"
                var e1 = p
                while _rd(base + e1, 1) != 0x0A:
                    e1 += 1
                var e2 = e1 + 1
                while _rd(base + e2, 1) != 0x0A:
                    e2 += 1
                var v = PkVal(KIND_GLOBAL)
                v.s = _str_at(base + p, e1 - p) + "." + _str_at(base + e1 + 1, e2 - e1 - 1)
                p = e2 + 1
                stack.append(self._new(v^))
            elif op == 0x93:    # STACK_GLOBAL
                var nm = stack.pop()
                var md = stack.pop()
                var v = PkVal(KIND_GLOBAL)
                v.s = self.vals[md].s + "." + self.vals[nm].s
                stack.append(self._new(v^))
            elif op == 0x51:    # BINPERSID
                var pid = stack.pop()
                var v = PkVal(KIND_STORAGE)
                ref t = self.vals[pid]
                if t.kind == KIND_TUPLE and len(t.items) >= 3:
                    v.s2 = _storage_dtype(self.vals[t.items[1]].s)
                    v.s = self.vals[t.items[2]].s
                    var key = prefix + "data/" + v.s
                    if key not in self.members:
                        raise Error("torch_pt: missing storage " + key)
                    v.i = self.members[key]
                stack.append(self._new(v^))
            elif op == 0x52:    # REDUCE
                var args = stack.pop()
                var callee = stack.pop()
                stack.append(self._reduce(callee, args))
            elif op == 0x62:    # BUILD: state is dropped (OrderedDict / tensor setstate)
                _ = stack.pop()
            elif op == 0x2E:    # STOP
                break
            else:
                raise Error("torch_pt: unsupported pickle opcode " + String(op) + " at " + String(p - 1))
        if len(stack) == 0:
            raise Error("torch_pt: empty pickle in " + self.path)
        return stack[len(stack) - 1]

    def _reduce(mut self, callee: Int, args: Int) raises -> Int:
        var name = self.vals[callee].s
        if name == "collections.OrderedDict":
            return self._new(PkVal(KIND_DICT))
        if name == "torch._utils._rebuild_parameter" or name == "torch._utils._rebuild_parameter_with_state":
            return self.vals[args].items[0]
        if name == "torch._utils._rebuild_tensor_v2" or name == "torch._utils._rebuild_tensor":
            ref a = self.vals[args]
            var v = PkVal(KIND_TENSOR)
            for k in range(min(4, len(a.items))):
                v.items.append(a.items[k])
            return self._new(v^)
        var o = PkVal(KIND_OBJECT)
        o.s = name
        return self._new(o^)

    # ------------------------------------------------------------------ navigation

    def kind(self, node: Int) -> Int:
        if node < 0:
            return KIND_NONE
        return self.vals[node].kind

    def size(self, node: Int) -> Int:
        if node < 0:
            return 0
        return len(self.vals[node].items)

    def at(self, node: Int, i: Int) -> Int:
        if node < 0 or i >= len(self.vals[node].items):
            return -1
        return self.vals[node].items[i]

    def key_at(self, node: Int, i: Int) -> String:
        if node < 0 or i >= len(self.vals[node].keys):
            return String()
        return self.vals[self.vals[node].keys[i]].s

    def get(self, node: Int, key: String) -> Int:
        """Value of `key` in a dict node, -1 when absent."""
        if node < 0 or self.vals[node].kind != KIND_DICT:
            return -1
        ref v = self.vals[node]
        for i in range(len(v.keys)):
            if self.vals[v.keys[i]].s == key:
                return v.items[i]
        return -1

    def text(self, node: Int) -> String:
        if node < 0:
            return String()
        return self.vals[node].s

    def integer(self, node: Int, default: Int = 0) -> Int:
        if node < 0:
            return default
        var k = self.vals[node].kind
        if k == KIND_INT or k == KIND_BOOL:
            return self.vals[node].i
        if k == KIND_FLOAT:
            return Int(self.vals[node].f)
        return default

    def number(self, node: Int, default: Float64 = 0.0) -> Float64:
        if node < 0:
            return default
        var k = self.vals[node].kind
        if k == KIND_FLOAT:
            return self.vals[node].f
        if k == KIND_INT or k == KIND_BOOL:
            return Float64(self.vals[node].i)
        return default

    def tensor(self, node: Int) raises -> PtTensor:
        if self.kind(node) != KIND_TENSOR:
            raise Error("torch_pt: not a tensor")
        ref v = self.vals[node]
        ref st = self.vals[v.items[0]]
        var t = PtTensor()
        t.dtype = st.s2
        var off = self.integer(v.items[1])
        for k in range(self.size(v.items[2])):
            t.shape.append(self.integer(self.at(v.items[2], k)))
        for k in range(self.size(v.items[3])):
            t.strides.append(self.integer(self.at(v.items[3], k)))
        t.addr = st.i + off * dtype_size(t.dtype)
        var expect = 1
        var r = len(t.shape) - 1
        while r >= 0:
            if t.shape[r] != 1 and t.strides[r] != expect:
                t.contiguous = False
            expect *= t.shape[r]
            r -= 1
        return t^

    def tensors(self, node: Int, prefix: String = "") raises -> Dict[String, PtTensor]:
        """All tensors below a dict / list node, by dotted name."""
        var out = Dict[String, PtTensor]()
        self._collect(node, prefix, out)
        return out^

    def _collect(self, node: Int, prefix: String, mut out: Dict[String, PtTensor]) raises:
        var k = self.kind(node)
        if k == KIND_TENSOR:
            out[prefix] = self.tensor(node)
        elif k == KIND_DICT:
            for i in range(self.size(node)):
                var name = self.key_at(node, i)
                self._collect(self.at(node, i), name if prefix == "" else prefix + "." + name, out)
        elif k == KIND_LIST or k == KIND_TUPLE:
            for i in range(self.size(node)):
                self._collect(self.at(node, i), String(i) if prefix == "" else prefix + "." + String(i), out)


def _f64_from_bits(bits: UInt64) -> Float64:
    return bitcast[DType.float64, 1](SIMD[DType.uint64, 1](bits))[0]
