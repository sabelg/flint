"""Checkpoint access for `flint.ml.depth`: `.safetensors` or `torch.save` `.pt`, read in place.

`Weights.open(path, prefix)` maps either format once; every tensor is then addressed by its
name with `prefix` stripped (Depth Anything 3 stores `model.backbone...`, MoGe `model` /
`model_config` dicts inside the pickle). F32 tensors are borrowed from the mapping; F16 / BF16
are converted. `config` exposes the pickled `model_config` of a MoGe checkpoint.
"""

from std.collections import Dict
from std.ffi import external_call
from flint.ml.tensor import Tensor, f32_from_bits
from flint.ml.safetensors import SafeTensors
from flint.ml.torch_pt import TorchPt, PtTensor, KIND_LIST, KIND_INT, KIND_STR, KIND_NONE


struct WEntry(Copyable, Movable):
    var dtype: String
    var shape: List[Int]
    var addr: Int

    def __init__(out self, dtype: String, var shape: List[Int], addr: Int):
        self.dtype = dtype
        self.shape = shape^
        self.addr = addr


def _half_to_f32(h: UInt32) -> Float32:
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


struct Weights(Movable):
    var path: String
    var st: SafeTensors
    var pt: TorchPt
    var is_pt: Bool
    var entries: Dict[String, WEntry]
    var config: Int          # torch_pt node of `model_config` (-1 for safetensors)

    def __init__(out self):
        self.path = String()
        self.st = SafeTensors()
        self.pt = TorchPt()
        self.is_pt = False
        self.entries = Dict[String, WEntry]()
        self.config = -1

    @staticmethod
    def open(path: String, prefix: String = "") raises -> Weights:
        var w = Weights()
        w.path = path
        var plen = prefix.byte_length()
        if path.endswith(".safetensors"):
            w.st = SafeTensors.open(path)
            for name in w.st.names:
                var e = w.st.entries[name].copy()
                var key = name
                if plen > 0:
                    if not name.startswith(prefix):
                        continue
                    key = String(name[byte=plen:])
                w.entries[key] = WEntry(e.dtype, e.shape.copy(), w.st.data_start + e.begin)
        else:
            w.pt = TorchPt.open(path)
            w.is_pt = True
            var node = w.pt.root
            var sd = w.pt.get(node, "model")
            if sd < 0:
                sd = w.pt.get(node, "state_dict")
            if sd < 0:
                sd = node
            w.config = w.pt.get(node, "model_config")
            var ts = w.pt.tensors(sd)
            for e in ts.items():
                if not e.value.contiguous:
                    raise Error("depth: non-contiguous tensor " + e.key + " in " + path)
                var key = e.key
                if plen > 0:
                    if not key.startswith(prefix):
                        continue
                    var tail = String(key[byte=plen:])
                    key = tail^
                w.entries[key] = WEntry(e.value.dtype, e.value.shape.copy(), e.value.addr)
        return w^

    def has(self, name: String) -> Bool:
        return name in self.entries

    def shape(self, name: String) raises -> List[Int]:
        var e = self.entries.get(name)
        if not e:
            raise Error("depth: no tensor " + name + " in " + self.path)
        return e.value().shape.copy()

    def tensor(self, name: String) raises -> Tensor:
        """`name` as float32 (borrowed for F32, converted for F16 / BF16)."""
        var found = self.entries.get(name)
        if not found:
            raise Error("depth: no tensor " + name + " in " + self.path)
        var e = found.value().copy()
        var n = 1
        for d in e.shape:
            n *= d
        if e.dtype == "F32":
            return Tensor.borrowed(e.shape.copy(), e.addr)
        var out = List[Float32](length=n, fill=0.0)
        var ph = Pointer[UInt16, ImmutAnyOrigin](unsafe_from_address=e.addr)
        if e.dtype == "BF16":
            for i in range(n):
                out[i] = f32_from_bits(UInt32(ph.unsafe_offset(i).unsafe_load[alignment=1]()) << 16)
        elif e.dtype == "F16":
            for i in range(n):
                out[i] = _half_to_f32(UInt32(ph.unsafe_offset(i).unsafe_load[alignment=1]()))
        else:
            raise Error("depth: unsupported dtype " + e.dtype + " for " + name)
        return Tensor.floats(e.shape.copy(), out^)

    def floats(self, name: String) raises -> List[Float32]:
        """An owned float32 copy of `name`."""
        var t = self.tensor(name)
        var n = t.numel()
        var out = List[Float32](length=n, fill=0.0)
        _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), t.fptr(), n * 4)
        _ = t.numel()  # keep the (possibly converted) tensor alive past the raw copy
        return out^

    # ---- MoGe `model_config` access (dotted paths into the pickled dict)

    def _cfg(self, path: String) -> Int:
        var node = self.config
        for part in path.split("."):
            node = self.pt.get(node, String(part))
            if node < 0:
                return -1
        return node

    def cfg_has(self, path: String) -> Bool:
        return self._cfg(path) >= 0 and self.pt.kind(self._cfg(path)) != KIND_NONE

    def cfg_int(self, path: String, default: Int = 0) -> Int:
        return self.pt.integer(self._cfg(path), default)

    def cfg_str(self, path: String) -> String:
        return self.pt.text(self._cfg(path))

    def cfg_ints(self, path: String) -> List[Int]:
        """An int list (None entries read as -1); a bare int is a one-element list."""
        var node = self._cfg(path)
        var out = List[Int]()
        var k = self.pt.kind(node)
        if k == KIND_INT:
            out.append(self.pt.integer(node))
            return out^
        for i in range(self.pt.size(node)):
            var c = self.pt.at(node, i)
            out.append(self.pt.integer(c, -1) if self.pt.kind(c) == KIND_INT else -1)
        return out^

    def cfg_strs(self, path: String) -> List[String]:
        var node = self._cfg(path)
        var out = List[String]()
        if self.pt.kind(node) == KIND_STR:
            out.append(self.pt.text(node))
            return out^
        for i in range(self.pt.size(node)):
            out.append(self.pt.text(self.pt.at(node, i)))
        return out^
