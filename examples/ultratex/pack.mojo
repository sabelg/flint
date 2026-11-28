"""Repackage official UltraTex weights as native SafeTensors. No Python runtime."""
from std.sys import argv
from flint.assets.cooker_v2 import sha256_bytes
from flint.assets.json_write import json_quote
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import numel_of
from flint.platform.os import file_size, read_bytes, write_bytes


struct _Pack(Movable):
    var entries: List[String]
    var payload: List[UInt8]

    def __init__(out self):
        self.entries = List[String]()
        self.payload = List[UInt8]()

    def append(
        mut self, key: String, dtype: String, shape: List[Int], source: List[UInt8], start: Int, size: Int
    ) raises:
        if (
            size <= 0
            or start < 0
            or start > len(source) - size
            or (dtype != "BF16" and dtype != "F16" and dtype != "F32")
            or numel_of(shape) != size // (4 if dtype == "F32" else 2)
            or size % (4 if dtype == "F32" else 2)
        ):
            raise Error("Invalid tensor storage: " + key)
        var dims = String("[")
        for i in range(len(shape)):
            if i:
                dims += ","
            dims += String(shape[i])
        dims += "]"
        var offset = len(self.payload)
        self.entries.append(
            json_quote(key)
            + ':{"dtype":'
            + json_quote(dtype)
            + ',"shape":'
            + dims
            + ',"data_offsets":['
            + String(offset)
            + ","
            + String(offset + size)
            + "]}"
        )
        self.payload.extend(source[start : start + size])

    def save(self, path: String) raises:
        var header = String("{")
        for i in range(len(self.entries)):
            if i:
                header += ","
            header += self.entries[i]
        header += "}"
        while header.byte_length() % 8:
            header += " "
        var out = List[UInt8](capacity=8 + header.byte_length() + len(self.payload))
        comptime for j in range(8):
            out.append(UInt8(UInt64(header.byte_length()) >> UInt64(j * 8)))
        out.extend(header.as_bytes())
        out.extend(self.payload.copy())
        if not write_bytes(path, out):
            raise Error("Failed to write " + path)


def _encoder(src: String, dest: String) raises:
    var w = SafeTensorWeights(src, 1 << 30)
    var pack = _Pack()
    for key in w.header.nodes[w.header.root].keys:
        if not (
            key.startswith("encoder.")
            or key.startswith("quant_conv.")
            or key == "bn.running_mean"
            or key == "bn.running_var"
        ):
            continue
        var node = w.header.object_get(w.header.root, key)
        var dims = w.header.object_get(node, "shape")
        var shape = List[Int]()
        for j in range(w.header.array_size(dims)):
            shape.append(w.header.integer(w.header.at(dims, j)))
        var name = key.copy()
        for i in range(4):
            var level = String(i)
            name = name.replace("down_blocks." + level + ".resnets.", "down." + level + ".block.")
            name = name.replace("down_blocks." + level + ".downsamplers.0.", "down." + level + ".downsample.")
        if name.startswith("quant_conv."):
            name = "encoder." + name
        name = name.replace("conv_shortcut.", "nin_shortcut.").replace("conv_norm_out.", "norm_out.")
        name = (
            name.replace("mid_block.resnets.0.", "mid.block_1.")
            .replace("mid_block.resnets.1.", "mid.block_2.")
            .replace("mid_block.attentions.0.", "mid.attn_1.")
        )
        name = name.replace(".attn_1.group_norm.", ".attn_1.norm.").replace(
            ".attn_1.to_out.0.", ".attn_1.proj_out."
        )
        name = (
            name.replace(".attn_1.to_q.", ".attn_1.q.")
            .replace(".attn_1.to_k.", ".attn_1.k.")
            .replace(".attn_1.to_v.", ".attn_1.v.")
        )
        if ".attn_1." in name and len(shape) == 2:
            shape.extend([1, 1])
        var offsets = w.header.object_get(node, "data_offsets")
        var start = w.header.integer(w.header.at(offsets, 0))
        var end = w.header.integer(w.header.at(offsets, 1))
        if start < 0 or end < start or end > len(w.archive) - w.data_offset:
            raise Error("Invalid autoencoder tensor offset")
        pack.append(
            name,
            w.header.text(w.header.object_get(node, "dtype")),
            shape,
            w.archive,
            w.data_offset + start,
            end - start,
        )
    if (
        w.header.object_get(w.header.root, "encoder.conv_in.weight") < 0
        or w.header.object_get(w.header.root, "bn.running_var") < 0
        or (
            w.header.object_get(w.header.root, "encoder.quant_conv.weight") < 0
            and w.header.object_get(w.header.root, "quant_conv.weight") < 0
        )
    ):
        raise Error("Expected a FLUX.2 VAE")
    pack.save(dest)


def _little(data: List[UInt8], at: Int, size: Int) -> Int:
    var v = 0
    for j in range(size):
        v |= Int(data[at + j]) << (8 * j)
    return v


def _integer(data: List[UInt8], mut at: Int) raises -> Int:
    var op = data[at]
    at += 1
    var size = 1 if op == 75 else (2 if op == 77 else (4 if op == 74 else 0))
    if size == 0:
        raise Error("Unexpected decoder integer encoding")
    var n = _little(data, at, size)
    at += size
    return n


def _decoder(src: String, dest: String) raises:
    # Only accept the released FLUX.2 decoder. That archive has a protocol-2
    # metadata member and one stored BF16 tensor buffer. Read data descriptors
    # directly: no pickle globals or reducers are invoked.
    if file_size(src) != 99261764:
        raise Error("Expected the official UltraTex FLUX.2 decoder.pt")
    var data = read_bytes(src)
    var hash = sha256_bytes(data)
    var expected: List[UInt32] = [
        0x981DC4C5,
        0x2481687C,
        0x6DA4E621,
        0x1617615A,
        0x2B2BCC33,
        0x456E1953,
        0xEE6C5478,
        0xAA00025B,
    ]
    for i in range(8):
        if hash[i] != expected[i]:
            raise Error("FLUX.2 decoder SHA-256 differs from the published checkpoint")
    var pack = _Pack()
    var key = String()
    var at = 64
    while at < 64 + 20144 and len(pack.entries) < 140:
        var op = data[at]
        at += 1
        if op == 88:  # BINUNICODE
            var n = _little(data, at, 4)
            at += 4
            var s = String(unsafe_from_utf8=data[at : at + n])
            at += n
            if s.endswith(".weight") or s.endswith(".bias"):
                key = s^
        elif op == 81:  # BINPERSID followed by storage offset and shape
            var offset = _integer(data, at)
            if data[at] == 40:
                at += 1
            var shape = List[Int]()
            while data[at] != 116 and data[at] != 133:
                shape.append(_integer(data, at))
            at += 1
            pack.append(key, "BF16", shape, data, 20416 + 2 * offset, 2 * numel_of(shape))
        elif op == 99:  # GLOBAL has two newline-terminated names
            for _ in range(2):
                while data[at] != 10:
                    at += 1
                at += 1
        elif op == 128 or op == 113 or op == 104 or op == 75:
            at += 1
        elif op == 114 or op == 106 or op == 74:
            at += 4
        elif op == 77:
            at += 2
    if len(pack.entries) != 140:
        raise Error("Incomplete UltraTex decoder")
    pack.save(dest)


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error("Usage: pack encoder|decoder INPUT OUTPUT")
    if String(args[2]) == String(args[3]):
        raise Error("Supply distinct input and output checkpoint paths")
    if String(args[1]) == "encoder":
        _encoder(String(args[2]), String(args[3]))
    elif String(args[1]) == "decoder":
        _decoder(String(args[2]), String(args[3]))
    else:
        raise Error("Expected encoder or decoder")
