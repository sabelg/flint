"""LZ4-compressed, 64-bit integrity-hashed binary state checkpoints (`flint.assets.checkpoint`).

Bridges `flint.core.serial` (`BlobWriter` / `BlobReader`) and `flint.core.lz4` (`lz4_pack` /
`lz4_unpack`) with `flint.platform.os` (`write_bytes` / `read_bytes`) so any Flint simulation
can save and restore bit-exact versioned state snapshots in a single call.
"""
from flint.core.lz4 import lz4_pack, lz4_unpack
from flint.core.serial import BlobWriter, BlobReader
from flint.platform.os import read_bytes, write_bytes


def pack_checkpoint_lz4(writer: BlobWriter) -> List[UInt8]:
    """Compresses `writer.data` into a framed LZ4 block with magic + 64-bit raw hash."""
    return lz4_pack(writer.data)


def unpack_checkpoint_lz4(packed: List[UInt8]) raises -> BlobReader:
    """Verifies magic, size bounds, and 64-bit hash of `packed` and returns a `BlobReader`."""
    var raw = lz4_unpack(packed)
    if len(raw) == 0:
        raise Error("corrupt or unreadable LZ4 checkpoint frame")
    return BlobReader(raw^)


def save_checkpoint_lz4(path: String, writer: BlobWriter) -> Bool:
    """Packs `writer` with LZ4 + integrity hash and writes it to `path`."""
    if len(writer.data) == 0:
        return False
    var packed = lz4_pack(writer.data)
    return write_bytes(path, packed)


def load_checkpoint_lz4(path: String) raises -> BlobReader:
    """Reads `path`, verifies the LZ4 frame & 64-bit hash, and returns a `BlobReader`."""
    var packed = read_bytes(path)
    if len(packed) == 0:
        raise Error("checkpoint file missing or empty: " + path)
    return unpack_checkpoint_lz4(packed)
