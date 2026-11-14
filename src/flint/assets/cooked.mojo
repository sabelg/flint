"""The header on every cooked file.

The pipeline cooks each manifest to build/cooked/<asset>.sdcooked: this header
(kind 1), then the JSON; the game reads manifests as JSON. Cooked models
(package 1.4, assets/cooked_model.mojo) start with the same 32 bytes, kind 2,
and hash their payload with core/lz4.mojo's block_hash rather than FNV.
"""
from flint.core.replay import ByteReader, Fnv64, put_u32, put_u64
from flint.assets.manifest import ENGINE_FORMAT

comptime COOKED_MAGIC: UInt32 = 0x4B434453  # "SDCK"
comptime COOKED_HEADER_BYTES: Int = 32


def hash_bytes(data: List[UInt8]) -> UInt64:
    var hash = Fnv64()
    for byte in data:
        hash.add_byte(byte)
    return hash.value


struct CookedHeader(Copyable, Movable):
    var engine_format: Int
    var kind: Int  # 1 = manifest
    var asset_id: UInt64
    var payload_hash: UInt64
    var payload_bytes: Int

    def __init__(out self):
        self.engine_format = 0
        self.kind = 0
        self.asset_id = 0
        self.payload_hash = 0
        self.payload_bytes = 0


def encode_cooked(kind: Int, asset_id: UInt64, payload: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    put_u32(out, COOKED_MAGIC)
    put_u32(out, UInt32(ENGINE_FORMAT))
    put_u32(out, UInt32(kind))
    put_u32(out, UInt32(len(payload)))
    put_u64(out, asset_id)
    put_u64(out, hash_bytes(payload))
    for byte in payload:
        out.append(byte)
    return out^


def decode_cooked(var data: List[UInt8]) -> CookedHeader:
    """Returns a header with engine_format 0 for anything this engine must not
    read: a short file, another format version, or a damaged payload."""
    var header = CookedHeader()
    if len(data) < COOKED_HEADER_BYTES:
        return header^
    var total = len(data)
    var reader = ByteReader(data^)
    if reader.u32() != COOKED_MAGIC:
        return header^
    var engine_format = Int(reader.u32())
    var kind = Int(reader.u32())
    var size = Int(reader.u32())
    var asset_id = reader.u64()
    var payload_hash = reader.u64()
    if engine_format != ENGINE_FORMAT or size != total - COOKED_HEADER_BYTES or hash_bytes(reader.bytes(size)) != payload_hash:
        return header^
    header.engine_format = engine_format
    header.kind = kind
    header.asset_id = asset_id
    header.payload_hash = payload_hash
    header.payload_bytes = size
    return header^
