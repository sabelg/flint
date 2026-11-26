"""Flint v5 Zero-Parse `.flpk` v3 Archive, 128-Bit `AssetUuid`, `.flmeta` Sidecars & Generational Handles (`flpk_v3.mojo`).

Implements:
  1. `AssetUuid`: 128-bit UUID with canonical 36-char `"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"` parser/formatter
     and deterministic path-based UUIDv5-style synthesis.
  2. `AssetMeta` (`.flmeta`): JSON sidecar reader/writer (`flint.assets.json` / `json_write`) storing stable UUIDs
     and content-addressed import settings.
  3. Generational `UntypedHandle` (`kind: 8b | slot: 32b | generation: 24b`) and strongly-typed `Handle[T: AssetPayload]`.
  4. `.flpk` v3 Zero-Parse Container:
     - Exact 64-byte `FlpkV3Header` & 64-byte `FlpkTocEntry`
     - Embedded O(1) Robin-Hood `UUID` (`32 B/slot`) & `path-FNV64` (`16 B/slot`) hash tables
     - 64-byte cache-line & 4096-byte OS page-aligned payload slabs
     - `CHUNK_COMP_NONE` (direct zero-copy `mmap` slice!) & `CHUNK_COMP_LZ4`
     - ECMA-182 `CRC64` (`0x42F0E1EBA9EA3693`) header/TOC & per-chunk integrity verification
     - `CookedMeshSlabHeader` (64 B) for pre-built `AaaGeometryPool` (`vertex_words`, `index_words`, `meshlet_words` + QEM `LOD0..3`)
     - Streamable `BC7`/`BC5`/`BC6H` mip sub-chunks (`sub_id = 0` mip-tail + `sub_id = 1..15` high-res mips)
     - `sub_id = 16` (`SUB_ID_BVH8_COLLIDER`) pre-built `Bvh8MeshCollider` static collider slab.
"""
from std.ffi import external_call
from std.math import isfinite
from std.memory import bitcast
from flint.core.lz4 import address_of, lz4_compress, lz4_decompress_into
from flint.core.math import Vec3
from flint.core.serial import float32_bits, float32_from_bits, fnv64
from flint.assets.json import parse_json
from flint.assets.json_write import json_quote
from flint.assets.meshlet import (
    AAA_MESHLET_WORDS,
    AaaGeometryPool,
    CookedMeshEntry,
    MeshLodSlice,
    MeshletCluster,
)
from flint.physics.bvh import Bvh8MeshCollider, Bvh8Node, MeshTriangle
from flint.platform.os import close_fd, file_stamp, open_read, read_bytes, write_bytes

# ============================================================================
# 1. Constants & Asset Kinds
# ============================================================================

comptime FLPK_V3_MAGIC: UInt32 = 0x4B504C46    # "FLPK" little-endian
comptime FLPK_V3_VERSION: UInt32 = 3
comptime FLPK_HEADER_BYTES: Int = 64
comptime FLPK_TOC_ENTRY_BYTES: Int = 64
comptime FLPK_UUID_SLOT_BYTES: Int = 32
comptime FLPK_PATH_SLOT_BYTES: Int = 16

comptime FLPK_MAX_ENTRIES: UInt64 = 1_048_576
comptime FLPK_MAX_HASH_CAP: UInt64 = 2_097_152
comptime FLPK_MAX_UNCOMPRESSED_BYTES: UInt64 = 1_073_741_824  # 1 GiB
comptime FLPK_EMPTY_SLOT_SENTINEL: UInt32 = 0xFFFFFFFF

comptime FLPK_FLAG_PAGE_ALIGNED: UInt32 = 1 << 0

comptime CHUNK_COMP_NONE: UInt8 = 0
comptime CHUNK_COMP_LZ4: UInt8 = 1

comptime ASSET_KIND_RAW: UInt8 = 0
comptime ASSET_KIND_MESH: UInt8 = 1
comptime ASSET_KIND_TEXTURE: UInt8 = 2
comptime ASSET_KIND_HDR: UInt8 = 3
comptime ASSET_KIND_FONT: UInt8 = 4
comptime ASSET_KIND_AUDIO: UInt8 = 5

comptime TEX_FMT_BC7_SRGB: UInt8 = 1
comptime TEX_FMT_BC7_UNORM: UInt8 = 2
comptime TEX_FMT_BC5_UNORM: UInt8 = 3
comptime TEX_FMT_BC6H_UFLOAT: UInt8 = 4

comptime SUB_ID_PRIMARY_OR_TAIL: UInt16 = 0
comptime SUB_ID_MIP_BASE: UInt16 = 1           # sub_id 1..15 -> high-res mips 0..14
comptime SUB_ID_BVH8_COLLIDER: UInt16 = 16
comptime SUB_ID_RIG: UInt16 = 17

comptime COOKED_MESH_SLAB_MAGIC: UInt32 = 0x48534D46  # "FMSH"
comptime COOKED_TEX_SLAB_MAGIC: UInt32 = 0x58455446   # "FTEX"
comptime COOKED_BVH8_SLAB_MAGIC: UInt32 = 0x38485642  # "BVH8"
comptime COOKED_AUD_SLAB_MAGIC: UInt32 = 0x44554146   # "FAUD"
comptime COOKED_RIG_SLAB_MAGIC: UInt32 = 0x47495246   # "FRIG"


@always_inline
def mip_to_sub_id(mip_level: Int) -> UInt16:
    """Maps high-res mip level `0..14` to `.flpk` v3 `sub_id = 1..15`."""
    return UInt16(max(1, min(15, mip_level + 1)))


@always_inline
def sub_id_to_mip(sub_id: UInt16) -> Int:
    """Maps `.flpk` v3 `sub_id = 1..15` back to mip level `0..14`."""
    return Int(sub_id) - 1


# ============================================================================
# 2. ECMA-182 64-Bit CRC (`0x42F0E1EBA9EA3693`) & Binary Helpers
# ============================================================================

comptime CRC64_ECMA_POLY: UInt64 = 0x42F0E1EBA9EA3693


@always_inline
def _mix64(x: UInt64) -> UInt64:
    var z = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def fnv64_str(s: String) -> UInt64:
    var value = UInt64(0xCBF29CE484222325)
    for b in s.as_bytes():
        value = (value ^ UInt64(b)) * 0x100000001B3
    return value


def crc64_ecma_ptr(addr: Int, length: Int, seed: UInt64 = 0) -> UInt64:
    """Computes standard ECMA-182 64-bit CRC (`poly = 0x42F0E1EBA9EA3693`, normal representation)
    over `length` bytes at `addr` with zero heap allocations."""
    if addr == 0 or length <= 0:
        return seed
    # Build 16-entry nibble table on stack (128 bytes, L1-resident)
    var tbl = Array[UInt64, 16](fill=0)
    for n in range(16):
        var c = UInt64(n) << 60
        for _ in range(4):
            if (c & 0x8000000000000000) != 0:
                c = (c << 1) ^ CRC64_ECMA_POLY
            else:
                c = c << 1
        tbl[n] = c

    var crc = ~seed
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr)
    for i in range(length):
        var b = UInt64(p[unsafe_offset=i])
        var hi_nib = Int(((crc >> 60) ^ (b >> 4)) & 0xF)
        crc = (crc << 4) ^ tbl[hi_nib]
        var lo_nib = Int(((crc >> 60) ^ (b & 0xF)) & 0xF)
        crc = (crc << 4) ^ tbl[lo_nib]
    return ~crc


def crc64_ecma(data: List[UInt8], seed: UInt64 = 0) -> UInt64:
    if len(data) == 0:
        return seed
    var crc = crc64_ecma_ptr(address_of(data), len(data), seed)
    _ = len(data)
    return crc


@always_inline
def _put_u16_le(mut out: List[UInt8], v: UInt16):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))


@always_inline
def _put_u32_le(mut out: List[UInt8], v: UInt32):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))


@always_inline
def _put_u64_le(mut out: List[UInt8], v: UInt64):
    for s in range(8):
        out.append(UInt8((v >> UInt64(s * 8)) & 0xFF))


@always_inline
def _write_u64_at(mut out: List[UInt8], at: Int, v: UInt64):
    for s in range(8):
        out[at + s] = UInt8((v >> UInt64(s * 8)) & 0xFF)


@always_inline
def _ptr_u8(addr: Int, off: Int) -> UInt8:
    return Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr)[unsafe_offset=off]


@always_inline
def _ptr_u16(addr: Int, off: Int) -> UInt16:
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr)
    return UInt16(p[unsafe_offset=off]) | (UInt16(p[unsafe_offset=off + 1]) << 8)


@always_inline
def _ptr_u32(addr: Int, off: Int) -> UInt32:
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr)
    return (
        UInt32(p[unsafe_offset=off])
        | (UInt32(p[unsafe_offset=off + 1]) << 8)
        | (UInt32(p[unsafe_offset=off + 2]) << 16)
        | (UInt32(p[unsafe_offset=off + 3]) << 24)
    )


@always_inline
def _ptr_u64(addr: Int, off: Int) -> UInt64:
    return UInt64(_ptr_u32(addr, off)) | (UInt64(_ptr_u32(addr, off + 4)) << 32)


# ============================================================================
# 3. 128-Bit `AssetUuid` & Canonical Parser/Formatter
# ============================================================================

@always_inline
def _hex_nibble(b: UInt8) -> Int:
    if b >= 48 and b <= 57:
        return Int(b - 48)
    if b >= 97 and b <= 102:
        return Int(b - 97 + 10)
    if b >= 65 and b <= 70:
        return Int(b - 65 + 10)
    return -1


@fieldwise_init
struct AssetUuid(Copyable, ImplicitlyCopyable, Movable, Equatable):
    """128-bit asset identifier with canonical `"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"` formatting."""
    var hi: UInt64
    var lo: UInt64

    def __init__(out self):
        self.hi = 0
        self.lo = 0

    @always_inline
    def is_nil(self) -> Bool:
        return self.hi == 0 and self.lo == 0

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.hi == other.hi and self.lo == other.lo

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self.hi != other.hi or self.lo != other.lo

    @always_inline
    def hash64(self) -> UInt64:
        return _mix64(self.hi ^ _mix64(self.lo ^ 0x9E3779B97F4A7C15))

    @staticmethod
    def from_path(rel_path: String) -> Self:
        """Synthesizes a deterministic RFC-4122 v5-shaped 128-bit UUID from a canonical relative asset path."""
        var h0 = fnv64_str("flint.uuid.ns:" + rel_path)
        var h1 = _mix64(h0 ^ fnv64_str(rel_path + "#lo"))
        var hi_v5 = (_mix64(h0) & ~UInt64(0xF000)) | UInt64(0x5000)
        var lo_v5 = (h1 & ~UInt64(0xC000000000000000)) | UInt64(0x8000000000000000)
        if hi_v5 == 0 and lo_v5 == 0:
            lo_v5 = 1
        return Self(hi_v5, lo_v5)

    @staticmethod
    def parse(text: String) -> Tuple[Bool, Self]:
        """Parses canonical 36-char `"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"` (or 32-char compact hex)."""
        var b = text.as_bytes()
        var n = len(b)
        if n != 36 and n != 32:
            return (False, Self())
        if n == 36:
            if b[8] != 45 or b[13] != 45 or b[18] != 45 or b[23] != 45:
                return (False, Self())
        var hi_acc = UInt64(0)
        var lo_acc = UInt64(0)
        var hex_idx = 0
        for i in range(n):
            if n == 36 and (i == 8 or i == 13 or i == 18 or i == 23):
                continue
            var nib = _hex_nibble(b[i])
            if nib < 0:
                return (False, Self())
            if hex_idx < 16:
                hi_acc = (hi_acc << 4) | UInt64(nib)
            else:
                lo_acc = (lo_acc << 4) | UInt64(nib)
            hex_idx += 1
        if hex_idx != 32:
            return (False, Self())
        return (True, Self(hi_acc, lo_acc))

    def format(self) -> String:
        """Formats as canonical lowercase 36-char `"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"`."""
        var hex = "0123456789abcdef".as_bytes()
        var out = List[UInt8](capacity=36)
        for nib_i in range(32):
            if nib_i == 8 or nib_i == 12 or nib_i == 16 or nib_i == 20:
                out.append(45)
            var v: Int
            if nib_i < 16:
                var shift = UInt64((15 - nib_i) * 4)
                v = Int((self.hi >> shift) & 0xF)
            else:
                var shift = UInt64((31 - nib_i) * 4)
                v = Int((self.lo >> shift) & 0xF)
            out.append(hex[v])
        return String(unsafe_from_utf8=out)


def parse_asset_uuid(text: String) raises -> AssetUuid:
    var res = AssetUuid.parse(text)
    if not res[0]:
        raise Error("invalid AssetUuid string: " + text)
    return res[1]


# ============================================================================
# 4. `.flmeta` JSON Sidecar Reader / Writer
# ============================================================================

struct AssetMeta(Copyable, Movable):
    """Asset sidecar metadata (`<asset>.flmeta`) controlling deterministic UUID & cooker settings."""
    var uuid: AssetUuid
    var asset_kind: UInt8
    var srgb: Bool
    var generate_mips: Bool
    var build_lods: Bool
    var build_collider: Bool
    var compress_lz4: Bool
    var tail_first_mip: Int
    var max_tex_dim: Int
    var toksvig_scale: Float32
    var source_path: String

    def __init__(out self):
        self.uuid = AssetUuid()
        self.asset_kind = ASSET_KIND_RAW
        self.srgb = True
        self.generate_mips = True
        self.build_lods = True
        self.build_collider = True
        self.compress_lz4 = False
        self.tail_first_mip = 4
        self.max_tex_dim = 2048
        self.toksvig_scale = Float32(1.0)
        self.source_path = String()

    @staticmethod
    def _safe_toksvig(v: Float32) -> Float32:
        if not isfinite(v):
            return Float32(1.0)
        return max(Float32(0.0), min(Float32(1.0e6), v))

    def settings_hash(self) -> UInt64:
        var safe_tok = Self._safe_toksvig(self.toksvig_scale)
        var s = (
            self.uuid.format()
            + "|k=" + String(Int(self.asset_kind))
            + "|srgb=" + ("1" if self.srgb else "0")
            + "|mips=" + ("1" if self.generate_mips else "0")
            + "|lods=" + ("1" if self.build_lods else "0")
            + "|col=" + ("1" if self.build_collider else "0")
            + "|lz4=" + ("1" if self.compress_lz4 else "0")
            + "|tail=" + String(self.tail_first_mip)
            + "|maxd=" + String(self.max_tex_dim)
            + "|tok=" + String(Int(Float64(safe_tok) * 1000.0))
        )
        return fnv64_str(s)

    def to_json(self) -> String:
        var safe_tok = Self._safe_toksvig(self.toksvig_scale)
        return (
            "{\n"
            + '  "uuid": ' + json_quote(self.uuid.format()) + ",\n"
            + '  "asset_kind": ' + String(Int(self.asset_kind)) + ",\n"
            + '  "srgb": ' + ("true" if self.srgb else "false") + ",\n"
            + '  "generate_mips": ' + ("true" if self.generate_mips else "false") + ",\n"
            + '  "build_lods": ' + ("true" if self.build_lods else "false") + ",\n"
            + '  "build_collider": ' + ("true" if self.build_collider else "false") + ",\n"
            + '  "compress_lz4": ' + ("true" if self.compress_lz4 else "false") + ",\n"
            + '  "tail_first_mip": ' + String(self.tail_first_mip) + ",\n"
            + '  "max_tex_dim": ' + String(self.max_tex_dim) + ",\n"
            + '  "toksvig_scale": ' + String(Float64(safe_tok)) + ",\n"
            + '  "source_path": ' + json_quote(self.source_path) + "\n"
            + "}\n"
        )


def parse_flmeta_bytes(raw: List[UInt8]) -> Tuple[Bool, AssetMeta]:
    var meta = AssetMeta()
    if len(raw) == 0:
        return (False, meta^)
    var doc = parse_json(raw)
    if not doc.valid or doc.root < 0 or doc.nodes[doc.root].kind != 5:
        return (False, meta^)
    var u_txt = doc.text(doc.object_get(doc.root, "uuid"))
    var u_parsed = AssetUuid.parse(u_txt)
    if not u_parsed[0]:
        return (False, meta^)
    meta.uuid = u_parsed[1]
    meta.asset_kind = UInt8(max(0, min(255, doc.integer(doc.object_get(doc.root, "asset_kind"), 0))))
    meta.srgb = doc.boolean(doc.object_get(doc.root, "srgb"), True)
    meta.generate_mips = doc.boolean(doc.object_get(doc.root, "generate_mips"), True)
    meta.build_lods = doc.boolean(doc.object_get(doc.root, "build_lods"), True)
    meta.build_collider = doc.boolean(doc.object_get(doc.root, "build_collider"), True)
    meta.compress_lz4 = doc.boolean(doc.object_get(doc.root, "compress_lz4"), False)
    meta.tail_first_mip = max(0, min(15, doc.integer(doc.object_get(doc.root, "tail_first_mip"), 4)))
    meta.max_tex_dim = max(4, doc.integer(doc.object_get(doc.root, "max_tex_dim"), 2048))
    var tok_node = doc.object_get(doc.root, "toksvig_scale")
    if tok_node >= 0:
        var tok_d = doc.number(tok_node, 1.0)
        var tok_f = Float32(tok_d)
        if not isfinite(tok_d) or not isfinite(tok_f):
            return (False, meta^)
        meta.toksvig_scale = AssetMeta._safe_toksvig(tok_f)
    else:
        meta.toksvig_scale = Float32(1.0)
    meta.source_path = doc.text(doc.object_get(doc.root, "source_path"))
    return (True, meta^)


def read_flmeta(meta_path: String) -> Tuple[Bool, AssetMeta]:
    return parse_flmeta_bytes(read_bytes(meta_path))


def write_flmeta(meta_path: String, meta: AssetMeta) -> Bool:
    var txt = meta.to_json()
    var raw = List[UInt8](capacity=txt.byte_length())
    for b in txt.as_bytes():
        raw.append(b)
    return write_bytes(meta_path, raw)


def load_or_init_flmeta(
    full_source_path: String,
    rel_path: String,
    default_kind: UInt8,
    persist_if_missing: Bool = False,
) -> AssetMeta:
    var meta_path = full_source_path + ".flmeta"
    var got = read_flmeta(meta_path)
    if got[0]:
        return got[1].copy()
    var m = AssetMeta()
    m.uuid = AssetUuid.from_path(rel_path)
    m.asset_kind = default_kind
    m.source_path = rel_path
    if rel_path.endswith("_normal.png") or rel_path.endswith("_nrm.png") or rel_path.endswith("_orm.png"):
        m.srgb = False
    if default_kind == ASSET_KIND_FONT or default_kind == ASSET_KIND_AUDIO:
        m.compress_lz4 = True
    if persist_if_missing:
        _ = write_flmeta(meta_path, m)
    return m^


# ============================================================================
# 5. Generational `UntypedHandle` & Strongly-Typed `Handle[T: AssetPayload]`
# ============================================================================

trait AssetPayload(Copyable, Movable, Deinitable):
    """Trait implemented by all typed engine assets stored in `AssetServer`."""

    @staticmethod
    def asset_kind() -> UInt8:
        ...

    def byte_footprint(self) -> Int:
        ...


@fieldwise_init
struct UntypedHandle(Copyable, ImplicitlyCopyable, Movable, Equatable):
    """64-bit generational handle: `kind: 8b (bits 56..63) | slot: 32b (bits 24..55) | generation: 24b (bits 0..23)`."""
    var packed: UInt64

    def __init__(out self):
        self.packed = 0

    def __init__(out self, kind: UInt8, slot: UInt32, generation: UInt32):
        self.packed = (UInt64(kind) << 56) | (UInt64(slot) << 24) | UInt64(generation & 0xFFFFFF)

    @always_inline
    def kind(self) -> UInt8:
        return UInt8((self.packed >> 56) & 0xFF)

    @always_inline
    def slot(self) -> UInt32:
        return UInt32((self.packed >> 24) & 0xFFFFFFFF)

    @always_inline
    def generation(self) -> UInt32:
        return UInt32(self.packed & 0xFFFFFF)

    @always_inline
    def is_valid(self) -> Bool:
        return self.generation() != 0

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.packed == other.packed

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self.packed != other.packed


struct Handle[T: AssetPayload](Copyable, ImplicitlyCopyable, Movable, Equatable):
    """Strongly-typed generational handle `Handle[T]` wrapping `UntypedHandle`."""
    var raw: UntypedHandle

    def __init__(out self):
        self.raw = UntypedHandle()

    def __init__(out self, raw: UntypedHandle):
        self.raw = raw

    def __init__(out self, slot: UInt32, generation: UInt32):
        self.raw = UntypedHandle(Self.T.asset_kind(), slot, generation)

    @always_inline
    def slot(self) -> UInt32:
        return self.raw.slot()

    @always_inline
    def generation(self) -> UInt32:
        return self.raw.generation()

    @always_inline
    def kind(self) -> UInt8:
        return self.raw.kind()

    @always_inline
    def is_valid(self) -> Bool:
        return self.raw.is_valid() and self.raw.kind() == Self.T.asset_kind()

    @always_inline
    def untyped(self) -> UntypedHandle:
        return self.raw

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.raw == other.raw

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self.raw != other.raw


# ============================================================================
# 6. `.flpk` v3 64-Byte Header, 64-Byte TOC Entry & Zero-Copy Archive
# ============================================================================

@fieldwise_init
struct FlpkV3Header(Copyable, ImplicitlyCopyable, Movable):
    """Exact 64-byte `.flpk` v3 archive header."""
    var magic: UInt32               # [0..4]   0x4B504C46 ("FLPK")
    var version: UInt32             # [4..8]   3
    var flags: UInt32               # [8..12]  FLPK_FLAG_*
    var toc_count: UInt32           # [12..16] number of 64B FlpkTocEntry records
    var toc_offset: UInt64          # [16..24] byte offset of TOC array
    var uuid_table_offset: UInt64   # [24..32] byte offset of Robin-Hood UUID table
    var path_table_offset: UInt64   # [32..40] byte offset of Robin-Hood path-FNV64 table
    var hash_table_capacity: UInt32 # [40..44] power-of-2 slot count of each Robin-Hood table
    var string_pool_bytes: UInt32   # [44..48] length of UTF-8 relative path string pool
    var total_file_bytes: UInt64    # [48..56] total archive size in bytes
    var toc_crc64: UInt64           # [56..64] ECMA-182 CRC64 over bytes [0..56] + metadata region

    def __init__(out self):
        self.magic = FLPK_V3_MAGIC
        self.version = FLPK_V3_VERSION
        self.flags = 0
        self.toc_count = 0
        self.toc_offset = 64
        self.uuid_table_offset = 64
        self.path_table_offset = 64
        self.hash_table_capacity = 0
        self.string_pool_bytes = 0
        self.total_file_bytes = 64
        self.toc_crc64 = 0


@fieldwise_init
struct FlpkTocEntry(Copyable, ImplicitlyCopyable, Movable):
    """Exact 64-byte `.flpk` v3 Table-of-Contents entry."""
    var uuid: AssetUuid             # [0..16]  128-bit AssetUuid (hi, lo)
    var path_fnv64: UInt64          # [16..24] FNV-1a 64 hash of normalized relative path
    var payload_offset: UInt64      # [24..32] 64B / 4096B aligned byte offset in archive
    var compressed_bytes: UInt64    # [32..40] stored byte size in archive
    var uncompressed_bytes: UInt64  # [40..48] decoded byte size
    var payload_crc64: UInt64       # [48..56] ECMA-182 CRC64 of stored payload slab
    var sub_id: UInt16              # [56..58] 0=tail/primary, 1..15=high-res mips, 16=Bvh8 collider
    var asset_kind: UInt8           # [58]     ASSET_KIND_*
    var compression: UInt8          # [59]     CHUNK_COMP_NONE (0) or CHUNK_COMP_LZ4 (1)
    var path_str_offset: UInt32     # [60..64] offset in string pool (u16 off | u16 len packed: lo20=off, hi12=len)

    def __init__(out self):
        self.uuid = AssetUuid()
        self.path_fnv64 = 0
        self.payload_offset = 0
        self.compressed_bytes = 0
        self.uncompressed_bytes = 0
        self.payload_crc64 = 0
        self.sub_id = 0
        self.asset_kind = ASSET_KIND_RAW
        self.compression = CHUNK_COMP_NONE
        self.path_str_offset = 0


struct FlpkBuildChunk(Copyable, Movable):
    """Input chunk specification for `encode_flpk_v3`."""
    var uuid: AssetUuid
    var rel_path: String
    var sub_id: UInt16
    var asset_kind: UInt8
    var compression: UInt8
    var raw_payload: List[UInt8]

    def __init__(
        out self,
        uuid: AssetUuid,
        rel_path: String,
        sub_id: UInt16,
        asset_kind: UInt8,
        var raw_payload: List[UInt8],
        compression: UInt8 = CHUNK_COMP_NONE,
    ):
        self.uuid = uuid
        self.rel_path = rel_path
        self.sub_id = sub_id
        self.asset_kind = asset_kind
        self.compression = compression
        self.raw_payload = raw_payload^


@fieldwise_init
struct FlpkZeroCopySlice(Copyable, ImplicitlyCopyable, Movable):
    """Zero-copy pointer slice directly into a persistent `FlpkV3MappedArchive` (`CHUNK_COMP_NONE`)."""
    var addr: Int
    var size: Int
    var valid: Bool

    def __init__(out self):
        self.addr = 0
        self.size = 0
        self.valid = False


@always_inline
def _robin_uuid_hash(uuid: AssetUuid, sub_id: UInt16) -> UInt64:
    return _mix64(uuid.hash64() ^ (UInt64(sub_id) * 0x9E3779B97F4A7C15))


@always_inline
def _robin_path_hash(path_fnv: UInt64, sub_id: UInt16) -> UInt64:
    return _mix64(path_fnv ^ (UInt64(sub_id) * 0xBF58476D1CE4E5B9))


@always_inline
def _align_up(val: Int, align: Int) -> Int:
    return (val + align - 1) & ~(align - 1)


@always_inline
def _range_within_u64(offset: UInt64, length: UInt64, total_size: UInt64) -> Bool:
    """Overflow-safe unsigned range check (`[offset, offset + length] <= total_size`)."""
    return offset <= total_size and length <= total_size - offset


@always_inline
def _range_within(offset: Int, length: Int, total_size: Int) -> Bool:
    if offset < 0 or length < 0 or total_size < 0:
        return False
    return offset <= total_size and length <= total_size - offset


def encode_flpk_v3(chunks: List[FlpkBuildChunk], page_align: Bool = True) -> List[UInt8]:
    """Packs `chunks` into a zero-parse `.flpk` v3 image with O(1) Robin-Hood UUID & Path tables,
    64B/4096B aligned payload slabs, optional LZ4 compression, and ECMA-182 CRC64 checksums."""
    var n = len(chunks)
    var hcap = 0 if n == 0 else 16
    while n > 0 and hcap < max(16, n * 2):
        hcap <<= 1
    var hmask = (hcap - 1) if hcap > 0 else 0

    # Prepare stored payloads & string pool
    var stored_slabs = List[List[UInt8]](capacity=n)
    var comp_modes = List[UInt8](capacity=n)
    var str_pool = List[UInt8]()
    var str_packed = List[UInt32](capacity=n)

    for i in range(n):
        ref ch = chunks[i]
        var p_off = len(str_pool)
        var pb = ch.rel_path.as_bytes()
        var p_len = min(4095, len(pb))
        if p_off + p_len <= 0xFFFFF:
            for k in range(p_len):
                str_pool.append(pb[k])
            str_packed.append(UInt32(p_off & 0xFFFFF) | (UInt32(p_len & 0xFFF) << 20))
        else:
            str_packed.append(0)

        if ch.compression == CHUNK_COMP_LZ4 and len(ch.raw_payload) >= 64:
            var lz = lz4_compress(ch.raw_payload)
            if len(lz) > 0 and len(lz) < len(ch.raw_payload):
                stored_slabs.append(lz^)
                comp_modes.append(CHUNK_COMP_LZ4)
            else:
                stored_slabs.append(ch.raw_payload.copy())
                comp_modes.append(CHUNK_COMP_NONE)
        else:
            stored_slabs.append(ch.raw_payload.copy())
            comp_modes.append(CHUNK_COMP_NONE)

    var toc_off = FLPK_HEADER_BYTES
    var uuid_tbl_off = toc_off + n * FLPK_TOC_ENTRY_BYTES
    var path_tbl_off = uuid_tbl_off + hcap * FLPK_UUID_SLOT_BYTES
    var str_pool_off = path_tbl_off + hcap * FLPK_PATH_SLOT_BYTES
    var meta_end = _align_up(str_pool_off + len(str_pool), 64)

    var slab_align = 4096 if page_align else 64
    var cur_off = _align_up(meta_end, slab_align) if n > 0 else meta_end
    var offsets = List[Int](capacity=n)
    for i in range(n):
        cur_off = _align_up(cur_off, slab_align if len(stored_slabs[i]) >= 4096 and page_align else 64)
        offsets.append(cur_off)
        cur_off += len(stored_slabs[i])
    var total_bytes = _align_up(cur_off, 64)

    # Build Robin-Hood UUID table (32 B/slot) & Path-FNV64 table (16 B/slot)
    var u_hi = List[UInt64](length=hcap, fill=0)
    var u_lo = List[UInt64](length=hcap, fill=0)
    var u_toc = List[UInt32](length=hcap, fill=FLPK_EMPTY_SLOT_SENTINEL)
    var u_psl = List[UInt32](length=hcap, fill=0)  # 0 = empty, >= 1 = probe_dist + 1
    var u_sub = List[UInt16](length=hcap, fill=0)

    var p_fnv = List[UInt64](length=hcap, fill=0)
    var p_toc = List[UInt32](length=hcap, fill=FLPK_EMPTY_SLOT_SENTINEL)
    var p_psl = List[UInt16](length=hcap, fill=0)  # 0 = empty, >= 1 = probe_dist + 1
    var p_sub = List[UInt16](length=hcap, fill=0)

    for i in range(n):
        ref ch = chunks[i]
        # Insert into UUID Robin-Hood table
        var ins_hi = ch.uuid.hi
        var ins_lo = ch.uuid.lo
        var ins_toc = UInt32(i)
        var ins_sub = ch.sub_id
        var ins_psl = UInt32(1)
        var slot = Int(_robin_uuid_hash(ch.uuid, ch.sub_id)) & hmask
        for _ in range(hcap):
            if u_psl[slot] == 0:
                u_hi[slot] = ins_hi
                u_lo[slot] = ins_lo
                u_toc[slot] = ins_toc
                u_psl[slot] = ins_psl
                u_sub[slot] = ins_sub
                break
            if u_psl[slot] < ins_psl:
                var th = u_hi[slot]; var tl = u_lo[slot]; var tt = u_toc[slot]; var tp = u_psl[slot]; var ts = u_sub[slot]
                u_hi[slot] = ins_hi; u_lo[slot] = ins_lo; u_toc[slot] = ins_toc; u_psl[slot] = ins_psl; u_sub[slot] = ins_sub
                ins_hi = th; ins_lo = tl; ins_toc = tt; ins_psl = tp; ins_sub = ts
            slot = (slot + 1) & hmask
            ins_psl += 1

        # Insert into Path-FNV64 Robin-Hood table
        var fnv = fnv64_str(ch.rel_path)
        var pk_fnv = fnv
        var pk_toc = UInt32(i)
        var pk_sub = ch.sub_id
        var pk_psl = UInt16(1)
        var pslot = Int(_robin_path_hash(fnv, ch.sub_id)) & hmask
        for _ in range(hcap):
            if p_psl[pslot] == 0:
                p_fnv[pslot] = pk_fnv
                p_toc[pslot] = pk_toc
                p_psl[pslot] = pk_psl
                p_sub[pslot] = pk_sub
                break
            if p_psl[pslot] < pk_psl:
                var tf = p_fnv[pslot]; var tt2 = p_toc[pslot]; var tp2 = p_psl[pslot]; var ts2 = p_sub[pslot]
                p_fnv[pslot] = pk_fnv; p_toc[pslot] = pk_toc; p_psl[pslot] = pk_psl; p_sub[pslot] = pk_sub
                pk_fnv = tf; pk_toc = tt2; pk_psl = tp2; pk_sub = ts2
            pslot = (pslot + 1) & hmask
            pk_psl += 1

    # Emit header [0..64] (with placeholder toc_crc64 at [56..64])
    var out = List[UInt8](capacity=total_bytes)
    _put_u32_le(out, FLPK_V3_MAGIC)
    _put_u32_le(out, FLPK_V3_VERSION)
    _put_u32_le(out, FLPK_FLAG_PAGE_ALIGNED if page_align else UInt32(0))
    _put_u32_le(out, UInt32(n))
    _put_u64_le(out, UInt64(toc_off))
    _put_u64_le(out, UInt64(uuid_tbl_off))
    _put_u64_le(out, UInt64(path_tbl_off))
    _put_u32_le(out, UInt32(hcap))
    _put_u32_le(out, UInt32(len(str_pool)))
    _put_u64_le(out, UInt64(total_bytes))
    _put_u64_le(out, UInt64(0))  # filled below with ECMA-182 CRC64

    # Emit 64-byte TOC entries
    for i in range(n):
        ref ch = chunks[i]
        var slab_crc = crc64_ecma(stored_slabs[i])
        _put_u64_le(out, ch.uuid.hi)
        _put_u64_le(out, ch.uuid.lo)
        _put_u64_le(out, fnv64_str(ch.rel_path))
        _put_u64_le(out, UInt64(offsets[i]))
        _put_u64_le(out, UInt64(len(stored_slabs[i])))
        _put_u64_le(out, UInt64(len(ch.raw_payload)))
        _put_u64_le(out, slab_crc)
        _put_u16_le(out, ch.sub_id)
        out.append(ch.asset_kind)
        out.append(comp_modes[i])
        _put_u32_le(out, str_packed[i])

    # Emit UUID Robin-Hood table (32 B/slot)
    for s in range(hcap):
        _put_u64_le(out, u_hi[s])
        _put_u64_le(out, u_lo[s])
        _put_u32_le(out, u_toc[s])
        _put_u32_le(out, u_psl[s])
        _put_u16_le(out, u_sub[s])
        _put_u16_le(out, 0)
        _put_u32_le(out, 0)

    # Emit Path-FNV64 Robin-Hood table (16 B/slot)
    for s in range(hcap):
        _put_u64_le(out, p_fnv[s])
        _put_u32_le(out, p_toc[s])
        _put_u16_le(out, p_psl[s])
        _put_u16_le(out, p_sub[s])

    # Emit string pool + pad to `meta_end`
    for b in str_pool:
        out.append(b)
    while len(out) < meta_end:
        out.append(0)

    # Compute ECMA-182 CRC64 over header[0..56] + TOC + Robin-Hood tables + string pool!
    var hdr_crc = crc64_ecma_ptr(address_of(out), 56, 0)
    var meta_crc = crc64_ecma_ptr(address_of(out) + 64, meta_end - 64, hdr_crc)
    _write_u64_at(out, 56, meta_crc)

    # Emit aligned payload slabs
    for i in range(n):
        while len(out) < offsets[i]:
            out.append(0)
        for b in stored_slabs[i]:
            out.append(b)
    while len(out) < total_bytes:
        out.append(0)
    return out^


def write_flpk_v3(output_path: String, chunks: List[FlpkBuildChunk], page_align: Bool = True) -> Bool:
    var img = encode_flpk_v3(chunks, page_align)
    return write_bytes(output_path, img)


struct FlpkV3MappedArchive(Movable):
    """Persistent `mmap`-backed `.flpk` v3 archive with O(1) Robin-Hood UUID & Path-FNV64 lookup,
    hostile-input bounds/overflow verification, and zero per-read `mmap`/`munmap` allocations."""
    var path: String
    var fd: Int
    var addr: Int
    var size: Int
    var entry_count: Int
    var valid: Bool
    var header: FlpkV3Header
    var error_note: String

    def __init__(out self):
        self.path = String()
        self.fd = -1
        self.addr = 0
        self.size = 0
        self.entry_count = 0
        self.valid = False
        self.header = FlpkV3Header()
        self.error_note = String()

    def __init__(out self, *, deinit move: Self):
        self.path = move.path^
        self.fd = move.fd
        self.addr = move.addr
        self.size = move.size
        self.entry_count = move.entry_count
        self.valid = move.valid
        self.header = move.header
        self.error_note = move.error_note^

    def __deinit__(deinit self):
        if self.addr != 0 and self.size > 0:
            _ = external_call["munmap", Int32](Int(self.addr), Int64(self.size))
        if self.fd >= 0:
            close_fd(self.fd)

    def close(mut self):
        if self.addr != 0 and self.size > 0:
            _ = external_call["munmap", Int32](Int(self.addr), Int64(self.size))
            self.addr = 0
        if self.fd >= 0:
            close_fd(self.fd)
            self.fd = -1
        self.size = 0
        self.entry_count = 0
        self.valid = False

    @staticmethod
    def open(pack_path: String) -> Self:
        var arch = Self()
        arch.path = pack_path
        var st = file_stamp(pack_path)
        if st[0] < FLPK_HEADER_BYTES:
            arch.error_note = "file missing or smaller than 64B FlpkV3Header"
            return arch^
        arch.size = st[0]
        arch.fd = open_read(pack_path)
        if arch.fd < 0:
            arch.size = 0
            arch.error_note = "open_read failed"
            return arch^
        # PROT_READ = 1, MAP_PRIVATE = 2
        var ptr = external_call["mmap", Int](Int(0), Int64(arch.size), Int32(1), Int32(2), Int32(arch.fd), Int64(0))
        if ptr == 0 or ptr == -1:
            close_fd(arch.fd)
            arch.fd = -1
            arch.size = 0
            arch.error_note = "mmap failed"
            return arch^
        arch.addr = ptr
        arch._validate_mapped_header()
        if not arch.valid:
            var saved_note = arch.error_note
            arch.close()
            arch.error_note = saved_note
        return arch^

    def _validate_mapped_header(mut self):
        self.valid = False
        self.entry_count = 0
        if self.addr == 0 or self.size < FLPK_HEADER_BYTES:
            self.error_note = "unmapped or undersized archive"
            return

        var mapped_size = UInt64(self.size)
        var h = FlpkV3Header()
        h.magic = _ptr_u32(self.addr, 0)
        h.version = _ptr_u32(self.addr, 4)
        h.flags = _ptr_u32(self.addr, 8)
        h.toc_count = _ptr_u32(self.addr, 12)
        h.toc_offset = _ptr_u64(self.addr, 16)
        h.uuid_table_offset = _ptr_u64(self.addr, 24)
        h.path_table_offset = _ptr_u64(self.addr, 32)
        h.hash_table_capacity = _ptr_u32(self.addr, 40)
        h.string_pool_bytes = _ptr_u32(self.addr, 44)
        h.total_file_bytes = _ptr_u64(self.addr, 48)
        h.toc_crc64 = _ptr_u64(self.addr, 56)

        if h.magic != FLPK_V3_MAGIC or h.version != FLPK_V3_VERSION:
            self.error_note = "bad FLPK v3 magic or version"
            return

        if h.total_file_bytes < UInt64(FLPK_HEADER_BYTES) or h.total_file_bytes > mapped_size:
            self.error_note = "truncated or invalid total_file_bytes"
            return

        var eff_size = h.total_file_bytes
        var entry_cnt_u64 = UInt64(h.toc_count)
        var hcap_u64 = UInt64(h.hash_table_capacity)

        if entry_cnt_u64 > FLPK_MAX_ENTRIES:
            self.error_note = "toc_count exceeds FLPK_MAX_ENTRIES (1048576)"
            return
        if hcap_u64 > FLPK_MAX_HASH_CAP:
            self.error_note = "hash_table_capacity exceeds FLPK_MAX_HASH_CAP (2097152)"
            return
        if entry_cnt_u64 == 0:
            if hcap_u64 != 0:
                self.error_note = "hash_table_capacity must be 0 when toc_count is 0"
                return
        else:
            if hcap_u64 < entry_cnt_u64 or (hcap_u64 & (hcap_u64 - 1)) != 0:
                self.error_note = "invalid Robin-Hood table capacity"
                return

        # Since entry_cnt_u64 <= 1_048_576 and hcap_u64 <= 2_097_152, these products are <= 64 MiB (zero overflow).
        var toc_bytes = entry_cnt_u64 * UInt64(FLPK_TOC_ENTRY_BYTES)
        var uuid_bytes = hcap_u64 * UInt64(FLPK_UUID_SLOT_BYTES)
        var path_bytes = hcap_u64 * UInt64(FLPK_PATH_SLOT_BYTES)
        var str_bytes = UInt64(h.string_pool_bytes)

        # Validate every metadata table range BEFORE touching memory or computing CRC64!
        if h.toc_offset < UInt64(FLPK_HEADER_BYTES) or not _range_within_u64(h.toc_offset, toc_bytes, eff_size):
            self.error_note = "toc_offset range out of bounds"
            return
        if h.uuid_table_offset < UInt64(FLPK_HEADER_BYTES) or not _range_within_u64(h.uuid_table_offset, uuid_bytes, eff_size):
            self.error_note = "uuid_table_offset range out of bounds"
            return
        if h.path_table_offset < UInt64(FLPK_HEADER_BYTES) or not _range_within_u64(h.path_table_offset, path_bytes, eff_size):
            self.error_note = "path_table_offset range out of bounds"
            return

        # Safe: path_table_offset + path_bytes <= eff_size <= mapped_size
        var str_off = h.path_table_offset + path_bytes
        if not _range_within_u64(str_off, str_bytes, eff_size):
            self.error_note = "string_pool range out of bounds"
            return

        var str_end = str_off + str_bytes
        var pad64 = (UInt64(64) - (str_end & UInt64(63))) & UInt64(63)
        if not _range_within_u64(str_end, pad64, eff_size):
            self.error_note = "meta_end alignment out of bounds"
            return
        var meta_end = str_end + pad64
        if meta_end < UInt64(FLPK_HEADER_BYTES) or meta_end > eff_size or meta_end > mapped_size:
            self.error_note = "metadata region out of bounds"
            return

        # Every metadata table must also reside within [64, meta_end] so toc_crc64 covers it completely!
        if (
            not _range_within_u64(h.toc_offset, toc_bytes, meta_end)
            or not _range_within_u64(h.uuid_table_offset, uuid_bytes, meta_end)
            or not _range_within_u64(h.path_table_offset, path_bytes, meta_end)
            or not _range_within_u64(str_off, str_bytes, meta_end)
        ):
            self.error_note = "metadata table escapes CRC64-protected region"
            return

        var c0 = crc64_ecma_ptr(self.addr, 56, 0)
        var calc_crc = crc64_ecma_ptr(self.addr + 64, Int(meta_end - UInt64(64)), c0)
        if calc_crc != h.toc_crc64:
            self.error_note = "TOC ECMA-182 CRC64 mismatch"
            return

        # Validate every FlpkTocEntry payload range, uncompressed cap (<= 1 GiB), compression & string slice!
        var n_entries = Int(entry_cnt_u64)
        var toc_base = self.addr + Int(h.toc_offset)
        for i in range(n_entries):
            var e_base = toc_base + i * FLPK_TOC_ENTRY_BYTES
            var p_off = _ptr_u64(e_base, 24)
            var c_bytes = _ptr_u64(e_base, 32)
            var u_bytes = _ptr_u64(e_base, 40)
            var comp = _ptr_u8(e_base, 59)
            var p_str = _ptr_u32(e_base, 60)

            if not _range_within_u64(p_off, c_bytes, eff_size) or not _range_within_u64(p_off, c_bytes, mapped_size):
                self.error_note = "TOC entry payload range out of bounds"
                return
            if u_bytes > FLPK_MAX_UNCOMPRESSED_BYTES:
                self.error_note = "TOC entry uncompressed_bytes exceeds 1 GiB limit"
                return
            if comp != CHUNK_COMP_NONE and comp != CHUNK_COMP_LZ4:
                self.error_note = "TOC entry has unknown compression mode"
                return
            if comp == CHUNK_COMP_NONE and c_bytes != u_bytes:
                self.error_note = "CHUNK_COMP_NONE entry has mismatched compressed/uncompressed sizes"
                return

            var s_off = UInt64(p_str & 0xFFFFF)
            var s_len = UInt64((p_str >> 20) & 0xFFF)
            if not _range_within_u64(s_off, s_len, str_bytes):
                self.error_note = "TOC entry string pool slice out of bounds"
                return

        # Validate every Robin-Hood UUID & Path hash slot index (< entry_count or empty 0xFFFFFFFF)!
        var hcap = Int(hcap_u64)
        var uuid_base = self.addr + Int(h.uuid_table_offset)
        var path_base = self.addr + Int(h.path_table_offset)
        for s in range(hcap):
            var u_addr = uuid_base + s * FLPK_UUID_SLOT_BYTES
            var u_toc = _ptr_u32(u_addr, 16)
            var u_psl = _ptr_u32(u_addr, 20)
            if u_psl == 0:
                if u_toc != FLPK_EMPTY_SLOT_SENTINEL and UInt64(u_toc) >= entry_cnt_u64:
                    self.error_note = "corrupt empty UUID hash slot index"
                    return
            else:
                if UInt64(u_toc) >= entry_cnt_u64 or UInt64(u_psl) > hcap_u64:
                    self.error_note = "UUID hash slot index out of bounds"
                    return

            var p_addr = path_base + s * FLPK_PATH_SLOT_BYTES
            var p_toc = _ptr_u32(p_addr, 8)
            var p_psl = _ptr_u16(p_addr, 12)
            if p_psl == 0:
                if p_toc != FLPK_EMPTY_SLOT_SENTINEL and UInt64(p_toc) >= entry_cnt_u64:
                    self.error_note = "corrupt empty Path hash slot index"
                    return
            else:
                if UInt64(p_toc) >= entry_cnt_u64 or UInt64(p_psl) > hcap_u64:
                    self.error_note = "Path hash slot index out of bounds"
                    return

        self.header = h
        self.entry_count = n_entries
        self.valid = True

    @always_inline
    def toc_count(self) -> Int:
        return self.entry_count if self.valid else 0

    def toc_entry(self, toc_idx: Int) -> FlpkTocEntry:
        var e = FlpkTocEntry()
        if not self.valid or self.addr == 0 or toc_idx < 0 or toc_idx >= self.entry_count:
            return e
        var entry_off = Int(self.header.toc_offset) + toc_idx * FLPK_TOC_ENTRY_BYTES
        if not _range_within(entry_off, FLPK_TOC_ENTRY_BYTES, self.size):
            return e
        var base = self.addr + entry_off
        e.uuid = AssetUuid(_ptr_u64(base, 0), _ptr_u64(base, 8))
        e.path_fnv64 = _ptr_u64(base, 16)
        e.payload_offset = _ptr_u64(base, 24)
        e.compressed_bytes = _ptr_u64(base, 32)
        e.uncompressed_bytes = _ptr_u64(base, 40)
        e.payload_crc64 = _ptr_u64(base, 48)
        e.sub_id = _ptr_u16(base, 56)
        e.asset_kind = _ptr_u8(base, 58)
        e.compression = _ptr_u8(base, 59)
        e.path_str_offset = _ptr_u32(base, 60)
        return e

    def entry_path(self, toc_idx: Int) -> String:
        if not self.valid or self.addr == 0 or toc_idx < 0 or toc_idx >= self.entry_count:
            return String()
        var e = self.toc_entry(toc_idx)
        var off = UInt64(e.path_str_offset & 0xFFFFF)
        var slen = UInt64((e.path_str_offset >> 20) & 0xFFF)
        var str_bytes = UInt64(self.header.string_pool_bytes)
        if slen == 0 or not _range_within_u64(off, slen, str_bytes):
            return String()
        var str_pool_off = self.header.path_table_offset + UInt64(self.header.hash_table_capacity) * UInt64(FLPK_PATH_SLOT_BYTES)
        if not _range_within_u64(str_pool_off, str_bytes, UInt64(self.size)):
            return String()
        var str_base = self.addr + Int(str_pool_off + off)
        var n_len = Int(slen)
        var b = List[UInt8](capacity=n_len)
        for i in range(n_len):
            b.append(_ptr_u8(str_base, i))
        return String(unsafe_from_utf8=b)

    def find_by_uuid(self, uuid: AssetUuid, sub_id: UInt16 = 0) -> Int:
        """O(1) Robin-Hood lookup in the mapped UUID table; returns `toc_idx` or `-1`."""
        if not self.valid or self.addr == 0 or self.entry_count <= 0:
            return -1
        var hcap = Int(self.header.hash_table_capacity)
        if hcap <= 0 or (hcap & (hcap - 1)) != 0:
            return -1
        var tbl_off = Int(self.header.uuid_table_offset)
        if not _range_within(tbl_off, hcap * FLPK_UUID_SLOT_BYTES, self.size):
            return -1
        var hmask = hcap - 1
        var slot = Int(_robin_uuid_hash(uuid, sub_id)) & hmask
        var psl = UInt32(1)
        var tbl_base = self.addr + tbl_off
        for _ in range(hcap):
            var s_addr = tbl_base + slot * FLPK_UUID_SLOT_BYTES
            var s_psl = _ptr_u32(s_addr, 20)
            if s_psl == 0 or psl > s_psl:
                return -1
            if _ptr_u64(s_addr, 0) == uuid.hi and _ptr_u64(s_addr, 8) == uuid.lo and _ptr_u16(s_addr, 24) == sub_id:
                var raw_idx = _ptr_u32(s_addr, 16)
                if raw_idx == FLPK_EMPTY_SLOT_SENTINEL or Int(raw_idx) >= self.entry_count:
                    return -1
                return Int(raw_idx)
            slot = (slot + 1) & hmask
            psl += 1
        return -1

    def find_by_path(self, rel_path: String, sub_id: UInt16 = 0) -> Int:
        """O(1) Robin-Hood lookup in the mapped Path-FNV64 table; returns `toc_idx` or `-1`."""
        if not self.valid or self.addr == 0 or self.entry_count <= 0:
            return -1
        var hcap = Int(self.header.hash_table_capacity)
        if hcap <= 0 or (hcap & (hcap - 1)) != 0:
            return -1
        var tbl_off = Int(self.header.path_table_offset)
        if not _range_within(tbl_off, hcap * FLPK_PATH_SLOT_BYTES, self.size):
            return -1
        var fnv = fnv64_str(rel_path)
        var hmask = hcap - 1
        var slot = Int(_robin_path_hash(fnv, sub_id)) & hmask
        var psl = UInt16(1)
        var tbl_base = self.addr + tbl_off
        for _ in range(hcap):
            var s_addr = tbl_base + slot * FLPK_PATH_SLOT_BYTES
            var s_psl = _ptr_u16(s_addr, 12)
            if s_psl == 0 or psl > s_psl:
                return -1
            if _ptr_u64(s_addr, 0) == fnv and _ptr_u16(s_addr, 14) == sub_id:
                var raw_idx = _ptr_u32(s_addr, 8)
                if raw_idx == FLPK_EMPTY_SLOT_SENTINEL or Int(raw_idx) >= self.entry_count:
                    return -1
                return Int(raw_idx)
            slot = (slot + 1) & hmask
            psl += 1
        return -1

    def verify_entry_crc64(self, toc_idx: Int) -> Bool:
        if not self.valid or self.addr == 0 or toc_idx < 0 or toc_idx >= self.entry_count:
            return False
        var e = self.toc_entry(toc_idx)
        if not _range_within_u64(e.payload_offset, e.compressed_bytes, UInt64(self.size)):
            return False
        var off = Int(e.payload_offset)
        var csz = Int(e.compressed_bytes)
        var actual = crc64_ecma_ptr(self.addr + off, csz, 0)
        return actual == e.payload_crc64

    def mmap_slice_by_toc(self, toc_idx: Int, verify_crc: Bool = True) -> FlpkZeroCopySlice:
        """Returns a direct zero-copy `mmap` pointer slice for `CHUNK_COMP_NONE` entries."""
        var sl = FlpkZeroCopySlice()
        if not self.valid or self.addr == 0 or toc_idx < 0 or toc_idx >= self.entry_count:
            return sl
        var e = self.toc_entry(toc_idx)
        if e.compression != CHUNK_COMP_NONE:
            return sl
        if not _range_within_u64(e.payload_offset, e.compressed_bytes, UInt64(self.size)):
            return sl
        var off = Int(e.payload_offset)
        var csz = Int(e.compressed_bytes)
        if verify_crc and crc64_ecma_ptr(self.addr + off, csz, 0) != e.payload_crc64:
            return sl
        sl.addr = self.addr + off
        sl.size = csz
        sl.valid = True
        return sl

    def mmap_slice_by_uuid(self, uuid: AssetUuid, sub_id: UInt16 = 0, verify_crc: Bool = True) -> FlpkZeroCopySlice:
        return self.mmap_slice_by_toc(self.find_by_uuid(uuid, sub_id), verify_crc)

    def read_payload_by_toc(self, toc_idx: Int, verify_crc: Bool = True) -> Tuple[Bool, List[UInt8]]:
        """Reads (or LZ4-decompresses) TOC entry `toc_idx`, rejecting any CRC64 or frame corruption."""
        if not self.valid or self.addr == 0 or toc_idx < 0 or toc_idx >= self.entry_count:
            return (False, List[UInt8]())
        var e = self.toc_entry(toc_idx)
        if (
            not _range_within_u64(e.payload_offset, e.compressed_bytes, UInt64(self.size))
            or e.uncompressed_bytes > FLPK_MAX_UNCOMPRESSED_BYTES
        ):
            return (False, List[UInt8]())
        var off = Int(e.payload_offset)
        var csz = Int(e.compressed_bytes)
        var usz = Int(e.uncompressed_bytes)
        if verify_crc:
            var actual_crc = crc64_ecma_ptr(self.addr + off, csz, 0)
            if actual_crc != e.payload_crc64:
                return (False, List[UInt8]())
        if usz == 0:
            return (True, List[UInt8]())
        if e.compression == CHUNK_COMP_NONE:
            if csz != usz:
                return (False, List[UInt8]())
            var out = List[UInt8](capacity=usz)
            out.resize(unsafe_uninit_length=usz)
            _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), self.addr + off, Int64(usz))
            return (True, out^)
        elif e.compression == CHUNK_COMP_LZ4:
            if csz == 0 or UInt64(usz) > UInt64(csz) * 256:
                return (False, List[UInt8]())
            var out = List[UInt8](capacity=usz + 1)
            out.resize(unsafe_uninit_length=usz)
            var got = lz4_decompress_into(self.addr + off, csz, Int(out.unsafe_ptr()), usz)
            if got != usz:
                return (False, List[UInt8]())
            return (True, out^)
        return (False, List[UInt8]())

    def read_entry_bytes(self, idx: Int, verify_crc: Bool = True) -> List[UInt8]:
        """Safe wrapper returning decoded payload bytes for TOC index `idx` (empty if invalid)."""
        if not self.valid or self.addr == 0 or idx < 0 or idx >= self.entry_count:
            return List[UInt8]()
        var res = self.read_payload_by_toc(idx, verify_crc)
        return res[1].copy() if res[0] else List[UInt8]()

    def read_by_uuid(self, uuid: AssetUuid, sub_id: UInt16 = 0, verify_crc: Bool = True) -> Tuple[Bool, List[UInt8]]:
        return self.read_payload_by_toc(self.find_by_uuid(uuid, sub_id), verify_crc)

    def read_by_path(self, rel_path: String, sub_id: UInt16 = 0, verify_crc: Bool = True) -> Tuple[Bool, List[UInt8]]:
        return self.read_payload_by_toc(self.find_by_path(rel_path, sub_id), verify_crc)


# ============================================================================
# 7. `CookedMeshSlabHeader` (`AaaGeometryPool`), Texture Mip Slabs & `Bvh8` Slab
# ============================================================================

@fieldwise_init
struct CookedMeshSlabHeader(Copyable, ImplicitlyCopyable, Movable):
    """64-byte header for pre-built `AaaGeometryPool` slabs (`vertex_words`, `index_words`, `meshlet_words` + QEM `LOD0..3`)."""
    var magic: UInt32
    var version: UInt32
    var mesh_count: UInt32
    var meshlet_cluster_count: UInt32
    var vertex_word_count: UInt32
    var index_word_count: UInt32
    var meshlet_word_count: UInt32
    var max_lod_count: UInt32
    var bounds_center_x: Float32
    var bounds_center_y: Float32
    var bounds_center_z: Float32
    var bounds_radius: Float32
    var reserved0: UInt64
    var reserved1: UInt64

    def __init__(out self):
        self.magic = COOKED_MESH_SLAB_MAGIC
        self.version = 1
        self.mesh_count = 0
        self.meshlet_cluster_count = 0
        self.vertex_word_count = 0
        self.index_word_count = 0
        self.meshlet_word_count = 0
        self.max_lod_count = 1
        self.bounds_center_x = 0.0
        self.bounds_center_y = 0.0
        self.bounds_center_z = 0.0
        self.bounds_radius = 0.0
        self.reserved0 = 0
        self.reserved1 = 0


def encode_cooked_mesh_slab(pool: AaaGeometryPool) -> List[UInt8]:
    """Serializes a pre-cooked `AaaGeometryPool` (`LOD 0..3` QEM meshlets + GPU SSBO words) into an aligned binary slab."""
    var max_lods = 1
    var cx = Float32(0.0); var cy = Float32(0.0); var cz = Float32(0.0); var rad = Float32(0.0)
    if len(pool.meshes) > 0:
        cx = pool.meshes[0].bounds_center.x
        cy = pool.meshes[0].bounds_center.y
        cz = pool.meshes[0].bounds_center.z
        rad = pool.meshes[0].bounds_radius
        for i in range(len(pool.meshes)):
            if pool.meshes[i].lod_count > max_lods:
                max_lods = pool.meshes[i].lod_count

    var est = (
        64
        + len(pool.meshes) * 112
        + len(pool.meshlets) * 44
        + (len(pool.vertex_words) + len(pool.index_words) + len(pool.meshlet_words)) * 4
    )
    var out = List[UInt8](capacity=est)
    # 64-byte CookedMeshSlabHeader
    _put_u32_le(out, COOKED_MESH_SLAB_MAGIC)
    _put_u32_le(out, 1)
    _put_u32_le(out, UInt32(len(pool.meshes)))
    _put_u32_le(out, UInt32(len(pool.meshlets)))
    _put_u32_le(out, UInt32(len(pool.vertex_words)))
    _put_u32_le(out, UInt32(len(pool.index_words)))
    _put_u32_le(out, UInt32(len(pool.meshlet_words)))
    _put_u32_le(out, UInt32(max_lods))
    _put_u32_le(out, float32_bits(cx))
    _put_u32_le(out, float32_bits(cy))
    _put_u32_le(out, float32_bits(cz))
    _put_u32_le(out, float32_bits(rad))
    _put_u64_le(out, 0)
    _put_u64_le(out, 0)

    for i in range(len(pool.meshes)):
        ref m = pool.meshes[i]
        _put_u32_le(out, UInt32(m.mesh_id))
        _put_u32_le(out, UInt32(m.material_id))
        _put_u32_le(out, UInt32(m.first_vertex))
        _put_u32_le(out, UInt32(m.vertex_count))
        _put_u32_le(out, UInt32(m.first_index))
        _put_u32_le(out, UInt32(m.triangle_count))
        _put_u32_le(out, UInt32(m.first_meshlet))
        _put_u32_le(out, UInt32(m.meshlet_count))
        _put_u32_le(out, UInt32(m.lod0_meshlet_count))
        _put_u32_le(out, UInt32(m.lod_count))
        _put_u32_le(out, float32_bits(m.bounds_center.x))
        _put_u32_le(out, float32_bits(m.bounds_center.y))
        _put_u32_le(out, float32_bits(m.bounds_center.z))
        _put_u32_le(out, float32_bits(m.bounds_radius))
        for l in range(4):
            _put_u32_le(out, UInt32(m.lods[l].first_meshlet))
            _put_u32_le(out, UInt32(m.lods[l].meshlet_count))
            _put_u32_le(out, UInt32(m.lods[l].triangle_count))
            _put_u32_le(out, float32_bits(m.lods[l].object_error))

    for i in range(len(pool.meshlets)):
        ref ml = pool.meshlets[i]
        _put_u32_le(out, float32_bits(ml.center.x))
        _put_u32_le(out, float32_bits(ml.center.y))
        _put_u32_le(out, float32_bits(ml.center.z))
        _put_u32_le(out, float32_bits(ml.radius))
        _put_u32_le(out, float32_bits(ml.cone_axis.x))
        _put_u32_le(out, float32_bits(ml.cone_axis.y))
        _put_u32_le(out, float32_bits(ml.cone_axis.z))
        _put_u32_le(out, float32_bits(ml.cone_sin))
        _put_u32_le(out, UInt32(ml.first_tri))
        _put_u32_le(out, UInt32(ml.tri_count))
        _put_u32_le(out, UInt32(ml.mesh_id))
        _put_u32_le(out, UInt32(ml.material_id))
        _put_u32_le(out, UInt32(ml.lod_level))
        _put_u32_le(out, float32_bits(ml.self_error))
        _put_u32_le(out, float32_bits(ml.parent_error))

    for w in pool.vertex_words:
        _put_u32_le(out, w)
    for w in pool.index_words:
        _put_u32_le(out, w)
    for w in pool.meshlet_words:
        _put_u32_le(out, w)
    return out^


def decode_cooked_mesh_slab_ptr_into(
    addr: Int, size: Int, mut pool: AaaGeometryPool
) -> Tuple[Bool, CookedMeshSlabHeader]:
    """Decodes a cooked mesh slab directly into `pool` with zero intermediate `clone_geometry_pool` copies."""
    var hdr = CookedMeshSlabHeader()
    if addr == 0 or size < 64:
        return (False, hdr)
    hdr.magic = _ptr_u32(addr, 0)
    hdr.version = _ptr_u32(addr, 4)
    if hdr.magic != COOKED_MESH_SLAB_MAGIC or hdr.version != 1:
        return (False, hdr)
    hdr.mesh_count = _ptr_u32(addr, 8)
    hdr.meshlet_cluster_count = _ptr_u32(addr, 12)
    hdr.vertex_word_count = _ptr_u32(addr, 16)
    hdr.index_word_count = _ptr_u32(addr, 20)
    hdr.meshlet_word_count = _ptr_u32(addr, 24)
    hdr.max_lod_count = _ptr_u32(addr, 28)
    hdr.bounds_center_x = float32_from_bits(_ptr_u32(addr, 32))
    hdr.bounds_center_y = float32_from_bits(_ptr_u32(addr, 36))
    hdr.bounds_center_z = float32_from_bits(_ptr_u32(addr, 40))
    hdr.bounds_radius = float32_from_bits(_ptr_u32(addr, 44))

    var mc_u64 = UInt64(hdr.mesh_count)
    var mlc_u64 = UInt64(hdr.meshlet_cluster_count)
    var vwc_u64 = UInt64(hdr.vertex_word_count)
    var iwc_u64 = UInt64(hdr.index_word_count)
    var mwc_u64 = UInt64(hdr.meshlet_word_count)
    if (
        mc_u64 > 65536
        or mlc_u64 > 4_194_304
        or vwc_u64 > 268_435_456
        or iwc_u64 > 268_435_456
        or mwc_u64 > 268_435_456
        or mwc_u64 != mlc_u64 * UInt64(AAA_MESHLET_WORDS)
    ):
        return (False, hdr)
    var needed_u64 = UInt64(64) + mc_u64 * 120 + mlc_u64 * 60 + (vwc_u64 + iwc_u64 + mwc_u64) * 4
    if needed_u64 > UInt64(size):
        return (False, hdr)
    var mc = Int(mc_u64)
    var mlc = Int(mlc_u64)
    var vwc = Int(vwc_u64)
    var iwc = Int(iwc_u64)
    var mwc = Int(mwc_u64)

    var cur = 64
    pool.meshes.clear()
    pool.meshes.reserve(mc)
    for _ in range(mc):
        var m = CookedMeshEntry()
        m.mesh_id = Int(_ptr_u32(addr, cur)); cur += 4
        m.material_id = Int(_ptr_u32(addr, cur)); cur += 4
        m.first_vertex = Int(_ptr_u32(addr, cur)); cur += 4
        m.vertex_count = Int(_ptr_u32(addr, cur)); cur += 4
        m.first_index = Int(_ptr_u32(addr, cur)); cur += 4
        m.triangle_count = Int(_ptr_u32(addr, cur)); cur += 4
        m.first_meshlet = Int(_ptr_u32(addr, cur)); cur += 4
        m.meshlet_count = Int(_ptr_u32(addr, cur)); cur += 4
        m.lod0_meshlet_count = Int(_ptr_u32(addr, cur)); cur += 4
        m.lod_count = Int(_ptr_u32(addr, cur)); cur += 4
        var bx = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        var by = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        var bz = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        m.bounds_center = Vec3(bx, by, bz)
        m.bounds_radius = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        for l in range(4):
            var fm = Int(_ptr_u32(addr, cur)); cur += 4
            var mc_l = Int(_ptr_u32(addr, cur)); cur += 4
            var tc_l = Int(_ptr_u32(addr, cur)); cur += 4
            var oe = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
            m.lods[l] = MeshLodSlice(fm, mc_l, tc_l, oe)
        pool.meshes.append(m^)

    pool.meshlets.clear()
    pool.meshlets.reserve(mlc)
    for _ in range(mlc):
        var ml = MeshletCluster()
        var mx = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        var my = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        var mz = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        ml.center = Vec3(mx, my, mz)
        ml.radius = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        var ax = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        var ay = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        var az = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        ml.cone_axis = Vec3(ax, ay, az)
        ml.cone_sin = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        ml.first_tri = Int(_ptr_u32(addr, cur)); cur += 4
        ml.tri_count = Int(_ptr_u32(addr, cur)); cur += 4
        ml.mesh_id = Int(_ptr_u32(addr, cur)); cur += 4
        ml.material_id = Int(_ptr_u32(addr, cur)); cur += 4
        ml.lod_level = Int(_ptr_u32(addr, cur)); cur += 4
        ml.self_error = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        ml.parent_error = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
        pool.meshlets.append(ml^)

    pool.vertex_words.clear()
    if vwc > 0:
        pool.vertex_words.resize(unsafe_uninit_length=vwc)
        _ = external_call["memcpy", Int](Int(pool.vertex_words.unsafe_ptr()), addr + cur, Int64(vwc * 4))
        cur += vwc * 4
    pool.index_words.clear()
    if iwc > 0:
        pool.index_words.resize(unsafe_uninit_length=iwc)
        _ = external_call["memcpy", Int](Int(pool.index_words.unsafe_ptr()), addr + cur, Int64(iwc * 4))
        cur += iwc * 4
    pool.meshlet_words.clear()
    if mwc > 0:
        pool.meshlet_words.resize(unsafe_uninit_length=mwc)
        _ = external_call["memcpy", Int](Int(pool.meshlet_words.unsafe_ptr()), addr + cur, Int64(mwc * 4))
        cur += mwc * 4
    return (True, hdr)


def decode_cooked_mesh_slab_ptr(addr: Int, size: Int) -> Tuple[Bool, CookedMeshSlabHeader, AaaGeometryPool]:
    var pool = AaaGeometryPool()
    var res = decode_cooked_mesh_slab_ptr_into(addr, size, pool)
    return (res[0], res[1], pool^)


def decode_cooked_mesh_slab_into(data: List[UInt8], mut pool: AaaGeometryPool) -> Tuple[Bool, CookedMeshSlabHeader]:
    if len(data) == 0:
        return (False, CookedMeshSlabHeader())
    var res = decode_cooked_mesh_slab_ptr_into(address_of(data), len(data), pool)
    _ = len(data)
    return res


def clone_geometry_pool(src: AaaGeometryPool) -> AaaGeometryPool:
    var dst = AaaGeometryPool()
    dst.meshes = src.meshes.copy()
    dst.meshlets = src.meshlets.copy()
    dst.vertex_words = src.vertex_words.copy()
    dst.index_words = src.index_words.copy()
    dst.meshlet_words = src.meshlet_words.copy()
    return dst^


def decode_cooked_mesh_slab(data: List[UInt8]) -> Tuple[Bool, CookedMeshSlabHeader, AaaGeometryPool]:
    var pool = AaaGeometryPool()
    var res = decode_cooked_mesh_slab_into(data, pool)
    return (res[0], res[1], pool^)


@fieldwise_init
struct CookedTextureTailHeader(Copyable, Movable):
    """Header stored in `sub_id = 0` texture tail chunk alongside packed tail mip bytes."""
    var magic: UInt32
    var width: UInt32
    var height: UInt32
    var mip_count: UInt16
    var tail_first_mip: UInt16
    var tex_format: UInt8
    var pad0: UInt8
    var pad1: UInt16
    var mip_sizes: Array[UInt32, 16]

    def __init__(out self):
        self.magic = COOKED_TEX_SLAB_MAGIC
        self.width = 0
        self.height = 0
        self.mip_count = 0
        self.tail_first_mip = 0
        self.tex_format = TEX_FMT_BC7_SRGB
        self.pad0 = 0
        self.pad1 = 0
        self.mip_sizes = Array[UInt32, 16](fill=0)


def encode_texture_tail_slab(
    width: Int,
    height: Int,
    tex_format: UInt8,
    tail_first_mip: Int,
    all_mips: List[List[UInt8]],
) -> List[UInt8]:
    var mcnt = min(16, len(all_mips))
    var tf = max(0, min(max(0, mcnt - 1), tail_first_mip))
    var out = List[UInt8]()
    _put_u32_le(out, COOKED_TEX_SLAB_MAGIC)
    _put_u32_le(out, UInt32(width))
    _put_u32_le(out, UInt32(height))
    _put_u16_le(out, UInt16(mcnt))
    _put_u16_le(out, UInt16(tf))
    out.append(tex_format)
    out.append(0)
    _put_u16_le(out, 0)
    for m in range(16):
        var sz = len(all_mips[m]) if m < mcnt else 0
        _put_u32_le(out, UInt32(sz))
    for m in range(tf, mcnt):
        for b in all_mips[m]:
            out.append(b)
    return out^


def decode_texture_tail_slab(data: List[UInt8]) -> Tuple[Bool, CookedTextureTailHeader, List[List[UInt8]]]:
    var hdr = CookedTextureTailHeader()
    var mips = List[List[UInt8]]()
    if len(data) < 84:
        return (False, hdr^, mips^)
    var addr = address_of(data)
    hdr.magic = _ptr_u32(addr, 0)
    if hdr.magic != COOKED_TEX_SLAB_MAGIC:
        return (False, hdr^, mips^)
    hdr.width = _ptr_u32(addr, 4)
    hdr.height = _ptr_u32(addr, 8)
    hdr.mip_count = _ptr_u16(addr, 12)
    hdr.tail_first_mip = _ptr_u16(addr, 14)
    hdr.tex_format = _ptr_u8(addr, 16)
    for m in range(16):
        hdr.mip_sizes[m] = _ptr_u32(addr, 20 + m * 4)
    var mcnt = Int(hdr.mip_count)
    var tf = Int(hdr.tail_first_mip)
    if mcnt <= 0 or mcnt > 16 or tf < 0 or tf >= mcnt:
        return (False, hdr^, mips^)
    for _ in range(mcnt):
        mips.append(List[UInt8]())
    var cur = 84
    for m in range(tf, mcnt):
        var sz = Int(hdr.mip_sizes[m])
        if not _range_within(cur, sz, len(data)):
            return (False, hdr^, List[List[UInt8]]())
        var sub = List[UInt8](capacity=sz)
        for k in range(sz):
            sub.append(data[cur + k])
        mips[m] = sub^
        cur += sz
    _ = len(data)
    return (True, hdr^, mips^)


def encode_bvh8_collider_slab(col: Bvh8MeshCollider) -> List[UInt8]:
    """Serializes pre-built `Bvh8MeshCollider` (`sub_id = 16`) for zero-SAH-rebuild static mesh loading."""
    var n_tri = len(col.triangles)
    var n_nodes = len(col.nodes)
    var out = List[UInt8](capacity=16 + n_tri * 52 + n_nodes * 256)
    _put_u32_le(out, COOKED_BVH8_SLAB_MAGIC)
    _put_u32_le(out, UInt32(n_tri))
    _put_u32_le(out, UInt32(n_nodes))
    _put_u32_le(out, bitcast[DType.uint32](Int32(col.root_node)))

    for i in range(n_tri):
        ref t = col.triangles[i]
        _put_u32_le(out, float32_bits(t.v0.x)); _put_u32_le(out, float32_bits(t.v0.y)); _put_u32_le(out, float32_bits(t.v0.z))
        _put_u32_le(out, float32_bits(t.v1.x)); _put_u32_le(out, float32_bits(t.v1.y)); _put_u32_le(out, float32_bits(t.v1.z))
        _put_u32_le(out, float32_bits(t.v2.x)); _put_u32_le(out, float32_bits(t.v2.y)); _put_u32_le(out, float32_bits(t.v2.z))
        _put_u32_le(out, float32_bits(t.normal.x)); _put_u32_le(out, float32_bits(t.normal.y)); _put_u32_le(out, float32_bits(t.normal.z))
        out.append(t.active_edges)
        out.append(0)
        _put_u16_le(out, t.material_id)

    for i in range(n_nodes):
        ref nd = col.nodes[i]
        for lane in range(8):
            _put_u32_le(out, float32_bits(nd.min_x[lane]))
            _put_u32_le(out, float32_bits(nd.min_y[lane]))
            _put_u32_le(out, float32_bits(nd.min_z[lane]))
            _put_u32_le(out, float32_bits(nd.max_x[lane]))
            _put_u32_le(out, float32_bits(nd.max_y[lane]))
            _put_u32_le(out, float32_bits(nd.max_z[lane]))
            _put_u32_le(out, bitcast[DType.uint32](nd.child_indices[lane]))
            _put_u32_le(out, bitcast[DType.uint32](nd.child_counts[lane]))
    return out^


def decode_bvh8_collider_slab(data: List[UInt8]) -> Tuple[Bool, Bvh8MeshCollider]:
    var col = Bvh8MeshCollider()
    if len(data) < 16:
        return (False, col^)
    var addr = address_of(data)
    if _ptr_u32(addr, 0) != COOKED_BVH8_SLAB_MAGIC:
        return (False, col^)
    var n_tri_u64 = UInt64(_ptr_u32(addr, 4))
    var n_nodes_u64 = UInt64(_ptr_u32(addr, 8))
    col.root_node = Int(bitcast[DType.int32](_ptr_u32(addr, 12)))
    if n_tri_u64 > 16_777_216 or n_nodes_u64 > 16_777_216:
        return (False, col^)
    var needed_bvh = UInt64(16) + n_tri_u64 * 52 + n_nodes_u64 * 256
    if needed_bvh > UInt64(len(data)):
        return (False, col^)
    var n_tri = Int(n_tri_u64)
    var n_nodes = Int(n_nodes_u64)
    var cur = 16
    col.triangles.reserve(n_tri)
    for _ in range(n_tri):
        var v0 = Vec3(float32_from_bits(_ptr_u32(addr, cur)), float32_from_bits(_ptr_u32(addr, cur + 4)), float32_from_bits(_ptr_u32(addr, cur + 8))); cur += 12
        var v1 = Vec3(float32_from_bits(_ptr_u32(addr, cur)), float32_from_bits(_ptr_u32(addr, cur + 4)), float32_from_bits(_ptr_u32(addr, cur + 8))); cur += 12
        var v2 = Vec3(float32_from_bits(_ptr_u32(addr, cur)), float32_from_bits(_ptr_u32(addr, cur + 4)), float32_from_bits(_ptr_u32(addr, cur + 8))); cur += 12
        var nrm = Vec3(float32_from_bits(_ptr_u32(addr, cur)), float32_from_bits(_ptr_u32(addr, cur + 4)), float32_from_bits(_ptr_u32(addr, cur + 8))); cur += 12
        var act = _ptr_u8(addr, cur)
        var mat_id = _ptr_u16(addr, cur + 2)
        cur += 4
        col.triangles.append(MeshTriangle(v0, v1, v2, nrm, act, mat_id))

    col.nodes.reserve(n_nodes)
    for _ in range(n_nodes):
        var nd = Bvh8Node()
        for lane in range(8):
            nd.min_x[lane] = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
            nd.min_y[lane] = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
            nd.min_z[lane] = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
            nd.max_x[lane] = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
            nd.max_y[lane] = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
            nd.max_z[lane] = float32_from_bits(_ptr_u32(addr, cur)); cur += 4
            nd.child_indices[lane] = bitcast[DType.int32](_ptr_u32(addr, cur)); cur += 4
            nd.child_counts[lane] = bitcast[DType.int32](_ptr_u32(addr, cur)); cur += 4
        col.nodes.append(nd)
    _ = len(data)

    if n_nodes == 0:
        if col.root_node >= 0:
            return (False, Bvh8MeshCollider())
    elif col.root_node < 0 or col.root_node >= n_nodes:
        return (False, Bvh8MeshCollider())

    for ni in range(n_nodes):
        ref nd = col.nodes[ni]
        for lane in range(8):
            var c_idx = Int(nd.child_indices[lane])
            var c_cnt = Int(nd.child_counts[lane])
            if c_cnt < 0:
                return (False, Bvh8MeshCollider())
            if c_cnt > 0:
                var leaf_first = -c_idx - 1
                if leaf_first < 0 or leaf_first > n_tri or c_cnt > n_tri - leaf_first:
                    return (False, Bvh8MeshCollider())
            else:
                if c_idx < 0 or c_idx >= n_nodes:
                    return (False, Bvh8MeshCollider())
                if c_idx > 0 and c_idx == ni:
                    return (False, Bvh8MeshCollider())

    if n_nodes > 0:
        var color = List[UInt8](length=n_nodes, fill=0)  # 0=unvisited, 1=visiting, 2=done
        var dfs_node = List[Int]()
        var dfs_lane = List[Int]()
        for start_n in range(n_nodes):
            var seed = col.root_node if start_n == 0 else start_n
            if color[seed] != 0:
                continue
            color[seed] = 1
            dfs_node.append(seed)
            dfs_lane.append(0)
            while len(dfs_node) > 0:
                var top = len(dfs_node) - 1
                var ni = dfs_node[top]
                var li = dfs_lane[top]
                if li >= 8:
                    color[ni] = 2
                    _ = dfs_node.pop()
                    _ = dfs_lane.pop()
                    continue
                dfs_lane[top] = li + 1
                var c_idx = Int(col.nodes[ni].child_indices[li])
                var c_cnt = Int(col.nodes[ni].child_counts[li])
                if c_cnt == 0 and c_idx > 0:
                    if color[c_idx] == 1:
                        return (False, Bvh8MeshCollider())
                    if color[c_idx] == 0:
                        color[c_idx] = 1
                        dfs_node.append(c_idx)
                        dfs_lane.append(0)

    return (True, col^)


def encode_audio_pcm_slab(channels: Int, sample_rate: Int, samples: List[Int16]) -> List[UInt8]:
    var out = List[UInt8](capacity=16 + len(samples) * 2)
    _put_u32_le(out, COOKED_AUD_SLAB_MAGIC)
    _put_u32_le(out, UInt32(max(1, channels)))
    _put_u32_le(out, UInt32(max(1, sample_rate)))
    _put_u32_le(out, UInt32(len(samples)))
    for s in samples:
        var u = UInt16(s)
        _put_u16_le(out, u)
    return out^


def decode_audio_pcm_slab(data: List[UInt8]) -> Tuple[Bool, Int, Int, List[Int16]]:
    if len(data) < 16:
        return (False, 0, 0, List[Int16]())
    var addr = address_of(data)
    if _ptr_u32(addr, 0) != COOKED_AUD_SLAB_MAGIC:
        return (False, 0, 0, List[Int16]())
    var ch = Int(_ptr_u32(addr, 4))
    var sr = Int(_ptr_u32(addr, 8))
    var cnt_u64 = UInt64(_ptr_u32(addr, 12))
    if (
        ch <= 0
        or ch > 8
        or sr <= 0
        or sr > 768_000
        or cnt_u64 == 0
        or cnt_u64 > 536_870_912
        or UInt64(16) + cnt_u64 * 2 > UInt64(len(data))
    ):
        return (False, 0, 0, List[Int16]())
    var cnt = Int(cnt_u64)
    if cnt % ch != 0:
        return (False, 0, 0, List[Int16]())
    var pcm = List[Int16](capacity=cnt)
    for i in range(cnt):
        var u = _ptr_u16(addr, 16 + i * 2)
        pcm.append(Int16(u))
    _ = len(data)
    return (True, ch, sr, pcm^)


def encode_cooked_rig_slab(rig_bytes: List[UInt8]) -> List[UInt8]:
    """Wraps or validates a `'FRIG'` (`COOKED_RIG_SLAB_MAGIC`) binary rig slab (`sub_id = 17`)."""
    if len(rig_bytes) >= 12:
        var addr = address_of(rig_bytes)
        var magic = _ptr_u32(addr, 0)
        _ = len(rig_bytes)
        if magic == COOKED_RIG_SLAB_MAGIC:
            return rig_bytes.copy()
    var out = List[UInt8](capacity=12 + len(rig_bytes))
    _put_u32_le(out, COOKED_RIG_SLAB_MAGIC)
    _put_u32_le(out, UInt32(1))
    _put_u32_le(out, UInt32(len(rig_bytes)))
    for b in rig_bytes:
        out.append(b)
    return out^


def decode_cooked_rig_slab(data: List[UInt8]) -> Tuple[Bool, List[UInt8]]:
    """Validates a `'FRIG'` binary rig slab (`sub_id = 17`) and returns its payload bytes."""
    if len(data) < 12:
        return (False, List[UInt8]())
    var addr = address_of(data)
    var magic = _ptr_u32(addr, 0)
    var ver = _ptr_u32(addr, 4)
    var plen = Int(_ptr_u32(addr, 8))
    _ = len(data)
    if magic != COOKED_RIG_SLAB_MAGIC or ver < 1 or plen < 0 or 12 + plen > len(data):
        return (False, List[UInt8]())
    return (True, data.copy())
