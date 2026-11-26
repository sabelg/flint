"""Project-relative `AssetRef` with UUID + path normalization and binary/JSON codecs."""
from flint.assets.flpk_v3 import ASSET_KIND_RAW, AssetUuid
from flint.assets.json import JsonDocument
from flint.assets.json_write import json_quote
from flint.core.serial import BlobReader, BlobWriter


def normalize_asset_rel_path(path: String) -> Tuple[Bool, String]:
    """Normalizes a project-relative asset path, rejecting `..` traversal and absolute paths."""
    var raw = path.as_bytes()
    if len(raw) == 0:
        return (True, String())

    # Reject NUL bytes, leading '/', leading '\', or Windows drive prefix (e.g. "C:")
    for i in range(len(raw)):
        if raw[i] == 0:
            return (False, String())
    if raw[0] == 47 or raw[0] == 92:
        return (False, String())
    if len(raw) >= 2 and raw[1] == 58:
        return (False, String())

    var segs = List[String]()
    var cur = List[UInt8]()
    for i in range(len(raw)):
        var b = raw[i]
        if b == 47 or b == 92:
            if len(cur) > 0:
                var seg = String(unsafe_from_utf8=cur)
                if seg == "..":
                    return (False, String())
                if seg != ".":
                    segs.append(seg)
                cur.clear()
        else:
            cur.append(b)
    if len(cur) > 0:
        var seg = String(unsafe_from_utf8=cur)
        if seg == "..":
            return (False, String())
        if seg != ".":
            segs.append(seg)

    var out = String()
    for i in range(len(segs)):
        if i > 0:
            out += "/"
        out += segs[i]
    return (True, out^)


def is_safe_asset_rel_path(path: String) -> Bool:
    return normalize_asset_rel_path(path)[0]


struct AssetRef(Copyable, Movable, Equatable):
    """Persistent project-relative asset reference (`uuid` + `rel_path` + `kind`)."""
    var uuid: AssetUuid
    var rel_path: String
    var kind: UInt8

    def __init__(out self):
        self.uuid = AssetUuid()
        self.rel_path = String()
        self.kind = ASSET_KIND_RAW

    def __init__(out self, uuid: AssetUuid, rel_path: String, kind: UInt8 = ASSET_KIND_RAW):
        var norm = normalize_asset_rel_path(rel_path)
        self.uuid = uuid
        self.rel_path = norm[1] if norm[0] else rel_path
        self.kind = kind

    @staticmethod
    def from_path(rel_path: String, kind: UInt8 = ASSET_KIND_RAW) -> Self:
        var norm = normalize_asset_rel_path(rel_path)
        if not norm[0] or norm[1].byte_length() == 0:
            return Self(AssetUuid(), String(), kind)
        return Self(AssetUuid.from_path(norm[1]), norm[1], kind)

    @always_inline
    def is_empty(self) -> Bool:
        return self.uuid.is_nil() and self.rel_path.byte_length() == 0

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.uuid == other.uuid and self.rel_path == other.rel_path and self.kind == other.kind

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def write_bin(self, mut out: BlobWriter):
        out.u64(self.uuid.hi)
        out.u64(self.uuid.lo)
        out.text(self.rel_path)
        out.u8(self.kind)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var hi = reader.u64()
        var lo = reader.u64()
        var rp = reader.text()
        var k = reader.u8()
        if reader.failed:
            raise Error("AssetRef: truncated binary blob")
        var norm = normalize_asset_rel_path(rp)
        if not norm[0]:
            raise Error("AssetRef: unsafe relative path in binary blob: " + rp)
        return Self(AssetUuid(hi, lo), norm[1], k)

    def write_json(self) -> String:
        return (
            "{\"uuid\": " + json_quote(self.uuid.format())
            + ", \"rel_path\": " + json_quote(self.rel_path)
            + ", \"kind\": " + String(Int(self.kind))
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        if node < 0:
            return Self()
        var kind_id = doc.nodes[node].kind
        if kind_id == 3:
            var rp_str = doc.text(node, "")
            var norm = normalize_asset_rel_path(rp_str)
            if not norm[0]:
                raise Error("AssetRef: unsafe relative path: " + rp_str)
            return Self.from_path(norm[1], ASSET_KIND_RAW)
        if kind_id != 5:
            raise Error("AssetRef: expected JSON object or string")
        var rp_node = doc.object_get(node, "rel_path")
        var rp = doc.text(rp_node, "") if rp_node >= 0 else String()
        var norm = normalize_asset_rel_path(rp)
        if not norm[0]:
            raise Error("AssetRef: unsafe relative path: " + rp)

        var u = AssetUuid()
        var u_node = doc.object_get(node, "uuid")
        if u_node >= 0:
            var u_str = doc.text(u_node, "")
            if u_str.byte_length() > 0:
                var parsed = AssetUuid.parse(u_str)
                if not parsed[0]:
                    raise Error("AssetRef: invalid UUID string: " + u_str)
                u = parsed[1]

        var k_node = doc.object_get(node, "kind")
        var k_int = doc.integer(k_node, Int(ASSET_KIND_RAW))
        var k = UInt8(max(0, min(255, k_int)))
        return Self(u, norm[1], k)
