"""Flint v5 Scene Graph, ComponentRegistry, Nested Prefabs & `.flscn` / `.flscnb` Persistence (`flint.scene.scene`).

Provides:
- Re-exports `trait SerializableComponent(Component)` (`SCHEMA_NAME`, `SCHEMA_FOURCC`,
  `SCHEMA_VERSION`, `write_bin`, `read_bin`, `write_json`, `read_json`)
- `ComponentRegistry`: type-erased binary + JSON component codec vtable
- `Prefab`, `PrefabNode`, `PrefabOverride`, `NestedPrefabRef`: composable nested prefabs with
  per-instance component overrides and cycle-guarded instantiation
- `Scene`: deterministic `.flscn` JSON (`assets/json.mojo` + `json_write.mojo`), compressed `.flscnb`
  LZ4 binary (`BlobWriter` + `lz4_pack`), and `<0.2 ms` Play-In-Editor (`enter_pie` / `exit_pie`)
  bit-exact world snapshot & restoration.
"""
from std.collections import List
from std.ffi import external_call
from std.math import isfinite, max
from std.memory import bitcast
from flint.app.hot_reload import (
    POD_FIELD_BOOL,
    POD_FIELD_COLOR_RGB,
    POD_FIELD_F32,
    POD_FIELD_INT,
    POD_FIELD_QUAT,
    POD_FIELD_VEC3,
    PodComponentSchema,
)
from flint.assets.asset_ref import AssetRef
from flint.assets.flpk_v3 import ASSET_KIND_AUDIO, ASSET_KIND_MESH, ASSET_KIND_RAW
from flint.core.detsort import stable_order
from flint.core.lz4 import lz4_pack, lz4_unpack
from flint.core.math import Vec3
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.core.transform import Quat
from flint.assets.json import JsonDocument, parse_json
from flint.assets.json_write import json_quote, json_text
from flint.platform.os import read_bytes, write_bytes, write_file_durable
from flint.ecs.world import (
    World,
    Entity,
    NULL_ENTITY,
    BitSet256,
    Component,
    STORAGE_SPARSE,
    STORAGE_TABLE,
    comp_hash64,
)
from flint.scene.transform_hierarchy import (
    SerializableComponent,
    LocalTransform,
    GlobalTransform,
    PrevGlobalTransform,
    Parent,
    register_transform_components,
    set_parent,
)

comptime SCENE_BIN_MAGIC: UInt32 = 0x4E435346  # "FSCN"
comptime SCENE_SCHEMA_VERSION: Int = 2
comptime MAX_PREFAB_NEST_DEPTH: Int = 16
comptime SCENE_UNKNOWN_JSON_FOURCC: UInt32 = fourcc("UKJS")
comptime _SCENE_UNKNOWN_BIN_KEY: StaticString = "__unknown_bin__"


def _bytes_to_hex(data: List[UInt8]) -> String:
    var out = List[UInt8](capacity=len(data) * 2)
    for i in range(len(data)):
        var b = Int(data[i])
        var hi = (b >> 4) & 0xF
        var lo = b & 0xF
        out.append(UInt8(48 + hi) if hi < 10 else UInt8(87 + hi))
        out.append(UInt8(48 + lo) if lo < 10 else UInt8(87 + lo))
    return String(unsafe_from_utf8=out)


def _hex_nibble(c: UInt8) raises -> Int:
    if c >= 48 and c <= 57:
        return Int(c - 48)
    if c >= 97 and c <= 102:
        return Int(c - 87)
    if c >= 65 and c <= 70:
        return Int(c - 55)
    raise Error("invalid hex character in __unknown_bin__")


def _hex_to_bytes(s: String) raises -> List[UInt8]:
    var raw = s.as_bytes()
    if (len(raw) % 2) != 0:
        raise Error("odd hex length in __unknown_bin__")
    var out = List[UInt8](capacity=len(raw) // 2)
    var i = 0
    while i < len(raw):
        var hi = _hex_nibble(raw[i])
        var lo = _hex_nibble(raw[i + 1])
        out.append(UInt8((hi << 4) | lo))
        i += 2
    return out^


def _parse_unknown_bin_json(
    doc: JsonDocument, node: Int, mut world: World, entity: Entity
) raises:
    if node < 0 or node >= len(doc.nodes) or doc.nodes[node].kind != 4:
        raise Error("invalid __unknown_bin__ array in .flscn")
    var n = doc.array_size(node)
    for i in range(n):
        var elem = doc.at(node, i)
        if elem < 0 or doc.nodes[elem].kind != 5:
            raise Error("invalid __unknown_bin__ entry object in .flscn")
        var fcc_n = doc.object_get(elem, "fourcc")
        var ver_n = doc.object_get(elem, "version")
        var hex_n = doc.object_get(elem, "hex")
        if (
            fcc_n < 0
            or doc.nodes[fcc_n].kind != 2
            or ver_n < 0
            or doc.nodes[ver_n].kind != 2
            or hex_n < 0
            or doc.nodes[hex_n].kind != 3
        ):
            raise Error("missing or invalid fields in __unknown_bin__ entry")
        var tag_i = doc.integer(fcc_n, -1)
        var ver_i = doc.integer(ver_n, -1)
        if tag_i < 0 or ver_i < 0:
            raise Error("negative fourcc/version in __unknown_bin__ entry")
        var payload = _hex_to_bytes(doc.text(hex_n, ""))
        var unk_ptr = world.get_ptr_mut[UnknownComponentsComponent](entity)
        if unk_ptr:
            unk_ptr.value()[unsafe_offset=0].add_bin(UInt32(tag_i), UInt32(ver_i), payload^)
        else:
            var unk = UnknownComponentsComponent()
            unk.add_bin(UInt32(tag_i), UInt32(ver_i), payload^)
            _ = world.insert[UnknownComponentsComponent](entity, unk^)


# ---------------------------------------------------------------------------
# Opaque Unknown-Component Preservation & Scene Migration Hooks
# ---------------------------------------------------------------------------

struct UnknownComponentsComponent(Component):
    """Stores unknown JSON component payloads and unregistered FourCC binary blobs losslessly."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.scene.UnknownComponentsComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_FOURCC: UInt32 = fourcc("UNKC")
    comptime SCHEMA_VERSION: Int = 1

    var json_names: List[String]
    var json_payloads: List[String]
    var bin_fourccs: List[UInt32]
    var bin_versions: List[UInt32]
    var bin_payloads: List[List[UInt8]]

    def __init__(out self):
        self.json_names = List[String]()
        self.json_payloads = List[String]()
        self.bin_fourccs = List[UInt32]()
        self.bin_versions = List[UInt32]()
        self.bin_payloads = List[List[UInt8]]()

    def __init__(out self, *, copy: Self):
        self.json_names = copy.json_names.copy()
        self.json_payloads = copy.json_payloads.copy()
        self.bin_fourccs = copy.bin_fourccs.copy()
        self.bin_versions = copy.bin_versions.copy()
        self.bin_payloads = copy.bin_payloads.copy()

    def add_json(mut self, name: String, payload: String):
        for i in range(len(self.json_names)):
            if self.json_names[i] == name:
                self.json_payloads[i] = payload
                return
        self.json_names.append(name)
        self.json_payloads.append(payload)

    def add_bin(mut self, tag: UInt32, version: UInt32, var payload: List[UInt8]):
        for i in range(len(self.bin_fourccs)):
            if self.bin_fourccs[i] == tag:
                self.bin_versions[i] = version
                self.bin_payloads[i] = payload^
                return
        self.bin_fourccs.append(tag)
        self.bin_versions.append(version)
        self.bin_payloads.append(payload^)

    def sorted_json_order(self) -> List[Int]:
        var n = len(self.json_names)
        var order = List[Int](capacity=n)
        for i in range(n):
            order.append(i)
        for i in range(1, n):
            var key_idx = order[i]
            var key_name = self.json_names[key_idx]
            var j = i - 1
            while j >= 0 and self.json_names[order[j]] > key_name:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key_idx
        return order^

    def sorted_bin_order(self) -> List[Int]:
        var n = len(self.bin_fourccs)
        var order = List[Int](capacity=n)
        for i in range(n):
            order.append(i)
        for i in range(1, n):
            var key_idx = order[i]
            var key_tag = self.bin_fourccs[key_idx]
            var key_ver = self.bin_versions[key_idx]
            var j = i - 1
            while j >= 0 and (
                self.bin_fourccs[order[j]] > key_tag
                or (self.bin_fourccs[order[j]] == key_tag and self.bin_versions[order[j]] > key_ver)
            ):
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key_idx
        return order^

    def write_bin(self, mut out: BlobWriter):
        var jorder = self.sorted_json_order()
        out.u32(UInt32(len(jorder)))
        for idx in jorder:
            out.text(self.json_names[idx])
            out.text(self.json_payloads[idx])
        var border = self.sorted_bin_order()
        out.u32(UInt32(len(border)))
        for idx in border:
            out.u32(self.bin_fourccs[idx])
            out.u32(self.bin_versions[idx])
            out.u32(UInt32(len(self.bin_payloads[idx])))
            if len(self.bin_payloads[idx]) > 0:
                out.write_bytes_raw(Int(self.bin_payloads[idx].unsafe_ptr()), len(self.bin_payloads[idx]))

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var out = Self()
        var nj = Int(reader.u32())
        if reader.failed or nj < 0 or nj > 65536 or nj > max(0, reader.limit() - reader.offset) // 8:
            raise Error("UnknownComponentsComponent: invalid json count")
        for _ in range(nj):
            var jn = reader.text()
            var jp = reader.text()
            if reader.failed:
                raise Error("UnknownComponentsComponent: truncated json entry")
            out.add_json(jn, jp)
        var nb = Int(reader.u32())
        if reader.failed or nb < 0 or nb > 65536 or nb > max(0, reader.limit() - reader.offset) // 12:
            raise Error("UnknownComponentsComponent: invalid bin count")
        for _ in range(nb):
            var tag = reader.u32()
            var bver = reader.u32()
            var blen = Int(reader.u32())
            if reader.failed or blen < 0 or blen > reader.limit() - reader.offset:
                raise Error("UnknownComponentsComponent: truncated bin payload")
            var pay = List[UInt8](capacity=blen)
            if blen > 0:
                pay.resize(unsafe_uninit_length=blen)
                if not reader.read_bytes_raw(Int(pay.unsafe_ptr()), blen):
                    raise Error("UnknownComponentsComponent: failed reading bin payload")
            out.add_bin(tag, bver, pay^)
        if reader.failed:
            raise Error("UnknownComponentsComponent: truncated blob")
        return out^


def _unknown_bin_write(comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[UnknownComponentsComponent, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.begin(UnknownComponentsComponent.SCHEMA_FOURCC, UnknownComponentsComponent.SCHEMA_VERSION)
    comp.write_bin(out)
    out.end()


def _unknown_bin_read(comp_addr: Int, reader_addr: Int) -> Bool:
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    try:
        var ver = reader.enter(UnknownComponentsComponent.SCHEMA_FOURCC, UnknownComponentsComponent.SCHEMA_VERSION)
        var val = UnknownComponentsComponent.read_bin(reader, ver)
        reader.leave()
        var dst = Pointer[UnknownComponentsComponent, MutAnyOrigin](unsafe_from_address=comp_addr)
        __get_address_as_uninit_lvalue(dst._mlir_value) = val^
        return True
    except:
        return False


def register_unknown_components(mut world: World):
    var tid = world.register_component[UnknownComponentsComponent]()
    if tid >= 0 and not world.components[tid].has_custom_bin:
        world.set_component_binary_hooks(tid, _unknown_bin_write, _unknown_bin_read)


comptime SceneJsonMigrationFn = def(String) thin raises -> String


@fieldwise_init
struct SceneMigrationEntry(Copyable, Movable):
    var from_version: Int
    var to_version: Int
    var hook: SceneJsonMigrationFn


struct SceneMigrationRegistry(Copyable, Movable):
    """Stores ordered schema-version migration hooks `(from_version -> to_version)` for `.flscn`."""
    var entries: List[SceneMigrationEntry]

    def __init__(out self):
        self.entries = List[SceneMigrationEntry]()

    def __init__(out self, *, copy: Self):
        self.entries = copy.entries.copy()

    def register(mut self, from_version: Int, to_version: Int, hook: SceneJsonMigrationFn):
        self.entries.append(SceneMigrationEntry(from_version, to_version, hook))

    def migrate_json(self, from_version: Int, to_version: Int, payload: String) raises -> String:
        var cur = payload
        var v = from_version
        while v < to_version:
            var applied = False
            for i in range(len(self.entries)):
                if self.entries[i].from_version == v and self.entries[i].to_version == v + 1:
                    cur = self.entries[i].hook(cur)
                    applied = True
                    break
            if not applied:
                pass
            v += 1
        return cur^


# ---------------------------------------------------------------------------
# ComponentRegistry: Type-Erased Binary + JSON Codec Vtable
# ---------------------------------------------------------------------------

comptime _CodecRegisterWorldFn = def(Int) thin -> Int
comptime _CodecHasFn = def(Int, Entity) thin -> Bool
comptime _CodecWriteBinFn = def(Int, Entity, Int) thin -> Bool
comptime _CodecReadBinFn = def(Int, Entity, Int, Int) thin -> Bool
comptime _CodecWriteJsonFn = def(Int, Entity, Int) thin -> Bool
comptime _CodecReadJsonFn = def(Int, Entity, Int, Int) thin -> Bool
comptime _CodecWriteBinSlotsFn = def(Int, Entity, Int, Int) thin -> Bool
comptime _CodecWriteJsonSlotsFn = def(Int, Entity, Int, Int) thin -> Bool
comptime _CodecResolveSlotsFn = def(Int, Entity, Int) thin -> None
comptime _CodecDefaultInsertFn = def(Int, Entity) thin -> Bool
comptime _CodecRemoveFn = def(Int, Entity) thin -> Bool


def _noop_reg_world(_w: Int) -> Int:
    return -1


def _noop_has(_w: Int, _e: Entity) -> Bool:
    return False


def _noop_write_bin(_w: Int, _e: Entity, _out: Int) -> Bool:
    return False


def _noop_read_bin(_w: Int, _e: Entity, _r: Int, _v: Int) -> Bool:
    return False


def _noop_write_json(_w: Int, _e: Entity, _out: Int) -> Bool:
    return False


def _noop_read_json(_w: Int, _e: Entity, _doc: Int, _n: Int) -> Bool:
    return False


def _noop_write_slots(_w: Int, _e: Entity, _m: Int, _out: Int) -> Bool:
    return False


def _noop_resolve_slots(_w: Int, _e: Entity, _sp: Int):
    pass


def _noop_default_insert(_w: Int, _e: Entity) -> Bool:
    return False


def _noop_remove(_w: Int, _e: Entity) -> Bool:
    return False


def _pod_read_u8(base_ptr: Int, offset: Int) -> UInt8:
    return Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=base_ptr + offset)[unsafe_offset=0]


def _pod_write_u8(base_ptr: Int, offset: Int, val: UInt8):
    Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base_ptr + offset)[unsafe_offset=0] = val


def _pod_read_u32(base_ptr: Int, offset: Int) -> UInt32:
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=base_ptr + offset)
    return (
        UInt32(p[unsafe_offset=0])
        | (UInt32(p[unsafe_offset=1]) << 8)
        | (UInt32(p[unsafe_offset=2]) << 16)
        | (UInt32(p[unsafe_offset=3]) << 24)
    )


def _pod_write_u32(base_ptr: Int, offset: Int, val: UInt32):
    var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base_ptr + offset)
    p[unsafe_offset=0] = UInt8(val & 0xFF)
    p[unsafe_offset=1] = UInt8((val >> 8) & 0xFF)
    p[unsafe_offset=2] = UInt8((val >> 16) & 0xFF)
    p[unsafe_offset=3] = UInt8((val >> 24) & 0xFF)


def _pod_read_i32(base_ptr: Int, offset: Int) -> Int32:
    return bitcast[DType.int32](_pod_read_u32(base_ptr, offset))


def _pod_write_i32(base_ptr: Int, offset: Int, val: Int32):
    _pod_write_u32(base_ptr, offset, bitcast[DType.uint32](val))


def _pod_read_u64(base_ptr: Int, offset: Int) -> UInt64:
    var lo = UInt64(_pod_read_u32(base_ptr, offset))
    var hi = UInt64(_pod_read_u32(base_ptr, offset + 4))
    return lo | (hi << 32)


def _pod_write_u64(base_ptr: Int, offset: Int, val: UInt64):
    _pod_write_u32(base_ptr, offset, UInt32(val & 0xFFFFFFFF))
    _pod_write_u32(base_ptr, offset + 4, UInt32((val >> 32) & 0xFFFFFFFF))


def _pod_read_i64(base_ptr: Int, offset: Int) -> Int64:
    return bitcast[DType.int64](_pod_read_u64(base_ptr, offset))


def _pod_write_i64(base_ptr: Int, offset: Int, val: Int64):
    _pod_write_u64(base_ptr, offset, bitcast[DType.uint64](val))


def _pod_read_f32(base_ptr: Int, offset: Int) -> Float32:
    var v = bitcast[DType.float32](_pod_read_u32(base_ptr, offset))
    return v if isfinite(v) else Float32(0.0)


def _pod_write_f32(base_ptr: Int, offset: Int, val: Float32):
    var safe = val if isfinite(val) else Float32(0.0)
    _pod_write_u32(base_ptr, offset, bitcast[DType.uint32](safe))


def _pod_write_json(schema: PodComponentSchema, base_ptr: Int) -> String:
    if len(schema.fields) == 0:
        var raw = List[UInt8](length=max(0, schema.byte_size), fill=0)
        if schema.byte_size > 0 and base_ptr != 0:
            var src = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=base_ptr)
            for bi in range(schema.byte_size):
                raw[bi] = src[unsafe_offset=bi]
        return "{\"__raw_hex__\": " + json_quote(_bytes_to_hex(raw)) + "}"
    var out = String("{")
    for i in range(len(schema.fields)):
        if i > 0:
            out += ", "
        ref f = schema.fields[i]
        out += json_quote(f.name) + ": "
        if f.kind == POD_FIELD_BOOL:
            var b = (_pod_read_u8(base_ptr, f.offset) != 0) if f.size == 1 else (_pod_read_u32(base_ptr, f.offset) != 0)
            out += "true" if b else "false"
        elif f.kind == POD_FIELD_INT:
            var iv = Int(_pod_read_i32(base_ptr, f.offset)) if f.size == 4 else Int(_pod_read_i64(base_ptr, f.offset))
            out += String(iv)
        elif f.kind == POD_FIELD_F32:
            out += String(Float64(_pod_read_f32(base_ptr, f.offset)))
        elif f.kind == POD_FIELD_VEC3 or f.kind == POD_FIELD_COLOR_RGB:
            var x = _pod_read_f32(base_ptr, f.offset)
            var y = _pod_read_f32(base_ptr, f.offset + 4)
            var z = _pod_read_f32(base_ptr, f.offset + 8)
            out += "[" + String(Float64(x)) + ", " + String(Float64(y)) + ", " + String(Float64(z)) + "]"
        elif f.kind == POD_FIELD_QUAT:
            var x = _pod_read_f32(base_ptr, f.offset)
            var y = _pod_read_f32(base_ptr, f.offset + 4)
            var z = _pod_read_f32(base_ptr, f.offset + 8)
            var w = _pod_read_f32(base_ptr, f.offset + 12)
            out += "[" + String(Float64(x)) + ", " + String(Float64(y)) + ", " + String(Float64(z)) + ", " + String(Float64(w)) + "]"
        else:
            out += "null"
    out += "}"
    return out^


def _pod_read_json(
    schema: PodComponentSchema, doc: JsonDocument, node: Int, mut buf: List[UInt8]
) -> Bool:
    if node < 0 or node >= len(doc.nodes) or doc.nodes[node].kind != 5:
        return False
    buf = List[UInt8](length=max(1, schema.byte_size), fill=0)
    if len(schema.fields) == 0:
        var hex_n = doc.object_get(node, "__raw_hex__")
        if hex_n >= 0 and doc.nodes[hex_n].kind == 3:
            try:
                var raw = _hex_to_bytes(doc.text(hex_n, ""))
                if len(raw) != schema.byte_size:
                    return False
                if schema.byte_size > 0:
                    buf = raw^
            except:
                return False
        return True
    var base_ptr = Int(buf.unsafe_ptr())

    for i in range(len(schema.fields)):
        ref f = schema.fields[i]
        var fnode = doc.object_get(node, f.name)
        if fnode < 0:
            continue
        if f.kind == POD_FIELD_BOOL:
            var b = doc.boolean(fnode, False) if doc.nodes[fnode].kind == 1 else (doc.integer(fnode, 0) != 0)
            if f.size == 1:
                _pod_write_u8(base_ptr, f.offset, UInt8(1) if b else UInt8(0))
            else:
                _pod_write_u32(base_ptr, f.offset, UInt32(1) if b else UInt32(0))
        elif f.kind == POD_FIELD_INT:
            var iv = doc.i64(fnode, 0)
            if f.size == 4:
                _pod_write_i32(base_ptr, f.offset, Int32(iv))
            else:
                _pod_write_i64(base_ptr, f.offset, iv)
        elif f.kind == POD_FIELD_F32:
            _pod_write_f32(base_ptr, f.offset, Float32(doc.number(fnode, 0.0)))
        elif f.kind == POD_FIELD_VEC3 or f.kind == POD_FIELD_COLOR_RGB:
            if doc.nodes[fnode].kind == 4 and doc.array_size(fnode) >= 3:
                _pod_write_f32(base_ptr, f.offset, Float32(doc.number(doc.at(fnode, 0), 0.0)))
                _pod_write_f32(base_ptr, f.offset + 4, Float32(doc.number(doc.at(fnode, 1), 0.0)))
                _pod_write_f32(base_ptr, f.offset + 8, Float32(doc.number(doc.at(fnode, 2), 0.0)))
        elif f.kind == POD_FIELD_QUAT:
            if doc.nodes[fnode].kind == 4 and doc.array_size(fnode) >= 4:
                _pod_write_f32(base_ptr, f.offset, Float32(doc.number(doc.at(fnode, 0), 0.0)))
                _pod_write_f32(base_ptr, f.offset + 4, Float32(doc.number(doc.at(fnode, 1), 0.0)))
                _pod_write_f32(base_ptr, f.offset + 8, Float32(doc.number(doc.at(fnode, 2), 0.0)))
                _pod_write_f32(base_ptr, f.offset + 12, Float32(doc.number(doc.at(fnode, 3), 1.0)))
    return True


def _serializable_bin_write[C: SerializableComponent](comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[C, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.begin(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
    comp.write_bin(out)
    out.end()


def _serializable_bin_read[C: SerializableComponent](comp_addr: Int, reader_addr: Int) -> Bool:
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    try:
        var ver = reader.enter(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
        var val = C.read_bin(reader, ver)
        reader.leave()
        var dst = Pointer[C, MutAnyOrigin](unsafe_from_address=comp_addr)
        __get_address_as_uninit_lvalue(dst._mlir_value) = val^
        return True
    except:
        return False


def _codec_register_world[C: SerializableComponent](world_addr: Int) -> Int:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    var tid = world.register_component[C]()
    comptime is_triv = C.__copy_ctor_is_trivial and C.__move_ctor_is_trivial and C.__del__is_trivial
    if not is_triv and tid >= 0 and not world.components[tid].has_custom_bin:
        world.set_component_binary_hooks(
            tid, _serializable_bin_write[C], _serializable_bin_read[C]
        )
    return tid


def _codec_has[C: SerializableComponent](world_addr: Int, entity: Entity) -> Bool:
    ref world = Pointer[World, ImmutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return world.has[C](entity)


def _codec_write_bin[C: SerializableComponent](world_addr: Int, entity: Entity, writer_addr: Int) -> Bool:
    ref world = Pointer[World, ImmutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    var ptr_opt = world.get_ptr[C](entity)
    if not ptr_opt:
        return False
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.begin(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
    ptr_opt.value()[unsafe_offset=0].write_bin(out)
    out.end()
    return True


def _codec_read_bin[C: SerializableComponent](
    world_addr: Int, entity: Entity, reader_addr: Int, newest_ver: Int
) -> Bool:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    try:
        var ver = reader.enter(C.SCHEMA_FOURCC, newest_ver)
        var val = C.read_bin(reader, ver)
        reader.leave()
        return world.insert[C](entity, val^)
    except:
        return False


def _codec_write_json[C: SerializableComponent](world_addr: Int, entity: Entity, str_out_addr: Int) -> Bool:
    ref world = Pointer[World, ImmutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    var ptr_opt = world.get_ptr[C](entity)
    if not ptr_opt:
        return False
    ref s_out = Pointer[String, MutAnyOrigin](unsafe_from_address=str_out_addr)[unsafe_offset=0]
    s_out = ptr_opt.value()[unsafe_offset=0].write_json()
    return True


def _codec_read_json[C: SerializableComponent](
    world_addr: Int, entity: Entity, doc_addr: Int, node: Int
) -> Bool:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    ref doc = Pointer[JsonDocument, ImmutAnyOrigin](unsafe_from_address=doc_addr)[unsafe_offset=0]
    try:
        var val = C.read_json(doc, node)
        return world.insert[C](entity, val^)
    except:
        return False


def _codec_insert_default[C: SerializableComponent](world_addr: Int, entity: Entity) -> Bool:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    if not world.is_alive(entity) or world.has[C](entity):
        return False
    try:
        var bytes = List[UInt8](capacity=2)
        bytes.append(UInt8(123))  # '{'
        bytes.append(UInt8(125))  # '}'
        var empty_doc = parse_json(bytes)
        var val = C.read_json(empty_doc, empty_doc.root)
        return world.insert[C](entity, val^)
    except:
        return False


def _codec_remove[C: SerializableComponent](world_addr: Int, entity: Entity) -> Bool:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    if not world.is_alive(entity):
        return False
    return world.remove[C](entity)


@fieldwise_init
struct PrefabOverride(Copyable, Movable):
    """Per-instance component override targeting a prefab node by slot index or node name."""
    var target_slot: Int
    var target_name: String
    var schema_name: String
    var json_payload: String

    @staticmethod
    def for_slot[C: SerializableComponent](slot: Int, comp: C) -> Self:
        return Self(slot, "", String(C.SCHEMA_NAME), comp.write_json())

    @staticmethod
    def for_name[C: SerializableComponent](node_name: String, comp: C) -> Self:
        return Self(-1, node_name, String(C.SCHEMA_NAME), comp.write_json())


struct PrefabInstanceComponent(SerializableComponent):
    """Tracks an entity's link to a `.flprefab` asset and sparse per-instance overrides."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.scene.PrefabInstanceComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "PrefabInstanceComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("PRFI")
    comptime SCHEMA_VERSION: Int = 1

    var prefab_name: String
    var prefab_ref: AssetRef
    var node_slot: Int
    var overrides: List[PrefabOverride]

    def __init__(
        out self,
        prefab_name: String = "",
        var prefab_ref: AssetRef = AssetRef(),
        node_slot: Int = 0,
    ):
        self.prefab_name = prefab_name
        self.prefab_ref = prefab_ref^
        self.node_slot = node_slot
        self.overrides = List[PrefabOverride]()

    def __init__(out self, *, copy: Self):
        self.prefab_name = copy.prefab_name
        self.prefab_ref = copy.prefab_ref.copy()
        self.node_slot = copy.node_slot
        self.overrides = copy.overrides.copy()

    def write_bin(self, mut out: BlobWriter):
        out.text(self.prefab_name)
        self.prefab_ref.write_bin(out)
        out.integer(self.node_slot)
        out.u32(UInt32(len(self.overrides)))
        for i in range(len(self.overrides)):
            ref ov = self.overrides[i]
            out.integer(ov.target_slot)
            out.text(ov.target_name)
            out.text(ov.schema_name)
            out.text(ov.json_payload)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var pn = reader.text()
        var pref = AssetRef.read_bin(reader, 1)
        var ns = reader.integer()
        var n_ov = Int(reader.u32())
        if reader.failed or n_ov < 0 or n_ov > 4096 or n_ov > max(0, reader.limit() - reader.offset) // 12:
            raise Error("PrefabInstanceComponent: invalid override count")
        var out = Self(pn, pref^, ns)
        for _ in range(n_ov):
            var ts = reader.integer()
            var tn = reader.text()
            var sn = reader.text()
            var jp = reader.text()
            if reader.failed:
                raise Error("PrefabInstanceComponent: truncated override entry")
            out.overrides.append(PrefabOverride(ts, tn, sn, jp))
        if reader.failed:
            raise Error("PrefabInstanceComponent: truncated blob")
        return out^

    def write_json(self) -> String:
        var out = (
            "{\"prefab_name\": " + json_quote(self.prefab_name)
            + ", \"prefab_ref\": " + self.prefab_ref.write_json()
            + ", \"node_slot\": " + String(self.node_slot)
            + ", \"overrides\": ["
        )
        for i in range(len(self.overrides)):
            if i > 0:
                out += ", "
            ref ov = self.overrides[i]
            out += (
                "{\"target_slot\": " + String(ov.target_slot)
                + ", \"target_name\": " + json_quote(ov.target_name)
                + ", \"schema_name\": " + json_quote(ov.schema_name)
                + ", \"json_payload\": " + json_quote(ov.json_payload)
                + "}"
            )
        out += "]}"
        return out^

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var pn = doc.text(doc.object_get(node, "prefab_name"), "")
        var pref_node = doc.object_get(node, "prefab_ref")
        var pref = AssetRef.read_json(doc, pref_node) if pref_node >= 0 else AssetRef()
        var ns = doc.integer(doc.object_get(node, "node_slot"), 0)
        var out = Self(pn, pref^, ns)
        var ov_arr = doc.object_get(node, "overrides")
        if ov_arr >= 0 and doc.nodes[ov_arr].kind == 4:
            for i in range(doc.array_size(ov_arr)):
                var ovn = doc.at(ov_arr, i)
                var ts = doc.integer(doc.object_get(ovn, "target_slot"), -1)
                var tn = doc.text(doc.object_get(ovn, "target_name"), "")
                var sn = doc.text(doc.object_get(ovn, "schema_name"), "")
                var jp = doc.text(doc.object_get(ovn, "json_payload"), "{}")
                out.overrides.append(PrefabOverride(ts, tn, sn, jp))
        return out^


@fieldwise_init
struct ComponentCodecEntry(Copyable, Movable):
    var component_hash: UInt64
    var schema_name: String
    var schema_fourcc: UInt32
    var schema_version: Int
    var reg_world_fn: _CodecRegisterWorldFn
    var has_fn: _CodecHasFn
    var write_bin_fn: _CodecWriteBinFn
    var read_bin_fn: _CodecReadBinFn
    var write_json_fn: _CodecWriteJsonFn
    var read_json_fn: _CodecReadJsonFn
    var has_slot_hooks: Bool
    var write_bin_slots_fn: _CodecWriteBinSlotsFn
    var write_json_slots_fn: _CodecWriteJsonSlotsFn
    var resolve_slots_fn: _CodecResolveSlotsFn
    var default_insert_fn: _CodecDefaultInsertFn
    var remove_fn: _CodecRemoveFn
    var is_pod: Bool
    var pod_schema_idx: Int


struct ComponentRegistry(Movable):
    """Type-erased registry of `SerializableComponent` and runtime POD binary & JSON codecs."""
    var entries: List[ComponentCodecEntry]
    var pod_schemas: List[PodComponentSchema]

    def __init__(out self):
        self.entries = List[ComponentCodecEntry]()
        self.pod_schemas = List[PodComponentSchema]()

    @staticmethod
    def with_builtins() -> Self:
        var reg = Self()
        reg.register[LocalTransform]()
        reg.register[GlobalTransform]()
        reg.register[PrevGlobalTransform]()
        reg.register[PrefabInstanceComponent]()
        return reg^

    def find_by_hash(self, comp_hash: UInt64) -> Int:
        for i in range(len(self.entries)):
            if self.entries[i].component_hash == comp_hash:
                return i
        return -1

    def find_by_name(self, name: String) -> Int:
        for i in range(len(self.entries)):
            if self.entries[i].schema_name == name:
                return i
        return -1

    def find_by_fourcc(self, tag: UInt32) -> Int:
        for i in range(len(self.entries)):
            if self.entries[i].schema_fourcc == tag:
                return i
        return -1

    def register[C: SerializableComponent](mut self):
        if self.find_by_hash(C.COMPONENT_ID) >= 0:
            return
        self.entries.append(
            ComponentCodecEntry(
                C.COMPONENT_ID,
                String(C.SCHEMA_NAME),
                C.SCHEMA_FOURCC,
                C.SCHEMA_VERSION,
                _codec_register_world[C],
                _codec_has[C],
                _codec_write_bin[C],
                _codec_read_bin[C],
                _codec_write_json[C],
                _codec_read_json[C],
                False,
                _noop_write_slots,
                _noop_write_slots,
                _noop_resolve_slots,
                _codec_insert_default[C],
                _codec_remove[C],
                False,
                -1,
            )
        )

    def register_pod_schema(mut self, schema: PodComponentSchema) -> Bool:
        if not schema.validate():
            return False
        var existing_name = self.find_by_name(schema.name)
        if existing_name >= 0:
            if not self.entries[existing_name].is_pod:
                return False
            var pidx = self.entries[existing_name].pod_schema_idx
            if pidx < 0 or pidx >= len(self.pod_schemas):
                return False
            return self.pod_schemas[pidx].matches_layout(schema)
        if self.find_by_hash(schema.component_hash) >= 0 or self.find_by_fourcc(schema.schema_fourcc) >= 0:
            return False
        var pidx = len(self.pod_schemas)
        self.pod_schemas.append(schema.copy())
        self.entries.append(
            ComponentCodecEntry(
                schema.component_hash,
                schema.name,
                schema.schema_fourcc,
                1,
                _noop_reg_world,
                _noop_has,
                _noop_write_bin,
                _noop_read_bin,
                _noop_write_json,
                _noop_read_json,
                False,
                _noop_write_slots,
                _noop_write_slots,
                _noop_resolve_slots,
                _noop_default_insert,
                _noop_remove,
                True,
                pidx,
            )
        )
        return True

    def insert_default_on_entity(self, mut world: World, entity: Entity, comp_name: String) -> Bool:
        if not world.is_alive(entity):
            return False
        var idx = self.find_by_name(comp_name)
        if idx < 0:
            return False
        ref entry = self.entries[idx]
        if entry.is_pod:
            if entry.pod_schema_idx < 0 or entry.pod_schema_idx >= len(self.pod_schemas):
                return False
            ref schema = self.pod_schemas[entry.pod_schema_idx]
            var tid = world.register_pod_component(
                schema.component_hash, schema.byte_size, schema.alignment, schema.storage_mode
            )
            if tid < 0 or world.has_by_type_id(entity, tid):
                return False
            return world.insert_raw_pod(entity, tid, 0)
        var w_addr = Int(Pointer(to=world))
        var reg_fn = entry.reg_world_fn
        _ = reg_fn(w_addr)
        var ins_fn = entry.default_insert_fn
        return ins_fn(w_addr, entity)

    def remove_from_entity(self, mut world: World, entity: Entity, comp_name: String) -> Bool:
        if not world.is_alive(entity):
            return False
        var idx = self.find_by_name(comp_name)
        if idx < 0:
            return False
        ref entry = self.entries[idx]
        if entry.is_pod:
            var tid = world.find_component_by_hash(entry.component_hash)
            if tid < 0:
                return False
            return world.remove_by_type_id(entity, tid)
        var rem_fn = entry.remove_fn
        return rem_fn(Int(Pointer(to=world)), entity)

    def set_slot_hooks[C: SerializableComponent](
        mut self,
        write_bin_slots_fn: _CodecWriteBinSlotsFn,
        write_json_slots_fn: _CodecWriteJsonSlotsFn,
        resolve_slots_fn: _CodecResolveSlotsFn,
    ):
        var idx = self.find_by_hash(C.COMPONENT_ID)
        if idx < 0:
            self.register[C]()
            idx = self.find_by_hash(C.COMPONENT_ID)
        if idx >= 0:
            self.entries[idx].has_slot_hooks = True
            self.entries[idx].write_bin_slots_fn = write_bin_slots_fn
            self.entries[idx].write_json_slots_fn = write_json_slots_fn
            self.entries[idx].resolve_slots_fn = resolve_slots_fn

    def ensure_registered_in_world(self, mut world: World):
        register_transform_components(world)
        register_unknown_components(world)
        var w_addr = Int(Pointer(to=world))
        for i in range(len(self.entries)):
            if self.entries[i].is_pod:
                var pidx = self.entries[i].pod_schema_idx
                if pidx >= 0 and pidx < len(self.pod_schemas):
                    ref schema = self.pod_schemas[pidx]
                    _ = world.register_pod_component(
                        schema.component_hash, schema.byte_size, schema.alignment, schema.storage_mode
                    )
            else:
                var fn_ptr = self.entries[i].reg_world_fn
                _ = fn_ptr(w_addr)

    def sorted_entry_indices(self) -> List[Int]:
        """Returns deterministic schema-hash-sorted indices across `self.entries`."""
        var keys = List[Int](capacity=len(self.entries))
        for i in range(len(self.entries)):
            keys.append(Int(self.entries[i].schema_fourcc))
        return stable_order(keys)

    def has_on_entity(self, idx: Int, world: World, entity: Entity) -> Bool:
        if idx < 0 or idx >= len(self.entries):
            return False
        if self.entries[idx].is_pod:
            var tid = world.find_component_by_hash(self.entries[idx].component_hash)
            return tid >= 0 and world.has_by_type_id(entity, tid)
        var fn_ptr = self.entries[idx].has_fn
        return fn_ptr(Int(Pointer(to=world)), entity)

    def write_entity_bin(self, idx: Int, world: World, entity: Entity, mut out: BlobWriter) -> Bool:
        if idx < 0 or idx >= len(self.entries):
            return False
        if self.entries[idx].is_pod:
            var pidx = self.entries[idx].pod_schema_idx
            if pidx < 0 or pidx >= len(self.pod_schemas):
                return False
            ref schema = self.pod_schemas[pidx]
            var tid = world.find_component_by_hash(schema.component_hash)
            if tid < 0 or not world.has_by_type_id(entity, tid):
                return False
            var ptr = world.get_raw_ptr(entity, tid)
            if schema.byte_size > 0 and ptr == 0:
                return False
            out.begin(schema.schema_fourcc, 1)
            out.integer(schema.byte_size)
            if schema.byte_size > 0:
                out.write_bytes_raw(ptr, schema.byte_size)
            out.end()
            return True
        var fn_ptr = self.entries[idx].write_bin_fn
        return fn_ptr(Int(Pointer(to=world)), entity, Int(Pointer(to=out)))

    def write_entity_bin_slots(
        self, idx: Int, world: World, entity: Entity, ent_to_slot: List[Int], mut out: BlobWriter
    ) -> Bool:
        if self.entries[idx].has_slot_hooks:
            var fn_ptr = self.entries[idx].write_bin_slots_fn
            return fn_ptr(Int(Pointer(to=world)), entity, Int(Pointer(to=ent_to_slot)), Int(Pointer(to=out)))
        return self.write_entity_bin(idx, world, entity, out)

    def read_entity_bin(self, idx: Int, mut world: World, entity: Entity, mut reader: BlobReader) -> Bool:
        if idx < 0 or idx >= len(self.entries):
            return False
        if self.entries[idx].is_pod:
            var pidx = self.entries[idx].pod_schema_idx
            if pidx < 0 or pidx >= len(self.pod_schemas):
                return False
            ref schema = self.pod_schemas[pidx]
            try:
                _ = reader.enter(schema.schema_fourcc, 1)
                var bsz = reader.integer()
                if reader.failed or reader.offset < 0 or bsz != schema.byte_size or bsz < 0 or bsz > reader.limit() - reader.offset:
                    return False
                var tid = world.register_pod_component(
                    schema.component_hash, schema.byte_size, schema.alignment, schema.storage_mode
                )
                if tid < 0:
                    return False
                var ok: Bool
                if bsz > 0:
                    ok = world.insert_raw_pod_bytes(
                        entity, tid, Span(reader.data)[reader.offset : reader.offset + bsz]
                    )
                    if ok:
                        reader.offset += bsz
                else:
                    ok = world.insert_raw_pod(entity, tid, 0)
                reader.leave()
                return ok
            except:
                return False
        var fn_ptr = self.entries[idx].read_bin_fn
        return fn_ptr(
            Int(Pointer(to=world)),
            entity,
            Int(Pointer(to=reader)),
            self.entries[idx].schema_version,
        )

    def write_entity_json(self, idx: Int, world: World, entity: Entity, mut out_json: String) -> Bool:
        if idx < 0 or idx >= len(self.entries):
            return False
        if self.entries[idx].is_pod:
            var pidx = self.entries[idx].pod_schema_idx
            if pidx < 0 or pidx >= len(self.pod_schemas):
                return False
            ref schema = self.pod_schemas[pidx]
            var tid = world.find_component_by_hash(schema.component_hash)
            if tid < 0 or not world.has_by_type_id(entity, tid):
                return False
            var ptr = world.get_raw_ptr(entity, tid)
            if schema.byte_size > 0 and ptr == 0:
                return False
            out_json = _pod_write_json(schema, ptr)
            return True
        var fn_ptr = self.entries[idx].write_json_fn
        return fn_ptr(Int(Pointer(to=world)), entity, Int(Pointer(to=out_json)))

    def write_entity_json_slots(
        self, idx: Int, world: World, entity: Entity, ent_to_slot: List[Int], mut out_json: String
    ) -> Bool:
        if self.entries[idx].has_slot_hooks:
            var fn_ptr = self.entries[idx].write_json_slots_fn
            return fn_ptr(Int(Pointer(to=world)), entity, Int(Pointer(to=ent_to_slot)), Int(Pointer(to=out_json)))
        return self.write_entity_json(idx, world, entity, out_json)

    def read_entity_json(
        self, idx: Int, mut world: World, entity: Entity, doc: JsonDocument, node: Int
    ) -> Bool:
        if idx < 0 or idx >= len(self.entries):
            return False
        if self.entries[idx].is_pod:
            var pidx = self.entries[idx].pod_schema_idx
            if pidx < 0 or pidx >= len(self.pod_schemas):
                return False
            ref schema = self.pod_schemas[pidx]
            var tid = world.register_pod_component(
                schema.component_hash, schema.byte_size, schema.alignment, schema.storage_mode
            )
            if tid < 0:
                return False
            var buf = List[UInt8]()
            if not _pod_read_json(schema, doc, node, buf):
                return False
            if schema.byte_size > 0:
                return world.insert_raw_pod_bytes(entity, tid, Span(buf))
            return world.insert_raw_pod(entity, tid, 0)
        var fn_ptr = self.entries[idx].read_json_fn
        return fn_ptr(Int(Pointer(to=world)), entity, Int(Pointer(to=doc)), node)

    def resolve_entity_slots(self, idx: Int, mut world: World, entity: Entity, spawned: List[Entity]):
        if self.entries[idx].has_slot_hooks:
            var fn_ptr = self.entries[idx].resolve_slots_fn
            fn_ptr(Int(Pointer(to=world)), entity, Int(Pointer(to=spawned)))


# ---------------------------------------------------------------------------
# Prefab, NestedPrefabRef & .flprefab Codecs
# ---------------------------------------------------------------------------


@fieldwise_init
struct NestedPrefabRef(Copyable, Movable):
    """Embeds a child `Prefab` under `attach_parent_slot` with local root offset and overrides."""
    var prefab_name: String
    var attach_parent_slot: Int
    var local_offset: LocalTransform
    var overrides: List[PrefabOverride]

    def __init__(
        out self,
        prefab_name: String,
        attach_parent_slot: Int = -1,
        local_offset: LocalTransform = LocalTransform.identity(),
    ):
        self.prefab_name = prefab_name
        self.attach_parent_slot = attach_parent_slot
        self.local_offset = local_offset
        self.overrides = List[PrefabOverride]()

    def add_override(mut self, var ov: PrefabOverride):
        self.overrides.append(ov^)


struct PrefabNode(Copyable, Movable):
    var name: String
    var parent_slot: Int
    var schema_names: List[String]
    var json_payloads: List[String]

    def __init__(out self, name: String, parent_slot: Int = -1):
        self.name = name
        self.parent_slot = parent_slot
        self.schema_names = List[String]()
        self.json_payloads = List[String]()

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.parent_slot = copy.parent_slot
        self.schema_names = copy.schema_names.copy()
        self.json_payloads = copy.json_payloads.copy()

    def set_raw_component(mut self, schema_name: String, payload: String):
        for i in range(len(self.schema_names)):
            if self.schema_names[i] == schema_name:
                self.json_payloads[i] = payload
                return
        self.schema_names.append(schema_name)
        self.json_payloads.append(payload)


def _apply_json_component_str(
    mut world: World,
    registry: ComponentRegistry,
    entity: Entity,
    schema_name: String,
    json_str: String,
) raises:
    var cidx = registry.find_by_name(schema_name)
    if cidx < 0:
        raise Error("unregistered component in prefab/scene: " + schema_name)
    var bytes = List[UInt8]()
    for b in json_str.as_bytes():
        bytes.append(b)
    var doc = parse_json(bytes)
    if not doc.valid:
        raise Error("invalid component JSON payload for " + schema_name)
    if not registry.read_entity_json(cidx, world, entity, doc, doc.root):
        raise Error("failed to decode component JSON for " + schema_name)


struct Prefab(Copyable, Movable):
    """Reusable hierarchical entity template supporting nested prefabs and overrides."""
    var name: String
    var nodes: List[PrefabNode]
    var nested_refs: List[NestedPrefabRef]

    def __init__(out self, name: String):
        self.name = name
        self.nodes = List[PrefabNode]()
        self.nested_refs = List[NestedPrefabRef]()

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.nodes = copy.nodes.copy()
        self.nested_refs = copy.nested_refs.copy()

    def add_node(mut self, name: String, parent_slot: Int = -1) -> Int:
        var idx = len(self.nodes)
        self.nodes.append(PrefabNode(name, parent_slot))
        return idx

    def set_component[C: SerializableComponent](mut self, node_slot: Int, comp: C):
        if node_slot >= 0 and node_slot < len(self.nodes):
            self.nodes[node_slot].set_raw_component(String(C.SCHEMA_NAME), comp.write_json())

    def add_nested(mut self, var nested_ref: NestedPrefabRef):
        self.nested_refs.append(nested_ref^)

    def _validate_no_cycles(
        self,
        library: List[Prefab],
        mut active_stack: List[String],
        depth: Int,
    ) raises:
        if depth > MAX_PREFAB_NEST_DEPTH:
            raise Error("nested prefab recursion exceeded MAX_PREFAB_NEST_DEPTH")
        for i in range(len(active_stack)):
            if active_stack[i] == self.name:
                raise Error("cyclic nested prefab detected: " + self.name)
        active_stack.append(self.name)
        for n_idx in range(len(self.nested_refs)):
            ref nref = self.nested_refs[n_idx]
            var found_idx = -1
            for k in range(len(library)):
                if library[k].name == nref.prefab_name:
                    found_idx = k
                    break
            if found_idx < 0:
                _ = active_stack.pop()
                raise Error("nested prefab not found in library: " + nref.prefab_name)
            library[found_idx]._validate_no_cycles(library, active_stack, depth + 1)
        _ = active_stack.pop()

    def instantiate(
        self,
        mut world: World,
        registry: ComponentRegistry,
        library: List[Prefab] = List[Prefab](),
        overrides: List[PrefabOverride] = List[PrefabOverride](),
        root_parent: Entity = NULL_ENTITY,
    ) raises -> List[Entity]:
        var active_stack = List[String]()
        self._validate_no_cycles(library, active_stack, 0)
        registry.ensure_registered_in_world(world)
        var rollback_list = List[Entity]()
        try:
            return self._instantiate_impl(
                world, registry, library, overrides, root_parent, 0, rollback_list
            )
        except err:
            for i in range(len(rollback_list) - 1, -1, -1):
                if world.is_alive(rollback_list[i]):
                    _ = world.despawn(rollback_list[i])
            raise err^

    def _instantiate_impl(
        self,
        mut world: World,
        registry: ComponentRegistry,
        library: List[Prefab],
        overrides: List[PrefabOverride],
        root_parent: Entity,
        depth: Int,
        mut rollback_list: List[Entity],
    ) raises -> List[Entity]:
        if depth > MAX_PREFAB_NEST_DEPTH:
            raise Error("nested prefab recursion exceeded MAX_PREFAB_NEST_DEPTH")

        var spawned = List[Entity](capacity=len(self.nodes))
        var spawned_names = List[String](capacity=len(self.nodes))
        for i in range(len(self.nodes)):
            var e = world.spawn_empty()
            rollback_list.append(e)
            spawned.append(e)
            spawned_names.append(self.nodes[i].name)
            for c in range(len(self.nodes[i].schema_names)):
                _apply_json_component_str(
                    world,
                    registry,
                    e,
                    self.nodes[i].schema_names[c],
                    self.nodes[i].json_payloads[c],
                )

        # Wire internal hierarchy + optional attachment to root_parent
        for i in range(len(self.nodes)):
            var p_slot = self.nodes[i].parent_slot
            if p_slot >= 0 and p_slot < len(spawned):
                _ = set_parent(world, spawned[i], spawned[p_slot])
            elif root_parent.valid() and world.is_alive(root_parent):
                _ = set_parent(world, spawned[i], root_parent)

        # Instantiate nested prefabs
        for n_idx in range(len(self.nested_refs)):
            ref nref = self.nested_refs[n_idx]
            var found_idx = -1
            for k in range(len(library)):
                if library[k].name == nref.prefab_name:
                    found_idx = k
                    break
            if found_idx < 0:
                raise Error("nested prefab not found in library: " + nref.prefab_name)
            var attach_ent = root_parent
            if nref.attach_parent_slot >= 0 and nref.attach_parent_slot < len(spawned):
                attach_ent = spawned[nref.attach_parent_slot]

            var child_base = len(spawned)
            var child_ents = library[found_idx]._instantiate_impl(
                world, registry, library, nref.overrides, attach_ent, depth + 1, rollback_list
            )
            # Apply local_offset to nested prefab root nodes (nodes whose parent_slot < 0)
            for ci in range(len(library[found_idx].nodes)):
                if library[found_idx].nodes[ci].parent_slot < 0 and ci < len(child_ents):
                    var re = child_ents[ci]
                    var lt_opt = world.get_ptr_mut[LocalTransform](re)
                    if lt_opt:
                        var lt_ptr = lt_opt.value()
                        var cur = lt_ptr[unsafe_offset=0]
                        lt_ptr[unsafe_offset=0] = LocalTransform(
                            nref.local_offset.translation + nref.local_offset.rotation.rotate(cur.translation),
                            (nref.local_offset.rotation * cur.rotation).normalized(),
                            cur.scale,
                        )
                    else:
                        _ = world.insert[LocalTransform](re, nref.local_offset)
                if ci < len(child_ents):
                    spawned.append(child_ents[ci])
                    spawned_names.append(library[found_idx].nodes[ci].name)
            for extra in range(len(library[found_idx].nodes), len(child_ents)):
                spawned.append(child_ents[extra])
                spawned_names.append("")
            _ = child_base

        # Apply caller overrides (overrides win over base and nested defaults)
        for ov_i in range(len(overrides)):
            ref ov = overrides[ov_i]
            var target_e = NULL_ENTITY
            if ov.target_slot >= 0 and ov.target_slot < len(spawned):
                target_e = spawned[ov.target_slot]
            elif ov.target_name.byte_length() > 0:
                for si in range(len(spawned_names)):
                    if spawned_names[si] == ov.target_name:
                        target_e = spawned[si]
                        break
            if target_e.valid():
                _apply_json_component_str(world, registry, target_e, ov.schema_name, ov.json_payload)

        return spawned^


def prefab_to_json(prefab: Prefab) -> String:
    var out = String("{\n")
    out += '  "format": "flprefab",\n'
    out += '  "version": 1,\n'
    out += '  "name": ' + json_quote(prefab.name) + ",\n"
    out += '  "nodes": ['
    for i in range(len(prefab.nodes)):
        if i > 0:
            out += ", "
        ref nd = prefab.nodes[i]
        out += '{"name": ' + json_quote(nd.name) + ', "parent_slot": ' + String(nd.parent_slot) + ', "components": {'
        for c in range(len(nd.schema_names)):
            if c > 0:
                out += ", "
            out += json_quote(nd.schema_names[c]) + ": " + nd.json_payloads[c]
        out += "}}"
    out += "],\n"
    out += '  "nested_refs": ['
    for i in range(len(prefab.nested_refs)):
        if i > 0:
            out += ", "
        ref nr = prefab.nested_refs[i]
        out += '{"prefab_name": ' + json_quote(nr.prefab_name)
        out += ', "attach_parent_slot": ' + String(nr.attach_parent_slot)
        out += ', "local_offset": ' + nr.local_offset.write_json()
        out += ', "overrides": ['
        for oi in range(len(nr.overrides)):
            if oi > 0:
                out += ", "
            ref ov = nr.overrides[oi]
            out += '{"target_slot": ' + String(ov.target_slot)
            out += ', "target_name": ' + json_quote(ov.target_name)
            out += ', "schema_name": ' + json_quote(ov.schema_name)
            out += ', "json_payload": ' + json_quote(ov.json_payload)
            out += "}"
        out += "]}"
    out += "]\n"
    out += "}\n"
    return out^


def save_flprefab_file(path: String, prefab: Prefab) -> Bool:
    var txt = prefab_to_json(prefab)
    var raw = _string_to_bytes(txt)
    return write_bytes(path, raw)


def load_flprefab_bytes(data: List[UInt8]) raises -> Prefab:
    if len(data) == 0:
        raise Error("empty .flprefab payload")
    var doc = parse_json(data)
    if not doc.valid or doc.root < 0 or doc.nodes[doc.root].kind != 5:
        raise Error("invalid .flprefab JSON document")

    var fmt_node = doc.object_get(doc.root, "format")
    if fmt_node >= 0:
        var fmt = doc.text(fmt_node, "")
        if fmt.byte_length() > 0 and fmt != "flprefab":
            raise Error("unexpected .flprefab format: " + fmt)

    var name = doc.text(doc.object_get(doc.root, "name"), "Prefab")
    var pf = Prefab(name)

    var nodes_node = doc.object_get(doc.root, "nodes")
    if nodes_node >= 0:
        if doc.nodes[nodes_node].kind != 4:
            raise Error("invalid nodes array in .flprefab")
        for i in range(doc.array_size(nodes_node)):
            var n_item = doc.at(nodes_node, i)
            if doc.nodes[n_item].kind != 5:
                raise Error("invalid node entry in .flprefab")
            var n_name = doc.text(doc.object_get(n_item, "name"), "Node" + String(i))
            var p_slot = doc.integer(doc.object_get(n_item, "parent_slot"), -1)
            var slot = pf.add_node(n_name, p_slot)
            var comps_node = doc.object_get(n_item, "components")
            if comps_node >= 0:
                if doc.nodes[comps_node].kind != 5:
                    raise Error("invalid components object in .flprefab node")
                for k in range(len(doc.nodes[comps_node].keys)):
                    var cname = doc.nodes[comps_node].keys[k]
                    var cval = doc.nodes[comps_node].children[k]
                    pf.nodes[slot].set_raw_component(cname, json_text(doc, cval))

    var entities_node = doc.object_get(doc.root, "entities")
    if entities_node >= 0:
        if doc.nodes[entities_node].kind != 4:
            raise Error("invalid entities array in .flprefab")
        for i in range(doc.array_size(entities_node)):
            var e_item = doc.at(entities_node, i)
            if doc.nodes[e_item].kind != 5:
                raise Error("invalid entity entry in .flprefab")
            var e_name = doc.text(doc.object_get(e_item, "name"), "Entity" + String(i))
            var p_slot = doc.integer(doc.object_get(e_item, "parent"), doc.integer(doc.object_get(e_item, "parent_slot"), -1))
            var slot = pf.add_node(e_name, p_slot)

            var comps_node = doc.object_get(e_item, "components")
            if comps_node >= 0 and doc.nodes[comps_node].kind == 5:
                for k in range(len(doc.nodes[comps_node].keys)):
                    var cname = doc.nodes[comps_node].keys[k]
                    var cval = doc.nodes[comps_node].children[k]
                    pf.nodes[slot].set_raw_component(cname, json_text(doc, cval))

            # Shorthand simplified entity fields (`mesh`, `material`, `rig`, `clip`, `position`, `audio`)
            var pos_node = doc.object_get(e_item, "position")
            var tr_node = doc.object_get(e_item, "translation")
            var rot_node = doc.object_get(e_item, "rotation")
            var scl_node = doc.object_get(e_item, "scale")
            if pos_node >= 0 or tr_node >= 0 or rot_node >= 0 or scl_node >= 0:
                var lt_str = String("{")
                var need_comma = False
                if tr_node >= 0:
                    lt_str += '"translation": ' + json_text(doc, tr_node)
                    need_comma = True
                elif pos_node >= 0:
                    lt_str += '"translation": ' + json_text(doc, pos_node)
                    need_comma = True
                if rot_node >= 0:
                    if need_comma:
                        lt_str += ", "
                    lt_str += '"rotation": ' + json_text(doc, rot_node)
                    need_comma = True
                if scl_node >= 0:
                    if need_comma:
                        lt_str += ", "
                    lt_str += '"scale": ' + json_text(doc, scl_node)
                lt_str += "}"
                pf.nodes[slot].set_raw_component("LocalTransform", lt_str)

            var mesh_node = doc.object_get(e_item, "mesh_ref")
            if mesh_node < 0:
                mesh_node = doc.object_get(e_item, "mesh")
            var mref = AssetRef()
            if mesh_node >= 0:
                if doc.nodes[mesh_node].kind == 3:
                    var mp = doc.text(mesh_node, "")
                    if mp.byte_length() > 0:
                        mref = AssetRef.from_path(mp, ASSET_KIND_MESH)
                elif doc.nodes[mesh_node].kind == 5:
                    mref = AssetRef.read_json(doc, mesh_node)
                    if mref.kind == ASSET_KIND_RAW:
                        mref.kind = ASSET_KIND_MESH

            var mat_node = doc.object_get(e_item, "material_ref")
            if mat_node < 0:
                mat_node = doc.object_get(e_item, "material")
            var matref = AssetRef()
            if mat_node >= 0:
                if doc.nodes[mat_node].kind == 3:
                    var mp = doc.text(mat_node, "")
                    if mp.byte_length() > 0:
                        matref = AssetRef.from_path(mp, ASSET_KIND_RAW)
                elif doc.nodes[mat_node].kind == 5:
                    matref = AssetRef.read_json(doc, mat_node)

            if not mref.is_empty():
                var mr_json = (
                    '{"mesh_id": 0, "material_id": -1, "cast_shadows": true, "visible": true'
                    + ', "mesh_ref": ' + mref.write_json()
                    + ', "material_ref": ' + matref.write_json()
                    + "}"
                )
                pf.nodes[slot].set_raw_component("MeshRendererComponent", mr_json)

            if not matref.is_empty():
                var mo_json = '{"material_id": -1, "enabled": true, "material_ref": ' + matref.write_json() + "}"
                pf.nodes[slot].set_raw_component("MaterialOverrideComponent", mo_json)

            var rig_node = doc.object_get(e_item, "rig_ref")
            if rig_node < 0:
                rig_node = doc.object_get(e_item, "rig")
            var rref = AssetRef()
            if rig_node >= 0:
                if doc.nodes[rig_node].kind == 3:
                    var rp = doc.text(rig_node, "")
                    if rp.byte_length() > 0:
                        rref = AssetRef.from_path(rp, ASSET_KIND_RAW)
                elif doc.nodes[rig_node].kind == 5:
                    rref = AssetRef.read_json(doc, rig_node)
            if not rref.is_empty():
                var rc_json = '{"rig_ref": ' + rref.write_json() + ', "bound": false, "joint_count": 0}'
                pf.nodes[slot].set_raw_component("RigComponent", rc_json)

            var graph_node = doc.object_get(e_item, "graph_ref")
            if graph_node < 0:
                graph_node = doc.object_get(e_item, "graph")
            var gref = AssetRef()
            if graph_node >= 0:
                if doc.nodes[graph_node].kind == 3:
                    var gp = doc.text(graph_node, "")
                    if gp.byte_length() > 0:
                        gref = AssetRef.from_path(gp, ASSET_KIND_RAW)
                elif doc.nodes[graph_node].kind == 5:
                    gref = AssetRef.read_json(doc, graph_node)
            var clip_name = doc.text(doc.object_get(e_item, "clip"), "")
            if clip_name.byte_length() > 0 or not gref.is_empty():
                var ac_json = (
                    '{"graph_id": 0, "clip_name": ' + json_quote(clip_name)
                    + ', "graph_ref": ' + gref.write_json()
                    + ', "playing": true, "loop": true}'
                )
                pf.nodes[slot].set_raw_component("AnimatorComponent", ac_json)

            var audio_node = doc.object_get(e_item, "audio_ref")
            if audio_node < 0:
                audio_node = doc.object_get(e_item, "audio")
            var aref = AssetRef()
            if audio_node >= 0:
                if doc.nodes[audio_node].kind == 3:
                    var ap = doc.text(audio_node, "")
                    if ap.byte_length() > 0:
                        aref = AssetRef.from_path(ap, ASSET_KIND_AUDIO)
                elif doc.nodes[audio_node].kind == 5:
                    aref = AssetRef.read_json(doc, audio_node)
                    if aref.kind == ASSET_KIND_RAW:
                        aref.kind = ASSET_KIND_AUDIO
            if not aref.is_empty():
                var ae_json = (
                    '{"asset_id": -1, "asset_path": ' + json_quote(aref.rel_path)
                    + ', "audio_ref": ' + aref.write_json()
                    + "}"
                )
                pf.nodes[slot].set_raw_component("AudioEmitterComponent", ae_json)

            var mass_node = doc.object_get(e_item, "mass")
            if mass_node >= 0 and doc.nodes[mass_node].kind == 2:
                var mval = Float32(doc.number(mass_node, 0.0))
                if isfinite(mval) and mval > Float32(0.0):
                    var rb_json = '{"motion_mode": 2, "mass": ' + String(Float64(mval)) + "}"
                    pf.nodes[slot].set_raw_component("RigidBodyComponent", rb_json)
                else:
                    pf.nodes[slot].set_raw_component("RigidBodyComponent", '{"motion_mode": 0, "mass": 0.0}')

    var nrefs_node = doc.object_get(doc.root, "nested_refs")
    if nrefs_node >= 0:
        if doc.nodes[nrefs_node].kind != 4:
            raise Error("invalid nested_refs array in .flprefab")
        for i in range(doc.array_size(nrefs_node)):
            var nr_item = doc.at(nrefs_node, i)
            if doc.nodes[nr_item].kind != 5:
                raise Error("invalid nested_ref entry in .flprefab")
            var child_name = doc.text(doc.object_get(nr_item, "prefab_name"), "")
            var attach_slot = doc.integer(doc.object_get(nr_item, "attach_parent_slot"), -1)
            var lo_node = doc.object_get(nr_item, "local_offset")
            var lo = LocalTransform.read_json(doc, lo_node) if lo_node >= 0 else LocalTransform.identity()
            var nref = NestedPrefabRef(child_name, attach_slot, lo)
            var ovs_node = doc.object_get(nr_item, "overrides")
            if ovs_node >= 0 and doc.nodes[ovs_node].kind == 4:
                for oi in range(doc.array_size(ovs_node)):
                    var ov_item = doc.at(ovs_node, oi)
                    var ts = doc.integer(doc.object_get(ov_item, "target_slot"), -1)
                    var tn = doc.text(doc.object_get(ov_item, "target_name"), "")
                    var sn = doc.text(doc.object_get(ov_item, "schema_name"), "")
                    var jp = doc.text(doc.object_get(ov_item, "json_payload"), "{}")
                    nref.add_override(PrefabOverride(ts, tn, sn, jp))
            pf.add_nested(nref^)

    return pf^


def load_flprefab_file(path: String) raises -> Prefab:
    var data = read_bytes(path)
    if len(data) == 0:
        raise Error("failed to read .flprefab file: " + path)
    return load_flprefab_bytes(data)


# ---------------------------------------------------------------------------
# Scene: .flscn Deterministic JSON + .flscnb LZ4 Binary + PIE Snapshot/Restore
# ---------------------------------------------------------------------------

def _string_to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def _collect_sorted_world_entities(world: World) -> List[Entity]:
    """Returns all live entities ordered deterministically by ascending `Entity.index`."""
    var out = List[Entity](capacity=world.entity_count())
    for idx in range(len(world.entities)):
        if world.entities[idx].alive:
            out.append(Entity(UInt32(idx), world.entities[idx].generation))
    return out^


def duplicate_entities(mut world: World, registry: ComponentRegistry, sources: List[Entity]) -> List[Entity]:
    """Copies `sources` with every registered component (through the registry's binary codecs, so a
    newly registered component type is copied without touching this code) plus unknown components.
    References inside the set (parent links, slot-hooked fields such as bone sockets) point at the
    copies; references to entities outside the set keep pointing at the originals, and a source whose
    parent is outside the set is parented to that same parent. Returns the copies in `sources` order,
    or an empty list (and a world without any partial copy) when a component fails to round-trip."""
    var n = len(sources)
    var out = List[Entity](capacity=n)
    if n == 0:
        return out^
    var max_idx = len(world.entities)
    var ent_to_slot = List[Int](length=max(1, max_idx), fill=-1)
    for k in range(n):
        if not world.is_alive(sources[k]) or ent_to_slot[Int(sources[k].index)] >= 0:
            return out^  # dead or repeated source
        ent_to_slot[Int(sources[k].index)] = k
    # Live entities outside the set get slots after the copies that resolve back to themselves.
    var outside = List[Entity]()
    for idx in range(max_idx):
        if world.entities[idx].alive and ent_to_slot[idx] < 0:
            ent_to_slot[idx] = n + len(outside)
            outside.append(Entity(UInt32(idx), world.entities[idx].generation))
    var sorted_codecs = registry.sorted_entry_indices()
    var blobs = List[List[UInt8]](capacity=n)
    var counts = List[Int](capacity=n)
    for k in range(n):
        var w = BlobWriter()
        var c = 0
        for ci in range(len(sorted_codecs)):
            var cidx = sorted_codecs[ci]
            if registry.has_on_entity(cidx, world, sources[k]):
                if not registry.write_entity_bin_slots(cidx, world, sources[k], ent_to_slot, w):
                    return out^
                c += 1
        var data = List[UInt8]()
        swap(data, w.data)
        blobs.append(data^)
        counts.append(c)
    registry.ensure_registered_in_world(world)
    for _ in range(n):
        out.append(world.spawn_empty())
    var ok = True
    for k in range(n):
        if not ok:
            break
        var r = BlobReader(blobs[k].copy())
        for _ in range(counts[k]):
            var cidx = registry.find_by_fourcc(r.peek_tag())
            if cidx < 0 or not registry.read_entity_bin(cidx, world, out[k], r) or r.failed:
                ok = False
                break
        if ok:
            var unk_opt = world.get_ptr[UnknownComponentsComponent](sources[k])
            if unk_opt:
                var unk = unk_opt.value()[unsafe_offset=0].copy()
                _ = world.insert[UnknownComponentsComponent](out[k], unk^)
    if not ok:
        for k in range(n):
            _ = world.despawn(out[k])
        return List[Entity]()
    for k in range(n):
        var p_opt = world.get_ptr[Parent](sources[k])
        if not p_opt:
            continue
        var pe = p_opt.value()[unsafe_offset=0].entity
        if not world.is_alive(pe):
            continue
        var ps = ent_to_slot[Int(pe.index)] if Int(pe.index) < len(ent_to_slot) else -1
        var target = out[ps] if ps >= 0 and ps < n else pe
        _ = set_parent(world, out[k], target)
    var spawned = out.copy()
    for oi in range(len(outside)):
        spawned.append(outside[oi])
    for ci in range(len(registry.entries)):
        if registry.entries[ci].has_slot_hooks:
            for k in range(n):
                if registry.has_on_entity(ci, world, out[k]):
                    registry.resolve_entity_slots(ci, world, out[k], spawned)
    return out^


struct Scene(Movable):
    """Flint v5 Scene container supporting `.flscn` JSON, `.flscnb` LZ4 binary, and PIE snapshots."""
    var name: String
    var pie_snapshot: List[UInt8]
    var has_pie_snapshot: Bool
    var migrations: SceneMigrationRegistry

    def __init__(out self, name: String = "UntitledScene"):
        self.name = name
        self.pie_snapshot = List[UInt8]()
        self.has_pie_snapshot = False
        self.migrations = SceneMigrationRegistry()

    # ---- Instant Play-In-Editor (PIE) World Snapshot / Restore (<0.2 ms) ----

    def enter_pie(mut self, world: World) raises:
        """Captures an instant binary snapshot of `world` prior to Play-In-Editor simulation."""
        var writer = BlobWriter()
        world.snapshot_blob(writer)
        self.pie_snapshot = List[UInt8]()
        swap(self.pie_snapshot, writer.data)
        self.has_pie_snapshot = True

    def exit_pie(self, mut world: World) raises:
        """Restores `world` bit-for-bit to its pre-PIE state in `<0.2 ms`."""
        if not self.has_pie_snapshot or len(self.pie_snapshot) == 0:
            raise Error("Scene.exit_pie called without an active PIE snapshot")
        var reader = BlobReader(self.pie_snapshot.copy())
        world.restore_blob(reader)

    # ---- .flscn Deterministic JSON Format (via assets/json.mojo + json_write.mojo) ----

    def save_flscn_string(self, world: World, registry: ComponentRegistry) raises -> String:
        """Serializes `world` into canonical, deterministic `.flscn` JSON text."""
        var ents = _collect_sorted_world_entities(world)
        var max_idx = len(world.entities)
        var ent_to_slot = List[Int](length=max(1, max_idx), fill=-1)
        for slot in range(len(ents)):
            ent_to_slot[Int(ents[slot].index)] = slot

        var sorted_codecs = registry.sorted_entry_indices()

        var raw = String("{\"format\": \"flscn\", \"version\": ") + String(SCENE_SCHEMA_VERSION)
        raw += ", \"name\": " + json_quote(self.name) + ", \"entities\": ["

        for slot in range(len(ents)):
            if slot > 0:
                raw += ", "
            var e = ents[slot]
            var parent_slot = -1
            var p_opt = world.get_ptr[Parent](e)
            if p_opt:
                var pe = p_opt.value()[unsafe_offset=0].entity
                if world.is_alive(pe) and Int(pe.index) < len(ent_to_slot):
                    parent_slot = ent_to_slot[Int(pe.index)]

            raw += "{\"slot\": " + String(slot) + ", \"parent\": " + String(parent_slot) + ", \"components\": {"
            var wrote_comp = 0
            for ci in range(len(sorted_codecs)):
                var cidx = sorted_codecs[ci]
                if registry.has_on_entity(cidx, world, e):
                    var comp_json = String()
                    if registry.write_entity_json_slots(cidx, world, e, ent_to_slot, comp_json):
                        if wrote_comp > 0:
                            raw += ", "
                        raw += json_quote(registry.entries[cidx].schema_name) + ": " + comp_json
                        wrote_comp += 1
            var unk_opt = world.get_ptr[UnknownComponentsComponent](e)
            if unk_opt:
                ref unk = unk_opt.value()[unsafe_offset=0]
                var jorder = unk.sorted_json_order()
                for ui in range(len(jorder)):
                    var uidx = jorder[ui]
                    if wrote_comp > 0:
                        raw += ", "
                    raw += json_quote(unk.json_names[uidx]) + ": " + unk.json_payloads[uidx]
                    wrote_comp += 1
                if len(unk.bin_fourccs) > 0:
                    if wrote_comp > 0:
                        raw += ", "
                    raw += json_quote(String(_SCENE_UNKNOWN_BIN_KEY)) + ": ["
                    var border = unk.sorted_bin_order()
                    for bi in range(len(border)):
                        if bi > 0:
                            raw += ", "
                        var bidx = border[bi]
                        raw += (
                            "{\"fourcc\": " + String(Int(unk.bin_fourccs[bidx]))
                            + ", \"version\": " + String(Int(unk.bin_versions[bidx]))
                            + ", \"hex\": " + json_quote(_bytes_to_hex(unk.bin_payloads[bidx]))
                            + "}"
                        )
                    raw += "]"
                    wrote_comp += 1
            raw += "}}"
        raw += "]}"

        # Canonicalize & validate through assets/json.mojo + assets/json_write.mojo
        var doc = parse_json(_string_to_bytes(raw))
        if not doc.valid:
            raise Error("generated .flscn JSON failed strict validation")
        return json_text(doc, doc.root) + "\n"

    def save_flscn_bytes(self, world: World, registry: ComponentRegistry) raises -> List[UInt8]:
        return _string_to_bytes(self.save_flscn_string(world, registry))

    def save_flscn_file(self, path: String, world: World, registry: ComponentRegistry) raises -> Bool:
        var data = self.save_flscn_bytes(world, registry)
        return write_file_durable(path, data)  # tmp + fsync + rename: a crash never leaves half a scene

    def load_flscn_bytes(
        mut self,
        mut world: World,
        registry: ComponentRegistry,
        data: List[UInt8],
        clear_existing: Bool = True,
    ) raises -> List[Entity]:
        """Parses `.flscn` JSON and populates `world`, running `self.migrations` when `version < SCENE_SCHEMA_VERSION`."""
        var mig = self.migrations.copy()
        return self._load_flscn_bytes_impl(world, registry, data, clear_existing, mig)

    def load_flscn_bytes_with_migrations(
        mut self,
        mut world: World,
        registry: ComponentRegistry,
        data: List[UInt8],
        migrations: SceneMigrationRegistry,
        clear_existing: Bool = True,
    ) raises -> List[Entity]:
        """Parses `.flscn` JSON using an explicit `SceneMigrationRegistry`."""
        return self._load_flscn_bytes_impl(world, registry, data, clear_existing, migrations)

    def _load_flscn_bytes_impl(
        mut self,
        mut world: World,
        registry: ComponentRegistry,
        data: List[UInt8],
        clear_existing: Bool,
        migrations: SceneMigrationRegistry,
    ) raises -> List[Entity]:
        """Parses `.flscn` JSON and populates `world`, reconstructing components and hierarchy."""
        var doc = parse_json(data)
        if not doc.valid:
            raise Error("invalid .flscn JSON document")
        var fmt = doc.text(doc.object_get(doc.root, "format"), "")
        var ver = doc.integer(doc.object_get(doc.root, "version"), -1)
        if fmt != "flscn" or ver < 1 or ver > SCENE_SCHEMA_VERSION:
            raise Error("unsupported .flscn header or version")

        if ver < SCENE_SCHEMA_VERSION and len(migrations.entries) > 0:
            var migrated = migrations.migrate_json(ver, SCENE_SCHEMA_VERSION, json_text(doc, doc.root))
            doc = parse_json(_string_to_bytes(migrated))
            if not doc.valid:
                raise Error("migrated .flscn JSON failed validation")

        var new_name = doc.text(doc.object_get(doc.root, "name"), self.name)
        var ents_arr = doc.object_get(doc.root, "entities")
        if ents_arr >= 0 and doc.nodes[ents_arr].kind != 4:
            raise Error("invalid entities array in .flscn")
        var count = doc.array_size(ents_arr)

        # Stage parse, component decoding, parent range & cycle checks in an isolated World
        # before mutating `world` or `self.name`.
        var staging = World()
        registry.ensure_registered_in_world(staging)
        var spawned = List[Entity](capacity=count)
        var parent_slots = List[Int](length=count, fill=-1)

        for _ in range(count):
            spawned.append(staging.spawn_empty())

        for i in range(count):
            var enode = doc.at(ents_arr, i)
            parent_slots[i] = doc.integer(doc.object_get(enode, "parent"), -1)
            var comps_obj = doc.object_get(enode, "components")
            if comps_obj >= 0 and doc.nodes[comps_obj].kind == 5:
                var num_keys = len(doc.nodes[comps_obj].keys)
                for k in range(num_keys):
                    var sname = doc.nodes[comps_obj].keys[k]
                    var cnode = doc.nodes[comps_obj].children[k]
                    if sname == String(_SCENE_UNKNOWN_BIN_KEY):
                        _parse_unknown_bin_json(doc, cnode, staging, spawned[i])
                        continue
                    var cidx = registry.find_by_name(sname)
                    if cidx < 0:
                        var unk_ptr = staging.get_ptr_mut[UnknownComponentsComponent](spawned[i])
                        if unk_ptr:
                            unk_ptr.value()[unsafe_offset=0].add_json(sname, json_text(doc, cnode))
                        else:
                            var unk = UnknownComponentsComponent()
                            unk.add_json(sname, json_text(doc, cnode))
                            _ = staging.insert[UnknownComponentsComponent](spawned[i], unk^)
                    else:
                        if not registry.read_entity_json(cidx, staging, spawned[i], doc, cnode):
                            raise Error("failed to decode .flscn component: " + sname)

        for i in range(count):
            var p = parent_slots[i]
            if p < -1 or p >= count:
                raise Error("out-of-range parent slot in .flscn scene")
            if p >= 0:
                if not set_parent(staging, spawned[i], spawned[p]):
                    raise Error("cycle or invalid parent in .flscn scene")

        for ci in range(len(registry.entries)):
            if registry.entries[ci].has_slot_hooks:
                for i in range(count):
                    if registry.has_on_entity(ci, staging, spawned[i]):
                        registry.resolve_entity_slots(ci, staging, spawned[i], spawned)

        registry.ensure_registered_in_world(world)
        if clear_existing:
            var snap = staging.snapshot_blob()
            world.restore_blob(snap)
            self.name = new_name
            return spawned^

        var live_spawned = List[Entity](capacity=count)
        for _ in range(count):
            live_spawned.append(world.spawn_empty())
        for i in range(count):
            var enode = doc.at(ents_arr, i)
            var comps_obj = doc.object_get(enode, "components")
            if comps_obj >= 0 and doc.nodes[comps_obj].kind == 5:
                var num_keys = len(doc.nodes[comps_obj].keys)
                for k in range(num_keys):
                    var sname = doc.nodes[comps_obj].keys[k]
                    var cnode = doc.nodes[comps_obj].children[k]
                    if sname == String(_SCENE_UNKNOWN_BIN_KEY):
                        _parse_unknown_bin_json(doc, cnode, world, live_spawned[i])
                        continue
                    var cidx = registry.find_by_name(sname)
                    if cidx >= 0:
                        _ = registry.read_entity_json(cidx, world, live_spawned[i], doc, cnode)
                    else:
                        var unk_ptr = world.get_ptr_mut[UnknownComponentsComponent](live_spawned[i])
                        if unk_ptr:
                            unk_ptr.value()[unsafe_offset=0].add_json(sname, json_text(doc, cnode))
                        else:
                            var unk = UnknownComponentsComponent()
                            unk.add_json(sname, json_text(doc, cnode))
                            _ = world.insert[UnknownComponentsComponent](live_spawned[i], unk^)
        for i in range(count):
            var p = parent_slots[i]
            if p >= 0 and p < count:
                _ = set_parent(world, live_spawned[i], live_spawned[p])
        for ci in range(len(registry.entries)):
            if registry.entries[ci].has_slot_hooks:
                for i in range(count):
                    if registry.has_on_entity(ci, world, live_spawned[i]):
                        registry.resolve_entity_slots(ci, world, live_spawned[i], live_spawned)
        self.name = new_name
        return live_spawned^

    def load_flscn_file(
        mut self,
        path: String,
        mut world: World,
        registry: ComponentRegistry,
        clear_existing: Bool = True,
    ) raises -> List[Entity]:
        var data = read_bytes(path)
        if len(data) == 0:
            raise Error("failed to read .flscn file: " + path)
        return self.load_flscn_bytes(world, registry, data, clear_existing)

    def load_flscn_file_with_migrations(
        mut self,
        path: String,
        mut world: World,
        registry: ComponentRegistry,
        migrations: SceneMigrationRegistry,
        clear_existing: Bool = True,
    ) raises -> List[Entity]:
        var data = read_bytes(path)
        if len(data) == 0:
            raise Error("failed to read .flscn file: " + path)
        return self.load_flscn_bytes_with_migrations(world, registry, data, migrations, clear_existing)

    # ---- .flscnb LZ4 Binary Format (BlobWriter + lz4_pack) ----

    def save_flscnb_bytes(self, world: World, registry: ComponentRegistry) -> List[UInt8]:
        """Serializes `world` into versioned `BlobWriter` chunks and compresses with `lz4_pack`."""
        var ents = _collect_sorted_world_entities(world)
        var max_idx = len(world.entities)
        var ent_to_slot = List[Int](length=max(1, max_idx), fill=-1)
        for slot in range(len(ents)):
            ent_to_slot[Int(ents[slot].index)] = slot

        var sorted_codecs = registry.sorted_entry_indices()
        var writer = BlobWriter()
        writer.begin(SCENE_BIN_MAGIC, SCENE_SCHEMA_VERSION)
        writer.text(self.name)
        writer.integer(len(ents))

        for slot in range(len(ents)):
            var e = ents[slot]
            var parent_slot = -1
            var p_opt = world.get_ptr[Parent](e)
            if p_opt:
                var pe = p_opt.value()[unsafe_offset=0].entity
                if world.is_alive(pe) and Int(pe.index) < len(ent_to_slot):
                    parent_slot = ent_to_slot[Int(pe.index)]
            writer.integer(parent_slot)

            var present_indices = List[Int]()
            for ci in range(len(sorted_codecs)):
                var cidx = sorted_codecs[ci]
                if registry.has_on_entity(cidx, world, e):
                    present_indices.append(cidx)

            var unk_opt = world.get_ptr[UnknownComponentsComponent](e)
            var unk_bin_count = len(unk_opt.value()[unsafe_offset=0].bin_fourccs) if unk_opt else 0
            var has_unk_json = Bool(unk_opt) and len(unk_opt.value()[unsafe_offset=0].json_names) > 0
            writer.integer(len(present_indices) + unk_bin_count + (1 if has_unk_json else 0))
            for pi in range(len(present_indices)):
                _ = registry.write_entity_bin_slots(present_indices[pi], world, e, ent_to_slot, writer)
            if unk_opt:
                ref unk = unk_opt.value()[unsafe_offset=0]
                var border = unk.sorted_bin_order()
                for bi in range(len(border)):
                    var bidx = border[bi]
                    writer.begin(unk.bin_fourccs[bidx], Int(unk.bin_versions[bidx]))
                    if len(unk.bin_payloads[bidx]) > 0:
                        writer.write_bytes_raw(
                            Int(unk.bin_payloads[bidx].unsafe_ptr()),
                            len(unk.bin_payloads[bidx]),
                        )
                    writer.end()
                if has_unk_json:
                    writer.begin(SCENE_UNKNOWN_JSON_FOURCC, 1)
                    var jorder = unk.sorted_json_order()
                    writer.u32(UInt32(len(jorder)))
                    for ji in range(len(jorder)):
                        var jidx = jorder[ji]
                        writer.text(unk.json_names[jidx])
                        writer.text(unk.json_payloads[jidx])
                    writer.end()

        writer.end()
        return lz4_pack(writer.data)

    def save_flscnb_file(self, path: String, world: World, registry: ComponentRegistry) -> Bool:
        var packed = self.save_flscnb_bytes(world, registry)
        return write_file_durable(path, packed)  # tmp + fsync + rename: a crash never leaves half a scene

    def load_flscnb_bytes(
        mut self,
        mut world: World,
        registry: ComponentRegistry,
        packed: List[UInt8],
        clear_existing: Bool = True,
    ) raises -> List[Entity]:
        """Decompresses and validates `.flscnb` LZ4 binary scene into `world`."""
        var raw = lz4_unpack(packed)
        if len(raw) == 0:
            raise Error("corrupted or invalid LZ4 frame in .flscnb")
        var raw_backup = List[UInt8]()
        if not clear_existing:
            raw_backup = raw.copy()
        var reader = BlobReader(raw^)
        var version = reader.enter(SCENE_BIN_MAGIC, SCENE_SCHEMA_VERSION)
        # v1 (pre unknown-component blobs) wrote entity references as raw handles, not scene slots;
        # v2 adds UKJS / unknown FourCC blobs and slot-relative references. Anything else is refused.
        if version != 1 and version != SCENE_SCHEMA_VERSION:
            raise Error(".flscnb version " + String(version) + " is not supported by this build")
        var legacy_v1 = version == 1
        var new_name = reader.text()
        var count = reader.integer()
        if (
            reader.failed
            or count < 0
            or count > 1 << 20
            or count > max(0, reader.limit() - reader.offset) // 16
        ):
            raise Error("corrupted entity count in .flscnb")

        # Stage parse, component decoding, parent range & cycle checks in an isolated World
        # before mutating `world` or `self.name`.
        var staging = World()
        registry.ensure_registered_in_world(staging)
        var spawned = List[Entity](capacity=count)
        var parent_slots = List[Int](length=count, fill=-1)
        for _ in range(count):
            spawned.append(staging.spawn_empty())

        for slot in range(count):
            parent_slots[slot] = reader.integer()
            var num_comps = reader.integer()
            if (
                reader.failed
                or num_comps < 0
                or num_comps > 65536
                or num_comps > max(0, reader.limit() - reader.offset) // 12
            ):
                raise Error("corrupted component count in .flscnb")
            for _ in range(num_comps):
                var tag = reader.peek_tag()
                var cidx = registry.find_by_fourcc(tag)
                if cidx < 0 and legacy_v1:
                    raise Error("unregistered component FourCC in v1 .flscnb")
                if cidx < 0:
                    if tag == SCENE_UNKNOWN_JSON_FOURCC:
                        var jver = reader.enter(SCENE_UNKNOWN_JSON_FOURCC, 1)
                        _ = jver
                        var nj = Int(reader.u32())
                        if (
                            reader.failed
                            or nj < 0
                            or nj > 65536
                            or nj > max(0, reader.limit() - reader.offset) // 8
                        ):
                            raise Error("corrupted UKJS unknown json count in .flscnb")
                        for _ in range(nj):
                            var jn = reader.text()
                            var jp = reader.text()
                            if reader.failed:
                                raise Error("truncated UKJS unknown json entry in .flscnb")
                            var unk_ptr = staging.get_ptr_mut[UnknownComponentsComponent](spawned[slot])
                            if unk_ptr:
                                unk_ptr.value()[unsafe_offset=0].add_json(jn, jp)
                            else:
                                var unk = UnknownComponentsComponent()
                                unk.add_json(jn, jp)
                                _ = staging.insert[UnknownComponentsComponent](spawned[slot], unk^)
                        reader.leave()
                    else:
                        var raw_blob = reader.read_raw_blob()
                        var unk_ptr = staging.get_ptr_mut[UnknownComponentsComponent](spawned[slot])
                        if unk_ptr:
                            unk_ptr.value()[unsafe_offset=0].add_bin(raw_blob[0], raw_blob[1], raw_blob[2].copy())
                        else:
                            var unk = UnknownComponentsComponent()
                            unk.add_bin(raw_blob[0], raw_blob[1], raw_blob[2].copy())
                            _ = staging.insert[UnknownComponentsComponent](spawned[slot], unk^)
                else:
                    if not registry.read_entity_bin(cidx, staging, spawned[slot], reader):
                        raise Error("failed to decode binary component in .flscnb")

        reader.leave()
        if reader.failed:
            raise Error("truncated or corrupted .flscnb scene tail")

        for slot in range(count):
            var p = parent_slots[slot]
            if p < -1 or p >= count:
                raise Error("out-of-range parent slot in .flscnb")
            if p >= 0:
                if not set_parent(staging, spawned[slot], spawned[p]):
                    raise Error("cycle or invalid parent in .flscnb")

        for ci in range(len(registry.entries)):
            if registry.entries[ci].has_slot_hooks and not legacy_v1:  # v1 refs are raw handles
                for slot in range(count):
                    if registry.has_on_entity(ci, staging, spawned[slot]):
                        registry.resolve_entity_slots(ci, staging, spawned[slot], spawned)

        registry.ensure_registered_in_world(world)
        if clear_existing:
            var snap = staging.snapshot_blob()
            world.restore_blob(snap)
            self.name = new_name
            return spawned^

        var reader2 = BlobReader(raw_backup^)
        _ = reader2.enter(SCENE_BIN_MAGIC, SCENE_SCHEMA_VERSION)
        _ = reader2.text()
        _ = reader2.integer()
        var live_spawned = List[Entity](capacity=count)
        for _ in range(count):
            live_spawned.append(world.spawn_empty())
        for slot in range(count):
            _ = reader2.integer()
            var num_comps = reader2.integer()
            for _ in range(num_comps):
                var tag = reader2.peek_tag()
                var cidx = registry.find_by_fourcc(tag)
                if cidx >= 0:
                    _ = registry.read_entity_bin(cidx, world, live_spawned[slot], reader2)
                elif tag == SCENE_UNKNOWN_JSON_FOURCC:
                    _ = reader2.enter(SCENE_UNKNOWN_JSON_FOURCC, 1)
                    var nj = Int(reader2.u32())
                    for _ in range(nj):
                        var jn = reader2.text()
                        var jp = reader2.text()
                        var unk_ptr = world.get_ptr_mut[UnknownComponentsComponent](live_spawned[slot])
                        if unk_ptr:
                            unk_ptr.value()[unsafe_offset=0].add_json(jn, jp)
                        else:
                            var unk = UnknownComponentsComponent()
                            unk.add_json(jn, jp)
                            _ = world.insert[UnknownComponentsComponent](live_spawned[slot], unk^)
                    reader2.leave()
                else:
                    var raw_blob = reader2.read_raw_blob()
                    var unk_ptr = world.get_ptr_mut[UnknownComponentsComponent](live_spawned[slot])
                    if unk_ptr:
                        unk_ptr.value()[unsafe_offset=0].add_bin(raw_blob[0], raw_blob[1], raw_blob[2].copy())
                    else:
                        var unk = UnknownComponentsComponent()
                        unk.add_bin(raw_blob[0], raw_blob[1], raw_blob[2].copy())
                        _ = world.insert[UnknownComponentsComponent](live_spawned[slot], unk^)
        reader2.leave()
        for slot in range(count):
            var p = parent_slots[slot]
            if p >= 0 and p < count:
                _ = set_parent(world, live_spawned[slot], live_spawned[p])
        for ci in range(len(registry.entries)):
            if registry.entries[ci].has_slot_hooks and not legacy_v1:  # v1 refs are raw handles
                for slot in range(count):
                    if registry.has_on_entity(ci, world, live_spawned[slot]):
                        registry.resolve_entity_slots(ci, world, live_spawned[slot], live_spawned)
        self.name = new_name
        return live_spawned^

    def load_flscnb_file(
        mut self,
        path: String,
        mut world: World,
        registry: ComponentRegistry,
        clear_existing: Bool = True,
    ) raises -> List[Entity]:
        var packed = read_bytes(path)
        if len(packed) == 0:
            raise Error("failed to read .flscnb file: " + path)
        return self.load_flscnb_bytes(world, registry, packed, clear_existing)
