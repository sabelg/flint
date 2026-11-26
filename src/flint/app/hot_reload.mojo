"""Flint v5 `<300 ms` Gameplay `.so` Hot-Reload Host & C-ABI Bridge (`flint.app.hot_reload`).

Provides:
- `FLINT_PLUGIN_API_VERSION` (`2`), `FLINT_PLUGIN_API_V2_MAGIC`, `FlintPluginApiV2` (with magic/version/
  `struct_size` validation, ECS/Resource/POD-schema hooks), and `FlintPluginApiV1` (opt-in compat).
- `PodFieldDesc`, `PodComponentSchema`, and `PodSchemaRegistryResource` for runtime gameplay POD schemas
  marshaled by value across the `.so` boundary (zero guest heap/function pointers in host `World`).
- `HotReloadHost`: loads, steps, hot-swaps, and unloads `.so` plugins via `dl_open` / `dl_sym` /
  `dl_close` while keeping `World` and `Resources` 100% intact across reloads.
"""
from std.collections import List
from std.ffi import external_call
from std.math import isfinite, max
from std.sys import size_of
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.ecs.world import (
    BitSet256,
    Entity,
    STORAGE_SPARSE,
    STORAGE_TABLE,
    World,
    comp_hash64,
    comp_hash64_str,
)
from flint.app.engine_app import Resource, Resources
from flint.platform.dynlib import dl_open, dl_sym, dl_close, dl_error, RTLD_NOW, RTLD_LOCAL
from flint.platform.os import close_fd, file_size, get_time, open_path_fd, open_read, remove_file

comptime FLINT_PLUGIN_API_VERSION: UInt32 = 2
comptime FLINT_PLUGIN_API_V2_MAGIC: UInt32 = 0x32415046  # "FPA2"

comptime POD_FIELD_BOOL: Int = 1
comptime POD_FIELD_INT: Int = 2
comptime POD_FIELD_F32: Int = 3
comptime POD_FIELD_STRING: Int = 4
comptime POD_FIELD_VEC3: Int = 5
comptime POD_FIELD_QUAT: Int = 6
comptime POD_FIELD_COLOR_RGB: Int = 7
comptime POD_SCHEMA_BLOB_FOURCC: UInt32 = fourcc("PODS")

# Host C-ABI vtable function pointer types
comptime ApiSpawnEmptyFn = def(Int) thin abi("C") -> UInt64
comptime ApiDespawnFn = def(Int, UInt64) thin abi("C") -> Int32
comptime ApiIsAliveFn = def(Int, UInt64) thin abi("C") -> Int32
comptime ApiEntityCountFn = def(Int) thin abi("C") -> Int
comptime ApiRegisterCompFn = def(Int, UInt64, Int, Int, Int) thin abi("C") -> Int
comptime ApiInsertCompFn = def(Int, UInt64, Int, Int) thin abi("C") -> Int32
comptime ApiGetCompMutFn = def(Int, UInt64, Int) thin abi("C") -> Int
comptime ApiGuestRowCb = def(UInt64, Int, Float32) thin abi("C") -> None
comptime ApiForEach1Fn = def(Int, Int, ApiGuestRowCb, Float32) thin abi("C") -> Int
comptime ApiUpsertResFn = def(Int, UInt64, Int, Int) thin abi("C") -> Int
comptime ApiGetResFn = def(Int, UInt64) thin abi("C") -> Int
comptime ApiGetCompFn = def(Int, UInt64, Int) thin abi("C") -> Int
comptime ApiHasCompFn = def(Int, UInt64, Int) thin abi("C") -> Int32
comptime ApiRemoveCompFn = def(Int, UInt64, Int) thin abi("C") -> Int32
comptime ApiRegisterPodSchemaFn = def(Int, Int, Int, Int) thin abi("C") -> Int

def _sym_to_fn[FnT: ImplicitlyCopyable & Deinitable](sym_addr: Int) -> FnT:
    var slot = sym_addr
    return Pointer(to=slot).unsafe_bitcast[FnT]()[unsafe_offset=0]
comptime GuestApiVersionFn = def() thin abi("C") -> UInt32
comptime GuestOnLoadFn = def(Int, Int, Int, Int32) thin abi("C") -> Int32
comptime GuestOnStepFn = def(Int, Int, Int, Float32) thin abi("C") -> Int32
comptime GuestOnUnloadFn = def(Int, Int, Int, Int32) thin abi("C") -> None


# ---------------------------------------------------------------------------
# Runtime POD Component Schemas (Value-Marshaled Across `.so` Boundary)
# ---------------------------------------------------------------------------

struct PodFieldDesc(Copyable, Movable):
    """Describes a single reflected field inside a hot-reloadable POD component."""
    var name: String
    var kind: Int
    var offset: Int
    var size: Int
    var min_val: Float32
    var max_val: Float32
    var step: Float32

    def __init__(
        out self,
        name: String = "",
        kind: Int = POD_FIELD_F32,
        offset: Int = 0,
        size: Int = 4,
        min_val: Float32 = 0.0,
        max_val: Float32 = 0.0,
        step: Float32 = 0.05,
    ):
        self.name = name
        self.kind = kind
        self.offset = offset
        self.size = size
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.kind = copy.kind
        self.offset = copy.offset
        self.size = copy.size
        self.min_val = copy.min_val
        self.max_val = copy.max_val
        self.step = copy.step

    def is_valid(self, struct_byte_size: Int) -> Bool:
        if self.name.byte_length() == 0 or self.offset < 0 or self.size <= 0:
            return False
        if self.offset + self.size > struct_byte_size:
            return False
        if not isfinite(self.min_val) or not isfinite(self.max_val) or not isfinite(self.step):
            return False
        if self.kind == POD_FIELD_BOOL:
            return self.size == 1 or self.size == 4
        if self.kind == POD_FIELD_INT:
            return self.size == 4 or self.size == 8
        if self.kind == POD_FIELD_F32:
            return self.size == 4
        if self.kind == POD_FIELD_VEC3:
            return self.size == 12
        if self.kind == POD_FIELD_QUAT:
            return self.size == 16
        if self.kind == POD_FIELD_COLOR_RGB:
            return self.size == 12
        return False


struct PodComponentSchema(Copyable, Movable):
    """Value-marshaled layout & reflection schema for a gameplay POD component."""
    var name: String
    var byte_size: Int
    var alignment: Int
    var storage_mode: Int
    var component_hash: UInt64
    var schema_fourcc: UInt32
    var fields: List[PodFieldDesc]

    def __init__(
        out self,
        name: String = "",
        byte_size: Int = 0,
        alignment: Int = 4,
        storage_mode: Int = STORAGE_TABLE,
    ):
        self.name = name
        self.byte_size = byte_size
        self.alignment = alignment
        self.storage_mode = storage_mode
        if name.byte_length() > 0:
            self.component_hash = comp_hash64_str(name)
            self.schema_fourcc = UInt32(0x80000000) | UInt32(self.component_hash & 0x7FFFFFFF)
        else:
            self.component_hash = 0
            self.schema_fourcc = 0
        self.fields = List[PodFieldDesc]()

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.byte_size = copy.byte_size
        self.alignment = copy.alignment
        self.storage_mode = copy.storage_mode
        self.component_hash = copy.component_hash
        self.schema_fourcc = copy.schema_fourcc
        self.fields = copy.fields.copy()

    def add_field(mut self, desc: PodFieldDesc) -> Bool:
        if self.byte_size < 0 or not desc.is_valid(self.byte_size):
            return False
        for i in range(len(self.fields)):
            if self.fields[i].name == desc.name:
                return False
        self.fields.append(desc.copy())
        return True

    def validate(self) -> Bool:
        if self.name.byte_length() == 0 or self.byte_size < 0:
            return False
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            return False
        if self.storage_mode != STORAGE_TABLE and self.storage_mode != STORAGE_SPARSE:
            return False
        for i in range(len(self.fields)):
            if not self.fields[i].is_valid(self.byte_size):
                return False
            for j in range(i + 1, len(self.fields)):
                if self.fields[i].name == self.fields[j].name:
                    return False
        return True

    def matches_layout(self, other: Self) -> Bool:
        if (
            self.name != other.name
            or self.byte_size != other.byte_size
            or self.alignment != other.alignment
            or self.storage_mode != other.storage_mode
            or len(self.fields) != len(other.fields)
        ):
            return False
        for i in range(len(self.fields)):
            ref a = self.fields[i]
            ref b = other.fields[i]
            if a.name != b.name or a.kind != b.kind or a.offset != b.offset or a.size != b.size:
                return False
        return True

    def encode_bytes(self) -> List[UInt8]:
        var out = BlobWriter()
        out.begin(POD_SCHEMA_BLOB_FOURCC, 1)
        out.text(self.name)
        out.integer(self.byte_size)
        out.integer(self.alignment)
        out.integer(self.storage_mode)
        out.u32(UInt32(len(self.fields)))
        for i in range(len(self.fields)):
            ref f = self.fields[i]
            out.text(f.name)
            out.integer(f.kind)
            out.integer(f.offset)
            out.integer(f.size)
            out.f32(f.min_val)
            out.f32(f.max_val)
            out.f32(f.step)
        out.end()
        var result = List[UInt8]()
        swap(result, out.data)
        return result^

    @staticmethod
    def decode_bytes(var data: List[UInt8]) raises -> Self:
        var reader = BlobReader(data^)
        _ = reader.enter(POD_SCHEMA_BLOB_FOURCC, 1)
        var name = reader.text()
        var bsz = reader.integer()
        var aln = reader.integer()
        var smode = reader.integer()
        var n_fields = Int(reader.u32())
        if (
            reader.failed
            or n_fields < 0
            or n_fields > 1024
            or n_fields > max(0, reader.limit() - reader.offset) // 16
        ):
            raise Error("PodComponentSchema: invalid field count")
        var out = Self(name, bsz, aln, smode)
        for _ in range(n_fields):
            var fname = reader.text()
            var fkind = reader.integer()
            var foff = reader.integer()
            var fsize = reader.integer()
            var fmin = reader.f32()
            var fmax = reader.f32()
            var fstep = reader.f32()
            if reader.failed:
                raise Error("PodComponentSchema: truncated field")
            if not out.add_field(PodFieldDesc(fname, fkind, foff, fsize, fmin, fmax, fstep)):
                raise Error("PodComponentSchema: invalid field in decoded schema")
        reader.leave()
        if reader.failed or not out.validate():
            raise Error("PodComponentSchema: invalid schema layout")
        return out^


struct PodSchemaRegistryResource(Resource, Movable):
    """Host-resident registry of POD component schemas registered by gameplay `.so` plugins."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.app.PodSchemaRegistryResource")
    var schemas: List[PodComponentSchema]
    var epoch: UInt64
    var refusal_count: Int
    var last_refusal: String

    def __init__(out self):
        self.schemas = List[PodComponentSchema]()
        self.epoch = 0
        self.refusal_count = 0
        self.last_refusal = ""

    def _refuse(mut self, reason: String) -> Bool:
        self.refusal_count += 1
        self.last_refusal = reason
        return False

    def find_schema(self, name: String) -> Int:
        for i in range(len(self.schemas)):
            if self.schemas[i].name == name:
                return i
        return -1

    def register_schema(mut self, schema: PodComponentSchema) -> Bool:
        """Registers `schema`; refusals bump `refusal_count` and set `last_refusal` so the hot-reload
        host can roll back a plugin whose schemas were refused."""
        if not schema.validate():
            return self._refuse("invalid POD schema '" + schema.name + "'")
        var idx = self.find_schema(schema.name)
        if idx >= 0:
            if self.schemas[idx].matches_layout(schema):
                return True
            return self._refuse(
                "POD schema '" + schema.name + "' changed layout (live components keep the old layout)"
            )
        for i in range(len(self.schemas)):
            if (
                self.schemas[i].component_hash == schema.component_hash
                or self.schemas[i].schema_fourcc == schema.schema_fourcc
            ):
                return self._refuse(
                    "POD schema '" + schema.name + "' collides with '" + self.schemas[i].name + "'"
                )
        self.schemas.append(schema.copy())
        self.epoch += 1
        return True


# ---------------------------------------------------------------------------
# Host-side C-ABI Implementations (all code pages permanently resident in host)
# ---------------------------------------------------------------------------

def _host_spawn_empty(world_addr: Int) abi("C") -> UInt64:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return world.spawn_empty().to_bits()


def _host_despawn(world_addr: Int, entity_bits: UInt64) abi("C") -> Int32:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return Int32(1) if world.despawn(Entity.from_bits(entity_bits)) else Int32(0)


def _host_is_alive(world_addr: Int, entity_bits: UInt64) abi("C") -> Int32:
    ref world = Pointer[World, ImmutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return Int32(1) if world.is_alive(Entity.from_bits(entity_bits)) else Int32(0)


def _host_entity_count(world_addr: Int) abi("C") -> Int:
    ref world = Pointer[World, ImmutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return world.entity_count()


def _host_register_comp(
    world_addr: Int, comp_hash: UInt64, byte_size: Int, alignment: Int, storage_mode: Int
) abi("C") -> Int:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return world.register_pod_component(comp_hash, byte_size, alignment, storage_mode)


def _host_insert_comp(world_addr: Int, entity_bits: UInt64, tid: Int, src_ptr: Int) abi("C") -> Int32:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return Int32(1) if world.insert_raw_pod(Entity.from_bits(entity_bits), tid, src_ptr) else Int32(0)


def _host_get_comp(world_addr: Int, entity_bits: UInt64, tid: Int) abi("C") -> Int:
    ref world = Pointer[World, ImmutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return world.get_raw_ptr(Entity.from_bits(entity_bits), tid)


def _host_get_comp_mut(world_addr: Int, entity_bits: UInt64, tid: Int) abi("C") -> Int:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return world.get_raw_ptr_mut(Entity.from_bits(entity_bits), tid)


def _host_has_comp(world_addr: Int, entity_bits: UInt64, tid: Int) abi("C") -> Int32:
    ref world = Pointer[World, ImmutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return Int32(1) if world.has_by_type_id(Entity.from_bits(entity_bits), tid) else Int32(0)


def _host_remove_comp(world_addr: Int, entity_bits: UInt64, tid: Int) abi("C") -> Int32:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return Int32(1) if world.remove_by_type_id(Entity.from_bits(entity_bits), tid) else Int32(0)


def _host_for_each1(world_addr: Int, tid: Int, cb: ApiGuestRowCb, dt: Float32) abi("C") -> Int:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    if tid < 0 or tid >= len(world.components):
        return 0
    var mask = BitSet256().with_bit(tid)
    var ents = world.query_entities(mask)
    for i in range(len(ents)):
        var e = ents[i]
        var ptr = world.get_raw_ptr_mut(e, tid)
        if ptr != 0:
            cb(e.to_bits(), ptr, dt)
    return len(ents)


def _host_upsert_res(res_addr: Int, res_id: UInt64, byte_size: Int, src_ptr: Int) abi("C") -> Int:
    ref res = Pointer[Resources, MutAnyOrigin](unsafe_from_address=res_addr)[unsafe_offset=0]
    return res.insert_pod_raw(res_id, byte_size, src_ptr)


def _host_get_res(res_addr: Int, res_id: UInt64) abi("C") -> Int:
    ref res = Pointer[Resources, ImmutAnyOrigin](unsafe_from_address=res_addr)[unsafe_offset=0]
    return res.get_raw_ptr(res_id)


def _host_register_pod_schema(
    world_addr: Int, res_addr: Int, bytes_ptr: Int, bytes_len: Int
) abi("C") -> Int:
    if world_addr == 0 or bytes_ptr == 0 or bytes_len <= 0 or bytes_len > (1 << 20):
        return -1
    var buf = List[UInt8](capacity=bytes_len)
    buf.resize(unsafe_uninit_length=bytes_len)
    var src = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=bytes_ptr)
    for bi in range(bytes_len):
        buf[bi] = src[unsafe_offset=bi]
    try:
        var schema = PodComponentSchema.decode_bytes(buf^)
        ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
        var tid = world.register_pod_component(
            schema.component_hash, schema.byte_size, schema.alignment, schema.storage_mode
        )
        if tid < 0:
            if res_addr != 0:
                ref res_r = Pointer[Resources, MutAnyOrigin](unsafe_from_address=res_addr)[unsafe_offset=0]
                if not res_r.has[PodSchemaRegistryResource]():
                    res_r.insert[PodSchemaRegistryResource](PodSchemaRegistryResource())
                var reg_r = res_r.get_ptr_mut[PodSchemaRegistryResource]()
                if reg_r:
                    _ = reg_r.value()[unsafe_offset=0]._refuse(
                        "POD component '" + schema.name + "' changed byte size or storage mode"
                    )
            return -1
        if res_addr != 0:
            ref res = Pointer[Resources, MutAnyOrigin](unsafe_from_address=res_addr)[unsafe_offset=0]
            if not res.has[PodSchemaRegistryResource]():
                res.insert[PodSchemaRegistryResource](PodSchemaRegistryResource())
            var reg_ptr = res.get_ptr_mut[PodSchemaRegistryResource]()
            if not reg_ptr:
                return -1
            if not reg_ptr.value()[unsafe_offset=0].register_schema(schema):
                return -1
        return tid
    except:
        return -1


@fieldwise_init
struct FlintPluginApiV1(Copyable, ImplicitlyCopyable, Movable):
    """Legacy C-ABI v1 host table for opt-in `allow_v1=True` compatibility."""
    var abi_version: UInt64
    var spawn_empty: ApiSpawnEmptyFn
    var despawn: ApiDespawnFn
    var is_alive: ApiIsAliveFn
    var entity_count: ApiEntityCountFn
    var register_pod_component: ApiRegisterCompFn
    var insert_raw_pod: ApiInsertCompFn
    var get_raw_ptr_mut: ApiGetCompMutFn
    var for_each_pod1: ApiForEach1Fn
    var upsert_pod_resource: ApiUpsertResFn
    var get_resource_ptr: ApiGetResFn

    def __init__(out self):
        self.abi_version = UInt64(1)
        self.spawn_empty = _host_spawn_empty
        self.despawn = _host_despawn
        self.is_alive = _host_is_alive
        self.entity_count = _host_entity_count
        self.register_pod_component = _host_register_comp
        self.insert_raw_pod = _host_insert_comp
        self.get_raw_ptr_mut = _host_get_comp_mut
        self.for_each_pod1 = _host_for_each1
        self.upsert_pod_resource = _host_upsert_res
        self.get_resource_ptr = _host_get_res


@no_inline
def _invoke_register_pod_schema(
    fn_ptr: ApiRegisterPodSchemaFn, world_addr: Int, res_addr: Int, var data: List[UInt8]
) -> Int:
    var tid = fn_ptr(world_addr, res_addr, Int(data.unsafe_ptr()), len(data))
    data.clear()
    return tid


@fieldwise_init
struct FlintPluginApiV2(Copyable, ImplicitlyCopyable, Movable):
    """Validated C-ABI v2 host table passed to hot-reloadable `.so` gameplay plugins."""
    var magic: UInt32
    var version: UInt32
    var struct_size: UInt64
    var spawn_empty: ApiSpawnEmptyFn
    var despawn: ApiDespawnFn
    var is_alive: ApiIsAliveFn
    var entity_count: ApiEntityCountFn
    var register_pod_component: ApiRegisterCompFn
    var insert_raw_pod: ApiInsertCompFn
    var get_raw_ptr_mut: ApiGetCompMutFn
    var for_each_pod1: ApiForEach1Fn
    var upsert_pod_resource: ApiUpsertResFn
    var get_resource_ptr: ApiGetResFn
    var get_raw_ptr: ApiGetCompFn
    var has_component: ApiHasCompFn
    var remove_component: ApiRemoveCompFn
    var register_pod_schema: ApiRegisterPodSchemaFn

    def __init__(out self):
        self.magic = FLINT_PLUGIN_API_V2_MAGIC
        self.version = FLINT_PLUGIN_API_VERSION
        self.struct_size = UInt64(size_of[Self]())
        self.spawn_empty = _host_spawn_empty
        self.despawn = _host_despawn
        self.is_alive = _host_is_alive
        self.entity_count = _host_entity_count
        self.register_pod_component = _host_register_comp
        self.insert_raw_pod = _host_insert_comp
        self.get_raw_ptr_mut = _host_get_comp_mut
        self.for_each_pod1 = _host_for_each1
        self.upsert_pod_resource = _host_upsert_res
        self.get_resource_ptr = _host_get_res
        self.get_raw_ptr = _host_get_comp
        self.has_component = _host_has_comp
        self.remove_component = _host_remove_comp
        self.register_pod_schema = _host_register_pod_schema

    @always_inline
    def is_valid(self) -> Bool:
        return (
            self.magic == FLINT_PLUGIN_API_V2_MAGIC
            and self.version == FLINT_PLUGIN_API_VERSION
            and Int(self.struct_size) == size_of[Self]()
        )

    def register_schema(self, world_addr: Int, res_addr: Int, schema: PodComponentSchema) -> Int:
        return _invoke_register_pod_schema(
            self.register_pod_schema, world_addr, res_addr, schema.encode_bytes()
        )

    def register_schema_in(
        self, mut world: World, mut res: Resources, schema: PodComponentSchema
    ) -> Int:
        var tid = self.register_schema(Int(Pointer(to=world)), Int(Pointer(to=res)), schema)
        world.keep_alive()
        _ = res.find_index(0)
        return tid


def _copy_file_fast(src_path: String, dst_path: String) -> Bool:
    var rfd = open_read(src_path)
    if rfd < 0:
        return False
    var wfd = open_path_fd(dst_path, Int32(1 | 64 | 512), Int32(493))
    if wfd < 0:
        close_fd(rfd)
        return False
    var buf = List[UInt8](length=65536, fill=0)
    var ok = True
    while True:
        var n = Int(external_call["read", Int64](Int32(rfd), buf.unsafe_ptr(), Int64(65536)))
        if n < 0:
            ok = False
            break
        if n == 0:
            break
        var off = 0
        while off < n:
            var w = Int(
                external_call["write", Int64](Int32(wfd), buf.unsafe_ptr().unsafe_offset(off), Int64(n - off))
            )
            if w <= 0:
                ok = False
                break
            off += w
        if not ok:
            break
    buf.clear()
    close_fd(wfd)
    close_fd(rfd)
    return ok


struct HotReloadHost(Movable):
    """Manages `<300 ms` dynamic `.so` gameplay plugin loading, stepping, and state-preserving hot-swap."""
    var api_v1_box: List[FlintPluginApiV1]
    var api_v2_box: List[FlintPluginApiV2]
    var allow_v1: Bool
    var active_api_version: UInt32
    var lib_handle: Int
    var fn_api_version: Int
    var fn_on_load: Int
    var fn_on_step: Int
    var fn_on_unload: Int
    var active_temp_path: String
    var source_path: String
    var reload_generation: Int
    var last_reload_ms: Float64
    var last_error: String
    var stage_prefix: String
    var rollback_count: Int
    var stage_counter: Int

    def __init__(
        out self,
        stage_prefix: String = "/tmp/flint-hotreload-gen",
        allow_v1: Bool = False,
    ):
        self.api_v1_box = List[FlintPluginApiV1](capacity=1)
        self.api_v1_box.append(FlintPluginApiV1())
        self.api_v2_box = List[FlintPluginApiV2](capacity=1)
        self.api_v2_box.append(FlintPluginApiV2())
        self.allow_v1 = allow_v1
        self.active_api_version = FLINT_PLUGIN_API_VERSION
        self.lib_handle = 0
        self.fn_api_version = 0
        self.fn_on_load = 0
        self.fn_on_step = 0
        self.fn_on_unload = 0
        self.active_temp_path = ""
        self.source_path = ""
        self.reload_generation = 0
        self.last_reload_ms = 0.0
        self.last_error = ""
        self.stage_prefix = stage_prefix
        self.rollback_count = 0
        self.stage_counter = 0

    def __deinit__(deinit self):
        if self.lib_handle != 0:
            _ = dl_close(self.lib_handle)
        if self.active_temp_path.byte_length() > 0:
            _ = remove_file(self.active_temp_path)

    @always_inline
    def is_loaded(self) -> Bool:
        return self.lib_handle != 0

    @always_inline
    def _active_api_addr(self) -> Int:
        if self.active_api_version == 1:
            return Int(self.api_v1_box.unsafe_ptr())
        return Int(self.api_v2_box.unsafe_ptr())

    def unload(mut self, mut world: World, mut resources: Resources, is_reload: Bool = False) -> Bool:
        if self.lib_handle == 0:
            return True
        if self.fn_on_unload != 0:
            var ufn = _sym_to_fn[GuestOnUnloadFn](self.fn_on_unload)
            ufn(
                self._active_api_addr(),
                Int(Pointer(to=world)),
                Int(Pointer(to=resources)),
                Int32(1) if is_reload else Int32(0),
            )
        var rc = dl_close(self.lib_handle)
        self.lib_handle = 0
        self.fn_api_version = 0
        self.fn_on_load = 0
        self.fn_on_step = 0
        self.fn_on_unload = 0
        self.active_api_version = FLINT_PLUGIN_API_VERSION
        if self.active_temp_path.byte_length() > 0:
            _ = remove_file(self.active_temp_path)
            self.active_temp_path = ""
        if rc != 0:
            self.last_error = dl_error()
            return False
        return True

    def load_or_reload(mut self, so_path: String, mut world: World, mut resources: Resources) -> Bool:
        """Loads `so_path` (or hot-swaps from a currently loaded `.so`) while keeping `World` and `Resources` intact."""
        if so_path.byte_length() == 0 or file_size(so_path) <= 0:
            self.last_error = "shared library not found or empty: " + so_path
            return False

        var t0 = get_time()
        var next_gen = self.reload_generation + 1
        var pid = Int(external_call["getpid", Int32]())
        # Unique staged path per attempt: a failed attempt's path may still be cached by the dynamic
        # loader (libraries that cannot really unload), so a retry must never reuse it.
        self.stage_counter += 1
        var staged = self.stage_prefix + "-" + String(pid) + "-" + String(self.stage_counter) + ".so"
        if not _copy_file_fast(so_path, staged):
            self.last_error = "failed to stage shared library from " + so_path
            _ = remove_file(staged)
            return False

        var handle = dl_open(staged, RTLD_NOW | RTLD_LOCAL)
        if handle == 0:
            self.last_error = dl_error()
            _ = remove_file(staged)
            return False

        var sym_ver = dl_sym(handle, "flint_plugin_api_version")
        if sym_ver == 0:
            self.last_error = "missing required plugin entrypoints: " + dl_error()
            _ = dl_close(handle)
            _ = remove_file(staged)
            return False
        var sym_step = dl_sym(handle, "flint_plugin_on_step")
        if sym_step == 0:
            self.last_error = "missing required plugin entrypoints: " + dl_error()
            _ = dl_close(handle)
            _ = remove_file(staged)
            return False
        var sym_load = dl_sym(handle, "flint_plugin_on_load")
        var sym_unload = dl_sym(handle, "flint_plugin_on_unload")

        var ver_fn = _sym_to_fn[GuestApiVersionFn](sym_ver)
        var found_ver = ver_fn()
        var ver_ok = (found_ver == FLINT_PLUGIN_API_VERSION) or (self.allow_v1 and found_ver == 1)
        if not ver_ok:
            self.last_error = (
                "plugin ABI mismatch: expected "
                + String(FLINT_PLUGIN_API_VERSION)
                + ", found "
                + String(found_ver)
            )
            _ = dl_close(handle)
            _ = remove_file(staged)
            return False

        if not resources.has[PodSchemaRegistryResource]():
            resources.insert[PodSchemaRegistryResource](PodSchemaRegistryResource())

        # Keep the previous plugin mapped until the new one has fully succeeded, so a failing
        # reload (on_load error or schema refusal) can restore it instead of leaving no plugin.
        var was_loaded = self.is_loaded()
        var prev_handle = self.lib_handle
        var prev_fn_api_version = self.fn_api_version
        var prev_fn_on_load = self.fn_on_load
        var prev_fn_on_step = self.fn_on_step
        var prev_fn_on_unload = self.fn_on_unload
        var prev_temp_path = self.active_temp_path
        var prev_source_path = self.source_path
        var prev_api_version = self.active_api_version

        if was_loaded and prev_fn_on_unload != 0:
            var old_ufn = _sym_to_fn[GuestOnUnloadFn](prev_fn_on_unload)
            old_ufn(self._active_api_addr(), Int(Pointer(to=world)), Int(Pointer(to=resources)), Int32(1))

        self.active_api_version = found_ver
        var refusals_before = self._schema_refusals(resources)
        var failure = String("")
        if sym_load != 0:
            var lfn = _sym_to_fn[GuestOnLoadFn](sym_load)
            var code = lfn(
                self._active_api_addr(),
                Int(Pointer(to=world)),
                Int(Pointer(to=resources)),
                Int32(1) if was_loaded else Int32(0),
            )
            if code != 0:
                failure = "flint_plugin_on_load returned error code " + String(code)
        if failure.byte_length() == 0 and self._schema_refusals(resources) > refusals_before:
            failure = "POD schema refused: " + self._last_schema_refusal(resources)

        if failure.byte_length() > 0:
            # Let the failed plugin release what it acquired, then drop it.
            if sym_unload != 0:
                var new_ufn = _sym_to_fn[GuestOnUnloadFn](sym_unload)
                new_ufn(
                    self._active_api_addr(),
                    Int(Pointer(to=world)),
                    Int(Pointer(to=resources)),
                    Int32(1) if was_loaded else Int32(0),
                )
            _ = dl_close(handle)
            _ = remove_file(staged)
            self.active_api_version = prev_api_version
            if not was_loaded:
                self.last_error = failure
                return False
            # Restore the previous plugin: it was told it is being reloaded, so re-run its on_load
            # as a reload to reattach to the preserved World/Resources.
            self.lib_handle = prev_handle
            self.fn_api_version = prev_fn_api_version
            self.fn_on_load = prev_fn_on_load
            self.fn_on_step = prev_fn_on_step
            self.fn_on_unload = prev_fn_on_unload
            self.active_temp_path = prev_temp_path
            self.source_path = prev_source_path
            var restored = True
            if prev_fn_on_load != 0:
                var plfn = _sym_to_fn[GuestOnLoadFn](prev_fn_on_load)
                var rcode = plfn(
                    self._active_api_addr(), Int(Pointer(to=world)), Int(Pointer(to=resources)), Int32(1)
                )
                if rcode != 0:
                    restored = False
                    failure += "; restoring the previous plugin also failed (on_load code " + String(rcode) + ")"
            if restored:
                self.last_error = "hot-reload of " + so_path + " rolled back to the previous plugin: " + failure
            else:
                _ = self.unload(world, resources, is_reload=False)
                self.last_error = "hot-reload of " + so_path + " failed: " + failure
            self.rollback_count += 1
            return False

        # Success: retire the previous plugin now that the new one is live.
        var close_err = String("")
        if was_loaded:
            var rc = dl_close(prev_handle)
            if prev_temp_path.byte_length() > 0:
                _ = remove_file(prev_temp_path)
            if rc != 0:
                close_err = "previous plugin dl_close failed: " + dl_error()
        self.lib_handle = handle
        self.fn_api_version = sym_ver
        self.fn_on_load = sym_load
        self.fn_on_step = sym_step
        self.fn_on_unload = sym_unload
        self.active_temp_path = staged
        self.source_path = so_path
        self.reload_generation = next_gen
        self.active_api_version = found_ver
        self.last_error = close_err

        self.last_reload_ms = (get_time() - t0) * 1000.0
        return True

    @staticmethod
    def _schema_refusals(resources: Resources) -> Int:
        var reg = resources.get_ptr[PodSchemaRegistryResource]()
        return reg.value()[unsafe_offset=0].refusal_count if reg else 0

    @staticmethod
    def _last_schema_refusal(resources: Resources) -> String:
        var reg = resources.get_ptr[PodSchemaRegistryResource]()
        return reg.value()[unsafe_offset=0].last_refusal if reg else String("")

    def step(mut self, mut world: World, mut resources: Resources, dt: Float32) -> Int32:
        """Invokes `flint_plugin_on_step` in the active `.so` plugin."""
        if self.lib_handle == 0 or self.fn_on_step == 0:
            return -1
        var sfn = _sym_to_fn[GuestOnStepFn](self.fn_on_step)
        return sfn(
            self._active_api_addr(),
            Int(Pointer(to=world)),
            Int(Pointer(to=resources)),
            dt,
        )
