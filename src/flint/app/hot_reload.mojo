"""Flint v5 `<300 ms` Gameplay `.so` Hot-Reload Host & C-ABI Bridge (`flint.app.hot_reload`).

Provides:
- `FLINT_PLUGIN_API_VERSION` (`1`) and `FlintPluginApiV1`: host-resident C-ABI vtable allowing
  compiled `.so` gameplay modules (`--emit shared-lib`) to spawn/despawn entities, register POD
  components, read/write component columns, and access `Resources` without embedding host drop
  function pointers inside unmapped `.so` segments!
- `HotReloadHost`: loads, steps, hot-swaps, and unloads `.so` plugins via `dl_open` / `dl_sym` /
  `dl_close` while keeping `World` and `Resources` 100% intact across reloads.
"""
from std.collections import List
from std.ffi import external_call
from flint.ecs.world import World, Entity, BitSet256
from flint.app.engine_app import Resources
from flint.platform.dynlib import dl_open, dl_sym, dl_close, dl_error, RTLD_NOW, RTLD_LOCAL
from flint.platform.os import get_time, remove_file

comptime FLINT_PLUGIN_API_VERSION: UInt32 = 1

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

def _sym_to_fn[FnT: ImplicitlyCopyable & Deinitable](sym_addr: Int) -> FnT:
    var slot = List[Int](length=1, fill=sym_addr)
    var res = Pointer[FnT, ImmutAnyOrigin](unsafe_from_address=Int(slot.unsafe_ptr()))[unsafe_offset=0]
    _ = len(slot)
    return res
comptime GuestApiVersionFn = def() thin abi("C") -> UInt32
comptime GuestOnLoadFn = def(Int, Int, Int, Int32) thin abi("C") -> Int32
comptime GuestOnStepFn = def(Int, Int, Int, Float32) thin abi("C") -> Int32
comptime GuestOnUnloadFn = def(Int, Int, Int, Int32) thin abi("C") -> None


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


def _host_get_comp_mut(world_addr: Int, entity_bits: UInt64, tid: Int) abi("C") -> Int:
    ref world = Pointer[World, MutAnyOrigin](unsafe_from_address=world_addr)[unsafe_offset=0]
    return world.get_raw_ptr_mut(Entity.from_bits(entity_bits), tid)


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


@fieldwise_init
struct FlintPluginApiV1(Copyable, ImplicitlyCopyable, Movable):
    """Stable C-ABI v1 host table passed to hot-reloadable `.so` gameplay plugins."""
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
        self.abi_version = UInt64(FLINT_PLUGIN_API_VERSION)
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


def _copy_file_fast(src_path: String, dst_path: String) -> Bool:
    var s = src_path
    var d = dst_path
    var rfd = external_call["open", Int32](s.as_c_string_span(), Int32(0), Int32(0))
    if rfd < 0:
        return False
    var wfd = external_call["open", Int32](d.as_c_string_span(), Int32(1 | 64 | 512), Int32(493))
    if wfd < 0:
        _ = external_call["close", Int32](rfd)
        return False
    var buf = List[UInt8](length=65536, fill=0)
    var ok = True
    while True:
        var n = Int(external_call["read", Int64](rfd, buf.unsafe_ptr(), Int64(65536)))
        if n < 0:
            ok = False
            break
        if n == 0:
            break
        var off = 0
        while off < n:
            var w = Int(
                external_call["write", Int64](wfd, buf.unsafe_ptr().unsafe_offset(off), Int64(n - off))
            )
            if w <= 0:
                ok = False
                break
            off += w
        if not ok:
            break
    _ = external_call["close", Int32](wfd)
    _ = external_call["close", Int32](rfd)
    return ok


struct HotReloadHost(Movable):
    """Manages `<300 ms` dynamic `.so` gameplay plugin loading, stepping, and state-preserving hot-swap."""
    var api_box: List[FlintPluginApiV1]
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

    def __init__(out self, stage_prefix: String = "/tmp/flint-hotreload-gen"):
        self.api_box = List[FlintPluginApiV1](capacity=1)
        self.api_box.append(FlintPluginApiV1())
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

    def __deinit__(deinit self):
        if self.lib_handle != 0:
            _ = dl_close(self.lib_handle)
        if self.active_temp_path.byte_length() > 0:
            _ = remove_file(self.active_temp_path)

    @always_inline
    def is_loaded(self) -> Bool:
        return self.lib_handle != 0

    def unload(mut self, mut world: World, mut resources: Resources, is_reload: Bool = False) -> Bool:
        if self.lib_handle == 0:
            return True
        if self.fn_on_unload != 0:
            var ufn = _sym_to_fn[GuestOnUnloadFn](self.fn_on_unload)
            ufn(
                Int(self.api_box.unsafe_ptr()),
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
        if self.active_temp_path.byte_length() > 0:
            _ = remove_file(self.active_temp_path)
            self.active_temp_path = ""
        if rc != 0:
            self.last_error = dl_error()
            return False
        return True

    def load_or_reload(mut self, so_path: String, mut world: World, mut resources: Resources) -> Bool:
        """Loads `so_path` (or hot-swaps from a currently loaded `.so`) while keeping `World` and `Resources` intact."""
        var t0 = get_time()
        var was_loaded = self.is_loaded()
        if was_loaded:
            if not self.unload(world, resources, is_reload=True):
                return False

        var next_gen = self.reload_generation + 1
        var pid = Int(external_call["getpid", Int32]())
        var staged = self.stage_prefix + "-" + String(pid) + "-" + String(next_gen) + ".so"
        if not _copy_file_fast(so_path, staged):
            self.last_error = "failed to stage shared library from " + so_path
            return False

        var handle = dl_open(staged, RTLD_NOW | RTLD_LOCAL)
        if handle == 0:
            self.last_error = dl_error()
            _ = remove_file(staged)
            return False

        var sym_ver = dl_sym(handle, "flint_plugin_api_version")
        var sym_load = dl_sym(handle, "flint_plugin_on_load")
        var sym_step = dl_sym(handle, "flint_plugin_on_step")
        var sym_unload = dl_sym(handle, "flint_plugin_on_unload")
        if sym_ver == 0 or sym_step == 0:
            self.last_error = "missing required plugin entrypoints: " + dl_error()
            _ = dl_close(handle)
            _ = remove_file(staged)
            return False

        var ver_fn = _sym_to_fn[GuestApiVersionFn](sym_ver)
        var found_ver = ver_fn()
        if found_ver != FLINT_PLUGIN_API_VERSION:
            self.last_error = (
                "plugin ABI mismatch: expected "
                + String(FLINT_PLUGIN_API_VERSION)
                + ", found "
                + String(found_ver)
            )
            _ = dl_close(handle)
            _ = remove_file(staged)
            return False

        self.lib_handle = handle
        self.fn_api_version = sym_ver
        self.fn_on_load = sym_load
        self.fn_on_step = sym_step
        self.fn_on_unload = sym_unload
        self.active_temp_path = staged
        self.source_path = so_path
        self.reload_generation = next_gen
        self.last_error = ""

        if self.fn_on_load != 0:
            var lfn = _sym_to_fn[GuestOnLoadFn](self.fn_on_load)
            var code = lfn(
                Int(self.api_box.unsafe_ptr()),
                Int(Pointer(to=world)),
                Int(Pointer(to=resources)),
                Int32(1) if was_loaded else Int32(0),
            )
            if code != 0:
                self.last_error = "flint_plugin_on_load returned error code " + String(code)
                _ = self.unload(world, resources, is_reload=False)
                return False

        self.last_reload_ms = (get_time() - t0) * 1000.0
        return True

    def step(mut self, mut world: World, mut resources: Resources, dt: Float32) -> Int32:
        """Invokes `flint_plugin_on_step` in the active `.so` plugin."""
        if self.lib_handle == 0 or self.fn_on_step == 0:
            return -1
        var sfn = _sym_to_fn[GuestOnStepFn](self.fn_on_step)
        return sfn(
            Int(self.api_box.unsafe_ptr()),
            Int(Pointer(to=world)),
            Int(Pointer(to=resources)),
            dt,
        )
