"""Hot-reloadable FlintPluginApiV2 `.so` gameplay plugin for `test_harness_game`."""
from std.sys import size_of
from flint.app.hot_reload import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2
from gameplay_shared import (
    HARNESS_PLUGIN_RES_ID,
    HarnessPluginTickStats,
    HarnessStats,
    build_harness_stats_schema,
)


def _on_harness_stats_row(_entity_bits: UInt64, comp_ptr: Int, _dt: Float32) abi("C") -> None:
    if comp_ptr == 0:
        return
    ref stats = Pointer[HarnessStats, MutAnyOrigin](unsafe_from_address=comp_ptr)[unsafe_offset=0]
    if stats.max_hp < Int32(1):
        stats.max_hp = Int32(100)
    if stats.hp < Int32(0):
        stats.hp = Int32(0)
    elif stats.hp > stats.max_hp:
        stats.hp = stats.max_hp
    if stats.enemy_hp < Int32(0):
        stats.enemy_hp = Int32(0)


@export("flint_plugin_api_version")
def flint_plugin_api_version() abi("C") -> UInt32:
    return FLINT_PLUGIN_API_VERSION


@export("flint_plugin_on_load")
def flint_plugin_on_load(
    api_ptr: Int, world_ptr: Int, res_ptr: Int, is_reload: Int32
) abi("C") -> Int32:
    if api_ptr == 0 or world_ptr == 0:
        return Int32(-1)
    ref api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]
    if not api.is_valid():
        return Int32(-2)
    var schema = build_harness_stats_schema()
    var tid = api.register_schema(world_ptr, res_ptr, schema)
    if tid < 0:
        return Int32(-3)
    if res_ptr != 0:
        var prev_ptr = api.get_resource_ptr(res_ptr, HARNESS_PLUGIN_RES_ID)
        var pst = HarnessPluginTickStats(Int32(0), Int32(0), Int32(0), Int32(1) if is_reload != 0 else Int32(0))
        if prev_ptr != 0:
            ref prev = Pointer[HarnessPluginTickStats, ImmutAnyOrigin](unsafe_from_address=prev_ptr)[unsafe_offset=0]
            pst.plugin_steps = prev.plugin_steps
            pst.entities_seen = prev.entities_seen
            pst.last_score = prev.last_score
            pst.reload_count = prev.reload_count + (Int32(1) if is_reload != 0 else Int32(0))
        var dst_ptr = api.upsert_pod_resource(
            res_ptr, HARNESS_PLUGIN_RES_ID, size_of[HarnessPluginTickStats](), 0
        )
        if dst_ptr != 0:
            Pointer[HarnessPluginTickStats, MutAnyOrigin](unsafe_from_address=dst_ptr)[unsafe_offset=0] = pst
    return Int32(0)


@export("flint_plugin_on_step")
def flint_plugin_on_step(
    api_ptr: Int, world_ptr: Int, res_ptr: Int, dt: Float32
) abi("C") -> Int32:
    if api_ptr == 0 or world_ptr == 0:
        return Int32(-1)
    ref api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]
    if not api.is_valid():
        return Int32(-2)
    var tid = api.register_pod_component(
        world_ptr, HarnessStats.COMPONENT_ID, size_of[HarnessStats](), 4, 0
    )
    var count = 0
    if tid >= 0:
        count = api.for_each_pod1(world_ptr, tid, _on_harness_stats_row, dt)
    if res_ptr != 0:
        var ptr = api.get_resource_ptr(res_ptr, HARNESS_PLUGIN_RES_ID)
        if ptr != 0:
            ref pst = Pointer[HarnessPluginTickStats, MutAnyOrigin](unsafe_from_address=ptr)[unsafe_offset=0]
            pst.plugin_steps += Int32(1)
            pst.entities_seen = Int32(count)
    return Int32(1)


@export("flint_plugin_on_unload")
def flint_plugin_on_unload(
    _api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32
) abi("C") -> None:
    pass
