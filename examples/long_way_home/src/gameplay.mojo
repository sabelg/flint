"""Hot-reloadable FlintPluginApiV2 `.so` gameplay plugin for *The Long Way Home*."""
from std.sys import size_of
from flint.app.hot_reload import (
    FLINT_PLUGIN_API_VERSION,
    FlintPluginApiV2,
    POD_FIELD_F32,
    POD_FIELD_INT,
    PodComponentSchema,
    PodFieldDesc,
)
from flint.ecs.world import STORAGE_TABLE, comp_hash64_str

comptime STORY_PROGRESS_COMP_ID: UInt64 = comp_hash64_str("StoryJourneyState")
comptime STORY_RES_ID: UInt64 = comp_hash64_str("long_way_home.StoryRuntimeStats")


@fieldwise_init
struct StoryJourneyState(Copyable, ImplicitlyCopyable, Movable):
    """POD component tracking cozy chapter progress & ambient interactions."""
    var chapter_idx: Int32      # 0=Emberleaf, 1=Water City, 2=Starfall Dunes, 3=Thousand Gates
    var path_progress: Float32  # 0.0 .. 1.0 along current chapter
    var bell_resonance: Float32 # decays gently after ringing the Kyoto temple bell
    var dog_joy_hop: Float32    # Shiba playful leap phase (0..1)
    var leaves_scattered: Int32


@fieldwise_init
struct StoryRuntimeStats(Copyable, ImplicitlyCopyable, Movable):
    var ticks: Int32
    var reload_count: Int32
    var ambient_clock: Float32


def build_story_journey_schema() -> PodComponentSchema:
    var s = PodComponentSchema(
        "StoryJourneyState",
        size_of[StoryJourneyState](),
        4,
        STORAGE_TABLE,
    )
    _ = s.add_field(PodFieldDesc("chapter_idx", POD_FIELD_INT, 0, 4, Float32(0.0), Float32(3.0), Float32(1.0)))
    _ = s.add_field(PodFieldDesc("path_progress", POD_FIELD_F32, 4, 4, Float32(0.0), Float32(1.0), Float32(0.01)))
    _ = s.add_field(PodFieldDesc("bell_resonance", POD_FIELD_F32, 8, 4, Float32(0.0), Float32(1.0), Float32(0.02)))
    _ = s.add_field(PodFieldDesc("dog_joy_hop", POD_FIELD_F32, 12, 4, Float32(0.0), Float32(1.0), Float32(0.05)))
    _ = s.add_field(PodFieldDesc("leaves_scattered", POD_FIELD_INT, 16, 4, Float32(0.0), Float32(999.0), Float32(1.0)))
    return s^


def _step_story_row(_entity_bits: UInt64, comp_ptr: Int, dt: Float32) abi("C") -> None:
    if comp_ptr == 0:
        return
    ref st = Pointer[StoryJourneyState, MutAnyOrigin](unsafe_from_address=comp_ptr)[unsafe_offset=0]
    if st.bell_resonance > Float32(0.0):
        st.bell_resonance = max(Float32(0.0), st.bell_resonance - dt * Float32(0.35))
    if st.dog_joy_hop > Float32(0.0):
        st.dog_joy_hop = max(Float32(0.0), st.dog_joy_hop - dt * Float32(1.4))


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
    var schema = build_story_journey_schema()
    if api.register_schema(world_ptr, res_ptr, schema) < 0:
        return Int32(-3)
    if res_ptr != 0:
        var prev_ptr = api.get_resource_ptr(res_ptr, STORY_RES_ID)
        var st = StoryRuntimeStats(Int32(0), Int32(1) if is_reload != 0 else Int32(0), Float32(0.0))
        if prev_ptr != 0:
            ref prev = Pointer[StoryRuntimeStats, ImmutAnyOrigin](unsafe_from_address=prev_ptr)[unsafe_offset=0]
            st.ticks = prev.ticks
            st.reload_count = prev.reload_count + (Int32(1) if is_reload != 0 else Int32(0))
            st.ambient_clock = prev.ambient_clock
        var dst_ptr = api.upsert_pod_resource(res_ptr, STORY_RES_ID, size_of[StoryRuntimeStats](), 0)
        if dst_ptr != 0:
            Pointer[StoryRuntimeStats, MutAnyOrigin](unsafe_from_address=dst_ptr)[unsafe_offset=0] = st
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
        world_ptr, STORY_PROGRESS_COMP_ID, size_of[StoryJourneyState](), 4, 0
    )
    if tid >= 0:
        _ = api.for_each_pod1(world_ptr, tid, _step_story_row, dt)
    if res_ptr != 0:
        var ptr = api.get_resource_ptr(res_ptr, STORY_RES_ID)
        if ptr != 0:
            ref st = Pointer[StoryRuntimeStats, MutAnyOrigin](unsafe_from_address=ptr)[unsafe_offset=0]
            st.ticks += Int32(1)
            st.ambient_clock += dt
    return Int32(1)


@export("flint_plugin_on_unload")
def flint_plugin_on_unload(
    _api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32
) abi("C") -> None:
    pass
