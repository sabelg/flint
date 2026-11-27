"""Regression Test Suite for Milestone M2 (`tests/test_studio_core_bugs.mojo`).

Covers Hidden Bugs #1–#5 (F06–F10) & M0 `UiContext` widgets:
  1. Bug #1 (F06): Edit vs. PIE frame/stage scheduling (`FrameTime`, `FixedTime.advance_frame`,
     no `STAGE_UPDATE` in Edit/Paused, no double `STAGE_PRE_UPDATE`/`STAGE_POST_UPDATE`,
     `event_swap_hooks` invocation, bit-identical `pie_stop` restore).
  2. Bug #2 (F07): Post-clear `RenderWorld.overlay_lines` survival across `clear_frame()` and
     `RenderExtractor.begin_extraction`, Tier-0 rendering + post-draw clear, and Studio
     fallback lights preserved via `RenderPluginStateResource.extra_lights`.
  3. Bug #3 (F08): Per-entity `MaterialOverrideComponent` slot isolation, expanded 12-field
     v1/v2 binary & JSON round-trips, slot recycling, and `StudioGpuMaterials` generation safety.
  4. Bug #4 (F09): `UiContext.wants_keyboard()` / `wants_mouse()` / `wants_mouse_at()`, M0
     widgets (`drag_float`, `combo_box`, `color_edit3`) + unclipped deferred popups,
     `Ctrl+Q` replacing bare `Esc` quit, and `process_editor_input` capture barriers.
  5. Bug #5a (F10): `KccTriggerProxy` + `PhysicsWorld._update_triggers()` (`OBB`/`Sphere`/`Capsule`,
     `CollisionFilter`, `PHYW` v3 round-trip) + `PhysicsPlugin` `CharacterControllerComponent`
     trigger events and `lookup_entity()` resolution + despawn `TRIGGER_EXIT`.
  6. Bug #5b (F10): Edit-mode camera isolation (`editor_camera_override`) preventing active
     `CameraComponent` from hijacking Edit/Paused viewport and restoring on `pie_stop()`.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_false, assert_true

from flint.ai import BehaviorTree, BehaviorTreeComponent
from flint.app import (
    FLINT_PLUGIN_API_VERSION,
    FlintPluginApiV2,
    POD_FIELD_BOOL,
    POD_FIELD_F32,
    POD_FIELD_INT,
    POD_FIELD_VEC3,
    PodComponentSchema,
    PodFieldDesc,
)
from flint.input.input_map import ACTION_BOOL, InputContext, InputMap

from flint.anim.flanim_ecs import BoneSocketAttachment, compute_volume_preserving_smear_scale, save_flanim_bytes
from flint.anim.flrig import FrigJointDef, FrigRigDef, save_flrig_file
from flint.anim.graph import AnimGraphDef, StateDef
from flint.anim.plugin import AnimAssetCache, AnimatorComponent, BoneSocketComponent, RigComponent, SkeletonComponent
from flint.assets.flmat import FlmatAsset, save_flmat_file
from flint.assets.asset_plugin import SceneAssetResolver
from flint.assets.server import AssetServer
from flint.assets.watcher import AssetWatcher
from flint.assets import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    ASSET_KIND_TEXTURE,
    AssetRef,
)
from flint.assets.glb_writer import write_native_glb
from flint.assets.image import write_png
from flint.assets.model import NativeMesh, NativeModel, NativeNode, NativeSkin
from flint.app.window_plugin import WindowStateResource
from flint.audio.plugin import AudioEmitterComponent
from flint.audio.server import AudioServer
from flint.nav.plugin import NavPluginStateResource
from flint.app.engine_app import (
    AppConfig,
    EngineApp,
    FixedTime,
    FrameTime,
    Resource,
    Resources,
    STAGE_FIXED_UPDATE,
    STAGE_LAST,
    STAGE_POST_UPDATE,
    STAGE_PRE_UPDATE,
    STAGE_UPDATE,
)
from flint.assets.json import parse_json
from flint.assets.mesh_builder import MeshBuilder
from flint.core.jobs import JobPool
from flint.core.math import Quat, Vec3
from flint.core.serial import BlobReader, BlobWriter
from flint.ecs.world import Entity, Events, ParallelCommands, World, comp_hash64
from flint.physics.character import CharacterController, CharacterControllerConfig
from flint.physics.filter import CAT_ALL, CAT_DEBRIS, CAT_PROP_PIECE, CAT_SENSOR, CollisionFilter
from flint.physics.plugin import (
    CharacterControllerComponent,
    ColliderComponent,
    PhysicsPlugin,
    PhysicsSyncStateResource,
    TriggerEventsResource,
    register_physics_codecs,
)
from flint.physics.world import (
    BODY_SHAPE_BOX,
    PhysicsWorld,
    TRIGGER_ENTER,
    TRIGGER_EXIT,
    TRIGGER_STAY,
    TriggerVolume,
)
from flint.platform.input import InputState
from flint.platform.os import env_now, env_value, file_size, make_dirs, read_bytes, remove_file, write_bytes
from flint.render.aaa_clustered_lights import AAA_LIGHT_SPOT, AaaClusterLight
from flint.render.material_graph import (
    MAT_FLAG_DOUBLE_SIDED,
    MAT_FLAG_RECEIVE_DECALS,
    MaterialAsset,
)
from flint.render.plugin import (
    CameraComponent,
    EnvironmentComponent,
    MaterialOverrideComponent,
    MeshRendererComponent,
    PointLightComponent,
    RenderPlugin,
    RenderPluginStateResource,
    SpotLightComponent,
    register_render_codecs,
)
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RENDER_TIER_1_FORWARD_CLUSTERED,
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    RenderCameraPacket,
    RenderExtractor,
    RenderWorld,
    ScalableRenderer,
)
from flint.ui.draw_list import UiDrawList
from flint.scene.scene import ComponentRegistry, PrefabInstanceComponent, Scene, UnknownComponentsComponent
from flint.scene.transform_hierarchy import (
    GlobalTransform,
    LocalTransform,
    PrevGlobalTransform,
    TransformPlugin,
)
from flint.studio.gizmo import GIZMO_MODE_ROTATE, GIZMO_MODE_SCALE, GIZMO_MODE_TRANSLATE
from flint.studio.studio_app import (
    FieldVisitor,
    FlintStudioState,
    PIE_PAUSED,
    PIE_PLAYING,
    PIE_STOPPED,
    SHADING_NPR_CEL,
    SHADING_PBR,
    SHADING_STYLIZED_VFX,
    StudioEntityMetaComponent,
    StudioStyleComponent,
    _resolve_default_scene_path,
    emit_studio_ui_draw_list,
    populate_default_studio_scene,
    upload_default_studio_gpu_assets,
)
from flint.ui.context import (
    UI_KEY_A,
    UI_KEY_BACKSPACE,
    UI_KEY_DELETE,
    UI_KEY_DOWN,
    UI_KEY_ENTER,
    UI_KEY_ESCAPE,
    UI_KEY_LEFT,
    UI_KEY_LEFT_CTRL,
    UI_KEY_LEFT_SHIFT,
    UI_KEY_RIGHT,
    UI_KEY_TAB,
    UI_KEY_UP,
    UiContext,
    hash_ui_id,
    pack_rgba8,
)
from flint.ui.flexbox_dock import DOCK_SPLIT_H, DockSpace, FlexStyle, RectF


# ---------------------------------------------------------------------------
# Helpers for Bug #1 Probe
# ---------------------------------------------------------------------------

struct _StageProbeResource(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.test.StageProbeResource")
    var pre_count: Int
    var fixed_count: Int
    var update_count: Int
    var post_count: Int
    var last_update_dt: Float32
    var events: Events[Int]

    def __init__(out self):
        self.pre_count = 0
        self.fixed_count = 0
        self.update_count = 0
        self.post_count = 0
        self.last_update_dt = 0.0
        self.events = Events[Int]()


def _probe_pre(
    _w: Pointer[World, MutAnyOrigin],
    r_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var p = r_ptr[unsafe_offset=0].get_ptr_mut[_StageProbeResource]()
    if p:
        p.value()[unsafe_offset=0].pre_count += 1


def _probe_fixed(
    _w: Pointer[World, MutAnyOrigin],
    r_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var p = r_ptr[unsafe_offset=0].get_ptr_mut[_StageProbeResource]()
    if p:
        p.value()[unsafe_offset=0].fixed_count += 1


def _probe_update(
    _w: Pointer[World, MutAnyOrigin],
    r_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref res = r_ptr[unsafe_offset=0]
    var p = res.get_ptr_mut[_StageProbeResource]()
    var ft = res.get_ptr[FrameTime]()
    if p:
        p.value()[unsafe_offset=0].update_count += 1
        if ft:
            p.value()[unsafe_offset=0].last_update_dt = ft.value()[unsafe_offset=0].delta_seconds
        _ = p.value()[unsafe_offset=0].events.send(p.value()[unsafe_offset=0].update_count)


def _probe_post(
    _w: Pointer[World, MutAnyOrigin],
    r_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var p = r_ptr[unsafe_offset=0].get_ptr_mut[_StageProbeResource]()
    if p:
        p.value()[unsafe_offset=0].post_count += 1


def _probe_swap(r_ptr: Pointer[Resources, MutAnyOrigin]):
    var p = r_ptr[unsafe_offset=0].get_ptr_mut[_StageProbeResource]()
    if p:
        p.value()[unsafe_offset=0].events.swap_buffers()


# ---------------------------------------------------------------------------
# 1. Bug #1 (F06): Edit vs. PIE Frame & Stage Scheduling
# ---------------------------------------------------------------------------

def test_bug1_edit_vs_pie_scheduling() raises:
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
    )
    _ = studio.spawn_entity("Box", position=Vec3(0.0, 2.0, 0.0), mass=1.0, record_undo=False)

    studio.app.insert_resource[_StageProbeResource](_StageProbeResource())
    _ = studio.app.add_system(STAGE_PRE_UPDATE, "probe_pre", _probe_pre)
    _ = studio.app.add_system(STAGE_FIXED_UPDATE, "probe_fixed", _probe_fixed)
    _ = studio.app.add_system(STAGE_UPDATE, "probe_update", _probe_update)
    _ = studio.app.add_system(STAGE_POST_UPDATE, "probe_post", _probe_post)
    studio.app.add_event_swap_hook(_probe_swap)

    # A. Edit mode: render_editor_frame updates FrameTime, runs PRE/POST, NEVER STAGE_UPDATE or FIXED
    for _ in range(3):
        studio.render_editor_frame(160, 120, dt=Float32(1.0 / 30.0))

    var probe = studio.app.resources.get_ptr[_StageProbeResource]()
    assert_true(Bool(probe))
    assert_equal(probe.value()[unsafe_offset=0].pre_count, 3)
    assert_equal(probe.value()[unsafe_offset=0].fixed_count, 0)
    assert_equal(probe.value()[unsafe_offset=0].update_count, 0)
    assert_equal(probe.value()[unsafe_offset=0].post_count, 3)

    var ft = studio.app.resources.get_ptr[FrameTime]()
    assert_true(Bool(ft))
    assert_equal(ft.value()[unsafe_offset=0].frame_index, UInt64(3))
    assert_true(abs(ft.value()[unsafe_offset=0].delta_seconds - Float32(1.0 / 30.0)) < Float32(1.0e-5))
    assert_true(abs(ft.value()[unsafe_offset=0].elapsed_seconds - 0.1) < 1.0e-4)

    # B. Paused mode: render_editor_frame still MUST NOT run STAGE_UPDATE
    assert_true(studio.pie_play())
    assert_true(studio.pie_pause())
    studio.render_editor_frame(160, 120, dt=Float32(1.0 / 60.0))
    assert_equal(probe.value()[unsafe_offset=0].update_count, 0)
    assert_equal(probe.value()[unsafe_offset=0].fixed_count, 0)

    # C. Interactive PIE frame: step_pie_frame + render_editor_frame runs PRE/POST/UPDATE once (no double-run!)
    assert_true(studio.pie_play())  # resumes from PAUSED -> PLAYING
    var pre_before = probe.value()[unsafe_offset=0].pre_count
    var post_before = probe.value()[unsafe_offset=0].post_count
    var fixed_before = probe.value()[unsafe_offset=0].fixed_count
    var upd_before = probe.value()[unsafe_offset=0].update_count
    var fix_ticks_before = studio.app.resources.get_ptr[FixedTime]().value()[unsafe_offset=0].tick_count

    var steps = studio.step_pie_frame(Float32(1.0 / 60.0))
    assert_equal(steps, 1)
    assert_true(abs(probe.value()[unsafe_offset=0].last_update_dt - Float32(1.0 / 60.0)) < Float32(1.0e-5))
    studio.render_editor_frame(160, 120, dt=Float32(1.0 / 60.0))

    assert_equal(probe.value()[unsafe_offset=0].pre_count - pre_before, 1)
    assert_equal(probe.value()[unsafe_offset=0].post_count - post_before, 1)
    assert_equal(probe.value()[unsafe_offset=0].fixed_count - fixed_before, 1)
    assert_equal(probe.value()[unsafe_offset=0].update_count - upd_before, 1)
    assert_equal(
        studio.app.resources.get_ptr[FixedTime]().value()[unsafe_offset=0].tick_count - fix_ticks_before,
        UInt64(1),
    )

    # Verify event_swap_hooks flushed the event sent during STAGE_UPDATE
    var ev_cursor = 0
    var evs_after_render = studio.app.resources.get_ptr_mut[_StageProbeResource]().value()[unsafe_offset=0].events.read_since(ev_cursor)
    assert_equal(len(evs_after_render), 1)
    assert_equal(evs_after_render[0], 1)

    # D. FixedTime accumulator across sub-tick dt frames (0.008 + 0.010 = 0.018 >= 1/60)
    var s0 = studio.step_pie_frame(Float32(0.008))
    studio.render_editor_frame(160, 120, dt=Float32(0.008))
    assert_equal(s0, 0)
    var s1 = studio.step_pie_frame(Float32(0.010))
    studio.render_editor_frame(160, 120, dt=Float32(0.010))
    assert_equal(s1, 1)
    assert_equal(probe.value()[unsafe_offset=0].update_count, 3)

    # E. Deterministic pie_step(5) runs STAGE_UPDATE 5x and swaps events each tick; pie_stop restores hash
    var pre_pie_hash = studio.pie.pre_play_hash
    var upd_pre_step = probe.value()[unsafe_offset=0].update_count
    var stepped = studio.pie_step(5, Float32(1.0 / 60.0))
    assert_equal(stepped, 5)
    assert_equal(probe.value()[unsafe_offset=0].update_count - upd_pre_step, 5)
    assert_true(studio.pie_stop())
    assert_equal(studio.state_hash(), pre_pie_hash)

    studio.shutdown()


# ---------------------------------------------------------------------------
# 2. Bug #2 (F07): Post-Clear `overlay_lines` & Studio Fallback Lights
# ---------------------------------------------------------------------------

def test_bug2_overlay_lines_and_studio_lights() raises:
    # A. RenderWorld.clear_frame preserves overlay_lines unless clear_overlay=True
    var rw = RenderWorld()
    rw.debug_lines.add_line(Vec3(0, 0, 0), Vec3(1, 0, 0), pack_rgba8(255, 0, 0, 255))
    rw.overlay_lines.add_line(Vec3(0, 0, 0), Vec3(0, 1, 0), pack_rgba8(0, 255, 0, 255))
    rw.clear_frame(1, Float32(1.0 / 60.0))
    assert_equal(rw.debug_lines.segment_count(), 0)
    assert_equal(rw.overlay_lines.segment_count(), 1)

    var ex = RenderExtractor()
    ex.begin_extraction(rw, 2, Float32(1.0 / 60.0))
    assert_equal(rw.overlay_lines.segment_count(), 1)
    rw.clear_frame(3, Float32(1.0 / 60.0), clear_overlay=True)
    assert_equal(rw.overlay_lines.segment_count(), 0)

    # B. EngineApp + RenderPlugin Tier-0 renders overlay_lines & clears them post-draw
    var app = EngineApp(AppConfig(title="overlay_test", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    app.startup()

    var cam_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(0.0, 0.5, 3.0)))
    _ = app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
    _ = app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=55.0, active=True))

    app.step_one_frame(Float32(1.0 / 60.0))
    var px_base = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()

    var rw_ptr = app.resources.get_ptr_mut[RenderWorld]()
    rw_ptr.value()[unsafe_offset=0].overlay_lines.add_line(
        Vec3(-0.6, 0.5, 0.0), Vec3(0.6, 0.5, 0.0), pack_rgba8(255, 40, 40, 255), thickness_px=3.0
    )
    app.step_one_frame(Float32(1.0 / 60.0))
    var px_with_line = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    assert_equal(rw_ptr.value()[unsafe_offset=0].overlay_lines.segment_count(), 0)

    var diff_count = 0
    for i in range(len(px_base)):
        if px_base[i] != px_with_line[i]:
            diff_count += 1
    assert_true(diff_count > 10)

    # Frame 3 with no overlay lines pushed must match baseline (zero stale leak)
    app.step_one_frame(Float32(1.0 / 60.0))
    var px_after_clear = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    var stale_diffs = 0
    for i in range(len(px_base)):
        if px_base[i] != px_after_clear[i]:
            stale_diffs += 1
    assert_equal(stale_diffs, 0)
    app.shutdown()

    # C. Studio fallback lights preserved past begin_extraction when 0 PointLightComponents exist
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
    )
    studio.render_editor_frame(160, 120)
    var s_rw = studio.app.resources.get_ptr[RenderWorld]()
    var s_rps = studio.app.resources.get_ptr[RenderPluginStateResource]()
    assert_equal(len(s_rw.value()[unsafe_offset=0].cluster_lights), 2)
    assert_equal(len(s_rps.value()[unsafe_offset=0].extra_lights), 0)

    # Adding an enabled PointLightComponent uses scene lights instead of Studio fallback lights
    var lgt_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](lgt_e, LocalTransform(Vec3(0.0, 3.0, 0.0)))
    _ = studio.app.world.insert[GlobalTransform](lgt_e, GlobalTransform())
    _ = studio.app.world.insert[PointLightComponent](lgt_e, PointLightComponent(Vec3(6.0, 5.0, 4.0), 12.0, True))
    studio.render_editor_frame(160, 120)
    assert_equal(len(s_rw.value()[unsafe_offset=0].cluster_lights), 1)
    assert_true(abs(s_rw.value()[unsafe_offset=0].cluster_lights[0].radiance.x - Float32(6.0)) < Float32(1.0e-5))
    studio.shutdown()


# ---------------------------------------------------------------------------
# 3. Bug #3 (F08): Per-Entity Material Override Isolation & Expanded Fields
# ---------------------------------------------------------------------------

def test_bug3_material_override_isolation_and_expanded_fields() raises:
    var app = EngineApp(AppConfig(title="mat_iso_test", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    app.startup()

    var rend = app.resources.get_ptr_mut[ScalableRenderer]()
    var shared_mat_id = rend.value()[unsafe_offset=0].register_material(
        MaterialAsset(base_color=Vec3(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0)
    )
    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(), Vec3(0.5, 0.5, 0.5))
    var mesh_id = rend.value()[unsafe_offset=0].register_mesh(mb^.finish(), shared_mat_id)

    var cam_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(0.0, 0.0, 5.0)))
    _ = app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = app.world.insert[CameraComponent](cam_e, CameraComponent())

    var ent_a = app.world.spawn()
    _ = app.world.insert[LocalTransform](ent_a, LocalTransform(Vec3(-2.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](ent_a, GlobalTransform())
    _ = app.world.insert[MeshRendererComponent](ent_a, MeshRendererComponent(mesh_id, shared_mat_id))
    _ = app.world.insert[MaterialOverrideComponent](
        ent_a,
        MaterialOverrideComponent(
            -1,
            Vec3(0.9, 0.1, 0.1),
            0.2,
            0.8,
            Vec3(1.5, 0.2, 0.0),
            True,
            opacity=0.75,
            alpha_cutoff=0.25,
            clearcoat=0.9,
            clearcoat_roughness=0.12,
            sheen=0.4,
            subsurface=0.3,
        ),
    )

    var ent_b = app.world.spawn()
    _ = app.world.insert[LocalTransform](ent_b, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](ent_b, GlobalTransform())
    _ = app.world.insert[MeshRendererComponent](ent_b, MeshRendererComponent(mesh_id, shared_mat_id))
    _ = app.world.insert[MaterialOverrideComponent](
        ent_b,
        MaterialOverrideComponent(
            -1,
            Vec3(0.1, 0.2, 0.9),
            0.7,
            0.0,
            Vec3(),
            True,
            opacity=0.5,
            alpha_cutoff=0.0,
            clearcoat=0.0,
            clearcoat_roughness=0.08,
            sheen=0.8,
            subsurface=0.6,
        ),
    )

    var ent_c = app.world.spawn()
    _ = app.world.insert[LocalTransform](ent_c, LocalTransform(Vec3(2.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](ent_c, GlobalTransform())
    _ = app.world.insert[MeshRendererComponent](ent_c, MeshRendererComponent(mesh_id, shared_mat_id))

    app.step_one_frame(Float32(1.0 / 60.0))

    # Shared base slot MUST NOT be mutated!
    assert_true(abs(rend.value()[unsafe_offset=0].materials[shared_mat_id].base_color.x - Float32(0.5)) < Float32(1.0e-5))

    var rw = app.resources.get_ptr[RenderWorld]()
    assert_equal(len(rw.value()[unsafe_offset=0].instances), 3)
    var slot_a = -1
    var slot_b = -1
    var slot_c = -1
    for i in range(len(rw.value()[unsafe_offset=0].instances)):
        ref inst = rw.value()[unsafe_offset=0].instances[i]
        if inst.entity_id == ent_a.to_bits():
            slot_a = inst.material_id
        elif inst.entity_id == ent_b.to_bits():
            slot_b = inst.material_id
        elif inst.entity_id == ent_c.to_bits():
            slot_c = inst.material_id

    assert_true(slot_a >= 0 and slot_b >= 0)
    assert_equal(slot_c, shared_mat_id)
    assert_true(slot_a != shared_mat_id and slot_b != shared_mat_id and slot_a != slot_b)

    ref mat_a = rend.value()[unsafe_offset=0].materials[slot_a]
    assert_true(abs(mat_a.base_color.x - Float32(0.9)) < Float32(1.0e-5))
    assert_true(abs(mat_a.opacity - Float32(0.75)) < Float32(1.0e-5))
    assert_true(abs(mat_a.alpha_cutoff - Float32(0.25)) < Float32(1.0e-5))
    assert_true(abs(mat_a.clearcoat - Float32(0.9)) < Float32(1.0e-5))
    assert_true(abs(mat_a.clearcoat_roughness - Float32(0.12)) < Float32(1.0e-5))
    assert_true(abs(mat_a.sheen_color.x - Float32(0.4)) < Float32(1.0e-5))
    assert_true(abs(mat_a.subsurface_amount - Float32(0.3)) < Float32(1.0e-5))

    ref mat_b = rend.value()[unsafe_offset=0].materials[slot_b]
    assert_true(abs(mat_b.base_color.z - Float32(0.9)) < Float32(1.0e-5))
    assert_true(abs(mat_b.opacity - Float32(0.5)) < Float32(1.0e-5))
    assert_true(abs(mat_b.sheen_color.x - Float32(0.8)) < Float32(1.0e-5))
    assert_true(abs(mat_b.subsurface_amount - Float32(0.6)) < Float32(1.0e-5))

    # Disabling override on A frees its slot and reverts A to shared_mat_id; spawning D recycles freed slot
    var total_mats_before = len(rend.value()[unsafe_offset=0].materials)
    app.world.get_ptr_mut[MaterialOverrideComponent](ent_a).value()[unsafe_offset=0].enabled = False
    app.step_one_frame(Float32(1.0 / 60.0))
    var rstate = app.resources.get_ptr[RenderPluginStateResource]()
    assert_equal(len(rstate.value()[unsafe_offset=0].free_override_slots), 1)
    assert_equal(rstate.value()[unsafe_offset=0].free_override_slots[0], slot_a)

    var ent_d = app.world.spawn()
    _ = app.world.insert[LocalTransform](ent_d, LocalTransform(Vec3(0.0, 2.0, 0.0)))
    _ = app.world.insert[GlobalTransform](ent_d, GlobalTransform())
    _ = app.world.insert[MeshRendererComponent](ent_d, MeshRendererComponent(mesh_id, shared_mat_id))
    _ = app.world.insert[MaterialOverrideComponent](
        ent_d, MaterialOverrideComponent(-1, Vec3(0.2, 0.9, 0.3), 0.3, 0.2, Vec3(), True)
    )
    app.step_one_frame(Float32(1.0 / 60.0))
    assert_equal(len(rend.value()[unsafe_offset=0].materials), total_mats_before)
    assert_equal(len(rstate.value()[unsafe_offset=0].free_override_slots), 0)
    app.shutdown()

    # Binary v2 + v1 fallback + JSON round-trips for MaterialOverrideComponent
    var orig = MaterialOverrideComponent(
        4, Vec3(0.25, 0.5, 0.75), 0.3, 0.6, Vec3(0.1, 0.2, 0.3), True, 0.85, 0.15, 0.7, 0.09, 0.45, 0.35
    )
    var bw = BlobWriter()
    orig.write_bin(bw)
    var br = BlobReader(bw.data.copy())
    var decoded = MaterialOverrideComponent.read_bin(br, MaterialOverrideComponent.SCHEMA_VERSION)
    assert_equal(decoded.material_id, 4)
    assert_true(abs(decoded.opacity - Float32(0.85)) < Float32(1.0e-5))
    assert_true(abs(decoded.alpha_cutoff - Float32(0.15)) < Float32(1.0e-5))
    assert_true(abs(decoded.clearcoat - Float32(0.7)) < Float32(1.0e-5))
    assert_true(abs(decoded.clearcoat_roughness - Float32(0.09)) < Float32(1.0e-5))
    assert_true(abs(decoded.sheen - Float32(0.45)) < Float32(1.0e-5))
    assert_true(abs(decoded.subsurface - Float32(0.35)) < Float32(1.0e-5))

    # v1 binary fallback
    var bw1 = BlobWriter()
    bw1.integer(2)
    bw1.vec3(Vec3(0.4, 0.5, 0.6))
    bw1.f32(0.4)
    bw1.f32(0.1)
    bw1.vec3(Vec3())
    bw1.boolean(True)
    var br1 = BlobReader(bw1.data.copy())
    var dec1 = MaterialOverrideComponent.read_bin(br1, 1)
    assert_equal(dec1.material_id, 2)
    assert_true(abs(dec1.opacity - Float32(1.0)) < Float32(1.0e-5))
    assert_true(abs(dec1.clearcoat - Float32(0.0)) < Float32(1.0e-5))

    var jstr = orig.write_json()
    var jbytes = List[UInt8](capacity=jstr.byte_length())
    for b in jstr.as_bytes():
        jbytes.append(b)
    var jdoc = parse_json(jbytes)
    var dec_j = MaterialOverrideComponent.read_json(jdoc, 0)
    assert_true(abs(dec_j.opacity - Float32(0.85)) < Float32(1.0e-5))
    assert_true(abs(dec_j.sheen - Float32(0.45)) < Float32(1.0e-5))
    assert_true(abs(dec_j.subsurface - Float32(0.35)) < Float32(1.0e-5))

    # StudioGpuMaterials generation tracking & lifecycle invalidation
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var e0 = studio.spawn_entity("MatEnt", position=Vec3(0.0, 1.0, 0.0), record_undo=False)
    studio.render_editor_frame(160, 120)
    assert_equal(studio.gpu_materials.entity_generations[e0], studio.app.world.entities[e0].generation)
    studio.gpu_materials.entity_pbr_mat_ids[e0] = 7
    assert_true(studio.despawn_entity(e0, record_undo=False))
    assert_equal(studio.gpu_materials.entity_pbr_mat_ids[e0], -1)
    studio.shutdown()


# ---------------------------------------------------------------------------
# 4. Bug #4 (F09): `UiContext` Capture Barrier, M0 Widgets & `Esc` Guard
# ---------------------------------------------------------------------------

def test_bug4_ui_context_capture_m0_widgets_and_esc_guard() raises:
    var ui = UiContext()
    var inp = InputState()

    # A. Non-text button auto-focuses (focused_id != 0) yet wants_keyboard() MUST be False
    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)  # away from widgets
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.button("btn_ok", "OK", 100.0, 28.0)
    ui.end_frame()
    assert_true(ui.focused_id != 0)
    assert_false(ui.wants_keyboard())
    assert_false(ui.wants_mouse())

    # B. Clicking text_input focuses text -> wants_keyboard() == True; clicking outside clears it
    var ui2 = UiContext()
    var txt = String("Hero")
    inp.begin_frame()
    inp.move_mouse(40.0, 45.0)  # inside second row (text_input at y=32..60)
    inp.set_mouse(0, True)
    ui2.begin_frame(inp, 1000.0, 700.0)
    _ = ui2.button("btn_ok", "OK", 100.0, 28.0)
    _ = ui2.text_input("name_in", txt, 200.0)
    ui2.end_frame()
    assert_true(ui2.wants_keyboard())
    assert_true(ui2.wants_mouse())
    assert_true(ui2.wants_mouse_at(40.0, 45.0))
    assert_true(ui2.wants_mouse_at(800.0, 500.0))  # active_id != 0 while LMB held inside text_input

    # Release mouse, keep typing focus on steady-state frame
    inp.begin_frame()
    inp.set_mouse(0, False)
    inp.move_mouse(800.0, 500.0)
    ui2.begin_frame(inp, 1000.0, 700.0)
    _ = ui2.button("btn_ok", "OK", 100.0, 28.0)
    _ = ui2.text_input("name_in", txt, 200.0)
    ui2.end_frame()
    assert_true(ui2.wants_keyboard())
    assert_false(ui2.wants_mouse())
    assert_false(ui2.wants_mouse_at(800.0, 500.0))

    # Pressing Esc inside focused text_input (not IME) blurs focus cleanly
    inp.begin_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, True)
    ui2.begin_frame(inp, 1000.0, 700.0)
    _ = ui2.button("btn_ok", "OK", 100.0, 28.0)
    _ = ui2.text_input("name_in", txt, 200.0)
    ui2.end_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, False)
    assert_false(ui2.wants_keyboard())

    # Re-focus text_input then click outside on viewport -> clears focus
    inp.begin_frame()
    inp.move_mouse(40.0, 45.0)
    inp.set_mouse(0, True)
    ui2.begin_frame(inp, 1000.0, 700.0)
    _ = ui2.button("btn_ok", "OK", 100.0, 28.0)
    _ = ui2.text_input("name_in", txt, 200.0)
    ui2.end_frame()
    assert_true(ui2.wants_keyboard())

    inp.begin_frame()
    inp.set_mouse(0, False)
    ui2.begin_frame(inp, 1000.0, 700.0)
    _ = ui2.button("btn_ok", "OK", 100.0, 28.0)
    _ = ui2.text_input("name_in", txt, 200.0)
    ui2.end_frame()

    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    inp.set_mouse(0, True)
    ui2.begin_frame(inp, 1000.0, 700.0)
    _ = ui2.button("btn_ok", "OK", 100.0, 28.0)
    _ = ui2.text_input("name_in", txt, 200.0)
    ui2.end_frame()
    inp.set_mouse(0, False)
    assert_false(ui2.wants_keyboard())

    # C. DockSpace DOCK_TABS captures mouse while DOCK_CENTRAL_VIEWPORT does not
    var dock = DockSpace()
    var vp = dock.add_central_viewport("Viewport")
    _ = dock.split_node(vp, DOCK_SPLIT_H, 0.25, [String("Hierarchy")], place_new_first=True)
    inp.begin_frame()
    inp.move_mouse(60.0, 100.0)  # inside Hierarchy DOCK_TABS
    ui.begin_frame(inp, 1000.0, 700.0)
    ui.update_dock_space(dock, RectF(0.0, 0.0, 1000.0, 700.0))
    ui.end_frame()
    assert_true(ui.wants_mouse())

    inp.begin_frame()
    inp.move_mouse(600.0, 350.0)  # inside DOCK_CENTRAL_VIEWPORT
    ui.begin_frame(inp, 1000.0, 700.0)
    ui.update_dock_space(dock, RectF(0.0, 0.0, 1000.0, 700.0))
    ui.end_frame()
    assert_false(ui.wants_mouse())

    # D. M0 Widgets: drag_float (drag + Shift fine-tune + NaN clamp)
    var dv = Float32(1.0)
    inp.begin_frame()
    inp.move_mouse(50.0, 20.0)
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.drag_float("df_x", "Roughness", dv, speed=0.1, min_val=0.0, max_val=2.0)
    ui.end_frame()

    inp.begin_frame()
    inp.move_mouse(55.0, 20.0)  # delta_x = +5.0 -> +0.5
    ui.begin_frame(inp, 1000.0, 700.0)
    assert_true(ui.drag_float("df_x", "Roughness", dv, speed=0.1, min_val=0.0, max_val=2.0))
    ui.end_frame()
    assert_true(abs(dv - Float32(1.5)) < Float32(1.0e-4))

    # Shift fine-tune (0.2x)
    inp.begin_frame()
    inp.set_logical_key(UI_KEY_LEFT_SHIFT, True)
    inp.move_mouse(60.0, 20.0)  # delta_x = +5.0 * 0.1 * 0.2 = +0.1
    ui.begin_frame(inp, 1000.0, 700.0)
    assert_true(ui.drag_float("df_x", "Roughness", dv, speed=0.1, min_val=0.0, max_val=2.0))
    ui.end_frame()
    inp.set_logical_key(UI_KEY_LEFT_SHIFT, False)
    inp.set_mouse(0, False)
    assert_true(abs(dv - Float32(1.6)) < Float32(1.0e-4))

    var nan_v = Float32(0.0) / Float32(0.0)
    assert_true(isnan(nan_v))
    inp.begin_frame()
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.drag_float("df_nan", "Clamp", nan_v, speed=0.05, min_val=0.25, max_val=1.0)
    ui.end_frame()
    assert_true(abs(nan_v - Float32(0.25)) < Float32(1.0e-4))

    # E. M0 Widgets: combo_box (open, unclipped deferred popup, select item, close)
    var items = List[String]()
    items.append(String("PBR"))
    items.append(String("Cel"))
    items.append(String("VFX"))
    var sel = 0
    inp.begin_frame()
    inp.move_mouse(50.0, 20.0)
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.combo_box("cmb", "Shader", sel, items)
    ui.end_frame()
    assert_true(ui.wants_keyboard())
    assert_true(ui.wants_mouse())

    inp.begin_frame()
    inp.set_mouse(0, False)
    inp.set_logical_key(UI_KEY_DOWN, True)
    ui.begin_frame(inp, 1000.0, 700.0)
    assert_true(ui.combo_box("cmb", "Shader", sel, items))
    ui.end_frame()
    inp.set_logical_key(UI_KEY_DOWN, False)
    assert_equal(sel, 1)

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, True)
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.combo_box("cmb", "Shader", sel, items)
    ui.end_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, False)
    assert_false(ui.wants_keyboard())

    # E2. Regression (challenger_9): Tab-focus combo_box with pointer outside box_r (hovered == False)
    # and press Enter -> popup MUST open and remain open after end_frame(), then Enter on next frame closes it.
    var ui_cb = UiContext()
    var inp_cb = InputState()
    var sel_enter = 0
    inp_cb.begin_frame()
    inp_cb.move_mouse(800.0, 500.0)  # outside box_r
    inp_cb.set_logical_key(UI_KEY_TAB, True)
    ui_cb.begin_frame(inp_cb, 1000.0, 700.0)
    _ = ui_cb.button("btn_prev", "Prev")
    _ = ui_cb.combo_box("cmb_enter", "Shader", sel_enter, items)
    ui_cb.end_frame()
    inp_cb.set_logical_key(UI_KEY_TAB, False)
    var cmb_wid = ui_cb.widget_id("cmb_enter")
    assert_equal(ui_cb.focused_id, cmb_wid)
    assert_equal(ui_cb._open_popup_id, UInt64(0))

    inp_cb.begin_frame()
    inp_cb.move_mouse(800.0, 500.0)
    inp_cb.set_logical_key(UI_KEY_ENTER, True)
    ui_cb.begin_frame(inp_cb, 1000.0, 700.0)
    _ = ui_cb.button("btn_prev", "Prev")
    _ = ui_cb.combo_box("cmb_enter", "Shader", sel_enter, items)
    ui_cb.end_frame()
    inp_cb.set_logical_key(UI_KEY_ENTER, False)
    assert_equal(ui_cb._open_popup_id, cmb_wid)
    assert_equal(ui_cb._popup_kind, 1)
    assert_true(ui_cb.wants_keyboard())
    assert_true(ui_cb.wants_mouse())

    inp_cb.begin_frame()
    inp_cb.set_logical_key(UI_KEY_DOWN, True)
    ui_cb.begin_frame(inp_cb, 1000.0, 700.0)
    _ = ui_cb.button("btn_prev", "Prev")
    assert_true(ui_cb.combo_box("cmb_enter", "Shader", sel_enter, items))
    ui_cb.end_frame()
    inp_cb.set_logical_key(UI_KEY_DOWN, False)
    assert_equal(sel_enter, 1)
    assert_equal(ui_cb._open_popup_id, cmb_wid)

    inp_cb.begin_frame()
    inp_cb.set_logical_key(UI_KEY_ENTER, True)
    ui_cb.begin_frame(inp_cb, 1000.0, 700.0)
    _ = ui_cb.button("btn_prev", "Prev")
    _ = ui_cb.combo_box("cmb_enter", "Shader", sel_enter, items)
    ui_cb.end_frame()
    inp_cb.set_logical_key(UI_KEY_ENTER, False)
    assert_equal(ui_cb._open_popup_id, UInt64(0))
    assert_equal(ui_cb._popup_kind, 0)
    assert_equal(sel_enter, 1)
    assert_false(ui_cb.wants_keyboard())
    assert_false(ui_cb.wants_mouse())

    # F. M0 Widgets: color_edit3 (drag channel + swatch popup)
    var col = Vec3(0.2, 0.4, 0.6)
    inp.begin_frame()
    inp.move_mouse(18.0, 20.0)  # inside swatch
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.color_edit3("col", "Albedo", col)
    ui.end_frame()
    assert_equal(ui._popup_kind, 2)
    inp.begin_frame()
    inp.set_mouse(0, False)
    inp.set_logical_key(UI_KEY_ESCAPE, True)
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.color_edit3("col", "Albedo", col)
    ui.end_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, False)
    assert_equal(ui._popup_kind, 0)

    # G. Studio input capture & Esc quit guard
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var e0 = studio.spawn_entity("GuardTarget", position=Vec3(0.0, 1.0, 0.0), record_undo=False)
    studio.select_entity(e0)

    # Bare Esc MUST NOT quit!
    var s_inp = InputState()
    s_inp.begin_frame()
    s_inp.keys[UI_KEY_ESCAPE] = True
    s_inp.pressed[UI_KEY_ESCAPE] = True
    assert_true(studio.process_editor_input(s_inp, Float32(1.0 / 60.0)))
    s_inp.keys[UI_KEY_ESCAPE] = False

    # When UiContext wants_keyboard() == True, hotkeys (E, Del, F5, Ctrl+Q) MUST be blocked
    studio.ui_ctx._focused_text_id = UInt64(12345)
    s_inp.begin_frame()
    s_inp.keys[69] = True   # E -> Rotate
    s_inp.pressed[69] = True
    s_inp.keys[261] = True  # Del -> delete
    s_inp.pressed[261] = True
    s_inp.keys[294] = True  # F5 -> PIE
    s_inp.pressed[294] = True
    assert_true(studio.process_editor_input(s_inp, Float32(1.0 / 60.0)))
    assert_equal(studio.gizmo_mode, GIZMO_MODE_TRANSLATE)
    assert_true(studio.is_alive_id(e0))
    assert_equal(studio.pie.mode, PIE_STOPPED)
    s_inp.keys[69] = False
    s_inp.keys[261] = False
    s_inp.keys[294] = False

    # Clear text focus -> hotkeys work and Ctrl+Q quits
    studio.ui_ctx._focused_text_id = 0
    s_inp.begin_frame()
    s_inp.keys[UI_KEY_LEFT_CTRL] = True
    s_inp.keys[81] = True  # Q -> Ctrl+Q
    s_inp.pressed[81] = True
    assert_false(studio.process_editor_input(s_inp, Float32(1.0 / 60.0)))
    studio.shutdown()


# ---------------------------------------------------------------------------
# 5. Bug #5a (F10): KCC Triggers (`PhysicsWorld` + `PhysicsPlugin`)
# ---------------------------------------------------------------------------

def test_bug5a_kcc_triggers() raises:
    # A. Standalone PhysicsWorld + CharacterController across OBB, Sphere, Capsule triggers
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_trigger(TriggerVolume.obb(10, Vec3(2.0, 1.0, 0.0), Vec3(0.75, 1.5, 0.75)))
    _ = pw.add_trigger(TriggerVolume.sphere(11, Vec3(5.0, 1.0, 0.0), 0.9))
    _ = pw.add_trigger(TriggerVolume.capsule(12, Vec3(8.0, 1.0, 0.0), 0.7, 1.0))
    # Masked trigger that rejects CAT_DEBRIS (KCC default category)
    _ = pw.add_trigger(
        TriggerVolume.obb(13, Vec3(2.0, 1.0, 0.0), Vec3(0.75, 1.5, 0.75), filter=CollisionFilter(CAT_SENSOR, CAT_PROP_PIECE, 0))
    )

    var cfg = CharacterControllerConfig()
    cfg.gravity = 0.0
    var kcc = CharacterController(Vec3(0.0, 1.0, 0.0), cfg)
    var kcc_id = pw.register_kcc_proxy(kcc.pos, kcc.cfg.radius, kcc.cfg.half_height, kcc.cfg.up, kcc.cfg.filter)
    assert_equal(len(pw.bodies), 0)  # Must NOT pollute self.bodies!
    assert_equal(pw.find_slot_by_id(kcc_id), -1)

    var saw_obb_enter = False
    var saw_obb_stay = False
    var saw_obb_exit = False
    var saw_sph_enter = False
    var saw_cap_enter = False
    var saw_masked = False

    for _ in range(95):
        var p = kcc.move_and_slide(pw, Vec3(6.0, 0.0, 0.0), Float32(1.0 / 60.0))
        assert_true(pw.update_kcc_proxy(kcc_id, p))
        pw.step(Float32(1.0 / 60.0))
        for ti in range(len(pw.trigger_events)):
            var ev = pw.trigger_events[ti]
            assert_equal(ev.body_id, kcc_id)
            if ev.trigger_id == 10:
                if ev.kind == TRIGGER_ENTER:
                    saw_obb_enter = True
                elif ev.kind == TRIGGER_STAY:
                    saw_obb_stay = True
                elif ev.kind == TRIGGER_EXIT:
                    saw_obb_exit = True
            elif ev.trigger_id == 11 and ev.kind == TRIGGER_ENTER:
                saw_sph_enter = True
            elif ev.trigger_id == 12 and ev.kind == TRIGGER_ENTER:
                saw_cap_enter = True
            elif ev.trigger_id == 13:
                saw_masked = True

    assert_true(saw_obb_enter and saw_obb_stay and saw_obb_exit)
    assert_true(saw_sph_enter and saw_cap_enter)
    assert_false(saw_masked)
    assert_true(kcc.pos.x > Float32(8.5))  # Non-blocking traversal!

    # PHYW v3 serialization round-trip preserves kcc_proxies
    var bw = BlobWriter()
    pw.serialize(bw)
    var pw2 = PhysicsWorld()
    var br = BlobReader(bw.data.copy())
    pw2.deserialize(br)
    assert_equal(len(pw2.kcc_proxies), 1)
    assert_equal(pw2.kcc_proxies[0].id, kcc_id)

    # B. ECS EngineApp + PhysicsPlugin + CharacterControllerComponent + ColliderComponent(is_trigger=True)
    var app = EngineApp(AppConfig(title="kcc_trig_ecs", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.startup()

    var trig_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](trig_e, LocalTransform(Vec3(2.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](trig_e, GlobalTransform())
    _ = app.world.insert[ColliderComponent](trig_e, ColliderComponent.box(Vec3(0.8, 1.5, 0.8), is_trigger=True))

    var kcc_cfg = CharacterControllerConfig()
    kcc_cfg.gravity = 0.0
    var kcc_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](kcc_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](kcc_e, GlobalTransform())
    _ = app.world.insert[CharacterControllerComponent](
        kcc_e, CharacterControllerComponent(Vec3(0.0, 1.0, 0.0), kcc_cfg, Vec3(6.0, 0.0, 0.0))
    )

    var tev_cursor = 0
    var ecs_enter = False
    var ecs_stay = False
    var resolved_entity_ok = False
    for _ in range(30):
        app.step_one_frame(Float32(1.0 / 60.0))
        var tevs = app.resources.get_ptr[TriggerEventsResource]().value()[unsafe_offset=0].events.read_since(tev_cursor)
        ref sync = app.resources.get_ptr[PhysicsSyncStateResource]().value()[unsafe_offset=0]
        for ti in range(len(tevs)):
            var ev = tevs[ti]
            if ev.kind == TRIGGER_ENTER:
                ecs_enter = True
                if sync.lookup_entity(ev.body_id) == kcc_e:
                    resolved_entity_ok = True
            elif ev.kind == TRIGGER_STAY:
                ecs_stay = True
        if ecs_enter and ecs_stay:
            break

    assert_true(ecs_enter and ecs_stay and resolved_entity_ok)

    # Despawning kcc_e inside the trigger unregisters the proxy and emits TRIGGER_EXIT
    assert_true(app.world.despawn(kcc_e))
    app.step_one_frame(Float32(1.0 / 60.0))
    var exit_tevs = app.resources.get_ptr[TriggerEventsResource]().value()[unsafe_offset=0].events.read_since(tev_cursor)
    var saw_despawn_exit = False
    for ti in range(len(exit_tevs)):
        if exit_tevs[ti].kind == TRIGGER_EXIT:
            saw_despawn_exit = True
    assert_true(saw_despawn_exit)
    assert_equal(len(app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].kcc_proxies), 0)
    app.shutdown()


# ---------------------------------------------------------------------------
# 6. Bug #5b (F10): Edit-Mode Camera Isolation
# ---------------------------------------------------------------------------

def test_bug5b_edit_camera_isolation() raises:
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
    )
    studio.look_at(Vec3(0.0, 2.0, -5.0), Vec3(0.0, 1.0, 0.0))

    var cam_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(50.0, 20.0, 50.0)))
    _ = studio.app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
    _ = studio.app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=65.0, active=True))

    # Edit mode: editor camera MUST NOT be hijacked by CameraComponent
    studio.render_editor_frame(160, 120)
    var rw = studio.app.resources.get_ptr[RenderWorld]()
    assert_true(abs(rw.value()[unsafe_offset=0].camera.eye.x - Float32(0.0)) < Float32(1.0e-3))
    assert_true(abs(rw.value()[unsafe_offset=0].camera.eye.y - Float32(2.0)) < Float32(1.0e-3))
    assert_true(abs(rw.value()[unsafe_offset=0].camera.eye.z - Float32(-5.0)) < Float32(1.0e-3))

    # Paused mode: still stays on editor camera
    assert_true(studio.pie_play())
    assert_true(studio.pie_pause())
    studio.render_editor_frame(160, 120)
    assert_true(abs(rw.value()[unsafe_offset=0].camera.eye.x - Float32(0.0)) < Float32(1.0e-3))

    # Playing mode: active CameraComponent takes over
    assert_true(studio.pie_play())
    _ = studio.step_pie_frame(Float32(1.0 / 60.0))
    studio.render_editor_frame(160, 120)
    assert_true(abs(rw.value()[unsafe_offset=0].camera.eye.x - Float32(50.0)) < Float32(1.0e-3))
    assert_true(abs(rw.value()[unsafe_offset=0].camera.eye.y - Float32(20.0)) < Float32(1.0e-3))
    assert_true(abs(rw.value()[unsafe_offset=0].camera.eye.z - Float32(50.0)) < Float32(1.0e-3))

    # Stopping PIE restores editor camera
    assert_true(studio.pie_stop())
    studio.render_editor_frame(160, 120)
    assert_true(abs(rw.value()[unsafe_offset=0].camera.eye.x - Float32(0.0)) < Float32(1.0e-3))
    assert_true(abs(rw.value()[unsafe_offset=0].camera.eye.y - Float32(2.0)) < Float32(1.0e-3))
    assert_true(abs(rw.value()[unsafe_offset=0].camera.eye.z - Float32(-5.0)) < Float32(1.0e-3))

    studio.shutdown()


# ---------------------------------------------------------------------------
# 7. Iteration 2 Challenger Regressions (F09-1..4, F10b-1, F07 Tier-0, F08 Studio)
# ---------------------------------------------------------------------------

def test_iteration2_challenger_regressions() raises:
    # 1. F09-1: Tab into text_input updates wants_keyboard() on the exact same frame
    var ui_tab = UiContext()
    var inp = InputState()
    var txt = String("Name")
    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui_tab.begin_frame(inp, 1000.0, 700.0)
    _ = ui_tab.button("btn_first", "First", 100.0, 28.0)
    _ = ui_tab.text_input("txt_second", txt, 200.0)
    ui_tab.end_frame()
    assert_false(ui_tab.wants_keyboard())

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_TAB, True)
    ui_tab.begin_frame(inp, 1000.0, 700.0)
    _ = ui_tab.button("btn_first", "First", 100.0, 28.0)
    _ = ui_tab.text_input("txt_second", txt, 200.0)
    ui_tab.end_frame()
    inp.set_logical_key(UI_KEY_TAB, False)
    assert_equal(ui_tab.focused_id, ui_tab.widget_id("txt_second"))
    assert_true(ui_tab.wants_keyboard())

    # Shift+Tab back to button clears wants_keyboard() on the exact same frame
    inp.begin_frame()
    inp.set_logical_key(UI_KEY_LEFT_SHIFT, True)
    inp.set_logical_key(UI_KEY_TAB, True)
    ui_tab.begin_frame(inp, 1000.0, 700.0)
    _ = ui_tab.button("btn_first", "First", 100.0, 28.0)
    _ = ui_tab.text_input("txt_second", txt, 200.0)
    ui_tab.end_frame()
    inp.set_logical_key(UI_KEY_TAB, False)
    inp.set_logical_key(UI_KEY_LEFT_SHIFT, False)
    assert_equal(ui_tab.focused_id, ui_tab.widget_id("btn_first"))
    assert_false(ui_tab.wants_keyboard())

    # 2. F09-2: text_input as first/only widget never auto-focuses on startup or after Esc
    var ui_first = UiContext()
    var only_txt = String("Solo")
    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui_first.begin_frame(inp, 1000.0, 700.0)
    _ = ui_first.text_input("solo_in", only_txt, 200.0)
    ui_first.end_frame()
    assert_equal(ui_first.focused_id, UInt64(0))
    assert_false(ui_first.wants_keyboard())

    # Click text_input to focus, then Esc to blur, then verify next frame stays blurred
    inp.begin_frame()
    inp.move_mouse(40.0, 14.0)
    inp.set_mouse(0, True)
    ui_first.begin_frame(inp, 1000.0, 700.0)
    _ = ui_first.text_input("solo_in", only_txt, 200.0)
    ui_first.end_frame()
    assert_true(ui_first.wants_keyboard())

    inp.begin_frame()
    inp.set_mouse(0, False)
    inp.set_logical_key(UI_KEY_ESCAPE, True)
    ui_first.begin_frame(inp, 1000.0, 700.0)
    _ = ui_first.text_input("solo_in", only_txt, 200.0)
    ui_first.end_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, False)
    assert_false(ui_first.wants_keyboard())

    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui_first.begin_frame(inp, 1000.0, 700.0)
    _ = ui_first.text_input("solo_in", only_txt, 200.0)
    ui_first.end_frame()
    assert_equal(ui_first.focused_id, UInt64(0))
    assert_false(ui_first.wants_keyboard())

    # 3. F09-3: Open combo_box & color_edit3 popup click never bleeds through to underlying widgets
    var ui_pop = UiContext()
    var pop_items = List[String]()
    pop_items.append(String("Alpha"))
    pop_items.append(String("Beta"))
    pop_items.append(String("Gamma"))
    var pop_sel = 0
    var under_clicked = False
    var under_checked = False

    # Frame 1: open combo_box (box is at y~2..22, drop_r starts around y~26)
    inp.begin_frame()
    inp.move_mouse(40.0, 12.0)
    inp.set_mouse(0, True)
    ui_pop.begin_frame(inp, 1000.0, 700.0)
    _ = ui_pop.combo_box("cmb_top", "Mode", pop_sel, pop_items, 220.0)
    if ui_pop.button("btn_under", "Under", 200.0, 28.0):
        under_clicked = True
    _ = ui_pop.checkbox("chk_under", "CheckUnder", under_checked)
    ui_pop.end_frame()
    assert_true(ui_pop._open_popup_id != 0)
    var item1_x = ui_pop._popup_rect.x + Float32(20.0)
    var item1_y = ui_pop._popup_rect.y + Float32(2.0) + ui_pop.theme.row_height * Float32(1.5)

    # Frame 2: release mouse before clicking popup item
    inp.begin_frame()
    inp.set_mouse(0, False)
    ui_pop.begin_frame(inp, 1000.0, 700.0)
    _ = ui_pop.combo_box("cmb_top", "Mode", pop_sel, pop_items, 220.0)
    if ui_pop.button("btn_under", "Under", 200.0, 28.0):
        under_clicked = True
    _ = ui_pop.checkbox("chk_under", "CheckUnder", under_checked)
    ui_pop.end_frame()

    # Frame 3: press inside open popup item 1 (overlaps btn_under / chk_under!)
    inp.begin_frame()
    inp.move_mouse(item1_x, item1_y)
    inp.set_mouse(0, True)
    ui_pop.begin_frame(inp, 1000.0, 700.0)
    var cmb_changed = ui_pop.combo_box("cmb_top", "Mode", pop_sel, pop_items, 220.0)
    if ui_pop.button("btn_under", "Under", 200.0, 28.0):
        under_clicked = True
    _ = ui_pop.checkbox("chk_under", "CheckUnder", under_checked)
    ui_pop.end_frame()
    assert_true(cmb_changed)
    assert_equal(pop_sel, 1)
    assert_equal(ui_pop._open_popup_id, UInt64(0))
    assert_false(under_checked)
    assert_true(ui_pop.active_id != ui_pop.widget_id("btn_under"))

    # Frame 4: release mouse at same coordinate -> btn_under MUST NOT activate!
    inp.begin_frame()
    inp.set_mouse(0, False)
    ui_pop.begin_frame(inp, 1000.0, 700.0)
    _ = ui_pop.combo_box("cmb_top", "Mode", pop_sel, pop_items, 220.0)
    if ui_pop.button("btn_under", "Under", 200.0, 28.0):
        under_clicked = True
    _ = ui_pop.checkbox("chk_under", "CheckUnder", under_checked)
    ui_pop.end_frame()
    assert_false(under_clicked)

    # Also verify color_edit3 popup click does not bleed through to underlying button
    var ui_col_pop = UiContext()
    var col_val = Vec3(0.2, 0.4, 0.6)
    var col_under_clicked = False
    inp.begin_frame()
    inp.move_mouse(10.0, 12.0)
    inp.set_mouse(0, True)
    ui_col_pop.begin_frame(inp, 1000.0, 700.0)
    _ = ui_col_pop.color_edit3("col_top", "Tint", col_val)
    if ui_col_pop.button("btn_under_col", "UnderCol", 200.0, 28.0):
        col_under_clicked = True
    ui_col_pop.end_frame()
    assert_equal(ui_col_pop._popup_kind, 2)
    var bar_x = ui_col_pop._popup_rect.x + Float32(40.0)
    var bar_y = ui_col_pop._popup_rect.y + Float32(14.0)

    inp.begin_frame()
    inp.set_mouse(0, False)
    ui_col_pop.begin_frame(inp, 1000.0, 700.0)
    _ = ui_col_pop.color_edit3("col_top", "Tint", col_val)
    if ui_col_pop.button("btn_under_col", "UnderCol", 200.0, 28.0):
        col_under_clicked = True
    ui_col_pop.end_frame()

    inp.begin_frame()
    inp.move_mouse(bar_x, bar_y)
    inp.set_mouse(0, True)
    ui_col_pop.begin_frame(inp, 1000.0, 700.0)
    var col_changed = ui_col_pop.color_edit3("col_top", "Tint", col_val)
    if ui_col_pop.button("btn_under_col", "UnderCol", 200.0, 28.0):
        col_under_clicked = True
    ui_col_pop.end_frame()
    assert_true(col_changed)
    assert_true(ui_col_pop.active_id != ui_col_pop.widget_id("btn_under_col"))
    inp.begin_frame()
    inp.set_mouse(0, False)
    ui_col_pop.begin_frame(inp, 1000.0, 700.0)
    _ = ui_col_pop.color_edit3("col_top", "Tint", col_val)
    if ui_col_pop.button("btn_under_col", "UnderCol", 200.0, 28.0):
        col_under_clicked = True
    ui_col_pop.end_frame()
    assert_false(col_under_clicked)

    # 4. F09-4 & 5. F10b-1: RMB fly requires fresh mouse_pressed(1), and pie_step sets editor_camera_override=True
    var studio_ui = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    studio_ui.ui_ctx.hovered_id = UInt64(999)
    studio_ui.ui_ctx.hovered_rect = RectF(0.0, 0.0, 80.0, 80.0)
    var s_inp = InputState()
    s_inp.begin_frame()
    s_inp.has_mouse_position = True
    s_inp.last_mouse_x = 20.0
    s_inp.last_mouse_y = 20.0
    s_inp.set_mouse(1, True)  # RMB pressed while over UI widget
    assert_true(studio_ui.process_editor_input(s_inp, Float32(1.0 / 60.0)))
    assert_false(studio_ui._rmb_fly_active)

    # Drag into viewport while STILL holding RMB (mouse_pressed(1) == False) -> MUST NOT engage fly!
    studio_ui.ui_ctx.hovered_id = 0
    s_inp.begin_frame()
    s_inp.has_mouse_position = True
    s_inp.last_mouse_x = 120.0
    s_inp.last_mouse_y = 60.0
    assert_true(studio_ui.process_editor_input(s_inp, Float32(1.0 / 60.0)))
    assert_false(studio_ui._rmb_fly_active)

    # Release and press RMB freshly in viewport -> engages fly!
    s_inp.begin_frame()
    s_inp.set_mouse(1, False)
    assert_true(studio_ui.process_editor_input(s_inp, Float32(1.0 / 60.0)))
    s_inp.begin_frame()
    s_inp.set_mouse(1, True)
    assert_true(studio_ui.process_editor_input(s_inp, Float32(1.0 / 60.0)))
    assert_true(studio_ui._rmb_fly_active)

    # F10b-1: pie_step() from PIE_STOPPED sets PIE_PAUSED + editor_camera_override=True
    var stepped_n = studio_ui.pie_step(1, Float32(1.0 / 60.0))
    assert_equal(stepped_n, 1)
    assert_equal(studio_ui.pie.mode, PIE_PAUSED)
    var rps_step = studio_ui.app.resources.get_ptr[RenderPluginStateResource]()
    assert_true(Bool(rps_step))
    assert_true(rps_step.value()[unsafe_offset=0].editor_camera_override)
    assert_true(studio_ui.pie_stop())
    studio_ui.shutdown()

    # 6. F07 Tier-0 Studio: DOCK_CENTRAL_VIEWPORT fill_rgba8=0 does not black out viewport / overlay_lines
    var studio_t0 = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=640,
        height=360,
        worker_threads=0,
    )
    studio_t0.look_at(Vec3(0.0, 1.5, 4.0), Vec3(0.0, 0.5, 0.0))
    studio_t0.render_editor_frame(640, 360)
    var px_t0_base = studio_t0.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    var center_px = px_t0_base[(360 // 2) * 640 + (640 // 2)]
    assert_true(center_px != UInt32(0xFF000000))

    var rw_t0 = studio_t0.app.resources.get_ptr_mut[RenderWorld]()
    rw_t0.value()[unsafe_offset=0].overlay_lines.add_line(
        Vec3(-1.2, 0.5, 0.0), Vec3(1.2, 0.5, 0.0), pack_rgba8(255, 240, 20, 255), thickness_px=3.0
    )
    studio_t0.render_editor_frame(640, 360)
    var px_t0_ov = studio_t0.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    var ov_diffs = 0
    for i in range(len(px_t0_base)):
        if px_t0_base[i] != px_t0_ov[i]:
            ov_diffs += 1
    assert_true(ov_diffs > 20)
    studio_t0.shutdown()

    # 7. F08 Studio Vulkan: shared mat0 on plain entity + Studio entity is never clobbered
    var studio_vk = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_1_FORWARD_CLUSTERED,
        width=160,
        height=120,
        worker_threads=0,
    )
    var mat0 = -1
    var vk_rend_opt = studio_vk.app.resources.get_ptr_mut[ScalableRenderer]()
    if vk_rend_opt:
        mat0 = vk_rend_opt.value()[unsafe_offset=0].register_material(
            MaterialAsset(base_color=Vec3(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0)
        )
    var plain_e = studio_vk.app.world.spawn()
    _ = studio_vk.app.world.insert[LocalTransform](plain_e, LocalTransform(Vec3(-1.5, 1.0, 0.0)))
    _ = studio_vk.app.world.insert[GlobalTransform](plain_e, GlobalTransform())
    _ = studio_vk.app.world.insert[PrevGlobalTransform](plain_e, PrevGlobalTransform())
    _ = studio_vk.app.world.insert[MeshRendererComponent](plain_e, MeshRendererComponent(0, mat0))

    var orb_id = studio_vk.spawn_entity(
        "Orb_0",
        position=Vec3(1.5, 1.0, 0.0),
        base_color=Vec3(0.1, 0.7, 0.2),
        record_undo=False,
    )
    var orb_e = studio_vk.resolve_entity(orb_id)
    studio_vk.app.world.get_ptr_mut[MeshRendererComponent](orb_e).value()[unsafe_offset=0].material_id = mat0
    studio_vk.app.world.get_ptr_mut[MaterialOverrideComponent](orb_e).value()[unsafe_offset=0].material_id = mat0

    studio_vk.render_editor_frame(160, 120)
    assert_true(studio_vk.pie_play())
    _ = studio_vk.step_pie_frame(Float32(1.0 / 60.0))
    studio_vk.render_editor_frame(160, 120)
    assert_true(studio_vk.pie_stop())
    studio_vk.render_editor_frame(160, 120)

    var base_mat_r = studio_vk.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].materials[mat0].base_color.x
    assert_true(abs(base_mat_r - Float32(0.5)) < Float32(1.0e-5))
    studio_vk.shutdown()


def _run_slot_recycling_suite(tier: UInt32, cycles: Int) raises:
    var studio = FlintStudioState(
        headless=True,
        render_tier=tier,
        width=160,
        height=120,
        worker_threads=0,
    )
    var rend_ptr = studio.app.resources.get_ptr_mut[ScalableRenderer]()
    assert_true(Bool(rend_ptr))
    if not rend_ptr.value()[unsafe_offset=0].uses_vulkan():
        studio.shutdown()
        return

    var mat0 = rend_ptr.value()[unsafe_offset=0].register_material(
        MaterialAsset(base_color=Vec3(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0)
    )
    var plain_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](plain_e, LocalTransform(Vec3(-2.5, 1.0, 0.0)))
    _ = studio.app.world.insert[GlobalTransform](plain_e, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](plain_e, PrevGlobalTransform())
    _ = studio.app.world.insert[MeshRendererComponent](plain_e, MeshRendererComponent(0, mat0))

    var pbr_ids = List[Int]()
    for k in range(5):
        var oid = studio.spawn_entity(
            "Orb_" + String(k),
            position=Vec3(Float32(k) * Float32(1.2), 1.0, 0.0),
            base_color=Vec3(Float32(0.15) * Float32(k + 1), Float32(0.2), Float32(0.8) - Float32(0.1) * Float32(k)),
            record_undo=False,
        )
        var oe = studio.resolve_entity(oid)
        studio.app.world.get_ptr_mut[MeshRendererComponent](oe).value()[unsafe_offset=0].material_id = mat0
        studio.app.world.get_ptr_mut[MaterialOverrideComponent](oe).value()[unsafe_offset=0].material_id = mat0
        pbr_ids.append(oid)

    studio.render_editor_frame(160, 120)

    # 1. No duplicate 2x allocation in RenderPlugin.override_slots on SHADING_PBR
    var rstate = studio.app.resources.get_ptr[RenderPluginStateResource]()
    assert_true(Bool(rstate))
    assert_equal(len(rstate.value()[unsafe_offset=0].override_slots), 0)
    assert_equal(len(rend_ptr.value()[unsafe_offset=0].materials), 6)  # mat0 + 5 PBR orbs (NOT 11!)
    assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].materials), 6)
    assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].pending_material_copies), 0)

    # 2. Add 1 SHADING_NPR_CEL + 1 SHADING_STYLIZED_VFX entity
    var cel_id = studio.spawn_entity("CelEnt", position=Vec3(-1.0, 1.0, 2.0), base_color=Vec3(0.9, 0.3, 0.2), record_undo=False)
    assert_true(studio.configure_entity_npr_cel(cel_id, cel_outline_width_px=2.0))
    var vfx_id = studio.spawn_entity("VfxEnt", position=Vec3(1.0, 1.0, 2.0), base_color=Vec3(0.2, 0.8, 0.9), record_undo=False)
    assert_true(studio.configure_entity_stylized_vfx(vfx_id, vfx_erosion=0.4))

    studio.render_editor_frame(160, 120)
    var steady_rend_mats = len(rend_ptr.value()[unsafe_offset=0].materials)
    var steady_gpu_mats = len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].materials)
    assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].cel_materials), 1)
    assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].vfx_materials), 1)
    assert_equal(len(rstate.value()[unsafe_offset=0].override_slots), 0)
    assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].pending_material_copies), 0)

    # 3. Repeated PIE play/step/stop/render cycles must have ZERO slot leak across PBR, Cel, and VFX
    for _ in range(cycles):
        assert_true(studio.pie_play())
        _ = studio.step_pie_frame(Float32(1.0 / 60.0))
        studio.render_editor_frame(160, 120)
        assert_true(studio.pie_stop())
        studio.render_editor_frame(160, 120)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].materials), steady_rend_mats)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].materials), steady_gpu_mats)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].cel_materials), 1)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].vfx_materials), 1)
        assert_equal(len(rstate.value()[unsafe_offset=0].override_slots), 0)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].pending_material_copies), 0)
        assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[mat0].base_color.x - Float32(0.5)) < Float32(1.0e-5))
        assert_true(abs(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].materials[mat0].base_color.x - Float32(0.5)) < Float32(1.0e-5))

    # 3b. Repeated Cel outline_width_px toggles (2.0 -> 0.0 -> 2.0) must reuse outline slot with ZERO growth
    for _ in range(cycles):
        assert_true(studio.configure_entity_npr_cel(cel_id, cel_outline_width_px=0.0))
        studio.render_editor_frame(160, 120)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].materials), steady_gpu_mats)
        assert_true(studio.configure_entity_npr_cel(cel_id, cel_outline_width_px=2.0))
        studio.render_editor_frame(160, 120)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].materials), steady_gpu_mats)

    # 4. Repeated despawn -> respawn cycles must recycle PBR/Cel/VFX slots with ZERO growth
    for cyc in range(cycles):
        for k in range(len(pbr_ids)):
            assert_true(studio.despawn_entity(pbr_ids[k], record_undo=False))
        assert_true(studio.despawn_entity(cel_id, record_undo=False))
        assert_true(studio.despawn_entity(vfx_id, record_undo=False))
        pbr_ids.clear()
        for k in range(5):
            var nid = studio.spawn_entity(
                "Orb_R" + String(cyc) + "_" + String(k),
                position=Vec3(Float32(k) * Float32(1.2), 1.0, 0.0),
                base_color=Vec3(Float32(0.1) * Float32(k + 2), Float32(0.45), Float32(0.25)),
                record_undo=False,
            )
            var ne = studio.resolve_entity(nid)
            studio.app.world.get_ptr_mut[MeshRendererComponent](ne).value()[unsafe_offset=0].material_id = mat0
            studio.app.world.get_ptr_mut[MaterialOverrideComponent](ne).value()[unsafe_offset=0].material_id = mat0
            pbr_ids.append(nid)
        cel_id = studio.spawn_entity("CelEnt_R" + String(cyc), position=Vec3(-1.0, 1.0, 2.0), base_color=Vec3(0.4, 0.9, 0.3), record_undo=False)
        assert_true(studio.configure_entity_npr_cel(cel_id, cel_outline_width_px=2.5))
        vfx_id = studio.spawn_entity("VfxEnt_R" + String(cyc), position=Vec3(1.0, 1.0, 2.0), base_color=Vec3(0.8, 0.2, 0.9), record_undo=False)
        assert_true(studio.configure_entity_stylized_vfx(vfx_id, vfx_erosion=0.55))

        studio.render_editor_frame(160, 120)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].materials), steady_rend_mats)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].materials), steady_gpu_mats)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].cel_materials), 1)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].vfx_materials), 1)
        assert_equal(len(rstate.value()[unsafe_offset=0].override_slots), 0)
        assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[mat0].base_color.x - Float32(0.5)) < Float32(1.0e-5))

    # 5. Repeated load_scene_file cycles must recycle PBR/Cel/VFX slots with ZERO growth
    var scene_path = "/tmp/flint_test_f08_slot_recycle.flscn"
    assert_true(studio.save_scene_file(scene_path))
    for _ in range(cycles):
        assert_true(studio.load_scene_file(scene_path))
        studio.render_editor_frame(160, 120)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].materials), steady_rend_mats)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].materials), steady_gpu_mats)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].cel_materials), 1)
        assert_equal(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].vfx_materials), 1)
        assert_equal(len(rstate.value()[unsafe_offset=0].override_slots), 0)
        assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[mat0].base_color.x - Float32(0.5)) < Float32(1.0e-5))
    _ = remove_file(scene_path)

    # 6. Shading mode switching on a live entity recycles slots without unbounded growth
    var switch_id = pbr_ids[0]
    for _ in range(cycles):
        assert_true(studio.configure_entity_npr_cel(switch_id, cel_outline_width_px=2.0))
        studio.render_editor_frame(160, 120)
        assert_true(studio.configure_entity_stylized_vfx(switch_id, vfx_erosion=0.35))
        studio.render_editor_frame(160, 120)
        var sw_v = studio.entities[switch_id].copy()
        sw_v.shading_mode = SHADING_PBR
        assert_true(studio.commit_node_edit(switch_id, sw_v^, "Back to PBR"))
        studio.render_editor_frame(160, 120)
    assert_true(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].cel_materials) <= 2)
    assert_true(len(rend_ptr.value()[unsafe_offset=0].gpu_slot[0].vfx_materials) <= 2)
    assert_equal(len(rend_ptr.value()[unsafe_offset=0].materials), steady_rend_mats)
    studio.shutdown()


def test_bug3_studio_gpu_materials_slot_recycling() raises:
    _run_slot_recycling_suite(RENDER_TIER_1_FORWARD_CLUSTERED, 6)


def test_iteration4_tier2_slot_recycling() raises:
    _run_slot_recycling_suite(RENDER_TIER_2_AAA_VISBUFFER_RT, 2)


def test_iteration4_vulkan_pbr_serialization_and_mo_enabled_parity() raises:
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_1_FORWARD_CLUSTERED,
        width=160,
        height=120,
        worker_threads=0,
    )
    var rend_ptr = studio.app.resources.get_ptr_mut[ScalableRenderer]()
    assert_true(Bool(rend_ptr))
    if not rend_ptr.value()[unsafe_offset=0].uses_vulkan():
        studio.shutdown()
        return

    var mat0 = rend_ptr.value()[unsafe_offset=0].register_material(
        MaterialAsset(base_color=Vec3(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0)
    )
    var base_tex_mat = MaterialAsset(base_color=Vec3(0.2, 0.2, 0.2), roughness=0.6, metallic=0.1)
    base_tex_mat.set_albedo_slot(7)
    base_tex_mat.set_normal_slot(11)
    base_tex_mat.set_orm_slot(13)
    base_tex_mat.specular_factor = Float32(0.42)
    base_tex_mat.iridescence = Float32(0.65)
    base_tex_mat.flags = MAT_FLAG_RECEIVE_DECALS | MAT_FLAG_DOUBLE_SIDED
    var tex_mat_id = rend_ptr.value()[unsafe_offset=0].register_material(base_tex_mat)

    var red_id = studio.spawn_entity(
        "RedOrb",
        position=Vec3(0.0, 1.0, 0.0),
        base_color=Vec3(0.95, 0.05, 0.05),
        roughness=0.22,
        metallic=0.75,
        record_undo=False,
    )
    var red_e = studio.resolve_entity(red_id)
    studio.app.world.get_ptr_mut[MeshRendererComponent](red_e).value()[unsafe_offset=0].material_id = mat0
    studio.app.world.get_ptr_mut[MaterialOverrideComponent](red_e).value()[unsafe_offset=0].material_id = mat0
    studio.app.world.get_ptr_mut[MaterialOverrideComponent](red_e).value()[unsafe_offset=0].clearcoat = Float32(0.70)
    assert_true(studio.app.world.get_ptr[MaterialOverrideComponent](red_e).value()[unsafe_offset=0].enabled)

    # Also test base material texture/flag cloning via `build_overridden_material`
    var tex_ent_id = studio.spawn_entity(
        "TexturedClone",
        position=Vec3(2.0, 1.0, 0.0),
        base_color=Vec3(0.1, 0.85, 0.25),
        roughness=0.33,
        metallic=0.25,
        record_undo=False,
    )
    var tex_e = studio.resolve_entity(tex_ent_id)
    studio.app.world.get_ptr_mut[MeshRendererComponent](tex_e).value()[unsafe_offset=0].material_id = tex_mat_id
    studio.app.world.get_ptr_mut[MaterialOverrideComponent](tex_e).value()[unsafe_offset=0].material_id = tex_mat_id

    var hash_before = studio.state_hash()
    studio.render_editor_frame(160, 120)

    # 2a: ECS components & state_hash() remain untouched across Vulkan render_editor_frame
    assert_true(studio.app.world.get_ptr[MaterialOverrideComponent](red_e).value()[unsafe_offset=0].enabled)
    assert_equal(studio.app.world.get_ptr[MaterialOverrideComponent](red_e).value()[unsafe_offset=0].material_id, mat0)
    assert_equal(studio.app.world.get_ptr[MeshRendererComponent](red_e).value()[unsafe_offset=0].material_id, mat0)
    assert_equal(studio.state_hash(), hash_before)

    var rstate = studio.app.resources.get_ptr[RenderPluginStateResource]()
    assert_true(Bool(rstate))
    assert_equal(len(rstate.value()[unsafe_offset=0].override_slots), 0)
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[mat0].base_color.x - Float32(0.5)) < Float32(1.0e-5))

    var rw_vk = studio.app.resources.get_ptr[RenderWorld]()
    assert_true(Bool(rw_vk))
    var red_ext_id = -1
    var tex_ext_id = -1
    for ii in range(len(rw_vk.value()[unsafe_offset=0].instances)):
        ref inst = rw_vk.value()[unsafe_offset=0].instances[ii]
        if inst.entity_id == red_e.to_bits():
            red_ext_id = inst.material_id
        elif inst.entity_id == tex_e.to_bits():
            tex_ext_id = inst.material_id
    assert_true(red_ext_id >= 0 and red_ext_id != mat0)
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[red_ext_id].base_color.x - Float32(0.95)) < Float32(1.0e-4))
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[red_ext_id].base_color.y - Float32(0.05)) < Float32(1.0e-4))
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[red_ext_id].base_color.z - Float32(0.05)) < Float32(1.0e-4))
    assert_true(tex_ext_id >= 0 and tex_ext_id != tex_mat_id)
    assert_equal(Int(rend_ptr.value()[unsafe_offset=0].materials[tex_ext_id].albedo_slot), 7)
    assert_equal(Int(rend_ptr.value()[unsafe_offset=0].materials[tex_ext_id].normal_slot), 11)
    assert_equal(Int(rend_ptr.value()[unsafe_offset=0].materials[tex_ext_id].orm_slot), 13)
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[tex_ext_id].specular_factor - Float32(0.42)) < Float32(1.0e-4))
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[tex_ext_id].iridescence - Float32(0.65)) < Float32(1.0e-4))
    assert_true((rend_ptr.value()[unsafe_offset=0].materials[tex_ext_id].flags & MAT_FLAG_DOUBLE_SIDED) != 0)

    # 2b: Save .flscn after Vulkan render -> reload in Tier-0 Studio & standalone EngineApp
    var scene_path = "/tmp/flint_test_iter4_pbr_parity.flscn"
    assert_true(studio.save_scene_file(scene_path))

    var st_t0 = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
    )
    var t0_mat0 = st_t0.app.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0].register_material(
        MaterialAsset(base_color=Vec3(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0)
    )
    assert_true(st_t0.load_scene_file(scene_path))
    st_t0.render_editor_frame(160, 120)
    var rstate_t0 = st_t0.app.resources.get_ptr[RenderPluginStateResource]()
    assert_true(len(rstate_t0.value()[unsafe_offset=0].override_slots) >= 1)
    var t0_red_e = st_t0.resolve_entity(red_id)
    var rw_t0 = st_t0.app.resources.get_ptr[RenderWorld]()
    var rend_t0 = st_t0.app.resources.get_ptr[ScalableRenderer]()
    var t0_ext_id = -1
    for ii in range(len(rw_t0.value()[unsafe_offset=0].instances)):
        if rw_t0.value()[unsafe_offset=0].instances[ii].entity_id == t0_red_e.to_bits():
            t0_ext_id = rw_t0.value()[unsafe_offset=0].instances[ii].material_id
            break
    assert_true(t0_ext_id >= 0 and t0_ext_id != t0_mat0)
    assert_true(rend_t0.value()[unsafe_offset=0].materials[t0_ext_id].base_color.x > Float32(0.90))
    assert_true(rend_t0.value()[unsafe_offset=0].materials[t0_ext_id].base_color.y < Float32(0.10))
    assert_true(abs(rend_t0.value()[unsafe_offset=0].materials[t0_ext_id].clearcoat - Float32(0.70)) < Float32(1.0e-4))
    assert_true(abs(rend_t0.value()[unsafe_offset=0].materials[t0_mat0].base_color.x - Float32(0.5)) < Float32(1.0e-5))
    st_t0.shutdown()

    var sa = EngineApp(AppConfig(title="iter4_sa", headless=True, worker_threads=0))
    sa.add_plugin(TransformPlugin())
    sa.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    sa.startup()
    var game_mat0 = sa.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0].register_material(
        MaterialAsset(base_color=Vec3(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0)
    )
    var reg = ComponentRegistry.with_builtins()
    register_physics_codecs(reg)
    register_render_codecs(reg)
    reg.register[StudioEntityMetaComponent]()
    reg.register[StudioStyleComponent]()
    reg.ensure_registered_in_world(sa.world)
    var scn = Scene("Parity")
    _ = scn.load_flscn_file(scene_path, sa.world, reg, clear_existing=True)
    var sa_cam = sa.world.spawn()
    _ = sa.world.insert[LocalTransform](sa_cam, LocalTransform(Vec3(0.0, 1.0, 5.0)))
    _ = sa.world.insert[GlobalTransform](sa_cam, GlobalTransform())
    _ = sa.world.insert[PrevGlobalTransform](sa_cam, PrevGlobalTransform())
    _ = sa.world.insert[CameraComponent](sa_cam, CameraComponent(fov_y_deg=55.0, active=True))
    sa.step_one_frame(Float32(1.0 / 60.0))

    var rstate_sa = sa.resources.get_ptr[RenderPluginStateResource]()
    assert_true(len(rstate_sa.value()[unsafe_offset=0].override_slots) >= 1)
    var rw_sa = sa.resources.get_ptr[RenderWorld]()
    var rend_sa = sa.resources.get_ptr[ScalableRenderer]()
    assert_true(len(rw_sa.value()[unsafe_offset=0].instances) >= 1)
    var sa_mat_id = -1
    for ii in range(len(rw_sa.value()[unsafe_offset=0].instances)):
        if rw_sa.value()[unsafe_offset=0].instances[ii].entity_id == red_e.to_bits():
            sa_mat_id = rw_sa.value()[unsafe_offset=0].instances[ii].material_id
            break
    if sa_mat_id < 0:
        sa_mat_id = rw_sa.value()[unsafe_offset=0].instances[0].material_id
    assert_true(sa_mat_id != game_mat0 and sa_mat_id >= 0 and sa_mat_id < len(rend_sa.value()[unsafe_offset=0].materials))
    assert_true(abs(rend_sa.value()[unsafe_offset=0].materials[sa_mat_id].base_color.x - Float32(0.95)) < Float32(1.0e-4))
    assert_true(abs(rend_sa.value()[unsafe_offset=0].materials[sa_mat_id].base_color.y - Float32(0.05)) < Float32(1.0e-4))
    assert_true(abs(rend_sa.value()[unsafe_offset=0].materials[sa_mat_id].base_color.z - Float32(0.05)) < Float32(1.0e-4))
    assert_true(abs(rend_sa.value()[unsafe_offset=0].materials[sa_mat_id].clearcoat - Float32(0.70)) < Float32(1.0e-4))
    assert_true(abs(rend_sa.value()[unsafe_offset=0].materials[game_mat0].base_color.x - Float32(0.5)) < Float32(1.0e-5))
    sa.shutdown()
    _ = remove_file(scene_path)

    # 2c: Intentional mo.enabled = False on SHADING_PBR is honored in Vulkan Studio and preserved across commit_node_edit
    studio.app.world.get_ptr_mut[MaterialOverrideComponent](red_e).value()[unsafe_offset=0].enabled = False
    studio.app.world.get_ptr_mut[MaterialOverrideComponent](red_e).value()[unsafe_offset=0].clearcoat = Float32(0.0)
    studio._sync_compat_view()
    var edit_v = studio.entities[red_id].copy()
    edit_v.translation = Vec3(0.5, 1.0, 0.0)
    assert_true(studio.commit_node_edit(red_id, edit_v^, "Move disabled override"))
    assert_false(studio.app.world.get_ptr[MaterialOverrideComponent](red_e).value()[unsafe_offset=0].enabled)

    var mats_before_toggle = len(rend_ptr.value()[unsafe_offset=0].materials)
    studio.render_editor_frame(160, 120)
    assert_false(studio.app.world.get_ptr[MaterialOverrideComponent](red_e).value()[unsafe_offset=0].enabled)
    assert_equal(studio.app.world.get_ptr[MeshRendererComponent](red_e).value()[unsafe_offset=0].material_id, mat0)
    assert_equal(len(rstate.value()[unsafe_offset=0].override_slots), 0)
    assert_equal(studio.gpu_materials.entity_pbr_mat_ids[red_id], -1)

    var off_ext_id = -1
    for ii in range(len(rw_vk.value()[unsafe_offset=0].instances)):
        if rw_vk.value()[unsafe_offset=0].instances[ii].entity_id == red_e.to_bits():
            off_ext_id = rw_vk.value()[unsafe_offset=0].instances[ii].material_id
            break
    assert_equal(off_ext_id, mat0)
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[off_ext_id].base_color.x - Float32(0.5)) < Float32(1.0e-5))

    # Re-enable mo.enabled = True -> recycles freed PBR slot without material leak
    studio.app.world.get_ptr_mut[MaterialOverrideComponent](red_e).value()[unsafe_offset=0].enabled = True
    studio.render_editor_frame(160, 120)
    var on_ext_id = -1
    for ii in range(len(rw_vk.value()[unsafe_offset=0].instances)):
        if rw_vk.value()[unsafe_offset=0].instances[ii].entity_id == red_e.to_bits():
            on_ext_id = rw_vk.value()[unsafe_offset=0].instances[ii].material_id
            break
    assert_true(on_ext_id >= 0 and on_ext_id != mat0)
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[on_ext_id].base_color.x - Float32(0.95)) < Float32(1.0e-4))
    assert_equal(len(rend_ptr.value()[unsafe_offset=0].materials), mats_before_toggle)
    studio.shutdown()


def test_iteration4_cel_outline_and_smear_pie_regressions() raises:
    # 1. Tier-0 non-destructive smear_stretch check & PIE restore
    var st_t0 = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
    )
    var t0_id = st_t0.spawn_entity("SmearT0", position=Vec3(0.0, 1.0, 0.0), record_undo=False)
    var t0_e = st_t0.resolve_entity(t0_id)
    var t0_v = st_t0.entities[t0_id].copy()
    t0_v.smear_stretch = Float32(1.5)
    assert_true(st_t0.commit_node_edit(t0_id, t0_v^, "Set smear_stretch T0"))
    var exp_sm = compute_volume_preserving_smear_scale(Float32(1.5))

    for _ in range(4):
        st_t0.render_editor_frame(160, 120)
        ref gt_t0 = st_t0.app.world.get_ptr[GlobalTransform](t0_e).value()[unsafe_offset=0]
        assert_true(abs(gt_t0.matrix.at(0, 0) - Float32(1.0)) < Float32(1.0e-5))
        assert_true(abs(gt_t0.matrix.at(1, 1) - Float32(1.0)) < Float32(1.0e-5))
        assert_true(abs(gt_t0.matrix.at(2, 2) - Float32(1.0)) < Float32(1.0e-5))
        var rw_t0 = st_t0.app.resources.get_ptr[RenderWorld]()
        var found_t0 = False
        for ii in range(len(rw_t0.value()[unsafe_offset=0].instances)):
            ref inst = rw_t0.value()[unsafe_offset=0].instances[ii]
            if inst.entity_id == t0_e.to_bits():
                assert_true(abs(inst.world_mat.at(1, 1) - exp_sm.y) < Float32(1.0e-4))
                found_t0 = True
                break
        assert_true(found_t0)

    var t0_h = st_t0.state_hash()
    assert_true(st_t0.pie_play())
    _ = st_t0.step_pie_frame(Float32(1.0 / 60.0))
    st_t0.render_editor_frame(160, 120)
    assert_true(st_t0.pie_stop())
    assert_true(st_t0.pie.last_restore_verified)
    assert_equal(st_t0.state_hash(), t0_h)
    st_t0.shutdown()

    # 2. Vulkan Cel outline_width_px 2.0 -> 0.0 -> 2.0 slot constancy + F06 smear_stretch PIE restore
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_1_FORWARD_CLUSTERED,
        width=160,
        height=120,
        worker_threads=0,
    )
    var rend_ptr = studio.app.resources.get_ptr_mut[ScalableRenderer]()
    assert_true(Bool(rend_ptr))
    if not rend_ptr.value()[unsafe_offset=0].uses_vulkan():
        studio.shutdown()
        return

    var cel_id = studio.spawn_entity(
        "CelToggle", position=Vec3(0.0, 1.0, 0.0), base_color=Vec3(0.9, 0.3, 0.2), record_undo=False
    )
    assert_true(studio.configure_entity_npr_cel(cel_id, cel_outline_width_px=2.0))
    studio.render_editor_frame(160, 120)

    ref gpu = rend_ptr.value()[unsafe_offset=0].gpu_slot[0]
    var cel_slot = studio.gpu_materials.entity_cel_mat_ids[cel_id]
    assert_true(cel_slot >= 0 and gpu.is_cel_outlined_material(cel_slot))
    var base_gpu_mats = len(gpu.materials)
    var base_cel_mats = len(gpu.cel_materials)

    for _ in range(6):
        assert_true(studio.configure_entity_npr_cel(cel_id, cel_outline_width_px=0.0))
        studio.render_editor_frame(160, 120)
        assert_false(gpu.is_cel_outlined_material(cel_slot))
        assert_equal(len(gpu.materials), base_gpu_mats)
        assert_true(studio.configure_entity_npr_cel(cel_id, cel_outline_width_px=2.0))
        studio.render_editor_frame(160, 120)
        assert_true(gpu.is_cel_outlined_material(cel_slot))
        assert_equal(len(gpu.materials), base_gpu_mats)
        assert_equal(len(gpu.cel_materials), base_cel_mats)

    # F06 smear_stretch = 1.5 across Cel and PBR entities + PIE cycles
    assert_true(studio.configure_entity_npr_cel(cel_id, cel_outline_width_px=2.0, smear_stretch=1.5))
    var pbr_smear_id = studio.spawn_entity(
        "SmearPbr", position=Vec3(2.0, 1.0, 0.0), base_color=Vec3(0.3, 0.7, 0.9), record_undo=False
    )
    var pbr_v = studio.entities[pbr_smear_id].copy()
    pbr_v.smear_stretch = Float32(1.5)
    assert_true(studio.commit_node_edit(pbr_smear_id, pbr_v^, "Set PBR smear"))

    var cel_e = studio.resolve_entity(cel_id)
    var pbr_e = studio.resolve_entity(pbr_smear_id)
    for _ in range(3):
        studio.render_editor_frame(160, 120)
        ref gt_cel = studio.app.world.get_ptr[GlobalTransform](cel_e).value()[unsafe_offset=0]
        ref gt_pbr = studio.app.world.get_ptr[GlobalTransform](pbr_e).value()[unsafe_offset=0]
        assert_true(abs(gt_cel.matrix.at(1, 1) - Float32(1.0)) < Float32(1.0e-5))
        assert_true(abs(gt_pbr.matrix.at(1, 1) - Float32(1.0)) < Float32(1.0e-5))
        var rw_vk = studio.app.resources.get_ptr[RenderWorld]()
        var found_pbr = False
        for ii in range(len(rw_vk.value()[unsafe_offset=0].instances)):
            ref inst = rw_vk.value()[unsafe_offset=0].instances[ii]
            if inst.entity_id == pbr_e.to_bits():
                assert_true(abs(inst.world_mat.at(1, 1) - exp_sm.y) < Float32(1.0e-4))
                found_pbr = True
                break
        assert_true(found_pbr)

    for _ in range(2):
        studio.render_editor_frame(160, 120)
        var h_pre = studio.state_hash()
        var sum_pre = studio.app.world.state_checksum()
        assert_true(studio.pie_play())
        _ = studio.step_pie_frame(Float32(1.0 / 60.0))
        studio.render_editor_frame(160, 120)
        assert_true(studio.pie_stop())
        assert_true(studio.pie.last_restore_verified)
        assert_equal(studio.state_hash(), h_pre)
        assert_equal(studio.app.world.state_checksum(), sum_pre)
        assert_true(
            abs(studio.app.world.get_ptr[StudioStyleComponent](cel_e).value()[unsafe_offset=0].smear_stretch - Float32(1.5))
            < Float32(1.0e-5)
        )
        assert_true(
            abs(studio.app.world.get_ptr[StudioStyleComponent](pbr_e).value()[unsafe_offset=0].smear_stretch - Float32(1.5))
            < Float32(1.0e-5)
        )
    studio.shutdown()


def test_iteration4_default_studio_foliage_materials_non_aliasing() raises:
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_1_FORWARD_CLUSTERED,
        width=160,
        height=120,
        worker_threads=0,
    )
    var rend_ptr = studio.app.resources.get_ptr_mut[ScalableRenderer]()
    assert_true(Bool(rend_ptr))
    if not rend_ptr.value()[unsafe_offset=0].uses_vulkan():
        studio.shutdown()
        return

    populate_default_studio_scene(studio)
    upload_default_studio_gpu_assets(studio)

    var trunk_mat = studio.gpu_materials.mat_trunk_id
    var foliage_mat = studio.gpu_materials.mat_foliage_id
    assert_true(trunk_mat >= 0 and foliage_mat >= 0 and trunk_mat != foliage_mat)

    for k in range(len(studio.gpu_materials.entity_pbr_mat_ids)):
        assert_true(studio.gpu_materials.entity_pbr_mat_ids[k] != trunk_mat)
        assert_true(studio.gpu_materials.entity_pbr_mat_ids[k] != foliage_mat)

    assert_equal(rend_ptr.value()[unsafe_offset=0].cpu_mesh_materials[studio.gpu_materials.foliage_trunk_id], trunk_mat)
    assert_equal(rend_ptr.value()[unsafe_offset=0].cpu_mesh_materials[studio.gpu_materials.foliage_canopy_id], foliage_mat)

    assert_true(studio.despawn_entity(4, record_undo=False))
    assert_true(studio.despawn_entity(5, record_undo=False))
    for k in range(len(studio.gpu_materials.free_pbr_mat_ids)):
        assert_true(studio.gpu_materials.free_pbr_mat_ids[k] != trunk_mat)
        assert_true(studio.gpu_materials.free_pbr_mat_ids[k] != foliage_mat)

    _ = studio.spawn_entity("BrightA", position=Vec3(1.0, 1.0, 0.0), base_color=Vec3(0.95, 0.05, 0.95), record_undo=False)
    _ = studio.spawn_entity("BrightB", position=Vec3(2.0, 1.0, 0.0), base_color=Vec3(0.95, 0.05, 0.95), record_undo=False)
    studio.render_editor_frame(160, 120)

    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[trunk_mat].base_color.x - Float32(0.34)) < Float32(1.0e-4))
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[trunk_mat].base_color.y - Float32(0.23)) < Float32(1.0e-4))
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[trunk_mat].base_color.z - Float32(0.14)) < Float32(1.0e-4))
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[foliage_mat].base_color.x - Float32(0.18)) < Float32(1.0e-4))
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[foliage_mat].base_color.y - Float32(0.46)) < Float32(1.0e-4))
    assert_true(abs(rend_ptr.value()[unsafe_offset=0].materials[foliage_mat].base_color.z - Float32(0.22)) < Float32(1.0e-4))
    studio.shutdown()


def _verify_saved_cel_vfx_scene_parity(
    scene_path: String,
    hero_id: Int,
    hero_bits: UInt64,
    exp_r: Float32,
    exp_g: Float32,
    exp_b: Float32,
) raises:
    # 1. Reload into Tier-0 Studio
    var st_t0 = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
    )
    var t0_mat0 = st_t0.app.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0].register_material(
        MaterialAsset(base_color=Vec3(0.1, 0.2, 0.9), roughness=0.55, metallic=0.0)
    )
    assert_true(st_t0.load_scene_file(scene_path))
    var t0_hero_e = st_t0.resolve_entity(hero_id)
    assert_true(st_t0.app.world.get_ptr[MaterialOverrideComponent](t0_hero_e).value()[unsafe_offset=0].enabled)
    st_t0.render_editor_frame(160, 120)
    var rw_t0 = st_t0.app.resources.get_ptr[RenderWorld]()
    var rend_t0 = st_t0.app.resources.get_ptr[ScalableRenderer]()
    var t0_ext_id = -1
    for ii in range(len(rw_t0.value()[unsafe_offset=0].instances)):
        if rw_t0.value()[unsafe_offset=0].instances[ii].entity_id == t0_hero_e.to_bits():
            t0_ext_id = rw_t0.value()[unsafe_offset=0].instances[ii].material_id
            break
    assert_true(t0_ext_id >= 0 and t0_ext_id != t0_mat0)
    assert_true(abs(rend_t0.value()[unsafe_offset=0].materials[t0_ext_id].base_color.x - exp_r) < Float32(1.0e-4))
    assert_true(abs(rend_t0.value()[unsafe_offset=0].materials[t0_ext_id].base_color.y - exp_g) < Float32(1.0e-4))
    assert_true(abs(rend_t0.value()[unsafe_offset=0].materials[t0_ext_id].base_color.z - exp_b) < Float32(1.0e-4))
    assert_true(abs(rend_t0.value()[unsafe_offset=0].materials[t0_mat0].base_color.z - Float32(0.9)) < Float32(1.0e-4))
    st_t0.shutdown()

    # 2. Reload into standalone EngineApp + RenderPlugin
    var sa = EngineApp(AppConfig(title="iter5_sa", headless=True, worker_threads=0))
    sa.add_plugin(TransformPlugin())
    sa.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    sa.startup()
    var game_mat0 = sa.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0].register_material(
        MaterialAsset(base_color=Vec3(0.1, 0.2, 0.9), roughness=0.55, metallic=0.0)
    )
    var reg = ComponentRegistry.with_builtins()
    register_physics_codecs(reg)
    register_render_codecs(reg)
    reg.register[StudioEntityMetaComponent]()
    reg.register[StudioStyleComponent]()
    reg.ensure_registered_in_world(sa.world)
    var scn = Scene("CelVfxParity")
    _ = scn.load_flscn_file(scene_path, sa.world, reg, clear_existing=True)
    var sa_cam = sa.world.spawn()
    _ = sa.world.insert[LocalTransform](sa_cam, LocalTransform(Vec3(0.0, 1.0, 5.0)))
    _ = sa.world.insert[GlobalTransform](sa_cam, GlobalTransform())
    _ = sa.world.insert[PrevGlobalTransform](sa_cam, PrevGlobalTransform())
    _ = sa.world.insert[CameraComponent](sa_cam, CameraComponent(fov_y_deg=55.0, active=True))
    sa.step_one_frame(Float32(1.0 / 60.0))

    var rw_sa = sa.resources.get_ptr[RenderWorld]()
    var rend_sa = sa.resources.get_ptr[ScalableRenderer]()
    var sa_mat_id = -1
    for ii in range(len(rw_sa.value()[unsafe_offset=0].instances)):
        if rw_sa.value()[unsafe_offset=0].instances[ii].entity_id == hero_bits:
            sa_mat_id = rw_sa.value()[unsafe_offset=0].instances[ii].material_id
            break
    if sa_mat_id < 0 and len(rw_sa.value()[unsafe_offset=0].instances) > 0:
        sa_mat_id = rw_sa.value()[unsafe_offset=0].instances[0].material_id
    assert_true(sa_mat_id >= 0 and sa_mat_id != game_mat0)
    assert_true(abs(rend_sa.value()[unsafe_offset=0].materials[sa_mat_id].base_color.x - exp_r) < Float32(1.0e-4))
    assert_true(abs(rend_sa.value()[unsafe_offset=0].materials[sa_mat_id].base_color.y - exp_g) < Float32(1.0e-4))
    assert_true(abs(rend_sa.value()[unsafe_offset=0].materials[sa_mat_id].base_color.z - exp_b) < Float32(1.0e-4))
    assert_true(abs(rend_sa.value()[unsafe_offset=0].materials[game_mat0].base_color.z - Float32(0.9)) < Float32(1.0e-4))
    sa.shutdown()


def test_iteration5_cel_vfx_mo_enabled_parity() raises:
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_1_FORWARD_CLUSTERED,
        width=160,
        height=120,
        worker_threads=0,
    )
    var rend_ptr = studio.app.resources.get_ptr_mut[ScalableRenderer]()
    assert_true(Bool(rend_ptr))
    if not rend_ptr.value()[unsafe_offset=0].uses_vulkan():
        studio.shutdown()
        return

    var mat0 = rend_ptr.value()[unsafe_offset=0].register_material(
        MaterialAsset(base_color=Vec3(0.1, 0.2, 0.9), roughness=0.55, metallic=0.0)
    )
    var other_mat = rend_ptr.value()[unsafe_offset=0].register_material(
        MaterialAsset(base_color=Vec3(0.85, 0.85, 0.1), roughness=0.9, metallic=0.9)
    )

    # 1. HeroCel with mo.enabled=True preserves mo.enabled across Cel & VFX + .flscn Tier-0 / standalone reload
    var hero_id = studio.spawn_entity(
        "HeroCel",
        position=Vec3(0.0, 1.0, 0.0),
        base_color=Vec3(0.95, 0.05, 0.05),
        record_undo=False,
    )
    var hero_e = studio.resolve_entity(hero_id)
    studio.app.world.get_ptr_mut[MeshRendererComponent](hero_e).value()[unsafe_offset=0].material_id = mat0
    studio.app.world.get_ptr_mut[MaterialOverrideComponent](hero_e).value()[unsafe_offset=0].material_id = mat0
    studio._sync_compat_view()
    assert_true(studio.app.world.get_ptr[MaterialOverrideComponent](hero_e).value()[unsafe_offset=0].enabled)

    assert_true(studio.configure_entity_npr_cel(hero_id, cel_outline_width_px=2.0))
    assert_true(studio.app.world.get_ptr[MaterialOverrideComponent](hero_e).value()[unsafe_offset=0].enabled)
    studio.render_editor_frame(160, 120)
    var cel_slot = studio.gpu_materials.entity_cel_mat_ids[hero_id]
    assert_true(cel_slot >= 0)
    ref gpu = rend_ptr.value()[unsafe_offset=0].gpu_slot[0]
    assert_true(abs(gpu.materials[cel_slot].base_color.x - Float32(0.95)) < Float32(1.0e-4))
    assert_true(abs(gpu.materials[cel_slot].base_color.y - Float32(0.05)) < Float32(1.0e-4))
    assert_true(abs(gpu.materials[cel_slot].base_color.z - Float32(0.05)) < Float32(1.0e-4))

    var cel_scene_path = "/tmp/flint_test_iter5_cel_parity.flscn"
    assert_true(studio.save_scene_file(cel_scene_path))
    _verify_saved_cel_vfx_scene_parity(
        cel_scene_path, hero_id, hero_e.to_bits(), Float32(0.95), Float32(0.05), Float32(0.05)
    )
    _ = remove_file(cel_scene_path)

    assert_true(studio.configure_entity_stylized_vfx(hero_id, vfx_erosion=0.42))
    assert_true(studio.app.world.get_ptr[MaterialOverrideComponent](hero_e).value()[unsafe_offset=0].enabled)
    studio.render_editor_frame(160, 120)
    var vfx_slot = studio.gpu_materials.entity_vfx_mat_ids[hero_id]
    assert_true(vfx_slot >= 0)
    assert_true(abs(gpu.materials[vfx_slot].base_color.x - Float32(0.95)) < Float32(1.0e-4))
    assert_true(abs(gpu.materials[vfx_slot].base_color.y - Float32(0.05)) < Float32(1.0e-4))
    assert_true(abs(gpu.materials[vfx_slot].base_color.z - Float32(0.05)) < Float32(1.0e-4))

    var vfx_scene_path = "/tmp/flint_test_iter5_vfx_parity.flscn"
    assert_true(studio.save_scene_file(vfx_scene_path))
    _verify_saved_cel_vfx_scene_parity(
        vfx_scene_path, hero_id, hero_e.to_bits(), Float32(0.95), Float32(0.05), Float32(0.05)
    )
    _ = remove_file(vfx_scene_path)

    # 2. Entity starting with mo.enabled=False preserves False across PBR -> Cel -> VFX -> PBR
    # and renders mat0 (0.1, 0.2, 0.9) instead of mo.base_color (0.95, 0.05, 0.05) or mo.material_id (other_mat)
    var dis_id = studio.spawn_entity(
        "DisabledCel",
        position=Vec3(2.0, 1.0, 0.0),
        base_color=Vec3(0.95, 0.05, 0.05),
        record_undo=False,
    )
    var dis_e = studio.resolve_entity(dis_id)
    studio.app.world.get_ptr_mut[MeshRendererComponent](dis_e).value()[unsafe_offset=0].material_id = mat0
    studio.app.world.get_ptr_mut[MaterialOverrideComponent](dis_e).value()[unsafe_offset=0].material_id = other_mat
    studio.app.world.get_ptr_mut[MaterialOverrideComponent](dis_e).value()[unsafe_offset=0].enabled = False
    studio._sync_compat_view()
    assert_false(studio.app.world.get_ptr[MaterialOverrideComponent](dis_e).value()[unsafe_offset=0].enabled)

    assert_true(studio.configure_entity_npr_cel(dis_id, cel_outline_width_px=2.0))
    assert_false(studio.app.world.get_ptr[MaterialOverrideComponent](dis_e).value()[unsafe_offset=0].enabled)
    studio.render_editor_frame(160, 120)
    var dis_cel_slot = studio.gpu_materials.entity_cel_mat_ids[dis_id]
    assert_true(dis_cel_slot >= 0)
    assert_true(abs(gpu.materials[dis_cel_slot].base_color.x - Float32(0.1)) < Float32(1.0e-4))
    assert_true(abs(gpu.materials[dis_cel_slot].base_color.y - Float32(0.2)) < Float32(1.0e-4))
    assert_true(abs(gpu.materials[dis_cel_slot].base_color.z - Float32(0.9)) < Float32(1.0e-4))

    assert_true(studio.configure_entity_stylized_vfx(dis_id, vfx_erosion=0.35))
    assert_false(studio.app.world.get_ptr[MaterialOverrideComponent](dis_e).value()[unsafe_offset=0].enabled)
    studio.render_editor_frame(160, 120)
    var dis_vfx_slot = studio.gpu_materials.entity_vfx_mat_ids[dis_id]
    assert_true(dis_vfx_slot >= 0)
    assert_true(abs(gpu.materials[dis_vfx_slot].base_color.x - Float32(0.1)) < Float32(1.0e-4))
    assert_true(abs(gpu.materials[dis_vfx_slot].base_color.y - Float32(0.2)) < Float32(1.0e-4))
    assert_true(abs(gpu.materials[dis_vfx_slot].base_color.z - Float32(0.9)) < Float32(1.0e-4))

    var pbr_v = studio.entities[dis_id].copy()
    pbr_v.shading_mode = SHADING_PBR
    assert_true(studio.commit_node_edit(dis_id, pbr_v^, "Back to PBR Disabled"))
    assert_false(studio.app.world.get_ptr[MaterialOverrideComponent](dis_e).value()[unsafe_offset=0].enabled)

    # Clearcoat-only in Cel mode when mo.enabled == False still preserves mat0 base_color
    assert_true(studio.configure_entity_npr_cel(dis_id, cel_outline_width_px=2.0))
    studio.app.world.get_ptr_mut[StudioStyleComponent](dis_e).value()[unsafe_offset=0].clearcoat = Float32(0.65)
    studio.render_editor_frame(160, 120)
    var dis_cel_cc_slot = studio.gpu_materials.entity_cel_mat_ids[dis_id]
    assert_true(dis_cel_cc_slot >= 0)
    assert_true(abs(gpu.materials[dis_cel_cc_slot].base_color.x - Float32(0.1)) < Float32(1.0e-4))
    assert_true(abs(gpu.materials[dis_cel_cc_slot].base_color.y - Float32(0.2)) < Float32(1.0e-4))
    assert_true(abs(gpu.materials[dis_cel_cc_slot].base_color.z - Float32(0.9)) < Float32(1.0e-4))

    # 3. state_hash() detects GlobalTransform changes on entities lacking LocalTransform
    var gt_only_e = studio.app.world.spawn()
    _ = studio.app.world.insert[GlobalTransform](gt_only_e, GlobalTransform.from_local(LocalTransform(Vec3(1.0, 2.0, 3.0))))
    var h_gt1 = studio.state_hash()
    _ = studio.app.world.insert[GlobalTransform](gt_only_e, GlobalTransform.from_local(LocalTransform(Vec3(42.0, 7.0, -3.0))))
    var h_gt2 = studio.state_hash()
    assert_true(h_gt1 != h_gt2)

    studio.shutdown()


def test_m3a_dockspace_json_validation_and_env_spotlight_tier0() raises:
    # 1. DockSpace.load_json strict JSON & structural validation without wiping layout on error
    var dock = DockSpace.create_studio_5panel()
    var before_json = dock.save_json()
    assert_false(dock.load_json("[]"))
    assert_false(dock.load_json('{"version": 2, "root": -1, "nodes": []}'))
    assert_false(dock.load_json('{"version": "1", "root": -1, "nodes": []}'))
    assert_false(dock.load_json('{"version": 1, "root": "0", "nodes": []}'))
    assert_false(dock.load_json('{"version": 1, "root": -1, "nodes": "not_an_array"}'))
    assert_false(dock.load_json('{"version": 1, "root": 0, "nodes": ["bad_node"]}'))
    assert_false(dock.load_json('{"version": 1, "root": 0, "nodes": [{"id": 0, "child_a": -1, "child_b": -1}]}'))
    assert_false(dock.load_json('{"version": 1, "root": 0, "nodes": [{"id": 0, "kind": 999, "child_a": -1, "child_b": -1}]}'))
    assert_false(dock.load_json('{"version": 1, "root": 0, "nodes": [{"id": 0, "kind": 2, "child_a": -1, "child_b": -1, "tabs": "not_arr"}]}'))
    assert_false(dock.load_json('{"version": 1, "root": 0, "nodes": [{"id": 0, "kind": 2, "child_a": 0, "child_b": -1, "tabs": ["A"]}]}'))
    assert_false(dock.load_json('{"version": 1.5, "root": 0, "nodes": [{"id": 0, "kind": 2, "child_a": -1, "child_b": -1}]}'))
    assert_false(dock.load_json('{"version": 1, "root": 0.5, "nodes": [{"id": 0, "kind": 2, "child_a": -1, "child_b": -1}]}'))
    assert_false(dock.load_json('{"version": 1, "root": 0, "nodes": [{"id": 0, "kind": 2.9, "child_a": -1, "child_b": -1}]}'))
    assert_false(dock.load_json('{"version": 1, "root": 0, "nodes": [{"id": 0, "kind": 2, "child_a": "0", "child_b": -1}]}'))
    assert_false(dock.load_json('{"version": 1, "root": 0, "nodes": [{"id": 0, "kind": 2, "child_a": -1, "child_b": "0"}]}'))
    assert_equal(dock.save_json(), before_json)
    assert_true(dock.load_json(before_json))

    # 2. EnvironmentComponent & SpotLightComponent codecs, extraction, and Tier-0 sun/exposure modulation
    var reg = ComponentRegistry.with_builtins()
    register_render_codecs(reg)
    assert_true(reg.find_by_name("EnvironmentComponent") >= 0)
    assert_true(reg.find_by_name("SpotLightComponent") >= 0)

    var env_def = EnvironmentComponent()
    var env_bw = BlobWriter()
    env_def.write_bin(env_bw)
    var env_br = BlobReader(env_bw.data.copy())
    var env_rt = EnvironmentComponent.read_bin(env_br, EnvironmentComponent.SCHEMA_VERSION)
    assert_true(abs(env_rt.exposure - Float32(1.0)) < Float32(1.0e-5))

    var spt_def = SpotLightComponent()
    var spt_bw = BlobWriter()
    spt_def.write_bin(spt_bw)
    var spt_br = BlobReader(spt_bw.data.copy())
    var spt_rt = SpotLightComponent.read_bin(spt_br, SpotLightComponent.SCHEMA_VERSION)
    assert_true(abs(spt_rt.outer_deg - Float32(35.0)) < Float32(1.0e-5))

    var app = EngineApp(AppConfig(title="m3a_env_spot", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    app.startup()

    var rend = app.resources.get_ptr_mut[ScalableRenderer]()
    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(), Vec3(1.5, 1.5, 1.5))
    var mesh_id = rend.value()[unsafe_offset=0].register_mesh(mb^.finish(), 0)

    var cam_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(0.0, 0.0, 5.0)))
    _ = app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
    _ = app.world.insert[CameraComponent](cam_e, CameraComponent(active=True))

    var box_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](box_e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](box_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](box_e, PrevGlobalTransform())
    _ = app.world.insert[MeshRendererComponent](box_e, MeshRendererComponent(mesh_id, -1))

    var env_e = app.world.spawn()
    _ = app.world.insert[EnvironmentComponent](
        env_e, EnvironmentComponent(sun_dir=Vec3(0.0, -1.0, 0.0), sun_radiance=Vec3(0.4, 0.4, 0.4), exposure=0.4)
    )

    var spot_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](spot_e, LocalTransform(Vec3(1.0, 3.0, 2.0)))
    _ = app.world.insert[GlobalTransform](spot_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](spot_e, PrevGlobalTransform())
    _ = app.world.insert[SpotLightComponent](spot_e, SpotLightComponent(radiance=Vec3(9.0, 6.0, 3.0), range=14.0))

    var spot_dis = app.world.spawn()
    _ = app.world.insert[LocalTransform](spot_dis, LocalTransform(Vec3(0.0, 2.0, 0.0)))
    _ = app.world.insert[GlobalTransform](spot_dis, GlobalTransform())
    _ = app.world.insert[SpotLightComponent](spot_dis, SpotLightComponent(enabled=False))

    app.step_one_frame(Float32(1.0 / 60.0))
    var rw_ptr = app.resources.get_ptr[RenderWorld]()
    assert_true(Bool(rw_ptr))
    assert_equal(len(rw_ptr.value()[unsafe_offset=0].cluster_lights), 1)
    assert_equal(rw_ptr.value()[unsafe_offset=0].cluster_lights[0].kind, AAA_LIGHT_SPOT)
    var px1 = rend.value()[unsafe_offset=0].readback_pixels()

    var env_mut = app.world.get_ptr_mut[EnvironmentComponent](env_e)
    env_mut.value()[unsafe_offset=0].sun_radiance = Vec3(8.0, 6.0, 4.0)
    env_mut.value()[unsafe_offset=0].exposure = 2.5
    app.step_one_frame(Float32(1.0 / 60.0))
    var px2 = rend.value()[unsafe_offset=0].readback_pixels()
    var diffs = 0
    for i in range(len(px1)):
        if px1[i] != px2[i]:
            diffs += 1
    assert_true(diffs > 50)
    app.shutdown()


def _m3b_write_text(path: String, text: String):
    var sb = text.as_bytes()
    var buf = List[UInt8](capacity=len(sb))
    for i in range(len(sb)):
        buf.append(sb[i])
    _ = write_bytes(path, buf)


def _m3b_rm_rf(path: String):
    var cmd = "rm -rf '" + path + "'"
    _ = external_call["system", Int32](cmd.as_c_string_span())


def test_m3b_f11_input_context_hotkeys_dirty_modal_and_layout() raises:
    var tmp = "/tmp/flint_test_m3b_f11"
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp)
    var state_dir = tmp + "/state"
    var prev_state_dir = env_value("FLINT_STATE_DIR")
    var env_k = String("FLINT_STATE_DIR")
    _ = external_call["setenv", Int32](env_k.as_c_string_span(), state_dir.as_c_string_span(), Int32(1))

    try:
        # 1. StudioEntityMetaComponent v1 backward compat + v2 enabled round-trip
        var v1w = BlobWriter()
        v1w.text("LegacyEnt")
        v1w.boolean(False)
        v1w.u64(0)
        v1w.u64(0)
        v1w.integer(2)
        v1w.vec3(Vec3(0.5, 0.5, 0.5))
        v1w.f32(0.32)
        v1w.f32(0.55)
        v1w.integer(-1)
        var v1r = BlobReader(v1w.data.copy())
        var m1 = StudioEntityMetaComponent.read_bin(v1r, 1)
        assert_equal(m1.name, "LegacyEnt")
        assert_true(m1.enabled)

        var m2 = StudioEntityMetaComponent(name="V2Ent", enabled=False)
        var v2w = BlobWriter()
        m2.write_bin(v2w)
        var v2r = BlobReader(v2w.data.copy())
        var m2_rt = StudioEntityMetaComponent.read_bin(v2r, 2)
        assert_false(m2_rt.enabled)
        var m2_j = m2.write_json()
        var jbytes = List[UInt8](capacity=m2_j.byte_length())
        for b in m2_j.as_bytes():
            jbytes.append(b)
        var jdoc = parse_json(jbytes)
        var m2_jrt = StudioEntityMetaComponent.read_json(jdoc, 0)
        assert_false(m2_jrt.enabled)

        # 2. InputContext split, F/Ctrl+Shift+Z/Ctrl+C/Ctrl+V/X, dirty * + Ctrl+Q modal + layout persistence
        var s = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=1280, height=720, worker_threads=0)
        var eid = s.spawn_entity("Target", position=Vec3(6.0, 2.0, -4.0), rotation=Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), Float32(0.8)), record_undo=False)
        s.select_entity(eid)
        assert_false(s.dirty)

        # Clean Ctrl+Q quits immediately
        var inp = InputState()
        inp.begin_frame()
        inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
        inp.set_logical_key(81, True)
        assert_false(s.process_editor_input(inp, Float32(1.0 / 60.0)))
        inp.begin_frame()
        inp.set_logical_key(UI_KEY_LEFT_CTRL, False)
        inp.set_logical_key(81, False)
        s.quit_requested = False

        # Frame selection (F = 70)
        inp.begin_frame()
        inp.set_logical_key(70, True)
        assert_true(s.process_editor_input(inp, Float32(1.0 / 60.0)))
        assert_true(abs(s.cam_target.x - Float32(6.0)) < Float32(1.0e-3))
        inp.begin_frame()
        inp.set_logical_key(70, False)

        # Toggle Local/World gizmo (X = 88)
        assert_false(s.gizmo_local_space)
        inp.begin_frame()
        inp.set_logical_key(88, True)
        _ = s.process_editor_input(inp, Float32(1.0 / 60.0))
        assert_true(s.gizmo_local_space)
        var lrot = s.selection_gizmo_rotation()
        assert_true(abs(lrot.y) > Float32(0.1))
        inp.begin_frame()
        inp.set_logical_key(88, False)

        # Ctrl+C + Ctrl+V duplicate (undoable & redoable via Ctrl+Shift+Z)
        inp.begin_frame()
        inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
        inp.set_logical_key(67, True)
        _ = s.process_editor_input(inp, Float32(1.0 / 60.0))
        inp.begin_frame()
        inp.set_logical_key(67, False)
        inp.set_logical_key(86, True)
        _ = s.process_editor_input(inp, Float32(1.0 / 60.0))
        assert_equal(s.live_entity_count(), 2)
        assert_true(s.dirty)
        assert_true(s.app.config.title.endswith("*"))
        assert_true(s.app.resources.get_ptr[WindowStateResource]().value()[unsafe_offset=0].title.endswith("*"))
        inp.begin_frame()
        inp.set_logical_key(86, False)
        inp.set_logical_key(UI_KEY_LEFT_CTRL, False)

        # Undo then Ctrl+Shift+Z redo
        assert_true(s.undo())
        assert_equal(s.live_entity_count(), 1)
        inp.begin_frame()
        inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
        inp.set_logical_key(UI_KEY_LEFT_SHIFT, True)
        inp.set_logical_key(90, True)
        _ = s.process_editor_input(inp, Float32(1.0 / 60.0))
        assert_equal(s.live_entity_count(), 2)
        inp.begin_frame()
        inp.set_logical_key(90, False)
        inp.set_logical_key(UI_KEY_LEFT_SHIFT, False)
        inp.set_logical_key(UI_KEY_LEFT_CTRL, False)

        # StudioPlay blocks Delete / Esc / X in PIE
        assert_true(s.pie_play())
        inp.begin_frame()
        inp.set_logical_key(261, True)  # Delete
        inp.set_logical_key(256, True)  # Esc
        inp.set_logical_key(88, True)   # X
        assert_true(s.process_editor_input(inp, Float32(1.0 / 60.0)))
        assert_equal(s.live_entity_count(), 2)
        assert_true(s.gizmo_local_space)
        inp.begin_frame()
        inp.set_logical_key(261, False)
        inp.set_logical_key(256, False)
        inp.set_logical_key(88, False)
        assert_true(s.pie_stop())

        # Dirty Ctrl+Q opens confirm modal instead of exiting; new_scene(force=False) also deferred
        inp.begin_frame()
        inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
        inp.set_logical_key(81, True)
        assert_true(s.process_editor_input(inp, Float32(1.0 / 60.0)))
        assert_true(s.confirm_modal_open)
        assert_equal(s.confirm_modal_action, 1)
        s.confirm_modal_open = False
        s.confirm_modal_action = 0
        assert_false(s.new_scene(force=False))
        assert_true(s.confirm_modal_open)
        assert_equal(s.confirm_modal_action, 2)
        s.confirm_modal_open = False
        assert_true(s.save_scene_file(tmp + "/f11.flscn"))
        assert_false(s.dirty)
        assert_true(len(s.toasts) > 0)

        # DockSpace layout save + reload persistence
        s.dock.header_height = Float32(28.0)
        s.shutdown()
        var s2 = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=1280, height=720, worker_threads=0)
        assert_true(abs(s2.dock.header_height - Float32(28.0)) < Float32(0.01))
        s2.shutdown()
    finally:
        if prev_state_dir.byte_length() > 0:
            _ = external_call["setenv", Int32](env_k.as_c_string_span(), prev_state_dir.as_c_string_span(), Int32(1))
        else:
            _ = external_call["unsetenv", Int32](env_k.as_c_string_span())
        _m3b_rm_rf(tmp)


def test_inspector_sections_and_frame_selection() raises:
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=1280, height=720, worker_threads=0)
    var target = Vec3(7.0, 2.0, -3.0)
    var eid = studio.spawn_entity("Bridge support", position=target, record_undo=False)
    studio.select_entity(eid)
    var input = InputState()
    for _ in range(2):
        input.begin_frame()
        studio.render_editor_frame(1280, 720, input=input)
    var dock = hash_ui_id(UInt64(0xCBF29CE484222325), "dock:Inspector")
    var scroll = hash_ui_id(dock, "insp_scroll")
    var header = hash_ui_id(scroll, "insp_header_actions")
    var frame = studio.ui_ctx.cached_rects[hash_ui_id(header, "insp_frame_entity")]
    var view = studio.ui_ctx.cached_rects[scroll]
    assert_true(frame.x >= view.x and frame.right() <= view.right())
    assert_true((studio.cam_target - target).length() > Float32(0.1))
    input.begin_frame()
    input.move_mouse(frame.center_x(), frame.center_y())
    input.set_mouse(0, True)
    studio.render_editor_frame(1280, 720, input=input)
    input.begin_frame()
    input.set_mouse(0, False)
    studio.render_editor_frame(1280, 720, input=input)
    assert_true((studio.cam_target - target).length() < Float32(0.001))

    var kind_id = hash_ui_id(scroll, "col_kind")
    var kind = studio.ui_ctx.cached_rects[kind_id]
    input.begin_frame()
    input.move_mouse(kind.x + 5, kind.center_y())
    input.set_mouse(0, True)
    studio.render_editor_frame(1280, 720, input=input)
    assert_true(studio.ui_ctx._open_popup_id != kind_id)
    input.begin_frame()
    input.set_mouse(0, False)
    studio.render_editor_frame(1280, 720, input=input)
    input.begin_frame()
    input.move_mouse(kind.right() - 12, kind.center_y())
    input.set_mouse(0, True)
    studio.render_editor_frame(1280, 720, input=input)
    assert_equal(studio.ui_ctx._open_popup_id, kind_id)
    var menu = studio.ui_ctx._popup_rect
    input.begin_frame()
    input.set_mouse(0, False)
    studio.render_editor_frame(1280, 720, input=input)
    input.begin_frame()
    input.move_mouse(menu.center_x(), menu.y + 2 + studio.ui_ctx.theme.row_height * 1.5)
    input.set_mouse(0, True)
    studio.render_editor_frame(1280, 720, input=input)
    assert_equal(studio.entities[eid].collider_kind, 1)
    assert_true(studio.undo())
    assert_equal(studio.entities[eid].collider_kind, 2)
    input.begin_frame()
    input.set_mouse(0, False)
    studio.render_editor_frame(1280, 720, input=input)

    var transform = hash_ui_id(scroll, "insp_trs")
    var section = studio.ui_ctx.cached_rects[transform]
    assert_true(studio.ui_ctx.state_map[transform].tree_open)
    input.begin_frame()
    input.move_mouse(section.center_x(), section.center_y())
    input.set_mouse(0, True)
    studio.render_editor_frame(1280, 720, input=input)
    assert_false(studio.ui_ctx.state_map[transform].tree_open)
    assert_equal(studio.selected_entity_id, eid)

    studio.set_ui_scale(1.5)
    for _ in range(2):
        input.begin_frame()
        input.set_mouse(0, False)
        studio.render_editor_frame(1280, 720, input=input)
    var albedo = studio.ui_ctx.cached_rects[hash_ui_id(scroll, "mat_col")]
    assert_true(albedo.h >= studio.ui_ctx.theme.row_height * 2)
    var ui_scale = studio.effective_ui_scale()
    input.begin_frame()
    input.move_mouse((albedo.right() - 12) * ui_scale, (albedo.bottom() - 10) * ui_scale)
    studio.render_editor_frame(1280, 720, input=input)
    assert_equal(studio.ui_ctx.hovered_id, hash_ui_id(scroll, "mat_col#c2"))
    var blue = studio.ui_ctx.hovered_rect
    assert_true(blue.w >= studio.ui_ctx._measure_text("1.00", studio.ui_ctx.theme.font_size * 0.84) + 8)
    assert_true(blue.right() <= albedo.right() + 0.01)

    section = studio.ui_ctx.cached_rects[transform]
    input.begin_frame()
    input.move_mouse(section.center_x() * ui_scale, section.center_y() * ui_scale)
    input.set_mouse(0, True)
    studio.render_editor_frame(1280, 720, input=input)
    assert_true(studio.ui_ctx.state_map[transform].tree_open)
    input.begin_frame()
    input.set_mouse(0, False)
    studio.render_editor_frame(1280, 720, input=input)
    var position = studio.ui_ctx.cached_rects[hash_ui_id(scroll, "prow:Position")]
    assert_true(position.h > 24)
    var z_x = (position.right() - 12) * ui_scale
    var z_y = (position.bottom() - 10) * ui_scale
    input.begin_frame()
    input.move_mouse(z_x, z_y)
    input.set_mouse(0, True)
    studio.render_editor_frame(1280, 720, input=input)
    assert_equal(studio.ui_ctx.active_id, hash_ui_id(scroll, "tr_pos#v2_2"))
    assert_true(studio.ui_ctx.hovered_rect.w >= studio.ui_ctx._measure_text("-3.00", studio.studio_theme.font_mono) + 17)
    input.begin_frame()
    input.move_mouse(z_x + 12, z_y)
    studio.render_editor_frame(1280, 720, input=input)
    assert_true(studio.entities[eid].translation.z > target.z)
    assert_equal(studio.entities[eid].translation.x, target.x)
    assert_equal(studio.entities[eid].translation.y, target.y)
    studio.shutdown()


def test_m3b_f12_inspector_guards_and_enabled_sync() raises:
    var s = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=1280, height=720, worker_threads=0)
    var eid = s.spawn_entity("GuardTarget", position=Vec3(0.0, 1.0, 0.0), record_undo=False)
    var e = s.resolve_entity(eid)
    # Direct ECS trigger mutation must NOT be wiped by stale StudioEntityView commit
    var col_ptr = s.app.world.get_ptr_mut[ColliderComponent](e)
    assert_true(Bool(col_ptr))
    col_ptr.value()[unsafe_offset=0].is_trigger = True
    var stale_view = s.entities[eid].copy()
    stale_view.name = "RenamedGuard"
    assert_true(s.commit_node_edit(eid, stale_view^, "Rename Only"))
    assert_true(s.app.world.get_ptr[ColliderComponent](e).value()[unsafe_offset=0].is_trigger)
    assert_true(s.entities[eid].is_trigger)

    # Explicitly toggling is_trigger updates ECS
    var trig_view = s.entities[eid].copy()
    trig_view.is_trigger = False
    assert_true(s.commit_node_edit(eid, trig_view^, "Clear Trigger"))
    assert_false(s.app.world.get_ptr[ColliderComponent](e).value()[unsafe_offset=0].is_trigger)

    # Lowering clearcoat syncs StudioStyleComponent.clearcoat
    var cc_up = s.entities[eid].copy()
    cc_up.clearcoat = Float32(0.85)
    assert_true(s.commit_node_edit(eid, cc_up^, "Set CC"))
    assert_true(abs(s.app.world.get_ptr[StudioStyleComponent](e).value()[unsafe_offset=0].clearcoat - Float32(0.85)) < Float32(1.0e-4))
    var cc_down = s.entities[eid].copy()
    cc_down.clearcoat = Float32(0.15)
    assert_true(s.commit_node_edit(eid, cc_down^, "Lower CC"))
    assert_true(abs(s.app.world.get_ptr[StudioStyleComponent](e).value()[unsafe_offset=0].clearcoat - Float32(0.15)) < Float32(1.0e-4))

    # enabled syncs StudioEntityMetaComponent.enabled and MeshRendererComponent.visible
    var dis_v = s.entities[eid].copy()
    dis_v.enabled = False
    assert_true(s.commit_node_edit(eid, dis_v^, "Disable Entity"))
    assert_false(s.app.world.get_ptr[StudioEntityMetaComponent](e).value()[unsafe_offset=0].enabled)
    assert_false(s.app.world.get_ptr[MeshRendererComponent](e).value()[unsafe_offset=0].visible)

    # inspect_fields includes all F12 reflected fields
    var fv = FieldVisitor()
    s.entities[eid].inspect_fields(fv)
    var seen = 0
    for fi in range(len(fv.fields)):
        var fname = fv.fields[fi].name
        if fname == "friction" or fname == "restitution" or fname == "opacity" or fname == "clearcoat" or fname == "sheen" or fname == "subsurface" or fname == "is_trigger" or fname == "enabled":
            seen += 1
    assert_equal(seen, 8)
    s.shutdown()


def test_m3b_f13_recursive_asset_browser_and_spawn_cards() raises:
    var tmp = "/tmp/flint_test_m3b_f13"
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp + "/materials/sub")
    _ = make_dirs(tmp + "/prefabs")
    _ = make_dirs(tmp + "/scenes")
    _ = make_dirs(tmp + "/.git")
    _ = make_dirs(tmp + "/build")
    try:
        for metadata in [".claude", ".codex", ".gemini"]:
            var checkout = tmp + "/" + metadata + "/worktrees/editor"
            _ = make_dirs(checkout)
            _m3b_write_text(checkout + "/duplicate.flscn", "{}")
        for i in range(8):
            _m3b_write_text(tmp + "/materials/sub/mat_" + String(i) + ".flmat", "{}")
        for i in range(4):
            _m3b_write_text(tmp + "/prefabs/pf_" + String(i) + ".flprefab", "{}")
        _m3b_write_text(tmp + "/scenes/demo.flscn", '{"schema":2,"entities":[]}')
        _m3b_write_text(tmp + "/.git/ignored.flmat", "{}")
        _m3b_write_text(tmp + "/build/ignored.flscn", "{}")

        var s = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=1600, height=900, worker_threads=0)
        var n_found = s.scan_asset_browser(tmp)
        assert_equal(n_found, 13)
        s.render_editor_frame(1600, 900)
        var ab_focus_count = len(s.ui_ctx.bg_focusables)
        assert_true(ab_focus_count >= 13 + 7)  # 6 spawn cards + rescan + 13 asset items

        # Spawn all 6 cards and verify component kinds
        var e0 = s.spawn_card_entity(0)  # Empty
        var e1 = s.spawn_card_entity(1)  # Cube
        var e2 = s.spawn_card_entity(2)  # Point Light
        var e3 = s.spawn_card_entity(3)  # Spot Light
        var e4 = s.spawn_card_entity(4)  # Camera
        var e5 = s.spawn_card_entity(5)  # Audio Emitter
        assert_false(s.app.world.has[MeshRendererComponent](s.resolve_entity(e0)))
        assert_true(s.app.world.has[MeshRendererComponent](s.resolve_entity(e1)))
        assert_true(s.app.world.has[PointLightComponent](s.resolve_entity(e2)))
        assert_true(s.app.world.has[SpotLightComponent](s.resolve_entity(e3)))
        assert_true(s.app.world.has[CameraComponent](s.resolve_entity(e4)))
        assert_true(s.app.world.has[AudioEmitterComponent](s.resolve_entity(e5)))
        assert_equal(s.live_entity_count(), 6)
        assert_true(s.undo())
        assert_equal(s.live_entity_count(), 5)
        s.shutdown()
    finally:
        _m3b_rm_rf(tmp)



def _click_library_control(mut studio: FlintStudioState, mut input: InputState, id: UInt64) raises:
    var rect = studio.ui_ctx.cached_rects[id]
    assert_true(rect.w > 0 and rect.h > 0)
    for down in [True, False]:
        input.begin_frame()
        input.move_mouse(rect.center_x(), rect.center_y())
        input.set_mouse(0, down)
        studio.render_editor_frame(1280, 720, input=input)


def test_project_assets_import_search_and_create() raises:
    var tmp = "/tmp/flint-library-ui-" + String(external_call["getpid", Int32]())
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp + "/project")
    _m3b_write_text(tmp + "/palette.json", "{}")
    try:
        var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=1280, height=720, worker_threads=0)
        var input = InputState()
        var root = UInt64(0xCBF29CE484222325)
        var dock = hash_ui_id(root, "dock:AssetBrowser")
        var bar = hash_ui_id(dock, "ab_toolbar")
        var modal = hash_ui_id(root, "modal:ab_dialog")
        for _ in range(3):
            studio.render_editor_frame(1280, 720, input=input)
        _click_library_control(studio, input, hash_ui_id(bar, "ab_import"))
        assert_true(studio.asset_browser_state.dialog != 0)
        input.begin_frame()
        studio.render_editor_frame(1280, 720, input=input)
        _click_library_control(studio, input, hash_ui_id(modal, "ab_source"))
        input.begin_frame()
        input.push_text_utf8(tmp + "/palette.json")
        studio.render_editor_frame(1280, 720, input=input)
        _click_library_control(studio, input, hash_ui_id(modal, "ab_project"))
        input.begin_frame()
        input.push_text_utf8(tmp + "/project")
        studio.render_editor_frame(1280, 720, input=input)
        _click_library_control(studio, input, hash_ui_id(hash_ui_id(modal, "ab_dialog_actions"), "ab_confirm"))
        assert_equal(studio.asset_browser_state.dialog, 0)
        assert_equal(file_size(tmp + "/project/assets/palette.json"), 2)
        assert_equal(len(studio.asset_browser_entries), 1)
        input.begin_frame()
        studio.render_editor_frame(1280, 720, input=input)
        _click_library_control(studio, input, hash_ui_id(bar, "ab_search"))
        input.begin_frame()
        input.push_text_utf8("unmatched")
        studio.render_editor_frame(1280, 720, input=input)
        assert_equal(studio.asset_browser_state.search_query, "unmatched")
        var entry_id = hash_ui_id(hash_ui_id(dock, "ab_entries_scroll"), "ab_item_0")
        for id in studio.ui_ctx.bg_focusables:
            assert_true(id != entry_id)
        _click_library_control(studio, input, hash_ui_id(bar, "ab_create"))
        input.begin_frame()
        studio.render_editor_frame(1280, 720, input=input)
        var count = studio.live_entity_count()
        _click_library_control(studio, input, hash_ui_id(hash_ui_id(modal, "ab_create_row_0"), "card_cube"))
        assert_equal(studio.live_entity_count(), count + 1)
        assert_equal(studio.asset_browser_state.dialog, 0)
        studio.shutdown()
    finally:
        _m3b_rm_rf(tmp)


def test_m3b_f13_28_assets_spawn_cards_layout_and_clicks() raises:
    var tmp = "/tmp/flint_test_m3b_f13_28"
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp + "/assets/mats/sub")
    _ = make_dirs(tmp + "/assets/prefabs/nested")
    _ = make_dirs(tmp + "/assets/anims")
    _ = make_dirs(tmp + "/assets/audio")
    _ = make_dirs(tmp + "/scenes")
    _ = make_dirs(tmp + "/.git")
    _ = make_dirs(tmp + "/build")
    try:
        for i in range(10):
            _m3b_write_text(tmp + "/assets/mats/sub/mat_" + String(i) + ".flmat", "{}")
        for i in range(6):
            _m3b_write_text(tmp + "/assets/prefabs/nested/pf_" + String(i) + ".flprefab", "{}")
        for i in range(4):
            _m3b_write_text(tmp + "/assets/anims/anim_" + String(i) + ".flanim", "{}")
        _m3b_write_text(tmp + "/assets/anims/boss.flrig", "FRIG")
        _m3b_write_text(tmp + "/assets/hero.glb", "glTF")
        _m3b_write_text(tmp + "/assets/albedo.png", "PNG")
        _m3b_write_text(tmp + "/assets/sky.hdr", "HDR")
        _m3b_write_text(tmp + "/assets/audio/hit.wav", "RIFF")
        _m3b_write_text(tmp + "/assets/audio/bgm.qoa", "qoaf")
        _m3b_write_text(tmp + "/assets/config.json", "{}")
        _m3b_write_text(tmp + "/scenes/level1.flscn", '{"schema":2,"entities":[]}')
        # Ignored non-asset / excluded directories
        _m3b_write_text(tmp + "/assets/readme.txt", "ignore")
        _m3b_write_text(tmp + "/.git/ignored.flmat", "{}")
        _m3b_write_text(tmp + "/build/ignored.flscn", "{}")

        var s = FlintStudioState(
            headless=True,
            render_tier=RENDER_TIER_0_HEADLESS_SIMD,
            width=1600,
            height=900,
            worker_threads=0,
        )
        var n_found = s.scan_asset_browser(tmp)
        assert_equal(n_found, 28)

        # Spawn one entity so Inspector scroll view + text_input are also active alongside 28+ assets
        var probe_e = s.spawn_entity("InspectMe", position=Vec3(0.0, 1.0, 0.0), record_undo=False)
        s.select_entity(probe_e)

        # Render 3 frames at 1600x900 (covers Frame 1 cold layout + Frame 2/3 cached_rects)
        var inp = InputState()
        for _ in range(3):
            inp.begin_frame()
            inp.move_mouse(10.0, 10.0)
            s.render_editor_frame(1600, 900, input=inp)

        var ab_info = s.dock.find_panel("AssetBrowser")
        assert_true(ab_info[0] >= 0 and ab_info[2])
        var ab_rect = ab_info[3]

        var root_seed = UInt64(0xCBF29CE484222325)
        var ab_dock_id = hash_ui_id(root_seed, "dock:AssetBrowser")
        var ab_cards_id = hash_ui_id(ab_dock_id, "ab_spawn_cards")
        var ab_scroll_id = hash_ui_id(ab_dock_id, "ab_entries_scroll")

        var scr_opt = s.ui_ctx.cached_rects.get(ab_scroll_id)
        assert_true(Bool(scr_opt))
        var scr = scr_opt.value()
        assert_true(scr.h >= Float32(40.0))
        assert_true(scr.y >= ab_rect.y and scr.bottom() <= ab_rect.bottom())

        # Also verify Inspector's insp_scroll & insp_name did not collapse
        var insp_dock_id = hash_ui_id(root_seed, "dock:Inspector")
        var insp_scroll_id = hash_ui_id(insp_dock_id, "insp_scroll")
        var insp_scr_opt = s.ui_ctx.cached_rects.get(insp_scroll_id)
        assert_true(Bool(insp_scr_opt))
        assert_true(insp_scr_opt.value().h >= Float32(100.0))
        var insp_name_id = hash_ui_id(insp_scroll_id, "insp_name")
        var insp_name_opt = s.ui_ctx.cached_rects.get(insp_name_id)
        assert_true(Bool(insp_name_opt))
        assert_true(insp_name_opt.value().h >= Float32(20.0))

        var card_ids = List[String]()
        card_ids.append("card_empty")
        card_ids.append("card_cube")
        card_ids.append("card_point_light")
        card_ids.append("card_spot_light")
        card_ids.append("card_camera")
        card_ids.append("card_audio_emitter")

        var expected_prefixes = List[String]()
        expected_prefixes.append("Empty_")
        expected_prefixes.append("Cube_")
        expected_prefixes.append("PointLight_")
        expected_prefixes.append("SpotLight_")
        expected_prefixes.append("Camera_")
        expected_prefixes.append("AudioEmitter_")

        for ci in range(len(card_ids)):
            var cid = hash_ui_id(ab_cards_id, card_ids[ci])
            var cr_opt = s.ui_ctx.cached_rects.get(cid)
            assert_true(Bool(cr_opt))
            var cr = cr_opt.value()
            assert_true(cr.h >= Float32(20.0))
            assert_true(cr.x >= ab_rect.x and cr.right() <= ab_rect.right())
            assert_true(cr.y >= ab_rect.y and cr.bottom() <= ab_rect.bottom())

            var before_cnt = s.live_entity_count()
            var click_x = cr.x + cr.w * Float32(0.5)
            var click_y = cr.y + Float32(12.0)

            inp.begin_frame()
            inp.move_mouse(click_x, click_y)
            inp.set_mouse(0, True)
            s.render_editor_frame(1600, 900, input=inp)

            inp.begin_frame()
            inp.move_mouse(click_x, click_y)
            inp.set_mouse(0, False)
            s.render_editor_frame(1600, 900, input=inp)

            assert_equal(s.live_entity_count(), before_cnt + 1)
            var sid = s.selected_entity_id
            assert_true(sid >= 0 and s.is_alive_id(sid))
            assert_true(s.entities[sid].name.startswith(expected_prefixes[ci]))
            var se = s.resolve_entity(sid)
            if ci == 0:
                assert_false(s.app.world.has[MeshRendererComponent](se))
            elif ci == 1:
                assert_true(s.app.world.has[MeshRendererComponent](se))
            elif ci == 2:
                assert_true(s.app.world.has[PointLightComponent](se))
            elif ci == 3:
                assert_true(s.app.world.has[SpotLightComponent](se))
            elif ci == 4:
                assert_true(s.app.world.has[CameraComponent](se))
            elif ci == 5:
                assert_true(s.app.world.has[AudioEmitterComponent](se))

        s.shutdown()
    finally:
        _m3b_rm_rf(tmp)



def test_m3b_f14_f15_plugins_navbake_overlays_spotlight_and_scenes_root() raises:
    var tmp = "/tmp/flint_test_m3b_f15"
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp + "/scenes")
    try:
        var s = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=640, height=480, worker_threads=0, project_root=tmp)
        assert_true(s.scene_file_path == tmp + "/scenes/studio_scene.flscn")
        _ = make_dirs(tmp + "/assets")
        _m3b_write_text(tmp + "/assets/main.flscn", '{"schema":2,"entities":[]}')
        assert_equal(_resolve_default_scene_path(tmp), tmp + "/assets/main.flscn")

        # Plugins & codecs installed
        assert_true(s.registry.find_by_name("AudioEmitterComponent") >= 0)
        assert_true(s.registry.find_by_name("NavMeshComponent") >= 0)
        assert_true(s.app.resources.has[AudioServer]())
        assert_true(s.app.resources.has[NavPluginStateResource]())

        # NavMesh bake + Collider overlay (including BODY_SHAPE_CONVEX_HULL and BODY_SHAPE_SPHERE)
        _ = s.spawn_entity("Floor", position=Vec3(0.0, 0.0, 0.0), collider_kind=2, collider_half_extents=Vec3(8.0, 0.25, 8.0), record_undo=False)
        var hull_pts = List[Vec3]()
        hull_pts.append(Vec3(-1.0, 0.0, -1.0))
        hull_pts.append(Vec3(1.0, 0.0, -1.0))
        hull_pts.append(Vec3(1.0, 0.0, 1.0))
        hull_pts.append(Vec3(-1.0, 0.0, 1.0))
        hull_pts.append(Vec3(0.0, 2.0, 0.0))
        var hull_e = s.app.world.spawn()
        _ = s.app.world.insert[LocalTransform](hull_e, LocalTransform(Vec3(2.0, 0.25, 0.0)))
        _ = s.app.world.insert[GlobalTransform](hull_e, GlobalTransform())
        _ = s.app.world.insert[ColliderComponent](hull_e, ColliderComponent.convex_hull(hull_pts))
        var sph_e = s.app.world.spawn()
        _ = s.app.world.insert[LocalTransform](sph_e, LocalTransform(Vec3(-2.5, 0.5, 0.0)))
        _ = s.app.world.insert[GlobalTransform](sph_e, GlobalTransform())
        _ = s.app.world.insert[ColliderComponent](sph_e, ColliderComponent.sphere(0.6))
        var polys = s.bake_navmesh()
        assert_true(polys > 0)
        s.show_collider_overlay = True
        s.render_editor_frame(640, 480)

        # SpotLight fallback yield: disabled keeps 2 fallback lights, enabled yields to 0
        var spt = s.spawn_card_entity(3)
        var sp_ptr = s.app.world.get_ptr_mut[SpotLightComponent](s.resolve_entity(spt))
        sp_ptr.value()[unsafe_offset=0].enabled = False
        s._sync_studio_pre_render(640, 480)
        assert_equal(len(s.app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0].extra_lights), 2)
        sp_ptr.value()[unsafe_offset=0].enabled = True
        s._sync_studio_pre_render(640, 480)
        assert_equal(len(s.app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0].extra_lights), 0)
        s.shutdown()
    finally:
        _m3b_rm_rf(tmp)



def test_m3b_iter3_scroll_clip_cull_and_empty_regressions() raises:
    var root_seed = UInt64(0xCBF29CE484222325)

    # (a) Scroll ab_entries_scroll down 2 notches (scroll_y=72) at 1600x900 and click card_empty
    #     -> MUST spawn Empty_* (NOT clipped ab_item_2!)
    var tmp = "/tmp/flint_test_m3b_iter3_clip"
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp + "/assets/mats")
    try:
        for i in range(12):
            var idx_s = ("0" + String(i)) if i < 10 else String(i)
            _m3b_write_text(tmp + "/assets/mats/asset_" + idx_s + ".flmat", "{}")

        var s = FlintStudioState(
            headless=True,
            render_tier=RENDER_TIER_0_HEADLESS_SIMD,
            width=1600,
            height=900,
            worker_threads=0,
        )
        assert_equal(s.scan_asset_browser(tmp), 12)
        var inp = InputState()
        for _ in range(2):
            inp.begin_frame()
            inp.move_mouse(10.0, 10.0)
            s.render_editor_frame(1600, 900, input=inp)

        var ab_dock_id = hash_ui_id(root_seed, "dock:AssetBrowser")
        var ab_cards_id = hash_ui_id(ab_dock_id, "ab_spawn_cards")
        var ab_scroll_id = hash_ui_id(ab_dock_id, "ab_entries_scroll")
        var card_empty_id = hash_ui_id(ab_cards_id, "card_empty")

        var scr_opt = s.ui_ctx.cached_rects.get(ab_scroll_id)
        var cr_opt = s.ui_ctx.cached_rects.get(card_empty_id)
        assert_true(Bool(scr_opt) and Bool(cr_opt))
        var scr = scr_opt.value()
        var cr = cr_opt.value()

        # Scroll down 2 notches (2 * 36 = 72 px)
        for _ in range(2):
            inp.begin_frame()
            inp.move_mouse(scr.x + Float32(20.0), scr.y + Float32(20.0))
            inp.wheel = Float32(-1.0)
            s.render_editor_frame(1600, 900, input=inp)

        var st_ab = s.ui_ctx.state_map.get(ab_scroll_id)
        assert_true(Bool(st_ab))
        assert_true(abs(st_ab.value().scroll_y - Float32(72.0)) < Float32(0.5))

        # Click card_empty (overlaps clipped ab_item_2 vertically!)
        var cx = cr.x + cr.w * Float32(0.5)
        var cy = cr.y + Float32(12.0)
        inp.begin_frame()
        inp.move_mouse(cx, cy)
        inp.set_mouse(0, True)
        s.render_editor_frame(1600, 900, input=inp)

        inp.begin_frame()
        inp.move_mouse(cx, cy)
        inp.set_mouse(0, False)
        s.render_editor_frame(1600, 900, input=inp)

        assert_equal(s.live_entity_count(), 1)
        assert_true(s.selected_entity_id >= 0 and s.is_alive_id(s.selected_entity_id))
        assert_true(s.entities[s.selected_entity_id].name.startswith("Empty_"))
        s.shutdown()
    finally:
        _m3b_rm_rf(tmp)


    # (b) Scroll insp_scroll down 2 notches at 1280x720, click Toolbar over clipped insp_name
    #     -> wants_keyboard() MUST remain False!
    var s_b = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=1280,
        height=720,
        worker_threads=0,
    )
    var eid_b = s_b.spawn_entity("ClipNameEnt", position=Vec3(0.0, 1.0, 0.0), collider_kind=2, record_undo=False)
    assert_true(s_b.configure_entity_npr_cel(eid_b, cel_outline_width_px=2.0))
    s_b.select_entity(eid_b)
    var inp_b = InputState()
    for _ in range(2):
        inp_b.begin_frame()
        inp_b.move_mouse(10.0, 10.0)
        s_b.render_editor_frame(1280, 720, input=inp_b)

    var insp_dock_id = hash_ui_id(root_seed, "dock:Inspector")
    var insp_scroll_id = hash_ui_id(insp_dock_id, "insp_scroll")
    var insp_name_id = hash_ui_id(insp_scroll_id, "insp_name")
    var insp_scr_b = s_b.ui_ctx.cached_rects[insp_scroll_id]

    for _ in range(2):
        inp_b.begin_frame()
        inp_b.move_mouse(insp_scr_b.x + Float32(20.0), insp_scr_b.y + Float32(40.0))
        inp_b.wheel = Float32(-1.0)
        s_b.render_editor_frame(1280, 720, input=inp_b)

    var name_r = s_b.ui_ctx.cached_rects[insp_name_id]
    assert_true(name_r.y < insp_scr_b.y)
    var click_tb_x = name_r.x + Float32(30.0)
    var click_tb_y = max(Float32(4.0), min(insp_scr_b.y - Float32(4.0), name_r.y + Float32(10.0)))
    inp_b.begin_frame()
    inp_b.move_mouse(click_tb_x, click_tb_y)
    inp_b.set_mouse(0, True)
    s_b.render_editor_frame(1280, 720, input=inp_b)

    inp_b.begin_frame()
    inp_b.move_mouse(click_tb_x, click_tb_y)
    inp_b.set_mouse(0, False)
    s_b.render_editor_frame(1280, 720, input=inp_b)

    assert_false(s_b.ui_ctx.wants_keyboard())
    s_b.shutdown()

    # (c) Scroll insp_scroll to ~99.8 (culling Parent: Root label) at 1600x900,
    #     click cached_rects[shd_vfx] -> sets SHADING_STYLIZED_VFX without collapsing insp_npr_on!
    var s_c = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=1600,
        height=900,
        worker_threads=0,
    )
    var eid_c = s_c.spawn_entity("CullFlexEnt", position=Vec3(0.0, 1.0, 0.0), collider_kind=2, record_undo=False)
    assert_true(s_c.configure_entity_npr_cel(eid_c, cel_outline_width_px=2.0))
    s_c.select_entity(eid_c)
    var inp_c = InputState()
    for _ in range(2):
        inp_c.begin_frame()
        inp_c.move_mouse(10.0, 10.0)
        s_c.render_editor_frame(1600, 900, input=inp_c)

    var insp_scr_c = s_c.ui_ctx.cached_rects[insp_scroll_id]
    # Scroll by 2 notches (72.0) + ~0.7722 notch (27.8) = 99.8 px to cull Parent: Root
    for _ in range(2):
        inp_c.begin_frame()
        inp_c.move_mouse(insp_scr_c.x + Float32(20.0), insp_scr_c.y + Float32(40.0))
        inp_c.wheel = Float32(-1.0)
        s_c.render_editor_frame(1600, 900, input=inp_c)
    inp_c.begin_frame()
    inp_c.move_mouse(insp_scr_c.x + Float32(20.0), insp_scr_c.y + Float32(40.0))
    inp_c.wheel = Float32(-27.8 / 36.0)
    s_c.render_editor_frame(1600, 900, input=inp_c)

    assert_true(s_c.ui_ctx.culled_items_this_frame > 0)
    var insp_npr_on_id = hash_ui_id(insp_scroll_id, "insp_npr_on")
    var npr_row_id = hash_ui_id(insp_npr_on_id, "npr_mode_row")
    var shd_vfx_id = hash_ui_id(npr_row_id, "shd_vfx")
    var vfx_opt = s_c.ui_ctx.cached_rects.get(shd_vfx_id)
    assert_true(Bool(vfx_opt))
    var vfx_r = vfx_opt.value()
    assert_true(vfx_r.y >= insp_scr_c.y and vfx_r.bottom() <= insp_scr_c.bottom())

    var vx = vfx_r.x + vfx_r.w * Float32(0.5)
    var vy = vfx_r.y + vfx_r.h * Float32(0.5)
    inp_c.begin_frame()
    inp_c.move_mouse(vx, vy)
    inp_c.set_mouse(0, True)
    s_c.render_editor_frame(1600, 900, input=inp_c)

    inp_c.begin_frame()
    inp_c.move_mouse(vx, vy)
    inp_c.set_mouse(0, False)
    s_c.render_editor_frame(1600, 900, input=inp_c)

    assert_equal(s_c.entities[eid_c].shading_mode, SHADING_STYLIZED_VFX)
    var st_npr = s_c.ui_ctx.state_map.get(insp_npr_on_id)
    assert_true(Bool(st_npr) and st_npr.value().tree_open)
    s_c.shutdown()

    # (d) Empty begin_scroll_view("sv_zero", 240.0, 0.0) & h=4.0 -> scroll_max_y == 0.0
    #     + culled tree_node/label FlexTree alignment + slider drag continuation outside clip
    var ui_d = UiContext()
    var inp_d = InputState()
    inp_d.begin_frame()
    ui_d.begin_frame(inp_d, 800.0, 600.0)
    ui_d.begin_scroll_view("sv_zero", Float32(240.0), Float32(0.0))
    ui_d.end_scroll_view()
    ui_d.begin_scroll_view("sv_tiny", Float32(240.0), Float32(4.0))
    ui_d.end_scroll_view()
    ui_d.end_frame()

    var sv_zero_id = ui_d.widget_id("sv_zero")
    var sv_tiny_id = ui_d.widget_id("sv_tiny")
    assert_equal(ui_d.state_map[sv_zero_id].scroll_max_y, Float32(0.0))
    assert_equal(ui_d.state_map[sv_tiny_id].scroll_max_y, Float32(0.0))
    assert_equal(len(ui_d.draw_list.prims), 2)  # 1 rect per empty scroll view, 0 phantom scrollbar prims

    # Culled label + tree_node keeps cached_rects synced and slider drag continues outside clip
    var ui_e = UiContext()
    var inp_e = InputState()
    var sl_v = Float32(0.5)
    var btn_after_clicked = False
    for fr in range(3):
        inp_e.begin_frame()
        if fr == 2:
            inp_e.move_mouse(20.0, 20.0)
            inp_e.wheel = Float32(-3.0)  # scroll 108 px -> culls L0..L2 + tn_top
        ui_e.begin_frame(inp_e, 800.0, 600.0)
        ui_e.begin_scroll_view("sv_cull", Float32(260.0), Float32(90.0))
        ui_e.label("L0")
        ui_e.label("L1")
        ui_e.label("L2")
        if ui_e.tree_node("tn_top", "TopNode", default_open=True):
            ui_e.label("InsideTree")
            ui_e.tree_pop()
        _ = ui_e.slider_float("sl_cull", "Val", sl_v, Float32(0.0), Float32(1.0), w=180.0)
        if ui_e.button("btn_after_cull", "Target", Float32(100.0), Float32(24.0)):
            btn_after_clicked = True
        ui_e.label("L_bot0")
        ui_e.label("L_bot1")
        ui_e.end_scroll_view()
        ui_e.end_frame()

    assert_true(ui_e.culled_items_this_frame >= 4)
    var sv_cull_id = ui_e.widget_id("sv_cull")
    var btn_r = ui_e.cached_rects[hash_ui_id(sv_cull_id, "btn_after_cull")]
    inp_e.begin_frame()
    inp_e.move_mouse(btn_r.x + Float32(20.0), btn_r.y + Float32(12.0))
    inp_e.set_mouse(0, True)
    ui_e.begin_frame(inp_e, 800.0, 600.0)
    ui_e.begin_scroll_view("sv_cull", Float32(260.0), Float32(90.0))
    ui_e.label("L0")
    ui_e.label("L1")
    ui_e.label("L2")
    if ui_e.tree_node("tn_top", "TopNode", default_open=True):
        ui_e.label("InsideTree")
        ui_e.tree_pop()
    _ = ui_e.slider_float("sl_cull", "Val", sl_v, Float32(0.0), Float32(1.0), w=180.0)
    if ui_e.button("btn_after_cull", "Target", Float32(100.0), Float32(24.0)):
        btn_after_clicked = True
    ui_e.label("L_bot0")
    ui_e.label("L_bot1")
    ui_e.end_scroll_view()
    ui_e.end_frame()

    inp_e.begin_frame()
    inp_e.move_mouse(btn_r.x + Float32(20.0), btn_r.y + Float32(12.0))
    inp_e.set_mouse(0, False)
    ui_e.begin_frame(inp_e, 800.0, 600.0)
    ui_e.begin_scroll_view("sv_cull", Float32(260.0), Float32(90.0))
    ui_e.label("L0")
    ui_e.label("L1")
    ui_e.label("L2")
    if ui_e.tree_node("tn_top", "TopNode", default_open=True):
        ui_e.label("InsideTree")
        ui_e.tree_pop()
    _ = ui_e.slider_float("sl_cull", "Val", sl_v, Float32(0.0), Float32(1.0), w=180.0)
    if ui_e.button("btn_after_cull", "Target", Float32(100.0), Float32(24.0)):
        btn_after_clicked = True
    ui_e.label("L_bot0")
    ui_e.label("L_bot1")
    ui_e.end_scroll_view()
    ui_e.end_frame()
    assert_true(btn_after_clicked)

    # Slider drag continuation outside scroll clip
    var sl_r = ui_e.cached_rects[hash_ui_id(sv_cull_id, "sl_cull")]
    inp_e.begin_frame()
    inp_e.move_mouse(sl_r.x + Float32(10.0), sl_r.y + Float32(10.0))
    inp_e.set_mouse(0, True)
    ui_e.begin_frame(inp_e, 800.0, 600.0)
    ui_e.begin_scroll_view("sv_cull", Float32(260.0), Float32(90.0))
    ui_e.label("L0")
    ui_e.label("L1")
    ui_e.label("L2")
    if ui_e.tree_node("tn_top", "TopNode", default_open=True):
        ui_e.label("InsideTree")
        ui_e.tree_pop()
    _ = ui_e.slider_float("sl_cull", "Val", sl_v, Float32(0.0), Float32(1.0), w=180.0)
    _ = ui_e.button("btn_after_cull", "Target", Float32(100.0), Float32(24.0))
    ui_e.label("L_bot0")
    ui_e.label("L_bot1")
    ui_e.end_scroll_view()
    ui_e.end_frame()

    inp_e.begin_frame()
    inp_e.move_mouse(Float32(500.0), Float32(500.0))  # way outside scroll clip!
    ui_e.begin_frame(inp_e, 800.0, 600.0)
    ui_e.begin_scroll_view("sv_cull", Float32(260.0), Float32(90.0))
    ui_e.label("L0")
    ui_e.label("L1")
    ui_e.label("L2")
    if ui_e.tree_node("tn_top", "TopNode", default_open=True):
        ui_e.label("InsideTree")
        ui_e.tree_pop()
    var sl_dragged = ui_e.slider_float("sl_cull", "Val", sl_v, Float32(0.0), Float32(1.0), w=180.0)
    _ = ui_e.button("btn_after_cull", "Target", Float32(100.0), Float32(24.0))
    ui_e.label("L_bot0")
    ui_e.label("L_bot1")
    ui_e.end_scroll_view()
    ui_e.end_frame()
    assert_true(sl_dragged)
    assert_true(abs(sl_v - Float32(1.0)) < Float32(1.0e-4))


def _rect_inside(inner: RectF, outer: RectF) -> Bool:
    return (
        inner.w > Float32(0.0)
        and inner.h > Float32(0.0)
        and inner.x >= outer.x - Float32(0.01)
        and inner.y >= outer.y - Float32(0.01)
        and inner.right() <= outer.right() + Float32(0.01)
        and inner.bottom() <= outer.bottom() + Float32(0.01)
    )


def test_m3b_iter4_modal_frame2_layout_and_clicks() raises:
    var root_seed = UInt64(0xCBF29CE484222325)
    var modal_id = hash_ui_id(root_seed, "modal:unsaved_confirm_modal")
    var btns_id = hash_ui_id(modal_id, "unsaved_modal_btns")
    var save_id = hash_ui_id(btns_id, "modal_save_cont")
    var discard_id = hash_ui_id(btns_id, "modal_discard")
    var cancel_id = hash_ui_id(btns_id, "modal_cancel")

    var tmp = "/tmp/flint_test_m3b_iter4_modal"
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp)
    try:
        var s = FlintStudioState(
            headless=True,
            render_tier=RENDER_TIER_0_HEADLESS_SIMD,
            width=1280,
            height=720,
            worker_threads=0,
        )
        s.scene_file_path = tmp + "/saved_modal.flscn"
        _ = s.spawn_entity("ModalProp", position=Vec3(1.0, 0.0, 0.0), record_undo=False)
        s.mark_dirty(True)
        assert_true(s.request_quit())
        assert_true(s.confirm_modal_open)
        assert_equal(s.confirm_modal_action, 1)

        var mrect_720 = RectF(445.0, 291.0, 390.0, 138.0)
        var body_720 = RectF(457.0, 329.0, 366.0, 90.0)
        var inp = InputState()

        # Frame 1 & Frame 2 at 1280x720: cached_rects must stay strictly inside mrect/body
        for _ in range(2):
            inp.begin_frame()
            inp.move_mouse(10.0, 10.0)
            s.render_editor_frame(1280, 720, input=inp)
            assert_true(s.ui_ctx.wants_keyboard())
            assert_true(s.ui_ctx.wants_mouse())
            assert_true(_rect_inside(s.ui_ctx.cached_rects[btns_id], body_720))
            assert_true(_rect_inside(s.ui_ctx.cached_rects[save_id], mrect_720))
            assert_true(_rect_inside(s.ui_ctx.cached_rects[save_id], body_720))
            assert_true(_rect_inside(s.ui_ctx.cached_rects[discard_id], mrect_720))
            assert_true(_rect_inside(s.ui_ctx.cached_rects[discard_id], body_720))
            assert_true(_rect_inside(s.ui_ctx.cached_rects[cancel_id], mrect_720))
            assert_true(_rect_inside(s.ui_ctx.cached_rects[cancel_id], body_720))

        # Click exact centered Discard button coordinate (643.0, 372.0) on Frame 2+
        assert_true(s.ui_ctx.cached_rects[discard_id].contains(Float32(643.0), Float32(372.0)))
        inp.begin_frame()
        inp.move_mouse(643.0, 372.0)
        inp.set_mouse(0, True)
        s.render_editor_frame(1280, 720, input=inp)

        inp.begin_frame()
        inp.move_mouse(643.0, 372.0)
        inp.set_mouse(0, False)
        s.render_editor_frame(1280, 720, input=inp)

        assert_true(s.quit_requested)
        assert_false(s.confirm_modal_open)
        assert_equal(s.confirm_modal_action, 0)

        # Re-open modal at 1280x720: outside click is blocked; Cancel on Frame 2+ closes without quitting
        s.quit_requested = False
        s.mark_dirty(True)
        assert_true(s.request_quit())
        for _ in range(2):
            inp.begin_frame()
            inp.move_mouse(10.0, 10.0)
            s.render_editor_frame(1280, 720, input=inp)

        # Click outside centered modal over top-left Toolbar (20, 20) -> must NOT trigger Cancel/Discard
        inp.begin_frame()
        inp.move_mouse(20.0, 20.0)
        inp.set_mouse(0, True)
        s.render_editor_frame(1280, 720, input=inp)
        inp.begin_frame()
        inp.move_mouse(20.0, 20.0)
        inp.set_mouse(0, False)
        s.render_editor_frame(1280, 720, input=inp)
        assert_true(s.confirm_modal_open)
        assert_false(s.quit_requested)

        # Click Cancel on Frame 2+
        var can_r = s.ui_ctx.cached_rects[cancel_id]
        assert_true(_rect_inside(can_r, body_720))
        inp.begin_frame()
        inp.move_mouse(can_r.center_x(), can_r.center_y())
        inp.set_mouse(0, True)
        s.render_editor_frame(1280, 720, input=inp)
        inp.begin_frame()
        inp.move_mouse(can_r.center_x(), can_r.center_y())
        inp.set_mouse(0, False)
        s.render_editor_frame(1280, 720, input=inp)
        assert_false(s.confirm_modal_open)
        assert_equal(s.confirm_modal_action, 0)
        assert_false(s.quit_requested)

        # Resize to 1600x900 and open modal: Frame 1 & Frame 2+ stay centered in 1600x900 modal
        # and Save & Continue saves the scene, clears dirty, closes modal, and sets quit_requested
        assert_true(s.request_quit())
        var mrect_900 = RectF(605.0, 381.0, 390.0, 138.0)
        var body_900 = RectF(617.0, 419.0, 366.0, 90.0)
        for _ in range(2):
            inp.begin_frame()
            inp.move_mouse(10.0, 10.0)
            s.render_editor_frame(1600, 900, input=inp)
            assert_true(_rect_inside(s.ui_ctx.cached_rects[btns_id], body_900))
            assert_true(_rect_inside(s.ui_ctx.cached_rects[save_id], mrect_900))
            assert_true(_rect_inside(s.ui_ctx.cached_rects[save_id], body_900))
            assert_true(_rect_inside(s.ui_ctx.cached_rects[discard_id], body_900))
            assert_true(_rect_inside(s.ui_ctx.cached_rects[cancel_id], body_900))

        var save_r = s.ui_ctx.cached_rects[save_id]
        inp.begin_frame()
        inp.move_mouse(save_r.center_x(), save_r.center_y())
        inp.set_mouse(0, True)
        s.render_editor_frame(1600, 900, input=inp)
        inp.begin_frame()
        inp.move_mouse(save_r.center_x(), save_r.center_y())
        inp.set_mouse(0, False)
        s.render_editor_frame(1600, 900, input=inp)

        assert_false(s.confirm_modal_open)
        assert_false(s.dirty)
        assert_true(s.quit_requested)
        assert_true(len(read_bytes(tmp + "/saved_modal.flscn")) > 0)
        s.shutdown()
    finally:
        _m3b_rm_rf(tmp)


    # Standalone UiContext nested begin_box row inside begin_modal across Frames 1..3
    var ui = UiContext()
    var inp_u = InputState()
    var m_open = True
    var bg_clicked = False
    var nb_clicked = False
    var modal_w = Float32(420.0)
    var modal_h = Float32(160.0)
    var body_u = RectF(
        (Float32(1280.0) - modal_w) * Float32(0.5) + Float32(12.0),
        (Float32(720.0) - modal_h) * Float32(0.5) + Float32(28.0) + Float32(10.0),
        modal_w - Float32(24.0),
        modal_h - Float32(28.0) - Float32(20.0),
    )

    for fr in range(3):
        inp_u.begin_frame()
        inp_u.move_mouse(20.0, 20.0)
        if fr == 1:
            inp_u.set_mouse(0, True)  # Try pressing bg_btn while modal is open
        elif fr == 2:
            inp_u.set_mouse(0, False)  # Try releasing bg_btn while modal is open
        ui.begin_frame(inp_u, 1280.0, 720.0)
        if ui.button("bg_btn", "BG", 100.0, 26.0):
            bg_clicked = True
        if ui.begin_modal("nested_modal", "Nested Modal", modal_w, modal_h, m_open):
            ui.label("Prompt")
            ui.begin_box("nested_row", FlexStyle.row(gap=10.0, pad=2.0))
            _ = ui.button("nbtn_a", "Alpha", 90.0, 26.0)
            if ui.button("nbtn_b", "Beta", 90.0, 26.0):
                nb_clicked = True
            ui.end_box()
            ui.end_modal()
        ui.end_frame()

    assert_false(bg_clicked)
    var n_modal_id = hash_ui_id(root_seed, "modal:nested_modal")
    var n_row_id = hash_ui_id(n_modal_id, "nested_row")
    var n_a_id = hash_ui_id(n_row_id, "nbtn_a")
    var n_b_id = hash_ui_id(n_row_id, "nbtn_b")
    assert_true(_rect_inside(ui.cached_rects[n_row_id], body_u))
    assert_true(_rect_inside(ui.cached_rects[n_a_id], body_u))
    assert_true(_rect_inside(ui.cached_rects[n_b_id], body_u))

    var nb_r = ui.cached_rects[n_b_id]
    for step in range(2):
        inp_u.begin_frame()
        inp_u.move_mouse(nb_r.center_x(), nb_r.center_y())
        inp_u.set_mouse(0, step == 0)
        ui.begin_frame(inp_u, 1280.0, 720.0)
        if ui.button("bg_btn", "BG", 100.0, 26.0):
            bg_clicked = True
        if ui.begin_modal("nested_modal", "Nested Modal", modal_w, modal_h, m_open):
            ui.label("Prompt")
            ui.begin_box("nested_row", FlexStyle.row(gap=10.0, pad=2.0))
            _ = ui.button("nbtn_a", "Alpha", 90.0, 26.0)
            if ui.button("nbtn_b", "Beta", 90.0, 26.0):
                nb_clicked = True
            ui.end_box()
            ui.end_modal()
        ui.end_frame()
    assert_true(nb_clicked)
    assert_false(bg_clicked)


def test_m4a_state_hash_unknown_components_and_bone_socket_slots() raises:
    # 1. FlintStudioState.state_hash() sensitivity to UnknownComponentsComponent & PIE parity
    var scn_a = (
        '{"format": "flscn", "version": 2, "name": "UnkHashA", "entities": ['
        '{"slot": 0, "parent": -1, "components": {'
        '"LocalTransform": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]}, '
        '"ThirdParty": {"hp": 100}'
        '}}]}'
    )
    var scn_b = (
        '{"format": "flscn", "version": 2, "name": "UnkHashA", "entities": ['
        '{"slot": 0, "parent": -1, "components": {'
        '"LocalTransform": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]}, '
        '"ThirdParty": {"hp": 999}'
        '}}]}'
    )
    var ba = List[UInt8]()
    for c in scn_a.as_bytes():
        ba.append(c)
    var bb = List[UInt8]()
    for c in scn_b.as_bytes():
        bb.append(c)

    var s1 = FlintStudioState(headless=True)
    var s2 = FlintStudioState(headless=True)
    _ = s1.scene.load_flscn_bytes(s1.app.world, s1.registry, ba, clear_existing=True)
    _ = s2.scene.load_flscn_bytes(s2.app.world, s2.registry, bb, clear_existing=True)
    var h1 = s1.state_hash()
    var h2 = s2.state_hash()
    assert_true(h1 != h2)

    # PIE enter/exit preserves UnknownComponentsComponent and state_hash()
    assert_true(s1.pie_play())
    _ = s1.step_simulation(1, Float32(1.0 / 60.0))
    assert_true(s1.pie_stop())
    assert_equal(s1.state_hash(), h1)

    # 2. state_hash() hashes BoneSocketComponent using canonical entity slots (not raw Entity bits)
    var s_slot1 = FlintStudioState(headless=True)
    var s_slot2 = FlintStudioState(headless=True)
    s_slot1.registry.ensure_registered_in_world(s_slot1.app.world)
    s_slot2.registry.ensure_registered_in_world(s_slot2.app.world)

    # Shift entity index/generation in s_slot2 without changing live slot order
    var dummy = s_slot2.app.world.spawn_empty()
    _ = s_slot2.app.world.despawn(dummy)

    var sk1 = s_slot1.app.world.spawn_empty()
    _ = s_slot1.app.world.insert[LocalTransform](sk1, LocalTransform.identity())
    _ = s_slot1.app.world.insert[SkeletonComponent](sk1, SkeletonComponent(42))
    var sock1 = s_slot1.app.world.spawn_empty()
    _ = s_slot1.app.world.insert[LocalTransform](sock1, LocalTransform.identity())
    _ = s_slot1.app.world.insert[BoneSocketComponent](
        sock1, BoneSocketComponent(sk1, BoneSocketAttachment("Hand", 1), -1)
    )

    var sk2 = s_slot2.app.world.spawn_empty()
    _ = s_slot2.app.world.insert[LocalTransform](sk2, LocalTransform.identity())
    _ = s_slot2.app.world.insert[SkeletonComponent](sk2, SkeletonComponent(42))
    var sock2 = s_slot2.app.world.spawn_empty()
    _ = s_slot2.app.world.insert[LocalTransform](sock2, LocalTransform.identity())
    _ = s_slot2.app.world.insert[BoneSocketComponent](
        sock2, BoneSocketComponent(sk2, BoneSocketAttachment("Hand", 1), -1)
    )
    assert_true(sk1.to_bits() != sk2.to_bits())
    assert_equal(s_slot1.state_hash(), s_slot2.state_hash())

    s1.shutdown()
    s2.shutdown()
    s_slot1.shutdown()
    s_slot2.shutdown()


def _m4b_push_u32_le(mut out: List[UInt8], v: UInt32):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))


def _m4b_push_f32_le(mut out: List[UInt8], v: Float32):
    _m4b_push_u32_le(out, bitcast[DType.uint32](v))


def _m4b_make_rigged_glb() -> List[UInt8]:
    var bin_data = List[UInt8]()
    var pos_off = len(bin_data)
    _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0)
    _m4b_push_f32_le(bin_data, 1.0); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0)
    _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 1.0); _m4b_push_f32_le(bin_data, 0.0)
    var pos_len = len(bin_data) - pos_off

    var norm_off = len(bin_data)
    for _ in range(3):
        _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 1.0)
    var norm_len = len(bin_data) - norm_off

    var jnt_off = len(bin_data)
    for _ in range(3):
        bin_data.append(0); bin_data.append(1); bin_data.append(0); bin_data.append(0)
    var jnt_len = len(bin_data) - jnt_off

    var wgt_off = len(bin_data)
    for _ in range(3):
        _m4b_push_f32_le(bin_data, 0.75); _m4b_push_f32_le(bin_data, 0.25)
        _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0)
    var wgt_len = len(bin_data) - wgt_off

    var ibm_off = len(bin_data)
    for _ in range(2):
        for r in range(4):
            for c in range(4):
                _m4b_push_f32_le(bin_data, Float32(1.0) if r == c else Float32(0.0))
    var ibm_len = len(bin_data) - ibm_off

    var t_off = len(bin_data)
    _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 1.0)
    var t_len = len(bin_data) - t_off

    var v_off = len(bin_data)
    _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.5); _m4b_push_f32_le(bin_data, 0.0)
    _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.9); _m4b_push_f32_le(bin_data, 0.0)
    var v_len = len(bin_data) - v_off

    while len(bin_data) % 4 != 0:
        bin_data.append(0)

    var json = (
        '{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,2]}],'
        '"buffers":[{"byteLength":' + String(len(bin_data)) + '}],'
        '"bufferViews":['
        '{"buffer":0,"byteOffset":' + String(pos_off) + ',"byteLength":' + String(pos_len) + '},'
        '{"buffer":0,"byteOffset":' + String(norm_off) + ',"byteLength":' + String(norm_len) + '},'
        '{"buffer":0,"byteOffset":' + String(jnt_off) + ',"byteLength":' + String(jnt_len) + '},'
        '{"buffer":0,"byteOffset":' + String(wgt_off) + ',"byteLength":' + String(wgt_len) + '},'
        '{"buffer":0,"byteOffset":' + String(ibm_off) + ',"byteLength":' + String(ibm_len) + '},'
        '{"buffer":0,"byteOffset":' + String(t_off) + ',"byteLength":' + String(t_len) + '},'
        '{"buffer":0,"byteOffset":' + String(v_off) + ',"byteLength":' + String(v_len) + '}'
        '],"accessors":['
        '{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},'
        '{"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"},'
        '{"bufferView":2,"componentType":5121,"count":3,"type":"VEC4"},'
        '{"bufferView":3,"componentType":5126,"count":3,"type":"VEC4"},'
        '{"bufferView":4,"componentType":5126,"count":2,"type":"MAT4"},'
        '{"bufferView":5,"componentType":5126,"count":2,"type":"SCALAR","min":[0],"max":[1]},'
        '{"bufferView":6,"componentType":5126,"count":2,"type":"VEC3"}'
        '],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"JOINTS_0":2,"WEIGHTS_0":3}}]}],'
        '"nodes":[{"name":"Root","children":[1]},{"name":"Hand","translation":[0,0.5,0]},{"name":"MeshNode","mesh":0,"skin":0}],'
        '"skins":[{"joints":[0,1],"inverseBindMatrices":4}],'
        '"animations":[{"name":"Idle","samplers":[{"input":5,"output":6,"interpolation":"LINEAR"}],'
        '"channels":[{"sampler":0,"target":{"node":1,"path":"translation"}}]}]}'
    )
    var json_bytes = List[UInt8]()
    for b in json.as_bytes():
        json_bytes.append(b)
    while len(json_bytes) % 4 != 0:
        json_bytes.append(0x20)

    var total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    var out = List[UInt8](capacity=total_len)
    _m4b_push_u32_le(out, 0x46546C67)
    _m4b_push_u32_le(out, 2)
    _m4b_push_u32_le(out, UInt32(total_len))
    _m4b_push_u32_le(out, UInt32(len(json_bytes)))
    _m4b_push_u32_le(out, 0x4E4F534A)
    for b in json_bytes:
        out.append(b)
    _m4b_push_u32_le(out, UInt32(len(bin_data)))
    _m4b_push_u32_le(out, 0x004E4942)
    for b in bin_data:
        out.append(b)
    return out^


def _m4b_make_test_wav(n_samples: Int = 128) -> List[UInt8]:
    var data_bytes = n_samples * 2
    var riff_size = 36 + data_bytes
    var out = List[UInt8](capacity=44 + data_bytes)
    for b in "RIFF".as_bytes():
        out.append(b)
    _m4b_push_u32_le(out, UInt32(riff_size))
    for b in "WAVEfmt ".as_bytes():
        out.append(b)
    for b in [UInt8(16), 0, 0, 0, 1, 0, 1, 0]:
        out.append(b)
    _m4b_push_u32_le(out, UInt32(48000))
    _m4b_push_u32_le(out, UInt32(96000))
    for b in [UInt8(2), 0, 16, 0]:
        out.append(b)
    for b in "data".as_bytes():
        out.append(b)
    _m4b_push_u32_le(out, UInt32(data_bytes))
    for i in range(n_samples):
        var v = Int16((i * 250) % 10000)
        var u = UInt16(v)
        out.append(UInt8(u & 0xFF))
        out.append(UInt8((u >> 8) & 0xFF))
    return out^


def test_m4b_edit_mode_idle_preview_invariant_and_pie_parity() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_test_m4b_idle_preview_" + String(pid)
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp + "/assets/models")
    _ = make_dirs(tmp + "/scenes")
    try:
        var glb_b = _m4b_make_rigged_glb()
        assert_true(write_bytes(tmp + "/assets/models/hero.glb", glb_b))

        var s = FlintStudioState(
            headless=True,
            render_tier=RENDER_TIER_0_HEADLESS_SIMD,
            width=160,
            height=120,
            worker_threads=0,
            project_root=tmp,
        )
        s.app.resources.insert(_StageProbeResource())
        _ = s.app.add_system(STAGE_UPDATE, "test.m4b_idle_update_probe", _probe_update, exclusive=True)

        # Spawn rigged entity via activate_asset_entry + BoneSocketComponent child
        var hero_id = s.activate_asset_entry("assets/models/hero.glb")
        assert_true(hero_id >= 0)
        var hero_e = s.resolve_entity(hero_id)
        assert_true(s.app.world.has[SkeletonComponent](hero_e))
        assert_true(s.app.world.has[AnimatorComponent](hero_e))

        var sock_id = s.spawn_entity("SwordSocket", position=Vec3(0.0, 0.0, 0.0), record_undo=False)
        var sock_e = s.resolve_entity(sock_id)
        var att = BoneSocketAttachment("HandSlot", 1, -1, Vec3(0.1, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
        _ = s.app.world.insert[BoneSocketComponent](sock_e, BoneSocketComponent(hero_e, att^, -1))

        # Frame 0 resolves mesh + AnimAssetCache before measuring Edit-mode invariants
        s.render_editor_frame(160, 120)
        assert_true(s.viewport_anim_preview)
        assert_equal(s.pie.mode, PIE_STOPPED)
        var h0 = s.state_hash()
        var upd0 = s.app.resources.get_ptr[_StageProbeResource]().value()[].update_count
        var sync0 = s.app.resources.get_ptr[RenderPluginStateResource]().value()[].skinned_sync_count

        for _ in range(10):
            s.render_editor_frame(160, 120)
            assert_equal(s.app.resources.get_ptr[_StageProbeResource]().value()[].update_count, upd0)
            assert_equal(s.state_hash(), h0)

        var sync1 = s.app.resources.get_ptr[RenderPluginStateResource]().value()[].skinned_sync_count
        assert_true(sync1 >= sync0 + 10)
        assert_true(s.preview_anim_time > Float32(0.0))

        # PIE play resets preview_anim_time, runs STAGE_UPDATE, and pie_stop restores exact state_hash()
        assert_true(s.pie_play())
        assert_true(abs(s.preview_anim_time) < Float32(1.0e-6))
        assert_true(s.step_simulation(2, Float32(1.0 / 60.0)))
        assert_true(s.app.resources.get_ptr[_StageProbeResource]().value()[].update_count >= upd0 + 2)
        assert_true(s.pie_stop())
        assert_equal(s.state_hash(), h0)

        s.shutdown()
    finally:
        _m3b_rm_rf(tmp)



def test_m4b_studio_asset_import_drag_drop_activate_and_inspector_undo() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_test_m4b_studio_assets_" + String(pid)
    _m3b_rm_rf(tmp)
    var proj = tmp + "/proj"
    var ext = tmp + "/ext"
    _ = make_dirs(proj + "/assets")
    _ = make_dirs(proj + "/scenes")
    _ = make_dirs(ext)
    try:
        # Prepare valid external assets of all M1 kinds
        var px_brick = List[UInt32](length=8 * 8, fill=UInt32(0xFF884422))
        assert_true(write_png(ext + "/brick.png", px_brick, 8, 8, stamped=False))

        var mdl = NativeModel()
        var m = NativeMesh()
        m.positions.append(Float32(-0.5)); m.positions.append(Float32(-0.5)); m.positions.append(Float32(0.0))
        m.positions.append(Float32(0.5)); m.positions.append(Float32(-0.5)); m.positions.append(Float32(0.0))
        m.positions.append(Float32(0.0)); m.positions.append(Float32(0.5)); m.positions.append(Float32(0.0))
        m.normals.append(Float32(0.0)); m.normals.append(Float32(0.0)); m.normals.append(Float32(1.0))
        m.normals.append(Float32(0.0)); m.normals.append(Float32(0.0)); m.normals.append(Float32(1.0))
        m.normals.append(Float32(0.0)); m.normals.append(Float32(0.0)); m.normals.append(Float32(1.0))
        m.texcoords.append(Float32(0.0)); m.texcoords.append(Float32(0.0))
        m.texcoords.append(Float32(1.0)); m.texcoords.append(Float32(0.0))
        m.texcoords.append(Float32(0.5)); m.texcoords.append(Float32(1.0))
        m.indices.append(0); m.indices.append(1); m.indices.append(2)
        mdl.meshes.append(m^)
        assert_true(write_native_glb(ext + "/crate.glb", mdl))

        var fmat = FlmatAsset("BrickMat")
        fmat.base_color = Vec3(0.9, 0.35, 0.25)
        fmat.roughness = Float32(0.5)
        fmat.albedo_tex = AssetRef.from_path("textures/brick.png", ASSET_KIND_TEXTURE)
        assert_true(save_flmat_file(ext + "/brick.flmat", fmat))

        var rig = FrigRigDef("HeroRig")
        rig.joints.append(FrigJointDef("Root", -1, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig.joints.append(FrigJointDef("Spine", 0, Vec3(0.0, 0.8, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        assert_true(save_flrig_file(ext + "/hero.flrig", rig))

        var flanim_g = AnimGraphDef()
        flanim_g.name = "HeroGraph"
        var st_idle = StateDef()
        st_idle.name = "Idle"
        st_idle.clip = 0
        flanim_g.states.append(st_idle^)
        var flanim_b = save_flanim_bytes(flanim_g)
        assert_true(write_bytes(ext + "/hero.flanim", flanim_b))
        _m3b_write_text(
            ext + "/crate.flprefab",
            '{"format":"flprefab","version":1,"name":"CratePrefab","entities":['
            '{"name":"CrateRoot","position":[1.0,2.0,-1.0],"mass":4.5,"mesh_ref":"models/crate.glb","material_ref":"materials/brick.flmat"}]}',
        )
        var wav_hit = _m4b_make_test_wav(128)
        assert_true(write_bytes(ext + "/hit.wav", wav_hit))
        _m3b_write_text(ext + "/bad.glb", "NOT_A_GLB")

        var s = FlintStudioState(
            headless=True,
            render_tier=RENDER_TIER_0_HEADLESS_SIMD,
            width=320,
            height=240,
            worker_threads=0,
            project_root=proj,
        )

        # 1. import_asset_to_project rejects traversal/system/corrupt files, accepts valid assets
        var r_bad1 = s.import_asset_to_project("../outside.glb", "models")
        assert_false(r_bad1[0])
        var r_bad2 = s.import_asset_to_project("/etc/passwd", "assets")
        assert_false(r_bad2[0])
        var r_bad3 = s.import_asset_to_project(ext + "/bad.glb", "models")
        assert_false(r_bad3[0])

        var imp_tex = s.import_asset_to_project(ext + "/brick.png", "textures")
        var imp_glb = s.import_asset_to_project(ext + "/crate.glb", "models")
        var imp_mat = s.import_asset_to_project(ext + "/brick.flmat", "materials")
        var imp_rig = s.import_asset_to_project(ext + "/hero.flrig", "rigs")
        var imp_anm = s.import_asset_to_project(ext + "/hero.flanim", "anims")
        var imp_pf = s.import_asset_to_project(ext + "/crate.flprefab", "prefabs")
        var imp_wav = s.import_asset_to_project(ext + "/hit.wav", "audio")
        assert_true(imp_tex[0] and imp_tex[1].rel_path == "textures/brick.png")
        assert_true(imp_glb[0] and imp_glb[1].rel_path == "models/crate.glb")
        assert_true(imp_mat[0] and imp_mat[1].rel_path == "materials/brick.flmat")
        assert_true(imp_rig[0] and imp_rig[1].rel_path == "rigs/hero.flrig")
        assert_true(imp_anm[0] and imp_anm[1].rel_path == "anims/hero.flanim")
        assert_true(imp_pf[0] and imp_pf[1].rel_path == "prefabs/crate.flprefab")
        assert_true(imp_wav[0] and imp_wav[1].rel_path == "audio/hit.wav")

        # A clean project with a renamed asset + its imported sidecar resolves the saved reference.
        var relocated = tmp + "/checkout/renamed"
        assert_true(make_dirs(relocated))
        var imported = proj + "/materials/brick.flmat"
        assert_true(file_size(imported + ".flmeta") > 0)
        var imported_bytes = read_bytes(imported)
        var imported_meta = read_bytes(imported + ".flmeta")
        assert_true(write_bytes(relocated + "/stone.flmat", imported_bytes))
        assert_true(write_bytes(relocated + "/stone.flmat.flmeta", imported_meta))
        var fresh = AssetServer(enable_watcher=False)
        fresh.loose_root = tmp + "/checkout"
        fresh.index_loose_directory()
        var handle = fresh.resolve_raw(imp_mat[1])
        var asset = fresh.raws.get_ptr(handle)
        assert_true(Bool(asset))
        assert_true(asset.value()[].bytes == read_bytes(imported))
        _ = fresh.raws.release(handle)

        # 2. activate_asset_entry dispatches GLB, FLMAT, FLPREFAB, FLRIG, FLANIM, WAV with undo support
        var mesh_eid = s.activate_asset_entry("models/crate.glb", drop_pos=Vec3(2.0, 0.5, -1.0), from_drop=True)
        assert_true(mesh_eid >= 0)
        var mesh_e = s.resolve_entity(mesh_eid)
        assert_equal(s.app.world.get_ptr[MeshRendererComponent](mesh_e).value()[].mesh_ref.rel_path, "models/crate.glb")
        assert_true(s.app.world.get_ptr[MeshRendererComponent](mesh_e).value()[].mesh_id >= 0)

        var h_pre_mat = s.state_hash()
        assert_equal(s.activate_asset_entry("materials/brick.flmat", target_entity_id=mesh_eid, from_drop=True), mesh_eid)
        var h_post_mat = s.state_hash()
        assert_true(h_post_mat != h_pre_mat)
        assert_true(s.app.world.has[MaterialOverrideComponent](mesh_e))
        assert_equal(s.app.world.get_ptr[MeshRendererComponent](mesh_e).value()[].material_ref.rel_path, "materials/brick.flmat")

        # Undo/redo invertibility on asset assignment + commit_ecs_edit
        assert_true(s.undo())
        assert_equal(s.state_hash(), h_pre_mat)
        assert_true(s.redo())
        assert_equal(s.state_hash(), h_post_mat)

        assert_equal(s.activate_asset_entry("rigs/hero.flrig", target_entity_id=mesh_eid), mesh_eid)
        assert_true(s.app.world.has[RigComponent](mesh_e))
        assert_equal(s.app.world.get_ptr[RigComponent](mesh_e).value()[].rig_ref.rel_path, "rigs/hero.flrig")

        assert_equal(s.activate_asset_entry("anims/hero.flanim", target_entity_id=mesh_eid), mesh_eid)
        assert_true(s.app.world.has[AnimatorComponent](mesh_e))
        assert_equal(s.app.world.get_ptr[AnimatorComponent](mesh_e).value()[].graph_ref.rel_path, "anims/hero.flanim")

        var before_blob = s._capture_world_blob()
        var ac_m = s.app.world.get_ptr_mut[AnimatorComponent](mesh_e)
        ac_m.value()[].speed = Float32(2.25)
        assert_true(s.commit_ecs_edit(mesh_eid, "Edit Speed", before_blob^))
        var h_spd = s.state_hash()
        assert_true(s.undo())
        assert_true(abs(s.app.world.get_ptr[AnimatorComponent](mesh_e).value()[].speed - Float32(1.0)) < Float32(1.0e-4))
        assert_true(s.redo())
        assert_equal(s.state_hash(), h_spd)

        var pf_eid = s.activate_asset_entry("prefabs/crate.flprefab", drop_pos=Vec3(5.0, 0.0, 0.0), from_drop=True)
        assert_true(pf_eid >= 0)
        var pf_e = s.resolve_entity(pf_eid)
        assert_true(s.app.world.has[PrefabInstanceComponent](pf_e))
        assert_equal(s.app.world.get_ptr[PrefabInstanceComponent](pf_e).value()[].prefab_ref.rel_path, "prefabs/crate.flprefab")

        var au_eid = s.activate_asset_entry("audio/hit.wav")
        assert_true(au_eid >= 0)
        assert_equal(
            s.app.world.get_ptr[AudioEmitterComponent](s.resolve_entity(au_eid)).value()[].audio_ref.rel_path,
            "audio/hit.wav",
        )

        # 3. UiContext begin_drag_source -> accept_drag_target_rect rect hit-test
        var ui = UiContext()
        var inp = InputState()
        var vp_rect = RectF(200.0, 40.0, 400.0, 300.0)
        inp.begin_frame()
        inp.move_mouse(20.0, 14.0)
        inp.set_mouse(0, True)
        ui.begin_frame(inp, 1000.0, 700.0)
        _ = ui.button("ab_item_0", "crate.glb", 100.0, 22.0)
        _ = ui.begin_drag_source("ASSET_ENTRY", 0, "models/crate.glb")
        ui.end_frame()

        inp.begin_frame()
        inp.move_mouse(300.0, 150.0)
        inp.set_mouse(0, True)
        ui.begin_frame(inp, 1000.0, 700.0)
        ui.end_frame()

        inp.begin_frame()
        inp.move_mouse(320.0, 160.0)
        inp.set_mouse(0, False)
        ui.begin_frame(inp, 1000.0, 700.0)
        var drop_res = ui.accept_drag_target_rect("ASSET_ENTRY", vp_rect)
        ui.end_frame()
        assert_true(drop_res[0])
        assert_equal(drop_res[1], 0)
        assert_equal(drop_res[2], "models/crate.glb")

        # 4. Gate 1 Remediation regressions:
        #    (a) enable_watcher=False .flmat disk edit + _assign_flmat_to_entity must NOT revert to stale cache
        var mat_disk = proj + "/materials/brick.flmat"
        var fmat_upd = FlmatAsset("BrickMat")
        fmat_upd.base_color = Vec3(0.12, 0.85, 0.30)
        fmat_upd.roughness = Float32(0.22)
        assert_true(save_flmat_file(mat_disk, fmat_upd))
        assert_true(s._assign_flmat_to_entity(mesh_eid, "materials/brick.flmat"))
        ref mo_after = s.app.world.get_ptr[MaterialOverrideComponent](mesh_e).value()[unsafe_offset=0]
        assert_true(abs(mo_after.base_color.x - Float32(0.12)) < Float32(1.0e-4))
        assert_true(abs(mo_after.base_color.y - Float32(0.85)) < Float32(1.0e-4))
        assert_true(abs(mo_after.base_color.z - Float32(0.30)) < Float32(1.0e-4))

        #    (b) .flscene, .flpk, and uppercase .PNG import + scan_asset_browser
        _m3b_write_text(ext + "/empty.flscene", '{"format":"flscn","version":2,"name":"Empty","entities":[]}')
        var flpk_hdr = List[UInt8]()
        flpk_hdr.append(0x46); flpk_hdr.append(0x4C); flpk_hdr.append(0x50); flpk_hdr.append(0x4B)
        flpk_hdr.append(3); flpk_hdr.append(0); flpk_hdr.append(0); flpk_hdr.append(0)
        assert_true(write_bytes(ext + "/bundle.flpk", flpk_hdr))
        assert_true(write_png(ext + "/UPPER_TEX.PNG", px_brick, 8, 8, stamped=False))
        var imp_scn = s.import_asset_to_project(ext + "/empty.flscene", "scenes")
        var imp_pk = s.import_asset_to_project(ext + "/bundle.flpk", "packs")
        var imp_up = s.import_asset_to_project(ext + "/UPPER_TEX.PNG", "textures")
        assert_true(imp_scn[0] and imp_pk[0] and imp_up[0])
        _ = s.scan_asset_browser()
        var found_scn = False
        var found_pk = False
        var found_up = False
        for ab_e in s.asset_browser_entries:
            if ab_e.endswith("empty.flscene"):
                found_scn = True
            elif ab_e.endswith("bundle.flpk"):
                found_pk = True
            elif ab_e.endswith("UPPER_TEX.PNG"):
                found_up = True
        assert_true(found_scn and found_pk and found_up)

        #    (c) .flprefab shorthand "mesh_ref" / "material_ref" spawns MeshRendererComponent & MaterialOverrideComponent
        assert_true(s.app.world.has[MeshRendererComponent](pf_e))
        assert_true(s.app.world.has[MaterialOverrideComponent](pf_e))
        assert_equal(s.app.world.get_ptr[MeshRendererComponent](pf_e).value()[].mesh_ref.rel_path, "models/crate.glb")
        assert_equal(s.app.world.get_ptr[MaterialOverrideComponent](pf_e).value()[].material_ref.rel_path, "materials/brick.flmat")

        #    (d) .flrig drop onto existing SkeletonComponent (e.g. from GLB) applies new joints/parents & clears dirty_topology,
        #        and .flrig drop with selected_entity_id == -1 spawns/returns valid entity >= 0
        var rig3 = FrigRigDef("NewThreeJointRig")
        rig3.joints.append(FrigJointDef("Root", -1, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig3.joints.append(FrigJointDef("Spine", 0, Vec3(0.0, 1.1, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        rig3.joints.append(FrigJointDef("Head", 1, Vec3(0.0, 0.6, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        assert_true(save_flrig_file(ext + "/rig3.flrig", rig3))
        var imp_rig3 = s.import_asset_to_project(ext + "/rig3.flrig", "rigs")
        assert_true(imp_rig3[0])
        assert_equal(s.activate_asset_entry("rigs/rig3.flrig", target_entity_id=mesh_eid, from_drop=True), mesh_eid)
        ref sk_after_rig = s.app.world.get_ptr[SkeletonComponent](mesh_e).value()[]
        assert_equal(sk_after_rig.rig_ref.rel_path, "rigs/rig3.flrig")
        assert_equal(sk_after_rig.skeleton.node_count, 3)
        assert_false(sk_after_rig.dirty_topology)
        assert_true(abs(sk_after_rig.skeleton.pose.t[1].y - Float32(1.1)) < Float32(1.0e-4))

        s.clear_selection()
        assert_equal(s.selected_entity_id, -1)
        var rig_spawn_id = s.activate_asset_entry("rigs/rig3.flrig", target_entity_id=-1, drop_pos=Vec3(3.0, 0.0, 0.0), from_drop=True)
        assert_true(rig_spawn_id >= 0)
        assert_true(s.app.world.has[RigComponent](s.resolve_entity(rig_spawn_id)))
        assert_true(s.app.world.has[SkeletonComponent](s.resolve_entity(rig_spawn_id)))

        #    (e) Inspector MeshRef / MaterialRef / entity-level .flrig drop targets via emit_studio_ui_draw_list
        var unrigged_eid = s.spawn_entity("StaticCube", position=Vec3(0.0, 0.0, 0.0), record_undo=False)
        s.select_entity(unrigged_eid)
        var insp_inp = InputState()
        insp_inp.begin_frame()
        s.render_editor_frame(1280, 720, input=insp_inp)
        var insp_info = s.dock.find_panel("Inspector")
        assert_true(insp_info[0] >= 0)
        var drop_x = insp_info[3].x + insp_info[3].w * Float32(0.5)
        var drop_y = insp_info[3].y + insp_info[3].h - Float32(40.0)
        s.ui_ctx.drag_drop.active = True
        s.ui_ctx.drag_drop.kind = "ASSET_ENTRY"
        s.ui_ctx.drag_drop.str_value = "rigs/rig3.flrig"
        var drop_inp = InputState()
        drop_inp.begin_frame()
        drop_inp.move_mouse(drop_x, drop_y)
        drop_inp.set_mouse(0, False)
        s.render_editor_frame(1280, 720, input=drop_inp)
        var unrigged_e = s.resolve_entity(unrigged_eid)
        assert_true(s.app.world.has[RigComponent](unrigged_e))
        assert_true(s.app.world.has[SkeletonComponent](unrigged_e))
        assert_equal(s.app.world.get_ptr[RigComponent](unrigged_e).value()[].rig_ref.rel_path, "rigs/rig3.flrig")

        s.shutdown()
    finally:
        _m3b_rm_rf(tmp)



def _m4b_make_prop_glb(base_color: UInt32 = 0xFFFFFFFF) -> NativeModel:
    var mdl = NativeModel()
    var m = NativeMesh()
    m.base_color = base_color
    m.positions.append(Float32(-0.8)); m.positions.append(Float32(-0.8)); m.positions.append(Float32(0.0))
    m.positions.append(Float32(0.8)); m.positions.append(Float32(-0.8)); m.positions.append(Float32(0.0))
    m.positions.append(Float32(0.0)); m.positions.append(Float32(0.8)); m.positions.append(Float32(0.0))
    for _ in range(3):
        m.normals.append(Float32(0.0)); m.normals.append(Float32(0.0)); m.normals.append(Float32(1.0))
    m.texcoords.append(Float32(0.0)); m.texcoords.append(Float32(0.0))
    m.texcoords.append(Float32(1.0)); m.texcoords.append(Float32(0.0))
    m.texcoords.append(Float32(0.5)); m.texcoords.append(Float32(1.0))
    m.indices.append(0); m.indices.append(1); m.indices.append(2)
    mdl.meshes.append(m^)
    return mdl^


def _m4b_render_3d_pixels(mut s: FlintStudioState) -> List[UInt32]:
    s.look_at(Vec3(0.0, 1.2, 4.5), Vec3(0.0, 0.2, 0.0))
    s.render_editor_frame(160, 120)
    var rw = s.app.resources.get_ptr_mut[RenderWorld]().value()
    var ren = s.app.resources.get_ptr_mut[ScalableRenderer]().value()
    var empty_ui = UiDrawList()
    _ = ren[unsafe_offset=0].render_world(
        rw[unsafe_offset=0], empty_ui, capture_readback=True, force_readback=True
    )
    return ren[unsafe_offset=0].readback_pixels()


def _m4b_pixel_diffs(a: List[UInt32], b: List[UInt32]) -> Tuple[Int, Int]:
    var nonzero = 0
    var diffs = 0
    for i in range(len(b)):
        if b[i] != UInt32(0):
            nonzero += 1
        if i < len(a) and a[i] != b[i]:
            diffs += 1
    return (nonzero, diffs)


def test_m4b_glb_default_material_non_aliasing_and_reload_parity() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_test_m4b_glb_mat_reload_" + String(pid)
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp + "/models")
    _ = make_dirs(tmp + "/materials")
    _ = make_dirs(tmp + "/scenes")
    try:
        var hero_glb = _m4b_make_rigged_glb()
        assert_true(write_bytes(tmp + "/models/hero.glb", hero_glb))
        var prop_mdl = _m4b_make_prop_glb(0xFFFFFFFF)
        assert_true(write_native_glb(tmp + "/models/prop.glb", prop_mdl))
        var hero_mat = FlmatAsset("HeroBlue")
        hero_mat.base_color = Vec3(0.05, 0.25, 0.95)
        hero_mat.roughness = Float32(0.35)
        hero_mat.metallic = Float32(0.15)
        assert_true(save_flmat_file(tmp + "/materials/hero.flmat", hero_mat))

        # 1. Asset-only scene with hero.glb + hero.flmat AND prop.glb (material_ref empty)
        var s1 = FlintStudioState(
            headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0, project_root=tmp
        )
        var hero_id = s1.activate_asset_entry("models/hero.glb", drop_pos=Vec3(-1.0, 0.0, 0.0), from_drop=True)
        assert_true(hero_id >= 0)
        assert_equal(s1.activate_asset_entry("materials/hero.flmat", target_entity_id=hero_id, from_drop=True), hero_id)
        var prop_id = s1.activate_asset_entry("models/prop.glb", drop_pos=Vec3(1.0, 0.0, 0.0), from_drop=True)
        assert_true(prop_id >= 0)

        var px1 = _m4b_render_3d_pixels(s1)
        var h1 = s1.state_hash()
        assert_true(s1.save_scene_file("scenes/mixed.flscn"))
        assert_true(s1.save_scene_file("scenes/mixed.flscnb"))

        # Same-session reload clears mesh_default_mat_ids and preserves 0-diff parity & state_hash()
        assert_true(s1.load_scene_file("scenes/mixed.flscn"))
        var px_same = _m4b_render_3d_pixels(s1)
        assert_equal(s1.state_hash(), h1)
        var same_stat = _m4b_pixel_diffs(px1, px_same)
        assert_true(same_stat[0] > 100 and same_stat[1] == 0)
        s1.shutdown()

        # Fresh Studio reloads (.flscn and .flscnb)
        for fmt in range(2):
            var s2 = FlintStudioState(
                headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0, project_root=tmp
            )
            var spath = "scenes/mixed.flscn" if fmt == 0 else "scenes/mixed.flscnb"
            assert_true(s2.load_scene_file(spath))
            assert_equal(s2.state_hash(), h1)
            var px2 = _m4b_render_3d_pixels(s2)
            var dstat = _m4b_pixel_diffs(px1, px2)
            assert_true(dstat[0] > 100 and dstat[1] == 0)

            var h_e = s2.resolve_entity(hero_id)
            var p_e = s2.resolve_entity(prop_id)
            ref h_mr = s2.app.world.get_ptr[MeshRendererComponent](h_e).value()[unsafe_offset=0]
            ref p_mr = s2.app.world.get_ptr[MeshRendererComponent](p_e).value()[unsafe_offset=0]
            assert_true(p_mr.material_ref.is_empty())
            assert_true(p_mr.material_id >= 0 and p_mr.material_id != h_mr.material_id)
            ref ren2 = s2.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0]
            var p_bc = ren2.materials[p_mr.material_id].base_color
            assert_true(abs(p_bc.x - Float32(1.0)) < Float32(1.0e-4))
            assert_true(abs(p_bc.y - Float32(1.0)) < Float32(1.0e-4))
            assert_true(abs(p_bc.z - Float32(1.0)) < Float32(1.0e-4))
            s2.shutdown()

        # 2. Single hero.glb WITHOUT .flmat across save -> fresh reload (.flscn and .flscnb)
        var sh1 = FlintStudioState(
            headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0, project_root=tmp
        )
        var only_id = sh1.activate_asset_entry("models/hero.glb", drop_pos=Vec3(0.0, 0.0, 0.0), from_drop=True)
        assert_true(only_id >= 0)
        var px_only1 = _m4b_render_3d_pixels(sh1)
        var h_only1 = sh1.state_hash()
        assert_true(sh1.save_scene_file("scenes/hero_only.flscn"))
        assert_true(sh1.save_scene_file("scenes/hero_only.flscnb"))
        sh1.shutdown()

        for fmt in range(2):
            var sh2 = FlintStudioState(
                headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0, project_root=tmp
            )
            var spath = "scenes/hero_only.flscn" if fmt == 0 else "scenes/hero_only.flscnb"
            assert_true(sh2.load_scene_file(spath))
            assert_equal(sh2.state_hash(), h_only1)
            var px_only2 = _m4b_render_3d_pixels(sh2)
            var dstat = _m4b_pixel_diffs(px_only1, px_only2)
            assert_true(dstat[0] > 100 and dstat[1] == 0)
            var oe = sh2.resolve_entity(only_id)
            ref omr = sh2.app.world.get_ptr[MeshRendererComponent](oe).value()[unsafe_offset=0]
            assert_true(omr.material_id >= 0)
            ref ren_o = sh2.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0]
            var obc = ren_o.materials[omr.material_id].base_color
            assert_true(abs(obc.x - Float32(1.0)) < Float32(1.0e-4))
            assert_true(abs(obc.y - Float32(1.0)) < Float32(1.0e-4))
            assert_true(abs(obc.z - Float32(1.0)) < Float32(1.0e-4))
            sh2.shutdown()

        # 3. Tier 2 Vulkan asset-only mixed reload non-aliasing check (guarded by uses_vulkan())
        var sv1 = FlintStudioState(
            headless=True, render_tier=RENDER_TIER_2_AAA_VISBUFFER_RT, width=160, height=120, worker_threads=0, project_root=tmp
        )
        if sv1.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].uses_vulkan():
            var vh_id = sv1.activate_asset_entry("models/hero.glb", drop_pos=Vec3(-1.0, 0.0, 0.0), from_drop=True)
            assert_equal(sv1.activate_asset_entry("materials/hero.flmat", target_entity_id=vh_id, from_drop=True), vh_id)
            var vp_id = sv1.activate_asset_entry("models/prop.glb", drop_pos=Vec3(1.0, 0.0, 0.0), from_drop=True)
            _ = _m4b_render_3d_pixels(sv1)
            var hv1 = sv1.state_hash()
            assert_true(sv1.save_scene_file("scenes/mixed_vk.flscn"))
            sv1.shutdown()

            var sv2 = FlintStudioState(
                headless=True, render_tier=RENDER_TIER_2_AAA_VISBUFFER_RT, width=160, height=120, worker_threads=0, project_root=tmp
            )
            assert_true(sv2.load_scene_file("scenes/mixed_vk.flscn"))
            assert_equal(sv2.state_hash(), hv1)
            _ = _m4b_render_3d_pixels(sv2)
            var vp_e2 = sv2.resolve_entity(vp_id)
            var vh_e2 = sv2.resolve_entity(vh_id)
            ref vp_mr2 = sv2.app.world.get_ptr[MeshRendererComponent](vp_e2).value()[unsafe_offset=0]
            ref vh_mr2 = sv2.app.world.get_ptr[MeshRendererComponent](vh_e2).value()[unsafe_offset=0]
            assert_true(vp_mr2.material_id >= 0 and vp_mr2.material_id != vh_mr2.material_id)
            assert_true(vp_mr2.material_id != sv2.gpu_materials.entity_pbr_mat_ids[vh_id])
            ref ren_v2 = sv2.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0]
            var v_bc = ren_v2.materials[vp_mr2.material_id].base_color
            assert_true(abs(v_bc.x - Float32(1.0)) < Float32(1.0e-4))
            assert_true(abs(v_bc.y - Float32(1.0)) < Float32(1.0e-4))
            assert_true(abs(v_bc.z - Float32(1.0)) < Float32(1.0e-4))
            var g_bc = ren_v2.gpu_slot[0].materials[vp_mr2.material_id].base_color
            assert_true(abs(g_bc.x - Float32(1.0)) < Float32(1.0e-4))
            assert_true(abs(g_bc.y - Float32(1.0)) < Float32(1.0e-4))
            assert_true(abs(g_bc.z - Float32(1.0)) < Float32(1.0e-4))
            sv2.shutdown()
        else:
            sv1.shutdown()

        # 4. Leading whitespace in JSON .flscene loads cleanly
        _m3b_write_text(tmp + "/scenes/ws.flscene", '  \n  {"format":"flscn","version":2,"name":"WS","entities":[]}')
        var sws = FlintStudioState(
            headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0, project_root=tmp
        )
        assert_true(sws.load_scene_file("scenes/ws.flscene"))
        assert_equal(sws.live_entity_count(), 0)
        sws.shutdown()
    finally:
        _m3b_rm_rf(tmp)



def _m4b_make_rigged_glb_clips(n_clips: Int) -> List[UInt8]:
    var bin_data = List[UInt8]()
    var pos_off = len(bin_data)
    _m4b_push_f32_le(bin_data, -0.5); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0)
    _m4b_push_f32_le(bin_data, 0.5); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0)
    _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 1.0); _m4b_push_f32_le(bin_data, 0.0)
    var pos_len = len(bin_data) - pos_off

    var norm_off = len(bin_data)
    for _ in range(3):
        _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 1.0)
    var norm_len = len(bin_data) - norm_off

    var jnt_off = len(bin_data)
    bin_data.append(0); bin_data.append(0); bin_data.append(0); bin_data.append(0)
    bin_data.append(0); bin_data.append(0); bin_data.append(0); bin_data.append(0)
    bin_data.append(1); bin_data.append(0); bin_data.append(0); bin_data.append(0)
    var jnt_len = len(bin_data) - jnt_off

    var wgt_off = len(bin_data)
    for _ in range(3):
        _m4b_push_f32_le(bin_data, 1.0); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0)
    var wgt_len = len(bin_data) - wgt_off

    var ibm_off = len(bin_data)
    for _ in range(2):
        _m4b_push_f32_le(bin_data, 1.0); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0)
        _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 1.0); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0)
        _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 1.0); _m4b_push_f32_le(bin_data, 0.0)
        _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 1.0)
    var ibm_len = len(bin_data) - ibm_off

    var t_off = len(bin_data)
    _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 1.0)
    var t_len = len(bin_data) - t_off

    var v_off = len(bin_data)
    _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.5); _m4b_push_f32_le(bin_data, 0.0)
    _m4b_push_f32_le(bin_data, 0.0); _m4b_push_f32_le(bin_data, 0.9); _m4b_push_f32_le(bin_data, 0.0)
    var v_len = len(bin_data) - v_off

    while len(bin_data) % 4 != 0:
        bin_data.append(0)

    var anims_json = (
        '[{"name":"Idle","samplers":[{"input":5,"output":6,"interpolation":"LINEAR"}],'
        '"channels":[{"sampler":0,"target":{"node":1,"path":"translation"}}]}'
    )
    if n_clips >= 2:
        anims_json += (
            ',{"name":"Run","samplers":[{"input":5,"output":6,"interpolation":"LINEAR"}],'
            '"channels":[{"sampler":0,"target":{"node":1,"path":"translation"}}]}'
        )
    anims_json += "]"

    var json = (
        '{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,2]}],'
        '"buffers":[{"byteLength":' + String(len(bin_data)) + '}],'
        '"bufferViews":['
        '{"buffer":0,"byteOffset":' + String(pos_off) + ',"byteLength":' + String(pos_len) + '},'
        '{"buffer":0,"byteOffset":' + String(norm_off) + ',"byteLength":' + String(norm_len) + '},'
        '{"buffer":0,"byteOffset":' + String(jnt_off) + ',"byteLength":' + String(jnt_len) + '},'
        '{"buffer":0,"byteOffset":' + String(wgt_off) + ',"byteLength":' + String(wgt_len) + '},'
        '{"buffer":0,"byteOffset":' + String(ibm_off) + ',"byteLength":' + String(ibm_len) + '},'
        '{"buffer":0,"byteOffset":' + String(t_off) + ',"byteLength":' + String(t_len) + '},'
        '{"buffer":0,"byteOffset":' + String(v_off) + ',"byteLength":' + String(v_len) + '}'
        '],"accessors":['
        '{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-0.5,0,0],"max":[0.5,1,0]},'
        '{"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"},'
        '{"bufferView":2,"componentType":5121,"count":3,"type":"VEC4"},'
        '{"bufferView":3,"componentType":5126,"count":3,"type":"VEC4"},'
        '{"bufferView":4,"componentType":5126,"count":2,"type":"MAT4"},'
        '{"bufferView":5,"componentType":5126,"count":2,"type":"SCALAR","min":[0],"max":[1]},'
        '{"bufferView":6,"componentType":5126,"count":2,"type":"VEC3"}'
        '],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"JOINTS_0":2,"WEIGHTS_0":3}}]}],'
        '"nodes":[{"name":"Root","children":[1]},{"name":"Hand","translation":[0,0.5,0]},{"name":"MeshNode","mesh":0,"skin":0}],'
        '"skins":[{"joints":[0,1],"inverseBindMatrices":4}],'
        '"animations":' + anims_json + "}"
    )
    var json_bytes = List[UInt8]()
    for b in json.as_bytes():
        json_bytes.append(b)
    while len(json_bytes) % 4 != 0:
        json_bytes.append(0x20)

    var total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    var out = List[UInt8](capacity=total_len)
    _m4b_push_u32_le(out, 0x46546C67)
    _m4b_push_u32_le(out, 2)
    _m4b_push_u32_le(out, UInt32(total_len))
    _m4b_push_u32_le(out, UInt32(len(json_bytes)))
    _m4b_push_u32_le(out, 0x4E4F534A)
    for b in json_bytes:
        out.append(b)
    _m4b_push_u32_le(out, UInt32(len(bin_data)))
    _m4b_push_u32_le(out, 0x004E4942)
    for b in bin_data:
        out.append(b)
    return out^


def test_m4b_pie_hot_reload_preserved_across_pie_stop_and_new_scene_cache_reset() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_test_m4b_pie_hr_" + String(pid)
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp + "/models")
    _ = make_dirs(tmp + "/materials")
    _ = make_dirs(tmp + "/scenes")
    try:
        var hero2_bytes = _m4b_make_rigged_glb_clips(2)
        assert_true(write_bytes(tmp + "/models/hero.glb", hero2_bytes))
        var prop_mdl = _m4b_make_prop_glb(0xFFFFFFFF)
        assert_true(write_native_glb(tmp + "/models/prop.glb", prop_mdl))
        var hero_mat = FlmatAsset("HeroInit")
        hero_mat.base_color = Vec3(0.15, 0.90, 0.25)
        hero_mat.roughness = Float32(0.4)
        assert_true(save_flmat_file(tmp + "/materials/hero.flmat", hero_mat))

        var s = FlintStudioState(
            headless=True,
            render_tier=RENDER_TIER_2_AAA_VISBUFFER_RT,
            width=160,
            height=120,
            worker_threads=0,
            project_root=tmp,
        )
        ref srv_init = s.app.resources.get_ptr_mut[AssetServer]().value()[unsafe_offset=0]
        srv_init.watcher = AssetWatcher(enabled=True)
        assert_true(srv_init.watch_loose_directory(tmp) > 0)

        var hero_id = s.activate_asset_entry("models/hero.glb", drop_pos=Vec3(-1.0, 0.0, 0.0), from_drop=True)
        assert_true(hero_id >= 0)
        assert_equal(s.activate_asset_entry("materials/hero.flmat", target_entity_id=hero_id, from_drop=True), hero_id)
        var prop_id = s.activate_asset_entry("models/prop.glb", drop_pos=Vec3(1.0, 0.0, 0.0), from_drop=True)
        assert_true(prop_id >= 0)
        var hero_e = s.resolve_entity(hero_id)
        s.app.world.get_ptr_mut[AnimatorComponent](hero_e).value()[unsafe_offset=0].clip_index = 1
        s.render_editor_frame(160, 120)
        assert_equal(s.app.world.get_ptr[AnimatorComponent](hero_e).value()[unsafe_offset=0].clip_index, 1)

        # 1. Normal PIE round-trip with NO hot-reload must remain 100% bit-exact in state_hash()
        var h_no_hr = s.state_hash()
        assert_true(s.pie_play())
        assert_equal(s.pie_step(10), 10)
        assert_true(s.pie_stop())
        assert_true(s.pie.last_restore_verified)
        assert_equal(s.state_hash(), h_no_hr)

        # 2. PIE-time live inotify .flmat + .glb (2->1 clips) hot-reload -> pie_stop() preserves reloads!
        assert_true(s.pie_play())
        assert_equal(s.pie_step(2), 2)

        var hero_mat_new = FlmatAsset("HeroHotReload")
        hero_mat_new.base_color = Vec3(0.92, 0.12, 0.45)
        hero_mat_new.roughness = Float32(0.18)
        assert_true(save_flmat_file(tmp + "/materials/hero.flmat", hero_mat_new))
        var hero1_bytes = _m4b_make_rigged_glb_clips(1)
        assert_true(write_bytes(tmp + "/models/hero.glb", hero1_bytes))

        ref srv_pie = s.app.resources.get_ptr_mut[AssetServer]().value()[unsafe_offset=0]
        _ = srv_pie.tick(1000.0)
        _ = srv_pie.tick(1001.0)
        _ = s.step_simulation(1)

        ref mo_pie = s.app.world.get_ptr[MaterialOverrideComponent](hero_e).value()[unsafe_offset=0]
        assert_true(abs(mo_pie.base_color.x - Float32(0.92)) < Float32(1.0e-4))
        assert_true(abs(mo_pie.base_color.y - Float32(0.12)) < Float32(1.0e-4))
        assert_equal(s.app.world.get_ptr[AnimatorComponent](hero_e).value()[unsafe_offset=0].clip_index, 0)

        assert_true(s.pie_stop())
        assert_true(s.pie.last_restore_verified)

        # Immediately after pie_stop() (and after next editor render), hot-reloaded values are preserved
        ref mo_post = s.app.world.get_ptr[MaterialOverrideComponent](hero_e).value()[unsafe_offset=0]
        assert_true(abs(mo_post.base_color.x - Float32(0.92)) < Float32(1.0e-4))
        assert_true(abs(mo_post.base_color.y - Float32(0.12)) < Float32(1.0e-4))
        assert_true(abs(mo_post.base_color.z - Float32(0.45)) < Float32(1.0e-4))
        assert_equal(s.app.world.get_ptr[AnimatorComponent](hero_e).value()[unsafe_offset=0].clip_index, 0)
        assert_true(abs(s.entities[hero_id].base_color.x - Float32(0.92)) < Float32(1.0e-4))
        assert_equal(s.entities[hero_id].clip_index, 0)

        s.render_editor_frame(160, 120)
        ref ren = s.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0]
        if ren.uses_vulkan():
            var pbr_slot = s.gpu_materials.entity_pbr_mat_ids[hero_id]
            assert_true(pbr_slot >= 0)
            var g_bc = ren.gpu_slot[0].materials[pbr_slot].base_color
            assert_true(abs(g_bc.x - Float32(0.92)) < Float32(1.0e-4))
            assert_true(abs(g_bc.y - Float32(0.12)) < Float32(1.0e-4))
            assert_true(abs(g_bc.z - Float32(0.45)) < Float32(1.0e-4))
        ref prop_mr = s.app.world.get_ptr[MeshRendererComponent](s.resolve_entity(prop_id)).value()[unsafe_offset=0]
        assert_true(abs(ren.materials[prop_mr.material_id].base_color.x - Float32(1.0)) < Float32(1.0e-4))

        # 3. new_scene(force=True) clears RenderPluginStateResource & SceneAssetResolver scene caches
        var rps_pre = s.app.resources.get_ptr[RenderPluginStateResource]()
        assert_true(len(rps_pre.value()[unsafe_offset=0].skinned_entries) > 0 or not ren.uses_vulkan())
        assert_true(len(s.app.resources.get_ptr[SceneAssetResolver]().value()[unsafe_offset=0].mesh_default_mat_ids) > 0)
        assert_true(s.new_scene(force=True))
        ref rps_post = s.app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0]
        ref res_post = s.app.resources.get_ptr[SceneAssetResolver]().value()[unsafe_offset=0]
        assert_equal(len(rps_post.override_slots), 0)
        assert_equal(len(rps_post.studio_override_mat_ids), 0)
        assert_equal(len(rps_post.skinned_entries), 0)
        assert_equal(len(rps_post.skinned_model_ids), 0)
        assert_equal(len(rps_post.skinned_invalidated), 0)
        assert_equal(len(res_post.mesh_default_mat_ids), 0)
        assert_equal(len(res_post.skel_entity_rig_keys), 0)

        # Re-spawning prop.glb after new_scene resolves a valid default material
        var prop2_id = s.activate_asset_entry("models/prop.glb", drop_pos=Vec3(0.0, 0.0, 0.0), from_drop=True)
        assert_true(prop2_id >= 0)
        s.render_editor_frame(160, 120)
        ref prop2_mr = s.app.world.get_ptr[MeshRendererComponent](s.resolve_entity(prop2_id)).value()[unsafe_offset=0]
        assert_true(prop2_mr.material_id >= 0)
        assert_true(abs(ren.materials[prop2_mr.material_id].base_color.x - Float32(1.0)) < Float32(1.0e-4))
        s.shutdown()
    finally:
        _m3b_rm_rf(tmp)



def test_m4b_iter5_pie_new_load_guard_flmat_style_undo_and_audio_read_bin() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_test_m4b_iter5_" + String(pid)
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp + "/models")
    _ = make_dirs(tmp + "/materials")
    _ = make_dirs(tmp + "/textures")
    _ = make_dirs(tmp + "/audio")
    _ = make_dirs(tmp + "/scenes")
    try:
        var hero2_bytes = _m4b_make_rigged_glb_clips(2)
        assert_true(write_bytes(tmp + "/models/hero.glb", hero2_bytes))
        var prop_mdl = _m4b_make_prop_glb(0xFFFFFFFF)
        assert_true(write_native_glb(tmp + "/models/prop.glb", prop_mdl))
        var px_a = List[UInt32](length=4 * 4, fill=UInt32(0xFFC8C8C8))
        var px_b = List[UInt32](length=4 * 4, fill=UInt32(0xFF5A5A5A))
        assert_true(write_png(tmp + "/textures/alb1.png", px_a, 4, 4, stamped=False))
        assert_true(write_png(tmp + "/textures/alb2.png", px_b, 4, 4, stamped=False))

        var wav_bytes = List[UInt8]()
        var wav_hdr: StaticString = "RIFF"
        for b in wav_hdr.as_bytes():
            wav_bytes.append(b)
        _m4b_push_u32_le(wav_bytes, 44)  # chunk size
        for b in StaticString("WAVEfmt ").as_bytes():
            wav_bytes.append(b)
        _m4b_push_u32_le(wav_bytes, 16)
        wav_bytes.append(1); wav_bytes.append(0)  # PCM
        wav_bytes.append(1); wav_bytes.append(0)  # mono
        _m4b_push_u32_le(wav_bytes, 8000)
        _m4b_push_u32_le(wav_bytes, 16000)
        wav_bytes.append(2); wav_bytes.append(0)
        wav_bytes.append(16); wav_bytes.append(0)
        for b in StaticString("data").as_bytes():
            wav_bytes.append(b)
        _m4b_push_u32_le(wav_bytes, 8)
        for _ in range(8):
            wav_bytes.append(0)
        assert_true(write_bytes(tmp + "/audio/sfx.wav", wav_bytes))

        var hero_mat = FlmatAsset("HeroIter5")
        hero_mat.base_color = Vec3(0.20, 0.30, 0.40)
        hero_mat.roughness = Float32(0.50)
        hero_mat.shading_mode = SHADING_PBR
        hero_mat.albedo_tex = AssetRef.from_path("textures/alb1.png", ASSET_KIND_TEXTURE)
        assert_true(save_flmat_file(tmp + "/materials/hero.flmat", hero_mat))

        var s = FlintStudioState(
            headless=True,
            render_tier=RENDER_TIER_2_AAA_VISBUFFER_RT,
            width=160,
            height=120,
            worker_threads=0,
            project_root=tmp,
        )
        ref srv_init = s.app.resources.get_ptr_mut[AssetServer]().value()[unsafe_offset=0]
        srv_init.watcher = AssetWatcher(enabled=True)
        assert_true(srv_init.watch_loose_directory(tmp) > 0)

        # Save a 1-entity alternate scene and a corrupt scene file for Defect 1 load tests
        var alt_eid = s.spawn_card_entity(1)
        assert_true(alt_eid >= 0)
        assert_true(s.save_scene_file("scenes/other.flscn"))
        _m3b_write_text(tmp + "/scenes/corrupt.flscn", "{broken_json")
        assert_true(s.new_scene(force=True))

        # Populate main scene: hero + prop + unresolved AudioEmitter (Defect 3)
        var hero_id = s.activate_asset_entry("models/hero.glb", drop_pos=Vec3(-1.0, 0.0, 0.0), from_drop=True)
        assert_true(hero_id >= 0)
        assert_equal(s.activate_asset_entry("materials/hero.flmat", target_entity_id=hero_id, from_drop=True), hero_id)
        var prop_id = s.activate_asset_entry("models/prop.glb", drop_pos=Vec3(1.0, 0.0, 0.0), from_drop=True)
        assert_true(prop_id >= 0)

        # Defect 3: AudioEmitterComponent with audio_ref set directly and asset_path == "" BEFORE resolve_world()
        var au_e = s.app.world.spawn_empty()
        _ = s.app.world.insert[LocalTransform](au_e, LocalTransform(Vec3(2.0, 0.0, 0.0), Quat.identity(), Vec3(1.0, 1.0, 1.0)))
        _ = s.app.world.insert[GlobalTransform](au_e, GlobalTransform.from_local(LocalTransform(Vec3(2.0, 0.0, 0.0), Quat.identity(), Vec3(1.0, 1.0, 1.0))))
        _ = s.app.world.insert[StudioEntityMetaComponent](au_e, StudioEntityMetaComponent(name="UnresolvedAudio"))
        var au_comp = AudioEmitterComponent()
        au_comp.audio_ref = AssetRef.from_path("audio/sfx.wav", ASSET_KIND_AUDIO)
        au_comp.asset_path = ""
        _ = s.app.world.insert[AudioEmitterComponent](au_e, au_comp^)

        var h_pre_au_pie = s.state_hash()
        assert_true(s.pie_play())
        assert_equal(s.pie_step(3), 3)
        assert_true(s.pie_stop())
        assert_true(s.pie.last_restore_verified)
        assert_equal(s.state_hash(), h_pre_au_pie)
        ref au_restored = s.app.world.get_ptr[AudioEmitterComponent](au_e).value()[unsafe_offset=0]
        assert_equal(au_restored.asset_path, "")
        assert_equal(au_restored.audio_ref.rel_path, "audio/sfx.wav")

        # Now run an editor frame and test normal in-Studio Inspector edits + undo/redo invertibility
        s.render_editor_frame(160, 120)
        var hero_e = s.resolve_entity(hero_id)
        var h_before_cel = s.state_hash()
        var hero_mat_cel = hero_mat.copy()
        hero_mat_cel.shading_mode = SHADING_NPR_CEL
        hero_mat_cel.cel_params.outline_width_px = Float32(3.0)
        hero_mat_cel.albedo_tex = AssetRef.from_path("textures/alb2.png", ASSET_KIND_TEXTURE)
        assert_true(save_flmat_file(tmp + "/materials/hero.flmat", hero_mat_cel))
        assert_true(s._assign_flmat_to_entity(hero_id, "materials/hero.flmat"))
        var h_after_cel = s.state_hash()
        assert_true(h_after_cel != h_before_cel)

        var v_rgh = s.entities[hero_id].copy()
        v_rgh.roughness = Float32(0.77)
        assert_true(s.commit_node_edit(hero_id, v_rgh^, "Inspector mat_rgh"))
        var h_after_rgh = s.state_hash()
        assert_true(h_after_rgh != h_after_cel)

        # Normal in-Studio undo/redo MUST invert state_hash() and properties bit-exactly
        assert_true(s.undo())
        assert_equal(s.state_hash(), h_after_cel)
        assert_true(abs(s.app.world.get_ptr[MaterialOverrideComponent](hero_e).value()[unsafe_offset=0].roughness - Float32(0.50)) < Float32(1.0e-4))
        assert_true(s.undo())
        assert_equal(s.state_hash(), h_before_cel)
        assert_equal(s.app.world.get_ptr[StudioStyleComponent](hero_e).value()[unsafe_offset=0].shading_mode, SHADING_PBR)
        assert_true(s.redo())
        assert_equal(s.state_hash(), h_after_cel)
        assert_equal(s.app.world.get_ptr[StudioStyleComponent](hero_e).value()[unsafe_offset=0].shading_mode, SHADING_NPR_CEL)
        assert_true(s.redo())
        assert_equal(s.state_hash(), h_after_rgh)
        assert_true(abs(s.app.world.get_ptr[MaterialOverrideComponent](hero_e).value()[unsafe_offset=0].roughness - Float32(0.77)) < Float32(1.0e-4))

        # Reset roughness back to .flmat default (0.50) via Inspector .flmat edit
        var hero_mat_tex = hero_mat_cel.copy()
        hero_mat_tex.albedo_tex = AssetRef.from_path("textures/alb1.png", ASSET_KIND_TEXTURE)
        assert_true(save_flmat_file(tmp + "/materials/hero.flmat", hero_mat_tex))
        assert_true(s._assign_flmat_to_entity(hero_id, "materials/hero.flmat"))

        # Defect 2 Part A (Edit-mode): External .flmat hot-reload syncs StudioStyleComponent too
        var hero_mat_vfx = hero_mat_tex.copy()
        hero_mat_vfx.base_color = Vec3(0.65, 0.25, 0.85)
        hero_mat_vfx.shading_mode = SHADING_STYLIZED_VFX
        hero_mat_vfx.vfx_params.erosion = Float32(0.72)
        hero_mat_vfx.vfx_params.uv_scroll_v = Float32(6.5)
        assert_true(save_flmat_file(tmp + "/materials/hero.flmat", hero_mat_vfx))
        ref srv_tick = s.app.resources.get_ptr_mut[AssetServer]().value()[unsafe_offset=0]
        _ = srv_tick.tick(1000.0)
        _ = srv_tick.tick(1001.0)
        s.render_editor_frame(160, 120)
        assert_true(abs(s.app.world.get_ptr[MaterialOverrideComponent](hero_e).value()[unsafe_offset=0].base_color.x - Float32(0.65)) < Float32(1.0e-4))
        assert_equal(s.app.world.get_ptr[StudioStyleComponent](hero_e).value()[unsafe_offset=0].shading_mode, SHADING_STYLIZED_VFX)
        assert_true(abs(s.app.world.get_ptr[StudioStyleComponent](hero_e).value()[unsafe_offset=0].vfx_erosion - Float32(0.72)) < Float32(1.0e-4))
        assert_equal(s.entities[hero_id].shading_mode, SHADING_STYLIZED_VFX)

        # Defect 2 Parts A + B (PIE -> External .flmat hot-reload -> pie_stop -> undo -> redo)
        assert_true(s.pie_play())
        assert_equal(s.pie_step(3), 3)
        var hero_mat_pie = hero_mat_vfx.copy()
        hero_mat_pie.base_color = Vec3(0.91, 0.14, 0.33)
        hero_mat_pie.roughness = Float32(0.22)
        hero_mat_pie.shading_mode = SHADING_NPR_CEL
        hero_mat_pie.cel_params.outline_width_px = Float32(4.5)
        assert_true(save_flmat_file(tmp + "/materials/hero.flmat", hero_mat_pie))
        _ = srv_tick.tick(1002.0)
        _ = srv_tick.tick(1003.0)
        assert_true(s.pie_stop())
        assert_true(s.pie.last_restore_verified)

        ref mo_post = s.app.world.get_ptr[MaterialOverrideComponent](hero_e).value()[unsafe_offset=0]
        ref st_post = s.app.world.get_ptr[StudioStyleComponent](hero_e).value()[unsafe_offset=0]
        assert_true(abs(mo_post.base_color.x - Float32(0.91)) < Float32(1.0e-4))
        assert_true(abs(mo_post.roughness - Float32(0.22)) < Float32(1.0e-4))
        assert_equal(st_post.shading_mode, SHADING_NPR_CEL)
        assert_true(abs(st_post.cel_outline_width_px - Float32(4.5)) < Float32(1.0e-4))
        assert_equal(s.entities[hero_id].shading_mode, SHADING_NPR_CEL)

        s.render_editor_frame(160, 120)
        var h_post_pie_stop = s.state_hash()

        # Post-PIE undo() and redo() on pre-hot-reload TX must keep .flmat-backed fields synced to disk!
        assert_true(s.undo())
        ref mo_u = s.app.world.get_ptr[MaterialOverrideComponent](hero_e).value()[unsafe_offset=0]
        ref st_u = s.app.world.get_ptr[StudioStyleComponent](hero_e).value()[unsafe_offset=0]
        assert_true(abs(mo_u.base_color.x - Float32(0.91)) < Float32(1.0e-4))
        assert_equal(st_u.shading_mode, SHADING_NPR_CEL)
        assert_true(abs(st_u.cel_outline_width_px - Float32(4.5)) < Float32(1.0e-4))

        assert_true(s.redo())
        ref mo_r = s.app.world.get_ptr[MaterialOverrideComponent](hero_e).value()[unsafe_offset=0]
        ref st_r = s.app.world.get_ptr[StudioStyleComponent](hero_e).value()[unsafe_offset=0]
        assert_true(abs(mo_r.base_color.x - Float32(0.91)) < Float32(1.0e-4))
        assert_equal(st_r.shading_mode, SHADING_NPR_CEL)
        assert_true(abs(st_r.cel_outline_width_px - Float32(4.5)) < Float32(1.0e-4))
        assert_equal(s.state_hash(), h_post_pie_stop)

        # Post-PIE new mat_rgh edit still inverts cleanly via undo/redo
        var v_post_rgh = s.entities[hero_id].copy()
        v_post_rgh.roughness = Float32(0.88)
        assert_true(s.commit_node_edit(hero_id, v_post_rgh^, "Post-PIE mat_rgh"))
        var h_post_rgh = s.state_hash()
        assert_true(s.undo())
        assert_equal(s.state_hash(), h_post_pie_stop)
        assert_true(s.redo())
        assert_equal(s.state_hash(), h_post_rgh)

        # Defect 1: Menu disabling + failed load does not abort PIE + new_scene/load_scene_file abort PIE cleanly
        var pre_pie_count = s.live_entity_count()
        assert_true(pre_pie_count >= 3)
        assert_true(s.pie_play())
        assert_equal(s.pie_step(2), 2)
        assert_false(s.undo())
        assert_false(s.redo())

        var menu_inp = InputState()
        menu_inp.begin_frame()
        _ = s.handle_command_menu(menu_inp, 1280, 720)
        assert_false(s.menu.bar.groups[0].items[0].enabled)  # SAVE
        assert_false(s.menu.bar.groups[0].items[1].enabled)  # LOAD
        assert_false(s.menu.bar.groups[0].items[2].enabled)  # NEW_SCENE
        assert_false(s.menu.bar.groups[1].items[0].enabled)  # UNDO
        assert_false(s.menu.bar.groups[1].items[1].enabled)  # REDO

        # Failed load during PIE must NOT stop or corrupt PIE
        assert_false(s.load_scene_file("scenes/corrupt.flscn"))
        assert_equal(s.pie.mode, PIE_PLAYING)
        assert_true(len(s.pie.checkpoint_lz4) > 0)
        assert_true("StudioPlay" in s.input_map.active_stack)
        assert_equal(s.live_entity_count(), pre_pie_count)

        # Dirty new_scene(force=False) during PIE defers to confirm modal without aborting PIE yet
        s.mark_dirty(True)
        assert_false(s.new_scene(force=False))
        assert_true(s.confirm_modal_open)
        assert_equal(s.pie.mode, PIE_PLAYING)
        s.confirm_modal_open = False
        s.confirm_modal_action = 0

        # new_scene(force=True) during PIE aborts PIE cleanly; pie_stop() returns False & never resurrects!
        assert_true(s.new_scene(force=True))
        assert_equal(s.live_entity_count(), 0)
        assert_equal(s.pie.mode, PIE_STOPPED)
        assert_equal(len(s.pie.checkpoint_lz4), 0)
        assert_false("StudioPlay" in s.input_map.active_stack)
        assert_equal(s._pie_entry_hot_reloads, 0)
        assert_false(s.pie_stop())
        assert_equal(s.live_entity_count(), 0)

        # load_scene_file() during PIE_PAUSED aborts PIE cleanly; pie_stop() returns False & never resurrects!
        assert_true(s.spawn_card_entity(1) >= 0)
        assert_true(s.spawn_card_entity(2) >= 0)
        assert_true(s.pie_play())
        assert_true(s.pie_pause())
        assert_equal(s.pie.mode, PIE_PAUSED)
        assert_true(s.load_scene_file("scenes/other.flscn"))
        assert_equal(s.live_entity_count(), 1)
        assert_equal(s.pie.mode, PIE_STOPPED)
        assert_equal(len(s.pie.checkpoint_lz4), 0)
        assert_false("StudioPlay" in s.input_map.active_stack)
        assert_false(s.pie_stop())
        assert_equal(s.live_entity_count(), 1)

        s.shutdown()
    finally:
        _m3b_rm_rf(tmp)



def _m5b_discover_mojo_build_cmd(src_v1: String, so_v1: String, src_v2: String, so_v2: String) -> String:
    var src_dir = "src"
    if file_size("src/flint/core/math.mojo") <= 0 and file_size("../src/flint/core/math.mojo") > 0:
        src_dir = "../src"
    var env_root = env_now("FLINT_ROOT")
    if env_root.byte_length() > 0 and file_size(env_root + "/src/flint/core/math.mojo") > 0:
        src_dir = env_root + "/src"
    var repo_root = "."
    if src_dir == "../src":
        repo_root = ".."
    elif src_dir.byte_length() > 4:
        repo_root = String(src_dir[byte=0 : src_dir.byte_length() - 4])

    var chosen_pfx = String("")
    var env_pfx = env_now("MOJO_PREFIX")
    var home_dir = env_now("HOME")
    if env_pfx.byte_length() > 0 and file_size(env_pfx + "/bin/mojo") > 0:
        chosen_pfx = env_pfx
    elif file_size(repo_root + "/.pixi/envs/default/bin/mojo") > 0:
        chosen_pfx = repo_root + "/.pixi/envs/default"
    elif file_size(repo_root + "/../.pixi/envs/default/bin/mojo") > 0:
        chosen_pfx = repo_root + "/../.pixi/envs/default"
    elif home_dir.byte_length() > 0 and file_size(home_dir + "/.pixi/envs/turbomojo/bin/mojo") > 0:
        chosen_pfx = home_dir + "/.pixi/envs/turbomojo"

    var mojo_bin = (chosen_pfx + "/bin/mojo") if chosen_pfx.byte_length() > 0 else String("mojo")
    var pfx_env = (
        "export MOJO_PREFIX=" + chosen_pfx
        + "; export MODULAR_HOME=${MOJO_PREFIX}/share/max; "
        + "export LD_LIBRARY_PATH=${MOJO_PREFIX}/lib:${LD_LIBRARY_PATH:-}; "
    ) if chosen_pfx.byte_length() > 0 else String("")
    return (
        pfx_env
        + mojo_bin + " build -I " + src_dir + " " + src_v1 + " --emit shared-lib -o " + so_v1 + " -O0 --Werror && "
        + mojo_bin + " build -I " + src_dir + " " + src_v2 + " --emit shared-lib -o " + so_v2 + " -O0 --Werror"
    )


def _m5b_guest_so_src(step_delta: String, ver_tag: String) -> String:
    var s = String("from flint.app import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2, POD_FIELD_INT, PodComponentSchema, PodFieldDesc\n")
    s += "from flint.ecs.world import comp_hash64_str\n\n"
    s += "@export(\"flint_plugin_api_version\")\n"
    s += "def flint_plugin_api_version() abi(\"C\") -> UInt32:\n"
    s += "    return FLINT_PLUGIN_API_VERSION\n\n"
    s += "def _step_hr(_ebits: UInt64, ptr: Int, _dt: Float32) abi(\"C\"):\n"
    s += "    Pointer[Int32, MutAnyOrigin](unsafe_from_address=ptr)[unsafe_offset=0] += Int32(" + step_delta + ")\n\n"
    s += "@export(\"flint_plugin_on_load\")\n"
    s += "def flint_plugin_on_load(api_ptr: Int, world_ptr: Int, res_ptr: Int, is_reload: Int32) abi(\"C\") -> Int32:\n"
    s += "    var api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]\n"
    s += "    if not api.is_valid():\n"
    s += "        return -2\n"
    s += "    var sc = PodComponentSchema(\"M5bHrStats\", 4, 4, 0)\n"
    s += "    _ = sc.add_field(PodFieldDesc(\"score\", POD_FIELD_INT, 0, 4))\n"
    s += "    var tid = api.register_schema(world_ptr, res_ptr, sc)\n"
    s += "    if is_reload == 0:\n"
    s += "        var e = api.spawn_empty(world_ptr)\n"
    s += "        var init_v = List[Int32](length=1, fill=Int32(10))\n"
    s += "        _ = api.insert_raw_pod(world_ptr, e, tid, Int(init_v.unsafe_ptr()))\n"
    s += "        if init_v[0] < 0:\n"
    s += "            return -3\n"
    s += "    var r_init = List[Int64](length=1, fill=Int64(" + ver_tag + "))\n"
    s += "    var rptr = api.upsert_pod_resource(res_ptr, UInt64(0x5B5B000000000001), 8, Int(r_init.unsafe_ptr()))\n"
    s += "    if r_init[0] < 0:\n"
    s += "        return -4\n"
    s += "    if rptr != 0:\n"
    s += "        Pointer[Int64, MutAnyOrigin](unsafe_from_address=rptr)[unsafe_offset=0] = Int64(" + ver_tag + ")\n"
    s += "    return 0\n\n"
    s += "@export(\"flint_plugin_on_step\")\n"
    s += "def flint_plugin_on_step(api_ptr: Int, world_ptr: Int, _res_ptr: Int, dt: Float32) abi(\"C\") -> Int32:\n"
    s += "    var api = Pointer[FlintPluginApiV2, ImmutAnyOrigin](unsafe_from_address=api_ptr)[unsafe_offset=0]\n"
    s += "    if not api.is_valid():\n"
    s += "        return -2\n"
    s += "    var tid = api.register_pod_component(world_ptr, comp_hash64_str(\"M5bHrStats\"), 4, 4, 0)\n"
    s += "    return Int32(api.for_each_pod1(world_ptr, tid, _step_hr, dt))\n\n"
    s += "@export(\"flint_plugin_on_unload\")\n"
    s += "def flint_plugin_on_unload(_api_ptr: Int, _world_ptr: Int, _res_ptr: Int, _is_reload: Int32) abi(\"C\") -> None:\n"
    s += "    pass\n"
    return s^


def test_m5b_f22b_pie_gameplay_hot_reload_and_stop_restore() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_test_m5b_hr_" + String(pid)
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp)
    try:
        var src_v1 = tmp + "/gp_v1.mojo"
        var src_v2 = tmp + "/gp_v2.mojo"
        var so_v1 = tmp + "/gp_v1.so"
        var so_v2 = tmp + "/gp_v2.so"
        _m3b_write_text(src_v1, _m5b_guest_so_src("1", "101"))
        _m3b_write_text(src_v2, _m5b_guest_so_src("10", "202"))
        var cmd = _m5b_discover_mojo_build_cmd(src_v1, so_v1, src_v2, so_v2) + "\x00"
        var rc = Int(external_call["system", Int32](cmd.as_bytes().unsafe_ptr()))
        assert_equal(rc, 0)
        assert_true(file_size(so_v1) > 0 and file_size(so_v2) > 0)

        var s = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0, project_root=tmp)
        var cube_id = s.spawn_card_entity(1)
        assert_true(cube_id >= 0)
        var pre_hash = s.state_hash()

        # Negative loads
        assert_false(s.load_gameplay_plugin(""))
        assert_false(s.load_gameplay_plugin(tmp + "/does_not_exist.so"))
        assert_false(s.reload_gameplay_plugin())
        assert_equal(s.state_hash(), pre_hash)

        assert_true(s.pie_play())
        assert_true(s.load_gameplay_plugin(so_v1))
        assert_true(s.hot_reload_host.is_loaded())
        assert_equal(s.hot_reload_host.reload_generation, 1)
        assert_true(s.registry.find_by_name("M5bHrStats") >= 0)

        # Find guest-spawned entity and step with v1 (+1 per step)
        var tid = s.app.world.find_component_by_hash(s.registry.pod_schemas[s.registry.entries[s.registry.find_by_name("M5bHrStats")].pod_schema_idx].component_hash)
        assert_true(tid >= 0)
        var hr_e = Entity()
        for i in range(len(s.app.world.entities)):
            if s.app.world.entities[i].alive:
                var cand = Entity(UInt32(i), s.app.world.entities[i].generation)
                if s.app.world.has_by_type_id(cand, tid):
                    hr_e = cand
                    break
        assert_true(hr_e.valid())
        var p1 = s.app.world.get_raw_ptr(hr_e, tid)
        assert_equal(Int(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=p1)[unsafe_offset=0]), 10)

        _ = s.step_pie_frame(Float32(1.0 / 60.0))
        _ = s.step_pie_frame(Float32(1.0 / 60.0))
        assert_equal(Int(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(hr_e, tid))[unsafe_offset=0]), 12)

        # Hot-reload v2 in PIE_PLAYING: preserves entity + increments with v2 (+10) & updates resource tag to 202
        var ent_before = s.app.world.entity_count()
        assert_true(s.reload_gameplay_plugin(so_v2))
        assert_equal(s.pie.mode, PIE_PLAYING)
        assert_equal(s.hot_reload_host.reload_generation, 2)
        assert_equal(s.app.world.entity_count(), ent_before)
        assert_true(s.app.world.is_alive(hr_e))
        var rptr = s.app.resources.get_raw_ptr(UInt64(0x5B5B000000000001))
        assert_true(rptr != 0)
        assert_equal(Int(Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=rptr)[unsafe_offset=0]), 202)

        _ = s.step_pie_frame(Float32(1.0 / 60.0))
        assert_equal(Int(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(hr_e, tid))[unsafe_offset=0]), 22)

        # pie_stop unloads .so and restores exact pre_hash while keeping gameplay_so_path
        assert_true(s.pie_stop())
        assert_false(s.hot_reload_host.is_loaded())
        assert_equal(s.state_hash(), pre_hash)
        assert_equal(s.gameplay_so_path, so_v2)

        # Render in Edit mode never steps .so; next pie_play auto-loads gameplay_so_path
        s.render_editor_frame(160, 120)
        assert_false(s.hot_reload_host.is_loaded())
        assert_true(s.pie_play())
        assert_true(s.hot_reload_host.is_loaded())
        assert_true(s.pie_stop())
        assert_false(s.hot_reload_host.is_loaded())
        assert_equal(s.state_hash(), pre_hash)
        s.shutdown()
    finally:
        _m3b_rm_rf(tmp)



def test_m5b_f23b_reflection_inspector_add_remove_undo_redo() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_test_m5b_refl_" + String(pid)
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp + "/scenes")
    try:
        var s = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=640, height=480, worker_threads=0, project_root=tmp)
        var schema = PodComponentSchema("GameplayStats", 24, 4, 0)
        assert_true(schema.add_field(PodFieldDesc("active", POD_FIELD_BOOL, 0, 1)))
        assert_true(schema.add_field(PodFieldDesc("hp", POD_FIELD_INT, 4, 4, Float32(0.0), Float32(999.0))))
        assert_true(schema.add_field(PodFieldDesc("speed", POD_FIELD_F32, 8, 4, Float32(0.0), Float32(50.0))))
        assert_true(schema.add_field(PodFieldDesc("vel", POD_FIELD_VEC3, 12, 12)))
        assert_true(s.register_pod_schema(schema))

        var eid = s.spawn_entity("Hero", position=Vec3(1.0, 2.0, 3.0))
        assert_true(eid >= 0)
        s.select_entity(eid)
        var h0 = s.state_hash()

        # Negative boundaries must return False and leave state_hash untouched
        assert_false(s.register_pod_schema(PodComponentSchema("", 8, 4)))
        assert_false(s.add_component_by_name(-1, "PointLightComponent"))
        assert_false(s.add_component_by_name(999999, "PointLightComponent"))
        assert_false(s.remove_component_by_name(-1, "LocalTransform"))
        assert_false(s.remove_component_by_name(999999, "LocalTransform"))
        assert_false(s.add_component_by_name(eid, "NonExistentM2Component"))
        assert_false(s.add_component_by_name(eid, "LocalTransform"))
        assert_false(s.remove_component_by_name(eid, "SpotLightComponent"))
        assert_equal(s.state_hash(), h0)

        # Add built-ins, BehaviorTreeComponent, and POD schema
        assert_true(s.add_component_by_name(eid, "PointLightComponent"))
        var h_pl = s.state_hash()
        assert_true(h_pl != h0)
        assert_true(s.add_component_by_name(eid, "BehaviorTreeComponent"))
        assert_true(s.add_component_by_name(eid, "GameplayStats"))
        var h_all = s.state_hash()
        assert_true(h_all != h_pl)

        # Attach UnknownComponentsComponent and verify Reflection Inspector UI emission
        var se = s.resolve_entity(eid)
        var unk = UnknownComponentsComponent()
        unk.add_json("CustomLegacyBlob", '{"k": 42}')
        _ = s.app.world.insert[UnknownComponentsComponent](se, unk^)
        var h_unk = s.state_hash()

        var ui = UiDrawList()
        var font = s.ui_ctx.font.copy()
        var inp = InputState()
        inp.begin_frame()
        emit_studio_ui_draw_list(ui, s, 0, font, 640, 480, input=inp, dt=Float32(1.0 / 60.0))
        assert_true(len(ui.prims) > 0)

        # Duplicate copies BehaviorTreeComponent + POD component
        var dup_id = s.duplicate_entity(eid, record_undo=True)
        assert_true(dup_id >= 0)
        var dup_e = s.resolve_entity(dup_id)
        assert_true(s.app.world.has[BehaviorTreeComponent](dup_e))
        var gp_tid = s.app.world.find_component_by_hash(schema.component_hash)
        assert_true(gp_tid >= 0 and s.app.world.has_by_type_id(dup_e, gp_tid))
        assert_true(s.undo())
        assert_equal(s.state_hash(), h_unk)

        # Remove POD & built-in + undo/redo hash parity
        assert_true(s.remove_component_by_name(eid, "GameplayStats"))
        assert_true(s.undo())
        assert_equal(s.state_hash(), h_unk)
        assert_true(s.redo())
        assert_true(s.undo())
        assert_equal(s.state_hash(), h_unk)

        # Save .flscn + .flscnb + PIE round-trip parity
        assert_true(s.pie_play())
        _ = s.pie_step(2)
        assert_true(s.pie_stop())
        assert_equal(s.state_hash(), h_unk)

        assert_true(s.save_scene_file("scenes/refl.flscn"))
        assert_true(s.save_scene_file("scenes/refl.flscnb"))
        s.shutdown()

        for fmt in range(2):
            var s2 = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=640, height=480, worker_threads=0, project_root=tmp)
            assert_true(s2.register_pod_schema(schema))
            var sp = "scenes/refl.flscn" if fmt == 0 else "scenes/refl.flscnb"
            assert_true(s2.load_scene_file(sp))
            assert_equal(s2.state_hash(), h_unk)
            s2.shutdown()
    finally:
        _m3b_rm_rf(tmp)



def test_m5b_f24a_pie_camera_possession_shift_f1_and_edit_lock() raises:
    var s = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=320, height=240, worker_threads=0)
    s.look_at(Vec3(0.0, 2.0, -6.0), Vec3(0.0, 0.0, 0.0))
    var cam_id = s.spawn_entity("GameCam", position=Vec3(40.0, 15.0, 40.0))
    assert_true(cam_id >= 0)
    var cam_e = s.resolve_entity(cam_id)
    _ = s.app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=60.0, active=True))

    # Edit mode enforces editor_camera_override == True
    s.render_editor_frame(320, 240)
    assert_true(s.app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0].editor_camera_override)
    assert_true(abs(s.app.resources.get_ptr[RenderWorld]().value()[unsafe_offset=0].camera.eye.x) < Float32(0.5))

    # Enter PIE -> game camera possessed
    assert_true(s.pie_play())
    assert_false(s.pie_camera_ejected)
    _ = s.step_pie_frame(Float32(1.0 / 60.0))
    s.render_editor_frame(320, 240)
    assert_false(s.app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0].editor_camera_override)
    assert_true(abs(s.app.resources.get_ptr[RenderWorld]().value()[unsafe_offset=0].camera.eye.x - Float32(40.0)) < Float32(0.5))

    # Esc (pie_release_cursor) must NOT eject game camera
    var inp_esc = InputState()
    inp_esc.begin_frame()
    inp_esc.set_logical_key(UI_KEY_ESCAPE, True)
    _ = s.process_editor_input(inp_esc, Float32(1.0 / 60.0))
    s.render_editor_frame(320, 240)
    assert_false(s.pie_camera_ejected)
    assert_false(s.app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0].editor_camera_override)

    # Shift+F1 -> eject to editor camera while remaining PIE_PLAYING
    var inp_eject = InputState()
    inp_eject.begin_frame()
    inp_eject.set_logical_key(UI_KEY_LEFT_SHIFT, True)
    inp_eject.set_logical_key(290, True)
    _ = s.process_editor_input(inp_eject, Float32(1.0 / 60.0))
    s.render_editor_frame(320, 240)
    assert_equal(s.pie.mode, PIE_PLAYING)
    assert_true(s.pie_camera_ejected)
    assert_true(s.app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0].editor_camera_override)

    # pie_pause enforces editor camera lock; Shift+F1 again re-possesses
    assert_true(s.pie_pause())
    s.render_editor_frame(320, 240)
    assert_true(s.app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0].editor_camera_override)
    assert_true(s.pie_play())
    var inp_possess = InputState()
    inp_possess.begin_frame()
    inp_possess.set_logical_key(UI_KEY_LEFT_SHIFT, True)
    inp_possess.set_logical_key(290, True)
    _ = s.process_editor_input(inp_possess, Float32(1.0 / 60.0))
    s.render_editor_frame(320, 240)
    assert_false(s.pie_camera_ejected)
    assert_false(s.app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0].editor_camera_override)

    # _abort_pie_session resets pie_camera_ejected and restores editor camera
    var inp_eject2 = InputState()
    inp_eject2.begin_frame()
    inp_eject2.set_logical_key(UI_KEY_LEFT_SHIFT, True)
    inp_eject2.set_logical_key(290, True)
    _ = s.process_editor_input(inp_eject2, Float32(1.0 / 60.0))
    assert_true(s.pie_camera_ejected)
    s._abort_pie_session()
    assert_false(s.pie_camera_ejected)
    assert_equal(s.pie.mode, PIE_STOPPED)
    assert_true(s.app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0].editor_camera_override)

    # Zero-camera fallback in PIE keeps valid editor camera fov_y_deg > 0
    _ = s.app.world.remove[CameraComponent](cam_e)
    assert_true(s.pie_play())
    _ = s.step_pie_frame(Float32(1.0 / 60.0))
    s.render_editor_frame(320, 240)
    assert_true(s.app.resources.get_ptr[RenderWorld]().value()[unsafe_offset=0].camera.fov_y_deg > Float32(0.0))
    assert_true(s.pie_stop())
    s.shutdown()


def test_m5b_f25b_input_map_dockspace_migration_rebind_and_ai_plugin() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_test_m5b_imap_" + String(pid)
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp)
    try:
        # 1. DockSpace has InputMap tab without displacing AssetBrowser as active tab 0, and open_input_map_panel migrates legacy layouts
        var s0 = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=640, height=480, worker_threads=0, project_root=tmp)
        assert_true(s0.dock.find_panel("InputMap")[0] >= 0)
        assert_equal(s0.dock.find_panel("AssetBrowser")[1], 0)
        assert_true(s0.dock.find_panel("AssetBrowser")[2])

        var leg_dock = DockSpace()
        var leg_vp = leg_dock.add_central_viewport("SceneViewport")
        _ = leg_dock.split_node(leg_vp, DOCK_SPLIT_H, 0.72, [String("AssetBrowser"), String("MCP v2")], place_new_first=False)
        assert_true(leg_dock.find_panel("InputMap")[0] < 0)
        s0.dock = leg_dock^
        s0.open_input_map_panel()
        assert_true(s0.dock.find_panel("InputMap")[0] >= 0)
        assert_true(s0.dock.find_panel("InputMap")[2])

        # 2. AiPlugin + register_ai_codecs wired in __init__
        var bt_id = s0.spawn_entity("BtAgent")
        s0.select_entity(bt_id)
        s0.gizmo_mode = GIZMO_MODE_TRANSLATE
        assert_true(s0.add_component_by_name(bt_id, "BehaviorTreeComponent"))
        var bt_e = s0.resolve_entity(bt_id)
        var bt_p = s0.app.world.get_ptr_mut[BehaviorTreeComponent](bt_e)
        assert_true(Bool(bt_p))
        _ = bt_p.value()[].tree.add_action(1)
        var pre_bt_hash = s0.state_hash()
        assert_true(s0.pie_play())
        _ = s0.step_pie_frame(Float32(1.0 / 60.0))
        var bt_p_post = s0.app.world.get_ptr[BehaviorTreeComponent](bt_e)
        assert_true(Bool(bt_p_post) and bt_p_post.value()[].state.time_s > Float32(0.0))
        assert_true(s0.pie_stop())
        assert_equal(s0.state_hash(), pre_bt_hash)

        # 3. InputMap panel rebind capture + Esc cancel + JSON persistence + hotkey suppression
        var ctx = InputContext("Gameplay", 0, False)
        ctx.bind_key("Jump", 32)
        _ = s0.input_map.register_action("Jump", ACTION_BOOL)
        s0.input_map.add_context(ctx^)
        s0.input_map.push_context("Gameplay")

        assert_false(s0.begin_input_rebind("NoSuchCtx", "Jump"))
        assert_false(s0.begin_input_rebind("Gameplay", "NoSuchAct"))
        s0.open_input_map_panel()
        assert_true(s0.dock.find_panel("InputMap")[2])
        assert_true(s0.begin_input_rebind("Gameplay", "Jump"))
        assert_true(s0.ui_ctx.wants_keyboard())

        var ui = UiDrawList()
        var font = s0.ui_ctx.font.copy()
        var inp_neutral = InputState()
        inp_neutral.begin_frame()
        emit_studio_ui_draw_list(ui, s0, 0, font, 640, 480, input=inp_neutral, dt=Float32(1.0 / 60.0))
        assert_true(len(ui.prims) > 0)
        assert_true(s0.ui_ctx.wants_keyboard())

        var inp_cap = InputState()
        inp_cap.begin_frame()
        inp_cap.set_logical_key(UI_KEY_DELETE, True)
        inp_cap.set_logical_key(69, True)
        _ = s0.process_editor_input(inp_cap, Float32(1.0 / 60.0))
        assert_true(s0.is_alive_id(bt_id))
        assert_equal(s0.gizmo_mode, GIZMO_MODE_TRANSLATE)

        # Esc cancels cleanly
        var inp_esc = InputState()
        inp_esc.begin_frame()
        inp_esc.set_logical_key(UI_KEY_ESCAPE, True)
        emit_studio_ui_draw_list(ui, s0, 0, font, 640, 480, input=inp_esc, dt=Float32(1.0 / 60.0))
        assert_false(s0.input_rebind_active)
        assert_false(s0.ui_ctx.wants_keyboard())

        # Rebind to 'K' (75) and save/reload JSON
        assert_true(s0.begin_input_rebind("Gameplay", "Jump"))
        var inp_rebind = InputState()
        inp_rebind.begin_frame()
        inp_rebind.set_logical_key(75, True)
        emit_studio_ui_draw_list(ui, s0, 0, font, 640, 480, input=inp_rebind, dt=Float32(1.0 / 60.0))
        assert_false(s0.input_rebind_active)
        assert_true(s0.save_input_map_file(tmp + "/config/input_map.json"))

        var reloaded = InputMap()
        var raw_im = read_bytes(tmp + "/config/input_map.json")
        assert_true(reloaded.load_json(String(unsafe_from_utf8=raw_im)))
        var inp_k = InputState()
        inp_k.begin_frame()
        inp_k.set_logical_key(75, True)
        reloaded.update(inp_k, Float32(1.0 / 60.0))
        assert_true(reloaded.just_pressed("Jump"))
        s0.shutdown()
    finally:
        _m3b_rm_rf(tmp)


def test_m5b_iter2_pie_step_so_coalesce_and_pod_int_keyboard() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_test_m5b_r2_" + String(pid)
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp)
    try:
        # Part A: pie_step() .so stepping in PIE_PAUSED vs render_editor_frame / step_physics_deterministic
        var src_v1 = tmp + "/gp_v1.mojo"
        var src_v2 = tmp + "/gp_v2.mojo"
        var so_v1 = tmp + "/gp_v1.so"
        var so_v2 = tmp + "/gp_v2.so"
        _m3b_write_text(src_v1, _m5b_guest_so_src("1", "101"))
        _m3b_write_text(src_v2, _m5b_guest_so_src("10", "202"))
        var cmd = _m5b_discover_mojo_build_cmd(src_v1, so_v1, src_v2, so_v2) + "\x00"
        assert_equal(Int(external_call["system", Int32](cmd.as_bytes().unsafe_ptr())), 0)

        var s = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=1280, height=1400, worker_threads=0, project_root=tmp)
        var cube_id = s.spawn_card_entity(1)
        assert_true(cube_id >= 0)
        var pre_hash = s.state_hash()

        s.gameplay_so_path = so_v1
        # 1. PIE_STOPPED -> pie_step(1) auto-loads .so, enters PIE_PAUSED, and steps .so once (10 -> 11)
        assert_equal(s.pie_step(1, Float32(1.0 / 60.0)), 1)
        assert_equal(s.pie.mode, PIE_PAUSED)
        assert_true(s.hot_reload_host.is_loaded())
        var tid = s.app.world.find_component_by_hash(
            s.registry.pod_schemas[s.registry.entries[s.registry.find_by_name("M5bHrStats")].pod_schema_idx].component_hash
        )
        assert_true(tid >= 0)
        var hr_e = Entity()
        for i in range(len(s.app.world.entities)):
            if s.app.world.entities[i].alive:
                var cand = Entity(UInt32(i), s.app.world.entities[i].generation)
                if s.app.world.has_by_type_id(cand, tid):
                    hr_e = cand
                    break
        assert_true(hr_e.valid())
        assert_equal(Int(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(hr_e, tid))[unsafe_offset=0]), 11)

        # 2. render_editor_frame & step_physics_deterministic in PIE_PAUSED do NOT step .so
        s.render_editor_frame(160, 120)
        s.render_editor_frame(160, 120)
        _ = s.step_physics_deterministic(2, Float32(1.0 / 60.0))
        assert_equal(Int(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(hr_e, tid))[unsafe_offset=0]), 11)

        # 3. pie_play() -> pie_pause() -> pie_step(1) steps .so in PIE_PAUSED (11 -> 12)
        assert_true(s.pie_play())
        assert_true(s.pie_pause())
        assert_equal(s.pie.mode, PIE_PAUSED)
        assert_equal(s.pie_step(1, Float32(1.0 / 60.0)), 1)
        assert_equal(Int(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(hr_e, tid))[unsafe_offset=0]), 12)

        assert_true(s.pie_stop())
        assert_false(s.hot_reload_host.is_loaded())
        assert_equal(s.state_hash(), pre_hash)
        _ = s.step_physics_deterministic(2, Float32(1.0 / 60.0))
        assert_false(s.hot_reload_host.is_loaded())
        s.gameplay_so_path = String()

        # Part B: Cross-widget mouse-drag undo coalescing & single-widget drag coalescing
        var edge_schema = PodComponentSchema("EdgePod", 20, 4, 0)
        assert_true(edge_schema.add_field(PodFieldDesc("hp", POD_FIELD_INT, 0, 4)))
        assert_true(edge_schema.add_field(PodFieldDesc("speed", POD_FIELD_F32, 4, 4)))
        assert_true(edge_schema.add_field(PodFieldDesc("vel", POD_FIELD_VEC3, 8, 12)))
        assert_true(s.register_pod_schema(edge_schema))

        var eid = s.spawn_entity("UiProbeEnt")
        assert_true(eid >= 0)
        s.select_entity(eid)
        assert_true(s.add_component_by_name(eid, "EdgePod", record_undo=False))
        var se = s.resolve_entity(eid)
        var edge_tid = s.app.world.find_component_by_hash(edge_schema.component_hash)
        assert_true(edge_tid >= 0)

        var ui = UiDrawList()
        var font = s.ui_ctx.font.copy()
        for _ in range(2):
            var inp0 = InputState()
            inp0.begin_frame()
            emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=inp0, dt=Float32(1.0 / 60.0))

        var root_id = UInt64(0xCBF29CE484222325)
        var dock_id = hash_ui_id(root_id, "dock:Inspector")
        var scroll_id = hash_ui_id(dock_id, "insp_scroll")
        var refl_id = hash_ui_id(scroll_id, "insp_reflect_add")
        var pod_id = hash_ui_id(refl_id, "insp_pod_EdgePod")
        var wid_speed = hash_ui_id(pod_id, "insp_pod_EdgePod_speed_1")
        var wid_vel = hash_ui_id(pod_id, "insp_pod_EdgePod_vel_2")
        var row_hp_id = hash_ui_id(pod_id, "insp_pod_EdgePod_hp_0_row")
        var wid_hp_txt = hash_ui_id(row_hp_id, "insp_pod_EdgePod_hp_0_txt")

        var r_sp_opt = s.ui_ctx.cached_rects.get(wid_speed)
        var r_vel_opt = s.ui_ctx.cached_rects.get(wid_vel)
        var r_txt_opt = s.ui_ctx.cached_rects.get(wid_hp_txt)
        assert_true(Bool(r_sp_opt) and Bool(r_vel_opt) and Bool(r_txt_opt))
        var r_sp = r_sp_opt.value()
        var r_vel = r_vel_opt.value()
        var r_txt = r_txt_opt.value()

        var undo_before = len(s.undo_stack)
        var sp_x0 = r_sp.x + Float32(48.0)
        var sp_y = r_sp.center_y()

        # Drag speed across 2 move frames -> must coalesce into 1 transaction
        var inp_drag = InputState()
        inp_drag.begin_frame()
        inp_drag.move_mouse(sp_x0, sp_y)
        inp_drag.set_mouse(0, True)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=inp_drag, dt=Float32(1.0 / 60.0))

        inp_drag.begin_frame()
        inp_drag.move_mouse(sp_x0 + Float32(20.0), sp_y)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=inp_drag, dt=Float32(1.0 / 60.0))

        inp_drag.begin_frame()
        inp_drag.move_mouse(sp_x0 + Float32(40.0), sp_y)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=inp_drag, dt=Float32(1.0 / 60.0))

        inp_drag.begin_frame()
        inp_drag.set_mouse(0, False)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=inp_drag, dt=Float32(1.0 / 60.0))

        assert_equal(len(s.undo_stack), undo_before + 1)
        var p_edge = s.app.world.get_raw_ptr(se, edge_tid)
        var sp_val = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=p_edge + 4)[unsafe_offset=0]
        assert_true(sp_val > Float32(0.5))

        # Drag vel -> must create a 2nd separate transaction (NOT coalesce into speed's tx)
        var vel_x0 = r_vel.x + Float32(48.0)
        var vel_y = r_vel.center_y()
        inp_drag.begin_frame()
        inp_drag.move_mouse(vel_x0, vel_y)
        inp_drag.set_mouse(0, True)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=inp_drag, dt=Float32(1.0 / 60.0))

        inp_drag.begin_frame()
        inp_drag.move_mouse(vel_x0 + Float32(30.0), vel_y)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=inp_drag, dt=Float32(1.0 / 60.0))

        inp_drag.begin_frame()
        inp_drag.set_mouse(0, False)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=inp_drag, dt=Float32(1.0 / 60.0))

        assert_equal(len(s.undo_stack), undo_before + 2)
        var vel_x_val = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(se, edge_tid) + 8)[unsafe_offset=0]
        assert_true(vel_x_val > Float32(0.5))

        # One undo reverts ONLY vel.x while keeping speed
        assert_true(s.undo())
        var sp_after_1 = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(se, edge_tid) + 4)[unsafe_offset=0]
        var vx_after_1 = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(se, edge_tid) + 8)[unsafe_offset=0]
        assert_true(abs(vx_after_1) < Float32(1.0e-5))
        assert_true(sp_after_1 > Float32(0.5))
        assert_true(s.undo())
        var sp_after_2 = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(se, edge_tid) + 4)[unsafe_offset=0]
        assert_true(abs(sp_after_2) < Float32(1.0e-5))

        # Part C: POD_FIELD_INT keyboard typing "" -> "-" -> "4" -> "2" = -42 + blur recovery
        var tx_cx = r_txt.x + Float32(48.0)
        var tx_cy = r_txt.center_y()
        inp_drag.begin_frame()
        inp_drag.move_mouse(tx_cx, tx_cy)
        inp_drag.set_mouse(0, True)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=inp_drag, dt=Float32(1.0 / 60.0))

        inp_drag.begin_frame()
        inp_drag.set_mouse(0, False)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=inp_drag, dt=Float32(1.0 / 60.0))
        assert_equal(s.ui_ctx.focused_id, wid_hp_txt)

        # Ctrl+A + Backspace -> ""
        var sel_del = InputState()
        sel_del.begin_frame()
        sel_del.set_logical_key(UI_KEY_LEFT_CTRL, True)
        sel_del.set_logical_key(UI_KEY_A, True)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=sel_del, dt=Float32(1.0 / 60.0))

        var bs = InputState()
        bs.begin_frame()
        bs.set_logical_key(UI_KEY_BACKSPACE, True)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=bs, dt=Float32(1.0 / 60.0))
        assert_equal(s._int_edit_wid, wid_hp_txt)
        assert_equal(s._int_edit_buf, "")
        assert_equal(Int(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(se, edge_tid))[unsafe_offset=0]), 0)

        # Type "-", "4", "2" -> -42
        var t_minus = InputState()
        t_minus.begin_frame()
        t_minus.push_text_utf8("-")
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=t_minus, dt=Float32(1.0 / 60.0))
        assert_equal(s._int_edit_buf, "-")
        assert_equal(Int(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(se, edge_tid))[unsafe_offset=0]), 0)

        var t_4 = InputState()
        t_4.begin_frame()
        t_4.push_text_utf8("4")
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=t_4, dt=Float32(1.0 / 60.0))
        assert_equal(Int(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(se, edge_tid))[unsafe_offset=0]), -4)

        var t_2 = InputState()
        t_2.begin_frame()
        t_2.push_text_utf8("2")
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=t_2, dt=Float32(1.0 / 60.0))
        assert_equal(Int(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(se, edge_tid))[unsafe_offset=0]), -42)

        # Clear to "-" and blur -> recovers cleanly without corrupting ECS -42
        var sel_del2 = InputState()
        sel_del2.begin_frame()
        sel_del2.set_logical_key(UI_KEY_LEFT_CTRL, True)
        sel_del2.set_logical_key(UI_KEY_A, True)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=sel_del2, dt=Float32(1.0 / 60.0))
        var bs2 = InputState()
        bs2.begin_frame()
        bs2.set_logical_key(UI_KEY_BACKSPACE, True)
        bs2.push_text_utf8("-")
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=bs2, dt=Float32(1.0 / 60.0))
        assert_equal(s._int_edit_buf, "-")

        inp_drag.begin_frame()
        inp_drag.move_mouse(Float32(10.0), Float32(10.0))
        inp_drag.set_mouse(0, True)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=inp_drag, dt=Float32(1.0 / 60.0))
        inp_drag.begin_frame()
        inp_drag.set_mouse(0, False)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 1400, input=inp_drag, dt=Float32(1.0 / 60.0))
        assert_equal(s._int_edit_wid, UInt64(0))
        assert_equal(s._int_edit_buf, "")
        assert_equal(Int(Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=s.app.world.get_raw_ptr(se, edge_tid))[unsafe_offset=0]), -42)

        s.shutdown()
    finally:
        _m3b_rm_rf(tmp)


def test_t5_inspector_drag_coalesce_skinned_preview_off_and_text_focus() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp = "/tmp/flint_test_t5_studio_" + String(pid)
    _m3b_rm_rf(tmp)
    _ = make_dirs(tmp)
    try:
        # Part A: 5-frame drag coalescing on Animator, BoneSocket, and AudioEmitter sliders/drags
        var s = FlintStudioState(
            headless=True,
            render_tier=RENDER_TIER_0_HEADLESS_SIMD,
            width=1280,
            height=1600,
            worker_threads=0,
            project_root=tmp,
        )
        var eid = s.spawn_entity("T5DragTarget", record_undo=False)
        assert_true(eid >= 0)
        var se = s.resolve_entity(eid)
        _ = s.app.world.insert[AnimatorComponent](se, AnimatorComponent())
        _ = s.app.world.insert[BoneSocketComponent](se, BoneSocketComponent())
        _ = s.app.world.insert[AudioEmitterComponent](se, AudioEmitterComponent())
        s._sync_ecs_and_physics()
        s.select_entity(eid)

        var ui = UiDrawList()
        var font = s.ui_ctx.font.copy()
        for _ in range(2):
            var inp0 = InputState()
            inp0.begin_frame()
            emit_studio_ui_draw_list(ui, s, 0, font, 1280, 2200, input=inp0, dt=Float32(1.0 / 60.0))

        var root_id = UInt64(0xCBF29CE484222325)
        var dock_id = hash_ui_id(root_id, "dock:Inspector")
        var scroll_id = hash_ui_id(dock_id, "insp_scroll")
        var anim_node_id = hash_ui_id(scroll_id, "insp_animator")
        var rig_node_id = hash_ui_id(scroll_id, "insp_rig_skel")
        var aud_node_id = hash_ui_id(scroll_id, "insp_audio")

        var wid_anim_spd = hash_ui_id(anim_node_id, "insp_anim_spd")
        var wid_bs_pos = hash_ui_id(rig_node_id, "insp_bs_pos")
        var wid_bs_scl = hash_ui_id(rig_node_id, "insp_bs_scl")
        var wid_au_vol = hash_ui_id(aud_node_id, "insp_au_vol")
        var wid_au_pch = hash_ui_id(aud_node_id, "insp_au_pch")

        var r_aspd = s.ui_ctx.cached_rects.get(wid_anim_spd).value()
        var r_bpos = s.ui_ctx.cached_rects.get(wid_bs_pos).value()
        var r_bscl = s.ui_ctx.cached_rects.get(wid_bs_scl).value()
        var r_avol = s.ui_ctx.cached_rects.get(wid_au_vol).value()
        var r_apch = s.ui_ctx.cached_rects.get(wid_au_pch).value()

        # Helper lambda-style loop over 5 controls: each 5-frame drag must add exactly +1 undo entry
        var drag_xs: List[Float32] = [
            r_aspd.x + r_aspd.w * Float32(0.45),
            r_bpos.x + Float32(28.0),
            r_bscl.x + Float32(28.0),
            r_avol.x + r_avol.w * Float32(0.45),
            r_apch.x + r_apch.w * Float32(0.45),
        ]
        var drag_ys: List[Float32] = [
            r_aspd.center_y(),
            r_bpos.center_y(),
            r_bscl.center_y(),
            r_avol.center_y(),
            r_apch.center_y(),
        ]
        for ci in range(5):
            var u0 = len(s.undo_stack)
            var x0 = drag_xs[ci]
            var y0 = drag_ys[ci]
            var inp = InputState()
            inp.begin_frame()
            inp.move_mouse(x0, y0)
            inp.set_mouse(0, True)
            emit_studio_ui_draw_list(ui, s, 0, font, 1280, 2200, input=inp, dt=Float32(1.0 / 60.0))
            for fi in range(1, 6):
                inp.begin_frame()
                inp.move_mouse(x0 + Float32(fi * 8), y0)
                emit_studio_ui_draw_list(ui, s, 0, font, 1280, 2200, input=inp, dt=Float32(1.0 / 60.0))
            inp.begin_frame()
            inp.set_mouse(0, False)
            emit_studio_ui_draw_list(ui, s, 0, font, 1280, 2200, input=inp, dt=Float32(1.0 / 60.0))
            assert_equal(len(s.undo_stack), u0 + 1)

        # Part B: UiContext text_input focused_id cleared on entity deselect (no re-capture on reselect)
        var wid_name = hash_ui_id(scroll_id, "insp_name")
        var r_name = s.ui_ctx.cached_rects.get(wid_name).value()
        var inp_click = InputState()
        inp_click.begin_frame()
        inp_click.move_mouse(r_name.x + Float32(40.0), r_name.center_y())
        inp_click.set_mouse(0, True)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 2200, input=inp_click, dt=Float32(1.0 / 60.0))
        inp_click.begin_frame()
        inp_click.set_mouse(0, False)
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 2200, input=inp_click, dt=Float32(1.0 / 60.0))
        assert_true(s.ui_ctx.wants_keyboard(), "clicking insp_name must assert wants_keyboard()")
        assert_true(s.ui_ctx.focused_id != UInt64(0))

        s.clear_selection()
        var inp_idle = InputState()
        inp_idle.begin_frame()
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 2200, input=inp_idle, dt=Float32(1.0 / 60.0))
        assert_false(s.ui_ctx.wants_keyboard())
        assert_equal(s.ui_ctx.focused_id, UInt64(0))

        s.select_entity(eid)
        inp_idle.begin_frame()
        emit_studio_ui_draw_list(ui, s, 0, font, 1280, 2200, input=inp_idle, dt=Float32(1.0 / 60.0))
        assert_false(s.ui_ctx.wants_keyboard(), "reselecting entity must NOT re-capture keyboard")
        assert_true(s.ui_ctx.focused_id != wid_name, "reselecting entity must NOT restore focus to insp_name")
        assert_equal(s.ui_ctx._focused_text_id, UInt64(0))

        # Part C: Skinned mesh allocated & synced when viewport_anim_preview = False
        s.viewport_anim_preview = False
        var sk_mdl = NativeModel()
        var smesh = NativeMesh()
        smesh.positions.append(Float32(0.0)); smesh.positions.append(Float32(0.0)); smesh.positions.append(Float32(0.0))
        smesh.positions.append(Float32(1.0)); smesh.positions.append(Float32(0.0)); smesh.positions.append(Float32(0.0))
        smesh.positions.append(Float32(0.0)); smesh.positions.append(Float32(1.0)); smesh.positions.append(Float32(0.0))
        for _ in range(3):
            smesh.normals.append(Float32(0.0)); smesh.normals.append(Float32(0.0)); smesh.normals.append(Float32(1.0))
            smesh.texcoords.append(Float32(0.0)); smesh.texcoords.append(Float32(0.0))
            smesh.joints.append(UInt16(0)); smesh.joints.append(UInt16(0)); smesh.joints.append(UInt16(0)); smesh.joints.append(UInt16(0))
            smesh.weights.append(Float32(1.0)); smesh.weights.append(Float32(0.0)); smesh.weights.append(Float32(0.0)); smesh.weights.append(Float32(0.0))
        smesh.indices.append(0); smesh.indices.append(1); smesh.indices.append(2)
        sk_mdl.meshes.append(smesh^)
        var rnode = NativeNode()
        rnode.name = "Root"
        rnode.mesh = 0
        rnode.skin = 0
        sk_mdl.nodes.append(rnode^)
        var skn = NativeSkin()
        skn.joints.append(0)
        for mi in range(16):
            skn.inverse_bind.append(Float32(1.0) if (mi % 5 == 0) else Float32(0.0))
        sk_mdl.skins.append(skn^)

        var mdl_id = UInt64(990011)
        s.app.resources.get_ptr_mut[AnimAssetCache]().value()[unsafe_offset=0].register_model(mdl_id, sk_mdl^)
        var sk_eid = s.spawn_entity("SkinnedNoPreview", record_undo=False)
        var sk_e = s.resolve_entity(sk_eid)
        _ = s.app.world.insert[MeshRendererComponent](sk_e, MeshRendererComponent(-1, -1))
        _ = s.app.world.insert[SkeletonComponent](sk_e, SkeletonComponent(mdl_id))

        s.render_editor_frame(320, 240)
        ref rps = s.app.resources.get_ptr[RenderPluginStateResource]().value()[unsafe_offset=0]
        assert_true(rps.skinned_sync_count >= 1, "skinned mesh must sync even with viewport_anim_preview=False")
        ref mr_sk = s.app.world.get_ptr[MeshRendererComponent](sk_e).value()[unsafe_offset=0]
        assert_true(mr_sk.mesh_id >= 0, "skinned mesh_id must be allocated even with viewport_anim_preview=False")

        s.shutdown()
    finally:
        _m3b_rm_rf(tmp)


def main() raises:
    test_bug1_edit_vs_pie_scheduling()
    print("[PASS] test_bug1_edit_vs_pie_scheduling")
    test_bug2_overlay_lines_and_studio_lights()
    print("[PASS] test_bug2_overlay_lines_and_studio_lights")
    test_bug3_material_override_isolation_and_expanded_fields()
    print("[PASS] test_bug3_material_override_isolation_and_expanded_fields")
    test_bug3_studio_gpu_materials_slot_recycling()
    print("[PASS] test_bug3_studio_gpu_materials_slot_recycling")
    test_bug4_ui_context_capture_m0_widgets_and_esc_guard()
    print("[PASS] test_bug4_ui_context_capture_m0_widgets_and_esc_guard")
    test_bug5a_kcc_triggers()
    print("[PASS] test_bug5a_kcc_triggers")
    test_bug5b_edit_camera_isolation()
    print("[PASS] test_bug5b_edit_camera_isolation")
    test_iteration2_challenger_regressions()
    print("[PASS] test_iteration2_challenger_regressions")
    test_iteration4_vulkan_pbr_serialization_and_mo_enabled_parity()
    print("[PASS] test_iteration4_vulkan_pbr_serialization_and_mo_enabled_parity")
    test_iteration4_cel_outline_and_smear_pie_regressions()
    print("[PASS] test_iteration4_cel_outline_and_smear_pie_regressions")
    test_iteration4_tier2_slot_recycling()
    print("[PASS] test_iteration4_tier2_slot_recycling")
    test_iteration4_default_studio_foliage_materials_non_aliasing()
    print("[PASS] test_iteration4_default_studio_foliage_materials_non_aliasing")
    test_iteration5_cel_vfx_mo_enabled_parity()
    print("[PASS] test_iteration5_cel_vfx_mo_enabled_parity")
    test_m3a_dockspace_json_validation_and_env_spotlight_tier0()
    print("[PASS] test_m3a_dockspace_json_validation_and_env_spotlight_tier0")
    test_m3b_f11_input_context_hotkeys_dirty_modal_and_layout()
    print("[PASS] test_m3b_f11_input_context_hotkeys_dirty_modal_and_layout")
    test_inspector_sections_and_frame_selection()
    test_m3b_f12_inspector_guards_and_enabled_sync()
    print("[PASS] test_m3b_f12_inspector_guards_and_enabled_sync")
    test_m3b_f13_recursive_asset_browser_and_spawn_cards()
    print("[PASS] test_m3b_f13_recursive_asset_browser_and_spawn_cards")
    test_project_assets_import_search_and_create()
    test_m3b_f13_28_assets_spawn_cards_layout_and_clicks()
    print("[PASS] test_m3b_f13_28_assets_spawn_cards_layout_and_clicks")
    test_m3b_f14_f15_plugins_navbake_overlays_spotlight_and_scenes_root()
    print("[PASS] test_m3b_f14_f15_plugins_navbake_overlays_spotlight_and_scenes_root")
    test_m3b_iter3_scroll_clip_cull_and_empty_regressions()
    print("[PASS] test_m3b_iter3_scroll_clip_cull_and_empty_regressions")
    test_m3b_iter4_modal_frame2_layout_and_clicks()
    print("[PASS] test_m3b_iter4_modal_frame2_layout_and_clicks")
    test_m4a_state_hash_unknown_components_and_bone_socket_slots()
    print("[PASS] test_m4a_state_hash_unknown_components_and_bone_socket_slots")
    test_m4b_edit_mode_idle_preview_invariant_and_pie_parity()
    print("[PASS] test_m4b_edit_mode_idle_preview_invariant_and_pie_parity")
    test_m4b_studio_asset_import_drag_drop_activate_and_inspector_undo()
    print("[PASS] test_m4b_studio_asset_import_drag_drop_activate_and_inspector_undo")
    test_m4b_glb_default_material_non_aliasing_and_reload_parity()
    print("[PASS] test_m4b_glb_default_material_non_aliasing_and_reload_parity")
    test_m4b_pie_hot_reload_preserved_across_pie_stop_and_new_scene_cache_reset()
    print("[PASS] test_m4b_pie_hot_reload_preserved_across_pie_stop_and_new_scene_cache_reset")
    test_m4b_iter5_pie_new_load_guard_flmat_style_undo_and_audio_read_bin()
    print("[PASS] test_m4b_iter5_pie_new_load_guard_flmat_style_undo_and_audio_read_bin")
    test_m5b_f22b_pie_gameplay_hot_reload_and_stop_restore()
    print("[PASS] test_m5b_f22b_pie_gameplay_hot_reload_and_stop_restore")
    test_m5b_f23b_reflection_inspector_add_remove_undo_redo()
    print("[PASS] test_m5b_f23b_reflection_inspector_add_remove_undo_redo")
    test_m5b_f24a_pie_camera_possession_shift_f1_and_edit_lock()
    print("[PASS] test_m5b_f24a_pie_camera_possession_shift_f1_and_edit_lock")
    test_m5b_f25b_input_map_dockspace_migration_rebind_and_ai_plugin()
    print("[PASS] test_m5b_f25b_input_map_dockspace_migration_rebind_and_ai_plugin")
    test_m5b_iter2_pie_step_so_coalesce_and_pod_int_keyboard()
    print("[PASS] test_m5b_iter2_pie_step_so_coalesce_and_pod_int_keyboard")
    test_t5_inspector_drag_coalesce_skinned_preview_off_and_text_focus()
    print("[PASS] test_t5_inspector_drag_coalesce_skinned_preview_off_and_text_focus")
    print("[ALL PASS] M2+M3a+M3b+M4a+M4b+M5b+Tier5 Studio Core Bugs Regression Suite (33/33)")
