# AREA: m0
"""Tier 1 Opaque-Box Feature Coverage: Area B — Bugs #1-#5 (F06-F10) & Studio M0 Core (F11-F14)."""

from std.collections import List
from std.math import abs, isnan
from flint.anim.plugin import AnimAssetCache, AnimPlugin, AnimatorComponent, register_anim_codecs
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
from flint.app.window_plugin import WindowStateResource
from flint.assets.json import parse_json
from flint.assets.mesh_builder import MeshBuilder
from flint.audio.qoa import QoaAsset, encode_qoa
from flint.audio.server import AudioServer, BUS_SFX
from flint.camera.rig import CAM_MODE_FREE_FLY, CAM_MODE_ORBIT_FOLLOW
from flint.core.jobs import JobPool
from flint.core.math import Quat, Vec3
from flint.core.serial import BlobReader, BlobWriter
from flint.ecs.world import Entity, Events, ParallelCommands, World, comp_hash64
from flint.input.input_map import MOD_CTRL, MOD_SHIFT
from flint.physics.character import CharacterController, CharacterControllerConfig
from flint.physics.filter import CAT_PROP_PIECE, CAT_SENSOR, CollisionFilter
from flint.physics.plugin import (
    CharacterControllerComponent,
    ColliderComponent,
    PhysicsPlugin,
    PhysicsSyncStateResource,
    RigidBodyComponent,
    TriggerEventsResource,
)
from flint.physics.world import (
    BODY_SHAPE_CAPSULE,
    PhysicsWorld,
    TRIGGER_ENTER,
    TRIGGER_EXIT,
    TRIGGER_STAY,
    TriggerVolume,
)
from flint.platform.audio import decode_wav
from flint.platform.input import InputState
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import (
    CameraComponent,
    MaterialOverrideComponent,
    MeshRendererComponent,
    PointLightComponent,
    RenderPlugin,
    RenderPluginStateResource,
)
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RENDER_TIER_1_FORWARD_CLUSTERED,
    RenderExtractor,
    RenderWorld,
    ScalableRenderer,
)
from flint.scene.scene import ComponentRegistry, Scene
from flint.scene.transform_hierarchy import (
    GlobalTransform,
    LocalTransform,
    PrevGlobalTransform,
    TransformPlugin,
)
from flint.studio.gizmo import GIZMO_MODE_TRANSLATE
from flint.studio.studio_app import (
    COLLIDER_BOX,
    COLLIDER_CAPSULE,
    FieldVisitor,
    FlintStudioState,
    PIE_PAUSED,
    PIE_PLAYING,
    PIE_STOPPED,
    StudioEntityView,
)
from flint.ui.context import (
    UI_KEY_C,
    UI_KEY_DOWN,
    UI_KEY_ESCAPE,
    UI_KEY_LEFT_CTRL,
    UI_KEY_LEFT_SHIFT,
    UI_KEY_TAB,
    UI_KEY_V,
    UiContext,
    pack_rgba8,
)
from flint.ui.flexbox_dock import DOCK_SPLIT_H, DockSpace, RectF
from e2e_studio.common.studio_test_utils import (
    StudioSuiteRunner,
    make_tmp_dir,
    read_bytes,
    rm_rf_tmp,
    write_minimal_glb,
    write_minimal_png,
    write_minimal_wav,
    write_text,
)


struct ProbeResource(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.e2e.studio.ProbeResource")
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


def probe_pre(
    _w: Pointer[World, MutAnyOrigin],
    r_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var p = r_ptr[unsafe_offset=0].get_ptr_mut[ProbeResource]()
    if p:
        p.value()[unsafe_offset=0].pre_count += 1


def probe_fixed(
    _w: Pointer[World, MutAnyOrigin],
    r_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var p = r_ptr[unsafe_offset=0].get_ptr_mut[ProbeResource]()
    if p:
        p.value()[unsafe_offset=0].fixed_count += 1


def probe_update(
    _w: Pointer[World, MutAnyOrigin],
    r_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref res = r_ptr[unsafe_offset=0]
    var p = res.get_ptr_mut[ProbeResource]()
    var ft = res.get_ptr[FrameTime]()
    if p:
        p.value()[unsafe_offset=0].update_count += 1
        if ft:
            p.value()[unsafe_offset=0].last_update_dt = ft.value()[unsafe_offset=0].delta_seconds
        _ = p.value()[unsafe_offset=0].events.send(p.value()[unsafe_offset=0].update_count)


def probe_post(
    _w: Pointer[World, MutAnyOrigin],
    r_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var p = r_ptr[unsafe_offset=0].get_ptr_mut[ProbeResource]()
    if p:
        p.value()[unsafe_offset=0].post_count += 1


def probe_swap(r_ptr: Pointer[Resources, MutAnyOrigin]):
    var p = r_ptr[unsafe_offset=0].get_ptr_mut[ProbeResource]()
    if p:
        p.value()[unsafe_offset=0].events.swap_buffers()


def run_f06_edit_vs_pie_scheduling(mut runner: StudioSuiteRunner):
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
    )
    _ = studio.spawn_entity("FallingBox", position=Vec3(0.0, 2.0, 0.0), mass=1.0, record_undo=False)
    studio.app.insert_resource[ProbeResource](ProbeResource())
    _ = studio.app.add_system(STAGE_PRE_UPDATE, "e2e_probe_pre", probe_pre)
    _ = studio.app.add_system(STAGE_FIXED_UPDATE, "e2e_probe_fixed", probe_fixed)
    _ = studio.app.add_system(STAGE_UPDATE, "e2e_probe_update", probe_update)
    _ = studio.app.add_system(STAGE_POST_UPDATE, "e2e_probe_post", probe_post)
    studio.app.add_event_swap_hook(probe_swap)

    for _ in range(3):
        studio.render_editor_frame(160, 120, dt=Float32(1.0 / 30.0))

    var probe = studio.app.resources.get_ptr[ProbeResource]()
    var ft = studio.app.resources.get_ptr[FrameTime]()
    var edit_ok = (
        Bool(probe)
        and probe.value()[unsafe_offset=0].pre_count == 3
        and probe.value()[unsafe_offset=0].fixed_count == 0
        and probe.value()[unsafe_offset=0].update_count == 0
        and probe.value()[unsafe_offset=0].post_count == 3
        and Bool(ft)
        and ft.value()[unsafe_offset=0].frame_index == UInt64(3)
        and abs(ft.value()[unsafe_offset=0].delta_seconds - Float32(1.0 / 30.0)) < Float32(1.0e-5)
        and abs(ft.value()[unsafe_offset=0].elapsed_seconds - 0.1) < 1.0e-4
    )
    _ = runner.check("t1_m0_f06_edit_mode_skips_update_and_updates_frametime", edit_ok, "edit scheduling mismatch")

    var pause_ok = studio.pie_play() and studio.pie_pause()
    studio.render_editor_frame(160, 120, dt=Float32(1.0 / 60.0))
    pause_ok = (
        pause_ok
        and probe.value()[unsafe_offset=0].update_count == 0
        and probe.value()[unsafe_offset=0].fixed_count == 0
    )
    _ = runner.check("t1_m0_f06_paused_mode_skips_update", pause_ok, "paused mode ran update/fixed")

    var resume_ok = studio.pie_play()
    var pre_before = probe.value()[unsafe_offset=0].pre_count
    var post_before = probe.value()[unsafe_offset=0].post_count
    var fixed_before = probe.value()[unsafe_offset=0].fixed_count
    var upd_before = probe.value()[unsafe_offset=0].update_count
    var fix_ticks_before = studio.app.resources.get_ptr[FixedTime]().value()[unsafe_offset=0].tick_count

    var steps = studio.step_pie_frame(Float32(1.0 / 60.0))
    studio.render_editor_frame(160, 120, dt=Float32(1.0 / 60.0))
    var ev_cursor = 0
    var evs = studio.app.resources.get_ptr_mut[ProbeResource]().value()[unsafe_offset=0].events.read_since(ev_cursor)
    var pie_frame_ok = (
        resume_ok
        and steps == 1
        and abs(probe.value()[unsafe_offset=0].last_update_dt - Float32(1.0 / 60.0)) < Float32(1.0e-5)
        and (probe.value()[unsafe_offset=0].pre_count - pre_before) == 1
        and (probe.value()[unsafe_offset=0].post_count - post_before) == 1
        and (probe.value()[unsafe_offset=0].fixed_count - fixed_before) == 1
        and (probe.value()[unsafe_offset=0].update_count - upd_before) == 1
        and (studio.app.resources.get_ptr[FixedTime]().value()[unsafe_offset=0].tick_count - fix_ticks_before) == UInt64(1)
        and len(evs) == 1
        and evs[0] == 1
    )
    _ = runner.check("t1_m0_f06_pie_frame_single_run_and_event_swap", pie_frame_ok, "double-run or missed event swap")

    var s0 = studio.step_pie_frame(Float32(0.008))
    studio.render_editor_frame(160, 120, dt=Float32(0.008))
    var s1 = studio.step_pie_frame(Float32(0.010))
    studio.render_editor_frame(160, 120, dt=Float32(0.010))
    var subtick_ok = s0 == 0 and s1 == 1 and probe.value()[unsafe_offset=0].update_count == 3
    _ = runner.check("t1_m0_f06_fixed_time_subtick_accumulator", subtick_ok, "sub-tick accumulation failed")

    var pre_pie_hash = studio.pie.pre_play_hash
    var upd_pre_step = probe.value()[unsafe_offset=0].update_count
    var stepped = studio.pie_step(5, Float32(1.0 / 60.0))
    var stop_ok: Bool
    try:
        stop_ok = studio.pie_stop()
    except:
        stop_ok = False
    var step_restore_ok = (
        stepped == 5
        and (probe.value()[unsafe_offset=0].update_count - upd_pre_step) == 5
        and stop_ok
        and studio.state_hash() == pre_pie_hash
    )
    _ = runner.check("t1_m0_f06_pie_step_and_stop_restores_exact_hash", step_restore_ok, "pie_step/pie_stop hash mismatch")
    studio.shutdown()


def run_f07_overlay_lines_and_lights(mut runner: StudioSuiteRunner):
    var rw = RenderWorld()
    rw.debug_lines.add_line(Vec3(0, 0, 0), Vec3(1, 0, 0), pack_rgba8(255, 0, 0, 255))
    rw.overlay_lines.add_line(Vec3(0, 0, 0), Vec3(0, 1, 0), pack_rgba8(0, 255, 0, 255))
    rw.clear_frame(1, Float32(1.0 / 60.0))
    var survive_ok = rw.debug_lines.segment_count() == 0 and rw.overlay_lines.segment_count() == 1
    var ex = RenderExtractor()
    ex.begin_extraction(rw, 2, Float32(1.0 / 60.0))
    survive_ok = survive_ok and rw.overlay_lines.segment_count() == 1
    rw.clear_frame(3, Float32(1.0 / 60.0), clear_overlay=True)
    survive_ok = survive_ok and rw.overlay_lines.segment_count() == 0
    _ = runner.check("t1_m0_f07_overlay_lines_survive_clear_and_extraction", survive_ok, "overlay_lines cleared prematurely")

    var app = EngineApp(AppConfig(title="e2e_overlay", headless=True, worker_threads=0))
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
    var px_line = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    var post_draw_cleared = rw_ptr.value()[unsafe_offset=0].overlay_lines.segment_count() == 0

    var diff_count = 0
    for i in range(len(px_base)):
        if px_base[i] != px_line[i]:
            diff_count += 1

    app.step_one_frame(Float32(1.0 / 60.0))
    var px_next = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    var stale_diffs = 0
    for i in range(len(px_base)):
        if px_base[i] != px_next[i]:
            stale_diffs += 1
    app.shutdown()
    _ = runner.check(
        "t1_m0_f07_tier0_draws_overlay_then_clears_without_stale_leak",
        post_draw_cleared and diff_count > 10 and stale_diffs == 0,
        "diff=" + String(diff_count) + " stale=" + String(stale_diffs),
    )

    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=640,
        height=360,
        worker_threads=0,
    )
    studio.look_at(Vec3(0.0, 1.5, 4.0), Vec3(0.0, 0.5, 0.0))
    studio.render_editor_frame(640, 360)
    var px_prev = studio.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    var px_settled = px_prev.copy()
    var settled_drift = -1
    for _ in range(15):
        studio.render_editor_frame(640, 360)
        var px_cur = studio.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
        settled_drift = 0
        for i in range(len(px_cur)):
            if px_cur[i] != px_prev[i]:
                settled_drift += 1
        px_settled = px_cur.copy()
        px_prev = px_cur^
        if settled_drift == 0:
            break

    var s_rw = studio.app.resources.get_ptr[RenderWorld]()
    var s_rps = studio.app.resources.get_ptr[RenderPluginStateResource]()
    var center_px = px_settled[(360 // 2) * 640 + (640 // 2)]
    var fallback_ok = (
        settled_drift == 0
        and len(s_rw.value()[unsafe_offset=0].cluster_lights) == 2
        and len(s_rps.value()[unsafe_offset=0].extra_lights) == 0
        and center_px != UInt32(0xFF000000)
    )

    var rw_s = studio.app.resources.get_ptr_mut[RenderWorld]()
    rw_s.value()[unsafe_offset=0].overlay_lines.add_line(
        Vec3(-1.2, 0.5, 0.0), Vec3(1.2, 0.5, 0.0), pack_rgba8(255, 240, 20, 255), thickness_px=3.0
    )
    studio.render_editor_frame(640, 360)
    var px_ov = studio.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    var studio_ov_cleared = s_rw.value()[unsafe_offset=0].overlay_lines.segment_count() == 0
    var studio_ov_diffs = 0
    var yellow_hits = 0
    for i in range(len(px_settled)):
        if px_settled[i] != px_ov[i]:
            studio_ov_diffs += 1
            var p = px_ov[i]
            var r = Int((p >> 16) & 0xFF)
            var g = Int((p >> 8) & 0xFF)
            var b = Int(p & 0xFF)
            if r >= 220 and g >= 200 and b <= 60:
                yellow_hits += 1

    # Next frame with zero overlay_lines must return to settled baseline (0 drift)
    studio.render_editor_frame(640, 360)
    var px_after = studio.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    var post_drift = 0
    for i in range(len(px_settled)):
        if px_settled[i] != px_after[i]:
            post_drift += 1

    var lgt_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](lgt_e, LocalTransform(Vec3(0.0, 3.0, 0.0)))
    _ = studio.app.world.insert[GlobalTransform](lgt_e, GlobalTransform())
    _ = studio.app.world.insert[PointLightComponent](lgt_e, PointLightComponent(Vec3(6.0, 5.0, 4.0), 12.0, True))
    studio.render_editor_frame(640, 360)
    var scene_light_ok = (
        len(s_rw.value()[unsafe_offset=0].cluster_lights) == 1
        and abs(s_rw.value()[unsafe_offset=0].cluster_lights[0].radiance.x - Float32(6.0)) < Float32(1.0e-5)
    )
    studio.shutdown()
    _ = runner.check(
        "t1_m0_f07_studio_viewport_overlay_and_fallback_lights",
        fallback_ok and studio_ov_cleared and studio_ov_diffs > 20 and yellow_hits > 100 and post_drift == 0 and scene_light_ok,
        "settled_drift=" + String(settled_drift)
        + " ov_diffs=" + String(studio_ov_diffs)
        + " yellow=" + String(yellow_hits)
        + " post_drift=" + String(post_drift),
    )


def run_f08_material_override_isolation(mut runner: StudioSuiteRunner):
    var app = EngineApp(AppConfig(title="e2e_mat_iso", headless=True, worker_threads=0))
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

    var base_intact = abs(rend.value()[unsafe_offset=0].materials[shared_mat_id].base_color.x - Float32(0.5)) < Float32(1.0e-5)
    var rw = app.resources.get_ptr[RenderWorld]()
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

    var iso_ok = (
        base_intact
        and len(rw.value()[unsafe_offset=0].instances) == 3
        and slot_a >= 0
        and slot_b >= 0
        and slot_c == shared_mat_id
        and slot_a != shared_mat_id
        and slot_b != shared_mat_id
        and slot_a != slot_b
    )
    if iso_ok:
        ref mat_a = rend.value()[unsafe_offset=0].materials[slot_a]
        ref mat_b = rend.value()[unsafe_offset=0].materials[slot_b]
        iso_ok = (
            abs(mat_a.base_color.x - Float32(0.9)) < Float32(1.0e-5)
            and abs(mat_a.opacity - Float32(0.75)) < Float32(1.0e-5)
            and abs(mat_a.alpha_cutoff - Float32(0.25)) < Float32(1.0e-5)
            and abs(mat_a.clearcoat - Float32(0.9)) < Float32(1.0e-5)
            and abs(mat_a.clearcoat_roughness - Float32(0.12)) < Float32(1.0e-5)
            and abs(mat_a.sheen_color.x - Float32(0.4)) < Float32(1.0e-5)
            and abs(mat_a.subsurface_amount - Float32(0.3)) < Float32(1.0e-5)
            and abs(mat_b.base_color.z - Float32(0.9)) < Float32(1.0e-5)
            and abs(mat_b.opacity - Float32(0.5)) < Float32(1.0e-5)
            and abs(mat_b.sheen_color.x - Float32(0.8)) < Float32(1.0e-5)
            and abs(mat_b.subsurface_amount - Float32(0.6)) < Float32(1.0e-5)
        )
    _ = runner.check("t1_m0_f08_per_entity_override_isolation_12_fields", iso_ok, "material override leaked or clobbered")

    var total_mats_before = len(rend.value()[unsafe_offset=0].materials)
    app.world.get_ptr_mut[MaterialOverrideComponent](ent_a).value()[unsafe_offset=0].enabled = False
    app.step_one_frame(Float32(1.0 / 60.0))
    var rstate = app.resources.get_ptr[RenderPluginStateResource]()
    var freed_ok = len(rstate.value()[unsafe_offset=0].free_override_slots) == 1 and rstate.value()[unsafe_offset=0].free_override_slots[0] == slot_a

    var ent_d = app.world.spawn()
    _ = app.world.insert[LocalTransform](ent_d, LocalTransform(Vec3(0.0, 2.0, 0.0)))
    _ = app.world.insert[GlobalTransform](ent_d, GlobalTransform())
    _ = app.world.insert[MeshRendererComponent](ent_d, MeshRendererComponent(mesh_id, shared_mat_id))
    _ = app.world.insert[MaterialOverrideComponent](
        ent_d, MaterialOverrideComponent(-1, Vec3(0.2, 0.9, 0.3), 0.3, 0.2, Vec3(), True)
    )
    app.step_one_frame(Float32(1.0 / 60.0))
    var recycled_ok = (
        freed_ok
        and len(rend.value()[unsafe_offset=0].materials) == total_mats_before
        and len(rstate.value()[unsafe_offset=0].free_override_slots) == 0
    )
    app.shutdown()
    _ = runner.check("t1_m0_f08_override_slot_recycling", recycled_ok, "override slot not recycled")

    var orig = MaterialOverrideComponent(
        4, Vec3(0.25, 0.5, 0.75), 0.3, 0.6, Vec3(0.1, 0.2, 0.3), True, 0.85, 0.15, 0.7, 0.09, 0.45, 0.35
    )
    var bw = BlobWriter()
    orig.write_bin(bw)
    var codec_ok: Bool
    try:
        var br = BlobReader(bw.data.copy())
        var dec2 = MaterialOverrideComponent.read_bin(br, MaterialOverrideComponent.SCHEMA_VERSION)
        var bw1 = BlobWriter()
        bw1.integer(2)
        bw1.vec3(Vec3(0.4, 0.5, 0.6))
        bw1.f32(0.4)
        bw1.f32(0.1)
        bw1.vec3(Vec3())
        bw1.boolean(True)
        var br1 = BlobReader(bw1.data.copy())
        var dec1 = MaterialOverrideComponent.read_bin(br1, 1)
        var jstr = orig.write_json()
        var jbytes = List[UInt8](capacity=jstr.byte_length())
        for b in jstr.as_bytes():
            jbytes.append(b)
        var jdoc = parse_json(jbytes)
        var dec_j = MaterialOverrideComponent.read_json(jdoc, 0)
        codec_ok = (
            dec2.material_id == 4
            and abs(dec2.opacity - Float32(0.85)) < Float32(1.0e-5)
            and abs(dec2.clearcoat - Float32(0.7)) < Float32(1.0e-5)
            and abs(dec2.sheen - Float32(0.45)) < Float32(1.0e-5)
            and abs(dec2.subsurface - Float32(0.35)) < Float32(1.0e-5)
            and dec1.material_id == 2
            and abs(dec1.opacity - Float32(1.0)) < Float32(1.0e-5)
            and abs(dec_j.opacity - Float32(0.85)) < Float32(1.0e-5)
            and abs(dec_j.subsurface - Float32(0.35)) < Float32(1.0e-5)
        )
    except:
        codec_ok = False
    _ = runner.check("t1_m0_f08_material_override_v2_v1_json_codecs", codec_ok, "codec round-trip mismatch")

    var studio_vk = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_1_FORWARD_CLUSTERED,
        width=160,
        height=120,
        worker_threads=0,
    )
    var mat0 = studio_vk.app.resources.get_ptr_mut[ScalableRenderer]().value()[unsafe_offset=0].register_material(
        MaterialAsset(base_color=Vec3(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0)
    )
    var plain_e = studio_vk.app.world.spawn()
    _ = studio_vk.app.world.insert[LocalTransform](plain_e, LocalTransform(Vec3(-1.5, 1.0, 0.0)))
    _ = studio_vk.app.world.insert[GlobalTransform](plain_e, GlobalTransform())
    _ = studio_vk.app.world.insert[PrevGlobalTransform](plain_e, PrevGlobalTransform())
    _ = studio_vk.app.world.insert[MeshRendererComponent](plain_e, MeshRendererComponent(0, mat0))

    var orb_id = studio_vk.spawn_entity("Orb_0", position=Vec3(1.5, 1.0, 0.0), base_color=Vec3(0.1, 0.7, 0.2), record_undo=False)
    var orb_e = studio_vk.resolve_entity(orb_id)
    studio_vk.app.world.get_ptr_mut[MeshRendererComponent](orb_e).value()[unsafe_offset=0].material_id = mat0
    studio_vk.app.world.get_ptr_mut[MaterialOverrideComponent](orb_e).value()[unsafe_offset=0].material_id = mat0

    studio_vk.render_editor_frame(160, 120)
    var pie_ok = studio_vk.pie_play()
    _ = studio_vk.step_pie_frame(Float32(1.0 / 60.0))
    studio_vk.render_editor_frame(160, 120)
    try:
        pie_ok = pie_ok and studio_vk.pie_stop()
    except:
        pie_ok = False
    studio_vk.render_editor_frame(160, 120)
    var base_r = studio_vk.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].materials[mat0].base_color.x
    var gen_ok = studio_vk.gpu_materials.entity_generations[orb_id] == studio_vk.app.world.entities[orb_id].generation
    var despawn_ok = studio_vk.despawn_entity(orb_id, record_undo=False) and studio_vk.gpu_materials.entity_pbr_mat_ids[orb_id] == -1
    studio_vk.shutdown()
    _ = runner.check(
        "t1_m0_f08_studio_shared_mat_and_gpu_gen_lifecycle",
        pie_ok and abs(base_r - Float32(0.5)) < Float32(1.0e-5) and gen_ok and despawn_ok,
        "Studio shared mat clobbered or generation mismatch",
    )


def run_f09_ui_capture_widgets_and_esc_guard(mut runner: StudioSuiteRunner):
    # A. Non-text auto-focus never triggers wants_keyboard; text_input alone never auto-focuses; Tab/Shift+Tab updates same frame
    var ui = UiContext()
    var inp = InputState()
    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.button("btn_ok", "OK", 100.0, 28.0)
    ui.end_frame()
    var btn_autofocus_ok = ui.focused_id != 0 and (not ui.wants_keyboard()) and (not ui.wants_mouse())

    var ui_tab = UiContext()
    var tab_txt = String("Name")
    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui_tab.begin_frame(inp, 1000.0, 700.0)
    _ = ui_tab.button("btn_first", "First", 100.0, 28.0)
    _ = ui_tab.text_input("txt_second", tab_txt, 200.0)
    ui_tab.end_frame()
    var tab_ok = not ui_tab.wants_keyboard()

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_TAB, True)
    ui_tab.begin_frame(inp, 1000.0, 700.0)
    _ = ui_tab.button("btn_first", "First", 100.0, 28.0)
    _ = ui_tab.text_input("txt_second", tab_txt, 200.0)
    ui_tab.end_frame()
    inp.set_logical_key(UI_KEY_TAB, False)
    tab_ok = tab_ok and ui_tab.focused_id == ui_tab.widget_id("txt_second") and ui_tab.wants_keyboard()

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_LEFT_SHIFT, True)
    inp.set_logical_key(UI_KEY_TAB, True)
    ui_tab.begin_frame(inp, 1000.0, 700.0)
    _ = ui_tab.button("btn_first", "First", 100.0, 28.0)
    _ = ui_tab.text_input("txt_second", tab_txt, 200.0)
    ui_tab.end_frame()
    inp.set_logical_key(UI_KEY_TAB, False)
    inp.set_logical_key(UI_KEY_LEFT_SHIFT, False)
    tab_ok = tab_ok and ui_tab.focused_id == ui_tab.widget_id("btn_first") and (not ui_tab.wants_keyboard())

    # Solo text_input never auto-focuses, focuses on click at last_item_rect, blurs on Esc & stays blurred
    var ui_solo = UiContext()
    var solo_txt = String("Solo")
    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui_solo.begin_frame(inp, 1000.0, 700.0)
    _ = ui_solo.text_input("solo_in", solo_txt, 200.0)
    var solo_r = ui_solo.last_item_rect
    ui_solo.end_frame()
    var solo_ok = ui_solo.focused_id == UInt64(0) and (not ui_solo.wants_keyboard())

    inp.begin_frame()
    inp.move_mouse(solo_r.center_x(), solo_r.center_y())
    inp.set_mouse(0, True)
    ui_solo.begin_frame(inp, 1000.0, 700.0)
    _ = ui_solo.text_input("solo_in", solo_txt, 200.0)
    ui_solo.end_frame()
    solo_ok = solo_ok and ui_solo.wants_keyboard() and ui_solo.wants_mouse() and ui_solo.wants_mouse_at(solo_r.center_x(), solo_r.center_y()) and ui_solo.wants_mouse_at(800.0, 500.0)

    inp.begin_frame()
    inp.set_mouse(0, False)
    inp.set_logical_key(UI_KEY_ESCAPE, True)
    ui_solo.begin_frame(inp, 1000.0, 700.0)
    _ = ui_solo.text_input("solo_in", solo_txt, 200.0)
    ui_solo.end_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, False)
    solo_ok = solo_ok and (not ui_solo.wants_keyboard())

    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui_solo.begin_frame(inp, 1000.0, 700.0)
    _ = ui_solo.text_input("solo_in", solo_txt, 200.0)
    ui_solo.end_frame()
    solo_ok = solo_ok and ui_solo.focused_id == UInt64(0) and (not ui_solo.wants_keyboard())
    _ = runner.check("t1_m0_f09_wants_keyboard_tab_esc_and_autofocus", btn_autofocus_ok and tab_ok and solo_ok, "text focus/wants_keyboard invariant broken")

    # B. DockSpace DOCK_TABS captures mouse while DOCK_CENTRAL_VIEWPORT does not
    var dock = DockSpace()
    var vp = dock.add_central_viewport("SceneViewport")
    _ = dock.split_node(vp, DOCK_SPLIT_H, 0.25, [String("Hierarchy")], place_new_first=True)
    dock.solve_layout(RectF(0.0, 0.0, 1000.0, 700.0))
    var h_panel = dock.find_panel("Hierarchy")
    var vp_panel = dock.find_panel("SceneViewport")

    inp.begin_frame()
    inp.move_mouse(h_panel[3].center_x(), h_panel[3].center_y())
    ui.begin_frame(inp, 1000.0, 700.0)
    ui.update_dock_space(dock, RectF(0.0, 0.0, 1000.0, 700.0))
    ui.end_frame()
    var dock_cap_ok = ui.wants_mouse()

    inp.begin_frame()
    inp.move_mouse(vp_panel[3].center_x(), vp_panel[3].center_y())
    ui.begin_frame(inp, 1000.0, 700.0)
    ui.update_dock_space(dock, RectF(0.0, 0.0, 1000.0, 700.0))
    ui.end_frame()
    dock_cap_ok = dock_cap_ok and (not ui.wants_mouse())
    _ = runner.check("t1_m0_f09_dockspace_tabs_vs_viewport_wants_mouse", dock_cap_ok, "DockSpace mouse capture mismatch")

    # C. M0 widgets (drag_float + Shift, combo_box + popup click isolation, color_edit3 + popup click isolation)
    var ui_w = UiContext()
    var dv = Float32(1.0)
    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui_w.begin_frame(inp, 1000.0, 700.0)
    _ = ui_w.drag_float("df_x", "Roughness", dv, speed=0.1, min_val=0.0, max_val=2.0)
    var df_r = ui_w.last_item_rect
    ui_w.end_frame()

    inp.begin_frame()
    inp.move_mouse(df_r.x + Float32(30.0), df_r.center_y())
    inp.set_mouse(0, True)
    ui_w.begin_frame(inp, 1000.0, 700.0)
    _ = ui_w.drag_float("df_x", "Roughness", dv, speed=0.1, min_val=0.0, max_val=2.0)
    ui_w.end_frame()

    inp.begin_frame()
    inp.move_mouse(df_r.x + Float32(35.0), df_r.center_y())
    ui_w.begin_frame(inp, 1000.0, 700.0)
    var df_ch1 = ui_w.drag_float("df_x", "Roughness", dv, speed=0.1, min_val=0.0, max_val=2.0)
    ui_w.end_frame()
    var drag_ok = df_ch1 and abs(dv - Float32(1.5)) < Float32(1.0e-4)

    inp.begin_frame()
    inp.set_logical_key(UI_KEY_LEFT_SHIFT, True)
    inp.move_mouse(df_r.x + Float32(40.0), df_r.center_y())
    ui_w.begin_frame(inp, 1000.0, 700.0)
    var df_ch2 = ui_w.drag_float("df_x", "Roughness", dv, speed=0.1, min_val=0.0, max_val=2.0)
    ui_w.end_frame()
    inp.set_logical_key(UI_KEY_LEFT_SHIFT, False)
    inp.set_mouse(0, False)
    drag_ok = drag_ok and df_ch2 and abs(dv - Float32(1.6)) < Float32(1.0e-4)

    # combo_box + click bleed-through test using public last_item_rect (no _popup_rect!)
    var ui_pop = UiContext()
    var pop_items = List[String]()
    pop_items.append(String("Alpha"))
    pop_items.append(String("Beta"))
    pop_items.append(String("Gamma"))
    var pop_sel = 0
    var under_clicked = False
    var under_checked = False

    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui_pop.begin_frame(inp, 1000.0, 700.0)
    _ = ui_pop.combo_box("cmb_top", "Mode", pop_sel, pop_items, 220.0)
    var cmb_r = ui_pop.last_item_rect
    if ui_pop.button("btn_under", "Under", 200.0, 28.0):
        under_clicked = True
    _ = ui_pop.checkbox("chk_under", "CheckUnder", under_checked)
    ui_pop.end_frame()

    inp.begin_frame()
    inp.move_mouse(cmb_r.x + Float32(25.0), cmb_r.center_y())
    inp.set_mouse(0, True)
    ui_pop.begin_frame(inp, 1000.0, 700.0)
    _ = ui_pop.combo_box("cmb_top", "Mode", pop_sel, pop_items, 220.0)
    if ui_pop.button("btn_under", "Under", 200.0, 28.0):
        under_clicked = True
    _ = ui_pop.checkbox("chk_under", "CheckUnder", under_checked)
    ui_pop.end_frame()
    var cmb_open_ok = ui_pop.wants_keyboard() and ui_pop.wants_mouse()

    inp.begin_frame()
    inp.set_mouse(0, False)
    inp.set_logical_key(UI_KEY_DOWN, True)
    ui_pop.begin_frame(inp, 1000.0, 700.0)
    var cmb_key_ok = ui_pop.combo_box("cmb_top", "Mode", pop_sel, pop_items, 220.0)
    if ui_pop.button("btn_under", "Under", 200.0, 28.0):
        under_clicked = True
    _ = ui_pop.checkbox("chk_under", "CheckUnder", under_checked)
    ui_pop.end_frame()
    inp.set_logical_key(UI_KEY_DOWN, False)
    cmb_open_ok = cmb_open_ok and cmb_key_ok and pop_sel == 1

    # Click item 2 inside open combo popup directly over underlying button/checkbox
    var item2_y = cmb_r.bottom() + Float32(2.0) + ui_pop.theme.row_height * Float32(2.5)
    inp.begin_frame()
    inp.move_mouse(cmb_r.x + Float32(25.0), item2_y)
    inp.set_mouse(0, True)
    ui_pop.begin_frame(inp, 1000.0, 700.0)
    var cmb_click_ok = ui_pop.combo_box("cmb_top", "Mode", pop_sel, pop_items, 220.0)
    if ui_pop.button("btn_under", "Under", 200.0, 28.0):
        under_clicked = True
    _ = ui_pop.checkbox("chk_under", "CheckUnder", under_checked)
    ui_pop.end_frame()

    inp.begin_frame()
    inp.set_mouse(0, False)
    ui_pop.begin_frame(inp, 1000.0, 700.0)
    _ = ui_pop.combo_box("cmb_top", "Mode", pop_sel, pop_items, 220.0)
    if ui_pop.button("btn_under", "Under", 200.0, 28.0):
        under_clicked = True
    _ = ui_pop.checkbox("chk_under", "CheckUnder", under_checked)
    ui_pop.end_frame()
    var cmb_no_bleed = cmb_open_ok and cmb_click_ok and pop_sel == 2 and (not ui_pop.wants_keyboard()) and (not under_clicked) and (not under_checked)

    # color_edit3 + popup click isolation using last_item_rect
    var ui_col = UiContext()
    var col_val = Vec3(0.2, 0.4, 0.6)
    var col_under_clicked = False
    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui_col.begin_frame(inp, 1000.0, 700.0)
    _ = ui_col.color_edit3("col_top", "Tint", col_val)
    var col_r = ui_col.last_item_rect
    if ui_col.button("btn_under_col", "UnderCol", 200.0, 28.0):
        col_under_clicked = True
    ui_col.end_frame()

    inp.begin_frame()
    inp.move_mouse(col_r.x + Float32(8.0), col_r.center_y())
    inp.set_mouse(0, True)
    ui_col.begin_frame(inp, 1000.0, 700.0)
    _ = ui_col.color_edit3("col_top", "Tint", col_val)
    if ui_col.button("btn_under_col", "UnderCol", 200.0, 28.0):
        col_under_clicked = True
    ui_col.end_frame()
    var col_open_ok = ui_col.wants_keyboard()

    inp.begin_frame()
    inp.set_mouse(0, False)
    ui_col.begin_frame(inp, 1000.0, 700.0)
    _ = ui_col.color_edit3("col_top", "Tint", col_val)
    if ui_col.button("btn_under_col", "UnderCol", 200.0, 28.0):
        col_under_clicked = True
    ui_col.end_frame()

    inp.begin_frame()
    inp.move_mouse(col_r.x + Float32(60.0), col_r.bottom() + Float32(12.0))
    inp.set_mouse(0, True)
    ui_col.begin_frame(inp, 1000.0, 700.0)
    var col_changed = ui_col.color_edit3("col_top", "Tint", col_val)
    if ui_col.button("btn_under_col", "UnderCol", 200.0, 28.0):
        col_under_clicked = True
    ui_col.end_frame()

    inp.begin_frame()
    inp.set_mouse(0, False)
    inp.set_logical_key(UI_KEY_ESCAPE, True)
    ui_col.begin_frame(inp, 1000.0, 700.0)
    _ = ui_col.color_edit3("col_top", "Tint", col_val)
    if ui_col.button("btn_under_col", "UnderCol", 200.0, 28.0):
        col_under_clicked = True
    ui_col.end_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, False)
    var col_ok = col_open_ok and col_changed and (not col_under_clicked) and (not ui_col.wants_keyboard())
    _ = runner.check("t1_m0_f09_m0_widgets_and_popup_bleed_isolation", drag_ok and cmb_no_bleed and col_ok, "widget or popup isolation failed")

    # D. Studio Esc quit guard + text_input keyboard capture gate via public UiContext
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var e0 = studio.spawn_entity("GuardTarget", position=Vec3(0.0, 1.0, 0.0), record_undo=False)
    studio.select_entity(e0)

    var s_inp = InputState()
    s_inp.begin_frame()
    s_inp.set_logical_key(UI_KEY_ESCAPE, True)
    var esc_stay_open = studio.process_editor_input(s_inp, Float32(1.0 / 60.0))
    s_inp.set_logical_key(UI_KEY_ESCAPE, False)

    var stxt = String("EditMe")
    var ui_drv = InputState()
    ui_drv.begin_frame()
    ui_drv.move_mouse(800.0, 500.0)
    studio.ui_ctx.begin_frame(ui_drv, 1000.0, 700.0)
    _ = studio.ui_ctx.text_input("studio_guard_ti", stxt, 200.0)
    var sti_r = studio.ui_ctx.last_item_rect
    studio.ui_ctx.end_frame()

    ui_drv.begin_frame()
    ui_drv.move_mouse(sti_r.center_x(), sti_r.center_y())
    ui_drv.set_mouse(0, True)
    studio.ui_ctx.begin_frame(ui_drv, 1000.0, 700.0)
    _ = studio.ui_ctx.text_input("studio_guard_ti", stxt, 200.0)
    studio.ui_ctx.end_frame()
    ui_drv.set_mouse(0, False)
    var captured_ok = studio.ui_ctx.wants_keyboard()

    s_inp.begin_frame()
    s_inp.set_logical_key(69, True)   # E
    s_inp.set_logical_key(261, True)  # Del
    s_inp.set_logical_key(294, True)  # F5
    s_inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
    s_inp.set_logical_key(81, True)   # Ctrl+Q
    var blocked_ok = (
        studio.process_editor_input(s_inp, Float32(1.0 / 60.0))
        and studio.gizmo_mode == GIZMO_MODE_TRANSLATE
        and studio.is_alive_id(e0)
        and studio.pie.mode == PIE_STOPPED
    )
    s_inp.begin_frame()
    s_inp.set_logical_key(69, False)
    s_inp.set_logical_key(261, False)
    s_inp.set_logical_key(294, False)
    s_inp.set_logical_key(UI_KEY_LEFT_CTRL, False)
    s_inp.set_logical_key(81, False)
    _ = studio.process_editor_input(s_inp, Float32(1.0 / 60.0))

    ui_drv.begin_frame()
    ui_drv.set_logical_key(UI_KEY_ESCAPE, True)
    studio.ui_ctx.begin_frame(ui_drv, 1000.0, 700.0)
    _ = studio.ui_ctx.text_input("studio_guard_ti", stxt, 200.0)
    studio.ui_ctx.end_frame()
    ui_drv.set_logical_key(UI_KEY_ESCAPE, False)
    var unblur_ok = not studio.ui_ctx.wants_keyboard()

    s_inp.begin_frame()
    s_inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
    s_inp.set_logical_key(81, True)
    var ctrl_q_quits = not studio.process_editor_input(s_inp, Float32(1.0 / 60.0))
    studio.shutdown()
    _ = runner.check(
        "t1_m0_f09_studio_esc_guard_and_capture_barrier",
        esc_stay_open and captured_ok and blocked_ok and unblur_ok and ctrl_q_quits,
        "esc=" + String(esc_stay_open) + " cap=" + String(captured_ok) + " blk=" + String(blocked_ok) + " unb=" + String(unblur_ok) + " quit=" + String(ctrl_q_quits),
    )


def run_f10_kcc_triggers_and_edit_camera(mut runner: StudioSuiteRunner):
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_trigger(TriggerVolume.obb(10, Vec3(2.0, 1.0, 0.0), Vec3(0.75, 1.5, 0.75)))
    _ = pw.add_trigger(TriggerVolume.sphere(11, Vec3(5.0, 1.0, 0.0), 0.9))
    _ = pw.add_trigger(TriggerVolume.capsule(12, Vec3(8.0, 1.0, 0.0), 0.7, 1.0))
    _ = pw.add_trigger(
        TriggerVolume.obb(13, Vec3(2.0, 1.0, 0.0), Vec3(0.75, 1.5, 0.75), filter=CollisionFilter(CAT_SENSOR, CAT_PROP_PIECE, 0))
    )

    var cfg = CharacterControllerConfig()
    cfg.gravity = 0.0
    var kcc = CharacterController(Vec3(0.0, 1.0, 0.0), cfg)
    var kcc_id = pw.register_kcc_proxy(kcc.pos, kcc.cfg.radius, kcc.cfg.half_height, kcc.cfg.up, kcc.cfg.filter)
    var no_body_pollute = len(pw.bodies) == 0 and pw.find_slot_by_id(kcc_id) == -1

    var saw_obb_enter = False
    var saw_obb_stay = False
    var saw_obb_exit = False
    var saw_sph_enter = False
    var saw_cap_enter = False
    var saw_masked = False
    for _ in range(95):
        var p = kcc.move_and_slide(pw, Vec3(6.0, 0.0, 0.0), Float32(1.0 / 60.0))
        _ = pw.update_kcc_proxy(kcc_id, p)
        pw.step(Float32(1.0 / 60.0))
        for ti in range(len(pw.trigger_events)):
            var ev = pw.trigger_events[ti]
            if ev.trigger_id == 10 and ev.body_id == kcc_id:
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

    var phyw_rt_ok: Bool
    var bw = BlobWriter()
    pw.serialize(bw)
    try:
        var pw2 = PhysicsWorld()
        var br = BlobReader(bw.data.copy())
        pw2.deserialize(br)
        phyw_rt_ok = len(pw2.kcc_proxies) == 1 and pw2.kcc_proxies[0].id == kcc_id
    except:
        phyw_rt_ok = False

    _ = runner.check(
        "t1_m0_f10_kcc_triggers_shapes_filter_and_phyw_v3",
        no_body_pollute and saw_obb_enter and saw_obb_stay and saw_obb_exit and saw_sph_enter and saw_cap_enter and (not saw_masked) and kcc.pos.x > Float32(8.5) and phyw_rt_ok,
        "KCC trigger overlap or PHYW v3 round-trip failed",
    )

    var app = EngineApp(AppConfig(title="e2e_kcc_ecs", headless=True, worker_threads=0))
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
    var lookup_ok = False
    for _ in range(30):
        app.step_one_frame(Float32(1.0 / 60.0))
        var tevs = app.resources.get_ptr[TriggerEventsResource]().value()[unsafe_offset=0].events.read_since(tev_cursor)
        ref sync = app.resources.get_ptr[PhysicsSyncStateResource]().value()[unsafe_offset=0]
        for ti in range(len(tevs)):
            var ev = tevs[ti]
            if ev.kind == TRIGGER_ENTER:
                ecs_enter = True
                if sync.lookup_entity(ev.body_id) == kcc_e:
                    lookup_ok = True
            elif ev.kind == TRIGGER_STAY:
                ecs_stay = True
        if ecs_enter and ecs_stay:
            break

    var despawned = app.world.despawn(kcc_e)
    app.step_one_frame(Float32(1.0 / 60.0))
    var exit_tevs = app.resources.get_ptr[TriggerEventsResource]().value()[unsafe_offset=0].events.read_since(tev_cursor)
    var saw_exit = False
    for ti in range(len(exit_tevs)):
        if exit_tevs[ti].kind == TRIGGER_EXIT:
            saw_exit = True
    var proxies_empty = len(app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0].kcc_proxies) == 0
    app.shutdown()
    _ = runner.check(
        "t1_m0_f10_ecs_kcc_trigger_lookup_and_despawn_exit",
        ecs_enter and ecs_stay and lookup_ok and despawned and saw_exit and proxies_empty,
        "ECS KCC trigger or despawn TRIGGER_EXIT failed",
    )

    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0)
    studio.look_at(Vec3(0.0, 2.0, -5.0), Vec3(0.0, 1.0, 0.0))
    var cam_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(50.0, 20.0, 50.0)))
    _ = studio.app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
    _ = studio.app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=65.0, active=True))

    studio.render_editor_frame(160, 120)
    var rw = studio.app.resources.get_ptr[RenderWorld]()
    var edit_cam_ok = (
        abs(rw.value()[unsafe_offset=0].camera.eye.x - Float32(0.0)) < Float32(1.0e-3)
        and abs(rw.value()[unsafe_offset=0].camera.eye.y - Float32(2.0)) < Float32(1.0e-3)
        and abs(rw.value()[unsafe_offset=0].camera.eye.z - Float32(-5.0)) < Float32(1.0e-3)
    )
    var p_ok = studio.pie_play() and studio.pie_pause()
    studio.render_editor_frame(160, 120)
    var paused_cam_ok = abs(rw.value()[unsafe_offset=0].camera.eye.x - Float32(0.0)) < Float32(1.0e-3)

    p_ok = p_ok and studio.pie_play()
    _ = studio.step_pie_frame(Float32(1.0 / 60.0))
    studio.render_editor_frame(160, 120)
    var pie_cam_ok = (
        abs(rw.value()[unsafe_offset=0].camera.eye.x - Float32(50.0)) < Float32(1.0e-3)
        and abs(rw.value()[unsafe_offset=0].camera.eye.y - Float32(20.0)) < Float32(1.0e-3)
        and abs(rw.value()[unsafe_offset=0].camera.eye.z - Float32(50.0)) < Float32(1.0e-3)
    )
    var stopped_ok: Bool
    try:
        stopped_ok = studio.pie_stop()
    except:
        stopped_ok = False
    studio.render_editor_frame(160, 120)
    var restore_cam_ok = (
        stopped_ok
        and abs(rw.value()[unsafe_offset=0].camera.eye.x - Float32(0.0)) < Float32(1.0e-3)
        and abs(rw.value()[unsafe_offset=0].camera.eye.z - Float32(-5.0)) < Float32(1.0e-3)
    )
    var stepped_ok = studio.pie_step(1, Float32(1.0 / 60.0)) == 1 and studio.pie.mode == PIE_PAUSED
    var rps = studio.app.resources.get_ptr[RenderPluginStateResource]()
    var step_override_ok = Bool(rps) and rps.value()[unsafe_offset=0].editor_camera_override
    studio.shutdown()
    _ = runner.check(
        "t1_m0_f10_edit_camera_isolation_across_edit_pause_play_step_stop",
        edit_cam_ok and p_ok and paused_cam_ok and pie_cam_ok and restore_cam_ok and stepped_ok and step_override_ok,
        "Edit camera isolation failed",
    )


def run_f11_dockspace_and_undo(mut runner: StudioSuiteRunner):
    var d1 = DockSpace.create_studio_5panel()
    _ = d1.activate_tab("Outliner")
    d1.solve_layout(RectF(0.0, 0.0, 1280.0, 720.0))
    var saved = d1.save_json()

    var d2 = DockSpace()
    var loaded = d2.load_json(saved)
    d2.solve_layout(RectF(0.0, 0.0, 1280.0, 720.0))
    var h1 = d1.find_panel("Outliner")
    var h2 = d2.find_panel("Outliner")
    var v1 = d1.find_panel("SceneViewport")
    var v2 = d2.find_panel("SceneViewport")
    var dock_ok = (
        loaded
        and h1[0] == h2[0]
        and h2[2]
        and abs(h1[3].w - h2[3].w) < Float32(0.5)
        and abs(v1[3].w - v2[3].w) < Float32(0.5)
    )
    _ = runner.check("t1_m0_f11_dockspace_json_persistence", dock_ok, "DockSpace JSON round-trip mismatch")

    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var h_init = studio.state_hash()
    var e1 = studio.spawn_entity("Hero", position=Vec3(1.0, 2.0, 3.0), record_undo=True)
    var h_after_spawn = studio.state_hash()
    var moved = studio.set_entity_transform(e1, Vec3(4.0, 5.0, 6.0), Quat(0.0, 0.0, 0.0, 1.0), Vec3(1.0, 1.0, 1.0))
    var h_after_move = studio.state_hash()
    var dup_id = studio.duplicate_entity(e1, record_undo=True)
    var h_after_dup = studio.state_hash()

    var undo_ok = (
        e1 >= 0
        and moved
        and dup_id >= 0
        and studio.live_entity_count() == 2
        and studio.undo()
        and studio.state_hash() == h_after_move
        and studio.undo()
        and studio.state_hash() == h_after_spawn
        and studio.undo()
        and studio.state_hash() == h_init
        and studio.redo()
        and studio.state_hash() == h_after_spawn
        and studio.redo()
        and studio.state_hash() == h_after_move
        and studio.redo()
        and studio.state_hash() == h_after_dup
    )
    studio.shutdown()
    _ = runner.check("t1_m0_f11_studio_undo_redo_hash_algebra", undo_ok, "Undo/redo hash algebra mismatch")


def run_f12_inspector_fields_and_roundtrip(mut runner: StudioSuiteRunner, tmp: String):
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=320,
        height=180,
        worker_threads=0,
    )
    var eid = studio.spawn_entity("InspTarget", position=Vec3(1.0, 2.0, 3.0), mass=2.5, record_undo=True)
    var e = studio.resolve_entity(eid)
    var col_ptr = studio.app.world.get_ptr_mut[ColliderComponent](e)
    if col_ptr:
        col_ptr.value()[unsafe_offset=0].is_trigger = True

    var view = studio.entities[eid].copy()
    view.name = "RenamedCapsule"
    view.collider_kind = COLLIDER_CAPSULE
    view.collider_half_extents = Vec3(0.45, 1.25, 0.45)
    view.friction = 0.85
    view.restitution = 0.65
    view.emissive = Vec3(1.2, 0.5, 0.1)
    view.clearcoat = 0.8
    view.sheen = 0.5
    view.subsurface = 0.35
    view.opacity = 0.7
    var committed = studio.commit_node_edit(eid, view^, "F12 Inspector Edit")

    var sc_j = tmp + "/f12_insp.flscn"
    var sc_b = tmp + "/f12_insp.flscnb"
    var saved_j = studio.save_scene_file(sc_j)
    var saved_b = studio.save_scene_file(sc_b)
    studio.shutdown()

    var roundtrip_ok = committed and saved_j and saved_b
    var paths = [sc_j, sc_b]
    for pi in range(len(paths)):
        var st = FlintStudioState(
            headless=True,
            render_tier=RENDER_TIER_0_HEADLESS_SIMD,
            width=320,
            height=180,
            worker_threads=0,
        )
        var loaded = st.load_scene_file(paths[pi])
        var found_id = -1
        for i in range(len(st.entities)):
            if st.entities[i].alive and st.entities[i].name == "RenamedCapsule":
                found_id = i
                break
        if (not loaded) or found_id < 0:
            roundtrip_ok = False
        else:
            ref rv = st.entities[found_id]
            var re = st.resolve_entity(found_id)
            var rcol = st.app.world.get_ptr[ColliderComponent](re)
            var rrb = st.app.world.get_ptr[RigidBodyComponent](re)
            var rmo = st.app.world.get_ptr[MaterialOverrideComponent](re)
            var fields_ok = (
                rv.name == "RenamedCapsule"
                and rv.collider_kind == COLLIDER_CAPSULE
                and abs(rv.collider_half_extents.x - Float32(0.45)) < Float32(1.0e-4)
                and abs(rv.collider_half_extents.y - Float32(1.25)) < Float32(1.0e-4)
                and abs(rv.friction - Float32(0.85)) < Float32(1.0e-4)
                and abs(rv.restitution - Float32(0.65)) < Float32(1.0e-4)
                and abs(rv.emissive.x - Float32(1.2)) < Float32(1.0e-4)
                and abs(rv.clearcoat - Float32(0.8)) < Float32(1.0e-4)
                and abs(rv.sheen - Float32(0.5)) < Float32(1.0e-4)
                and abs(rv.subsurface - Float32(0.35)) < Float32(1.0e-4)
                and abs(rv.opacity - Float32(0.7)) < Float32(1.0e-4)
                and Bool(rcol)
                and rcol.value()[unsafe_offset=0].shape == BODY_SHAPE_CAPSULE
                and rcol.value()[unsafe_offset=0].is_trigger
                and Bool(rrb)
                and abs(rrb.value()[unsafe_offset=0].friction - Float32(0.85)) < Float32(1.0e-4)
                and abs(rrb.value()[unsafe_offset=0].restitution - Float32(0.65)) < Float32(1.0e-4)
                and Bool(rmo)
                and abs(rmo.value()[unsafe_offset=0].emissive.x - Float32(1.2)) < Float32(1.0e-4)
                and abs(rmo.value()[unsafe_offset=0].clearcoat - Float32(0.8)) < Float32(1.0e-4)
                and abs(rmo.value()[unsafe_offset=0].sheen - Float32(0.5)) < Float32(1.0e-4)
                and abs(rmo.value()[unsafe_offset=0].subsurface - Float32(0.35)) < Float32(1.0e-4)
                and abs(rmo.value()[unsafe_offset=0].opacity - Float32(0.7)) < Float32(1.0e-4)
            )
            if not fields_ok:
                roundtrip_ok = False
        st.shutdown()

    _ = runner.check(
        "t1_m0_f12_inspector_fields_and_roundtrip",
        roundtrip_ok,
        "F12 Inspector fields commit_node_edit or .flscn/.flscnb round-trip failed",
    )


def run_f14_anim_audio_plugins_and_codecs(mut runner: StudioSuiteRunner, tmp: String):
    var wav_path = tmp + "/tone.wav"
    var wav_written = write_minimal_wav(wav_path, 0.05, 48000, 2)
    var wav = decode_wav(read_bytes(wav_path))
    var qoa_bytes = encode_qoa(wav.samples, wav.channels, wav.sample_rate)
    var qoa_asset = QoaAsset.from_bytes(qoa_bytes.copy())

    var reg = ComponentRegistry.with_builtins()
    register_anim_codecs(reg)
    var codecs_ok = (
        reg.find_by_name("AnimatorComponent") >= 0
        and reg.find_by_name("SkeletonComponent") >= 0
    )

    var app = EngineApp(AppConfig(title="e2e_anim_audio", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(AnimPlugin())
    app.startup()

    var has_resources = app.resources.has[AnimAssetCache]()
    var srv = AudioServer(sample_rate=48000, quantum_frames=256, ring_capacity=512)
    var aid_wav = srv.register_wav_asset(wav)
    var aid_qoa = srv.register_qoa_asset(qoa_bytes^)
    var h_snd = srv.play_2d(aid_wav, volume=0.6, bus_id=BUS_SFX, looping=True)
    var stopped = srv.stop_sound(h_snd)
    srv.stop()

    var e = app.world.spawn()
    _ = app.world.insert[LocalTransform](e, LocalTransform(Vec3(1.0, 0.0, 2.0)))
    _ = app.world.insert[GlobalTransform](e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](e, PrevGlobalTransform())
    _ = app.world.insert[AnimatorComponent](e, AnimatorComponent(graph_id=UInt64(7)))
    app.step_one_frame(Float32(1.0 / 60.0))

    var sc = Scene("anim_rt")
    var scene_ok: Bool
    try:
        var j_bytes = sc.save_flscn_bytes(app.world, reg)
        var b_bytes = sc.save_flscnb_bytes(app.world, reg)
        var w2 = World()
        reg.ensure_registered_in_world(w2)
        var n_j = sc.load_flscn_bytes(w2, reg, j_bytes, clear_existing=True)
        var w3 = World()
        reg.ensure_registered_in_world(w3)
        var n_b = sc.load_flscnb_bytes(w3, reg, b_bytes, clear_existing=True)
        scene_ok = len(n_j) == 1 and len(n_b) == 1 and w2.entity_count() == 1 and w3.entity_count() == 1
    except:
        scene_ok = False
    app.shutdown()

    _ = runner.check(
        "t1_m0_f14_anim_audio_plugins_qoa_wav_and_scene_codecs",
        wav_written and wav.valid and qoa_asset.valid and codecs_ok and has_resources and aid_wav >= 0 and aid_qoa >= 0 and stopped and scene_ok,
        "Anim plugin or AudioServer WAV/QOA codec check failed",
    )


def run_pending_m3_m0_runtime_checks(mut runner: StudioSuiteRunner, tmp: String):
    # 1. F11: StudioPlay blocks Esc and Del in PIE
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var e0 = studio.spawn_entity("PieProtected", position=Vec3(0.0, 1.0, 0.0), record_undo=False)
    studio.select_entity(e0)
    _ = studio.pie_play()
    var has_studio_play = False
    for i in range(len(studio.input_map.active_stack)):
        if studio.input_map.active_stack[i] == "StudioPlay":
            has_studio_play = True
    var inp = InputState()
    inp.begin_frame()
    inp.set_logical_key(261, True)  # Delete
    inp.set_logical_key(256, True)  # Esc
    var stay_open = studio.process_editor_input(inp, Float32(1.0 / 60.0))
    var still_alive = studio.is_alive_id(e0)
    studio.shutdown()
    _ = runner.check(
        "t1_m0_f11_pending_studio_play_blocks_esc_del_in_pie",
        has_studio_play and stay_open and still_alive,
        "StudioPlay context not active or Delete deleted entity in PIE",
    )

    # 2a. F11: M0 hotkeys (F frame, Ctrl+Shift+Z redo)
    var studio_hk = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var eh = studio_hk.spawn_entity("FrameTarget", position=Vec3(20.0, 5.0, 0.0), record_undo=True)
    studio_hk.select_entity(eh)
    _ = studio_hk.set_entity_transform(eh, Vec3(22.0, 5.0, 0.0), Quat(0.0, 0.0, 0.0, 1.0), Vec3(1.0, 1.0, 1.0))
    _ = studio_hk.undo()
    var h_inp = InputState()
    h_inp.begin_frame()
    h_inp.set_logical_key(341, True)  # Left Ctrl
    h_inp.set_logical_key(340, True)  # Left Shift
    h_inp.set_logical_key(90, True)   # Z
    _ = studio_hk.process_editor_input(h_inp, Float32(1.0 / 60.0))
    var redo_pos_ok = studio_hk.is_alive_id(eh) and eh < len(studio_hk.entities) and abs(studio_hk.entities[eh].translation.x - Float32(22.0)) < Float32(1.0e-3)

    h_inp.begin_frame()
    h_inp.set_logical_key(341, False)
    h_inp.set_logical_key(340, False)
    h_inp.set_logical_key(90, False)
    h_inp.set_logical_key(70, True)   # F -> frame selection
    _ = studio_hk.process_editor_input(h_inp, Float32(1.0 / 60.0))
    var framed_ok = abs(studio_hk.cam_target.x - Float32(22.0)) < Float32(1.0)
    studio_hk.shutdown()
    _ = runner.check(
        "t1_m0_f11_pending_hotkeys_f_frame_and_ctrl_shift_z_redo",
        redo_pos_ok and framed_ok,
        "Ctrl+Shift+Z redo or F frame hotkey not yet wired",
    )

    # 2b. F11: Ctrl+C + Ctrl+V duplicate via InputState + process_editor_input (undoable)
    var studio_cv = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var e_cv = studio_cv.spawn_entity("CopyTarget", position=Vec3(1.0, 1.0, 0.0), record_undo=True)
    studio_cv.select_entity(e_cv)
    var cv_inp = InputState()
    cv_inp.begin_frame()
    cv_inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
    cv_inp.set_logical_key(UI_KEY_C, True)
    _ = studio_cv.process_editor_input(cv_inp, Float32(1.0 / 60.0))
    cv_inp.begin_frame()
    cv_inp.set_logical_key(UI_KEY_C, False)
    _ = studio_cv.process_editor_input(cv_inp, Float32(1.0 / 60.0))
    cv_inp.begin_frame()
    cv_inp.set_logical_key(UI_KEY_V, True)
    _ = studio_cv.process_editor_input(cv_inp, Float32(1.0 / 60.0))
    cv_inp.begin_frame()
    cv_inp.set_logical_key(UI_KEY_LEFT_CTRL, False)
    cv_inp.set_logical_key(UI_KEY_V, False)
    _ = studio_cv.process_editor_input(cv_inp, Float32(1.0 / 60.0))
    var cv_dup_ok = studio_cv.live_entity_count() == 2 and studio_cv.undo() and studio_cv.live_entity_count() == 1 and studio_cv.redo() and studio_cv.live_entity_count() == 2
    studio_cv.shutdown()
    _ = runner.check(
        "t1_m0_f11_pending_ctrl_c_v_duplicate",
        cv_dup_ok,
        "Ctrl+C / Ctrl+V duplicate hotkey not yet wired",
    )

    # 2c. F11: Local/world gizmo toggle via InputMap action or CommandMenu item
    var studio_gw = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var has_gizmo_space_action = False
    for ai in range(len(studio_gw.input_map.actions)):
        var an = studio_gw.input_map.actions[ai].name
        if an.find("gizmo") >= 0 and (an.find("space") >= 0 or an.find("local") >= 0 or an.find("world") >= 0):
            has_gizmo_space_action = True
    var has_gizmo_space_menu = False
    for gi in range(len(studio_gw.menu.bar.groups)):
        for ii in range(len(studio_gw.menu.bar.groups[gi].items)):
            var mt = studio_gw.menu.bar.groups[gi].items[ii].title
            if mt.find("Local") >= 0 or mt.find("World") >= 0 or mt.find("Space") >= 0:
                has_gizmo_space_menu = True
    studio_gw.shutdown()
    _ = runner.check(
        "t1_m0_f11_pending_local_world_gizmo_toggle",
        has_gizmo_space_action or has_gizmo_space_menu,
        "Local/world gizmo toggle action/menu not yet exposed",
    )

    # 2d. F11: Dirty * marker & Ctrl+Q unsaved-changes guard
    var studio_dq = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=1280, height=720)
    _ = studio_dq.spawn_entity("DirtyBox", position=Vec3(1.0, 2.0, 3.0), record_undo=True)
    studio_dq.render_editor_frame(1280, 720)
    var dirty_prims = studio_dq.ui_ctx.draw_list.count()
    var win_ptr = studio_dq.app.resources.get_ptr[WindowStateResource]()
    var has_star = (
        studio_dq.menu.message.find("*") >= 0
        or studio_dq.app.config.title.find("*") >= 0
        or (Bool(win_ptr) and win_ptr.value()[unsafe_offset=0].title.find("*") >= 0)
    )
    var dq_inp = InputState()
    dq_inp.begin_frame()
    dq_inp.set_logical_key(UI_KEY_LEFT_CTRL, True)
    dq_inp.set_logical_key(81, True)  # Ctrl+Q
    var stay_open_dirty = studio_dq.process_editor_input(dq_inp, Float32(1.0 / 60.0))
    dq_inp.begin_frame()
    dq_inp.set_logical_key(UI_KEY_LEFT_CTRL, False)
    dq_inp.set_logical_key(81, False)
    studio_dq.render_editor_frame(1280, 720, input=dq_inp)
    var modal_opened = studio_dq.ui_ctx.modal_depth > 0 or studio_dq.ui_ctx.had_modal_prev_frame
    var saved_clean = studio_dq.save_scene_file(tmp + "/dirty_guard_saved.flscn")
    studio_dq.render_editor_frame(1280, 720)
    var clean_prims = studio_dq.ui_ctx.draw_list.count()
    var dirty_ui_diff = dirty_prims != clean_prims
    studio_dq.shutdown()
    _ = runner.check(
        "t1_m0_f11_pending_dirty_marker_and_ctrl_q_guard",
        stay_open_dirty and modal_opened and saved_clean and (has_star or dirty_ui_diff),
        "Dirty * marker or Ctrl+Q unsaved-changes guard not yet implemented",
    )

    # 3. F12: Inspector UI & StudioEntityView.inspect_fields() expose M0 rename/collider/PBR fields
    var studio_f12 = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=1280, height=1200)
    var ef12 = studio_f12.spawn_entity("InspUiTarget", position=Vec3(0.0, 1.0, 0.0), record_undo=False)
    studio_f12.select_entity(ef12)
    var f12_vis = FieldVisitor()
    studio_f12.entities[ef12].inspect_fields(f12_vis)
    var has_fric = False
    var has_rest = False
    var has_opac = False
    var has_cc = False
    var has_sh = False
    var has_sss = False
    var has_trig = False
    for fi in range(len(f12_vis.fields)):
        var fname = f12_vis.fields[fi].name
        if fname == "friction":
            has_fric = True
        elif fname == "restitution":
            has_rest = True
        elif fname == "opacity":
            has_opac = True
        elif fname == "clearcoat":
            has_cc = True
        elif fname == "sheen":
            has_sh = True
        elif fname == "subsurface":
            has_sss = True
        elif fname == "is_trigger":
            has_trig = True
    var refl_ok = has_fric and has_rest and has_opac and has_cc and has_sh and has_sss and has_trig

    # Check Inspector exposes interactive rename text_input via Tab cycling (wants_keyboard becomes True)
    studio_f12.render_editor_frame(1280, 1200)
    var n_focus = len(studio_f12.ui_ctx.bg_focusables)
    var found_text_input = False
    var tab_inp = InputState()
    for _ in range(n_focus + 2):
        tab_inp.begin_frame()
        tab_inp.set_logical_key(UI_KEY_TAB, True)
        studio_f12.render_editor_frame(1280, 1200, input=tab_inp)
        tab_inp.set_logical_key(UI_KEY_TAB, False)
        if studio_f12.ui_ctx.wants_keyboard():
            found_text_input = True
            break
    studio_f12.shutdown()
    _ = runner.check(
        "t1_m0_f12_pending_inspector_ui_and_reflection_fields",
        refl_ok and found_text_input,
        "refl_ok=" + String(refl_ok) + " rename_text_input=" + String(found_text_input),
    )

    # 4. F13: Recursive Asset Browser across nested subdirs (>10 items, no min(10, ...) UI cap)
    var ab10_dir = tmp + "/assets_10"
    for i in range(10):
        _ = write_minimal_glb(ab10_dir + "/m10_" + String(i) + ".glb")
    var ab_dir = tmp + "/nested_assets"
    for i in range(6):
        _ = write_minimal_glb(ab_dir + "/models/sub/m_" + String(i) + ".glb")
    for i in range(5):
        _ = write_minimal_png(ab_dir + "/textures/t_" + String(i) + ".png", 8, 8, 0xFF3399FF)
    for i in range(4):
        _ = write_minimal_wav(ab_dir + "/audio/a_" + String(i) + ".wav", 0.02, 48000, 1)
    _ = write_text(ab_dir + "/scenes/main.flscn", '{"schema":2,"entities":[]}\n')

    var studio_ab = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=1600, height=2400)
    var found_10 = studio_ab.scan_asset_browser(ab10_dir)
    studio_ab.render_editor_frame(1600, 2400)
    var focus_10 = len(studio_ab.ui_ctx.bg_focusables)
    var prims_10 = studio_ab.ui_ctx.draw_list.count()

    var found_n = studio_ab.scan_asset_browser(ab_dir)
    studio_ab.render_editor_frame(1600, 2400)
    var focus_16 = len(studio_ab.ui_ctx.bg_focusables)
    var prims_16 = studio_ab.ui_ctx.draw_list.count()
    studio_ab.shutdown()
    _ = runner.check(
        "t1_m0_f13_pending_recursive_asset_browser_gt_10",
        found_10 == 10 and found_n == 16 and len(studio_ab.asset_browser_entries) == 16 and focus_16 >= focus_10 + 6 and prims_16 > prims_10,
        "found_n=" + String(found_n) + " focus_10=" + String(focus_10) + " focus_16=" + String(focus_16),
    )

    # 5. F13: M0 Spawn Cards (Empty, Cube, Point Light, Camera, etc.) in Asset Browser / Studio UI
    var studio_sc = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=1600, height=900)
    studio_sc.render_editor_frame(1600, 900)
    var ab_info = studio_sc.dock.find_panel("AssetBrowser")
    var ab_rect = ab_info[3]
    var card_centers_x = List[Float32]()
    var card_centers_y = List[Float32]()
    for fi in range(len(studio_sc.ui_ctx.bg_focusables)):
        var wid = studio_sc.ui_ctx.bg_focusables[fi]
        var r_opt = studio_sc.ui_ctx.cached_rects.get(wid)
        if r_opt:
            var cr = r_opt.value()
            var cx = cr.center_x()
            var cy = cr.center_y()
            if ab_rect.contains(cx, cy):
                card_centers_x.append(cx)
                card_centers_y.append(cy)
    var sc_inp = InputState()
    for ci in range(len(card_centers_x)):
        sc_inp.begin_frame()
        sc_inp.move_mouse(card_centers_x[ci], card_centers_y[ci])
        sc_inp.set_mouse(0, True)
        studio_sc.render_editor_frame(1600, 900, input=sc_inp)
        sc_inp.begin_frame()
        sc_inp.set_mouse(0, False)
        studio_sc.render_editor_frame(1600, 900, input=sc_inp)
        # The toolbar's Import / Create buttons open a modal that would swallow the later card clicks.
        studio_sc.asset_browser_state.dialog = 0
        sc_inp.begin_frame()
        studio_sc.render_editor_frame(1600, 900, input=sc_inp)
    var spawned_empty = False
    var spawned_cube = False
    var spawned_point_light = False
    var spawned_camera = False
    for i in range(len(studio_sc.app.world.entities)):
        if not studio_sc.app.world.entities[i].alive:
            continue
        var ent = Entity(UInt32(i), studio_sc.app.world.entities[i].generation)
        var has_mesh = studio_sc.app.world.has[MeshRendererComponent](ent)
        var has_pl = studio_sc.app.world.has[PointLightComponent](ent)
        var has_cam = studio_sc.app.world.has[CameraComponent](ent)
        if has_pl:
            spawned_point_light = True
        if has_cam:
            spawned_camera = True
        if has_mesh and (not has_pl) and (not has_cam):
            spawned_cube = True
        if (not has_mesh) and (not has_pl) and (not has_cam):
            spawned_empty = True
    studio_sc.shutdown()
    _ = runner.check(
        "t1_m0_f13_pending_spawn_cards",
        spawned_empty and spawned_cube and spawned_point_light and spawned_camera,
        "empty=" + String(spawned_empty)
        + " cube=" + String(spawned_cube)
        + " point_light=" + String(spawned_point_light)
        + " camera=" + String(spawned_camera),
    )

    # 6. F14: FlintStudioState installs AnimPlugin & AudioServer
    var studio_plg = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var has_plg = (
        studio_plg.registry.find_by_name("AnimatorComponent") >= 0
        and studio_plg.app.resources.has[AnimAssetCache]()
        and studio_plg.app.resources.has[AudioServer]()
    )
    studio_plg.shutdown()
    _ = runner.check(
        "t1_m0_f14_pending_studio_installs_anim_and_audio_plugins",
        has_plg,
        "FlintStudioState does not yet install AnimPlugin/AudioServer",
    )


def main() raises:
    var tmp = make_tmp_dir("t1_m0_core")
    var runner = StudioSuiteRunner("t1_m0_studio_core")
    try:
        run_f06_edit_vs_pie_scheduling(runner)
        run_f07_overlay_lines_and_lights(runner)
        run_f08_material_override_isolation(runner)
        run_f09_ui_capture_widgets_and_esc_guard(runner)
        run_f10_kcc_triggers_and_edit_camera(runner)
        run_f11_dockspace_and_undo(runner)
        run_f12_inspector_fields_and_roundtrip(runner, tmp)
        run_f14_anim_audio_plugins_and_codecs(runner, tmp)
        run_pending_m3_m0_runtime_checks(runner, tmp)
    finally:
        rm_rf_tmp(tmp)
    runner.finish()
