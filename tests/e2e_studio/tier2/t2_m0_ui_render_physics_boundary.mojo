# AREA: m0
"""Tier 2 Boundary & Negative Coverage: Area B — Bugs #1-#5 & M0 UI/Render/Physics/Dock/Audio/Nav (F06-F14)."""

from std.collections import List
from std.math import abs, isfinite, isnan
from flint.app.engine_app import AppConfig, EngineApp, FixedTime, FrameTime
from flint.assets.mesh_builder import MeshBuilder
from flint.audio.server import AudioServer, SoundHandle
from flint.camera.rig import CAM_MODE_FREE_FLY
from flint.core.math import Vec3
from flint.core.serial import BlobReader, BlobWriter
from flint.ecs.world import NULL_ENTITY
from flint.nav.navmesh import NavMesh
from flint.physics.character import CharacterController, CharacterControllerConfig
from flint.physics.filter import CAT_PROP_PIECE, CAT_SENSOR, CollisionFilter
from flint.physics.plugin import (
    CharacterControllerComponent,
    ColliderComponent,
    PhysicsPlugin,
    PhysicsSyncStateResource,
    TriggerEventsResource,
)
from flint.physics.world import PhysicsWorld, TRIGGER_EXIT, TriggerVolume
from flint.platform.audio import decode_wav
from flint.platform.input import InputState
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import (
    CameraComponent,
    MaterialOverrideComponent,
    MeshRendererComponent,
    PointLightComponent,
    RenderPlugin,
)
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, RenderWorld, ScalableRenderer
from flint.scene.transform_hierarchy import GlobalTransform, LocalTransform, PrevGlobalTransform, TransformPlugin
from flint.studio.studio_app import FlintStudioState, PIE_PAUSED, PIE_PLAYING, PIE_STOPPED, StudioEntityView
from flint.ui.context import UI_KEY_ESCAPE, UiContext, pack_rgba8
from flint.ui.flexbox_dock import DOCK_SPLIT_H, DockSpace, RectF
from e2e_studio.common.studio_test_utils import (
    StudioSuiteRunner,
    make_tmp_dir,
    rm_rf_tmp,
)


def test_bug1_scheduling_dt_boundaries(mut runner: StudioSuiteRunner):
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
    )
    _ = studio.spawn_entity("DtProbe", position=Vec3(0.0, 2.0, 0.0), mass=1.0, record_undo=False)
    var h0 = studio.state_hash()

    # Invalid transitions when PIE_STOPPED
    var invalid_trans_ok = (not studio.pie_pause())
    var stop_when_stopped: Bool
    try:
        stop_when_stopped = studio.pie_stop()
    except:
        stop_when_stopped = False
    invalid_trans_ok = invalid_trans_ok and (not stop_when_stopped)
    invalid_trans_ok = invalid_trans_ok and studio.pie_step(0) == 0 and studio.pie_step(-3) == 0
    invalid_trans_ok = invalid_trans_ok and studio.step_pie_frame(Float32(1.0 / 60.0)) == 0

    # Bad dt values in Edit mode
    var nan_dt = Float32(0.0) / Float32(0.0)
    var inf_dt = Float32(1.0) / Float32(0.0)
    studio.render_editor_frame(160, 120, dt=Float32(0.0))
    studio.render_editor_frame(160, 120, dt=Float32(-1.0))
    studio.render_editor_frame(160, 120, dt=nan_dt)
    studio.render_editor_frame(160, 120, dt=inf_dt)

    var ft = studio.app.resources.get_ptr[FrameTime]()
    var edit_finite_ok = (
        Bool(ft)
        and isfinite(ft.value()[unsafe_offset=0].delta_seconds)
        and isfinite(Float32(ft.value()[unsafe_offset=0].elapsed_seconds))
        and studio.state_hash() == h0
    )

    # PIE spiral-of-death & non-finite dt boundaries
    var entered = studio.pie_play()
    var s_zero = studio.step_pie_frame(Float32(0.0))
    var s_neg = studio.step_pie_frame(Float32(-5.0))
    var s_nan = studio.step_pie_frame(nan_dt)
    var s_inf = studio.step_pie_frame(inf_dt)
    var s_huge = studio.step_pie_frame(Float32(10.0))
    var fix_ptr = studio.app.resources.get_ptr[FixedTime]()
    var max_steps = fix_ptr.value()[unsafe_offset=0].max_steps_per_frame if Bool(fix_ptr) else 8

    var paused = studio.pie_pause()
    var s_while_paused = studio.step_pie_frame(Float32(1.0 / 60.0))

    var stopped: Bool
    try:
        stopped = studio.pie_stop()
    except:
        stopped = False

    studio.shutdown()
    _ = runner.check(
        "t2_m0_f06_scheduling_dt_and_transition_boundaries",
        invalid_trans_ok
        and edit_finite_ok
        and entered
        and s_zero == 0
        and s_neg == 0
        and s_nan == 0
        and s_inf >= 0
        and s_huge <= max_steps
        and s_huge >= 1
        and paused
        and s_while_paused == 0
        and stopped,
        "PIE scheduling boundary or spiral clamp failed",
    )


def test_bug2_overlay_lines_and_disabled_light_boundaries(mut runner: StudioSuiteRunner):
    var app = EngineApp(AppConfig(title="t2_overlay", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    app.startup()

    var cam_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(0.0, 0.5, 3.0)))
    _ = app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
    _ = app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=55.0, near_z=0.1, active=True))

    app.step_one_frame(Float32(1.0 / 60.0))
    var px_base = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()

    var nan_v = Float32(0.0) / Float32(0.0)
    var inf_v = Float32(1.0) / Float32(0.0)
    var rw_ptr = app.resources.get_ptr_mut[RenderWorld]()
    rw_ptr.value()[unsafe_offset=0].overlay_lines.add_line(Vec3(nan_v, 0.0, 0.0), Vec3(1.0, 1.0, 0.0), pack_rgba8(255, 0, 0, 255))
    rw_ptr.value()[unsafe_offset=0].overlay_lines.add_line(Vec3(0.0, inf_v, 0.0), Vec3(0.0, 1.0, 0.0), pack_rgba8(0, 255, 0, 255))
    rw_ptr.value()[unsafe_offset=0].overlay_lines.add_line(Vec3(1.0, 1.0, 1.0), Vec3(1.0, 1.0, 1.0), pack_rgba8(0, 0, 255, 255), thickness_px=-5.0)
    # Segment straddling behind near plane (camera at z=3 looking -Z)
    rw_ptr.value()[unsafe_offset=0].overlay_lines.add_line(Vec3(-0.2, 0.5, 6.0), Vec3(0.2, 0.5, 1.0), pack_rgba8(255, 255, 0, 255), thickness_px=2.0)

    app.step_one_frame(Float32(1.0 / 60.0))
    var cleared_post = rw_ptr.value()[unsafe_offset=0].overlay_lines.segment_count() == 0
    var px_after = app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
    var render_ok = len(px_after) == len(px_base) and cleared_post
    app.shutdown()

    # Disabled PointLightComponent must NOT suppress Studio's 2 fallback lights
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0)
    var lgt_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](lgt_e, LocalTransform(Vec3(0.0, 3.0, 0.0)))
    _ = studio.app.world.insert[GlobalTransform](lgt_e, GlobalTransform())
    _ = studio.app.world.insert[PointLightComponent](lgt_e, PointLightComponent(Vec3(6.0, 5.0, 4.0), 12.0, enabled=False))
    studio.render_editor_frame(160, 120)
    var s_rw = studio.app.resources.get_ptr[RenderWorld]()
    var fallback_preserved = Bool(s_rw) and len(s_rw.value()[unsafe_offset=0].cluster_lights) == 2
    studio.shutdown()

    _ = runner.check(
        "t2_m0_f07_overlay_nan_behind_near_and_disabled_light",
        render_ok and fallback_preserved,
        "overlay boundary or disabled light handling failed",
    )


def test_bug3_material_override_invalid_id_nan_trunc(mut runner: StudioSuiteRunner):
    var app = EngineApp(AppConfig(title="t2_mat_bounds", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    app.startup()

    var rend = app.resources.get_ptr_mut[ScalableRenderer]()
    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(), Vec3(0.5, 0.5, 0.5))
    var mesh_id = rend.value()[unsafe_offset=0].register_mesh(mb^.finish(), 0)

    var cam_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(0.0, 0.0, 4.0)))
    _ = app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = app.world.insert[CameraComponent](cam_e, CameraComponent())

    var nan_f = Float32(0.0) / Float32(0.0)
    var inf_f = Float32(1.0) / Float32(0.0)
    var ent_bad = app.world.spawn()
    _ = app.world.insert[LocalTransform](ent_bad, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](ent_bad, GlobalTransform())
    _ = app.world.insert[MeshRendererComponent](ent_bad, MeshRendererComponent(mesh_id, 999999))
    _ = app.world.insert[MaterialOverrideComponent](
        ent_bad,
        MaterialOverrideComponent(
            -999,
            Vec3(nan_f, -2.0, 5.0),
            -2.0,
            5.0,
            Vec3(-1.0, nan_f, inf_f),
            True,
            opacity=2.5,
            alpha_cutoff=-1.0,
            clearcoat=nan_f,
            clearcoat_roughness=-0.5,
            sheen=inf_f,
            subsurface=-3.0,
        ),
    )
    app.step_one_frame(Float32(1.0 / 60.0))

    var rw = app.resources.get_ptr[RenderWorld]()
    var finite_mat_ok = False
    if Bool(rw) and len(rw.value()[unsafe_offset=0].instances) == 1:
        var slot = rw.value()[unsafe_offset=0].instances[0].material_id
        if slot >= 0 and slot < len(rend.value()[unsafe_offset=0].materials):
            ref m = rend.value()[unsafe_offset=0].materials[slot]
            finite_mat_ok = (
                isfinite(m.base_color.x)
                and isfinite(m.roughness)
                and isfinite(m.metallic)
                and isfinite(m.opacity)
                and isfinite(m.alpha_cutoff)
                and isfinite(m.clearcoat)
                and isfinite(m.clearcoat_roughness)
                and isfinite(m.emissive.y)
                and isfinite(m.emissive.z)
                and m.roughness >= Float32(0.0)
                and m.roughness <= Float32(1.0)
                and m.opacity >= Float32(0.0)
                and m.opacity <= Float32(1.0)
            )
    app.shutdown()

    # Truncated binary blob MUST raise Error cleanly
    var bw = BlobWriter()
    bw.integer(4)
    bw.f32(0.5)  # truncated far short of full MaterialOverrideComponent
    var trunc_raised = False
    try:
        var br = BlobReader(bw.data.copy())
        _ = MaterialOverrideComponent.read_bin(br, MaterialOverrideComponent.SCHEMA_VERSION)
    except:
        trunc_raised = True

    _ = runner.check(
        "t2_m0_f08_material_override_invalid_id_nan_and_truncated_bin",
        finite_mat_ok and trunc_raised,
        "OOB material_id, NaN sanitization, or truncated read_bin failed",
    )


def test_bug4_ui_widgets_nan_inverted_bounds_and_modal(mut runner: StudioSuiteRunner):
    var ui = UiContext()
    var inp = InputState()
    var nan_v = Float32(0.0) / Float32(0.0)
    var df_val = nan_v
    var col_val = Vec3(nan_v, -5.0, 99.0)
    var empty_items = List[String]()
    var bad_sel = 999

    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.drag_float("df_inv", "Inv", df_val, speed=-0.5, min_val=5.0, max_val=1.0)
    _ = ui.color_edit3("col_inv", "ColInv", col_val, min_val=0.8, max_val=0.2)
    _ = ui.combo_box("cmb_empty", "Empty", bad_sel, empty_items)
    var cmb_r = ui.last_item_rect
    ui.begin_scroll_view("sc_empty", 200.0, 80.0)
    var vis_zero = ui.scroll_visible_range(0, Float32(0.0))
    var vis_nan = ui.scroll_visible_range(10, nan_v)
    ui.end_scroll_view()
    ui.end_frame()

    # Click empty combo_box: must NOT open popup or panic
    inp.begin_frame()
    inp.move_mouse(cmb_r.center_x(), cmb_r.center_y())
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.combo_box("cmb_empty", "Empty", bad_sel, empty_items)
    ui.end_frame()
    inp.set_mouse(0, False)

    var bounds_ok = (
        isfinite(df_val)
        and df_val >= Float32(1.0)
        and df_val <= Float32(5.0)
        and isfinite(col_val.x)
        and isfinite(col_val.y)
        and isfinite(col_val.z)
        and col_val.x >= Float32(0.2)
        and col_val.z <= Float32(0.8)
        and (not ui.wants_keyboard())
        and vis_zero[0] == 0
        and vis_zero[1] == 0
        and vis_nan[0] == 0
        and vis_nan[1] == 10
    )

    # Modal blocks background button & traps keyboard; Esc closes modal
    var ui_m = UiContext()
    var modal_open = True
    var bg_clicked = False
    var bg_r: RectF
    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui_m.begin_frame(inp, 1000.0, 700.0)
    if ui_m.button("bg_btn", "BG", 120.0, 28.0):
        bg_clicked = True
    bg_r = ui_m.last_item_rect
    if ui_m.begin_modal("m_guard", "Confirm", 300.0, 160.0, modal_open):
        _ = ui_m.button("m_ok", "OK", 80.0, 26.0)
        ui_m.end_modal()
    ui_m.end_frame()
    var modal_active_cap = ui_m.wants_keyboard() and ui_m.wants_mouse()

    # Try clicking background button while modal is open
    inp.begin_frame()
    inp.move_mouse(bg_r.center_x(), bg_r.center_y())
    inp.set_mouse(0, True)
    ui_m.begin_frame(inp, 1000.0, 700.0)
    if ui_m.button("bg_btn", "BG", 120.0, 28.0):
        bg_clicked = True
    if ui_m.begin_modal("m_guard", "Confirm", 300.0, 160.0, modal_open):
        _ = ui_m.button("m_ok", "OK", 80.0, 26.0)
        ui_m.end_modal()
    ui_m.end_frame()

    inp.begin_frame()
    inp.set_mouse(0, False)
    inp.set_logical_key(UI_KEY_ESCAPE, True)
    ui_m.begin_frame(inp, 1000.0, 700.0)
    if ui_m.button("bg_btn", "BG", 120.0, 28.0):
        bg_clicked = True
    if ui_m.begin_modal("m_guard", "Confirm", 300.0, 160.0, modal_open):
        _ = ui_m.button("m_ok", "OK", 80.0, 26.0)
        ui_m.end_modal()
    ui_m.end_frame()
    inp.set_logical_key(UI_KEY_ESCAPE, False)

    _ = runner.check(
        "t2_m0_f09_ui_empty_nan_inverted_bounds_and_modal_trap",
        bounds_ok and modal_active_cap and (not bg_clicked) and (not modal_open),
        "UI boundary or modal trap failed",
    )


def test_bug4_rmb_cross_boundary_drag_and_outside_blur(mut runner: StudioSuiteRunner):
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=1000, height=700, worker_threads=0)
    var s_inp = InputState()
    studio.render_editor_frame(1000, 700, input=s_inp)

    var hier = studio.dock.find_panel("Hierarchy")[3]
    var vp = studio.dock.find_panel("SceneViewport")[3]
    var fwd_init = studio.cam_fwd

    # Press RMB over Hierarchy (DOCK_TABS -> wants_mouse() == True)
    s_inp.begin_frame()
    s_inp.move_mouse(hier.center_x(), hier.center_y())
    s_inp.set_mouse(1, True)
    studio.render_editor_frame(1000, 700, input=s_inp)
    var stay1 = studio.process_editor_input(s_inp, Float32(1.0 / 60.0))
    var no_fly_in_ui = studio.camera_rig.mode != CAM_MODE_FREE_FLY

    # Drag into SceneViewport while STILL holding RMB (mouse_pressed(1) == False)
    s_inp.begin_frame()
    s_inp.move_mouse(vp.center_x(), vp.center_y())
    studio.render_editor_frame(1000, 700, input=s_inp)
    var stay2 = studio.process_editor_input(s_inp, Float32(1.0 / 60.0))
    var no_fly_cross_drag = (
        studio.camera_rig.mode != CAM_MODE_FREE_FLY
        and abs(studio.cam_fwd.x - fwd_init.x) < Float32(1.0e-4)
        and abs(studio.cam_fwd.y - fwd_init.y) < Float32(1.0e-4)
    )

    # Release RMB, then press RMB freshly inside SceneViewport -> engages CAM_MODE_FREE_FLY
    s_inp.begin_frame()
    s_inp.set_mouse(1, False)
    studio.render_editor_frame(1000, 700, input=s_inp)
    _ = studio.process_editor_input(s_inp, Float32(1.0 / 60.0))

    s_inp.begin_frame()
    s_inp.move_mouse(vp.center_x(), vp.center_y())
    s_inp.set_mouse(1, True)
    studio.render_editor_frame(1000, 700, input=s_inp)
    var stay3 = studio.process_editor_input(s_inp, Float32(1.0 / 60.0))
    var fly_engaged = studio.camera_rig.mode == CAM_MODE_FREE_FLY
    studio.shutdown()

    # text_input focus persists across neutral frames & blurs on outside LMB click
    var ui = UiContext()
    var inp = InputState()
    var txt = String("BlurMe")
    inp.begin_frame()
    inp.move_mouse(800.0, 500.0)
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.text_input("ti_blur", txt, 180.0)
    var ti_r = ui.last_item_rect
    ui.end_frame()

    inp.begin_frame()
    inp.move_mouse(ti_r.center_x(), ti_r.center_y())
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.text_input("ti_blur", txt, 180.0)
    ui.end_frame()
    var focused_on_click = ui.wants_keyboard()

    inp.begin_frame()
    inp.set_mouse(0, False)
    inp.move_mouse(800.0, 500.0)
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.text_input("ti_blur", txt, 180.0)
    ui.end_frame()
    var persists_neutral = ui.wants_keyboard()

    inp.begin_frame()
    inp.set_mouse(0, True)
    ui.begin_frame(inp, 1000.0, 700.0)
    _ = ui.text_input("ti_blur", txt, 180.0)
    ui.end_frame()
    var blurs_outside = not ui.wants_keyboard()

    _ = runner.check(
        "t2_m0_f09_rmb_cross_boundary_drag_and_text_blur",
        stay1 and stay2 and stay3 and no_fly_in_ui and no_fly_cross_drag and fly_engaged and focused_on_click and persists_neutral and blurs_outside,
        "cross-boundary RMB fly or outside click blur failed",
    )


def test_bug5_kcc_and_camera_boundaries(mut runner: StudioSuiteRunner):
    var pw = PhysicsWorld(gravity=0.0)
    var bad_update = pw.update_kcc_proxy(99999, Vec3(1.0, 2.0, 3.0))
    pw.unregister_kcc_proxy(-1)

    # Filter mismatch (mask=0) suppresses KCC trigger events
    _ = pw.add_trigger(
        TriggerVolume.obb(20, Vec3(0.0, 1.0, 0.0), Vec3(2.0, 2.0, 2.0), filter=CollisionFilter(CAT_SENSOR, 0, 0))
    )
    var cfg = CharacterControllerConfig()
    cfg.gravity = 0.0
    var kcc = CharacterController(Vec3(0.0, 1.0, 0.0), cfg)
    var kcc_id = pw.register_kcc_proxy(kcc.pos, kcc.cfg.radius, kcc.cfg.half_height, kcc.cfg.up, kcc.cfg.filter)
    pw.step(Float32(1.0 / 60.0))
    var zero_mismatch_evs = len(pw.trigger_events) == 0

    # Despawning TriggerVolume entity mid-overlap while KCC is inside leaves zero dangling state
    var app = EngineApp(AppConfig(title="t2_kcc_trig_despawn", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.startup()
    var bad_lookup = app.resources.get_ptr[PhysicsSyncStateResource]().value()[unsafe_offset=0].lookup_entity(-1) == NULL_ENTITY

    var trig_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](trig_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](trig_e, GlobalTransform())
    _ = app.world.insert[ColliderComponent](trig_e, ColliderComponent.box(Vec3(1.5, 1.5, 1.5), is_trigger=True))

    var kcc_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](kcc_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](kcc_e, GlobalTransform())
    _ = app.world.insert[CharacterControllerComponent](kcc_e, CharacterControllerComponent(Vec3(0.0, 1.0, 0.0), cfg, Vec3()))

    var tev_cursor = 0
    app.step_one_frame(Float32(1.0 / 60.0))
    _ = app.resources.get_ptr[TriggerEventsResource]().value()[unsafe_offset=0].events.read_since(tev_cursor)
    var kcc_body_id = -1
    ref pw_mid = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    if len(pw_mid.kcc_proxies) > 0:
        kcc_body_id = pw_mid.kcc_proxies[0].id

    var despawn_trig = app.world.despawn(trig_e)
    app.step_one_frame(Float32(1.0 / 60.0))  # despawn frame emits TRIGGER_EXIT and deactivates trigger
    var despawn_evs = app.resources.get_ptr[TriggerEventsResource]().value()[unsafe_offset=0].events.read_since(tev_cursor)
    var saw_trig_despawn_exit = False
    for ei in range(len(despawn_evs)):
        ref ev = despawn_evs[ei]
        if ev.kind == TRIGGER_EXIT and ev.body_id == kcc_body_id:
            saw_trig_despawn_exit = True

    app.step_one_frame(Float32(1.0 / 60.0))  # subsequent frame must emit 0 events
    var next_evs = app.resources.get_ptr[TriggerEventsResource]().value()[unsafe_offset=0].events.read_since(tev_cursor)
    ref pw_after = app.resources.get_ptr[PhysicsWorld]().value()[unsafe_offset=0]
    var all_trigs_inactive = True
    for ti in range(len(pw_after.triggers)):
        if pw_after.triggers[ti].active:
            all_trigs_inactive = False
    var post_evs = pw_after.trigger_events.copy()
    var clean_after_trig_despawn = (
        kcc_body_id >= 0
        and saw_trig_despawn_exit
        and len(next_evs) == 0
        and all_trigs_inactive
        and len(post_evs) == 0
    )
    app.shutdown()

    # Inactive CameraComponent does not hijack PIE; zero look_target_offset remains finite
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0)
    studio.look_at(Vec3(0.0, 2.0, -5.0), Vec3(0.0, 1.0, 0.0))
    var c_inact = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](c_inact, LocalTransform(Vec3(40.0, 10.0, 40.0)))
    _ = studio.app.world.insert[GlobalTransform](c_inact, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](c_inact, PrevGlobalTransform())
    _ = studio.app.world.insert[CameraComponent](c_inact, CameraComponent(active=False))
    studio.render_editor_frame(160, 120)
    _ = studio.pie_play()
    _ = studio.step_pie_frame(Float32(1.0 / 60.0))
    studio.render_editor_frame(160, 120)
    var rw = studio.app.resources.get_ptr[RenderWorld]()
    var inactive_ignored = abs(rw.value()[unsafe_offset=0].camera.eye.x - Float32(0.0)) < Float32(1.0e-3)
    var stop_ok: Bool
    try:
        stop_ok = studio.pie_stop()
    except:
        stop_ok = False

    var c_zero = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](c_zero, LocalTransform(Vec3(1.0, 2.0, 3.0)))
    _ = studio.app.world.insert[GlobalTransform](c_zero, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](c_zero, PrevGlobalTransform())
    _ = studio.app.world.insert[CameraComponent](c_zero, CameraComponent(active=True, look_target_offset=Vec3(0.0, 0.0, 0.0)))
    _ = studio.pie_play()
    _ = studio.step_pie_frame(Float32(1.0 / 60.0))
    studio.render_editor_frame(160, 120)
    var zero_target_finite = (
        isfinite(rw.value()[unsafe_offset=0].camera.target.x)
        and isfinite(rw.value()[unsafe_offset=0].camera.target.y)
        and isfinite(rw.value()[unsafe_offset=0].camera.target.z)
    )
    studio.shutdown()

    _ = runner.check(
        "t2_m0_f10_kcc_invalid_ids_filter_despawn_and_cam_boundaries",
        (not bad_update)
        and kcc_id >= 0
        and zero_mismatch_evs
        and bad_lookup
        and despawn_trig
        and clean_after_trig_despawn
        and inactive_ignored
        and stop_ok
        and zero_target_finite,
        "bad_up=" + String(not bad_update)
        + " zmis=" + String(zero_mismatch_evs)
        + " blook=" + String(bad_lookup)
        + " dtrig=" + String(despawn_trig)
        + " clean=" + String(clean_after_trig_despawn)
        + " inact=" + String(inactive_ignored)
        + " stop=" + String(stop_ok)
        + " zfin=" + String(zero_target_finite),
    )


def test_f11_dockspace_malformed_json_reparent_and_empty_undo(mut runner: StudioSuiteRunner):
    var dock = DockSpace.create_studio_5panel()
    var before_json = dock.save_json()

    var bad_jsons = List[String]()
    bad_jsons.append(String("{bad"))
    bad_jsons.append(String('{"version": 1, "root": 0, "nodes": []}'))
    bad_jsons.append(String('{"version": 1, "root": 0, "nodes": [{"id": 0, "kind": 0, "child_a": 99, "child_b": 0}]}'))
    bad_jsons.append(String('{"version": 1, "root": 0, "nodes": [{"id": 0, "kind": 0, "child_a": 0, "child_b": 0}]}'))
    bad_jsons.append(
        String('{"version": 1, "root": 0, "nodes": [{"id": 0, "kind": 0, "child_a": 1, "child_b": 1}, {"id": 1, "kind": 2, "child_a": -1, "child_b": -1}]}')
    )
    bad_jsons.append(
        String('{"version": 1, "root": 0, "nodes": [{"id": 0, "kind": 0, "child_a": 1, "child_b": 2}, {"id": 1, "kind": 0, "child_a": 0, "child_b": 2}, {"id": 2, "kind": 2, "child_a": -1, "child_b": -1}]}')
    )

    var all_rejected = True
    for i in range(len(bad_jsons)):
        if dock.load_json(bad_jsons[i]):
            all_rejected = False
    var layout_intact = dock.save_json() == before_json

    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var empty_undo_ok = (not studio.undo()) and (not studio.redo())
    var parent_id = studio.spawn_entity("Parent", record_undo=False)
    var child_id = studio.spawn_entity("Child", parent_id=parent_id, record_undo=False)
    var self_rej = not studio.reparent_entity(parent_id, parent_id)
    var cycle_rej = not studio.reparent_entity(parent_id, child_id)
    var invalid_rej = not studio.reparent_entity(-1, parent_id)
    var hierarchy_ok = studio.parent_of(child_id) == parent_id and studio.parent_of(parent_id) == -1
    studio.shutdown()

    _ = runner.check(
        "t2_m0_f11_dockspace_malformed_json_reparent_cycle_and_empty_undo",
        all_rejected and layout_intact and empty_undo_ok and self_rej and cycle_rej and invalid_rej and hierarchy_ok,
        "Malformed DockSpace JSON, reparent cycle, or empty undo failed",
    )

    var extra_bad_nodes_rej = not dock.load_json(String('{"version": 1, "root": -1, "nodes": "not_an_array"}'))
    var extra_bad_kind_rej = not dock.load_json(
        String('{"version": 1, "root": 0, "nodes": [{"id": 0, "kind": 999, "child_a": -1, "child_b": -1, "tabs": ["A"]}]}')
    )
    var extra_layout_intact = dock.save_json() == before_json
    _ = runner.check(
        "t2_m0_f11_dockspace_malformed_nodes_type_and_invalid_kind",
        extra_bad_nodes_rej and extra_bad_kind_rej and extra_layout_intact,
        "nodes_non_array_rej=" + String(extra_bad_nodes_rej)
        + " invalid_kind_rej=" + String(extra_bad_kind_rej)
        + " layout_intact=" + String(extra_layout_intact),
    )


def test_f12_f13_f14_browser_inspector_nav_audio_boundaries(mut runner: StudioSuiteRunner, tmp: String):
    var studio = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var empty_dir = tmp + "/empty_assets"
    var n_empty = studio.scan_asset_browser(empty_dir)
    var n_missing = studio.scan_asset_browser(tmp + "/no_such_dir")

    var e0 = studio.spawn_entity("SanitizeTarget", position=Vec3(0.0, 1.0, 0.0), record_undo=False)
    var bad_view = StudioEntityView(e0, "BadBounds")
    var nan_f = Float32(0.0) / Float32(0.0)
    bad_view.collider_half_extents = Vec3(-2.0, nan_f, 0.0)
    bad_view.friction = -1.0
    bad_view.restitution = 5.0
    var edit_ok = studio.commit_node_edit(e0, bad_view^, "Boundary Edit")
    var bad_id_rej = not studio.commit_node_edit(-1, StudioEntityView(-1, "NoEntity"))
    var finite_he = (
        isfinite(studio.entities[e0].collider_half_extents.x)
        and isfinite(studio.entities[e0].collider_half_extents.y)
        and studio.entities[e0].collider_half_extents.x > Float32(0.0)
        and studio.entities[e0].collider_half_extents.y > Float32(0.0)
    )
    studio.shutdown()

    # Empty world NavMesh bake yields 0 polys cleanly
    var empty_pw = PhysicsWorld()
    var nm = NavMesh.build_from_physics(empty_pw)
    var nav_empty_ok = nm.poly_count() == 0

    # AudioServer invalid handle stop_sound & empty/corrupt WAV decode
    var srv = AudioServer(sample_rate=48000, quantum_frames=256, ring_capacity=64)
    var bad_stop = not srv.stop_sound(SoundHandle())
    srv.stop()
    var empty_wav = decode_wav(List[UInt8]())
    var corrupt_bytes = List[UInt8]()
    for b in String("NOT_A_WAV_FILE").as_bytes():
        corrupt_bytes.append(b)
    var corrupt_wav = decode_wav(corrupt_bytes)

    _ = runner.check(
        "t2_m0_f12_f13_f14_browser_inspector_nav_audio_boundaries",
        n_empty == 0
        and n_missing == 0
        and edit_ok
        and bad_id_rej
        and finite_he
        and nav_empty_ok
        and bad_stop
        and (not empty_wav.valid)
        and (not corrupt_wav.valid),
        "AssetBrowser/Inspector/Nav/Audio boundary check failed",
    )


def main() raises:
    var tmp = make_tmp_dir("t2_m0_bounds")
    var runner = StudioSuiteRunner("t2_m0_ui_render_physics_boundary")
    try:
        test_bug1_scheduling_dt_boundaries(runner)
        test_bug2_overlay_lines_and_disabled_light_boundaries(runner)
        test_bug3_material_override_invalid_id_nan_trunc(runner)
        test_bug4_ui_widgets_nan_inverted_bounds_and_modal(runner)
        test_bug4_rmb_cross_boundary_drag_and_outside_blur(runner)
        test_bug5_kcc_and_camera_boundaries(runner)
        test_f11_dockspace_malformed_json_reparent_and_empty_undo(runner)
        test_f12_f13_f14_browser_inspector_nav_audio_boundaries(runner, tmp)
    finally:
        rm_rf_tmp(tmp)
    runner.finish()
