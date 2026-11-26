# AREA: m2
"""Tier 2 Boundary & Negative Opaque API Tests: Area D — Behaviour & Runtime Boundaries (F24, F25)."""

from std.collections import List
from std.math import isfinite, isnan, isinf
from flint.ai import (
    BT_ABORT_NONE,
    BT_CMP_EQ,
    BT_FAILURE,
    BehaviorTree,
    Blackboard,
    BtExecutionState,
)
from flint.app import AppConfig, EngineApp
from flint.audio import AudioServer
from flint.audio.plugin import AudioEmitterComponent, AudioListenerComponent, AudioPlugin
from flint.core.math import Quat, Vec3
from flint.ecs import NULL_ENTITY
from flint.input import ACTION_AXIS2D, ACTION_BOOL, InputContext, InputMap
from flint.nav import NavMesh, NavMeshConfig, NavObstacle
from flint.physics.character import CharacterControllerConfig
from flint.physics.filter import CAT_PROP_PIECE, CAT_SENSOR, CollisionFilter
from flint.physics.plugin import (
    CharacterControllerComponent,
    ColliderComponent,
    PhysicsPlugin,
    PhysicsSyncStateResource,
    TriggerEventsResource,
)
from flint.physics.world import PhysicsWorld, TRIGGER_ENTER, TRIGGER_EXIT, TRIGGER_STAY
from flint.platform.input import InputState
from flint.render.plugin import CameraComponent, RenderPluginStateResource
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, RenderWorld
from flint.scene import (
    GlobalTransform,
    LocalTransform,
    PrevGlobalTransform,
    TransformPlugin,
)
from flint.studio import FlintStudioState
from e2e_studio.common.studio_test_utils import StudioSuiteRunner


def _case_kcc_despawn_inside_trigger_and_stale_lookup() -> Tuple[Bool, String]:
    var app = EngineApp(AppConfig(title="t2_kcc_despawn", headless=True, fixed_hz=60.0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.startup()

    var trig_e = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](trig_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](trig_e, GlobalTransform.from_local(LocalTransform(Vec3(0.0, 1.0, 0.0))))
    _ = app.world.insert[PrevGlobalTransform](trig_e, PrevGlobalTransform())
    _ = app.world.insert[ColliderComponent](trig_e, ColliderComponent.box(Vec3(1.5, 1.5, 1.5), is_trigger=True))

    var kcc_e = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](kcc_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](kcc_e, GlobalTransform.from_local(LocalTransform(Vec3(0.0, 1.0, 0.0))))
    _ = app.world.insert[PrevGlobalTransform](kcc_e, PrevGlobalTransform())
    _ = app.world.insert[CharacterControllerComponent](
        kcc_e,
        CharacterControllerComponent(Vec3(0.0, 1.0, 0.0), CharacterControllerConfig(), Vec3()),
    )

    var cursor = 0
    var saw_enter = False
    var enter_body_id = -1
    var lookup_matched = False

    for _ in range(3):
        app.step_one_frame(Float32(1.0 / 60.0))
        var tev_opt = app.resources.get_ptr[TriggerEventsResource]()
        var sync_opt = app.resources.get_ptr[PhysicsSyncStateResource]()
        if tev_opt and sync_opt:
            var evs = tev_opt.value()[unsafe_offset=0].events.read_since(cursor)
            for i in range(len(evs)):
                if evs[i].kind == TRIGGER_ENTER:
                    saw_enter = True
                    enter_body_id = evs[i].body_id
                    if sync_opt.value()[unsafe_offset=0].lookup_entity(enter_body_id) == kcc_e:
                        lookup_matched = True

    if not saw_enter or not lookup_matched or enter_body_id < 0:
        return (False, "failed initial TRIGGER_ENTER or lookup_entity match before despawn")

    # Despawn KCC while inside trigger
    _ = app.world.despawn(kcc_e)
    var saw_exit_kcc = False
    for _ in range(3):
        app.step_one_frame(Float32(1.0 / 60.0))
        var tev_opt = app.resources.get_ptr[TriggerEventsResource]()
        if tev_opt:
            var evs = tev_opt.value()[unsafe_offset=0].events.read_since(cursor)
            for i in range(len(evs)):
                if evs[i].kind == TRIGGER_EXIT and evs[i].body_id == enter_body_id:
                    saw_exit_kcc = True

    var pw_opt = app.resources.get_ptr[PhysicsWorld]()
    var sync_opt2 = app.resources.get_ptr[PhysicsSyncStateResource]()
    var kcc_proxies_empty = Bool(pw_opt) and len(pw_opt.value()[unsafe_offset=0].kcc_proxies) == 0
    var stale_is_null = Bool(sync_opt2) and sync_opt2.value()[unsafe_offset=0].lookup_entity(enter_body_id) == NULL_ENTITY

    if not saw_exit_kcc or not kcc_proxies_empty or not stale_is_null:
        return (
            False,
            "kcc despawn cleanup failed: exit="
            + String(saw_exit_kcc)
            + " proxies_empty="
            + String(kcc_proxies_empty)
            + " stale_null="
            + String(stale_is_null),
        )

    # Also despawn the trigger entity itself while a second KCC is inside
    var kcc_e2 = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](kcc_e2, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](kcc_e2, GlobalTransform.from_local(LocalTransform(Vec3(0.0, 1.0, 0.0))))
    _ = app.world.insert[PrevGlobalTransform](kcc_e2, PrevGlobalTransform())
    _ = app.world.insert[CharacterControllerComponent](
        kcc_e2,
        CharacterControllerComponent(Vec3(0.0, 1.0, 0.0), CharacterControllerConfig(), Vec3()),
    )

    var saw_enter2 = False
    for _ in range(3):
        app.step_one_frame(Float32(1.0 / 60.0))
        var tev_opt = app.resources.get_ptr[TriggerEventsResource]()
        if tev_opt:
            var evs = tev_opt.value()[unsafe_offset=0].events.read_since(cursor)
            for i in range(len(evs)):
                if evs[i].kind == TRIGGER_ENTER:
                    saw_enter2 = True

    _ = app.world.despawn(trig_e)
    var saw_exit_trig = False
    for _ in range(3):
        app.step_one_frame(Float32(1.0 / 60.0))
        var tev_opt = app.resources.get_ptr[TriggerEventsResource]()
        if tev_opt:
            var evs = tev_opt.value()[unsafe_offset=0].events.read_since(cursor)
            for i in range(len(evs)):
                if evs[i].kind == TRIGGER_EXIT:
                    saw_exit_trig = True

    if not saw_enter2 or not saw_exit_trig:
        return (False, "trigger despawn did not emit TRIGGER_EXIT: enter2=" + String(saw_enter2) + " exit=" + String(saw_exit_trig))
    return (True, "")


def _case_bt_empty_nan_dt_and_unclosed_composite() -> Tuple[Bool, String]:
    var empty_bt = BehaviorTree()
    var bb = Blackboard()
    var st_empty = BtExecutionState()
    var res_empty = empty_bt.tick(bb, st_empty, Float32(1.0 / 60.0))
    if res_empty != BT_FAILURE:
        return (False, "empty BehaviorTree must return BT_FAILURE")

    var bt = BehaviorTree()
    _ = bt.begin_sequence()
    _ = bt.add_blackboard_cmp(999, BT_CMP_EQ, Float32(0.0), BT_ABORT_NONE)  # out-of-range key
    _ = bt.add_action(1)
    # Intentionally omit end_composite() to test unclosed composite robustness

    var st = BtExecutionState()
    var nan_dt = Float32(0.0) / Float32(0.0)
    _ = bt.tick(bb, st, Float32(-1.0))
    _ = bt.tick(bb, st, nan_dt)
    if not isfinite(st.time_s) or st.time_s < Float32(0.0):
        return (False, "BehaviorTree state.time_s became non-finite or negative under negative/NaN dt")
    return (True, "")


def _case_input_map_corrupt_json_and_stack_boundaries() -> Tuple[Bool, String]:
    var imap = InputMap()
    _ = imap.register_action("Jump", ACTION_BOOL)
    var ctx = InputContext("Gameplay", 0, False)
    ctx.bind_key("Jump", 32)
    imap.add_context(ctx^)
    imap.push_context("Gameplay")

    var empty_ok = imap.load_json("")
    var bad_syntax_ok = imap.load_json("{bad json")
    var missing_keys_ok = imap.load_json('{"version":1}')
    if empty_ok or bad_syntax_ok or missing_keys_ok:
        return (False, "load_json unexpectedly succeeded on corrupt/incomplete JSON")

    # Verify pre-existing valid bindings survived failed load_json calls
    var inp = InputState()
    inp.begin_frame()
    inp.set_logical_key(32, True)
    imap.update(inp, Float32(1.0 / 60.0))
    if not imap.just_pressed("Jump"):
        return (False, "pre-existing Jump binding lost after failed load_json")

    if imap.rebind_key("NonExistentCtx", "Jump", 65):
        return (False, "rebind_key succeeded on NonExistentCtx")

    var imap_empty = InputMap()
    if imap_empty.pop_context(""):
        return (False, "pop_context succeeded on empty active_stack")

    var inp_nan = InputState()
    inp_nan.begin_frame()
    inp_nan.add_mouse_delta(Float32(0.0) / Float32(0.0), Float32(0.0) / Float32(0.0))
    _ = imap.register_action("Look", ACTION_AXIS2D)
    var look_ctx = InputContext("LookCtx", 1, False)
    look_ctx.bind_mouse_look("Look")
    imap.add_context(look_ctx^)
    imap.push_context("LookCtx")
    imap.update(inp_nan, Float32(-0.5))
    imap.update(inp_nan, Float32(0.0) / Float32(0.0))
    var look_v = imap.axis_2d("Look")
    if not isfinite(look_v[0]) or not isfinite(look_v[1]) or not isfinite(imap.elapsed_time):
        return (False, "NaN mouse delta or negative/NaN dt corrupted InputMap state")
    return (True, "")


def _case_navmesh_unreachable_blocked_and_empty_targets() -> Tuple[Bool, String]:
    # 1. Empty NavMesh
    var empty_nav = NavMesh()
    var p_empty = empty_nav.find_path(Vec3(), Vec3(1.0, 0.0, 1.0))
    if p_empty.found or len(p_empty.waypoints) != 0:
        return (False, "empty NavMesh returned path.found=True")

    # 2. Disconnected islands
    var pw_disc = PhysicsWorld(gravity=0.0)
    _ = pw_disc.add_box(Vec3(-10.0, -0.5, 0.0), Vec3(3.0, 0.5, 3.0), Float32(0.0))
    _ = pw_disc.add_box(Vec3(10.0, -0.5, 0.0), Vec3(3.0, 0.5, 3.0), Float32(0.0))
    var nav_disc = NavMesh.build_from_physics(pw_disc, NavMeshConfig())
    var p_disc = nav_disc.find_path(Vec3(-10.0, 0.1, 0.0), Vec3(10.0, 0.1, 0.0), Float32(5.0))
    if p_disc.found or len(p_disc.waypoints) != 0:
        return (False, "disconnected NavMesh islands returned found=True")

    # 3. Blocking NavObstacle across narrow bridge, then removed
    var pw_conn = PhysicsWorld(gravity=0.0)
    _ = pw_conn.add_box(Vec3(0.0, -0.5, 0.0), Vec3(6.0, 0.5, 1.5), Float32(0.0))
    var nav_conn = NavMesh.build_from_physics(pw_conn, NavMeshConfig())
    if nav_conn.poly_count() == 0:
        return (False, "failed to bake connected floor NavMesh")

    var p_open = nav_conn.find_path(Vec3(-4.5, 0.1, 0.0), Vec3(4.5, 0.1, 0.0), Float32(4.0))
    if not p_open.found:
        return (False, "open corridor path not found before adding NavObstacle")

    _ = nav_conn.add_obstacle(NavObstacle.box(77, Vec3(0.0, 0.5, 0.0), Vec3(1.2, 1.5, 4.0)))
    var p_blocked = nav_conn.find_path(Vec3(-4.5, 0.1, 0.0), Vec3(4.5, 0.1, 0.0), Float32(4.0))
    if p_blocked.found:
        return (False, "blocked NavMesh returned found=True while NavObstacle spanned corridor")

    nav_conn.remove_obstacle(77)
    var p_reopen = nav_conn.find_path(Vec3(-4.5, 0.1, 0.0), Vec3(4.5, 0.1, 0.0), Float32(4.0))
    if not p_reopen.found:
        return (False, "path not restored after removing NavObstacle")
    return (True, "")


def _case_audio_plugin_missing_asset_ref_graceful_noop() -> Tuple[Bool, String]:
    var app = EngineApp(AppConfig(title="t2_audio_missing", headless=True, fixed_hz=60.0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(AudioPlugin(headless=True))
    app.startup()

    var lst = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](lst, LocalTransform(Vec3()))
    _ = app.world.insert[GlobalTransform](lst, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](lst, PrevGlobalTransform())
    _ = app.world.insert[AudioListenerComponent](lst, AudioListenerComponent())

    var em_e = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](em_e, LocalTransform(Vec3(2.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](em_e, GlobalTransform.from_local(LocalTransform(Vec3(2.0, 0.0, 0.0))))
    _ = app.world.insert[PrevGlobalTransform](em_e, PrevGlobalTransform())
    _ = app.world.insert[AudioEmitterComponent](
        em_e,
        AudioEmitterComponent(asset_id=-1, asset_path="audio/missing_sfx.wav", play_on_spawn=True),
    )

    for _ in range(10):
        app.step_one_frame(Float32(1.0 / 60.0))

    var em_opt = app.world.get_ptr[AudioEmitterComponent](em_e)
    if not em_opt or em_opt.value()[unsafe_offset=0].handle.is_valid():
        return (False, "missing asset_id=-1 emitter unexpectedly returned valid SoundHandle")

    var srv = AudioServer(48000, 256)
    _ = srv.play_3d(-1, Vec3())
    _ = srv.play_2d(999999)
    var out_lr = List[Float32]()
    for _ in range(4):
        srv.render_offline_quantum(out_lr, 256)
        for i in range(len(out_lr)):
            if isnan(out_lr[i]) or isinf(out_lr[i]):
                return (False, "non-finite sample rendered for invalid audio asset_id")
    return (True, "")


def _case_pie_camera_no_active_camera_and_multiple_cameras() -> Tuple[Bool, String]:
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
    )
    studio.look_at(Vec3(0.0, 3.0, -8.0), Vec3(0.0, 1.0, 0.0))

    # Edge A: zero CameraComponent entities -> fallback to editor camera without NaN/zero FOV
    if not studio.pie_play():
        return (False, "pie_play failed with zero CameraComponents")
    _ = studio.step_pie_frame(Float32(1.0 / 60.0))
    studio.render_editor_frame(160, 120)
    var rw_opt = studio.app.resources.get_ptr[RenderWorld]()
    if not rw_opt:
        return (False, "missing RenderWorld")
    var cam0 = rw_opt.value()[unsafe_offset=0].camera
    if not isfinite(cam0.fov_y_deg) or cam0.fov_y_deg <= Float32(0.0):
        return (False, "fallback PIE camera has invalid FOV")
    try:
        _ = studio.pie_stop()
    except:
        pass

    # Edge B: two CameraComponents, deactivate first during PIE -> second takes over
    var c1 = studio.app.world.spawn_empty()
    _ = studio.app.world.insert[LocalTransform](c1, LocalTransform(Vec3(15.0, 5.0, 0.0)))
    _ = studio.app.world.insert[GlobalTransform](c1, GlobalTransform.from_local(LocalTransform(Vec3(15.0, 5.0, 0.0))))
    _ = studio.app.world.insert[PrevGlobalTransform](c1, PrevGlobalTransform())
    _ = studio.app.world.insert[CameraComponent](c1, CameraComponent(fov_y_deg=50.0, active=True))

    var c2 = studio.app.world.spawn_empty()
    _ = studio.app.world.insert[LocalTransform](c2, LocalTransform(Vec3(-25.0, 12.0, 5.0)))
    _ = studio.app.world.insert[GlobalTransform](c2, GlobalTransform.from_local(LocalTransform(Vec3(-25.0, 12.0, 5.0))))
    _ = studio.app.world.insert[PrevGlobalTransform](c2, PrevGlobalTransform())
    _ = studio.app.world.insert[CameraComponent](c2, CameraComponent(fov_y_deg=70.0, active=True))

    if not studio.pie_play():
        return (False, "pie_play failed with two CameraComponents")
    _ = studio.step_pie_frame(Float32(1.0 / 60.0))
    studio.render_editor_frame(160, 120)

    # Deactivate c1 and verify c2 is extracted
    var c1_ptr = studio.app.world.get_ptr_mut[CameraComponent](c1)
    if c1_ptr:
        c1_ptr.value()[unsafe_offset=0].active = False
    _ = studio.step_pie_frame(Float32(1.0 / 60.0))
    studio.render_editor_frame(160, 120)

    var rw_opt2 = studio.app.resources.get_ptr[RenderWorld]()
    if not rw_opt2:
        return (False, "missing RenderWorld after camera switch")
    var cam2 = rw_opt2.value()[unsafe_offset=0].camera
    var switched_ok = abs(cam2.eye.x - Float32(-25.0)) < Float32(0.5) and abs(cam2.fov_y_deg - Float32(70.0)) < Float32(0.5)

    var stopped_ok: Bool
    try:
        stopped_ok = studio.pie_stop()
    except:
        stopped_ok = False
    studio.render_editor_frame(160, 120)
    var rps_opt = studio.app.resources.get_ptr[RenderPluginStateResource]()
    var restored_override = Bool(rps_opt) and rps_opt.value()[unsafe_offset=0].editor_camera_override
    if not switched_ok or not stopped_ok or not restored_override:
        return (
            False,
            "multi-camera switch or pie_stop restore failed: switched="
            + String(switched_ok)
            + " stopped="
            + String(stopped_ok)
            + " override="
            + String(restored_override),
        )
    return (True, "")


def _case_input_map_duplicate_push_and_blocks_lower_all() -> Tuple[Bool, String]:
    var imap = InputMap()
    _ = imap.register_action("Fire", ACTION_BOOL)
    _ = imap.register_action("MenuPing", ACTION_BOOL)

    var gp = InputContext("Gameplay", 0, False)
    gp.bind_key("Fire", 70)  # 'F'
    imap.add_context(gp^)

    var ov = InputContext("Overlay", 10, True)  # blocks_lower_all=True, does NOT bind 'F'
    ov.bind_key("MenuPing", 77)  # 'M'
    imap.add_context(ov^)

    imap.push_context("Gameplay")
    imap.push_context("Overlay")
    imap.push_context("Gameplay")  # duplicate push re-orders without duplicating

    if len(imap.active_stack) != 2:
        return (False, "duplicate push_context produced stack length " + String(len(imap.active_stack)))

    var inp = InputState()
    inp.begin_frame()
    inp.set_logical_key(70, True)  # 'F' (unbound in Overlay)
    inp.set_logical_key(77, True)  # 'M'
    imap.update(inp, Float32(1.0 / 60.0))
    if imap.is_down("Fire") or not imap.is_down("MenuPing"):
        return (False, "blocks_lower_all=True failed to block unbound lower action 'Fire'")
    return (True, "")


def _case_kcc_filter_mismatch_on_trigger_volume() -> Tuple[Bool, String]:
    var app = EngineApp(AppConfig(title="t2_kcc_filter", headless=True, fixed_hz=60.0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.startup()

    var trig_e = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](trig_e, LocalTransform(Vec3(1.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](trig_e, GlobalTransform.from_local(LocalTransform(Vec3(1.0, 1.0, 0.0))))
    _ = app.world.insert[PrevGlobalTransform](trig_e, PrevGlobalTransform())
    var trig_col = ColliderComponent.box(Vec3(0.75, 1.5, 0.75), is_trigger=True)
    trig_col.filter = CollisionFilter(CAT_SENSOR, CAT_PROP_PIECE, 0)  # rejects default CAT_DEBRIS KCC
    _ = app.world.insert[ColliderComponent](trig_e, trig_col^)

    var kcc_e = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](kcc_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](kcc_e, GlobalTransform.from_local(LocalTransform(Vec3(0.0, 1.0, 0.0))))
    _ = app.world.insert[PrevGlobalTransform](kcc_e, PrevGlobalTransform())
    _ = app.world.insert[CharacterControllerComponent](
        kcc_e,
        CharacterControllerComponent(Vec3(0.0, 1.0, 0.0), CharacterControllerConfig(), Vec3(6.0, 0.0, 0.0)),
    )

    var cursor = 0
    var seen_events = 0
    for _ in range(40):
        app.step_one_frame(Float32(1.0 / 60.0))
        var tev_opt = app.resources.get_ptr[TriggerEventsResource]()
        if tev_opt:
            var evs = tev_opt.value()[unsafe_offset=0].events.read_since(cursor)
            for i in range(len(evs)):
                if evs[i].kind == TRIGGER_ENTER or evs[i].kind == TRIGGER_STAY or evs[i].kind == TRIGGER_EXIT:
                    seen_events += 1

    if seen_events != 0:
        return (False, "mismatched trigger filter emitted " + String(seen_events) + " events")
    return (True, "")


def main() raises:
    var runner = StudioSuiteRunner("t2_m2_behaviour_boundary")

    var r1 = _case_kcc_despawn_inside_trigger_and_stale_lookup()
    _ = runner.check("t2_m2_kcc_despawn_inside_trigger_and_stale_lookup", r1[0], r1[1])

    var r2 = _case_bt_empty_nan_dt_and_unclosed_composite()
    _ = runner.check("t2_m2_bt_empty_nan_dt_and_unclosed_composite", r2[0], r2[1])

    var r3 = _case_input_map_corrupt_json_and_stack_boundaries()
    _ = runner.check("t2_m2_input_map_corrupt_json_and_stack_boundaries", r3[0], r3[1])

    var r4 = _case_navmesh_unreachable_blocked_and_empty_targets()
    _ = runner.check("t2_m2_navmesh_unreachable_blocked_and_empty_targets", r4[0], r4[1])

    var r5 = _case_audio_plugin_missing_asset_ref_graceful_noop()
    _ = runner.check("t2_m2_audio_plugin_missing_asset_ref_graceful_noop", r5[0], r5[1])

    var r6 = _case_pie_camera_no_active_camera_and_multiple_cameras()
    _ = runner.check("t2_m2_pie_camera_no_active_camera_and_multiple_cameras", r6[0], r6[1])

    var r7 = _case_input_map_duplicate_push_and_blocks_lower_all()
    _ = runner.check("t2_m2_input_map_duplicate_push_and_blocks_lower_all", r7[0], r7[1])

    var r8 = _case_kcc_filter_mismatch_on_trigger_volume()
    _ = runner.check("t2_m2_kcc_filter_mismatch_on_trigger_volume", r8[0], r8[1])

    runner.finish()
