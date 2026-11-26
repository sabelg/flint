# AREA: m2
"""Tier 1 Opaque API Tests: Area D — Plugins, AI, InputMap, HUD, Camera & Hot-Reload ABI (F22, F24, F25)."""

from std.collections import List
from flint.ai import (
    BT_ABORT_LOWER_PRIORITY,
    BT_CMP_GT,
    BT_FAILURE,
    BT_RUNNING,
    BT_SUCCESS,
    BehaviorTree,
    Blackboard,
    BtExecutionState,
    CURVE_LINEAR,
    UtilityAiScorer,
    UtilityConsideration,
    UtilityOption,
)
from flint.app import AppConfig, EngineApp, FLINT_PLUGIN_API_VERSION, HotReloadHost
from flint.audio.plugin import AudioEmitterComponent, AudioListenerComponent, AudioPlugin, register_audio_codecs
from flint.core.math import Vec3
from flint.ecs import Entity
from flint.input import ACTION_AXIS2D, ACTION_BOOL, InputContext, InputMap
from flint.physics.character import CharacterControllerConfig
from flint.physics.plugin import (
    CharacterControllerComponent,
    ColliderComponent,
    PhysicsPlugin,
    PhysicsSyncStateResource,
    TriggerEventsResource,
    register_physics_codecs,
)
from flint.physics.world import TRIGGER_ENTER, TRIGGER_EXIT, TRIGGER_STAY
from flint.platform.input import InputState
from flint.render.plugin import (
    CameraComponent,
    RenderPlugin,
    RenderPluginStateResource,
    register_render_codecs,
)
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, RenderWorld, ScalableRenderer
from flint.scene import (
    ComponentRegistry,
    GlobalTransform,
    LocalTransform,
    PrevGlobalTransform,
    Scene,
    TransformPlugin,
)
from flint.studio import FlintStudioState, GIZMO_MODE_TRANSLATE, PIE_PLAYING
from flint.ui.context import UI_KEY_DELETE, UI_KEY_LEFT_SHIFT, UiContext, UiTheme
from flint.ui.flexbox_dock import FlexStyle
from e2e_studio.common.studio_test_utils import (
    StudioSuiteRunner,
    make_tmp_dir,
    read_text,
    rm_rf_tmp,
    write_text,
)


def _case_plugin_api_version_2(tmp: String) -> Tuple[Bool, String]:
    var host = HotReloadHost(tmp + "/hr_v2_baseline")
    var baseline_ok = (not host.is_loaded()) and host.reload_generation == 0
    if not baseline_ok:
        return (False, "HotReloadHost initial state invalid")
    if Int(FLINT_PLUGIN_API_VERSION) != 2:
        return (
            False,
            "FLINT_PLUGIN_API_VERSION mismatch: expected 2, got "
            + String(FLINT_PLUGIN_API_VERSION)
            + " (M5 FlintPluginApiV2 pending)",
        )
    return (True, "")


def _case_pie_camera_possession_shift_f1() -> Tuple[Bool, String]:
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
    )
    studio.look_at(Vec3(0.0, 2.0, -5.0), Vec3(0.0, 1.0, 0.0))

    var cam_e = studio.app.world.spawn_empty()
    _ = studio.app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(40.0, 15.0, 40.0)))
    _ = studio.app.world.insert[GlobalTransform](cam_e, GlobalTransform.from_local(LocalTransform(Vec3(40.0, 15.0, 40.0))))
    _ = studio.app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
    _ = studio.app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=60.0, active=True))

    # Edit mode: editor camera active
    studio.render_editor_frame(160, 120)
    var rps_ed = studio.app.resources.get_ptr[RenderPluginStateResource]()
    var rw_ed = studio.app.resources.get_ptr[RenderWorld]()
    if not rps_ed or not rw_ed:
        return (False, "missing RenderPluginStateResource or RenderWorld in Edit mode")
    if not rps_ed.value()[unsafe_offset=0].editor_camera_override:
        return (False, "Edit mode did not enforce editor_camera_override == True")
    if abs(rw_ed.value()[unsafe_offset=0].camera.eye.x) > Float32(0.5):
        return (False, "Edit mode camera hijacked by CameraComponent")

    # Enter PIE: game camera possessed
    if not studio.pie_play():
        return (False, "pie_play failed")
    _ = studio.step_pie_frame(Float32(1.0 / 60.0))
    studio.render_editor_frame(160, 120)
    var rps_pie = studio.app.resources.get_ptr[RenderPluginStateResource]()
    var rw_pie = studio.app.resources.get_ptr[RenderWorld]()
    if not rps_pie or not rw_pie:
        return (False, "missing Render resources in PIE")
    if rps_pie.value()[unsafe_offset=0].editor_camera_override:
        return (False, "PIE did not possess game camera (editor_camera_override remained True)")
    if abs(rw_pie.value()[unsafe_offset=0].camera.eye.x - Float32(40.0)) > Float32(0.5):
        return (False, "PIE camera eye did not track CameraComponent at (40,15,40)")

    # Shift+F1 -> eject to editor camera while PIE remains PIE_PLAYING
    var inp_eject = InputState()
    inp_eject.begin_frame()
    inp_eject.set_logical_key(UI_KEY_LEFT_SHIFT, True)
    inp_eject.set_logical_key(290, True)  # F1
    _ = studio.process_editor_input(inp_eject, Float32(1.0 / 60.0))
    studio.render_editor_frame(160, 120)

    var rps_ej = studio.app.resources.get_ptr[RenderPluginStateResource]()
    var ejected_ok = (
        studio.pie.mode == PIE_PLAYING
        and Bool(rps_ej)
        and rps_ej.value()[unsafe_offset=0].editor_camera_override
    )
    if not ejected_ok:
        try:
            _ = studio.pie_stop()
        except:
            pass
        return (False, "Shift+F1 did not eject to editor camera during PIE (M5 Shift+F1 pending)")

    # Shift+F1 again -> re-possess game camera
    var inp_possess = InputState()
    inp_possess.begin_frame()
    inp_possess.set_logical_key(UI_KEY_LEFT_SHIFT, True)
    inp_possess.set_logical_key(290, True)  # F1
    _ = studio.process_editor_input(inp_possess, Float32(1.0 / 60.0))
    studio.render_editor_frame(160, 120)
    var rps_re = studio.app.resources.get_ptr[RenderPluginStateResource]()
    if not rps_re or rps_re.value()[unsafe_offset=0].editor_camera_override:
        try:
            _ = studio.pie_stop()
        except:
            pass
        return (False, "second Shift+F1 did not re-possess game camera")

    var stopped_ok: Bool
    try:
        stopped_ok = studio.pie_stop()
    except:
        stopped_ok = False
    studio.render_editor_frame(160, 120)
    var rps_stop = studio.app.resources.get_ptr[RenderPluginStateResource]()
    if not stopped_ok or not rps_stop or not rps_stop.value()[unsafe_offset=0].editor_camera_override:
        return (False, "pie_stop did not restore editor camera")
    return (True, "")


def _case_kcc_trigger_enter_stay_exit_and_lookup() -> Tuple[Bool, String]:
    var app = EngineApp(AppConfig(title="t1_kcc_trig", headless=True, fixed_hz=60.0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.startup()

    var trig_e = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](trig_e, LocalTransform(Vec3(2.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](trig_e, GlobalTransform.from_local(LocalTransform(Vec3(2.0, 1.0, 0.0))))
    _ = app.world.insert[PrevGlobalTransform](trig_e, PrevGlobalTransform())
    _ = app.world.insert[ColliderComponent](trig_e, ColliderComponent.box(Vec3(0.8, 1.5, 0.8), is_trigger=True))

    var kcc_e = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](kcc_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](kcc_e, GlobalTransform.from_local(LocalTransform(Vec3(0.0, 1.0, 0.0))))
    _ = app.world.insert[PrevGlobalTransform](kcc_e, PrevGlobalTransform())
    _ = app.world.insert[CharacterControllerComponent](
        kcc_e,
        CharacterControllerComponent(Vec3(0.0, 1.0, 0.0), CharacterControllerConfig(), Vec3(6.0, 0.0, 0.0)),
    )

    var cursor = 0
    var saw_enter = False
    var saw_stay = False
    var saw_exit = False
    var lookup_ok = False

    for _ in range(60):
        app.step_one_frame(Float32(1.0 / 60.0))
        var tev_opt = app.resources.get_ptr[TriggerEventsResource]()
        var sync_opt = app.resources.get_ptr[PhysicsSyncStateResource]()
        if tev_opt and sync_opt:
            var evs = tev_opt.value()[unsafe_offset=0].events.read_since(cursor)
            for i in range(len(evs)):
                var ev = evs[i]
                if ev.kind == TRIGGER_ENTER:
                    saw_enter = True
                    if sync_opt.value()[unsafe_offset=0].lookup_entity(ev.body_id) == kcc_e:
                        lookup_ok = True
                elif ev.kind == TRIGGER_STAY:
                    saw_stay = True
                    if sync_opt.value()[unsafe_offset=0].lookup_entity(ev.body_id) == kcc_e:
                        lookup_ok = True
                elif ev.kind == TRIGGER_EXIT:
                    saw_exit = True

    if not (saw_enter and saw_stay and saw_exit and lookup_ok):
        return (
            False,
            "enter="
            + String(saw_enter)
            + " stay="
            + String(saw_stay)
            + " exit="
            + String(saw_exit)
            + " lookup="
            + String(lookup_ok),
        )
    return (True, "")


def _case_behavior_tree_reactive_and_utility() -> Tuple[Bool, String]:
    var bb = Blackboard()
    var k_alert = bb.def_key("alert", Float32(0.0))
    var k_patrol = bb.def_key("patrol", Float32(0.9))
    var k_idle = bb.def_key("idle", Float32(0.5))

    var scorer = UtilityAiScorer(use_geometric_mean=True)
    var opt_patrol = UtilityOption(2, Float32(1.0))
    opt_patrol.add_consideration(UtilityConsideration(k_patrol, Float32(0.0), Float32(1.0), CURVE_LINEAR))
    var opt_idle = UtilityOption(3, Float32(0.2))
    opt_idle.add_consideration(UtilityConsideration(k_idle, Float32(0.0), Float32(1.0), CURVE_LINEAR))
    _ = scorer.add_option(opt_patrol^)
    _ = scorer.add_option(opt_idle^)

    var bt = BehaviorTree()
    _ = bt.begin_selector()
    # High-priority branch with BT_ABORT_LOWER_PRIORITY + cooldown + inverter
    _ = bt.begin_sequence()
    _ = bt.add_blackboard_cmp(k_alert, BT_CMP_GT, Float32(0.5), BT_ABORT_LOWER_PRIORITY)
    _ = bt.begin_cooldown(Float32(0.25))
    _ = bt.begin_inverter()
    _ = bt.add_action(10)  # action 10 returns BT_FAILURE -> inverted to BT_SUCCESS
    bt.end_composite()     # inverter
    bt.end_composite()     # cooldown
    bt.end_composite()     # high-priority sequence
    # Low-priority utility branch
    _ = bt.begin_utility_select(scorer^)
    _ = bt.add_action(2)
    _ = bt.add_action(3)
    bt.end_composite()     # utility select
    bt.end_composite()     # root selector

    var st = BtExecutionState()
    st.set_action_status(2, BT_RUNNING)
    st.set_action_status(3, BT_SUCCESS)
    st.set_action_status(10, BT_FAILURE)  # inverter turns this into BT_SUCCESS

    var r1 = bt.tick(bb, st, Float32(1.0 / 60.0))
    if r1 != BT_RUNNING or st.active_action_id != 2:
        return (False, "expected low-priority BT_RUNNING on action 2, got status=" + String(r1) + " action=" + String(st.active_action_id))

    # Flip blackboard alert > 0.5 -> reactive abort preempts running action 2!
    bb.set_float(k_alert, Float32(1.0))
    var r2 = bt.tick(bb, st, Float32(1.0 / 60.0))
    if r2 != BT_SUCCESS or st.aborts_triggered != 1 or st.active_action_id != 10:
        return (
            False,
            "expected reactive abort to action 10: status="
            + String(r2)
            + " aborts="
            + String(st.aborts_triggered)
            + " action="
            + String(st.active_action_id),
        )
    return (True, "")


def _case_input_map_contexts_rebind_json_and_studio_play(tmp: String) -> Tuple[Bool, String]:
    # Subcase A: InputMap rebind_key, JSON round-trip, and context stack masking
    var imap = InputMap()
    _ = imap.register_action("Move", ACTION_AXIS2D)
    _ = imap.register_action("Jump", ACTION_BOOL)
    _ = imap.register_action("Pause", ACTION_BOOL)

    var gp = InputContext("Gameplay", 0, False)
    gp.bind_key("Jump", 32)  # Space
    gp.bind_key("Move", 68, scale_x=Float32(1.0), scale_y=Float32(0.0))
    imap.add_context(gp^)

    var pm = InputContext("PauseMenu", 10, False)
    pm.bind_key("Pause", 32, consume=True)
    imap.add_context(pm^)
    imap.push_context("Gameplay")

    if not imap.rebind_key("Gameplay", "Jump", 74):  # 'J'
        return (False, "rebind_key failed")

    var json_path = tmp + "/bindings.json"
    if not write_text(json_path, imap.save_json()):
        return (False, "failed to write bindings.json")

    var imap2 = InputMap()
    if not imap2.load_json(read_text(json_path)):
        return (False, "InputMap.load_json failed on saved bindings.json")

    var inp_j = InputState()
    inp_j.begin_frame()
    inp_j.set_logical_key(74, True)
    imap2.update(inp_j, Float32(1.0 / 60.0))
    if not imap2.just_pressed("Jump"):
        return (False, "rebound key 'J' did not trigger Jump after JSON reload")

    # Also test context masking on a shared key (Space)
    var imap_mask = InputMap()
    _ = imap_mask.register_action("Jump", ACTION_BOOL)
    _ = imap_mask.register_action("Pause", ACTION_BOOL)
    var g_ctx = InputContext("Gameplay", 0, False)
    g_ctx.bind_key("Jump", 32)
    var p_ctx = InputContext("PauseMenu", 10, False)
    p_ctx.bind_key("Pause", 32, consume=True)
    imap_mask.add_context(g_ctx^)
    imap_mask.add_context(p_ctx^)
    imap_mask.push_context("Gameplay")
    imap_mask.push_context("PauseMenu")

    var inp_sp = InputState()
    inp_sp.begin_frame()
    inp_sp.set_logical_key(32, True)
    imap_mask.update(inp_sp, Float32(1.0 / 60.0))
    if not imap_mask.just_pressed("Pause") or imap_mask.just_pressed("Jump"):
        return (False, "PauseMenu did not mask Space from Gameplay")

    # Subcase B: StudioPlay context suppresses StudioEditor hotkeys in PIE_PLAYING
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
    )
    var eid = studio.spawn_entity("PieHotkeyTarget")
    studio.select_entity(eid)
    studio.gizmo_mode = GIZMO_MODE_TRANSLATE
    if not studio.pie_play():
        return (False, "pie_play failed in StudioPlay check")

    var inp_del = InputState()
    inp_del.begin_frame()
    inp_del.set_logical_key(UI_KEY_DELETE, True)
    inp_del.set_logical_key(69, True)  # 'E' (gizmo_rotate in StudioEditor)
    _ = studio.process_editor_input(inp_del, Float32(1.0 / 60.0))

    var alive_after = studio.is_alive_id(eid)
    var gizmo_unchanged = studio.gizmo_mode == GIZMO_MODE_TRANSLATE
    try:
        _ = studio.pie_stop()
    except:
        pass
    if not alive_after or not gizmo_unchanged:
        return (
            False,
            "StudioEditor hotkeys fired during PIE_PLAYING (alive="
            + String(alive_after)
            + " gizmo_unchanged="
            + String(gizmo_unchanged)
            + "; M5 StudioPlay context pending)",
        )
    return (True, "")


def _case_game_hud_overlay_render() -> Tuple[Bool, String]:
    var app = EngineApp(AppConfig(title="t1_hud_render", headless=True, fixed_hz=60.0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    app.startup()

    app.step_one_frame(Float32(1.0 / 60.0))
    var rend_opt1 = app.resources.get_ptr[ScalableRenderer]()
    if not rend_opt1:
        return (False, "missing ScalableRenderer")
    var px_base = rend_opt1.value()[unsafe_offset=0].readback_pixels()

    var hud = UiContext(preset_hud=True)
    var dark = UiTheme.dark_studio()
    if hud.theme.bg_panel == dark.bg_panel:
        return (False, "UiContext(preset_hud=True) did not apply UiTheme.game_hud()")

    hud.begin_frame(InputState(), Float32(160.0), Float32(120.0), Float32(1.0 / 60.0))
    hud.begin_box("hud_root", FlexStyle.col(gap=4.0, pad=8.0))
    hud.label("SCORE 420")
    _ = hud.button("hud_btn", "PLAY", Float32(110.0), Float32(32.0))
    hud.end_box()
    hud.end_frame()

    var rps_opt = app.resources.get_ptr_mut[RenderPluginStateResource]()
    if not rps_opt:
        return (False, "missing RenderPluginStateResource")
    for i in range(len(hud.draw_list.prims)):
        rps_opt.value()[unsafe_offset=0].ui_draw_list.prims.append(hud.draw_list.prims[i])

    app.step_one_frame(Float32(1.0 / 60.0))
    var rend_opt2 = app.resources.get_ptr[ScalableRenderer]()
    if not rend_opt2:
        return (False, "missing ScalableRenderer after HUD draw")
    var px_hud = rend_opt2.value()[unsafe_offset=0].readback_pixels()

    if len(px_base) != len(px_hud) or len(px_hud) == 0:
        return (False, "invalid readback pixel buffer lengths")
    var diff_count = 0
    for i in range(len(px_hud)):
        if px_base[i] != px_hud[i]:
            diff_count += 1
    if diff_count <= 20:
        return (False, "HUD overlay produced only " + String(diff_count) + " changed pixels")
    return (True, "")


def _case_scene_transition_roundtrip(tmp: String) -> Tuple[Bool, String]:
    var reg = ComponentRegistry.with_builtins()
    register_physics_codecs(reg)
    register_render_codecs(reg)
    register_audio_codecs(reg)

    var w1 = EngineApp(AppConfig(title="t1_scene_rt", headless=True))
    w1.add_plugin(TransformPlugin())
    w1.add_plugin(PhysicsPlugin(gravity=0.0))
    w1.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=64, height=64))
    w1.add_plugin(AudioPlugin(headless=True))
    reg.ensure_registered_in_world(w1.world)

    var e_cam = w1.world.spawn_empty()
    _ = w1.world.insert[LocalTransform](e_cam, LocalTransform(Vec3(0.0, 2.0, -4.0)))
    _ = w1.world.insert[GlobalTransform](e_cam, GlobalTransform.from_local(LocalTransform(Vec3(0.0, 2.0, -4.0))))
    _ = w1.world.insert[PrevGlobalTransform](e_cam, PrevGlobalTransform())
    _ = w1.world.insert[CameraComponent](e_cam, CameraComponent(fov_y_deg=65.0, active=True))

    var e_trig = w1.world.spawn_empty()
    _ = w1.world.insert[LocalTransform](e_trig, LocalTransform(Vec3(3.0, 1.0, 0.0)))
    _ = w1.world.insert[GlobalTransform](e_trig, GlobalTransform.from_local(LocalTransform(Vec3(3.0, 1.0, 0.0))))
    _ = w1.world.insert[PrevGlobalTransform](e_trig, PrevGlobalTransform())
    _ = w1.world.insert[ColliderComponent](e_trig, ColliderComponent.box(Vec3(1.0, 1.0, 1.0), is_trigger=True))

    var e_kcc = w1.world.spawn_empty()
    _ = w1.world.insert[LocalTransform](e_kcc, LocalTransform(Vec3(-2.0, 1.0, 0.0)))
    _ = w1.world.insert[GlobalTransform](e_kcc, GlobalTransform.from_local(LocalTransform(Vec3(-2.0, 1.0, 0.0))))
    _ = w1.world.insert[PrevGlobalTransform](e_kcc, PrevGlobalTransform())
    _ = w1.world.insert[CharacterControllerComponent](e_kcc, CharacterControllerComponent(Vec3(-2.0, 1.0, 0.0)))

    var e_aud = w1.world.spawn_empty()
    _ = w1.world.insert[LocalTransform](e_aud, LocalTransform(Vec3(0.0, 0.0, 2.0)))
    _ = w1.world.insert[GlobalTransform](e_aud, GlobalTransform.from_local(LocalTransform(Vec3(0.0, 0.0, 2.0))))
    _ = w1.world.insert[PrevGlobalTransform](e_aud, PrevGlobalTransform())
    _ = w1.world.insert[AudioEmitterComponent](e_aud, AudioEmitterComponent(asset_id=-1, asset_path="audio/sfx.wav"))
    _ = w1.world.insert[AudioListenerComponent](e_aud, AudioListenerComponent())

    var scn = Scene("TransitionLevel")
    var flscn_path = tmp + "/level.flscn"
    var flscnb_path = tmp + "/level.flscnb"
    var saved_ok: Bool
    try:
        saved_ok = scn.save_flscn_file(flscn_path, w1.world, reg) and scn.save_flscnb_file(flscnb_path, w1.world, reg)
    except:
        saved_ok = False
    if not saved_ok:
        return (False, "failed to save .flscn/.flscnb scene files")

    var w_json = EngineApp(AppConfig(title="t1_scene_json", headless=True))
    var w_bin = EngineApp(AppConfig(title="t1_scene_bin", headless=True))
    var scn_j = Scene()
    var scn_b = Scene()
    var lj: List[Entity]
    var lb: List[Entity]
    try:
        lj = scn_j.load_flscn_file(flscn_path, w_json.world, reg)
        lb = scn_b.load_flscnb_file(flscnb_path, w_bin.world, reg)
    except e:
        return (False, "failed loading scene roundtrip: " + String(e))

    if len(lj) != 4 or len(lb) != 4 or w_json.world.entity_count() != 4 or w_bin.world.entity_count() != 4:
        return (False, "entity count mismatch after .flscn/.flscnb roundtrip")

    if not w_json.world.has[CameraComponent](lj[0]) or not w_bin.world.has[CameraComponent](lb[0]):
        return (False, "CameraComponent missing after .flscn/.flscnb roundtrip")
    if (
        abs(w_json.world.get[CameraComponent](lj[0]).fov_y_deg - Float32(65.0)) > Float32(1.0e-3)
        or abs(w_bin.world.get[CameraComponent](lb[0]).fov_y_deg - Float32(65.0)) > Float32(1.0e-3)
        or not w_json.world.get[CameraComponent](lj[0]).active
        or not w_bin.world.get[CameraComponent](lb[0]).active
    ):
        return (False, "CameraComponent field values mismatch after .flscn/.flscnb roundtrip")

    if not w_json.world.has[ColliderComponent](lj[1]) or not w_bin.world.has[ColliderComponent](lb[1]):
        return (False, "ColliderComponent missing after .flscn/.flscnb roundtrip")
    if not w_json.world.get[ColliderComponent](lj[1]).is_trigger or not w_bin.world.get[ColliderComponent](lb[1]).is_trigger:
        return (False, "ColliderComponent.is_trigger lost after .flscn/.flscnb roundtrip")

    if not w_json.world.has[CharacterControllerComponent](lj[2]) or not w_bin.world.has[CharacterControllerComponent](lb[2]):
        return (False, "CharacterControllerComponent missing after .flscn/.flscnb roundtrip")

    if not w_json.world.has[AudioEmitterComponent](lj[3]) or not w_bin.world.has[AudioEmitterComponent](lb[3]):
        return (False, "AudioEmitterComponent missing after .flscn/.flscnb roundtrip")
    if (
        w_json.world.get[AudioEmitterComponent](lj[3]).asset_path != "audio/sfx.wav"
        or w_bin.world.get[AudioEmitterComponent](lb[3]).asset_path != "audio/sfx.wav"
    ):
        return (False, "AudioEmitterComponent.asset_path mismatch after .flscn/.flscnb roundtrip")

    if not w_json.world.has[AudioListenerComponent](lj[3]) or not w_bin.world.has[AudioListenerComponent](lb[3]):
        return (False, "AudioListenerComponent missing after .flscn/.flscnb roundtrip")
    return (True, "")


def main() raises:
    var tmp = make_tmp_dir("t1_m2_flow")
    var runner = StudioSuiteRunner("t1_m2_plugins_ai_flow")

    var r1 = _case_plugin_api_version_2(tmp)
    _ = runner.check("t1_m2_f22_plugin_api_version_2", r1[0], r1[1])

    var r2 = _case_pie_camera_possession_shift_f1()
    _ = runner.check("t1_m2_f24_pie_camera_possession_shift_f1", r2[0], r2[1])

    var r3 = _case_kcc_trigger_enter_stay_exit_and_lookup()
    _ = runner.check("t1_m2_f24_kcc_trigger_enter_stay_exit_and_lookup", r3[0], r3[1])

    var r4 = _case_behavior_tree_reactive_and_utility()
    _ = runner.check("t1_m2_f25_behavior_tree_reactive_and_utility", r4[0], r4[1])

    var r5 = _case_input_map_contexts_rebind_json_and_studio_play(tmp)
    _ = runner.check("t1_m2_f25_input_map_contexts_rebind_json_and_studio_play", r5[0], r5[1])

    var r6 = _case_game_hud_overlay_render()
    _ = runner.check("t1_m2_f25_game_hud_overlay_render", r6[0], r6[1])

    var r7 = _case_scene_transition_roundtrip(tmp)
    _ = runner.check("t1_m2_f25_scene_transition_roundtrip", r7[0], r7[1])

    rm_rf_tmp(tmp)
    runner.finish()
