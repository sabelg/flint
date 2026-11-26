# AREA: m0 m1 m2 m3
"""Tier 3 Cross-Feature Pairwise Tests (Pure Mojo 1.2): Content, Assets, Plugins, Scene v2 & Edit/PIE Runtime."""

from std.collections import List
from std.ffi import external_call
from std.math import abs

from flint.ai import (
    BT_ABORT_LOWER_PRIORITY,
    BT_CMP_GT,
    BT_RUNNING,
    BT_SUCCESS,
    BehaviorTree,
    Blackboard,
    BtExecutionState,
)
from flint.ai.behavior_tree import BehaviorTreeComponent
from flint.anim.flrig import FrigJointDef, FrigRigDef, load_flrig_file, save_flrig_file
from flint.anim.plugin import AnimPlugin, AnimatorComponent, RigComponent, register_anim_codecs
from flint.app.engine_app import (
    AppConfig,
    EngineApp,
    FrameTime,
    Resource,
    Resources,
    STAGE_POST_UPDATE,
    STAGE_PRE_UPDATE,
    STAGE_UPDATE,
)
from flint.app.flow import GameFlow
from flint.app.hot_reload import FLINT_PLUGIN_API_VERSION, FlintPluginApiV2, PodComponentSchema
from flint.assets.asset_ref import AssetRef
from flint.assets.flmat import FlmatAsset, MAT_SHADING_PBR, load_flmat_file, save_flmat_file
from flint.assets.flpk_v3 import ASSET_KIND_RAW, AssetUuid
from flint.assets.server import AssetServer
from flint.audio.plugin import AudioEmitterComponent, AudioListenerComponent, AudioPlugin
from flint.audio.server import AudioServer, BUS_SFX
from flint.core.jobs import JobPool
from flint.core.math import Quat, Vec3
from flint.ecs.world import Entity, Events, ParallelCommands, World, comp_hash64
from flint.nav.navmesh import NavMesh
from flint.nav.plugin import NavPlugin
from flint.physics.character import CharacterControllerConfig
from flint.physics.plugin import (
    CharacterControllerComponent,
    ColliderComponent,
    OnEnter,
    OnExit,
    PhysicsPlugin,
    RigidBodyComponent,
    TriggerEventsResource,
    entity_from_body,
)
from flint.physics.world import PhysicsWorld, TRIGGER_ENTER, TRIGGER_EXIT, TriggerVolume
from flint.platform.audio import decode_wav
from flint.platform.os import make_dirs
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import (
    CameraComponent,
    MaterialOverrideComponent,
    MeshRendererComponent,
    RenderPlugin,
    RenderPluginStateResource,
    register_render_codecs,
)
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RenderWorld,
    ScalableRenderer,
)
from flint.scene.scene import SCENE_SCHEMA_VERSION, ComponentRegistry, Scene
from flint.scene.transform_hierarchy import (
    GlobalTransform,
    LocalTransform,
    PrevGlobalTransform,
    TransformPlugin,
)
from flint.studio.studio_app import FlintStudioState, PIE_PAUSED, PIE_PLAYING, PIE_STOPPED
from e2e_studio.common.studio_test_utils import (
    StudioSuiteRunner,
    make_tmp_dir,
    read_bytes,
    rm_rf_tmp,
    write_minimal_wav,
    write_text,
)


struct PairProbeResource(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.e2e_studio.PairProbeResource")
    var pre_count: Int
    var update_count: Int
    var post_count: Int
    var events: Events[Int]

    def __init__(out self):
        self.pre_count = 0
        self.update_count = 0
        self.post_count = 0
        self.events = Events[Int]()


def probe_pre_sys(
    _w: Pointer[World, MutAnyOrigin],
    r_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var p = r_ptr[unsafe_offset=0].get_ptr_mut[PairProbeResource]()
    if p:
        p.value()[unsafe_offset=0].pre_count += 1


def probe_update_sys(
    _w: Pointer[World, MutAnyOrigin],
    r_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var p = r_ptr[unsafe_offset=0].get_ptr_mut[PairProbeResource]()
    if p:
        p.value()[unsafe_offset=0].update_count += 1
        _ = p.value()[unsafe_offset=0].events.send(p.value()[unsafe_offset=0].update_count)


def probe_post_sys(
    _w: Pointer[World, MutAnyOrigin],
    r_ptr: Pointer[Resources, MutAnyOrigin],
    _p: Pointer[JobPool, MutAnyOrigin],
    _c: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    var p = r_ptr[unsafe_offset=0].get_ptr_mut[PairProbeResource]()
    if p:
        p.value()[unsafe_offset=0].post_count += 1


def probe_swap_hook(r_ptr: Pointer[Resources, MutAnyOrigin]):
    var p = r_ptr[unsafe_offset=0].get_ptr_mut[PairProbeResource]()
    if p:
        p.value()[unsafe_offset=0].events.swap_buffers()


# ---------------------------------------------------------------------------
# Pair 1: AssetRef + .flmat + MaterialOverrideComponent isolation + RenderWorld.overlay_lines
# ---------------------------------------------------------------------------
def test_pair1_asset_ref_flmat_override_and_overlay_lines(
    mut runner: StudioSuiteRunner, tmp: String
) raises:
    var asset_root = tmp + "/assets"
    _ = make_dirs(asset_root + "/materials")
    var mat_path = asset_root + "/materials/shared.flmat"
    var flmat = FlmatAsset("SharedMat")
    flmat.shading_mode = MAT_SHADING_PBR
    flmat.base_color = Vec3(0.4, 0.4, 0.4)
    flmat.roughness = Float32(0.6)
    flmat.metallic = Float32(0.1)
    var saved_mat = save_flmat_file(mat_path, flmat)
    var ref_mat = AssetRef(AssetUuid.from_path("materials/shared.flmat"), "materials/shared.flmat", ASSET_KIND_RAW)

    var asrv = AssetServer(enable_watcher=False)
    _ = asrv.watch_loose_directory(asset_root)
    var h_raw = asrv.load_raw(ref_mat.rel_path)
    var p_raw = asrv.raws.get_ptr(h_raw)
    var raw_ok = Bool(p_raw) and len(p_raw.value()[unsafe_offset=0].bytes) > 0
    var loaded_mat = load_flmat_file(mat_path)
    var mat_ok = (
        saved_mat
        and raw_ok
        and loaded_mat.shading_mode == MAT_SHADING_PBR
        and abs(loaded_mat.base_color.x - Float32(0.4)) < Float32(1.0e-4)
        and abs(loaded_mat.roughness - Float32(0.6)) < Float32(1.0e-4)
    )
    asrv.close()

    var app = EngineApp(AppConfig(title="pair1_mat_overlay", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120))
    app.startup()

    var rend = app.resources.get_ptr_mut[ScalableRenderer]()
    var rw_ptr = app.resources.get_ptr_mut[RenderWorld]()
    var ok = Bool(rend) and Bool(rw_ptr) and mat_ok and ref_mat.rel_path == "materials/shared.flmat"
    if ok:
        var shared_mat = MaterialAsset(
            base_color=loaded_mat.base_color,
            roughness=loaded_mat.roughness,
            metallic=loaded_mat.metallic,
        )
        var shared_id = rend.value()[unsafe_offset=0].register_material(shared_mat)

        var e_a = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_a, LocalTransform(Vec3(-1.5, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e_a, GlobalTransform())
        _ = app.world.insert[PrevGlobalTransform](e_a, PrevGlobalTransform())
        _ = app.world.insert[MeshRendererComponent](e_a, MeshRendererComponent(0, shared_id))
        var ov_a = MaterialOverrideComponent(-1, Vec3(0.9, 0.1, 0.1), 0.2, 0.8, Vec3(), True, opacity=0.75, clearcoat=0.8)
        ov_a.material_ref = ref_mat.copy()
        _ = app.world.insert[MaterialOverrideComponent](e_a, ov_a^)

        var e_b = app.world.spawn()
        _ = app.world.insert[LocalTransform](e_b, LocalTransform(Vec3(1.5, 0.0, 0.0)))
        _ = app.world.insert[GlobalTransform](e_b, GlobalTransform())
        _ = app.world.insert[PrevGlobalTransform](e_b, PrevGlobalTransform())
        _ = app.world.insert[MeshRendererComponent](e_b, MeshRendererComponent(0, shared_id))

        rw_ptr.value()[unsafe_offset=0].overlay_lines.add_line(
            Vec3(0.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0), 0xFF00FF00, thickness_px=2.0
        )
        rw_ptr.value()[unsafe_offset=0].clear_frame(1, Float32(1.0 / 60.0), clear_overlay=False)
        ok = ok and rw_ptr.value()[unsafe_offset=0].overlay_lines.segment_count() == 1

        app.step_one_frame(Float32(1.0 / 60.0))
        ok = ok and abs(rend.value()[unsafe_offset=0].materials[shared_id].base_color.x - Float32(0.4)) < Float32(1.0e-5)

    app.shutdown()
    _ = runner.check("t3_mojo_pair1_asset_ref_flmat_override_and_overlay_lines", ok, "pair 1 failed")


# ---------------------------------------------------------------------------
# Pair 2: FRIG .flrig + AnimPlugin + RigComponent + Scene v2 round-trip
# ---------------------------------------------------------------------------
def test_pair2_frig_anim_plugin_rig_and_scene_v2_roundtrip(
    mut runner: StudioSuiteRunner, tmp: String
) raises:
    var rig = FrigRigDef("Pair2Rig")
    rig.joints.append(FrigJointDef("Root", -1, Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    rig.joints.append(FrigJointDef("Spine", 0, Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    var flrig_path = tmp + "/pair2_hero.flrig"
    var saved_rig = save_flrig_file(flrig_path, rig)
    var rig_rt = load_flrig_file(flrig_path)

    var app = EngineApp(AppConfig(title="pair2_frig_scene_v2", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(AnimPlugin())

    var reg = ComponentRegistry.with_builtins()
    register_anim_codecs(reg)
    reg.register[RigComponent]()
    reg.ensure_registered_in_world(app.world)
    app.startup()

    var e = app.world.spawn()
    _ = app.world.insert[LocalTransform](e, LocalTransform(Vec3(1.0, 0.0, 2.0)))
    _ = app.world.insert[GlobalTransform](e, GlobalTransform())
    var rc = RigComponent()
    rc.rig_ref = AssetRef(AssetUuid.from_path("rigs/pair2_hero.flrig"), "rigs/pair2_hero.flrig", ASSET_KIND_RAW)
    _ = app.world.insert[RigComponent](e, rc^)

    var sc = Scene("pair2_scene")
    var json_str = sc.save_flscn_string(app.world, reg)
    var bin_bytes = sc.save_flscnb_bytes(app.world, reg)

    var w_rt = World()
    reg.ensure_registered_in_world(w_rt)
    var sc_rt = Scene()
    var jbytes = List[UInt8](capacity=json_str.byte_length())
    for b in json_str.as_bytes():
        jbytes.append(b)
    var ents_j = sc_rt.load_flscn_bytes(w_rt, reg, jbytes)
    var w_bin = World()
    reg.ensure_registered_in_world(w_bin)
    var ents_b = sc_rt.load_flscnb_bytes(w_bin, reg, bin_bytes)

    var schema_ver = SCENE_SCHEMA_VERSION
    var ok = (
        saved_rig
        and len(rig_rt.joints) == 2
        and schema_ver == 2
        and len(ents_j) == 1
        and len(ents_b) == 1
        and w_rt.has[RigComponent](ents_j[0])
        and w_bin.has[RigComponent](ents_b[0])
        and w_rt.get_ptr[RigComponent](ents_j[0]).value()[].rig_ref.rel_path == "rigs/pair2_hero.flrig"
        and w_bin.get_ptr[RigComponent](ents_b[0]).value()[].rig_ref.rel_path == "rigs/pair2_hero.flrig"
    )
    app.shutdown()
    _ = runner.check(
        "t3_mojo_pair2_frig_anim_plugin_rig_and_scene_v2_roundtrip",
        ok,
        "SCENE_SCHEMA_VERSION=" + String(SCENE_SCHEMA_VERSION),
    )


# ---------------------------------------------------------------------------
# Pair 3: CharacterControllerComponent + TriggerVolume (OnEnter/OnExit + entity_from_body)
#         + BehaviorTreeComponent + NavPlugin/NavMesh + AudioPlugin/AudioServer
# ---------------------------------------------------------------------------
def test_pair3_kcc_triggers_bt_nav_and_audio_plugins(
    mut runner: StudioSuiteRunner, tmp: String
) raises:
    var wav_path = tmp + "/pair3_sfx.wav"
    var wav_wrote = write_minimal_wav(wav_path, 0.15, 48000, 2)
    var wav = decode_wav(read_bytes(wav_path))

    var app = EngineApp(AppConfig(title="pair3_gameplay_plugins", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0.0))
    app.add_plugin(NavPlugin())
    app.add_plugin(AudioPlugin(headless=True))
    app.startup()

    var floor_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](floor_e, LocalTransform(Vec3(0.0, -0.5, 0.0)))
    _ = app.world.insert[GlobalTransform](floor_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](floor_e, PrevGlobalTransform())
    _ = app.world.insert[ColliderComponent](floor_e, ColliderComponent.box(Vec3(10.0, 0.5, 10.0)))
    _ = app.world.insert[RigidBodyComponent](floor_e, RigidBodyComponent.static_body())

    var trig_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](trig_e, LocalTransform(Vec3(2.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](trig_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](trig_e, PrevGlobalTransform())
    _ = app.world.insert[ColliderComponent](trig_e, ColliderComponent.box(Vec3(0.8, 1.5, 0.8), is_trigger=True))

    var kcc_cfg = CharacterControllerConfig()
    kcc_cfg.gravity = 0.0
    var player_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](player_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
    _ = app.world.insert[GlobalTransform](player_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](player_e, PrevGlobalTransform())
    _ = app.world.insert[CharacterControllerComponent](
        player_e, CharacterControllerComponent(Vec3(0.0, 1.0, 0.0), kcc_cfg, Vec3(6.0, 0.0, 0.0))
    )

    var enemy_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](enemy_e, LocalTransform(Vec3(6.0, 0.0, 0.0)))
    _ = app.world.insert[GlobalTransform](enemy_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](enemy_e, PrevGlobalTransform())
    _ = app.world.insert[BehaviorTreeComponent](enemy_e, BehaviorTreeComponent())

    var bb = Blackboard()
    var k_alert = bb.def_key("alert", Float32(0.0))
    var bt = BehaviorTree()
    _ = bt.begin_selector()
    _ = bt.begin_sequence()
    _ = bt.add_blackboard_cmp(k_alert, BT_CMP_GT, Float32(0.5), BT_ABORT_LOWER_PRIORITY)
    _ = bt.add_action(10)
    bt.end_composite()
    _ = bt.add_action(2)
    bt.end_composite()
    var bt_st = BtExecutionState()
    bt_st.set_action_status(2, BT_RUNNING)
    bt_st.set_action_status(10, BT_SUCCESS)
    var bt_r1 = bt.tick(bb, bt_st, Float32(1.0 / 60.0))
    bb.set_float(k_alert, Float32(1.0))
    var bt_r2 = bt.tick(bb, bt_st, Float32(1.0 / 60.0))
    var bt_ok = app.world.has[BehaviorTreeComponent](enemy_e) and bt_r1 == BT_RUNNING and bt_r2 == BT_SUCCESS

    var cursor = 0
    var saw_enter = False
    var saw_exit = False
    var resolved_player = False
    for _ in range(60):
        app.step_one_frame(Float32(1.0 / 60.0))
        var evs = app.resources.get_ptr[TriggerEventsResource]().value()[unsafe_offset=0].events.read_since(cursor)
        for i in range(len(evs)):
            if OnEnter(evs[i], trig_e, app.world, app.resources):
                saw_enter = True
                if entity_from_body(evs[i].body_id, app.resources) == player_e:
                    resolved_player = True
            if OnExit(evs[i], trig_e, app.world, app.resources):
                if entity_from_body(evs[i].body_id, app.resources) == player_e:
                    saw_exit = True
        if saw_enter and resolved_player and saw_exit:
            break

    var nav_ok = False
    var pw_ptr = app.resources.get_ptr[PhysicsWorld]()
    if pw_ptr:
        var nm = NavMesh.build_from_physics(pw_ptr.value()[unsafe_offset=0])
        var path = nm.find_path(Vec3(-4.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0))
        nav_ok = nm.poly_count() > 0 and path.found and len(path.waypoints) >= 2

    var audio_ok = False
    var audio_srv = app.resources.get_ptr_mut[AudioServer]()
    if audio_srv and wav_wrote and wav.valid:
        var aid = audio_srv.value()[unsafe_offset=0].register_wav_asset(wav)
        var sh = audio_srv.value()[unsafe_offset=0].play_2d(aid, volume=0.5, bus_id=BUS_SFX, looping=True)
        var sh_valid = sh.is_valid()
        if not audio_srv.value()[unsafe_offset=0].rt_running:
            var out_lr = List[Float32]()
            audio_srv.value()[unsafe_offset=0].render_offline_quantum(out_lr, 256)
        for _ in range(10):
            app.step_one_frame(Float32(1.0 / 60.0))
            if app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count() >= 1:
                break
            if app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].rt_running:
                _ = external_call["usleep", Int32](UInt32(2000))
        var playing_before = app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count()
        app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].stop_all()
        if not app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].rt_running:
            var out_lr = List[Float32]()
            app.resources.get_ptr_mut[AudioServer]().value()[unsafe_offset=0].render_offline_quantum(out_lr, 256)
        for _ in range(10):
            app.step_one_frame(Float32(1.0 / 60.0))
            if app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count() == 0:
                break
            if app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].rt_running:
                _ = external_call["usleep", Int32](UInt32(2000))
        var playing_after = app.resources.get_ptr[AudioServer]().value()[unsafe_offset=0].active_logical_count()
        audio_ok = sh_valid and playing_before >= 1 and playing_after == 0

    var ok = saw_enter and saw_exit and resolved_player and bt_ok and nav_ok and audio_ok
    app.shutdown()
    _ = runner.check(
        "t3_mojo_pair3_kcc_triggers_bt_nav_and_audio_plugins",
        ok,
        "enter=" + String(saw_enter) + " exit=" + String(saw_exit) + " player=" + String(resolved_player) + " bt=" + String(bt_ok) + " nav=" + String(nav_ok) + " audio=" + String(audio_ok),
    )


# ---------------------------------------------------------------------------
# Pair 4: FlintPluginApiV2 (PodComponentSchema) + Scene v2 unknown-component
#         opaque blob preservation + GameFlow save slots
# ---------------------------------------------------------------------------
def test_pair4_plugin_api_v2_scene_unknown_blob_and_gameflow(
    mut runner: StudioSuiteRunner, tmp: String
) raises:
    var api_ver = FLINT_PLUGIN_API_VERSION
    var api_ok = api_ver == 2
    var schema = PodComponentSchema("GameplayHealth", 16)
    var api_v2 = FlintPluginApiV2()

    var unknown_scene = (
        '{"format": "flscn", "version": 1, "name": "opaque_rt", "entities": ['
        + '{"slot": 0, "parent": -1, "components": {'
        + '"LocalTransform": {"position": [1.0, 2.0, 3.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]}, '
        + '"CustomGameHealth": {"hp": 85, "team": "alpha"}'
        + '}}]}\n'
    )
    var in_bytes = List[UInt8](capacity=unknown_scene.byte_length())
    for b in unknown_scene.as_bytes():
        in_bytes.append(b)

    var reg = ComponentRegistry.with_builtins()
    var w1 = World()
    var sc = Scene()
    var ents1 = sc.load_flscn_bytes(w1, reg, in_bytes)
    var saved = sc.save_flscn_string(w1, reg)

    var w2 = World()
    var saved_bytes = List[UInt8](capacity=saved.byte_length())
    for b in saved.as_bytes():
        saved_bytes.append(b)
    var ents2 = sc.load_flscn_bytes(w2, reg, saved_bytes)

    var flow = GameFlow(tmp + "/saves")
    var slot_ok = flow.save_slot(0, "checkpoint_1") and flow.load_slot(0) == "checkpoint_1"

    var ok = (
        api_ok
        and schema.byte_size == 16
        and api_v2.version == 2
        and len(ents1) == 1
        and len(ents2) == 1
        and saved.find("CustomGameHealth") >= 0
        and saved.find("85") >= 0
        and slot_ok
    )
    _ = runner.check("t3_mojo_pair4_plugin_api_v2_scene_unknown_blob_and_gameflow", ok, "PluginApiV2/SceneV2/GameFlow failed")


# ---------------------------------------------------------------------------
# Pair 5: Edit vs PIE STAGE_UPDATE + CameraComponent isolation + Events[T] swaps
# ---------------------------------------------------------------------------
def test_pair5_edit_vs_pie_camera_isolation_and_events_swap(
    mut runner: StudioSuiteRunner
) raises:
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=160,
        height=120,
        worker_threads=0,
    )
    studio.look_at(Vec3(0.0, 2.0, -5.0), Vec3(0.0, 1.0, 0.0))

    var cam_e = studio.app.world.spawn()
    _ = studio.app.world.insert[LocalTransform](cam_e, LocalTransform(Vec3(30.0, 12.0, 40.0)))
    _ = studio.app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = studio.app.world.insert[PrevGlobalTransform](cam_e, PrevGlobalTransform())
    _ = studio.app.world.insert[CameraComponent](cam_e, CameraComponent(fov_y_deg=60.0, active=True))

    studio.app.insert_resource[PairProbeResource](PairProbeResource())
    _ = studio.app.add_system(STAGE_PRE_UPDATE, "t3_probe_pre", probe_pre_sys)
    _ = studio.app.add_system(STAGE_UPDATE, "t3_probe_upd", probe_update_sys)
    _ = studio.app.add_system(STAGE_POST_UPDATE, "t3_probe_post", probe_post_sys)
    studio.app.add_event_swap_hook(probe_swap_hook)
    studio.app.schedule.compile_all()

    # Edit mode: STAGE_UPDATE must NOT run; editor_camera_override keeps editor camera
    studio.render_editor_frame(160, 120, dt=Float32(1.0 / 60.0))
    var probe = studio.app.resources.get_ptr[PairProbeResource]()
    var rps = studio.app.resources.get_ptr[RenderPluginStateResource]()
    var rw = studio.app.resources.get_ptr[RenderWorld]()
    var edit_ok = (
        Bool(probe)
        and Bool(rps)
        and Bool(rw)
        and probe.value()[unsafe_offset=0].update_count == 0
        and probe.value()[unsafe_offset=0].pre_count == 1
        and probe.value()[unsafe_offset=0].post_count == 1
        and rps.value()[unsafe_offset=0].editor_camera_override
        and abs(rw.value()[unsafe_offset=0].camera.eye.x - Float32(0.0)) < Float32(1.0e-3)
    )

    # Enter PIE playing: editor_camera_override becomes False, STAGE_UPDATE + Events[T] swap run
    var pie_started = studio.pie_play()
    var steps = studio.step_pie_frame(Float32(1.0 / 60.0))
    studio.render_editor_frame(160, 120, dt=Float32(1.0 / 60.0))
    var ev_cursor = 0
    var evs = studio.app.resources.get_ptr_mut[PairProbeResource]().value()[unsafe_offset=0].events.read_since(ev_cursor)
    var pie_ok = (
        pie_started
        and studio.pie.mode == PIE_PLAYING
        and steps == 1
        and probe.value()[unsafe_offset=0].update_count == 1
        and not rps.value()[unsafe_offset=0].editor_camera_override
        and abs(rw.value()[unsafe_offset=0].camera.eye.x - Float32(30.0)) < Float32(1.0e-3)
        and len(evs) == 1
        and evs[0] == 1
    )

    # Stop PIE: editor_camera_override restored to True, editor camera restored
    var pie_stopped = studio.pie_stop()
    studio.render_editor_frame(160, 120, dt=Float32(1.0 / 60.0))
    var stop_ok = (
        pie_stopped
        and studio.pie.mode == PIE_STOPPED
        and rps.value()[unsafe_offset=0].editor_camera_override
        and abs(rw.value()[unsafe_offset=0].camera.eye.x - Float32(0.0)) < Float32(1.0e-3)
    )

    studio.shutdown()
    _ = runner.check(
        "t3_mojo_pair5_edit_vs_pie_camera_isolation_and_events_swap",
        edit_ok and pie_ok and stop_ok,
        "edit_ok=" + String(edit_ok) + " pie_ok=" + String(pie_ok) + " stop_ok=" + String(stop_ok),
    )


def main() raises:
    var tmp = make_tmp_dir("t3_content_runtime")
    var runner = StudioSuiteRunner("t3_cross_feature_content_runtime")

    try:
        test_pair1_asset_ref_flmat_override_and_overlay_lines(runner, tmp)
        test_pair2_frig_anim_plugin_rig_and_scene_v2_roundtrip(runner, tmp)
        test_pair3_kcc_triggers_bt_nav_and_audio_plugins(runner, tmp)
        test_pair4_plugin_api_v2_scene_unknown_blob_and_gameflow(runner, tmp)
        test_pair5_edit_vs_pie_camera_isolation_and_events_swap(runner)
    finally:
        rm_rf_tmp(tmp)
    runner.finish()
