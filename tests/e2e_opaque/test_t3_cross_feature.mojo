"""Tier 3 Opaque-Box E2E Suite: Pairwise Cross-Feature Integration Tests (10 pairs).

Uses ONLY public Flint Engine v5 APIs (zero private `_` symbols from `flint.*`).
All disk outputs live under an isolated `/tmp/flint_e2e_m3_t3_<pid>` directory cleaned on exit.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, sin, sqrt
from std.testing import assert_equal, assert_false, assert_true

from flint.app import AppConfig, EngineApp
from flint.app.engine_app import WorldTickSyncResource
from flint.anim.flanim_ecs import (
    FLANIM_FOURCC,
    RAGDOLL_SM_ACTIVE_PD,
    RAGDOLL_SM_ANIMATED,
    RAGDOLL_SM_GETUP_PRONE,
    RAGDOLL_SM_GETUP_SUPINE,
    BoneSocketAttachment,
    RagdollStateMachine,
    load_flanim_bytes,
    save_flanim_bytes,
    solve_foot_plant_world,
)
from flint.anim.graph import (
    OP_GT,
    AnimGraphDef,
    BlendSampleDef,
    BlendSpaceDef,
    Condition,
    StateDef,
)
from flint.anim.pose import Pose, SkeletonInstance
from flint.anim.procedural import FootPlantSolver, Placement
from flint.assets.cooker_v2 import cook_workspace
from flint.assets.flpk_v3 import FlpkV3MappedArchive
from flint.assets.glb_native import parse_native_glb
from flint.assets.hdr_image import (
    decode_hdr_or_exr,
    encode_exr_fp16,
    encode_radiance_hdr,
)
from flint.assets.jpeg import decode_jpeg, encode_jpeg_baseline
from flint.assets.json import JsonDocument, parse_json
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import (
    NativeAnimation,
    NativeChannel,
    NativeModel,
    NativeNode,
)
from flint.assets.server import AssetServer
from flint.audio import AudioServer, BUS_SFX, decode_qoa, encode_qoa
from flint.core.ik import angle_axis
from flint.core.jobs import JobPool
from flint.core.lz4 import hash_list
from flint.core.math import Mat4, Quat, Vec3
from flint.devtools.mcp_v2 import McpServerV2
from flint.ecs import (
    STORAGE_TABLE,
    Component,
    Entity,
    World,
    comp_hash64,
)
from flint.input import ACTION_AXIS2D, ACTION_BOOL, InputContext, InputMap
from flint.lab import EngineBenchSuite
from flint.net import (
    IPV6_SAFE_MTU_BYTES,
    JITTER_TRIANGULAR,
    NetEndpoint,
    NetHub,
    RollbackSession,
    VirtualEndpoint,
    VirtualNetSim,
)
from flint.net.rollback import PhysicsRollbackAdapter, PhysicsRollbackInput
from flint.physics.character import CharacterController, CharacterControllerConfig
from flint.physics.world import CAT_ALL, CollisionFilter, PhysicsWorld
from flint.platform.input import InputState
from flint.platform.os import file_size, make_dirs, read_bytes, write_bytes
from flint.render.flip import FLIP_PPD, flip_error_map, flip_mean
from flint.render.font import SdfFont
from flint.render.image_diff import compare_pixels
from flint.render.material_graph import MaterialAsset
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RenderCameraPacket,
    RenderExtractor,
    RenderWorld,
    ScalableRenderer,
)
from flint.scene import (
    ComponentRegistry,
    GlobalTransform,
    LocalTransform,
    Parent,
    PrevGlobalTransform,
    Scene,
    TransformPlugin,
    set_parent,
)
from flint.studio import FlintStudioState, bake_procedural_to_glb
from flint.text.sdf_bake import bake_font, dsf_bytes
from flint.text.ttf import TrueTypeFont
from flint.ui.context import UiContext
from flint.ui.flexbox_dock import DockSpace, RectF


comptime TAU64: Float64 = 6.283185307179586


def _rm_rf_tmp(path: String):
    if path.startswith("/tmp/flint_e2e_m3_"):
        var cmd = "rm -rf " + path
        _ = external_call["system", Int32](cmd.as_c_string_span())


def _str_to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def _parse_json_str(s: String) -> JsonDocument:
    var raw = _str_to_bytes(s)
    return parse_json(raw)


@fieldwise_init
struct T3BodyLink(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e_opaque.t3.T3BodyLink")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var body_id: Int


@fieldwise_init
struct T3SyncedPos(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e_opaque.t3.T3SyncedPos")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var pos: Vec3


def _make_procedural_biped() -> NativeModel:
    var m = NativeModel()
    var names = [
        String("Pelvis"), String("Chest"), String("Head"),
        String("UpperArm.R"), String("Hand.R"),
        String("Thigh.L"), String("Shin.L"), String("Foot.L"),
        String("Thigh.R"), String("Shin.R"), String("Foot.R"),
    ]
    var parents = [-1, 0, 1, 1, 3, 0, 5, 6, 0, 8, 9]
    var offsets = [
        Vec3(0.0, 0.92, 0.0), Vec3(0.0, 0.42, 0.0), Vec3(0.0, 0.32, 0.0),
        Vec3(0.26, 0.24, 0.0), Vec3(0.0, -0.48, 0.0),
        Vec3(-0.14, 0.0, 0.0), Vec3(0.0, -0.44, 0.0), Vec3(0.0, -0.40, 0.0),
        Vec3(0.14, 0.0, 0.0), Vec3(0.0, -0.44, 0.0), Vec3(0.0, -0.40, 0.0),
    ]
    for i in range(len(names)):
        var nd = NativeNode()
        nd.name = names[i]
        nd.translation = offsets[i]
        nd.base_translation = offsets[i]
        nd.rotation = Quat()
        nd.base_rotation = Quat()
        nd.scale = Vec3(1.0, 1.0, 1.0)
        nd.base_scale = Vec3(1.0, 1.0, 1.0)
        m.nodes.append(nd^)
        var p = parents[i]
        if p >= 0:
            m.nodes[p].children.append(i)
        else:
            m.roots.append(i)

    for cname in [String("GetupSupine"), String("GetupProne")]:
        var anim = NativeAnimation()
        anim.name = cname
        anim.duration = 0.40
        var ch = NativeChannel()
        ch.node = 0
        ch.path = 0
        ch.times = [Float32(0.0), Float32(0.40)]
        ch.values = [Float32(0.0), Float32(0.92), Float32(0.0), Float32(0.0), Float32(0.92), Float32(0.0)]
        anim.channels.append(ch^)
        m.animations.append(anim^)
    return m^


# ===========================================================================
# Pair 1: EngineApp + Scene + TransformPlugin
# ===========================================================================

def test_t3_p1_engine_app_scene_transform_plugin() raises:
    var app = EngineApp(AppConfig(title="T3_Pair1", headless=True, worker_threads=2))
    app.add_plugin(TransformPlugin())

    var reg = ComponentRegistry.with_builtins()
    var root_e = app.world.spawn_empty()
    var child_e = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](
        root_e,
        LocalTransform(Vec3(10.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)),
    )
    _ = app.world.insert[GlobalTransform](root_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](root_e, PrevGlobalTransform())

    _ = app.world.insert[LocalTransform](
        child_e,
        LocalTransform(Vec3(0.0, 5.0, 0.0), Quat(), Vec3(2.0, 2.0, 2.0)),
    )
    _ = app.world.insert[GlobalTransform](child_e, GlobalTransform())
    _ = app.world.insert[PrevGlobalTransform](child_e, PrevGlobalTransform())
    assert_true(set_parent(app.world, child_e, root_e))

    app.startup()
    app.step_one_frame(Float32(1.0 / 60.0))

    var gt_child = app.world.get[GlobalTransform](child_e)
    assert_true(abs(gt_child.translation.x - Float32(10.0)) < Float32(1e-4))
    assert_true(abs(gt_child.translation.y - Float32(5.0)) < Float32(1e-4))
    assert_true(abs(gt_child.scale.x - Float32(2.0)) < Float32(1e-4))

    var lt_root_ptr = app.world.get_ptr_mut[LocalTransform](root_e)
    assert_true(Bool(lt_root_ptr))
    lt_root_ptr.value()[unsafe_offset=0].translation = Vec3(20.0, 1.0, -3.0)
    app.step_one_frame(Float32(1.0 / 60.0))

    var gt_child2 = app.world.get[GlobalTransform](child_e)
    assert_true(abs(gt_child2.translation.x - Float32(20.0)) < Float32(1e-4))
    assert_true(abs(gt_child2.translation.y - Float32(6.0)) < Float32(1e-4))
    assert_true(abs(gt_child2.translation.z - Float32(-3.0)) < Float32(1e-4))

    assert_true(app.resources.has[WorldTickSyncResource]())
    var prev_child2_ptr = app.world.get_ptr[PrevGlobalTransform](child_e)
    assert_true(Bool(prev_child2_ptr))
    ref prev_child2 = prev_child2_ptr.value()[unsafe_offset=0]
    assert_true(abs(prev_child2.value.translation.x - Float32(10.0)) < Float32(1e-4))
    assert_true(abs(prev_child2.value.translation.y - Float32(5.0)) < Float32(1e-4))

    var scene = Scene("T3Hierarchy")
    var flscn_text = scene.save_flscn_string(app.world, reg)
    assert_true(flscn_text.find("T3Hierarchy") >= 0)

    var app2 = EngineApp(AppConfig(title="T3_Pair1_Reload", headless=True, worker_threads=2))
    app2.add_plugin(TransformPlugin())
    var scene2 = Scene()
    var loaded = scene2.load_flscn_bytes(app2.world, reg, _str_to_bytes(flscn_text))
    assert_equal(len(loaded), 2)
    app2.startup()
    app2.step_one_frame(Float32(1.0 / 60.0))

    var found_child = False
    for i in range(len(loaded)):
        if app2.world.has[Parent](loaded[i]):
            var g = app2.world.get[GlobalTransform](loaded[i])
            assert_true(abs(g.translation.x - Float32(20.0)) < Float32(1e-4))
            assert_true(abs(g.translation.y - Float32(6.0)) < Float32(1e-4))
            found_child = True
    assert_true(found_child)

    app2.shutdown()
    app.shutdown()
    print("[PASS] test_t3_p1_engine_app_scene_transform_plugin")


# ===========================================================================
# Pair 2: World + PhysicsWorld
# ===========================================================================

def test_t3_p2_world_physics_world() raises:
    var world = World()
    _ = world.register_component[T3BodyLink]()
    _ = world.register_component[T3SyncedPos]()

    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    var sphere_slot = pw.add_sphere(Vec3(0.0, 5.0, 0.0), 0.5, mass=1.0)
    var sphere_id = pw.body_id(sphere_slot)
    pw.rebuild_broadphase()

    var e = world.spawn_empty()
    assert_true(world.insert[T3BodyLink](e, T3BodyLink(sphere_id)))
    assert_true(world.insert[T3SyncedPos](e, T3SyncedPos(Vec3(0.0, 5.0, 0.0))))

    var dt = Float32(1.0 / 60.0)
    for _ in range(30):
        pw.step(dt)
        var link = world.get[T3BodyLink](e)
        var slot = pw.find_slot_by_id(link.body_id)
        assert_true(slot >= 0)
        var sp_ptr = world.get_ptr_mut[T3SyncedPos](e)
        assert_true(Bool(sp_ptr))
        sp_ptr.value()[unsafe_offset=0].pos = pw.bodies[slot].pos

    var synced = world.get[T3SyncedPos](e)
    assert_true(synced.pos.y < Float32(4.2))
    var hit = pw.raycast(
        Vec3(synced.pos.x, synced.pos.y + 5.0, synced.pos.z),
        Vec3(0.0, -1.0, 0.0),
        Float32(20.0),
    )
    assert_true(hit.hit)
    assert_equal(hit.body_id, sphere_id)
    print("[PASS] test_t3_p2_world_physics_world")


# ===========================================================================
# Pair 3: PhysicsWorld + CharacterController + InputMap
# ===========================================================================

def test_t3_p3_physics_kcc_input_map() raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    _ = pw.add_box(Vec3(2.0, 1.0, 0.0), Vec3(0.5, 1.5, 5.0), mass=0.0)
    pw.rebuild_broadphase()

    var kcc = CharacterController(Vec3(0.0, 1.05, 0.0), CharacterControllerConfig())
    var inp = InputState()
    var imap = InputMap()
    _ = imap.register_action("Move", ACTION_AXIS2D)
    _ = imap.register_action("Jump", ACTION_BOOL)

    var ctx = InputContext("Gameplay", priority=0)
    ctx.bind_key("Move", 68, scale_x=1.0, scale_y=0.0)   # D -> +X
    ctx.bind_key("Move", 65, scale_x=-1.0, scale_y=0.0)  # A -> -X
    ctx.bind_key("Move", 87, scale_x=0.0, scale_y=1.0)   # W -> +Y
    ctx.bind_key("Move", 83, scale_x=0.0, scale_y=-1.0)  # S -> -Y
    ctx.bind_key("Jump", 32)                             # Space
    imap.add_context(ctx^)
    imap.push_context("Gameplay")

    var dt = Float32(1.0 / 60.0)
    # Settle KCC onto ground
    for _ in range(15):
        inp.begin_frame()
        imap.update(inp, dt)
        _ = kcc.move_and_slide(pw, Vec3(), dt)
        imap.set_grounded(kcc.grounded)
    assert_true(kcc.grounded)
    assert_true(imap.coyote_valid())

    var start_x = kcc.pos.x
    inp.set_logical_key(68, True)  # Hold 'D'
    var hit_wall_ever = False
    for _ in range(45):
        inp.begin_frame()
        imap.update(inp, dt)
        var mv = imap.axis_2d("Move")
        assert_true(mv[0] > Float32(0.5))
        _ = kcc.move_and_slide(pw, Vec3(mv[0] * Float32(4.0), 0.0, mv[1] * Float32(4.0)), dt)
        if kcc.hit_wall:
            hit_wall_ever = True
    inp.set_logical_key(68, False)

    assert_true(kcc.pos.x > start_x + Float32(0.5))
    assert_true(hit_wall_ever)
    assert_true(kcc.pos.x < Float32(1.5))

    var pre_jump_y = kcc.pos.y
    inp.begin_frame()
    inp.set_logical_key(32, True)
    imap.update(inp, dt)
    assert_true(imap.just_pressed("Jump"))
    _ = kcc.move_and_slide(pw, Vec3(), dt, jump_speed=6.0)
    inp.set_logical_key(32, False)
    assert_true(kcc.pos.y > pre_jump_y)

    inp.close()
    print("[PASS] test_t3_p3_physics_kcc_input_map")


# ===========================================================================
# Pair 4: ArticulatedRagdoll + flanim_ecs + PhysicsWorld
# ===========================================================================

def test_t3_p4_ragdoll_flanim_physics_world() raises:
    var g = AnimGraphDef()
    g.name = "T3RagdollGraph"
    var bs = BlendSpaceDef()
    bs.name = "Move"
    bs.param = 0
    bs.samples.append(BlendSampleDef(0, 0.0))
    bs.samples.append(BlendSampleDef(1, 4.0))
    g.blend_spaces.append(bs^)
    var st = StateDef()
    st.name = "Locomotion"
    st.blend = 0
    st.when.append(Condition(0, OP_GT, 0.1))
    g.states.append(st^)
    var fl_bytes = save_flanim_bytes(g)
    var fl_tag = UInt32(fl_bytes[0]) | (UInt32(fl_bytes[1]) << 8) | (UInt32(fl_bytes[2]) << 16) | (UInt32(fl_bytes[3]) << 24)
    assert_equal(fl_tag, FLANIM_FOURCC)
    var g2 = load_flanim_bytes(fl_bytes)
    assert_equal(g2.name, "T3RagdollGraph")

    var dt = Float32(1.0 / 60.0)
    var pw = PhysicsWorld()
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(30.0, 0.5, 30.0), mass=0.0)
    _ = pw.add_box(Vec3(0.14, 0.125, 0.0), Vec3(0.25, 0.125, 0.40), mass=0.0)
    _ = pw.add_box(Vec3(1.1, 0.25, -0.8), Vec3(0.4, 0.25, 0.4), mass=0.0)
    var shield_slot = pw.add_kinematic_box(Vec3(), Vec3(0.2, 0.3, 0.05))
    var shield_id = pw.body_id(shield_slot)
    pw.rebuild_broadphase()

    var biped = _make_procedural_biped()
    var skel = SkeletonInstance()
    skel.reset_to_rest(biped)
    var place = Placement(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), 0.0)

    var sock = BoneSocketAttachment("Shield", 4, 50, Vec3(0.05, 0.0, 0.1), Quat(), Vec3(1.0, 1.0, 1.0), shield_id)
    sock.sync_to_physics(skel, place, pw, dt)
    skel.pose.r[3] = angle_axis(0.8, Vec3(0.0, 1.0, 0.0))
    skel.update_matrices(biped)
    sock.sync_to_physics(skel, place, pw, dt)
    assert_true(pw.bodies[shield_slot].linear_vel.length() > Float32(1.0))

    var fp = FootPlantSolver(0)
    fp.add_leg(5, 6, 7, 0.04)
    fp.add_leg(8, 9, 10, 0.04)
    var stance: List[Bool] = [True, True]
    var ray_hits = 0
    for _ in range(15):
        skel.reset_to_rest(biped)
        ray_hits = solve_foot_plant_world(fp, pw, skel, biped, place, stance, dt)
    assert_equal(ray_hits, 2)
    assert_true(fp.foot_world_position(1, skel, place).y > fp.foot_world_position(0, skel, place).y + Float32(0.18))

    var rsm = RagdollStateMachine()
    rsm.supine_getup_clip = 0
    rsm.prone_getup_clip = 1
    _ = rsm.ragdoll.add_limb_from_model(biped, 0, -1, 15.0, 0.15, 0.35)
    _ = rsm.ragdoll.add_limb_from_model(biped, 1, 0, 12.0, 0.15, 0.35)
    _ = rsm.ragdoll.add_limb_from_model(biped, 5, 0, 6.0, 0.10, 0.40, is_hinge=True)
    _ = rsm.ragdoll.add_limb_from_model(biped, 8, 0, 6.0, 0.10, 0.40, is_hinge=True)

    var rest_pose = Pose()
    rest_pose.rest(biped)
    skel.reset_to_rest(biped)

    rsm.apply_hit_impulse(skel, place, biped, 1, Vec3(15.0, 0.0, 0.0), Vec3(0.0, 1.3, 0.0), go_limp_threshold=100.0, dt=dt)
    assert_equal(rsm.state, RAGDOLL_SM_ACTIVE_PD)
    rsm.update(pw, skel, place, biped, rest_pose, dt)

    rsm.apply_hit_impulse(skel, place, biped, 1, Vec3(0.0, 15.0, -160.0), Vec3(0.0, 1.35, 0.0), go_limp_threshold=100.0, dt=dt)
    var saw_getup = False
    for _ in range(180):
        rsm.update(pw, skel, place, biped, rest_pose, dt)
        if rsm.state == RAGDOLL_SM_GETUP_SUPINE or rsm.state == RAGDOLL_SM_GETUP_PRONE:
            saw_getup = True
        if saw_getup and rsm.state == RAGDOLL_SM_ANIMATED:
            break
    assert_true(saw_getup)
    assert_equal(rsm.state, RAGDOLL_SM_ANIMATED)
    assert_true(rsm.last_pelvis_snap_m < Float32(1.0e-4))
    print("[PASS] test_t3_p4_ragdoll_flanim_physics_world")


# ===========================================================================
# Pair 5: CookerV2/.flpk + AssetServer + glb_native/jpeg/hdr_image
# ===========================================================================

def test_t3_p5_cooker_flpk_asset_server_codecs(tmp_root: String) raises:
    var p5_dir = tmp_root + "/pair5"
    var ws = p5_dir + "/ws"
    var cache_dir = p5_dir + "/cache"
    var pack_path = p5_dir + "/bundle.flpk"
    assert_true(make_dirs(ws + "/meshes"))
    assert_true(make_dirs(ws + "/textures"))
    assert_true(make_dirs(ws + "/env"))

    # 1. Procedural GLB + direct parse_native_glb verification
    var glb_path = ws + "/meshes/orb.glb"
    var baked = bake_procedural_to_glb("material_orb", glb_path, seed=7)
    assert_true(baked[0])
    var parsed_glb = parse_native_glb(read_bytes(glb_path))
    assert_true(len(parsed_glb.meshes) > 0)
    assert_true(len(parsed_glb.nodes) > 0)

    # 2. Baseline JPEG + direct decode_jpeg verification
    var jpg_px = List[UInt32](length=32 * 32, fill=0)
    for y in range(32):
        for x in range(32):
            jpg_px[y * 32 + x] = UInt32(0xFF000000) | (UInt32(x * 8) << 16) | (UInt32(y * 8) << 8) | UInt32(128)
    var jpg_bytes = encode_jpeg_baseline(jpg_px, 32, 32, quality=92)
    assert_true(len(jpg_bytes) > 32)
    var dec_jpg = decode_jpeg(jpg_bytes, 1024)
    assert_true(dec_jpg.valid)
    assert_equal(dec_jpg.width, 32)
    assert_equal(dec_jpg.height, 32)
    assert_true(write_bytes(ws + "/textures/brick.jpg", jpg_bytes))

    # 3. Radiance .hdr + OpenEXR .exr codec verification
    var hdr_px = List[Vec3](length=16 * 8, fill=Vec3(2.5, 1.2, 0.5))
    var hdr_bytes = encode_radiance_hdr(hdr_px, 16, 8)
    var exr_bytes = encode_exr_fp16(hdr_px, 16, 8)
    var dec_hdr = decode_hdr_or_exr(hdr_bytes)
    var dec_exr = decode_hdr_or_exr(exr_bytes)
    assert_true(dec_hdr.valid and dec_hdr.width == 16 and dec_hdr.height == 8)
    assert_true(dec_exr.valid and dec_exr.width == 16 and dec_exr.height == 8)
    assert_true(write_bytes(ws + "/env/sky.hdr", hdr_bytes))
    assert_true(write_bytes(ws + "/env/sky_exr.exr", exr_bytes))

    var pool = JobPool()
    _ = pool.start(2)
    var assets = AssetServer(enable_watcher=False)
    try:
        var rep = cook_workspace(ws, pack_path, pool, cache_dir=cache_dir)
        assert_true(rep.ok)
        assert_equal(rep.total_assets, 4)

        var arch = FlpkV3MappedArchive.open(pack_path)
        assert_true(arch.valid)
        assert_true(arch.find_by_path("meshes/orb.glb", 0) >= 0)
        assert_true(arch.find_by_path("textures/brick.jpg", 0) >= 0)
        assert_true(arch.find_by_path("env/sky.hdr", 0) >= 0)
        assert_true(arch.find_by_path("env/sky_exr.exr", 0) >= 0)
        arch.close()

        assert_true(assets.mount_flpk(pack_path))
        var h_mesh = assets.load_mesh("meshes/orb.glb")
        var h_tex = assets.load_texture("textures/brick.jpg", progressive=False)
        var h_hdr = assets.load_hdr("env/sky.hdr")
        var h_exr = assets.load_hdr("env/sky_exr.exr")
        assert_true(assets.meshes.is_loaded(h_mesh))
        assert_true(assets.textures.is_loaded(h_tex))
        assert_true(assets.hdrs.is_loaded(h_hdr))
        assert_true(assets.hdrs.is_loaded(h_exr))
        assert_true(assets.total_resident_bytes() > 0)
    finally:
        assets.close()
        pool.stop()

    print("[PASS] test_t3_p5_cooker_flpk_asset_server_codecs")


# ===========================================================================
# Pair 6: ScalableRenderer + UiContext + DockSpace + sdf_bake
# ===========================================================================

def test_t3_p6_scalable_renderer_ui_dockspace_sdf_font() raises:
    comptime FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
    var ttf = TrueTypeFont()
    assert_true(ttf.load(FONT_PATH))

    var codes = List[Int]()
    for c in range(32, 127):
        codes.append(c)
    var baked = bake_font(ttf, codes, 36, 5, 512)
    var dsf = dsf_bytes(baked)
    var sdf_font = SdfFont()
    assert_true(sdf_font.load_bytes(dsf))
    assert_true(sdf_font.valid)
    assert_true(sdf_font.measure("STUDIO", 18.0) > Float32(10.0))

    var ui = UiContext()
    ui.font = sdf_font.copy()
    var dock = DockSpace.create_studio_5panel()
    var inp = InputState()
    var ren = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, width=320, height=180)

    try:
        var mat_id = ren.register_material(MaterialAsset(base_color=Vec3(0.9, 0.35, 0.15)))
        var mb = MeshBuilder()
        mb.add_beveled_box(Vec3(0.0, 0.5, 0.0), Vec3(0.6, 0.6, 0.6))
        var mesh_id = ren.register_mesh(mb^.finish(), mat_id)

        var rw = RenderWorld()
        rw.camera = RenderCameraPacket(eye=Vec3(0.0, 1.2, 3.0), target=Vec3(0.0, 0.5, 0.0))
        var ext = RenderExtractor()
        ext.begin_extraction(rw, 1)
        ext.extract_entity(rw, UInt64(77), mesh_id, Mat4(), mat_id)
        ext.end_extraction()

        inp.begin_frame()
        ui.begin_frame(inp, 320.0, 180.0, Float32(1.0 / 60.0))
        ui.update_dock_space(dock, RectF(0.0, 0.0, 320.0, 180.0))
        if ui.begin_dock_panel(dock, "Outliner"):
            _ = ui.button("btn_add", "Add Entity")
            ui.end_dock_panel()
        ui.end_frame()

        assert_true(ui.draw_list.count() > 0)
        assert_true(ren.render_world(rw, ui.draw_list, capture_readback=True, force_readback=True))
        var px = ren.readback_pixels()
        assert_equal(len(px), 320 * 180)
        var first_c = px[0]
        var has_diff_px = False
        for i in range(len(px)):
            if px[i] != first_c:
                has_diff_px = True
                break
        assert_true(has_diff_px)
    finally:
        inp.close()
        ren.shutdown()

    print("[PASS] test_t3_p6_scalable_renderer_ui_dockspace_sdf_font")


# ===========================================================================
# Pair 7: AudioServer + Qoa + 3D Emitter Movement
# ===========================================================================

def _measure_rms(buf: List[Float32]) -> Float64:
    if len(buf) == 0:
        return 0.0
    var acc = Float64(0.0)
    for i in range(len(buf)):
        var v = Float64(buf[i])
        acc += v * v
    return sqrt(acc / Float64(len(buf)))


def test_t3_p7_audio_server_qoa_3d_emitter() raises:
    comptime SR: Int = 48000
    comptime FRAMES: Int = 4800
    var pcm = List[Int16](capacity=FRAMES)
    for i in range(FRAMES):
        var s = 0.85 * sin(TAU64 * 440.0 * Float64(i) / Float64(SR))
        pcm.append(Int16(Int(s * 32767.0)))

    var qoa_bytes = encode_qoa(pcm, 1, SR)
    assert_true(len(qoa_bytes) > 16)
    var qoa_dec = decode_qoa(qoa_bytes)
    assert_true(qoa_dec.valid)

    var audio = AudioServer(sample_rate=SR, quantum_frames=256)
    try:
        var aid = audio.register_qoa_asset(qoa_bytes^)
        audio.set_listener(
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(1.0, 0.0, 0.0),
        )
        var h = audio.play_3d(
            aid,
            Vec3(2.0, 0.0, 0.0),
            looping=True,
            bus_id=BUS_SFX,
            ref_dist=5.0,
            max_dist=100.0,
        )
        assert_true(h.is_valid())

        var buf = List[Float32]()
        var near_rms = Float64(0.0)
        for _ in range(8):
            audio.render_offline_quantum(buf, 256)
            near_rms += _measure_rms(buf)
        near_rms /= 8.0
        assert_true(near_rms > 0.01)
        assert_true(audio.momentary_lufs() > Float32(-60.0))

        assert_true(audio.update_emitter(h, Vec3(80.0, 0.0, 0.0)))
        var far_rms = Float64(0.0)
        for q in range(12):
            audio.render_offline_quantum(buf, 256)
            if q >= 4:
                far_rms += _measure_rms(buf)
        far_rms /= 8.0
        assert_true(far_rms < near_rms * 0.5)
        assert_true(audio.stop_sound(h))
    finally:
        audio.stop()

    print("[PASS] test_t3_p7_audio_server_qoa_3d_emitter")


# ===========================================================================
# Pair 8: VirtualNetSim + RollbackSession + NetHub + PhysicsWorld
# ===========================================================================

def _make_phys_input(peer: Int, tick: Int) -> PhysicsRollbackInput:
    var phase = Float32((tick + peer * 7) % 20)
    var ix = Float32(6.0) if (phase < Float32(10.0)) else Float32(-6.0)
    var iz = Float32(4.0) if (peer == 0) else Float32(-4.0)
    return PhysicsRollbackInput(ix, 0.0, iz, 0.0, 1 if (tick % 15 == 0) else 0)


def test_t3_p8_virtual_net_sim_rollback_nethub_physics() raises:
    var sim = VirtualNetSim.from_rtt(
        rtt_ms=40.0,
        jitter_ms=10.0,
        loss_rate=0.05,
        dup_rate=0.01,
        mtu_bytes=IPV6_SAFE_MTU_BYTES,
        seed=0x9988776655443322,
        jitter_mode=JITTER_TRIANGULAR,
    )
    var ep0 = sim.create_ipv4_peer(0, 9700)
    var ep1 = sim.create_ipv4_peer(1, 9700)

    var hub_host = NetHub[VirtualEndpoint](sim.create_ipv4_peer(2, 9800), True)
    var hub_cli = NetHub[VirtualEndpoint](sim.create_ipv4_peer(3, 9800), False)
    var cli_cid = hub_cli.connect(hub_host.local_endpoint())
    var msg_out = List[UInt8](capacity=64)
    for i in range(64):
        msg_out.append(UInt8((i * 13 + 5) & 0xFF))
    assert_true(hub_cli.send_reliable(cli_cid, msg_out))

    var peer_addrs = List[NetEndpoint]()
    peer_addrs.append(ep0.local_endpoint())
    peer_addrs.append(ep1.local_endpoint())

    var pw0 = PhysicsRollbackAdapter.create_arena(num_peers=2)
    var pw1 = PhysicsRollbackAdapter.create_arena(num_peers=2)

    var s0 = RollbackSession[PhysicsRollbackInput](
        local_peer_id=0,
        peer_endpoints=peer_addrs,
        input_delay_ticks=2,
        max_rollback_ticks=8,
        tick_hz=60.0,
        checksum_interval=4,
        record_full_hash_history=True,
    )
    var s1 = RollbackSession[PhysicsRollbackInput](
        local_peer_id=1,
        peer_endpoints=peer_addrs,
        input_delay_ticks=2,
        max_rollback_ticks=8,
        tick_hz=60.0,
        checksum_interval=4,
        record_full_hash_history=True,
    )
    s0.enable_rift_pacing = False
    s1.enable_rift_pacing = False
    s0.bind_world(pw0)
    s1.bind_world(pw1)

    var hub_received = False
    comptime TARGET_TICKS: Int = 120
    for _ in range(260):
        var now = sim.now_ms()
        var c0 = _make_phys_input(0, s0.current_tick)
        var c1 = _make_phys_input(1, s1.current_tick)
        _ = s0.step_frame(c0, pw0, ep0, now)
        _ = s1.step_frame(c1, pw1, ep1, now)

        hub_host.poll(now)
        hub_cli.poll(now)
        for cid in range(hub_host.count):
            var popped = hub_host.pop_reliable(cid)
            if popped[0]:
                assert_equal(hash_list(popped[1]), hash_list(msg_out))
                hub_received = True

        if (
            len(s0.confirmed_hash_history) >= TARGET_TICKS
            and len(s1.confirmed_hash_history) >= TARGET_TICKS
            and hub_received
        ):
            break
        sim.advance_ms(1000.0 / 60.0)

    assert_true(hub_received)
    assert_false(s0.desync_detected)
    assert_false(s1.desync_detected)
    assert_true(len(s0.confirmed_hash_history) >= TARGET_TICKS)
    assert_true(len(s1.confirmed_hash_history) >= TARGET_TICKS)
    for t in range(TARGET_TICKS):
        assert_equal(s0.confirmed_hash_history[t], s1.confirmed_hash_history[t])

    print("[PASS] test_t3_p8_virtual_net_sim_rollback_nethub_physics")


# ===========================================================================
# Pair 9: FlintStudioState + McpServerV2
# ===========================================================================

def test_t3_p9_studio_state_mcp_v2_live_sync(tmp_root: String) raises:
    var ws = tmp_root + "/pair9_ws"
    assert_true(make_dirs(ws))

    var studio = FlintStudioState()
    var mcp = McpServerV2(workspace_root=ws)
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()

    try:
        var h0 = studio.state_hash()
        _ = mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}',
            studio,
            assets,
            pool,
        )

        var spawn_resp = mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ecs.spawn_entity","arguments":{"name":"LiveBox","position":[1.0,6.0,-2.0],"mass":3.0}}}',
            studio,
            assets,
            pool,
        )
        var spawn_doc = _parse_json_str(spawn_resp)
        var spawn_sc = spawn_doc.object_get(spawn_doc.object_get(spawn_doc.root, "result"), "structuredContent")
        var eid = spawn_doc.integer(spawn_doc.object_get(spawn_sc, "entity_id"), -1)
        assert_true(eid >= 0)
        assert_equal(studio.live_entity_count(), 1)
        var h1 = studio.state_hash()
        assert_true(h1 != h0)

        _ = mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ecs.patch_components","arguments":{"entity_id":' + String(eid) + ',"position":[1.0,8.0,-2.0]}}}',
            studio,
            assets,
            pool,
        )
        assert_equal(studio.entities[eid].translation.y, Float32(8.0))
        var h2 = studio.state_hash()

        _ = mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"play"}}}',
            studio,
            assets,
            pool,
        )
        _ = mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"step","ticks":20}}}',
            studio,
            assets,
            pool,
        )
        assert_true(studio.entities[eid].translation.y < Float32(8.0))

        var stop_resp = mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"studio.pie_control","arguments":{"action":"stop"}}}',
            studio,
            assets,
            pool,
        )
        var stop_doc = _parse_json_str(stop_resp)
        var stop_sc = stop_doc.object_get(stop_doc.object_get(stop_doc.root, "result"), "structuredContent")
        assert_true(stop_doc.boolean(stop_doc.object_get(stop_sc, "restored_exact"), False))
        assert_equal(studio.state_hash(), h2)

        _ = mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"studio.undo_redo","arguments":{"action":"undo","steps":1}}}',
            studio,
            assets,
            pool,
        )
        assert_equal(studio.state_hash(), h1)
        assert_equal(studio.entities[eid].translation.y, Float32(6.0))

        _ = mcp.handle_jsonrpc(
            '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"studio.undo_redo","arguments":{"action":"redo","steps":1}}}',
            studio,
            assets,
            pool,
        )
        assert_equal(studio.state_hash(), h2)
    finally:
        assets.close()
        pool.stop()

    print("[PASS] test_t3_p9_studio_state_mcp_v2_live_sync")


# ===========================================================================
# Pair 10: EngineBenchSuite + flip / image_diff
# ===========================================================================

def test_t3_p10_bench_suite_flip_image_diff(tmp_root: String) raises:
    var p10_dir = tmp_root + "/pair10"
    assert_true(make_dirs(p10_dir))

    comptime W: Int = 32
    comptime H: Int = 32
    var ref_px = List[UInt32](length=W * H, fill=UInt32(0xFF204080))
    var exact_px = ref_px.copy()
    var pert_px = ref_px.copy()
    for y in range(H):
        for x in range(W):
            if x >= 8 and x < 24 and y >= 8 and y < 24:
                pert_px[y * W + x] = UInt32(0xFFE0F020)

    var err_same = flip_error_map(ref_px, exact_px, W, H, ppd=FLIP_PPD)
    assert_equal(flip_mean(err_same), 0.0)
    var diff_same = compare_pixels(exact_px, ref_px, W, H, tolerance=2)
    assert_true(diff_same.passes(500, 0.05))

    var err_pert = flip_error_map(ref_px, pert_px, W, H, ppd=FLIP_PPD)
    var mean_pert = flip_mean(err_pert)
    assert_true(mean_pert > 0.05)
    var diff_pert = compare_pixels(pert_px, ref_px, W, H, tolerance=2)
    assert_false(diff_pert.passes(500, 0.05))

    var suite = EngineBenchSuite("t3_pair10")
    assert_true(
        suite.verify_golden_pixels(
            "exact_frame",
            exact_px,
            ref_px,
            W,
            H,
            heatmap_out_path=p10_dir + "/unused.png",
        )
    )
    var heat_path = p10_dir + "/pert_heat.png"
    assert_false(
        suite.verify_golden_pixels(
            "perturbed_frame",
            pert_px,
            ref_px,
            W,
            H,
            heatmap_out_path=heat_path,
        )
    )
    assert_true(file_size(heat_path) > 0)

    var samples = List[Float64]()
    samples.append(0.8)
    samples.append(1.1)
    samples.append(1.4)
    assert_true(suite.record_samples("frame_step", samples, budget_p99_ms=5.0))
    assert_equal(suite.passed_cases, 2)
    assert_equal(suite.failed_cases, 1)

    var csv_path = p10_dir + "/t3_perf.csv"
    assert_true(suite.append_perf_history_csv(csv_path, package_name="e2e-t3", date_str="2026-11-20"))
    assert_true(file_size(csv_path) > 0)
    print("[PASS] test_t3_p10_bench_suite_flip_image_diff")


def main() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp_root = "/tmp/flint_e2e_m3_t3_" + String(pid)
    _rm_rf_tmp(tmp_root)
    assert_true(make_dirs(tmp_root))

    try:
        test_t3_p1_engine_app_scene_transform_plugin()
        test_t3_p2_world_physics_world()
        test_t3_p3_physics_kcc_input_map()
        test_t3_p4_ragdoll_flanim_physics_world()
        test_t3_p5_cooker_flpk_asset_server_codecs(tmp_root)
        test_t3_p6_scalable_renderer_ui_dockspace_sdf_font()
        test_t3_p7_audio_server_qoa_3d_emitter()
        test_t3_p8_virtual_net_sim_rollback_nethub_physics()
        test_t3_p9_studio_state_mcp_v2_live_sync(tmp_root)
        test_t3_p10_bench_suite_flip_image_diff(tmp_root)
    except e:
        _rm_rf_tmp(tmp_root)
        raise e^

    _rm_rf_tmp(tmp_root)
    print("[SUMMARY] 10/10 passed")
