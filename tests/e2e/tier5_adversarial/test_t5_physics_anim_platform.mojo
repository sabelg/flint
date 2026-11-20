"""Tier 5 Adversarial Coverage Hardening: Physics, Anim, Core, Platform & DevTools (14 tests)."""
from std.collections import List
from std.ffi import external_call
from std.math import abs, isinf, isnan, sqrt
from flint.anim.graph import BlendSample2DDef, BlendSpace2DDef
from flint.anim.motion_matching import MotionDatabase, MotionFeature
from flint.anim.pose import SkeletonInstance
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeModel, NativeNode
from flint.audio import AudioCmd, AudioMpscRing, AudioServer, BUS_SFX
from flint.core.cvars import CVAR_COUNT, CVAR_DEBUG_DRAW, CVAR_DEBUG_RANGE, ConsoleVars
from flint.core.ik import fabrik
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat
from flint.devtools.mcp_v2 import FlintMcpUnixListener, McpUnixSessionClient, encode_mcp_frame
from flint.devtools.overlays import (
    FlintDevOverlays,
    KEY_DOWN,
    KEY_ENTER,
    KEY_F2,
    KEY_F4,
    KEY_F5,
    KEY_GRAVE,
    KEY_LEFT,
    KEY_RIGHT,
)
from flint.ecs import (
    Component,
    ComponentBitSet,
    Entity,
    ParallelCommands,
    STORAGE_TABLE,
    World,
    comp_hash64,
)
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.islands import ConstraintGraph, pair_key
from flint.physics.joints import Joint, JointRow
from flint.physics.narrowphase import ConvexHull, collide_box_box, collide_convex_convex
from flint.physics.solver import ContactPoint, Manifold, SolverSettings, ccd_cpus, solve
from flint.physics.solver_bodies import SolverBodies
from flint.physics.tires import PacejkaTire
from flint.physics.vehicle import VehicleChassis
from flint.physics.world import PhysicsWorld
from flint.platform.audio import AUDIO_CMD_PLAY, AUDIO_CMD_SET_LISTENER, AudioCommand, AudioSpscQueue, NativeAudio
from flint.platform.dsp import GranularBiomeSynth, ItdDelayLine
from flint.platform.input import InputState, PAD_AXIS_LEFT_X, PAD_AXIS_LEFT_Y, PadState
from flint.platform.wayland import WaylandWindow
from flint.platform.x11 import X11Window, x11_put32
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.simd_canvas import SimdCanvas
from e2e.common.harness import TestContext
from e2e.common.v5_fixtures import (
    create_mock_asset_server,
    create_mock_mcp_server,
    create_mock_pcm16_sine,
    create_mock_studio,
)


def _pid_str() -> String:
    return String(Int(external_call["getpid", Int32]()))


@fieldwise_init
struct _T5Pos(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e.t5.Pos")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var x: Float32
    var y: Float32


struct _T5NonTrivialUnreg(Component):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e.t5.NonTrivialUnreg")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var label: String

    def __init__(out self, label: String = "unreg"):
        self.label = label


def test_t5_p01_bvh8_coplanar_edge_symmetry(mut ctx: TestContext) raises:
    var verts = List[Vec3]()
    verts.append(Vec3(0, 0, 0)); verts.append(Vec3(0, 0, 2))
    verts.append(Vec3(2, 0, 0)); verts.append(Vec3(2, 0, 2))
    var idxs = List[Int32]()
    idxs.append(0); idxs.append(1); idxs.append(2)
    idxs.append(2); idxs.append(1); idxs.append(3)
    var bvh = Bvh8MeshCollider.build(verts, idxs)
    var cp1 = bvh.triangles[1].closest_point(Vec3(0.5, 0.2, 0.5))
    ctx.assert_vec3_approx(cp1[1], Vec3(0, 1, 0), 0.01, "coplanar tri1 shared edge must not snag with tilted normal")


def test_t5_p02_bvh8_sweep_overlap_and_parallel(mut ctx: TestContext) raises:
    var verts = List[Vec3]()
    verts.append(Vec3(0, 0, 0)); verts.append(Vec3(0, 0, 2)); verts.append(Vec3(2, 0, 0))
    var idxs = List[Int32](); idxs.append(0); idxs.append(1); idxs.append(2)
    var bvh = Bvh8MeshCollider.build(verts, idxs)
    var sh = bvh.sweep_sphere(Vec3(0.5, 0.2, 0.5), Vec3(0.0, -1.0, 0.0), 0.5, 1.0)
    ctx.assert_true(sh.hit and sh.toi == 0.0, "sphere already intersecting triangle at t=0 must report toi=0")
    var sh_par = bvh.sweep_sphere(Vec3(-0.3, 0.2, 0.5), Vec3(1.0, 0.0, 0.0), 0.5, 2.0)
    ctx.assert_true(sh_par.hit, "parallel sphere sweep hitting triangle edge must report hit")

    var sh_edge = bvh.sweep_sphere(Vec3(-0.6, 0.0, 1.0), Vec3(1.0, 0.0, 0.0), 0.5, 2.0)
    ctx.assert_true(sh_edge.hit, "horizontal edge sweep must hit active triangle edge")
    ctx.assert_approx(sh_edge.toi, Float32(0.1), Float32(0.02), "horizontal edge sweep TOI must be ~0.1")
    ctx.assert_vec3_approx(sh_edge.normal, Vec3(-1.0, 0.0, 0.0), Float32(0.05), "active edge sweep must preserve outward edge normal (-1,0,0)")

    var q_verts = List[Vec3]()
    q_verts.append(Vec3(0, 0, 0)); q_verts.append(Vec3(0, 0, 2))
    q_verts.append(Vec3(2, 0, 0)); q_verts.append(Vec3(2, 0, 2))
    var q_idxs = List[Int32]()
    q_idxs.append(0); q_idxs.append(1); q_idxs.append(2)
    q_idxs.append(2); q_idxs.append(1); q_idxs.append(3)
    var q_bvh = Bvh8MeshCollider.build(q_verts, q_idxs)
    var sh_corner = q_bvh.sweep_sphere(Vec3(2.7, 0.0, -0.05), Vec3(-1.0, 0.0, 0.0), 0.5, 2.0)
    ctx.assert_true(sh_corner.hit, "horizontal sweep into coplanar quad outer corner must hit")
    ctx.assert_approx(sh_corner.toi, Float32(0.2025), Float32(0.01), "corner sweep TOI must match sphere-vertex impact")
    ctx.assert_vec3_approx(
        sh_corner.normal,
        Vec3(Float32(0.9949874), Float32(0.0), Float32(-0.1)),
        Float32(0.05),
        "coplanar quad corner sweep must return outward horizontal normal (not smoothed tri.normal)",
    )
    var sh_diag = q_bvh.sweep_sphere(Vec3(1.0, 0.8, 1.0), Vec3(0.2, -1.0, 0.2), 0.5, 2.0)
    ctx.assert_true(sh_diag.hit, "sweep onto inactive diagonal must hit")
    ctx.assert_vec3_approx(sh_diag.normal, Vec3(0.0, 1.0, 0.0), Float32(0.05), "inactive diagonal interior sweep must remain smoothed to face normal")


def test_t5_p03_box_box_15_axis_sat(mut ctx: TestContext) raises:
    var rot_a = Quat(0, 0, Float32(0.38268343), Float32(0.92387953))
    var rot_b = Quat(Float32(0.38268343), 0, 0, Float32(0.92387953))
    var half = Vec3(1.0, 1.0, 1.0)
    var pb = Vec3(-2.0, 0.5, 2.4)
    var m = collide_box_box(Vec3(0, 0, 0), rot_a, half, pb, rot_b, half)
    ctx.assert_false(m.has_collision, "15-axis SAT must detect edge-edge cross-axis separation")

    var m_face = collide_box_box(Vec3(0, 0, 0), Quat(), half, Vec3(0, 1.8, 0), Quat(), half)
    ctx.assert_true(m_face.has_collision and m_face.count == 4, "box-box face overlap clips 4 contact points")

    var rot_45y = Quat.from_scaled_angle(Vec3(0.0, Float32(0.78539816), 0.0))
    var rot_tilt = Quat.from_scaled_angle(Vec3(Float32(0.35), Float32(0.78539816), Float32(0.15)))
    var pb_ov = Vec3(0.1, 1.45, 0.05)
    var m_rot = collide_box_box(Vec3(0, 0, 0), rot_45y, half, pb_ov, rot_tilt, half)
    ctx.assert_true(m_rot.has_collision and m_rot.count == 4, "rotated overlapping boxes yield 4-point manifold")
    ctx.assert_true(m_rot.p0.depth >= m_rot.p1.depth and m_rot.p0.depth >= m_rot.p2.depth and m_rot.p0.depth >= m_rot.p3.depth, "p0 must hold maximum penetration depth")
    var m_sep_after = collide_box_box(Vec3(0, 0, 0), rot_45y, half, pb_ov + m_rot.p0.normal * (m_rot.p0.depth + Float32(2.0e-3)), rot_tilt, half)
    ctx.assert_false(m_sep_after.has_collision, "pushing along p0.normal * p0.depth must separate rotated overlapping boxes")


def test_t5_p04_gjk_epa_accuracy(mut ctx: TestContext) raises:
    var rot_a = Quat(0, 0, Float32(0.38268343), Float32(0.92387953))
    var rot_b = Quat(Float32(0.38268343), 0, 0, Float32(0.92387953))
    var verts = List[Vec3]()
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                verts.append(Vec3(Float32(sx), Float32(sy), Float32(sz)))
    var ha = ConvexHull(verts)
    var hb = ConvexHull(verts)
    var m_sep = collide_convex_convex(ha, Vec3(0, 0, 0), rot_a, hb, Vec3(1.5, 2.4, 0.0), rot_b)
    ctx.assert_false(m_sep.has_collision, "GJK must not false-positive on separated rotated cubes")
    var m_ov = collide_convex_convex(ha, Vec3(0, 0, 0), Quat(), hb, Vec3(1.2, 0, 0), Quat())
    ctx.assert_true(m_ov.has_collision, "overlapping hulls collide")
    ctx.assert_approx(m_ov.p0.depth, 0.8, 0.05, "EPA must compute true penetration depth 0.8 (not hardcoded 0.05)")


def test_t5_p05_vehicle_brake_dt0_pacejka(mut ctx: TestContext) raises:
    var pw_coast = PhysicsWorld(gravity=9.81)
    _ = pw_coast.add_box(Vec3(0, -0.5, 0), Vec3(80, 0.5, 80), mass=0.0)
    var c_coast = VehicleChassis(1450.0); c_coast.setup_four_wheel_car()
    c_coast.pos = Vec3(0, 0.72, 0); c_coast.linear_vel = Vec3(0, 0, 20.0)
    var s_coast = pw_coast.add_vehicle(c_coast^)
    pw_coast.rebuild_broadphase()
    pw_coast.set_vehicle_controls(s_coast, Float32(0.0), Float32(0.0), Float32(0.0))

    var pw_brake = PhysicsWorld(gravity=9.81)
    _ = pw_brake.add_box(Vec3(0, -0.5, 0), Vec3(80, 0.5, 80), mass=0.0)
    var c_brake = VehicleChassis(1450.0); c_brake.setup_four_wheel_car()
    c_brake.pos = Vec3(0, 0.72, 0); c_brake.linear_vel = Vec3(0, 0, 20.0)
    var s_brake = pw_brake.add_vehicle(c_brake^)
    pw_brake.rebuild_broadphase()
    pw_brake.set_vehicle_controls(s_brake, Float32(0.0), Float32(1.0), Float32(0.0))

    for _ in range(10):
        pw_coast.step(Float32(1.0 / 60.0))
        pw_brake.step(Float32(1.0 / 60.0))

    var v_coast = pw_coast.vehicle_state(s_coast).linear_vel.z
    var v_brake = pw_brake.vehicle_state(s_brake).linear_vel.z
    ctx.assert_true(v_brake < v_coast - Float32(0.5), "applying brake=1.0 decelerates VehicleChassis in PhysicsWorld")

    var car_dt0 = VehicleChassis(); car_dt0.setup_four_wheel_car()
    car_dt0.pos = Vec3(0, 0.4, 0)
    car_dt0.substep_suspension(0.0, 0.0, 0.0, 0.0)
    ctx.assert_false(isnan(car_dt0.pos.y) or isinf(car_dt0.wheel_states[0].compression_rate), "dt=0 must not produce NaN/Inf")

    var tire = PacejkaTire()
    var f90 = tire.calc_forces(0.1, Float32(1.5707963), 4000.0, 20.0)
    ctx.assert_false(isnan(f90[0]) or isnan(f90[1]), "90-deg slip angle must remain finite")
    var f_launch = tire.calc_forces(1.0, 0.0, 4000.0, 0.0)
    ctx.assert_true(f_launch[0] > 10.0, "tire generates longitudinal launch traction at standstill with nonzero slip")


def test_t5_p06_solver_serial_parallel_parity(mut ctx: TestContext) raises:
    var cg = ConstraintGraph()
    var keys = List[Int](); var ra = List[Int](); var rb = List[Int]()
    for i in range(1, 30):
        keys.append(pair_key(0, i)); ra.append(0); rb.append(i)
    var cols = cg.assign(keys, ra, rb, 30)
    ctx.assert_equal(cg.overflow, 5, "29 constraints on body 0 spills 5 into OVERFLOW_COLOUR")
    ctx.assert_equal(cols[28], 24, "overflow colour index is 24")

    var bs = SolverBodies(); bs.reset(2)
    bs.px[0] = 0; bs.py[0] = 0.45; bs.vy[0] = -5.0; bs.inv_mass[0] = 1.0; bs.ixx[0] = 1; bs.iyy[0] = 1; bs.izz[0] = 1
    var bp = SolverBodies(); bp.reset(2)
    bp.px[0] = 0; bp.py[0] = 0.45; bp.vy[0] = -5.0; bp.inv_mass[0] = 1.0; bp.ixx[0] = 1; bp.iyy[0] = 1; bp.izz[0] = 1
    var ms = List[Manifold](); var m0 = Manifold(0, -1, False, 0, Vec3(0, 1, 0))
    m0.ia = 0; m0.ib = bs.static_index; m0.colour = 23
    m0.points.append(ContactPoint(Vec3(0, 0.0, 0), 0.05, 0))
    ms.append(m0^)
    var mp = ms.copy()
    var js = List[Joint](); var jp = List[Joint](); var jr = List[JointRow]()
    var ss = SolverSettings(); ss.force_serial = True
    var sp = SolverSettings(); sp.force_serial = False
    var pool = JobPool(); _ = pool.allocate(); _ = pool.start(2)
    var cpus = ccd_cpus()
    _ = solve(bs, ms, js, jr, ss, 1.0 / 60.0, 0, cpus)
    _ = solve(bp, mp, jp, jr, sp, 1.0 / 60.0, Int(Pointer(to=pool)), cpus)
    pool.stop()
    ctx.assert_approx(bs.vy[0], bp.vy[0], 1.0e-6, "serial and parallel solver must match bit-for-bit")


def test_t5_a01_skeleton_instance_reverse_hierarchy(mut ctx: TestContext) raises:
    var model = NativeModel()
    var child = NativeNode(); child.name = "child"; child.base_translation = Vec3(1, 0, 0)
    var parent = NativeNode(); parent.name = "parent"; parent.base_translation = Vec3(0, 5, 0)
    parent.children.append(0)
    model.nodes.append(child^); model.nodes.append(parent^)
    var skel = SkeletonInstance()
    skel.reset_to_rest(model)
    var world = skel.world_matrices[0].transform_point(Vec3(0, 0, 0))
    ctx.assert_vec3_approx(world, Vec3(1, 5, 0), 0.001, "child must inherit parent world transform even when roots is empty")


def test_t5_a02_blendspace2d_and_motion_wrap(mut ctx: TestContext) raises:
    var bs2 = BlendSpace2DDef()
    bs2.is_polar = True
    bs2.samples.append(BlendSample2DDef(0, 3.13, 1.0))
    bs2.samples.append(BlendSample2DDef(1, 0.0, 1.0))
    var w = bs2.weights(-3.13, 1.0)
    ctx.assert_true(w[0] > 0.9, "polar 2D blend space must wrap angle across +-pi")

    var db = MotionDatabase()
    var fp = MotionFeature(); fp.traj_facing_200 = 3.14; fp.traj_facing_400 = 3.14; fp.traj_facing_800 = 3.14
    db.add_frame(0, 0, 0.0, fp)
    var qn = MotionFeature(); qn.traj_facing_200 = -3.14; qn.traj_facing_400 = -3.14; qn.traj_facing_800 = -3.14
    var res = db.find_best_match(qn)
    ctx.assert_true(res[2] < 0.01, "MotionDatabase facing metric must wrap +-pi")


def test_t5_c01_ecs_despawn_state_transition_leak(mut ctx: TestContext) raises:
    var w = World()
    _ = w.register_component[_T5Pos]()

    var e0 = w.spawn()
    _ = w.insert[_T5Pos](e0, _T5Pos(1.0, 2.0))
    ctx.assert_true(w.is_alive(e0), "spawned entity is alive")
    ctx.assert_true(w.despawn(e0), "despawn succeeds")
    ctx.assert_false(w.is_alive(e0), "despawned handle rejected")
    ctx.assert_equal(w.entity_count(), 0, "entity_count returns to 0 after despawn")

    var e_reused = w.spawn()
    ctx.assert_false(w.is_alive(e0), "old generation handle stays dead after slot reuse")
    ctx.assert_true(w.is_alive(e_reused), "reused slot new generation handle is alive")

    # ParallelCommands: spawn + despawn existing gen-0 entity in same flush
    var cmds = ParallelCommands()
    var prov = cmds.spawn_empty(0, order_key=0)
    cmds.insert[_T5Pos](0, prov, _T5Pos(9.0, 10.0), order_key=1)
    cmds.despawn(0, e_reused, order_key=2)
    w.flush_commands(cmds)
    ctx.assert_false(w.is_alive(e_reused), "deferred despawn kills existing entity")
    ctx.assert_equal(w.entity_count(), 1, "only newly spawned deferred entity remains live")

    # Same-command-buffer spawn + despawn of provisional handle must not leak
    var rem = w.query_entities(w.bit_for[_T5Pos]())
    ctx.assert_equal(len(rem), 1, "one live entity with _T5Pos")
    var live_e = rem[0]
    var cmds_sd = ParallelCommands()
    var prov_sd = cmds_sd.spawn_empty(0, order_key=0)
    cmds_sd.despawn(0, prov_sd, order_key=1)
    cmds_sd.despawn(0, live_e, order_key=2)
    w.flush_commands(cmds_sd)
    ctx.assert_equal(w.entity_count(), 0, "same-buffer provisional spawn+despawn leaves zero entity leaks")

    # Mutate component via STORAGE_TABLE get_ptr_mut & verify state_checksum changes
    var e_mut = w.spawn()
    _ = w.insert[_T5Pos](e_mut, _T5Pos(3.0, 4.0))
    var cs0 = w.state_checksum()
    w.get_ptr_mut[_T5Pos](e_mut).value()[unsafe_offset=0].x = Float32(42.0)
    var cs1 = w.state_checksum()
    ctx.assert_true(cs0 != cs1, "World.state_checksum reflects STORAGE_TABLE component mutation")

    var cv = ConsoleVars()
    ctx.assert_approx(cv.get(-1), 0.0, 0.0001, "ConsoleVars.get guards negative index")
    ctx.assert_approx(cv.get(CVAR_COUNT), 0.0, 0.0001, "ConsoleVars.get guards out-of-range index")


def test_t5_c02_ecs_multichunk_jobpool_fabrik(mut ctx: TestContext) raises:
    var w = World()
    _ = w.register_component[_T5Pos]()
    # Register >260 POD components & verify extensible ComponentBitSet (>256 cap lifted)
    var base_hash = comp_hash64("e2e.t5.pod")
    for k in range(265):
        _ = w.register_pod_component(base_hash ^ UInt64(k + 1), 8)
    var bs = ComponentBitSet()
    bs.set(265)
    ctx.assert_true(bs.has(265), "ComponentBitSet supports component IDs > 256")
    ctx.assert_true(len(w.components) >= 266, "World supports >256 registered components")

    var hs = List[Entity]()
    for i in range(132):
        var e = w.spawn()
        _ = w.insert[_T5Pos](e, _T5Pos(Float32(i), Float32(i * 2)))
        hs.append(e)
    _ = w.despawn(hs[63]); _ = w.despawn(hs[64]); _ = w.despawn(hs[65])

    var blob = w.snapshot_blob()
    var w2 = World()
    _ = w2.register_component[_T5Pos]()
    w2.restore_blob(blob)
    ctx.assert_false(w2.is_alive(hs[64]), "despawned stale handle remains dead across snapshot_blob/restore_blob")
    ctx.assert_true(w2.is_alive(hs[100]), "live handle remains alive across snapshot_blob/restore_blob")
    ctx.assert_approx(w2.get[_T5Pos](hs[100]).x, Float32(100.0), 0.001, "component value preserved across snapshot_blob")

    # Unregistered non-trivial component must raise explicit error on snapshot_blob (no Clone Vault!)
    var w_bad = World()
    _ = w_bad.register_component[_T5NonTrivialUnreg]()
    var eb = w_bad.spawn()
    _ = w_bad.insert[_T5NonTrivialUnreg](eb, _T5NonTrivialUnreg("boom"))
    var raised = False
    try:
        _ = w_bad.snapshot_blob()
    except:
        raised = True
    ctx.assert_true(raised, "unregistered non-trivial component raises explicit error in snapshot_blob")

    var pts = List[Vec3]()
    pts.append(Vec3(0, 0, 0)); pts.append(Vec3(0, 1, 0)); pts.append(Vec3(0, 2, 0)); pts.append(Vec3(0, 3, 0))
    ctx.assert_true(fabrik(pts, Vec3(1.5, 1.5, 0.0), 20, 0.001), "FABRIK reaches 50% span target")
    ctx.assert_approx((pts[1] - pts[0]).length(), 1.0, 0.001, "FABRIK preserves bone length")


def test_t5_pl01_itd_delay_line_ramp(mut ctx: TestContext) raises:
    var itd = ItdDelayLine()
    for i in range(10):
        itd.push(Float32(i))
    var d05 = itd.read_delayed(0.5)
    ctx.assert_approx(d05, 8.5, 0.1, "0.5-sample delay on ramp 0..9 must interpolate between 9.0 and 8.0 (~8.5)")


def test_t5_pl02_audio_synths_spsc_and_mix(mut ctx: TestContext) raises:
    var gs = GranularBiomeSynth()
    var s_g = gs.step(1.0)
    ctx.assert_true(not isnan(s_g) and abs(s_g) <= 1.0, "GranularBiomeSynth compiles and outputs bounded audio")
    var q0 = AudioSpscQueue(0)
    ctx.assert_true(q0.capacity >= 2, "AudioSpscQueue clamps minimum capacity safely")
    var q = AudioSpscQueue(8)
    for i in range(10): _ = q.push(AudioCommand(AUDIO_CMD_PLAY, i))
    ctx.assert_true(q.is_full(), "capacity 8 ring fills at 7 items")
    var cmd = AudioCommand()
    for i in range(4):
        ctx.assert_true(q.pop(cmd) and cmd.sound_id == i, "FIFO order before wrap")
    for i in range(10, 14):
        ctx.assert_true(q.push(AudioCommand(AUDIO_CMD_PLAY, i)), "wraparound push succeeds")

    var audio = NativeAudio()
    _ = audio.send_command(AudioCommand(AUDIO_CMD_SET_LISTENER, pos_x=4.0, pos_y=5.0, pos_z=6.0))
    audio.process_commands()
    ctx.assert_vec3_approx(audio.listener_pos, Vec3(4.0, 5.0, 6.0), 0.001, "AUDIO_CMD_SET_LISTENER updates listener_pos")

    # AudioMpscRing wraparound + AudioServer TruePeakLimiter offline quantum guard
    var ring = AudioMpscRing.create(8)
    ctx.assert_true(ring.allocated(), "AudioMpscRing allocated")
    for i in range(4):
        var ac = AudioCmd(); ac.kind = UInt32(1); ac.slot = Int32(i)
        ctx.assert_true(ring.push(ac), "AudioMpscRing push succeeds")
    var ac_out = AudioCmd()
    for i in range(4):
        ctx.assert_true(ring.pop(ac_out) and Int(ac_out.slot) == i, "AudioMpscRing FIFO order")
    for i in range(10, 14):
        var ac2 = AudioCmd(); ac2.kind = UInt32(1); ac2.slot = Int32(i)
        ctx.assert_true(ring.push(ac2), "AudioMpscRing wraparound push succeeds")
    ring.destroy()

    var srv = AudioServer(48000, 256)
    var aid = srv.register_pcm16_asset(create_mock_pcm16_sine(880.0, 4800, 48000), 1, 48000)
    _ = srv.play_2d(aid, volume=3.5, bus_id=BUS_SFX, looping=True, direct_stereo=True)
    var out_lr = List[Float32](length=512, fill=0.0)
    for _ in range(4):
        srv.render_offline_quantum(out_lr, 256)
    var peak = Float32(0.0)
    for i in range(512):
        if abs(out_lr[i]) > peak:
            peak = abs(out_lr[i])
    ctx.assert_true(peak > Float32(0.1) and peak <= Float32(0.989), "AudioServer mixes active voice within TruePeakLimiter ceiling")


def test_t5_pl03_input_diagonal_toggle_mash_and_x11_symmetry(mut ctx: TestContext) raises:
    var pad = PadState()
    pad.apply_event(2, PAD_AXIS_LEFT_X, 32767)
    pad.apply_event(2, PAD_AXIS_LEFT_Y, 32767)
    var mag = sqrt(pad.axes[PAD_AXIS_LEFT_X] * pad.axes[PAD_AXIS_LEFT_X] + pad.axes[PAD_AXIS_LEFT_Y] * pad.axes[PAD_AXIS_LEFT_Y])
    ctx.assert_approx(mag, 1.0, 0.01, "diagonal stick clamps to unit circle")

    var wl = WaylandWindow(); var x11 = X11Window()
    ctx.assert_equal(wl.pointer_locked, x11.pointer_locked, "initial lock symmetric")
    x11.set_pointer_lock(True)
    ctx.assert_true(x11.pointer_locked, "X11 set_pointer_lock toggles state")
    x11.set_pointer_lock(False)
    x11.incoming.resize(64, 0)
    x11.incoming[0] = 3; x11.incoming[1] = 25; x11_put32(x11.incoming, 4, 12345)
    x11.incoming[32] = 2; x11.incoming[33] = 25; x11_put32(x11.incoming, 36, 12345)
    x11.process_events()
    ctx.assert_equal(len(x11.pending_keys), 0, "X11 filters synthetic KeyRelease+KeyPress auto-repeat")


def test_t5_d01_mcp_and_overlays(mut ctx: TestContext) raises:
    var studio = create_mock_studio()
    var assets = create_mock_asset_server()
    var pool = JobPool()
    var mcp = create_mock_mcp_server("/tmp")

    # 1. JSON-RPC 2.0 & MCP v2 error validation
    ctx.assert_true("-32700" in mcp.handle_jsonrpc("{bad json", studio, assets, pool), "McpServerV2 returns -32700 on parse error")
    ctx.assert_true("-32600" in mcp.handle_jsonrpc("{\"jsonrpc\":\"2.0\",\"id\":1}", studio, assets, pool), "McpServerV2 returns -32600 on missing method")
    ctx.assert_true("-32601" in mcp.handle_jsonrpc("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"no.such.method\",\"params\":{}}", studio, assets, pool), "McpServerV2 returns -32601 on unknown method")
    ctx.assert_true('"isError":true' in mcp.handle_jsonrpc("{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"bogus.tool\",\"arguments\":{}}}", studio, assets, pool), "McpServerV2 rejects unknown tool")
    ctx.assert_true('"isError":true' in mcp.handle_jsonrpc("{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"ecs.patch_components\",\"arguments\":{\"entity_id\":99999,\"position\":[1,2,3]}}}", studio, assets, pool), "McpServerV2 rejects invalid entity_id")
    ctx.assert_true('"isError":true' in mcp.handle_jsonrpc("{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"render.capture_viewport\",\"arguments\":{\"output_path\":\"../escape.png\"}}}", studio, assets, pool), "McpServerV2 rejects path traversal '..'")

    # 2. Live ECS + physics + renderer inspection via McpServerV2
    var r_sp = mcp.handle_jsonrpc("{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"tools/call\",\"params\":{\"name\":\"ecs.spawn_entity\",\"arguments\":{\"name\":\"AdvCube\"}}}", studio, assets, pool)
    ctx.assert_true('"isError":false' in r_sp and studio.live_entity_count() == 1, "ecs.spawn_entity creates live studio entity")
    var r_q = mcp.handle_jsonrpc("{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/call\",\"params\":{\"name\":\"ecs.query_entities\",\"arguments\":{}}}", studio, assets, pool)
    ctx.assert_true("AdvCube" in r_q, "ecs.query_entities returns spawned entity")
    var r_step = mcp.handle_jsonrpc("{\"jsonrpc\":\"2.0\",\"id\":8,\"method\":\"tools/call\",\"params\":{\"name\":\"physics.step_deterministic\",\"arguments\":{\"ticks\":3}}}", studio, assets, pool)
    ctx.assert_true('"isError":false' in r_step and studio.sim_tick == 3, "physics.step_deterministic advances 3 ticks")

    var ren = FlintAaaVulkanRenderer(width=64, height=64)
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active")
    _ = ren.add_material(AaaMaterial())
    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(-0.5, -0.5, -0.5), Vec3(0.5, 0.5, 0.5), 0.05)
    var me = ren.add_mesh(mb^.finish(), 0)
    ren.begin_frame()
    ren.draw_instance(me.mesh_id, Mat4())
    _ = ren.render_frame()
    var r_gpu = mcp.handle_jsonrpc_with_renderer(
        "{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"tools/call\",\"params\":{\"name\":\"render.inspect_gpu_frame\",\"arguments\":{}}}",
        studio, assets, pool, ren,
    )
    ctx.assert_true("VULKAN-AAA" in r_gpu and "aliasing_savings_ratio" in r_gpu, "render.inspect_gpu_frame reports VULKAN-AAA metrics")

    # 3. FlintMcpUnixListener + McpUnixSessionClient partial Content-Length framing & abrupt disconnect safety
    var sock_path = "/tmp/flint_e2e_t5_mcp_" + _pid_str() + ".sock"
    ctx.cleanup_artifact(sock_path)
    var srv = FlintMcpUnixListener(sock_path)
    try:
        ctx.assert_true(srv.active, "FlintMcpUnixListener active")

        var drop_cli = McpUnixSessionClient(use_content_length=True)
        ctx.assert_true(drop_cli.connect(sock_path), "drop client connects")
        _ = drop_cli.send_raw_request("{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"ping\",\"params\":{}}")
        drop_cli.close()
        _ = srv.poll_once(mcp, studio, assets, pool)

        var slow_cli = McpUnixSessionClient(use_content_length=True)
        ctx.assert_true(slow_cli.connect(sock_path), "slow partial-frame client connects")
        var frame_bytes = encode_mcp_frame("{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"ping\",\"params\":{}}", use_content_length=True)
        var split_at = len(frame_bytes) // 2
        var part1 = List[UInt8](capacity=split_at)
        for i in range(split_at): part1.append(frame_bytes[i])
        var part2 = List[UInt8](capacity=len(frame_bytes) - split_at)
        for i in range(split_at, len(frame_bytes)): part2.append(frame_bytes[i])

        ctx.assert_true(slow_cli.send_raw_bytes(part1), "send first half of Content-Length frame")
        ctx.assert_equal(srv.poll_once(mcp, studio, assets, pool), 0, "partial frame yields 0 processed requests")
        ctx.assert_equal(srv.client_count(), 1, "listener retains slow partial-frame client FD across frames")

        ctx.assert_true(slow_cli.send_raw_bytes(part2), "send second half of Content-Length frame")
        ctx.assert_equal(srv.poll_once(mcp, studio, assets, pool), 1, "listener answers once complete frame arrives")
        var resp = slow_cli.recv_response_blocking(2000)
        ctx.assert_true(resp[0] and '"id":11' in resp[1], "slow client receives framed JSON-RPC response")
        slow_cli.close()
    except e:
        srv.close()
        ctx.cleanup_artifact(sock_path)
        ren.shutdown()
        assets.close()
        studio.shutdown()
        pool.stop()
        raise e^

    srv.close()
    ctx.cleanup_artifact(sock_path)

    # 4. FlintDevOverlays GPU inspector sync + interactive CVar sliders
    var ov = FlintDevOverlays()
    var inp = InputState()
    inp.begin_frame(); inp.set_key(KEY_GRAVE, True)
    ctx.assert_true(ov.handle_input(inp), "grave opens console")
    inp.begin_frame(); inp.set_key(KEY_F2, True)
    ctx.assert_true(ov.handle_input(inp), "console swallows F2 while open")
    inp.begin_frame(); inp.set_key(KEY_GRAVE, False); inp.set_key(KEY_F2, False)
    inp.begin_frame(); inp.set_key(KEY_GRAVE, True)
    ctx.assert_false(ov.handle_input(inp), "grave closes console")
    inp.begin_frame(); inp.set_key(KEY_F2, True); inp.set_key(KEY_F4, True); inp.set_key(KEY_F5, True)
    _ = ov.handle_input(inp)
    ctx.assert_true(ov.show_profiler and ov.show_gpu_inspector and ov.show_cvars, "F2/F4/F5 toggle overlays when console closed")

    var init_draw = ov.cvars.get(CVAR_DEBUG_DRAW)
    inp.begin_frame(); inp.set_key(KEY_ENTER, True)
    ctx.assert_false(ov.handle_input(inp), "F5 slider interaction must not swallow game input")
    ctx.assert_true(ov.cvars.get(CVAR_DEBUG_DRAW) != init_draw, "ENTER toggles selected debug.draw CVar flag")
    var init_range = ov.cvars.get(CVAR_DEBUG_RANGE)
    inp.begin_frame(); inp.set_key(KEY_DOWN, True); inp.set_key(KEY_RIGHT, True)
    _ = ov.handle_input(inp)
    ctx.assert_approx(ov.cvars.get(CVAR_DEBUG_RANGE), init_range + Float32(5.0), Float32(0.01), "DOWN+RIGHT increments debug.range CVar by 5")
    inp.begin_frame(); inp.set_key(KEY_LEFT, True)
    _ = ov.handle_input(inp)
    ctx.assert_approx(ov.cvars.get(CVAR_DEBUG_RANGE), init_range, Float32(0.01), "LEFT decrements debug.range CVar back")

    var canvas = SimdCanvas(900, 600)
    canvas.clear()
    ov.render(canvas)
    ov.sync_aaa_renderer(ren)
    ctx.assert_true(ov.gpu_stats.attached, "sync_aaa_renderer attaches GPU stats")
    ctx.assert_equal(ov.gpu_stats.total_meshlets, ren.geom.total_meshlets(), "gpu_stats tracks real meshlet count")
    ctx.assert_equal(ov.gpu_stats.staging_kib, ren.buf_staging.size // 1024, "gpu_stats tracks real staging buffer KiB")
    ov.render(canvas)
    ren.shutdown()
    assets.close()
    studio.shutdown()
    pool.stop()


def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T5_Physics_Anim_Platform")
    try: test_t5_p01_bvh8_coplanar_edge_symmetry(ctx); ctx.record_pass("test_t5_p01_bvh8_coplanar_edge_symmetry") except e: ctx.record_fail("test_t5_p01_bvh8_coplanar_edge_symmetry", String(e))
    try: test_t5_p02_bvh8_sweep_overlap_and_parallel(ctx); ctx.record_pass("test_t5_p02_bvh8_sweep_overlap_and_parallel") except e: ctx.record_fail("test_t5_p02_bvh8_sweep_overlap_and_parallel", String(e))
    try: test_t5_p03_box_box_15_axis_sat(ctx); ctx.record_pass("test_t5_p03_box_box_15_axis_sat") except e: ctx.record_fail("test_t5_p03_box_box_15_axis_sat", String(e))
    try: test_t5_p04_gjk_epa_accuracy(ctx); ctx.record_pass("test_t5_p04_gjk_epa_accuracy") except e: ctx.record_fail("test_t5_p04_gjk_epa_accuracy", String(e))
    try: test_t5_p05_vehicle_brake_dt0_pacejka(ctx); ctx.record_pass("test_t5_p05_vehicle_brake_dt0_pacejka") except e: ctx.record_fail("test_t5_p05_vehicle_brake_dt0_pacejka", String(e))
    try: test_t5_p06_solver_serial_parallel_parity(ctx); ctx.record_pass("test_t5_p06_solver_serial_parallel_parity") except e: ctx.record_fail("test_t5_p06_solver_serial_parallel_parity", String(e))
    try: test_t5_a01_skeleton_instance_reverse_hierarchy(ctx); ctx.record_pass("test_t5_a01_skeleton_instance_reverse_hierarchy") except e: ctx.record_fail("test_t5_a01_skeleton_instance_reverse_hierarchy", String(e))
    try: test_t5_a02_blendspace2d_and_motion_wrap(ctx); ctx.record_pass("test_t5_a02_blendspace2d_and_motion_wrap") except e: ctx.record_fail("test_t5_a02_blendspace2d_and_motion_wrap", String(e))
    try: test_t5_c01_ecs_despawn_state_transition_leak(ctx); ctx.record_pass("test_t5_c01_ecs_despawn_state_transition_leak") except e: ctx.record_fail("test_t5_c01_ecs_despawn_state_transition_leak", String(e))
    try: test_t5_c02_ecs_multichunk_jobpool_fabrik(ctx); ctx.record_pass("test_t5_c02_ecs_multichunk_jobpool_fabrik") except e: ctx.record_fail("test_t5_c02_ecs_multichunk_jobpool_fabrik", String(e))
    try: test_t5_pl01_itd_delay_line_ramp(ctx); ctx.record_pass("test_t5_pl01_itd_delay_line_ramp") except e: ctx.record_fail("test_t5_pl01_itd_delay_line_ramp", String(e))
    try: test_t5_pl02_audio_synths_spsc_and_mix(ctx); ctx.record_pass("test_t5_pl02_audio_synths_spsc_and_mix") except e: ctx.record_fail("test_t5_pl02_audio_synths_spsc_and_mix", String(e))
    try: test_t5_pl03_input_diagonal_toggle_mash_and_x11_symmetry(ctx); ctx.record_pass("test_t5_pl03_input_diagonal_toggle_mash_and_x11_symmetry") except e: ctx.record_fail("test_t5_pl03_input_diagonal_toggle_mash_and_x11_symmetry", String(e))
    try: test_t5_d01_mcp_and_overlays(ctx); ctx.record_pass("test_t5_d01_mcp_and_overlays") except e: ctx.record_fail("test_t5_d01_mcp_and_overlays", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("test_t5_physics_anim_platform failed")
