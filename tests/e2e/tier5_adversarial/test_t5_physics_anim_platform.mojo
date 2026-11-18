"""Tier 5 Adversarial Coverage Hardening: Physics, Anim, Core, Platform & DevTools."""
from std.collections import List
from std.ffi import external_call
from std.math import abs, isnan, isinf, sqrt
from flint.core.math import Vec3
from flint.core.transform import Quat, trs
from flint.core.ecs import ArchetypeSlots, EntityHandle, EntityCommandBuffer, SimdColumn
from flint.core.jobs import JobPool
from flint.core.ik import fabrik
from flint.core.cvars import ConsoleVars, CVAR_COUNT, CVAR_DEBUG_DRAW, CVAR_DEBUG_RANGE
from flint.core.serial import BlobWriter, BlobReader
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.narrowphase import collide_box_box, collide_convex_convex, ConvexHull
from flint.physics.solver_bodies import SolverBodies
from flint.physics.solver import SolverSettings, Manifold, ContactPoint, solve, ccd_cpus
from flint.physics.joints import Joint, JointRow
from flint.physics.islands import ConstraintGraph, pair_key
from flint.physics.vehicle import VehicleChassis
from flint.physics.tires import PacejkaTire
from flint.anim.pose import SkeletonInstance
from flint.anim.graph import BlendSpace2DDef, BlendSample2DDef
from flint.anim.motion_matching import MotionDatabase, MotionFeature
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeModel, NativeNode
from flint.platform.dsp import ItdDelayLine, GranularBiomeSynth
from flint.platform.audio import NativeAudio, AudioSpscQueue, AudioCommand, AUDIO_CMD_PLAY, AUDIO_CMD_SET_LISTENER
from flint.platform.input import InputState, PadState, PAD_AXIS_LEFT_X, PAD_AXIS_LEFT_Y
from flint.platform.wayland import WaylandWindow
from flint.platform.x11 import X11Window, x11_put32
from flint.platform.os import close_fd, remove_file
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_clustered_lights import AaaClusterLight, AAA_LIGHT_SPOT
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.simd_canvas import SimdCanvas
from flint.devtools.mcp_server import (
    FlintMcpLiveEndpoint,
    FlintMcpInfo,
    handle_flint_mcp_query,
    apply_mcp_scene_mutations,
    mcp_scene_mutation_seq,
    mcp_scene_pending_count,
    mcp_scene_reset,
    query_flint_mcp_unix,
    _make_sockaddr_un,
)
from flint.devtools.overlays import (
    FlintDevOverlays,
    KEY_GRAVE,
    KEY_ENTER,
    KEY_RIGHT,
    KEY_LEFT,
    KEY_DOWN,
    KEY_UP,
    KEY_F2,
    KEY_F4,
    KEY_F5,
)
from e2e.common.harness import TestContext


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

    # Horizontal sweep into active vertical/lateral edge at x=0 must preserve outward edge normal (-1, 0, 0)
    var sh_edge = bvh.sweep_sphere(Vec3(-0.6, 0.0, 1.0), Vec3(1.0, 0.0, 0.0), 0.5, 2.0)
    ctx.assert_true(sh_edge.hit, "horizontal edge sweep must hit active triangle edge")
    ctx.assert_approx(sh_edge.toi, Float32(0.1), Float32(0.02), "horizontal edge sweep TOI must be ~0.1")
    ctx.assert_vec3_approx(sh_edge.normal, Vec3(-1.0, 0.0, 0.0), Float32(0.05), "active edge sweep must preserve outward edge normal (-1,0,0)")

    # Coplanar 2-triangle quad: horizontal sweep into outer corner vertex shared with inactive diagonal
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
    # Sweeping down onto inactive diagonal interior must still smooth to face normal (0, 1, 0)
    var sh_diag = q_bvh.sweep_sphere(Vec3(1.0, 0.8, 1.0), Vec3(0.2, -1.0, 0.2), 0.5, 2.0)
    ctx.assert_true(sh_diag.hit, "sweep onto inactive diagonal must hit")
    ctx.assert_vec3_approx(sh_diag.normal, Vec3(0.0, 1.0, 0.0), Float32(0.05), "inactive diagonal interior sweep must remain smoothed to face normal")


def test_t5_p03_box_box_15_axis_sat(mut ctx: TestContext) raises:
    var rot_a = Quat(0, 0, Float32(0.38268343), Float32(0.92387953)) # 45 deg Z
    var rot_b = Quat(Float32(0.38268343), 0, 0, Float32(0.92387953)) # 45 deg X
    var half = Vec3(1.0, 1.0, 1.0)
    var pb = Vec3(-2.0, 0.5, 2.4) # Overlapping on all 6 face axes, separated on cross axis
    var m = collide_box_box(Vec3(0, 0, 0), rot_a, half, pb, rot_b, half)
    ctx.assert_false(m.has_collision, "15-axis SAT must detect edge-edge cross-axis separation")

    # Face-face overlap should generate multi-point clipped contact manifold
    var m_face = collide_box_box(Vec3(0, 0, 0), Quat(), half, Vec3(0, 1.8, 0), Quat(), half)
    ctx.assert_true(m_face.has_collision and m_face.count == 4, "box-box face overlap clips 4 contact points")

    # 8-vertex clipped polygon (45-deg rotated + tilted) must select deepest point in p0 so push-out separates
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
    var car_coast = VehicleChassis(); car_coast.setup_four_wheel_car()
    car_coast.pos = Vec3(0, 0.55, 0); car_coast.linear_vel = Vec3(0, 0, 20.0)
    car_coast.substep_suspension(0.1, 0.0, 0.0, 0.0)
    var car_brake = VehicleChassis(); car_brake.setup_four_wheel_car()
    car_brake.pos = Vec3(0, 0.55, 0); car_brake.linear_vel = Vec3(0, 0, 20.0)
    car_brake.substep_suspension(0.1, 0.0, 1.0, 0.0)
    ctx.assert_true(car_brake.linear_vel.z < car_coast.linear_vel.z - 0.5, "applying brake=1.0 must decelerate vehicle")

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
    var slots = ArchetypeSlots(1, 16)
    var h = slots.spawn()
    ctx.assert_true(slots.despawn(h, deferred=True), "deferred despawn succeeds")
    _ = slots.despawn(h, deferred=False)
    slots.commit_tick()
    ctx.assert_equal(slots.live_count, 0, "live_count must return to 0 after immediate despawn of ROW_DESPAWNING")
    ctx.assert_equal(slots.pending, 0, "pending must be 0 after commit_tick")

    var ecb = EntityCommandBuffer()
    var eh = ecb.spawn(1)
    ctx.assert_true(eh.valid(), "ECB spawn returns valid provisional handle")
    ecb.playback(slots)
    slots.commit_tick()
    ctx.assert_equal(slots.live_count, 1, "ECB playback spawns 1 entity cleanly")

    # Gen-0 real handle must not collide with provisional handle; despawning real gen-0 in same ECB must despawn it
    var slots_g0 = ArchetypeSlots(1, 16)
    var h0 = slots_g0.spawn()
    var ecb_g0 = EntityCommandBuffer()
    var prov0 = ecb_g0.spawn(1)
    ctx.assert_true(prov0.generation != h0.generation, "ECB provisional handle generation must be disjoint from gen-0 real handle")
    ecb_g0.despawn(h0)
    ecb_g0.playback(slots_g0)
    slots_g0.commit_tick()
    ctx.assert_equal(slots_g0.resolve(h0), -1, "gen-0 real entity despawned via ECB must be dead")
    ctx.assert_equal(slots_g0.live_count, 1, "only the newly spawned ECB entity remains live")

    # Spawning and despawning the same provisional handle within one ECB must not leak a live entity
    var slots_sd = ArchetypeSlots(1, 16)
    var ecb_sd = EntityCommandBuffer()
    var prov_sd = ecb_sd.spawn(1)
    ecb_sd.despawn(prov_sd)
    ecb_sd.playback(slots_sd)
    slots_sd.commit_tick()
    ctx.assert_equal(slots_sd.live_count, 0, "same-ECB spawn+despawn must leave live_count == 0")
    ctx.assert_equal(slots_sd.pending, 0, "same-ECB spawn+despawn must leave pending == 0")

    var col = SimdColumn[DType.float32](8)
    var oob = col.load8(4)
    ctx.assert_approx(oob[0], 0.0, 0.0001, "SimdColumn.load8 guards OOB tail safely")

    var cv = ConsoleVars()
    ctx.assert_approx(cv.get(-1), 0.0, 0.0001, "ConsoleVars.get guards negative index")
    ctx.assert_approx(cv.get(CVAR_COUNT), 0.0, 0.0001, "ConsoleVars.get guards out-of-range index")


def test_t5_c02_ecs_multichunk_jobpool_fabrik(mut ctx: TestContext) raises:
    var slots = ArchetypeSlots(2, 130)
    var hs = List[EntityHandle]()
    for _ in range(130): hs.append(slots.spawn())
    ctx.assert_false(slots.spawn().valid(), "full pool rejects 131st spawn")
    _ = slots.despawn(hs[63]); _ = slots.despawn(hs[64]); _ = slots.despawn(hs[65])
    var r = slots.spawn()
    ctx.assert_equal(r.row, 63, "lowest-free reuse across chunk boundary picks row 63")
    var bw = BlobWriter(); slots.write_blob(bw)
    var br = BlobReader(bw.data.copy())
    var slots2 = ArchetypeSlots(2, 8)
    ctx.assert_true(slots2.read_blob(br), "blob deserializes")
    ctx.assert_equal(slots2.resolve(r), 63, "generation preserved across blob")

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
    # X11 auto-repeat filter check
    x11.incoming.resize(64, 0)
    x11.incoming[0] = 3; x11.incoming[1] = 25; x11_put32(x11.incoming, 4, 12345)
    x11.incoming[32] = 2; x11.incoming[33] = 25; x11_put32(x11.incoming, 36, 12345)
    x11.process_events()
    ctx.assert_equal(len(x11.pending_keys), 0, "X11 filters synthetic KeyRelease+KeyPress auto-repeat")


def test_t5_d01_mcp_and_overlays(mut ctx: TestContext) raises:
    var info = FlintMcpInfo()
    _ = handle_flint_mcp_query("flint_set_cvar", '{"name":"debug.draw","value":0.0}', info)
    var r_get = handle_flint_mcp_query("flint_get_cvar", '{"name":"debug.draw"}', info)
    ctx.assert_true('"value":0' in r_get, "flint_get_cvar must reflect prior flint_set_cvar mutation")

    # Verify genuine flint_mutate_scene validation & scene mutation pipeline
    _ = mcp_scene_reset()
    var seq0 = mcp_scene_mutation_seq()
    ctx.assert_true('"ok":false' in handle_flint_mcp_query("flint_mutate_scene", '{bad json', info), "flint_mutate_scene rejects invalid JSON")
    ctx.assert_true('"ok":false' in handle_flint_mcp_query("flint_mutate_scene", '{}', info), "flint_mutate_scene rejects missing op")
    ctx.assert_true('"ok":false' in handle_flint_mcp_query("flint_mutate_scene", '{"op":"bogus_op"}', info), "flint_mutate_scene rejects unsupported op")
    ctx.assert_true('"ok":false' in handle_flint_mcp_query("flint_mutate_scene", '{"op":"set_transform","position":[1,2,3]}', info), "flint_mutate_scene rejects missing entity_id")
    ctx.assert_true('"ok":false' in handle_flint_mcp_query("flint_mutate_scene", '{"op":"set_material","material_id":0,"roughness":1.5}', info), "flint_mutate_scene rejects out-of-range roughness")
    ctx.assert_true('"ok":false' in handle_flint_mcp_query("flint_mutate_scene", '{"op":"set_light","target":"sun","direction":[0,0,0]}', info), "flint_mutate_scene rejects zero-length light direction")
    ctx.assert_true('"ok":false' in handle_flint_mcp_query("flint_mutate_scene", '{"op":"set_cvar","name":"nonexistent.cvar","value":1}', info), "flint_mutate_scene rejects unknown cvar")

    var r_xform = handle_flint_mcp_query("flint_mutate_scene", '{"op":"set_transform","entity_id":0,"position":[3.0,4.0,-2.0],"scale":2.0}', info)
    var r_mat = handle_flint_mcp_query("flint_mutate_scene", '{"op":"set_material","material_id":0,"roughness":0.75,"base_color":[0.2,0.5,0.9]}', info)
    var r_light = handle_flint_mcp_query("flint_mutate_scene", '{"op":"set_light","target":"sun","direction":[0.0,1.0,0.0],"color":[4.0,3.0,2.0]}', info)
    var r_cam = handle_flint_mcp_query("flint_mutate_scene", '{"op":"set_camera","position":[1.0,2.0,-8.0],"target":[0.0,0.0,0.0],"fov_deg":70.0}', info)
    var r_cvar = handle_flint_mcp_query("flint_mutate_scene", '{"op":"set_cvar","name":"debug.range","value":75.0}', info)
    ctx.assert_true('"ok":true' in r_xform and '"ok":true' in r_mat and '"ok":true' in r_light and '"ok":true' in r_cam and '"ok":true' in r_cvar, "valid flint_mutate_scene ops succeed")
    ctx.assert_true(mcp_scene_mutation_seq() > seq0, "flint_mutate_scene advances mutation seq")
    ctx.assert_equal(mcp_scene_pending_count(), 4, "4 renderer scene mutations queued")

    var ren = FlintAaaVulkanRenderer(width=64, height=64)
    _ = ren.add_material(AaaMaterial())
    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(-0.5, -0.5, -0.5), Vec3(0.5, 0.5, 0.5), 0.05)
    var me = ren.add_mesh(mb^.finish(), 0)
    ren.begin_frame()
    ren.draw_instance(me.mesh_id, trs(Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1)))
    var applied = apply_mcp_scene_mutations(ren)
    ctx.assert_equal(applied, 4, "apply_mcp_scene_mutations applies all 4 queued scene mutations")
    ctx.assert_approx(ren.frame_instances[0].world_mat.values[12], Float32(3.0), Float32(0.01), "MCP set_transform updated instance X position")
    ctx.assert_approx(ren.materials[0].roughness, Float32(0.75), Float32(0.01), "MCP set_material updated material roughness")
    ctx.assert_vec3_approx(ren.sun_dir, Vec3(0.0, 1.0, 0.0), Float32(0.01), "MCP set_light updated sun_dir")
    ctx.assert_approx(ren.fov_y_deg, Float32(70.0), Float32(0.01), "MCP set_camera updated fov_y_deg")
    ctx.assert_equal(mcp_scene_pending_count(), 0, "mutation queue drained after apply")

    # Verify set_light with simultaneous position + direction on both existing and newly created cluster lights
    ren.add_cluster_light(AaaClusterLight.point(Vec3(1.0, 1.0, 1.0), Vec3(2.0, 2.0, 2.0), Float32(8.0)))
    var r_cl_exist = handle_flint_mcp_query(
        "flint_mutate_scene",
        '{"op":"set_light","light_id":0,"position":[5.0,6.0,7.0],"direction":[0.0,-1.0,0.0],"color":[2.0,3.0,4.0]}',
        info,
    )
    var r_cl_new = handle_flint_mcp_query(
        "flint_mutate_scene",
        '{"op":"set_light","light_id":1,"position":[-3.0,4.5,2.0],"direction":[1.0,0.0,0.0],"intensity":2.5}',
        info,
    )
    var r_cl_dir_only = handle_flint_mcp_query(
        "flint_mutate_scene",
        '{"op":"set_light","light_id":2,"direction":[0.0,0.0,-1.0]}',
        info,
    )
    ctx.assert_true(
        '"ok":true' in r_cl_exist and '"ok":true' in r_cl_new and '"ok":true' in r_cl_dir_only,
        "cluster light set_light mutations accepted",
    )
    var cl_applied = apply_mcp_scene_mutations(ren)
    ctx.assert_equal(cl_applied, 3, "all 3 cluster light mutations applied")
    ctx.assert_equal(len(ren.frame_cluster_lights), 3, "2 new cluster lights added alongside existing light")
    ctx.assert_vec3_approx(
        ren.frame_cluster_lights[0].position,
        Vec3(5.0, 6.0, 7.0),
        Float32(0.01),
        "existing cluster light keeps explicit position when direction also supplied",
    )
    ctx.assert_vec3_approx(
        ren.frame_cluster_lights[0].axis_u,
        Vec3(0.0, -1.0, 0.0),
        Float32(0.01),
        "existing cluster light updates axis_u when position also supplied",
    )
    ctx.assert_true(ren.frame_cluster_lights[0].kind == AAA_LIGHT_SPOT, "existing cluster light promoted to spot when direction supplied")
    ctx.assert_true(ren.frame_cluster_lights[0].param0 > Float32(0.8), "promoted cluster spotlight initializes cos_outer (param0)")
    ctx.assert_true(ren.frame_cluster_lights[0].param1 > Float32(8.0), "promoted cluster spotlight initializes inv_cos_range (param1)")
    ctx.assert_vec3_approx(
        ren.frame_cluster_lights[1].position,
        Vec3(-3.0, 4.5, 2.0),
        Float32(0.01),
        "new cluster light sets explicit position when direction also supplied",
    )
    ctx.assert_vec3_approx(
        ren.frame_cluster_lights[1].axis_u,
        Vec3(1.0, 0.0, 0.0),
        Float32(0.01),
        "new cluster light sets axis_u when position also supplied",
    )
    ctx.assert_true(ren.frame_cluster_lights[1].kind == AAA_LIGHT_SPOT, "new cluster light sets spot kind when direction supplied")
    ctx.assert_true(ren.frame_cluster_lights[1].param0 > Float32(0.8), "new cluster spotlight initializes cos_outer (param0)")
    ctx.assert_true(ren.frame_cluster_lights[1].param1 > Float32(8.0), "new cluster spotlight initializes inv_cos_range (param1)")
    ctx.assert_vec3_approx(
        ren.frame_cluster_lights[2].position,
        Vec3(0.0, 0.0, 0.0),
        Float32(0.01),
        "new cluster light with direction-only defaults position to origin (not direction)",
    )
    ctx.assert_vec3_approx(
        ren.frame_cluster_lights[2].axis_u,
        Vec3(0.0, 0.0, -1.0),
        Float32(0.01),
        "new cluster light with direction-only sets axis_u correctly",
    )
    ctx.assert_true(ren.frame_cluster_lights[2].kind == AAA_LIGHT_SPOT, "new direction-only cluster light sets spot kind")
    ctx.assert_true(ren.frame_cluster_lights[2].param0 > Float32(0.8), "new direction-only cluster spotlight initializes cos_outer (param0)")
    ctx.assert_true(ren.frame_cluster_lights[2].param1 > Float32(8.0), "new direction-only cluster spotlight initializes inv_cos_range (param1)")

    # Verify promoting an existing LTC tube light (param0=1.0, param1=0.25) resets spotlight cone params
    ren.add_cluster_light(AaaClusterLight.tube_ltc(Vec3(0.0, 2.0, 0.0), Vec3(2.0, 2.0, 0.0), Float32(0.25), Vec3(5.0, 5.0, 5.0)))
    var r_cl_tube = handle_flint_mcp_query(
        "flint_mutate_scene",
        '{"op":"set_light","light_id":3,"direction":[0.0,-1.0,0.0]}',
        info,
    )
    ctx.assert_true('"ok":true' in r_cl_tube, "tube cluster light set_light mutation accepted")
    ctx.assert_equal(apply_mcp_scene_mutations(ren), 1, "tube cluster light promotion applied")
    ctx.assert_true(ren.frame_cluster_lights[3].kind == AAA_LIGHT_SPOT, "promoted tube light becomes spot")
    ctx.assert_true(
        ren.frame_cluster_lights[3].param0 > Float32(0.8) and ren.frame_cluster_lights[3].param0 < Float32(0.9),
        "promoted tube light resets param0 from 1.0 to spot cos_outer",
    )
    ctx.assert_true(ren.frame_cluster_lights[3].param1 > Float32(8.0), "promoted tube light resets param1 from tube_radius to inv_cos_range")

    # Verify live Unix socket survives client disconnect without SIGPIPE (exit 141) AND retains slow EAGAIN clients across frames
    var sock_path = String("/tmp/flint_t5_mcp_sigpipe.sock")
    var srv = FlintMcpLiveEndpoint(sock_path)
    ctx.assert_true(srv.active, "MCP endpoint active")
    var cfd = Int(external_call["socket", Int32](Int32(1), Int32(1), Int32(0)))
    if cfd >= 0:
        var addr = _make_sockaddr_un(sock_path)
        if external_call["connect", Int32](Int32(cfd), addr.unsafe_ptr(), UInt32(110)) == 0:
            var req = String('{"method":"flint_get_cvar","args":{"name":"debug.draw"}}')
            var rb = List[UInt8]()
            for b in req.as_bytes(): rb.append(b)
            _ = external_call["write", Int64](Int32(cfd), rb.unsafe_ptr(), Int64(len(rb)))
        close_fd(cfd)
        _ = srv.poll_once(info)

    # Slow client: connect on frame 1 (EAGAIN), send on frame 2, verify answered on frame 2
    var slow_fd = Int(external_call["socket", Int32](Int32(1), Int32(1), Int32(0)))
    if slow_fd >= 0:
        var addr2 = _make_sockaddr_un(sock_path)
        if external_call["connect", Int32](Int32(slow_fd), addr2.unsafe_ptr(), UInt32(110)) == 0:
            ctx.assert_equal(srv.poll_once(info), 0, "slow EAGAIN client must yield 0 answers without dropping FD")
            ctx.assert_equal(len(srv.client_fds), 1, "slow EAGAIN client FD retained across poll_once")
            var req2 = String('{"method":"ping","args":"{}"}')
            var rb2 = List[UInt8]()
            for b in req2.as_bytes(): rb2.append(b)
            _ = external_call["write", Int64](Int32(slow_fd), rb2.unsafe_ptr(), Int64(len(rb2)))
            ctx.assert_equal(srv.poll_once(info), 1, "slow client answered once payload arrives")
            var rbuf = List[UInt8](length=256, fill=0)
            var rn = Int(external_call["read", Int64](Int32(slow_fd), rbuf.unsafe_ptr(), Int64(255)))
            ctx.assert_true(rn > 0, "slow client receives valid response without EPIPE")
        close_fd(slow_fd)
    srv.close()
    _ = remove_file(sock_path)
    var q_closed = query_flint_mcp_unix(sock_path, "ping", "{}")
    ctx.assert_true('"ok":false' in q_closed, "query_flint_mcp_unix returns error JSON when endpoint closed")

    # Verify DevOverlays GPU inspector sync + interactive CVar sliders
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

    # Interactive [F5] CVar slider adjustment (row 0 = debug.draw flag, row 1 = debug.range float)
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

    # GPU inspector unattached & attached telemetry + canvas render
    var canvas = SimdCanvas(900, 600)
    canvas.clear()
    ov.render(canvas)
    ov.sync_aaa_renderer(ren)
    ctx.assert_true(ov.gpu_stats.attached, "sync_aaa_renderer attaches GPU stats")
    ctx.assert_equal(ov.gpu_stats.total_meshlets, ren.geom.total_meshlets(), "gpu_stats tracks real meshlet count")
    ctx.assert_equal(ov.gpu_stats.staging_kib, ren.buf_staging.size // 1024, "gpu_stats tracks real staging buffer KiB (8192 KiB)")
    ov.render(canvas)
    ren.shutdown()


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
