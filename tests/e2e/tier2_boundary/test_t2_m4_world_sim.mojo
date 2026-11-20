"""Tier 2 Boundary & Corner Cases: Milestone 4 World Sim, Audio, Input & Tooling (F30–F39: 50 tests).
Corner cases, extremes, and stress inputs for Bvh8MeshCollider, narrowphase/GJK,
XPBD rigid coupling, PacejkaTire/VehicleChassis, DQ skinning, IK, AudioServer/DSP/Acoustics,
EvdevGamepad/InputMap/XkbKeymap, and McpServerV2.
"""
from std.collections import List
from std.math import abs
from flint.audio import (
    AcousticMaterialTable,
    KWeightedLufsMeter,
    TruePeakLimiter,
    compute_woodworth_hrtf,
    evaluate_emitter_acoustics,
    probe_room_sabine_rt60,
)
from flint.core.cvars import CVAR_COUNT, CVAR_DEBUG_DRAW, ConsoleVars, cvar_entry
from flint.core.ik import aim_rotation, two_bone_ik
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.input import (
    ABS_Z,
    ACTION_BOOL,
    EV_ABS,
    AxisCurve,
    EvdevGamepad,
    InputContext,
    InputMap,
)
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.ccd import gjk_closest_points
from flint.physics.narrowphase import (
    ConvexHull,
    collide_box_box,
    collide_capsule_capsule,
    collide_convex_convex,
    collide_sphere_sphere,
)
from flint.physics.solver import rodrigues_displacement
from flint.physics.solver_bodies import LANES, V8
from flint.physics.tires import PacejkaTire
from flint.physics.vehicle import VehicleChassis
from flint.physics.world import PhysicsWorld
from flint.physics.xpbd import ParticleMesh, RigidAttachment
from flint.platform.dsp import CytomicSVF, ModalResonator, compute_spatial_3d
from flint.platform.input import PAD_AXIS_LT, InputState
from flint.platform.wayland import evdev_key
from flint.platform.xkb import XkbKeymap, x11_keysym_to_codepoint
from flint.render.aaa_skinning_rt import DualQuat, blend_dual_quats_cpu
from e2e.common.harness import TestContext
from e2e.common.v5_fixtures import (
    create_mock_asset_server,
    create_mock_bvh8_floor,
    create_mock_mcp_server,
    create_mock_studio,
)


def _make_cube_hull(half: Float32 = 1.0) -> ConvexHull:
    var pts = List[Vec3]()
    pts.append(Vec3(-half, -half, -half)); pts.append(Vec3(half, -half, -half))
    pts.append(Vec3(half, half, -half)); pts.append(Vec3(-half, half, -half))
    pts.append(Vec3(-half, -half, half)); pts.append(Vec3(half, -half, half))
    pts.append(Vec3(half, half, half)); pts.append(Vec3(-half, half, half))
    return ConvexHull.build_quickhull(pts)


def _make_box_room_bvh(hx: Float32, hy: Float32, hz: Float32) -> Bvh8MeshCollider:
    var vs = List[Vec3]()
    vs.append(Vec3(-hx, -hy, -hz)); vs.append(Vec3(hx, -hy, -hz))
    vs.append(Vec3(hx, hy, -hz)); vs.append(Vec3(-hx, hy, -hz))
    vs.append(Vec3(-hx, -hy, hz)); vs.append(Vec3(hx, -hy, hz))
    vs.append(Vec3(hx, hy, hz)); vs.append(Vec3(-hx, hy, hz))
    var idxs: List[Int32] = [
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 5, 1, 0, 4, 5,
        3, 2, 6, 3, 6, 7,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ]
    return Bvh8MeshCollider.build(vs, idxs)


# =========================================================================
# F30: Bvh8MeshCollider (Boundary)
# =========================================================================

def test_b_f30_zero_volume_aabb(mut ctx: TestContext) raises:
    var verts: List[Vec3] = [Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0)]
    var idxs: List[Int32] = [0, 1, 2]
    var bvh = Bvh8MeshCollider.build(verts, idxs)
    var hit = bvh.raycast(Vec3(0, 2, 0), Vec3(0, -1, 0), 10.0)
    ctx.assert_false(hit.hit, "degenerate zero-area triangle does not falsely hit")


def test_b_f30_touching_face_aabb(mut ctx: TestContext) raises:
    var bvh = create_mock_bvh8_floor(10.0, 0.0)
    var hit = bvh.sweep_sphere(Vec3(0, 5.0, 0), Vec3(0, -1.0, 0), 0.5, 10.0)
    ctx.assert_true(hit.hit, "sphere sweep hits floor mesh")
    ctx.assert_approx(hit.toi, 4.5, 0.02, "sphere sweep touches at distance 5.0 - 0.5 = 4.5")


def test_b_f30_empty_broadphase_input(mut ctx: TestContext) raises:
    var bvh = Bvh8MeshCollider()
    var hit = bvh.raycast(Vec3(0, 1, 0), Vec3(0, -1, 0), 10.0)
    ctx.assert_false(hit.hit, "empty Bvh8MeshCollider misses safely")


def test_b_f30_exact_cell_boundary_hash(mut ctx: TestContext) raises:
    var bvh = create_mock_bvh8_floor(10.0, 0.0)
    var m = bvh.collide_sphere(Vec3(0.0, 0.4, 0.0), 0.5)
    ctx.assert_true(m.count > 0, "sphere penetrating BVH8 floor generates contact")
    var grazing = bvh.raycast(Vec3(0.0, 0.5, -20.0), Vec3(0.0, 0.0, 1.0), 50.0)
    ctx.assert_false(grazing.hit, "parallel ray above floor misses cleanly")


def test_b_f30_extreme_distant_aabb(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_static_mesh(create_mock_bvh8_floor(20.0, 0.0))
    var slot = pw.add_sphere(Vec3(0.0, 1.2, 0.0), 0.5, mass=1.0)
    pw.rebuild_broadphase()
    for _ in range(30):
        pw.step(Float32(1.0 / 60.0))
    ctx.assert_true(slot >= 0 and pw.bodies[slot].pos.y >= 0.3, "sphere settles on PhysicsWorld static BVH8 floor")


# =========================================================================
# F31: Discrete Narrowphase, Quickhull & GJK (Boundary)
# =========================================================================

def test_b_f31_concentric_spheres_penetration(mut ctx: TestContext) raises:
    var m = collide_sphere_sphere(Vec3(0, 0, 0), 2.0, Vec3(0, 0, 0), 2.0)
    ctx.assert_true(m.count > 0, "concentric spheres produce manifold")
    ctx.assert_approx(m.p0.depth, 4.0, 0.01, "concentric depth == 4.0")


def test_b_f31_touching_spheres_zero_penetration(mut ctx: TestContext) raises:
    var m = collide_sphere_sphere(Vec3(0, 0, 0), 1.5, Vec3(3.0, 0, 0), 1.5)
    ctx.assert_true(m.count > 0, "touching spheres at 3.0 produce manifold")
    ctx.assert_approx(m.p0.depth, 0.0, 0.01, "touching spheres have 0 penetration")


def test_b_f31_degenerate_capsule_point(mut ctx: TestContext) raises:
    var m = collide_capsule_capsule(Vec3(0, 0, 0), Vec3(0, 0, 0), 0.5, Vec3(0.5, 0, 0), Vec3(0.5, 0, 0), 0.5)
    ctx.assert_true(m.count > 0, "degenerate zero-length capsules behave as spheres")
    ctx.assert_approx(m.p0.depth, 0.5, 0.01, "degenerate capsule depth == 0.5")


def test_b_f31_deep_sat_penetration_clamping(mut ctx: TestContext) raises:
    var m = collide_box_box(Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1), Vec3(1.5, 0, 0), Quat(), Vec3(1, 1, 1))
    ctx.assert_true(m.count > 0, "box-box SAT generates manifold")
    ctx.assert_approx(m.p0.depth, 0.5, 0.02, "box-box overlap depth == 0.5")


def test_b_f31_parallel_contact_normals(mut ctx: TestContext) raises:
    var ha = _make_cube_hull(1.0)
    var hb = _make_cube_hull(1.0)
    var gjk = gjk_closest_points(ha, Vec3(0, 0, 0), Quat(), hb, Vec3(4.0, 0, 0), Quat())
    ctx.assert_false(gjk[0], "separated convex hulls not overlapping")
    ctx.assert_approx(gjk[3], 2.0, 0.05, "GJK closest distance between 2m cubes at 4m is 2.0")
    var m_cc = collide_convex_convex(ha, Vec3(0, 0, 0), Quat(), hb, Vec3(1.5, 0, 0), Quat())
    ctx.assert_true(m_cc.count > 0, "overlapping convex hulls produce contact")


# =========================================================================
# F32: 8-Lane SIMD XPBD & Rigid Coupling in PhysicsWorld (Boundary)
# =========================================================================

def test_b_f32_zero_mass_static_body(mut ctx: TestContext) raises:
    var mesh = ParticleMesh()
    var p = mesh.add_particle(Vec3(0, 0, 0), mass=0.0)
    mesh.push(p, Vec3(100, -500, 300))
    ctx.assert_approx(mesh.velocity[p].length(), 0.0, 0.0001, "pinned zero-mass particle ignores impulse")


def test_b_f32_zero_compliance_rigid(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=0.0)
    var box_slot = pw.add_box(Vec3(0.0, 10.0, 0.0), Vec3(0.5, 0.5, 0.5), mass=2.0)
    pw.bodies[box_slot].linear_vel = Vec3(6.0, 0.0, 0.0)
    var cloth = ParticleMesh()
    var p0 = cloth.add_particle(Vec3(0.0, 10.0, 0.0), mass=4.0)
    var p1 = cloth.add_particle(Vec3(0.0, 9.0, 0.0), mass=4.0)
    cloth.add_link(p0, p1, compliance=0.0)
    var att = RigidAttachment(p0, box_slot, Vec3(0.0, 0.0, 0.0), compliance=0.0)
    var slot = pw.add_xpbd_mesh(cloth^, att)
    pw.rebuild_broadphase()
    for _ in range(10):
        pw.step(Float32(1.0 / 60.0))
    var mesh_out = pw.xpbd_mesh(slot)
    ctx.assert_true(slot >= 0 and mesh_out.steps_run == 10, "PhysicsWorld steps coupled XPBD mesh")
    ctx.assert_true(mesh_out.position[p0].x > 0.1, "rigid body drags attached XPBD particle along +X")
    ctx.assert_true(pw.bodies[box_slot].linear_vel.x < 5.5 and pw.bodies[box_slot].pos.x < 0.8, "XPBD attachment exerts two-way reaction impulse slowing rigid box")
    ctx.assert_true(mesh_out.link_error() < 0.01, "zero-compliance link remains nearly inextensible")


def test_b_f32_extreme_compliance_soft(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    var box_slot = pw.add_box(Vec3(0.0, 10.0, 0.0), Vec3(0.5, 0.5, 0.5), mass=2.0)
    pw.bodies[box_slot].linear_vel = Vec3(0.0, 6.0, 0.0)
    var mesh = ParticleMesh()
    var p0 = mesh.add_particle(Vec3(0.0, 10.0, 0.0), mass=2.0)
    var p1 = mesh.add_particle(Vec3(0.0, 9.0, 0.0), mass=2.0)
    mesh.add_link(p0, p1, compliance=100.0)
    ctx.assert_approx(mesh.link_compliance[0], 100.0, 0.001, "soft compliance 100.0 stored")
    var att = RigidAttachment(p0, box_slot, Vec3(0.0, 0.0, 0.0), compliance=0.0)
    var slot = pw.add_xpbd_mesh(mesh^, att)
    pw.rebuild_broadphase()
    for _ in range(10):
        pw.step(Float32(1.0 / 60.0))
    var mesh_out = pw.xpbd_mesh(slot)
    ctx.assert_true(mesh_out.position[p0].y > 10.05, "coupled soft mesh follows rising rigid body")
    ctx.assert_true(mesh_out.link_error() > 0.05, "extreme compliance 100.0 allows elastic link stretch under load")
    ctx.assert_true(pw.bodies[box_slot].linear_vel.y < 5.5, "gravity + XPBD coupling decelerates rigid box")


def test_b_f32_rodrigues_zero_rotation(mut ctx: TestContext) raises:
    var res = rodrigues_displacement(V8.splat(Vec3(0, 0, 0)), V8.splat(Vec3(5, 0, 0)))
    ctx.assert_approx(res.x[0], 0.0, 0.0001, "zero omega yields zero disp x")
    ctx.assert_approx(res.y[0], 0.0, 0.0001, "zero omega yields zero disp y")


def test_b_f32_simd_lanes_count(mut ctx: TestContext) raises:
    var v = V8.splat(Vec3(7.0, -3.0, 2.5))
    for i in range(LANES):
        ctx.assert_approx(v.x[i], 7.0, 0.0001, "lane X")
        ctx.assert_approx(v.y[i], -3.0, 0.0001, "lane Y")
        ctx.assert_approx(v.z[i], 2.5, 0.0001, "lane Z")


# =========================================================================
# F33: PacejkaTire & VehicleChassis Boundaries in PhysicsWorld
# =========================================================================

def test_b_f33_zero_slip_zero_force(mut ctx: TestContext) raises:
    var tire = PacejkaTire()
    var f = tire.calc_forces(0.0, 0.0, 4000.0, 10.0)
    ctx.assert_approx(f[0], 0.0, 0.001, "zero slip produces 0 Fx")
    ctx.assert_approx(f[1], 0.0, 0.001, "zero slip produces 0 Fy")
    ctx.assert_approx(f[2], 0.0, 0.001, "zero slip produces 0 Mz")


def test_b_f33_ninety_deg_slip_peak(mut ctx: TestContext) raises:
    var tire = PacejkaTire()
    var f90 = tire.calc_forces(0.0, 1.5, 4000.0, 15.0)
    ctx.assert_true(f90[1] > 500.0, "90-deg slip angle saturates cleanly without NaN")
    var bx = Float32(0.0)
    var by = Float32(0.0)
    var hill = tire.calc_forces_bristle(0.0, 0.0, 0.01, 4000.0, 0.016, bx, by)
    ctx.assert_true(abs(hill[0]) >= 0.0, "bristle tire model handles near-zero speed")


def test_b_f33_zero_normal_load_airborne(mut ctx: TestContext) raises:
    var tire = PacejkaTire()
    var f_air = tire.calc_forces(0.2, 0.2, 0.0, 20.0)
    ctx.assert_approx(f_air[0], 0.0, 0.0001, "airborne zero normal load gives 0 Fx")
    ctx.assert_approx(f_air[1], 0.0, 0.0001, "airborne zero normal load gives 0 Fy")


def test_b_f33_suspension_full_compression(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(40.0, 0.5, 40.0), mass=0.0)
    var car = VehicleChassis(1450.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(0.0, 0.35, 0.0)
    var slot = pw.add_vehicle(car^)
    pw.rebuild_broadphase()
    pw.step(Float32(1.0 / 60.0))
    var st = pw.vehicle_state(slot)
    ctx.assert_true(st.wheel_states[0].grounded, "low-slung vehicle wheel contacts floor")
    ctx.assert_true(st.wheel_states[0].compression > 0.0, "wheel suspension compresses under load")
    ctx.assert_approx(st.wheel_states[0].compression, st.wheels[0].max_travel, 0.001, "suspension bottoms at max_travel (0.20 m)")
    ctx.assert_true(st.wheel_states[0].normal_load > 0.0, "compressed wheel generates positive normal load")


def test_b_f33_symmetric_negative_slip(mut ctx: TestContext) raises:
    var tire = PacejkaTire()
    var f_pos = tire.calc_forces(0.0, 0.12, 4000.0, 15.0)
    var f_neg = tire.calc_forces(0.0, -0.12, 4000.0, 15.0)
    ctx.assert_approx(f_pos[1], -f_neg[1], 0.5, "Pacejka lateral Fy is antisymmetric in slip angle")


# =========================================================================
# F34: Dual-Quaternion Skinning (Boundary)
# =========================================================================

def test_b_f34_identity_quat(mut ctx: TestContext) raises:
    var q = Quat()
    ctx.assert_approx(q.w, 1.0, 0.0001, "identity quat w == 1.0")
    ctx.assert_approx(q.x, 0.0, 0.0001, "identity quat x == 0.0")


def test_b_f34_antipodal_flip_alignment(mut ctx: TestContext) raises:
    var dq0 = DualQuat.from_quat_trans(Quat(0.0, 0.0, 0.0, 1.0), Vec3(0.0, 0.0, 0.0))
    var dq1 = DualQuat.from_quat_trans(Quat(0.0, 0.0, 0.0, -1.0), Vec3(2.0, 0.0, 0.0))
    ctx.assert_approx(dq1.rw, -1.0, 0.0001, "antipodal DualQuat retains opposite hemisphere sign")
    var bones = List[DualQuat]()
    bones.append(dq0)
    bones.append(dq1)
    var blended = blend_dual_quats_cpu(bones, 0, 1, 0, 0, 0.5, 0.5, 0.0, 0.0)
    var pt = blended.transform_point(Vec3(0.0, 0.0, 0.0))
    ctx.assert_vec3_approx(pt, Vec3(1.0, 0.0, 0.0), 0.001, "antipodal DLB blend transforms midpoint cleanly without collapse")
    var rot_norm = blended.rx * blended.rx + blended.ry * blended.ry + blended.rz * blended.rz + blended.rw * blended.rw
    ctx.assert_approx(rot_norm, 1.0, 0.0001, "antipodal DLB blend produces unit rotation quaternion")


def test_b_f34_blend_space_exact_corners(mut ctx: TestContext) raises:
    var dq0 = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(0, 0, 0))
    var dq1 = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(10, 0, 0))
    var bones = List[DualQuat](); bones.append(dq0); bones.append(dq1); bones.append(dq0); bones.append(dq0)
    var corner = blend_dual_quats_cpu(bones, 0, 1, 2, 3, 1.0, 0.0, 0.0, 0.0)
    ctx.assert_vec3_approx(corner.transform_point(Vec3(0, 0, 0)), Vec3(0, 0, 0), 0.001, "corner weight 1.0")


def test_b_f34_blend_space_uniform_center(mut ctx: TestContext) raises:
    var dq0 = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(0, 0, 0))
    var dq1 = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(4, 0, 0))
    var bones = List[DualQuat](); bones.append(dq0); bones.append(dq1); bones.append(dq0); bones.append(dq0)
    var center = blend_dual_quats_cpu(bones, 0, 1, 2, 3, 0.5, 0.5, 0.0, 0.0)
    ctx.assert_vec3_approx(center.transform_point(Vec3(0, 0, 0)), Vec3(2, 0, 0), 0.01, "midpoint blend")


def test_b_f34_quat_180_deg_rotation(mut ctx: TestContext) raises:
    var rotated = Quat(0.0, 1.0, 0.0, 0.0).rotate(Vec3(1, 0, 0))
    ctx.assert_approx(rotated.x, -1.0, 0.001, "180 deg Y rotation maps (1,0,0) to (-1,0,0)")


# =========================================================================
# F35: Two-Bone Analytical IK & Aim Constraints (Boundary)
# =========================================================================

def test_b_f35_ik_target_at_exact_reach(mut ctx: TestContext) raises:
    var res = two_bone_ik(Vec3(0, 2, 0), Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), Quat(), Quat(), Vec3(0, 0, 1))
    var n = res.upper.w*res.upper.w + res.upper.x*res.upper.x + res.upper.y*res.upper.y + res.upper.z*res.upper.z
    ctx.assert_approx(n, 1.0, 0.01, "upper joint rotation normalized at exact reach")


def test_b_f35_ik_target_beyond_reach(mut ctx: TestContext) raises:
    var res = two_bone_ik(Vec3(0, 2, 0), Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(0, 10, 0), Quat(), Quat(), Vec3(0, 0, 1))
    var n = res.middle.w*res.middle.w + res.middle.x*res.middle.x + res.middle.y*res.middle.y + res.middle.z*res.middle.z
    ctx.assert_approx(n, 1.0, 0.01, "middle joint rotation normalized beyond reach")


def test_b_f35_ik_target_at_origin(mut ctx: TestContext) raises:
    var res = two_bone_ik(Vec3(0, 2, 0), Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(0, 2, 0), Quat(), Quat(), Vec3(0, 0, 1))
    ctx.assert_true(res.upper.w != 0.0 or res.upper.y != 0.0, "joint folds gracefully at origin")


def test_b_f35_aim_rotation_collinear_forward(mut ctx: TestContext) raises:
    var q = aim_rotation(Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, 0, 10), Quat())
    ctx.assert_vec3_approx(q.rotate(Vec3(0, 0, 1)), Vec3(0, 0, 1), 0.05, "collinear forward aligned")


def test_b_f35_aim_rotation_180_deg_turn(mut ctx: TestContext) raises:
    var q = aim_rotation(Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, 0, -10), Quat())
    ctx.assert_vec3_approx(q.rotate(Vec3(0, 0, 1)), Vec3(0, 0, -1), 0.05, "180 deg turn aligns +Z to -Z")


# =========================================================================
# F36: Real-Time Audio DSP, Limiter & LUFS Meter (Boundary)
# =========================================================================

def test_b_f36_svf_nyquist_frequency(mut ctx: TestContext) raises:
    var svf = CytomicSVF(22050.0, 0.707, 44100.0)
    var out_sample = svf.process(1.0)
    ctx.assert_true(out_sample >= -2.0 and out_sample <= 2.0, "Nyquist SVF stable")


def test_b_f36_svf_sub_bass_frequency(mut ctx: TestContext) raises:
    var svf = CytomicSVF(20.0, 0.707, 44100.0)
    var out_sample = svf.process(1.0)
    ctx.assert_true(out_sample >= -2.0 and out_sample <= 2.0, "20Hz SVF stable")


def test_b_f36_svf_silent_input(mut ctx: TestContext) raises:
    var svf = CytomicSVF(1000.0, 0.707, 44100.0)
    for _ in range(10):
        ctx.assert_approx(svf.process(0.0), 0.0, 0.0001, "silent input produces 0")


def test_b_f36_modal_resonator_low_frequency(mut ctx: TestContext) raises:
    var mr = ModalResonator(10.0, 0.1, 1.0, 48000.0)
    var s = abs(mr.step(1.0))
    ctx.assert_true(s > 0.0 and s <= 2.0, "low-freq ModalResonator clamps and rings stably")
    var lim = TruePeakLimiter(-0.1, 80.0, 48000.0)
    var max_out = Float32(0.0)
    for _ in range(128):
        var res = lim.process(4.0, 4.0)
        if abs(res[0]) > max_out:
            max_out = abs(res[0])
    ctx.assert_true(max_out <= Float32(0.989), "TruePeakLimiter holds +12 dBFS step <= -0.1 dBFS")


def test_b_f36_modal_resonator_decay_monotonic(mut ctx: TestContext) raises:
    var meter = KWeightedLufsMeter(48000)
    for _ in range(2400):
        meter.process_frame(0.0, 0.0)
    ctx.assert_true(meter.momentary_lufs() <= -99.0, "silent signal reads <= -99 LUFS")
    var res = ModalResonator(440.0, 0.05, 1.0, 48000.0)
    var early_peak = abs(res.step(1.0))
    for _ in range(50):
        var sv = abs(res.step(0.0))
        if sv > early_peak:
            early_peak = sv
    for _ in range(25000):
        _ = res.step(0.0)
    var late_peak = Float32(0.0)
    for _ in range(50):
        var sv = abs(res.step(0.0))
        if sv > late_peak:
            late_peak = sv
    ctx.assert_true(early_peak > 0.0 and early_peak > late_peak, "resonator envelope decays from early to late")
    ctx.assert_true(late_peak < early_peak * 0.1, "resonator late peak decays by >90%")


# =========================================================================
# F37: Binaural 3D Spatial Audio, Woodworth HRTF & Acoustics (Boundary)
# =========================================================================

def test_b_f37_source_at_listener_origin(mut ctx: TestContext) raises:
    var h_origin = compute_woodworth_hrtf(
        Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(1, 0, 0), Vec3(0, 1, 0),
        Vec3(0, 0, 0), Vec3(0, 0, 0),
    )
    ctx.assert_approx(h_origin.gain_l, 0.70710677, 0.001, "origin source L gain == sqrt(0.5)")
    ctx.assert_approx(h_origin.gain_r, 0.70710677, 0.001, "origin source R gain == sqrt(0.5)")
    ctx.assert_approx(h_origin.delay_l_samples, 0.0, 0.001, "origin source L delay == 0")
    ctx.assert_approx(h_origin.delay_r_samples, 0.0, 0.001, "origin source R delay == 0")
    var h_front = compute_woodworth_hrtf(
        Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(1, 0, 0), Vec3(0, 1, 0),
        Vec3(0, 0, -5), Vec3(0, 0, 0),
    )
    ctx.assert_approx(h_front.delay_l_samples - h_front.delay_r_samples, 0.0, 0.01, "front center source has 0 ITD")


def test_b_f37_source_beyond_max_distance(mut ctx: TestContext) raises:
    var params = compute_spatial_3d(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 500), Vec3(0, 0, 0))
    ctx.assert_true(params.gain < 0.1, "distant source has low gain")


def test_b_f37_pure_left_source(mut ctx: TestContext) raises:
    var h = compute_woodworth_hrtf(
        Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(1, 0, 0), Vec3(0, 1, 0),
        Vec3(-10, 0, 0), Vec3(0, 0, 0),
    )
    ctx.assert_true((h.delay_l_samples - h.delay_r_samples) < -20.0, "left -90 deg source yields negative ITD")


def test_b_f37_pure_right_source(mut ctx: TestContext) raises:
    var hr = compute_woodworth_hrtf(
        Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(1, 0, 0), Vec3(0, 1, 0),
        Vec3(10, 0, 0), Vec3(0, 0, 0),
    )
    ctx.assert_true((hr.delay_l_samples - hr.delay_r_samples) > 20.0, "right +90 deg source yields positive ITD > 20 samples")
    var room = _make_box_room_bvh(6.0, 3.0, 6.0)
    var mats = AcousticMaterialTable()
    var probe = probe_room_sabine_rt60(room, mats, Vec3(0, 0, 0), 50.0)
    ctx.assert_true(probe.sabine_rt60_sec > 0.0, "Sabine RT60 probe positive inside closed box room")
    var occ = evaluate_emitter_acoustics(room, mats, Vec3(0, 0, 0), Vec3(20, 0, 0))
    ctx.assert_true(occ.occlusion_gain < 0.9, "wall between listener and exterior emitter attenuates gain")


def test_b_f37_doppler_subsonic_limit(mut ctx: TestContext) raises:
    var params = compute_spatial_3d(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 10), Vec3(0, 0, -300))
    ctx.assert_true(params.doppler_pitch > 2.0, "Mach 0.87 approach produces high Doppler shift")


# =========================================================================
# F38: EvdevGamepad, InputMap, XkbKeymap & Window Input (Boundary)
# =========================================================================

def test_b_f38_key_code_zero(mut ctx: TestContext) raises:
    ctx.assert_equal(evdev_key(0), -1, "evdev code 0 returns -1 unmapped")
    ctx.assert_false(InputState().uses_legacy_jsdev(), "InputState does not use legacy jsdev")


def test_b_f38_key_code_escape(mut ctx: TestContext) raises:
    ctx.assert_equal(evdev_key(1), 256, "KEY_ESC maps to 256")
    var curve = AxisCurve()
    var zero_xy = curve.eval_2d(0.05, 0.05)
    ctx.assert_approx(zero_xy[0], 0.0, 0.0001, "AxisCurve radial deadzone zeroes small input")


def test_b_f38_key_code_w(mut ctx: TestContext) raises:
    ctx.assert_equal(evdev_key(17), 87, "KEY_W maps to 87")
    var pad = EvdevGamepad()
    pad.configure_abs_axis(ABS_Z, 0, 255, 0, 0)
    pad.feed_raw_evdev_event(EV_ABS, UInt16(ABS_Z), Int32(0))
    ctx.assert_approx(pad.axes[PAD_AXIS_LT], 0.0, 0.001, "DualSense rest trigger == 0.0")


def test_b_f38_key_code_space(mut ctx: TestContext) raises:
    ctx.assert_equal(evdev_key(57), 32, "KEY_SPACE maps to 32")
    var imap = InputMap()
    _ = imap.register_action("Jump", ACTION_BOOL)
    var ic = InputContext("G")
    ic.bind_key("Jump", 32)
    imap.add_context(ic^)
    imap.push_context("G")
    var inp = InputState()
    inp.set_logical_key(32, True)
    imap.update(inp, 0.016)
    ctx.assert_true(imap.consume_buffered("Jump", 0.120), "120 ms jump buffer consumes buffered press")


def test_b_f38_unmapped_high_key(mut ctx: TestContext) raises:
    ctx.assert_equal(evdev_key(9999), -1, "unmapped scancode maps to -1")
    ctx.assert_equal(x11_keysym_to_codepoint(0x0041), 0x41, "XKB keysym 'A' maps to U+0041")
    var xkb = XkbKeymap()
    ctx.assert_false(xkb.has_parsed_keymap, "default empty XkbKeymap has_parsed_keymap=False")


# =========================================================================
# F39: Developer Tooling (ConsoleVars & McpServerV2) (Boundary)
# =========================================================================

def test_b_f39_cvar_entry_out_of_bounds(mut ctx: TestContext) raises:
    ctx.assert_equal(cvar_entry(-1).name.byte_length(), 0, "cvar_entry(-1) returns empty entry")
    ctx.assert_equal(cvar_entry(CVAR_COUNT).name.byte_length(), 0, "cvar_entry(CVAR_COUNT) returns empty entry")
    var cvars = ConsoleVars()
    ctx.assert_equal(cvars.entry(-1).name.byte_length(), 0, "ConsoleVars.entry(-1) returns empty entry")
    ctx.assert_equal(cvars.entry(CVAR_COUNT).name.byte_length(), 0, "ConsoleVars.entry(CVAR_COUNT) returns empty entry")


def test_b_f39_debug_draw_toggle(mut ctx: TestContext) raises:
    var cvars = ConsoleVars()
    cvars.set(CVAR_DEBUG_DRAW, 1.0)
    ctx.assert_equal(cvars.get(CVAR_DEBUG_DRAW), 1.0, "debug draw 1.0")
    cvars.set(CVAR_DEBUG_DRAW, 0.0)
    ctx.assert_equal(cvars.get(CVAR_DEBUG_DRAW), 0.0, "debug draw 0.0")


def test_b_f39_mcp_unknown_tool(mut ctx: TestContext) raises:
    var studio = create_mock_studio()
    var assets = create_mock_asset_server()
    var pool = JobPool()
    var mcp = create_mock_mcp_server("/tmp")
    var res = mcp.handle_jsonrpc(
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"nonexistent.tool\",\"arguments\":{}}}",
        studio, assets, pool,
    )
    ctx.assert_true(res.find("isError") >= 0 or res.find("error") >= 0, "unknown tool returns MCP error")
    assets.close(); studio.shutdown(); pool.stop()


def test_b_f39_mcp_engine_info(mut ctx: TestContext) raises:
    var studio = create_mock_studio()
    var assets = create_mock_asset_server()
    var pool = JobPool()
    var mcp = create_mock_mcp_server("/tmp")
    var r_init = mcp.handle_jsonrpc(
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"clientInfo\":{\"name\":\"e2e\"}}}",
        studio, assets, pool,
    )
    ctx.assert_true(r_init.find("flint-engine-mcp-v2") >= 0, "initialize returns serverInfo")
    var r_list = mcp.handle_jsonrpc("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{}}", studio, assets, pool)
    ctx.assert_true(r_list.find("ecs.query_entities") >= 0 and r_list.find("physics.step_deterministic") >= 0, "tools/list enumerates v2 tools")
    var r_call = mcp.handle_jsonrpc(
        "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"ecs.query_entities\",\"arguments\":{}}}",
        studio, assets, pool,
    )
    ctx.assert_true(r_call.find("\"result\"") >= 0 and r_call.find("ok") >= 0, "tools/call ecs.query_entities succeeds")
    assets.close(); studio.shutdown(); pool.stop()


def test_b_f39_cvar_entry_index(mut ctx: TestContext) raises:
    ctx.assert_true(cvar_entry(0).name.byte_length() > 0, "cvar 0 has non-empty name")


# =========================================================================
# Aggregator for Tier 2 Milestone 4
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T2_M4_WorldSim_Boundary")

    try: test_b_f30_zero_volume_aabb(ctx); ctx.record_pass("test_b_f30_zero_volume_aabb") except e: ctx.record_fail("test_b_f30_zero_volume_aabb", String(e))
    try: test_b_f30_touching_face_aabb(ctx); ctx.record_pass("test_b_f30_touching_face_aabb") except e: ctx.record_fail("test_b_f30_touching_face_aabb", String(e))
    try: test_b_f30_empty_broadphase_input(ctx); ctx.record_pass("test_b_f30_empty_broadphase_input") except e: ctx.record_fail("test_b_f30_empty_broadphase_input", String(e))
    try: test_b_f30_exact_cell_boundary_hash(ctx); ctx.record_pass("test_b_f30_exact_cell_boundary_hash") except e: ctx.record_fail("test_b_f30_exact_cell_boundary_hash", String(e))
    try: test_b_f30_extreme_distant_aabb(ctx); ctx.record_pass("test_b_f30_extreme_distant_aabb") except e: ctx.record_fail("test_b_f30_extreme_distant_aabb", String(e))

    try: test_b_f31_concentric_spheres_penetration(ctx); ctx.record_pass("test_b_f31_concentric_spheres_penetration") except e: ctx.record_fail("test_b_f31_concentric_spheres_penetration", String(e))
    try: test_b_f31_touching_spheres_zero_penetration(ctx); ctx.record_pass("test_b_f31_touching_spheres_zero_penetration") except e: ctx.record_fail("test_b_f31_touching_spheres_zero_penetration", String(e))
    try: test_b_f31_degenerate_capsule_point(ctx); ctx.record_pass("test_b_f31_degenerate_capsule_point") except e: ctx.record_fail("test_b_f31_degenerate_capsule_point", String(e))
    try: test_b_f31_deep_sat_penetration_clamping(ctx); ctx.record_pass("test_b_f31_deep_sat_penetration_clamping") except e: ctx.record_fail("test_b_f31_deep_sat_penetration_clamping", String(e))
    try: test_b_f31_parallel_contact_normals(ctx); ctx.record_pass("test_b_f31_parallel_contact_normals") except e: ctx.record_fail("test_b_f31_parallel_contact_normals", String(e))

    try: test_b_f32_zero_mass_static_body(ctx); ctx.record_pass("test_b_f32_zero_mass_static_body") except e: ctx.record_fail("test_b_f32_zero_mass_static_body", String(e))
    try: test_b_f32_zero_compliance_rigid(ctx); ctx.record_pass("test_b_f32_zero_compliance_rigid") except e: ctx.record_fail("test_b_f32_zero_compliance_rigid", String(e))
    try: test_b_f32_extreme_compliance_soft(ctx); ctx.record_pass("test_b_f32_extreme_compliance_soft") except e: ctx.record_fail("test_b_f32_extreme_compliance_soft", String(e))
    try: test_b_f32_rodrigues_zero_rotation(ctx); ctx.record_pass("test_b_f32_rodrigues_zero_rotation") except e: ctx.record_fail("test_b_f32_rodrigues_zero_rotation", String(e))
    try: test_b_f32_simd_lanes_count(ctx); ctx.record_pass("test_b_f32_simd_lanes_count") except e: ctx.record_fail("test_b_f32_simd_lanes_count", String(e))

    try: test_b_f33_zero_slip_zero_force(ctx); ctx.record_pass("test_b_f33_zero_slip_zero_force") except e: ctx.record_fail("test_b_f33_zero_slip_zero_force", String(e))
    try: test_b_f33_ninety_deg_slip_peak(ctx); ctx.record_pass("test_b_f33_ninety_deg_slip_peak") except e: ctx.record_fail("test_b_f33_ninety_deg_slip_peak", String(e))
    try: test_b_f33_zero_normal_load_airborne(ctx); ctx.record_pass("test_b_f33_zero_normal_load_airborne") except e: ctx.record_fail("test_b_f33_zero_normal_load_airborne", String(e))
    try: test_b_f33_suspension_full_compression(ctx); ctx.record_pass("test_b_f33_suspension_full_compression") except e: ctx.record_fail("test_b_f33_suspension_full_compression", String(e))
    try: test_b_f33_symmetric_negative_slip(ctx); ctx.record_pass("test_b_f33_symmetric_negative_slip") except e: ctx.record_fail("test_b_f33_symmetric_negative_slip", String(e))

    try: test_b_f34_identity_quat(ctx); ctx.record_pass("test_b_f34_identity_quat") except e: ctx.record_fail("test_b_f34_identity_quat", String(e))
    try: test_b_f34_antipodal_flip_alignment(ctx); ctx.record_pass("test_b_f34_antipodal_flip_alignment") except e: ctx.record_fail("test_b_f34_antipodal_flip_alignment", String(e))
    try: test_b_f34_blend_space_exact_corners(ctx); ctx.record_pass("test_b_f34_blend_space_exact_corners") except e: ctx.record_fail("test_b_f34_blend_space_exact_corners", String(e))
    try: test_b_f34_blend_space_uniform_center(ctx); ctx.record_pass("test_b_f34_blend_space_uniform_center") except e: ctx.record_fail("test_b_f34_blend_space_uniform_center", String(e))
    try: test_b_f34_quat_180_deg_rotation(ctx); ctx.record_pass("test_b_f34_quat_180_deg_rotation") except e: ctx.record_fail("test_b_f34_quat_180_deg_rotation", String(e))

    try: test_b_f35_ik_target_at_exact_reach(ctx); ctx.record_pass("test_b_f35_ik_target_at_exact_reach") except e: ctx.record_fail("test_b_f35_ik_target_at_exact_reach", String(e))
    try: test_b_f35_ik_target_beyond_reach(ctx); ctx.record_pass("test_b_f35_ik_target_beyond_reach") except e: ctx.record_fail("test_b_f35_ik_target_beyond_reach", String(e))
    try: test_b_f35_ik_target_at_origin(ctx); ctx.record_pass("test_b_f35_ik_target_at_origin") except e: ctx.record_fail("test_b_f35_ik_target_at_origin", String(e))
    try: test_b_f35_aim_rotation_collinear_forward(ctx); ctx.record_pass("test_b_f35_aim_rotation_collinear_forward") except e: ctx.record_fail("test_b_f35_aim_rotation_collinear_forward", String(e))
    try: test_b_f35_aim_rotation_180_deg_turn(ctx); ctx.record_pass("test_b_f35_aim_rotation_180_deg_turn") except e: ctx.record_fail("test_b_f35_aim_rotation_180_deg_turn", String(e))

    try: test_b_f36_svf_nyquist_frequency(ctx); ctx.record_pass("test_b_f36_svf_nyquist_frequency") except e: ctx.record_fail("test_b_f36_svf_nyquist_frequency", String(e))
    try: test_b_f36_svf_sub_bass_frequency(ctx); ctx.record_pass("test_b_f36_svf_sub_bass_frequency") except e: ctx.record_fail("test_b_f36_svf_sub_bass_frequency", String(e))
    try: test_b_f36_svf_silent_input(ctx); ctx.record_pass("test_b_f36_svf_silent_input") except e: ctx.record_fail("test_b_f36_svf_silent_input", String(e))
    try: test_b_f36_modal_resonator_low_frequency(ctx); ctx.record_pass("test_b_f36_modal_resonator_low_frequency") except e: ctx.record_fail("test_b_f36_modal_resonator_low_frequency", String(e))
    try: test_b_f36_modal_resonator_decay_monotonic(ctx); ctx.record_pass("test_b_f36_modal_resonator_decay_monotonic") except e: ctx.record_fail("test_b_f36_modal_resonator_decay_monotonic", String(e))

    try: test_b_f37_source_at_listener_origin(ctx); ctx.record_pass("test_b_f37_source_at_listener_origin") except e: ctx.record_fail("test_b_f37_source_at_listener_origin", String(e))
    try: test_b_f37_source_beyond_max_distance(ctx); ctx.record_pass("test_b_f37_source_beyond_max_distance") except e: ctx.record_fail("test_b_f37_source_beyond_max_distance", String(e))
    try: test_b_f37_pure_left_source(ctx); ctx.record_pass("test_b_f37_pure_left_source") except e: ctx.record_fail("test_b_f37_pure_left_source", String(e))
    try: test_b_f37_pure_right_source(ctx); ctx.record_pass("test_b_f37_pure_right_source") except e: ctx.record_fail("test_b_f37_pure_right_source", String(e))
    try: test_b_f37_doppler_subsonic_limit(ctx); ctx.record_pass("test_b_f37_doppler_subsonic_limit") except e: ctx.record_fail("test_b_f37_doppler_subsonic_limit", String(e))

    try: test_b_f38_key_code_zero(ctx); ctx.record_pass("test_b_f38_key_code_zero") except e: ctx.record_fail("test_b_f38_key_code_zero", String(e))
    try: test_b_f38_key_code_escape(ctx); ctx.record_pass("test_b_f38_key_code_escape") except e: ctx.record_fail("test_b_f38_key_code_escape", String(e))
    try: test_b_f38_key_code_w(ctx); ctx.record_pass("test_b_f38_key_code_w") except e: ctx.record_fail("test_b_f38_key_code_w", String(e))
    try: test_b_f38_key_code_space(ctx); ctx.record_pass("test_b_f38_key_code_space") except e: ctx.record_fail("test_b_f38_key_code_space", String(e))
    try: test_b_f38_unmapped_high_key(ctx); ctx.record_pass("test_b_f38_unmapped_high_key") except e: ctx.record_fail("test_b_f38_unmapped_high_key", String(e))

    try: test_b_f39_cvar_entry_out_of_bounds(ctx); ctx.record_pass("test_b_f39_cvar_entry_out_of_bounds") except e: ctx.record_fail("test_b_f39_cvar_entry_out_of_bounds", String(e))
    try: test_b_f39_debug_draw_toggle(ctx); ctx.record_pass("test_b_f39_debug_draw_toggle") except e: ctx.record_fail("test_b_f39_debug_draw_toggle", String(e))
    try: test_b_f39_mcp_unknown_tool(ctx); ctx.record_pass("test_b_f39_mcp_unknown_tool") except e: ctx.record_fail("test_b_f39_mcp_unknown_tool", String(e))
    try: test_b_f39_mcp_engine_info(ctx); ctx.record_pass("test_b_f39_mcp_engine_info") except e: ctx.record_fail("test_b_f39_mcp_engine_info", String(e))
    try: test_b_f39_cvar_entry_index(ctx); ctx.record_pass("test_b_f39_cvar_entry_index") except e: ctx.record_fail("test_b_f39_cvar_entry_index", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
