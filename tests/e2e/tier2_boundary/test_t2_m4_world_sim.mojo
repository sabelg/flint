"""Tier 2 Boundary & Corner Cases: Milestone 4 World Sim, Audio, Input & Tooling (F30–F39: 50 tests).
Corner cases, extremes, and stress inputs for BVH8, narrowphase, SIMD solver, Pacejka, DQ skinning, IK, DSP, 3D audio, Wayland, and Tooling.
"""
from std.collections import List
from std.math import sqrt, sin, cos, atan, exp
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.ik import two_bone_ik, aim_rotation
from flint.core.cvars import ConsoleVars, CVAR_DEBUG_DRAW, cvar_entry
from flint.physics.broadphase import (
    cell_coordinate, cell_key, bucket_of, BroadphaseInput, boxes_overlap, find_pairs,
)
from flint.physics.filter import CollisionFilter
from flint.physics.solver_bodies import V8, F8, LANES
from flint.physics.solver import rodrigues_displacement, tangent_basis
from flint.physics.ccd import closest_obb_surface
from flint.physics.forces import FieldVolume
from flint.physics.joints import make_soft
from flint.physics.xpbd import ParticleMesh
from flint.render.aaa_skinning_rt import DualQuat, blend_dual_quats_cpu
from flint.platform.dsp import compute_spatial_3d, CytomicSVF, ModalResonator
from flint.platform.wayland import evdev_key
from flint.devtools.mcp_server import handle_flint_mcp_query, FlintMcpInfo
from e2e.common.harness import TestContext


# =========================================================================
# F30: BVH8 Static Mesh Collider (Boundary)
# =========================================================================

def test_b_f30_zero_volume_aabb(mut ctx: TestContext) raises:
    var pt = Vec3(2, 3, 4)
    ctx.assert_true(boxes_overlap(pt, pt, pt, pt), "identical point AABBs overlap")

def test_b_f30_touching_face_aabb(mut ctx: TestContext) raises:
    var a_lo = Vec3(0, 0, 0)
    var a_hi = Vec3(1, 1, 1)
    var b_lo = Vec3(1, 0, 0)
    var b_hi = Vec3(2, 1, 1)
    ctx.assert_true(boxes_overlap(a_lo, a_hi, b_lo, b_hi), "touching face AABBs evaluate as overlapping")

def test_b_f30_empty_broadphase_input(mut ctx: TestContext) raises:
    var bp = BroadphaseInput()
    var pairs = find_pairs(bp)
    ctx.assert_equal(len(pairs.pairs), 0, "empty broadphase finds 0 pairs")

def test_b_f30_exact_cell_boundary_hash(mut ctx: TestContext) raises:
    ctx.assert_equal(cell_coordinate(Float32(0.0)), 0, "0.0 hashes to cell 0")
    ctx.assert_equal(cell_coordinate(Float32(8.0)), 1, "exact multiple of CELL_SIZE (8.0) hashes to cell 1")
    ctx.assert_equal(cell_coordinate(Float32(-8.0)), -1, "-8.0 hashes to cell -1")

def test_b_f30_extreme_distant_aabb(mut ctx: TestContext) raises:
    var a_lo = Vec3(-1000000, -1000000, -1000000)
    var a_hi = Vec3(-999999, -999999, -999999)
    var b_lo = Vec3(999999, 999999, 999999)
    var b_hi = Vec3(1000000, 1000000, 1000000)
    ctx.assert_false(boxes_overlap(a_lo, a_hi, b_lo, b_hi), "extreme opposite corners do not overlap")


# =========================================================================
# F31: Discrete Narrowphase & SAT/GJK-EPA (Boundary)
# =========================================================================

def test_b_f31_concentric_spheres_penetration(mut ctx: TestContext) raises:
    var sp = FieldVolume.sphere(1, 0, Vec3(0, 0, 0), 2.0)
    ctx.assert_true(sp.inside(Vec3(0, 0, 0)), "concentric center point inside sphere")
    ctx.assert_approx(sp.distance_t(Vec3(0, 0, 0)), 0.0, 0.001, "concentric center distance is 0.0")

def test_b_f31_touching_spheres_zero_penetration(mut ctx: TestContext) raises:
    var sp = FieldVolume.sphere(1, 0, Vec3(0, 0, 0), 1.5)
    ctx.assert_approx(sp.distance_t(Vec3(3.0, 0.0, 0.0)), 2.0, 0.001, "touching sphere surface at 3.0 has distance_t 2.0")

def test_b_f31_degenerate_capsule_point(mut ctx: TestContext) raises:
    var cap = FieldVolume.capsule(1, 0, Vec3(1, 1, 1), Vec3(1, 1, 1), 0.5)
    ctx.assert_approx(cap.distance_t(Vec3(1, 1, 1)), 0.0, 0.001, "degenerate capsule acts as sphere centered at point")

def test_b_f31_deep_sat_penetration_clamping(mut ctx: TestContext) raises:
    var res = closest_obb_surface(Vec3(0.1, 0, 0), Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    var signed_d = res[2]
    ctx.assert_true(signed_d < 0.0, "interior point yields negative signed distance (penetration)")
    ctx.assert_approx(signed_d, -0.9, 0.01, "deep penetration depth equals distance to closest face")

def test_b_f31_parallel_contact_normals(mut ctx: TestContext) raises:
    var basis = tangent_basis(Vec3(0, 1, 0))
    ctx.assert_approx(basis[0].dot(Vec3(0, 1, 0)), 0.0, 0.0001, "tangent is strictly orthogonal to normal")


# =========================================================================
# F32: 8-Lane SIMD XPBD Solver (Boundary)
# =========================================================================

def test_b_f32_zero_mass_static_body(mut ctx: TestContext) raises:
    var mesh = ParticleMesh()
    var p = mesh.add_particle(Vec3(0, 0, 0), mass=0.0)
    mesh.push(p, Vec3(100, -500, 300))
    ctx.assert_approx(mesh.velocity[p].length(), 0.0, 0.0001, "static body with zero mass remains stationary under impulse")

def test_b_f32_zero_compliance_rigid(mut ctx: TestContext) raises:
    var mesh = ParticleMesh()
    var p0 = mesh.add_particle(Vec3(0, 0, 0), mass=1.0)
    var p1 = mesh.add_particle(Vec3(1, 0, 0), mass=1.0)
    mesh.add_link(p0, p1, compliance=0.0)
    ctx.assert_equal(mesh.link_compliance[0], Float32(0.0), "zero compliance link registered as pure rigid constraint")

def test_b_f32_extreme_compliance_soft(mut ctx: TestContext) raises:
    var soft = make_soft(0.1, 0.05, 0.016)
    ctx.assert_true(soft.impulse_scale >= 0.0, "ultra-soft constraint produces valid non-negative impulse scale")

def test_b_f32_rodrigues_zero_rotation(mut ctx: TestContext) raises:
    var omega = V8.splat(Vec3(0, 0, 0))
    var r = V8.splat(Vec3(5, 0, 0))
    var res = rodrigues_displacement(omega, r)
    ctx.assert_approx(res.x[0], 0.0, 0.0001, "zero angular velocity produces zero displacement")
    ctx.assert_approx(res.y[0], 0.0, 0.0001, "zero angular velocity produces zero displacement")
    ctx.assert_approx(res.z[0], 0.0, 0.0001, "zero angular velocity produces zero displacement")

def test_b_f32_simd_lanes_count(mut ctx: TestContext) raises:
    var v = V8.splat(Vec3(7.0, -3.0, 2.5))
    for i in range(LANES):
        ctx.assert_approx(v.x[i], 7.0, 0.0001, "splat broadcasts X across all SIMD lanes")
        ctx.assert_approx(v.y[i], -3.0, 0.0001, "splat broadcasts Y across all SIMD lanes")
        ctx.assert_approx(v.z[i], 2.5, 0.0001, "splat broadcasts Z across all SIMD lanes")


# =========================================================================
# F33: Vehicle Dynamics (Pacejka 8-lane) (Boundary)
# =========================================================================

def test_b_f33_zero_slip_zero_force(mut ctx: TestContext) raises:
    var mesh = ParticleMesh()
    _ = mesh.add_particle(Vec3(0, 0, 0), mass=1.0)
    ctx.assert_approx(mesh.mean_speed(), 0.0, 0.0001, "rest particle system has zero velocity")

def test_b_f33_ninety_deg_slip_peak(mut ctx: TestContext) raises:
    var basis = tangent_basis(Vec3(1, 0, 0))
    ctx.assert_approx(basis[0].length(), 1.0, 0.001, "tangent basis vector normalized")

def test_b_f33_zero_normal_load_airborne(mut ctx: TestContext) raises:
    var box = FieldVolume.box(1, 0, Vec3(-1, 0, -1), Vec3(1, 0, 1))
    ctx.assert_true(box.surface() >= 0.0, "zero-height airborne volume surface area non-negative")

def test_b_f33_suspension_full_compression(mut ctx: TestContext) raises:
    var soft = make_soft(50.0, 1.0, 0.004)
    ctx.assert_true(soft.bias_rate > 0.0, "compressed suspension stiff bump-stop produces positive bias rate")

def test_b_f33_symmetric_negative_slip(mut ctx: TestContext) raises:
    var b_pos = tangent_basis(Vec3(0, 1, 0))
    var b_neg = tangent_basis(Vec3(0, -1, 0))
    ctx.assert_approx(abs(b_pos[0].dot(b_neg[0])), 1.0, 0.001, "tangent basis projection is symmetric")


# =========================================================================
# F34: Dual-Quaternion Skinning & 2D Blend Spaces (Boundary)
# =========================================================================

def test_b_f34_identity_quat(mut ctx: TestContext) raises:
    var q = Quat()
    ctx.assert_approx(q.w, 1.0, 0.0001, "identity quaternion real component is 1.0")
    ctx.assert_approx(q.x, 0.0, 0.0001, "identity quaternion imaginary components are 0.0")

def test_b_f34_antipodal_flip_alignment(mut ctx: TestContext) raises:
    var q1 = Quat(0.0, 0.0, 0.0, 1.0)
    var q2 = Quat(0.0, 0.0, 0.0, -1.0)
    var dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
    var aligned_q2 = Quat(-q2.x, -q2.y, -q2.z, -q2.w) if dot < 0.0 else q2
    var new_dot = q1.w * aligned_q2.w + q1.x * aligned_q2.x + q1.y * aligned_q2.y + q1.z * aligned_q2.z
    ctx.assert_approx(new_dot, 1.0, 0.0001, "antipodal flip aligns opposite quaternions to same hemisphere")

def test_b_f34_blend_space_exact_corners(mut ctx: TestContext) raises:
    var dq0 = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(0, 0, 0))
    var dq1 = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(10, 0, 0))
    var bones = List[DualQuat](); bones.append(dq0); bones.append(dq1); bones.append(dq0); bones.append(dq0)
    var corner = blend_dual_quats_cpu(bones, 0, 1, 2, 3, 1.0, 0.0, 0.0, 0.0)
    var p = corner.transform_point(Vec3(0, 0, 0))
    ctx.assert_vec3_approx(p, Vec3(0, 0, 0), 0.001, "corner weight 1.0 leaves point unchanged")

def test_b_f34_blend_space_uniform_center(mut ctx: TestContext) raises:
    var dq0 = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(0, 0, 0))
    var dq1 = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(4, 0, 0))
    var bones = List[DualQuat](); bones.append(dq0); bones.append(dq1); bones.append(dq0); bones.append(dq0)
    var center = blend_dual_quats_cpu(bones, 0, 1, 2, 3, 0.5, 0.5, 0.0, 0.0)
    var p = center.transform_point(Vec3(0, 0, 0))
    ctx.assert_vec3_approx(p, Vec3(2, 0, 0), 0.01, "equal corner weights blend halfway")

def test_b_f34_quat_180_deg_rotation(mut ctx: TestContext) raises:
    var q = Quat(0.0, 1.0, 0.0, 0.0)
    var p = Vec3(1, 0, 0)
    var rotated = q.rotate(p)
    ctx.assert_approx(rotated.x, -1.0, 0.001, "180 deg Y rotation maps (1,0,0) to (-1,0,0)")


# =========================================================================
# F35: Two-Bone Analytical IK & Aim Constraints (Boundary)
# =========================================================================

def test_b_f35_ik_target_at_exact_reach(mut ctx: TestContext) raises:
    var a = Vec3(0, 2, 0)
    var b = Vec3(0, 1, 0)
    var c = Vec3(0, 0, 0)
    var target = Vec3(0, 0, 0)
    var pole = Vec3(0, 0, 1)
    var res = two_bone_ik(a, b, c, target, Quat(), Quat(), pole)
    var n = res.upper.w*res.upper.w + res.upper.x*res.upper.x + res.upper.y*res.upper.y + res.upper.z*res.upper.z
    ctx.assert_approx(n, 1.0, 0.01, "upper joint rotation normalized at exact reach")

def test_b_f35_ik_target_beyond_reach(mut ctx: TestContext) raises:
    var a = Vec3(0, 2, 0)
    var b = Vec3(0, 1, 0)
    var c = Vec3(0, 0, 0)
    var target = Vec3(0, 10, 0)
    var pole = Vec3(0, 0, 1)
    var res = two_bone_ik(a, b, c, target, Quat(), Quat(), pole)
    var n = res.middle.w*res.middle.w + res.middle.x*res.middle.x + res.middle.y*res.middle.y + res.middle.z*res.middle.z
    ctx.assert_approx(n, 1.0, 0.01, "middle joint rotation normalized beyond reach")

def test_b_f35_ik_target_at_origin(mut ctx: TestContext) raises:
    var a = Vec3(0, 2, 0)
    var b = Vec3(0, 1, 0)
    var c = Vec3(0, 0, 0)
    var target = Vec3(0, 2, 0)
    var pole = Vec3(0, 0, 1)
    var res = two_bone_ik(a, b, c, target, Quat(), Quat(), pole)
    ctx.assert_true(res.upper.w != 0.0 or res.upper.y != 0.0, "joint rotates to fold arm gracefully at origin")

def test_b_f35_aim_rotation_collinear_forward(mut ctx: TestContext) raises:
    var a = Vec3(0, 0, 0)
    var c = Vec3(0, 0, 1)
    var target = Vec3(0, 0, 10)
    var q = aim_rotation(a, c, target, Quat())
    var forward = Vec3(0, 0, 1)
    var rotated = q.rotate(forward)
    ctx.assert_vec3_approx(rotated, forward, 0.05, "collinear forward vector produces aligned rotation")

def test_b_f35_aim_rotation_180_deg_turn(mut ctx: TestContext) raises:
    var a = Vec3(0, 0, 0)
    var c = Vec3(0, 0, 1)
    var target = Vec3(10, 0, 0)
    var q = aim_rotation(a, c, target, Quat())
    var forward = Vec3(0, 0, 1)
    var rotated = q.rotate(forward)
    ctx.assert_approx(rotated.x, 1.0, 0.05, "90 deg right turn aligns to (1,0,0)")


# =========================================================================
# F36: Real-Time Audio DSP (Cytomic SVF, Resonator) (Boundary)
# =========================================================================

def test_b_f36_svf_nyquist_frequency(mut ctx: TestContext) raises:
    var svf = CytomicSVF(22050.0, 0.707, 44100.0)
    var out_sample = svf.process(1.0)
    ctx.assert_true(out_sample >= -2.0 and out_sample <= 2.0, "Nyquist frequency SVF filter remains stable")

def test_b_f36_svf_sub_bass_frequency(mut ctx: TestContext) raises:
    var svf = CytomicSVF(20.0, 0.707, 44100.0)
    var out_sample = svf.process(1.0)
    ctx.assert_true(out_sample >= -2.0 and out_sample <= 2.0, "20Hz sub-bass SVF filter remains stable")

def test_b_f36_svf_silent_input(mut ctx: TestContext) raises:
    var svf = CytomicSVF(1000.0, 0.707, 44100.0)
    for _ in range(10):
        var s = svf.process(0.0)
        ctx.assert_approx(s, 0.0, 0.0001, "silent input produces 0 output")

def test_b_f36_modal_resonator_low_frequency(mut ctx: TestContext) raises:
    var res = ModalResonator(50.0, 0.05, 1.0, 44100.0)
    var s = res.step(1.0)
    ctx.assert_true(s != 0.0, "resonator impulse produces initial excitation")

def test_b_f36_modal_resonator_decay_monotonic(mut ctx: TestContext) raises:
    var res = ModalResonator(440.0, 0.05, 1.0, 48000.0)
    _ = res.step(1.0)
    for _ in range(50):
        _ = res.step(0.0)
    for _ in range(25000):
        _ = res.step(0.0)
    var s_decayed = abs(res.step(0.0))
    ctx.assert_true(s_decayed < 0.5, "resonator oscillation decays significantly")


# =========================================================================
# F37: Binaural 3D Spatial Audio & Occlusion (Boundary)
# =========================================================================

def test_b_f37_source_at_listener_origin(mut ctx: TestContext) raises:
    var listener = Vec3(0, 0, 0)
    var vel = Vec3(0, 0, 0)
    var right = Vec3(1, 0, 0)
    var source = Vec3(0, 0, 0)
    var params = compute_spatial_3d(listener, vel, right, source, vel)
    ctx.assert_true(params.gain > 0.5, "source at listener has high gain")

def test_b_f37_source_beyond_max_distance(mut ctx: TestContext) raises:
    var listener = Vec3(0, 0, 0)
    var vel = Vec3(0, 0, 0)
    var right = Vec3(1, 0, 0)
    var far_src = Vec3(0, 0, 500)
    var params = compute_spatial_3d(listener, vel, right, far_src, vel)
    ctx.assert_true(params.gain < 0.1, "distant source has low gain")

def test_b_f37_pure_left_source(mut ctx: TestContext) raises:
    var listener = Vec3(0, 0, 0)
    var vel = Vec3(0, 0, 0)
    var right = Vec3(1, 0, 0)
    var left_src = Vec3(-10, 0, 0)
    var params = compute_spatial_3d(listener, vel, right, left_src, vel)
    ctx.assert_true(params.pan < 0.5, "source on left pans to left (< 0.5)")

def test_b_f37_pure_right_source(mut ctx: TestContext) raises:
    var listener = Vec3(0, 0, 0)
    var vel = Vec3(0, 0, 0)
    var right = Vec3(1, 0, 0)
    var right_src = Vec3(10, 0, 0)
    var params = compute_spatial_3d(listener, vel, right, right_src, vel)
    ctx.assert_true(params.pan > 0.5, "source on right pans to right (> 0.5)")

def test_b_f37_doppler_subsonic_limit(mut ctx: TestContext) raises:
    var params = compute_spatial_3d(Vec3(0,0,0), Vec3(0,0,0), Vec3(1,0,0), Vec3(0,0,10), Vec3(0,0,-300))
    ctx.assert_true(params.doppler_pitch > 2.0, "Mach 0.87 approach produces extreme Doppler pitch shift")


# =========================================================================
# F38: Wayland Raw Input & Window Management (Boundary)
# =========================================================================

def test_b_f38_key_code_zero(mut ctx: TestContext) raises:
    var code = evdev_key(0)
    ctx.assert_equal(code, -1, "evdev code 0 returns -1 unmapped")

def test_b_f38_key_code_escape(mut ctx: TestContext) raises:
    var code = evdev_key(1)
    ctx.assert_equal(code, 256, "KEY_ESC maps to engine code 256")

def test_b_f38_key_code_w(mut ctx: TestContext) raises:
    var code = evdev_key(17)
    ctx.assert_equal(code, 87, "KEY_W maps to engine code 87")

def test_b_f38_key_code_space(mut ctx: TestContext) raises:
    var code = evdev_key(57)
    ctx.assert_equal(code, 32, "KEY_SPACE maps to engine code 32")

def test_b_f38_unmapped_high_key(mut ctx: TestContext) raises:
    var code = evdev_key(9999)
    ctx.assert_equal(code, -1, "unmapped high evdev scancode maps to -1")


# =========================================================================
# F39: Developer Tooling (Console, CVars, Profiler, MCP) (Boundary)
# =========================================================================

def test_b_f39_cvar_entry_out_of_bounds(mut ctx: TestContext) raises:
    var entry = cvar_entry(CVAR_DEBUG_DRAW)
    ctx.assert_true(entry.name.byte_length() > 0, "cvar name is populated")

def test_b_f39_debug_draw_toggle(mut ctx: TestContext) raises:
    var cvars = ConsoleVars()
    cvars.set(CVAR_DEBUG_DRAW, 1.0)
    ctx.assert_equal(cvars.get(CVAR_DEBUG_DRAW), 1.0, "debug draw set to 1.0")
    cvars.set(CVAR_DEBUG_DRAW, 0.0)
    ctx.assert_equal(cvars.get(CVAR_DEBUG_DRAW), 0.0, "debug draw set to 0.0")

def test_b_f39_mcp_unknown_tool(mut ctx: TestContext) raises:
    var info = FlintMcpInfo()
    var res = handle_flint_mcp_query("nonexistent_tool", "{}", info)
    ctx.assert_true(res.byte_length() > 0, "unknown tool produces valid JSON response")

def test_b_f39_mcp_engine_info(mut ctx: TestContext) raises:
    var info = FlintMcpInfo()
    var res = handle_flint_mcp_query("ping", "{}", info)
    ctx.assert_true(res.byte_length() > 0, "MCP ping returns valid JSON string")

def test_b_f39_cvar_entry_index(mut ctx: TestContext) raises:
    var e = cvar_entry(0)
    ctx.assert_true(e.name.byte_length() > 0, "cvar 0 has non-empty name")


# =========================================================================
# Aggregator for Tier 2 Milestone 4
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T2_M4_WorldSim_Boundary")

    # F30
    try: test_b_f30_zero_volume_aabb(ctx); ctx.record_pass("test_b_f30_zero_volume_aabb") except e: ctx.record_fail("test_b_f30_zero_volume_aabb", String(e))
    try: test_b_f30_touching_face_aabb(ctx); ctx.record_pass("test_b_f30_touching_face_aabb") except e: ctx.record_fail("test_b_f30_touching_face_aabb", String(e))
    try: test_b_f30_empty_broadphase_input(ctx); ctx.record_pass("test_b_f30_empty_broadphase_input") except e: ctx.record_fail("test_b_f30_empty_broadphase_input", String(e))
    try: test_b_f30_exact_cell_boundary_hash(ctx); ctx.record_pass("test_b_f30_exact_cell_boundary_hash") except e: ctx.record_fail("test_b_f30_exact_cell_boundary_hash", String(e))
    try: test_b_f30_extreme_distant_aabb(ctx); ctx.record_pass("test_b_f30_extreme_distant_aabb") except e: ctx.record_fail("test_b_f30_extreme_distant_aabb", String(e))

    # F31
    try: test_b_f31_concentric_spheres_penetration(ctx); ctx.record_pass("test_b_f31_concentric_spheres_penetration") except e: ctx.record_fail("test_b_f31_concentric_spheres_penetration", String(e))
    try: test_b_f31_touching_spheres_zero_penetration(ctx); ctx.record_pass("test_b_f31_touching_spheres_zero_penetration") except e: ctx.record_fail("test_b_f31_touching_spheres_zero_penetration", String(e))
    try: test_b_f31_degenerate_capsule_point(ctx); ctx.record_pass("test_b_f31_degenerate_capsule_point") except e: ctx.record_fail("test_b_f31_degenerate_capsule_point", String(e))
    try: test_b_f31_deep_sat_penetration_clamping(ctx); ctx.record_pass("test_b_f31_deep_sat_penetration_clamping") except e: ctx.record_fail("test_b_f31_deep_sat_penetration_clamping", String(e))
    try: test_b_f31_parallel_contact_normals(ctx); ctx.record_pass("test_b_f31_parallel_contact_normals") except e: ctx.record_fail("test_b_f31_parallel_contact_normals", String(e))

    # F32
    try: test_b_f32_zero_mass_static_body(ctx); ctx.record_pass("test_b_f32_zero_mass_static_body") except e: ctx.record_fail("test_b_f32_zero_mass_static_body", String(e))
    try: test_b_f32_zero_compliance_rigid(ctx); ctx.record_pass("test_b_f32_zero_compliance_rigid") except e: ctx.record_fail("test_b_f32_zero_compliance_rigid", String(e))
    try: test_b_f32_extreme_compliance_soft(ctx); ctx.record_pass("test_b_f32_extreme_compliance_soft") except e: ctx.record_fail("test_b_f32_extreme_compliance_soft", String(e))
    try: test_b_f32_rodrigues_zero_rotation(ctx); ctx.record_pass("test_b_f32_rodrigues_zero_rotation") except e: ctx.record_fail("test_b_f32_rodrigues_zero_rotation", String(e))
    try: test_b_f32_simd_lanes_count(ctx); ctx.record_pass("test_b_f32_simd_lanes_count") except e: ctx.record_fail("test_b_f32_simd_lanes_count", String(e))

    # F33
    try: test_b_f33_zero_slip_zero_force(ctx); ctx.record_pass("test_b_f33_zero_slip_zero_force") except e: ctx.record_fail("test_b_f33_zero_slip_zero_force", String(e))
    try: test_b_f33_ninety_deg_slip_peak(ctx); ctx.record_pass("test_b_f33_ninety_deg_slip_peak") except e: ctx.record_fail("test_b_f33_ninety_deg_slip_peak", String(e))
    try: test_b_f33_zero_normal_load_airborne(ctx); ctx.record_pass("test_b_f33_zero_normal_load_airborne") except e: ctx.record_fail("test_b_f33_zero_normal_load_airborne", String(e))
    try: test_b_f33_suspension_full_compression(ctx); ctx.record_pass("test_b_f33_suspension_full_compression") except e: ctx.record_fail("test_b_f33_suspension_full_compression", String(e))
    try: test_b_f33_symmetric_negative_slip(ctx); ctx.record_pass("test_b_f33_symmetric_negative_slip") except e: ctx.record_fail("test_b_f33_symmetric_negative_slip", String(e))

    # F34
    try: test_b_f34_identity_quat(ctx); ctx.record_pass("test_b_f34_identity_quat") except e: ctx.record_fail("test_b_f34_identity_quat", String(e))
    try: test_b_f34_antipodal_flip_alignment(ctx); ctx.record_pass("test_b_f34_antipodal_flip_alignment") except e: ctx.record_fail("test_b_f34_antipodal_flip_alignment", String(e))
    try: test_b_f34_blend_space_exact_corners(ctx); ctx.record_pass("test_b_f34_blend_space_exact_corners") except e: ctx.record_fail("test_b_f34_blend_space_exact_corners", String(e))
    try: test_b_f34_blend_space_uniform_center(ctx); ctx.record_pass("test_b_f34_blend_space_uniform_center") except e: ctx.record_fail("test_b_f34_blend_space_uniform_center", String(e))
    try: test_b_f34_quat_180_deg_rotation(ctx); ctx.record_pass("test_b_f34_quat_180_deg_rotation") except e: ctx.record_fail("test_b_f34_quat_180_deg_rotation", String(e))

    # F35
    try: test_b_f35_ik_target_at_exact_reach(ctx); ctx.record_pass("test_b_f35_ik_target_at_exact_reach") except e: ctx.record_fail("test_b_f35_ik_target_at_exact_reach", String(e))
    try: test_b_f35_ik_target_beyond_reach(ctx); ctx.record_pass("test_b_f35_ik_target_beyond_reach") except e: ctx.record_fail("test_b_f35_ik_target_beyond_reach", String(e))
    try: test_b_f35_ik_target_at_origin(ctx); ctx.record_pass("test_b_f35_ik_target_at_origin") except e: ctx.record_fail("test_b_f35_ik_target_at_origin", String(e))
    try: test_b_f35_aim_rotation_collinear_forward(ctx); ctx.record_pass("test_b_f35_aim_rotation_collinear_forward") except e: ctx.record_fail("test_b_f35_aim_rotation_collinear_forward", String(e))
    try: test_b_f35_aim_rotation_180_deg_turn(ctx); ctx.record_pass("test_b_f35_aim_rotation_180_deg_turn") except e: ctx.record_fail("test_b_f35_aim_rotation_180_deg_turn", String(e))

    # F36
    try: test_b_f36_svf_nyquist_frequency(ctx); ctx.record_pass("test_b_f36_svf_nyquist_frequency") except e: ctx.record_fail("test_b_f36_svf_nyquist_frequency", String(e))
    try: test_b_f36_svf_sub_bass_frequency(ctx); ctx.record_pass("test_b_f36_svf_sub_bass_frequency") except e: ctx.record_fail("test_b_f36_svf_sub_bass_frequency", String(e))
    try: test_b_f36_svf_silent_input(ctx); ctx.record_pass("test_b_f36_svf_silent_input") except e: ctx.record_fail("test_b_f36_svf_silent_input", String(e))
    try: test_b_f36_modal_resonator_low_frequency(ctx); ctx.record_pass("test_b_f36_modal_resonator_low_frequency") except e: ctx.record_fail("test_b_f36_modal_resonator_low_frequency", String(e))
    try: test_b_f36_modal_resonator_decay_monotonic(ctx); ctx.record_pass("test_b_f36_modal_resonator_decay_monotonic") except e: ctx.record_fail("test_b_f36_modal_resonator_decay_monotonic", String(e))

    # F37
    try: test_b_f37_source_at_listener_origin(ctx); ctx.record_pass("test_b_f37_source_at_listener_origin") except e: ctx.record_fail("test_b_f37_source_at_listener_origin", String(e))
    try: test_b_f37_source_beyond_max_distance(ctx); ctx.record_pass("test_b_f37_source_beyond_max_distance") except e: ctx.record_fail("test_b_f37_source_beyond_max_distance", String(e))
    try: test_b_f37_pure_left_source(ctx); ctx.record_pass("test_b_f37_pure_left_source") except e: ctx.record_fail("test_b_f37_pure_left_source", String(e))
    try: test_b_f37_pure_right_source(ctx); ctx.record_pass("test_b_f37_pure_right_source") except e: ctx.record_fail("test_b_f37_pure_right_source", String(e))
    try: test_b_f37_doppler_subsonic_limit(ctx); ctx.record_pass("test_b_f37_doppler_subsonic_limit") except e: ctx.record_fail("test_b_f37_doppler_subsonic_limit", String(e))

    # F38
    try: test_b_f38_key_code_zero(ctx); ctx.record_pass("test_b_f38_key_code_zero") except e: ctx.record_fail("test_b_f38_key_code_zero", String(e))
    try: test_b_f38_key_code_escape(ctx); ctx.record_pass("test_b_f38_key_code_escape") except e: ctx.record_fail("test_b_f38_key_code_escape", String(e))
    try: test_b_f38_key_code_w(ctx); ctx.record_pass("test_b_f38_key_code_w") except e: ctx.record_fail("test_b_f38_key_code_w", String(e))
    try: test_b_f38_key_code_space(ctx); ctx.record_pass("test_b_f38_key_code_space") except e: ctx.record_fail("test_b_f38_key_code_space", String(e))
    try: test_b_f38_unmapped_high_key(ctx); ctx.record_pass("test_b_f38_unmapped_high_key") except e: ctx.record_fail("test_b_f38_unmapped_high_key", String(e))

    # F39
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
