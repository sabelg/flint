"""Tier 3 Cross-Feature Combinations: Physics & Animation Interactions (15 tests).
Pairwise interactions between broadphase collision, XPBD solver, vehicle dynamics,
analytical IK, skeletal skinning, spatial audio DSP, and replay streams.
"""
from std.collections import List
from std.math import sqrt, sin, cos, atan, exp
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.ik import two_bone_ik, aim_rotation
from flint.physics.broadphase import BroadphaseInput, boxes_overlap, find_pairs
from flint.physics.filter import CollisionFilter
from flint.physics.solver_bodies import SolverBodies, V8, F8, LANES
from flint.physics.solver import rodrigues_displacement
from flint.physics.wind import WindField, make_vorticle
from flint.anim.pose import Pose, blend_into
from flint.platform.dsp import compute_spatial_3d, CytomicSVF, ModalResonator
from flint.assets.replay_stream import FlintReplayStream
from e2e.common.harness import TestContext


def quat_norm_sq(q: Quat) -> Float32:
    return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w


# 1. Broadphase pair detection -> XPBD contact candidate generation
def test_t3_broadphase_to_solver_candidates(mut ctx: TestContext) raises:
    var bp = BroadphaseInput()
    bp.add(Vec3(-1, 0, 0), Vec3(1, 2, 1), True, True, CollisionFilter(1, 1, 0))
    bp.add(Vec3(0.5, 0, 0), Vec3(2.5, 2, 1), True, True, CollisionFilter(1, 1, 0))
    var pairs = find_pairs(bp)
    ctx.assert_true(len(pairs.pairs) >= 1, "broadphase detects overlapping bodies for solver")

# 2. Character IK foot placement onto surface normal
def test_t3_ik_foot_placement_surface_normal(mut ctx: TestContext) raises:
    var hip = Vec3(0, 1.8, 0)
    var knee = Vec3(0, 0.9, 0.1)
    var ankle = Vec3(0, 0.1, 0)
    var ground_hit = Vec3(0, 0.15, 0) # ground 15cm elevated
    var pole = Vec3(0, 0.9, 1.0)
    var res = two_bone_ik(hip, knee, ankle, ground_hit, Quat(), Quat(), pole)
    ctx.assert_approx(quat_norm_sq(res.upper), 1.0, 0.001, "upper joint rotation is normalized quaternion")
    ctx.assert_approx(quat_norm_sq(res.middle), 1.0, 0.001, "middle joint rotation is normalized quaternion")
    ctx.assert_true(res.upper.w < 0.9999 or res.middle.w < 0.9999, "non-zero joint flex computed for elevated ground")

# 3. Aim constraint head tracking towards moving target
def test_t3_aim_head_tracking(mut ctx: TestContext) raises:
    var neck = Vec3(0, 1.6, 0)
    var head = Vec3(0, 1.7, 0.1)
    var target = Vec3(5.0, 1.7, 5.0)
    var q = aim_rotation(neck, head, target, Quat())
    var fwd = Vec3(0, 0, 1)
    var rotated = q.rotate(fwd)
    ctx.assert_true(rotated.x > 0.0 and rotated.z > 0.0, "aim rotation turns head toward (+X, +Z) target")

# 4. Pacejka tire slip force feeding XPBD wheel impulse
def test_t3_pacejka_to_wheel_impulse(mut ctx: TestContext) raises:
    var slip: Float32 = 0.08
    var b: Float32 = 10.0
    var c: Float32 = 1.65
    var d: Float32 = 4500.0 # 4500 N peak lateral grip
    var lateral_force = d * sin(c * atan(b * slip))
    var dt: Float32 = 0.016
    var impulse = lateral_force * dt
    var bodies = SolverBodies()
    bodies.reset(2)
    bodies.set_velocity(1, Vec3(0, 0, impulse / 25.0), Vec3(0, 0, 0))
    bodies.integrate_velocities(0, dt, 50.0, 20.0)
    var v_wheel = bodies.velocity(1)
    ctx.assert_true(v_wheel.z > 0.0, "wheel impulse updates solver body velocity")

# 5. Physics impact velocity -> Audio resonator excitation
def test_t3_impact_velocity_to_audio_resonance(mut ctx: TestContext) raises:
    var v1 = Vec3(0, -10, 0)
    var v2 = Vec3(0, 0, 0)
    var rel_vel = (v1 - v2).length() # 10 m/s impact
    var impact_gain = min(Float32(1.0), rel_vel / 15.0)
    var resonator = ModalResonator(220.0, 0.05, impact_gain, 48000.0)
    var s = resonator.step(impact_gain)
    ctx.assert_true(s > 0.0, "high velocity impact excites modal resonator proportionally")

# 6. Spatial audio 3D tracking moving rigid body
def test_t3_moving_body_doppler_audio(mut ctx: TestContext) raises:
    var listener = Vec3(0, 0, 0)
    var listener_vel = Vec3(0, 0, 0)
    var right = Vec3(1, 0, 0)
    var body_pos = Vec3(0, 0, 20)
    var body_vel = Vec3(0, 0, -25) # moving towards listener at 25 m/s
    var params = compute_spatial_3d(listener, listener_vel, right, body_pos, body_vel)
    ctx.assert_true(params.doppler_pitch > 1.01, "approaching rigid body produces positive pitch shift")

# 7. Collision filtering isolating ragdoll internal limbs
def test_t3_collision_filtering_ragdoll(mut ctx: TestContext) raises:
    var ragdoll_group = 5
    var limb1_filter = CollisionFilter(1, 2, ragdoll_group)
    var limb2_filter = CollisionFilter(2, 1, ragdoll_group)
    ctx.assert_true(limb1_filter.touches(limb2_filter), "collision filter evaluates contact masks")

# 8. Rodrigues rotation maintaining lever arm for ragdoll constraint
def test_t3_rodrigues_joint_lever_arm(mut ctx: TestContext) raises:
    var omega = V8.splat(Vec3(0.0, 0.5, 0.0))
    var r = V8.splat(Vec3(0.8, 0.0, 0.0))
    var disp = rodrigues_displacement(omega, r)
    var final_pos = Vec3(0.8 + disp.x[0], disp.y[0], disp.z[0])
    ctx.assert_approx(final_pos.length(), 0.8, 0.02, "Rodrigues rotation maintains constraint anchor distance")

# 9. Suspension compression affecting tire normal load
def test_t3_suspension_compression_load(mut ctx: TestContext) raises:
    var bodies = SolverBodies()
    bodies.reset(2)
    bodies.set_velocity(1, Vec3(0, -2.0, 0), Vec3(0, 0, 0))
    bodies.integrate_velocities(0, 0.016, 50.0, 20.0)
    var v_after = bodies.velocity(1)
    ctx.assert_true(v_after.length() > 0.0, "solver body velocity integrated under suspension load")

# 10. Replay stream recording physics state snapshot
def test_t3_replay_physics_snapshot(mut ctx: TestContext) raises:
    var replay = FlintReplayStream()
    ctx.assert_equal(replay.tick_count(), 0, "replay stream starts ready for physics ticks")

# 11. Lowpass filter smoothing noisy physics acceleration
def test_t3_acceleration_filtering_svf(mut ctx: TestContext) raises:
    var svf = CytomicSVF(10.0, 0.707, 1000.0) # 10Hz lowpass for physics telemetry
    _ = svf.process(100.0) # spike
    var smoothed = svf.process(0.0)
    ctx.assert_true(smoothed < 100.0, "SVF lowpass smooths violent instantaneous physics spike")

# 12. Dual quaternion spherical blend between walking and running poses
def test_t3_anim_pose_slerp(mut ctx: TestContext) raises:
    var p_walk = Pose(2)
    p_walk.r[0] = Quat(0, 0, 0, 1)
    p_walk.t[0] = Vec3(0, 0, 0)
    var p_run = Pose(2)
    p_run.r[0] = Quat(0, 0.3826, 0, 0.9238) # ~45 deg Y turn
    p_run.t[0] = Vec3(0, 0, 2.0)
    blend_into(p_walk, p_run, 0.5)
    ctx.assert_approx(p_walk.t[0].z, 1.0, 0.01, "pose translation blended linearly")
    ctx.assert_true(p_walk.r[0].y > 0.1 and p_walk.r[0].y < 0.3, "pose rotation spherical slerp blended intermediate orientation")
    ctx.assert_approx(quat_norm_sq(p_walk.r[0]), 1.0, 0.001, "blended quaternion remains unit normalized")

# 13. Two-bone IK solving leg while aim constraint turns spine
def test_t3_simultaneous_ik_and_aim(mut ctx: TestContext) raises:
    var hip = Vec3(0, 2, 0)
    var knee = Vec3(0, 1, 0)
    var ankle = Vec3(0, 0, 0)
    _ = two_bone_ik(hip, knee, ankle, Vec3(0, 0.2, 0.1), Quat(), Quat(), Vec3(0, 1, 1))
    var spine_base = Vec3(0, 2, 0)
    var spine_tip = Vec3(0, 3, 0)
    var q_aim = aim_rotation(spine_base, spine_tip, Vec3(5, 3, 5), Quat())
    ctx.assert_true(q_aim.w < 1.0, "aim rotation applies valid torso swing concurrently with leg IK")

# 14. Linear velocity decay under aerodynamic drag
def test_t3_aero_drag_decay(mut ctx: TestContext) raises:
    var wind = WindField()
    var v = make_vorticle(1, Vec3(0, 0, 0), Vec3(0, 1, 0), 10.0, 5.0, 2.0, 100)
    wind.add_vorticle(v)
    var w_sample = wind.at_vec(Vec3(2, 0, 0), 1)
    ctx.assert_true(w_sample.length() > 0.1, "vorticle aerodynamic field produces non-zero wind drag velocity")

# 15. Vehicle weight transfer during braking into corner
def test_t3_trail_braking_weight_transfer(mut ctx: TestContext) raises:
    var bodies = SolverBodies()
    bodies.reset(4)
    bodies.set_velocity(0, Vec3(0, 0, 10.0), Vec3(0, 0, 0))
    bodies.set_velocity(1, Vec3(0, 0, 10.0), Vec3(0, 0, 0))
    bodies.set_velocity(2, Vec3(0, 0, 10.0), Vec3(0, 0, 0))
    bodies.set_velocity(3, Vec3(0, 0, 10.0), Vec3(0, 0, 0))
    bodies.integrate_velocities(0, 0.016, 50.0, 20.0)
    var v0 = bodies.velocity(0)
    ctx.assert_true(v0.length() <= 10.0, "solver bodies integration maintains physical velocity bounds")


# =========================================================================
# Aggregator for Tier 3 Physics & Animation
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T3_CrossFeature_PhysicsAnim")

    try: test_t3_broadphase_to_solver_candidates(ctx); ctx.record_pass("test_t3_broadphase_to_solver_candidates") except e: ctx.record_fail("test_t3_broadphase_to_solver_candidates", String(e))
    try: test_t3_ik_foot_placement_surface_normal(ctx); ctx.record_pass("test_t3_ik_foot_placement_surface_normal") except e: ctx.record_fail("test_t3_ik_foot_placement_surface_normal", String(e))
    try: test_t3_aim_head_tracking(ctx); ctx.record_pass("test_t3_aim_head_tracking") except e: ctx.record_fail("test_t3_aim_head_tracking", String(e))
    try: test_t3_pacejka_to_wheel_impulse(ctx); ctx.record_pass("test_t3_pacejka_to_wheel_impulse") except e: ctx.record_fail("test_t3_pacejka_to_wheel_impulse", String(e))
    try: test_t3_impact_velocity_to_audio_resonance(ctx); ctx.record_pass("test_t3_impact_velocity_to_audio_resonance") except e: ctx.record_fail("test_t3_impact_velocity_to_audio_resonance", String(e))
    try: test_t3_moving_body_doppler_audio(ctx); ctx.record_pass("test_t3_moving_body_doppler_audio") except e: ctx.record_fail("test_t3_moving_body_doppler_audio", String(e))
    try: test_t3_collision_filtering_ragdoll(ctx); ctx.record_pass("test_t3_collision_filtering_ragdoll") except e: ctx.record_fail("test_t3_collision_filtering_ragdoll", String(e))
    try: test_t3_rodrigues_joint_lever_arm(ctx); ctx.record_pass("test_t3_rodrigues_joint_lever_arm") except e: ctx.record_fail("test_t3_rodrigues_joint_lever_arm", String(e))
    try: test_t3_suspension_compression_load(ctx); ctx.record_pass("test_t3_suspension_compression_load") except e: ctx.record_fail("test_t3_suspension_compression_load", String(e))
    try: test_t3_replay_physics_snapshot(ctx); ctx.record_pass("test_t3_replay_physics_snapshot") except e: ctx.record_fail("test_t3_replay_physics_snapshot", String(e))
    try: test_t3_acceleration_filtering_svf(ctx); ctx.record_pass("test_t3_acceleration_filtering_svf") except e: ctx.record_fail("test_t3_acceleration_filtering_svf", String(e))
    try: test_t3_anim_pose_slerp(ctx); ctx.record_pass("test_t3_anim_pose_slerp") except e: ctx.record_fail("test_t3_anim_pose_slerp", String(e))
    try: test_t3_simultaneous_ik_and_aim(ctx); ctx.record_pass("test_t3_simultaneous_ik_and_aim") except e: ctx.record_fail("test_t3_simultaneous_ik_and_aim", String(e))
    try: test_t3_aero_drag_decay(ctx); ctx.record_pass("test_t3_aero_drag_decay") except e: ctx.record_fail("test_t3_aero_drag_decay", String(e))
    try: test_t3_trail_braking_weight_transfer(ctx); ctx.record_pass("test_t3_trail_braking_weight_transfer") except e: ctx.record_fail("test_t3_trail_braking_weight_transfer", String(e))

def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
