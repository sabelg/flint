"""Tier 4 Real-World Application Scenarios: S5 Character Combat & Dynamic Ragdoll.
Simulates character combat interaction: hierarchical skeletal rig, two-bone IK weapon aim,
foot placement onto dynamic terrain, XPBD impulse solver response, and impact audio resonance.
"""
from std.collections import List
from std.math import sqrt, sin, cos
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.ik import two_bone_ik, aim_rotation
from flint.physics.solver_bodies import SolverBodies, V8, F8, LANES
from flint.physics.solver import rodrigues_displacement
from flint.platform.dsp import compute_spatial_3d, ModalResonator
from e2e.common.harness import TestContext


def quat_norm_sq(q: Quat) -> Float32:
    return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w


def test_t4_character_combat_full_sequence(mut ctx: TestContext) raises:
    # 1. Character skeletal posture
    var pelvis = Vec3(0, 1.0, 0)
    var spine = Vec3(0, 1.4, 0)
    ctx.assert_true(spine.y > pelvis.y, "spine is elevated above pelvis")
    var shoulder_r = Vec3(0.3, 1.4, 0)
    var elbow_r = Vec3(0.6, 1.2, 0.2)
    var wrist_r = Vec3(0.8, 1.0, 0.4)

    # 2. Weapon aim IK targeting enemy combatant at (10, 1.4, 10)
    var enemy_target = Vec3(10.0, 1.4, 10.0)
    var weapon_aim_q = aim_rotation(spine, shoulder_r, enemy_target, Quat())
    var arm_dir = (shoulder_r - spine).normalized()
    var aim_dir = weapon_aim_q.rotate(arm_dir)
    ctx.assert_true(aim_dir.x > 0.0 and aim_dir.z > 0.0, "weapon aim rotation directs toward enemy combatant")

    # Arm two-bone IK tracking weapon grip
    var pole = Vec3(0.5, 2.0, 0.5)
    var ik_arm = two_bone_ik(shoulder_r, elbow_r, wrist_r, wrist_r + aim_dir * 0.2, Quat(), Quat(), pole)
    ctx.assert_approx(quat_norm_sq(ik_arm.upper), 1.0, 0.001, "arm IK upper quaternion is unit normalized")
    ctx.assert_approx(quat_norm_sq(ik_arm.middle), 1.0, 0.001, "arm IK middle quaternion is unit normalized")
    ctx.assert_true(ik_arm.upper.w < 0.9999 or ik_arm.middle.w < 0.9999, "arm IK solves non-zero joint flexion for weapon grip")

    # 3. Dynamic foot placement on uneven terrain
    var hip_l = Vec3(-0.2, 0.9, 0)
    var knee_l = Vec3(-0.2, 0.5, 0.1)
    var ankle_l = Vec3(-0.2, 0.1, 0)
    var ground_terrain_pos = Vec3(-0.2, 0.08, 0) # 8cm rock outcrop
    var ik_leg = two_bone_ik(hip_l, knee_l, ankle_l, ground_terrain_pos, Quat(), Quat(), Vec3(-0.2, 0.5, 1.0))
    ctx.assert_approx(quat_norm_sq(ik_leg.upper), 1.0, 0.001, "leg IK upper quaternion is unit normalized")
    ctx.assert_approx(quat_norm_sq(ik_leg.middle), 1.0, 0.001, "leg IK middle quaternion is unit normalized")
    ctx.assert_true(ik_leg.upper.w < 0.9999 or ik_leg.middle.w < 0.9999, "leg IK firmly plants foot by flexing knee")

    # 4. Incoming projectile hit reaction: XPBD momentum impulse
    var projectile_vel = Vec3(-30.0, 0.0, -10.0) # 31.6 m/s bullet/arrow
    var projectile_mass: Float32 = 0.05 # 50g
    var momentum = projectile_vel * projectile_mass
    var char_mass: Float32 = 80.0 # 80kg
    var delta_v = momentum * (1.0 / char_mass)
    var bodies = SolverBodies()
    bodies.reset(2)
    bodies.set_velocity(1, delta_v, Vec3(0, 0.2, 0))
    bodies.integrate_velocities(0, 0.016, 50.0, 20.0)
    var char_v = bodies.velocity(1)
    ctx.assert_true(char_v.length() > 0.01, "projectile delivers knockback impulse to solver body")

    # Rodrigues angular displacement of chest joint under impact torque
    var torque_impulse = V8.splat(Vec3(0.0, 0.2, 0.0))
    var chest_offset = V8.splat(Vec3(0.0, 0.4, 0.1))
    var disp = rodrigues_displacement(torque_impulse, chest_offset)
    ctx.assert_approx(disp.y[0], 0.0, 0.05, "vertical displacement minimal under horizontal torque")

    # 5. Flesh/armor impact audio resonance
    var hit_resonator = ModalResonator(520.0, 0.04, 1.0, 48000.0) # metallic armor ding
    var sound_out = hit_resonator.step(1.0)
    ctx.assert_true(sound_out > 0.0, "impact triggers metallic armor resonance")

    # Spatial audio parameters to camera listener
    var cam_listener = Vec3(0, 1.5, -3.0) # 3rd person follow camera
    var cam_vel = Vec3(0, 0, 0)
    var cam_right = Vec3(1, 0, 0)
    var audio_params = compute_spatial_3d(cam_listener, cam_vel, cam_right, spine, delta_v)
    ctx.assert_true(audio_params.gain > 0.5, "impact sound loud and clear to 3rd person camera")


# =========================================================================
# Aggregator for Tier 4 Character Combat Scenario
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T4_RealWorld_CharacterCombat")
    try: test_t4_character_combat_full_sequence(ctx); ctx.record_pass("test_t4_character_combat_full_sequence") except e: ctx.record_fail("test_t4_character_combat_full_sequence", String(e))

def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
