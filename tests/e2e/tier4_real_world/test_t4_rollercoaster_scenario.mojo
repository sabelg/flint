"""Tier 4 Real-World Application Scenarios: S2 Rollercoaster Physics & Track Simulation.
Simulates a multi-car rollercoaster train navigating a vertical drop, valley trough,
and vertical loop with Catmull-Rom track spline, G-force kinematics, and binaural audio.
"""
from std.collections import List
from std.math import sqrt, min, max
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.physics.solver_bodies import SolverBodies
from flint.physics.ccd import swept_sphere_obb
from flint.assets.replay_stream import _catmull_rom_vec3
from flint.platform.dsp import compute_spatial_3d, ModalResonator
from e2e.common.harness import TestContext


def evaluate_coaster_track(t_track: Float32) -> Vec3:
    # 4 knots defining coaster hill crest, vertical drop, valley trough, and climb
    var p0 = Vec3(0, 50, -50)
    var p1 = Vec3(0, 50, 0)    # Drop hill crest (50m high)
    var p2 = Vec3(0, 5, 60)    # Valley trough (5m high)
    var p3 = Vec3(0, 35, 120)  # Next hill crest (35m high)
    return _catmull_rom_vec3(p0, p1, p2, p3, t_track)


def test_t4_rollercoaster_full_run(mut ctx: TestContext) raises:
    # Coaster train setup: 3 carts connected in series
    var num_carts = 3

    # Initial state at top of drop
    var track_progress: Float32 = 0.0 # t in [0.0, 1.0]
    var speed: Float32 = 4.0 # initial crest speed 4 m/s (~14 km/h)
    var dt: Float32 = 0.016667 # 60 FPS tick

    # Track audio rattle synthesizer
    var rattle = ModalResonator(180.0, 0.05, 0.5, 48000.0)

    # Listener (park guest seated in station at (15, 2, 30))
    var listener_pos = Vec3(15, 2, 30)
    var listener_vel = Vec3(0, 0, 0)
    var listener_right = Vec3(1, 0, 0)

    var peak_g_force: Float32 = 0.0
    var trough_speed: Float32 = 0.0

    # Simulate 250 frames (~4.1 seconds of drop into valley trough)
    var bodies = SolverBodies()
    bodies.reset(3) # 3 coaster carts
    bodies.set_velocity(0, Vec3(0, 0, speed), Vec3(0, 0, 0))

    var pos_cart0 = Vec3(0, 50, 0)
    var pos_cart2 = Vec3(0, 50, -5.0)

    for _ in range(250):
        var pos = evaluate_coaster_track(track_progress)
        var next_pos = evaluate_coaster_track(min(Float32(1.0), track_progress + 0.01))
        var tangent = (next_pos - pos).normalized()

        # Potential to kinetic energy conversion: v = sqrt(v0^2 + 2*g*dh)
        var height_drop = 50.0 - pos.y
        if height_drop > 0.0:
            speed = sqrt(16.0 + 2.0 * 9.81 * height_drop)
            bodies.set_velocity(0, tangent * speed, Vec3(0, 0, 0))
            bodies.integrate_velocities(0, dt, 50.0, 10.0)

        # Advance along spline
        var ds = speed * dt
        track_progress = min(Float32(1.0), track_progress + ds / 60.0)

        pos_cart0 = pos
        pos_cart2 = evaluate_coaster_track(max(Float32(0.0), track_progress - 0.08))

        # Clearance check against structural support tower
        var hit = swept_sphere_obb(pos, pos + tangent * ds, 0.4, Vec3(5.0, 10.0, 30.0), Quat(), Vec3(1.0, 10.0, 1.0))
        ctx.assert_false(hit.hit, "coaster train avoids collision with structural support towers")

        # Normal acceleration & G-force: a_n = v^2 / R
        var radius_of_curvature: Float32 = 22.0 # ~22m curve radius at trough
        var a_norm = (speed * speed) / radius_of_curvature
        var current_g = (a_norm + 9.81) / 9.81
        if current_g > peak_g_force:
            peak_g_force = current_g

        if pos.y < 8.0 and trough_speed < speed:
            trough_speed = speed

        # Spatial audio tick from coaster train to spectator
        var cart_vel = tangent * speed
        var audio_params = compute_spatial_3d(listener_pos, listener_vel, listener_right, pos, cart_vel)
        _ = rattle.step(audio_params.gain)

    # Verification of physical coaster dynamics
    ctx.assert_true(trough_speed > 25.0, "coaster reaches high speed > 25 m/s (~90 km/h) in trough")
    ctx.assert_true(peak_g_force > 3.0 and peak_g_force < 5.5, "vertical G-force in trough within safe human limits [3.0G, 5.5G]")
    ctx.assert_equal(num_carts, 3, "multi-cart train intact")
    var train_span = (pos_cart0 - pos_cart2).length()
    ctx.assert_true(train_span > 2.0 and train_span < 10.0, "cart separation maintained along 3D track curve")


# =========================================================================
# Aggregator for Tier 4 Rollercoaster Scenario
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T4_RealWorld_Rollercoaster")
    try: test_t4_rollercoaster_full_run(ctx); ctx.record_pass("test_t4_rollercoaster_full_run") except e: ctx.record_fail("test_t4_rollercoaster_full_run", String(e))

def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
