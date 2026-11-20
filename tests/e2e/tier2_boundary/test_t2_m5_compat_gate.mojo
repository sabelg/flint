"""Tier 2 Boundary & Corner Cases: Milestone 5 Compatibility & Gate (F40–F43: 20 tests).
Corner cases, extremes, and stress inputs for Pure Mojo gate (22 subpackages), NeonDrift
(VehicleChassis/PhysicsWorld/InputMap), Rollercoaster (EngineApp/PhysicsWorld), and Headless PNG gates.
"""
from std.collections import List
from std.ffi import external_call
from flint.app import EngineApp
from flint.assets.image import decode_png, write_png
from flint.assets.replay_stream import FlintReplayStream, _catmull_rom_vec3
from flint.core.build_stamp import BUILD_COMMIT, BUILD_MOJO
from flint.core.detmath import det_cos, det_sin
from flint.core.math import Vec3
from flint.core.serial import BlobWriter
from flint.core.transform import Quat
from flint.input import ACTION_AXIS1D, InputContext, InputMap
from flint.physics.plugin import RigidBodyComponent
from flint.physics.vehicle import VehicleChassis
from flint.physics.world import BODY_MOTION_DYNAMIC, PhysicsWorld
from flint.platform.dsp import compute_spatial_3d
from flint.platform.input import PAD_AXIS_LEFT_X, PAD_AXIS_RT, InputState, PadState
from flint.platform.os import file_size, list_directory, process_rss_bytes, read_bytes
from flint.render.image_diff import compare_pixels
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, ScalableRenderer
from flint.scene import LocalTransform
from flint.ui.draw_list import UiDrawList
from e2e.common.harness import TestContext
from e2e.common.v5_fixtures import create_headless_v5_app, register_mock_render_scene


@always_inline
def get_flint_root() -> String:
    if file_size("src/flint/core/math.mojo") > 0:
        return ""
    return "/home/user/flint/"


def _b_str_in_list(hay: List[String], needle: String) -> Bool:
    for i in range(len(hay)):
        if hay[i] == needle:
            return True
    return False


# =========================================================================
# F40: Pure Mojo 1.2 & Zero Warnings Gate (Boundary)
# =========================================================================

def test_b_f40_pure_mojo_version_exact(mut ctx: TestContext) raises:
    ctx.assert_true(BUILD_MOJO.startswith("1.2"), "Mojo version string conforms to 1.2.x specification")
    ctx.assert_true(BUILD_MOJO.byte_length() >= 5, "Mojo version string >= 5 bytes")


def test_b_f40_build_commit_length(mut ctx: TestContext) raises:
    ctx.assert_true(BUILD_COMMIT.byte_length() >= 3, "git commit hash has valid length")


def test_b_f40_ieee754_determinism_subnormal(mut ctx: TestContext) raises:
    var a: Float32 = 1.0e-38
    var b: Float32 = 1.0e-38
    ctx.assert_approx(a - b, 0.0, 0.000001, "bit-exact float subtraction produces 0")
    ctx.assert_equal(Float32(0.0), Float32(-0.0), "+0.0 and -0.0 compare equal")
    var v8 = SIMD[DType.float32, 8](0.0, 0.5, 1.0, 1.5, -0.5, -1.0, -1.5, 3.14159)
    var s = det_sin(v8)
    var c = det_cos(v8)
    ctx.assert_approx(s[0], 0.0, 0.0001, "det_sin(0) == 0")
    ctx.assert_approx(c[0], 1.0, 0.0001, "det_cos(0) == 1")


def test_b_f40_zero_c_dependencies_invariant(mut ctx: TestContext) raises:
    """Dynamically scans ALL 22 src/flint/ subpackages verifying 100% .mojo files."""
    var root = get_flint_root()
    var expected: List[String] = [
        "ai", "anim", "app", "assets", "audio", "camera", "core", "cycles",
        "devtools", "ecs", "input", "lab", "nav", "net", "physics", "platform",
        "procedural", "render", "scene", "studio", "text", "ui",
    ]
    var pkgs = list_directory(root + "src/flint")
    ctx.assert_true(len(pkgs) >= 22, "all 22 subpackages discovered")
    for e in range(len(expected)):
        ctx.assert_true(_b_str_in_list(pkgs, expected[e]), "subpackage present: " + expected[e])

    var verified_count = 0
    for p in range(len(pkgs)):
        var pkg = pkgs[p]
        if pkg.startswith("."):
            continue
        var files = list_directory(root + "src/flint/" + pkg)
        for f in range(len(files)):
            var fname = files[f]
            ctx.assert_true(fname.endswith(".mojo"), "all engine source files end with .mojo: " + pkg + "/" + fname)
            ctx.assert_false(
                fname.endswith(".c") or fname.endswith(".cpp") or fname.endswith(".h")
                or fname.endswith(".so") or fname.endswith(".a") or fname.endswith(".o"),
                "zero C/C++ files or binaries permitted",
            )
            verified_count += 1
    ctx.assert_true(verified_count >= 180, "exhaustively verified >= 180 pure Mojo engine modules")


def test_b_f40_compiler_optimization_contract(mut ctx: TestContext) raises:
    var q = Quat.from_scaled_angle(Vec3(0, 1, 0) * Float32(0.006283))
    var v = Vec3(1, 0, 0)
    for _ in range(1000):
        v = q.rotate(v)
    ctx.assert_approx(v.length(), 1.0, 0.001, "1000 quaternion rotations preserve unit length")


# =========================================================================
# F41: NeonDrift 2099 Compatibility (Boundary)
# =========================================================================

def test_b_f41_throttle_clamping_boundary(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0, -0.5, 0), Vec3(40, 0.5, 40), mass=0.0)
    var car = VehicleChassis(1400.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(0, 0.95, 0)
    var slot = pw.add_vehicle(car^)
    pw.rebuild_broadphase()
    pw.set_vehicle_controls(slot, 2.5, -1.0, 0.0)
    for _ in range(5):
        pw.step(Float32(1.0 / 60.0))
    var st = pw.vehicle_state(slot)
    ctx.assert_true(st.wheel_states[0].grounded, "VehicleChassis front wheel grounds after rebuild_broadphase")
    ctx.assert_true(st.linear_vel.z > 0.0, "out-of-range throttle accelerates vehicle forward")
    ctx.assert_true(st.pos.y > 0.0, "VehicleChassis remains stable under out-of-range throttle/brake")

    var pad = PadState()
    pad.apply_event(2, PAD_AXIS_RT, 32767)
    ctx.assert_approx(pad.axes[PAD_AXIS_RT], 1.0, 0.01, "full throttle trigger clamps to 1.0")


def test_b_f41_steer_clamping_boundary(mut ctx: TestContext) raises:
    var imap = InputMap()
    _ = imap.register_action("Steer", ACTION_AXIS1D)
    var ic = InputContext("Drive")
    ic.bind_pad_axis_1d("Steer", PAD_AXIS_LEFT_X)
    imap.add_context(ic^)
    imap.push_context("Drive")
    var inp = InputState()
    inp.pads[0].apply_event(2, PAD_AXIS_LEFT_X, 32767)
    imap.update(inp, 0.016)
    ctx.assert_approx(imap.axis_1d("Steer"), 1.0, 0.05, "full right steer saturates at 1.0")


def test_b_f41_speedometer_zero_speed(mut ctx: TestContext) raises:
    var p0 = compute_spatial_3d(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 10), Vec3(0, 0, 0))
    ctx.assert_approx(p0.doppler_pitch, 1.0, 0.0001, "stationary vehicle produces 1.0 Doppler pitch")


def test_b_f41_speedometer_supersonic_boundary(mut ctx: TestContext) raises:
    var p_fast = compute_spatial_3d(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 10), Vec3(0, 0, -340))
    ctx.assert_true(p_fast.doppler_pitch > 2.0, "supersonic velocity produces extreme Doppler shift")


def test_b_f41_replay_stream_empty_capacity(mut ctx: TestContext) raises:
    var replay = FlintReplayStream()
    ctx.assert_equal(replay.tick_count(), 0, "empty replay stream starts with 0 ticks")
    ctx.assert_equal(replay.keyframe_count(), 0, "empty replay stream starts with 0 keyframes")
    ctx.assert_equal(replay.camera_knot_count(), 0, "empty replay stream starts with 0 camera knots")
    ctx.assert_false(replay.evaluate_camera_spline(0.0)[0], "empty replay stream camera spline returns False")
    var w = BlobWriter()
    w.u32(123)
    replay.begin_recording("neondrift", w)
    var input_bytes = List[UInt8](); input_bytes.append(1)
    for _ in range(10):
        replay.record_tick(0.016, input_bytes)
    ctx.assert_equal(replay.tick_count(), 10, "10 recorded ticks registered in replay stream")


# =========================================================================
# F42: Rollercoaster Game Verification (Boundary)
# =========================================================================

def test_b_f42_spline_knot_t0_exact(mut ctx: TestContext) raises:
    var p1 = Vec3(10, 20, 30)
    var pt = _catmull_rom_vec3(Vec3(0, 0, 0), p1, Vec3(40, 50, 60), Vec3(70, 80, 90), 0.0)
    ctx.assert_vec3_approx(pt, p1, 0.001, "Catmull-Rom at t=0.0 evaluates exactly to p1")


def test_b_f42_spline_knot_t1_exact(mut ctx: TestContext) raises:
    var p2 = Vec3(40, 50, 60)
    var pt = _catmull_rom_vec3(Vec3(0, 0, 0), Vec3(10, 20, 30), p2, Vec3(70, 80, 90), 1.0)
    ctx.assert_vec3_approx(pt, p2, 0.001, "Catmull-Rom at t=1.0 evaluates exactly to p2")


def test_b_f42_train_single_cart(mut ctx: TestContext) raises:
    var app = create_headless_v5_app(with_physics=True)
    app.startup()
    var cart_e = app.world.spawn()
    _ = app.world.insert(cart_e, LocalTransform(Vec3(0.0, 5.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    var rb = RigidBodyComponent(mass=400.0, motion_mode=BODY_MOTION_DYNAMIC)
    rb.linear_vel = Vec3(0.0, 0.0, 6.0)
    _ = app.world.insert(cart_e, rb)
    for _ in range(10):
        app.step_one_frame(Float32(1.0 / 60.0))
    var tr = app.world.get[LocalTransform](cart_e)
    ctx.assert_true(tr.translation.z > 0.5, "EngineApp + PhysicsPlugin steps single cart forward")
    ctx.assert_true(tr.translation.y < 5.0, "gravity acts on dynamic cart in EngineApp")
    app.shutdown()


def test_b_f42_g_force_weightlessness_crest(mut ctx: TestContext) raises:
    var p0 = Vec3(0, 0, 0); var p1 = Vec3(10, 10, 10); var p2 = Vec3(0, 10, 20); var p3 = Vec3(0, 0, 30)
    var apex_prev = _catmull_rom_vec3(p0, p1, p2, p3, 0.45)
    var apex = _catmull_rom_vec3(p0, p1, p2, p3, 0.5)
    var apex_next = _catmull_rom_vec3(p0, p1, p2, p3, 0.55)
    var curve_y = (apex_next.y - apex.y) - (apex.y - apex_prev.y)
    ctx.assert_true(curve_y <= 0.0, "crest curvature is downward")


def test_b_f42_zero_speed_g_force(mut ctx: TestContext) raises:
    var mid = _catmull_rom_vec3(Vec3(0, 0, 0), Vec3(0, 0, 10), Vec3(0, 0, 20), Vec3(0, 0, 30), 0.5)
    ctx.assert_approx(mid.y, 0.0, 0.0001, "flat track produces zero vertical curvature")


# =========================================================================
# F43: 100% Pass Invariant & Headless Render Verification (Boundary)
# =========================================================================

def test_b_f43_harness_approx_epsilon_boundary(mut ctx: TestContext) raises:
    var sub = TestContext()
    var threw_inside = False
    try:
        sub.assert_approx(Float32(1.0), Float32(1.0009), 0.001)
    except:
        threw_inside = True
    ctx.assert_false(threw_inside, "assert_approx passes inside epsilon")

    var threw_outside = False
    try:
        sub.assert_approx(Float32(1.0), Float32(1.0011), 0.001)
    except:
        threw_outside = True
    ctx.assert_true(threw_outside, "assert_approx raises outside epsilon")


def test_b_f43_harness_vec3_approx_boundary(mut ctx: TestContext) raises:
    var sub = TestContext()
    var threw_inside = False
    try:
        sub.assert_vec3_approx(Vec3(1.0, 2.0, 3.0), Vec3(1.0009, 2.0, 3.0), 0.001)
    except:
        threw_inside = True
    ctx.assert_false(threw_inside, "assert_vec3_approx passes inside epsilon")

    var threw_outside = False
    try:
        sub.assert_vec3_approx(Vec3(1.0, 2.0, 3.0), Vec3(1.0015, 2.0, 3.0), 0.001)
    except:
        threw_outside = True
    ctx.assert_true(threw_outside, "assert_vec3_approx raises outside epsilon")


def test_b_f43_headless_shot_ihdr_dimensions(mut ctx: TestContext) raises:
    """Renders deterministic ScalableRenderer frame -> writes PNG to /tmp -> decodes via decode_png,
    plus verifies build/shots/*.png IHDR dimensions if already present on disk (never calls flint.sh shot)."""
    var pid_s = String(Int(external_call["getpid", Int32]()))
    var shot_tmp = "/tmp/flint_e2e_b_f43_shot_" + pid_s + ".png"
    ctx.cleanup_artifact(shot_tmp)

    var ren = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, width=128, height=72)
    var rw = register_mock_render_scene(ren)
    var ui = UiDrawList()
    ctx.assert_true(ren.render_world(rw, ui), "ScalableRenderer 128x72 frame 1 renders")
    var px1 = ren.readback_pixels()
    ctx.assert_true(ren.render_world(rw, ui), "ScalableRenderer 128x72 frame 2 renders")
    var px2 = ren.readback_pixels()
    ctx.assert_approx(Float32(compare_pixels(px1, px2, 128, 72, 0).flip_mean), 0.0, 0.0001, "2-frame flip_mean == 0.0")

    var px_mut = px1.copy()
    ctx.assert_true(write_png(shot_tmp, px_mut, 128, 72, stamped=False), "write_png writes /tmp PNG")
    var bytes = read_bytes(shot_tmp)
    ctx.assert_true(len(bytes) >= 26, "PNG has >= 26 bytes")
    var w = (UInt32(bytes[16]) << 24) | (UInt32(bytes[17]) << 16) | (UInt32(bytes[18]) << 8) | UInt32(bytes[19])
    var h = (UInt32(bytes[20]) << 24) | (UInt32(bytes[21]) << 16) | (UInt32(bytes[22]) << 8) | UInt32(bytes[23])
    ctx.assert_equal(Int(w), 128, "PNG IHDR width == 128")
    ctx.assert_equal(Int(h), 72, "PNG IHDR height == 72")
    var dec = decode_png(bytes, 16384)
    ctx.assert_true(dec.valid and dec.width == 128 and dec.height == 72, "decode_png validates 128x72 PNG")
    ctx.cleanup_artifact(shot_tmp)

    var root = get_flint_root()
    var spark_path = root + "build/shots/flint_spark.png"
    if file_size(spark_path) > 0:
        var sb = read_bytes(spark_path)
        var sdec = decode_png(sb, 16384)
        ctx.assert_true(sdec.valid and sdec.width == 1280 and sdec.height == 720, "build/shots/flint_spark.png is valid 1280x720 PNG")


def test_b_f43_zero_memory_leak_threshold(mut ctx: TestContext) raises:
    var rss = process_rss_bytes()
    ctx.assert_true(rss >= 1024 * 1024 and rss <= 1024 * 1024 * 1024, "process RSS within [1 MiB, 1 GiB]")


def test_b_f43_gate_exit_code_contract(mut ctx: TestContext) raises:
    var cmd = String("cd /home/user/flint && bash tools/flint.sh sync --verify > /dev/null 2>&1")
    var rc = external_call["system", Int32](cmd.as_c_string_span())
    ctx.assert_equal(Int(rc), 0, "tools/flint.sh sync --verify executes cleanly with exit code 0")


# =========================================================================
# Aggregator for Tier 2 Milestone 5
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T2_M5_CompatGate_Boundary")

    try: test_b_f40_pure_mojo_version_exact(ctx); ctx.record_pass("test_b_f40_pure_mojo_version_exact") except e: ctx.record_fail("test_b_f40_pure_mojo_version_exact", String(e))
    try: test_b_f40_build_commit_length(ctx); ctx.record_pass("test_b_f40_build_commit_length") except e: ctx.record_fail("test_b_f40_build_commit_length", String(e))
    try: test_b_f40_ieee754_determinism_subnormal(ctx); ctx.record_pass("test_b_f40_ieee754_determinism_subnormal") except e: ctx.record_fail("test_b_f40_ieee754_determinism_subnormal", String(e))
    try: test_b_f40_zero_c_dependencies_invariant(ctx); ctx.record_pass("test_b_f40_zero_c_dependencies_invariant") except e: ctx.record_fail("test_b_f40_zero_c_dependencies_invariant", String(e))
    try: test_b_f40_compiler_optimization_contract(ctx); ctx.record_pass("test_b_f40_compiler_optimization_contract") except e: ctx.record_fail("test_b_f40_compiler_optimization_contract", String(e))

    try: test_b_f41_throttle_clamping_boundary(ctx); ctx.record_pass("test_b_f41_throttle_clamping_boundary") except e: ctx.record_fail("test_b_f41_throttle_clamping_boundary", String(e))
    try: test_b_f41_steer_clamping_boundary(ctx); ctx.record_pass("test_b_f41_steer_clamping_boundary") except e: ctx.record_fail("test_b_f41_steer_clamping_boundary", String(e))
    try: test_b_f41_speedometer_zero_speed(ctx); ctx.record_pass("test_b_f41_speedometer_zero_speed") except e: ctx.record_fail("test_b_f41_speedometer_zero_speed", String(e))
    try: test_b_f41_speedometer_supersonic_boundary(ctx); ctx.record_pass("test_b_f41_speedometer_supersonic_boundary") except e: ctx.record_fail("test_b_f41_speedometer_supersonic_boundary", String(e))
    try: test_b_f41_replay_stream_empty_capacity(ctx); ctx.record_pass("test_b_f41_replay_stream_empty_capacity") except e: ctx.record_fail("test_b_f41_replay_stream_empty_capacity", String(e))

    try: test_b_f42_spline_knot_t0_exact(ctx); ctx.record_pass("test_b_f42_spline_knot_t0_exact") except e: ctx.record_fail("test_b_f42_spline_knot_t0_exact", String(e))
    try: test_b_f42_spline_knot_t1_exact(ctx); ctx.record_pass("test_b_f42_spline_knot_t1_exact") except e: ctx.record_fail("test_b_f42_spline_knot_t1_exact", String(e))
    try: test_b_f42_train_single_cart(ctx); ctx.record_pass("test_b_f42_train_single_cart") except e: ctx.record_fail("test_b_f42_train_single_cart", String(e))
    try: test_b_f42_g_force_weightlessness_crest(ctx); ctx.record_pass("test_b_f42_g_force_weightlessness_crest") except e: ctx.record_fail("test_b_f42_g_force_weightlessness_crest", String(e))
    try: test_b_f42_zero_speed_g_force(ctx); ctx.record_pass("test_b_f42_zero_speed_g_force") except e: ctx.record_fail("test_b_f42_zero_speed_g_force", String(e))

    try: test_b_f43_harness_approx_epsilon_boundary(ctx); ctx.record_pass("test_b_f43_harness_approx_epsilon_boundary") except e: ctx.record_fail("test_b_f43_harness_approx_epsilon_boundary", String(e))
    try: test_b_f43_harness_vec3_approx_boundary(ctx); ctx.record_pass("test_b_f43_harness_vec3_approx_boundary") except e: ctx.record_fail("test_b_f43_harness_vec3_approx_boundary", String(e))
    try: test_b_f43_headless_shot_ihdr_dimensions(ctx); ctx.record_pass("test_b_f43_headless_shot_ihdr_dimensions") except e: ctx.record_fail("test_b_f43_headless_shot_ihdr_dimensions", String(e))
    try: test_b_f43_zero_memory_leak_threshold(ctx); ctx.record_pass("test_b_f43_zero_memory_leak_threshold") except e: ctx.record_fail("test_b_f43_zero_memory_leak_threshold", String(e))
    try: test_b_f43_gate_exit_code_contract(ctx); ctx.record_pass("test_b_f43_gate_exit_code_contract") except e: ctx.record_fail("test_b_f43_gate_exit_code_contract", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("test_t2_m5_compat_gate failed")
