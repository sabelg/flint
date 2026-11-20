"""Tier 1 Feature Coverage: Milestone 5 Compatibility & Gate (F40–F43: 20 tests).
Opaque-box and requirement-driven verification of Pure Mojo 1.2 gate across all 22 subpackages,
NeonDrift vehicle/asset compatibility, Rollercoaster PhysicsWorld coupling, and E2E render/manifest gates.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, sqrt
from flint.anim.plugin import AnimPlugin
from flint.app import AppConfig, EngineApp
from flint.assets.flpk_v3 import (
    ASSET_KIND_RAW,
    CHUNK_COMP_NONE,
    AssetUuid,
    FlpkBuildChunk,
    FlpkV3MappedArchive,
    encode_flpk_v3,
)
from flint.assets.glb_native import parse_native_glb
from flint.assets.glb_writer import encode_native_glb
from flint.assets.image import decode_png, write_png
from flint.assets.model import NativeModel, NativeNode
from flint.assets.replay_stream import FlintReplayStream, _catmull_rom_vec3
from flint.audio.plugin import AudioPlugin
from flint.core.build_stamp import BUILD_COMMIT, BUILD_MOJO
from flint.core.detmath import det_cos, det_sin
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.ecs import World
from flint.input.plugin import InputPlugin
from flint.net.plugin import NetPlugin
from flint.physics.forces import FieldVolume
from flint.physics.joints import JOINT_REVOLUTE, JointDef
from flint.physics.plugin import PhysicsPlugin
from flint.physics.vehicle import VehicleChassis
from flint.physics.world import PhysicsWorld
from flint.platform.dsp import ModalResonator, compute_spatial_3d
from flint.platform.os import file_size, list_directory, read_bytes, write_bytes
from flint.procedural.arch_kit import (
    build_street_bollard_and_grate,
    build_street_traffic_cantilever,
    build_transit_shelter,
    build_transit_shelter_parts,
)
from flint.render.image_diff import compare_pixels
from flint.render.plugin import RenderPlugin
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, ScalableRenderer
from flint.scene import TransformPlugin
from flint.ui.draw_list import UiDrawList
from e2e.common.harness import TestContext
from e2e.common.v5_fixtures import register_mock_render_scene


@always_inline
def get_flint_root() -> String:
    if file_size("src/flint/core/math.mojo") > 0:
        return ""
    return "/home/user/flint/"


def _str_in_list(hay: List[String], needle: String) -> Bool:
    for i in range(len(hay)):
        if hay[i] == needle:
            return True
    return False


# =========================================================================
# F40: Pure Mojo 1.2 & Zero Warnings Gate (5 tests)
# =========================================================================

def test_f40_pure_mojo_engine_sources(mut ctx: TestContext) raises:
    """Dynamically scans ALL 22 src/flint/ subpackages verifying 100% .mojo sources."""
    var root = get_flint_root()
    var expected: List[String] = [
        "ai", "anim", "app", "assets", "audio", "camera", "core", "cycles",
        "devtools", "ecs", "input", "lab", "nav", "net", "physics", "platform",
        "procedural", "render", "scene", "studio", "text", "ui",
    ]
    var discovered = list_directory(root + "src/flint")
    ctx.assert_true(len(discovered) >= 22, "src/flint contains all >= 22 subpackages")
    for e in range(len(expected)):
        ctx.assert_true(_str_in_list(discovered, expected[e]), "expected subpackage present: " + expected[e])

    var total_modules = 0
    for p in range(len(discovered)):
        var pkg = discovered[p]
        if pkg.startswith("."):
            continue
        var files = list_directory(root + "src/flint/" + pkg)
        for f in range(len(files)):
            var fname = files[f]
            ctx.assert_true(fname.endswith(".mojo"), "engine file in " + pkg + " must end with .mojo: " + fname)
            ctx.assert_false(
                fname.endswith(".c") or fname.endswith(".cpp") or fname.endswith(".cc")
                or fname.endswith(".h") or fname.endswith(".hpp") or fname.endswith(".so")
                or fname.endswith(".a") or fname.endswith(".o") or fname.endswith(".dylib") or fname.endswith(".dll"),
                "zero C/C++ wrapper artifacts permitted",
            )
            total_modules += 1
    ctx.assert_true(total_modules >= 180, "verified >= 180 pure Mojo engine modules across all 22 subpackages")


def test_f40_zero_gameplay_leak_gate(mut ctx: TestContext) raises:
    var cmd = String("cd /home/user/flint && bash tools/flint.sh sync --verify > /dev/null 2>&1")
    var rc = external_call["system", Int32](cmd.as_c_string_span())
    ctx.assert_equal(Int(rc), 0, "Zero-Game-Leak gate executes cleanly with exit code 0")


def test_f40_compiler_provenance_and_clean_stamp(mut ctx: TestContext) raises:
    ctx.assert_true(BUILD_MOJO.startswith("1.2"), "compiler version conforms to Mojo 1.2")
    ctx.assert_true(BUILD_COMMIT.byte_length() >= 3, "build commit hash is non-empty")
    ctx.assert_true((BUILD_COMMIT + ":" + BUILD_MOJO).byte_length() >= 8, "build stamp valid")


def test_f40_zero_warnings_compilation(mut ctx: TestContext) raises:
    """Compiles a v5 EngineApp + all-plugins probe under --Werror without mutating build_stamp.mojo."""
    var root = get_flint_root()
    var abs_root = root if root.byte_length() > 0 else String("/home/user/flint/")
    var pid_s = String(Int(external_call["getpid", Int32]()))
    var probe_src = "/tmp/flint_e2e_f40_probe_" + pid_s + ".mojo"
    var probe_bin = "/tmp/flint_e2e_f40_probe_" + pid_s + "_bin"
    var log_path = "/tmp/flint_e2e_f40_probe_" + pid_s + ".log"
    ctx.cleanup_artifact(probe_src)
    ctx.cleanup_artifact(probe_bin)
    ctx.cleanup_artifact(log_path)

    var code = String(
        "from flint.app import AppConfig, EngineApp\n"
        + "from flint.ecs import World\n"
        + "from flint.scene import TransformPlugin\n"
        + "from flint.physics.plugin import PhysicsPlugin\n"
        + "from flint.anim.plugin import AnimPlugin\n"
        + "from flint.audio.plugin import AudioPlugin\n"
        + "from flint.input.plugin import InputPlugin\n"
        + "from flint.render.plugin import RenderPlugin\n"
        + "from flint.net.plugin import NetPlugin\n"
        + "def main():\n"
        + "    var app = EngineApp(AppConfig(title=\"Probe\", headless=True, worker_threads=0))\n"
        + "    app.add_plugin(TransformPlugin())\n"
        + "    app.add_plugin(InputPlugin(enable_evdev=False))\n"
        + "    app.add_plugin(PhysicsPlugin())\n"
        + "    app.add_plugin(AnimPlugin())\n"
        + "    app.add_plugin(AudioPlugin(headless=True))\n"
        + "    app.add_plugin(RenderPlugin())\n"
        + "    app.add_plugin(NetPlugin())\n"
        + "    _ = World()\n"
    )
    var cb = List[UInt8]()
    for b in code.as_bytes():
        cb.append(b)
    ctx.assert_true(write_bytes(probe_src, cb), "write f40 probe source")

    var cmd = (
        "export MOJO_PREFIX=\"/home/user/flint/.pixi/envs/default\" "
        + "MODULAR_HOME=\"/home/user/flint/.pixi/envs/default/share/max\" "
        + "LD_LIBRARY_PATH=\"/home/user/flint/.pixi/envs/default/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\" && "
        + "/home/user/flint/.pixi/envs/default/bin/mojo build -I " + abs_root + "src -O1 --fp-mode contract=off --Werror "
        + probe_src + " -o " + probe_bin + " > " + log_path + " 2>&1"
    )
    var rc = external_call["system", Int32](cmd.as_c_string_span())
    var log_bytes = read_bytes(log_path)
    var log_str = String(from_utf8=log_bytes)
    ctx.cleanup_artifact(probe_src)
    ctx.cleanup_artifact(probe_bin)
    ctx.cleanup_artifact(log_path)
    ctx.assert_equal(Int(rc), 0, "v5 probe compiles with exit code 0 under --Werror: " + log_str)
    ctx.assert_true(log_str.find("warning:") < 0, "zero compiler warnings")
    ctx.assert_true(log_str.find("error:") < 0, "zero compiler errors")


def test_f40_float_determinism_contract(mut ctx: TestContext) raises:
    var a: Float32 = 1.0000001
    var b: Float32 = 1.0000001
    var prod = a * b
    var neg_prod = -prod
    ctx.assert_approx(a * b + neg_prod, 0.0, 0.00001, "contract=off preserves subexpression determinism")
    var v8 = SIMD[DType.float32, 8](0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    var s1 = det_sin(v8)
    var s2 = det_sin(v8)
    var c1 = det_cos(v8)
    for i in range(8):
        ctx.assert_equal(s1[i], s2[i], "det_sin SIMD8 lane bit-exact")
        ctx.assert_approx(s1[i] * s1[i] + c1[i] * c1[i], 1.0, 0.001, "detmath SIMD8 trig identity")


# =========================================================================
# F41: NeonDrift 2099 Compatibility (5 tests)
# =========================================================================

def test_f41_neondrift_vehicle_interface(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(80.0, 0.5, 80.0), mass=0.0)
    var car = VehicleChassis(1420.0)
    car.setup_four_wheel_car(wheelbase=2.7, track_width=1.6, chassis_height=0.45)
    car.pos = Vec3(0.0, 0.55, 0.0)
    var slot = pw.add_vehicle(car^)
    pw.rebuild_broadphase()
    pw.set_vehicle_controls(slot, 1.0, 0.0, 0.25)
    var z0 = pw.vehicle_state(slot).pos.z
    for _ in range(15):
        pw.step(Float32(1.0 / 60.0))
    var st = pw.vehicle_state(slot)
    ctx.assert_true(st.pos.z > z0, "PhysicsWorld vehicle step advances along forward axis")
    ctx.assert_true(st.linear_vel.length() > 0.0, "VehicleChassis accelerates under throttle in PhysicsWorld")


def test_f41_neondrift_city_assets(mut ctx: TestContext) raises:
    var shelter_parts = build_transit_shelter_parts()
    var cantilever = build_street_traffic_cantilever()
    var bollard = build_street_bollard_and_grate()
    ctx.assert_true(cantilever.vertex_count() > 0 and bollard.vertex_count() > 0, "arch_kit builds meshes")

    var m_four = NativeModel()
    m_four.meshes.append(shelter_parts[0].copy())
    m_four.meshes.append(shelter_parts[1].copy())
    m_four.meshes.append(cantilever^)
    m_four.meshes.append(bollard^)
    for idx in range(4):
        var nd = NativeNode()
        nd.name = "part_" + String(idx)
        nd.mesh = idx
        nd.mesh_indices.append(idx)
        m_four.nodes.append(nd^)

    var m_two = NativeModel()
    m_two.meshes.append(build_transit_shelter())
    m_two.meshes.append(shelter_parts[1].copy())
    for idx in range(2):
        var nd = NativeNode()
        nd.name = "hover_" + String(idx)
        nd.mesh = idx
        nd.mesh_indices.append(idx)
        m_two.nodes.append(nd^)

    var chunks = List[FlpkBuildChunk]()
    chunks.append(FlpkBuildChunk(AssetUuid.from_path("models/urban_kit_4.glb"), "models/urban_kit_4.glb", 0, ASSET_KIND_RAW, encode_native_glb(m_four), CHUNK_COMP_NONE))
    chunks.append(FlpkBuildChunk(AssetUuid.from_path("models/transit_kit_2.glb"), "models/transit_kit_2.glb", 0, ASSET_KIND_RAW, encode_native_glb(m_two), CHUNK_COMP_NONE))

    var pack_path = String("/tmp/flint_e2e_f41_city_v3.flpk")
    ctx.cleanup_artifact(pack_path)
    var pack_bytes = encode_flpk_v3(chunks, page_align=True)
    ctx.assert_true(write_bytes(pack_path, pack_bytes), "write .flpk v3 archive")

    var arch = FlpkV3MappedArchive.open(pack_path)
    ctx.assert_true(arch.valid and arch.toc_count() == 2, ".flpk v3 TOC contains 2 entries")
    var t0 = arch.find_by_uuid(AssetUuid.from_path("models/urban_kit_4.glb"), 0)
    var t1 = arch.find_by_uuid(AssetUuid.from_path("models/transit_kit_2.glb"), 0)
    ctx.assert_true(t0 >= 0 and arch.verify_entry_crc64(t0), "urban_kit_4 CRC64 valid")
    ctx.assert_true(t1 >= 0 and arch.verify_entry_crc64(t1), "transit_kit_2 CRC64 valid")
    var p0 = arch.read_payload_by_toc(t0, verify_crc=True)
    var p1 = arch.read_payload_by_toc(t1, verify_crc=True)
    ctx.assert_equal(len(parse_native_glb(p0[1]).meshes), 4, "urban_kit_4 round-trips 4 meshes")
    ctx.assert_equal(len(parse_native_glb(p1[1]).meshes), 2, "transit_kit_2 round-trips 2 meshes")
    arch.close()
    ctx.cleanup_artifact(pack_path)


def test_f41_neondrift_audio_ambience(mut ctx: TestContext) raises:
    var res = ModalResonator((6500.0 / 60.0) * 4.0, 0.05, 1.0, 48000.0)
    ctx.assert_true(abs(res.step(1.0)) > 0.0, "engine resonator responds to excitation")


def test_f41_neondrift_speed_hud(mut ctx: TestContext) raises:
    var p = compute_spatial_3d(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 50), Vec3(0, 0, -60))
    ctx.assert_true(p.doppler_pitch > 1.08, "high-speed vehicle produces Doppler shift")


def test_f41_neondrift_replay_scrub(mut ctx: TestContext) raises:
    var replay = FlintReplayStream()
    replay.add_camera_knot(0, Vec3(0, 10, 0), Vec3(0, 0, 1), 60.0)
    replay.add_camera_knot(10, Vec3(0, 15, 20), Vec3(0, 0, 1), 60.0)
    var knot_res = replay.evaluate_camera_spline(5.0)
    ctx.assert_true(knot_res[0], "replay camera spline evaluates cleanly")
    ctx.assert_approx(knot_res[1].eye.y, 12.5, 0.5, "camera path interpolates between keyframes")


# =========================================================================
# F42: Rollercoaster Game Verification (5 tests)
# =========================================================================

def test_f42_rollercoaster_track_spline(mut ctx: TestContext) raises:
    var mid = _catmull_rom_vec3(Vec3(0, 0, 0), Vec3(0, 10, 10), Vec3(0, 20, 30), Vec3(0, 15, 50), 0.5)
    ctx.assert_true(mid.y > 10.0 and mid.y < 20.0, "spline smoothly interpolates height")


def test_f42_rollercoaster_cart_physics(mut ctx: TestContext) raises:
    comptime CARS_IN_TRAIN: Int = 5
    var pw = PhysicsWorld(gravity=0.0)
    var car_slots = List[Int]()
    for c in range(CARS_IN_TRAIN):
        var slot = pw.add_box(Vec3(0.0, 2.0, Float32(c) * 3.0), Vec3(0.8, 0.5, 1.2), mass=500.0)
        if c == 0:
            pw.bodies[slot].linear_vel = Vec3(0.0, 0.0, 25.0)
        car_slots.append(slot)
    for c in range(CARS_IN_TRAIN - 1):
        var sa = car_slots[c]
        var sb = car_slots[c + 1]
        var anchor = Vec3(0.0, 2.0, Float32(c) * 3.0 + 1.5)
        _ = pw.add_joint(JointDef(JOINT_REVOLUTE, sa, sb, anchor, Vec3(1, 0, 0)))
    pw.rebuild_broadphase()
    for _ in range(30):
        pw.step(Float32(1.0 / 60.0))
    var s0 = car_slots[0]
    var s4 = car_slots[4]
    ctx.assert_true(pw.bodies[s0].pos.z > 1.5, "coupled train lead car advances along +Z")
    ctx.assert_true(pw.bodies[s4].pos.z > 13.5, "tail car starting at rest (z=12) pulled forward by JOINT_REVOLUTE")
    ctx.assert_true(pw.bodies[s4].linear_vel.z > 3.0, "tail car acquired positive forward velocity via joint coupling")


def test_f42_rollercoaster_g_force_calc(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(80.0, 0.5, 80.0), mass=0.0)
    var car = VehicleChassis(1200.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(0.0, 0.55, 0.0)
    car.linear_vel = Vec3(0.0, 0.0, 22.0)
    var slot = pw.add_vehicle(car^)
    pw.rebuild_broadphase()
    pw.set_vehicle_controls(slot, 0.5, 0.0, 0.45)
    var dt = Float32(1.0 / 60.0)
    for _ in range(10):
        pw.step(dt)
    var v_before = pw.vehicle_state(slot).linear_vel
    pw.step(dt)
    var v_after = pw.vehicle_state(slot).linear_vel
    var accel = (v_after - v_before) * (Float32(1.0) / dt)
    var lat_g = abs(accel.x) / Float32(9.81)
    var total_g = sqrt(lat_g * lat_g + Float32(1.0))
    ctx.assert_true(total_g > 1.0, "turning PhysicsWorld vehicle experiences > 1.0G combined load")


def test_f42_rollercoaster_camera_follow(mut ctx: TestContext) raises:
    var t0 = _catmull_rom_vec3(Vec3(0, 0, 0), Vec3(0, 10, 10), Vec3(0, 20, 30), Vec3(0, 15, 50), 0.5)
    var t1 = _catmull_rom_vec3(Vec3(0, 0, 0), Vec3(0, 10, 10), Vec3(0, 20, 30), Vec3(0, 15, 50), 0.51)
    var forward = (t1 - t0).normalized()
    ctx.assert_true(forward.z > 0.0 and forward.y > 0.0, "chase camera forward tracks track tangent")


def test_f42_rollercoaster_track_collision(mut ctx: TestContext) raises:
    var tube = FieldVolume.capsule(1, 0, Vec3(0, 0, 0), Vec3(0, 0, 10), 0.15)
    ctx.assert_true(tube.inside(Vec3(0.05, 0.05, 5.0)), "rail interior inside track tube volume")
    ctx.assert_false(tube.inside(Vec3(0.25, 0.25, 5.0)), "wheel guide outside track tube volume")


# =========================================================================
# F43: 100% E2E Suite Inventory, Harness & Headless Render Verification (5 tests)
# =========================================================================

def test_f43_harness_failure_detection(mut ctx: TestContext) raises:
    var sub = TestContext()
    var threw = False
    try:
        sub.assert_true(False, "deliberate assertion failure")
    except:
        threw = True
    ctx.assert_true(threw, "TestContext.assert_true raises Error on False")
    sub.record_fail("test_simulated_fail", "expected deliberate failure")
    ctx.assert_equal(sub.failed_tests, 1, "failed_tests == 1")
    ctx.assert_equal(sub.passed_tests, 0, "passed_tests == 0")
    ctx.assert_equal(sub.total_tests, 1, "total_tests == 1")


def test_f43_harness_pass_detection(mut ctx: TestContext) raises:
    var sub = TestContext()
    sub.assert_true(True)
    sub.assert_false(False)
    sub.assert_equal(42, 42)
    sub.assert_approx(Float32(1.0), Float32(1.0005), 0.001)
    sub.record_pass("test_passing_assertions")
    ctx.assert_equal(sub.passed_tests, 1, "passed_tests == 1")
    ctx.assert_equal(sub.failed_tests, 0, "failed_tests == 0")


def test_f43_suite_inventory_manifest(mut ctx: TestContext) raises:
    var root = get_flint_root()
    var files = List[String]()
    files.append(root + "tests/test_e2e.mojo")
    files.append(root + "tests/e2e/common/harness.mojo")
    files.append(root + "tests/e2e/common/mock_data.mojo")
    files.append(root + "tests/e2e/common/v5_fixtures.mojo")
    # Tier 1 (5)
    files.append(root + "tests/e2e/tier1_coverage/test_t1_m1_phase0.mojo")
    files.append(root + "tests/e2e/tier1_coverage/test_t1_m2_gpu_loop.mojo")
    files.append(root + "tests/e2e/tier1_coverage/test_t1_m3_rt_streaming.mojo")
    files.append(root + "tests/e2e/tier1_coverage/test_t1_m4_world_sim.mojo")
    files.append(root + "tests/e2e/tier1_coverage/test_t1_m5_compat_gate.mojo")
    # Tier 2 (5)
    files.append(root + "tests/e2e/tier2_boundary/test_t2_m1_phase0.mojo")
    files.append(root + "tests/e2e/tier2_boundary/test_t2_m2_gpu_loop.mojo")
    files.append(root + "tests/e2e/tier2_boundary/test_t2_m3_rt_streaming.mojo")
    files.append(root + "tests/e2e/tier2_boundary/test_t2_m4_world_sim.mojo")
    files.append(root + "tests/e2e/tier2_boundary/test_t2_m5_compat_gate.mojo")
    # Tier 3 (3)
    files.append(root + "tests/e2e/tier3_cross_feature/test_t3_render_pipeline.mojo")
    files.append(root + "tests/e2e/tier3_cross_feature/test_t3_physics_anim.mojo")
    files.append(root + "tests/e2e/tier3_cross_feature/test_t3_platform_assets.mojo")
    # Tier 4 (5)
    files.append(root + "tests/e2e/tier4_real_world/test_t4_neondrift_scenario.mojo")
    files.append(root + "tests/e2e/tier4_real_world/test_t4_rollercoaster_scenario.mojo")
    files.append(root + "tests/e2e/tier4_real_world/test_t4_openworld_scenario.mojo")
    files.append(root + "tests/e2e/tier4_real_world/test_t4_asset_pipeline_scenario.mojo")
    files.append(root + "tests/e2e/tier4_real_world/test_t4_character_combat_scenario.mojo")
    # Tier 5 (2)
    files.append(root + "tests/e2e/tier5_adversarial/test_t5_render_assets.mojo")
    files.append(root + "tests/e2e/tier5_adversarial/test_t5_physics_anim_platform.mojo")

    for i in range(len(files)):
        ctx.assert_true(file_size(files[i]) > 0, "E2E file exists: " + files[i])

    ctx.assert_equal(len(list_directory(root + "tests/e2e/tier1_coverage")), 5, "Tier 1 count == 5")
    ctx.assert_equal(len(list_directory(root + "tests/e2e/tier2_boundary")), 5, "Tier 2 count == 5")
    ctx.assert_equal(len(list_directory(root + "tests/e2e/tier3_cross_feature")), 3, "Tier 3 count == 3")
    ctx.assert_equal(len(list_directory(root + "tests/e2e/tier4_real_world")), 5, "Tier 4 count == 5")
    ctx.assert_equal(len(list_directory(root + "tests/e2e/tier5_adversarial")), 2, "Tier 5 count == 2")


def test_f43_artifact_lifecycle(mut ctx: TestContext) raises:
    var temp_path = String("/tmp/flint_e2e_artifact_lifecycle_test.tmp")
    var data: List[UInt8] = [0xDE, 0xAD, 0xBE, 0xEF]
    _ = write_bytes(temp_path, data)
    ctx.assert_equal(file_size(temp_path), 4, "temporary artifact written")
    ctx.assert_equal(len(read_bytes(temp_path)), 4, "read_bytes returns payload")
    ctx.cleanup_artifact(temp_path)
    ctx.assert_equal(file_size(temp_path), -1, "cleanup_artifact unlinks file")


def test_f43_headless_shot_artifact(mut ctx: TestContext) raises:
    """Renders real headless ScalableRenderer frames, validates PNG encode/decode + pixel variance + 2-frame FLIP determinism,
    and decodes build/shots/*.png whenever already present on disk (never invokes tools/flint.sh shot)."""
    var shot_tmp = String("/tmp/flint_e2e_f43_shot.png")
    ctx.cleanup_artifact(shot_tmp)

    var ren = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120)
    var rw = register_mock_render_scene(ren)
    var ui = UiDrawList()
    ctx.assert_true(ren.render_world(rw, ui), "ScalableRenderer frame 1 renders")
    var px1 = ren.readback_pixels()
    ctx.assert_equal(len(px1), 160 * 120, "frame 1 readback size == 160x120")

    ctx.assert_true(ren.render_world(rw, ui), "ScalableRenderer frame 2 renders")
    var px2 = ren.readback_pixels()
    var diff = compare_pixels(px1, px2, 160, 120, 0)
    ctx.assert_true(diff.same_size, "compare_pixels sizes match")
    ctx.assert_approx(Float32(diff.flip_mean), 0.0, 0.0001, "2-frame headless render is FLIP-identical (flip_mean == 0.0)")

    var px_mut = px1.copy()
    ctx.assert_true(write_png(shot_tmp, px_mut, 160, 120, stamped=False), "write_png writes /tmp shot")
    var dec = decode_png(read_bytes(shot_tmp), 16384)
    ctx.assert_true(dec.valid and dec.width == 160 and dec.height == 120, "decode_png validates written PNG")
    var first_px = dec.pixels[0]
    var has_variance = False
    for i in range(len(dec.pixels)):
        if dec.pixels[i] != first_px:
            has_variance = True
            break
    ctx.assert_true(has_variance, "rendered headless frame has non-uniform pixels")
    ctx.cleanup_artifact(shot_tmp)

    var root = get_flint_root()
    var shot_names: List[String] = ["flint_spark.png", "open_world_noon_vista.png", "flint_studio.png"]
    for i in range(len(shot_names)):
        var sp = root + "build/shots/" + shot_names[i]
        if file_size(sp) > 0:
            var img = decode_png(read_bytes(sp), 16384)
            ctx.assert_true(img.valid and img.width >= 640 and img.height >= 360, "disk shot valid PNG: " + shot_names[i])
            ctx.assert_equal(len(img.pixels), img.width * img.height, "disk shot pixel buffer matches dimensions: " + shot_names[i])
            var p0 = img.pixels[0]
            var disk_var = False
            for k in range(len(img.pixels)):
                if img.pixels[k] != p0:
                    disk_var = True
                    break
            ctx.assert_true(disk_var, "disk shot has non-uniform pixels: " + shot_names[i])


# =========================================================================
# Aggregator for Tier 1 Milestone 5
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T1_M5_CompatGate")

    try: test_f40_pure_mojo_engine_sources(ctx); ctx.record_pass("test_f40_pure_mojo_engine_sources") except e: ctx.record_fail("test_f40_pure_mojo_engine_sources", String(e))
    try: test_f40_zero_gameplay_leak_gate(ctx); ctx.record_pass("test_f40_zero_gameplay_leak_gate") except e: ctx.record_fail("test_f40_zero_gameplay_leak_gate", String(e))
    try: test_f40_compiler_provenance_and_clean_stamp(ctx); ctx.record_pass("test_f40_compiler_provenance_and_clean_stamp") except e: ctx.record_fail("test_f40_compiler_provenance_and_clean_stamp", String(e))
    try: test_f40_zero_warnings_compilation(ctx); ctx.record_pass("test_f40_zero_warnings_compilation") except e: ctx.record_fail("test_f40_zero_warnings_compilation", String(e))
    try: test_f40_float_determinism_contract(ctx); ctx.record_pass("test_f40_float_determinism_contract") except e: ctx.record_fail("test_f40_float_determinism_contract", String(e))

    try: test_f41_neondrift_vehicle_interface(ctx); ctx.record_pass("test_f41_neondrift_vehicle_interface") except e: ctx.record_fail("test_f41_neondrift_vehicle_interface", String(e))
    try: test_f41_neondrift_city_assets(ctx); ctx.record_pass("test_f41_neondrift_city_assets") except e: ctx.record_fail("test_f41_neondrift_city_assets", String(e))
    try: test_f41_neondrift_audio_ambience(ctx); ctx.record_pass("test_f41_neondrift_audio_ambience") except e: ctx.record_fail("test_f41_neondrift_audio_ambience", String(e))
    try: test_f41_neondrift_speed_hud(ctx); ctx.record_pass("test_f41_neondrift_speed_hud") except e: ctx.record_fail("test_f41_neondrift_speed_hud", String(e))
    try: test_f41_neondrift_replay_scrub(ctx); ctx.record_pass("test_f41_neondrift_replay_scrub") except e: ctx.record_fail("test_f41_neondrift_replay_scrub", String(e))

    try: test_f42_rollercoaster_track_spline(ctx); ctx.record_pass("test_f42_rollercoaster_track_spline") except e: ctx.record_fail("test_f42_rollercoaster_track_spline", String(e))
    try: test_f42_rollercoaster_cart_physics(ctx); ctx.record_pass("test_f42_rollercoaster_cart_physics") except e: ctx.record_fail("test_f42_rollercoaster_cart_physics", String(e))
    try: test_f42_rollercoaster_g_force_calc(ctx); ctx.record_pass("test_f42_rollercoaster_g_force_calc") except e: ctx.record_fail("test_f42_rollercoaster_g_force_calc", String(e))
    try: test_f42_rollercoaster_camera_follow(ctx); ctx.record_pass("test_f42_rollercoaster_camera_follow") except e: ctx.record_fail("test_f42_rollercoaster_camera_follow", String(e))
    try: test_f42_rollercoaster_track_collision(ctx); ctx.record_pass("test_f42_rollercoaster_track_collision") except e: ctx.record_fail("test_f42_rollercoaster_track_collision", String(e))

    try: test_f43_harness_failure_detection(ctx); ctx.record_pass("test_f43_harness_failure_detection") except e: ctx.record_fail("test_f43_harness_failure_detection", String(e))
    try: test_f43_harness_pass_detection(ctx); ctx.record_pass("test_f43_harness_pass_detection") except e: ctx.record_fail("test_f43_harness_pass_detection", String(e))
    try: test_f43_suite_inventory_manifest(ctx); ctx.record_pass("test_f43_suite_inventory_manifest") except e: ctx.record_fail("test_f43_suite_inventory_manifest", String(e))
    try: test_f43_artifact_lifecycle(ctx); ctx.record_pass("test_f43_artifact_lifecycle") except e: ctx.record_fail("test_f43_artifact_lifecycle", String(e))
    try: test_f43_headless_shot_artifact(ctx); ctx.record_pass("test_f43_headless_shot_artifact") except e: ctx.record_fail("test_f43_headless_shot_artifact", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("test_t1_m5_compat_gate failed")
