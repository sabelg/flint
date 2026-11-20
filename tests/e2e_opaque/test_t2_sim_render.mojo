"""Opaque-box E2E Tier 2 (Boundary, Corner & Invalid Input) suite for FA3 and FA4.

Covers >= 6 boundary/negative tests for FA3 (zero/tiny/negative dt, degenerate
raycast, invalid IDs/filters/corrupt blob, steep cliff KCC, 0-dt/0-wheel vehicle &
aero boundaries, unpinned cloth & degenerate XPBD, empty/0-dt ragdoll, corrupt
`.flanim` & empty footplant) and >= 6 boundary/negative tests for FA4 (zero-draw
frame, tiny/0 resize clamp, `AaaRenderGraph` invalid limits & empty graph,
`FlintAaaVulkanRenderer` zero-draw headless-safe, 0x0/mismatch FLIP, 0x0/mismatch
`image_diff`, missing/corrupt PNG `compare_with_png`).
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs
from std.testing import assert_true, assert_false, assert_equal

from flint.core.math import Vec3, Quat, Mat4
from flint.core.ik import angle_axis
from flint.core.serial import BlobReader, float32_bits

from flint.physics.world import PhysicsWorld
from flint.physics.joints import JointDef, JOINT_WELD
from flint.physics.filter import CollisionFilter, CAT_STATIC, CAT_PROP_PIECE, CAT_ALL
from flint.physics.character import CharacterController, CharacterControllerConfig
from flint.physics.vehicle import VehicleChassis
from flint.physics.aero6dof import (
    Aero6DofConfig,
    Aero6DofControls,
    Aero6DofState,
    step_aero6dof,
    eval_isa_atmosphere,
    ISA_RHO0,
)
from flint.physics.xpbd import ParticleChain, ParticleMesh, ChainCollider

from flint.assets.model import NativeModel, NativeNode
from flint.platform.os import make_dirs, remove_file, write_bytes

from flint.anim.pose import SkeletonInstance
from flint.anim.procedural import Placement, FootPlantSolver
from flint.anim.ragdoll import ArticulatedRagdoll
from flint.anim.flanim_ecs import load_flanim_bytes, solve_foot_plant_world

from flint.render.render_world import (
    ScalableRenderer,
    RenderWorld,
    RenderInstancePacket,
    RENDER_TIER_0_HEADLESS_SIMD,
)
from flint.render.aaa_render_graph import (
    AaaRenderGraph,
    RgResourceAccess,
    RG_QUEUE_GRAPHICS,
    RG_USAGE_COMPUTE_WRITE,
    RG_UNIT_BYTES,
)
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.flip import flip_error_map, flip_mean, flip_heat_image
from flint.render.image_diff import compare_pixels, compare_with_png
from flint.ui.draw_list import UiDrawList


def _make_simple_biped() -> NativeModel:
    var m = NativeModel()
    var names = [
        String("Pelvis"), String("Chest"),
        String("Thigh.L"), String("Shin.L"), String("Foot.L"),
        String("Thigh.R"), String("Shin.R"), String("Foot.R"),
    ]
    var parents = [-1, 0, 0, 2, 3, 0, 5, 6]
    var offsets = [
        Vec3(0.0, 0.92, 0.0), Vec3(0.0, 0.42, 0.0),
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
        nd.scale = Vec3(1, 1, 1)
        nd.base_scale = Vec3(1, 1, 1)
        m.nodes.append(nd^)
        var p = parents[i]
        if p >= 0:
            m.nodes[p].children.append(i)
        else:
            m.roots.append(i)
    return m^


# ============================================================================
# FA3 Tier 2: Boundary, Corner & Negative Simulation Tests
# ============================================================================

def test_fa3_t2_physics_zero_tiny_negative_dt_and_degenerate_raycast() raises:
    var w = PhysicsWorld(gravity=9.81)
    _ = w.add_box(Vec3(0.0, -0.5, 0.0), Vec3(10.0, 0.5, 10.0), mass=0.0)
    var s = w.add_sphere(Vec3(0.0, 3.0, 0.0), 0.5, mass=1.0)
    w.rebuild_broadphase()

    w.step(Float32(0.0))
    w.step(Float32(1.0e-9))
    w.step(Float32(-0.016))

    assert_equal(w.tick, 0)
    assert_equal(float32_bits(w.time_s), float32_bits(Float32(0.0)))
    assert_equal(float32_bits(w.bodies[s].pos.y), float32_bits(Float32(3.0)))

    var h_zero_dir = w.raycast(Vec3(0.0, 3.0, -2.0), Vec3(0.0, 0.0, 0.0), 100.0)
    assert_false(h_zero_dir.hit)

    var h_zero_dist = w.raycast(Vec3(0.0, 3.0, -2.0), Vec3(0.0, 0.0, 1.0), 0.0)
    assert_false(h_zero_dist.hit)

    var h_neg_dist = w.raycast(Vec3(0.0, 3.0, -2.0), Vec3(0.0, 0.0, 1.0), -5.0)
    assert_false(h_neg_dist.hit)

    var h_all = w.raycast_all(Vec3(0.0, 3.0, -2.0), Vec3(0.0, 0.0, 0.0), 100.0)
    assert_equal(len(h_all), 0)


def test_fa3_t2_physics_invalid_ids_filters_and_corrupt_blob() raises:
    var w = PhysicsWorld()
    var s_slot = w.add_sphere(
        Vec3(0.0, 1.0, 4.0),
        0.5,
        mass=0.0,
        filter=CollisionFilter(CAT_PROP_PIECE, CAT_ALL, 0),
    )
    assert_true(w.body_id(s_slot) > 0)
    assert_equal(w.find_slot_by_id(-999), -1)
    assert_equal(w.body_id(-1), -1)
    assert_equal(w.body_id(9999), -1)
    assert_equal(w.add_joint(JointDef(JOINT_WELD, -1, 9999, Vec3())), -1)

    w.rebuild_broadphase()
    # Filter excluding CAT_PROP_PIECE must miss the prop sphere
    var flt_only_static = CollisionFilter(CAT_ALL, CAT_STATIC, 0)
    var h_miss = w.raycast(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), 20.0, filter=flt_only_static)
    assert_false(h_miss.hit)
    var ov_miss = w.overlap_sphere(Vec3(0.0, 1.0, 4.0), 2.0, filter=flt_only_static)
    assert_equal(len(ov_miss), 0)

    # Corrupt / truncated blob must raise an explicit Error
    var bad_bytes: List[UInt8] = [UInt8(0x00), UInt8(0x01), UInt8(0x02), UInt8(0x03)]
    var br = BlobReader(bad_bytes^)
    var w2 = PhysicsWorld()
    var raised = False
    try:
        w2.deserialize(br)
    except:
        raised = True
    assert_true(raised)


def test_fa3_t2_kcc_zero_dt_and_steep_cliff_no_wall_surf() raises:
    var w = PhysicsWorld(gravity=9.81)
    _ = w.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    var ang70 = Float32(70.0 * 0.01745329252)
    _ = w.add_box(Vec3(0.0, 2.5, 4.0), Vec3(3.0, 0.25, 5.0), mass=0.0, rot=angle_axis(-ang70, Vec3(1, 0, 0)))
    w.rebuild_broadphase()

    var cfg = CharacterControllerConfig()
    var kcc = CharacterController(Vec3(0.0, 0.90, 0.8), cfg)
    var pos0 = kcc.pos
    var out0 = kcc.move_and_slide(w, Vec3(10.0, 0.0, 10.0), Float32(0.0))
    assert_equal(float32_bits(out0.x), float32_bits(pos0.x))
    assert_equal(float32_bits(out0.y), float32_bits(pos0.y))
    assert_equal(float32_bits(out0.z), float32_bits(pos0.z))

    var dt = Float32(1.0 / 60.0)
    for _ in range(75):
        _ = kcc.move_and_slide(w, Vec3(0.0, 0.0, 5.0), dt)
    assert_true(kcc.foot_position().y < Float32(0.15))

    # Drop onto steep 70 deg slope -> flags on_steep_slope and slides downhill
    var kcc_drop = CharacterController(Vec3(0.0, 3.8, 3.8), cfg)
    var saw_steep = False
    for _ in range(45):
        _ = kcc_drop.move_and_slide(w, Vec3(0.0, 0.0, 0.0), dt)
        if kcc_drop.on_steep_slope:
            saw_steep = True
    assert_true(saw_steep)
    assert_true(kcc_drop.pos.y < Float32(3.3))


def test_fa3_t2_vehicle_and_aero_boundaries() raises:
    var car = VehicleChassis(1450.0)
    # 0-wheel unconfigured chassis must safely no-op
    car.substep_suspension(Float32(1.0 / 60.0), throttle=1.0)
    assert_equal(len(car.wheels), 0)

    car.setup_four_wheel_car()
    car.pos = Vec3(0.0, 0.54, 0.0)
    car.substep_suspension(Float32(0.0), throttle=1.0)
    car.substep_suspension(Float32(-0.1), throttle=1.0)
    assert_equal(float32_bits(car.pos.z), float32_bits(Float32(0.0)))

    # ISA atmosphere negative and extreme altitude boundaries
    var isa_neg = eval_isa_atmosphere(-500.0)
    assert_true(abs(isa_neg[0] - ISA_RHO0) < Float32(1.0e-4))
    var isa_high = eval_isa_atmosphere(200000.0)
    assert_true(isa_high[0] > Float32(0.0) and isa_high[0] == isa_high[0])

    # Zero dt and zero airspeed must produce finite state with no NaN
    var st = Aero6DofState(Vec3(0.0, 500.0, 0.0), Vec3(0.0, 0.0, 0.0))
    var acfg = Aero6DofConfig()
    var actrl = Aero6DofControls()
    step_aero6dof(st, acfg, actrl, Float32(0.0))
    assert_equal(float32_bits(st.pos.y), float32_bits(Float32(500.0)))

    for _ in range(10):
        step_aero6dof(st, acfg, actrl, Float32(1.0 / 60.0))
    assert_true(st.pos.x == st.pos.x and st.pos.y == st.pos.y and st.pos.z == st.pos.z)


def test_fa3_t2_xpbd_unpinned_cloth_and_degenerate_constraints() raises:
    var dt = Float32(1.0 / 60.0)
    # Unpinned cloth falls under gravity while preserving links
    var free_mesh = ParticleMesh()
    var a0 = free_mesh.add_particle(Vec3(0.0, 4.0, 0.0), 1.0, 0.05)
    var a1 = free_mesh.add_particle(Vec3(1.0, 4.0, 0.0), 1.0, 0.05)
    var a2 = free_mesh.add_particle(Vec3(0.0, 4.0, 1.0), 1.0, 0.05)
    var a3 = free_mesh.add_particle(Vec3(1.0, 4.0, 1.0), 1.0, 0.05)
    free_mesh.add_link(a0, a1, 0.0)
    free_mesh.add_link(a0, a2, 0.0)
    free_mesh.add_link(a1, a3, 0.0)
    free_mesh.add_link(a2, a3, 0.0)
    for _ in range(25):
        free_mesh.step(dt, 2, 9.81, List[Vec3](), List[ChainCollider](), iterations=2)
    assert_true(free_mesh.position[0].y < Float32(3.5))
    assert_true(free_mesh.link_error() < Float32(0.05))

    # Colocated particles, self-link, out-of-range pin, and dt=0
    var deg = ParticleMesh()
    var d0 = deg.add_particle(Vec3(0.0, 2.0, 0.0), 1.0, 0.05)
    var d1 = deg.add_particle(Vec3(0.0, 2.0, 0.0), 1.0, 0.05)
    deg.add_link(d0, d1, 0.0)
    deg.add_link(d0, d0, 0.0)   # self-link ignored
    deg.add_link(d0, 99, 0.0)   # out-of-range ignored
    deg.pin(99, Vec3())         # out-of-range pin ignored
    deg.step(Float32(0.0), 2, 9.81, List[Vec3](), List[ChainCollider](), iterations=2)
    assert_equal(deg.steps_run, 0)
    deg.step(dt, 2, 9.81, List[Vec3](), List[ChainCollider](), iterations=2)
    assert_true(deg.position[0].y == deg.position[0].y and deg.position[1].y == deg.position[1].y)

    # ParticleChain clamped to minimum 2 particles
    var short_chain = ParticleChain.line(Vec3(), Vec3(1.0, 0.0, 0.0), 1)
    assert_equal(short_chain.count(), 2)


def test_fa3_t2_ragdoll_empty_zero_dt_and_invalid_bone_impulse() raises:
    var empty_rag = ArticulatedRagdoll()
    assert_equal(float32_bits(empty_rag.mean_speed()), float32_bits(Float32(0.0)))
    assert_false(empty_rag.settled())
    empty_rag.step(Float32(1.0 / 60.0))
    assert_equal(empty_rag.steps_taken, 0)
    empty_rag.apply_impulse_at_bone(999, Vec3(10.0, 0.0, 0.0))

    var biped = _make_simple_biped()
    var rag = ArticulatedRagdoll()
    _ = rag.add_limb_from_model(biped, 0, -1, 15.0, 0.15, 0.35)
    _ = rag.add_limb_from_model(biped, 1, 0, 12.0, 0.15, 0.35)
    rag.apply_impulse_at_bone(999, Vec3(50.0, 0.0, 0.0))
    rag.step(Float32(0.0))
    rag.step(Float32(-0.01))
    assert_equal(rag.steps_taken, 0)


def test_fa3_t2_flanim_corrupt_bytes_and_empty_footplant() raises:
    var raised_empty = False
    try:
        _ = load_flanim_bytes(List[UInt8]())
    except:
        raised_empty = True
    assert_true(raised_empty)

    var bad_magic: List[UInt8] = [UInt8(0x42), UInt8(0x41), UInt8(0x44), UInt8(0x30), UInt8(1), UInt8(0), UInt8(0), UInt8(0)]
    var raised_magic = False
    try:
        _ = load_flanim_bytes(bad_magic)
    except:
        raised_magic = True
    assert_true(raised_magic)

    var w = PhysicsWorld()
    _ = w.add_box(Vec3(0.0, -0.5, 0.0), Vec3(10.0, 0.5, 10.0), mass=0.0)
    w.rebuild_broadphase()
    var biped = _make_simple_biped()
    var skel = SkeletonInstance()
    skel.reset_to_rest(biped)

    var empty_fp = FootPlantSolver(0)
    var hits_empty = solve_foot_plant_world(empty_fp, w, skel, biped, Placement(), List[Bool](), Float32(1.0 / 60.0))
    assert_equal(hits_empty, 0)

    var high_fp = FootPlantSolver(0)
    high_fp.add_leg(2, 3, 4, 0.04)
    high_fp.add_leg(5, 6, 7, 0.04)
    var stance: List[Bool] = [True, True]
    var hits_miss = solve_foot_plant_world(
        high_fp, w, skel, biped, Placement(Vec3(0.0, 50.0, 0.0)), stance, Float32(1.0 / 60.0)
    )
    assert_equal(hits_miss, 0)


# ============================================================================
# FA4 Tier 2: Boundary, Corner & Negative Rendering & Verification Tests
# ============================================================================

def test_fa4_t2_scalable_renderer_zero_draw_frame() raises:
    var ren = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, width=128, height=96)
    var rw = RenderWorld()
    rw.clear_frame(0)
    var ui = UiDrawList()
    assert_true(ren.render_world(rw, ui, capture_readback=True))
    var px = ren.readback_pixels()
    assert_equal(len(px), 128 * 96)
    assert_equal(ren.cpu_canvas.triangles_drawn, 0)

    # Invalid mesh_id must be skipped safely without crashing
    rw.instances.append(RenderInstancePacket(UInt64(99), 999, Mat4(), Mat4()))
    assert_true(ren.render_world(rw, ui, capture_readback=True))
    assert_equal(len(ren.readback_pixels()), 128 * 96)
    ren.shutdown()


def test_fa4_t2_scalable_renderer_tiny_resize_clamp() raises:
    var ren = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, width=0, height=-10)
    assert_equal(ren.width, 64)
    assert_equal(ren.height, 64)

    assert_true(ren.resize(0, 0))
    assert_equal(ren.width, 16)
    assert_equal(ren.height, 16)
    assert_equal(ren.cpu_canvas.width, 64)
    assert_equal(ren.cpu_canvas.height, 64)

    var rw = RenderWorld()
    var ui = UiDrawList()
    assert_true(ren.render_world(rw, ui, capture_readback=True))
    assert_equal(len(ren.readback_pixels()), ren.cpu_canvas.width * ren.cpu_canvas.height)

    assert_true(ren.resize(96, 80))
    assert_equal(ren.width, 96)
    assert_equal(ren.height, 80)
    assert_true(ren.render_world(rw, ui, capture_readback=True))
    assert_equal(len(ren.readback_pixels()), 96 * 80)
    ren.shutdown()


def test_fa4_t2_aaa_render_graph_invalid_limits_and_empty() raises:
    var rg_empty = AaaRenderGraph()
    assert_true(rg_empty.compile(heap_capacity_bytes=0))
    assert_equal(rg_empty.aliasing_metrics().unaliased_bytes, 0)

    # Transient active resource with 0 capacity must fail compile
    var rg_zero = AaaRenderGraph()
    var r0 = rg_zero.add_resource("buf", 4096, transient=True)
    var acc0 = List[RgResourceAccess]()
    acc0.append(RgResourceAccess(r0, RG_USAGE_COMPUTE_WRITE))
    _ = rg_zero.add_pass("p0", acc0, RG_QUEUE_GRAPHICS)
    assert_false(rg_zero.compile(heap_capacity_bytes=0))

    # Resource larger than heap capacity must fail compile
    var rg_oom = AaaRenderGraph()
    var r_big = rg_oom.add_resource("huge", 1024 * 1024, transient=True)
    var acc_big = List[RgResourceAccess]()
    acc_big.append(RgResourceAccess(r_big, RG_USAGE_COMPUTE_WRITE))
    _ = rg_oom.add_pass("p_big", acc_big, RG_QUEUE_GRAPHICS)
    assert_false(rg_oom.compile(heap_capacity_bytes=4096))

    # Non-transient resource size clamps to RG_UNIT_BYTES and does not count as transient
    var rg_perm = AaaRenderGraph()
    var r_perm = rg_perm.add_resource("tiny_perm", 1, transient=False)
    assert_equal(rg_perm.resources[r_perm].size_bytes, RG_UNIT_BYTES)
    var acc_perm = List[RgResourceAccess]()
    acc_perm.append(RgResourceAccess(r_perm, RG_USAGE_COMPUTE_WRITE))
    _ = rg_perm.add_pass("p_perm", acc_perm, RG_QUEUE_GRAPHICS)
    assert_true(rg_perm.compile(heap_capacity_bytes=4096))
    assert_equal(rg_perm.aliasing_metrics().unaliased_bytes, 0)


def test_fa4_t2_aaa_renderer_zero_draw_headless_safe() raises:
    var ren = FlintAaaVulkanRenderer(
        width=64,
        height=64,
        tsr_scale=1.0,
        max_draws=256,
        max_instances=256,
        env_cube_dim=16,
    )
    if ren.active:
        ren.begin_frame()
        assert_true(ren.render_frame(capture_readback=True))
        assert_equal(len(ren.readback_pixels()), 64 * 64)
        assert_true(ren.resize(0, 0))
        assert_true(ren.output_w >= 16 and ren.output_h >= 16)
    ren.shutdown()


def test_fa4_t2_flip_zero_dims_and_length_mismatch() raises:
    var dummy = List[UInt32](length=16, fill=0xFF102030)
    assert_equal(len(flip_error_map(dummy, dummy, 0, 0)), 0)
    assert_equal(len(flip_error_map(dummy, dummy, -4, 4)), 0)
    assert_equal(len(flip_error_map(dummy, dummy, 8, 8)), 0)  # 16 < 64
    assert_equal(flip_mean(List[Float32]()), Float64(1.0))
    assert_equal(len(flip_heat_image(List[Float32]())), 0)


def test_fa4_t2_image_diff_0x0_empty_and_mismatch() raises:
    var d0 = compare_pixels(List[UInt32](), List[UInt32](), 0, 0, 2)
    assert_false(d0.same_size)
    assert_false(d0.passes(0, 1.0))
    assert_equal(d0.summary(), "size_mismatch")

    var short_a = List[UInt32](length=10, fill=0xFF000000)
    var full_b = List[UInt32](length=256, fill=0xFF000000)
    var d_short = compare_pixels(short_a, full_b, 16, 16, 2)
    assert_false(d_short.same_size)
    assert_false(d_short.passes(1000000, 1.0))

    # Tolerance / spread / FLIP threshold interaction
    var img_a = List[UInt32](length=64, fill=0xFF808080)
    var img_b = img_a.copy()
    img_b[0] = 0xFF818080  # 1 pixel off by +1 in R (15,625 ppm)
    var d_tol = compare_pixels(img_a, img_b, 8, 8, tolerance=0)
    assert_true(d_tol.same_size)
    assert_equal(d_tol.over_tolerance, 1)
    assert_false(d_tol.passes(0, 1.0))
    assert_true(d_tol.passes(20000, 1.0))
    assert_false(d_tol.passes(20000, 0.0))


def test_fa4_t2_image_diff_missing_or_corrupt_png() raises:
    var px = List[UInt32](length=32 * 32, fill=0xFF204060)
    var d_missing = compare_with_png(px, 32, 32, "/tmp/nonexistent_flint_e2e_opaque_ref_xyz.png", 2)
    assert_false(d_missing.same_size)
    assert_false(d_missing.passes(1000000, 1.0))

    var pid = Int(external_call["getpid", Int32]())
    var tmp_dir = "/tmp/flint-e2e-opaque-t2-" + String(pid)
    assert_true(make_dirs(tmp_dir))
    var bad_png = tmp_dir + "/corrupt.png"
    var junk: List[UInt8] = [UInt8(0x4E), UInt8(0x4F), UInt8(0x50), UInt8(0x45)]
    assert_true(write_bytes(bad_png, junk))

    var d_corrupt = compare_with_png(px, 32, 32, bad_png, 2)
    _ = remove_file(bad_png)
    var owned_dir = tmp_dir
    _ = external_call["rmdir", Int32](owned_dir.as_c_string_span())

    assert_false(d_corrupt.same_size)
    assert_false(d_corrupt.passes(1000000, 1.0))


def main() raises:
    var passed = 0
    var total = 14

    test_fa3_t2_physics_zero_tiny_negative_dt_and_degenerate_raycast()
    print("[PASS] test_fa3_t2_physics_zero_tiny_negative_dt_and_degenerate_raycast")
    passed += 1

    test_fa3_t2_physics_invalid_ids_filters_and_corrupt_blob()
    print("[PASS] test_fa3_t2_physics_invalid_ids_filters_and_corrupt_blob")
    passed += 1

    test_fa3_t2_kcc_zero_dt_and_steep_cliff_no_wall_surf()
    print("[PASS] test_fa3_t2_kcc_zero_dt_and_steep_cliff_no_wall_surf")
    passed += 1

    test_fa3_t2_vehicle_and_aero_boundaries()
    print("[PASS] test_fa3_t2_vehicle_and_aero_boundaries")
    passed += 1

    test_fa3_t2_xpbd_unpinned_cloth_and_degenerate_constraints()
    print("[PASS] test_fa3_t2_xpbd_unpinned_cloth_and_degenerate_constraints")
    passed += 1

    test_fa3_t2_ragdoll_empty_zero_dt_and_invalid_bone_impulse()
    print("[PASS] test_fa3_t2_ragdoll_empty_zero_dt_and_invalid_bone_impulse")
    passed += 1

    test_fa3_t2_flanim_corrupt_bytes_and_empty_footplant()
    print("[PASS] test_fa3_t2_flanim_corrupt_bytes_and_empty_footplant")
    passed += 1

    test_fa4_t2_scalable_renderer_zero_draw_frame()
    print("[PASS] test_fa4_t2_scalable_renderer_zero_draw_frame")
    passed += 1

    test_fa4_t2_scalable_renderer_tiny_resize_clamp()
    print("[PASS] test_fa4_t2_scalable_renderer_tiny_resize_clamp")
    passed += 1

    test_fa4_t2_aaa_render_graph_invalid_limits_and_empty()
    print("[PASS] test_fa4_t2_aaa_render_graph_invalid_limits_and_empty")
    passed += 1

    test_fa4_t2_aaa_renderer_zero_draw_headless_safe()
    print("[PASS] test_fa4_t2_aaa_renderer_zero_draw_headless_safe")
    passed += 1

    test_fa4_t2_flip_zero_dims_and_length_mismatch()
    print("[PASS] test_fa4_t2_flip_zero_dims_and_length_mismatch")
    passed += 1

    test_fa4_t2_image_diff_0x0_empty_and_mismatch()
    print("[PASS] test_fa4_t2_image_diff_0x0_empty_and_mismatch")
    passed += 1

    test_fa4_t2_image_diff_missing_or_corrupt_png()
    print("[PASS] test_fa4_t2_image_diff_missing_or_corrupt_png")
    passed += 1

    print("[SUMMARY] " + String(passed) + "/" + String(total) + " passed")
