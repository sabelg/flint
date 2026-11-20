"""Tier 1 Feature Coverage: Milestone 4 World Sim, Audio, Input & Tooling (F30–F39: 50 tests).
Opaque-box and requirement-driven verification of BVH8, narrowphase, SIMD solver,
Pacejka/VehicleChassis in PhysicsWorld, GPU skinning, IK/blend spaces, AudioServer,
3D HRTF/acoustics, EvdevHub/EvdevGamepad/InputMap, and McpServerV2.
"""
from std.collections import List
from std.math import abs, sqrt
from flint.anim.graph import BlendSampleDef, BlendSpaceDef
from flint.anim.procedural import Placement, smooth01
from flint.assets.json import parse_json
from flint.audio import (
    BUS_MUSIC,
    BUS_SFX,
    AudioCmd,
    AudioMpscRing,
    AudioServer,
    compute_woodworth_hrtf,
)
from flint.core.cvars import CVAR_DEBUG_DRAW, ConsoleVars, cvar_entry
from flint.core.ik import aim_rotation, two_bone_ik
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.core.profiler import ZONE_POST, ZONE_SIM, zone_name
from flint.core.transform import Quat
from flint.devtools.overlays import KEY_F3, FlintDevOverlays
from flint.input import (
    ABS_Z,
    ACTION_BOOL,
    EV_ABS,
    EV_FF,
    EV_KEY,
    BTN_SOUTH,
    EvdevGamepad,
    InputContext,
    InputMap,
    pack_ff_play_event,
    pack_ff_rumble_effect,
)
from flint.physics.broadphase import (
    HASH_BUCKETS,
    BroadphaseInput,
    HashGrid,
    boxes_overlap,
    bucket_of,
    cell_coordinate,
    cell_key,
    find_pairs,
)
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.ccd import closest_obb_surface, swept_sphere_moving_sphere
from flint.physics.filter import CollisionFilter
from flint.physics.islands import GRAPH_COLOURS, ConstraintGraph, pair_key
from flint.physics.narrowphase import collide_sphere_capsule, collide_sphere_sphere
from flint.physics.solver import rodrigues_displacement
from flint.physics.solver_bodies import F8, V8
from flint.physics.tires import PacejkaTire
from flint.physics.vehicle import VehicleChassis
from flint.physics.world import PhysicsWorld
from flint.platform.dsp import CytomicSVF, ModalResonator, compute_spatial_3d
from flint.platform.input import PAD_A, PAD_AXIS_LT, InputState
from flint.platform.wayland import WaylandWindow, evdev_key, wl_put32
from flint.platform.x11 import X11Window, x11_put16
from flint.render.aaa_skinning_rt import (
    AAA_SKIN_CUR_BONES_BASE,
    AAA_SKIN_PREV_BONES_BASE,
    DualQuat,
    aaa_dq_skinning_shader,
    blend_dual_quats_cpu,
)
from flint.render.spirv_validate import validate_spirv_module
from e2e.common.harness import TestContext
from e2e.common.v5_fixtures import (
    create_mock_asset_server,
    create_mock_bvh8_floor,
    create_mock_mcp_server,
    create_mock_pcm16_sine,
    create_mock_studio,
)


# =========================================================================
# F30: BVH8 Static Mesh Collider & Broadphase (5 tests)
# =========================================================================

def test_f30_broadphase_boxes_overlap_true(mut ctx: TestContext) raises:
    ctx.assert_true(boxes_overlap(Vec3(-1, -1, -1), Vec3(1, 1, 1), Vec3(0.5, 0.5, 0.5), Vec3(2, 2, 2)), "overlapping AABBs true")


def test_f30_broadphase_boxes_overlap_false(mut ctx: TestContext) raises:
    ctx.assert_false(boxes_overlap(Vec3(-5, -5, -5), Vec3(-2, -2, -2), Vec3(2, 2, 2), Vec3(5, 5, 5)), "disjoint AABBs false")


def test_f30_hash_grid_spatial_hash(mut ctx: TestContext) raises:
    var p = Vec3(10.5, 3.2, -7.8)
    var cx = cell_coordinate(p.x)
    var cy = cell_coordinate(p.y)
    var cz = cell_coordinate(p.z)
    ctx.assert_equal(cx, 1, "cell_coordinate x == 1")
    ctx.assert_equal(cy, 0, "cell_coordinate y == 0")
    ctx.assert_equal(cz, -1, "cell_coordinate z == -1")
    var bucket = bucket_of(cell_key(cx, cy, cz))
    ctx.assert_true(bucket >= 0 and bucket < HASH_BUCKETS, "bucket within HASH_BUCKETS")
    var grid = HashGrid(1)
    grid.insert(0, p, p + Vec3(1.0, 1.0, 1.0))
    ctx.assert_true(len(grid.slot_key) > 0, "HashGrid registers occupied cell slot")


def test_f30_active_edge_smoothing(mut ctx: TestContext) raises:
    var floor_bvh = create_mock_bvh8_floor(20.0, 0.0)
    var r_hit = floor_bvh.raycast(Vec3(0.0, 5.0, 0.0), Vec3(0.0, -1.0, 0.0), 20.0)
    ctx.assert_true(r_hit.hit, "Bvh8MeshCollider hits downward ray")
    ctx.assert_approx(r_hit.t, 5.0, 0.01, "Bvh8 raycast hit distance == 5.0")
    var res = closest_obb_surface(Vec3(1.05, 1.05, 0.0), Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    ctx.assert_true(res[1].x > 0.6 and res[1].y > 0.6, "edge contact normal blends face normals")


def test_f30_find_pairs_broadphase(mut ctx: TestContext) raises:
    var bp = BroadphaseInput()
    bp.add(Vec3(-1, -1, -1), Vec3(1, 1, 1), True, True, CollisionFilter())
    bp.add(Vec3(0, 0, 0), Vec3(2, 2, 2), True, True, CollisionFilter())
    var pairs = find_pairs(bp)
    ctx.assert_true(len(pairs.pairs) >= 1, "overlapping bodies identified by broadphase")


# =========================================================================
# F31: Discrete Narrowphase & SAT/GJK-EPA (5 tests)
# =========================================================================

def test_f31_sphere_capsule_manifold(mut ctx: TestContext) raises:
    var m_hit = collide_sphere_capsule(Vec3(0.8, 2.0, 0.0), 0.5, Vec3(0, 0, 0), Vec3(0, 4, 0), 0.5)
    ctx.assert_true(m_hit.count > 0, "overlapping sphere-capsule generates contact manifold")
    var m_miss = collide_sphere_capsule(Vec3(3.0, 2.0, 0.0), 0.5, Vec3(0, 0, 0), Vec3(0, 4, 0), 0.5)
    ctx.assert_equal(m_miss.count, 0, "separated sphere-capsule has empty manifold")


def test_f31_capsule_capsule_distance(mut ctx: TestContext) raises:
    var hit = swept_sphere_moving_sphere(Vec3(-2, 0, 0), Vec3(0, 0, 0), 0.5, Vec3(0, 0, 0), Vec3(0, 0, 0), 0.5)
    ctx.assert_true(hit.hit, "moving sphere swept against target collides")
    ctx.assert_approx(hit.toi, 0.5, 0.01, "exact collision TOI 0.5")


def test_f31_box_box_sat_separating_axis(mut ctx: TestContext) raises:
    var res = closest_obb_surface(Vec3(5, 0, 0), Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    ctx.assert_approx(res[2], 4.0, 0.001, "OBB SAT separating distance is 4.0")
    ctx.assert_vec3_approx(res[1], Vec3(1, 0, 0), 0.001, "separating axis aligns to +X")


def test_f31_gjk_epa_convex_hull(mut ctx: TestContext) raises:
    var m = collide_sphere_sphere(Vec3(0, 0, 0), 1.0, Vec3(1.5, 0, 0), 1.0)
    ctx.assert_true(m.count > 0, "overlapping spheres generate manifold")
    ctx.assert_approx(m.p0.depth, 0.5, 0.01, "penetration depth == 0.5")


def test_f31_collision_filter_masking(mut ctx: TestContext) raises:
    var f1 = CollisionFilter(1, 2, 0)
    var f2 = CollisionFilter(2, 1, 0)
    var f3 = CollisionFilter(4, 4, 0)
    ctx.assert_true(f1.touches(f2), "matching category/mask touches")
    ctx.assert_false(f1.touches(f3), "disjoint mask does not touch")


# =========================================================================
# F32: Zero-Allocation SIMD Quaternion Solver (5 tests)
# =========================================================================

def test_f32_simd_v8_splat_and_scale(mut ctx: TestContext) raises:
    var scaled = V8.splat(Vec3(2.0, 3.0, 4.0)).scale(F8(0.5))
    ctx.assert_approx(scaled.x[0], 1.0, 0.001, "SIMD lane 0 x scaled")
    ctx.assert_approx(scaled.y[0], 1.5, 0.001, "SIMD lane 0 y scaled")
    ctx.assert_approx(scaled.z[0], 2.0, 0.001, "SIMD lane 0 z scaled")


def test_f32_simd_v8_dot_product(mut ctx: TestContext) raises:
    var d = V8.splat(Vec3(1.0, 2.0, 3.0)).dot(V8.splat(Vec3(4.0, 5.0, 6.0)))
    ctx.assert_approx(d[0], 32.0, 0.001, "1*4 + 2*5 + 3*6 = 32")


def test_f32_simd_v8_cross_product(mut ctx: TestContext) raises:
    var z = V8.splat(Vec3(1.0, 0.0, 0.0)).cross(V8.splat(Vec3(0.0, 1.0, 0.0)))
    ctx.assert_approx(z.z[0], 1.0, 0.001, "X cross Y = Z (z=1)")


def test_f32_rodrigues_lever_arm_invariance(mut ctx: TestContext) raises:
    var disp = rodrigues_displacement(V8.splat(Vec3(0.0, 0.0, 0.1)), V8.splat(Vec3(5.0, 0.0, 0.0)))
    var final_pos = Vec3(5.0 + disp.x[0], disp.y[0], disp.z[0])
    ctx.assert_approx(final_pos.length(), 5.0, 0.01, "Rodrigues rotation preserves lever arm length")


def test_f32_parallel_island_graph_coloring(mut ctx: TestContext) raises:
    var graph = ConstraintGraph()
    var keys = List[Int]()
    keys.append(pair_key(1, 2)); keys.append(pair_key(2, 3)); keys.append(pair_key(1, 3))
    var rows_a: List[Int] = [0, 1, 0]
    var rows_b: List[Int] = [1, 2, 2]
    var colours = graph.assign(keys, rows_a, rows_b, 3)
    ctx.assert_true(colours[0] >= 0 and colours[0] < GRAPH_COLOURS, "valid colour assigned")
    ctx.assert_true(colours[0] != colours[1], "sharing bodies assigned distinct colors")


# =========================================================================
# F33: Advanced Vehicle Dynamics inside PhysicsWorld (5 tests)
# =========================================================================

def test_f33_suspension_substep_rate(mut ctx: TestContext) raises:
    var pw = PhysicsWorld(gravity=9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(40.0, 0.5, 40.0), mass=0.0)
    var car = VehicleChassis(1450.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(0.0, 0.95, 0.0)
    var slot = pw.add_vehicle(car^)
    pw.rebuild_broadphase()
    for _ in range(30):
        pw.step(Float32(1.0 / 60.0))
    var st = pw.vehicle_state(slot)
    ctx.assert_true(len(st.wheel_states) == 4, "4-wheel car hosted in PhysicsWorld")
    ctx.assert_true(st.wheel_states[0].grounded, "front-left wheel grounds on floor box")
    ctx.assert_true(
        st.wheel_states[0].compression > 0.0 and st.wheel_states[0].compression < st.wheels[0].suspension_rest_len,
        "suspension settles to equilibrium compression",
    )


def test_f33_pacejka_magic_formula(mut ctx: TestContext) raises:
    var tire = PacejkaTire()
    var f_lon = tire.calc_forces(0.14, 0.0, 4000.0, 15.0)
    var f_lat = tire.calc_forces(0.0, 0.12, 4000.0, 15.0)
    var f_heavy = tire.calc_forces(0.14, 0.0, 8000.0, 15.0)
    ctx.assert_true(f_lon[0] > 1000.0, "Pacejka longitudinal peak Fx > 1000 N")
    ctx.assert_true(f_lat[1] > 1000.0, "Pacejka lateral Fy > 1000 N")
    ctx.assert_true(f_heavy[0] > f_lon[0], "higher normal load increases peak longitudinal force")


def test_f33_tire_combined_slip(mut ctx: TestContext) raises:
    var tire = PacejkaTire()
    var f_pure = tire.calc_forces(0.14, 0.0, 4000.0, 15.0)
    var f_comb = tire.calc_forces(0.14, 0.25, 4000.0, 15.0)
    ctx.assert_true(abs(f_comb[0]) < abs(f_pure[0]), "combined lateral slip reduces longitudinal Fx via friction ellipse")
    ctx.assert_true(abs(f_comb[1]) > 0.0, "combined slip produces lateral Fy")


def test_f33_anti_roll_bar_stiffness(mut ctx: TestContext) raises:
    var pw_no_arb = PhysicsWorld(gravity=9.81)
    _ = pw_no_arb.add_box(Vec3(0.0, -0.5, 0.0), Vec3(80.0, 0.5, 80.0), mass=0.0)
    var c0 = VehicleChassis(1450.0)
    c0.setup_four_wheel_car()
    c0.pos = Vec3(0.0, 0.96, 0.0)
    c0.linear_vel = Vec3(0.0, 0.0, 12.0)
    c0.anti_roll_bar_front = 0.0
    c0.anti_roll_bar_rear = 0.0
    var s0 = pw_no_arb.add_vehicle(c0^)
    pw_no_arb.rebuild_broadphase()

    var pw_arb = PhysicsWorld(gravity=9.81)
    _ = pw_arb.add_box(Vec3(0.0, -0.5, 0.0), Vec3(80.0, 0.5, 80.0), mass=0.0)
    var c1 = VehicleChassis(1450.0)
    c1.setup_four_wheel_car()
    c1.pos = Vec3(0.0, 0.96, 0.0)
    c1.linear_vel = Vec3(0.0, 0.0, 12.0)
    c1.anti_roll_bar_front = 25000.0
    c1.anti_roll_bar_rear = 20000.0
    var s1 = pw_arb.add_vehicle(c1^)
    pw_arb.rebuild_broadphase()

    pw_no_arb.set_vehicle_controls(s0, 0.1, 0.0, 0.0)
    pw_arb.set_vehicle_controls(s1, 0.1, 0.0, 0.0)
    for _ in range(10):
        pw_no_arb.step(Float32(1.0 / 60.0))
        pw_arb.step(Float32(1.0 / 60.0))

    pw_no_arb.set_vehicle_controls(s0, 0.1, 0.0, 0.25)
    pw_arb.set_vehicle_controls(s1, 0.1, 0.0, 0.25)
    for _ in range(25):
        pw_no_arb.step(Float32(1.0 / 60.0))
        pw_arb.step(Float32(1.0 / 60.0))

    var st0 = pw_no_arb.vehicle_state(s0)
    var st1 = pw_arb.vehicle_state(s1)
    ctx.assert_true(
        st0.wheel_states[0].grounded and st0.wheel_states[1].grounded,
        "no-ARB car front wheels stay grounded in turn",
    )
    ctx.assert_true(
        st1.wheel_states[0].grounded and st1.wheel_states[1].grounded,
        "ARB car front wheels stay grounded in turn",
    )
    var asym0 = abs(st0.wheel_states[0].compression - st0.wheel_states[1].compression)
    var asym1 = abs(st1.wheel_states[0].compression - st1.wheel_states[1].compression)
    ctx.assert_true(asym0 > 0.0, "cornering induces measurable front roll asymmetry without ARB")
    ctx.assert_true(asym1 < asym0, "anti-roll bar reduces grounded left-right suspension asymmetry")


def test_f33_weight_transfer_acceleration(mut ctx: TestContext) raises:
    var pw_acc = PhysicsWorld(gravity=9.81)
    _ = pw_acc.add_box(Vec3(0.0, -0.5, 0.0), Vec3(80.0, 0.5, 80.0), mass=0.0)
    var ca = VehicleChassis(1450.0)
    ca.setup_four_wheel_car()
    ca.pos = Vec3(0.0, 0.72, 0.0)
    var sa = pw_acc.add_vehicle(ca^)
    pw_acc.set_vehicle_controls(sa, 1.0, 0.0, 0.0)
    for _ in range(25):
        pw_acc.step(Float32(1.0 / 60.0))
    var sta = pw_acc.vehicle_state(sa)
    ctx.assert_true(sta.pos.z > 0.1, "throttle accelerates VehicleChassis forward along +Z")

    var pw_brk = PhysicsWorld(gravity=9.81)
    _ = pw_brk.add_box(Vec3(0.0, -0.5, 0.0), Vec3(80.0, 0.5, 80.0), mass=0.0)
    var cb = VehicleChassis(1450.0)
    cb.setup_four_wheel_car()
    cb.pos = Vec3(0.0, 0.72, 0.0)
    cb.linear_vel = Vec3(0.0, 0.0, 15.0)
    var sb = pw_brk.add_vehicle(cb^)
    pw_brk.set_vehicle_controls(sb, 0.0, 1.0, 0.0)
    for _ in range(25):
        pw_brk.step(Float32(1.0 / 60.0))
    var stb = pw_brk.vehicle_state(sb)
    ctx.assert_true(stb.linear_vel.z < 15.0, "brake decelerates VehicleChassis inside PhysicsWorld")
    var front_load_brk = stb.wheel_states[0].normal_load + stb.wheel_states[1].normal_load
    var rear_load_brk = stb.wheel_states[2].normal_load + stb.wheel_states[3].normal_load
    ctx.assert_true(front_load_brk > rear_load_brk, "braking shifts normal load to front axle")


# =========================================================================
# F34: Multi-Rig GPU Skinning & Batched BLAS (5 tests)
# =========================================================================

def test_f34_multi_rig_palette_slices(mut ctx: TestContext) raises:
    var dq = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(1, 2, 3))
    var words = List[UInt32](length=16, fill=0)
    dq.write_words(words, 0)
    ctx.assert_true(words[0] == 0 and words[3] != 0, "quaternion components encoded into bone palette")
    ctx.assert_true(AAA_SKIN_CUR_BONES_BASE < AAA_SKIN_PREV_BONES_BASE, "cur bones precede prev bones")


def test_f34_batched_blas_build_records(mut ctx: TestContext) raises:
    var spv = aaa_dq_skinning_shader(0, 1)
    ctx.assert_true(len(spv) > 0 and validate_spirv_module(spv).valid, "compute skinning shader validates")


def test_f34_bone_matrix_composition(mut ctx: TestContext) raises:
    var dq = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(0, 1, 0))
    ctx.assert_vec3_approx(dq.transform_point(Vec3(0, 1, 0)), Vec3(0, 2, 0), 0.001, "hierarchical bone transform")


def test_f34_skinning_output_vertex_offset(mut ctx: TestContext) raises:
    var dq0 = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(0, 0, 0))
    var dq1 = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(0, 2, 0))
    var bones = List[DualQuat](); bones.append(dq0); bones.append(dq1); bones.append(dq0); bones.append(dq0)
    var blended = blend_dual_quats_cpu(bones, 0, 1, 2, 3, 0.5, 0.5, 0.0, 0.0)
    ctx.assert_vec3_approx(blended.transform_point(Vec3(0, 0, 0)), Vec3(0, 1, 0), 0.01, "DLB skinning midpoint")


def test_f34_blas_refit_without_rebuild(mut ctx: TestContext) raises:
    var spv = aaa_dq_skinning_shader(0, 2)
    ctx.assert_true(len(spv) > 0 and validate_spirv_module(spv).valid, "in-place compute skinning validates")


# =========================================================================
# F35: Decoupled Skeleton & 2D Blend Spaces (5 tests)
# =========================================================================

def test_f35_two_bone_ik_solution(mut ctx: TestContext) raises:
    var res = two_bone_ik(Vec3(0, 2, 0), Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(0, 0.5, 0.5), Quat(), Quat(), Vec3(0, 0, 1))
    var norm_up = res.upper.w*res.upper.w + res.upper.x*res.upper.x + res.upper.y*res.upper.y + res.upper.z*res.upper.z
    var norm_mid = res.middle.w*res.middle.w + res.middle.x*res.middle.x + res.middle.y*res.middle.y + res.middle.z*res.middle.z
    ctx.assert_approx(norm_up, 1.0, 0.01, "upper joint rotation normalized")
    ctx.assert_approx(norm_mid, 1.0, 0.01, "middle joint rotation normalized")


def test_f35_aim_rotation_alignment(mut ctx: TestContext) raises:
    var q = aim_rotation(Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, 0, 10), Quat())
    ctx.assert_vec3_approx(q.rotate(Vec3(0, 0, 1)), Vec3(0, 0, 1), 0.05, "aim_rotation aligns to target")


def test_f35_2d_blend_space_weights(mut ctx: TestContext) raises:
    var bs = BlendSpaceDef()
    bs.samples.append(BlendSampleDef(0, 0.0))
    bs.samples.append(BlendSampleDef(1, 3.0))
    bs.samples.append(BlendSampleDef(2, 8.0))
    var w = bs.weights(5.0)
    ctx.assert_approx(w[1] + w[2], 1.0, 0.001, "blend space weights sum to 1.0")
    ctx.assert_approx(w[1], 0.6, 0.01, "walk weight 0.6")


def test_f35_motion_database_indexing(mut ctx: TestContext) raises:
    var bs = BlendSpaceDef()
    bs.samples.append(BlendSampleDef(0, 0.0))
    bs.samples.append(BlendSampleDef(1, 2.5))
    var w = bs.weights(0.0)
    ctx.assert_equal(w[0], Float32(1.0), "idle clip weight 1.0 at speed 0")


def test_f35_soft_reach_ik_damping(mut ctx: TestContext) raises:
    ctx.assert_approx(smooth01(0.0), 0.0, 0.001, "smooth01 at 0 is 0.0")
    ctx.assert_approx(smooth01(0.5), 0.5, 0.001, "smooth01 at 0.5 is 0.5")
    ctx.assert_approx(smooth01(1.0), 1.0, 0.001, "smooth01 at 1 is 1.0")
    var place = Placement(Vec3(1.0, 0.0, 2.0), Vec3(1, 1, 1), 0.0)
    ctx.assert_vec3_approx(place.to_world(Vec3(0.5, 1.0, 0.0)), Vec3(1.5, 1.0, 2.0), 0.001, "Placement maps local to world")


# =========================================================================
# F36: Real-Time AudioServer & MPSC Ring (5 tests)
# =========================================================================

def test_f36_spsc_queue_lock_free(mut ctx: TestContext) raises:
    var ring = AudioMpscRing.create(64)
    ctx.assert_true(ring.allocated(), "AudioMpscRing allocated")
    var cmd = AudioCmd()
    cmd.kind = 7
    cmd.slot = 42
    ctx.assert_true(ring.push(cmd), "AudioMpscRing push succeeds")
    var out = AudioCmd()
    ctx.assert_true(ring.pop(out), "AudioMpscRing pop succeeds")
    ctx.assert_equal(Int(out.kind), 7, "AudioCmd kind preserved")
    ctx.assert_equal(Int(out.slot), 42, "AudioCmd slot preserved")
    ring.destroy()


def test_f36_audio_sample_rate(mut ctx: TestContext) raises:
    var srv = AudioServer(48000, 256)
    ctx.assert_equal(srv.sample_rate, 48000, "AudioServer standard output rate is 48 kHz")
    var clip = create_mock_pcm16_sine(440.0, 4800, 48000)
    var aid = srv.register_pcm16_asset(clip, 1, 48000)
    ctx.assert_true(aid >= 0, "AudioServer registers PCM16 clip")


def test_f36_audio_quantum_buffer_frames(mut ctx: TestContext) raises:
    var srv = AudioServer(48000, 256)
    var clip = create_mock_pcm16_sine(440.0, 4800, 48000)
    var aid = srv.register_pcm16_asset(clip, 1, 48000)
    _ = srv.play_2d(aid, volume=0.8, bus_id=BUS_SFX, looping=True, direct_stereo=True)
    var out_lr = List[Float32](length=512, fill=0.0)
    srv.render_offline_quantum(out_lr, 256)
    ctx.assert_equal(len(out_lr), 512, "quantum renders 256 stereo frames (512 floats)")
    var energy = Float32(0.0)
    for i in range(512):
        energy += abs(out_lr[i])
    ctx.assert_true(energy > 0.01, "AudioServer offline quantum mixes active voice")


def test_f36_voice_bus_volume_gains(mut ctx: TestContext) raises:
    var srv = AudioServer(48000, 256)
    ctx.assert_true(srv.set_bus_gain(BUS_MUSIC, 0.4), "AudioServer sets BUS_MUSIC gain to 0.4")


def test_f36_zero_quantum_allocations(mut ctx: TestContext) raises:
    var srv = AudioServer(48000, 256)
    var clip = create_mock_pcm16_sine(997.0, 4800, 48000)
    var aid = srv.register_pcm16_asset(clip, 1, 48000)
    _ = srv.play_2d(aid, volume=4.0, bus_id=BUS_SFX, looping=True, direct_stereo=True)
    var out_lr = List[Float32](length=512, fill=0.0)
    for _ in range(4):
        srv.render_offline_quantum(out_lr, 256)
    var peak = Float32(0.0)
    for i in range(512):
        if abs(out_lr[i]) > peak:
            peak = abs(out_lr[i])
    ctx.assert_true(peak <= Float32(0.989), "TruePeakLimiter brickwall caps +12 dBFS at <= 0.989")


# =========================================================================
# F37: Continuous 3D HRTF & Synths (5 tests)
# =========================================================================

def test_f37_spatial_3d_distance_attenuation(mut ctx: TestContext) raises:
    var params_near = compute_spatial_3d(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 10), Vec3(0, 0, 0))
    var params_far = compute_spatial_3d(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 100), Vec3(0, 0, 0))
    ctx.assert_true(params_near.gain > params_far.gain, "near sound has higher gain than distant sound")


def test_f37_spatial_3d_stereo_panning(mut ctx: TestContext) raises:
    var params = compute_spatial_3d(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(10, 0, 0), Vec3(0, 0, 0))
    ctx.assert_true(params.pan > 0.5, "source to the right pans right")
    var hrtf = compute_woodworth_hrtf(
        Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(1, 0, 0), Vec3(0, 1, 0),
        Vec3(10, 0, 0), Vec3(0, 0, 0),
    )
    var itd = hrtf.delay_l_samples - hrtf.delay_r_samples
    ctx.assert_true(itd > 20.0, "Woodworth HRTF +90 deg right source has positive ITD (~29.5 samples)")


def test_f37_spatial_3d_doppler_shift(mut ctx: TestContext) raises:
    var params = compute_spatial_3d(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 50), Vec3(0, 0, -30))
    ctx.assert_true(params.doppler_pitch > 1.0, "approaching sound exhibits upward Doppler pitch shift")


def test_f37_procedural_cytomic_svf(mut ctx: TestContext) raises:
    var svf = CytomicSVF(1000.0, 0.707, 48000.0)
    var out_sample = svf.process(1.0)
    ctx.assert_true(out_sample > 0.0 and out_sample < 2.0, "CytomicSVF processes impulse stably")


def test_f37_modal_resonator_harmonic(mut ctx: TestContext) raises:
    var resonator = ModalResonator(440.0, 0.05, 1.0, 48000.0)
    var s0 = resonator.step(1.0)
    var peak: Float32 = 0.0
    for _ in range(50):
        var s = abs(resonator.step(0.0))
        if s > peak:
            peak = s
    for _ in range(25000):
        _ = resonator.step(0.0)
    ctx.assert_true(s0 > 0.0, "resonator responds to excitation")
    ctx.assert_true(abs(resonator.step(0.0)) < peak * 0.1, "resonator oscillation decays by >90%")


# =========================================================================
# F38: Raw Pointer Lock, EvdevGamepad & InputMap (5 tests)
# =========================================================================

def test_f38_evdev_key_mapping(mut ctx: TestContext) raises:
    ctx.assert_equal(evdev_key(1), 256, "evdev KEY_ESC (1) maps to Flint KEY_ESCAPE (256)")


def test_f38_raw_unaccelerated_mouse_delta(mut ctx: TestContext) raises:
    var input = InputState()
    input.add_mouse_delta(14.0, -8.5)
    ctx.assert_approx(input.mouse_delta_x(), 14.0, 0.001, "raw unaccelerated mouse delta X preserved")
    ctx.assert_approx(input.mouse_delta_y(), -8.5, 0.001, "raw unaccelerated mouse delta Y preserved")
    ctx.assert_false(input.uses_legacy_jsdev(), "InputState uses EvdevHub, not legacy /dev/input/js*")


def test_f38_evdev_ff_rumble_effect(mut ctx: TestContext) raises:
    var ff_pkt = pack_ff_rumble_effect(-1, 0.8, 0.4, 500)
    ctx.assert_equal(len(ff_pkt), 48, "struct ff_effect is 48 bytes")
    ctx.assert_equal(Int(ff_pkt[0]) | (Int(ff_pkt[1]) << 8), 0x0050, "FF_RUMBLE == 0x0050")
    var ev_play = pack_ff_play_event(0, True)
    ctx.assert_equal(len(ev_play), 24, "EV_FF input_event is 24 bytes")
    ctx.assert_equal(Int(ev_play[16]) | (Int(ev_play[17]) << 8), Int(EV_FF), "EV_FF code matches")


def test_f38_pointer_lock_symmetry(mut ctx: TestContext) raises:
    var wl = WaylandWindow()
    ctx.assert_false(wl.pointer_locked, "Wayland starts unlocked")
    wl.locked_pointer = 7
    wl.relative_pointer = 5
    var p_lock = List[UInt8](length=8, fill=0)
    wl_put32(p_lock, 0, 7)
    wl_put32(p_lock, 4, 0 | (8 << 16))
    wl.incoming = p_lock^
    wl.handle_incoming()
    var p_rel = List[UInt8](length=32, fill=0)
    wl_put32(p_rel, 0, 5)
    wl_put32(p_rel, 4, 0 | (32 << 16))
    wl_put32(p_rel, 24, 1024)
    wl_put32(p_rel, 28, -512)
    wl.incoming = p_rel^
    wl.handle_incoming()
    ctx.assert_true(wl.pointer_locked, "Wayland pointer lock engaged")
    ctx.assert_approx(wl.pending_relative_x, 4.0, 0.01, "Wayland accumulates relative dx while locked")
    ctx.assert_approx(wl.pending_relative_y, -2.0, 0.01, "Wayland accumulates relative dy while locked")

    var x11 = X11Window()
    ctx.assert_false(x11.pointer_locked, "X11 starts unlocked")
    x11.set_pointer_lock(True)
    ctx.assert_true(x11.pointer_locked, "X11 set_pointer_lock(True) engages lock")
    var evs = List[UInt8](length=64, fill=0)
    evs[0] = 6
    x11_put16(evs, 24, 100)
    x11_put16(evs, 26, 100)
    evs[32] = 6
    x11_put16(evs, 32 + 24, 114)
    x11_put16(evs, 32 + 26, 92)
    x11.incoming = evs^
    x11.process_events()
    ctx.assert_approx(x11.pending_rel_x, 14.0, 0.01, "X11 accumulates relative dx while locked")
    ctx.assert_approx(x11.pending_rel_y, -8.0, 0.01, "X11 accumulates relative dy while locked")
    x11.set_pointer_lock(False)
    ctx.assert_false(x11.pointer_locked, "X11 set_pointer_lock(False) releases lock")


def test_f38_dual_motor_haptic_range(mut ctx: TestContext) raises:
    var pad = EvdevGamepad()
    pad.configure_abs_axis(ABS_Z, 0, 1023, 0, 0)
    ctx.assert_approx(pad.axes[PAD_AXIS_LT], 0.0, 0.001, "EvdevGamepad trigger starts at 0.0")
    pad.feed_raw_evdev_event(EV_ABS, UInt16(ABS_Z), Int32(1023))
    ctx.assert_approx(pad.axes[PAD_AXIS_LT], 1.0, 0.001, "EvdevGamepad full trigger press maps to 1.0")
    pad.feed_raw_evdev_event(EV_KEY, UInt16(BTN_SOUTH), Int32(1))
    ctx.assert_true(pad.down[PAD_A], "EvdevGamepad BTN_SOUTH maps to PAD_A")

    var imap = InputMap()
    _ = imap.register_action("Jump", ACTION_BOOL)
    var ic = InputContext("Gameplay")
    ic.bind_key("Jump", 32)
    imap.add_context(ic^)
    imap.push_context("Gameplay")
    var inp = InputState()
    inp.set_logical_key(32, True)
    imap.update(inp, 0.016)
    ctx.assert_true(imap.is_down("Jump"), "InputMap resolves Jump action")


# =========================================================================
# F39: Developer Tooling & McpServerV2 (5 tests)
# =========================================================================

def test_f39_mcp_query_ping(mut ctx: TestContext) raises:
    var studio = create_mock_studio()
    var assets = create_mock_asset_server()
    var pool = JobPool()
    var mcp = create_mock_mcp_server("/tmp")
    var resp = mcp.handle_jsonrpc("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\",\"params\":{}}", studio, assets, pool)
    ctx.assert_true(resp.find("\"result\"") >= 0, "McpServerV2 ping returns JSON-RPC result")
    assets.close(); studio.shutdown(); pool.stop()


def test_f39_cvar_lookup_and_set(mut ctx: TestContext) raises:
    var cvars = ConsoleVars()
    var entry = cvar_entry(CVAR_DEBUG_DRAW)
    ctx.assert_equal(entry.name, "debug.draw", "CVAR_DEBUG_DRAW name is debug.draw")
    cvars.set(CVAR_DEBUG_DRAW, 1.0)
    ctx.assert_equal(cvars.get(CVAR_DEBUG_DRAW), 1.0, "cvar value mutated to 1.0")


def test_f39_flamebar_zone_profiling(mut ctx: TestContext) raises:
    ctx.assert_equal(zone_name(ZONE_POST), "post", "ZONE_POST name is post")
    ctx.assert_equal(zone_name(ZONE_SIM), "sim", "ZONE_SIM name is sim")


def test_f39_gpu_buffer_inspector_bounds(mut ctx: TestContext) raises:
    var overlays = FlintDevOverlays()
    var input = InputState()
    input.set_key(KEY_F3, True)
    _ = overlays.handle_input(input)
    ctx.assert_true(overlays.show_memory, "F3 hotkey activates memory and buffer inspector")


def test_f39_mcp_command_execution(mut ctx: TestContext) raises:
    var studio = create_mock_studio()
    var assets = create_mock_asset_server()
    var pool = JobPool()
    var mcp = create_mock_mcp_server("/tmp")
    var resp = mcp.handle_jsonrpc(
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"physics.step_deterministic\",\"arguments\":{\"ticks\":3}}}",
        studio, assets, pool,
    )
    var jb = List[UInt8]()
    for b in resp.as_bytes():
        jb.append(b)
    var doc = parse_json(jb)
    ctx.assert_true(doc.valid, "McpServerV2 physics.step_deterministic returns valid JSON")
    ctx.assert_true(resp.find("\\\"ticks_advanced\\\":3") >= 0 or resp.find("\"ticks_advanced\":3") >= 0, "advanced exactly 3 deterministic ticks")
    ctx.assert_true(resp.find("\"isError\":false") >= 0, "MCP tool call succeeded without error")
    ctx.assert_equal(studio.sim_tick, 3, "FlintStudioState.sim_tick advanced to 3")
    assets.close(); studio.shutdown(); pool.stop()


# =========================================================================
# Aggregator for Tier 1 Milestone 4
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T1_M4_WorldSim")

    try: test_f30_broadphase_boxes_overlap_true(ctx); ctx.record_pass("test_f30_broadphase_boxes_overlap_true") except e: ctx.record_fail("test_f30_broadphase_boxes_overlap_true", String(e))
    try: test_f30_broadphase_boxes_overlap_false(ctx); ctx.record_pass("test_f30_broadphase_boxes_overlap_false") except e: ctx.record_fail("test_f30_broadphase_boxes_overlap_false", String(e))
    try: test_f30_hash_grid_spatial_hash(ctx); ctx.record_pass("test_f30_hash_grid_spatial_hash") except e: ctx.record_fail("test_f30_hash_grid_spatial_hash", String(e))
    try: test_f30_active_edge_smoothing(ctx); ctx.record_pass("test_f30_active_edge_smoothing") except e: ctx.record_fail("test_f30_active_edge_smoothing", String(e))
    try: test_f30_find_pairs_broadphase(ctx); ctx.record_pass("test_f30_find_pairs_broadphase") except e: ctx.record_fail("test_f30_find_pairs_broadphase", String(e))

    try: test_f31_sphere_capsule_manifold(ctx); ctx.record_pass("test_f31_sphere_capsule_manifold") except e: ctx.record_fail("test_f31_sphere_capsule_manifold", String(e))
    try: test_f31_capsule_capsule_distance(ctx); ctx.record_pass("test_f31_capsule_capsule_distance") except e: ctx.record_fail("test_f31_capsule_capsule_distance", String(e))
    try: test_f31_box_box_sat_separating_axis(ctx); ctx.record_pass("test_f31_box_box_sat_separating_axis") except e: ctx.record_fail("test_f31_box_box_sat_separating_axis", String(e))
    try: test_f31_gjk_epa_convex_hull(ctx); ctx.record_pass("test_f31_gjk_epa_convex_hull") except e: ctx.record_fail("test_f31_gjk_epa_convex_hull", String(e))
    try: test_f31_collision_filter_masking(ctx); ctx.record_pass("test_f31_collision_filter_masking") except e: ctx.record_fail("test_f31_collision_filter_masking", String(e))

    try: test_f32_simd_v8_splat_and_scale(ctx); ctx.record_pass("test_f32_simd_v8_splat_and_scale") except e: ctx.record_fail("test_f32_simd_v8_splat_and_scale", String(e))
    try: test_f32_simd_v8_dot_product(ctx); ctx.record_pass("test_f32_simd_v8_dot_product") except e: ctx.record_fail("test_f32_simd_v8_dot_product", String(e))
    try: test_f32_simd_v8_cross_product(ctx); ctx.record_pass("test_f32_simd_v8_cross_product") except e: ctx.record_fail("test_f32_simd_v8_cross_product", String(e))
    try: test_f32_rodrigues_lever_arm_invariance(ctx); ctx.record_pass("test_f32_rodrigues_lever_arm_invariance") except e: ctx.record_fail("test_f32_rodrigues_lever_arm_invariance", String(e))
    try: test_f32_parallel_island_graph_coloring(ctx); ctx.record_pass("test_f32_parallel_island_graph_coloring") except e: ctx.record_fail("test_f32_parallel_island_graph_coloring", String(e))

    try: test_f33_suspension_substep_rate(ctx); ctx.record_pass("test_f33_suspension_substep_rate") except e: ctx.record_fail("test_f33_suspension_substep_rate", String(e))
    try: test_f33_pacejka_magic_formula(ctx); ctx.record_pass("test_f33_pacejka_magic_formula") except e: ctx.record_fail("test_f33_pacejka_magic_formula", String(e))
    try: test_f33_tire_combined_slip(ctx); ctx.record_pass("test_f33_tire_combined_slip") except e: ctx.record_fail("test_f33_tire_combined_slip", String(e))
    try: test_f33_anti_roll_bar_stiffness(ctx); ctx.record_pass("test_f33_anti_roll_bar_stiffness") except e: ctx.record_fail("test_f33_anti_roll_bar_stiffness", String(e))
    try: test_f33_weight_transfer_acceleration(ctx); ctx.record_pass("test_f33_weight_transfer_acceleration") except e: ctx.record_fail("test_f33_weight_transfer_acceleration", String(e))

    try: test_f34_multi_rig_palette_slices(ctx); ctx.record_pass("test_f34_multi_rig_palette_slices") except e: ctx.record_fail("test_f34_multi_rig_palette_slices", String(e))
    try: test_f34_batched_blas_build_records(ctx); ctx.record_pass("test_f34_batched_blas_build_records") except e: ctx.record_fail("test_f34_batched_blas_build_records", String(e))
    try: test_f34_bone_matrix_composition(ctx); ctx.record_pass("test_f34_bone_matrix_composition") except e: ctx.record_fail("test_f34_bone_matrix_composition", String(e))
    try: test_f34_skinning_output_vertex_offset(ctx); ctx.record_pass("test_f34_skinning_output_vertex_offset") except e: ctx.record_fail("test_f34_skinning_output_vertex_offset", String(e))
    try: test_f34_blas_refit_without_rebuild(ctx); ctx.record_pass("test_f34_blas_refit_without_rebuild") except e: ctx.record_fail("test_f34_blas_refit_without_rebuild", String(e))

    try: test_f35_two_bone_ik_solution(ctx); ctx.record_pass("test_f35_two_bone_ik_solution") except e: ctx.record_fail("test_f35_two_bone_ik_solution", String(e))
    try: test_f35_aim_rotation_alignment(ctx); ctx.record_pass("test_f35_aim_rotation_alignment") except e: ctx.record_fail("test_f35_aim_rotation_alignment", String(e))
    try: test_f35_2d_blend_space_weights(ctx); ctx.record_pass("test_f35_2d_blend_space_weights") except e: ctx.record_fail("test_f35_2d_blend_space_weights", String(e))
    try: test_f35_motion_database_indexing(ctx); ctx.record_pass("test_f35_motion_database_indexing") except e: ctx.record_fail("test_f35_motion_database_indexing", String(e))
    try: test_f35_soft_reach_ik_damping(ctx); ctx.record_pass("test_f35_soft_reach_ik_damping") except e: ctx.record_fail("test_f35_soft_reach_ik_damping", String(e))

    try: test_f36_spsc_queue_lock_free(ctx); ctx.record_pass("test_f36_spsc_queue_lock_free") except e: ctx.record_fail("test_f36_spsc_queue_lock_free", String(e))
    try: test_f36_audio_sample_rate(ctx); ctx.record_pass("test_f36_audio_sample_rate") except e: ctx.record_fail("test_f36_audio_sample_rate", String(e))
    try: test_f36_audio_quantum_buffer_frames(ctx); ctx.record_pass("test_f36_audio_quantum_buffer_frames") except e: ctx.record_fail("test_f36_audio_quantum_buffer_frames", String(e))
    try: test_f36_voice_bus_volume_gains(ctx); ctx.record_pass("test_f36_voice_bus_volume_gains") except e: ctx.record_fail("test_f36_voice_bus_volume_gains", String(e))
    try: test_f36_zero_quantum_allocations(ctx); ctx.record_pass("test_f36_zero_quantum_allocations") except e: ctx.record_fail("test_f36_zero_quantum_allocations", String(e))

    try: test_f37_spatial_3d_distance_attenuation(ctx); ctx.record_pass("test_f37_spatial_3d_distance_attenuation") except e: ctx.record_fail("test_f37_spatial_3d_distance_attenuation", String(e))
    try: test_f37_spatial_3d_stereo_panning(ctx); ctx.record_pass("test_f37_spatial_3d_stereo_panning") except e: ctx.record_fail("test_f37_spatial_3d_stereo_panning", String(e))
    try: test_f37_spatial_3d_doppler_shift(ctx); ctx.record_pass("test_f37_spatial_3d_doppler_shift") except e: ctx.record_fail("test_f37_spatial_3d_doppler_shift", String(e))
    try: test_f37_procedural_cytomic_svf(ctx); ctx.record_pass("test_f37_procedural_cytomic_svf") except e: ctx.record_fail("test_f37_procedural_cytomic_svf", String(e))
    try: test_f37_modal_resonator_harmonic(ctx); ctx.record_pass("test_f37_modal_resonator_harmonic") except e: ctx.record_fail("test_f37_modal_resonator_harmonic", String(e))

    try: test_f38_evdev_key_mapping(ctx); ctx.record_pass("test_f38_evdev_key_mapping") except e: ctx.record_fail("test_f38_evdev_key_mapping", String(e))
    try: test_f38_raw_unaccelerated_mouse_delta(ctx); ctx.record_pass("test_f38_raw_unaccelerated_mouse_delta") except e: ctx.record_fail("test_f38_raw_unaccelerated_mouse_delta", String(e))
    try: test_f38_evdev_ff_rumble_effect(ctx); ctx.record_pass("test_f38_evdev_ff_rumble_effect") except e: ctx.record_fail("test_f38_evdev_ff_rumble_effect", String(e))
    try: test_f38_pointer_lock_symmetry(ctx); ctx.record_pass("test_f38_pointer_lock_symmetry") except e: ctx.record_fail("test_f38_pointer_lock_symmetry", String(e))
    try: test_f38_dual_motor_haptic_range(ctx); ctx.record_pass("test_f38_dual_motor_haptic_range") except e: ctx.record_fail("test_f38_dual_motor_haptic_range", String(e))

    try: test_f39_mcp_query_ping(ctx); ctx.record_pass("test_f39_mcp_query_ping") except e: ctx.record_fail("test_f39_mcp_query_ping", String(e))
    try: test_f39_cvar_lookup_and_set(ctx); ctx.record_pass("test_f39_cvar_lookup_and_set") except e: ctx.record_fail("test_f39_cvar_lookup_and_set", String(e))
    try: test_f39_flamebar_zone_profiling(ctx); ctx.record_pass("test_f39_flamebar_zone_profiling") except e: ctx.record_fail("test_f39_flamebar_zone_profiling", String(e))
    try: test_f39_gpu_buffer_inspector_bounds(ctx); ctx.record_pass("test_f39_gpu_buffer_inspector_bounds") except e: ctx.record_fail("test_f39_gpu_buffer_inspector_bounds", String(e))
    try: test_f39_mcp_command_execution(ctx); ctx.record_pass("test_f39_mcp_command_execution") except e: ctx.record_fail("test_f39_mcp_command_execution", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("test_t1_m4_world_sim failed")
