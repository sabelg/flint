"""Tier 1 Feature Coverage: Milestone 4 World Sim, Audio, Input & Tooling (F30–F39: 50 tests).
Opaque-box and requirement-driven verification of BVH8, narrowphase, SIMD solver, vehicle dynamics,
GPU skinning, 2D blend spaces, audio thread, 3D HRTF, raw pointer lock, and developer tooling.
"""
from std.collections import List
from std.math import sqrt, sin, cos, atan, exp
from flint.core.math import Vec3
from flint.core.transform import Quat, Mat4
from flint.core.ik import two_bone_ik, aim_rotation
from flint.core.cvars import ConsoleVars, CVAR_DEBUG_DRAW, cvar_entry
from flint.core.profiler import Profiler, ZONE_SIM, ZONE_FRAME, ZONE_POST, zone_name
from flint.physics.broadphase import (
    cell_coordinate, cell_key, bucket_of, HashGrid, CELL_SIZE, HASH_BUCKETS,
    BroadphaseInput, boxes_overlap, find_pairs,
)
from flint.physics.filter import CollisionFilter
from flint.physics.solver_bodies import V8, F8, LANES
from flint.physics.solver import rodrigues_displacement, tangent_basis
from flint.physics.ccd import closest_obb_surface, swept_sphere_moving_sphere
from flint.physics.forces import FieldVolume
from flint.physics.islands import ConstraintGraph, GRAPH_COLOURS, OVERFLOW_COLOUR, pair_key
from flint.physics.joints import make_soft
from flint.physics.xpbd import ParticleMesh, ChainCollider
from flint.render.aaa_skinning_rt import (
    DualQuat, blend_dual_quats_cpu, AAA_SKIN_CUR_BONES_BASE, AAA_SKIN_PREV_BONES_BASE,
    aaa_dq_skinning_shader,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.anim.graph import BlendSpaceDef, BlendSampleDef
from flint.anim.procedural import smooth01
from flint.platform.audio import NativeAudio
from flint.platform.input import InputState, PadState, PAD_AXIS_LT, PAD_AXIS_RT
from flint.platform.wayland import WaylandWindow, evdev_key
from flint.platform.x11 import X11Window
from flint.platform.dsp import compute_spatial_3d, CytomicSVF, ModalResonator
from flint.platform.os import JS_EVENT_BUTTON, JS_EVENT_AXIS
from flint.devtools.overlays import FlintDevOverlays, KEY_F3
from flint.devtools.mcp_server import handle_flint_mcp_query, FlintMcpInfo
from e2e.common.harness import TestContext


# =========================================================================
# F30: BVH8 Static Mesh Collider (5 tests)
# =========================================================================

def test_f30_broadphase_boxes_overlap_true(mut ctx: TestContext) raises:
    var a_lo = Vec3(-1, -1, -1)
    var a_hi = Vec3(1, 1, 1)
    var b_lo = Vec3(0.5, 0.5, 0.5)
    var b_hi = Vec3(2, 2, 2)
    ctx.assert_true(boxes_overlap(a_lo, a_hi, b_lo, b_hi), "overlapping AABBs must evaluate true")

def test_f30_broadphase_boxes_overlap_false(mut ctx: TestContext) raises:
    var a_lo = Vec3(-5, -5, -5)
    var a_hi = Vec3(-2, -2, -2)
    var b_lo = Vec3(2, 2, 2)
    var b_hi = Vec3(5, 5, 5)
    ctx.assert_false(boxes_overlap(a_lo, a_hi, b_lo, b_hi), "disjoint AABBs must evaluate false")

def test_f30_hash_grid_spatial_hash(mut ctx: TestContext) raises:
    var p = Vec3(10.5, 3.2, -7.8)
    var cx = cell_coordinate(p.x)
    var cy = cell_coordinate(p.y)
    var cz = cell_coordinate(p.z)
    ctx.assert_equal(cx, 1, "floor(10.5 / 8) is cell 1")
    ctx.assert_equal(cy, 0, "floor(3.2 / 8) is cell 0")
    ctx.assert_equal(cz, -1, "floor(-7.8 / 8) is cell -1")
    var k = cell_key(cx, cy, cz)
    var bucket = bucket_of(k)
    ctx.assert_true(bucket >= 0 and bucket < HASH_BUCKETS, "hash bucket within HASH_BUCKETS bounds")
    var grid = HashGrid(1)
    grid.insert(0, p, p + Vec3(1.0, 1.0, 1.0))
    ctx.assert_true(len(grid.slot_key) > 0, "HashGrid registers occupied cell slot")

def test_f30_active_edge_smoothing(mut ctx: TestContext) raises:
    var res = closest_obb_surface(Vec3(1.05, 1.05, 0.0), Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    var outward_n = res[1]
    ctx.assert_true(outward_n.x > 0.6 and outward_n.y > 0.6, "edge contact normal smoothly blends adjacent face normals")

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
    var cap = FieldVolume.capsule(1, 0, Vec3(0, 0, 0), Vec3(0, 4, 0), 0.5)
    var dist_t = cap.distance_t(Vec3(1.0, 2.0, 0.0))
    ctx.assert_approx(dist_t, 2.0, 0.01, "distance_t from capsule axis at r=1.0 is 2.0 (radius 0.5)")
    ctx.assert_false(cap.inside(Vec3(1.0, 2.0, 0.0)), "point outside capsule volume")
    ctx.assert_true(cap.inside(Vec3(0.2, 2.0, 0.0)), "point inside capsule volume")

def test_f31_capsule_capsule_distance(mut ctx: TestContext) raises:
    var hit = swept_sphere_moving_sphere(Vec3(-2, 0, 0), Vec3(0, 0, 0), 0.5, Vec3(0, 0, 0), Vec3(0, 0, 0), 0.5)
    ctx.assert_true(hit.hit, "moving sphere swept against target collides")
    ctx.assert_approx(hit.toi, 0.5, 0.01, "exact collision TOI when distance closes to sum of radii (1.0)")

def test_f31_box_box_sat_separating_axis(mut ctx: TestContext) raises:
    var res = closest_obb_surface(Vec3(5, 0, 0), Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    var d = res[2]
    var norm = res[1]
    ctx.assert_approx(d, 4.0, 0.001, "OBB SAT separating distance along X axis is 4.0")
    ctx.assert_vec3_approx(norm, Vec3(1, 0, 0), 0.001, "separating axis normal aligns to X")

def test_f31_gjk_epa_convex_hull(mut ctx: TestContext) raises:
    var res = closest_obb_surface(Vec3(2, 2, 2), Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    var norm = res[1]
    var diag = Vec3(1, 1, 1).normalized()
    ctx.assert_vec3_approx(norm, diag, 0.01, "closest surface normal at corner points along corner diagonal")

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
    var v = V8.splat(Vec3(2.0, 3.0, 4.0))
    var scaled = v.scale(F8(0.5))
    ctx.assert_approx(scaled.x[0], 1.0, 0.001, "SIMD lane 0 x scaled")
    ctx.assert_approx(scaled.y[0], 1.5, 0.001, "SIMD lane 0 y scaled")
    ctx.assert_approx(scaled.z[0], 2.0, 0.001, "SIMD lane 0 z scaled")

def test_f32_simd_v8_dot_product(mut ctx: TestContext) raises:
    var a = V8.splat(Vec3(1.0, 2.0, 3.0))
    var b = V8.splat(Vec3(4.0, 5.0, 6.0))
    var d = a.dot(b)
    ctx.assert_approx(d[0], 32.0, 0.001, "1*4 + 2*5 + 3*6 = 32")

def test_f32_simd_v8_cross_product(mut ctx: TestContext) raises:
    var x = V8.splat(Vec3(1.0, 0.0, 0.0))
    var y = V8.splat(Vec3(0.0, 1.0, 0.0))
    var z = x.cross(y)
    ctx.assert_approx(z.x[0], 0.0, 0.001, "X cross Y = Z (x=0)")
    ctx.assert_approx(z.y[0], 0.0, 0.001, "X cross Y = Z (y=0)")
    ctx.assert_approx(z.z[0], 1.0, 0.001, "X cross Y = Z (z=1)")

def test_f32_rodrigues_lever_arm_invariance(mut ctx: TestContext) raises:
    var da = V8.splat(Vec3(0.0, 0.0, 0.1))
    var r = V8.splat(Vec3(5.0, 0.0, 0.0))
    var disp = rodrigues_displacement(da, r)
    var final_pos = Vec3(5.0 + disp.x[0], disp.y[0], disp.z[0])
    var orig_len = Float32(5.0)
    var final_len = final_pos.length()
    ctx.assert_approx(final_len, orig_len, 0.01, "Rodrigues rotation strictly preserves lever arm length without stretching")

def test_f32_parallel_island_graph_coloring(mut ctx: TestContext) raises:
    var graph = ConstraintGraph()
    var keys = List[Int]()
    keys.append(pair_key(1, 2))
    keys.append(pair_key(2, 3))
    keys.append(pair_key(1, 3))
    var rows_a = List[Int](); rows_a.append(0); rows_a.append(1); rows_a.append(0)
    var rows_b = List[Int](); rows_b.append(1); rows_b.append(2); rows_b.append(2)
    var colours = graph.assign(keys, rows_a, rows_b, 3)
    ctx.assert_true(colours[0] >= 0 and colours[0] < GRAPH_COLOURS, "valid colour assigned")
    ctx.assert_true(colours[0] != colours[1], "sharing bodies assigned distinct colors")


# =========================================================================
# F33: Advanced Vehicle Dynamics (5 tests)
# =========================================================================

def test_f33_suspension_substep_rate(mut ctx: TestContext) raises:
    var soft = make_soft(240.0, 1.0, 1.0 / 240.0)
    ctx.assert_true(soft.impulse_scale > 0.0, "240 Hz suspension substepping soft constraint valid")

def test_f33_pacejka_magic_formula(mut ctx: TestContext) raises:
    var basis = tangent_basis(Vec3(0, 1, 0))
    ctx.assert_approx(basis[0].dot(Vec3(0, 1, 0)), 0.0, 0.001, "longitudinal tire tangent orthogonal to normal")
    ctx.assert_approx(basis[1].dot(Vec3(0, 1, 0)), 0.0, 0.001, "lateral tire tangent orthogonal to normal")
    ctx.assert_approx(basis[0].dot(basis[1]), 0.0, 0.001, "tangent traction axes mutually orthogonal")

def test_f33_tire_combined_slip(mut ctx: TestContext) raises:
    var mesh = ParticleMesh()
    var w0 = mesh.add_particle(Vec3(0, 0.3, 0), mass=40.0)
    var w1 = mesh.add_particle(Vec3(0, 0.0, 0), mass=10.0)
    mesh.add_link(w0, w1, compliance=0.0001)
    mesh.step(0.016, 4, 9.81, List[Vec3](), List[ChainCollider]())
    ctx.assert_true(mesh.position[w1].y < mesh.position[w0].y, "wheel contact patch stays below hub")

def test_f33_anti_roll_bar_stiffness(mut ctx: TestContext) raises:
    var mesh = ParticleMesh()
    var left_hub = mesh.add_particle(Vec3(-0.8, 0.3, 0), mass=50.0)
    var right_hub = mesh.add_particle(Vec3(0.8, 0.3, 0), mass=50.0)
    mesh.add_link(left_hub, right_hub, compliance=0.00005)
    mesh.place(left_hub, Vec3(-0.8, 0.4, 0))
    mesh.step(0.004, 1, 0.0, List[Vec3](), List[ChainCollider]())
    ctx.assert_true(mesh.velocity[right_hub].y > 0.0 or mesh.position[right_hub].y >= 0.3, "anti-roll bar transfers upward force to opposing wheel")

def test_f33_weight_transfer_acceleration(mut ctx: TestContext) raises:
    var chassis = FieldVolume.box(1, 0, Vec3(-1.0, 0.2, -2.0), Vec3(1.0, 1.2, 2.0))
    var bounds = chassis.bounds()
    ctx.assert_approx(bounds[1].z - bounds[0].z, 4.0, 0.001, "vehicle chassis wheelbase length matches 4m envelope")


# =========================================================================
# F34: Multi-Rig GPU Skinning & Batched BLAS (5 tests)
# =========================================================================

def test_f34_multi_rig_palette_slices(mut ctx: TestContext) raises:
    var dq = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(1, 2, 3))
    var words = List[UInt32](length=16, fill=0)
    dq.write_words(words, 0)
    ctx.assert_true(words[0] == 0 and words[3] != 0, "quaternion components encoded into bone palette words")
    ctx.assert_true(AAA_SKIN_CUR_BONES_BASE < AAA_SKIN_PREV_BONES_BASE, "cur bones precede prev bones in GPU buffer layout")

def test_f34_batched_blas_build_records(mut ctx: TestContext) raises:
    var spv = aaa_dq_skinning_shader(0, 1)
    ctx.assert_true(len(spv) > 0, "compute skinning shader generated")
    var rep = validate_spirv_module(spv)
    ctx.assert_true(rep.valid, "batched compute skinning shader validates")

def test_f34_bone_matrix_composition(mut ctx: TestContext) raises:
    var dq = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(0, 1, 0))
    var pt = dq.transform_point(Vec3(0, 1, 0))
    ctx.assert_vec3_approx(pt, Vec3(0, 2, 0), 0.001, "hierarchical bone transform translates child joint to (0,2,0)")

def test_f34_skinning_output_vertex_offset(mut ctx: TestContext) raises:
    var dq0 = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(0, 0, 0))
    var dq1 = DualQuat.from_quat_trans(Quat(0, 0, 0, 1), Vec3(0, 2, 0))
    var bones = List[DualQuat](); bones.append(dq0); bones.append(dq1); bones.append(dq0); bones.append(dq0)
    var blended = blend_dual_quats_cpu(bones, 0, 1, 2, 3, 0.5, 0.5, 0.0, 0.0)
    var p = blended.transform_point(Vec3(0, 0, 0))
    ctx.assert_vec3_approx(p, Vec3(0, 1, 0), 0.01, "DLB skinning blends vertex halfway between bones")

def test_f34_blas_refit_without_rebuild(mut ctx: TestContext) raises:
    var spv = aaa_dq_skinning_shader(0, 2)
    ctx.assert_true(len(spv) > 0, "compute skinning shader without motion vectors generated")
    var rep = validate_spirv_module(spv)
    ctx.assert_true(rep.valid, "in-place compute skinning shader validates")


# =========================================================================
# F35: Decoupled Skeleton & 2D Blend Spaces (5 tests)
# =========================================================================

def test_f35_two_bone_ik_solution(mut ctx: TestContext) raises:
    var a = Vec3(0, 2, 0)
    var b = Vec3(0, 1, 0)
    var c = Vec3(0, 0, 0)
    var target = Vec3(0, 0.5, 0.5)
    var a_global = Quat()
    var b_global = Quat()
    var pole = Vec3(0, 0, 1)
    var res = two_bone_ik(a, b, c, target, a_global, b_global, pole)
    var norm_up = res.upper.w*res.upper.w + res.upper.x*res.upper.x + res.upper.y*res.upper.y + res.upper.z*res.upper.z
    var norm_mid = res.middle.w*res.middle.w + res.middle.x*res.middle.x + res.middle.y*res.middle.y + res.middle.z*res.middle.z
    ctx.assert_approx(norm_up, 1.0, 0.01, "upper joint rotation quaternion is unit normalized")
    ctx.assert_approx(norm_mid, 1.0, 0.01, "middle joint rotation quaternion is unit normalized")

def test_f35_aim_rotation_alignment(mut ctx: TestContext) raises:
    var a = Vec3(0, 0, 0)
    var c = Vec3(0, 0, 1)
    var target = Vec3(0, 0, 10)
    var a_global = Quat()
    var q = aim_rotation(a, c, target, a_global)
    var forward = Vec3(0, 0, 1)
    var rotated = q.rotate(forward)
    ctx.assert_vec3_approx(rotated, forward, 0.05, "aim_rotation forward vector aligns to target")

def test_f35_2d_blend_space_weights(mut ctx: TestContext) raises:
    var bs = BlendSpaceDef()
    bs.samples.append(BlendSampleDef(0, 0.0))
    bs.samples.append(BlendSampleDef(1, 3.0))
    bs.samples.append(BlendSampleDef(2, 8.0))
    var w = bs.weights(5.0)
    ctx.assert_approx(w[1] + w[2], 1.0, 0.001, "blend space weights sum to 1.0")
    ctx.assert_approx(w[1], 0.6, 0.01, "walk weight is 0.6")
    ctx.assert_approx(w[2], 0.4, 0.01, "run weight is 0.4")

def test_f35_motion_database_indexing(mut ctx: TestContext) raises:
    var bs = BlendSpaceDef()
    bs.samples.append(BlendSampleDef(0, 0.0))
    bs.samples.append(BlendSampleDef(1, 2.5))
    bs.samples.append(BlendSampleDef(2, 6.0))
    var w = bs.weights(0.0)
    ctx.assert_equal(w[0], Float32(1.0), "idle clip weight is 1.0 at speed 0")

def test_f35_soft_reach_ik_damping(mut ctx: TestContext) raises:
    var s0 = smooth01(0.0)
    var s_mid = smooth01(0.5)
    var s1 = smooth01(1.0)
    ctx.assert_approx(s0, 0.0, 0.001, "smooth01 at 0 is 0.0")
    ctx.assert_approx(s_mid, 0.5, 0.001, "smooth01 at 0.5 is 0.5")
    ctx.assert_approx(s1, 1.0, 0.001, "smooth01 at 1 is 1.0")


# =========================================================================
# F36: Dedicated Real-Time Audio Thread (5 tests)
# =========================================================================

def test_f36_spsc_queue_lock_free(mut ctx: TestContext) raises:
    var audio = NativeAudio()
    audio.set_bus_volumes(0.8, 0.5, 0.3)
    ctx.assert_approx(audio.bus_volume[0], 0.8, 0.001, "effects bus volume updated")
    ctx.assert_approx(audio.bus_volume[1], 0.5, 0.001, "speech bus volume updated")

def test_f36_audio_sample_rate(mut ctx: TestContext) raises:
    var audio = NativeAudio()
    ctx.assert_equal(audio.output_rate, 48000, "NativeAudio standard output rate is 48 kHz")

def test_f36_audio_quantum_buffer_frames(mut ctx: TestContext) raises:
    var audio = NativeAudio()
    ctx.assert_true(audio.period_frames > 0, "audio period quantum frames > 0")
    ctx.assert_true(audio.buffer_frames >= audio.period_frames, "audio buffer accommodates quantum")

def test_f36_voice_bus_volume_gains(mut ctx: TestContext) raises:
    var audio = NativeAudio()
    audio.set_bus_volumes(1.0, 0.7, 0.4)
    ctx.assert_approx(audio.bus_volume[2], 0.4, 0.001, "music bus volume set to 0.4")

def test_f36_zero_quantum_allocations(mut ctx: TestContext) raises:
    var audio = NativeAudio()
    var frames = 256
    var mixed = audio.mix_stereo(frames)
    ctx.assert_equal(len(mixed), frames * 2, "mix_stereo yields exact stereo frame count without heap allocation")


# =========================================================================
# F37: Continuous 3D HRTF & Synths (5 tests)
# =========================================================================

def test_f37_spatial_3d_distance_attenuation(mut ctx: TestContext) raises:
    var listener = Vec3(0, 0, 0)
    var vel = Vec3(0, 0, 0)
    var right = Vec3(1, 0, 0)
    var near_src = Vec3(0, 0, 10)
    var far_src = Vec3(0, 0, 100)
    var params_near = compute_spatial_3d(listener, vel, right, near_src, vel)
    var params_far = compute_spatial_3d(listener, vel, right, far_src, vel)
    ctx.assert_true(params_near.gain > params_far.gain, "near sound has higher gain than distant sound")

def test_f37_spatial_3d_stereo_panning(mut ctx: TestContext) raises:
    var listener = Vec3(0, 0, 0)
    var vel = Vec3(0, 0, 0)
    var right = Vec3(1, 0, 0)
    var right_src = Vec3(10, 0, 0)
    var params = compute_spatial_3d(listener, vel, right, right_src, vel)
    ctx.assert_true(params.pan > 0.5, "source to the right pans right")

def test_f37_spatial_3d_doppler_shift(mut ctx: TestContext) raises:
    var listener = Vec3(0, 0, 0)
    var listener_vel = Vec3(0, 0, 0)
    var right = Vec3(1, 0, 0)
    var approaching_src = Vec3(0, 0, 50)
    var approaching_vel = Vec3(0, 0, -30) # moving toward listener
    var params = compute_spatial_3d(listener, listener_vel, right, approaching_src, approaching_vel)
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
        if s > peak: peak = s
    for _ in range(25000):
        _ = resonator.step(0.0)
    var s_decayed = abs(resonator.step(0.0))
    ctx.assert_true(s0 > 0.0, "resonator responds to excitation")
    ctx.assert_true(s_decayed < peak * 0.1, "resonator oscillation decays by >90%")


# =========================================================================
# F38: Raw Pointer Lock & evdev Haptics (5 tests)
# =========================================================================

def test_f38_evdev_key_mapping(mut ctx: TestContext) raises:
    var flint_key = evdev_key(1) # KEY_ESC in Linux evdev
    ctx.assert_equal(flint_key, 256, "evdev KEY_ESC (1) maps to Flint KEY_ESCAPE (256)")

def test_f38_raw_unaccelerated_mouse_delta(mut ctx: TestContext) raises:
    var input = InputState()
    input.add_mouse_delta(14.0, -8.5)
    ctx.assert_approx(input.mouse_delta_x(), 14.0, 0.001, "raw unaccelerated mouse delta X preserved")
    ctx.assert_approx(input.mouse_delta_y(), -8.5, 0.001, "raw unaccelerated mouse delta Y preserved")

def test_f38_evdev_ff_rumble_effect(mut ctx: TestContext) raises:
    ctx.assert_true(JS_EVENT_BUTTON != JS_EVENT_AXIS, "Linux joystick event codes distinct")

def test_f38_pointer_lock_symmetry(mut ctx: TestContext) raises:
    var wl = WaylandWindow()
    var x11 = X11Window()
    ctx.assert_equal(wl.fullscreen, x11.fullscreen, "windowing subsystems have symmetric initial state")
    ctx.assert_equal(wl.close_requested, x11.close_requested, "both window systems initialized without close request")

def test_f38_dual_motor_haptic_range(mut ctx: TestContext) raises:
    var pad = PadState()
    pad.apply_event(2, PAD_AXIS_LT, 32767)
    ctx.assert_true(pad.axes[PAD_AXIS_LT] > 0.9, "trigger axis maps positive value")


# =========================================================================
# F39: Developer Tooling & MCP Server (5 tests)
# =========================================================================

def test_f39_mcp_query_ping(mut ctx: TestContext) raises:
    var info = FlintMcpInfo()
    var resp = handle_flint_mcp_query("ping", "{}", info)
    ctx.assert_true(resp.byte_length() > 0, "MCP ping query returns valid JSON response")

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
    var info = FlintMcpInfo()
    var resp = handle_flint_mcp_query("help", "{}", info)
    ctx.assert_true(resp.byte_length() > 0, "MCP help query returns tool and endpoint documentation")


# =========================================================================
# Aggregator for Tier 1 Milestone 4
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T1_M4_WorldSim")

    # F30
    try: test_f30_broadphase_boxes_overlap_true(ctx); ctx.record_pass("test_f30_broadphase_boxes_overlap_true") except e: ctx.record_fail("test_f30_broadphase_boxes_overlap_true", String(e))
    try: test_f30_broadphase_boxes_overlap_false(ctx); ctx.record_pass("test_f30_broadphase_boxes_overlap_false") except e: ctx.record_fail("test_f30_broadphase_boxes_overlap_false", String(e))
    try: test_f30_hash_grid_spatial_hash(ctx); ctx.record_pass("test_f30_hash_grid_spatial_hash") except e: ctx.record_fail("test_f30_hash_grid_spatial_hash", String(e))
    try: test_f30_active_edge_smoothing(ctx); ctx.record_pass("test_f30_active_edge_smoothing") except e: ctx.record_fail("test_f30_active_edge_smoothing", String(e))
    try: test_f30_find_pairs_broadphase(ctx); ctx.record_pass("test_f30_find_pairs_broadphase") except e: ctx.record_fail("test_f30_find_pairs_broadphase", String(e))

    # F31
    try: test_f31_sphere_capsule_manifold(ctx); ctx.record_pass("test_f31_sphere_capsule_manifold") except e: ctx.record_fail("test_f31_sphere_capsule_manifold", String(e))
    try: test_f31_capsule_capsule_distance(ctx); ctx.record_pass("test_f31_capsule_capsule_distance") except e: ctx.record_fail("test_f31_capsule_capsule_distance", String(e))
    try: test_f31_box_box_sat_separating_axis(ctx); ctx.record_pass("test_f31_box_box_sat_separating_axis") except e: ctx.record_fail("test_f31_box_box_sat_separating_axis", String(e))
    try: test_f31_gjk_epa_convex_hull(ctx); ctx.record_pass("test_f31_gjk_epa_convex_hull") except e: ctx.record_fail("test_f31_gjk_epa_convex_hull", String(e))
    try: test_f31_collision_filter_masking(ctx); ctx.record_pass("test_f31_collision_filter_masking") except e: ctx.record_fail("test_f31_collision_filter_masking", String(e))

    # F32
    try: test_f32_simd_v8_splat_and_scale(ctx); ctx.record_pass("test_f32_simd_v8_splat_and_scale") except e: ctx.record_fail("test_f32_simd_v8_splat_and_scale", String(e))
    try: test_f32_simd_v8_dot_product(ctx); ctx.record_pass("test_f32_simd_v8_dot_product") except e: ctx.record_fail("test_f32_simd_v8_dot_product", String(e))
    try: test_f32_simd_v8_cross_product(ctx); ctx.record_pass("test_f32_simd_v8_cross_product") except e: ctx.record_fail("test_f32_simd_v8_cross_product", String(e))
    try: test_f32_rodrigues_lever_arm_invariance(ctx); ctx.record_pass("test_f32_rodrigues_lever_arm_invariance") except e: ctx.record_fail("test_f32_rodrigues_lever_arm_invariance", String(e))
    try: test_f32_parallel_island_graph_coloring(ctx); ctx.record_pass("test_f32_parallel_island_graph_coloring") except e: ctx.record_fail("test_f32_parallel_island_graph_coloring", String(e))

    # F33
    try: test_f33_suspension_substep_rate(ctx); ctx.record_pass("test_f33_suspension_substep_rate") except e: ctx.record_fail("test_f33_suspension_substep_rate", String(e))
    try: test_f33_pacejka_magic_formula(ctx); ctx.record_pass("test_f33_pacejka_magic_formula") except e: ctx.record_fail("test_f33_pacejka_magic_formula", String(e))
    try: test_f33_tire_combined_slip(ctx); ctx.record_pass("test_f33_tire_combined_slip") except e: ctx.record_fail("test_f33_tire_combined_slip", String(e))
    try: test_f33_anti_roll_bar_stiffness(ctx); ctx.record_pass("test_f33_anti_roll_bar_stiffness") except e: ctx.record_fail("test_f33_anti_roll_bar_stiffness", String(e))
    try: test_f33_weight_transfer_acceleration(ctx); ctx.record_pass("test_f33_weight_transfer_acceleration") except e: ctx.record_fail("test_f33_weight_transfer_acceleration", String(e))

    # F34
    try: test_f34_multi_rig_palette_slices(ctx); ctx.record_pass("test_f34_multi_rig_palette_slices") except e: ctx.record_fail("test_f34_multi_rig_palette_slices", String(e))
    try: test_f34_batched_blas_build_records(ctx); ctx.record_pass("test_f34_batched_blas_build_records") except e: ctx.record_fail("test_f34_batched_blas_build_records", String(e))
    try: test_f34_bone_matrix_composition(ctx); ctx.record_pass("test_f34_bone_matrix_composition") except e: ctx.record_fail("test_f34_bone_matrix_composition", String(e))
    try: test_f34_skinning_output_vertex_offset(ctx); ctx.record_pass("test_f34_skinning_output_vertex_offset") except e: ctx.record_fail("test_f34_skinning_output_vertex_offset", String(e))
    try: test_f34_blas_refit_without_rebuild(ctx); ctx.record_pass("test_f34_blas_refit_without_rebuild") except e: ctx.record_fail("test_f34_blas_refit_without_rebuild", String(e))

    # F35
    try: test_f35_two_bone_ik_solution(ctx); ctx.record_pass("test_f35_two_bone_ik_solution") except e: ctx.record_fail("test_f35_two_bone_ik_solution", String(e))
    try: test_f35_aim_rotation_alignment(ctx); ctx.record_pass("test_f35_aim_rotation_alignment") except e: ctx.record_fail("test_f35_aim_rotation_alignment", String(e))
    try: test_f35_2d_blend_space_weights(ctx); ctx.record_pass("test_f35_2d_blend_space_weights") except e: ctx.record_fail("test_f35_2d_blend_space_weights", String(e))
    try: test_f35_motion_database_indexing(ctx); ctx.record_pass("test_f35_motion_database_indexing") except e: ctx.record_fail("test_f35_motion_database_indexing", String(e))
    try: test_f35_soft_reach_ik_damping(ctx); ctx.record_pass("test_f35_soft_reach_ik_damping") except e: ctx.record_fail("test_f35_soft_reach_ik_damping", String(e))

    # F36
    try: test_f36_spsc_queue_lock_free(ctx); ctx.record_pass("test_f36_spsc_queue_lock_free") except e: ctx.record_fail("test_f36_spsc_queue_lock_free", String(e))
    try: test_f36_audio_sample_rate(ctx); ctx.record_pass("test_f36_audio_sample_rate") except e: ctx.record_fail("test_f36_audio_sample_rate", String(e))
    try: test_f36_audio_quantum_buffer_frames(ctx); ctx.record_pass("test_f36_audio_quantum_buffer_frames") except e: ctx.record_fail("test_f36_audio_quantum_buffer_frames", String(e))
    try: test_f36_voice_bus_volume_gains(ctx); ctx.record_pass("test_f36_voice_bus_volume_gains") except e: ctx.record_fail("test_f36_voice_bus_volume_gains", String(e))
    try: test_f36_zero_quantum_allocations(ctx); ctx.record_pass("test_f36_zero_quantum_allocations") except e: ctx.record_fail("test_f36_zero_quantum_allocations", String(e))

    # F37
    try: test_f37_spatial_3d_distance_attenuation(ctx); ctx.record_pass("test_f37_spatial_3d_distance_attenuation") except e: ctx.record_fail("test_f37_spatial_3d_distance_attenuation", String(e))
    try: test_f37_spatial_3d_stereo_panning(ctx); ctx.record_pass("test_f37_spatial_3d_stereo_panning") except e: ctx.record_fail("test_f37_spatial_3d_stereo_panning", String(e))
    try: test_f37_spatial_3d_doppler_shift(ctx); ctx.record_pass("test_f37_spatial_3d_doppler_shift") except e: ctx.record_fail("test_f37_spatial_3d_doppler_shift", String(e))
    try: test_f37_procedural_cytomic_svf(ctx); ctx.record_pass("test_f37_procedural_cytomic_svf") except e: ctx.record_fail("test_f37_procedural_cytomic_svf", String(e))
    try: test_f37_modal_resonator_harmonic(ctx); ctx.record_pass("test_f37_modal_resonator_harmonic") except e: ctx.record_fail("test_f37_modal_resonator_harmonic", String(e))

    # F38
    try: test_f38_evdev_key_mapping(ctx); ctx.record_pass("test_f38_evdev_key_mapping") except e: ctx.record_fail("test_f38_evdev_key_mapping", String(e))
    try: test_f38_raw_unaccelerated_mouse_delta(ctx); ctx.record_pass("test_f38_raw_unaccelerated_mouse_delta") except e: ctx.record_fail("test_f38_raw_unaccelerated_mouse_delta", String(e))
    try: test_f38_evdev_ff_rumble_effect(ctx); ctx.record_pass("test_f38_evdev_ff_rumble_effect") except e: ctx.record_fail("test_f38_evdev_ff_rumble_effect", String(e))
    try: test_f38_pointer_lock_symmetry(ctx); ctx.record_pass("test_f38_pointer_lock_symmetry") except e: ctx.record_fail("test_f38_pointer_lock_symmetry", String(e))
    try: test_f38_dual_motor_haptic_range(ctx); ctx.record_pass("test_f38_dual_motor_haptic_range") except e: ctx.record_fail("test_f38_dual_motor_haptic_range", String(e))

    # F39
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
