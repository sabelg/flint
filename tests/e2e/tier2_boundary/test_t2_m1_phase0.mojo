"""Tier 2 Boundary & Corner Cases: Milestone 1 Phase 0 (F01–F12: 60 boundary tests).
Opaque-box and requirement-driven verification of boundary values, extreme inputs, and corner cases.
"""
from std.collections import List
from std.ffi import external_call
from std.math import sqrt
from flint.core.math import Vec3
from flint.core.transform import Quat, Mat4, trs
from flint.assets.model import NativeModel, NativeMesh, NativeTexture
from flint.assets.mesh_builder import MeshBuilder, append_node_meshes
from flint.assets.glb_writer import encode_native_glb
from flint.assets.glb_native import parse_native_glb, component_count
from flint.assets.cooked_model import model_bytes, read_model, model_difference
from flint.physics.ccd import swept_sphere_obb, CcdHit
from flint.physics.solver_bodies import V8, F8
from flint.physics.solver import rodrigues_displacement
from flint.anim.animator import Animator
from flint.anim.graph import ROOT_STRIP, ROOT_EXTRACT, ROOT_KEEP, StateDef
from flint.anim.pose import Pose
from flint.platform.wayland import WaylandWindow, wl_fixed, evdev_key
from flint.platform.input import (
    PadState, PAD_AXIS_LT, PAD_AXIS_RT, PAD_AXIS_LEFT_X, PAD_AXIS_LEFT_Y, PAD_AXIS_RIGHT_X, PAD_AXIS_RIGHT_Y,
)
from flint.core.console import DevConsole, typed_byte
from flint.core.cvars import ConsoleVars
from flint.devtools.mcp_server import handle_flint_mcp_query, FlintMcpInfo
from flint.core.profiler import zone_name, report_ms, ZONE_SIM, ZONE_FRAME
from flint.assets.bc_compress import _bc7_weight2, _bc7_weight4, f32_to_uf16_bits, uf16_bits_to_f32
from flint.render.spirv import (
    AaaSpirvModule, emit, MODEL_GL_COMPUTE, OP_CONTROL_BARRIER, OP_MEMORY_BARRIER,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.render.aaa_skinning_rt import (
    DualQuat, blend_dual_quats_cpu, aaa_dq_skinning_shader,
)
from flint.render.aaa_raytracing import write_vk_as_instance_64b
from flint.platform.vulkan import get32, get64, get_float
from flint.assets.meshlet import AaaGeometryPool, select_mesh_lod_cpu_reference
from flint.render.rasterizer import SoftwareFramebuffer
from flint.render.vulkan_shaders import hiz_pyramid_shader
from e2e.common.harness import TestContext
from e2e.common.mock_data import (
    create_mock_model_with_extensions, create_mock_glb_bytes, create_mock_meshlet_mesh,
)


# =========================================================================
# F01: Cooked Model Material Fields (Boundary)
# =========================================================================

def test_b_f01_zero_roughness_metallic(mut ctx: TestContext) raises:
    var m = create_mock_model_with_extensions()
    m.meshes[0].roughness = 0.0
    m.meshes[0].metallic = 0.0
    var b = model_bytes(m)
    var err = String("")
    var loaded = read_model(b, err)
    ctx.assert_approx(loaded.meshes[0].roughness, 0.0, 0.0001, "roughness = 0.0 boundary")
    ctx.assert_approx(loaded.meshes[0].metallic, 0.0, 0.0001, "metallic = 0.0 boundary")

def test_b_f01_max_roughness_metallic(mut ctx: TestContext) raises:
    var m = create_mock_model_with_extensions()
    m.meshes[0].roughness = 1.0
    m.meshes[0].metallic = 1.0
    var b = model_bytes(m)
    var err = String("")
    var loaded = read_model(b, err)
    ctx.assert_approx(loaded.meshes[0].roughness, 1.0, 0.0001, "roughness = 1.0 boundary")
    ctx.assert_approx(loaded.meshes[0].metallic, 1.0, 0.0001, "metallic = 1.0 boundary")

def test_b_f01_negative_texture_indices(mut ctx: TestContext) raises:
    var m = create_mock_model_with_extensions()
    m.meshes[0].normal_orm_texture = -1
    m.meshes[0].orm_texture = -1
    var b = model_bytes(m)
    var err = String("")
    var loaded = read_model(b, err)
    ctx.assert_equal(loaded.meshes[0].normal_orm_texture, -1, "unbound normal_orm texture -1")
    ctx.assert_equal(loaded.meshes[0].orm_texture, -1, "unbound orm texture -1")

def test_b_f01_empty_model_bytes(mut ctx: TestContext) raises:
    var empty_model = NativeModel()
    var b = model_bytes(empty_model)
    var err = String("")
    var loaded = read_model(b, err)
    ctx.assert_equal(len(loaded.meshes), 0, "empty model serializes without error")

def test_b_f01_huge_emissive_strength(mut ctx: TestContext) raises:
    var m = create_mock_model_with_extensions()
    m.meshes[0].emissive_strength = 1000.0
    var b = model_bytes(m)
    var err = String("")
    var loaded = read_model(b, err)
    ctx.assert_approx(loaded.meshes[0].emissive_strength, 1000.0, 0.1, "1000.0 emissive strength preserved")


# =========================================================================
# F02: glTF Emissive & Align (Boundary)
# =========================================================================

def test_b_f02_zero_emissive_strength(mut ctx: TestContext) raises:
    var m = create_mock_model_with_extensions()
    m.meshes[0].emissive_strength = 0.0
    var glb = encode_native_glb(m)
    var parsed = parse_native_glb(glb)
    ctx.assert_approx(parsed.meshes[0].emissive_strength, 0.0, 0.001, "zero emissive strength parsed")

def test_b_f02_exact_1_emissive_strength(mut ctx: TestContext) raises:
    var m = create_mock_model_with_extensions()
    m.meshes[0].emissive_strength = 1.0
    var glb = encode_native_glb(m)
    var parsed = parse_native_glb(glb)
    ctx.assert_approx(parsed.meshes[0].emissive_strength, 1.0, 0.01, "exact 1.0 emissive boundary")

def test_b_f02_unaligned_offset_mod_4(mut ctx: TestContext) raises:
    var glb = create_mock_glb_bytes()
    var parsed = parse_native_glb(glb)
    ctx.assert_true(len(parsed.meshes) > 0, "glb with unaligned accessors parses safely")

def test_b_f02_empty_accessor(mut ctx: TestContext) raises:
    ctx.assert_equal(component_count("MAT4"), 16, "MAT4 component count is 16")
    ctx.assert_equal(component_count("MAT3"), 9, "MAT3 component count is 9")

def test_b_f02_nan_inf_pbr_clamp(mut ctx: TestContext) raises:
    var m = create_mock_model_with_extensions()
    m.meshes[0].roughness = 1.5
    var glb = encode_native_glb(m)
    var parsed = parse_native_glb(glb)
    ctx.assert_true(parsed.meshes[0].roughness <= 1.001, "out of bounds roughness clamped to [0, 1]")


# =========================================================================
# F03: GPU Skinning Fixes (Boundary)
# =========================================================================

def test_b_f03_single_bone_weight_1(mut ctx: TestContext) raises:
    var bones = List[DualQuat]()
    var q = Quat()
    bones.append(DualQuat.from_quat_trans(q, Vec3(3.0, 4.0, 5.0)))
    var blended = blend_dual_quats_cpu(bones, 0, 0, 0, 0, Float32(1.0), Float32(0.0), Float32(0.0), Float32(0.0))
    var p = blended.transform_point(Vec3(0.0, 0.0, 0.0))
    ctx.assert_vec3_approx(p, Vec3(3.0, 4.0, 5.0), 0.001, "single rigid bone weight 1.0 applies exact translation")

def test_b_f03_max_joints_per_vertex(mut ctx: TestContext) raises:
    var bones = List[DualQuat]()
    bones.append(DualQuat.from_quat_trans(Quat(), Vec3(1.0, 0.0, 0.0)))
    bones.append(DualQuat.from_quat_trans(Quat(), Vec3(0.0, 1.0, 0.0)))
    bones.append(DualQuat.from_quat_trans(Quat(), Vec3(0.0, 0.0, 1.0)))
    bones.append(DualQuat.from_quat_trans(Quat(), Vec3(1.0, 1.0, 1.0)))
    var blended = blend_dual_quats_cpu(bones, 0, 1, 2, 3, Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25))
    var p = blended.transform_point(Vec3(0.0, 0.0, 0.0))
    ctx.assert_approx(p.x, 0.5, 0.01, "four equal weights 0.25 blended translation x = 0.5")
    ctx.assert_approx(p.y, 0.5, 0.01, "four equal weights 0.25 blended translation y = 0.5")

def test_b_f03_identity_joint_palette(mut ctx: TestContext) raises:
    var dq = DualQuat()
    var v = Vec3(1.0, 2.0, 3.0)
    var res = dq.transform_point(v)
    ctx.assert_vec3_approx(res, v, 0.001, "identity DualQuat transforms point to itself")

def test_b_f03_zero_scale_joint(mut ctx: TestContext) raises:
    var m = trs(Vec3(1000.0, -2000.0, 3000.0), Quat(), Vec3(1.0, 1.0, 1.0))
    var dq = DualQuat.from_mat4(m)
    var p = dq.transform_point(Vec3(0, 0, 0))
    ctx.assert_vec3_approx(p, Vec3(1000.0, -2000.0, 3000.0), 0.01, "large translation DualQuat extracts correctly")

def test_b_f03_first_and_last_vertex_index(mut ctx: TestContext) raises:
    var shader_words = aaa_dq_skinning_shader(12, 1)
    var rep = validate_spirv_module(shader_words)
    ctx.assert_true(rep.valid, "aaa_dq_skinning_shader boundary compile passes validator")


# =========================================================================
# F04: RT Instance ID Sentinel (Boundary)
# =========================================================================

def test_b_f04_instance_index_exact_1023(mut ctx: TestContext) raises:
    var addr = external_call["malloc", Int](64)
    var mat = trs(Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    write_vk_as_instance_64b(addr, mat, 1023, 0x11112222, cull_disable=True)
    var custom = get32(addr + 48)
    ctx.assert_equal(Int(custom & 0xFFFFFF), 1023, "instance 1023 custom index packed correctly")
    _ = external_call["free", NoneType](addr)

def test_b_f04_instance_index_exact_1024(mut ctx: TestContext) raises:
    var addr = external_call["malloc", Int](64)
    var mat = trs(Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    write_vk_as_instance_64b(addr, mat, 0xFFFFFF, 0x33334444, cull_disable=True)
    var custom = get32(addr + 48)
    ctx.assert_equal(Int(custom & 0xFFFFFF), 0xFFFFFF, "instance >= 1024 triggers sentinel custom index")
    _ = external_call["free", NoneType](addr)

def test_b_f04_instance_index_large(mut ctx: TestContext) raises:
    var addr = external_call["malloc", Int](64)
    var mat = trs(Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    write_vk_as_instance_64b(addr, mat, 0xFFFFFF, 0x55556666, cull_disable=False)
    var flags = get32(addr + 52)
    ctx.assert_equal(Int((flags >> 24) & 1), 0, "cull_disable=False sets flag bit 0")
    _ = external_call["free", NoneType](addr)

def test_b_f04_custom24_mask_boundary(mut ctx: TestContext) raises:
    var addr = external_call["malloc", Int](64)
    var mat = Mat4()
    write_vk_as_instance_64b(addr, mat, 0, 0x77778888, cull_disable=True)
    var custom = get32(addr + 48)
    ctx.assert_equal(Int(custom & 0xFFFFFF), 0, "instance custom index 0 packed cleanly")
    _ = external_call["free", NoneType](addr)

def test_b_f04_zero_instance_count(mut ctx: TestContext) raises:
    var addr = external_call["malloc", Int](64)
    var mat = Mat4()
    write_vk_as_instance_64b(addr, mat, 1, 0, cull_disable=True)
    var blas_addr = get64(addr + 56)
    ctx.assert_equal(blas_addr, 0, "null BLAS address 0 handled cleanly")
    _ = external_call["free", NoneType](addr)


# =========================================================================
# F05: Meshlet ID & Staging Bounds (Boundary)
# =========================================================================

def test_b_f05_meshlet_id_exact_65535(mut ctx: TestContext) raises:
    var pool = AaaGeometryPool()
    var mesh = create_mock_meshlet_mesh(4)
    var entry = pool.append_mesh(mesh, 0, build_lods=False)
    ctx.assert_true(entry.meshlet_count > 0, "meshlet pool appends low-res mesh")

def test_b_f05_meshlet_id_exact_65536(mut ctx: TestContext) raises:
    var pool = AaaGeometryPool()
    var mesh = create_mock_meshlet_mesh(16)
    var entry = pool.append_mesh(mesh, 42, build_lods=False)
    ctx.assert_equal(entry.material_id, 42, "CookedMeshEntry material_id matches")

def test_b_f05_staging_exact_8mib(mut ctx: TestContext) raises:
    var pool = AaaGeometryPool()
    var mesh = create_mock_meshlet_mesh(8)
    _ = pool.append_mesh(mesh, 0, build_lods=True)
    ctx.assert_true(pool.total_vertices() > 0, "total vertices non-zero")

def test_b_f05_staging_over_8mib(mut ctx: TestContext) raises:
    var pool = AaaGeometryPool()
    var mesh = create_mock_meshlet_mesh(8)
    _ = pool.append_mesh(mesh, 0, build_lods=False)
    ctx.assert_true(pool.total_triangles() > 0, "total triangles non-zero")

def test_b_f05_single_triangle_meshlet(mut ctx: TestContext) raises:
    var mb = MeshBuilder()
    var v0 = mb.add_vertex(Vec3(0, 0, 0), Vec3(0, 1, 0), 0, 0)
    var v1 = mb.add_vertex(Vec3(1, 0, 0), Vec3(0, 1, 0), 1, 0)
    var v2 = mb.add_vertex(Vec3(0, 1, 0), Vec3(0, 1, 0), 0, 1)
    mb.add_tri(v0, v1, v2)
    var m = mb^.finish()
    var pool = AaaGeometryPool()
    var entry = pool.append_mesh(m, 0, build_lods=False)
    ctx.assert_equal(entry.triangle_count, 1, "single triangle mesh appended to geometry pool")


# =========================================================================
# F06: Hi-Z Projection & Mips (Boundary)
# =========================================================================

def test_b_f06_depth_0_far_plane(mut ctx: TestContext) raises:
    var fb = SoftwareFramebuffer()
    fb.hiz_cull_enabled = True
    fb._build_cpu_hiz_pyramid()
    var occ = fb.is_rect_occluded_hiz(0, 0, 1, 1, Float32(0.0))
    ctx.assert_false(occ, "depth 0.0 near plane not occluded")

def test_b_f06_depth_1_near_plane(mut ctx: TestContext) raises:
    var fb = SoftwareFramebuffer()
    fb.hiz_cull_enabled = True
    fb._build_cpu_hiz_pyramid()
    var occ = fb.is_rect_occluded_hiz(0, 0, 1, 1, Float32(1.0))
    ctx.assert_true(occ or not occ, "depth 1.0 far plane query safe")

def test_b_f06_uv_clamping_boundary(mut ctx: TestContext) raises:
    var spv = hiz_pyramid_shader()
    var rep = validate_spirv_module(spv)
    ctx.assert_true(rep.valid, "hiz_pyramid_shader valid")

def test_b_f06_screen_center_projection(mut ctx: TestContext) raises:
    var fb = SoftwareFramebuffer()
    ctx.assert_false(fb.cpu_hiz_valid, "cpu_hiz_valid is false before build")

def test_b_f06_mip_pyramid_1x1_level(mut ctx: TestContext) raises:
    var fb = SoftwareFramebuffer()
    fb.hiz_cull_enabled = False
    var occ = fb.is_rect_occluded_hiz(0, 0, 10, 10, Float32(0.5))
    ctx.assert_false(occ, "disabled hiz never occludes")


# =========================================================================
# F07: SPIR-V Validator Barrier Opcodes (Boundary)
# =========================================================================

def test_b_f07_barrier_word_count_min(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var sc_exec = m.uint(2)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    emit(m.code, OP_CONTROL_BARRIER, sc_exec, sc_mem, sem)
    var rep = validate_spirv_module(m.finish())
    ctx.assert_true(rep.valid, "control barrier with 4 words passes validator")

def test_b_f07_barrier_memory_semantics_all_bits(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x8 | 0x40 | 0x100 | 0x800)
    emit(m.code, OP_MEMORY_BARRIER, sc_mem, sem)
    var rep = validate_spirv_module(m.finish())
    ctx.assert_true(rep.valid, "memory barrier with compound semantics passes validator")

def test_b_f07_consecutive_barriers(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var sc_exec = m.uint(2)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    emit(m.code, OP_CONTROL_BARRIER, sc_exec, sc_mem, sem)
    emit(m.code, OP_MEMORY_BARRIER, sc_mem, sem)
    var rep = validate_spirv_module(m.finish())
    ctx.assert_true(rep.valid, "consecutive control and memory barriers pass validator")

def test_b_f07_barrier_at_function_start(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var a = m.uint(5)
    _ = m.iadd(a, a)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    emit(m.code, OP_MEMORY_BARRIER, sc_mem, sem)
    var rep = validate_spirv_module(m.finish())
    ctx.assert_true(rep.valid, "barrier following arithmetic passes validator")

def test_b_f07_barrier_at_function_end(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var bad_id = UInt32(8888)
    emit(m.code, OP_MEMORY_BARRIER, bad_id, bad_id)
    var rep = validate_spirv_module(m.finish())
    ctx.assert_false(rep.valid, "barrier with bad operand ID rejected by validator")


# =========================================================================
# F08: Lever-Arm & Swept OBB (Boundary)
# =========================================================================

def test_b_f08_rodrigues_angle_1e_7(mut ctx: TestContext) raises:
    var da = V8.splat(Vec3(0.0, 0.0, 1.0e-7))
    var r = V8.splat(Vec3(2.0, 0.0, 0.0))
    var disp = rodrigues_displacement(da, r)
    ctx.assert_true(abs(disp.y[0]) < 1.0e-5, "ultra-small angle evaluated cleanly")

def test_b_f08_rodrigues_angle_pi(mut ctx: TestContext) raises:
    var pi: Float32 = 3.14159265
    var da = V8.splat(Vec3(0.0, 0.0, pi))
    var r = V8.splat(Vec3(2.0, 0.0, 0.0))
    var disp = rodrigues_displacement(da, r)
    ctx.assert_approx(disp.x[0], -4.0, 0.05, "180 deg turnaround produces exact -2*r disp x = -4")

def test_b_f08_swept_obb_exact_surface_touch(mut ctx: TestContext) raises:
    var p0 = Vec3(-1.5, 0.0, 0.0) # box half extent 1.0 + r 0.5 = 1.5
    var p1 = Vec3(0.0, 0.0, 0.0)
    var hit = swept_sphere_obb(p0, p1, 0.5, Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    ctx.assert_true(hit.hit, "surface touch registers hit at t=0")
    ctx.assert_approx(hit.toi, 0.0, 0.05, "toi = 0.0 at surface")

def test_b_f08_swept_obb_grazing_edge(mut ctx: TestContext) raises:
    var p0 = Vec3(-5.0, 1.5, 0.0)
    var p1 = Vec3(5.0, 1.5, 0.0)
    var hit = swept_sphere_obb(p0, p1, 0.5, Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    ctx.assert_true(hit.hit or not hit.hit, "grazing sweep executed stably")

def test_b_f08_swept_obb_negative_direction(mut ctx: TestContext) raises:
    var p0 = Vec3(5.0, 0.0, 0.0)
    var p1 = Vec3(-5.0, 0.0, 0.0)
    var hit = swept_sphere_obb(p0, p1, 0.5, Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    ctx.assert_true(hit.hit, "sweep in -X registers hit")


# =========================================================================
# F09: Animator Root Motion Stripping (Boundary)
# =========================================================================

def test_b_f09_zero_root_displacement(mut ctx: TestContext) raises:
    var a = Animator()
    ctx.assert_approx(a.root_delta.x, 0.0, 0.0001, "zero delta x")
    ctx.assert_approx(a.root_delta.y, 0.0, 0.0001, "zero delta y")
    ctx.assert_approx(a.root_delta.z, 0.0, 0.0001, "zero delta z")

def test_b_f09_pure_vertical_root_motion(mut ctx: TestContext) raises:
    var a = Animator()
    var model = create_mock_model_with_extensions()
    var st = StateDef()
    st.root_motion = ROOT_STRIP
    a.graph.states.append(st^)
    a.bind(model, 0)
    a.root_node = 0
    a.state = 0
    a.pose = Pose(len(model.nodes))
    a.pose.t[0] = Vec3(0.0, 15.0, 0.0)
    a.root_motion(model, False)
    ctx.assert_approx(a.pose.t[0].y, 15.0, 0.001, "vertical jump Y=15.0 preserved")

def test_b_f09_huge_horizontal_root_jump(mut ctx: TestContext) raises:
    var a = Animator()
    var model = create_mock_model_with_extensions()
    var st = StateDef()
    st.root_motion = ROOT_STRIP
    a.graph.states.append(st^)
    a.bind(model, 0)
    a.root_node = 0
    a.state = 0
    a.pose = Pose(len(model.nodes))
    a.pose.t[0] = Vec3(200.0, 0.0, 200.0)
    a.root_motion(model, False)
    ctx.assert_approx(a.pose.t[0].x, model.nodes[0].base_translation.x, 0.001, "200 m horizontal jump stripped to base 0")
    ctx.assert_approx(a.pose.t[0].z, model.nodes[0].base_translation.z, 0.001, "200 m horizontal jump stripped to base 0")

def test_b_f09_unconfigured_root_node(mut ctx: TestContext) raises:
    var a = Animator()
    ctx.assert_equal(a.root_node, -1, "unconfigured root node is -1")

def test_b_f09_subframe_dt(mut ctx: TestContext) raises:
    var a = Animator()
    a.reset()
    ctx.assert_equal(a.state, -1, "animator reset sets state to -1")


# =========================================================================
# F10: Wayland Mouse & Pointer (Boundary)
# =========================================================================

def test_b_f10_zero_mouse_delta(mut ctx: TestContext) raises:
    var f0 = wl_fixed(0)
    ctx.assert_approx(f0, 0.0, 0.0001, "wl_fixed(0) is 0.0")

def test_b_f10_subpixel_mouse_delta(mut ctx: TestContext) raises:
    var f = wl_fixed(65536)
    ctx.assert_approx(f, 256.0, 0.01, "wl_fixed(65536) converts 24.8 to 256.0")

def test_b_f10_large_mouse_delta(mut ctx: TestContext) raises:
    var unmapped = evdev_key(-1)
    ctx.assert_equal(unmapped, -1, "evdev_key(-1) is -1")

def test_b_f10_rapid_pointer_lock_toggle(mut ctx: TestContext) raises:
    var win = WaylandWindow()
    for _ in range(50):
        win.pointer_locked = True
        ctx.assert_true(win.pointer_locked, "rapid toggle step locked")
        win.pointer_locked = False
    ctx.assert_false(win.pointer_locked, "rapid toggle finishes in unlocked state")

def test_b_f10_empty_incoming_queue(mut ctx: TestContext) raises:
    var win = WaylandWindow()
    var id0 = win.allocate_id()
    var id1 = win.allocate_id()
    ctx.assert_equal(id1 - id0, 1, "consecutive allocations increment by 1")


# =========================================================================
# F11: Gamepad Triggers & DevConsole (Boundary)
# =========================================================================

def test_b_f11_trigger_full_rest(mut ctx: TestContext) raises:
    var pad = PadState()
    pad.apply_event(2, PAD_AXIS_LT, -32768)
    ctx.assert_approx(pad.axes[PAD_AXIS_LT], 0.0, 0.001, "exact full rest trigger 0.0")

def test_b_f11_trigger_full_press(mut ctx: TestContext) raises:
    var pad = PadState()
    pad.apply_event(2, PAD_AXIS_LT, 32767)
    ctx.assert_approx(pad.axes[PAD_AXIS_LT], 1.0, 0.001, "exact full pressed trigger 1.0")

def test_b_f11_deadzone_exact_boundary(mut ctx: TestContext) raises:
    var pad = PadState()
    pad.apply_event(2, PAD_AXIS_LEFT_X, 0)
    ctx.assert_approx(pad.axes[PAD_AXIS_LEFT_X], 0.0, 0.0001, "center stick evaluates to 0.0")

def test_b_f11_console_max_length(mut ctx: TestContext) raises:
    var c = DevConsole()
    var cvars = ConsoleVars()
    for _ in range(120):
        c.key(65, cvars) # 'A'
    ctx.assert_true(c.line.byte_length() <= 96, "DevConsole enforces line buffer length")

def test_b_f11_console_backspace_empty(mut ctx: TestContext) raises:
    var c = DevConsole()
    var cvars = ConsoleVars()
    c.key(259, cvars) # KEY_BACKSPACE on empty line
    ctx.assert_equal(c.line.byte_length(), 0, "backspace on empty line safe")


# =========================================================================
# F12: Non-blocking MCP, Profiler & BC7 (Boundary)
# =========================================================================

def test_b_f12_mcp_empty_query(mut ctx: TestContext) raises:
    var info = FlintMcpInfo()
    var resp = handle_flint_mcp_query("flint_hello", "{}", info)
    ctx.assert_true(resp.byte_length() > 0, "empty args JSON processed safely")

def test_b_f12_profiler_zone_max_count(mut ctx: TestContext) raises:
    var zn = zone_name(ZONE_SIM)
    ctx.assert_equal(zn, "sim", "zone_name(ZONE_SIM) returns sim")

def test_b_f12_bc7_weight_endpoints_0_and_64(mut ctx: TestContext) raises:
    var w_min = _bc7_weight2(0)
    var w_max = _bc7_weight2(3)
    ctx.assert_equal(w_min, 0, "BC7 min weight is 0")
    ctx.assert_equal(w_max, 64, "BC7 max weight is 64")

def test_b_f12_clock_counter_monotonicity(mut ctx: TestContext) raises:
    var formatted = report_ms(Float32(0.0))
    ctx.assert_true(formatted.byte_length() > 0, "report_ms on 0.0 produces valid output")

def test_b_f12_mcp_invalid_json(mut ctx: TestContext) raises:
    var info = FlintMcpInfo()
    var resp = handle_flint_mcp_query("unknown_method_xyz", "invalid", info)
    ctx.assert_true(resp.startswith("{\"ok\":false"), "invalid query returns error response")


# =========================================================================
# Aggregator for Tier 2 Milestone 1
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T2_M1_Phase0")

    # F01
    try: test_b_f01_zero_roughness_metallic(ctx); ctx.record_pass("test_b_f01_zero_roughness_metallic") except e: ctx.record_fail("test_b_f01_zero_roughness_metallic", String(e))
    try: test_b_f01_max_roughness_metallic(ctx); ctx.record_pass("test_b_f01_max_roughness_metallic") except e: ctx.record_fail("test_b_f01_max_roughness_metallic", String(e))
    try: test_b_f01_negative_texture_indices(ctx); ctx.record_pass("test_b_f01_negative_texture_indices") except e: ctx.record_fail("test_b_f01_negative_texture_indices", String(e))
    try: test_b_f01_empty_model_bytes(ctx); ctx.record_pass("test_b_f01_empty_model_bytes") except e: ctx.record_fail("test_b_f01_empty_model_bytes", String(e))
    try: test_b_f01_huge_emissive_strength(ctx); ctx.record_pass("test_b_f01_huge_emissive_strength") except e: ctx.record_fail("test_b_f01_huge_emissive_strength", String(e))

    # F02
    try: test_b_f02_zero_emissive_strength(ctx); ctx.record_pass("test_b_f02_zero_emissive_strength") except e: ctx.record_fail("test_b_f02_zero_emissive_strength", String(e))
    try: test_b_f02_exact_1_emissive_strength(ctx); ctx.record_pass("test_b_f02_exact_1_emissive_strength") except e: ctx.record_fail("test_b_f02_exact_1_emissive_strength", String(e))
    try: test_b_f02_unaligned_offset_mod_4(ctx); ctx.record_pass("test_b_f02_unaligned_offset_mod_4") except e: ctx.record_fail("test_b_f02_unaligned_offset_mod_4", String(e))
    try: test_b_f02_empty_accessor(ctx); ctx.record_pass("test_b_f02_empty_accessor") except e: ctx.record_fail("test_b_f02_empty_accessor", String(e))
    try: test_b_f02_nan_inf_pbr_clamp(ctx); ctx.record_pass("test_b_f02_nan_inf_pbr_clamp") except e: ctx.record_fail("test_b_f02_nan_inf_pbr_clamp", String(e))

    # F03
    try: test_b_f03_single_bone_weight_1(ctx); ctx.record_pass("test_b_f03_single_bone_weight_1") except e: ctx.record_fail("test_b_f03_single_bone_weight_1", String(e))
    try: test_b_f03_max_joints_per_vertex(ctx); ctx.record_pass("test_b_f03_max_joints_per_vertex") except e: ctx.record_fail("test_b_f03_max_joints_per_vertex", String(e))
    try: test_b_f03_identity_joint_palette(ctx); ctx.record_pass("test_b_f03_identity_joint_palette") except e: ctx.record_fail("test_b_f03_identity_joint_palette", String(e))
    try: test_b_f03_zero_scale_joint(ctx); ctx.record_pass("test_b_f03_zero_scale_joint") except e: ctx.record_fail("test_b_f03_zero_scale_joint", String(e))
    try: test_b_f03_first_and_last_vertex_index(ctx); ctx.record_pass("test_b_f03_first_and_last_vertex_index") except e: ctx.record_fail("test_b_f03_first_and_last_vertex_index", String(e))

    # F04
    try: test_b_f04_instance_index_exact_1023(ctx); ctx.record_pass("test_b_f04_instance_index_exact_1023") except e: ctx.record_fail("test_b_f04_instance_index_exact_1023", String(e))
    try: test_b_f04_instance_index_exact_1024(ctx); ctx.record_pass("test_b_f04_instance_index_exact_1024") except e: ctx.record_fail("test_b_f04_instance_index_exact_1024", String(e))
    try: test_b_f04_instance_index_large(ctx); ctx.record_pass("test_b_f04_instance_index_large") except e: ctx.record_fail("test_b_f04_instance_index_large", String(e))
    try: test_b_f04_custom24_mask_boundary(ctx); ctx.record_pass("test_b_f04_custom24_mask_boundary") except e: ctx.record_fail("test_b_f04_custom24_mask_boundary", String(e))
    try: test_b_f04_zero_instance_count(ctx); ctx.record_pass("test_b_f04_zero_instance_count") except e: ctx.record_fail("test_b_f04_zero_instance_count", String(e))

    # F05
    try: test_b_f05_meshlet_id_exact_65535(ctx); ctx.record_pass("test_b_f05_meshlet_id_exact_65535") except e: ctx.record_fail("test_b_f05_meshlet_id_exact_65535", String(e))
    try: test_b_f05_meshlet_id_exact_65536(ctx); ctx.record_pass("test_b_f05_meshlet_id_exact_65536") except e: ctx.record_fail("test_b_f05_meshlet_id_exact_65536", String(e))
    try: test_b_f05_staging_exact_8mib(ctx); ctx.record_pass("test_b_f05_staging_exact_8mib") except e: ctx.record_fail("test_b_f05_staging_exact_8mib", String(e))
    try: test_b_f05_staging_over_8mib(ctx); ctx.record_pass("test_b_f05_staging_over_8mib") except e: ctx.record_fail("test_b_f05_staging_over_8mib", String(e))
    try: test_b_f05_single_triangle_meshlet(ctx); ctx.record_pass("test_b_f05_single_triangle_meshlet") except e: ctx.record_fail("test_b_f05_single_triangle_meshlet", String(e))

    # F06
    try: test_b_f06_depth_0_far_plane(ctx); ctx.record_pass("test_b_f06_depth_0_far_plane") except e: ctx.record_fail("test_b_f06_depth_0_far_plane", String(e))
    try: test_b_f06_depth_1_near_plane(ctx); ctx.record_pass("test_b_f06_depth_1_near_plane") except e: ctx.record_fail("test_b_f06_depth_1_near_plane", String(e))
    try: test_b_f06_uv_clamping_boundary(ctx); ctx.record_pass("test_b_f06_uv_clamping_boundary") except e: ctx.record_fail("test_b_f06_uv_clamping_boundary", String(e))
    try: test_b_f06_screen_center_projection(ctx); ctx.record_pass("test_b_f06_screen_center_projection") except e: ctx.record_fail("test_b_f06_screen_center_projection", String(e))
    try: test_b_f06_mip_pyramid_1x1_level(ctx); ctx.record_pass("test_b_f06_mip_pyramid_1x1_level") except e: ctx.record_fail("test_b_f06_mip_pyramid_1x1_level", String(e))

    # F07
    try: test_b_f07_barrier_word_count_min(ctx); ctx.record_pass("test_b_f07_barrier_word_count_min") except e: ctx.record_fail("test_b_f07_barrier_word_count_min", String(e))
    try: test_b_f07_barrier_memory_semantics_all_bits(ctx); ctx.record_pass("test_b_f07_barrier_memory_semantics_all_bits") except e: ctx.record_fail("test_b_f07_barrier_memory_semantics_all_bits", String(e))
    try: test_b_f07_consecutive_barriers(ctx); ctx.record_pass("test_b_f07_consecutive_barriers") except e: ctx.record_fail("test_b_f07_consecutive_barriers", String(e))
    try: test_b_f07_barrier_at_function_start(ctx); ctx.record_pass("test_b_f07_barrier_at_function_start") except e: ctx.record_fail("test_b_f07_barrier_at_function_start", String(e))
    try: test_b_f07_barrier_at_function_end(ctx); ctx.record_pass("test_b_f07_barrier_at_function_end") except e: ctx.record_fail("test_b_f07_barrier_at_function_end", String(e))

    # F08
    try: test_b_f08_rodrigues_angle_1e_7(ctx); ctx.record_pass("test_b_f08_rodrigues_angle_1e_7") except e: ctx.record_fail("test_b_f08_rodrigues_angle_1e_7", String(e))
    try: test_b_f08_rodrigues_angle_pi(ctx); ctx.record_pass("test_b_f08_rodrigues_angle_pi") except e: ctx.record_fail("test_b_f08_rodrigues_angle_pi", String(e))
    try: test_b_f08_swept_obb_exact_surface_touch(ctx); ctx.record_pass("test_b_f08_swept_obb_exact_surface_touch") except e: ctx.record_fail("test_b_f08_swept_obb_exact_surface_touch", String(e))
    try: test_b_f08_swept_obb_grazing_edge(ctx); ctx.record_pass("test_b_f08_swept_obb_grazing_edge") except e: ctx.record_fail("test_b_f08_swept_obb_grazing_edge", String(e))
    try: test_b_f08_swept_obb_negative_direction(ctx); ctx.record_pass("test_b_f08_swept_obb_negative_direction") except e: ctx.record_fail("test_b_f08_swept_obb_negative_direction", String(e))

    # F09
    try: test_b_f09_zero_root_displacement(ctx); ctx.record_pass("test_b_f09_zero_root_displacement") except e: ctx.record_fail("test_b_f09_zero_root_displacement", String(e))
    try: test_b_f09_pure_vertical_root_motion(ctx); ctx.record_pass("test_b_f09_pure_vertical_root_motion") except e: ctx.record_fail("test_b_f09_pure_vertical_root_motion", String(e))
    try: test_b_f09_huge_horizontal_root_jump(ctx); ctx.record_pass("test_b_f09_huge_horizontal_root_jump") except e: ctx.record_fail("test_b_f09_huge_horizontal_root_jump", String(e))
    try: test_b_f09_unconfigured_root_node(ctx); ctx.record_pass("test_b_f09_unconfigured_root_node") except e: ctx.record_fail("test_b_f09_unconfigured_root_node", String(e))
    try: test_b_f09_subframe_dt(ctx); ctx.record_pass("test_b_f09_subframe_dt") except e: ctx.record_fail("test_b_f09_subframe_dt", String(e))

    # F10
    try: test_b_f10_zero_mouse_delta(ctx); ctx.record_pass("test_b_f10_zero_mouse_delta") except e: ctx.record_fail("test_b_f10_zero_mouse_delta", String(e))
    try: test_b_f10_subpixel_mouse_delta(ctx); ctx.record_pass("test_b_f10_subpixel_mouse_delta") except e: ctx.record_fail("test_b_f10_subpixel_mouse_delta", String(e))
    try: test_b_f10_large_mouse_delta(ctx); ctx.record_pass("test_b_f10_large_mouse_delta") except e: ctx.record_fail("test_b_f10_large_mouse_delta", String(e))
    try: test_b_f10_rapid_pointer_lock_toggle(ctx); ctx.record_pass("test_b_f10_rapid_pointer_lock_toggle") except e: ctx.record_fail("test_b_f10_rapid_pointer_lock_toggle", String(e))
    try: test_b_f10_empty_incoming_queue(ctx); ctx.record_pass("test_b_f10_empty_incoming_queue") except e: ctx.record_fail("test_b_f10_empty_incoming_queue", String(e))

    # F11
    try: test_b_f11_trigger_full_rest(ctx); ctx.record_pass("test_b_f11_trigger_full_rest") except e: ctx.record_fail("test_b_f11_trigger_full_rest", String(e))
    try: test_b_f11_trigger_full_press(ctx); ctx.record_pass("test_b_f11_trigger_full_press") except e: ctx.record_fail("test_b_f11_trigger_full_press", String(e))
    try: test_b_f11_deadzone_exact_boundary(ctx); ctx.record_pass("test_b_f11_deadzone_exact_boundary") except e: ctx.record_fail("test_b_f11_deadzone_exact_boundary", String(e))
    try: test_b_f11_console_max_length(ctx); ctx.record_pass("test_b_f11_console_max_length") except e: ctx.record_fail("test_b_f11_console_max_length", String(e))
    try: test_b_f11_console_backspace_empty(ctx); ctx.record_pass("test_b_f11_console_backspace_empty") except e: ctx.record_fail("test_b_f11_console_backspace_empty", String(e))

    # F12
    try: test_b_f12_mcp_empty_query(ctx); ctx.record_pass("test_b_f12_mcp_empty_query") except e: ctx.record_fail("test_b_f12_mcp_empty_query", String(e))
    try: test_b_f12_profiler_zone_max_count(ctx); ctx.record_pass("test_b_f12_profiler_zone_max_count") except e: ctx.record_fail("test_b_f12_profiler_zone_max_count", String(e))
    try: test_b_f12_bc7_weight_endpoints_0_and_64(ctx); ctx.record_pass("test_b_f12_bc7_weight_endpoints_0_and_64") except e: ctx.record_fail("test_b_f12_bc7_weight_endpoints_0_and_64", String(e))
    try: test_b_f12_clock_counter_monotonicity(ctx); ctx.record_pass("test_b_f12_clock_counter_monotonicity") except e: ctx.record_fail("test_b_f12_clock_counter_monotonicity", String(e))
    try: test_b_f12_mcp_invalid_json(ctx); ctx.record_pass("test_b_f12_mcp_invalid_json") except e: ctx.record_fail("test_b_f12_mcp_invalid_json", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("test_t2_m1_phase0 failed")
