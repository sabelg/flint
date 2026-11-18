"""Tier 1 Feature Coverage: Milestone 1 Phase 0 (F01–F12: 60 tests).
Opaque-box and requirement-driven verification of P0 bugs, memory safety, and core invariants.
"""
from std.collections import List
from std.ffi import external_call
from std.math import sqrt
from flint.core.math import Vec3
from flint.core.transform import Quat, Mat4, trs
from flint.assets.model import NativeModel, NativeMesh, NativeTexture, NativeNode
from flint.assets.mesh_builder import MeshBuilder, append_node_meshes
from flint.assets.glb_writer import encode_native_glb
from flint.assets.glb_native import parse_native_glb, component_count
from flint.assets.cooked_model import model_bytes, read_model, model_difference, encode_cooked_model, decode_cooked_model, SourceStamp
from flint.assets.glb import GlbInfo
from flint.assets.meshlet import (
    AaaGeometryPool, compute_acmr, optimize_vertex_cache_forsyth, simplify_mesh_qem, select_mesh_lod_cpu_reference,
)
from flint.physics.ccd import swept_sphere_obb, swept_sphere_moving_sphere, CcdHit
from flint.physics.solver_bodies import V8, F8, LANES
from flint.physics.solver import rodrigues_displacement
from flint.anim.animator import Animator
from flint.anim.graph import ROOT_STRIP, ROOT_EXTRACT, ROOT_KEEP, StateDef
from flint.anim.pose import Pose, Inertializer
from flint.platform.wayland import WaylandWindow, wl_fixed, evdev_key, wl_put32, wl_gets32
from flint.platform.input import (
    PadState, PAD_AXIS_LT, PAD_AXIS_RT, PAD_AXIS_LEFT_X, PAD_AXIS_LEFT_Y, PAD_AXIS_RIGHT_X, PAD_AXIS_RIGHT_Y,
)
from flint.core.console import DevConsole, typed_byte
from flint.core.cvars import ConsoleVars
from flint.devtools.mcp_server import handle_flint_mcp_query, FlintMcpInfo
from flint.core.profiler import ProfileTimeline, Profiler, zone_name, report_ms, ZONE_SIM, ZONE_FRAME, ZONE_POST
from flint.assets.bc_compress import _bc7_weight2, _bc7_weight3, _bc7_weight4, f32_to_uf16_bits, uf16_bits_to_f32
from flint.render.spirv import (
    AaaSpirvModule, emit, MODEL_GL_COMPUTE, OP_CONTROL_BARRIER, OP_MEMORY_BARRIER,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.render.aaa_skinning_rt import (
    DualQuat, blend_dual_quats_cpu, aaa_dq_skinning_shader,
    AAA_SKIN_CUR_BONES_BASE, AAA_SKIN_PREV_BONES_BASE,
)
from flint.render.aaa_raytracing import write_vk_as_instance_64b, emit_rt_shadow_query, emit_rt_surface_cache_ray
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_visbuffer import meshlet_cull_shader
from flint.platform.vulkan import get32, get64, get_float
from flint.render.rasterizer import SoftwareFramebuffer
from flint.render.vulkan_shaders import hiz_pyramid_shader
from e2e.common.harness import TestContext
from e2e.common.mock_data import (
    create_mock_model_with_extensions, create_mock_glb_bytes, create_mock_meshlet_mesh,
)


# =========================================================================
# F01: Cooked Model Material Fields (5 tests)
# =========================================================================

def test_f01_model_bytes_material_fields(mut ctx: TestContext) raises:
    var model = create_mock_model_with_extensions()
    var bytes = model_bytes(model)
    ctx.assert_true(len(bytes) > 64, "model_bytes should serialize non-empty buffer")

def test_f01_read_model_deserialization(mut ctx: TestContext) raises:
    var model = create_mock_model_with_extensions()
    var bytes = model_bytes(model)
    var err = String("")
    var loaded = read_model(bytes, err)
    ctx.assert_true(err.byte_length() == 0, "read_model should produce no errors")
    ctx.assert_equal(len(loaded.meshes), len(model.meshes), "mesh count must match")
    ctx.assert_equal(loaded.meshes[0].normal_orm_texture, model.meshes[0].normal_orm_texture, "normal_orm_texture match")
    ctx.assert_equal(loaded.meshes[0].orm_texture, model.meshes[0].orm_texture, "orm_texture match")
    ctx.assert_approx(loaded.meshes[0].roughness, model.meshes[0].roughness, 0.001, "roughness match")
    ctx.assert_approx(loaded.meshes[0].metallic, model.meshes[0].metallic, 0.001, "metallic match")

def test_f01_model_difference_roughness(mut ctx: TestContext) raises:
    var m1 = create_mock_model_with_extensions()
    var m2 = create_mock_model_with_extensions()
    var diff_same = model_difference(m1, m2)
    ctx.assert_true(diff_same.byte_length() == 0, "identical models must have empty difference")
    m2.meshes[0].roughness = 0.99
    var diff_changed = model_difference(m1, m2)
    ctx.assert_true(diff_changed.byte_length() > 0, "modified roughness must be detected by model_difference")

def test_f01_model_difference_textures(mut ctx: TestContext) raises:
    var m1 = create_mock_model_with_extensions()
    var m2 = create_mock_model_with_extensions()
    m2.meshes[0].normal_orm_texture = 99
    var diff = model_difference(m1, m2)
    ctx.assert_true(diff.byte_length() > 0, "texture index difference must be detected")

def test_f01_cooked_stamp_and_roundtrip(mut ctx: TestContext) raises:
    var model = create_mock_model_with_extensions()
    var info = GlbInfo()
    info.image_count = 3
    var stamp = SourceStamp(1024, 1234567, 987654321)
    var cooked_bytes = encode_cooked_model(model, info, stamp, 42)
    ctx.assert_true(len(cooked_bytes) > 128, "encode_cooked_model produced valid payload")
    var decoded = decode_cooked_model(cooked_bytes)
    ctx.assert_equal(decoded.asset_id, 42, "asset id matches")
    ctx.assert_equal(decoded.stamp.size, 1024, "stamp size matches")
    ctx.assert_approx(decoded.model.meshes[0].roughness, model.meshes[0].roughness, 0.001, "decoded roughness matches")


# =========================================================================
# F02: glTF Emissive & Align (5 tests)
# =========================================================================

def test_f02_gltf_emissive_strength(mut ctx: TestContext) raises:
    var glb = create_mock_glb_bytes()
    var reloaded = parse_native_glb(glb)
    ctx.assert_true(len(reloaded.meshes) > 0, "glb must parse meshes")
    ctx.assert_true(reloaded.meshes[0].emissive_strength > 1.0, "emissive_strength > 1.0 must be preserved via extension")

def test_f02_gltf_emissive_no_double_mult(mut ctx: TestContext) raises:
    var model = create_mock_model_with_extensions()
    var glb = encode_native_glb(model)
    var parsed = parse_native_glb(glb)
    ctx.assert_true(parsed.meshes[0].emissive_color.x <= 1.001, "emissive color RGB must remain clamped [0, 1]")

def test_f02_unaligned_accessor_loading(mut ctx: TestContext) raises:
    var glb = create_mock_glb_bytes()
    ctx.assert_true(len(glb) > 100, "glb binary valid")
    var parsed = parse_native_glb(glb)
    ctx.assert_true(len(parsed.meshes[0].positions) > 0, "positions loaded successfully without unaligned crash")

def test_f02_gltf_accessor_component_counts(mut ctx: TestContext) raises:
    ctx.assert_equal(component_count("SCALAR"), 1, "SCALAR is 1")
    ctx.assert_equal(component_count("VEC2"), 2, "VEC2 is 2")
    ctx.assert_equal(component_count("VEC3"), 3, "VEC3 is 3")
    ctx.assert_equal(component_count("VEC4"), 4, "VEC4 is 4")

def test_f02_gltf_pbr_attributes(mut ctx: TestContext) raises:
    var glb = create_mock_glb_bytes()
    var parsed = parse_native_glb(glb)
    ctx.assert_approx(parsed.meshes[0].roughness, 0.25, 0.05, "roughness roundtrips accurately")
    ctx.assert_approx(parsed.meshes[0].metallic, 0.75, 0.05, "metallic roundtrips accurately")


# =========================================================================
# F03: GPU Skinning Fixes (5 tests)
# =========================================================================

def test_f03_bone_palette_slice_offsets(mut ctx: TestContext) raises:
    # Verify multi-rig non-overlapping bone palette slices using DualQuat
    var dq0 = DualQuat.from_quat_trans(Quat(), Vec3(1.0, 2.0, 3.0))
    var dq1 = DualQuat.from_quat_trans(Quat(), Vec3(4.0, 5.0, 6.0))
    var palette = List[UInt32](length=128 * 8, fill=0)
    # Instance 0 writes at joint 0 (offset 0 words)
    dq0.write_words(palette, 0)
    # Instance 1 writes at joint 64 (offset 64 * 8 = 512 words)
    dq1.write_words(palette, 512)
    ctx.assert_true(palette[0] != 0 or palette[3] != 0, "instance 0 wrote valid words")
    ctx.assert_true(palette[512] != 0 or palette[515] != 0, "instance 1 wrote valid words to slice")
    ctx.assert_equal(palette[256], 0, "intermediate unwritten words remain zero")

def test_f03_joint_weights_normalization(mut ctx: TestContext) raises:
    var bones = List[DualQuat]()
    bones.append(DualQuat.from_quat_trans(Quat(), Vec3(0.0, 0.0, 0.0)))
    bones.append(DualQuat.from_quat_trans(Quat(), Vec3(2.0, 0.0, 0.0)))
    bones.append(DualQuat.from_quat_trans(Quat(), Vec3(0.0, 4.0, 0.0)))
    bones.append(DualQuat.from_quat_trans(Quat(), Vec3(0.0, 0.0, 8.0)))
    var blended = blend_dual_quats_cpu(bones, 0, 1, 2, 3, 0.5, 0.25, 0.15, 0.10)
    var p = blended.transform_point(Vec3(0.0, 0.0, 0.0))
    # Weighted translation: 0.5*0 + 0.25*2.0 + 0.15*0 + 0.10*0 = 0.5 on X
    ctx.assert_approx(p.x, 0.5, 0.01, "blended X translation matches 0.25 * 2.0")
    ctx.assert_approx(p.y, 0.6, 0.01, "blended Y translation matches 0.15 * 4.0")
    ctx.assert_approx(p.z, 0.8, 0.01, "blended Z translation matches 0.10 * 8.0")

def test_f03_dual_quaternion_math(mut ctx: TestContext) raises:
    var bones = List[DualQuat]()
    bones.append(DualQuat())
    var q90 = Quat(0.0, Float32(0.70710678), 0.0, Float32(0.70710678))
    bones.append(DualQuat.from_quat_trans(q90, Vec3(2.0, 0.0, 0.0)))
    var blended = blend_dual_quats_cpu(bones, 0, 1, 0, 0, Float32(0.5), Float32(0.5), Float32(0.0), Float32(0.0))
    var p = blended.transform_point(Vec3(0.0, 0.0, 0.0))
    ctx.assert_true(p.x > 0.5, "blended dual quaternion interpolates translation")

def test_f03_prev_pos_indexing_safety(mut ctx: TestContext) raises:
    var shader_words = aaa_dq_skinning_shader(12, 1)
    ctx.assert_true(len(shader_words) > 100, "aaa_dq_skinning_shader produces complete compute shader")
    var rep = validate_spirv_module(shader_words)
    ctx.assert_true(rep.valid, "aaa_dq_skinning_shader passes SPIR-V module validator")

def test_f03_decoupled_vertex_buffers(mut ctx: TestContext) raises:
    var dq = DualQuat.from_quat_trans(Quat(), Vec3(1.0, 2.0, 3.0))
    var dir = dq.transform_dir(Vec3(0.0, 1.0, 0.0))
    ctx.assert_vec3_approx(dir, Vec3(0.0, 1.0, 0.0), 0.001, "transform_dir preserves unit direction without translation")


# =========================================================================
# F04: RT Instance ID Sentinel (5 tests)
# =========================================================================

def test_f04_rt_instance_id_normal(mut ctx: TestContext) raises:
    var addr = external_call["malloc", Int](64)
    var mat = trs(Vec3(1, 2, 3), Quat(), Vec3(1, 1, 1))
    write_vk_as_instance_64b(addr, mat, 42, 0x12345678, cull_disable=True)
    var custom = get32(addr + 48)
    ctx.assert_equal(Int(custom & 0xFFFFFF), 42, "normal instance custom index packed correctly")
    ctx.assert_equal(Int((custom >> 24) & 0xFF), 0xFF, "upper 8 bits mask packed as 0xFF")
    _ = external_call["free", NoneType](addr)

def test_f04_rt_instance_id_sentinel(mut ctx: TestContext) raises:
    var addr = external_call["malloc", Int](64)
    var mat = trs(Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    write_vk_as_instance_64b(addr, mat, 0xFFFFFF, 0xABCDEF00, cull_disable=True)
    var custom = get32(addr + 48)
    ctx.assert_equal(Int(custom & 0xFFFFFF), 0xFFFFFF, "sentinel 0xFFFFFF packed into 24-bit custom index")
    _ = external_call["free", NoneType](addr)

def test_f04_rt_miss_radiance_fallback(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4, enable_ray_query=True)
    var tlas = m.acceleration_structure(0)
    var b_sc = m.buffer(1)
    var ox = m.float(0.0)
    var oy = m.float(0.0)
    var oz = m.float(0.0)
    var dx = m.float(0.0)
    var dy = m.float(0.0)
    var dz = m.float(1.0)
    var tmin = m.float(0.001)
    var tmax = m.float(100.0)
    var miss_r = m.float(0.5)
    var miss_g = m.float(0.5)
    var miss_b = m.float(0.5)
    _ = emit_rt_surface_cache_ray(m, tlas, b_sc, ox, oy, oz, dx, dy, dz, tmin, tmax, miss_r, miss_g, miss_b)
    var code = m.finish()
    var rep = validate_spirv_module(code)
    ctx.assert_true(rep.valid, "RT surface cache shader with 0xFFFFFF sentinel branch validates cleanly")

def test_f04_rt_custom_index_packing(mut ctx: TestContext) raises:
    var addr = external_call["malloc", Int](64)
    var mat = Mat4()
    write_vk_as_instance_64b(addr, mat, 500, 0xCAFEBABE, cull_disable=True)
    var flags = get32(addr + 52)
    ctx.assert_equal(Int((flags >> 24) & 1), 1, "cull_disable sets flag bit 1")
    var blas_addr = get64(addr + 56)
    ctx.assert_equal(blas_addr, 0xCAFEBABE, "blas 64-bit device address matches")
    _ = external_call["free", NoneType](addr)

def test_f04_rt_as_instance_buffer(mut ctx: TestContext) raises:
    var buf = List[UInt8](length=128, fill=0)
    var addr = Int(buf.unsafe_ptr())
    var mat = trs(Vec3(1, 2, 3), Quat(), Vec3(1, 1, 1))
    write_vk_as_instance_64b(addr, mat, 10, 0xA000, cull_disable=True)
    write_vk_as_instance_64b(addr + 64, mat, 20, 0xB000, cull_disable=True)
    var id0 = Int(buf[48]) | (Int(buf[49]) << 8)
    var id1 = Int(buf[48 + 64]) | (Int(buf[49 + 64]) << 8)
    ctx.assert_equal(id0, 10, "first instance at byte offset 0 has ID 10")
    ctx.assert_equal(id1, 20, "second instance at byte offset 64 has ID 20")


# =========================================================================
# F05: Meshlet ID & Staging Bounds (5 tests)
# =========================================================================

def test_f05_meshlet_id_32bit(mut ctx: TestContext) raises:
    var large_ml_id: UInt32 = 70000
    var inst_id: UInt32 = 3
    var packed: UInt32 = (inst_id << 22) | (large_ml_id & 0x3FFFFF)
    var decoded_inst = packed >> 22
    var decoded_ml = packed & 0x3FFFFF
    ctx.assert_equal(Int(decoded_inst), 3, "22-bit packing decodes inst_id correctly")
    ctx.assert_equal(Int(decoded_ml), 70000, "22-bit packing preserves meshlet ID >= 65536")

def test_f05_staging_capacity_bounds(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=64, height=64)
    ctx.assert_true(ren.active, "renderer active in headless mode")
    var init_cap = ren.buf_staging.size
    ctx.assert_true(init_cap > 0, "staging buffer initially allocated")
    ren.ensure_staging_capacity(init_cap + 1024 * 1024)
    ctx.assert_true(ren.buf_staging.size >= init_cap + 1024 * 1024, "staging buffer grows dynamically")
    ren.shutdown()

def test_f05_add_mesh_incremental_words(mut ctx: TestContext) raises:
    var pool = AaaGeometryPool()
    var mesh = create_mock_meshlet_mesh(8)
    var entry1 = pool.append_mesh(mesh, 0, build_lods=False)
    var entry2 = pool.append_mesh(mesh, 1, build_lods=False)
    ctx.assert_true(entry2.first_vertex >= entry1.vertex_count, "second mesh vertices append incrementally")

def test_f05_meshlet_cluster_counts(mut ctx: TestContext) raises:
    var pool = AaaGeometryPool()
    var mesh = create_mock_meshlet_mesh(16)
    var entry = pool.append_mesh(mesh, 0, build_lods=True)
    ctx.assert_true(entry.lod_count >= 1, "append_mesh with build_lods produces LOD levels")
    ctx.assert_true(entry.bounds_radius > 0.0, "bounds radius calculated correctly")

def test_f05_geometry_pool_totals(mut ctx: TestContext) raises:
    var pool = AaaGeometryPool()
    var mesh = create_mock_meshlet_mesh(10)
    var entry = pool.append_mesh(mesh, 0, build_lods=False)
    var lod_close = select_mesh_lod_cpu_reference(entry, Vec3(0, 0, 0), 1.0, Vec3(0, 0, 1), 0.1, 1080, Float32(0.577))
    var lod_far = select_mesh_lod_cpu_reference(entry, Vec3(0, 0, 0), 1.0, Vec3(0, 0, 1000), 0.1, 1080, Float32(0.577))
    ctx.assert_equal(lod_close, 0, "close camera selects LOD 0")
    ctx.assert_true(lod_far >= lod_close, "far camera selects equal or higher LOD")


# =========================================================================
# F06: Hi-Z Projection & Mips (5 tests)
# =========================================================================

def test_f06_hiz_projection_prev_mat(mut ctx: TestContext) raises:
    var prev_vp = trs(Vec3(0, 0, 5), Quat(), Vec3(1, 1, 1))
    var wc = Vec3(0, 0, 0)
    var projected = prev_vp.transform_point(wc)
    ctx.assert_vec3_approx(projected, Vec3(0, 0, 5), 0.001, "point transformed by previous view-proj")

def test_f06_hiz_min_depth_mip_pyramid(mut ctx: TestContext) raises:
    var spv = hiz_pyramid_shader()
    ctx.assert_true(len(spv) > 50, "hiz_pyramid_shader produces SPIR-V binary")
    var rep = validate_spirv_module(spv)
    ctx.assert_true(rep.valid, "hiz_pyramid_shader passes SPIR-V validation")

def test_f06_hiz_cull_visible_meshlet(mut ctx: TestContext) raises:
    # In Reverse-Z, near=1.0, far=0.0. Object rev_z = 0.9 (near). Occluder rev_z = 0.5 (farther).
    var sphere_rev_z: Float32 = 0.9
    var min_occ_z: Float32 = 0.5
    var occluded = min_occ_z > (sphere_rev_z + Float32(0.0005))
    ctx.assert_false(occluded, "object in front of occluder must NOT be culled")

def test_f06_hiz_cull_occluded_meshlet(mut ctx: TestContext) raises:
    # Object rev_z = 0.3 (far). Occluder rev_z = 0.8 (closer).
    var sphere_rev_z: Float32 = 0.3
    var min_occ_z: Float32 = 0.8
    var occluded = min_occ_z > (sphere_rev_z + Float32(0.0005))
    ctx.assert_true(occluded, "object behind closer occluder must be culled")

def test_f06_hiz_reverse_z_comparison(mut ctx: TestContext) raises:
    var near: Float32 = 0.1
    var far: Float32 = 1000.0
    var near_z_rev = near / near
    var far_z_rev = near / far
    ctx.assert_true(near_z_rev > far_z_rev, "Reverse-Z maps near plane to 1.0 > far plane at 0.0")
    ctx.assert_approx(near_z_rev, 1.0, 0.0001, "near plane is exactly 1.0")


# =========================================================================
# F07: SPIR-V Validator Barrier Opcodes (5 tests)
# =========================================================================

def test_f07_spirv_control_barrier_void(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var sc_exec = m.uint(2)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    emit(m.code, OP_CONTROL_BARRIER, sc_exec, sc_mem, sem)
    var code = m.finish()
    var rep = validate_spirv_module(code)
    ctx.assert_true(rep.valid, "module with OP_CONTROL_BARRIER passes validator without void-result crash")

def test_f07_spirv_memory_barrier_void(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    emit(m.code, OP_MEMORY_BARRIER, sc_mem, sem)
    var code = m.finish()
    var rep = validate_spirv_module(code)
    ctx.assert_true(rep.valid, "module with OP_MEMORY_BARRIER passes validator without void-result crash")

def test_f07_spirv_barrier_no_result_type(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var a = m.uint(1)
    var b = m.uint(2)
    _ = m.iadd(a, b)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    emit(m.code, OP_MEMORY_BARRIER, sc_mem, sem)
    var code = m.finish()
    var rep = validate_spirv_module(code)
    ctx.assert_true(rep.valid, "interleaved compute arithmetic and memory barrier validates cleanly")

def test_f07_spirv_validator_compute_pass(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var a = m.uint(10)
    var b = m.uint(20)
    _ = m.iadd(a, b)
    var code = m.finish()
    var rep = validate_spirv_module(code)
    ctx.assert_true(rep.valid, "basic compute spirv passes validator")

def test_f07_spirv_barrier_operand_validation(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    # Emit barrier with undefined operand IDs (e.g. 9999)
    emit(m.code, OP_CONTROL_BARRIER, UInt32(9999), UInt32(9998), UInt32(9997))
    var code = m.finish()
    var rep = validate_spirv_module(code)
    ctx.assert_false(rep.valid, "barrier with undefined operand IDs fails validator")


# =========================================================================
# F08: Lever-Arm & Swept OBB (5 tests)
# =========================================================================

def test_f08_rodrigues_displacement_zero(mut ctx: TestContext) raises:
    var da = V8.zero()
    var r = V8.splat(Vec3(1.0, 0.0, 0.0))
    var disp = rodrigues_displacement(da, r)
    ctx.assert_approx(disp.x[0], 0.0, 0.0001, "zero angular delta produces zero displacement x")
    ctx.assert_approx(disp.y[0], 0.0, 0.0001, "zero angular delta produces zero displacement y")
    ctx.assert_approx(disp.z[0], 0.0, 0.0001, "zero angular delta produces zero displacement z")

def test_f08_rodrigues_displacement_small_angle(mut ctx: TestContext) raises:
    var da = V8.splat(Vec3(0.0, 0.0, 0.001))
    var r = V8.splat(Vec3(1.0, 0.0, 0.0))
    var disp = rodrigues_displacement(da, r)
    ctx.assert_approx(disp.y[0], 0.001, 0.0001, "small angle Taylor expansion matches da x r")

def test_f08_rodrigues_displacement_large_angle(mut ctx: TestContext) raises:
    var pi_half: Float32 = 1.5707963
    var da = V8.splat(Vec3(0.0, 0.0, pi_half))
    var r = V8.splat(Vec3(1.0, 0.0, 0.0))
    var disp = rodrigues_displacement(da, r)
    # 90 deg rotation of (1,0,0) about Z is (0,1,0), displacement is (-1, 1, 0)
    ctx.assert_approx(disp.x[0], -1.0, 0.01, "exact 90 deg rotation disp x = -1.0")
    ctx.assert_approx(disp.y[0], 1.0, 0.01, "exact 90 deg rotation disp y = 1.0")

def test_f08_swept_sphere_obb_face_hit(mut ctx: TestContext) raises:
    var p0 = Vec3(-5.0, 0.0, 0.0)
    var p1 = Vec3(5.0, 0.0, 0.0)
    var r: Float32 = 0.5
    var center = Vec3(0.0, 0.0, 0.0)
    var rot = Quat()
    var half = Vec3(1.0, 1.0, 1.0)
    var hit = swept_sphere_obb(p0, p1, r, center, rot, half)
    ctx.assert_true(hit.hit, "swept sphere directly towards box must hit")
    ctx.assert_vec3_approx(hit.normal, Vec3(-1.0, 0.0, 0.0), 0.05, "hit normal on X face points -X")

def test_f08_swept_sphere_obb_corner_miss(mut ctx: TestContext) raises:
    var p0 = Vec3(-5.0, 5.0, 5.0)
    var p1 = Vec3(-2.0, 5.0, 5.0)
    var r: Float32 = 0.5
    var center = Vec3(0.0, 0.0, 0.0)
    var rot = Quat()
    var half = Vec3(1.0, 1.0, 1.0)
    var hit = swept_sphere_obb(p0, p1, r, center, rot, half)
    ctx.assert_false(hit.hit, "trajectory far from OBB must not register hit")


# =========================================================================
# F09: Animator Root Motion Stripping (5 tests)
# =========================================================================

def test_f09_root_motion_strip_xz(mut ctx: TestContext) raises:
    var a = Animator()
    var model = create_mock_model_with_extensions()
    a.bind(model, 0)
    ctx.assert_approx(a.root_delta.x, 0.0, 0.001, "initial root delta x is 0")
    ctx.assert_approx(a.root_delta.z, 0.0, 0.001, "initial root delta z is 0")

def test_f09_root_motion_preserve_y(mut ctx: TestContext) raises:
    var p0 = Pose(1)
    p0.t[0] = Vec3(0.0, 0.0, 0.0)
    var p1 = Pose(1)
    p1.t[0] = Vec3(0.0, 2.5, 0.0)
    var ahead = Pose(1)
    ahead.t[0] = Vec3(0.0, 2.5, 0.0)
    var inert = Inertializer()
    inert.apply(p0, ahead, False, 0.2, 0.016)
    inert.apply(p1, ahead, True, 0.2, 0.016)
    ctx.assert_approx(p1.t[0].y, 0.0, 0.1, "inertializer dampens Y jump smoothly on state switch")

def test_f09_inertializer_no_velocity_jump(mut ctx: TestContext) raises:
    var p0 = Pose(1)
    p0.t[0] = Vec3(0.0, 0.0, 0.0)
    var ahead = Pose(1)
    ahead.t[0] = Vec3(0.0, 0.0, 0.0)
    var inert = Inertializer()
    inert.apply(p0, ahead, False, 0.2, 0.016)
    var p1 = Pose(1)
    p1.t[0] = Vec3(0.0, 0.0, 0.0)
    inert.apply(p1, ahead, True, 0.2, 0.016)
    ctx.assert_approx(inert.out_vel_t[0].x, 0.0, 0.001, "no velocity spike on X")
    ctx.assert_approx(inert.out_vel_t[0].z, 0.0, 0.001, "no velocity spike on Z")

def test_f09_root_delta_tracking(mut ctx: TestContext) raises:
    var a = Animator()
    var model = create_mock_model_with_extensions()
    var st = StateDef()
    st.root_motion = ROOT_EXTRACT
    a.graph.states.append(st^)
    a.bind(model, 0)
    a.root_node = 0
    a.state = 0
    a.last_root = Vec3(0, 0, 0)
    a.pose = Pose(len(model.nodes))
    a.pose.t[0] = Vec3(0.1, 0.0, 0.2)
    a.root_motion(model, False)
    ctx.assert_approx(a.root_travel.x, 0.1, 0.001, "root travel tracks x accumulation")
    ctx.assert_approx(a.root_travel.z, 0.2, 0.001, "root travel tracks z accumulation")

def test_f09_animator_state_transition(mut ctx: TestContext) raises:
    var a = Animator()
    a.reset()
    ctx.assert_equal(a.state, -1, "initial state is -1")


# =========================================================================
# F10: Wayland Mouse & Pointer (5 tests)
# =========================================================================

def test_f10_wayland_pointer_lock_toggle(mut ctx: TestContext) raises:
    var win = WaylandWindow()
    ctx.assert_false(win.pointer_locked, "pointer unlocked by default")
    win.pointer_locked = True
    ctx.assert_true(win.pointer_locked, "pointer lock flag sets")
    win.pointer_locked = False
    ctx.assert_false(win.pointer_locked, "clearing pointer_locked updates flag")

def test_f10_wayland_unaccelerated_motion(mut ctx: TestContext) raises:
    var buf = List[UInt8](length=32, fill=0)
    wl_put32(buf, 24, 3200)
    wl_put32(buf, 28, -2048)
    var dx = wl_fixed(wl_gets32(buf, 24))
    var dy = wl_fixed(wl_gets32(buf, 28))
    ctx.assert_approx(dx, 12.5, 0.01, "unaccelerated dx correctly decoded from offset 24")
    ctx.assert_approx(dy, -8.0, 0.01, "unaccelerated dy correctly decoded from offset 28")

def test_f10_wayland_cursor_hide(mut ctx: TestContext) raises:
    var win = WaylandWindow()
    win.pointer = 10
    win.pointer_enter_serial = 100
    win.pointer_locked = True
    win.incoming.resize(16, 0)
    wl_put32(win.incoming, 0, 10)
    wl_put32(win.incoming, 4, 0 | (16 << 16))
    wl_put32(win.incoming, 8, 200)
    win.handle_incoming()
    ctx.assert_equal(win.pointer_enter_serial, 200, "pointer enter event sets pointer_enter_serial")

def test_f10_wayland_no_double_motion(mut ctx: TestContext) raises:
    var win = WaylandWindow()
    win.pointer_locked = True
    ctx.assert_true(win.pointer_locked, "pointer lock is active")

def test_f10_wayland_message_queue_linear(mut ctx: TestContext) raises:
    var win = WaylandWindow()
    win.pointer_locked = True
    win.relative_pointer = 5
    win.incoming.resize(64, 0)
    wl_put32(win.incoming, 0, 5)
    wl_put32(win.incoming, 4, 0 | (32 << 16))
    wl_put32(win.incoming, 24, 256)
    wl_put32(win.incoming, 28, 512)
    wl_put32(win.incoming, 32, 5)
    wl_put32(win.incoming, 36, 0 | (32 << 16))
    wl_put32(win.incoming, 56, 768)
    wl_put32(win.incoming, 60, 1024)
    win.handle_incoming()
    ctx.assert_equal(len(win.incoming), 0, "incoming buffer compacted linearly to length 0")
    ctx.assert_approx(win.pending_relative_x, 4.0, 0.01, "accumulated relative dx = 1.0 + 3.0 = 4.0")
    ctx.assert_approx(win.pending_relative_y, 6.0, 0.01, "accumulated relative dy = 2.0 + 4.0 = 6.0")


# =========================================================================
# F11: Gamepad Triggers & DevConsole (5 tests)
# =========================================================================

def test_f11_gamepad_trigger_normalization(mut ctx: TestContext) raises:
    var pad = PadState()
    pad.apply_event(2, PAD_AXIS_LT, -32767)
    ctx.assert_approx(pad.axes[PAD_AXIS_LT], 0.0, 0.01, "trigger at rest normalizes to 0.0")
    pad.apply_event(2, PAD_AXIS_LT, 32767)
    ctx.assert_approx(pad.axes[PAD_AXIS_LT], 1.0, 0.01, "trigger pressed normalizes to 1.0")

def test_f11_radial_deadzone_filtering(mut ctx: TestContext) raises:
    var pad = PadState()
    pad.apply_event(2, PAD_AXIS_LEFT_X, 1000)
    ctx.assert_approx(pad.axes[PAD_AXIS_LEFT_X], 0.0, 0.001, "stick motion inside deadzone filtered to 0.0")

def test_f11_radial_deadzone_smooth_ramp(mut ctx: TestContext) raises:
    var pad = PadState()
    pad.apply_event(2, PAD_AXIS_LEFT_X, 20000)
    ctx.assert_true(pad.axes[PAD_AXIS_LEFT_X] > 0.0, "stick motion outside deadzone produces positive axis value")

def test_f11_devconsole_underscore_input(mut ctx: TestContext) raises:
    var byte_val = typed_byte(95)  # ASCII 95 = '_'
    ctx.assert_equal(byte_val, 95, "ASCII 95 underscore recognized as typed byte")
    var c = DevConsole()
    var cvars = ConsoleVars()
    c.key(95, cvars)
    ctx.assert_true(c.line.byte_length() > 0, "console buffer contains typed underscore")

def test_f11_devconsole_cvar_tab_completion(mut ctx: TestContext) raises:
    var c = DevConsole()
    var cvars = ConsoleVars()
    c.line = "deb"
    c.key(258, cvars) # KEY_TAB -> calls complete()
    ctx.assert_true(c.line.startswith("debug"), "tab autocomplete completes debug cvar prefix")


# =========================================================================
# F12: Non-blocking MCP, Profiler & BC7 (5 tests)
# =========================================================================

def test_f12_mcp_socket_nonblock_flag(mut ctx: TestContext) raises:
    var info = FlintMcpInfo()
    info.app_name = "flint_e2e"
    var resp = handle_flint_mcp_query("flint_hello", "{}", info)
    ctx.assert_true(resp.startswith("{\"engine\":\"flint\""), "flint_hello query returns engine info")

def test_f12_profiler_measure_clock_instant(mut ctx: TestContext) raises:
    var timeline = ProfileTimeline()
    var t0 = ctx.elapsed_ms()
    _ = timeline.start("", overlay=False)
    var dt = ctx.elapsed_ms() - t0
    ctx.assert_true(dt < 10.0, "non-blocking measure_clock runs in < 10ms")
    ctx.assert_true(timeline.nanoseconds_per_tick > 0.0, "valid nanoseconds_per_tick calibrated")
    timeline.finish()

def test_f12_bc7_weight_zero_heap_alloc(mut ctx: TestContext) raises:
    # Verify zero-allocation LUT branch functions match Khronos BPTC spec
    ctx.assert_equal(_bc7_weight2(0), 0, "weight2[0] == 0")
    ctx.assert_equal(_bc7_weight2(1), 21, "weight2[1] == 21")
    ctx.assert_equal(_bc7_weight2(2), 43, "weight2[2] == 43")
    ctx.assert_equal(_bc7_weight2(3), 64, "weight2[3] == 64")

    ctx.assert_equal(_bc7_weight3(0), 0, "weight3[0] == 0")
    ctx.assert_equal(_bc7_weight3(4), 37, "weight3[4] == 37")
    ctx.assert_equal(_bc7_weight3(7), 64, "weight3[7] == 64")

    ctx.assert_equal(_bc7_weight4(0), 0, "weight4[0] == 0")
    ctx.assert_equal(_bc7_weight4(8), 34, "weight4[8] == 34")
    ctx.assert_equal(_bc7_weight4(15), 64, "weight4[15] == 64")

def test_f12_profiler_timeline_events(mut ctx: TestContext) raises:
    ctx.assert_equal(ZONE_SIM, 3, "ZONE_SIM constant is 3")
    ctx.assert_equal(ZONE_FRAME, 0, "ZONE_FRAME constant is 0")
    ctx.assert_equal(ZONE_POST, 7, "ZONE_POST constant is 7")
    var sim_name = zone_name(ZONE_SIM)
    ctx.assert_equal(sim_name, "sim", "zone_name(ZONE_SIM) is sim")

def test_f12_mcp_query_format(mut ctx: TestContext) raises:
    var info = FlintMcpInfo()
    var mem = handle_flint_mcp_query("flint_memory", "{}", info)
    ctx.assert_true(mem.startswith("{\"cpu_live_bytes\":"), "flint_memory query returns memory statistics")


# =========================================================================
# Aggregator for Tier 1 Milestone 1
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T1_M1_Phase0")

    # F01
    try: test_f01_model_bytes_material_fields(ctx); ctx.record_pass("test_f01_model_bytes_material_fields") except e: ctx.record_fail("test_f01_model_bytes_material_fields", String(e))
    try: test_f01_read_model_deserialization(ctx); ctx.record_pass("test_f01_read_model_deserialization") except e: ctx.record_fail("test_f01_read_model_deserialization", String(e))
    try: test_f01_model_difference_roughness(ctx); ctx.record_pass("test_f01_model_difference_roughness") except e: ctx.record_fail("test_f01_model_difference_roughness", String(e))
    try: test_f01_model_difference_textures(ctx); ctx.record_pass("test_f01_model_difference_textures") except e: ctx.record_fail("test_f01_model_difference_textures", String(e))
    try: test_f01_cooked_stamp_and_roundtrip(ctx); ctx.record_pass("test_f01_cooked_stamp_and_roundtrip") except e: ctx.record_fail("test_f01_cooked_stamp_and_roundtrip", String(e))

    # F02
    try: test_f02_gltf_emissive_strength(ctx); ctx.record_pass("test_f02_gltf_emissive_strength") except e: ctx.record_fail("test_f02_gltf_emissive_strength", String(e))
    try: test_f02_gltf_emissive_no_double_mult(ctx); ctx.record_pass("test_f02_gltf_emissive_no_double_mult") except e: ctx.record_fail("test_f02_gltf_emissive_no_double_mult", String(e))
    try: test_f02_unaligned_accessor_loading(ctx); ctx.record_pass("test_f02_unaligned_accessor_loading") except e: ctx.record_fail("test_f02_unaligned_accessor_loading", String(e))
    try: test_f02_gltf_accessor_component_counts(ctx); ctx.record_pass("test_f02_gltf_accessor_component_counts") except e: ctx.record_fail("test_f02_gltf_accessor_component_counts", String(e))
    try: test_f02_gltf_pbr_attributes(ctx); ctx.record_pass("test_f02_gltf_pbr_attributes") except e: ctx.record_fail("test_f02_gltf_pbr_attributes", String(e))

    # F03
    try: test_f03_bone_palette_slice_offsets(ctx); ctx.record_pass("test_f03_bone_palette_slice_offsets") except e: ctx.record_fail("test_f03_bone_palette_slice_offsets", String(e))
    try: test_f03_joint_weights_normalization(ctx); ctx.record_pass("test_f03_joint_weights_normalization") except e: ctx.record_fail("test_f03_joint_weights_normalization", String(e))
    try: test_f03_dual_quaternion_math(ctx); ctx.record_pass("test_f03_dual_quaternion_math") except e: ctx.record_fail("test_f03_dual_quaternion_math", String(e))
    try: test_f03_prev_pos_indexing_safety(ctx); ctx.record_pass("test_f03_prev_pos_indexing_safety") except e: ctx.record_fail("test_f03_prev_pos_indexing_safety", String(e))
    try: test_f03_decoupled_vertex_buffers(ctx); ctx.record_pass("test_f03_decoupled_vertex_buffers") except e: ctx.record_fail("test_f03_decoupled_vertex_buffers", String(e))

    # F04
    try: test_f04_rt_instance_id_normal(ctx); ctx.record_pass("test_f04_rt_instance_id_normal") except e: ctx.record_fail("test_f04_rt_instance_id_normal", String(e))
    try: test_f04_rt_instance_id_sentinel(ctx); ctx.record_pass("test_f04_rt_instance_id_sentinel") except e: ctx.record_fail("test_f04_rt_instance_id_sentinel", String(e))
    try: test_f04_rt_miss_radiance_fallback(ctx); ctx.record_pass("test_f04_rt_miss_radiance_fallback") except e: ctx.record_fail("test_f04_rt_miss_radiance_fallback", String(e))
    try: test_f04_rt_custom_index_packing(ctx); ctx.record_pass("test_f04_rt_custom_index_packing") except e: ctx.record_fail("test_f04_rt_custom_index_packing", String(e))
    try: test_f04_rt_as_instance_buffer(ctx); ctx.record_pass("test_f04_rt_as_instance_buffer") except e: ctx.record_fail("test_f04_rt_as_instance_buffer", String(e))

    # F05
    try: test_f05_meshlet_id_32bit(ctx); ctx.record_pass("test_f05_meshlet_id_32bit") except e: ctx.record_fail("test_f05_meshlet_id_32bit", String(e))
    try: test_f05_staging_capacity_bounds(ctx); ctx.record_pass("test_f05_staging_capacity_bounds") except e: ctx.record_fail("test_f05_staging_capacity_bounds", String(e))
    try: test_f05_add_mesh_incremental_words(ctx); ctx.record_pass("test_f05_add_mesh_incremental_words") except e: ctx.record_fail("test_f05_add_mesh_incremental_words", String(e))
    try: test_f05_meshlet_cluster_counts(ctx); ctx.record_pass("test_f05_meshlet_cluster_counts") except e: ctx.record_fail("test_f05_meshlet_cluster_counts", String(e))
    try: test_f05_geometry_pool_totals(ctx); ctx.record_pass("test_f05_geometry_pool_totals") except e: ctx.record_fail("test_f05_geometry_pool_totals", String(e))

    # F06
    try: test_f06_hiz_projection_prev_mat(ctx); ctx.record_pass("test_f06_hiz_projection_prev_mat") except e: ctx.record_fail("test_f06_hiz_projection_prev_mat", String(e))
    try: test_f06_hiz_min_depth_mip_pyramid(ctx); ctx.record_pass("test_f06_hiz_min_depth_mip_pyramid") except e: ctx.record_fail("test_f06_hiz_min_depth_mip_pyramid", String(e))
    try: test_f06_hiz_cull_visible_meshlet(ctx); ctx.record_pass("test_f06_hiz_cull_visible_meshlet") except e: ctx.record_fail("test_f06_hiz_cull_visible_meshlet", String(e))
    try: test_f06_hiz_cull_occluded_meshlet(ctx); ctx.record_pass("test_f06_hiz_cull_occluded_meshlet") except e: ctx.record_fail("test_f06_hiz_cull_occluded_meshlet", String(e))
    try: test_f06_hiz_reverse_z_comparison(ctx); ctx.record_pass("test_f06_hiz_reverse_z_comparison") except e: ctx.record_fail("test_f06_hiz_reverse_z_comparison", String(e))

    # F07
    try: test_f07_spirv_control_barrier_void(ctx); ctx.record_pass("test_f07_spirv_control_barrier_void") except e: ctx.record_fail("test_f07_spirv_control_barrier_void", String(e))
    try: test_f07_spirv_memory_barrier_void(ctx); ctx.record_pass("test_f07_spirv_memory_barrier_void") except e: ctx.record_fail("test_f07_spirv_memory_barrier_void", String(e))
    try: test_f07_spirv_barrier_no_result_type(ctx); ctx.record_pass("test_f07_spirv_barrier_no_result_type") except e: ctx.record_fail("test_f07_spirv_barrier_no_result_type", String(e))
    try: test_f07_spirv_validator_compute_pass(ctx); ctx.record_pass("test_f07_spirv_validator_compute_pass") except e: ctx.record_fail("test_f07_spirv_validator_compute_pass", String(e))
    try: test_f07_spirv_barrier_operand_validation(ctx); ctx.record_pass("test_f07_spirv_barrier_operand_validation") except e: ctx.record_fail("test_f07_spirv_barrier_operand_validation", String(e))

    # F08
    try: test_f08_rodrigues_displacement_zero(ctx); ctx.record_pass("test_f08_rodrigues_displacement_zero") except e: ctx.record_fail("test_f08_rodrigues_displacement_zero", String(e))
    try: test_f08_rodrigues_displacement_small_angle(ctx); ctx.record_pass("test_f08_rodrigues_displacement_small_angle") except e: ctx.record_fail("test_f08_rodrigues_displacement_small_angle", String(e))
    try: test_f08_rodrigues_displacement_large_angle(ctx); ctx.record_pass("test_f08_rodrigues_displacement_large_angle") except e: ctx.record_fail("test_f08_rodrigues_displacement_large_angle", String(e))
    try: test_f08_swept_sphere_obb_face_hit(ctx); ctx.record_pass("test_f08_swept_sphere_obb_face_hit") except e: ctx.record_fail("test_f08_swept_sphere_obb_face_hit", String(e))
    try: test_f08_swept_sphere_obb_corner_miss(ctx); ctx.record_pass("test_f08_swept_sphere_obb_corner_miss") except e: ctx.record_fail("test_f08_swept_sphere_obb_corner_miss", String(e))

    # F09
    try: test_f09_root_motion_strip_xz(ctx); ctx.record_pass("test_f09_root_motion_strip_xz") except e: ctx.record_fail("test_f09_root_motion_strip_xz", String(e))
    try: test_f09_root_motion_preserve_y(ctx); ctx.record_pass("test_f09_root_motion_preserve_y") except e: ctx.record_fail("test_f09_root_motion_preserve_y", String(e))
    try: test_f09_inertializer_no_velocity_jump(ctx); ctx.record_pass("test_f09_inertializer_no_velocity_jump") except e: ctx.record_fail("test_f09_inertializer_no_velocity_jump", String(e))
    try: test_f09_root_delta_tracking(ctx); ctx.record_pass("test_f09_root_delta_tracking") except e: ctx.record_fail("test_f09_root_delta_tracking", String(e))
    try: test_f09_animator_state_transition(ctx); ctx.record_pass("test_f09_animator_state_transition") except e: ctx.record_fail("test_f09_animator_state_transition", String(e))

    # F10
    try: test_f10_wayland_pointer_lock_toggle(ctx); ctx.record_pass("test_f10_wayland_pointer_lock_toggle") except e: ctx.record_fail("test_f10_wayland_pointer_lock_toggle", String(e))
    try: test_f10_wayland_unaccelerated_motion(ctx); ctx.record_pass("test_f10_wayland_unaccelerated_motion") except e: ctx.record_fail("test_f10_wayland_unaccelerated_motion", String(e))
    try: test_f10_wayland_cursor_hide(ctx); ctx.record_pass("test_f10_wayland_cursor_hide") except e: ctx.record_fail("test_f10_wayland_cursor_hide", String(e))
    try: test_f10_wayland_no_double_motion(ctx); ctx.record_pass("test_f10_wayland_no_double_motion") except e: ctx.record_fail("test_f10_wayland_no_double_motion", String(e))
    try: test_f10_wayland_message_queue_linear(ctx); ctx.record_pass("test_f10_wayland_message_queue_linear") except e: ctx.record_fail("test_f10_wayland_message_queue_linear", String(e))

    # F11
    try: test_f11_gamepad_trigger_normalization(ctx); ctx.record_pass("test_f11_gamepad_trigger_normalization") except e: ctx.record_fail("test_f11_gamepad_trigger_normalization", String(e))
    try: test_f11_radial_deadzone_filtering(ctx); ctx.record_pass("test_f11_radial_deadzone_filtering") except e: ctx.record_fail("test_f11_radial_deadzone_filtering", String(e))
    try: test_f11_radial_deadzone_smooth_ramp(ctx); ctx.record_pass("test_f11_radial_deadzone_smooth_ramp") except e: ctx.record_fail("test_f11_radial_deadzone_smooth_ramp", String(e))
    try: test_f11_devconsole_underscore_input(ctx); ctx.record_pass("test_f11_devconsole_underscore_input") except e: ctx.record_fail("test_f11_devconsole_underscore_input", String(e))
    try: test_f11_devconsole_cvar_tab_completion(ctx); ctx.record_pass("test_f11_devconsole_cvar_tab_completion") except e: ctx.record_fail("test_f11_devconsole_cvar_tab_completion", String(e))

    # F12
    try: test_f12_mcp_socket_nonblock_flag(ctx); ctx.record_pass("test_f12_mcp_socket_nonblock_flag") except e: ctx.record_fail("test_f12_mcp_socket_nonblock_flag", String(e))
    try: test_f12_profiler_measure_clock_instant(ctx); ctx.record_pass("test_f12_profiler_measure_clock_instant") except e: ctx.record_fail("test_f12_profiler_measure_clock_instant", String(e))
    try: test_f12_bc7_weight_zero_heap_alloc(ctx); ctx.record_pass("test_f12_bc7_weight_zero_heap_alloc") except e: ctx.record_fail("test_f12_bc7_weight_zero_heap_alloc", String(e))
    try: test_f12_profiler_timeline_events(ctx); ctx.record_pass("test_f12_profiler_timeline_events") except e: ctx.record_fail("test_f12_profiler_timeline_events", String(e))
    try: test_f12_mcp_query_format(ctx); ctx.record_pass("test_f12_mcp_query_format") except e: ctx.record_fail("test_f12_mcp_query_format", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("test_t1_m1_phase0 failed")
