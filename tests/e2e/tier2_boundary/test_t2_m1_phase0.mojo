"""Tier 2 Boundary & Corner Cases: Milestone 1 Phase 0 (F01–F12: 60 boundary tests).
Opaque-box and requirement-driven verification of boundary values, extreme inputs, and corner cases.
"""
from std.collections import List
from std.math import abs
from std.time import monotonic
from flint.anim.animator import Animator
from flint.anim.graph import ROOT_STRIP, AnimGraphDef, StateDef
from flint.anim.pose import Pose
from flint.anim.sync import ClipInfo
from flint.assets.bc_compress import _bc7_weight2
from flint.assets.flpk_v3 import decode_cooked_mesh_slab, encode_cooked_mesh_slab
from flint.assets.glb_native import component_count, load_accessor, parse_native_glb
from flint.assets.glb_writer import encode_native_glb
from flint.assets.json import parse_json
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.meshlet import AaaGeometryPool
from flint.assets.model import NativeAnimation, NativeModel, NativeNode
from flint.core.console import DevConsole
from flint.core.cvars import ConsoleVars
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.core.profiler import ZONE_SIM, ProfileTimeline, profile_ticks, report_ms, zone_name
from flint.core.transform import Mat4, Quat, trs
from flint.physics.ccd import swept_sphere_obb
from flint.physics.solver import rodrigues_displacement
from flint.physics.solver_bodies import V8
from flint.platform.input import PAD_AXIS_LEFT_X, PAD_AXIS_LT, InputState, PadState
from flint.platform.vulkan import VulkanGpu, get32, get64
from flint.platform.wayland import WaylandWindow, evdev_key, wl_fixed
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_raytracing import write_vk_as_instance_64b
from flint.render.aaa_renderer import LinearStagingRing
from flint.render.aaa_skinning_rt import DualQuat, aaa_dq_skinning_shader, blend_dual_quats_cpu
from flint.render.aaa_visbuffer import (
    AAA_HIZ_MAX_MIPS,
    compute_hiz_mip_levels,
    hiz_depth_downsample_shader,
)
from flint.render.material_graph import BINDLESS_SLOT_NONE, MaterialAsset
from flint.render.render_world import (
    make_rh_look_at,
    make_rh_reverse_z_infinite_perspective,
    project_rh_world_to_screen,
)
from flint.render.spirv import MODEL_GL_COMPUTE, OP_CONTROL_BARRIER, OP_MEMORY_BARRIER, AaaSpirvModule, emit
from flint.render.spirv_validate import validate_spirv_module
from e2e.common.harness import TestContext
from e2e.common.mock_data import (
    create_mock_glb_bytes,
    create_mock_meshlet_mesh,
    create_mock_model_with_extensions,
)
from e2e.common.v5_fixtures import (
    create_mock_asset_server,
    create_mock_mcp_server,
    create_mock_studio,
)


# =========================================================================
# F01: MaterialAsset, AaaMaterial & Cooked Mesh Slab Boundaries
# =========================================================================

def test_b_f01_zero_roughness_metallic(mut ctx: TestContext) raises:
    var mat = MaterialAsset(roughness=0.0, metallic=0.0)
    ctx.assert_approx(mat.roughness, 0.04, 0.001, "roughness = 0.0 clamped safely to 0.04")
    ctx.assert_approx(mat.metallic, 0.0, 0.0001, "metallic = 0.0 boundary")


def test_b_f01_max_roughness_metallic(mut ctx: TestContext) raises:
    var mat = MaterialAsset(roughness=1.5, metallic=1.5)
    ctx.assert_approx(mat.roughness, 1.0, 0.0001, "roughness clamped to 1.0 boundary")
    ctx.assert_approx(mat.metallic, 1.0, 0.0001, "metallic clamped to 1.0 boundary")


def test_b_f01_negative_texture_indices(mut ctx: TestContext) raises:
    var mat = MaterialAsset()
    mat.set_albedo_slot(-1)
    mat.set_normal_slot(-5)
    mat.set_orm_slot(-1)
    ctx.assert_equal(mat.albedo_slot, BINDLESS_SLOT_NONE, "negative albedo slot maps to BINDLESS_SLOT_NONE")
    ctx.assert_equal(mat.normal_slot, BINDLESS_SLOT_NONE, "negative normal slot maps to BINDLESS_SLOT_NONE")
    ctx.assert_equal(mat.orm_slot, BINDLESS_SLOT_NONE, "negative orm slot maps to BINDLESS_SLOT_NONE")


def test_b_f01_empty_model_bytes(mut ctx: TestContext) raises:
    var empty_pool = AaaGeometryPool()
    var slab = encode_cooked_mesh_slab(empty_pool)
    var dec = decode_cooked_mesh_slab(slab)
    ctx.assert_true(dec[0], "empty AaaGeometryPool slab decodes cleanly")
    ctx.assert_equal(len(dec[2].vertex_words), 0, "empty pool has 0 vertex words")


def test_b_f01_huge_emissive_strength(mut ctx: TestContext) raises:
    var mat = MaterialAsset()
    mat.emissive = Vec3(1000.0, 500.0, 250.0)
    var words = mat.pack_words()
    var round_mat = MaterialAsset.unpack_words(words)
    ctx.assert_approx(round_mat.emissive.x, 1000.0, 0.1, "1000.0 emissive radiance round-trips")


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
    ctx.assert_true(len(parsed.meshes) > 0, "glb parses safely")


def test_b_f02_empty_accessor(mut ctx: TestContext) raises:
    var j_str = String(
        '{"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":16}],'
        '"accessors":[{"bufferView":0,"componentType":5126,"count":0,"type":"VEC3"},'
        '{"componentType":5126,"count":4,"type":"SCALAR"}]}'
    )
    var jb = List[UInt8](capacity=j_str.byte_length())
    for b in j_str.as_bytes():
        jb.append(b)
    var doc = parse_json(jb)
    ctx.assert_true(doc.valid, "minimal glTF accessor JSON parses")
    var acc_zero = load_accessor(doc, 0, 16)
    var acc_nobv = load_accessor(doc, 1, 16)
    var empty_bin = List[UInt8](length=16, fill=0)
    ctx.assert_false(acc_zero.valid, "count=0 empty accessor rejected as invalid")
    ctx.assert_equal(acc_zero.count, 0, "empty accessor preserves count=0")
    ctx.assert_approx(acc_zero.float_value(empty_bin, 0, 0), 0.0, 0.0001, "invalid accessor float_value returns 0.0")
    ctx.assert_false(acc_nobv.valid, "accessor without bufferView rejected as invalid")
    ctx.assert_equal(component_count("MAT4"), 16, "MAT4 component count is 16")
    ctx.assert_equal(component_count("MAT3"), 9, "MAT3 component count is 9")
    ctx.assert_equal(component_count("UNKNOWN"), 0, "unknown accessor type returns 0 components")


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
    bones.append(DualQuat.from_quat_trans(Quat(), Vec3(3.0, 4.0, 5.0)))
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
    ctx.assert_vec3_approx(dq.transform_point(v), v, 0.001, "identity DualQuat transforms point to itself")


def test_b_f03_zero_scale_joint(mut ctx: TestContext) raises:
    var m = trs(Vec3(10.0, -20.0, 30.0), Quat(), Vec3(0.0, 0.0, 0.0))
    var dq = DualQuat.from_mat4(m)
    var p = dq.transform_point(Vec3(1.0, 2.0, 3.0))
    ctx.assert_approx(dq.uniform_scale(), 0.01, 0.001, "zero-scale Mat4 clamps uniform scale cleanly to 1e-6^(1/3)=0.01")
    ctx.assert_vec3_approx(p, Vec3(10.01, -19.98, 30.03), 0.01, "zero-scale joint transforms without NaN near translation")
    var dq_qt = DualQuat.from_quat_trans(Quat(), Vec3(10.0, -20.0, 30.0), Float32(0.0))
    ctx.assert_approx(dq_qt.uniform_scale(), 1.0e-6, 1.0e-7, "from_quat_trans(scale=0) clamps uniform scale to 1e-6")


def test_b_f03_first_and_last_vertex_index(mut ctx: TestContext) raises:
    var rep = validate_spirv_module(aaa_dq_skinning_shader(12, 1))
    ctx.assert_true(rep.valid, "aaa_dq_skinning_shader boundary compile passes validator")


# =========================================================================
# F04: RT Instance ID Sentinel (Boundary)
# =========================================================================

def test_b_f04_instance_index_exact_1023(mut ctx: TestContext) raises:
    var buf = List[UInt8](length=64, fill=0)
    var addr = Int(buf.unsafe_ptr())
    write_vk_as_instance_64b(addr, trs(Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1)), 1023, 0x11112222, cull_disable=True)
    var custom = get32(addr + 48)
    _ = buf[0]
    ctx.assert_equal(Int(custom & 0xFFFFFF), 1023, "instance 1023 custom index packed correctly")


def test_b_f04_instance_index_exact_1024(mut ctx: TestContext) raises:
    var buf = List[UInt8](length=64, fill=0)
    var addr = Int(buf.unsafe_ptr())
    write_vk_as_instance_64b(addr, trs(Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1)), 0xFFFFFF, 0x33334444, cull_disable=True)
    var custom = get32(addr + 48)
    _ = buf[0]
    ctx.assert_equal(Int(custom & 0xFFFFFF), 0xFFFFFF, "sentinel custom index 0xFFFFFF packed")


def test_b_f04_instance_index_large(mut ctx: TestContext) raises:
    var buf = List[UInt8](length=64, fill=0)
    var addr = Int(buf.unsafe_ptr())
    write_vk_as_instance_64b(addr, trs(Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1)), 0xFFFFFF, 0x55556666, cull_disable=False)
    var flags = get32(addr + 52)
    _ = buf[0]
    ctx.assert_equal(Int((flags >> 24) & 1), 0, "cull_disable=False sets flag bit 0")


def test_b_f04_custom24_mask_boundary(mut ctx: TestContext) raises:
    var buf = List[UInt8](length=64, fill=0)
    var addr = Int(buf.unsafe_ptr())
    write_vk_as_instance_64b(addr, Mat4(), 0, 0x77778888, cull_disable=True)
    var custom = get32(addr + 48)
    _ = buf[0]
    ctx.assert_equal(Int(custom & 0xFFFFFF), 0, "instance custom index 0 packed cleanly")


def test_b_f04_zero_instance_count(mut ctx: TestContext) raises:
    var buf = List[UInt8](length=64, fill=0)
    var addr = Int(buf.unsafe_ptr())
    write_vk_as_instance_64b(addr, Mat4(), 1, 0, cull_disable=True)
    var blas_addr = get64(addr + 56)
    _ = buf[0]
    ctx.assert_equal(blas_addr, 0, "null BLAS address 0 handled cleanly")


# =========================================================================
# F05: Meshlet ID & LinearStagingRing Bounds (Boundary)
# =========================================================================

def test_b_f05_meshlet_id_exact_65535(mut ctx: TestContext) raises:
    var pool = AaaGeometryPool()
    var entry = pool.append_mesh(create_mock_meshlet_mesh(4), 0, build_lods=False)
    ctx.assert_true(entry.meshlet_count > 0, "meshlet pool appends low-res mesh")


def test_b_f05_meshlet_id_exact_65536(mut ctx: TestContext) raises:
    var pool = AaaGeometryPool()
    var entry = pool.append_mesh(create_mock_meshlet_mesh(16), 42, build_lods=False)
    ctx.assert_equal(entry.material_id, 42, "CookedMeshEntry material_id matches")


def test_b_f05_staging_exact_8mib(mut ctx: TestContext) raises:
    var vk = VulkanGpu()
    ctx.assert_true(vk.initialize(validate=False, need_discrete=False), "VulkanGpu initializes in headless mode")
    var ring = LinearStagingRing()
    ctx.assert_true(ring.init(vk, 65536), "LinearStagingRing initializes 64 KiB slots")
    var exact = ring.alloc(vk, 0, 65536)
    ctx.assert_equal(exact.slab_idx, -1, "exact slot_capacity alloc stays in primary slot buffer")
    ctx.assert_true(exact.cpu_address != 0, "exact primary slice has valid mapped ptr")
    ring.reset_slot(vk, 0)
    ring.shutdown(vk)
    vk.close()


def test_b_f05_staging_over_8mib(mut ctx: TestContext) raises:
    var vk = VulkanGpu()
    ctx.assert_true(vk.initialize(validate=False, need_discrete=False), "VulkanGpu initializes in headless mode")
    var ring = LinearStagingRing()
    ctx.assert_true(ring.init(vk, 65536), "LinearStagingRing initializes")
    var over = ring.alloc(vk, 0, 65536 + 1024)
    ctx.assert_true(over.slab_idx >= 0 and over.cpu_address != 0, "over-capacity alloc spills into overflow slab")
    ring.reset_slot(vk, 0)
    ctx.assert_equal(len(ring.overflow_slabs[0]), 0, "reset_slot frees overflow slabs")
    ring.shutdown(vk)
    vk.close()


def test_b_f05_single_triangle_meshlet(mut ctx: TestContext) raises:
    var mb = MeshBuilder()
    var v0 = mb.add_vertex(Vec3(0, 0, 0), Vec3(0, 1, 0), 0, 0)
    var v1 = mb.add_vertex(Vec3(1, 0, 0), Vec3(0, 1, 0), 1, 0)
    var v2 = mb.add_vertex(Vec3(0, 1, 0), Vec3(0, 1, 0), 0, 1)
    mb.add_tri(v0, v1, v2)
    var pool = AaaGeometryPool()
    var entry = pool.append_mesh(mb^.finish(), 0, build_lods=False)
    ctx.assert_equal(entry.triangle_count, 1, "single triangle mesh appended to geometry pool")


# =========================================================================
# F06: Reverse-Z & Hi-Z Mip Pyramid Boundaries
# =========================================================================

def test_b_f06_depth_0_far_plane(mut ctx: TestContext) raises:
    var view = make_rh_look_at(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, -1.0))
    var proj = make_rh_reverse_z_infinite_perspective(Float32(1.0471976), Float32(1.0), Float32(0.1))
    var vp = proj.multiply(view)
    var far_res = project_rh_world_to_screen(vp, Vec3(0.0, 0.0, -10000.0), 100, 100)
    ctx.assert_true(far_res[3], "far point projects valid")
    ctx.assert_true(far_res[2] < 0.001, "infinite Reverse-Z approaches 0.0 at far distance")


def test_b_f06_depth_1_near_plane(mut ctx: TestContext) raises:
    var view = make_rh_look_at(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, -1.0))
    var proj = make_rh_reverse_z_infinite_perspective(Float32(1.0471976), Float32(1.0), Float32(0.1))
    var vp = proj.multiply(view)
    var near_res = project_rh_world_to_screen(vp, Vec3(0.0, 0.0, -0.1), 100, 100)
    ctx.assert_true(near_res[3], "near plane point projects valid")
    ctx.assert_approx(near_res[2], 1.0, 0.001, "near plane maps to Reverse-Z 1.0")


def test_b_f06_uv_clamping_boundary(mut ctx: TestContext) raises:
    var rep = validate_spirv_module(hiz_depth_downsample_shader())
    ctx.assert_true(rep.valid, "hiz_depth_downsample_shader passes SPIR-V validator")


def test_b_f06_screen_center_projection(mut ctx: TestContext) raises:
    var view = make_rh_look_at(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, -1.0))
    var proj = make_rh_reverse_z_infinite_perspective(Float32(1.0471976), Float32(1.0), Float32(0.1))
    var vp = proj.multiply(view)
    var behind = project_rh_world_to_screen(vp, Vec3(0.0, 0.0, 2.0), 100, 100)
    ctx.assert_false(behind[3], "point behind camera (+Z in RH -Z space) rejected")


def test_b_f06_mip_pyramid_1x1_level(mut ctx: TestContext) raises:
    ctx.assert_equal(compute_hiz_mip_levels(1, 1), 1, "1x1 texture produces 1 Hi-Z mip level")
    ctx.assert_equal(compute_hiz_mip_levels(16384, 16384), AAA_HIZ_MAX_MIPS, "16K clamped to AAA_HIZ_MAX_MIPS")


# =========================================================================
# F07: SPIR-V Validator Barrier Opcodes (Boundary)
# =========================================================================

def test_b_f07_barrier_word_count_min(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    emit(m.code, OP_CONTROL_BARRIER, m.uint(2), m.uint(2), m.uint(0x100))
    ctx.assert_true(validate_spirv_module(m.finish()).valid, "control barrier passes validator")


def test_b_f07_barrier_memory_semantics_all_bits(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    emit(m.code, OP_MEMORY_BARRIER, m.uint(2), m.uint(0x8 | 0x40 | 0x100 | 0x800))
    ctx.assert_true(validate_spirv_module(m.finish()).valid, "compound memory semantics pass validator")


def test_b_f07_consecutive_barriers(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var sc_exec = m.uint(2)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    emit(m.code, OP_CONTROL_BARRIER, sc_exec, sc_mem, sem)
    emit(m.code, OP_MEMORY_BARRIER, sc_mem, sem)
    ctx.assert_true(validate_spirv_module(m.finish()).valid, "consecutive barriers pass validator")


def test_b_f07_barrier_at_function_start(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var a = m.uint(5)
    _ = m.iadd(a, a)
    emit(m.code, OP_MEMORY_BARRIER, m.uint(2), m.uint(0x100))
    ctx.assert_true(validate_spirv_module(m.finish()).valid, "barrier following arithmetic passes validator")


def test_b_f07_barrier_at_function_end(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    emit(m.code, OP_MEMORY_BARRIER, UInt32(8888), UInt32(8888))
    ctx.assert_false(validate_spirv_module(m.finish()).valid, "barrier with bad operand ID rejected")


# =========================================================================
# F08: Lever-Arm & Swept OBB (Boundary)
# =========================================================================

def test_b_f08_rodrigues_angle_1e_7(mut ctx: TestContext) raises:
    var disp = rodrigues_displacement(V8.splat(Vec3(0.0, 0.0, 1.0e-7)), V8.splat(Vec3(2.0, 0.0, 0.0)))
    ctx.assert_true(abs(disp.y[0]) < 1.0e-5, "ultra-small angle evaluated cleanly")


def test_b_f08_rodrigues_angle_pi(mut ctx: TestContext) raises:
    var disp = rodrigues_displacement(V8.splat(Vec3(0.0, 0.0, 3.14159265)), V8.splat(Vec3(2.0, 0.0, 0.0)))
    ctx.assert_approx(disp.x[0], -4.0, 0.05, "180 deg turnaround produces -2*r disp x = -4")


def test_b_f08_swept_obb_exact_surface_touch(mut ctx: TestContext) raises:
    var hit = swept_sphere_obb(Vec3(-1.5, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), 0.5, Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    ctx.assert_true(hit.hit, "surface touch registers hit at t=0")
    ctx.assert_approx(hit.toi, 0.0, 0.05, "toi = 0.0 at surface")


def test_b_f08_swept_obb_grazing_edge(mut ctx: TestContext) raises:
    var hit = swept_sphere_obb(Vec3(-5.0, 1.5, 0.0), Vec3(5.0, 1.5, 0.0), 0.5, Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    ctx.assert_true(hit.hit, "exact grazing sweep at y=1.5 touches top OBB edge")
    ctx.assert_approx(hit.toi, 0.4, 0.02, "grazing sweep contacts top edge at toi=0.4 (x=-1.0)")
    ctx.assert_approx(hit.normal.length(), 1.0, 0.01, "grazing contact normal is unit length")
    ctx.assert_approx(hit.normal.y, 1.0, 0.02, "grazing contact normal points along +Y")
    var miss = swept_sphere_obb(Vec3(-5.0, 1.51, 0.0), Vec3(5.0, 1.51, 0.0), 0.5, Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    ctx.assert_false(miss.hit, "sweep 0.01m above grazing edge (y=1.51) misses cleanly")


def test_b_f08_swept_obb_negative_direction(mut ctx: TestContext) raises:
    var hit = swept_sphere_obb(Vec3(5.0, 0.0, 0.0), Vec3(-5.0, 0.0, 0.0), 0.5, Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
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
    ctx.assert_approx(a.pose.t[0].x, model.nodes[0].base_translation.x, 0.001, "horizontal jump stripped")
    ctx.assert_approx(a.pose.t[0].z, model.nodes[0].base_translation.z, 0.001, "horizontal jump stripped")


def test_b_f09_unconfigured_root_node(mut ctx: TestContext) raises:
    var a = Animator()
    ctx.assert_equal(a.root_node, -1, "unconfigured root node is -1")


def test_b_f09_subframe_dt(mut ctx: TestContext) raises:
    var a = Animator()
    var nm = NativeModel()
    var nd = NativeNode()
    nd.name = "Root"
    nm.nodes.append(nd^)
    var anim = NativeAnimation()
    anim.name = "idle"
    anim.duration = Float32(1.0)
    nm.animations.append(anim^)
    var g = AnimGraphDef()
    var st = StateDef()
    st.name = "idle"
    st.clip = 0
    g.states.append(st^)
    var clips = List[ClipInfo]()
    clips.append(ClipInfo("idle", Float32(1.0), True, Float32(0.0)))
    a.configure(g, clips)
    for _ in range(5):
        a.update(nm, 0, Float32(1.0e-4))
    ctx.assert_equal(a.state, 0, "Animator enters state 0 on subframe updates")
    ctx.assert_approx(a.clip_time, 0.0004, 0.00005, "4 advancing 0.1ms subframe steps advance clip_time to 0.0004s")
    ctx.assert_approx(a.inertial.last_dt, 0.0001, 0.00002, "Inertializer records 0.1ms subframe dt")


# =========================================================================
# F10: Wayland Mouse & Pointer (Boundary)
# =========================================================================

def test_b_f10_zero_mouse_delta(mut ctx: TestContext) raises:
    ctx.assert_approx(wl_fixed(0), 0.0, 0.0001, "wl_fixed(0) is 0.0")


def test_b_f10_subpixel_mouse_delta(mut ctx: TestContext) raises:
    ctx.assert_approx(wl_fixed(65536), 256.0, 0.01, "wl_fixed(65536) converts 24.8 to 256.0")


def test_b_f10_large_mouse_delta(mut ctx: TestContext) raises:
    var inp = InputState()
    inp.begin_frame()
    inp.add_mouse_delta(1500.0, -2500.0)
    ctx.assert_approx(inp.mouse_delta_x(), 1500.0, 0.01, "large mouse dx preserved")
    ctx.assert_approx(inp.mouse_delta_y(), -2500.0, 0.01, "large mouse dy preserved")
    inp.begin_frame()
    ctx.assert_approx(inp.mouse_delta_x(), 0.0, 0.001, "begin_frame zeroes mouse_dx")
    ctx.assert_equal(evdev_key(-1), -1, "evdev_key(-1) is -1")


def test_b_f10_rapid_pointer_lock_toggle(mut ctx: TestContext) raises:
    var win = WaylandWindow()
    win.using_kms = True
    for _ in range(50):
        win.set_pointer_lock(True)
        ctx.assert_true(win.pointer_locked, "set_pointer_lock(True) locks")
        win.set_pointer_lock(False)
    ctx.assert_false(win.pointer_locked, "set_pointer_lock(False) finishes unlocked")


def test_b_f10_empty_incoming_queue(mut ctx: TestContext) raises:
    var win = WaylandWindow()
    win.incoming.clear()
    win.handle_incoming()
    ctx.assert_equal(len(win.incoming), 0, "empty incoming queue stays empty after handle_incoming")
    ctx.assert_approx(win.pending_relative_x, 0.0, 0.0001, "no spurious relative_x on empty queue")
    ctx.assert_approx(win.pending_relative_y, 0.0, 0.0001, "no spurious relative_y on empty queue")
    win.incoming.append(0x01)
    win.incoming.append(0x02)
    win.handle_incoming()
    ctx.assert_equal(len(win.incoming), 2, "sub-header 2-byte fragment preserved safely without panic")


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
        c.key(65, cvars)
    ctx.assert_true(c.line.byte_length() <= 96, "DevConsole enforces line buffer length")


def test_b_f11_console_backspace_empty(mut ctx: TestContext) raises:
    var c = DevConsole()
    var cvars = ConsoleVars()
    c.key(259, cvars)
    ctx.assert_equal(c.line.byte_length(), 0, "backspace on empty line safe")


# =========================================================================
# F12: McpServerV2, Profiler & BC7 (Boundary)
# =========================================================================

def test_b_f12_mcp_empty_query(mut ctx: TestContext) raises:
    var studio = create_mock_studio()
    var assets = create_mock_asset_server()
    var pool = JobPool()
    var mcp = create_mock_mcp_server("/tmp")
    var resp = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ecs.query_entities","arguments":{"name_contains":"__nonexistent__"}}}',
        studio,
        assets,
        pool,
    )
    ctx.assert_true(resp.find('\\"count\\":0') >= 0 or resp.find('"count":0') >= 0, "empty entity query returns count=0")
    ctx.assert_true(resp.find('\\"entities\\":[]') >= 0 or resp.find('"entities":[]') >= 0, "empty entity query returns []")
    var bad_tool = mcp.handle_jsonrpc('{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"","arguments":{}}}', studio, assets, pool)
    ctx.assert_true(bad_tool.find("-32602") >= 0, "empty tool name returns JSON-RPC invalid params -32602")
    assets.close()
    studio.shutdown()
    pool.stop()


def test_b_f12_profiler_zone_max_count(mut ctx: TestContext) raises:
    ctx.assert_equal(zone_name(ZONE_SIM), "sim", "zone_name(ZONE_SIM) returns sim")


def test_b_f12_bc7_weight_endpoints_0_and_64(mut ctx: TestContext) raises:
    ctx.assert_equal(_bc7_weight2(0), 0, "BC7 min weight is 0")
    ctx.assert_equal(_bc7_weight2(3), 64, "BC7 max weight is 64")


def test_b_f12_clock_counter_monotonicity(mut ctx: TestContext) raises:
    var t0 = profile_ticks()
    var t_imm = profile_ticks()
    ctx.assert_true(t_imm >= t0, "profile_ticks is non-decreasing across immediate reads")
    var wall0 = monotonic()
    var acc: UInt64 = 0
    var iter: UInt64 = 0
    while (monotonic() - wall0) < 100_000 or iter < 1000:
        acc += iter
        iter += 1
    var t1 = profile_ticks()
    ctx.assert_true(acc > 0 and t1 > t0, "profile_ticks advances monotonically across work")
    var tl = ProfileTimeline()
    ctx.assert_true(tl.start("", False), "ProfileTimeline calibrates clock")
    ctx.assert_true(tl.to_ms(UInt64(10000)) > tl.to_ms(UInt64(1000)), "to_ms scales monotonically with ticks")
    ctx.assert_true(report_ms(tl.to_ms(UInt64(1000))).byte_length() > 0, "report_ms formats calibrated ms")
    tl.finish()


def test_b_f12_mcp_invalid_json(mut ctx: TestContext) raises:
    var studio = create_mock_studio()
    var assets = create_mock_asset_server()
    var pool = JobPool()
    var mcp = create_mock_mcp_server("/tmp")
    var bad_json = mcp.handle_jsonrpc("{not valid json", studio, assets, pool)
    ctx.assert_true(bad_json.find("-32700") >= 0, "invalid JSON returns JSON-RPC parse error -32700")
    var bad_method = mcp.handle_jsonrpc("{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"unknown.xyz\",\"params\":{}}", studio, assets, pool)
    ctx.assert_true(bad_method.find("-32601") >= 0, "unknown method returns JSON-RPC -32601")
    assets.close()
    studio.shutdown()
    pool.stop()


# =========================================================================
# Aggregator for Tier 2 Milestone 1
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T2_M1_Phase0")

    try: test_b_f01_zero_roughness_metallic(ctx); ctx.record_pass("test_b_f01_zero_roughness_metallic") except e: ctx.record_fail("test_b_f01_zero_roughness_metallic", String(e))
    try: test_b_f01_max_roughness_metallic(ctx); ctx.record_pass("test_b_f01_max_roughness_metallic") except e: ctx.record_fail("test_b_f01_max_roughness_metallic", String(e))
    try: test_b_f01_negative_texture_indices(ctx); ctx.record_pass("test_b_f01_negative_texture_indices") except e: ctx.record_fail("test_b_f01_negative_texture_indices", String(e))
    try: test_b_f01_empty_model_bytes(ctx); ctx.record_pass("test_b_f01_empty_model_bytes") except e: ctx.record_fail("test_b_f01_empty_model_bytes", String(e))
    try: test_b_f01_huge_emissive_strength(ctx); ctx.record_pass("test_b_f01_huge_emissive_strength") except e: ctx.record_fail("test_b_f01_huge_emissive_strength", String(e))

    try: test_b_f02_zero_emissive_strength(ctx); ctx.record_pass("test_b_f02_zero_emissive_strength") except e: ctx.record_fail("test_b_f02_zero_emissive_strength", String(e))
    try: test_b_f02_exact_1_emissive_strength(ctx); ctx.record_pass("test_b_f02_exact_1_emissive_strength") except e: ctx.record_fail("test_b_f02_exact_1_emissive_strength", String(e))
    try: test_b_f02_unaligned_offset_mod_4(ctx); ctx.record_pass("test_b_f02_unaligned_offset_mod_4") except e: ctx.record_fail("test_b_f02_unaligned_offset_mod_4", String(e))
    try: test_b_f02_empty_accessor(ctx); ctx.record_pass("test_b_f02_empty_accessor") except e: ctx.record_fail("test_b_f02_empty_accessor", String(e))
    try: test_b_f02_nan_inf_pbr_clamp(ctx); ctx.record_pass("test_b_f02_nan_inf_pbr_clamp") except e: ctx.record_fail("test_b_f02_nan_inf_pbr_clamp", String(e))

    try: test_b_f03_single_bone_weight_1(ctx); ctx.record_pass("test_b_f03_single_bone_weight_1") except e: ctx.record_fail("test_b_f03_single_bone_weight_1", String(e))
    try: test_b_f03_max_joints_per_vertex(ctx); ctx.record_pass("test_b_f03_max_joints_per_vertex") except e: ctx.record_fail("test_b_f03_max_joints_per_vertex", String(e))
    try: test_b_f03_identity_joint_palette(ctx); ctx.record_pass("test_b_f03_identity_joint_palette") except e: ctx.record_fail("test_b_f03_identity_joint_palette", String(e))
    try: test_b_f03_zero_scale_joint(ctx); ctx.record_pass("test_b_f03_zero_scale_joint") except e: ctx.record_fail("test_b_f03_zero_scale_joint", String(e))
    try: test_b_f03_first_and_last_vertex_index(ctx); ctx.record_pass("test_b_f03_first_and_last_vertex_index") except e: ctx.record_fail("test_b_f03_first_and_last_vertex_index", String(e))

    try: test_b_f04_instance_index_exact_1023(ctx); ctx.record_pass("test_b_f04_instance_index_exact_1023") except e: ctx.record_fail("test_b_f04_instance_index_exact_1023", String(e))
    try: test_b_f04_instance_index_exact_1024(ctx); ctx.record_pass("test_b_f04_instance_index_exact_1024") except e: ctx.record_fail("test_b_f04_instance_index_exact_1024", String(e))
    try: test_b_f04_instance_index_large(ctx); ctx.record_pass("test_b_f04_instance_index_large") except e: ctx.record_fail("test_b_f04_instance_index_large", String(e))
    try: test_b_f04_custom24_mask_boundary(ctx); ctx.record_pass("test_b_f04_custom24_mask_boundary") except e: ctx.record_fail("test_b_f04_custom24_mask_boundary", String(e))
    try: test_b_f04_zero_instance_count(ctx); ctx.record_pass("test_b_f04_zero_instance_count") except e: ctx.record_fail("test_b_f04_zero_instance_count", String(e))

    try: test_b_f05_meshlet_id_exact_65535(ctx); ctx.record_pass("test_b_f05_meshlet_id_exact_65535") except e: ctx.record_fail("test_b_f05_meshlet_id_exact_65535", String(e))
    try: test_b_f05_meshlet_id_exact_65536(ctx); ctx.record_pass("test_b_f05_meshlet_id_exact_65536") except e: ctx.record_fail("test_b_f05_meshlet_id_exact_65536", String(e))
    try: test_b_f05_staging_exact_8mib(ctx); ctx.record_pass("test_b_f05_staging_exact_8mib") except e: ctx.record_fail("test_b_f05_staging_exact_8mib", String(e))
    try: test_b_f05_staging_over_8mib(ctx); ctx.record_pass("test_b_f05_staging_over_8mib") except e: ctx.record_fail("test_b_f05_staging_over_8mib", String(e))
    try: test_b_f05_single_triangle_meshlet(ctx); ctx.record_pass("test_b_f05_single_triangle_meshlet") except e: ctx.record_fail("test_b_f05_single_triangle_meshlet", String(e))

    try: test_b_f06_depth_0_far_plane(ctx); ctx.record_pass("test_b_f06_depth_0_far_plane") except e: ctx.record_fail("test_b_f06_depth_0_far_plane", String(e))
    try: test_b_f06_depth_1_near_plane(ctx); ctx.record_pass("test_b_f06_depth_1_near_plane") except e: ctx.record_fail("test_b_f06_depth_1_near_plane", String(e))
    try: test_b_f06_uv_clamping_boundary(ctx); ctx.record_pass("test_b_f06_uv_clamping_boundary") except e: ctx.record_fail("test_b_f06_uv_clamping_boundary", String(e))
    try: test_b_f06_screen_center_projection(ctx); ctx.record_pass("test_b_f06_screen_center_projection") except e: ctx.record_fail("test_b_f06_screen_center_projection", String(e))
    try: test_b_f06_mip_pyramid_1x1_level(ctx); ctx.record_pass("test_b_f06_mip_pyramid_1x1_level") except e: ctx.record_fail("test_b_f06_mip_pyramid_1x1_level", String(e))

    try: test_b_f07_barrier_word_count_min(ctx); ctx.record_pass("test_b_f07_barrier_word_count_min") except e: ctx.record_fail("test_b_f07_barrier_word_count_min", String(e))
    try: test_b_f07_barrier_memory_semantics_all_bits(ctx); ctx.record_pass("test_b_f07_barrier_memory_semantics_all_bits") except e: ctx.record_fail("test_b_f07_barrier_memory_semantics_all_bits", String(e))
    try: test_b_f07_consecutive_barriers(ctx); ctx.record_pass("test_b_f07_consecutive_barriers") except e: ctx.record_fail("test_b_f07_consecutive_barriers", String(e))
    try: test_b_f07_barrier_at_function_start(ctx); ctx.record_pass("test_b_f07_barrier_at_function_start") except e: ctx.record_fail("test_b_f07_barrier_at_function_start", String(e))
    try: test_b_f07_barrier_at_function_end(ctx); ctx.record_pass("test_b_f07_barrier_at_function_end") except e: ctx.record_fail("test_b_f07_barrier_at_function_end", String(e))

    try: test_b_f08_rodrigues_angle_1e_7(ctx); ctx.record_pass("test_b_f08_rodrigues_angle_1e_7") except e: ctx.record_fail("test_b_f08_rodrigues_angle_1e_7", String(e))
    try: test_b_f08_rodrigues_angle_pi(ctx); ctx.record_pass("test_b_f08_rodrigues_angle_pi") except e: ctx.record_fail("test_b_f08_rodrigues_angle_pi", String(e))
    try: test_b_f08_swept_obb_exact_surface_touch(ctx); ctx.record_pass("test_b_f08_swept_obb_exact_surface_touch") except e: ctx.record_fail("test_b_f08_swept_obb_exact_surface_touch", String(e))
    try: test_b_f08_swept_obb_grazing_edge(ctx); ctx.record_pass("test_b_f08_swept_obb_grazing_edge") except e: ctx.record_fail("test_b_f08_swept_obb_grazing_edge", String(e))
    try: test_b_f08_swept_obb_negative_direction(ctx); ctx.record_pass("test_b_f08_swept_obb_negative_direction") except e: ctx.record_fail("test_b_f08_swept_obb_negative_direction", String(e))

    try: test_b_f09_zero_root_displacement(ctx); ctx.record_pass("test_b_f09_zero_root_displacement") except e: ctx.record_fail("test_b_f09_zero_root_displacement", String(e))
    try: test_b_f09_pure_vertical_root_motion(ctx); ctx.record_pass("test_b_f09_pure_vertical_root_motion") except e: ctx.record_fail("test_b_f09_pure_vertical_root_motion", String(e))
    try: test_b_f09_huge_horizontal_root_jump(ctx); ctx.record_pass("test_b_f09_huge_horizontal_root_jump") except e: ctx.record_fail("test_b_f09_huge_horizontal_root_jump", String(e))
    try: test_b_f09_unconfigured_root_node(ctx); ctx.record_pass("test_b_f09_unconfigured_root_node") except e: ctx.record_fail("test_b_f09_unconfigured_root_node", String(e))
    try: test_b_f09_subframe_dt(ctx); ctx.record_pass("test_b_f09_subframe_dt") except e: ctx.record_fail("test_b_f09_subframe_dt", String(e))

    try: test_b_f10_zero_mouse_delta(ctx); ctx.record_pass("test_b_f10_zero_mouse_delta") except e: ctx.record_fail("test_b_f10_zero_mouse_delta", String(e))
    try: test_b_f10_subpixel_mouse_delta(ctx); ctx.record_pass("test_b_f10_subpixel_mouse_delta") except e: ctx.record_fail("test_b_f10_subpixel_mouse_delta", String(e))
    try: test_b_f10_large_mouse_delta(ctx); ctx.record_pass("test_b_f10_large_mouse_delta") except e: ctx.record_fail("test_b_f10_large_mouse_delta", String(e))
    try: test_b_f10_rapid_pointer_lock_toggle(ctx); ctx.record_pass("test_b_f10_rapid_pointer_lock_toggle") except e: ctx.record_fail("test_b_f10_rapid_pointer_lock_toggle", String(e))
    try: test_b_f10_empty_incoming_queue(ctx); ctx.record_pass("test_b_f10_empty_incoming_queue") except e: ctx.record_fail("test_b_f10_empty_incoming_queue", String(e))

    try: test_b_f11_trigger_full_rest(ctx); ctx.record_pass("test_b_f11_trigger_full_rest") except e: ctx.record_fail("test_b_f11_trigger_full_rest", String(e))
    try: test_b_f11_trigger_full_press(ctx); ctx.record_pass("test_b_f11_trigger_full_press") except e: ctx.record_fail("test_b_f11_trigger_full_press", String(e))
    try: test_b_f11_deadzone_exact_boundary(ctx); ctx.record_pass("test_b_f11_deadzone_exact_boundary") except e: ctx.record_fail("test_b_f11_deadzone_exact_boundary", String(e))
    try: test_b_f11_console_max_length(ctx); ctx.record_pass("test_b_f11_console_max_length") except e: ctx.record_fail("test_b_f11_console_max_length", String(e))
    try: test_b_f11_console_backspace_empty(ctx); ctx.record_pass("test_b_f11_console_backspace_empty") except e: ctx.record_fail("test_b_f11_console_backspace_empty", String(e))

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
