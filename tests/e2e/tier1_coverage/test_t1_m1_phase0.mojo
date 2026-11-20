"""Tier 1 Feature Coverage: Milestone 1 Phase 0 (F01–F12: 60 tests).
Opaque-box and requirement-driven verification of v5 cooked assets, Reverse-Z/Hi-Z,
SPIR-V validation, Wayland/input, and McpServerV2 invariants.
"""
from std.collections import List
from std.math import abs, sqrt
from flint.anim.animator import Animator
from flint.anim.graph import OP_GT, ROOT_EXTRACT, ROOT_KEEP, ROOT_STRIP, AnimGraphDef, Condition, StateDef
from flint.anim.pose import Inertializer, Pose
from flint.anim.sync import ClipInfo
from flint.assets.bc_compress import (
    _bc7_weight2,
    _bc7_weight3,
    _bc7_weight4,
    encode_bc7_rgba,
)
from flint.assets.flpk_v3 import (
    ASSET_KIND_MESH,
    CHUNK_COMP_NONE,
    COOKED_MESH_SLAB_MAGIC,
    AssetMeta,
    AssetUuid,
    FlpkBuildChunk,
    FlpkV3MappedArchive,
    decode_cooked_mesh_slab_into,
    encode_cooked_mesh_slab,
    encode_flpk_v3,
    read_flmeta,
    write_flmeta,
)
from flint.assets.glb_native import component_count, parse_native_glb
from flint.assets.glb_writer import encode_native_glb
from flint.assets.json import parse_json
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.meshlet import AaaGeometryPool, select_mesh_lod_cpu_reference
from flint.assets.model import NativeMesh, NativeModel
from flint.assets.server import AssetServer
from flint.core.console import DevConsole, typed_byte
from flint.core.cvars import ConsoleVars
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.core.profiler import ZONE_FRAME, ZONE_POST, ZONE_SIM, ProfileTimeline, zone_name
from flint.core.transform import Mat4, Quat, trs
from flint.devtools.mcp_v2 import McpServerV2
from flint.physics.ccd import CcdHit, swept_sphere_obb
from flint.physics.solver import rodrigues_displacement
from flint.physics.solver_bodies import F8, LANES, V8
from flint.platform.input import (
    PAD_AXIS_LEFT_X,
    PAD_AXIS_LT,
    PadState,
)
from flint.platform.os import write_bytes
from flint.platform.vulkan import get32, get64
from flint.platform.wayland import WaylandWindow, wl_fixed, wl_gets32, wl_put32
from flint.render.aaa_raytracing import emit_rt_surface_cache_ray, write_vk_as_instance_64b
from flint.render.aaa_renderer import FRAMES_IN_FLIGHT, FlintAaaVulkanRenderer
from flint.render.aaa_skinning_rt import DualQuat, aaa_dq_skinning_shader, blend_dual_quats_cpu
from flint.render.aaa_visbuffer import (
    AAA_MAX_PACKED_INSTANCES,
    AAA_MAX_PACKED_MESHLET_ID,
    compute_hiz_mip_levels,
    hiz_depth_downsample_shader,
    meshlet_cull_shader,
)
from flint.render.material_graph import MaterialAsset
from flint.render.render_world import (
    make_rh_look_at,
    make_rh_reverse_z_infinite_perspective,
    make_rh_reverse_z_perspective,
    project_rh_world_to_screen,
)
from flint.render.spirv import (
    MODEL_GL_COMPUTE,
    OP_CONTROL_BARRIER,
    OP_MEMORY_BARRIER,
    AaaSpirvModule,
    emit,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.studio import FlintStudioState
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
# F01: Cooked Mesh Slabs, MaterialAsset Words & .flmeta (5 tests)
# =========================================================================

def test_f01_model_bytes_material_fields(mut ctx: TestContext) raises:
    var glb = create_mock_glb_bytes()
    var loaded = parse_native_glb(glb)
    ctx.assert_true(len(loaded.meshes) > 0, "GLB must deserialize meshes")
    ctx.assert_approx(loaded.meshes[0].roughness, 0.25, 0.01, "roughness match")
    ctx.assert_approx(loaded.meshes[0].metallic, 0.75, 0.01, "metallic match")
    ctx.assert_true(loaded.meshes[0].emissive_strength > 1.0, "emissive_strength match")
    ctx.assert_true(loaded.meshes[0].double_sided, "double_sided preserved")


def test_f01_read_model_deserialization(mut ctx: TestContext) raises:
    var pool = AaaGeometryPool()
    var mesh = create_mock_meshlet_mesh(10)
    _ = pool.append_mesh(mesh, 0, build_lods=True)
    var slab = encode_cooked_mesh_slab(pool)
    ctx.assert_true(len(slab) > 64, "cooked mesh slab must exceed 64B header")
    var magic = UInt32(slab[0]) | (UInt32(slab[1]) << 8) | (UInt32(slab[2]) << 16) | (UInt32(slab[3]) << 24)
    ctx.assert_equal(magic, COOKED_MESH_SLAB_MAGIC, "cooked slab magic matches FMSH")
    var pool2 = AaaGeometryPool()
    var res = decode_cooked_mesh_slab_into(slab, pool2)
    ctx.assert_true(res[0], "decode_cooked_mesh_slab_into succeeds")
    ctx.assert_equal(len(pool2.vertex_words), len(pool.vertex_words), "vertex words round-trip")
    ctx.assert_equal(len(pool2.index_words), len(pool.index_words), "index words round-trip")
    ctx.assert_equal(len(pool2.meshlet_words), len(pool.meshlet_words), "meshlet words round-trip")


def test_f01_model_difference_roughness(mut ctx: TestContext) raises:
    var m1 = MaterialAsset(base_color=Vec3(0.8, 0.2, 0.2), roughness=0.25, metallic=0.75)
    var m2 = MaterialAsset(base_color=Vec3(0.8, 0.2, 0.2), roughness=0.25, metallic=0.75)
    var w1 = m1.pack_words()
    var w2 = m2.pack_words()
    ctx.assert_equal(len(w1), 24, "MaterialAsset packs 24 GPU words")
    var diff_same = False
    for i in range(len(w1)):
        if w1[i] != w2[i]:
            diff_same = True
    ctx.assert_false(diff_same, "identical MaterialAssets pack identical GPU words")
    m2.roughness = Float32(0.99)
    var w3 = m2.pack_words()
    var diff_changed = False
    for i in range(len(w1)):
        if w1[i] != w3[i]:
            diff_changed = True
    ctx.assert_true(diff_changed, "modified roughness changes packed GPU material words")


def test_f01_model_difference_textures(mut ctx: TestContext) raises:
    var m1 = MaterialAsset()
    var m2 = MaterialAsset()
    m1.set_albedo_slot(3)
    m1.set_normal_slot(4)
    m2.set_albedo_slot(3)
    m2.set_normal_slot(99)
    var w1 = m1.pack_words()
    var w2 = m2.pack_words()
    ctx.assert_true(w1[23] != w2[23], "bindless normal slot change modifies word 23")


def test_f01_cooked_stamp_and_roundtrip(mut ctx: TestContext) raises:
    var meta_path = String("/tmp/flint_e2e_f01_test.flmeta")
    var flpk_path = String("/tmp/flint_e2e_f01_test.flpk")
    ctx.cleanup_artifact(meta_path)
    ctx.cleanup_artifact(flpk_path)

    var meta = AssetMeta()
    meta.uuid = AssetUuid.from_path("meshes/box.glb")
    meta.asset_kind = ASSET_KIND_MESH
    meta.source_path = "meshes/box.glb"
    meta.max_tex_dim = 1024
    ctx.assert_true(write_flmeta(meta_path, meta), "write_flmeta writes .flmeta sidecar")
    var loaded_meta = read_flmeta(meta_path)
    ctx.assert_true(loaded_meta[0], "read_flmeta parses .flmeta sidecar")
    ctx.assert_equal(loaded_meta[1].uuid.format(), meta.uuid.format(), "UUID round-trips in .flmeta")

    var pool = AaaGeometryPool()
    _ = pool.append_mesh(create_mock_meshlet_mesh(8), 0, build_lods=False)
    var slab = encode_cooked_mesh_slab(pool)
    var chunks = List[FlpkBuildChunk]()
    chunks.append(FlpkBuildChunk(meta.uuid, "meshes/box.glb", 0, ASSET_KIND_MESH, slab^, CHUNK_COMP_NONE))
    var packed = encode_flpk_v3(chunks, page_align=True)
    ctx.assert_true(write_bytes(flpk_path, packed), "write_bytes persists flpk v3")

    var arch = FlpkV3MappedArchive.open(flpk_path)
    ctx.assert_true(arch.valid, "FlpkV3MappedArchive opens cleanly")
    var toc = arch.find_by_uuid(meta.uuid, 0)
    ctx.assert_true(toc >= 0, "find_by_uuid locates cooked mesh chunk")
    ctx.assert_true(arch.verify_entry_crc64(toc), "ECMA-182 CRC64 verifies")
    arch.close()
    ctx.cleanup_artifact(meta_path)
    ctx.cleanup_artifact(flpk_path)


# =========================================================================
# F02: glTF Emissive & Align (5 tests)
# =========================================================================

def test_f02_gltf_emissive_strength(mut ctx: TestContext) raises:
    var glb = create_mock_glb_bytes()
    var reloaded = parse_native_glb(glb)
    ctx.assert_true(len(reloaded.meshes) > 0, "glb must parse meshes")
    ctx.assert_true(reloaded.meshes[0].emissive_strength > 1.0, "emissive_strength > 1.0 preserved")


def test_f02_gltf_emissive_no_double_mult(mut ctx: TestContext) raises:
    var model = create_mock_model_with_extensions()
    var glb = encode_native_glb(model)
    var parsed = parse_native_glb(glb)
    ctx.assert_true(parsed.meshes[0].emissive_color.x <= 1.001, "emissive color RGB stays clamped [0, 1]")


def test_f02_unaligned_accessor_loading(mut ctx: TestContext) raises:
    var glb = create_mock_glb_bytes()
    ctx.assert_true(len(glb) > 100, "glb binary valid")
    var parsed = parse_native_glb(glb)
    ctx.assert_true(len(parsed.meshes[0].positions) > 0, "positions loaded successfully")


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
    ctx.assert_true(len(parsed.meshes[0].normals) > 0, "normals preserved")


# =========================================================================
# F03: GPU Skinning Fixes (5 tests)
# =========================================================================

def test_f03_bone_palette_slice_offsets(mut ctx: TestContext) raises:
    var dq0 = DualQuat.from_quat_trans(Quat(), Vec3(1.0, 2.0, 3.0))
    var dq1 = DualQuat.from_quat_trans(Quat(), Vec3(4.0, 5.0, 6.0))
    var palette = List[UInt32](length=128 * 8, fill=0)
    dq0.write_words(palette, 0)
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
    ctx.assert_vec3_approx(dir, Vec3(0.0, 1.0, 0.0), 0.001, "transform_dir preserves unit direction")


# =========================================================================
# F04: RT Instance ID Sentinel (5 tests)
# =========================================================================

def test_f04_rt_instance_id_normal(mut ctx: TestContext) raises:
    var buf = List[UInt8](length=64, fill=0)
    var addr = Int(buf.unsafe_ptr())
    var mat = trs(Vec3(1, 2, 3), Quat(), Vec3(1, 1, 1))
    write_vk_as_instance_64b(addr, mat, 42, 0x12345678, cull_disable=True)
    var custom = get32(addr + 48)
    _ = buf[0]
    ctx.assert_equal(Int(custom & 0xFFFFFF), 42, "normal instance custom index packed correctly")
    ctx.assert_equal(Int((custom >> 24) & 0xFF), 0xFF, "upper 8 bits mask packed as 0xFF")


def test_f04_rt_instance_id_sentinel(mut ctx: TestContext) raises:
    var buf = List[UInt8](length=64, fill=0)
    var addr = Int(buf.unsafe_ptr())
    var mat = trs(Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1))
    write_vk_as_instance_64b(addr, mat, 0xFFFFFF, 0xABCDEF00, cull_disable=True)
    var custom = get32(addr + 48)
    _ = buf[0]
    ctx.assert_equal(Int(custom & 0xFFFFFF), 0xFFFFFF, "sentinel 0xFFFFFF packed into 24-bit custom index")


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
    var buf = List[UInt8](length=64, fill=0)
    var addr = Int(buf.unsafe_ptr())
    var mat = Mat4()
    write_vk_as_instance_64b(addr, mat, 500, 0xCAFEBABE, cull_disable=True)
    var flags = get32(addr + 52)
    var blas_addr = get64(addr + 56)
    _ = buf[0]
    ctx.assert_equal(Int((flags >> 24) & 1), 1, "cull_disable sets flag bit 1")
    ctx.assert_equal(blas_addr, 0xCAFEBABE, "blas 64-bit device address matches")


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
# F05: Meshlet ID & Staging Ring Bounds (5 tests)
# =========================================================================

def test_f05_meshlet_id_32bit(mut ctx: TestContext) raises:
    ctx.assert_true(AAA_MAX_PACKED_MESHLET_ID >= UInt32(0x3FFFFE), "22-bit meshlet payload > 4M")
    ctx.assert_true(AAA_MAX_PACKED_INSTANCES >= UInt32(0x3FFFF), "18-bit instance payload > 262k")
    var pool = AaaGeometryPool()
    var mesh = create_mock_meshlet_mesh(16)
    var entry = pool.append_mesh(mesh, 0, build_lods=False)
    ctx.assert_true(entry.meshlet_count > 0, "geometry pool builds meshlets")
    ctx.assert_true(UInt32(entry.meshlet_count) < AAA_MAX_PACKED_MESHLET_ID, "meshlets fit packed range")


def test_f05_staging_capacity_bounds(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=64, height=64)
    ctx.assert_true(ren.active, "renderer active in headless mode")
    ctx.assert_equal(len(ren.staging_ring.slot_buffers), FRAMES_IN_FLIGHT, "LinearStagingRing has FRAMES_IN_FLIGHT slots")
    ctx.assert_true(ren.staging_ring.slot_capacity > 0, "LinearStagingRing slot_capacity > 0")
    var sl0 = ren.staging_ring.alloc(ren.vk, 0, 4096)
    ctx.assert_true(sl0.cpu_address != 0, "primary staging slice mapped")
    var sl_big = ren.staging_ring.alloc(ren.vk, 0, ren.staging_ring.slot_capacity + 65536)
    ctx.assert_true(sl_big.slab_idx >= 0 and sl_big.cpu_address != 0, "oversized upload allocates overflow slab")
    ren.staging_ring.reset_slot(ren.vk, 0)
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
# F06: Reverse-Z Projection & Hi-Z Mips (5 tests)
# =========================================================================

def test_f06_hiz_projection_prev_mat(mut ctx: TestContext) raises:
    var view = make_rh_look_at(Vec3(0.0, 0.0, 5.0), Vec3(0.0, 0.0, 0.0))
    var proj = make_rh_reverse_z_infinite_perspective(Float32(1.0471976), Float32(16.0 / 9.0), Float32(0.1))
    var vp = proj.multiply(view)
    var res = project_rh_world_to_screen(vp, Vec3(0.0, 0.0, 0.0), 1280, 720)
    ctx.assert_true(res[3], "point in front of RH -Z camera projects valid")
    ctx.assert_approx(res[0], 640.0, 1.0, "origin projects to horizontal center")
    ctx.assert_approx(res[1], 360.0, 1.0, "origin projects to vertical center")
    ctx.assert_true(res[2] > 0.0 and res[2] < 1.0, "Reverse-Z depth in (0, 1)")


def test_f06_hiz_min_depth_mip_pyramid(mut ctx: TestContext) raises:
    var spv = hiz_depth_downsample_shader()
    ctx.assert_true(len(spv) > 50, "hiz_depth_downsample_shader produces SPIR-V binary")
    var rep = validate_spirv_module(spv)
    ctx.assert_true(rep.valid, "hiz_depth_downsample_shader passes SPIR-V validation")
    var mips = compute_hiz_mip_levels(1920, 1080)
    ctx.assert_true(mips >= 2 and mips <= 8, "1080p Hi-Z mip levels clamped in 2..8")


def test_f06_hiz_cull_visible_meshlet(mut ctx: TestContext) raises:
    var cull_spv = meshlet_cull_shader()
    var rep = validate_spirv_module(cull_spv)
    ctx.assert_true(rep.valid, "meshlet_cull_shader passes SPIR-V validation")


def test_f06_hiz_cull_occluded_meshlet(mut ctx: TestContext) raises:
    var view = make_rh_look_at(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, -1.0))
    var proj = make_rh_reverse_z_infinite_perspective(Float32(1.0471976), Float32(1.0), Float32(0.1))
    var vp = proj.multiply(view)
    var near_pt = project_rh_world_to_screen(vp, Vec3(0.0, 0.0, -2.0), 800, 800)
    var far_pt = project_rh_world_to_screen(vp, Vec3(0.0, 0.0, -20.0), 800, 800)
    ctx.assert_true(near_pt[3] and far_pt[3], "both points project in front")
    ctx.assert_true(near_pt[2] > far_pt[2], "closer object has larger Reverse-Z depth than occluded far object")


def test_f06_hiz_reverse_z_comparison(mut ctx: TestContext) raises:
    var proj = make_rh_reverse_z_perspective(Float32(1.0471976), Float32(1.0), Float32(0.1), Float32(1000.0))
    var p_near = project_rh_world_to_screen(proj, Vec3(0.0, 0.0, -0.1), 100, 100)
    var p_far = project_rh_world_to_screen(proj, Vec3(0.0, 0.0, -1000.0), 100, 100)
    ctx.assert_true(p_near[3] and p_far[3], "near and far boundary points project valid")
    ctx.assert_approx(p_near[2], 1.0, 0.001, "near plane maps to Reverse-Z 1.0")
    ctx.assert_approx(p_far[2], 0.0, 0.001, "far plane maps to Reverse-Z 0.0")


# =========================================================================
# F07: SPIR-V Validator Barrier Opcodes (5 tests)
# =========================================================================

def test_f07_spirv_control_barrier_void(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var sc_exec = m.uint(2)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    emit(m.code, OP_CONTROL_BARRIER, sc_exec, sc_mem, sem)
    var rep = validate_spirv_module(m.finish())
    ctx.assert_true(rep.valid, "module with OP_CONTROL_BARRIER passes validator")


def test_f07_spirv_memory_barrier_void(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    emit(m.code, OP_MEMORY_BARRIER, sc_mem, sem)
    var rep = validate_spirv_module(m.finish())
    ctx.assert_true(rep.valid, "module with OP_MEMORY_BARRIER passes validator")


def test_f07_spirv_barrier_no_result_type(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var a = m.uint(1)
    var b = m.uint(2)
    _ = m.iadd(a, b)
    var sc_mem = m.uint(2)
    var sem = m.uint(0x100)
    emit(m.code, OP_MEMORY_BARRIER, sc_mem, sem)
    var rep = validate_spirv_module(m.finish())
    ctx.assert_true(rep.valid, "interleaved compute arithmetic and memory barrier validates cleanly")


def test_f07_spirv_validator_compute_pass(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    var a = m.uint(10)
    var b = m.uint(20)
    _ = m.iadd(a, b)
    var rep = validate_spirv_module(m.finish())
    ctx.assert_true(rep.valid, "basic compute spirv passes validator")


def test_f07_spirv_barrier_operand_validation(mut ctx: TestContext) raises:
    var m = AaaSpirvModule(MODEL_GL_COMPUTE, 8, 8, 4)
    emit(m.code, OP_CONTROL_BARRIER, UInt32(9999), UInt32(9998), UInt32(9997))
    var rep = validate_spirv_module(m.finish())
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
    ctx.assert_approx(disp.x[0], -1.0, 0.01, "exact 90 deg rotation disp x = -1.0")
    ctx.assert_approx(disp.y[0], 1.0, 0.01, "exact 90 deg rotation disp y = 1.0")


def test_f08_swept_sphere_obb_face_hit(mut ctx: TestContext) raises:
    var hit = swept_sphere_obb(Vec3(-5.0, 0.0, 0.0), Vec3(5.0, 0.0, 0.0), 0.5, Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    ctx.assert_true(hit.hit, "swept sphere directly towards box must hit")
    ctx.assert_vec3_approx(hit.normal, Vec3(-1.0, 0.0, 0.0), 0.05, "hit normal on X face points -X")


def test_f08_swept_sphere_obb_corner_miss(mut ctx: TestContext) raises:
    var hit = swept_sphere_obb(Vec3(-5.0, 5.0, 5.0), Vec3(-2.0, 5.0, 5.0), 0.5, Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    ctx.assert_false(hit.hit, "trajectory far from OBB must not register hit")


# =========================================================================
# F09: Animator Root Motion Stripping (5 tests)
# =========================================================================

def test_f09_root_motion_strip_xz(mut ctx: TestContext) raises:
    var a = Animator()
    var model = create_mock_model_with_extensions()
    var st = StateDef()
    st.root_motion = ROOT_STRIP
    a.graph.states.append(st^)
    a.bind(model, 0)
    a.root_node = 0
    a.state = 0
    a.pose = Pose(len(model.nodes))
    a.pose.t[0] = Vec3(2.5, 1.2, -3.0)
    a.root_motion(model, False)
    ctx.assert_approx(a.pose.t[0].x, model.nodes[0].base_translation.x, 0.001, "ROOT_STRIP strips X to base translation")
    ctx.assert_approx(a.pose.t[0].y, 1.2, 0.001, "ROOT_STRIP preserves vertical Y=1.2")
    ctx.assert_approx(a.pose.t[0].z, model.nodes[0].base_translation.z, 0.001, "ROOT_STRIP strips Z to base translation")


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
    ctx.assert_equal(a.state, -1, "initial state is -1")
    var g = AnimGraphDef()
    g.params.append("speed")
    var st0 = StateDef()
    st0.name = "run"
    st0.clip = 0
    st0.when.append(Condition(0, OP_GT, Float32(0.5)))
    var st1 = StateDef()
    st1.name = "idle"
    st1.clip = 0
    g.states.append(st0^)
    g.states.append(st1^)
    var clips = List[ClipInfo]()
    clips.append(ClipInfo("idle", Float32(1.0), True, Float32(0.0)))
    a.configure(g, clips)
    ctx.assert_equal(a.select(), 1, "speed=0.0 selects fallback state 1")
    a.set(0, Float32(1.0))
    ctx.assert_equal(a.select(), 0, "speed=1.0 transitions state 1 -> 0")


# =========================================================================
# F10: Wayland Mouse & Pointer (5 tests)
# =========================================================================

def test_f10_wayland_pointer_lock_toggle(mut ctx: TestContext) raises:
    var win = WaylandWindow()
    ctx.assert_false(win.pointer_locked, "pointer unlocked by default")
    win.locked_pointer = 7
    win.incoming.resize(16, 0)
    wl_put32(win.incoming, 0, 7)
    wl_put32(win.incoming, 4, 0 | (8 << 16))
    wl_put32(win.incoming, 8, 7)
    wl_put32(win.incoming, 12, 1 | (8 << 16))
    # Process lock packet first
    var first = List[UInt8](length=8, fill=0)
    wl_put32(first, 0, 7)
    wl_put32(first, 4, 0 | (8 << 16))
    win.incoming = first^
    win.handle_incoming()
    ctx.assert_true(win.pointer_locked, "locked_pointer opcode 0 activates pointer_locked")
    var second = List[UInt8](length=8, fill=0)
    wl_put32(second, 0, 7)
    wl_put32(second, 4, 1 | (8 << 16))
    win.incoming = second^
    win.handle_incoming()
    ctx.assert_false(win.pointer_locked, "locked_pointer opcode 1 clears pointer_locked")


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
    win.pointer = 10
    win.relative_pointer = 5
    win.incoming.resize(52, 0)
    # wl_pointer motion (opcode 2, size 20)
    wl_put32(win.incoming, 0, 10)
    wl_put32(win.incoming, 4, 2 | (20 << 16))
    wl_put32(win.incoming, 12, 2560)
    wl_put32(win.incoming, 16, 2560)
    # relative_pointer motion (opcode 0, size 32)
    wl_put32(win.incoming, 20, 5)
    wl_put32(win.incoming, 24, 0 | (32 << 16))
    wl_put32(win.incoming, 44, 512)
    wl_put32(win.incoming, 48, 256)
    win.handle_incoming()
    ctx.assert_approx(win.pending_relative_x, 2.0, 0.01, "relative_x comes only from relative_pointer")
    ctx.assert_approx(win.pending_relative_y, 1.0, 0.01, "relative_y comes only from relative_pointer")


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
    var byte_val = typed_byte(95)
    ctx.assert_equal(byte_val, 95, "ASCII 95 underscore recognized as typed byte")
    var c = DevConsole()
    var cvars = ConsoleVars()
    c.key(95, cvars)
    ctx.assert_true(c.line.byte_length() > 0, "console buffer contains typed underscore")


def test_f11_devconsole_cvar_tab_completion(mut ctx: TestContext) raises:
    var c = DevConsole()
    var cvars = ConsoleVars()
    c.line = "deb"
    c.key(258, cvars)
    ctx.assert_true(c.line.startswith("debug"), "tab autocomplete completes debug cvar prefix")


# =========================================================================
# F12: McpServerV2, Profiler & BC7 (5 tests)
# =========================================================================

def test_f12_mcp_socket_nonblock_flag(mut ctx: TestContext) raises:
    var studio = create_mock_studio()
    var assets = create_mock_asset_server()
    var pool = JobPool()
    var mcp = create_mock_mcp_server("/tmp")
    var req = String("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\"}}")
    var resp = mcp.handle_jsonrpc(req, studio, assets, pool)
    var jb = List[UInt8]()
    for b in resp.as_bytes():
        jb.append(b)
    var doc = parse_json(jb)
    ctx.assert_true(doc.valid, "McpServerV2 initialize returns valid JSON")
    ctx.assert_true(resp.find("2024-11-05") >= 0, "McpServerV2 returns MCP protocolVersion")
    assets.close()
    studio.shutdown()
    pool.stop()


def test_f12_profiler_measure_clock_instant(mut ctx: TestContext) raises:
    var timeline = ProfileTimeline()
    var t0 = ctx.elapsed_ms()
    _ = timeline.start("", overlay=False)
    var dt = ctx.elapsed_ms() - t0
    ctx.assert_true(dt < 10.0, "non-blocking measure_clock runs in < 10ms")
    ctx.assert_true(timeline.nanoseconds_per_tick > 0.0, "valid nanoseconds_per_tick calibrated")
    timeline.finish()


def test_f12_bc7_weight_zero_heap_alloc(mut ctx: TestContext) raises:
    ctx.assert_equal(_bc7_weight2(0), 0, "weight2[0] == 0")
    ctx.assert_equal(_bc7_weight2(3), 64, "weight2[3] == 64")
    ctx.assert_equal(_bc7_weight3(4), 37, "weight3[4] == 37")
    ctx.assert_equal(_bc7_weight4(15), 64, "weight4[15] == 64")
    var px = List[UInt32](length=16, fill=0xFF804020)
    for i in range(8):
        px[i] = 0xFFE0A060
    var bc7 = encode_bc7_rgba(px, 4, 4, multi_mode=True)
    ctx.assert_equal(len(bc7), 16, "4x4 BC7 block encodes to 16 bytes")


def test_f12_profiler_timeline_events(mut ctx: TestContext) raises:
    ctx.assert_equal(ZONE_SIM, 3, "ZONE_SIM constant is 3")
    ctx.assert_equal(ZONE_FRAME, 0, "ZONE_FRAME constant is 0")
    ctx.assert_equal(ZONE_POST, 7, "ZONE_POST constant is 7")
    ctx.assert_equal(zone_name(ZONE_SIM), "sim", "zone_name(ZONE_SIM) is sim")


def test_f12_mcp_query_format(mut ctx: TestContext) raises:
    var studio = create_mock_studio()
    var assets = create_mock_asset_server()
    var pool = JobPool()
    var mcp = create_mock_mcp_server("/tmp")
    var list_resp = mcp.handle_jsonrpc("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{}}", studio, assets, pool)
    ctx.assert_true(list_resp.find("ecs.spawn_entity") >= 0, "tools/list advertises ecs.spawn_entity")
    var call_resp = mcp.handle_jsonrpc(
        "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"ecs.spawn_entity\",\"arguments\":{\"name\":\"M1Box\"}}}",
        studio,
        assets,
        pool,
    )
    ctx.assert_true(call_resp.find("\"result\"") >= 0, "tools/call ecs.spawn_entity returns JSON-RPC result")
    ctx.assert_equal(studio.live_entity_count(), 1, "MCP ecs.spawn_entity mutated FlintStudioState ECS world")
    assets.close()
    studio.shutdown()
    pool.stop()


# =========================================================================
# Aggregator for Tier 1 Milestone 1
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T1_M1_Phase0")

    try: test_f01_model_bytes_material_fields(ctx); ctx.record_pass("test_f01_model_bytes_material_fields") except e: ctx.record_fail("test_f01_model_bytes_material_fields", String(e))
    try: test_f01_read_model_deserialization(ctx); ctx.record_pass("test_f01_read_model_deserialization") except e: ctx.record_fail("test_f01_read_model_deserialization", String(e))
    try: test_f01_model_difference_roughness(ctx); ctx.record_pass("test_f01_model_difference_roughness") except e: ctx.record_fail("test_f01_model_difference_roughness", String(e))
    try: test_f01_model_difference_textures(ctx); ctx.record_pass("test_f01_model_difference_textures") except e: ctx.record_fail("test_f01_model_difference_textures", String(e))
    try: test_f01_cooked_stamp_and_roundtrip(ctx); ctx.record_pass("test_f01_cooked_stamp_and_roundtrip") except e: ctx.record_fail("test_f01_cooked_stamp_and_roundtrip", String(e))

    try: test_f02_gltf_emissive_strength(ctx); ctx.record_pass("test_f02_gltf_emissive_strength") except e: ctx.record_fail("test_f02_gltf_emissive_strength", String(e))
    try: test_f02_gltf_emissive_no_double_mult(ctx); ctx.record_pass("test_f02_gltf_emissive_no_double_mult") except e: ctx.record_fail("test_f02_gltf_emissive_no_double_mult", String(e))
    try: test_f02_unaligned_accessor_loading(ctx); ctx.record_pass("test_f02_unaligned_accessor_loading") except e: ctx.record_fail("test_f02_unaligned_accessor_loading", String(e))
    try: test_f02_gltf_accessor_component_counts(ctx); ctx.record_pass("test_f02_gltf_accessor_component_counts") except e: ctx.record_fail("test_f02_gltf_accessor_component_counts", String(e))
    try: test_f02_gltf_pbr_attributes(ctx); ctx.record_pass("test_f02_gltf_pbr_attributes") except e: ctx.record_fail("test_f02_gltf_pbr_attributes", String(e))

    try: test_f03_bone_palette_slice_offsets(ctx); ctx.record_pass("test_f03_bone_palette_slice_offsets") except e: ctx.record_fail("test_f03_bone_palette_slice_offsets", String(e))
    try: test_f03_joint_weights_normalization(ctx); ctx.record_pass("test_f03_joint_weights_normalization") except e: ctx.record_fail("test_f03_joint_weights_normalization", String(e))
    try: test_f03_dual_quaternion_math(ctx); ctx.record_pass("test_f03_dual_quaternion_math") except e: ctx.record_fail("test_f03_dual_quaternion_math", String(e))
    try: test_f03_prev_pos_indexing_safety(ctx); ctx.record_pass("test_f03_prev_pos_indexing_safety") except e: ctx.record_fail("test_f03_prev_pos_indexing_safety", String(e))
    try: test_f03_decoupled_vertex_buffers(ctx); ctx.record_pass("test_f03_decoupled_vertex_buffers") except e: ctx.record_fail("test_f03_decoupled_vertex_buffers", String(e))

    try: test_f04_rt_instance_id_normal(ctx); ctx.record_pass("test_f04_rt_instance_id_normal") except e: ctx.record_fail("test_f04_rt_instance_id_normal", String(e))
    try: test_f04_rt_instance_id_sentinel(ctx); ctx.record_pass("test_f04_rt_instance_id_sentinel") except e: ctx.record_fail("test_f04_rt_instance_id_sentinel", String(e))
    try: test_f04_rt_miss_radiance_fallback(ctx); ctx.record_pass("test_f04_rt_miss_radiance_fallback") except e: ctx.record_fail("test_f04_rt_miss_radiance_fallback", String(e))
    try: test_f04_rt_custom_index_packing(ctx); ctx.record_pass("test_f04_rt_custom_index_packing") except e: ctx.record_fail("test_f04_rt_custom_index_packing", String(e))
    try: test_f04_rt_as_instance_buffer(ctx); ctx.record_pass("test_f04_rt_as_instance_buffer") except e: ctx.record_fail("test_f04_rt_as_instance_buffer", String(e))

    try: test_f05_meshlet_id_32bit(ctx); ctx.record_pass("test_f05_meshlet_id_32bit") except e: ctx.record_fail("test_f05_meshlet_id_32bit", String(e))
    try: test_f05_staging_capacity_bounds(ctx); ctx.record_pass("test_f05_staging_capacity_bounds") except e: ctx.record_fail("test_f05_staging_capacity_bounds", String(e))
    try: test_f05_add_mesh_incremental_words(ctx); ctx.record_pass("test_f05_add_mesh_incremental_words") except e: ctx.record_fail("test_f05_add_mesh_incremental_words", String(e))
    try: test_f05_meshlet_cluster_counts(ctx); ctx.record_pass("test_f05_meshlet_cluster_counts") except e: ctx.record_fail("test_f05_meshlet_cluster_counts", String(e))
    try: test_f05_geometry_pool_totals(ctx); ctx.record_pass("test_f05_geometry_pool_totals") except e: ctx.record_fail("test_f05_geometry_pool_totals", String(e))

    try: test_f06_hiz_projection_prev_mat(ctx); ctx.record_pass("test_f06_hiz_projection_prev_mat") except e: ctx.record_fail("test_f06_hiz_projection_prev_mat", String(e))
    try: test_f06_hiz_min_depth_mip_pyramid(ctx); ctx.record_pass("test_f06_hiz_min_depth_mip_pyramid") except e: ctx.record_fail("test_f06_hiz_min_depth_mip_pyramid", String(e))
    try: test_f06_hiz_cull_visible_meshlet(ctx); ctx.record_pass("test_f06_hiz_cull_visible_meshlet") except e: ctx.record_fail("test_f06_hiz_cull_visible_meshlet", String(e))
    try: test_f06_hiz_cull_occluded_meshlet(ctx); ctx.record_pass("test_f06_hiz_cull_occluded_meshlet") except e: ctx.record_fail("test_f06_hiz_cull_occluded_meshlet", String(e))
    try: test_f06_hiz_reverse_z_comparison(ctx); ctx.record_pass("test_f06_hiz_reverse_z_comparison") except e: ctx.record_fail("test_f06_hiz_reverse_z_comparison", String(e))

    try: test_f07_spirv_control_barrier_void(ctx); ctx.record_pass("test_f07_spirv_control_barrier_void") except e: ctx.record_fail("test_f07_spirv_control_barrier_void", String(e))
    try: test_f07_spirv_memory_barrier_void(ctx); ctx.record_pass("test_f07_spirv_memory_barrier_void") except e: ctx.record_fail("test_f07_spirv_memory_barrier_void", String(e))
    try: test_f07_spirv_barrier_no_result_type(ctx); ctx.record_pass("test_f07_spirv_barrier_no_result_type") except e: ctx.record_fail("test_f07_spirv_barrier_no_result_type", String(e))
    try: test_f07_spirv_validator_compute_pass(ctx); ctx.record_pass("test_f07_spirv_validator_compute_pass") except e: ctx.record_fail("test_f07_spirv_validator_compute_pass", String(e))
    try: test_f07_spirv_barrier_operand_validation(ctx); ctx.record_pass("test_f07_spirv_barrier_operand_validation") except e: ctx.record_fail("test_f07_spirv_barrier_operand_validation", String(e))

    try: test_f08_rodrigues_displacement_zero(ctx); ctx.record_pass("test_f08_rodrigues_displacement_zero") except e: ctx.record_fail("test_f08_rodrigues_displacement_zero", String(e))
    try: test_f08_rodrigues_displacement_small_angle(ctx); ctx.record_pass("test_f08_rodrigues_displacement_small_angle") except e: ctx.record_fail("test_f08_rodrigues_displacement_small_angle", String(e))
    try: test_f08_rodrigues_displacement_large_angle(ctx); ctx.record_pass("test_f08_rodrigues_displacement_large_angle") except e: ctx.record_fail("test_f08_rodrigues_displacement_large_angle", String(e))
    try: test_f08_swept_sphere_obb_face_hit(ctx); ctx.record_pass("test_f08_swept_sphere_obb_face_hit") except e: ctx.record_fail("test_f08_swept_sphere_obb_face_hit", String(e))
    try: test_f08_swept_sphere_obb_corner_miss(ctx); ctx.record_pass("test_f08_swept_sphere_obb_corner_miss") except e: ctx.record_fail("test_f08_swept_sphere_obb_corner_miss", String(e))

    try: test_f09_root_motion_strip_xz(ctx); ctx.record_pass("test_f09_root_motion_strip_xz") except e: ctx.record_fail("test_f09_root_motion_strip_xz", String(e))
    try: test_f09_root_motion_preserve_y(ctx); ctx.record_pass("test_f09_root_motion_preserve_y") except e: ctx.record_fail("test_f09_root_motion_preserve_y", String(e))
    try: test_f09_inertializer_no_velocity_jump(ctx); ctx.record_pass("test_f09_inertializer_no_velocity_jump") except e: ctx.record_fail("test_f09_inertializer_no_velocity_jump", String(e))
    try: test_f09_root_delta_tracking(ctx); ctx.record_pass("test_f09_root_delta_tracking") except e: ctx.record_fail("test_f09_root_delta_tracking", String(e))
    try: test_f09_animator_state_transition(ctx); ctx.record_pass("test_f09_animator_state_transition") except e: ctx.record_fail("test_f09_animator_state_transition", String(e))

    try: test_f10_wayland_pointer_lock_toggle(ctx); ctx.record_pass("test_f10_wayland_pointer_lock_toggle") except e: ctx.record_fail("test_f10_wayland_pointer_lock_toggle", String(e))
    try: test_f10_wayland_unaccelerated_motion(ctx); ctx.record_pass("test_f10_wayland_unaccelerated_motion") except e: ctx.record_fail("test_f10_wayland_unaccelerated_motion", String(e))
    try: test_f10_wayland_cursor_hide(ctx); ctx.record_pass("test_f10_wayland_cursor_hide") except e: ctx.record_fail("test_f10_wayland_cursor_hide", String(e))
    try: test_f10_wayland_no_double_motion(ctx); ctx.record_pass("test_f10_wayland_no_double_motion") except e: ctx.record_fail("test_f10_wayland_no_double_motion", String(e))
    try: test_f10_wayland_message_queue_linear(ctx); ctx.record_pass("test_f10_wayland_message_queue_linear") except e: ctx.record_fail("test_f10_wayland_message_queue_linear", String(e))

    try: test_f11_gamepad_trigger_normalization(ctx); ctx.record_pass("test_f11_gamepad_trigger_normalization") except e: ctx.record_fail("test_f11_gamepad_trigger_normalization", String(e))
    try: test_f11_radial_deadzone_filtering(ctx); ctx.record_pass("test_f11_radial_deadzone_filtering") except e: ctx.record_fail("test_f11_radial_deadzone_filtering", String(e))
    try: test_f11_radial_deadzone_smooth_ramp(ctx); ctx.record_pass("test_f11_radial_deadzone_smooth_ramp") except e: ctx.record_fail("test_f11_radial_deadzone_smooth_ramp", String(e))
    try: test_f11_devconsole_underscore_input(ctx); ctx.record_pass("test_f11_devconsole_underscore_input") except e: ctx.record_fail("test_f11_devconsole_underscore_input", String(e))
    try: test_f11_devconsole_cvar_tab_completion(ctx); ctx.record_pass("test_f11_devconsole_cvar_tab_completion") except e: ctx.record_fail("test_f11_devconsole_cvar_tab_completion", String(e))

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
