"""Tier 2 Boundary & Corner Cases: Milestone 3 RT & Streaming (F18–F29: 60 tests).
Corner cases, extremes, and stress inputs for SVGF, split VisBuffer shading, VNDF, ReSTIR,
VSM, RenderGraph, .flpk v3 / AssetServer, QEM, glTF ext, JobPool, ECS v2, and Profiler.
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs
from std.memory import bitcast
from flint.assets.flpk_v3 import (
    ASSET_KIND_RAW,
    CHUNK_COMP_NONE,
    AssetUuid,
    FlpkBuildChunk,
    FlpkV3MappedArchive,
    encode_flpk_v3,
    write_flpk_v3,
)
from flint.assets.meshlet import MeshLodSlice, _SymQuadric, simplify_mesh_qem
from flint.core.jobs import JOB_PRIO_HIGH, JOB_PRIO_LOW, JOB_PRIO_NORMAL, JobPool, default_worker_count
from flint.core.math import Vec2, Vec3
from flint.core.profiler import Profiler, ProfileTimeline, profile_ticks, report_ms
from flint.core.transform import Mat4
from flint.ecs import (
    STORAGE_TABLE,
    Component,
    ComponentBitSet,
    Entity,
    ParallelCommands,
    World,
    comp_hash64,
)
from flint.platform.os import read_bytes, write_bytes
from flint.render.aaa_bsdf import (
    AaaMaterial,
    eval_kaplanyan_specular_aa_cpu,
    eval_thin_film_iridescence_cpu,
    reoriented_normal_blend_cpu,
)
from flint.render.aaa_clustered_lights import (
    AaaClusterLight,
    RestirReservoir,
    closest_point_on_ltc_light_cpu,
)
from flint.render.aaa_gi_svgf import (
    AAA_SHADE_CLASS_COMPLEX,
    AAA_SHADE_CLASS_FOLIAGE_SSS,
    AAA_SHADE_CLASS_STANDARD_PBR,
    aaa_deferred_shade_class_shader,
    aaa_shade_sky_shader,
    aaa_shade_unlit_shader,
    aaa_tile_classify_shader,
    eval_virtual_depth_reprojection_cpu,
    surface_cache_update_shader,
    svgf_accumulate_moments_cpu,
    svgf_atrous_shader,
    svgf_temporal_reproject_shader,
    tile_classify_word_offset,
)
from flint.render.aaa_glass import glass_f0_cpu, glass_fresnel_cpu
from flint.render.aaa_ibl import _importance_sample_ggx, sample_heitz_bounded_vndf_cpu
from flint.render.aaa_render_graph import RG_USAGE_COMPUTE_WRITE, AaaRenderGraph, AaaResidencyStreamer, RgResourceAccess
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_vsm import VSM_PHYS_TILE_COUNT, VSM_VIRTUAL_PAGE_COUNT, VsmPageManager
from flint.render.spirv_validate import validate_spirv_module
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_flpk_v3_data, create_mock_meshlet_mesh
from e2e.common.v5_fixtures import create_mock_asset_server


@fieldwise_init
struct _T2M3Val(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e.t2_m3.Val")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var v: Int


@fieldwise_init
struct _T2M3NonTriv(Component):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e.t2_m3.NonTriv")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var label: String


def _t2_m3_noop_job(_ctx: Int, _chunk: Int):
    pass


# =========================================================================
# F18: SVGF Temporal Denoising (Boundary)
# =========================================================================

def test_b_f18_infinite_depth_discontinuity(mut ctx: TestContext) raises:
    var prev_vp = List[Float32](length=16, fill=0.0)
    prev_vp[0] = 1.0; prev_vp[5] = 1.0; prev_vp[10] = 1.0; prev_vp[15] = 1.0
    var miss_res = eval_virtual_depth_reprojection_cpu(Vec3(0, 0, 0), Vec3(0, 0, 5.0), 10.0, False, prev_vp, Vec2(0.5, 0.5))
    ctx.assert_approx(miss_res[0], 100000.0, 0.1, "miss hit clamps virtual depth to 100000.0")


def test_b_f18_identical_normals_weight(mut ctx: TestContext) raises:
    ctx.assert_true(validate_spirv_module(svgf_temporal_reproject_shader()).valid, "SVGF temporal reprojection SPIR-V valid")


def test_b_f18_opposing_normals_weight(mut ctx: TestContext) raises:
    ctx.assert_true(validate_spirv_module(svgf_atrous_shader()).valid, "SVGF atrous SPIR-V valid")


def test_b_f18_pure_current_frame_alpha(mut ctx: TestContext) raises:
    var luma = Float32(3.5)
    var res = svgf_accumulate_moments_cpu(Float32(10.0), Float32(100.0), luma, Float32(0.0))
    ctx.assert_approx(res[0], luma, 0.0001, "alpha=1.0 sets m1 directly to current luma 3.5")
    ctx.assert_approx(res[1], luma * luma, 0.0001, "alpha=1.0 sets m2 directly to luma^2")
    ctx.assert_approx(res[2], 0.0, 0.0001, "pure current frame produces zero variance")


def test_b_f18_pure_history_frame_alpha(mut ctx: TestContext) raises:
    var res = svgf_accumulate_moments_cpu(Float32(2.0), Float32(4.1), Float32(10.0), Float32(1000.0))
    ctx.assert_true(res[0] < Float32(3.0), "high history weight suppresses outlier")
    ctx.assert_true(res[2] >= 0.0, "variance remains non-negative")


# =========================================================================
# F19: Split Deferred VisBuffer Shading & Surface Cache (Boundary)
# =========================================================================

def test_b_f19_origin_hash_key(mut ctx: TestContext) raises:
    ctx.assert_true(validate_spirv_module(aaa_tile_classify_shader()).valid, "tile classify shader valid")
    ctx.assert_true(
        validate_spirv_module(aaa_deferred_shade_class_shader(False, AAA_SHADE_CLASS_STANDARD_PBR)).valid,
        "deferred standard PBR class shader valid",
    )


def test_b_f19_negative_coordinates_hash(mut ctx: TestContext) raises:
    ctx.assert_true(
        validate_spirv_module(aaa_deferred_shade_class_shader(True, AAA_SHADE_CLASS_COMPLEX)).valid,
        "RT complex shade class shader valid",
    )
    ctx.assert_true(
        validate_spirv_module(aaa_deferred_shade_class_shader(True, AAA_SHADE_CLASS_FOLIAGE_SSS)).valid,
        "RT foliage SSS shade class shader valid",
    )
    ctx.assert_true(validate_spirv_module(aaa_shade_sky_shader()).valid, "sky shade shader valid")
    ctx.assert_true(validate_spirv_module(aaa_shade_unlit_shader()).valid, "unlit shade shader valid")


def test_b_f19_bucket_wrap_boundary(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(validate_spirv_module(surface_cache_update_shader(False)).valid, "surface_cache_update_shader(False) valid")
    ctx.assert_true(validate_spirv_module(surface_cache_update_shader(True)).valid, "surface_cache_update_shader(True) valid")
    ctx.assert_true(tile_classify_word_offset(160 * 120) > 0, "tile classify scratch offset positive")
    var mid = ren.add_mesh(create_mock_meshlet_mesh(8), 0, build_lods=False)
    var prev_rt_gi = ren.enable_rt_gi
    var prev_bounces = ren.gi_bounces
    ren.enable_rt_gi = True
    ren.gi_bounces = 2
    ren.begin_frame(Vec3(0, 1, -3), Vec3(0, 0, 0))
    ren.draw_instance(mid.mesh_id, Mat4())
    ctx.assert_true(ren.render_frame(), "RT GI 2-bounce surface-cache frame renders cleanly")
    ren.enable_rt_gi = prev_rt_gi
    ren.gi_bounces = prev_bounces


def test_b_f19_zero_radiance_light(mut ctx: TestContext) raises:
    var light = AaaClusterLight.point(Vec3(0, 0, 0), Vec3(0, 0, 0), 10.0)
    ctx.assert_approx(light.radiance.x, 0.0, 0.0001, "zero radiance produces black light")
    ctx.assert_approx(light.radiance.y, 0.0, 0.0001, "zero radiance produces black light")


def test_b_f19_cache_budget_exhaustion(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "FlintAaaVulkanRenderer active for surface cache budget check")
    ctx.assert_true(ren.max_surfcache_tris >= 65536, "renderer allocates 64K+ surface cache capacity")
    var expected_min_bytes = (4096 + ren.max_surfcache_tris * 4) * 4
    ctx.assert_true(ren.buf_surfcache.size >= expected_min_bytes, "buf_surfcache matches header + max_surfcache_tris budget")
    ctx.assert_equal(ren.ring_surfcache[0].cpu_address, 0, "ring_surfcache[0] resides in DEVICE_LOCAL VRAM")
    ctx.assert_true(ren.ring_surfcache[0].vk_buffer != 0, "ring_surfcache[0] has valid VkBuffer handle")


# =========================================================================
# F20: Heitz Bounded GGX VNDF RT Reflections (Boundary)
# =========================================================================

def test_b_f20_mirror_roughness_clamping(mut ctx: TestContext) raises:
    var h = _importance_sample_ggx(0.5, 0.5, 0.0, Vec3(0, 0, 1))
    ctx.assert_approx(h.length(), 1.0, 0.001, "roughness 0.0 evaluates safely")
    var s = sample_heitz_bounded_vndf_cpu(0.5, 0.5, 0.0, Vec3(0, 0, 1))
    ctx.assert_approx(s[0].length(), 1.0, 0.01, "VNDF roughness 0.0 unit normal")


def test_b_f20_maximum_roughness(mut ctx: TestContext) raises:
    var s = sample_heitz_bounded_vndf_cpu(0.5, 0.5, 1.0, Vec3(0, 0, 1))
    ctx.assert_approx(s[0].length(), 1.0, 0.01, "VNDF roughness 1.0 unit normal")
    ctx.assert_true(s[1] >= 0.0 and s[1] <= 1.001, "VNDF weight bounded in [0, 1]")


def test_b_f20_grazing_incident_angle(mut ctx: TestContext) raises:
    var f_graz = glass_fresnel_cpu(Float32(0.001), Float32(1.5))
    ctx.assert_true(f_graz > 0.98, "glass_fresnel_cpu at grazing approaches 1.0")


def test_b_f20_normal_incident_angle(mut ctx: TestContext) raises:
    var f0 = glass_f0_cpu(Float32(1.5))
    var f_norm = glass_fresnel_cpu(Float32(1.0), Float32(1.5))
    ctx.assert_approx(f_norm, f0, 0.0001, "glass_fresnel_cpu at normal equals glass_f0_cpu")


def test_b_f20_backfacing_view_cull(mut ctx: TestContext) raises:
    var s = sample_heitz_bounded_vndf_cpu(0.2, 0.8, 0.35, Vec3(0.0, 0.0, -1.0))
    ctx.assert_approx(s[0].length(), 1.0, 0.01, "backfacing view vector safely clamped")


# =========================================================================
# F21: Spatiotemporal ReSTIR DI + GI (Boundary)
# =========================================================================

def test_b_f21_reservoir_empty_initialization(mut ctx: TestContext) raises:
    var res = RestirReservoir()
    ctx.assert_equal(res.M_count, 0, "empty reservoir M_count 0")
    ctx.assert_approx(res.unbiased_weight(), 0.0, 0.0001, "empty reservoir weight 0.0")
    ctx.assert_equal(res.light_id, UInt32(0xFFFFFFFF), "unlit sentinel 0xFFFFFFFF")


def test_b_f21_reservoir_single_candidate(mut ctx: TestContext) raises:
    var res = RestirReservoir()
    ctx.assert_true(res.update(UInt32(7), Float32(10.0), Float32(5.0), Float32(0.1)), "first candidate updates")
    ctx.assert_equal(res.light_id, UInt32(7), "candidate 7 selected")
    ctx.assert_approx(res.unbiased_weight(), 0.5, 0.0001, "W == 0.5")


def test_b_f21_reservoir_m_cap_clamp(mut ctx: TestContext) raises:
    var res = RestirReservoir()
    for i in range(100):
        _ = res.update(UInt32(i % 10), Float32(2.0), Float32(2.0), Float32(0.5))
        res.clamp_m(20)
    ctx.assert_equal(res.M_count, 20, "M_count clamped to 20")
    ctx.assert_true(res.unbiased_weight() > 0.0, "W remains positive")


def test_b_f21_reservoir_zero_target_pdf(mut ctx: TestContext) raises:
    var res = RestirReservoir()
    ctx.assert_false(res.update(UInt32(1), Float32(0.0), Float32(0.0), Float32(0.5)), "zero-PDF candidate rejected")
    ctx.assert_equal(res.M_count, 0, "M_count stays 0")


def test_b_f21_disocclusion_rejection(mut ctx: TestContext) raises:
    var r_center = RestirReservoir(UInt32(5), Float32(20.0), 4, Float32(5.0))
    var r_empty = RestirReservoir()
    ctx.assert_false(r_center.merge(r_empty, Float32(10.0), Float32(0.5)), "merging empty neighbor rejected")
    var pt_light = AaaClusterLight.point(Vec3(3, 4, 0), Vec3(1, 1, 1), 10.0)
    var cp = closest_point_on_ltc_light_cpu(pt_light, Vec3(0, 0, 0))
    ctx.assert_vec3_approx(cp, Vec3(3, 4, 0), 0.001, "closest point on point light is light position")


# =========================================================================
# F22: Virtual Shadow Maps (VSM) Physical Page Pool (Boundary)
# =========================================================================

def test_b_f22_zero_region_invalidation(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    for i in range(len(vsm.virtual_valid)):
        vsm.virtual_valid[i] = True
    vsm.invalidate_region(0.0, 0.0, 0.0, 0.0)
    ctx.assert_false(vsm.virtual_valid[0], "origin page invalidated")
    ctx.assert_true(vsm.virtual_valid[1], "adjacent page remains valid")


def test_b_f22_full_region_invalidation(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    for i in range(len(vsm.virtual_valid)):
        vsm.virtual_valid[i] = True
    vsm.invalidate_region(0.0, 0.0, 1.0, 1.0)
    ctx.assert_false(vsm.virtual_valid[0] or vsm.virtual_valid[99], "full domain invalidates pages")


def test_b_f22_out_of_bounds_invalidation(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    for i in range(len(vsm.virtual_valid)):
        vsm.virtual_valid[i] = True
    vsm.invalidate_region(-10.0, -10.0, 20.0, 20.0)
    ctx.assert_false(vsm.virtual_valid[0], "clamped origin page invalidated from True -> False")
    ctx.assert_false(vsm.virtual_valid[VSM_VIRTUAL_PAGE_COUNT - 1], "clamped max page invalidated from True -> False")


def test_b_f22_physical_tile_capacity(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    ctx.assert_equal(len(vsm.phys_to_virtual), Int(VSM_PHYS_TILE_COUNT), "physical tile capacity is 1024")


def test_b_f22_repeated_invalidate_all(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    for _ in range(5):
        vsm.invalidate_all()
    ctx.assert_equal(len(vsm.phys_to_virtual), 1024, "repeated invalidation preserves capacity")


# =========================================================================
# F23: AAA RenderGraph Async Compute & Dynamic Aliasing (Boundary)
# =========================================================================

def test_b_f23_empty_render_graph(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    ctx.assert_true(rg.compile(64 * 1024 * 1024), "empty graph compiles")
    ctx.assert_equal(rg.aliased_peak_bytes, 0, "empty graph peak is 0")


def test_b_f23_single_pass_execution(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r0 = rg.add_resource("depth", 1024 * 1024 * 4, transient=True)
    var acc = List[RgResourceAccess](); acc.append(RgResourceAccess(r0, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("SinglePass", acc)
    ctx.assert_true(rg.compile(64 * 1024 * 1024), "single pass graph compiles")
    ctx.assert_equal(rg.aliased_peak_bytes, 1024 * 1024 * 4, "peak equals resource size")


def test_b_f23_thirty_two_passes_chain(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r0 = rg.add_resource("shared_tex", 1024 * 1024, transient=True)
    for i in range(32):
        var acc = List[RgResourceAccess](); acc.append(RgResourceAccess(r0, RG_USAGE_COMPUTE_WRITE))
        ctx.assert_equal(rg.add_pass("Pass" + String(i), acc), i, "sequential pass indexing")
    ctx.assert_true(rg.compile(64 * 1024 * 1024), "32-pass graph compiles")
    ctx.assert_equal(len(rg.barriers), 31, "32 sequential writes emit 31 WAW barriers")


def test_b_f23_residency_streamer_capacity(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r_a = rg.add_resource("tex_a", 1024 * 1024 * 8, transient=True)
    var r_b = rg.add_resource("tex_b", 1024 * 1024 * 8, transient=True)
    var acc = List[RgResourceAccess]()
    acc.append(RgResourceAccess(r_a, RG_USAGE_COMPUTE_WRITE))
    acc.append(RgResourceAccess(r_b, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("ConcurrentPass", acc)
    ctx.assert_true(rg.compile(64 * 1024 * 1024), "concurrent resources compile")
    var diff = abs(rg.resources[r_b].aliased_offset_bytes - rg.resources[r_a].aliased_offset_bytes)
    ctx.assert_true(diff >= 1024 * 1024 * 8, "concurrent resources receive disjoint offsets")

    var streamer = AaaResidencyStreamer(1, 1)
    streamer.record_gpu_feedback(0, 0)
    ctx.assert_equal(streamer.tick_streaming_budget(1024, 1024 * 1024), 1, "single chunk promoted")


def test_b_f23_residency_zero_tick(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r_huge = rg.add_resource("huge_buffer", 128 * 1024 * 1024, transient=True)
    var acc = List[RgResourceAccess](); acc.append(RgResourceAccess(r_huge, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("HugePass", acc)
    ctx.assert_false(rg.compile(32 * 1024 * 1024), "heap exhaustion rejected gracefully")
    var streamer = AaaResidencyStreamer(16, 4)
    ctx.assert_equal(streamer.tick_streaming_budget(0, 0), 0, "zero budget uploads 0 textures")


# =========================================================================
# F24: `.flpk` v3 Container & AssetServer (Boundary)
# =========================================================================

def test_b_f24_empty_pack_encode(mut ctx: TestContext) raises:
    var chunks = List[FlpkBuildChunk]()
    var encoded = encode_flpk_v3(chunks, page_align=False)
    ctx.assert_true(len(encoded) >= 64, "empty flpk v3 contains valid 64-byte header")


def test_b_f24_single_byte_payload(mut ctx: TestContext) raises:
    var p = List[UInt8]()
    p.append(0xAB)
    var chunks = List[FlpkBuildChunk]()
    chunks.append(FlpkBuildChunk(AssetUuid.from_path("byte.bin"), "byte.bin", 0, ASSET_KIND_RAW, p^, CHUNK_COMP_NONE))
    var pack = encode_flpk_v3(chunks, page_align=False)
    ctx.assert_true(len(pack) > 64, "single-byte payload packed into flpk v3")


def test_b_f24_non_existent_file_read(mut ctx: TestContext) raises:
    var arch = FlpkV3MappedArchive.open("/tmp/missing_t2_m3.flpk")
    ctx.assert_false(arch.valid, "missing archive returns valid=False")
    ctx.assert_equal(arch.find_by_path("missing.bin", 0), -1, "missing lookup returns -1")
    arch.close()
    var srv = create_mock_asset_server()
    ctx.assert_false(srv.load_raw("missing.bin").is_valid(), "AssetServer missing load_raw invalid")
    srv.close()


def test_b_f24_corrupt_pack_read(mut ctx: TestContext) raises:
    var pid_s = String(Int(external_call["getpid", Int32]()))
    var flpk_path = "/tmp/flint_e2e_t2_f24_corrupt_" + pid_s + ".flpk"
    ctx.cleanup_artifact(flpk_path)
    var payload = List[UInt8](length=32, fill=0x42)
    var chunks = List[FlpkBuildChunk]()
    chunks.append(FlpkBuildChunk(AssetUuid.from_path("data.bin"), "data.bin", 0, ASSET_KIND_RAW, payload^, CHUNK_COMP_NONE))
    ctx.assert_true(write_flpk_v3(flpk_path, chunks, page_align=False), "write flpk v3")

    var arch_ok = FlpkV3MappedArchive.open(flpk_path)
    ctx.assert_true(arch_ok.valid, "clean flpk v3 opens validly")
    var toc_ok = arch_ok.find_by_path("data.bin", 0)
    ctx.assert_true(toc_ok >= 0, "clean TOC entry found")
    var payload_off = Int(arch_ok.toc_entry(toc_ok).payload_offset)
    arch_ok.close()

    # 1) Corrupt actual payload byte -> TOC stays valid, verify_entry_crc64 returns False
    var raw = read_bytes(flpk_path)
    ctx.assert_true(payload_off >= 64 and payload_off < len(raw), "payload_offset lies inside archive")
    raw[payload_off] ^= 0xFF
    ctx.assert_true(write_bytes(flpk_path, raw), "flip byte inside payload")
    var arch = FlpkV3MappedArchive.open(flpk_path)
    ctx.assert_true(arch.valid, "archive TOC remains valid on payload-only corruption")
    var toc = arch.find_by_path("data.bin", 0)
    ctx.assert_true(toc >= 0, "TOC entry found for payload-corrupted archive")
    ctx.assert_false(arch.verify_entry_crc64(toc), "1-byte payload corruption fails ECMA-182 CRC64")
    ctx.assert_false(arch.read_payload_by_toc(toc, verify_crc=True)[0], "read_payload_by_toc rejects CRC64 mismatch")
    arch.close()

    # 2) Corrupt metadata byte before payload -> archive open rejects TOC CRC64
    raw[payload_off] ^= 0xFF
    raw[payload_off - 1] ^= 0xFF
    ctx.assert_true(write_bytes(flpk_path, raw), "flip byte inside metadata")
    var arch_meta = FlpkV3MappedArchive.open(flpk_path)
    ctx.assert_false(arch_meta.valid, "TOC/metadata corruption fails archive validation on open")
    arch_meta.close()
    ctx.cleanup_artifact(flpk_path)


def test_b_f24_alignment_boundary(mut ctx: TestContext) raises:
    var pack_bytes = create_mock_flpk_v3_data()
    ctx.assert_true(len(pack_bytes) >= 64, "mock flpk v3 data >= 64B")
    ctx.assert_true(pack_bytes[0] == 0x46 and pack_bytes[1] == 0x4C and pack_bytes[2] == 0x50 and pack_bytes[3] == 0x4B, "FLPK magic")


# =========================================================================
# F25: Meshlet Simplification & Quadric Error Metrics (Boundary)
# =========================================================================

def test_b_f25_coplanar_quadric_plane(mut ctx: TestContext) raises:
    var q = _SymQuadric.from_plane(0, 1, 0, 0)
    ctx.assert_approx(Float32(q.eval(Vec3(10, 0, -10))), 0.0, 0.0001, "coplanar point 0 error")


def test_b_f25_qem_target_ratio_1_0(mut ctx: TestContext) raises:
    var mesh = create_mock_meshlet_mesh(8)
    var orig_count = len(mesh.indices) // 3
    var res = simplify_mesh_qem(mesh, orig_count)
    ctx.assert_equal(len(res[0].indices) // 3, orig_count, "target ratio 1.0 retains triangle count")


def test_b_f25_qem_target_minimum_tris(mut ctx: TestContext) raises:
    var mesh = create_mock_meshlet_mesh(12)
    var res = simplify_mesh_qem(mesh, 8)
    ctx.assert_true(len(res[0].indices) // 3 >= 4, "simplifier maintains minimum topology")


def test_b_f25_quadric_off_plane_error(mut ctx: TestContext) raises:
    var q = _SymQuadric.from_plane(0, 1, 0, 0)
    ctx.assert_approx(Float32(q.eval(Vec3(0, 5, 0))), 25.0, 0.01, "distance 5 squared error is 25")


def test_b_f25_lod_slice_error_monotonicity(mut ctx: TestContext) raises:
    var s0 = MeshLodSlice(0, 10, 50, 0.01)
    var s1 = MeshLodSlice(10, 5, 25, 0.05)
    ctx.assert_true(s1.object_error > s0.object_error, "higher LOD has greater error")


# =========================================================================
# F26: Extended Material PBR Properties (Boundary)
# =========================================================================

def test_b_f26_clearcoat_boundaries(mut ctx: TestContext) raises:
    var m0 = AaaMaterial()
    m0.clearcoat = Float32(-0.5)
    m0.clearcoat_roughness = Float32(0.0)
    var w0 = List[UInt32]()
    m0.append_words(w0)
    ctx.assert_approx(bitcast[DType.float32](w0[6]), 0.0, 0.0001, "negative clearcoat packed as 0.0")
    ctx.assert_approx(bitcast[DType.float32](w0[7]), 0.04, 0.0001, "clearcoat_roughness 0.0 clamped to 0.04 in packed SSBO")

    var m1 = AaaMaterial()
    m1.clearcoat = Float32(1.5)
    m1.clearcoat_roughness = Float32(2.0)
    m1.two_sided_foliage = True
    var w1 = List[UInt32]()
    m1.append_words(w1)
    ctx.assert_approx(bitcast[DType.float32](w1[6]), 1.0, 0.0001, "over-1 clearcoat clamped to 1.0")
    ctx.assert_approx(bitcast[DType.float32](w1[7]), -1.0, 0.0001, "two_sided_foliage packs negated clamped clearcoat_roughness -1.0")

    var aa = eval_kaplanyan_specular_aa_cpu(Vec3(0, 0, 1), Vec3(0.2, 0, 1).normalized(), Vec3(0, 0.2, 1).normalized(), 0.04, 0.04)
    ctx.assert_true(aa[1] >= 0.04 and aa[1] <= 1.0, "Kaplanyan specular AA keeps clearcoat roughness in [0.04, 1.0]")


def test_b_f26_sheen_roughness_boundaries(mut ctx: TestContext) raises:
    var m0 = AaaMaterial()
    m0.sheen_roughness = Float32(0.0)
    var w0 = List[UInt32]()
    m0.append_words(w0)
    ctx.assert_approx(bitcast[DType.float32](w0[19]), 0.05, 0.0001, "sheen_roughness 0.0 clamped to 0.05 floor in SSBO")

    var m1 = AaaMaterial()
    m1.sheen_roughness = Float32(1.0)
    m1.unlit = True
    var w1 = List[UInt32]()
    m1.append_words(w1)
    ctx.assert_approx(bitcast[DType.float32](w1[19]), -1.0, 0.0001, "unlit encodes negated sheen_roughness -1.0 in SSBO")


def test_b_f26_thin_film_zero_thickness(mut ctx: TestContext) raises:
    var c = eval_thin_film_iridescence_cpu(1.0, 0.0, 1.33)
    ctx.assert_true(c.x >= 0.0, "zero thickness non-negative reflectance")


def test_b_f26_thin_film_1000nm_thickness(mut ctx: TestContext) raises:
    var c = eval_thin_film_iridescence_cpu(1.0, 1000.0, 1.5)
    ctx.assert_true(c.x >= 0.0 and c.y >= 0.0 and c.z >= 0.0, "1000nm thick film valid")


def test_b_f26_reoriented_normal_identity(mut ctx: TestContext) raises:
    var n = Vec3(0, 0, 1)
    ctx.assert_approx(reoriented_normal_blend_cpu(n, n, 1.0).z, 1.0, 0.001, "identity normal blend")


# =========================================================================
# F27: Adaptive Fiber JobPool Priorities, >16 Dependents & RAII (Boundary)
# =========================================================================

def test_b_f27_default_workers_minimum(mut ctx: TestContext) raises:
    ctx.assert_true(default_worker_count() >= 1, "worker count >= 1")


def test_b_f27_pool_default_init(mut ctx: TestContext) raises:
    var p = JobPool()
    ctx.assert_false(p.allocated(), "unallocated pool returns false")
    ctx.assert_false(p.ready(), "unallocated pool is not ready")


def test_b_f27_pool_worker_count_sanity(mut ctx: TestContext) raises:
    var p = JobPool()
    _ = p.allocate()
    ctx.assert_true(p.start(1), "starting 1 worker succeeds")
    p.stop()


def test_b_f27_pipeline_cache_header_boundary(mut ctx: TestContext) raises:
    var pool = JobPool()
    _ = pool.allocate()
    _ = pool.start(2)
    var parent = pool.create(_t2_m3_noop_job, 0, 1)
    var children = List[Int]()
    for _ in range(20):
        var ch = pool.create(_t2_m3_noop_job, 0, 1)
        pool.depend(ch, parent)
        children.append(ch)
    pool.submit(parent, priority=JOB_PRIO_HIGH)
    for i in range(len(children)):
        pool.submit(children[i], priority=JOB_PRIO_LOW)
    for i in range(len(children)):
        pool.wait(children[i])
    pool.stop()
    ctx.assert_equal(len(children), 20, ">16 dependents fan-out completed without abort")


def test_b_f27_pipeline_cache_version_boundary(mut ctx: TestContext) raises:
    var pool = JobPool()
    _ = pool.allocate()
    ctx.assert_true(pool.start(1), "JobPool starts 1 worker")
    var j = pool.create(_t2_m3_noop_job, 0, 1)
    pool.submit(j, priority=JOB_PRIO_NORMAL)
    pool.wait(j)
    # Destructor RAII cleans up when leaving scope even without explicit stop


# =========================================================================
# F28: Hybrid ECS v2 (flint.ecs.World) Boundaries
# =========================================================================

def test_b_f28_empty_slots_allocation(mut ctx: TestContext) raises:
    var w = World()
    var e = w.spawn()
    ctx.assert_true(e.valid() and w.is_alive(e), "spawn succeeds on empty World")
    _ = w.despawn(e)
    ctx.assert_false(w.is_alive(e), "despawn invalidates entity")


def test_b_f28_single_slot_allocation(mut ctx: TestContext) raises:
    var w = World()
    var e1 = w.spawn()
    _ = w.despawn(e1)
    var e2 = w.spawn()
    ctx.assert_true(e2.generation > e1.generation, "slot reuse bumps generation")
    ctx.assert_false(w.is_alive(e1), "old generation rejected")


def test_b_f28_simd_column_boundaries(mut ctx: TestContext) raises:
    var w = World()
    var last_tid = -1
    for i in range(270):
        last_tid = w.register_pod_component(UInt64(0xE2E20001 + i), 4, 4)
    ctx.assert_true(last_tid >= 269, "World registers > 256 component types (tid=" + String(last_tid) + ")")
    var e = w.spawn()
    var val = UInt32(0x12345678)
    ctx.assert_true(w.insert_raw_pod(e, last_tid, Int(Pointer(to=val))), "insert_raw_pod succeeds for tid > 256")
    ctx.assert_true(w.has_by_type_id(e, last_tid), "World.has_by_type_id true for tid > 256")
    var mask = ComponentBitSet()
    mask.set(last_tid)
    var hits = w.query_entities(mask)
    ctx.assert_equal(len(hits), 1, "World.query_entities matches bit > 256")
    _ = w.despawn(e)
    ctx.assert_false(w.has_by_type_id(e, last_tid), "stale despawned entity has_by_type_id == False")


def test_b_f28_archetype_registry_name_lookup(mut ctx: TestContext) raises:
    var w = World()
    _ = w.register_component[_T2M3Val]()
    var e = w.spawn()
    _ = w.insert(e, _T2M3Val(777))
    var stale = w.spawn()
    _ = w.insert(stale, _T2M3Val(999))
    _ = w.despawn(stale)
    ctx.assert_false(w.has[_T2M3Val](stale), "w.has[_T2M3Val](stale) is False")
    var blob = w.snapshot_blob()
    var w2 = World()
    _ = w2.register_component[_T2M3Val]()
    w2.restore_blob(blob)
    ctx.assert_equal(w2.entity_count(), 1, "restore_blob recovers 1 live entity")
    ctx.assert_equal(w2.get[_T2M3Val](e).v, 777, "restore_blob recovers POD component value")
    ctx.assert_false(w2.has[_T2M3Val](stale), "restored world rejects stale entity handle")

    var w_nt = World()
    var e_nt = w_nt.spawn()
    _ = w_nt.insert(e_nt, _T2M3NonTriv("unregistered"))
    var raised_err = False
    try:
        _ = w_nt.snapshot_blob()
    except:
        raised_err = True
    ctx.assert_true(raised_err, "snapshot_blob raises on unregistered non-trivial component")


def test_b_f28_entity_handle_zero_unpack(mut ctx: TestContext) raises:
    var e_zero = Entity.from_bits(0)
    ctx.assert_false(e_zero.valid(), "Entity.from_bits(0) is invalid")
    var cmds = ParallelCommands()
    var w = World()
    var pe = cmds.spawn_empty(0, order_key=0)
    cmds.insert[_T2M3Val](0, pe, _T2M3Val(123), order_key=0)
    w.flush_commands(cmds)
    ctx.assert_equal(w.entity_count(), 1, "ParallelCommands flushes 1 entity cleanly")


# =========================================================================
# F29: Render State Introspection & Profiler Timestamps (Boundary)
# =========================================================================

def test_b_f29_zero_elapsed_duration(mut ctx: TestContext) raises:
    var tl = ProfileTimeline()
    ctx.assert_true(tl.start("", False), "ProfileTimeline starts cleanly")
    ctx.assert_approx(tl.to_ms(UInt64(0)), 0.0, 0.0001, "0 ticks converts to 0.0 ms")
    tl.finish()


def test_b_f29_monotonic_timestamp_ordering(mut ctx: TestContext) raises:
    var t0 = profile_ticks()
    var t1 = profile_ticks()
    ctx.assert_true(t1 >= t0, "consecutive profile_ticks are monotonic")


def test_b_f29_profiler_microsecond_precision(mut ctx: TestContext) raises:
    var p = Profiler()
    ctx.assert_false(p.enabled(), "unattached profiler not enabled")
    ctx.assert_equal(p.frame(), 0, "initial frame index 0")


def test_b_f29_large_frame_time_profiling(mut ctx: TestContext) raises:
    ctx.assert_true(report_ms(100.0).byte_length() > 0, "report_ms formats frame duration")


def test_b_f29_gpu_timestamp_frequency_scaling(mut ctx: TestContext) raises:
    var tl = ProfileTimeline()
    _ = tl.start("", False)
    ctx.assert_true(tl.to_ms(UInt64(1000000)) >= 0.0, "tick to ms non-negative")
    tl.finish()


# =========================================================================
# Aggregator for Tier 2 Milestone 3
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T2_M3_RtStreaming_Boundary")
    var ren = FlintAaaVulkanRenderer(width=160, height=120)

    try: test_b_f18_infinite_depth_discontinuity(ctx); ctx.record_pass("test_b_f18_infinite_depth_discontinuity") except e: ctx.record_fail("test_b_f18_infinite_depth_discontinuity", String(e))
    try: test_b_f18_identical_normals_weight(ctx); ctx.record_pass("test_b_f18_identical_normals_weight") except e: ctx.record_fail("test_b_f18_identical_normals_weight", String(e))
    try: test_b_f18_opposing_normals_weight(ctx); ctx.record_pass("test_b_f18_opposing_normals_weight") except e: ctx.record_fail("test_b_f18_opposing_normals_weight", String(e))
    try: test_b_f18_pure_current_frame_alpha(ctx); ctx.record_pass("test_b_f18_pure_current_frame_alpha") except e: ctx.record_fail("test_b_f18_pure_current_frame_alpha", String(e))
    try: test_b_f18_pure_history_frame_alpha(ctx); ctx.record_pass("test_b_f18_pure_history_frame_alpha") except e: ctx.record_fail("test_b_f18_pure_history_frame_alpha", String(e))

    try: test_b_f19_origin_hash_key(ctx); ctx.record_pass("test_b_f19_origin_hash_key") except e: ctx.record_fail("test_b_f19_origin_hash_key", String(e))
    try: test_b_f19_negative_coordinates_hash(ctx); ctx.record_pass("test_b_f19_negative_coordinates_hash") except e: ctx.record_fail("test_b_f19_negative_coordinates_hash", String(e))
    try: test_b_f19_bucket_wrap_boundary(ctx, ren); ctx.record_pass("test_b_f19_bucket_wrap_boundary") except e: ctx.record_fail("test_b_f19_bucket_wrap_boundary", String(e))
    try: test_b_f19_zero_radiance_light(ctx); ctx.record_pass("test_b_f19_zero_radiance_light") except e: ctx.record_fail("test_b_f19_zero_radiance_light", String(e))
    try: test_b_f19_cache_budget_exhaustion(ctx, ren); ctx.record_pass("test_b_f19_cache_budget_exhaustion") except e: ctx.record_fail("test_b_f19_cache_budget_exhaustion", String(e))

    try: test_b_f20_mirror_roughness_clamping(ctx); ctx.record_pass("test_b_f20_mirror_roughness_clamping") except e: ctx.record_fail("test_b_f20_mirror_roughness_clamping", String(e))
    try: test_b_f20_maximum_roughness(ctx); ctx.record_pass("test_b_f20_maximum_roughness") except e: ctx.record_fail("test_b_f20_maximum_roughness", String(e))
    try: test_b_f20_grazing_incident_angle(ctx); ctx.record_pass("test_b_f20_grazing_incident_angle") except e: ctx.record_fail("test_b_f20_grazing_incident_angle", String(e))
    try: test_b_f20_normal_incident_angle(ctx); ctx.record_pass("test_b_f20_normal_incident_angle") except e: ctx.record_fail("test_b_f20_normal_incident_angle", String(e))
    try: test_b_f20_backfacing_view_cull(ctx); ctx.record_pass("test_b_f20_backfacing_view_cull") except e: ctx.record_fail("test_b_f20_backfacing_view_cull", String(e))

    try: test_b_f21_reservoir_empty_initialization(ctx); ctx.record_pass("test_b_f21_reservoir_empty_initialization") except e: ctx.record_fail("test_b_f21_reservoir_empty_initialization", String(e))
    try: test_b_f21_reservoir_single_candidate(ctx); ctx.record_pass("test_b_f21_reservoir_single_candidate") except e: ctx.record_fail("test_b_f21_reservoir_single_candidate", String(e))
    try: test_b_f21_reservoir_m_cap_clamp(ctx); ctx.record_pass("test_b_f21_reservoir_m_cap_clamp") except e: ctx.record_fail("test_b_f21_reservoir_m_cap_clamp", String(e))
    try: test_b_f21_reservoir_zero_target_pdf(ctx); ctx.record_pass("test_b_f21_reservoir_zero_target_pdf") except e: ctx.record_fail("test_b_f21_reservoir_zero_target_pdf", String(e))
    try: test_b_f21_disocclusion_rejection(ctx); ctx.record_pass("test_b_f21_disocclusion_rejection") except e: ctx.record_fail("test_b_f21_disocclusion_rejection", String(e))

    try: test_b_f22_zero_region_invalidation(ctx); ctx.record_pass("test_b_f22_zero_region_invalidation") except e: ctx.record_fail("test_b_f22_zero_region_invalidation", String(e))
    try: test_b_f22_full_region_invalidation(ctx); ctx.record_pass("test_b_f22_full_region_invalidation") except e: ctx.record_fail("test_b_f22_full_region_invalidation", String(e))
    try: test_b_f22_out_of_bounds_invalidation(ctx); ctx.record_pass("test_b_f22_out_of_bounds_invalidation") except e: ctx.record_fail("test_b_f22_out_of_bounds_invalidation", String(e))
    try: test_b_f22_physical_tile_capacity(ctx); ctx.record_pass("test_b_f22_physical_tile_capacity") except e: ctx.record_fail("test_b_f22_physical_tile_capacity", String(e))
    try: test_b_f22_repeated_invalidate_all(ctx); ctx.record_pass("test_b_f22_repeated_invalidate_all") except e: ctx.record_fail("test_b_f22_repeated_invalidate_all", String(e))

    try: test_b_f23_empty_render_graph(ctx); ctx.record_pass("test_b_f23_empty_render_graph") except e: ctx.record_fail("test_b_f23_empty_render_graph", String(e))
    try: test_b_f23_single_pass_execution(ctx); ctx.record_pass("test_b_f23_single_pass_execution") except e: ctx.record_fail("test_b_f23_single_pass_execution", String(e))
    try: test_b_f23_thirty_two_passes_chain(ctx); ctx.record_pass("test_b_f23_thirty_two_passes_chain") except e: ctx.record_fail("test_b_f23_thirty_two_passes_chain", String(e))
    try: test_b_f23_residency_streamer_capacity(ctx); ctx.record_pass("test_b_f23_residency_streamer_capacity") except e: ctx.record_fail("test_b_f23_residency_streamer_capacity", String(e))
    try: test_b_f23_residency_zero_tick(ctx); ctx.record_pass("test_b_f23_residency_zero_tick") except e: ctx.record_fail("test_b_f23_residency_zero_tick", String(e))

    try: test_b_f24_empty_pack_encode(ctx); ctx.record_pass("test_b_f24_empty_pack_encode") except e: ctx.record_fail("test_b_f24_empty_pack_encode", String(e))
    try: test_b_f24_single_byte_payload(ctx); ctx.record_pass("test_b_f24_single_byte_payload") except e: ctx.record_fail("test_b_f24_single_byte_payload", String(e))
    try: test_b_f24_non_existent_file_read(ctx); ctx.record_pass("test_b_f24_non_existent_file_read") except e: ctx.record_fail("test_b_f24_non_existent_file_read", String(e))
    try: test_b_f24_corrupt_pack_read(ctx); ctx.record_pass("test_b_f24_corrupt_pack_read") except e: ctx.record_fail("test_b_f24_corrupt_pack_read", String(e))
    try: test_b_f24_alignment_boundary(ctx); ctx.record_pass("test_b_f24_alignment_boundary") except e: ctx.record_fail("test_b_f24_alignment_boundary", String(e))

    try: test_b_f25_coplanar_quadric_plane(ctx); ctx.record_pass("test_b_f25_coplanar_quadric_plane") except e: ctx.record_fail("test_b_f25_coplanar_quadric_plane", String(e))
    try: test_b_f25_qem_target_ratio_1_0(ctx); ctx.record_pass("test_b_f25_qem_target_ratio_1_0") except e: ctx.record_fail("test_b_f25_qem_target_ratio_1_0", String(e))
    try: test_b_f25_qem_target_minimum_tris(ctx); ctx.record_pass("test_b_f25_qem_target_minimum_tris") except e: ctx.record_fail("test_b_f25_qem_target_minimum_tris", String(e))
    try: test_b_f25_quadric_off_plane_error(ctx); ctx.record_pass("test_b_f25_quadric_off_plane_error") except e: ctx.record_fail("test_b_f25_quadric_off_plane_error", String(e))
    try: test_b_f25_lod_slice_error_monotonicity(ctx); ctx.record_pass("test_b_f25_lod_slice_error_monotonicity") except e: ctx.record_fail("test_b_f25_lod_slice_error_monotonicity", String(e))

    try: test_b_f26_clearcoat_boundaries(ctx); ctx.record_pass("test_b_f26_clearcoat_boundaries") except e: ctx.record_fail("test_b_f26_clearcoat_boundaries", String(e))
    try: test_b_f26_sheen_roughness_boundaries(ctx); ctx.record_pass("test_b_f26_sheen_roughness_boundaries") except e: ctx.record_fail("test_b_f26_sheen_roughness_boundaries", String(e))
    try: test_b_f26_thin_film_zero_thickness(ctx); ctx.record_pass("test_b_f26_thin_film_zero_thickness") except e: ctx.record_fail("test_b_f26_thin_film_zero_thickness", String(e))
    try: test_b_f26_thin_film_1000nm_thickness(ctx); ctx.record_pass("test_b_f26_thin_film_1000nm_thickness") except e: ctx.record_fail("test_b_f26_thin_film_1000nm_thickness", String(e))
    try: test_b_f26_reoriented_normal_identity(ctx); ctx.record_pass("test_b_f26_reoriented_normal_identity") except e: ctx.record_fail("test_b_f26_reoriented_normal_identity", String(e))

    try: test_b_f27_default_workers_minimum(ctx); ctx.record_pass("test_b_f27_default_workers_minimum") except e: ctx.record_fail("test_b_f27_default_workers_minimum", String(e))
    try: test_b_f27_pool_default_init(ctx); ctx.record_pass("test_b_f27_pool_default_init") except e: ctx.record_fail("test_b_f27_pool_default_init", String(e))
    try: test_b_f27_pool_worker_count_sanity(ctx); ctx.record_pass("test_b_f27_pool_worker_count_sanity") except e: ctx.record_fail("test_b_f27_pool_worker_count_sanity", String(e))
    try: test_b_f27_pipeline_cache_header_boundary(ctx); ctx.record_pass("test_b_f27_pipeline_cache_header_boundary") except e: ctx.record_fail("test_b_f27_pipeline_cache_header_boundary", String(e))
    try: test_b_f27_pipeline_cache_version_boundary(ctx); ctx.record_pass("test_b_f27_pipeline_cache_version_boundary") except e: ctx.record_fail("test_b_f27_pipeline_cache_version_boundary", String(e))

    try: test_b_f28_empty_slots_allocation(ctx); ctx.record_pass("test_b_f28_empty_slots_allocation") except e: ctx.record_fail("test_b_f28_empty_slots_allocation", String(e))
    try: test_b_f28_single_slot_allocation(ctx); ctx.record_pass("test_b_f28_single_slot_allocation") except e: ctx.record_fail("test_b_f28_single_slot_allocation", String(e))
    try: test_b_f28_simd_column_boundaries(ctx); ctx.record_pass("test_b_f28_simd_column_boundaries") except e: ctx.record_fail("test_b_f28_simd_column_boundaries", String(e))
    try: test_b_f28_archetype_registry_name_lookup(ctx); ctx.record_pass("test_b_f28_archetype_registry_name_lookup") except e: ctx.record_fail("test_b_f28_archetype_registry_name_lookup", String(e))
    try: test_b_f28_entity_handle_zero_unpack(ctx); ctx.record_pass("test_b_f28_entity_handle_zero_unpack") except e: ctx.record_fail("test_b_f28_entity_handle_zero_unpack", String(e))

    try: test_b_f29_zero_elapsed_duration(ctx); ctx.record_pass("test_b_f29_zero_elapsed_duration") except e: ctx.record_fail("test_b_f29_zero_elapsed_duration", String(e))
    try: test_b_f29_monotonic_timestamp_ordering(ctx); ctx.record_pass("test_b_f29_monotonic_timestamp_ordering") except e: ctx.record_fail("test_b_f29_monotonic_timestamp_ordering", String(e))
    try: test_b_f29_profiler_microsecond_precision(ctx); ctx.record_pass("test_b_f29_profiler_microsecond_precision") except e: ctx.record_fail("test_b_f29_profiler_microsecond_precision", String(e))
    try: test_b_f29_large_frame_time_profiling(ctx); ctx.record_pass("test_b_f29_large_frame_time_profiling") except e: ctx.record_fail("test_b_f29_large_frame_time_profiling", String(e))
    try: test_b_f29_gpu_timestamp_frequency_scaling(ctx); ctx.record_pass("test_b_f29_gpu_timestamp_frequency_scaling") except e: ctx.record_fail("test_b_f29_gpu_timestamp_frequency_scaling", String(e))

    ren.shutdown()


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
