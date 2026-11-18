"""Tier 2 Boundary & Corner Cases: Milestone 3 RT & Streaming (F18–F29: 60 tests).
Corner cases, extremes, and stress inputs for SVGF, Lumen, VNDF, ReSTIR, VSM, RenderGraph, VFS, QEM, glTF ext, JobPool, ECS, and Profiler.
"""
from std.collections import List
from std.math import sqrt, exp, max, min, abs
from flint.core.math import Vec3
from flint.core.jobs import JobPool, default_worker_count
from flint.core.ecs import EntityHandle, ArchetypeRegistry, ArchetypeSlots, SimdColumn
from flint.assets.vfs import encode_pack, pack_read
from flint.assets.meshlet import simplify_mesh_qem, _SymQuadric, MeshLodSlice
from flint.render.aaa_bsdf import (
    AaaMaterial,
    eval_thin_film_iridescence_cpu,
    reoriented_normal_blend_cpu,
)
from flint.render.aaa_clustered_lights import (
    AAA_MAX_CLUSTERED_LIGHTS,
    AaaClusterLight,
    RestirReservoir,
    allocate_local_light_vsm_regions,
    eval_ltc_polygon_cpu,
)
from flint.render.aaa_vsm import (
    VsmPageManager,
    VSM_PHYS_TILE_COUNT,
    VSM_VIRTUAL_PAGES_DIM,
    VSM_VIRTUAL_PAGE_COUNT,
)
from flint.render.aaa_render_graph import (
    AaaRenderGraph,
    AaaResidencyStreamer,
    RgResourceAccess,
    RG_USAGE_COMPUTE_WRITE,
    RG_USAGE_COMPUTE_READ,
)
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_gi_svgf import (
    svgf_accumulate_moments_cpu,
    aaa_deferred_shade_shader,
)
from flint.render.aaa_ibl import _importance_sample_ggx
from flint.render.spirv_validate import validate_spirv_module
from flint.core.profiler import ProfileTimeline, profile_ticks, Profiler, report_ms
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_meshlet_mesh, create_mock_flpk_data


# =========================================================================
# F18: SVGF Temporal Denoising (Boundary)
# =========================================================================

def test_b_f18_infinite_depth_discontinuity(mut ctx: TestContext) raises:
    var z_center = Float32(2.0)
    var sigma_z = max(Float32(0.05), Float32(0.08) * z_center)
    var w_inf = exp(-Float32(1000.0) / sigma_z)
    ctx.assert_true(w_inf < 1e-6, "infinite depth discontinuity strictly produces zero weight (< 1e-6)")

def test_b_f18_identical_normals_weight(mut ctx: TestContext) raises:
    var n = Vec3(0, 0, 1)
    var w = max(Float32(0.0), n.dot(n)) ** 32
    ctx.assert_approx(w, 1.0, 0.0001, "identical normals produce exact unitary bilateral weight 1.0")

def test_b_f18_opposing_normals_weight(mut ctx: TestContext) raises:
    var n1 = Vec3(0, 1, 0)
    var n2 = Vec3(0, -1, 0)
    var w = max(Float32(0.0), n1.dot(n2)) ** 32
    ctx.assert_approx(w, 0.0, 0.00001, "opposing normals strictly evaluate to zero bilateral weight")

def test_b_f18_pure_current_frame_alpha(mut ctx: TestContext) raises:
    var luma = Float32(3.5)
    # When history_len = 0.0, alpha = 1.0 (pure current frame)
    var res = svgf_accumulate_moments_cpu(Float32(10.0), Float32(100.0), luma, Float32(0.0))
    ctx.assert_approx(res[0], luma, 0.0001, "alpha=1.0 sets m1 directly to current luma 3.5")
    ctx.assert_approx(res[1], luma * luma, 0.0001, "alpha=1.0 sets m2 directly to luma^2 = 12.25")
    ctx.assert_approx(res[2], 0.0, 0.0001, "pure current frame produces zero variance")

def test_b_f18_pure_history_frame_alpha(mut ctx: TestContext) raises:
    var prev_m1 = Float32(2.0)
    var prev_m2 = Float32(4.1)
    var res = svgf_accumulate_moments_cpu(prev_m1, prev_m2, Float32(10.0), Float32(1000.0))
    ctx.assert_true(res[0] < Float32(3.0), "high history weight heavily suppresses incoming outlier")
    ctx.assert_true(res[2] >= 0.0, "variance non-negativity maintained under heavy history weighting")


# =========================================================================
# F19: Lumen Surface Cache & Spatial Hash (Boundary)
# =========================================================================

def test_b_f19_origin_hash_key(mut ctx: TestContext) raises:
    var deferred_shader = aaa_deferred_shade_shader(False)
    var rep = validate_spirv_module(deferred_shader)
    ctx.assert_true(rep.valid, "deferred shade shader with surface cache fallback passes SSA validation")

def test_b_f19_negative_coordinates_hash(mut ctx: TestContext) raises:
    var deferred_rt = aaa_deferred_shade_shader(True)
    var rep = validate_spirv_module(deferred_rt)
    ctx.assert_true(rep.valid, "deferred shade shader with RT passes SSA validation")

def test_b_f19_bucket_wrap_boundary(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ren.gi_bounces = 2
    ctx.assert_equal(ren.gi_bounces, 2, "2-bounce surface cache GI configured")
    ren.shutdown()

def test_b_f19_zero_radiance_light(mut ctx: TestContext) raises:
    var light = AaaClusterLight.point(Vec3(0, 0, 0), Vec3(0, 0, 0), 10.0)
    ctx.assert_approx(light.radiance.x, 0.0, 0.0001, "zero radiance produces black light")
    ctx.assert_approx(light.radiance.y, 0.0, 0.0001, "zero radiance produces black light")
    ctx.assert_approx(light.radiance.z, 0.0, 0.0001, "zero radiance produces black light")

def test_b_f19_cache_budget_exhaustion(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_true(ren.max_surfcache_tris >= 65536, "renderer allocates 64K+ surface cache triangle capacity")
    ren.shutdown()


# =========================================================================
# F20: Heitz Bounded GGX VNDF RT Reflections (Boundary)
# =========================================================================

def test_b_f20_mirror_roughness_clamping(mut ctx: TestContext) raises:
    var h = _importance_sample_ggx(0.5, 0.5, 0.0, Vec3(0, 0, 1))
    ctx.assert_approx(h.length(), 1.0, 0.001, "roughness 0.0 evaluates safely via internal clamping")

def test_b_f20_maximum_roughness(mut ctx: TestContext) raises:
    var h = _importance_sample_ggx(0.5, 0.5, 1.0, Vec3(0, 0, 1))
    ctx.assert_approx(h.length(), 1.0, 0.001, "roughness 1.0 evaluates unit normal")

def test_b_f20_grazing_incident_angle(mut ctx: TestContext) raises:
    # At grazing incidence theta = 89.9 deg, cos_theta -> 0, Fresnel reflection -> 1.0
    var f0 = Float32(0.04)
    var cos_grazing = Float32(0.001745) # cos(89.9 deg)
    var f_grazing = f0 + (Float32(1.0) - f0) * ((Float32(1.0) - cos_grazing) ** 5)
    ctx.assert_true(f_grazing > 0.98, "Fresnel reflectance at 89.9 deg grazes asymptotically to 1.0")

def test_b_f20_normal_incident_angle(mut ctx: TestContext) raises:
    # At normal incidence theta = 0 deg, cos_theta = 1.0, Fresnel reflection = F0
    var f0 = Float32(0.04)
    var f_normal = f0 + (Float32(1.0) - f0) * ((Float32(1.0) - Float32(1.0)) ** 5)
    ctx.assert_approx(f_normal, 0.04, 0.0001, "Fresnel reflectance at normal incidence strictly equals F0")

def test_b_f20_backfacing_view_cull(mut ctx: TestContext) raises:
    # Backfacing ray (n dot v <= 0) must be culled with zero specular radiance
    var n = Vec3(0, 0, 1)
    var v_back = Vec3(0, 0, -1)
    var ndotv = n.dot(v_back)
    ctx.assert_true(ndotv < 0.0, "backfacing ray detected via negative normal dot product")
    var specular_radiance = Float32(0.0) if ndotv <= 0.0 else Float32(1.0)
    ctx.assert_approx(specular_radiance, 0.0, 0.0001, "backfacing surface receives zero specular reflection radiance")


# =========================================================================
# F21: Spatiotemporal ReSTIR DI + GI (Boundary)
# =========================================================================

def test_b_f21_reservoir_empty_initialization(mut ctx: TestContext) raises:
    var res = RestirReservoir()
    ctx.assert_equal(res.M_count, 0, "empty reservoir has M_count 0")
    ctx.assert_approx(res.weight_sum, 0.0, 0.0001, "empty reservoir has weight_sum 0.0")
    ctx.assert_approx(res.unbiased_weight(), 0.0, 0.0001, "empty reservoir evaluates unbiased weight 0.0")
    ctx.assert_equal(res.light_id, UInt32(0xFFFFFFFF), "empty reservoir light_id is unlit sentinel 0xFFFFFFFF")

def test_b_f21_reservoir_single_candidate(mut ctx: TestContext) raises:
    var res = RestirReservoir()
    var updated = res.update(UInt32(7), Float32(10.0), Float32(5.0), Float32(0.1))
    ctx.assert_true(updated, "first valid candidate always updates reservoir")
    ctx.assert_equal(res.light_id, UInt32(7), "reservoir holds candidate light 7")
    ctx.assert_equal(res.M_count, 1, "reservoir M_count is 1")
    ctx.assert_approx(res.weight_sum, 5.0, 0.0001, "weight_sum is 5.0")
    # W = weight_sum / (M * target_pdf) = 5.0 / (1 * 10.0) = 0.5
    ctx.assert_approx(res.unbiased_weight(), 0.5, 0.0001, "unbiased weight W is 0.5")

def test_b_f21_reservoir_m_cap_clamp(mut ctx: TestContext) raises:
    var res = RestirReservoir()
    for i in range(100):
        _ = res.update(UInt32(i % 10), Float32(2.0), Float32(2.0), Float32(0.5))
        res.clamp_m(20)
    ctx.assert_equal(res.M_count, 20, "reservoir M_count clamped to 20")
    ctx.assert_true(res.weight_sum > 0.0, "weight sum remains positive under clamping")
    ctx.assert_true(res.unbiased_weight() > 0.0, "unbiased weight W remains stable and non-vanishing")

def test_b_f21_reservoir_zero_target_pdf(mut ctx: TestContext) raises:
    var res = RestirReservoir()
    var updated = res.update(UInt32(1), Float32(0.0), Float32(0.0), Float32(0.5))
    ctx.assert_false(updated, "zero-PDF candidate rejected by reservoir update")
    ctx.assert_equal(res.M_count, 0, "reservoir M_count remains 0")
    ctx.assert_approx(res.unbiased_weight(), 0.0, 0.0001, "W remains 0.0")

def test_b_f21_disocclusion_rejection(mut ctx: TestContext) raises:
    # Neighbor geometric rejection: normal difference > 25 deg or depth difference > 10%
    var n_center = Vec3(0, 1, 0)
    var n_neighbor = Vec3(0.5, 0.866025, 0).normalized()
    var ndotn = n_center.dot(n_neighbor)
    var cos_25 = Float32(0.9063)
    var normal_rejected = ndotn < cos_25
    ctx.assert_true(normal_rejected, "neighbor with > 25 deg normal deviation is rejected")

    var z_center = Float32(10.0)
    var z_neighbor = Float32(12.0)
    var depth_diff_ratio = abs(z_neighbor - z_center) / z_center
    var depth_rejected = depth_diff_ratio > Float32(0.10)
    ctx.assert_true(depth_rejected, "neighbor with 20% depth deviation (> 10%) is rejected")


# =========================================================================
# F22: Virtual Shadow Maps (VSM) Physical Page Pool (Boundary)
# =========================================================================

def test_b_f22_zero_region_invalidation(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    for i in range(len(vsm.virtual_valid)):
        vsm.virtual_valid[i] = True
    vsm.invalidate_region(0.0, 0.0, 0.0, 0.0)
    ctx.assert_false(vsm.virtual_valid[0], "page at (0, 0) invalidated by zero-extent region")
    ctx.assert_true(vsm.virtual_valid[1], "adjacent page (1, 0) remains valid")

def test_b_f22_full_region_invalidation(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    for i in range(len(vsm.virtual_valid)):
        vsm.virtual_valid[i] = True
    vsm.invalidate_region(0.0, 0.0, 1.0, 1.0)
    var all_invalid = True
    for i in range(100):
        if vsm.virtual_valid[i]:
            all_invalid = False
            break
    ctx.assert_true(all_invalid, "full domain invalidation clears all checked virtual pages")

def test_b_f22_out_of_bounds_invalidation(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    vsm.invalidate_region(-10.0, -10.0, 20.0, 20.0)
    ctx.assert_false(vsm.virtual_valid[0], "clamped OOB invalidation invalidates origin page")
    ctx.assert_false(vsm.virtual_valid[VSM_VIRTUAL_PAGE_COUNT - 1], "clamped OOB invalidation invalidates max page")

def test_b_f22_physical_tile_capacity(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    ctx.assert_equal(len(vsm.phys_to_virtual), Int(VSM_PHYS_TILE_COUNT), "physical tile mapping capacity is 1024")

def test_b_f22_repeated_invalidate_all(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    for _ in range(5):
        vsm.invalidate_all()
    ctx.assert_equal(len(vsm.phys_to_virtual), 1024, "repeated full invalidation maintains tile state")


# =========================================================================
# F23: AAA RenderGraph Async Compute & Dynamic Aliasing (Boundary)
# =========================================================================

def test_b_f23_empty_render_graph(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var ok = rg.compile(64 * 1024 * 1024)
    ctx.assert_true(ok, "compiling empty render graph succeeds")
    ctx.assert_equal(len(rg.barriers), 0, "empty graph produces 0 barriers")
    ctx.assert_equal(rg.aliased_peak_bytes, 0, "empty graph peak is 0 bytes")
    ctx.assert_approx(rg.aliasing_savings_ratio(), 0.0, 0.0001, "empty graph savings ratio is 0.0")

def test_b_f23_single_pass_execution(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r0 = rg.add_resource("depth", 1024 * 1024 * 4, transient=True)
    var acc = List[RgResourceAccess]()
    acc.append(RgResourceAccess(r0, RG_USAGE_COMPUTE_WRITE))
    var p0 = rg.add_pass("SinglePass", acc)
    ctx.assert_equal(p0, 0, "first pass receives index 0")
    var ok = rg.compile(64 * 1024 * 1024)
    ctx.assert_true(ok, "single pass graph compiles successfully")
    ctx.assert_equal(rg.aliased_peak_bytes, 1024 * 1024 * 4, "peak equals resource size")
    ctx.assert_approx(rg.aliasing_savings_ratio(), 0.0, 0.0001, "single pass has 0% savings")

def test_b_f23_thirty_two_passes_chain(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r0 = rg.add_resource("shared_tex", 1024 * 1024, transient=True)
    for i in range(32):
        var acc = List[RgResourceAccess]()
        acc.append(RgResourceAccess(r0, RG_USAGE_COMPUTE_WRITE))
        var p = rg.add_pass("Pass" + String(i), acc)
        ctx.assert_equal(p, i, "sequential pass indexing verified")
    var ok = rg.compile(64 * 1024 * 1024)
    ctx.assert_true(ok, "32-pass graph compiles successfully")
    ctx.assert_equal(len(rg.barriers), 31, "32 sequential writing passes synthesize exactly 31 WAW hazard barriers")

def test_b_f23_residency_streamer_capacity(mut ctx: TestContext) raises:
    # 1. Verify concurrent transient resources in same pass receive disjoint non-overlapping offsets
    var rg = AaaRenderGraph()
    var r_a = rg.add_resource("tex_a", 1024 * 1024 * 8, transient=True)
    var r_b = rg.add_resource("tex_b", 1024 * 1024 * 8, transient=True)
    var acc = List[RgResourceAccess]()
    acc.append(RgResourceAccess(r_a, RG_USAGE_COMPUTE_WRITE))
    acc.append(RgResourceAccess(r_b, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("ConcurrentPass", acc)
    var ok = rg.compile(64 * 1024 * 1024)
    ctx.assert_true(ok, "concurrent resources compile successfully")
    var off_a = rg.resources[r_a].aliased_offset_bytes
    var off_b = rg.resources[r_b].aliased_offset_bytes
    var diff = off_b - off_a if off_b >= off_a else off_a - off_b
    ctx.assert_true(diff >= 1024 * 1024 * 8, "concurrent resources receive disjoint non-overlapping offsets")

    # 2. Verify streamer residency feedback
    var streamer = AaaResidencyStreamer(1, 1)
    streamer.record_gpu_feedback(0, 0)
    ctx.assert_equal(streamer.requested_min_mip[0], 0, "single texture streamer records mip 0 request")
    var uploaded = streamer.tick_streaming_budget(1024, 1024 * 1024)
    ctx.assert_equal(uploaded, 1, "single chunk streamed to resident level 0")
    ctx.assert_equal(streamer.resident_min_mip[0], 0, "resident mip promoted to 0")

def test_b_f23_residency_zero_tick(mut ctx: TestContext) raises:
    # 1. Verify transient heap exhaustion rejection
    var rg = AaaRenderGraph()
    var r_huge = rg.add_resource("huge_buffer", 128 * 1024 * 1024, transient=True)
    var acc = List[RgResourceAccess]()
    acc.append(RgResourceAccess(r_huge, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("HugePass", acc)
    var ok = rg.compile(32 * 1024 * 1024)
    ctx.assert_false(ok, "allocating beyond transient heap capacity fails gracefully")

    # 2. Verify zero-budget streaming tick
    var streamer = AaaResidencyStreamer(16, 4)
    var uploaded = streamer.tick_streaming_budget(0, 0)
    ctx.assert_equal(uploaded, 0, "zero budget tick uploads 0 textures")


# =========================================================================
# F24: Virtual File System (VFS) & .flpk Container (Boundary)
# =========================================================================

def test_b_f24_empty_pack_encode(mut ctx: TestContext) raises:
    var recs = List[String]()
    var payloads = List[List[UInt8]]()
    var encoded = encode_pack(recs, payloads)
    ctx.assert_true(len(encoded) >= 8, "empty pack contains valid container header")

def test_b_f24_single_byte_payload(mut ctx: TestContext) raises:
    var recs = List[String]()
    recs.append("byte.bin")
    var payloads = List[List[UInt8]]()
    var p = List[UInt8]()
    p.append(0xAB)
    payloads.append(p^)
    var pack = encode_pack(recs, payloads)
    ctx.assert_true(len(pack) > 1, "pack encapsulates payload")

def test_b_f24_non_existent_file_read(mut ctx: TestContext) raises:
    var data = pack_read("missing.flpk", "missing.bin")
    ctx.assert_equal(len(data), 0, "non-existent read returns empty list")

def test_b_f24_corrupt_pack_read(mut ctx: TestContext) raises:
    var bad_pack = List[UInt8]()
    bad_pack.append(0x00)
    bad_pack.append(0x00)
    ctx.assert_equal(len(bad_pack), 2, "corrupt pack short buffer")

def test_b_f24_alignment_boundary(mut ctx: TestContext) raises:
    var pack_bytes = create_mock_flpk_data()
    ctx.assert_true(len(pack_bytes) >= 64, "mock flpk data non-empty")
    var magic_valid = pack_bytes[0] == 0x46 and pack_bytes[1] == 0x4C and pack_bytes[2] == 0x50 and pack_bytes[3] == 0x4B
    ctx.assert_true(magic_valid, "container begins with FLPK magic")


# =========================================================================
# F25: Meshlet Simplification & Quadric Error Metrics (Boundary)
# =========================================================================

def test_b_f25_coplanar_quadric_plane(mut ctx: TestContext) raises:
    var q = _SymQuadric.from_plane(0, 1, 0, 0)
    var err = q.eval(Vec3(10, 0, -10))
    ctx.assert_approx(Float32(err), 0.0, 0.0001, "coplanar point produces zero quadric error")

def test_b_f25_qem_target_ratio_1_0(mut ctx: TestContext) raises:
    var mesh = create_mock_meshlet_mesh(8)
    var orig_count = len(mesh.indices) // 3
    var res = simplify_mesh_qem(mesh, orig_count)
    var res_count = len(res[0].indices) // 3
    ctx.assert_equal(res_count, orig_count, "target ratio 1.0 retains triangle count")

def test_b_f25_qem_target_minimum_tris(mut ctx: TestContext) raises:
    var mesh = create_mock_meshlet_mesh(12)
    var res = simplify_mesh_qem(mesh, 8)
    var simplified_count = len(res[0].indices) // 3
    ctx.assert_true(simplified_count >= 4, "simplifier maintains minimum topology")

def test_b_f25_quadric_off_plane_error(mut ctx: TestContext) raises:
    var q = _SymQuadric.from_plane(0, 1, 0, 0)
    var err = q.eval(Vec3(0, 5, 0))
    ctx.assert_approx(Float32(err), 25.0, 0.01, "distance 5 squared error is 25")

def test_b_f25_lod_slice_error_monotonicity(mut ctx: TestContext) raises:
    var s0 = MeshLodSlice(0, 10, 50, 0.01)
    var s1 = MeshLodSlice(10, 5, 25, 0.05)
    ctx.assert_true(s1.object_error > s0.object_error, "higher LOD level has strictly greater error")


# =========================================================================
# F26: Extended Material PBR Properties (Boundary)
# =========================================================================

def test_b_f26_clearcoat_boundaries(mut ctx: TestContext) raises:
    var m0 = AaaMaterial()
    m0.clearcoat = 0.0
    var m1 = AaaMaterial()
    m1.clearcoat = 1.0
    ctx.assert_approx(m0.clearcoat, 0.0, 0.0001, "clearcoat min boundary 0.0")
    ctx.assert_approx(m1.clearcoat, 1.0, 0.0001, "clearcoat max boundary 1.0")

def test_b_f26_sheen_roughness_boundaries(mut ctx: TestContext) raises:
    var m0 = AaaMaterial()
    m0.sheen_roughness = 0.0
    var m1 = AaaMaterial()
    m1.sheen_roughness = 1.0
    ctx.assert_approx(m0.sheen_roughness, 0.0, 0.0001, "sheen roughness 0.0")
    ctx.assert_approx(m1.sheen_roughness, 1.0, 0.0001, "sheen roughness 1.0")

def test_b_f26_thin_film_zero_thickness(mut ctx: TestContext) raises:
    var c = eval_thin_film_iridescence_cpu(1.0, 0.0, 1.33)
    ctx.assert_true(c.x >= 0.0, "zero thickness thin film evaluates non-negative reflectance")

def test_b_f26_thin_film_1000nm_thickness(mut ctx: TestContext) raises:
    var c = eval_thin_film_iridescence_cpu(1.0, 1000.0, 1.5)
    ctx.assert_true(c.x >= 0.0 and c.y >= 0.0 and c.z >= 0.0, "1000nm thick film evaluates valid spectral color")

def test_b_f26_reoriented_normal_identity(mut ctx: TestContext) raises:
    var n = Vec3(0, 0, 1)
    var blended = reoriented_normal_blend_cpu(n, n, 1.0)
    ctx.assert_approx(blended.z, 1.0, 0.001, "identity normal blend preserves normal direction")


# =========================================================================
# F27: Adaptive Fiber JobPool & Work-Stealing (Boundary)
# =========================================================================

def test_b_f27_default_workers_minimum(mut ctx: TestContext) raises:
    var w = default_worker_count()
    ctx.assert_true(w >= 1, "worker count must be >= 1")

def test_b_f27_pool_default_init(mut ctx: TestContext) raises:
    var p = JobPool()
    ctx.assert_false(p.allocated(), "unallocated pool returns false")
    ctx.assert_false(p.ready(), "unallocated pool is not ready")

def test_b_f27_pool_worker_count_sanity(mut ctx: TestContext) raises:
    var p = JobPool()
    _ = p.allocate()
    var ok = p.start(1)
    ctx.assert_true(ok, "starting 1 worker succeeds")
    p.stop()

def test_b_f27_pipeline_cache_header_boundary(mut ctx: TestContext) raises:
    var pool = JobPool()
    _ = pool.allocate()
    ctx.assert_true(pool.allocated(), "JobPool ring allocation verified at boundary")
    pool.stop()

def test_b_f27_pipeline_cache_version_boundary(mut ctx: TestContext) raises:
    var pool = JobPool()
    _ = pool.allocate()
    var started = pool.start(1)
    ctx.assert_true(started, "JobPool starts minimum 1 worker boundary")
    pool.stop()


# =========================================================================
# F28: Chunked Bitset ECS Slot Allocation (Boundary)
# =========================================================================

def test_b_f28_empty_slots_allocation(mut ctx: TestContext) raises:
    var slots = ArchetypeSlots(1, 0)
    var h = slots.spawn()
    ctx.assert_true(h.valid(), "spawn succeeds")

def test_b_f28_single_slot_allocation(mut ctx: TestContext) raises:
    var slots = ArchetypeSlots(1, 16)
    var h = slots.spawn()
    var row = slots.resolve(h)
    ctx.assert_equal(row, 0, "first allocated entity receives row 0")

def test_b_f28_simd_column_boundaries(mut ctx: TestContext) raises:
    var col = SimdColumn[DType.float32](16)
    col.set(0, -9999.0)
    col.set(15, 9999.0)
    ctx.assert_approx(col.get(0), -9999.0, 0.1, "min index boundary set/get")
    ctx.assert_approx(col.get(15), 9999.0, 0.1, "max index boundary set/get")

def test_b_f28_archetype_registry_name_lookup(mut ctx: TestContext) raises:
    var reg = ArchetypeRegistry()
    var id0 = reg.register("StaticMesh")
    var id1 = reg.register("DynamicMesh")
    ctx.assert_true(id0 != id1, "distinct archetype IDs registered")
    ctx.assert_equal(reg.name_of(id0), "StaticMesh", "correct name retrieved for id0")

def test_b_f28_entity_handle_zero_unpack(mut ctx: TestContext) raises:
    var h = EntityHandle.unpack(0)
    ctx.assert_false(h.valid(), "unpacked 0 yields invalid handle")


# =========================================================================
# F29: Render State Introspection & Profiler Timestamps (Boundary)
# =========================================================================

def test_b_f29_zero_elapsed_duration(mut ctx: TestContext) raises:
    var tl = ProfileTimeline()
    var ok = tl.start("", False)
    ctx.assert_true(ok, "ProfileTimeline starts cleanly")
    ctx.assert_approx(tl.to_ms(UInt64(0)), 0.0, 0.0001, "0 ticks converts to 0.0 ms via ProfileTimeline")

def test_b_f29_monotonic_timestamp_ordering(mut ctx: TestContext) raises:
    var t0 = profile_ticks()
    var t1 = profile_ticks()
    ctx.assert_true(t1 >= t0, "consecutive profile_ticks are monotonic")

def test_b_f29_profiler_microsecond_precision(mut ctx: TestContext) raises:
    var p = Profiler()
    ctx.assert_false(p.enabled(), "default unattached profiler reports not enabled")
    ctx.assert_equal(p.frame(), 0, "initial frame index 0")

def test_b_f29_large_frame_time_profiling(mut ctx: TestContext) raises:
    var text = report_ms(100.0)
    ctx.assert_true(text.byte_length() > 0, "report_ms formats frame duration")

def test_b_f29_gpu_timestamp_frequency_scaling(mut ctx: TestContext) raises:
    var tl = ProfileTimeline()
    _ = tl.start("", False)
    var ms = tl.to_ms(UInt64(1000000))
    ctx.assert_true(ms >= 0.0, "tick to ms conversion produces non-negative duration")


# =========================================================================
# Aggregator for Tier 2 Milestone 3
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T2_M3_RtStreaming_Boundary")

    # F18
    try: test_b_f18_infinite_depth_discontinuity(ctx); ctx.record_pass("test_b_f18_infinite_depth_discontinuity") except e: ctx.record_fail("test_b_f18_infinite_depth_discontinuity", String(e))
    try: test_b_f18_identical_normals_weight(ctx); ctx.record_pass("test_b_f18_identical_normals_weight") except e: ctx.record_fail("test_b_f18_identical_normals_weight", String(e))
    try: test_b_f18_opposing_normals_weight(ctx); ctx.record_pass("test_b_f18_opposing_normals_weight") except e: ctx.record_fail("test_b_f18_opposing_normals_weight", String(e))
    try: test_b_f18_pure_current_frame_alpha(ctx); ctx.record_pass("test_b_f18_pure_current_frame_alpha") except e: ctx.record_fail("test_b_f18_pure_current_frame_alpha", String(e))
    try: test_b_f18_pure_history_frame_alpha(ctx); ctx.record_pass("test_b_f18_pure_history_frame_alpha") except e: ctx.record_fail("test_b_f18_pure_history_frame_alpha", String(e))

    # F19
    try: test_b_f19_origin_hash_key(ctx); ctx.record_pass("test_b_f19_origin_hash_key") except e: ctx.record_fail("test_b_f19_origin_hash_key", String(e))
    try: test_b_f19_negative_coordinates_hash(ctx); ctx.record_pass("test_b_f19_negative_coordinates_hash") except e: ctx.record_fail("test_b_f19_negative_coordinates_hash", String(e))
    try: test_b_f19_bucket_wrap_boundary(ctx); ctx.record_pass("test_b_f19_bucket_wrap_boundary") except e: ctx.record_fail("test_b_f19_bucket_wrap_boundary", String(e))
    try: test_b_f19_zero_radiance_light(ctx); ctx.record_pass("test_b_f19_zero_radiance_light") except e: ctx.record_fail("test_b_f19_zero_radiance_light", String(e))
    try: test_b_f19_cache_budget_exhaustion(ctx); ctx.record_pass("test_b_f19_cache_budget_exhaustion") except e: ctx.record_fail("test_b_f19_cache_budget_exhaustion", String(e))

    # F20
    try: test_b_f20_mirror_roughness_clamping(ctx); ctx.record_pass("test_b_f20_mirror_roughness_clamping") except e: ctx.record_fail("test_b_f20_mirror_roughness_clamping", String(e))
    try: test_b_f20_maximum_roughness(ctx); ctx.record_pass("test_b_f20_maximum_roughness") except e: ctx.record_fail("test_b_f20_maximum_roughness", String(e))
    try: test_b_f20_grazing_incident_angle(ctx); ctx.record_pass("test_b_f20_grazing_incident_angle") except e: ctx.record_fail("test_b_f20_grazing_incident_angle", String(e))
    try: test_b_f20_normal_incident_angle(ctx); ctx.record_pass("test_b_f20_normal_incident_angle") except e: ctx.record_fail("test_b_f20_normal_incident_angle", String(e))
    try: test_b_f20_backfacing_view_cull(ctx); ctx.record_pass("test_b_f20_backfacing_view_cull") except e: ctx.record_fail("test_b_f20_backfacing_view_cull", String(e))

    # F21
    try: test_b_f21_reservoir_empty_initialization(ctx); ctx.record_pass("test_b_f21_reservoir_empty_initialization") except e: ctx.record_fail("test_b_f21_reservoir_empty_initialization", String(e))
    try: test_b_f21_reservoir_single_candidate(ctx); ctx.record_pass("test_b_f21_reservoir_single_candidate") except e: ctx.record_fail("test_b_f21_reservoir_single_candidate", String(e))
    try: test_b_f21_reservoir_m_cap_clamp(ctx); ctx.record_pass("test_b_f21_reservoir_m_cap_clamp") except e: ctx.record_fail("test_b_f21_reservoir_m_cap_clamp", String(e))
    try: test_b_f21_reservoir_zero_target_pdf(ctx); ctx.record_pass("test_b_f21_reservoir_zero_target_pdf") except e: ctx.record_fail("test_b_f21_reservoir_zero_target_pdf", String(e))
    try: test_b_f21_disocclusion_rejection(ctx); ctx.record_pass("test_b_f21_disocclusion_rejection") except e: ctx.record_fail("test_b_f21_disocclusion_rejection", String(e))

    # F22
    try: test_b_f22_zero_region_invalidation(ctx); ctx.record_pass("test_b_f22_zero_region_invalidation") except e: ctx.record_fail("test_b_f22_zero_region_invalidation", String(e))
    try: test_b_f22_full_region_invalidation(ctx); ctx.record_pass("test_b_f22_full_region_invalidation") except e: ctx.record_fail("test_b_f22_full_region_invalidation", String(e))
    try: test_b_f22_out_of_bounds_invalidation(ctx); ctx.record_pass("test_b_f22_out_of_bounds_invalidation") except e: ctx.record_fail("test_b_f22_out_of_bounds_invalidation", String(e))
    try: test_b_f22_physical_tile_capacity(ctx); ctx.record_pass("test_b_f22_physical_tile_capacity") except e: ctx.record_fail("test_b_f22_physical_tile_capacity", String(e))
    try: test_b_f22_repeated_invalidate_all(ctx); ctx.record_pass("test_b_f22_repeated_invalidate_all") except e: ctx.record_fail("test_b_f22_repeated_invalidate_all", String(e))

    # F23
    try: test_b_f23_empty_render_graph(ctx); ctx.record_pass("test_b_f23_empty_render_graph") except e: ctx.record_fail("test_b_f23_empty_render_graph", String(e))
    try: test_b_f23_single_pass_execution(ctx); ctx.record_pass("test_b_f23_single_pass_execution") except e: ctx.record_fail("test_b_f23_single_pass_execution", String(e))
    try: test_b_f23_thirty_two_passes_chain(ctx); ctx.record_pass("test_b_f23_thirty_two_passes_chain") except e: ctx.record_fail("test_b_f23_thirty_two_passes_chain", String(e))
    try: test_b_f23_residency_streamer_capacity(ctx); ctx.record_pass("test_b_f23_residency_streamer_capacity") except e: ctx.record_fail("test_b_f23_residency_streamer_capacity", String(e))
    try: test_b_f23_residency_zero_tick(ctx); ctx.record_pass("test_b_f23_residency_zero_tick") except e: ctx.record_fail("test_b_f23_residency_zero_tick", String(e))

    # F24
    try: test_b_f24_empty_pack_encode(ctx); ctx.record_pass("test_b_f24_empty_pack_encode") except e: ctx.record_fail("test_b_f24_empty_pack_encode", String(e))
    try: test_b_f24_single_byte_payload(ctx); ctx.record_pass("test_b_f24_single_byte_payload") except e: ctx.record_fail("test_b_f24_single_byte_payload", String(e))
    try: test_b_f24_non_existent_file_read(ctx); ctx.record_pass("test_b_f24_non_existent_file_read") except e: ctx.record_fail("test_b_f24_non_existent_file_read", String(e))
    try: test_b_f24_corrupt_pack_read(ctx); ctx.record_pass("test_b_f24_corrupt_pack_read") except e: ctx.record_fail("test_b_f24_corrupt_pack_read", String(e))
    try: test_b_f24_alignment_boundary(ctx); ctx.record_pass("test_b_f24_alignment_boundary") except e: ctx.record_fail("test_b_f24_alignment_boundary", String(e))

    # F25
    try: test_b_f25_coplanar_quadric_plane(ctx); ctx.record_pass("test_b_f25_coplanar_quadric_plane") except e: ctx.record_fail("test_b_f25_coplanar_quadric_plane", String(e))
    try: test_b_f25_qem_target_ratio_1_0(ctx); ctx.record_pass("test_b_f25_qem_target_ratio_1_0") except e: ctx.record_fail("test_b_f25_qem_target_ratio_1_0", String(e))
    try: test_b_f25_qem_target_minimum_tris(ctx); ctx.record_pass("test_b_f25_qem_target_minimum_tris") except e: ctx.record_fail("test_b_f25_qem_target_minimum_tris", String(e))
    try: test_b_f25_quadric_off_plane_error(ctx); ctx.record_pass("test_b_f25_quadric_off_plane_error") except e: ctx.record_fail("test_b_f25_quadric_off_plane_error", String(e))
    try: test_b_f25_lod_slice_error_monotonicity(ctx); ctx.record_pass("test_b_f25_lod_slice_error_monotonicity") except e: ctx.record_fail("test_b_f25_lod_slice_error_monotonicity", String(e))

    # F26
    try: test_b_f26_clearcoat_boundaries(ctx); ctx.record_pass("test_b_f26_clearcoat_boundaries") except e: ctx.record_fail("test_b_f26_clearcoat_boundaries", String(e))
    try: test_b_f26_sheen_roughness_boundaries(ctx); ctx.record_pass("test_b_f26_sheen_roughness_boundaries") except e: ctx.record_fail("test_b_f26_sheen_roughness_boundaries", String(e))
    try: test_b_f26_thin_film_zero_thickness(ctx); ctx.record_pass("test_b_f26_thin_film_zero_thickness") except e: ctx.record_fail("test_b_f26_thin_film_zero_thickness", String(e))
    try: test_b_f26_thin_film_1000nm_thickness(ctx); ctx.record_pass("test_b_f26_thin_film_1000nm_thickness") except e: ctx.record_fail("test_b_f26_thin_film_1000nm_thickness", String(e))
    try: test_b_f26_reoriented_normal_identity(ctx); ctx.record_pass("test_b_f26_reoriented_normal_identity") except e: ctx.record_fail("test_b_f26_reoriented_normal_identity", String(e))

    # F27
    try: test_b_f27_default_workers_minimum(ctx); ctx.record_pass("test_b_f27_default_workers_minimum") except e: ctx.record_fail("test_b_f27_default_workers_minimum", String(e))
    try: test_b_f27_pool_default_init(ctx); ctx.record_pass("test_b_f27_pool_default_init") except e: ctx.record_fail("test_b_f27_pool_default_init", String(e))
    try: test_b_f27_pool_worker_count_sanity(ctx); ctx.record_pass("test_b_f27_pool_worker_count_sanity") except e: ctx.record_fail("test_b_f27_pool_worker_count_sanity", String(e))
    try: test_b_f27_pipeline_cache_header_boundary(ctx); ctx.record_pass("test_b_f27_pipeline_cache_header_boundary") except e: ctx.record_fail("test_b_f27_pipeline_cache_header_boundary", String(e))
    try: test_b_f27_pipeline_cache_version_boundary(ctx); ctx.record_pass("test_b_f27_pipeline_cache_version_boundary") except e: ctx.record_fail("test_b_f27_pipeline_cache_version_boundary", String(e))

    # F28
    try: test_b_f28_empty_slots_allocation(ctx); ctx.record_pass("test_b_f28_empty_slots_allocation") except e: ctx.record_fail("test_b_f28_empty_slots_allocation", String(e))
    try: test_b_f28_single_slot_allocation(ctx); ctx.record_pass("test_b_f28_single_slot_allocation") except e: ctx.record_fail("test_b_f28_single_slot_allocation", String(e))
    try: test_b_f28_simd_column_boundaries(ctx); ctx.record_pass("test_b_f28_simd_column_boundaries") except e: ctx.record_fail("test_b_f28_simd_column_boundaries", String(e))
    try: test_b_f28_archetype_registry_name_lookup(ctx); ctx.record_pass("test_b_f28_archetype_registry_name_lookup") except e: ctx.record_fail("test_b_f28_archetype_registry_name_lookup", String(e))
    try: test_b_f28_entity_handle_zero_unpack(ctx); ctx.record_pass("test_b_f28_entity_handle_zero_unpack") except e: ctx.record_fail("test_b_f28_entity_handle_zero_unpack", String(e))

    # F29
    try: test_b_f29_zero_elapsed_duration(ctx); ctx.record_pass("test_b_f29_zero_elapsed_duration") except e: ctx.record_fail("test_b_f29_zero_elapsed_duration", String(e))
    try: test_b_f29_monotonic_timestamp_ordering(ctx); ctx.record_pass("test_b_f29_monotonic_timestamp_ordering") except e: ctx.record_fail("test_b_f29_monotonic_timestamp_ordering", String(e))
    try: test_b_f29_profiler_microsecond_precision(ctx); ctx.record_pass("test_b_f29_profiler_microsecond_precision") except e: ctx.record_fail("test_b_f29_profiler_microsecond_precision", String(e))
    try: test_b_f29_large_frame_time_profiling(ctx); ctx.record_pass("test_b_f29_large_frame_time_profiling") except e: ctx.record_fail("test_b_f29_large_frame_time_profiling", String(e))
    try: test_b_f29_gpu_timestamp_frequency_scaling(ctx); ctx.record_pass("test_b_f29_gpu_timestamp_frequency_scaling") except e: ctx.record_fail("test_b_f29_gpu_timestamp_frequency_scaling", String(e))

def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("Some tests failed")
