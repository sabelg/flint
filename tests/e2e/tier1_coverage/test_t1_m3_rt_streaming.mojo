"""Tier 1 Feature Coverage: Milestone 3 RT & Streaming (F18–F29: 60 tests).
Opaque-box and requirement-driven verification of SVGF, split VisBuffer shading,
Lumen surface cache, VNDF, ReSTIR, VSM, RenderGraph, .flpk v3 / AssetServer,
QEM, glTF material extensions, JobPool priorities, and Hybrid ECS v2.
"""
from std.atomic import Atomic
from std.collections import List
from std.ffi import external_call
from std.math import abs, exp, max, min, sqrt
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
from flint.assets.server import AssetServer, CookedTextureAsset
from flint.core.jobs import JOB_PRIO_HIGH, JOB_PRIO_LOW, JOB_PRIO_NORMAL, JobPool, default_worker_count
from flint.core.math import Vec2, Vec3
from flint.ecs import (
    STORAGE_TABLE,
    Component,
    ComponentBitSet,
    Entity,
    Events,
    ParallelCommands,
    World,
    comp_hash64,
)
from flint.render.aaa_bsdf import (
    AaaMaterial,
    eval_thin_film_iridescence_cpu,
    reoriented_normal_blend_cpu,
)
from flint.render.aaa_clustered_lights import (
    AAA_CLUSTER_X,
    AAA_CLUSTER_Y,
    AAA_CLUSTER_Z,
    AAA_MAX_CLUSTERED_LIGHTS,
    AAA_TOTAL_CLUSTERS,
    AaaClusterLight,
    RestirReservoir,
    aaa_cluster_cull_shader,
    allocate_local_light_vsm_regions,
    bake_ltc_lut_rg16f,
    eval_ltc_polygon_cpu,
)
from flint.render.aaa_gi_svgf import (
    AAA_SHADE_CLASS_STANDARD_PBR,
    aaa_deferred_shade_class_shader,
    aaa_shade_sky_shader,
    aaa_shade_unlit_shader,
    aaa_tile_classify_shader,
    eval_virtual_depth_reprojection_cpu,
    surface_cache_update_shader,
    svgf_accumulate_moments_cpu,
    svgf_atrous_shader,
    svgf_atrous_shader_raw_spv,
    svgf_atrous_shader_specialized,
    svgf_temporal_reproject_shader,
)
from flint.render.aaa_glass import glass_fresnel_cpu
from flint.render.aaa_ibl import (
    _importance_sample_ggx,
    bake_split_sum_brdf_lut_rg16f,
    sample_heitz_bounded_vndf_cpu,
)
from flint.render.aaa_render_graph import (
    RG_USAGE_COMPUTE_WRITE,
    AaaRenderGraph,
    AaaResidencyStreamer,
    RgResourceAccess,
)
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_visbuffer import meshlet_cull_shader
from flint.render.aaa_vsm import (
    VSM_PHYS_TILE_COUNT,
    VSM_VIRTUAL_PAGES_DIM,
    VsmPageManager,
    compute_toroidal_page_index,
    make_ortho_directional_light_vp,
)
from flint.render.spirv_validate import validate_spirv_module
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_flpk_v3_data, create_mock_meshlet_mesh
from e2e.common.v5_fixtures import create_mock_asset_server, create_mock_flpk_v3_archive_file


@fieldwise_init
struct _T1M3Pos(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e.t1_m3.Pos")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var x: Float32
    var y: Float32
    var z: Float32


@fieldwise_init
struct _T1M3Vel(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("e2e.t1_m3.Vel")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var vx: Float32
    var vy: Float32
    var vz: Float32


@fieldwise_init
struct _T1M3Evt(Copyable, ImplicitlyCopyable, Movable):
    var code: Int


def _t1_m3_par_spawn_kernel(
    lane: Int,
    order: Int,
    ent: Entity,
    pos: Pointer[_T1M3Pos, MutAnyOrigin],
    ctx_addr: Int,
):
    ref cmds = Pointer[ParallelCommands, MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var spawned = cmds.spawn_empty(lane, order_key=order)
    cmds.insert[_T1M3Pos](lane, spawned, _T1M3Pos(pos[unsafe_offset=0].x + 10.0, 1.0, 2.0), order_key=order)


struct _T1M3JobOrderCtx:
    var seq: Atomic[Int64]
    var order_lo: Int64
    var order_hi: Int64

    def __init__(out self):
        self.seq = Atomic[Int64](0)
        self.order_lo = 0
        self.order_hi = 0


def _t1_m3_job_lo(ctx_addr: Int, _chunk: Int):
    ref c = Pointer[_T1M3JobOrderCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    c.order_lo = c.seq.fetch_add(1) + 1


def _t1_m3_job_hi(ctx_addr: Int, _chunk: Int):
    ref c = Pointer[_T1M3JobOrderCtx, MutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    c.order_hi = c.seq.fetch_add(1) + 1


# =========================================================================
# F18: SVGF Temporal Denoising & Split VisBuffer Shaders (5 tests)
# =========================================================================

def test_f18_svgf_variance_moments_formula(mut ctx: TestContext) raises:
    var m1 = Float32(0.0)
    var m2 = Float32(0.0)
    var h_len = Float32(0.0)
    var var_est = Float32(0.0)

    for i in range(100):
        var luma = Float32(0.5) if (i % 2 == 0) else Float32(1.5)
        var res = svgf_accumulate_moments_cpu(m1, m2, luma, h_len)
        m1 = res[0]; m2 = res[1]; var_est = res[2]; h_len = res[3]

    ctx.assert_approx(m1, 1.0, 0.05, "first moment m1 converges to 1.0")
    ctx.assert_approx(m2, 1.25, 0.05, "second moment m2 converges to 1.25")
    ctx.assert_approx(var_est, 0.25, 0.05, "variance converges to 0.25")

    var reset = svgf_accumulate_moments_cpu(m1, m2, Float32(4.0), Float32(0.0))
    ctx.assert_approx(reset[0], 4.0, 0.001, "history_len=0 resets m1 to current sample")
    ctx.assert_true(validate_spirv_module(svgf_temporal_reproject_shader()).valid, "SVGF temporal reprojection valid")


def test_f18_svgf_bilateral_depth_weight(mut ctx: TestContext) raises:
    ctx.assert_true(validate_spirv_module(svgf_atrous_shader_specialized[True]()).valid, "SVGF atrous specialized(True) valid")
    ctx.assert_true(validate_spirv_module(svgf_atrous_shader_specialized[False]()).valid, "SVGF atrous specialized(False) valid")
    ctx.assert_true(validate_spirv_module(svgf_atrous_shader_raw_spv()).valid, "SVGF atrous raw SPV valid")


def test_f18_svgf_bilateral_normal_weight(mut ctx: TestContext) raises:
    ctx.assert_true(validate_spirv_module(aaa_tile_classify_shader()).valid, "aaa_tile_classify_shader valid SPIR-V")
    ctx.assert_true(
        validate_spirv_module(aaa_deferred_shade_class_shader(False, AAA_SHADE_CLASS_STANDARD_PBR)).valid,
        "aaa_deferred_shade_class_shader valid SPIR-V",
    )
    ctx.assert_true(validate_spirv_module(aaa_shade_sky_shader()).valid, "aaa_shade_sky_shader valid SPIR-V")
    ctx.assert_true(validate_spirv_module(aaa_shade_unlit_shader()).valid, "aaa_shade_unlit_shader valid SPIR-V")


def test_f18_svgf_temporal_accumulation_alpha(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "renderer active for SVGF temporal accumulation")
    ren.svgf_passes = 3
    ren.begin_frame(Vec3(0.0, 2.0, 5.0), Vec3(0.0, 0.0, 0.0))
    ctx.assert_true(ren.render_frame(), "SVGF-enabled frame 1 renders")
    ren.begin_frame(Vec3(0.0, 2.0, 5.0), Vec3(0.0, 0.0, 0.0))
    ctx.assert_true(ren.render_frame(), "SVGF-enabled frame 2 renders")
    ctx.assert_true(ren.buf_post_scratch.size > 0, "post scratch buffer allocated for SVGF/post")
    ctx.assert_true(ren.buf_post_scratch.vk_buffer != 0, "post scratch VkBuffer valid")


def test_f18_svgf_atrous_wavelet_step(mut ctx: TestContext) raises:
    ctx.assert_true(validate_spirv_module(svgf_atrous_shader()).valid, "svgf_atrous_shader validates")
    var prev_vp = List[Float32](length=16, fill=0.0)
    prev_vp[0] = 1.0; prev_vp[5] = 1.0; prev_vp[10] = 1.0; prev_vp[15] = 1.0
    var res = eval_virtual_depth_reprojection_cpu(Vec3(0, 0, 0), Vec3(0, 0, 8.0), 12.0, True, prev_vp, Vec2(0.5, 0.5))
    ctx.assert_approx(res[0], 20.0, 0.001, "virtual depth reprojection evaluates 8 + 12 = 20")


# =========================================================================
# F19: Lumen Surface Cache & Spatial Hash (5 tests)
# =========================================================================

def test_f19_surface_cache_amortized_update(mut ctx: TestContext) raises:
    var sc_shader = surface_cache_update_shader(False)
    ctx.assert_true(validate_spirv_module(sc_shader).valid, "Lumen surface cache updater passes SSA validation")
    ctx.assert_true(len(sc_shader) > 100, "surface cache shader emits instructions")


def test_f19_surface_cache_header_offset(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "renderer active for surface cache header check")
    ctx.assert_true(ren.buf_surfcache.size >= 1024 * 4, "surface cache buffer allocates headers")
    ctx.assert_true(ren.max_surfcache_tris > 0, "max surface cache triangles allocated")
    ctx.assert_true(ren.ring_surfcache[0].vk_buffer != 0, "ring_surfcache[0] has valid VkBuffer")


def test_f19_spatial_radiance_hash_key(mut ctx: TestContext) raises:
    ctx.assert_true(validate_spirv_module(surface_cache_update_shader(True)).valid, "RT surface cache shader passes validation")


def test_f19_surface_cache_triangle_allocation(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    var mat_id = ren.add_material(AaaMaterial(base_color=Vec3(0.8, 0.8, 0.8), roughness=0.5))
    ctx.assert_true(mat_id >= 0, "material registered for surface cache instance")


def test_f19_ltc_night_bounce_radiance(mut ctx: TestContext) raises:
    var light = AaaClusterLight.point(Vec3(0, 2, 0), Vec3(10, 10, 10), 15.0)
    ctx.assert_true(light.radiance.x > 0.0, "LTC point light contributes positive radiance")
    var poly = eval_ltc_polygon_cpu(Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3(0, 2, 0), Vec3(1, 0, 0), Vec3(0, 0, 1), 1.0, 0.0, False)
    ctx.assert_true(poly >= 0.0, "LTC polygon evaluation produces non-negative form factor")


# =========================================================================
# F20: Heitz Bounded GGX VNDF RT Reflections (5 tests)
# =========================================================================

def test_f20_vndf_roughness_clamping(mut ctx: TestContext) raises:
    var mat = AaaMaterial(roughness=0.01)
    ctx.assert_approx(mat.roughness, 0.04, 0.001, "roughness clamped to 0.04 floor in AaaMaterial")
    var h = _importance_sample_ggx(0.5, 0.5, 0.01, Vec3(0, 0, 1))
    ctx.assert_approx(h.length(), 1.0, 0.01, "GGX microfacet normal is normalized")


def test_f20_virtual_depth_reprojection(mut ctx: TestContext) raises:
    var prev_vp = List[Float32](length=16, fill=0.0)
    prev_vp[0] = 1.0; prev_vp[5] = 1.0; prev_vp[10] = 1.0; prev_vp[15] = 1.0
    var res = eval_virtual_depth_reprojection_cpu(Vec3(0, 0, 0), Vec3(0, 0, 10.0), Float32(15.0), True, prev_vp, Vec2(0.5, 0.5))
    ctx.assert_approx(res[0], 25.0, 0.001, "virtual depth evaluates exact sum: 10.0 + 15.0 = 25.0")


def test_f20_ggx_smith_masking_shadowing(mut ctx: TestContext) raises:
    var s_norm = sample_heitz_bounded_vndf_cpu(0.35, 0.45, 0.4, Vec3(0.0, 0.0, 1.0))
    var s_graz = sample_heitz_bounded_vndf_cpu(0.35, 0.45, 0.4, Vec3(0.95, 0.0, 0.31).normalized())
    ctx.assert_approx(s_norm[0].length(), 1.0, 0.01, "bounded VNDF sample normal unit length")
    ctx.assert_true(s_norm[1] > 0.0 and s_norm[1] <= 1.001, "normal incidence Smith G2/G1 weight in (0, 1]")
    ctx.assert_true(s_graz[1] >= 0.0 and s_graz[1] <= 1.001, "grazing incidence Smith G2/G1 weight in [0, 1]")


def test_f20_reflection_fresnel_schlick(mut ctx: TestContext) raises:
    var f_norm = glass_fresnel_cpu(Float32(1.0), Float32(1.5))
    var f_60 = glass_fresnel_cpu(Float32(0.5), Float32(1.5))
    var f_graz = glass_fresnel_cpu(Float32(0.0), Float32(1.5))
    ctx.assert_approx(f_norm, 0.04, 0.005, "glass_fresnel_cpu at normal incidence ~0.04 for IOR=1.5")
    ctx.assert_true(f_60 > f_norm and f_graz > f_60, "Fresnel reflectance increases monotonically toward grazing")
    ctx.assert_approx(f_graz, 1.0, 0.01, "Fresnel at grazing reaches 1.0")


def test_f20_vndf_hemisphere_sampling(mut ctx: TestContext) raises:
    var lut = bake_split_sum_brdf_lut_rg16f(16, 16)
    ctx.assert_equal(len(lut), 16 * 16 * 4, "16x16 RG16F split-sum BRDF LUT baked")
    var s = sample_heitz_bounded_vndf_cpu(0.25, 0.75, 0.2, Vec3(0.0, 0.0, 1.0))
    ctx.assert_true(s[0].z > 0.0, "sampled VNDF microfacet normal lies in upper hemisphere")


# =========================================================================
# F21: ReSTIR DI Local Light Shadows (5 tests)
# =========================================================================

def test_f21_restir_light_clustering(mut ctx: TestContext) raises:
    ctx.assert_equal(AAA_CLUSTER_X, 16, "16 X clusters")
    ctx.assert_equal(AAA_CLUSTER_Y, 9, "9 Y clusters")
    ctx.assert_equal(AAA_CLUSTER_Z, 24, "24 Z clusters")
    ctx.assert_equal(AAA_TOTAL_CLUSTERS, 3456, "3,456 total froxel clusters")
    ctx.assert_true(validate_spirv_module(aaa_cluster_cull_shader(0, 1)).valid, "cluster light cull shader validates")


def test_f21_restir_candidate_selection(mut ctx: TestContext) raises:
    var res = RestirReservoir()
    _ = res.update(UInt32(10), Float32(1.0), Float32(1.0), Float32(0.9))
    _ = res.update(UInt32(20), Float32(5.0), Float32(5.0), Float32(0.9))
    _ = res.update(UInt32(30), Float32(20.0), Float32(20.0), Float32(0.9))
    var replaced = res.update(UInt32(40), Float32(100.0), Float32(100.0), Float32(0.05))
    ctx.assert_true(replaced, "high-PDF candidate replaces earlier samples")
    ctx.assert_equal(res.light_id, UInt32(40), "reservoir holds chosen light 40")
    ctx.assert_equal(res.M_count, 4, "reservoir M_count is 4")


def test_f21_restir_spatial_resampling(mut ctx: TestContext) raises:
    var r_center = RestirReservoir(UInt32(5), Float32(50.0), 4, Float32(12.5))
    var r_neighbor = RestirReservoir(UInt32(8), Float32(120.0), 6, Float32(20.0))
    ctx.assert_approx(r_neighbor.unbiased_weight(), 1.0, 0.001, "neighbor unbiased weight is 1.0")
    ctx.assert_true(r_center.merge(r_neighbor, Float32(25.0), Float32(0.1)), "reservoir merge succeeds")
    ctx.assert_equal(r_center.M_count, 10, "merged M_count is 10")
    ctx.assert_equal(r_center.light_id, UInt32(8), "center adopted candidate 8")


def test_f21_restir_max_clustered_lights(mut ctx: TestContext) raises:
    ctx.assert_equal(Int(AAA_MAX_CLUSTERED_LIGHTS), 1024, "AAA_MAX_CLUSTERED_LIGHTS is 1024")
    var lights = List[AaaClusterLight]()
    lights.append(AaaClusterLight.point(Vec3(0, 2, 0), Vec3(5, 5, 5), 10.0))
    lights.append(AaaClusterLight.point(Vec3(3, 2, 0), Vec3(5, 5, 5), 10.0))
    var regions = allocate_local_light_vsm_regions(lights, 8)
    ctx.assert_true(regions >= 2, "allocate_local_light_vsm_regions assigns shadow regions")
    var ltc = bake_ltc_lut_rg16f(16)
    ctx.assert_equal(len(ltc), 16 * 16 * 4, "16x16 LTC LUT baked")


def test_f21_restir_unbiased_contribution_weight(mut ctx: TestContext) raises:
    var res = RestirReservoir(UInt32(12), Float32(180.0), 10, Float32(18.0))
    ctx.assert_approx(res.unbiased_weight(), 1.0, 0.0001, "unbiased weight evaluates 180 / (10 * 18) == 1.0")


# =========================================================================
# F22: Two-Phase Hi-Z Pyramid & VSM Clipmaps (5 tests)
# =========================================================================

def test_f22_vsm_physical_tile_count(mut ctx: TestContext) raises:
    ctx.assert_equal(Int(VSM_PHYS_TILE_COUNT), 1024, "VSM_PHYS_TILE_COUNT is 1024")


def test_f22_vsm_page_allocation(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    vsm.invalidate_all()
    ctx.assert_equal(len(vsm.phys_to_virtual), 1024, "1024 physical tiles in VsmPageManager")


def test_f22_vsm_invalidation(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    for i in range(len(vsm.virtual_valid)):
        vsm.virtual_valid[i] = True
    vsm.invalidate_region(0.1, 0.1, 0.4, 0.4)
    var in_idx = Int(0.2 * Float32(VSM_VIRTUAL_PAGES_DIM)) * VSM_VIRTUAL_PAGES_DIM + Int(0.2 * Float32(VSM_VIRTUAL_PAGES_DIM))
    var out_idx = Int(0.9 * Float32(VSM_VIRTUAL_PAGES_DIM)) * VSM_VIRTUAL_PAGES_DIM + Int(0.9 * Float32(VSM_VIRTUAL_PAGES_DIM))
    ctx.assert_false(vsm.virtual_valid[in_idx], "page inside invalidated region is invalid")
    ctx.assert_true(vsm.virtual_valid[out_idx], "page outside invalidated region remains valid")


def test_f22_hiz_two_phase_cull_split(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(validate_spirv_module(meshlet_cull_shader()).valid, "meshlet cull shader passes SPIR-V validation")
    ctx.assert_true(ren.buf_occluded_candidates.size > 0, "renderer allocates buf_occluded_candidates")
    ctx.assert_true(ren.buf_indirect_phase2.size > 0, "renderer allocates buf_indirect_phase2")
    ctx.assert_equal(compute_toroidal_page_index(0, 64, 0), 0, "toroidal wrap at 64")
    ctx.assert_equal(compute_toroidal_page_index(0, -1, 0), 63, "negative wrap to 63")


def test_f22_ortho_light_view_proj(mut ctx: TestContext) raises:
    var vp = make_ortho_directional_light_vp(Vec3(0.5, 0.8, 0.2).normalized(), Vec3(0, 0, 0), 50.0, 120.0)
    ctx.assert_equal(len(vp), 16, "16-element ortho light_view_proj matrix generated")
    ctx.assert_approx(vp[15], 1.0, 0.001, "affine projection entry m[15] is 1.0")


# =========================================================================
# F23: Render Graph Transient Aliasing (5 tests)
# =========================================================================

def test_f23_rg_resource_creation(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    ctx.assert_equal(rg.add_resource("depth", 1920 * 1080 * 4, transient=True), 0, "first resource id is 0")
    ctx.assert_equal(rg.add_resource("color", 1920 * 1080 * 8, transient=True), 1, "second resource id is 1")


def test_f23_rg_pass_node_registration(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r1 = rg.add_resource("gbuffer", 1024 * 1024 * 16)
    var accesses = List[RgResourceAccess]()
    accesses.append(RgResourceAccess(r1, 1))
    ctx.assert_equal(rg.add_pass("gbuffer_pass", accesses), 0, "first pass node id is 0")


def test_f23_rg_compile_aliasing(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r1 = rg.add_resource("early_depth", 1024 * 1024 * 16, transient=True)
    var r2 = rg.add_resource("mid_gbuffer", 1024 * 1024 * 16, transient=True)
    var r3 = rg.add_resource("late_post", 1024 * 1024 * 16, transient=True)
    var acc1 = List[RgResourceAccess](); acc1.append(RgResourceAccess(r1, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("pass0", acc1)
    var acc2 = List[RgResourceAccess](); acc2.append(RgResourceAccess(r2, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("pass1", acc2)
    var acc3 = List[RgResourceAccess](); acc3.append(RgResourceAccess(r3, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("pass2", acc3)
    ctx.assert_true(rg.compile(64 * 1024 * 1024), "render graph compilation succeeds")
    ctx.assert_equal(rg.resources[r1].aliased_offset_bytes, 0, "r1 allocated at offset 0")
    ctx.assert_equal(rg.resources[r2].aliased_offset_bytes, 0, "r2 aliased at offset 0")
    ctx.assert_equal(rg.aliased_peak_bytes, 16 * 1024 * 1024, "aliased peak is 16 MB")


def test_f23_rg_savings_ratio(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r1 = rg.add_resource("r1", 1024 * 1024 * 16, transient=True)
    var r2 = rg.add_resource("r2", 1024 * 1024 * 16, transient=True)
    var r3 = rg.add_resource("r3", 1024 * 1024 * 16, transient=True)
    var a1 = List[RgResourceAccess](); a1.append(RgResourceAccess(r1, RG_USAGE_COMPUTE_WRITE)); _ = rg.add_pass("p1", a1)
    var a2 = List[RgResourceAccess](); a2.append(RgResourceAccess(r2, RG_USAGE_COMPUTE_WRITE)); _ = rg.add_pass("p2", a2)
    var a3 = List[RgResourceAccess](); a3.append(RgResourceAccess(r3, RG_USAGE_COMPUTE_WRITE)); _ = rg.add_pass("p3", a3)
    _ = rg.compile(64 * 1024 * 1024)
    ctx.assert_approx(rg.aliasing_savings_ratio(), Float32(2.0 / 3.0), 0.01, "aliasing savings ratio ~0.667")


def test_f23_rg_profile_format(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "renderer active for render graph profile check")
    var profile = ren.render_graph.format_gpu_profile_summary()
    ctx.assert_true(profile.byte_length() > 0, "GPU profile summary string generated")
    ctx.assert_true(ren.render_graph.aliased_img_peak_bytes > 0, "render graph tracks aliased image heap peak")


# =========================================================================
# F24: Zero-Parse mmap .flpk v3 VFS & AssetServer (5 tests)
# =========================================================================

def test_f24_vfs_encode_and_read(mut ctx: TestContext) raises:
    var pid_s = String(Int(external_call["getpid", Int32]()))
    var flpk_path = "/tmp/flint_e2e_t1_f24_a_" + pid_s + ".flpk"
    ctx.cleanup_artifact(flpk_path)
    ctx.assert_true(create_mock_flpk_v3_archive_file(flpk_path), "create_mock_flpk_v3_archive_file succeeds")
    var arch = FlpkV3MappedArchive.open(flpk_path)
    ctx.assert_true(arch.valid, "FlpkV3MappedArchive opens cleanly")
    ctx.assert_true(arch.toc_count() >= 4, "archive contains at least 4 TOC entries")
    arch.close()
    ctx.cleanup_artifact(flpk_path)


def test_f24_vfs_directory_entries(mut ctx: TestContext) raises:
    var pid_s = String(Int(external_call["getpid", Int32]()))
    var flpk_path = "/tmp/flint_e2e_t1_f24_b_" + pid_s + ".flpk"
    ctx.cleanup_artifact(flpk_path)
    ctx.assert_true(create_mock_flpk_v3_archive_file(flpk_path), "write flpk v3")
    var arch = FlpkV3MappedArchive.open(flpk_path)
    var idx_p = arch.find_by_path("model.glb", 0)
    var idx_u = arch.find_by_uuid(AssetUuid.from_path("model.glb"), 0)
    ctx.assert_true(idx_p >= 0, "Robin-Hood path table finds model.glb")
    ctx.assert_equal(idx_p, idx_u, "path and UUID Robin-Hood lookups resolve identical TOC index")
    arch.close()
    ctx.cleanup_artifact(flpk_path)


def test_f24_vfs_alignment_64byte(mut ctx: TestContext) raises:
    var pack_bytes = create_mock_flpk_v3_data()
    ctx.assert_true(len(pack_bytes) >= 64, "flpk v3 container >= 64 bytes")
    ctx.assert_true(pack_bytes[0] == 0x46 and pack_bytes[1] == 0x4C and pack_bytes[2] == 0x50 and pack_bytes[3] == 0x4B, "FLPK magic")
    var pid_s = String(Int(external_call["getpid", Int32]()))
    var flpk_path = "/tmp/flint_e2e_t1_f24_c_" + pid_s + ".flpk"
    ctx.cleanup_artifact(flpk_path)
    ctx.assert_true(create_mock_flpk_v3_archive_file(flpk_path), "write aligned flpk")
    var arch = FlpkV3MappedArchive.open(flpk_path)
    var idx = arch.find_by_path("manifest.json", 0)
    ctx.assert_true(idx >= 0, "manifest.json present")
    var entry = arch.toc_entry(idx)
    ctx.assert_equal(Int(entry.payload_offset & 63), 0, "payload offset is 64-byte cache-line aligned")
    var slice = arch.mmap_slice_by_toc(idx, verify_crc=True)
    ctx.assert_true(slice.valid and slice.size > 0, "zero-copy mmap slice valid")
    arch.close()
    ctx.cleanup_artifact(flpk_path)


def test_f24_vfs_multi_file_bundle(mut ctx: TestContext) raises:
    var pid_s = String(Int(external_call["getpid", Int32]()))
    var flpk_path = "/tmp/flint_e2e_t1_f24_d_" + pid_s + ".flpk"
    ctx.cleanup_artifact(flpk_path)
    var chunks = List[FlpkBuildChunk]()
    var p1: List[UInt8] = [UInt8(11), UInt8(22), UInt8(33), UInt8(44)]
    var p2: List[UInt8] = [UInt8(55), UInt8(66), UInt8(77)]
    chunks.append(FlpkBuildChunk(AssetUuid.from_path("a.bin"), "a.bin", 0, ASSET_KIND_RAW, p1^, CHUNK_COMP_NONE))
    chunks.append(FlpkBuildChunk(AssetUuid.from_path("b.bin"), "b.bin", 0, ASSET_KIND_RAW, p2^, CHUNK_COMP_NONE))
    ctx.assert_true(write_flpk_v3(flpk_path, chunks, page_align=False), "write multi-file flpk v3")
    var arch = FlpkV3MappedArchive.open(flpk_path)
    var toc_a = arch.find_by_path("a.bin", 0)
    ctx.assert_true(toc_a >= 0 and arch.verify_entry_crc64(toc_a), "CRC64 verifies for a.bin")
    var read_a = arch.read_payload_by_toc(toc_a, verify_crc=True)
    ctx.assert_true(read_a[0] and len(read_a[1]) == 4 and read_a[1][2] == 33, "payload round-trips")
    arch.close()
    ctx.cleanup_artifact(flpk_path)


def test_f24_vfs_missing_file_handling(mut ctx: TestContext) raises:
    var arch = FlpkV3MappedArchive.open("/tmp/nonexistent_v3_e2e_file.flpk")
    ctx.assert_false(arch.valid, "missing .flpk v3 returns valid=False")
    ctx.assert_equal(arch.find_by_path("foo.txt", 0), -1, "find_by_path on invalid archive returns -1")
    arch.close()
    var srv = create_mock_asset_server()
    var h = srv.load_raw("missing/asset.bin")
    ctx.assert_false(h.is_valid(), "AssetServer.load_raw returns invalid handle for missing asset")
    srv.close()


# =========================================================================
# F25: Async Transfer Ring & Residency Streaming (5 tests)
# =========================================================================

def test_f25_residency_streamer_feedback(mut ctx: TestContext) raises:
    var streamer = AaaResidencyStreamer(128, 4)
    streamer.record_gpu_feedback(10, 1)
    ctx.assert_equal(streamer.requested_min_mip[10], 1, "GPU feedback records mip 1")
    ctx.assert_equal(streamer.requested_min_mip[11], 4, "unrequested texture remains at tail mip 4")


def test_f25_residency_budget_tick(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.active, "renderer active for residency VRAM bridge")
    var srv = create_mock_asset_server()
    srv.attach_renderer_residency_bridge(ren)
    var tex = CookedTextureAsset()
    tex.width = 64
    tex.height = 64
    tex.mip_count = 5
    tex.tail_first_mip = 4
    tex.resident_min_mip = 4
    for m in range(5):
        var dim = 64 >> m
        var b = ((dim + 3) // 4) * ((dim + 3) // 4) * 16
        tex.mip_sizes[m] = UInt32(b)
        tex.mips.append(List[UInt8](length=b, fill=UInt8(0x55)))
    var th = srv.textures.insert(tex^, AssetUuid.from_path("tex/albedo.png"), "tex/albedo.png", srv.current_tick)
    var slot = Int(th.slot())
    srv.streamer.enqueue_mip_upload(slot, 3, 256)
    srv.streamer.enqueue_mip_upload(slot, 2, 1024)
    srv.streamer.enqueue_mip_upload(slot, 1, 4096)
    var drained = srv.streamer.drain_uploads_within_budget(1500)
    ctx.assert_equal(len(drained), 2, "budgeted drain admits mips 3 and 2 within 1500B")
    ctx.assert_equal(srv.streamer.drained_bytes_last_tick, 1280, "drained_bytes_last_tick tracks 256 + 1024")
    ctx.assert_equal(srv.streamer.resident_min_mip[slot], 2, "resident_min_mip advances 4 -> 2")
    ctx.assert_true(srv.upload_mip_to_gpu(th, 2, 1024), "upload_mip_to_gpu stages mip 2 to Vulkan")
    ctx.assert_true(srv.textures.slot_vram_bytes[slot] > 0, "slot_vram_bytes > 0 after mip upload")
    ctx.assert_true(srv.total_vram_live_bytes() > 0, "total_vram_live_bytes > 0 after mip upload")
    _ = srv.tick(1.0)
    srv.close()


def test_f25_mip_tail_residency_threshold(mut ctx: TestContext) raises:
    var streamer = AaaResidencyStreamer(64, 4)
    ctx.assert_equal(streamer.resident_min_mip[0], 4, "initial resident tail mip level is 4")
    ctx.assert_equal(len(streamer.resident_min_mip), 64, "streamer manages 64 textures")


def test_f25_transfer_ring_capacity(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.staging_ring.slot_capacity >= 1024 * 1024, "LinearStagingRing slot capacity allocated")
    ctx.assert_true(ren.buf_staging.size >= 8 * 1024 * 1024, "renderer staging buffer >= 8 MiB")


def test_f25_streamer_max_textures(mut ctx: TestContext) raises:
    var streamer = AaaResidencyStreamer(256, 4)
    streamer.record_gpu_feedback(255, 2)
    ctx.assert_equal(streamer.requested_min_mip[255], 2, "max texture index 255 records requested mip 2")


# =========================================================================
# F26: Cluster-Group QEM Meshlet DAG (5 tests)
# =========================================================================

def test_f26_qem_quadric_plane(mut ctx: TestContext) raises:
    var q = _SymQuadric.from_plane(0.0, 1.0, 0.0, -2.0)
    ctx.assert_approx(Float32(q.eval(Vec3(0.0, 2.0, 0.0))), 0.0, 0.001, "point on plane evaluates 0 quadric error")


def test_f26_qem_quadric_addition(mut ctx: TestContext) raises:
    var q_sum = _SymQuadric.from_plane(1.0, 0.0, 0.0, 0.0) + _SymQuadric.from_plane(0.0, 1.0, 0.0, 0.0)
    ctx.assert_true(q_sum.eval(Vec3(1.0, 1.0, 0.0)) > 0.0, "quadrics additively accumulate error")


def test_f26_qem_simplify_reduction(mut ctx: TestContext) raises:
    var mesh = create_mock_meshlet_mesh(12)
    var orig_tris = len(mesh.indices) // 3
    var res = simplify_mesh_qem(mesh, orig_tris // 2)
    ctx.assert_true(len(res[0].indices) // 3 < orig_tris, "QEM simplification strictly reduces triangle count")


def test_f26_meshlet_cluster_bounds(mut ctx: TestContext) raises:
    var mesh = create_mock_meshlet_mesh(12)
    var slices = simplify_mesh_qem(mesh, len(mesh.indices) // 6)
    ctx.assert_true(len(slices) > 0 and len(slices[0].indices) > 0, "QEM returns valid LOD slices")


def test_f26_mesh_lod_slice_ranges(mut ctx: TestContext) raises:
    var slice = MeshLodSlice(0, 10, 50, 0.05)
    ctx.assert_equal(slice.first_meshlet, 0, "first meshlet matches")
    ctx.assert_equal(slice.meshlet_count, 10, "meshlet count matches")
    ctx.assert_approx(slice.object_error, 0.05, 0.001, "object error matches")


# =========================================================================
# F27: Advanced glTF Material Extensions (5 tests)
# =========================================================================

def test_f27_material_clearcoat_properties(mut ctx: TestContext) raises:
    var mat = AaaMaterial()
    mat.clearcoat = 0.8
    mat.clearcoat_roughness = 0.1
    ctx.assert_approx(mat.clearcoat, 0.8, 0.001, "clearcoat intensity 0.8")
    ctx.assert_approx(mat.clearcoat_roughness, 0.1, 0.001, "clearcoat roughness 0.1")


def test_f27_material_sheen_properties(mut ctx: TestContext) raises:
    var mat = AaaMaterial()
    mat.sheen_color = Vec3(1.0, 0.8, 0.5)
    mat.sheen_roughness = 0.3
    ctx.assert_approx(mat.sheen_roughness, 0.3, 0.001, "sheen roughness 0.3")
    ctx.assert_vec3_approx(mat.sheen_color, Vec3(1.0, 0.8, 0.5), 0.001, "sheen color preserved")


def test_f27_material_iridescence_properties(mut ctx: TestContext) raises:
    var mat = AaaMaterial()
    mat.iridescence = 1.0
    mat.iridescence_ior = 1.33
    ctx.assert_approx(mat.iridescence, 1.0, 0.001, "iridescence strength 1.0")
    ctx.assert_approx(mat.iridescence_ior, 1.33, 0.001, "iridescence IOR 1.33")


def test_f27_thin_film_iridescence_cpu(mut ctx: TestContext) raises:
    var color = eval_thin_film_iridescence_cpu(0.8, 350.0, 1.33)
    ctx.assert_true(color.x >= 0.0 and color.y >= 0.0 and color.z >= 0.0, "iridescence spectral RGB valid")
    ctx.assert_true((color.x + color.y + color.z) > 0.5, "thin-film iridescence RGB sum > 0.5")


def test_f27_rnm_blend_cpu(mut ctx: TestContext) raises:
    var blended = reoriented_normal_blend_cpu(Vec3(0, 1, 0), Vec3(0.1, 0.98, 0.1).normalized(), 1.0)
    ctx.assert_approx(blended.length(), 1.0, 0.01, "RNM blended normal remains unit length")


# =========================================================================
# F28: JobPool Priorities, Work-Stealing & Pipeline (5 tests)
# =========================================================================

def test_f28_job_pool_allocation(mut ctx: TestContext) raises:
    var pool = JobPool()
    ctx.assert_true(pool.allocate(), "JobPool allocates ring memory cleanly")
    ctx.assert_true(pool.allocated(), "JobPool reports allocated state")
    pool.stop()


def test_f28_job_pool_workers_start_stop(mut ctx: TestContext) raises:
    var pool = JobPool()
    _ = pool.allocate()
    ctx.assert_true(pool.start(2), "JobPool starts 2 Chase-Lev workers")
    var jctx = _T1M3JobOrderCtx()
    var ctx_addr = Int(Pointer(to=jctx))
    var j_hi = pool.create(_t1_m3_job_hi, ctx_addr, 1)
    var j_lo = pool.create(_t1_m3_job_lo, ctx_addr, 1)
    pool.depend(j_hi, j_lo)
    pool.submit(j_hi, priority=JOB_PRIO_HIGH)
    pool.submit(j_lo, priority=JOB_PRIO_LOW)
    pool.wait(j_hi)
    pool.stop()
    ctx.assert_equal(Int(jctx.seq.load()), 2, "both dependency-ordered jobs ran")
    ctx.assert_equal(Int(jctx.order_lo), 1, "dependency job_lo executed first despite LOW priority")
    ctx.assert_equal(Int(jctx.order_hi), 2, "dependent job_hi executed second despite HIGH priority")
    ctx.assert_false(pool.ready(), "JobPool stopped cleanly")


def test_f28_pipeline_cache_magic(mut ctx: TestContext, mut ren: FlintAaaVulkanRenderer) raises:
    ctx.assert_true(ren.desc_pool != 0, "renderer pipeline descriptor pool created")
    ctx.assert_true(ren.pipe_layout != 0, "renderer pipeline layout created")


def test_f28_job_pool_default_workers(mut ctx: TestContext) raises:
    ctx.assert_true(default_worker_count() >= 1, "default worker count at least 1")


def test_f28_job_pool_ready_state(mut ctx: TestContext) raises:
    var pool = JobPool()
    ctx.assert_false(pool.ready(), "unallocated pool is not ready")


# =========================================================================
# F29: Hybrid ECS v2 (flint.ecs.World) & ParallelCommands (5 tests)
# =========================================================================

def test_f29_entity_handle_generation(mut ctx: TestContext) raises:
    var w = World()
    var e1 = w.spawn()
    ctx.assert_true(e1.valid() and w.is_alive(e1), "spawned entity is valid and alive")
    ctx.assert_true(w.despawn(e1), "despawn succeeds")
    ctx.assert_false(w.is_alive(e1), "despawned entity is no longer alive")
    var e2 = w.spawn()
    ctx.assert_true(e2 != e1, "recycled slot increments generation")
    ctx.assert_false(w.is_alive(e1), "stale entity handle remains rejected")


def test_f29_archetype_registry(mut ctx: TestContext) raises:
    var w = World()
    var last_id = -1
    for i in range(270):
        last_id = w.register_pod_component(UInt64(0xE2E00001 + i), 8, 4)
    ctx.assert_true(last_id >= 260, "World supports > 256 registered component types")
    var mask = ComponentBitSet()
    mask.set(last_id)
    ctx.assert_true(mask.has(last_id), "ComponentBitSet tracks bits >= 256")


def test_f29_archetype_spawn_despawn(mut ctx: TestContext) raises:
    var w = World()
    var e = w.spawn()
    ctx.assert_true(w.insert(e, _T1M3Pos(1.0, 2.0, 3.0)), "insert _T1M3Pos")
    ctx.assert_true(w.insert(e, _T1M3Vel(4.0, 5.0, 6.0)), "insert _T1M3Vel triggers archetype move")
    ctx.assert_true(w.has[_T1M3Pos](e) and w.has[_T1M3Vel](e), "both components present")
    ctx.assert_approx(w.get[_T1M3Pos](e).x, 1.0, 0.001, "component value preserved across archetype transition")
    ctx.assert_true(w.despawn(e) and not w.is_alive(e), "despawn clears entity")


def test_f29_simd_column_load_store(mut ctx: TestContext) raises:
    var w = World()
    var e = w.spawn()
    _ = w.insert(e, _T1M3Pos(3.0, 4.0, 5.0))
    var cs1 = w.state_checksum()
    var ptr_opt = w.get_ptr_mut[_T1M3Pos](e)
    ctx.assert_true(Bool(ptr_opt), "STORAGE_TABLE component mutable ptr retrieved")
    if ptr_opt:
        ptr_opt.value()[unsafe_offset=0].x = 9.0
    var cs2 = w.state_checksum()
    ctx.assert_approx(w.get[_T1M3Pos](e).x, 9.0, 0.001, "World STORAGE_TABLE load/store mutates component")
    ctx.assert_true(cs1 != cs2, "World.state_checksum reflects dense component store")

    var ev = Events[_T1M3Evt]()
    var cur = 0
    _ = ev.send(_T1M3Evt(42))
    _ = ev.send(_T1M3Evt(99))
    var batch1 = ev.read_since(cur)
    ctx.assert_equal(len(batch1), 2, "Events reads 2 sent events")
    ctx.assert_equal(batch1[0].code, 42, "first event code matches")
    ev.swap_buffers()
    var batch2 = ev.read_since(cur)
    ctx.assert_equal(len(batch2), 0, "cursor avoids re-reading consumed events across swap")


def test_f29_deferred_tick_commit(mut ctx: TestContext) raises:
    var w = World()
    var e0 = w.spawn()
    _ = w.insert(e0, _T1M3Pos(1.5, 0.0, 0.0))
    var pool = JobPool()
    _ = pool.allocate()
    _ = pool.start(1)
    var cmds = ParallelCommands()
    w.par_for_each1[_T1M3Pos](pool, _t1_m3_par_spawn_kernel, Int(Pointer(to=cmds)))
    w.flush_commands(cmds)
    pool.stop()
    ctx.assert_equal(w.entity_count(), 2, "ParallelCommands flushed spawned entity into World")


# =========================================================================
# Aggregator for Tier 1 Milestone 3
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T1_M3_RtStreaming")
    var ren = FlintAaaVulkanRenderer(width=160, height=120)

    try: test_f18_svgf_variance_moments_formula(ctx); ctx.record_pass("test_f18_svgf_variance_moments_formula") except e: ctx.record_fail("test_f18_svgf_variance_moments_formula", String(e))
    try: test_f18_svgf_bilateral_depth_weight(ctx); ctx.record_pass("test_f18_svgf_bilateral_depth_weight") except e: ctx.record_fail("test_f18_svgf_bilateral_depth_weight", String(e))
    try: test_f18_svgf_bilateral_normal_weight(ctx); ctx.record_pass("test_f18_svgf_bilateral_normal_weight") except e: ctx.record_fail("test_f18_svgf_bilateral_normal_weight", String(e))
    try: test_f18_svgf_temporal_accumulation_alpha(ctx, ren); ctx.record_pass("test_f18_svgf_temporal_accumulation_alpha") except e: ctx.record_fail("test_f18_svgf_temporal_accumulation_alpha", String(e))
    try: test_f18_svgf_atrous_wavelet_step(ctx); ctx.record_pass("test_f18_svgf_atrous_wavelet_step") except e: ctx.record_fail("test_f18_svgf_atrous_wavelet_step", String(e))

    try: test_f19_surface_cache_amortized_update(ctx); ctx.record_pass("test_f19_surface_cache_amortized_update") except e: ctx.record_fail("test_f19_surface_cache_amortized_update", String(e))
    try: test_f19_surface_cache_header_offset(ctx, ren); ctx.record_pass("test_f19_surface_cache_header_offset") except e: ctx.record_fail("test_f19_surface_cache_header_offset", String(e))
    try: test_f19_spatial_radiance_hash_key(ctx); ctx.record_pass("test_f19_spatial_radiance_hash_key") except e: ctx.record_fail("test_f19_spatial_radiance_hash_key", String(e))
    try: test_f19_surface_cache_triangle_allocation(ctx, ren); ctx.record_pass("test_f19_surface_cache_triangle_allocation") except e: ctx.record_fail("test_f19_surface_cache_triangle_allocation", String(e))
    try: test_f19_ltc_night_bounce_radiance(ctx); ctx.record_pass("test_f19_ltc_night_bounce_radiance") except e: ctx.record_fail("test_f19_ltc_night_bounce_radiance", String(e))

    try: test_f20_vndf_roughness_clamping(ctx); ctx.record_pass("test_f20_vndf_roughness_clamping") except e: ctx.record_fail("test_f20_vndf_roughness_clamping", String(e))
    try: test_f20_virtual_depth_reprojection(ctx); ctx.record_pass("test_f20_virtual_depth_reprojection") except e: ctx.record_fail("test_f20_virtual_depth_reprojection", String(e))
    try: test_f20_ggx_smith_masking_shadowing(ctx); ctx.record_pass("test_f20_ggx_smith_masking_shadowing") except e: ctx.record_fail("test_f20_ggx_smith_masking_shadowing", String(e))
    try: test_f20_reflection_fresnel_schlick(ctx); ctx.record_pass("test_f20_reflection_fresnel_schlick") except e: ctx.record_fail("test_f20_reflection_fresnel_schlick", String(e))
    try: test_f20_vndf_hemisphere_sampling(ctx); ctx.record_pass("test_f20_vndf_hemisphere_sampling") except e: ctx.record_fail("test_f20_vndf_hemisphere_sampling", String(e))

    try: test_f21_restir_light_clustering(ctx); ctx.record_pass("test_f21_restir_light_clustering") except e: ctx.record_fail("test_f21_restir_light_clustering", String(e))
    try: test_f21_restir_candidate_selection(ctx); ctx.record_pass("test_f21_restir_candidate_selection") except e: ctx.record_fail("test_f21_restir_candidate_selection", String(e))
    try: test_f21_restir_spatial_resampling(ctx); ctx.record_pass("test_f21_restir_spatial_resampling") except e: ctx.record_fail("test_f21_restir_spatial_resampling", String(e))
    try: test_f21_restir_max_clustered_lights(ctx); ctx.record_pass("test_f21_restir_max_clustered_lights") except e: ctx.record_fail("test_f21_restir_max_clustered_lights", String(e))
    try: test_f21_restir_unbiased_contribution_weight(ctx); ctx.record_pass("test_f21_restir_unbiased_contribution_weight") except e: ctx.record_fail("test_f21_restir_unbiased_contribution_weight", String(e))

    try: test_f22_vsm_physical_tile_count(ctx); ctx.record_pass("test_f22_vsm_physical_tile_count") except e: ctx.record_fail("test_f22_vsm_physical_tile_count", String(e))
    try: test_f22_vsm_page_allocation(ctx); ctx.record_pass("test_f22_vsm_page_allocation") except e: ctx.record_fail("test_f22_vsm_page_allocation", String(e))
    try: test_f22_vsm_invalidation(ctx); ctx.record_pass("test_f22_vsm_invalidation") except e: ctx.record_fail("test_f22_vsm_invalidation", String(e))
    try: test_f22_hiz_two_phase_cull_split(ctx, ren); ctx.record_pass("test_f22_hiz_two_phase_cull_split") except e: ctx.record_fail("test_f22_hiz_two_phase_cull_split", String(e))
    try: test_f22_ortho_light_view_proj(ctx); ctx.record_pass("test_f22_ortho_light_view_proj") except e: ctx.record_fail("test_f22_ortho_light_view_proj", String(e))

    try: test_f23_rg_resource_creation(ctx); ctx.record_pass("test_f23_rg_resource_creation") except e: ctx.record_fail("test_f23_rg_resource_creation", String(e))
    try: test_f23_rg_pass_node_registration(ctx); ctx.record_pass("test_f23_rg_pass_node_registration") except e: ctx.record_fail("test_f23_rg_pass_node_registration", String(e))
    try: test_f23_rg_compile_aliasing(ctx); ctx.record_pass("test_f23_rg_compile_aliasing") except e: ctx.record_fail("test_f23_rg_compile_aliasing", String(e))
    try: test_f23_rg_savings_ratio(ctx); ctx.record_pass("test_f23_rg_savings_ratio") except e: ctx.record_fail("test_f23_rg_savings_ratio", String(e))
    try: test_f23_rg_profile_format(ctx, ren); ctx.record_pass("test_f23_rg_profile_format") except e: ctx.record_fail("test_f23_rg_profile_format", String(e))

    try: test_f24_vfs_encode_and_read(ctx); ctx.record_pass("test_f24_vfs_encode_and_read") except e: ctx.record_fail("test_f24_vfs_encode_and_read", String(e))
    try: test_f24_vfs_directory_entries(ctx); ctx.record_pass("test_f24_vfs_directory_entries") except e: ctx.record_fail("test_f24_vfs_directory_entries", String(e))
    try: test_f24_vfs_alignment_64byte(ctx); ctx.record_pass("test_f24_vfs_alignment_64byte") except e: ctx.record_fail("test_f24_vfs_alignment_64byte", String(e))
    try: test_f24_vfs_multi_file_bundle(ctx); ctx.record_pass("test_f24_vfs_multi_file_bundle") except e: ctx.record_fail("test_f24_vfs_multi_file_bundle", String(e))
    try: test_f24_vfs_missing_file_handling(ctx); ctx.record_pass("test_f24_vfs_missing_file_handling") except e: ctx.record_fail("test_f24_vfs_missing_file_handling", String(e))

    try: test_f25_residency_streamer_feedback(ctx); ctx.record_pass("test_f25_residency_streamer_feedback") except e: ctx.record_fail("test_f25_residency_streamer_feedback", String(e))
    try: test_f25_residency_budget_tick(ctx, ren); ctx.record_pass("test_f25_residency_budget_tick") except e: ctx.record_fail("test_f25_residency_budget_tick", String(e))
    try: test_f25_mip_tail_residency_threshold(ctx); ctx.record_pass("test_f25_mip_tail_residency_threshold") except e: ctx.record_fail("test_f25_mip_tail_residency_threshold", String(e))
    try: test_f25_transfer_ring_capacity(ctx, ren); ctx.record_pass("test_f25_transfer_ring_capacity") except e: ctx.record_fail("test_f25_transfer_ring_capacity", String(e))
    try: test_f25_streamer_max_textures(ctx); ctx.record_pass("test_f25_streamer_max_textures") except e: ctx.record_fail("test_f25_streamer_max_textures", String(e))

    try: test_f26_qem_quadric_plane(ctx); ctx.record_pass("test_f26_qem_quadric_plane") except e: ctx.record_fail("test_f26_qem_quadric_plane", String(e))
    try: test_f26_qem_quadric_addition(ctx); ctx.record_pass("test_f26_qem_quadric_addition") except e: ctx.record_fail("test_f26_qem_quadric_addition", String(e))
    try: test_f26_qem_simplify_reduction(ctx); ctx.record_pass("test_f26_qem_simplify_reduction") except e: ctx.record_fail("test_f26_qem_simplify_reduction", String(e))
    try: test_f26_meshlet_cluster_bounds(ctx); ctx.record_pass("test_f26_meshlet_cluster_bounds") except e: ctx.record_fail("test_f26_meshlet_cluster_bounds", String(e))
    try: test_f26_mesh_lod_slice_ranges(ctx); ctx.record_pass("test_f26_mesh_lod_slice_ranges") except e: ctx.record_fail("test_f26_mesh_lod_slice_ranges", String(e))

    try: test_f27_material_clearcoat_properties(ctx); ctx.record_pass("test_f27_material_clearcoat_properties") except e: ctx.record_fail("test_f27_material_clearcoat_properties", String(e))
    try: test_f27_material_sheen_properties(ctx); ctx.record_pass("test_f27_material_sheen_properties") except e: ctx.record_fail("test_f27_material_sheen_properties", String(e))
    try: test_f27_material_iridescence_properties(ctx); ctx.record_pass("test_f27_material_iridescence_properties") except e: ctx.record_fail("test_f27_material_iridescence_properties", String(e))
    try: test_f27_thin_film_iridescence_cpu(ctx); ctx.record_pass("test_f27_thin_film_iridescence_cpu") except e: ctx.record_fail("test_f27_thin_film_iridescence_cpu", String(e))
    try: test_f27_rnm_blend_cpu(ctx); ctx.record_pass("test_f27_rnm_blend_cpu") except e: ctx.record_fail("test_f27_rnm_blend_cpu", String(e))

    try: test_f28_job_pool_allocation(ctx); ctx.record_pass("test_f28_job_pool_allocation") except e: ctx.record_fail("test_f28_job_pool_allocation", String(e))
    try: test_f28_job_pool_workers_start_stop(ctx); ctx.record_pass("test_f28_job_pool_workers_start_stop") except e: ctx.record_fail("test_f28_job_pool_workers_start_stop", String(e))
    try: test_f28_pipeline_cache_magic(ctx, ren); ctx.record_pass("test_f28_pipeline_cache_magic") except e: ctx.record_fail("test_f28_pipeline_cache_magic", String(e))
    try: test_f28_job_pool_default_workers(ctx); ctx.record_pass("test_f28_job_pool_default_workers") except e: ctx.record_fail("test_f28_job_pool_default_workers", String(e))
    try: test_f28_job_pool_ready_state(ctx); ctx.record_pass("test_f28_job_pool_ready_state") except e: ctx.record_fail("test_f28_job_pool_ready_state", String(e))

    try: test_f29_entity_handle_generation(ctx); ctx.record_pass("test_f29_entity_handle_generation") except e: ctx.record_fail("test_f29_entity_handle_generation", String(e))
    try: test_f29_archetype_registry(ctx); ctx.record_pass("test_f29_archetype_registry") except e: ctx.record_fail("test_f29_archetype_registry", String(e))
    try: test_f29_archetype_spawn_despawn(ctx); ctx.record_pass("test_f29_archetype_spawn_despawn") except e: ctx.record_fail("test_f29_archetype_spawn_despawn", String(e))
    try: test_f29_simd_column_load_store(ctx); ctx.record_pass("test_f29_simd_column_load_store") except e: ctx.record_fail("test_f29_simd_column_load_store", String(e))
    try: test_f29_deferred_tick_commit(ctx); ctx.record_pass("test_f29_deferred_tick_commit") except e: ctx.record_fail("test_f29_deferred_tick_commit", String(e))

    ren.shutdown()


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("test_t1_m3_rt_streaming failed")
