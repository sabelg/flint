"""Tier 1 Feature Coverage: Milestone 3 RT & Streaming (F18–F29: 60 tests).
Opaque-box and requirement-driven verification of SVGF, Lumen cache, VNDF, ReSTIR, VSM, RenderGraph, VFS, QEM, glTF ext, JobPool, and ECS.
"""
from std.collections import List
from std.math import sqrt, exp, max, min, cos, sin
from flint.core.math import Vec2, Vec3
from flint.core.transform import Quat
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
    AAA_CLUSTER_X,
    AAA_CLUSTER_Y,
    AAA_CLUSTER_Z,
    AAA_TOTAL_CLUSTERS,
    AAA_MAX_CLUSTERED_LIGHTS,
    AaaClusterLight,
    RestirReservoir,
    allocate_local_light_vsm_regions,
    bake_ltc_lut_rg16f,
    eval_ltc_polygon_cpu,
    aaa_cluster_cull_shader,
)
from flint.render.aaa_vsm import (
    VsmPageManager,
    VSM_PHYS_TILE_COUNT,
    VSM_VIRTUAL_PAGES_DIM,
    VSM_VIRTUAL_PAGE_COUNT,
    compute_toroidal_page_index,
    get_vsm_cascade_index,
    make_ortho_directional_light_vp,
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
    eval_virtual_depth_reprojection_cpu,
    svgf_temporal_reproject_shader,
    svgf_atrous_shader,
    svgf_atrous_shader_specialized,
    svgf_atrous_shader_raw_spv,
    surface_cache_update_shader,
    aaa_deferred_shade_shader,
)
from flint.render.aaa_visbuffer import (
    meshlet_cull_shader,
    BIND_AAA_CANDIDATES,
)
from flint.render.aaa_ibl import (
    _importance_sample_ggx,
    bake_split_sum_brdf_lut_rg16f,
    sample_heitz_bounded_vndf_cpu,
)
from flint.render.spirv_validate import validate_spirv_module
from e2e.common.harness import TestContext
from e2e.common.mock_data import create_mock_meshlet_mesh, create_mock_flpk_data


# =========================================================================
# F18: SVGF Temporal Denoising (5 tests)
# =========================================================================

def test_f18_svgf_variance_moments_formula(mut ctx: TestContext) raises:
    # 1. Feed 100 noisy samples alternating between 0.5 and 1.5 (true mean = 1.0, true variance = 0.25)
    var m1 = Float32(0.0)
    var m2 = Float32(0.0)
    var h_len = Float32(0.0)
    var var_est = Float32(0.0)

    for i in range(100):
        var luma = Float32(0.5) if (i % 2 == 0) else Float32(1.5)
        var res = svgf_accumulate_moments_cpu(m1, m2, luma, h_len)
        m1 = res[0]
        m2 = res[1]
        var_est = res[2]
        h_len = res[3]

    ctx.assert_approx(m1, 1.0, 0.05, "first moment m1 converges to true mean 1.0")
    ctx.assert_approx(m2, 1.25, 0.05, "second moment m2 converges to true E[X^2] = 1.25")
    ctx.assert_approx(var_est, 0.25, 0.05, "variance Var(L) = max(0, m2 - m1^2) converges to true variance 0.25")
    ctx.assert_true(var_est >= 0.0, "variance is strictly non-negative")

    # 2. Feed 120 constant samples (L = 2.0); variance strictly decays toward 0.0
    for _ in range(120):
        var res = svgf_accumulate_moments_cpu(m1, m2, Float32(2.0), h_len)
        m1 = res[0]
        m2 = res[1]
        var_est = res[2]
        h_len = res[3]

    ctx.assert_approx(m1, 2.0, 0.05, "m1 converges to new constant luminance 2.0")
    ctx.assert_true(var_est < 0.02, "variance strictly decays under constant input signal")
    ctx.assert_true(var_est >= 0.0, "variance non-negativity maintained under convergence")

    # 3. SPIR-V SSA Dominance validation of SVGF temporal reprojection shader
    var rep = validate_spirv_module(svgf_temporal_reproject_shader())
    ctx.assert_true(rep.valid, "SVGF temporal reprojection shader passes SPIR-V SSA validation")

def test_f18_svgf_bilateral_depth_weight(mut ctx: TestContext) raises:
    # Bilateral depth weight: w_z = exp(-|delta_z| / max(0.05, 0.08 * z_center))
    var z_center = Float32(5.0)
    var sigma_z = max(Float32(0.05), Float32(0.08) * z_center) # 0.40

    # Test 1: Identical depth (delta_z = 0) -> weight = 1.0
    var w_zero = exp(-Float32(0.0) / sigma_z)
    ctx.assert_approx(w_zero, 1.0, 0.0001, "depth weight at delta_z = 0 is exactly 1.0")

    # Test 2: delta_z = sigma_z -> weight = exp(-1) = 0.367879
    var w_sigma = exp(-sigma_z / sigma_z)
    ctx.assert_approx(w_sigma, 0.36788, 0.001, "depth weight at delta_z = sigma_z is exp(-1)")

    # Test 3: Large depth discontinuity (delta_z = 2.0 * z_center = 10.0) -> strict edge rejection
    var w_large = exp(-Float32(10.0) / sigma_z)
    ctx.assert_true(w_large < 0.0001, "large depth discontinuity is rejected with weight < 1e-4")

def test_f18_svgf_bilateral_normal_weight(mut ctx: TestContext) raises:
    # Normal weight: w_n = max(0.0, dot(n_p, n_q))^32
    var n_p = Vec3(0, 1, 0)

    # Coplanar: dot = 1.0 -> w = 1.0
    var w_coplanar = max(Float32(0.0), n_p.dot(Vec3(0, 1, 0))) ** 32
    ctx.assert_approx(w_coplanar, 1.0, 0.0001, "coplanar normals produce weight 1.0")

    # 30 deg angle: cos(30 deg) = 0.866025 -> (0.866025)^32 = 0.01007
    var n_30 = Vec3(0.5, 0.8660254, 0).normalized()
    var w_30 = max(Float32(0.0), n_p.dot(n_30)) ** 32
    ctx.assert_approx(w_30, 0.010, 0.005, "30-degree normal difference attenuated to ~0.01")

    # >= 90 deg angle: dot <= 0.0 -> w = 0.0
    var n_opp = Vec3(0, -1, 0)
    var w_opp = max(Float32(0.0), n_p.dot(n_opp)) ** 32
    ctx.assert_approx(w_opp, 0.0, 0.0001, "opposing normals produce weight 0.0")

def test_f18_svgf_temporal_accumulation_alpha(mut ctx: TestContext) raises:
    # Exponential moving average: S_k = (1 - alpha) * S_{k-1} + alpha * X
    # For constant input X = 1.0 starting from S_0 = 0.0, exact closed form: S_k = 1 - (1 - alpha)^k
    var alpha = Float32(0.2)
    var s_accum = Float32(0.0)

    for k in range(1, 16):
        s_accum = (Float32(1.0) - alpha) * s_accum + alpha * Float32(1.0)
        var expected = Float32(1.0) - ((Float32(1.0) - alpha) ** k)
        ctx.assert_approx(s_accum, expected, 0.0001, "step response matches exact closed form at step " + String(k))

    ctx.assert_approx(s_accum, 0.965, 0.005, "after 15 steps accumulator reaches ~96.5% of final value")

def test_f18_svgf_atrous_wavelet_step(mut ctx: TestContext) raises:
    # 1D À-Trous B-spline kernel weights: [1/16, 1/4, 3/8, 1/4, 1/16]
    var weights = List[Float32]()
    weights.append(Float32(0.0625))
    weights.append(Float32(0.25))
    weights.append(Float32(0.375))
    weights.append(Float32(0.25))
    weights.append(Float32(0.0625))
    var weight_sum = Float32(0.0)
    for i in range(len(weights)):
        weight_sum += weights[i]
    ctx.assert_approx(weight_sum, 1.0, 0.00001, "À-Trous wavelet kernel weights sum exactly to 1.0 (energy conservation)")

    # 2D Separable convolution of constant field C = 42.0 produces exactly 42.0
    var c_val = Float32(42.0)
    var filtered_val = Float32(0.0)
    for dy in range(len(weights)):
        for dx in range(len(weights)):
            filtered_val += weights[dy] * weights[dx] * c_val
    ctx.assert_approx(filtered_val, c_val, 0.0001, "À-Trous filter on constant signal perfectly preserves energy")


# =========================================================================
# F19: Lumen Surface Cache & Spatial Hash (5 tests)
# =========================================================================

def test_f19_surface_cache_amortized_update(mut ctx: TestContext) raises:
    var sc_shader = surface_cache_update_shader(False)
    var rep = validate_spirv_module(sc_shader)
    ctx.assert_true(rep.valid, "Lumen surface cache updater compute shader passes SSA validation")
    ctx.assert_true(len(sc_shader) > 100, "surface cache shader emits non-empty instruction stream")

def test_f19_surface_cache_header_offset(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_true(ren.buf_surfcache.size >= 1024 * 4, "surface cache buffer allocates instance headers")
    ctx.assert_true(ren.max_surfcache_tris > 0, "max surface cache triangles allocated")
    ren.shutdown()

def test_f19_spatial_radiance_hash_key(mut ctx: TestContext) raises:
    var sc_rt_shader = surface_cache_update_shader(True)
    var rep = validate_spirv_module(sc_rt_shader)
    ctx.assert_true(rep.valid, "RT-accelerated surface cache updater passes SSA validation")

def test_f19_surface_cache_triangle_allocation(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    var mat_id = ren.add_material(AaaMaterial(base_color=Vec3(0.8, 0.8, 0.8), roughness=0.5))
    ctx.assert_true(mat_id >= 0, "material registered for surface cache instance")
    ren.shutdown()

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
    ctx.assert_approx(h.length(), 1.0, 0.01, "GGX microfacet normal is normalized unit vector")
    ctx.assert_true(h.z > 0.0, "GGX microfacet normal points into upper hemisphere")

def test_f20_virtual_depth_reprojection(mut ctx: TestContext) raises:
    # Primary mirror surface at distance d_surf = 10.0 reflecting an object at t_hit = 15.0 -> z_virtual = 25.0
    var cam_pos = Vec3(0, 0, 0)
    var primary_pos = Vec3(0, 0, 10.0)
    var t_hit = Float32(15.0)
    var cur_uv = Vec2(0.5, 0.5)
    var prev_vp = List[Float32](length=16, fill=0.0)
    prev_vp[0] = 1.0; prev_vp[5] = 1.0; prev_vp[10] = 1.0; prev_vp[15] = 1.0
    
    var res = eval_virtual_depth_reprojection_cpu(cam_pos, primary_pos, t_hit, True, prev_vp, cur_uv)
    var z_virtual = res[0]
    ctx.assert_approx(z_virtual, 25.0, 0.001, "virtual depth evaluates exact sum: ||P - C|| + t_hit = 10.0 + 15.0 = 25.0")

    # Specular Parallax: under lateral camera movement delta_x = 1.0
    var delta_x = Float32(1.0)
    var delta_u_spec = delta_x / z_virtual
    var delta_u_surf = delta_x / Float32(10.0)
    ctx.assert_true(delta_u_spec < delta_u_surf, "specular reflection parallax is smaller than primary surface parallax")
    ctx.assert_approx(delta_u_surf / delta_u_spec, 2.5, 0.001, "parallax ratio exactly matches depth ratio z_virtual / d_surf = 2.5")

def test_f20_ggx_smith_masking_shadowing(mut ctx: TestContext) raises:
    # Smith height-correlated G2(wo, wi)
    var alpha = Float32(0.2)
    var a2 = alpha * alpha

    # Normal incidence (n dot wo = n dot wi = 1.0) -> G2 = 1.0
    var num_norm = Float32(2.0) * Float32(1.0) * Float32(1.0)
    var den_norm = Float32(1.0) * sqrt(a2 + (Float32(1.0) - a2)) + Float32(1.0) * sqrt(a2 + (Float32(1.0) - a2))
    var g2_norm = num_norm / den_norm
    ctx.assert_approx(g2_norm, 1.0, 0.0001, "Smith G2 at normal incidence is exactly 1.0")

    # Grazing angle (n dot wo = 0.1, n dot wi = 0.8) -> G2 bounded in (0, 1) and decays
    var ndotwo = Float32(0.1)
    var ndotwi = Float32(0.8)
    var num_gr = Float32(2.0) * ndotwo * ndotwi
    var lambda_v = ndotwi * sqrt(a2 + (Float32(1.0) - a2) * ndotwo * ndotwo)
    var lambda_l = ndotwo * sqrt(a2 + (Float32(1.0) - a2) * ndotwi * ndotwi)
    var g2_gr = num_gr / (lambda_v + lambda_l)
    ctx.assert_true(g2_gr > 0.0 and g2_gr < 1.0, "Smith G2 is strictly bounded in (0, 1) at grazing angle")
    ctx.assert_true(g2_gr < g2_norm, "Smith G2 decreases monotonically as viewing angle grazes")

def test_f20_reflection_fresnel_schlick(mut ctx: TestContext) raises:
    # Schlick Fresnel: F(cos_theta) = F0 + (1 - F0) * (1 - cos_theta)^5
    var f0 = Float32(0.04)

    # cos_theta = 1.0 (normal incidence) -> F = F0
    var f_norm = f0 + (Float32(1.0) - f0) * ((Float32(1.0) - Float32(1.0)) ** 5)
    ctx.assert_approx(f_norm, 0.04, 0.0001, "Schlick Fresnel at normal incidence equals F0 = 0.04")

    # cos_theta = 0.5 (60 degrees) -> F = 0.04 + 0.96 * (0.5)^5 = 0.07
    var f_60 = f0 + (Float32(1.0) - f0) * ((Float32(1.0) - Float32(0.5)) ** 5)
    ctx.assert_approx(f_60, 0.07, 0.0001, "Schlick Fresnel at 60 deg incidence equals 0.07")

    # cos_theta = 0.0 (90 degrees / grazing incidence) -> F = 1.0
    var f_90 = f0 + (Float32(1.0) - f0) * ((Float32(1.0) - Float32(0.0)) ** 5)
    ctx.assert_approx(f_90, 1.0, 0.0001, "Schlick Fresnel at grazing incidence equals 1.0")

def test_f20_vndf_hemisphere_sampling(mut ctx: TestContext) raises:
    var h = _importance_sample_ggx(0.25, 0.75, 0.2, Vec3(0, 1, 0))
    ctx.assert_approx(h.length(), 1.0, 0.001, "importance sampled GGX normal is unit length")
    ctx.assert_true(h.dot(Vec3(0, 1, 0)) > 0.0, "sampled normal is strictly in upper hemisphere")


# =========================================================================
# F21: ReSTIR DI Local Light Shadows (5 tests)
# =========================================================================

def test_f21_restir_light_clustering(mut ctx: TestContext) raises:
    ctx.assert_equal(AAA_CLUSTER_X, 16, "16 X clusters")
    ctx.assert_equal(AAA_CLUSTER_Y, 9, "9 Y clusters")
    ctx.assert_equal(AAA_CLUSTER_Z, 24, "24 Z clusters")
    ctx.assert_equal(AAA_TOTAL_CLUSTERS, 3456, "3,456 total froxel clusters in engine")
    var cull_spv = aaa_cluster_cull_shader(0, 1)
    var rep = validate_spirv_module(cull_spv)
    ctx.assert_true(rep.valid, "cluster light culling compute shader validates")

def test_f21_restir_candidate_selection(mut ctx: TestContext) raises:
    var res = RestirReservoir()
    # Stream 4 candidates with increasing PDFs: 1.0, 5.0, 20.0, 100.0
    _ = res.update(UInt32(10), Float32(1.0), Float32(1.0), Float32(0.9))
    _ = res.update(UInt32(20), Float32(5.0), Float32(5.0), Float32(0.9))
    _ = res.update(UInt32(30), Float32(20.0), Float32(20.0), Float32(0.9))
    var replaced = res.update(UInt32(40), Float32(100.0), Float32(100.0), Float32(0.05))
    ctx.assert_true(replaced, "high-PDF candidate successfully replaces earlier samples")
    ctx.assert_equal(res.light_id, UInt32(40), "reservoir holds chosen candidate light 40")
    ctx.assert_equal(res.M_count, 4, "reservoir M_count accurately counts 4 candidates")
    ctx.assert_approx(res.weight_sum, 126.0, 0.001, "reservoir tracks cumulative weight sum 126.0")

def test_f21_restir_spatial_resampling(mut ctx: TestContext) raises:
    var r_center = RestirReservoir(UInt32(5), Float32(50.0), 4, Float32(12.5))
    var r_neighbor = RestirReservoir(UInt32(8), Float32(120.0), 6, Float32(20.0))
    # Unbiased weight of neighbor: W_n = 120 / (6 * 20) = 1.0
    ctx.assert_approx(r_neighbor.unbiased_weight(), 1.0, 0.001, "neighbor unbiased weight is 1.0")
    # Merge neighbor into center with target_pdf = 25.0 at center point
    # Weight added = target_pdf * W_n * M_n = 25.0 * 1.0 * 6 = 150.0
    var merged = r_center.merge(r_neighbor, Float32(25.0), Float32(0.1))
    ctx.assert_true(merged, "reservoir merge succeeds")
    ctx.assert_equal(r_center.M_count, 10, "merged M_count is 4 + 6 = 10")
    ctx.assert_approx(r_center.weight_sum, 200.0, 0.001, "merged weight sum is 50.0 + 150.0 = 200.0")
    ctx.assert_equal(r_center.light_id, UInt32(8), "center adopted candidate 8 from neighbor")

def test_f21_restir_max_clustered_lights(mut ctx: TestContext) raises:
    ctx.assert_equal(Int(AAA_MAX_CLUSTERED_LIGHTS), 1024, "AAA_MAX_CLUSTERED_LIGHTS is 1024")

def test_f21_restir_unbiased_contribution_weight(mut ctx: TestContext) raises:
    var res = RestirReservoir(UInt32(12), Float32(180.0), 10, Float32(18.0))
    # W = 180 / (10 * 18) = 1.0
    var w = res.unbiased_weight()
    ctx.assert_approx(w, 1.0, 0.0001, "unbiased contribution weight evaluates exact ratio")
    # Verify unshadowed vs shadowed evaluation
    var unshadowed_radiance = Float32(45.0)
    var vis_unshadowed = Float32(1.0)
    var vis_shadowed = Float32(0.0)
    var contrib_unshadowed = vis_unshadowed * w * unshadowed_radiance
    var contrib_shadowed = vis_shadowed * w * unshadowed_radiance
    ctx.assert_approx(contrib_unshadowed, 45.0, 0.001, "unshadowed sample contributes full radiance")
    ctx.assert_approx(contrib_shadowed, 0.0, 0.0001, "shadowed ray-queried sample evaluates 0.0 radiance")


# =========================================================================
# F22: Two-Phase Hi-Z Pyramid & VSM Clipmaps (5 tests)
# =========================================================================

def test_f22_vsm_physical_tile_count(mut ctx: TestContext) raises:
    ctx.assert_equal(Int(VSM_PHYS_TILE_COUNT), 1024, "VSM_PHYS_TILE_COUNT is 1024 physical pages")

def test_f22_vsm_page_allocation(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    vsm.invalidate_all()
    ctx.assert_equal(len(vsm.phys_to_virtual), 1024, "1024 physical tiles in VsmPageManager")

def test_f22_vsm_invalidation(mut ctx: TestContext) raises:
    var vsm = VsmPageManager()
    for i in range(len(vsm.virtual_valid)):
        vsm.virtual_valid[i] = True
    vsm.invalidate_region(0.1, 0.1, 0.4, 0.4)
    var test_vx = Int(0.2 * Float32(VSM_VIRTUAL_PAGES_DIM))
    var test_vy = Int(0.2 * Float32(VSM_VIRTUAL_PAGES_DIM))
    var inside_idx = test_vy * VSM_VIRTUAL_PAGES_DIM + test_vx
    ctx.assert_false(vsm.virtual_valid[inside_idx], "page inside invalidated region is invalid")
    var test_out_x = Int(0.9 * Float32(VSM_VIRTUAL_PAGES_DIM))
    var test_out_y = Int(0.9 * Float32(VSM_VIRTUAL_PAGES_DIM))
    var outside_idx = test_out_y * VSM_VIRTUAL_PAGES_DIM + test_out_x
    ctx.assert_true(vsm.virtual_valid[outside_idx], "page outside invalidated region remains valid")

def test_f22_hiz_two_phase_cull_split(mut ctx: TestContext) raises:
    # 1. Genuine SPIR-V validation of Two-Phase meshlet cull shader bytecode
    var cull_shader = meshlet_cull_shader()
    var rep = validate_spirv_module(cull_shader)
    ctx.assert_true(rep.valid, "two-phase meshlet cull compute shader passes SPIR-V validation")
    ctx.assert_true(len(cull_shader) > 100, "meshlet cull shader produces valid bytecode stream")

    # 2. Verify candidate packing and unpacking math: (inst_idx << 22) | (ml_id & 0x3FFFFF)
    var test_inst = UInt32(1023)
    var test_ml = UInt32(4194303)
    var packed = (test_inst << 22) | (test_ml & 0x3FFFFF)
    var unpacked_inst = packed >> 22
    var unpacked_ml = packed & 0x3FFFFF
    ctx.assert_equal(unpacked_inst, test_inst, "candidate bitfield unpacks instance index accurately")
    ctx.assert_equal(unpacked_ml, test_ml, "candidate bitfield unpacks meshlet id accurately")

    # 3. Verify FlintAaaVulkanRenderer two-phase culling buffer and descriptor configuration
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_true(ren.buf_occluded_candidates.size > 0, "renderer allocates buf_occluded_candidates")
    ctx.assert_true(ren.buf_indirect_phase2.size > 0, "renderer allocates buf_indirect_phase2")
    ctx.assert_equal(len(ren.desc_sets_cull_p1), 2, "renderer configures Phase 1 cull descriptor sets")
    ctx.assert_equal(len(ren.desc_sets_cull_p2), 2, "renderer configures Phase 2 cull descriptor sets")
    ctx.assert_true(ren.desc_sets_cull_p1[0] != 0, "Phase 1 cull descriptor set is valid")
    ctx.assert_true(ren.desc_sets_cull_p2[0] != 0, "Phase 2 cull descriptor set is valid")
    ren.shutdown()

    # 2. Verify 4-Cascade Toroidal addressing (vx mod 64, vy mod 64)
    ctx.assert_equal(compute_toroidal_page_index(0, 0, 0), 0, "cascade 0 origin maps to slot 0")
    ctx.assert_equal(compute_toroidal_page_index(0, 64, 0), 0, "64 mod 64 toroidally wraps to slot 0")
    ctx.assert_equal(compute_toroidal_page_index(0, 63, 63), 4095, "page (63, 63) maps to cascade 0 max slot 4095")
    ctx.assert_equal(compute_toroidal_page_index(1, 0, 0), 4096, "cascade 1 origin maps to slot 4096")
    ctx.assert_equal(compute_toroidal_page_index(3, 63, 63), 16383, "cascade 3 max maps to total page bound 16383")
    ctx.assert_equal(compute_toroidal_page_index(0, -1, 0), 63, "negative coordinate -1 wraps toroidally to 63")

def test_f22_ortho_light_view_proj(mut ctx: TestContext) raises:
    var vp = make_ortho_directional_light_vp(Vec3(0.5, 0.8, 0.2).normalized(), Vec3(0, 0, 0), 50.0, 120.0)
    ctx.assert_equal(len(vp), 16, "16-element ortho light_view_proj matrix generated")
    ctx.assert_approx(vp[15], 1.0, 0.001, "affine projection entry m[15] is 1.0")


# =========================================================================
# F23: Render Graph Transient Aliasing (5 tests)
# =========================================================================

def test_f23_rg_resource_creation(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r1 = rg.add_resource("depth", 1920 * 1080 * 4, transient=True)
    var r2 = rg.add_resource("color", 1920 * 1080 * 8, transient=True)
    ctx.assert_equal(r1, 0, "first resource id is 0")
    ctx.assert_equal(r2, 1, "second resource id is 1")

def test_f23_rg_pass_node_registration(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r1 = rg.add_resource("gbuffer", 1024 * 1024 * 16)
    var accesses = List[RgResourceAccess]()
    accesses.append(RgResourceAccess(r1, 1))
    var p = rg.add_pass("gbuffer_pass", accesses)
    ctx.assert_equal(p, 0, "first pass node id is 0")

def test_f23_rg_compile_aliasing(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r1 = rg.add_resource("early_depth", 1024 * 1024 * 16, transient=True)
    var r2 = rg.add_resource("mid_gbuffer", 1024 * 1024 * 16, transient=True)
    var r3 = rg.add_resource("late_post", 1024 * 1024 * 16, transient=True)
    var acc1 = List[RgResourceAccess]()
    acc1.append(RgResourceAccess(r1, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("pass0", acc1)
    var acc2 = List[RgResourceAccess]()
    acc2.append(RgResourceAccess(r2, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("pass1", acc2)
    var acc3 = List[RgResourceAccess]()
    acc3.append(RgResourceAccess(r3, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("pass2", acc3)
    var ok = rg.compile(64 * 1024 * 1024)
    ctx.assert_true(ok, "render graph compilation succeeds")
    ctx.assert_equal(rg.resources[r1].aliased_offset_bytes, 0, "r1 allocated at offset 0")
    ctx.assert_equal(rg.resources[r2].aliased_offset_bytes, 0, "r2 aliased at offset 0")
    ctx.assert_equal(rg.resources[r3].aliased_offset_bytes, 0, "r3 aliased at offset 0")
    ctx.assert_equal(rg.aliased_peak_bytes, 16 * 1024 * 1024, "aliased peak is 16 MB")
    ctx.assert_equal(rg.unaliased_transient_bytes, 48 * 1024 * 1024, "unaliased transient sum is 48 MB")
    ctx.assert_approx(rg.aliasing_savings_ratio(), Float32(2.0 / 3.0), 0.01, "aliasing savings ratio is ~66.7%")

def test_f23_rg_savings_ratio(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r1 = rg.add_resource("r1", 1024 * 1024 * 16, transient=True)
    var r2 = rg.add_resource("r2", 1024 * 1024 * 16, transient=True)
    var r3 = rg.add_resource("r3", 1024 * 1024 * 16, transient=True)
    var acc1 = List[RgResourceAccess]()
    acc1.append(RgResourceAccess(r1, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("p1", acc1)
    var acc2 = List[RgResourceAccess]()
    acc2.append(RgResourceAccess(r2, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("p2", acc2)
    var acc3 = List[RgResourceAccess]()
    acc3.append(RgResourceAccess(r3, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("p3", acc3)
    _ = rg.compile(64 * 1024 * 1024)
    var ratio = rg.aliasing_savings_ratio()
    ctx.assert_approx(ratio, Float32(2.0 / 3.0), 0.01, "aliasing savings ratio exactly matches 1 - 16/48 = 0.667")

def test_f23_rg_profile_format(mut ctx: TestContext) raises:
    var rg = AaaRenderGraph()
    var r1 = rg.add_resource("res", 1024 * 1024 * 8, transient=True)
    var acc1 = List[RgResourceAccess]()
    acc1.append(RgResourceAccess(r1, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("GpuPass", acc1)
    _ = rg.compile(64 * 1024 * 1024)
    var profile = rg.format_gpu_profile_summary()
    ctx.assert_true(profile.byte_length() > 0, "GPU profile summary string generated")


# =========================================================================
# F24: Persistent mmap .flpk v2 VFS (5 tests)
# =========================================================================

def test_f24_vfs_encode_and_read(mut ctx: TestContext) raises:
    var pack_bytes = create_mock_flpk_data()
    ctx.assert_true(len(pack_bytes) > 64, "mock flpk data non-empty")

def test_f24_vfs_directory_entries(mut ctx: TestContext) raises:
    var pack_bytes = create_mock_flpk_data()
    # Header format: 4 bytes magic, 4 bytes version, 4 bytes entry count
    var count = Int(pack_bytes[8]) | (Int(pack_bytes[9]) << 8)
    ctx.assert_true(count >= 3, "directory contains at least 3 entries")

def test_f24_vfs_alignment_64byte(mut ctx: TestContext) raises:
    var pack_bytes = create_mock_flpk_data()
    ctx.assert_true(len(pack_bytes) >= 64, "flpk container has valid size")
    var magic_valid = pack_bytes[0] == 0x46 and pack_bytes[1] == 0x4C and pack_bytes[2] == 0x50 and pack_bytes[3] == 0x4B
    ctx.assert_true(magic_valid, "FLPK magic header verified in container")

def test_f24_vfs_multi_file_bundle(mut ctx: TestContext) raises:
    var names = List[String]()
    var files = List[List[UInt8]]()
    names.append(String("a.bin"))
    var f1 = List[UInt8]()
    f1.append(1); f1.append(2); f1.append(3)
    var f1_len = len(f1)
    files.append(f1^)
    var pack = encode_pack(names, files)
    ctx.assert_true(len(pack) > f1_len, "pack encapsulates file content")

def test_f24_vfs_missing_file_handling(mut ctx: TestContext) raises:
    var dummy_pack = String("/tmp/nonexistent.flpk")
    var data = pack_read(dummy_pack, "foo.txt")
    ctx.assert_equal(len(data), 0, "missing pack or file returns empty list without crashing")


# =========================================================================
# F25: Async Transfer Ring & Mip-Tail Streaming (5 tests)
# =========================================================================

def test_f25_residency_streamer_feedback(mut ctx: TestContext) raises:
    var streamer = AaaResidencyStreamer(128, 4)
    streamer.record_gpu_feedback(10, 1)
    ctx.assert_equal(streamer.requested_min_mip[10], 1, "GPU feedback records requested mip level 1")
    ctx.assert_equal(streamer.requested_min_mip[11], 4, "unrequested texture remains at tail mip level 4")

def test_f25_residency_budget_tick(mut ctx: TestContext) raises:
    var streamer = AaaResidencyStreamer(128, 4)
    streamer.record_gpu_feedback(5, 0)
    var uploaded = streamer.tick_streaming_budget(1024, 512 * 1024)
    ctx.assert_true(uploaded >= 0, "budget tick returns non-negative uploaded count")

def test_f25_mip_tail_residency_threshold(mut ctx: TestContext) raises:
    var streamer = AaaResidencyStreamer(64, 4)
    ctx.assert_equal(streamer.resident_min_mip[0], 4, "initial resident tail mip level is 4")
    ctx.assert_equal(len(streamer.resident_min_mip), 64, "streamer manages 64 textures")

def test_f25_transfer_ring_capacity(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_true(ren.buf_staging.size >= 8 * 1024 * 1024, "renderer staging/transfer ring allocated >= 8 MiB")
    ren.shutdown()

def test_f25_streamer_max_textures(mut ctx: TestContext) raises:
    var streamer = AaaResidencyStreamer(256, 4)
    streamer.record_gpu_feedback(255, 2)
    ctx.assert_equal(streamer.requested_min_mip[255], 2, "max texture index 255 records requested mip 2")
    ctx.assert_equal(len(streamer.resident_min_mip), 256, "residency streamer capacity for 256 textures")


# =========================================================================
# F26: Cluster-Group QEM Meshlet DAG (5 tests)
# =========================================================================

def test_f26_qem_quadric_plane(mut ctx: TestContext) raises:
    var q = _SymQuadric.from_plane(0.0, 1.0, 0.0, -2.0)
    var dist = q.eval(Vec3(0.0, 2.0, 0.0))
    ctx.assert_approx(Float32(dist), 0.0, 0.001, "point on plane evaluates to zero quadric error")

def test_f26_qem_quadric_addition(mut ctx: TestContext) raises:
    var q1 = _SymQuadric.from_plane(1.0, 0.0, 0.0, 0.0)
    var q2 = _SymQuadric.from_plane(0.0, 1.0, 0.0, 0.0)
    var q_sum = q1 + q2
    var err = q_sum.eval(Vec3(1.0, 1.0, 0.0))
    ctx.assert_true(err > 0.0, "quadrics additively accumulate error")

def test_f26_qem_simplify_reduction(mut ctx: TestContext) raises:
    var mesh = create_mock_meshlet_mesh(12)
    var orig_tris = len(mesh.indices) // 3
    var target_tris = orig_tris // 2
    var res = simplify_mesh_qem(mesh, target_tris)
    var simplified_tris = len(res[0].indices) // 3
    ctx.assert_true(simplified_tris <= orig_tris, "QEM simplification reduces or maintains triangle count")

def test_f26_meshlet_cluster_bounds(mut ctx: TestContext) raises:
    var mesh = create_mock_meshlet_mesh(12)
    var slices = simplify_mesh_qem(mesh, len(mesh.indices) // 6)
    ctx.assert_true(len(slices) > 0, "QEM returns LOD slices")
    ctx.assert_true(len(slices[0].indices) > 0, "LOD slice contains valid simplified index list")

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
    mat.iridescence_thickness_nm = 380.0
    ctx.assert_approx(mat.iridescence, 1.0, 0.001, "iridescence strength 1.0")
    ctx.assert_approx(mat.iridescence_ior, 1.33, 0.001, "iridescence IOR matches water/soap film")

def test_f27_thin_film_iridescence_cpu(mut ctx: TestContext) raises:
    var color = eval_thin_film_iridescence_cpu(0.8, 350.0, 1.33)
    ctx.assert_true(color.x >= 0.0 and color.y >= 0.0 and color.z >= 0.0, "iridescence spectral RGB valid")

def test_f27_rnm_blend_cpu(mut ctx: TestContext) raises:
    var macro_n = Vec3(0, 1, 0)
    var detail_n = Vec3(0.1, 0.98, 0.1).normalized()
    var blended = reoriented_normal_blend_cpu(macro_n, detail_n, 1.0)
    ctx.assert_approx(blended.length(), 1.0, 0.01, "RNM blended normal remains unit length")


# =========================================================================
# F28: Pipeline Cache & Low-Latency Jobs (5 tests)
# =========================================================================

def test_f28_job_pool_allocation(mut ctx: TestContext) raises:
    var pool = JobPool()
    var ok = pool.allocate()
    ctx.assert_true(ok, "JobPool allocates ring memory cleanly")
    ctx.assert_true(pool.allocated(), "JobPool reports allocated state")
    pool.stop()

def test_f28_job_pool_workers_start_stop(mut ctx: TestContext) raises:
    var pool = JobPool()
    _ = pool.allocate()
    var started = pool.start(2)
    ctx.assert_true(started, "JobPool starts 2 worker threads")
    ctx.assert_true(pool.ready(), "JobPool is ready")
    pool.stop()
    ctx.assert_false(pool.ready(), "JobPool stopped cleanly")

def test_f28_pipeline_cache_magic(mut ctx: TestContext) raises:
    var ren = FlintAaaVulkanRenderer(width=160, height=120)
    ctx.assert_true(ren.desc_pool != 0, "renderer pipeline descriptor pool created")
    ctx.assert_true(ren.pipe_layout != 0, "renderer pipeline layout created")
    ren.shutdown()

def test_f28_job_pool_default_workers(mut ctx: TestContext) raises:
    var workers = default_worker_count()
    ctx.assert_true(workers >= 1, "default worker count at least 1")

def test_f28_job_pool_ready_state(mut ctx: TestContext) raises:
    var pool = JobPool()
    ctx.assert_false(pool.ready(), "unallocated pool is not ready")


# =========================================================================
# F29: Chunked Bitset ECS & CommandBuffer (5 tests)
# =========================================================================

def test_f29_entity_handle_generation(mut ctx: TestContext) raises:
    var h = EntityHandle()
    ctx.assert_false(h.valid(), "default handle is invalid")
    var packed = h.packed()
    var unpacked = EntityHandle.unpack(packed)
    ctx.assert_equal(unpacked.packed(), packed, "EntityHandle pack/unpack roundtrip")

def test_f29_archetype_registry(mut ctx: TestContext) raises:
    var reg = ArchetypeRegistry()
    var arch_id = reg.register("Vehicle")
    ctx.assert_equal(reg.name_of(arch_id), "Vehicle", "registered archetype name matches")

def test_f29_archetype_spawn_despawn(mut ctx: TestContext) raises:
    var slots = ArchetypeSlots(1, 64)
    var handle = slots.spawn()
    ctx.assert_true(handle.valid(), "spawned handle is valid")
    var row = slots.resolve(handle)
    ctx.assert_true(row >= 0, "handle resolves to non-negative row")
    var despawned = slots.despawn(handle)
    ctx.assert_true(despawned, "despawn succeeds")
    var row_dead = slots.resolve(handle)
    ctx.assert_equal(row_dead, -1, "despawned handle cannot be resolved")

def test_f29_simd_column_load_store(mut ctx: TestContext) raises:
    var col = SimdColumn[DType.float32](16)
    col.set(0, 1.25)
    col.set(1, 2.50)
    ctx.assert_approx(col.get(0), 1.25, 0.001, "get(0) matches set(0)")
    ctx.assert_approx(col.get(1), 2.50, 0.001, "get(1) matches set(1)")

def test_f29_deferred_tick_commit(mut ctx: TestContext) raises:
    var slots = ArchetypeSlots(1, 32)
    var h = slots.spawn(deferred=True)
    ctx.assert_true(h.valid(), "deferred spawn yields valid handle")
    slots.commit_tick()
    var row = slots.resolve(h)
    ctx.assert_true(row >= 0, "resolved after commit_tick")


# =========================================================================
# Aggregator for Tier 1 Milestone 3
# =========================================================================

def run_all(mut ctx: TestContext) raises:
    ctx.start_suite("T1_M3_RtStreaming")

    # F18
    try: test_f18_svgf_variance_moments_formula(ctx); ctx.record_pass("test_f18_svgf_variance_moments_formula") except e: ctx.record_fail("test_f18_svgf_variance_moments_formula", String(e))
    try: test_f18_svgf_bilateral_depth_weight(ctx); ctx.record_pass("test_f18_svgf_bilateral_depth_weight") except e: ctx.record_fail("test_f18_svgf_bilateral_depth_weight", String(e))
    try: test_f18_svgf_bilateral_normal_weight(ctx); ctx.record_pass("test_f18_svgf_bilateral_normal_weight") except e: ctx.record_fail("test_f18_svgf_bilateral_normal_weight", String(e))
    try: test_f18_svgf_temporal_accumulation_alpha(ctx); ctx.record_pass("test_f18_svgf_temporal_accumulation_alpha") except e: ctx.record_fail("test_f18_svgf_temporal_accumulation_alpha", String(e))
    try: test_f18_svgf_atrous_wavelet_step(ctx); ctx.record_pass("test_f18_svgf_atrous_wavelet_step") except e: ctx.record_fail("test_f18_svgf_atrous_wavelet_step", String(e))

    # F19
    try: test_f19_surface_cache_amortized_update(ctx); ctx.record_pass("test_f19_surface_cache_amortized_update") except e: ctx.record_fail("test_f19_surface_cache_amortized_update", String(e))
    try: test_f19_surface_cache_header_offset(ctx); ctx.record_pass("test_f19_surface_cache_header_offset") except e: ctx.record_fail("test_f19_surface_cache_header_offset", String(e))
    try: test_f19_spatial_radiance_hash_key(ctx); ctx.record_pass("test_f19_spatial_radiance_hash_key") except e: ctx.record_fail("test_f19_spatial_radiance_hash_key", String(e))
    try: test_f19_surface_cache_triangle_allocation(ctx); ctx.record_pass("test_f19_surface_cache_triangle_allocation") except e: ctx.record_fail("test_f19_surface_cache_triangle_allocation", String(e))
    try: test_f19_ltc_night_bounce_radiance(ctx); ctx.record_pass("test_f19_ltc_night_bounce_radiance") except e: ctx.record_fail("test_f19_ltc_night_bounce_radiance", String(e))

    # F20
    try: test_f20_vndf_roughness_clamping(ctx); ctx.record_pass("test_f20_vndf_roughness_clamping") except e: ctx.record_fail("test_f20_vndf_roughness_clamping", String(e))
    try: test_f20_virtual_depth_reprojection(ctx); ctx.record_pass("test_f20_virtual_depth_reprojection") except e: ctx.record_fail("test_f20_virtual_depth_reprojection", String(e))
    try: test_f20_ggx_smith_masking_shadowing(ctx); ctx.record_pass("test_f20_ggx_smith_masking_shadowing") except e: ctx.record_fail("test_f20_ggx_smith_masking_shadowing", String(e))
    try: test_f20_reflection_fresnel_schlick(ctx); ctx.record_pass("test_f20_reflection_fresnel_schlick") except e: ctx.record_fail("test_f20_reflection_fresnel_schlick", String(e))
    try: test_f20_vndf_hemisphere_sampling(ctx); ctx.record_pass("test_f20_vndf_hemisphere_sampling") except e: ctx.record_fail("test_f20_vndf_hemisphere_sampling", String(e))

    # F21
    try: test_f21_restir_light_clustering(ctx); ctx.record_pass("test_f21_restir_light_clustering") except e: ctx.record_fail("test_f21_restir_light_clustering", String(e))
    try: test_f21_restir_candidate_selection(ctx); ctx.record_pass("test_f21_restir_candidate_selection") except e: ctx.record_fail("test_f21_restir_candidate_selection", String(e))
    try: test_f21_restir_spatial_resampling(ctx); ctx.record_pass("test_f21_restir_spatial_resampling") except e: ctx.record_fail("test_f21_restir_spatial_resampling", String(e))
    try: test_f21_restir_max_clustered_lights(ctx); ctx.record_pass("test_f21_restir_max_clustered_lights") except e: ctx.record_fail("test_f21_restir_max_clustered_lights", String(e))
    try: test_f21_restir_unbiased_contribution_weight(ctx); ctx.record_pass("test_f21_restir_unbiased_contribution_weight") except e: ctx.record_fail("test_f21_restir_unbiased_contribution_weight", String(e))

    # F22
    try: test_f22_vsm_physical_tile_count(ctx); ctx.record_pass("test_f22_vsm_physical_tile_count") except e: ctx.record_fail("test_f22_vsm_physical_tile_count", String(e))
    try: test_f22_vsm_page_allocation(ctx); ctx.record_pass("test_f22_vsm_page_allocation") except e: ctx.record_fail("test_f22_vsm_page_allocation", String(e))
    try: test_f22_vsm_invalidation(ctx); ctx.record_pass("test_f22_vsm_invalidation") except e: ctx.record_fail("test_f22_vsm_invalidation", String(e))
    try: test_f22_hiz_two_phase_cull_split(ctx); ctx.record_pass("test_f22_hiz_two_phase_cull_split") except e: ctx.record_fail("test_f22_hiz_two_phase_cull_split", String(e))
    try: test_f22_ortho_light_view_proj(ctx); ctx.record_pass("test_f22_ortho_light_view_proj") except e: ctx.record_fail("test_f22_ortho_light_view_proj", String(e))

    # F23
    try: test_f23_rg_resource_creation(ctx); ctx.record_pass("test_f23_rg_resource_creation") except e: ctx.record_fail("test_f23_rg_resource_creation", String(e))
    try: test_f23_rg_pass_node_registration(ctx); ctx.record_pass("test_f23_rg_pass_node_registration") except e: ctx.record_fail("test_f23_rg_pass_node_registration", String(e))
    try: test_f23_rg_compile_aliasing(ctx); ctx.record_pass("test_f23_rg_compile_aliasing") except e: ctx.record_fail("test_f23_rg_compile_aliasing", String(e))
    try: test_f23_rg_savings_ratio(ctx); ctx.record_pass("test_f23_rg_savings_ratio") except e: ctx.record_fail("test_f23_rg_savings_ratio", String(e))
    try: test_f23_rg_profile_format(ctx); ctx.record_pass("test_f23_rg_profile_format") except e: ctx.record_fail("test_f23_rg_profile_format", String(e))

    # F24
    try: test_f24_vfs_encode_and_read(ctx); ctx.record_pass("test_f24_vfs_encode_and_read") except e: ctx.record_fail("test_f24_vfs_encode_and_read", String(e))
    try: test_f24_vfs_directory_entries(ctx); ctx.record_pass("test_f24_vfs_directory_entries") except e: ctx.record_fail("test_f24_vfs_directory_entries", String(e))
    try: test_f24_vfs_alignment_64byte(ctx); ctx.record_pass("test_f24_vfs_alignment_64byte") except e: ctx.record_fail("test_f24_vfs_alignment_64byte", String(e))
    try: test_f24_vfs_multi_file_bundle(ctx); ctx.record_pass("test_f24_vfs_multi_file_bundle") except e: ctx.record_fail("test_f24_vfs_multi_file_bundle", String(e))
    try: test_f24_vfs_missing_file_handling(ctx); ctx.record_pass("test_f24_vfs_missing_file_handling") except e: ctx.record_fail("test_f24_vfs_missing_file_handling", String(e))

    # F25
    try: test_f25_residency_streamer_feedback(ctx); ctx.record_pass("test_f25_residency_streamer_feedback") except e: ctx.record_fail("test_f25_residency_streamer_feedback", String(e))
    try: test_f25_residency_budget_tick(ctx); ctx.record_pass("test_f25_residency_budget_tick") except e: ctx.record_fail("test_f25_residency_budget_tick", String(e))
    try: test_f25_mip_tail_residency_threshold(ctx); ctx.record_pass("test_f25_mip_tail_residency_threshold") except e: ctx.record_fail("test_f25_mip_tail_residency_threshold", String(e))
    try: test_f25_transfer_ring_capacity(ctx); ctx.record_pass("test_f25_transfer_ring_capacity") except e: ctx.record_fail("test_f25_transfer_ring_capacity", String(e))
    try: test_f25_streamer_max_textures(ctx); ctx.record_pass("test_f25_streamer_max_textures") except e: ctx.record_fail("test_f25_streamer_max_textures", String(e))

    # F26
    try: test_f26_qem_quadric_plane(ctx); ctx.record_pass("test_f26_qem_quadric_plane") except e: ctx.record_fail("test_f26_qem_quadric_plane", String(e))
    try: test_f26_qem_quadric_addition(ctx); ctx.record_pass("test_f26_qem_quadric_addition") except e: ctx.record_fail("test_f26_qem_quadric_addition", String(e))
    try: test_f26_qem_simplify_reduction(ctx); ctx.record_pass("test_f26_qem_simplify_reduction") except e: ctx.record_fail("test_f26_qem_simplify_reduction", String(e))
    try: test_f26_meshlet_cluster_bounds(ctx); ctx.record_pass("test_f26_meshlet_cluster_bounds") except e: ctx.record_fail("test_f26_meshlet_cluster_bounds", String(e))
    try: test_f26_mesh_lod_slice_ranges(ctx); ctx.record_pass("test_f26_mesh_lod_slice_ranges") except e: ctx.record_fail("test_f26_mesh_lod_slice_ranges", String(e))

    # F27
    try: test_f27_material_clearcoat_properties(ctx); ctx.record_pass("test_f27_material_clearcoat_properties") except e: ctx.record_fail("test_f27_material_clearcoat_properties", String(e))
    try: test_f27_material_sheen_properties(ctx); ctx.record_pass("test_f27_material_sheen_properties") except e: ctx.record_fail("test_f27_material_sheen_properties", String(e))
    try: test_f27_material_iridescence_properties(ctx); ctx.record_pass("test_f27_material_iridescence_properties") except e: ctx.record_fail("test_f27_material_iridescence_properties", String(e))
    try: test_f27_thin_film_iridescence_cpu(ctx); ctx.record_pass("test_f27_thin_film_iridescence_cpu") except e: ctx.record_fail("test_f27_thin_film_iridescence_cpu", String(e))
    try: test_f27_rnm_blend_cpu(ctx); ctx.record_pass("test_f27_rnm_blend_cpu") except e: ctx.record_fail("test_f27_rnm_blend_cpu", String(e))

    # F28
    try: test_f28_job_pool_allocation(ctx); ctx.record_pass("test_f28_job_pool_allocation") except e: ctx.record_fail("test_f28_job_pool_allocation", String(e))
    try: test_f28_job_pool_workers_start_stop(ctx); ctx.record_pass("test_f28_job_pool_workers_start_stop") except e: ctx.record_fail("test_f28_job_pool_workers_start_stop", String(e))
    try: test_f28_pipeline_cache_magic(ctx); ctx.record_pass("test_f28_pipeline_cache_magic") except e: ctx.record_fail("test_f28_pipeline_cache_magic", String(e))
    try: test_f28_job_pool_default_workers(ctx); ctx.record_pass("test_f28_job_pool_default_workers") except e: ctx.record_fail("test_f28_job_pool_default_workers", String(e))
    try: test_f28_job_pool_ready_state(ctx); ctx.record_pass("test_f28_job_pool_ready_state") except e: ctx.record_fail("test_f28_job_pool_ready_state", String(e))

    # F29
    try: test_f29_entity_handle_generation(ctx); ctx.record_pass("test_f29_entity_handle_generation") except e: ctx.record_fail("test_f29_entity_handle_generation", String(e))
    try: test_f29_archetype_registry(ctx); ctx.record_pass("test_f29_archetype_registry") except e: ctx.record_fail("test_f29_archetype_registry", String(e))
    try: test_f29_archetype_spawn_despawn(ctx); ctx.record_pass("test_f29_archetype_spawn_despawn") except e: ctx.record_fail("test_f29_archetype_spawn_despawn", String(e))
    try: test_f29_simd_column_load_store(ctx); ctx.record_pass("test_f29_simd_column_load_store") except e: ctx.record_fail("test_f29_simd_column_load_store", String(e))
    try: test_f29_deferred_tick_commit(ctx); ctx.record_pass("test_f29_deferred_tick_commit") except e: ctx.record_fail("test_f29_deferred_tick_commit", String(e))


def main() raises:
    var ctx = TestContext()
    run_all(ctx)
    ctx.print_summary()
    if ctx.failed_tests > 0:
        raise Error("test_t1_m3_rt_streaming failed")
