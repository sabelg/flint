"""End-to-End Verification Suite for Flint AAA Engine v2 (`test_aaa_engine_v2.mojo`):
  - Pure-Mojo SPIR-V 1.5 SSA Dominance Validation (`validate_spirv_module`) on all v2 Shaders
  - Pillar I: Barycentric Textured + Froxel-Lit Ray-Hit Shading (`emit_rt_textured_hit_shade`),
    Heitz Bounded `VNDF`, Clearcoat RT Routing, `NRD ReBLUR`-Lite Specular Denoiser,
    `RTAO` + Bent Normals + `GTSO` Specular Occlusion, and Top-2 Shadowed Hero Local/LTC Lights
  - Pillar II: `FlintTSR v3` (16-Phase Radical-Inverse Halton `(2,3)`, 5-Tap Catmull-Rom,
    Karis YCoCg Variance Ray-Clip, Pre-Bloom Resolve), `160x90x64` Temporal Froxel Volumetrics,
    GPU Froxel-Lit 3D Rain Streaks, `180°` Tile-Max Shutter Motion Blur, Starburst Glare,
    Bounded 95%-Trimmed Auto-Exposure, Cinema `AgX` Shadow-Toe, Windshield Rain & Fresnel Hologram OIT
  - Pillar III: ForzaTech Physical Road Wetness, GPU Tangent-Space 3D Ray-Box Interior Parallax
    (`AAA_MAT_INTERIOR_PARALLAX`), 4-Layer Height-Modulated Terrain Splatting (`vertex_words[11]`),
    Coupled `Toksvig` Normal+ORM Mip Baker, and Dielectric `Beer-Lambert` Refracting/Caustic/Foaming Water.
"""
from std.math import abs, max, min
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat, trs
from flint.assets.model import NativeMesh, NativeModel, NativeNode
from flint.assets.meshlet import simplify_mesh_qem
from flint.assets.mesh_builder import MeshBuilder
from flint.render.spirv_validate import validate_spirv_module
from flint.render.aaa_visbuffer import (
    BIND_AAA_HDR_OUT,
    BIND_AAA_POST_SCRATCH,
    BIND_AAA_SKINNING,
    BIND_AAA_VERTICES,
    meshlet_cull_shader,
    visbuffer_alphamask_fragment_shader,
    visbuffer_alphamask_vertex_shader,
)
from flint.render.aaa_vsm import (
    vsm_alphamask_fragment_shader,
    vsm_alphamask_vertex_shader,
)
from flint.render.aaa_bsdf import (
    AAA_MAT_DET_NONE,
    AAA_MAT_INTERIOR_PARALLAX,
    AAA_MATERIAL_WORDS,
    AaaMaterial,
    eval_kaplanyan_specular_aa_cpu,
)
from flint.render.aaa_clustered_lights import (
    AaaClusterLight,
    closest_point_on_ltc_light_cpu,
)
from flint.render.aaa_gi_svgf import (
    aaa_deferred_shade_shader,
    eval_cofactor_gram_schmidt_tbn_cpu,
    surface_cache_update_shader,
)
from flint.render.aaa_skinning_rt import (
    AAA_MAX_SKIN_BONES,
    AAA_SKIN_BIND_STRIDE_WORDS,
    AAA_SKIN_BIND_VERTS_BASE,
    AAA_SKIN_BUFFER_WORDS,
    AAA_SKIN_CUR_BONES_BASE,
    AAA_SKIN_MAX_VERTS,
    AAA_SKIN_PREV_BONES_BASE,
    AAA_SKIN_PREV_POS_BASE,
    DualQuat,
    aaa_dq_skinning_shader,
    normalize_and_sort_skin_weights_4,
    skin_vertex_dq_cpu,
)
from flint.render.aaa_spec_reblur import (
    eval_anisotropic_ggx_blur_radii_cpu,
    spec_reblur_atrous_shader,
    spec_reblur_temporal_shader,
)
from flint.render.aaa_froxel_volumetrics import (
    AAA_FROXEL_Z,
    eval_cornette_shanks_phase_cpu,
    eval_hillaire_slice_integration_cpu,
    froxel_inject_and_reproject_shader,
    froxel_prefix_integrate_shader,
    froxel_raymarch_integrate_shader,
)
from flint.render.aaa_water import (
    aaa_water_refract_shader,
    eval_beer_lambert_water_cpu,
    eval_dielectric_water_fresnel_cpu,
    eval_nyquist_lean_water_alpha2_cpu,
)
from flint.render.aaa_sky_atmosphere import (
    BrunetonAtmosphereParams,
    eval_nubis_cloud_density_cpu,
)
from flint.render.aaa_gpu_particles import (
    AaaParticleEmitter,
    gpu_particle_clear_tiles_shader,
    gpu_particle_sim_and_bin_shader,
    gpu_particle_tile_composite_shader,
)
from flint.render.aaa_oit_volumetrics import (
    AaaVolumetricPlume,
    eval_hologram_fresnel_rim_cpu,
    oit_and_6way_volumetric_shader,
)
from flint.render.aaa_post_tsr import (
    AAA_DISPLAY_AGX_REC709,
    AAA_DISPLAY_LINEAR_SRGB,
    aaa_adapt_exposure_shader,
    aaa_bloom_upsample_shader,
    aaa_display_tonemap_shader,
    aaa_motion_blur_shader,
    aaa_tsr_v3_resolve_shader,
    eval_agx_tonemap_cpu,
    halton_2_3_jitter_px,
)
from flint.render.aaa_renderer import FlintAaaVulkanRenderer


def _assert_spv_valid(name: String, words: List[UInt32]) raises:
    var rep = validate_spirv_module(words)
    if not rep.valid:
        var err_msg = "SPIR-V validation failed for " + name + ":"
        for e in rep.errors:
            err_msg += "\n  - " + e
        print(err_msg)
    assert_true(rep.valid)


def _make_uv_quad(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    nrm: Vec3,
    u_scale: Float32 = 1.0,
    v_scale: Float32 = 1.0,
    double_sided: Bool = True,
) -> NativeMesh:
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.5, metallic=0.0, double_sided=double_sided)
    var i0 = mb.add_vertex(p0, nrm, 0.0, 0.0)
    var i1 = mb.add_vertex(p1, nrm, u_scale, 0.0)
    var i2 = mb.add_vertex(p2, nrm, u_scale, v_scale)
    var i3 = mb.add_vertex(p3, nrm, 0.0, v_scale)
    mb.add_quad(i0, i1, i2, i3)
    var m = mb^.finish()
    m.colors.clear()
    return m^


def test_1_spirv_ssa_validation_all_v2_shaders() raises:
    _assert_spv_valid("aaa_deferred_shade_shader(RT=True)", aaa_deferred_shade_shader(True))
    _assert_spv_valid("aaa_deferred_shade_shader(RT=False)", aaa_deferred_shade_shader(False))
    _assert_spv_valid("surface_cache_update_shader(RT=True)", surface_cache_update_shader(True))
    _assert_spv_valid("surface_cache_update_shader(RT=False)", surface_cache_update_shader(False))
    _assert_spv_valid("spec_reblur_temporal_shader", spec_reblur_temporal_shader())
    _assert_spv_valid("spec_reblur_atrous_shader", spec_reblur_atrous_shader())
    _assert_spv_valid("froxel_inject_and_reproject_shader", froxel_inject_and_reproject_shader())
    _assert_spv_valid("froxel_raymarch_integrate_shader", froxel_raymarch_integrate_shader())
    _assert_spv_valid("aaa_water_refract_shader(RT=True)", aaa_water_refract_shader(True))
    _assert_spv_valid("aaa_water_refract_shader(RT=False)", aaa_water_refract_shader(False))
    _assert_spv_valid("oit_and_6way_volumetric_shader", oit_and_6way_volumetric_shader())
    _assert_spv_valid("aaa_tsr_v3_resolve_shader", aaa_tsr_v3_resolve_shader())
    _assert_spv_valid("aaa_motion_blur_shader", aaa_motion_blur_shader())
    _assert_spv_valid("aaa_display_tonemap_shader", aaa_display_tonemap_shader())
    _assert_spv_valid("aaa_adapt_exposure_shader", aaa_adapt_exposure_shader(BIND_AAA_POST_SCRATCH))
    _assert_spv_valid("aaa_bloom_upsample_shader", aaa_bloom_upsample_shader(BIND_AAA_HDR_OUT, BIND_AAA_POST_SCRATCH))
    _assert_spv_valid("meshlet_cull_shader", meshlet_cull_shader())
    _assert_spv_valid("visbuffer_alphamask_vertex_shader", visbuffer_alphamask_vertex_shader())
    _assert_spv_valid("visbuffer_alphamask_fragment_shader", visbuffer_alphamask_fragment_shader())
    _assert_spv_valid("vsm_alphamask_vertex_shader", vsm_alphamask_vertex_shader())
    _assert_spv_valid("vsm_alphamask_fragment_shader", vsm_alphamask_fragment_shader())
    _assert_spv_valid("froxel_prefix_integrate_shader", froxel_prefix_integrate_shader())
    _assert_spv_valid("gpu_particle_clear_tiles_shader", gpu_particle_clear_tiles_shader())
    _assert_spv_valid("gpu_particle_sim_and_bin_shader", gpu_particle_sim_and_bin_shader())
    _assert_spv_valid("gpu_particle_tile_composite_shader", gpu_particle_tile_composite_shader())
    print("  [PASS] 1. All 25 AAA v2+v3 SPIR-V 1.5 Shaders Validated (0 SSA / CFG / Type Errors)")


def test_2_cpu_models_bsdf_ltc_reblur_halton_froxel_water_agx() raises:
    # Material 24-word layout, bit 30 interior parallax & road_wetness_mode
    var mat = AaaMaterial(base_color=Vec3(0.4, 0.4, 0.4), roughness=0.7, metallic=0.0)
    mat.interior_parallax_enabled = True
    mat.detail_normal_tex = 12
    mat.road_wetness_mode = True
    mat.detail_uv_scale = Float32(44.0)
    mat.subsurface_amount = Float32(0.35)
    var words = List[UInt32]()
    mat.append_words(words)
    assert_equal(len(words), AAA_MATERIAL_WORDS)
    assert_true((words[23] & AAA_MAT_INTERIOR_PARALLAX) != 0)
    assert_equal((words[23] >> 24) & AAA_MAT_DET_NONE, UInt32(12))
    assert_true((words[23] & UInt32(0x80000000)) != 0)
    assert_true(bitcast[DType.float32](words[11]) < Float32(-43.9))

    # Kaplanyan Specular AA filters high curvature differential normals on BOTH substrate & clearcoat
    var kap = eval_kaplanyan_specular_aa_cpu(
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.35, 0.936, 0.0).normalized(),
        Vec3(0.0, 0.936, 0.35).normalized(),
        Float32(0.04),
        Float32(0.03),
    )
    assert_true(kap[0] > Float32(0.18) and kap[1] > Float32(0.18))

    # Closest point on LTC Tube & Rect
    var tube = AaaClusterLight.tube_ltc(Vec3(-3.0, 2.0, 5.0), Vec3(3.0, 2.0, 5.0), 0.12, Vec3(5.0, 5.0, 5.0), range_radius=10.0)
    var cp_tube = closest_point_on_ltc_light_cpu(tube, Vec3(8.0, 0.0, 5.0))
    assert_true(abs(cp_tube.x - Float32(3.0)) < Float32(1e-4) and abs(cp_tube.y - Float32(2.0)) < Float32(1e-4))

    var rect = AaaClusterLight.rect_ltc(
        Vec3(0.0, 4.0, 6.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
        2.0, 1.5, Vec3(5.0, 5.0, 5.0), range_radius=12.0,
    )
    var cp_rect = closest_point_on_ltc_light_cpu(rect, Vec3(-5.0, 9.0, 2.0))
    assert_true(abs(cp_rect.x - Float32(-2.0)) < Float32(1e-4) and abs(cp_rect.y - Float32(5.5)) < Float32(1e-4))

    # ReBLUR Anisotropic GGX radii: 0 on mirror (`rough < 0.05`) and contact (`hit_t -> 0`), stretched at grazing angle
    var r_mir = eval_anisotropic_ggx_blur_radii_cpu(Float32(0.02), Float32(0.50), Float32(10.0), Float32(5.0), Float32(0.52), 720)
    assert_true(r_mir[0] == Float32(0.0) and r_mir[1] == Float32(0.0))
    var r_gloss = eval_anisotropic_ggx_blur_radii_cpu(Float32(0.25), Float32(0.20), Float32(12.0), Float32(4.0), Float32(0.52), 720)
    assert_true(r_gloss[1] > r_gloss[0] * Float32(1.8))

    # 16-phase Halton (2, 3): phase 0 is (0, 0), all 16 phases in [-0.5, +0.5) and distinct
    var h0 = halton_2_3_jitter_px(0)
    assert_true(abs(h0[0]) < Float32(1e-6) and abs(h0[1]) < Float32(1e-6))
    for i in range(16):
        var hi = halton_2_3_jitter_px(i)
        assert_true(hi[0] >= Float32(-0.5) and hi[0] < Float32(0.5))
        assert_true(hi[1] >= Float32(-0.5) and hi[1] < Float32(0.5))
        for j in range(i + 1, 16):
            var hj = halton_2_3_jitter_px(j)
            assert_true(abs(hi[0] - hj[0]) + abs(hi[1] - hj[1]) > Float32(0.02))

    # Froxel phase + Hillaire analytical step + Beer-Lambert water + AgX shadow-toe + Hologram rim
    assert_equal(AAA_FROXEL_Z, 64)
    assert_true(eval_cornette_shanks_phase_cpu(Float32(0.95), Float32(0.72)) > eval_cornette_shanks_phase_cpu(Float32(-0.5), Float32(0.72)) * Float32(5.0))
    var hil = eval_hillaire_slice_integration_cpu(Vec3(0.2, 0.3, 0.5), Float32(0.1), Float32(2.0))
    assert_true(hil[1] < Float32(0.85) and hil[0].z > Float32(0.0))

    var bl_shallow = eval_beer_lambert_water_cpu(Vec3(1.0, 1.0, 1.0), Vec3(0.45, 0.08, 0.03), Float32(0.2), Vec3(0.01, 0.04, 0.08))
    var bl_deep = eval_beer_lambert_water_cpu(Vec3(1.0, 1.0, 1.0), Vec3(0.45, 0.08, 0.03), Float32(4.0), Vec3(0.01, 0.04, 0.08))
    assert_true(bl_deep.x < bl_shallow.x * Float32(0.30) and bl_deep.z > bl_deep.x * Float32(3.0))
    assert_true(eval_dielectric_water_fresnel_cpu(Float32(0.05)) > Float32(0.75))

    var agx_dark = eval_agx_tonemap_cpu(Vec3(0.002, 0.003, 0.008))
    var agx_gunmetal = eval_agx_tonemap_cpu(Vec3(0.012, 0.012, 0.014))
    var agx_hi = eval_agx_tonemap_cpu(Vec3(24.0, 18.0, 12.0))
    assert_true(agx_dark.z > Float32(0.0) and agx_dark.z < Float32(0.06))  # Smooth C2 rational toe: zero hard black clip!
    assert_true(agx_gunmetal.x > agx_dark.x * Float32(3.0))
    assert_true(agx_hi.x > Float32(0.85) and agx_hi.x <= Float32(1.0))

    # v3 Alpha-cutout material packing + LEAN water slope-variance transfer + Nubis 3D cumulus density
    var mat_am = AaaMaterial(base_color=Vec3(0.3, 0.6, 0.2), roughness=0.7, metallic=0.0)
    mat_am.alpha_cutoff = Float32(0.48)
    assert_true(mat_am.is_alpha_masked())
    var words_am = List[UInt32]()
    mat_am.append_words(words_am)
    assert_equal(len(words_am), AAA_MATERIAL_WORDS)
    assert_true(bitcast[DType.float32](words_am[3]) < Float32(0.0))

    var a2_near = eval_nyquist_lean_water_alpha2_cpu(Float32(0.02))
    var a2_far = eval_nyquist_lean_water_alpha2_cpu(Float32(5.0))
    assert_true(a2_near < Float32(0.0015) and a2_far > a2_near * Float32(2.5))

    assert_equal(eval_nubis_cloud_density_cpu(Vec3(1.2, 2.4, -0.8), Float32(0.35), Float32(0.0), Vec3()), Float32(0.0))
    var nubis_max = Float32(0.0)
    for k_c in range(16):
        var d_c = eval_nubis_cloud_density_cpu(
            Vec3(Float32(k_c) * Float32(0.47), Float32(2.3), Float32(k_c) * Float32(0.31)),
            Float32(0.32),
            Float32(0.75),
            Vec3(),
        )
        if d_c > nubis_max:
            nubis_max = d_c
    assert_true(nubis_max > Float32(0.02))

    var holo_center = eval_hologram_fresnel_rim_cpu(Float32(0.98), Float32(1.0))
    var holo_rim = eval_hologram_fresnel_rim_cpu(Float32(0.05), Float32(1.0))
    assert_true(holo_rim[0] > holo_center[0] * Float32(3.0) and holo_rim[1] > holo_center[1] * Float32(3.0))
    print("  [PASS] 2. CPU Analytic Models Verified (BSDF Bits, Kaplanyan, LTC, ReBLUR, 16-Halton, Froxels, Water, AgX)")


def test_3_white_furnace_energy_conservation_v2() raises:
    var ren = FlintAaaVulkanRenderer(width=128, height=128)
    assert_true(ren.active)
    ren.furnace_white_dome = True
    ren.exposure = Float32(1.0)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB

    var sb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.5, metallic=0.0)
    sb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.35, 1.35, 1.35), rings=20, segments=32)
    var sph_e = ren.add_mesh(sb^.finish(), 0)

    for met_i in range(2):
        for cc_i in range(2):
            for r_val in [Float32(0.10), Float32(0.45), Float32(0.90)]:
                var fm = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=r_val, metallic=Float32(met_i))
                fm.clearcoat = Float32(cc_i)
                fm.clearcoat_roughness = Float32(0.12)
                var mid = ren.add_material(fm)
                ren.begin_frame(Vec3(0.0, 0.0, -3.2), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
                ren.draw_instance(sph_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=mid)
                assert_true(ren.render_frame())
                var hdr = ren.readback_resolved_hdr()
                var sum_l = Float64(0.0)
                var cnt = 0
                for y in range(44, 84):
                    for x in range(44, 84):
                        var c = hdr[y * 128 + x]
                        sum_l += Float64(c.x + c.y + c.z) / 3.0
                        cnt += 1
                var mean_l = Float32(sum_l / Float64(cnt))
                assert_true(abs(mean_l - Float32(1.0)) <= Float32(0.010))

    ren.shutdown()
    print("  [PASS] 3. White-Furnace Energy Conservation (1.0000 +/- 0.010 across Dielectric, Metal & Clearcoat)")


def test_4_barycentric_textured_ray_hit_and_clearcoat_rt() raises:
    var w = 128
    var h = 128
    var ren = FlintAaaVulkanRenderer(width=w, height=h)
    assert_true(ren.active)
    if not ren.vk.has_ray_query:
        ren.shutdown()
        print("  [PASS] 4. Barycentric Ray-Hit & Clearcoat RT Skipped (No VK_KHR_ray_query on ICD)")
        return

    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.enable_vsm = False
    ren.enable_rt_reflections = True
    ren.enable_rt_shadows = False
    ren.enable_rt_gi = False
    ren.tsr_history_weight = Float32(0.0)
    ren.ibl_intensity = Float32(0.0)
    ren.sun_dir = Vec3(0.0, 0.5, -0.8660254).normalized()
    ren.sun_radiance = Vec3(8.0, 8.0, 8.0)

    # Create a 64x64 BC7 texture: left half (`u < 0.5`) pure Crimson Red, right half (`u >= 0.5`) pure Cyan!
    # Applied to a SINGLE 2-triangle quad wall: triangle-centroid sampling would fail, only true barycentric UV succeeds!
    var stripe_px = List[UInt32](length=64 * 64, fill=0)
    for y in range(64):
        for x in range(64):
            stripe_px[y * 64 + x] = UInt32(0xFFFF0808) if x < 32 else UInt32(0xFF08F0FF)
    var stripe_tex = ren.upload_bc7_texture(stripe_px, 64, 64, srgb=True, generate_mips=True)
    assert_true(stripe_tex >= 0)

    var mat_wall = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.9, metallic=0.0)
    mat_wall.albedo_tex = stripe_tex
    var id_wall = ren.add_material(mat_wall)

    # Dark dielectric car-hood clearcoat material (`metallic=0.0, base_color=0.01, roughness=0.80, clearcoat=1.0, ccoat_r=0.02`)!
    var mat_ccoat_hood = AaaMaterial(base_color=Vec3(0.01, 0.01, 0.01), roughness=0.80, metallic=0.0)
    mat_ccoat_hood.clearcoat = Float32(1.0)
    mat_ccoat_hood.clearcoat_roughness = Float32(0.02)
    var id_hood = ren.add_material(mat_ccoat_hood)

    # Mirror chrome floor (`metallic=1.0, roughness=0.02`)
    var mat_mirror = AaaMaterial(base_color=Vec3(0.98, 0.98, 0.98), roughness=0.02, metallic=1.0)
    var id_mirror = ren.add_material(mat_mirror)

    var wall_m = _make_uv_quad(
        Vec3(-3.0, 0.0, 3.0), Vec3(3.0, 0.0, 3.0), Vec3(3.0, 4.0, 3.0), Vec3(-3.0, 4.0, 3.0),
        Vec3(0.0, 0.0, -1.0),
    )
    var flr_m = _make_uv_quad(
        Vec3(-3.0, 0.0, -2.0), Vec3(3.0, 0.0, -2.0), Vec3(3.0, 0.0, 3.0), Vec3(-3.0, 0.0, 3.0),
        Vec3(0.0, 1.0, 0.0),
    )
    var e_wall = ren.add_mesh(wall_m, id_wall)
    var e_flr = ren.add_mesh(flr_m, id_mirror)

    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))

    # 4a. Mirror floor reflects Red on left (`x=36, y=96`) and Cyan on right (`x=92, y=96`) from the 2-triangle wall!
    ren.begin_frame(Vec3(0.0, 1.4, -2.8), Vec3(0.0, 0.5, 2.0), fov_y_deg=55.0)
    ren.draw_instance(e_wall.mesh_id, ident)
    ren.draw_instance(e_flr.mesh_id, ident, material_override=id_mirror)
    assert_true(ren.render_frame())
    var hdr_mir = ren.readback_resolved_hdr()
    var left_refl = hdr_mir[96 * w + 36]
    var right_refl = hdr_mir[96 * w + 92]
    assert_true(left_refl.x > left_refl.z * Float32(2.0))
    assert_true(right_refl.z > right_refl.x * Float32(2.0))

    # 4b. Dark Dielectric Clearcoat Hood (`roughness=0.80, clearcoat=1.0, ccoat_r=0.02`) ALSO receives RT reflections!
    ren.begin_frame(Vec3(0.0, 0.65, -2.8), Vec3(0.0, 0.25, 2.0), fov_y_deg=55.0)
    ren.draw_instance(e_wall.mesh_id, ident)
    ren.draw_instance(e_flr.mesh_id, ident, material_override=id_hood)
    assert_true(ren.render_frame())
    var hdr_cc = ren.readback_resolved_hdr()
    var cc_left = hdr_cc[96 * w + 36]
    var cc_right = hdr_cc[96 * w + 92]
    assert_true(cc_left.x > cc_left.z * Float32(1.4) and cc_right.z > cc_right.x * Float32(1.4))

    ren.shutdown()
    print("  [PASS] 4. Barycentric Textured Ray-Hit Shading & Dual-Lobe Clearcoat RT Routing Verified")


def test_5_reblur_denoiser_and_top2_hero_local_shadows() raises:
    var w = 128
    var h = 128
    var ren = FlintAaaVulkanRenderer(width=w, height=h)
    assert_true(ren.active)
    if not ren.vk.has_ray_query:
        ren.shutdown()
        print("  [PASS] 5. ReBLUR & Top-2 Hero Local Shadows Skipped (No VK_KHR_ray_query on ICD)")
        return

    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.enable_vsm = False
    ren.ibl_intensity = Float32(0.0)
    ren.sun_radiance = Vec3(0.0, 0.0, 0.0)

    var flr_m = _make_uv_quad(
        Vec3(-4.0, 0.0, -2.0), Vec3(4.0, 0.0, -2.0), Vec3(4.0, 0.0, 6.0), Vec3(-4.0, 0.0, 6.0),
        Vec3(0.0, 1.0, 0.0),
    )
    var blk_m = _make_uv_quad(
        Vec3(0.0, 0.0, -1.0), Vec3(0.0, 0.0, 5.0), Vec3(0.0, 2.5, 5.0), Vec3(0.0, 2.5, -1.0),
        Vec3(-1.0, 0.0, 0.0),
    )
    var mat_matte = ren.add_material(AaaMaterial(base_color=Vec3(0.8, 0.8, 0.8), roughness=0.9, metallic=0.0))
    var mat_glossy = ren.add_material(AaaMaterial(base_color=Vec3(0.85, 0.85, 0.85), roughness=0.22, metallic=1.0))
    var mat_neon_wall = AaaMaterial(base_color=Vec3(0.1, 0.1, 0.1), roughness=0.9, metallic=0.0)
    mat_neon_wall.emissive = Vec3(4.5, 0.8, 0.2)
    var id_neon = ren.add_material(mat_neon_wall)

    var wall_m = _make_uv_quad(
        Vec3(-1.2, 0.0, 5.0), Vec3(1.2, 0.0, 5.0), Vec3(1.2, 3.5, 5.0), Vec3(-1.2, 3.5, 5.0),
        Vec3(0.0, 0.0, -1.0),
    )
    var e_flr = ren.add_mesh(flr_m, mat_matte)
    var e_blk = ren.add_mesh(blk_m, mat_matte)
    var e_wall = ren.add_mesh(wall_m, id_neon)
    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))

    # 5a. Top-2 Hero Local LTC Tube Light on `-X` side (`x = -1.8, y = 1.2, z = 2.0`) blocked from `+X` floor (`x = +1.5`) by wall at `x = 0`!
    ren.enable_rt_shadows = True
    ren.enable_restir_di = False
    ren.enable_rt_reflections = False
    ren.tsr_history_weight = Float32(0.0)
    var hero_tube = AaaClusterLight.tube_ltc(
        Vec3(-1.8, 1.2, 0.8), Vec3(-1.8, 1.2, 3.2), 0.14, Vec3(18.0, 16.0, 12.0), range_radius=14.0
    )
    ren.begin_frame(Vec3(0.0, 3.8, -2.5), Vec3(0.0, 0.0, 2.0), fov_y_deg=55.0)
    ren.add_cluster_light(hero_tube)
    ren.draw_instance(e_flr.mesh_id, ident, material_override=mat_matte)
    ren.draw_instance(e_blk.mesh_id, ident)
    assert_true(ren.render_frame())
    var sh_hdr = ren.readback_resolved_hdr()
    var lit_side = sh_hdr[72 * w + 36]
    var occ_side = sh_hdr[72 * w + 92]
    assert_true(lit_side.x > occ_side.x * Float32(4.0))

    # 5b. Multi-frame Heitz Bounded VNDF + NRD ReBLUR-Lite Specular Denoiser (`roughness = 0.22`)
    ren.enable_rt_reflections = True
    ren.enable_spec_reblur = True
    ren.tsr_history_weight = Float32(0.85)
    for _ in range(5):
        ren.begin_frame(Vec3(0.0, 1.2, -2.5), Vec3(0.0, 0.4, 3.0), fov_y_deg=55.0)
        ren.draw_instance(e_flr.mesh_id, ident, material_override=mat_glossy)
        ren.draw_instance(e_wall.mesh_id, ident)
        assert_true(ren.render_frame())
    var rb_hdr = ren.readback_resolved_hdr()
    var center_gloss = rb_hdr[92 * w + 64]
    assert_true(center_gloss.x > Float32(0.15) and center_gloss.x < Float32(20.0))

    ren.shutdown()
    print("  [PASS] 5. Top-2 Hero Local LTC Shadows & Multi-Frame ReBLUR Glossy VNDF Denoiser Verified")


def test_6_splat_terrain_interior_parallax_wetness_and_toksvig() raises:
    var w = 128
    var h = 128
    var ren = FlintAaaVulkanRenderer(width=w, height=h)
    assert_true(ren.active)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.enable_vsm = False
    ren.tsr_history_weight = Float32(0.0)
    ren.sun_dir = Vec3(0.0, 0.85, -0.526).normalized()
    ren.sun_radiance = Vec3(3.2, 3.2, 3.2)
    ren.ibl_intensity = Float32(0.2)

    # 6a. Upload Toksvig coupled BC5 Normal + BC7 ORM mip chain
    var nrms = List[Vec3](capacity=32 * 32)
    var orms = List[UInt32](capacity=32 * 32)
    for y in range(32):
        for x in range(32):
            var sx = Float32(0.45) if ((x + y) & 1) == 0 else Float32(-0.45)
            nrms.append(Vec3(sx, Float32(0.0), Float32(0.893)).normalized())
            orms.append(UInt32(0xFFFF1800))  # AO=255, Rough=24 (smooth), Metal=0
    var tok_slots = ren.upload_toksvig_normal_orm(nrms, orms, 32, 32, toksvig_scale=1.0)
    assert_true(tok_slots[0] >= 0 and tok_slots[1] >= 0)

    # 6b. 4-Layer Height-Modulated Terrain Splatting (`(t_alb + k*2, t_nrm + k*2)` for k in 0..3)
    var flat_nrm = List[Vec3](length=16 * 16, fill=Vec3(0.0, 0.0, 1.0))
    var layer_colors = [
        UInt32(0x9020D020),  # k=0 Grass (Green, height=0.56)
        UInt32(0x98E0C050),  # k=1 Sand  (Gold, height=0.60)
        UInt32(0xA0707078),  # k=2 Rock  (Grey, height=0.63)
        UInt32(0xA8D03020),  # k=3 Cobble (Terracotta Red, height=0.66)
    ]
    var base_alb_slot = -1
    var base_nrm_slot = -1
    for k in range(4):
        var px_k = List[UInt32](length=16 * 16, fill=layer_colors[k])
        var sa = ren.upload_bc7_texture(px_k, 16, 16, srgb=True, generate_mips=False)
        var sn = ren.upload_bc5_normal_texture(flat_nrm, 16, 16, generate_mips=False)
        if k == 0:
            base_alb_slot = sa
            base_nrm_slot = sn
    var mat_splat = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.85, metallic=0.0)
    mat_splat.albedo_tex = base_alb_slot
    mat_splat.normal_tex = base_nrm_slot
    var id_splat = ren.add_material(mat_splat)

    var terr_q = _make_uv_quad(
        Vec3(-3.0, 0.0, -1.0), Vec3(3.0, 0.0, -1.0), Vec3(3.0, 0.0, 4.0), Vec3(-3.0, 0.0, 4.0),
        Vec3(0.0, 1.0, 0.0),
    )
    # Left vertices (v0, v3): Grass (`u24 = 0`), Right vertices (v1, v2): Cobble (`w_cobble = 255 -> 0xFF0000`)!
    var splat_w = List[UInt32]()
    splat_w.append(UInt32(0x000000))
    splat_w.append(UInt32(0xFF0000))
    splat_w.append(UInt32(0xFF0000))
    splat_w.append(UInt32(0x000000))
    var e_terr = ren.upload_splat_mesh(terr_q, id_splat, splat_w)

    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    ren.begin_frame(Vec3(0.0, 3.2, -2.2), Vec3(0.0, 0.0, 1.5), fov_y_deg=55.0)
    ren.draw_instance(e_terr.mesh_id, ident)
    assert_true(ren.render_frame())
    var terr_hdr = ren.readback_resolved_hdr()
    var grass_px = terr_hdr[72 * w + 22]
    var cobble_px = terr_hdr[72 * w + 106]
    assert_true(grass_px.y > grass_px.x * Float32(1.8))
    assert_true(cobble_px.x > cobble_px.y * Float32(1.8))

    # 6c. GPU Ray-Box Interior Parallax (`AAA_MAT_INTERIOR_PARALLAX`) produces lit interior rooms at night
    var room_px = List[UInt32](length=32 * 32, fill=UInt32(0xFFFFD8A8))
    var room_slot = ren.upload_bc7_texture(room_px, 32, 32, srgb=True, generate_mips=False)
    assert_true(room_slot >= 0 and room_slot < 63)
    var mat_win = AaaMaterial(base_color=Vec3(0.06, 0.08, 0.10), roughness=0.06, metallic=0.0)
    mat_win.interior_parallax_enabled = True
    mat_win.detail_normal_tex = room_slot
    mat_win.detail_uv_scale = Float32(4.0)
    var id_win = ren.add_material(mat_win)

    var win_q = _make_uv_quad(
        Vec3(-2.5, -2.0, 2.0), Vec3(2.5, -2.0, 2.0), Vec3(2.5, 2.0, 2.0), Vec3(-2.5, 2.0, 2.0),
        Vec3(0.0, 0.0, -1.0),
    )
    var e_win = ren.add_mesh(win_q, id_win)
    ren.sun_radiance = Vec3(0.0, 0.0, 0.0)
    ren.ibl_intensity = Float32(0.0)
    ren.begin_frame(Vec3(0.0, 0.0, -2.0), Vec3(0.0, 0.0, 2.0), fov_y_deg=50.0)
    ren.draw_instance(e_win.mesh_id, ident)
    assert_true(ren.render_frame())
    var win_hdr = ren.readback_resolved_hdr()
    var sum_win = Float32(0.0)
    for py in range(32, 96):
        for px in range(32, 96):
            sum_win += win_hdr[py * w + px].x + win_hdr[py * w + px].y
    assert_true(sum_win > Float32(25.0))

    # 6d. ForzaTech Physical Road Wetness (`road_wetness_mode = True`): wet porous asphalt is darker via Lekner-Dorf TIR!
    ren.sun_radiance = Vec3(3.0, 3.0, 3.0)
    var mat_road = AaaMaterial(base_color=Vec3(0.32, 0.31, 0.30), roughness=0.82, metallic=0.0)
    mat_road.road_wetness_mode = True
    mat_road.detail_uv_scale = Float32(24.0)
    var id_road = ren.add_material(mat_road)
    var e_road = ren.add_mesh(
        _make_uv_quad(Vec3(-3.0, 0.0, -1.0), Vec3(3.0, 0.0, -1.0), Vec3(3.0, 0.0, 4.0), Vec3(-3.0, 0.0, 4.0), Vec3(0.0, 1.0, 0.0)),
        id_road,
    )
    ren.weather_wetness = Float32(0.0)
    ren.begin_frame(Vec3(0.0, 3.5, -2.0), Vec3(0.0, 0.0, 1.5), fov_y_deg=50.0)
    ren.draw_instance(e_road.mesh_id, ident)
    assert_true(ren.render_frame())
    var dry_hdr = ren.readback_resolved_hdr()[64 * w + 64]

    ren.weather_wetness = Float32(0.80)
    ren.rain_intensity = Float32(0.65)
    ren.begin_frame(Vec3(0.0, 3.5, -2.0), Vec3(0.0, 0.0, 1.5), fov_y_deg=50.0)
    ren.draw_instance(e_road.mesh_id, ident)
    assert_true(ren.render_frame())
    var wet_hdr = ren.readback_resolved_hdr()[64 * w + 64]
    assert_true(wet_hdr.x < dry_hdr.x * Float32(0.65))

    ren.shutdown()
    print("  [PASS] 6. 4-Layer Terrain Splatting, Interior Parallax, ForzaTech Wetness & Toksvig Mips Verified")


def test_7_froxel_volumetrics_rain_and_beer_lambert_water() raises:
    var w = 128
    var h = 128
    var ren = FlintAaaVulkanRenderer(width=w, height=h)
    assert_true(ren.active)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.tsr_history_weight = Float32(0.0)
    ren.ibl_intensity = Float32(0.05)
    ren.sun_dir = Vec3(0.15, 0.55, 0.82).normalized()
    ren.sun_radiance = Vec3(3.5, 3.4, 3.2)

    # 7a. 160x90x64 Froxel Volumetric Fog + VSM + Spot Searchlight + GPU 3D Rain
    var id_black = ren.add_material(AaaMaterial(base_color=Vec3(0.01, 0.01, 0.01), roughness=0.95, metallic=0.0))
    var e_bg = ren.add_mesh(
        _make_uv_quad(Vec3(-20.0, -10.0, 35.0), Vec3(20.0, -10.0, 35.0), Vec3(20.0, 15.0, 35.0), Vec3(-20.0, 15.0, 35.0), Vec3(0.0, 0.0, -1.0)),
        id_black,
    )
    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))

    ren.enable_froxel_fog = False
    ren.begin_frame(Vec3(0.0, 1.5, -4.0), Vec3(0.0, 1.5, 20.0), fov_y_deg=55.0)
    ren.draw_instance(e_bg.mesh_id, ident)
    assert_true(ren.render_frame())
    var no_fog_c = ren.readback_resolved_hdr()[64 * w + 64]

    ren.enable_froxel_fog = True
    ren.fog_density = Float32(0.035)
    ren.enable_gpu_rain = True
    ren.rain_intensity = Float32(0.80)
    var spot = AaaClusterLight.spot(
        Vec3(0.0, 5.0, 10.0), Vec3(0.0, -1.0, 0.0), Vec3(40.0, 90.0, 120.0), 24.0, inner_deg=18.0, outer_deg=38.0,
    )
    for _ in range(3):
        ren.begin_frame(Vec3(0.0, 1.5, -4.0), Vec3(0.0, 1.5, 20.0), fov_y_deg=55.0)
        ren.add_cluster_light(spot)
        ren.draw_instance(e_bg.mesh_id, ident)
        assert_true(ren.render_frame())
    var fog_c = ren.readback_resolved_hdr()[64 * w + 64]
    assert_true((fog_c.x + fog_c.y + fog_c.z) > (no_fog_c.x + no_fog_c.y + no_fog_c.z) + Float32(0.05))

    # 7b. Dielectric Beer-Lambert Water Pass over sloped white sand lakebed (`y = -0.25` left -> `y = -3.8` right)
    ren.enable_froxel_fog = False
    ren.enable_gpu_rain = False
    ren.rain_intensity = Float32(0.0)
    ren.enable_vsm = False
    ren.enable_water_plane = True
    ren.water_level_y = Float32(0.0)
    ren.water_absorption_rgb = Vec3(0.55, 0.10, 0.03)

    var id_sand = ren.add_material(AaaMaterial(base_color=Vec3(0.92, 0.90, 0.85), roughness=0.9, metallic=0.0))
    var bed_q = _make_uv_quad(
        Vec3(-4.0, -0.25, -1.0), Vec3(4.0, -3.80, -1.0), Vec3(4.0, -3.80, 6.0), Vec3(-4.0, -0.25, 6.0),
        Vec3(0.40, 0.916, 0.0).normalized(),
    )
    var e_bed = ren.add_mesh(bed_q, id_sand)
    ren.begin_frame(Vec3(0.0, 4.5, -2.5), Vec3(0.0, -1.0, 2.5), fov_y_deg=55.0)
    ren.draw_instance(e_bed.mesh_id, ident)
    assert_true(ren.render_frame())
    var wat_hdr = ren.readback_resolved_hdr()
    var shallow_px = wat_hdr[64 * w + 28]
    var deep_px = wat_hdr[64 * w + 100]
    var shallow_br = shallow_px.z / max(Float32(1e-4), shallow_px.x)
    var deep_br = deep_px.z / max(Float32(1e-4), deep_px.x)
    assert_true(deep_br > shallow_br * Float32(1.6))

    ren.shutdown()
    print("  [PASS] 7. 160x90x64 Froxel Volumetrics + GPU 3D Rain & Dielectric Beer-Lambert Water Verified")


def test_8_tsr_v3_halton16_motion_blur_exposure_agx_and_hologram_oit() raises:
    var w = 128
    var h = 128
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=0.75)
    assert_true(ren.active)
    ren.display_mode = AAA_DISPLAY_AGX_REC709
    ren.tsr_history_weight = Float32(0.85)
    ren.bloom_intensity = Float32(0.15)
    ren.bloom_threshold = Float32(0.8)
    ren.glare_streak_strength = Float32(0.65)
    ren.enable_motion_blur = True
    ren.motion_blur_shutter = Float32(0.5)
    ren.enable_auto_exposure = True
    ren.auto_exposure_speed = Float32(1.0)
    ren.min_auto_exposure = Float32(0.25)
    ren.max_auto_exposure = Float32(1.75)
    ren.windshield_rain = Float32(0.55)

    # 8a. Verify Bounded Auto-Exposure never exceeds `max_auto_exposure = 1.75` even in pitch-dark night!
    ren.sun_radiance = Vec3(0.0, 0.0, 0.0)
    ren.ibl_intensity = Float32(0.001)
    var id_dark = ren.add_material(AaaMaterial(base_color=Vec3(0.02, 0.02, 0.03), roughness=0.9, metallic=0.0))
    var sb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.5, metallic=0.0)
    sb.add_ellipsoid(Vec3(0.0, 0.0, 4.0), Vec3(1.2, 1.2, 1.2), rings=12, segments=16)
    var e_box = ren.add_mesh(sb^.finish(), id_dark)

    # Run 16-phase Halton TSR v3 + moving camera motion blur + windshield rain + glare
    for f_i in range(16):
        var cx = Float32(f_i) * Float32(0.15)
        ren.begin_frame(Vec3(cx, 0.5, -3.0), Vec3(cx, 0.0, 4.0), fov_y_deg=55.0)
        ren.draw_instance(e_box.mesh_id, trs(Vec3(cx, 0.0, 4.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        assert_true(ren.render_frame())

    var ae = ren.get_adapted_exposure()
    assert_true(ae[0] >= Float32(0.24) and ae[0] <= Float32(1.76))

    # 8b. Fresnel Rim Hologram OIT order-independence (`A->B == B->A` to the exact bit!)
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_motion_blur = False
    ren.enable_auto_exposure = False
    ren.windshield_rain = Float32(0.0)
    ren.bloom_intensity = Float32(0.0)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB

    var holo_a = AaaVolumetricPlume.hologram(Vec3(-0.2, 0.0, 4.2), 1.2, Vec3(0.1, 0.9, 1.0), density=0.65, emission=Vec3(0.4, 2.5, 3.8))
    var holo_b = AaaVolumetricPlume.hologram(Vec3(0.2, 0.1, 5.0), 1.3, Vec3(1.0, 0.3, 0.8), density=0.60, emission=Vec3(3.2, 0.5, 2.0))

    ren.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 4.5), fov_y_deg=50.0)
    ren.draw_volumetric_plume(holo_a)
    ren.draw_volumetric_plume(holo_b)
    assert_true(ren.render_frame())
    var px_ab = ren.readback_pixels()

    ren.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 4.5), fov_y_deg=50.0)
    ren.draw_volumetric_plume(holo_b)
    ren.draw_volumetric_plume(holo_a)
    assert_true(ren.render_frame())
    var px_ba = ren.readback_pixels()

    var diff = 0
    for i in range(len(px_ab)):
        if px_ab[i] != px_ba[i]:
            diff += 1
    assert_equal(diff, 0)

    ren.shutdown()
    print("  [PASS] 8. FlintTSR v3 16-Halton, 180-Deg Motion Blur, Bounded Exposure, Glare, Rain & Hologram OIT Verified")


def test_9_alphamask_visbuffer_vsm_and_nonopaque_rayquery() raises:
    var w = 128
    var h = 128
    var ren = FlintAaaVulkanRenderer(width=w, height=h)
    assert_true(ren.active)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.tsr_history_weight = Float32(0.0)
    ren.ibl_intensity = Float32(0.02)
    ren.sun_dir = Vec3(0.0, 0.0, -1.0)
    ren.sun_radiance = Vec3(4.0, 4.0, 4.0)
    ren.enable_vsm = True
    ren.enable_rt_shadows = True

    # Build 64x64 BC7 alpha-cutout texture: left half `u < 0.5` is transparent (`alpha = 0`), right half `u >= 0.5` is opaque green (`alpha = 255`)
    var grate_px = List[UInt32](length=64 * 64, fill=0)
    for gy in range(64):
        for gx in range(64):
            grate_px[gy * 64 + gx] = UInt32(0x0020D030) if gx < 32 else UInt32(0xFF20D030)
    var grate_slot = ren.upload_bc7_texture(grate_px, 64, 64, srgb=False, generate_mips=True)
    assert_true(grate_slot >= 0)

    var mat_red = AaaMaterial(base_color=Vec3(0.90, 0.08, 0.06), roughness=0.8, metallic=0.0)
    var id_red = ren.add_material(mat_red)
    var e_wall = ren.add_mesh(
        _make_uv_quad(Vec3(-4.0, -4.0, 5.0), Vec3(4.0, -4.0, 5.0), Vec3(4.0, 4.0, 5.0), Vec3(-4.0, 4.0, 5.0), Vec3(0.0, 0.0, -1.0)),
        id_red,
    )

    var mat_grate = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.7, metallic=0.0)
    mat_grate.albedo_tex = grate_slot
    mat_grate.alpha_cutoff = Float32(0.48)
    var id_grate = ren.add_material(mat_grate)
    var grate_mesh = _make_uv_quad(
        Vec3(-2.0, -2.0, 2.5), Vec3(2.0, -2.0, 2.5), Vec3(2.0, 2.0, 2.5), Vec3(-2.0, 2.0, 2.5), Vec3(0.0, 0.0, -1.0),
    )
    grate_mesh.alpha_mode = 1
    var e_grate = ren.add_mesh(grate_mesh, id_grate)

    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    # Render 2 frames so VSM pages warm up and rasterize with `pipe_vsm_alphamask_gfx`!
    for _ in range(2):
        ren.begin_frame(Vec3(-1.4, 0.0, -2.5), Vec3(0.2, 0.0, 5.0), fov_y_deg=50.0)
        ren.draw_instance(e_wall.mesh_id, ident)
        ren.draw_instance(e_grate.mesh_id, ident)
        assert_true(ren.render_frame())

    var hdr = ren.readback_resolved_hdr()
    # Left side looks through `alpha = 0` cutout straight onto sunlit red wall (`VSM` + `Non-Opaque RayQuery` both pass through!)
    var px_cutout_lit_wall = hdr[64 * w + 38]
    # Center-right hits solid green `alpha = 255` half of the alpha-masked quad
    var px_solid_green_grate = hdr[64 * w + 76]
    # Far right looks past the grate edge onto the red wall shadowed by the `alpha = 255` half (`sun_dir = (0,0,-1)` -> `+Z` shadow!)
    ren.sun_dir = Vec3(-0.32, 0.0, -0.947).normalized()
    ren.vsm.invalidate_all()
    for _ in range(2):
        ren.begin_frame(Vec3(0.0, 0.0, -2.5), Vec3(0.0, 0.0, 5.0), fov_y_deg=55.0)
        ren.draw_instance(e_wall.mesh_id, ident)
        ren.draw_instance(e_grate.mesh_id, ident)
        assert_true(ren.render_frame())
    var hdr2 = ren.readback_resolved_hdr()
    var wall_through_hole = hdr2[64 * w + 46]     # `x < 0` hole -> sees sunlit red wall!
    var wall_behind_solid = hdr2[64 * w + 106]    # `x > 2.0` wall receiving oblique shadow from `x in [0..2]` solid half!

    assert_true(px_cutout_lit_wall.x > px_cutout_lit_wall.y * Float32(3.5))
    assert_true(px_solid_green_grate.y > px_solid_green_grate.x * Float32(2.5))
    assert_true(wall_through_hole.x > wall_behind_solid.x * Float32(2.2))

    ren.shutdown()
    print("  [PASS] 9. Two-Bucket Alpha-Cutout (VisBuffer + VSM + Non-Opaque RayQuery Transparency) Verified")


def test_10_nubis_clouds_starfield_vertex_color_fireballs_and_gpu_particles() raises:
    var w = 128
    var h = 128
    var ren = FlintAaaVulkanRenderer(width=w, height=h)
    assert_true(ren.active)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.tsr_history_weight = Float32(0.0)

    # 10a. 24-Bit sRGB Vertex Color (`vertex_words[11][8..31]`) decoded in `aaa_deferred_shade_shader`
    var vcol_mesh = _make_uv_quad(
        Vec3(-2.0, -2.0, 3.0), Vec3(2.0, -2.0, 3.0), Vec3(2.0, 2.0, 3.0), Vec3(-2.0, 2.0, 3.0), Vec3(0.0, 0.0, -1.0),
    )
    var id_white = ren.add_material(AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.8, metallic=0.0))
    var e_vc = ren.add_mesh(vcol_mesh, id_white)
    # Directly stamp 24-bit sRGB emerald `(R=24, G=245, B=36)` into `vertex_words[11][8..31]` (`ao_u6 = 63` in `bits 2..7`)
    var emerald_meta = (UInt32(63) << 2) | (UInt32(24) << 8) | (UInt32(245) << 16) | (UInt32(36) << 24)
    for v_i in range(e_vc.vertex_count):
        var vw_idx = (e_vc.first_vertex + v_i) * 12 + 11
        ren.geom.vertex_words[vw_idx] = emerald_meta
    from flint.platform.vulkan import put32
    for v_i in range(e_vc.vertex_count):
        var vw_idx = (e_vc.first_vertex + v_i) * 12 + 11
        put32(ren.buf_vertices.cpu_address + vw_idx * 4, emerald_meta)

    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    ren.sun_dir = Vec3(0.0, 0.2, -0.98).normalized()
    ren.sun_radiance = Vec3(3.0, 3.0, 3.0)
    ren.begin_frame(Vec3(0.0, 0.0, -1.5), Vec3(0.0, 0.0, 3.0), fov_y_deg=50.0)
    ren.draw_instance(e_vc.mesh_id, ident)
    assert_true(ren.render_frame())
    var vc_px = ren.readback_resolved_hdr()[64 * w + 64]
    assert_true(vc_px.y > vc_px.x * Float32(8.0) and vc_px.y > vc_px.z * Float32(5.0))

    # 10b. Nubis 3D Perlin-Worley Volumetric Cloudscape + Night Blackbody Starfield & Cratered Moon
    var atmo = BrunetonAtmosphereParams()
    atmo.sun_dir = Vec3(0.25, -0.25, 0.935).normalized()
    atmo.cloud_coverage = Float32(0.58)
    atmo.star_intensity = Float32(1.0)
    _ = ren.set_bruneton_atmosphere(atmo)
    ren.begin_frame(Vec3(0.0, 1.5, 0.0), Vec3(0.0, 4.5, 8.0), fov_y_deg=65.0)
    assert_true(ren.render_frame())
    var sky_hdr = ren.readback_resolved_hdr()
    var max_sky = Float32(0.0)
    for sp in sky_hdr:
        var lum_s = sp.x + sp.y + sp.z
        if lum_s > max_sky:
            max_sky = lum_s
    assert_true(max_sky > Float32(0.02))

    # 10c. 4-Step Pyroclastic Fireball Plumes (exact OIT order-independence!) + `64K` GPU Compute Particles
    ren.cloud_coverage = Float32(0.0)
    ren.star_intensity = Float32(0.0)
    ren.ibl_intensity = Float32(0.02)
    ren.sun_radiance = Vec3(0.0, 0.0, 0.0)
    var fb_a = AaaVolumetricPlume(Vec3(-0.25, 0.0, 4.5), 1.3, Vec3(0.25, 0.22, 0.20), density=1.1, emission=Vec3(9.5, 3.8, 0.6))
    var fb_b = AaaVolumetricPlume(Vec3(0.25, 0.1, 5.1), 1.4, Vec3(0.22, 0.20, 0.18), density=0.95, emission=Vec3(6.5, 2.2, 0.4))

    ren.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), fov_y_deg=55.0)
    ren.draw_volumetric_plume(fb_a)
    ren.draw_volumetric_plume(fb_b)
    assert_true(ren.render_frame())
    var fb_ab = ren.readback_pixels()

    ren.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), fov_y_deg=55.0)
    ren.draw_volumetric_plume(fb_b)
    ren.draw_volumetric_plume(fb_a)
    assert_true(ren.render_frame())
    var fb_ba = ren.readback_pixels()
    var fb_diff = 0
    for i in range(len(fb_ab)):
        if fb_ab[i] != fb_ba[i]:
            fb_diff += 1
    assert_equal(fb_diff, 0)

    # Submit all 4 GPU compute particle archetypes (`SPARK_STREAK`, `WATER_SPRAY`, `DRIFT_SMOKE`, `EXHAUST_PLASMA`)
    ren.enable_froxel_fog = True
    ren.fog_density = Float32(0.015)
    ren.begin_frame(Vec3(0.0, 0.5, -2.0), Vec3(0.0, 0.5, 4.0), fov_y_deg=60.0)
    assert_true(ren.render_frame())
    var base_sum = Float32(0.0)
    for bp in ren.readback_resolved_hdr():
        base_sum += bp.x + bp.y + bp.z

    for _ in range(3):
        ren.begin_frame(Vec3(0.0, 0.5, -2.0), Vec3(0.0, 0.5, 4.0), fov_y_deg=60.0)
        ren.submit_particle_emitter(AaaParticleEmitter.sparks(Vec3(-0.4, 0.5, 3.2), Vec3(0.5, 2.5, 0.0), spawn_count=128))
        ren.submit_particle_emitter(AaaParticleEmitter.exhaust_plasma(Vec3(0.4, 0.4, 3.0), Vec3(0.0, 0.0, -8.0), spawn_count=64))
        ren.submit_particle_emitter(AaaParticleEmitter.water_spray(Vec3(-0.2, 0.2, 3.5), Vec3(-0.4, 1.2, 0.0), spawn_count=48))
        ren.submit_particle_emitter(AaaParticleEmitter.drift_smoke(Vec3(0.2, 0.3, 3.8), Vec3(0.2, 0.8, 0.0), spawn_count=48))
        assert_true(ren.render_frame())

    var part_sum = Float32(0.0)
    for pp in ren.readback_resolved_hdr():
        part_sum += pp.x + pp.y + pp.z
    assert_true(part_sum > base_sum + Float32(8.0))

    ren.shutdown()
    print("  [PASS] 10. 24-Bit Vertex Colors, Nubis 3D Clouds + Starfield/Moon, Pyroclastic Fireballs & 64K GPU Particles Verified")


def test_11_v4_skinning_arena_qem_articulated_mv_cofactor_tbn_and_wind() raises:
    # 11a. Expanded 2,048-Bone / 98,304-Vertex Skinning Arena & SPIR-V Validation
    assert_equal(AAA_MAX_SKIN_BONES, 2048)
    assert_equal(AAA_SKIN_CUR_BONES_BASE, 0)
    assert_equal(AAA_SKIN_PREV_BONES_BASE, 16384)
    assert_equal(AAA_SKIN_BIND_VERTS_BASE, 32768)
    assert_equal(AAA_SKIN_MAX_VERTS, 98304)
    assert_equal(AAA_SKIN_PREV_POS_BASE, 32768 + 98304 * AAA_SKIN_BIND_STRIDE_WORDS)
    assert_equal(AAA_SKIN_BUFFER_WORDS, AAA_SKIN_PREV_POS_BASE + 98304 * 3)
    _assert_spv_valid("aaa_dq_skinning_shader", aaa_dq_skinning_shader(BIND_AAA_VERTICES, BIND_AAA_SKINNING))

    # 11b. Dominant-Pivot Weight Sort, 4-Weight Normalization (Zero `j3=0` Root Leak) & Uniform Bone Scale
    var ns1 = normalize_and_sort_skin_weights_4(5, 0, 0, 0, Float32(0.35), Float32(0.0), Float32(0.0), Float32(0.0))
    assert_equal(ns1[0] & UInt32(0xFF), UInt32(5))
    assert_true(abs(ns1[1] - Float32(1.0)) < Float32(1e-6))
    assert_equal(ns1[2], Float32(0.0))
    assert_equal(ns1[3], Float32(0.0))
    assert_equal(Float32(1.0) - (ns1[1] + ns1[2] + ns1[3]), Float32(0.0))

    # 2-bone unsorted & unnormalized `(j0=4, w=0.15), (j1=7, w=0.45)` -> `j0=7 (0.75), j1=4 (0.25)` with zero leak
    var ns2 = normalize_and_sort_skin_weights_4(4, 7, 0, 0, Float32(0.15), Float32(0.45), Float32(0.0), Float32(0.0))
    assert_equal(ns2[0] & UInt32(0xFF), UInt32(7))
    assert_equal((ns2[0] >> UInt32(8)) & UInt32(0xFF), UInt32(4))
    assert_true(abs(ns2[1] - Float32(0.75)) < Float32(1e-6))
    assert_true(abs(ns2[2] - Float32(0.25)) < Float32(1e-6))
    assert_equal(ns2[3], Float32(0.0))
    assert_equal(Float32(1.0) - (ns2[1] + ns2[2] + ns2[3]), Float32(0.0))

    var leak_bones = List[DualQuat](length=8, fill=DualQuat())
    # Move root bone 0 far away `(500, -500, 500)`; 1-bone/2-bone vertices on bones 4,5,7 must NOT leak onto bone 0!
    leak_bones[0] = DualQuat.from_quat_trans(Quat(), Vec3(500.0, -500.0, 500.0))
    leak_bones[4] = DualQuat.from_quat_trans(Quat(), Vec3(0.0, 4.0, 0.0))
    leak_bones[5] = DualQuat.from_quat_trans(Quat(), Vec3(0.0, 3.0, 0.0))
    leak_bones[7] = DualQuat.from_quat_trans(Quat(), Vec3(0.0, 8.0, 0.0))
    var sk1_cpu = skin_vertex_dq_cpu(
        leak_bones, 5, 0, 0, 0, Float32(0.35), Float32(0.0), Float32(0.0), Float32(0.0), Vec3(1.0, 0.0, 0.0)
    )
    assert_true((sk1_cpu[0] - Vec3(1.0, 3.0, 0.0)).length() < Float32(1e-4))
    var sk2_cpu = skin_vertex_dq_cpu(
        leak_bones, 4, 7, 0, 0, Float32(0.15), Float32(0.45), Float32(0.0), Float32(0.0), Vec3(0.0, 0.0, 0.0)
    )
    assert_true((sk2_cpu[0] - Vec3(0.0, 7.0, 0.0)).length() < Float32(1e-4))

    # Uniform bone scale in `DualQuat.from_mat4`
    var q_rot = Quat(0.0, Float32(0.70710678), 0.0, Float32(0.70710678)).normalized()
    var m_scaled = trs(Vec3(3.0, -2.0, 5.0), q_rot, Vec3(2.5, 2.5, 2.5))
    var dq_scaled = DualQuat.from_mat4(m_scaled)
    var p_test = Vec3(2.0, 1.0, 0.0)
    var p_dq = dq_scaled.transform_point(p_test)
    var p_m = Vec3(
        m_scaled.values[0] * p_test.x + m_scaled.values[4] * p_test.y + m_scaled.values[8] * p_test.z + m_scaled.values[12],
        m_scaled.values[1] * p_test.x + m_scaled.values[5] * p_test.y + m_scaled.values[9] * p_test.z + m_scaled.values[13],
        m_scaled.values[2] * p_test.x + m_scaled.values[6] * p_test.y + m_scaled.values[10] * p_test.z + m_scaled.values[14],
    )
    assert_true((p_dq - p_m).length() < Float32(1e-3))

    # 11c. QEM Skin-LOD Top-4 Joint/Weight Blending & Renormalization (`sum == 1.0`)
    var grid_m = NativeMesh()
    comptime GX: Int = 9
    comptime GY: Int = 9
    for gy in range(GY):
        for gx in range(GX):
            grid_m.positions.append(Float32(gx) * Float32(0.25))
            grid_m.positions.append(Float32(gy) * Float32(0.25))
            grid_m.positions.append(Float32(0.0))
            grid_m.normals.append(Float32(0.0)); grid_m.normals.append(Float32(0.0)); grid_m.normals.append(Float32(1.0))
            grid_m.texcoords.append(Float32(gx) / Float32(GX - 1))
            grid_m.texcoords.append(Float32(gy) / Float32(GY - 1))
            var b_base = UInt16((gx + gy) % 6)
            grid_m.joints.append(b_base)
            grid_m.joints.append((b_base + UInt16(1)) % UInt16(8))
            grid_m.joints.append((b_base + UInt16(2)) % UInt16(8))
            grid_m.joints.append((b_base + UInt16(3)) % UInt16(8))
            grid_m.weights.append(Float32(0.45))
            grid_m.weights.append(Float32(0.25))
            grid_m.weights.append(Float32(0.20))
            grid_m.weights.append(Float32(0.10))
    for gy in range(GY - 1):
        for gx in range(GX - 1):
            var i0 = gy * GX + gx
            var i1 = i0 + 1
            var i2 = i0 + GX + 1
            var i3 = i0 + GX
            grid_m.indices.append(i0); grid_m.indices.append(i1); grid_m.indices.append(i2)
            grid_m.indices.append(i0); grid_m.indices.append(i2); grid_m.indices.append(i3)
    var qem_sk = simplify_mesh_qem(grid_m, 32)
    var sv_cnt = qem_sk[0].vertex_count()
    assert_true(qem_sk[0].triangle_count() <= 40 and sv_cnt > 0)
    assert_equal(len(qem_sk[0].joints), sv_cnt * 4)
    assert_equal(len(qem_sk[0].weights), sv_cnt * 4)
    for sv in range(sv_cnt):
        var sw0 = qem_sk[0].weights[sv * 4 + 0]
        var sw1 = qem_sk[0].weights[sv * 4 + 1]
        var sw2 = qem_sk[0].weights[sv * 4 + 2]
        var sw3 = qem_sk[0].weights[sv * 4 + 3]
        assert_true(sw0 >= sw1 - Float32(1e-5) and sw1 >= sw2 - Float32(1e-5) and sw2 >= sw3 - Float32(1e-5) and sw3 >= Float32(0.0))
        assert_true(abs((sw0 + sw1 + sw2 + sw3) - Float32(1.0)) < Float32(1e-5))

    # 11d. Cofactor Normal Matrix + Gram-Schmidt Orthonormal TBN under extreme non-uniform scale
    var m_nonuni = trs(Vec3(0.0, 0.0, 0.0), Quat(0.0, 0.0, Float32(0.3826834), Float32(0.9238795)), Vec3(4.0, 0.25, 1.5))
    var ln_test = Vec3(0.6, 0.8, 0.0).normalized()
    var lt_test = Vec3(-0.8, 0.6, 0.2).normalized()
    var tbn = eval_cofactor_gram_schmidt_tbn_cpu(m_nonuni, ln_test, lt_test, Float32(1.0))
    assert_true(abs(tbn[0].length() - Float32(1.0)) < Float32(1e-5))
    assert_true(abs(tbn[1].length() - Float32(1.0)) < Float32(1e-5))
    assert_true(abs(tbn[2].length() - Float32(1.0)) < Float32(1e-5))
    assert_true(abs(tbn[0].dot(tbn[1])) < Float32(1e-5))
    assert_true(abs(tbn[0].dot(tbn[2])) < Float32(1e-5))
    assert_true(abs(tbn[1].dot(tbn[2])) < Float32(1e-5))
    # Verify surface tangent vector `M * (-0.8, 0.6, 0)` is strictly orthogonal to cofactor world normal `tbn[0]`!
    var surf_edge_l = Vec3(-0.8, 0.6, 0.0)
    var surf_edge_w = Vec3(
        m_nonuni.values[0] * surf_edge_l.x + m_nonuni.values[4] * surf_edge_l.y,
        m_nonuni.values[1] * surf_edge_l.x + m_nonuni.values[5] * surf_edge_l.y,
        m_nonuni.values[2] * surf_edge_l.x + m_nonuni.values[6] * surf_edge_l.y,
    )
    assert_true(abs(tbn[0].dot(surf_edge_w.normalized())) < Float32(1e-5))

    # 11e. GPU Hardware Verification: Auto-Allocated Bone Palettes, Zero Root Leak, Zero Halton Static Velocity,
    #      `draw_native_model_articulated` Sub-Node Motion Vectors & GPU Particle Ambient Wind Advection!
    var w = 128
    var h = 128
    var ren = FlintAaaVulkanRenderer(width=w, height=h)
    assert_true(ren.active)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.enable_vsm = False
    ren.svgf_passes = 0
    ren.tsr_history_weight = Float32(0.85)  # Active 16-phase Halton TSR sub-pixel jitter!

    # Auto-allocated `bone_palette_offset` never overwrites slot 0, and GPU compute skinning has 0 root-bone leak
    var q_sk = _make_uv_quad(
        Vec3(-1.5, -1.5, 0.0), Vec3(1.5, -1.5, 0.0), Vec3(1.5, 1.5, 0.0), Vec3(-1.5, 1.5, 0.0), Vec3(0.0, 0.0, -1.0)
    )
    var jp_test = List[UInt32](length=4, fill=UInt32(2))
    var jw_unnorm = List[Vec3](length=4, fill=Vec3(0.40, 0.0, 0.0))  # Unnormalized 1-bone weight -> must NOT leak 0.60 onto bone 0!
    var id_wht = ren.add_material(AaaMaterial(base_color=Vec3(0.9, 0.9, 0.9), roughness=0.7, metallic=0.0))
    var sk_a = ren.add_skinned_mesh(q_sk, jp_test, jw_unnorm, id_wht)
    var sk_b = ren.add_skinned_mesh(q_sk, jp_test, jw_unnorm, id_wht)
    assert_equal(sk_a.bone_count, 64)
    assert_equal(sk_a.bone_palette_offset, 0)
    assert_equal(sk_b.bone_palette_offset, 64 * 8)

    var gpu_bones = List[DualQuat](length=4, fill=DualQuat())
    gpu_bones[0] = DualQuat.from_quat_trans(Quat(), Vec3(999.0, 999.0, 999.0))  # Root bone 0 far offscreen
    gpu_bones[2] = DualQuat.from_quat_trans(Quat(), Vec3(0.0, 0.0, 3.5))        # Bone 2 centered in front of camera
    _ = ren.update_skinned_mesh_bones(sk_b, gpu_bones, gpu_bones, defer_to_frame=True)

    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    for _ in range(2):
        ren.begin_frame(Vec3(0.0, 0.0, -2.0), Vec3(0.0, 0.0, 3.5), fov_y_deg=55.0)
        _ = ren.update_skinned_mesh_bones(sk_b, gpu_bones, gpu_bones, defer_to_frame=True)
        ren.draw_instance(sk_b.mesh_entry.mesh_id, ident, ident, has_prev=True)
        assert_true(ren.render_frame(capture_readback=True))

    var hdr_ptr = Pointer[UInt32, ImmutAnyOrigin](unsafe_from_address=ren.buf_hdr_out.cpu_address)
    var center_base = (64 * w + 64) * 8
    var z_sk = bitcast[DType.float32](hdr_ptr[unsafe_offset=center_base + 3])
    var vu_static_sk = bitcast[DType.float32](hdr_ptr[unsafe_offset=center_base + 4])
    var vv_static_sk = bitcast[DType.float32](hdr_ptr[unsafe_offset=center_base + 5])
    assert_true(abs(z_sk - Float32(5.5)) < Float32(0.15))  # Hit deformed quad at z = 3.5 (cam at z = -2.0 -> dist 5.5), zero bone-0 leak!
    assert_true(abs(vu_static_sk) < Float32(1e-5) and abs(vv_static_sk) < Float32(1e-5))

    # Build 2-node articulated `NativeModel` (Node 0 = stationary torso at x=-1.2, Node 1 = articulating wing)
    var art_mdl = NativeModel()
    var torso_m = _make_uv_quad(
        Vec3(-0.7, -1.2, 3.0), Vec3(0.7, -1.2, 3.0), Vec3(0.7, 1.2, 3.0), Vec3(-0.7, 1.2, 3.0), Vec3(0.0, 0.0, -1.0)
    )
    var wing_m = _make_uv_quad(
        Vec3(-0.7, -1.2, 3.0), Vec3(0.7, -1.2, 3.0), Vec3(0.7, 1.2, 3.0), Vec3(-0.7, 1.2, 3.0), Vec3(0.0, 0.0, -1.0)
    )
    art_mdl.meshes.append(torso_m^)
    art_mdl.meshes.append(wing_m^)
    var n_torso = NativeNode()
    n_torso.name = "Torso"
    n_torso.mesh = 0
    n_torso.translation = Vec3(-1.3, 0.0, 0.0)
    n_torso.children.append(1)
    var n_wing = NativeNode()
    n_wing.name = "Wing"
    n_wing.mesh = 1
    n_wing.translation = Vec3(2.6, 0.0, 0.0)  # World x = -1.3 + 2.6 = +1.3
    art_mdl.nodes.append(n_torso^)
    art_mdl.nodes.append(n_wing^)
    art_mdl.roots.append(0)
    var art_up = ren.upload_native_model(art_mdl, build_lods=False)

    # Stationary multi-node frames with active 16-phase Halton jitter MUST emit exact `(0.0, 0.0)` velocity!
    var g_rest = art_mdl.global_transforms()
    for _ in range(3):
        ren.begin_frame(Vec3(0.0, 0.0, -2.5), Vec3(0.0, 0.0, 3.0), fov_y_deg=55.0)
        ren.draw_native_model_articulated(art_mdl, art_up, g_rest, g_rest, ident, ident, has_prev=True)
        assert_true(ren.render_frame(capture_readback=True))
    var wing_px_x = 92
    var torso_px_x = 36
    var wing_stat_vu = bitcast[DType.float32](hdr_ptr[unsafe_offset=(64 * w + wing_px_x) * 8 + 4])
    var wing_stat_vv = bitcast[DType.float32](hdr_ptr[unsafe_offset=(64 * w + wing_px_x) * 8 + 5])
    assert_true(abs(wing_stat_vu) < Float32(1e-5) and abs(wing_stat_vv) < Float32(1e-5))

    # Sweep ONLY sub-node 1 (`Wing`) from world `x = +0.5` (`prev_globals`) to `x = +1.3` (`cur_globals`) while root is stationary!
    var g_prev_art = List[Mat4]()
    g_prev_art.append(g_rest[0].copy())
    g_prev_art.append(trs(Vec3(0.5, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    ren.begin_frame(Vec3(0.0, 0.0, -2.5), Vec3(0.0, 0.0, 3.0), fov_y_deg=55.0)
    ren.draw_native_model_articulated(art_mdl, art_up, g_rest, g_prev_art, ident, ident, has_prev=True)
    assert_true(ren.render_frame(capture_readback=True))

    var torso_mv_u = bitcast[DType.float32](hdr_ptr[unsafe_offset=(64 * w + torso_px_x) * 8 + 4])
    var wing_mv_u = bitcast[DType.float32](hdr_ptr[unsafe_offset=(64 * w + wing_px_x) * 8 + 4])
    assert_true(abs(torso_mv_u) < Float32(1e-5))     # Stationary torso node has 0 velocity
    assert_true(wing_mv_u > Float32(0.08))           # Articulated wing sub-node emits strong +U motion vector!

    # Ambient Wind Advection on `64K` GPU Compute Smoke Particles (`b_globals[128..130]`)
    ren.tsr_history_weight = Float32(0.0)
    ren.ibl_intensity = Float32(0.0)
    ren.sun_dir = Vec3(0.0, 1.0, 0.0)
    ren.sun_radiance = Vec3(6.0, 6.0, 6.0)
    ren.wind_velocity = Vec3(45.0, 0.0, 0.0)
    var id_blk_bg = ren.add_material(AaaMaterial(base_color=Vec3(0.0, 0.0, 0.0), roughness=1.0, metallic=0.0))
    var e_blk_bg = ren.add_mesh(
        _make_uv_quad(Vec3(-20.0, -20.0, 12.0), Vec3(20.0, -20.0, 12.0), Vec3(20.0, 20.0, 12.0), Vec3(-20.0, 20.0, 12.0), Vec3(0.0, 0.0, -1.0)),
        id_blk_bg,
    )
    for step_w in range(14):
        ren.begin_frame(Vec3(0.0, 0.5, -2.5), Vec3(0.0, 0.5, 4.0), fov_y_deg=60.0)
        ren.draw_instance(e_blk_bg.mesh_id, ident)
        if step_w == 0:
            ren.submit_particle_emitter(
                AaaParticleEmitter.drift_smoke(
                    Vec3(0.0, 0.5, 4.0),
                    Vec3(0.0, 0.0, 0.0),
                    spawn_count=256,
                    size=0.65,
                    lifetime=1.8,
                    tint=Vec3(3.0, 3.0, 3.0),
                )
            )
        assert_true(ren.render_frame())
    var wind_hdr = ren.readback_resolved_hdr()
    var left_smoke = Float32(0.0)
    var right_smoke = Float32(0.0)
    for py_s in range(24, 104):
        for px_s in range(8, 56):
            var cl = wind_hdr[py_s * w + px_s]
            left_smoke += cl.x + cl.y + cl.z
        for px_s in range(72, 120):
            var cr = wind_hdr[py_s * w + px_s]
            right_smoke += cr.x + cr.y + cr.z
    assert_true(right_smoke > left_smoke + Float32(25.0))

    ren.shutdown()
    print("  [PASS] 11. v4 2048-Bone Skinning Arena, QEM Skin-LOD, Cofactor TBN, Zero-Jitter Articulated MV & Particle Wind Verified")


def main() raises:
    print("[Flint AAA v4] Running Full SPIR-V SSA + Vulkan Hardware Verification Suite...")
    test_1_spirv_ssa_validation_all_v2_shaders()
    test_2_cpu_models_bsdf_ltc_reblur_halton_froxel_water_agx()
    test_3_white_furnace_energy_conservation_v2()
    test_4_barycentric_textured_ray_hit_and_clearcoat_rt()
    test_5_reblur_denoiser_and_top2_hero_local_shadows()
    test_6_splat_terrain_interior_parallax_wetness_and_toksvig()
    test_7_froxel_volumetrics_rain_and_beer_lambert_water()
    test_8_tsr_v3_halton16_motion_blur_exposure_agx_and_hologram_oit()
    test_9_alphamask_visbuffer_vsm_and_nonopaque_rayquery()
    test_10_nubis_clouds_starfield_vertex_color_fireballs_and_gpu_particles()
    test_11_v4_skinning_arena_qem_articulated_mv_cofactor_tbn_and_wind()
    print("[Flint AAA v4] ALL 11 FLAGSHIP VERIFICATION SUITES PASSED WITH 0 ERRORS!")
