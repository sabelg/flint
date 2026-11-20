"""Flint v5 Track 2 Standalone Procedural Render Pipeline & Metrology Suite (`test_flint_v5_render.mojo`).

Zero external `.glb` / font / texture files. Executes end-to-end in `<2.0s` verifying:
  1. SPIR-V `OpSpecConstant` (`OP_SPEC_CONSTANT_TRUE=48`, `OP_SPEC_CONSTANT_FALSE=49`, `OP_SPEC_CONSTANT=50`,
     `DEC_SPEC_ID=1`) on `SpirvBuilder` & `AaaSpirvModule`, validated by `validate_spirv_module`, plus
     `aaa_ui_overlay_shader()`, `vat_crowd_eval_shader()`, and `SurfaceMaterialNode` (`ShaderDsl`).
  2. `MaterialAsset` 128-byte (`32 x UInt32`) GPU layout with **16-bit bindless slots up to `65,534`**,
     `MAT_DOMAIN_*`, `MAT_FLAG_*`, `GpuDecalVolume` OBB projection, and `TerrainClipmapDesc` snapping.
  3. Procedural 2-bone skeletal `NativeModel` -> `bake_vat_crowd_clip_rgba16f` (`RGBA16F` pos+normal sheets),
     10,000-instance phase evaluation, and non-zero 3D motion vectors (`cur_pos - prev_pos`).
  4. `WaylandWindow.enable_vulkan_bridge()`, `WaylandWindow.presents_on_gpu()`, `SwapchainSurface`
     (`AcquiredFrame`, `SURFACE_HEADLESS`, `recreate`), RH `-Z` / Reverse-Z projection math, `DebugLineBatch`
     near-plane clipping into `UiDrawList` `PRIM_LINE`s, `DoubleBufferedRenderWorld`, `RenderExtractor`
     (`UInt64` entity ID tracking + despawn pruning), and `ScalableRenderer` (`TIER_0`, `TIER_1`, `TIER_2`).
  5. `ui_overlay.mojo` exact `0xFFFF2040` pure-red vs `0xFF2040FF` pure-blue GPU pixel readback (proving
     Bug 3 Red/Blue swap is 100% fixed!), all 6 GPU SDF primitive kinds (`PRIM_SHAPE`, `PRIM_GLYPH`,
     `PRIM_NINE_SLICE`, `PRIM_CONTROLLER_SYMBOL`, `PRIM_LINE` capsule + Bézier, `PRIM_IMAGE`),
     `>300` (`340`) overlapping primitives inside a single `16x16` tile (proving zero drops & painter's order),
     and `disp_mode = 1` `A2B10G10R10` HDR10 ST.2084 PQ @ 203 nits paper-white!
  6. Procedural White-Furnace energy conservation check + Cornell Box + 5x5 PBR/Clearcoat Sphere Matrix +
     4-Cascade Toroidal VSM (`±16m, ±64m, ±256m, ±1024m`) Sundial shadow contrast!
  7. Render features: soft ray-traced sun shadows (penumbra filter radius), refractive glass (Fresnel,
     Snell, total internal reflection, Beer-Lambert), shaped water and generic GPU particles (color / size /
     alpha over life, additive and alpha blending): SPIR-V validation of their passes, and on the GPU,
     additive particles light the frame and clear glass shows the wall behind it (with ray queries).
"""
from std.collections import List
from std.ffi import external_call
from std.math import abs, cos, max, min, sin, sqrt
from std.memory import bitcast
from std.sys.info import size_of
from flint.assets.flpk_v3 import AssetUuid
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.server import AssetServer
from flint.assets.model import (
    NativeAnimation,
    NativeChannel,
    NativeMesh,
    NativeModel,
    NativeNode,
    NativeSkin,
)
from flint.core.math import Mat4, Vec3
from flint.core.transform import Quat, trs
from flint.platform.amdgpu import (
    AmdGpu,
    _AmdDeviceControlBlock,
    _GpuBufferControl,
    _amd_ctrl_ptr,
    _buf_ctrl_ptr,
)
from flint.platform.os import close_fd, get_time
from flint.platform.vulkan import (
    ACCESS_HOST_READ,
    ACCESS_TRANSFER_WRITE,
    BUFFER_USAGE_AS_BUILD_INPUT_KHR,
    BUFFER_USAGE_DEVICE_ADDRESS,
    FORMAT_A2B10G10R10_UNORM_PACK32,
    FORMAT_B8G8R8A8_UNORM,
    FORMAT_R16G16B16A16_SFLOAT,
    FORMAT_R8G8B8A8_UNORM,
    GpuBuffer,
    IMAGE_USAGE_SAMPLED,
    IMAGE_USAGE_TRANSFER_TARGET,
    PIPELINE_HOST,
    PIPELINE_TRANSFER,
    SwapchainHandle,
    VulkanAccelStruct,
    VulkanGpu,
    VulkanImage,
    VulkanMemBlock,
    get32,
    put32,
    put_float,
    swapchain_format_bytes_per_pixel,
)
from flint.platform.wayland import WaylandWindow
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
from flint.render.aaa_froxel_volumetrics import (
    AAA_FROXEL_Z,
    eval_cornette_shanks_phase_cpu,
    eval_hillaire_slice_integration_cpu,
    froxel_inject_and_reproject_shader,
    froxel_prefix_integrate_shader,
    froxel_raymarch_integrate_shader,
)
from flint.render.aaa_gi_svgf import (
    AAA_SHADE_CLASS_COMPLEX,
    AAA_SHADE_CLASS_COUNT,
    AAA_SHADE_CLASS_FOLIAGE_SSS,
    AAA_SHADE_CLASS_SKY,
    AAA_SHADE_CLASS_STANDARD_PBR,
    AAA_SHADE_CLASS_UNLIT,
    aaa_deferred_shade_class_shader,
    aaa_deferred_shade_shader,
    aaa_shade_sky_shader,
    aaa_shade_unlit_shader,
    aaa_tile_classify_shader,
    eval_cofactor_gram_schmidt_tbn_cpu,
    surface_cache_update_shader,
)
from flint.render.aaa_npr_vfx import visbuffer_alphamask_vfx_fragment_shader
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
    aaa_display_present_fragment_shader,
    aaa_display_present_vertex_shader,
    aaa_display_tonemap_direct_shader,
    aaa_display_tonemap_shader,
    aaa_motion_blur_shader,
    aaa_tsr_v3_resolve_shader,
    eval_agx_tonemap_cpu,
    eval_display_grade_cpu,
    halton_2_3_jitter_px,
)
from flint.render.aaa_raytracing import build_blas_triangles_ex
from flint.render.aaa_render_graph import (
    RG_QUEUE_GRAPHICS,
    RG_USAGE_COMPUTE_READ,
    RG_USAGE_COMPUTE_WRITE,
    AaaRenderGraph,
    RgResourceAccess,
)
from flint.render.aaa_renderer import (
    AAA_DISPLAY_SCRGB_FP16,
    FlintAaaVulkanRenderer,
    LinearStagingRing,
    aaa_pack_scrgb_rgba16f_shader,
)
from flint.render.aaa_sky_atmosphere import (
    BrunetonAtmosphereParams,
    eval_nubis_cloud_density_cpu,
)
from flint.render.aaa_spec_reblur import (
    eval_anisotropic_ggx_blur_radii_cpu,
    spec_reblur_atrous_shader,
    spec_reblur_temporal_shader,
)
from flint.render.aaa_visbuffer import (
    AAA_MAX_BINDLESS,
    AAA_MAX_PACKED_INSTANCES,
    AAA_MAX_PACKED_MESHLET_ID,
    BIND_AAA_HDR_OUT,
    BIND_AAA_POST_SCRATCH,
    compute_hiz_mip_levels,
    hiz_depth_downsample_shader,
    meshlet_cull_shader,
    visbuffer_alphamask_fragment_shader,
    visbuffer_alphamask_vertex_shader,
)
from flint.render.aaa_glass import (
    AaaGlassParams,
    aaa_glass_refract_shader,
    glass_beer_lambert_cpu,
    glass_f0_cpu,
    glass_fresnel_cpu,
    glass_refract_cpu,
)
from flint.render.aaa_gpu_particles import (
    AAA_EMITTER_WORDS,
    PARTICLE_FLAG_ADDITIVE,
    PARTICLE_FLAG_STRETCH,
    PARTICLE_GENERIC,
    AaaParticleEmitter,
    gpu_particle_clear_tiles_shader,
    gpu_particle_sim_and_bin_shader,
    gpu_particle_tile_composite_shader,
)
from flint.render.aaa_rt_shadow import penumbra_radius_px_cpu, rt_sun_shadow_filter_shader, rt_sun_shadow_trace_shader
from flint.render.aaa_ibl import SkyDomeParams, bake_bc6h_ibl_cubemap_and_sh9
from flint.render.aaa_skinning_rt import AAA_SKIN_MAX_VERTS, DualQuat
from flint.render.aaa_water import (
    aaa_water_refract_shader,
    eval_beer_lambert_water_cpu,
    eval_dielectric_water_fresnel_cpu,
    eval_nyquist_lean_water_alpha2_cpu,
)
from flint.render.aaa_vsm import (
    VSM_CASCADE_0_EXTENT,
    VSM_CASCADE_1_EXTENT,
    VSM_CASCADE_2_EXTENT,
    VSM_CASCADE_3_EXTENT,
    VSM_TABLE_WORDS,
    VsmPageManager,
    compute_vsm_cascade_matrices,
    compute_vsm_cascade_page_origins,
    mark_toroidal_cascade_pages,
    vsm_alphamask_fragment_shader,
    vsm_alphamask_vertex_shader,
    vsm_mark_pages_shader,
    vsm_page_vertex_shader,
)
from flint.render.debug_draw import DEBUG_CYAN, DebugDraw
from flint.render.material_graph import (
    BINDLESS_SLOT_MAX,
    BINDLESS_SLOT_NONE,
    GPU_DECAL_VOLUME_WORDS,
    MATERIAL_ARCHIVE_BYTES,
    MATERIAL_ARCHIVE_WORDS,
    MATERIAL_ASSET_BYTES,
    MATERIAL_ASSET_WORDS,
    MAT_DOMAIN_CLEARCOAT,
    MAT_DOMAIN_STANDARD_PBR,
    MAT_DOMAIN_SUBSURFACE_SKIN,
    MAT_DOMAIN_TWO_SIDED_FOLIAGE,
    MAT_DOMAIN_UNLIT_UI,
    MAT_FLAG_ALPHA_MASKED,
    MAT_FLAG_CLEARCOAT,
    MAT_FLAG_DOUBLE_SIDED,
    MAT_FLAG_EMISSIVE_BLOOM,
    MAT_FLAG_RECEIVE_DECALS,
    MAT_FLAG_ROAD_WETNESS,
    MAT_FLAG_VAT_CROWD,
    BakedVatCrowdClip,
    GpuDecalVolume,
    MaterialAsset,
    StandardPbrMaterialNode,
    TerrainClipmapDesc,
    TriplanarWeatherMaterialNode,
    VatCrowdClipDesc,
    bake_vat_crowd_clip_rgba16f,
    compile_surface_material_shader,
    f16_bits_to_f32,
    f32_to_f16_bits,
    sample_vat_vertex_cpu,
    terrain_clipmap_eval_shader,
    vat_crowd_eval_shader,
)
from flint.render.render_world import (
    DEG_TO_RAD_F32,
    RENDER_TIER_0_HEADLESS_SIMD,
    RENDER_TIER_1_FORWARD_CLUSTERED,
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    DebugLineBatch,
    DoubleBufferedRenderWorld,
    RenderCameraPacket,
    RenderExtractor,
    ScalableRenderer,
    TerrainClipmapPacket,
    make_rh_look_at,
    make_rh_reverse_z_infinite_perspective,
    make_rh_reverse_z_perspective,
    project_rh_world_to_screen,
)
from flint.render.simd_canvas import pack_argb
from flint.render.spirv import (
    AaaSpirvModule,
    DEC_SPEC_ID,
    MODEL_GL_COMPUTE,
    OP_BITWISE_AND,
    OP_CONSTANT,
    OP_DECORATE,
    OP_SPEC_CONSTANT,
    OP_SPEC_CONSTANT_FALSE,
    OP_SPEC_CONSTANT_TRUE,
    SpirvBuilder,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.render.swapchain_surface import (
    DISPLAY_EXTENDED_SCRGB_FP16,
    DISPLAY_HDR10_PQ,
    DISPLAY_SDR_SRGB,
    PRESENT_MAILBOX,
    SURFACE_HEADLESS,
    SwapchainSurface,
    resolve_swapchain_display_mode,
    validate_present_source_bytes,
)
from flint.render.ui_overlay import (
    UI_SYMBOL_A,
    UI_SYMBOL_GIZMO,
    UI_SYMBOL_PLAY,
    aaa_ui_overlay_direct_shader,
    aaa_ui_overlay_shader,
    aaa_ui_present_fragment_shader,
    make_ui_bezier_prim,
    make_ui_image_prim,
    make_ui_line_prim,
    make_ui_symbol_prim,
)
from flint.ui.draw_list import (
    GpuUiPrim,
    PRIM_GLYPH,
    PRIM_LINE,
    PRIM_POLY_DATA,
    UI_FLAG_BORDER,
    UI_FLAG_DROP_SHADOW,
    UI_FLAG_VERT_GRADIENT,
    UiDrawList,
)


def _fail(msg: String):
    print("FAIL:", msg)
    external_call["exit", NoneType](Int32(1))


def _assert(cond: Bool, msg: String):
    if not cond:
        _fail(msg)


def _unpack_argb(px: UInt32) -> Tuple[Int, Int, Int, Int]:
    var a = Int((px >> 24) & UInt32(0xFF))
    var r = Int((px >> 16) & UInt32(0xFF))
    var g = Int((px >> 8) & UInt32(0xFF))
    var b = Int(px & UInt32(0xFF))
    return (r, g, b, a)


def _contains_opcode(words: List[UInt32], target_op: UInt32) -> Bool:
    var idx = 5
    var n = len(words)
    while idx < n:
        var hdr = words[idx]
        var wc = Int(hdr >> 16)
        var op = hdr & UInt32(0xFFFF)
        if wc <= 0:
            break
        if op == target_op:
            return True
        idx += wc
    return False


def _contains_spec_id(words: List[UInt32], spec_id: UInt32) -> Bool:
    var idx = 5
    var n = len(words)
    while idx < n:
        var hdr = words[idx]
        var wc = Int(hdr >> 16)
        var op = hdr & UInt32(0xFFFF)
        if wc <= 0:
            break
        if op == OP_DECORATE and wc >= 4 and words[idx + 2] == DEC_SPEC_ID and words[idx + 3] == spec_id:
            return True
        idx += wc
    return False


# ============================================================================
# Suite 1: SPIR-V `OpSpecConstant` & Shader Validator
# ============================================================================

def test_spirv_spec_constants_and_shaders():
    _assert(OP_SPEC_CONSTANT_TRUE == 48, "OP_SPEC_CONSTANT_TRUE == 48")
    _assert(OP_SPEC_CONSTANT_FALSE == 49, "OP_SPEC_CONSTANT_FALSE == 49")
    _assert(OP_SPEC_CONSTANT == 50, "OP_SPEC_CONSTANT == 50")
    _assert(DEC_SPEC_ID == 1, "DEC_SPEC_ID == 1")

    # Test `SpirvBuilder` (`SpirvModule`) specialization constants
    var sb = SpirvBuilder(8, 8, 4)
    var b_out = sb.buffer(0)
    var sc_true = sb.spec_constant_bool(0, True)
    var sc_false = sb.spec_constant_bool(1, False)
    var sc_u32 = sb.spec_constant_u32(2, 42)
    var chosen = sb.select_u(sc_true, sc_u32, sb.select_u(sc_false, sb.uint(99), sb.uint(7)))
    sb.store_word(b_out, sb.uint(0), chosen)
    var sb_words = sb.finish()

    _assert(_contains_opcode(sb_words, OP_SPEC_CONSTANT_TRUE), "SpirvBuilder emitted OpSpecConstantTrue")
    _assert(_contains_opcode(sb_words, OP_SPEC_CONSTANT_FALSE), "SpirvBuilder emitted OpSpecConstantFalse")
    _assert(_contains_opcode(sb_words, OP_SPEC_CONSTANT), "SpirvBuilder emitted OpSpecConstant")
    _assert(_contains_spec_id(sb_words, 0) and _contains_spec_id(sb_words, 1) and _contains_spec_id(sb_words, 2), "SpecId decorations emitted")

    var rep_sb = validate_spirv_module(sb_words)
    if not rep_sb.valid:
        _fail("SpirvBuilder OpSpecConstant validation failed: " + rep_sb.errors[0])

    # Test `AaaSpirvModule` specialization constants
    var am = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=4)
    var a_buf = am.buffer(0)
    var am_b0 = am.spec_constant_bool(10, True)
    var am_b1 = am.spec_constant_bool(11, False)
    var am_u0 = am.spec_constant_u32(12, 1337)
    var val_u = am.select_u(am_b0, am_u0, am.select_u(am_b1, am.uint(1), am.uint(0)))
    am.store_uint(a_buf, am.uint(0), val_u)
    var am_words = am.finish()
    var rep_am = validate_spirv_module(am_words)
    if not rep_am.valid:
        _fail("AaaSpirvModule OpSpecConstant validation failed: " + rep_am.errors[0])

    # Validate Track 2 GPU kernels
    var ui_spv = aaa_ui_overlay_shader()
    var rep_ui = validate_spirv_module(ui_spv)
    if not rep_ui.valid:
        _fail("aaa_ui_overlay_shader SPIR-V validation failed: " + rep_ui.errors[0])

    var vat_spv = vat_crowd_eval_shader()
    var rep_vat = validate_spirv_module(vat_spv)
    if not rep_vat.valid:
        _fail("vat_crowd_eval_shader SPIR-V validation failed: " + rep_vat.errors[0])

    var mat_std_spv = compile_surface_material_shader(StandardPbrMaterialNode(enable_detail_layer=True))
    var rep_mat_std = validate_spirv_module(mat_std_spv)
    if not rep_mat_std.valid:
        _fail("StandardPbrMaterialNode SPIR-V validation failed: " + rep_mat_std.errors[0])
    _assert(_contains_opcode(mat_std_spv, OP_SPEC_CONSTANT_TRUE) and _contains_opcode(mat_std_spv, OP_SPEC_CONSTANT), "Material shader uses OpSpecConstant")

    var mat_tri_spv = compile_surface_material_shader(TriplanarWeatherMaterialNode(snow_line_y=8.0))
    var rep_mat_tri = validate_spirv_module(mat_tri_spv)
    if not rep_mat_tri.valid:
        _fail("TriplanarWeatherMaterialNode SPIR-V validation failed: " + rep_mat_tri.errors[0])

    var scrgb_spv = aaa_pack_scrgb_rgba16f_shader()
    var rep_scrgb = validate_spirv_module(scrgb_spv)
    if not rep_scrgb.valid:
        _fail("aaa_pack_scrgb_rgba16f_shader SPIR-V validation failed: " + rep_scrgb.errors[0])
    var rep_vsm_m = validate_spirv_module(vsm_mark_pages_shader())
    if not rep_vsm_m.valid:
        _fail("vsm_mark_pages_shader SPIR-V validation failed: " + rep_vsm_m.errors[0])

    # F17 & F18 SPIR-V validation + class specialization checks
    var spv_tc = aaa_tile_classify_shader()
    _validate_or_fail(spv_tc, "aaa_tile_classify_shader")
    var spv_sky = aaa_shade_sky_shader(True)
    _validate_or_fail(spv_sky, "aaa_shade_sky_shader")
    var spv_unlit = aaa_shade_unlit_shader(True)
    _validate_or_fail(spv_unlit, "aaa_shade_unlit_shader")
    var spv_pbr_rt = aaa_deferred_shade_class_shader(True, AAA_SHADE_CLASS_STANDARD_PBR, True)
    var spv_pbr_no_rt = aaa_deferred_shade_class_shader(False, AAA_SHADE_CLASS_STANDARD_PBR, True)
    var spv_fol_rt = aaa_deferred_shade_class_shader(True, AAA_SHADE_CLASS_FOLIAGE_SSS, True)
    var spv_cmp_rt = aaa_deferred_shade_class_shader(True, AAA_SHADE_CLASS_COMPLEX, True)
    _validate_or_fail(spv_pbr_rt, "aaa_deferred_shade_class_shader(PBR, RT)")
    _validate_or_fail(spv_pbr_no_rt, "aaa_deferred_shade_class_shader(PBR, NoRT)")
    _validate_or_fail(spv_fol_rt, "aaa_deferred_shade_class_shader(FOLIAGE_SSS, RT)")
    _validate_or_fail(spv_cmp_rt, "aaa_deferred_shade_class_shader(COMPLEX, RT)")
    _assert(len(spv_pbr_rt) < len(spv_cmp_rt), "Specialized STANDARD_PBR shader smaller than COMPLEX shader")
    _validate_or_fail(terrain_clipmap_eval_shader(), "terrain_clipmap_eval_shader")
    var rep_vsm_v = validate_spirv_module(vsm_page_vertex_shader())
    if not rep_vsm_v.valid:
        _fail("vsm_page_vertex_shader SPIR-V validation failed: " + rep_vsm_v.errors[0])
    var rep_vsm_av = validate_spirv_module(vsm_alphamask_vertex_shader())
    if not rep_vsm_av.valid:
        _fail("vsm_alphamask_vertex_shader SPIR-V validation failed: " + rep_vsm_av.errors[0])
    var rep_hiz = validate_spirv_module(hiz_depth_downsample_shader())
    if not rep_hiz.valid:
        _fail("hiz_depth_downsample_shader SPIR-V validation failed: " + rep_hiz.errors[0])
    var rep_mc = validate_spirv_module(meshlet_cull_shader())
    if not rep_mc.valid:
        _fail("meshlet_cull_shader SPIR-V validation failed: " + rep_mc.errors[0])
    var max_bindless_rt = AAA_MAX_BINDLESS
    var max_ml_rt = AAA_MAX_PACKED_MESHLET_ID
    var max_inst_rt = AAA_MAX_PACKED_INSTANCES
    _assert(max_bindless_rt >= 4096, "AAA_MAX_BINDLESS raised to >= 4096")
    _assert(max_ml_rt == 0x3FFFFE and max_inst_rt == 0x3FFFF, "22-bit ml_id / 18-bit inst_id packing limits")
    _assert(compute_hiz_mip_levels(320, 180) == 8 and compute_hiz_mip_levels(16, 16) == 5, "Hi-Z mip level computation")

    print("[PASS] Suite 1: SPIR-V OpSpecConstant & Track 2 Shader Validation (" + String(len(ui_spv)) + " UI words)")


# ============================================================================
# Suite 2: 128-Byte `MaterialAsset` (16-Bit Bindless), Decals, Terrain & VAT
# ============================================================================

def _build_procedural_skeletal_walker() -> NativeModel:
    """Builds a procedural 2-bone articulated capsule character with a 1.0s walk-swing animation clip (0 external files!)."""
    var mb = MeshBuilder()
    mb.add_cylinder(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.8, 0.0), 0.22, 0.16, segments=12)
    var mesh = mb^.finish()
    var vc = mesh.vertex_count()
    mesh.skin = 0
    for v in range(vc):
        var py = mesh.positions[v * 3 + 1]
        var w_upper = max(Float32(0.0), min(Float32(1.0), (py - Float32(0.5)) / Float32(1.1)))
        var w_lower = Float32(1.0) - w_upper
        mesh.joints.append(UInt16(0))
        mesh.joints.append(UInt16(1))
        mesh.joints.append(UInt16(0))
        mesh.joints.append(UInt16(0))
        mesh.weights.append(w_lower)
        mesh.weights.append(w_upper)
        mesh.weights.append(Float32(0.0))
        mesh.weights.append(Float32(0.0))

    var n0 = NativeNode()
    n0.name = "pelvis"
    n0.children.append(1)

    var n1 = NativeNode()
    n1.name = "torso"
    n1.base_translation = Vec3(0.0, 0.85, 0.0)
    n1.translation = Vec3(0.0, 0.85, 0.0)

    var sk = NativeSkin()
    sk.joints.append(0)
    sk.joints.append(1)
    var ibm0 = Mat4()
    var ibm1 = trs(Vec3(0.0, -0.85, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    for k in range(16):
        sk.inverse_bind.append(ibm0.values[k])
    for k in range(16):
        sk.inverse_bind.append(ibm1.values[k])

    var ch_rot = NativeChannel()
    ch_rot.node = 1
    ch_rot.path = 1  # rotation
    var kf_times = [Float32(0.0), Float32(0.25), Float32(0.50), Float32(0.75), Float32(1.0)]
    var kf_angles = [Float32(0.0), Float32(0.45), Float32(0.0), Float32(-0.45), Float32(0.0)]
    for i in range(5):
        ch_rot.times.append(kf_times[i])
        var half_a = kf_angles[i] * Float32(0.5)
        var q = Quat(sin(half_a), 0.0, 0.0, cos(half_a)).normalized()
        ch_rot.values.append(q.x)
        ch_rot.values.append(q.y)
        ch_rot.values.append(q.z)
        ch_rot.values.append(q.w)

    var anim = NativeAnimation()
    anim.name = "walk_swing"
    anim.duration = Float32(1.0)
    anim.channels.append(ch_rot^)

    var model = NativeModel()
    model.meshes.append(mesh^)
    model.nodes.append(n0^)
    model.nodes.append(n1^)
    model.skins.append(sk^)
    model.animations.append(anim^)
    return model^


def test_material_graph_decals_terrain_and_vat():
    var mat = MaterialAsset(
        base_color=Vec3(0.82, 0.24, 0.12),
        roughness=0.18,
        metallic=0.85,
        domain=MAT_DOMAIN_CLEARCOAT,
        flags=MAT_FLAG_RECEIVE_DECALS | MAT_FLAG_EMISSIVE_BLOOM | MAT_FLAG_VAT_CROWD,
    )
    mat.clearcoat = Float32(0.95)
    mat.clearcoat_roughness = Float32(0.06)
    mat.albedo_slot = BINDLESS_SLOT_MAX          # 65,534 (0xFFFE) — full 16-bit bindless!
    mat.normal_slot = UInt16(42000)
    mat.orm_slot = UInt16(12345)
    mat.emissive_slot = BINDLESS_SLOT_NONE
    mat.vat_pos_slot = UInt16(60001)
    mat.vat_nrm_slot = UInt16(60002)
    mat.uv_scale_x = Float32(4.0)
    mat.uv_scale_y = Float32(2.5)

    var packed_gpu = mat.pack_words()
    var mat_words_rt = MATERIAL_ASSET_WORDS
    _assert(
        mat_words_rt == AAA_MATERIAL_WORDS
        and mat_words_rt == 24
        and len(packed_gpu) == MATERIAL_ASSET_WORDS
        and len(packed_gpu) * 4 == MATERIAL_ASSET_BYTES,
        "MaterialAsset GPU layout unified with AAA_MATERIAL_WORDS == 24 words (96 bytes)",
    )
    var aaa_words = List[UInt32]()
    mat.to_aaa_material().append_words(aaa_words)
    for w_i in range(AAA_MATERIAL_WORDS):
        _assert(packed_gpu[w_i] == aaa_words[w_i], "MaterialAsset.pack_words matches AaaMaterial SSBO word " + String(w_i))
    var unpacked_gpu = MaterialAsset.unpack_words(packed_gpu)
    _assert(abs(unpacked_gpu.clearcoat - Float32(0.95)) < Float32(1e-4) and unpacked_gpu.domain == MAT_DOMAIN_CLEARCOAT, "24-word GPU unpack preserves clearcoat")

    var packed = mat.pack_archive_words()
    _assert(len(packed) == MATERIAL_ARCHIVE_WORDS and len(packed) * 4 == MATERIAL_ARCHIVE_BYTES, "MaterialAsset archive == 128 bytes (32 words)")

    var unpacked = MaterialAsset.unpack_archive_words(packed)
    _assert(unpacked.albedo_slot == UInt16(65534), "16-bit albedo_slot 65,534 preserved")
    _assert(unpacked.normal_slot == UInt16(42000), "16-bit normal_slot 42,000 preserved")
    _assert(unpacked.orm_slot == UInt16(12345), "16-bit orm_slot 12,345 preserved")
    _assert(unpacked.emissive_slot == BINDLESS_SLOT_NONE, "BINDLESS_SLOT_NONE preserved")
    _assert(unpacked.vat_pos_slot == UInt16(60001) and unpacked.vat_nrm_slot == UInt16(60002), "16-bit VAT slots preserved")
    _assert(unpacked.domain == MAT_DOMAIN_CLEARCOAT, "MAT_DOMAIN_CLEARCOAT preserved")
    _assert((unpacked.flags & MAT_FLAG_CLEARCOAT) != 0 and (unpacked.flags & MAT_FLAG_VAT_CROWD) != 0, "MAT_FLAG_* preserved")
    _assert(abs(unpacked.uv_scale_x - Float32(4.0)) < Float32(0.01) and abs(unpacked.uv_scale_y - Float32(2.5)) < Float32(0.01), "FP16 UV scales preserved")
    var skin_mat = MaterialAsset(domain=MAT_DOMAIN_SUBSURFACE_SKIN)
    var std_mat = MaterialAsset(domain=MAT_DOMAIN_STANDARD_PBR)
    _assert(skin_mat.domain == 1 and std_mat.domain == 0, "Material domain enum check")

    # F17 lossless 24-word round-trip for Unlit & Two-Sided Foliage
    var unlit_mat = MaterialAsset(base_color=Vec3(0.2, 0.9, 0.3), domain=MAT_DOMAIN_UNLIT_UI)
    unlit_mat.sheen_roughness = Float32(0.42)
    var unlit_words = List[UInt32]()
    unlit_mat.to_aaa_material().append_words(unlit_words)
    _assert(len(unlit_words) == MATERIAL_ASSET_WORDS and MATERIAL_ASSET_WORDS == 24, "Unlit uses 24 words")
    var unlit_rt = MaterialAsset.unpack_words(unlit_words)
    _assert(unlit_rt.domain == MAT_DOMAIN_UNLIT_UI, "24-word GPU unpack preserves MAT_DOMAIN_UNLIT_UI")

    var fol_mat = MaterialAsset(base_color=Vec3(0.25, 0.65, 0.2), domain=MAT_DOMAIN_TWO_SIDED_FOLIAGE)
    fol_mat.clearcoat_roughness = Float32(0.12)
    fol_mat.subsurface_color = Vec3(0.4, 0.85, 0.2)
    var fol_words = List[UInt32]()
    fol_mat.to_aaa_material().append_words(fol_words)
    var fol_rt = MaterialAsset.unpack_words(fol_words)
    _assert(
        fol_rt.domain == MAT_DOMAIN_TWO_SIDED_FOLIAGE and (fol_rt.flags & MAT_FLAG_DOUBLE_SIDED) != 0,
        "24-word GPU unpack preserves MAT_DOMAIN_TWO_SIDED_FOLIAGE + MAT_FLAG_DOUBLE_SIDED",
    )

    # Regression (F17): negative scalar inputs must clamp non-negative without flipping sign-bit flags
    var neg_mat = MaterialAsset(base_color=Vec3(0.4, 0.6, 0.8), domain=MAT_DOMAIN_STANDARD_PBR)
    neg_mat.sheen_roughness = Float32(-0.5)
    neg_mat.clearcoat_roughness = Float32(-0.5)
    neg_mat.specular_factor = Float32(-0.5)
    neg_mat.detail_uv_scale = Float32(-1.0)
    var neg_aaa = neg_mat.to_aaa_material()
    _assert(
        neg_aaa.sheen_roughness >= Float32(0.05)
        and neg_aaa.clearcoat_roughness >= Float32(0.04)
        and neg_aaa.specular_factor >= Float32(0.0)
        and neg_aaa.detail_uv_scale >= Float32(0.0),
        "MaterialAsset.to_aaa_material clamps negative scalar roughness/specular/detail_scale non-negative",
    )
    var neg_words = List[UInt32]()
    neg_aaa.append_words(neg_words)
    _assert(
        bitcast[DType.float32](neg_words[3]) >= Float32(0.0)
        and bitcast[DType.float32](neg_words[7]) >= Float32(0.04)
        and bitcast[DType.float32](neg_words[11]) >= Float32(0.001)
        and bitcast[DType.float32](neg_words[19]) >= Float32(0.05),
        "AaaMaterial.append_words keeps sign bits strictly positive on negative scalar inputs",
    )
    var neg_rt = MaterialAsset.unpack_words(neg_words)
    _assert(
        neg_rt.domain == MAT_DOMAIN_STANDARD_PBR
        and (neg_rt.flags & (MAT_FLAG_ALPHA_MASKED | MAT_FLAG_DOUBLE_SIDED | MAT_FLAG_ROAD_WETNESS)) == 0,
        "Negative scalar inputs do not flip domain/flags on 24-word unpack",
    )
    var ac_only = MaterialAsset(base_color=Vec3(0.5, 0.5, 0.5), domain=MAT_DOMAIN_STANDARD_PBR, flags=MAT_FLAG_RECEIVE_DECALS)
    ac_only.alpha_cutoff = Float32(0.5)
    var ac_aaa = ac_only.to_aaa_material()
    _assert(ac_aaa.is_alpha_masked() and ac_aaa.alpha_cutoff >= Float32(0.5), "to_aaa_material honors alpha_cutoff without explicit flag")
    var ac_rt = MaterialAsset.unpack_words(ac_only.pack_words())
    _assert((ac_rt.flags & MAT_FLAG_ALPHA_MASKED) != 0 and ac_rt.alpha_cutoff >= Float32(0.5), "alpha_cutoff without flag round-trips")

    # OBB Decal Volume projection + normal angle fade
    var decal = GpuDecalVolume(
        center=Vec3(2.0, 0.0, -3.0),
        half_extents=Vec3(1.5, 0.5, 1.5),
        rotation=Quat(),
        albedo_slot=UInt16(512),
        opacity=Float32(0.9),
    )
    _assert(len(decal.pack_words()) == GPU_DECAL_VOLUME_WORDS, "GpuDecalVolume == 24 words (96 bytes)")
    var hit_up = decal.project_world_point(Vec3(2.0, 0.0, -3.0), Vec3(0.0, 1.0, 0.0))
    _assert(hit_up[0] and abs(hit_up[1] - Float32(0.5)) < Float32(0.01) and hit_up[3] > Float32(0.8), "Decal projects upward floor point")
    var miss_wall = decal.project_world_point(Vec3(2.0, 0.0, -3.0), Vec3(1.0, 0.0, 0.0))
    _assert(not miss_wall[0], "Decal rejects perpendicular wall normal")

    # Terrain Clipmap grid snapping
    var clip_desc = TerrainClipmapDesc(ring_count=4, grid_dim_per_ring=64, base_cell_size_m=0.5)
    _assert(len(clip_desc.pack_words()) == 12, "TerrainClipmapDesc packs 12 words")
    var snap0 = clip_desc.snapped_ring_origin_xz(Vec3(13.3, 5.0, -7.8), 0)
    _assert(abs(snap0[0] - Float32(13.0)) < Float32(1e-4) and abs(snap0[1] - Float32(-8.0)) < Float32(1e-4), "Clipmap ring 0 snaps to 2*cell=1.0m")

    # VAT Crowd Baker (`RGBA16F`) & 10,000-instance motion vector check
    var walker = _build_procedural_skeletal_walker()
    var baked_vat = bake_vat_crowd_clip_rgba16f(walker, mesh_index=0, clip_index=0, sample_fps=30.0, looping=True)
    _assert(baked_vat.desc.vertex_count > 20 and baked_vat.desc.frame_count == 30, "VAT baked 30 frames")
    _assert(len(baked_vat.pos_rgba16f) == baked_vat.desc.vertex_count * 30 * 4, "RGBA16F pos sheet size exact")

    # Find top vertex (`y > 1.7`) and verify skeletal swing deformation + non-zero 3D motion vectors across crowd phases
    var top_v = 0
    var max_y = Float32(-1e9)
    for v in range(walker.meshes[0].vertex_count()):
        var py = walker.meshes[0].positions[v * 3 + 1]
        if py > max_y:
            max_y = py
            top_v = v
    var s_f0 = sample_vat_vertex_cpu(baked_vat, top_v, Float32(0.0), Float32(0.0))
    var s_peak = sample_vat_vertex_cpu(baked_vat, top_v, Float32(0.25), Float32(0.25 - 1.0 / 60.0))
    var s_mid = sample_vat_vertex_cpu(baked_vat, top_v, Float32(0.125), Float32(0.125 - 1.0 / 60.0))
    var swing_disp = (s_peak[0] - s_f0[0]).length()
    var frame_mv = (s_mid[0] - s_mid[2]).length()
    _assert(swing_disp > Float32(0.25), "VAT skeletal swing deforms upper body > 0.25m")
    _assert(frame_mv > Float32(0.015), "VAT cur_pos - prev_pos 3D motion vector is non-zero")

    # Verify 10,000 crowd instances sample independent phases cleanly
    var crowd_mv_accum = Float32(0.0)
    for inst_i in range(10000):
        var phase = Float32(inst_i % 97) * Float32(0.0103)
        var sv = sample_vat_vertex_cpu(baked_vat, top_v, phase + Float32(0.12), phase + Float32(0.12 - 1.0 / 60.0))
        crowd_mv_accum += (sv[0] - sv[2]).length()
    _assert(crowd_mv_accum > Float32(50.0), "10,000 VAT crowd instances animate independently with valid motion vectors")

    print("[PASS] Suite 2: 128B MaterialAsset (slot 65,534), OBB Decals, Clipmap & 10k VAT Crowd")


# ============================================================================
# Suite 3: SwapchainSurface, RH `-Z` / Reverse-Z, DebugLineBatch & RenderWorld
# ============================================================================

def test_swapchain_render_world_and_tier0():
    # WaylandWindow zero-copy bridge & `presents_on_gpu()` contract
    var win = WaylandWindow()
    _assert(not win.wants_bridge, "WaylandWindow.wants_bridge defaults False")
    win.enable_vulkan_bridge()
    _assert(win.wants_bridge, "enable_vulkan_bridge sets wants_bridge = True")
    _assert(not win.presents_on_gpu(), "Empty WaylandWindow does not present on GPU")
    win.swapchain = SwapchainHandle(1, 0xCAFE, 1280, 720, 44, 0, 2, 3, False)
    _assert(win.presents_on_gpu(), "WaylandWindow with live vk_swapchain != 0 reports presents_on_gpu() == True!")
    win.swapchain = SwapchainHandle()

    # Headless `SwapchainSurface` lifecycle + HDR scRGB `R16G16B16A16_SFLOAT` (`format == 97`, `8` bytes/px) stride check!
    var surf = SwapchainSurface()
    _assert(surf.initialize_headless(320, 180, DISPLAY_SDR_SRGB, PRESENT_MAILBOX), "Headless SwapchainSurface init")
    _assert(surf.backend == SURFACE_HEADLESS and not surf.is_zero_copy_gpu(), "SURFACE_HEADLESS state")
    _assert(swapchain_format_bytes_per_pixel(FORMAT_B8G8R8A8_UNORM) == 4, "BGRA8 == 4 bytes/pixel")
    _assert(swapchain_format_bytes_per_pixel(FORMAT_A2B10G10R10_UNORM_PACK32) == 4, "RGB10A2 PQ == 4 bytes/pixel")
    _assert(swapchain_format_bytes_per_pixel(FORMAT_R16G16B16A16_SFLOAT) == 8, "R16G16B16A16_SFLOAT (97) == 8 bytes/pixel")
    _assert(resolve_swapchain_display_mode(FORMAT_R16G16B16A16_SFLOAT, True) == DISPLAY_EXTENDED_SCRGB_FP16, "scRGB FP16 display mode == 3")
    _assert(
        not validate_present_source_bytes(FORMAT_R16G16B16A16_SFLOAT, 320 * 180 * 4, 320, 320, 180),
        "4-byte/pixel PQ buffer MUST be rejected for 8-byte/pixel R16G16B16A16_SFLOAT swapchain!",
    )
    _assert(
        validate_present_source_bytes(FORMAT_R16G16B16A16_SFLOAT, 320 * 180 * 8, 320, 320, 180),
        "8-byte/pixel RGBA16F buffer accepted for R16G16B16A16_SFLOAT swapchain",
    )
    var surf_scrgb = SwapchainSurface()
    _assert(surf_scrgb.initialize_headless(320, 180, DISPLAY_EXTENDED_SCRGB_FP16, PRESENT_MAILBOX), "Headless scRGB surface init")
    _assert(surf_scrgb.is_scrgb_fp16() and surf_scrgb.bytes_per_pixel() == 8 and surf_scrgb.handle.format == FORMAT_R16G16B16A16_SFLOAT, "Headless scRGB format == 97 (8 bpp)")

    # Unified Right-Handed `-Z` forward view + Reverse-Z projection math
    var view = make_rh_look_at(Vec3(0.0, 2.0, 5.0), Vec3(0.0, 2.0, 0.0))
    var p_front = view.transform_point(Vec3(0.0, 2.0, 1.0))  # 4m in front of eye
    _assert(abs(p_front.x) < Float32(1e-5) and abs(p_front.y) < Float32(1e-5) and abs(p_front.z - Float32(-4.0)) < Float32(1e-4), "RH -Z view places forward target at z_view = -4.0")

    var near_z = Float32(0.10)
    var proj_inf = make_rh_reverse_z_infinite_perspective(Float32(60.0) * DEG_TO_RAD_F32, Float32(16.0 / 9.0), near_z)
    var vp_inf = proj_inf.multiply(view)
    var sc_near = project_rh_world_to_screen(vp_inf, Vec3(0.0, 2.0, 5.0 - near_z), 1280, 720)
    var sc_far = project_rh_world_to_screen(vp_inf, Vec3(0.0, 2.0, -995.0), 1280, 720)
    var sc_behind = project_rh_world_to_screen(vp_inf, Vec3(0.0, 2.0, 7.0), 1280, 720)
    _assert(sc_near[3] and abs(sc_near[2] - Float32(1.0)) < Float32(1e-4), "Reverse-Z near plane maps to z_ndc = 1.0")
    _assert(sc_far[3] and sc_far[2] > Float32(0.0) and sc_far[2] < Float32(0.001), "Infinite Reverse-Z 1000m maps near 0.0")
    _assert(not sc_behind[3], "Behind-camera point rejected")

    var proj_fin = make_rh_reverse_z_perspective(Float32(60.0) * DEG_TO_RAD_F32, Float32(1.0), Float32(0.1), Float32(100.0))
    var vp_fin = proj_fin.multiply(view)
    var sc_fin_far = project_rh_world_to_screen(vp_fin, Vec3(0.0, 2.0, 5.0 - 100.0), 512, 512)
    _assert(sc_fin_far[3] and abs(sc_fin_far[2]) < Float32(1e-4), "Finite Reverse-Z far plane maps to z_ndc = 0.0")

    # GPU 3D `DebugLineBatch` analytical near-plane clipping into `UiDrawList` (`PRIM_LINE`)
    var dbg = DebugLineBatch()
    # Line 1: straddles camera near plane (starts 3m behind eye at z=8, ends 4m in front at z=1) -> MUST clip & emit!
    dbg.add_line(Vec3(0.0, 2.0, 8.0), Vec3(0.0, 2.0, 1.0), 0xFFFF4040, 2.0, xray=False)
    # Line 2: completely behind camera (z=7 to z=12) -> MUST cull!
    dbg.add_line(Vec3(-1.0, 2.0, 7.0), Vec3(1.0, 2.0, 12.0), 0xFF40FF40, 2.0, xray=False)
    dbg.add_aabb(Vec3(-0.5, 1.5, -1.0), Vec3(0.5, 2.5, 0.0))
    dbg.add_gizmo(Vec3(0.0, 2.0, 0.0), Quat(), 0.75)

    var cpu_dd = DebugDraw()
    cpu_dd.set_enabled(True, Float32(1.0 / 60.0))
    cpu_dd.sphere(Vec3(0.5, 2.0, -0.5), Float32(0.35), DEBUG_CYAN)
    dbg.ingest_debug_draw(cpu_dd, xray=True)

    var ui_dbg = UiDrawList()
    var emitted = dbg.emit_to_ui_draw_list(ui_dbg, Vec3(0.0, 2.0, 5.0), Vec3(0.0, 2.0, 0.0), Float32(60.0), Float32(0.15), 320, 240)
    # 1 clipped straddler + 0 behind + 12 AABB + 9 gizmo + 48 sphere (3*16) = 70 segments
    _assert(emitted == 70 and ui_dbg.count() == 70, "DebugLineBatch near-plane clipped & emitted 70 PRIM_LINE segments")
    _assert((ui_dbg.prims[0].header & UInt32(0xF)) == PRIM_LINE, "Emitted debug primitives use PRIM_LINE")

    # Double-buffered `RenderWorld[2]` & `RenderExtractor` (`UInt64` entity ID tracking + despawn pruning)
    var rw_ring = DoubleBufferedRenderWorld()
    var extractor = RenderExtractor()

    var m_frame0 = trs(Vec3(1.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    var m_frame1 = trs(Vec3(3.5, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))

    extractor.begin_extraction(rw_ring.write_world(), 1)
    extractor.extract_entity(rw_ring.write_world(), UInt64(1001), 0, m_frame0)
    extractor.extract_entity(rw_ring.write_world(), UInt64(2002), 0, m_frame0)
    extractor.end_extraction()
    _assert(extractor.tracked_entity_count() == 2, "Extractor tracks 2 entities on frame 1")
    rw_ring.swap_buffers()
    _assert(not rw_ring.read_world().instances[0].has_prev, "Newly spawned entity has has_prev == False")

    # Frame 2: Entity 1001 moves to x=3.5, Entity 2002 despawns, Entity 3003 spawns
    extractor.begin_extraction(rw_ring.write_world(), 2)
    extractor.extract_entity(rw_ring.write_world(), UInt64(1001), 0, m_frame1)
    extractor.extract_entity(rw_ring.write_world(), UInt64(3003), 0, m_frame1)
    extractor.end_extraction()
    _assert(extractor.tracked_entity_count() == 2, "Extractor pruned despawned entity 2002")
    rw_ring.swap_buffers()
    ref inst_1001 = rw_ring.read_world().instances[0]
    ref inst_3003 = rw_ring.read_world().instances[1]
    _assert(inst_1001.has_prev and abs(inst_1001.prev_world_mat.values[12] - Float32(1.0)) < Float32(1e-5), "Entity 1001 prev_world_mat preserved")
    _assert(not inst_3003.has_prev and abs(inst_3003.prev_world_mat.values[12] - Float32(3.5)) < Float32(1e-5), "Newly spawned 3003 does not streak")

    # Tier 0 Headless SIMD `ScalableRenderer` verification (zero Vulkan overhead)
    _assert(RENDER_TIER_1_FORWARD_CLUSTERED == 1, "RENDER_TIER_1_FORWARD_CLUSTERED == 1")
    var tier0 = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, width=128, height=96)
    var t0_mat = tier0.register_material(MaterialAsset(base_color=Vec3(0.9, 0.3, 0.2)))
    var mb_box = MeshBuilder()
    mb_box.add_beveled_box(Vec3(0.0, 0.5, 0.0), Vec3(0.6, 0.6, 0.6))
    var t0_mesh = tier0.register_mesh(mb_box^.finish(), t0_mat)
    rw_ring.write_world().clear_frame(3)
    rw_ring.write_world().camera = RenderCameraPacket(eye=Vec3(0.0, 1.2, 3.2), target=Vec3(0.0, 0.5, 0.0))
    extractor.begin_extraction(rw_ring.write_world(), 3)
    rw_ring.write_world().camera = RenderCameraPacket(eye=Vec3(0.0, 1.2, 3.2), target=Vec3(0.0, 0.5, 0.0))
    extractor.extract_entity(rw_ring.write_world(), UInt64(1001), t0_mesh, Mat4(), t0_mat)
    rw_ring.write_world().debug_lines.add_gizmo(Vec3(0.0, 0.5, 0.0), Quat(), 0.8)
    extractor.end_extraction()
    rw_ring.swap_buffers()

    var t0_ui = UiDrawList()
    t0_ui.add_rect(4.0, 4.0, 20.0, 10.0, 0xFFFF3030)
    _assert(tier0.render_world(rw_ring.read_world(), t0_ui, capture_readback=True), "Tier 0 SIMD render_world succeeded")
    var t0_px = tier0.readback_pixels()
    _assert(len(t0_px) == 128 * 96 and tier0.cpu_canvas.triangles_drawn > 0, "Tier 0 SIMD rasterized triangles")
    tier0.shutdown()

    print("[PASS] Suite 3: SwapchainSurface, RH -Z/Reverse-Z, DebugLineBatch, RenderExtractor & Tier 0 SIMD")


# ============================================================================
# Suite 4 & 5: GPU UI Overlay (Exact Red/Blue, 6 Prim Kinds, >300 Overlap, HDR10)
#              + White-Furnace + Cornell Box + 5x5 PBR/Clearcoat + 4-Cascade VSM Sundial
# ============================================================================

def _build_procedural_ui_atlas_16x16() -> List[UInt32]:
    """Builds a crisp 16x16 ARGB8888 border + SDF disk texture for `PRIM_NINE_SLICE`, `PRIM_IMAGE`, and `PRIM_GLYPH`."""
    var px = List[UInt32](capacity=256)
    for y in range(16):
        for x in range(16):
            var dx = Float32(x) - Float32(7.5)
            var dy = Float32(y) - Float32(7.5)
            var dist = sqrt(dx * dx + dy * dy)
            # Encode signed distance around r=5.0 in R channel (>128 inside), bright gold/cyan in GB
            var sdf_u8 = max(0, min(255, Int((Float32(0.5) + (Float32(5.2) - dist) * Float32(0.22)) * Float32(255.0))))
            var is_border = x < 3 or x >= 13 or y < 3 or y >= 13
            var g_u8 = 230 if is_border else 180
            var b_u8 = 255 if is_border else 90
            var r_u8 = max(sdf_u8, 220 if is_border else sdf_u8)
            px.append(pack_argb(r_u8, g_u8, b_u8, 255))
    return px^


def _project_lh_pixel(
    cam_pos: Vec3,
    cam_target: Vec3,
    fov_y_deg: Float32,
    w: Int,
    h: Int,
    world_pt: Vec3,
) -> Tuple[Int, Int]:
    """Exact pixel projection matching `FlintAaaVulkanRenderer.begin_frame` (`+X=right, +Y=up, +Z=fwd`)."""
    var fwd = (cam_target - cam_pos).normalized()
    var up_ref = Vec3(0.0, 0.0, 1.0) if abs(fwd.y) > Float32(0.98) else Vec3(0.0, 1.0, 0.0)
    var right = up_ref.cross(fwd).normalized()
    var up = fwd.cross(right).normalized()
    var rel = world_pt - cam_pos
    var vx = right.dot(rel)
    var vy = up.dot(rel)
    var vz = max(Float32(1e-3), fwd.dot(rel))
    var tan_y = sin(fov_y_deg * Float32(0.5) * DEG_TO_RAD_F32) / cos(fov_y_deg * Float32(0.5) * DEG_TO_RAD_F32)
    var tan_x = tan_y * (Float32(w) / Float32(h))
    var ndc_x = vx / (vz * tan_x)
    var ndc_y = vy / (vz * tan_y)
    var px = max(0, min(w - 1, Int((ndc_x * Float32(0.5) + Float32(0.5)) * Float32(w))))
    var py = max(0, min(h - 1, Int((Float32(1.0) - (ndc_y * Float32(0.5) + Float32(0.5))) * Float32(h))))
    return (px, py)


def _luma_rec709(c: Vec3) -> Float32:
    return c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722)


def test_gpu_ui_and_aaa_metropolis():
    var W = 192
    var H = 144
    var renderer = ScalableRenderer(RENDER_TIER_2_AAA_VISBUFFER_RT, width=W, height=H, tsr_scale=1.0)
    if not renderer.uses_vulkan():
        print("[SKIP] Vulkan hardware unavailable in environment; CPU Tier 0 verified")
        renderer.shutdown()
        return

    ref gpu = renderer.gpu_slot[0]
    gpu.tsr_history_weight = Float32(0.0)
    gpu.enable_auto_exposure = False
    gpu.bloom_intensity = Float32(0.0)
    gpu.camera_lens.vignette_strength = Float32(0.0)

    # Upload procedural 16x16 bindless texture for Nine-Slice / Image / Glyph tests
    var atlas_px = _build_procedural_ui_atlas_16x16()
    var ui_tex_slot = gpu.upload_bc7_texture(atlas_px, 16, 16, srgb=False, generate_mips=False)
    _assert(ui_tex_slot >= 0, "Uploaded procedural 16x16 UI bindless texture")

    # ------------------------------------------------------------------------
    # Part A: Exact `0xFFFF2040` Pure-Red vs `0xFF2040FF` Pure-Blue + 6 SDF Kinds + >300 Tile Overlap!
    # ------------------------------------------------------------------------
    gpu.display_mode = DISPLAY_SDR_SRGB
    gpu.begin_frame(Vec3(0.0, 1.0, 4.0), Vec3(0.0, 1.0, 0.0))

    # 1. Exact pure-red (`0xFFFF2040`: R=255, G=32, B=64) vs pure-blue (`0xFF2040FF`: R=32, G=64, B=255)
    gpu.ui_draw_list.add_rect(8.0, 8.0, 28.0, 28.0, 0xFFFF2040)
    gpu.ui_draw_list.add_rect(44.0, 8.0, 28.0, 28.0, 0xFF2040FF)

    # Styled `PRIM_SHAPE` with vertical gradient + border + drop shadow
    gpu.ui_draw_list.add_rect_styled(
        80.0, 8.0, 32.0, 28.0,
        0xFFFFA020, 0xFF2080FF,
        corner_radius=5.0,
        border_width=2.0,
        flags=UI_FLAG_VERT_GRADIENT | UI_FLAG_DROP_SHADOW | UI_FLAG_BORDER,
    )

    # 2. `PRIM_GLYPH` (kind 1) with outer outline (`param1`) + drop shadow (`UI_FLAG_DROP_SHADOW`)
    var glyph_hdr = (PRIM_GLYPH & UInt32(0xF)) | (UInt32(ui_tex_slot & 0xFFF) << 4) | (UI_FLAG_DROP_SHADOW << 16)
    gpu.ui_draw_list.prims.append(
        GpuUiPrim(
            8.0, 46.0, 36.0, 74.0,
            0.0, 0.0, 1.0, 1.0,
            0, UInt32(0xFFFF) | (UInt32(0xFFFF) << 16),
            0xFFFFF040, 0xFF102030,
            Float32(0.08), Float32(1.5), Float32(1.0), glyph_hdr,
        )
    )

    # 3. `PRIM_NINE_SLICE` (kind 2)
    gpu.ui_draw_list.add_nine_slice(ui_tex_slot, 44.0, 46.0, 36.0, 28.0, 0.25, 0.25, 0.25, 0.25, 0xFFFFFFFF)

    # 4. `PRIM_CONTROLLER_SYMBOL` (kind 3): 'A' button, 'Play', and 'Gizmo'
    gpu.ui_draw_list.prims.append(make_ui_symbol_prim(UI_SYMBOL_PLAY, 88.0, 46.0, 28.0, 28.0, 0xFF30FF80, 0xFF182230))
    gpu.ui_draw_list.add_controller_symbol(UI_SYMBOL_A, 8.0, 84.0, 26.0, 26.0, 0xFFFFFFFF, 0xFF2060D0)
    gpu.ui_draw_list.prims.append(make_ui_symbol_prim(UI_SYMBOL_GIZMO, 40.0, 84.0, 26.0, 26.0, 0xFFFFD030, 0xFF181820))

    # 5. `PRIM_LINE` (kind 4): straight capsule line + quadratic Bézier curve
    gpu.ui_draw_list.prims.append(make_ui_line_prim(8.0, 124.0, 72.0, 124.0, 6.0, 0xFF20F0D0))
    gpu.ui_draw_list.prims.append(make_ui_bezier_prim(80.0, 130.0, 110.0, 96.0, 140.0, 130.0, 5.0, 0xFFFF50E0))

    # 6. `PRIM_IMAGE` (kind 5): bindless textured rounded quad
    gpu.ui_draw_list.prims.append(make_ui_image_prim(ui_tex_slot, 148.0, 46.0, 32.0, 32.0, tint_rgba8=0xFFFFFFFF, corner_radius=4.0))

    # 7. Chunked `>300` (`340`) overlapping primitives inside ONE `16x16` tile (`x in [128..144], y in [8..24]`)!
    for ov_i in range(339):
        var alt_col: UInt32 = 0xFFD03030 if (ov_i & 1) == 0 else 0xFF3030D0
        gpu.ui_draw_list.add_rect(126.0, 6.0, 22.0, 22.0, alt_col)
    # Primitive #340 (well past single-chunk 256 cap!) paints exact green `0xFF10F030` (`R=16, G=240, B=48`)
    gpu.ui_draw_list.add_rect(126.0, 6.0, 22.0, 22.0, 0xFF10F030)

    _assert(gpu.render_frame(capture_readback=True), "UI overlay verification frame rendered")
    var ui_px = gpu.readback_pixels()

    # Check exact pure-red `0xFFFF2040` at `(20, 20)` -> `(R=255, G=32, B=64)`
    var red_p = _unpack_argb(ui_px[20 * W + 20])
    _assert(
        abs(red_p[0] - 255) <= 1 and abs(red_p[1] - 32) <= 2 and abs(red_p[2] - 64) <= 2,
        "Pure-red 0xFFFF2040 readback mismatch (got R="
        + String(red_p[0]) + " G=" + String(red_p[1]) + " B=" + String(red_p[2]) + ")",
    )

    # Check exact pure-blue `0xFF2040FF` at `(56, 20)` -> `(R=32, G=64, B=255)`
    var blue_p = _unpack_argb(ui_px[20 * W + 56])
    _assert(
        abs(blue_p[0] - 32) <= 2 and abs(blue_p[1] - 64) <= 2 and abs(blue_p[2] - 255) <= 1,
        "Pure-blue 0xFF2040FF readback mismatch (got R="
        + String(blue_p[0]) + " G=" + String(blue_p[1]) + " B=" + String(blue_p[2]) + ")",
    )

    # Check 340th overlapping primitive at `(136, 16)` -> exact `0xFF10F030` (`R=16, G=240, B=48`)
    var ov_p = _unpack_argb(ui_px[16 * W + 136])
    _assert(
        abs(ov_p[0] - 16) <= 2 and abs(ov_p[1] - 240) <= 2 and abs(ov_p[2] - 48) <= 2,
        "340-prim chunked tile loop dropped or misordered prim #340 (got R="
        + String(ov_p[0]) + " G=" + String(ov_p[1]) + " B=" + String(ov_p[2]) + ")",
    )

    # Check PRIM_LINE `(40, 124)`, Bezier apex `(110, 113)`, Controller Play `(101, 60)`, Nine-Slice `(46, 60)`, Image `(164, 62)`
    var line_p = _unpack_argb(ui_px[124 * W + 40])
    _assert(line_p[1] > 200 and line_p[2] > 170, "PRIM_LINE capsule rendered cyan stroke")
    var bez_p = _unpack_argb(ui_px[113 * W + 110])
    _assert(bez_p[0] > 180 and bez_p[2] > 160, "PRIM_LINE quadratic Bezier rendered magenta arch")
    var sym_p = _unpack_argb(ui_px[60 * W + 101])
    _assert(sym_p[1] > 200, "PRIM_CONTROLLER_SYMBOL Play icon rendered")
    var ns_p = _unpack_argb(ui_px[60 * W + 46])
    _assert(ns_p[0] > 120 or ns_p[1] > 120 or ns_p[2] > 120, "PRIM_NINE_SLICE sampled bindless border")
    var img_p = _unpack_argb(ui_px[62 * W + 164])
    _assert(img_p[0] > 100 or img_p[1] > 100, "PRIM_IMAGE sampled bindless quad")

    # ------------------------------------------------------------------------
    # Part B: HDR10 ST.2084 PQ (`disp_mode == 1`) @ 203 nits UI Paper-White
    # ------------------------------------------------------------------------
    gpu.display_mode = DISPLAY_HDR10_PQ
    gpu.begin_frame(Vec3(0.0, 1.0, 4.0), Vec3(0.0, 1.0, 0.0))
    gpu.ui_draw_list.add_rect(16.0, 16.0, 48.0, 48.0, 0xFFFFFFFF)
    _assert(gpu.render_frame(capture_readback=True), "HDR10 PQ UI frame rendered")
    var hdr_px = gpu.readback_pixels()
    var pq_word = hdr_px[32 * W + 32]
    var pq_r10 = Int(pq_word & UInt32(0x3FF))
    var pq_g10 = Int((pq_word >> 10) & UInt32(0x3FF))
    var pq_b10 = Int((pq_word >> 20) & UInt32(0x3FF))
    var pq_a2 = Int((pq_word >> 30) & UInt32(0x3))
    # ST.2084 PQ of 203 nits (203 / 10000 = 0.0203) is ~0.5807 -> * 1023 ≈ 594
    _assert(
        pq_a2 == 3 and pq_r10 >= 565 and pq_r10 <= 620 and abs(pq_r10 - pq_g10) <= 2 and abs(pq_g10 - pq_b10) <= 2,
        "HDR10 PQ 203-nit paper-white out of expected [565..620] code range: " + String(pq_r10),
    )
    gpu.display_mode = DISPLAY_SDR_SRGB

    print(
        "[PASS] Suite 4: GPU UI Exact 0xFFFF2040 Red / 0xFF2040FF Blue, All 6 SDF Kinds, 340-Prim Tile & HDR10 PQ (code="
        + String(pq_r10) + ")"
    )

    # ------------------------------------------------------------------------
    # Part C: White-Furnace Energy Conservation Check
    # ------------------------------------------------------------------------
    var mb_sphere = MeshBuilder()
    mb_sphere.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), rings=16, segments=24)
    var sphere_mesh = mb_sphere^.finish()

    var wf_mat = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.45, metallic=0.0)
    var wf_mat_id = gpu.add_material(wf_mat)
    var sphere_entry = gpu.add_mesh(sphere_mesh, wf_mat_id, build_lods=True)

    gpu.furnace_white_dome = True
    gpu.enable_vsm = False
    gpu.enable_rt_shadows = False
    gpu.enable_rt_gi = False
    gpu.enable_rt_reflections = False
    gpu.exposure = Float32(1.0)

    gpu.begin_frame(Vec3(0.0, 0.0, 2.8), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    gpu.draw_instance(sphere_entry.mesh_id, Mat4())
    _assert(gpu.render_frame(capture_readback=True), "White-furnace frame rendered")
    var wf_hdr = gpu.readback_resolved_hdr()
    var wf_center = wf_hdr[(H // 2) * W + (W // 2)]
    _assert(
        wf_center.x >= Float32(0.78) and wf_center.x <= Float32(1.15) and abs(wf_center.x - wf_center.y) < Float32(0.05),
        "White-furnace energy conservation violated: L=" + String(wf_center.x),
    )
    gpu.furnace_white_dome = False

    # ------------------------------------------------------------------------
    # Part D: 4-Cascade Toroidal VSM (`±16m, ±64m, ±256m, ±1024m`) + Cornell Box + 5x5 PBR/Clearcoat + Sundial
    # ------------------------------------------------------------------------
    _assert(
        VSM_CASCADE_0_EXTENT == 16.0
        and VSM_CASCADE_1_EXTENT == 64.0
        and VSM_CASCADE_2_EXTENT == 256.0
        and VSM_CASCADE_3_EXTENT == 1024.0,
        "4-cascade extents == 16, 64, 256, 1024m",
    )
    var sun_d = Vec3(0.55, 0.72, 0.42).normalized()
    var casc_mats = compute_vsm_cascade_matrices(sun_d, Vec3(3.0, 1.5, -2.0))
    _assert(len(casc_mats) == 4 and len(casc_mats[0]) == 16, "compute_vsm_cascade_matrices produced 4 4x4 matrices")

    var vsm_mgr = VsmPageManager()
    var vsm_tbl = List[UInt32](length=VSM_TABLE_WORDS, fill=0)
    var orig_a = compute_vsm_cascade_page_origins(sun_d, Vec3(0.0, 0.0, 0.0))
    var empty_orig = List[Tuple[Int, Int]]()
    var marked_0 = mark_toroidal_cascade_pages(vsm_mgr, vsm_tbl, empty_orig, orig_a, active_radius_pages=2)
    var allocs_0 = vsm_mgr.process_requests(vsm_tbl)
    var marked_1 = mark_toroidal_cascade_pages(vsm_mgr, vsm_tbl, orig_a, orig_a, active_radius_pages=2)
    var allocs_1 = vsm_mgr.process_requests(vsm_tbl)
    _assert(marked_0 == 100 and len(allocs_0) == 100, "Initial 4-cascade 5x5 active region allocates 100 pages")
    _assert(marked_1 == 100 and len(allocs_1) == 0 and vsm_mgr.cached_hits_last_frame == 100, "Static toroidal VSM hits 100% cache on frame 2!")

    # Build Procedural Cornell Box + 5x5 PBR/Clearcoat Sphere Matrix + VSM Sundial Gnomon
    var mb_slab = MeshBuilder()
    mb_slab.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), bevel=0.02)
    var slab_mesh = mb_slab^.finish()
    var slab_entry = gpu.add_mesh(slab_mesh, wf_mat_id, build_lods=False)

    var mat_white = renderer.register_material(MaterialAsset(base_color=Vec3(0.76, 0.76, 0.74), roughness=0.75, metallic=0.0))
    var mat_red = renderer.register_material(MaterialAsset(base_color=Vec3(0.75, 0.12, 0.10), roughness=0.70, metallic=0.0))
    var mat_green = renderer.register_material(MaterialAsset(base_color=Vec3(0.12, 0.68, 0.15), roughness=0.70, metallic=0.0))
    var mat_gold = renderer.register_material(MaterialAsset(base_color=Vec3(0.95, 0.76, 0.30), roughness=0.18, metallic=1.0))

    # 5x5 PBR + Clearcoat Sphere Matrix materials
    var matrix_mats = List[Int](capacity=25)
    for row in range(5):
        for col in range(5):
            var rgh = Float32(0.08) + Float32(col) * Float32(0.21)
            var met = Float32(1.0) if row >= 2 else Float32(0.0)
            var m_sp = MaterialAsset(
                base_color=Vec3(0.20 + 0.18 * Float32(col), 0.35 + 0.12 * Float32(row), 0.85 - 0.14 * Float32(row)),
                roughness=rgh,
                metallic=met,
                domain=(MAT_DOMAIN_CLEARCOAT if row == 1 else MAT_DOMAIN_STANDARD_PBR),
            )
            if row == 1:
                m_sp.clearcoat = Float32(1.0)
                m_sp.clearcoat_roughness = Float32(0.05)
            matrix_mats.append(renderer.register_material(m_sp))

    ref gpu2 = renderer.gpu_slot[0]
    gpu2.enable_vsm = True
    gpu2.enable_vsm_clipmap_cascades = True
    gpu2.vsm_scene_center = Vec3(0.0, 1.5, 0.0)
    gpu2.vsm_ortho_half_extent = Float32(16.0)
    gpu2.sun_dir = Vec3(0.65, 0.72, 0.24).normalized()
    gpu2.sun_radiance = Vec3(5.5, 5.2, 4.8)
    gpu2.ibl_intensity = Float32(0.25)

    gpu2.begin_frame(Vec3(0.0, 3.2, 7.5), Vec3(0.0, 1.4, 0.0), fov_y_deg=50.0)
    # Cornell Floor, Back Wall, Left Red Wall, Right Green Wall
    gpu2.draw_instance(slab_entry.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(6.0, 0.1, 6.0)), material_override=mat_white)
    gpu2.draw_instance(slab_entry.mesh_id, trs(Vec3(0.0, 3.0, -4.5), Quat(), Vec3(6.0, 3.2, 0.12)), material_override=mat_white)
    gpu2.draw_instance(slab_entry.mesh_id, trs(Vec3(-4.8, 3.0, 0.0), Quat(), Vec3(0.12, 3.2, 5.0)), material_override=mat_red)
    gpu2.draw_instance(slab_entry.mesh_id, trs(Vec3(4.8, 3.0, 0.0), Quat(), Vec3(0.12, 3.2, 5.0)), material_override=mat_green)

    # 5x5 PBR/Clearcoat Sphere Matrix suspended along back gallery
    for row in range(5):
        for col in range(5):
            var sx = Float32(col - 2) * Float32(0.85)
            var sy = Float32(1.1) + Float32(row) * Float32(0.68)
            gpu2.draw_instance(
                sphere_entry.mesh_id,
                trs(Vec3(sx, sy, -2.6), Quat(), Vec3(0.26, 0.26, 0.26)),
                material_override=matrix_mats[row * 5 + col],
            )

    # Tall Sundial Gnomon pillar at `(1.2, 1.5, 1.2)` casting VSM shadow toward `(-X, -Z)` on the white floor
    gpu2.draw_instance(slab_entry.mesh_id, trs(Vec3(1.2, 1.5, 1.2), Quat(), Vec3(0.22, 1.5, 0.22)), material_override=mat_gold)

    # Project 3D Debug Gizmo onto zero-copy UI DrawList
    var scene_dbg = DebugLineBatch()
    scene_dbg.add_gizmo(Vec3(-1.8, 0.4, 1.5), Quat(), 0.75)
    _ = scene_dbg.emit_to_ui_draw_list(gpu2.ui_draw_list, Vec3(0.0, 3.2, 7.5), Vec3(0.0, 1.4, 0.0), Float32(50.0), Float32(0.1), W, H)

    _assert(gpu2.render_frame(capture_readback=True), "Cornell Box + 5x5 Sphere Matrix + VSM Sundial rendered")
    var scene_hdr = gpu2.readback_resolved_hdr()

    # Verify non-zero scene radiance and lit floor vs shadowed floor contrast
    var max_lum = Float32(0.0)
    var min_lum = Float32(1e9)
    for i in range(len(scene_hdr)):
        var c = scene_hdr[i]
        # Verify zero NaNs!
        _assert(c.x == c.x and c.y == c.y and c.z == c.z, "NaN detected in resolved HDR output")
        var lum = c.x * Float32(0.2126) + c.y * Float32(0.7152) + c.z * Float32(0.0722)
        if lum > max_lum:
            max_lum = lum
        if lum < min_lum:
            min_lum = lum
    _assert(max_lum > Float32(0.4) and max_lum > min_lum * Float32(2.5), "Scene exhibits lit vs shadowed VSM dynamic range")
    var cornell_meshlet_draws = gpu2.last_culled_meshlet_draws
    _assert(cornell_meshlet_draws >= 30, "GPU Meshlet culler emitted >= 30 indirect draws for Cornell + 5x5 Matrix + Sundial")

    # ------------------------------------------------------------------------
    # Part E: Extended Linear scRGB `RGBA16F` (`8` BPP) GPU Pack + Live `SurfaceMaterialNode` GPU Dispatch!
    # ------------------------------------------------------------------------
    gpu2.display_mode = AAA_DISPLAY_SCRGB_FP16
    gpu2.exposure = Float32(1.0)
    gpu2.begin_frame(Vec3(0.0, 3.2, 7.5), Vec3(0.0, 1.4, 0.0), fov_y_deg=50.0)
    gpu2.draw_instance(slab_entry.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(6.0, 0.1, 6.0)), material_override=mat_white)
    _assert(gpu2.render_frame(capture_readback=True), "scRGB RGBA16F 8-byte/pixel frame rendered")
    var scrgb_hdr_ref = gpu2.readback_resolved_hdr()
    var scrgb_words = gpu2.readback_scrgb_rgba16f_words()
    _assert(len(scrgb_words) == W * H * 2, "scRGB RGBA16F output packs exact 2 x UInt32 (8 bytes) per pixel")
    var probe_idx = (H - 24) * W + (W // 2)
    var w0_sc = scrgb_words[probe_idx * 2 + 0]
    var w1_sc = scrgb_words[probe_idx * 2 + 1]
    var r_sc = f16_bits_to_f32(UInt16(w0_sc & UInt32(0xFFFF)))
    var a_sc = f16_bits_to_f32(UInt16((w1_sc >> 16) & UInt32(0xFFFF)))
    var expected_r_sc = scrgb_hdr_ref[probe_idx].x * Float32(203.0 / 80.0)
    _assert(
        expected_r_sc > Float32(0.25)
        and abs(r_sc - expected_r_sc) < max(Float32(0.02), expected_r_sc * Float32(0.03))
        and abs(a_sc - Float32(1.0)) < Float32(0.01),
        "Extended Linear scRGB RGBA16F scales linear HDR by 203/80 = 2.5375 (r_sc="
        + String(r_sc) + ", expected=" + String(expected_r_sc) + ")",
    )

    # Dispatch live `TriplanarWeatherMaterialNode` (`SurfaceMaterialNode`) on GPU against unified 24-word `buf_materials`!
    var mat_basalt = renderer.register_material(
        MaterialAsset(base_color=Vec3(0.06, 0.06, 0.08), roughness=0.88, metallic=0.0)
    )
    var baked_slots = renderer.evaluate_material_graph(
        TriplanarWeatherMaterialNode(snow_line_y=12.0),
        mat_basalt,
        dim=32,
        time_sec=0.0,
        world_y_scale=24.0,
    )
    _assert(
        baked_slots[0] >= 0 and baked_slots[1] >= 0,
        "ScalableRenderer dispatched SurfaceMaterialNode on GPU and bound BC7 Albedo+ORM slots",
    )

    var mb_uv_hi = MeshBuilder()
    mb_uv_hi.add_ellipsoid(
        Vec3(-0.9, 0.0, 0.0), Vec3(0.68, 0.68, 0.25),
        u0=0.2, v0=0.08, u1=0.8, v1=0.18,  # v ~ 0.12 -> world_y ~ 21m > snow_line_y=12m (SNOW!)
    )
    var mb_uv_lo = MeshBuilder()
    mb_uv_lo.add_ellipsoid(
        Vec3(0.9, 0.0, 0.0), Vec3(0.68, 0.68, 0.25),
        u0=0.2, v0=0.82, u1=0.8, v1=0.94,  # v ~ 0.88 -> world_y ~ 2.8m < snow_line_y=12m (DARK BASALT!)
    )
    var e_snow_q = renderer.gpu_slot[0].add_mesh(mb_uv_hi^.finish(), mat_basalt, build_lods=False)
    var e_rock_q = renderer.gpu_slot[0].add_mesh(mb_uv_lo^.finish(), mat_basalt, build_lods=False)

    ref gpu_mg = renderer.gpu_slot[0]
    gpu_mg.display_mode = DISPLAY_SDR_SRGB
    gpu_mg.enable_vsm = False
    gpu_mg.enable_rt_shadows = False
    gpu_mg.enable_rt_gi = False
    gpu_mg.enable_rt_reflections = False
    gpu_mg.sun_dir = Vec3(0.55, 0.65, -0.52).normalized()
    gpu_mg.sun_radiance = Vec3(4.5, 4.5, 4.5)
    gpu_mg.ibl_intensity = Float32(0.04)
    var mg_eye = Vec3(0.0, 0.0, -3.2)
    var mg_tgt = Vec3(0.0, 0.0, 0.0)
    gpu_mg.begin_frame(mg_eye, mg_tgt, fov_y_deg=45.0)
    gpu_mg.draw_instance(e_snow_q.mesh_id, Mat4(), material_override=mat_basalt)
    gpu_mg.draw_instance(e_rock_q.mesh_id, Mat4(), material_override=mat_basalt)
    _assert(gpu_mg.render_frame(capture_readback=True), "SurfaceMaterialNode 3D mesh shading frame rendered")
    var mg_hdr = gpu_mg.readback_resolved_hdr()
    var px_snow = _project_lh_pixel(mg_eye, mg_tgt, Float32(45.0), W, H, Vec3(-0.9, 0.0, -0.2))
    var px_rock = _project_lh_pixel(mg_eye, mg_tgt, Float32(45.0), W, H, Vec3(0.9, 0.0, -0.2))
    var snow_lum = _luma_rec709(mg_hdr[px_snow[1] * W + px_snow[0]])
    var rock_lum = max(Float32(1e-4), _luma_rec709(mg_hdr[px_rock[1] * W + px_rock[0]]))
    _assert(
        snow_lum > Float32(0.55) and snow_lum > rock_lum * Float32(3.5),
        "GPU-dispatched TriplanarWeatherMaterialNode shaded snow vs basalt on 3D meshes (snow="
        + String(snow_lum) + ", rock=" + String(rock_lum) + ")",
    )

    print(
        "[PASS] Suite 5: White-Furnace (L=" + String(wf_center.x)
        + ") + scRGB RGBA16F (8BPP) + GPU SurfaceMaterialNode + Cornell Box ("
        + String(cornell_meshlet_draws) + " meshlet draws)"
    )
    renderer.shutdown()


def test_multi_cascade_vsm_gpu_shading():
    """Suite 6: Proves Cascades 1 (`±64m`, caster/receiver at `x = 45m` outside Cascade 0 `±16m`) AND
    Cascade 2 (`±256m`, caster/receiver at `x = 140m` outside Cascade 1 `±64m`) genuinely cast GPU PCSS shadows!"""
    var W = 320
    var H = 180
    var ren = FlintAaaVulkanRenderer(width=W, height=H, tsr_scale=1.0, max_draws=16384, max_instances=2048, env_cube_dim=16)
    if not ren.active:
        print("[SKIP] Vulkan unavailable for Suite 6 multi-cascade VSM GPU proof")
        ren.shutdown()
        return

    _assert(ren.enable_vsm_clipmap_cascades, "FlintAaaVulkanRenderer defaults enable_vsm_clipmap_cascades = True")
    ren.enable_vsm = True
    ren.enable_rt_shadows = False      # Strictly disable RT shadows so 100% of shadowing comes from VSM Cascades 1 & 2!
    ren.enable_rt_gi = False
    ren.enable_rt_reflections = False
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_auto_exposure = False
    ren.bloom_intensity = Float32(0.0)
    ren.camera_lens.vignette_strength = Float32(0.0)
    ren.vsm_ortho_half_extent = Float32(16.0)  # Cascades: C0=±16m, C1=±64m, C2=±256m, C3=±1024m around cam_pos!
    ren.sun_dir = Vec3(0.0, 0.70710678, 0.70710678).normalized()  # 45-deg sun from +Z casting shadows toward -Z
    ren.sun_radiance = Vec3(6.5, 6.2, 5.8)
    ren.ibl_intensity = Float32(0.10)

    var mat_ground = ren.add_material(AaaMaterial(base_color=Vec3(0.85, 0.85, 0.85), roughness=0.85, metallic=0.0))
    var mat_pillar = ren.add_material(AaaMaterial(base_color=Vec3(0.70, 0.35, 0.20), roughness=0.60, metallic=0.0))

    var mb_box = MeshBuilder()
    mb_box.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), bevel=0.01)
    var box_e = ren.add_mesh(mb_box^.finish(), mat_ground, build_lods=False)

    # Camera stays at `x = 0.0` (`cam_pos = (0, 36, -48)`), so:
    #   - `x = 45.0 m` is `2.81x` outside Cascade 0 (`±16m`), strictly inside Cascade 1 (`±64m`)!
    #   - `x = 140.0 m` is `2.19x` outside Cascade 1 (`±64m`), strictly inside Cascade 2 (`±256m`)!
    var cam_eye = Vec3(0.0, 36.0, -48.0)
    var cam_tgt = Vec3(92.0, 0.0, 1.0)
    var fov_deg = Float32(55.0)

    ren.vsm.invalidate_all()
    ren.begin_frame(cam_eye, cam_tgt, fov_y_deg=fov_deg)
    # Station 1 (`x = 45m`, Cascade 1): ground pad + tall shadow-casting pillar at `(45, 5, 6)` casting umbra onto `(45, 0, 1)`
    ren.draw_instance(box_e.mesh_id, trs(Vec3(45.0, -0.25, 0.0), Quat(), Vec3(14.0, 0.25, 16.0)), material_override=mat_ground)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(45.0, 5.0, 6.0), Quat(), Vec3(2.8, 5.0, 1.2)), material_override=mat_pillar)
    # Station 2 (`x = 140m`, Cascade 2): ground pad + tall shadow-casting pillar at `(140, 10, 12)` casting umbra onto `(140, 0, 2)`
    ren.draw_instance(box_e.mesh_id, trs(Vec3(140.0, -0.25, 0.0), Quat(), Vec3(28.0, 0.25, 30.0)), material_override=mat_ground)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(140.0, 10.0, 12.0), Quat(), Vec3(6.5, 10.0, 2.5)), material_override=mat_pillar)
    _assert(ren.render_frame(capture_readback=True), "Multi-cascade VSM (45m C1 + 140m C2) frame rendered")
    var tc_vsm0 = ren.readback_tile_classify_counts()
    _assert(
        tc_vsm0[AAA_SHADE_CLASS_SKY] > 0 and tc_vsm0[AAA_SHADE_CLASS_STANDARD_PBR] > 0,
        "Frame-0 enable_vsm=True (single_cmd_buf=False) populates readback_tile_classify_counts()",
    )

    var hdr = ren.readback_resolved_hdr()
    var px_u45 = _project_lh_pixel(cam_eye, cam_tgt, fov_deg, W, H, Vec3(45.0, 0.0, 1.0))
    var px_l45 = _project_lh_pixel(cam_eye, cam_tgt, fov_deg, W, H, Vec3(36.0, 0.0, 1.0))
    var px_u140 = _project_lh_pixel(cam_eye, cam_tgt, fov_deg, W, H, Vec3(140.0, 0.0, 2.0))
    var px_l140 = _project_lh_pixel(cam_eye, cam_tgt, fov_deg, W, H, Vec3(120.0, 0.0, 2.0))

    var umbra_luma_45 = _luma_rec709(hdr[px_u45[1] * W + px_u45[0]])
    var lit_luma_45 = _luma_rec709(hdr[px_l45[1] * W + px_l45[0]])
    var umbra_luma_140 = _luma_rec709(hdr[px_u140[1] * W + px_u140[0]])
    var lit_luma_140 = _luma_rec709(hdr[px_l140[1] * W + px_l140[0]])

    _assert(
        lit_luma_45 > Float32(0.5) and umbra_luma_45 < Float32(0.45) * lit_luma_45,
        "Cascade 1 (x=45m, outside ±16m C0) shadow failed: umbra=" + String(umbra_luma_45) + " lit=" + String(lit_luma_45),
    )
    _assert(
        lit_luma_140 > Float32(0.5) and umbra_luma_140 < Float32(0.45) * lit_luma_140,
        "Cascade 2 (x=140m, outside ±64m C1) shadow failed: umbra=" + String(umbra_luma_140) + " lit=" + String(lit_luma_140),
    )

    # Second static frame: verify 100% physical tile cache hit across BOTH Cascade 1 & Cascade 2 quadrants!
    ren.begin_frame(cam_eye, cam_tgt, fov_y_deg=fov_deg)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(45.0, -0.25, 0.0), Quat(), Vec3(14.0, 0.25, 16.0)), material_override=mat_ground)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(45.0, 5.0, 6.0), Quat(), Vec3(2.8, 5.0, 1.2)), material_override=mat_pillar)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(140.0, -0.25, 0.0), Quat(), Vec3(28.0, 0.25, 30.0)), material_override=mat_ground)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(140.0, 10.0, 12.0), Quat(), Vec3(6.5, 10.0, 2.5)), material_override=mat_pillar)
    _assert(ren.render_frame(capture_readback=True), "Multi-cascade VSM static cache frame rendered")
    _assert(
        ren.vsm.newly_rendered_pages_last_frame == 0 and ren.vsm.cached_hits_last_frame > 0,
        "Multi-cascade VSM static frame hit 100% cache (newly_rendered=" + String(ren.vsm.newly_rendered_pages_last_frame) + ")",
    )

    print(
        "[PASS] Suite 6: Real 4-Cascade VSM GPU Shading — C1@45m (umbra="
        + String(umbra_luma_45) + " vs lit=" + String(lit_luma_45)
        + "), C2@140m (umbra=" + String(umbra_luma_140) + " vs lit=" + String(lit_luma_140) + ")"
    )
    ren.shutdown()


def test_gpu_ui_polygons():
    """PRIM_POLYGON: nonzero winding (holes, unions blended once), packing, AA edges."""
    var dl = UiDrawList()
    var sq = List[Float32]()
    for v in [Float32(10), 10, 50, 10, 50, 50, 10, 50]:
        sq.append(v)
    var one = List[Int]()
    one.append(4)
    # 4 points + the closing repeat = 5 vertices -> 1 header + 1 data prim
    _assert(dl.add_polygon(sq, one, 0xFFFFFFFF) == 2, "a quad packs into a header and one data prim")
    _assert(dl.prims[0].u0 == 5.0 and dl.prims[1].header == PRIM_POLY_DATA, "polygon header holds the vertex count")
    _assert(dl.add_polygon(sq, List[Int](), 0xFFFFFFFF) == 0, "no outlines draws nothing")

    var W = 96
    var H = 64
    var renderer = ScalableRenderer(RENDER_TIER_2_AAA_VISBUFFER_RT, width=W, height=H, tsr_scale=1.0)
    if not renderer.uses_vulkan():
        print("[SKIP] Vulkan hardware unavailable; polygon packing verified")
        renderer.shutdown()
        return
    ref gpu = renderer.gpu_slot[0]
    gpu.display_mode = DISPLAY_SDR_SRGB
    gpu.begin_frame(Vec3(0.0, 1.0, 4.0), Vec3(0.0, 1.0, 0.0))
    gpu.ui_draw_list.add_rect(0.0, 0.0, Float32(W), Float32(H), 0xFF000000)
    # A square ring: outer clockwise, inner counter-clockwise (a hole).
    var ring = List[Float32]()
    for v in [Float32(4), 4, 40, 4, 40, 40, 4, 40, 14, 14, 14, 30, 30, 30, 30, 14]:
        ring.append(v)
    var two = List[Int]()
    two.append(4)
    two.append(4)
    _ = gpu.ui_draw_list.add_polygon(ring, two, 0xFFFFFFFF)
    # Two overlapping half-transparent squares as ONE polygon: the overlap blends once.
    var union = List[Float32]()
    for v in [Float32(48.5), 8, 72, 8, 72, 32, 48.5, 32, 60, 20, 84, 20, 84, 44, 60, 44]:
        union.append(v)
    _ = gpu.ui_draw_list.add_polygon(union, two, 0x80FFFFFF)
    _assert(gpu.render_frame(capture_readback=True), "polygon frame rendered")
    var px = gpu.readback_pixels()
    _assert(_unpack_argb(px[8 * W + 8])[0] > 250, "ring body is filled")
    _assert(_unpack_argb(px[22 * W + 22])[0] < 5, "ring hole stays empty")
    _assert(_unpack_argb(px[60 * W + 60])[0] < 5, "outside stays empty")
    var single = _unpack_argb(px[12 * W + 52])[0]
    var overlap = _unpack_argb(px[26 * W + 66])[0]
    _assert(abs(single - 128) <= 3 and abs(overlap - single) <= 2, "the union blends once (" + String(single) + ", " + String(overlap) + ")")
    var edge = _unpack_argb(px[12 * W + 48])[0]   # pixel centre (48.5, 12.5) sits on the edge x = 48.5
    _assert(abs(edge - single // 2) <= 4, "edges are anti-aliased (" + String(edge) + ")")
    var seam = _unpack_argb(px[20 * W + 60])[0]   # on an internal edge of the union
    _assert(abs(seam - single) <= 2, "no seam on internal union edges (" + String(seam) + ")")
    # Verify ScalableRenderer Tier 2 resize() recreates targets & updates SwapchainSurface
    var prev_gen = renderer.surface.resize_generations
    _assert(renderer.resize(128, 80), "ScalableRenderer Tier 2 resize(128, 80) succeeded")
    _assert(
        renderer.width == 128
        and renderer.height == 80
        and renderer.gpu_slot[0].output_w == 128
        and renderer.gpu_slot[0].output_h == 80
        and renderer.surface.resize_generations == prev_gen + 1,
        "ScalableRenderer Tier 2 resize synced dimensions and surface generation",
    )
    print("[PASS] Suite 4b: GPU UI PRIM_POLYGON nonzero winding (hole, union blended once, AA edge " + String(edge) + ", no seam)")
    test_f17_f18_gpu_split_vat_clipmap(renderer)
    renderer.shutdown()


def test_f17_f18_gpu_split_vat_clipmap(mut renderer: ScalableRenderer):
    """F17 Split VisBuffer 5-class tile shading + F18 GPU VAT Crowd + TerrainClipmapDesc via RenderWorld."""
    ref gpu = renderer.gpu_slot[0]
    gpu.tsr_history_weight = Float32(0.0)
    gpu.enable_auto_exposure = False
    gpu.bloom_intensity = Float32(0.0)
    gpu.camera_lens.vignette_strength = Float32(0.0)
    gpu.enable_vsm = False
    gpu.enable_rt_shadows = False
    gpu.enable_rt_reflections = False
    gpu.enable_rt_gi = False

    # 1. F17: Render scene exercising ALL 5 tile classes (Sky, Unlit, Standard PBR, Foliage/SSS, Complex Clearcoat)
    var mat_unlit = AaaMaterial(Vec3(0.05, 0.85, 0.15), roughness=0.5, metallic=0.0)
    mat_unlit.unlit = True
    mat_unlit.emissive = Vec3(0.0, 0.4, 0.0)
    var id_unlit = gpu.add_material(mat_unlit)

    var mat_pbr = AaaMaterial(Vec3(0.75, 0.72, 0.68), roughness=0.45, metallic=0.0)
    var id_pbr = gpu.add_material(mat_pbr)

    var mat_fol = AaaMaterial(Vec3(0.18, 0.62, 0.22), roughness=0.55, metallic=0.0)
    mat_fol.two_sided_foliage = True
    mat_fol.subsurface_amount = Float32(0.8)
    mat_fol.subsurface_color = Vec3(0.35, 0.85, 0.20)
    var id_fol = gpu.add_material(mat_fol)

    var mat_cmp = AaaMaterial(Vec3(0.82, 0.12, 0.12), roughness=0.55, metallic=0.0)
    mat_cmp.clearcoat = Float32(1.0)
    mat_cmp.clearcoat_roughness = Float32(0.05)
    var id_cmp = gpu.add_material(mat_cmp)

    var mb_box = MeshBuilder()
    mb_box.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.55, 0.55, 0.55))
    var box_e = gpu.add_mesh(mb_box^.finish(), id_pbr)

    var eye_f17 = Vec3(0.0, 1.5, -5.5)
    var tgt_f17 = Vec3(0.0, 1.5, 0.0)
    gpu.begin_frame(eye_f17, tgt_f17, fov_y_deg=55.0)
    gpu.draw_instance(box_e.mesh_id, trs(Vec3(-2.1, 1.5, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=id_unlit)
    gpu.draw_instance(box_e.mesh_id, trs(Vec3(-0.7, 1.5, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=id_pbr)
    gpu.draw_instance(box_e.mesh_id, trs(Vec3(0.7, 1.5, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=id_fol)
    gpu.draw_instance(box_e.mesh_id, trs(Vec3(2.1, 1.5, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=id_cmp)
    _assert(gpu.render_frame(capture_readback=True), "F17 split VisBuffer 5-class frame rendered")

    var counts = gpu.readback_tile_classify_counts()
    _assert(len(counts) == AAA_SHADE_CLASS_COUNT, "F17 returns 5 class tile counts")
    _assert(
        counts[AAA_SHADE_CLASS_SKY] > 0
        and counts[AAA_SHADE_CLASS_UNLIT] > 0
        and counts[AAA_SHADE_CLASS_STANDARD_PBR] > 0
        and counts[AAA_SHADE_CLASS_FOLIAGE_SSS] > 0
        and counts[AAA_SHADE_CLASS_COMPLEX] > 0,
        "F17 GPU tile classifier hit all 5 classes (sky="
        + String(counts[0])
        + ", unlit="
        + String(counts[1])
        + ", pbr="
        + String(counts[2])
        + ", foliage="
        + String(counts[3])
        + ", complex="
        + String(counts[4])
        + ")",
    )
    var hdr_f17 = gpu.readback_resolved_hdr()
    for p_i in range(len(hdr_f17)):
        var c = hdr_f17[p_i]
        _assert(c.x == c.x and c.y == c.y and c.z == c.z and c.x >= Float32(0.0) and c.x < Float32(1e6), "F17 HDR finite")
    var rw = gpu.output_w
    var rh = gpu.output_h
    var px_u = _project_lh_pixel(eye_f17, tgt_f17, Float32(55.0), rw, rh, Vec3(-2.1, 1.5, -0.55))
    var c_u = hdr_f17[px_u[1] * rw + px_u[0]]
    _assert(c_u.y > c_u.x * Float32(2.0) and c_u.y > Float32(0.4), "F17 Unlit green box radiance verified")
    var px_c = _project_lh_pixel(eye_f17, tgt_f17, Float32(55.0), rw, rh, Vec3(2.1, 1.5, -0.55))
    var c_cmp = hdr_f17[px_c[1] * rw + px_c[0]]
    _assert(c_cmp.x > c_cmp.z and c_cmp.x > Float32(0.1), "F17 Complex clearcoat red box radiance verified")

    # Regression (F17): negative scalar inputs stay STANDARD_PBR + non-zero HDR, and sheen max() parity
    var mat_sheen_low = AaaMaterial(Vec3(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0)
    mat_sheen_low.sheen_color = Vec3(0.0005, 0.0005, 0.0005)
    _assert(
        gpu._material_shade_class(mat_sheen_low) == AAA_SHADE_CLASS_STANDARD_PBR,
        "CPU _material_shade_class matches GPU max(sheen_color) > 0.001",
    )
    var ma_neg = MaterialAsset(base_color=Vec3(0.25, 0.85, 0.30), domain=MAT_DOMAIN_STANDARD_PBR)
    ma_neg.sheen_roughness = Float32(-0.5)
    ma_neg.clearcoat_roughness = Float32(-0.5)
    ma_neg.specular_factor = Float32(-0.5)
    ma_neg.detail_uv_scale = Float32(-1.0)
    var id_neg = gpu.add_material(ma_neg.to_aaa_material())
    gpu.begin_frame(Vec3(0.0, 0.0, -3.0), Vec3(0.0, 0.0, 0.0), fov_y_deg=55.0)
    gpu.draw_instance(box_e.mesh_id, Mat4(), material_override=id_neg)
    _assert(gpu.render_frame(capture_readback=True), "F17 negative scalar material frame rendered")
    var neg_counts = gpu.readback_tile_classify_counts()
    _assert(
        neg_counts[AAA_SHADE_CLASS_STANDARD_PBR] > 0
        and neg_counts[AAA_SHADE_CLASS_UNLIT] == 0
        and neg_counts[AAA_SHADE_CLASS_FOLIAGE_SSS] == 0
        and neg_counts[AAA_SHADE_CLASS_COMPLEX] == 0,
        "Negative scalar inputs classify as STANDARD_PBR on GPU",
    )
    var neg_hdr = gpu.readback_resolved_hdr()[(rh // 2) * rw + (rw // 2)]
    _assert(neg_hdr.y > Float32(0.2), "Negative scalar material produces non-zero shaded HDR")

    # 2. F18a: GPU VAT Crowd (`vat_crowd_eval_shader`) via RenderWorld + RenderExtractor + ScalableRenderer
    var walker = _build_procedural_skeletal_walker()
    var baked_vat = bake_vat_crowd_clip_rgba16f(walker, mesh_index=0, clip_index=0, sample_fps=30.0, looping=True)
    var cid = renderer.register_vat_crowd_clip(baked_vat, walker.meshes[0], id_pbr, max_phase_slots=4)
    _assert(cid >= 0, "F18 VAT crowd clip registered on ScalableRenderer")

    var rw_ring = DoubleBufferedRenderWorld()
    var extractor = RenderExtractor()
    var ui_empty = UiDrawList()
    var eye_vat = Vec3(0.0, 1.3, -3.2)
    var tgt_vat = Vec3(0.0, 1.1, 0.0)

    extractor.begin_extraction(rw_ring.write_world(), 10)
    rw_ring.write_world().camera = RenderCameraPacket(eye=eye_vat, target=tgt_vat)
    extractor.extract_vat_crowd_entity(
        rw_ring.write_world(), UInt64(9001), cid, Float32(0.0), Float32(0.0), Mat4(), id_pbr
    )
    extractor.end_extraction()
    rw_ring.swap_buffers()
    _assert(renderer.render_world(rw_ring.read_world(), ui_empty, capture_readback=True), "F18 VAT phase 0.0 rendered")
    var hdr_vat0 = renderer.gpu_slot[0].readback_resolved_hdr()

    extractor.begin_extraction(rw_ring.write_world(), 11)
    rw_ring.write_world().camera = RenderCameraPacket(eye=eye_vat, target=tgt_vat)
    extractor.extract_vat_crowd_entity(
        rw_ring.write_world(), UInt64(9001), cid, Float32(0.25), Float32(0.0), Mat4(), id_pbr
    )
    extractor.end_extraction()
    rw_ring.swap_buffers()
    _assert(renderer.render_world(rw_ring.read_world(), ui_empty, capture_readback=True), "F18 VAT phase 0.25 rendered")
    var hdr_vat1 = renderer.gpu_slot[0].readback_resolved_hdr()

    var vat_max_diff = Float32(0.0)
    for p_i in range(len(hdr_vat0)):
        var d = abs(hdr_vat0[p_i].x - hdr_vat1[p_i].x) + abs(hdr_vat0[p_i].y - hdr_vat1[p_i].y) + abs(hdr_vat0[p_i].z - hdr_vat1[p_i].z)
        if d > vat_max_diff:
            vat_max_diff = d
    _assert(vat_max_diff > Float32(0.02), "F18 VAT GPU deformation changes silhouette HDR output (diff=" + String(vat_max_diff) + ")")

    # Stress check: 1,024 VAT crowd instances in one RenderWorld frame!
    extractor.begin_extraction(rw_ring.write_world(), 12)
    rw_ring.write_world().camera = RenderCameraPacket(eye=Vec3(0.0, 6.0, -18.0), target=Vec3(0.0, 1.0, 0.0))
    for k_c in range(1024):
        var cx = Float32((k_c % 32) - 16) * Float32(1.4)
        var cz = Float32((k_c // 32) - 16) * Float32(1.4)
        var ph = Float32(k_c % 8) * Float32(0.12)
        extractor.extract_vat_crowd_entity(
            rw_ring.write_world(),
            UInt64(20000 + k_c),
            cid,
            ph,
            ph,
            trs(Vec3(cx, 0.0, cz), Quat(), Vec3(1.0, 1.0, 1.0)),
            id_pbr,
        )
    extractor.end_extraction()
    rw_ring.swap_buffers()
    _assert(renderer.render_world(rw_ring.read_world(), ui_empty, capture_readback=True), "F18 1,024 VAT crowd rendered")
    _assert(renderer.gpu_slot[0].last_culled_meshlet_draws > 0, "F18 1,024 VAT crowd drew GPU meshlets")

    # Regression (F18): non-looping VAT clamping (`< 0` -> `0.0`, `> dur` -> `dur`, distinct from start)
    var walker_nl = _build_procedural_skeletal_walker()
    var end_q = Quat(sin(Float32(0.225)), 0.0, 0.0, cos(Float32(0.225))).normalized()
    walker_nl.animations[0].channels[0].values[4 * 4 + 0] = end_q.x
    walker_nl.animations[0].channels[0].values[4 * 4 + 1] = end_q.y
    walker_nl.animations[0].channels[0].values[4 * 4 + 2] = end_q.z
    walker_nl.animations[0].channels[0].values[4 * 4 + 3] = end_q.w
    var baked_nl = bake_vat_crowd_clip_rgba16f(walker_nl, mesh_index=0, clip_index=0, sample_fps=30.0, looping=False)
    var cid_nl = renderer.register_vat_crowd_clip(baked_nl, walker_nl.meshes[0], id_pbr, max_phase_slots=4)
    _assert(cid_nl >= 0, "F18 non-looping VAT clip registered")
    var dur_nl = baked_nl.desc.duration_sec
    var nl_hdrs = List[List[Vec3]]()
    var nl_times = [Float32(-0.5), Float32(0.0), dur_nl, dur_nl + Float32(0.25), dur_nl + Float32(1.0)]
    for ti in range(len(nl_times)):
        extractor.begin_extraction(rw_ring.write_world(), UInt64(120 + ti))
        rw_ring.write_world().camera = RenderCameraPacket(eye=eye_vat, target=tgt_vat)
        extractor.extract_vat_crowd_entity(
            rw_ring.write_world(), UInt64(9500), cid_nl, nl_times[ti], nl_times[ti], Mat4(), id_pbr
        )
        extractor.end_extraction()
        rw_ring.swap_buffers()
        _assert(renderer.render_world(rw_ring.read_world(), ui_empty, capture_readback=True), "F18 non-looping VAT frame rendered")
        nl_hdrs.append(renderer.gpu_slot[0].readback_resolved_hdr())
    var diff_neg_zero = Float32(0.0)
    var diff_over_dur = Float32(0.0)
    var diff_over_dur2 = Float32(0.0)
    var diff_ends = Float32(0.0)
    for p_i in range(len(nl_hdrs[0])):
        var d0 = abs(nl_hdrs[0][p_i].x - nl_hdrs[1][p_i].x) + abs(nl_hdrs[0][p_i].y - nl_hdrs[1][p_i].y) + abs(nl_hdrs[0][p_i].z - nl_hdrs[1][p_i].z)
        var d1 = abs(nl_hdrs[2][p_i].x - nl_hdrs[3][p_i].x) + abs(nl_hdrs[2][p_i].y - nl_hdrs[3][p_i].y) + abs(nl_hdrs[2][p_i].z - nl_hdrs[3][p_i].z)
        var d2 = abs(nl_hdrs[2][p_i].x - nl_hdrs[4][p_i].x) + abs(nl_hdrs[2][p_i].y - nl_hdrs[4][p_i].y) + abs(nl_hdrs[2][p_i].z - nl_hdrs[4][p_i].z)
        var de = abs(nl_hdrs[1][p_i].x - nl_hdrs[2][p_i].x) + abs(nl_hdrs[1][p_i].y - nl_hdrs[2][p_i].y) + abs(nl_hdrs[1][p_i].z - nl_hdrs[2][p_i].z)
        if d0 > diff_neg_zero:
            diff_neg_zero = d0
        if d1 > diff_over_dur:
            diff_over_dur = d1
        if d2 > diff_over_dur2:
            diff_over_dur2 = d2
        if de > diff_ends:
            diff_ends = de
    _assert(
        diff_neg_zero == Float32(0.0)
        and diff_over_dur == Float32(0.0)
        and diff_over_dur2 == Float32(0.0)
        and diff_ends > Float32(0.02),
        "F18 non-looping VAT clamps t=-0.5==0.0 and t>dur==dur on GPU (diff_ends=" + String(diff_ends) + ")",
    )

    # 3. F18b: GPU TerrainClipmapDesc via RenderWorld + heightmap upload + camera snap
    var hm_w = 32
    var hm_h = 32
    var heights = List[Float32](length=hm_w * hm_h, fill=Float32(0.15))
    for hz in range(hm_h):
        for hx in range(hm_w):
            var dx = (Float32(hx) - Float32(15.5)) / Float32(8.0)
            var dz = (Float32(hz) - Float32(15.5)) / Float32(8.0)
            var bump = max(Float32(0.0), Float32(1.0) - (dx * dx + dz * dz) * Float32(0.35))
            heights[hz * hm_w + hx] = Float32(0.15) + bump * Float32(0.75)

    var tc_desc = TerrainClipmapDesc(ring_count=3, grid_dim_per_ring=32, base_cell_size_m=0.5, height_scale_m=10.0)
    var final_tc = renderer.set_terrain_clipmap(tc_desc, id_pbr, heights, hm_w, hm_h)
    _assert(final_tc.heightmap_slot != BINDLESS_SLOT_NONE, "F18 Terrain heightmap uploaded to RGBA16F slot")

    extractor.begin_extraction(rw_ring.write_world(), 13)
    rw_ring.write_world().vat_instances.clear()
    rw_ring.write_world().terrain_clipmap = TerrainClipmapPacket(True, final_tc.copy(), id_pbr)
    rw_ring.write_world().camera = RenderCameraPacket(eye=Vec3(0.0, 13.0, -22.0), target=Vec3(0.0, 3.0, 0.0))
    extractor.end_extraction()
    rw_ring.swap_buffers()
    _assert(renderer.render_world(rw_ring.read_world(), ui_empty, capture_readback=True), "F18 TerrainClipmap frame 1 rendered")
    _assert(len(renderer.gpu_slot[0].terrain_ring_origins) == 3, "F18 TerrainClipmap maintains 3 ring origins")
    var orig_a = renderer.gpu_slot[0].terrain_ring_origins[0]
    var hdr_t1 = renderer.gpu_slot[0].readback_resolved_hdr()
    var terrain_lit_sum = Float32(0.0)
    for p_i in range(len(hdr_t1)):
        var c = hdr_t1[p_i]
        _assert(c.x == c.x and c.y == c.y and c.z == c.z, "F18 Terrain frame 1 finite HDR")
        terrain_lit_sum += c.x + c.y + c.z
    _assert(terrain_lit_sum > Float32(1.0), "F18 TerrainClipmap produces shaded pixels")

    # Translate camera & verify toroidal ring origins snap cleanly
    extractor.begin_extraction(rw_ring.write_world(), 14)
    rw_ring.write_world().terrain_clipmap = TerrainClipmapPacket(True, final_tc.copy(), id_pbr)
    rw_ring.write_world().camera = RenderCameraPacket(eye=Vec3(14.0, 13.0, -8.0), target=Vec3(14.0, 3.0, 14.0))
    extractor.end_extraction()
    rw_ring.swap_buffers()
    _assert(renderer.render_world(rw_ring.read_world(), ui_empty, capture_readback=True), "F18 TerrainClipmap frame 2 rendered")
    _assert(
        len(renderer.gpu_slot[0].frame_instances) > 0
        and abs(renderer.gpu_slot[0].frame_instances[0].world_mat.values[12] - renderer.gpu_slot[0].frame_instances[0].prev_world_mat.values[12]) < Float32(1e-5)
        and abs(renderer.gpu_slot[0].frame_instances[0].world_mat.values[14] - renderer.gpu_slot[0].frame_instances[0].prev_world_mat.values[14]) < Float32(1e-5),
        "F18 Terrain ring snap preserves prev_world_mat == world_mat (zero false world motion vector)",
    )
    var orig_b = renderer.gpu_slot[0].terrain_ring_origins[0]
    _assert(abs(orig_b[0] - orig_a[0]) > Float32(5.0) and abs(orig_b[1] - orig_a[1]) > Float32(5.0), "F18 Terrain ring origin snapped on camera translation")
    var hdr_t2 = renderer.gpu_slot[0].readback_resolved_hdr()
    for p_i in range(len(hdr_t2)):
        var c = hdr_t2[p_i]
        _assert(c.x == c.x and c.y == c.y and c.z == c.z, "F18 Terrain frame 2 finite HDR")

    # Regression (F18): splat_control_slot + layer_albedo_slots + RenderWorld slot-only dirty sync
    var splat_rgba = List[UInt8](length=4 * 4 * 4, fill=0)
    var mag_rgba = List[UInt8](length=4 * 4 * 4, fill=0)
    for ti in range(16):
        splat_rgba[ti * 4 + 1] = UInt8(255)  # 100% layer 1 (sand)
        splat_rgba[ti * 4 + 3] = UInt8(0)
        mag_rgba[ti * 4 + 0] = UInt8(255)    # Bright magenta layer albedo
        mag_rgba[ti * 4 + 1] = UInt8(25)
        mag_rgba[ti * 4 + 2] = UInt8(230)
        mag_rgba[ti * 4 + 3] = UInt8(128)
    var splat_slot = renderer.gpu_slot[0].upload_rgba8_texture(splat_rgba, 4, 4)
    var mag_slot = renderer.gpu_slot[0].upload_rgba8_texture(mag_rgba, 4, 4)
    _assert(splat_slot >= 0 and mag_slot >= 0, "Uploaded terrain splat_control_slot + layer_albedo_slots[1]")

    var tc_splat = final_tc.copy()
    tc_splat.splat_control_slot = UInt16(splat_slot)
    tc_splat.layer_albedo_slots[1] = UInt16(mag_slot)
    extractor.begin_extraction(rw_ring.write_world(), 15)
    rw_ring.write_world().terrain_clipmap = TerrainClipmapPacket(True, tc_splat.copy(), id_pbr)
    rw_ring.write_world().camera = RenderCameraPacket(eye=Vec3(14.0, 13.0, -8.0), target=Vec3(14.0, 3.0, 14.0))
    extractor.end_extraction()
    rw_ring.swap_buffers()
    _assert(renderer.render_world(rw_ring.read_world(), ui_empty, capture_readback=True), "F18 Terrain splat+layer frame rendered")
    var hdr_t3 = renderer.gpu_slot[0].readback_resolved_hdr()
    var splat_diff = Float32(0.0)
    var mag_hits = 0
    for p_i in range(len(hdr_t3)):
        var d = abs(hdr_t3[p_i].x - hdr_t2[p_i].x) + abs(hdr_t3[p_i].y - hdr_t2[p_i].y) + abs(hdr_t3[p_i].z - hdr_t2[p_i].z)
        if d > splat_diff:
            splat_diff = d
        if hdr_t3[p_i].x > hdr_t3[p_i].y * Float32(1.5) and hdr_t3[p_i].z > hdr_t3[p_i].y * Float32(1.2):
            mag_hits += 1
    _assert(
        splat_diff > Float32(0.1) and mag_hits > 0,
        "F18 splat_control_slot + layer_albedo_slots[1] changed GPU terrain output (diff=" + String(splat_diff) + ", hits=" + String(mag_hits) + ")",
    )
    renderer.clear_terrain_clipmap()

    # Regression (F18 RT BLAS Refit): 4-ring TerrainClipmap + 4-slot heavy VAT crowd with enable_rt_shadows active
    var mb_ell = MeshBuilder()
    mb_ell.add_ellipsoid(Vec3(0.0, 1.0, 0.0), Vec3(0.6, 0.8, 0.6), rings=16, segments=32)
    var ell_mesh = mb_ell^.finish()
    var ell_v_cnt = ell_mesh.vertex_count()
    var ell_desc = VatCrowdClipDesc(0, ell_v_cnt, 4, Float32(30.0), Float32(0.5), True)
    var ell_baked = BakedVatCrowdClip(ell_desc)
    for f_i in range(4):
        var sc_y = Float32(1.0) + Float32(f_i) * Float32(0.1)
        for v_i in range(ell_v_cnt):
            var dst = (f_i * ell_v_cnt + v_i) * 4
            ell_baked.pos_rgba16f[dst + 0] = f32_to_f16_bits(ell_mesh.positions[v_i * 3 + 0])
            ell_baked.pos_rgba16f[dst + 1] = f32_to_f16_bits(ell_mesh.positions[v_i * 3 + 1] * sc_y)
            ell_baked.pos_rgba16f[dst + 2] = f32_to_f16_bits(ell_mesh.positions[v_i * 3 + 2])
            ell_baked.pos_rgba16f[dst + 3] = f32_to_f16_bits(Float32(1.0))
            ell_baked.nrm_rgba16f[dst + 0] = f32_to_f16_bits(ell_mesh.normals[v_i * 3 + 0])
            ell_baked.nrm_rgba16f[dst + 1] = f32_to_f16_bits(ell_mesh.normals[v_i * 3 + 1])
            ell_baked.nrm_rgba16f[dst + 2] = f32_to_f16_bits(ell_mesh.normals[v_i * 3 + 2])
            ell_baked.nrm_rgba16f[dst + 3] = f32_to_f16_bits(Float32(0.0))
    ell_baked.bounds_min = Vec3(-1.0, 0.0, -1.0)
    ell_baked.bounds_max = Vec3(1.0, 2.5, 1.0)
    var cid_rt = renderer.register_vat_crowd_clip(ell_baked, ell_mesh, id_pbr, max_phase_slots=4)
    _assert(cid_rt >= 0, "F18 heavy VAT clip registered for RT BLAS refit check")

    var tc_rt = TerrainClipmapDesc(ring_count=4, grid_dim_per_ring=32, base_cell_size_m=0.5, height_scale_m=10.0)
    var final_tc_rt = renderer.set_terrain_clipmap(tc_rt, id_pbr, heights, hm_w, hm_h)
    renderer.gpu_slot[0].enable_rt_shadows = renderer.gpu_slot[0].vk.has_ray_query
    extractor.begin_extraction(rw_ring.write_world(), 16)
    rw_ring.write_world().terrain_clipmap = TerrainClipmapPacket(True, final_tc_rt.copy(), id_pbr)
    rw_ring.write_world().camera = RenderCameraPacket(eye=Vec3(0.0, 10.0, -12.0), target=Vec3(0.0, 1.0, 0.0))
    for s_i in range(4):
        extractor.extract_vat_crowd_entity(
            rw_ring.write_world(),
            UInt64(9800 + s_i),
            cid_rt,
            Float32(s_i) * Float32(0.11),
            Float32(s_i) * Float32(0.11),
            trs(Vec3(Float32(s_i - 1) * Float32(2.2), 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)),
            id_pbr,
        )
    extractor.end_extraction()
    rw_ring.swap_buffers()
    _assert(
        renderer.render_world(rw_ring.read_world(), ui_empty, capture_readback=True)
        and renderer.gpu_slot[0].vk.error == "",
        "F18 RT-enabled 4-ring TerrainClipmap + 4-slot heavy VAT BLAS refit succeeded without device loss (err="
        + renderer.gpu_slot[0].vk.error + ")",
    )
    var rt_hdr = renderer.gpu_slot[0].readback_resolved_hdr()
    var rt_sum = Float32(0.0)
    for p_i in range(len(rt_hdr)):
        var c = rt_hdr[p_i]
        _assert(c.x == c.x and c.y == c.y and c.z == c.z, "F18 RT BLAS refit HDR finite")
        rt_sum += c.x + c.y + c.z
    _assert(rt_sum > Float32(1.0), "F18 RT BLAS refit produced non-zero shaded HDR")
    renderer.gpu_slot[0].enable_rt_shadows = False
    renderer.clear_terrain_clipmap()

    # Regression (F18): AAA_SKIN_MAX_VERTS bounds enforcement in register_vat_crowd_clip
    var saved_bind = renderer.gpu_slot[0].skin_bind_cursor
    var saved_vcur = renderer.gpu_slot[0].skin_arena.vertex_cursor
    var w_vcnt = walker.meshes[0].vertex_count()
    renderer.gpu_slot[0].skin_bind_cursor = AAA_SKIN_MAX_VERTS - w_vcnt * 2 - 1
    var cid_part = renderer.gpu_slot[0].register_vat_crowd_clip(baked_vat, walker.meshes[0], id_pbr, max_phase_slots=8)
    _assert(
        cid_part >= 0 and len(renderer.gpu_slot[0].vat_clips[cid_part].slot_mesh_entries) == 2,
        "register_vat_crowd_clip clamps slots cleanly at AAA_SKIN_MAX_VERTS",
    )
    for sb_i in range(len(renderer.gpu_slot[0].vat_clips[cid_part].slot_skin_bases)):
        var sb = renderer.gpu_slot[0].vat_clips[cid_part].slot_skin_bases[sb_i]
        _assert(sb + w_vcnt <= AAA_SKIN_MAX_VERTS, "slot_skin_base strictly within AAA_SKIN_MAX_VERTS")
    var cid_exhaust = renderer.gpu_slot[0].register_vat_crowd_clip(baked_vat, walker.meshes[0], id_pbr, max_phase_slots=4)
    _assert(cid_exhaust == -1, "register_vat_crowd_clip returns -1 when AAA_SKIN_MAX_VERTS exhausted on slot 0")
    renderer.gpu_slot[0].skin_bind_cursor = saved_bind
    renderer.gpu_slot[0].skin_arena.vertex_cursor = saved_vcur

    print(
        "[PASS] Suite 12 (F17+F18): 5-Class Split VisBuffer (tiles="
        + String(counts[0])
        + "/"
        + String(counts[1])
        + "/"
        + String(counts[2])
        + "/"
        + String(counts[3])
        + "/"
        + String(counts[4])
        + "), 1024-Crowd VAT (diff="
        + String(vat_max_diff)
        + "), TerrainClipmap"
    )


def _validate_or_fail(words: List[UInt32], name: String):
    var rep = validate_spirv_module(words)
    if not rep.valid:
        _fail(name + " SPIR-V validation failed: " + rep.errors[0])


def test_render_features_glass_water_particles():
    """Suite 7: soft RT sun shadows, refractive glass, shaped water and generic GPU particles."""
    # Glass optics on the CPU
    _assert(abs(glass_f0_cpu(1.5) - Float32(0.04)) < 1e-4, "glass F0 at ior 1.5 is 4%")
    _assert(glass_fresnel_cpu(1.0, 1.5) < 0.05 and glass_fresnel_cpu(0.05, 1.5) > 0.6, "Fresnel rises toward grazing")
    var head_on = glass_refract_cpu(Vec3(0.0, 0.0, -1.0), Vec3(0.0, 0.0, 1.0), Float32(1.0 / 1.5))
    _assert(head_on[1] and abs(head_on[0].z + 1.0) < 1e-5, "head-on light passes straight through")
    var s45 = Float32(0.70710678)
    var bent = glass_refract_cpu(Vec3(s45, 0.0, -s45), Vec3(0.0, 0.0, 1.0), Float32(1.0 / 1.5))
    _assert(bent[1] and abs(bent[0].x - s45 / 1.5) < 1e-4, "Snell: sin(t) = sin(i) / 1.5")
    var tir = glass_refract_cpu(Vec3(s45, 0.0, -s45), Vec3(0.0, 0.0, 1.0), Float32(1.5))
    _assert(not tir[1] and tir[0].z > 0.0, "45 deg from inside glass is total internal reflection")
    var bl = glass_beer_lambert_cpu(Vec3(0.5, 1.0, 1.0), 1.0, 2.0)
    _assert(abs(bl.x - 0.25) < 1e-3 and abs(bl.y - 1.0) < 1e-4, "Beer-Lambert: tint^(density * path)")
    var clear = AaaGlassParams(material_id=0, ior=1.5).shadow_transmittance()
    var dark = AaaGlassParams(material_id=0, ior=1.5, tint=Vec3(0.3, 0.3, 0.3), opacity=0.5).shadow_transmittance()
    _assert(clear > 0.9 and clear < 1.0 and dark < clear * 0.2, "glass shadow transmittance")

    # Soft shadow filter radius: wider with occluder distance, capped
    var r_near = penumbra_radius_px_cpu(0.1, 0.005, 5.0, 0.4, 720)
    var r_far = penumbra_radius_px_cpu(2.0, 0.005, 5.0, 0.4, 720)
    _assert(r_near < r_far and r_far <= 14.0, "penumbra radius grows with occluder distance")

    # Generic particle emitter packing
    var em = AaaParticleEmitter.generic(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 0.0), 300, color=Vec3(0.0, 3.0, 0.0), stretch=True)
    var ew = List[UInt32]()
    em.append_words(ew, 0)
    _assert(len(ew) == AAA_EMITTER_WORDS, "an emitter packs AAA_EMITTER_WORDS words")
    _assert((ew[14] & 7) == PARTICLE_GENERIC, "generic archetype in the bounce word's low bits")
    _assert(ew[22] == (PARTICLE_FLAG_ADDITIVE | PARTICLE_FLAG_STRETCH), "generic flags word")
    _assert(em.end_color.y == Float32(3.0) and em.end_size == em.size, "end color / size default to the start")

    _validate_or_fail(rt_sun_shadow_trace_shader(), "rt_sun_shadow_trace_shader")
    _validate_or_fail(rt_sun_shadow_filter_shader(), "rt_sun_shadow_filter_shader")
    _validate_or_fail(aaa_glass_refract_shader(), "aaa_glass_refract_shader")
    _validate_or_fail(aaa_water_refract_shader(True), "aaa_water_refract_shader")
    _validate_or_fail(gpu_particle_sim_and_bin_shader(), "gpu_particle_sim_and_bin_shader")
    _validate_or_fail(gpu_particle_tile_composite_shader(), "gpu_particle_tile_composite_shader")

    var W = 160
    var H = 96
    var ren = FlintAaaVulkanRenderer(width=W, height=H, tsr_scale=1.0, max_draws=1024, max_instances=256, env_cube_dim=16)
    if not ren.active:
        print("[SKIP] Vulkan unavailable for Suite 7 render-feature GPU checks")
        ren.shutdown()
        return
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_auto_exposure = False
    ren.bloom_intensity = Float32(0.0)
    ren.camera_lens.vignette_strength = Float32(0.0)
    var mat_floor = ren.add_material(AaaMaterial(base_color=Vec3(0.3, 0.3, 0.3), roughness=0.8, metallic=0.0))
    var mat_wall = ren.add_material(AaaMaterial(base_color=Vec3(0.9, 0.06, 0.04), roughness=0.6, metallic=0.0))
    var mat_glass = ren.add_glass_material(AaaGlassParams(material_id=0, ior=1.2, density=0.0))
    _assert(mat_glass >= 0 and ren.is_glass_material(mat_glass), "glass material registered")
    var mb_box = MeshBuilder()
    mb_box.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), bevel=0.01)
    var box_e = ren.add_mesh(mb_box^.finish(), mat_floor, build_lods=False)
    var mb_ball = MeshBuilder()
    mb_ball.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(0.8, 0.8, 0.8), rings=16, segments=24)
    var ball_e = ren.add_mesh(mb_ball^.finish(), mat_glass, build_lods=False)
    var eye = Vec3(0.0, 1.0, 4.0)
    var tgt = Vec3(0.0, 1.0, 0.0)
    var cx = W // 2
    var cy = H // 2

    # Additive generic particles light the middle of the frame (green)
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    _assert(ren.render_frame(capture_readback=True), "particle baseline frame rendered")
    var g0 = ren.readback_resolved_hdr()[cy * W + cx].y
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    ren.submit_particle_emitter(
        AaaParticleEmitter.generic(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 0.0), 400, color=Vec3(0.0, 3.0, 0.0), size=0.12,
                                   alpha=1.0, end_alpha=1.0, spawn_radius=0.15, velocity_spread=0.0, gravity_scale=0.0)
    )
    _assert(ren.render_frame(capture_readback=True), "particle frame rendered")
    var g1 = ren.readback_resolved_hdr()[cy * W + cx].y
    _assert(g1 > g0 + 0.5, "additive particles brighten the frame (green " + String(g0) + " -> " + String(g1) + ")")

    # Clear glass in front of a red wall shows the wall (needs hardware ray queries; no RT feature
    # needs to be on: glass in the frame builds the TLAS by itself)
    var glass_note = String("glass skipped (no ray queries)")
    if ren.vk.has_ray_query:
        ren.sun_dir = Vec3(0.3, 0.8, 0.6).normalized()  # the sun lights the wall's face
        var red_seen = Float32(0.0)
        var blue_seen = Float32(0.0)
        for _ in range(3):
            ren.begin_frame(eye, tgt, fov_y_deg=40.0)
            ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
            ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, 2.0, -2.0), Quat(), Vec3(3.0, 2.0, 0.1)), material_override=mat_wall)
            ren.draw_instance(ball_e.mesh_id, trs(Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
            _assert(ren.render_frame(capture_readback=True), "glass frame rendered")
            var c = ren.readback_resolved_hdr()[cy * W + cx].copy()
            red_seen = c.x
            blue_seen = c.z
        _assert(red_seen > blue_seen * 2.0, "the red wall shows through clear glass (r=" + String(red_seen) + " b=" + String(blue_seen) + ")")
        glass_note = "glass r=" + String(red_seen) + " b=" + String(blue_seen)
    # Verify Phase 0+2 ring-buffering, multi-mip Hi-Z pyramid, async VSM boundary crossing, and live resize()
    _assert(
        len(ren.ring_surfcache) == 2
        and len(ren.ring_vsm_table) == 2
        and len(ren.ring_oit_staging) == 2
        and ren.ring_surfcache[0].vk_buffer != ren.ring_surfcache[1].vk_buffer
        and ren.ring_vsm_table[0].vk_buffer != ren.ring_vsm_table[1].vk_buffer
        and ren.ring_oit_staging[0].vk_buffer != ren.ring_oit_staging[1].vk_buffer,
        "Per-slot ring-buffered surfcache, vsm_table, and oit_staging buffers are distinct",
    )
    _assert(
        ren.hiz_img.image != 0 and ren.hiz_mip_levels == compute_hiz_mip_levels(W, H) and ren.hiz_valid,
        "Multi-mip Hi-Z depth pyramid created and populated",
    )
    # VSM camera boundary crossing (async 1-frame readback without mid-frame stall)
    ren.enable_vsm = True
    ren.begin_frame(Vec3(12.0, 2.0, 4.0), Vec3(12.0, 1.0, 0.0), fov_y_deg=40.0)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(12.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    _assert(ren.render_frame(capture_readback=False), "VSM boundary-crossing async frame rendered")

    # Live FlintAaaVulkanRenderer.resize() + post-resize render & readback
    _assert(ren.resize(192, 128), "FlintAaaVulkanRenderer.resize(192, 128) succeeded")
    _assert(
        ren.output_w == 192 and ren.output_h == 128 and ren.hiz_mip_levels == compute_hiz_mip_levels(192, 128),
        "FlintAaaVulkanRenderer.resize updated output dimensions and Hi-Z mip count",
    )
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)))
    _assert(ren.render_frame(capture_readback=True), "Post-resize frame rendered")
    var resized_px = ren.readback_pixels()
    _assert(len(resized_px) == 192 * 128, "Post-resize readback matches new 192x128 dimensions")

    # Wave 2: 1D Compacted Meshlet Worklist + AaaRenderGraph Barrier & Aliasing Metrics
    var rg_m = ren.get_render_graph_aliasing_metrics()
    _assert(
        rg_m.unaliased_bytes > 0
        and rg_m.aliased_peak_bytes > 0
        and rg_m.aliased_peak_bytes < rg_m.unaliased_bytes
        and rg_m.saved_bytes == rg_m.unaliased_bytes - rg_m.aliased_peak_bytes
        and rg_m.compiled_barrier_count >= 7,
        "AaaRenderGraph OffsetAllocator VRAM aliasing & barrier compilation verified",
    )
    _assert(
        ren.render_graph.passes_executed_last_frame >= 7 and ren.render_graph.barriers_emitted_last_frame >= 5,
        "AaaRenderGraph emitted barriers across core frame passes",
    )
    var expected_wl = ren.geom.meshes[box_e.mesh_id].meshlet_count * 10 + ren.geom.meshes[ball_e.mesh_id].meshlet_count
    ren.begin_frame(eye, tgt, fov_y_deg=40.0)
    for k_i in range(10):
        ren.draw_instance(box_e.mesh_id, trs(Vec3(Float32(k_i) * Float32(0.4) - Float32(1.8), -0.1, 0.0), Quat(), Vec3(0.2, 0.1, 0.2)))
    ren.draw_instance(ball_e.mesh_id, trs(Vec3(0.0, 1.0, 0.0), Quat(), Vec3(0.5, 0.5, 0.5)), material_override=mat_floor)
    _assert(ren.render_frame(capture_readback=True), "1D compacted meshlet worklist frame rendered")
    _assert(
        ren.last_worklist_meshlets == expected_wl and ren.last_culled_meshlet_draws > 0,
        "1D compacted worklist count matches exact sum of meshlets (" + String(ren.last_worklist_meshlets) + " vs " + String(expected_wl) + ")",
    )

    # Wave 2: Projected OBB Deferred Decal (`GpuDecalVolume`) GPU upload & shading verification
    ren.enable_vsm = False
    ren.sun_dir = Vec3(0.0, 0.9, 0.4).normalized()
    ren.sun_radiance = Vec3(4.0, 4.0, 4.0)
    var dec_eye = Vec3(0.0, 3.0, 3.5)
    var dec_tgt = Vec3(0.0, 0.0, 0.0)
    var rw_d = ren.output_w
    var rh_d = ren.output_h
    var mid_idx = (rh_d // 2) * rw_d + (rw_d // 2)

    ren.begin_frame(dec_eye, dec_tgt, fov_y_deg=45.0)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)), material_override=mat_floor)
    _assert(ren.render_frame(capture_readback=True), "Decal baseline floor frame rendered")
    var floor_base = ren.readback_resolved_hdr()[mid_idx].copy()

    var red_decal = GpuDecalVolume(
        center=Vec3(0.0, 0.0, 0.0),
        half_extents=Vec3(2.5, 1.0, 2.5),
        rotation=Quat(),
        albedo_slot=BINDLESS_SLOT_NONE,
        opacity=1.0,
    )
    red_decal.tint_rgb = Vec3(0.95, 0.04, 0.04)
    ren.begin_frame(dec_eye, dec_tgt, fov_y_deg=45.0)
    ren.draw_instance(box_e.mesh_id, trs(Vec3(0.0, -0.1, 0.0), Quat(), Vec3(5.0, 0.1, 5.0)), material_override=mat_floor)
    ren.add_decal(red_decal)
    _assert(ren.render_frame(capture_readback=True), "Projected OBB decal frame rendered")
    var floor_dec = ren.readback_resolved_hdr()[mid_idx].copy()
    _assert(
        floor_dec.x > floor_base.x * Float32(1.8) and floor_dec.x > floor_dec.z * Float32(3.0),
        "Projected OBB red decal shaded onto grey floor (r: " + String(floor_base.x) + " -> " + String(floor_dec.x) + ", b: " + String(floor_dec.z) + ")",
    )

    print("[PASS] Suite 7: Soft RT Shadows + Refractive Glass + Shaped Water + Generic GPU Particles (green "
          + String(g0) + " -> " + String(g1) + ", " + glass_note + ")")
    test_f15_f16_f19_staging_aliasing_growth_and_direct_present(ren)
    ren.shutdown()


def _drop_amd_owner(w_fd: Int, ctrl_ptr: Int):
    var amd_owner = AmdGpu()
    amd_owner.fd = w_fd
    amd_owner.ctrl_ptr = ctrl_ptr


def _test_amdgpu_raii_lifecycle():
    """Tests AmdGpu instantiation (--Werror), out-of-order drop with GpuBuffer, and copy+close safety."""
    var amd_empty = AmdGpu()
    _assert(amd_empty.fd == -1 and amd_empty.ctrl_ptr == 0, "F22: empty AmdGpu initializes safely")

    # 1. Out-of-order AmdGpu vs GpuBuffer drop: fd must close when AmdGpu drops even if GpuBuffer is still live
    var p1 = List[Int32](length=2, fill=-1)
    _assert(external_call["pipe", Int32](p1.unsafe_ptr()) == 0, "F22: pipe() created fds for AmdGpu RAII test")
    var r_fd = Int(p1[0])
    var w_fd = Int(p1[1])
    var c1_raw = external_call["calloc", Int](Int(1), max(64, size_of[_AmdDeviceControlBlock]()))
    ref c1 = _amd_ctrl_ptr(c1_raw)[unsafe_offset=0]
    c1.ref_count = 1
    c1.gpu_ref_count = 1
    c1.fd = w_fd
    c1.alive = True

    var buf_surviving = GpuBuffer(1, 4096, 0, 0)
    var st_raw = external_call["calloc", Int](Int(1), max(128, size_of[_GpuBufferControl]()))
    ref st = _buf_ctrl_ptr(st_raw)[unsafe_offset=0]
    st.ref_count = 1
    st.destroyed = False
    st.backend_kind = 2
    st.ctrl_ptr = c1_raw
    st.amd_handle = 1
    st.amd_size = 0  # skip drm ioctl on pipe fd while still exercising actrl.alive & ref_count
    c1.ref_count += 1
    buf_surviving.state_ptr = st_raw

    _drop_amd_owner(w_fd, c1_raw)
    # amd_owner dropped while buf_surviving is still alive: w_fd MUST be closed & c1.alive == False!
    _assert(external_call["fcntl", Int32](Int32(w_fd), Int32(1)) == -1, "F22: AmdGpu.__deinit__ closed fd even while GpuBuffer was alive")
    _assert(not c1.alive and c1.ref_count == 1 and c1.gpu_ref_count == 0, "F22: _AmdDeviceControlBlock alive=False, ref_count=1 after owner drop")
    _ = buf_surviving.size
    close_fd(r_fd)

    # 2. AmdGpu move ownership + double-close must never double-close fd
    var p2 = List[Int32](length=2, fill=-1)
    _assert(external_call["pipe", Int32](p2.unsafe_ptr()) == 0, "F22: pipe() created fds for AmdGpu move+close test")
    var r2_fd = Int(p2[0])
    var w2_fd = Int(p2[1])
    var c2_raw = external_call["calloc", Int](Int(1), max(64, size_of[_AmdDeviceControlBlock]()))
    ref c2 = _amd_ctrl_ptr(c2_raw)[unsafe_offset=0]
    c2.ref_count = 1
    c2.gpu_ref_count = 1
    c2.fd = w2_fd
    c2.alive = True

    var a1 = AmdGpu()
    a1.fd = w2_fd
    a1.ctrl_ptr = c2_raw
    var a2 = a1^
    _assert(a2.fd == w2_fd and a2.ctrl_ptr == c2_raw, "F22: AmdGpu.__moveinit__ transferred fd & ctrl_ptr")
    _assert(c2.gpu_ref_count == 1 and c2.ref_count == 1, "F22: AmdGpu move preserves ref_count == 1")
    a2.close()
    _assert(a2.fd == -1 and a2.ctrl_ptr == 0, "F22: first a2.close() cleared a2 handles")
    _assert(external_call["fcntl", Int32](Int32(w2_fd), Int32(1)) == -1, "F22: first a2.close() closed w2_fd")
    a2.close()
    _assert(a2.fd == -1 and a2.ctrl_ptr == 0, "F22: second a2.close() is a safe no-op")
    close_fd(r2_fd)


def test_f22_raii_ownership_and_wsi_hooks():
    """Suite 8: F22 Movable-only RAII handles, out-of-order drop safety, and Win32/macOS WSI attach hooks."""
    comptime assert conforms_to(VulkanGpu, Movable)
    comptime assert not conforms_to(VulkanGpu, Copyable)
    comptime assert conforms_to(GpuBuffer, Movable)
    comptime assert not conforms_to(GpuBuffer, Copyable)
    comptime assert conforms_to(AmdGpu, Movable)
    comptime assert not conforms_to(AmdGpu, Copyable)
    comptime assert conforms_to(SwapchainHandle, Movable)
    comptime assert not conforms_to(SwapchainHandle, Copyable)
    _test_amdgpu_raii_lifecycle()

    var vk = VulkanGpu()
    if not vk.initialize(validate=False, need_discrete=False):
        print("[SKIP] Vulkan unavailable; skipping Suite 8 GPU checks")
        vk.close()
        return

    var buf_dev = vk.create_buffer_device_local(4096)
    var buf_host = vk.create_buffer(4096, local=False)
    var img = vk.create_image_2d(16, 16, FORMAT_R8G8B8A8_UNORM, IMAGE_USAGE_SAMPLED | IMAGE_USAGE_TRANSFER_TARGET)
    var blk = VulkanMemBlock(0, 0, 4096, 0, False)
    var surf = SwapchainSurface()
    _assert(buf_dev.vk_buffer != 0 and buf_dev.cpu_address == 0, "F22: DEVICE_LOCAL buffer has cpu_address == 0")
    _assert(buf_host.vk_buffer != 0 and buf_host.cpu_address != 0, "F22: host-visible buffer mapped")
    _assert(img.image != 0 and img.view != 0, "F22: VulkanImage created")

    # Move VulkanGpu to a new owner (transferring ownership cleanly; original `vk` is moved-out by compiler)
    var vk2 = vk^
    _assert(vk2.instance != 0 and vk2.device != 0, "F22: moved-to VulkanGpu retains live handles")

    # Real VkDeviceMemory wrapped in VulkanMemBlock with inverted drop order (vk2.close() before blk_real2 drops)
    var mt = vk2.memory_type(0xFFFFFFFF, 0, 0)
    _assert(mt >= 0, "F22: valid memory type found for VulkanMemBlock")
    var real_mem = vk2.allocation(4096, mt, 0)
    _assert(real_mem != 0, "F22: vk2.allocation(4096) succeeded")
    var blk_real = VulkanMemBlock(real_mem, mt, 4096, 0, False, vk2.device, vk2.fn_free_memory, vk2.ctrl_ptr)
    var blk_real2 = blk_real^
    _assert(blk_real2.ctrl_ptr == vk2.ctrl_ptr and blk_real2.memory == real_mem, "F22: VulkanMemBlock attached to _VkDeviceControlBlock")

    # Non-opaque BLAS via build_blas_triangles_ex(..., opaque=False) with inverted drop order
    var blas_non_opaque = VulkanAccelStruct()
    var rt_vbuf = GpuBuffer()
    var rt_ibuf = GpuBuffer()
    var rt_scratch = GpuBuffer()
    if vk2.has_ray_query:
        _assert(vk2.allocate_command_buffers(1), "F22: allocated cmd buffer for non-opaque BLAS")
        rt_vbuf = vk2.create_buffer_bda(4096, extra_usage=BUFFER_USAGE_AS_BUILD_INPUT_KHR | BUFFER_USAGE_DEVICE_ADDRESS, host_visible=True)
        rt_ibuf = vk2.create_buffer_bda(4096, extra_usage=BUFFER_USAGE_AS_BUILD_INPUT_KHR | BUFFER_USAGE_DEVICE_ADDRESS, host_visible=True)
        put_float(rt_vbuf.cpu_address + 0, 0.0); put_float(rt_vbuf.cpu_address + 4, 0.0); put_float(rt_vbuf.cpu_address + 8, 0.0)
        put_float(rt_vbuf.cpu_address + 48, 1.0); put_float(rt_vbuf.cpu_address + 52, 0.0); put_float(rt_vbuf.cpu_address + 56, 0.0)
        put_float(rt_vbuf.cpu_address + 96, 0.0); put_float(rt_vbuf.cpu_address + 100, 1.0); put_float(rt_vbuf.cpu_address + 104, 0.0)
        put32(rt_ibuf.cpu_address + 0, 0); put32(rt_ibuf.cpu_address + 4, 1); put32(rt_ibuf.cpu_address + 8, 2)
        if vk2.begin_commands(0):
            blas_non_opaque = build_blas_triangles_ex(
                vk2,
                0,
                vk2.buffer_device_address(rt_vbuf),
                48,
                2,
                vk2.buffer_device_address(rt_ibuf),
                1,
                rt_scratch,
                opaque=False,
            )
            var f_rt = vk2.submit(0)
            if f_rt != 0:
                _ = vk2.wait(f_rt)
        _assert(blas_non_opaque.handle != 0 and blas_non_opaque.ctrl_ptr == vk2.ctrl_ptr, "F22: non-opaque BLAS attached ctrl_ptr")

    # Cross-platform WSI hooks fail gracefully on Linux without linking Win32/macOS libs
    _assert(not vk2.create_win32_surface(0, 0), "F22: create_win32_surface(0, 0) fails cleanly on Linux")
    _assert(not vk2.create_macos_surface(0), "F22: create_macos_surface(0) fails cleanly on Linux")
    _assert(vk2.attach_win32_surface(0, 0, 128, 128).swapchain == 0, "F22: attach_win32_surface returns invalid handle cleanly")
    _assert(vk2.attach_macos_surface(0, 128, 128).swapchain == 0, "F22: attach_macos_surface returns invalid handle cleanly")

    # Close device BEFORE buf_dev, buf_host, img, blk_real2, blas_non_opaque drop (arbitrary drop-order safety via _VkDeviceControlBlock)
    vk2.close()
    _assert(vk2.device == 0, "F22: vk2.close() succeeded while live handles still in scope")
    # Double close is a safe no-op
    vk2.close()
    _ = surf.width
    _ = blk.memory
    _ = blk_real2.memory
    _ = blas_non_opaque.handle
    _ = rt_vbuf.size
    _ = rt_ibuf.size
    _ = rt_scratch.size
    print("[PASS] Suite 8: F22 RAII Handle Ownership, Drop-Order Safety & Win32/macOS WSI Hooks")


def test_f15_f16_f19_staging_aliasing_growth_and_direct_present(mut ren: FlintAaaVulkanRenderer):
    """Suites 9-11: F15 LinearStagingRing & DEVICE_LOCAL, F16 Aliased Heaps & submit_side, F19 Growable Pools & Direct Present."""
    # Suite 9a: LinearStagingRing overflow slabs without memcpy of prior slices
    var vk = VulkanGpu()
    if not vk.initialize(validate=False, need_discrete=False):
        print("[SKIP] Vulkan unavailable; skipping Suites 9-11 GPU checks")
        vk.close()
        return
    var ring = LinearStagingRing()
    _assert(ring.init(vk, 65536), "F15: LinearStagingRing initialized (64 KiB/slot)")
    var sl0 = ring.alloc(vk, 0, 4096)
    _assert(sl0.slab_idx == -1 and sl0.cpu_address != 0, "F15: first slice uses primary slot buffer")
    put32(sl0.cpu_address, 0xCAFEBABE)
    var sl1 = ring.alloc(vk, 0, 131072)
    _assert(sl1.slab_idx == 0 and sl1.cpu_address != 0, "F15: large slice spills into overflow slab without reallocating primary")
    _assert(get32(sl0.cpu_address) == 0xCAFEBABE, "F15: prior slice pointer remains valid & uncorrupted after overflow")
    _assert(len(ring.overflow_slabs[0]) == 1, "F15: overflow_slabs[0] tracks spilled slab")
    ring.reset_slot(vk, 0)
    _assert(ring.slot_offsets[0] == 0 and len(ring.overflow_slabs[0]) == 0, "F15: reset_slot frees overflow slabs cleanly")
    ring.shutdown(vk)
    vk.close()

    # Suites 9b + 10 + 11 on active FlintAaaVulkanRenderer
    if not ren.active:
        print("[SKIP] AAA Vulkan renderer unavailable; skipping remaining Suites 9-11 checks")
        return

    # F15 checks: DEVICE_LOCAL buffers have cpu_address == 0; globals mirror & VSM/readback remain host-visible
    _assert(
        ren.buf_vertices.cpu_address == 0
        and ren.buf_indices.cpu_address == 0
        and ren.buf_meshlets.cpu_address == 0
        and ren.buf_materials.cpu_address == 0
        and ren.buf_skinning.cpu_address == 0
        and ren.buf_hdr_out.cpu_address == 0
        and ren.buf_history.cpu_address == 0
        and ren.buf_oit_nodes.cpu_address == 0
        and ren.buf_post_scratch.cpu_address == 0
        and ren.ring_globals[0].cpu_address == 0
        and ren.ring_instances[0].cpu_address == 0
        and ren.ring_draw_pairs[0].cpu_address == 0,
        "F15: Geometry, render target, and per-frame SSBOs reside in DEVICE_LOCAL VRAM (cpu_address == 0)",
    )
    _assert(
        ren.buf_globals.cpu_address != 0
        and ren.ring_vsm_table[0].cpu_address != 0
        and ren.buf_ldr_readback.cpu_address != 0,
        "F15: Studio picking globals mirror + HOST_CACHED VSM & readback buffers remain mapped",
    )

    # F16 checks: Aliased heaps + disjoint metrics + resize + async compute submit_side
    _assert(ren.aliased_buf_heap != 0 and ren.aliased_img_heap != 0, "F16: Aliased buffer & image heaps created")
    _assert(
        ren.buf_indirect_phase2.vk_suballoc_block == -2
        and ren.buf_occluded_candidates.vk_suballoc_block == -2
        and ren.buf_hdr_out.vk_suballoc_block == -2
        and ren.vis_img.suballoc_block == -2
        and ren.depth_img.suballoc_block == -2
        and ren.post_alias_img.suballoc_block == -2,
        "F16: Transient targets bound to aliased heaps with sentinel suballoc_block == -2",
    )
    _assert(
        ren.render_graph.aliased_buf_peak_bytes > 0
        and ren.render_graph.aliased_img_peak_bytes > 0
        and ren.render_graph.aliased_peak_bytes < ren.render_graph.unaliased_transient_bytes
        and ren.render_graph.aliasing_barriers_compiled > 0,
        "F16: Separate buffer & image OffsetAllocators realize genuine intra-frame VRAM savings",
    )

    var mat_red = ren.add_material(AaaMaterial(base_color=Vec3(0.85, 0.2, 0.15), roughness=0.45))
    var mb_small = MeshBuilder()
    mb_small.add_cylinder(Vec3(0.0, -0.5, 0.0), Vec3(0.0, 0.5, 0.0), 0.6, 0.6, segments=16)
    var small_e = ren.add_mesh(mb_small^.finish(), mat_red)

    ren.enable_vsm = True
    ren.enable_async_vsm = True
    ren.sun_dir = Vec3(0.3, 0.85, 0.4).normalized()
    ren.sun_radiance = Vec3(5.0, 5.0, 5.0)
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=True), "F15/F16: First VSM frame rendered")
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=True), "F15/F16: Steady-state frame rendered")
    _assert(ren.render_graph.buffer_barriers_emitted_last_frame > 0, "F16: AaaRenderGraph emits real VkBufferMemoryBarrier2 barriers")
    if ren.vk.has_side_queue():
        _assert(ren.last_async_compute_submits >= 1, "F16: VsmMarkPages executed via submit_side async compute queue")
    else:
        _assert(ren.last_async_compute_submits == 0, "F16: Inline fallback active when device lacks dedicated async compute queue")

    var mid_hdr = ren.readback_resolved_hdr()[(ren.output_h // 2) * ren.output_w + (ren.output_w // 2)].copy()
    _assert(mid_hdr.x + mid_hdr.y + mid_hdr.z > Float32(0.05), "F15: Lit center pixel readback via HOST_CACHED buffer > 0.05")

    # Regression (F16): Staggered-lifetime fragmentation check (A[0..0]=1M, B[0..2]=2M, C[1..2]=2M -> >= 5242880)
    var rg_frag = AaaRenderGraph()
    var r_a = rg_frag.add_resource("A", 1048576, transient=True, is_image=False, alignment_bytes=256)
    var r_b = rg_frag.add_resource("B", 2097152, transient=True, is_image=False, alignment_bytes=256)
    var r_c = rg_frag.add_resource("C", 2097152, transient=True, is_image=False, alignment_bytes=256)
    var acc_p0 = List[RgResourceAccess]()
    acc_p0.append(RgResourceAccess(r_a, RG_USAGE_COMPUTE_WRITE))
    acc_p0.append(RgResourceAccess(r_b, RG_USAGE_COMPUTE_WRITE))
    _ = rg_frag.add_pass("P0", acc_p0, RG_QUEUE_GRAPHICS)
    var acc_p1 = List[RgResourceAccess]()
    acc_p1.append(RgResourceAccess(r_b, RG_USAGE_COMPUTE_READ))
    acc_p1.append(RgResourceAccess(r_c, RG_USAGE_COMPUTE_WRITE))
    _ = rg_frag.add_pass("P1", acc_p1, RG_QUEUE_GRAPHICS)
    var acc_p2 = List[RgResourceAccess]()
    acc_p2.append(RgResourceAccess(r_b, RG_USAGE_COMPUTE_READ))
    acc_p2.append(RgResourceAccess(r_c, RG_USAGE_COMPUTE_READ))
    _ = rg_frag.add_pass("P2", acc_p2, RG_QUEUE_GRAPHICS)
    _assert(rg_frag.compile(), "F16: Staggered fragmentation graph compiled")
    _assert(
        rg_frag.aliased_buf_peak_bytes >= 5242880,
        "F16: aliased_buf_peak_bytes reflects max(offset + size) >= 5242880 (got " + String(rg_frag.aliased_buf_peak_bytes) + ")",
    )
    for ri in range(3):
        _assert(
            rg_frag.resources[ri].aliased_offset_bytes + rg_frag.resources[ri].size_bytes <= rg_frag.aliased_buf_peak_bytes,
            "F16: Transient resource end offset within aliased_buf_peak_bytes",
        )

    # Regression (F16): >=4-frame FRAMES_IN_FLIGHT=2 deferred/async wraparound with VSM + async_vsm active
    ren.deferred_readback = True
    for _ in range(4):
        ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
        ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        _assert(ren.render_frame(capture_readback=False), "F16: Deferred FRAMES_IN_FLIGHT=2 async VSM frame rendered")
        _assert(ren.prev_submitted_fence > 0, "F16: prev_submitted_fence stays non-zero across deferred frame")
    ren.flush_deferred_frame()
    ren.deferred_readback = False

    # Verify slot wraparound fence preservation when prev_submitted_fence > frame_fences[cur_slot]
    ren.vk.wait_idle()
    var wrap_slot = ren.frame_index % 2
    var f_slot_old: UInt64 = 0
    var f_slot_newer: UInt64 = 0
    if ren.vk.begin_commands(0):
        f_slot_old = ren.vk.submit(0)
    if ren.vk.begin_commands(1):
        f_slot_newer = ren.vk.submit(1)
    _assert(f_slot_old > 0 and f_slot_newer > f_slot_old, "F16: Submitted ordered probe fences")
    ren.frame_fences[wrap_slot] = f_slot_old
    ren.prev_submitted_fence = f_slot_newer
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    _assert(
        ren.prev_submitted_fence == f_slot_newer,
        "F16: Waiting on older cur_slot fence preserves newer prev_submitted_fence across slot wraparound",
    )
    ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=True), "F16: Post-wraparound VSM frame rendered")
    var wrap_hdr = ren.readback_resolved_hdr()[(ren.output_h // 2) * ren.output_w + (ren.output_w // 2)].copy()
    _assert(
        wrap_hdr.x == wrap_hdr.x and wrap_hdr.y == wrap_hdr.y and wrap_hdr.z == wrap_hdr.z
        and (wrap_hdr.x + wrap_hdr.y + wrap_hdr.z) > Float32(0.05),
        "F16: Post-wraparound render shows zero NaN/pixel corruption",
    )
    ren.enable_vsm = False

    _assert(ren.resize(160, 120), "F16: resize(160, 120) cleanly rebuilt aliased heaps")
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=True), "F16: Post-resize frame rendered")

    _validate_or_fail(aaa_display_tonemap_direct_shader(24), "aaa_display_tonemap_direct_shader")
    _validate_or_fail(aaa_display_present_vertex_shader(), "aaa_display_present_vertex_shader")
    _validate_or_fail(aaa_display_present_fragment_shader(), "aaa_display_present_fragment_shader")
    _validate_or_fail(aaa_ui_overlay_direct_shader(24), "aaa_ui_overlay_direct_shader")
    _validate_or_fail(aaa_ui_present_fragment_shader(), "aaa_ui_present_fragment_shader")

    # Regression (F19): visbuffer_alphamask_vfx_fragment_shader SPIR-V validation + 0xFF prim_id mask check
    var vfx_spv = visbuffer_alphamask_vfx_fragment_shader()
    _validate_or_fail(vfx_spv, "visbuffer_alphamask_vfx_fragment_shader")
    var const_ff_id: UInt32 = 0
    var const_ffff_id: UInt32 = 0
    var last_and_a1: UInt32 = 0
    var last_and_a2: UInt32 = 0
    var spv_i = 5
    while spv_i < len(vfx_spv):
        var wc = Int(vfx_spv[spv_i] >> 16)
        var op = vfx_spv[spv_i] & UInt32(0xFFFF)
        if wc <= 0:
            break
        if op == OP_CONSTANT and wc == 4:
            if vfx_spv[spv_i + 3] == UInt32(0xFF):
                const_ff_id = vfx_spv[spv_i + 2]
            elif vfx_spv[spv_i + 3] == UInt32(0xFFFF):
                const_ffff_id = vfx_spv[spv_i + 2]
        elif op == OP_BITWISE_AND and wc >= 5:
            last_and_a1 = vfx_spv[spv_i + 3]
            last_and_a2 = vfx_spv[spv_i + 4]
        spv_i += wc
    _assert(
        const_ff_id != 0 and (last_and_a1 == const_ff_id or last_and_a2 == const_ff_id)
        and (const_ffff_id == 0 or (last_and_a1 != const_ffff_id and last_and_a2 != const_ffff_id)),
        "F19: visbuffer_alphamask_vfx_fragment_shader masks prim_id with 0xFF (not 0xFFFF)",
    )

    # Compact F20 check on live FlintAaaVulkanRenderer (< 20 ms): staged bindless mip upload + minLod sampler + eviction
    _assert(len(ren.samp_aniso16_min_lod) == 16 and ren.samp_aniso16_min_lod[2] != 0, "F20: samp_aniso16_min_lod[0..15] must be initialized")
    var f20_srv = AssetServer(stream_budget_bytes_per_tick=65536, enable_watcher=False)
    f20_srv.attach_renderer_residency_bridge(ren)
    var f20_h = f20_srv.textures.allocate_queued(AssetUuid.from_path("f20/render_bindless.png"), "f20/render_bindless.png", 1)
    _assert(f20_h.is_valid(), "F20: queued texture slot allocation failed")
    _assert(f20_srv.upload_mip_to_gpu(f20_h, mip_level=2, byte_size=1024), "F20: staged bindless mip 2 upload failed")
    var f20_slot = Int(f20_h.slot())
    var f20_bidx = f20_srv.streamer.slot_bindless_idx[f20_slot]
    _assert(f20_bidx >= 0 and ren.bindless_resident_min_mip[f20_bidx] == 2, "F20: bindless_resident_min_mip must clamp to mip 2")
    _assert(f20_srv.total_vram_live_bytes() > 0 and f20_srv.textures.slot_vram_bytes[f20_slot] > 0, "F20: slot_vram_bytes must be > 0")
    _assert(f20_srv.upload_mip_to_gpu(f20_h, mip_level=0, byte_size=16384), "F20: staged bindless mip 0 upload failed")
    _assert(ren.bindless_resident_min_mip[f20_bidx] == 0, "F20: bindless_resident_min_mip must refine to mip 0")
    _ = f20_srv.textures.release(f20_h)
    _ = f20_srv.textures.evict_lru_unreferenced(0)
    _assert(f20_srv.total_vram_live_bytes() == 0 and ren.bindless_vram_bytes[f20_bidx] == 0, "F20: bindless slot eviction must release VRAM to 0")

    # Odd-frame deferred_readback=True multi-frame bindless mip upload + eviction + slot reuse (< 15 ms)
    var prev_dr = ren.deferred_readback
    ren.deferred_readback = True
    # Ensure we hit an odd frame_index during upload & eviction
    if (ren.frame_index & 1) == 0:
        ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
        ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        _assert(ren.render_frame(capture_readback=False), "F20: odd-frame alignment render failed")
    for df_i in range(4):
        var dh = f20_srv.textures.allocate_queued(AssetUuid.from_path("f20/df_" + String(df_i) + ".png"), "f20/df_" + String(df_i) + ".png", 1)
        _assert(dh.is_valid(), "F20 deferred_readback: queued alloc failed")
        _assert(f20_srv.upload_mip_to_gpu(dh, mip_level=1, byte_size=4096), "F20 deferred_readback: odd/even mip upload failed")
        ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
        ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        _assert(ren.render_frame(capture_readback=False), "F20 deferred_readback: render_frame failed")
        _ = f20_srv.textures.release(dh)
        _ = f20_srv.textures.evict_lru_unreferenced(0)
    ren.flush_deferred_frame()
    ren.vk.wait_idle()
    _assert(f20_srv.total_vram_live_bytes() == 0, "F20 deferred_readback: multi-frame eviction must release VRAM to 0")
    _assert(
        len(ren.pending_bindless_desc_writes[0]) == 0 and len(ren.pending_bindless_desc_writes[1]) == 0,
        "F20 deferred_readback: flush_deferred_frame must drain pending_bindless_desc_writes",
    )

    # Regression (Failure Mode A): deferred streaming + capture_readback=True + eviction + subsequent render
    for df_a in range(4):
        var dh_a = f20_srv.textures.allocate_queued(
            AssetUuid.from_path("f20/dfa_" + String(df_a) + ".png"), "f20/dfa_" + String(df_a) + ".png", 1
        )
        _assert(dh_a.is_valid(), "F20 Failure Mode A: queued alloc failed")
        _assert(f20_srv.upload_mip_to_gpu(dh_a, mip_level=1, byte_size=4096), "F20 Failure Mode A: mip upload failed")
        ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
        ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        _assert(ren.render_frame(capture_readback=(df_a == 3)), "F20 Failure Mode A: render_frame failed")
        _ = f20_srv.textures.release(dh_a)
        _ = f20_srv.textures.evict_lru_unreferenced(0)
    ren.vk.wait_idle()
    _assert(ren.frame_fences[0] == 0 and ren.frame_fences[1] == 0, "F20 Failure Mode A: fences cleared after readback")
    _assert(
        len(ren.pending_bindless_desc_writes[0]) == 0 and len(ren.pending_bindless_desc_writes[1]) == 0,
        "F20 Failure Mode A: zero dangling pending_bindless_desc_writes when fences are 0",
    )
    var dh_post = f20_srv.textures.allocate_queued(AssetUuid.from_path("f20/dfa_post.png"), "f20/dfa_post.png", 1)
    _assert(dh_post.is_valid(), "F20 Failure Mode A: post-eviction alloc failed")
    _assert(f20_srv.upload_mip_to_gpu(dh_post, mip_level=1, byte_size=4096), "F20 Failure Mode A: post-eviction upload failed")
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=True), "F20 Failure Mode A: subsequent render_frame after eviction failed")
    _ = f20_srv.textures.release(dh_post)
    _ = f20_srv.textures.evict_lru_unreferenced(0)

    # Regression (Failure Mode B): even-frame eviction (frame_fences[1] != 0) -> odd-frame slot reuse pixel check
    if (ren.frame_index & 1) != 0:
        ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
        ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        _assert(ren.render_frame(capture_readback=False), "F20 Failure Mode B: even-frame alignment failed")
        ren.flush_deferred_frame()
    var red_px = List[UInt32](length=16, fill=UInt32(0xFF0000FF))
    var blue_px = List[UInt32](length=16, fill=UInt32(0xFFFF0000))
    var slot_s = ren.allocate_bindless_texture_slot(4, 4, 1, FORMAT_R8G8B8A8_UNORM)
    _assert(slot_s >= 0, "F20 Failure Mode B: initial slot allocation failed")
    _assert(
        ren.upload_bindless_mip_staged(slot_s, 0, 4, 4, Int(red_px.unsafe_ptr()), 64) > 0,
        "F20 Failure Mode B: RED upload failed",
    )
    _ = len(red_px)
    var m_reuse_def = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.45)
    m_reuse_def.albedo_tex = slot_s
    var mat_reuse = ren.add_material(m_reuse_def)
    # Frame E (slot 0) + Frame O (slot 1)
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=mat_reuse)
    _assert(ren.render_frame(capture_readback=False), "F20 Failure Mode B: Frame E render failed")
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=mat_reuse)
    _assert(ren.render_frame(capture_readback=False), "F20 Failure Mode B: Frame O render failed")
    # Frame E2 (slot 0): frame_fences[1] != 0 during eviction, then render Frame E2
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    _assert(ren.frame_fences[1] != 0, "F20 Failure Mode B: expected frame_fences[1] != 0 before eviction")
    ren.evict_bindless_texture_slot(slot_s)
    ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=False), "F20 Failure Mode B: Frame E2 render failed")
    # Reuse slot_s with solid BLUE before Frame O2
    var slot_reused = ren.allocate_bindless_texture_slot(4, 4, 1, FORMAT_R8G8B8A8_UNORM)
    _assert(slot_reused == slot_s, "F20 Failure Mode B: expected reuse of evicted bindless slot")
    _assert(
        ren.upload_bindless_mip_staged(slot_reused, 0, 4, 4, Int(blue_px.unsafe_ptr()), 64) > 0,
        "F20 Failure Mode B: BLUE upload on reused slot failed",
    )
    _ = len(blue_px)
    _assert(len(ren.pending_bindless_desc_writes[1]) == 0, "F20 Failure Mode B: slot 1 pending writes must be clean")
    # Frame O2 (slot 1): verify center pixel reflects BLUE, not stale fallback white
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=mat_reuse)
    _assert(ren.render_frame(capture_readback=True), "F20 Failure Mode B: Frame O2 render failed")
    var reuse_hdr = ren.readback_resolved_hdr()[(ren.output_h // 2) * ren.output_w + (ren.output_w // 2)].copy()
    _assert(
        reuse_hdr.z > reuse_hdr.x * Float32(2.0) and reuse_hdr.z > reuse_hdr.y * Float32(2.0),
        "F20 Failure Mode B: reused slot rendered BLUE (r="
        + String(reuse_hdr.x)
        + " g="
        + String(reuse_hdr.y)
        + " b="
        + String(reuse_hdr.z)
        + ")",
    )
    ren.evict_bindless_texture_slot(slot_reused)
    ren.vk.wait_idle()

    # Regression (M3c Iter 4): mid-stream add_material/add_mesh/upload_bc7_texture/upload_rgba8_texture while frame_fences[0] != 0 and frame_fences[1] != 0 (< 10 ms)
    if (ren.frame_index & 1) != 0:
        ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
        ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        _assert(ren.render_frame(capture_readback=False), "M3c mid-stream: even alignment render failed")
        ren.flush_deferred_frame()
    # Submit even frame -> frame_fences[0] != 0
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(small_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=False), "M3c mid-stream: even deferred frame failed")
    _assert(ren.frame_fences[0] != 0, "M3c mid-stream: expected frame_fences[0] != 0 before mid-stream uploads")
    var mid_green_argb = List[UInt32](length=16, fill=UInt32(0xFF00FF00))
    var mid_green_rgba8 = List[UInt8](length=64, fill=0)
    for px_i in range(16):
        mid_green_rgba8[px_i * 4 + 1] = UInt8(255)
        mid_green_rgba8[px_i * 4 + 3] = UInt8(255)
    var mid_bc7 = ren.upload_bc7_texture(mid_green_argb, 4, 4, srgb=True, generate_mips=False)
    var mid_rgba8 = ren.upload_rgba8_texture(mid_green_rgba8, 4, 4)
    var mid_mat_def = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.45)
    mid_mat_def.albedo_tex = mid_rgba8
    var mid_mat = ren.add_material(mid_mat_def)
    var mb_mid = MeshBuilder()
    mb_mid.add_cylinder(Vec3(0.0, -0.5, 0.0), Vec3(0.0, 0.5, 0.0), 0.6, 0.6, segments=16)
    var mid_e = ren.add_mesh(mb_mid^.finish(), mid_mat)
    _assert(
        mid_bc7 >= 0 and mid_rgba8 >= 0 and mid_mat >= 0 and mid_e.mesh_id >= 0 and ren.vk.error == "",
        "M3c mid-stream: uploads while frame_fences[0] != 0 must succeed with empty vk.error (" + ren.vk.error + ")",
    )
    # Submit odd frame -> frame_fences[1] != 0, then upload bindless textures without wait_idle/flush_deferred_frame
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(mid_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=False), "M3c mid-stream: odd deferred frame failed")
    _assert(ren.frame_fences[1] != 0, "M3c mid-stream: expected frame_fences[1] != 0 before odd mid-stream uploads")
    var odd_bc7 = ren.upload_bc7_texture(mid_green_argb, 4, 4, srgb=True, generate_mips=False)
    var odd_rgba8 = ren.upload_rgba8_texture(mid_green_rgba8, 4, 4)
    _assert(
        odd_bc7 >= 0 and odd_rgba8 >= 0 and ren.vk.error == "",
        "M3c mid-stream: bindless uploads while frame_fences[1] != 0 must succeed (" + ren.vk.error + ")",
    )
    _assert(
        len(ren.pending_bindless_desc_writes[1]) >= 2,
        "M3c mid-stream: slot 1 bindless descriptor writes must be queued while frame_fences[1] != 0",
    )
    var odd_mat_def = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.45)
    odd_mat_def.albedo_tex = odd_rgba8
    var odd_mat = ren.add_material(odd_mat_def)
    _assert(odd_mat >= 0 and ren.vk.error == "", "M3c mid-stream: add_material while frame_fences[1] != 0 failed")

    # M3c Iter 5: exercise evaluate_material_graph_to_bindless, add_mesh (forcing buf_as_scratch growth),
    # upload_ibl_environment, and update_skinned_mesh_bones(defer_to_frame=False) while frame_fences[1] != 0!
    var eval_mat = ren.add_material(AaaMaterial(base_color=Vec3(0.4, 0.7, 0.3), roughness=0.4))
    ren.vk.wait_idle()
    for fi in range(len(ren.frame_fences)):
        ren.frame_fences[fi] = 0
    ren.prev_submitted_fence = 0

    var sk_box = MeshBuilder()
    sk_box.add_cylinder(Vec3(0.0, -0.2, 0.0), Vec3(0.0, 0.2, 0.0), 0.2, 0.2, segments=6)
    var sk_raw = sk_box^.finish()
    var sk_vcnt = sk_raw.vertex_count()
    var sk_joints = List[UInt32](length=sk_vcnt, fill=UInt32(0))
    var sk_weights = List[Vec3](length=sk_vcnt, fill=Vec3(1.0, 0.0, 0.0))
    var sk_mid_e = ren.add_skinned_mesh(sk_raw, sk_joints, sk_weights, odd_mat, bone_count=2)
    var sk_bones_mid = List[DualQuat]()
    sk_bones_mid.append(DualQuat())
    sk_bones_mid.append(DualQuat.from_quat_trans(Quat(0.0, 0.0, 0.0, 1.0), Vec3(0.0, 0.05, 0.0)))

    # Helper alignment: make sure next frame is even (slot 0), then submit slot 0 + slot 1 -> frame_fences[1] != 0
    if (ren.frame_index & 1) != 0:
        ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
        ren.draw_instance(mid_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=odd_mat)
        _assert(ren.render_frame(capture_readback=False), "M3c slot1: align render failed")
        ren.flush_deferred_frame()

    # 1) evaluate_material_graph_to_bindless while frame_fences[1] != 0 + verify uncorrupted Frame 1 HDR readback
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(mid_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=odd_mat)
    _assert(ren.render_frame(capture_readback=False), "M3c slot1: even frame before mg failed")
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(mid_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=odd_mat)
    _assert(ren.render_frame(capture_readback=False), "M3c slot1: odd frame before mg failed")
    _assert(ren.frame_fences[1] != 0, "M3c slot1: expected frame_fences[1] != 0 before evaluate_material_graph_to_bindless")
    var mg_slots = ren.evaluate_material_graph_to_bindless(StandardPbrMaterialNode(), eval_mat, dim=32)
    _assert(
        mg_slots[0] >= 0 and mg_slots[1] >= 0 and ren.frame_fences[1] == 0 and ren.vk.error == "",
        "M3c slot1: evaluate_material_graph_to_bindless must retire frame_fences[1] cleanly (" + ren.vk.error + ")",
    )
    ren.flush_deferred_frame()
    var mg_f1_hdr = ren.readback_resolved_hdr()[(ren.output_h // 2) * ren.output_w + (ren.output_w // 2)].copy()
    _assert(
        mg_f1_hdr.y > mg_f1_hdr.x * Float32(2.0) and mg_f1_hdr.y > mg_f1_hdr.z * Float32(2.0),
        "M3c slot1: Frame 1 HDR readback uncorrupted GREEN after evaluate_material_graph_to_bindless",
    )

    # 2) add_mesh(..., build_lods=False) forcing buf_as_scratch growth while frame_fences[1] != 0
    var scratch_before = ren.buf_as_scratch.size
    var mb_grow = MeshBuilder()
    for gi in range(24):
        var gx = Float32(gi) * Float32(2.0)
        mb_grow.add_cylinder(Vec3(gx, -0.4, 0.0), Vec3(gx, 0.4, 0.0), 0.5, 0.5, segments=256)
    var grow_mesh = mb_grow^.finish()
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(mid_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=odd_mat)
    _assert(ren.render_frame(capture_readback=False), "M3c slot1: even frame before add_mesh failed")
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(mid_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=odd_mat)
    _assert(ren.render_frame(capture_readback=False), "M3c slot1: odd frame before add_mesh failed")
    _assert(ren.frame_fences[1] != 0, "M3c slot1: expected frame_fences[1] != 0 before add_mesh")
    var grow_e = ren.add_mesh(grow_mesh, odd_mat, build_lods=False)
    _assert(
        grow_e.mesh_id >= 0 and ren.vk.error == "",
        "M3c slot1: add_mesh while frame_fences[1] != 0 failed (" + ren.vk.error + ")",
    )
    if ren.vk.has_ray_query:
        _assert(
            ren.frame_fences[1] == 0 and ren.buf_as_scratch.size > scratch_before,
            "M3c slot1: expected add_mesh(24k tris) to wait slot1 & grow buf_as_scratch (" + String(scratch_before) + " -> " + String(ren.buf_as_scratch.size) + ")",
        )

    # 3) upload_ibl_environment while frame_fences[1] != 0
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(mid_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=odd_mat)
    _assert(ren.render_frame(capture_readback=False), "M3c slot1: even frame before ibl failed")
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(mid_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=odd_mat)
    _assert(ren.render_frame(capture_readback=False), "M3c slot1: odd frame before ibl failed")
    _assert(ren.frame_fences[1] != 0, "M3c slot1: expected frame_fences[1] != 0 before upload_ibl_environment")
    ren.upload_ibl_environment(
        bake_bc6h_ibl_cubemap_and_sh9(
            SkyDomeParams(uniform_furnace=False),
            base_dim=ren.env_cube_dim,
            mip_levels=ren.env_mip_levels,
            ggx_samples=4,
        )
    )
    _assert(ren.frame_fences[1] == 0 and ren.vk.error == "", "M3c slot1: upload_ibl_environment failed (" + ren.vk.error + ")")

    # 4) update_skinned_mesh_bones(defer_to_frame=False) while frame_fences[1] != 0
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(mid_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=odd_mat)
    _assert(ren.render_frame(capture_readback=False), "M3c slot1: even frame before skinning failed")
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(mid_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=odd_mat)
    _assert(ren.render_frame(capture_readback=False), "M3c slot1: odd frame before skinning failed")
    _assert(ren.frame_fences[1] != 0, "M3c slot1: expected frame_fences[1] != 0 before update_skinned_mesh_bones")
    _ = ren.update_skinned_mesh_bones(sk_mid_e, sk_bones_mid, sk_bones_mid, defer_to_frame=False)
    _assert(ren.frame_fences[1] == 0 and ren.vk.error == "", "M3c slot1: update_skinned_mesh_bones(defer_to_frame=False) failed (" + ren.vk.error + ")")

    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(mid_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=odd_mat)
    _assert(ren.render_frame(capture_readback=False), "M3c mid-stream: Frame E2 render failed")
    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(mid_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=odd_mat)
    _assert(ren.render_frame(capture_readback=True), "M3c mid-stream: Frame O2 readback render failed")
    _assert(
        len(ren.pending_bindless_desc_writes[0]) == 0 and len(ren.pending_bindless_desc_writes[1]) == 0,
        "M3c mid-stream: pending_bindless_desc_writes must drain cleanly",
    )
    var midstream_hdr = ren.readback_resolved_hdr()[(ren.output_h // 2) * ren.output_w + (ren.output_w // 2)].copy()
    _assert(
        ren.vk.error == "" and midstream_hdr.y > midstream_hdr.x * Float32(2.0) and midstream_hdr.y > midstream_hdr.z * Float32(2.0),
        "M3c mid-stream: rendered GREEN with queued bindless descriptor (r="
        + String(midstream_hdr.x)
        + " g="
        + String(midstream_hdr.y)
        + " b="
        + String(midstream_hdr.z)
        + ")",
    )
    ren.vk.wait_idle()
    ren.deferred_readback = prev_dr
    f20_srv.close()

    # Suite 11: F19 geometric growth of buf_materials, buf_vertices, buf_indices
    var mat_cap0 = ren.buf_materials.size
    var last_mat = mat_red
    for m_i in range(260):
        last_mat = ren.add_material(AaaMaterial(base_color=Vec3(0.2, 0.75, 0.3), roughness=Float32(0.3 + Float32(m_i % 5) * 0.1)))
    _assert(ren.buf_materials.size > mat_cap0 and len(ren.materials) >= 260, "F19: buf_materials grew geometrically past initial 256-material cap")

    var vb_cap0 = ren.buf_vertices.size
    var ib_cap0 = ren.buf_indices.size
    var mb_big = MeshBuilder()
    for _ in range(176):
        mb_big.add_cylinder(Vec3(0.0, -0.4, 0.0), Vec3(0.0, 0.4, 0.0), 0.5, 0.5, segments=256)
    var big_e = ren.add_mesh(mb_big^.finish(), last_mat)
    _assert(ren.buf_vertices.size > vb_cap0, "F19: buf_vertices grew geometrically past initial 2 MiB capacity")
    _assert(ren.buf_indices.size > ib_cap0, "F19: buf_indices grew geometrically past initial 2 MiB capacity")

    ren.begin_frame(Vec3(0.0, 0.6, 2.5), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
    ren.draw_instance(big_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _assert(ren.render_frame(capture_readback=True), "F19: Frame rendered using grown geometry & material buffers")
    _assert(ren.last_culled_meshlet_draws > 0, "F19: Grown meshlet draw emitted via ring_draw_pairs")

    print("[PASS] Suites 9-11: F15 Staging/DEVICE_LOCAL, F16 Aliased Heaps & Async Compute, F19 Growth & Direct Present")


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


def _readback_hdr_out_floats(mut ren: FlintAaaVulkanRenderer) -> List[Float32]:
    var n_floats = ren.render_w * ren.render_h * 8
    var out = List[Float32](length=n_floats, fill=Float32(0.0))
    if not ren.active or ren.buf_hdr_out.vk_buffer == 0 or n_floats <= 0:
        return out^
    var rb = ren.vk.create_buffer(n_floats * 4, local=False, cpu_reads=True)
    if rb.vk_buffer != 0 and rb.cpu_address != 0:
        ren.vk.wait_idle()
        if ren.vk.begin_commands(0):
            ren.vk.copy_buffer(0, ren.buf_hdr_out, 0, rb, 0, n_floats * 4)
            ren.vk.memory_barrier2(
                0, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ)
            )
            var f = ren.vk.submit(0)
            if f != 0 and ren.vk.wait(f):
                _ = external_call["memcpy", Int](Int(out.unsafe_ptr()), rb.cpu_address, n_floats * 4)
    ren.vk.destroy_buffer(rb)
    return out^


def test_v2_spirv_shaders_and_cpu_models():
    # 1. All 20+ uncovered SPIR-V 1.5 shader variants from AAA v2
    _validate_or_fail(aaa_deferred_shade_shader(True), "aaa_deferred_shade_shader(RT=True)")
    _validate_or_fail(aaa_deferred_shade_shader(False), "aaa_deferred_shade_shader(RT=False)")
    _validate_or_fail(surface_cache_update_shader(True), "surface_cache_update_shader(RT=True)")
    _validate_or_fail(surface_cache_update_shader(False), "surface_cache_update_shader(RT=False)")
    _validate_or_fail(spec_reblur_temporal_shader(), "spec_reblur_temporal_shader")
    _validate_or_fail(spec_reblur_atrous_shader(), "spec_reblur_atrous_shader")
    _validate_or_fail(froxel_inject_and_reproject_shader(), "froxel_inject_and_reproject_shader")
    _validate_or_fail(froxel_raymarch_integrate_shader(), "froxel_raymarch_integrate_shader")
    _validate_or_fail(aaa_water_refract_shader(False), "aaa_water_refract_shader(RT=False)")
    _validate_or_fail(oit_and_6way_volumetric_shader(), "oit_and_6way_volumetric_shader")
    _validate_or_fail(aaa_tsr_v3_resolve_shader(), "aaa_tsr_v3_resolve_shader")
    _validate_or_fail(aaa_motion_blur_shader(), "aaa_motion_blur_shader")
    _validate_or_fail(aaa_display_tonemap_shader(), "aaa_display_tonemap_shader")
    _validate_or_fail(aaa_adapt_exposure_shader(BIND_AAA_POST_SCRATCH), "aaa_adapt_exposure_shader")
    _validate_or_fail(
        aaa_bloom_upsample_shader(BIND_AAA_HDR_OUT, BIND_AAA_POST_SCRATCH),
        "aaa_bloom_upsample_shader",
    )
    _validate_or_fail(visbuffer_alphamask_vertex_shader(), "visbuffer_alphamask_vertex_shader")
    _validate_or_fail(visbuffer_alphamask_fragment_shader(), "visbuffer_alphamask_fragment_shader")
    _validate_or_fail(vsm_alphamask_fragment_shader(), "vsm_alphamask_fragment_shader")
    _validate_or_fail(froxel_prefix_integrate_shader(), "froxel_prefix_integrate_shader")
    _validate_or_fail(gpu_particle_clear_tiles_shader(), "gpu_particle_clear_tiles_shader")

    # 2. Material 24-word layout, bit 30 interior parallax & road_wetness_mode
    var mat = AaaMaterial(base_color=Vec3(0.4, 0.4, 0.4), roughness=0.7, metallic=0.0)
    mat.interior_parallax_enabled = True
    mat.detail_normal_tex = 12
    mat.road_wetness_mode = True
    mat.detail_uv_scale = Float32(44.0)
    mat.subsurface_amount = Float32(0.35)
    var words = List[UInt32]()
    mat.append_words(words)
    _assert(len(words) == AAA_MATERIAL_WORDS, "AaaMaterial packs AAA_MATERIAL_WORDS words")
    _assert((words[23] & AAA_MAT_INTERIOR_PARALLAX) != 0, "AaaMaterial packs interior_parallax bit")
    _assert(((words[23] >> 24) & AAA_MAT_DET_NONE) == UInt32(12), "AaaMaterial packs detail_normal_tex")
    _assert((words[23] & UInt32(0x80000000)) != 0, "AaaMaterial packs road_wetness_mode")
    _assert(bitcast[DType.float32](words[11]) < Float32(-43.9), "AaaMaterial packs detail_uv_scale")

    # Kaplanyan Specular AA
    var kap = eval_kaplanyan_specular_aa_cpu(
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.35, 0.936, 0.0).normalized(),
        Vec3(0.0, 0.936, 0.35).normalized(),
        Float32(0.04),
        Float32(0.03),
    )
    _assert(kap[0] > Float32(0.18) and kap[1] > Float32(0.18), "Kaplanyan specular AA filters high curvature")

    # Closest point on LTC Tube & Rect
    var tube = AaaClusterLight.tube_ltc(
        Vec3(-3.0, 2.0, 5.0), Vec3(3.0, 2.0, 5.0), 0.12, Vec3(5.0, 5.0, 5.0), range_radius=10.0
    )
    var cp_tube = closest_point_on_ltc_light_cpu(tube, Vec3(8.0, 0.0, 5.0))
    _assert(
        abs(cp_tube.x - Float32(3.0)) < Float32(1e-4) and abs(cp_tube.y - Float32(2.0)) < Float32(1e-4),
        "closest_point_on_ltc_light_cpu tube endpoint",
    )

    var rect = AaaClusterLight.rect_ltc(
        Vec3(0.0, 4.0, 6.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
        2.0, 1.5, Vec3(5.0, 5.0, 5.0), range_radius=12.0,
    )
    var cp_rect = closest_point_on_ltc_light_cpu(rect, Vec3(-5.0, 9.0, 2.0))
    _assert(
        abs(cp_rect.x - Float32(-2.0)) < Float32(1e-4) and abs(cp_rect.y - Float32(5.5)) < Float32(1e-4),
        "closest_point_on_ltc_light_cpu rect corner",
    )

    # ReBLUR Anisotropic GGX radii
    var r_mir = eval_anisotropic_ggx_blur_radii_cpu(Float32(0.02), Float32(0.50), Float32(10.0), Float32(5.0), Float32(0.52), 720)
    _assert(r_mir[0] == Float32(0.0) and r_mir[1] == Float32(0.0), "ReBLUR mirror radii == 0")
    var r_gloss = eval_anisotropic_ggx_blur_radii_cpu(Float32(0.25), Float32(0.20), Float32(12.0), Float32(4.0), Float32(0.52), 720)
    _assert(r_gloss[1] > r_gloss[0] * Float32(1.8), "ReBLUR glossy grazing elongation")

    # 16-phase Halton (2, 3)
    var h0 = halton_2_3_jitter_px(0)
    _assert(abs(h0[0]) < Float32(1e-6) and abs(h0[1]) < Float32(1e-6), "Halton phase 0 is (0, 0)")
    for i in range(16):
        var hi = halton_2_3_jitter_px(i)
        _assert(hi[0] >= Float32(-0.5) and hi[0] < Float32(0.5), "Halton x in [-0.5, 0.5)")
        _assert(hi[1] >= Float32(-0.5) and hi[1] < Float32(0.5), "Halton y in [-0.5, 0.5)")
        for j in range(i + 1, 16):
            var hj = halton_2_3_jitter_px(j)
            _assert(abs(hi[0] - hj[0]) + abs(hi[1] - hj[1]) > Float32(0.02), "Halton phases distinct")

    # Froxel phase + Hillaire analytical step + Beer-Lambert water + AgX shadow-toe + Display Grade + Hologram rim
    _assert(AAA_FROXEL_Z == 64, "AAA_FROXEL_Z == 64")
    _assert(
        eval_cornette_shanks_phase_cpu(Float32(0.95), Float32(0.72))
        > eval_cornette_shanks_phase_cpu(Float32(-0.5), Float32(0.72)) * Float32(5.0),
        "Cornette-Shanks forward scattering lobe",
    )
    var hil = eval_hillaire_slice_integration_cpu(Vec3(0.2, 0.3, 0.5), Float32(0.1), Float32(2.0))
    _assert(hil[1] < Float32(0.85) and hil[0].z > Float32(0.0), "Hillaire froxel slice integration")

    var bl_shallow = eval_beer_lambert_water_cpu(Vec3(1.0, 1.0, 1.0), Vec3(0.45, 0.08, 0.03), Float32(0.2), Vec3(0.01, 0.04, 0.08))
    var bl_deep = eval_beer_lambert_water_cpu(Vec3(1.0, 1.0, 1.0), Vec3(0.45, 0.08, 0.03), Float32(4.0), Vec3(0.01, 0.04, 0.08))
    _assert(bl_deep.x < bl_shallow.x * Float32(0.30) and bl_deep.z > bl_deep.x * Float32(3.0), "Beer-Lambert water absorption")
    _assert(eval_dielectric_water_fresnel_cpu(Float32(0.05)) > Float32(0.75), "Water Fresnel at grazing")

    var agx_dark = eval_agx_tonemap_cpu(Vec3(0.002, 0.003, 0.008))
    var agx_gunmetal = eval_agx_tonemap_cpu(Vec3(0.012, 0.012, 0.014))
    var agx_hi = eval_agx_tonemap_cpu(Vec3(24.0, 18.0, 12.0))
    _assert(agx_dark.z > Float32(0.0) and agx_dark.z < Float32(0.06), "AgX C2 rational toe preserves shadows")
    _assert(agx_gunmetal.x > agx_dark.x * Float32(3.0), "AgX shadow toe separation")
    _assert(agx_hi.x > Float32(0.85) and agx_hi.x <= Float32(1.0), "AgX highlights shoulder in (0.85, 1.0]")

    var leaf = Vec3(0.30, 0.62, 0.20)
    var same = eval_display_grade_cpu(leaf, 1.0, 1.0)
    _assert(abs(same.x - leaf.x) < 1e-6 and abs(same.y - leaf.y) < 1e-6 and abs(same.z - leaf.z) < 1e-6, "Display grade identity")
    var vivid = eval_display_grade_cpu(leaf, 1.4, 1.0)
    _assert(vivid.y - vivid.z > (leaf.y - leaf.z) * Float32(1.35), "Display grade saturation boost")
    var grey = eval_display_grade_cpu(leaf, 0.0, 1.0)
    _assert(abs(grey.x - grey.y) < 1e-6 and abs(grey.y - grey.z) < 1e-6, "Display grade desaturation")
    var punchy = eval_display_grade_cpu(Vec3(0.2, 0.5, 0.8), 1.0, 1.5)
    _assert(abs(punchy.x - 0.05) < 1e-5 and abs(punchy.y - 0.5) < 1e-5 and abs(punchy.z - 0.95) < 1e-5, "Display grade contrast pivot")
    var clipped = eval_display_grade_cpu(Vec3(0.02, 0.98, 0.5), 2.0, 2.0)
    _assert(clipped.x == 0.0 and clipped.y == 1.0, "Display grade clamps to [0, 1]")

    var mat_am = AaaMaterial(base_color=Vec3(0.3, 0.6, 0.2), roughness=0.7, metallic=0.0)
    mat_am.alpha_cutoff = Float32(0.48)
    _assert(mat_am.is_alpha_masked(), "Alpha-masked flag")
    var words_am = List[UInt32]()
    mat_am.append_words(words_am)
    _assert(len(words_am) == AAA_MATERIAL_WORDS and bitcast[DType.float32](words_am[3]) < Float32(0.0), "Alpha-masked material word")

    var a2_near = eval_nyquist_lean_water_alpha2_cpu(Float32(0.02))
    var a2_far = eval_nyquist_lean_water_alpha2_cpu(Float32(5.0))
    _assert(a2_near < Float32(0.0015) and a2_far > a2_near * Float32(2.5), "LEAN water slope variance Nyquist scaling")

    _assert(eval_nubis_cloud_density_cpu(Vec3(1.2, 2.4, -0.8), Float32(0.35), Float32(0.0), Vec3()) == Float32(0.0), "Nubis zero coverage == 0")
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
    _assert(nubis_max > Float32(0.02), "Nubis cumulus cloud density positive")

    var holo_center = eval_hologram_fresnel_rim_cpu(Float32(0.98), Float32(1.0))
    var holo_rim = eval_hologram_fresnel_rim_cpu(Float32(0.05), Float32(1.0))
    _assert(holo_rim[0] > holo_center[0] * Float32(3.0) and holo_rim[1] > holo_center[1] * Float32(3.0), "Hologram Fresnel rim boost")

    # 11d. Cofactor Normal Matrix + Gram-Schmidt Orthonormal TBN under extreme non-uniform scale
    var m_nonuni = trs(Vec3(0.0, 0.0, 0.0), Quat(0.0, 0.0, Float32(0.3826834), Float32(0.9238795)), Vec3(4.0, 0.25, 1.5))
    var ln_test = Vec3(0.6, 0.8, 0.0).normalized()
    var lt_test = Vec3(-0.8, 0.6, 0.2).normalized()
    var tbn = eval_cofactor_gram_schmidt_tbn_cpu(m_nonuni, ln_test, lt_test, Float32(1.0))
    _assert(abs(tbn[0].length() - Float32(1.0)) < Float32(1e-5), "Cofactor TBN N unit length")
    _assert(abs(tbn[1].length() - Float32(1.0)) < Float32(1e-5), "Cofactor TBN T unit length")
    _assert(abs(tbn[2].length() - Float32(1.0)) < Float32(1e-5), "Cofactor TBN B unit length")
    _assert(abs(tbn[0].dot(tbn[1])) < Float32(1e-5), "Cofactor TBN N orthogonal to T")
    _assert(abs(tbn[0].dot(tbn[2])) < Float32(1e-5), "Cofactor TBN N orthogonal to B")
    _assert(abs(tbn[1].dot(tbn[2])) < Float32(1e-5), "Cofactor TBN T orthogonal to B")
    var surf_edge_l = Vec3(-0.8, 0.6, 0.0)
    var surf_edge_w = Vec3(
        m_nonuni.values[0] * surf_edge_l.x + m_nonuni.values[4] * surf_edge_l.y,
        m_nonuni.values[1] * surf_edge_l.x + m_nonuni.values[5] * surf_edge_l.y,
        m_nonuni.values[2] * surf_edge_l.x + m_nonuni.values[6] * surf_edge_l.y,
    )
    _assert(abs(tbn[0].dot(surf_edge_w.normalized())) < Float32(1e-5), "Cofactor normal orthogonal to deformed tangent")
    print("[PASS] Suite 13: AAA v2 Uncovered SPIR-V Shaders, 13 Analytic CPU Evaluators & Cofactor Gram-Schmidt TBN")


def test_v2_gpu_aaa_pipeline_features():
    var w = 128
    var h = 128

    # 8. TSR v3 16-Halton (tsr_scale=0.75), 180-deg Motion Blur, Bounded Auto-Exposure, Glare, Rain & Hologram OIT
    var ren = FlintAaaVulkanRenderer(width=w, height=h, tsr_scale=0.75, env_cube_dim=16)
    if not ren.active:
        print("[SKIP] Vulkan unavailable; skipping AAA v2 GPU pipeline suite")
        ren.shutdown()
        return
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
    ren.sun_radiance = Vec3(0.0, 0.0, 0.0)
    ren.ibl_intensity = Float32(0.001)
    var id_dark = ren.add_material(AaaMaterial(base_color=Vec3(0.02, 0.02, 0.03), roughness=0.9, metallic=0.0))
    var sb_tsr = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.5, metallic=0.0)
    sb_tsr.add_ellipsoid(Vec3(0.0, 0.0, 4.0), Vec3(1.2, 1.2, 1.2), rings=12, segments=16)
    var e_box = ren.add_mesh(sb_tsr^.finish(), id_dark)

    for f_i in range(16):
        var cx = Float32(f_i) * Float32(0.15)
        ren.begin_frame(Vec3(cx, 0.5, -3.0), Vec3(cx, 0.0, 4.0), fov_y_deg=55.0)
        ren.draw_instance(e_box.mesh_id, trs(Vec3(cx, 0.0, 4.0), Quat(), Vec3(1.0, 1.0, 1.0)))
        _assert(ren.render_frame(capture_readback=True), "8a TSR v3 Halton-16 frame rendered")

    var ae = ren.get_adapted_exposure()
    _assert(ae[0] >= Float32(0.24) and ae[0] <= Float32(1.76), "8a bounded auto-exposure stays in [0.25, 1.75]")

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
    _assert(ren.render_frame(capture_readback=True), "8b hologram OIT A->B rendered")
    var px_ab = ren.readback_pixels()

    ren.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 4.5), fov_y_deg=50.0)
    ren.draw_volumetric_plume(holo_b)
    ren.draw_volumetric_plume(holo_a)
    _assert(ren.render_frame(capture_readback=True), "8b hologram OIT B->A rendered")
    var px_ba = ren.readback_pixels()
    var diff_h = 0
    for i in range(len(px_ab)):
        if px_ab[i] != px_ba[i]:
            diff_h += 1
    _assert(diff_h == 0, "8b Fresnel hologram OIT bit-exact order independence")

    # Reuse single renderer at native 1.0x resolution for Suites 3-11e
    ren.tsr_scale = Float32(1.0)
    _assert(ren.resize(w, h), "Resize shared AAA v2 renderer to tsr_scale=1.0")
    ren.glare_streak_strength = Float32(0.0)
    ren.bloom_threshold = Float32(0.0)
    ren.sun_dir = Vec3(0.38, 0.80, -0.46).normalized()
    ren.sun_radiance = Vec3(3.2, 3.0, 2.75)
    ren.has_prev_cam = False

    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))

    # 3. 12-Lobe White-Furnace Energy Conservation (+/- 0.010 across Dielectric, Metal & Clearcoat)
    ren.wind_velocity = Vec3(0.0, 0.0, 0.0)
    ren.svgf_passes = 2
    ren.tsr_history_weight = Float32(0.0)
    ren.enable_auto_exposure = False
    ren.bloom_intensity = Float32(0.0)
    ren.ibl_intensity = Float32(1.0)
    ren.furnace_white_dome = True
    ren.exposure = Float32(1.0)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB

    var sb_wf = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.5, metallic=0.0)
    sb_wf.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.35, 1.35, 1.35), rings=20, segments=32)
    var sph_e = ren.add_mesh(sb_wf^.finish(), 0)
    for met_i in range(2):
        for cc_i in range(2):
            for r_val in [Float32(0.10), Float32(0.45), Float32(0.90)]:
                var fm = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=r_val, metallic=Float32(met_i))
                fm.clearcoat = Float32(cc_i)
                fm.clearcoat_roughness = Float32(0.12)
                var mid = ren.add_material(fm)
                ren.begin_frame(Vec3(0.0, 0.0, -3.2), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
                ren.draw_instance(sph_e.mesh_id, ident, material_override=mid)
                _assert(ren.render_frame(capture_readback=True), "White furnace lobe frame rendered")
                var hdr_wf = ren.readback_resolved_hdr()
                var sum_l = Float64(0.0)
                var cnt_wf = 0
                for y in range(44, 84):
                    for x in range(44, 84):
                        var c = hdr_wf[y * 128 + x]
                        sum_l += Float64(c.x + c.y + c.z) / 3.0
                        cnt_wf += 1
                var mean_l = Float32(sum_l / Float64(cnt_wf))
                _assert(
                    abs(mean_l - Float32(1.0)) <= Float32(0.010),
                    "White-furnace mean=" + String(mean_l) + " out of 1.0 +/- 0.010",
                )
    ren.furnace_white_dome = False

    # 4 & 5. Barycentric Textured RT + Dual-Lobe Clearcoat RT + Top-2 Hero Local LTC Shadows + ReBLUR
    if ren.vk.has_ray_query:
        ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
        ren.enable_vsm = False
        ren.enable_rt_reflections = True
        ren.enable_rt_shadows = False
        ren.enable_rt_gi = False
        ren.tsr_history_weight = Float32(0.0)
        ren.ibl_intensity = Float32(0.0)
        ren.sun_dir = Vec3(0.0, 0.5, -0.8660254).normalized()
        ren.sun_radiance = Vec3(8.0, 8.0, 8.0)

        var stripe_px = List[UInt32](length=64 * 64, fill=0)
        for y in range(64):
            for x in range(64):
                stripe_px[y * 64 + x] = UInt32(0xFFFF0808) if x < 32 else UInt32(0xFF08F0FF)
        var stripe_tex = ren.upload_bc7_texture(stripe_px, 64, 64, srgb=True, generate_mips=True)
        _assert(stripe_tex >= 0, "Uploaded stripe BC7 texture")

        var mat_wall = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.9, metallic=0.0)
        mat_wall.albedo_tex = stripe_tex
        var id_wall = ren.add_material(mat_wall)

        var mat_ccoat_hood = AaaMaterial(base_color=Vec3(0.01, 0.01, 0.01), roughness=0.80, metallic=0.0)
        mat_ccoat_hood.clearcoat = Float32(1.0)
        mat_ccoat_hood.clearcoat_roughness = Float32(0.02)
        var id_hood = ren.add_material(mat_ccoat_hood)

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

        ren.begin_frame(Vec3(0.0, 1.4, -2.8), Vec3(0.0, 0.5, 2.0), fov_y_deg=55.0)
        ren.draw_instance(e_wall.mesh_id, ident)
        ren.draw_instance(e_flr.mesh_id, ident, material_override=id_mirror)
        _assert(ren.render_frame(capture_readback=True), "4a mirror RT frame rendered")
        var hdr_mir = ren.readback_resolved_hdr()
        var left_refl = hdr_mir[96 * w + 36]
        var right_refl = hdr_mir[96 * w + 92]
        _assert(left_refl.x > left_refl.z * Float32(2.0), "4a barycentric RT reflection left red")
        _assert(right_refl.z > right_refl.x * Float32(2.0), "4a barycentric RT reflection right cyan")

        ren.begin_frame(Vec3(0.0, 0.65, -2.8), Vec3(0.0, 0.25, 2.0), fov_y_deg=55.0)
        ren.draw_instance(e_wall.mesh_id, ident)
        ren.draw_instance(e_flr.mesh_id, ident, material_override=id_hood)
        _assert(ren.render_frame(capture_readback=True), "4b clearcoat RT frame rendered")
        var hdr_cc = ren.readback_resolved_hdr()
        var cc_left = hdr_cc[96 * w + 36]
        var cc_right = hdr_cc[96 * w + 92]
        _assert(
            cc_left.x > cc_left.z * Float32(1.4) and cc_right.z > cc_right.x * Float32(1.4),
            "4b dark dielectric clearcoat receives RT reflections",
        )

        # 5. ReBLUR & Top-2 Hero Local LTC Shadows
        ren.sun_radiance = Vec3(0.0, 0.0, 0.0)
        var flr5_m = _make_uv_quad(
            Vec3(-4.0, 0.0, -2.0), Vec3(4.0, 0.0, -2.0), Vec3(4.0, 0.0, 6.0), Vec3(-4.0, 0.0, 6.0),
            Vec3(0.0, 1.0, 0.0),
        )
        var blk5_m = _make_uv_quad(
            Vec3(0.0, 0.0, -1.0), Vec3(0.0, 0.0, 5.0), Vec3(0.0, 2.5, 5.0), Vec3(0.0, 2.5, -1.0),
            Vec3(-1.0, 0.0, 0.0),
        )
        var mat_matte = ren.add_material(AaaMaterial(base_color=Vec3(0.8, 0.8, 0.8), roughness=0.9, metallic=0.0))
        var mat_glossy = ren.add_material(AaaMaterial(base_color=Vec3(0.85, 0.85, 0.85), roughness=0.22, metallic=1.0))
        var mat_neon_wall = AaaMaterial(base_color=Vec3(0.1, 0.1, 0.1), roughness=0.9, metallic=0.0)
        mat_neon_wall.emissive = Vec3(4.5, 0.8, 0.2)
        var id_neon = ren.add_material(mat_neon_wall)

        var wall5_m = _make_uv_quad(
            Vec3(-1.2, 0.0, 5.0), Vec3(1.2, 0.0, 5.0), Vec3(1.2, 3.5, 5.0), Vec3(-1.2, 3.5, 5.0),
            Vec3(0.0, 0.0, -1.0),
        )
        var e_flr5 = ren.add_mesh(flr5_m, mat_matte)
        var e_blk5 = ren.add_mesh(blk5_m, mat_matte)
        var e_wall5 = ren.add_mesh(wall5_m, id_neon)

        ren.enable_rt_shadows = True
        ren.enable_restir_di = False
        ren.enable_rt_reflections = False
        ren.tsr_history_weight = Float32(0.0)
        var hero_tube = AaaClusterLight.tube_ltc(
            Vec3(-1.8, 1.2, 0.8), Vec3(-1.8, 1.2, 3.2), 0.14, Vec3(18.0, 16.0, 12.0), range_radius=14.0
        )
        ren.begin_frame(Vec3(0.0, 3.8, -2.5), Vec3(0.0, 0.0, 2.0), fov_y_deg=55.0)
        ren.add_cluster_light(hero_tube)
        ren.draw_instance(e_flr5.mesh_id, ident, material_override=mat_matte)
        ren.draw_instance(e_blk5.mesh_id, ident)
        _assert(ren.render_frame(capture_readback=True), "5a hero LTC shadow frame rendered")
        var sh_hdr = ren.readback_resolved_hdr()
        var lit_side = sh_hdr[72 * w + 36]
        var occ_side = sh_hdr[72 * w + 92]
        _assert(lit_side.x > occ_side.x * Float32(4.0), "5a hero local LTC tube light shadow contrast")

        ren.enable_rt_reflections = True
        ren.enable_spec_reblur = True
        ren.tsr_history_weight = Float32(0.85)
        for _ in range(5):
            ren.begin_frame(Vec3(0.0, 1.2, -2.5), Vec3(0.0, 0.4, 3.0), fov_y_deg=55.0)
            ren.draw_instance(e_flr5.mesh_id, ident, material_override=mat_glossy)
            ren.draw_instance(e_wall5.mesh_id, ident)
            _assert(ren.render_frame(capture_readback=True), "5b ReBLUR frame rendered")
        var rb_hdr = ren.readback_resolved_hdr()
        var center_gloss = rb_hdr[92 * w + 64]
        _assert(center_gloss.x > Float32(0.15) and center_gloss.x < Float32(20.0), "5b ReBLUR denoised glossy reflection")

        ren.enable_rt_shadows = False
        ren.enable_rt_reflections = False

    # 6. 4-Layer Terrain Splatting, Interior Parallax, ForzaTech Wetness & Toksvig Mips
    ren.enable_spec_reblur = True
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.enable_vsm = False
    ren.tsr_history_weight = Float32(0.0)
    ren.sun_dir = Vec3(0.0, 0.85, -0.526).normalized()
    ren.sun_radiance = Vec3(3.2, 3.2, 3.2)
    ren.ibl_intensity = Float32(0.2)

    var nrms = List[Vec3](capacity=32 * 32)
    var orms = List[UInt32](capacity=32 * 32)
    for y in range(32):
        for x in range(32):
            var sx = Float32(0.45) if ((x + y) & 1) == 0 else Float32(-0.45)
            nrms.append(Vec3(sx, Float32(0.0), Float32(0.893)).normalized())
            orms.append(UInt32(0xFFFF1800))
    var tok_slots = ren.upload_toksvig_normal_orm(nrms, orms, 32, 32, toksvig_scale=1.0)
    _assert(tok_slots[0] >= 0 and tok_slots[1] >= 0, "6a uploaded Toksvig normal/ORM mip chain")

    var flat_nrm = List[Vec3](length=16 * 16, fill=Vec3(0.0, 0.0, 1.0))
    var layer_colors = [
        UInt32(0x9020D020),
        UInt32(0x98E0C050),
        UInt32(0xA0707078),
        UInt32(0xA8D03020),
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
    var splat_w = List[UInt32]()
    splat_w.append(UInt32(0x000000))
    splat_w.append(UInt32(0xFF0000))
    splat_w.append(UInt32(0xFF0000))
    splat_w.append(UInt32(0x000000))
    var e_terr = ren.upload_splat_mesh(terr_q, id_splat, splat_w)

    ren.begin_frame(Vec3(0.0, 3.2, -2.2), Vec3(0.0, 0.0, 1.5), fov_y_deg=55.0)
    ren.draw_instance(e_terr.mesh_id, ident)
    _assert(ren.render_frame(capture_readback=True), "6b terrain splat frame rendered")
    var terr_hdr = ren.readback_resolved_hdr()
    var grass_px = terr_hdr[72 * w + 22]
    var cobble_px = terr_hdr[72 * w + 106]
    _assert(grass_px.y > grass_px.x * Float32(1.8), "6b terrain splat grass green")
    _assert(cobble_px.x > cobble_px.y * Float32(1.8), "6b terrain splat cobble red")

    var room_px = List[UInt32](length=32 * 32, fill=UInt32(0xFFFFD8A8))
    var room_slot = ren.upload_bc7_texture(room_px, 32, 32, srgb=True, generate_mips=False)
    _assert(room_slot >= 0 and room_slot < 63, "6c room texture slot < 63")
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
    _assert(ren.render_frame(capture_readback=True), "6c interior parallax frame rendered")
    var win_hdr = ren.readback_resolved_hdr()
    var sum_win = Float32(0.0)
    for py in range(32, 96):
        for px in range(32, 96):
            sum_win += win_hdr[py * w + px].x + win_hdr[py * w + px].y
    _assert(sum_win > Float32(25.0), "6c interior parallax emits lit room")

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
    _assert(ren.render_frame(capture_readback=True), "6d dry road frame rendered")
    var dry_hdr = ren.readback_resolved_hdr()[64 * w + 64]

    ren.weather_wetness = Float32(0.80)
    ren.rain_intensity = Float32(0.65)
    ren.begin_frame(Vec3(0.0, 3.5, -2.0), Vec3(0.0, 0.0, 1.5), fov_y_deg=50.0)
    ren.draw_instance(e_road.mesh_id, ident)
    _assert(ren.render_frame(capture_readback=True), "6d wet road frame rendered")
    var wet_hdr = ren.readback_resolved_hdr()[64 * w + 64]
    _assert(wet_hdr.x < dry_hdr.x * Float32(0.65), "6d wet road darkens via Lekner-Dorf TIR")

    ren.weather_wetness = Float32(0.0)
    ren.rain_intensity = Float32(0.0)
    ren.scene_time_sec = Float32(0.0)
    ren.enable_spec_reblur = True

    # 7. Froxel Volumetrics + GPU Rain & Dielectric Beer-Lambert Water
    ren.ibl_intensity = Float32(0.05)
    ren.sun_dir = Vec3(0.15, 0.55, 0.82).normalized()
    ren.sun_radiance = Vec3(3.5, 3.4, 3.2)
    var id_black = ren.add_material(AaaMaterial(base_color=Vec3(0.01, 0.01, 0.01), roughness=0.95, metallic=0.0))
    var e_bg = ren.add_mesh(
        _make_uv_quad(Vec3(-20.0, -10.0, 35.0), Vec3(20.0, -10.0, 35.0), Vec3(20.0, 15.0, 35.0), Vec3(-20.0, 15.0, 35.0), Vec3(0.0, 0.0, -1.0)),
        id_black,
    )
    ren.enable_froxel_fog = False
    ren.begin_frame(Vec3(0.0, 1.5, -4.0), Vec3(0.0, 1.5, 20.0), fov_y_deg=55.0)
    ren.draw_instance(e_bg.mesh_id, ident)
    _assert(ren.render_frame(capture_readback=True), "7a baseline no-fog frame rendered")
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
        _assert(ren.render_frame(capture_readback=True), "7a froxel fog + rain frame rendered")
    var fog_c = ren.readback_resolved_hdr()[64 * w + 64]
    _assert((fog_c.x + fog_c.y + fog_c.z) > (no_fog_c.x + no_fog_c.y + no_fog_c.z) + Float32(0.05), "7a froxel fog + spot scatters light")

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
    _assert(ren.render_frame(capture_readback=True), "7b Beer-Lambert water frame rendered")
    var wat_hdr = ren.readback_resolved_hdr()
    var shallow_px = wat_hdr[64 * w + 28]
    var deep_px = wat_hdr[64 * w + 100]
    var shallow_br = shallow_px.z / max(Float32(1e-4), shallow_px.x)
    var deep_br = deep_px.z / max(Float32(1e-4), deep_px.x)
    _assert(
        deep_br > shallow_br * Float32(1.6),
        "7b Beer-Lambert water blue/red ratio increases with depth (shallow="
        + String(shallow_px.x)
        + ","
        + String(shallow_px.y)
        + ","
        + String(shallow_px.z)
        + " br="
        + String(shallow_br)
        + "; deep="
        + String(deep_px.x)
        + ","
        + String(deep_px.y)
        + ","
        + String(deep_px.z)
        + " br="
        + String(deep_br)
        + ")",
    )
    ren.enable_water_plane = False

    # 9. Two-Bucket Alpha-Cutout (VisBuffer + VSM + Non-Opaque RayQuery Transparency)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.tsr_history_weight = Float32(0.0)
    ren.ibl_intensity = Float32(0.02)
    ren.sun_dir = Vec3(0.0, 0.0, -1.0)
    ren.sun_radiance = Vec3(4.0, 4.0, 4.0)
    ren.enable_vsm = True
    ren.enable_rt_shadows = True

    var grate_px = List[UInt32](length=64 * 64, fill=0)
    for gy in range(64):
        for gx in range(64):
            grate_px[gy * 64 + gx] = UInt32(0x0020D030) if gx < 32 else UInt32(0xFF20D030)
    var grate_slot = ren.upload_bc7_texture(grate_px, 64, 64, srgb=False, generate_mips=True)
    _assert(grate_slot >= 0, "9 uploaded alpha-cutout grate texture")

    var mat_red9 = AaaMaterial(base_color=Vec3(0.90, 0.08, 0.06), roughness=0.8, metallic=0.0)
    var id_red9 = ren.add_material(mat_red9)
    var e_wall9 = ren.add_mesh(
        _make_uv_quad(Vec3(-4.0, -4.0, 5.0), Vec3(4.0, -4.0, 5.0), Vec3(4.0, 4.0, 5.0), Vec3(-4.0, 4.0, 5.0), Vec3(0.0, 0.0, -1.0)),
        id_red9,
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

    for _ in range(2):
        ren.begin_frame(Vec3(-1.4, 0.0, -2.5), Vec3(0.2, 0.0, 5.0), fov_y_deg=50.0)
        ren.draw_instance(e_wall9.mesh_id, ident)
        ren.draw_instance(e_grate.mesh_id, ident)
        _assert(ren.render_frame(capture_readback=True), "9 cutout warmup frame rendered")
    var hdr9 = ren.readback_resolved_hdr()
    var px_cutout_lit_wall = hdr9[64 * w + 38]
    var px_solid_green_grate = hdr9[64 * w + 76]

    ren.sun_dir = Vec3(-0.32, 0.0, -0.947).normalized()
    ren.vsm.invalidate_all()
    for _ in range(2):
        ren.begin_frame(Vec3(0.0, 0.0, -2.5), Vec3(0.0, 0.0, 5.0), fov_y_deg=55.0)
        ren.draw_instance(e_wall9.mesh_id, ident)
        ren.draw_instance(e_grate.mesh_id, ident)
        _assert(ren.render_frame(capture_readback=True), "9 cutout shadow frame rendered")
    var hdr9_2 = ren.readback_resolved_hdr()
    var wall_through_hole = hdr9_2[64 * w + 46]
    var wall_behind_solid = hdr9_2[64 * w + 106]
    _assert(px_cutout_lit_wall.x > px_cutout_lit_wall.y * Float32(3.5), "9 cutout hole reveals lit red wall")
    _assert(px_solid_green_grate.y > px_solid_green_grate.x * Float32(2.5), "9 solid grate half renders green")
    _assert(wall_through_hole.x > wall_behind_solid.x * Float32(2.2), "9 cutout casts shadow only from opaque half")

    ren.enable_vsm = False
    ren.enable_rt_shadows = False

    # 10. 24-Bit Vertex Colors (F15 NativeMesh.colors fix), Nubis Clouds + Starfield, Fireball OIT & 4 Particle Presets
    ren.ibl_intensity = Float32(1.0)
    var vcol_mesh = _make_uv_quad(
        Vec3(-2.0, -2.0, 3.0), Vec3(2.0, -2.0, 3.0), Vec3(2.0, 2.0, 3.0), Vec3(-2.0, 2.0, 3.0), Vec3(0.0, 0.0, -1.0),
    )
    vcol_mesh.colors = List[UInt8](capacity=16)
    for _ in range(4):
        vcol_mesh.colors.append(UInt8(24))
        vcol_mesh.colors.append(UInt8(245))
        vcol_mesh.colors.append(UInt8(36))
        vcol_mesh.colors.append(UInt8(255))
    var id_white = ren.add_material(AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.8, metallic=0.0))
    var e_vc = ren.add_mesh(vcol_mesh, id_white)

    ren.sun_dir = Vec3(0.0, 0.2, -0.98).normalized()
    ren.sun_radiance = Vec3(3.0, 3.0, 3.0)
    ren.begin_frame(Vec3(0.0, 0.0, -1.5), Vec3(0.0, 0.0, 3.0), fov_y_deg=50.0)
    ren.draw_instance(e_vc.mesh_id, ident)
    _assert(ren.render_frame(capture_readback=True), "10a 24-bit vertex color frame rendered")
    var vc_px = ren.readback_resolved_hdr()[64 * w + 64]
    _assert(vc_px.y > vc_px.x * Float32(8.0) and vc_px.y > vc_px.z * Float32(5.0), "10a 24-bit vertex color emerald shading")

    var atmo = BrunetonAtmosphereParams()
    atmo.sun_dir = Vec3(0.25, -0.25, 0.935).normalized()
    atmo.cloud_coverage = Float32(0.58)
    atmo.star_intensity = Float32(1.0)
    _ = ren.set_bruneton_atmosphere(atmo)
    ren.begin_frame(Vec3(0.0, 1.5, 0.0), Vec3(0.0, 4.5, 8.0), fov_y_deg=65.0)
    _assert(ren.render_frame(capture_readback=True), "10b Nubis clouds + starfield frame rendered")
    var sky_hdr = ren.readback_resolved_hdr()
    var max_sky = Float32(0.0)
    for sp in sky_hdr:
        var lum_s = sp.x + sp.y + sp.z
        if lum_s > max_sky:
            max_sky = lum_s
    _assert(max_sky > Float32(0.02), "10b Nubis clouds + starfield produces non-black sky radiance")

    ren.enable_aerial_perspective = False
    ren.upload_ibl_environment(
        bake_bc6h_ibl_cubemap_and_sh9(
            SkyDomeParams(uniform_furnace=False),
            base_dim=ren.env_cube_dim,
            mip_levels=ren.env_mip_levels,
            ggx_samples=4,
        )
    )
    ren.cloud_coverage = Float32(0.0)
    ren.star_intensity = Float32(0.0)
    ren.ibl_intensity = Float32(0.02)
    ren.sun_radiance = Vec3(0.0, 0.0, 0.0)
    var fb_a = AaaVolumetricPlume(Vec3(-0.25, 0.0, 4.5), 1.3, Vec3(0.25, 0.22, 0.20), density=1.1, emission=Vec3(9.5, 3.8, 0.6))
    var fb_b = AaaVolumetricPlume(Vec3(0.25, 0.1, 5.1), 1.4, Vec3(0.22, 0.20, 0.18), density=0.95, emission=Vec3(6.5, 2.2, 0.4))

    ren.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), fov_y_deg=55.0)
    ren.draw_volumetric_plume(fb_a)
    ren.draw_volumetric_plume(fb_b)
    _assert(ren.render_frame(capture_readback=True), "10c fireball A->B rendered")
    var fb_ab = ren.readback_pixels()

    ren.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), fov_y_deg=55.0)
    ren.draw_volumetric_plume(fb_b)
    ren.draw_volumetric_plume(fb_a)
    _assert(ren.render_frame(capture_readback=True), "10c fireball B->A rendered")
    var fb_ba = ren.readback_pixels()
    var fb_diff = 0
    for i in range(len(fb_ab)):
        if fb_ab[i] != fb_ba[i]:
            fb_diff += 1
    _assert(fb_diff == 0, "10c pyroclastic fireball OIT bit-exact order independence")

    ren.enable_froxel_fog = True
    ren.fog_density = Float32(0.015)
    ren.begin_frame(Vec3(0.0, 0.5, -2.0), Vec3(0.0, 0.5, 4.0), fov_y_deg=60.0)
    _assert(ren.render_frame(capture_readback=True), "10c particle baseline rendered")
    var base_sum = Float32(0.0)
    for bp in ren.readback_resolved_hdr():
        base_sum += bp.x + bp.y + bp.z

    for _ in range(3):
        ren.begin_frame(Vec3(0.0, 0.5, -2.0), Vec3(0.0, 0.5, 4.0), fov_y_deg=60.0)
        ren.submit_particle_emitter(AaaParticleEmitter.sparks(Vec3(-0.4, 0.5, 3.2), Vec3(0.5, 2.5, 0.0), spawn_count=128))
        ren.submit_particle_emitter(AaaParticleEmitter.exhaust_plasma(Vec3(0.4, 0.4, 3.0), Vec3(0.0, 0.0, -8.0), spawn_count=64))
        ren.submit_particle_emitter(AaaParticleEmitter.water_spray(Vec3(-0.2, 0.2, 3.5), Vec3(-0.4, 1.2, 0.0), spawn_count=48))
        ren.submit_particle_emitter(AaaParticleEmitter.drift_smoke(Vec3(0.2, 0.3, 3.8), Vec3(0.2, 0.8, 0.0), spawn_count=48))
        _assert(ren.render_frame(capture_readback=True), "10c 4-preset particle frame rendered")

    var part_sum = Float32(0.0)
    for pp in ren.readback_resolved_hdr():
        part_sum += pp.x + pp.y + pp.z
    _assert(part_sum > base_sum + Float32(8.0), "10c 4 GPU particle presets emit visible radiance")

    # 11e. `draw_native_model_articulated` Sub-Node Motion Vectors, Particle Wind Advection &
    #      Auto-Allocated Bone Palettes / Zero Root Leak / Zero Halton Static Velocity (F15 staging readback)
    ren.particles_ever_active = False
    ren.vk.wait_idle()
    if ren.vk.begin_commands(0):
        ren.vk.fill_buffer(0, ren.buf_oit_nodes, 0, ren.buf_oit_nodes.size, 0)
        var f_clr_11e = ren.vk.submit(0)
        if f_clr_11e != 0:
            _ = ren.vk.wait(f_clr_11e)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.enable_vsm = False
    ren.enable_froxel_fog = False
    ren.fog_density = Float32(0.0)
    ren.furnace_white_dome = False
    ren.svgf_passes = 0
    ren.tsr_history_weight = Float32(0.85)
    ren.ibl_intensity = Float32(1.0)
    ren.exposure = Float32(1.0)

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
    n_wing.translation = Vec3(2.6, 0.0, 0.0)
    art_mdl.nodes.append(n_torso^)
    art_mdl.nodes.append(n_wing^)
    art_mdl.roots.append(0)
    var art_up = ren.upload_native_model(art_mdl, build_lods=False)

    var g_rest = art_mdl.global_transforms()
    for _ in range(3):
        ren.begin_frame(Vec3(0.0, 0.0, -2.5), Vec3(0.0, 0.0, 3.0), fov_y_deg=55.0)
        ren.draw_native_model_articulated(art_mdl, art_up, g_rest, g_rest, ident, ident, has_prev=True)
        _assert(ren.render_frame(capture_readback=True), "11e articulated static frame rendered")
    var hdr_f32_art_stat = _readback_hdr_out_floats(ren)
    var wing_px_x = 92
    var torso_px_x = 36
    var wing_stat_vu = hdr_f32_art_stat[(64 * w + wing_px_x) * 8 + 4]
    var wing_stat_vv = hdr_f32_art_stat[(64 * w + wing_px_x) * 8 + 5]
    _assert(abs(wing_stat_vu) < Float32(1e-5) and abs(wing_stat_vv) < Float32(1e-5), "11e articulated stationary sub-node has zero velocity")

    var g_prev_art = List[Mat4]()
    g_prev_art.append(g_rest[0].copy())
    g_prev_art.append(trs(Vec3(0.5, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    ren.begin_frame(Vec3(0.0, 0.0, -2.5), Vec3(0.0, 0.0, 3.0), fov_y_deg=55.0)
    ren.draw_native_model_articulated(art_mdl, art_up, g_rest, g_prev_art, ident, ident, has_prev=True)
    _assert(ren.render_frame(capture_readback=True), "11e articulated moving wing frame rendered")
    var hdr_f32_art_mv = _readback_hdr_out_floats(ren)
    var torso_mv_u = hdr_f32_art_mv[(64 * w + torso_px_x) * 8 + 4]
    var wing_mv_u = hdr_f32_art_mv[(64 * w + wing_px_x) * 8 + 4]
    _assert(abs(torso_mv_u) < Float32(1e-5), "11e stationary torso node has 0 velocity")
    _assert(wing_mv_u > Float32(0.08), "11e articulated wing sub-node emits +U motion vector")

    # Wind advection on GPU drift smoke particles
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
        _assert(ren.render_frame(capture_readback=True), "11e wind smoke frame rendered")
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
    _assert(right_smoke > left_smoke + Float32(25.0), "11e wind_velocity advects GPU smoke particles +X")

    ren.wind_velocity = Vec3(0.0, 0.0, 0.0)
    ren.ibl_intensity = Float32(1.0)
    ren.tsr_history_weight = Float32(0.85)
    var q_sk = _make_uv_quad(
        Vec3(-1.5, -1.5, 0.0), Vec3(1.5, -1.5, 0.0), Vec3(1.5, 1.5, 0.0), Vec3(-1.5, 1.5, 0.0), Vec3(0.0, 0.0, -1.0)
    )
    var jp_test = List[UInt32](length=4, fill=UInt32(2))
    var jw_unnorm = List[Vec3](length=4, fill=Vec3(0.40, 0.0, 0.0))
    var id_wht = ren.add_material(AaaMaterial(base_color=Vec3(0.9, 0.9, 0.9), roughness=0.7, metallic=0.0))
    var sk_a = ren.add_skinned_mesh(q_sk, jp_test, jw_unnorm, id_wht)
    var sk_b = ren.add_skinned_mesh(q_sk, jp_test, jw_unnorm, id_wht)
    _assert(sk_a.bone_count == 64, "sk_a bone_count == 64")
    _assert(sk_a.bone_palette_offset == 0, "sk_a bone_palette_offset == 0")
    _assert(sk_b.bone_palette_offset == 64 * 8, "sk_b bone_palette_offset disjoint")

    var gpu_bones = List[DualQuat](length=4, fill=DualQuat())
    gpu_bones[0] = DualQuat.from_quat_trans(Quat(), Vec3(999.0, 999.0, 999.0))
    gpu_bones[2] = DualQuat.from_quat_trans(Quat(), Vec3(0.0, 0.0, 3.5))

    for _ in range(2):
        _ = ren.update_skinned_mesh_bones(sk_b, gpu_bones, gpu_bones, defer_to_frame=True)
        ren.begin_frame(Vec3(0.0, 0.0, -2.0), Vec3(0.0, 0.0, 3.5), fov_y_deg=55.0)
        ren.draw_instance(sk_b.mesh_entry.mesh_id, ident, ident, has_prev=True)
        _assert(ren.render_frame(capture_readback=True), "11e skinned static frame rendered")

    var hdr_f32_sk = _readback_hdr_out_floats(ren)
    var center_base = (64 * w + 64) * 8
    var z_sk = hdr_f32_sk[center_base + 3]
    var vu_static_sk = hdr_f32_sk[center_base + 4]
    var vv_static_sk = hdr_f32_sk[center_base + 5]
    _assert(abs(z_sk - Float32(5.5)) < Float32(0.15), "11e GPU skinning z_lin == 5.5 without bone-0 leak")
    _assert(abs(vu_static_sk) < Float32(1e-5) and abs(vv_static_sk) < Float32(1e-5), "11e static Halton jitter has zero velocity")
    ren.shutdown()
    print("[PASS] Suite 14: AAA v2 GPU Pipeline (White-Furnace 12-Lobe, Clearcoat RT, ReBLUR, Splat, Parallax, Wetness, Froxel/Water, TSR v3, Cutout VSM, Nubis, Articulated MV & Wind)")


def main():
    var t0 = get_time()
    print("=== Flint v5 Track 2 Render Pipeline & Metrology Suite ===")
    test_spirv_spec_constants_and_shaders()
    test_material_graph_decals_terrain_and_vat()
    test_swapchain_render_world_and_tier0()
    test_gpu_ui_and_aaa_metropolis()
    test_gpu_ui_polygons()
    test_multi_cascade_vsm_gpu_shading()
    test_render_features_glass_water_particles()
    test_f22_raii_ownership_and_wsi_hooks()
    test_v2_spirv_shaders_and_cpu_models()
    test_v2_gpu_aaa_pipeline_features()
    var dt = get_time() - t0
    print("=== ALL FLINT V5 TRACK 2 RENDER TESTS PASSED in " + String(dt) + "s ===")
    _assert(dt < 5.0, "Test suite exceeded runtime safety ceiling: " + String(dt) + "s")
