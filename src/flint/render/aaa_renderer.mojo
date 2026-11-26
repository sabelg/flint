"""Unified Vulkan 1.3/1.4 Hardware AAA Renderer (`FlintAaaVulkanRenderer`, `aaa_renderer.mojo`).

Integrates all 4 Milestones of the Flint Vulkan-Only 100% AAA Plan:
  - Milestone 1 (`V12`): `VK_KHR_dynamic_rendering` 64-Bit Visibility Buffer (`R32G32_UINT` +
    Reverse-Z `D32_SFLOAT`), 64v/124t GPU Meshlet `Hi-Z` + Normal-Cone Culling (`vkCmdDrawIndirectCount`),
    Analytic 3-Ray Differential `ddx/ddy` Barycentrics & Bindless `16x` Aniso `BC7 / BC5 / BC6H` TMU Sampling.
  - Milestone 2 (`V13`): Enterprise Multi-Lobe BSDF (Fdez-Aguera / Kulla-Conty Multi-Scatter GGX,
    Clearcoat, Dual-UV `RNM` Micro-Detail, Thin-Film Iridescence, Charlie Sheen, Burley SSS) &
    `16,384 x 16,384` Demand-Paged Virtual Shadow Maps (`VSM`) + Blocker-Search `PCSS`.
  - Milestone 3 (`V14`): Hardware `VK_KHR_acceleration_structure` (`BLAS`/`TLAS`) & `SPV_KHR_ray_query`
    Contact Shadows, Reflections, Split-Sum `BC6H` Cubemaps, 2-Bounce Lumen Surface-Cache GI & `SVGF`.
  - Milestone 4 (`V15`): Disocclusion-Reactive `FlintTSR v2` (`50% -> 100%/4K`), Per-Pixel 4-Node
    Exact `OIT` + 6-Way Lightmapped Beer-Powder Smoke, Declarative `VK_KHR_synchronization2` `AaaRenderGraph`
    with `OffsetAllocator` Aliasing, Async Compute Side-Queue Support & `AgX / HDR10 PQ / Metrology` Output.
"""
from std.ffi import external_call
from std.math import cos, exp2, log2, max, min, sin, sqrt, tan
from std.memory import bitcast
from flint.core.math import Vec3
from flint.core.transform import Mat4
from flint.render.camera_basis import view_right
from flint.platform.os import get_time
from flint.assets.model import NativeMesh, NativeModel
from flint.assets.bc_compress import (
    bake_toksvig_normal_orm_pyramid,
    downsample_argb_2x,
    downsample_normals_2x,
    encode_bc5_normals,
    encode_bc6h_hdr,
    encode_bc7_rgba,
)
from flint.assets.meshlet import AAA_VERTEX_BYTES, AAA_MESHLET_WORDS, AAA_VERTEX_WORDS, AaaGeometryPool, CookedMeshEntry, GeomSlot
from flint.platform.amdgpu import GpuBuffer
from flint.platform.vulkan import (
    ACCESS_AS_READ_KHR,
    ACCESS_AS_WRITE_KHR,
    ACCESS_COLOR_ATTACHMENT_WRITE,
    ACCESS_DEPTH_ATTACHMENT_READ,
    ACCESS_DEPTH_ATTACHMENT_WRITE,
    ACCESS_HOST_READ,
    ACCESS_INDIRECT_READ,
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    ACCESS_TRANSFER_READ,
    ACCESS_TRANSFER_WRITE,
    BUFFER_USAGE_AS_BUILD_INPUT_KHR,
    BUFFER_USAGE_DEVICE_ADDRESS,
    COMPARE_GREATER_OR_EQUAL,
    COMPARE_LESS_OR_EQUAL,
    CULL_MODE_NONE,
    CULL_MODE_FRONT,
    CULL_MODE_BACK,
    FRONT_FACE_CW,
    FRONT_FACE_CCW,
    INDEX_TYPE_UINT32,
    NO_IMAGE,
    FORMAT_R8_UNORM,
    FORMAT_BC5_UNORM,
    FORMAT_BC6H_UFLOAT,
    FORMAT_BC7_SRGB,
    FORMAT_BC7_UNORM,
    FORMAT_D32_SFLOAT,
    FORMAT_R16G16_SFLOAT,
    FORMAT_R16G16B16A16_SFLOAT,
    FORMAT_R32G32_UINT,
    IMAGE_USAGE_COLOR_ATTACHMENT,
    IMAGE_USAGE_DEPTH_ATTACHMENT,
    IMAGE_USAGE_SAMPLED,
    IMAGE_USAGE_TRANSFER_TARGET,
    LAYOUT_COLOR_ATTACHMENT,
    LAYOUT_DEPTH_ATTACHMENT,
    LAYOUT_SHADER_READ_ONLY,
    LAYOUT_TRANSFER_TARGET,
    PIPELINE_AS_BUILD_KHR,
    PIPELINE_BOTTOM,
    PIPELINE_COLOR_ATTACHMENT_OUTPUT,
    PIPELINE_COMPUTE,
    PIPELINE_DRAW_INDIRECT,
    PIPELINE_EARLY_FRAGMENT_TESTS,
    PIPELINE_FRAGMENT_SHADER,
    PIPELINE_HOST,
    PIPELINE_LATE_FRAGMENT_TESTS,
    PIPELINE_TOP,
    PIPELINE_TRANSFER,
    PIPELINE_VERTEX_SHADER,
    SwapchainHandle,
    VulkanAccelStruct,
    VulkanGpu,
    VulkanImage,
    get32,
    get64,
    put32,
    put_float,
)
from flint.render.font import SdfFont
from flint.ui.draw_list import UiDrawList, GpuUiPrim
from flint.render.ui_overlay import aaa_ui_overlay_shader, AAA_POST_UI_PRIMS_BASE_OFFSET
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
# --- sota:geometry hook ---
from flint.render.sota_geometry_cull import SOTA_MAX_GLOBAL_MESHLETS, SotaGeometryGpu, sota_geometry_grow_size
from flint.render.sota_geometry_visid import SOTA_MAX_GLOBAL_VERTICES, sota_draw_pack
from flint.render.sota_geometry_cull import SOTA_BIND_CANDIDATES, SOTA_BIND_DAG, SOTA_BIND_MESHLETS
from flint.render.sota_geometry_vt import SotaVirtualTextures
from flint.render.sota_geometry_cluster_dag import cook_cluster_dag_into_pool
# --- end sota:geometry hook ---
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_MAX_BINDLESS,
    AAA_PUSH_BYTES,
    AAA_PUSH_WORDS,
    AAA_SINGLE_SAMPLERS,
    AAA_STORAGE_COUNT,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_BRDF_LUT,
    BIND_AAA_CANDIDATES,
    BIND_AAA_CLUSTERS,
    BIND_AAA_DEPTH_TEX,
    BIND_AAA_ENV_CUBE,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    BIND_AAA_INDICES,
    BIND_AAA_INDIRECT,
    BIND_AAA_INSTANCES,
    BIND_AAA_LTC_LUT,
    BIND_AAA_MATERIALS,
    BIND_AAA_MESHLETS,
    BIND_AAA_OIT_NODES,
    BIND_AAA_POST_SCRATCH,
    BIND_AAA_SKINNING,
    BIND_AAA_SURFCACHE,
    BIND_AAA_TLAS,
    BIND_AAA_VERTICES,
    BIND_AAA_VIS_TEX,
    BIND_AAA_VSM_TABLE,
    BIND_AAA_VSM_TEX,
    make_reverse_z_perspective,
    meshlet_cull_shader,
    visbuffer_alphamask_vertex_shader,
    visbuffer_fragment_shader,
    visbuffer_vertex_shader,
)
from flint.render.aaa_bsdf import AAA_MATERIAL_WORDS, AaaMaterial
# --- sota:E hook ---
from flint.render.sota_e_characters import AAA_FLAG_SOTA_E_CHARACTERS
from flint.render.sota_e_pipeline_cache import SotaEPsoCache, sota_e_pso_cache_open, sota_e_pso_cache_save
from flint.render.sota_e_parallel_pso import sota_e_pso_batch_begin, sota_e_pso_batch_flush
from flint.render.sota_e_gpu_driven import SotaEGpuDriven, SOTA_E_INSTANCE_OUTLINE_BIT, SOTA_E_ROUTE_MAX_DRAWS
from flint.render.sota_e_hair_shadow import SotaEHairShadow, sota_e_dom_words
# --- end sota:E hook ---
# --- sota:mat hook --- (area I: layered material model, cvar r.sota_mat.layered / FLINT_SOTA_MAT=1)
from flint.render.sota_mat_core import (
    AAA_FLAG_SOTA_MAT,
    SOTA_MAT_EXTRA_BYTES,
    SOTA_MAT_MAX_MATERIALS,
    SotaMatLayers,
    SotaMatState,
    sota_mat_clear_layers,
    sota_mat_write_layers,
)
# --- end sota:mat hook ---
from flint.render.material_graph import (
    MaterialAsset,
    SurfaceMaterialNode,
    compile_surface_material_shader,
)
from flint.render.aaa_vsm import (
    VSM_CASCADE_COUNT,
    VSM_CASCADE_PAGES_DIM,
    VSM_GLOBALS_CASCADE_FLAG_WORD,
    VSM_GLOBALS_CASCADE_1_WORD,
    VSM_GLOBALS_CASCADE_2_WORD,
    VSM_GLOBALS_CASCADE_3_WORD,
    VSM_PAGE_TEXELS,
    VSM_PHYS_ATLAS_RES,
    VSM_TABLE_WORDS,
    VsmPageManager,
    compute_vsm_cascade_matrices,
    compute_vsm_cascade_page_origins,
    make_ortho_directional_light_vp,
    vsm_alphamask_fragment_shader,
    vsm_alphamask_vertex_shader,
    vsm_cascade_from_virtual_page,
    vsm_cascade_globals_word_offset,
    vsm_cascade_quadrant_origin,
    vsm_mark_pages_shader,
    vsm_page_vertex_shader,
)
from flint.render.aaa_ibl import (
    BakedIblEnvironment,
    SkyDomeParams,
    bake_bc6h_ibl_cubemap_and_sh9,
    bake_split_sum_brdf_lut_rg16f,
)
from flint.render.aaa_raytracing import (
    AAA_GLOBALS_GLASS_SHADOW_WORD,
    AAA_GLOBALS_SHADOW_CONE_WORD,
    build_blas_triangles_ex,
    ensure_tlas_capacity_mixed_opaque,
    record_build_tlas_mixed_opaque,
    write_vk_as_instance_64b,
)
from flint.render.aaa_gi_svgf import (
    AAA_FLAG_AERIAL_PERSPECTIVE,
    AAA_FLAG_CLUSTERED_LIGHTS,
    AAA_FLAG_FURNACE_WHITE_DOME,
    AAA_FLAG_GLASS,
    AAA_FLAG_RESTIR_DI,
    AAA_FLAG_RT_REFLECTIONS,
    AAA_FLAG_RT_SHADOWS,
    AAA_FLAG_RT_SURFACE_GI,
    AAA_FLAG_VSM,
    aaa_deferred_shade_shader,
    surface_cache_update_shader,
    svgf_atrous_shader,
    svgf_temporal_reproject_shader,
)
# --- sota:gi hook --- (area B: world radiance cache + path-traced GI)
from flint.render.sota_gi_core import (
    SOTA_GI_CTRL_WORDS,
    SOTA_GI_MODE_HYBRID,
    SOTA_GI_MODE_LEGACY,
    SOTA_GI_SURFCACHE_TRIS,
    sota_gi_region_words,
    sota_svgf_scratch_words,
)
from flint.render.sota_gi_world_cache import (
    SotaGiSettings,
    sota_gi_resolve_shader,
    sota_gi_upsample_shader,
    sota_wrc_update_shader,
)
# --- end sota:gi hook ---
from flint.render.aaa_clustered_lights import (
    AAA_CLUSTER_BUFFER_WORDS,
    AAA_CLUSTER_LIGHTS_BASE,
    AAA_CLUSTER_Z,
    AAA_MAX_CLUSTERED_LIGHTS,
    AaaClusterLight,
    aaa_cluster_cull_shader,
    bake_ltc_lut_rg16f,
)
from flint.render.aaa_skinning_rt import (
    AAA_MAX_SKIN_BONES,
    AAA_SKIN_BIND_STRIDE_WORDS,
    AAA_SKIN_BIND_VERTS_BASE,
    AAA_SKIN_BUFFER_WORDS,
    AAA_SKIN_CUR_BONES_BASE,
    AAA_SKIN_CUR_MORPH_WEIGHTS_BASE,
    AAA_SKIN_CUR_SMEAR_BASE,
    AAA_SKIN_MAX_MORPH_DELTAS,
    AAA_SKIN_MAX_MORPH_SLOTS,
    AAA_SKIN_MAX_VERTS,
    AAA_SKIN_MORPH_DELTAS_BASE,
    AAA_SKIN_PREV_BONES_BASE,
    AAA_SKIN_PREV_MORPH_WEIGHTS_BASE,
    AAA_SKIN_PREV_POS_BASE,
    AAA_SKIN_PREV_SMEAR_BASE,
    BoneSmear,
    DualQuat,
    FlintAaaSkinningArena,
    aaa_dq_skinning_shader,
    normalize_and_sort_skin_weights_4,
)
from flint.render.aaa_sky_atmosphere import (
    BrunetonAtmosphereParams,
    BrunetonSkyLuts,
    bake_bruneton_sky_ibl,
)
from flint.render.aaa_oit_volumetrics import (
    AAA_MAX_OIT_PRIMS,
    AaaVolumetricPlume,
    oit_and_6way_volumetric_shader,
)
from flint.render.aaa_spec_reblur import (
    spec_reblur_atrous_shader,
    spec_reblur_hist_ping_offset,
    spec_reblur_hist_pong_offset,
    spec_reblur_raw_offset,
    spec_reblur_scratch_offset,
    spec_reblur_temporal_shader,
)
from flint.render.aaa_froxel_volumetrics import (
    AAA_FROXEL_GRID_WORDS,
    AAA_FROXEL_X,
    AAA_FROXEL_Y,
    AAA_FROXEL_Z,
    froxel_inject_and_reproject_shader,
    froxel_ping_word_offset,
    froxel_pong_word_offset,
    froxel_prefix_integrate_shader,
    froxel_prefix_lut_word_offset,
    froxel_raymarch_integrate_shader,
)
from flint.render.aaa_gpu_particles import (
    AAA_MAX_PARTICLES,
    AAA_EMITTER_WORDS,
    AAA_MAX_PARTICLE_EMITTERS,
    AaaParticleEmitter,
    gpu_particle_clear_tiles_shader,
    gpu_particle_sim_and_bin_shader,
    gpu_particle_tile_composite_shader,
    gpu_particles_emitters_word_offset,
    gpu_particles_pool_word_offset,
    gpu_particles_tile_bins_word_offset,
    gpu_particles_tile_counts_word_offset,
    gpu_particles_total_oit_words,
)
from flint.render.aaa_water import AAA_GLOBALS_WATER_SHAPE_WORD, aaa_water_refract_shader
from flint.render.aaa_visbuffer import AAA_GLOBALS_WIND_WORD, AAA_INSTANCE_WIND_SHIFT
from flint.render.aaa_glass import (
    AAA_GLASS_ENTRY_WORDS,
    AAA_GLOBALS_GLASS_COUNT_WORD,
    AAA_GLOBALS_GLASS_TABLE_WORD,
    AAA_INSTANCE_GLASS_BIT,
    AAA_MAX_GLASS_MATERIALS,
    AaaGlassParams,
    aaa_glass_refract_shader,
)
from flint.render.aaa_npr_cel import (
    AAA_CEL_ENTRY_WORDS,
    AAA_CEL_NO_OUTLINE_MAT,
    AAA_GLOBALS_CEL_COUNT_WORD,
    AAA_GLOBALS_CEL_TABLE_WORD,
    AAA_MAX_CEL_MATERIALS,
    AaaCelParams,
    aaa_npr_cel_shade_shader,
    npr_outline_fragment_shader,
    npr_outline_vertex_shader,
)
from flint.render.aaa_npr_vfx import (
    AAA_GLOBALS_IMPACT_FRAME_WORD,
    AAA_GLOBALS_IMPACT_PAPER_GAIN_WORD,
    AAA_GLOBALS_SHOCKWAVE_COUNT_WORD,
    AAA_GLOBALS_SHOCKWAVE_TABLE_WORD,
    AAA_GLOBALS_VFX_COUNT_WORD,
    AAA_GLOBALS_VFX_TABLE_WORD,
    AAA_MAX_SHOCKWAVES,
    AAA_MAX_VFX_MATERIALS,
    AAA_SHOCKWAVE_WORDS,
    AAA_VFX_ENTRY_WORDS,
    AaaImpactFrameParams,
    AaaShockwaveRing,
    AaaStylizedVfxParams,
    aaa_npr_post_fx_shader,
    visbuffer_alphamask_vfx_fragment_shader,
)
from flint.render.aaa_rt_shadow import (
    rt_shadow_filtered_offset,
    rt_shadow_raw_offset,
    rt_shadow_total_words,
    rt_sun_shadow_filter_shader,
    rt_sun_shadow_trace_shader,
)
# --- sota:ml hook ---
from flint.render.sota_ml_denoise import SotaMlDenoiser
from flint.render.sota_ml_upscale import SotaMlUpscaler
# --- end sota:ml hook ---
from flint.render.aaa_post_tsr import (
    AAA_DISPLAY_AGX_REC709,
    AAA_POST_ADAPTED_EXP_WORD,
    AAA_POST_AVG_LUMA_WORD,
    AAA_POST_BLOOM_BASE,
    AAA_POST_BLOOM_MIPS,
    AaaPhysicalCamera,
    aaa_adapt_exposure_shader,
    aaa_bloom_downsample_shader,
    aaa_bloom_upsample_shader,
    aaa_bokeh_dof_shader,
    aaa_display_tonemap_shader,
    aaa_luminance_histogram_shader,
    aaa_motion_blur_shader,
    aaa_tsr_v3_resolve_shader,
    bloom_mip_word_offset,
    dof_scratch_word_offset,
    halton_2_3_jitter_px,
    srgb_u8_to_linear,
    tsr_v2_and_display_shader,
)
# --- sota:camera hook (import) ---
from flint.render.sota_perf_vsm import SotaVsmCache  # --- sota:perf hook ---
from flint.render.sota_shadow_vsm import (  # --- sota:shadow hook ---
    SOTA_SHADOW_GLOBALS_CONTACT_WORD,
    SOTA_SHADOW_GLOBALS_TOROIDAL_OFFSETS_WORD,
    SOTA_SHADOW_GLOBALS_TOROIDAL_WORD,
    SotaShadowState,
    sota_shadow_depth_snap_index,
    sota_shadow_local_from_slot,
    sota_shadow_vsm_cmd_target,
)
from flint.render.sota_shadow_gpu import (  # --- sota:shadow hook ---
    SOTA_SHADOW_ITEM_MESHLETS,
    SotaShadowGpu,
    sota_shadow_reset_tables,
)
from flint.core.jobs import JobPool, default_worker_count  # --- sota:perf hook ---
from flint.render.sota_geometry_cluster_dag import ClusterDag  # --- sota:perf hook ---
from flint.render.sota_geometry_cull import SOTA_BIND_INDIRECT  # --- sota:perf hook ---
from flint.render.sota_perf_async import sota_pipelined_default, sota_route_capacity_for, sota_route_draws_limit  # --- sota:perf hook ---
from flint.render.sota_perf_upload import (  # --- sota:perf hook --- upload profile + parallel prepass
    SOTA_PREP_ALBEDO,
    SOTA_PREP_NORMAL,
    SOTA_PREP_ORM,
    SotaPreparedModel,
    SotaUploadStats,
    sota_now,
    sota_append_mesh_prepared,
    sota_prepare_model,
    sota_upload_parallel_enabled,
    sota_upload_worker_count,
)
from flint.render.sota_camera_post import (
    SotaCameraPost,
    sota_camera_record_ca,
    sota_camera_record_dof,
    sota_camera_record_lens,
    sota_camera_record_vrs_classify,
    sota_camera_record_vrs_fill,
    sota_camera_record_vrs_rate,
    sota_camera_vrs_begin_frame,
)
from flint.render.sota_camera_hdr import (  # --- sota:camera hook ---
    SOTA_DISPLAY_AGX_HDR,
    sota_hdr_record_scrgb,
    sota_hdr_tonemap_mode,
    sota_hdr_tonemap_push,
    sota_hdr_ui_mode,
    sota_hdr_ui_paper_push,
    sota_hdr_update_metadata,
)
from flint.render.sota_camera_exposure import sota_ae_record  # --- sota:camera hook (round 4) ---
from flint.render.sota_atmo_core import rgb_luma  # --- sota:camera hook (round 4: sky calibration) ---
# --- end sota:camera hook ---
# --- sota:atmo hook (import) ---
from flint.render.sota_atmo_gpu import SotaAtmosphere, sota_atmo_record_apply, sota_atmo_record_luts, sota_atmo_region_words
# --- end sota:atmo hook ---
from flint.render.aaa_render_graph import (
    AaaGpuPassTiming,
    AaaRenderGraph,
    AaaResidencyStreamer,
    RG_QUEUE_ASYNC_COMPUTE,
    RG_QUEUE_GRAPHICS,
    RG_USAGE_COLOR_ATT_WRITE,
    RG_USAGE_COMPUTE_READ,
    RG_USAGE_COMPUTE_WRITE,
    RG_USAGE_DEPTH_ATT_WRITE,
    RG_USAGE_INDIRECT_ARGS,
    RgResourceAccess,
)


comptime AAA_DISPLAY_SCRGB_FP16: UInt32 = 3  # Extended Linear scRGB (`VK_FORMAT_R16G16B16A16_SFLOAT`, `1.0 = 80 nits`, `8` bytes/px)


def aaa_pack_scrgb_rgba16f_shader() -> List[UInt32]:
    """Packs resolved linear Rec.709 HDR (`BIND_AAA_HDR_OUT`) into `8`-byte/pixel `R16G16B16A16_SFLOAT` (`RGBA16F`)
    Extended Linear scRGB (`1.0 = 80 nits`, `paper_white_nits / 80.0 = 2.5375`) inside `BIND_AAA_POST_SCRATCH`.
    Push[0]=out_w, Push[1]=out_h, Push[2]=out_scrgb_word_off, Push[3]=scrgb_scale_f32.
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)

    var dx = s.invocation_axis(0)
    var dy = s.invocation_axis(1)
    var dst_w = s.push_word(0)
    var dst_h = s.push_word(1)
    var out_off = s.push_word(2)
    var sc_scale = s.push_float(3)
    s.return_if(s.logical_or(s.uge(dx, dst_w), s.uge(dy, dst_h)))

    var dst_pix = s.iadd(s.imul(dy, dst_w), dx)
    var h_base = s.imul(dst_pix, s.uint(8))
    var r_lin = s.fmul(s.load_float(b_hdr, h_base), sc_scale)
    var g_lin = s.fmul(s.load_float(b_hdr, s.iadd(h_base, s.uint(1))), sc_scale)
    var b_lin = s.fmul(s.load_float(b_hdr, s.iadd(h_base, s.uint(2))), sc_scale)

    var w0 = s.pack_f16x2(r_lin, g_lin)
    var w1 = s.pack_f16x2(b_lin, s.float(1.0))
    var dst_word = s.iadd(out_off, s.imul(dst_pix, s.uint(2)))
    s.store_word(b_post, dst_word, w0)
    s.store_word(b_post, s.iadd(dst_word, s.uint(1)), w1)
    return s.finish()


struct AaaDrawInstance(Copyable, Movable):
    var mesh_id: Int
    var world_mat: Mat4
    var prev_world_mat: Mat4
    var material_override: Int
    var uniform_scale: Float32
    var wind: Float32             # 0..1: how much the global wind sways this draw (see set_wind)

    def __init__(
        out self,
        mesh_id: Int,
        world_mat: Mat4,
        prev_world_mat: Mat4 = Mat4(),
        material_override: Int = -1,
        uniform_scale: Float32 = 1.0,
        has_prev: Bool = False,
        wind: Float32 = 0.0,
    ):
        self.mesh_id = mesh_id
        self.world_mat = world_mat.copy()
        self.prev_world_mat = prev_world_mat.copy() if has_prev else world_mat.copy()
        self.material_override = material_override
        self.uniform_scale = max(Float32(0.001), uniform_scale)
        self.wind = min(Float32(1.0), max(Float32(0.0), wind))


def _mat4_mul_col_major(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = List[Float32](length=16, fill=0.0)
    for col in range(4):
        for row in range(4):
            var s = Float32(0.0)
            for k in range(4):
                s += a[k * 4 + row] * b[col * 4 + k]
            out[col * 4 + row] = s
    return out^


def _build_camera_view_col_major(eye: Vec3, fwd: Vec3, right: Vec3, up: Vec3) -> List[Float32]:
    """Left-handed camera view matrix (`+X = right, +Y = up, +Z = fwd`)."""
    var m = List[Float32](length=16, fill=0.0)
    m[0] = right.x; m[4] = right.y; m[8] = right.z;  m[12] = -right.dot(eye)
    m[1] = up.x;    m[5] = up.y;    m[9] = up.z;     m[13] = -up.dot(eye)
    m[2] = fwd.x;   m[6] = fwd.y;   m[10] = fwd.z;   m[14] = -fwd.dot(eye)
    m[3] = 0.0;     m[7] = 0.0;     m[11] = 0.0;     m[15] = Float32(1.0)
    return m^


struct AaaSkinnedMeshEntry(Copyable, Movable):
    var mesh_entry: CookedMeshEntry
    var skin_bind_first_slot: Int
    var vertex_count: Int
    var bone_count: Int
    var bone_palette_offset: Int
    var morph_count: Int
    var morph_weight_first_slot: Int
    var morph_delta_first_slot: Int

    def __init__(out self):
        self.mesh_entry = CookedMeshEntry()
        self.skin_bind_first_slot = 0
        self.vertex_count = 0
        self.bone_count = 0
        self.bone_palette_offset = 0
        self.morph_count = 0
        self.morph_weight_first_slot = 0
        self.morph_delta_first_slot = 0


struct AaaUploadedModel(Movable):
    """Turnkey GPU residency handle returned by `upload_native_model(model)` (`V19`)."""
    var mesh_entries: List[CookedMeshEntry]
    var skinned_entries: List[AaaSkinnedMeshEntry]
    var material_ids: List[Int]
    var albedo_bindless_slots: List[Int]
    var normal_bindless_slots: List[Int]
    var orm_bindless_slots: List[Int]

    def __init__(out self):
        self.mesh_entries = List[CookedMeshEntry]()
        self.skinned_entries = List[AaaSkinnedMeshEntry]()
        self.material_ids = List[Int]()
        self.albedo_bindless_slots = List[Int]()
        self.normal_bindless_slots = List[Int]()
        self.orm_bindless_slots = List[Int]()


comptime UploadedModelSlots = AaaUploadedModel


comptime FRAMES_IN_FLIGHT: Int = 2
comptime AAA_UI_PRIM_REGION_WORDS: Int = 32768 * 16  # 32768 GpuUiPrim (64 bytes = 2 MiB) per frame in flight
# Per-slot timestamp query range: 64 per-pass queries + 1 whole-frame TOP/BOTTOM pair.
comptime AAA_QUERY_SLOT_STRIDE: Int = 66
comptime AAA_QUERY_FRAME_PAIR: Int = 64
comptime _AAA_STAGE_ALL_COMMANDS: Int = 0x10000
comptime _AAA_ACCESS_MEMORY_READ_WRITE: Int = 0x18000


struct FlintAaaVulkanRenderer(Movable):
    """100% Pure-Mojo Vulkan 1.3/1.4 Hardware AAA Renderer (`V12 -> V15`)."""
    var vk: VulkanGpu
    var active: Bool
    var output_w: Int
    var output_h: Int
    var render_w: Int
    var render_h: Int
    var tsr_scale: Float32
    var max_draws: Int
    var max_instances: Int
    var max_surfcache_tris: Int

    # Ring-Buffered CPU/GPU Multi-Buffering (Feature 14)
    var ring_globals: List[GpuBuffer]
    var ring_instances: List[GpuBuffer]
    var ring_as_instances: List[GpuBuffer]
    var ring_indirect: List[GpuBuffer]
    var ring_alphamask_indirect: List[GpuBuffer]
    var ring_outline_indirect: List[GpuBuffer]
    var ring_clusters: List[GpuBuffer]
    var ring_tlas: List[VulkanAccelStruct]
    var ring_tlas_cap: List[Int]  # instance capacity each slot's TLAS + shared scratch were sized for
    var desc_sets: List[Int]
    var desc_sets_cull_p1: List[Int]
    var desc_sets_cull_p2: List[Int]
    var frame_fences: List[UInt64]

    # Frame-time instrumentation (refreshed by every `render_frame`; read-only for callers)
    var last_cpu_prev_wait_ms: Float64  # deferred mode: wait on the previous frame's fence
    var last_cpu_pack_ms: Float64  # instance / globals / cluster / TLAS-capacity packing
    var last_cpu_vsm_ms: Float64  # VSM request processing + indirect page packing
    var last_cpu_record_ms: Float64  # command-buffer recording
    var last_cpu_submit_wait_ms: Float64  # vkQueueSubmit (+ fence wait in synchronous mode)
    var geometry_mb: Int  # --- sota:perf hook ---
    var sota_vsm: SotaVsmCache  # --- sota:perf hook --- VSM page invalidation for light / caster motion
    var sota_shadow: SotaShadowState  # --- sota:shadow hook --- Area G (VSM command sizing, toroidal pages, contact shadows)
    var sota_shadow_gpu: SotaShadowGpu  # --- sota:shadow hook --- GPU page allocation / cull / raster
    var geometry_overflow_meshes: Int  # --- sota:perf hook --- meshes rejected because the pools were full
    var sota_upload: SotaUploadStats  # --- sota:perf hook --- per-phase upload wall time
    var sota_prep: SotaPreparedModel  # --- sota:perf hook --- parallel upload prepass results
    var sota_upload_pool: JobPool  # --- sota:perf hook ---
    var sota_upload_pool_started: Bool  # --- sota:perf hook ---
    var sota_route_draws: Int  # --- sota:perf hook --- capacity of the alpha-mask / outline indirect rings (was a fixed 16384)
    var sota_route_cpu_need: Int  # --- sota:perf hook --- draws the CPU packers wanted last frame (beyond capacity = dropped)
    var sota_route_grow_events: Int  # --- sota:perf hook ---
    var last_frame_gpu_ms: Float64  # whole-frame TOP->BOTTOM timestamp pair (incl. TLAS, skin, readback copy)
    var last_vsm_dirty_pages: Int
    var last_cpu_vsm_table_ms: Float64  # part of `last_cpu_vsm_ms`: VSM table alloc + host read/write copies
    var last_cpu_vsm_requests_ms: Float64  # part of `last_cpu_vsm_ms`: `vsm.process_requests`

    # Opt-in deferred readback (default False = unchanged synchronous behaviour). When True and no swapchain
    # is attached, `render_frame()` submits without waiting; the next `render_frame()` waits on that fence,
    # so `readback_pixels_into()` returns the previous frame. `capture_readback=True` stays synchronous.
    var deferred_readback: Bool
    var force_readback: Bool
    var enable_vsm_clipmap_cascades: Bool
    var vsm_cascade_matrices: List[List[Float32]]
    var vsm_cascade_page_origins: List[Tuple[Int, Int]]
    var readback_extra: List[GpuBuffer]  # per-slot readback buffers for slots >= 1 (slot 0 = `buf_ldr_readback`)
    var readback_ready_slot: Int  # slot whose readback buffer holds the latest completed frame
    var pending_slot: Int  # slot of a submitted-but-not-yet-waited deferred frame (-1 = none)
    var pending_prof_pairs: List[Tuple[String, Int]]
    var pending_frame_q: Int
    var pending_n_inst: Int

    # UI Overlay Pass (Feature 15)
    var pipe_ui_overlay: Int
    var pipe_scrgb_pack: Int
    var ui_draw_list: UiDrawList
    var ui_font: SdfFont
    var ui_font_img: VulkanImage
    var ui_font_atlas_slot: Int

    # Descriptor & Pipeline handles
    var dsl: Int
    var pipe_layout: Int
    var desc_pool: Int
    var desc_set: Int
    var pipe_meshlet_cull: Int
    var pipe_visbuffer_gfx: Int
    var pipe_visbuffer_alphamask_gfx: Int
    var pipe_vsm_mark: Int
    var pipe_vsm_gfx: Int
    var pipe_vsm_alphamask_gfx: Int
    var pipe_surfcache: Int
    var pipe_cluster_cull: Int
    var pipe_dq_skinning: Int
    var pipe_deferred_shade: Int
    var pipe_svgf_reproject: Int
    var pipe_svgf_atrous: Int
    var pipe_oit_vol: Int
    var pipe_luma_hist: Int
    var pipe_adapt_exp: Int
    var pipe_bloom_down: Int
    var pipe_bloom_up: Int
    var pipe_bokeh_dof: Int
    var pipe_tsr_display: Int
    var pipe_spec_reblur_temp: Int
    var pipe_spec_reblur_atrous: Int
    var pipe_froxel_inject: Int
    var pipe_froxel_prefix: Int
    var pipe_froxel_march: Int
    var pipe_particle_clear: Int
    var pipe_particle_sim: Int
    var pipe_particle_comp: Int
    var pipe_water_refract: Int
    var pipe_rt_shadow_trace: Int
    var pipe_rt_shadow_filter: Int
    var pipe_glass_refract: Int
    var pipe_tsr_v3_resolve: Int
    var pipe_motion_blur: Int
    var pipe_display_tonemap: Int
    var pipe_npr_outline_gfx: Int
    var pipe_npr_cel_shade: Int
    var pipe_npr_post_fx: Int

    # Samplers & Images
    var samp_aniso16: Int
    var samp_linear_clamp: Int
    var samp_nearest: Int
    var vis_img: VulkanImage
    var depth_img: VulkanImage
    var vsm_img: VulkanImage
    var brdf_lut_img: VulkanImage
    var ltc_lut_img: VulkanImage
    var env_cube_img: VulkanImage
    var bindless_images: List[VulkanImage]

    # Storage & Staging Buffers
    var buf_vertices: GpuBuffer
    var buf_indices: GpuBuffer
    var buf_meshlets: GpuBuffer
    var buf_instances: GpuBuffer
    var buf_materials: GpuBuffer
    var buf_indirect: GpuBuffer
    var buf_globals: GpuBuffer
    var buf_hdr_out: GpuBuffer
    var buf_history: GpuBuffer
    var buf_vsm_table: GpuBuffer
    var buf_surfcache: GpuBuffer
    var buf_oit_nodes: GpuBuffer
    var buf_clusters: GpuBuffer
    var buf_skinning: GpuBuffer
    var buf_post_scratch: GpuBuffer
    var buf_staging: GpuBuffer

    # Ray-Tracing Acceleration Structures
    var mesh_blas: List[VulkanAccelStruct]
    var frame_tlas: VulkanAccelStruct
    var buf_as_instances: GpuBuffer
    var buf_as_scratch: GpuBuffer
    var buf_vsm_indirect: GpuBuffer
    var buf_ldr_readback: GpuBuffer
    var buf_indirect_phase2: GpuBuffer
    var buf_alphamask_indirect: GpuBuffer
    var buf_outline_indirect: GpuBuffer
    var buf_occluded_candidates: GpuBuffer
    var pending_skin_updates: List[AaaSkinnedMeshEntry]

    # CPU Scene & Subsystems
    var geom: AaaGeometryPool
    var materials: List[AaaMaterial]
    var glass_materials: List[AaaGlassParams]  # optics of the transparent materials (`aaa_glass.mojo`)
    var cel_materials: List[AaaCelParams]
    var vfx_materials: List[AaaStylizedVfxParams]
    var frame_shockwaves: List[AaaShockwaveRing]
    var impact_frame: AaaImpactFrameParams
    var frame_instances: List[AaaDrawInstance]
    var frame_plumes: List[AaaVolumetricPlume]
    var frame_particle_emitters: List[AaaParticleEmitter]
    var particle_ring_cursor: Int
    var particles_ever_active: Bool
    var particle_dt_sec: Float32  # GPU particle simulation step per frame (set it to the game's frame time)
    var frame_cluster_lights: List[AaaClusterLight]
    var cluster_far_z: Float32
    var skinned_mesh_flags: List[Bool]
    var skinned_mesh_bind_slots: List[Int]
    var mesh_double_sided_flags: List[Bool]
    var mesh_alpha_masked_flags: List[Bool]
    var skin_bind_cursor: Int
    var skin_bone_cursor: Int
    var skin_morph_weight_cursor: Int
    var skin_morph_delta_cursor: Int
    var skin_arena: FlintAaaSkinningArena
    var enable_aerial_perspective: Bool
    var aerial_sigma_ext: Vec3
    var aerial_r_term: Vec3
    var aerial_m_term: Vec3
    var aerial_mie_g: Float32
    var vsm: VsmPageManager
    var render_graph: AaaRenderGraph
    var streamer: AaaResidencyStreamer
    var sh9_irradiance: List[Vec3]
    var env_cube_dim: Int
    var env_mip_levels: Int

    # Camera, Light & Frame State
    var cam_pos: Vec3
    var cam_fwd: Vec3
    var cam_right: Vec3
    var cam_up: Vec3
    var fov_y_deg: Float32
    var near_z: Float32
    var prev_view_proj: List[Float32]
    var has_prev_vp: Bool
    var sun_dir: Vec3
    var sun_radiance: Vec3
    var sun_angular_radius: Float32
    var sun_shadow_softness: Float32  # angular radius (rad) of the RT sun-shadow cone; < 0 = use `sun_angular_radius`
    var vsm_scene_center: Vec3
    var vsm_ortho_half_extent: Float32
    var ibl_intensity: Float32
    var exposure: Float32
    var enable_cone_cull: Bool
    var enable_hiz_cull: Bool
    var enable_vsm: Bool
    # --- sota:E hook ---
    var sota_e_character_shading: Bool  # skin / eye / hair material models in the deferred pass
    var sota_e_pso: SotaEPsoCache  # validated on-disk pipeline cache (`FLINT_PSO_CACHE=0` disables)
    var sota_e_gpu: SotaEGpuDriven  # GPU-packed alpha-mask / outline draw lists (cvar r.sota_e.gpu_driven)
    var sota_e_hair: SotaEHairShadow  # strand-hair deep opacity shadow map (cvar r.sota_e.hair_dom)
    # --- end sota:E hook ---
    var sota_mat: SotaMatState  # --- sota:mat hook --- layered material model (cvar r.sota_mat.layered, default off)
    var enable_rt_shadows: Bool
    var enable_restir_di: Bool
    var enable_rt_reflections: Bool
    var enable_rt_gi: Bool
    var furnace_white_dome: Bool
    var gi_rays_per_pixel: Int
    var svgf_passes: Int
    var svgf_sigma_luma: Float32
    var display_mode: UInt32
    var tsr_history_weight: Float32
    var tsr_sharpness: Float32
    var max_lod_error_px: Float32
    var wind_dir_x: Float32       # the global wind (set_wind): unit direction in xz,
    var wind_dir_z: Float32
    var wind_sway: Float32        # how far a fully swaying tip moves (m),
    var wind_hz: Float32          # how fast (Hz),
    var wind_gust: Float32        # and how gusty (0..1)
    var wind_t0: Float64
    var camera_lens: AaaPhysicalCamera
    # --- sota:camera hook (field) ---
    var sota_cam: SotaCameraPost  # physical camera, scatter-as-gather DoF, lens flare/dirt/CA/vignette, sensor noise
    # --- end sota:camera hook ---
    var sota_atmo: SotaAtmosphere  # --- sota:atmo hook --- physical sky / aerial perspective / height fog / time of day
    var enable_auto_exposure: Bool
    var auto_exposure_speed: Float32
    var auto_exposure_key: Float32
    var ev_compensation: Float32
    var bloom_intensity: Float32
    var bloom_threshold: Float32
    var grade_saturation: Float32  # display grade after tone mapping: 1.0 = unchanged, >1 more vivid
    var grade_contrast: Float32  # 1.0 = unchanged, >1 punchier around mid-grey
    var enable_bokeh_dof: Bool
    var enable_gpu_profiler: Bool
    var enable_async_vsm: Bool
    var vsm_warmed_up: Bool
    var gi_bounces: Int
    var sota_gi: SotaGiSettings  # --- sota:gi hook --- area-B GI cvars (`mode = 0` keeps the v5 path)
    var sota_gi_region_ready: Bool  # --- sota:gi hook --- surface-cache buffer grown for the area-B region
    var frame_index: Int
    var last_culled_meshlet_draws: Int
    # --- sota:geometry hook ---
    var sota_geom: SotaGeometryGpu
    var sota_vt: SotaVirtualTextures
    # --- end sota:geometry hook ---
    var prev_cam_pos: Vec3
    var has_prev_cam: Bool
    var scene_time_sec: Float32
    var enable_spec_reblur: Bool
    # --- sota:ml hook ---
    var sota_ml: SotaMlDenoiser  # learned GI denoiser; opt-in (`enabled`), replaces SVGF when weights load (`sota_ml_denoise.mojo`)
    var sota_ml_up: SotaMlUpscaler  # learned TSR resolve; opt-in (`enabled`), TSR v3 fallback (`sota_ml_upscale.mojo`)
    # --- end sota:ml hook ---
    var spec_reblur_passes: Int
    var spec_reblur_valid_frames: Int
    var enable_froxel_fog: Bool
    var froxel_valid_frames: Int
    var fog_density: Float32
    var fog_height_falloff: Float32
    var fog_anisotropy_g: Float32
    var fog_base_height_y: Float32
    var fog_albedo: Vec3
    var froxel_far_z: Float32
    var enable_gpu_rain: Bool
    var rain_intensity: Float32
    var weather_wetness: Float32
    var windshield_rain: Float32
    var wind_velocity: Vec3
    var enable_water_plane: Bool
    var water_level_y: Float32
    var water_absorption_rgb: Vec3
    var water_bounds_min_xz: Tuple[Float32, Float32]
    var water_bounds_max_xz: Tuple[Float32, Float32]
    var water_elliptical: Bool  # True: the water is the ellipse inscribed in its bounds (a round pond)
    var water_wave_scale: Float32  # wave amplitude (1 = open-water swell, ~0.3 = calm pond, 0 = mirror)
    var water_foam_band: Float32  # depth (m) below which shoreline foam forms
    var water_foam: Float32  # foam amount (0 = none, 1 = default)
    var enable_motion_blur: Bool
    var motion_blur_shutter: Float32
    var motion_blur_max_px: Float32
    var glare_streak_strength: Float32
    var min_auto_exposure: Float32
    var max_auto_exposure: Float32
    var cloud_coverage: Float32
    var cloud_wind_offset: Vec3
    var star_intensity: Float32
    var sky_zenith_clear: Vec3
    var sky_horizon_clear: Vec3

    def __init__(
        out self,
        width: Int = 640,
        height: Int = 360,
        tsr_scale: Float32 = 1.0,
        max_draws: Int = 131072,
        max_instances: Int = 32768,
        env_cube_dim: Int = 64,
        geometry_mb: Int = 32,  # --- sota:perf hook --- vertex + index pools (MB each); meshlets get half
    ):
        self.vk = VulkanGpu()
        self.geometry_mb = max(8, geometry_mb)  # --- sota:perf hook ---
        self.geometry_overflow_meshes = 0  # --- sota:perf hook ---
        self.sota_vsm = SotaVsmCache()  # --- sota:perf hook ---
        self.sota_shadow = SotaShadowState()  # --- sota:shadow hook ---
        self.sota_shadow_gpu = SotaShadowGpu()  # --- sota:shadow hook ---
        self.sota_upload = SotaUploadStats()  # --- sota:perf hook ---
        self.sota_prep = SotaPreparedModel()  # --- sota:perf hook ---
        self.sota_upload_pool = JobPool()  # --- sota:perf hook ---
        self.sota_upload_pool_started = False  # --- sota:perf hook ---
        self.sota_route_draws = SOTA_E_ROUTE_MAX_DRAWS  # --- sota:perf hook ---
        self.sota_route_cpu_need = 0  # --- sota:perf hook ---
        self.sota_route_grow_events = 0  # --- sota:perf hook ---
        self.active = False
        self.output_w = max(16, width)
        self.output_h = max(16, height)
        self.tsr_scale = max(Float32(0.50), min(Float32(1.0), tsr_scale))
        self.render_w = max(8, Int(Float32(self.output_w) * self.tsr_scale + Float32(0.5)))
        self.render_h = max(8, Int(Float32(self.output_h) * self.tsr_scale + Float32(0.5)))
        self.max_draws = max(256, max_draws)
        self.max_instances = min(32768, max(256, max_instances))
        self.max_surfcache_tris = 262144

        self.dsl = 0
        self.pipe_layout = 0
        self.desc_pool = 0
        self.desc_set = 0
        self.pipe_meshlet_cull = 0
        self.pipe_visbuffer_gfx = 0
        self.pipe_visbuffer_alphamask_gfx = 0
        self.pipe_vsm_mark = 0
        self.pipe_vsm_gfx = 0
        self.pipe_vsm_alphamask_gfx = 0
        self.pipe_surfcache = 0
        self.pipe_cluster_cull = 0
        self.pipe_dq_skinning = 0
        self.pipe_deferred_shade = 0
        self.pipe_svgf_reproject = 0
        self.pipe_svgf_atrous = 0
        self.pipe_oit_vol = 0
        self.pipe_luma_hist = 0
        self.pipe_adapt_exp = 0
        self.pipe_bloom_down = 0
        self.pipe_bloom_up = 0
        self.pipe_bokeh_dof = 0
        self.pipe_tsr_display = 0
        self.pipe_spec_reblur_temp = 0
        self.pipe_spec_reblur_atrous = 0
        self.pipe_froxel_inject = 0
        self.pipe_froxel_prefix = 0
        self.pipe_froxel_march = 0
        self.pipe_particle_clear = 0
        self.pipe_particle_sim = 0
        self.pipe_particle_comp = 0
        self.pipe_water_refract = 0
        self.pipe_rt_shadow_trace = 0
        self.pipe_rt_shadow_filter = 0
        self.pipe_glass_refract = 0
        self.pipe_tsr_v3_resolve = 0
        self.pipe_motion_blur = 0
        self.pipe_display_tonemap = 0
        self.pipe_npr_outline_gfx = 0
        self.pipe_npr_cel_shade = 0
        self.pipe_npr_post_fx = 0

        self.samp_aniso16 = 0
        self.samp_linear_clamp = 0
        self.samp_nearest = 0
        self.vis_img = VulkanImage()
        self.depth_img = VulkanImage()
        self.vsm_img = VulkanImage()
        self.brdf_lut_img = VulkanImage()
        self.ltc_lut_img = VulkanImage()
        self.env_cube_img = VulkanImage()
        self.bindless_images = List[VulkanImage]()

        self.buf_vertices = GpuBuffer()
        self.buf_indices = GpuBuffer()
        self.buf_meshlets = GpuBuffer()
        self.buf_instances = GpuBuffer()
        self.buf_materials = GpuBuffer()
        self.buf_indirect = GpuBuffer()
        self.buf_globals = GpuBuffer()
        self.buf_hdr_out = GpuBuffer()
        self.buf_history = GpuBuffer()
        self.buf_vsm_table = GpuBuffer()
        self.buf_surfcache = GpuBuffer()
        self.buf_oit_nodes = GpuBuffer()
        self.buf_clusters = GpuBuffer()
        self.buf_skinning = GpuBuffer()
        self.buf_post_scratch = GpuBuffer()
        self.buf_staging = GpuBuffer()
        self.buf_indirect_phase2 = GpuBuffer()
        self.buf_alphamask_indirect = GpuBuffer()
        self.buf_outline_indirect = GpuBuffer()
        self.buf_occluded_candidates = GpuBuffer()

        self.ring_globals = List[GpuBuffer]()
        self.ring_instances = List[GpuBuffer]()
        self.ring_as_instances = List[GpuBuffer]()
        self.ring_indirect = List[GpuBuffer]()
        self.ring_alphamask_indirect = List[GpuBuffer]()
        self.ring_outline_indirect = List[GpuBuffer]()
        self.ring_clusters = List[GpuBuffer]()
        self.ring_tlas = List[VulkanAccelStruct]()
        self.ring_tlas_cap = List[Int]()
        self.desc_sets = List[Int]()
        self.desc_sets_cull_p1 = List[Int]()
        self.desc_sets_cull_p2 = List[Int]()
        self.frame_fences = List[UInt64]()

        self.last_cpu_prev_wait_ms = 0.0
        self.last_cpu_pack_ms = 0.0
        self.last_cpu_vsm_ms = 0.0
        self.last_cpu_record_ms = 0.0
        self.last_cpu_submit_wait_ms = 0.0
        self.last_frame_gpu_ms = 0.0
        self.last_vsm_dirty_pages = 0
        self.last_cpu_vsm_table_ms = 0.0
        self.last_cpu_vsm_requests_ms = 0.0
        self.deferred_readback = sota_pipelined_default()  # --- sota:perf hook --- cvar r.sota_perf.pipelined (default off)
        self.force_readback = False
        self.enable_vsm_clipmap_cascades = True
        self.vsm_cascade_matrices = List[List[Float32]]()
        self.vsm_cascade_page_origins = List[Tuple[Int, Int]]()
        self.readback_extra = List[GpuBuffer]()
        self.readback_ready_slot = 0
        self.pending_slot = -1
        self.pending_prof_pairs = List[Tuple[String, Int]]()
        self.pending_frame_q = -1
        self.pending_n_inst = 0

        self.pipe_ui_overlay = 0
        self.pipe_scrgb_pack = 0
        self.ui_draw_list = UiDrawList()
        self.ui_font = SdfFont()
        self.ui_font_img = VulkanImage()
        self.ui_font_atlas_slot = -1

        self.mesh_blas = List[VulkanAccelStruct]()
        self.frame_tlas = VulkanAccelStruct()
        self.buf_as_instances = GpuBuffer()
        self.buf_as_scratch = GpuBuffer()
        self.buf_vsm_indirect = GpuBuffer()
        self.buf_ldr_readback = GpuBuffer()
        self.pending_skin_updates = List[AaaSkinnedMeshEntry]()

        self.geom = AaaGeometryPool()
        self.materials = List[AaaMaterial]()
        self.frame_instances = List[AaaDrawInstance]()
        self.frame_plumes = List[AaaVolumetricPlume]()
        self.frame_particle_emitters = List[AaaParticleEmitter]()
        self.particle_ring_cursor = 0
        self.particles_ever_active = False
        self.particle_dt_sec = Float32(1.0 / 60.0)
        self.frame_cluster_lights = List[AaaClusterLight]()
        self.cluster_far_z = Float32(250.0)
        self.skinned_mesh_flags = List[Bool]()
        self.skinned_mesh_bind_slots = List[Int]()
        self.mesh_double_sided_flags = List[Bool]()
        self.mesh_alpha_masked_flags = List[Bool]()
        self.skin_bind_cursor = 0
        self.skin_bone_cursor = 0
        self.skin_morph_weight_cursor = 0
        self.skin_morph_delta_cursor = 0
        self.skin_arena = FlintAaaSkinningArena()
        self.enable_aerial_perspective = False
        self.aerial_sigma_ext = Vec3()
        self.aerial_r_term = Vec3()
        self.aerial_m_term = Vec3()
        self.aerial_mie_g = Float32(0.80)
        self.vsm = VsmPageManager()
        self.render_graph = AaaRenderGraph()
        self.streamer = AaaResidencyStreamer()
        self.sh9_irradiance = List[Vec3](length=9, fill=Vec3(0.0, 0.0, 0.0))
        self.env_cube_dim = max(16, env_cube_dim)
        var m_cnt = 1
        var tmp_d = self.env_cube_dim
        while tmp_d > 4 and m_cnt < 7:
            tmp_d >>= 1
            m_cnt += 1
        self.env_mip_levels = m_cnt

        self.cam_pos = Vec3(0.0, 1.5, -5.0)
        self.cam_fwd = Vec3(0.0, 0.0, 1.0)
        self.cam_right = Vec3(1.0, 0.0, 0.0)
        self.cam_up = Vec3(0.0, 1.0, 0.0)
        self.fov_y_deg = Float32(55.0)
        self.near_z = Float32(0.10)
        self.prev_view_proj = List[Float32](length=16, fill=0.0)
        self.has_prev_vp = False
        self.sun_dir = Vec3(0.38, 0.80, -0.46).normalized()
        self.sun_radiance = Vec3(3.2, 3.0, 2.75)
        self.sun_angular_radius = Float32(0.0035)
        self.sun_shadow_softness = Float32(0.005)
        self.glass_materials = List[AaaGlassParams]()
        self.cel_materials = List[AaaCelParams]()
        self.vfx_materials = List[AaaStylizedVfxParams]()
        self.frame_shockwaves = List[AaaShockwaveRing]()
        self.impact_frame = AaaImpactFrameParams()
        self.vsm_scene_center = Vec3(0.0, 0.0, 0.0)
        self.vsm_ortho_half_extent = Float32(16.0)
        self.ibl_intensity = Float32(1.0)
        self.exposure = Float32(1.0)
        self.enable_cone_cull = True
        self.enable_hiz_cull = True
        self.enable_vsm = True
        # --- sota:E hook ---
        self.sota_e_character_shading = True
        self.sota_e_pso = SotaEPsoCache()
        self.sota_e_gpu = SotaEGpuDriven()
        self.sota_e_hair = SotaEHairShadow()
        # --- end sota:E hook ---
        self.sota_mat = SotaMatState()  # --- sota:mat hook ---
        self.enable_rt_shadows = False
        self.enable_restir_di = False
        self.enable_rt_reflections = False
        self.enable_rt_gi = False
        self.furnace_white_dome = False
        self.gi_rays_per_pixel = 12
        self.svgf_passes = 2
        self.svgf_sigma_luma = Float32(0.65)
        self.display_mode = AAA_DISPLAY_AGX_REC709
        self.tsr_history_weight = Float32(0.82)
        self.tsr_sharpness = Float32(0.22)
        self.max_lod_error_px = Float32(1.0)
        self.wind_dir_x = Float32(1.0)
        self.wind_dir_z = Float32(0.0)
        self.wind_sway = Float32(0.0)
        self.wind_hz = Float32(0.6)
        self.wind_gust = Float32(0.3)
        self.wind_t0 = get_time()
        self.camera_lens = AaaPhysicalCamera()
        # --- sota:camera hook (init) ---
        self.sota_cam = SotaCameraPost()
        # --- end sota:camera hook ---
        self.sota_atmo = SotaAtmosphere()  # --- sota:atmo hook ---
        self.enable_auto_exposure = False
        self.auto_exposure_speed = Float32(0.45)
        self.auto_exposure_key = Float32(0.18)
        self.ev_compensation = Float32(0.0)
        self.bloom_intensity = Float32(0.0)
        self.bloom_threshold = Float32(0.0)
        self.grade_saturation = Float32(1.0)
        self.grade_contrast = Float32(1.0)
        self.enable_bokeh_dof = False
        self.enable_gpu_profiler = True
        self.enable_async_vsm = False
        self.vsm_warmed_up = False
        self.gi_bounces = 2
        self.sota_gi = SotaGiSettings()  # --- sota:gi hook ---
        self.sota_gi_region_ready = False  # --- sota:gi hook ---
        self.frame_index = 0
        self.last_culled_meshlet_draws = 0
        # --- sota:geometry hook ---
        self.sota_geom = SotaGeometryGpu()
        self.sota_vt = SotaVirtualTextures()
        # --- end sota:geometry hook ---
        self.prev_cam_pos = self.cam_pos
        self.has_prev_cam = False
        self.scene_time_sec = Float32(0.0)
        self.enable_spec_reblur = True
        self.spec_reblur_passes = 3
        self.spec_reblur_valid_frames = 0
        self.enable_froxel_fog = False
        self.froxel_valid_frames = 0
        self.fog_density = Float32(0.008)
        self.fog_height_falloff = Float32(0.08)
        self.fog_anisotropy_g = Float32(0.72)
        self.fog_base_height_y = Float32(0.0)
        self.fog_albedo = Vec3(0.88, 0.91, 0.96)
        self.froxel_far_z = Float32(180.0)
        self.enable_gpu_rain = False
        self.rain_intensity = Float32(0.0)
        self.weather_wetness = Float32(0.0)
        self.windshield_rain = Float32(0.0)
        self.wind_velocity = Vec3(2.2, -0.4, 1.3)
        self.enable_water_plane = False
        self.water_level_y = Float32(0.0)
        self.water_absorption_rgb = Vec3(0.32, 0.075, 0.035)
        self.water_bounds_min_xz = (Float32(-1.0e5), Float32(-1.0e5))
        self.water_bounds_max_xz = (Float32(1.0e5), Float32(1.0e5))
        self.water_elliptical = False
        self.water_wave_scale = Float32(1.0)
        self.water_foam_band = Float32(0.48)
        self.water_foam = Float32(1.0)
        self.enable_motion_blur = False
        self.motion_blur_shutter = Float32(0.5)
        self.motion_blur_max_px = Float32(24.0)
        self.glare_streak_strength = Float32(0.0)
        self.min_auto_exposure = Float32(0.15)
        self.max_auto_exposure = Float32(3.20)
        self.cloud_coverage = Float32(0.0)
        self.cloud_wind_offset = Vec3(0.0, 0.0, 0.0)
        self.star_intensity = Float32(0.0)
        self.sky_zenith_clear = Vec3(0.0, 0.0, 0.0)
        self.sky_horizon_clear = Vec3(0.0, 0.0, 0.0)

        # --- sota:ml hook ---
        self.sota_ml = SotaMlDenoiser()
        self.sota_ml_up = SotaMlUpscaler()
        # --- end sota:ml hook ---
        self.active = self._init_vulkan_pipeline()

    def _build_default_render_graph(mut self):
        self.render_graph = AaaRenderGraph()
        var pix_r = self.render_w * self.render_h
        var pix_o = self.output_w * self.output_h
        var est_draws = min(self.max_draws, max(2048, pix_r // 2))
        var r_indirect = self.render_graph.add_resource("IndirectDrawCmds", est_draws * 16, transient=True)
        var r_hiz = self.render_graph.add_resource("ReverseZHiZPyramid", pix_r * 4, transient=True)
        var r_vis = self.render_graph.add_resource("VisBuffer64", pix_r * 8, transient=True)
        var r_vsm_req = self.render_graph.add_resource("VsmPageRequests", VSM_TABLE_WORDS * 4, transient=True)
        var r_hdr = self.render_graph.add_resource("LinearHdrScratch", pix_r * 32, transient=True)
        var r_svgf = self.render_graph.add_resource("SvgfPingPong", pix_r * 32, transient=True)
        var r_oit = self.render_graph.add_resource("Oit4NodePool", pix_r * 32, transient=True)
        var r_tsr = self.render_graph.add_resource("TsrUpscaleScratch", pix_o * 16, transient=True)

        var p0 = List[RgResourceAccess]()
        p0.append(RgResourceAccess(r_indirect, RG_USAGE_COMPUTE_WRITE))
        p0.append(RgResourceAccess(r_hiz, RG_USAGE_COMPUTE_READ))
        _ = self.render_graph.add_pass("MeshletCull", p0, RG_QUEUE_GRAPHICS)

        var p1 = List[RgResourceAccess]()
        p1.append(RgResourceAccess(r_indirect, RG_USAGE_INDIRECT_ARGS))
        p1.append(RgResourceAccess(r_vis, RG_USAGE_COLOR_ATT_WRITE))
        p1.append(RgResourceAccess(r_hiz, RG_USAGE_DEPTH_ATT_WRITE))
        _ = self.render_graph.add_pass("VisBufferHwRaster", p1, RG_QUEUE_GRAPHICS)

        var p2 = List[RgResourceAccess]()
        p2.append(RgResourceAccess(r_vis, RG_USAGE_COMPUTE_READ))
        p2.append(RgResourceAccess(r_vsm_req, RG_USAGE_COMPUTE_WRITE))
        _ = self.render_graph.add_pass("VsmMarkPages", p2, RG_QUEUE_ASYNC_COMPUTE)

        var p3 = List[RgResourceAccess]()
        p3.append(RgResourceAccess(r_vis, RG_USAGE_COMPUTE_READ))
        p3.append(RgResourceAccess(r_vsm_req, RG_USAGE_COMPUTE_READ))
        p3.append(RgResourceAccess(r_hdr, RG_USAGE_COMPUTE_WRITE))
        _ = self.render_graph.add_pass("DeferredMultiLobeRtShade", p3, RG_QUEUE_GRAPHICS)

        var p4 = List[RgResourceAccess]()
        p4.append(RgResourceAccess(r_hdr, RG_USAGE_COMPUTE_READ))
        p4.append(RgResourceAccess(r_svgf, RG_USAGE_COMPUTE_WRITE))
        _ = self.render_graph.add_pass("SvgfAtrousWavelet", p4, RG_QUEUE_GRAPHICS)

        var p5 = List[RgResourceAccess]()
        p5.append(RgResourceAccess(r_svgf, RG_USAGE_COMPUTE_READ))
        p5.append(RgResourceAccess(r_oit, RG_USAGE_COMPUTE_WRITE))
        _ = self.render_graph.add_pass("Oit6WayVolumetrics", p5, RG_QUEUE_GRAPHICS)

        var p6 = List[RgResourceAccess]()
        p6.append(RgResourceAccess(r_svgf, RG_USAGE_COMPUTE_READ))
        p6.append(RgResourceAccess(r_tsr, RG_USAGE_COMPUTE_WRITE))
        _ = self.render_graph.add_pass("TsrV2AndDisplayOetf", p6, RG_QUEUE_GRAPHICS)

        _ = self.render_graph.compile()

    def _init_vulkan_pipeline(mut self) -> Bool:
        if not self.vk.initialize(validate=False, need_discrete=False):
            return False
        if not self.vk.has_dynamic_rendering or not self.vk.has_descriptor_indexing:
            return False
        if not self.vk.allocate_command_buffers(2):
            return False
        if self.vk.has_side_queue():
            _ = self.vk.allocate_side_command_buffers(1)

        var use_rt = self.vk.has_ray_query
        var layouts = self.vk.create_aaa_layouts(
            AAA_STORAGE_COUNT,
            AAA_SINGLE_SAMPLERS,
            AAA_MAX_BINDLESS,
            use_rt,
            AAA_PUSH_BYTES,
        )
        self.dsl = layouts[0]
        self.pipe_layout = layouts[1]
        if self.dsl == 0 or self.pipe_layout == 0:
            return False

        var pool_sets = self.vk.allocate_aaa_descriptor_sets(
            self.dsl,
            FRAMES_IN_FLIGHT * 3,
            AAA_STORAGE_COUNT,
            AAA_SINGLE_SAMPLERS,
            AAA_MAX_BINDLESS,
            use_rt,
        )
        self.desc_pool = pool_sets[0]
        var all_sets = pool_sets[1].copy()
        if len(all_sets) < FRAMES_IN_FLIGHT * 3:
            return False
        self.desc_sets = List[Int]()
        self.desc_sets_cull_p1 = List[Int]()
        self.desc_sets_cull_p2 = List[Int]()
        for i in range(FRAMES_IN_FLIGHT):
            self.desc_sets.append(all_sets[i])
            self.desc_sets_cull_p1.append(all_sets[FRAMES_IN_FLIGHT + i])
            self.desc_sets_cull_p2.append(all_sets[FRAMES_IN_FLIGHT * 2 + i])
        self.desc_set = self.desc_sets[0]

        # --- sota:E hook (validated, XDG-located, atomically saved pipeline cache) ---
        self.sota_e_pso = sota_e_pso_cache_open(self.vk)
        sota_e_pso_batch_begin(self.vk)  # compute pipelines below compile in parallel at the flush
        # --- end sota:E hook ---

        # Compile all 10 AAA Hardware Graphics & Compute SPIR-V 1.5 Pipelines!
        self.pipe_meshlet_cull = self.vk.create_aaa_compute_pipeline(self.pipe_layout, meshlet_cull_shader())
        self.pipe_visbuffer_gfx = self.vk.create_graphics_pipeline(
            self.pipe_layout,
            visbuffer_vertex_shader(),
            visbuffer_fragment_shader(),
            has_color=True,
            color_format=FORMAT_R32G32_UINT,
            depth_format=FORMAT_D32_SFLOAT,
            depth_test=True,
            depth_write=True,
            depth_compare=COMPARE_GREATER_OR_EQUAL,
            cull_mode=CULL_MODE_BACK,
            front_face=FRONT_FACE_CW,
        )
        self.pipe_visbuffer_alphamask_gfx = self.vk.create_graphics_pipeline(
            self.pipe_layout,
            visbuffer_alphamask_vertex_shader(),
            visbuffer_alphamask_vfx_fragment_shader(),
            has_color=True,
            color_format=FORMAT_R32G32_UINT,
            depth_format=FORMAT_D32_SFLOAT,
            depth_test=True,
            depth_write=True,
            depth_compare=COMPARE_GREATER_OR_EQUAL,
            cull_mode=CULL_MODE_NONE,
            front_face=FRONT_FACE_CW,
        )
        self.pipe_npr_outline_gfx = self.vk.create_graphics_pipeline(
            self.pipe_layout,
            npr_outline_vertex_shader(),
            npr_outline_fragment_shader(),
            has_color=True,
            color_format=FORMAT_R32G32_UINT,
            depth_format=FORMAT_D32_SFLOAT,
            depth_test=True,
            depth_write=True,
            depth_compare=COMPARE_GREATER_OR_EQUAL,
            cull_mode=CULL_MODE_FRONT,
            front_face=FRONT_FACE_CW,
        )
        self.pipe_vsm_mark = self.vk.create_aaa_compute_pipeline(self.pipe_layout, vsm_mark_pages_shader())
        self.pipe_vsm_gfx = self.vk.create_graphics_pipeline(
            self.pipe_layout,
            vsm_page_vertex_shader(),
            List[UInt32](),
            has_color=False,
            color_format=0,
            depth_format=FORMAT_D32_SFLOAT,
            depth_test=True,
            depth_write=True,
            depth_compare=COMPARE_LESS_OR_EQUAL,
            cull_mode=CULL_MODE_BACK,
            front_face=FRONT_FACE_CW,
        )
        self.pipe_vsm_alphamask_gfx = self.vk.create_graphics_pipeline(
            self.pipe_layout,
            vsm_alphamask_vertex_shader(),
            vsm_alphamask_fragment_shader(),
            has_color=False,
            color_format=0,
            depth_format=FORMAT_D32_SFLOAT,
            depth_test=True,
            depth_write=True,
            depth_compare=COMPARE_LESS_OR_EQUAL,
            cull_mode=CULL_MODE_NONE,
            front_face=FRONT_FACE_CW,
        )
        self.pipe_surfcache = self.vk.create_aaa_compute_pipeline(self.pipe_layout, surface_cache_update_shader(use_rt))
        self.pipe_cluster_cull = self.vk.create_aaa_compute_pipeline(
            self.pipe_layout, aaa_cluster_cull_shader(BIND_AAA_CLUSTERS, BIND_AAA_GLOBALS)
        )
        self.pipe_dq_skinning = self.vk.create_aaa_compute_pipeline(
            self.pipe_layout, aaa_dq_skinning_shader(BIND_AAA_VERTICES, BIND_AAA_SKINNING)
        )
        self.pipe_deferred_shade = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_deferred_shade_shader(use_rt))
        self.pipe_svgf_reproject = self.vk.create_aaa_compute_pipeline(self.pipe_layout, svgf_temporal_reproject_shader())
        self.pipe_svgf_atrous = self.vk.create_aaa_compute_pipeline(self.pipe_layout, svgf_atrous_shader())
        # --- sota:gi hook --- (optional: a zero handle just disables the area-B modes)
        if use_rt:
            self.sota_gi.pipe_wrc = self.vk.create_aaa_compute_pipeline(self.pipe_layout, sota_wrc_update_shader(use_rt))
            self.sota_gi.pipe_resolve = self.vk.create_aaa_compute_pipeline(self.pipe_layout, sota_gi_resolve_shader(use_rt))
            self.sota_gi.pipe_upsample = self.vk.create_aaa_compute_pipeline(self.pipe_layout, sota_gi_upsample_shader())
        # --- end sota:gi hook ---
        self.pipe_oit_vol = self.vk.create_aaa_compute_pipeline(self.pipe_layout, oit_and_6way_volumetric_shader())
        self.pipe_luma_hist = self.vk.create_aaa_compute_pipeline(
            self.pipe_layout, aaa_luminance_histogram_shader(BIND_AAA_HDR_OUT, BIND_AAA_POST_SCRATCH)
        )
        self.pipe_adapt_exp = self.vk.create_aaa_compute_pipeline(
            self.pipe_layout, aaa_adapt_exposure_shader(BIND_AAA_POST_SCRATCH)
        )
        self.pipe_bloom_down = self.vk.create_aaa_compute_pipeline(
            self.pipe_layout, aaa_bloom_downsample_shader(BIND_AAA_HDR_OUT, BIND_AAA_POST_SCRATCH)
        )
        self.pipe_bloom_up = self.vk.create_aaa_compute_pipeline(
            self.pipe_layout, aaa_bloom_upsample_shader(BIND_AAA_HDR_OUT, BIND_AAA_POST_SCRATCH)
        )
        self.pipe_bokeh_dof = self.vk.create_aaa_compute_pipeline(
            self.pipe_layout, aaa_bokeh_dof_shader(BIND_AAA_HDR_OUT, BIND_AAA_POST_SCRATCH)
        )
        self.pipe_tsr_display = self.vk.create_aaa_compute_pipeline(self.pipe_layout, tsr_v2_and_display_shader())
        self.pipe_spec_reblur_temp = self.vk.create_aaa_compute_pipeline(self.pipe_layout, spec_reblur_temporal_shader())
        self.pipe_spec_reblur_atrous = self.vk.create_aaa_compute_pipeline(self.pipe_layout, spec_reblur_atrous_shader())
        self.pipe_froxel_inject = self.vk.create_aaa_compute_pipeline(self.pipe_layout, froxel_inject_and_reproject_shader())
        self.pipe_froxel_prefix = self.vk.create_aaa_compute_pipeline(self.pipe_layout, froxel_prefix_integrate_shader())
        self.pipe_froxel_march = self.vk.create_aaa_compute_pipeline(self.pipe_layout, froxel_raymarch_integrate_shader())
        self.pipe_particle_clear = self.vk.create_aaa_compute_pipeline(self.pipe_layout, gpu_particle_clear_tiles_shader())
        self.pipe_particle_sim = self.vk.create_aaa_compute_pipeline(self.pipe_layout, gpu_particle_sim_and_bin_shader())
        self.pipe_particle_comp = self.vk.create_aaa_compute_pipeline(self.pipe_layout, gpu_particle_tile_composite_shader())
        self.pipe_water_refract = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_water_refract_shader(use_rt))
        if use_rt:
            self.pipe_rt_shadow_trace = self.vk.create_aaa_compute_pipeline(self.pipe_layout, rt_sun_shadow_trace_shader())
            self.pipe_rt_shadow_filter = self.vk.create_aaa_compute_pipeline(self.pipe_layout, rt_sun_shadow_filter_shader())
            self.pipe_glass_refract = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_glass_refract_shader())
        self.pipe_tsr_v3_resolve = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_tsr_v3_resolve_shader())
        self.pipe_motion_blur = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_motion_blur_shader())
        self.pipe_display_tonemap = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_display_tonemap_shader())
        # --- sota:camera hook (pipelines; optional, features skip when not ready) ---
        _ = self.sota_cam.create_pipelines(self.vk, self.pipe_layout)
        # --- end sota:camera hook ---
        _ = self.sota_atmo.create_pipelines(self.vk, self.pipe_layout)  # --- sota:atmo hook --- (optional)
        self.pipe_npr_cel_shade = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_npr_cel_shade_shader())
        self.pipe_npr_post_fx = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_npr_post_fx_shader())
        self.pipe_ui_overlay = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_ui_overlay_shader())
        self.pipe_scrgb_pack = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_pack_scrgb_rgba16f_shader())
        # --- sota:E hook (parallel compile of the batched compute pipelines) ---
        if sota_e_pso_batch_flush(self.vk) > 0:
            return False
        # --- end sota:E hook ---

        if (
            self.pipe_meshlet_cull == 0
            or self.pipe_visbuffer_gfx == 0
            or self.pipe_visbuffer_alphamask_gfx == 0
            or self.pipe_npr_outline_gfx == 0
            or self.pipe_vsm_mark == 0
            or self.pipe_vsm_gfx == 0
            or self.pipe_vsm_alphamask_gfx == 0
            or self.pipe_surfcache == 0
            or self.pipe_cluster_cull == 0
            or self.pipe_dq_skinning == 0
            or self.pipe_deferred_shade == 0
            or self.pipe_svgf_reproject == 0
            or self.pipe_svgf_atrous == 0
            or self.pipe_oit_vol == 0
            or self.pipe_luma_hist == 0
            or self.pipe_adapt_exp == 0
            or self.pipe_bloom_down == 0
            or self.pipe_bloom_up == 0
            or self.pipe_bokeh_dof == 0
            or self.pipe_tsr_display == 0
            or self.pipe_spec_reblur_temp == 0
            or self.pipe_spec_reblur_atrous == 0
            or self.pipe_froxel_inject == 0
            or self.pipe_froxel_prefix == 0
            or self.pipe_froxel_march == 0
            or self.pipe_particle_clear == 0
            or self.pipe_particle_sim == 0
            or self.pipe_particle_comp == 0
            or self.pipe_water_refract == 0
            or self.pipe_tsr_v3_resolve == 0
            or self.pipe_motion_blur == 0
            or self.pipe_display_tonemap == 0
            or self.pipe_npr_cel_shade == 0
            or self.pipe_npr_post_fx == 0
            or self.pipe_ui_overlay == 0
            or self.pipe_scrgb_pack == 0
        ):
            return False

        if self.vk.timestamps:
            _ = self.vk.create_query_pool(AAA_QUERY_SLOT_STRIDE * FRAMES_IN_FLIGHT)

        # Samplers
        self.samp_aniso16 = self.vk.create_sampler(linear=True, max_anisotropy=16.0, clamp_to_edge=False)
        self.samp_linear_clamp = self.vk.create_sampler(linear=True, max_anisotropy=1.0, clamp_to_edge=True)
        self.samp_nearest = self.vk.create_sampler(linear=False, max_anisotropy=1.0, clamp_to_edge=True)

        # Render Targets & LUTs
        self.vis_img = self.vk.create_image_2d(
            self.render_w, self.render_h, FORMAT_R32G32_UINT, IMAGE_USAGE_COLOR_ATTACHMENT | IMAGE_USAGE_SAMPLED
        )
        self.depth_img = self.vk.create_image_2d(
            self.render_w, self.render_h, FORMAT_D32_SFLOAT, IMAGE_USAGE_DEPTH_ATTACHMENT | IMAGE_USAGE_SAMPLED
        )
        self.vsm_img = self.vk.create_image_2d(
            VSM_PHYS_ATLAS_RES, VSM_PHYS_ATLAS_RES, FORMAT_D32_SFLOAT, IMAGE_USAGE_DEPTH_ATTACHMENT | IMAGE_USAGE_SAMPLED
        )
        self.brdf_lut_img = self.vk.create_image_2d(
            64, 64, FORMAT_R16G16_SFLOAT, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED
        )
        self.ltc_lut_img = self.vk.create_image_2d(
            64, 64, FORMAT_R16G16_SFLOAT, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED
        )
        self.env_cube_img = self.vk.create_image_2d(
            self.env_cube_dim,
            self.env_cube_dim,
            FORMAT_BC6H_UFLOAT,
            IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED,
            mip_levels=self.env_mip_levels,
            cube=True,
        )
        if (
            self.vis_img.image == 0
            or self.depth_img.image == 0
            or self.vsm_img.image == 0
            or self.brdf_lut_img.image == 0
            or self.ltc_lut_img.image == 0
            or self.env_cube_img.image == 0
        ):
            return False

        # UI Font atlas (Feature 15)
        _ = self.ui_font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36, 5)
        var fw = self.ui_font.atlas_width if self.ui_font.atlas_width > 0 else 512
        var fh = self.ui_font.atlas_height if self.ui_font.atlas_height > 0 else 512
        self.ui_font_img = self.vk.create_image_2d(
            fw, fh, FORMAT_R8_UNORM, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED
        )
        self.ui_font_atlas_slot = len(self.bindless_images)
        self.bindless_images.append(self.ui_font_img.copy())

        # Allocate Storage Buffers (`local=True` places GPU SSBOs in `DEVICE_LOCAL` Resizable BAR VRAM!)
        var max_pix = max(self.output_w * self.output_h, self.render_w * self.render_h)
        var bda_flags = BUFFER_USAGE_DEVICE_ADDRESS | BUFFER_USAGE_AS_BUILD_INPUT_KHR
        # --- sota:perf hook --- geometry pools sized by `geometry_mb` (default 32 MB = v5 sizes)
        self.buf_vertices = self.vk.create_buffer_bda(self.geometry_mb * 1024 * 1024, extra_usage=bda_flags, host_visible=True)
        self.buf_indices = self.vk.create_buffer_bda(self.geometry_mb * 1024 * 1024, extra_usage=bda_flags, host_visible=True)
        self.buf_meshlets = self.vk.create_buffer((8 if self.geometry_mb <= 32 else self.geometry_mb // 2) * 1024 * 1024, local=True)
        self.buf_materials = self.vk.create_buffer(1024 * AAA_MATERIAL_WORDS * 4 + SOTA_MAT_EXTRA_BYTES, local=True)  # sota:mat hook: + layer words + LUTs
        sota_mat_clear_layers(self.buf_materials.cpu_address)  # --- sota:mat hook ---
        self.buf_vsm_indirect = self.vk.create_buffer(64 + self.max_draws * 16, local=True)
        self.sota_shadow.vsm_cmd_capacity = self.max_draws  # --- sota:shadow hook ---
        self.buf_hdr_out = self.vk.create_buffer(max_pix * 8 * 4, local=True)
        var hist_words = rt_shadow_raw_offset(max_pix) + rt_shadow_total_words(max_pix)
        hist_words += sota_svgf_scratch_words(max_pix)  # --- sota:gi hook --- SVGF a-trous scratch
        # --- sota:ml hook --- (scratch appended past every other history region; offset recorded here)
        self.sota_ml.scratch_base = hist_words
        self.sota_ml.scratch_max_pix = max_pix
        hist_words += SotaMlDenoiser.scratch_words(max_pix)
        # --- end sota:ml hook ---
        self.sota_atmo.region_base = hist_words  # --- sota:atmo hook --- LUTs + aerial-perspective volume (1.66 MB)
        hist_words += sota_atmo_region_words()  # --- sota:atmo hook ---
        self.buf_history = self.vk.create_buffer(hist_words * 4, local=True)
        self.buf_vsm_table = self.vk.create_buffer(VSM_TABLE_WORDS * 4, local=False, cpu_reads=True)
        # --- sota:gi hook --- (+ the area-B control block; the full region is allocated on first use)
        self.buf_surfcache = self.vk.create_buffer((4096 + self.max_surfcache_tris * 4 + SOTA_GI_CTRL_WORDS) * 4, local=True)
        self.buf_oit_nodes = self.vk.create_buffer(gpu_particles_total_oit_words(max_pix) * 4, local=True)
        self.buf_skinning = self.vk.create_buffer(AAA_SKIN_BUFFER_WORDS * 4, local=True)
        # Per-pixel post regions end below word 1024 + max_pix * 10; the UI prims of each frame in flight follow.
        var scratch_words = AAA_POST_UI_PRIMS_BASE_OFFSET + max_pix * 10 + FRAMES_IN_FLIGHT * AAA_UI_PRIM_REGION_WORDS
        # --- sota:E hook --- hair deep opacity map region after the UI prims
        self.sota_e_hair.base_word = scratch_words
        scratch_words += sota_e_dom_words(self.sota_e_hair.res)
        # --- end sota:E hook ---
        self.buf_post_scratch = self.vk.create_buffer(scratch_words * 4, local=True)
        self.buf_ldr_readback = self.vk.create_buffer(64 + self.output_w * self.output_h * 4, local=False, cpu_reads=True)
        self.buf_staging = self.vk.create_buffer(8 * 1024 * 1024, local=False)
        self.buf_indirect_phase2 = self.vk.create_buffer(64 + self.max_draws * 20, local=True)
        self.buf_occluded_candidates = self.vk.create_buffer(64 + self.max_draws * 8, local=True)  # sota:geometry hook: (inst, meshlet) pairs

        # Ring-Buffered CPU-to-GPU Dynamic Buffers (Feature 14: FRAMES_IN_FLIGHT = 2)
        for _ in range(FRAMES_IN_FLIGHT):
            var b_glob = self.vk.create_buffer(16384, local=True)
            _ = external_call["memset", Int](b_glob.cpu_address, Int32(0), b_glob.size)
            var b_inst = self.vk.create_buffer(self.max_instances * AAA_INSTANCE_WORDS * 4, local=True)
            var b_ind = self.vk.create_buffer(64 + self.max_draws * 20, local=True)
            var b_am_ind = self.vk.create_buffer(64 + self.sota_route_draws * 20, local=True)  # --- sota:perf hook ---
            var b_out_ind = self.vk.create_buffer(64 + self.sota_route_draws * 20, local=True)  # --- sota:perf hook ---
            var b_clust = self.vk.create_buffer(AAA_CLUSTER_BUFFER_WORDS * 4, local=True)
            _ = external_call["memset", Int](b_clust.cpu_address, Int32(0), b_clust.size)
            self.ring_globals.append(b_glob^)
            self.ring_instances.append(b_inst^)
            self.ring_indirect.append(b_ind^)
            self.ring_alphamask_indirect.append(b_am_ind^)
            self.ring_outline_indirect.append(b_out_ind^)
            self.ring_clusters.append(b_clust^)
            if self.vk.has_ray_query:
                var b_as = self.vk.create_buffer_bda(self.max_instances * 64, extra_usage=bda_flags, host_visible=True)
                var tlas = VulkanAccelStruct()
                var init_inst_da = self.vk.buffer_device_address(b_as)
                _ = ensure_tlas_capacity_mixed_opaque(self.vk, tlas, init_inst_da, self.max_instances, self.buf_as_scratch)
                self.ring_as_instances.append(b_as^)
                self.ring_tlas.append(tlas^)
            self.frame_fences.append(0)

        self.buf_globals = self.ring_globals[0].copy()
        self.buf_instances = self.ring_instances[0].copy()
        self.buf_indirect = self.ring_indirect[0].copy()
        self.buf_alphamask_indirect = self.ring_alphamask_indirect[0].copy()
        self.buf_outline_indirect = self.ring_outline_indirect[0].copy()
        self.buf_clusters = self.ring_clusters[0].copy()
        if self.vk.has_ray_query and len(self.ring_as_instances) > 0:
            self.buf_as_instances = self.ring_as_instances[0].copy()
            self.frame_tlas = self.ring_tlas[0].copy()

        _ = external_call["memset", Int](self.buf_history.cpu_address, Int32(0), self.buf_history.size)
        _ = external_call["memset", Int](self.buf_vsm_table.cpu_address, Int32(0), self.buf_vsm_table.size)
        _ = external_call["memset", Int](self.buf_surfcache.cpu_address, Int32(0), self.buf_surfcache.size)
        _ = external_call["memset", Int](self.buf_oit_nodes.cpu_address, Int32(0), self.buf_oit_nodes.size)
        _ = external_call["memset", Int](self.buf_skinning.cpu_address, Int32(0), self.buf_skinning.size)
        _ = external_call["memset", Int](self.buf_post_scratch.cpu_address, Int32(0), self.buf_post_scratch.size)
        _ = external_call["memset", Int](self.buf_ldr_readback.cpu_address, Int32(0), self.buf_ldr_readback.size)

        # Bind Storage Buffers and Samplers across all descriptor sets
        for slot in range(FRAMES_IN_FLIGHT):
            var ds = self.desc_sets[slot]
            self.vk.write_descriptor(ds, BIND_AAA_VERTICES, self.buf_vertices)
            self.vk.write_descriptor(ds, BIND_AAA_INDICES, self.buf_indices)
            self.vk.write_descriptor(ds, BIND_AAA_MESHLETS, self.buf_meshlets)
            self.vk.write_descriptor(ds, BIND_AAA_INSTANCES, self.ring_instances[slot])
            self.vk.write_descriptor(ds, BIND_AAA_MATERIALS, self.buf_materials)
            self.vk.write_descriptor(ds, BIND_AAA_INDIRECT, self.ring_indirect[slot])
            self.vk.write_descriptor(ds, BIND_AAA_GLOBALS, self.ring_globals[slot])
            self.vk.write_descriptor(ds, BIND_AAA_HDR_OUT, self.buf_hdr_out)
            self.vk.write_descriptor(ds, BIND_AAA_HISTORY, self.buf_history)
            self.vk.write_descriptor(ds, BIND_AAA_VSM_TABLE, self.buf_vsm_table)
            self.vk.write_descriptor(ds, BIND_AAA_SURFCACHE, self.buf_surfcache)
            self.vk.write_descriptor(ds, BIND_AAA_OIT_NODES, self.buf_oit_nodes)
            self.vk.write_descriptor(ds, BIND_AAA_CLUSTERS, self.ring_clusters[slot])
            self.vk.write_descriptor(ds, BIND_AAA_SKINNING, self.buf_skinning)
            self.vk.write_descriptor(ds, BIND_AAA_POST_SCRATCH, self.buf_post_scratch)

            self.vk.write_aaa_texture(ds, BIND_AAA_VIS_TEX, 0, self.samp_nearest, self.vis_img.view)
            self.vk.write_aaa_texture(ds, BIND_AAA_DEPTH_TEX, 0, self.samp_linear_clamp, self.depth_img.view)
            self.vk.write_aaa_texture(ds, BIND_AAA_VSM_TEX, 0, self.samp_linear_clamp, self.vsm_img.view)
            self.vk.write_aaa_texture(ds, BIND_AAA_ENV_CUBE, 0, self.samp_linear_clamp, self.env_cube_img.view)
            self.vk.write_aaa_texture(ds, BIND_AAA_BRDF_LUT, 0, self.samp_linear_clamp, self.brdf_lut_img.view)
            self.vk.write_aaa_texture(ds, BIND_AAA_LTC_LUT, 0, self.samp_linear_clamp, self.ltc_lut_img.view)

            # Phase 1 Cull Descriptor Set
            var ds_p1 = self.desc_sets_cull_p1[slot]
            self.vk.write_descriptor(ds_p1, BIND_AAA_VERTICES, self.buf_vertices)
            self.vk.write_descriptor(ds_p1, BIND_AAA_INDICES, self.buf_indices)
            self.vk.write_descriptor(ds_p1, BIND_AAA_MESHLETS, self.buf_meshlets)
            self.vk.write_descriptor(ds_p1, BIND_AAA_INSTANCES, self.ring_instances[slot])
            self.vk.write_descriptor(ds_p1, BIND_AAA_MATERIALS, self.buf_materials)
            self.vk.write_descriptor(ds_p1, BIND_AAA_INDIRECT, self.ring_indirect[slot])
            self.vk.write_descriptor(ds_p1, BIND_AAA_GLOBALS, self.ring_globals[slot])
            self.vk.write_descriptor(ds_p1, BIND_AAA_CANDIDATES, self.buf_occluded_candidates)
            self.vk.write_aaa_texture(ds_p1, BIND_AAA_DEPTH_TEX, 0, self.samp_linear_clamp, self.depth_img.view)

            # Phase 2 Cull Descriptor Set: BIND_AAA_INDIRECT -> buf_indirect_phase2, BIND_AAA_CANDIDATES -> buf_occluded_candidates
            var ds_p2 = self.desc_sets_cull_p2[slot]
            self.vk.write_descriptor(ds_p2, BIND_AAA_VERTICES, self.buf_vertices)
            self.vk.write_descriptor(ds_p2, BIND_AAA_INDICES, self.buf_indices)
            self.vk.write_descriptor(ds_p2, BIND_AAA_MESHLETS, self.buf_meshlets)
            self.vk.write_descriptor(ds_p2, BIND_AAA_INSTANCES, self.ring_instances[slot])
            self.vk.write_descriptor(ds_p2, BIND_AAA_MATERIALS, self.buf_materials)
            self.vk.write_descriptor(ds_p2, BIND_AAA_INDIRECT, self.buf_indirect_phase2)
            self.vk.write_descriptor(ds_p2, BIND_AAA_GLOBALS, self.ring_globals[slot])
            self.vk.write_descriptor(ds_p2, BIND_AAA_CANDIDATES, self.buf_occluded_candidates)
            self.vk.write_aaa_texture(ds_p2, BIND_AAA_DEPTH_TEX, 0, self.samp_linear_clamp, self.depth_img.view)
        self.vk.write_aaa_texture(self.desc_set, BIND_AAA_ENV_CUBE, 0, self.samp_linear_clamp, self.env_cube_img.view)
        self.vk.write_aaa_texture(self.desc_set, BIND_AAA_BRDF_LUT, 0, self.samp_linear_clamp, self.brdf_lut_img.view)
        self.vk.write_aaa_texture(self.desc_set, BIND_AAA_LTC_LUT, 0, self.samp_linear_clamp, self.ltc_lut_img.view)

        # --- sota:geometry hook ---
        # Own layout/sets/pipelines; failure leaves `sota_geom.active = False` and the legacy culler runs.
        if not self.sota_geom.init(
            self.vk, FRAMES_IN_FLIGHT, self.render_w, self.render_h, self.buf_meshlets.size // (AAA_MESHLET_WORDS * 4),
            self.buf_meshlets, self.ring_instances, self.ring_indirect, self.ring_globals,
            self.buf_occluded_candidates, self.buf_indirect_phase2, self.depth_img.view, self.samp_nearest,
        ):
            self.sota_geom.active = False
        if not self.sota_vt.init(
            self.vk, FRAMES_IN_FLIGHT, self.buf_vertices, self.buf_indices, self.buf_meshlets, self.ring_instances,
            self.buf_materials, self.ring_globals, self.vis_img.view, self.samp_nearest,
        ):
            self.sota_vt.active = False
        # --- end sota:geometry hook ---
        # --- sota:E hook (GPU-driven alpha-mask / outline packing on Area A's culler) ---
        _ = self.sota_e_gpu.init(
            self.vk, self.sota_geom, FRAMES_IN_FLIGHT, self.buf_meshlets, self.ring_instances,
            self.ring_alphamask_indirect, self.ring_outline_indirect, self.ring_globals,
            self.buf_occluded_candidates, self.depth_img.view, self.samp_nearest,
        )
        _ = self.sota_e_hair.init(self.vk, self.pipe_layout, self.sota_e_hair.base_word)
        # --- end sota:E hook ---

        # Bake & Upload 64x64 RG16F Split-Sum BRDF LUT + 64x64 RG16F Heitz LTC LUT!
        var lut_bytes = bake_split_sum_brdf_lut_rg16f(64, 128)
        var ltc_bytes = bake_ltc_lut_rg16f(64)
        var ltc_st_off = (len(lut_bytes) + 63) & ~63
        _ = external_call["memcpy", Int](self.buf_staging.cpu_address, Int(lut_bytes.unsafe_ptr()), len(lut_bytes))
        _ = external_call["memcpy", Int](self.buf_staging.cpu_address + ltc_st_off, Int(ltc_bytes.unsafe_ptr()), len(ltc_bytes))
        _ = len(lut_bytes)
        _ = len(ltc_bytes)

        if not self.vk.begin_commands(0):
            return False
        self.vk.transition_image(0, self.brdf_lut_img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        self.vk.upload_image_mip(0, self.brdf_lut_img, self.buf_staging, 0, 0, 64, 64)
        self.vk.transition_image(0, self.brdf_lut_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)

        self.vk.transition_image(0, self.ltc_lut_img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        self.vk.upload_image_mip(0, self.ltc_lut_img, self.buf_staging, ltc_st_off, 0, 64, 64)
        self.vk.transition_image(0, self.ltc_lut_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)

        # Initialize Depth & VSM Atlas to valid `SHADER_READ_ONLY` layouts
        self.vk.transition_image(0, self.vis_img, LAYOUT_COLOR_ATTACHMENT, PIPELINE_TOP, 0, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE)
        self.vk.transition_image(0, self.depth_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_TOP, 0, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE)
        self.vk.begin_rendering(0, 0, 0, self.render_w, self.render_h, True, self.vis_img.view, True, 0, 0, True, self.depth_img.view, True, 0.0)
        self.vk.end_rendering(0)
        self.vk.transition_image(0, self.vis_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        self.vk.transition_image(0, self.depth_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)

        self.vk.transition_image(0, self.vsm_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_TOP, 0, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE)
        self.vk.begin_rendering(0, 0, 0, VSM_PHYS_ATLAS_RES, VSM_PHYS_ATLAS_RES, False, 0, False, 0, 0, True, self.vsm_img.view, True, 1.0)
        self.vk.end_rendering(0)
        self.vk.transition_image(0, self.vsm_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)

        # Upload initial UI font atlas if valid
        if self.ui_font.valid and self.ui_font_img.image != 0:
            var atlas_bytes = len(self.ui_font.atlas)
            var font_st_off = (ltc_st_off + len(ltc_bytes) + 63) & ~63
            self.ensure_staging_capacity(font_st_off + atlas_bytes)
            _ = external_call["memcpy", Int](self.buf_staging.cpu_address + font_st_off, Int(self.ui_font.atlas.unsafe_ptr()), atlas_bytes)
            self.vk.transition_image(0, self.ui_font_img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
            self.vk.upload_image_subregion(0, self.ui_font_img, self.buf_staging, font_st_off, 0, 0, self.ui_font.atlas_width, self.ui_font.atlas_height)
            self.vk.transition_image(0, self.ui_font_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
            self.ui_font.mark_clean()

        if self.vk.has_ray_query and len(self.ring_tlas) > 0:
            for slot in range(FRAMES_IN_FLIGHT):
                if self.ring_tlas[slot].handle != 0:
                    var inst_da = self.vk.buffer_device_address(self.ring_as_instances[slot])
                    record_build_tlas_mixed_opaque(self.vk, 0, self.ring_tlas[slot], inst_da, 0, self.buf_as_scratch)
                    self.vk.memory_barrier2(
                        0,
                        PIPELINE_AS_BUILD_KHR,
                        Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR),
                        PIPELINE_AS_BUILD_KHR | PIPELINE_COMPUTE,
                        Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
                    )

        var init_fence = self.vk.submit(0)
        if init_fence == 0 or not self.vk.wait(init_fence):
            return False

        if self.vk.has_ray_query and len(self.ring_tlas) > 0:
            for slot in range(FRAMES_IN_FLIGHT):
                if self.ring_tlas[slot].handle != 0:
                    self.vk.write_aaa_tlas(self.desc_sets[slot], BIND_AAA_TLAS, self.ring_tlas[slot].handle)

        # Upload default BC6H Golden-Hour / Sky IBL + White dummy BC7 texture at bindless[0]
        self.upload_ibl_environment(
            bake_bc6h_ibl_cubemap_and_sh9(
                SkyDomeParams(uniform_furnace=False),
                base_dim=self.env_cube_dim,
                mip_levels=self.env_mip_levels,
            )
        )
        if self.ui_font_img.view != 0:
            for ds_i in range(len(self.desc_sets)):
                self.vk.write_aaa_texture(
                    self.desc_sets[ds_i],
                    BIND_AAA_BINDLESS_TEX,
                    self.ui_font_atlas_slot,
                    self.samp_linear_clamp,
                    self.ui_font_img.view,
                )
        var white_px = List[UInt32](length=16, fill=0xFFFFFFFF)
        var white_slot = self.upload_bc7_texture(white_px, 4, 4, srgb=False, generate_mips=False)
        if white_slot >= 0 and white_slot < len(self.bindless_images):
            var fallback_view = self.bindless_images[white_slot].view
            if self.ui_font_img.view == 0:
                for ds_i in range(len(self.desc_sets)):
                    self.vk.write_aaa_texture(
                        self.desc_sets[ds_i],
                        BIND_AAA_BINDLESS_TEX,
                        self.ui_font_atlas_slot,
                        self.samp_aniso16,
                        fallback_view,
                    )
            for s_idx in range(len(self.bindless_images), AAA_MAX_BINDLESS):
                for ds_i in range(len(self.desc_sets)):
                    self.vk.write_aaa_texture(self.desc_sets[ds_i], BIND_AAA_BINDLESS_TEX, s_idx, self.samp_aniso16, fallback_view)

        self._build_default_render_graph()
        return True

    # ========================================================================
    # Asset Registration: Bindless BC7 / BC5 / BC6H Textures, Meshes & BLAS
    # ========================================================================

    def ensure_staging_capacity(mut self, required_bytes: Int):
        """Ensures buf_staging has at least required_bytes, growing dynamically."""
        if required_bytes <= self.buf_staging.size:
            return
        var old_size = self.buf_staging.size
        var new_cap = max(old_size * 2, required_bytes)
        new_cap = (new_cap + 65535) & (~65535)
        var new_buf = self.vk.create_buffer(new_cap, local=False)
        if self.buf_staging.cpu_address != 0 and new_buf.cpu_address != 0 and old_size > 0:
            _ = external_call["memcpy", Int](new_buf.cpu_address, self.buf_staging.cpu_address, old_size)
        self.vk.destroy_buffer(self.buf_staging)
        self.buf_staging = new_buf^

    def upload_ibl_environment(mut self, ibl: BakedIblEnvironment):
        self.sh9_irradiance = ibl.sh9.copy()
        self.sky_zenith_clear = Vec3(0.0, 0.0, 0.0)
        self.sky_horizon_clear = Vec3(0.0, 0.0, 0.0)
        var total_staging = 0
        for b_arr in ibl.mip_face_bytes:
            total_staging += (len(b_arr) + 15) & ~15
        self.ensure_staging_capacity(total_staging)
        var staging_off = 0
        for b_arr in ibl.mip_face_bytes:
            _ = external_call["memcpy", Int](self.buf_staging.cpu_address + staging_off, Int(b_arr.unsafe_ptr()), len(b_arr))
            staging_off += (len(b_arr) + 15) & ~15

        if not self.vk.begin_commands(0):
            return
        self.vk.transition_image(0, self.env_cube_img, LAYOUT_TRANSFER_TARGET, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        var cur_off = 0
        for mip in range(ibl.mip_levels):
            var m_dim = max(4, ibl.base_dim >> mip)
            for face in range(6):
                var idx = mip * 6 + face
                self.vk.upload_image_mip(0, self.env_cube_img, self.buf_staging, cur_off, mip, m_dim, m_dim, base_layer=face, layer_count=1)
                cur_off += (len(ibl.mip_face_bytes[idx]) + 15) & ~15
        self.vk.transition_image(0, self.env_cube_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        var f = self.vk.submit(0)
        if f != 0:
            _ = self.vk.wait(f)

    def upload_bc7_texture(
        mut self,
        pixels_argb: List[UInt32],
        width: Int,
        height: Int,
        srgb: Bool = True,
        generate_mips: Bool = True,
    ) -> Int:
        """Compresses `0xAARRGGBB` pixels into a `BC7` mipmap pyramid and binds to `BIND_AAA_BINDLESS_TEX`!"""
        if len(self.bindless_images) >= AAA_MAX_BINDLESS:
            return -1
        var mips = 1
        if generate_mips:
            var d = min(width, height)
            while d > 4 and mips < 8:
                d >>= 1
                mips += 1

        var fmt = FORMAT_BC7_SRGB if srgb else FORMAT_BC7_UNORM
        var img = self.vk.create_image_2d(width, height, fmt, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED, mip_levels=mips)
        if img.image == 0:
            return -1

        var sota_t0 = sota_now()  # --- sota:perf hook ---
        var cur_px = pixels_argb.copy()
        var cw = width
        var ch = height
        var mip_offsets = List[Int]()
        var mip_dims = List[Tuple[Int, Int]]()
        var st_off = 0

        for m in range(mips):
            var bc7 = encode_bc7_rgba(cur_px, cw, ch)
            self.ensure_staging_capacity(st_off + len(bc7))
            _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_off, Int(bc7.unsafe_ptr()), len(bc7))
            mip_offsets.append(st_off)
            mip_dims.append((cw, ch))
            st_off += (len(bc7) + 15) & ~15
            _ = len(bc7)
            if m + 1 < mips:
                var ds = downsample_argb_2x(cur_px, cw, ch)
                cur_px = ds[0].copy()
                cw = ds[1]
                ch = ds[2]

        var sota_t1 = sota_now()  # --- sota:perf hook ---
        self.sota_upload.tex_cpu_s += sota_t1 - sota_t0  # --- sota:perf hook ---
        if not self.vk.begin_commands(0):
            self.vk.destroy_image(img)
            return -1
        self.vk.transition_image(0, img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        for m in range(mips):
            self.vk.upload_image_mip(0, img, self.buf_staging, mip_offsets[m], m, mip_dims[m][0], mip_dims[m][1])
        self.vk.transition_image(0, img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        var f = self.vk.submit(0)
        if f != 0:
            _ = self.vk.wait(f)

        var slot = len(self.bindless_images)
        for ds_i in range(len(self.desc_sets)):
            self.vk.write_aaa_texture(self.desc_sets[ds_i], BIND_AAA_BINDLESS_TEX, slot, self.samp_aniso16, img.view)
        self.bindless_images.append(img^)
        self.sota_upload.tex_gpu_s += sota_now() - sota_t1  # --- sota:perf hook ---
        self.sota_upload.textures += 1  # --- sota:perf hook ---
        self.sota_upload.submits += 1  # --- sota:perf hook ---
        return slot

    def upload_bc5_normal_texture(
        mut self,
        normals: List[Vec3],
        width: Int,
        height: Int,
        generate_mips: Bool = True,
    ) -> Int:
        """Compresses unit tangent normals into a `BC5_UNORM` mipmap pyramid and binds to `BIND_AAA_BINDLESS_TEX`!"""
        if len(self.bindless_images) >= AAA_MAX_BINDLESS:
            return -1
        var mips = 1
        if generate_mips:
            var d = min(width, height)
            while d > 4 and mips < 8:
                d >>= 1
                mips += 1

        var img = self.vk.create_image_2d(width, height, FORMAT_BC5_UNORM, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED, mip_levels=mips)
        if img.image == 0:
            return -1

        var sota_t0 = sota_now()  # --- sota:perf hook ---
        var cur_n = normals.copy()
        var cw = width
        var ch = height
        var mip_offsets = List[Int]()
        var mip_dims = List[Tuple[Int, Int]]()
        var st_off = 0

        for m in range(mips):
            var bc5 = encode_bc5_normals(cur_n, cw, ch)
            self.ensure_staging_capacity(st_off + len(bc5))
            _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_off, Int(bc5.unsafe_ptr()), len(bc5))
            mip_offsets.append(st_off)
            mip_dims.append((cw, ch))
            st_off += (len(bc5) + 15) & ~15
            _ = len(bc5)
            if m + 1 < mips:
                var ds = downsample_normals_2x(cur_n, cw, ch)
                cur_n = ds[0].copy()
                cw = ds[1]
                ch = ds[2]

        var sota_t1 = sota_now()  # --- sota:perf hook ---
        self.sota_upload.tex_cpu_s += sota_t1 - sota_t0  # --- sota:perf hook ---
        if not self.vk.begin_commands(0):
            self.vk.destroy_image(img)
            return -1
        self.vk.transition_image(0, img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        for m in range(mips):
            self.vk.upload_image_mip(0, img, self.buf_staging, mip_offsets[m], m, mip_dims[m][0], mip_dims[m][1])
        self.vk.transition_image(0, img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        var f = self.vk.submit(0)
        if f != 0:
            _ = self.vk.wait(f)

        var slot = len(self.bindless_images)
        for ds_i in range(len(self.desc_sets)):
            self.vk.write_aaa_texture(self.desc_sets[ds_i], BIND_AAA_BINDLESS_TEX, slot, self.samp_aniso16, img.view)
        self.bindless_images.append(img^)
        self.sota_upload.tex_gpu_s += sota_now() - sota_t1  # --- sota:perf hook ---
        self.sota_upload.textures += 1  # --- sota:perf hook ---
        self.sota_upload.submits += 1  # --- sota:perf hook ---
        return slot

    def upload_bc6h_texture(mut self, hdr_pixels: List[Vec3], width: Int, height: Int) -> Int:
        if len(self.bindless_images) >= AAA_MAX_BINDLESS:
            return -1
        var img = self.vk.create_image_2d(width, height, FORMAT_BC6H_UFLOAT, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED, mip_levels=1)
        if img.image == 0:
            return -1
        var bc6 = encode_bc6h_hdr(hdr_pixels, width, height)
        self.ensure_staging_capacity(len(bc6))
        _ = external_call["memcpy", Int](self.buf_staging.cpu_address, Int(bc6.unsafe_ptr()), len(bc6))
        _ = len(bc6)
        if not self.vk.begin_commands(0):
            self.vk.destroy_image(img)
            return -1
        self.vk.transition_image(0, img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        self.vk.upload_image_mip(0, img, self.buf_staging, 0, 0, width, height)
        self.vk.transition_image(0, img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        var f = self.vk.submit(0)
        if f != 0:
            _ = self.vk.wait(f)
        var slot = len(self.bindless_images)
        for ds_i in range(len(self.desc_sets)):
            self.vk.write_aaa_texture(self.desc_sets[ds_i], BIND_AAA_BINDLESS_TEX, slot, self.samp_linear_clamp, img.view)
        self.bindless_images.append(img^)
        return slot

    def add_material(mut self, mat: AaaMaterial) -> Int:
        var max_mats = min(SOTA_MAT_MAX_MATERIALS, self.buf_materials.size // (AAA_MATERIAL_WORDS * 4))  # sota:mat hook
        if len(self.materials) >= max_mats:
            return -1
        var id = len(self.materials)
        self.materials.append(mat.copy())
        var words = List[UInt32]()
        mat.append_words(words)
        if self.buf_materials.cpu_address != 0:
            _ = external_call["memcpy", Int](
                self.buf_materials.cpu_address + id * AAA_MATERIAL_WORDS * 4,
                Int(words.unsafe_ptr()),
                len(words) * 4,
            )
        _ = len(words)
        return id

    def add_glass_material(mut self, glass: AaaGlassParams) -> Int:
        """Adds a transparent, refractive material (`aaa_glass.mojo`): a smooth dielectric surface of base
        color `tint * opacity` that the glass pass makes see-through. `glass.material_id` is ignored.
        Returns the material id, or -1 when the material or glass table is full."""
        if len(self.glass_materials) >= AAA_MAX_GLASS_MATERIALS:
            return -1
        var mat = AaaMaterial(
            base_color=Vec3(glass.tint.x * glass.opacity, glass.tint.y * glass.opacity, glass.tint.z * glass.opacity),
            roughness=glass.roughness,
            metallic=0.0,
        )
        var id = self.add_material(mat)
        if id < 0:
            return -1
        var g = glass.copy()
        g.material_id = id
        self.glass_materials.append(g^)
        return id

    def update_glass_material(mut self, mat_id: Int, glass: AaaGlassParams) -> Bool:
        """Changes a glass material's optics (and its surface to match); False if `mat_id` is not glass."""
        for i in range(len(self.glass_materials)):
            if self.glass_materials[i].material_id == mat_id:
                var g = glass.copy()
                g.material_id = mat_id
                var mat = self.materials[mat_id].copy()
                mat.base_color = Vec3(g.tint.x * g.opacity, g.tint.y * g.opacity, g.tint.z * g.opacity)
                mat.roughness = g.roughness
                mat.metallic = Float32(0.0)
                self.update_material(mat_id, mat)
                self.glass_materials[i] = g^
                return True
        return False

    def is_glass_material(self, mat_id: Int) -> Bool:
        for g in self.glass_materials:
            if g.material_id == mat_id:
                return True
        return False

    def add_cel_material(mut self, mat: AaaMaterial, cel: AaaCelParams) -> Int:
        """Registers an `AaaMaterial` paired with `AaaCelParams` (`ILM`, `SSS`, `Inner-Ink`, `Hair-Halo`, `Rim`,
        and automatic hardware `CULL_MODE_FRONT` clip-space outline material allocation when `outline_width_px > 0`)."""
        if len(self.cel_materials) >= AAA_MAX_CEL_MATERIALS:
            return -1
        var id = self.add_material(mat)
        if id < 0:
            return -1
        var c = cel.copy()
        c.material_id = id
        if c.outline_width_px > Float32(0.001):
            var out_mat = AaaMaterial(
                base_color=c.outline_color,
                roughness=1.0,
                metallic=0.0,
            )
            out_mat.specular_factor = Float32(0.0)
            var out_id = self.add_material(out_mat)
            c.outline_material_id = out_id if out_id >= 0 else AAA_CEL_NO_OUTLINE_MAT
        else:
            c.outline_material_id = AAA_CEL_NO_OUTLINE_MAT
        self.cel_materials.append(c^)
        return id

    def update_cel_material(mut self, mat_id: Int, cel: AaaCelParams) -> Bool:
        """Updates an existing `AaaCelParams` entry in-place (preserving or allocating its `outline_material_id`)."""
        for i in range(len(self.cel_materials)):
            if self.cel_materials[i].material_id == mat_id:
                var c = cel.copy()
                c.material_id = mat_id
                var prev_out = self.cel_materials[i].outline_material_id
                if c.outline_width_px > Float32(0.001):
                    if prev_out >= 0 and prev_out < AAA_CEL_NO_OUTLINE_MAT:
                        var om = self.materials[prev_out].copy()
                        om.base_color = c.outline_color
                        self.update_material(prev_out, om)
                        c.outline_material_id = prev_out
                    else:
                        var om = AaaMaterial(base_color=c.outline_color, roughness=1.0, metallic=0.0)
                        om.specular_factor = Float32(0.0)
                        var out_id = self.add_material(om)
                        c.outline_material_id = out_id if out_id >= 0 else AAA_CEL_NO_OUTLINE_MAT
                else:
                    c.outline_material_id = AAA_CEL_NO_OUTLINE_MAT
                self.cel_materials[i] = c^
                return True
        # Not cel-shaded yet (an uploaded glTF material, textures and all): pair it with these
        # params, so a loaded model can join a toon scene.
        if mat_id < 0 or mat_id >= len(self.materials) or len(self.cel_materials) >= AAA_MAX_CEL_MATERIALS:
            return False
        var c = cel.copy()
        c.material_id = mat_id
        if c.outline_width_px > Float32(0.001):
            var om = AaaMaterial(base_color=c.outline_color, roughness=1.0, metallic=0.0)
            om.specular_factor = Float32(0.0)
            var out_id = self.add_material(om)
            c.outline_material_id = out_id if out_id >= 0 else AAA_CEL_NO_OUTLINE_MAT
        else:
            c.outline_material_id = AAA_CEL_NO_OUTLINE_MAT
        self.cel_materials.append(c^)
        return True

    def is_cel_outlined_material(self, mat_id: Int) -> Bool:
        for i in range(len(self.cel_materials)):
            ref c = self.cel_materials[i]
            if c.material_id == mat_id and c.outline_width_px > Float32(0.001) and c.outline_material_id < AAA_CEL_NO_OUTLINE_MAT:
                return True
        return False

    def add_vfx_material(mut self, mat: AaaMaterial, vfx: AaaStylizedVfxParams) -> Int:
        """Registers an eroding, UV-scrolling 3D anime VFX material (`AaaStylizedVfxParams`).
        Automatically enables Bucket-2 hardware alpha-cutout (`alpha_cutoff = 0.48`) so eroded holes
        cut out geometry silhouettes cleanly in `VisBuffer`!"""
        if len(self.vfx_materials) >= AAA_MAX_VFX_MATERIALS:
            return -1
        var m = mat.copy()
        if m.alpha_cutoff <= Float32(0.001):
            m.alpha_cutoff = Float32(0.48)
        var id = self.add_material(m)
        if id < 0:
            return -1
        var v = vfx.copy()
        v.material_id = id
        self.vfx_materials.append(v^)
        return id

    def update_vfx_material(mut self, mat_id: Int, vfx: AaaStylizedVfxParams) -> Bool:
        for i in range(len(self.vfx_materials)):
            if self.vfx_materials[i].material_id == mat_id:
                var v = vfx.copy()
                v.material_id = mat_id
                self.vfx_materials[i] = v^
                return True
        return False

    def submit_shockwave(mut self, ring: AaaShockwaveRing):
        """Submits a screen-space R/G/B chromatic-dispersion radial shockwave ring for this frame (`<= 4`)."""
        if len(self.frame_shockwaves) < AAA_MAX_SHOCKWAVES:
            self.frame_shockwaves.append(ring.copy())

    def set_impact_frame(mut self, params: AaaImpactFrameParams):
        """Configures high-contrast radial-streak anime **Manga Impact Frame** post-processing (`blend = 0` disables)."""
        self.impact_frame = params.copy()

    def clear_impact_frame(mut self):
        self.impact_frame = AaaImpactFrameParams()

    # --- sota:mat hook ---
    def sota_mat_set_layers(mut self, mat_id: Int, layers: SotaMatLayers):
        """Attaches the layered-model extension (anisotropy, rotation, thin-walled transmission + tint) to `mat_id`.
        Only read when `sota_mat.enabled` (`FLINT_SOTA_MAT=1`); materials without it keep their legacy parameters."""
        if mat_id < 0 or mat_id >= SOTA_MAT_MAX_MATERIALS:
            return
        sota_mat_write_layers(self.buf_materials.cpu_address, mat_id, layers)

    # --- end sota:mat hook ---
    def update_material(mut self, mat_id: Int, mat: AaaMaterial):
        if mat_id < 0 or mat_id >= len(self.materials):
            return
        self.materials[mat_id] = mat.copy()
        var words = List[UInt32]()
        mat.append_words(words)
        _ = external_call[
            "memcpy", Int
        ](self.buf_materials.cpu_address + mat_id * AAA_MATERIAL_WORDS * 4, Int(words.unsafe_ptr()), len(words) * 4)
        _ = len(words)

    def add_material_asset(mut self, mat: MaterialAsset) -> Int:
        """Uploads a unified `24`-word (`AAA_MATERIAL_WORDS`) `MaterialAsset` directly into `buf_materials` (`Binding 4`)."""
        return self.add_material(mat.to_aaa_material())

    def update_material_asset(mut self, mat_id: Int, mat: MaterialAsset):
        """Updates an existing material slot in `buf_materials` (`Binding 4`) from a `MaterialAsset`."""
        self.update_material(mat_id, mat.to_aaa_material())

    def evaluate_material_graph_to_bindless[
        NodeT: SurfaceMaterialNode
    ](
        mut self,
        node: NodeT,
        mat_id: Int,
        dim: Int = 32,
        time_sec: Float32 = 0.0,
        world_y_scale: Float32 = 24.0,
    ) -> Tuple[Int, Int]:
        """Compiles & dispatches `node` (`SurfaceMaterialNode`) on Vulkan against live `buf_materials` (`Binding 4`, stride `24`)
        and `BIND_AAA_BINDLESS_TEX`, bakes the GPU-evaluated surface response into bindless `BC7` Albedo + ORM textures,
        and binds them onto `mat_id` for 3D VisBuffer + Hardware RT shading! Returns `(albedo_slot, orm_slot)`."""
        if not self.active or mat_id < 0 or mat_id >= len(self.materials):
            return (-1, -1)
        var tex_dim = max(8, min(256, (dim + 3) & ~3))
        var spv = compile_surface_material_shader(node)
        var pipe = self.vk.create_aaa_compute_pipeline(self.pipe_layout, spv)
        if pipe == 0:
            return (-1, -1)

        var out_word_off = AAA_POST_BLOOM_BASE
        var out_words = tex_dim * tex_dim * 8
        if (out_word_off + out_words) * 4 > self.buf_post_scratch.size:
            self.vk.destroy_pipeline(pipe)
            return (-1, -1)

        if not self.vk.begin_commands(0):
            self.vk.destroy_pipeline(pipe)
            return (-1, -1)
        var push_mg = List[UInt32](length=16, fill=0)
        push_mg[0] = UInt32(tex_dim)
        push_mg[1] = UInt32(tex_dim)
        push_mg[2] = UInt32(mat_id)
        push_mg[3] = bitcast[DType.uint32](time_sec)
        push_mg[4] = UInt32(out_word_off)
        push_mg[5] = bitcast[DType.uint32](world_y_scale)
        self.vk.dispatch_aaa(
            0,
            pipe,
            self.pipe_layout,
            self.desc_sets[0],
            push_mg,
            (tex_dim + 7) // 8,
            (tex_dim + 7) // 8,
        )
        self.vk.memory_barrier2(
            0, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ)
        )
        var f_mg = self.vk.submit(0)
        if f_mg != 0:
            _ = self.vk.wait(f_mg)
        self.vk.destroy_pipeline(pipe)
        if f_mg == 0:
            return (-1, -1)

        var eval_f = List[Float32](length=out_words, fill=0.0)
        _ = external_call[
            "memcpy", Int
        ](Int(eval_f.unsafe_ptr()), self.buf_post_scratch.cpu_address + out_word_off * 4, out_words * 4)

        var alb_argb = List[UInt32](length=tex_dim * tex_dim, fill=0xFFFFFFFF)
        var orm_argb = List[UInt32](length=tex_dim * tex_dim, fill=0xFFFFFFFF)
        for p_i in range(tex_dim * tex_dim):
            var b_off = p_i * 8
            var rl = max(Float32(0.0), min(Float32(1.0), eval_f[b_off + 0]))
            var gl = max(Float32(0.0), min(Float32(1.0), eval_f[b_off + 1]))
            var bl = max(Float32(0.0), min(Float32(1.0), eval_f[b_off + 2]))
            var rgh = max(Float32(0.04), min(Float32(1.0), eval_f[b_off + 3]))
            var met = max(Float32(0.0), min(Float32(1.0), eval_f[b_off + 4]))
            var opa = max(Float32(0.0), min(Float32(1.0), eval_f[b_off + 6]))

            # Encode linear RGB -> sRGB 8-bit (`BC7_SRGB` hardware decode restores exact linear RGB!)
            var rs = UInt32(min(255, max(0, Int(sqrt(rl) * Float32(255.0) + Float32(0.5)))))
            var gs = UInt32(min(255, max(0, Int(sqrt(gl) * Float32(255.0) + Float32(0.5)))))
            var bs = UInt32(min(255, max(0, Int(sqrt(bl) * Float32(255.0) + Float32(0.5)))))
            var a8 = UInt32(min(255, max(0, Int(opa * Float32(255.0) + Float32(0.5)))))
            alb_argb[p_i] = (a8 << 24) | (rs << 16) | (gs << 8) | bs

            var rgh8 = UInt32(min(255, max(10, Int(rgh * Float32(255.0) + Float32(0.5)))))
            var met8 = UInt32(min(255, max(0, Int(met * Float32(255.0) + Float32(0.5)))))
            orm_argb[p_i] = UInt32(0xFF000000) | (UInt32(255) << 16) | (rgh8 << 8) | met8
        _ = len(eval_f)

        var alb_slot = self.upload_bc7_texture(alb_argb, tex_dim, tex_dim, srgb=True, generate_mips=True)
        var orm_slot = self.upload_bc7_texture(orm_argb, tex_dim, tex_dim, srgb=False, generate_mips=True)
        if alb_slot >= 0 and orm_slot >= 0:
            var upd = self.materials[mat_id].copy()
            upd.base_color = Vec3(1.0, 1.0, 1.0)
            upd.roughness = Float32(1.0)
            upd.metallic = Float32(1.0)
            upd.albedo_tex = alb_slot
            upd.orm_tex = orm_slot
            self.update_material(mat_id, upd)
        return (alb_slot, orm_slot)

    def add_mesh(mut self, mesh: NativeMesh, material_id: Int, build_lods: Bool = False) -> CookedMeshEntry:
        """Partitions `mesh` into 64v/124t GPU Meshlets (and optional `QEM` `LOD 0..3` DAG), uploads SSBOs, and builds `BLAS`!
        Reuses room freed by `free_mesh` when there is any. `entry.mesh_id < 0`: the geometry buffers are full."""
        if len(self.geom.holes) > 0:
            return self.new_mesh(mesh, material_id, build_lods)
        var v_words_before = len(self.geom.vertex_words)
        var i_words_before = len(self.geom.index_words)
        var m_words_before = len(self.geom.meshlet_words)

        var sota_m0 = sota_now()  # --- sota:perf hook ---
        # --- sota:geometry hook ---
        var entry: CookedMeshEntry
        if build_lods and self.sota_geom.cluster_lod_auto and self.sota_geom.active and len(mesh.indices) // 3 >= 256 and mesh.alpha_mode != 1:
            if self.sota_prep.has_sorted(self.sota_prep.current_mesh):  # --- sota:perf hook --- prepass vertex-pack order
                self.geom.sota_next_sorted = self.sota_prep.take_sorted(self.sota_prep.current_mesh)
                self.geom.sota_next_sorted_valid = True
            entry = self.add_mesh_cluster_dag_entry(mesh, material_id)
            self.geom.sota_next_sorted_valid = False  # --- sota:perf hook ---
        elif build_lods and self.sota_prep.has_lods(self.sota_prep.current_mesh):  # --- sota:perf hook --- prepass LOD chain
            entry = sota_append_mesh_prepared(self.geom, mesh, material_id, self.sota_prep, self.sota_prep.current_mesh)
        else:
            entry = self.geom.append_mesh(mesh, material_id, build_lods=build_lods)
        # --- end sota:geometry hook ---
        var sota_m1 = sota_now()  # --- sota:perf hook ---
        self.sota_upload.mesh_cook_s += sota_m1 - sota_m0  # --- sota:perf hook ---
        _ = self._sota_ensure_geometry_capacity()  # sota:geometry hook: grow buffers to the scene

        var vb_start = v_words_before * 4
        var vb_len = (len(self.geom.vertex_words) - v_words_before) * 4
        if vb_start + vb_len <= self.buf_vertices.size and vb_len > 0:
            _ = external_call["memcpy", Int](
                self.buf_vertices.cpu_address + vb_start,
                Int(self.geom.vertex_words.unsafe_ptr()) + vb_start,
                vb_len,
            )

        var ib_start = i_words_before * 4
        var ib_len = (len(self.geom.index_words) - i_words_before) * 4
        if ib_start + ib_len <= self.buf_indices.size and ib_len > 0:
            _ = external_call["memcpy", Int](
                self.buf_indices.cpu_address + ib_start,
                Int(self.geom.index_words.unsafe_ptr()) + ib_start,
                ib_len,
            )

        var mb_start = m_words_before * 4
        var mb_len = (len(self.geom.meshlet_words) - m_words_before) * 4
        if mb_start + mb_len <= self.buf_meshlets.size and mb_len > 0:
            _ = external_call["memcpy", Int](
                self.buf_meshlets.cpu_address + mb_start,
                Int(self.geom.meshlet_words.unsafe_ptr()) + mb_start,
                mb_len,
            )

        var sota_m2 = sota_now()  # --- sota:perf hook ---
        self.sota_upload.mesh_copy_s += sota_m2 - sota_m1  # --- sota:perf hook ---
        var is_am_mesh = (mesh.alpha_mode == 1) or (
            material_id >= 0 and material_id < len(self.materials) and self.materials[material_id].is_alpha_masked()
        )
        if self.vk.has_ray_query and entry.triangle_count > 0 and entry.vertex_count > 0:
            # Note: `buf_indices` stores global vertex indices (`entry.first_vertex + local_v`) with `firstVertex = 0`.
            var v_addr = self.vk.buffer_device_address(self.buf_vertices)
            var i_addr = self.vk.buffer_device_address(self.buf_indices) + UInt64(entry.first_index * 4)
            var max_v = max(0, entry.first_vertex + entry.vertex_count - 1)
            if self.vk.begin_commands(0):
                var blas = build_blas_triangles_ex(
                    self.vk,
                    0,
                    v_addr,
                    AAA_VERTEX_BYTES,
                    max_v,
                    i_addr,
                    entry.triangle_count,
                    self.buf_as_scratch,
                    opaque=not is_am_mesh,
                )
                self.vk.memory_barrier2(
                    0,
                    PIPELINE_AS_BUILD_KHR,
                    Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR),
                    PIPELINE_AS_BUILD_KHR | PIPELINE_COMPUTE,
                    Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
                )
                var f = self.vk.submit(0)
                if f != 0:
                    _ = self.vk.wait(f)
                self.mesh_blas.append(blas^)
                self.sota_upload.blas_builds += 1  # --- sota:perf hook ---
                self.sota_upload.submits += 1  # --- sota:perf hook ---
            else:
                self.mesh_blas.append(VulkanAccelStruct())
        else:
            self.mesh_blas.append(VulkanAccelStruct())

        self.sota_upload.blas_s += sota_now() - sota_m2  # --- sota:perf hook ---
        self.sota_upload.meshes += 1  # --- sota:perf hook ---
        self.skinned_mesh_flags.append(False)
        self.skinned_mesh_bind_slots.append(0)
        self.mesh_double_sided_flags.append(mesh.double_sided or is_am_mesh)
        self.mesh_alpha_masked_flags.append(is_am_mesh)
        self.vsm.invalidate_all()
        if (len(self.geom.vertex_words) * 4 > self.buf_vertices.size or len(self.geom.index_words) * 4 > self.buf_indices.size
                or len(self.geom.meshlet_words) * 4 > self.buf_meshlets.size):
            entry.mesh_id = -1            # did not fit on the GPU: never draw it
            # --- sota:perf hook --- make the overflow visible instead of silently dropping geometry
            if self.geometry_overflow_meshes == 0:
                print("[Flint AAA] WARNING: geometry pools full (geometry_mb=" + String(self.geometry_mb)
                      + "): buffer growth failed, meshes from here on are not drawn")
            self.geometry_overflow_meshes += 1
            self._sota_log_once("geometry: mesh did not fit on the GPU (buffer growth failed); it will not be drawn")  # sota:geometry hook
        # --- sota:geometry hook --- draws and the visbuffer carry a 24-bit global meshlet id and a 24-bit vertex index
        if entry.mesh_id >= 0 and (len(self.geom.meshlet_words) // 12 > SOTA_MAX_GLOBAL_MESHLETS
                                   or len(self.geom.vertex_words) // 12 > SOTA_MAX_GLOBAL_VERTICES):
            entry.mesh_id = -1
            self.sota_geom.meshlet_id_overflows += 1
            self._sota_log_once("geometry: more than " + String(SOTA_MAX_GLOBAL_MESHLETS) + " meshlets or "
                                + String(SOTA_MAX_GLOBAL_VERTICES) + " vertices do not fit the 24-bit draw ids; further meshes are not drawn")
        # --- end sota:geometry hook ---
        return entry^

    # --- sota:geometry hook ---
    def sota_debug_pad_meshlets(mut self, count: Int) -> Int:
        """Test hook: appends `count` copies of the last meshlet record (never referenced by any mesh), so the next
        uploaded mesh gets global meshlet ids past `count`. Returns the new global meshlet count (-1 on failure)."""
        var n0 = len(self.geom.meshlets)
        if n0 == 0 or count <= 0 or len(self.geom.meshlet_words) < 12:
            return -1
        var m_words_before = len(self.geom.meshlet_words)
        var last = List[UInt32]()
        for k in range(12):
            last.append(self.geom.meshlet_words[m_words_before - 12 + k])
        var rec = self.geom.meshlets[n0 - 1].copy()
        for _ in range(count):
            for k in range(12):
                self.geom.meshlet_words.append(last[k])
            self.geom.meshlets.append(rec.copy())
        if not self._sota_ensure_geometry_capacity():
            return -1
        var mb_start = m_words_before * 4
        var mb_len = (len(self.geom.meshlet_words) - m_words_before) * 4
        if mb_start + mb_len > self.buf_meshlets.size:
            return -1
        _ = external_call["memcpy", Int](
            self.buf_meshlets.cpu_address + mb_start, Int(self.geom.meshlet_words.unsafe_ptr()) + mb_start, mb_len
        )
        return len(self.geom.meshlets)

    def _sota_log_once(mut self, msg: String):
        """Prints a renderer warning once per distinct message (no silent failures, no log spam)."""
        for m in self.sota_geom.logged:
            if m == msg:
                return
        self.sota_geom.logged.append(msg)
        print("[flint render] " + msg)

    def _sota_frame_fail(mut self, reason: String) -> Bool:
        self.sota_geom.frame_failures += 1
        self.sota_geom.last_frame_error = reason
        self._sota_log_once("render_frame failed: " + reason)
        return False

    def _sota_ensure_geometry_capacity(mut self) -> Bool:
        """Grows the vertex / index / meshlet buffers (and the DAG records) to fit the CPU geometry pool:
        waits for the GPU, copies the old contents, rebinds every descriptor set that holds them. Logs and
        returns False when the allocation fails (the caller then marks the mesh undrawable)."""
        var need_v = len(self.geom.vertex_words) * 4
        var need_i = len(self.geom.index_words) * 4
        var need_m = len(self.geom.meshlet_words) * 4
        if need_v <= self.buf_vertices.size and need_i <= self.buf_indices.size and need_m <= self.buf_meshlets.size:
            return True
        if not self.active:
            return False
        self.vk.wait_idle()
        var bda_flags = BUFFER_USAGE_DEVICE_ADDRESS | BUFFER_USAGE_AS_BUILD_INPUT_KHR
        if need_v > self.buf_vertices.size:
            var sz = sota_geometry_grow_size(self.buf_vertices.size, need_v)
            var nb = self.vk.create_buffer_bda(sz, extra_usage=bda_flags, host_visible=True)
            if nb.size == 0 or nb.cpu_address == 0:
                self.vk.destroy_buffer(nb)
                self._sota_log_once("geometry: out of memory growing the vertex buffer to " + String(sz // 1048576) + " MB")
                return False
            _ = external_call["memcpy", Int](nb.cpu_address, self.buf_vertices.cpu_address, self.buf_vertices.size)
            self.vk.destroy_buffer(self.buf_vertices)
            self.buf_vertices = nb^
        if need_i > self.buf_indices.size:
            var sz = sota_geometry_grow_size(self.buf_indices.size, need_i)
            var nb = self.vk.create_buffer_bda(sz, extra_usage=bda_flags, host_visible=True)
            if nb.size == 0 or nb.cpu_address == 0:
                self.vk.destroy_buffer(nb)
                self._sota_log_once("geometry: out of memory growing the index buffer to " + String(sz // 1048576) + " MB")
                return False
            _ = external_call["memcpy", Int](nb.cpu_address, self.buf_indices.cpu_address, self.buf_indices.size)
            self.vk.destroy_buffer(self.buf_indices)
            self.buf_indices = nb^
        if need_m > self.buf_meshlets.size:
            var sz = sota_geometry_grow_size(self.buf_meshlets.size, need_m)
            var nb = self.vk.create_buffer(sz, local=True)
            if nb.size == 0 or nb.cpu_address == 0:
                self.vk.destroy_buffer(nb)
                self._sota_log_once("geometry: out of memory growing the meshlet buffer to " + String(sz // 1048576) + " MB")
                return False
            _ = external_call["memcpy", Int](nb.cpu_address, self.buf_meshlets.cpu_address, self.buf_meshlets.size)
            self.vk.destroy_buffer(self.buf_meshlets)
            self.buf_meshlets = nb^
        for slot in range(FRAMES_IN_FLIGHT):
            var sets = List[Int]()
            sets.append(self.desc_sets[slot])
            sets.append(self.desc_sets_cull_p1[slot])
            sets.append(self.desc_sets_cull_p2[slot])
            for ds in sets:
                self.vk.write_descriptor(ds, BIND_AAA_VERTICES, self.buf_vertices)
                self.vk.write_descriptor(ds, BIND_AAA_INDICES, self.buf_indices)
                self.vk.write_descriptor(ds, BIND_AAA_MESHLETS, self.buf_meshlets)
        var ml_cap = self.buf_meshlets.size // (AAA_MESHLET_WORDS * 4)
        if not self.sota_geom.rebind_meshlets(self.vk, self.buf_meshlets, ml_cap):
            self._sota_log_once("geometry: could not grow the DAG record buffer; cluster LOD falls back to LOD0 for new meshes")
        self.sota_vt.rebind_geometry(self.vk, self.buf_vertices, self.buf_indices, self.buf_meshlets)
        self._sota_rebind_e_routes()
        self.sota_geom.grow_events += 1
        print("[flint render] geometry buffers grown: vertices " + String(self.buf_vertices.size // 1048576) + " MB, indices "
              + String(self.buf_indices.size // 1048576) + " MB, meshlets " + String(self.buf_meshlets.size // 1048576) + " MB")
        return True

    def _sota_rebind_e_routes(mut self):
        """Area E's GPU-driven alpha-mask / outline cull sets hold the meshlet, candidate and DAG buffers that
        geometry / draw growth replaces: point them at the live buffers (a stale set reads freed memory)."""
        for ds in self.sota_e_gpu.sets_alpha:
            self.vk.write_descriptor(ds, SOTA_BIND_MESHLETS, self.buf_meshlets)
            self.vk.write_descriptor(ds, SOTA_BIND_CANDIDATES, self.buf_occluded_candidates)
            self.vk.write_descriptor(ds, SOTA_BIND_DAG, self.sota_geom.buf_dag)
        for ds in self.sota_e_gpu.sets_outline:
            self.vk.write_descriptor(ds, SOTA_BIND_MESHLETS, self.buf_meshlets)
            self.vk.write_descriptor(ds, SOTA_BIND_CANDIDATES, self.buf_occluded_candidates)
            self.vk.write_descriptor(ds, SOTA_BIND_DAG, self.sota_geom.buf_dag)

    # --- sota:perf hook --- alpha-mask / outline indirect rings grow with the scene (fixed 16384 before)
    def _sota_check_route_overflow(mut self, slot: Int):
        """The GPU routes count every visible meshlet in word 0 (even past capacity) and the CPU packers
        report what they wanted: grow both rings when last frame's need passed the capacity. Runs after the
        slot's fence has signalled, so word 0 is that slot's last frame."""
        var need = self.sota_route_cpu_need
        self.sota_route_cpu_need = 0
        if slot < len(self.ring_alphamask_indirect) and self.ring_alphamask_indirect[slot].cpu_address != 0:
            need = max(need, Int(get32(self.ring_alphamask_indirect[slot].cpu_address)))
        if slot < len(self.ring_outline_indirect) and self.ring_outline_indirect[slot].cpu_address != 0:
            need = max(need, Int(get32(self.ring_outline_indirect[slot].cpu_address)))
        if need <= self.sota_route_draws or need > sota_route_draws_limit():
            return
        var new_cap = sota_route_capacity_for(need)
        self._sota_log_once("visible alpha-masked / outlined meshlets exceeded " + String(self.sota_route_draws)
                            + " (needed " + String(need) + "); growing the route rings")
        _ = self._sota_grow_route_draws(new_cap)

    def _sota_grow_route_draws(mut self, new_cap: Int) -> Bool:
        if new_cap <= self.sota_route_draws or not self.active:
            return False
        self.vk.wait_idle()
        var n_am = List[GpuBuffer]()
        var n_out = List[GpuBuffer]()
        var ok = True
        for _ in range(FRAMES_IN_FLIGHT):
            var a = self.vk.create_buffer(64 + new_cap * 20, local=True)
            var b = self.vk.create_buffer(64 + new_cap * 20, local=True)
            ok = ok and a.size > 0 and b.size > 0 and a.cpu_address != 0 and b.cpu_address != 0
            if a.cpu_address != 0:
                put32(a.cpu_address, 0)
            if b.cpu_address != 0:
                put32(b.cpu_address, 0)
            n_am.append(a^)
            n_out.append(b^)
        if not ok:
            for b in n_am:
                self.vk.destroy_buffer(b)
            for b in n_out:
                self.vk.destroy_buffer(b)
            self._sota_log_once("out of memory growing the alpha-mask / outline rings to " + String(new_cap))
            return False
        for i in range(len(self.ring_alphamask_indirect)):
            self.vk.destroy_buffer(self.ring_alphamask_indirect[i])
        for i in range(len(self.ring_outline_indirect)):
            self.vk.destroy_buffer(self.ring_outline_indirect[i])
        self.ring_alphamask_indirect = n_am^
        self.ring_outline_indirect = n_out^
        self.buf_alphamask_indirect = self.ring_alphamask_indirect[0].copy()
        self.buf_outline_indirect = self.ring_outline_indirect[0].copy()
        self.sota_route_draws = new_cap
        for f in range(len(self.sota_e_gpu.sets_alpha)):
            if f < len(self.ring_alphamask_indirect):
                self.vk.write_descriptor(self.sota_e_gpu.sets_alpha[f], SOTA_BIND_INDIRECT, self.ring_alphamask_indirect[f])
        for f in range(len(self.sota_e_gpu.sets_outline)):
            if f < len(self.ring_outline_indirect):
                self.vk.write_descriptor(self.sota_e_gpu.sets_outline[f], SOTA_BIND_INDIRECT, self.ring_outline_indirect[f])
        self.sota_route_grow_events += 1
        print("[flint render] alpha-mask / outline route capacity grown to " + String(new_cap))
        return True
    # --- end sota:perf hook ---

    def _sota_grow_draws(mut self, new_max: Int) -> Bool:
        """Grows every buffer sized by `max_draws` (indirect rings, phase-2 indirect, occluded candidates,
        VSM indirect) and rebinds them. Old contents are per-frame scratch, so nothing is copied."""
        if new_max <= self.max_draws or not self.active:
            return False
        self.vk.wait_idle()
        var n_vsm_cmds = max(new_max, self.sota_shadow.vsm_cmd_capacity)  # --- sota:shadow hook --- never shrink
        var n_vsm = self.vk.create_buffer(64 + n_vsm_cmds * 16, local=True)
        var n_p2 = self.vk.create_buffer(64 + new_max * 20, local=True)
        var n_cand = self.vk.create_buffer(64 + new_max * 8, local=True)
        var n_ring = List[GpuBuffer]()
        var ok = n_vsm.size > 0 and n_p2.size > 0 and n_cand.size > 0
        for _ in range(FRAMES_IN_FLIGHT):
            var b = self.vk.create_buffer(64 + new_max * 20, local=True)
            ok = ok and b.size > 0
            n_ring.append(b^)
        if not ok:
            self.vk.destroy_buffer(n_vsm)
            self.vk.destroy_buffer(n_p2)
            self.vk.destroy_buffer(n_cand)
            for b in n_ring:
                self.vk.destroy_buffer(b)
            self._sota_log_once("out of memory growing max_draws to " + String(new_max) + "; extra meshlets stay dropped (counted in sota_geom.draws_dropped_total)")
            return False
        self.vk.destroy_buffer(self.buf_vsm_indirect)
        self.vk.destroy_buffer(self.buf_indirect_phase2)
        self.vk.destroy_buffer(self.buf_occluded_candidates)
        for i in range(FRAMES_IN_FLIGHT):
            self.vk.destroy_buffer(self.ring_indirect[i])
        self.buf_vsm_indirect = n_vsm^
        self.sota_shadow.vsm_cmd_capacity = n_vsm_cmds  # --- sota:shadow hook ---
        self.buf_indirect_phase2 = n_p2^
        self.buf_occluded_candidates = n_cand^
        self.ring_indirect = n_ring^
        self.buf_indirect = self.ring_indirect[0].copy()
        self.max_draws = new_max
        for slot in range(FRAMES_IN_FLIGHT):
            self.vk.write_descriptor(self.desc_sets[slot], BIND_AAA_INDIRECT, self.ring_indirect[slot])
            self.vk.write_descriptor(self.desc_sets_cull_p1[slot], BIND_AAA_INDIRECT, self.ring_indirect[slot])
            self.vk.write_descriptor(self.desc_sets_cull_p1[slot], BIND_AAA_CANDIDATES, self.buf_occluded_candidates)
            self.vk.write_descriptor(self.desc_sets_cull_p2[slot], BIND_AAA_INDIRECT, self.buf_indirect_phase2)
            self.vk.write_descriptor(self.desc_sets_cull_p2[slot], BIND_AAA_CANDIDATES, self.buf_occluded_candidates)
        self.sota_geom.rebind_draws(self.vk, self.ring_indirect, self.buf_indirect_phase2, self.buf_occluded_candidates)
        self._sota_rebind_e_routes()
        self.sota_geom.draw_grow_events += 1
        print("[flint render] max_draws grown to " + String(new_max))
        return True

    # --- sota:perf hook --- parallel upload prepass (sota_perf_upload.mojo)
    def _sota_prepare_upload(mut self, model: NativeModel, cap_dim: Int, build_lods: Bool, upload_skinned: Bool):
        """Encodes every texture role and builds every cluster DAG that the serial walk of
        `upload_native_model` is about to need, on a JobPool. The decisions mirror the walk; anything not
        prepared (VT albedo, skinned meshes, legacy-LOD and alpha-masked meshes) still runs inline."""
        self.sota_prep.clear()
        if not self.active or not sota_upload_parallel_enabled():
            return
        if not self.sota_upload_pool_started:
            self.sota_upload_pool_started = True
            var n = sota_upload_worker_count()
            if n > 0:
                _ = self.sota_upload_pool.start(n)
        if not self.sota_upload_pool.ready():
            return
        var n_tex = len(model.textures)
        var want_alb = List[Bool](length=n_tex, fill=False)
        var want_nrm = List[Bool](length=n_tex, fill=False)
        var want_orm = List[Bool](length=n_tex, fill=False)
        var vt_alb = List[Bool](length=n_tex, fill=False)
        var want_dag = List[Bool](length=len(model.meshes), fill=False)
        var want_lods = List[Bool](length=len(model.meshes), fill=False)
        var dag_ok = build_lods and self.sota_geom.cluster_lod_auto and self.sota_geom.active and len(self.geom.holes) == 0
        for m_idx in range(len(model.meshes)):
            ref m = model.meshes[m_idx]
            if m.texture >= 0 and m.texture < n_tex:
                ref tx = model.textures[m.texture]
                if (m.alpha_mode != 1 and self.sota_vt.wants_model_texture(tx.width, tx.height)
                        and len(tx.pixels) >= tx.width * tx.height):
                    vt_alb[m.texture] = True  # may stream through VT: leave it to the walk
                else:
                    want_alb[m.texture] = True
            if m.normal_orm_texture >= 0 and m.normal_orm_texture < n_tex:
                want_nrm[m.normal_orm_texture] = True
            if m.orm_texture >= 0 and m.orm_texture < n_tex:
                want_orm[m.orm_texture] = True
            var mv_cnt = m.vertex_count()
            var skinned = upload_skinned and mv_cnt > 0 and len(m.joints) >= mv_cnt * 4 and len(m.weights) >= mv_cnt * 4
            if dag_ok and not skinned and len(m.indices) // 3 >= 256 and m.alpha_mode != 1:
                want_dag[m_idx] = True
            elif build_lods and not skinned and len(self.geom.holes) == 0 and len(m.indices) // 3 >= 1024 and mv_cnt > 0:
                want_lods[m_idx] = True  # legacy QEM LOD chain (alpha-masked and small meshes)
        for t in range(n_tex):
            if vt_alb[t]:
                want_alb[t] = False
        var mt_addr = 0
        if not self.sota_geom.dag_pool_started and self.sota_geom.dag_build_threads != 0:
            var nd = default_worker_count() if self.sota_geom.dag_build_threads < 0 else self.sota_geom.dag_build_threads
            if nd > 0:
                _ = self.sota_geom.dag_pool.start(nd)
            self.sota_geom.dag_pool_started = True
        if self.sota_geom.dag_pool.ready():
            mt_addr = Int(Pointer(to=self.sota_geom.dag_pool))
        var hits = sota_prepare_model(
            model, self.sota_prep, self.sota_upload_pool, want_alb, want_nrm, want_orm, want_dag, want_lods, cap_dim,
            self.sota_geom.dag_cache_dir, mt_addr,
        )
        self.sota_upload.prepare_s += self.sota_prep.prepare_s
        self.sota_upload.prepared_jobs += self.sota_prep.jobs
        self.sota_upload.prepared_workers = self.sota_prep.workers
        self.sota_upload.dag_cache_hits += hits
        self.sota_prep.current_mesh = -1

    def _sota_upload_prepared_texture(mut self, role: Int, t: Int) -> Int:
        """GPU half of `upload_bc7_texture` / `upload_bc5_normal_texture` for a prepass-encoded mip chain."""
        var prep = self.sota_prep.take_texture(role, t)
        if len(self.bindless_images) >= AAA_MAX_BINDLESS or len(prep.blobs) == 0:
            return -1
        var sota_t1 = sota_now()
        var mips = len(prep.blobs)
        var fmt = FORMAT_BC5_UNORM if prep.bc5 else (FORMAT_BC7_SRGB if prep.srgb else FORMAT_BC7_UNORM)
        var img = self.vk.create_image_2d(prep.widths[0], prep.heights[0], fmt, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED, mip_levels=mips)
        if img.image == 0:
            return -1
        var mip_offsets = List[Int]()
        var st_off = 0
        for m in range(mips):
            var n = len(prep.blobs[m])
            self.ensure_staging_capacity(st_off + n)
            _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_off, Int(prep.blobs[m].unsafe_ptr()), n)
            mip_offsets.append(st_off)
            st_off += (n + 15) & ~15
        if not self.vk.begin_commands(0):
            self.vk.destroy_image(img)
            return -1
        self.vk.transition_image(0, img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        for m in range(mips):
            self.vk.upload_image_mip(0, img, self.buf_staging, mip_offsets[m], m, prep.widths[m], prep.heights[m])
        self.vk.transition_image(0, img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        var f = self.vk.submit(0)
        if f != 0:
            _ = self.vk.wait(f)
        _ = len(prep.blobs)
        var slot = len(self.bindless_images)
        for ds_i in range(len(self.desc_sets)):
            self.vk.write_aaa_texture(self.desc_sets[ds_i], BIND_AAA_BINDLESS_TEX, slot, self.samp_aniso16, img.view)
        self.bindless_images.append(img^)
        self.sota_upload.tex_gpu_s += sota_now() - sota_t1
        self.sota_upload.textures += 1
        self.sota_upload.submits += 1
        return slot
    # --- end sota:perf hook ---

    def add_virtual_texture(mut self, pixels_argb: List[UInt32], dim: Int) -> Int:
        """Registers a square power-of-two (128..16384) sRGB ARGB image as a streamed virtual texture (only the
        pages the camera needs become resident; see sota_geometry_vt.mojo). Returns the VT index or -1."""
        return self.sota_vt.add_virtual_texture(
            self.vk, pixels_argb, dim, self.desc_sets, self.samp_nearest, self.samp_linear_clamp, self.bindless_images,
            AAA_MAX_BINDLESS,
        )

    def add_material_vt(mut self, mat: AaaMaterial, vt: Int) -> Int:
        """`mat` with its albedo from virtual texture `vt`. Base colour becomes the texture mean (used by RT/GI
        hits); `update_material` on the returned id drops the VT binding."""
        if vt < 0 or vt >= self.sota_vt.count():
            return -1
        var m = mat.copy()
        var mean = self.sota_vt.mean_rgb[vt]
        m.base_color = Vec3(mean[0], mean[1], mean[2])
        m.iridescence = Float32(0.0)
        var id = self.add_material(m)
        if id >= 0 and self.buf_materials.cpu_address != 0:
            put32(self.buf_materials.cpu_address + (id * AAA_MATERIAL_WORDS + 15) * 4, bitcast[DType.uint32](self.sota_vt.material_code(vt)))
        return id

    def add_mesh_cluster_dag_entry(mut self, mesh: NativeMesh, material_id: Int) -> CookedMeshEntry:
        """Cooks `mesh` as a continuous cluster-LOD DAG into the CPU pool and publishes its DAG records
        (the caller, `add_mesh`, uploads the pool range and builds the LOD0 BLAS)."""
        var sota_d0 = sota_now()  # --- sota:perf hook ---
        var dag: ClusterDag
        if self.sota_prep.has_dag(self.sota_prep.current_mesh):  # built by the parallel upload prepass
            dag = self.sota_prep.take_dag(self.sota_prep.current_mesh)
        else:
            dag = self.sota_geom.build_dag(mesh)       # .flpk cache + JobPool build
        self.sota_upload.dag_build_s += sota_now() - sota_d0  # --- sota:perf hook ---
        var cooked = cook_cluster_dag_into_pool(self.geom, mesh, material_id, dag)
        self.sota_geom.write_dag_records(cooked[0].first_meshlet, cooked[1])
        return cooked[0].copy()

    def add_mesh_cluster_lod(mut self, mesh: NativeMesh, material_id: Int) -> CookedMeshEntry:
        """Explicit cluster-LOD DAG mesh (independent of `sota_geom.cluster_lod_auto`)."""
        var keep = self.sota_geom.cluster_lod_auto
        self.sota_geom.cluster_lod_auto = True
        var e = self.add_mesh(mesh, material_id, build_lods=True)
        self.sota_geom.cluster_lod_auto = keep
        return e^
    # --- end sota:geometry hook ---

    def set_wind(mut self, dir_x: Float32, dir_z: Float32, sway_m: Float32, hz: Float32 = 0.6, gust: Float32 = 0.3):
        """The global wind for draws with wind > 0: direction (xz), how far tips sway (m), how fast
        (Hz), how gusty (0..1). Vertices sway by their uv.y (0 at the root .. 1 at the tip)."""
        var L = sqrt(dir_x * dir_x + dir_z * dir_z)
        self.wind_dir_x = dir_x / L if L > Float32(1e-6) else Float32(1.0)
        self.wind_dir_z = dir_z / L if L > Float32(1e-6) else Float32(0.0)
        self.wind_sway = max(Float32(0.0), sway_m)
        self.wind_hz = max(Float32(0.0), hz)
        self.wind_gust = min(Float32(1.0), max(Float32(0.0), gust))

    # -- freeing, replacing and streaming meshes ---------------------------------------------------------
    def _upload_slot(mut self, sl: GeomSlot) -> Bool:
        """Copy one mesh's room (vertices, index words, meshlets) from the CPU pool to the GPU buffers."""
        var vb = sl.first_vertex * AAA_VERTEX_WORDS * 4
        var vn = sl.vertex_cap * AAA_VERTEX_WORDS * 4
        var ib = sl.first_index * 4
        var in_ = sl.index_cap * 4
        var mb = sl.first_meshlet * AAA_MESHLET_WORDS * 4
        var mn = sl.meshlet_cap * AAA_MESHLET_WORDS * 4
        if vb + vn > self.buf_vertices.size or ib + in_ > self.buf_indices.size or mb + mn > self.buf_meshlets.size:
            return False
        if vn > 0:
            _ = external_call["memcpy", Int](self.buf_vertices.cpu_address + vb, Int(self.geom.vertex_words.unsafe_ptr()) + vb, vn)
        if in_ > 0:
            _ = external_call["memcpy", Int](self.buf_indices.cpu_address + ib, Int(self.geom.index_words.unsafe_ptr()) + ib, in_)
        if mn > 0:
            _ = external_call["memcpy", Int](self.buf_meshlets.cpu_address + mb, Int(self.geom.meshlet_words.unsafe_ptr()) + mb, mn)
        return True

    def _rebuild_blas(mut self, mesh_id: Int, opaque: Bool):
        if mesh_id < 0 or mesh_id >= len(self.mesh_blas):
            return
        self.vk.destroy_accel_struct(self.mesh_blas[mesh_id])
        self.mesh_blas[mesh_id] = VulkanAccelStruct()
        ref entry = self.geom.meshes[mesh_id]
        if not self.vk.has_ray_query or entry.triangle_count <= 0 or entry.vertex_count <= 0:
            return
        var v_addr = self.vk.buffer_device_address(self.buf_vertices)
        var i_addr = self.vk.buffer_device_address(self.buf_indices) + UInt64(entry.first_index * 4)
        var max_v = max(0, entry.first_vertex + entry.vertex_count - 1)
        if self.vk.begin_commands(0):
            var blas = build_blas_triangles_ex(
                self.vk, 0, v_addr, AAA_VERTEX_BYTES, max_v, i_addr, entry.triangle_count, self.buf_as_scratch, opaque=opaque
            )
            self.vk.memory_barrier2(
                0,
                PIPELINE_AS_BUILD_KHR,
                Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR),
                PIPELINE_AS_BUILD_KHR | PIPELINE_COMPUTE,
                Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
            )
            var f = self.vk.submit(0)
            if f != 0:
                _ = self.vk.wait(f)
            self.mesh_blas[mesh_id] = blas^

    def replace_mesh(
        mut self, mesh_id: Int, mesh: NativeMesh, material_id: Int, build_lods: Bool = False
    ) -> CookedMeshEntry:
        """Swap mesh `mesh_id`'s geometry for `mesh` (same id, so draws keep working): in place when
        it fits, else in room another mesh freed, else at the end. For streamed terrain tiles, edited
        level geometry, trails. Waits for the GPU first. `mesh_id < 0` in the result: no room."""
        var bad = CookedMeshEntry()
        bad.mesh_id = -1
        if mesh_id < 0 or mesh_id >= len(self.geom.meshes) or (mesh_id < len(self.skinned_mesh_flags) and self.skinned_mesh_flags[mesh_id]):
            return bad^
        self.vk.wait_idle()
        # --- sota:geometry hook ---
        var old_sl = self.geom.mesh_slot(mesh_id)
        self.sota_geom.clear_dag_range(old_sl.first_meshlet, old_sl.meshlet_cap)
        # --- end sota:geometry hook ---
        var res = self.geom.replace_mesh(mesh_id, mesh, material_id, build_lods)
        var sl = res[1]
        if not sl.empty() and not self._upload_slot(sl):
            _ = self.geom.free_mesh(mesh_id)          # no room on the GPU: leave it empty
            self._rebuild_blas(mesh_id, True)
            return bad^
        for h in self.geom.holes:                       # freed room holds meshlets that cover nothing
            _ = self._upload_slot(GeomSlot(h.first_vertex, 0, h.first_index, 0, h.first_meshlet, h.meshlet_cap))
        var is_am = (mesh.alpha_mode == 1) or (
            material_id >= 0 and material_id < len(self.materials) and self.materials[material_id].is_alpha_masked()
        )
        self.mesh_double_sided_flags[mesh_id] = mesh.double_sided or is_am
        self.mesh_alpha_masked_flags[mesh_id] = is_am
        self._rebuild_blas(mesh_id, not is_am)
        self.vsm.invalidate_all()
        return res[0].copy()

    def free_mesh(mut self, mesh_id: Int) -> Bool:
        """Give mesh `mesh_id`'s GPU room back for later meshes; it draws nothing until replaced."""
        if mesh_id < 0 or mesh_id >= len(self.geom.meshes):
            return False
        self.vk.wait_idle()
        var sl = self.geom.free_mesh(mesh_id)
        if sl.empty():
            return False
        self.sota_geom.clear_dag_range(sl.first_meshlet, sl.meshlet_cap)  # sota:geometry hook
        _ = self._upload_slot(GeomSlot(sl.first_vertex, 0, sl.first_index, 0, sl.first_meshlet, sl.meshlet_cap))
        self._rebuild_blas(mesh_id, True)
        self.vsm.invalidate_all()
        return True

    def new_mesh(mut self, mesh: NativeMesh, material_id: Int, build_lods: Bool = False) -> CookedMeshEntry:
        """A new mesh id, its geometry put in freed room when some fits."""
        var mesh_id = self.geom.reserve_mesh_id()
        self.mesh_blas.append(VulkanAccelStruct())
        self.skinned_mesh_flags.append(False)
        self.skinned_mesh_bind_slots.append(0)
        self.mesh_double_sided_flags.append(False)
        self.mesh_alpha_masked_flags.append(False)
        return self.replace_mesh(mesh_id, mesh, material_id, build_lods)

    def add_splat_mesh(
        mut self,
        mesh: NativeMesh,
        material_id: Int,
        splat_weights_u24: List[UInt32] = List[UInt32](),
        build_lods: Bool = False,
    ) -> CookedMeshEntry:
        """Cooks a 4-layer height-modulated PBR terrain splat mesh (`vertex_words[11]` bit 1 = 1 + 24-bit weights),
        uploads GPU Meshlet/Vertex/Index SSBOs, and builds its hardware `BLAS`!"""
        var v_words_before = len(self.geom.vertex_words)
        var i_words_before = len(self.geom.index_words)
        var m_words_before = len(self.geom.meshlet_words)

        var entry = self.geom.append_splat_mesh(mesh, material_id, splat_weights_u24, build_lods=build_lods)
        _ = self._sota_ensure_geometry_capacity()  # sota:geometry hook: grow buffers to the scene

        var vb_start = v_words_before * 4
        var vb_len = (len(self.geom.vertex_words) - v_words_before) * 4
        if vb_start + vb_len <= self.buf_vertices.size and vb_len > 0:
            _ = external_call["memcpy", Int](
                self.buf_vertices.cpu_address + vb_start,
                Int(self.geom.vertex_words.unsafe_ptr()) + vb_start,
                vb_len,
            )

        var ib_start = i_words_before * 4
        var ib_len = (len(self.geom.index_words) - i_words_before) * 4
        if ib_start + ib_len <= self.buf_indices.size and ib_len > 0:
            _ = external_call["memcpy", Int](
                self.buf_indices.cpu_address + ib_start,
                Int(self.geom.index_words.unsafe_ptr()) + ib_start,
                ib_len,
            )

        var mb_start = m_words_before * 4
        var mb_len = (len(self.geom.meshlet_words) - m_words_before) * 4
        if mb_start + mb_len <= self.buf_meshlets.size and mb_len > 0:
            _ = external_call["memcpy", Int](
                self.buf_meshlets.cpu_address + mb_start,
                Int(self.geom.meshlet_words.unsafe_ptr()) + mb_start,
                mb_len,
            )

        var is_am_mesh = (mesh.alpha_mode == 1) or (
            material_id >= 0 and material_id < len(self.materials) and self.materials[material_id].is_alpha_masked()
        )
        if self.vk.has_ray_query and entry.triangle_count > 0 and entry.vertex_count > 0:
            # Note: `buf_indices` stores global vertex indices (`entry.first_vertex + local_v`) with `firstVertex = 0`.
            var v_addr = self.vk.buffer_device_address(self.buf_vertices)
            var i_addr = self.vk.buffer_device_address(self.buf_indices) + UInt64(entry.first_index * 4)
            var max_v = max(0, entry.first_vertex + entry.vertex_count - 1)
            if self.vk.begin_commands(0):
                var blas = build_blas_triangles_ex(
                    self.vk,
                    0,
                    v_addr,
                    AAA_VERTEX_BYTES,
                    max_v,
                    i_addr,
                    entry.triangle_count,
                    self.buf_as_scratch,
                    opaque=not is_am_mesh,
                )
                self.vk.memory_barrier2(
                    0,
                    PIPELINE_AS_BUILD_KHR,
                    Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR),
                    PIPELINE_AS_BUILD_KHR | PIPELINE_COMPUTE,
                    Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
                )
                var f = self.vk.submit(0)
                if f != 0:
                    _ = self.vk.wait(f)
                self.mesh_blas.append(blas^)
            else:
                self.mesh_blas.append(VulkanAccelStruct())
        else:
            self.mesh_blas.append(VulkanAccelStruct())

        self.skinned_mesh_flags.append(False)
        self.skinned_mesh_bind_slots.append(0)
        self.mesh_double_sided_flags.append(mesh.double_sided or is_am_mesh)
        self.mesh_alpha_masked_flags.append(is_am_mesh)
        self.vsm.invalidate_all()
        return entry^

    def upload_splat_mesh(
        mut self,
        mesh: NativeMesh,
        material_id: Int,
        splat_weights_u24: List[UInt32] = List[UInt32](),
        build_lods: Bool = False,
    ) -> CookedMeshEntry:
        return self.add_splat_mesh(mesh, material_id, splat_weights_u24, build_lods)

    def upload_toksvig_normal_orm(
        mut self,
        normals: List[Vec3],
        orm_argb: List[UInt32],
        width: Int,
        height: Int,
        toksvig_scale: Float32 = 1.0,
    ) -> Tuple[Int, Int]:
        """Bakes coupled `BC5` tangent-normal + Toksvig normal-variance `BC7_UNORM` ORM mip pyramids
        and uploads both to `BIND_AAA_BINDLESS_TEX`, returning `(normal_slot, orm_slot)`!"""
        if len(self.bindless_images) + 2 > AAA_MAX_BINDLESS or width <= 0 or height <= 0:
            return (-1, -1)
        var pyr = bake_toksvig_normal_orm_pyramid(normals, orm_argb, width, height, toksvig_scale, min_dim=4)
        var mips = min(8, len(pyr[0]))
        if mips <= 0:
            return (-1, -1)

        var nrm_img = self.vk.create_image_2d(
            width, height, FORMAT_BC5_UNORM, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED, mip_levels=mips
        )
        var orm_img = self.vk.create_image_2d(
            width, height, FORMAT_BC7_UNORM, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED, mip_levels=mips
        )
        if nrm_img.image == 0 or orm_img.image == 0:
            if nrm_img.image != 0:
                self.vk.destroy_image(nrm_img)
            if orm_img.image != 0:
                self.vk.destroy_image(orm_img)
            return (-1, -1)

        var nrm_offs = List[Int]()
        var orm_offs = List[Int]()
        var st_off = 0
        for m in range(mips):
            var mw = pyr[2][m]
            var mh = pyr[3][m]
            var bc5 = encode_bc5_normals(pyr[0][m], mw, mh)
            self.ensure_staging_capacity(st_off + len(bc5))
            _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_off, Int(bc5.unsafe_ptr()), len(bc5))
            nrm_offs.append(st_off)
            st_off += (len(bc5) + 15) & ~15
            _ = len(bc5)

            var bc7 = encode_bc7_rgba(pyr[1][m], mw, mh)
            self.ensure_staging_capacity(st_off + len(bc7))
            _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_off, Int(bc7.unsafe_ptr()), len(bc7))
            orm_offs.append(st_off)
            st_off += (len(bc7) + 15) & ~15
            _ = len(bc7)

        if not self.vk.begin_commands(0):
            self.vk.destroy_image(nrm_img)
            self.vk.destroy_image(orm_img)
            return (-1, -1)
        self.vk.transition_image(0, nrm_img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        self.vk.transition_image(0, orm_img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        for m in range(mips):
            var mw = pyr[2][m]
            var mh = pyr[3][m]
            self.vk.upload_image_mip(0, nrm_img, self.buf_staging, nrm_offs[m], m, mw, mh)
            self.vk.upload_image_mip(0, orm_img, self.buf_staging, orm_offs[m], m, mw, mh)
        self.vk.transition_image(0, nrm_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        self.vk.transition_image(0, orm_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        var f = self.vk.submit(0)
        if f != 0:
            _ = self.vk.wait(f)

        var nrm_slot = len(self.bindless_images)
        for ds_i in range(len(self.desc_sets)):
            self.vk.write_aaa_texture(self.desc_sets[ds_i], BIND_AAA_BINDLESS_TEX, nrm_slot, self.samp_aniso16, nrm_img.view)
        self.bindless_images.append(nrm_img^)

        var orm_slot = len(self.bindless_images)
        for ds_i in range(len(self.desc_sets)):
            self.vk.write_aaa_texture(self.desc_sets[ds_i], BIND_AAA_BINDLESS_TEX, orm_slot, self.samp_aniso16, orm_img.view)
        self.bindless_images.append(orm_img^)
        return (nrm_slot, orm_slot)

    def add_skinned_mesh(
        mut self,
        mesh: NativeMesh,
        joint_indices_packed: List[UInt32],
        joint_weights_w012: List[Vec3],
        material_id: Int,
        bone_count: Int = 64,
        bone_palette_offset: Int = -1,
    ) -> AaaSkinnedMeshEntry:
        """Uploads a 4-bone Dual-Quaternion skinned mesh into `BIND_AAA_SKINNING` (sorting each vertex's
        4 bone weights descending so `j0` is always the dominant hemisphere pivot and normalizing to `1.0`),
        auto-allocates a disjoint `bone_palette_offset` slice, widens meshlet culling spheres, and builds
        an updatable `ALLOW_UPDATE_KHR` hardware `BLAS`!"""
        var v_words_before = len(self.geom.vertex_words)
        var i_words_before = len(self.geom.index_words)
        var m_words_before = len(self.geom.meshlet_words)

        var entry = self.geom.append_mesh(mesh, material_id, build_lods=False)
        # Expand meshlet culling spheres & disable backface cone culling so articulated skeletal poses never false-cull
        for m_idx in range(entry.first_meshlet, entry.first_meshlet + entry.meshlet_count):
            var w_rad = self.geom.meshlets[m_idx].radius * Float32(4.0) + Float32(4.0)
            self.geom.meshlets[m_idx].radius = w_rad
            self.geom.meshlets[m_idx].cone_sin = Float32(2.0)
            self.geom.meshlet_words[m_idx * 12 + 3] = bitcast[DType.uint32](w_rad)
            self.geom.meshlet_words[m_idx * 12 + 7] = bitcast[DType.uint32](Float32(2.0))
        _ = self._sota_ensure_geometry_capacity()  # sota:geometry hook: grow buffers to the scene

        var vb_start = v_words_before * 4
        var vb_len = (len(self.geom.vertex_words) - v_words_before) * 4
        if vb_start + vb_len <= self.buf_vertices.size and vb_len > 0:
            _ = external_call["memcpy", Int](
                self.buf_vertices.cpu_address + vb_start,
                Int(self.geom.vertex_words.unsafe_ptr()) + vb_start,
                vb_len,
            )

        var ib_start = i_words_before * 4
        var ib_len = (len(self.geom.index_words) - i_words_before) * 4
        if ib_start + ib_len <= self.buf_indices.size and ib_len > 0:
            _ = external_call["memcpy", Int](
                self.buf_indices.cpu_address + ib_start,
                Int(self.geom.index_words.unsafe_ptr()) + ib_start,
                ib_len,
            )

        var mb_start = m_words_before * 4
        var mb_len = (len(self.geom.meshlet_words) - m_words_before) * 4
        if mb_start + mb_len <= self.buf_meshlets.size and mb_len > 0:
            _ = external_call["memcpy", Int](
                self.buf_meshlets.cpu_address + mb_start,
                Int(self.geom.meshlet_words.unsafe_ptr()) + mb_start,
                mb_len,
            )

        self.skin_arena.vertex_cursor = self.skin_bind_cursor
        var v_alloc = self.skin_arena.alloc_vertices(entry.vertex_count)
        var first_slot = v_alloc[0]
        var v_cnt = v_alloc[1]
        self.skin_bind_cursor = self.skin_arena.vertex_cursor

        self.skin_arena.bone_cursor = self.skin_bone_cursor
        var pal_off = self.skin_arena.alloc_bone_palette(bone_count, bone_palette_offset)
        self.skin_bone_cursor = self.skin_arena.bone_cursor

        # Initialize any unpopulated bone slots in `CUR_BONES` & `PREV_BONES` to exact identity `DualQuat(0,0,0,1, 0,0,0,0)`
        var eff_bones_init = min(max(1, bone_count), max(0, AAA_MAX_SKIN_BONES - (pal_off // 8)))
        for b in range(eff_bones_init):
            var c_off = (AAA_SKIN_CUR_BONES_BASE + pal_off + b * 8) * 4
            if (
                get32(self.buf_skinning.cpu_address + c_off + 0) == UInt32(0)
                and get32(self.buf_skinning.cpu_address + c_off + 4) == UInt32(0)
                and get32(self.buf_skinning.cpu_address + c_off + 8) == UInt32(0)
                and get32(self.buf_skinning.cpu_address + c_off + 12) == UInt32(0)
            ):
                put_float(self.buf_skinning.cpu_address + c_off + 12, Float32(1.0))
            var p_off = (AAA_SKIN_PREV_BONES_BASE + pal_off + b * 8) * 4
            if (
                get32(self.buf_skinning.cpu_address + p_off + 0) == UInt32(0)
                and get32(self.buf_skinning.cpu_address + p_off + 4) == UInt32(0)
                and get32(self.buf_skinning.cpu_address + p_off + 8) == UInt32(0)
                and get32(self.buf_skinning.cpu_address + p_off + 12) == UInt32(0)
            ):
                put_float(self.buf_skinning.cpu_address + p_off + 12, Float32(1.0))

        var m_w_first = 0
        var m_d_first = 0
        var eff_morph_cnt = 0
        var mesh_v_cnt = max(v_cnt, mesh.vertex_count())
        if v_cnt > 0 and mesh.morph_count > 0 and len(mesh.morph_positions) >= mesh.morph_count * v_cnt * 3:
            self.skin_arena.morph_weight_cursor = self.skin_morph_weight_cursor
            self.skin_arena.morph_delta_cursor = self.skin_morph_delta_cursor
            var m_alloc = self.skin_arena.alloc_morph_targets(mesh.morph_count, v_cnt)
            m_w_first = m_alloc[0]
            m_d_first = m_alloc[1]
            eff_morph_cnt = m_alloc[2]
            self.skin_morph_weight_cursor = self.skin_arena.morph_weight_cursor
            self.skin_morph_delta_cursor = self.skin_arena.morph_delta_cursor

            if eff_morph_cnt > 0:
                _ = external_call[
                    "memset", Int
                ](
                    self.buf_skinning.cpu_address + (AAA_SKIN_CUR_MORPH_WEIGHTS_BASE + m_w_first) * 4,
                    Int32(0),
                    eff_morph_cnt * 4,
                )
                _ = external_call[
                    "memset", Int
                ](
                    self.buf_skinning.cpu_address + (AAA_SKIN_PREV_MORPH_WEIGHTS_BASE + m_w_first) * 4,
                    Int32(0),
                    eff_morph_cnt * 4,
                )
                var src_stride_v = mesh_v_cnt if len(mesh.morph_positions) >= mesh.morph_count * mesh_v_cnt * 3 else v_cnt
                var has_morph_nrm = len(mesh.morph_normals) >= mesh.morph_count * src_stride_v * 3
                for m_i in range(eff_morph_cnt):
                    for v in range(v_cnt):
                        var src_m3 = (m_i * src_stride_v + v) * 3
                        var dst_dw = AAA_SKIN_MORPH_DELTAS_BASE + (m_d_first + m_i * v_cnt + v) * 6
                        put_float(self.buf_skinning.cpu_address + (dst_dw + 0) * 4, mesh.morph_positions[src_m3 + 0])
                        put_float(self.buf_skinning.cpu_address + (dst_dw + 1) * 4, mesh.morph_positions[src_m3 + 1])
                        put_float(self.buf_skinning.cpu_address + (dst_dw + 2) * 4, mesh.morph_positions[src_m3 + 2])
                        var dnx = mesh.morph_normals[src_m3 + 0] if has_morph_nrm else Float32(0.0)
                        var dny = mesh.morph_normals[src_m3 + 1] if has_morph_nrm else Float32(0.0)
                        var dnz = mesh.morph_normals[src_m3 + 2] if has_morph_nrm else Float32(0.0)
                        put_float(self.buf_skinning.cpu_address + (dst_dw + 3) * 4, dnx)
                        put_float(self.buf_skinning.cpu_address + (dst_dw + 4) * 4, dny)
                        put_float(self.buf_skinning.cpu_address + (dst_dw + 5) * 4, dnz)

        var has_mesh_jw = len(mesh.joints) >= v_cnt * 4 and len(mesh.weights) >= v_cnt * 4

        for v in range(v_cnt):
            var gv = entry.first_vertex + v
            var src_vw = gv * 12
            var dst_sw = AAA_SKIN_BIND_VERTS_BASE + (first_slot + v) * AAA_SKIN_BIND_STRIDE_WORDS
            for k in range(12):
                put32(self.buf_skinning.cpu_address + (dst_sw + k) * 4, self.geom.vertex_words[src_vw + k])

            var j0_r = 0; var j1_r = 0; var j2_r = 0; var j3_r = 0
            var w0_r = Float32(1.0); var w1_r = Float32(0.0); var w2_r = Float32(0.0); var w3_r = Float32(0.0)
            if v < len(joint_indices_packed) and v < len(joint_weights_w012):
                var jp_in = joint_indices_packed[v]
                j0_r = Int(jp_in & UInt32(0xFF))
                j1_r = Int((jp_in >> UInt32(8)) & UInt32(0xFF))
                j2_r = Int((jp_in >> UInt32(16)) & UInt32(0xFF))
                j3_r = Int((jp_in >> UInt32(24)) & UInt32(0xFF))
                var wv = joint_weights_w012[v]
                w0_r = max(Float32(0.0), wv.x)
                w1_r = max(Float32(0.0), wv.y)
                w2_r = max(Float32(0.0), wv.z)
                if has_mesh_jw and Int(mesh.joints[v * 4 + 3]) == j3_r and Int(mesh.joints[v * 4 + 0]) == j0_r:
                    w3_r = max(Float32(0.0), mesh.weights[v * 4 + 3])
                elif w0_r > Float32(0.0) and w1_r > Float32(0.0) and w2_r > Float32(0.0) and j3_r != 0 and (w0_r + w1_r + w2_r) < Float32(1.0):
                    w3_r = Float32(1.0) - (w0_r + w1_r + w2_r)
            elif has_mesh_jw:
                j0_r = Int(mesh.joints[v * 4 + 0]) & 0xFF
                j1_r = Int(mesh.joints[v * 4 + 1]) & 0xFF
                j2_r = Int(mesh.joints[v * 4 + 2]) & 0xFF
                j3_r = Int(mesh.joints[v * 4 + 3]) & 0xFF
                w0_r = max(Float32(0.0), mesh.weights[v * 4 + 0])
                w1_r = max(Float32(0.0), mesh.weights[v * 4 + 1])
                w2_r = max(Float32(0.0), mesh.weights[v * 4 + 2])
                w3_r = max(Float32(0.0), mesh.weights[v * 4 + 3])

            var ns = normalize_and_sort_skin_weights_4(j0_r, j1_r, j2_r, j3_r, w0_r, w1_r, w2_r, w3_r)
            put32(self.buf_skinning.cpu_address + (dst_sw + 12) * 4, ns[0])
            put_float(self.buf_skinning.cpu_address + (dst_sw + 13) * 4, ns[1])
            put_float(self.buf_skinning.cpu_address + (dst_sw + 14) * 4, ns[2])
            put_float(self.buf_skinning.cpu_address + (dst_sw + 15) * 4, ns[3])

            var prev_off = AAA_SKIN_PREV_POS_BASE + (first_slot + v) * 3
            put32(self.buf_skinning.cpu_address + (prev_off + 0) * 4, self.geom.vertex_words[src_vw + 0])
            put32(self.buf_skinning.cpu_address + (prev_off + 1) * 4, self.geom.vertex_words[src_vw + 1])
            put32(self.buf_skinning.cpu_address + (prev_off + 2) * 4, self.geom.vertex_words[src_vw + 2])

        if self.vk.has_ray_query and entry.triangle_count > 0 and entry.vertex_count > 0:
            var v_addr = self.vk.buffer_device_address(self.buf_vertices)
            var i_addr = self.vk.buffer_device_address(self.buf_indices) + UInt64(entry.first_index * 4)
            var max_v = max(0, entry.first_vertex + entry.vertex_count - 1)
            if self.vk.begin_commands(0):
                var blas = self.vk.build_blas_triangles(
                    0,
                    v_addr,
                    AAA_VERTEX_BYTES,
                    max_v,
                    i_addr,
                    entry.triangle_count,
                    self.buf_as_scratch,
                    allow_update=True,
                )
                self.vk.memory_barrier2(
                    0,
                    PIPELINE_AS_BUILD_KHR,
                    Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR),
                    PIPELINE_AS_BUILD_KHR | PIPELINE_COMPUTE,
                    Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
                )
                var f = self.vk.submit(0)
                if f != 0:
                    _ = self.vk.wait(f)
                self.mesh_blas.append(blas^)
            else:
                self.mesh_blas.append(VulkanAccelStruct())
        else:
            self.mesh_blas.append(VulkanAccelStruct())

        self.skinned_mesh_flags.append(v_cnt > 0)
        self.skinned_mesh_bind_slots.append(first_slot)
        self.mesh_double_sided_flags.append(mesh.double_sided)
        self.mesh_alpha_masked_flags.append(False)
        self.vsm.invalidate_all()

        var res = AaaSkinnedMeshEntry()
        res.mesh_entry = entry^
        res.skin_bind_first_slot = first_slot
        res.vertex_count = v_cnt
        res.bone_count = max(1, bone_count)
        res.bone_palette_offset = pal_off
        res.morph_count = eff_morph_cnt
        res.morph_weight_first_slot = m_w_first
        res.morph_delta_first_slot = m_d_first
        return res^

    def _enqueue_pending_skin_update(mut self, skin_entry: AaaSkinnedMeshEntry):
        for k in range(len(self.pending_skin_updates)):
            if (
                self.pending_skin_updates[k].mesh_entry.mesh_id == skin_entry.mesh_entry.mesh_id
                and self.pending_skin_updates[k].skin_bind_first_slot == skin_entry.skin_bind_first_slot
            ):
                self.pending_skin_updates[k] = skin_entry.copy()
                return
        self.pending_skin_updates.append(skin_entry.copy())

    def _remove_pending_skin_update(mut self, skin_entry: AaaSkinnedMeshEntry):
        var i = 0
        while i < len(self.pending_skin_updates):
            if (
                self.pending_skin_updates[i].mesh_entry.mesh_id == skin_entry.mesh_entry.mesh_id
                and self.pending_skin_updates[i].skin_bind_first_slot == skin_entry.skin_bind_first_slot
            ):
                _ = self.pending_skin_updates.pop(i)
            else:
                i += 1

    def _record_skin_dispatch_and_blas_refit(mut self, frame: Int, skin_entry: AaaSkinnedMeshEntry):
        var push_sk = List[UInt32](length=16, fill=0)
        push_sk[0] = UInt32(skin_entry.vertex_count)
        push_sk[1] = UInt32(skin_entry.skin_bind_first_slot)
        push_sk[2] = UInt32(skin_entry.mesh_entry.first_vertex)
        push_sk[3] = UInt32(AAA_SKIN_CUR_BONES_BASE + skin_entry.bone_palette_offset)
        push_sk[4] = UInt32(AAA_SKIN_PREV_BONES_BASE + skin_entry.bone_palette_offset)
        push_sk[5] = UInt32(AAA_SKIN_CUR_SMEAR_BASE + skin_entry.bone_palette_offset)
        push_sk[6] = UInt32(AAA_SKIN_PREV_SMEAR_BASE + skin_entry.bone_palette_offset)
        push_sk[7] = UInt32(max(0, skin_entry.morph_count))
        push_sk[8] = UInt32(max(0, skin_entry.morph_weight_first_slot))
        push_sk[9] = UInt32(max(0, skin_entry.morph_delta_first_slot))
        self.vk.dispatch_aaa(
            frame,
            self.pipe_dq_skinning,
            self.pipe_layout,
            self.desc_set,
            push_sk,
            (skin_entry.vertex_count + 63) // 64,
            1,
        )

        var me = skin_entry.mesh_entry.copy()
        if self.vk.has_ray_query and me.mesh_id < len(self.mesh_blas) and self.mesh_blas[me.mesh_id].handle != 0:
            self.vk.memory_barrier2(
                frame,
                PIPELINE_COMPUTE,
                Int(ACCESS_SHADER_WRITE),
                PIPELINE_AS_BUILD_KHR | PIPELINE_VERTEX_SHADER | PIPELINE_COMPUTE,
                Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
            )
            var v_addr = self.vk.buffer_device_address(self.buf_vertices)
            var i_addr = self.vk.buffer_device_address(self.buf_indices) + UInt64(me.first_index * 4)
            var max_v = max(0, me.first_vertex + me.vertex_count - 1)
            self.vk.refit_blas_triangles(
                frame,
                self.mesh_blas[me.mesh_id],
                v_addr,
                AAA_VERTEX_BYTES,
                max_v,
                i_addr,
                me.triangle_count,
                self.buf_as_scratch,
            )
            self.vk.memory_barrier2(
                frame,
                PIPELINE_AS_BUILD_KHR,
                Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR),
                PIPELINE_AS_BUILD_KHR | PIPELINE_COMPUTE,
                Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
            )
        else:
            self.vk.memory_barrier2(
                frame,
                PIPELINE_COMPUTE,
                Int(ACCESS_SHADER_WRITE),
                PIPELINE_VERTEX_SHADER | PIPELINE_COMPUTE,
                Int(ACCESS_SHADER_READ),
            )

    def update_skinned_mesh_bones(
        mut self,
        skin_entry: AaaSkinnedMeshEntry,
        cur_bones: List[DualQuat],
        prev_bones: List[DualQuat],
        invalidate_vsm: Bool = False,
        defer_to_frame: Bool = False,
        reset_smears: Bool = True,
    ) -> Float64:
        """Dispatches 4-bone GPU Dual-Quaternion compute skinning into `buf_vertices` + `buf_skinning`
        and performs a zero-allocation in-place `VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR` `BLAS` refit!
        When `defer_to_frame=True`, records inline inside `render_frame()`'s single command buffer (`0` extra submit!)."""
        if not self.active or skin_entry.vertex_count <= 0:
            return 0.0

        var pal_off = max(0, skin_entry.bone_palette_offset)
        var max_avail = max(0, AAA_MAX_SKIN_BONES - (pal_off // 8))
        var cur_base = AAA_SKIN_CUR_BONES_BASE + pal_off
        var prev_base = AAA_SKIN_PREV_BONES_BASE + pal_off
        var n_cur = min(max_avail, len(cur_bones))
        for b in range(n_cur):
            var dq = cur_bones[b]
            var off = (cur_base + b * 8) * 4
            put_float(self.buf_skinning.cpu_address + off + 0, dq.rx)
            put_float(self.buf_skinning.cpu_address + off + 4, dq.ry)
            put_float(self.buf_skinning.cpu_address + off + 8, dq.rz)
            put_float(self.buf_skinning.cpu_address + off + 12, dq.rw)
            put_float(self.buf_skinning.cpu_address + off + 16, dq.dx)
            put_float(self.buf_skinning.cpu_address + off + 20, dq.dy)
            put_float(self.buf_skinning.cpu_address + off + 24, dq.dz)
            put_float(self.buf_skinning.cpu_address + off + 28, dq.dw)

        var n_prev = min(max_avail, len(prev_bones))
        for b in range(n_prev):
            var dq = prev_bones[b]
            var off = (prev_base + b * 8) * 4
            put_float(self.buf_skinning.cpu_address + off + 0, dq.rx)
            put_float(self.buf_skinning.cpu_address + off + 4, dq.ry)
            put_float(self.buf_skinning.cpu_address + off + 8, dq.rz)
            put_float(self.buf_skinning.cpu_address + off + 12, dq.rw)
            put_float(self.buf_skinning.cpu_address + off + 16, dq.dx)
            put_float(self.buf_skinning.cpu_address + off + 20, dq.dy)
            put_float(self.buf_skinning.cpu_address + off + 24, dq.dz)
            put_float(self.buf_skinning.cpu_address + off + 28, dq.dw)

        if reset_smears:
            var n_clr = max(n_cur, n_prev)
            if n_clr > 0:
                _ = external_call[
                    "memset", Int
                ](self.buf_skinning.cpu_address + (AAA_SKIN_CUR_SMEAR_BASE + pal_off) * 4, Int32(0), n_clr * 32)
                _ = external_call[
                    "memset", Int
                ](self.buf_skinning.cpu_address + (AAA_SKIN_PREV_SMEAR_BASE + pal_off) * 4, Int32(0), n_clr * 32)

        if invalidate_vsm:
            self.vsm.invalidate_all()

        if defer_to_frame:
            self._enqueue_pending_skin_update(skin_entry)
            return 0.0

        self._remove_pending_skin_update(skin_entry)
        var t0 = get_time()
        if not self.vk.begin_commands(0):
            return 0.0
        self._record_skin_dispatch_and_blas_refit(0, skin_entry)
        var f = self.vk.submit(0)
        if f != 0:
            _ = self.vk.wait(f)
        return (get_time() - t0) * 1000.0

    def update_skinned_mesh_morphs(
        mut self,
        skin_entry: AaaSkinnedMeshEntry,
        cur_weights: List[Float32],
        prev_weights: List[Float32] = List[Float32](),
    ):
        """Uploads `BlendShape` morph-target weights (`cur_weights` and `prev_weights`, defaulting `prev_weights`
        to `cur_weights` when empty) into `buf_skinning` and queues an inline GPU skinning + `BLAS` refit pass!"""
        if not self.active or skin_entry.vertex_count <= 0 or skin_entry.morph_count <= 0:
            return
        var m_first = max(0, skin_entry.morph_weight_first_slot)
        var max_m = min(skin_entry.morph_count, max(0, AAA_SKIN_MAX_MORPH_SLOTS - m_first))
        if max_m <= 0:
            return
        var has_prev = len(prev_weights) > 0
        for m in range(max_m):
            var cw = cur_weights[m] if m < len(cur_weights) else Float32(0.0)
            var pw = (prev_weights[m] if m < len(prev_weights) else Float32(0.0)) if has_prev else cw
            put_float(self.buf_skinning.cpu_address + (AAA_SKIN_CUR_MORPH_WEIGHTS_BASE + m_first + m) * 4, cw)
            put_float(self.buf_skinning.cpu_address + (AAA_SKIN_PREV_MORPH_WEIGHTS_BASE + m_first + m) * 4, pw)
        self._enqueue_pending_skin_update(skin_entry)

    def update_skinned_mesh_smear(
        mut self,
        skin_entry: AaaSkinnedMeshEntry,
        cur_bones: List[DualQuat],
        prev_bones: List[DualQuat],
        cur_smears: List[BoneSmear],
        prev_smears: List[BoneSmear] = List[BoneSmear](),
        invalidate_vsm: Bool = False,
        defer_to_frame: Bool = False,
    ) -> Float64:
        """Uploads Dual-Quaternion bones + non-uniform 3D affine `BoneSmear` stretch/squash/shear frames,
        deforms `buf_vertices` in-place with `J^{-T}` normals, and refits the mesh's hardware `BLAS`!"""
        if not self.active or skin_entry.vertex_count <= 0:
            return 0.0
        var pal_off = max(0, skin_entry.bone_palette_offset)
        var max_avail = max(0, AAA_MAX_SKIN_BONES - (pal_off // 8))
        var cur_sm_base = AAA_SKIN_CUR_SMEAR_BASE + pal_off
        var prev_sm_base = AAA_SKIN_PREV_SMEAR_BASE + pal_off
        var n_clear = min(max_avail, max(len(cur_bones), max(len(cur_smears), len(prev_smears))))
        if n_clear > 0:
            _ = external_call["memset", Int](self.buf_skinning.cpu_address + cur_sm_base * 4, Int32(0), n_clear * 32)
            _ = external_call["memset", Int](self.buf_skinning.cpu_address + prev_sm_base * 4, Int32(0), n_clear * 32)

        var n_cs = min(max_avail, len(cur_smears))
        for b in range(n_cs):
            var sm = cur_smears[b]
            var off = (cur_sm_base + b * 8) * 4
            put_float(self.buf_skinning.cpu_address + off + 0, sm.scale.x - Float32(1.0))
            put_float(self.buf_skinning.cpu_address + off + 4, sm.scale.y - Float32(1.0))
            put_float(self.buf_skinning.cpu_address + off + 8, sm.scale.z - Float32(1.0))
            put_float(self.buf_skinning.cpu_address + off + 12, sm.shear_xy)
            put_float(self.buf_skinning.cpu_address + off + 16, sm.pivot_bind.x)
            put_float(self.buf_skinning.cpu_address + off + 20, sm.pivot_bind.y)
            put_float(self.buf_skinning.cpu_address + off + 24, sm.pivot_bind.z)
            put_float(self.buf_skinning.cpu_address + off + 28, sm.shear_xz)

        var has_ps = len(prev_smears) > 0
        var n_ps = min(max_avail, len(prev_smears) if has_ps else len(cur_smears))
        for b in range(n_ps):
            var sm = prev_smears[b] if has_ps else cur_smears[b]
            var off = (prev_sm_base + b * 8) * 4
            put_float(self.buf_skinning.cpu_address + off + 0, sm.scale.x - Float32(1.0))
            put_float(self.buf_skinning.cpu_address + off + 4, sm.scale.y - Float32(1.0))
            put_float(self.buf_skinning.cpu_address + off + 8, sm.scale.z - Float32(1.0))
            put_float(self.buf_skinning.cpu_address + off + 12, sm.shear_xy)
            put_float(self.buf_skinning.cpu_address + off + 16, sm.pivot_bind.x)
            put_float(self.buf_skinning.cpu_address + off + 20, sm.pivot_bind.y)
            put_float(self.buf_skinning.cpu_address + off + 24, sm.pivot_bind.z)
            put_float(self.buf_skinning.cpu_address + off + 28, sm.shear_xz)

        return self.update_skinned_mesh_bones(
            skin_entry, cur_bones, prev_bones, invalidate_vsm=invalidate_vsm, defer_to_frame=defer_to_frame, reset_smears=False
        )

    def set_bruneton_atmosphere(
        mut self,
        params: BrunetonAtmosphereParams,
        enable_aerial: Bool = True,
    ) -> BrunetonSkyLuts:
        """Bakes Bruneton/Hillaire Transmittance (`256x64`), Multiple-Scattering (`32x32`), and Sky-View (`192x108`)
        LUTs, uploads the resulting `BC6H_UFLOAT` GGX cubemap + `SH9` irradiance, and configures GPU aerial perspective!"""
        var luts = BrunetonSkyLuts(params)
        var ibl = bake_bruneton_sky_ibl(luts, base_dim=self.env_cube_dim, mip_levels=self.env_mip_levels)
        self.upload_ibl_environment(ibl)
        self.sun_dir = params.sun_dir
        self.sun_angular_radius = params.sun_angular_radius
        var t_sun = luts.sun_transmittance_at_observer()
        self.sun_radiance = Vec3(
            params.solar_irradiance.x * t_sun.x,
            params.solar_irradiance.y * t_sun.y,
            params.solar_irradiance.z * t_sun.z,
        )
        var ap = luts.compute_aerial_perspective_constants()
        self.aerial_sigma_ext = ap[0]
        self.aerial_r_term = ap[1]
        self.aerial_m_term = ap[2]
        self.aerial_mie_g = ap[3]
        self.enable_aerial_perspective = enable_aerial
        self.cloud_coverage = params.cloud_coverage
        self.cloud_wind_offset = params.cloud_wind_offset
        self.star_intensity = params.star_intensity
        if params.cloud_coverage > Float32(0.01):
            var clr_p = params.copy()
            clr_p.cloud_coverage = Float32(0.0)
            var clr_luts = BrunetonSkyLuts(clr_p)
            self.sky_zenith_clear = clr_luts.sample_sky(Vec3(0.0, 1.0, 0.0))
            self.sky_horizon_clear = clr_luts.sample_sky(Vec3(0.7071, 0.06, 0.7071).normalized())
        else:
            self.sky_zenith_clear = luts.sample_sky(Vec3(0.0, 1.0, 0.0))
            self.sky_horizon_clear = luts.sample_sky(Vec3(0.7071, 0.06, 0.7071).normalized())
        return luts^

    def upload_native_model(
        mut self,
        mut model: NativeModel,
        build_lods: Bool = True,
        max_tex_dim: Int = 2048,
        upload_skinned: Bool = False,
    ) -> AaaUploadedModel:
        """Turnkey glTF 2.0 / `.glb` `NativeModel` -> GPU AAA Asset Uploader (`V19`):
        - Deduplicates & block-compresses embedded textures (`sRGB BC7` for `baseColorTexture`,
          `BC5` tangent normals for `normalTexture`, `UNORM BC7` linear ORM for `metallicRoughnessTexture`).
        - Registers `AaaMaterial` descriptors with exact IEC sRGB -> linear `baseColorFactor` & HDR emissive.
        - Partitions every `NativeMesh` into Forsyth-optimized `64v/124t` Meshlets + `QEM` `LOD 0..3` DAG + `BLAS`!
        """
        var out = AaaUploadedModel()
        var n_tex = len(model.textures)
        var cap_dim = max(16, max_tex_dim)
        out.albedo_bindless_slots = List[Int](length=n_tex, fill=-1)
        out.normal_bindless_slots = List[Int](length=n_tex, fill=-1)
        out.orm_bindless_slots = List[Int](length=n_tex, fill=-1)

        var sota_vt_of_tex = List[Int](length=n_tex, fill=-1)  # sota:geometry hook (VT albedo per texture)
        self._sota_prepare_upload(model, cap_dim, build_lods, upload_skinned)  # --- sota:perf hook ---
        for m_idx in range(len(model.meshes)):
            ref m = model.meshes[m_idx]
            self.sota_prep.current_mesh = m_idx  # --- sota:perf hook ---
            var alb_slot = -1
            # --- sota:geometry hook --- big square albedo maps stream through virtual texturing (r.sota.vt.models)
            var sota_vt_idx = -1
            if m.texture >= 0 and m.texture < n_tex and m.alpha_mode != 1:
                if sota_vt_of_tex[m.texture] < 0 and out.albedo_bindless_slots[m.texture] < 0:
                    ref vtx = model.textures[m.texture]
                    if self.sota_vt.wants_model_texture(vtx.width, vtx.height) and len(vtx.pixels) >= vtx.width * vtx.height:
                        sota_vt_of_tex[m.texture] = self.add_virtual_texture(vtx.pixels, vtx.width)
                sota_vt_idx = sota_vt_of_tex[m.texture]
            # --- end sota:geometry hook ---
            # --- sota:perf hook --- albedo encoded by the parallel prepass
            if (m.texture >= 0 and m.texture < n_tex and sota_vt_idx < 0 and out.albedo_bindless_slots[m.texture] < 0
                    and self.sota_prep.has_texture(SOTA_PREP_ALBEDO, m.texture)):
                out.albedo_bindless_slots[m.texture] = self._sota_upload_prepared_texture(SOTA_PREP_ALBEDO, m.texture)
            # --- end sota:perf hook ---
            if m.texture >= 0 and m.texture < n_tex and sota_vt_idx < 0:
                if out.albedo_bindless_slots[m.texture] < 0:
                    ref tx = model.textures[m.texture]
                    if tx.width > 0 and tx.height > 0 and len(tx.pixels) >= tx.width * tx.height:
                        var cur_px = tx.pixels.copy()
                        var cw = tx.width
                        var ch = tx.height
                        while (cw > cap_dim or ch > cap_dim) and cw >= 8 and ch >= 8:
                            var ds = downsample_argb_2x(cur_px, cw, ch)
                            cur_px = ds[0].copy(); cw = ds[1]; ch = ds[2]
                        out.albedo_bindless_slots[m.texture] = self.upload_bc7_texture(
                            cur_px, cw, ch, srgb=True, generate_mips=True
                        )
                if out.albedo_bindless_slots[m.texture] >= 0:
                    alb_slot = out.albedo_bindless_slots[m.texture]

            var nrm_slot = -1
            # --- sota:perf hook --- normal map encoded by the parallel prepass
            if (m.normal_orm_texture >= 0 and m.normal_orm_texture < n_tex and out.normal_bindless_slots[m.normal_orm_texture] < 0
                    and self.sota_prep.has_texture(SOTA_PREP_NORMAL, m.normal_orm_texture)):
                out.normal_bindless_slots[m.normal_orm_texture] = self._sota_upload_prepared_texture(SOTA_PREP_NORMAL, m.normal_orm_texture)
            # --- end sota:perf hook ---
            if m.normal_orm_texture >= 0 and m.normal_orm_texture < n_tex:
                if out.normal_bindless_slots[m.normal_orm_texture] < 0:
                    ref ntx = model.textures[m.normal_orm_texture]
                    if ntx.width > 0 and ntx.height > 0 and len(ntx.pixels) >= ntx.width * ntx.height:
                        var cur_np = ntx.pixels.copy()
                        var nw = ntx.width
                        var nh = ntx.height
                        while (nw > cap_dim or nh > cap_dim) and nw >= 8 and nh >= 8:
                            var ds = downsample_argb_2x(cur_np, nw, nh)
                            cur_np = ds[0].copy(); nw = ds[1]; nh = ds[2]
                        var n_vecs = List[Vec3](capacity=nw * nh)
                        for p in cur_np:
                            var nx = (Float32((p >> 16) & 0xFF) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
                            var ny = (Float32((p >> 8) & 0xFF) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
                            var nz = (Float32(p & 0xFF) / Float32(255.0)) * Float32(2.0) - Float32(1.0)
                            n_vecs.append(Vec3(nx, ny, max(Float32(0.05), nz)).normalized())
                        out.normal_bindless_slots[m.normal_orm_texture] = self.upload_bc5_normal_texture(
                            n_vecs, nw, nh, generate_mips=True
                        )
                if out.normal_bindless_slots[m.normal_orm_texture] >= 0:
                    nrm_slot = out.normal_bindless_slots[m.normal_orm_texture]

            var orm_slot = -1
            # --- sota:perf hook --- ORM encoded by the parallel prepass
            if (m.orm_texture >= 0 and m.orm_texture < n_tex and out.orm_bindless_slots[m.orm_texture] < 0
                    and self.sota_prep.has_texture(SOTA_PREP_ORM, m.orm_texture)):
                out.orm_bindless_slots[m.orm_texture] = self._sota_upload_prepared_texture(SOTA_PREP_ORM, m.orm_texture)
            # --- end sota:perf hook ---
            if m.orm_texture >= 0 and m.orm_texture < n_tex:
                if out.orm_bindless_slots[m.orm_texture] < 0:
                    ref otx = model.textures[m.orm_texture]
                    if otx.width > 0 and otx.height > 0 and len(otx.pixels) >= otx.width * otx.height:
                        var cur_op = otx.pixels.copy()
                        var ow = otx.width
                        var oh = otx.height
                        while (ow > cap_dim or oh > cap_dim) and ow >= 8 and oh >= 8:
                            var ds = downsample_argb_2x(cur_op, ow, oh)
                            cur_op = ds[0].copy(); ow = ds[1]; oh = ds[2]
                        # Ensure AO defaults to 255 (R channel) if glTF only populated G=Roughness, B=Metallic
                        var orm_px = List[UInt32](capacity=ow * oh)
                        for p in cur_op:
                            var r_ao = (p >> 16) & 0xFF
                            var use_ao = UInt32(255) if r_ao == 0 else r_ao
                            orm_px.append(UInt32(0xFF000000) | (use_ao << 16) | (p & 0x0000FFFF))
                        out.orm_bindless_slots[m.orm_texture] = self.upload_bc7_texture(
                            orm_px, ow, oh, srgb=False, generate_mips=True
                        )
                if out.orm_bindless_slots[m.orm_texture] >= 0:
                    orm_slot = out.orm_bindless_slots[m.orm_texture]

            var bc = m.base_color
            var lin_bc = Vec3(
                srgb_u8_to_linear(Int((bc >> 16) & 0xFF)),
                srgb_u8_to_linear(Int((bc >> 8) & 0xFF)),
                srgb_u8_to_linear(Int(bc & 0xFF)),
            )
            var mat = AaaMaterial(
                base_color=lin_bc,
                roughness=m.roughness,
                metallic=m.metallic,
            )
            mat.albedo_tex = alb_slot
            mat.normal_tex = nrm_slot
            mat.orm_tex = orm_slot
            if m.emissive_strength > Float32(1e-4):
                mat.emissive = m.emissive_color * m.emissive_strength
            mat.clearcoat = m.clearcoat
            mat.clearcoat_roughness = m.clearcoat_roughness
            mat.specular_factor = m.specular_factor
            mat.iridescence = m.iridescence
            mat.iridescence_thickness_nm = m.iridescence_thickness_nm
            mat.iridescence_ior = m.iridescence_ior
            mat.sheen_color = m.sheen_color
            mat.sheen_roughness = m.sheen_roughness
            if m.alpha_mode == 1:
                mat.alpha_cutoff = Float32(0.48)
            var mat_id = self.add_material_vt(mat, sota_vt_idx) if sota_vt_idx >= 0 else self.add_material(mat)  # sota:geometry hook
            var mv_cnt = m.vertex_count()
            if upload_skinned and mv_cnt > 0 and len(m.joints) >= mv_cnt * 4 and len(m.weights) >= mv_cnt * 4:
                var b_cnt = 64
                if m.skin >= 0 and m.skin < len(model.skins) and len(model.skins[m.skin].joints) > 0:
                    b_cnt = len(model.skins[m.skin].joints)
                var empty_jp = List[UInt32]()
                var empty_jw = List[Vec3]()
                var sk_entry = self.add_skinned_mesh(m, empty_jp, empty_jw, mat_id, bone_count=b_cnt)
                out.material_ids.append(mat_id)
                out.mesh_entries.append(sk_entry.mesh_entry.copy())
                out.skinned_entries.append(sk_entry^)
            else:
                var entry = self.add_mesh(m, mat_id, build_lods=build_lods)
                out.material_ids.append(mat_id)
                out.mesh_entries.append(entry^)

        self.sota_prep.clear()  # --- sota:perf hook ---
        return out^

    def draw_native_model(
        mut self,
        mut model: NativeModel,
        uploaded: AaaUploadedModel,
        root_world: Mat4 = Mat4(),
        prev_root_world: Mat4 = Mat4(),
        has_prev: Bool = False,
        prev_node_globals: List[Mat4] = List[Mat4](),
    ):
        """Traverses a hierarchical `NativeModel` scene graph (`nodes`, `mesh_indices`, `global_transforms()`)
        and submits every primitive with its exact world transform & bounding-sphere scale!"""
        var n_meshes = len(uploaded.mesh_entries)
        if n_meshes == 0:
            return
        if len(model.nodes) == 0:
            var sx = Vec3(root_world.values[0], root_world.values[1], root_world.values[2]).length()
            var sy = Vec3(root_world.values[4], root_world.values[5], root_world.values[6]).length()
            var sz = Vec3(root_world.values[8], root_world.values[9], root_world.values[10]).length()
            var u_sc = max(sx, max(sy, sz))
            for m_i in range(n_meshes):
                self.draw_instance(uploaded.mesh_entries[m_i].mesh_id, root_world, prev_root_world, -1, u_sc, has_prev)
            return

        var globals = model.global_transforms()
        var use_prev = prev_root_world.copy() if has_prev else root_world.copy()
        var has_prev_nodes = has_prev and len(prev_node_globals) == len(model.nodes)
        for n_i in range(len(model.nodes)):
            var node_m = root_world.multiply(globals[n_i])
            var prev_m = use_prev.multiply(prev_node_globals[n_i].copy() if has_prev_nodes else globals[n_i].copy())
            var sx = Vec3(node_m.values[0], node_m.values[1], node_m.values[2]).length()
            var sy = Vec3(node_m.values[4], node_m.values[5], node_m.values[6]).length()
            var sz = Vec3(node_m.values[8], node_m.values[9], node_m.values[10]).length()
            var u_sc = max(sx, max(sy, sz))

            if len(model.nodes[n_i].mesh_indices) > 0:
                for pm in model.nodes[n_i].mesh_indices:
                    if pm >= 0 and pm < n_meshes:
                        self.draw_instance(uploaded.mesh_entries[pm].mesh_id, node_m, prev_m, -1, u_sc, has_prev)
            elif model.nodes[n_i].mesh >= 0 and model.nodes[n_i].mesh < n_meshes:
                self.draw_instance(uploaded.mesh_entries[model.nodes[n_i].mesh].mesh_id, node_m, prev_m, -1, u_sc, has_prev)

    def draw_native_model_articulated(
        mut self,
        model: NativeModel,
        uploaded: UploadedModelSlots,
        cur_globals: List[Mat4],
        prev_globals: List[Mat4],
        root_world: Mat4 = Mat4(),
        prev_root_world: Mat4 = Mat4(),
        has_prev: Bool = True,
        material_override: Int = -1,
    ):
        """Submits an articulated multi-node `NativeModel` using independent `cur_globals` and `prev_globals`
        node hierarchies (`cur_m = root_world * cur_globals[n_i]`, `prev_m = prev_root_world * prev_globals[n_i]`)
        so transforming sub-nodes (wings, arms, legs, turrets) emit exact 3D motion vectors!"""
        var n_meshes = len(uploaded.mesh_entries)
        if n_meshes == 0:
            return
        var use_prev_root = prev_root_world.copy() if has_prev else root_world.copy()
        if len(model.nodes) == 0:
            var sx = Vec3(root_world.values[0], root_world.values[1], root_world.values[2]).length()
            var sy = Vec3(root_world.values[4], root_world.values[5], root_world.values[6]).length()
            var sz = Vec3(root_world.values[8], root_world.values[9], root_world.values[10]).length()
            var u_sc = max(sx, max(sy, sz))
            for m_i in range(n_meshes):
                self.draw_instance(
                    uploaded.mesh_entries[m_i].mesh_id,
                    root_world,
                    use_prev_root,
                    material_override,
                    u_sc,
                    has_prev,
                )
            return

        for n_i in range(len(model.nodes)):
            var cg = cur_globals[n_i].copy() if n_i < len(cur_globals) else model.nodes[n_i].local_transform()
            var pg = prev_globals[n_i].copy() if (has_prev and n_i < len(prev_globals)) else cg.copy()
            var cur_m = root_world.multiply(cg)
            var prev_m = use_prev_root.multiply(pg)
            var sx = Vec3(cur_m.values[0], cur_m.values[1], cur_m.values[2]).length()
            var sy = Vec3(cur_m.values[4], cur_m.values[5], cur_m.values[6]).length()
            var sz = Vec3(cur_m.values[8], cur_m.values[9], cur_m.values[10]).length()
            var u_sc = max(sx, max(sy, sz))

            if len(model.nodes[n_i].mesh_indices) > 0:
                for pm in model.nodes[n_i].mesh_indices:
                    if pm >= 0 and pm < n_meshes:
                        self.draw_instance(
                            uploaded.mesh_entries[pm].mesh_id, cur_m, prev_m, material_override, u_sc, has_prev
                        )
            elif model.nodes[n_i].mesh >= 0 and model.nodes[n_i].mesh < n_meshes:
                self.draw_instance(
                    uploaded.mesh_entries[model.nodes[n_i].mesh].mesh_id,
                    cur_m,
                    prev_m,
                    material_override,
                    u_sc,
                    has_prev,
                )

    # ========================================================================
    # Frame Submission & Execution
    # ========================================================================

    def begin_frame(
        mut self,
        cam_pos: Vec3 = Vec3(0.0, 1.5, -5.0),
        cam_target: Vec3 = Vec3(0.0, 0.0, 0.0),
        fov_y_deg: Float32 = 55.0,
        near_z: Float32 = 0.10,
        cam_roll: Float32 = 0.0,
        slot: Int = -1,
    ):
        var cur_slot = slot if (slot >= 0 and slot < len(self.ring_globals)) else (self.frame_index % FRAMES_IN_FLIGHT)
        if cur_slot < len(self.frame_fences) and self.frame_fences[cur_slot] != 0:
            _ = self.vk.wait(self.frame_fences[cur_slot])
            self.frame_fences[cur_slot] = 0

        self.buf_globals = self.ring_globals[cur_slot].copy()
        self.buf_instances = self.ring_instances[cur_slot].copy()
        self.buf_indirect = self.ring_indirect[cur_slot].copy()
        self.buf_alphamask_indirect = self.ring_alphamask_indirect[cur_slot].copy()
        self.buf_outline_indirect = self.ring_outline_indirect[cur_slot].copy()
        self.buf_clusters = self.ring_clusters[cur_slot].copy()
        self.desc_set = self.desc_sets[cur_slot]
        if self.vk.has_ray_query and len(self.ring_as_instances) > cur_slot:
            self.buf_as_instances = self.ring_as_instances[cur_slot].copy()
            self.frame_tlas = self.ring_tlas[cur_slot].copy()

        self.cam_pos = cam_pos
        var fwd = (cam_target - cam_pos).normalized()
        var r0 = view_right(fwd)
        var u0 = fwd.cross(r0).normalized()
        var right = r0
        var up = u0
        if abs(cam_roll) > Float32(0.0001):
            var cr = cos(cam_roll)
            var sr = sin(cam_roll)
            right = (r0 * cr + u0 * sr).normalized()
            up = (u0 * cr - r0 * sr).normalized()
        self.cam_fwd = fwd
        self.cam_right = right
        self.cam_up = up
        self.fov_y_deg = fov_y_deg
        self.near_z = near_z
        self.frame_instances.clear()
        self.frame_plumes.clear()
        self.frame_particle_emitters.clear()
        self.frame_cluster_lights.clear()
        self.frame_shockwaves.clear()
        self.ui_draw_list.clear()

    def begin_frame(mut self, slot: Int):
        self.begin_frame(Vec3(0.0, 1.5, -5.0), Vec3(0.0, 0.0, 0.0), slot=slot)

    def add_cluster_light(mut self, light: AaaClusterLight):
        if len(self.frame_cluster_lights) < AAA_MAX_CLUSTERED_LIGHTS:
            self.frame_cluster_lights.append(light.copy())

    def draw_instance(
        mut self,
        mesh_id: Int,
        world_mat: Mat4,
        prev_world_mat: Mat4 = Mat4(),
        material_override: Int = -1,
        uniform_scale: Float32 = 1.0,
        has_prev: Bool = False,
        wind: Float32 = 0.0,
    ):
        if (len(self.frame_instances) < self.max_instances and mesh_id >= 0 and mesh_id < len(self.geom.meshes)
                and self.geom.meshes[mesh_id].meshlet_count > 0):          # a freed / empty mesh draws nothing
            self.frame_instances.append(
                AaaDrawInstance(mesh_id, world_mat, prev_world_mat, material_override, uniform_scale, has_prev, wind)
            )

    def draw_volumetric_plume(mut self, plume: AaaVolumetricPlume):
        if len(self.frame_plumes) < AAA_MAX_OIT_PRIMS:
            self.frame_plumes.append(plume.copy())

    def submit_particle_emitter(mut self, emitter: AaaParticleEmitter):
        """Queues a GPU compute particle emitter burst/stream (`PARTICLE_SPARK_STREAK`, `WATER_SPRAY`, `DRIFT_SMOKE`,
        `EXHAUST_PLASMA`, `GENERIC`) for this frame: it spawns `spawn_count` particles, which then live on their own."""
        if len(self.frame_particle_emitters) < AAA_MAX_PARTICLE_EMITTERS and emitter.spawn_count > 0:
            self.frame_particle_emitters.append(emitter.copy())
            self.particles_ever_active = True

    def add_particle_emitter(mut self, emitter: AaaParticleEmitter):
        self.submit_particle_emitter(emitter)

    def _is_instance_alpha_masked(self, di: AaaDrawInstance, mat_final: Int) -> Bool:
        if di.mesh_id >= 0 and di.mesh_id < len(self.mesh_alpha_masked_flags) and self.mesh_alpha_masked_flags[di.mesh_id]:
            return True
        if mat_final >= 0 and mat_final < len(self.materials) and self.materials[mat_final].is_alpha_masked():
            return True
        return False

    def _pack_alphamask_visbuffer_commands(mut self, n_inst: Int, tan_x: Float32, tan_y: Float32) -> Int:
        """Packs frustum-visible alpha-masked meshlets (`VkDrawIndexedIndirectCommand`, stride 20) into `buf_alphamask_indirect`
        for Bucket 2 (`pipe_visbuffer_alphamask_gfx`), keeping Bucket 1 (`pipe_visbuffer_gfx`, `EarlyFragmentTests`) 100% pure!"""
        var cmd_cnt = 0
        var sota_want = 0  # --- sota:perf hook ---
        var base_ptr = self.buf_alphamask_indirect.cpu_address
        if base_ptr == 0 or n_inst <= 0:
            return 0
        var sec_x = sqrt(Float32(1.0) + tan_x * tan_x)
        var sec_y = sqrt(Float32(1.0) + tan_y * tan_y)
        var f_h = Float32(self.render_h)

        for i_idx in range(n_inst):
            ref di = self.frame_instances[i_idx]
            ref me = self.geom.meshes[di.mesh_id]
            var mat_final = me.material_id if di.material_override < 0 else di.material_override
            if not self._is_instance_alpha_masked(di, mat_final) or me.lod0_meshlet_count <= 0:
                continue

            ref wm = di.world_mat.values
            var bc = me.bounds_center
            var wx = wm[0] * bc.x + wm[4] * bc.y + wm[8] * bc.z + wm[12]
            var wy = wm[1] * bc.x + wm[5] * bc.y + wm[9] * bc.z + wm[13]
            var wz = wm[2] * bc.x + wm[6] * bc.y + wm[10] * bc.z + wm[14]
            var wr = me.bounds_radius * di.uniform_scale

            var rcx = wx - self.cam_pos.x
            var rcy = wy - self.cam_pos.y
            var rcz = wz - self.cam_pos.z
            var vz = rcx * self.cam_fwd.x + rcy * self.cam_fwd.y + rcz * self.cam_fwd.z
            if vz + wr < self.near_z:
                continue
            var vx_c = abs(rcx * self.cam_right.x + rcy * self.cam_right.y + rcz * self.cam_right.z)
            var vy_c = abs(rcx * self.cam_up.x + rcy * self.cam_up.y + rcz * self.cam_up.z)
            if vx_c - wr * sec_x > max(Float32(0.0), vz) * tan_x or vy_c - wr * sec_y > max(Float32(0.0), vz) * tan_y:
                continue

            # QEM LOD selection matching GPU meshlet culler
            var lod_idx = 0
            if me.lod_count > 1 and vz > Float32(1.0):
                var px_per_unit = f_h / (Float32(2.0) * vz * max(Float32(0.05), tan_y))
                for l_cand in range(1, me.lod_count):
                    if me.lods[l_cand].object_error * di.uniform_scale * px_per_unit <= self.max_lod_error_px:
                        lod_idx = l_cand
            var sl = me.lods[lod_idx]
            var first_ml = sl.first_meshlet if sl.meshlet_count > 0 else me.first_meshlet
            var ml_cnt = sl.meshlet_count if sl.meshlet_count > 0 else me.lod0_meshlet_count

            sota_want += ml_cnt  # --- sota:perf hook ---
            for m_loc in range(ml_cnt):
                if cmd_cnt >= self.sota_route_draws:  # --- sota:perf hook --- was a fixed 16384
                    break
                var ml_id = first_ml + m_loc
                ref ml = self.geom.meshlets[ml_id]
                var dst = base_ptr + 16 + cmd_cnt * 20
                put32(dst + 0, UInt32(ml.tri_count * 3))
                put32(dst + 4, UInt32(1))
                put32(dst + 8, UInt32(ml.first_tri * 3))
                var dpk = sota_draw_pack(i_idx, ml_id)  # --- sota:geometry hook --- 24-bit meshlet id (sota_geometry_visid)
                put32(dst + 12, dpk[1])
                put32(dst + 16, dpk[0])
                cmd_cnt += 1

        put32(base_ptr, UInt32(cmd_cnt))
        self.sota_route_cpu_need = max(self.sota_route_cpu_need, sota_want)  # --- sota:perf hook ---
        return cmd_cnt

    def _pack_outline_visbuffer_commands(mut self, n_inst: Int, tan_x: Float32, tan_y: Float32) -> Int:
        """Packs frustum-visible cel-outlined meshlets (`VkDrawIndexedIndirectCommand`, stride 20) into
        `buf_outline_indirect` for `pipe_npr_outline_gfx` (`CULL_MODE_FRONT`). Zero overhead when `len(cel_materials) == 0`!"""
        if len(self.cel_materials) == 0 or n_inst <= 0:
            return 0
        var base_ptr = self.buf_outline_indirect.cpu_address
        if base_ptr == 0:
            return 0
        var cmd_cnt = 0
        var sota_want = 0  # --- sota:perf hook ---
        var sec_x = sqrt(Float32(1.0) + tan_x * tan_x)
        var sec_y = sqrt(Float32(1.0) + tan_y * tan_y)
        var f_h = Float32(self.render_h)

        for i_idx in range(n_inst):
            ref di = self.frame_instances[i_idx]
            ref me = self.geom.meshes[di.mesh_id]
            var mat_final = me.material_id if di.material_override < 0 else di.material_override
            if not self.is_cel_outlined_material(mat_final) or me.lod0_meshlet_count <= 0:
                continue

            ref wm = di.world_mat.values
            var bc = me.bounds_center
            var wx = wm[0] * bc.x + wm[4] * bc.y + wm[8] * bc.z + wm[12]
            var wy = wm[1] * bc.x + wm[5] * bc.y + wm[9] * bc.z + wm[13]
            var wz = wm[2] * bc.x + wm[6] * bc.y + wm[10] * bc.z + wm[14]
            var wr = me.bounds_radius * di.uniform_scale
            if di.mesh_id >= 0 and di.mesh_id < len(self.skinned_mesh_flags) and self.skinned_mesh_flags[di.mesh_id]:
                wr = wr * Float32(4.0) + Float32(4.0)

            var rcx = wx - self.cam_pos.x
            var rcy = wy - self.cam_pos.y
            var rcz = wz - self.cam_pos.z
            var vz = rcx * self.cam_fwd.x + rcy * self.cam_fwd.y + rcz * self.cam_fwd.z
            if vz + wr < self.near_z:
                continue
            var vx_c = abs(rcx * self.cam_right.x + rcy * self.cam_right.y + rcz * self.cam_right.z)
            var vy_c = abs(rcx * self.cam_up.x + rcy * self.cam_up.y + rcz * self.cam_up.z)
            if vx_c - wr * sec_x > max(Float32(0.0), vz) * tan_x or vy_c - wr * sec_y > max(Float32(0.0), vz) * tan_y:
                continue

            var lod_idx = 0
            if me.lod_count > 1 and vz > Float32(1.0):
                var px_per_unit = f_h / (Float32(2.0) * vz * max(Float32(0.05), tan_y))
                for l_cand in range(1, me.lod_count):
                    if me.lods[l_cand].object_error * di.uniform_scale * px_per_unit <= self.max_lod_error_px:
                        lod_idx = l_cand
            var sl = me.lods[lod_idx]
            var first_ml = sl.first_meshlet if sl.meshlet_count > 0 else me.first_meshlet
            var ml_cnt = sl.meshlet_count if sl.meshlet_count > 0 else me.lod0_meshlet_count

            sota_want += ml_cnt  # --- sota:perf hook ---
            for m_loc in range(ml_cnt):
                if cmd_cnt >= self.sota_route_draws:  # --- sota:perf hook --- was a fixed 16384
                    break
                var ml_id = first_ml + m_loc
                ref ml = self.geom.meshlets[ml_id]
                var dst = base_ptr + 16 + cmd_cnt * 20
                put32(dst + 0, UInt32(ml.tri_count * 3))
                put32(dst + 4, UInt32(1))
                put32(dst + 8, UInt32(ml.first_tri * 3))
                var dpk = sota_draw_pack(i_idx, ml_id)  # --- sota:geometry hook --- 24-bit meshlet id (sota_geometry_visid)
                put32(dst + 12, dpk[1])
                put32(dst + 16, dpk[0])
                cmd_cnt += 1

        put32(base_ptr, UInt32(cmd_cnt))
        self.sota_route_cpu_need = max(self.sota_route_cpu_need, sota_want)  # --- sota:perf hook ---
        return cmd_cnt

    def _pack_vsm_indirect_commands(
        mut self,
        n_inst: Int,
        light_vp: List[Float32],
        vsm_allocs: List[Tuple[Int, Int, Int, Int]],
    ) -> Tuple[Int, List[Int], List[Int], List[Int], List[Int]]:
        """Per-page light-space bounding-sphere culler + QEM coarse shadow LOD packer into `buf_vsm_indirect`.
        Separates Opaque (`pipe_vsm_gfx`) vs Alpha-Masked (`pipe_vsm_alphamask_gfx`) commands per dirty `128x128` VSM page!"""
        var n_pages = len(vsm_allocs)
        var page_offsets = List[Int](length=n_pages, fill=0)
        var page_counts = List[Int](length=n_pages, fill=0)
        var page_am_offsets = List[Int](length=n_pages, fill=0)
        var page_am_counts = List[Int](length=n_pages, fill=0)
        if n_pages == 0 or n_inst == 0:
            return (0, page_offsets^, page_counts^, page_am_offsets^, page_am_counts^)

        var use_vsm_casc = self.enable_vsm_clipmap_cascades and len(self.vsm_cascade_matrices) >= VSM_CASCADE_COUNT
        var base_vsm_ext = max(Float32(1.0), self.vsm_ortho_half_extent)

        var casc_has_dirty = List[Bool](length=VSM_CASCADE_COUNT, fill=False)
        var casc_min_vx = List[Int](length=VSM_CASCADE_COUNT, fill=127)
        var casc_max_vx = List[Int](length=VSM_CASCADE_COUNT, fill=0)
        var casc_min_vy = List[Int](length=VSM_CASCADE_COUNT, fill=127)
        var casc_max_vy = List[Int](length=VSM_CASCADE_COUNT, fill=0)

        var page_slot = List[Int](length=16384, fill=-1)
        var min_pvx = 127; var max_pvx = 0
        var min_pvy = 127; var max_pvy = 0
        for p_i in range(n_pages):
            var pvx = vsm_allocs[p_i][0]
            var pvy = vsm_allocs[p_i][1]
            if use_vsm_casc and self.sota_shadow.toroidal_active and pvx >= 0 and pvx < 128 and pvy >= 0 and pvy < 128:
                # --- sota:shadow hook --- toroidal slot -> window-local page (the culling below is window-relative)
                var loc_p = sota_shadow_local_from_slot(pvx, pvy, self.sota_shadow.offsets)
                pvx = (pvx & 64) | loc_p[0]
                pvy = (pvy & 64) | loc_p[1]
            if pvx >= 0 and pvx < 128 and pvy >= 0 and pvy < 128:
                page_slot[pvy * 128 + pvx] = p_i
                if pvx < min_pvx: min_pvx = pvx
                if pvx > max_pvx: max_pvx = pvx
                if pvy < min_pvy: min_pvy = pvy
                if pvy > max_pvy: max_pvy = pvy
                if use_vsm_casc:
                    var c_idx = vsm_cascade_from_virtual_page(pvx, pvy)
                    casc_has_dirty[c_idx] = True
                    if pvx < casc_min_vx[c_idx]: casc_min_vx[c_idx] = pvx
                    if pvx > casc_max_vx[c_idx]: casc_max_vx[c_idx] = pvx
                    if pvy < casc_min_vy[c_idx]: casc_min_vy[c_idx] = pvy
                    if pvy > casc_max_vy[c_idx]: casc_max_vy[c_idx] = pvy

        var per_page = List[List[Tuple[UInt32, UInt32]]]()
        var per_page_am = List[List[Tuple[UInt32, UInt32]]]()
        for _ in range(n_pages):
            per_page.append(List[Tuple[UInt32, UInt32]]())
            per_page_am.append(List[Tuple[UInt32, UInt32]]())

        var page_scale_single = Float32(64.0) / base_vsm_ext
        var n_casc_pass = VSM_CASCADE_COUNT if use_vsm_casc else 1
        var pass_vps = List[List[Float32]](capacity=n_casc_pass)
        if use_vsm_casc:
            for c_p in range(VSM_CASCADE_COUNT):
                pass_vps.append(self.vsm_cascade_matrices[c_p].copy())
        else:
            pass_vps.append(light_vp.copy())

        for i_idx in range(n_inst):
            ref di = self.frame_instances[i_idx]
            ref me = self.geom.meshes[di.mesh_id]
            if me.lod0_meshlet_count <= 0:
                continue
            var mat_final_v = me.material_id if di.material_override < 0 else di.material_override
            if len(self.glass_materials) > 0 and self.is_glass_material(mat_final_v):
                continue  # glass casts no VSM shadow (RT shadows dim the sun through it instead)
            var is_am_v = self._is_instance_alpha_masked(di, mat_final_v)
            ref wm = di.world_mat.values
            var s0 = wm[0] * wm[0] + wm[1] * wm[1] + wm[2] * wm[2]
            var s1 = wm[4] * wm[4] + wm[5] * wm[5] + wm[6] * wm[6]
            var s2 = wm[8] * wm[8] + wm[9] * wm[9] + wm[10] * wm[10]
            var max_sc = sqrt(max(Float32(1e-6), max(s0, max(s1, s2))))

            var bc = me.bounds_center
            var wx = wm[0] * bc.x + wm[4] * bc.y + wm[8] * bc.z + wm[12]
            var wy = wm[1] * bc.x + wm[5] * bc.y + wm[9] * bc.z + wm[13]
            var wz = wm[2] * bc.x + wm[6] * bc.y + wm[10] * bc.z + wm[14]
            var wr = me.bounds_radius * max_sc

            # Select coarse QEM LOD for compact shadow casters (crowds, props, distant cars/trees)
            var lod_idx = 0
            if me.lod_count > 1 and wr < Float32(5.5):
                lod_idx = min(me.lod_count - 1, 2)
            var sl = me.lods[lod_idx]
            var first_ml = sl.first_meshlet if sl.meshlet_count > 0 else me.first_meshlet
            var ml_cnt = sl.meshlet_count if sl.meshlet_count > 0 else me.lod0_meshlet_count

            for c_pass in range(n_casc_pass):
                if use_vsm_casc and not casc_has_dirty[c_pass]:
                    continue
                ref cur_vp = pass_vps[c_pass]
                var q_orig = vsm_cascade_quadrant_origin(c_pass) if use_vsm_casc else (0, 0)
                var grid_f = Float32(VSM_CASCADE_PAGES_DIM) if use_vsm_casc else Float32(128.0)
                var page_scale = (
                    (Float32(32.0) / (base_vsm_ext * Float32(1 << (2 * c_pass))))
                    if use_vsm_casc
                    else page_scale_single
                )
                var b_min_vx = casc_min_vx[c_pass] if use_vsm_casc else min_pvx
                var b_max_vx = casc_max_vx[c_pass] if use_vsm_casc else max_pvx
                var b_min_vy = casc_min_vy[c_pass] if use_vsm_casc else min_pvy
                var b_max_vy = casc_max_vy[c_pass] if use_vsm_casc else max_pvy

                var lx = cur_vp[0] * wx + cur_vp[4] * wy + cur_vp[8] * wz + cur_vp[12]
                var ly = cur_vp[1] * wx + cur_vp[5] * wy + cur_vp[9] * wz + cur_vp[13]
                var lz = cur_vp[2] * wx + cur_vp[6] * wy + cur_vp[10] * wz + cur_vp[14]
                if lz < Float32(-0.5) or lz > Float32(1.5):
                    continue
                var fx = Float32(q_orig[0]) + (lx * Float32(0.5) + Float32(0.5)) * grid_f
                var fy = Float32(q_orig[1]) + (-ly * Float32(0.5) + Float32(0.5)) * grid_f
                var fr = wr * page_scale + Float32(0.35)

                if (
                    fx + fr < Float32(b_min_vx)
                    or fx - fr > Float32(b_max_vx + 1)
                    or fy + fr < Float32(b_min_vy)
                    or fy - fr > Float32(b_max_vy + 1)
                ):
                    continue

                var vx0 = max(b_min_vx, Int(fx - fr))
                var vx1 = min(b_max_vx, Int(fx + fr))
                var vy0 = max(b_min_vy, Int(fy - fr))
                var vy1 = min(b_max_vy, Int(fy + fr))
                if vx0 > vx1 or vy0 > vy1:
                    continue

                if fr <= Float32(2.5):
                    var hit_pages = List[Int]()
                    for py_i in range(vy0, vy1 + 1):
                        var row_off = py_i * 128
                        for px_i in range(vx0, vx1 + 1):
                            var ps = page_slot[row_off + px_i]
                            if ps >= 0:
                                hit_pages.append(ps)
                    if len(hit_pages) == 0:
                        continue
                    for m_loc in range(ml_cnt):
                        var ml_id = first_ml + m_loc
                        var v_cnt_u = UInt32(self.geom.meshlets[ml_id].tri_count * 3) | ((UInt32(ml_id) >> 17) << 16)  # --- sota:geometry hook --- 24-bit meshlet id (sota_geometry_visid): meshlet hi bits
                        var pk_u = UInt32((i_idx << 17) | (ml_id & 0x1FFFF))
                        for hp in hit_pages:
                            if is_am_v:
                                per_page_am[hp].append((v_cnt_u, pk_u))
                            else:
                                per_page[hp].append((v_cnt_u, pk_u))
                else:
                    # Fine per-meshlet light-space sphere culling for large terrain chunks, tracks & megatowers
                    var any_dirty = False
                    for py_i in range(vy0, vy1 + 1):
                        var row_off = py_i * 128
                        for px_i in range(vx0, vx1 + 1):
                            if page_slot[row_off + px_i] >= 0:
                                any_dirty = True
                                break
                        if any_dirty:
                            break
                    if not any_dirty:
                        continue
                    for m_loc in range(ml_cnt):
                        var ml_id = first_ml + m_loc
                        ref ml = self.geom.meshlets[ml_id]
                        var mc = ml.center
                        var mwx = wm[0] * mc.x + wm[4] * mc.y + wm[8] * mc.z + wm[12]
                        var mwy = wm[1] * mc.x + wm[5] * mc.y + wm[9] * mc.z + wm[13]
                        var mwz = wm[2] * mc.x + wm[6] * mc.y + wm[10] * mc.z + wm[14]
                        var mwr = ml.radius * max_sc
                        var mlx = cur_vp[0] * mwx + cur_vp[4] * mwy + cur_vp[8] * mwz + cur_vp[12]
                        var mly = cur_vp[1] * mwx + cur_vp[5] * mwy + cur_vp[9] * mwz + cur_vp[13]
                        var mfx = Float32(q_orig[0]) + (mlx * Float32(0.5) + Float32(0.5)) * grid_f
                        var mfy = Float32(q_orig[1]) + (-mly * Float32(0.5) + Float32(0.5)) * grid_f
                        var mfr = mwr * page_scale + Float32(0.30)
                        var mvx0 = max(vx0, Int(mfx - mfr))
                        var mvx1 = min(vx1, Int(mfx + mfr))
                        var mvy0 = max(vy0, Int(mfy - mfr))
                        var mvy1 = min(vy1, Int(mfy + mfr))
                        if mvx0 > mvx1 or mvy0 > mvy1:
                            continue
                        var v_cnt_u = UInt32(ml.tri_count * 3) | ((UInt32(ml_id) >> 17) << 16)  # --- sota:geometry hook --- 24-bit meshlet id (sota_geometry_visid): meshlet hi bits
                        var pk_u = UInt32((i_idx << 17) | (ml_id & 0x1FFFF))
                        for py_i in range(mvy0, mvy1 + 1):
                            var row_off = py_i * 128
                            for px_i in range(mvx0, mvx1 + 1):
                                var ps = page_slot[row_off + px_i]
                                if ps >= 0:
                                    if is_am_v:
                                        per_page_am[ps].append((v_cnt_u, pk_u))
                                    else:
                                        per_page[ps].append((v_cnt_u, pk_u))

        # --- sota:shadow hook --- size the command buffer by what this frame needs. It was capped at `max_draws`
        # (visible meshlets), truncated pages were cleared to lit and then cached as valid: Forest lost its shadows.
        var need_cmds = 0
        for p_i in range(n_pages):
            need_cmds += len(per_page[p_i]) + len(per_page_am[p_i])
        self.sota_shadow.vsm_cmds_last = need_cmds
        var cap_cmds = self._sota_shadow_ensure_vsm_cmds(need_cmds)
        # --- end sota:shadow hook ---
        var cmd_cnt = 0
        var base_ptr = self.buf_vsm_indirect.cpu_address
        for p_i in range(n_pages):
            page_offsets[p_i] = cmd_cnt
            ref plist = per_page[p_i]
            var written = 0
            for cmd in plist:
                if cmd_cnt >= cap_cmds:  # sota:shadow hook (was max_draws)
                    break
                var dst = base_ptr + cmd_cnt * 16
                put32(dst + 0, cmd[0] & UInt32(0xFFFF))
                put32(dst + 4, UInt32(1))
                put32(dst + 8, (cmd[0] >> 16) << 24)  # sota:geometry hook: firstVertex carries meshlet bits 17..23
                put32(dst + 12, cmd[1])
                cmd_cnt += 1
                written += 1
            page_counts[p_i] = written

            page_am_offsets[p_i] = cmd_cnt
            ref am_list = per_page_am[p_i]
            var written_am = 0
            for cmd in am_list:
                if cmd_cnt >= cap_cmds:  # sota:shadow hook (was max_draws)
                    break
                var dst = base_ptr + cmd_cnt * 16
                put32(dst + 0, cmd[0] & UInt32(0xFFFF))
                put32(dst + 4, UInt32(1))
                put32(dst + 8, (cmd[0] >> 16) << 24)  # sota:geometry hook: firstVertex carries meshlet bits 17..23
                put32(dst + 12, cmd[1])
                cmd_cnt += 1
                written_am += 1
            page_am_counts[p_i] = written_am
            # --- sota:shadow hook --- a page missing casters must not be cached: leave it invalid, retry next frame
            if written + written_am < len(plist) + len(am_list):
                self.sota_shadow.vsm_cmds_dropped_total += len(plist) + len(am_list) - written - written_am
                self.sota_shadow.vsm_pages_retried_total += 1
                var tvx = vsm_allocs[p_i][0]
                var tvy = vsm_allocs[p_i][1]
                if tvx >= 0 and tvx < 128 and tvy >= 0 and tvy < 128:
                    self.vsm.virtual_valid[tvy * 128 + tvx] = False
            # --- end sota:shadow hook ---

        return (cmd_cnt, page_offsets^, page_counts^, page_am_offsets^, page_am_counts^)

    # --- sota:shadow hook ---
    def _sota_shadow_ensure_vsm_cmds(mut self, need: Int) -> Int:
        """Grows `buf_vsm_indirect` to hold `need` commands (+25 %); returns the capacity actually available.
        Waits for the GPU (the previous frame may still read the old buffer); growth is rare (first big frame)."""
        var cap = self.sota_shadow.vsm_cmd_capacity
        if need <= cap or not self.active:
            return cap
        var target = sota_shadow_vsm_cmd_target(need)
        self.vk.wait_idle()
        var nb = self.vk.create_buffer(64 + target * 16, local=True)
        if nb.size == 0:
            self._sota_log_once("out of memory growing the VSM command buffer to " + String(target) + " commands; "
                                + "pages that do not fit stay invalid and are retried")
            return cap
        self.vk.destroy_buffer(self.buf_vsm_indirect)
        self.buf_vsm_indirect = nb^
        self.sota_shadow.vsm_cmd_capacity = target
        self.sota_shadow.vsm_cmd_grows += 1
        return target
    # --- end sota:shadow hook ---

    def ui_prims_word_offset(self, slot: Int) -> Int:
        """Where a frame slot's GPU UI prims live in `buf_post_scratch`: past every per-pixel post
        region (RT shade, water and motion blur stage 8 floats per pixel from word 1024, bloom and
        DoF sit below 1024 + max_pix * 10), so no pass of the frame overwrites them before
        `UiSdfOverlayPass` reads them, and one region per frame in flight, so recording the next
        frame never rewrites prims the GPU is still drawing."""
        var max_pix = max(self.output_w * self.output_h, self.render_w * self.render_h)
        return AAA_POST_UI_PRIMS_BASE_OFFSET + max_pix * 10 + (slot % FRAMES_IN_FLIGHT) * AAA_UI_PRIM_REGION_WORDS

    # --- sota:atmo hook ---
    def _sota_atmo_prepare(mut self):
        """Area H: sun / moon for this frame, the renderer's directional light from the atmosphere, and the IBL
        (SH9 + GGX cube) re-baked from the physical sky when a light moved more than `ibl_update_deg`."""
        if not self.sota_atmo.active() or self.furnace_white_dome:
            return
        self.sota_atmo.update_lights()
        if self.sota_atmo.drive_renderer_light:
            var pl = self.sota_atmo.primary_light(self.cam_pos.y)
            self.sun_dir = pl[0]
            self.sun_radiance = pl[1]
            self.sun_angular_radius = pl[2]
        if self.sota_atmo.needs_ibl_refresh():
            var t0 = get_time()
            var ibl = self.sota_atmo.bake_ibl(self.cam_pos.y, self.env_cube_dim, self.env_mip_levels)
            self.upload_ibl_environment(ibl)
            self.sota_atmo.last_ibl_ms = (get_time() - t0) * 1000.0
    # --- end sota:atmo hook ---

    def _sota_shadow_gpu_prepare(mut self, slot: Int, n_inst: Int, eligible: Bool) -> Bool:
        """--- sota:shadow hook --- `ren.sota_shadow.gpu_pages`: switches `BIND_AAA_VSM_TABLE` between the v5 table
        and the GPU path's buffer, uploads this frame's caster items + invalidations. False = run the CPU path."""
        var want = self.sota_shadow.gpu_pages and eligible and self.sota_shadow_gpu.init(self.vk, self.pipe_layout)
        if want:
            # caster items: same caster set / LOD choice / glass skip / alpha split as `_pack_vsm_indirect_commands`
            var items = List[UInt32]()
            for i_idx in range(min(n_inst, 32767)):
                ref di = self.frame_instances[i_idx]
                ref me = self.geom.meshes[di.mesh_id]
                if me.lod0_meshlet_count <= 0:
                    continue
                var mat_final_v = me.material_id if di.material_override < 0 else di.material_override
                if len(self.glass_materials) > 0 and self.is_glass_material(mat_final_v):
                    continue
                var am_bit = UInt32(0x80000000) if self._is_instance_alpha_masked(di, mat_final_v) else UInt32(0)
                ref wm = di.world_mat.values
                var s0 = wm[0] * wm[0] + wm[1] * wm[1] + wm[2] * wm[2]
                var s1 = wm[4] * wm[4] + wm[5] * wm[5] + wm[6] * wm[6]
                var s2 = wm[8] * wm[8] + wm[9] * wm[9] + wm[10] * wm[10]
                var wr = me.bounds_radius * sqrt(max(Float32(1e-6), max(s0, max(s1, s2))))
                var lod_idx = 0
                if me.lod_count > 1 and wr < Float32(5.5):
                    lod_idx = min(me.lod_count - 1, 2)
                var sl = me.lods[lod_idx]
                var first_ml = sl.first_meshlet if sl.meshlet_count > 0 else me.first_meshlet
                var ml_cnt = sl.meshlet_count if sl.meshlet_count > 0 else me.lod0_meshlet_count
                var m0 = 0
                while m0 < ml_cnt:
                    items.append(UInt32(i_idx) | am_bit)
                    items.append(UInt32(first_ml + m0))
                    items.append(UInt32(min(SOTA_SHADOW_ITEM_MESHLETS, ml_cnt - m0)))
                    m0 += SOTA_SHADOW_ITEM_MESHLETS
            self.sota_shadow_gpu.read_stats()
            var cmd_need = max(131072, self.sota_shadow_gpu.wanted_max)
            var switched = not self.sota_shadow.gpu_was_active
            if self.sota_shadow_gpu.ensure_buffer(self.vk, len(items) // 3, cmd_need) or switched:
                self.vk.wait_idle()
                if switched:
                    sota_shadow_reset_tables(self.sota_shadow_gpu.buf)
                    self.vsm.invalidate_all()
                for ds_i in range(len(self.desc_sets)):
                    self.vk.write_descriptor(self.desc_sets[ds_i], BIND_AAA_VSM_TABLE, self.sota_shadow_gpu.buf)
            self.sota_shadow_gpu.upload(slot, items, self.vsm.virtual_valid)
            self.last_vsm_dirty_pages = self.sota_shadow_gpu.dirty_last
        elif self.sota_shadow.gpu_was_active:
            # back to the CPU path: the v5 table / allocator start empty
            self.vk.wait_idle()
            self.vsm = VsmPageManager()
            _ = external_call["memset", Int](self.buf_vsm_table.cpu_address, Int32(0), self.buf_vsm_table.size)
            for ds_i in range(len(self.desc_sets)):
                self.vk.write_descriptor(self.desc_sets[ds_i], BIND_AAA_VSM_TABLE, self.buf_vsm_table)
        self.sota_shadow.gpu_was_active = want
        self.sota_shadow.gpu_active = want
        return want

    def render_frame(mut self, capture_readback: Bool = False, force_readback: Bool = False) -> Bool:
        if not self.active:
            return self._sota_frame_fail("renderer is not active (Vulkan init failed or shut down)")  # sota:geometry hook

        # Deferred readback: retire the previous frame first (no-op in the default synchronous mode). Waiting here,
        # before any host write, keeps every single-instance HOST_VISIBLE buffer race-free (Phase A).
        var t_wait0 = get_time()
        self._retire_pending_frame()
        var t_pack0 = get_time()
        self.last_cpu_prev_wait_ms = (t_pack0 - t_wait0) * 1000.0

        _ = self.streamer.tick_streaming_budget()
        self._sota_gi_prepare()  # --- sota:gi hook ---
        self._sota_atmo_prepare()  # --- sota:atmo hook ---

        # --- sota:camera hook (exposure triangle -> exposure, shutter, legacy lens) ---
        if self.sota_cam.ae.enable:  # round 4: photometric calibration of the EV100 meter (sky) and the camera
            if self.sota_atmo.active():
                self.sota_cam.ae.calibrate_to_sky(rgb_luma(self.sota_atmo.params.solar_irradiance))
            if self.sota_cam.ae.calibrate_camera:
                self.sota_cam.camera.ev100_unity = self.sota_cam.ae.ev100_unity()
        if self.sota_cam.drive_renderer:
            var ae_mul = Float32(1.0)
            if self.enable_auto_exposure and self.frame_index > 0:
                ae_mul = self.get_adapted_exposure()[0]
            var cam_sync = self.sota_cam.frame_sync(ae_mul, self.enable_auto_exposure)
            self.exposure = cam_sync[0]
            self.motion_blur_shutter = cam_sync[1]
            self.camera_lens.aperture_fstop = self.sota_cam.camera.fstop
            self.camera_lens.shutter_seconds = self.sota_cam.camera.shutter_s
            self.camera_lens.iso = self.sota_cam.camera.iso
            self.camera_lens.focal_length_mm = self.sota_cam.camera.focal_length_mm
            self.camera_lens.sensor_height_mm = self.sota_cam.camera.sensor_height_mm
            self.camera_lens.focus_distance_m = self.sota_cam.camera.focus_distance_m
        # --- end sota:camera hook ---

        var cur_slot = self.frame_index % FRAMES_IN_FLIGHT
        self.sota_geom.begin_slot(cur_slot)  # sota:geometry hook
        # --- sota:geometry hook --- visible meshlets over max_draws are counted, logged and the buffers grow
        var sota_need_draws = self.sota_geom.check_draw_overflow(self.max_draws)
        if sota_need_draws > 0:
            self._sota_log_once("visible meshlets exceeded max_draws " + String(self.max_draws) + " (needed "
                                + String(sota_need_draws) + "); growing the draw buffers")
            _ = self._sota_grow_draws(sota_need_draws + sota_need_draws // 4)
        # --- end sota:geometry hook ---
        self._sota_check_route_overflow(cur_slot)  # --- sota:perf hook --- alpha-mask / outline ring growth
        self.sota_vt.begin_frame(cur_slot, (self.frame_index + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT)  # sota:geometry hook
        var defer_this = self.deferred_readback and (not capture_readback) and self.vk.swapchain == 0
        if defer_this and cur_slot > 0 and len(self.readback_extra) < cur_slot:
            while len(self.readback_extra) < FRAMES_IN_FLIGHT - 1:
                var rb = self.vk.create_buffer(self.buf_ldr_readback.size, local=False, cpu_reads=True)
                if rb.cpu_address == 0:
                    break
                _ = external_call["memset", Int](rb.cpu_address, Int32(0), rb.size)
                self.readback_extra.append(rb^)
            if len(self.readback_extra) < cur_slot:
                defer_this = False
        var rb_buf = self._readback_buffer(cur_slot) if defer_this else self.buf_ldr_readback.copy()
        self.buf_globals = self.ring_globals[cur_slot].copy()
        self.buf_instances = self.ring_instances[cur_slot].copy()
        self.buf_indirect = self.ring_indirect[cur_slot].copy()
        self.buf_alphamask_indirect = self.ring_alphamask_indirect[cur_slot].copy()
        self.buf_outline_indirect = self.ring_outline_indirect[cur_slot].copy()
        self.buf_clusters = self.ring_clusters[cur_slot].copy()
        self.desc_set = self.desc_sets[cur_slot]
        if self.vk.has_ray_query and len(self.ring_as_instances) > cur_slot:
            self.buf_as_instances = self.ring_as_instances[cur_slot].copy()
            self.frame_tlas = self.ring_tlas[cur_slot].copy()

        var n_inst = len(self.frame_instances)
        var max_ml_per_inst = 1
        var last_large_sc_inst = -1
        var sc_tri_counts = List[Int](capacity=min(1024, max(1, n_inst)))
        var sc_cursor = 1024  # Leave words [0..4095] for up to 1,024 instance headers!
        var sc_inst_count = min(1024, n_inst)

        # 1. Upload Instance SSBO + Surface-Cache Headers + TLAS Instance Records
        var frame_has_glass = False
        self.sota_e_gpu.begin_frame()  # sota:E hook
        for i in range(n_inst):
            ref di = self.frame_instances[i]
            ref me = self.geom.meshes[di.mesh_id]
            if me.meshlet_count > max_ml_per_inst:
                max_ml_per_inst = me.meshlet_count
            var dst = self.buf_instances.cpu_address + i * AAA_INSTANCE_WORDS * 4
            for k in range(16):
                put_float(dst + k * 4, di.world_mat.values[k])
                put_float(dst + (16 + k) * 4, di.prev_world_mat.values[k])
            put32(dst + 32 * 4, UInt32(me.first_meshlet))
            put32(dst + 33 * 4, UInt32(me.meshlet_count))
            var is_sk = di.mesh_id < len(self.skinned_mesh_flags) and self.skinned_mesh_flags[di.mesh_id]
            var inst_fl = UInt32(0x00010000) if is_sk else UInt32(0)
            var mat_ov = UInt32(0xFFFF) if di.material_override < 0 else UInt32(di.material_override & 0xFFFF)
            var wind8 = UInt32(Int(di.wind * Float32(255.0) + Float32(0.5))) << UInt32(AAA_INSTANCE_WIND_SHIFT)
            put32(dst + 34 * 4, inst_fl | mat_ov | wind8)
            put_float(dst + 35 * 4, di.uniform_scale)
            var sk_slot = self.skinned_mesh_bind_slots[di.mesh_id] if (is_sk and di.mesh_id < len(self.skinned_mesh_bind_slots)) else 0
            put32(dst + 36 * 4, UInt32(sk_slot))
            put32(dst + 37 * 4, UInt32(me.first_vertex))

            var mat_final = me.material_id if di.material_override < 0 else di.material_override
            var is_am_inst = self._is_instance_alpha_masked(di, mat_final)
            var is_glass_inst = len(self.glass_materials) > 0 and self.is_glass_material(mat_final)
            if is_glass_inst:
                frame_has_glass = True
            var is_ds = (di.mesh_id < len(self.mesh_double_sided_flags) and self.mesh_double_sided_flags[di.mesh_id]) or is_am_inst
            put32(dst + 38 * 4, UInt32(me.first_index))
            # --- sota:E hook --- bit 19 routes cel-outlined instances to the GPU outline packer
            var sota_e_outl = len(self.cel_materials) > 0 and self.is_cel_outlined_material(mat_final)
            if is_am_inst:
                self.sota_e_gpu.frame_alpha_instances += 1
            if sota_e_outl:
                self.sota_e_gpu.frame_outline_instances += 1
            # --- end sota:E hook ---
            put32(
                dst + 39 * 4,
                UInt32(mat_final & 0xFFFF)
                | ((UInt32(1) if is_ds else UInt32(0)) << 16)
                | ((UInt32(1) if is_am_inst else UInt32(0)) << 17)
                | (AAA_INSTANCE_GLASS_BIT if is_glass_inst else UInt32(0))
                | (SOTA_E_INSTANCE_OUTLINE_BIT if sota_e_outl else UInt32(0)),  # sota:E hook
            )
            var my_sc_base = 0xFFFFFF
            if i < 1024:
                var sc_hdr = self.buf_surfcache.cpu_address + i * 16
                if me.triangle_count <= 4096 and sc_cursor + me.triangle_count < self.max_surfcache_tris:
                    my_sc_base = sc_cursor
                    sc_cursor += me.triangle_count
                    sc_tri_counts.append(me.triangle_count)
                    if me.triangle_count > 2048:
                        last_large_sc_inst = i
                    put32(sc_hdr + 0, UInt32(my_sc_base))
                    put32(sc_hdr + 4, UInt32(me.first_index))
                    put32(sc_hdr + 8, UInt32(me.triangle_count))
                    put32(sc_hdr + 12, UInt32(mat_final))
                else:
                    sc_tri_counts.append(0)
                    put32(sc_hdr + 0, UInt32(0xFFFFFF))
                    put32(sc_hdr + 4, UInt32(0))
                    put32(sc_hdr + 8, UInt32(0))
                    put32(sc_hdr + 12, UInt32(0))

            if self.vk.has_ray_query and di.mesh_id < len(self.mesh_blas):
                var blas_da = self.mesh_blas[di.mesh_id].device_address
                write_vk_as_instance_64b(
                    self.buf_as_instances.cpu_address + i * 64,
                    di.world_mat,
                    my_sc_base,
                    blas_da,
                    cull_disable=True,
                    alpha_masked=is_am_inst,
                    glass=is_glass_inst,
                )

        # Upload 6-Way OIT Volumetric Plumes
        if len(self.frame_plumes) > 0:
            var oit_w = List[UInt32]()
            for pl in self.frame_plumes:
                pl.append_words(oit_w)
            _ = external_call["memcpy", Int](self.buf_oit_nodes.cpu_address, Int(oit_w.unsafe_ptr()), len(oit_w) * 4)
            _ = len(oit_w)

        # 2. Upload Camera + Light + VSM + SH9 Globals SSBO
        var aspect = Float32(self.render_w) / Float32(max(1, self.render_h))
        var fov_rad = self.fov_y_deg * Float32(3.14159265 / 180.0)
        var tan_y = tan(fov_rad * Float32(0.5))
        var tan_x = tan_y * aspect
        var view_m = _build_camera_view_col_major(self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up)
        var proj_m = make_reverse_z_perspective(fov_rad, aspect, self.near_z)
        var vp_m = _mat4_mul_col_major(proj_m, view_m)
        if not self.has_prev_vp:
            self.prev_view_proj = vp_m.copy()
            self.has_prev_vp = True

        var base_vsm_ext = max(Float32(1.0), self.vsm_ortho_half_extent)
        var light_vp = make_ortho_directional_light_vp(
            self.sun_dir, self.vsm_scene_center, base_vsm_ext
        )
        self.vsm_cascade_matrices = compute_vsm_cascade_matrices(
            self.sun_dir, self.cam_pos, base_vsm_ext
        )
        var use_vsm_casc_frame = (
            self.enable_vsm_clipmap_cascades and len(self.vsm_cascade_matrices) >= VSM_CASCADE_COUNT
        )
        if use_vsm_casc_frame:
            light_vp = self.vsm_cascade_matrices[0].copy()
        # --- sota:shadow hook --- cascade window moves: toroidal scroll / depth-snap invalidation, this frame's offsets
        if self.enable_vsm and use_vsm_casc_frame:
            var sh_origins = compute_vsm_cascade_page_origins(self.sun_dir, self.cam_pos, base_vsm_ext)
            self.sota_shadow.update_cascades(
                self.vsm, self.vsm_cascade_page_origins, sh_origins,
                sota_shadow_depth_snap_index(self.sun_dir, self.cam_pos), True,
            )
            if self.sota_shadow.toroidal_active:
                self.vsm_cascade_page_origins = sh_origins^
        else:
            self.sota_shadow.update_cascades(self.vsm, self.vsm_cascade_page_origins, self.vsm_cascade_page_origins, 0, False)
        # --- end sota:shadow hook ---
        # --- sota:perf hook --- keep cached VSM pages valid under light rotation and moving / skinned casters
        if self.enable_vsm and self.sota_vsm.enabled:
            self.sota_vsm.update_light(self.vsm, self.sun_dir, self.sun_radiance)
            self.sota_vsm.begin_casters()
            if use_vsm_casc_frame:
                for inst_i in range(len(self.frame_instances)):
                    ref di = self.frame_instances[inst_i]
                    if di.mesh_id < 0 or di.mesh_id >= len(self.geom.meshes):
                        continue
                    var deforming = di.mesh_id < len(self.skinned_mesh_flags) and self.skinned_mesh_flags[di.mesh_id]
                    self.sota_vsm.caster(
                        self.vsm, self.vsm_cascade_matrices, di.world_mat, di.prev_world_mat,
                        self.geom.meshes[di.mesh_id].bounds_center, self.geom.meshes[di.mesh_id].bounds_radius,
                        di.uniform_scale, deforming, self.sota_shadow.offsets,  # sota:shadow hook (toroidal slots)
                    )
        elif self.sota_vsm.enabled:
            _ = self.sota_vsm.track_light(self.sun_dir, self.sun_radiance)
        # --- end sota:perf hook ---

        # True 16-phase radical-inverse Halton (2, 3) sub-pixel TSR jitter across ALL `tsr_scale` values (including native 1.0x!)
        var jx_px_cur = Float32(0.0)
        var jy_px_cur = Float32(0.0)
        var jx_clip = Float32(0.0)
        var jy_clip = Float32(0.0)
        var mip_bias = Float32(0.0)
        if self.tsr_history_weight > Float32(0.01) and not self.furnace_white_dome:
            var hj = halton_2_3_jitter_px(self.frame_index)
            jx_px_cur = hj[0]
            jy_px_cur = hj[1]
            jx_clip = (jx_px_cur * Float32(2.0)) / Float32(self.render_w)
            jy_clip = (jy_px_cur * Float32(2.0)) / Float32(self.render_h)
            mip_bias = Float32(-0.42)

        var g_addr = self.buf_globals.cpu_address
        for k in range(16):
            put_float(g_addr + k * 4, vp_m[k])
            put_float(g_addr + (16 + k) * 4, self.prev_view_proj[k])
            put_float(g_addr + (56 + k) * 4, light_vp[k])
        put_float(g_addr + 32 * 4, self.cam_pos.x)
        put_float(g_addr + 33 * 4, self.cam_pos.y)
        put_float(g_addr + 34 * 4, self.cam_pos.z)
        put_float(g_addr + 35 * 4, self.near_z)
        put_float(g_addr + 36 * 4, self.cam_fwd.x)
        put_float(g_addr + 37 * 4, self.cam_fwd.y)
        put_float(g_addr + 38 * 4, self.cam_fwd.z)
        put_float(g_addr + 39 * 4, tan_x)
        put_float(g_addr + 40 * 4, self.cam_right.x)
        put_float(g_addr + 41 * 4, self.cam_right.y)
        put_float(g_addr + 42 * 4, self.cam_right.z)
        put_float(g_addr + 43 * 4, tan_y)
        put_float(g_addr + 44 * 4, self.cam_up.x)
        put_float(g_addr + 45 * 4, self.cam_up.y)
        put_float(g_addr + 46 * 4, self.cam_up.z)
        put_float(g_addr + 47 * 4, self.exposure)
        put_float(g_addr + 48 * 4, self.sun_dir.x)
        put_float(g_addr + 49 * 4, self.sun_dir.y)
        put_float(g_addr + 50 * 4, self.sun_dir.z)
        put_float(g_addr + 51 * 4, self.sun_angular_radius)
        var shadow_cone = self.sun_shadow_softness if self.sun_shadow_softness >= Float32(0.0) else self.sun_angular_radius
        put_float(g_addr + AAA_GLOBALS_SHADOW_CONE_WORD * 4, tan(min(Float32(0.2), shadow_cone)))
        var n_glass = min(AAA_MAX_GLASS_MATERIALS, len(self.glass_materials))
        put32(g_addr + AAA_GLOBALS_GLASS_COUNT_WORD * 4, UInt32(n_glass))
        var glass_tr_sum = Float32(0.0)
        for g_i in range(n_glass):
            var g_words = List[UInt32](capacity=AAA_GLASS_ENTRY_WORDS)
            self.glass_materials[g_i].append_words(g_words)
            for g_w in range(AAA_GLASS_ENTRY_WORDS):
                put32(g_addr + (AAA_GLOBALS_GLASS_TABLE_WORD + g_i * AAA_GLASS_ENTRY_WORDS + g_w) * 4, g_words[g_w])
            glass_tr_sum += self.glass_materials[g_i].shadow_transmittance()
        put_float(g_addr + AAA_GLOBALS_GLASS_SHADOW_WORD * 4, glass_tr_sum / Float32(max(1, n_glass)))
        put_float(g_addr + 52 * 4, Float32(0.0) if self.furnace_white_dome else self.sun_radiance.x)
        put_float(g_addr + 53 * 4, Float32(0.0) if self.furnace_white_dome else self.sun_radiance.y)
        put_float(g_addr + 54 * 4, Float32(0.0) if self.furnace_white_dome else self.sun_radiance.z)
        put_float(g_addr + 55 * 4, self.ibl_intensity)
        for k in range(9):
            put_float(g_addr + (72 + k * 3 + 0) * 4, self.sh9_irradiance[k].x)
            put_float(g_addr + (72 + k * 3 + 1) * 4, self.sh9_irradiance[k].y)
            put_float(g_addr + (72 + k * 3 + 2) * 4, self.sh9_irradiance[k].z)

        var n_cl_lights = len(self.frame_cluster_lights)
        if n_cl_lights > 0:
            var cl_addr = self.buf_clusters.cpu_address
            var far_z_cl = max(self.near_z + Float32(1.0), self.cluster_far_z)
            put32(cl_addr + 0 * 4, UInt32(n_cl_lights))
            put_float(cl_addr + 1 * 4, self.near_z)
            put_float(cl_addr + 2 * 4, far_z_cl)
            put_float(cl_addr + 3 * 4, log2(far_z_cl / max(Float32(1e-4), self.near_z)))
            put_float(cl_addr + 4 * 4, tan_x)
            put_float(cl_addr + 5 * 4, tan_y)
            var cl_words = List[UInt32](capacity=n_cl_lights * 16)
            for cl_i in range(n_cl_lights):
                self.frame_cluster_lights[cl_i].append_words(cl_words)
            _ = external_call[
                "memcpy", Int
            ](cl_addr + AAA_CLUSTER_LIGHTS_BASE * 4, Int(cl_words.unsafe_ptr()), len(cl_words) * 4)
            _ = len(cl_words)
        else:
            put32(self.buf_clusters.cpu_address, 0)

        var r_flags: UInt32 = 0
        if self.enable_vsm and not self.furnace_white_dome:
            r_flags |= AAA_FLAG_VSM
        if self.vk.has_ray_query and not self.furnace_white_dome and n_inst > 0:
            if self.enable_rt_shadows:
                r_flags |= AAA_FLAG_RT_SHADOWS
                if self.enable_restir_di:
                    r_flags |= AAA_FLAG_RESTIR_DI
            if self.enable_rt_reflections:
                r_flags |= AAA_FLAG_RT_REFLECTIONS
            if self.enable_rt_gi:
                r_flags |= AAA_FLAG_RT_SURFACE_GI
        if self.furnace_white_dome:
            r_flags |= AAA_FLAG_FURNACE_WHITE_DOME
        if frame_has_glass:
            r_flags |= AAA_FLAG_GLASS
        if n_cl_lights > 0 and not self.furnace_white_dome:
            r_flags |= AAA_FLAG_CLUSTERED_LIGHTS
        if self.enable_aerial_perspective and not self.furnace_white_dome and not self.sota_atmo.replaces_legacy_ap():  # sota:atmo hook
            r_flags |= AAA_FLAG_AERIAL_PERSPECTIVE
        # --- sota:E hook ---
        if self.sota_e_character_shading:
            r_flags |= AAA_FLAG_SOTA_E_CHARACTERS
        self.sota_e_hair.write_globals(g_addr, self.sun_dir)
        # --- end sota:E hook ---
        # --- sota:mat hook (layered material model; LUTs baked + uploaded on first enabled frame) ---
        if self.sota_mat.enabled:
            self.sota_mat.prepare(self.buf_materials.cpu_address)
            if self.sota_mat.lut_uploaded:
                r_flags |= AAA_FLAG_SOTA_MAT
        # --- end sota:mat hook ---
        put32(g_addr + 99 * 4, r_flags)
        put32(g_addr + 100 * 4, UInt32(self.frame_index))
        put_float(g_addr + 101 * 4, jx_clip)
        put_float(g_addr + 102 * 4, jy_clip)
        put_float(g_addr + 103 * 4, Float32(self.env_mip_levels - 1))
        put_float(g_addr + 104 * 4, self.aerial_sigma_ext.x)
        put_float(g_addr + 105 * 4, self.aerial_sigma_ext.y)
        put_float(g_addr + 106 * 4, self.aerial_sigma_ext.z)
        put_float(g_addr + 107 * 4, self.aerial_mie_g)
        put_float(g_addr + 108 * 4, self.aerial_r_term.x)
        put_float(g_addr + 109 * 4, self.aerial_r_term.y)
        put_float(g_addr + 110 * 4, self.aerial_r_term.z)
        put_float(g_addr + 111 * 4, Float32(1.0))
        put_float(g_addr + 112 * 4, self.aerial_m_term.x)
        put_float(g_addr + 113 * 4, self.aerial_m_term.y)
        put_float(g_addr + 114 * 4, self.aerial_m_term.z)
        var eff_vig = Float32(0.0) if self.furnace_white_dome else self.camera_lens.vignette_strength
        put_float(g_addr + 115 * 4, eff_vig)
        put_float(g_addr + 116 * 4, self.bloom_intensity)
        put_float(g_addr + 117 * 4, self.camera_lens.focus_distance_m)
        put_float(g_addr + 118 * 4, self.camera_lens.compute_coc_scale_px(self.output_h))
        put32(g_addr + 119 * 4, UInt32(1) if (self.enable_auto_exposure and not self.furnace_white_dome) else UInt32(0))

        # Extended v2 AAA Globals (`b_globals[120..155]`): Froxel Volumetrics, Weather, Water, TSR v3 & Grading
        var prev_c = self.prev_cam_pos if self.has_prev_cam else self.cam_pos
        var cam_vel = (self.cam_pos - prev_c) * Float32(60.0)
        put_float(g_addr + 120 * 4, prev_c.x)
        put_float(g_addr + 121 * 4, prev_c.y)
        put_float(g_addr + 122 * 4, prev_c.z)
        put_float(g_addr + 123 * 4, self.scene_time_sec)
        put_float(g_addr + 124 * 4, Float32(0.0) if self.furnace_white_dome else self.weather_wetness)
        put_float(g_addr + 125 * 4, Float32(0.0) if self.furnace_white_dome else self.rain_intensity)
        put_float(g_addr + 126 * 4, Float32(0.0) if self.furnace_white_dome else self.windshield_rain)
        put_float(g_addr + 127 * 4, mip_bias)
        put_float(g_addr + 128 * 4, self.wind_velocity.x)
        put_float(g_addr + 129 * 4, self.wind_velocity.y)
        put_float(g_addr + 130 * 4, self.wind_velocity.z)
        put_float(g_addr + 131 * 4, max(self.near_z + Float32(2.0), self.froxel_far_z))
        put_float(
            g_addr + 132 * 4,
            Float32(0.0) if (self.furnace_white_dome or not self.enable_froxel_fog) else max(Float32(0.0), self.fog_density),
        )
        put_float(g_addr + 133 * 4, max(Float32(0.001), self.fog_height_falloff))
        put_float(g_addr + 134 * 4, max(Float32(-0.90), min(Float32(0.95), self.fog_anisotropy_g)))
        put_float(g_addr + 135 * 4, self.fog_base_height_y)
        put_float(g_addr + 136 * 4, self.fog_albedo.x)
        put_float(g_addr + 137 * 4, self.fog_albedo.y)
        put_float(g_addr + 138 * 4, self.fog_albedo.z)
        put_float(g_addr + 139 * 4, Float32(0.0) if self.furnace_white_dome else self.glare_streak_strength)
        put_float(g_addr + 140 * 4, self.water_level_y)
        put_float(g_addr + 141 * 4, self.water_absorption_rgb.x)
        put_float(g_addr + 142 * 4, self.water_absorption_rgb.y)
        put_float(g_addr + 143 * 4, self.water_absorption_rgb.z)
        put_float(g_addr + 144 * 4, self.water_bounds_min_xz[0])
        put_float(g_addr + 145 * 4, self.water_bounds_min_xz[1])
        put_float(g_addr + 146 * 4, self.water_bounds_max_xz[0])
        put_float(g_addr + 147 * 4, self.water_bounds_max_xz[1])
        put32(g_addr + AAA_GLOBALS_WATER_SHAPE_WORD * 4, UInt32(1) if self.water_elliptical else UInt32(0))
        put_float(g_addr + (AAA_GLOBALS_WATER_SHAPE_WORD + 1) * 4, self.water_wave_scale)
        put_float(g_addr + (AAA_GLOBALS_WATER_SHAPE_WORD + 2) * 4, self.water_foam_band)
        put_float(g_addr + (AAA_GLOBALS_WATER_SHAPE_WORD + 3) * 4, self.water_foam)
        put_float(g_addr + AAA_GLOBALS_WIND_WORD * 4, Float32(get_time() - self.wind_t0))   # wall clock: sway speed does not follow the frame rate
        put_float(g_addr + (AAA_GLOBALS_WIND_WORD + 1) * 4, self.wind_dir_x)
        put_float(g_addr + (AAA_GLOBALS_WIND_WORD + 2) * 4, self.wind_dir_z)
        put_float(g_addr + (AAA_GLOBALS_WIND_WORD + 3) * 4, self.wind_sway)
        put_float(g_addr + (AAA_GLOBALS_WIND_WORD + 4) * 4, self.wind_hz)
        put_float(g_addr + (AAA_GLOBALS_WIND_WORD + 5) * 4, self.wind_gust)
        put_float(g_addr + 148 * 4, self.min_auto_exposure)
        put_float(g_addr + 149 * 4, self.max_auto_exposure)
        put_float(g_addr + 150 * 4, self.motion_blur_shutter)
        put_float(g_addr + 151 * 4, jx_px_cur)
        put_float(g_addr + 152 * 4, jy_px_cur)
        put_float(g_addr + 153 * 4, cam_vel.x)
        put_float(g_addr + 154 * 4, cam_vel.y)
        put_float(g_addr + 155 * 4, cam_vel.z)

        # Extended v3 Flagship Sky Globals (`b_globals[156..166]`): Nubis 3D Clouds + Blackbody Starfield/Moon
        put_float(g_addr + 156 * 4, Float32(0.0) if self.furnace_white_dome else max(Float32(0.0), min(Float32(1.0), self.cloud_coverage)))
        put_float(g_addr + 157 * 4, self.cloud_wind_offset.x)
        put_float(g_addr + 158 * 4, self.cloud_wind_offset.y)
        put_float(g_addr + 159 * 4, self.cloud_wind_offset.z)
        put_float(g_addr + 160 * 4, Float32(0.0) if self.furnace_white_dome else max(Float32(0.0), self.star_intensity))
        put_float(g_addr + 161 * 4, self.sky_zenith_clear.x)
        put_float(g_addr + 162 * 4, self.sky_zenith_clear.y)
        put_float(g_addr + 163 * 4, self.sky_zenith_clear.z)
        put_float(g_addr + 164 * 4, self.sky_horizon_clear.x)
        put_float(g_addr + 165 * 4, self.sky_horizon_clear.y)
        put_float(g_addr + 166 * 4, self.sky_horizon_clear.z)
        # --- sota:atmo hook --- per-frame atmosphere constants (b_globals[3840 .. 3904))
        if self.sota_atmo.active() and not self.furnace_white_dome:
            self.sota_atmo.write_globals(
                g_addr, self.cam_pos.y, self.cloud_coverage > Float32(0.01) or self.star_intensity > Float32(0.01),
                self.enable_froxel_fog, self.fog_density, self.froxel_far_z, self.fog_height_falloff,
                self.fog_base_height_y, self.fog_albedo, self.fog_anisotropy_g,
            )
        # --- end sota:atmo hook ---

        # Extended v5 4-Cascade VSM Globals (`b_globals[167..215]` — `0..166` untouched!):
        put32(g_addr + VSM_GLOBALS_CASCADE_FLAG_WORD * 4, UInt32(1) if use_vsm_casc_frame else UInt32(0))
        # --- sota:shadow hook --- toroidal page addressing flag + per-cascade slot offsets
        var sh_tor = use_vsm_casc_frame and self.sota_shadow.toroidal_active
        put32(g_addr + SOTA_SHADOW_GLOBALS_TOROIDAL_WORD * 4, UInt32(1) if sh_tor else UInt32(0))
        for c_t in range(VSM_CASCADE_COUNT):
            var o_t = self.sota_shadow.offsets[c_t] if sh_tor else (0, 0)
            put32(g_addr + (SOTA_SHADOW_GLOBALS_TOROIDAL_OFFSETS_WORD + c_t) * 4, UInt32(o_t[0] | (o_t[1] << 8)))
        put_float(g_addr + SOTA_SHADOW_GLOBALS_CONTACT_WORD * 4, self.sota_shadow.contact_length_m)
        put_float(g_addr + (SOTA_SHADOW_GLOBALS_CONTACT_WORD + 1) * 4, self.sota_shadow.contact_thickness_m)
        put32(g_addr + (SOTA_SHADOW_GLOBALS_CONTACT_WORD + 2) * 4, UInt32(1) if self.sota_shadow.contact else UInt32(0))
        # --- end sota:shadow hook ---
        if use_vsm_casc_frame:
            for k in range(16):
                put_float(g_addr + (VSM_GLOBALS_CASCADE_1_WORD + k) * 4, self.vsm_cascade_matrices[1][k])
                put_float(g_addr + (VSM_GLOBALS_CASCADE_2_WORD + k) * 4, self.vsm_cascade_matrices[2][k])
                put_float(g_addr + (VSM_GLOBALS_CASCADE_3_WORD + k) * 4, self.vsm_cascade_matrices[3][k])

        # Extended v5.1 Anime NPR Cel, Stylized 3D Mesh VFX, Chromatic Shockwaves & Manga Impact Frame Globals
        var n_cel = 0 if self.furnace_white_dome else min(AAA_MAX_CEL_MATERIALS, len(self.cel_materials))
        put32(g_addr + AAA_GLOBALS_CEL_COUNT_WORD * 4, UInt32(n_cel))
        for c_i in range(n_cel):
            var c_words = List[UInt32](capacity=AAA_CEL_ENTRY_WORDS)
            self.cel_materials[c_i].append_words(c_words)
            for c_w in range(AAA_CEL_ENTRY_WORDS):
                put32(g_addr + (AAA_GLOBALS_CEL_TABLE_WORD + c_i * AAA_CEL_ENTRY_WORDS + c_w) * 4, c_words[c_w])

        var n_vfx = 0 if self.furnace_white_dome else min(AAA_MAX_VFX_MATERIALS, len(self.vfx_materials))
        put32(g_addr + AAA_GLOBALS_VFX_COUNT_WORD * 4, UInt32(n_vfx))
        for v_i in range(n_vfx):
            var v_words = List[UInt32](capacity=AAA_VFX_ENTRY_WORDS)
            self.vfx_materials[v_i].append_words(v_words)
            for v_w in range(AAA_VFX_ENTRY_WORDS):
                put32(g_addr + (AAA_GLOBALS_VFX_TABLE_WORD + v_i * AAA_VFX_ENTRY_WORDS + v_w) * 4, v_words[v_w])

        var n_sw = 0 if self.furnace_white_dome else min(AAA_MAX_SHOCKWAVES, len(self.frame_shockwaves))
        put32(g_addr + AAA_GLOBALS_SHOCKWAVE_COUNT_WORD * 4, UInt32(n_sw))
        for s_i in range(n_sw):
            var s_words = List[UInt32](capacity=AAA_SHOCKWAVE_WORDS)
            self.frame_shockwaves[s_i].append_words(s_words)
            for s_w in range(AAA_SHOCKWAVE_WORDS):
                put32(g_addr + (AAA_GLOBALS_SHOCKWAVE_TABLE_WORD + s_i * AAA_SHOCKWAVE_WORDS + s_w) * 4, s_words[s_w])

        var imp_blend = Float32(0.0) if self.furnace_white_dome else self.impact_frame.blend
        put_float(g_addr + AAA_GLOBALS_IMPACT_PAPER_GAIN_WORD * 4, self.impact_frame.paper_gain)
        put_float(g_addr + (AAA_GLOBALS_IMPACT_FRAME_WORD + 0) * 4, imp_blend)
        put_float(g_addr + (AAA_GLOBALS_IMPACT_FRAME_WORD + 1) * 4, self.impact_frame.center_u)
        put_float(g_addr + (AAA_GLOBALS_IMPACT_FRAME_WORD + 2) * 4, self.impact_frame.center_v)
        put_float(g_addr + (AAA_GLOBALS_IMPACT_FRAME_WORD + 3) * 4, self.impact_frame.luma_threshold)
        put_float(g_addr + (AAA_GLOBALS_IMPACT_FRAME_WORD + 4) * 4, self.impact_frame.streak_strength)
        var signed_acc = -self.impact_frame.accent_width if self.impact_frame.invert_polarity else self.impact_frame.accent_width
        put_float(g_addr + (AAA_GLOBALS_IMPACT_FRAME_WORD + 5) * 4, signed_acc)
        put_float(g_addr + (AAA_GLOBALS_IMPACT_FRAME_WORD + 6) * 4, self.impact_frame.ink_color.x)
        put_float(g_addr + (AAA_GLOBALS_IMPACT_FRAME_WORD + 7) * 4, self.impact_frame.ink_color.y)
        put_float(g_addr + (AAA_GLOBALS_IMPACT_FRAME_WORD + 8) * 4, self.impact_frame.ink_color.z)
        put_float(g_addr + (AAA_GLOBALS_IMPACT_FRAME_WORD + 9) * 4, self.impact_frame.accent_color.x)
        put_float(g_addr + (AAA_GLOBALS_IMPACT_FRAME_WORD + 10) * 4, self.impact_frame.accent_color.y)
        put_float(g_addr + (AAA_GLOBALS_IMPACT_FRAME_WORD + 11) * 4, self.impact_frame.accent_color.z)

        # Pack Bucket 2 alpha-masked VisBuffer draws & hardware `CULL_MODE_FRONT` clip-space outline draws
        # --- sota:E hook --- GPU-driven: the counts become max draw counts for vkCmdDrawIndexedIndirectCount
        var sota_e_gpu_lists = self.sota_e_gpu.use(self.sota_geom) and n_inst > 0
        var am_vis_draws: Int
        var outline_vis_draws: Int
        if sota_e_gpu_lists:
            self.sota_e_gpu.reset_counters(self.buf_alphamask_indirect, self.buf_outline_indirect)
            am_vis_draws = self.sota_route_draws if self.sota_e_gpu.frame_alpha_instances > 0 else 0  # sota:perf hook
            var outl_on = not self.furnace_white_dome and self.sota_e_gpu.frame_outline_instances > 0
            outline_vis_draws = self.sota_route_draws if outl_on else 0  # sota:perf hook
        else:
            am_vis_draws = self._pack_alphamask_visbuffer_commands(n_inst, tan_x, tan_y)
            outline_vis_draws = 0 if self.furnace_white_dome else self._pack_outline_visbuffer_commands(n_inst, tan_x, tan_y)
        # --- end sota:E hook ---

        # Reset indirect draw counters and candidate queue = 0
        put32(self.buf_indirect.cpu_address, 0)
        if self.buf_indirect_phase2.cpu_address != 0:
            put32(self.buf_indirect_phase2.cpu_address, 0)
        if self.buf_occluded_candidates.cpu_address != 0:
            put32(self.buf_occluded_candidates.cpu_address, 0)

        # 3. Ensure TLAS Capacity Before Recording (Zero per-frame `vkAllocateMemory` / `vkDestroyAccelerationStructureKHR`!)
        var use_rt_frame = self.vk.has_ray_query and n_inst > 0 and (
            self.enable_rt_shadows
            or self.enable_rt_reflections
            or self.enable_rt_gi
            or (self.enable_water_plane and not self.furnace_white_dome)
            or frame_has_glass
        )
        # --- sota:gi hook --- (effective area-B mode + control block for this frame)
        var sota_mode = SOTA_GI_MODE_LEGACY
        if (
            use_rt_frame
            and self.enable_rt_gi
            and not self.furnace_white_dome
            and self.sota_gi_region_ready
            and self.sota_gi.pipe_wrc != 0
            and self.sota_gi.pipe_resolve != 0
        ):
            sota_mode = self.sota_gi.mode
        var sota_restir_di = use_rt_frame and self.enable_restir_di and self.enable_rt_shadows and n_cl_lights > 0
        # --- sota:perf hook --- a light change restarts photo-mode accumulation and lets the WRC re-converge fast
        var sota_keep_hist = self.sota_gi.wrc_max_history
        if self.sota_vsm.light_changed_frame:
            self.sota_gi.reset_accumulation()
        self.sota_gi.wrc_max_history = self.sota_vsm.gi_history_cap(sota_keep_hist)
        self.sota_gi.write_ctrl(
            self.buf_surfcache.cpu_address, sota_mode, self.frame_index, self.cam_pos, self.cam_fwd,
            self.render_w, self.render_h, max(self.output_w * self.output_h, self.render_w * self.render_h), n_inst,
            restir_di=sota_restir_di, region_ready=self.sota_gi_region_ready,
        )
        self.sota_gi.wrc_max_history = sota_keep_hist
        # --- end sota:gi hook ---
        var inst_da: UInt64 = 0
        if use_rt_frame:
            inst_da = self.vk.buffer_device_address(self.buf_as_instances)
            # Build sizes depend only on the instance count, so skip `vkGetAccelerationStructureBuildSizesKHR`
            # (16k instances) once this slot's TLAS is known to be large enough.
            var tlas_cap = max(self.max_instances, n_inst)
            while len(self.ring_tlas_cap) < len(self.ring_tlas):
                self.ring_tlas_cap.append(0)
            var tlas_known_ok = (
                cur_slot < len(self.ring_tlas_cap)
                and self.frame_tlas.handle != 0
                and self.buf_as_scratch.size > 0
                and self.ring_tlas_cap[cur_slot] >= tlas_cap
            )
            if not tlas_known_ok:
                if ensure_tlas_capacity_mixed_opaque(self.vk, self.frame_tlas, inst_da, tlas_cap, self.buf_as_scratch):
                    self.vk.write_aaa_tlas(self.desc_set, BIND_AAA_TLAS, self.frame_tlas.handle)
                    if cur_slot < len(self.ring_tlas):
                        self.ring_tlas[cur_slot] = self.frame_tlas.copy()
                if cur_slot < len(self.ring_tlas_cap) and self.frame_tlas.handle != 0:
                    self.ring_tlas_cap[cur_slot] = max(self.ring_tlas_cap[cur_slot], tlas_cap)

        # Single-Command-Buffer Mode: asynchronous single-pass execution without CPU stalls in steady-state,
        # with automatic Frame-0 / camera-teleport warmup so multi-cascade VSM shadows are crisp on the very first frame!
        var vsm_active = (r_flags & AAA_FLAG_VSM) != 0 and n_inst > 0
        var cam_teleported = self.has_prev_cam and (self.cam_pos - self.prev_cam_pos).length() > Float32(4.0)
        var single_cmd_buf = True

        var t_vsm0 = get_time()
        self.last_cpu_pack_ms = (t_vsm0 - t_pack0) * 1000.0
        var vsm_allocs = List[Tuple[Int, Int, Int, Int]]()
        var vsm_page_offsets = List[Int]()
        var vsm_page_counts = List[Int]()
        var vsm_page_am_offsets = List[Int]()
        var vsm_page_am_counts = List[Int]()
        # --- sota:shadow hook --- GPU page path: no request readback, no CPU allocation / packing / sync submit
        var sh_gpu = self._sota_shadow_gpu_prepare(cur_slot, n_inst, vsm_active and use_vsm_casc_frame)
        if vsm_active and not sh_gpu:
            var origin_shifted = self.sota_shadow.forced_sync_frame  # --- sota:shadow hook ---
            if use_vsm_casc_frame and not self.sota_shadow.toroidal_active:  # sota:shadow hook (toroidal: done above)
                var next_origins = compute_vsm_cascade_page_origins(self.sun_dir, self.cam_pos, base_vsm_ext)
                if len(self.vsm_cascade_page_origins) >= VSM_CASCADE_COUNT:
                    for c_i in range(VSM_CASCADE_COUNT):
                        if (
                            next_origins[c_i][0] != self.vsm_cascade_page_origins[c_i][0]
                            or next_origins[c_i][1] != self.vsm_cascade_page_origins[c_i][1]
                        ):
                            self.vsm.invalidate_cascade(c_i)
                            origin_shifted = True
                self.vsm_cascade_page_origins = next_origins^

            var tbl = List[UInt32](length=VSM_TABLE_WORDS, fill=0)
            _ = external_call["memcpy", Int](Int(tbl.unsafe_ptr()), self.buf_vsm_table.cpu_address, VSM_TABLE_WORDS * 4)
            var has_gpu_reqs = False
            for r_i in range(16384, VSM_TABLE_WORDS):
                if tbl[r_i] != 0:
                    has_gpu_reqs = True
                    break
            single_cmd_buf = has_gpu_reqs and not cam_teleported and not origin_shifted
            if single_cmd_buf:
                var t_vsm_req0 = get_time()
                var pages = self.vsm.process_requests(tbl)
                var t_vsm_req1 = get_time()
                _ = external_call["memcpy", Int](self.buf_vsm_table.cpu_address, Int(tbl.unsafe_ptr()), VSM_TABLE_WORDS * 4)
                self.last_cpu_vsm_requests_ms = (t_vsm_req1 - t_vsm_req0) * 1000.0
                self.last_cpu_vsm_table_ms = (get_time() - t_vsm0) * 1000.0 - self.last_cpu_vsm_requests_ms
                for pg in pages:
                    vsm_allocs.append((pg.vx, pg.vy, pg.px, pg.py))
                if len(vsm_allocs) > 0:
                    var packed_vsm = self._pack_vsm_indirect_commands(n_inst, light_vp, vsm_allocs)
                    vsm_page_offsets = packed_vsm[1].copy()
                    vsm_page_counts = packed_vsm[2].copy()
                    vsm_page_am_offsets = packed_vsm[3].copy()
                    vsm_page_am_counts = packed_vsm[4].copy()
            _ = len(tbl)

        # ====================================================================
        # PASS GROUP A: Deferred Skinning + Inline TLAS + GPU Meshlet Cull + VisBuffer + VSM
        # ====================================================================
        var use_prof = self.enable_gpu_profiler and self.vk.timestamps and self.vk.query_pool != 0
        var prof_pairs = List[Tuple[String, Int]]()
        var q_base = cur_slot * AAA_QUERY_SLOT_STRIDE
        var q_limit = q_base + 64
        var next_q = q_base
        var frame_q = (q_base + AAA_QUERY_FRAME_PAIR) if (use_prof and single_cmd_buf) else -1
        var t_record0 = get_time()
        self.last_cpu_vsm_ms = (t_record0 - t_vsm0) * 1000.0

        if not self.vk.begin_commands(cur_slot):
            return self._sota_frame_fail("begin_commands failed (device lost or fence timeout)")  # sota:geometry hook
        if use_prof:
            self.vk.reset_queries(cur_slot, q_base, AAA_QUERY_SLOT_STRIDE)
        self.sota_vt.record_uploads(self.vk, cur_slot)  # sota:geometry hook
        if defer_this:
            # The CPU no longer waits between frames, so serialise this frame's GPU work behind the previous one
            # (single-instance buffers/images: history, post scratch, VSM atlas, surface cache, TLAS scratch...).
            self.vk.memory_barrier2(
                cur_slot, _AAA_STAGE_ALL_COMMANDS, _AAA_ACCESS_MEMORY_READ_WRITE, _AAA_STAGE_ALL_COMMANDS, _AAA_ACCESS_MEMORY_READ_WRITE
            )
        if frame_q >= 0:
            self.vk.write_timestamp(cur_slot, frame_q, PIPELINE_TOP)

        # Execute any deferred Dual-Quaternion skinning + BLAS refits inline (`0` extra submit!)
        if len(self.pending_skin_updates) > 0:
            for sk_i in range(len(self.pending_skin_updates)):
                var sk_e = self.pending_skin_updates[sk_i].copy()
                self._record_skin_dispatch_and_blas_refit(cur_slot, sk_e)
            self.pending_skin_updates.clear()

        # Inline TLAS Rebuild (`0` host allocations, `0` extra submit!)
        if use_rt_frame and self.frame_tlas.handle != 0:
            record_build_tlas_mixed_opaque(self.vk, cur_slot, self.frame_tlas, inst_da, n_inst, self.buf_as_scratch)
            self.vk.memory_barrier2(
                cur_slot,
                PIPELINE_AS_BUILD_KHR,
                Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR),
                PIPELINE_AS_BUILD_KHR | PIPELINE_COMPUTE,
                Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
            )

        # --- sota:E hook (hair deep opacity map, read by the deferred pass) ---
        self.sota_e_hair.record(self.vk, cur_slot, self.pipe_layout, self.desc_set)
        # --- end sota:E hook ---
        if n_inst > 0:
            var q_mc = next_q
            if use_prof:
                self.vk.write_timestamp(cur_slot, q_mc, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("MeshletCull"), q_mc))
            var push_cull = List[UInt32](length=16, fill=0)
            push_cull[0] = UInt32(n_inst)
            push_cull[1] = UInt32(self.max_draws)
            push_cull[2] = UInt32(1) if self.enable_cone_cull else UInt32(0)
            push_cull[3] = UInt32(1) if (self.enable_hiz_cull and self.frame_index > 0 and not cam_teleported) else UInt32(0)
            push_cull[4] = UInt32(self.render_w)
            push_cull[5] = UInt32(self.render_h)
            push_cull[6] = bitcast[DType.uint32](self.max_lod_error_px)
            push_cull[7] = 0  # Phase 1: cull_phase = 0
            # --- sota:geometry hook ---
            if self.sota_geom.use_cull():
                self.sota_geom.record_cull(
                    self.vk, cur_slot, 0, n_inst, self.max_draws, max_ml_per_inst, self.enable_cone_cull,
                    push_cull[3] != 0, self.render_w, self.render_h, self.max_lod_error_px, use_rt_frame,
                )
                # --- sota:E hook ---
                if sota_e_gpu_lists:
                    self.sota_e_gpu.record(
                        self.vk, self.sota_geom, cur_slot, n_inst, max_ml_per_inst, self.render_w, self.render_h,
                        self.max_lod_error_px, am_vis_draws > 0, outline_vis_draws > 0 and self.pipe_npr_outline_gfx != 0,
                        max_draws=self.sota_route_draws,  # --- sota:perf hook ---
                    )
                # --- end sota:E hook ---
            else:
                if not self.sota_geom.legacy_hiz_occlusion:
                    push_cull[3] = 0  # v5 5-tap test is not conservative; nothing is queued, so phase 2 is a no-op
                self.vk.dispatch_aaa(
                    cur_slot,
                    self.pipe_meshlet_cull,
                    self.pipe_layout,
                    self.desc_sets_cull_p1[cur_slot],
                    push_cull,
                    (n_inst + 7) // 8,
                    (max_ml_per_inst + 7) // 8,
                )
            # --- end sota:geometry hook ---
            if use_prof:
                self.vk.write_timestamp(cur_slot, q_mc + 1, PIPELINE_BOTTOM)
            self.render_graph.emit_pass_barriers(self.vk, cur_slot, 1)
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_DRAW_INDIRECT | PIPELINE_TRANSFER, Int(ACCESS_INDIRECT_READ | ACCESS_TRANSFER_READ)
            )
            self.vk.copy_buffer(cur_slot, self.buf_indirect, 0, rb_buf, 0, 4)

        # Hardware `VK_KHR_dynamic_rendering` 64-Bit Visibility Buffer + Reverse-Z Depth (Phase 1 Draw)
        var q_vb = next_q
        if use_prof:
            self.vk.write_timestamp(cur_slot, q_vb, PIPELINE_BOTTOM)
            next_q += 2
            prof_pairs.append((String("VisBufferHwRaster"), q_vb))
        self.vk.transition_image(
            cur_slot, self.vis_img, LAYOUT_COLOR_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE
        )
        self.vk.transition_image(
            cur_slot, self.depth_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE
        )
        self.vk.begin_rendering(
            cur_slot, 0, 0, self.render_w, self.render_h, True, self.vis_img.view, True, 0, 0, True, self.depth_img.view, True, 0.0
        )
        self.vk.set_viewport_and_scissor(cur_slot, 0, 0, self.render_w, self.render_h, flip_y=True)
        if n_inst > 0:
            var push_vb = List[UInt32](length=16, fill=0)
            push_vb[0] = 0
            push_vb[1] = bitcast[DType.uint32](jx_clip)
            push_vb[2] = bitcast[DType.uint32](jy_clip)
            self.vk.bind_graphics_pipeline(cur_slot, self.pipe_visbuffer_gfx, self.pipe_layout, self.desc_set, push_vb)
            self.vk.bind_index_buffer(cur_slot, self.buf_indices, 0, INDEX_TYPE_UINT32)
            self.vk.draw_indexed_indirect_count(
                cur_slot,
                self.buf_indirect,
                16,
                self.buf_indirect,
                0,
                min(self.max_draws, n_inst * max_ml_per_inst),
                stride=20,
            )
        self.vk.end_rendering(cur_slot)

        # Phase 2: Test occluded candidates against fresh current-frame depth and draw disocclusions
        if n_inst > 0 and self.enable_hiz_cull and self.frame_index > 0:
            # Construct / downsample current frame Hi-Z depth pyramid with barrier
            self.vk.transition_image(
                cur_slot, self.depth_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_LATE_FRAGMENT_TESTS, Int(ACCESS_DEPTH_ATTACHMENT_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ)
            )

            # Phase 2 cull over candidates in self.buf_occluded_candidates
            var push_cull_p2 = List[UInt32](length=16, fill=0)
            push_cull_p2[0] = UInt32(n_inst)
            push_cull_p2[1] = UInt32(self.max_draws)
            push_cull_p2[2] = UInt32(1) if self.enable_cone_cull else UInt32(0)
            push_cull_p2[3] = UInt32(1)
            push_cull_p2[4] = UInt32(self.render_w)
            push_cull_p2[5] = UInt32(self.render_h)
            push_cull_p2[6] = bitcast[DType.uint32](self.max_lod_error_px)
            push_cull_p2[7] = 1  # Phase 2: cull_phase = 1
            # --- sota:geometry hook ---
            if self.sota_geom.use_cull():
                # conservative Hi-Z pyramid from the phase-1 depth, then candidates vs current matrices
                self.sota_geom.record_hiz_build(self.vk, cur_slot)
                self.sota_geom.record_cull(
                    self.vk, cur_slot, 1, n_inst, self.max_draws, max_ml_per_inst, self.enable_cone_cull,
                    True, self.render_w, self.render_h, self.max_lod_error_px,
                )
            else:
                self.vk.dispatch_aaa(
                    cur_slot,
                    self.pipe_meshlet_cull,
                    self.pipe_layout,
                    self.desc_sets_cull_p2[cur_slot],
                    push_cull_p2,
                    (self.max_draws + 63) // 64,
                    1,
                )
            # --- end sota:geometry hook ---
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_DRAW_INDIRECT, Int(ACCESS_INDIRECT_READ)
            )

            # Phase 2 Disocclusion Draw (LOAD existing VisBuffer & Depth, append newly visible meshlets)
            self.vk.transition_image(
                cur_slot, self.vis_img, LAYOUT_COLOR_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE
            )
            self.vk.transition_image(
                cur_slot, self.depth_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE
            )
            self.vk.begin_rendering(
                cur_slot, 0, 0, self.render_w, self.render_h, True, self.vis_img.view, False, 0, 0, True, self.depth_img.view, False, 0.0
            )
            self.vk.set_viewport_and_scissor(cur_slot, 0, 0, self.render_w, self.render_h, flip_y=True)
            var push_vb2 = List[UInt32](length=16, fill=0)
            push_vb2[0] = 0
            push_vb2[1] = bitcast[DType.uint32](jx_clip)
            push_vb2[2] = bitcast[DType.uint32](jy_clip)
            self.vk.bind_graphics_pipeline(cur_slot, self.pipe_visbuffer_gfx, self.pipe_layout, self.desc_set, push_vb2)
            self.vk.bind_index_buffer(cur_slot, self.buf_indices, 0, INDEX_TYPE_UINT32)
            self.vk.draw_indexed_indirect_count(
                cur_slot,
                self.buf_indirect_phase2,
                16,
                self.buf_indirect_phase2,
                0,
                self.max_draws,
                stride=20,
            )
            self.vk.end_rendering(cur_slot)

        # Bucket 2 Hardware Alpha-Cutout VisBuffer Raster (`pipe_visbuffer_alphamask_gfx`, `early_depth = False`, `CULL_MODE_NONE`)
        if am_vis_draws > 0:
            self.vk.memory_barrier2(
                cur_slot,
                PIPELINE_COLOR_ATTACHMENT_OUTPUT | PIPELINE_LATE_FRAGMENT_TESTS,
                Int(ACCESS_COLOR_ATTACHMENT_WRITE | ACCESS_DEPTH_ATTACHMENT_WRITE),
                PIPELINE_COLOR_ATTACHMENT_OUTPUT | PIPELINE_EARLY_FRAGMENT_TESTS | PIPELINE_LATE_FRAGMENT_TESTS,
                Int(ACCESS_COLOR_ATTACHMENT_WRITE | ACCESS_DEPTH_ATTACHMENT_READ | ACCESS_DEPTH_ATTACHMENT_WRITE),
            )
            self.vk.begin_rendering(
                cur_slot, 0, 0, self.render_w, self.render_h, True, self.vis_img.view, False, 0, 0, True, self.depth_img.view, False, 0.0
            )
            self.vk.set_viewport_and_scissor(cur_slot, 0, 0, self.render_w, self.render_h, flip_y=True)
            var push_vb_am = List[UInt32](length=16, fill=0)
            push_vb_am[0] = 0
            push_vb_am[1] = bitcast[DType.uint32](jx_clip)
            push_vb_am[2] = bitcast[DType.uint32](jy_clip)
            self.vk.bind_graphics_pipeline(cur_slot, self.pipe_visbuffer_alphamask_gfx, self.pipe_layout, self.desc_set, push_vb_am)
            self.vk.bind_index_buffer(cur_slot, self.buf_indices, 0, INDEX_TYPE_UINT32)
            if sota_e_gpu_lists:  # sota:E hook
                self.vk.draw_indexed_indirect_count(cur_slot, self.buf_alphamask_indirect, 16, self.buf_alphamask_indirect, 0, am_vis_draws, stride=20)
            else:
                self.vk.draw_indexed_indirect(
                    cur_slot,
                    self.buf_alphamask_indirect,
                    16,
                    am_vis_draws,
                    stride=20,
                )
            self.vk.end_rendering(cur_slot)

        # Hardware Clip-Space Variable-Width Per-Vertex Outline Pass (`pipe_npr_outline_gfx`, `CULL_MODE_FRONT`, Reverse-Z)
        if outline_vis_draws > 0 and self.pipe_npr_outline_gfx != 0:
            self.vk.memory_barrier2(
                cur_slot,
                PIPELINE_COLOR_ATTACHMENT_OUTPUT | PIPELINE_LATE_FRAGMENT_TESTS,
                Int(ACCESS_COLOR_ATTACHMENT_WRITE | ACCESS_DEPTH_ATTACHMENT_WRITE),
                PIPELINE_COLOR_ATTACHMENT_OUTPUT | PIPELINE_EARLY_FRAGMENT_TESTS | PIPELINE_LATE_FRAGMENT_TESTS,
                Int(ACCESS_COLOR_ATTACHMENT_WRITE | ACCESS_DEPTH_ATTACHMENT_READ | ACCESS_DEPTH_ATTACHMENT_WRITE),
            )
            self.vk.begin_rendering(
                cur_slot, 0, 0, self.render_w, self.render_h, True, self.vis_img.view, False, 0, 0, True, self.depth_img.view, False, 0.0
            )
            self.vk.set_viewport_and_scissor(cur_slot, 0, 0, self.render_w, self.render_h, flip_y=True)
            var push_out = List[UInt32](length=16, fill=0)
            push_out[0] = 0
            push_out[1] = bitcast[DType.uint32](jx_clip)
            push_out[2] = bitcast[DType.uint32](jy_clip)
            push_out[3] = UInt32(self.render_w)
            push_out[4] = UInt32(self.render_h)
            self.vk.bind_graphics_pipeline(cur_slot, self.pipe_npr_outline_gfx, self.pipe_layout, self.desc_set, push_out)
            self.vk.bind_index_buffer(cur_slot, self.buf_indices, 0, INDEX_TYPE_UINT32)
            if sota_e_gpu_lists:  # sota:E hook
                self.vk.draw_indexed_indirect_count(cur_slot, self.buf_outline_indirect, 16, self.buf_outline_indirect, 0, outline_vis_draws, stride=20)
            else:
                self.vk.draw_indexed_indirect(
                    cur_slot,
                    self.buf_outline_indirect,
                    16,
                    outline_vis_draws,
                    stride=20,
                )
            self.vk.end_rendering(cur_slot)

        self.vk.transition_image(
            cur_slot, self.vis_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
        )
        self.vk.transition_image(
            cur_slot, self.depth_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
        )
        if use_prof:
            self.vk.write_timestamp(cur_slot, q_vb + 1, PIPELINE_BOTTOM)
        # --- sota:geometry hook --- virtual-texture feedback (1/16 of the visbuffer pixels)
        if self.sota_vt.count() > 0:
            var q_vt = next_q
            if use_prof and q_vt + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_vt, PIPELINE_TOP)
                next_q += 2
                prof_pairs.append((String("SotaVtFeedback"), q_vt))
            self.sota_vt.record_feedback(self.vk, cur_slot, self.render_w, self.render_h, self.frame_index + 1)
            if use_prof and q_vt + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_vt + 1, PIPELINE_BOTTOM)
        # --- end sota:geometry hook ---

        # VSM Mark Visible Virtual Shadow Pages (16k x 16k)
        if vsm_active:
            var q_vm = next_q
            if use_prof:
                self.vk.write_timestamp(cur_slot, q_vm, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("VsmMarkPages"), q_vm))
            var push_vm = List[UInt32](length=16, fill=0)
            push_vm[0] = UInt32(self.render_w)
            push_vm[1] = UInt32(self.render_h)
            self.vk.dispatch_aaa(
                cur_slot, self.pipe_vsm_mark, self.pipe_layout, self.desc_set, push_vm, (self.render_w + 7) // 8, (self.render_h + 7) // 8
            )
            if use_prof:
                self.vk.write_timestamp(cur_slot, q_vm + 1, PIPELINE_BOTTOM)
            self.vk.memory_barrier2(cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
            if sh_gpu:  # --- sota:shadow hook --- allocation + cull + one-pass atlas raster
                var q_sg = next_q
                if use_prof and q_sg + 1 < q_limit:
                    self.vk.write_timestamp(cur_slot, q_sg, PIPELINE_BOTTOM)
                    next_q += 2
                    prof_pairs.append((String("SotaShadowVsmGpu"), q_sg))
                self.sota_shadow_gpu.record(
                    self.vk, cur_slot, cur_slot, self.pipe_layout, self.desc_set, self.vsm_img, VSM_PAGE_TEXELS * 32
                )
                if use_prof and q_sg + 1 < q_limit:
                    self.vk.write_timestamp(cur_slot, q_sg + 1, PIPELINE_BOTTOM)

        # Synchronous Frame-0 / Camera-Teleport VSM Page Readback (skipped in warmed-up single-command-buffer gameplay!)
        if not single_cmd_buf:
            self.vk.memory_barrier2(cur_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
            var f_group_a = self.vk.submit(cur_slot)
            if f_group_a == 0 or not self.vk.wait(f_group_a):
                return self._sota_frame_fail("VSM readback submit/wait failed (device lost?)")  # sota:geometry hook
            self.last_culled_meshlet_draws = Int(get32(self.buf_ldr_readback.cpu_address))
            self.vsm_warmed_up = True

            var tbl = List[UInt32](length=VSM_TABLE_WORDS, fill=0)
            _ = external_call["memcpy", Int](Int(tbl.unsafe_ptr()), self.buf_vsm_table.cpu_address, VSM_TABLE_WORDS * 4)
            var saved_reqs = List[Int]()
            for v_i in range(16384):
                if tbl[16384 + v_i] != 0:
                    saved_reqs.append(v_i)
            var pages = self.vsm.process_requests(tbl)
            # Preserve GPU-marked request flags in `buf_vsm_table` so the next static frame runs `single_cmd_buf = True` and hits cache!
            for v_i in saved_reqs:
                tbl[16384 + v_i] = 1
            _ = external_call["memcpy", Int](self.buf_vsm_table.cpu_address, Int(tbl.unsafe_ptr()), VSM_TABLE_WORDS * 4)
            _ = len(tbl)
            for pg in pages:
                vsm_allocs.append((pg.vx, pg.vy, pg.px, pg.py))
            if len(vsm_allocs) > 0 and (self.last_culled_meshlet_draws > 0 or am_vis_draws > 0):
                var packed_vsm = self._pack_vsm_indirect_commands(n_inst, light_vp, vsm_allocs)
                vsm_page_offsets = packed_vsm[1].copy()
                vsm_page_counts = packed_vsm[2].copy()
                vsm_page_am_offsets = packed_vsm[3].copy()
                vsm_page_am_counts = packed_vsm[4].copy()

            if not self.vk.begin_commands(cur_slot):
                return self._sota_frame_fail("begin_commands (pass group B) failed")  # sota:geometry hook

        # ====================================================================
        # PASS GROUP B: VSM Indirect Tile Raster + 2-Tier Lumen Surface-Cache +
        #               Deferred Multi-Lobe RT Shade + SVGF + OIT + Post + TSR v2!
        # ====================================================================
        self.last_vsm_dirty_pages = len(vsm_allocs) if not sh_gpu else self.sota_shadow_gpu.dirty_last  # sota:shadow hook
        if len(vsm_allocs) > 0:
            var q_vr = next_q
            if use_prof and q_vr + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_vr, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("VsmTileRaster"), q_vr))
            self.vk.transition_image(
                cur_slot, self.vsm_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE
            )
            for p_idx in range(len(vsm_allocs)):
                var pg = vsm_allocs[p_idx]
                var vx = pg[0]; var vy = pg[1]; var px = pg[2]; var py = pg[3]
                var rx = px * VSM_PAGE_TEXELS
                var ry = py * VSM_PAGE_TEXELS
                var sc_f = Float32(128.0)
                var bx_f = Float32(127 - 2 * vx)
                var by_f = -Float32(127 - 2 * vy)
                var mat_word_off = UInt32(56)
                if use_vsm_casc_frame:
                    var casc = vsm_cascade_from_virtual_page(vx, vy)
                    var loc_vx = vx & (VSM_CASCADE_PAGES_DIM - 1)
                    var loc_vy = vy & (VSM_CASCADE_PAGES_DIM - 1)
                    if self.sota_shadow.toroidal_active:  # --- sota:shadow hook --- slot -> window-local page
                        var loc_t = sota_shadow_local_from_slot(vx, vy, self.sota_shadow.offsets)
                        loc_vx = loc_t[0]
                        loc_vy = loc_t[1]
                    sc_f = Float32(VSM_CASCADE_PAGES_DIM)
                    bx_f = Float32((VSM_CASCADE_PAGES_DIM - 1) - 2 * loc_vx)
                    by_f = -Float32((VSM_CASCADE_PAGES_DIM - 1) - 2 * loc_vy)
                    mat_word_off = UInt32(vsm_cascade_globals_word_offset(casc))
                self.vk.begin_rendering(
                    cur_slot, rx, ry, VSM_PAGE_TEXELS, VSM_PAGE_TEXELS, False, 0, False, 0, 0, True, self.vsm_img.view, True, 1.0
                )
                var p_cnt = vsm_page_counts[p_idx] if p_idx < len(vsm_page_counts) else 0
                var p_am_cnt = vsm_page_am_counts[p_idx] if p_idx < len(vsm_page_am_counts) else 0
                if p_cnt > 0 or p_am_cnt > 0:
                    self.vk.set_viewport_and_scissor(cur_slot, rx, ry, VSM_PAGE_TEXELS, VSM_PAGE_TEXELS, flip_y=True)
                    var push_vsm = List[UInt32](length=16, fill=0)
                    push_vsm[0] = bitcast[DType.uint32](sc_f)
                    push_vsm[1] = bitcast[DType.uint32](sc_f)
                    push_vsm[2] = bitcast[DType.uint32](bx_f)
                    push_vsm[3] = bitcast[DType.uint32](by_f)
                    push_vsm[4] = mat_word_off
                    if p_cnt > 0:
                        var p_off = vsm_page_offsets[p_idx]
                        self.vk.bind_graphics_pipeline(cur_slot, self.pipe_vsm_gfx, self.pipe_layout, self.desc_set, push_vsm)
                        self.vk.draw_indirect(cur_slot, self.buf_vsm_indirect, p_off * 16, p_cnt, stride=16)
                    if p_am_cnt > 0:
                        var p_am_off = vsm_page_am_offsets[p_idx]
                        self.vk.bind_graphics_pipeline(cur_slot, self.pipe_vsm_alphamask_gfx, self.pipe_layout, self.desc_set, push_vsm)
                        self.vk.draw_indirect(cur_slot, self.buf_vsm_indirect, p_am_off * 16, p_am_cnt, stride=16)
                self.vk.end_rendering(cur_slot)

            self.vk.transition_image(
                cur_slot, self.vsm_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
            )
            if use_prof and q_vr + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_vr + 1, PIPELINE_BOTTOM)

        # Update Lumen Direct + Indirect Mesh Surface Cache (only required when RT GI multi-bounce is active!)
        if use_rt_frame and self.enable_rt_gi and sc_inst_count > 0 and self.sota_gi.needs_surface_cache(sota_mode):  # sota:gi hook
            var q_sc = next_q
            if use_prof and q_sc + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_sc, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("LumenSurfaceCache"), q_sc))

            var head_end = (last_large_sc_inst + 1) if last_large_sc_inst >= 0 else 0
            var max_tri_head = 1
            var max_tri_tail = 1
            for s_i in range(len(sc_tri_counts)):
                var tc = sc_tri_counts[s_i]
                if s_i < head_end:
                    if tc > max_tri_head:
                        max_tri_head = tc
                else:
                    if tc > max_tri_tail:
                        max_tri_tail = tc

            var n_bounces = max(1, min(2, self.gi_bounces)) if self.enable_rt_gi else 1
            if sota_mode == SOTA_GI_MODE_HYBRID:  # --- sota:gi hook --- the WRC supplies the bounce
                n_bounces = 1
            for bounce in range(n_bounces):
                if head_end > 0:
                    var push_sc0 = List[UInt32](length=16, fill=0)
                    push_sc0[0] = UInt32(head_end)
                    push_sc0[1] = UInt32(bounce)
                    push_sc0[2] = UInt32(0)
                    self.vk.dispatch_aaa(
                        cur_slot,
                        self.pipe_surfcache,
                        self.pipe_layout,
                        self.desc_set,
                        push_sc0,
                        (head_end + 3) // 4,
                        (max_tri_head + 15) // 16,
                    )
                if head_end < sc_inst_count:
                    var tail_cnt = sc_inst_count - head_end
                    var push_sc1 = List[UInt32](length=16, fill=0)
                    push_sc1[0] = UInt32(sc_inst_count)
                    push_sc1[1] = UInt32(bounce)
                    push_sc1[2] = UInt32(head_end)
                    self.vk.dispatch_aaa(
                        cur_slot,
                        self.pipe_surfcache,
                        self.pipe_layout,
                        self.desc_set,
                        push_sc1,
                        (tail_cnt + 3) // 4,
                        (max_tri_tail + 15) // 16,
                    )
                self.vk.memory_barrier2(
                    cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
                )
            if use_prof and q_sc + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_sc + 1, PIPELINE_BOTTOM)

        # --- sota:gi hook --- (world radiance cache probe update: one 64-ray workgroup per probe)
        if self.sota_gi.uses_wrc(sota_mode):
            var q_wrc = next_q
            if use_prof and q_wrc + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_wrc, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("SotaWorldRadianceCache"), q_wrc))
            var push_wrc = List[UInt32](length=16, fill=0)
            self.vk.dispatch_aaa(cur_slot, self.sota_gi.pipe_wrc, self.pipe_layout, self.desc_set, push_wrc, self.sota_gi.probe_count(), 1)
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            if use_prof and q_wrc + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_wrc + 1, PIPELINE_BOTTOM)
        # --- end sota:gi hook ---

        # `16 x 9 x 24` Logarithmic-Z GPU Clustered Punctual + Heitz `LTC` Area Light Culler (`V18`)!
        if n_cl_lights > 0 and not self.furnace_white_dome:
            var q_cl = next_q
            if use_prof and q_cl + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_cl, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("ClusterLightCull"), q_cl))
            var push_cl = List[UInt32](length=16, fill=0)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_cluster_cull,
                self.pipe_layout,
                self.desc_set,
                push_cl,
                1,
                1,
                AAA_CLUSTER_Z,
            )
            if use_prof and q_cl + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_cl + 1, PIPELINE_BOTTOM)
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )

        var max_pix = max(self.output_w * self.output_h, self.render_w * self.render_h)

        # --- sota:atmo hook --- transmittance / multi-scattering (on change), sky-view and aerial-perspective LUTs
        var run_sota_atmo = self.sota_atmo.active() and not self.furnace_white_dome
        if run_sota_atmo:
            sota_atmo_record_luts(self.sota_atmo, self.vk, cur_slot, self.pipe_layout, self.desc_set, use_prof, next_q, q_limit, prof_pairs)
        # --- end sota:atmo hook ---

        # `160 x 90 x 64` Temporal Froxel Volumetric Scattering Injection + 3D Reprojection Pass!
        var run_froxel = self.enable_froxel_fog and self.fog_density > Float32(1e-6) and not self.furnace_white_dome
        var run_rain = self.enable_gpu_rain and self.rain_intensity > Float32(0.001) and not self.furnace_white_dome
        var frox_in = froxel_ping_word_offset(max_pix) if (self.frame_index & 1) == 0 else froxel_pong_word_offset(max_pix)
        var frox_out = froxel_pong_word_offset(max_pix) if (self.frame_index & 1) == 0 else froxel_ping_word_offset(max_pix)
        var frox_prefix = froxel_prefix_lut_word_offset(max_pix)
        if run_froxel or run_rain:
            var q_fi = next_q
            if use_prof and q_fi + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_fi, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("FroxelInjectAndReproject"), q_fi))
            var eff_frox_fr = 0 if (self.froxel_valid_frames == 0 or self.tsr_history_weight <= Float32(0.01)) else self.froxel_valid_frames
            var push_fi = List[UInt32](length=16, fill=0)
            push_fi[0] = UInt32(frox_in)
            push_fi[1] = UInt32(frox_out)
            push_fi[2] = UInt32(eff_frox_fr)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_froxel_inject,
                self.pipe_layout,
                self.desc_set,
                push_fi,
                (AAA_FROXEL_X + 7) // 8,
                (AAA_FROXEL_Y + 5) // 6,
                AAA_FROXEL_Z,
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            var push_fp = List[UInt32](length=16, fill=0)
            push_fp[0] = UInt32(frox_out)
            push_fp[1] = UInt32(frox_prefix)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_froxel_prefix,
                self.pipe_layout,
                self.desc_set,
                push_fp,
                (AAA_FROXEL_X + 7) // 8,
                (AAA_FROXEL_Y + 5) // 6,
                1,
            )
            if use_prof and q_fi + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_fi + 1, PIPELINE_BOTTOM)
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            self.froxel_valid_frames += 1
        else:
            self.froxel_valid_frames = 0

        # Soft RT sun shadows: area-sun trace + penumbra-sized edge-stopped denoise (`aaa_rt_shadow.mojo`)
        var rt_sun_flt_off = 0
        if (
            self.enable_rt_shadows
            and self.vk.has_ray_query
            and n_inst > 0
            and not self.furnace_white_dome
            and self.pipe_rt_shadow_trace != 0
            and self.pipe_rt_shadow_filter != 0
        ):
            var q_rs = next_q
            if use_prof and q_rs + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_rs, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("RtSoftSunShadows"), q_rs))
            rt_sun_flt_off = rt_shadow_filtered_offset(max_pix)
            var push_rs = List[UInt32](length=16, fill=0)
            push_rs[0] = UInt32(self.render_w)
            push_rs[1] = UInt32(self.render_h)
            push_rs[2] = UInt32(rt_shadow_raw_offset(max_pix))
            push_rs[3] = UInt32(rt_sun_flt_off)
            self.vk.dispatch_aaa(
                cur_slot, self.pipe_rt_shadow_trace, self.pipe_layout, self.desc_set, push_rs, (self.render_w + 7) // 8, (self.render_h + 7) // 8
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            self.vk.dispatch_aaa(
                cur_slot, self.pipe_rt_shadow_filter, self.pipe_layout, self.desc_set, push_rs, (self.render_w + 7) // 8, (self.render_h + 7) // 8
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            if use_prof and q_rs + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_rs + 1, PIPELINE_BOTTOM)

        # Full Deferred `VisBuffer` Multi-Lobe + VSM + `SPV_KHR_ray_query` Shading!
        var run_svgf = self.enable_rt_gi and self.svgf_passes > 0 and not self.furnace_white_dome
        var run_spec_reblur = (
            self.enable_rt_reflections
            and self.enable_spec_reblur
            and self.spec_reblur_passes > 0
            and not self.furnace_white_dome
            and self.vk.has_ray_query
            and n_inst > 0
        )
        var spec_in_off = spec_reblur_raw_offset(max_pix)
        # --- sota:camera hook (compute VRS: classify + compact the pixels to shade) ---
        var sota_vrs_list = 0
        if not self.furnace_white_dome:
            sota_camera_vrs_begin_frame(self.sota_cam, self.render_graph.pass_timings)
            sota_vrs_list = sota_camera_record_vrs_classify(
                self.sota_cam, self.vk, cur_slot, self.pipe_layout, self.desc_set,
                self.render_w, self.render_h, max_pix, use_prof, next_q, q_limit, prof_pairs,
            )
        # --- end sota:camera hook ---
        var q_sh = next_q
        if use_prof and q_sh + 1 < q_limit:
            self.vk.write_timestamp(cur_slot, q_sh, PIPELINE_BOTTOM)
            next_q += 2
            prof_pairs.append((String("DeferredMultiLobeRtShade"), q_sh))
        var push_sh = List[UInt32](length=16, fill=0)
        push_sh[0] = UInt32(self.render_w)
        push_sh[1] = UInt32(self.render_h)
        push_sh[2] = UInt32(max(1, self.gi_rays_per_pixel))
        push_sh[3] = UInt32(1) if run_svgf else UInt32(0)
        push_sh[4] = UInt32(1) if run_spec_reblur else UInt32(0)
        push_sh[5] = UInt32(spec_in_off)
        push_sh[6] = UInt32(rt_sun_flt_off)
        push_sh[7] = UInt32(sota_vrs_list)  # --- sota:camera hook (compute VRS shade list) ---
        push_sh[8] = UInt32((self.render_w + 7) // 8)  # --- sota:camera hook ---
        self.vk.dispatch_aaa(
            cur_slot,
            self.pipe_deferred_shade,
            self.pipe_layout,
            self.desc_set,
            push_sh,
            (self.render_w + 7) // 8,
            (self.render_h + 7) // 8,
        )
        if use_prof and q_sh + 1 < q_limit:
            self.vk.write_timestamp(cur_slot, q_sh + 1, PIPELINE_BOTTOM)
        # --- sota:camera hook (compute VRS: rebuild the skipped pixels from their anchors) ---
        if sota_vrs_list != 0:
            sota_camera_record_vrs_fill(
                self.sota_cam, self.vk, cur_slot, self.pipe_layout, self.desc_set, self.render_w, self.render_h,
                max_pix, run_svgf, run_spec_reblur, spec_in_off, use_prof, next_q, q_limit, prof_pairs,
            )
        # --- end sota:camera hook ---
        self.render_graph.emit_pass_barriers(self.vk, cur_slot, 4)

        # --- sota:gi hook --- (area-B GI resolve: probe GI / hybrid rays / path tracer -> demodulated GI)
        if sota_mode != SOTA_GI_MODE_LEGACY:
            var q_sgr = next_q
            if use_prof and q_sgr + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_sgr, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("SotaGiResolve"), q_sgr))
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            var push_sgr = List[UInt32](length=16, fill=0)
            push_sgr[0] = UInt32(self.render_w)
            push_sgr[1] = UInt32(self.render_h)
            push_sgr[2] = UInt32(1) if run_svgf else UInt32(0)
            # round 3: reduced-resolution resolve (scale, this / previous frame's block sub-pixel)
            var sgr_lowres = self.sota_gi.cur_scale > 1 and self.sota_gi.pipe_upsample != 0
            push_sgr[3] = UInt32(self.sota_gi.cur_scale) if sgr_lowres else UInt32(1)
            push_sgr[4] = UInt32(self.sota_gi.cur_jx) if sgr_lowres else UInt32(0)
            push_sgr[5] = UInt32(self.sota_gi.cur_jy) if sgr_lowres else UInt32(0)
            push_sgr[6] = UInt32(self.sota_gi.prev_jx) if sgr_lowres else UInt32(0)
            push_sgr[7] = UInt32(self.sota_gi.prev_jy) if sgr_lowres else UInt32(0)
            push_sgr[8] = bitcast[DType.uint32](self.sota_gi.upsample_tent)
            var sgr_sc = Int(push_sgr[3])
            var sgr_w = (self.render_w + sgr_sc - 1) // sgr_sc
            var sgr_h = (self.render_h + sgr_sc - 1) // sgr_sc
            self.vk.dispatch_aaa(
                cur_slot, self.sota_gi.pipe_resolve, self.pipe_layout, self.desc_set, push_sgr,
                (sgr_w + 7) // 8, (sgr_h + 7) // 8,
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            if use_prof and q_sgr + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_sgr + 1, PIPELINE_BOTTOM)
            if sgr_lowres:
                var q_sgu = next_q
                if use_prof and q_sgu + 1 < q_limit:
                    self.vk.write_timestamp(cur_slot, q_sgu, PIPELINE_BOTTOM)
                    next_q += 2
                    prof_pairs.append((String("SotaGiUpsample"), q_sgu))
                self.vk.dispatch_aaa(
                    cur_slot, self.sota_gi.pipe_upsample, self.pipe_layout, self.desc_set, push_sgr,
                    (self.render_w + 7) // 8, (self.render_h + 7) // 8,
                )
                self.vk.memory_barrier2(
                    cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
                )
                if use_prof and q_sgu + 1 < q_limit:
                    self.vk.write_timestamp(cur_slot, q_sgu + 1, PIPELINE_BOTTOM)
        self.sota_gi.end_frame(sota_mode)
        # --- end sota:gi hook ---

        # --- sota:ml hook --- opt-in learned denoiser replaces SVGF (weights present, GI mode allowed)
        var run_sota_ml = run_svgf and self.sota_ml.ready(self.vk, self.pipe_layout, self.sota_gi.mode)
        if run_sota_ml:
            var q_ml = next_q
            if use_prof and q_ml + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_ml, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("SotaMlDenoise"), q_ml))
            self.sota_ml.record(self.vk, cur_slot, self.pipe_layout, self.desc_set, self.render_w, self.render_h)
            if use_prof and q_ml + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_ml + 1, PIPELINE_BOTTOM)
        else:
            self.sota_ml.skipped()
        # --- end sota:ml hook ---

        # `SVGF` Temporal Reprojection + Moments Variance Edge-Stopping `À-Trous` Denoiser
        if run_svgf and not run_sota_ml:  # sota:ml
            var q_sv = next_q
            if use_prof and q_sv + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_sv, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("SvgfTemporalAndAtrous"), q_sv))

            var svgf_hist_in = (max_pix * 8) if (self.frame_index & 1) == 0 else 0
            var svgf_hist_out = 0 if (self.frame_index & 1) == 0 else (max_pix * 8)
            var push_rep = List[UInt32](length=16, fill=0)
            push_rep[0] = UInt32(self.render_w)
            push_rep[1] = UInt32(self.render_h)
            push_rep[2] = UInt32(svgf_hist_in)
            push_rep[3] = UInt32(svgf_hist_out)
            push_rep[4] = UInt32(self.frame_index)
            push_rep[5] = UInt32(1) if self.sota_gi.svgf_jitter_tol else UInt32(0)  # --- sota:gi hook --- (round 4)
            push_rep[6] = bitcast[DType.uint32](self.sota_gi.svgf_history_cap(sota_mode, self.sota_vsm.gi_boost_left > 0))  # sota:gi hook
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_svgf_reproject,
                self.pipe_layout,
                self.desc_set,
                push_rep,
                (self.render_w + 7) // 8,
                (self.render_h + 7) // 8,
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )

            var n_steps = ((self.svgf_passes + 1) // 2) * 2
            for st_i in range(n_steps):
                var push_sv = List[UInt32](length=16, fill=0)
                push_sv[0] = UInt32(self.render_w)
                push_sv[1] = UInt32(self.render_h)
                push_sv[2] = UInt32(1 << st_i)
                push_sv[3] = UInt32(st_i & 1)
                push_sv[4] = bitcast[DType.uint32](self.svgf_sigma_luma)
                push_sv[5] = UInt32(1) if (st_i + 1 == n_steps) else UInt32(0)
                # --- sota:gi hook --- a-trous ping-pong scratch past the history regions (0 = legacy bank A)
                if self.sota_gi.svgf_history_fix:
                    push_sv[6] = UInt32(rt_shadow_raw_offset(max_pix) + rt_shadow_total_words(max_pix))
                self.vk.dispatch_aaa(
                    cur_slot,
                    self.pipe_svgf_atrous,
                    self.pipe_layout,
                    self.desc_set,
                    push_sv,
                    (self.render_w + 7) // 8,
                    (self.render_h + 7) // 8,
                )
                self.vk.memory_barrier2(
                    cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
                )
            if use_prof and q_sv + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_sv + 1, PIPELINE_BOTTOM)

        # `NRD ReBLUR`-Lite Dual Virtual-Depth Spatiotemporal Specular Denoiser!
        if run_spec_reblur:
            var q_rb = next_q
            if use_prof and q_rb + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_rb, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("SpecReblurTemporalAtrous"), q_rb))
            var spec_h_in = spec_reblur_hist_ping_offset(max_pix) if (self.frame_index & 1) == 0 else spec_reblur_hist_pong_offset(max_pix)
            var spec_h_out = spec_reblur_hist_pong_offset(max_pix) if (self.frame_index & 1) == 0 else spec_reblur_hist_ping_offset(max_pix)
            var spec_scratch = spec_reblur_scratch_offset(max_pix)
            var eff_rb_frame = 0 if self.tsr_history_weight <= Float32(0.01) else self.spec_reblur_valid_frames

            var push_rbt = List[UInt32](length=16, fill=0)
            push_rbt[0] = UInt32(self.render_w)
            push_rbt[1] = UInt32(self.render_h)
            push_rbt[2] = UInt32(spec_in_off)
            push_rbt[3] = UInt32(spec_h_in)
            push_rbt[4] = UInt32(spec_h_out)
            push_rbt[5] = UInt32(spec_scratch)
            push_rbt[6] = UInt32(eff_rb_frame)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_spec_reblur_temp,
                self.pipe_layout,
                self.desc_set,
                push_rbt,
                (self.render_w + 7) // 8,
                (self.render_h + 7) // 8,
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )

            var n_rb = max(1, self.spec_reblur_passes)
            for rb_i in range(n_rb):
                var rb_src = spec_scratch if (rb_i & 1) == 0 else spec_h_in
                var rb_dst = spec_h_in if (rb_i & 1) == 0 else spec_scratch
                var push_rba = List[UInt32](length=16, fill=0)
                push_rba[0] = UInt32(self.render_w)
                push_rba[1] = UInt32(self.render_h)
                push_rba[2] = UInt32(1 << rb_i)
                push_rba[3] = UInt32(rb_src)
                push_rba[4] = UInt32(rb_dst)
                push_rba[5] = UInt32(spec_in_off)
                push_rba[6] = UInt32(1) if (rb_i + 1 == n_rb) else UInt32(0)
                push_rba[7] = UInt32(spec_scratch)
                self.vk.dispatch_aaa(
                    cur_slot,
                    self.pipe_spec_reblur_atrous,
                    self.pipe_layout,
                    self.desc_set,
                    push_rba,
                    (self.render_w + 7) // 8,
                    (self.render_h + 7) // 8,
                )
                self.vk.memory_barrier2(
                    cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
                )
            if use_prof and q_rb + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_rb + 1, PIPELINE_BOTTOM)
            self.spec_reblur_valid_frames += 1
        else:
            self.spec_reblur_valid_frames = 0

        # Stylized Anime NPR Cel, Outline & Eroding 3D Mesh VFX Resolve (`aaa_npr_cel.mojo`, `aaa_npr_vfx.mojo`)
        if (n_cel > 0 or n_vfx > 0) and self.pipe_npr_cel_shade != 0:
            var q_cel = next_q
            if use_prof and q_cel + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_cel, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("NprCelAndStylizedVfxShade"), q_cel))
            var push_cel = List[UInt32](length=16, fill=0)
            push_cel[0] = UInt32(self.render_w)
            push_cel[1] = UInt32(self.render_h)
            push_cel[2] = UInt32(rt_sun_flt_off)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_npr_cel_shade,
                self.pipe_layout,
                self.desc_set,
                push_cel,
                (self.render_w + 7) // 8,
                (self.render_h + 7) // 8,
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            if use_prof and q_cel + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_cel + 1, PIPELINE_BOTTOM)

        # Ray-traced refractive glass over the shaded, denoised frame (`aaa_glass.mojo`)
        if frame_has_glass and use_rt_frame and self.pipe_glass_refract != 0 and not self.furnace_white_dome:
            var q_gl = next_q
            if use_prof and q_gl + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_gl, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("RtRefractiveGlass"), q_gl))
            var push_gl = List[UInt32](length=16, fill=0)
            push_gl[0] = UInt32(self.render_w)
            push_gl[1] = UInt32(self.render_h)
            self.vk.dispatch_aaa(
                cur_slot, self.pipe_glass_refract, self.pipe_layout, self.desc_set, push_gl, (self.render_w + 7) // 8, (self.render_h + 7) // 8
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            if use_prof and q_gl + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_gl + 1, PIPELINE_BOTTOM)

        var run_water = self.enable_water_plane and not self.furnace_white_dome

        # When Water or GPU Rain will write partial reactive weights into `b_oit`, clear `b_oit[react]` first!
        if run_water or run_rain:
            var push_clr_react = List[UInt32](length=16, fill=0)
            push_clr_react[0] = UInt32(self.render_w)
            push_clr_react[1] = UInt32(self.render_h)
            push_clr_react[2] = UInt32(0)
            push_clr_react[3] = UInt32(0)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_oit_vol,
                self.pipe_layout,
                self.desc_set,
                push_clr_react,
                (self.render_w + 7) // 8,
                (self.render_h + 7) // 8,
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )

        # Dielectric `Beer-Lambert` Refracting + Caustic + Foaming Water Pass (`aaa_water.mojo`)!
        if run_water:
            var q_wt = next_q
            if use_prof and q_wt + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_wt, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("DielectricBeerLambertWater"), q_wt))
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ)
            )
            self.vk.copy_buffer(
                cur_slot,
                self.buf_hdr_out,
                0,
                self.buf_post_scratch,
                AAA_POST_BLOOM_BASE * 4,
                self.render_w * self.render_h * 8 * 4,
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            var push_wt = List[UInt32](length=16, fill=0)
            push_wt[0] = UInt32(self.render_w)
            push_wt[1] = UInt32(self.render_h)
            push_wt[2] = UInt32(AAA_POST_BLOOM_BASE)
            push_wt[3] = UInt32(1) if use_rt_frame else UInt32(0)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_water_refract,
                self.pipe_layout,
                self.desc_set,
                push_wt,
                (self.render_w + 7) // 8,
                (self.render_h + 7) // 8,
            )
            if use_prof and q_wt + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_wt + 1, PIPELINE_BOTTOM)
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )

        # --- sota:atmo hook --- sky / sun / moon, aerial perspective, height-fog tail (before the froxel composite)
        if run_sota_atmo:
            sota_atmo_record_apply(
                self.sota_atmo, self.vk, cur_slot, self.pipe_layout, self.desc_set, self.render_w, self.render_h,
                use_prof, next_q, q_limit, prof_pairs,
            )
        # --- end sota:atmo hook ---

        # 64-Slice Hillaire Froxel Prefix-LUT Composite + GPU Froxel-Lit 3D Rain Streaks!
        if run_froxel or run_rain:
            var q_fm = next_q
            if use_prof and q_fm + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_fm, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("FroxelRaymarchAndRain"), q_fm))
            var push_fm = List[UInt32](length=16, fill=0)
            push_fm[0] = UInt32(self.render_w)
            push_fm[1] = UInt32(self.render_h)
            push_fm[2] = UInt32(frox_prefix)
            push_fm[3] = UInt32(1) if run_rain else UInt32(0)
            push_fm[4] = UInt32(frox_out)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_froxel_march,
                self.pipe_layout,
                self.desc_set,
                push_fm,
                (self.render_w + 7) // 8,
                (self.render_h + 7) // 8,
            )
            if use_prof and q_fm + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_fm + 1, PIPELINE_BOTTOM)
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )

        # Per-Pixel 4-Node Exact `OIT` + 6-Way Lightmapped Volumetric Plumes + Fresnel Rim Holograms!
        if len(self.frame_plumes) > 0 or not (run_water or run_rain):
            var q_oit = next_q
            if use_prof and q_oit + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_oit, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("Oit6WayVolumetrics"), q_oit))
            var push_oit = List[UInt32](length=16, fill=0)
            push_oit[0] = UInt32(self.render_w)
            push_oit[1] = UInt32(self.render_h)
            push_oit[2] = UInt32(len(self.frame_plumes))
            push_oit[3] = UInt32(1) if (run_water or run_rain) else UInt32(0)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_oit_vol,
                self.pipe_layout,
                self.desc_set,
                push_oit,
                (self.render_w + 7) // 8,
                (self.render_h + 7) // 8,
            )
            if use_prof and q_oit + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_oit + 1, PIPELINE_BOTTOM)
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )

        # `64K` GPU Compute Particle Simulation, `16x16` Tile Binning & Froxel-Lit Composite (`aaa_gpu_particles.mojo`)!
        if self.particles_ever_active and not self.furnace_white_dome:
            var q_pt = next_q
            if use_prof and q_pt + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_pt, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("GpuComputeParticles64K"), q_pt))
            var em_off = gpu_particles_emitters_word_offset(max_pix)
            var pool_off = gpu_particles_pool_word_offset(max_pix)
            var t_cnt_off = gpu_particles_tile_counts_word_offset(max_pix)
            var t_bin_off = gpu_particles_tile_bins_word_offset(max_pix)
            var n_em = min(AAA_MAX_PARTICLE_EMITTERS, len(self.frame_particle_emitters))
            if n_em > 0:
                var em_words = List[UInt32](capacity=n_em * AAA_EMITTER_WORDS)
                for e_i in range(n_em):
                    var sc_e = max(0, min(4095, self.frame_particle_emitters[e_i].spawn_count))
                    self.frame_particle_emitters[e_i].append_words(em_words, self.particle_ring_cursor)
                    self.particle_ring_cursor = (self.particle_ring_cursor + sc_e) & 0xFFFF
                _ = external_call[
                    "memcpy", Int
                ](self.buf_oit_nodes.cpu_address + em_off * 4, Int(em_words.unsafe_ptr()), len(em_words) * 4)
                _ = len(em_words)

            var tiles_x = (self.render_w + 15) // 16
            var tiles_y = (self.render_h + 15) // 16
            var num_tiles = tiles_x * tiles_y

            var push_pc = List[UInt32](length=16, fill=0)
            push_pc[0] = UInt32(t_cnt_off)
            push_pc[1] = UInt32(num_tiles)
            self.vk.dispatch_aaa(
                cur_slot, self.pipe_particle_clear, self.pipe_layout, self.desc_set, push_pc, (num_tiles + 63) // 64, 1
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )

            var push_ps = List[UInt32](length=16, fill=0)
            push_ps[0] = UInt32(self.render_w)
            push_ps[1] = UInt32(self.render_h)
            push_ps[2] = bitcast[DType.uint32](max(Float32(0.0005), min(Float32(0.1), self.particle_dt_sec)))
            push_ps[3] = UInt32(n_em)
            push_ps[4] = UInt32(em_off)
            push_ps[5] = UInt32(pool_off)
            push_ps[6] = UInt32(t_cnt_off)
            push_ps[7] = UInt32(t_bin_off)
            self.vk.dispatch_aaa(
                cur_slot, self.pipe_particle_sim, self.pipe_layout, self.desc_set, push_ps, (AAA_MAX_PARTICLES + 63) // 64, 1
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )

            var push_pcomp = List[UInt32](length=16, fill=0)
            push_pcomp[0] = UInt32(self.render_w)
            push_pcomp[1] = UInt32(self.render_h)
            push_pcomp[2] = UInt32(pool_off)
            push_pcomp[3] = UInt32(t_cnt_off)
            push_pcomp[4] = UInt32(t_bin_off)
            push_pcomp[5] = UInt32(frox_prefix)
            push_pcomp[6] = UInt32(1) if (run_froxel or run_rain) else UInt32(0)
            self.vk.dispatch_aaa(
                cur_slot, self.pipe_particle_comp, self.pipe_layout, self.desc_set, push_pcomp, tiles_x, tiles_y
            )
            if use_prof and q_pt + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_pt + 1, PIPELINE_BOTTOM)
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )

        # `FlintTSR v3` Pre-Bloom Temporal Upscale & 16-Phase Halton Anti-Aliasing (`render_w x render_h -> output_w x output_h`)!
        # --- sota:ml hook --- learned resolve (same bindings and push layout as TSR v3)
        var use_ml_up = self.sota_ml_up.ready(self.vk, self.pipe_layout, self.render_w < self.output_w)
        self.sota_ml_up.record_capture(self.vk, cur_slot, self.buf_hdr_out, max_pix, self.render_w, self.render_h)
        # --- end sota:ml hook ---
        var q_tsr = next_q
        if use_prof and q_tsr + 1 < q_limit:
            self.vk.write_timestamp(cur_slot, q_tsr, PIPELINE_BOTTOM)
            next_q += 2
            prof_pairs.append((String("SotaMlUpscale") if use_ml_up else String("FlintTsrV3Resolve"), q_tsr))  # sota:ml
        var ping_off = max_pix * 16
        var pong_off = max_pix * 20
        var out_ldr_off = max_pix * 24
        var hist_in = ping_off if (self.frame_index & 1) == 0 else pong_off
        var hist_out = pong_off if (self.frame_index & 1) == 0 else ping_off
        var eff_hist_w = Float32(0.0) if (self.frame_index == 0 or self.furnace_white_dome) else self.tsr_history_weight
        eff_hist_w = self.sota_vsm.tsr_history_cap(eff_hist_w)  # --- sota:perf hook --- lighting cut drops stale TSR history

        var push_tsr = List[UInt32](length=16, fill=0)
        push_tsr[0] = UInt32(self.render_w)
        push_tsr[1] = UInt32(self.render_h)
        push_tsr[2] = UInt32(self.output_w)
        push_tsr[3] = UInt32(self.output_h)
        push_tsr[4] = bitcast[DType.uint32](eff_hist_w)
        push_tsr[5] = UInt32(hist_in)
        push_tsr[6] = UInt32(hist_out)
        push_tsr[7] = UInt32(AAA_POST_BLOOM_BASE)
        push_tsr[8] = bitcast[DType.uint32](self.tsr_sharpness)
        self.vk.dispatch_aaa(
            cur_slot,
            self.sota_ml_up.pipe if use_ml_up else self.pipe_tsr_v3_resolve,  # sota:ml
            self.pipe_layout,
            self.desc_set,
            push_tsr,
            (self.output_w + 7) // 8,
            (self.output_h + 7) // 8,
        )
        self.vk.memory_barrier2(
            cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ)
        )
        self.vk.copy_buffer(
            cur_slot,
            self.buf_post_scratch,
            AAA_POST_BLOOM_BASE * 4,
            self.buf_hdr_out,
            0,
            self.output_w * self.output_h * 8 * 4,
        )
        self.vk.memory_barrier2(
            cur_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
        )
        if use_prof and q_tsr + 1 < q_limit:
            self.vk.write_timestamp(cur_slot, q_tsr + 1, PIPELINE_BOTTOM)

        # --- sota:camera hook (scatter-as-gather DoF replaces the legacy pass when enabled) ---
        var sota_dof_on = self.sota_cam.enable_dof and self.sota_cam.pipelines_ready() and not self.furnace_white_dome
        if sota_dof_on:
            sota_camera_record_dof(
                self.sota_cam, self.vk, cur_slot, self.pipe_layout, self.desc_set,
                self.output_w, self.output_h, use_prof, next_q, q_limit, prof_pairs,
            )
        if self.sota_cam.enable_chromatic_aberration and self.sota_cam.pipelines_ready() and not self.furnace_white_dome:
            sota_camera_record_ca(  # lateral CA before bloom / motion blur (physical order)
                self.sota_cam, self.vk, cur_slot, self.pipe_layout, self.desc_set,
                self.output_w, self.output_h, max_pix, use_prof, next_q, q_limit, prof_pairs,
            )
        # --- end sota:camera hook ---
        # Thin-Lens Polygonal Diaphragm Bokeh Depth-of-Field at Native `output_w x output_h`!
        if self.enable_bokeh_dof and not self.furnace_white_dome and not sota_dof_on:
            var q_dof = next_q
            if use_prof and q_dof + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_dof, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("PolygonalBokehDoF"), q_dof))
            var dof_off = dof_scratch_word_offset(max_pix)
            var coc_sc = self.camera_lens.compute_coc_scale_px(self.output_h)
            for dof_mode in range(2):
                var push_dof = List[UInt32](length=16, fill=0)
                push_dof[0] = UInt32(self.output_w)
                push_dof[1] = UInt32(self.output_h)
                push_dof[2] = UInt32(dof_mode)
                push_dof[3] = UInt32(dof_off)
                push_dof[4] = bitcast[DType.uint32](self.camera_lens.focus_distance_m)
                push_dof[5] = bitcast[DType.uint32](coc_sc)
                push_dof[6] = bitcast[DType.uint32](self.camera_lens.max_coc_px)
                push_dof[7] = bitcast[DType.uint32](Float32(self.camera_lens.diaphragm_blades))
                self.vk.dispatch_aaa(
                    cur_slot,
                    self.pipe_bokeh_dof,
                    self.pipe_layout,
                    self.desc_set,
                    push_dof,
                    (self.output_w + 7) // 8,
                    (self.output_h + 7) // 8,
                )
                self.vk.memory_barrier2(
                    cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
                )
            if use_prof and q_dof + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_dof + 1, PIPELINE_BOTTOM)

        # Screen-Space Chromatic-Dispersion Shockwaves & Manga Impact Frame (`aaa_npr_vfx.mojo`)
        if (n_sw > 0 or imp_blend > Float32(0.001)) and self.pipe_npr_post_fx != 0:
            var q_npf = next_q
            if use_prof and q_npf + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_npf, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("NprShockwaveAndImpactPostFx"), q_npf))
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ)
            )
            self.vk.copy_buffer(
                cur_slot,
                self.buf_hdr_out,
                0,
                self.buf_post_scratch,
                AAA_POST_BLOOM_BASE * 4,
                self.output_w * self.output_h * 8 * 4,
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            var push_npf = List[UInt32](length=16, fill=0)
            push_npf[0] = UInt32(self.output_w)
            push_npf[1] = UInt32(self.output_h)
            push_npf[2] = UInt32(AAA_POST_BLOOM_BASE)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_npr_post_fx,
                self.pipe_layout,
                self.desc_set,
                push_npf,
                (self.output_w + 7) // 8,
                (self.output_h + 7) // 8,
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            if use_prof and q_npf + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_npf + 1, PIPELINE_BOTTOM)

        # Energy-Conserving 5-Mip Dual-Kawase + Starburst/Anamorphic HDR Bloom at Native `output_w x output_h`!
        var has_saved_bloom_mip = False
        var saved_bm_w = 0
        var saved_bm_h = 0
        var safe_bloom_mip_off = AAA_POST_BLOOM_BASE + max_pix * 8
        if self.bloom_intensity > Float32(1e-4) and not self.furnace_white_dome:
            var q_bl = next_q
            if use_prof and q_bl + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_bl, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("DualKawaseBloom5Mip"), q_bl))
            var mip_w = List[Int]()
            var mip_h = List[Int]()
            var mip_off = List[Int]()
            for m_i in range(AAA_POST_BLOOM_MIPS):
                mip_w.append(max(2, self.output_w >> (m_i + 1)))
                mip_h.append(max(2, self.output_h >> (m_i + 1)))
                mip_off.append(bloom_mip_word_offset(max_pix, m_i))

            for m_i in range(AAA_POST_BLOOM_MIPS):
                var sw = self.output_w if m_i == 0 else mip_w[m_i - 1]
                var sh = self.output_h if m_i == 0 else mip_h[m_i - 1]
                var dw = mip_w[m_i]
                var dh = mip_h[m_i]
                var s_off = 0 if m_i == 0 else mip_off[m_i - 1]
                var push_bd = List[UInt32](length=16, fill=0)
                push_bd[0] = UInt32(sw)
                push_bd[1] = UInt32(sh)
                push_bd[2] = UInt32(dw)
                push_bd[3] = UInt32(dh)
                push_bd[4] = UInt32(1) if m_i == 0 else UInt32(0)
                push_bd[5] = UInt32(s_off)
                push_bd[6] = UInt32(mip_off[m_i])
                push_bd[7] = bitcast[DType.uint32](max(Float32(0.0), self.bloom_threshold))
                self.vk.dispatch_aaa(
                    cur_slot, self.pipe_bloom_down, self.pipe_layout, self.desc_set, push_bd, (dw + 7) // 8, (dh + 7) // 8
                )
                self.vk.memory_barrier2(
                    cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
                )

            for rev_i in range(AAA_POST_BLOOM_MIPS):
                var m_src = (AAA_POST_BLOOM_MIPS - 1) - rev_i
                var sw = mip_w[m_src]
                var sh = mip_h[m_src]
                var is_final = m_src == 0
                var dw = self.output_w if is_final else mip_w[m_src - 1]
                var dh = self.output_h if is_final else mip_h[m_src - 1]
                var d_off = 0 if is_final else mip_off[m_src - 1]
                var push_bu = List[UInt32](length=16, fill=0)
                push_bu[0] = UInt32(sw)
                push_bu[1] = UInt32(sh)
                push_bu[2] = UInt32(dw)
                push_bu[3] = UInt32(dh)
                push_bu[4] = UInt32(1) if is_final else UInt32(0)
                push_bu[5] = UInt32(mip_off[m_src])
                push_bu[6] = UInt32(d_off)
                push_bu[7] = bitcast[DType.uint32](min(Float32(1.0), self.bloom_intensity))
                push_bu[8] = UInt32(1) if self.bloom_threshold > Float32(0.001) else UInt32(0)
                push_bu[9] = bitcast[DType.uint32](self.glare_streak_strength)
                push_bu[10] = UInt32(m_src)
                self.vk.dispatch_aaa(
                    cur_slot, self.pipe_bloom_up, self.pipe_layout, self.desc_set, push_bu, (dw + 7) // 8, (dh + 7) // 8
                )
                self.vk.memory_barrier2(
                    cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
                )
            if self.windshield_rain > Float32(0.001) and len(mip_w) > 1:
                saved_bm_w = mip_w[1]
                saved_bm_h = mip_h[1]
                var bm_bytes = saved_bm_w * saved_bm_h * 16
                if (safe_bloom_mip_off * 4 + bm_bytes) <= self.buf_post_scratch.size:
                    self.vk.memory_barrier2(
                        cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ)
                    )
                    self.vk.copy_buffer(
                        cur_slot,
                        self.buf_post_scratch,
                        mip_off[1] * 4,
                        self.buf_post_scratch,
                        safe_bloom_mip_off * 4,
                        bm_bytes,
                    )
                    self.vk.memory_barrier2(
                        cur_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
                    )
                    has_saved_bloom_mip = True
            if use_prof and q_bl + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_bl + 1, PIPELINE_BOTTOM)

        # Tile-Max `180°` Rotary-Shutter Motion Blur (`group = 16 x 16` at Native `output_w x output_h`)!
        if self.enable_motion_blur and self.motion_blur_shutter > Float32(0.001) and not self.furnace_white_dome:
            var q_mb = next_q
            if use_prof and q_mb + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_mb, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("TileMaxMotionBlur"), q_mb))
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ)
            )
            self.vk.copy_buffer(
                cur_slot,
                self.buf_hdr_out,
                0,
                self.buf_post_scratch,
                AAA_POST_BLOOM_BASE * 4,
                self.output_w * self.output_h * 8 * 4,
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            var push_mb = List[UInt32](length=16, fill=0)
            push_mb[0] = UInt32(self.output_w)
            push_mb[1] = UInt32(self.output_h)
            push_mb[2] = UInt32(AAA_POST_BLOOM_BASE)
            push_mb[3] = bitcast[DType.uint32](self.motion_blur_shutter)
            push_mb[4] = bitcast[DType.uint32](self.motion_blur_max_px)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_motion_blur,
                self.pipe_layout,
                self.desc_set,
                push_mb,
                (self.output_w + 15) // 16,
                (self.output_h + 15) // 16,
            )
            if use_prof and q_mb + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_mb + 1, PIPELINE_BOTTOM)
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )

        # --- sota:camera hook (round 4: EV100 auto exposure replaces the v5 meter when enabled) ---
        var sota_ae_on = self.sota_cam.ae.active(self.enable_auto_exposure) and not self.furnace_white_dome
        if sota_ae_on:
            sota_ae_record(
                self.sota_cam.ae, self.vk, cur_slot, self.pipe_layout, self.desc_set, self.output_w, self.output_h,
                self.sota_cam.frame_dt_s, self.ev_compensation, use_prof, next_q, q_limit, prof_pairs,
            )
        # --- end sota:camera hook ---
        # GPU 64-Bin Log-Luminance Histogram & Bounded 95%-Trimmed Auto-Exposure!
        if self.enable_auto_exposure and not self.furnace_white_dome and not sota_ae_on:
            var q_ae = next_q
            if use_prof and q_ae + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_ae, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("LuminanceAutoExposure"), q_ae))
            var push_lh = List[UInt32](length=16, fill=0)
            push_lh[0] = UInt32(self.output_w)
            push_lh[1] = UInt32(self.output_h)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_luma_hist,
                self.pipe_layout,
                self.desc_set,
                push_lh,
                (self.output_w + 7) // 8,
                (self.output_h + 7) // 8,
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
            )
            var push_ae = List[UInt32](length=16, fill=0)
            push_ae[0] = UInt32(self.output_w * self.output_h)
            push_ae[1] = bitcast[DType.uint32](self.auto_exposure_speed)
            push_ae[2] = bitcast[DType.uint32](self.auto_exposure_key)
            push_ae[3] = bitcast[DType.uint32](exp2(self.ev_compensation))
            push_ae[4] = bitcast[DType.uint32](self.min_auto_exposure)
            push_ae[5] = bitcast[DType.uint32](self.max_auto_exposure)
            self.vk.dispatch_aaa(cur_slot, self.pipe_adapt_exp, self.pipe_layout, self.desc_set, push_ae, 1, 1)
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE | PIPELINE_HOST, Int(ACCESS_SHADER_READ | ACCESS_HOST_READ)
            )
            if use_prof and q_ae + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_ae + 1, PIPELINE_BOTTOM)

        # --- sota:camera hook (compute VRS rate map for the next frame, from the finished image before lens noise) ---
        if not self.furnace_white_dome:
            sota_camera_record_vrs_rate(
                self.sota_cam, self.vk, cur_slot, self.pipe_layout, self.desc_set, self.output_w, self.output_h,
                self.render_w, self.render_h, max_pix, use_prof, next_q, q_limit, prof_pairs,
            )
        # --- end sota:camera hook ---
        # --- sota:camera hook (lens: CA, optical vignette, ghosts, dirt; then sensor noise) ---
        if self.sota_cam.any_lens_effect() and self.sota_cam.pipelines_ready() and not self.furnace_white_dome:
            sota_camera_record_lens(
                self.sota_cam, self.vk, cur_slot, self.pipe_layout, self.desc_set, self.buf_hdr_out,
                self.buf_post_scratch, self.output_w, self.output_h, max_pix, use_prof, next_q, q_limit, prof_pairs,
            )
        # --- end sota:camera hook ---
        # Cinema Display Tonemap (`AgX` Toe + `+5.5 EV` Shoulder / `HDR10 PQ` / `Linear sRGB` + Windshield Rain)!
        var q_dt = next_q
        if use_prof and q_dt + 1 < q_limit:
            self.vk.write_timestamp(cur_slot, q_dt, PIPELINE_BOTTOM)
            next_q += 2
            prof_pairs.append((String("CinemaDisplayTonemap"), q_dt))
        var disp_tonemap_mode = UInt32(2) if self.display_mode == AAA_DISPLAY_SCRGB_FP16 else self.display_mode
        # --- sota:camera hook (HDR display mapping after AgX when the output is HDR10 / scRGB) ---
        disp_tonemap_mode = sota_hdr_tonemap_mode(self.sota_cam.hdr, self.display_mode, disp_tonemap_mode)
        # --- end sota:camera hook ---
        var push_dt = List[UInt32](length=16, fill=0)
        push_dt[0] = UInt32(self.output_w)
        push_dt[1] = UInt32(self.output_h)
        push_dt[2] = UInt32(out_ldr_off)
        push_dt[3] = disp_tonemap_mode
        push_dt[4] = UInt32(1) if has_saved_bloom_mip else UInt32(0)
        push_dt[5] = UInt32(saved_bm_w)
        push_dt[6] = UInt32(saved_bm_h)
        push_dt[7] = UInt32(safe_bloom_mip_off)
        push_dt[8] = bitcast[DType.uint32](self.grade_saturation - Float32(1.0))
        push_dt[9] = bitcast[DType.uint32](self.grade_contrast - Float32(1.0))
        sota_hdr_tonemap_push(self.sota_cam.hdr, push_dt)  # --- sota:camera hook (Push[10..12]) ---
        self.vk.dispatch_aaa(
            cur_slot,
            self.pipe_display_tonemap,
            self.pipe_layout,
            self.desc_set,
            push_dt,
            (self.output_w + 7) // 8,
            (self.output_h + 7) // 8,
        )
        if use_prof and q_dt + 1 < q_limit:
            self.vk.write_timestamp(cur_slot, q_dt + 1, PIPELINE_BOTTOM)
        # --- sota:camera hook (compute VRS rate map from the display words, before the UI overlay) ---
        if not self.furnace_white_dome:
            sota_camera_record_vrs_rate(
                self.sota_cam, self.vk, cur_slot, self.pipe_layout, self.desc_set, self.output_w, self.output_h,
                self.render_w, self.render_h, max_pix, use_prof, next_q, q_limit, prof_pairs,
                out_ldr_off, sota_hdr_ui_mode(disp_tonemap_mode) == UInt32(1),
            )
        # --- end sota:camera hook ---

        # Feature 15: Tile-Binned GPU UiSdfOverlayPass (<0.04 ms GPU UI rendering)
        if self.ui_draw_list.count() > 0:
            var q_ui = next_q
            if use_prof and q_ui + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_ui, PIPELINE_BOTTOM)
                next_q += 2
                prof_pairs.append((String("UiSdfOverlay"), q_ui))

            # Sync dirty font atlas if modified
            if self.ui_font.is_dirty() and self.ui_font_img.image != 0:
                var atlas_bytes = len(self.ui_font.atlas)
                self.ensure_staging_capacity(atlas_bytes)
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address, Int(self.ui_font.atlas.unsafe_ptr()), atlas_bytes)
                self.vk.transition_image(cur_slot, self.ui_font_img, LAYOUT_TRANSFER_TARGET, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
                self.vk.upload_image_subregion(cur_slot, self.ui_font_img, self.buf_staging, 0, 0, 0, self.ui_font.atlas_width, self.ui_font.atlas_height)
                self.vk.transition_image(cur_slot, self.ui_font_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
                self.ui_font.mark_clean()

            # Upload GpuUiPrim structs into buf_post_scratch at AAA_POST_UI_PRIMS_BASE_OFFSET
            var prims_word_offset = self.ui_prims_word_offset(cur_slot)
            var max_prims_fit = AAA_UI_PRIM_REGION_WORDS // 16
            var prim_cnt = min(self.ui_draw_list.count(), min(32768, max_prims_fit))
            var prim_bytes = prim_cnt * 64
            var prim_dst = self.buf_post_scratch.cpu_address + prims_word_offset * 4
            if prim_bytes > 0:
                _ = external_call["memcpy", Int](prim_dst, Int(self.ui_draw_list.prims.unsafe_ptr()), prim_bytes)

            self.vk.memory_barrier2(
                cur_slot,
                PIPELINE_COMPUTE,
                Int(ACCESS_SHADER_WRITE),
                PIPELINE_COMPUTE,
                Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE),
            )

            var push_ui = List[UInt32](length=16, fill=0)
            push_ui[0] = UInt32(self.output_w)
            push_ui[1] = UInt32(self.output_h)
            push_ui[2] = UInt32(prim_cnt)
            push_ui[3] = UInt32(prims_word_offset)
            push_ui[4] = UInt32(out_ldr_off)
            push_ui[5] = sota_hdr_ui_mode(disp_tonemap_mode)  # --- sota:camera hook (mapped HDR composes in PQ) ---
            push_ui[6] = sota_hdr_ui_paper_push(self.sota_cam.hdr, disp_tonemap_mode)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_ui_overlay,
                self.pipe_layout,
                self.desc_set,
                push_ui,
                (self.output_w + 15) // 16,
                (self.output_h + 15) // 16,
            )
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE | PIPELINE_TRANSFER, Int(ACCESS_SHADER_READ | ACCESS_TRANSFER_READ)
            )
            if use_prof and q_ui + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_ui + 1, PIPELINE_BOTTOM)

        # Extended Linear scRGB `R16G16B16A16_SFLOAT` (`8` bytes/pixel `RGBA16F`, `1.0 = 80 nits`) Output Pack!
        var scrgb_swapchain = self.vk.swapchain != 0 and self.vk.swapchain_format == FORMAT_R16G16B16A16_SFLOAT
        var want_scrgb_fp16 = (self.display_mode == AAA_DISPLAY_SCRGB_FP16) or scrgb_swapchain
        # --- sota:camera hook (mapped HDR: PQ word incl. UI -> scRGB fp16; replaces the raw linear pack) ---
        if want_scrgb_fp16 and disp_tonemap_mode == SOTA_DISPLAY_AGX_HDR and self.sota_cam.pipe_hdr_scrgb != 0:
            sota_hdr_record_scrgb(
                self.vk, cur_slot, self.sota_cam.pipe_hdr_scrgb, self.pipe_layout, self.desc_set,
                self.output_w, self.output_h, out_ldr_off, AAA_POST_BLOOM_BASE,
            )
            want_scrgb_fp16 = False
        # --- end sota:camera hook ---
        if want_scrgb_fp16:
            var push_scrgb = List[UInt32](length=16, fill=0)
            push_scrgb[0] = UInt32(self.output_w)
            push_scrgb[1] = UInt32(self.output_h)
            push_scrgb[2] = UInt32(AAA_POST_BLOOM_BASE)
            push_scrgb[3] = bitcast[DType.uint32](self.exposure * Float32(203.0 / 80.0))
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_scrgb_pack,
                self.pipe_layout,
                self.desc_set,
                push_scrgb,
                (self.output_w + 7) // 8,
                (self.output_h + 7) // 8,
            )
            self.vk.memory_barrier2(
                cur_slot,
                PIPELINE_COMPUTE,
                Int(ACCESS_SHADER_WRITE),
                PIPELINE_TRANSFER | PIPELINE_HOST,
                Int(ACCESS_TRANSFER_READ | ACCESS_HOST_READ),
            )

        # Feature 13 & 14: Presentation & Ring Synchronization (skip PCIe DMA readback on live swapchain unless forced!)
        var do_readback = capture_readback or force_readback or self.force_readback or (self.vk.swapchain == 0)
        if do_readback:
            # DMA-copy final LDR frame + adapted exposure from `DEVICE_LOCAL` VRAM into `HOST_CACHED` readback memory
            # (`buf_ldr_readback`, or this slot's ring buffer in deferred-readback mode).
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ)
            )
            self.vk.copy_buffer(cur_slot, self.buf_post_scratch, AAA_POST_ADAPTED_EXP_WORD * 4, rb_buf, 4, 8)
            self.vk.copy_buffer(
                cur_slot,
                self.buf_history,
                out_ldr_off * 4,
                rb_buf,
                64,
                self.output_w * self.output_h * 4,
            )
            self.vk.memory_barrier2(cur_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
        if frame_q >= 0:
            self.vk.write_timestamp(cur_slot, frame_q + 1, PIPELINE_BOTTOM)

        var t_submit0 = get_time()
        self.last_cpu_record_ms = (t_submit0 - t_record0) * 1000.0
        if self.vk.swapchain != 0:
            _ = sota_hdr_update_metadata(self.sota_cam.hdr, self.vk)  # --- sota:camera hook (VK_EXT_hdr_metadata) ---
            var img_idx = self.vk.acquire_image(cur_slot)
            if img_idx != NO_IMAGE:
                var copy_w = min(self.output_w, self.vk.swapchain_width)
                var copy_h = min(self.output_h, self.vk.swapchain_height)
                if copy_w > 0 and copy_h > 0:
                    if scrgb_swapchain:
                        # Copy 8-byte/pixel `R16G16B16A16_SFLOAT` Extended Linear scRGB buffer (NEVER 4-byte PQ words!)
                        self.vk.copy_to_image(
                            cur_slot,
                            self.buf_post_scratch,
                            self.output_w,
                            img_idx,
                            UInt64(AAA_POST_BLOOM_BASE * 4),
                            copy_w,
                            copy_h,
                        )
                    else:
                        self.vk.copy_to_image(
                            cur_slot,
                            self.buf_history,
                            self.output_w,
                            img_idx,
                            UInt64(out_ldr_off * 4),
                            copy_w,
                            copy_h,
                        )
                var f = self.vk.submit_and_present(cur_slot, img_idx)
                self.frame_fences[cur_slot] = f
                if do_readback and f != 0:
                    _ = self.vk.wait(f)
                    self.frame_fences[cur_slot] = 0
            else:
                var f = self.vk.submit(cur_slot)
                self.frame_fences[cur_slot] = f
                if do_readback and f != 0:
                    _ = self.vk.wait(f)
                    self.frame_fences[cur_slot] = 0
        else:
            var f = self.vk.submit(cur_slot)
            self.frame_fences[cur_slot] = f
            if defer_this and f != 0:
                # Phase A: do not wait; the next `render_frame()` (or `flush_deferred_frame()`) retires it.
                self.pending_slot = cur_slot
                self.pending_prof_pairs = prof_pairs.copy()
                self.pending_frame_q = frame_q
                self.pending_n_inst = n_inst
            elif do_readback and f != 0:
                _ = self.vk.wait(f)
                self.frame_fences[cur_slot] = 0
        self.last_cpu_submit_wait_ms = (get_time() - t_submit0) * 1000.0

        if self.pending_slot < 0:
            if single_cmd_buf and do_readback:
                self.last_culled_meshlet_draws = Int(get32(self.buf_ldr_readback.cpu_address)) if n_inst > 0 else 0
                self.readback_ready_slot = 0

            # Read hardware `VkQueryPool` 64-bit GPU timestamps and convert delta ticks -> nanoseconds/milliseconds!
            self._read_gpu_timings(prof_pairs, frame_q)

        self.prev_view_proj = vp_m^
        self.prev_cam_pos = self.cam_pos
        self.has_prev_cam = True
        self.scene_time_sec += Float32(1.0 / 60.0)
        self.frame_index += 1
        return True

    # --- sota:gi hook ---
    def _sota_gi_prepare(mut self):
        """Grows the surface-cache buffer to hold the area-B region the first time a mode is enabled."""
        if self.sota_gi_region_ready:
            return
        var wants_restir_di = self.enable_restir_di and self.sota_gi.restir_di_reuse and self.enable_rt_shadows
        if self.sota_gi.mode == SOTA_GI_MODE_LEGACY and not wants_restir_di:
            return
        if self.max_surfcache_tris != SOTA_GI_SURFCACHE_TRIS or self.sota_gi.pipe_wrc == 0:
            return
        var max_pix = max(self.output_w * self.output_h, self.render_w * self.render_h)
        var words = 4096 + self.max_surfcache_tris * 4 + sota_gi_region_words(max_pix)
        var nb = self.vk.create_buffer(words * 4, local=True)
        if nb.cpu_address == 0:
            return
        self.vk.wait_idle()
        _ = external_call["memset", Int](nb.cpu_address, Int32(0), nb.size)
        self.vk.destroy_buffer(self.buf_surfcache)
        self.buf_surfcache = nb^
        for slot in range(FRAMES_IN_FLIGHT):
            self.vk.write_descriptor(self.desc_sets[slot], BIND_AAA_SURFCACHE, self.buf_surfcache)
        self.sota_gi.reset_cache()
        self.sota_gi_region_ready = True
    # --- end sota:gi hook ---

    def _read_gpu_timings(mut self, prof_pairs: List[Tuple[String, Int]], frame_q: Int):
        self.render_graph.pass_timings.clear()
        self.render_graph.total_gpu_ms = 0.0
        var q_buf = List[UInt64](length=2, fill=0)
        if len(prof_pairs) > 0:
            for pp in prof_pairs:
                if self.vk.read_queries(pp[1], 2, Int(q_buf.unsafe_ptr())):
                    var t0_q = q_buf[0]
                    var t1_q = q_buf[1]
                    var delta_ticks = (t1_q - t0_q) if t1_q >= t0_q else UInt64(0)
                    var ns_f = Float64(delta_ticks) * self.vk.timestamp_period
                    var ms_f = ns_f * 1.0e-6
                    self.render_graph.pass_timings.append(AaaGpuPassTiming(pp[0], UInt64(ns_f), ms_f))
                    self.render_graph.total_gpu_ms += ms_f
        if frame_q >= 0:
            q_buf[0] = 0
            q_buf[1] = 0
            if self.vk.read_queries(frame_q, 2, Int(q_buf.unsafe_ptr())) and q_buf[1] >= q_buf[0]:
                self.last_frame_gpu_ms = Float64(q_buf[1] - q_buf[0]) * self.vk.timestamp_period * 1.0e-6
        _ = len(q_buf)

    def _readback_buffer(self, slot: Int) -> GpuBuffer:
        """Readback buffer for ring `slot` (slot 0 and the synchronous path use `buf_ldr_readback`)."""
        if slot <= 0 or slot - 1 >= len(self.readback_extra):
            return self.buf_ldr_readback.copy()
        return self.readback_extra[slot - 1].copy()

    def _readback_address(self) -> Int:
        """`cpu_address` of the readback buffer holding the latest completed frame."""
        if self.readback_ready_slot <= 0 or self.readback_ready_slot - 1 >= len(self.readback_extra):
            return self.buf_ldr_readback.cpu_address
        return self.readback_extra[self.readback_ready_slot - 1].cpu_address

    def _retire_pending_frame(mut self):
        """Deferred readback: waits on the in-flight frame's fence, then publishes its readback buffer,
        meshlet counter and GPU timestamps. No-op when nothing is pending (default synchronous mode)."""
        if self.pending_slot < 0:
            return
        var slot = self.pending_slot
        self.pending_slot = -1
        if slot < len(self.frame_fences) and self.frame_fences[slot] != 0:
            _ = self.vk.wait(self.frame_fences[slot])
            self.frame_fences[slot] = 0
        self.readback_ready_slot = slot
        self.last_culled_meshlet_draws = Int(get32(self._readback_address())) if self.pending_n_inst > 0 else 0
        var pairs = self.pending_prof_pairs^
        self.pending_prof_pairs = List[Tuple[String, Int]]()
        self._read_gpu_timings(pairs, self.pending_frame_q)
        self.pending_frame_q = -1

    def flush_deferred_frame(mut self):
        """Deferred readback: block until the last submitted frame is done so `readback_pixels_into()`
        returns it (e.g. before a screenshot or when leaving deferred mode)."""
        self._retire_pending_frame()

    def get_adapted_exposure(self) -> Tuple[Float32, Float32]:
        """Returns `(adapted_exposure, avg_scene_luminance)` computed on GPU by `aaa_adapt_exposure_shader`."""
        var exp_u = get32(self._readback_address() + 4)
        var lum_u = get32(self._readback_address() + 8)
        return (bitcast[DType.float32](exp_u), bitcast[DType.float32](lum_u))

    def get_gpu_pass_timings(self) -> List[AaaGpuPassTiming]:
        return self.render_graph.pass_timings.copy()

    def readback_pixels_into(self, mut dst: List[UInt32]):
        """Zero-allocation `HOST_CACHED` frame readback straight into `dst` (`SimdCanvas.pixels`)!"""
        var n = self.output_w * self.output_h
        if len(dst) < n:
            dst.resize(n, 0)
        _ = external_call[
            "memcpy", Int
        ](Int(dst.unsafe_ptr()), self._readback_address() + 64, n * 4)

    def readback_pixels(self) -> List[UInt32]:
        var n = self.output_w * self.output_h
        var out = List[UInt32](length=n, fill=0)
        self.readback_pixels_into(out)
        return out^

    def readback_resolved_hdr(self) -> List[Vec3]:
        """Reads back linear HDR `(R, G, B)` resolved by `FlintTSR v2` before display OETF."""
        var n = self.output_w * self.output_h
        var max_pix = max(n, self.render_w * self.render_h)
        var last_written_hist = (max_pix * 20) if ((self.frame_index - 1) & 1) == 0 else (max_pix * 16)
        var raw_f = List[Float32](length=n * 4, fill=0.0)
        _ = external_call[
            "memcpy", Int
        ](Int(raw_f.unsafe_ptr()), self.buf_history.cpu_address + last_written_hist * 4, n * 16)
        var out = List[Vec3](length=n, fill=Vec3())
        for i in range(n):
            out[i] = Vec3(raw_f[i * 4 + 0], raw_f[i * 4 + 1], raw_f[i * 4 + 2])
        _ = len(raw_f)
        return out^

    def readback_scrgb_rgba16f_words(self) -> List[UInt32]:
        """Reads back `8`-byte/pixel `R16G16B16A16_SFLOAT` (`2 x UInt32` per pixel) Extended Linear scRGB words (`1.0 = 80 nits`)."""
        var n_words = self.output_w * self.output_h * 2
        var out = List[UInt32](length=n_words, fill=0)
        if self.buf_post_scratch.cpu_address != 0 and (AAA_POST_BLOOM_BASE + n_words) * 4 <= self.buf_post_scratch.size:
            _ = external_call[
                "memcpy", Int
            ](Int(out.unsafe_ptr()), self.buf_post_scratch.cpu_address + AAA_POST_BLOOM_BASE * 4, n_words * 4)
        return out^

    def shutdown(mut self):
        if not self.active and self.vk.instance == 0:
            self.vk._free_host_scratch()
            return
        if self.vk.device == 0:
            self.vk.close()
            self.active = False
            return
        self.vk.wait_idle()
        for i in range(len(self.mesh_blas)):
            self.vk.destroy_accel_struct(self.mesh_blas[i])
        self.mesh_blas.clear()
        for i in range(len(self.ring_tlas)):
            if self.ring_tlas[i].handle != 0:
                self.vk.destroy_accel_struct(self.ring_tlas[i])
        self.ring_tlas.clear()
        self.frame_tlas = VulkanAccelStruct()

        for i in range(len(self.ring_as_instances)):
            if self.ring_as_instances[i].size > 0:
                self.vk.destroy_buffer(self.ring_as_instances[i])
        self.ring_as_instances.clear()
        self.buf_as_instances = GpuBuffer()

        if self.buf_as_scratch.size > 0:
            self.vk.destroy_buffer(self.buf_as_scratch)
            self.buf_as_scratch = GpuBuffer()

        self.ui_font_img = VulkanImage()
        for i in range(len(self.bindless_images)):
            self.vk.destroy_image(self.bindless_images[i])
        self.bindless_images.clear()
        self.vk.destroy_image(self.vis_img)
        self.vk.destroy_image(self.depth_img)
        self.vk.destroy_image(self.vsm_img)
        self.vk.destroy_image(self.brdf_lut_img)
        self.vk.destroy_image(self.ltc_lut_img)
        self.vk.destroy_image(self.env_cube_img)

        self.vk.destroy_sampler(self.samp_aniso16)
        self.vk.destroy_sampler(self.samp_linear_clamp)
        self.vk.destroy_sampler(self.samp_nearest)

        self.vk.destroy_pipeline(self.pipe_meshlet_cull)
        self.sota_e_gpu.shutdown(self.vk)  # sota:E hook
        self.sota_e_hair.shutdown(self.vk)  # sota:E hook
        self.sota_geom.shutdown(self.vk)  # sota:geometry hook
        if self.sota_upload_pool_started and self.sota_upload_pool.ready():  # --- sota:perf hook ---
            self.sota_upload_pool.stop()
        self.sota_upload_pool_started = False  # --- sota:perf hook ---
        self.sota_vt.shutdown(self.vk)  # sota:geometry hook
        self.vk.destroy_pipeline(self.pipe_visbuffer_gfx)
        self.vk.destroy_pipeline(self.pipe_visbuffer_alphamask_gfx)
        if self.pipe_npr_outline_gfx != 0:
            self.vk.destroy_pipeline(self.pipe_npr_outline_gfx)
            self.pipe_npr_outline_gfx = 0
        self.vk.destroy_pipeline(self.pipe_vsm_mark)
        self.sota_shadow_gpu.destroy(self.vk)  # --- sota:shadow hook ---
        self.vk.destroy_pipeline(self.pipe_vsm_gfx)
        self.vk.destroy_pipeline(self.pipe_vsm_alphamask_gfx)
        self.vk.destroy_pipeline(self.pipe_surfcache)
        self.vk.destroy_pipeline(self.pipe_cluster_cull)
        self.vk.destroy_pipeline(self.pipe_dq_skinning)
        self.vk.destroy_pipeline(self.pipe_deferred_shade)
        self.vk.destroy_pipeline(self.pipe_svgf_reproject)
        self.sota_ml.destroy(self.vk)  # sota:ml hook
        self.sota_ml_up.destroy(self.vk)  # sota:ml hook
        self.sota_atmo.destroy_pipelines(self.vk)  # sota:atmo hook
        self.vk.destroy_pipeline(self.pipe_svgf_atrous)
        if self.sota_gi.pipe_wrc != 0:  # --- sota:gi hook ---
            self.vk.destroy_pipeline(self.sota_gi.pipe_wrc)
            self.sota_gi.pipe_wrc = 0
        if self.sota_gi.pipe_resolve != 0:  # --- sota:gi hook ---
            self.vk.destroy_pipeline(self.sota_gi.pipe_resolve)
            self.sota_gi.pipe_resolve = 0
        if self.sota_gi.pipe_upsample != 0:  # --- sota:gi hook --- (round 3 upsample)
            self.vk.destroy_pipeline(self.sota_gi.pipe_upsample)
            self.sota_gi.pipe_upsample = 0
        self.vk.destroy_pipeline(self.pipe_oit_vol)
        self.vk.destroy_pipeline(self.pipe_luma_hist)
        self.vk.destroy_pipeline(self.pipe_adapt_exp)
        self.vk.destroy_pipeline(self.pipe_bloom_down)
        self.vk.destroy_pipeline(self.pipe_bloom_up)
        self.vk.destroy_pipeline(self.pipe_bokeh_dof)
        self.vk.destroy_pipeline(self.pipe_tsr_display)
        self.vk.destroy_pipeline(self.pipe_spec_reblur_temp)
        self.vk.destroy_pipeline(self.pipe_spec_reblur_atrous)
        self.vk.destroy_pipeline(self.pipe_froxel_inject)
        self.vk.destroy_pipeline(self.pipe_froxel_prefix)
        self.vk.destroy_pipeline(self.pipe_froxel_march)
        self.vk.destroy_pipeline(self.pipe_particle_clear)
        self.vk.destroy_pipeline(self.pipe_particle_sim)
        self.vk.destroy_pipeline(self.pipe_particle_comp)
        self.vk.destroy_pipeline(self.pipe_water_refract)
        if self.pipe_rt_shadow_trace != 0:
            self.vk.destroy_pipeline(self.pipe_rt_shadow_trace)
        if self.pipe_rt_shadow_filter != 0:
            self.vk.destroy_pipeline(self.pipe_rt_shadow_filter)
        if self.pipe_glass_refract != 0:
            self.vk.destroy_pipeline(self.pipe_glass_refract)
        self.vk.destroy_pipeline(self.pipe_tsr_v3_resolve)
        self.vk.destroy_pipeline(self.pipe_motion_blur)
        self.vk.destroy_pipeline(self.pipe_display_tonemap)
        if self.pipe_npr_cel_shade != 0:
            self.vk.destroy_pipeline(self.pipe_npr_cel_shade)
            self.pipe_npr_cel_shade = 0
        if self.pipe_npr_post_fx != 0:
            self.vk.destroy_pipeline(self.pipe_npr_post_fx)
            self.pipe_npr_post_fx = 0
        if self.pipe_ui_overlay != 0:
            self.vk.destroy_pipeline(self.pipe_ui_overlay)
            self.pipe_ui_overlay = 0
        if self.pipe_scrgb_pack != 0:
            self.vk.destroy_pipeline(self.pipe_scrgb_pack)
            self.pipe_scrgb_pack = 0
        self.vk.destroy_aaa_descriptors(self.desc_pool, self.dsl, self.pipe_layout)

        for i in range(len(self.ring_globals)):
            if self.ring_globals[i].size > 0:
                self.vk.destroy_buffer(self.ring_globals[i])
        self.ring_globals.clear()
        self.buf_globals = GpuBuffer()

        for i in range(len(self.ring_instances)):
            if self.ring_instances[i].size > 0:
                self.vk.destroy_buffer(self.ring_instances[i])
        self.ring_instances.clear()
        self.buf_instances = GpuBuffer()

        for i in range(len(self.ring_indirect)):
            if self.ring_indirect[i].size > 0:
                self.vk.destroy_buffer(self.ring_indirect[i])
        self.ring_indirect.clear()
        self.buf_indirect = GpuBuffer()

        for i in range(len(self.ring_alphamask_indirect)):
            if self.ring_alphamask_indirect[i].size > 0:
                self.vk.destroy_buffer(self.ring_alphamask_indirect[i])
        self.ring_alphamask_indirect.clear()
        self.buf_alphamask_indirect = GpuBuffer()

        for i in range(len(self.ring_outline_indirect)):
            if self.ring_outline_indirect[i].size > 0:
                self.vk.destroy_buffer(self.ring_outline_indirect[i])
        self.ring_outline_indirect.clear()
        self.buf_outline_indirect = GpuBuffer()

        for i in range(len(self.ring_clusters)):
            if self.ring_clusters[i].size > 0:
                self.vk.destroy_buffer(self.ring_clusters[i])
        self.ring_clusters.clear()
        self.buf_clusters = GpuBuffer()

        self.vk.destroy_buffer(self.buf_vertices)
        self.vk.destroy_buffer(self.buf_indices)
        self.vk.destroy_buffer(self.buf_meshlets)
        self.vk.destroy_buffer(self.buf_instances)
        self.vk.destroy_buffer(self.buf_materials)
        self.vk.destroy_buffer(self.buf_indirect)
        self.vk.destroy_buffer(self.buf_vsm_indirect)
        self.vk.destroy_buffer(self.buf_globals)
        self.vk.destroy_buffer(self.buf_hdr_out)
        self.vk.destroy_buffer(self.buf_history)
        self.vk.destroy_buffer(self.buf_vsm_table)
        self.vk.destroy_buffer(self.buf_surfcache)
        self.vk.destroy_buffer(self.buf_oit_nodes)
        self.vk.destroy_buffer(self.buf_clusters)
        self.vk.destroy_buffer(self.buf_skinning)
        self.vk.destroy_buffer(self.buf_post_scratch)
        self.vk.destroy_buffer(self.buf_ldr_readback)
        for i in range(len(self.readback_extra)):
            self.vk.destroy_buffer(self.readback_extra[i])
        self.readback_extra.clear()
        self.pending_slot = -1
        self.readback_ready_slot = 0
        self.vk.destroy_buffer(self.buf_staging)
        self.vk.destroy_buffer(self.buf_indirect_phase2)
        self.vk.destroy_buffer(self.buf_occluded_candidates)

        # --- sota:E hook ---
        if self.vk.pipeline_cache != 0:
            _ = sota_e_pso_cache_save(self.vk, self.sota_e_pso)
        # --- end sota:E hook ---
        self.vk.close()
        self.active = False

    def __deinit__(deinit self):
        self.shutdown()
