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
from flint.assets.meshlet import AAA_VERTEX_BYTES, AAA_MESHLET_WORDS, AaaGeometryPool, CookedMeshEntry
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
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_MAX_BINDLESS,
    AAA_PUSH_BYTES,
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
    visbuffer_alphamask_fragment_shader,
    visbuffer_alphamask_vertex_shader,
    visbuffer_fragment_shader,
    visbuffer_vertex_shader,
)
from flint.render.aaa_bsdf import AAA_MATERIAL_WORDS, AaaMaterial
from flint.render.aaa_vsm import (
    VSM_PAGE_TEXELS,
    VSM_PHYS_ATLAS_RES,
    VSM_TABLE_WORDS,
    VsmPageManager,
    make_ortho_directional_light_vp,
    vsm_alphamask_fragment_shader,
    vsm_alphamask_vertex_shader,
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
    build_blas_triangles_ex,
    ensure_tlas_capacity_mixed_opaque,
    record_build_tlas_mixed_opaque,
    write_vk_as_instance_64b,
)
from flint.render.aaa_gi_svgf import (
    AAA_FLAG_AERIAL_PERSPECTIVE,
    AAA_FLAG_CLUSTERED_LIGHTS,
    AAA_FLAG_FURNACE_WHITE_DOME,
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
    AAA_SKIN_MAX_VERTS,
    AAA_SKIN_PREV_BONES_BASE,
    AAA_SKIN_PREV_POS_BASE,
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
from flint.render.aaa_water import aaa_water_refract_shader
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


struct AaaDrawInstance(Copyable, Movable):
    var mesh_id: Int
    var world_mat: Mat4
    var prev_world_mat: Mat4
    var material_override: Int
    var uniform_scale: Float32

    def __init__(
        out self,
        mesh_id: Int,
        world_mat: Mat4,
        prev_world_mat: Mat4 = Mat4(),
        material_override: Int = -1,
        uniform_scale: Float32 = 1.0,
        has_prev: Bool = False,
    ):
        self.mesh_id = mesh_id
        self.world_mat = world_mat.copy()
        self.prev_world_mat = prev_world_mat.copy() if has_prev else world_mat.copy()
        self.material_override = material_override
        self.uniform_scale = max(Float32(0.001), uniform_scale)


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

    def __init__(out self):
        self.mesh_entry = CookedMeshEntry()
        self.skin_bind_first_slot = 0
        self.vertex_count = 0
        self.bone_count = 0
        self.bone_palette_offset = 0


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
comptime AAA_UI_PRIM_REGION_WORDS: Int = 4096 * 16  # 4096 GpuUiPrim (64 bytes) per frame in flight
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
    var last_frame_gpu_ms: Float64  # whole-frame TOP->BOTTOM timestamp pair (incl. TLAS, skin, readback copy)
    var last_vsm_dirty_pages: Int
    var last_cpu_vsm_table_ms: Float64  # part of `last_cpu_vsm_ms`: VSM table alloc + host read/write copies
    var last_cpu_vsm_requests_ms: Float64  # part of `last_cpu_vsm_ms`: `vsm.process_requests`

    # Opt-in deferred readback (default False = unchanged synchronous behaviour). When True and no swapchain
    # is attached, `render_frame()` submits without waiting; the next `render_frame()` waits on that fence,
    # so `readback_pixels_into()` returns the previous frame. `capture_readback=True` stays synchronous.
    var deferred_readback: Bool
    var readback_extra: List[GpuBuffer]  # per-slot readback buffers for slots >= 1 (slot 0 = `buf_ldr_readback`)
    var readback_ready_slot: Int  # slot whose readback buffer holds the latest completed frame
    var pending_slot: Int  # slot of a submitted-but-not-yet-waited deferred frame (-1 = none)
    var pending_prof_pairs: List[Tuple[String, Int]]
    var pending_frame_q: Int
    var pending_n_inst: Int

    # UI Overlay Pass (Feature 15)
    var pipe_ui_overlay: Int
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
    var pipe_tsr_v3_resolve: Int
    var pipe_motion_blur: Int
    var pipe_display_tonemap: Int

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
    var buf_occluded_candidates: GpuBuffer
    var pending_skin_updates: List[AaaSkinnedMeshEntry]

    # CPU Scene & Subsystems
    var geom: AaaGeometryPool
    var materials: List[AaaMaterial]
    var frame_instances: List[AaaDrawInstance]
    var frame_plumes: List[AaaVolumetricPlume]
    var frame_particle_emitters: List[AaaParticleEmitter]
    var particle_ring_cursor: Int
    var particles_ever_active: Bool
    var frame_cluster_lights: List[AaaClusterLight]
    var cluster_far_z: Float32
    var skinned_mesh_flags: List[Bool]
    var skinned_mesh_bind_slots: List[Int]
    var mesh_double_sided_flags: List[Bool]
    var mesh_alpha_masked_flags: List[Bool]
    var skin_bind_cursor: Int
    var skin_bone_cursor: Int
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
    var vsm_scene_center: Vec3
    var vsm_ortho_half_extent: Float32
    var ibl_intensity: Float32
    var exposure: Float32
    var enable_cone_cull: Bool
    var enable_hiz_cull: Bool
    var enable_vsm: Bool
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
    var camera_lens: AaaPhysicalCamera
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
    var frame_index: Int
    var last_culled_meshlet_draws: Int
    var prev_cam_pos: Vec3
    var has_prev_cam: Bool
    var scene_time_sec: Float32
    var enable_spec_reblur: Bool
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
    ):
        self.vk = VulkanGpu()
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
        self.pipe_tsr_v3_resolve = 0
        self.pipe_motion_blur = 0
        self.pipe_display_tonemap = 0

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
        self.buf_occluded_candidates = GpuBuffer()

        self.ring_globals = List[GpuBuffer]()
        self.ring_instances = List[GpuBuffer]()
        self.ring_as_instances = List[GpuBuffer]()
        self.ring_indirect = List[GpuBuffer]()
        self.ring_alphamask_indirect = List[GpuBuffer]()
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
        self.deferred_readback = False
        self.readback_extra = List[GpuBuffer]()
        self.readback_ready_slot = 0
        self.pending_slot = -1
        self.pending_prof_pairs = List[Tuple[String, Int]]()
        self.pending_frame_q = -1
        self.pending_n_inst = 0

        self.pipe_ui_overlay = 0
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
        self.frame_cluster_lights = List[AaaClusterLight]()
        self.cluster_far_z = Float32(250.0)
        self.skinned_mesh_flags = List[Bool]()
        self.skinned_mesh_bind_slots = List[Int]()
        self.mesh_double_sided_flags = List[Bool]()
        self.mesh_alpha_masked_flags = List[Bool]()
        self.skin_bind_cursor = 0
        self.skin_bone_cursor = 0
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
        self.vsm_scene_center = Vec3(0.0, 0.0, 0.0)
        self.vsm_ortho_half_extent = Float32(16.0)
        self.ibl_intensity = Float32(1.0)
        self.exposure = Float32(1.0)
        self.enable_cone_cull = True
        self.enable_hiz_cull = True
        self.enable_vsm = True
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
        self.camera_lens = AaaPhysicalCamera()
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
        self.frame_index = 0
        self.last_culled_meshlet_draws = 0
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

        _ = self.vk.init_pipeline_cache("build/vk_pso.cache")

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
            visbuffer_alphamask_fragment_shader(),
            has_color=True,
            color_format=FORMAT_R32G32_UINT,
            depth_format=FORMAT_D32_SFLOAT,
            depth_test=True,
            depth_write=True,
            depth_compare=COMPARE_GREATER_OR_EQUAL,
            cull_mode=CULL_MODE_NONE,
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
        self.pipe_tsr_v3_resolve = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_tsr_v3_resolve_shader())
        self.pipe_motion_blur = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_motion_blur_shader())
        self.pipe_display_tonemap = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_display_tonemap_shader())
        self.pipe_ui_overlay = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_ui_overlay_shader())

        if (
            self.pipe_meshlet_cull == 0
            or self.pipe_visbuffer_gfx == 0
            or self.pipe_visbuffer_alphamask_gfx == 0
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
            or self.pipe_ui_overlay == 0
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
        self.buf_vertices = self.vk.create_buffer_bda(32 * 1024 * 1024, extra_usage=bda_flags, host_visible=True)
        self.buf_indices = self.vk.create_buffer_bda(32 * 1024 * 1024, extra_usage=bda_flags, host_visible=True)
        self.buf_meshlets = self.vk.create_buffer(8 * 1024 * 1024, local=True)
        self.buf_materials = self.vk.create_buffer(1024 * AAA_MATERIAL_WORDS * 4, local=True)
        self.buf_vsm_indirect = self.vk.create_buffer(64 + self.max_draws * 16, local=True)
        self.buf_hdr_out = self.vk.create_buffer(max_pix * 8 * 4, local=True)
        var hist_words = max_pix * 44 + AAA_FROXEL_GRID_WORDS * 3
        self.buf_history = self.vk.create_buffer(hist_words * 4, local=True)
        self.buf_vsm_table = self.vk.create_buffer(VSM_TABLE_WORDS * 4, local=False, cpu_reads=True)
        self.buf_surfcache = self.vk.create_buffer((4096 + self.max_surfcache_tris * 4) * 4, local=True)
        self.buf_oit_nodes = self.vk.create_buffer(gpu_particles_total_oit_words(max_pix) * 4, local=True)
        self.buf_skinning = self.vk.create_buffer(AAA_SKIN_BUFFER_WORDS * 4, local=True)
        # Per-pixel post regions end below word 1024 + max_pix * 10; the UI prims of each frame in flight follow.
        var scratch_words = AAA_POST_UI_PRIMS_BASE_OFFSET + max_pix * 10 + FRAMES_IN_FLIGHT * AAA_UI_PRIM_REGION_WORDS
        self.buf_post_scratch = self.vk.create_buffer(scratch_words * 4, local=True)
        self.buf_ldr_readback = self.vk.create_buffer(64 + self.output_w * self.output_h * 4, local=False, cpu_reads=True)
        self.buf_staging = self.vk.create_buffer(8 * 1024 * 1024, local=False)
        self.buf_indirect_phase2 = self.vk.create_buffer(64 + self.max_draws * 20, local=True)
        self.buf_occluded_candidates = self.vk.create_buffer(64 + self.max_draws * 4, local=True)

        # Ring-Buffered CPU-to-GPU Dynamic Buffers (Feature 14: FRAMES_IN_FLIGHT = 2)
        for _ in range(FRAMES_IN_FLIGHT):
            var b_glob = self.vk.create_buffer(4096, local=True)
            var b_inst = self.vk.create_buffer(self.max_instances * AAA_INSTANCE_WORDS * 4, local=True)
            var b_ind = self.vk.create_buffer(64 + self.max_draws * 20, local=True)
            var b_am_ind = self.vk.create_buffer(64 + 16384 * 20, local=True)
            var b_clust = self.vk.create_buffer(AAA_CLUSTER_BUFFER_WORDS * 4, local=True)
            _ = external_call["memset", Int](b_clust.cpu_address, Int32(0), b_clust.size)
            self.ring_globals.append(b_glob^)
            self.ring_instances.append(b_inst^)
            self.ring_indirect.append(b_ind^)
            self.ring_alphamask_indirect.append(b_am_ind^)
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
        var max_mats = self.buf_materials.size // (AAA_MATERIAL_WORDS * 4)
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

    def add_mesh(mut self, mesh: NativeMesh, material_id: Int, build_lods: Bool = False) -> CookedMeshEntry:
        """Partitions `mesh` into 64v/124t GPU Meshlets (and optional `QEM` `LOD 0..3` DAG), uploads SSBOs, and builds `BLAS`!"""
        var v_words_before = len(self.geom.vertex_words)
        var i_words_before = len(self.geom.index_words)
        var m_words_before = len(self.geom.meshlet_words)

        var entry = self.geom.append_mesh(mesh, material_id, build_lods=build_lods)

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
        return res^

    def _record_skin_dispatch_and_blas_refit(mut self, frame: Int, skin_entry: AaaSkinnedMeshEntry):
        var push_sk = List[UInt32](length=16, fill=0)
        push_sk[0] = UInt32(skin_entry.vertex_count)
        push_sk[1] = UInt32(skin_entry.skin_bind_first_slot)
        push_sk[2] = UInt32(skin_entry.mesh_entry.first_vertex)
        push_sk[3] = UInt32(AAA_SKIN_CUR_BONES_BASE + skin_entry.bone_palette_offset)
        push_sk[4] = UInt32(AAA_SKIN_PREV_BONES_BASE + skin_entry.bone_palette_offset)
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

        if invalidate_vsm:
            self.vsm.invalidate_all()

        if defer_to_frame:
            self.pending_skin_updates.append(skin_entry.copy())
            return 0.0

        var t0 = get_time()
        if not self.vk.begin_commands(0):
            return 0.0
        self._record_skin_dispatch_and_blas_refit(0, skin_entry)
        var f = self.vk.submit(0)
        if f != 0:
            _ = self.vk.wait(f)
        return (get_time() - t0) * 1000.0

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

        for m_idx in range(len(model.meshes)):
            ref m = model.meshes[m_idx]
            var alb_slot = -1
            if m.texture >= 0 and m.texture < n_tex:
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
            var mat_id = self.add_material(mat)
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
        self.buf_clusters = self.ring_clusters[cur_slot].copy()
        self.desc_set = self.desc_sets[cur_slot]
        if self.vk.has_ray_query and len(self.ring_as_instances) > cur_slot:
            self.buf_as_instances = self.ring_as_instances[cur_slot].copy()
            self.frame_tlas = self.ring_tlas[cur_slot].copy()

        self.cam_pos = cam_pos
        var fwd = (cam_target - cam_pos).normalized()
        var up_ref = Vec3(0.0, 0.0, 1.0) if abs(fwd.y) > Float32(0.95) else Vec3(0.0, 1.0, 0.0)
        var r0 = up_ref.cross(fwd).normalized()
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
    ):
        if len(self.frame_instances) < self.max_instances and mesh_id >= 0 and mesh_id < len(self.geom.meshes):
            self.frame_instances.append(
                AaaDrawInstance(mesh_id, world_mat, prev_world_mat, material_override, uniform_scale, has_prev)
            )

    def draw_volumetric_plume(mut self, plume: AaaVolumetricPlume):
        if len(self.frame_plumes) < AAA_MAX_OIT_PRIMS:
            self.frame_plumes.append(plume.copy())

    def submit_particle_emitter(mut self, emitter: AaaParticleEmitter):
        """Queues a GPU compute particle emitter burst/stream (`PARTICLE_SPARK_STREAK`, `WATER_SPRAY`, `DRIFT_SMOKE`, `EXHAUST_PLASMA`)."""
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

            for m_loc in range(ml_cnt):
                if cmd_cnt >= 16384:
                    break
                var ml_id = first_ml + m_loc
                ref ml = self.geom.meshlets[ml_id]
                var dst = base_ptr + 16 + cmd_cnt * 20
                put32(dst + 0, UInt32(ml.tri_count * 3))
                put32(dst + 4, UInt32(1))
                put32(dst + 8, UInt32(ml.first_tri * 3))
                put32(dst + 12, UInt32(0))
                put32(dst + 16, UInt32((i_idx << 17) | (ml_id & 0x1FFFF)))
                cmd_cnt += 1

        put32(base_ptr, UInt32(cmd_cnt))
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

        var page_slot = List[Int](length=16384, fill=-1)
        var min_pvx = 127; var max_pvx = 0
        var min_pvy = 127; var max_pvy = 0
        for p_i in range(n_pages):
            var pvx = vsm_allocs[p_i][0]
            var pvy = vsm_allocs[p_i][1]
            if pvx >= 0 and pvx < 128 and pvy >= 0 and pvy < 128:
                page_slot[pvy * 128 + pvx] = p_i
                if pvx < min_pvx: min_pvx = pvx
                if pvx > max_pvx: max_pvx = pvx
                if pvy < min_pvy: min_pvy = pvy
                if pvy > max_pvy: max_pvy = pvy

        var per_page = List[List[Tuple[UInt32, UInt32]]]()
        var per_page_am = List[List[Tuple[UInt32, UInt32]]]()
        for _ in range(n_pages):
            per_page.append(List[Tuple[UInt32, UInt32]]())
            per_page_am.append(List[Tuple[UInt32, UInt32]]())

        var page_scale = Float32(64.0) / max(Float32(1.0), self.vsm_ortho_half_extent)

        for i_idx in range(n_inst):
            ref di = self.frame_instances[i_idx]
            ref me = self.geom.meshes[di.mesh_id]
            if me.lod0_meshlet_count <= 0:
                continue
            var mat_final_v = me.material_id if di.material_override < 0 else di.material_override
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

            var lx = light_vp[0] * wx + light_vp[4] * wy + light_vp[8] * wz + light_vp[12]
            var ly = light_vp[1] * wx + light_vp[5] * wy + light_vp[9] * wz + light_vp[13]
            var fx = (lx * Float32(0.5) + Float32(0.5)) * Float32(128.0)
            var fy = (-ly * Float32(0.5) + Float32(0.5)) * Float32(128.0)
            var fr = wr * page_scale + Float32(0.35)

            if (
                fx + fr < Float32(min_pvx)
                or fx - fr > Float32(max_pvx + 1)
                or fy + fr < Float32(min_pvy)
                or fy - fr > Float32(max_pvy + 1)
            ):
                continue

            var vx0 = max(min_pvx, Int(fx - fr))
            var vx1 = min(max_pvx, Int(fx + fr))
            var vy0 = max(min_pvy, Int(fy - fr))
            var vy1 = min(max_pvy, Int(fy + fr))
            if vx0 > vx1 or vy0 > vy1:
                continue

            # Select coarse QEM LOD for compact shadow casters (crowds, props, distant cars/trees)
            var lod_idx = 0
            if me.lod_count > 1 and wr < Float32(5.5):
                lod_idx = min(me.lod_count - 1, 2)
            var sl = me.lods[lod_idx]
            var first_ml = sl.first_meshlet if sl.meshlet_count > 0 else me.first_meshlet
            var ml_cnt = sl.meshlet_count if sl.meshlet_count > 0 else me.lod0_meshlet_count

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
                    var v_cnt_u = UInt32(self.geom.meshlets[ml_id].tri_count * 3)
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
                    var mlx = light_vp[0] * mwx + light_vp[4] * mwy + light_vp[8] * mwz + light_vp[12]
                    var mly = light_vp[1] * mwx + light_vp[5] * mwy + light_vp[9] * mwz + light_vp[13]
                    var mfx = (mlx * Float32(0.5) + Float32(0.5)) * Float32(128.0)
                    var mfy = (-mly * Float32(0.5) + Float32(0.5)) * Float32(128.0)
                    var mfr = mwr * page_scale + Float32(0.30)
                    var mvx0 = max(vx0, Int(mfx - mfr))
                    var mvx1 = min(vx1, Int(mfx + mfr))
                    var mvy0 = max(vy0, Int(mfy - mfr))
                    var mvy1 = min(vy1, Int(mfy + mfr))
                    if mvx0 > mvx1 or mvy0 > mvy1:
                        continue
                    var v_cnt_u = UInt32(ml.tri_count * 3)
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

        var cmd_cnt = 0
        var base_ptr = self.buf_vsm_indirect.cpu_address
        for p_i in range(n_pages):
            page_offsets[p_i] = cmd_cnt
            ref plist = per_page[p_i]
            var written = 0
            for cmd in plist:
                if cmd_cnt >= self.max_draws:
                    break
                var dst = base_ptr + cmd_cnt * 16
                put32(dst + 0, cmd[0])
                put32(dst + 4, UInt32(1))
                put32(dst + 8, UInt32(0))
                put32(dst + 12, cmd[1])
                cmd_cnt += 1
                written += 1
            page_counts[p_i] = written

            page_am_offsets[p_i] = cmd_cnt
            ref am_list = per_page_am[p_i]
            var written_am = 0
            for cmd in am_list:
                if cmd_cnt >= self.max_draws:
                    break
                var dst = base_ptr + cmd_cnt * 16
                put32(dst + 0, cmd[0])
                put32(dst + 4, UInt32(1))
                put32(dst + 8, UInt32(0))
                put32(dst + 12, cmd[1])
                cmd_cnt += 1
                written_am += 1
            page_am_counts[p_i] = written_am

        return (cmd_cnt, page_offsets^, page_counts^, page_am_offsets^, page_am_counts^)

    def ui_prims_word_offset(self, slot: Int) -> Int:
        """Where a frame slot's GPU UI prims live in `buf_post_scratch`: past every per-pixel post
        region (RT shade, water and motion blur stage 8 floats per pixel from word 1024, bloom and
        DoF sit below 1024 + max_pix * 10), so no pass of the frame overwrites them before
        `UiSdfOverlayPass` reads them, and one region per frame in flight, so recording the next
        frame never rewrites prims the GPU is still drawing."""
        var max_pix = max(self.output_w * self.output_h, self.render_w * self.render_h)
        return AAA_POST_UI_PRIMS_BASE_OFFSET + max_pix * 10 + (slot % FRAMES_IN_FLIGHT) * AAA_UI_PRIM_REGION_WORDS

    def render_frame(mut self, capture_readback: Bool = False) -> Bool:
        if not self.active:
            return False

        # Deferred readback: retire the previous frame first (no-op in the default synchronous mode). Waiting here,
        # before any host write, keeps every single-instance HOST_VISIBLE buffer race-free (Phase A).
        var t_wait0 = get_time()
        self._retire_pending_frame()
        var t_pack0 = get_time()
        self.last_cpu_prev_wait_ms = (t_pack0 - t_wait0) * 1000.0

        _ = self.streamer.tick_streaming_budget()

        var cur_slot = self.frame_index % FRAMES_IN_FLIGHT
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
            put32(dst + 34 * 4, inst_fl | mat_ov)
            put_float(dst + 35 * 4, di.uniform_scale)
            var sk_slot = self.skinned_mesh_bind_slots[di.mesh_id] if (is_sk and di.mesh_id < len(self.skinned_mesh_bind_slots)) else 0
            put32(dst + 36 * 4, UInt32(sk_slot))
            put32(dst + 37 * 4, UInt32(me.first_vertex))

            var mat_final = me.material_id if di.material_override < 0 else di.material_override
            var is_am_inst = self._is_instance_alpha_masked(di, mat_final)
            var is_ds = (di.mesh_id < len(self.mesh_double_sided_flags) and self.mesh_double_sided_flags[di.mesh_id]) or is_am_inst
            put32(dst + 38 * 4, UInt32(me.first_index))
            put32(
                dst + 39 * 4,
                UInt32(mat_final & 0xFFFF)
                | ((UInt32(1) if is_ds else UInt32(0)) << 16)
                | ((UInt32(1) if is_am_inst else UInt32(0)) << 17),
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

        var light_vp = make_ortho_directional_light_vp(
            self.sun_dir, self.vsm_scene_center, self.vsm_ortho_half_extent
        )

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
        if n_cl_lights > 0 and not self.furnace_white_dome:
            r_flags |= AAA_FLAG_CLUSTERED_LIGHTS
        if self.enable_aerial_perspective and not self.furnace_white_dome:
            r_flags |= AAA_FLAG_AERIAL_PERSPECTIVE
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

        # Pack Bucket 2 alpha-masked VisBuffer draws (zero overhead when no alpha-masked meshes are submitted)
        var am_vis_draws = self._pack_alphamask_visbuffer_commands(n_inst, tan_x, tan_y)

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
        )
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

        # Single-Command-Buffer Mode: asynchronous single-pass execution without CPU stalls (Feature 22)
        var vsm_active = (r_flags & AAA_FLAG_VSM) != 0 and n_inst > 0
        var single_cmd_buf = True

        var t_vsm0 = get_time()
        self.last_cpu_pack_ms = (t_vsm0 - t_pack0) * 1000.0
        var vsm_allocs = List[Tuple[Int, Int, Int, Int]]()
        var vsm_page_offsets = List[Int]()
        var vsm_page_counts = List[Int]()
        var vsm_page_am_offsets = List[Int]()
        var vsm_page_am_counts = List[Int]()
        if single_cmd_buf and vsm_active:
            var tbl = List[UInt32](length=VSM_TABLE_WORDS, fill=0)
            _ = external_call["memcpy", Int](Int(tbl.unsafe_ptr()), self.buf_vsm_table.cpu_address, VSM_TABLE_WORDS * 4)
            var t_vsm_req0 = get_time()
            var pages = self.vsm.process_requests(tbl)
            var t_vsm_req1 = get_time()
            _ = external_call["memcpy", Int](self.buf_vsm_table.cpu_address, Int(tbl.unsafe_ptr()), VSM_TABLE_WORDS * 4)
            _ = len(tbl)
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
            return False
        if use_prof:
            self.vk.reset_queries(cur_slot, q_base, AAA_QUERY_SLOT_STRIDE if single_cmd_buf else 8)
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

        if n_inst > 0:
            var q_mc = next_q
            if use_prof:
                self.vk.write_timestamp(cur_slot, q_mc, PIPELINE_TOP)
                next_q += 2
                prof_pairs.append((String("MeshletCull"), q_mc))
            var push_cull = List[UInt32](length=16, fill=0)
            push_cull[0] = UInt32(n_inst)
            push_cull[1] = UInt32(self.max_draws)
            push_cull[2] = UInt32(1) if self.enable_cone_cull else UInt32(0)
            push_cull[3] = UInt32(1) if (self.enable_hiz_cull and self.frame_index > 0) else UInt32(0)
            push_cull[4] = UInt32(self.render_w)
            push_cull[5] = UInt32(self.render_h)
            push_cull[6] = bitcast[DType.uint32](self.max_lod_error_px)
            push_cull[7] = 0  # Phase 1: cull_phase = 0
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_meshlet_cull,
                self.pipe_layout,
                self.desc_sets_cull_p1[cur_slot],
                push_cull,
                (n_inst + 7) // 8,
                (max_ml_per_inst + 7) // 8,
            )
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
            self.vk.write_timestamp(cur_slot, q_vb, PIPELINE_TOP)
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
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_meshlet_cull,
                self.pipe_layout,
                self.desc_sets_cull_p2[cur_slot],
                push_cull_p2,
                (self.max_draws + 63) // 64,
                1,
            )
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
            self.vk.draw_indexed_indirect(
                cur_slot,
                self.buf_alphamask_indirect,
                16,
                am_vis_draws,
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

        # VSM Mark Visible Virtual Shadow Pages (16k x 16k)
        if vsm_active:
            var q_vm = next_q
            if use_prof:
                self.vk.write_timestamp(cur_slot, q_vm, PIPELINE_TOP)
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

        # Synchronous Frame-0 / Test-Suite VSM Page Readback (skipped in warmed-up single-command-buffer gameplay!)
        if not single_cmd_buf:
            self.vk.memory_barrier2(cur_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
            var f_group_a = self.vk.submit(cur_slot)
            if f_group_a == 0 or not self.vk.wait(f_group_a):
                return False
            self.last_culled_meshlet_draws = Int(get32(self.buf_ldr_readback.cpu_address))
            self.vsm_warmed_up = True

            var tbl = List[UInt32](length=VSM_TABLE_WORDS, fill=0)
            _ = external_call["memcpy", Int](Int(tbl.unsafe_ptr()), self.buf_vsm_table.cpu_address, VSM_TABLE_WORDS * 4)
            var pages = self.vsm.process_requests(tbl)
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
                return False
            next_q = q_base + 8
            if use_prof:
                self.vk.reset_queries(cur_slot, q_base + 8, 24)

        # ====================================================================
        # PASS GROUP B: VSM Indirect Tile Raster + 2-Tier Lumen Surface-Cache +
        #               Deferred Multi-Lobe RT Shade + SVGF + OIT + Post + TSR v2!
        # ====================================================================
        self.last_vsm_dirty_pages = len(vsm_allocs)
        if len(vsm_allocs) > 0:
            var q_vr = next_q
            if use_prof and q_vr + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_vr, PIPELINE_TOP)
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
                var bx_f = Float32(127 - 2 * vx)
                var by_f = -Float32(127 - 2 * vy)
                self.vk.begin_rendering(
                    cur_slot, rx, ry, VSM_PAGE_TEXELS, VSM_PAGE_TEXELS, False, 0, False, 0, 0, True, self.vsm_img.view, True, 1.0
                )
                var p_cnt = vsm_page_counts[p_idx] if p_idx < len(vsm_page_counts) else 0
                var p_am_cnt = vsm_page_am_counts[p_idx] if p_idx < len(vsm_page_am_counts) else 0
                if p_cnt > 0 or p_am_cnt > 0:
                    self.vk.set_viewport_and_scissor(cur_slot, rx, ry, VSM_PAGE_TEXELS, VSM_PAGE_TEXELS, flip_y=True)
                    var push_vsm = List[UInt32](length=16, fill=0)
                    push_vsm[0] = bitcast[DType.uint32](Float32(128.0))
                    push_vsm[1] = bitcast[DType.uint32](Float32(128.0))
                    push_vsm[2] = bitcast[DType.uint32](bx_f)
                    push_vsm[3] = bitcast[DType.uint32](by_f)
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
        if use_rt_frame and self.enable_rt_gi and sc_inst_count > 0:
            var q_sc = next_q
            if use_prof and q_sc + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_sc, PIPELINE_TOP)
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

        # `16 x 9 x 24` Logarithmic-Z GPU Clustered Punctual + Heitz `LTC` Area Light Culler (`V18`)!
        if n_cl_lights > 0 and not self.furnace_white_dome:
            var q_cl = next_q
            if use_prof and q_cl + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_cl, PIPELINE_TOP)
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

        # `160 x 90 x 64` Temporal Froxel Volumetric Scattering Injection + 3D Reprojection Pass!
        var run_froxel = self.enable_froxel_fog and self.fog_density > Float32(1e-6) and not self.furnace_white_dome
        var run_rain = self.enable_gpu_rain and self.rain_intensity > Float32(0.001) and not self.furnace_white_dome
        var frox_in = froxel_ping_word_offset(max_pix) if (self.frame_index & 1) == 0 else froxel_pong_word_offset(max_pix)
        var frox_out = froxel_pong_word_offset(max_pix) if (self.frame_index & 1) == 0 else froxel_ping_word_offset(max_pix)
        var frox_prefix = froxel_prefix_lut_word_offset(max_pix)
        if run_froxel or run_rain:
            var q_fi = next_q
            if use_prof and q_fi + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_fi, PIPELINE_TOP)
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
        var q_sh = next_q
        if use_prof and q_sh + 1 < q_limit:
            self.vk.write_timestamp(cur_slot, q_sh, PIPELINE_TOP)
            next_q += 2
            prof_pairs.append((String("DeferredMultiLobeRtShade"), q_sh))
        var push_sh = List[UInt32](length=16, fill=0)
        push_sh[0] = UInt32(self.render_w)
        push_sh[1] = UInt32(self.render_h)
        push_sh[2] = UInt32(max(1, self.gi_rays_per_pixel))
        push_sh[3] = UInt32(1) if run_svgf else UInt32(0)
        push_sh[4] = UInt32(1) if run_spec_reblur else UInt32(0)
        push_sh[5] = UInt32(spec_in_off)
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
        self.render_graph.emit_pass_barriers(self.vk, cur_slot, 4)

        # `SVGF` Temporal Reprojection + Moments Variance Edge-Stopping `À-Trous` Denoiser
        if run_svgf:
            var q_sv = next_q
            if use_prof and q_sv + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_sv, PIPELINE_TOP)
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
                self.vk.write_timestamp(cur_slot, q_rb, PIPELINE_TOP)
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
                self.vk.write_timestamp(cur_slot, q_wt, PIPELINE_TOP)
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

        # 64-Slice Hillaire Froxel Prefix-LUT Composite + GPU Froxel-Lit 3D Rain Streaks!
        if run_froxel or run_rain:
            var q_fm = next_q
            if use_prof and q_fm + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_fm, PIPELINE_TOP)
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
                self.vk.write_timestamp(cur_slot, q_oit, PIPELINE_TOP)
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
                self.vk.write_timestamp(cur_slot, q_pt, PIPELINE_TOP)
                next_q += 2
                prof_pairs.append((String("GpuComputeParticles64K"), q_pt))
            var em_off = gpu_particles_emitters_word_offset(max_pix)
            var pool_off = gpu_particles_pool_word_offset(max_pix)
            var t_cnt_off = gpu_particles_tile_counts_word_offset(max_pix)
            var t_bin_off = gpu_particles_tile_bins_word_offset(max_pix)
            var n_em = min(AAA_MAX_PARTICLE_EMITTERS, len(self.frame_particle_emitters))
            if n_em > 0:
                var em_words = List[UInt32](capacity=n_em * 16)
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
            push_ps[2] = bitcast[DType.uint32](Float32(1.0 / 60.0))
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
        var q_tsr = next_q
        if use_prof and q_tsr + 1 < q_limit:
            self.vk.write_timestamp(cur_slot, q_tsr, PIPELINE_TOP)
            next_q += 2
            prof_pairs.append((String("FlintTsrV3Resolve"), q_tsr))
        var ping_off = max_pix * 16
        var pong_off = max_pix * 20
        var out_ldr_off = max_pix * 24
        var hist_in = ping_off if (self.frame_index & 1) == 0 else pong_off
        var hist_out = pong_off if (self.frame_index & 1) == 0 else ping_off
        var eff_hist_w = Float32(0.0) if (self.frame_index == 0 or self.furnace_white_dome) else self.tsr_history_weight

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
            self.pipe_tsr_v3_resolve,
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

        # Thin-Lens Polygonal Diaphragm Bokeh Depth-of-Field at Native `output_w x output_h`!
        if self.enable_bokeh_dof and not self.furnace_white_dome:
            var q_dof = next_q
            if use_prof and q_dof + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_dof, PIPELINE_TOP)
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

        # Energy-Conserving 5-Mip Dual-Kawase + Starburst/Anamorphic HDR Bloom at Native `output_w x output_h`!
        var has_saved_bloom_mip = False
        var saved_bm_w = 0
        var saved_bm_h = 0
        var safe_bloom_mip_off = AAA_POST_BLOOM_BASE + max_pix * 8
        if self.bloom_intensity > Float32(1e-4) and not self.furnace_white_dome:
            var q_bl = next_q
            if use_prof and q_bl + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_bl, PIPELINE_TOP)
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
                self.vk.write_timestamp(cur_slot, q_mb, PIPELINE_TOP)
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

        # GPU 64-Bin Log-Luminance Histogram & Bounded 95%-Trimmed Auto-Exposure!
        if self.enable_auto_exposure and not self.furnace_white_dome:
            var q_ae = next_q
            if use_prof and q_ae + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_ae, PIPELINE_TOP)
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

        # Cinema Display Tonemap (`AgX` Toe + `+5.5 EV` Shoulder / `HDR10 PQ` / `Linear sRGB` + Windshield Rain)!
        var q_dt = next_q
        if use_prof and q_dt + 1 < q_limit:
            self.vk.write_timestamp(cur_slot, q_dt, PIPELINE_TOP)
            next_q += 2
            prof_pairs.append((String("CinemaDisplayTonemap"), q_dt))
        var push_dt = List[UInt32](length=16, fill=0)
        push_dt[0] = UInt32(self.output_w)
        push_dt[1] = UInt32(self.output_h)
        push_dt[2] = UInt32(out_ldr_off)
        push_dt[3] = self.display_mode
        push_dt[4] = UInt32(1) if has_saved_bloom_mip else UInt32(0)
        push_dt[5] = UInt32(saved_bm_w)
        push_dt[6] = UInt32(saved_bm_h)
        push_dt[7] = UInt32(safe_bloom_mip_off)
        push_dt[8] = bitcast[DType.uint32](self.grade_saturation - Float32(1.0))
        push_dt[9] = bitcast[DType.uint32](self.grade_contrast - Float32(1.0))
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

        # Feature 15: Tile-Binned GPU UiSdfOverlayPass (<0.04 ms GPU UI rendering)
        if self.ui_draw_list.count() > 0:
            var q_ui = next_q
            if use_prof and q_ui + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_ui, PIPELINE_TOP)
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
            var prim_cnt = min(self.ui_draw_list.count(), min(4096, max_prims_fit))
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
            push_ui[5] = self.display_mode
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

        # Feature 13 & 14: Presentation & Ring Synchronization
        var do_readback = capture_readback or (self.vk.swapchain == 0)
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
            var img_idx = self.vk.acquire_image(cur_slot)
            if img_idx != NO_IMAGE:
                self.vk.copy_to_image(cur_slot, self.buf_history, self.output_w, img_idx, UInt64(out_ldr_off * 4))
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

    def shutdown(mut self):
        if not self.active and self.vk.instance == 0:
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
        self.vk.destroy_pipeline(self.pipe_visbuffer_gfx)
        self.vk.destroy_pipeline(self.pipe_visbuffer_alphamask_gfx)
        self.vk.destroy_pipeline(self.pipe_vsm_mark)
        self.vk.destroy_pipeline(self.pipe_vsm_gfx)
        self.vk.destroy_pipeline(self.pipe_vsm_alphamask_gfx)
        self.vk.destroy_pipeline(self.pipe_surfcache)
        self.vk.destroy_pipeline(self.pipe_cluster_cull)
        self.vk.destroy_pipeline(self.pipe_dq_skinning)
        self.vk.destroy_pipeline(self.pipe_deferred_shade)
        self.vk.destroy_pipeline(self.pipe_svgf_reproject)
        self.vk.destroy_pipeline(self.pipe_svgf_atrous)
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
        self.vk.destroy_pipeline(self.pipe_tsr_v3_resolve)
        self.vk.destroy_pipeline(self.pipe_motion_blur)
        self.vk.destroy_pipeline(self.pipe_display_tonemap)
        if self.pipe_ui_overlay != 0:
            self.vk.destroy_pipeline(self.pipe_ui_overlay)
            self.pipe_ui_overlay = 0
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

        _ = self.vk.save_pipeline_cache("build/vk_pso.cache")
        self.vk.close()
        self.active = False
