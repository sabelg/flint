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
from std.math import cos, exp2, floor, log2, max, min, sin, sqrt, tan
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
    ACCESS_MEMORY_READ,
    ACCESS_MEMORY_WRITE,
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    ACCESS_TRANSFER_READ,
    ACCESS_TRANSFER_WRITE,
    BUFFER_USAGE,
    BUFFER_USAGE_AS_BUILD_INPUT_KHR,
    BUFFER_USAGE_DEVICE_ADDRESS,
    BUFFER_USAGE_INDEX,
    COMPARE_GREATER_OR_EQUAL,
    COMPARE_LESS_OR_EQUAL,
    CULL_MODE_NONE,
    CULL_MODE_FRONT,
    CULL_MODE_BACK,
    FRONT_FACE_CW,
    FRONT_FACE_CCW,
    INDEX_TYPE_UINT32,
    NO_IMAGE,
    FORMAT_B8G8R8A8_SRGB,
    FORMAT_R8_UNORM,
    FORMAT_R8G8B8A8_UNORM,
    FORMAT_BC5_UNORM,
    FORMAT_BC6H_UFLOAT,
    FORMAT_BC7_SRGB,
    FORMAT_BC7_UNORM,
    FORMAT_D32_SFLOAT,
    FORMAT_R32_SFLOAT,
    FORMAT_R16G16_SFLOAT,
    FORMAT_R16G16B16A16_SFLOAT,
    FORMAT_R32G32_UINT,
    IMAGE_USAGE_COLOR_ATTACHMENT,
    IMAGE_USAGE_DEPTH_ATTACHMENT,
    IMAGE_USAGE_SAMPLED,
    IMAGE_USAGE_STORAGE,
    IMAGE_USAGE_TRANSFER_SRC,
    IMAGE_USAGE_TRANSFER_TARGET,
    LAYOUT_COLOR_ATTACHMENT,
    LAYOUT_DEPTH_ATTACHMENT,
    LAYOUT_GENERAL,
    LAYOUT_PRESENT,
    LAYOUT_SHADER_READ_ONLY,
    LAYOUT_TRANSFER_TARGET,
    LAYOUT_UNDEFINED,
    PIPELINE_ALL_COMMANDS,
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
    VkMemReq,
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
from flint.render.ui_overlay import (
    aaa_ui_overlay_direct_shader,
    aaa_ui_overlay_shader,
    aaa_ui_present_fragment_shader,
    AAA_POST_UI_PRIMS_BASE_OFFSET,
)
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_MAX_BINDLESS,
    AAA_MAX_PACKED_INSTANCES,
    AAA_MAX_PACKED_MESHLET_ID,
    AAA_PUSH_BYTES,
    AAA_PUSH_WORDS,
    AAA_SINGLE_SAMPLERS,
    AAA_STORAGE_COUNT,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_BRDF_LUT,
    BIND_AAA_CANDIDATES,
    BIND_AAA_CLUSTERS,
    BIND_AAA_DEPTH_TEX,
    BIND_AAA_DRAW_PAIRS,
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
    BIND_AAA_SWAPCHAIN_IMG,
    BIND_AAA_TLAS,
    BIND_AAA_VERTICES,
    BIND_AAA_VIS_TEX,
    BIND_AAA_VSM_TABLE,
    BIND_AAA_VSM_TEX,
    BIND_AAA_WORKLIST,
    compute_hiz_mip_levels,
    hiz_depth_downsample_shader,
    make_reverse_z_perspective,
    meshlet_cull_shader,
    visbuffer_alphamask_vertex_shader,
    visbuffer_fragment_shader,
    visbuffer_vertex_shader,
)
from flint.render.aaa_bsdf import AAA_MATERIAL_WORDS, AaaMaterial
from flint.render.material_graph import (
    AAA_GLOBALS_DECAL_COUNT_WORD,
    AAA_GLOBALS_DECAL_TABLE_WORD,
    AAA_GLOBALS_TERRAIN_CLIPMAP_WORD,
    AAA_GLOBALS_TERRAIN_META_WORD,
    AAA_GLOBALS_TERRAIN_ORIGINS_WORD,
    AAA_MAX_DECALS,
    BakedVatCrowdClip,
    GPU_DECAL_VOLUME_WORDS,
    GpuDecalVolume,
    MaterialAsset,
    SurfaceMaterialNode,
    TerrainClipmapDesc,
    VatCrowdClipDesc,
    compile_surface_material_shader,
    f32_to_f16_bits,
    terrain_clipmap_eval_shader,
    vat_crowd_eval_shader,
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
    AAA_SHADE_CLASS_COMPLEX,
    AAA_SHADE_CLASS_COUNT,
    AAA_SHADE_CLASS_FOLIAGE_SSS,
    AAA_SHADE_CLASS_SKY,
    AAA_SHADE_CLASS_STANDARD_PBR,
    AAA_SHADE_CLASS_UNLIT,
    aaa_deferred_shade_class_shader,
    aaa_shade_sky_shader,
    aaa_shade_unlit_shader,
    aaa_tile_classify_shader,
    surface_cache_update_shader,
    svgf_atrous_shader,
    svgf_temporal_reproject_shader,
    tile_classify_word_offset,
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
    aaa_display_present_fragment_shader,
    aaa_display_present_vertex_shader,
    aaa_display_tonemap_direct_shader,
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
    RgAliasingMetrics,
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
    var emissive_bindless_slots: List[Int]

    def __init__(out self):
        self.mesh_entries = List[CookedMeshEntry]()
        self.skinned_entries = List[AaaSkinnedMeshEntry]()
        self.material_ids = List[Int]()
        self.albedo_bindless_slots = List[Int]()
        self.normal_bindless_slots = List[Int]()
        self.orm_bindless_slots = List[Int]()
        self.emissive_bindless_slots = List[Int]()


comptime UploadedModelSlots = AaaUploadedModel


@fieldwise_init
struct _VatRegisteredClip(Copyable, Movable):
    var desc: VatCrowdClipDesc
    var material_id: Int
    var slot_mesh_entries: List[CookedMeshEntry]
    var slot_skin_bases: List[Int]
    var slot_last_norm_phase: List[Float32]

    def __init__(out self):
        self.desc = VatCrowdClipDesc()
        self.material_id = 0
        self.slot_mesh_entries = List[CookedMeshEntry]()
        self.slot_skin_bases = List[Int]()
        self.slot_last_norm_phase = List[Float32]()


@fieldwise_init
struct _VatDrawRequest(Copyable, Movable):
    var clip_id: Int
    var phase_sec: Float32
    var prev_phase_sec: Float32
    var world_mat: Mat4
    var prev_world_mat: Mat4
    var material_override: Int
    var uniform_scale: Float32
    var has_prev: Bool


@fieldwise_init
struct _VatEvalDispatch(Copyable, Movable):
    var clip_id: Int
    var mesh_id: Int
    var first_vertex: Int
    var vertex_count: Int
    var prev_pos_base_word: Int
    var cur_norm_time: Float32
    var prev_norm_time: Float32



comptime FRAMES_IN_FLIGHT: Int = 2
comptime AAA_UI_PRIM_REGION_WORDS: Int = 32768 * 16  # 32768 GpuUiPrim (64 bytes = 2 MiB) per frame in flight
# Per-slot timestamp query range: 64 per-pass queries + 1 whole-frame TOP/BOTTOM pair.
comptime AAA_QUERY_SLOT_STRIDE: Int = 66
comptime AAA_QUERY_FRAME_PAIR: Int = 64
comptime _AAA_STAGE_ALL_COMMANDS: Int = 0x10000
comptime _AAA_ACCESS_MEMORY_READ_WRITE: Int = 0x18000


@fieldwise_init
struct StagingSlice(Copyable, ImplicitlyCopyable, Movable):
    var slab_idx: Int  # -1 = primary slot buffer, >= 0 = overflow_slabs[slot][slab_idx]
    var offset: Int
    var cpu_address: Int
    var size: Int

    def __init__(out self):
        self.slab_idx = -1
        self.offset = 0
        self.cpu_address = 0
        self.size = 0


struct LinearStagingRing(Movable):
    """Per-frame fence-synced host-visible linear upload ring with overflow slabs (zero memcpy of prior slices)."""
    var slot_buffers: List[GpuBuffer]
    var slot_offsets: List[Int]
    var overflow_slabs: List[List[GpuBuffer]]
    var slot_capacity: Int

    def __init__(out self):
        self.slot_buffers = List[GpuBuffer]()
        self.slot_offsets = List[Int]()
        self.overflow_slabs = List[List[GpuBuffer]]()
        self.slot_capacity = 0

    def init(mut self, mut vk: VulkanGpu, initial_slot_bytes: Int = 8 * 1024 * 1024) -> Bool:
        self.shutdown(vk)
        self.slot_capacity = max(65536, (initial_slot_bytes + 4095) & ~4095)
        for _ in range(FRAMES_IN_FLIGHT):
            var sb = vk.create_buffer(self.slot_capacity, local=True)
            if sb.vk_buffer == 0 or sb.cpu_address == 0:
                self.shutdown(vk)
                return False
            self.slot_buffers.append(sb^)
            self.slot_offsets.append(0)
            self.overflow_slabs.append(List[GpuBuffer]())
        return True

    def reset_slot(mut self, mut vk: VulkanGpu, slot: Int):
        """Reclaims `slot` after caller waits on `frame_fences[slot]`."""
        if slot < 0 or slot >= len(self.slot_offsets):
            return
        self.slot_offsets[slot] = 0
        ref slabs = self.overflow_slabs[slot]
        for i in range(len(slabs)):
            if slabs[i].size > 0:
                vk.destroy_buffer(slabs[i])
        slabs.clear()

    def alloc(mut self, mut vk: VulkanGpu, slot: Int, byte_size: Int, align: Int = 256) -> StagingSlice:
        """Allocates `byte_size` in `slot`'s staging ring (or a fresh overflow slab; never reallocates/memcpy prior slices)."""
        if byte_size <= 0 or slot < 0 or slot >= len(self.slot_buffers):
            return StagingSlice()
        var eff_align = max(16, align)
        var cur = (self.slot_offsets[slot] + eff_align - 1) & ~(eff_align - 1)
        if cur + byte_size <= self.slot_buffers[slot].size:
            self.slot_offsets[slot] = cur + byte_size
            return StagingSlice(-1, cur, self.slot_buffers[slot].cpu_address + cur, byte_size)
        var slab_bytes = max(self.slot_capacity, (byte_size + 4095) & ~4095)
        var ov = vk.create_buffer(slab_bytes, local=True)
        if ov.vk_buffer == 0 or ov.cpu_address == 0:
            return StagingSlice()
        var s_idx = len(self.overflow_slabs[slot])
        var addr = ov.cpu_address
        self.overflow_slabs[slot].append(ov^)
        return StagingSlice(s_idx, 0, addr, byte_size)

    def stage_bytes(mut self, mut vk: VulkanGpu, slot: Int, src_ptr: Int, byte_size: Int, align: Int = 256) -> StagingSlice:
        var sl = self.alloc(vk, slot, byte_size, align)
        if sl.cpu_address != 0 and src_ptr != 0 and byte_size > 0:
            _ = external_call["memcpy", Int](sl.cpu_address, src_ptr, byte_size)
        return sl

    def record_copy(
        self,
        mut vk: VulkanGpu,
        cmd_slot: Int,
        ring_slot: Int,
        slice: StagingSlice,
        dst: GpuBuffer,
        dst_offset: Int,
        byte_size: Int = -1,
    ):
        var sz = byte_size if byte_size >= 0 else slice.size
        if sz <= 0 or slice.cpu_address == 0:
            return
        if slice.slab_idx < 0:
            vk.copy_buffer(cmd_slot, self.slot_buffers[ring_slot], slice.offset, dst, dst_offset, sz)
        else:
            vk.copy_buffer(cmd_slot, self.overflow_slabs[ring_slot][slice.slab_idx], slice.offset, dst, dst_offset, sz)

    def upload_to_image_mip(
        self,
        mut vk: VulkanGpu,
        cmd_slot: Int,
        ring_slot: Int,
        slice: StagingSlice,
        img: VulkanImage,
        mip_level: Int,
        mip_w: Int,
        mip_h: Int,
    ):
        if slice.size <= 0 or slice.cpu_address == 0:
            return
        if slice.slab_idx < 0:
            vk.upload_image_mip(cmd_slot, img, self.slot_buffers[ring_slot], slice.offset, mip_level, mip_w, mip_h)
        else:
            vk.upload_image_mip(
                cmd_slot, img, self.overflow_slabs[ring_slot][slice.slab_idx], slice.offset, mip_level, mip_w, mip_h
            )

    def upload_to_image_subregion(
        self,
        mut vk: VulkanGpu,
        cmd_slot: Int,
        ring_slot: Int,
        slice: StagingSlice,
        img: VulkanImage,
        offset_x: Int,
        offset_y: Int,
        width: Int,
        height: Int,
        mip_level: Int = 0,
    ):
        if slice.size <= 0 or slice.cpu_address == 0:
            return
        if slice.slab_idx < 0:
            vk.upload_image_subregion(
                cmd_slot, img, self.slot_buffers[ring_slot], slice.offset, offset_x, offset_y, width, height, mip_level
            )
        else:
            vk.upload_image_subregion(
                cmd_slot,
                img,
                self.overflow_slabs[ring_slot][slice.slab_idx],
                slice.offset,
                offset_x,
                offset_y,
                width,
                height,
                mip_level,
            )

    def shutdown(mut self, mut vk: VulkanGpu):
        for i in range(len(self.overflow_slabs)):
            for j in range(len(self.overflow_slabs[i])):
                if self.overflow_slabs[i][j].size > 0:
                    vk.destroy_buffer(self.overflow_slabs[i][j])
            self.overflow_slabs[i].clear()
        self.overflow_slabs.clear()
        for i in range(len(self.slot_buffers)):
            if self.slot_buffers[i].size > 0:
                vk.destroy_buffer(self.slot_buffers[i])
        self.slot_buffers.clear()
        self.slot_offsets.clear()
        self.slot_capacity = 0


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
    var ring_surfcache: List[GpuBuffer]
    var ring_vsm_table: List[GpuBuffer]
    var ring_oit_staging: List[GpuBuffer]
    var ring_meshlet_worklist: List[GpuBuffer]
    var ring_draw_pairs: List[GpuBuffer]
    var draw_pairs_cap: Int
    var staging_ring: LinearStagingRing
    var meshlet_worklist_cap: Int
    var last_worklist_meshlets: Int
    var side_cmd_base: Int
    var last_async_compute_submits: Int
    var prev_submitted_fence: UInt64
    var swapchain_img_binding: Int
    var ring_tlas: List[VulkanAccelStruct]
    var ring_tlas_cap: List[Int]  # instance capacity each slot's TLAS + shared scratch were sized for
    var desc_sets: List[Int]
    var desc_sets_cull_p1: List[Int]
    var desc_sets_cull_p2: List[Int]
    var frame_fences: List[UInt64]
    var slot_prof_pairs: List[List[Tuple[String, Int]]]
    var slot_frame_q: List[Int]
    var slot_has_pending_queries: List[Bool]
    var swapchain_out_of_date: Bool

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
    var pipe_hiz_downsample: Int
    var pipe_visbuffer_gfx: Int
    var pipe_visbuffer_alphamask_gfx: Int
    var pipe_vsm_mark: Int
    var pipe_vsm_gfx: Int
    var pipe_vsm_alphamask_gfx: Int
    var pipe_surfcache: Int
    var pipe_cluster_cull: Int
    var pipe_dq_skinning: Int
    var pipe_tile_classify: Int
    var pipe_shade_sky: Int
    var pipe_shade_unlit: Int
    var pipe_shade_pbr: Int
    var pipe_shade_foliage_sss: Int
    var pipe_shade_complex: Int
    var pipe_vat_crowd_eval: Int
    var pipe_terrain_clipmap_eval: Int
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
    var pipe_display_tonemap_direct: Int
    var pipe_ui_overlay_direct: Int
    var pipe_display_present_gfx: Int
    var pipe_ui_present_gfx: Int
    var present_gfx_format: UInt32
    var pipe_npr_outline_gfx: Int
    var pipe_npr_cel_shade: Int
    var pipe_npr_post_fx: Int

    # Samplers & Images
    var samp_aniso16: Int
    var samp_aniso16_min_lod: List[Int]
    var samp_linear_clamp: Int
    var samp_nearest: Int
    var pending_bindless_desc_writes: List[List[Tuple[Int, Int, Int]]]
    var deferred_destroy_images: List[List[VulkanImage]]
    var bindless_free_slots: List[Int]
    var bindless_resident_min_mip: List[Int]
    var bindless_vram_bytes: List[Int]
    var bindless_layouts_init: List[Bool]
    var fallback_white_slot: Int
    var vis_img: VulkanImage
    var depth_img: VulkanImage
    var post_alias_img: VulkanImage
    var aliased_buf_heap: Int
    var aliased_img_heap: Int
    var hiz_img: VulkanImage
    var hiz_mip_levels: Int
    var hiz_valid: Bool
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
    var pending_skin_copies: List[Tuple[Int, List[UInt32]]]
    var skin_bones_initialized: List[Bool]

    # CPU Scene & Subsystems
    var geom: AaaGeometryPool
    var materials: List[AaaMaterial]
    var glass_materials: List[AaaGlassParams]  # optics of the transparent materials (`aaa_glass.mojo`)
    var cel_materials: List[AaaCelParams]
    var vfx_materials: List[AaaStylizedVfxParams]
    var frame_shockwaves: List[AaaShockwaveRing]
    var impact_frame: AaaImpactFrameParams
    var frame_instances: List[AaaDrawInstance]
    var frame_decals: List[GpuDecalVolume]
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
    var rg_res_vis: Int
    var rg_res_depth: Int
    var rg_res_indirect: Int
    var rg_res_indirect_p2: Int
    var rg_res_candidates: Int
    var rg_res_vsm_req: Int
    var rg_res_hdr: Int
    var rg_res_oit: Int
    var rg_res_post_img: Int
    var rg_res_svgf: Int
    var rg_res_tsr: Int
    var streamer: AaaResidencyStreamer
    var sh9_irradiance: List[Vec3]
    var env_cube_dim: Int
    var env_mip_levels: Int
    var last_tile_classify_counts: List[Int]

    # F18 VAT Crowd & Terrain Clipmap Runtime State
    var vat_clips: List[_VatRegisteredClip]
    var frame_vat_draws: List[_VatDrawRequest]
    var terrain_clipmap_active: Bool
    var terrain_clipmap_desc: TerrainClipmapDesc
    var terrain_material_id: Int
    var terrain_ring_meshes: List[CookedMeshEntry]
    var terrain_ring_origins: List[Tuple[Float32, Float32]]
    var terrain_clipmap_dirty: Bool

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
    ):
        self.vk = VulkanGpu()
        self.active = False
        self.output_w = max(16, width)
        self.output_h = max(16, height)
        self.tsr_scale = max(Float32(0.50), min(Float32(1.0), tsr_scale))
        self.render_w = max(8, Int(Float32(self.output_w) * self.tsr_scale + Float32(0.5)))
        self.render_h = max(8, Int(Float32(self.output_h) * self.tsr_scale + Float32(0.5)))
        self.max_draws = max(256, max_draws)
        self.max_instances = min(Int(AAA_MAX_PACKED_INSTANCES), max(256, max_instances))
        self.max_surfcache_tris = 262144

        self.dsl = 0
        self.pipe_layout = 0
        self.desc_pool = 0
        self.desc_set = 0
        self.pipe_meshlet_cull = 0
        self.pipe_hiz_downsample = 0
        self.pipe_visbuffer_gfx = 0
        self.pipe_visbuffer_alphamask_gfx = 0
        self.pipe_vsm_mark = 0
        self.pipe_vsm_gfx = 0
        self.pipe_vsm_alphamask_gfx = 0
        self.pipe_surfcache = 0
        self.pipe_cluster_cull = 0
        self.pipe_dq_skinning = 0
        self.pipe_tile_classify = 0
        self.pipe_shade_sky = 0
        self.pipe_shade_unlit = 0
        self.pipe_shade_pbr = 0
        self.pipe_shade_foliage_sss = 0
        self.pipe_shade_complex = 0
        self.pipe_vat_crowd_eval = 0
        self.pipe_terrain_clipmap_eval = 0
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
        self.pipe_display_tonemap_direct = 0
        self.pipe_ui_overlay_direct = 0
        self.pipe_display_present_gfx = 0
        self.pipe_ui_present_gfx = 0
        self.present_gfx_format = 0
        self.pipe_npr_outline_gfx = 0
        self.pipe_npr_cel_shade = 0
        self.pipe_npr_post_fx = 0

        self.samp_aniso16 = 0
        self.samp_aniso16_min_lod = List[Int]()
        self.samp_linear_clamp = 0
        self.samp_nearest = 0
        self.pending_bindless_desc_writes = List[List[Tuple[Int, Int, Int]]]()
        self.deferred_destroy_images = List[List[VulkanImage]]()
        for _ in range(FRAMES_IN_FLIGHT):
            self.pending_bindless_desc_writes.append(List[Tuple[Int, Int, Int]]())
            self.deferred_destroy_images.append(List[VulkanImage]())
        self.bindless_free_slots = List[Int]()
        self.bindless_resident_min_mip = List[Int]()
        self.bindless_vram_bytes = List[Int]()
        self.bindless_layouts_init = List[Bool]()
        self.fallback_white_slot = -1
        self.vis_img = VulkanImage()
        self.depth_img = VulkanImage()
        self.post_alias_img = VulkanImage()
        self.aliased_buf_heap = 0
        self.aliased_img_heap = 0
        self.hiz_img = VulkanImage()
        self.hiz_mip_levels = 1
        self.hiz_valid = False
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
        self.ring_surfcache = List[GpuBuffer]()
        self.ring_vsm_table = List[GpuBuffer]()
        self.ring_oit_staging = List[GpuBuffer]()
        self.ring_meshlet_worklist = List[GpuBuffer]()
        self.ring_draw_pairs = List[GpuBuffer]()
        self.draw_pairs_cap = max(16384, self.max_draws) * 5
        self.staging_ring = LinearStagingRing()
        self.meshlet_worklist_cap = max(self.max_draws * 2, 262144)
        self.last_worklist_meshlets = 0
        self.side_cmd_base = -1
        self.last_async_compute_submits = 0
        self.prev_submitted_fence = 0
        self.swapchain_img_binding = BIND_AAA_SWAPCHAIN_IMG
        self.ring_tlas = List[VulkanAccelStruct]()
        self.ring_tlas_cap = List[Int]()
        self.desc_sets = List[Int]()
        self.desc_sets_cull_p1 = List[Int]()
        self.desc_sets_cull_p2 = List[Int]()
        self.frame_fences = List[UInt64]()
        self.slot_prof_pairs = List[List[Tuple[String, Int]]]()
        self.slot_frame_q = List[Int]()
        self.slot_has_pending_queries = List[Bool]()
        self.swapchain_out_of_date = False

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
        self.pending_skin_copies = List[Tuple[Int, List[UInt32]]]()
        self.skin_bones_initialized = List[Bool](length=AAA_MAX_SKIN_BONES, fill=False)

        self.geom = AaaGeometryPool()
        self.materials = List[AaaMaterial]()
        self.frame_instances = List[AaaDrawInstance]()
        self.frame_decals = List[GpuDecalVolume]()
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
        self.rg_res_vis = -1
        self.rg_res_depth = -1
        self.rg_res_indirect = -1
        self.rg_res_indirect_p2 = -1
        self.rg_res_candidates = -1
        self.rg_res_vsm_req = -1
        self.rg_res_hdr = -1
        self.rg_res_oit = -1
        self.rg_res_post_img = -1
        self.rg_res_svgf = -1
        self.rg_res_tsr = -1
        self.streamer = AaaResidencyStreamer()
        self.sh9_irradiance = List[Vec3](length=9, fill=Vec3(0.0, 0.0, 0.0))
        self.env_cube_dim = max(16, env_cube_dim)
        var m_cnt = 1
        var tmp_d = self.env_cube_dim
        while tmp_d > 4 and m_cnt < 7:
            tmp_d >>= 1
            m_cnt += 1
        self.env_mip_levels = m_cnt
        self.last_tile_classify_counts = List[Int](length=AAA_SHADE_CLASS_COUNT, fill=0)

        self.vat_clips = List[_VatRegisteredClip]()
        self.frame_vat_draws = List[_VatDrawRequest]()
        self.terrain_clipmap_active = False
        self.terrain_clipmap_desc = TerrainClipmapDesc()
        self.terrain_material_id = 0
        self.terrain_ring_meshes = List[CookedMeshEntry]()
        self.terrain_ring_origins = List[Tuple[Float32, Float32]]()
        self.terrain_clipmap_dirty = False

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
        self.enable_async_vsm = True
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

        self.active = self._init_vulkan_pipeline()

    def _free_aliased_targets(mut self):
        if self.vis_img.image != 0:
            self.vk.destroy_image(self.vis_img)
        if self.depth_img.image != 0:
            self.vk.destroy_image(self.depth_img)
        if self.post_alias_img.image != 0:
            self.vk.destroy_image(self.post_alias_img)
        if self.aliased_img_heap != 0:
            self.vk.destroy_aliased_heap(self.aliased_img_heap)
            self.aliased_img_heap = 0
        if self.buf_indirect_phase2.vk_buffer != 0:
            self.vk.destroy_buffer(self.buf_indirect_phase2)
            self.buf_indirect_phase2 = GpuBuffer()
        if self.buf_occluded_candidates.vk_buffer != 0:
            self.vk.destroy_buffer(self.buf_occluded_candidates)
            self.buf_occluded_candidates = GpuBuffer()
        if self.buf_hdr_out.vk_buffer != 0:
            self.vk.destroy_buffer(self.buf_hdr_out)
            self.buf_hdr_out = GpuBuffer()
        if self.aliased_buf_heap != 0:
            self.vk.destroy_aliased_heap(self.aliased_buf_heap)
            self.aliased_buf_heap = 0

    def _build_default_render_graph(
        mut self,
        req_p2: VkMemReq = VkMemReq(),
        req_cand: VkMemReq = VkMemReq(),
        req_hdr: VkMemReq = VkMemReq(),
        req_vis: VkMemReq = VkMemReq(),
        req_depth: VkMemReq = VkMemReq(),
        req_post_img: VkMemReq = VkMemReq(),
    ):
        self.render_graph = AaaRenderGraph()
        var pix_r = self.render_w * self.render_h
        var pix_o = self.output_w * self.output_h
        var r_indirect = self.render_graph.add_resource("IndirectDrawCmds", 64 + self.max_draws * 20, transient=False, is_image=False)
        var r_indirect_p2 = self.render_graph.add_resource("IndirectDrawCmdsP2", 64 + self.max_draws * 20, transient=True, is_image=False)
        var r_candidates = self.render_graph.add_resource("OccludedCandidates", 64 + self.max_draws * 8, transient=True, is_image=False)
        var r_hiz = self.render_graph.add_resource("ReverseZDepth", pix_r * 4, transient=True, is_image=True)
        var r_vis = self.render_graph.add_resource("VisBuffer64", pix_r * 8, transient=True, is_image=True)
        var r_post_img = self.render_graph.add_resource("PostTransientImg", pix_r * 8, transient=True, is_image=True)
        var r_vsm_req = self.render_graph.add_resource("VsmPageRequests", VSM_TABLE_WORDS * 4, transient=False, is_image=False)
        var r_hdr = self.render_graph.add_resource("LinearHdrScratch", pix_r * 32, transient=True, is_image=False)
        var r_svgf = self.render_graph.add_resource("SvgfPingPong", pix_r * 32, transient=False, is_image=False)
        var r_oit = self.render_graph.add_resource("Oit4NodePool", max(4096, pix_r * 32), transient=False, is_image=False)
        var r_tsr = self.render_graph.add_resource("TsrUpscaleScratch", pix_o * 16, transient=False, is_image=False)
        if req_p2.size > 0:
            self.render_graph.set_resource_requirements(r_indirect_p2, req_p2)
        if req_cand.size > 0:
            self.render_graph.set_resource_requirements(r_candidates, req_cand)
        if req_hdr.size > 0:
            self.render_graph.set_resource_requirements(r_hdr, req_hdr)
        if req_vis.size > 0:
            self.render_graph.set_resource_requirements(r_vis, req_vis)
        if req_depth.size > 0:
            self.render_graph.set_resource_requirements(r_hiz, req_depth)
        if req_post_img.size > 0:
            self.render_graph.set_resource_requirements(r_post_img, req_post_img)

        self.rg_res_vis = r_vis
        self.rg_res_depth = r_hiz
        self.rg_res_indirect = r_indirect
        self.rg_res_indirect_p2 = r_indirect_p2
        self.rg_res_candidates = r_candidates
        self.rg_res_vsm_req = r_vsm_req
        self.rg_res_hdr = r_hdr
        self.rg_res_oit = r_oit
        self.rg_res_post_img = r_post_img
        self.rg_res_svgf = r_svgf
        self.rg_res_tsr = r_tsr

        var p0 = List[RgResourceAccess]()
        p0.append(RgResourceAccess(r_indirect, RG_USAGE_COMPUTE_WRITE))
        p0.append(RgResourceAccess(r_indirect_p2, RG_USAGE_COMPUTE_WRITE))
        p0.append(RgResourceAccess(r_candidates, RG_USAGE_COMPUTE_WRITE))
        _ = self.render_graph.add_pass("MeshletCull", p0, RG_QUEUE_GRAPHICS)

        var p1 = List[RgResourceAccess]()
        p1.append(RgResourceAccess(r_indirect, RG_USAGE_INDIRECT_ARGS))
        p1.append(RgResourceAccess(r_indirect_p2, RG_USAGE_INDIRECT_ARGS))
        p1.append(RgResourceAccess(r_candidates, RG_USAGE_COMPUTE_READ))
        p1.append(RgResourceAccess(r_vis, RG_USAGE_COLOR_ATT_WRITE))
        p1.append(RgResourceAccess(r_hiz, RG_USAGE_DEPTH_ATT_WRITE))
        _ = self.render_graph.add_pass("VisBufferHwRaster", p1, RG_QUEUE_GRAPHICS)

        var p2 = List[RgResourceAccess]()
        p2.append(RgResourceAccess(r_vis, RG_USAGE_COMPUTE_READ))
        p2.append(RgResourceAccess(r_hiz, RG_USAGE_COMPUTE_READ))
        p2.append(RgResourceAccess(r_vsm_req, RG_USAGE_COMPUTE_WRITE))
        _ = self.render_graph.add_pass("VsmMarkPages", p2, RG_QUEUE_ASYNC_COMPUTE)

        var p3 = List[RgResourceAccess]()
        p3.append(RgResourceAccess(r_vis, RG_USAGE_COMPUTE_READ))
        p3.append(RgResourceAccess(r_hiz, RG_USAGE_COMPUTE_READ))
        p3.append(RgResourceAccess(r_vsm_req, RG_USAGE_COMPUTE_READ))
        p3.append(RgResourceAccess(r_hdr, RG_USAGE_COMPUTE_WRITE))
        _ = self.render_graph.add_pass("DeferredMultiLobeRtShade", p3, RG_QUEUE_GRAPHICS)

        var p4 = List[RgResourceAccess]()
        p4.append(RgResourceAccess(r_vis, RG_USAGE_COMPUTE_READ))
        p4.append(RgResourceAccess(r_hiz, RG_USAGE_COMPUTE_READ))
        p4.append(RgResourceAccess(r_hdr, RG_USAGE_COMPUTE_READ))
        p4.append(RgResourceAccess(r_svgf, RG_USAGE_COMPUTE_WRITE))
        _ = self.render_graph.add_pass("SvgfAtrousWavelet", p4, RG_QUEUE_GRAPHICS)

        var p5 = List[RgResourceAccess]()
        p5.append(RgResourceAccess(r_hiz, RG_USAGE_COMPUTE_READ))
        p5.append(RgResourceAccess(r_svgf, RG_USAGE_COMPUTE_READ))
        p5.append(RgResourceAccess(r_hdr, RG_USAGE_COMPUTE_WRITE))
        p5.append(RgResourceAccess(r_oit, RG_USAGE_COMPUTE_WRITE))
        p5.append(RgResourceAccess(r_post_img, RG_USAGE_COLOR_ATT_WRITE))
        _ = self.render_graph.add_pass("Oit6WayVolumetrics", p5, RG_QUEUE_GRAPHICS)

        var p6 = List[RgResourceAccess]()
        p6.append(RgResourceAccess(r_hiz, RG_USAGE_COMPUTE_READ))
        p6.append(RgResourceAccess(r_svgf, RG_USAGE_COMPUTE_READ))
        p6.append(RgResourceAccess(r_hdr, RG_USAGE_COMPUTE_READ))
        p6.append(RgResourceAccess(r_oit, RG_USAGE_COMPUTE_READ))
        p6.append(RgResourceAccess(r_post_img, RG_USAGE_COMPUTE_READ))
        p6.append(RgResourceAccess(r_tsr, RG_USAGE_COMPUTE_WRITE))
        _ = self.render_graph.add_pass("TsrV2AndDisplayOetf", p6, RG_QUEUE_GRAPHICS)

        _ = self.render_graph.compile()

    def _allocate_and_bind_aliased_targets(mut self) -> Bool:
        self._free_aliased_targets()
        var max_pix = max(self.output_w * self.output_h, self.render_w * self.render_h)
        var img_conc = self.vk.has_side_queue()
        var vis_req = VkMemReq()
        var dep_req = VkMemReq()
        var post_img_req = VkMemReq()
        var p2_req = VkMemReq()
        var cand_req = VkMemReq()
        var hdr_req = VkMemReq()

        self.vis_img = self.vk.create_image_2d_unbound(
            self.render_w,
            self.render_h,
            FORMAT_R32G32_UINT,
            IMAGE_USAGE_COLOR_ATTACHMENT | IMAGE_USAGE_SAMPLED | IMAGE_USAGE_TRANSFER_SRC,
            vis_req,
            concurrent=img_conc,
        )
        self.depth_img = self.vk.create_image_2d_unbound(
            self.render_w,
            self.render_h,
            FORMAT_D32_SFLOAT,
            IMAGE_USAGE_DEPTH_ATTACHMENT | IMAGE_USAGE_SAMPLED,
            dep_req,
            concurrent=img_conc,
        )
        self.post_alias_img = self.vk.create_image_2d_unbound(
            self.render_w,
            self.render_h,
            FORMAT_R16G16B16A16_SFLOAT,
            IMAGE_USAGE_COLOR_ATTACHMENT | IMAGE_USAGE_SAMPLED | IMAGE_USAGE_TRANSFER_TARGET,
            post_img_req,
            concurrent=False,
        )
        self.buf_indirect_phase2 = self.vk.create_buffer_unbound(64 + self.max_draws * 20, p2_req, concurrent=False)
        self.buf_occluded_candidates = self.vk.create_buffer_unbound(64 + self.max_draws * 8, cand_req, concurrent=False)
        self.buf_hdr_out = self.vk.create_buffer_unbound(max_pix * 8 * 4, hdr_req, concurrent=False)

        if (
            self.vis_img.image == 0
            or self.depth_img.image == 0
            or self.post_alias_img.image == 0
            or self.buf_indirect_phase2.vk_buffer == 0
            or self.buf_occluded_candidates.vk_buffer == 0
            or self.buf_hdr_out.vk_buffer == 0
        ):
            self._free_aliased_targets()
            return False

        self._build_default_render_graph(
            p2_req, cand_req, hdr_req, vis_req, dep_req, post_img_req
        )

        self.aliased_img_heap = self.vk.create_aliased_heap(
            max(4096, self.render_graph.aliased_img_peak_bytes),
            self.render_graph.image_memory_type_bits,
            local=True,
        )
        self.aliased_buf_heap = self.vk.create_aliased_heap(
            max(4096, self.render_graph.aliased_buf_peak_bytes),
            self.render_graph.buffer_memory_type_bits,
            local=True,
        )
        if self.aliased_img_heap == 0 or self.aliased_buf_heap == 0:
            self._free_aliased_targets()
            return False

        var ok_bind = (
            self.vk.bind_image_aliased(
                self.vis_img, self.aliased_img_heap, self.render_graph.resources[self.rg_res_vis].aliased_offset_bytes
            )
            and self.vk.bind_image_aliased(
                self.depth_img, self.aliased_img_heap, self.render_graph.resources[self.rg_res_depth].aliased_offset_bytes
            )
            and self.vk.bind_image_aliased(
                self.post_alias_img,
                self.aliased_img_heap,
                self.render_graph.resources[self.rg_res_post_img].aliased_offset_bytes,
            )
            and self.vk.bind_buffer_aliased(
                self.buf_indirect_phase2,
                self.aliased_buf_heap,
                self.render_graph.resources[self.rg_res_indirect_p2].aliased_offset_bytes,
            )
            and self.vk.bind_buffer_aliased(
                self.buf_occluded_candidates,
                self.aliased_buf_heap,
                self.render_graph.resources[self.rg_res_candidates].aliased_offset_bytes,
            )
            and self.vk.bind_buffer_aliased(
                self.buf_hdr_out,
                self.aliased_buf_heap,
                self.render_graph.resources[self.rg_res_hdr].aliased_offset_bytes,
            )
        )
        if not ok_bind:
            self._free_aliased_targets()
            return False

        self.render_graph.bind_resource_buffer(self.rg_res_indirect_p2, self.buf_indirect_phase2.vk_buffer, 0, self.buf_indirect_phase2.size)
        self.render_graph.bind_resource_buffer(self.rg_res_candidates, self.buf_occluded_candidates.vk_buffer, 0, self.buf_occluded_candidates.size)
        self.render_graph.bind_resource_buffer(self.rg_res_hdr, self.buf_hdr_out.vk_buffer, 0, self.buf_hdr_out.size)
        return True

    def _bind_render_graph_persistent_buffers(mut self, slot: Int):
        if slot < len(self.ring_indirect):
            self.render_graph.bind_resource_buffer(self.rg_res_indirect, self.ring_indirect[slot].vk_buffer, 0, self.ring_indirect[slot].size)
        if slot < len(self.ring_vsm_table):
            self.render_graph.bind_resource_buffer(self.rg_res_vsm_req, self.ring_vsm_table[slot].vk_buffer, 0, self.ring_vsm_table[slot].size)
        if self.buf_oit_nodes.vk_buffer != 0:
            self.render_graph.bind_resource_buffer(self.rg_res_oit, self.buf_oit_nodes.vk_buffer, 0, self.buf_oit_nodes.size)
        if self.buf_post_scratch.vk_buffer != 0:
            self.render_graph.bind_resource_buffer(self.rg_res_svgf, self.buf_post_scratch.vk_buffer, 0, self.buf_post_scratch.size)
        if self.buf_history.vk_buffer != 0:
            self.render_graph.bind_resource_buffer(self.rg_res_tsr, self.buf_history.vk_buffer, 0, self.buf_history.size)

    def _wait_cmd_slot0(mut self):
        """Waits on `frame_fences[0]` if still in flight, resets its staging slice, and flushes retirements."""
        if len(self.frame_fences) > 0 and self.frame_fences[0] != 0:
            _ = self.vk.wait(self.frame_fences[0])
            if self.prev_submitted_fence <= self.frame_fences[0]:
                self.prev_submitted_fence = 0
            self.frame_fences[0] = 0
            self.staging_ring.reset_slot(self.vk, 0)
            self._flush_slot_retirements(0)

    def _wait_all_in_flight(mut self):
        """Waits for all in-flight frame slots, resets staging ring slices, and flushes retirements."""
        var any_in_flight = False
        for i in range(len(self.frame_fences)):
            if self.frame_fences[i] != 0:
                any_in_flight = True
                break
        if not any_in_flight:
            return
        self.vk.wait_idle()
        self.prev_submitted_fence = 0
        for i in range(len(self.frame_fences)):
            self.frame_fences[i] = 0
        for i in range(len(self.staging_ring.slot_buffers)):
            self.staging_ring.reset_slot(self.vk, i)
        var max_retire = max(len(self.pending_bindless_desc_writes), len(self.deferred_destroy_images))
        for i in range(max_retire):
            self._flush_slot_retirements(i)

    def _upload_to_device_local_sync(mut self, src_ptr: Int, byte_len: Int, dst_binding: Int, dst_offset: Int = 0):
        if byte_len <= 0 or src_ptr == 0:
            return
        var dst_ok = (
            (dst_binding == BIND_AAA_MATERIALS and self.buf_materials.vk_buffer != 0)
            or (dst_binding == BIND_AAA_SKINNING and self.buf_skinning.vk_buffer != 0)
        )
        if not dst_ok:
            return
        self._wait_cmd_slot0()
        self.ensure_staging_capacity(byte_len)
        if self.buf_staging.cpu_address == 0:
            return
        _ = external_call["memcpy", Int](self.buf_staging.cpu_address, src_ptr, byte_len)
        if not self.vk.begin_commands(0):
            return
        if dst_binding == BIND_AAA_MATERIALS:
            self.vk.copy_buffer(0, self.buf_staging, 0, self.buf_materials, dst_offset, byte_len)
        elif dst_binding == BIND_AAA_SKINNING:
            self.vk.copy_buffer(0, self.buf_staging, 0, self.buf_skinning, dst_offset, byte_len)
        self.vk.memory_barrier2(
            0,
            PIPELINE_TRANSFER,
            Int(ACCESS_TRANSFER_WRITE),
            PIPELINE_ALL_COMMANDS,
            Int(ACCESS_MEMORY_READ | ACCESS_SHADER_READ | ACCESS_INDIRECT_READ | ACCESS_AS_READ_KHR),
        )
        var f = self.vk.submit(0)
        if f != 0:
            _ = self.vk.wait(f)

    def _grow_storage_buffer(
        mut self,
        needed_bytes: Int,
        binding: Int,
        extra_usage: UInt32,
        bda: Bool,
        update_cull_sets: Bool,
    ) -> Bool:
        var old_size: Int
        var old_valid: Bool
        if binding == BIND_AAA_VERTICES:
            old_size = self.buf_vertices.size
            old_valid = self.buf_vertices.vk_buffer != 0
        elif binding == BIND_AAA_INDICES:
            old_size = self.buf_indices.size
            old_valid = self.buf_indices.vk_buffer != 0
        elif binding == BIND_AAA_MESHLETS:
            old_size = self.buf_meshlets.size
            old_valid = self.buf_meshlets.vk_buffer != 0
        elif binding == BIND_AAA_MATERIALS:
            old_size = self.buf_materials.size
            old_valid = self.buf_materials.vk_buffer != 0
        else:
            return False
        if needed_bytes <= old_size:
            return True
        self.vk.wait_idle()
        for i in range(len(self.frame_fences)):
            self.frame_fences[i] = 0
            self._flush_slot_retirements(i)
        self.prev_submitted_fence = 0
        var new_cap = max(max(65536, old_size * 2), needed_bytes)
        new_cap = (new_cap + 65535) & ~65535
        var new_buf = self.vk.create_buffer_device_local(new_cap, usage=BUFFER_USAGE | extra_usage, bda=bda)
        if new_buf.vk_buffer == 0:
            return False
        if old_valid and old_size > 0:
            if not self.vk.begin_commands(0):
                self.vk.destroy_buffer(new_buf)
                return False
            if binding == BIND_AAA_VERTICES:
                self.vk.copy_buffer(0, self.buf_vertices, 0, new_buf, 0, old_size)
            elif binding == BIND_AAA_INDICES:
                self.vk.copy_buffer(0, self.buf_indices, 0, new_buf, 0, old_size)
            elif binding == BIND_AAA_MESHLETS:
                self.vk.copy_buffer(0, self.buf_meshlets, 0, new_buf, 0, old_size)
            elif binding == BIND_AAA_MATERIALS:
                self.vk.copy_buffer(0, self.buf_materials, 0, new_buf, 0, old_size)
            self.vk.memory_barrier2(
                0,
                PIPELINE_TRANSFER,
                Int(ACCESS_TRANSFER_WRITE),
                PIPELINE_ALL_COMMANDS,
                Int(ACCESS_MEMORY_READ | ACCESS_SHADER_READ | ACCESS_INDIRECT_READ | ACCESS_AS_READ_KHR),
            )
            var f = self.vk.submit(0)
            if f == 0 or not self.vk.wait(f):
                self.vk.destroy_buffer(new_buf)
                return False
            if binding == BIND_AAA_VERTICES:
                self.vk.destroy_buffer(self.buf_vertices)
            elif binding == BIND_AAA_INDICES:
                self.vk.destroy_buffer(self.buf_indices)
            elif binding == BIND_AAA_MESHLETS:
                self.vk.destroy_buffer(self.buf_meshlets)
            elif binding == BIND_AAA_MATERIALS:
                self.vk.destroy_buffer(self.buf_materials)
        for slot in range(len(self.desc_sets)):
            self.vk.write_descriptor(self.desc_sets[slot], binding, new_buf)
            if update_cull_sets:
                if slot < len(self.desc_sets_cull_p1):
                    self.vk.write_descriptor(self.desc_sets_cull_p1[slot], binding, new_buf)
                if slot < len(self.desc_sets_cull_p2):
                    self.vk.write_descriptor(self.desc_sets_cull_p2[slot], binding, new_buf)
        if binding == BIND_AAA_VERTICES:
            self.buf_vertices = new_buf^
        elif binding == BIND_AAA_INDICES:
            self.buf_indices = new_buf^
        elif binding == BIND_AAA_MESHLETS:
            self.buf_meshlets = new_buf^
        elif binding == BIND_AAA_MATERIALS:
            self.buf_materials = new_buf^
        return True

    def _init_vulkan_pipeline(mut self) -> Bool:
        if not self.vk.initialize(validate=False, need_discrete=False):
            return False
        if not self.vk.has_dynamic_rendering or not self.vk.has_descriptor_indexing:
            return False
        if not self.vk.allocate_command_buffers(FRAMES_IN_FLIGHT * 2):
            return False
        if self.vk.has_side_queue():
            self.side_cmd_base = self.vk.allocate_side_command_buffers(FRAMES_IN_FLIGHT)

        var use_rt = self.vk.has_ray_query
        self.swapchain_img_binding = AAA_STORAGE_COUNT + AAA_SINGLE_SAMPLERS + 1 + (1 if use_rt else 0)
        var layouts = self.vk.create_aaa_layouts(
            AAA_STORAGE_COUNT,
            AAA_SINGLE_SAMPLERS,
            AAA_MAX_BINDLESS,
            use_rt,
            AAA_PUSH_BYTES,
            storage_image_count=1,
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
            storage_image_count=1,
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

        # Compile core hardware Graphics & Compute SPIR-V 1.5 Pipelines eagerly;
        # optional feature pipelines & specialized shade classes compile lazily on first use.
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
        self.pipe_tile_classify = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_tile_classify_shader())
        self.pipe_shade_sky = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_shade_sky_shader(True))
        self.pipe_shade_pbr = self.vk.create_aaa_compute_pipeline(
            self.pipe_layout, aaa_deferred_shade_class_shader(use_rt, AAA_SHADE_CLASS_STANDARD_PBR, True)
        )
        self.pipe_tsr_v3_resolve = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_tsr_v3_resolve_shader())
        self.pipe_display_tonemap = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_display_tonemap_shader())
        self.pipe_ui_overlay = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_ui_overlay_shader())

        if (
            self.pipe_meshlet_cull == 0
            or self.pipe_visbuffer_gfx == 0
            or self.pipe_vsm_mark == 0
            or self.pipe_vsm_gfx == 0
            or self.pipe_tile_classify == 0
            or self.pipe_shade_sky == 0
            or self.pipe_shade_pbr == 0
            or self.pipe_tsr_v3_resolve == 0
            or self.pipe_display_tonemap == 0
            or self.pipe_ui_overlay == 0
        ):
            return False

        if not self.staging_ring.init(self.vk, 8 * 1024 * 1024):
            return False

        # Aliased transient Render Targets & Buffers (`aliased_img_heap` + `aliased_buf_heap`)
        if not self._allocate_and_bind_aliased_targets():
            return False

        self.hiz_mip_levels = compute_hiz_mip_levels(self.render_w, self.render_h)
        self.hiz_img = self.vk.create_image_2d(
            self.render_w,
            self.render_h,
            FORMAT_R32_SFLOAT,
            IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED,
            mip_levels=self.hiz_mip_levels,
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
            or self.hiz_img.image == 0
            or self.vsm_img.image == 0
            or self.brdf_lut_img.image == 0
            or self.ltc_lut_img.image == 0
            or self.env_cube_img.image == 0
        ):
            return False

        self.samp_aniso16 = self.vk.create_sampler(linear=True, max_anisotropy=16.0, clamp_to_edge=False, max_lod=16.0)
        self.samp_aniso16_min_lod.clear()
        for m_lod in range(16):
            var ms = self.vk.create_sampler(
                linear=True,
                max_anisotropy=16.0,
                clamp_to_edge=False,
                max_lod=16.0,
                min_lod=Float32(m_lod),
            )
            self.samp_aniso16_min_lod.append(ms)
        self.samp_linear_clamp = self.vk.create_sampler(linear=True, max_anisotropy=1.0, clamp_to_edge=True, max_lod=16.0)
        self.samp_nearest = self.vk.create_sampler(linear=False, max_anisotropy=1.0, clamp_to_edge=True, max_lod=16.0)
        if self.samp_aniso16 == 0 or self.samp_linear_clamp == 0 or self.samp_nearest == 0:
            return False

        # UI Font atlas (Feature 15, owned directly in bindless_images without .copy())
        _ = self.ui_font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36, 5)
        var fw = self.ui_font.atlas_width if self.ui_font.atlas_width > 0 else 512
        var fh = self.ui_font.atlas_height if self.ui_font.atlas_height > 0 else 512
        var font_img = self.vk.create_image_2d(
            fw, fh, FORMAT_R8_UNORM, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED
        )
        self.ui_font_atlas_slot = len(self.bindless_images)
        self.bindless_images.append(font_img^)

        # Allocate true DEVICE_LOCAL Storage Buffers (`cpu_address == 0` where VRAM-only exists)
        var max_pix = max(self.output_w * self.output_h, self.render_w * self.render_h)
        var bda_flags = BUFFER_USAGE_DEVICE_ADDRESS | BUFFER_USAGE_AS_BUILD_INPUT_KHR
        self.buf_vertices = self.vk.create_buffer_device_local(2 * 1024 * 1024, usage=BUFFER_USAGE | BUFFER_USAGE_AS_BUILD_INPUT_KHR, bda=True)
        self.buf_indices = self.vk.create_buffer_device_local(2 * 1024 * 1024, usage=BUFFER_USAGE | BUFFER_USAGE_INDEX | BUFFER_USAGE_AS_BUILD_INPUT_KHR, bda=True)
        self.buf_meshlets = self.vk.create_buffer_device_local(1 * 1024 * 1024)
        self.buf_materials = self.vk.create_buffer_device_local(256 * AAA_MATERIAL_WORDS * 4)
        self.buf_vsm_indirect = self.vk.create_buffer_device_local(64 + self.max_draws * 16)
        var hist_words = rt_shadow_raw_offset(max_pix) + rt_shadow_total_words(max_pix)
        self.buf_history = self.vk.create_buffer_device_local(hist_words * 4)
        self.buf_oit_nodes = self.vk.create_buffer_device_local(gpu_particles_total_oit_words(max_pix) * 4)
        self.buf_skinning = self.vk.create_buffer_device_local(AAA_SKIN_BUFFER_WORDS * 4)
        var scratch_words = AAA_POST_UI_PRIMS_BASE_OFFSET + max_pix * 10 + FRAMES_IN_FLIGHT * AAA_UI_PRIM_REGION_WORDS
        self.buf_post_scratch = self.vk.create_buffer_device_local(scratch_words * 4)
        self.buf_ldr_readback = self.vk.create_buffer(64 + self.output_w * self.output_h * 4, local=False, cpu_reads=True)
        self.buf_staging = self.vk.create_buffer(8 * 1024 * 1024, local=False)
        self.buf_globals = self.vk.create_buffer(4096, local=False)
        if self.buf_globals.cpu_address != 0:
            _ = external_call["memset", Int](self.buf_globals.cpu_address, Int32(0), self.buf_globals.size)

        # Ring-Buffered CPU-to-GPU Dynamic Buffers (DEVICE_LOCAL, staged through LinearStagingRing)
        var oit_stage_bytes = (AAA_MAX_OIT_PRIMS * 16 + AAA_MAX_PARTICLE_EMITTERS * AAA_EMITTER_WORDS + 64) * 4
        var vsm_conc = self.vk.has_side_queue()
        for _ in range(FRAMES_IN_FLIGHT):
            var b_glob = self.vk.create_buffer_device_local(16384, concurrent=vsm_conc)
            var b_inst = self.vk.create_buffer_device_local(self.max_instances * AAA_INSTANCE_WORDS * 4)
            var b_ind = self.vk.create_buffer_device_local(64 + self.max_draws * 20)
            var b_am_ind = self.vk.create_buffer_device_local(64 + 16384 * 20)
            var b_out_ind = self.vk.create_buffer_device_local(64 + 16384 * 20)
            var b_clust = self.vk.create_buffer_device_local(AAA_CLUSTER_BUFFER_WORDS * 4)
            var b_sc = self.vk.create_buffer_device_local((4096 + self.max_surfcache_tris * 4) * 4)
            var b_vsm = self.vk.create_buffer(VSM_TABLE_WORDS * 4, local=False, cpu_reads=True, concurrent=vsm_conc)
            if b_vsm.cpu_address != 0:
                _ = external_call["memset", Int](b_vsm.cpu_address, Int32(0), b_vsm.size)
            var b_oit_st = self.vk.create_buffer(oit_stage_bytes, local=False)
            if b_oit_st.cpu_address != 0:
                _ = external_call["memset", Int](b_oit_st.cpu_address, Int32(0), b_oit_st.size)
            var b_wl = self.vk.create_buffer_device_local(self.meshlet_worklist_cap * 8)
            var b_dp = self.vk.create_buffer_device_local(self.draw_pairs_cap * 8)
            self.ring_globals.append(b_glob^)
            self.ring_instances.append(b_inst^)
            self.ring_indirect.append(b_ind^)
            self.ring_alphamask_indirect.append(b_am_ind^)
            self.ring_outline_indirect.append(b_out_ind^)
            self.ring_clusters.append(b_clust^)
            self.ring_surfcache.append(b_sc^)
            self.ring_vsm_table.append(b_vsm^)
            self.ring_oit_staging.append(b_oit_st^)
            self.ring_meshlet_worklist.append(b_wl^)
            self.ring_draw_pairs.append(b_dp^)
            if self.vk.has_ray_query:
                var b_as = self.vk.create_buffer_device_local(self.max_instances * 64, usage=BUFFER_USAGE | bda_flags, bda=True)
                var tlas = VulkanAccelStruct()
                var init_inst_da = self.vk.buffer_device_address(b_as)
                _ = ensure_tlas_capacity_mixed_opaque(self.vk, tlas, init_inst_da, self.max_instances, self.buf_as_scratch)
                self.ring_as_instances.append(b_as^)
                self.ring_tlas.append(tlas^)
            self.frame_fences.append(0)
            self.slot_prof_pairs.append(List[Tuple[String, Int]]())
            self.slot_frame_q.append(-1)
            self.slot_has_pending_queries.append(False)

        self.buf_instances = GpuBuffer(0, self.ring_instances[0].size, 0, 0)
        self.buf_indirect = GpuBuffer(0, self.ring_indirect[0].size, 0, 0)
        self.buf_alphamask_indirect = GpuBuffer(0, self.ring_alphamask_indirect[0].size, 0, 0)
        self.buf_outline_indirect = GpuBuffer(0, self.ring_outline_indirect[0].size, 0, 0)
        self.buf_clusters = GpuBuffer(0, self.ring_clusters[0].size, 0, 0)
        self.buf_surfcache = GpuBuffer(0, self.ring_surfcache[0].size, 0, 0)
        self.buf_vsm_table = GpuBuffer(0, self.ring_vsm_table[0].size, 0, 0)
        if self.vk.has_ray_query and len(self.ring_tlas) > 0:
            self.frame_tlas.handle = self.ring_tlas[0].handle

        if self.buf_ldr_readback.cpu_address != 0:
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
            self.vk.write_descriptor(ds, BIND_AAA_VSM_TABLE, self.ring_vsm_table[slot])
            self.vk.write_descriptor(ds, BIND_AAA_SURFCACHE, self.ring_surfcache[slot])
            self.vk.write_descriptor(ds, BIND_AAA_OIT_NODES, self.buf_oit_nodes)
            self.vk.write_descriptor(ds, BIND_AAA_CLUSTERS, self.ring_clusters[slot])
            self.vk.write_descriptor(ds, BIND_AAA_SKINNING, self.buf_skinning)
            self.vk.write_descriptor(ds, BIND_AAA_POST_SCRATCH, self.buf_post_scratch)
            self.vk.write_descriptor(ds, BIND_AAA_DRAW_PAIRS, self.ring_draw_pairs[slot])

            self.vk.write_aaa_texture(ds, BIND_AAA_VIS_TEX, 0, self.samp_nearest, self.vis_img.view)
            self.vk.write_aaa_texture(ds, BIND_AAA_DEPTH_TEX, 0, self.samp_linear_clamp, self.depth_img.view)
            self.vk.write_aaa_texture(ds, BIND_AAA_VSM_TEX, 0, self.samp_linear_clamp, self.vsm_img.view)
            self.vk.write_aaa_texture(ds, BIND_AAA_ENV_CUBE, 0, self.samp_linear_clamp, self.env_cube_img.view)
            self.vk.write_aaa_texture(ds, BIND_AAA_BRDF_LUT, 0, self.samp_linear_clamp, self.brdf_lut_img.view)
            self.vk.write_aaa_texture(ds, BIND_AAA_LTC_LUT, 0, self.samp_linear_clamp, self.ltc_lut_img.view)

            var ds_p1 = self.desc_sets_cull_p1[slot]
            self.vk.write_descriptor(ds_p1, BIND_AAA_VERTICES, self.buf_vertices)
            self.vk.write_descriptor(ds_p1, BIND_AAA_INDICES, self.buf_indices)
            self.vk.write_descriptor(ds_p1, BIND_AAA_MESHLETS, self.buf_meshlets)
            self.vk.write_descriptor(ds_p1, BIND_AAA_INSTANCES, self.ring_instances[slot])
            self.vk.write_descriptor(ds_p1, BIND_AAA_MATERIALS, self.buf_materials)
            self.vk.write_descriptor(ds_p1, BIND_AAA_INDIRECT, self.ring_indirect[slot])
            self.vk.write_descriptor(ds_p1, BIND_AAA_GLOBALS, self.ring_globals[slot])
            self.vk.write_descriptor(ds_p1, BIND_AAA_WORKLIST, self.ring_meshlet_worklist[slot])
            self.vk.write_descriptor(ds_p1, BIND_AAA_CANDIDATES, self.buf_occluded_candidates)
            self.vk.write_descriptor(ds_p1, BIND_AAA_DRAW_PAIRS, self.ring_draw_pairs[slot])
            self.vk.write_aaa_texture(ds_p1, BIND_AAA_DEPTH_TEX, 0, self.samp_nearest, self.hiz_img.view)

            var ds_p2 = self.desc_sets_cull_p2[slot]
            self.vk.write_descriptor(ds_p2, BIND_AAA_VERTICES, self.buf_vertices)
            self.vk.write_descriptor(ds_p2, BIND_AAA_INDICES, self.buf_indices)
            self.vk.write_descriptor(ds_p2, BIND_AAA_MESHLETS, self.buf_meshlets)
            self.vk.write_descriptor(ds_p2, BIND_AAA_INSTANCES, self.ring_instances[slot])
            self.vk.write_descriptor(ds_p2, BIND_AAA_MATERIALS, self.buf_materials)
            self.vk.write_descriptor(ds_p2, BIND_AAA_INDIRECT, self.buf_indirect_phase2)
            self.vk.write_descriptor(ds_p2, BIND_AAA_GLOBALS, self.ring_globals[slot])
            self.vk.write_descriptor(ds_p2, BIND_AAA_WORKLIST, self.ring_meshlet_worklist[slot])
            self.vk.write_descriptor(ds_p2, BIND_AAA_CANDIDATES, self.buf_occluded_candidates)
            self.vk.write_descriptor(ds_p2, BIND_AAA_DRAW_PAIRS, self.ring_draw_pairs[slot])
            self.vk.write_aaa_texture(ds_p2, BIND_AAA_DEPTH_TEX, 0, self.samp_nearest, self.hiz_img.view)
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

        self._wait_cmd_slot0()
        if not self.vk.begin_commands(0):
            return False
        # Zero-initialize DEVICE_LOCAL persistent SSBOs on GPU timeline
        self.vk.fill_buffer(0, self.buf_history, 0, self.buf_history.size, 0)
        self.vk.fill_buffer(0, self.buf_oit_nodes, 0, self.buf_oit_nodes.size, 0)
        self.vk.fill_buffer(0, self.buf_skinning, 0, self.buf_skinning.size, 0)
        self.vk.fill_buffer(0, self.buf_post_scratch, 0, self.buf_post_scratch.size, 0)
        for slot in range(FRAMES_IN_FLIGHT):
            self.vk.fill_buffer(0, self.ring_globals[slot], 0, self.ring_globals[slot].size, 0)
            self.vk.fill_buffer(0, self.ring_clusters[slot], 0, self.ring_clusters[slot].size, 0)
            self.vk.fill_buffer(0, self.ring_surfcache[slot], 0, self.ring_surfcache[slot].size, 0)
        self.vk.memory_barrier2(
            0, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_TRANSFER | PIPELINE_COMPUTE, Int(ACCESS_TRANSFER_READ | ACCESS_SHADER_READ)
        )

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
        self.vk.transition_image(0, self.hiz_img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        var hz_w = self.render_w
        var hz_h = self.render_h
        for m in range(self.hiz_mip_levels):
            self.vk.upload_image_mip(0, self.hiz_img, self.buf_post_scratch, AAA_POST_BLOOM_BASE * 4, m, hz_w, hz_h)
            hz_w = max(1, hz_w >> 1)
            hz_h = max(1, hz_h >> 1)
        self.vk.transition_image(0, self.hiz_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        self.hiz_valid = False

        self.vk.transition_image(0, self.vsm_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_TOP, 0, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE)
        self.vk.begin_rendering(0, 0, 0, VSM_PHYS_ATLAS_RES, VSM_PHYS_ATLAS_RES, False, 0, False, 0, 0, True, self.vsm_img.view, True, 1.0)
        self.vk.end_rendering(0)
        self.vk.transition_image(0, self.vsm_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)

        # Upload initial UI font atlas if valid
        if self.ui_font.valid and self.ui_font_atlas_slot >= 0 and self.bindless_images[self.ui_font_atlas_slot].image != 0:
            var atlas_bytes = len(self.ui_font.atlas)
            var font_st_off = (ltc_st_off + len(ltc_bytes) + 63) & ~63
            self.ensure_staging_capacity(font_st_off + atlas_bytes)
            _ = external_call["memcpy", Int](self.buf_staging.cpu_address + font_st_off, Int(self.ui_font.atlas.unsafe_ptr()), atlas_bytes)
            self.vk.transition_image(0, self.bindless_images[self.ui_font_atlas_slot], LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
            self.vk.upload_image_subregion(0, self.bindless_images[self.ui_font_atlas_slot], self.buf_staging, font_st_off, 0, 0, self.ui_font.atlas_width, self.ui_font.atlas_height)
            self.vk.transition_image(0, self.bindless_images[self.ui_font_atlas_slot], LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
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

        # Upload default BC6H Golden-Hour / Sky IBL + White dummy BC7 texture at bindless[1]
        self.upload_ibl_environment(
            bake_bc6h_ibl_cubemap_and_sh9(
                SkyDomeParams(uniform_furnace=False),
                base_dim=self.env_cube_dim,
                mip_levels=self.env_mip_levels,
            )
        )
        var font_view = self.bindless_images[self.ui_font_atlas_slot].view if self.ui_font_atlas_slot >= 0 else 0
        if font_view != 0:
            for ds_i in range(len(self.desc_sets)):
                self.vk.write_aaa_texture(
                    self.desc_sets[ds_i],
                    BIND_AAA_BINDLESS_TEX,
                    self.ui_font_atlas_slot,
                    self.samp_linear_clamp,
                    font_view,
                )
        var white_px = List[UInt32](length=16, fill=0xFFFFFFFF)
        var white_slot = self.upload_bc7_texture(white_px, 4, 4, srgb=False, generate_mips=False)
        self.fallback_white_slot = white_slot
        if white_slot >= 0 and white_slot < len(self.bindless_images):
            var fallback_view = self.bindless_images[white_slot].view
            if font_view == 0 and self.ui_font_atlas_slot >= 0:
                for ds_i in range(len(self.desc_sets)):
                    self.vk.write_aaa_texture(
                        self.desc_sets[ds_i],
                        BIND_AAA_BINDLESS_TEX,
                        self.ui_font_atlas_slot,
                        self.samp_aniso16,
                        fallback_view,
                    )
            var prefill_end = min(256, AAA_MAX_BINDLESS)
            for s_idx in range(len(self.bindless_images), prefill_end):
                for ds_i in range(len(self.desc_sets)):
                    self.vk.write_aaa_texture(self.desc_sets[ds_i], BIND_AAA_BINDLESS_TEX, s_idx, self.samp_aniso16, fallback_view)

        self._bind_render_graph_persistent_buffers(0)
        return True

    def resize(mut self, width: Int, height: Int) -> Bool:
        """Recreates all resolution-dependent render targets, aliased heaps, Hi-Z pyramid, and storage buffers cleanly."""
        if not self.active or self.vk.device == 0:
            return False
        var new_out_w = max(16, width)
        var new_out_h = max(16, height)
        var new_ren_w = max(8, Int(Float32(new_out_w) * self.tsr_scale + Float32(0.5)))
        var new_ren_h = max(8, Int(Float32(new_out_h) * self.tsr_scale + Float32(0.5)))
        if (
            new_out_w == self.output_w
            and new_out_h == self.output_h
            and new_ren_w == self.render_w
            and new_ren_h == self.render_h
        ):
            self.swapchain_out_of_date = False
            return True

        self.vk.wait_idle()
        for i in range(len(self.frame_fences)):
            self.frame_fences[i] = 0
            self._flush_slot_retirements(i)
        self.prev_submitted_fence = 0
        for i in range(len(self.slot_has_pending_queries)):
            self.slot_has_pending_queries[i] = False
            self.slot_prof_pairs[i].clear()
            self.slot_frame_q[i] = -1
        self.pending_slot = -1
        self.pending_prof_pairs.clear()
        self.pending_frame_q = -1
        self.readback_ready_slot = 0

        self._free_aliased_targets()
        self.vk.destroy_image(self.hiz_img)
        self.vk.destroy_buffer(self.buf_history)
        self.vk.destroy_buffer(self.buf_oit_nodes)
        self.vk.destroy_buffer(self.buf_post_scratch)
        self.vk.destroy_buffer(self.buf_ldr_readback)
        for i in range(len(self.readback_extra)):
            self.vk.destroy_buffer(self.readback_extra[i])
        self.readback_extra.clear()

        self.output_w = new_out_w
        self.output_h = new_out_h
        self.render_w = new_ren_w
        self.render_h = new_ren_h
        var max_pix = max(self.output_w * self.output_h, self.render_w * self.render_h)

        if not self._allocate_and_bind_aliased_targets():
            return False
        self.hiz_mip_levels = compute_hiz_mip_levels(self.render_w, self.render_h)
        self.hiz_img = self.vk.create_image_2d(
            self.render_w,
            self.render_h,
            FORMAT_R32_SFLOAT,
            IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED,
            mip_levels=self.hiz_mip_levels,
        )
        if self.vis_img.image == 0 or self.depth_img.image == 0 or self.hiz_img.image == 0:
            return False

        var hist_words = rt_shadow_raw_offset(max_pix) + rt_shadow_total_words(max_pix)
        self.buf_history = self.vk.create_buffer_device_local(hist_words * 4)
        self.buf_oit_nodes = self.vk.create_buffer_device_local(gpu_particles_total_oit_words(max_pix) * 4)
        var scratch_words = AAA_POST_UI_PRIMS_BASE_OFFSET + max_pix * 10 + FRAMES_IN_FLIGHT * AAA_UI_PRIM_REGION_WORDS
        self.buf_post_scratch = self.vk.create_buffer_device_local(scratch_words * 4)
        self.buf_ldr_readback = self.vk.create_buffer(64 + self.output_w * self.output_h * 4, local=False, cpu_reads=True)

        if self.buf_ldr_readback.cpu_address != 0:
            _ = external_call["memset", Int](self.buf_ldr_readback.cpu_address, Int32(0), self.buf_ldr_readback.size)

        for slot in range(FRAMES_IN_FLIGHT):
            var ds = self.desc_sets[slot]
            self.vk.write_descriptor(ds, BIND_AAA_HDR_OUT, self.buf_hdr_out)
            self.vk.write_descriptor(ds, BIND_AAA_HISTORY, self.buf_history)
            self.vk.write_descriptor(ds, BIND_AAA_OIT_NODES, self.buf_oit_nodes)
            self.vk.write_descriptor(ds, BIND_AAA_POST_SCRATCH, self.buf_post_scratch)
            self.vk.write_aaa_texture(ds, BIND_AAA_VIS_TEX, 0, self.samp_nearest, self.vis_img.view)
            self.vk.write_aaa_texture(ds, BIND_AAA_DEPTH_TEX, 0, self.samp_linear_clamp, self.depth_img.view)
            self.vk.write_descriptor(self.desc_sets_cull_p1[slot], BIND_AAA_CANDIDATES, self.buf_occluded_candidates)
            self.vk.write_descriptor(self.desc_sets_cull_p2[slot], BIND_AAA_INDIRECT, self.buf_indirect_phase2)
            self.vk.write_descriptor(self.desc_sets_cull_p2[slot], BIND_AAA_CANDIDATES, self.buf_occluded_candidates)
            self.vk.write_aaa_texture(self.desc_sets_cull_p1[slot], BIND_AAA_DEPTH_TEX, 0, self.samp_nearest, self.hiz_img.view)
            self.vk.write_aaa_texture(self.desc_sets_cull_p2[slot], BIND_AAA_DEPTH_TEX, 0, self.samp_nearest, self.hiz_img.view)

        self._wait_cmd_slot0()
        if not self.vk.begin_commands(0):
            return False
        self.vk.fill_buffer(0, self.buf_history, 0, self.buf_history.size, 0)
        self.vk.fill_buffer(0, self.buf_oit_nodes, 0, self.buf_oit_nodes.size, 0)
        self.vk.fill_buffer(0, self.buf_post_scratch, 0, self.buf_post_scratch.size, 0)
        self.vk.memory_barrier2(
            0, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_TRANSFER | PIPELINE_COMPUTE, Int(ACCESS_TRANSFER_READ | ACCESS_SHADER_READ)
        )
        self.vk.transition_image(0, self.vis_img, LAYOUT_COLOR_ATTACHMENT, PIPELINE_TOP, 0, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE)
        self.vk.transition_image(0, self.depth_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_TOP, 0, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE)
        self.vk.begin_rendering(0, 0, 0, self.render_w, self.render_h, True, self.vis_img.view, True, 0, 0, True, self.depth_img.view, True, 0.0)
        self.vk.end_rendering(0)
        self.vk.transition_image(0, self.vis_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        self.vk.transition_image(0, self.depth_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        self.vk.transition_image(0, self.hiz_img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        var hz_w = self.render_w
        var hz_h = self.render_h
        for m in range(self.hiz_mip_levels):
            self.vk.upload_image_mip(0, self.hiz_img, self.buf_post_scratch, AAA_POST_BLOOM_BASE * 4, m, hz_w, hz_h)
            hz_w = max(1, hz_w >> 1)
            hz_h = max(1, hz_h >> 1)
        self.vk.transition_image(0, self.hiz_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)
        var rf = self.vk.submit(0)
        if rf == 0 or not self.vk.wait(rf):
            return False

        self.has_prev_vp = False
        self.spec_reblur_valid_frames = 0
        self.froxel_valid_frames = 0
        self.hiz_valid = False
        self.swapchain_out_of_date = False
        self._bind_render_graph_persistent_buffers(0)
        return True

    def _record_hiz_downsample(mut self, cur_slot: Int):
        """Builds the multi-mip conservative Reverse-Z min-reduction Hi-Z pyramid (`hiz_img`) from `depth_img`
        (which must already be in `LAYOUT_SHADER_READ_ONLY`), using `buf_post_scratch[AAA_POST_BLOOM_BASE..]`."""
        _ = self._ensure_hiz_pipeline()
        if self.pipe_hiz_downsample == 0 or self.hiz_img.image == 0 or self.hiz_mip_levels <= 0:
            return
        var mip_offsets = List[Int](capacity=self.hiz_mip_levels)
        var mip_ws = List[Int](capacity=self.hiz_mip_levels)
        var mip_hs = List[Int](capacity=self.hiz_mip_levels)
        var cur_off = AAA_POST_BLOOM_BASE
        var mw = self.render_w
        var mh = self.render_h
        for m in range(self.hiz_mip_levels):
            mip_offsets.append(cur_off)
            mip_ws.append(mw)
            mip_hs.append(mh)
            var prev_w = mw if m == 0 else mip_ws[m - 1]
            var prev_h = mh if m == 0 else mip_hs[m - 1]
            var prev_off = cur_off if m == 0 else mip_offsets[m - 1]
            var push_hz = List[UInt32](length=16, fill=0)
            push_hz[0] = UInt32(mw)
            push_hz[1] = UInt32(mh)
            push_hz[2] = UInt32(prev_w)
            push_hz[3] = UInt32(prev_h)
            push_hz[4] = UInt32(prev_off)
            push_hz[5] = UInt32(cur_off)
            push_hz[6] = UInt32(m)
            self.vk.dispatch_aaa(
                cur_slot,
                self.pipe_hiz_downsample,
                self.pipe_layout,
                self.desc_set,
                push_hz,
                (mw + 7) // 8,
                (mh + 7) // 8,
            )
            if m + 1 < self.hiz_mip_levels:
                self.vk.memory_barrier2(
                    cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ)
                )
            cur_off += mw * mh
            mw = max(1, mw >> 1)
            mh = max(1, mh >> 1)

        self.vk.memory_barrier2(
            cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ)
        )
        self.vk.transition_image(
            cur_slot, self.hiz_img, LAYOUT_TRANSFER_TARGET, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE
        )
        for m in range(self.hiz_mip_levels):
            self.vk.upload_image_mip(
                cur_slot, self.hiz_img, self.buf_post_scratch, mip_offsets[m] * 4, m, mip_ws[m], mip_hs[m]
            )
        self.vk.transition_image(
            cur_slot, self.hiz_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
        )
        self.hiz_valid = True

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
        self._wait_all_in_flight()
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

    def _ensure_bindless_meta(mut self, slot: Int):
        while len(self.bindless_resident_min_mip) <= slot:
            self.bindless_resident_min_mip.append(16)
            self.bindless_vram_bytes.append(0)
            self.bindless_layouts_init.append(False)

    def _flush_slot_retirements(mut self, slot: Int):
        if slot < 0:
            return
        if slot < len(self.pending_bindless_desc_writes) and slot < len(self.desc_sets):
            for wi in range(len(self.pending_bindless_desc_writes[slot])):
                var w = self.pending_bindless_desc_writes[slot][wi]
                if w[1] != 0 and w[2] != 0:
                    self.vk.write_aaa_texture(self.desc_sets[slot], BIND_AAA_BINDLESS_TEX, w[0], w[1], w[2])
            self.pending_bindless_desc_writes[slot].clear()
        if slot < len(self.deferred_destroy_images):
            for di in range(len(self.deferred_destroy_images[slot])):
                self.vk.destroy_image(self.deferred_destroy_images[slot][di])
            self.deferred_destroy_images[slot].clear()

    def _write_or_queue_bindless(mut self, bindless_slot: Int, sampler: Int, view: Int, cur_slot: Int = -1):
        _ = cur_slot
        if view == 0 or sampler == 0:
            return
        for ds_i in range(len(self.desc_sets)):
            var in_flight = ds_i < len(self.frame_fences) and self.frame_fences[ds_i] != 0
            if in_flight and ds_i < len(self.pending_bindless_desc_writes):
                var replaced = False
                for wi in range(len(self.pending_bindless_desc_writes[ds_i])):
                    if self.pending_bindless_desc_writes[ds_i][wi][0] == bindless_slot:
                        self.pending_bindless_desc_writes[ds_i][wi] = (bindless_slot, sampler, view)
                        replaced = True
                        break
                if not replaced:
                    self.pending_bindless_desc_writes[ds_i].append((bindless_slot, sampler, view))
            else:
                if ds_i < len(self.pending_bindless_desc_writes):
                    var wi = len(self.pending_bindless_desc_writes[ds_i]) - 1
                    while wi >= 0:
                        if self.pending_bindless_desc_writes[ds_i][wi][0] == bindless_slot:
                            var last = self.pending_bindless_desc_writes[ds_i].pop()
                            if wi < len(self.pending_bindless_desc_writes[ds_i]):
                                self.pending_bindless_desc_writes[ds_i][wi] = last
                        wi -= 1
                self.vk.write_aaa_texture(self.desc_sets[ds_i], BIND_AAA_BINDLESS_TEX, bindless_slot, sampler, view)

    def allocate_bindless_texture_slot(
        mut self,
        width: Int,
        height: Int,
        mip_count: Int,
        vk_format: UInt32,
    ) -> Int:
        """Allocates a `VulkanImage` in `bindless_images` (reusing `bindless_free_slots`) for staged mip streaming."""
        if self.vk.device == 0:
            return -1
        var eff_w = max(1, width)
        var eff_h = max(1, height)
        var eff_mips = max(1, min(16, mip_count))
        var img = self.vk.create_image_2d(
            eff_w,
            eff_h,
            vk_format,
            IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED,
            mip_levels=eff_mips,
        )
        if img.image == 0:
            return -1
        var vram_sz = img.memory_bytes
        var slot: Int
        if len(self.bindless_free_slots) > 0:
            slot = self.bindless_free_slots.pop()
            self.bindless_images[slot] = img^
        else:
            if len(self.bindless_images) >= AAA_MAX_BINDLESS:
                self.vk.destroy_image(img)
                return -1
            slot = len(self.bindless_images)
            self.bindless_images.append(img^)
        self._ensure_bindless_meta(slot)
        self.bindless_resident_min_mip[slot] = eff_mips
        self.bindless_vram_bytes[slot] = vram_sz
        self.bindless_layouts_init[slot] = False
        return slot

    def upload_bindless_mip_staged(
        mut self,
        bindless_slot: Int,
        mip_level: Int,
        mip_w: Int,
        mip_h: Int,
        data_ptr: Int,
        data_bytes: Int,
        cur_slot: Int = -1,
        cmd_slot: Int = -1,
    ) -> Int:
        """Stages a single mip into `bindless_images[bindless_slot]` via `LinearStagingRing`, updates `minLod` sampler & descriptors."""
        if self.vk.device == 0 or bindless_slot < 0 or bindless_slot >= len(self.bindless_images):
            return 0
        if self.bindless_images[bindless_slot].image == 0:
            return 0
        self._ensure_bindless_meta(bindless_slot)

        var eff_slot = cur_slot if (cur_slot >= 0 and cur_slot < FRAMES_IN_FLIGHT) else (self.frame_index % FRAMES_IN_FLIGHT)
        var sync_submit = cmd_slot < 0
        if sync_submit and len(self.vk.command_buffers) == 0:
            if not self.vk.allocate_command_buffers(FRAMES_IN_FLIGHT):
                return 0
        var eff_cmd = (eff_slot if eff_slot < len(self.vk.command_buffers) else 0) if sync_submit else cmd_slot
        if sync_submit:
            if eff_slot < len(self.frame_fences) and self.frame_fences[eff_slot] != 0:
                _ = self.vk.wait(self.frame_fences[eff_slot])
                if self.prev_submitted_fence <= self.frame_fences[eff_slot]:
                    self.prev_submitted_fence = 0
                self.frame_fences[eff_slot] = 0
                self.staging_ring.reset_slot(self.vk, eff_slot)
                self._flush_slot_retirements(eff_slot)
            if eff_cmd == 0 and eff_slot != 0:
                self._wait_cmd_slot0()
            if not self.vk.begin_commands(eff_cmd):
                return 0

        var eff_mips = max(1, self.bindless_images[bindless_slot].mip_levels)
        var eff_mip = max(0, min(eff_mips - 1, mip_level))
        var stage_bytes = max(16, data_bytes)
        var sl = self.staging_ring.alloc(self.vk, eff_slot, stage_bytes, 256)
        if sl.size <= 0 or sl.cpu_address == 0:
            if sync_submit:
                var f_abort = self.vk.submit(eff_cmd)
                if f_abort != 0:
                    _ = self.vk.wait(f_abort)
            return 0
        if data_ptr != 0 and data_bytes > 0:
            var copy_len = min(stage_bytes, data_bytes)
            _ = external_call["memcpy", Int](sl.cpu_address, data_ptr, copy_len)
            if copy_len < stage_bytes:
                _ = external_call["memset", Int](sl.cpu_address + copy_len, Int32(0), stage_bytes - copy_len)
        else:
            _ = external_call["memset", Int](sl.cpu_address, Int32(0x7F), stage_bytes)

        if not self.bindless_layouts_init[bindless_slot]:
            self.vk.transition_image_mip_range(
                eff_cmd,
                self.bindless_images[bindless_slot],
                0,
                eff_mips,
                LAYOUT_UNDEFINED,
                LAYOUT_SHADER_READ_ONLY,
                PIPELINE_TOP,
                0,
                PIPELINE_COMPUTE,
                ACCESS_SHADER_READ,
            )
            self.bindless_layouts_init[bindless_slot] = True
            self.bindless_images[bindless_slot].layout = LAYOUT_SHADER_READ_ONLY

        self.vk.transition_image_mip_range(
            eff_cmd,
            self.bindless_images[bindless_slot],
            eff_mip,
            1,
            LAYOUT_SHADER_READ_ONLY,
            LAYOUT_TRANSFER_TARGET,
            PIPELINE_COMPUTE | PIPELINE_FRAGMENT_SHADER,
            ACCESS_SHADER_READ,
            PIPELINE_TRANSFER,
            ACCESS_TRANSFER_WRITE,
        )
        self.staging_ring.upload_to_image_mip(
            self.vk,
            eff_cmd,
            eff_slot,
            sl,
            self.bindless_images[bindless_slot],
            eff_mip,
            max(1, mip_w),
            max(1, mip_h),
        )
        self.vk.transition_image_mip_range(
            eff_cmd,
            self.bindless_images[bindless_slot],
            eff_mip,
            1,
            LAYOUT_TRANSFER_TARGET,
            LAYOUT_SHADER_READ_ONLY,
            PIPELINE_TRANSFER,
            ACCESS_TRANSFER_WRITE,
            PIPELINE_COMPUTE | PIPELINE_FRAGMENT_SHADER,
            ACCESS_SHADER_READ,
        )

        if sync_submit:
            var f = self.vk.submit(eff_cmd)
            if f != 0:
                _ = self.vk.wait(f)

        if eff_mip < self.bindless_resident_min_mip[bindless_slot]:
            self.bindless_resident_min_mip[bindless_slot] = eff_mip
        var min_mip = max(0, min(15, self.bindless_resident_min_mip[bindless_slot]))
        var samp = self.samp_aniso16
        if min_mip < len(self.samp_aniso16_min_lod) and self.samp_aniso16_min_lod[min_mip] != 0:
            samp = self.samp_aniso16_min_lod[min_mip]
        var view = self.bindless_images[bindless_slot].view
        self._write_or_queue_bindless(bindless_slot, samp, view, eff_slot)

        var vram_sz = max(stage_bytes, self.bindless_images[bindless_slot].memory_bytes)
        self.bindless_vram_bytes[bindless_slot] = vram_sz
        return vram_sz

    def evict_bindless_texture_slot(mut self, bindless_slot: Int):
        """Rebinds `bindless_slot` to fallback white and destroys/retires the `VulkanImage` safely across frames in flight."""
        if bindless_slot < 0 or bindless_slot >= len(self.bindless_images):
            return
        if self.bindless_images[bindless_slot].image == 0:
            return
        self._ensure_bindless_meta(bindless_slot)
        var cur_slot = self.frame_index % FRAMES_IN_FLIGHT
        var fb_view = 0
        if self.fallback_white_slot >= 0 and self.fallback_white_slot < len(self.bindless_images):
            fb_view = self.bindless_images[self.fallback_white_slot].view
        if fb_view != 0 and self.samp_aniso16 != 0:
            self._write_or_queue_bindless(bindless_slot, self.samp_aniso16, fb_view, cur_slot)

        var retire_slot = -1
        var max_f: UInt64 = 0
        for i in range(len(self.frame_fences)):
            if self.frame_fences[i] != 0 and self.frame_fences[i] >= max_f:
                max_f = self.frame_fences[i]
                retire_slot = i
        if retire_slot >= 0 and retire_slot < len(self.deferred_destroy_images):
            var retired = VulkanImage()
            swap(retired, self.bindless_images[bindless_slot])
            self.deferred_destroy_images[retire_slot].append(retired^)
        elif self.vk.device != 0:
            self.vk.destroy_image(self.bindless_images[bindless_slot])
        else:
            self.bindless_images[bindless_slot] = VulkanImage()

        self.bindless_resident_min_mip[bindless_slot] = 16
        self.bindless_vram_bytes[bindless_slot] = 0
        self.bindless_layouts_init[bindless_slot] = False
        self.bindless_free_slots.append(bindless_slot)

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

        self._wait_cmd_slot0()
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
        var view = img.view
        var vram_sz = img.memory_bytes
        self.bindless_images.append(img^)
        self._ensure_bindless_meta(slot)
        self.bindless_resident_min_mip[slot] = 0
        self.bindless_vram_bytes[slot] = vram_sz
        self.bindless_layouts_init[slot] = True
        self._write_or_queue_bindless(slot, self.samp_aniso16, view)
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

        self._wait_cmd_slot0()
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
        var view = img.view
        var vram_sz = img.memory_bytes
        self.bindless_images.append(img^)
        self._ensure_bindless_meta(slot)
        self.bindless_resident_min_mip[slot] = 0
        self.bindless_vram_bytes[slot] = vram_sz
        self.bindless_layouts_init[slot] = True
        self._write_or_queue_bindless(slot, self.samp_aniso16, view)
        return slot

    def upload_bc6h_texture(mut self, hdr_pixels: List[Vec3], width: Int, height: Int) -> Int:
        if len(self.bindless_images) >= AAA_MAX_BINDLESS:
            return -1
        var img = self.vk.create_image_2d(width, height, FORMAT_BC6H_UFLOAT, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED, mip_levels=1)
        if img.image == 0:
            return -1
        var bc6 = encode_bc6h_hdr(hdr_pixels, width, height)
        self._wait_cmd_slot0()
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
        var view = img.view
        var vram_sz = img.memory_bytes
        self.bindless_images.append(img^)
        self._ensure_bindless_meta(slot)
        self.bindless_resident_min_mip[slot] = 0
        self.bindless_vram_bytes[slot] = vram_sz
        self.bindless_layouts_init[slot] = True
        self._write_or_queue_bindless(slot, self.samp_linear_clamp, view)
        return slot

    def upload_rgba16f_texture(mut self, rgba16f: List[UInt16], width: Int, height: Int) -> Int:
        """Uploads raw packed `RGBA16F` texels (`len >= width * height * 4` `UInt16` half-floats)
        into `BIND_AAA_BINDLESS_TEX` (`FORMAT_R16G16B16A16_SFLOAT` + `samp_linear_clamp`)."""
        if len(self.bindless_images) >= AAA_MAX_BINDLESS or width <= 0 or height <= 0:
            return -1
        var need_u16 = width * height * 4
        if len(rgba16f) < need_u16:
            return -1
        var img = self.vk.create_image_2d(
            width, height, FORMAT_R16G16B16A16_SFLOAT, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED, mip_levels=1
        )
        if img.image == 0:
            return -1
        var byte_sz = need_u16 * 2
        self._wait_cmd_slot0()
        self.ensure_staging_capacity(byte_sz)
        _ = external_call["memcpy", Int](self.buf_staging.cpu_address, Int(rgba16f.unsafe_ptr()), byte_sz)
        _ = len(rgba16f)
        if not self.vk.begin_commands(0):
            self.vk.destroy_image(img)
            return -1
        self.vk.transition_image(0, img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        self.vk.upload_image_mip(0, img, self.buf_staging, 0, 0, width, height)
        self.vk.transition_image(
            0, img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
        )
        var f = self.vk.submit(0)
        if f != 0:
            _ = self.vk.wait(f)
        var slot = len(self.bindless_images)
        var view = img.view
        var vram_sz = img.memory_bytes
        self.bindless_images.append(img^)
        self._ensure_bindless_meta(slot)
        self.bindless_resident_min_mip[slot] = 0
        self.bindless_vram_bytes[slot] = vram_sz
        self.bindless_layouts_init[slot] = True
        self._write_or_queue_bindless(slot, self.samp_linear_clamp, view)
        return slot

    def upload_rgba8_texture(mut self, rgba8: List[UInt8], width: Int, height: Int) -> Int:
        """Uploads raw `RGBA8_UNORM` texels (`len >= width * height * 4`) into `BIND_AAA_BINDLESS_TEX`."""
        if len(self.bindless_images) >= AAA_MAX_BINDLESS or width <= 0 or height <= 0:
            return -1
        var need_u8 = width * height * 4
        if len(rgba8) < need_u8:
            return -1
        var img = self.vk.create_image_2d(
            width, height, FORMAT_R8G8B8A8_UNORM, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED, mip_levels=1
        )
        if img.image == 0:
            return -1
        self._wait_cmd_slot0()
        self.ensure_staging_capacity(need_u8)
        _ = external_call["memcpy", Int](self.buf_staging.cpu_address, Int(rgba8.unsafe_ptr()), need_u8)
        _ = len(rgba8)
        if not self.vk.begin_commands(0):
            self.vk.destroy_image(img)
            return -1
        self.vk.transition_image(0, img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        self.vk.upload_image_mip(0, img, self.buf_staging, 0, 0, width, height)
        self.vk.transition_image(
            0, img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
        )
        var f = self.vk.submit(0)
        if f != 0:
            _ = self.vk.wait(f)
        var slot = len(self.bindless_images)
        var view = img.view
        var vram_sz = img.memory_bytes
        self.bindless_images.append(img^)
        self._ensure_bindless_meta(slot)
        self.bindless_resident_min_mip[slot] = 0
        self.bindless_vram_bytes[slot] = vram_sz
        self.bindless_layouts_init[slot] = True
        self._write_or_queue_bindless(slot, self.samp_linear_clamp, view)
        return slot

    def upload_terrain_heightmap_rgba16f(mut self, heights: List[Float32], width: Int, height: Int) -> Int:
        """Packs scalar `[0..1]` (or normalized) terrain height samples into `RGBA16F` `(h, 0.5, 0.5, 1.0)`
        and uploads to `BIND_AAA_BINDLESS_TEX` (`samp_linear_clamp`)."""
        if width <= 0 or height <= 0 or len(heights) < width * height:
            return -1
        var texels = width * height
        var packed = List[UInt16](length=texels * 4, fill=0)
        var half_h = f32_to_f16_bits(Float32(0.5))
        var one_h = f32_to_f16_bits(Float32(1.0))
        for i in range(texels):
            packed[i * 4 + 0] = f32_to_f16_bits(heights[i])
            packed[i * 4 + 1] = half_h
            packed[i * 4 + 2] = half_h
            packed[i * 4 + 3] = one_h
        return self.upload_rgba16f_texture(packed, width, height)

    def add_material(mut self, mat: AaaMaterial) -> Int:
        var id = len(self.materials)
        var needed_bytes = (id + 1) * AAA_MATERIAL_WORDS * 4
        if not self._grow_storage_buffer(needed_bytes, BIND_AAA_MATERIALS, 0, False, True):
            return -1
        self.materials.append(mat.copy())
        var words = List[UInt32]()
        mat.append_words(words)
        self._upload_to_device_local_sync(Int(words.unsafe_ptr()), len(words) * 4, BIND_AAA_MATERIALS, id * AAA_MATERIAL_WORDS * 4)
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
        return False

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

    def update_material(mut self, mat_id: Int, mat: AaaMaterial):
        if mat_id < 0 or mat_id >= len(self.materials):
            return
        self.materials[mat_id] = mat.copy()
        var words = List[UInt32]()
        mat.append_words(words)
        self._wait_all_in_flight()
        self._upload_to_device_local_sync(Int(words.unsafe_ptr()), len(words) * 4, BIND_AAA_MATERIALS, mat_id * AAA_MATERIAL_WORDS * 4)
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

        var rb_mg = self.vk.create_buffer(out_words * 4, local=False, cpu_reads=True)
        if rb_mg.vk_buffer == 0 or rb_mg.cpu_address == 0:
            if rb_mg.size > 0:
                self.vk.destroy_buffer(rb_mg)
            self.vk.destroy_pipeline(pipe)
            return (-1, -1)

        self._wait_all_in_flight()
        if not self.vk.begin_commands(0):
            self.vk.destroy_buffer(rb_mg)
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
            0, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ)
        )
        self.vk.copy_buffer(0, self.buf_post_scratch, out_word_off * 4, rb_mg, 0, out_words * 4)
        self.vk.memory_barrier2(
            0, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ)
        )
        var f_mg = self.vk.submit(0)
        if f_mg != 0:
            _ = self.vk.wait(f_mg)
        self.vk.destroy_pipeline(pipe)
        if f_mg == 0:
            self.vk.destroy_buffer(rb_mg)
            return (-1, -1)

        var eval_f = List[Float32](length=out_words, fill=0.0)
        _ = external_call[
            "memcpy", Int
        ](Int(eval_f.unsafe_ptr()), rb_mg.cpu_address, out_words * 4)
        self.vk.destroy_buffer(rb_mg)

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

    def add_mesh(
        mut self,
        mesh: NativeMesh,
        material_id: Int,
        build_lods: Bool = False,
        allow_update: Bool = False,
        expand_bounds: Float32 = 0.0,
    ) -> CookedMeshEntry:
        """Partitions `mesh` into 64v/124t GPU Meshlets (and optional `QEM` `LOD 0..3` DAG), uploads SSBOs, and builds `BLAS`!"""
        var v_words_before = len(self.geom.vertex_words)
        var i_words_before = len(self.geom.index_words)
        var m_words_before = len(self.geom.meshlet_words)

        var entry = self.geom.append_mesh(mesh, material_id, build_lods=build_lods)
        if expand_bounds > Float32(0.0):
            for m_idx in range(entry.first_meshlet, entry.first_meshlet + entry.meshlet_count):
                var w_rad = self.geom.meshlets[m_idx].radius + expand_bounds
                self.geom.meshlets[m_idx].radius = w_rad
                self.geom.meshlets[m_idx].cone_sin = Float32(2.0)
                self.geom.meshlet_words[m_idx * 12 + 3] = bitcast[DType.uint32](w_rad)
                self.geom.meshlet_words[m_idx * 12 + 7] = bitcast[DType.uint32](Float32(2.0))

        var vb_start = v_words_before * 4
        var vb_len = (len(self.geom.vertex_words) - v_words_before) * 4
        var ib_start = i_words_before * 4
        var ib_len = (len(self.geom.index_words) - i_words_before) * 4
        var mb_start = m_words_before * 4
        var mb_len = (len(self.geom.meshlet_words) - m_words_before) * 4
        var ok_vb = self._grow_storage_buffer(vb_start + vb_len, BIND_AAA_VERTICES, BUFFER_USAGE_AS_BUILD_INPUT_KHR, True, True)
        var ok_ib = self._grow_storage_buffer(ib_start + ib_len, BIND_AAA_INDICES, BUFFER_USAGE_INDEX | BUFFER_USAGE_AS_BUILD_INPUT_KHR, True, True)
        var ok_mb = self._grow_storage_buffer(mb_start + mb_len, BIND_AAA_MESHLETS, 0, False, True)
        var grow_ok = ok_vb and ok_ib and ok_mb

        var st_vb = 0
        var st_ib = (vb_len + 15) & ~15
        var st_mb = st_ib + ((ib_len + 15) & ~15)
        var total_st = st_mb + ((mb_len + 15) & ~15)
        var is_am_mesh = (mesh.alpha_mode == 1) or (
            material_id >= 0 and material_id < len(self.materials) and self.materials[material_id].is_alpha_masked()
        )
        var need_blas = self.vk.has_ray_query and entry.triangle_count > 0 and entry.vertex_count > 0
        if grow_ok and (total_st > 0 or need_blas):
            if need_blas:
                self._wait_all_in_flight()
            else:
                self._wait_cmd_slot0()
        if grow_ok and total_st > 0:
            self.ensure_staging_capacity(total_st)
            if vb_len > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_vb, Int(self.geom.vertex_words.unsafe_ptr()) + vb_start, vb_len)
            if ib_len > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_ib, Int(self.geom.index_words.unsafe_ptr()) + ib_start, ib_len)
            if mb_len > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_mb, Int(self.geom.meshlet_words.unsafe_ptr()) + mb_start, mb_len)

        if grow_ok and (total_st > 0 or need_blas) and self.vk.begin_commands(0):
            if vb_len > 0:
                self.vk.copy_buffer(0, self.buf_staging, st_vb, self.buf_vertices, vb_start, vb_len)
            if ib_len > 0:
                self.vk.copy_buffer(0, self.buf_staging, st_ib, self.buf_indices, ib_start, ib_len)
            if mb_len > 0:
                self.vk.copy_buffer(0, self.buf_staging, st_mb, self.buf_meshlets, mb_start, mb_len)
            if total_st > 0:
                self.vk.memory_barrier2(
                    0,
                    PIPELINE_TRANSFER,
                    Int(ACCESS_TRANSFER_WRITE),
                    PIPELINE_ALL_COMMANDS,
                    Int(ACCESS_MEMORY_READ | ACCESS_SHADER_READ | ACCESS_INDIRECT_READ | ACCESS_AS_READ_KHR),
                )
            if need_blas:
                var v_addr = self.vk.buffer_device_address(self.buf_vertices)
                var i_addr = self.vk.buffer_device_address(self.buf_indices) + UInt64(entry.first_index * 4)
                var max_v = max(0, entry.first_vertex + entry.vertex_count - 1)
                var blas = build_blas_triangles_ex(
                    self.vk,
                    0,
                    v_addr,
                    AAA_VERTEX_BYTES,
                    max_v,
                    i_addr,
                    entry.triangle_count,
                    self.buf_as_scratch,
                    allow_update=allow_update,
                    opaque=not is_am_mesh,
                )
                self.vk.memory_barrier2(
                    0,
                    PIPELINE_AS_BUILD_KHR,
                    Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR),
                    PIPELINE_AS_BUILD_KHR | PIPELINE_COMPUTE,
                    Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
                )
                self.mesh_blas.append(blas^)
            else:
                self.mesh_blas.append(VulkanAccelStruct())
            var f = self.vk.submit(0)
            if f != 0:
                _ = self.vk.wait(f)
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
        allow_update: Bool = False,
        expand_bounds: Float32 = 0.0,
    ) -> CookedMeshEntry:
        """Cooks a 4-layer height-modulated PBR terrain splat mesh (`vertex_words[11]` bit 1 = 1 + 24-bit weights),
        uploads GPU Meshlet/Vertex/Index SSBOs, and builds its hardware `BLAS`!"""
        var v_words_before = len(self.geom.vertex_words)
        var i_words_before = len(self.geom.index_words)
        var m_words_before = len(self.geom.meshlet_words)

        var entry = self.geom.append_splat_mesh(mesh, material_id, splat_weights_u24, build_lods=build_lods)
        if expand_bounds > Float32(0.0):
            for m_idx in range(entry.first_meshlet, entry.first_meshlet + entry.meshlet_count):
                var w_rad = self.geom.meshlets[m_idx].radius + expand_bounds
                self.geom.meshlets[m_idx].radius = w_rad
                self.geom.meshlets[m_idx].cone_sin = Float32(2.0)
                self.geom.meshlet_words[m_idx * 12 + 3] = bitcast[DType.uint32](w_rad)
                self.geom.meshlet_words[m_idx * 12 + 7] = bitcast[DType.uint32](Float32(2.0))

        var vb_start = v_words_before * 4
        var vb_len = (len(self.geom.vertex_words) - v_words_before) * 4
        var ib_start = i_words_before * 4
        var ib_len = (len(self.geom.index_words) - i_words_before) * 4
        var mb_start = m_words_before * 4
        var mb_len = (len(self.geom.meshlet_words) - m_words_before) * 4
        var ok_vb = self._grow_storage_buffer(vb_start + vb_len, BIND_AAA_VERTICES, BUFFER_USAGE_AS_BUILD_INPUT_KHR, True, True)
        var ok_ib = self._grow_storage_buffer(ib_start + ib_len, BIND_AAA_INDICES, BUFFER_USAGE_INDEX | BUFFER_USAGE_AS_BUILD_INPUT_KHR, True, True)
        var ok_mb = self._grow_storage_buffer(mb_start + mb_len, BIND_AAA_MESHLETS, 0, False, True)
        var grow_ok = ok_vb and ok_ib and ok_mb

        var st_vb = 0
        var st_ib = (vb_len + 15) & ~15
        var st_mb = st_ib + ((ib_len + 15) & ~15)
        var total_st = st_mb + ((mb_len + 15) & ~15)
        var is_am_mesh = (mesh.alpha_mode == 1) or (
            material_id >= 0 and material_id < len(self.materials) and self.materials[material_id].is_alpha_masked()
        )
        var need_blas = self.vk.has_ray_query and entry.triangle_count > 0 and entry.vertex_count > 0
        if grow_ok and (total_st > 0 or need_blas):
            if need_blas:
                self._wait_all_in_flight()
            else:
                self._wait_cmd_slot0()
        if grow_ok and total_st > 0:
            self.ensure_staging_capacity(total_st)
            if vb_len > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_vb, Int(self.geom.vertex_words.unsafe_ptr()) + vb_start, vb_len)
            if ib_len > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_ib, Int(self.geom.index_words.unsafe_ptr()) + ib_start, ib_len)
            if mb_len > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_mb, Int(self.geom.meshlet_words.unsafe_ptr()) + mb_start, mb_len)

        if grow_ok and (total_st > 0 or need_blas) and self.vk.begin_commands(0):
            if vb_len > 0:
                self.vk.copy_buffer(0, self.buf_staging, st_vb, self.buf_vertices, vb_start, vb_len)
            if ib_len > 0:
                self.vk.copy_buffer(0, self.buf_staging, st_ib, self.buf_indices, ib_start, ib_len)
            if mb_len > 0:
                self.vk.copy_buffer(0, self.buf_staging, st_mb, self.buf_meshlets, mb_start, mb_len)
            if total_st > 0:
                self.vk.memory_barrier2(
                    0,
                    PIPELINE_TRANSFER,
                    Int(ACCESS_TRANSFER_WRITE),
                    PIPELINE_ALL_COMMANDS,
                    Int(ACCESS_MEMORY_READ | ACCESS_SHADER_READ | ACCESS_INDIRECT_READ | ACCESS_AS_READ_KHR),
                )
            if need_blas:
                var v_addr = self.vk.buffer_device_address(self.buf_vertices)
                var i_addr = self.vk.buffer_device_address(self.buf_indices) + UInt64(entry.first_index * 4)
                var max_v = max(0, entry.first_vertex + entry.vertex_count - 1)
                var blas = build_blas_triangles_ex(
                    self.vk,
                    0,
                    v_addr,
                    AAA_VERTEX_BYTES,
                    max_v,
                    i_addr,
                    entry.triangle_count,
                    self.buf_as_scratch,
                    allow_update=allow_update,
                    opaque=not is_am_mesh,
                )
                self.vk.memory_barrier2(
                    0,
                    PIPELINE_AS_BUILD_KHR,
                    Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR),
                    PIPELINE_AS_BUILD_KHR | PIPELINE_COMPUTE,
                    Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
                )
                self.mesh_blas.append(blas^)
            else:
                self.mesh_blas.append(VulkanAccelStruct())
            var f = self.vk.submit(0)
            if f != 0:
                _ = self.vk.wait(f)
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
        self._wait_cmd_slot0()
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
        var nrm_view = nrm_img.view
        var nrm_vram = nrm_img.memory_bytes
        self.bindless_images.append(nrm_img^)
        self._ensure_bindless_meta(nrm_slot)
        self.bindless_resident_min_mip[nrm_slot] = 0
        self.bindless_vram_bytes[nrm_slot] = nrm_vram
        self.bindless_layouts_init[nrm_slot] = True
        self._write_or_queue_bindless(nrm_slot, self.samp_aniso16, nrm_view)

        var orm_slot = len(self.bindless_images)
        var orm_view = orm_img.view
        var orm_vram = orm_img.memory_bytes
        self.bindless_images.append(orm_img^)
        self._ensure_bindless_meta(orm_slot)
        self.bindless_resident_min_mip[orm_slot] = 0
        self.bindless_vram_bytes[orm_slot] = orm_vram
        self.bindless_layouts_init[orm_slot] = True
        self._write_or_queue_bindless(orm_slot, self.samp_aniso16, orm_view)
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
        var ib_start = i_words_before * 4
        var ib_len = (len(self.geom.index_words) - i_words_before) * 4
        var mb_start = m_words_before * 4
        var mb_len = (len(self.geom.meshlet_words) - m_words_before) * 4
        var ok_vb = self._grow_storage_buffer(vb_start + vb_len, BIND_AAA_VERTICES, BUFFER_USAGE_AS_BUILD_INPUT_KHR, True, True)
        var ok_ib = self._grow_storage_buffer(ib_start + ib_len, BIND_AAA_INDICES, BUFFER_USAGE_INDEX | BUFFER_USAGE_AS_BUILD_INPUT_KHR, True, True)
        var ok_mb = self._grow_storage_buffer(mb_start + mb_len, BIND_AAA_MESHLETS, 0, False, True)
        var grow_ok = ok_vb and ok_ib and ok_mb

        self.skin_arena.vertex_cursor = self.skin_bind_cursor
        var v_alloc = self.skin_arena.alloc_vertices(entry.vertex_count)
        var first_slot = v_alloc[0]
        var v_cnt = v_alloc[1]
        self.skin_bind_cursor = self.skin_arena.vertex_cursor

        self.skin_arena.bone_cursor = self.skin_bone_cursor
        var pal_off = self.skin_arena.alloc_bone_palette(bone_count, bone_palette_offset)
        self.skin_bone_cursor = self.skin_arena.bone_cursor

        # Initialize unpopulated bone slots in `CUR_BONES` & `PREV_BONES` to identity DualQuat
        var eff_bones_init = min(max(1, bone_count), max(0, AAA_MAX_SKIN_BONES - (pal_off // 8)))
        var pal_slot_base = pal_off // 8
        var id_dq_words = List[UInt32](length=8, fill=0)
        id_dq_words[3] = bitcast[DType.uint32](Float32(1.0))
        var bone_copy_offsets = List[Int]()
        var bone_copy_words = List[UInt32]()
        for b in range(eff_bones_init):
            var bs = pal_slot_base + b
            if bs >= 0 and bs < len(self.skin_bones_initialized) and not self.skin_bones_initialized[bs]:
                self.skin_bones_initialized[bs] = True
                bone_copy_offsets.append(AAA_SKIN_CUR_BONES_BASE + pal_off + b * 8)
                for k in range(8):
                    bone_copy_words.append(id_dq_words[k])
                bone_copy_offsets.append(AAA_SKIN_PREV_BONES_BASE + pal_off + b * 8)
                for k in range(8):
                    bone_copy_words.append(id_dq_words[k])

        var m_w_first = 0
        var m_d_first = 0
        var eff_morph_cnt = 0
        var mesh_v_cnt = max(v_cnt, mesh.vertex_count())
        var morph_deltas_words = List[UInt32]()
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
                var src_stride_v = mesh_v_cnt if len(mesh.morph_positions) >= mesh.morph_count * mesh_v_cnt * 3 else v_cnt
                var has_morph_nrm = len(mesh.morph_normals) >= mesh.morph_count * src_stride_v * 3
                morph_deltas_words.resize(eff_morph_cnt * v_cnt * 6, 0)
                for m_i in range(eff_morph_cnt):
                    for v in range(v_cnt):
                        var src_m3 = (m_i * src_stride_v + v) * 3
                        var dst_dw = (m_i * v_cnt + v) * 6
                        morph_deltas_words[dst_dw + 0] = bitcast[DType.uint32](mesh.morph_positions[src_m3 + 0])
                        morph_deltas_words[dst_dw + 1] = bitcast[DType.uint32](mesh.morph_positions[src_m3 + 1])
                        morph_deltas_words[dst_dw + 2] = bitcast[DType.uint32](mesh.morph_positions[src_m3 + 2])
                        var dnx = mesh.morph_normals[src_m3 + 0] if has_morph_nrm else Float32(0.0)
                        var dny = mesh.morph_normals[src_m3 + 1] if has_morph_nrm else Float32(0.0)
                        var dnz = mesh.morph_normals[src_m3 + 2] if has_morph_nrm else Float32(0.0)
                        morph_deltas_words[dst_dw + 3] = bitcast[DType.uint32](dnx)
                        morph_deltas_words[dst_dw + 4] = bitcast[DType.uint32](dny)
                        morph_deltas_words[dst_dw + 5] = bitcast[DType.uint32](dnz)

        var has_mesh_jw = len(mesh.joints) >= v_cnt * 4 and len(mesh.weights) >= v_cnt * 4
        var bind_verts_words = List[UInt32](length=v_cnt * AAA_SKIN_BIND_STRIDE_WORDS, fill=0)
        var prev_pos_words = List[UInt32](length=v_cnt * 3, fill=0)

        for v in range(v_cnt):
            var gv = entry.first_vertex + v
            var src_vw = gv * 12
            var dst_sw = v * AAA_SKIN_BIND_STRIDE_WORDS
            for k in range(12):
                bind_verts_words[dst_sw + k] = self.geom.vertex_words[src_vw + k]

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
            bind_verts_words[dst_sw + 12] = ns[0]
            bind_verts_words[dst_sw + 13] = bitcast[DType.uint32](ns[1])
            bind_verts_words[dst_sw + 14] = bitcast[DType.uint32](ns[2])
            bind_verts_words[dst_sw + 15] = bitcast[DType.uint32](ns[3])

            var prev_off = v * 3
            prev_pos_words[prev_off + 0] = self.geom.vertex_words[src_vw + 0]
            prev_pos_words[prev_off + 1] = self.geom.vertex_words[src_vw + 1]
            prev_pos_words[prev_off + 2] = self.geom.vertex_words[src_vw + 2]

        var st_vb = 0
        var st_ib = (vb_len + 15) & ~15
        var st_mb = st_ib + ((ib_len + 15) & ~15)
        var st_bones = st_mb + ((mb_len + 15) & ~15)
        var bone_bytes = len(bone_copy_words) * 4
        var st_md = st_bones + ((bone_bytes + 15) & ~15)
        var md_bytes = len(morph_deltas_words) * 4
        var st_bv = st_md + ((md_bytes + 15) & ~15)
        var bv_bytes = len(bind_verts_words) * 4
        var st_pp = st_bv + ((bv_bytes + 15) & ~15)
        var pp_bytes = len(prev_pos_words) * 4
        var total_st = st_pp + ((pp_bytes + 15) & ~15)
        var need_blas = self.vk.has_ray_query and entry.triangle_count > 0 and entry.vertex_count > 0
        if grow_ok and (total_st > 0 or eff_morph_cnt > 0 or need_blas):
            if need_blas:
                self._wait_all_in_flight()
            else:
                self._wait_cmd_slot0()
        if grow_ok and total_st > 0:
            self.ensure_staging_capacity(total_st)
            if vb_len > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_vb, Int(self.geom.vertex_words.unsafe_ptr()) + vb_start, vb_len)
            if ib_len > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_ib, Int(self.geom.index_words.unsafe_ptr()) + ib_start, ib_len)
            if mb_len > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_mb, Int(self.geom.meshlet_words.unsafe_ptr()) + mb_start, mb_len)
            if bone_bytes > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_bones, Int(bone_copy_words.unsafe_ptr()), bone_bytes)
            if md_bytes > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_md, Int(morph_deltas_words.unsafe_ptr()), md_bytes)
            if bv_bytes > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_bv, Int(bind_verts_words.unsafe_ptr()), bv_bytes)
            if pp_bytes > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_pp, Int(prev_pos_words.unsafe_ptr()), pp_bytes)
        _ = len(bone_copy_words) + len(morph_deltas_words) + len(bind_verts_words) + len(prev_pos_words)

        if grow_ok and (total_st > 0 or eff_morph_cnt > 0 or need_blas) and self.vk.begin_commands(0):
            if vb_len > 0:
                self.vk.copy_buffer(0, self.buf_staging, st_vb, self.buf_vertices, vb_start, vb_len)
            if ib_len > 0:
                self.vk.copy_buffer(0, self.buf_staging, st_ib, self.buf_indices, ib_start, ib_len)
            if mb_len > 0:
                self.vk.copy_buffer(0, self.buf_staging, st_mb, self.buf_meshlets, mb_start, mb_len)
            for bi in range(len(bone_copy_offsets)):
                self.vk.copy_buffer(0, self.buf_staging, st_bones + bi * 32, self.buf_skinning, bone_copy_offsets[bi] * 4, 32)
            if eff_morph_cnt > 0:
                self.vk.fill_buffer(0, self.buf_skinning, (AAA_SKIN_CUR_MORPH_WEIGHTS_BASE + m_w_first) * 4, eff_morph_cnt * 4, 0)
                self.vk.fill_buffer(0, self.buf_skinning, (AAA_SKIN_PREV_MORPH_WEIGHTS_BASE + m_w_first) * 4, eff_morph_cnt * 4, 0)
            if md_bytes > 0:
                self.vk.copy_buffer(0, self.buf_staging, st_md, self.buf_skinning, (AAA_SKIN_MORPH_DELTAS_BASE + m_d_first * 6) * 4, md_bytes)
            if bv_bytes > 0:
                self.vk.copy_buffer(0, self.buf_staging, st_bv, self.buf_skinning, (AAA_SKIN_BIND_VERTS_BASE + first_slot * AAA_SKIN_BIND_STRIDE_WORDS) * 4, bv_bytes)
            if pp_bytes > 0:
                self.vk.copy_buffer(0, self.buf_staging, st_pp, self.buf_skinning, (AAA_SKIN_PREV_POS_BASE + first_slot * 3) * 4, pp_bytes)
            self.vk.memory_barrier2(
                0,
                PIPELINE_TRANSFER,
                Int(ACCESS_TRANSFER_WRITE),
                PIPELINE_ALL_COMMANDS,
                Int(ACCESS_MEMORY_READ | ACCESS_SHADER_READ | ACCESS_INDIRECT_READ | ACCESS_AS_READ_KHR),
            )
            if need_blas:
                var v_addr = self.vk.buffer_device_address(self.buf_vertices)
                var i_addr = self.vk.buffer_device_address(self.buf_indices) + UInt64(entry.first_index * 4)
                var max_v = max(0, entry.first_vertex + entry.vertex_count - 1)
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
                self.mesh_blas.append(blas^)
            else:
                self.mesh_blas.append(VulkanAccelStruct())
            var f = self.vk.submit(0)
            if f != 0:
                _ = self.vk.wait(f)
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
        _ = self._ensure_dq_skinning_pipeline()
        var ds_skin = self.desc_sets[frame] if (frame >= 0 and frame < len(self.desc_sets)) else self.desc_set
        self.vk.dispatch_aaa(
            frame,
            self.pipe_dq_skinning,
            self.pipe_layout,
            ds_skin,
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
        var cur_w = List[UInt32](length=n_cur * 8, fill=0)
        for b in range(n_cur):
            var dq = cur_bones[b]
            var off = b * 8
            cur_w[off + 0] = bitcast[DType.uint32](dq.rx)
            cur_w[off + 1] = bitcast[DType.uint32](dq.ry)
            cur_w[off + 2] = bitcast[DType.uint32](dq.rz)
            cur_w[off + 3] = bitcast[DType.uint32](dq.rw)
            cur_w[off + 4] = bitcast[DType.uint32](dq.dx)
            cur_w[off + 5] = bitcast[DType.uint32](dq.dy)
            cur_w[off + 6] = bitcast[DType.uint32](dq.dz)
            cur_w[off + 7] = bitcast[DType.uint32](dq.dw)

        var n_prev = min(max_avail, len(prev_bones))
        var prev_w = List[UInt32](length=n_prev * 8, fill=0)
        for b in range(n_prev):
            var dq = prev_bones[b]
            var off = b * 8
            prev_w[off + 0] = bitcast[DType.uint32](dq.rx)
            prev_w[off + 1] = bitcast[DType.uint32](dq.ry)
            prev_w[off + 2] = bitcast[DType.uint32](dq.rz)
            prev_w[off + 3] = bitcast[DType.uint32](dq.rw)
            prev_w[off + 4] = bitcast[DType.uint32](dq.dx)
            prev_w[off + 5] = bitcast[DType.uint32](dq.dy)
            prev_w[off + 6] = bitcast[DType.uint32](dq.dz)
            prev_w[off + 7] = bitcast[DType.uint32](dq.dw)

        var n_clr = max(n_cur, n_prev) if reset_smears else 0

        if invalidate_vsm:
            self.vsm.invalidate_all()

        if defer_to_frame:
            if n_cur > 0:
                self.pending_skin_copies.append((cur_base, cur_w^))
            if n_prev > 0:
                self.pending_skin_copies.append((prev_base, prev_w^))
            if n_clr > 0:
                self.pending_skin_copies.append((AAA_SKIN_CUR_SMEAR_BASE + pal_off, List[UInt32](length=n_clr * 8, fill=0)))
                self.pending_skin_copies.append((AAA_SKIN_PREV_SMEAR_BASE + pal_off, List[UInt32](length=n_clr * 8, fill=0)))
            self._enqueue_pending_skin_update(skin_entry)
            return 0.0

        self._remove_pending_skin_update(skin_entry)
        var t0 = get_time()
        var cur_bytes = len(cur_w) * 4
        var prev_bytes = len(prev_w) * 4
        var st_p = (cur_bytes + 15) & ~15
        var total_b = st_p + ((prev_bytes + 15) & ~15)
        self._wait_all_in_flight()
        if total_b > 0:
            self.ensure_staging_capacity(total_b)
            if cur_bytes > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address, Int(cur_w.unsafe_ptr()), cur_bytes)
            if prev_bytes > 0:
                _ = external_call["memcpy", Int](self.buf_staging.cpu_address + st_p, Int(prev_w.unsafe_ptr()), prev_bytes)
        _ = len(cur_w) + len(prev_w)
        if not self.vk.begin_commands(0):
            return 0.0
        if cur_bytes > 0:
            self.vk.copy_buffer(0, self.buf_staging, 0, self.buf_skinning, cur_base * 4, cur_bytes)
        if prev_bytes > 0:
            self.vk.copy_buffer(0, self.buf_staging, st_p, self.buf_skinning, prev_base * 4, prev_bytes)
        if n_clr > 0:
            self.vk.fill_buffer(0, self.buf_skinning, (AAA_SKIN_CUR_SMEAR_BASE + pal_off) * 4, n_clr * 32, 0)
            self.vk.fill_buffer(0, self.buf_skinning, (AAA_SKIN_PREV_SMEAR_BASE + pal_off) * 4, n_clr * 32, 0)
        if total_b > 0 or n_clr > 0:
            self.vk.memory_barrier2(
                0, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ)
            )
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
        var cw_words = List[UInt32](length=max_m, fill=0)
        var pw_words = List[UInt32](length=max_m, fill=0)
        for m in range(max_m):
            var cw = cur_weights[m] if m < len(cur_weights) else Float32(0.0)
            var pw = (prev_weights[m] if m < len(prev_weights) else Float32(0.0)) if has_prev else cw
            cw_words[m] = bitcast[DType.uint32](cw)
            pw_words[m] = bitcast[DType.uint32](pw)
        self.pending_skin_copies.append((AAA_SKIN_CUR_MORPH_WEIGHTS_BASE + m_first, cw_words^))
        self.pending_skin_copies.append((AAA_SKIN_PREV_MORPH_WEIGHTS_BASE + m_first, pw_words^))
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
        var n_cs = min(max_avail, len(cur_smears))
        var has_ps = len(prev_smears) > 0
        var n_ps = min(max_avail, len(prev_smears) if has_ps else len(cur_smears))
        var nw_sm = max(n_clear, max(n_cs, n_ps)) * 8
        var cur_sm_w = List[UInt32](length=max(1, nw_sm), fill=0)
        var prev_sm_w = List[UInt32](length=max(1, nw_sm), fill=0)

        for b in range(n_cs):
            var sm = cur_smears[b]
            var off = b * 8
            cur_sm_w[off + 0] = bitcast[DType.uint32](sm.scale.x - Float32(1.0))
            cur_sm_w[off + 1] = bitcast[DType.uint32](sm.scale.y - Float32(1.0))
            cur_sm_w[off + 2] = bitcast[DType.uint32](sm.scale.z - Float32(1.0))
            cur_sm_w[off + 3] = bitcast[DType.uint32](sm.shear_xy)
            cur_sm_w[off + 4] = bitcast[DType.uint32](sm.pivot_bind.x)
            cur_sm_w[off + 5] = bitcast[DType.uint32](sm.pivot_bind.y)
            cur_sm_w[off + 6] = bitcast[DType.uint32](sm.pivot_bind.z)
            cur_sm_w[off + 7] = bitcast[DType.uint32](sm.shear_xz)

        for b in range(n_ps):
            var sm = prev_smears[b] if has_ps else cur_smears[b]
            var off = b * 8
            prev_sm_w[off + 0] = bitcast[DType.uint32](sm.scale.x - Float32(1.0))
            prev_sm_w[off + 1] = bitcast[DType.uint32](sm.scale.y - Float32(1.0))
            prev_sm_w[off + 2] = bitcast[DType.uint32](sm.scale.z - Float32(1.0))
            prev_sm_w[off + 3] = bitcast[DType.uint32](sm.shear_xy)
            prev_sm_w[off + 4] = bitcast[DType.uint32](sm.pivot_bind.x)
            prev_sm_w[off + 5] = bitcast[DType.uint32](sm.pivot_bind.y)
            prev_sm_w[off + 6] = bitcast[DType.uint32](sm.pivot_bind.z)
            prev_sm_w[off + 7] = bitcast[DType.uint32](sm.shear_xz)

        if nw_sm > 0:
            if defer_to_frame:
                self.pending_skin_copies.append((cur_sm_base, cur_sm_w^))
                self.pending_skin_copies.append((prev_sm_base, prev_sm_w^))
            else:
                self._upload_to_device_local_sync(Int(cur_sm_w.unsafe_ptr()), nw_sm * 4, BIND_AAA_SKINNING, cur_sm_base * 4)
                self._upload_to_device_local_sync(Int(prev_sm_w.unsafe_ptr()), nw_sm * 4, BIND_AAA_SKINNING, prev_sm_base * 4)
                _ = len(cur_sm_w) + len(prev_sm_w)

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
        out.emissive_bindless_slots = List[Int](length=n_tex, fill=-1)

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
                var has_sep_occ = (
                    m.occlusion_texture >= 0
                    and m.occlusion_texture < n_tex
                    and m.occlusion_texture != m.orm_texture
                    and model.textures[m.occlusion_texture].width > 0
                    and model.textures[m.occlusion_texture].height > 0
                    and len(model.textures[m.occlusion_texture].pixels)
                    >= model.textures[m.occlusion_texture].width * model.textures[m.occlusion_texture].height
                )
                if has_sep_occ:
                    ref otx = model.textures[m.orm_texture]
                    ref octx = model.textures[m.occlusion_texture]
                    if otx.width > 0 and otx.height > 0 and len(otx.pixels) >= otx.width * otx.height:
                        var cur_op = otx.pixels.copy()
                        var ow = otx.width
                        var oh = otx.height
                        while (ow > cap_dim or oh > cap_dim) and ow >= 8 and oh >= 8:
                            var ds = downsample_argb_2x(cur_op, ow, oh)
                            cur_op = ds[0].copy(); ow = ds[1]; oh = ds[2]
                        var orm_px = List[UInt32](capacity=ow * oh)
                        for py in range(oh):
                            var sy = min(octx.height - 1, (py * octx.height) // oh)
                            for px in range(ow):
                                var p = cur_op[py * ow + px]
                                var sx = min(octx.width - 1, (px * octx.width) // ow)
                                var r_ao = Float32((octx.pixels[sy * octx.width + sx] >> 16) & 0xFF)
                                var scaled_ao = UInt32(max(0, min(255, Int(Float32(255.0) * (Float32(1.0) - m.occlusion_strength) + r_ao * m.occlusion_strength + Float32(0.5)))))
                                orm_px.append(UInt32(0xFF000000) | (scaled_ao << 16) | (p & 0x0000FFFF))
                        orm_slot = self.upload_bc7_texture(orm_px, ow, oh, srgb=False, generate_mips=True)
                else:
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
                                var use_ao: UInt32
                                if m.occlusion_texture == m.orm_texture:
                                    use_ao = UInt32(max(0, min(255, Int(Float32(255.0) * (Float32(1.0) - m.occlusion_strength) + Float32(r_ao) * m.occlusion_strength + Float32(0.5)))))
                                else:
                                    use_ao = UInt32(255) if r_ao == 0 else r_ao
                                orm_px.append(UInt32(0xFF000000) | (use_ao << 16) | (p & 0x0000FFFF))
                            out.orm_bindless_slots[m.orm_texture] = self.upload_bc7_texture(
                                orm_px, ow, oh, srgb=False, generate_mips=True
                            )
                    if out.orm_bindless_slots[m.orm_texture] >= 0:
                        orm_slot = out.orm_bindless_slots[m.orm_texture]
            elif m.occlusion_texture >= 0 and m.occlusion_texture < n_tex:
                if out.orm_bindless_slots[m.occlusion_texture] < 0:
                    ref octx = model.textures[m.occlusion_texture]
                    if octx.width > 0 and octx.height > 0 and len(octx.pixels) >= octx.width * octx.height:
                        var cur_op = octx.pixels.copy()
                        var ow = octx.width
                        var oh = octx.height
                        while (ow > cap_dim or oh > cap_dim) and ow >= 8 and oh >= 8:
                            var ds = downsample_argb_2x(cur_op, ow, oh)
                            cur_op = ds[0].copy(); ow = ds[1]; oh = ds[2]
                        var rough_u8 = UInt32(max(12, min(255, Int(m.roughness * Float32(255.0) + Float32(0.5)))))
                        var metal_u8 = UInt32(max(0, min(255, Int(m.metallic * Float32(255.0) + Float32(0.5)))))
                        var orm_px = List[UInt32](capacity=ow * oh)
                        for p in cur_op:
                            var r_ao = Float32((p >> 16) & 0xFF)
                            var scaled_ao = UInt32(max(0, min(255, Int(Float32(255.0) * (Float32(1.0) - m.occlusion_strength) + r_ao * m.occlusion_strength + Float32(0.5)))))
                            orm_px.append(UInt32(0xFF000000) | (scaled_ao << 16) | (rough_u8 << 8) | metal_u8)
                        out.orm_bindless_slots[m.occlusion_texture] = self.upload_bc7_texture(
                            orm_px, ow, oh, srgb=False, generate_mips=True
                        )
                if out.orm_bindless_slots[m.occlusion_texture] >= 0:
                    orm_slot = out.orm_bindless_slots[m.occlusion_texture]

            if m.emissive_texture >= 0 and m.emissive_texture < n_tex:
                if out.emissive_bindless_slots[m.emissive_texture] < 0:
                    ref etx = model.textures[m.emissive_texture]
                    if etx.width > 0 and etx.height > 0 and len(etx.pixels) >= etx.width * etx.height:
                        var cur_ep = etx.pixels.copy()
                        var ew = etx.width
                        var eh = etx.height
                        while (ew > cap_dim or eh > cap_dim) and ew >= 8 and eh >= 8:
                            var ds = downsample_argb_2x(cur_ep, ew, eh)
                            cur_ep = ds[0].copy(); ew = ds[1]; eh = ds[2]
                        out.emissive_bindless_slots[m.emissive_texture] = self.upload_bc7_texture(
                            cur_ep, ew, eh, srgb=True, generate_mips=True
                        )

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

    def _material_shade_class(self, mat: AaaMaterial) -> Int:
        if mat.unlit:
            return AAA_SHADE_CLASS_UNLIT
        var has_sheen = max(mat.sheen_color.x, max(mat.sheen_color.y, mat.sheen_color.z)) > Float32(0.001)
        if (
            mat.clearcoat > Float32(0.001)
            or mat.iridescence > Float32(0.001)
            or has_sheen
            or mat.road_wetness_mode
            or mat.interior_parallax_enabled
        ):
            return AAA_SHADE_CLASS_COMPLEX
        if mat.two_sided_foliage or mat.subsurface_amount > Float32(0.001) or mat.is_alpha_masked():
            return AAA_SHADE_CLASS_FOLIAGE_SSS
        return AAA_SHADE_CLASS_STANDARD_PBR

    def _has_material_class(self, cls: Int) -> Bool:
        if cls == AAA_SHADE_CLASS_SKY or cls == AAA_SHADE_CLASS_STANDARD_PBR:
            return True
        if cls == AAA_SHADE_CLASS_UNLIT and self.pipe_shade_unlit != 0:
            return True
        if cls == AAA_SHADE_CLASS_FOLIAGE_SSS and self.pipe_shade_foliage_sss != 0:
            return True
        if cls == AAA_SHADE_CLASS_COMPLEX and self.pipe_shade_complex != 0:
            return True
        for i in range(len(self.materials)):
            if self._material_shade_class(self.materials[i]) == cls:
                return True
        return False

    def _ensure_shade_class_pipeline(mut self, cls: Int) -> Int:
        var use_rt = self.vk.has_ray_query
        if cls == AAA_SHADE_CLASS_SKY:
            if self.pipe_shade_sky == 0:
                self.pipe_shade_sky = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_shade_sky_shader(True))
            return self.pipe_shade_sky
        if cls == AAA_SHADE_CLASS_UNLIT:
            if self.pipe_shade_unlit == 0:
                self.pipe_shade_unlit = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_shade_unlit_shader(True))
            return self.pipe_shade_unlit
        if cls == AAA_SHADE_CLASS_STANDARD_PBR:
            if self.pipe_shade_pbr == 0:
                self.pipe_shade_pbr = self.vk.create_aaa_compute_pipeline(
                    self.pipe_layout, aaa_deferred_shade_class_shader(use_rt, AAA_SHADE_CLASS_STANDARD_PBR, True)
                )
            return self.pipe_shade_pbr
        if cls == AAA_SHADE_CLASS_FOLIAGE_SSS:
            if self.pipe_shade_foliage_sss == 0:
                self.pipe_shade_foliage_sss = self.vk.create_aaa_compute_pipeline(
                    self.pipe_layout, aaa_deferred_shade_class_shader(use_rt, AAA_SHADE_CLASS_FOLIAGE_SSS, True)
                )
            return self.pipe_shade_foliage_sss
        if cls == AAA_SHADE_CLASS_COMPLEX:
            if self.pipe_shade_complex == 0:
                self.pipe_shade_complex = self.vk.create_aaa_compute_pipeline(
                    self.pipe_layout, aaa_deferred_shade_class_shader(use_rt, AAA_SHADE_CLASS_COMPLEX, True)
                )
            return self.pipe_shade_complex
        return 0

    def _ensure_vat_crowd_eval_pipeline(mut self) -> Int:
        if self.pipe_vat_crowd_eval == 0:
            self.pipe_vat_crowd_eval = self.vk.create_aaa_compute_pipeline(self.pipe_layout, vat_crowd_eval_shader())
        return self.pipe_vat_crowd_eval

    def _ensure_terrain_clipmap_eval_pipeline(mut self) -> Int:
        if self.pipe_terrain_clipmap_eval == 0:
            self.pipe_terrain_clipmap_eval = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, terrain_clipmap_eval_shader()
            )
        return self.pipe_terrain_clipmap_eval

    def _ensure_hiz_pipeline(mut self) -> Int:
        if self.pipe_hiz_downsample == 0:
            self.pipe_hiz_downsample = self.vk.create_aaa_compute_pipeline(self.pipe_layout, hiz_depth_downsample_shader())
        return self.pipe_hiz_downsample

    def _ensure_visbuffer_alphamask_pipeline(mut self) -> Int:
        if self.pipe_visbuffer_alphamask_gfx == 0:
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
        return self.pipe_visbuffer_alphamask_gfx

    def _ensure_vsm_alphamask_pipeline(mut self) -> Int:
        if self.pipe_vsm_alphamask_gfx == 0:
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
        return self.pipe_vsm_alphamask_gfx

    def _ensure_oit_vol_pipeline(mut self) -> Int:
        if self.pipe_oit_vol == 0:
            self.pipe_oit_vol = self.vk.create_aaa_compute_pipeline(self.pipe_layout, oit_and_6way_volumetric_shader())
        return self.pipe_oit_vol

    def _ensure_npr_outline_pipeline(mut self) -> Int:
        if self.pipe_npr_outline_gfx == 0:
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
        return self.pipe_npr_outline_gfx

    def _ensure_surfcache_pipeline(mut self) -> Int:
        if self.pipe_surfcache == 0:
            self.pipe_surfcache = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, surface_cache_update_shader(self.vk.has_ray_query)
            )
        return self.pipe_surfcache

    def _ensure_cluster_cull_pipeline(mut self) -> Int:
        if self.pipe_cluster_cull == 0:
            self.pipe_cluster_cull = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_cluster_cull_shader(BIND_AAA_CLUSTERS, BIND_AAA_GLOBALS)
            )
        return self.pipe_cluster_cull

    def _ensure_dq_skinning_pipeline(mut self) -> Int:
        if self.pipe_dq_skinning == 0:
            self.pipe_dq_skinning = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_dq_skinning_shader(BIND_AAA_VERTICES, BIND_AAA_SKINNING)
            )
        return self.pipe_dq_skinning

    def _ensure_svgf_pipelines(mut self):
        if self.pipe_svgf_reproject == 0:
            self.pipe_svgf_reproject = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, svgf_temporal_reproject_shader()
            )
        if self.pipe_svgf_atrous == 0:
            self.pipe_svgf_atrous = self.vk.create_aaa_compute_pipeline(self.pipe_layout, svgf_atrous_shader())

    def _ensure_spec_reblur_pipelines(mut self):
        if self.pipe_spec_reblur_temp == 0:
            self.pipe_spec_reblur_temp = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, spec_reblur_temporal_shader()
            )
        if self.pipe_spec_reblur_atrous == 0:
            self.pipe_spec_reblur_atrous = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, spec_reblur_atrous_shader()
            )

    def _ensure_rt_shadow_pipelines(mut self):
        if self.vk.has_ray_query:
            if self.pipe_rt_shadow_trace == 0:
                self.pipe_rt_shadow_trace = self.vk.create_aaa_compute_pipeline(
                    self.pipe_layout, rt_sun_shadow_trace_shader()
                )
            if self.pipe_rt_shadow_filter == 0:
                self.pipe_rt_shadow_filter = self.vk.create_aaa_compute_pipeline(
                    self.pipe_layout, rt_sun_shadow_filter_shader()
                )

    def _ensure_glass_refract_pipeline(mut self) -> Int:
        if self.vk.has_ray_query and self.pipe_glass_refract == 0:
            self.pipe_glass_refract = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_glass_refract_shader()
            )
        return self.pipe_glass_refract

    def _ensure_froxel_pipelines(mut self):
        if self.pipe_froxel_inject == 0:
            self.pipe_froxel_inject = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, froxel_inject_and_reproject_shader()
            )
        if self.pipe_froxel_prefix == 0:
            self.pipe_froxel_prefix = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, froxel_prefix_integrate_shader()
            )
        if self.pipe_froxel_march == 0:
            self.pipe_froxel_march = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, froxel_raymarch_integrate_shader()
            )

    def _ensure_particle_pipelines(mut self):
        if self.pipe_particle_clear == 0:
            self.pipe_particle_clear = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, gpu_particle_clear_tiles_shader()
            )
        if self.pipe_particle_sim == 0:
            self.pipe_particle_sim = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, gpu_particle_sim_and_bin_shader()
            )
        if self.pipe_particle_comp == 0:
            self.pipe_particle_comp = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, gpu_particle_tile_composite_shader()
            )

    def _ensure_water_refract_pipeline(mut self) -> Int:
        if self.pipe_water_refract == 0:
            self.pipe_water_refract = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_water_refract_shader(self.vk.has_ray_query)
            )
        return self.pipe_water_refract

    def _ensure_npr_cel_pipeline(mut self) -> Int:
        if self.pipe_npr_cel_shade == 0:
            self.pipe_npr_cel_shade = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_npr_cel_shade_shader()
            )
        return self.pipe_npr_cel_shade

    def _ensure_npr_post_fx_pipeline(mut self) -> Int:
        if self.pipe_npr_post_fx == 0:
            self.pipe_npr_post_fx = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_npr_post_fx_shader()
            )
        return self.pipe_npr_post_fx

    def _ensure_bokeh_dof_pipeline(mut self) -> Int:
        if self.pipe_bokeh_dof == 0:
            self.pipe_bokeh_dof = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_bokeh_dof_shader(BIND_AAA_HDR_OUT, BIND_AAA_POST_SCRATCH)
            )
        return self.pipe_bokeh_dof

    def _ensure_bloom_pipelines(mut self):
        if self.pipe_bloom_down == 0:
            self.pipe_bloom_down = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_bloom_downsample_shader(BIND_AAA_HDR_OUT, BIND_AAA_POST_SCRATCH)
            )
        if self.pipe_bloom_up == 0:
            self.pipe_bloom_up = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_bloom_upsample_shader(BIND_AAA_HDR_OUT, BIND_AAA_POST_SCRATCH)
            )

    def _ensure_motion_blur_pipeline(mut self) -> Int:
        if self.pipe_motion_blur == 0:
            self.pipe_motion_blur = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_motion_blur_shader())
        return self.pipe_motion_blur

    def _ensure_auto_exposure_pipelines(mut self):
        if self.pipe_luma_hist == 0:
            self.pipe_luma_hist = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_luminance_histogram_shader(BIND_AAA_HDR_OUT, BIND_AAA_POST_SCRATCH)
            )
        if self.pipe_adapt_exp == 0:
            self.pipe_adapt_exp = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_adapt_exposure_shader(BIND_AAA_POST_SCRATCH)
            )

    def _ensure_direct_present_compute_pipelines(mut self):
        if self.vk.has_storage_write_without_format and self.pipe_display_tonemap_direct == 0:
            self.pipe_display_tonemap_direct = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_display_tonemap_direct_shader(self.swapchain_img_binding)
            )
        if (
            self.vk.has_storage_write_without_format
            and self.vk.has_storage_read_without_format
            and self.pipe_ui_overlay_direct == 0
        ):
            self.pipe_ui_overlay_direct = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_ui_overlay_direct_shader(self.swapchain_img_binding)
            )

    def _ensure_scrgb_pack_pipeline(mut self) -> Int:
        if self.pipe_scrgb_pack == 0:
            self.pipe_scrgb_pack = self.vk.create_aaa_compute_pipeline(
                self.pipe_layout, aaa_pack_scrgb_rgba16f_shader()
            )
        return self.pipe_scrgb_pack

    def readback_tile_classify_counts(self) -> List[Int]:
        """Returns the 5 per-class 8x8 tile counts (`[SKY, UNLIT, STANDARD_PBR, FOLIAGE_SSS, COMPLEX]`)
        populated on GPU by `aaa_tile_classify_shader` during the last readback frame."""
        var out = List[Int](length=AAA_SHADE_CLASS_COUNT, fill=0)
        for i in range(min(AAA_SHADE_CLASS_COUNT, len(self.last_tile_classify_counts))):
            out[i] = self.last_tile_classify_counts[i]
        return out^

    def register_vat_crowd_clip(
        mut self,
        clip: BakedVatCrowdClip,
        template_mesh: NativeMesh,
        material_id: Int,
        max_phase_slots: Int = 8,
    ) -> Int:
        """Uploads a `BakedVatCrowdClip` (`pos_rgba16f` + `nrm_rgba16f`) and pre-allocates `max_phase_slots`
        disjoint deformable vertex/BLAS slots for GPU `vat_crowd_eval_shader` crowd evaluation."""
        if not self.active or clip.desc.vertex_count <= 0 or clip.desc.frame_count <= 0:
            return -1
        var v_need = template_mesh.vertex_count()
        if v_need <= 0 or self.skin_bind_cursor + v_need > AAA_SKIN_MAX_VERTS:
            return -1

        var pos_slot = self.upload_rgba16f_texture(
            clip.pos_rgba16f, clip.desc.vertex_count, clip.desc.frame_count
        )
        var nrm_slot = self.upload_rgba16f_texture(
            clip.nrm_rgba16f, clip.desc.vertex_count, clip.desc.frame_count
        )
        if pos_slot < 0 or nrm_slot < 0:
            return -1

        var reg = _VatRegisteredClip()
        reg.desc = clip.desc
        reg.desc.pos_tex_slot = UInt16(pos_slot)
        reg.desc.nrm_tex_slot = UInt16(nrm_slot)
        reg.material_id = max(0, material_id)
        var n_slots = max(1, min(32, max_phase_slots))
        var bmin = clip.bounds_min
        var bmax = clip.bounds_max
        var ext_r = max(Float32(4.0), (bmax - bmin).length() * Float32(1.5))
        for _ in range(n_slots):
            self.skin_arena.vertex_cursor = self.skin_bind_cursor
            var v_alloc = self.skin_arena.alloc_vertices(v_need)
            var first_slot = v_alloc[0]
            if v_alloc[1] != v_need or first_slot + v_need > AAA_SKIN_MAX_VERTS:
                self.skin_arena.vertex_cursor = self.skin_bind_cursor
                break
            self.skin_bind_cursor = self.skin_arena.vertex_cursor
            var entry = self.add_mesh(
                template_mesh, reg.material_id, build_lods=False, allow_update=True, expand_bounds=ext_r
            )
            if entry.mesh_id >= 0 and entry.mesh_id < len(self.skinned_mesh_flags):
                self.skinned_mesh_flags[entry.mesh_id] = True
                self.skinned_mesh_bind_slots[entry.mesh_id] = first_slot
            reg.slot_mesh_entries.append(entry^)
            reg.slot_skin_bases.append(first_slot)
            reg.slot_last_norm_phase.append(Float32(0.0))

        if len(reg.slot_mesh_entries) == 0:
            return -1

        var cid = len(self.vat_clips)
        self.vat_clips.append(reg^)
        return cid

    def draw_vat_crowd_instance(
        mut self,
        clip_id: Int,
        phase_sec: Float32,
        prev_phase_sec: Float32,
        world_mat: Mat4,
        prev_world_mat: Mat4 = Mat4(),
        material_override: Int = -1,
        uniform_scale: Float32 = 1.0,
        has_prev: Bool = False,
    ):
        """Queues a GPU VAT-deformed crowd instance for the current frame."""
        if clip_id < 0 or clip_id >= len(self.vat_clips):
            return
        self.frame_vat_draws.append(
            _VatDrawRequest(
                clip_id,
                phase_sec,
                prev_phase_sec,
                world_mat.copy(),
                prev_world_mat.copy(),
                material_override,
                uniform_scale,
                has_prev,
            )
        )

    def _build_terrain_clipmap_ring_mesh(self, ring_idx: Int, desc: TerrainClipmapDesc) -> NativeMesh:
        var n = max(16, (desc.grid_dim_per_ring // 2) * 2)
        var cs = desc.ring_cell_size(ring_idx)
        var half_n = Float32(n) * Float32(0.5)
        var mesh = NativeMesh()
        mesh.double_sided = False
        var stride = n + 1
        for iz in range(stride):
            var lz = (Float32(iz) - half_n) * cs
            for ix in range(stride):
                var lx = (Float32(ix) - half_n) * cs
                mesh.positions.append(lx); mesh.positions.append(Float32(0.0)); mesh.positions.append(lz)
                mesh.normals.append(Float32(0.0)); mesh.normals.append(Float32(1.0)); mesh.normals.append(Float32(0.0))
                mesh.texcoords.append(lx * Float32(0.25)); mesh.texcoords.append(lz * Float32(0.25))
                mesh.ao.append(UInt8(255))  # bit 2 = 1 -> interior vertex

        var cut_min = (n // 4) + 1
        var cut_max = (3 * n) // 4 - 2
        for iz in range(n):
            for ix in range(n):
                if ring_idx > 0 and ix >= cut_min and ix <= cut_max and iz >= cut_min and iz <= cut_max:
                    continue
                var i00 = iz * stride + ix
                var i10 = iz * stride + ix + 1
                var i01 = (iz + 1) * stride + ix
                var i11 = (iz + 1) * stride + ix + 1
                mesh.indices.append(i00); mesh.indices.append(i01); mesh.indices.append(i11)
                mesh.indices.append(i00); mesh.indices.append(i11); mesh.indices.append(i10)

        # Add perimeter skirt vertices (`ao = 251`, bit 2 = 0) along outer boundary so cracks between LODs never leak sky
        var skirt_base = mesh.vertex_count()
        for ix in range(stride):
            mesh.positions.append(mesh.positions[ix * 3 + 0])
            mesh.positions.append(mesh.positions[ix * 3 + 1])
            mesh.positions.append(mesh.positions[ix * 3 + 2])
            mesh.normals.append(Float32(0.0)); mesh.normals.append(Float32(1.0)); mesh.normals.append(Float32(0.0))
            mesh.texcoords.append(mesh.texcoords[ix * 2 + 0]); mesh.texcoords.append(mesh.texcoords[ix * 2 + 1])
            mesh.ao.append(UInt8(251))
        for ix in range(n):
            var top0 = ix
            var top1 = ix + 1
            var bot0 = skirt_base + ix
            var bot1 = skirt_base + ix + 1
            mesh.indices.append(top0); mesh.indices.append(top1); mesh.indices.append(bot1)
            mesh.indices.append(top0); mesh.indices.append(bot1); mesh.indices.append(bot0)

        skirt_base = mesh.vertex_count()
        for ix in range(stride):
            var vi = n * stride + ix
            mesh.positions.append(mesh.positions[vi * 3 + 0])
            mesh.positions.append(mesh.positions[vi * 3 + 1])
            mesh.positions.append(mesh.positions[vi * 3 + 2])
            mesh.normals.append(Float32(0.0)); mesh.normals.append(Float32(1.0)); mesh.normals.append(Float32(0.0))
            mesh.texcoords.append(mesh.texcoords[vi * 2 + 0]); mesh.texcoords.append(mesh.texcoords[vi * 2 + 1])
            mesh.ao.append(UInt8(251))
        for ix in range(n):
            var top0 = n * stride + ix
            var top1 = n * stride + ix + 1
            var bot0 = skirt_base + ix
            var bot1 = skirt_base + ix + 1
            mesh.indices.append(top0); mesh.indices.append(bot1); mesh.indices.append(top1)
            mesh.indices.append(top0); mesh.indices.append(bot0); mesh.indices.append(bot1)

        skirt_base = mesh.vertex_count()
        for iz in range(stride):
            var vi = iz * stride
            mesh.positions.append(mesh.positions[vi * 3 + 0])
            mesh.positions.append(mesh.positions[vi * 3 + 1])
            mesh.positions.append(mesh.positions[vi * 3 + 2])
            mesh.normals.append(Float32(0.0)); mesh.normals.append(Float32(1.0)); mesh.normals.append(Float32(0.0))
            mesh.texcoords.append(mesh.texcoords[vi * 2 + 0]); mesh.texcoords.append(mesh.texcoords[vi * 2 + 1])
            mesh.ao.append(UInt8(251))
        for iz in range(n):
            var top0 = iz * stride
            var top1 = (iz + 1) * stride
            var bot0 = skirt_base + iz
            var bot1 = skirt_base + iz + 1
            mesh.indices.append(top0); mesh.indices.append(bot1); mesh.indices.append(top1)
            mesh.indices.append(top0); mesh.indices.append(bot0); mesh.indices.append(bot1)

        skirt_base = mesh.vertex_count()
        for iz in range(stride):
            var vi = iz * stride + n
            mesh.positions.append(mesh.positions[vi * 3 + 0])
            mesh.positions.append(mesh.positions[vi * 3 + 1])
            mesh.positions.append(mesh.positions[vi * 3 + 2])
            mesh.normals.append(Float32(0.0)); mesh.normals.append(Float32(1.0)); mesh.normals.append(Float32(0.0))
            mesh.texcoords.append(mesh.texcoords[vi * 2 + 0]); mesh.texcoords.append(mesh.texcoords[vi * 2 + 1])
            mesh.ao.append(UInt8(251))
        for iz in range(n):
            var top0 = iz * stride + n
            var top1 = (iz + 1) * stride + n
            var bot0 = skirt_base + iz
            var bot1 = skirt_base + iz + 1
            mesh.indices.append(top0); mesh.indices.append(top1); mesh.indices.append(bot1)
            mesh.indices.append(top0); mesh.indices.append(bot1); mesh.indices.append(bot0)

        return mesh^

    def set_terrain_clipmap(mut self, desc: TerrainClipmapDesc, material_id: Int = -1):
        """Activates concentric GPU terrain clipmap rings evaluated per-frame by `terrain_clipmap_eval_shader`."""
        var mat_id = material_id
        if mat_id < 0 or mat_id >= len(self.materials):
            var tm = AaaMaterial(Vec3(1.0, 1.0, 1.0), Float32(0.75), Float32(0.0))
            mat_id = self.add_material(tm)
            if mat_id < 0:
                mat_id = 0
        self.terrain_clipmap_desc = desc.copy()
        self.terrain_clipmap_desc.ring_count = max(1, min(8, desc.ring_count))
        self.terrain_material_id = mat_id
        self.terrain_ring_meshes.clear()
        self.terrain_ring_origins.clear()
        var expand_h = abs(desc.height_scale_m) + abs(desc.height_offset_m) + Float32(32.0)
        for r in range(self.terrain_clipmap_desc.ring_count):
            var ring_mesh = self._build_terrain_clipmap_ring_mesh(r, self.terrain_clipmap_desc)
            var splat_w = List[UInt32](length=ring_mesh.vertex_count(), fill=0)
            var entry = self.add_splat_mesh(
                ring_mesh, mat_id, splat_w, build_lods=False, allow_update=True, expand_bounds=expand_h
            )
            self.terrain_ring_meshes.append(entry^)
            var orig = self.terrain_clipmap_desc.snapped_ring_origin_xz(self.cam_pos, r)
            self.terrain_ring_origins.append(orig)
        self.terrain_clipmap_active = True
        self.terrain_clipmap_dirty = True
        self.vsm.invalidate_all()

    def clear_terrain_clipmap(mut self):
        self.terrain_clipmap_active = False
        self.terrain_ring_meshes.clear()
        self.terrain_ring_origins.clear()
        self.terrain_clipmap_dirty = False
        self.vsm.invalidate_all()

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
            if self.prev_submitted_fence <= self.frame_fences[cur_slot]:
                self.prev_submitted_fence = 0
            self.frame_fences[cur_slot] = 0
        self.staging_ring.reset_slot(self.vk, cur_slot)
        self._flush_slot_retirements(cur_slot)
        if cur_slot < len(self.slot_has_pending_queries) and self.slot_has_pending_queries[cur_slot]:
            self.slot_has_pending_queries[cur_slot] = False
            var q_pairs = self.slot_prof_pairs[cur_slot].copy()
            var q_frame = self.slot_frame_q[cur_slot]
            self.slot_prof_pairs[cur_slot].clear()
            self.slot_frame_q[cur_slot] = -1
            self._read_gpu_timings(q_pairs, q_frame)

        self.desc_set = self.desc_sets[cur_slot]
        if self.vk.has_ray_query and len(self.ring_tlas) > cur_slot:
            self.frame_tlas.handle = self.ring_tlas[cur_slot].handle

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
        self.frame_vat_draws.clear()
        self.frame_decals.clear()
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
    ):
        if len(self.frame_instances) < self.max_instances and mesh_id >= 0 and mesh_id < len(self.geom.meshes):
            self.frame_instances.append(
                AaaDrawInstance(mesh_id, world_mat, prev_world_mat, material_override, uniform_scale, has_prev)
            )

    def draw_volumetric_plume(mut self, plume: AaaVolumetricPlume):
        if len(self.frame_plumes) < AAA_MAX_OIT_PRIMS:
            self.frame_plumes.append(plume.copy())

    def add_decal(mut self, decal: GpuDecalVolume):
        """Queues a 96-byte (`24`-word) projected OBB deferred decal volume for this frame (up to `AAA_MAX_DECALS`)."""
        if len(self.frame_decals) < AAA_MAX_DECALS:
            self.frame_decals.append(decal.copy())

    def get_render_graph_aliasing_metrics(self) -> RgAliasingMetrics:
        """Returns compiled `AaaRenderGraph` transient VRAM aliasing savings and barrier statistics."""
        return self.render_graph.aliasing_metrics()

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

    def _pack_alphamask_visbuffer_commands(
        mut self, cur_slot: Int, n_inst: Int, tan_x: Float32, tan_y: Float32
    ) -> Tuple[Int, StagingSlice, StagingSlice]:
        """Packs frustum-visible alpha-masked meshlets (`VkDrawIndexedIndirectCommand`, stride 20) +
        `(inst_id, ml_id)` draw pairs into `LinearStagingRing`, keeping `DEVICE_LOCAL` targets host-free."""
        var cmd_cnt = 0
        if n_inst <= 0:
            return (0, StagingSlice(), StagingSlice())
        var max_cmds = min(16384, self.max_draws)
        var draw_pairs_base = 2 * self.max_draws
        var cmds_host = List[UInt32]()
        cmds_host.resize(4, 0)
        var pairs_host = List[UInt32]()
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
                if cmd_cnt >= max_cmds:
                    break
                var ml_id = first_ml + m_loc
                if i_idx > Int(AAA_MAX_PACKED_INSTANCES) or ml_id > Int(AAA_MAX_PACKED_MESHLET_ID):
                    continue
                ref ml = self.geom.meshlets[ml_id]
                cmds_host.append(UInt32(ml.tri_count * 3))
                cmds_host.append(UInt32(1))
                cmds_host.append(UInt32(ml.first_tri * 3))
                cmds_host.append(UInt32(0))
                cmds_host.append(UInt32(draw_pairs_base + cmd_cnt))
                pairs_host.append(UInt32(i_idx))
                pairs_host.append(UInt32(ml_id))
                cmd_cnt += 1

        if cmd_cnt <= 0:
            return (0, StagingSlice(), StagingSlice())
        cmds_host[0] = UInt32(cmd_cnt)
        var sl_ind = self.staging_ring.stage_bytes(self.vk, cur_slot, Int(cmds_host.unsafe_ptr()), len(cmds_host) * 4)
        var sl_pairs = self.staging_ring.stage_bytes(self.vk, cur_slot, Int(pairs_host.unsafe_ptr()), len(pairs_host) * 4)
        _ = len(cmds_host)
        _ = len(pairs_host)
        return (cmd_cnt, sl_ind, sl_pairs)

    def _pack_outline_visbuffer_commands(
        mut self, cur_slot: Int, n_inst: Int, tan_x: Float32, tan_y: Float32
    ) -> Tuple[Int, StagingSlice, StagingSlice]:
        """Packs frustum-visible cel-outlined meshlets (`VkDrawIndexedIndirectCommand`, stride 20) +
        `(inst_id, ml_id)` pairs into `LinearStagingRing`."""
        if len(self.cel_materials) == 0 or n_inst <= 0:
            return (0, StagingSlice(), StagingSlice())
        var cmd_cnt = 0
        var max_cmds = min(16384, self.max_draws)
        var draw_pairs_base = 3 * self.max_draws
        var cmds_host = List[UInt32]()
        cmds_host.resize(4, 0)
        var pairs_host = List[UInt32]()
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

            for m_loc in range(ml_cnt):
                if cmd_cnt >= max_cmds:
                    break
                var ml_id = first_ml + m_loc
                if i_idx > Int(AAA_MAX_PACKED_INSTANCES) or ml_id > Int(AAA_MAX_PACKED_MESHLET_ID):
                    continue
                ref ml = self.geom.meshlets[ml_id]
                cmds_host.append(UInt32(ml.tri_count * 3))
                cmds_host.append(UInt32(1))
                cmds_host.append(UInt32(ml.first_tri * 3))
                cmds_host.append(UInt32(0))
                cmds_host.append(UInt32(draw_pairs_base + cmd_cnt))
                pairs_host.append(UInt32(i_idx))
                pairs_host.append(UInt32(ml_id))
                cmd_cnt += 1

        if cmd_cnt <= 0:
            return (0, StagingSlice(), StagingSlice())
        cmds_host[0] = UInt32(cmd_cnt)
        var sl_ind = self.staging_ring.stage_bytes(self.vk, cur_slot, Int(cmds_host.unsafe_ptr()), len(cmds_host) * 4)
        var sl_pairs = self.staging_ring.stage_bytes(self.vk, cur_slot, Int(pairs_host.unsafe_ptr()), len(pairs_host) * 4)
        _ = len(cmds_host)
        _ = len(pairs_host)
        return (cmd_cnt, sl_ind, sl_pairs)

    def _pack_vsm_indirect_commands(
        mut self,
        cur_slot: Int,
        n_inst: Int,
        light_vp: List[Float32],
        vsm_allocs: List[Tuple[Int, Int, Int, Int]],
    ) -> Tuple[Int, List[Int], List[Int], List[Int], List[Int], StagingSlice, StagingSlice]:
        """Per-page light-space bounding-sphere culler + QEM coarse shadow LOD packer staged into `LinearStagingRing`."""
        var n_pages = len(vsm_allocs)
        var page_offsets = List[Int](length=n_pages, fill=0)
        var page_counts = List[Int](length=n_pages, fill=0)
        var page_am_offsets = List[Int](length=n_pages, fill=0)
        var page_am_counts = List[Int](length=n_pages, fill=0)
        if n_pages == 0 or n_inst == 0:
            return (0, page_offsets^, page_counts^, page_am_offsets^, page_am_counts^, StagingSlice(), StagingSlice())

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

        var per_page = List[List[Tuple[UInt32, UInt32, UInt32]]]()
        var per_page_am = List[List[Tuple[UInt32, UInt32, UInt32]]]()
        for _ in range(n_pages):
            per_page.append(List[Tuple[UInt32, UInt32, UInt32]]())
            per_page_am.append(List[Tuple[UInt32, UInt32, UInt32]]())

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
                        if i_idx > Int(AAA_MAX_PACKED_INSTANCES) or ml_id > Int(AAA_MAX_PACKED_MESHLET_ID):
                            continue
                        var v_cnt_u = UInt32(self.geom.meshlets[ml_id].tri_count * 3)
                        for hp in hit_pages:
                            if is_am_v:
                                per_page_am[hp].append((v_cnt_u, UInt32(i_idx), UInt32(ml_id)))
                            else:
                                per_page[hp].append((v_cnt_u, UInt32(i_idx), UInt32(ml_id)))
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
                        if i_idx > Int(AAA_MAX_PACKED_INSTANCES) or ml_id > Int(AAA_MAX_PACKED_MESHLET_ID):
                            continue
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
                        var v_cnt_u = UInt32(ml.tri_count * 3)
                        for py_i in range(mvy0, mvy1 + 1):
                            var row_off = py_i * 128
                            for px_i in range(mvx0, mvx1 + 1):
                                var ps = page_slot[row_off + px_i]
                                if ps >= 0:
                                    if is_am_v:
                                        per_page_am[ps].append((v_cnt_u, UInt32(i_idx), UInt32(ml_id)))
                                    else:
                                        per_page[ps].append((v_cnt_u, UInt32(i_idx), UInt32(ml_id)))

        var cmd_cnt = 0
        var draw_pairs_base = 4 * self.max_draws
        var vsm_cmds = List[UInt32]()
        var vsm_pairs = List[UInt32]()
        for p_i in range(n_pages):
            page_offsets[p_i] = cmd_cnt
            ref plist = per_page[p_i]
            var written = 0
            for cmd in plist:
                if cmd_cnt >= self.max_draws:
                    break
                vsm_cmds.append(cmd[0])
                vsm_cmds.append(UInt32(1))
                vsm_cmds.append(UInt32(0))
                vsm_cmds.append(UInt32(draw_pairs_base + cmd_cnt))
                vsm_pairs.append(cmd[1])
                vsm_pairs.append(cmd[2])
                cmd_cnt += 1
                written += 1
            page_counts[p_i] = written

            page_am_offsets[p_i] = cmd_cnt
            ref am_list = per_page_am[p_i]
            var written_am = 0
            for cmd in am_list:
                if cmd_cnt >= self.max_draws:
                    break
                vsm_cmds.append(cmd[0])
                vsm_cmds.append(UInt32(1))
                vsm_cmds.append(UInt32(0))
                vsm_cmds.append(UInt32(draw_pairs_base + cmd_cnt))
                vsm_pairs.append(cmd[1])
                vsm_pairs.append(cmd[2])
                cmd_cnt += 1
                written_am += 1
            page_am_counts[p_i] = written_am

        var sl_ind = StagingSlice()
        var sl_pairs = StagingSlice()
        if cmd_cnt > 0:
            sl_ind = self.staging_ring.stage_bytes(self.vk, cur_slot, Int(vsm_cmds.unsafe_ptr()), len(vsm_cmds) * 4)
            sl_pairs = self.staging_ring.stage_bytes(self.vk, cur_slot, Int(vsm_pairs.unsafe_ptr()), len(vsm_pairs) * 4)
        _ = len(vsm_cmds)
        _ = len(vsm_pairs)
        return (cmd_cnt, page_offsets^, page_counts^, page_am_offsets^, page_am_counts^, sl_ind, sl_pairs)

    def ui_prims_word_offset(self, slot: Int) -> Int:
        """Where a frame slot's GPU UI prims live in `buf_post_scratch`: past every per-pixel post
        region (RT shade, water and motion blur stage 8 floats per pixel from word 1024, bloom and
        DoF sit below 1024 + max_pix * 10), so no pass of the frame overwrites them before
        `UiSdfOverlayPass` reads them, and one region per frame in flight, so recording the next
        frame never rewrites prims the GPU is still drawing."""
        var max_pix = max(self.output_w * self.output_h, self.render_w * self.render_h)
        return AAA_POST_UI_PRIMS_BASE_OFFSET + max_pix * 10 + (slot % FRAMES_IN_FLIGHT) * AAA_UI_PRIM_REGION_WORDS

    def _ensure_direct_present_pipelines(mut self, color_fmt: UInt32) -> Bool:
        if color_fmt == 0:
            return False
        if self.present_gfx_format != color_fmt:
            if self.pipe_display_present_gfx != 0:
                self.vk.destroy_pipeline(self.pipe_display_present_gfx)
                self.pipe_display_present_gfx = 0
            if self.pipe_ui_present_gfx != 0:
                self.vk.destroy_pipeline(self.pipe_ui_present_gfx)
                self.pipe_ui_present_gfx = 0
            self.present_gfx_format = color_fmt
        if self.pipe_display_present_gfx == 0:
            self.pipe_display_present_gfx = self.vk.create_graphics_pipeline(
                self.pipe_layout,
                aaa_display_present_vertex_shader(),
                aaa_display_present_fragment_shader(),
                has_color=True,
                color_format=color_fmt,
                depth_format=0,
                depth_test=False,
                depth_write=False,
                cull_mode=CULL_MODE_NONE,
                front_face=FRONT_FACE_CW,
                alpha_blend=False,
            )
        if self.pipe_ui_present_gfx == 0:
            self.pipe_ui_present_gfx = self.vk.create_graphics_pipeline(
                self.pipe_layout,
                aaa_display_present_vertex_shader(),
                aaa_ui_present_fragment_shader(),
                has_color=True,
                color_format=color_fmt,
                depth_format=0,
                depth_test=False,
                depth_write=False,
                cull_mode=CULL_MODE_NONE,
                front_face=FRONT_FACE_CW,
                alpha_blend=True,
            )
        return self.pipe_display_present_gfx != 0 and self.pipe_ui_present_gfx != 0

    def render_frame(mut self, capture_readback: Bool = False, force_readback: Bool = False) -> Bool:
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
        if cur_slot < len(self.frame_fences) and self.frame_fences[cur_slot] != 0:
            _ = self.vk.wait(self.frame_fences[cur_slot])
            if self.prev_submitted_fence <= self.frame_fences[cur_slot]:
                self.prev_submitted_fence = 0
            self.frame_fences[cur_slot] = 0
        self.staging_ring.reset_slot(self.vk, cur_slot)
        self._flush_slot_retirements(cur_slot)
        self.last_async_compute_submits = 0
        self._bind_render_graph_persistent_buffers(cur_slot)
        if cur_slot < len(self.slot_has_pending_queries) and self.slot_has_pending_queries[cur_slot]:
            self.slot_has_pending_queries[cur_slot] = False
            var q_pairs = self.slot_prof_pairs[cur_slot].copy()
            var q_frame = self.slot_frame_q[cur_slot]
            self.slot_prof_pairs[cur_slot].clear()
            self.slot_frame_q[cur_slot] = -1
            self._read_gpu_timings(q_pairs, q_frame)
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
        var rb_slot = cur_slot if defer_this else 0
        self.desc_set = self.desc_sets[cur_slot]
        if self.vk.has_ray_query and len(self.ring_tlas) > cur_slot:
            self.frame_tlas.handle = self.ring_tlas[cur_slot].handle

        var do_readback = capture_readback or force_readback or self.force_readback or (self.vk.swapchain == 0)

        # F18 Pre-Resolve: VAT Crowd phase quantization + Terrain Clipmap ring origin snapping
        var vat_eval_tasks = List[_VatEvalDispatch]()
        if len(self.frame_vat_draws) > 0:
            var vat_slot_used = List[List[Int]]()
            var vat_slot_phase = List[List[Float32]]()
            var vat_slot_prev_phase = List[List[Float32]]()
            for c_i in range(len(self.vat_clips)):
                var ns = len(self.vat_clips[c_i].slot_mesh_entries)
                vat_slot_used.append(List[Int](length=ns, fill=0))
                vat_slot_phase.append(List[Float32](length=ns, fill=Float32(-1.0)))
                vat_slot_prev_phase.append(List[Float32](length=ns, fill=Float32(-1.0)))

            for v_i in range(len(self.frame_vat_draws)):
                ref vd = self.frame_vat_draws[v_i]
                var cid = vd.clip_id
                if cid < 0 or cid >= len(self.vat_clips):
                    continue
                ref vc = self.vat_clips[cid]
                var n_slots = len(vc.slot_mesh_entries)
                if n_slots <= 0:
                    continue
                var dur = max(Float32(1e-4), vc.desc.duration_sec)
                var norm_c: Float32
                var norm_p: Float32
                if vc.desc.looping:
                    var tc = vd.phase_sec - floor(vd.phase_sec / dur) * dur
                    var tp = vd.prev_phase_sec - floor(vd.prev_phase_sec / dur) * dur
                    norm_c = min(Float32(1.0), max(Float32(0.0), tc / dur))
                    norm_p = min(Float32(1.0), max(Float32(0.0), tp / dur))
                else:
                    norm_c = min(Float32(1.0), max(Float32(0.0), vd.phase_sec / dur))
                    norm_p = min(Float32(1.0), max(Float32(0.0), vd.prev_phase_sec / dur))
                var base_slot = min(n_slots - 1, Int(floor(norm_c * Float32(n_slots))))
                var chosen = base_slot
                for probe in range(n_slots):
                    var cand = (base_slot + probe) % n_slots
                    if (
                        vat_slot_used[cid][cand] == 0
                        or (
                            abs(vat_slot_phase[cid][cand] - norm_c) < Float32(1e-4)
                            and abs(vat_slot_prev_phase[cid][cand] - norm_p) < Float32(1e-4)
                        )
                    ):
                        chosen = cand
                        break
                if vat_slot_used[cid][chosen] == 0:
                    vat_slot_used[cid][chosen] = 1
                    vat_slot_phase[cid][chosen] = norm_c
                    vat_slot_prev_phase[cid][chosen] = norm_p
                    var me_v = vc.slot_mesh_entries[chosen].copy()
                    var prev_b = AAA_SKIN_PREV_POS_BASE + vc.slot_skin_bases[chosen] * 3
                    vat_eval_tasks.append(
                        _VatEvalDispatch(
                            cid,
                            me_v.mesh_id,
                            me_v.first_vertex,
                            min(vc.desc.vertex_count, me_v.vertex_count),
                            prev_b,
                            norm_c,
                            norm_p,
                        )
                    )
                var draw_mid = vc.slot_mesh_entries[chosen].mesh_id
                var vd_wm = vd.world_mat.copy()
                var vd_pwm = vd.prev_world_mat.copy()
                var vd_mo = vd.material_override
                var vd_us = vd.uniform_scale
                var vd_hp = vd.has_prev
                self.draw_instance(
                    draw_mid,
                    vd_wm,
                    vd_pwm,
                    vd_mo,
                    vd_us,
                    vd_hp,
                )

        var terrain_eval_rings = List[Tuple[Int, Int, Int, Int, Float32, Float32, Float32]]()
        if self.terrain_clipmap_active and len(self.terrain_ring_meshes) > 0:
            var r_lim = min(len(self.terrain_ring_meshes), self.terrain_clipmap_desc.ring_count)
            var any_ring_shifted = False
            for r in range(r_lim):
                var orig = self.terrain_clipmap_desc.snapped_ring_origin_xz(self.cam_pos, r)
                var prev_orig = orig
                if r < len(self.terrain_ring_origins):
                    prev_orig = self.terrain_ring_origins[r]
                else:
                    while len(self.terrain_ring_origins) <= r:
                        self.terrain_ring_origins.append(orig)
                var shifted = (
                    abs(orig[0] - prev_orig[0]) > Float32(1e-4)
                    or abs(orig[1] - prev_orig[1]) > Float32(1e-4)
                    or self.terrain_clipmap_dirty
                )
                if shifted:
                    self.terrain_ring_origins[r] = orig
                    any_ring_shifted = True
                ref re = self.terrain_ring_meshes[r]
                if shifted:
                    terrain_eval_rings.append(
                        (r, re.mesh_id, re.first_vertex, re.vertex_count, self.terrain_clipmap_desc.ring_cell_size(r), orig[0], orig[1])
                    )
                var cur_m = Mat4()
                cur_m.values[12] = orig[0]
                cur_m.values[14] = orig[1]
                var prv_m = cur_m.copy()
                self.draw_instance(
                    re.mesh_id, cur_m, prv_m, self.terrain_material_id, Float32(1.0), True
                )
            if any_ring_shifted:
                self.vsm.invalidate_all()
            self.terrain_clipmap_dirty = False

        var n_inst = len(self.frame_instances)
        var last_large_sc_inst = -1
        var sc_tri_counts = List[Int](capacity=min(1024, max(1, n_inst)))
        var sc_cursor = 1024  # Leave words [0..4095] for up to 1,024 instance headers!
        var sc_inst_count = min(1024, n_inst)

        # 1. Stage Instance SSBO + 1D 2-Word Meshlet Worklist + Surface-Cache Headers + TLAS Instance Records
        var sl_inst = StagingSlice()
        if n_inst > 0:
            sl_inst = self.staging_ring.alloc(self.vk, cur_slot, n_inst * AAA_INSTANCE_WORDS * 4)
        var sl_sc = StagingSlice()
        var sc_hdr_cnt = min(1024, n_inst)
        if sc_hdr_cnt > 0:
            sl_sc = self.staging_ring.alloc(self.vk, cur_slot, sc_hdr_cnt * 16)
        var sl_as = StagingSlice()
        if self.vk.has_ray_query and n_inst > 0:
            sl_as = self.staging_ring.alloc(self.vk, cur_slot, n_inst * 64)

        var host_worklist = List[UInt32]()
        var worklist_count = 0
        var frame_has_glass = False
        for i in range(n_inst):
            ref di = self.frame_instances[i]
            ref me = self.geom.meshes[di.mesh_id]
            var mat_final = me.material_id if di.material_override < 0 else di.material_override
            var is_am_inst = self._is_instance_alpha_masked(di, mat_final)
            var is_glass_inst = len(self.glass_materials) > 0 and self.is_glass_material(mat_final)
            if is_glass_inst:
                frame_has_glass = True
            var is_ds = (di.mesh_id < len(self.mesh_double_sided_flags) and self.mesh_double_sided_flags[di.mesh_id]) or is_am_inst
            if sl_inst.cpu_address != 0:
                var dst = sl_inst.cpu_address + i * AAA_INSTANCE_WORDS * 4
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
                put32(dst + 38 * 4, UInt32(me.first_index))
                put32(
                    dst + 39 * 4,
                    UInt32(mat_final & 0xFFFF)
                    | ((UInt32(1) if is_ds else UInt32(0)) << 16)
                    | ((UInt32(1) if is_am_inst else UInt32(0)) << 17)
                    | (AAA_INSTANCE_GLASS_BIT if is_glass_inst else UInt32(0)),
                )
            # Compact opaque `(inst_idx, ml_id)` pairs into flat 2-word 1D meshlet cull worklist
            if not is_am_inst and i <= Int(AAA_MAX_PACKED_INSTANCES):
                var first_ml = me.first_meshlet
                var ml_cnt = me.meshlet_count
                for m_loc in range(ml_cnt):
                    var ml_id = first_ml + m_loc
                    if ml_id > Int(AAA_MAX_PACKED_MESHLET_ID):
                        break
                    host_worklist.append(UInt32(i))
                    host_worklist.append(UInt32(ml_id))
                    worklist_count += 1
            var my_sc_base = 0xFFFFFF
            if i < 1024:
                if me.triangle_count <= 4096 and sc_cursor + me.triangle_count < self.max_surfcache_tris:
                    my_sc_base = sc_cursor
                    sc_cursor += me.triangle_count
                    sc_tri_counts.append(me.triangle_count)
                    if me.triangle_count > 2048:
                        last_large_sc_inst = i
                    if sl_sc.cpu_address != 0:
                        var sc_hdr = sl_sc.cpu_address + i * 16
                        put32(sc_hdr + 0, UInt32(my_sc_base))
                        put32(sc_hdr + 4, UInt32(me.first_index))
                        put32(sc_hdr + 8, UInt32(me.triangle_count))
                        put32(sc_hdr + 12, UInt32(mat_final))
                else:
                    sc_tri_counts.append(0)
                    if sl_sc.cpu_address != 0:
                        var sc_hdr = sl_sc.cpu_address + i * 16
                        put32(sc_hdr + 0, UInt32(0xFFFFFF))
                        put32(sc_hdr + 4, UInt32(0))
                        put32(sc_hdr + 8, UInt32(0))
                        put32(sc_hdr + 12, UInt32(0))

            if self.vk.has_ray_query and sl_as.cpu_address != 0 and di.mesh_id < len(self.mesh_blas):
                var blas_da = self.mesh_blas[di.mesh_id].device_address
                write_vk_as_instance_64b(
                    sl_as.cpu_address + i * 64,
                    di.world_mat,
                    my_sc_base,
                    blas_da,
                    cull_disable=True,
                    alpha_masked=is_am_inst,
                    glass=is_glass_inst,
                )
        var sl_worklist = StagingSlice()
        if worklist_count > 0:
            var wl_bytes = len(host_worklist) * 4
            if wl_bytes > self.ring_meshlet_worklist[cur_slot].size:
                self.vk.wait_idle()
                for fi in range(len(self.frame_fences)):
                    self.frame_fences[fi] = 0
                    self._flush_slot_retirements(fi)
                self.prev_submitted_fence = 0
                var new_wl_cap = max(self.ring_meshlet_worklist[cur_slot].size * 2, (wl_bytes + 65535) & ~65535)
                var new_wl = self.vk.create_buffer_device_local(new_wl_cap)
                if new_wl.vk_buffer != 0:
                    self.vk.destroy_buffer(self.ring_meshlet_worklist[cur_slot])
                    self.ring_meshlet_worklist[cur_slot] = new_wl^
                    self.vk.write_descriptor(self.desc_sets[cur_slot], BIND_AAA_WORKLIST, self.ring_meshlet_worklist[cur_slot])
                    self.vk.write_descriptor(self.desc_sets_cull_p1[cur_slot], BIND_AAA_WORKLIST, self.ring_meshlet_worklist[cur_slot])
                    self.vk.write_descriptor(self.desc_sets_cull_p2[cur_slot], BIND_AAA_WORKLIST, self.ring_meshlet_worklist[cur_slot])
                    self.meshlet_worklist_cap = max(self.meshlet_worklist_cap, new_wl_cap // 8)
            worklist_count = min(worklist_count, self.ring_meshlet_worklist[cur_slot].size // 8)
            sl_worklist = self.staging_ring.stage_bytes(self.vk, cur_slot, Int(host_worklist.unsafe_ptr()), worklist_count * 8)
        _ = len(host_worklist)

        # Stage 6-Way OIT Volumetric Plumes into per-frame staging buffer (copied on GPU timeline to avoid CPU-GPU race!)
        var oit_plume_copy_bytes = 0
        if len(self.frame_plumes) > 0:
            var oit_w = List[UInt32]()
            for pl in self.frame_plumes:
                pl.append_words(oit_w)
            oit_plume_copy_bytes = min(len(oit_w) * 4, AAA_MAX_OIT_PRIMS * 16 * 4)
            if oit_plume_copy_bytes > 0:
                _ = external_call["memcpy", Int](
                    self.ring_oit_staging[cur_slot].cpu_address, Int(oit_w.unsafe_ptr()), oit_plume_copy_bytes
                )
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

        var sl_glob = self.staging_ring.alloc(self.vk, cur_slot, 16384)
        var g_addr = sl_glob.cpu_address
        if g_addr == 0:
            return False
        _ = external_call["memset", Int](g_addr, Int32(0), sl_glob.size)
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
        var sl_clust = StagingSlice()
        if n_cl_lights > 0:
            var cl_words = List[UInt32](capacity=n_cl_lights * 16)
            for cl_i in range(n_cl_lights):
                self.frame_cluster_lights[cl_i].append_words(cl_words)
            var cl_bytes = (AAA_CLUSTER_LIGHTS_BASE + len(cl_words)) * 4
            sl_clust = self.staging_ring.alloc(self.vk, cur_slot, cl_bytes)
            if sl_clust.cpu_address != 0:
                var cl_addr = sl_clust.cpu_address
                var far_z_cl = max(self.near_z + Float32(1.0), self.cluster_far_z)
                put32(cl_addr + 0 * 4, UInt32(n_cl_lights))
                put_float(cl_addr + 1 * 4, self.near_z)
                put_float(cl_addr + 2 * 4, far_z_cl)
                put_float(cl_addr + 3 * 4, log2(far_z_cl / max(Float32(1e-4), self.near_z)))
                put_float(cl_addr + 4 * 4, tan_x)
                put_float(cl_addr + 5 * 4, tan_y)
                _ = external_call[
                    "memcpy", Int
                ](cl_addr + AAA_CLUSTER_LIGHTS_BASE * 4, Int(cl_words.unsafe_ptr()), len(cl_words) * 4)
            _ = len(cl_words)

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
        put32(g_addr + AAA_GLOBALS_WATER_SHAPE_WORD * 4, UInt32(1) if self.water_elliptical else UInt32(0))
        put_float(g_addr + (AAA_GLOBALS_WATER_SHAPE_WORD + 1) * 4, self.water_wave_scale)
        put_float(g_addr + (AAA_GLOBALS_WATER_SHAPE_WORD + 2) * 4, self.water_foam_band)
        put_float(g_addr + (AAA_GLOBALS_WATER_SHAPE_WORD + 3) * 4, self.water_foam)
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

        # Extended v5 4-Cascade VSM Globals (`b_globals[167..215]` — `0..166` untouched!):
        put32(g_addr + VSM_GLOBALS_CASCADE_FLAG_WORD * 4, UInt32(1) if use_vsm_casc_frame else UInt32(0))
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

        # Projected OBB Deferred Decals (`GpuDecalVolume`, ring-buffered in `buf_globals[3008..3783]`)
        var n_dec = min(AAA_MAX_DECALS, len(self.frame_decals))
        put32(g_addr + AAA_GLOBALS_DECAL_COUNT_WORD * 4, UInt32(n_dec))
        for d_i in range(n_dec):
            var d_words = self.frame_decals[d_i].pack_words()
            var d_base = AAA_GLOBALS_DECAL_TABLE_WORD + d_i * GPU_DECAL_VOLUME_WORDS
            for d_w in range(GPU_DECAL_VOLUME_WORDS):
                put32(g_addr + (d_base + d_w) * 4, d_words[d_w])

        # F18 Terrain Clipmap Globals (`b_globals[3784..3815]`)
        if self.terrain_clipmap_active:
            var tc_words = self.terrain_clipmap_desc.pack_words()
            for tw in range(len(tc_words)):
                put32(g_addr + (AAA_GLOBALS_TERRAIN_CLIPMAP_WORD + tw) * 4, tc_words[tw])
            put32(g_addr + AAA_GLOBALS_TERRAIN_META_WORD * 4, UInt32(1))
            for r in range(min(8, len(self.terrain_ring_origins))):
                put_float(g_addr + (AAA_GLOBALS_TERRAIN_ORIGINS_WORD + r * 2 + 0) * 4, self.terrain_ring_origins[r][0])
                put_float(g_addr + (AAA_GLOBALS_TERRAIN_ORIGINS_WORD + r * 2 + 1) * 4, self.terrain_ring_origins[r][1])
        else:
            put32(g_addr + AAA_GLOBALS_TERRAIN_META_WORD * 4, UInt32(0))

        # Mirror first 4096 bytes into host-visible shadow `self.buf_globals` for `studio/picking.mojo`
        if self.buf_globals.cpu_address != 0 and g_addr != 0:
            _ = external_call["memcpy", Int](
                self.buf_globals.cpu_address, g_addr, min(4096, sl_glob.size)
            )

        # Pack Bucket 2 alpha-masked VisBuffer draws & hardware `CULL_MODE_FRONT` clip-space outline draws
        var am_res = self._pack_alphamask_visbuffer_commands(cur_slot, n_inst, tan_x, tan_y)
        var am_vis_draws = am_res[0]
        var out_res = (0, StagingSlice(), StagingSlice()) if self.furnace_white_dome else self._pack_outline_visbuffer_commands(cur_slot, n_inst, tan_x, tan_y)
        var outline_vis_draws = out_res[0]

        # Stage deferred skinning uniforms/morphs before recording
        var skin_stage_copies = List[Tuple[Int, StagingSlice]]()
        if len(self.pending_skin_copies) > 0:
            for sc_i in range(len(self.pending_skin_copies)):
                var w_off = self.pending_skin_copies[sc_i][0]
                ref w_arr = self.pending_skin_copies[sc_i][1]
                if len(w_arr) > 0:
                    var sl_sk = self.staging_ring.stage_bytes(
                        self.vk, cur_slot, Int(w_arr.unsafe_ptr()), len(w_arr) * 4
                    )
                    skin_stage_copies.append((w_off * 4, sl_sk))
            self.pending_skin_copies.clear()

        # 3. Ensure TLAS Capacity Before Recording (Zero per-frame `vkAllocateMemory` / `vkDestroyAccelerationStructureKHR`!)
        var use_rt_frame = self.vk.has_ray_query and n_inst > 0 and (
            self.enable_rt_shadows
            or self.enable_rt_reflections
            or self.enable_rt_gi
            or (self.enable_water_plane and not self.furnace_white_dome)
            or frame_has_glass
        )
        var inst_da: UInt64 = 0
        if use_rt_frame and cur_slot < len(self.ring_as_instances):
            inst_da = self.vk.buffer_device_address(self.ring_as_instances[cur_slot])
            # Build sizes depend only on the instance count, so skip `vkGetAccelerationStructureBuildSizesKHR`
            # (16k instances) once this slot's TLAS is known to be large enough.
            var tlas_cap = max(self.max_instances, n_inst)
            while len(self.ring_tlas_cap) < len(self.ring_tlas):
                self.ring_tlas_cap.append(0)
            var tlas_known_ok = (
                cur_slot < len(self.ring_tlas_cap)
                and cur_slot < len(self.ring_tlas)
                and self.ring_tlas[cur_slot].handle != 0
                and self.buf_as_scratch.size > 0
                and self.ring_tlas_cap[cur_slot] >= tlas_cap
            )
            if not tlas_known_ok and cur_slot < len(self.ring_tlas):
                if ensure_tlas_capacity_mixed_opaque(self.vk, self.ring_tlas[cur_slot], inst_da, tlas_cap, self.buf_as_scratch):
                    self.vk.write_aaa_tlas(self.desc_set, BIND_AAA_TLAS, self.ring_tlas[cur_slot].handle)
                if cur_slot < len(self.ring_tlas_cap) and self.ring_tlas[cur_slot].handle != 0:
                    self.ring_tlas_cap[cur_slot] = max(self.ring_tlas_cap[cur_slot], tlas_cap)
            if cur_slot < len(self.ring_tlas):
                self.frame_tlas.handle = self.ring_tlas[cur_slot].handle

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
        var vsm_draws = 0
        var sl_vsm_ind = StagingSlice()
        var sl_vsm_pairs = StagingSlice()
        if vsm_active:
            if use_vsm_casc_frame:
                var next_origins = compute_vsm_cascade_page_origins(self.sun_dir, self.cam_pos, base_vsm_ext)
                if len(self.vsm_cascade_page_origins) >= VSM_CASCADE_COUNT:
                    for c_i in range(VSM_CASCADE_COUNT):
                        if (
                            next_origins[c_i][0] != self.vsm_cascade_page_origins[c_i][0]
                            or next_origins[c_i][1] != self.vsm_cascade_page_origins[c_i][1]
                        ):
                            self.vsm.invalidate_cascade(c_i)
                self.vsm_cascade_page_origins = next_origins^

            # Ring-buffered 1-frame-latency async readback of `ring_vsm_table`:
            # Read from the most recently completed slot (`prev_slot` if its fence already retired, else `cur_slot`
            # which was just waited in `begin_frame`). Never stall mid-frame on camera boundary crossings (`origin_shifted`)!
            var prev_slot = (cur_slot + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT
            var vsm_read_slot = cur_slot
            if self.frame_index > 0 and prev_slot < len(self.frame_fences) and self.frame_fences[prev_slot] == 0:
                vsm_read_slot = prev_slot

            var tbl = List[UInt32](length=VSM_TABLE_WORDS, fill=0)
            _ = external_call["memcpy", Int](
                Int(tbl.unsafe_ptr()), self.ring_vsm_table[vsm_read_slot].cpu_address, VSM_TABLE_WORDS * 4
            )
            var has_gpu_reqs = False
            for r_i in range(16384, VSM_TABLE_WORDS):
                if tbl[r_i] != 0:
                    has_gpu_reqs = True
                    break
            # Only do a one-time Frame-0 synchronous warmup when capturing an immediate single-frame readback
            # before any GPU VSM requests exist; steady-state & VSM camera boundary crossings ALWAYS stay single_cmd_buf!
            var need_sync_vsm_warmup = capture_readback and not self.vsm_warmed_up and not has_gpu_reqs
            single_cmd_buf = not need_sync_vsm_warmup
            if single_cmd_buf:
                var t_vsm_req0 = get_time()
                var pages = self.vsm.process_requests(tbl)
                var t_vsm_req1 = get_time()
                _ = external_call["memcpy", Int](
                    self.ring_vsm_table[cur_slot].cpu_address, Int(tbl.unsafe_ptr()), VSM_TABLE_WORDS * 4
                )
                if has_gpu_reqs:
                    self.vsm_warmed_up = True
                self.last_cpu_vsm_requests_ms = (t_vsm_req1 - t_vsm_req0) * 1000.0
                self.last_cpu_vsm_table_ms = (get_time() - t_vsm0) * 1000.0 - self.last_cpu_vsm_requests_ms
                for pg in pages:
                    vsm_allocs.append((pg.vx, pg.vy, pg.px, pg.py))
                if len(vsm_allocs) > 0:
                    var packed_vsm = self._pack_vsm_indirect_commands(cur_slot, n_inst, light_vp, vsm_allocs)
                    vsm_draws = packed_vsm[0]
                    vsm_page_offsets = packed_vsm[1].copy()
                    vsm_page_counts = packed_vsm[2].copy()
                    vsm_page_am_offsets = packed_vsm[3].copy()
                    vsm_page_am_counts = packed_vsm[4].copy()
                    sl_vsm_ind = packed_vsm[5]
                    sl_vsm_pairs = packed_vsm[6]
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

        self.last_worklist_meshlets = worklist_count
        self.render_graph.begin_frame()
        var use_async_vsm = vsm_active and self.enable_async_vsm and self.vk.has_side_queue() and self.side_cmd_base >= 0
        var vis_slot = (FRAMES_IN_FLIGHT + cur_slot) if use_async_vsm else cur_slot
        if not self.vk.begin_commands(vis_slot):
            return False
        if use_prof:
            self.vk.reset_queries(vis_slot, q_base, AAA_QUERY_SLOT_STRIDE)
        if defer_this:
            # The CPU no longer waits between frames, so serialise this frame's GPU work behind the previous one
            # (single-instance buffers/images: history, post scratch, VSM atlas, surface cache, TLAS scratch...).
            self.vk.memory_barrier2(
                vis_slot, _AAA_STAGE_ALL_COMMANDS, _AAA_ACCESS_MEMORY_READ_WRITE, _AAA_STAGE_ALL_COMMANDS, _AAA_ACCESS_MEMORY_READ_WRITE
            )
        if frame_q >= 0:
            self.vk.write_timestamp(vis_slot, frame_q, PIPELINE_TOP)

        # Batched host->DEVICE_LOCAL staging copies + indirect zero fills + single transfer barrier
        if sl_inst.size > 0:
            self.staging_ring.record_copy(self.vk, vis_slot, cur_slot, sl_inst, self.ring_instances[cur_slot], 0)
        if sl_sc.size > 0:
            self.staging_ring.record_copy(self.vk, vis_slot, cur_slot, sl_sc, self.ring_surfcache[cur_slot], 0)
        if self.vk.has_ray_query and sl_as.size > 0 and cur_slot < len(self.ring_as_instances):
            self.staging_ring.record_copy(self.vk, vis_slot, cur_slot, sl_as, self.ring_as_instances[cur_slot], 0)
        if sl_worklist.size > 0:
            self.staging_ring.record_copy(self.vk, vis_slot, cur_slot, sl_worklist, self.ring_meshlet_worklist[cur_slot], 0)
        self.staging_ring.record_copy(self.vk, vis_slot, cur_slot, sl_glob, self.ring_globals[cur_slot], 0)
        if n_cl_lights > 0 and sl_clust.size > 0:
            self.staging_ring.record_copy(self.vk, vis_slot, cur_slot, sl_clust, self.ring_clusters[cur_slot], 0)
        else:
            self.vk.fill_buffer(vis_slot, self.ring_clusters[cur_slot], 0, 16, 0)
        if am_vis_draws > 0:
            self.staging_ring.record_copy(self.vk, vis_slot, cur_slot, am_res[1], self.ring_alphamask_indirect[cur_slot], 0)
            self.staging_ring.record_copy(self.vk, vis_slot, cur_slot, am_res[2], self.ring_draw_pairs[cur_slot], (2 * self.max_draws) * 8)
        if outline_vis_draws > 0:
            self.staging_ring.record_copy(self.vk, vis_slot, cur_slot, out_res[1], self.ring_outline_indirect[cur_slot], 0)
            self.staging_ring.record_copy(self.vk, vis_slot, cur_slot, out_res[2], self.ring_draw_pairs[cur_slot], (3 * self.max_draws) * 8)
        if vsm_draws > 0:
            self.staging_ring.record_copy(self.vk, vis_slot, cur_slot, sl_vsm_ind, self.buf_vsm_indirect, 0)
            self.staging_ring.record_copy(self.vk, vis_slot, cur_slot, sl_vsm_pairs, self.ring_draw_pairs[cur_slot], (4 * self.max_draws) * 8)
        for sc_i in range(len(skin_stage_copies)):
            self.staging_ring.record_copy(
                self.vk, vis_slot, cur_slot, skin_stage_copies[sc_i][1], self.buf_skinning, skin_stage_copies[sc_i][0]
            )
        self.vk.fill_buffer(vis_slot, self.ring_indirect[cur_slot], 0, 16, 0)
        self.vk.fill_buffer(vis_slot, self.buf_indirect_phase2, 0, 16, 0)
        self.vk.fill_buffer(vis_slot, self.buf_occluded_candidates, 0, 16, 0)
        self.vk.memory_barrier2(
            vis_slot,
            PIPELINE_TRANSFER,
            Int(ACCESS_TRANSFER_WRITE),
            PIPELINE_AS_BUILD_KHR | PIPELINE_DRAW_INDIRECT | PIPELINE_VERTEX_SHADER | PIPELINE_FRAGMENT_SHADER | PIPELINE_COMPUTE,
            Int(ACCESS_TRANSFER_READ | ACCESS_INDIRECT_READ | ACCESS_SHADER_READ | ACCESS_AS_READ_KHR),
        )

        # Execute any deferred Dual-Quaternion skinning + BLAS refits inline (`0` extra submit!)
        if len(self.pending_skin_updates) > 0:
            for sk_i in range(len(self.pending_skin_updates)):
                var sk_e = self.pending_skin_updates[sk_i].copy()
                self._record_skin_dispatch_and_blas_refit(vis_slot, sk_e)
            self.pending_skin_updates.clear()

        # F18 GPU Compute Pre-Pass: VAT Crowd & Terrain Clipmap vertex generation BEFORE MeshletCull & TLAS!
        var did_geom_eval = False
        if len(vat_eval_tasks) > 0 and self._ensure_vat_crowd_eval_pipeline() != 0:
            did_geom_eval = True
            for vt_i in range(len(vat_eval_tasks)):
                ref vt = vat_eval_tasks[vt_i]
                ref vclip = self.vat_clips[vt.clip_id]
                var push_vt = List[UInt32](length=16, fill=0)
                push_vt[0] = UInt32(vt.vertex_count)
                push_vt[1] = UInt32(vclip.desc.frame_count)
                push_vt[2] = UInt32(vclip.desc.pos_tex_slot)
                push_vt[3] = UInt32(vclip.desc.nrm_tex_slot)
                push_vt[4] = UInt32(vt.first_vertex)
                push_vt[5] = UInt32(vt.prev_pos_base_word)
                push_vt[6] = bitcast[DType.uint32](vt.cur_norm_time)
                push_vt[7] = bitcast[DType.uint32](vt.prev_norm_time)
                self.vk.dispatch_aaa(
                    vis_slot,
                    self.pipe_vat_crowd_eval,
                    self.pipe_layout,
                    self.desc_set,
                    push_vt,
                    (vt.vertex_count + 63) // 64,
                    1,
                )

        if len(terrain_eval_rings) > 0 and self._ensure_terrain_clipmap_eval_pipeline() != 0:
            did_geom_eval = True
            var world_sz_m = max(
                Float32(64.0),
                self.terrain_clipmap_desc.ring_half_extent(self.terrain_clipmap_desc.ring_count - 1) * Float32(2.0),
            )
            var world_ox = -world_sz_m * Float32(0.5)
            var world_oz = -world_sz_m * Float32(0.5)
            for tr_i in range(len(terrain_eval_rings)):
                var tr = terrain_eval_rings[tr_i]
                var r_idx = tr[0]
                var v_cnt = tr[3]
                var cell_sz = tr[4]
                var ox = tr[5]
                var oz = tr[6]
                var half_ext = self.terrain_clipmap_desc.ring_half_extent(r_idx)
                var skirt_drop = max(Float32(2.0), cell_sz * Float32(2.0))
                var push_tr = List[UInt32](length=16, fill=0)
                push_tr[0] = UInt32(v_cnt)
                push_tr[1] = UInt32(tr[2])
                push_tr[2] = UInt32(self.terrain_clipmap_desc.heightmap_slot)
                push_tr[3] = UInt32(self.terrain_clipmap_desc.splat_control_slot)
                push_tr[4] = bitcast[DType.uint32](ox)
                push_tr[5] = bitcast[DType.uint32](oz)
                push_tr[6] = bitcast[DType.uint32](cell_sz)
                push_tr[7] = bitcast[DType.uint32](half_ext)
                push_tr[8] = bitcast[DType.uint32](self.terrain_clipmap_desc.height_scale_m)
                push_tr[9] = bitcast[DType.uint32](self.terrain_clipmap_desc.height_offset_m)
                push_tr[10] = bitcast[DType.uint32](self.terrain_clipmap_desc.morph_start_ratio)
                push_tr[11] = bitcast[DType.uint32](world_sz_m)
                push_tr[12] = bitcast[DType.uint32](world_ox)
                push_tr[13] = bitcast[DType.uint32](world_oz)
                push_tr[14] = bitcast[DType.uint32](skirt_drop)
                push_tr[15] = UInt32(self.terrain_clipmap_desc.grid_dim_per_ring)
                self.vk.dispatch_aaa(
                    vis_slot,
                    self.pipe_terrain_clipmap_eval,
                    self.pipe_layout,
                    self.desc_set,
                    push_tr,
                    (v_cnt + 63) // 64,
                    1,
                )

        if did_geom_eval:
            self.vk.memory_barrier2(
                vis_slot,
                PIPELINE_COMPUTE,
                Int(ACCESS_SHADER_WRITE),
                PIPELINE_AS_BUILD_KHR | PIPELINE_VERTEX_SHADER | PIPELINE_COMPUTE,
                Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
            )
            if use_rt_frame:
                var v_addr = self.vk.buffer_device_address(self.buf_vertices)
                for vt_i in range(len(vat_eval_tasks)):
                    var mid = vat_eval_tasks[vt_i].mesh_id
                    if mid >= 0 and mid < len(self.mesh_blas) and self.mesh_blas[mid].handle != 0:
                        ref me = self.geom.meshes[mid]
                        var i_addr = self.vk.buffer_device_address(self.buf_indices) + UInt64(me.first_index * 4)
                        var max_v = max(0, me.first_vertex + me.vertex_count - 1)
                        self.vk.refit_blas_triangles(
                            vis_slot,
                            self.mesh_blas[mid],
                            v_addr,
                            AAA_VERTEX_BYTES,
                            max_v,
                            i_addr,
                            me.triangle_count,
                            self.buf_as_scratch,
                        )
                        self.vk.memory_barrier2(
                            vis_slot,
                            PIPELINE_AS_BUILD_KHR,
                            Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR),
                            PIPELINE_AS_BUILD_KHR | PIPELINE_COMPUTE,
                            Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
                        )
                for tr_i in range(len(terrain_eval_rings)):
                    var mid = terrain_eval_rings[tr_i][1]
                    if mid >= 0 and mid < len(self.mesh_blas) and self.mesh_blas[mid].handle != 0:
                        ref me = self.geom.meshes[mid]
                        var i_addr = self.vk.buffer_device_address(self.buf_indices) + UInt64(me.first_index * 4)
                        var max_v = max(0, me.first_vertex + me.vertex_count - 1)
                        self.vk.refit_blas_triangles(
                            vis_slot,
                            self.mesh_blas[mid],
                            v_addr,
                            AAA_VERTEX_BYTES,
                            max_v,
                            i_addr,
                            me.triangle_count,
                            self.buf_as_scratch,
                        )
                        self.vk.memory_barrier2(
                            vis_slot,
                            PIPELINE_AS_BUILD_KHR,
                            Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR),
                            PIPELINE_AS_BUILD_KHR | PIPELINE_COMPUTE,
                            Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
                        )

        # Inline TLAS Rebuild (`0` host allocations, `0` extra submit!)
        if use_rt_frame and cur_slot < len(self.ring_tlas) and self.ring_tlas[cur_slot].handle != 0:
            record_build_tlas_mixed_opaque(self.vk, vis_slot, self.ring_tlas[cur_slot], inst_da, n_inst, self.buf_as_scratch)
            self.vk.memory_barrier2(
                vis_slot,
                PIPELINE_AS_BUILD_KHR,
                Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR),
                PIPELINE_AS_BUILD_KHR | PIPELINE_COMPUTE,
                Int(ACCESS_AS_WRITE_KHR | ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
            )

        self.render_graph.emit_pass_barriers(self.vk, vis_slot, 0)
        if worklist_count > 0:
            var q_mc = next_q
            if use_prof:
                self.vk.write_timestamp(vis_slot, q_mc, PIPELINE_TOP)
                next_q += 2
                prof_pairs.append((String("MeshletCull"), q_mc))
            var push_cull = List[UInt32](length=16, fill=0)
            push_cull[0] = UInt32(worklist_count)
            push_cull[1] = UInt32(self.max_draws)
            push_cull[2] = UInt32(1) if self.enable_cone_cull else UInt32(0)
            push_cull[3] = UInt32(1) if (self.enable_hiz_cull and self.hiz_valid and self.frame_index > 0 and not cam_teleported) else UInt32(0)
            push_cull[4] = UInt32(self.render_w)
            push_cull[5] = UInt32(self.render_h)
            push_cull[6] = bitcast[DType.uint32](self.max_lod_error_px)
            push_cull[7] = 0  # Phase 1: cull_phase = 0
            push_cull[8] = UInt32(self.hiz_mip_levels)
            push_cull[9] = UInt32(0)  # draw_pairs_base = 0
            self.vk.dispatch_aaa(
                vis_slot,
                self.pipe_meshlet_cull,
                self.pipe_layout,
                self.desc_sets_cull_p1[cur_slot],
                push_cull,
                (worklist_count + 63) // 64,
                1,
            )
            if use_prof:
                self.vk.write_timestamp(vis_slot, q_mc + 1, PIPELINE_BOTTOM)
            self.render_graph.emit_pass_barriers(self.vk, vis_slot, 1)
            self.vk.memory_barrier2(
                vis_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ)
            )
            if rb_slot <= 0 or rb_slot - 1 >= len(self.readback_extra):
                self.vk.copy_buffer(vis_slot, self.ring_indirect[cur_slot], 0, self.buf_ldr_readback, 0, 4)
            else:
                self.vk.copy_buffer(vis_slot, self.ring_indirect[cur_slot], 0, self.readback_extra[rb_slot - 1], 0, 4)

        # Hardware `VK_KHR_dynamic_rendering` 64-Bit Visibility Buffer + Reverse-Z Depth (Phase 1 Draw)
        var q_vb = next_q
        if use_prof:
            self.vk.write_timestamp(vis_slot, q_vb, PIPELINE_TOP)
            next_q += 2
            prof_pairs.append((String("VisBufferHwRaster"), q_vb))
        if not self.render_graph.emit_pass_image_transition(self.vk, vis_slot, 1, self.rg_res_vis, self.vis_img):
            self.vk.transition_image(
                vis_slot, self.vis_img, LAYOUT_COLOR_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE
            )
        if not self.render_graph.emit_pass_image_transition(self.vk, vis_slot, 1, self.rg_res_depth, self.depth_img):
            self.vk.transition_image(
                vis_slot, self.depth_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE
            )
        self.vk.begin_rendering(
            vis_slot, 0, 0, self.render_w, self.render_h, True, self.vis_img.view, True, 0, 0, True, self.depth_img.view, True, 0.0
        )
        self.vk.set_viewport_and_scissor(vis_slot, 0, 0, self.render_w, self.render_h, flip_y=True)
        if worklist_count > 0:
            var push_vb = List[UInt32](length=16, fill=0)
            push_vb[0] = 0
            push_vb[1] = bitcast[DType.uint32](jx_clip)
            push_vb[2] = bitcast[DType.uint32](jy_clip)
            self.vk.bind_graphics_pipeline(vis_slot, self.pipe_visbuffer_gfx, self.pipe_layout, self.desc_set, push_vb)
            self.vk.bind_index_buffer(vis_slot, self.buf_indices, 0, INDEX_TYPE_UINT32)
            self.vk.draw_indexed_indirect_count(
                vis_slot,
                self.ring_indirect[cur_slot],
                16,
                self.ring_indirect[cur_slot],
                0,
                min(self.max_draws, worklist_count),
                stride=20,
            )
        self.vk.end_rendering(vis_slot)

        # Phase 2: Test occluded candidates against fresh current-frame multi-mip Hi-Z pyramid and draw disocclusions
        if n_inst > 0 and self.enable_hiz_cull and self.hiz_valid and self.frame_index > 0 and not cam_teleported:
            # Construct / downsample current frame Hi-Z depth pyramid (`D32_SFLOAT` -> multi-mip `R32_SFLOAT` min-reduction)
            self.vk.transition_image(
                vis_slot, self.depth_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
            )
            self._record_hiz_downsample(vis_slot)

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
            push_cull_p2[8] = UInt32(self.hiz_mip_levels)
            push_cull_p2[9] = UInt32(self.max_draws)  # draw_pairs_base = max_draws
            self.vk.dispatch_aaa(
                vis_slot,
                self.pipe_meshlet_cull,
                self.pipe_layout,
                self.desc_sets_cull_p2[cur_slot],
                push_cull_p2,
                (self.max_draws + 63) // 64,
                1,
            )
            self.vk.memory_barrier2(
                vis_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_DRAW_INDIRECT, Int(ACCESS_INDIRECT_READ)
            )

            # Phase 2 Disocclusion Draw (LOAD existing VisBuffer & Depth, append newly visible meshlets)
            self.vk.memory_barrier2(
                vis_slot,
                PIPELINE_COLOR_ATTACHMENT_OUTPUT,
                Int(ACCESS_COLOR_ATTACHMENT_WRITE),
                PIPELINE_COLOR_ATTACHMENT_OUTPUT,
                Int(ACCESS_COLOR_ATTACHMENT_WRITE),
            )
            self.vk.transition_image(
                vis_slot, self.depth_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE
            )
            self.vk.begin_rendering(
                vis_slot, 0, 0, self.render_w, self.render_h, True, self.vis_img.view, False, 0, 0, True, self.depth_img.view, False, 0.0
            )
            self.vk.set_viewport_and_scissor(vis_slot, 0, 0, self.render_w, self.render_h, flip_y=True)
            var push_vb2 = List[UInt32](length=16, fill=0)
            push_vb2[0] = 0
            push_vb2[1] = bitcast[DType.uint32](jx_clip)
            push_vb2[2] = bitcast[DType.uint32](jy_clip)
            self.vk.bind_graphics_pipeline(vis_slot, self.pipe_visbuffer_gfx, self.pipe_layout, self.desc_set, push_vb2)
            self.vk.bind_index_buffer(vis_slot, self.buf_indices, 0, INDEX_TYPE_UINT32)
            self.vk.draw_indexed_indirect_count(
                vis_slot,
                self.buf_indirect_phase2,
                16,
                self.buf_indirect_phase2,
                0,
                self.max_draws,
                stride=20,
            )
            self.vk.end_rendering(vis_slot)

        # Bucket 2 Hardware Alpha-Cutout VisBuffer Raster (`pipe_visbuffer_alphamask_gfx`, `early_depth = False`, `CULL_MODE_NONE`)
        if am_vis_draws > 0 and self._ensure_visbuffer_alphamask_pipeline() != 0:
            self.vk.memory_barrier2(
                vis_slot,
                PIPELINE_COLOR_ATTACHMENT_OUTPUT | PIPELINE_LATE_FRAGMENT_TESTS,
                Int(ACCESS_COLOR_ATTACHMENT_WRITE | ACCESS_DEPTH_ATTACHMENT_WRITE),
                PIPELINE_COLOR_ATTACHMENT_OUTPUT | PIPELINE_EARLY_FRAGMENT_TESTS | PIPELINE_LATE_FRAGMENT_TESTS,
                Int(ACCESS_COLOR_ATTACHMENT_WRITE | ACCESS_DEPTH_ATTACHMENT_READ | ACCESS_DEPTH_ATTACHMENT_WRITE),
            )
            self.vk.begin_rendering(
                vis_slot, 0, 0, self.render_w, self.render_h, True, self.vis_img.view, False, 0, 0, True, self.depth_img.view, False, 0.0
            )
            self.vk.set_viewport_and_scissor(vis_slot, 0, 0, self.render_w, self.render_h, flip_y=True)
            var push_vb_am = List[UInt32](length=16, fill=0)
            push_vb_am[0] = 0
            push_vb_am[1] = bitcast[DType.uint32](jx_clip)
            push_vb_am[2] = bitcast[DType.uint32](jy_clip)
            self.vk.bind_graphics_pipeline(vis_slot, self.pipe_visbuffer_alphamask_gfx, self.pipe_layout, self.desc_set, push_vb_am)
            self.vk.bind_index_buffer(vis_slot, self.buf_indices, 0, INDEX_TYPE_UINT32)
            self.vk.draw_indexed_indirect(
                vis_slot,
                self.ring_alphamask_indirect[cur_slot],
                16,
                am_vis_draws,
                stride=20,
            )
            self.vk.end_rendering(vis_slot)

        # Hardware Clip-Space Variable-Width Per-Vertex Outline Pass (`pipe_npr_outline_gfx`, `CULL_MODE_FRONT`, Reverse-Z)
        if outline_vis_draws > 0 and self._ensure_npr_outline_pipeline() != 0:
            self.vk.memory_barrier2(
                vis_slot,
                PIPELINE_COLOR_ATTACHMENT_OUTPUT | PIPELINE_LATE_FRAGMENT_TESTS,
                Int(ACCESS_COLOR_ATTACHMENT_WRITE | ACCESS_DEPTH_ATTACHMENT_WRITE),
                PIPELINE_COLOR_ATTACHMENT_OUTPUT | PIPELINE_EARLY_FRAGMENT_TESTS | PIPELINE_LATE_FRAGMENT_TESTS,
                Int(ACCESS_COLOR_ATTACHMENT_WRITE | ACCESS_DEPTH_ATTACHMENT_READ | ACCESS_DEPTH_ATTACHMENT_WRITE),
            )
            self.vk.begin_rendering(
                vis_slot, 0, 0, self.render_w, self.render_h, True, self.vis_img.view, False, 0, 0, True, self.depth_img.view, False, 0.0
            )
            self.vk.set_viewport_and_scissor(vis_slot, 0, 0, self.render_w, self.render_h, flip_y=True)
            var push_out = List[UInt32](length=16, fill=0)
            push_out[0] = 0
            push_out[1] = bitcast[DType.uint32](jx_clip)
            push_out[2] = bitcast[DType.uint32](jy_clip)
            push_out[3] = UInt32(self.render_w)
            push_out[4] = UInt32(self.render_h)
            self.vk.bind_graphics_pipeline(vis_slot, self.pipe_npr_outline_gfx, self.pipe_layout, self.desc_set, push_out)
            self.vk.bind_index_buffer(vis_slot, self.buf_indices, 0, INDEX_TYPE_UINT32)
            self.vk.draw_indexed_indirect(
                vis_slot,
                self.ring_outline_indirect[cur_slot],
                16,
                outline_vis_draws,
                stride=20,
            )
            self.vk.end_rendering(vis_slot)

        self.render_graph.emit_pass_barriers(self.vk, vis_slot, 2)
        if not self.render_graph.emit_pass_image_transition(self.vk, vis_slot, 2, self.rg_res_vis, self.vis_img):
            self.vk.transition_image(
                vis_slot, self.vis_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
            )
        if not self.render_graph.emit_pass_image_transition(self.vk, vis_slot, 2, self.rg_res_depth, self.depth_img):
            self.vk.transition_image(
                vis_slot, self.depth_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
            )
        # Refresh multi-mip Hi-Z pyramid from the final frame depth for next frame's Phase 1 occlusion cull
        if n_inst > 0 and self.enable_hiz_cull:
            self._record_hiz_downsample(vis_slot)
        if use_prof:
            self.vk.write_timestamp(vis_slot, q_vb + 1, PIPELINE_BOTTOM)

        # VSM Mark Visible Virtual Shadow Pages (16k x 16k): Async Compute (`submit_side`) when available, else inline!
        var f_side: UInt64 = 0
        if use_async_vsm:
            if not single_cmd_buf:
                self.vk.memory_barrier2(vis_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
            var f_vis = self.vk.submit_with_wait(vis_slot, self.vk.timeline, self.prev_submitted_fence, PIPELINE_TOP)
            if f_vis == 0:
                return False
            self.prev_submitted_fence = 0

            var side_slot = self.side_cmd_base + cur_slot
            if not self.vk.begin_commands(side_slot):
                return False
            var push_vm = List[UInt32](length=16, fill=0)
            push_vm[0] = UInt32(self.render_w)
            push_vm[1] = UInt32(self.render_h)
            self.vk.dispatch_aaa(
                side_slot, self.pipe_vsm_mark, self.pipe_layout, self.desc_set, push_vm, (self.render_w + 7) // 8, (self.render_h + 7) // 8
            )
            self.vk.memory_barrier2(side_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_HOST | PIPELINE_COMPUTE, Int(ACCESS_HOST_READ | ACCESS_SHADER_READ))
            f_side = self.vk.submit_side(side_slot, self.vk.timeline, f_vis, PIPELINE_COMPUTE)
            if f_side == 0:
                return False
            self.last_async_compute_submits += 1

            if not single_cmd_buf:
                if not self.vk.wait(f_vis) or not self.vk.wait_side(f_side):
                    return False
                f_side = 0
            else:
                if not self.vk.begin_commands(cur_slot):
                    return False
        elif vsm_active:
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

        # Synchronous Frame-0 / Camera-Teleport VSM Page Readback (skipped in warmed-up single-command-buffer gameplay!)
        if not single_cmd_buf:
            if not use_async_vsm:
                self.vk.memory_barrier2(cur_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
                var f_group_a = self.vk.submit_with_wait(cur_slot, self.vk.timeline, self.prev_submitted_fence, PIPELINE_TOP)
                self.prev_submitted_fence = 0
                if f_group_a == 0 or not self.vk.wait(f_group_a):
                    return False
            self.last_culled_meshlet_draws = Int(get32(self.buf_ldr_readback.cpu_address))
            self.vsm_warmed_up = True

            var tbl = List[UInt32](length=VSM_TABLE_WORDS, fill=0)
            _ = external_call["memcpy", Int](Int(tbl.unsafe_ptr()), self.ring_vsm_table[cur_slot].cpu_address, VSM_TABLE_WORDS * 4)
            var saved_reqs = List[Int]()
            for v_i in range(16384):
                if tbl[16384 + v_i] != 0:
                    saved_reqs.append(v_i)
            var pages = self.vsm.process_requests(tbl)
            # Preserve GPU-marked request flags in `ring_vsm_table` so the next static frame runs `single_cmd_buf = True` and hits cache!
            for v_i in saved_reqs:
                tbl[16384 + v_i] = 1
            _ = external_call["memcpy", Int](self.ring_vsm_table[cur_slot].cpu_address, Int(tbl.unsafe_ptr()), VSM_TABLE_WORDS * 4)
            _ = len(tbl)
            for pg in pages:
                vsm_allocs.append((pg.vx, pg.vy, pg.px, pg.py))
            if len(vsm_allocs) > 0 and (self.last_culled_meshlet_draws > 0 or am_vis_draws > 0):
                var packed_vsm = self._pack_vsm_indirect_commands(cur_slot, n_inst, light_vp, vsm_allocs)
                vsm_draws = packed_vsm[0]
                vsm_page_offsets = packed_vsm[1].copy()
                vsm_page_counts = packed_vsm[2].copy()
                vsm_page_am_offsets = packed_vsm[3].copy()
                vsm_page_am_counts = packed_vsm[4].copy()
                sl_vsm_ind = packed_vsm[5]
                sl_vsm_pairs = packed_vsm[6]

            if not self.vk.begin_commands(cur_slot):
                return False
            if vsm_draws > 0:
                self.staging_ring.record_copy(self.vk, cur_slot, cur_slot, sl_vsm_ind, self.buf_vsm_indirect, 0)
                self.staging_ring.record_copy(self.vk, cur_slot, cur_slot, sl_vsm_pairs, self.ring_draw_pairs[cur_slot], (4 * self.max_draws) * 8)
                self.vk.memory_barrier2(
                    cur_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_DRAW_INDIRECT | PIPELINE_VERTEX_SHADER, Int(ACCESS_INDIRECT_READ | ACCESS_SHADER_READ)
                )

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
                var sc_f = Float32(128.0)
                var bx_f = Float32(127 - 2 * vx)
                var by_f = -Float32(127 - 2 * vy)
                var mat_word_off = UInt32(56)
                if use_vsm_casc_frame:
                    var casc = vsm_cascade_from_virtual_page(vx, vy)
                    var loc_vx = vx & (VSM_CASCADE_PAGES_DIM - 1)
                    var loc_vy = vy & (VSM_CASCADE_PAGES_DIM - 1)
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
                    if p_am_cnt > 0 and self._ensure_vsm_alphamask_pipeline() != 0:
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
            _ = self._ensure_surfcache_pipeline()
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
            _ = self._ensure_cluster_cull_pipeline()
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
            self._ensure_froxel_pipelines()
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

        # Soft RT sun shadows: area-sun trace + penumbra-sized edge-stopped denoise (`aaa_rt_shadow.mojo`)
        var rt_sun_flt_off = 0
        if self.enable_rt_shadows and self.vk.has_ray_query and n_inst > 0 and not self.furnace_white_dome:
            self._ensure_rt_shadow_pipelines()
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
                self.vk.write_timestamp(cur_slot, q_rs, PIPELINE_TOP)
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
        self.render_graph.emit_pass_barriers(self.vk, cur_slot, 3)
        var spec_in_off = spec_reblur_raw_offset(max_pix)
        var tile_base = tile_classify_word_offset(max_pix)
        var tiles_x_sh = (self.render_w + 7) // 8
        var tiles_y_sh = (self.render_h + 7) // 8
        var num_tiles_sh = max(1, tiles_x_sh * tiles_y_sh)
        var q_sh = next_q
        if use_prof and q_sh + 1 < q_limit:
            self.vk.write_timestamp(cur_slot, q_sh, PIPELINE_TOP)
            next_q += 2
            prof_pairs.append((String("DeferredMultiLobeRtShade"), q_sh))

        self.vk.fill_buffer(cur_slot, self.buf_post_scratch, tile_base * 4, AAA_SHADE_CLASS_COUNT * 4 * 4, 0)
        self.vk.memory_barrier2(
            cur_slot,
            PIPELINE_TRANSFER,
            Int(ACCESS_TRANSFER_WRITE),
            PIPELINE_COMPUTE,
            Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE),
        )

        var push_tc = List[UInt32](length=16, fill=0)
        push_tc[0] = UInt32(self.render_w)
        push_tc[1] = UInt32(self.render_h)
        push_tc[2] = UInt32(len(self.materials))
        push_tc[7] = UInt32(tile_base)
        push_tc[8] = UInt32(num_tiles_sh)
        self.vk.dispatch_aaa(
            cur_slot,
            self.pipe_tile_classify,
            self.pipe_layout,
            self.desc_set,
            push_tc,
            tiles_x_sh,
            tiles_y_sh,
        )
        self.vk.memory_barrier2(
            cur_slot,
            PIPELINE_COMPUTE,
            Int(ACCESS_SHADER_WRITE),
            PIPELINE_DRAW_INDIRECT | PIPELINE_COMPUTE | PIPELINE_TRANSFER,
            Int(ACCESS_INDIRECT_READ | ACCESS_SHADER_READ | ACCESS_TRANSFER_READ),
        )
        if do_readback:
            if rb_slot <= 0 or rb_slot - 1 >= len(self.readback_extra):
                for cls_i in range(AAA_SHADE_CLASS_COUNT):
                    self.vk.copy_buffer(
                        cur_slot,
                        self.buf_post_scratch,
                        (tile_base + cls_i * 4) * 4,
                        self.buf_ldr_readback,
                        12 + cls_i * 4,
                        4,
                    )
            else:
                for cls_i in range(AAA_SHADE_CLASS_COUNT):
                    self.vk.copy_buffer(
                        cur_slot,
                        self.buf_post_scratch,
                        (tile_base + cls_i * 4) * 4,
                        self.readback_extra[rb_slot - 1],
                        12 + cls_i * 4,
                        4,
                    )

        var push_sh = List[UInt32](length=16, fill=0)
        push_sh[0] = UInt32(self.render_w)
        push_sh[1] = UInt32(self.render_h)
        push_sh[2] = UInt32(max(1, self.gi_rays_per_pixel))
        push_sh[3] = UInt32(1) if run_svgf else UInt32(0)
        push_sh[4] = UInt32(1) if run_spec_reblur else UInt32(0)
        push_sh[5] = UInt32(spec_in_off)
        push_sh[6] = UInt32(rt_sun_flt_off)
        push_sh[8] = UInt32(len(self.materials))
        for cls_i in range(AAA_SHADE_CLASS_COUNT):
            if self._has_material_class(cls_i):
                var c_pipe = self._ensure_shade_class_pipeline(cls_i)
                if c_pipe != 0:
                    push_sh[7] = UInt32(tile_base + AAA_SHADE_CLASS_COUNT * 4 + cls_i * num_tiles_sh)
                    self.vk.dispatch_indirect_aaa(
                        cur_slot,
                        c_pipe,
                        self.pipe_layout,
                        self.desc_set,
                        push_sh,
                        self.buf_post_scratch,
                        (tile_base + cls_i * 4) * 4,
                    )
        if use_prof and q_sh + 1 < q_limit:
            self.vk.write_timestamp(cur_slot, q_sh + 1, PIPELINE_BOTTOM)
        self.render_graph.emit_pass_barriers(self.vk, cur_slot, 4)

        # `SVGF` Temporal Reprojection + Moments Variance Edge-Stopping `À-Trous` Denoiser
        if run_svgf:
            self._ensure_svgf_pipelines()
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
            self._ensure_spec_reblur_pipelines()
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

        # Stylized Anime NPR Cel, Outline & Eroding 3D Mesh VFX Resolve (`aaa_npr_cel.mojo`, `aaa_npr_vfx.mojo`)
        if (n_cel > 0 or n_vfx > 0) and self._ensure_npr_cel_pipeline() != 0:
            var q_cel = next_q
            if use_prof and q_cel + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_cel, PIPELINE_TOP)
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
        if frame_has_glass and use_rt_frame and not self.furnace_white_dome and self._ensure_glass_refract_pipeline() != 0:
            var q_gl = next_q
            if use_prof and q_gl + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_gl, PIPELINE_TOP)
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
            _ = self._ensure_oit_vol_pipeline()
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
            _ = self._ensure_water_refract_pipeline()
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
            self._ensure_froxel_pipelines()
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
        self.render_graph.emit_pass_barriers(self.vk, cur_slot, 5)
        if len(self.frame_plumes) > 0 or not (run_water or run_rain):
            _ = self._ensure_oit_vol_pipeline()
            if oit_plume_copy_bytes > 0:
                self.vk.copy_buffer(
                    cur_slot, self.ring_oit_staging[cur_slot], 0, self.buf_oit_nodes, 0, oit_plume_copy_bytes
                )
                self.vk.memory_barrier2(
                    cur_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ)
                )
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
                var em_words = List[UInt32](capacity=n_em * AAA_EMITTER_WORDS)
                for e_i in range(n_em):
                    var sc_e = max(0, min(4095, self.frame_particle_emitters[e_i].spawn_count))
                    self.frame_particle_emitters[e_i].append_words(em_words, self.particle_ring_cursor)
                    self.particle_ring_cursor = (self.particle_ring_cursor + sc_e) & 0xFFFF
                var em_stage_off = AAA_MAX_OIT_PRIMS * 16 * 4
                var em_bytes = len(em_words) * 4
                _ = external_call["memcpy", Int](
                    self.ring_oit_staging[cur_slot].cpu_address + em_stage_off, Int(em_words.unsafe_ptr()), em_bytes
                )
                self.vk.copy_buffer(
                    cur_slot, self.ring_oit_staging[cur_slot], em_stage_off, self.buf_oit_nodes, em_off * 4, em_bytes
                )
                self.vk.memory_barrier2(
                    cur_slot,
                    PIPELINE_TRANSFER,
                    Int(ACCESS_TRANSFER_WRITE),
                    PIPELINE_COMPUTE,
                    Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE),
                )
                _ = len(em_words)

            var tiles_x = (self.render_w + 15) // 16
            var tiles_y = (self.render_h + 15) // 16
            var num_tiles = tiles_x * tiles_y

            self._ensure_particle_pipelines()
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
        self.render_graph.emit_pass_barriers(self.vk, cur_slot, 6)
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
            _ = self._ensure_bokeh_dof_pipeline()
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

        # Screen-Space Chromatic-Dispersion Shockwaves & Manga Impact Frame (`aaa_npr_vfx.mojo`)
        if (n_sw > 0 or imp_blend > Float32(0.001)) and self._ensure_npr_post_fx_pipeline() != 0:
            var q_npf = next_q
            if use_prof and q_npf + 1 < q_limit:
                self.vk.write_timestamp(cur_slot, q_npf, PIPELINE_TOP)
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
            self._ensure_bloom_pipelines()
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
            _ = self._ensure_motion_blur_pipeline()
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
            self._ensure_auto_exposure_pipelines()
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

        var scrgb_swapchain = self.vk.swapchain != 0 and self.vk.swapchain_format == FORMAT_R16G16B16A16_SFLOAT
        # Keep `vis_img` in `LAYOUT_SHADER_READ_ONLY` across both readback and live-swapchain frames so
        # post-frame `VisBufferGpuPicker` readbacks remain valid without aliasing hazards (`post_alias_img`
        # has no live pass consumer).

        var skipped_before = self.vk.skipped
        var img_idx = NO_IMAGE
        if self.vk.swapchain != 0:
            img_idx = self.vk.acquire_image(cur_slot)

        # Upload dirty SDF font atlas and stage UI primitives into DEVICE_LOCAL `buf_post_scratch` BEFORE any render pass
        var prim_cnt = 0
        var prims_word_offset = self.ui_prims_word_offset(cur_slot)
        if self.ui_draw_list.count() > 0:
            if (
                self.ui_font.is_dirty()
                and self.ui_font_atlas_slot >= 0
                and self.ui_font_atlas_slot < len(self.bindless_images)
                and self.bindless_images[self.ui_font_atlas_slot].image != 0
            ):
                var atlas_bytes = len(self.ui_font.atlas)
                if atlas_bytes > 0:
                    var sl_font = self.staging_ring.stage_bytes(
                        self.vk, cur_slot, Int(self.ui_font.atlas.unsafe_ptr()), atlas_bytes, align=16
                    )
                    if sl_font.cpu_address != 0:
                        var src_st = (
                            PIPELINE_TOP
                            if self.bindless_images[self.ui_font_atlas_slot].layout == LAYOUT_UNDEFINED
                            else (PIPELINE_COMPUTE | PIPELINE_FRAGMENT_SHADER)
                        )
                        var src_ac = (
                            UInt32(0)
                            if self.bindless_images[self.ui_font_atlas_slot].layout == LAYOUT_UNDEFINED
                            else ACCESS_SHADER_READ
                        )
                        self.vk.transition_image(
                            cur_slot,
                            self.bindless_images[self.ui_font_atlas_slot],
                            LAYOUT_TRANSFER_TARGET,
                            src_st,
                            src_ac,
                            PIPELINE_TRANSFER,
                            ACCESS_TRANSFER_WRITE,
                        )
                        self.staging_ring.upload_to_image_subregion(
                            self.vk,
                            cur_slot,
                            cur_slot,
                            sl_font,
                            self.bindless_images[self.ui_font_atlas_slot],
                            0,
                            0,
                            self.ui_font.atlas_width,
                            self.ui_font.atlas_height,
                        )
                        self.vk.transition_image(
                            cur_slot,
                            self.bindless_images[self.ui_font_atlas_slot],
                            LAYOUT_SHADER_READ_ONLY,
                            PIPELINE_TRANSFER,
                            ACCESS_TRANSFER_WRITE,
                            PIPELINE_COMPUTE | PIPELINE_FRAGMENT_SHADER,
                            ACCESS_SHADER_READ,
                        )
                        self.ui_font.mark_clean()

            var max_prims_fit = AAA_UI_PRIM_REGION_WORDS // 16
            prim_cnt = min(self.ui_draw_list.count(), min(32768, max_prims_fit))
            var prim_bytes = prim_cnt * 64
            if prim_bytes > 0:
                var sl_ui = self.staging_ring.stage_bytes(
                    self.vk, cur_slot, Int(self.ui_draw_list.prims.unsafe_ptr()), prim_bytes
                )
                self.staging_ring.record_copy(self.vk, cur_slot, cur_slot, sl_ui, self.buf_post_scratch, prims_word_offset * 4)
            self.vk.memory_barrier2(
                cur_slot,
                PIPELINE_TRANSFER | PIPELINE_COMPUTE,
                Int(ACCESS_TRANSFER_WRITE | ACCESS_SHADER_WRITE),
                PIPELINE_COMPUTE | PIPELINE_FRAGMENT_SHADER,
                Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE),
            )

        # Direct-to-Swapchain Present vs Headless/Readback Display Tonemap + UI Overlay
        var disp_tonemap_mode = UInt32(2) if self.display_mode == AAA_DISPLAY_SCRGB_FP16 else self.display_mode
        var copy_w = min(self.output_w, self.vk.swapchain_width) if self.vk.swapchain != 0 else self.output_w
        var copy_h = min(self.output_h, self.vk.swapchain_height) if self.vk.swapchain != 0 else self.output_h
        var direct_live = (
            self.vk.swapchain != 0
            and (not scrgb_swapchain)
            and (not do_readback)
            and img_idx != NO_IMAGE
            and copy_w > 0
            and copy_h > 0
        )
        var is_srgb_sw = (self.vk.swapchain_format == 50) or (self.vk.swapchain_format == 43)

        var q_dt = next_q
        if use_prof and q_dt + 1 < q_limit:
            self.vk.write_timestamp(cur_slot, q_dt, PIPELINE_TOP)
            next_q += 2
            prof_pairs.append((String("CinemaDisplayTonemap"), q_dt))
        var push_dt = List[UInt32](length=16, fill=0)
        push_dt[0] = UInt32(copy_w if direct_live else self.output_w)
        push_dt[1] = UInt32(copy_h if direct_live else self.output_h)
        push_dt[2] = UInt32(out_ldr_off)
        push_dt[3] = disp_tonemap_mode
        push_dt[4] = UInt32(1) if has_saved_bloom_mip else UInt32(0)
        push_dt[5] = UInt32(saved_bm_w)
        push_dt[6] = UInt32(saved_bm_h)
        push_dt[7] = UInt32(safe_bloom_mip_off)
        push_dt[8] = bitcast[DType.uint32](self.grade_saturation - Float32(1.0))
        push_dt[9] = bitcast[DType.uint32](self.grade_contrast - Float32(1.0))
        push_dt[10] = UInt32(1) if is_srgb_sw else UInt32(0)

        var direct_used_storage = False
        var direct_used_gfx = False
        if direct_live:
            self._ensure_direct_present_compute_pipelines()
            var use_storage_direct = (
                self.vk.swapchain_has_storage
                and self.vk.has_storage_write_without_format
                and self.pipe_display_tonemap_direct != 0
                and img_idx < len(self.vk.image_views)
                and self.vk.image_views[img_idx] != 0
            )
            if use_storage_direct:
                direct_used_storage = True
                self.vk.write_aaa_storage_image(
                    self.desc_set, self.swapchain_img_binding, self.vk.image_views[img_idx], LAYOUT_GENERAL
                )
                self.vk.transition_swapchain_image(
                    cur_slot, img_idx, LAYOUT_GENERAL, PIPELINE_TOP, 0, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE
                )
                self.vk.dispatch_aaa(
                    cur_slot,
                    self.pipe_display_tonemap_direct,
                    self.pipe_layout,
                    self.desc_set,
                    push_dt,
                    (copy_w + 7) // 8,
                    (copy_h + 7) // 8,
                )
            elif (
                img_idx < len(self.vk.image_views)
                and self.vk.image_views[img_idx] != 0
                and self._ensure_direct_present_pipelines(self.vk.swapchain_format)
            ):
                direct_used_gfx = True
                self.vk.transition_swapchain_image(
                    cur_slot, img_idx, LAYOUT_COLOR_ATTACHMENT, PIPELINE_TOP, 0, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE
                )
                self.vk.begin_rendering(
                    cur_slot, 0, 0, copy_w, copy_h, True, self.vk.image_views[img_idx], False, 0, 0, False, 0, False, 0.0
                )
                self.vk.set_viewport_and_scissor(cur_slot, 0, 0, copy_w, copy_h, flip_y=False)
                self.vk.bind_graphics_pipeline(cur_slot, self.pipe_display_present_gfx, self.pipe_layout, self.desc_set, push_dt)
                self.vk.draw(cur_slot, 3, 1)
                if self.ui_draw_list.count() == 0:
                    self.vk.end_rendering(cur_slot)
                    self.vk.transition_swapchain_image(
                        cur_slot, img_idx, LAYOUT_PRESENT, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE, PIPELINE_BOTTOM, 0
                    )
            else:
                direct_live = False

        if not direct_live:
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

            var push_ui = List[UInt32](length=16, fill=0)
            push_ui[0] = UInt32(copy_w if direct_live else self.output_w)
            push_ui[1] = UInt32(copy_h if direct_live else self.output_h)
            push_ui[2] = UInt32(prim_cnt)
            push_ui[3] = UInt32(prims_word_offset)
            push_ui[4] = UInt32(out_ldr_off)
            push_ui[5] = disp_tonemap_mode
            push_ui[10] = UInt32(1) if is_srgb_sw else UInt32(0)

            if direct_used_gfx:
                self.vk.bind_graphics_pipeline(cur_slot, self.pipe_ui_present_gfx, self.pipe_layout, self.desc_set, push_ui)
                self.vk.draw(cur_slot, 3, 1)
                self.vk.end_rendering(cur_slot)
                self.vk.transition_swapchain_image(
                    cur_slot, img_idx, LAYOUT_PRESENT, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE, PIPELINE_BOTTOM, 0
                )
            elif direct_used_storage:
                if self.pipe_ui_overlay_direct != 0:
                    self.vk.memory_barrier2(
                        cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
                    )
                    self.vk.dispatch_aaa(
                        cur_slot,
                        self.pipe_ui_overlay_direct,
                        self.pipe_layout,
                        self.desc_set,
                        push_ui,
                        (copy_w + 15) // 16,
                        (copy_h + 15) // 16,
                    )
                    self.vk.transition_swapchain_image(
                        cur_slot, img_idx, LAYOUT_PRESENT, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_BOTTOM, 0
                    )
                elif self._ensure_direct_present_pipelines(self.vk.swapchain_format):
                    self.vk.transition_swapchain_image(
                        cur_slot,
                        img_idx,
                        LAYOUT_COLOR_ATTACHMENT,
                        PIPELINE_COMPUTE,
                        ACCESS_SHADER_WRITE,
                        PIPELINE_COLOR_ATTACHMENT_OUTPUT,
                        UInt32(0x80) | ACCESS_COLOR_ATTACHMENT_WRITE,
                    )
                    self.vk.begin_rendering(
                        cur_slot, 0, 0, copy_w, copy_h, True, self.vk.image_views[img_idx], False, 0, 0, False, 0, False, 0.0
                    )
                    self.vk.set_viewport_and_scissor(cur_slot, 0, 0, copy_w, copy_h, flip_y=False)
                    self.vk.bind_graphics_pipeline(cur_slot, self.pipe_ui_present_gfx, self.pipe_layout, self.desc_set, push_ui)
                    self.vk.draw(cur_slot, 3, 1)
                    self.vk.end_rendering(cur_slot)
                    self.vk.transition_swapchain_image(
                        cur_slot, img_idx, LAYOUT_PRESENT, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE, PIPELINE_BOTTOM, 0
                    )
                else:
                    self.vk.transition_swapchain_image(
                        cur_slot, img_idx, LAYOUT_PRESENT, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_BOTTOM, 0
                    )
            else:
                self.vk.memory_barrier2(
                    cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
                )
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
        elif direct_used_storage:
            self.vk.transition_swapchain_image(
                cur_slot, img_idx, LAYOUT_PRESENT, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_BOTTOM, 0
            )

        # Extended Linear scRGB `R16G16B16A16_SFLOAT` (`8` bytes/pixel `RGBA16F`, `1.0 = 80 nits`) Output Pack!
        var want_scrgb_fp16 = (self.display_mode == AAA_DISPLAY_SCRGB_FP16) or scrgb_swapchain
        if want_scrgb_fp16:
            _ = self._ensure_scrgb_pack_pipeline()
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
        if do_readback:
            # DMA-copy final LDR frame + adapted exposure from `DEVICE_LOCAL` VRAM into `HOST_CACHED` readback memory
            self.vk.memory_barrier2(
                cur_slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_READ)
            )
            if rb_slot <= 0 or rb_slot - 1 >= len(self.readback_extra):
                self.vk.copy_buffer(cur_slot, self.buf_post_scratch, AAA_POST_ADAPTED_EXP_WORD * 4, self.buf_ldr_readback, 4, 8)
                self.vk.copy_buffer(cur_slot, self.buf_history, out_ldr_off * 4, self.buf_ldr_readback, 64, self.output_w * self.output_h * 4)
            else:
                self.vk.copy_buffer(cur_slot, self.buf_post_scratch, AAA_POST_ADAPTED_EXP_WORD * 4, self.readback_extra[rb_slot - 1], 4, 8)
                self.vk.copy_buffer(cur_slot, self.buf_history, out_ldr_off * 4, self.readback_extra[rb_slot - 1], 64, self.output_w * self.output_h * 4)
            self.vk.memory_barrier2(cur_slot, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
        if frame_q >= 0:
            self.vk.write_timestamp(cur_slot, frame_q + 1, PIPELINE_BOTTOM)

        var t_submit0 = get_time()
        self.last_cpu_record_ms = (t_submit0 - t_record0) * 1000.0
        var async_swapchain_submit = False
        var wait_sem = self.vk.side_timeline if f_side > 0 else (self.vk.timeline if self.prev_submitted_fence > 0 else 0)
        var wait_val = f_side if f_side > 0 else self.prev_submitted_fence
        var wait_stg = PIPELINE_TOP
        if self.vk.swapchain != 0:
            if img_idx != NO_IMAGE:
                if (not direct_live) and copy_w > 0 and copy_h > 0:
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
                var f = self.vk.submit_and_present(
                    cur_slot,
                    img_idx,
                    extra_wait_sem=wait_sem,
                    extra_wait_val=wait_val,
                    extra_wait_stage=wait_stg,
                )
                self.frame_fences[cur_slot] = f
                self.prev_submitted_fence = f
                if self.vk.skipped > skipped_before:
                    # `vkQueuePresentKHR` returned `VK_ERROR_OUT_OF_DATE_KHR`: flag swapchain recreation.
                    self.swapchain_out_of_date = True
                if do_readback and f != 0:
                    var ok_w = self.vk.wait(f)
                    self.frame_fences[cur_slot] = 0
                    self.prev_submitted_fence = 0
                    self._flush_slot_retirements(cur_slot)
                    if not ok_w:
                        return False
                elif f != 0:
                    async_swapchain_submit = True
            else:
                # `NO_IMAGE` / `VK_ERROR_OUT_OF_DATE_KHR`: finish command buffer, flag swapchain resize, and sync targets!
                var f = self.vk.submit_with_wait(cur_slot, wait_sem, wait_val, wait_stg)
                self.frame_fences[cur_slot] = f
                if f != 0:
                    var ok_w = self.vk.wait(f)
                    self.frame_fences[cur_slot] = 0
                    self.prev_submitted_fence = 0
                    self._flush_slot_retirements(cur_slot)
                    if not ok_w:
                        return False
                self.prev_submitted_fence = 0
                self.swapchain_out_of_date = True
                if (
                    self.vk.swapchain_width > 0
                    and self.vk.swapchain_height > 0
                    and (self.vk.swapchain_width != self.output_w or self.vk.swapchain_height != self.output_h)
                ):
                    _ = self.resize(self.vk.swapchain_width, self.vk.swapchain_height)
        else:
            var f = self.vk.submit_with_wait(cur_slot, wait_sem, wait_val, wait_stg)
            self.frame_fences[cur_slot] = f
            self.prev_submitted_fence = f
            if f == 0:
                return False
            if defer_this:
                # Phase A: do not wait; the next `render_frame()` (or `flush_deferred_frame()`) retires it.
                self.pending_slot = cur_slot
                self.pending_prof_pairs = prof_pairs.copy()
                self.pending_frame_q = frame_q
                self.pending_n_inst = n_inst
            elif do_readback:
                var ok_w = self.vk.wait(f)
                self.frame_fences[cur_slot] = 0
                self.prev_submitted_fence = 0
                self._flush_slot_retirements(cur_slot)
                if not ok_w:
                    return False
        self.last_cpu_submit_wait_ms = (get_time() - t_submit0) * 1000.0

        if async_swapchain_submit:
            # Defer `VkQueryPool` timestamp readback by `FRAMES_IN_FLIGHT` frames until `vk.wait(frame_fences[cur_slot])`
            # completes in `begin_frame()` — never read unwritten queries right after async swapchain submit!
            if cur_slot < len(self.slot_has_pending_queries):
                self.slot_prof_pairs[cur_slot] = prof_pairs^
                self.slot_frame_q[cur_slot] = frame_q
                self.slot_has_pending_queries[cur_slot] = True
        elif self.pending_slot < 0:
            if do_readback:
                self.last_culled_meshlet_draws = Int(get32(self.buf_ldr_readback.cpu_address)) if n_inst > 0 else 0
                self.readback_ready_slot = 0
                var tc_rb = self._readback_address()
                if tc_rb != 0:
                    for cls_i in range(AAA_SHADE_CLASS_COUNT):
                        self.last_tile_classify_counts[cls_i] = Int(get32(tc_rb + 12 + cls_i * 4))

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
            if self.prev_submitted_fence <= self.frame_fences[slot]:
                self.prev_submitted_fence = 0
            self.frame_fences[slot] = 0
        self._flush_slot_retirements(slot)
        self.readback_ready_slot = slot
        var tc_rb = self._readback_address()
        self.last_culled_meshlet_draws = Int(get32(tc_rb)) if self.pending_n_inst > 0 else 0
        if tc_rb != 0:
            for cls_i in range(AAA_SHADE_CLASS_COUNT):
                self.last_tile_classify_counts[cls_i] = Int(get32(tc_rb + 12 + cls_i * 4))
        var pairs = self.pending_prof_pairs^
        self.pending_prof_pairs = List[Tuple[String, Int]]()
        self._read_gpu_timings(pairs, self.pending_frame_q)
        self.pending_frame_q = -1

    def flush_deferred_frame(mut self):
        """Deferred readback: block until the last submitted frame is done so `readback_pixels_into()`
        returns it (e.g. before a screenshot or when leaving deferred mode)."""
        self._retire_pending_frame()
        for s_i in range(len(self.frame_fences)):
            if self.frame_fences[s_i] == 0:
                self._flush_slot_retirements(s_i)

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

    def readback_resolved_hdr(mut self) -> List[Vec3]:
        """Reads back linear HDR `(R, G, B)` resolved by `FlintTSR v2` before display OETF."""
        var n = self.output_w * self.output_h
        var max_pix = max(n, self.render_w * self.render_h)
        var last_written_hist = (max_pix * 20) if ((self.frame_index - 1) & 1) == 0 else (max_pix * 16)
        var out = List[Vec3](length=n, fill=Vec3())
        if not self.active or self.buf_history.vk_buffer == 0 or n <= 0:
            return out^
        var rb = self.vk.create_buffer(n * 16, local=False, cpu_reads=True)
        if rb.vk_buffer == 0 or rb.cpu_address == 0:
            self.vk.destroy_buffer(rb)
            return out^
        self.vk.wait_idle()
        if self.vk.begin_commands(0):
            self.vk.copy_buffer(0, self.buf_history, last_written_hist * 4, rb, 0, n * 16)
            self.vk.memory_barrier2(0, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))
            var f = self.vk.submit(0)
            if f != 0 and self.vk.wait(f):
                var raw_f = List[Float32](length=n * 4, fill=0.0)
                _ = external_call[
                    "memcpy", Int
                ](Int(raw_f.unsafe_ptr()), rb.cpu_address, n * 16)
                for i in range(n):
                    out[i] = Vec3(raw_f[i * 4 + 0], raw_f[i * 4 + 1], raw_f[i * 4 + 2])
                _ = len(raw_f)
        self.vk.destroy_buffer(rb)
        return out^

    def readback_scrgb_rgba16f_words(mut self) -> List[UInt32]:
        """Reads back `8`-byte/pixel `R16G16B16A16_SFLOAT` (`2 x UInt32` per pixel) Extended Linear scRGB words (`1.0 = 80 nits`)."""
        var n_words = self.output_w * self.output_h * 2
        var out = List[UInt32](length=n_words, fill=0)
        if not self.active or self.buf_post_scratch.vk_buffer == 0 or n_words <= 0:
            return out^
        if (AAA_POST_BLOOM_BASE + n_words) * 4 <= self.buf_post_scratch.size:
            var rb = self.vk.create_buffer(n_words * 4, local=False, cpu_reads=True)
            if rb.vk_buffer != 0 and rb.cpu_address != 0:
                self.vk.wait_idle()
                if self.vk.begin_commands(0):
                    self.vk.copy_buffer(0, self.buf_post_scratch, AAA_POST_BLOOM_BASE * 4, rb, 0, n_words * 4)
                    self.vk.memory_barrier2(
                        0, PIPELINE_TRANSFER, Int(ACCESS_TRANSFER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ)
                    )
                    var f = self.vk.submit(0)
                    if f != 0 and self.vk.wait(f):
                        _ = external_call[
                            "memcpy", Int
                        ](Int(out.unsafe_ptr()), rb.cpu_address, n_words * 4)
            self.vk.destroy_buffer(rb)
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
        for fi in range(len(self.frame_fences)):
            self.frame_fences[fi] = 0
        self.prev_submitted_fence = 0
        for i in range(len(self.mesh_blas)):
            self.vk.destroy_accel_struct(self.mesh_blas[i])
        self.mesh_blas.clear()
        for i in range(len(self.ring_tlas)):
            if self.ring_tlas[i].handle != 0:
                self.vk.destroy_accel_struct(self.ring_tlas[i])
        self.ring_tlas.clear()
        self.frame_tlas.handle = 0
        self.frame_tlas.buffer = GpuBuffer()
        self.frame_tlas.ctrl_ptr = 0

        for i in range(len(self.ring_as_instances)):
            if self.ring_as_instances[i].size > 0:
                self.vk.destroy_buffer(self.ring_as_instances[i])
        self.ring_as_instances.clear()
        self.buf_as_instances = GpuBuffer()

        if self.buf_as_scratch.size > 0:
            self.vk.destroy_buffer(self.buf_as_scratch)
            self.buf_as_scratch = GpuBuffer()

        self.ui_font_img.image = 0
        self.ui_font_img.view = 0
        self.ui_font_img.ctrl_ptr = 0
        for i in range(len(self.bindless_images)):
            self.vk.destroy_image(self.bindless_images[i])
        self.bindless_images.clear()
        for s_i in range(len(self.deferred_destroy_images)):
            for d_i in range(len(self.deferred_destroy_images[s_i])):
                self.vk.destroy_image(self.deferred_destroy_images[s_i][d_i])
            self.deferred_destroy_images[s_i].clear()
        for s_i in range(len(self.pending_bindless_desc_writes)):
            self.pending_bindless_desc_writes[s_i].clear()
        self.bindless_free_slots.clear()
        self.bindless_resident_min_mip.clear()
        self.bindless_vram_bytes.clear()
        self.bindless_layouts_init.clear()
        self._free_aliased_targets()
        self.vk.destroy_image(self.hiz_img)
        self.vk.destroy_image(self.vsm_img)
        self.vk.destroy_image(self.brdf_lut_img)
        self.vk.destroy_image(self.ltc_lut_img)
        self.vk.destroy_image(self.env_cube_img)

        self.vk.destroy_sampler(self.samp_aniso16)
        for m_i in range(len(self.samp_aniso16_min_lod)):
            self.vk.destroy_sampler(self.samp_aniso16_min_lod[m_i])
        self.samp_aniso16_min_lod.clear()
        self.vk.destroy_sampler(self.samp_linear_clamp)
        self.vk.destroy_sampler(self.samp_nearest)

        if self.pipe_meshlet_cull != 0:
            self.vk.destroy_pipeline(self.pipe_meshlet_cull)
            self.pipe_meshlet_cull = 0
        if self.pipe_hiz_downsample != 0:
            self.vk.destroy_pipeline(self.pipe_hiz_downsample)
            self.pipe_hiz_downsample = 0
        if self.pipe_visbuffer_gfx != 0:
            self.vk.destroy_pipeline(self.pipe_visbuffer_gfx)
            self.pipe_visbuffer_gfx = 0
        if self.pipe_visbuffer_alphamask_gfx != 0:
            self.vk.destroy_pipeline(self.pipe_visbuffer_alphamask_gfx)
            self.pipe_visbuffer_alphamask_gfx = 0
        if self.pipe_npr_outline_gfx != 0:
            self.vk.destroy_pipeline(self.pipe_npr_outline_gfx)
            self.pipe_npr_outline_gfx = 0
        if self.pipe_vsm_mark != 0:
            self.vk.destroy_pipeline(self.pipe_vsm_mark)
            self.pipe_vsm_mark = 0
        if self.pipe_vsm_gfx != 0:
            self.vk.destroy_pipeline(self.pipe_vsm_gfx)
            self.pipe_vsm_gfx = 0
        if self.pipe_vsm_alphamask_gfx != 0:
            self.vk.destroy_pipeline(self.pipe_vsm_alphamask_gfx)
            self.pipe_vsm_alphamask_gfx = 0
        if self.pipe_surfcache != 0:
            self.vk.destroy_pipeline(self.pipe_surfcache)
            self.pipe_surfcache = 0
        if self.pipe_cluster_cull != 0:
            self.vk.destroy_pipeline(self.pipe_cluster_cull)
            self.pipe_cluster_cull = 0
        if self.pipe_dq_skinning != 0:
            self.vk.destroy_pipeline(self.pipe_dq_skinning)
            self.pipe_dq_skinning = 0
        if self.pipe_tile_classify != 0:
            self.vk.destroy_pipeline(self.pipe_tile_classify)
            self.pipe_tile_classify = 0
        if self.pipe_shade_sky != 0:
            self.vk.destroy_pipeline(self.pipe_shade_sky)
            self.pipe_shade_sky = 0
        if self.pipe_shade_unlit != 0:
            self.vk.destroy_pipeline(self.pipe_shade_unlit)
            self.pipe_shade_unlit = 0
        if self.pipe_shade_pbr != 0:
            self.vk.destroy_pipeline(self.pipe_shade_pbr)
            self.pipe_shade_pbr = 0
        if self.pipe_shade_foliage_sss != 0:
            self.vk.destroy_pipeline(self.pipe_shade_foliage_sss)
            self.pipe_shade_foliage_sss = 0
        if self.pipe_shade_complex != 0:
            self.vk.destroy_pipeline(self.pipe_shade_complex)
            self.pipe_shade_complex = 0
        if self.pipe_vat_crowd_eval != 0:
            self.vk.destroy_pipeline(self.pipe_vat_crowd_eval)
            self.pipe_vat_crowd_eval = 0
        if self.pipe_terrain_clipmap_eval != 0:
            self.vk.destroy_pipeline(self.pipe_terrain_clipmap_eval)
            self.pipe_terrain_clipmap_eval = 0
        if self.pipe_svgf_reproject != 0:
            self.vk.destroy_pipeline(self.pipe_svgf_reproject)
            self.pipe_svgf_reproject = 0
        if self.pipe_svgf_atrous != 0:
            self.vk.destroy_pipeline(self.pipe_svgf_atrous)
            self.pipe_svgf_atrous = 0
        if self.pipe_oit_vol != 0:
            self.vk.destroy_pipeline(self.pipe_oit_vol)
            self.pipe_oit_vol = 0
        if self.pipe_luma_hist != 0:
            self.vk.destroy_pipeline(self.pipe_luma_hist)
            self.pipe_luma_hist = 0
        if self.pipe_adapt_exp != 0:
            self.vk.destroy_pipeline(self.pipe_adapt_exp)
            self.pipe_adapt_exp = 0
        if self.pipe_bloom_down != 0:
            self.vk.destroy_pipeline(self.pipe_bloom_down)
            self.pipe_bloom_down = 0
        if self.pipe_bloom_up != 0:
            self.vk.destroy_pipeline(self.pipe_bloom_up)
            self.pipe_bloom_up = 0
        if self.pipe_bokeh_dof != 0:
            self.vk.destroy_pipeline(self.pipe_bokeh_dof)
            self.pipe_bokeh_dof = 0
        if self.pipe_tsr_display != 0:
            self.vk.destroy_pipeline(self.pipe_tsr_display)
            self.pipe_tsr_display = 0
        if self.pipe_spec_reblur_temp != 0:
            self.vk.destroy_pipeline(self.pipe_spec_reblur_temp)
            self.pipe_spec_reblur_temp = 0
        if self.pipe_spec_reblur_atrous != 0:
            self.vk.destroy_pipeline(self.pipe_spec_reblur_atrous)
            self.pipe_spec_reblur_atrous = 0
        if self.pipe_froxel_inject != 0:
            self.vk.destroy_pipeline(self.pipe_froxel_inject)
            self.pipe_froxel_inject = 0
        if self.pipe_froxel_prefix != 0:
            self.vk.destroy_pipeline(self.pipe_froxel_prefix)
            self.pipe_froxel_prefix = 0
        if self.pipe_froxel_march != 0:
            self.vk.destroy_pipeline(self.pipe_froxel_march)
            self.pipe_froxel_march = 0
        if self.pipe_particle_clear != 0:
            self.vk.destroy_pipeline(self.pipe_particle_clear)
            self.pipe_particle_clear = 0
        if self.pipe_particle_sim != 0:
            self.vk.destroy_pipeline(self.pipe_particle_sim)
            self.pipe_particle_sim = 0
        if self.pipe_particle_comp != 0:
            self.vk.destroy_pipeline(self.pipe_particle_comp)
            self.pipe_particle_comp = 0
        if self.pipe_water_refract != 0:
            self.vk.destroy_pipeline(self.pipe_water_refract)
            self.pipe_water_refract = 0
        if self.pipe_rt_shadow_trace != 0:
            self.vk.destroy_pipeline(self.pipe_rt_shadow_trace)
            self.pipe_rt_shadow_trace = 0
        if self.pipe_rt_shadow_filter != 0:
            self.vk.destroy_pipeline(self.pipe_rt_shadow_filter)
            self.pipe_rt_shadow_filter = 0
        if self.pipe_glass_refract != 0:
            self.vk.destroy_pipeline(self.pipe_glass_refract)
            self.pipe_glass_refract = 0
        if self.pipe_tsr_v3_resolve != 0:
            self.vk.destroy_pipeline(self.pipe_tsr_v3_resolve)
            self.pipe_tsr_v3_resolve = 0
        if self.pipe_motion_blur != 0:
            self.vk.destroy_pipeline(self.pipe_motion_blur)
            self.pipe_motion_blur = 0
        if self.pipe_display_tonemap != 0:
            self.vk.destroy_pipeline(self.pipe_display_tonemap)
            self.pipe_display_tonemap = 0
        if self.pipe_display_tonemap_direct != 0:
            self.vk.destroy_pipeline(self.pipe_display_tonemap_direct)
            self.pipe_display_tonemap_direct = 0
        if self.pipe_ui_overlay_direct != 0:
            self.vk.destroy_pipeline(self.pipe_ui_overlay_direct)
            self.pipe_ui_overlay_direct = 0
        if self.pipe_display_present_gfx != 0:
            self.vk.destroy_pipeline(self.pipe_display_present_gfx)
            self.pipe_display_present_gfx = 0
        if self.pipe_ui_present_gfx != 0:
            self.vk.destroy_pipeline(self.pipe_ui_present_gfx)
            self.pipe_ui_present_gfx = 0
        self.present_gfx_format = 0
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

        for i in range(len(self.ring_surfcache)):
            if self.ring_surfcache[i].size > 0:
                self.vk.destroy_buffer(self.ring_surfcache[i])
        self.ring_surfcache.clear()
        self.buf_surfcache = GpuBuffer()

        for i in range(len(self.ring_vsm_table)):
            if self.ring_vsm_table[i].size > 0:
                self.vk.destroy_buffer(self.ring_vsm_table[i])
        self.ring_vsm_table.clear()
        self.buf_vsm_table = GpuBuffer()

        for i in range(len(self.ring_oit_staging)):
            if self.ring_oit_staging[i].size > 0:
                self.vk.destroy_buffer(self.ring_oit_staging[i])
        self.ring_oit_staging.clear()

        for i in range(len(self.ring_meshlet_worklist)):
            if self.ring_meshlet_worklist[i].size > 0:
                self.vk.destroy_buffer(self.ring_meshlet_worklist[i])
        self.ring_meshlet_worklist.clear()

        for i in range(len(self.ring_draw_pairs)):
            if self.ring_draw_pairs[i].size > 0:
                self.vk.destroy_buffer(self.ring_draw_pairs[i])
        self.ring_draw_pairs.clear()

        self.staging_ring.shutdown(self.vk)

        self.slot_prof_pairs.clear()
        self.slot_frame_q.clear()
        self.slot_has_pending_queries.clear()

        self.vk.destroy_buffer(self.buf_vertices)
        self.vk.destroy_buffer(self.buf_indices)
        self.vk.destroy_buffer(self.buf_meshlets)
        self.vk.destroy_buffer(self.buf_materials)
        self.vk.destroy_buffer(self.buf_vsm_indirect)
        self.vk.destroy_buffer(self.buf_globals)
        self.vk.destroy_buffer(self.buf_history)
        self.vk.destroy_buffer(self.buf_oit_nodes)
        self.vk.destroy_buffer(self.buf_skinning)
        self.vk.destroy_buffer(self.buf_post_scratch)
        self.vk.destroy_buffer(self.buf_ldr_readback)
        for i in range(len(self.readback_extra)):
            self.vk.destroy_buffer(self.readback_extra[i])
        self.readback_extra.clear()
        self.pending_slot = -1
        self.readback_ready_slot = 0
        self.vk.destroy_buffer(self.buf_staging)

        if self.vk.pipeline_cache != 0:
            _ = self.vk.save_pipeline_cache("build/vk_pso.cache")
        self.vk.close()
        self.active = False

    def __deinit__(deinit self):
        self.shutdown()
