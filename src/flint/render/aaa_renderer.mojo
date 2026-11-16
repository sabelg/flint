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
from std.math import cos, max, min, sin, sqrt, tan
from std.memory import bitcast
from flint.core.math import Vec3
from flint.core.transform import Mat4
from flint.assets.model import NativeMesh
from flint.assets.bc_compress import (
     downsample_argb_2x,
    downsample_normals_2x,
    encode_bc5_normals,
    encode_bc6h_hdr,
    encode_bc7_rgba,
)
from flint.assets.meshlet import AAA_VERTEX_BYTES, AaaGeometryPool, CookedMeshEntry
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
    ACCESS_TRANSFER_WRITE,
    BUFFER_USAGE_AS_BUILD_INPUT_KHR,
    BUFFER_USAGE_DEVICE_ADDRESS,
    COMPARE_GREATER_OR_EQUAL,
    COMPARE_LESS_OR_EQUAL,
    CULL_MODE_NONE,
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
    PIPELINE_COLOR_ATTACHMENT_OUTPUT,
    PIPELINE_COMPUTE,
    PIPELINE_DRAW_INDIRECT,
    PIPELINE_EARLY_FRAGMENT_TESTS,
    PIPELINE_FRAGMENT_SHADER,
    PIPELINE_HOST,
    PIPELINE_LATE_FRAGMENT_TESTS,
    PIPELINE_TOP,
    PIPELINE_TRANSFER,
    VulkanAccelStruct,
    VulkanGpu,
    VulkanImage,
    get32,
    put32,
    put_float,
)
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_MAX_BINDLESS,
    AAA_PUSH_BYTES,
    AAA_SINGLE_SAMPLERS,
    AAA_STORAGE_COUNT,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_BRDF_LUT,
    BIND_AAA_DEPTH_TEX,
    BIND_AAA_ENV_CUBE,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    BIND_AAA_INDICES,
    BIND_AAA_INDIRECT,
    BIND_AAA_INSTANCES,
    BIND_AAA_MATERIALS,
    BIND_AAA_MESHLETS,
    BIND_AAA_OIT_NODES,
    BIND_AAA_SURFCACHE,
    BIND_AAA_TLAS,
    BIND_AAA_VERTICES,
    BIND_AAA_VIS_TEX,
    BIND_AAA_VSM_TABLE,
    BIND_AAA_VSM_TEX,
    make_reverse_z_perspective,
    meshlet_cull_shader,
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
    vsm_mark_pages_shader,
    vsm_page_vertex_shader,
)
from flint.render.aaa_ibl import (
    BakedIblEnvironment,
    SkyDomeParams,
    bake_bc6h_ibl_cubemap_and_sh9,
    bake_split_sum_brdf_lut_rg16f,
)
from flint.render.aaa_raytracing import write_vk_as_instance_64b
from flint.render.aaa_gi_svgf import (
    AAA_FLAG_FURNACE_WHITE_DOME,
    AAA_FLAG_RT_REFLECTIONS,
    AAA_FLAG_RT_SHADOWS,
    AAA_FLAG_RT_SURFACE_GI,
    AAA_FLAG_VSM,
    aaa_deferred_shade_shader,
     surface_cache_update_shader,
    svgf_atrous_shader,
)
from flint.render.aaa_oit_volumetrics import (
    AAA_MAX_OIT_PRIMS,
    AaaVolumetricPlume,
    oit_and_6way_volumetric_shader,
)
from flint.render.aaa_post_tsr import (
    AAA_DISPLAY_AGX_REC709,
    tsr_v2_and_display_shader,
)
from flint.render.aaa_render_graph import (
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

    # Descriptor & Pipeline handles
    var dsl: Int
    var pipe_layout: Int
    var desc_pool: Int
    var desc_set: Int
    var pipe_meshlet_cull: Int
    var pipe_visbuffer_gfx: Int
    var pipe_vsm_mark: Int
    var pipe_vsm_gfx: Int
    var pipe_surfcache: Int
    var pipe_deferred_shade: Int
    var pipe_svgf_atrous: Int
    var pipe_oit_vol: Int
    var pipe_tsr_display: Int

    # Samplers & Images
    var samp_aniso16: Int
    var samp_linear_clamp: Int
    var samp_nearest: Int
    var vis_img: VulkanImage
    var depth_img: VulkanImage
    var vsm_img: VulkanImage
    var brdf_lut_img: VulkanImage
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
    var buf_staging: GpuBuffer

    # Ray-Tracing Acceleration Structures
    var mesh_blas: List[VulkanAccelStruct]
    var frame_tlas: VulkanAccelStruct
    var buf_as_instances: GpuBuffer
    var buf_as_scratch: GpuBuffer

    # CPU Scene & Subsystems
    var geom: AaaGeometryPool
    var materials: List[AaaMaterial]
    var frame_instances: List[AaaDrawInstance]
    var frame_plumes: List[AaaVolumetricPlume]
    var vsm: VsmPageManager
    var render_graph: AaaRenderGraph
    var streamer: AaaResidencyStreamer
    var sh9_irradiance: List[Vec3]
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
    var enable_rt_reflections: Bool
    var enable_rt_gi: Bool
    var furnace_white_dome: Bool
    var gi_rays_per_pixel: Int
    var svgf_passes: Int
    var display_mode: UInt32
    var tsr_history_weight: Float32
    var frame_index: Int
    var last_culled_meshlet_draws: Int

    def __init__(
        out self,
        width: Int = 640,
        height: Int = 360,
        tsr_scale: Float32 = 1.0,
        max_draws: Int = 131072,
        max_instances: Int = 32768,
    ):
        self.vk = VulkanGpu()
        self.active = False
        self.output_w = max(16, width)
        self.output_h = max(16, height)
        self.tsr_scale = max(Float32(0.50), min(Float32(1.0), tsr_scale))
        self.render_w = max(8, Int(Float32(self.output_w) * self.tsr_scale + Float32(0.5)))
        self.render_h = max(8, Int(Float32(self.output_h) * self.tsr_scale + Float32(0.5)))
        self.max_draws = max(256, max_draws)
        self.max_instances = max(256, max_instances)
        self.max_surfcache_tris = 65536

        self.dsl = 0
        self.pipe_layout = 0
        self.desc_pool = 0
        self.desc_set = 0
        self.pipe_meshlet_cull = 0
        self.pipe_visbuffer_gfx = 0
        self.pipe_vsm_mark = 0
        self.pipe_vsm_gfx = 0
        self.pipe_surfcache = 0
        self.pipe_deferred_shade = 0
        self.pipe_svgf_atrous = 0
        self.pipe_oit_vol = 0
        self.pipe_tsr_display = 0

        self.samp_aniso16 = 0
        self.samp_linear_clamp = 0
        self.samp_nearest = 0
        self.vis_img = VulkanImage()
        self.depth_img = VulkanImage()
        self.vsm_img = VulkanImage()
        self.brdf_lut_img = VulkanImage()
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
        self.buf_staging = GpuBuffer()

        self.mesh_blas = List[VulkanAccelStruct]()
        self.frame_tlas = VulkanAccelStruct()
        self.buf_as_instances = GpuBuffer()
        self.buf_as_scratch = GpuBuffer()

        self.geom = AaaGeometryPool()
        self.materials = List[AaaMaterial]()
        self.frame_instances = List[AaaDrawInstance]()
        self.frame_plumes = List[AaaVolumetricPlume]()
        self.vsm = VsmPageManager()
        self.render_graph = AaaRenderGraph()
        self.streamer = AaaResidencyStreamer()
        self.sh9_irradiance = List[Vec3](length=9, fill=Vec3(0.0, 0.0, 0.0))
        self.env_mip_levels = 5

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
        self.enable_rt_reflections = False
        self.enable_rt_gi = False
        self.furnace_white_dome = False
        self.gi_rays_per_pixel = 12
        self.svgf_passes = 2
        self.display_mode = AAA_DISPLAY_AGX_REC709
        self.tsr_history_weight = Float32(0.82)
        self.frame_index = 0
        self.last_culled_meshlet_draws = 0

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

        var pool_set = self.vk.allocate_aaa_descriptor_set(
            self.dsl,
            AAA_STORAGE_COUNT,
            AAA_SINGLE_SAMPLERS,
            AAA_MAX_BINDLESS,
            use_rt,
        )
        self.desc_pool = pool_set[0]
        self.desc_set = pool_set[1]
        if self.desc_set == 0:
            return False

        # Compile all 9 AAA Hardware Graphics & Compute SPIR-V 1.5 Pipelines!
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
            cull_mode=CULL_MODE_NONE,
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
            cull_mode=CULL_MODE_NONE,
        )
        self.pipe_surfcache = self.vk.create_aaa_compute_pipeline(self.pipe_layout, surface_cache_update_shader(use_rt))
        self.pipe_deferred_shade = self.vk.create_aaa_compute_pipeline(self.pipe_layout, aaa_deferred_shade_shader(use_rt))
        self.pipe_svgf_atrous = self.vk.create_aaa_compute_pipeline(self.pipe_layout, svgf_atrous_shader())
        self.pipe_oit_vol = self.vk.create_aaa_compute_pipeline(self.pipe_layout, oit_and_6way_volumetric_shader())
        self.pipe_tsr_display = self.vk.create_aaa_compute_pipeline(self.pipe_layout, tsr_v2_and_display_shader())

        if (
            self.pipe_meshlet_cull == 0
            or self.pipe_visbuffer_gfx == 0
            or self.pipe_vsm_mark == 0
            or self.pipe_vsm_gfx == 0
            or self.pipe_surfcache == 0
            or self.pipe_deferred_shade == 0
            or self.pipe_svgf_atrous == 0
            or self.pipe_oit_vol == 0
            or self.pipe_tsr_display == 0
        ):
            return False

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
        self.env_cube_img = self.vk.create_image_2d(
            64, 64, FORMAT_BC6H_UFLOAT, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED, mip_levels=self.env_mip_levels, cube=True
        )
        if self.vis_img.image == 0 or self.depth_img.image == 0 or self.vsm_img.image == 0 or self.brdf_lut_img.image == 0 or self.env_cube_img.image == 0:
            return False

        # Allocate Storage Buffers
        var max_pix = max(self.output_w * self.output_h, self.render_w * self.render_h)
        var bda_flags = BUFFER_USAGE_DEVICE_ADDRESS | BUFFER_USAGE_AS_BUILD_INPUT_KHR
        self.buf_vertices = self.vk.create_buffer_bda(4 * 1024 * 1024, extra_usage=bda_flags, host_visible=True)
        self.buf_indices = self.vk.create_buffer_bda(4 * 1024 * 1024, extra_usage=bda_flags, host_visible=True)
        self.buf_meshlets = self.vk.create_buffer(2 * 1024 * 1024, local=False)
        self.buf_instances = self.vk.create_buffer(self.max_instances * AAA_INSTANCE_WORDS * 4, local=False)
        self.buf_materials = self.vk.create_buffer(1024 * AAA_MATERIAL_WORDS * 4, local=False)
        self.buf_indirect = self.vk.create_buffer(64 + self.max_draws * 16, local=False, cpu_reads=True)
        self.buf_globals = self.vk.create_buffer(4096, local=False)
        self.buf_hdr_out = self.vk.create_buffer(max_pix * 8 * 4, local=True)
        self.buf_history = self.vk.create_buffer(max_pix * 20 * 4, local=False, cpu_reads=True)
        self.buf_vsm_table = self.vk.create_buffer(VSM_TABLE_WORDS * 4, local=False, cpu_reads=True)
        self.buf_surfcache = self.vk.create_buffer((1024 + self.max_surfcache_tris * 4) * 4, local=False)
        self.buf_oit_nodes = self.vk.create_buffer((4096 + max_pix) * 4, local=False)
        self.buf_staging = self.vk.create_buffer(8 * 1024 * 1024, local=False)
        if self.vk.has_ray_query:
            self.buf_as_instances = self.vk.create_buffer_bda(self.max_instances * 64, extra_usage=bda_flags, host_visible=True)

        _ = external_call["memset", Int](self.buf_history.cpu_address, Int32(0), self.buf_history.size)
        _ = external_call["memset", Int](self.buf_vsm_table.cpu_address, Int32(0), self.buf_vsm_table.size)

        # Bind all 12 Storage Buffers + 5 Single Samplers
        self.vk.write_descriptor(self.desc_set, BIND_AAA_VERTICES, self.buf_vertices)
        self.vk.write_descriptor(self.desc_set, BIND_AAA_INDICES, self.buf_indices)
        self.vk.write_descriptor(self.desc_set, BIND_AAA_MESHLETS, self.buf_meshlets)
        self.vk.write_descriptor(self.desc_set, BIND_AAA_INSTANCES, self.buf_instances)
        self.vk.write_descriptor(self.desc_set, BIND_AAA_MATERIALS, self.buf_materials)
        self.vk.write_descriptor(self.desc_set, BIND_AAA_INDIRECT, self.buf_indirect)
        self.vk.write_descriptor(self.desc_set, BIND_AAA_GLOBALS, self.buf_globals)
        self.vk.write_descriptor(self.desc_set, BIND_AAA_HDR_OUT, self.buf_hdr_out)
        self.vk.write_descriptor(self.desc_set, BIND_AAA_HISTORY, self.buf_history)
        self.vk.write_descriptor(self.desc_set, BIND_AAA_VSM_TABLE, self.buf_vsm_table)
        self.vk.write_descriptor(self.desc_set, BIND_AAA_SURFCACHE, self.buf_surfcache)
        self.vk.write_descriptor(self.desc_set, BIND_AAA_OIT_NODES, self.buf_oit_nodes)

        self.vk.write_aaa_texture(self.desc_set, BIND_AAA_VIS_TEX, 0, self.samp_nearest, self.vis_img.view)
        self.vk.write_aaa_texture(self.desc_set, BIND_AAA_DEPTH_TEX, 0, self.samp_linear_clamp, self.depth_img.view)
        self.vk.write_aaa_texture(self.desc_set, BIND_AAA_VSM_TEX, 0, self.samp_linear_clamp, self.vsm_img.view)
        self.vk.write_aaa_texture(self.desc_set, BIND_AAA_ENV_CUBE, 0, self.samp_linear_clamp, self.env_cube_img.view)
        self.vk.write_aaa_texture(self.desc_set, BIND_AAA_BRDF_LUT, 0, self.samp_linear_clamp, self.brdf_lut_img.view)

        # Bake & Upload 64x64 RG16F Split-Sum BRDF LUT + Default BC6H Sky Cubemap + Initial Depth Transitions!
        var lut_bytes = bake_split_sum_brdf_lut_rg16f(64, 128)
        _ = external_call["memcpy", Int](self.buf_staging.cpu_address, Int(lut_bytes.unsafe_ptr()), len(lut_bytes))
        _ = len(lut_bytes)

        if not self.vk.begin_commands(0):
            return False
        self.vk.transition_image(0, self.brdf_lut_img, LAYOUT_TRANSFER_TARGET, PIPELINE_TOP, 0, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE)
        self.vk.upload_image_mip(0, self.brdf_lut_img, self.buf_staging, 0, 0, 64, 64)
        self.vk.transition_image(0, self.brdf_lut_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_TRANSFER, ACCESS_TRANSFER_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ)

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

        if self.vk.has_ray_query:
            var inst_da = self.vk.buffer_device_address(self.buf_as_instances)
            self.frame_tlas = self.vk.build_tlas_instances(0, inst_da, 0, self.buf_as_scratch)
            self.vk.memory_barrier2(
                0,
                PIPELINE_AS_BUILD_KHR,
                Int(ACCESS_AS_WRITE_KHR),
                PIPELINE_COMPUTE,
                Int(ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
            )

        var init_fence = self.vk.submit(0)
        if init_fence == 0 or not self.vk.wait(init_fence):
            return False

        if self.vk.has_ray_query and self.frame_tlas.handle != 0:
            self.vk.write_aaa_tlas(self.desc_set, BIND_AAA_TLAS, self.frame_tlas.handle)

        # Upload default BC6H Golden-Hour / Sky IBL + White dummy BC7 texture at bindless[0]
        self.upload_ibl_environment(bake_bc6h_ibl_cubemap_and_sh9(SkyDomeParams(uniform_furnace=False), base_dim=64, mip_levels=self.env_mip_levels))
        var white_px = List[UInt32](length=16, fill=0xFFFFFFFF)
        var white_slot = self.upload_bc7_texture(white_px, 4, 4, srgb=False, generate_mips=False)
        if white_slot == 0 and len(self.bindless_images) > 0:
            for s_idx in range(1, AAA_MAX_BINDLESS):
                self.vk.write_aaa_texture(self.desc_set, BIND_AAA_BINDLESS_TEX, s_idx, self.samp_aniso16, self.bindless_images[0].view)

        self._build_default_render_graph()
        return True

    # ========================================================================
    # Asset Registration: Bindless BC7 / BC5 / BC6H Textures, Meshes & BLAS
    # ========================================================================

    def upload_ibl_environment(mut self, ibl: BakedIblEnvironment):
        self.sh9_irradiance = ibl.sh9.copy()
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
        self.vk.write_aaa_texture(self.desc_set, BIND_AAA_BINDLESS_TEX, slot, self.samp_aniso16, img.view)
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
        self.vk.write_aaa_texture(self.desc_set, BIND_AAA_BINDLESS_TEX, slot, self.samp_aniso16, img.view)
        self.bindless_images.append(img^)
        return slot

    def upload_bc6h_texture(mut self, hdr_pixels: List[Vec3], width: Int, height: Int) -> Int:
        if len(self.bindless_images) >= AAA_MAX_BINDLESS:
            return -1
        var img = self.vk.create_image_2d(width, height, FORMAT_BC6H_UFLOAT, IMAGE_USAGE_TRANSFER_TARGET | IMAGE_USAGE_SAMPLED, mip_levels=1)
        if img.image == 0:
            return -1
        var bc6 = encode_bc6h_hdr(hdr_pixels, width, height)
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
        self.vk.write_aaa_texture(self.desc_set, BIND_AAA_BINDLESS_TEX, slot, self.samp_linear_clamp, img.view)
        self.bindless_images.append(img^)
        return slot

    def add_material(mut self, mat: AaaMaterial) -> Int:
        var id = len(self.materials)
        self.materials.append(mat.copy())
        var words = List[UInt32]()
        for m in self.materials:
            m.append_words(words)
        _ = external_call["memcpy", Int](self.buf_materials.cpu_address, Int(words.unsafe_ptr()), len(words) * 4)
        _ = len(words)
        return id

    def add_mesh(mut self, mesh: NativeMesh, material_id: Int) -> CookedMeshEntry:
        """Partitions `mesh` into 64v/124t GPU Meshlets, uploads SSBOs, and builds Hardware `BLAS`!"""
        var entry = self.geom.append_mesh(mesh, material_id)
        _ = external_call[
            "memcpy", Int
        ](self.buf_vertices.cpu_address, Int(self.geom.vertex_words.unsafe_ptr()), len(self.geom.vertex_words) * 4)
        _ = external_call[
            "memcpy", Int
        ](self.buf_indices.cpu_address, Int(self.geom.index_words.unsafe_ptr()), len(self.geom.index_words) * 4)
        _ = external_call[
            "memcpy", Int
        ](self.buf_meshlets.cpu_address, Int(self.geom.meshlet_words.unsafe_ptr()), len(self.geom.meshlet_words) * 4)

        if self.vk.has_ray_query and entry.triangle_count > 0:
            var v_addr = self.vk.buffer_device_address(self.buf_vertices)
            var i_addr = self.vk.buffer_device_address(self.buf_indices) + UInt64(entry.first_index * 4)
            var max_v = entry.first_vertex + entry.vertex_count - 1
            if self.vk.begin_commands(0):
                var blas = self.vk.build_blas_triangles(
                    0,
                    v_addr,
                    AAA_VERTEX_BYTES,
                    max_v,
                    i_addr,
                    entry.triangle_count,
                    self.buf_as_scratch,
                )
                self.vk.memory_barrier2(0, PIPELINE_AS_BUILD_KHR, Int(ACCESS_AS_WRITE_KHR), PIPELINE_AS_BUILD_KHR, Int(ACCESS_AS_READ_KHR))
                var f = self.vk.submit(0)
                if f != 0:
                    _ = self.vk.wait(f)
                self.mesh_blas.append(blas^)
            else:
                self.mesh_blas.append(VulkanAccelStruct())
        else:
            self.mesh_blas.append(VulkanAccelStruct())

        self.vsm.invalidate_all()
        return entry^

    # ========================================================================
    # Frame Submission & Execution
    # ========================================================================

    def begin_frame(
        mut self,
        cam_pos: Vec3,
        cam_target: Vec3,
        fov_y_deg: Float32 = 55.0,
        near_z: Float32 = 0.10,
    ):
        self.cam_pos = cam_pos
        var fwd = (cam_target - cam_pos).normalized()
        var up_ref = Vec3(0.0, 0.0, 1.0) if abs(fwd.y) > Float32(0.95) else Vec3(0.0, 1.0, 0.0)
        var right = up_ref.cross(fwd).normalized()
        var up = fwd.cross(right).normalized()
        self.cam_fwd = fwd
        self.cam_right = right
        self.cam_up = up
        self.fov_y_deg = fov_y_deg
        self.near_z = near_z
        self.frame_instances.clear()
        self.frame_plumes.clear()

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

    def render_frame(mut self) -> Bool:
        if not self.active:
            return False

        var n_inst = len(self.frame_instances)
        var max_ml_per_inst = 1
        var max_tri_per_sc_inst = 1
        var sc_cursor = 256  # Leave words [0..1023] for up to 256 instance headers!
        var sc_inst_count = min(256, n_inst)

        # 1. Upload Instance SSBO + Surface-Cache Headers + TLAS Instance Records
        for i in range(n_inst):
            var di = self.frame_instances[i].copy()
            var me = self.geom.meshes[di.mesh_id].copy()
            if me.meshlet_count > max_ml_per_inst:
                max_ml_per_inst = me.meshlet_count
            var dst = self.buf_instances.cpu_address + i * AAA_INSTANCE_WORDS * 4
            for k in range(16):
                put_float(dst + k * 4, di.world_mat.values[k])
                put_float(dst + (16 + k) * 4, di.prev_world_mat.values[k])
            put32(dst + 32 * 4, UInt32(me.first_meshlet))
            put32(dst + 33 * 4, UInt32(me.meshlet_count))
            put32(dst + 34 * 4, UInt32(0xFFFFFFFF) if di.material_override < 0 else UInt32(di.material_override))
            put_float(dst + 35 * 4, di.uniform_scale)

            var mat_final = me.material_id if di.material_override < 0 else di.material_override
            var my_sc_base = 256
            if i < 256 and sc_cursor + me.triangle_count < self.max_surfcache_tris:
                my_sc_base = sc_cursor
                sc_cursor += me.triangle_count
                if me.triangle_count > max_tri_per_sc_inst:
                    max_tri_per_sc_inst = me.triangle_count
                var sc_hdr = self.buf_surfcache.cpu_address + i * 16
                put32(sc_hdr + 0, UInt32(my_sc_base))
                put32(sc_hdr + 4, UInt32(me.first_index))
                put32(sc_hdr + 8, UInt32(me.triangle_count))
                put32(sc_hdr + 12, UInt32(mat_final))

            if self.vk.has_ray_query and di.mesh_id < len(self.mesh_blas):
                var blas_da = self.mesh_blas[di.mesh_id].device_address
                write_vk_as_instance_64b(
                    self.buf_as_instances.cpu_address + i * 64,
                    di.world_mat,
                    my_sc_base,
                    blas_da,
                    cull_disable=True,
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

        # Halton (2, 3) sub-pixel TSR jitter when sub-native TSR is active
        var jx_clip = Float32(0.0)
        var jy_clip = Float32(0.0)
        if self.tsr_scale < Float32(0.99) and self.tsr_history_weight > Float32(0.01):
            var h_idx = (self.frame_index & 7) + 1
            var jx_px = (Float32(h_idx & 1) * Float32(0.5) + Float32((h_idx >> 1) & 1) * Float32(0.25)) - Float32(0.375)
            var jy_px = (Float32(h_idx % 3) / Float32(3.0)) - Float32(0.333333)
            jx_clip = (jx_px * Float32(2.0)) / Float32(self.render_w)
            jy_clip = (jy_px * Float32(2.0)) / Float32(self.render_h)

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

        var r_flags: UInt32 = 0
        if self.enable_vsm and not self.furnace_white_dome:
            r_flags |= AAA_FLAG_VSM
        if self.vk.has_ray_query and not self.furnace_white_dome and n_inst > 0:
            if self.enable_rt_shadows:
                r_flags |= AAA_FLAG_RT_SHADOWS
            if self.enable_rt_reflections:
                r_flags |= AAA_FLAG_RT_REFLECTIONS
            if self.enable_rt_gi:
                r_flags |= AAA_FLAG_RT_SURFACE_GI
        if self.furnace_white_dome:
            r_flags |= AAA_FLAG_FURNACE_WHITE_DOME
        put32(g_addr + 99 * 4, r_flags)
        put32(g_addr + 100 * 4, UInt32(self.frame_index))
        put_float(g_addr + 101 * 4, jx_clip)
        put_float(g_addr + 102 * 4, jy_clip)
        put_float(g_addr + 103 * 4, Float32(self.env_mip_levels - 1))

        # Reset indirect draw counter = 0
        put32(self.buf_indirect.cpu_address, 0)

        # 3. Rebuild per-frame Hardware TLAS if Ray Tracing is enabled!
        var use_rt_frame = self.vk.has_ray_query and n_inst > 0 and (
            self.enable_rt_shadows or self.enable_rt_reflections or self.enable_rt_gi
        )
        if use_rt_frame:
            if self.frame_tlas.handle != 0:
                self.vk.destroy_accel_struct(self.frame_tlas)
            if not self.vk.begin_commands(0):
                return False
            var inst_da = self.vk.buffer_device_address(self.buf_as_instances)
            self.frame_tlas = self.vk.build_tlas_instances(0, inst_da, n_inst, self.buf_as_scratch)
            self.vk.memory_barrier2(
                0,
                PIPELINE_AS_BUILD_KHR,
                Int(ACCESS_AS_WRITE_KHR),
                PIPELINE_COMPUTE,
                Int(ACCESS_AS_READ_KHR | ACCESS_SHADER_READ),
            )
            var f_tlas = self.vk.submit(0)
            if f_tlas == 0 or not self.vk.wait(f_tlas):
                return False
            if self.frame_tlas.handle != 0:
                self.vk.write_aaa_tlas(self.desc_set, BIND_AAA_TLAS, self.frame_tlas.handle)

        # ====================================================================
        # PASS GROUP A: GPU Meshlet Cull -> Hardware VisBuffer -> VSM Mark Pages
        # ====================================================================
        if not self.vk.begin_commands(0):
            return False

        if n_inst > 0:
            var push_cull = List[UInt32](length=16, fill=0)
            push_cull[0] = UInt32(n_inst)
            push_cull[1] = UInt32(self.max_draws)
            push_cull[2] = UInt32(1) if self.enable_cone_cull else UInt32(0)
            push_cull[3] = UInt32(1) if (self.enable_hiz_cull and self.frame_index > 0) else UInt32(0)
            push_cull[4] = UInt32(self.render_w)
            push_cull[5] = UInt32(self.render_h)
            self.vk.dispatch_aaa(
                0,
                self.pipe_meshlet_cull,
                self.pipe_layout,
                self.desc_set,
                push_cull,
                (n_inst + 7) // 8,
                (max_ml_per_inst + 7) // 8,
            )
            self.render_graph.emit_pass_barriers(self.vk, 0, 1)

        # Hardware `VK_KHR_dynamic_rendering` 64-Bit Visibility Buffer + Reverse-Z Depth
        self.vk.transition_image(
            0, self.vis_img, LAYOUT_COLOR_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE
        )
        self.vk.transition_image(
            0, self.depth_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE
        )
        self.vk.begin_rendering(
            0, 0, 0, self.render_w, self.render_h, True, self.vis_img.view, True, 0, 0, True, self.depth_img.view, True, 0.0
        )
        self.vk.set_viewport_and_scissor(0, 0, 0, self.render_w, self.render_h, flip_y=True)
        if n_inst > 0:
            var push_vb = List[UInt32](length=16, fill=0)
            push_vb[0] = 0
            push_vb[1] = bitcast[DType.uint32](jx_clip)
            push_vb[2] = bitcast[DType.uint32](jy_clip)
            self.vk.bind_graphics_pipeline(0, self.pipe_visbuffer_gfx, self.pipe_layout, self.desc_set, push_vb)
            self.vk.draw_indirect_count(
                0,
                self.buf_indirect,
                16,
                self.buf_indirect,
                0,
                min(self.max_draws, n_inst * max_ml_per_inst),
                stride=16,
            )
        self.vk.end_rendering(0)

        self.vk.transition_image(
            0, self.vis_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_COLOR_ATTACHMENT_OUTPUT, ACCESS_COLOR_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
        )
        self.vk.transition_image(
            0, self.depth_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
        )

        # VSM Mark Visible Virtual Shadow Pages (16k x 16k)
        if (r_flags & AAA_FLAG_VSM) != 0 and n_inst > 0:
            var push_vm = List[UInt32](length=16, fill=0)
            push_vm[0] = UInt32(self.render_w)
            push_vm[1] = UInt32(self.render_h)
            self.vk.dispatch_aaa(
                0, self.pipe_vsm_mark, self.pipe_layout, self.desc_set, push_vm, (self.render_w + 7) // 8, (self.render_h + 7) // 8
            )
            self.vk.memory_barrier2(0, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))

        var f_group_a = self.vk.submit(0)
        if f_group_a == 0 or not self.vk.wait(f_group_a):
            return False

        self.last_culled_meshlet_draws = Int(get32(self.buf_indirect.cpu_address))

        # ====================================================================
        # PASS GROUP B: VSM Demand-Paged Tile Raster + Lumen Surface-Cache +
        #               Deferred Multi-Lobe RT Shade + SVGF + OIT + TSR v2!
        # ====================================================================
        var vsm_allocs = List[Tuple[Int, Int, Int, Int]]()
        if (r_flags & AAA_FLAG_VSM) != 0 and n_inst > 0:
            var tbl = List[UInt32](length=VSM_TABLE_WORDS, fill=0)
            _ = external_call["memcpy", Int](Int(tbl.unsafe_ptr()), self.buf_vsm_table.cpu_address, VSM_TABLE_WORDS * 4)
            var pages = self.vsm.process_requests(tbl)
            _ = external_call["memcpy", Int](self.buf_vsm_table.cpu_address, Int(tbl.unsafe_ptr()), VSM_TABLE_WORDS * 4)
            _ = len(tbl)
            for pg in pages:
                vsm_allocs.append((pg.vx, pg.vy, pg.px, pg.py))

        if not self.vk.begin_commands(0):
            return False

        # Rasterize ONLY newly allocated / dirty `128x128` VSM physical tiles (cached static pages cost 0.0 ms!)
        if len(vsm_allocs) > 0 and self.last_culled_meshlet_draws > 0:
            self.vk.transition_image(
                0, self.vsm_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_EARLY_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE
            )
            for pg in vsm_allocs:
                var vx = pg[0]; var vy = pg[1]; var px = pg[2]; var py = pg[3]
                var rx = px * VSM_PAGE_TEXELS
                var ry = py * VSM_PAGE_TEXELS
                # Sub-frustum zoom mapping virtual page `(vx, vy)` into NDC `[-1..+1]^2`:
                # Note lu = 0.5 * lclip.x + 0.5 in [vx/128 .. (vx+1)/128] -> page_cx = 128 * lclip.x + (127 - 2*vx)
                # Note lv = -0.5 * lclip.y + 0.5 in [vy/128 .. (vy+1)/128] -> page_cy = 128 * lclip.y - (127 - 2*vy)
                var bx_f = Float32(127 - 2 * vx)
                var by_f = -Float32(127 - 2 * vy)
                self.vk.begin_rendering(
                    0, rx, ry, VSM_PAGE_TEXELS, VSM_PAGE_TEXELS, False, 0, False, 0, 0, True, self.vsm_img.view, True, 1.0
                )
                self.vk.set_viewport_and_scissor(0, rx, ry, VSM_PAGE_TEXELS, VSM_PAGE_TEXELS, flip_y=True)
                var push_vsm = List[UInt32](length=16, fill=0)
                push_vsm[0] = bitcast[DType.uint32](Float32(128.0))
                push_vsm[1] = bitcast[DType.uint32](Float32(128.0))
                push_vsm[2] = bitcast[DType.uint32](bx_f)
                push_vsm[3] = bitcast[DType.uint32](by_f)
                self.vk.bind_graphics_pipeline(0, self.pipe_vsm_gfx, self.pipe_layout, self.desc_set, push_vsm)
                # Draw all instance meshlets into this shadow page
                for i_idx in range(n_inst):
                    var di = self.frame_instances[i_idx].copy()
                    var me = self.geom.meshes[di.mesh_id].copy()
                    for m_loc in range(me.meshlet_count):
                        var ml_id = me.first_meshlet + m_loc
                        var tri_c = self.geom.meshlets[ml_id].tri_count
                        self.vk.draw(0, tri_c * 3, 1, 0, (i_idx << 16) | ml_id)
                self.vk.end_rendering(0)

            self.vk.transition_image(
                0, self.vsm_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE, PIPELINE_COMPUTE, ACCESS_SHADER_READ
            )

        # Update 2-Bounce Lumen Direct + Indirect Mesh Surface Cache!
        if use_rt_frame and (self.enable_rt_reflections or self.enable_rt_gi) and sc_inst_count > 0:
            for bounce in range(2 if self.enable_rt_gi else 1):
                var push_sc = List[UInt32](length=16, fill=0)
                push_sc[0] = UInt32(sc_inst_count)
                push_sc[1] = UInt32(bounce)
                self.vk.dispatch_aaa(
                    0,
                    self.pipe_surfcache,
                    self.pipe_layout,
                    self.desc_set,
                    push_sc,
                    (sc_inst_count + 3) // 4,
                    (max_tri_per_sc_inst + 15) // 16,
                )
                self.vk.memory_barrier2(
                    0, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
                )

        # Full Deferred `VisBuffer` Multi-Lobe + VSM + `SPV_KHR_ray_query` Shading!
        var push_sh = List[UInt32](length=16, fill=0)
        push_sh[0] = UInt32(self.render_w)
        push_sh[1] = UInt32(self.render_h)
        push_sh[2] = UInt32(max(1, self.gi_rays_per_pixel))
        self.vk.dispatch_aaa(
            0,
            self.pipe_deferred_shade,
            self.pipe_layout,
            self.desc_set,
            push_sh,
            (self.render_w + 7) // 8,
            (self.render_h + 7) // 8,
        )
        self.render_graph.emit_pass_barriers(self.vk, 0, 4)

        # `SVGF` Edge-Stopping `À-Trous` Wavelet Denoiser (even number of ping-pong steps returns result to `buf_hdr_out`!)
        if self.enable_rt_gi and self.svgf_passes > 0:
            var n_steps = ((self.svgf_passes + 1) // 2) * 2
            for st_i in range(n_steps):
                var push_sv = List[UInt32](length=16, fill=0)
                push_sv[0] = UInt32(self.render_w)
                push_sv[1] = UInt32(self.render_h)
                push_sv[2] = UInt32(1 << st_i)
                push_sv[3] = UInt32(st_i & 1)
                push_sv[4] = bitcast[DType.uint32](Float32(0.65))
                self.vk.dispatch_aaa(
                    0,
                    self.pipe_svgf_atrous,
                    self.pipe_layout,
                    self.desc_set,
                    push_sv,
                    (self.render_w + 7) // 8,
                    (self.render_h + 7) // 8,
                )
                self.vk.memory_barrier2(
                    0, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
                )

        # Per-Pixel 4-Node Exact `OIT` + 6-Way Lightmapped Volumetric Plumes + Reactive Mask!
        var push_oit = List[UInt32](length=16, fill=0)
        push_oit[0] = UInt32(self.render_w)
        push_oit[1] = UInt32(self.render_h)
        push_oit[2] = UInt32(len(self.frame_plumes))
        self.vk.dispatch_aaa(
            0,
            self.pipe_oit_vol,
            self.pipe_layout,
            self.desc_set,
            push_oit,
            (self.render_w + 7) // 8,
            (self.render_h + 7) // 8,
        )
        self.vk.memory_barrier2(
            0, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
        )

        # Disocclusion-Reactive `FlintTSR v2` (`render_w x render_h -> output_w x output_h`) + Display Output!
        var max_pix = max(self.output_w * self.output_h, self.render_w * self.render_h)
        var ping_off = max_pix * 8
        var pong_off = max_pix * 12
        var out_ldr_off = max_pix * 16
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
        push_tsr[7] = UInt32(out_ldr_off)
        push_tsr[8] = self.display_mode
        self.vk.dispatch_aaa(
            0,
            self.pipe_tsr_display,
            self.pipe_layout,
            self.desc_set,
            push_tsr,
            (self.output_w + 7) // 8,
            (self.output_h + 7) // 8,
        )
        self.vk.memory_barrier2(0, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_HOST, Int(ACCESS_HOST_READ))

        var f_end = self.vk.submit(0)
        if f_end == 0 or not self.vk.wait(f_end):
            return False

        self.prev_view_proj = vp_m^
        self.frame_index += 1
        return True

    def readback_pixels(self) -> List[UInt32]:
        var n = self.output_w * self.output_h
        var max_pix = max(n, self.render_w * self.render_h)
        var out_ldr_off = max_pix * 16
        var out = List[UInt32](length=n, fill=0)
        _ = external_call[
            "memcpy", Int
        ](Int(out.unsafe_ptr()), self.buf_history.cpu_address + out_ldr_off * 4, n * 4)
        return out^

    def readback_resolved_hdr(self) -> List[Vec3]:
        """Reads back linear HDR `(R, G, B)` resolved by `FlintTSR v2` before display OETF."""
        var n = self.output_w * self.output_h
        var max_pix = max(n, self.render_w * self.render_h)
        var last_written_hist = (max_pix * 12) if ((self.frame_index - 1) & 1) == 0 else (max_pix * 8)
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
        self.vk.destroy_accel_struct(self.frame_tlas)
        if self.buf_as_instances.size > 0:
            self.vk.destroy_buffer(self.buf_as_instances)
            self.buf_as_instances = GpuBuffer()
        if self.buf_as_scratch.size > 0:
            self.vk.destroy_buffer(self.buf_as_scratch)
            self.buf_as_scratch = GpuBuffer()

        for i in range(len(self.bindless_images)):
            self.vk.destroy_image(self.bindless_images[i])
        self.bindless_images.clear()
        self.vk.destroy_image(self.vis_img)
        self.vk.destroy_image(self.depth_img)
        self.vk.destroy_image(self.vsm_img)
        self.vk.destroy_image(self.brdf_lut_img)
        self.vk.destroy_image(self.env_cube_img)

        self.vk.destroy_sampler(self.samp_aniso16)
        self.vk.destroy_sampler(self.samp_linear_clamp)
        self.vk.destroy_sampler(self.samp_nearest)

        self.vk.destroy_pipeline(self.pipe_meshlet_cull)
        self.vk.destroy_pipeline(self.pipe_visbuffer_gfx)
        self.vk.destroy_pipeline(self.pipe_vsm_mark)
        self.vk.destroy_pipeline(self.pipe_vsm_gfx)
        self.vk.destroy_pipeline(self.pipe_surfcache)
        self.vk.destroy_pipeline(self.pipe_deferred_shade)
        self.vk.destroy_pipeline(self.pipe_svgf_atrous)
        self.vk.destroy_pipeline(self.pipe_oit_vol)
        self.vk.destroy_pipeline(self.pipe_tsr_display)
        self.vk.destroy_aaa_descriptors(self.desc_pool, self.dsl, self.pipe_layout)

        self.vk.destroy_buffer(self.buf_vertices)
        self.vk.destroy_buffer(self.buf_indices)
        self.vk.destroy_buffer(self.buf_meshlets)
        self.vk.destroy_buffer(self.buf_instances)
        self.vk.destroy_buffer(self.buf_materials)
        self.vk.destroy_buffer(self.buf_indirect)
        self.vk.destroy_buffer(self.buf_globals)
        self.vk.destroy_buffer(self.buf_hdr_out)
        self.vk.destroy_buffer(self.buf_history)
        self.vk.destroy_buffer(self.buf_vsm_table)
        self.vk.destroy_buffer(self.buf_surfcache)
        self.vk.destroy_buffer(self.buf_oit_nodes)
        self.vk.destroy_buffer(self.buf_staging)

        self.vk.close()
        self.active = False
