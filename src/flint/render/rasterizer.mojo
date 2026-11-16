"""Dependency-free CPU framebuffer, triangle rasterizer, and bitmap UI."""
from flint.core.log import LOG_GPU, log_error
from std.collections import Dict
from std.ffi import external_call
from std.math import sqrt, tan, sin, cos, floor, exp, log2
from std.memory import bitcast
from flint.core.math import Vec3
from flint.core.profiler import ZONE_DEBUG_DRAW, ZONE_HUD_LAYER
from flint.render.debug_draw import DebugDraw, DebugView
from flint.render.grading import build_tone_table
from flint.render.font import SdfFont, bitmap_glyph, decode_utf8
from flint.text.rich_text import parse_rich, layout_rich, plain_text, SPAN_BUTTON, BUTTON_LABEL
from flint.ui.button_glyphs import (
    glyph_shape, glyph_label, pad_button, face_color, SHAPE_KEYCAP, SHAPE_FACE, SHAPE_PILL, SHAPE_DPAD,
    BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_DPAD_UP, BUTTON_DPAD_RIGHT, BUTTON_DPAD_DOWN,
)
from flint.ui.layout import UiFrame, ui_frame, MAX_SAFE_MARGIN
from flint.ui.nine_slice import NineSlice, slate_panel, focus_frame, slice_map
from flint.render.lighting import LightingProfile, LocalLight
from flint.render.dither import DitherView, dither_view
from flint.render.gpu_layout import (
    ADDITIVE_FLAG, ALPHA_ERODE_FLAG, BLACKBODY_FLAG, CONSTANT_CAMERA, CONSTANT_CSM,
    CONSTANT_DECALS, CONSTANT_DITHER, CONSTANT_KEY, CONSTANT_LIGHTS, CONSTANT_PREV_CAMERA,
    CONSTANT_PROBES, CONSTANT_RADIANCE_CASCADES, CONSTANT_SDF_OCCLUDERS, CONSTANT_SPHERES,
    CONSTANT_V10_POST, CONSTANT_V11_POST, CONSTANT_V7_POST, CONSTANT_V8_ATMOS, CONSTANT_V9_CLOUDS,
    CONSTANT_VOL_POST, CONSTANT_WORLD, CSM_RES, DITHER_FLAG, EFFECT_AUTO_EXPOSURE, EFFECT_BILINEAR,
    EFFECT_CHROMATIC_REFRACT, EFFECT_CSM, EFFECT_DOF, EFFECT_DYN_REFLECT, EFFECT_DYRL_ANIME, EFFECT_FROXEL_VOL,
    EFFECT_GTAO_CONTACT, EFFECT_HIZ_CULL, EFFECT_PHYSICAL_SKY, EFFECT_RADIANCE_GI,
    EFFECT_SDF_SHADOWS, EFFECT_SHADOWS, EFFECT_SSGI_BENT, EFFECT_SSR,
    EFFECT_TAA, EFFECT_VOL_CLOUDS, EFFECT_VOLUMETRIC, EFFECT_WEATHER_PRECIP, FROXEL_X, FROXEL_Y, FROXEL_Z,
    GPU_CULL_IN_STRIDE_WORDS, GPU_CULL_MAX_INSTANCES, GPU_PARTICLE_MAX, GPU_PARTICLE_STRIDE_WORDS,
    MAX_IRRADIANCE_PROBES, MAX_RADIANCE_CASCADES, MAX_SDF_OCCLUDERS, MAX_SHADOW_SPHERES,
    MAX_SURFACE_DECALS, MIPMAPPED_FLAG,
    NO_DEPTH_WRITE_FLAG, NORM_ORM_FLAG, OBJECT_VELOCITY_FLAG, PBR_FLAG, PER_PIXEL_FLAG,
    RECORD_ANISO, RECORD_BLACKBODY_HEAT, RECORD_ERODE_THRESHOLD, RECORD_FLOATS, RECORD_METALLIC,
    RECORD_NORM_ORM_TEX, RECORD_REFRACT_STRENGTH, RECORD_ROUGHNESS, RECORD_SOFT_INV_DIST,
    RECORD_VEL_X, RECORD_VEL_Y, REFRACT_FLAG, SKY_LUT_H, SKY_LUT_W, SOFT_PARTICLE_FLAG,
    TEXTURED_FLAG, WATER_WAVE_FLAG,
)
from flint.render.gpu_raster import BACKEND_AMDGPU, BACKEND_VULKAN, GpuRaster

comptime CPU_CSM_RES: Int = 256
comptime CPU_CSM_CASCADE_WORDS: Int = CPU_CSM_RES * CPU_CSM_RES


@always_inline
def _parabolic_sin(phase: Float32) -> Float32:
    """Matches SPIR-V _parabolic_sin: C1 parabolic sin(2*pi*phase) accurate to 99.8%."""
    var f = phase - floor(phase)
    var u = f * Float32(2.0) - Float32(1.0)
    var abs_u = u if u >= Float32(0.0) else -u
    return Float32(4.0) * u * (Float32(1.0) - abs_u)


@always_inline
def _gyroid3d_unit_cpu(px: Float32, py: Float32, pz: Float32) -> Float32:
    """Matches SPIR-V _gyroid3d_unit bit-for-bit."""
    var sx = _parabolic_sin(px)
    var sy = _parabolic_sin(py)
    var sz = _parabolic_sin(pz)
    var cx = _parabolic_sin(px + Float32(0.25))
    var cy = _parabolic_sin(py + Float32(0.25))
    var cz = _parabolic_sin(pz + Float32(0.25))
    return (sx * cz + sy * cx + sz * cy) * Float32(0.16666667) + Float32(0.5)


struct HudMapLayer(Copyable, Movable):
    """Generic ARGB HUD overlay bitmap (used by pause map, radar, and minimap slots)."""
    var valid: Bool
    var width: Int
    var height: Int
    var generation: Int
    var pixels: List[UInt32]

    def __init__(out self):
        self.valid = False
        self.width = 0
        self.height = 0
        self.generation = 0
        self.pixels = List[UInt32]()

    def set_image(mut self, width: Int, height: Int, mut src_pixels: List[UInt32]):
        if width <= 0 or height <= 0 or len(src_pixels) < width * height:
            self.valid = False
            return
        self.width = width
        self.height = height
        self.pixels = src_pixels.copy()
        self.valid = True
        self.generation += 1


comptime PauseMap = HudMapLayer
comptime MiniMap = HudMapLayer


comptime TEXT_SPRITE_LIMIT: Int = 512


@fieldwise_init
struct TextSprite(Copyable, Movable):
    """One finished line of HUD text: premultiplied pixels, shadow and outline included."""
    var width: Int
    var height: Int
    var pixels: List[UInt32]


def dot(a: Vec3, b: Vec3) -> Float32:
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def clamp_byte(value: Float32) -> UInt8:
    return UInt8(max(0, min(255, Int(value))))


def rgb(red: UInt8, green: UInt8, blue: UInt8) -> UInt32:
    return UInt32(red) << 16 | UInt32(green) << 8 | UInt32(blue)


struct RenderVertex(Copyable, ImplicitlyCopyable, Movable):
    var position: Vec3
    var normal: Vec3
    var u: Float32
    var v: Float32
    var color: UInt32
    var light: Vec3
    var fog: Float32
    var emissive: Float32
    var occlusion: Float32
    var roughness: Float32
    var metallic: Float32
    var vel_x: Float32
    var vel_y: Float32
    var has_obj_vel: Bool

    def __init__(out self, position: Vec3 = Vec3(), normal: Vec3 = Vec3(0, 1, 0), u: Float32 = 0, v: Float32 = 0, color: UInt32 = 0xFFFFFFFF):
        self.position = position
        self.normal = normal
        self.u = u
        self.v = v
        self.color = color
        self.light = Vec3(0.90, 0.90, 0.90)
        self.fog = 0
        self.emissive = 0
        self.occlusion = -1
        self.roughness = 0.55
        self.metallic = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.has_obj_vel = False


struct CameraVertex(Copyable, ImplicitlyCopyable, Movable):
    var position: Vec3
    var normal: Vec3
    var u: Float32
    var v: Float32
    var color: UInt32
    var light: Vec3
    var fog: Float32
    var emissive: Float32
    var occlusion: Float32
    var roughness: Float32
    var metallic: Float32
    var vel_x: Float32
    var vel_y: Float32
    var has_obj_vel: Bool

    def __init__(out self, position: Vec3 = Vec3(), normal: Vec3 = Vec3(0, 1, 0), u: Float32 = 0, v: Float32 = 0, color: UInt32 = 0xFFFFFFFF):
        self.position = position
        self.normal = normal
        self.u = u
        self.v = v
        self.color = color
        self.light = Vec3(0.90, 0.90, 0.90)
        self.fog = 0
        self.emissive = 0
        self.occlusion = -1
        self.roughness = 0.55
        self.metallic = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.has_obj_vel = False

    def interpolate(self, other: Self, amount: Float32) -> Self:
        var inverse = 1.0 - amount
        var result = Self(
            self.position * inverse + other.position * amount,
            self.normal * inverse + other.normal * amount,
            self.u * inverse + other.u * amount,
            self.v * inverse + other.v * amount,
            self.color,
        )
        result.light = self.light * inverse + other.light * amount
        result.fog = self.fog * inverse + other.fog * amount
        result.emissive = self.emissive * inverse + other.emissive * amount
        result.occlusion = self.occlusion * inverse + other.occlusion * amount
        result.roughness = self.roughness * inverse + other.roughness * amount
        result.metallic = self.metallic * inverse + other.metallic * amount
        result.vel_x = self.vel_x * inverse + other.vel_x * amount
        result.vel_y = self.vel_y * inverse + other.vel_y * amount
        result.has_obj_vel = self.has_obj_vel and other.has_obj_vel
        return result^


struct ScreenVertex(Copyable, ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var inverse_z: Float32
    var normal: Vec3
    var u: Float32
    var v: Float32
    var color: UInt32
    var light: Vec3
    var fog: Float32
    var emissive: Float32
    var occlusion: Float32
    var roughness: Float32
    var metallic: Float32
    var vel_x: Float32
    var vel_y: Float32
    var has_obj_vel: Bool

    def __init__(out self):
        self.x = 0
        self.y = 0
        self.inverse_z = 1
        self.normal = Vec3(0, 1, 0)
        self.u = 0
        self.v = 0
        self.color = 0xFFFFFFFF
        self.light = Vec3(0.90, 0.90, 0.90)
        self.fog = 0
        self.emissive = 0
        self.occlusion = -1
        self.roughness = 0.55
        self.metallic = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.has_obj_vel = False


@fieldwise_init
struct IrradianceProbe(Copyable, ImplicitlyCopyable, Movable):
    var position: Vec3
    var inv_radius2: Float32
    var color: Vec3
    var sky_occlusion: Float32


@fieldwise_init
struct SurfaceDecal(Copyable, ImplicitlyCopyable, Movable):
    var center: Vec3
    var inv_hx: Float32
    var axis_right: Vec3
    var inv_hy: Float32
    var axis_forward: Vec3
    var inv_hz: Float32
    var albedo: Vec3
    var signed_opacity: Float32


@fieldwise_init
struct RadianceCascadeNode(Copyable, ImplicitlyCopyable, Movable):
    var position: Vec3
    var inv_radius2: Float32
    var radiance: Vec3
    var dir_bias_y: Float32


@fieldwise_init
struct SdfOccluder(Copyable, ImplicitlyCopyable, Movable):
    var p0: Vec3
    var radius: Float32
    var seg: Vec3
    var opacity: Float32


struct SoftwareFramebuffer(Copyable, Movable):
    var width: Int
    var height: Int
    var pixels: List[UInt32]
    var depth: List[Float32]
    var camera: Vec3
    var forward: Vec3
    var right: Vec3
    var up: Vec3
    var focal: Float32
    var fov_y: Float32  # degrees, kept across resizes (package 3.2: a setting)
    var fog_color: UInt32
    var light: Vec3
    var drawn_triangles: Int
    var culled_triangles: Int
    var frame_seconds: Float32
    var fps: Int
    var last_present: Float64
    var pixels_tested: Int
    var pixels_depth_rejected: Int
    var pixels_shaded: Int
    var bloom: List[Float32]
    var bloom_temporary: List[Float32]
    var bloom_bytes: List[UInt8]
    var bloom_width: Int
    var bloom_height: Int
    var bloom_enabled: Bool
    var bloom_pixels: Int
    var tone_lut: List[UInt8]
    var post_saturation_steps: Int
    var gpu: GpuRaster
    var bloom_strength: Int
    var gpu_effects: UInt32
    var gpu_smoothing: Bool
    var allow_gpu_effects: Bool
    var drop_shadows: Bool
    var drop_smoothing: Bool
    var hud_pixels: List[UInt32]
    var hud_ops: List[Int32]
    var hud_texts: List[String]
    var hud_rectangles: List[Int32]
    var hud_hash: UInt64
    var hud_recording: Bool
    var hud_drawing: Bool
    var hud_redraws: Int
    var text_sprites: List[TextSprite]
    var text_sprite_slots: Dict[String, Int]
    var text_canvas: List[UInt32]
    var ui_scale: Float32
    var ui_x: Int
    var ui_y: Int
    var text_scale: Float32     # player text-size setting, 1.0 to 1.5
    var blur_cache: List[UInt32]  # the finished menu backdrop; the world is frozen behind menus
    var blur_cache_valid: Bool
    var pause_map: PauseMap
    var backdrop_generation: Int
    var debug: DebugDraw
    # Package 3.2: triangles drawn while `occluder_pass` is on carry
    # DITHER_FLAG and thin out inside this frame's `dither` disc.
    var occluder_pass: Bool
    var dither: DitherView
    # Package 3.12: nine-slice images (ui/nine_slice.mojo): 0 slate panel, 1 focus frame.
    var nine_slices: List[NineSlice]
    # The fonts (package 3.12: with their TrueType sources) and the UI settings.
    var display_font: SdfFont   # Lilita One: titles, banners, big numbers
    var ui_font: SdfFont        # Atkinson Hyperlegible Next SemiBold: everything else
    var safe_margin: Float32    # package 3.12: safe-area margin per side (0 to 0.1)
    var button_pad: Bool        # package 3.12: button glyphs show the controller
    var button_sony: Bool       # ... with PlayStation shapes
    var button_keys: List[String]  # package 3.15: keycap labels of remapped keys by glyph action ("" = default)
    var button_keys_generation: Int  # bumped when they change, so the recorded HUD redraws
    var minimap: MiniMap        # package 3.18: the HUD's top-down map; last, to keep the older fields where they were
    var shadow_sphere_count: Int
    var min_text_em: Float32
    var cpu_to_key: Vec3
    var cpu_key_color: Vec3
    var cpu_env_sky: Vec3
    var cpu_env_ground: Vec3
    var cpu_env_horizon: Vec3
    var cpu_lights: List[LocalLight]
    var cpu_sphere_centers: List[Vec3]
    var cpu_sphere_radii_sq: List[Float32]
    var no_depth_write: Bool
    var additive_blend: Bool
    var vfx_soft_inv_dist: Float32
    var vfx_erode_threshold: Float32
    var vfx_blackbody_heat: Float32
    var vfx_refract_strength: Float32
    var water_wave_normals: Bool
    var prev_camera: Vec3
    var prev_right: Vec3
    var prev_up: Vec3
    var prev_forward: Vec3
    var prev_focal: Float32
    var temporal_frames: Int
    var history_parity: Int
    var taa_jitter_x: Float32
    var taa_jitter_y: Float32
    var prev_jitter_x: Float32
    var vol_density: Float32
    var vol_density_override: Float32
    var vol_height_falloff: Float32
    var ssr_strength: Float32
    var hdr_rgb: List[Float32]
    var gbuf_norm_rough: List[UInt32]
    var cpu_refract: List[UInt32]
    var cpu_history: List[UInt32]
    var cpu_csm_depth: List[Float32]
    var csm_right: Vec3
    var csm_up: Vec3
    var csm_c0_center: Vec3
    var csm_c0_inv_half: Float32
    var csm_c0_z_bias: Float32
    var csm_c0_z_scale: Float32
    var csm_c1_center: Vec3
    var csm_c1_inv_half: Float32
    var csm_c1_z_bias: Float32
    var csm_c1_z_scale: Float32
    var csm_has_casters: Bool
    var mesh_anisotropy: Float32
    var water_time: Float32
    var dof_focus_dist: Float32
    var dof_aperture: Float32
    var dof_max_coc: Float32
    var auto_exp_speed: Float32
    var auto_exp_min: Float32
    var auto_exp_max: Float32
    var auto_exp_target: Float32
    var adapted_exposure: Float32
    var cpu_velocity: List[UInt32]
    var vfx_norm_orm_ptr: Int
    var vfx_norm_orm_dim: Int
    var vfx_norm_orm_gpu_meta: UInt32
    var physical_sky_enabled: Bool
    var cam_altitude_km: Float32
    var planet_radius_km: Float32
    var atmos_thickness_km: Float32
    var rayleigh_density: Float32
    var mie_density: Float32
    var ozone_strength: Float32
    var aerial_strength: Float32
    var froxel_vol_enabled: Bool
    var froxel_max_dist: Float32
    var hiz_cull_enabled: Bool
    var cpu_probes: List[IrradianceProbe]
    var cpu_decals: List[SurfaceDecal]
    var cpu_froxel_hist: List[Float32]
    var cpu_froxel_valid: Bool
    var cpu_hiz_min_iz: List[Float32]
    var cpu_hiz_tiles_x: Int
    var cpu_hiz_tiles_y: Int
    var cpu_hiz_valid: Bool
    var vol_clouds_enabled: Bool
    var cloud_base_y: Float32
    var cloud_top_y: Float32
    var cloud_coverage: Float32
    var cloud_density: Float32
    var cloud_wind_x: Float32
    var cloud_wind_z: Float32
    var cloud_shadow_strength: Float32
    var cloud_scale: Float32
    var ssgi_enabled: Bool
    var ssgi_intensity: Float32
    var chromatic_refract_enabled: Bool
    var dispersion_strength: Float32
    var heat_haze_amp: Float32
    var radiance_gi_enabled: Bool
    var radiance_gi_intensity: Float32
    var cpu_radiance_nodes: List[RadianceCascadeNode]
    var dyrl_anime_enabled: Bool
    var dyrl_ink_strength: Float32
    var dyrl_cel_bands: Float32
    var dyrl_glint_strength: Float32
    var dyrl_halation: Float32
    var tsr_enabled: Bool
    var tsr_scale: Float32
    var tsr_sharpness: Float32
    var render_w: Int
    var render_h: Int
    var render_focal: Float32
    var tsr_subnative_active: Bool
    var sdf_shadows_enabled: Bool
    var sdf_softness_k: Float32
    var cpu_sdf_occluders: List[SdfOccluder]
    var weather_enabled: Bool
    var weather_kind: Int
    var weather_intensity: Float32
    var wetness: Float32
    var weather_wind_x: Float32
    var weather_wind_z: Float32
    var dyn_reflect_enabled: Bool
    var dyn_reflect_intensity: Float32
    var cpu_dyn_lobe_dir: Vec3
    var cpu_dyn_lobe_rgb: Vec3

    def __init__(out self, width: Int = 1280, height: Int = 720):
        self.width = width
        self.height = height
        self.pixels = List[UInt32](length=width * height, fill=0)
        self.depth = List[Float32](length=width * height, fill=0)
        self.camera = Vec3()
        self.forward = Vec3(0, 0, 1)
        self.right = Vec3(1, 0, 0)
        self.up = Vec3(0, 1, 0)
        self.focal = 1
        self.fov_y = 58
        self.fog_color = 0
        self.light = Vec3(-0.42, 0.82, -0.38).normalized()
        self.drawn_triangles = 0
        self.culled_triangles = 0
        self.frame_seconds = 1.0 / 60.0
        self.fps = 60
        self.last_present = 0
        self.pixels_tested = 0
        self.pixels_depth_rejected = 0
        self.pixels_shaded = 0
        self.bloom_width = 160
        self.bloom_height = 90
        self.bloom = List[Float32](length=self.bloom_width * self.bloom_height * 3, fill=0)
        self.bloom_temporary = List[Float32](length=self.bloom_width * self.bloom_height * 3, fill=0)
        self.bloom_bytes = List[UInt8](length=self.bloom_width * self.bloom_height * 3, fill=0)
        self.bloom_enabled = True
        self.bloom_pixels = 0
        self.tone_lut = List[UInt8](length=768, fill=0)
        self.post_saturation_steps = 0
        self.gpu = GpuRaster()
        self.bloom_strength = 5
        self.gpu_effects = 0
        self.gpu_smoothing = False
        self.allow_gpu_effects = True
        self.drop_shadows = False
        self.drop_smoothing = False
        self.occluder_pass = False
        self.dither = DitherView()
        self.nine_slices = List[NineSlice]()
        self.nine_slices.append(slate_panel())
        self.nine_slices.append(focus_frame())
        self.hud_pixels = List[UInt32]()
        self.hud_ops = List[Int32]()
        self.hud_texts = List[String]()
        self.hud_rectangles = List[Int32]()
        self.hud_hash = 0
        self.hud_recording = False
        self.hud_drawing = False
        self.hud_redraws = 0
        self.text_sprites = List[TextSprite]()
        self.text_sprite_slots = Dict[String, Int]()
        self.text_canvas = List[UInt32]()
        self.ui_scale = 1
        self.ui_x = 0
        self.ui_y = 0
        self.display_font = SdfFont()
        self.ui_font = SdfFont()
        self.text_scale = 1
        self.safe_margin = 0
        self.button_pad = False
        self.button_sony = False
        self.button_keys = List[String]()
        self.button_keys_generation = 0
        self.blur_cache = List[UInt32]()
        self.blur_cache_valid = False
        self.pause_map = PauseMap()
        self.backdrop_generation = 0
        self.debug = DebugDraw()
        self.minimap = MiniMap()
        self.shadow_sphere_count = 0
        self.min_text_em = 12.0
        self.cpu_to_key = Vec3(0.42, 0.82, 0.38).normalized()
        self.cpu_key_color = Vec3(0.75, 0.75, 0.75)
        self.cpu_env_sky = Vec3(0.32, 0.48, 0.72)
        self.cpu_env_ground = Vec3(0.10, 0.14, 0.22)
        self.cpu_env_horizon = Vec3(0.52, 0.64, 0.82)
        self.cpu_lights = List[LocalLight]()
        self.cpu_sphere_centers = List[Vec3]()
        self.cpu_sphere_radii_sq = List[Float32]()
        self.no_depth_write = False
        self.additive_blend = False
        self.vfx_soft_inv_dist = 0.0
        self.vfx_erode_threshold = 0.0
        self.vfx_blackbody_heat = -1.0
        self.vfx_refract_strength = 0.0
        self.water_wave_normals = False
        self.prev_camera = Vec3(0, 0, -5)
        self.prev_right = Vec3(1, 0, 0)
        self.prev_up = Vec3(0, 1, 0)
        self.prev_forward = Vec3(0, 0, 1)
        self.prev_focal = 1.0
        self.temporal_frames = 0
        self.history_parity = 0
        self.taa_jitter_x = 0.0
        self.taa_jitter_y = 0.0
        self.prev_jitter_x = 0.0
        self.vol_density = 0.0
        self.vol_density_override = -1.0
        self.vol_height_falloff = 0.020
        self.ssr_strength = 0.46
        self.hdr_rgb = List[Float32](length=width * height * 3, fill=0.0)
        self.gbuf_norm_rough = List[UInt32](length=width * height, fill=0)
        self.cpu_refract = List[UInt32](length=width * height, fill=0)
        self.cpu_history = List[UInt32](length=width * height, fill=0)
        self.cpu_csm_depth = List[Float32](length=CPU_CSM_CASCADE_WORDS * 2, fill=0.0)
        self.csm_right = Vec3(1, 0, 0)
        self.csm_up = Vec3(0, 0, 1)
        self.csm_c0_center = Vec3()
        self.csm_c0_inv_half = 1.0 / 44.0
        self.csm_c0_z_bias = 0.0
        self.csm_c0_z_scale = 0.5 / 220.0
        self.csm_c1_center = Vec3()
        self.csm_c1_inv_half = 1.0 / 260.0
        self.csm_c1_z_bias = 0.0
        self.csm_c1_z_scale = 0.5 / 520.0
        self.csm_has_casters = False
        self.mesh_anisotropy = 0.0
        self.water_time = 0.0
        self.dof_focus_dist = 12.0
        self.dof_aperture = 0.0
        self.dof_max_coc = 6.5
        self.auto_exp_speed = 0.16
        self.auto_exp_min = 0.88
        self.auto_exp_max = 1.12
        self.auto_exp_target = 0.0
        self.adapted_exposure = 1.0
        self.cpu_velocity = List[UInt32](length=width * height, fill=0)
        self.vfx_norm_orm_ptr = 0
        self.vfx_norm_orm_dim = 0
        self.vfx_norm_orm_gpu_meta = 0
        self.physical_sky_enabled = False
        self.cam_altitude_km = 0.05
        self.planet_radius_km = 6360.0
        self.atmos_thickness_km = 60.0
        self.rayleigh_density = 1.0
        self.mie_density = 1.0
        self.ozone_strength = 1.0
        self.aerial_strength = 0.32
        self.froxel_vol_enabled = False
        self.froxel_max_dist = 175.0
        self.hiz_cull_enabled = False
        self.cpu_probes = List[IrradianceProbe]()
        self.cpu_decals = List[SurfaceDecal]()
        self.cpu_froxel_hist = List[Float32]()
        self.cpu_froxel_valid = False
        self.cpu_hiz_min_iz = List[Float32]()
        self.cpu_hiz_tiles_x = 0
        self.cpu_hiz_tiles_y = 0
        self.cpu_hiz_valid = False
        self.vol_clouds_enabled = False
        self.cloud_base_y = 420.0
        self.cloud_top_y = 980.0
        self.cloud_coverage = 0.52
        self.cloud_density = 0.026
        self.cloud_wind_x = 12.0
        self.cloud_wind_z = -6.0
        self.cloud_shadow_strength = 0.42
        self.cloud_scale = 0.0022
        self.ssgi_enabled = True
        self.ssgi_intensity = 0.34
        self.chromatic_refract_enabled = True
        self.dispersion_strength = 0.22
        self.heat_haze_amp = 1.35
        self.radiance_gi_enabled = True
        self.radiance_gi_intensity = 0.28
        self.cpu_radiance_nodes = List[RadianceCascadeNode]()
        self.dyrl_anime_enabled = False
        self.dyrl_ink_strength = 0.85
        self.dyrl_cel_bands = 3.0
        self.dyrl_glint_strength = 0.65
        self.dyrl_halation = 0.28
        self.tsr_enabled = False
        self.tsr_scale = 0.75
        self.tsr_sharpness = 0.42
        self.render_w = width
        self.render_h = height
        self.render_focal = 1.0
        self.tsr_subnative_active = False
        self.sdf_shadows_enabled = True
        self.sdf_softness_k = 8.0
        self.cpu_sdf_occluders = List[SdfOccluder]()
        self.weather_enabled = False
        self.weather_kind = 1
        self.weather_intensity = 0.0
        self.wetness = 0.0
        self.weather_wind_x = 8.0
        self.weather_wind_z = -4.0
        self.dyn_reflect_enabled = True
        self.dyn_reflect_intensity = 0.44
        self.cpu_dyn_lobe_dir = Vec3(0.0, 1.0, 0.0)
        self.cpu_dyn_lobe_rgb = Vec3(0.0, 0.0, 0.0)
        self.update_ui_transform()
        for index in range(256):
            self.tone_lut[index] = UInt8(index)
            self.tone_lut[index + 256] = UInt8(index)
            self.tone_lut[index + 512] = UInt8(index)
        self.set_camera(Vec3(0, 0, -5), Vec3(), 58)

    def load_fonts(
        mut self,
        ui_path: String,
        display_path: String = "",
        ui_weight: Float32 = 0,
        ui_cache_dsf: String = "",
        display_cache_dsf: String = "",
    ) -> Bool:
        """Load UI and optional Display SDF fonts from either .ttf or .dsf files."""
        var ok_ui = False
        if ui_path.endswith(".dsf"):
            ok_ui = self.ui_font.load(ui_path)
        elif ui_path.byte_length() > 0:
            ok_ui = self.ui_font.load_ttf(ui_path, 42, 6, ui_weight, ui_cache_dsf)

        var disp = display_path if display_path.byte_length() > 0 else ui_path
        var disp_cache = display_cache_dsf if display_path.byte_length() > 0 else ui_cache_dsf
        var ok_disp = False
        if disp.endswith(".dsf"):
            ok_disp = self.display_font.load(disp)
        elif disp.byte_length() > 0:
            ok_disp = self.display_font.load_ttf(disp, 48, 6, ui_weight, disp_cache)

        self.text_sprites.clear()
        self.text_sprite_slots.clear()
        self.hud_hash = 0
        return ok_ui and ok_disp

    def em_size(self, size: Int) -> Float32:
        return max(Float32(size), self.min_text_em)

    def ui_frame(self, player: Int = 0) -> UiFrame:
        """The visible canvas and its safe area (package 3.12, ui/layout.mojo)."""
        return ui_frame(self.width, self.height, self.ui_x, self.ui_y, self.ui_scale, self.safe_margin, player=player)

    def set_safe_margin(mut self, margin: Float32):
        self.safe_margin = max(Float32(0), min(MAX_SAFE_MARGIN, margin))

    def update_ui_transform(mut self):
        """Fit the 1280x720 UI canvas to the screen at any scale (1.5x at
        1080p), centred. It used to scale by whole numbers only, so 1080p
        drew the UI at 1x and the smallest text was 7 px tall."""
        self.ui_scale = max(Float32(0.25), min(Float32(self.width) / 1280.0, Float32(self.height) / 720.0))
        self.ui_x = max(0, Int((Float32(self.width) - 1280.0 * self.ui_scale) * 0.5))
        self.ui_y = max(0, Int((Float32(self.height) - 720.0 * self.ui_scale) * 0.5))

    def blur_scene(mut self, amount: Float32) -> Bool:
        """The menu backdrop. On the CPU path the frame in `pixels` is
        blurred in place. On the GPU path the frame that opens the menu also
        composes a copy without the HUD layer; the next frame reads that copy
        back once, blurs it on the CPU, and lays it into the HUD layer under
        the menu (the world is frozen behind menus, so once is enough).
        Returns False when no backdrop is ready; the caller dims instead."""
        if not self.gpu.active:
            self.blur_pixels(amount)
            return True
        if amount <= 0:
            self.blur_cache_valid = False
            self.gpu.forget_backdrop()
            return True
        if self.hud_recording:
            if not self.blur_cache_valid:
                if not self.gpu.backdrop_ready:
                    self.gpu.backdrop_wanted = True
                    return False
                var read = self.gpu.read_back_backdrop(self.pixels)
                self.stop_recording_if_gpu_lost()
                if not read:
                    return False
                self.blur_pixels(1)
                self.gpu.forget_backdrop()
            if not self.blur_cache_valid:
                return False
            self.record_hud(5, 0, 0, 0, 0, self.backdrop_generation, 0, 0, 0, 255)
        return True

    def draw_backdrop(mut self):
        """Replay of op 5: the cached backdrop, opaque, across the HUD layer."""
        if not self.blur_cache_valid or len(self.blur_cache) != len(self.hud_pixels):
            return
        self.add_hud_rectangle(0, 0, self.width, self.height)
        for index in range(len(self.blur_cache)):
            self.hud_pixels[index] = self.blur_cache[index] | UInt32(0xFF000000)

    def blur_pixels(mut self, amount: Float32):
        """Blur the finished 3D frame for a menu backdrop: average down by 4,
        blur that twice with a 5-tap box, then scale back up bilinearly,
        darkened and pulled toward a cool blue. `amount` 0..1 fades it in."""
        if amount <= 0:
            self.blur_cache_valid = False
            return
        if amount >= 1 and self.blur_cache_valid and len(self.blur_cache) == len(self.pixels):
            _ = external_call["memcpy", Int](Int(self.pixels.unsafe_ptr()), Int(self.blur_cache.unsafe_ptr()), len(self.pixels) * 4)
            return
        var factor = 4
        var small_width = max(1, self.width // factor)
        var small_height = max(1, self.height // factor)
        var small = List[Float32](length=small_width * small_height * 3, fill=0)
        var weight = 1.0 / Float32(factor * factor)
        for sy in range(small_height):
            for sx in range(small_width):
                var r: Float32 = 0
                var g: Float32 = 0
                var b: Float32 = 0
                for dy in range(factor):
                    var row = (sy * factor + dy) * self.width + sx * factor
                    for dx in range(factor):
                        var c = self.pixels[row + dx]
                        r += Float32((c >> 16) & 255)
                        g += Float32((c >> 8) & 255)
                        b += Float32(c & 255)
                var o = (sy * small_width + sx) * 3
                small[o] = r * weight
                small[o + 1] = g * weight
                small[o + 2] = b * weight
        var temp = List[Float32](length=len(small), fill=0)
        for _ in range(2):
            box_blur_pass(small, temp, small_width, small_height, True)
            box_blur_pass(temp, small, small_width, small_height, False)
        var keep = 1.0 - amount
        var dim = 1.0 - 0.38 * amount
        var tint_r = 14.0 * amount
        var tint_g = 24.0 * amount
        var tint_b = 38.0 * amount
        for y in range(self.height):
            var fy = (Float32(y) + 0.5) / Float32(factor) - 0.5
            var y0 = max(0, min(small_height - 1, Int(floor(fy))))
            var y1 = min(small_height - 1, y0 + 1)
            var ty = max(Float32(0), min(Float32(1), fy - Float32(y0)))
            for x in range(self.width):
                var fx = (Float32(x) + 0.5) / Float32(factor) - 0.5
                var x0 = max(0, min(small_width - 1, Int(floor(fx))))
                var x1 = min(small_width - 1, x0 + 1)
                var tx = max(Float32(0), min(Float32(1), fx - Float32(x0)))
                var index = y * self.width + x
                var c = self.pixels[index]
                var result = UInt32(0)
                for channel in range(3):
                    var a = small[(y0 * small_width + x0) * 3 + channel]
                    var b = small[(y0 * small_width + x1) * 3 + channel]
                    var cc = small[(y1 * small_width + x0) * 3 + channel]
                    var d = small[(y1 * small_width + x1) * 3 + channel]
                    var blurred = (a + (b - a) * tx) + ((cc + (d - cc) * tx) - (a + (b - a) * tx)) * ty
                    var shift = UInt32(16 - channel * 8)
                    var original = Float32((c >> shift) & 255)
                    var tint = tint_r if channel == 0 else (tint_g if channel == 1 else tint_b)
                    var value = (original * keep + blurred * amount) * dim + tint
                    result |= UInt32(max(0, min(255, Int(value)))) << shift
                self.pixels[index] = (c & 0xFF000000) | result
        if amount >= 1:
            self.blur_cache = self.pixels.copy()
            self.blur_cache_valid = True
            self.backdrop_generation += 1
        else:
            self.blur_cache_valid = False

    def draw_pause_map(mut self, ui_left: Int, ui_top: Int):
        """Blit the pause map with its top-left at a canvas point."""
        if self.hud_recording:
            self.record_hud(4, ui_left, ui_top, 0, 0, self.pause_map.generation, 0, 0, 0, 255)
            return
        if not self.pause_map.valid:
            return
        var x0 = self.ux(ui_left)
        var y0 = self.uy(ui_top)
        if self.hud_drawing:
            self.add_hud_rectangle(x0, y0, self.pause_map.width, self.pause_map.height)
        for y in range(self.pause_map.height):
            var sy = y0 + y
            if sy < 0 or sy >= self.height:
                continue
            for x in range(self.pause_map.width):
                var sx = x0 + x
                if sx < 0 or sx >= self.width:
                    continue
                var c = self.pause_map.pixels[y * self.pause_map.width + x]
                if (c >> 24) == 0:
                    continue
                self.blend_pixel(sy * self.width + sx, c & 0xFFFFFF, 255)

    def draw_minimap(mut self, ui_left: Int, ui_top: Int, ui_width: Int, ui_height: Int, alpha: UInt8 = 255):
        """Package 3.18: the minimap scaled (nearest texel) into a canvas
        rectangle. Recorded as HUD op 8 with the map's generation, so the
        HUD redraws when the picture changes. Columns with no floor leave
        what is under them; the rest are laid over with `alpha`, the
        premultiplied over inlined on the raw pointers (blend_pixel per
        pixel cost 3.12 1.5 ms of p99 at 3440x1440)."""
        if self.hud_recording:
            self.record_hud(8, ui_left, ui_top, ui_width, ui_height, self.minimap.generation, 0, 0, 0, alpha)
            return
        if not self.minimap.valid or alpha == 0 or self.minimap.width <= 0 or self.minimap.height <= 0:
            return
        var x0 = self.ux(ui_left)
        var y0 = self.uy(ui_top)
        var w = self.ux(ui_left + ui_width) - x0
        var h = self.uy(ui_top + ui_height) - y0
        if w <= 0 or h <= 0:
            return
        if self.hud_drawing:
            self.add_hud_rectangle(x0, y0, w, h)
        var left = max(0, x0)
        var right = min(self.width, x0 + w)
        var top = max(0, y0)
        var bottom = min(self.height, y0 + h)
        if right <= left or bottom <= top:
            return
        # Texel column of each screen column, once.
        var columns = List[Int](length=right - left, fill=0)
        for px in range(left, right):
            columns[px - left] = min(self.minimap.width - 1, (px - x0) * self.minimap.width // w)
        var amount = UInt32(alpha)
        var inverse = 255 - amount
        var texels = self.minimap.pixels.unsafe_ptr()
        var hud = self.hud_drawing
        var target = self.hud_pixels.unsafe_ptr() if hud else self.pixels.unsafe_ptr()
        for py in range(top, bottom):
            var tz = min(self.minimap.height - 1, (py - y0) * self.minimap.height // h)
            var source_row = tz * self.minimap.width
            var row = py * self.width
            for px in range(left, right):
                var c = texels[unsafe_offset=source_row + columns[px - left]]
                if (c >> 24) == 0:
                    continue
                var index = row + px
                var under = target[unsafe_offset=index]
                var r = (c >> 16) & 255
                var g = (c >> 8) & 255
                var b = c & 255
                if hud:
                    # Premultiplied over, as blend_hud_pixel does it.
                    var a = amount + ((under >> 24) * inverse) // 255
                    r = (r * amount + ((under >> 16) & 255) * inverse) // 255
                    g = (g * amount + ((under >> 8) & 255) * inverse) // 255
                    b = (b * amount + (under & 255) * inverse) // 255
                    target[unsafe_offset=index] = a << 24 | r << 16 | g << 8 | b
                else:
                    r = (r * amount + ((under >> 16) & 255) * inverse) // 255
                    g = (g * amount + ((under >> 8) & 255) * inverse) // 255
                    b = (b * amount + (under & 255) * inverse) // 255
                    target[unsafe_offset=index] = r << 16 | g << 8 | b

    def ux(self, x: Int) -> Int:
        """Screen x of a UI-canvas x."""
        return self.ui_x + Int(floor(Float32(x) * self.ui_scale + 0.5))

    def uy(self, y: Int) -> Int:
        return self.ui_y + Int(floor(Float32(y) * self.ui_scale + 0.5))

    def set_text_scale(mut self, scale: Float32):
        var clamped = max(Float32(1), min(Float32(1.5), scale))
        if clamped != self.text_scale:
            self.text_scale = clamped
            self.hud_hash = 0  # the recorded draw calls are the same; force a redraw

    def resize(mut self, width: Int, height: Int):
        if width <= 0 or height <= 0 or (width == self.width and height == self.height and (not self.gpu.active or (self.gpu.width == width and self.gpu.height == height))):
            return
        self.width = width
        self.height = height
        self.render_w = width
        self.render_h = height
        self.tsr_subnative_active = False
        self.pixels = List[UInt32](length=width * height, fill=0)
        self.depth = List[Float32](length=width * height, fill=0)
        self.hdr_rgb = List[Float32](length=width * height * 3, fill=0.0)
        self.gbuf_norm_rough = List[UInt32](length=width * height, fill=0)
        self.cpu_refract = List[UInt32](length=width * height, fill=0)
        self.cpu_history = List[UInt32](length=width * height, fill=0)
        self.cpu_velocity = List[UInt32](length=width * height, fill=0)
        self.cpu_froxel_valid = False
        self.cpu_hiz_valid = False
        self.temporal_frames = 0
        self.history_parity = 0
        self.update_ui_transform()
        self.hud_pixels = List[UInt32]()
        self.hud_ops.clear()
        self.hud_texts.clear()
        self.hud_rectangles.clear()
        self.hud_hash = 0
        self.hud_recording = False
        self.text_sprites.clear()
        self.text_sprite_slots.clear()
        self.text_canvas = List[UInt32]()
        self.debug.written.clear()  # those pixels went with the old images
        # Both were made for the old size, and the map's pixels follow the UI scale.
        self.blur_cache_valid = False
        self.pause_map.valid = False
        if self.gpu.active:
            if not self.gpu.resize(width, height):
                self.gpu.active = False
        var position = self.camera
        var target = self.camera + self.forward
        self.set_camera(position, target, self.fov_y)

    def clear(mut self, red: UInt8, green: UInt8, blue: UInt8):
        var color = rgb(red, green, blue)
        if self.gpu.active:
            self.hud_ops.clear()
            self.hud_texts.clear()
            self.hud_recording = True
        else:
            self.hud_recording = False
            var fr = Float32(red)
            var fg = Float32(green)
            var fb = Float32(blue)
            for index in range(len(self.pixels)):
                self.pixels[index] = color
                self.depth[index] = 1000000000.0
                self.hdr_rgb[index * 3] = fr
                self.hdr_rgb[index * 3 + 1] = fg
                self.hdr_rgb[index * 3 + 2] = fb
                self.gbuf_norm_rough[index] = UInt32(0x00FF8080)
                self.cpu_refract[index] = 0
                self.cpu_velocity[index] = 0
            self.clear_cpu_csm()
        self.fog_color = color
        self.drawn_triangles = 0
        self.culled_triangles = 0
        self.pixels_tested = 0
        self.pixels_depth_rejected = 0
        self.pixels_shaded = 0
        self.bloom_pixels = 0
        self.vfx_soft_inv_dist = 0.0
        self.vfx_erode_threshold = 0.0
        self.vfx_blackbody_heat = -1.0
        self.vfx_refract_strength = 0.0
        self.water_wave_normals = False
        self.mesh_anisotropy = 0.0
        self.vfx_norm_orm_ptr = 0
        self.vfx_norm_orm_dim = 0
        self.vfx_norm_orm_gpu_meta = 0
        var dyn_cpu = self._compute_dynamic_reflection_lobe()
        self.cpu_dyn_lobe_dir = dyn_cpu[0]
        self.cpu_dyn_lobe_rgb = dyn_cpu[1]
        if self.bloom_enabled:
            for index in range(len(self.bloom)):
                self.bloom[index] = 0

    def configure_post(mut self, profile: LightingProfile, quality: Int, reduced_effects: Bool):
        var desired_width = 0 if quality <= 0 or reduced_effects else (160 if quality == 1 else (240 if quality == 2 else 384))
        var desired_height = 0 if desired_width == 0 else (90 if quality == 1 else (135 if quality == 2 else 216))
        if desired_width > 0:
            desired_width = min(desired_width, self.width)
            desired_height = min(desired_height, self.height)
        self.bloom_enabled = desired_width > 0
        self.bloom_strength = 5 if quality == 1 else 4
        # Visual extras never reach gameplay, and low quality or reduced effects drop them.
        var extras = self.allow_gpu_effects and quality >= 1 and not reduced_effects
        self.gpu_effects = (
            EFFECT_BILINEAR | EFFECT_SHADOWS | EFFECT_CSM | EFFECT_SSR | EFFECT_VOLUMETRIC | EFFECT_TAA | EFFECT_GTAO_CONTACT | EFFECT_AUTO_EXPOSURE
        ) if extras else UInt32(0)
        if extras and self.dof_aperture > Float32(0.0001):
            self.gpu_effects |= EFFECT_DOF
        if self.drop_shadows:
            self.gpu_effects &= ~(EFFECT_SHADOWS | EFFECT_CSM)
        if quality < 2 and self.vol_density_override < 0.0:
            self.gpu_effects &= ~EFFECT_VOLUMETRIC
        var fog_luma = profile.fog_color.x * 0.26 + profile.fog_color.y * 0.58 + profile.fog_color.z * 0.16
        if self.vol_density_override >= 0.0:
            self.vol_density = self.vol_density_override
        elif (self.gpu_effects & EFFECT_VOLUMETRIC) != 0:
            self.vol_density = Float32(0.16) if (fog_luma >= Float32(40.0) or profile.height_fog > Float32(0.001)) else Float32(0.032)
        else:
            self.vol_density = 0.0
        if self.vol_density <= Float32(0.0005):
            self.gpu_effects &= ~EFFECT_VOLUMETRIC
        if extras and self.physical_sky_enabled:
            self.gpu_effects |= EFFECT_PHYSICAL_SKY
        if extras and self.froxel_vol_enabled and (self.gpu_effects & EFFECT_VOLUMETRIC) != 0:
            self.gpu_effects |= EFFECT_FROXEL_VOL
        if extras and self.hiz_cull_enabled:
            self.gpu_effects |= EFFECT_HIZ_CULL
        if extras and self.vol_clouds_enabled:
            self.gpu_effects |= EFFECT_VOL_CLOUDS
        if extras and quality >= 2 and self.ssgi_enabled:
            self.gpu_effects |= EFFECT_SSGI_BENT
        if extras and self.chromatic_refract_enabled:
            self.gpu_effects |= EFFECT_CHROMATIC_REFRACT
        if extras and quality >= 2 and self.radiance_gi_enabled:
            self.gpu_effects |= EFFECT_RADIANCE_GI
        if extras and self.dyrl_anime_enabled:
            self.gpu_effects |= EFFECT_DYRL_ANIME
        if extras and self.sdf_shadows_enabled and len(self.cpu_sdf_occluders) > 0:
            self.gpu_effects |= EFFECT_SDF_SHADOWS
        if extras and self.weather_enabled and (self.weather_intensity > Float32(0.01) or self.wetness > Float32(0.01)):
            self.gpu_effects |= EFFECT_WEATHER_PRECIP
        if extras and self.dyn_reflect_enabled and self.dyn_reflect_intensity > Float32(0.005):
            self.gpu_effects |= EFFECT_DYN_REFLECT
        if self.tsr_enabled and self.tsr_scale < Float32(0.97) and self.width >= 160 and self.height >= 90:
            self.render_w = max(64, min(self.width, (Int(Float32(self.width) * self.tsr_scale + Float32(0.5)) // 8) * 8))
            self.render_h = max(64, min(self.height, (Int(Float32(self.height) * self.tsr_scale + Float32(0.5)) // 4) * 4))
            self.render_focal = self.focal * (Float32(self.render_h) / Float32(max(1, self.height)))
            self.tsr_subnative_active = self.render_w < self.width or self.render_h < self.height
        else:
            self.render_w = self.width
            self.render_h = self.height
            self.render_focal = self.focal
            self.tsr_subnative_active = False
        if self.gpu.active:
            self.gpu.render_width = self.render_w
            self.gpu.render_height = self.render_h
        self.vol_height_falloff = max(Float32(0.016), profile.height_fog)
        self.ssr_strength = Float32(0.46) if extras else Float32(0.0)
        self.gpu_smoothing = extras and not self.drop_smoothing
        self.post_saturation_steps = Int((profile.saturation - 1.0) * 64.0) if quality > 0 else 0
        if desired_width > 0 and (desired_width != self.bloom_width or desired_height != self.bloom_height):
            self.bloom_width = desired_width
            self.bloom_height = desired_height
            self.bloom = List[Float32](length=desired_width * desired_height * 3, fill=0)
            self.bloom_temporary = List[Float32](length=desired_width * desired_height * 3, fill=0)
            self.bloom_bytes = List[UInt8](length=desired_width * desired_height * 3, fill=0)
        if not self.bloom_enabled:
            self.bloom_pixels = 0
        if profile.filmic <= 0 and profile.cdl_slope.x == 1 and profile.cdl_slope.y == 1 and profile.cdl_slope.z == 1 and profile.cdl_offset.length_squared() == 0 and profile.cdl_power.x == 1 and profile.cdl_power.y == 1 and profile.cdl_power.z == 1:
            # Ungraded profiles keep the original highlight shoulder exactly.
            for index in range(256):
                var value = Float32(index) / 255.0 * profile.exposure
                # Preserve the midtones and compress only the highlight shoulder.
                if value > 0.72:
                    value = 0.72 + (value - 0.72) / (1.0 + (value - 0.72) * 0.90)
                self.tone_lut[index] = clamp_byte(value * 255.0 * profile.grade_gain.x)
                self.tone_lut[index + 256] = clamp_byte(value * 255.0 * profile.grade_gain.y)
                self.tone_lut[index + 512] = clamp_byte(value * 255.0 * profile.grade_gain.z)
        else:
            build_tone_table(profile, self.tone_lut)

    def set_camera(mut self, position: Vec3, target: Vec3, fovy: Float32 = 58):
        self.camera = position
        self.forward = (target - position).normalized()
        if self.forward.length_squared() < 0.5:
            self.forward = Vec3(0, 0, 1)
        self.right = cross(Vec3(0, 1, 0), self.forward).normalized()
        if self.right.length_squared() < 0.5:
            self.right = Vec3(1, 0, 0)
        self.up = cross(self.forward, self.right).normalized()
        self.fov_y = fovy
        self.focal = Float32(self.height) * 0.5 / tan(fovy * 0.01745329252 * 0.5)
        self.render_focal = self.focal * (Float32(self.render_h) / Float32(max(1, self.height)))
        self.dither = DitherView()

    def set_camera_roll(mut self, roll_rad: Float32):
        """Bank the camera's right/up basis around `forward` by `roll_rad` radians."""
        if abs(roll_rad) < 0.0001:
            return
        var c = cos(roll_rad)
        var s = sin(roll_rad)
        var r0 = self.right
        var u0 = self.up
        self.right = (r0 * c + u0 * s).normalized()
        self.up = (u0 * c - r0 * s).normalized()

    def set_dither(mut self, target: Vec3, radius: Float32, margin: Float32, keep: Float32):
        """Thin occluders out around `target` this frame (render/dither.mojo);
        call after set_camera, which turns the disc off."""
        self.dither = dither_view(self.camera, self.right, self.up, self.forward, self.focal, self.width, self.height, target, radius, margin, keep)

    def upload_dither(self):
        self.gpu.write_float(CONSTANT_DITHER, self.dither.center_x)
        self.gpu.write_float(CONSTANT_DITHER + 4, self.dither.center_y)
        self.gpu.write_float(CONSTANT_DITHER + 8, self.dither.scale)
        self.gpu.write_float(CONSTANT_DITHER + 12, self.dither.offset)
        self.gpu.write_float(CONSTANT_DITHER + 16, self.dither.near_inverse_z)

    def sphere_visible(self, center: Vec3, radius: Float32) -> Bool:
        var relative = center - self.camera
        var depth = dot(relative, self.forward)
        if depth + radius < 0.06:
            return False
        var horizontal = dot(relative, self.right)
        var vertical = dot(relative, self.up)
        var horizontal_slope = Float32(self.width) * 0.5 / self.focal
        var vertical_slope = Float32(self.height) * 0.5 / self.focal
        var horizontal_margin = radius * sqrt(1.0 + horizontal_slope * horizontal_slope)
        var vertical_margin = radius * sqrt(1.0 + vertical_slope * vertical_slope)
        if horizontal - horizontal_slope * depth > horizontal_margin:
            return False
        if -horizontal - horizontal_slope * depth > horizontal_margin:
            return False
        if vertical - vertical_slope * depth > vertical_margin:
            return False
        if -vertical - vertical_slope * depth > vertical_margin:
            return False
        return True

    def camera_vertex(self, vertex: RenderVertex) -> CameraVertex:
        var relative = vertex.position - self.camera
        var result = CameraVertex(
            Vec3(dot(relative, self.right), dot(relative, self.up), dot(relative, self.forward)),
            vertex.normal,
            vertex.u,
            vertex.v,
            vertex.color,
        )
        result.light = vertex.light
        result.fog = vertex.fog
        result.emissive = vertex.emissive
        result.occlusion = vertex.occlusion
        result.roughness = vertex.roughness
        result.metallic = vertex.metallic
        result.vel_x = vertex.vel_x
        result.vel_y = vertex.vel_y
        result.has_obj_vel = vertex.has_obj_vel
        return result^

    def project(self, vertex: CameraVertex) -> ScreenVertex:
        var result = ScreenVertex()
        var inverse_z = 1.0 / max(Float32(0.0001), vertex.position.z)
        var pw = self.render_w if self.tsr_subnative_active else self.width
        var ph = self.render_h if self.tsr_subnative_active else self.height
        var pf = self.render_focal if self.tsr_subnative_active else self.focal
        result.x = Float32(pw) * 0.5 + vertex.position.x * pf * inverse_z
        result.y = Float32(ph) * 0.5 - vertex.position.y * pf * inverse_z
        result.inverse_z = inverse_z
        result.normal = vertex.normal
        result.u = vertex.u
        result.v = vertex.v
        result.color = vertex.color
        result.light = vertex.light
        result.fog = vertex.fog
        result.emissive = vertex.emissive
        result.occlusion = vertex.occlusion
        result.roughness = vertex.roughness
        result.metallic = vertex.metallic
        result.vel_x = vertex.vel_x
        result.vel_y = vertex.vel_y
        result.has_obj_vel = vertex.has_obj_vel
        return result^

    @staticmethod
    def _lerp_packed_argb(a: UInt32, b: UInt32, w256: Int) -> UInt32:
        var iw = 256 - w256
        var lo = (((a & 0x00FF00FF) * UInt32(iw) + (b & 0x00FF00FF) * UInt32(w256)) >> 8) & 0x00FF00FF
        var hi = ((((a >> 8) & 0x00FF00FF) * UInt32(iw) + ((b >> 8) & 0x00FF00FF) * UInt32(w256)) >> 8) & 0x00FF00FF
        return (hi << 8) | lo

    def sample_texture_bilinear(self, mut texture: List[UInt32], texture_width: Int, texture_height: Int, u: Float32, v: Float32) -> UInt32:
        if texture_width <= 0 or texture_height <= 0 or len(texture) < texture_width * texture_height:
            return 0xFFFFFFFF
        var wu = u - floor(u)
        var wv = v - floor(v)
        var fx = wu * Float32(texture_width) - 0.5
        var fy = wv * Float32(texture_height) - 0.5
        var ix = Int(floor(fx))
        var iy = Int(floor(fy))
        var wx = max(0, min(256, Int((fx - Float32(ix)) * 256.0)))
        var wy = max(0, min(256, Int((fy - Float32(iy)) * 256.0)))
        var x0 = ix % texture_width
        if x0 < 0:
            x0 += texture_width
        var x1 = (x0 + 1) % texture_width
        var y0 = iy % texture_height
        if y0 < 0:
            y0 += texture_height
        var y1 = (y0 + 1) % texture_height
        var ptr = texture.unsafe_ptr()
        var c00 = ptr[unsafe_offset=y0 * texture_width + x0]
        var c10 = ptr[unsafe_offset=y0 * texture_width + x1]
        var c01 = ptr[unsafe_offset=y1 * texture_width + x0]
        var c11 = ptr[unsafe_offset=y1 * texture_width + x1]
        return Self._lerp_packed_argb(
            Self._lerp_packed_argb(c00, c10, wx),
            Self._lerp_packed_argb(c01, c11, wx),
            wy,
        )

    def sample_texture_filtered(
        self,
        mut texture: List[UInt32],
        texture_width: Int,
        texture_height: Int,
        u: Float32,
        v: Float32,
        du_dx: Float32,
        dv_dx: Float32,
        du_dy: Float32,
        dv_dy: Float32,
    ) -> UInt32:
        if (self.gpu_effects & EFFECT_BILINEAR) == 0:
            return self.sample_texture(texture, texture_width, texture_height, u, v)
        var wf = Float32(texture_width)
        var hf = Float32(texture_height)
        var lx2 = (du_dx * wf) * (du_dx * wf) + (dv_dx * hf) * (dv_dx * hf)
        var ly2 = (du_dy * wf) * (du_dy * wf) + (dv_dy * hf) * (dv_dy * hf)
        var x_maj = lx2 >= ly2
        var maj2 = lx2 if x_maj else ly2
        var min2 = max(Float32(0.000001), ly2 if x_maj else lx2)
        var m_du = du_dx if x_maj else du_dy
        var m_dv = dv_dx if x_maj else dv_dy
        if maj2 > min2 * Float32(3.2) and maj2 > Float32(1.2):
            var s0 = self.sample_texture_bilinear(texture, texture_width, texture_height, u - m_du * 0.375, v - m_dv * 0.375)
            var s1 = self.sample_texture_bilinear(texture, texture_width, texture_height, u - m_du * 0.125, v - m_dv * 0.125)
            var s2 = self.sample_texture_bilinear(texture, texture_width, texture_height, u + m_du * 0.125, v + m_dv * 0.125)
            var s3 = self.sample_texture_bilinear(texture, texture_width, texture_height, u + m_du * 0.375, v + m_dv * 0.375)
            return Self._lerp_packed_argb(Self._lerp_packed_argb(s0, s1, 128), Self._lerp_packed_argb(s2, s3, 128), 128)
        elif maj2 > min2 * Float32(1.5) and maj2 > Float32(1.1):
            var b0 = self.sample_texture_bilinear(texture, texture_width, texture_height, u - m_du * 0.25, v - m_dv * 0.25)
            var b1 = self.sample_texture_bilinear(texture, texture_width, texture_height, u + m_du * 0.25, v + m_dv * 0.25)
            return Self._lerp_packed_argb(b0, b1, 128)
        return self.sample_texture_bilinear(texture, texture_width, texture_height, u, v)

    def sample_texture(self, mut texture: List[UInt32], texture_width: Int, texture_height: Int, u: Float32, v: Float32) -> UInt32:
        if texture_width <= 0 or texture_height <= 0 or len(texture) < texture_width * texture_height:
            return 0xFFFFFFFF
        var wrapped_u = u - floor(u)
        var wrapped_v = v - floor(v)
        var x = max(0, min(texture_width - 1, Int(wrapped_u * Float32(texture_width))))
        var y = max(0, min(texture_height - 1, Int(wrapped_v * Float32(texture_height))))
        return texture[y * texture_width + x]

    def blend_hud_pixel(mut self, index: Int, color: UInt32, alpha: UInt8):
        """Premultiplied over, so the GPU can later place the layer on any scene."""
        var previous = self.hud_pixels[index]
        var inverse = 255 - Int(alpha)
        var a = Int(alpha) + (Int(previous >> 24) * inverse) // 255
        var red = (Int((color >> 16) & 255) * Int(alpha) + Int((previous >> 16) & 255) * inverse) // 255
        var green = (Int((color >> 8) & 255) * Int(alpha) + Int((previous >> 8) & 255) * inverse) // 255
        var blue = (Int(color & 255) * Int(alpha) + Int(previous & 255) * inverse) // 255
        self.hud_pixels[index] = UInt32(a) << 24 | UInt32(red) << 16 | UInt32(green) << 8 | UInt32(blue)

    def blend_pixel(mut self, index: Int, color: UInt32, alpha: UInt8):
        if self.hud_drawing:
            self.blend_hud_pixel(index, color, alpha)
            return
        if alpha == 255:
            self.pixels[index] = color & 0x00FFFFFF
            return
        if alpha == 0:
            return
        var previous = self.pixels[index]
        var amount = Float32(alpha) / 255.0
        var inverse = 1.0 - amount
        var red = Float32((color >> 16) & 255) * amount + Float32((previous >> 16) & 255) * inverse
        var green = Float32((color >> 8) & 255) * amount + Float32((previous >> 8) & 255) * inverse
        var blue = Float32(color & 255) * amount + Float32(previous & 255) * inverse
        self.pixels[index] = rgb(clamp_byte(red), clamp_byte(green), clamp_byte(blue))

    @staticmethod
    def _compress_hdr_byte(raw_hdr: Float32) -> Int:
        var excess = max(Float32(0.0), raw_hdr - Float32(175.0))
        var mapped = min(raw_hdr, Float32(175.0)) + (excess * Float32(80.0)) / (excess + Float32(95.0))
        return max(0, min(255, Int(mapped)))

    @always_inline
    def _lookup_tone_interp(self, val: Float32, ch_offset: Int) -> Float32:
        var v = max(Float32(0.0), min(Float32(255.0), val))
        var i0 = min(255, Int(v))
        var i1 = min(255, i0 + 1)
        var frac = v - Float32(i0)
        var l0 = Float32(self.tone_lut[ch_offset + i0])
        var l1 = Float32(self.tone_lut[ch_offset + i1])
        return l0 + (l1 - l0) * frac

    def _grade_hdr_pixel(self, red: Float32, green: Float32, blue: Float32, dither: Float32 = 0.0) -> UInt32:
        var hdr_luma = red * 0.26 + green * 0.58 + blue * 0.16
        var peak0 = max(max(red, green), max(blue, Float32(0.0001)))
        var hi_t = max(Float32(0.0), min(Float32(1.0), (peak0 - Float32(172.0)) * Float32(0.0048)))
        var agx_inset = Float32(0.045) + Float32(0.30) * hi_t
        var in_r = red + (hdr_luma - red) * agx_inset
        var in_g = green + (hdr_luma - green) * agx_inset
        var in_b = blue + (hdr_luma - blue) * agx_inset
        var peak_in = max(max(in_r, in_g), max(in_b, Float32(0.0001)))
        var excess = max(Float32(0.0), peak_in - Float32(175.0))
        var mapped_peak = min(peak_in, Float32(175.0)) + (excess * Float32(80.0)) / (excess + Float32(95.0))
        var hdr_scale = mapped_peak / peak_in
        var mapped_red = max(0, min(255, Int(self._lookup_tone_interp(in_r * hdr_scale, 0) + dither)))
        var mapped_green = max(0, min(255, Int(self._lookup_tone_interp(in_g * hdr_scale, 256) + dither)))
        var mapped_blue = max(0, min(255, Int(self._lookup_tone_interp(in_b * hdr_scale, 512) + dither)))
        if self.post_saturation_steps != 0:
            var luminance = (mapped_red * 66 + mapped_green * 150 + mapped_blue * 40) >> 8
            mapped_red = luminance + ((mapped_red - luminance) * (64 + self.post_saturation_steps) >> 6)
            mapped_green = luminance + ((mapped_green - luminance) * (64 + self.post_saturation_steps) >> 6)
            mapped_blue = luminance + ((mapped_blue - luminance) * (64 + self.post_saturation_steps) >> 6)
        return rgb(UInt8(max(0, min(255, mapped_red))), UInt8(max(0, min(255, mapped_green))), UInt8(max(0, min(255, mapped_blue))))

    def shade(self, base: UInt32, light: Vec3, fog: Float32, emissive: Float32, spec_rgb: Vec3 = Vec3()) -> UInt32:
        var red = Float32((base >> 16) & 255) * (light.x + emissive * 1.25) + spec_rgb.x
        var green = Float32((base >> 8) & 255) * (light.y + emissive * 1.25) + spec_rgb.y
        var blue = Float32(base & 255) * (light.z + emissive * 1.25) + spec_rgb.z
        red = red * (1.0 - fog) + Float32((self.fog_color >> 16) & 255) * fog
        green = green * (1.0 - fog) + Float32((self.fog_color >> 8) & 255) * fog
        blue = blue * (1.0 - fog) + Float32(self.fog_color & 255) * fog
        return (base & 0xFF000000) | self._grade_hdr_pixel(red, green, blue)

    def record_bloom(mut self, x: Int, y: Int, base: UInt32, emissive: Float32, alpha: Int):
        if not self.bloom_enabled or emissive < 0.16 or alpha < 24:
            return
        var bx = min(self.bloom_width - 1, x * self.bloom_width // self.width)
        var by = min(self.bloom_height - 1, y * self.bloom_height // self.height)
        var index = (by * self.bloom_width + bx) * 3
        var amount = emissive * (Float32(alpha) / 255.0)
        var red = Float32((base >> 16) & 255) * amount
        var green = Float32((base >> 8) & 255) * amount
        var blue = Float32(base & 255) * amount
        # Max-combine prevents overlapping emissive triangles from accumulating
        # unbounded energy in the low-resolution target.
        self.bloom[index] = max(self.bloom[index], red)
        self.bloom[index + 1] = max(self.bloom[index + 1], green)
        self.bloom[index + 2] = max(self.bloom[index + 2], blue)
        self.bloom_pixels += 1

    def blur_bloom(mut self):
        if not self.bloom_enabled or self.bloom_pixels == 0:
            return
        var w0 = self.bloom_width
        var h0 = self.bloom_height
        for y in range(h0):
            for x in range(w0):
                var index = (y * w0 + x) * 3
                var l2 = max(0, x - 2)
                var l1 = max(0, x - 1)
                var r1 = min(w0 - 1, x + 1)
                var r2 = min(w0 - 1, x + 2)
                for channel in range(3):
                    self.bloom_temporary[index + channel] = (
                        self.bloom[(y * w0 + l2) * 3 + channel] * 0.0625
                        + self.bloom[(y * w0 + l1) * 3 + channel] * 0.25
                        + self.bloom[index + channel] * 0.375
                        + self.bloom[(y * w0 + r1) * 3 + channel] * 0.25
                        + self.bloom[(y * w0 + r2) * 3 + channel] * 0.0625
                    )
        for y in range(h0):
            var u2 = max(0, y - 2)
            var u1 = max(0, y - 1)
            var d1 = min(h0 - 1, y + 1)
            var d2 = min(h0 - 1, y + 2)
            for x in range(w0):
                var index = (y * w0 + x) * 3
                for channel in range(3):
                    self.bloom[index + channel] = (
                        self.bloom_temporary[(u2 * w0 + x) * 3 + channel] * 0.0625
                        + self.bloom_temporary[(u1 * w0 + x) * 3 + channel] * 0.25
                        + self.bloom_temporary[index + channel] * 0.375
                        + self.bloom_temporary[(d1 * w0 + x) * 3 + channel] * 0.25
                        + self.bloom_temporary[(d2 * w0 + x) * 3 + channel] * 0.0625
                    )

        # Multi-scale Dual-Kawase pyramid (Mip0 -> Mip1 -> Mip2 -> Tent Upsample) matching Vulkan
        if w0 >= 16 and h0 >= 12:
            var w1 = max(2, w0 // 2)
            var h1 = max(2, h0 // 2)
            var w2 = max(2, w1 // 2)
            var h2 = max(2, h1 // 2)
            var mip1 = List[Float32](length=w1 * h1 * 3, fill=0)
            var mip2 = List[Float32](length=w2 * h2 * 3, fill=0)
            for y in range(h1):
                var cy = min(h0 - 1, y * 2)
                var uy = max(0, cy - 1)
                var dy = min(h0 - 1, cy + 1)
                for x in range(w1):
                    var cx = min(w0 - 1, x * 2)
                    var lx = max(0, cx - 2)
                    var rx = min(w0 - 1, cx + 2)
                    var o = (y * w1 + x) * 3
                    for ch in range(3):
                        var mid_v = self.bloom[(cy * w0 + cx) * 3 + ch]
                        var crn = (
                            self.bloom[(uy * w0 + lx) * 3 + ch]
                            + self.bloom[(uy * w0 + rx) * 3 + ch]
                            + self.bloom[(dy * w0 + lx) * 3 + ch]
                            + self.bloom[(dy * w0 + rx) * 3 + ch]
                        )
                        mip1[o + ch] = mid_v * 0.5 + crn * 0.125
            for y in range(h2):
                var cy = min(h1 - 1, y * 2)
                var uy = max(0, cy - 1)
                var dy = min(h1 - 1, cy + 1)
                for x in range(w2):
                    var cx = min(w1 - 1, x * 2)
                    var lx = max(0, cx - 2)
                    var rx = min(w1 - 1, cx + 2)
                    var o = (y * w2 + x) * 3
                    for ch in range(3):
                        var mid_v = mip1[(cy * w1 + cx) * 3 + ch]
                        var crn = (
                            mip1[(uy * w1 + lx) * 3 + ch]
                            + mip1[(uy * w1 + rx) * 3 + ch]
                            + mip1[(dy * w1 + lx) * 3 + ch]
                            + mip1[(dy * w1 + rx) * 3 + ch]
                        )
                        mip2[o + ch] = mid_v * 0.5 + crn * 0.125
            for y in range(h1):
                var uy = min(h2 - 1, y // 2)
                for x in range(w1):
                    var ux = min(w2 - 1, x // 2)
                    var o = (y * w1 + x) * 3
                    for ch in range(3):
                        mip1[o + ch] += mip2[(uy * w2 + ux) * 3 + ch] * 0.74
            for y in range(h0):
                var fy = max(Float32(0.0), (Float32(y) + 0.5) * Float32(h1) / Float32(h0) - 0.5)
                var y0 = min(h1 - 1, Int(fy))
                var y1 = min(h1 - 1, y0 + 1)
                var ty = fy - Float32(y0)
                for x in range(w0):
                    var fx = max(Float32(0.0), (Float32(x) + 0.5) * Float32(w1) / Float32(w0) - 0.5)
                    var x0 = min(w1 - 1, Int(fx))
                    var x1 = min(w1 - 1, x0 + 1)
                    var tx = fx - Float32(x0)
                    var o = (y * w0 + x) * 3
                    for ch in range(3):
                        var s00 = mip1[(y0 * w1 + x0) * 3 + ch]
                        var s10 = mip1[(y0 * w1 + x1) * 3 + ch]
                        var s01 = mip1[(y1 * w1 + x0) * 3 + ch]
                        var s11 = mip1[(y1 * w1 + x1) * 3 + ch]
                        var wide = (s00 + (s10 - s00) * tx) * (1.0 - ty) + (s01 + (s11 - s01) * tx) * ty
                        self.bloom[o + ch] = self.bloom[o + ch] * 0.68 + wide * 0.66

        for index in range(len(self.bloom)):
            self.bloom_bytes[index] = UInt8(max(0, min(255, Int(self.bloom[index]))))

    @staticmethod
    def _halton_jitter(frame_idx: Int) -> Tuple[Float32, Float32]:
        var idx = (frame_idx % 8) + 1
        var h2 = Float32(0.0)
        var f2 = Float32(0.5)
        var n2 = idx
        while n2 > 0:
            h2 += Float32(n2 & 1) * f2
            n2 >>= 1
            f2 *= 0.5
        var h3 = Float32(0.0)
        var f3 = Float32(1.0 / 3.0)
        var n3 = idx
        while n3 > 0:
            h3 += Float32(n3 % 3) * f3
            n3 //= 3
            f3 *= Float32(1.0 / 3.0)
        return ((h2 - 0.5) * Float32(0.68), (h3 - 0.5) * Float32(0.68))

    @staticmethod
    def _decode_oct_normal(gbuf: UInt32) -> Vec3:
        var ox = Float32(gbuf & 255) * Float32(2.0 / 255.0) - Float32(1.0)
        var oz = Float32((gbuf >> 8) & 255) * Float32(2.0 / 255.0) - Float32(1.0)
        var ny = Float32(1.0) - abs(ox) - abs(oz)
        var nx = ox
        var nz = oz
        if ny < 0.0:
            nx = (Float32(1.0) - abs(oz)) * (Float32(1.0) if ox >= 0.0 else Float32(-1.0))
            nz = (Float32(1.0) - abs(ox)) * (Float32(1.0) if oz >= 0.0 else Float32(-1.0))
        return Vec3(nx, ny, nz).normalized()

    @staticmethod
    def _encode_oct_normal(n: Vec3) -> Tuple[UInt32, UInt32]:
        var inv_l1 = Float32(1.0) / max(Float32(0.000001), abs(n.x) + abs(n.y) + abs(n.z))
        var ox = n.x * inv_l1
        var oz = n.z * inv_l1
        if n.y < 0.0:
            var ox_old = ox
            ox = (Float32(1.0) - abs(oz)) * (Float32(1.0) if ox_old >= 0.0 else Float32(-1.0))
            oz = (Float32(1.0) - abs(ox_old)) * (Float32(1.0) if oz >= 0.0 else Float32(-1.0))
        var bx = UInt32(max(0, min(255, Int((ox * 0.5 + 0.5) * 255.0 + 0.5))))
        var bz = UInt32(max(0, min(255, Int((oz * 0.5 + 0.5) * 255.0 + 0.5))))
        return (bx, bz)

    @staticmethod
    def _rgb_to_ycocg_f(col: UInt32) -> Vec3:
        var r = Float32((col >> 16) & 255)
        var g = Float32((col >> 8) & 255)
        var b = Float32(col & 255)
        var co = (r - b) * 0.5
        var tmp = b + co
        var cg = (g - tmp) * 0.5
        var y = tmp + cg
        return Vec3(y, co, cg)

    def _compute_temporal_params(mut self, taa_enabled: Bool) -> Tuple[Float32, Float32, Float32, Float32]:
        var cam_cut = (
            self.temporal_frames == 0
            or (self.camera - self.prev_camera).length_squared() > Float32(324.0)
            or dot(self.forward, self.prev_forward) < Float32(0.70)
            or abs(self.focal - self.prev_focal) > self.focal * Float32(0.15)
        )
        if not taa_enabled or cam_cut:
            self.prev_camera = self.camera
            self.prev_right = self.right
            self.prev_up = self.up
            self.prev_forward = self.forward
            self.prev_focal = self.focal
            self.taa_jitter_x = 0.0
            self.taa_jitter_y = 0.0
            return (Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0))
        var jit = Self._halton_jitter(self.temporal_frames)
        self.taa_jitter_x = jit[0]
        self.taa_jitter_y = jit[1]
        return (Float32(0.62), Float32(0.52), jit[0], jit[1])

    def _commit_temporal_camera(mut self):
        self.prev_camera = self.camera
        self.prev_right = self.right
        self.prev_up = self.up
        self.prev_forward = self.forward
        self.prev_focal = self.focal
        self.prev_jitter_x = self.taa_jitter_x
        self.history_parity = 1 - self.history_parity
        self.temporal_frames += 1

    def _build_cpu_sky_lut(self) -> List[Float32]:
        """Build 128x64 Hillaire Sky-View LUT on CPU matching `sky_lut_shader()`."""
        var lut = List[Float32](length=SKY_LUT_W * SKY_LUT_H * 4, fill=0.0)
        var to_key = self.cpu_to_key
        var key_color = self.cpu_key_color
        var env_sky = self.cpu_env_sky
        var env_ground = self.cpu_env_ground
        var env_horizon = self.cpu_env_horizon
        var sun_h = sqrt(max(Float32(0.0), to_key.x * to_key.x + to_key.z * to_key.z))
        var sun_loc = Vec3(sun_h, to_key.y, 0.0)
        var alt_km = max(Float32(0.02), self.cam_altitude_km)
        var r_planet = max(Float32(100.0), self.planet_radius_km)
        var atmos_h = max(Float32(12.0), self.atmos_thickness_km)
        var rayleigh_m = max(Float32(0.0), self.rayleigh_density)
        var mie_m = max(Float32(0.0), self.mie_density)
        var ozone_m = max(Float32(0.0), self.ozone_strength)
        var r_cam = r_planet + alt_km
        var r_top = r_planet + atmos_h
        var c_top = r_cam * r_cam - r_top * r_top
        var c_pl = r_cam * r_cam - r_planet * r_planet

        for row in range(SKY_LUT_H):
            var v_n = (Float32(row) + Float32(0.5)) / Float32(SKY_LUT_H)
            var s_v = Float32(1.0) - v_n * Float32(2.0)
            var dir_y = max(Float32(-0.9999), min(Float32(0.9999), s_v * abs(s_v)))
            var horiz_r = sqrt(max(Float32(0.0), Float32(1.0) - dir_y * dir_y))
            var b_dot = r_cam * dir_y
            var disc_top = b_dot * b_dot - c_top
            var disc_pl = b_dot * b_dot - c_pl
            var t_pl_near = -b_dot - sqrt(max(Float32(0.0), disc_pl))
            var hits_planet = disc_pl > Float32(0.0) and t_pl_near > Float32(0.0)

            for col in range(SKY_LUT_W):
                var u_n = (Float32(col) + Float32(0.5)) / Float32(SKY_LUT_W)
                var cos_az = Float32(1.0) - u_n * Float32(2.0)
                var sin_az = sqrt(max(Float32(0.0), Float32(1.0) - cos_az * cos_az))
                var ray_loc = Vec3(horiz_r * cos_az, dir_y, horiz_r * sin_az)
                var mu = dot(ray_loc, sun_loc)
                var phase_r = Float32(0.0596831) * (Float32(1.0) + mu * mu)
                var g_den = Float32(1.6084) - Float32(1.56) * mu
                var phase_m = min(Float32(2.4), Float32(0.0454) / max(Float32(0.015), g_den * sqrt(max(Float32(0.0001), g_den))))

                var rad_r = Float32(0.0)
                var rad_g = Float32(0.0)
                var rad_b = Float32(0.0)
                var tr_r = Float32(1.0)
                var tr_g = Float32(1.0)
                var tr_b = Float32(1.0)

                if disc_top > Float32(0.0):
                    var sq_top = sqrt(disc_top)
                    var t_start = max(Float32(0.0), -b_dot - sq_top)
                    var t_exit = max(Float32(0.0), sq_top - b_dot)
                    var t_end = min(t_exit, t_pl_near) if hits_planet else t_exit
                    var seg_len = max(Float32(0.0), t_end - t_start)
                    var dt = seg_len * Float32(0.1666667)
                    if seg_len > Float32(0.001):
                        for si in range(6):
                            var t_cur = t_start + (Float32(si) + Float32(0.5)) * dt
                            var px = ray_loc.x * t_cur
                            var py = r_cam + ray_loc.y * t_cur
                            var pz = ray_loc.z * t_cur
                            var r_here = sqrt(px * px + py * py + pz * pz + Float32(0.001))
                            var h_km = max(Float32(0.0), r_here - r_planet)
                            var rho_r = exp(-h_km * Float32(0.125)) * rayleigh_m
                            var rho_m = exp(-h_km * Float32(0.80)) * mie_m
                            var rho_o = max(Float32(0.0), Float32(1.0) - abs(h_km - Float32(25.0)) * Float32(0.066)) * ozone_m
                            var ext_r = rho_r * Float32(0.0068) + rho_m * Float32(0.0048) + rho_o * Float32(0.0018)
                            var ext_g = rho_r * Float32(0.0138) + rho_m * Float32(0.0048) + rho_o * Float32(0.0030)
                            var ext_b = rho_r * Float32(0.0245) + rho_m * Float32(0.0048) + rho_o * Float32(0.0002)
                            var step_tr_r = exp(-ext_r * dt)
                            var step_tr_g = exp(-ext_g * dt)
                            var step_tr_b = exp(-ext_b * dt)
                            var up_s = Vec3(px / r_here, py / r_here, pz / r_here)
                            var sun_zen = dot(up_s, sun_loc)
                            if sun_zen > Float32(-0.20):
                                var chap = Float32(1.0) / max(Float32(0.05), sun_zen + Float32(0.22))
                                var st_r = exp(-(rho_r * Float32(0.078) + rho_m * Float32(0.009)) * chap * Float32(0.69314718))
                                var st_g = exp(-(rho_r * Float32(0.158) + rho_o * Float32(0.036) + rho_m * Float32(0.009)) * chap * Float32(0.69314718))
                                var st_b = exp(-(rho_r * Float32(0.285) + rho_m * Float32(0.009)) * chap * Float32(0.69314718))
                                var sc_r = rho_r * Float32(0.0068) * phase_r + rho_m * Float32(0.0044) * phase_m
                                var sc_g = rho_r * Float32(0.0138) * phase_r + rho_m * Float32(0.0044) * phase_m
                                var sc_b = rho_r * Float32(0.0245) * phase_r + rho_m * Float32(0.0044) * phase_m
                                var sun_e = Float32(1380.0)
                                var in_r = sc_r * st_r * key_color.x * sun_e + env_horizon.x * rho_r * Float32(0.68)
                                var in_g = sc_g * st_g * key_color.y * sun_e + env_horizon.y * rho_r * Float32(0.68)
                                var in_b = sc_b * st_b * key_color.z * sun_e + env_horizon.z * rho_r * Float32(0.68)
                                var int_r = (Float32(1.0) - step_tr_r) / max(Float32(0.00001), ext_r)
                                var int_g = (Float32(1.0) - step_tr_g) / max(Float32(0.00001), ext_g)
                                var int_b = (Float32(1.0) - step_tr_b) / max(Float32(0.00001), ext_b)
                                rad_r += tr_r * in_r * int_r
                                rad_g += tr_g * in_g * int_g
                                rad_b += tr_b * in_b * int_b
                            tr_r *= step_tr_r
                            tr_g *= step_tr_g
                            tr_b *= step_tr_b

                if hits_planet:
                    var hx_p = ray_loc.x * t_pl_near
                    var hy_p = r_cam + ray_loc.y * t_pl_near
                    var hz_p = ray_loc.z * t_pl_near
                    var n_pl = Vec3(hx_p / r_planet, hy_p / r_planet, hz_p / r_planet)
                    var ndl_pl = max(Float32(0.0), dot(n_pl, sun_loc))
                    var surf_lit = (ndl_pl * Float32(1.35) + Float32(0.10)) * Float32(185.0)
                    rad_r += (env_ground.x * Float32(0.85) + env_horizon.x * Float32(0.32)) * surf_lit * tr_r
                    rad_g += (env_ground.y * Float32(0.92) + env_horizon.y * Float32(0.32)) * surf_lit * tr_g
                    rad_b += (env_ground.z * Float32(0.96) + env_horizon.z * Float32(0.32)) * surf_lit * tr_b
                    tr_r = 0.0
                    tr_g = 0.0
                    tr_b = 0.0
                else:
                    var space_w = max(Float32(0.15), min(Float32(1.0), alt_km * Float32(0.012)))
                    var sp_scale = space_w * Float32(28.0)
                    rad_r += env_sky.x * sp_scale * tr_r
                    rad_g += env_sky.y * sp_scale * tr_g
                    rad_b += env_sky.z * sp_scale * tr_b

                var out_i = (row * SKY_LUT_W + col) * 4
                lut[out_i] = rad_r
                lut[out_i + 1] = rad_g
                lut[out_i + 2] = rad_b
                lut[out_i + 3] = (tr_r + tr_g + tr_b) * Float32(0.3333333)
        return lut^

    def _sample_cpu_sky_lut(self, lut: List[Float32], ray_dir: Vec3) -> Vec3:
        var rh_len = sqrt(max(Float32(0.0), ray_dir.x * ray_dir.x + ray_dir.z * ray_dir.z))
        var kh_len = sqrt(max(Float32(0.0), self.cpu_to_key.x * self.cpu_to_key.x + self.cpu_to_key.z * self.cpu_to_key.z))
        var cos_az = max(
            Float32(-1.0),
            min(Float32(1.0), (ray_dir.x * self.cpu_to_key.x + ray_dir.z * self.cpu_to_key.z) / max(Float32(0.0001), rh_len * kh_len)),
        )
        var u_norm = Float32(0.5) - cos_az * Float32(0.5)
        var sign_y = Float32(1.0) if ray_dir.y >= Float32(0.0) else Float32(-1.0)
        var v_norm = max(Float32(0.0), min(Float32(1.0), Float32(0.5) - sign_y * sqrt(abs(ray_dir.y)) * Float32(0.5)))
        var uf = max(Float32(0.0), min(Float32(SKY_LUT_W - 1), u_norm * Float32(SKY_LUT_W) - Float32(0.5)))
        var vf = max(Float32(0.0), min(Float32(SKY_LUT_H - 1), v_norm * Float32(SKY_LUT_H) - Float32(0.5)))
        var x0 = min(SKY_LUT_W - 1, max(0, Int(uf)))
        var x1 = min(SKY_LUT_W - 1, x0 + 1)
        var y0 = min(SKY_LUT_H - 1, max(0, Int(vf)))
        var y1 = min(SKY_LUT_H - 1, y0 + 1)
        var tx = uf - Float32(x0)
        var ty = vf - Float32(y0)
        var i00 = (y0 * SKY_LUT_W + x0) * 4
        var i10 = (y0 * SKY_LUT_W + x1) * 4
        var i01 = (y1 * SKY_LUT_W + x0) * 4
        var i11 = (y1 * SKY_LUT_W + x1) * 4
        var sr = (lut[i00] + (lut[i10] - lut[i00]) * tx) * (1.0 - ty) + (lut[i01] + (lut[i11] - lut[i01]) * tx) * ty
        var sg = (lut[i00 + 1] + (lut[i10 + 1] - lut[i00 + 1]) * tx) * (1.0 - ty) + (lut[i01 + 1] + (lut[i11 + 1] - lut[i01 + 1]) * tx) * ty
        var sb = (lut[i00 + 2] + (lut[i10 + 2] - lut[i00 + 2]) * tx) * (1.0 - ty) + (lut[i01 + 2] + (lut[i11 + 2] - lut[i01 + 2]) * tx) * ty
        var tr_sun = lut[i00 + 3]
        var disc = max(Float32(0.0), min(Float32(1.0), (dot(ray_dir, self.cpu_to_key) - Float32(0.99935)) * Float32(2200.0)))
        var sun_gain = disc * disc * tr_sun * Float32(320.0)
        return Vec3(
            sr + self.cpu_key_color.x * sun_gain,
            sg + self.cpu_key_color.y * sun_gain,
            sb + self.cpu_key_color.z * sun_gain * Float32(0.88),
        )

    def _build_cpu_froxel_grid(mut self, fog_col: Vec3, temp_blend: Float32) -> List[Float32]:
        """Build 64x36x16 temporally reprojected 3D froxel scattering & prefix integration grid on CPU."""
        var total_cells = FROXEL_X * FROXEL_Y * FROXEL_Z * 4
        var next_hist = List[Float32](length=total_cells, fill=0.0)
        var integ = List[Float32](length=total_cells, fill=0.0)
        var half_w = Float32(self.width) * Float32(0.5)
        var half_h = Float32(self.height) * Float32(0.5)
        var inv_focal = Float32(1.0) / max(Float32(0.0001), self.focal)
        var frox_max = max(Float32(10.0), self.froxel_max_dist)
        var use_hist = self.cpu_froxel_valid and len(self.cpu_froxel_hist) == total_cells and temp_blend > Float32(0.001)
        var n_vlights = min(8, len(self.cpu_lights))
        var frame_phase = Float32(self.temporal_frames & 15) * Float32(0.618034)

        for fy in range(FROXEL_Y):
            var v_scr = (Float32(fy) + Float32(0.5)) / Float32(FROXEL_Y)
            var vy = (half_h - v_scr * half_h * Float32(2.0)) * inv_focal
            for fx in range(FROXEL_X):
                var u_scr = (Float32(fx) + Float32(0.5)) / Float32(FROXEL_X)
                var vx = (u_scr * half_w * Float32(2.0) - half_w) * inv_focal
                var ray_dir = (self.forward + self.right * vx + self.up * vy).normalized()
                var cos_th = dot(ray_dir, self.cpu_to_key)
                var hg_den = Float32(1.4096) - Float32(1.28) * cos_th
                var hg_phase = min(Float32(1.55), Float32(0.04698) / max(Float32(0.02), hg_den * sqrt(max(Float32(0.01), hg_den))))
                var sun_r = self.cpu_key_color.x * hg_phase * Float32(195.0)
                var sun_g = self.cpu_key_color.y * hg_phase * Float32(190.0)
                var sun_b = self.cpu_key_color.z * hg_phase * Float32(172.0)
                var ign_i = Float32(fx) * Float32(0.06711056) + Float32(fy) * Float32(0.00583715)
                var ign_0 = Float32(52.9829189) * (ign_i - floor(ign_i)) + frame_phase
                var ign_f = ign_0 - floor(ign_0)

                var cum_r = Float32(0.0)
                var cum_g = Float32(0.0)
                var cum_b = Float32(0.0)
                var cum_tr = Float32(1.0)

                for zi in range(FROXEL_Z):
                    var zf0 = Float32(zi) / Float32(FROXEL_Z)
                    var zf1 = Float32(zi + 1) / Float32(FROXEL_Z)
                    var d0 = frox_max * zf0 * zf0
                    var d1 = frox_max * zf1 * zf1
                    var dt = max(Float32(0.05), d1 - d0)
                    var d_jit = d0 + dt * (Float32(0.2) + ign_f * Float32(0.6))
                    var pw = self.camera + ray_dir * d_jit
                    var alt_h = max(Float32(0.0), pw.y + Float32(6.0))
                    var rho_cur = exp(-alt_h * self.vol_height_falloff * Float32(0.69314718)) * (self.vol_density * Float32(0.0036))
                    var vis_s = self.cpu_csm_shadow_1tap(pw, self.cpu_to_key) * self.cpu_sphere_shadow(pw, self.cpu_to_key)
                    var loc_r = Float32(0.0)
                    var loc_g = Float32(0.0)
                    var loc_b = Float32(0.0)
                    for li in range(n_vlights):
                        var pl = self.cpu_lights[li]
                        var d2 = (pl.position - pw).length_squared() + Float32(1.5)
                        var r2 = max(Float32(0.04), pl.radius * pl.radius)
                        if d2 < r2:
                            var fall = Float32(1.0) - d2 / r2
                            var l_att = fall * fall * pl.intensity * Float32(460.0)
                            loc_r += pl.color.x * l_att
                            loc_g += pl.color.y * l_att
                            loc_b += pl.color.z * l_att
                    var sr_fin = (fog_col.x * Float32(0.62) + sun_r * vis_s + loc_r) * rho_cur
                    var sg_fin = (fog_col.y * Float32(0.62) + sun_g * vis_s + loc_g) * rho_cur
                    var sb_fin = (fog_col.z * Float32(0.62) + sun_b * vis_s + loc_b) * rho_cur
                    var rho_fin = rho_cur

                    if use_hist:
                        var d_mid = (d0 + d1) * Float32(0.5)
                        var p_mid = (self.camera - self.prev_camera) + ray_dir * d_mid
                        var prev_vz = dot(p_mid, self.prev_forward)
                        if prev_vz > Float32(0.25):
                            var pu = (half_w + dot(p_mid, self.prev_right) * self.prev_focal / prev_vz) / max(Float32(1.0), half_w * Float32(2.0))
                            var pv = (half_h - dot(p_mid, self.prev_up) * self.prev_focal / prev_vz) / max(Float32(1.0), half_h * Float32(2.0))
                            if pu >= 0.0 and pu < 1.0 and pv >= 0.0 and pv < 1.0:
                                var pz_n = sqrt(max(Float32(0.0), min(Float32(1.0), p_mid.length() / frox_max)))
                                var pfx = min(FROXEL_X - 1, max(0, Int(pu * Float32(FROXEL_X))))
                                var pfy = min(FROXEL_Y - 1, max(0, Int(pv * Float32(FROXEL_Y))))
                                var pfz = min(FROXEL_Z - 1, max(0, Int(pz_n * Float32(FROXEL_Z))))
                                var p_i = ((pfz * FROXEL_Y + pfy) * FROXEL_X + pfx) * 4
                                sr_fin += (self.cpu_froxel_hist[p_i] - sr_fin) * temp_blend
                                sg_fin += (self.cpu_froxel_hist[p_i + 1] - sg_fin) * temp_blend
                                sb_fin += (self.cpu_froxel_hist[p_i + 2] - sb_fin) * temp_blend
                                rho_fin += (self.cpu_froxel_hist[p_i + 3] - rho_fin) * temp_blend

                    var c_i = ((zi * FROXEL_Y + fy) * FROXEL_X + fx) * 4
                    next_hist[c_i] = sr_fin
                    next_hist[c_i + 1] = sg_fin
                    next_hist[c_i + 2] = sb_fin
                    next_hist[c_i + 3] = rho_fin

                    var ext_sl = min(Float32(0.65), rho_fin * dt)
                    var tr_sl = exp(-ext_sl)
                    var step_int = (Float32(1.0) - tr_sl) / max(Float32(0.00001), rho_fin)
                    var w_int = cum_tr * step_int
                    cum_r += sr_fin * w_int
                    cum_g += sg_fin * w_int
                    cum_b += sb_fin * w_int
                    cum_tr *= tr_sl
                    integ[c_i] = cum_r
                    integ[c_i + 1] = cum_g
                    integ[c_i + 2] = cum_b
                    integ[c_i + 3] = cum_tr

        self.cpu_froxel_hist = next_hist^
        self.cpu_froxel_valid = True
        return integ^

    @staticmethod
    def _sample_cpu_froxel_trilinear(integ: List[Float32], fu: Float32, fv: Float32, fz: Float32) -> Tuple[Float32, Float32, Float32, Float32]:
        var x0 = min(FROXEL_X - 1, max(0, Int(fu)))
        var x1 = min(FROXEL_X - 1, x0 + 1)
        var y0 = min(FROXEL_Y - 1, max(0, Int(fv)))
        var y1 = min(FROXEL_Y - 1, y0 + 1)
        var z0 = min(FROXEL_Z - 1, max(0, Int(fz)))
        var z1 = min(FROXEL_Z - 1, z0 + 1)
        var tx = max(Float32(0.0), min(Float32(1.0), fu - Float32(x0)))
        var ty = max(Float32(0.0), min(Float32(1.0), fv - Float32(y0)))
        var tz = max(Float32(0.0), min(Float32(1.0), fz - Float32(z0)))
        var i000 = ((z0 * FROXEL_Y + y0) * FROXEL_X + x0) * 4
        var i100 = ((z0 * FROXEL_Y + y0) * FROXEL_X + x1) * 4
        var i010 = ((z0 * FROXEL_Y + y1) * FROXEL_X + x0) * 4
        var i110 = ((z0 * FROXEL_Y + y1) * FROXEL_X + x1) * 4
        var i001 = ((z1 * FROXEL_Y + y0) * FROXEL_X + x0) * 4
        var i101 = ((z1 * FROXEL_Y + y0) * FROXEL_X + x1) * 4
        var i011 = ((z1 * FROXEL_Y + y1) * FROXEL_X + x0) * 4
        var i111 = ((z1 * FROXEL_Y + y1) * FROXEL_X + x1) * 4
        var outs = Array[Float32, 4](fill=0.0)
        for ch in range(4):
            var x00 = integ[i000 + ch] + (integ[i100 + ch] - integ[i000 + ch]) * tx
            var x10 = integ[i010 + ch] + (integ[i110 + ch] - integ[i010 + ch]) * tx
            var x01 = integ[i001 + ch] + (integ[i101 + ch] - integ[i001 + ch]) * tx
            var x11 = integ[i011 + ch] + (integ[i111 + ch] - integ[i011 + ch]) * tx
            var y0_v = x00 + (x10 - x00) * ty
            var y1_v = x01 + (x11 - x01) * ty
            outs[ch] = y0_v + (y1_v - y0_v) * tz
        return (outs[0], outs[1], outs[2], outs[3])

    def _build_cpu_hiz_pyramid(mut self):
        """Build 16x16 coarse Hierarchical-Z min(inverse_z) pyramid on CPU matching `SHADER_HIZ_PYRAMID`."""
        var tx_cnt = (self.width + 15) // 16
        var ty_cnt = (self.height + 15) // 16
        if tx_cnt <= 0 or ty_cnt <= 0:
            self.cpu_hiz_valid = False
            return
        if len(self.cpu_hiz_min_iz) != tx_cnt * ty_cnt:
            self.cpu_hiz_min_iz = List[Float32](length=tx_cnt * ty_cnt, fill=0.0)
        self.cpu_hiz_tiles_x = tx_cnt
        self.cpu_hiz_tiles_y = ty_cnt
        for ty in range(ty_cnt):
            var y0 = ty * 16
            var y1 = min(self.height, y0 + 16)
            for tx in range(tx_cnt):
                var x0 = tx * 16
                var x1 = min(self.width, x0 + 16)
                var min_iz = Float32(1e9)
                for py in range(y0, y1):
                    var r_off = py * self.width
                    for px in range(x0, x1):
                        var d = self.depth[r_off + px]
                        var iz = Float32(0.0) if d >= Float32(99999999.0) else (Float32(1.0) / max(Float32(0.0001), d))
                        if iz < min_iz:
                            min_iz = iz
                self.cpu_hiz_min_iz[ty * tx_cnt + tx] = min_iz
        self.cpu_hiz_valid = True

    def is_rect_occluded_hiz(
        self,
        min_px_x: Int,
        min_px_y: Int,
        max_px_x: Int,
        max_px_y: Int,
        nearest_iz: Float32,
    ) -> Bool:
        """Conservative 16x16 Hi-Z occlusion check against the most recent completed frame."""
        if not self.hiz_cull_enabled or nearest_iz <= Float32(0.0):
            return False
        if (self.camera - self.prev_camera).length_squared() > Float32(16.0) or dot(self.forward, self.prev_forward) < Float32(0.96):
            return False
        if self.gpu.active:
            return self.gpu.is_rect_occluded_hiz(min_px_x, min_px_y, max_px_x, max_px_y, nearest_iz)
        if not self.cpu_hiz_valid or self.cpu_hiz_tiles_x <= 0 or self.cpu_hiz_tiles_y <= 0:
            return False
        var tx0 = max(0, min(self.cpu_hiz_tiles_x - 1, min_px_x >> 4))
        var ty0 = max(0, min(self.cpu_hiz_tiles_y - 1, min_px_y >> 4))
        var tx1 = max(0, min(self.cpu_hiz_tiles_x - 1, max_px_x >> 4))
        var ty1 = max(0, min(self.cpu_hiz_tiles_y - 1, max_px_y >> 4))
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 256:
            return False
        var margin_iz = nearest_iz * Float32(1.06) + Float32(0.0002)
        for ty in range(ty0, ty1 + 1):
            var r_off = ty * self.cpu_hiz_tiles_x
            for tx in range(tx0, tx1 + 1):
                if self.cpu_hiz_min_iz[r_off + tx] <= margin_iz:
                    return False
        return True

    @staticmethod
    def skin_vertices_cpu(
        joint_matrices_3x4: List[Float32],
        positions: List[Float32],
        normals: List[Float32],
        joints_4x_packed: List[UInt32],
        weights_4x: List[Float32],
        vert_count: Int,
        mut out_pos_norm: List[Float32],
    ):
        """4-bone Linear-Blend Skinning CPU parity matching `SHADER_GPU_SKIN`."""
        if len(out_pos_norm) < vert_count * 6:
            out_pos_norm.resize(vert_count * 6, 0.0)
        var max_joints = len(joint_matrices_3x4) // 12
        var has_n = len(normals) >= vert_count * 3
        for v in range(vert_count):
            var px = positions[v * 3]
            var py = positions[v * 3 + 1]
            var pz = positions[v * 3 + 2]
            var nx = normals[v * 3] if has_n else Float32(0.0)
            var ny = normals[v * 3 + 1] if has_n else Float32(1.0)
            var nz = normals[v * 3 + 2] if has_n else Float32(0.0)
            var j_pk = joints_4x_packed[v] if v < len(joints_4x_packed) else UInt32(0)
            var opx = Float32(0.0)
            var opy = Float32(0.0)
            var opz = Float32(0.0)
            var onx = Float32(0.0)
            var ony = Float32(0.0)
            var onz = Float32(0.0)
            for inf in range(4):
                var w_inf = weights_4x[v * 4 + inf] if (v * 4 + inf) < len(weights_4x) else (Float32(1.0) if inf == 0 else Float32(0.0))
                if w_inf > Float32(0.00001):
                    var j_idx = min(max(0, max_joints - 1), Int((j_pk >> UInt32(inf * 8)) & UInt32(63)))
                    var mb = j_idx * 12
                    var r0x = joint_matrices_3x4[mb]
                    var r0y = joint_matrices_3x4[mb + 1]
                    var r0z = joint_matrices_3x4[mb + 2]
                    var r0w = joint_matrices_3x4[mb + 3]
                    var r1x = joint_matrices_3x4[mb + 4]
                    var r1y = joint_matrices_3x4[mb + 5]
                    var r1z = joint_matrices_3x4[mb + 6]
                    var r1w = joint_matrices_3x4[mb + 7]
                    var r2x = joint_matrices_3x4[mb + 8]
                    var r2y = joint_matrices_3x4[mb + 9]
                    var r2z = joint_matrices_3x4[mb + 10]
                    var r2w = joint_matrices_3x4[mb + 11]
                    opx += (r0x * px + r0y * py + r0z * pz + r0w) * w_inf
                    opy += (r1x * px + r1y * py + r1z * pz + r1w) * w_inf
                    opz += (r2x * px + r2y * py + r2z * pz + r2w) * w_inf
                    onx += (r0x * nx + r0y * ny + r0z * nz) * w_inf
                    ony += (r1x * nx + r1y * ny + r1z * nz) * w_inf
                    onz += (r2x * nx + r2y * ny + r2z * nz) * w_inf
            var inv_nl = Float32(1.0) / sqrt(max(Float32(0.000001), onx * onx + ony * ony + onz * onz))
            var dst = v * 6
            out_pos_norm[dst] = opx
            out_pos_norm[dst + 1] = opy
            out_pos_norm[dst + 2] = opz
            out_pos_norm[dst + 3] = onx * inv_nl
            out_pos_norm[dst + 4] = ony * inv_nl
            out_pos_norm[dst + 5] = onz * inv_nl

    def _cloud_density_cpu(self, wx: Float32, wy: Float32, wz: Float32) -> Float32:
        """Matches SPIR-V _cloud_density_3d: 3-octave Gyroid-Worley FBM cumulus density."""
        var span_y = max(Float32(10.0), self.cloud_top_y - self.cloud_base_y)
        var h_t = max(Float32(0.0), min(Float32(1.0), (wy - self.cloud_base_y) / span_y))
        var bot_m = max(Float32(0.0), min(Float32(1.0), h_t * Float32(4.2)))
        var top_m = max(Float32(0.0), min(Float32(1.0), (Float32(1.0) - h_t) * Float32(2.1)))
        var anvil = bot_m * top_m
        var sc = self.cloud_scale * Float32(0.22)
        var ax0 = (wx + self.cloud_wind_x * self.water_time) * sc
        var ay0 = wy * (sc * Float32(1.35))
        var az0 = (wz + self.cloud_wind_z * self.water_time) * sc
        var n0 = _gyroid3d_unit_cpu(ax0 + Float32(0.11), ay0 + Float32(0.23), az0 + Float32(0.37))
        var n1 = _gyroid3d_unit_cpu(
            ax0 * Float32(2.17) + Float32(0.41),
            ay0 * Float32(2.17) + Float32(0.19),
            az0 * Float32(2.17) + Float32(0.53),
        )
        var worley1 = Float32(1.0) - abs(n1 * Float32(2.0) - Float32(1.0))
        var n2 = _gyroid3d_unit_cpu(
            ax0 * Float32(4.43) + Float32(0.67),
            ay0 * Float32(4.43) + Float32(0.31),
            az0 * Float32(4.43) + Float32(0.13),
        )
        var fbm = n0 * Float32(0.54) + worley1 * Float32(0.30) + n2 * Float32(0.16)
        var thresh = (Float32(1.0) - self.cloud_coverage) * Float32(0.68) + Float32(0.10)
        return max(Float32(0.0), min(Float32(1.0), (fbm * anvil - thresh) * Float32(2.8)))

    def _integrate_volumetric_clouds_cpu(
        self, ray_dir: Vec3, max_dist: Float32, ign: Float32
    ) -> Tuple[Float32, Float32, Float32, Float32]:
        """Matches SPIR-V _integrate_volumetric_clouds 6-step cumulus ray-march."""
        if self.cloud_density <= Float32(0.0001):
            return (Float32(0.0), Float32(0.0), Float32(0.0), Float32(1.0))
        var ry = ray_dir.y
        var abs_ry = abs(ry)
        var safe_ry = max(Float32(0.015), ry) if ry >= Float32(0.0) else -max(Float32(0.015), abs_ry)
        var t0_raw = (self.cloud_base_y - self.camera.y) / safe_ry
        var t1_raw = (self.cloud_top_y - self.camera.y) / safe_ry
        var in_slab = self.camera.y >= self.cloud_base_y and self.camera.y <= self.cloud_top_y
        var t_min = Float32(2.0) if in_slab else max(Float32(2.0), min(t0_raw, t1_raw))
        var t_max_raw = (t1_raw if ry >= Float32(0.0) else t0_raw) if in_slab else max(t0_raw, t1_raw)
        var t_end = min(min(t_max_raw, max_dist), Float32(8500.0))
        if t_end <= t_min or (abs_ry + (Float32(1.0) if in_slab else Float32(0.0))) <= Float32(0.025):
            return (Float32(0.0), Float32(0.0), Float32(0.0), Float32(1.0))
        var span = min(t_end - t_min, Float32(2400.0))
        var dt = span * Float32(0.16666667)
        var cos_th = dot(ray_dir, self.cpu_to_key)
        var denom_hg = max(Float32(0.04), Float32(1.3844) - Float32(1.24) * cos_th)
        var hg_fwd = Float32(0.6156) / max(Float32(0.04), denom_hg * sqrt(denom_hg))
        var hg_back = Float32(0.42) - cos_th * Float32(0.18)
        var phase = min(Float32(2.35), hg_fwd * Float32(0.68) + hg_back * Float32(0.45))
        var sun_r = self.cpu_key_color.x * Float32(228.0)
        var sun_g = self.cpu_key_color.y * Float32(218.0)
        var sun_b = self.cpu_key_color.z * Float32(202.0)
        var out_r = Float32(0.0)
        var out_g = Float32(0.0)
        var out_b = Float32(0.0)
        var out_tr = Float32(1.0)
        var inv_slab_h = Float32(1.0) / max(Float32(10.0), self.cloud_top_y - self.cloud_base_y)
        for si in range(6):
            if out_tr < Float32(0.03):
                break
            var t_s = t_min + (Float32(si) + ign) * dt
            var px = self.camera.x + ray_dir.x * t_s
            var py = self.camera.y + ray_dir.y * t_s
            var pz = self.camera.z + ray_dir.z * t_s
            var d_loc = self._cloud_density_cpu(px, py, pz)
            if d_loc > Float32(0.002):
                var h_amb = max(Float32(0.0), min(Float32(1.0), (py - self.cloud_base_y) * inv_slab_h))
                var opt_sun = d_loc * (Float32(2.6) - h_amb * Float32(1.5))
                var beer_sun = exp(-opt_sun * Float32(0.6931472))
                var powder = Float32(1.0) - exp(-d_loc * Float32(2.3567)) * Float32(0.42)
                var lit_e = beer_sun * powder * phase
                var amb_w = Float32(0.55) + h_amb * Float32(0.45)
                var c_r = sun_r * lit_e + Float32(138.0) * amb_w
                var c_g = sun_g * lit_e + Float32(144.0) * amb_w
                var c_b = sun_b * lit_e + Float32(156.0) * amb_w
                var sig_dt = min(Float32(0.88), d_loc * self.cloud_density * dt)
                var w_int = out_tr * sig_dt
                out_r += c_r * w_int
                out_g += c_g * w_int
                out_b += c_b * w_int
                out_tr *= Float32(1.0) - sig_dt
        return (out_r, out_g, out_b, out_tr)

    def step_particles_cpu(
        self,
        mut packed_particles: List[UInt32],
        count: Int,
        dt: Float32,
        gravity: Float32 = -9.8,
        drag: Float32 = 0.85,
        depth_bounce: Bool = True,
        restitution: Float32 = 0.45,
    ):
        """CPU parity mirror of SHADER_GPU_PARTICLES (up to 65,536 particles + screen-space depth bounce)."""
        var n = min(count, min(GPU_PARTICLE_MAX, len(packed_particles) // GPU_PARTICLE_STRIDE_WORDS))
        if n <= 0:
            return
        var damp = max(Float32(0.0), min(Float32(1.0), Float32(1.0) - drag * dt))
        var half_w = Float32(self.width) * Float32(0.5)
        var half_h = Float32(self.height) * Float32(0.5)
        var has_depth = depth_bounce and len(self.depth) == self.width * self.height and self.width > 4 and self.height > 4
        for pid in range(n):
            var b = pid * GPU_PARTICLE_STRIDE_WORDS
            var life0 = bitcast[DType.float32](packed_particles[b + 3])
            if life0 <= Float32(0.0):
                packed_particles[b + 3] = bitcast[DType.uint32](Float32(0.0))
                continue
            var px0 = bitcast[DType.float32](packed_particles[b])
            var py0 = bitcast[DType.float32](packed_particles[b + 1])
            var pz0 = bitcast[DType.float32](packed_particles[b + 2])
            var vx0 = bitcast[DType.float32](packed_particles[b + 4])
            var vy0 = bitcast[DType.float32](packed_particles[b + 5])
            var vz0 = bitcast[DType.float32](packed_particles[b + 6])
            var life1 = max(Float32(0.0), life0 - dt)
            var curl_p = life0 * Float32(0.65) + px0 * Float32(0.053) + pz0 * Float32(0.041)
            var curl_x = _parabolic_sin(curl_p)
            var curl_z = _parabolic_sin(curl_p + Float32(0.25))
            var nvx = vx0 * damp + (self.cloud_wind_x * Float32(0.08) + curl_x * Float32(1.25)) * dt
            var nvy = vy0 * damp + gravity * dt
            var nvz = vz0 * damp + (self.cloud_wind_z * Float32(0.08) + curl_z * Float32(1.25)) * dt
            var npx = px0 + nvx * dt
            var npy = py0 + nvy * dt
            var npz = pz0 + nvz * dt
            if has_depth:
                var rel = Vec3(npx - self.camera.x, npy - self.camera.y, npz - self.camera.z)
                var vz_cam = dot(rel, self.forward)
                if vz_cam > Float32(0.35):
                    var inv_vz = Float32(1.0) / vz_cam
                    var sx = half_w + dot(rel, self.right) * self.focal * inv_vz
                    var sy = half_h - dot(rel, self.up) * self.focal * inv_vz
                    if sx >= Float32(1.0) and sx < Float32(self.width - 2) and sy >= Float32(1.0) and sy < Float32(self.height - 2):
                        var z_b = self.depth[Int(sy) * self.width + Int(sx)]
                        var pen = vz_cam - z_b
                        if z_b < Float32(500.0) and pen > Float32(0.0) and pen < Float32(2.20):
                            var push_out = pen + Float32(0.05)
                            npx -= self.forward.x * push_out
                            npy -= self.forward.y * push_out
                            npz -= self.forward.z * push_out
                            nvx = -nvx * restitution
                            nvy = abs(nvy) * restitution + Float32(0.45)
                            nvz = -nvz * restitution
            packed_particles[b] = bitcast[DType.uint32](npx)
            packed_particles[b + 1] = bitcast[DType.uint32](npy)
            packed_particles[b + 2] = bitcast[DType.uint32](npz)
            packed_particles[b + 3] = bitcast[DType.uint32](life1)
            packed_particles[b + 4] = bitcast[DType.uint32](nvx)
            packed_particles[b + 5] = bitcast[DType.uint32](nvy)
            packed_particles[b + 6] = bitcast[DType.uint32](nvz)

    def _gather_radiance_cascades(self) -> List[RadianceCascadeNode]:
        """Build 2-cascade World-Space Radiance GI nodes from explicit emitters, dynamic lights & glowing scorch decals."""
        var out = List[RadianceCascadeNode]()
        for ri in range(min(MAX_RADIANCE_CASCADES, len(self.cpu_radiance_nodes))):
            out.append(self.cpu_radiance_nodes[ri])
        # Cascade 0 (near-field) & Cascade 1 (wide bounce) harvested from dynamic point/tube lights
        var li = 0
        while li < len(self.cpu_lights) and len(out) < MAX_RADIANCE_CASCADES:
            var pl = self.cpu_lights[li]
            if pl.intensity > Float32(0.05) and pl.radius > Float32(0.5):
                var r_bounce = pl.radius * Float32(1.85)
                var inv_r2 = Float32(1.0) / max(Float32(1.0), r_bounce * r_bounce)
                var e_scale = min(Float32(145.0), pl.intensity * Float32(42.0))
                out.append(
                    RadianceCascadeNode(
                        pl.position,
                        inv_r2,
                        Vec3(pl.color.x * e_scale, pl.color.y * e_scale, pl.color.z * e_scale),
                        Float32(0.15),
                    )
                )
            li += 1
        # Harvest glowing plasma scorch decals (signed_opacity < 0) as warm surface bounce nodes
        var di = 0
        while di < len(self.cpu_decals) and len(out) < MAX_RADIANCE_CASCADES:
            var dc = self.cpu_decals[di]
            if dc.signed_opacity < Float32(-0.05):
                var rad_d = Float32(6.0) / max(Float32(0.05), min(dc.inv_hx, dc.inv_hz))
                var inv_r2 = Float32(1.0) / max(Float32(4.0), rad_d * rad_d)
                var glow = min(Float32(1.2), -dc.signed_opacity) * Float32(0.45)
                out.append(
                    RadianceCascadeNode(
                        dc.center,
                        inv_r2,
                        Vec3(dc.albedo.x * glow, dc.albedo.y * glow, dc.albedo.z * glow),
                        Float32(-0.25),
                    )
                )
            di += 1
        return out^

    def _compute_dynamic_reflection_lobe(self) -> Tuple[Vec3, Vec3]:
        """Harvest a warm/balanced real-time dynamic environment reflection lobe from live scene lights & emitters."""
        var sum_dir = self.cpu_to_key * Float32(0.15)
        var sum_rgb = self.cpu_key_color * Float32(0.12)
        var n_lights = min(16, len(self.cpu_lights))
        for li in range(n_lights):
            var pl = self.cpu_lights[li]
            var rel = (pl.position + pl.seg_delta * Float32(0.5)) - self.camera
            var d = max(Float32(4.0), rel.length())
            var w = min(Float32(1.4), (pl.intensity * max(Float32(2.0), pl.radius)) / (d * Float32(0.85) + Float32(12.0)))
            if w > Float32(0.01):
                sum_dir = sum_dir + (rel * (Float32(1.0) / d)) * w
                sum_rgb = sum_rgb + pl.color * (w * Float32(0.32))
        var n_rad = min(8, len(self.cpu_radiance_nodes))
        for ri in range(n_rad):
            var rn = self.cpu_radiance_nodes[ri]
            var rel_r = rn.position - self.camera
            var dr = max(Float32(6.0), rel_r.length())
            var wr = min(Float32(0.9), Float32(12.0) / (dr * Float32(0.6) + Float32(14.0)))
            sum_dir = sum_dir + (rel_r * (Float32(1.0) / dr)) * wr
            sum_rgb = sum_rgb + rn.radiance * (wr * Float32(0.0022))
        var out_dir = sum_dir.normalized() if sum_dir.length_squared() > Float32(0.0001) else Vec3(0.0, 1.0, 0.0)
        var r_c = min(Float32(1.45), sum_rgb.x)
        var g_c = min(Float32(1.40), sum_rgb.y)
        # Gamut-safe warm/neutral cap so chrome visors & wet decks never clip electric blue
        var b_cap = max(Float32(0.18), max(r_c, g_c) * Float32(1.06))
        var b_c = min(b_cap, min(Float32(1.35), sum_rgb.z))
        return (out_dir, Vec3(r_c, g_c, b_c))

    @staticmethod
    def _sample_packed_bilinear_cpu(
        buf: List[UInt32], w: Int, h: Int, fx: Float32, fy: Float32
    ) -> UInt32:
        var cx = max(Float32(0.0), fx)
        var cy = max(Float32(0.0), fy)
        var max_x = max(0, w - 1)
        var max_y = max(0, h - 1)
        var x0 = min(max_x, max(0, Int(floor(cx))))
        var x1 = min(max_x, x0 + 1)
        var y0 = min(max_y, max(0, Int(floor(cy))))
        var y1 = min(max_y, y0 + 1)
        var wx = max(0, min(256, Int((cx - Float32(x0)) * Float32(256.0))))
        var wy = max(0, min(256, Int((cy - Float32(y0)) * Float32(256.0))))
        var c00 = buf[y0 * w + x0]
        var c10 = buf[y0 * w + x1]
        var c01 = buf[y1 * w + x0]
        var c11 = buf[y1 * w + x1]
        return Self._lerp_packed_argb(
            Self._lerp_packed_argb(c00, c10, wx),
            Self._lerp_packed_argb(c01, c11, wx),
            wy,
        )

    def tsr_upscale_cpu(
        self,
        src_pixels: List[UInt32],
        src_w: Int,
        src_h: Int,
        dst_w: Int,
        dst_h: Int,
        mut out_pixels: List[UInt32],
        sharpness: Float32 = 0.42,
        history_blend: Float32 = 0.0,
        jitter_x: Float32 = 0.0,
        jitter_y: Float32 = 0.0,
    ):
        """CPU parity mirror of V10 SHADER_TSR_UPSCALE (5-Tap Catmull-Rom + YCoCg history clamp)."""
        if src_w < 2 or src_h < 2 or dst_w < 2 or dst_h < 2 or len(src_pixels) < src_w * src_h:
            return
        out_pixels.resize(dst_w * dst_h, 0xFF000000)
        var scale_x = Float32(src_w) / Float32(dst_w)
        var scale_y = Float32(src_h) / Float32(dst_h)
        var cat_w = sharpness * Float32(0.45) + Float32(0.18)
        var use_hist = (
            history_blend > Float32(0.005)
            and dst_w == self.width
            and dst_h == self.height
            and len(self.cpu_history) == dst_w * dst_h
        )
        for y in range(dst_h):
            var v_src = (Float32(y) + Float32(0.5) + jitter_y) * scale_y - Float32(0.5)
            for x in range(dst_w):
                var u_src = (Float32(x) + Float32(0.5) + jitter_x) * scale_x - Float32(0.5)
                var c_ctr = Self._sample_packed_bilinear_cpu(src_pixels, src_w, src_h, u_src, v_src)
                var c_n = Self._sample_packed_bilinear_cpu(src_pixels, src_w, src_h, u_src, v_src - Float32(1.0))
                var c_s = Self._sample_packed_bilinear_cpu(src_pixels, src_w, src_h, u_src, v_src + Float32(1.0))
                var c_w = Self._sample_packed_bilinear_cpu(src_pixels, src_w, src_h, u_src - Float32(1.0), v_src)
                var c_e = Self._sample_packed_bilinear_cpu(src_pixels, src_w, src_h, u_src + Float32(1.0), v_src)
                var resolved = UInt32(0xFF000000)
                for ch in range(3):
                    var sh = UInt32(16 - ch * 8)
                    var cc = Float32((c_ctr >> sh) & 255)
                    var cn = Float32((c_n >> sh) & 255)
                    var cs = Float32((c_s >> sh) & 255)
                    var cw = Float32((c_w >> sh) & 255)
                    var ce = Float32((c_e >> sh) & 255)
                    var lo_c = min(cc, min(min(cn, cs), min(cw, ce)))
                    var hi_c = max(cc, max(max(cn, cs), max(cw, ce)))
                    var avg4 = (cn + cs + cw + ce) * Float32(0.25)
                    var val = max(lo_c, min(hi_c, cc + (cc - avg4) * cat_w))
                    resolved |= UInt32(max(0, min(255, Int(val + Float32(0.5))))) << sh
                if use_hist:
                    var idx = y * dst_w + x
                    var vw = self.cpu_velocity[idx] if idx < len(self.cpu_velocity) else UInt32(0)
                    var vx = (Float32(vw & 65535) - Float32(32768.0)) * Float32(0.015625) if vw != 0 else Float32(0.0)
                    var vy = (Float32(vw >> 16) - Float32(32768.0)) * Float32(0.015625) if vw != 0 else Float32(0.0)
                    var px_h = Float32(x) - vx
                    var py_h = Float32(y) - vy
                    if px_h >= Float32(0.0) and px_h < Float32(dst_w - 1) and py_h >= Float32(0.0) and py_h < Float32(dst_h - 1):
                        var h_col = Self._sample_packed_bilinear_cpu(self.cpu_history, dst_w, dst_h, px_h, py_h)
                        var yc_c = Self._rgb_to_ycocg_f(resolved)
                        var yc_n = Self._rgb_to_ycocg_f(c_n)
                        var yc_s = Self._rgb_to_ycocg_f(c_s)
                        var yc_w = Self._rgb_to_ycocg_f(c_w)
                        var yc_e = Self._rgb_to_ycocg_f(c_e)
                        var lo_y = min(yc_c.x, min(min(yc_n.x, yc_s.x), min(yc_w.x, yc_e.x)))
                        var hi_y = max(yc_c.x, max(max(yc_n.x, yc_s.x), max(yc_w.x, yc_e.x)))
                        var lo_co = min(yc_c.y, min(min(yc_n.y, yc_s.y), min(yc_w.y, yc_e.y)))
                        var hi_co = max(yc_c.y, max(max(yc_n.y, yc_s.y), max(yc_w.y, yc_e.y)))
                        var lo_cg = min(yc_c.z, min(min(yc_n.z, yc_s.z), min(yc_w.z, yc_e.z)))
                        var hi_cg = max(yc_c.z, max(max(yc_n.z, yc_s.z), max(yc_w.z, yc_e.z)))
                        var yc_h = Self._rgb_to_ycocg_f(h_col)
                        var cl_y = max(lo_y, min(hi_y, yc_h.x))
                        var cl_co = max(lo_co, min(hi_co, yc_h.y))
                        var cl_cg = max(lo_cg, min(hi_cg, yc_h.z))
                        var tmp_h = cl_y - cl_cg
                        var hr_cl = max(Float32(0.0), min(Float32(255.0), tmp_h + cl_co))
                        var hg_cl = max(Float32(0.0), min(Float32(255.0), cl_y + cl_cg))
                        var hb_cl = max(Float32(0.0), min(Float32(255.0), tmp_h - cl_co))
                        var spd = sqrt(vx * vx + vy * vy)
                        var eff_b = history_blend / (Float32(1.0) + spd * Float32(0.12))
                        var tr = clamp_byte(Float32((resolved >> 16) & 255) * (Float32(1.0) - eff_b) + hr_cl * eff_b)
                        var tg = clamp_byte(Float32((resolved >> 8) & 255) * (Float32(1.0) - eff_b) + hg_cl * eff_b)
                        var tb = clamp_byte(Float32(resolved & 255) * (Float32(1.0) - eff_b) + hb_cl * eff_b)
                        resolved = rgb(tr, tg, tb)
                out_pixels[y * dst_w + x] = resolved

    def cull_instances_cpu(
        self,
        packed_instances: List[UInt32],
        instance_count: Int,
        mut out_visibility: List[UInt32],
        mut out_indirect_cmds: List[UInt32],
        use_hiz: Bool = True,
    ) -> Int:
        """CPU parity mirror of V10 SHADER_GPU_CULL (Frustum + 16x16 Hi-Z -> Indirect Draw Buffer)."""
        var n = min(instance_count, min(GPU_CULL_MAX_INSTANCES, len(packed_instances) // GPU_CULL_IN_STRIDE_WORDS))
        out_visibility.resize(max(0, n), 0)
        out_indirect_cmds.clear()
        if n <= 0:
            return 0
        var half_w = Float32(self.width) * Float32(0.5)
        var half_h = Float32(self.height) * Float32(0.5)
        var inv_focal = Float32(1.0) / max(Float32(0.0001), self.focal)
        var tan_x = half_w * inv_focal
        var tan_y = half_h * inv_focal
        var sec_x = sqrt(tan_x * tan_x + Float32(1.0))
        var sec_y = sqrt(tan_y * tan_y + Float32(1.0))
        var vis_cnt = 0
        for iid in range(n):
            var b = iid * GPU_CULL_IN_STRIDE_WORDS
            var cx = bitcast[DType.float32](packed_instances[b])
            var cy = bitcast[DType.float32](packed_instances[b + 1])
            var cz = bitcast[DType.float32](packed_instances[b + 2])
            var rad = bitcast[DType.float32](packed_instances[b + 3])
            var idx_cnt = packed_instances[b + 4]
            var first_idx = packed_instances[b + 5]
            var vtx_off = packed_instances[b + 6]
            var flags = packed_instances[b + 7]
            var rel = Vec3(cx - self.camera.x, cy - self.camera.y, cz - self.camera.z)
            var vx = dot(rel, self.right)
            var vy = dot(rel, self.up)
            var vz = dot(rel, self.forward)
            var vis = (
                vz + rad >= Float32(0.20)
                and abs(vx) <= vz * tan_x + rad * sec_x
                and abs(vy) <= vz * tan_y + rad * sec_y
            )
            var near_z = vz - rad
            if vis and use_hiz and (flags & UInt32(1)) != 0 and near_z > Float32(0.35):
                var inv_vz = Float32(1.0) / max(Float32(0.35), vz)
                var sc_x = half_w + vx * self.focal * inv_vz
                var sc_y = half_h - vy * self.focal * inv_vz
                var reach = (rad * self.focal / near_z) * Float32(1.12)
                var min_px = Int(floor(sc_x - reach))
                var max_px = Int(floor(sc_x + reach + Float32(1.0)))
                var min_py = Int(floor(sc_y - reach))
                var max_py = Int(floor(sc_y + reach + Float32(1.0)))
                if max_px >= 0 and min_px < self.width and max_py >= 0 and min_py < self.height:
                    if self.is_rect_occluded_hiz(min_px, min_py, max_px, max_py, Float32(1.0) / near_z):
                        vis = False
            if vis:
                out_visibility[iid] = 1
                out_indirect_cmds.append(idx_cnt)
                out_indirect_cmds.append(1)
                out_indirect_cmds.append(first_idx)
                out_indirect_cmds.append(vtx_off)
                out_indirect_cmds.append(UInt32(iid))
                vis_cnt += 1
            else:
                out_visibility[iid] = 0
        return vis_cnt

    def apply_post(mut self, profile: LightingProfile, quality: Int, reduced_effects: Bool):
        """True Linear Float32 HDR post-compose pipeline on CPU matching Vulkan:
        Refraction -> Depth-Buffer SSR -> CSM/Sphere-Shadowed Volumetric Fog & HG God-Rays ->
        8-tap Affine-1/z SSAO -> Linear HDR Dual-Kawase Bloom -> Post-Compose AgX + Tone LUT ->
        Halo-Free CAS -> YCoCg 3x3 Clamped TAA + 6-Tap Velocity Motion Blur."""
        if self.gpu.active:
            return
        var w = self.width
        var h = self.height
        if w <= 0 or h <= 0:
            return

        # Fold firefly-clamped Linear Float32 HDR peaks (> 210.0) into bloom cells before blurring
        if self.bloom_enabled and self.bloom_width > 0 and self.bloom_height > 0:
            for by in range(self.bloom_height):
                var sy = min(h - 1, (by * h + h // 2) // self.bloom_height)
                for bx in range(self.bloom_width):
                    var sx = min(w - 1, (bx * w + w // 2) // self.bloom_width)
                    var p_idx = (sy * w + sx) * 3
                    var b_idx = (by * self.bloom_width + bx) * 3
                    var er = min(Float32(380.0), max(Float32(0.0), (self.hdr_rgb[p_idx] - Float32(210.0)) * Float32(0.38)))
                    var eg = min(Float32(380.0), max(Float32(0.0), (self.hdr_rgb[p_idx + 1] - Float32(210.0)) * Float32(0.38)))
                    var eb = min(Float32(380.0), max(Float32(0.0), (self.hdr_rgb[p_idx + 2] - Float32(210.0)) * Float32(0.38)))
                    if er + eg + eb > 0.5:
                        self.bloom[b_idx] = max(self.bloom[b_idx], er)
                        self.bloom[b_idx + 1] = max(self.bloom[b_idx + 1], eg)
                        self.bloom[b_idx + 2] = max(self.bloom[b_idx + 2], eb)
                        self.bloom_pixels += 1
        self.blur_bloom()
        var has_bloom = self.bloom_enabled and self.bloom_pixels > 0 and self.bloom_width > 0 and self.bloom_height > 0
        var inv_bloom_str = Float32(1.0) / Float32(max(1, self.bloom_strength))

        if quality < 1 or reduced_effects or w < 8 or h < 8:
            if has_bloom:
                for y in range(h):
                    var by = min(self.bloom_height - 1, y * self.bloom_height // h)
                    for x in range(w):
                        var idx = y * w + x
                        if (self.gbuf_norm_rough[idx] >> 31) == 0:
                            continue
                        var bx = min(self.bloom_width - 1, x * self.bloom_width // w)
                        var b_o = (by * self.bloom_width + bx) * 3
                        var hr = self.hdr_rgb[idx * 3] + self.bloom[b_o] * inv_bloom_str
                        var hg = self.hdr_rgb[idx * 3 + 1] + self.bloom[b_o + 1] * inv_bloom_str
                        var hb = self.hdr_rgb[idx * 3 + 2] + self.bloom[b_o + 2] * inv_bloom_str
                        self.pixels[idx] = self._grade_hdr_pixel(hr, hg, hb)
            return

        var half_w = Float32(w) * 0.5
        var half_h = Float32(h) * 0.5
        var inv_focal = Float32(1.0) / max(Float32(0.0001), self.focal)
        var ssr_on = (self.gpu_effects & EFFECT_SSR) != 0 and self.ssr_strength > 0.001
        var vol_on = (self.gpu_effects & EFFECT_VOLUMETRIC) != 0 and self.vol_density > 0.0005
        var phys_sky_on = (self.gpu_effects & EFFECT_PHYSICAL_SKY) != 0
        var vol_clouds_on = (self.gpu_effects & EFFECT_VOL_CLOUDS) != 0
        var ssgi_on = (self.gpu_effects & EFFECT_SSGI_BENT) != 0
        var chrom_on = (self.gpu_effects & EFFECT_CHROMATIC_REFRACT) != 0
        var rad_gi_on = (self.gpu_effects & EFFECT_RADIANCE_GI) != 0 and self.radiance_gi_intensity > Float32(0.001)
        var dyrl_on = (self.gpu_effects & EFFECT_DYRL_ANIME) != 0
        var cpu_rad_nodes = self._gather_radiance_cascades() if rad_gi_on else List[RadianceCascadeNode]()
        var froxel_on = vol_on and (self.gpu_effects & EFFECT_FROXEL_VOL) != 0
        var gtao_on = (self.gpu_effects & EFFECT_GTAO_CONTACT) != 0
        var shadows_on = (self.gpu_effects & EFFECT_SHADOWS) != 0
        var fog_col = profile.fog_color
        var composed = List[UInt32](length=w * h, fill=self.fog_color)
        var cpu_sky_lut = self._build_cpu_sky_lut() if phys_sky_on else List[Float32]()
        var cam_cut_frox = (
            self.temporal_frames == 0
            or (self.camera - self.prev_camera).length_squared() > Float32(324.0)
            or dot(self.forward, self.prev_forward) < Float32(0.70)
            or abs(self.focal - self.prev_focal) > self.focal * Float32(0.15)
        )
        var cpu_froxel_integ = self._build_cpu_froxel_grid(
            fog_col, Float32(0.0) if cam_cut_frox else Float32(0.80)
        ) if froxel_on else List[Float32]()

        # 64-Bin Log2 Luminance Histogram + 12%..95% Percentile-Trimmed Auto-Exposure
        var ae_scale = Float32(1.0)
        if (self.gpu_effects & EFFECT_AUTO_EXPOSURE) != 0:
            var bins = List[Int](length=64, fill=0)
            var total_hist = 0
            var py_h = 1
            while py_h < h:
                var row_h = py_h * w
                var px_h = 1
                while px_h < w:
                    var idx_h = row_h + px_h
                    if (self.gbuf_norm_rough[idx_h] >> 31) != 0:
                        var p_h = idx_h * 3
                        var lum_h = self.hdr_rgb[p_h] * Float32(0.26) + self.hdr_rgb[p_h + 1] * Float32(0.58) + self.hdr_rgb[p_h + 2] * Float32(0.16)
                        var log_l = max(Float32(0.0), min(Float32(10.0), log2(max(Float32(1.0), lum_h))))
                        var bin_i = max(0, min(63, Int(log_l * Float32(6.399))))
                        bins[bin_i] += 1
                        total_hist += 1
                    px_h += 4
                py_h += 4
            var target_exp = Float32(1.0)
            if total_hist >= 16:
                var lo_cut = Float32(total_hist) * Float32(0.12)
                var hi_cut = Float32(total_hist) * Float32(0.95)
                var accum_f = Float32(0.0)
                var w_sum = Float32(0.0)
                var k_sum = Float32(0.0)
                for bj in range(64):
                    var prev_a = accum_f
                    var next_a = prev_a + Float32(bins[bj])
                    accum_f = next_a
                    var keep = max(Float32(0.0), min(next_a, hi_cut) - max(prev_a, lo_cut))
                    w_sum += (Float32(bj) + Float32(0.5)) * keep
                    k_sum += keep
                var mean_bin = w_sum / max(Float32(1.0), k_sum)
                var avg_luma = exp(mean_bin * Float32(0.15625 * 0.69314718))
                var raw_exp = Float32(1.0)
                if self.auto_exp_target > Float32(0.1):
                    raw_exp = self.auto_exp_target / max(Float32(1.0), avg_luma)
                elif avg_luma > Float32(96.0):
                    raw_exp = Float32(96.0) / avg_luma
                elif avg_luma < Float32(34.0):
                    raw_exp = Float32(34.0) / max(Float32(4.0), avg_luma)
                target_exp = max(self.auto_exp_min, min(self.auto_exp_max, raw_exp))
            var sp_ae = max(Float32(0.02), min(Float32(1.0), self.auto_exp_speed))
            self.adapted_exposure = max(
                self.auto_exp_min,
                min(self.auto_exp_max, self.adapted_exposure + (target_exp - self.adapted_exposure) * sp_ae),
            )
            ae_scale = self.adapted_exposure

        for y in range(h):
            var ym2 = max(0, y - 2)
            var yp2 = min(h - 1, y + 2)
            var row = y * w
            var by_f = max(Float32(0.0), (Float32(y) + 0.5) * Float32(self.bloom_height) / Float32(h) - 0.5) if has_bloom else Float32(0.0)
            var by0 = min(max(0, self.bloom_height - 1), Int(by_f))
            var by1 = min(max(0, self.bloom_height - 1), by0 + 1)
            var bty = by_f - Float32(by0)
            var vy_cam = (half_h - (Float32(y) + 0.5)) * inv_focal

            for x in range(w):
                var idx = row + x
                var gbuf = self.gbuf_norm_rough[idx]
                var ref_pk = self.cpu_refract[idx]
                var vx_cam = (Float32(x) + 0.5 - half_w) * inv_focal
                var ray = self.forward + self.right * vx_cam + self.up * vy_cam
                var ign_in = Float32(x) * Float32(0.06711056) + Float32(y) * Float32(0.00583715)
                var ign_fr = ign_in - floor(ign_in)
                var ign = Float32(52.9829189) * ign_fr
                ign = ign - floor(ign)
                var wx_sky_e = Float32(0.0)
                if (self.gpu_effects & EFFECT_WEATHER_PRECIP) != 0 and self.weather_intensity > Float32(0.01):
                    var uv_x0 = Float32(x) / Float32(max(1, w))
                    var uv_y0 = Float32(y) / Float32(max(1, h))
                    var slant0 = max(Float32(-0.35), min(Float32(0.35), self.weather_wind_x * Float32(0.012)))
                    var is_snow0 = self.weather_kind == 2
                    var fall_spd0 = Float32(0.85) if is_snow0 else Float32(2.90)
                    var freq_x0 = Float32(38.0) if is_snow0 else Float32(86.0)
                    var freq_y0 = Float32(22.0) if is_snow0 else Float32(9.5)
                    var u_wx0 = (uv_x0 + uv_y0 * slant0) * freq_x0
                    var cell_wx0 = floor(u_wx0)
                    var fu_wx0 = u_wx0 - cell_wx0
                    var h_in0 = (cell_wx0 + Float32(17.31)) * Float32(0.1031)
                    var cell_h0 = h_in0 - floor(h_in0)
                    var v_wx0 = (uv_y0 - self.water_time * fall_spd0) * freq_y0 + cell_h0 * Float32(13.7)
                    var fv_wx0 = v_wx0 - floor(v_wx0)
                    var dx_wx0 = abs(fu_wx0 - (Float32(0.25) + cell_h0 * Float32(0.5)))
                    var streak_w0 = Float32(0.16) if is_snow0 else Float32(0.075)
                    var mx_wx0 = max(Float32(0.0), min(Float32(1.0), (streak_w0 - dx_wx0) * Float32(16.0)))
                    var my_wx0 = max(Float32(0.0), min(Float32(1.0), (Float32(0.42) - fv_wx0) * Float32(3.2)))
                    wx_sky_e = mx_wx0 * my_wx0 * self.weather_intensity * Float32(56.0)

                if (gbuf >> 31) == 0 and ref_pk == 0:
                    if phys_sky_on or vol_clouds_on or wx_sky_e > Float32(0.01):
                        var sky_dir = ray.normalized()
                        var sky_hdr = Vec3(
                            Float32((self.fog_color >> 16) & 255),
                            Float32((self.fog_color >> 8) & 255),
                            Float32(self.fog_color & 255),
                        )
                        if phys_sky_on:
                            sky_hdr = self._sample_cpu_sky_lut(cpu_sky_lut, sky_dir)
                        if vol_clouds_on:
                            var cl_res = self._integrate_volumetric_clouds_cpu(sky_dir, Float32(9000.0), ign)
                            sky_hdr = Vec3(
                                sky_hdr.x * cl_res[3] + cl_res[0],
                                sky_hdr.y * cl_res[3] + cl_res[1],
                                sky_hdr.z * cl_res[3] + cl_res[2],
                            )
                        sky_hdr = Vec3(
                            sky_hdr.x + wx_sky_e,
                            sky_hdr.y + wx_sky_e * Float32(0.97),
                            sky_hdr.z + wx_sky_e * Float32(0.90),
                        )
                        var bl_sky_r = Float32(0.0)
                        var bl_sky_g = Float32(0.0)
                        var bl_sky_b = Float32(0.0)
                        if has_bloom:
                            var bx_f = max(Float32(0.0), (Float32(x) + 0.5) * Float32(self.bloom_width) / Float32(w) - 0.5)
                            var bx0 = min(self.bloom_width - 1, Int(bx_f))
                            var bx1 = min(self.bloom_width - 1, bx0 + 1)
                            var btx = bx_f - Float32(bx0)
                            var i00 = (by0 * self.bloom_width + bx0) * 3
                            var i10 = (by0 * self.bloom_width + bx1) * 3
                            var i01 = (by1 * self.bloom_width + bx0) * 3
                            var i11 = (by1 * self.bloom_width + bx1) * 3
                            bl_sky_r = ((self.bloom[i00] + (self.bloom[i10] - self.bloom[i00]) * btx) * (1.0 - bty) + (self.bloom[i01] + (self.bloom[i11] - self.bloom[i01]) * btx) * bty) * inv_bloom_str
                            bl_sky_g = ((self.bloom[i00 + 1] + (self.bloom[i10 + 1] - self.bloom[i00 + 1]) * btx) * (1.0 - bty) + (self.bloom[i01 + 1] + (self.bloom[i11 + 1] - self.bloom[i01 + 1]) * btx) * bty) * inv_bloom_str
                            bl_sky_b = ((self.bloom[i00 + 2] + (self.bloom[i10 + 2] - self.bloom[i00 + 2]) * btx) * (1.0 - bty) + (self.bloom[i01 + 2] + (self.bloom[i11 + 2] - self.bloom[i01 + 2]) * btx) * bty) * inv_bloom_str
                        composed[idx] = self._grade_hdr_pixel(
                            (sky_hdr.x + bl_sky_r) * ae_scale,
                            (sky_hdr.y + bl_sky_g) * ae_scale,
                            (sky_hdr.z + bl_sky_b) * ae_scale,
                            (ign - 0.5) * 0.55,
                        )
                    else:
                        composed[idx] = self.fog_color
                    continue

                var sample_idx = idx
                var sample_idx_r = idx
                var sample_idx_b = idx
                if (ref_pk >> 16) != 0:
                    var rdx_f = Float32(Int(ref_pk & 255) - 128)
                    var rdy_f = Float32(Int((ref_pk >> 8) & 255) - 128)
                    if chrom_on:
                        rdx_f += _parabolic_sin(Float32(y) * Float32(0.018) + self.water_time * Float32(2.4)) * self.heat_haze_amp
                        rdy_f += _parabolic_sin(Float32(x) * Float32(0.015) + self.water_time * Float32(1.9)) * self.heat_haze_amp
                    var rx = max(0, min(w - 1, x + Int(rdx_f)))
                    var ry = max(0, min(h - 1, y + Int(rdy_f)))
                    var cand_g = ry * w + rx
                    if self.depth[cand_g] * Float32(1.15) >= self.depth[idx]:
                        sample_idx = cand_g
                        sample_idx_r = cand_g
                        sample_idx_b = cand_g
                    if chrom_on:
                        var r_mul = Float32(1.0) - self.dispersion_strength
                        var b_mul = Float32(1.0) + self.dispersion_strength
                        var cand_r = max(0, min(h - 1, y + Int(rdy_f * r_mul))) * w + max(0, min(w - 1, x + Int(rdx_f * r_mul)))
                        var cand_b = max(0, min(h - 1, y + Int(rdy_f * b_mul))) * w + max(0, min(w - 1, x + Int(rdx_f * b_mul)))
                        if self.depth[cand_r] * Float32(1.15) >= self.depth[idx]:
                            sample_idx_r = cand_r
                        if self.depth[cand_b] * Float32(1.15) >= self.depth[idx]:
                            sample_idx_b = cand_b

                var cur_r = self.hdr_rgb[sample_idx_r * 3]
                var cur_g = self.hdr_rgb[sample_idx * 3 + 1]
                var cur_b = self.hdr_rgb[sample_idx_b * 3 + 2]
                var d_here = self.depth[idx]

                # 0. Draw-order-independent CPU 2-Cascade Poisson PCF Shadow Evaluation in Linear HDR
                if self.csm_has_casters and (self.gpu_effects & EFFECT_CSM) != 0 and d_here > Float32(0.05) and d_here < Float32(500.0):
                    var csm_n = Self._decode_oct_normal(gbuf)
                    if dot(csm_n, self.cpu_to_key) > Float32(0.0):
                        var csm_vis = self.cpu_csm_shadow(self.camera + ray * d_here, csm_n, self.cpu_to_key, x, y)
                        if csm_vis < Float32(0.999):
                            var sh_atten = Float32(0.32) + Float32(0.68) * csm_vis
                            cur_r *= sh_atten
                            cur_g *= sh_atten
                            cur_b *= sh_atten

                # 1. CPU Depth-Buffer Screen-Space Reflections (SSR) in Linear HDR
                var ssr_refl_u = Int((gbuf >> 24) & 127)
                if ssr_on and ssr_refl_u > 2 and d_here > Float32(0.05) and d_here < Float32(500.0):
                    var world_p = self.camera + ray * d_here
                    var surf_n = Self._decode_oct_normal(gbuf)
                    var inc_dir = ray.normalized()
                    var r_dir = inc_dir - surf_n * (Float32(2.0) * dot(inc_dir, surf_n))
                    if dot(r_dir, self.forward) > Float32(-0.20):
                        var step_dt = max(Float32(0.55), d_here * Float32(0.045))
                        var march_t = step_dt * (Float32(0.75) + ign * Float32(0.55))
                        var prev_t = Float32(0.0)
                        for _ in range(10):
                            var q = world_p + r_dir * march_t
                            var rel_q = q - self.camera
                            var q_vz = dot(rel_q, self.forward)
                            if q_vz < Float32(0.25):
                                break
                            var sx_f = half_w + dot(rel_q, self.right) * self.focal / q_vz
                            var sy_f = half_h - dot(rel_q, self.up) * self.focal / q_vz
                            if sx_f < 2.0 or sx_f >= Float32(w - 2) or sy_f < 2.0 or sy_f >= Float32(h - 2):
                                break
                            var spx = Int(sy_f) * w + Int(sx_f)
                            var z_b = self.depth[spx]
                            var diff = q_vz - z_b
                            var thick = max(Float32(1.20), march_t * Float32(0.35))
                            if z_b < Float32(625.0) and diff > 0.0 and diff < thick:
                                var t_mid = (prev_t + march_t) * 0.5
                                var mq = world_p + r_dir * t_mid
                                var rel_m = mq - self.camera
                                var m_vz = max(Float32(0.25), dot(rel_m, self.forward))
                                var msx = max(1, min(w - 2, Int(half_w + dot(rel_m, self.right) * self.focal / m_vz)))
                                var msy = max(1, min(h - 2, Int(half_h - dot(rel_m, self.up) * self.focal / m_vz)))
                                var mpx = msy * w + msx
                                var hpx = mpx if m_vz > self.depth[mpx] else spx
                                var fx_n = abs(sx_f / Float32(w) * 2.0 - 1.0)
                                var fy_n = abs(sy_f / Float32(h) * 2.0 - 1.0)
                                var edge_v = max(Float32(0.0), min(Float32(1.0), (Float32(1.0) - max(fx_n, fy_n)) * Float32(4.5)))
                                var w_ssr = min(Float32(0.65), (Float32(ssr_refl_u) / Float32(127.0)) * edge_v * self.ssr_strength)
                                var keep_b = Float32(1.0) - w_ssr * Float32(0.45)
                                cur_r = cur_r * keep_b + min(Float32(340.0), self.hdr_rgb[hpx * 3]) * w_ssr
                                cur_g = cur_g * keep_b + min(Float32(340.0), self.hdr_rgb[hpx * 3 + 1]) * w_ssr
                                cur_b = cur_b * keep_b + min(Float32(340.0), self.hdr_rgb[hpx * 3 + 2]) * w_ssr
                                break
                            prev_t = march_t
                            step_dt *= Float32(1.38)
                            march_t += step_dt

                # 2. CPU Volumetric Fog & CSM/Sphere-Shadowed Henyey-Greenstein Crepuscular Light Shafts
                if vol_on:
                    var ray_len = ray.length()
                    var ray_dir = ray * (Float32(1.0) / max(Float32(0.0001), ray_len))
                    if froxel_on:
                        var frox_max = max(Float32(10.0), self.froxel_max_dist)
                        var surf_w_dist = min(frox_max, min(d_here, Float32(5000.0)) * ray_len)
                        var fu_c = max(Float32(0.0), min(Float32(FROXEL_X - 1), ((Float32(x) + 0.5) / Float32(w)) * Float32(FROXEL_X) - 0.5))
                        var fv_c = max(Float32(0.0), min(Float32(FROXEL_Y - 1), ((Float32(y) + 0.5) / Float32(h)) * Float32(FROXEL_Y) - 0.5))
                        var fz_norm = sqrt(max(Float32(0.0), min(Float32(1.0), surf_w_dist / frox_max)))
                        var fz_c = max(Float32(0.0), min(Float32(FROXEL_Z - 1), fz_norm * Float32(FROXEL_Z) - 0.5))
                        var frx = Self._sample_cpu_froxel_trilinear(cpu_froxel_integ, fu_c, fv_c, fz_c)
                        cur_r = cur_r * frx[3] + frx[0]
                        cur_g = cur_g * frx[3] + frx[1]
                        cur_b = cur_b * frx[3] + frx[2]
                    else:
                        var max_dist = min(Float32(175.0), min(d_here, Float32(500.0)) * ray_len)
                        var cos_th = dot(ray_dir, self.cpu_to_key)
                        var hg_den = Float32(1.4096) - Float32(1.28) * cos_th
                        var hg_phase = min(Float32(1.55), Float32(0.04698) / max(Float32(0.02), hg_den * sqrt(max(Float32(0.01), hg_den))))
                        var sun_r = self.cpu_key_color.x * hg_phase * Float32(195.0)
                        var sun_g = self.cpu_key_color.y * hg_phase * Float32(190.0)
                        var sun_b = self.cpu_key_color.z * hg_phase * Float32(172.0)
                        var trans = Float32(1.0)
                        var scat_r = Float32(0.0)
                        var scat_g = Float32(0.0)
                        var scat_b = Float32(0.0)
                        var n_vlights = min(4, len(self.cpu_lights))
                        for step_i in range(6):
                            var u_s = (Float32(step_i) + ign) * Float32(1.0 / 6.0)
                            var dist_s = max_dist * (u_s * u_s + u_s * Float32(0.05))
                            var dt_s = max_dist * (u_s * Float32(0.31) + Float32(0.01))
                            var px_s = self.camera + ray_dir * dist_s
                            var alt_h = max(Float32(0.0), px_s.y + Float32(6.0))
                            var rho = exp(-alt_h * self.vol_height_falloff * Float32(0.69314718)) * (self.vol_density * Float32(0.0036))
                            var ext = min(Float32(0.25), rho * dt_s)
                            var vis_s = self.cpu_csm_shadow_1tap(px_s, self.cpu_to_key) * self.cpu_sphere_shadow(px_s, self.cpu_to_key)
                            var loc_r = Float32(0.0)
                            var loc_g = Float32(0.0)
                            var loc_b = Float32(0.0)
                            for li in range(n_vlights):
                                var pl = self.cpu_lights[li]
                                var d2 = (pl.position - px_s).length_squared() + Float32(1.5)
                                var r2 = max(Float32(0.04), pl.radius * pl.radius)
                                if d2 < r2:
                                    var fall = Float32(1.0) - d2 / r2
                                    var l_att = fall * fall * pl.intensity * Float32(460.0)
                                    loc_r += pl.color.x * l_att
                                    loc_g += pl.color.y * l_att
                                    loc_b += pl.color.z * l_att
                            var w_step = trans * ext
                            scat_r += (fog_col.x * Float32(0.62) + sun_r * vis_s + loc_r) * w_step
                            scat_g += (fog_col.y * Float32(0.62) + sun_g * vis_s + loc_g) * w_step
                            scat_b += (fog_col.z * Float32(0.62) + sun_b * vis_s + loc_b) * w_step
                            trans *= Float32(1.0) - ext
                        cur_r = cur_r * trans + scat_r
                        cur_g = cur_g * trans + scat_g
                        cur_b = cur_b * trans + scat_b

                # 2b. Altitude-aware Hillaire Aerial Perspective on distant terrain/hulls when EFFECT_PHYSICAL_SKY is enabled
                if phys_sky_on and self.aerial_strength > Float32(0.001):
                    var dens_ap = exp(-max(Float32(0.0), self.cam_altitude_km) * Float32(0.125))
                    var aer_t = max(Float32(0.0), min(Float32(0.58), (Float32(1.0) - exp(-d_here * Float32(0.00125))) * self.aerial_strength * dens_ap))
                    if aer_t > Float32(0.004):
                        var ap_sky = self._sample_cpu_sky_lut(cpu_sky_lut, ray.normalized())
                        cur_r += (ap_sky.x - cur_r) * aer_t
                        cur_g += (ap_sky.y - cur_g) * aer_t
                        cur_b += (ap_sky.z - cur_b) * aer_t

                # 2c. V9 Ray-Marched 3D Volumetric Cloudscapes in front of high terrain/distant hulls
                if vol_clouds_on:
                    var ray_len_cl = ray.length()
                    var dir_cl = ray * (Float32(1.0) / max(Float32(0.0001), ray_len_cl))
                    var cl_g = self._integrate_volumetric_clouds_cpu(dir_cl, d_here * ray_len_cl, ign)
                    cur_r = cur_r * cl_g[3] + cl_g[0]
                    cur_g = cur_g * cl_g[3] + cl_g[1]
                    cur_b = cur_b * cl_g[3] + cl_g[2]

                # 3. Horizon-Based GTAO + V9 SSGI & Bent Normals + 8-Step Contact Shadows (or fallback SSAO)
                var ao_mult = Float32(1.0)
                if d_here > Float32(0.05) and d_here < Float32(625.0):
                    if gtao_on:
                        var surf_n_g = Self._decode_oct_normal(gbuf)
                        var n_rx = dot(surf_n_g, self.right)
                        var n_uy = dot(surf_n_g, self.up)
                        var n_fz = dot(surf_n_g, self.forward)
                        var cp_x = vx_cam * d_here
                        var cp_y = vy_cam * d_here
                        var rot_ph = ign * Float32(0.5)
                        var s_g = _parabolic_sin(rot_ph)
                        var c_g = _parabolic_sin(rot_ph + Float32(0.25))
                        var max_r2 = max(Float32(0.38), d_here * d_here * Float32(0.038))
                        var inv_max_r2 = Float32(1.0) / max_r2
                        var h_sum = Float32(0.0)
                        var bent_x = n_rx
                        var bent_y = n_uy
                        var bent_z = n_fz
                        var ssgi_r = Float32(0.0)
                        var ssgi_g = Float32(0.0)
                        var ssgi_b = Float32(0.0)
                        var px_cx_g = Float32(x) + Float32(0.5)
                        var px_cy_g = Float32(y) + Float32(0.5)
                        for d_idx in range(4):
                            var dx_dir = c_g if d_idx == 0 else (-s_g if d_idx == 1 else (-c_g if d_idx == 2 else s_g))
                            var dy_dir = s_g if d_idx == 0 else (c_g if d_idx == 1 else (-s_g if d_idx == 2 else -c_g))
                            var max_h = Float32(0.09)
                            for r_step in range(2):
                                var rad_px = Float32(2.6) if r_step == 0 else Float32(6.8)
                                var sx_g = max(Float32(0.5), min(Float32(w) - Float32(0.5), px_cx_g + dx_dir * rad_px))
                                var sy_g = max(Float32(0.5), min(Float32(h) - Float32(0.5), px_cy_g + dy_dir * rad_px))
                                var spx_g = Int(sy_g) * w + Int(sx_g)
                                var d_s = self.depth[spx_g]
                                if d_s > Float32(0.05) and d_s < Float32(625.0):
                                    var vx_s = (sx_g - half_w) * inv_focal
                                    var vy_s = (half_h - sy_g) * inv_focal
                                    var dv_x = vx_s * d_s - cp_x
                                    var dv_y = vy_s * d_s - cp_y
                                    var dv_z = d_s - d_here
                                    var dist2 = dv_x * dv_x + dv_y * dv_y + dv_z * dv_z + Float32(0.0001)
                                    var inv_d = Float32(1.0) / sqrt(dist2)
                                    var sin_h = (dv_x * n_rx + dv_y * n_uy + dv_z * n_fz) * inv_d
                                    var rng_w = max(Float32(0.0), min(Float32(1.0), Float32(1.0) - dist2 * inv_max_r2))
                                    var h_val = sin_h * rng_w
                                    max_h = max(max_h, h_val)
                                    if ssgi_on and h_val > Float32(0.10):
                                        var push_b = h_val * inv_d * Float32(0.35)
                                        bent_x -= dv_x * push_b
                                        bent_y -= dv_y * push_b
                                        bent_z -= dv_z * push_b
                                        var gi_w = min(Float32(0.85), h_val) * Float32(0.125)
                                        ssgi_r += min(Float32(240.0), self.hdr_rgb[spx_g * 3]) * gi_w
                                        ssgi_g += min(Float32(240.0), self.hdr_rgb[spx_g * 3 + 1]) * gi_w
                                        ssgi_b += min(Float32(240.0), self.hdr_rgb[spx_g * 3 + 2]) * gi_w
                            h_sum += max_h - Float32(0.09)
                        if ssgi_on:
                            var inv_bl = Float32(1.0) / sqrt(max(Float32(0.0001), bent_x * bent_x + bent_y * bent_y + bent_z * bent_z))
                            var bent_world_y = (bent_x * self.right.y + bent_y * self.up.y + bent_z * self.forward.y) * inv_bl
                            var bent_sky = Float32(1.0) + (bent_world_y - surf_n_g.y) * Float32(0.12)
                            cur_r = cur_r * bent_sky + ssgi_r * self.ssgi_intensity
                            cur_g = cur_g * bent_sky + ssgi_g * self.ssgi_intensity
                            cur_b = cur_b * bent_sky + ssgi_b * self.ssgi_intensity
                        var gtao_occ = max(Float32(0.0), min(Float32(0.34), h_sum * Float32(0.115)))
                        var cs_atten = Float32(1.0)
                        if shadows_on and d_here < Float32(95.0) and dot(surf_n_g, self.cpu_to_key) > Float32(0.04):
                            var lk_x = dot(self.cpu_to_key, self.right)
                            var lk_y = dot(self.cpu_to_key, self.up)
                            var lk_z = dot(self.cpu_to_key, self.forward)
                            var step_base = Float32(0.135) * max(Float32(0.35), min(Float32(1.35), d_here * Float32(0.055)))
                            var min_bias = Float32(0.014) + d_here * Float32(0.0024)
                            for ci in range(8):
                                var t_cs = (Float32(ci) + Float32(0.35) + ign * Float32(0.65)) * step_base
                                var rz_c = d_here + lk_z * t_cs
                                if rz_c > Float32(0.20):
                                    var sx_c = half_w + (cp_x + lk_x * t_cs) * self.focal / rz_c
                                    var sy_c = half_h - (cp_y + lk_y * t_cs) * self.focal / rz_c
                                    if sx_c >= Float32(1.0) and sx_c < Float32(w - 1) and sy_c >= Float32(1.0) and sy_c < Float32(h - 1):
                                        var bz_cs = self.depth[Int(sy_c) * w + Int(sx_c)]
                                        var dz_cs = rz_c - bz_cs
                                        if bz_cs < Float32(625.0) and dz_cs > min_bias and dz_cs < Float32(0.045) + t_cs * Float32(0.28):
                                            cs_atten = Float32(0.70)
                                            break
                        ao_mult = (Float32(1.0) - gtao_occ) * cs_atten
                    else:
                        var z_c = Float32(1.0) / d_here
                        var xm2 = max(0, x - 2)
                        var xp2 = min(w - 1, x + 2)
                        var d_xp = self.depth[row + xp2]
                        var d_xn = self.depth[row + xm2]
                        var d_yp = self.depth[yp2 * w + x]
                        var d_yn = self.depth[ym2 * w + x]
                        var occ_sum = Float32(0.0)
                        if d_xp > Float32(0.05) and d_xn > Float32(0.05):
                            var zp = Float32(1.0) / d_xp
                            var zn = Float32(1.0) / d_xn
                            if max(abs(zp - z_c), abs(zn - z_c)) < z_c * Float32(0.14):
                                var concav = ((zp + zn) * Float32(0.5) - z_c) / z_c
                                occ_sum += max(Float32(0.0), min(Float32(0.14), (concav - Float32(0.0014)) * Float32(13.5)))
                        if d_yp > Float32(0.05) and d_yn > Float32(0.05):
                            var zp = Float32(1.0) / d_yp
                            var zn = Float32(1.0) / d_yn
                            if max(abs(zp - z_c), abs(zn - z_c)) < z_c * Float32(0.14):
                                var concav = ((zp + zn) * Float32(0.5) - z_c) / z_c
                                occ_sum += max(Float32(0.0), min(Float32(0.14), (concav - Float32(0.0014)) * Float32(13.5)))
                        ao_mult = Float32(1.0) - min(Float32(0.32), occ_sum)

                var bl_r = Float32(0.0)
                var bl_g = Float32(0.0)
                var bl_b = Float32(0.0)
                if has_bloom:
                    var bx_f = max(Float32(0.0), (Float32(x) + 0.5) * Float32(self.bloom_width) / Float32(w) - 0.5)
                    var bx0 = min(self.bloom_width - 1, Int(bx_f))
                    var bx1 = min(self.bloom_width - 1, bx0 + 1)
                    var btx = bx_f - Float32(bx0)
                    var i00 = (by0 * self.bloom_width + bx0) * 3
                    var i10 = (by0 * self.bloom_width + bx1) * 3
                    var i01 = (by1 * self.bloom_width + bx0) * 3
                    var i11 = (by1 * self.bloom_width + bx1) * 3
                    bl_r = ((self.bloom[i00] + (self.bloom[i10] - self.bloom[i00]) * btx) * (1.0 - bty) + (self.bloom[i01] + (self.bloom[i11] - self.bloom[i01]) * btx) * bty) * inv_bloom_str
                    bl_g = ((self.bloom[i00 + 1] + (self.bloom[i10 + 1] - self.bloom[i00 + 1]) * btx) * (1.0 - bty) + (self.bloom[i01 + 1] + (self.bloom[i11 + 1] - self.bloom[i01 + 1]) * btx) * bty) * inv_bloom_str
                    bl_b = ((self.bloom[i00 + 2] + (self.bloom[i10 + 2] - self.bloom[i00 + 2]) * btx) * (1.0 - bty) + (self.bloom[i01 + 2] + (self.bloom[i11 + 2] - self.bloom[i01 + 2]) * btx) * bty) * inv_bloom_str

                # 3b. V10 World-Space Radiance Cascades GI (CPU parity with compose_shader)
                if rad_gi_on and d_here > Float32(0.05) and d_here < Float32(625.0):
                    var rgi_wp = self.camera + ray * d_here
                    var rgi_n = Self._decode_oct_normal(gbuf)
                    var up_w = rgi_n.y * Float32(0.5) + Float32(0.5)
                    var gnd_w = Float32(1.0) - up_w
                    var hor_w = max(Float32(0.0), Float32(1.0) - abs(rgi_n.y))
                    var gi_r = (self.cpu_env_sky.x * up_w + self.cpu_env_ground.x * gnd_w + self.cpu_env_horizon.x * hor_w * Float32(0.35)) * Float32(18.0)
                    var gi_g = (self.cpu_env_sky.y * up_w + self.cpu_env_ground.y * gnd_w + self.cpu_env_horizon.y * hor_w * Float32(0.35)) * Float32(18.0)
                    var gi_b = (self.cpu_env_sky.z * up_w + self.cpu_env_ground.z * gnd_w + self.cpu_env_horizon.z * hor_w * Float32(0.35)) * Float32(18.0)
                    for ri in range(len(cpu_rad_nodes)):
                        var rn = cpu_rad_nodes[ri]
                        var dv = rn.position - rgi_wp
                        var d2 = dv.length_squared()
                        var t_rc = Float32(1.0) - d2 * rn.inv_radius2
                        if t_rc > Float32(0.0):
                            var inv_dl = Float32(1.0) / sqrt(d2 + Float32(0.25))
                            var ndl_rc = max(Float32(0.12), dot(dv, rgi_n) * inv_dl * Float32(0.45) + Float32(0.55))
                            var aniso_w = max(Float32(0.25), Float32(1.0) + rgi_n.y * rn.dir_bias_y)
                            var w_rc = t_rc * t_rc * ndl_rc * aniso_w
                            gi_r += rn.radiance.x * w_rc
                            gi_g += rn.radiance.y * w_rc
                            gi_b += rn.radiance.z * w_rc
                    var rgi_w = self.radiance_gi_intensity * (ao_mult * Float32(0.75) + Float32(0.25))
                    cur_r += gi_r * rgi_w
                    cur_g += gi_g * rgi_w
                    cur_b += gi_b * rgi_w

                # 3c. V10 1984 DYRL 35mm Cel-Film / PBR Hybrid Mode (CPU parity with compose_shader)
                if dyrl_on and d_here > Float32(0.05) and d_here < Float32(625.0):
                    var bands = max(Float32(2.0), self.dyrl_cel_bands)
                    var lum_cel = cur_r * Float32(0.2588) + cur_g * Float32(0.5882) + cur_b * Float32(0.1529)
                    var u_cel = lum_cel / (lum_cel + Float32(92.0))
                    var u_sc = u_cel * bands
                    var band_fl = floor(u_sc)
                    var soft_step = max(Float32(0.0), min(Float32(1.0), (u_sc - band_fl - Float32(0.36)) * Float32(3.4)))
                    var q_u = (band_fl + soft_step + Float32(0.18)) / (bands + Float32(0.18))
                    var tgt_lum = (Float32(92.0) * q_u) / max(Float32(0.06), Float32(1.0) - min(Float32(0.90), q_u))
                    var raw_cel_s = max(Float32(0.54), min(Float32(1.38), tgt_lum / max(Float32(1.0), lum_cel)))
                    var cel_scale = Float32(1.0) + (raw_cel_s - Float32(1.0)) * Float32(0.72)
                    cur_r *= cel_scale
                    cur_g *= cel_scale
                    cur_b *= cel_scale

                    var xm1_d = max(0, x - 1)
                    var xp1_d = min(w - 1, x + 1)
                    var ym1_d = max(0, y - 1)
                    var yp1_d = min(h - 1, y + 1)
                    var idx_n = ym1_d * w + x
                    var idx_s = yp1_d * w + x
                    var idx_w = row + xm1_d
                    var idx_e = row + xp1_d
                    var iz_c = Float32(1.0) / d_here
                    var iz_n = Float32(0.0) if self.depth[idx_n] >= Float32(99999999.0) else (Float32(1.0) / max(Float32(0.01), self.depth[idx_n]))
                    var iz_s = Float32(0.0) if self.depth[idx_s] >= Float32(99999999.0) else (Float32(1.0) / max(Float32(0.01), self.depth[idx_s]))
                    var iz_w = Float32(0.0) if self.depth[idx_w] >= Float32(99999999.0) else (Float32(1.0) / max(Float32(0.01), self.depth[idx_w]))
                    var iz_e = Float32(0.0) if self.depth[idx_e] >= Float32(99999999.0) else (Float32(1.0) / max(Float32(0.01), self.depth[idx_e]))
                    var dz_rel = (abs(iz_n - iz_c) + abs(iz_s - iz_c) + abs(iz_w - iz_c) + abs(iz_e - iz_c)) / max(Float32(0.0025), iz_c)
                    var nw_o = Self._decode_oct_normal(self.gbuf_norm_rough[idx_w])
                    var ne_o = Self._decode_oct_normal(self.gbuf_norm_rough[idx_e])
                    var nn_o = Self._decode_oct_normal(self.gbuf_norm_rough[idx_n])
                    var ns_o = Self._decode_oct_normal(self.gbuf_norm_rough[idx_s])
                    var near_fade = max(Float32(0.15), min(Float32(1.0), iz_c * Float32(22.0)))
                    var dn_crease = ((Float32(1.0) - dot(ne_o, nw_o)) + (Float32(1.0) - dot(nn_o, ns_o))) * near_fade
                    var edge_raw = max(Float32(0.0), dz_rel - Float32(0.085)) * Float32(3.8) + max(Float32(0.0), dn_crease - Float32(0.24)) * Float32(0.95)
                    var ink_w = max(Float32(0.0), min(Float32(0.84), edge_raw * self.dyrl_ink_strength))
                    var keep_ink = Float32(1.0) - ink_w
                    cur_r = cur_r * keep_ink + Float32(9.0) * ink_w
                    cur_g = cur_g * keep_ink + Float32(8.0) * ink_w
                    cur_b = cur_b * keep_ink + Float32(10.0) * ink_w

                    var pk_h = (self.hdr_rgb[(row + max(0, x - 3)) * 3] + self.hdr_rgb[(row + min(w - 1, x + 3)) * 3]) * Float32(0.5)
                    var pk_v = (self.hdr_rgb[(max(0, y - 3) * w + x) * 3] + self.hdr_rgb[(min(h - 1, y + 3) * w + x) * 3]) * Float32(0.5)
                    var glint_e = max(Float32(0.0), min(Float32(1.0), (max(pk_h, pk_v) - Float32(235.0)) * Float32(0.008))) * self.dyrl_glint_strength * Float32(68.0)
                    var grain = (ign - Float32(0.5)) * self.dyrl_halation * Float32(7.5)
                    cur_r = max(Float32(0.0), cur_r + glint_e + grain)
                    cur_g = max(Float32(0.0), cur_g + glint_e * Float32(0.95) + grain)
                    cur_b = max(Float32(0.0), cur_b + glint_e * Float32(0.82) + grain * Float32(0.85))
                    bl_r *= Float32(1.0) + self.dyrl_halation * Float32(0.65)
                    bl_g *= Float32(1.0) + self.dyrl_halation * Float32(0.38)
                    bl_b *= Float32(1.0) + self.dyrl_halation * Float32(0.12)

                # 3d. V11 Dynamic Weather Precipitation & Canopy Streaks (EFFECT_WEATHER_PRECIP)
                if wx_sky_e > Float32(0.01):
                    var dep_ok = Float32(1.0) if d_here > Float32(5.5) else Float32(0.25)
                    var str_e = wx_sky_e * dep_ok
                    cur_r += str_e
                    cur_g += str_e * Float32(0.97)
                    cur_b += str_e * Float32(0.90)

                composed[idx] = self._grade_hdr_pixel(
                    (cur_r * ao_mult + bl_r) * ae_scale,
                    (cur_g * ao_mult + bl_g) * ae_scale,
                    (cur_b * ao_mult + bl_b) * ae_scale,
                    (ign - 0.5) * 0.55,
                )

        # 4. Halo-Free 5-Tap CAS + 3x3 YCoCg Neighborhood-Clamped TAA + Velocity Motion Blur
        var tp = self._compute_temporal_params((self.gpu_effects & EFFECT_TAA) != 0)
        var taa_blend = tp[0]
        var mblur_scale = tp[1]
        var next_hist = List[UInt32](length=w * h, fill=0)

        for y in range(h):
            var ym1 = max(0, y - 1)
            var yp1 = min(h - 1, y + 1)
            var row = y * w
            var px_cy = Float32(y) + 0.5
            var vy_c = (half_h - px_cy) * inv_focal
            for x in range(w):
                var idx = row + x
                var xm1 = max(0, x - 1)
                var xp1 = min(w - 1, x + 1)
                var px_c = composed[idx]
                var px_n = composed[ym1 * w + x]
                var px_s = composed[yp1 * w + x]
                var px_w = composed[row + xm1]
                var px_e = composed[row + xp1]
                var lc = (Int((px_c >> 16) & 255) * 54 + Int((px_c >> 8) & 255) * 183 + Int(px_c & 255) * 19) >> 8
                var ln = (Int((px_n >> 16) & 255) * 54 + Int((px_n >> 8) & 255) * 183 + Int(px_n & 255) * 19) >> 8
                var ls = (Int((px_s >> 16) & 255) * 54 + Int((px_s >> 8) & 255) * 183 + Int(px_s & 255) * 19) >> 8
                var lw = (Int((px_w >> 16) & 255) * 54 + Int((px_w >> 8) & 255) * 183 + Int(px_w & 255) * 19) >> 8
                var le = (Int((px_e >> 16) & 255) * 54 + Int((px_e >> 8) & 255) * 183 + Int(px_e & 255) * 19) >> 8
                var max_l = max(lc, max(max(ln, ls), max(lw, le)))
                var min_l = min(lc, min(min(ln, ls), min(lw, le)))
                var head = min(min_l, 255 - max_l)
                var cas_gain = max(Float32(0.0), min(Float32(0.26), Float32(head) / max(Float32(24.0), Float32(max_l)) * Float32(0.40)))
                var resolved = UInt32(0)
                for ch in range(3):
                    var sh = UInt32(16 - ch * 8)
                    var cc = Int((px_c >> sh) & 255)
                    var cn = Int((px_n >> sh) & 255)
                    var cs = Int((px_s >> sh) & 255)
                    var cw = Int((px_w >> sh) & 255)
                    var ce = Int((px_e >> sh) & 255)
                    var lo_f = Float32(min(cc, min(min(cn, cs), min(cw, ce))))
                    var hi_f = Float32(max(cc, max(max(cn, cs), max(cw, ce))))
                    var avg4 = Float32(cn + cs + cw + ce) * Float32(0.25)
                    var crisp = max(lo_f, min(hi_f, Float32(cc) + (Float32(cc) - avg4) * cas_gain))
                    resolved |= UInt32(max(0, min(255, Int(crisp)))) << sh

                var px_cx = Float32(x) + 0.5
                var vx_c = (px_cx - half_w) * inv_focal
                var d_c = min(Float32(10000.0), max(Float32(0.1), self.depth[idx]))
                var world_p = self.camera + (self.forward + self.right * vx_c + self.up * vy_c) * d_c
                var rel_prev = world_p - self.prev_camera
                var prev_vz = max(Float32(0.1), dot(rel_prev, self.prev_forward))
                var cam_prev_sx = half_w + dot(rel_prev, self.prev_right) * self.prev_focal / prev_vz
                var cam_prev_sy = half_h - dot(rel_prev, self.prev_up) * self.prev_focal / prev_vz
                var vel_word = self.cpu_velocity[idx]
                var has_obj_v = vel_word != 0
                var vel_x = (Float32(vel_word & 65535) - Float32(32768.0)) * Float32(0.015625) if has_obj_v else (px_cx - cam_prev_sx)
                var vel_y = (Float32(vel_word >> 16) - Float32(32768.0)) * Float32(0.015625) if has_obj_v else (px_cy - cam_prev_sy)
                var prev_sx = px_cx - vel_x
                var prev_sy = px_cy - vel_y

                if taa_blend > 0.001 and prev_sx >= 0.5 and prev_sx < Float32(w) - 0.5 and prev_sy >= 0.5 and prev_sy < Float32(h) - 0.5:
                    var hx0 = max(0, min(w - 1, Int(prev_sx - 0.5)))
                    var hy0 = max(0, min(h - 1, Int(prev_sy - 0.5)))
                    var hist_col = self.cpu_history[hy0 * w + hx0]
                    var yc_c = Self._rgb_to_ycocg_f(resolved)
                    var yc_n = Self._rgb_to_ycocg_f(px_n)
                    var yc_s = Self._rgb_to_ycocg_f(px_s)
                    var yc_w = Self._rgb_to_ycocg_f(px_w)
                    var yc_e = Self._rgb_to_ycocg_f(px_e)
                    var lo_y = min(yc_c.x, min(min(yc_n.x, yc_s.x), min(yc_w.x, yc_e.x)))
                    var hi_y = max(yc_c.x, max(max(yc_n.x, yc_s.x), max(yc_w.x, yc_e.x)))
                    var lo_co = min(yc_c.y, min(min(yc_n.y, yc_s.y), min(yc_w.y, yc_e.y)))
                    var hi_co = max(yc_c.y, max(max(yc_n.y, yc_s.y), max(yc_w.y, yc_e.y)))
                    var lo_cg = min(yc_c.z, min(min(yc_n.z, yc_s.z), min(yc_w.z, yc_e.z)))
                    var hi_cg = max(yc_c.z, max(max(yc_n.z, yc_s.z), max(yc_w.z, yc_e.z)))
                    var yc_h = Self._rgb_to_ycocg_f(hist_col)
                    var cl_y = max(lo_y, min(hi_y, yc_h.x))
                    var cl_co = max(lo_co, min(hi_co, yc_h.y))
                    var cl_cg = max(lo_cg, min(hi_cg, yc_h.z))
                    var tmp_h = cl_y - cl_cg
                    var hr_cl = max(Float32(0.0), min(Float32(255.0), tmp_h + cl_co))
                    var hg_cl = max(Float32(0.0), min(Float32(255.0), cl_y + cl_cg))
                    var hb_cl = max(Float32(0.0), min(Float32(255.0), tmp_h - cl_co))
                    var speed = sqrt(vel_x * vel_x + vel_y * vel_y)
                    var eff_b = taa_blend / (Float32(1.0) + speed * Float32(0.09))
                    var tr = clamp_byte(Float32((resolved >> 16) & 255) * (1.0 - eff_b) + hr_cl * eff_b)
                    var tg = clamp_byte(Float32((resolved >> 8) & 255) * (1.0 - eff_b) + hg_cl * eff_b)
                    var tb = clamp_byte(Float32(resolved & 255) * (1.0 - eff_b) + hb_cl * eff_b)
                    resolved = rgb(tr, tg, tb)

                next_hist[idx] = resolved
                var out_px = resolved
                var nx_r = (px_cx - half_w) / max(Float32(1.0), half_w)
                var ny_r = (px_cy - half_h) / max(Float32(1.0), half_h)
                var rad_w = Float32(1.0) if has_obj_v else (Float32(0.25) + min(Float32(1.0), nx_r * nx_r + ny_r * ny_r) * Float32(0.75))
                var bvx = vel_x * mblur_scale * rad_w
                var bvy = vel_y * mblur_scale * rad_w
                var blen = sqrt(bvx * bvx + bvy * bvy)
                if mblur_scale > 0.01 and blen > Float32(1.35):
                    var cap = min(Float32(16.0), blen) / blen
                    bvx *= cap
                    bvy *= cap
                    var sr = Float32((resolved >> 16) & 255)
                    var sg = Float32((resolved >> 8) & 255)
                    var sb = Float32(resolved & 255)
                    var sw = Float32(1.0)
                    for k in range(6):
                        var t_m = Float32(-0.45) + Float32(k) * Float32(0.18)
                        var mx = max(0, min(w - 1, Int(Float32(x) + bvx * t_m)))
                        var my = max(0, min(h - 1, Int(Float32(y) + bvy * t_m)))
                        var m_i = my * w + mx
                        var w_t = Float32(0.15) if self.depth[m_i] * Float32(1.30) < d_c else Float32(0.85)
                        var mc = composed[m_i]
                        sr += Float32((mc >> 16) & 255) * w_t
                        sg += Float32((mc >> 8) & 255) * w_t
                        sb += Float32(mc & 255) * w_t
                        sw += w_t
                    out_px = rgb(clamp_byte(sr / sw), clamp_byte(sg / sw), clamp_byte(sb / sw))

                if (self.gpu_effects & EFFECT_DOF) != 0 and self.dof_aperture > Float32(0.0001):
                    var coc = max(Float32(0.0), min(self.dof_max_coc, self.dof_aperture * abs(d_c - self.dof_focus_dist) / max(Float32(0.5), d_c)))
                    if coc > Float32(0.65):
                        var ign_d_in = Float32(x) * Float32(0.06711056) + Float32(y) * Float32(0.00583715)
                        var ign_d = Float32(52.9829189) * (ign_d_in - floor(ign_d_in))
                        ign_d = ign_d - floor(ign_d)
                        var s_dof = _parabolic_sin(ign_d) * coc
                        var c_dof = _parabolic_sin(ign_d + Float32(0.25)) * coc
                        var dr = Float32((out_px >> 16) & 255)
                        var dg = Float32((out_px >> 8) & 255)
                        var db = Float32(out_px & 255)
                        var dw = Float32(1.0)
                        var vx_taps = [Float32(0.2500), Float32(-0.3191), Float32(0.0501), Float32(0.4150), Float32(-0.7330), Float32(0.6657), Float32(-0.2224), Float32(-0.3910)]
                        var vy_taps = [Float32(0.0000), Float32(0.2923), Float32(-0.5568), Float32(0.5293), Float32(-0.1583), Float32(-0.4945), Float32(0.8752), Float32(-0.8861)]
                        for dtap in range(8):
                            var ox_d = vx_taps[dtap]
                            var oy_d = vy_taps[dtap]
                            var sx_d = max(0, min(w - 1, Int(Float32(x) + ox_d * c_dof - oy_d * s_dof)))
                            var sy_d = max(0, min(h - 1, Int(Float32(y) + ox_d * s_dof + oy_d * c_dof)))
                            var d_idx = sy_d * w + sx_d
                            var t_dep = min(Float32(10000.0), max(Float32(0.1), self.depth[d_idx]))
                            var t_coc = max(Float32(0.0), min(self.dof_max_coc, self.dof_aperture * abs(t_dep - self.dof_focus_dist) / max(Float32(0.5), t_dep)))
                            var fg_guard = max(Float32(0.05), min(Float32(1.0), t_coc * Float32(0.5))) if (t_dep + Float32(0.5) < d_c) else Float32(1.0)
                            var tc = composed[d_idx]
                            dr += Float32((tc >> 16) & 255) * fg_guard
                            dg += Float32((tc >> 8) & 255) * fg_guard
                            db += Float32(tc & 255) * fg_guard
                            dw += fg_guard
                        out_px = rgb(clamp_byte(dr / dw), clamp_byte(dg / dw), clamp_byte(db / dw))
                self.pixels[idx] = out_px

        if (self.gpu_effects & EFFECT_HIZ_CULL) != 0 or self.hiz_cull_enabled:
            self._build_cpu_hiz_pyramid()
        if self.tsr_subnative_active:
            var sub_src = List[UInt32](length=self.render_w * self.render_h, fill=0xFF000000)
            for sy_t in range(self.render_h):
                for sx_t in range(self.render_w):
                    sub_src[sy_t * self.render_w + sx_t] = self.pixels[sy_t * w + sx_t]
            var up_out = List[UInt32]()
            self.tsr_upscale_cpu(
                sub_src,
                self.render_w,
                self.render_h,
                w,
                h,
                up_out,
                sharpness=self.tsr_sharpness,
                history_blend=taa_blend,
            )
            if len(up_out) == w * h:
                self.pixels = up_out.copy()
                next_hist = up_out^
        self.cpu_history = next_hist^
        self._commit_temporal_camera()

    def enable_gpu(mut self, backend: Int = BACKEND_AMDGPU) -> Bool:
        """Hand scene rasterization to the GPU. The CPU path stays in charge on failure."""
        return self.gpu.initialize(self.width, self.height, backend)

    def per_pixel_lighting(self) -> Bool:
        return self.gpu.active and self.gpu_effects != 0

    def mark_light_on_screen(mut self, light: Int, position: Vec3, radius: Float32):
        """Find the tiles a light's sphere can touch. Near the camera that is all of them."""
        var mw = self.render_w if self.tsr_subnative_active else self.width
        var mh = self.render_h if self.tsr_subnative_active else self.height
        var mf = self.render_focal if self.tsr_subnative_active else self.focal
        var relative = position - self.camera
        var depth = dot(relative, self.forward)
        if depth < -radius:
            return
        if depth - radius < 0.25:
            self.gpu.mark_tile_lights(light, 0, 0, mw - 1, mh - 1)
            return
        var scale = mf / (depth - radius)
        var center_x = Float32(mw) * 0.5 + dot(relative, self.right) * mf / depth
        var center_y = Float32(mh) * 0.5 - dot(relative, self.up) * mf / depth
        var reach = radius * scale * 1.1
        self.gpu.mark_tile_lights(light, Int(max(Float32(-1), center_x - reach)), Int(max(Float32(-1), center_y - reach)), Int(min(Float32(mw), center_x + reach)), Int(min(Float32(mh), center_y + reach)))

    def upload_view(mut self):
        """Give the shader what it needs to rebuild each pixel's world position."""
        var vw = self.render_w if self.tsr_subnative_active else self.width
        var vh = self.render_h if self.tsr_subnative_active else self.height
        var vf = self.render_focal if self.tsr_subnative_active else self.focal
        var base = CONSTANT_CAMERA
        var axes = [self.camera, self.right, self.up, self.forward]
        for index in range(4):
            self.gpu.write_float(base + index * 12, axes[index].x)
            self.gpu.write_float(base + index * 12 + 4, axes[index].y)
            self.gpu.write_float(base + index * 12 + 8, axes[index].z)
        self.gpu.write_float(base + 48, Float32(vw) * 0.5)
        self.gpu.write_float(base + 52, Float32(vh) * 0.5)
        self.gpu.write_float(base + 56, 1.0 / vf)

    def submit_gpu_frame(mut self) -> Bool:
        """Finish the HUD layer and queue the frame's GPU passes."""
        self.hud_recording = False
        with self.gpu.profiler.zone(ZONE_HUD_LAYER):
            self.update_hud_layer()
        self.paint_debug()
        self.upload_dither()
        var csm_live = self.gpu.csm_item_count > 0 and (self.gpu_effects & EFFECT_CSM) != 0
        self.gpu.constants.write32((CONSTANT_CSM + 72) // 4, UInt32(1) if csm_live else UInt32(0))
        var shadows_active = self.gpu.voxels_ready or self.shadow_sphere_count > 0 or csm_live
        var eff = self.gpu_effects
        if not shadows_active:
            eff &= ~(EFFECT_SHADOWS | EFFECT_CSM)
        elif not csm_live:
            eff &= ~EFFECT_CSM
        if self.dof_aperture > Float32(0.0001) and self.allow_gpu_effects and (self.gpu_effects & EFFECT_BILINEAR) != 0:
            eff |= EFFECT_DOF
        else:
            eff &= ~EFFECT_DOF

        if self.gpu.backend == BACKEND_VULKAN:
            var prev_cam_snap = self.prev_camera
            var prev_right_snap = self.prev_right
            var prev_up_snap = self.prev_up
            var prev_fwd_snap = self.prev_forward
            var prev_focal_snap = self.prev_focal
            var tp = self._compute_temporal_params((eff & EFFECT_TAA) != 0)
            if tp[0] <= 0.0:
                prev_cam_snap = self.camera
                prev_right_snap = self.right
                prev_up_snap = self.up
                prev_fwd_snap = self.forward
                prev_focal_snap = self.focal
            var pb = CONSTANT_PREV_CAMERA
            self.gpu.write_float(pb + 0, prev_cam_snap.x)
            self.gpu.write_float(pb + 4, prev_cam_snap.y)
            self.gpu.write_float(pb + 8, prev_cam_snap.z)
            self.gpu.write_float(pb + 12, prev_right_snap.x)
            self.gpu.write_float(pb + 16, prev_right_snap.y)
            self.gpu.write_float(pb + 20, prev_right_snap.z)
            self.gpu.write_float(pb + 24, prev_up_snap.x)
            self.gpu.write_float(pb + 28, prev_up_snap.y)
            self.gpu.write_float(pb + 32, prev_up_snap.z)
            self.gpu.write_float(pb + 36, prev_fwd_snap.x)
            self.gpu.write_float(pb + 40, prev_fwd_snap.y)
            self.gpu.write_float(pb + 44, prev_fwd_snap.z)
            self.gpu.write_float(pb + 48, prev_focal_snap)
            self.gpu.write_float(pb + 52, tp[0])
            self.gpu.write_float(pb + 56, tp[1])
            self.gpu.write_float(pb + 60, tp[2])
            self.gpu.write_float(pb + 64, tp[3])
            self.gpu.write_float(pb + 68, self.prev_jitter_x)
            self.gpu.constants.write32((pb + 72) // 4, UInt32(self.history_parity))

            var vb = CONSTANT_VOL_POST
            self.gpu.write_float(vb + 0, Float32((self.fog_color >> 16) & 255))
            self.gpu.write_float(vb + 4, Float32((self.fog_color >> 8) & 255))
            self.gpu.write_float(vb + 8, Float32(self.fog_color & 255))
            self.gpu.write_float(vb + 20, self.vol_height_falloff)
            self.gpu.write_float(vb + 24, self.vol_density)
            self.gpu.write_float(vb + 28, self.ssr_strength)

            var v7 = CONSTANT_V7_POST
            self.gpu.write_float(v7 + 0, self.water_time)
            self.gpu.write_float(v7 + 4, self.dof_focus_dist)
            self.gpu.write_float(v7 + 8, self.dof_aperture)
            self.gpu.write_float(v7 + 12, self.dof_max_coc)
            self.gpu.write_float(v7 + 16, self.auto_exp_speed)
            self.gpu.write_float(v7 + 20, self.auto_exp_min)
            self.gpu.write_float(v7 + 24, self.auto_exp_max)
            self.gpu.write_float(v7 + 28, self.auto_exp_target)

            var v8 = CONSTANT_V8_ATMOS
            var probe_cnt = min(MAX_IRRADIANCE_PROBES, len(self.cpu_probes))
            var decal_cnt = min(MAX_SURFACE_DECALS, len(self.cpu_decals))
            self.gpu.write_float(v8 + 0, self.cam_altitude_km)
            self.gpu.write_float(v8 + 4, self.planet_radius_km)
            self.gpu.write_float(v8 + 8, self.atmos_thickness_km)
            self.gpu.write_float(v8 + 12, self.rayleigh_density)
            self.gpu.write_float(v8 + 16, self.mie_density)
            self.gpu.write_float(v8 + 20, self.ozone_strength)
            self.gpu.write_float(v8 + 24, self.froxel_max_dist)
            self.gpu.write_float(v8 + 28, Float32(0.80) if tp[0] > Float32(0.0) else Float32(0.0))
            self.gpu.constants.write32((v8 + 32) // 4, UInt32(probe_cnt))
            self.gpu.constants.write32((v8 + 36) // 4, UInt32(decal_cnt))
            self.gpu.write_float(v8 + 40, self.aerial_strength)

            for pi in range(probe_cnt):
                var pr = self.cpu_probes[pi]
                var pbase = CONSTANT_PROBES + pi * 32
                self.gpu.write_float(pbase + 0, pr.position.x)
                self.gpu.write_float(pbase + 4, pr.position.y)
                self.gpu.write_float(pbase + 8, pr.position.z)
                self.gpu.write_float(pbase + 12, pr.inv_radius2)
                self.gpu.write_float(pbase + 16, pr.color.x)
                self.gpu.write_float(pbase + 20, pr.color.y)
                self.gpu.write_float(pbase + 24, pr.color.z)
                self.gpu.write_float(pbase + 28, pr.sky_occlusion)

            for di in range(decal_cnt):
                var dc = self.cpu_decals[di]
                var dbase = CONSTANT_DECALS + di * 64
                self.gpu.write_float(dbase + 0, dc.center.x)
                self.gpu.write_float(dbase + 4, dc.center.y)
                self.gpu.write_float(dbase + 8, dc.center.z)
                self.gpu.write_float(dbase + 12, dc.inv_hx)
                self.gpu.write_float(dbase + 16, dc.axis_right.x)
                self.gpu.write_float(dbase + 20, dc.axis_right.y)
                self.gpu.write_float(dbase + 24, dc.axis_right.z)
                self.gpu.write_float(dbase + 28, dc.inv_hy)
                self.gpu.write_float(dbase + 32, dc.axis_forward.x)
                self.gpu.write_float(dbase + 36, dc.axis_forward.y)
                self.gpu.write_float(dbase + 40, dc.axis_forward.z)
                self.gpu.write_float(dbase + 44, dc.inv_hz)
                self.gpu.write_float(dbase + 48, dc.albedo.x)
                self.gpu.write_float(dbase + 52, dc.albedo.y)
                self.gpu.write_float(dbase + 56, dc.albedo.z)
                self.gpu.write_float(dbase + 60, dc.signed_opacity)

            var v9 = CONSTANT_V9_CLOUDS
            self.gpu.write_float(v9 + 0, self.cloud_base_y)
            self.gpu.write_float(v9 + 4, self.cloud_top_y)
            self.gpu.write_float(v9 + 8, self.cloud_coverage)
            self.gpu.write_float(v9 + 12, self.cloud_density)
            self.gpu.write_float(v9 + 16, self.cloud_wind_x)
            self.gpu.write_float(v9 + 20, self.cloud_wind_z)
            self.gpu.write_float(v9 + 24, self.cloud_shadow_strength)
            self.gpu.write_float(v9 + 28, self.cloud_scale)
            self.gpu.write_float(v9 + 32, self.ssgi_intensity)
            self.gpu.write_float(v9 + 36, self.dispersion_strength)
            self.gpu.write_float(v9 + 40, self.heat_haze_amp)
            self.gpu.write_float(v9 + 44, self.water_time)

            var v10 = CONSTANT_V10_POST
            var rnodes = self._gather_radiance_cascades() if (eff & EFFECT_RADIANCE_GI) != 0 else List[RadianceCascadeNode]()
            var rcnt = min(MAX_RADIANCE_CASCADES, len(rnodes))
            self.gpu.write_float(v10 + 0, self.radiance_gi_intensity)
            self.gpu.constants.write32((v10 + 4) // 4, UInt32(rcnt))
            self.gpu.write_float(v10 + 8, self.dyrl_ink_strength)
            self.gpu.write_float(v10 + 12, self.dyrl_cel_bands)
            self.gpu.write_float(v10 + 16, self.dyrl_glint_strength)
            self.gpu.write_float(v10 + 20, self.dyrl_halation)
            self.gpu.write_float(v10 + 24, self.tsr_sharpness)
            self.gpu.write_float(v10 + 28, self.tsr_scale)
            for ri in range(rcnt):
                var rn = rnodes[ri]
                var rbase = CONSTANT_RADIANCE_CASCADES + ri * 32
                self.gpu.write_float(rbase + 0, rn.position.x)
                self.gpu.write_float(rbase + 4, rn.position.y)
                self.gpu.write_float(rbase + 8, rn.position.z)
                self.gpu.write_float(rbase + 12, rn.inv_radius2)
                self.gpu.write_float(rbase + 16, rn.radiance.x)
                self.gpu.write_float(rbase + 20, rn.radiance.y)
                self.gpu.write_float(rbase + 24, rn.radiance.z)
                self.gpu.write_float(rbase + 28, rn.dir_bias_y)

            var v11 = CONSTANT_V11_POST
            var sdf_cnt = min(MAX_SDF_OCCLUDERS, len(self.cpu_sdf_occluders)) if (eff & EFFECT_SDF_SHADOWS) != 0 else 0
            var dyn_lobe = self._compute_dynamic_reflection_lobe()
            self.gpu.constants.write32((v11 + 0) // 4, UInt32(sdf_cnt))
            self.gpu.write_float(v11 + 4, self.sdf_softness_k)
            self.gpu.constants.write32((v11 + 8) // 4, UInt32(max(0, self.weather_kind)))
            self.gpu.write_float(v11 + 12, self.weather_intensity)
            self.gpu.write_float(v11 + 16, self.wetness)
            self.gpu.write_float(v11 + 20, self.weather_wind_x)
            self.gpu.write_float(v11 + 24, self.weather_wind_z)
            self.gpu.write_float(v11 + 28, self.dyn_reflect_intensity)
            self.gpu.write_float(v11 + 32, dyn_lobe[0].x)
            self.gpu.write_float(v11 + 36, dyn_lobe[0].y)
            self.gpu.write_float(v11 + 40, dyn_lobe[0].z)
            self.gpu.write_float(v11 + 44, Float32(4.0))
            self.gpu.write_float(v11 + 48, dyn_lobe[1].x)
            self.gpu.write_float(v11 + 52, dyn_lobe[1].y)
            self.gpu.write_float(v11 + 56, dyn_lobe[1].z)
            self.gpu.constants.write32((v11 + 60) // 4, UInt32(self.render_w if self.tsr_subnative_active else self.width))
            self.gpu.constants.write32((v11 + 64) // 4, UInt32(self.render_h if self.tsr_subnative_active else self.height))
            for si in range(sdf_cnt):
                var so = self.cpu_sdf_occluders[si]
                var sbase = CONSTANT_SDF_OCCLUDERS + si * 32
                self.gpu.write_float(sbase + 0, so.p0.x)
                self.gpu.write_float(sbase + 4, so.p0.y)
                self.gpu.write_float(sbase + 8, so.p0.z)
                self.gpu.write_float(sbase + 12, so.radius)
                self.gpu.write_float(sbase + 16, so.seg.x)
                self.gpu.write_float(sbase + 20, so.seg.y)
                self.gpu.write_float(sbase + 24, so.seg.z)
                self.gpu.write_float(sbase + 28, so.opacity)
            self.gpu.render_width = self.render_w
            self.gpu.render_height = self.render_h

        if not self.gpu.submit(self.tone_lut, self.post_saturation_steps, self.fog_color, self.bloom_enabled, self.bloom_width, self.bloom_height, self.bloom_strength, eff, self.gpu_smoothing):
            print("GPU raster stopped: " + self.gpu.error)
            log_error(LOG_GPU, "GPU raster stopped: " + self.gpu.error)
            return False
        if self.gpu.backend == BACKEND_VULKAN:
            self._commit_temporal_camera()
        return True

    def paint_debug(mut self):
        """Debug shapes go on last. The CPU's frame is redrawn from nothing each
        time; the GPU's HUD layer is not, so what the last frame painted there is
        first put back from the CPU's copy of that layer."""
        if not self.debug.has_work():
            return
        with self.gpu.profiler.zone(ZONE_DEBUG_DRAW):
            var view = DebugView(self.camera, self.right, self.up, self.forward, self.focal, self.width, self.height)
            if self.gpu.active:
                self.restore_hud_pixels()
                self.debug.paint(view, self.gpu.hud.cpu_address, self.gpu.row_pixels, 0xFF000000, True)
            else:
                self.debug.paint(view, Int(self.pixels.unsafe_ptr()), self.width, 0, False)

    def restore_hud_pixels(mut self):
        var layer = Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=self.gpu.hud.cpu_address)
        var clean = self.hud_pixels.unsafe_ptr()
        for packed in self.debug.written:
            var x = Int(packed & 0xFFFF)
            var y = Int(packed >> 16)
            layer[unsafe_offset=y * self.gpu.row_pixels + x] = clean[unsafe_offset=y * self.width + x]
        self.debug.written.clear()

    def read_back_gpu(mut self) -> Bool:
        """Copy the newest finished frame into `pixels`. This waits for the GPU."""
        var read = self.gpu.active and self.gpu.read_back(self.pixels)
        self.stop_recording_if_gpu_lost()
        return read

    def stop_recording_if_gpu_lost(mut self):
        """With the GPU gone no frame will replay the HUD, so draw it directly."""
        if not self.gpu.active:
            self.hud_recording = False

    def record_hud(mut self, kind: Int, a: Int, b: Int, c: Int, d: Int, e: Int, red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8):
        self.hud_ops.append(Int32(kind))
        self.hud_ops.append(Int32(a))
        self.hud_ops.append(Int32(b))
        self.hud_ops.append(Int32(c))
        self.hud_ops.append(Int32(d))
        self.hud_ops.append(Int32(e))
        self.hud_ops.append(Int32(rgb(red, green, blue)))
        self.hud_ops.append(Int32(alpha))

    def recorded_hash(self) -> UInt64:
        var value = UInt64(0xCBF29CE484222325)
        for index in range(len(self.hud_ops)):
            value = (value ^ UInt64(UInt32(self.hud_ops[index]))) * UInt64(0x100000001B3)
        for text in self.hud_texts:
            for byte in text.as_bytes():
                value = (value ^ UInt64(byte)) * UInt64(0x100000001B3)
            # Above any byte, so "AB", "C" cannot hash like "A", "BC".
            value = (value ^ UInt64(0x100)) * UInt64(0x100000001B3)
        return value

    def add_hud_rectangle(mut self, x: Int, y: Int, width: Int, height: Int):
        var left = max(0, x)
        var top = max(0, y)
        var right = min(self.width, x + width)
        var bottom = min(self.height, y + height)
        if right <= left or bottom <= top:
            return
        self.hud_rectangles.append(Int32(left))
        self.hud_rectangles.append(Int32(top))
        self.hud_rectangles.append(Int32(right))
        self.hud_rectangles.append(Int32(bottom))

    def replay_hud(mut self):
        var ops = self.hud_ops.copy()
        var offset = 0
        while offset < len(ops):
            var kind = Int(ops[offset])
            var a = Int(ops[offset + 1])
            var b = Int(ops[offset + 2])
            var c = Int(ops[offset + 3])
            var d = Int(ops[offset + 4])
            var e = Int(ops[offset + 5])
            var color = UInt32(ops[offset + 6])
            var red = UInt8((color >> 16) & 255)
            var green = UInt8((color >> 8) & 255)
            var blue = UInt8(color & 255)
            var alpha = UInt8(Int(ops[offset + 7]))
            if kind == 0:
                self.draw_rectangle(a, b, c, d, red, green, blue, alpha)
            elif kind == 1:
                self.draw_rectangle_rounded(a, b, c, d, Float32(e) / 1000.0, red, green, blue, alpha)
            elif kind == 2:
                self.draw_circle(a, b, Float32(c), red, green, blue, alpha)
            elif kind == 4:
                self.draw_pause_map(a, b)
            elif kind == 8:
                self.draw_minimap(a, b, c, d, alpha)
            elif kind == 5:
                self.draw_backdrop()
            elif kind == 6:
                var rich = self.hud_texts[e]
                _ = self.draw_rich_text(rich, a, b, c, d // 4, d % 4, red, green, blue)
            elif kind == 7:
                self.draw_nine_slice(e % 64, a, b, c, d, alpha, Float32(e // 64) / 256.0)
            else:
                var text = self.hud_texts[e]
                self.draw_text(text, a, b, c, red, green, blue, d % 4, d // 4)
            offset += 8

    def update_hud_layer(mut self):
        """Redraw the premultiplied HUD only when this frame's draw calls changed."""
        var hash = self.recorded_hash()
        if hash == self.hud_hash:
            return
        self.hud_hash = hash
        self.hud_redraws += 1
        if len(self.hud_pixels) != self.width * self.height:
            self.hud_pixels = List[UInt32](length=self.width * self.height, fill=0)
        var previous = self.hud_rectangles.copy()
        self.hud_rectangles.clear()
        for index in range(0, len(previous), 4):
            for y in range(Int(previous[index + 1]), Int(previous[index + 3])):
                _ = external_call["memset", Int](Int(self.hud_pixels.unsafe_ptr()) + (y * self.width + Int(previous[index])) * 4, Int32(0), (Int(previous[index + 2]) - Int(previous[index])) * 4)
        self.hud_drawing = True
        self.replay_hud()
        self.hud_drawing = False
        if not self.gpu.active or self.gpu.hud.size == 0 or self.gpu.hud.cpu_address == 0:
            return
        var source = Int(self.hud_pixels.unsafe_ptr())
        var target = self.gpu.hud.cpu_address
        var changed = previous^
        if changed != self.hud_rectangles:
            changed.extend(self.hud_rectangles.copy())
        var max_w = min(self.width, self.gpu.width)
        var max_h = min(self.height, self.gpu.height)
        for index in range(0, len(changed), 4):
            var left = max(0, Int(changed[index]))
            var right = min(max_w, Int(changed[index + 2]))
            var top = max(0, Int(changed[index + 1]))
            var bottom = min(max_h, Int(changed[index + 3]))
            if right <= left or bottom <= top:
                continue
            var span = (right - left) * 4
            for y in range(top, bottom):
                _ = external_call["memcpy", Int](target + (y * self.gpu.row_pixels + left) * 4, source + (y * self.width + left) * 4, span)

    def fill_gpu_slot(self, slot: Int, a: ScreenVertex, b: ScreenVertex, c: ScreenVertex, offset: Int, texture_width: Int, texture_height: Int, alpha: UInt8, double_sided: Bool):
        """Turn one triangle into a GPU record. Touches only its own slot, so any thread may call it."""
        # The view keeps world +X on screen right, which mirrors winding: faces toward
        # the camera have a positive area here.
        var area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        if abs(area) <= 0.00001 or (not double_sided and area <= 0):
            return
        var rw_lim = self.render_w if self.tsr_subnative_active else self.width
        var rh_lim = self.render_h if self.tsr_subnative_active else self.height
        var minimum_x = max(0, Int(floor(min(a.x, min(b.x, c.x)) - 0.5)))
        var maximum_x = min(rw_lim - 1, Int(floor(max(a.x, max(b.x, c.x)) + 0.5)))
        var minimum_y = max(0, Int(floor(min(a.y, min(b.y, c.y)) - 0.5)))
        var maximum_y = min(rh_lim - 1, Int(floor(max(a.y, max(b.y, c.y)) + 0.5)))
        if minimum_x > maximum_x or minimum_y > maximum_y:
            return
        var inverse_area = 1.0 / area
        var step_a_x = (b.y - c.y) * inverse_area
        var step_b_x = (c.y - a.y) * inverse_area
        var step_a_y = (c.x - b.x) * inverse_area
        var step_b_y = (a.x - c.x) * inverse_area
        self.gpu.store_bounds(slot, minimum_x, minimum_y, maximum_x, maximum_y)
        var address = self.gpu.record_address(slot)
        var record = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=address)
        # Weights at the centre of pixel (0, 0), so shaders pass whole-number coordinates.
        var a0 = ((b.x - 0.5) * (c.y - 0.5) - (b.y - 0.5) * (c.x - 0.5)) * inverse_area
        var b0 = ((c.x - 0.5) * (a.y - 0.5) - (c.y - 0.5) * (a.x - 0.5)) * inverse_area
        record[unsafe_offset=0] = step_a_x
        record[unsafe_offset=1] = step_a_y
        record[unsafe_offset=2] = a0
        record[unsafe_offset=3] = step_b_x
        record[unsafe_offset=4] = step_b_y
        record[unsafe_offset=5] = b0

        def plane(index: Int, va: Float32, vb: Float32, vc: Float32) capturing:
            var da = va - vc
            var db = vb - vc
            record[unsafe_offset=index] = da * step_a_x + db * step_b_x
            record[unsafe_offset=index + 1] = da * step_a_y + db * step_b_y
            record[unsafe_offset=index + 2] = da * a0 + db * b0 + vc

        plane(6, a.inverse_z, b.inverse_z, c.inverse_z)
        plane(9, a.u * a.inverse_z, b.u * b.inverse_z, c.u * c.inverse_z)
        plane(12, a.v * a.inverse_z, b.v * b.inverse_z, c.v * c.inverse_z)
        plane(16, a.light.x, b.light.x, c.light.x)
        plane(19, a.light.y, b.light.y, c.light.y)
        plane(22, a.light.z, b.light.z, c.light.z)
        plane(25, a.fog, b.fog, c.fog)
        plane(28, a.emissive, b.emissive, c.emissive)
        var per_pixel = a.occlusion >= 0 and b.occlusion >= 0 and c.occlusion >= 0
        if per_pixel:
            plane(40, a.normal.x, b.normal.x, c.normal.x)
            plane(43, a.normal.y, b.normal.y, c.normal.y)
            plane(46, a.normal.z, b.normal.z, c.normal.z)
            plane(49, a.occlusion, b.occlusion, c.occlusion)
        var has_vel = a.has_obj_vel and b.has_obj_vel and c.has_obj_vel
        if has_vel:
            plane(RECORD_VEL_X, a.vel_x * a.inverse_z, b.vel_x * b.inverse_z, c.vel_x * c.inverse_z)
            plane(RECORD_VEL_Y, a.vel_y * a.inverse_z, b.vel_y * b.inverse_z, c.vel_y * c.inverse_z)
        var words = Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=address)
        words[unsafe_offset=15] = a.color
        var is_mip = offset >= 0 and self.gpu.backend == BACKEND_VULKAN and GpuRaster.is_mipmapped(texture_width, texture_height)
        var has_norm_orm = per_pixel and self.vfx_norm_orm_gpu_meta != 0 and self.gpu.backend == BACKEND_VULKAN
        words[unsafe_offset=31] = (
            UInt32(alpha)
            | (TEXTURED_FLAG if offset >= 0 else UInt32(0))
            | (MIPMAPPED_FLAG if is_mip else UInt32(0))
            | (PER_PIXEL_FLAG if per_pixel else UInt32(0))
            | (PBR_FLAG if per_pixel else UInt32(0))
            | (DITHER_FLAG if self.occluder_pass else UInt32(0))
            | (NO_DEPTH_WRITE_FLAG if (self.no_depth_write or self.additive_blend) else UInt32(0))
            | (ADDITIVE_FLAG if self.additive_blend else UInt32(0))
            | (SOFT_PARTICLE_FLAG if self.vfx_soft_inv_dist > Float32(0.0) else UInt32(0))
            | (ALPHA_ERODE_FLAG if self.vfx_erode_threshold > Float32(0.0) else UInt32(0))
            | (BLACKBODY_FLAG if self.vfx_blackbody_heat >= Float32(0.0) else UInt32(0))
            | (REFRACT_FLAG if self.vfx_refract_strength > Float32(0.0) else UInt32(0))
            | (WATER_WAVE_FLAG if self.water_wave_normals else UInt32(0))
            | (OBJECT_VELOCITY_FLAG if has_vel else UInt32(0))
            | (NORM_ORM_FLAG if has_norm_orm else UInt32(0))
        )
        words[unsafe_offset=32] = UInt32(max(offset, 0))
        record[unsafe_offset=33] = Float32(texture_width)
        record[unsafe_offset=34] = Float32(texture_height)
        words[unsafe_offset=35] = UInt32(texture_width)
        words[unsafe_offset=36] = UInt32(max(texture_width - 1, 0))
        words[unsafe_offset=37] = UInt32(max(texture_height - 1, 0))
        record[unsafe_offset=RECORD_ROUGHNESS] = max(Float32(0.05), min(Float32(1.0), (a.roughness + b.roughness + c.roughness) * 0.33333334))
        record[unsafe_offset=RECORD_METALLIC] = max(Float32(0.0), min(Float32(1.0), (a.metallic + b.metallic + c.metallic) * 0.33333334))
        record[unsafe_offset=RECORD_SOFT_INV_DIST] = self.vfx_soft_inv_dist
        record[unsafe_offset=RECORD_ERODE_THRESHOLD] = self.vfx_erode_threshold
        record[unsafe_offset=RECORD_BLACKBODY_HEAT] = max(Float32(0.0), self.vfx_blackbody_heat)
        record[unsafe_offset=RECORD_REFRACT_STRENGTH] = self.vfx_refract_strength
        record[unsafe_offset=RECORD_ANISO] = self.mesh_anisotropy
        words[unsafe_offset=RECORD_NORM_ORM_TEX] = self.vfx_norm_orm_gpu_meta if has_norm_orm else UInt32(0)

    def clear_cpu_csm(mut self):
        self.csm_has_casters = False
        for i in range(len(self.cpu_csm_depth)):
            self.cpu_csm_depth[i] = 0.0

    def add_cpu_csm_triangle(
        mut self,
        cascade: Int,
        ax_full: Float32,
        ay_full: Float32,
        az: Float32,
        bx_full: Float32,
        by_full: Float32,
        bz: Float32,
        cx_full: Float32,
        cy_full: Float32,
        cz: Float32,
    ):
        var scale = Float32(CPU_CSM_RES) / Float32(CSM_RES)
        var ax = ax_full * scale
        var ay = ay_full * scale
        var bx = bx_full * scale
        var by = by_full * scale
        var cx = cx_full * scale
        var cy = cy_full * scale
        var area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) <= 0.005:
            return
        var max_z = max(az, max(bz, cz))
        var min_z = min(az, min(bz, cz))
        if max_z <= 0.002 or min_z >= 0.998:
            return
        var min_px = max(0, Int(floor(min(ax, min(bx, cx)))))
        var max_px = min(CPU_CSM_RES - 1, Int(floor(max(ax, max(bx, cx)))))
        var min_py = max(0, Int(floor(min(ay, min(by, cy)))))
        var max_py = min(CPU_CSM_RES - 1, Int(floor(max(ay, max(by, cy)))))
        if min_px > max_px or min_py > max_py:
            return
        self.csm_has_casters = True
        var inv_area = Float32(1.0) / area
        var wa_a = (by - cy) * inv_area
        var wa_b = (cx - bx) * inv_area
        var wa_c = (bx * cy - by * cx) * inv_area
        var wb_a = (cy - ay) * inv_area
        var wb_b = (ax - cx) * inv_area
        var wb_c = (cx * ay - cy * ax) * inv_area
        var dz_a = az - cz
        var dz_b = bz - cz
        var z_a = dz_a * wa_a + dz_b * wb_a
        var z_b = dz_a * wa_b + dz_b * wb_b
        var z_c = cz + dz_a * wa_c + dz_b * wb_c
        var casc_off = (cascade & 1) * CPU_CSM_CASCADE_WORDS
        for py in range(min_py, max_py + 1):
            var fy = Float32(py) + 0.5
            var row_off = casc_off + py * CPU_CSM_RES
            for px in range(min_px, max_px + 1):
                var fx = Float32(px) + 0.5
                var wa = wa_a * fx + wa_b * fy + wa_c
                var wb = wb_a * fx + wb_b * fy + wb_c
                var wc = Float32(1.0) - wa - wb
                if wa >= -0.0002 and wb >= -0.0002 and wc >= -0.0002:
                    var zs = z_a * fx + z_b * fy + z_c
                    if zs > 0.0005 and zs < 0.9995:
                        var idx = row_off + px
                        if zs > self.cpu_csm_depth[idx]:
                            self.cpu_csm_depth[idx] = zs

    def cpu_csm_shadow_1tap(self, world_pos: Vec3, to_key: Vec3) -> Float32:
        if not self.csm_has_casters or (self.gpu_effects & EFFECT_CSM) == 0:
            return 1.0
        var rel0 = world_pos - self.csm_c0_center
        var u0 = dot(rel0, self.csm_right) * self.csm_c0_inv_half
        var v0 = dot(rel0, self.csm_up) * self.csm_c0_inv_half
        var use_c0 = max(abs(u0), abs(v0)) < Float32(0.93)
        var rel1 = world_pos - self.csm_c1_center
        var u1 = dot(rel1, self.csm_right) * self.csm_c1_inv_half
        var v1 = dot(rel1, self.csm_up) * self.csm_c1_inv_half
        if not use_c0 and max(abs(u1), abs(v1)) >= Float32(0.96):
            return 1.0
        var us = u0 if use_c0 else u1
        var vs = v0 if use_c0 else v1
        var zb = self.csm_c0_z_bias if use_c0 else self.csm_c1_z_bias
        var zsc = self.csm_c0_z_scale if use_c0 else self.csm_c1_z_scale
        var cbase = 0 if use_c0 else CPU_CSM_CASCADE_WORDS
        var tx = max(0, min(CPU_CSM_RES - 1, Int((us * 0.5 + 0.5) * Float32(CPU_CSM_RES))))
        var ty = max(0, min(CPU_CSM_RES - 1, Int((0.5 - vs * 0.5) * Float32(CPU_CSM_RES))))
        var z_rec = (dot(world_pos, to_key) - zb) * zsc + Float32(0.503)
        if z_rec > 0.002 and z_rec < 0.998 and z_rec < self.cpu_csm_depth[cbase + ty * CPU_CSM_RES + tx]:
            return 0.08
        return 1.0

    def cpu_csm_shadow(self, world_pos: Vec3, normal: Vec3, to_key: Vec3, px_x: Int, px_y: Int) -> Float32:
        if not self.csm_has_casters or (self.gpu_effects & EFFECT_CSM) == 0:
            return 1.0
        var rel0 = world_pos - self.csm_c0_center
        var u0 = dot(rel0, self.csm_right) * self.csm_c0_inv_half
        var v0 = dot(rel0, self.csm_up) * self.csm_c0_inv_half
        var use_c0 = max(abs(u0), abs(v0)) < Float32(0.93)
        var rel1 = world_pos - self.csm_c1_center
        var u1 = dot(rel1, self.csm_right) * self.csm_c1_inv_half
        var v1 = dot(rel1, self.csm_up) * self.csm_c1_inv_half
        var max_uv1 = max(abs(u1), abs(v1))
        if not use_c0 and max_uv1 >= Float32(0.96):
            return 1.0
        var cc = self.csm_c0_center if use_c0 else self.csm_c1_center
        var inv_h = self.csm_c0_inv_half if use_c0 else self.csm_c1_inv_half
        var zb = self.csm_c0_z_bias if use_c0 else self.csm_c1_z_bias
        var zsc = self.csm_c0_z_scale if use_c0 else self.csm_c1_z_scale
        var cbase = 0 if use_c0 else CPU_CSM_CASCADE_WORDS
        var ndl = max(Float32(0.0), min(Float32(1.0), dot(normal, to_key)))
        var slope = sqrt(max(Float32(0.0), Float32(1.0) - ndl * ndl))
        var tex_world = Float32(2.0) / (Float32(CPU_CSM_RES) * max(Float32(0.0001), inv_h))
        var b_pos = world_pos + normal * (tex_world * (Float32(1.35) + slope * Float32(2.1))) + to_key * (tex_world * Float32(1.65))
        var b_rel = b_pos - cc
        var su = (dot(b_rel, self.csm_right) * inv_h * 0.5 + 0.5) * Float32(CPU_CSM_RES)
        var sv = (0.5 - dot(b_rel, self.csm_up) * inv_h * 0.5) * Float32(CPU_CSM_RES)
        var z_rec = (dot(b_pos, to_key) - zb) * zsc + Float32(0.5)
        if z_rec <= 0.002 or z_rec >= 0.998:
            return 1.0
        var ign_i = Float32(px_x) * Float32(0.06711056) + Float32(px_y) * Float32(0.00583715)
        var ign = Float32(52.9829189) * (ign_i - floor(ign_i))
        ign = ign - floor(ign)
        var sa = _parabolic_sin(ign)
        var ca = _parabolic_sin(ign + Float32(0.25))
        var lit_sum = Float32(0.0)
        for tap in range(4):
            var ox = Float32(-0.62) if (tap & 1) == 0 else Float32(0.62)
            var oy = Float32(-0.62) if tap < 2 else Float32(0.62)
            var tx = max(0, min(CPU_CSM_RES - 1, Int(su + ox * ca - oy * sa)))
            var ty = max(0, min(CPU_CSM_RES - 1, Int(sv + ox * sa + oy * ca)))
            if z_rec >= self.cpu_csm_depth[cbase + ty * CPU_CSM_RES + tx]:
                lit_sum += 0.25
        if not use_c0:
            var fade = max(Float32(0.0), min(Float32(1.0), (max_uv1 - Float32(0.86)) * Float32(10.0)))
            return lit_sum + (Float32(1.0) - lit_sum) * fade
        return lit_sum

    def cpu_sphere_shadow(self, world_pos: Vec3, to_key: Vec3) -> Float32:
        if len(self.cpu_sphere_centers) == 0:
            return 1.0
        var lit = Float32(1.0)
        for i in range(len(self.cpu_sphere_centers)):
            var rel = self.cpu_sphere_centers[i] - world_pos
            var r2 = self.cpu_sphere_radii_sq[i]
            if rel.length_squared() <= r2 * 1.15:
                continue
            var proj = dot(rel, to_key)
            if proj <= 0.0:
                continue
            var miss2 = rel.length_squared() - proj * proj
            if miss2 < r2 * 1.25:
                var pen = max(Float32(0.0), min(Float32(1.0), (miss2 - r2 * 0.55) / max(Float32(0.0001), r2 * 0.70)))
                var dist_atten = Float32(1.0) / (Float32(1.0) + (proj * proj) / (r2 * Float32(36.0) + Float32(0.001)))
                var dark = (Float32(1.0) - (0.22 + 0.78 * pen)) * dist_atten
                lit = min(lit, Float32(1.0) - dark)
        return lit

    def cpu_sdf_shadow(self, world_pos: Vec3, normal: Vec3, to_key: Vec3) -> Float32:
        """CPU parity mirror of V11 EFFECT_SDF_SHADOWS closed-form capsule/sphere penumbra."""
        var sdf_cnt = min(MAX_SDF_OCCLUDERS, len(self.cpu_sdf_occluders))
        if sdf_cnt <= 0:
            return 1.0
        var wb = world_pos + normal * Float32(0.18)
        var sdf_k = max(Float32(1.5), self.sdf_softness_k)
        var vis = Float32(1.0)
        for si in range(sdf_cnt):
            var so = self.cpu_sdf_occluders[si]
            var oa = so.p0 - wb
            var s_len2 = so.seg.length_squared()
            var u_seg = Float32(0.0)
            if s_len2 > Float32(0.0001):
                var d_s = dot(to_key, so.seg)
                var oa_d = dot(oa, to_key)
                var oa_s = dot(oa, so.seg)
                var den_s = max(Float32(0.0001), s_len2 - d_s * d_s)
                u_seg = max(Float32(0.0), min(Float32(1.0), (oa_d * d_s - oa_s) / den_s))
            var rc = oa + so.seg * u_seg
            var rc2 = rc.length_squared()
            var t_ray = dot(rc, to_key)
            if rc2 > so.radius * so.radius * Float32(1.08) and t_ray > Float32(0.10):
                var perp = sqrt(max(Float32(0.0), rc2 - t_ray * t_ray))
                var h_clr = perp - so.radius
                var pen_den = max(Float32(0.45), t_ray * Float32(0.35) + so.radius)
                var pen_v = max(Float32(0.0), min(Float32(1.0), Float32(0.5) + Float32(0.5) * sdf_k * h_clr / pen_den))
                var opac_s = max(Float32(0.0), min(Float32(0.95), so.opacity))
                vis = min(vis, Float32(1.0) - opac_s * (Float32(1.0) - pen_v))
        return vis

    def raster_triangle(mut self, a: ScreenVertex, b: ScreenVertex, c: ScreenVertex, mut texture: List[UInt32], texture_width: Int, texture_height: Int, alpha: UInt8, double_sided: Bool = False):
        if self.gpu.active:
            var slot = self.gpu.reserve_slots(1)
            if slot >= 0:
                var offset = -1
                if texture_width > 0 and texture_height > 0 and len(texture) >= texture_width * texture_height:
                    offset = self.gpu.texture_offset(Int(texture.unsafe_ptr()), texture_width, texture_height)
                if self.vfx_norm_orm_ptr != 0 and self.vfx_norm_orm_dim > 0 and self.vfx_norm_orm_gpu_meta == 0:
                    var n_off = self.gpu.texture_offset(self.vfx_norm_orm_ptr, self.vfx_norm_orm_dim, self.vfx_norm_orm_dim)
                    if n_off >= 0:
                        var n_log2 = 0
                        while (1 << n_log2) < self.vfx_norm_orm_dim and n_log2 < 11:
                            n_log2 += 1
                        self.vfx_norm_orm_gpu_meta = (UInt32(n_log2) << 28) | UInt32(n_off >> 2)
                self.fill_gpu_slot(slot, a, b, c, offset, texture_width, texture_height, alpha, double_sided)
            return
        # The view keeps world +X on screen right, which mirrors winding: faces toward
        # the camera have a positive area here.
        var area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        if abs(area) <= 0.00001 or (not double_sided and area <= 0):
            self.culled_triangles += 1
            return
        var rw_lim = self.render_w if self.tsr_subnative_active else self.width
        var rh_lim = self.render_h if self.tsr_subnative_active else self.height
        var minimum_x = max(0, Int(floor(min(a.x, min(b.x, c.x)) - 0.5)))
        var maximum_x = min(rw_lim - 1, Int(floor(max(a.x, max(b.x, c.x)) + 0.5)))
        var minimum_y = max(0, Int(floor(min(a.y, min(b.y, c.y)) - 0.5)))
        var maximum_y = min(rh_lim - 1, Int(floor(max(a.y, max(b.y, c.y)) + 0.5)))
        if minimum_x > maximum_x or minimum_y > maximum_y:
            self.culled_triangles += 1
            return
        var inverse_area = 1.0 / area
        var textured = texture_width > 0 and texture_height > 0 and len(texture) >= texture_width * texture_height
        var has_norm_orm_cpu = self.vfx_norm_orm_ptr != 0 and self.vfx_norm_orm_dim >= 4
        var per_pixel = a.occlusion >= 0 and b.occlusion >= 0 and c.occlusion >= 0
        var rough = max(Float32(0.05), min(Float32(1.0), (a.roughness + b.roughness + c.roughness) * 0.33333334))
        var metal = max(Float32(0.0), min(Float32(1.0), (a.metallic + b.metallic + c.metallic) * 0.33333334))
        var jx = Float32(0.0)
        var jy = Float32(0.0)
        if (
            (self.gpu_effects & EFFECT_TAA) != 0
            and self.temporal_frames > 0
            and (self.camera - self.prev_camera).length_squared() <= Float32(324.0)
            and dot(self.forward, self.prev_forward) >= Float32(0.70)
            and abs(self.focal - self.prev_focal) <= self.focal * Float32(0.15)
        ):
            var jit = Self._halton_jitter(self.temporal_frames)
            jx = jit[0]
            jy = jit[1]
        var start_x = Float32(minimum_x) + 0.5 + jx
        var start_y = Float32(minimum_y) + 0.5 + jy
        var row_a = ((b.x - start_x) * (c.y - start_y) - (b.y - start_y) * (c.x - start_x)) * inverse_area
        var row_b = ((c.x - start_x) * (a.y - start_y) - (c.y - start_y) * (a.x - start_x)) * inverse_area
        var step_a_x = (b.y - c.y) * inverse_area
        var step_b_x = (c.y - a.y) * inverse_area
        var step_a_y = (c.x - b.x) * inverse_area
        var step_b_y = (a.x - c.x) * inverse_area
        var step_c_x = -step_a_x - step_b_x
        var step_c_y = -step_a_y - step_b_y
        var iz_dx = step_a_x * a.inverse_z + step_b_x * b.inverse_z + step_c_x * c.inverse_z
        var iz_dy = step_a_y * a.inverse_z + step_b_y * b.inverse_z + step_c_y * c.inverse_z
        var uz_dx = step_a_x * (a.u * a.inverse_z) + step_b_x * (b.u * b.inverse_z) + step_c_x * (c.u * c.inverse_z)
        var uz_dy = step_a_y * (a.u * a.inverse_z) + step_b_y * (b.u * b.inverse_z) + step_c_y * (c.u * c.inverse_z)
        var vz_dx = step_a_x * (a.v * a.inverse_z) + step_b_x * (b.v * b.inverse_z) + step_c_x * (c.v * c.inverse_z)
        var vz_dy = step_a_y * (a.v * a.inverse_z) + step_b_y * (b.v * b.inverse_z) + step_c_y * (c.v * c.inverse_z)
        var inv_focal = 1.0 / (self.render_focal if self.tsr_subnative_active else self.focal)
        var half_w = Float32(self.render_w if self.tsr_subnative_active else self.width) * 0.5
        var half_h = Float32(self.render_h if self.tsr_subnative_active else self.height) * 0.5
        var fog_r_f = Float32((self.fog_color >> 16) & 255)
        var fog_g_f = Float32((self.fog_color >> 8) & 255)
        var fog_b_f = Float32(self.fog_color & 255)
        var tw_f = Float32(max(1, texture_width))
        var th_f = Float32(max(1, texture_height))
        var inv_tw = Float32(1.0) / tw_f
        var inv_th = Float32(1.0) / th_f
        self.drawn_triangles += 1
        var dithered = self.occluder_pass and self.dither.active()
        for y in range(minimum_y, maximum_y + 1):
            var weight_a = row_a
            var weight_b = row_b
            var index = y * self.width + minimum_x
            var sample_py = Float32(y) + 0.5 + jy
            for x in range(minimum_x, maximum_x + 1):
                var weight_c = 1.0 - weight_a - weight_b
                if weight_a >= -0.0001 and weight_b >= -0.0001 and weight_c >= -0.0001:
                    self.pixels_tested += 1
                    var inverse_z = weight_a * a.inverse_z + weight_b * b.inverse_z + weight_c * c.inverse_z
                    if inverse_z > 0:
                        var view_depth = 1.0 / inverse_z
                        var prev_depth = self.depth[index]
                        if view_depth < prev_depth and not (dithered and self.dither.drops(x, y, inverse_z)):
                            var u = Float32(0.5)
                            var v = Float32(0.5)
                            var du_dx = Float32(0.0)
                            var dv_dx = Float32(0.0)
                            var du_dy = Float32(0.0)
                            var dv_dy = Float32(0.0)
                            var sampled = UInt32(0xFFFFFFFF)
                            if textured or has_norm_orm_cpu:
                                u = (weight_a * a.u * a.inverse_z + weight_b * b.u * b.inverse_z + weight_c * c.u * c.inverse_z) * view_depth
                                v = (weight_a * a.v * a.inverse_z + weight_b * b.v * b.inverse_z + weight_c * c.v * c.inverse_z) * view_depth
                                du_dx = view_depth * (uz_dx - u * iz_dx)
                                du_dy = view_depth * (uz_dy - u * iz_dy)
                                dv_dx = view_depth * (vz_dx - v * iz_dx)
                                dv_dy = view_depth * (vz_dy - v * iz_dy)
                                if textured:
                                    sampled = self.sample_texture_filtered(texture, texture_width, texture_height, u, v, du_dx, dv_dx, du_dy, dv_dy)
                            var light = a.light * weight_a + b.light * weight_b + c.light * weight_c
                            var fog = a.fog * weight_a + b.fog * weight_b + c.fog * weight_c
                            var emissive = a.emissive * weight_a + b.emissive * weight_b + c.emissive * weight_c
                            var tint = a.color
                            var red = Int((sampled >> 16) & 255) * Int((tint >> 16) & 255) // 255
                            var green = Int((sampled >> 8) & 255) * Int((tint >> 8) & 255) // 255
                            var blue = Int(sampled & 255) * Int(tint & 255) // 255
                            if self.vfx_blackbody_heat >= Float32(0.0):
                                var bb_h = self.vfx_blackbody_heat
                                var cell_n = Float32((sampled >> 16) & 255) / Float32(255.0)
                                var t_kelvin = max(Float32(0.0), min(Float32(1.35), cell_n * (Float32(0.30) + Float32(1.28) * bb_h) - (Float32(1.0) - bb_h) * Float32(0.26)))
                                var r_th = Float32(44.0) + max(Float32(0.0), min(Float32(1.0), t_kelvin * Float32(2.25))) * Float32(211.0)
                                var g_th = Float32(36.0) + max(Float32(0.0), min(Float32(1.0), (t_kelvin - Float32(0.22)) * Float32(1.42))) * Float32(212.0)
                                var b_th = Float32(34.0) + max(Float32(0.0), min(Float32(1.0), (t_kelvin - Float32(0.62)) * Float32(1.45))) * Float32(184.0)
                                red = max(0, min(255, Int(r_th * Float32((tint >> 16) & 255) / Float32(255.0))))
                                green = max(0, min(255, Int(g_th * Float32((tint >> 8) & 255) / Float32(255.0))))
                                blue = max(0, min(255, Int(b_th * Float32(tint & 255) / Float32(255.0))))
                                emissive += t_kelvin * t_kelvin * bb_h * Float32(2.32)
                            var spec_rgb = Vec3()
                            var surf_ox = UInt32(128)
                            var surf_oz = UInt32(128)
                            var surf_rough = UInt32(255)
                            var surf_refl = UInt32(0)
                            if per_pixel:
                                var px_scale = inv_focal * view_depth
                                var vx = (Float32(x) + 0.5 + jx - half_w) * px_scale
                                var vy = (half_h - sample_py) * px_scale
                                var world_pos = self.camera + self.right * vx + self.up * vy + self.forward * view_depth
                                var raw_n = a.normal * weight_a + b.normal * weight_b + c.normal * weight_c
                                var px_rough = rough
                                var water_crest = Float32(0.0)
                                var water_near = Float32(0.0)
                                var cot_t = Vec3()
                                var has_cot = False
                                var ao_orm_mul = Float32(1.0)
                                if self.water_wave_normals:
                                    var vdir_w = (world_pos - self.camera).normalized()
                                    var graz_sin = max(Float32(0.06), abs(vdir_w.y))
                                    var px_foot = (view_depth * inv_focal) / graz_sin
                                    var near_att = max(Float32(0.0), min(Float32(1.0), Float32(1.0) - px_foot * Float32(0.11)))
                                    var fine_att = max(Float32(0.0), min(Float32(1.0), Float32(1.0) - px_foot * Float32(0.62)))
                                    px_rough = min(Float32(1.0), px_rough + (Float32(1.0) - fine_att) * Float32(0.075) + (Float32(1.0) - near_att) * Float32(0.055))
                                    var ph1 = world_pos.x * Float32(0.021) + world_pos.z * Float32(0.014) + self.water_time * Float32(0.22)
                                    var ph2 = world_pos.z * Float32(0.043) - world_pos.x * Float32(0.019) - self.water_time * Float32(0.31)
                                    var ph3 = world_pos.x * Float32(0.118) + world_pos.z * Float32(0.089) + self.water_time * Float32(0.53)
                                    var ph4 = world_pos.z * Float32(0.340) - world_pos.x * Float32(0.235) - self.water_time * Float32(0.79)
                                    var w1 = _parabolic_sin(ph1)
                                    var w2 = _parabolic_sin(ph2)
                                    var w3 = _parabolic_sin(ph3)
                                    var w4 = _parabolic_sin(ph4)
                                    var c1 = _parabolic_sin(ph1 + Float32(0.25))
                                    var c2 = _parabolic_sin(ph2 + Float32(0.25))
                                    var c3 = _parabolic_sin(ph3 + Float32(0.25))
                                    var c4 = _parabolic_sin(ph4 + Float32(0.25))
                                    raw_n.x += (w1 * Float32(0.108) - w2 * Float32(0.072)) * near_att + (w3 * Float32(0.086) - w4 * Float32(0.068)) * fine_att
                                    raw_n.z += (w1 * Float32(0.072) + w2 * Float32(0.098)) * near_att + (w3 * Float32(0.068) + w4 * Float32(0.092)) * fine_att
                                    water_crest = max(Float32(0.0), min(Float32(1.0), (c1 * Float32(0.42) + c2 * Float32(0.30)) * near_att + (c3 * Float32(0.18) + c4 * Float32(0.12)) * fine_att))
                                    water_near = near_att
                                    var jacobian = Float32(1.0) - ((c1 * Float32(0.34) + c2 * Float32(0.28)) * near_att + (c3 * Float32(0.24) + c4 * Float32(0.20)) * fine_att)
                                    var whitecap = max(Float32(0.0), min(Float32(0.72), (Float32(0.52) - jacobian) * Float32(2.1)))
                                    var shore_foam = Float32(0.0)
                                    if prev_depth < Float32(5000.0):
                                        var col_depth = prev_depth - view_depth
                                        if col_depth > Float32(0.0) and col_depth < Float32(2.8):
                                            var shallows = max(Float32(0.0), min(Float32(1.0), Float32(1.0) - col_depth * Float32(0.357143)))
                                            var surf_band = _parabolic_sin(col_depth * Float32(1.15) - self.water_time * Float32(0.42) + w1 * Float32(0.18)) * Float32(0.5) + Float32(0.5)
                                            shore_foam = shallows * shallows * (Float32(0.38) + surf_band * Float32(0.62))
                                    var foam = max(Float32(0.0), min(Float32(0.86), max(whitecap, shore_foam)))
                                    if foam > Float32(0.005):
                                        red = max(0, min(255, Int(Float32(red) + (Float32(216.0) - Float32(red)) * foam + 0.5)))
                                        green = max(0, min(255, Int(Float32(green) + (Float32(224.0) - Float32(green)) * foam + 0.5)))
                                        blue = max(0, min(255, Int(Float32(blue) + (Float32(226.0) - Float32(blue)) * foam + 0.5)))
                                        px_rough = px_rough + (Float32(0.58) - px_rough) * foam
                                elif has_norm_orm_cpu:
                                    var n0 = raw_n.normalized()
                                    var dpx = self.right * px_scale
                                    var dpy = self.up * (-px_scale)
                                    var tx_v = dpx - n0 * dot(n0, dpx)
                                    var ty_v = dpy - n0 * dot(n0, dpy)
                                    var tan_raw = tx_v * dv_dy - ty_v * dv_dx
                                    var bit_raw = ty_v * du_dx - tx_v * du_dy
                                    var tan_l2 = tan_raw.length_squared()
                                    var bit_l2 = bit_raw.length_squared()
                                    if tan_l2 > Float32(1e-11) and bit_l2 > Float32(1e-11):
                                        var inv_tl = Float32(1.0) / sqrt(tan_l2)
                                        var inv_bl = Float32(1.0) / sqrt(bit_l2)
                                        var tan_u = tan_raw * inv_tl
                                        var bit_u = bit_raw * inv_bl
                                        cot_t = tan_u
                                        has_cot = True
                                        var ndim = self.vfx_norm_orm_dim
                                        var ndim_f = Float32(ndim)
                                        var u_wr = u - floor(u)
                                        var v_wr = v - floor(v)
                                        var fx_n = u_wr * ndim_f - Float32(0.5)
                                        var fy_n = v_wr * ndim_f - Float32(0.5)
                                        var ix0 = Int(floor(fx_n))
                                        var iy0 = Int(floor(fy_n))
                                        var tx_f = fx_n - Float32(ix0)
                                        var ty_f = fy_n - Float32(iy0)
                                        var mask_n = ndim - 1
                                        var x0_n = ix0 & mask_n
                                        var x1_n = (ix0 + 1) & mask_n
                                        var y0_n = iy0 & mask_n
                                        var y1_n = (iy0 + 1) & mask_n
                                        var n_ptr = Pointer[UInt32, origin=ImmutAnyOrigin](unsafe_from_address=self.vfx_norm_orm_ptr)
                                        var p00 = n_ptr[unsafe_offset=y0_n * ndim + x0_n]
                                        var p10 = n_ptr[unsafe_offset=y0_n * ndim + x1_n]
                                        var p01 = n_ptr[unsafe_offset=y1_n * ndim + x0_n]
                                        var p11 = n_ptr[unsafe_offset=y1_n * ndim + x1_n]
                                        var w00 = (Float32(1.0) - tx_f) * (Float32(1.0) - ty_f)
                                        var w10 = tx_f * (Float32(1.0) - ty_f)
                                        var w01 = (Float32(1.0) - tx_f) * ty_f
                                        var w11 = tx_f * ty_f
                                        var r_n = Float32((p00 >> 16) & 255) * w00 + Float32((p10 >> 16) & 255) * w10 + Float32((p01 >> 16) & 255) * w01 + Float32((p11 >> 16) & 255) * w11
                                        var g_n = Float32((p00 >> 8) & 255) * w00 + Float32((p10 >> 8) & 255) * w10 + Float32((p01 >> 8) & 255) * w01 + Float32((p11 >> 8) & 255) * w11
                                        var b_n = Float32(p00 & 255) * w00 + Float32(p10 & 255) * w10 + Float32(p01 & 255) * w01 + Float32(p11 & 255) * w11
                                        var a_n = Float32((p00 >> 24) & 255) * w00 + Float32((p10 >> 24) & 255) * w10 + Float32((p01 >> 24) & 255) * w01 + Float32((p11 >> 24) & 255) * w11
                                        var grad_m = max(abs(du_dx) + abs(dv_dx), abs(du_dy) + abs(dv_dy)) * ndim_f
                                        var approx_lod = min(Float32(3.0), max(Float32(0.0), log2(max(Float32(1.0), grad_m)) * Float32(0.85)))
                                        var n_gain = max(Float32(0.25), Float32(1.0) - approx_lod * Float32(0.24)) * Float32(0.72)
                                        var nx_m = (r_n - Float32(128.0)) * Float32(0.007874016) * n_gain
                                        var ny_m = (g_n - Float32(128.0)) * Float32(0.007874016) * n_gain
                                        var orm_rough = max(Float32(0.06), min(Float32(1.0), b_n * Float32(0.003921569)))
                                        ao_orm_mul = max(Float32(0.25), min(Float32(1.0), a_n * Float32(0.003921569)))
                                        raw_n = raw_n + tan_u * nx_m + bit_u * ny_m
                                        px_rough = max(Float32(0.06), min(Float32(1.0), px_rough + (orm_rough - px_rough) * Float32(0.55)))
                                elif textured and (self.gpu_effects & EFFECT_BILINEAR) != 0 and texture_width >= 16 and texture_height >= 16:
                                    var fx2 = (du_dx * tw_f) * (du_dx * tw_f) + (dv_dx * th_f) * (dv_dx * th_f)
                                    var fy2 = (du_dy * tw_f) * (du_dy * tw_f) + (dv_dy * th_f) * (dv_dy * th_f)
                                    var fmax2 = max(fx2, fy2)
                                    if fmax2 < Float32(48.0):
                                        var c_up = self.sample_texture_bilinear(texture, texture_width, texture_height, u + inv_tw, v)
                                        var c_un = self.sample_texture_bilinear(texture, texture_width, texture_height, u - inv_tw, v)
                                        var c_vp = self.sample_texture_bilinear(texture, texture_width, texture_height, u, v + inv_th)
                                        var c_vn = self.sample_texture_bilinear(texture, texture_width, texture_height, u, v - inv_th)
                                        var l_up = (Float32((c_up >> 16) & 255) * 0.26 + Float32((c_up >> 8) & 255) * 0.58 + Float32(c_up & 255) * 0.16) / 255.0
                                        var l_un = (Float32((c_un >> 16) & 255) * 0.26 + Float32((c_un >> 8) & 255) * 0.58 + Float32(c_un & 255) * 0.16) / 255.0
                                        var l_vp = (Float32((c_vp >> 16) & 255) * 0.26 + Float32((c_vp >> 8) & 255) * 0.58 + Float32(c_vp & 255) * 0.16) / 255.0
                                        var l_vn = (Float32((c_vn >> 16) & 255) * 0.26 + Float32((c_vn >> 8) & 255) * 0.58 + Float32(c_vn & 255) * 0.16) / 255.0
                                        var dh_u = l_up - l_un
                                        var dh_v = l_vp - l_vn
                                        var n0 = raw_n.normalized()
                                        var dpx = self.right * px_scale
                                        var dpy = self.up * (-px_scale)
                                        var tx_v = dpx - n0 * dot(n0, dpx)
                                        var ty_v = dpy - n0 * dot(n0, dpy)
                                        var tan_raw = tx_v * dv_dy - ty_v * dv_dx
                                        var bit_raw = ty_v * du_dx - tx_v * du_dy
                                        var tan_l2 = tan_raw.length_squared()
                                        var bit_l2 = bit_raw.length_squared()
                                        if tan_l2 > Float32(1e-11) and bit_l2 > Float32(1e-11):
                                            var inv_tl = Float32(1.0) / sqrt(tan_l2)
                                            cot_t = tan_raw * inv_tl
                                            has_cot = True
                                            var det_fade = max(Float32(0.0), Float32(1.0) - sqrt(fmax2) * Float32(0.14)) * Float32(0.18)
                                            raw_n = raw_n - tan_raw * (dh_u * inv_tl * det_fade) - bit_raw * (dh_v / sqrt(bit_l2) * det_fade)
                                            px_rough = min(Float32(1.0), max(Float32(0.06), px_rough + min(Float32(0.09), (abs(dh_u) + abs(dh_v)) * Float32(0.22))))
                                if not self.water_wave_normals and len(self.cpu_decals) > 0:
                                    var base_rf = Float32(red)
                                    var base_gf = Float32(green)
                                    var base_bf = Float32(blue)
                                    var n_dec = min(MAX_SURFACE_DECALS, len(self.cpu_decals))
                                    for di in range(n_dec):
                                        var dc = self.cpu_decals[di]
                                        var d_rel = world_pos - dc.center
                                        var ax_n = cross(dc.axis_right, dc.axis_forward)
                                        var du_d = dot(d_rel, dc.axis_right) * dc.inv_hx
                                        var dv_d = dot(d_rel, dc.axis_forward) * dc.inv_hy
                                        var dn_d = abs(dot(d_rel, ax_n) * dc.inv_hz)
                                        var r2_d = du_d * du_d + dv_d * dv_d
                                        if r2_d < Float32(1.0) and dn_d < Float32(1.0):
                                            var fall_d = (Float32(1.0) - r2_d) * (Float32(1.0) - dn_d)
                                            var dec_op = max(Float32(0.0), min(Float32(0.94), abs(dc.signed_opacity) * fall_d * Float32(1.35)))
                                            base_rf = base_rf + (dc.albedo.x - base_rf) * dec_op
                                            base_gf = base_gf + (dc.albedo.y - base_gf) * dec_op
                                            base_bf = base_bf + (dc.albedo.z - base_bf) * dec_op
                                            px_rough = px_rough + (Float32(0.80) - px_rough) * (dec_op * Float32(0.65))
                                            if dc.signed_opacity < Float32(0.0):
                                                var ring = max(Float32(0.0), Float32(1.0) - abs(r2_d - Float32(0.30)) * Float32(3.2))
                                                var ember = ring * (Float32(1.0) - dn_d) * max(Float32(0.0), min(Float32(2.0), -dc.signed_opacity))
                                                spec_rgb = spec_rgb + Vec3(Float32(245.0), Float32(118.0), Float32(28.0)) * ember
                                    red = max(0, min(255, Int(base_rf + Float32(0.5))))
                                    green = max(0, min(255, Int(base_gf + Float32(0.5))))
                                    blue = max(0, min(255, Int(base_bf + Float32(0.5))))
                                if not self.water_wave_normals and (self.gpu_effects & EFFECT_WEATHER_PRECIP) != 0 and self.wetness > Float32(0.01):
                                    var up_wet = max(Float32(0.0), min(Float32(1.0), raw_n.y * Float32(0.85) + Float32(0.25))) * self.wetness
                                    var dark_f = Float32(1.0) - up_wet * (Float32(1.0) - metal) * Float32(0.42)
                                    red = max(0, min(255, Int(Float32(red) * dark_f + Float32(0.5))))
                                    green = max(0, min(255, Int(Float32(green) * dark_f + Float32(0.5))))
                                    blue = max(0, min(255, Int(Float32(blue) * dark_f + Float32(0.5))))
                                    var wet_g = up_wet * Float32(0.62)
                                    px_rough = px_rough + (Float32(0.075) - px_rough) * wet_g
                                    if self.weather_kind == 1 and raw_n.y > Float32(0.65):
                                        var rip_ph = world_pos.x * Float32(1.45) + world_pos.z * Float32(1.32) - self.water_time * Float32(2.4)
                                        raw_n.x += _parabolic_sin(rip_ph) * self.wetness * Float32(0.045)
                                        raw_n.z += _parabolic_sin(rip_ph + Float32(0.25)) * self.wetness * Float32(0.045)
                                var n = raw_n.normalized()
                                var fb_t = Vec3(n.z, 0.0, -n.x) if abs(n.y) < Float32(0.92) else Vec3(0.0, -n.z, n.y)
                                var t0 = cot_t if has_cot else fb_t
                                var tan_dir = (t0 - n * dot(n, t0)).normalized()
                                var bit_dir = cross(n, tan_dir)
                                var oct = Self._encode_oct_normal(n)
                                surf_ox = oct[0]
                                surf_oz = oct[1]
                                surf_rough = UInt32(max(0, min(255, Int(px_rough * 255.0 + 0.5))))
                                var ao = max(Float32(0.0), min(Float32(1.0), (a.occlusion * weight_a + b.occlusion * weight_b + c.occlusion * weight_c) * ao_orm_mul))
                                var view_dir = (self.camera - world_pos).normalized()
                                var ndotv = max(Float32(0.02), dot(n, view_dir))
                                var f0 = Vec3(0.04 + (Float32(red) - 0.04 * 255.0) / 255.0 * metal, 0.04 + (Float32(green) - 0.04 * 255.0) / 255.0 * metal, 0.04 + (Float32(blue) - 0.04 * 255.0) / 255.0 * metal)
                                var kd = (1.0 - metal) * (1.0 - 0.04)
                                light = light * (0.35 + 0.65 * kd)
                                var probe_col = Vec3()
                                var sky_occ = Float32(0.0)
                                var n_probes = min(MAX_IRRADIANCE_PROBES, len(self.cpu_probes))
                                for pi in range(n_probes):
                                    var pr = self.cpu_probes[pi]
                                    var dp_v = pr.position - world_pos
                                    var dp2 = dp_v.length_squared() + Float32(0.001)
                                    var tp0 = max(Float32(0.0), Float32(1.0) - dp2 * pr.inv_radius2)
                                    var wp_pr = tp0 * tp0
                                    if wp_pr > Float32(0.0005):
                                        var sh_dir = max(Float32(0.0), dot(n, dp_v) / sqrt(dp2)) * Float32(0.35) + Float32(0.65)
                                        probe_col = probe_col + pr.color * (wp_pr * sh_dir)
                                        sky_occ = max(sky_occ, wp_pr * max(Float32(0.0), min(Float32(0.95), pr.sky_occlusion)))
                                if sky_occ > Float32(0.0) or probe_col.length_squared() > Float32(0.0):
                                    light = light * (Float32(1.0) - sky_occ * Float32(0.65)) + probe_col * (ao * kd)
                                var alpha_r = px_rough * px_rough
                                var auto_aniso = Float32(0.0) if self.water_wave_normals else max(Float32(0.0), min(Float32(0.36), (metal - Float32(0.35)) * Float32(0.55)))
                                var aniso = max(max(Float32(0.0), min(Float32(0.85), self.mesh_anisotropy)), auto_aniso)
                                var ax = max(Float32(0.03), alpha_r * (Float32(1.0) + aniso * Float32(0.85)))
                                var ay = max(Float32(0.03), alpha_r * (Float32(1.0) - aniso * Float32(0.55)))
                                var k_ibl = (px_rough + 1.0) * (px_rough + 1.0) * 0.125
                                # Brian Karis Split-Sum GGX Environment BRDF + Kulla-Conty Multiscattering Energy Compensation
                                var kr_x = 1.0 - px_rough
                                var kr_y = Float32(0.0425) - px_rough * Float32(0.0275)
                                var kr_z = Float32(1.04) - px_rough * Float32(0.572)
                                var kr_w = px_rough * Float32(0.022) - Float32(0.04)
                                var exp_nv = exp(Float32(-6.432408) * ndotv)
                                var a004 = min(kr_x * kr_x, exp_nv) * kr_x + kr_y
                                var brdf_scale = max(Float32(0.0), min(Float32(1.0), kr_z - a004 * Float32(1.04)))
                                var brdf_bias = max(Float32(0.0), min(Float32(1.0), kr_w + a004 * Float32(1.04)))
                                var dir_albedo = max(Float32(0.18), min(Float32(1.0), brdf_scale + brdf_bias))
                                var f0_norm = max(Float32(0.0), min(Float32(1.0), (f0.x + f0.y + f0.z) * Float32(0.33333334)))
                                var ms_loss = max(Float32(0.0), min(Float32(1.35), (Float32(1.0) - dir_albedo) / dir_albedo))
                                var ms_comp = Float32(1.0) + f0_norm * ms_loss * Float32(0.58)
                                var ndotl_key = dot(n, self.cpu_to_key)
                                if ndotl_key > 0.0:
                                    var shadow = Float32(1.0)
                                    if (self.gpu_effects & EFFECT_SHADOWS) != 0:
                                        shadow = self.cpu_sphere_shadow(world_pos + n * 0.08, self.cpu_to_key)
                                    if (self.gpu_effects & EFFECT_SDF_SHADOWS) != 0:
                                        shadow *= self.cpu_sdf_shadow(world_pos, n, self.cpu_to_key)
                                    if (self.gpu_effects & EFFECT_VOL_CLOUDS) != 0 and self.cpu_to_key.y > Float32(0.05):
                                        var cl_mid_y = (self.cloud_base_y + self.cloud_top_y) * Float32(0.5)
                                        if world_pos.y < cl_mid_y:
                                            var t_cl_sh = (cl_mid_y - world_pos.y) / max(Float32(0.08), self.cpu_to_key.y)
                                            if t_cl_sh > Float32(0.0) and t_cl_sh < Float32(6500.0):
                                                var cl_d_sh = self._cloud_density_cpu(
                                                    world_pos.x + self.cpu_to_key.x * t_cl_sh,
                                                    cl_mid_y,
                                                    world_pos.z + self.cpu_to_key.z * t_cl_sh,
                                                )
                                                shadow *= max(Float32(0.22), min(Float32(1.0), Float32(1.0) - cl_d_sh * self.cloud_shadow_strength))
                                    var direct_ao = 0.78 + ao * 0.22
                                    light = light + self.cpu_key_color * (ndotl_key * direct_ao * shadow * kd)
                                    var h = (view_dir + self.cpu_to_key).normalized()
                                    var ndoth = max(Float32(0.0), dot(n, h))
                                    var tx_h = dot(tan_dir, h) / ax
                                    var by_h = dot(bit_dir, h) / ay
                                    var ellip = tx_h * tx_h + by_h * by_h + ndoth * ndoth
                                    var vdoth = max(Float32(0.0), dot(view_dir, h))
                                    var d_ggx = Float32(1.0) / max(Float32(0.0001), Float32(3.14159265) * ax * ay * ellip * ellip)
                                    var g_v = ndotv / (ndotv * (1.0 - k_ibl) + k_ibl)
                                    var g_l = ndotl_key / (ndotl_key * (1.0 - k_ibl) + k_ibl)
                                    var omv = 1.0 - vdoth
                                    var omv2 = omv * omv
                                    var omv5 = omv2 * omv2 * omv
                                    var fres = f0 + (Vec3(1, 1, 1) - f0) * omv5
                                    var brdf = min(Float32(6.0), d_ggx * g_v * g_l / max(Float32(0.05), 4.0 * ndotv * ndotl_key) * ndotl_key) * ms_comp
                                    var spec_scale = brdf * shadow * 255.0
                                    spec_rgb = spec_rgb + Vec3(fres.x * self.cpu_key_color.x * spec_scale, fres.y * self.cpu_key_color.y * spec_scale, fres.z * self.cpu_key_color.z * spec_scale)
                                    # 2nd Automotive Clear-Coat Lacquer Lobe
                                    var cc_gain = Float32(0.36) * max(Float32(0.0), Float32(1.0) - px_rough * Float32(0.72))
                                    if cc_gain > Float32(0.001):
                                        var denom_cc = ndoth * ndoth * (Float32(0.0085) - Float32(1.0)) + Float32(1.0)
                                        var d_cc = Float32(0.0085) / max(Float32(0.0001), Float32(3.14159265) * denom_cc * denom_cc)
                                        var gv_cc = ndotv * Float32(0.851) + Float32(0.149)
                                        var gl_cc = ndotl_key * Float32(0.851) + Float32(0.149)
                                        var vis_cc = Float32(0.25) / max(Float32(0.0001), gv_cc * gl_cc)
                                        var lobe_cc = min(Float32(6.0), d_cc * vis_cc * ndotl_key * shadow)
                                        var fres_cc = (Float32(0.04) + Float32(0.96) * omv5) * Float32(255.0)
                                        var cc_scale = lobe_cc * fres_cc * cc_gain
                                        spec_rgb = spec_rgb + self.cpu_key_color * cc_scale
                                if self.water_wave_normals and water_crest > Float32(0.0):
                                    var back_align = max(Float32(0.0), -dot(view_dir, self.cpu_to_key) * Float32(0.65) + Float32(0.35))
                                    var sss_w = water_crest * back_align * water_near * Float32(0.38)
                                    light = light + Vec3(self.cpu_key_color.x * sss_w * Float32(0.08), self.cpu_key_color.y * sss_w * Float32(0.44), self.cpu_key_color.z * sss_w * Float32(0.42))
                                var refl_dir = Vec3(2.0 * ndotv * n.x - view_dir.x, 2.0 * ndotv * n.y - view_dir.y, 2.0 * ndotv * n.z - view_dir.z)
                                var refl_y = refl_dir.y
                                var up_t = max(Float32(0.0), min(Float32(1.0), refl_y * 1.35))
                                var dn_t = max(Float32(0.0), min(Float32(1.0), -refl_y * 1.55))
                                var env_col = self.cpu_env_horizon + (self.cpu_env_sky - self.cpu_env_horizon) * up_t + (self.cpu_env_ground - self.cpu_env_horizon) * dn_t
                                var smoothness = 1.0 - px_rough * 0.80
                                var sun_align = max(Float32(0.0), dot(refl_dir, self.cpu_to_key))
                                var sun_a2 = sun_align * sun_align
                                var horiz_mask = max(Float32(0.0), min(Float32(1.0), Float32(1.0) - abs(refl_y) * Float32(0.68)))
                                env_col = (env_col + self.cpu_key_color * (sun_a2 * sun_a2 * horiz_mask * smoothness * Float32(0.28))) * (Float32(1.0) - sky_occ) + probe_col * Float32(0.42)
                                var f90 = Vec3(max(smoothness, f0.x), max(smoothness, f0.y), max(smoothness, f0.z))
                                var fe = Vec3(
                                    (f0.x * brdf_scale + f90.x * brdf_bias) * ms_comp,
                                    (f0.y * brdf_scale + f90.y * brdf_bias) * ms_comp,
                                    (f0.z * brdf_scale + f90.z * brdf_bias) * ms_comp,
                                )
                                var ao_ibl = 0.50 + 0.50 * ao
                                var ibl_w = smoothness * ao_ibl * Float32(0.84) * Float32(255.0)
                                spec_rgb = spec_rgb + Vec3(fe.x * env_col.x * ibl_w, fe.y * env_col.y * ibl_w, fe.z * env_col.z * ibl_w)
                                if (self.gpu_effects & EFFECT_DYN_REFLECT) != 0 and self.dyn_reflect_intensity > Float32(0.005):
                                    var d_align = max(Float32(0.0), dot(refl_dir, self.cpu_dyn_lobe_dir))
                                    var d_a2 = d_align * d_align
                                    var d_lobe = (d_a2 * d_a2 * smoothness + d_align * Float32(0.22)) * self.dyn_reflect_intensity * ao_ibl * Float32(255.0)
                                    spec_rgb = spec_rgb + Vec3(fe.x * self.cpu_dyn_lobe_rgb.x * d_lobe, fe.y * self.cpu_dyn_lobe_rgb.y * d_lobe, fe.z * self.cpu_dyn_lobe_rgb.z * d_lobe)
                                var gloss = max(Float32(0.0), min(Float32(1.0), (Float32(0.58) - px_rough) * Float32(2.1)))
                                var fe_avg = max(Float32(0.0), min(Float32(1.0), (fe.x + fe.y + fe.z) * Float32(0.33333334)))
                                var water_boost = Float32(1.35) if self.water_wave_normals else Float32(1.0)
                                var ssr_f = max(Float32(0.0), min(Float32(1.0), gloss * gloss * (fe_avg * Float32(0.72) + Float32(0.28)) * water_boost))
                                surf_refl = UInt32(min(127, max(0, Int(ssr_f * Float32(127.0) + Float32(0.5)))))
                                var om_nv = 1.0 - ndotv
                                var om_nv2 = om_nv * om_nv
                                var back_sun = 0.55 + 0.45 * max(Float32(0.0), -dot(view_dir, self.cpu_to_key))
                                var rim_w = (om_nv2 * om_nv2) * back_sun * ao_ibl * 0.14
                                light = light + self.cpu_key_color * rim_w
                                for li in range(len(self.cpu_lights)):
                                    var pl = self.cpu_lights[li]
                                    var l0 = pl.position - world_pos
                                    var seg_len2 = pl.seg_delta.length_squared()
                                    var is_tube = seg_len2 > Float32(0.0001)
                                    var t_diff = max(Float32(0.0), min(Float32(1.0), -dot(l0, pl.seg_delta) / seg_len2)) if is_tube else Float32(0.0)
                                    var offset_l = l0 + pl.seg_delta * t_diff
                                    var d2 = offset_l.length_squared()
                                    var r2 = max(Float32(0.01), pl.radius * pl.radius)
                                    if d2 < r2:
                                        var att0 = 1.0 - d2 / r2
                                        var att = att0 * att0 * pl.intensity
                                        var ldir = offset_l.normalized()
                                        var ndotl = max(Float32(0.0), dot(n, ldir))
                                        var lambert = 0.24 + ndotl * 0.76
                                        light = light + pl.color * (att * lambert * ao * kd)
                                    var sp_vec = offset_l
                                    var sp_d2 = d2
                                    var tube_norm = Float32(1.0)
                                    if is_tube:
                                        var r_dot_sd = dot(refl_dir, pl.seg_delta)
                                        var denom_k = max(Float32(0.0001), seg_len2 - r_dot_sd * r_dot_sd)
                                        var t_spec = max(Float32(0.0), min(Float32(1.0), (dot(refl_dir, l0) * r_dot_sd - dot(l0, pl.seg_delta)) / denom_k))
                                        sp_vec = l0 + pl.seg_delta * t_spec
                                        sp_d2 = sp_vec.length_squared()
                                        var sp_dist = sqrt(max(Float32(0.0001), sp_d2))
                                        var a_pr = max(max(alpha_r, Float32(0.02)), min(Float32(1.0), alpha_r + sqrt(seg_len2) * Float32(0.25) / sp_dist))
                                        var ratio_a = max(alpha_r, Float32(0.02)) / a_pr
                                        tube_norm = ratio_a * ratio_a
                                    if sp_d2 < r2:
                                        var s_dir = sp_vec.normalized()
                                        var s_ndl = max(Float32(0.0), dot(n, s_dir))
                                        if s_ndl > Float32(0.0):
                                            var sf0 = 1.0 - sp_d2 / r2
                                            var s_att = sf0 * sf0 * pl.intensity * tube_norm
                                            var h_l = (view_dir + s_dir).normalized()
                                            var ndoth_l = max(Float32(0.0), dot(n, h_l))
                                            var tx_hl = dot(tan_dir, h_l) / ax
                                            var by_hl = dot(bit_dir, h_l) / ay
                                            var ellip_l = tx_hl * tx_hl + by_hl * by_hl + ndoth_l * ndoth_l
                                            var d_l = Float32(1.0) / max(Float32(0.0001), Float32(3.14159265) * ax * ay * ellip_l * ellip_l)
                                            var brdf_l = min(Float32(6.0), d_l * 0.25 * s_ndl) * ms_comp * s_att * 255.0
                                            spec_rgb = spec_rgb + Vec3(f0.x * pl.color.x * brdf_l, f0.y * pl.color.y * brdf_l, f0.z * pl.color.z * brdf_l)
                            var base = (UInt32(255) << 24) | (UInt32(red) << 16) | (UInt32(green) << 8) | UInt32(blue)
                            var clearness = 1.0 - fog
                            var raw_r = max(Float32(0.0), (Float32(red) * (light.x + emissive * 1.25) + spec_rgb.x) * clearness + fog_r_f * fog)
                            var raw_g = max(Float32(0.0), (Float32(green) * (light.y + emissive * 1.25) + spec_rgb.y) * clearness + fog_g_f * fog)
                            var raw_b = max(Float32(0.0), (Float32(blue) * (light.z + emissive * 1.25) + spec_rgb.z) * clearness + fog_b_f * fog)
                            var shaded = (base & 0xFF000000) | self._grade_hdr_pixel(raw_r, raw_g, raw_b)
                            var tex_a = Int((sampled >> 24) & 255)
                            if self.vfx_erode_threshold > Float32(0.0):
                                var er = max(Float32(0.0), min(Float32(0.96), self.vfx_erode_threshold))
                                var er_t = max(Float32(0.0), min(Float32(1.0), (Float32(tex_a) / Float32(255.0) - er) / max(Float32(0.06), Float32(1.0) - er)))
                                tex_a = max(0, min(255, Int(er_t * er_t * (Float32(3.0) - Float32(2.0) * er_t) * Float32(255.0))))
                            var effective_alpha = Int(alpha) * Int((tint >> 24) & 255) // 255
                            effective_alpha = effective_alpha * tex_a // 255
                            if self.vfx_soft_inv_dist > Float32(0.0) and effective_alpha > 0:
                                var dz_t = max(Float32(0.0), min(Float32(1.0), (self.depth[index] - view_depth) * self.vfx_soft_inv_dist))
                                var near_t = max(Float32(0.0), min(Float32(1.0), (view_depth - Float32(0.22)) * Float32(0.85)))
                                var soft_t = dz_t * near_t
                                var soft_h = soft_t * soft_t * (Float32(3.0) - Float32(2.0) * soft_t)
                                effective_alpha = max(0, min(255, Int(Float32(effective_alpha) * soft_h)))
                            if effective_alpha > 0:
                                self.pixels_shaded += 1
                                var op = Float32(effective_alpha) / 255.0
                                var h_idx = index * 3
                                if self.vfx_refract_strength > Float32(0.0):
                                    var wu = u - floor(u) if textured else Float32(0.5)
                                    var wv = v - floor(v) if textured else Float32(0.5)
                                    var r_amp = self.vfx_refract_strength * op
                                    var dx_b = UInt32(max(0, min(255, Int(max(Float32(-14.0), min(Float32(14.0), (wu - 0.5) * r_amp)) + 128.5))))
                                    var dy_b = UInt32(max(0, min(255, Int(max(Float32(-14.0), min(Float32(14.0), (wv - 0.5) * r_amp)) + 128.5))))
                                    self.cpu_refract[index] = UInt32(0x00010000) | (dy_b << 8) | dx_b
                                if self.additive_blend:
                                    var prev_px = self.pixels[index]
                                    var ar = clamp_byte(Float32((prev_px >> 16) & 255) + Float32((shaded >> 16) & 255) * op)
                                    var ag = clamp_byte(Float32((prev_px >> 8) & 255) + Float32((shaded >> 8) & 255) * op)
                                    var ab = clamp_byte(Float32(prev_px & 255) + Float32(shaded & 255) * op)
                                    self.pixels[index] = rgb(ar, ag, ab)
                                    self.hdr_rgb[h_idx] += raw_r * op
                                    self.hdr_rgb[h_idx + 1] += raw_g * op
                                    self.hdr_rgb[h_idx + 2] += raw_b * op
                                else:
                                    self.blend_pixel(index, shaded, UInt8(effective_alpha))
                                    if effective_alpha == 255:
                                        self.hdr_rgb[h_idx] = raw_r
                                        self.hdr_rgb[h_idx + 1] = raw_g
                                        self.hdr_rgb[h_idx + 2] = raw_b
                                    else:
                                        var inv_op = 1.0 - op
                                        var glint_r = min(Float32(520.0), spec_rgb.x * clearness) * inv_op * Float32(0.88) if per_pixel else Float32(0.0)
                                        var glint_g = min(Float32(520.0), spec_rgb.y * clearness) * inv_op * Float32(0.88) if per_pixel else Float32(0.0)
                                        var glint_b = min(Float32(520.0), spec_rgb.z * clearness) * inv_op * Float32(0.88) if per_pixel else Float32(0.0)
                                        self.hdr_rgb[h_idx] = self.hdr_rgb[h_idx] * inv_op + raw_r * op + glint_r
                                        self.hdr_rgb[h_idx + 1] = self.hdr_rgb[h_idx + 1] * inv_op + raw_g * op + glint_g
                                        self.hdr_rgb[h_idx + 2] = self.hdr_rgb[h_idx + 2] * inv_op + raw_b * op + glint_b
                                    if per_pixel and effective_alpha < 255:
                                        var rem = (1.0 - op) * 0.88 * clearness
                                        var cur_px = self.pixels[index]
                                        var gr = clamp_byte(Float32((cur_px >> 16) & 255) + min(Float32(235.0), spec_rgb.x) * rem)
                                        var gg = clamp_byte(Float32((cur_px >> 8) & 255) + min(Float32(235.0), spec_rgb.y) * rem)
                                        var gb = clamp_byte(Float32(cur_px & 255) + min(Float32(235.0), spec_rgb.z) * rem)
                                        self.pixels[index] = rgb(gr, gg, gb)
                                self.gbuf_norm_rough[index] |= UInt32(1) << 31
                                var spec_peak = max(spec_rgb.x, max(spec_rgb.y, spec_rgb.z))
                                var bloom_emissive = emissive
                                var bloom_base = base
                                if spec_peak > 135.0:
                                    var spec_em = min(Float32(2.4), (spec_peak - 135.0) / 130.0)
                                    if spec_em > bloom_emissive:
                                        bloom_emissive = spec_em
                                        bloom_base = rgb(clamp_byte(spec_rgb.x * 0.72), clamp_byte(spec_rgb.y * 0.72), clamp_byte(spec_rgb.z * 0.72))
                                self.record_bloom(index % self.width, index // self.width, bloom_base, bloom_emissive, 255 if (per_pixel and spec_peak > 135.0) else effective_alpha)
                                if effective_alpha >= 128 and not self.no_depth_write and not self.additive_blend:
                                    self.depth[index] = view_depth
                                    var eff_refl = UInt32(min(127, max(0, Int(Float32(surf_refl) * clearness + 0.5))))
                                    self.gbuf_norm_rough[index] = (
                                        surf_ox
                                        | (surf_oz << 8)
                                        | (surf_rough << 16)
                                        | (eff_refl << 24)
                                        | (UInt32(1) << 31)
                                    )
                                    if a.has_obj_vel and b.has_obj_vel and c.has_obj_vel:
                                        var ivx = (weight_a * a.vel_x * a.inverse_z + weight_b * b.vel_x * b.inverse_z + weight_c * c.vel_x * c.inverse_z) * view_depth
                                        var ivy = (weight_a * a.vel_y * a.inverse_z + weight_b * b.vel_y * b.inverse_z + weight_c * c.vel_y * c.inverse_z) * view_depth
                                        var cvx = max(Float32(-250.0), min(Float32(250.0), ivx))
                                        var cvy = max(Float32(-250.0), min(Float32(250.0), ivy))
                                        var qx = UInt32(max(1, min(65535, Int(cvx * Float32(64.0) + Float32(32768.5)))))
                                        var qy = UInt32(max(1, min(65535, Int(cvy * Float32(64.0) + Float32(32768.5)))))
                                        self.cpu_velocity[index] = qx | (qy << 16)
                                    else:
                                        self.cpu_velocity[index] = 0
                        else:
                            self.pixels_depth_rejected += 1
                weight_a += step_a_x
                weight_b += step_b_x
                index += 1
            row_a += step_a_y
            row_b += step_b_y

    def draw_triangle(mut self, a: RenderVertex, b: RenderVertex, c: RenderVertex, mut texture: List[UInt32], texture_width: Int = 0, texture_height: Int = 0, alpha: UInt8 = 255, double_sided: Bool = False):
        var near = Float32(0.06)
        var camera_a = self.camera_vertex(a)
        var camera_b = self.camera_vertex(b)
        var camera_c = self.camera_vertex(c)
        if camera_a.position.z >= near and camera_b.position.z >= near and camera_c.position.z >= near:
            self.raster_triangle(self.project(camera_a), self.project(camera_b), self.project(camera_c), texture, texture_width, texture_height, alpha, double_sided)
            return
        var polygon = List[CameraVertex]()
        polygon.append(camera_a)
        polygon.append(camera_b)
        polygon.append(camera_c)
        var clipped = List[CameraVertex]()
        for index in range(len(polygon)):
            var current = polygon[index]
            var previous = polygon[(index + len(polygon) - 1) % len(polygon)]
            var current_inside = current.position.z >= near
            var previous_inside = previous.position.z >= near
            if current_inside != previous_inside:
                var amount = (near - previous.position.z) / (current.position.z - previous.position.z)
                clipped.append(previous.interpolate(current, amount))
            if current_inside:
                clipped.append(current)
        if len(clipped) < 3:
            self.culled_triangles += 1
            return
        for index in range(1, len(clipped) - 1):
            self.raster_triangle(self.project(clipped[0]), self.project(clipped[index]), self.project(clipped[index + 1]), texture, texture_width, texture_height, alpha, double_sided)

    def fill_gpu_clipped(self, first_slot: Int, a: RenderVertex, b: RenderVertex, c: RenderVertex, offset: Int, texture_width: Int, texture_height: Int, alpha: UInt8, double_sided: Bool):
        """Near-plane clip a triangle into its two reserved slots. Any thread may call it."""
        var near = Float32(0.06)
        var polygon = List[CameraVertex]()
        polygon.append(self.camera_vertex(a))
        polygon.append(self.camera_vertex(b))
        polygon.append(self.camera_vertex(c))
        var clipped = List[CameraVertex]()
        for index in range(3):
            var current = polygon[index]
            var previous = polygon[(index + 2) % 3]
            var current_inside = current.position.z >= near
            var previous_inside = previous.position.z >= near
            if current_inside != previous_inside:
                var amount = (near - previous.position.z) / (current.position.z - previous.position.z)
                clipped.append(previous.interpolate(current, amount))
            if current_inside:
                clipped.append(current)
        for index in range(1, len(clipped) - 1):
            self.fill_gpu_slot(first_slot + index - 1, self.project(clipped[0]), self.project(clipped[index]), self.project(clipped[index + 1]), offset, texture_width, texture_height, alpha, double_sided)

    def draw_line_3d(mut self, start: Vec3, end: Vec3, red: UInt8, green: UInt8, blue: UInt8):
        var a = self.project(self.camera_vertex(RenderVertex(start)))
        var b = self.project(self.camera_vertex(RenderVertex(end)))
        if self.camera_vertex(RenderVertex(start)).position.z < 0.06 or self.camera_vertex(RenderVertex(end)).position.z < 0.06:
            return
        if self.gpu.active:
            self.emit_gpu_line(a, b, rgb(red, green, blue) | UInt32(0xFF000000))
            return
        self.plot_line(a.x, a.y, b.x, b.y, rgb(red, green, blue))

    def emit_gpu_line(mut self, a: ScreenVertex, b: ScreenVertex, color: UInt32):
        """A depth-tested screen-space quad about one pixel wide per 720 lines."""
        var dx = b.x - a.x
        var dy = b.y - a.y
        var length = sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return
        var half_width = max(Float32(0.5), Float32(self.height) / 1440.0)
        var offset_x = -dy / length * half_width
        var offset_y = dx / length * half_width
        var corners = List[ScreenVertex]()
        for index in range(4):
            var corner = a.copy() if index < 2 else b.copy()
            var side = Float32(1) if index == 0 or index == 3 else Float32(-1)
            corner.x += offset_x * side
            corner.y += offset_y * side
            # A slight pull toward the camera keeps lines on surfaces from flickering.
            corner.inverse_z *= 1.002
            corner.color = color
            corner.light = Vec3(1, 1, 1)
            corner.fog = 0
            corner.emissive = 0
            corners.append(corner)
        var empty = List[UInt32]()
        self.raster_triangle(corners[0], corners[1], corners[2], empty, 0, 0, 255, True)
        self.raster_triangle(corners[0], corners[2], corners[3], empty, 0, 0, 255, True)

    def plot_line(mut self, start_x: Float32, start_y: Float32, end_x: Float32, end_y: Float32, color: UInt32):
        var steps = max(1, Int(max(abs(end_x - start_x), abs(end_y - start_y))))
        for index in range(steps + 1):
            var amount = Float32(index) / Float32(steps)
            var x = Int(start_x + (end_x - start_x) * amount)
            var y = Int(start_y + (end_y - start_y) * amount)
            if x >= 0 and y >= 0 and x < self.width and y < self.height:
                self.pixels[y * self.width + x] = color

    def draw_rectangle(mut self, x: Int, y: Int, width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8 = 255):
        if self.hud_recording:
            self.record_hud(0, x, y, width, height, 0, red, green, blue, alpha)
            return
        if x == 0 and y == 0 and width == 1280 and height == 720:
            # Full-screen dimming overlays reach the real screen edges.
            if self.hud_drawing:
                self.add_hud_rectangle(0, 0, self.width, self.height)
            self.fill_rectangle(0, 0, self.width, self.height, red, green, blue, alpha)
            return
        var left = self.ux(x)
        var top = self.uy(y)
        var w = self.ux(x + width) - left
        var h = self.uy(y + height) - top
        if self.hud_drawing:
            self.add_hud_rectangle(left, top, w, h)
        self.fill_rectangle(left, top, w, h, red, green, blue, alpha)

    def fill_rectangle(mut self, x: Int, y: Int, width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8 = 255):
        var color = rgb(red, green, blue)
        var left = max(0, x)
        var right = min(self.width, x + width)
        if self.hud_drawing:
            self.fill_hud_rectangle(left, max(0, y), right, min(self.height, y + height), color, alpha)
            return
        for py in range(max(0, y), min(self.height, y + height)):
            for px in range(left, right):
                self.blend_pixel(py * self.width + px, color, alpha)

    def fill_hud_rectangle(mut self, left: Int, top: Int, right: Int, bottom: Int, color: UInt32, alpha: UInt8):
        """Premultiplied over for a whole span, matching `blend_hud_pixel` exactly."""
        if alpha == 0 or right <= left:
            return
        var amount = UInt32(alpha)
        var inverse = 255 - amount
        var red = ((color >> 16) & 255) * amount
        var green = ((color >> 8) & 255) * amount
        var blue = (color & 255) * amount
        # Most of the layer is still empty when a panel lands on it.
        var over_nothing = amount << 24 | (red // 255) << 16 | (green // 255) << 8 | (blue // 255)
        var pixels = self.hud_pixels.unsafe_ptr()
        for y in range(top, bottom):
            var row = y * self.width
            for index in range(row + left, row + right):
                var under = pixels[unsafe_offset=index]
                if under == 0 or inverse == 0:
                    pixels[unsafe_offset=index] = over_nothing
                    continue
                var a = amount + ((under >> 24) * inverse) // 255
                var r = (red + ((under >> 16) & 255) * inverse) // 255
                var g = (green + ((under >> 8) & 255) * inverse) // 255
                var b = (blue + (under & 255) * inverse) // 255
                pixels[unsafe_offset=index] = a << 24 | r << 16 | g << 8 | b

    def draw_rectangle_rounded(mut self, ui_left: Int, ui_top: Int, ui_width: Int, ui_height: Int, roundness: Float32, red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8 = 255):
        if self.hud_recording:
            self.record_hud(1, ui_left, ui_top, ui_width, ui_height, Int(roundness * 1000.0 + 0.5), red, green, blue, alpha)
            return
        var x = self.ux(ui_left)
        var y = self.uy(ui_top)
        var width = self.ux(ui_left + ui_width) - x
        var height = self.uy(ui_top + ui_height) - y
        if self.hud_drawing:
            self.add_hud_rectangle(x, y, width, height)
        var radius = max(0, min(min(width, height) // 2, Int(roundness * Float32(min(width, height)))))
        self.fill_rectangle(x + radius, y, width - radius * 2, height, red, green, blue, alpha)
        self.fill_rectangle(x, y + radius, radius, height - radius * 2, red, green, blue, alpha)
        self.fill_rectangle(x + width - radius, y + radius, radius, height - radius * 2, red, green, blue, alpha)
        # Each corner is a quarter disc centred `radius` in from both edges.
        # (The right and bottom corners used to reuse the top-left distances,
        # which drew them inside out.)
        var r2 = Float32(radius) * Float32(radius)
        for py in range(radius):
            # Whole spans, not single pixels: each row of a corner is inside the disc
            # from `first` to the straight edge, and the far corners mirror it.
            var near_y = Float32(radius - py) - 0.5
            var far_y = Float32(py) + 0.5
            var first = radius
            var first_far = radius
            for px in range(radius):
                var near_x = Float32(radius - px) - 0.5
                if first == radius and near_x * near_x + near_y * near_y <= r2:
                    first = px
                if first_far == radius and near_x * near_x + far_y * far_y <= r2:
                    first_far = px
            var span = radius - first
            var span_far = radius - first_far
            self.fill_rectangle(x + first, y + py, span, 1, red, green, blue, alpha)
            self.fill_rectangle(x + width - radius, y + py, span, 1, red, green, blue, alpha)
            self.fill_rectangle(x + first_far, y + height - radius + py, span_far, 1, red, green, blue, alpha)
            self.fill_rectangle(x + width - radius, y + height - radius + py, span_far, 1, red, green, blue, alpha)

    def draw_rectangle_lines(mut self, x: Int, y: Int, width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8):
        self.draw_rectangle(x, y, width, 1, red, green, blue)
        self.draw_rectangle(x, y + height - 1, width, 1, red, green, blue)
        self.draw_rectangle(x, y, 1, height, red, green, blue)
        self.draw_rectangle(x + width - 1, y, 1, height, red, green, blue)

    def draw_circle(mut self, ui_x: Int, ui_y: Int, radius: Float32, red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8 = 255):
        if self.hud_recording:
            self.record_hud(2, ui_x, ui_y, Int(radius), 0, 0, red, green, blue, alpha)
            return
        var x = self.ux(ui_x)
        var y = self.uy(ui_y)
        var r = max(1, Int(radius * self.ui_scale + 0.5))
        if self.hud_drawing:
            self.add_hud_rectangle(x - r, y - r, r * 2 + 1, r * 2 + 1)
        var color = rgb(red, green, blue)
        for py in range(-r, r + 1):
            for px in range(-r, r + 1):
                if px * px + py * py > r * r:
                    continue
                var sx = x + px
                var sy = y + py
                if sx >= 0 and sy >= 0 and sx < self.width and sy < self.height:
                    self.blend_pixel(sy * self.width + sx, color, alpha)

    def draw_nine_slice(mut self, slice_index: Int, ui_left: Int, ui_top: Int, ui_width: Int, ui_height: Int, alpha: UInt8 = 255, corner_scale: Float32 = 1):
        """Package 3.12: a nine-slice image (ui/nine_slice.mojo) stretched over
        a canvas rectangle. Corners keep their size times the UI scale (and
        `corner_scale`), edges stretch along one axis, the centre along both."""
        if slice_index < 0 or slice_index >= len(self.nine_slices) or alpha == 0:
            return
        if self.hud_recording:
            self.record_hud(7, ui_left, ui_top, ui_width, ui_height, slice_index + 64 * Int(corner_scale * 256.0 + 0.5), 0, 0, 0, alpha)
            return
        var x = self.ux(ui_left)
        var y = self.uy(ui_top)
        var width = self.ux(ui_left + ui_width) - x
        var height = self.uy(ui_top + ui_height) - y
        if width <= 0 or height <= 0:
            return
        if self.hud_drawing:
            self.add_hud_rectangle(x, y, width, height)
        var scale = self.ui_scale * corner_scale
        var columns = slice_map(width, self.nine_slices[slice_index].width, self.nine_slices[slice_index].left, self.nine_slices[slice_index].right, scale)
        var rows = slice_map(height, self.nine_slices[slice_index].height, self.nine_slices[slice_index].top, self.nine_slices[slice_index].bottom, scale)
        var source_width = self.nine_slices[slice_index].width
        # A copy (48x48 at most): blend_pixel takes `mut self`.
        var image = self.nine_slices[slice_index].pixels.copy()
        var source = image.unsafe_ptr()
        var amount = UInt32(alpha)
        if self.hud_drawing:
            # The HUD layer: blend_hud_pixel's premultiplied over, inlined
            # on raw rows (the same integers, so the same pixels).
            var target = self.hud_pixels.unsafe_ptr()
            var column_ptr = columns.unsafe_ptr()
            for py in range(max(0, -y), min(height, self.height - y)):
                var row = rows[py] * source_width
                var target_row = (y + py) * self.width + x
                for px in range(max(0, -x), min(width, self.width - x)):
                    var texel = source[unsafe_offset=row + column_ptr[unsafe_offset=px]]
                    var coverage = Int((texel >> 24) * amount // 255)
                    if coverage == 0:
                        continue
                    var previous = target[unsafe_offset=target_row + px]
                    var inverse = 255 - coverage
                    var a = coverage + (Int(previous >> 24) * inverse) // 255
                    var red = (Int((texel >> 16) & 255) * coverage + Int((previous >> 16) & 255) * inverse) // 255
                    var green = (Int((texel >> 8) & 255) * coverage + Int((previous >> 8) & 255) * inverse) // 255
                    var blue = (Int(texel & 255) * coverage + Int(previous & 255) * inverse) // 255
                    target[unsafe_offset=target_row + px] = UInt32(a) << 24 | UInt32(red) << 16 | UInt32(green) << 8 | UInt32(blue)
            _ = len(image)
            _ = len(columns)
            return
        for py in range(max(0, -y), min(height, self.height - y)):
            var row = rows[py] * source_width
            var target_row = (y + py) * self.width
            for px in range(max(0, -x), min(width, self.width - x)):
                var texel = source[unsafe_offset=row + columns[px]]
                var coverage = (texel >> 24) * amount // 255
                if coverage == 0:
                    continue
                self.blend_pixel(target_row + x + px, texel & 0xFFFFFF, UInt8(coverage))
        _ = len(image)  # keep the copy alive past the raw pointer's last use

    def draw_text(mut self, text: String, ui_left: Int, ui_top: Int, size: Int, red: UInt8 = 240, green: UInt8 = 240, blue: UInt8 = 235, align: Int = 0, fit: Int = 0):
        """Text on the UI canvas. `ui_top` is the top of the capitals; `size`
        is pixels per em on the canvas. align: 0 left, 1 centre, 2 right
        (of `ui_left`). `fit` > 0 is the room on the canvas: wider text
        shrinks to fit, down to 80% of its base size. Sizes of 30 and up use
        the display face."""
        if self.hud_recording:
            self.hud_texts.append(text)
            self.record_hud(3, ui_left, ui_top, size, align + fit * 4, len(self.hud_texts) - 1, red, green, blue, 255)
            return
        var font_index = 1 if size >= 30 else 0
        if not self.font_ready(font_index):
            self.draw_bitmap_text(text, ui_left, ui_top, size, red, green, blue, align)
            return
        var px = self.em_size(size) * self.text_scale * self.ui_scale
        var width = self.measure_font(font_index, text, px)
        if fit > 0 and width > Float32(fit) * self.ui_scale:
            var floor_px = self.em_size(size) * self.ui_scale * 0.8
            px = max(floor_px, px * Float32(fit) * self.ui_scale / width)
            width = self.measure_font(font_index, text, px)
        var x = Float32(self.ux(ui_left))
        if align == 1:
            x -= width * 0.5
        elif align == 2:
            x -= width
        var cap = self.font_cap(font_index) * px / self.font_bake(font_index)
        var baseline = Float32(self.uy(ui_top)) + cap
        # Illuminated HUD/menu typography: crisp black micro-outline, directional
        # black drop shadow, and soft SDF optical bloom over transparent backdrops.
        var outline = max(Float32(1.10), px * (Float32(0.082) if font_index == 1 else Float32(0.068)))
        var shadow = max(Float32(1.85), px * Float32(0.105))
        if self.hud_drawing:
            self.draw_text_sprite(font_index, text, x, baseline, px, width, outline, shadow, rgb(red, green, blue))
            return
        self.draw_sdf_ink(font_index, text, x, baseline, px, outline, shadow, rgb(red, green, blue))

    def draw_sdf_ink(mut self, font_index: Int, text: String, x: Float32, baseline: Float32, px: Float32, outline: Float32, shadow: Float32, color: UInt32):
        var cr = Int((color >> 16) & 255)
        var cg = Int((color >> 8) & 255)
        var cb = Int(color & 255)
        # 1. Soft black ambient shadow penumbra + crisp directional black drop shadow
        self.draw_sdf_run(font_index, text, x + shadow * Float32(0.85), baseline + shadow * Float32(0.95), px, rgb(0, 0, 0), 170, Float32(0), rgb(0, 0, 0), max(Float32(2.2), px * Float32(0.14)))
        self.draw_sdf_run(font_index, text, x + shadow, baseline + shadow, px, rgb(0, 0, 0), 225, max(Float32(0.75), outline * Float32(0.65)), rgb(0, 0, 0), Float32(0))
        # 2. Luminous SDF optical light halo around the glyph ("text with light")
        var gr = UInt8(min(255, max(72 if cb > 200 else 0, cr + 28)))
        var gg = UInt8(min(255, cg + 24))
        var gb = UInt8(min(255, cb + 18))
        self.draw_sdf_run(font_index, text, x, baseline, px, rgb(gr, gg, gb), 84, Float32(0), rgb(gr, gg, gb), max(Float32(2.6), px * Float32(0.175)))
        # 3. Jet-black micro-stroke contour + incandescent self-lit glyph core
        var lr = UInt8(min(255, cr + (255 - cr) * 26 // 100))
        var lg = UInt8(min(255, cg + (252 - cg) * 24 // 100))
        var lb = UInt8(min(255, cb + (242 - cb) * 20 // 100))
        self.draw_sdf_run(font_index, text, x, baseline, px, rgb(lr, lg, lb), 255, max(Float32(1.05), outline), rgb(2, 4, 8), Float32(0))

    def text_sprite_slot(mut self, key: String, font_index: Int, text: String, fraction_x: Float32, fraction_y: Float32, px: Float32, width: Float32, outline: Float32, shadow: Float32, color: UInt32, reach: Int, above: Int) raises -> Int:
        """Find this line's finished pixels, drawing them once if they are new.

        Sampling a distance field costs about 8 ms for a full HUD at 3440x1440, and
        the HUD redraws whenever one number changes. Only the changed line pays now.
        """
        if key in self.text_sprite_slots:
            return self.text_sprite_slots[key]
        if len(self.text_sprites) >= TEXT_SPRITE_LIMIT:
            self.text_sprites.clear()
            self.text_sprite_slots.clear()
        if len(self.text_canvas) != len(self.hud_pixels):
            self.text_canvas = List[UInt32](length=len(self.hud_pixels), fill=0)
        var sprite_width = min(self.width, Int(width) + reach * 2 + 2)
        var sprite_height = min(self.height, Int(px * 1.5) + reach * 2 + 2)
        swap(self.hud_pixels, self.text_canvas)
        self.draw_sdf_ink(font_index, text, Float32(reach) + fraction_x, Float32(above) + fraction_y, px, outline, shadow, color)
        swap(self.hud_pixels, self.text_canvas)
        var pixels = List[UInt32](length=sprite_width * sprite_height, fill=0)
        for y in range(sprite_height):
            var row = Int(self.text_canvas.unsafe_ptr()) + y * self.width * 4
            _ = external_call["memcpy", Int](Int(pixels.unsafe_ptr()) + y * sprite_width * 4, row, sprite_width * 4)
            _ = external_call["memset", Int](row, Int32(0), sprite_width * 4)
        self.text_sprites.append(TextSprite(sprite_width, sprite_height, pixels^))
        self.text_sprite_slots[key] = len(self.text_sprites) - 1
        return len(self.text_sprites) - 1

    def draw_text_sprite(mut self, font_index: Int, text: String, x: Float32, baseline: Float32, px: Float32, width: Float32, outline: Float32, shadow: Float32, color: UInt32):
        """Lay one line of text into the HUD layer from its finished pixels."""
        var reach = Int(outline + shadow + px * 0.34) + 5
        var above = Int(px * 1.15) + reach
        var whole_x = Int(floor(x))
        var whole_y = Int(floor(baseline))
        var fraction_x = x - Float32(whole_x)
        var fraction_y = baseline - Float32(whole_y)
        var key = String(font_index) + ":" + String(Int(px * 256)) + ":" + String(Int(fraction_x * 256)) + ":" + String(Int(fraction_y * 256)) + ":" + String(color) + ":" + text
        var slot: Int
        try:
            slot = self.text_sprite_slot(key, font_index, text, fraction_x, fraction_y, px, width, outline, shadow, color, reach, above)
        except:
            self.draw_sdf_ink(font_index, text, x, baseline, px, outline, shadow, color)
            return
        var left = whole_x - reach
        var top = whole_y - above
        var sprite_width = self.text_sprites[slot].width
        var sprite_height = self.text_sprites[slot].height
        self.add_hud_rectangle(left, top, sprite_width, sprite_height)
        var source = self.text_sprites[slot].pixels.unsafe_ptr()
        var target = self.hud_pixels.unsafe_ptr()
        for y in range(max(0, -top), min(sprite_height, self.height - top)):
            var from_row = y * sprite_width
            var to_row = (top + y) * self.width + left
            for column in range(max(0, -left), min(sprite_width, self.width - left)):
                var ink = source[unsafe_offset=from_row + column]
                var coverage = ink >> 24
                if coverage == 0:
                    continue
                var under = target[unsafe_offset=to_row + column]
                if coverage == 255 or under == 0:
                    target[unsafe_offset=to_row + column] = ink
                    continue
                # Premultiplied over: the same result as blending the line straight in.
                var inverse = 255 - coverage
                var alpha = coverage + ((under >> 24) * inverse) // 255
                var red = ((ink >> 16) & 255) + (((under >> 16) & 255) * inverse) // 255
                var green = ((ink >> 8) & 255) + (((under >> 8) & 255) * inverse) // 255
                var blue = (ink & 255) + ((under & 255) * inverse) // 255
                target[unsafe_offset=to_row + column] = alpha << 24 | min(red, UInt32(255)) << 16 | min(green, UInt32(255)) << 8 | min(blue, UInt32(255))

    def text_width(self, text: String, size: Int) -> Int:
        """Width on the UI canvas of `draw_text` at this size."""
        var font_index = 1 if size >= 30 else 0
        if not self.font_ready(font_index):
            return text.byte_length() * 6 * max(1, size // 8)
        var em = self.em_size(size) * self.text_scale
        return Int(self.measure_font(font_index, text, em) + 0.5)

    def font_ready(self, index: Int) -> Bool:
        return self.display_font.valid if index == 1 else self.ui_font.valid

    def font_cap(self, index: Int) -> Float32:
        return self.display_font.cap_height if index == 1 else self.ui_font.cap_height

    def font_bake(self, index: Int) -> Float32:
        return self.display_font.bake_px if index == 1 else self.ui_font.bake_px

    def measure_font(self, index: Int, text: String, px: Float32) -> Float32:
        return self.display_font.measure(text, px) if index == 1 else self.ui_font.measure(text, px)

    def draw_sdf_run(mut self, font_index: Int, text: String, x: Float32, baseline: Float32, px: Float32, color: UInt32, opacity: Int, outline: Float32, outline_color: UInt32, softness: Float32):
        """One line of SDF glyphs. Coverage comes from the distance in screen
        pixels: 0.5 + d is a one-pixel anti-aliased edge; `outline` grows the
        shape by that many pixels in `outline_color`; `softness` widens the
        edge for shadows."""
        if font_index == 1:
            self.display_font.ensure_text(text)
        else:
            self.ui_font.ensure_text(text)
        var codes = decode_utf8(text)
        var pen = x
        for index in range(len(codes)):
            var glyph_index = self.ui_font.glyph_for(codes[index]) if font_index == 0 else self.display_font.glyph_for(codes[index])
            if glyph_index < 0:
                continue
            self.draw_sdf_glyph(font_index, glyph_index, pen, baseline, px, color, opacity, outline, outline_color, softness)
            var bake = self.font_bake(font_index)
            var advance = self.display_font.glyphs[glyph_index].advance if font_index == 1 else self.ui_font.glyphs[glyph_index].advance
            pen += advance * px / bake
            if index + 1 < len(codes):
                var kern = self.display_font.kern(codes[index], codes[index + 1]) if font_index == 1 else self.ui_font.kern(codes[index], codes[index + 1])
                pen += kern * px / bake

    def font_sample(self, font_index: Int, x: Float32, y: Float32) -> Float32:
        return self.display_font.sample(x, y) if font_index == 1 else self.ui_font.sample(x, y)

    def draw_sdf_glyph(mut self, font_index: Int, glyph_index: Int, pen: Float32, baseline: Float32, px: Float32, color: UInt32, opacity: Int, outline: Float32, outline_color: UInt32, softness: Float32):
        var glyph = self.display_font.glyphs[glyph_index] if font_index == 1 else self.ui_font.glyphs[glyph_index]
        if glyph.w == 0 or glyph.h == 0:
            return
        var spread = self.display_font.spread if font_index == 1 else self.ui_font.spread
        var scale = px / self.font_bake(font_index)  # screen px per bake px
        var left = pen + glyph.xoff * scale
        var top = baseline + glyph.yoff * scale
        var x0 = max(0, Int(floor(left)))
        var y0 = max(0, Int(floor(top)))
        var x1 = min(self.width, Int(floor(left + Float32(glyph.w) * scale)) + 1)
        var y1 = min(self.height, Int(floor(top + Float32(glyph.h) * scale)) + 1)
        # Distance units (0-255) to screen pixels.
        var to_screen = spread * scale / 127.0
        var edge = 0.5 + max(Float32(0), softness - 0.5)
        var width_edge = max(Float32(1), softness * 2)
        var fill_red = Int((color >> 16) & 255)
        var fill_green = Int((color >> 8) & 255)
        var fill_blue = Int(color & 255)
        var line_red = Int((outline_color >> 16) & 255)
        var line_green = Int((outline_color >> 8) & 255)
        var line_blue = Int(outline_color & 255)
        for sy in range(y0, y1):
            var v = (Float32(sy) + 0.5 - top) / scale
            if v < 0 or v > Float32(glyph.h):
                continue
            for sx in range(x0, x1):
                var u = (Float32(sx) + 0.5 - left) / scale
                if u < 0 or u > Float32(glyph.w):
                    continue
                var distance = (self.font_sample(font_index, Float32(glyph.x) + u, Float32(glyph.y) + v) - 128.0) * to_screen
                var fill = max(Float32(0), min(Float32(1), (distance + edge) / width_edge))
                var total = fill
                if outline > 0:
                    total = max(Float32(0), min(Float32(1), (distance + outline + 0.5)))
                if total <= 0.003:
                    continue
                var mixed = color
                if outline > 0 and total > 0:
                    var t = fill / total
                    var red = line_red + Int(Float32(fill_red - line_red) * t)
                    var green = line_green + Int(Float32(fill_green - line_green) * t)
                    var blue = line_blue + Int(Float32(fill_blue - line_blue) * t)
                    mixed = UInt32(red) << 16 | UInt32(green) << 8 | UInt32(blue)
                var alpha = UInt8(min(255, Int(total * Float32(opacity) + 0.5)))
                self.blend_pixel(sy * self.width + sx, mixed, alpha)

    def set_button_style(mut self, pad: Bool, sony: Bool):
        """Package 3.12: button glyphs in rich text follow the last device used."""
        self.button_pad = pad
        self.button_sony = sony

    def set_button_keys(mut self, keys: List[String]):
        """Package 3.15: keycaps show the keys the player bound."""
        var same = len(keys) == len(self.button_keys)
        if same:
            for index in range(len(keys)):
                if keys[index] != self.button_keys[index]:
                    same = False
        if not same:
            self.button_keys = keys.copy()
            self.button_keys_generation += 1

    def rich_text_lines(self, text: String, size: Int, max_width: Int) -> Int:
        """Package 3.22: how many lines `draw_rich_text` wraps `text` to at
        `max_width` canvas units (the caption box holds two)."""
        var base_face = 1 if size >= 30 else 0
        var px = self.em_size(size) * self.text_scale * self.ui_scale
        var layout = layout_rich(parse_rich(text), self.ui_font, self.display_font, px, base_face, Float32(max_width) * self.ui_scale, 1, self.button_pad, self.button_sony, self.button_keys)
        return max(1, layout.lines)

    def draw_rich_text(mut self, text: String, ui_left: Int, ui_top: Int, size: Int, max_width: Int = 0, align: Int = 0, red: UInt8 = 240, green: UInt8 = 240, blue: UInt8 = 235) -> Int:
        """Package 3.12: text with tags (text/rich_text.mojo: colour, display
        face, size, button glyphs), wrapped at `max_width` canvas units (0:
        one line unless '\\n'). `ui_top` is the top of the first line's
        capitals, as for `draw_text`; align 0 left, 1 centred on `ui_left`, 2
        right of it. Returns the height used on the canvas."""
        var base_face = 1 if size >= 30 else 0
        var px = self.em_size(size) * self.text_scale * self.ui_scale
        var spans = parse_rich(text)
        var layout = layout_rich(spans, self.ui_font, self.display_font, px, base_face, Float32(max_width) * self.ui_scale, align, self.button_pad, self.button_sony, self.button_keys)
        var height = Int(layout.height / self.ui_scale + 0.5)
        if self.hud_recording:
            self.hud_texts.append(text)
            # The button style rides in the alpha field, so a device change redraws.
            var style = UInt8((1 if self.button_pad else 0) + (2 if self.button_sony else 0) + 4 * (self.button_keys_generation % 64))
            self.record_hud(6, ui_left, ui_top, size, align + max_width * 4, len(self.hud_texts) - 1, red, green, blue, style)
            return height
        if not self.font_ready(base_face) or not self.font_ready(0):
            self.draw_bitmap_text(plain_text(spans), ui_left, ui_top, size, red, green, blue, align)
            return height
        var box = Float32(max_width) * self.ui_scale if max_width > 0 else layout.width
        var left = Float32(self.ux(ui_left))
        if align == 1:
            left -= box * 0.5
        elif align == 2:
            left -= box
        var top = Float32(self.uy(ui_top))
        var base = rgb(red, green, blue)
        for run in layout.runs:
            var x = left + run.x
            var baseline = top + run.baseline
            var color = UInt32(run.color) if run.color >= 0 else base
            if run.kind == SPAN_BUTTON:
                var cap = self.font_cap(0) * run.px / self.font_bake(0)
                var h = run.px * 1.15
                self.draw_button_glyph(run.action, x, baseline - cap - (h - cap) * 0.5, run.width, h)
                continue
            var outline = max(Float32(1.10), run.px * (Float32(0.082) if run.face == 1 else Float32(0.068)))
            var shadow = max(Float32(1.85), run.px * Float32(0.105))
            if self.hud_drawing:
                self.draw_text_sprite(run.face, run.text, x, baseline, run.px, run.width, outline, shadow, color)
            else:
                self.draw_sdf_ink(run.face, run.text, x, baseline, run.px, outline, shadow, color)
        return height

    def fill_round_screen(mut self, x: Float32, y: Float32, w: Float32, h: Float32, radius: Float32, color: UInt32, alpha: Int = 255):
        """An anti-aliased rounded box in screen pixels (a circle when the
        radius is half the size), for button glyphs."""
        var r = max(Float32(0), min(radius, min(w, h) * 0.5))
        var x0 = max(0, Int(floor(x)))
        var y0 = max(0, Int(floor(y)))
        var x1 = min(self.width, Int(floor(x + w)) + 1)
        var y1 = min(self.height, Int(floor(y + h)) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        if self.hud_drawing:
            self.add_hud_rectangle(x0, y0, x1 - x0, y1 - y0)
        var cx = x + w * 0.5
        var cy = y + h * 0.5
        var hx = w * 0.5 - r
        var hy = h * 0.5 - r
        for py in range(y0, y1):
            var qy = abs(Float32(py) + 0.5 - cy) - hy
            var dy = max(Float32(0), qy)
            for px in range(x0, x1):
                var qx = abs(Float32(px) + 0.5 - cx) - hx
                var dx = max(Float32(0), qx)
                # The rounded box's signed distance (negative inside).
                var distance = sqrt(dx * dx + dy * dy) + min(max(qx, qy), Float32(0)) - r
                var coverage = max(Float32(0), min(Float32(1), 0.5 - distance))
                if coverage <= 0.003:
                    continue
                self.blend_pixel(py * self.width + px, color, UInt8(min(255, Int(coverage * Float32(alpha) + 0.5))))

    def fill_triangle_up_screen(mut self, cx: Float32, top: Float32, size: Float32, color: UInt32):
        """An upward triangle outline (the PlayStation triangle)."""
        var thickness = max(Float32(1.5), size * 0.16)
        var rows = Int(size + 0.5)
        for row in range(rows):
            var t = Float32(row) / max(Float32(1), size)
            var half = size * 0.58 * t
            var y = floor(top) + Float32(row)
            if Float32(row) >= size - thickness:
                self.fill_round_screen(floor(cx - half), y, floor(half * 2 + 0.5), 1, 0, color)
            else:
                var edge = floor(min(thickness * 1.15, half) + 0.5)
                self.fill_round_screen(floor(cx - half), y, edge, 1, 0, color)
                self.fill_round_screen(floor(cx + half) - edge, y, edge, 1, 0, color)

    def draw_button_glyph(mut self, action: Int, x: Float32, y: Float32, w: Float32, h: Float32):
        """One button glyph in screen pixels with its top-left at (x, y): a
        keycap for the keyboard, else the controller button's own shape. The
        colours match the menu prompts' glyphs (render/interface.mojo)."""
        var shape = glyph_shape(action, self.button_pad)
        var label = glyph_label(action, self.button_pad, self.button_sony, self.button_keys)
        var label_px = h / Float32(1.15) * BUTTON_LABEL
        var dark = rgb(32, 40, 52)
        if shape == SHAPE_KEYCAP:
            var cap_h = h * 0.9
            self.fill_round_screen(x, y + h * 0.1, w, cap_h, h * 0.2, rgb(120, 128, 120))
            self.fill_round_screen(x, y, w, cap_h, h * 0.2, rgb(226, 230, 218))
            self.button_label(label, x + w * 0.5, y + cap_h * 0.5, label_px, rgb(30, 36, 40))
            return
        var button = pad_button(action)
        if shape == SHAPE_FACE:
            if self.button_sony:
                var symbol = face_color(button, True)
                self.fill_round_screen(x, y, h, h, h * 0.5, dark)
                if button == BUTTON_A:
                    self.button_label("×", x + h * 0.5, y + h * 0.5, h * 0.95, symbol)
                elif button == BUTTON_B:
                    self.fill_round_screen(x + h * 0.22, y + h * 0.22, h * 0.56, h * 0.56, h * 0.28, symbol)
                    self.fill_round_screen(x + h * 0.33, y + h * 0.33, h * 0.34, h * 0.34, h * 0.17, dark)
                elif button == BUTTON_X:
                    self.fill_round_screen(x + h * 0.26, y + h * 0.26, h * 0.48, h * 0.48, h * 0.04, symbol)
                    self.fill_round_screen(x + h * 0.35, y + h * 0.35, h * 0.30, h * 0.30, 0, dark)
                else:
                    self.fill_triangle_up_screen(x + h * 0.5, y + h * 0.24, h * 0.5, symbol)
                return
            self.fill_round_screen(x, y, h, h, h * 0.5, face_color(button, False))
            self.button_label(label, x + h * 0.5, y + h * 0.5, label_px, rgb(250, 250, 245))
            return
        if shape == SHAPE_PILL:
            self.fill_round_screen(x, y + h * 0.08, w, h * 0.84, h * 0.3, rgb(40, 48, 60))
            self.button_label(label, x + w * 0.5, y + h * 0.5, label_px, rgb(230, 236, 228))
            return
        if shape == SHAPE_DPAD:
            var light = rgb(210, 214, 204)
            var arm = h * 0.32
            self.fill_round_screen(x + (h - arm) * 0.5, y, arm, h, arm * 0.3, light)
            self.fill_round_screen(x, y + (h - arm) * 0.5, h, arm, arm * 0.3, light)
            var gold = rgb(255, 211, 112)
            var tip = arm * 0.7
            var mid = (h - tip) * 0.5
            if button == BUTTON_DPAD_UP:
                self.fill_round_screen(x + mid, y + arm * 0.15, tip, tip, tip * 0.2, gold)
            elif button == BUTTON_DPAD_RIGHT:
                self.fill_round_screen(x + h - arm * 0.15 - tip, y + mid, tip, tip, tip * 0.2, gold)
            elif button == BUTTON_DPAD_DOWN:
                self.fill_round_screen(x + mid, y + h - arm * 0.15 - tip, tip, tip, tip * 0.2, gold)
            else:
                self.fill_round_screen(x + arm * 0.15, y + mid, tip, tip, tip * 0.2, gold)
            return
        # A stick: a light ring around a dark cap.
        self.fill_round_screen(x, y, h, h, h * 0.5, rgb(210, 214, 204))
        self.fill_round_screen(x + h * 0.14, y + h * 0.14, h * 0.72, h * 0.72, h * 0.36, rgb(40, 48, 60))
        self.button_label(label, x + h * 0.5, y + h * 0.5, label_px * 0.85, rgb(230, 236, 228))

    def button_label(mut self, label: String, cx: Float32, cy: Float32, px: Float32, color: UInt32):
        """A label centred on (cx, cy) in the UI face, no shadow."""
        if label.byte_length() == 0 or not self.font_ready(0):
            return
        var width = self.measure_font(0, label, px)
        var cap = self.font_cap(0) * px / self.font_bake(0)
        self.draw_sdf_run(0, label, cx - width * 0.5, cy + cap * 0.5, px, color, 255, Float32(0), color, Float32(0))

    def draw_bitmap_text(mut self, text: String, ui_left: Int, ui_top: Int, size: Int, red: UInt8, green: UInt8, blue: UInt8, align: Int):
        """The original 5x7 font: the fallback when the font assets are missing."""
        var scale = max(1, Int(Float32(max(1, size // 8)) * self.ui_scale + 0.5))
        var cursor = self.ux(ui_left)
        var width = text.byte_length() * 6 * scale
        if align == 1:
            cursor -= width // 2
        elif align == 2:
            cursor -= width
        var y = self.uy(ui_top)
        var color = rgb(red, green, blue)
        if self.hud_drawing:
            color |= UInt32(0xFF000000)
            self.add_hud_rectangle(cursor, y, width, 7 * scale)
        for index in range(text.byte_length()):
            var code = Int(text.as_bytes()[index])
            if code == 32:
                cursor += 6 * scale
                continue
            var glyph = bitmap_glyph(code)
            for row in range(7):
                for column in range(5):
                    if (glyph >> UInt64(row * 5 + column)) & 1:
                        for sy in range(scale):
                            var py = y + row * scale + sy
                            if py < 0 or py >= self.height:
                                continue
                            for sx in range(scale):
                                var px = cursor + column * scale + sx
                                if px >= 0 and px < self.width:
                                    if self.hud_drawing:
                                        self.hud_pixels[py * self.width + px] = color
                                    else:
                                        self.pixels[py * self.width + px] = color
            cursor += 6 * scale


def box_blur_pass(source: List[Float32], mut target: List[Float32], width: Int, height: Int, horizontal: Bool):
    """One 5-tap box blur along rows or columns of an RGB float image."""
    for y in range(height):
        for x in range(width):
            var r: Float32 = 0
            var g: Float32 = 0
            var b: Float32 = 0
            for k in range(-2, 3):
                var sx = max(0, min(width - 1, x + k)) if horizontal else x
                var sy = y if horizontal else max(0, min(height - 1, y + k))
                var o = (sy * width + sx) * 3
                r += source[o]
                g += source[o + 1]
                b += source[o + 2]
            var t = (y * width + x) * 3
            target[t] = r * 0.2
            target[t + 1] = g * 0.2
            target[t + 2] = b * 0.2


def text_em(size: Int) -> Float32:
    """Em size on the UI canvas for a `draw_text` size."""
    return Float32(max(size, 12))
