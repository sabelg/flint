"""High-level game-agnostic 3D PBR scene and volumetric VFX renderer.

Wraps `SoftwareFramebuffer`, `LightingEnvironment`, and `FxTextures` so any 3D
application can render glTF 2.0 `NativeModel` hierarchies, GGX PBR materials,
analytic ray-traced sphere shadows, dynamic point/event lights, volumetric
ribbons/beams/billboards, and SDF HUDs on Vulkan, AMDGPU, or CPU fallback.
"""
from std.collections import Dict
from std.math import sqrt, sin, cos, floor
from std.memory import bitcast
from flint.core.math import Vec3
from flint.core.transform import Mat4
from flint.assets.model import NativeMesh, NativeTexture, NativeModel
from flint.render.lighting import LightingProfile, LightingEnvironment, LocalLight
from flint.render.fx_atlas import (
    FxTextures,
    TEX_VIGNETTE,
    TEX_DROPLET,
    TEX_RIBBON,
    TEX_FLAME,
    TEX_SHOCK_RING,
    TEX_ANAMORPHIC,
    TEX_CUMULUS,
    TEX_FIREBALL,
    TEX_VORONOI_BEAM,
    TEX_HEX_SHIELD,
    TEX_VAPOR_CONE,
    TEX_LIGHTNING,
)
from flint.render.gpu_layout import (
    CONSTANT_CLUSTER_LIGHTS,
    CONSTANT_KEY,
    CONSTANT_WORLD,
    CONSTANT_ENV_SKY,
    CONSTANT_ENV_GROUND,
    CONSTANT_ENV_HORIZON,
    CONSTANT_LIGHTS,
    CONSTANT_SPHERES,
    CONSTANT_CSM,
    CONSTANT_V7_POST,
    CSM_RES,
    EFFECT_CHROMATIC_REFRACT,
    EFFECT_CSM,
    EFFECT_DYN_REFLECT,
    EFFECT_DYRL_ANIME,
    EFFECT_FROXEL_VOL,
    EFFECT_HIZ_CULL,
    EFFECT_PHYSICAL_SKY,
    EFFECT_RADIANCE_GI,
    EFFECT_SDF_SHADOWS,
    EFFECT_SSGI_BENT,
    EFFECT_VOL_CLOUDS,
    EFFECT_WEATHER_PRECIP,
    GPU_CULL_IN_STRIDE_WORDS,
    GPU_CULL_MAX_INSTANCES,
    GPU_PARTICLE_MAX,
    GPU_PARTICLE_STRIDE_WORDS,
    EFFECT_VOLUMETRIC,
    GPU_SKIN_MAX_JOINTS,
    MAX_CLUSTER_LIGHTS,
    MAX_IRRADIANCE_PROBES,
    MAX_RADIANCE_CASCADES,
    MAX_SDF_OCCLUDERS,
    MAX_SHADOW_SPHERES,
    MAX_SURFACE_DECALS,
)
from flint.render.gpu_raster import (
    BACKEND_CPU,
    BACKEND_AMDGPU,
    BACKEND_VULKAN,
    BACKEND_AUTOMATIC,
    FRAME_SETS,
    requested_backend,
    backend_name,
)
from flint.render.rasterizer import (
    SoftwareFramebuffer,
    IrradianceProbe,
    RadianceCascadeNode,
    SdfOccluder,
    SurfaceDecal,
    RenderVertex,
    CameraVertex,
    ScreenVertex,
    dot,
    cross,
)


comptime FX_SCORCH: Int = 1
comptime FX_BOLT: Int = 5
comptime FX_STRIPES: Int = 6
comptime FX_SOFT: Int = 7
comptime FX_VIGNETTE: Int = TEX_VIGNETTE
comptime FX_RIBBON: Int = TEX_RIBBON
comptime FX_FLAME: Int = TEX_FLAME
comptime FX_SHOCK_RING: Int = TEX_SHOCK_RING
comptime FX_ANAMORPHIC: Int = TEX_ANAMORPHIC
comptime FX_CUMULUS: Int = TEX_CUMULUS
comptime FX_FIREBALL: Int = TEX_FIREBALL
comptime FX_VORONOI_BEAM: Int = TEX_VORONOI_BEAM
comptime FX_HEX_SHIELD: Int = TEX_HEX_SHIELD
comptime FX_VAPOR_CONE: Int = TEX_VAPOR_CONE
comptime FX_LIGHTNING: Int = TEX_LIGHTNING


def mul_argb(a: UInt32, b: UInt32) -> UInt32:
    var aa = ((a >> 24) & 255) * ((b >> 24) & 255) // 255
    var ar = ((a >> 16) & 255) * ((b >> 16) & 255) // 255
    var ag = ((a >> 8) & 255) * ((b >> 8) & 255) // 255
    var ab = (a & 255) * (b & 255) // 255
    return (aa << 24) | (ar << 16) | (ag << 8) | ab


@always_inline
def _mat4_max_scale(m: Mat4) -> Float32:
    var s0 = m.at(0, 0) * m.at(0, 0) + m.at(1, 0) * m.at(1, 0) + m.at(2, 0) * m.at(2, 0)
    var s1 = m.at(0, 1) * m.at(0, 1) + m.at(1, 1) * m.at(1, 1) + m.at(2, 1) * m.at(2, 1)
    var s2 = m.at(0, 2) * m.at(0, 2) + m.at(1, 2) * m.at(1, 2) + m.at(2, 2) * m.at(2, 2)
    return max(Float32(0.001), sqrt(max(s0, max(s1, s2))))


@always_inline
def _mat4_differs(a: Mat4, b: Mat4) -> Bool:
    for i in range(16):
        if abs(a.values[i] - b.values[i]) > Float32(0.0001):
            return True
    return False


@always_inline
def _clip_near_tri(
    c0: CameraVertex,
    c1: CameraVertex,
    c2: CameraVertex,
    mut out_poly: Array[CameraVertex, 4],
) -> Int:
    var near = Float32(0.06)
    var poly = Array[CameraVertex, 3](fill=CameraVertex())
    poly[0] = c0
    poly[1] = c1
    poly[2] = c2
    var count = 0
    for idx in range(3):
        var cur = poly[idx]
        var prev = poly[(idx + 2) % 3]
        var cur_in = cur.position.z >= near
        var prev_in = prev.position.z >= near
        if cur_in != prev_in and count < 4:
            var t = (near - prev.position.z) / (cur.position.z - prev.position.z)
            out_poly[count] = prev.interpolate(cur, t)
            count += 1
        if cur_in and count < 4:
            out_poly[count] = cur
            count += 1
    return count


struct GpuSceneRenderer(Movable):
    var fb: SoftwareFramebuffer
    var lighting: LightingEnvironment
    var fx: FxTextures
    var empty_tex: List[UInt32]
    var sphere_centers: List[Vec3]
    var sphere_radii: List[Float32]
    var active_profile: LightingProfile
    var active_quality: Int
    var active_reduced: Bool
    var frame_index: Int
    var scene_ended: Bool
    var scratch_cam: List[CameraVertex]
    var scratch_screen: List[ScreenVertex]
    var scratch_in_front: List[UInt8]
    var scratch_csm_proj: List[Vec3]
    var scratch_world_pts: List[Vec3]
    var scratch_dists: List[Float32]
    var prev_mesh_mats: Dict[UInt64, Mat4]
    var cur_mesh_mats: Dict[UInt64, Mat4]
    var mesh_ordinals: Dict[UInt64, Int]
    var hiz_culled_meshes: Int

    def __init__(out self, width: Int = 1280, height: Int = 720, prefer_gpu: Bool = True):
        self.fb = SoftwareFramebuffer(width, height)
        self.lighting = LightingEnvironment()
        self.fx = FxTextures()
        self.empty_tex = List[UInt32]()
        self.sphere_centers = List[Vec3]()
        self.sphere_radii = List[Float32]()
        self.active_profile = LightingProfile()
        self.active_quality = 2
        self.active_reduced = False
        self.frame_index = 0
        self.scene_ended = False
        self.scratch_cam = List[CameraVertex]()
        self.scratch_screen = List[ScreenVertex]()
        self.scratch_in_front = List[UInt8]()
        self.scratch_csm_proj = List[Vec3]()
        self.scratch_world_pts = List[Vec3]()
        self.scratch_dists = List[Float32]()
        self.prev_mesh_mats = Dict[UInt64, Mat4]()
        self.cur_mesh_mats = Dict[UInt64, Mat4]()
        self.mesh_ordinals = Dict[UInt64, Int]()
        self.hiz_culled_meshes = 0
        self.lighting.per_pixel = True

        if prefer_gpu:
            var wanted = requested_backend()
            if wanted == BACKEND_VULKAN:
                _ = self.fb.enable_gpu(BACKEND_VULKAN)
            elif wanted == BACKEND_AMDGPU:
                _ = self.fb.enable_gpu(BACKEND_AMDGPU)
            elif wanted == BACKEND_AUTOMATIC:
                # Try Vulkan first (works across NVIDIA, AMD, Intel, Mesa), then direct AMDGPU.
                if not self.fb.enable_gpu(BACKEND_VULKAN):
                    _ = self.fb.enable_gpu(BACKEND_AMDGPU)


    def shutdown(mut self):
        """Release the GPU device now (safe to call more than once; the CPU path keeps working)."""
        self.fb.gpu.shutdown()

    def __deinit__(deinit self):
        # Close the Vulkan device with its owner, so no VkDevice outlives the renderer
        # and the driver's own exit handlers never see a half-torn-down device.
        self.fb.gpu.shutdown()

    def active_backend_name(self) -> String:
        if self.fb.gpu.active:
            return backend_name(self.fb.gpu.backend)
        return "cpu"

    def using_gpu(self) -> Bool:
        return self.fb.gpu.active

    def begin_frame(
        mut self,
        cam_pos: Vec3,
        cam_target: Vec3,
        fov_y: Float32,
        profile: LightingProfile,
        dt: Float32 = 1.0 / 60.0,
        quality: Int = 2,
        reduced_effects: Bool = False,
        cam_roll: Float32 = 0.0,
    ):
        self.active_profile = profile
        self.active_quality = quality
        self.active_reduced = reduced_effects
        self.scene_ended = False
        self.sphere_centers.clear()
        self.sphere_radii.clear()
        self.fb.shadow_sphere_count = 0
        self.fb.no_depth_write = False
        self.fb.additive_blend = False
        self.fb.cpu_sphere_centers.clear()
        self.fb.cpu_sphere_radii_sq.clear()
        self.fb.cpu_lights.clear()
        self.fb.cpu_probes.clear()
        self.fb.cpu_decals.clear()
        self.fb.cpu_radiance_nodes.clear()
        self.fb.cpu_sdf_occluders.clear()
        self.hiz_culled_meshes = 0

        if self.fb.gpu.active:
            var slot = self.frame_index % FRAME_SETS
            if not self.fb.gpu.begin(slot):
                self.fb.stop_recording_if_gpu_lost()
            self.frame_index += 1

        self.fb.set_camera(cam_pos, cam_target, fov_y)
        self.fb.set_camera_roll(cam_roll)
        var cam_cut = (
            self.fb.temporal_frames == 0
            or (self.fb.camera - self.fb.prev_camera).length_squared() > Float32(324.0)
            or dot(self.fb.forward, self.fb.prev_forward) < Float32(0.70)
            or abs(self.fb.focal - self.fb.prev_focal) > self.fb.focal * Float32(0.15)
        )
        self.prev_mesh_mats = self.cur_mesh_mats^
        self.cur_mesh_mats = Dict[UInt64, Mat4]()
        self.mesh_ordinals.clear()
        if cam_cut:
            self.prev_mesh_mats.clear()
            self.fb.cpu_hiz_valid = False
            self.fb.gpu.hiz_synced_slot = -1
            self.fb.gpu.hiz_valid_mask = 0

        self.fb.configure_post(profile, quality, reduced_effects)
        var fog_rgb = profile.fog_rgb()
        self.fb.clear(fog_rgb[0], fog_rgb[1], fog_rgb[2])

        self.lighting.time += max(Float32(0), dt)
        self.fb.water_time = self.lighting.time
        self.lighting.update_events(dt)
        self.lighting.begin_frame_with(profile, cam_pos, quality, reduced_effects)

    def add_point_light(
        mut self,
        position: Vec3,
        color: Vec3,
        radius: Float32,
        intensity: Float32,
        priority: Int = 50,
        stable_id: Int = 0,
    ):
        self.lighting.register(position, color, radius, intensity, priority, stable_id)

    def add_tube_light(
        mut self,
        start_pos: Vec3,
        end_pos: Vec3,
        color: Vec3,
        radius: Float32,
        intensity: Float32,
        priority: Int = 60,
        stable_id: Int = 0,
    ):
        """Register a Brian Karis representative-point capsule / tube area light."""
        self.lighting.register_tube(start_pos, end_pos, color, radius, intensity, priority, stable_id)

    def emit_event_light(
        mut self,
        position: Vec3,
        color: Vec3,
        radius: Float32,
        intensity: Float32,
        duration: Float32,
        stable_id: Int = -1,
    ):
        self.lighting.emit_event(position, color, radius, intensity, duration, stable_id)

    def emit_tube_event_light(
        mut self,
        start_pos: Vec3,
        end_pos: Vec3,
        color: Vec3,
        radius: Float32,
        intensity: Float32,
        duration: Float32,
        stable_id: Int = -1,
    ):
        """Emit a short-lived capsule / tube area light (e.g. mega-particle beam or laser bolt)."""
        self.lighting.emit_tube_event(start_pos, end_pos, color, radius, intensity, duration, stable_id)

    def renew_event_light(
        mut self,
        position: Vec3,
        color: Vec3,
        radius: Float32,
        intensity: Float32,
        duration: Float32,
        stable_id: Int,
        priority: Int,
    ):
        """Update or hold a persistent dynamic event light in-place by stable_id."""
        self.lighting.renew_event(
            position, color, radius, intensity, duration, stable_id, priority
        )

    def add_shadow_sphere(mut self, center: Vec3, radius: Float32):
        if radius <= 0.01 or len(self.sphere_centers) >= MAX_SHADOW_SPHERES:
            return
        self.sphere_centers.append(center)
        self.sphere_radii.append(radius)

    def set_physical_atmosphere(
        mut self,
        enabled: Bool,
        altitude_km: Float32 = 0.05,
        rayleigh: Float32 = 1.0,
        mie: Float32 = 1.0,
        ozone: Float32 = 1.0,
        aerial_strength: Float32 = 0.32,
    ):
        """Enable or disable the Hillaire physical Sky-View LUT + aerial perspective (0 km..400 km)."""
        self.fb.physical_sky_enabled = enabled
        self.fb.cam_altitude_km = max(Float32(0.02), altitude_km)
        self.fb.rayleigh_density = max(Float32(0.0), rayleigh)
        self.fb.mie_density = max(Float32(0.0), mie)
        self.fb.ozone_strength = max(Float32(0.0), ozone)
        self.fb.aerial_strength = max(Float32(0.0), min(Float32(1.0), aerial_strength))
        var extras = self.fb.allow_gpu_effects and self.active_quality >= 1 and not self.active_reduced
        if enabled and extras:
            self.fb.gpu_effects |= EFFECT_PHYSICAL_SKY
        else:
            self.fb.gpu_effects &= ~EFFECT_PHYSICAL_SKY

    def set_froxel_volumetrics(mut self, enabled: Bool, max_dist: Float32 = 175.0):
        """Enable or disable the 64x36x16 temporally reprojected 3D froxel volumetric scattering grid."""
        self.fb.froxel_vol_enabled = enabled
        self.fb.froxel_max_dist = max(Float32(25.0), max_dist)
        var extras = self.fb.allow_gpu_effects and self.active_quality >= 1 and not self.active_reduced
        if enabled and extras and (self.fb.gpu_effects & EFFECT_VOLUMETRIC) != 0:
            self.fb.gpu_effects |= EFFECT_FROXEL_VOL
        else:
            self.fb.gpu_effects &= ~EFFECT_FROXEL_VOL

    def set_hiz_culling(mut self, enabled: Bool):
        """Enable or disable 16x16 coarse Hierarchical-Z depth pyramid occlusion culling."""
        self.fb.hiz_cull_enabled = enabled
        var extras = self.fb.allow_gpu_effects and self.active_quality >= 1 and not self.active_reduced
        if enabled and extras:
            self.fb.gpu_effects |= EFFECT_HIZ_CULL
        else:
            self.fb.gpu_effects &= ~EFFECT_HIZ_CULL

    def set_volumetric_clouds(
        mut self,
        enabled: Bool,
        base_y: Float32 = 420.0,
        top_y: Float32 = 980.0,
        coverage: Float32 = 0.52,
        density: Float32 = 0.026,
        shadow_strength: Float32 = 0.42,
        wind_x: Float32 = 12.0,
        wind_z: Float32 = -6.0,
        scale: Float32 = 0.0022,
    ):
        """Enable or disable ray-marched 3D Worley-Perlin cumulus cloudscapes + projected world cloud shadows."""
        self.fb.vol_clouds_enabled = enabled
        self.fb.cloud_base_y = base_y
        self.fb.cloud_top_y = max(base_y + Float32(40.0), top_y)
        self.fb.cloud_coverage = max(Float32(0.0), min(Float32(1.0), coverage))
        self.fb.cloud_density = max(Float32(0.0), density)
        self.fb.cloud_shadow_strength = max(Float32(0.0), min(Float32(0.85), shadow_strength))
        self.fb.cloud_wind_x = wind_x
        self.fb.cloud_wind_z = wind_z
        self.fb.cloud_scale = max(Float32(0.0002), scale)
        var extras = self.fb.allow_gpu_effects and self.active_quality >= 1 and not self.active_reduced
        if enabled and extras:
            self.fb.gpu_effects |= EFFECT_VOL_CLOUDS
        else:
            self.fb.gpu_effects &= ~EFFECT_VOL_CLOUDS

    def set_ssgi_bent_normals(mut self, enabled: Bool, intensity: Float32 = 0.34):
        """Enable or disable 1-bounce Screen-Space Diffuse GI (SSGI) color bleeding + GTAO bent normals."""
        self.fb.ssgi_enabled = enabled
        self.fb.ssgi_intensity = max(Float32(0.0), min(Float32(1.5), intensity))
        var extras = self.fb.allow_gpu_effects and self.active_quality >= 2 and not self.active_reduced
        if enabled and extras:
            self.fb.gpu_effects |= EFFECT_SSGI_BENT
        else:
            self.fb.gpu_effects &= ~EFFECT_SSGI_BENT

    def set_chromatic_refraction(
        mut self,
        enabled: Bool,
        dispersion: Float32 = 0.22,
        heat_haze_amp: Float32 = 1.35,
    ):
        """Enable or disable Cauchy RGB canopy dispersion & animated turbine/beam heat-haze distortion."""
        self.fb.chromatic_refract_enabled = enabled
        self.fb.dispersion_strength = max(Float32(0.0), min(Float32(0.65), dispersion))
        self.fb.heat_haze_amp = max(Float32(0.0), min(Float32(6.0), heat_haze_amp))
        var extras = self.fb.allow_gpu_effects and self.active_quality >= 1 and not self.active_reduced
        if enabled and extras:
            self.fb.gpu_effects |= EFFECT_CHROMATIC_REFRACT
        else:
            self.fb.gpu_effects &= ~EFFECT_CHROMATIC_REFRACT

    def set_radiance_gi(mut self, enabled: Bool, intensity: Float32 = 0.28):
        """Enable or disable V10 World-Space Radiance Cascades multi-bounce GI (`EFFECT_RADIANCE_GI`)."""
        self.fb.radiance_gi_enabled = enabled
        self.fb.radiance_gi_intensity = max(Float32(0.0), min(Float32(1.5), intensity))
        var extras = self.fb.allow_gpu_effects and self.active_quality >= 2 and not self.active_reduced
        if enabled and extras:
            self.fb.gpu_effects |= EFFECT_RADIANCE_GI
        else:
            self.fb.gpu_effects &= ~EFFECT_RADIANCE_GI

    def add_radiance_emitter(
        mut self,
        position: Vec3,
        radius: Float32,
        radiance_rgb: Vec3,
        dir_bias_y: Float32 = 0.15,
    ):
        """Register an explicit world-space radiance cascade emitter (e.g. off-screen beam, bridge glow, or lava/fire)."""
        if radius <= Float32(0.1) or len(self.fb.cpu_radiance_nodes) >= MAX_RADIANCE_CASCADES:
            return
        var inv_r2 = Float32(1.0) / max(Float32(0.25), radius * radius)
        self.fb.cpu_radiance_nodes.append(
            RadianceCascadeNode(
                position,
                inv_r2,
                radiance_rgb,
                max(Float32(-1.0), min(Float32(1.0), dir_bias_y)),
            )
        )

    def set_dyrl_anime_mode(
        mut self,
        enabled: Bool,
        ink_strength: Float32 = 0.85,
        cel_bands: Float32 = 3.0,
        glint_strength: Float32 = 0.65,
        halation: Float32 = 0.28,
    ):
        """Enable or disable V10 1984 *DYRL* 35mm Cel-Film / PBR Hybrid mode (`EFFECT_DYRL_ANIME`)."""
        self.fb.dyrl_anime_enabled = enabled
        self.fb.dyrl_ink_strength = max(Float32(0.0), min(Float32(1.5), ink_strength))
        self.fb.dyrl_cel_bands = max(Float32(2.0), min(Float32(6.0), cel_bands))
        self.fb.dyrl_glint_strength = max(Float32(0.0), min(Float32(2.0), glint_strength))
        self.fb.dyrl_halation = max(Float32(0.0), min(Float32(1.0), halation))
        var extras = self.fb.allow_gpu_effects and self.active_quality >= 1 and not self.active_reduced
        if enabled and extras:
            self.fb.gpu_effects |= EFFECT_DYRL_ANIME
        else:
            self.fb.gpu_effects &= ~EFFECT_DYRL_ANIME

    def set_tsr_upscale(mut self, enabled: Bool, scale: Float32 = 0.75, sharpness: Float32 = 0.42):
        """Configure V10/V11 Zero-Copy Sub-Native Temporal Super-Resolution (`SHADER_TSR_UPSCALE`, internal scale 50%..100%)."""
        self.fb.tsr_enabled = enabled
        self.fb.tsr_scale = max(Float32(0.50), min(Float32(1.0), scale))
        self.fb.tsr_sharpness = max(Float32(0.0), min(Float32(1.5), sharpness))
        if enabled and self.fb.tsr_scale < Float32(0.97) and self.fb.width >= 160 and self.fb.height >= 90:
            self.fb.render_w = max(64, min(self.fb.width, (Int(Float32(self.fb.width) * self.fb.tsr_scale + Float32(0.5)) // 8) * 8))
            self.fb.render_h = max(64, min(self.fb.height, (Int(Float32(self.fb.height) * self.fb.tsr_scale + Float32(0.5)) // 4) * 4))
            self.fb.render_focal = self.fb.focal * (Float32(self.fb.render_h) / Float32(max(1, self.fb.height)))
            self.fb.tsr_subnative_active = self.fb.render_w < self.fb.width or self.fb.render_h < self.fb.height
        else:
            self.fb.render_w = self.fb.width
            self.fb.render_h = self.fb.height
            self.fb.render_focal = self.fb.focal
            self.fb.tsr_subnative_active = False
        if self.fb.gpu.active:
            self.fb.gpu.render_width = self.fb.render_w
            self.fb.gpu.render_height = self.fb.render_h

    def set_sdf_shadows(mut self, enabled: Bool, softness_k: Float32 = 8.0):
        """Enable or disable V11 World-Space 3D SDF Soft-Penumbra Capsule/Sphere Shadows (`EFFECT_SDF_SHADOWS`)."""
        self.fb.sdf_shadows_enabled = enabled
        self.fb.sdf_softness_k = max(Float32(1.5), min(Float32(24.0), softness_k))
        var extras = self.fb.allow_gpu_effects and self.active_quality >= 1 and not self.active_reduced
        if enabled and extras and len(self.fb.cpu_sdf_occluders) > 0:
            self.fb.gpu_effects |= EFFECT_SDF_SHADOWS
        else:
            self.fb.gpu_effects &= ~EFFECT_SDF_SHADOWS

    def add_sdf_occluder(
        mut self,
        start_pos: Vec3,
        end_pos: Vec3,
        radius: Float32,
        opacity: Float32 = 0.88,
    ):
        """Register an analytical 3D SDF capsule or sphere (`start_pos == end_pos`) shadow caster (`MAX_SDF_OCCLUDERS = 16`)."""
        if radius <= Float32(0.05) or len(self.fb.cpu_sdf_occluders) >= MAX_SDF_OCCLUDERS:
            return
        self.fb.cpu_sdf_occluders.append(
            SdfOccluder(
                start_pos,
                radius,
                end_pos - start_pos,
                max(Float32(0.05), min(Float32(0.95), opacity)),
            )
        )
        var extras = self.fb.allow_gpu_effects and self.active_quality >= 1 and not self.active_reduced
        if self.fb.sdf_shadows_enabled and extras:
            self.fb.gpu_effects |= EFFECT_SDF_SHADOWS

    def set_weather(
        mut self,
        enabled: Bool,
        kind: Int = 1,
        intensity: Float32 = 0.55,
        wetness: Float32 = 0.65,
        wind_x: Float32 = 8.0,
        wind_z: Float32 = -4.0,
    ):
        """Configure V11 Dynamic Weather (`kind`: 1 = Monsoon Rain, 2 = Polar Blizzard, 3 = Ice Crystals) & Wet-PBR."""
        self.fb.weather_enabled = enabled
        self.fb.weather_kind = max(0, min(3, kind))
        self.fb.weather_intensity = max(Float32(0.0), min(Float32(1.0), intensity)) if enabled else Float32(0.0)
        self.fb.wetness = max(Float32(0.0), min(Float32(1.0), wetness)) if enabled else Float32(0.0)
        self.fb.weather_wind_x = wind_x
        self.fb.weather_wind_z = wind_z
        var extras = self.fb.allow_gpu_effects and self.active_quality >= 1 and not self.active_reduced
        if enabled and extras and (self.fb.weather_intensity > Float32(0.01) or self.fb.wetness > Float32(0.01)):
            self.fb.gpu_effects |= EFFECT_WEATHER_PRECIP
        else:
            self.fb.gpu_effects &= ~EFFECT_WEATHER_PRECIP

    def set_dynamic_reflections(mut self, enabled: Bool, intensity: Float32 = 0.44):
        """Enable or disable V11 Real-Time Dynamic Environment Reflection Lobe (`EFFECT_DYN_REFLECT`)."""
        self.fb.dyn_reflect_enabled = enabled
        self.fb.dyn_reflect_intensity = max(Float32(0.0), min(Float32(1.5), intensity))
        var extras = self.fb.allow_gpu_effects and self.active_quality >= 1 and not self.active_reduced
        if enabled and extras and self.fb.dyn_reflect_intensity > Float32(0.005):
            self.fb.gpu_effects |= EFFECT_DYN_REFLECT
        else:
            self.fb.gpu_effects &= ~EFFECT_DYN_REFLECT

    def upscale_tsr(
        mut self,
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
        """Reconstruct `dst_w x dst_h` ARGB image from `src_w x src_h` via Vulkan `SHADER_TSR_UPSCALE` or CPU parity."""
        var used_gpu = False
        if self.fb.gpu.active and self.fb.gpu.backend == BACKEND_VULKAN:
            used_gpu = self.fb.gpu.run_tsr_upscale(
                src_pixels,
                src_w,
                src_h,
                dst_w,
                dst_h,
                out_pixels,
                sharpness=sharpness,
                history_blend=history_blend,
                jitter_x=jitter_x,
                jitter_y=jitter_y,
                history_parity=self.fb.history_parity,
            )
        if not used_gpu:
            self.fb.tsr_upscale_cpu(
                src_pixels,
                src_w,
                src_h,
                dst_w,
                dst_h,
                out_pixels,
                sharpness=sharpness,
                history_blend=history_blend,
                jitter_x=jitter_x,
                jitter_y=jitter_y,
            )

    def cull_instances_indirect(
        mut self,
        packed_instances: List[UInt32],
        instance_count: Int,
        mut out_visibility: List[UInt32],
        mut out_indirect_cmds: List[UInt32],
        use_hiz: Bool = True,
    ) -> Int:
        """Cull up to 8,192 bounding-sphere instances against frustum + 16x16 Hi-Z into compact indirect draw commands."""
        var n = min(instance_count, min(GPU_CULL_MAX_INSTANCES, len(packed_instances) // GPU_CULL_IN_STRIDE_WORDS))
        if n <= 0:
            out_visibility.clear()
            out_indirect_cmds.clear()
            return 0
        if self.fb.gpu.active and self.fb.gpu.backend == BACKEND_VULKAN:
            if self.fb.gpu.constants.cpu_address != 0:
                self.fb.upload_view()
            var gpu_vis = self.fb.gpu.run_gpu_cull(
                packed_instances,
                n,
                out_visibility,
                out_indirect_cmds,
                use_hiz=use_hiz and self.fb.hiz_cull_enabled,
            )
            if gpu_vis >= 0:
                return gpu_vis
        return self.fb.cull_instances_cpu(
            packed_instances,
            n,
            out_visibility,
            out_indirect_cmds,
            use_hiz=use_hiz and self.fb.hiz_cull_enabled,
        )

    def step_gpu_particles(
        mut self,
        mut packed_particles: List[UInt32],
        count: Int,
        dt: Float32,
        gravity: Float32 = -9.8,
        drag: Float32 = 0.85,
        depth_bounce: Bool = True,
        restitution: Float32 = 0.45,
    ):
        """Step up to 65,536 ballistic + depth-bouncing particles on Vulkan compute or 1:1 CPU fallback."""
        var n = min(count, min(GPU_PARTICLE_MAX, len(packed_particles) // GPU_PARTICLE_STRIDE_WORDS))
        if n <= 0:
            return
        var used_gpu = False
        if self.fb.gpu.active and self.fb.gpu.backend == BACKEND_VULKAN:
            used_gpu = self.fb.gpu.run_gpu_particles(
                packed_particles,
                n,
                dt,
                gravity=gravity,
                drag=drag,
                depth_bounce=depth_bounce,
                restitution=restitution,
            )
        if not used_gpu:
            self.fb.step_particles_cpu(
                packed_particles,
                n,
                dt,
                gravity=gravity,
                drag=drag,
                depth_bounce=depth_bounce,
                restitution=restitution,
            )

    def draw_gpu_particles(
        mut self,
        packed_particles: List[UInt32],
        count: Int,
        max_draw: Int = 1536,
        emissive: Float32 = 2.1,
        additive: Bool = True,
    ) -> Int:
        """Render live frustum-visible particles from a `SHADER_GPU_PARTICLES` buffer as soft HDR sparks/debris."""
        var n = min(count, min(GPU_PARTICLE_MAX, len(packed_particles) // GPU_PARTICLE_STRIDE_WORDS))
        if n <= 0 or max_draw <= 0:
            return 0
        var stride = max(1, (n + max_draw - 1) // max_draw)
        var drawn = 0
        var idx = 0
        while idx < n and drawn < max_draw:
            var b = idx * GPU_PARTICLE_STRIDE_WORDS
            var life = bitcast[DType.float32](packed_particles[b + 3])
            if life > Float32(0.01):
                var px = bitcast[DType.float32](packed_particles[b])
                var py = bitcast[DType.float32](packed_particles[b + 1])
                var pz = bitcast[DType.float32](packed_particles[b + 2])
                var p = Vec3(px, py, pz)
                var meta = packed_particles[b + 7]
                var size_f = max(Float32(0.12), Float32((meta >> UInt32(24)) & UInt32(255)) * Float32(0.04))
                if self.fb.sphere_visible(p, size_f * Float32(1.5)):
                    var col = UInt32(0xFF000000) | (meta & UInt32(0x00FFFFFF))
                    var alpha_u = UInt8(max(24, min(240, Int(min(Float32(1.0), life * Float32(1.6)) * Float32(225.0)))))
                    self.draw_billboard(
                        p,
                        size_f,
                        FX_SOFT,
                        col,
                        alpha_u,
                        emissive=emissive,
                        additive=additive,
                        soft_depth=max(Float32(0.35), size_f * Float32(1.4)),
                    )
                    drawn += 1
            idx += stride
        return drawn

    def is_sphere_occluded_hiz(self, center: Vec3, radius: Float32) -> Bool:
        """Project bounding sphere to conservative screen rectangle and query the 16x16 Hi-Z pyramid."""
        if not self.fb.hiz_cull_enabled or radius <= Float32(0.01):
            return False
        var rel = center - self.fb.camera
        var vz = dot(rel, self.fb.forward)
        var nearest_z = vz - radius
        if nearest_z <= Float32(0.35):
            return False
        var scale = self.fb.focal / nearest_z
        var half_w = Float32(self.fb.width) * Float32(0.5)
        var half_h = Float32(self.fb.height) * Float32(0.5)
        var cx = half_w + dot(rel, self.fb.right) * (self.fb.focal / max(Float32(0.35), vz))
        var cy = half_h - dot(rel, self.fb.up) * (self.fb.focal / max(Float32(0.35), vz))
        var reach = radius * scale * Float32(1.12)
        var min_x = Int(floor(cx - reach))
        var max_x = Int(floor(cx + reach + Float32(1.0)))
        var min_y = Int(floor(cy - reach))
        var max_y = Int(floor(cy + reach + Float32(1.0)))
        if max_x < 0 or min_x >= self.fb.width or max_y < 0 or min_y >= self.fb.height:
            return False
        return self.fb.is_rect_occluded_hiz(min_x, min_y, max_x, max_y, Float32(1.0) / nearest_z)

    def add_irradiance_probe(
        mut self,
        position: Vec3,
        radius: Float32,
        color: Vec3,
        sky_occlusion: Float32 = 0.80,
    ):
        """Register an enclosed-interior SH/ambient irradiance probe with exterior sky occlusion."""
        if radius <= Float32(0.05) or len(self.fb.cpu_probes) >= MAX_IRRADIANCE_PROBES:
            return
        var inv_r2 = Float32(1.0) / max(Float32(0.01), radius * radius)
        self.fb.cpu_probes.append(IrradianceProbe(position, inv_r2, color, sky_occlusion))

    def add_surface_decal(
        mut self,
        center: Vec3,
        axis_right: Vec3,
        axis_forward: Vec3,
        half_extents: Vec3,
        color_argb: UInt32,
        opacity: Float32 = 0.85,
        scorch_glow: Bool = False,
    ):
        """Register an oriented-box surface decal (paint marking, blast crater, or glowing plasma scorch)."""
        if len(self.fb.cpu_decals) >= MAX_SURFACE_DECALS:
            return
        var col_rgb = Vec3(
            Float32((color_argb >> 16) & 255),
            Float32((color_argb >> 8) & 255),
            Float32(color_argb & 255),
        )
        var op = max(Float32(0.01), min(Float32(1.5), opacity))
        var signed_op = -op if scorch_glow else op
        var inv_hx = Float32(1.0) / max(Float32(0.02), half_extents.x)
        var inv_hy = Float32(1.0) / max(Float32(0.02), half_extents.y)
        var inv_hz = Float32(1.0) / max(Float32(0.02), half_extents.z)
        self.fb.cpu_decals.append(
            SurfaceDecal(
                center,
                inv_hx,
                axis_right.normalized(),
                inv_hy,
                axis_forward.normalized(),
                inv_hz,
                col_rgb,
                signed_op,
            )
        )

    def skin_mesh_vertices(
        mut self,
        bind_mesh: NativeMesh,
        joints_4x_packed: List[UInt32],
        weights_4x: List[Float32],
        joint_matrices_3x4: List[Float32],
        mut out_mesh: NativeMesh,
    ):
        """Deform `bind_mesh` into `out_mesh` using 4-bone Linear-Blend Skinning on GPU compute or CPU fallback."""
        var vcount = bind_mesh.vertex_count()
        if vcount <= 0 or len(joint_matrices_3x4) < 12:
            return
        if len(out_mesh.positions) != vcount * 3:
            out_mesh = bind_mesh.copy()
        if len(out_mesh.normals) < vcount * 3:
            out_mesh.normals.resize(vcount * 3, 0.0)
        var out_pn = List[Float32](length=vcount * 6, fill=0.0)
        var used_gpu = False
        if self.fb.gpu.active and self.fb.gpu.backend == BACKEND_VULKAN:
            var has_n = len(bind_mesh.normals) >= vcount * 3
            var packed = List[UInt32](length=vcount * 12, fill=0)
            for v in range(vcount):
                var wb = v * 12
                packed[wb + 0] = bitcast[DType.uint32](bind_mesh.positions[v * 3])
                packed[wb + 1] = bitcast[DType.uint32](bind_mesh.positions[v * 3 + 1])
                packed[wb + 2] = bitcast[DType.uint32](bind_mesh.positions[v * 3 + 2])
                packed[wb + 3] = bitcast[DType.uint32](bind_mesh.normals[v * 3] if has_n else Float32(0.0))
                packed[wb + 4] = bitcast[DType.uint32](bind_mesh.normals[v * 3 + 1] if has_n else Float32(1.0))
                packed[wb + 5] = bitcast[DType.uint32](bind_mesh.normals[v * 3 + 2] if has_n else Float32(0.0))
                packed[wb + 6] = joints_4x_packed[v] if v < len(joints_4x_packed) else UInt32(0)
                for inf in range(4):
                    var wf = weights_4x[v * 4 + inf] if (v * 4 + inf) < len(weights_4x) else (Float32(1.0) if inf == 0 else Float32(0.0))
                    packed[wb + 7 + inf] = bitcast[DType.uint32](wf)
                packed[wb + 11] = 0
            used_gpu = self.fb.gpu.run_gpu_skin(joint_matrices_3x4, packed, vcount, out_pn)
        if not used_gpu:
            SoftwareFramebuffer.skin_vertices_cpu(
                joint_matrices_3x4,
                bind_mesh.positions,
                bind_mesh.normals,
                joints_4x_packed,
                weights_4x,
                vcount,
                out_pn,
            )
        var min_p = Vec3(1e9, 1e9, 1e9)
        var max_p = Vec3(-1e9, -1e9, -1e9)
        for v in range(vcount):
            var px = out_pn[v * 6]
            var py = out_pn[v * 6 + 1]
            var pz = out_pn[v * 6 + 2]
            out_mesh.positions[v * 3] = px
            out_mesh.positions[v * 3 + 1] = py
            out_mesh.positions[v * 3 + 2] = pz
            out_mesh.normals[v * 3] = out_pn[v * 6 + 3]
            out_mesh.normals[v * 3 + 1] = out_pn[v * 6 + 4]
            out_mesh.normals[v * 3 + 2] = out_pn[v * 6 + 5]
            if px < min_p.x:
                min_p.x = px
            if py < min_p.y:
                min_p.y = py
            if pz < min_p.z:
                min_p.z = pz
            if px > max_p.x:
                max_p.x = px
            if py > max_p.y:
                max_p.y = py
            if pz > max_p.z:
                max_p.z = pz
        out_mesh.bounds_center = (min_p + max_p) * Float32(0.5)
        out_mesh.bounds_radius = max(Float32(0.05), (max_p - min_p).length() * Float32(0.5))
        out_mesh.bounds_verts = vcount

    def commit_lighting(mut self):
        """Select priority lights and upload camera, key light, point lights, and shadow spheres."""
        var per_pixel = self.active_quality >= 1 and not self.active_reduced
        self.lighting.per_pixel = per_pixel
        self.lighting.finish_frame()

        var to_key = Vec3(
            -self.active_profile.key_direction.x,
            -self.active_profile.key_direction.y,
            -self.active_profile.key_direction.z,
        ).normalized()
        self.fb.cpu_to_key = to_key
        self.fb.cpu_key_color = self.active_profile.key_color
        var fog_rgb = self.active_profile.fog_rgb()
        var raw_sky = Vec3(
            min(Float32(1.35), self.active_profile.hemisphere_sky.x * 1.24),
            min(Float32(1.35), self.active_profile.hemisphere_sky.y * 1.24),
            min(Float32(1.35), self.active_profile.hemisphere_sky.z * 1.24),
        )
        var sky_luma = raw_sky.x * 0.26 + raw_sky.y * 0.58 + raw_sky.z * 0.16
        var env_sky = raw_sky * 0.82 + Vec3(sky_luma, sky_luma, sky_luma) * 0.18
        var env_ground = self.active_profile.hemisphere_ground
        var raw_hor = Vec3(
            min(Float32(1.25), Float32(fog_rgb[0]) / 255.0 * 0.98),
            min(Float32(1.25), Float32(fog_rgb[1]) / 255.0 * 0.98),
            min(Float32(1.25), Float32(fog_rgb[2]) / 255.0 * 0.98),
        )
        var hor_luma = raw_hor.x * 0.26 + raw_hor.y * 0.58 + raw_hor.z * 0.16
        var env_horizon = raw_hor * 0.80 + Vec3(hor_luma, hor_luma, hor_luma) * 0.20
        self.fb.cpu_env_sky = env_sky
        self.fb.cpu_env_ground = env_ground
        self.fb.cpu_env_horizon = env_horizon
        self.fb.cpu_lights.clear()
        for i in range(self.lighting.selected_count):
            self.fb.cpu_lights.append(self.lighting.selected[i])
        var dyn_cpu = self.fb._compute_dynamic_reflection_lobe()
        self.fb.cpu_dyn_lobe_dir = dyn_cpu[0]
        self.fb.cpu_dyn_lobe_rgb = dyn_cpu[1]

        var sphere_count = min(MAX_SHADOW_SPHERES, len(self.sphere_centers))
        self.fb.shadow_sphere_count = sphere_count
        self.fb.cpu_sphere_centers.clear()
        self.fb.cpu_sphere_radii_sq.clear()
        for i in range(sphere_count):
            self.fb.cpu_sphere_centers.append(self.sphere_centers[i])
            self.fb.cpu_sphere_radii_sq.append(self.sphere_radii[i] * self.sphere_radii[i])

        # 2-Cascade Directional Shadow Map (CSM) texel-snapped orthographic basis
        var ref_up = Vec3(0.0, 0.0, 1.0) if abs(to_key.y) > Float32(0.92) else Vec3(0.0, 1.0, 0.0)
        var csm_right = cross(ref_up, to_key).normalized()
        var csm_up = cross(to_key, csm_right).normalized()

        var half0 = Float32(44.0)
        var texel0 = (half0 * Float32(2.0)) / Float32(CSM_RES)
        var anchor0 = self.fb.camera + self.fb.forward * Float32(22.0)
        var sx0 = floor(dot(anchor0, csm_right) / texel0 + Float32(0.5)) * texel0
        var sy0 = floor(dot(anchor0, csm_up) / texel0 + Float32(0.5)) * texel0
        var sz0 = dot(anchor0, to_key)
        var c0_center = csm_right * sx0 + csm_up * sy0 + to_key * sz0
        var inv_h0 = Float32(1.0) / half0
        var z_scale0 = Float32(0.5 / 220.0)

        var half1 = Float32(260.0)
        var texel1 = (half1 * Float32(2.0)) / Float32(CSM_RES)
        var anchor1 = self.fb.camera + self.fb.forward * Float32(95.0)
        var sx1 = floor(dot(anchor1, csm_right) / texel1 + Float32(0.5)) * texel1
        var sy1 = floor(dot(anchor1, csm_up) / texel1 + Float32(0.5)) * texel1
        var sz1 = dot(anchor1, to_key)
        var c1_center = csm_right * sx1 + csm_up * sy1 + to_key * sz1
        var inv_h1 = Float32(1.0) / half1
        var z_scale1 = Float32(0.5 / 520.0)

        self.fb.csm_right = csm_right
        self.fb.csm_up = csm_up
        self.fb.csm_c0_center = c0_center
        self.fb.csm_c0_inv_half = inv_h0
        self.fb.csm_c0_z_bias = sz0
        self.fb.csm_c0_z_scale = z_scale0
        self.fb.csm_c1_center = c1_center
        self.fb.csm_c1_inv_half = inv_h1
        self.fb.csm_c1_z_bias = sz1
        self.fb.csm_c1_z_scale = z_scale1

        if not self.fb.gpu.active:
            return

        self.fb.upload_view()
        self.fb.gpu.write_float(CONSTANT_CSM + 0, csm_right.x)
        self.fb.gpu.write_float(CONSTANT_CSM + 4, csm_right.y)
        self.fb.gpu.write_float(CONSTANT_CSM + 8, csm_right.z)
        self.fb.gpu.write_float(CONSTANT_CSM + 12, csm_up.x)
        self.fb.gpu.write_float(CONSTANT_CSM + 16, csm_up.y)
        self.fb.gpu.write_float(CONSTANT_CSM + 20, csm_up.z)
        self.fb.gpu.write_float(CONSTANT_CSM + 24, c0_center.x)
        self.fb.gpu.write_float(CONSTANT_CSM + 28, c0_center.y)
        self.fb.gpu.write_float(CONSTANT_CSM + 32, c0_center.z)
        self.fb.gpu.write_float(CONSTANT_CSM + 36, inv_h0)
        self.fb.gpu.write_float(CONSTANT_CSM + 40, sz0)
        self.fb.gpu.write_float(CONSTANT_CSM + 44, z_scale0)
        self.fb.gpu.write_float(CONSTANT_CSM + 48, c1_center.x)
        self.fb.gpu.write_float(CONSTANT_CSM + 52, c1_center.y)
        self.fb.gpu.write_float(CONSTANT_CSM + 56, c1_center.z)
        self.fb.gpu.write_float(CONSTANT_CSM + 60, inv_h1)
        self.fb.gpu.write_float(CONSTANT_CSM + 64, sz1)
        self.fb.gpu.write_float(CONSTANT_CSM + 68, z_scale1)
        self.fb.gpu.write_float(CONSTANT_KEY + 0, to_key.x)
        self.fb.gpu.write_float(CONSTANT_KEY + 4, to_key.y)
        self.fb.gpu.write_float(CONSTANT_KEY + 8, to_key.z)
        self.fb.gpu.write_float(CONSTANT_KEY + 12, self.active_profile.key_color.x)
        self.fb.gpu.write_float(CONSTANT_KEY + 16, self.active_profile.key_color.y)
        self.fb.gpu.write_float(CONSTANT_KEY + 20, self.active_profile.key_color.z)

        var legacy_light_count = min(32, self.lighting.selected_count)
        var cluster_light_count = min(MAX_CLUSTER_LIGHTS, self.lighting.selected_count)
        var w_word = CONSTANT_WORLD // 4
        self.fb.gpu.constants.write32(w_word + 0, 0)
        self.fb.gpu.constants.write32(w_word + 1, 0)
        self.fb.gpu.constants.write32(w_word + 2, 0)
        self.fb.gpu.constants.write32(w_word + 3, UInt32(sphere_count))
        self.fb.gpu.constants.write32(w_word + 4, 0)
        self.fb.gpu.constants.write32(w_word + 5, 0)
        self.fb.gpu.constants.write32(w_word + 6, UInt32(legacy_light_count))
        self.fb.gpu.constants.write32(w_word + 7, 0)
        self.fb.gpu.write_float(CONSTANT_ENV_SKY + 0, env_sky.x)
        self.fb.gpu.write_float(CONSTANT_ENV_SKY + 4, env_sky.y)
        self.fb.gpu.write_float(CONSTANT_ENV_SKY + 8, env_sky.z)
        self.fb.gpu.write_float(CONSTANT_ENV_GROUND + 0, env_ground.x)
        self.fb.gpu.write_float(CONSTANT_ENV_GROUND + 4, env_ground.y)
        self.fb.gpu.write_float(CONSTANT_ENV_GROUND + 8, env_ground.z)
        self.fb.gpu.write_float(CONSTANT_ENV_HORIZON + 0, env_horizon.x)
        self.fb.gpu.write_float(CONSTANT_ENV_HORIZON + 4, env_horizon.y)
        self.fb.gpu.write_float(CONSTANT_ENV_HORIZON + 8, env_horizon.z)

        self.fb.gpu.clear_tile_lights()
        for i in range(legacy_light_count):
            var pl = self.lighting.selected[i]
            var seg_len = pl.seg_delta.length()
            var mid_pos = pl.position + pl.seg_delta * Float32(0.5)
            var eff_rad = pl.radius + seg_len * Float32(0.5)
            var base = CONSTANT_LIGHTS + i * 32
            var inv_r2 = 1.0 / max(Float32(0.01), eff_rad * eff_rad)
            self.fb.gpu.write_float(base + 0, mid_pos.x)
            self.fb.gpu.write_float(base + 4, mid_pos.y)
            self.fb.gpu.write_float(base + 8, mid_pos.z)
            self.fb.gpu.write_float(base + 12, 0.0)
            self.fb.gpu.write_float(base + 16, pl.color.x * pl.intensity)
            self.fb.gpu.write_float(base + 20, pl.color.y * pl.intensity)
            self.fb.gpu.write_float(base + 24, pl.color.z * pl.intensity)
            self.fb.gpu.write_float(base + 28, inv_r2)
            self.fb.mark_light_on_screen(i, mid_pos, eff_rad)

        if self.fb.gpu.backend == BACKEND_VULKAN:
            self.fb.gpu.constants.write32((CONSTANT_V7_POST + 32) // 4, UInt32(cluster_light_count))
            var cw_lim = self.fb.render_w if self.fb.tsr_subnative_active else self.fb.width
            var ch_lim = self.fb.render_h if self.fb.tsr_subnative_active else self.fb.height
            var cf_val = self.fb.render_focal if self.fb.tsr_subnative_active else self.fb.focal
            var half_w = Float32(cw_lim) * Float32(0.5)
            var half_h = Float32(ch_lim) * Float32(0.5)
            for i in range(cluster_light_count):
                var pl = self.lighting.selected[i]
                var seg_len2 = pl.seg_delta.length_squared()
                var seg_half = Float32(0.5) * sqrt(seg_len2) if seg_len2 > Float32(0.0001) else Float32(0.0)
                var inv_r2 = Float32(1.0) / max(Float32(0.01), pl.radius * pl.radius)
                var cbase = CONSTANT_CLUSTER_LIGHTS + i * 48
                self.fb.gpu.write_float(cbase + 0, pl.position.x)
                self.fb.gpu.write_float(cbase + 4, pl.position.y)
                self.fb.gpu.write_float(cbase + 8, pl.position.z)
                self.fb.gpu.write_float(cbase + 12, inv_r2)
                self.fb.gpu.write_float(cbase + 16, pl.color.x * pl.intensity)
                self.fb.gpu.write_float(cbase + 20, pl.color.y * pl.intensity)
                self.fb.gpu.write_float(cbase + 24, pl.color.z * pl.intensity)
                self.fb.gpu.write_float(cbase + 28, seg_len2)
                self.fb.gpu.write_float(cbase + 32, pl.seg_delta.x)
                self.fb.gpu.write_float(cbase + 36, pl.seg_delta.y)
                self.fb.gpu.write_float(cbase + 40, pl.seg_delta.z)
                self.fb.gpu.write_float(cbase + 44, seg_half)

                var p0 = pl.position
                var p1 = pl.position + pl.seg_delta
                var rel0 = p0 - self.fb.camera
                var rel1 = p1 - self.fb.camera
                var vz0 = dot(rel0, self.fb.forward)
                var vz1 = dot(rel1, self.fb.forward)
                var min_vz = min(vz0, vz1) - pl.radius
                var max_vz = max(vz0, vz1) + pl.radius
                if max_vz < Float32(0.06):
                    continue
                if min_vz < Float32(0.25):
                    self.fb.gpu.mark_cluster_light(i, 0, 0, cw_lim - 1, ch_lim - 1, max(Float32(0.06), min_vz), max_vz)
                else:
                    var r0_sc = cf_val / max(Float32(0.25), vz0 - pl.radius)
                    var cx0 = half_w + dot(rel0, self.fb.right) * cf_val / vz0
                    var cy0 = half_h - dot(rel0, self.fb.up) * cf_val / vz0
                    var rch0 = pl.radius * r0_sc * Float32(1.15)
                    var lx = cx0 - rch0
                    var rx = cx0 + rch0
                    var ty = cy0 - rch0
                    var by = cy0 + rch0
                    if seg_len2 > Float32(0.0001):
                        var r1_sc = cf_val / max(Float32(0.25), vz1 - pl.radius)
                        var cx1 = half_w + dot(rel1, self.fb.right) * cf_val / vz1
                        var cy1 = half_h - dot(rel1, self.fb.up) * cf_val / vz1
                        var rch1 = pl.radius * r1_sc * Float32(1.15)
                        lx = min(lx, cx1 - rch1)
                        rx = max(rx, cx1 + rch1)
                        ty = min(ty, cy1 - rch1)
                        by = max(by, cy1 + rch1)
                    self.fb.gpu.mark_cluster_light(
                        i,
                        Int(max(Float32(-1.0), lx)),
                        Int(max(Float32(-1.0), ty)),
                        Int(min(Float32(cw_lim), rx)),
                        Int(min(Float32(ch_lim), by)),
                        min_vz,
                        max_vz,
                    )

        for i in range(sphere_count):
            var sbase = CONSTANT_SPHERES + i * 16
            var c = self.sphere_centers[i]
            var r = self.sphere_radii[i]
            self.fb.gpu.write_float(sbase + 0, c.x)
            self.fb.gpu.write_float(sbase + 4, c.y)
            self.fb.gpu.write_float(sbase + 8, c.z)
            self.fb.gpu.write_float(sbase + 12, r * r)

    def make_lit_vertex(
        self,
        world_pos: Vec3,
        world_normal: Vec3,
        u: Float32,
        v: Float32,
        color: UInt32,
        ao: Float32,
        roughness: Float32,
        metallic: Float32,
        emissive: Float32,
        unlit: Bool = False,
        no_fog: Bool = False,
    ) -> RenderVertex:
        var rv = RenderVertex(world_pos, world_normal, u, v, color)
        rv.roughness = roughness
        rv.metallic = metallic
        rv.emissive = emissive
        if unlit:
            rv.light = Vec3(1.0, 1.0, 1.0)
            rv.occlusion = -1.0
        else:
            rv.light = self.lighting.sample(world_pos, world_normal, ao)
            rv.occlusion = self.lighting.ambient_occlusion(ao) if self.lighting.per_pixel else Float32(-1.0)
        if no_fog:
            rv.fog = 0.0
        else:
            var depth = max(Float32(0.0), dot(world_pos - self.fb.camera, self.fb.forward))
            rv.fog = self.lighting.fog_amount(world_pos, depth)
        return rv^

    def sample_smoke_light(
        self,
        world_pos: Vec3,
        normal_hint: Vec3,
        hg_g: Float32 = 0.56,
        preserve_base: Bool = False,
    ) -> Vec3:
        """Evaluates wrapped directional sunlight, Henyey-Greenstein forward solar silver-lining,
        3-way sky/horizon/ground ambient, and nearby dynamic point/event lights for smoke & clouds."""
        var to_key = self.fb.cpu_to_key
        var key_col = self.fb.cpu_key_color
        var view_vec = world_pos - self.fb.camera
        var v_len2 = view_vec.length_squared()
        var view_dir = view_vec * (Float32(1.0) / sqrt(max(Float32(0.0001), v_len2)))
        var cos_th = max(Float32(-1.0), min(Float32(1.0), dot(view_dir, to_key)))
        var g = max(Float32(0.0), min(Float32(0.82), hg_g))
        var g2 = g * g
        var denom_base = max(Float32(0.08), Float32(1.0) + g2 - Float32(2.0) * g * cos_th)
        var hg_phase = min(Float32(0.42), (Float32(1.0) - g2) / (denom_base * sqrt(denom_base)) * Float32(0.065))

        var lr: Float32
        var lg: Float32
        var lb: Float32
        if preserve_base:
            lr = Float32(1.0) + key_col.x * hg_phase
            lg = Float32(1.0) + key_col.y * hg_phase
            lb = Float32(1.0) + key_col.z * (hg_phase * Float32(0.82))
        else:
            var up_w = max(Float32(0.0), min(Float32(1.0), normal_hint.y * Float32(0.5) + Float32(0.5)))
            var amb = self.fb.cpu_env_ground * (Float32(1.0) - up_w) + self.fb.cpu_env_sky * up_w
            amb = amb * Float32(0.68) + self.fb.cpu_env_horizon * Float32(0.32)
            var w_ndl = max(Float32(0.0), (dot(normal_hint, to_key) + Float32(0.62)) / Float32(1.62))
            lr = Float32(0.34) + amb.x * Float32(0.68) + key_col.x * (w_ndl * Float32(0.46) + hg_phase)
            lg = Float32(0.33) + amb.y * Float32(0.68) + key_col.y * (w_ndl * Float32(0.46) + hg_phase)
            lb = Float32(0.31) + amb.z * Float32(0.64) + key_col.z * (w_ndl * Float32(0.40) + hg_phase * Float32(0.82))

        var n_lights = len(self.fb.cpu_lights)
        for li in range(n_lights):
            var pl = self.fb.cpu_lights[li]
            var l0 = pl.position - world_pos
            var sl2 = pl.seg_delta.length_squared()
            var t_s = max(Float32(0.0), min(Float32(1.0), -dot(l0, pl.seg_delta) / sl2)) if sl2 > Float32(0.0001) else Float32(0.0)
            var d2 = (l0 + pl.seg_delta * t_s).length_squared()
            var r2 = max(Float32(0.04), pl.radius * pl.radius)
            if d2 >= r2:
                continue
            var att = Float32(1.0) - d2 / r2
            var w = att * att * pl.intensity * Float32(0.48)
            lr += pl.color.x * w
            lg += pl.color.y * w
            lb += pl.color.z * w

        return Vec3(
            min(Float32(2.35), max(Float32(0.18), lr)),
            min(Float32(2.35), max(Float32(0.18), lg)),
            min(Float32(2.15), max(Float32(0.16), lb)),
        )

    @always_inline
    def _submit_gpu_tri(
        mut self,
        sa: ScreenVertex,
        sb: ScreenVertex,
        sc: ScreenVertex,
        tex_offset: Int,
        tw: Int,
        th: Int,
        alpha: UInt8,
        double_sided: Bool,
        min_area: Float32 = 0.00001,
    ):
        var area = (sb.x - sa.x) * (sc.y - sa.y) - (sb.y - sa.y) * (sc.x - sa.x)
        if abs(area) <= min_area or (not double_sided and area <= 0.0):
            self.fb.culled_triangles += 1
            return
        var max_x = max(sa.x, max(sb.x, sc.x))
        var min_x = min(sa.x, min(sb.x, sc.x))
        var max_y = max(sa.y, max(sb.y, sc.y))
        var min_y = min(sa.y, min(sb.y, sc.y))
        if max_x < 0.0 or min_x >= Float32(self.fb.width) or max_y < 0.0 or min_y >= Float32(self.fb.height):
            self.fb.culled_triangles += 1
            return
        var slot = self.fb.gpu.reserve_slots(1)
        if slot >= 0:
            self.fb.fill_gpu_slot(slot, sa, sb, sc, tex_offset, tw, th, alpha, double_sided)
            self.fb.drawn_triangles += 1

    def draw_mesh(
        mut self,
        mut mesh: NativeMesh,
        mut textures: List[NativeTexture],
        transform: Mat4,
        tint: UInt32 = 0xFFFFFFFF,
        emissive_boost: Float32 = 0.0,
        alpha_override: Int = -1,
        unlit: Bool = False,
        no_fog: Bool = False,
        water_waves: Bool = False,
        force_depth_write: Bool = False,
    ):
        var vcount = mesh.vertex_count()
        if vcount < 3:
            return
        mesh.ensure_bounds()
        var world_center = transform.transform_point(mesh.bounds_center)
        var world_radius = mesh.bounds_radius * _mat4_max_scale(transform)

        var base_tint = mul_argb(mesh.base_color, tint)
        var alpha_byte = UInt8(255)
        if alpha_override >= 0:
            alpha_byte = UInt8(max(0, min(255, alpha_override)))
        elif mesh.alpha_mode == 2:
            alpha_byte = UInt8((base_tint >> 24) & 255)
            base_tint = 0xFF000000 | (base_tint & 0x00FFFFFF)

        var casts_csm = (
            (self.fb.gpu_effects & EFFECT_CSM) != 0
            and not unlit
            and alpha_byte >= 200
            and mesh.alpha_mode != 2
            and not water_waves
            and not self.fb.additive_blend
            and (not self.fb.no_depth_write or force_depth_write)
            and world_radius < Float32(420.0)
        )
        if casts_csm and world_radius > Float32(12.0) and world_radius < Float32(45.0):
            if (
                (self.fb.camera - world_center).length_squared() < world_radius * world_radius
                and world_center.y > self.fb.camera.y - Float32(1.0)
            ):
                casts_csm = False

        var in_c0 = False
        var in_c1 = False
        if casts_csm:
            var rel_c0 = world_center - self.fb.csm_c0_center
            var r0_lim = Float32(44.0) + world_radius
            if (
                abs(dot(rel_c0, self.fb.csm_right)) <= r0_lim
                and abs(dot(rel_c0, self.fb.csm_up)) <= r0_lim
                and abs(dot(rel_c0, self.fb.cpu_to_key)) <= Float32(215.0) + world_radius
            ):
                in_c0 = True
            var rel_c1 = world_center - self.fb.csm_c1_center
            var r1_lim = Float32(260.0) + world_radius
            if (
                abs(dot(rel_c1, self.fb.csm_right)) <= r1_lim
                and abs(dot(rel_c1, self.fb.csm_up)) <= r1_lim
                and abs(dot(rel_c1, self.fb.cpu_to_key)) <= Float32(500.0) + world_radius
            ):
                in_c1 = True

        var in_frustum = self.fb.sphere_visible(world_center, world_radius)
        var mesh_addr = UInt64(Int(Pointer(to=mesh)))
        var ord = self.mesh_ordinals.get(mesh_addr, 0)
        self.mesh_ordinals[mesh_addr] = ord + 1
        var inst_key = (mesh_addr << 16) ^ UInt64(ord & 0xFFFF)
        var track_vel = (
            not water_waves
            and not unlit
            and alpha_byte >= 200
            and mesh.alpha_mode != 2
            and (not self.fb.no_depth_write or force_depth_write)
            and not self.fb.additive_blend
        )
        var has_obj_vel = False
        var prev_mat = Mat4()
        if track_vel:
            self.cur_mesh_mats[inst_key] = transform.copy()
            if inst_key in self.prev_mesh_mats:
                prev_mat = self.prev_mesh_mats.get(inst_key, Mat4())
                has_obj_vel = _mat4_differs(transform, prev_mat)

        if in_frustum and self.fb.hiz_cull_enabled and world_radius < Float32(180.0):
            if self.is_sphere_occluded_hiz(world_center, world_radius):
                self.hiz_culled_meshes += 1
                in_frustum = False

        if not in_frustum and not in_c0 and not in_c1:
            self.fb.culled_triangles += mesh.triangle_count()
            return

        var has_normals = len(mesh.normals) >= vcount * 3
        var has_uvs = len(mesh.texcoords) >= vcount * 2
        var has_colors = len(mesh.colors) >= vcount * 4
        var has_ao = len(mesh.ao) >= vcount
        var need_csm = in_c0 or in_c1
        if need_csm and len(self.scratch_csm_proj) < vcount:
            self.scratch_csm_proj.resize(vcount, Vec3())

        var emissive = max(Float32(0.0), mesh.emissive_strength + emissive_boost)
        if in_frustum and len(self.scratch_cam) < vcount:
            self.scratch_cam.resize(vcount, CameraVertex())
            self.scratch_screen.resize(vcount, ScreenVertex())
            self.scratch_in_front.resize(vcount, 0)

        var csm_r = self.fb.csm_right
        var csm_u = self.fb.csm_up
        var csm_k = self.fb.cpu_to_key
        var near = Float32(0.06)
        var half_wf = Float32(self.fb.width) * Float32(0.5)
        var half_hf = Float32(self.fb.height) * Float32(0.5)
        var any_in_front = False
        var pos_ptr = mesh.positions.unsafe_ptr()
        for vi in range(vcount):
            var p_off = vi * 3
            var lp = Vec3(pos_ptr[unsafe_offset=p_off], pos_ptr[unsafe_offset=p_off + 1], pos_ptr[unsafe_offset=p_off + 2])
            var wp = transform.transform_point(lp)
            if need_csm:
                self.scratch_csm_proj[vi] = Vec3(dot(wp, csm_r), dot(wp, csm_u), dot(wp, csm_k))
            if not in_frustum:
                continue
            var ln = Vec3(mesh.normals[p_off], mesh.normals[p_off + 1], mesh.normals[p_off + 2]) if has_normals else Vec3(0, 1, 0)
            var wn = transform.transform_vector(ln).normalized()
            var u = mesh.texcoords[vi * 2] if has_uvs else Float32(0.0)
            var v = mesh.texcoords[vi * 2 + 1] if has_uvs else Float32(0.0)
            var vcol = base_tint
            if has_colors:
                var c_off = vi * 4
                var cr = UInt32(mesh.colors[c_off])
                var cg = UInt32(mesh.colors[c_off + 1])
                var cb = UInt32(mesh.colors[c_off + 2])
                var ca = UInt32(mesh.colors[c_off + 3])
                vcol = mul_argb(base_tint, (ca << 24) | (cr << 16) | (cg << 8) | cb)
            var ao = Float32(mesh.ao[vi]) / 255.0 if has_ao else Float32(1.0)
            var rv = self.make_lit_vertex(
                wp,
                wn,
                u,
                v,
                vcol,
                ao,
                mesh.roughness,
                mesh.metallic,
                emissive,
                unlit=unlit,
                no_fog=no_fog,
            )
            var cv = self.fb.camera_vertex(rv)
            if has_obj_vel:
                var cur_vz = max(near, cv.position.z)
                var cur_sx = half_wf + cv.position.x * self.fb.focal / cur_vz
                var cur_sy = half_hf - cv.position.y * self.fb.focal / cur_vz
                var wp_prev = prev_mat.transform_point(lp)
                var rel_prev = wp_prev - self.fb.prev_camera
                var prev_vz = max(Float32(0.1), dot(rel_prev, self.fb.prev_forward))
                var prev_sx = half_wf + dot(rel_prev, self.fb.prev_right) * self.fb.prev_focal / prev_vz
                var prev_sy = half_hf - dot(rel_prev, self.fb.prev_up) * self.fb.prev_focal / prev_vz
                cv.vel_x = cur_sx - prev_sx
                cv.vel_y = cur_sy - prev_sy
                cv.has_obj_vel = True
            self.scratch_cam[vi] = cv
            if cv.position.z >= near:
                self.scratch_in_front[vi] = 1
                self.scratch_screen[vi] = self.fb.project(cv)
                any_in_front = True
            else:
                self.scratch_in_front[vi] = 0

        var tri_count = mesh.triangle_count()
        var indexed = len(mesh.indices) >= tri_count * 3
        var use_gpu = self.fb.gpu.active

        if need_csm:
            var half_res = Float32(CSM_RES) * Float32(0.5)
            var off_x0 = dot(self.fb.csm_c0_center, csm_r)
            var off_y0 = dot(self.fb.csm_c0_center, csm_u)
            var sc_xy0 = self.fb.csm_c0_inv_half * half_res
            var zb0 = self.fb.csm_c0_z_bias
            var zs0 = self.fb.csm_c0_z_scale
            var off_x1 = dot(self.fb.csm_c1_center, csm_r)
            var off_y1 = dot(self.fb.csm_c1_center, csm_u)
            var sc_xy1 = self.fb.csm_c1_inv_half * half_res
            var zb1 = self.fb.csm_c1_z_bias
            var zs1 = self.fb.csm_c1_z_scale

            for ti in range(tri_count):
                var i0 = mesh.indices[ti * 3] if indexed else ti * 3
                var i1 = mesh.indices[ti * 3 + 1] if indexed else ti * 3 + 1
                var i2 = mesh.indices[ti * 3 + 2] if indexed else ti * 3 + 2
                if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= vcount or i1 >= vcount or i2 >= vcount:
                    continue
                var pa = self.scratch_csm_proj[i0]
                var pb = self.scratch_csm_proj[i1]
                var pc = self.scratch_csm_proj[i2]
                if in_c0:
                    var ax0 = (pa.x - off_x0) * sc_xy0 + half_res
                    var ay0 = half_res - (pa.y - off_y0) * sc_xy0
                    var az0 = (pa.z - zb0) * zs0 + Float32(0.5)
                    var bx0 = (pb.x - off_x0) * sc_xy0 + half_res
                    var by0 = half_res - (pb.y - off_y0) * sc_xy0
                    var bz0 = (pb.z - zb0) * zs0 + Float32(0.5)
                    var cx0 = (pc.x - off_x0) * sc_xy0 + half_res
                    var cy0 = half_res - (pc.y - off_y0) * sc_xy0
                    var cz0 = (pc.z - zb0) * zs0 + Float32(0.5)
                    if use_gpu:
                        self.fb.gpu.add_csm_triangle(0, ax0, ay0, az0, bx0, by0, bz0, cx0, cy0, cz0)
                    else:
                        self.fb.add_cpu_csm_triangle(0, ax0, ay0, az0, bx0, by0, bz0, cx0, cy0, cz0)
                if in_c1:
                    var ax1 = (pa.x - off_x1) * sc_xy1 + half_res
                    var ay1 = half_res - (pa.y - off_y1) * sc_xy1
                    var az1 = (pa.z - zb1) * zs1 + Float32(0.5)
                    var bx1 = (pb.x - off_x1) * sc_xy1 + half_res
                    var by1 = half_res - (pb.y - off_y1) * sc_xy1
                    var bz1 = (pb.z - zb1) * zs1 + Float32(0.5)
                    var cx1 = (pc.x - off_x1) * sc_xy1 + half_res
                    var cy1 = half_res - (pc.y - off_y1) * sc_xy1
                    var cz1 = (pc.z - zb1) * zs1 + Float32(0.5)
                    if use_gpu:
                        self.fb.gpu.add_csm_triangle(1, ax1, ay1, az1, bx1, by1, bz1, cx1, cy1, cz1)
                    else:
                        self.fb.add_cpu_csm_triangle(1, ax1, ay1, az1, bx1, by1, bz1, cx1, cy1, cz1)

        if not in_frustum or not any_in_front:
            self.fb.culled_triangles += tri_count
            return

        var prev_noz = self.fb.no_depth_write
        var prev_waves = self.fb.water_wave_normals
        var prev_aniso = self.fb.mesh_anisotropy
        if force_depth_write:
            self.fb.no_depth_write = False
        elif mesh.alpha_mode == 2 or alpha_byte < 245:
            self.fb.no_depth_write = True
        if water_waves:
            self.fb.water_wave_normals = True
        elif not unlit and mesh.metallic >= Float32(0.35):
            self.fb.mesh_anisotropy = max(prev_aniso, Float32(0.38))
        var has_tex = (
            mesh.texture >= 0
            and mesh.texture < len(textures)
            and textures[mesh.texture].width > 0
            and textures[mesh.texture].height > 0
            and len(textures[mesh.texture].pixels) >= textures[mesh.texture].width * textures[mesh.texture].height
        )
        var tw = textures[mesh.texture].width if has_tex else 0
        var th = textures[mesh.texture].height if has_tex else 0
        var tex_offset = -1
        if use_gpu and has_tex:
            tex_offset = self.fb.gpu.texture_offset(Int(textures[mesh.texture].pixels.unsafe_ptr()), tw, th)

        var prev_norm_ptr = self.fb.vfx_norm_orm_ptr
        var prev_norm_dim = self.fb.vfx_norm_orm_dim
        var prev_norm_meta = self.fb.vfx_norm_orm_gpu_meta
        self.fb.vfx_norm_orm_ptr = 0
        self.fb.vfx_norm_orm_dim = 0
        self.fb.vfx_norm_orm_gpu_meta = 0
        if (
            not unlit
            and not water_waves
            and mesh.normal_orm_texture >= 0
            and mesh.normal_orm_texture < len(textures)
        ):
            var nw = textures[mesh.normal_orm_texture].width
            var nh = textures[mesh.normal_orm_texture].height
            if (
                nw >= 8
                and nw <= 1024
                and nw == nh
                and (nw & (nw - 1)) == 0
                and len(textures[mesh.normal_orm_texture].pixels) >= nw * nh
            ):
                var n_ptr_addr = Int(textures[mesh.normal_orm_texture].pixels.unsafe_ptr())
                self.fb.vfx_norm_orm_ptr = n_ptr_addr
                self.fb.vfx_norm_orm_dim = nw
                if use_gpu:
                    var n_off = self.fb.gpu.texture_offset(n_ptr_addr, nw, nh)
                    if n_off >= 0:
                        var n_log2 = 0
                        while (1 << n_log2) < nw and n_log2 < 11:
                            n_log2 += 1
                        self.fb.vfx_norm_orm_gpu_meta = (UInt32(n_log2) << 28) | UInt32(n_off >> 2)

        var clipped = Array[CameraVertex, 4](fill=CameraVertex())
        for ti in range(tri_count):
            var i0 = mesh.indices[ti * 3] if indexed else ti * 3
            var i1 = mesh.indices[ti * 3 + 1] if indexed else ti * 3 + 1
            var i2 = mesh.indices[ti * 3 + 2] if indexed else ti * 3 + 2
            if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= vcount or i1 >= vcount or i2 >= vcount:
                continue
            var f0 = self.scratch_in_front[i0]
            var f1 = self.scratch_in_front[i1]
            var f2 = self.scratch_in_front[i2]
            var fsum = Int(f0) + Int(f1) + Int(f2)
            if fsum == 0:
                self.fb.culled_triangles += 1
                continue
            if fsum == 3:
                var sa = self.scratch_screen[i0]
                var sb = self.scratch_screen[i1]
                var sc = self.scratch_screen[i2]
                if use_gpu:
                    self._submit_gpu_tri(sa, sb, sc, tex_offset, tw, th, alpha_byte, mesh.double_sided)
                elif has_tex:
                    self.fb.raster_triangle(sa, sb, sc, textures[mesh.texture].pixels, tw, th, alpha_byte, mesh.double_sided)
                else:
                    self.fb.raster_triangle(sa, sb, sc, self.empty_tex, 0, 0, alpha_byte, mesh.double_sided)
                continue

            var n_clip = _clip_near_tri(self.scratch_cam[i0], self.scratch_cam[i1], self.scratch_cam[i2], clipped)
            if n_clip < 3:
                self.fb.culled_triangles += 1
                continue
            var s0 = self.fb.project(clipped[0])
            for k in range(1, n_clip - 1):
                var s1 = self.fb.project(clipped[k])
                var s2 = self.fb.project(clipped[k + 1])
                if use_gpu:
                    self._submit_gpu_tri(s0, s1, s2, tex_offset, tw, th, alpha_byte, mesh.double_sided)
                elif has_tex:
                    self.fb.raster_triangle(s0, s1, s2, textures[mesh.texture].pixels, tw, th, alpha_byte, mesh.double_sided)
                else:
                    self.fb.raster_triangle(s0, s1, s2, self.empty_tex, 0, 0, alpha_byte, mesh.double_sided)
        self.fb.no_depth_write = prev_noz
        self.fb.water_wave_normals = prev_waves
        self.fb.mesh_anisotropy = prev_aniso
        self.fb.vfx_norm_orm_ptr = prev_norm_ptr
        self.fb.vfx_norm_orm_dim = prev_norm_dim
        self.fb.vfx_norm_orm_gpu_meta = prev_norm_meta

    def draw_model(
        mut self,
        mut model: NativeModel,
        world_mat: Mat4,
        tint: UInt32 = 0xFFFFFFFF,
        emissive_boost: Float32 = 0.0,
        alpha_override: Int = -1,
        unlit: Bool = False,
        no_fog: Bool = False,
        water_waves: Bool = False,
        force_depth_write: Bool = False,
    ):
        if len(model.meshes) == 0:
            return
        if len(model.nodes) == 0:
            for pass_blend in range(2):
                var want_blend = pass_blend == 1
                for mi in range(len(model.meshes)):
                    var is_blend = (model.meshes[mi].alpha_mode == 2 or (alpha_override >= 0 and alpha_override < 255)) and not force_depth_write
                    if is_blend == want_blend:
                        self.draw_mesh(
                            model.meshes[mi],
                            model.textures,
                            world_mat,
                            tint=tint,
                            emissive_boost=emissive_boost,
                            alpha_override=alpha_override,
                            unlit=unlit,
                            no_fog=no_fog,
                            water_waves=water_waves,
                            force_depth_write=force_depth_write,
                        )
            return

        var globals = model.global_transforms()
        for pass_blend in range(2):
            var want_blend = pass_blend == 1
            for ni in range(len(model.nodes)):
                var node_world = world_mat.multiply(globals[ni])
                var count = len(model.nodes[ni].mesh_indices)
                if count > 0:
                    for k in range(count):
                        var mi = model.nodes[ni].mesh_indices[k]
                        if mi >= 0 and mi < len(model.meshes):
                            var is_blend = (model.meshes[mi].alpha_mode == 2 or (alpha_override >= 0 and alpha_override < 255)) and not force_depth_write
                            if is_blend == want_blend:
                                self.draw_mesh(
                                    model.meshes[mi],
                                    model.textures,
                                    node_world,
                                    tint=tint,
                                    emissive_boost=emissive_boost,
                                    alpha_override=alpha_override,
                                    unlit=unlit,
                                    no_fog=no_fog,
                                    water_waves=water_waves,
                                    force_depth_write=force_depth_write,
                                )
                elif model.nodes[ni].mesh >= 0 and model.nodes[ni].mesh < len(model.meshes):
                    var mi = model.nodes[ni].mesh
                    var is_blend = (model.meshes[mi].alpha_mode == 2 or (alpha_override >= 0 and alpha_override < 255)) and not force_depth_write
                    if is_blend == want_blend:
                        self.draw_mesh(
                            model.meshes[mi],
                            model.textures,
                            node_world,
                            tint=tint,
                            emissive_boost=emissive_boost,
                            alpha_override=alpha_override,
                            unlit=unlit,
                            no_fog=no_fog,
                            water_waves=water_waves,
                            force_depth_write=force_depth_write,
                        )

    def draw_planar_reflection_mesh(
        mut self,
        mut mesh: NativeMesh,
        mut textures: List[NativeTexture],
        transform: Mat4,
        plane_y: Float32,
        tint: UInt32 = 0xFFDCE8F8,
    ):
        """Mirrors `mesh` across horizontal plane `y = plane_y` with inverted winding & Z-buffered PBR."""
        var vcount = mesh.vertex_count()
        if vcount < 3:
            return
        mesh.ensure_bounds()
        var world_center = transform.transform_point(mesh.bounds_center)
        var world_radius = mesh.bounds_radius * _mat4_max_scale(transform)
        if world_center.y + world_radius < plane_y - 0.05:
            return
        var refl_center = Vec3(world_center.x, 2.0 * plane_y - world_center.y, world_center.z)
        if not self.fb.sphere_visible(refl_center, world_radius):
            self.fb.culled_triangles += mesh.triangle_count()
            return

        var has_normals = len(mesh.normals) >= vcount * 3
        var has_uvs = len(mesh.texcoords) >= vcount * 2
        var has_colors = len(mesh.colors) >= vcount * 4
        var has_ao = len(mesh.ao) >= vcount
        var base_tint = mul_argb(mesh.base_color, tint)
        var alpha_byte = UInt8(255)
        if mesh.alpha_mode == 2:
            alpha_byte = UInt8((base_tint >> 24) & 255)
            base_tint = 0xFF000000 | (base_tint & 0x00FFFFFF)
        var emissive = max(Float32(0.0), mesh.emissive_strength)

        if len(self.scratch_cam) < vcount:
            self.scratch_cam.resize(vcount, CameraVertex())
            self.scratch_screen.resize(vcount, ScreenVertex())
            self.scratch_in_front.resize(vcount, 0)
        if len(self.scratch_dists) < vcount:
            self.scratch_dists.resize(vcount, 0.0)

        var near = Float32(0.06)
        var any_in_front = False
        var pos_ptr = mesh.positions.unsafe_ptr()
        for vi in range(vcount):
            var p_off = vi * 3
            var lp = Vec3(pos_ptr[unsafe_offset=p_off], pos_ptr[unsafe_offset=p_off + 1], pos_ptr[unsafe_offset=p_off + 2])
            var ln = Vec3(mesh.normals[p_off], mesh.normals[p_off + 1], mesh.normals[p_off + 2]) if has_normals else Vec3(0, 1, 0)
            var wp = transform.transform_point(lp)
            var wn = transform.transform_vector(ln).normalized()
            self.scratch_dists[vi] = wp.y - plane_y
            var wp_r = Vec3(wp.x, min(plane_y - Float32(0.004), 2.0 * plane_y - wp.y), wp.z)
            var wn_r = Vec3(wn.x, -wn.y, wn.z)
            var u = mesh.texcoords[vi * 2] if has_uvs else Float32(0.0)
            var v = mesh.texcoords[vi * 2 + 1] if has_uvs else Float32(0.0)
            var vcol = base_tint
            if has_colors:
                var c_off = vi * 4
                vcol = mul_argb(
                    base_tint,
                    (UInt32(mesh.colors[c_off + 3]) << 24)
                    | (UInt32(mesh.colors[c_off]) << 16)
                    | (UInt32(mesh.colors[c_off + 1]) << 8)
                    | UInt32(mesh.colors[c_off + 2]),
                )
            var ao = Float32(mesh.ao[vi]) / 255.0 if has_ao else Float32(1.0)
            var rv = self.make_lit_vertex(
                wp_r, wn_r, u, v, vcol, ao, mesh.roughness, mesh.metallic, emissive
            )
            var cv = self.fb.camera_vertex(rv)
            self.scratch_cam[vi] = cv
            if cv.position.z >= near:
                self.scratch_in_front[vi] = 1
                self.scratch_screen[vi] = self.fb.project(cv)
                any_in_front = True
            else:
                self.scratch_in_front[vi] = 0

        if not any_in_front:
            return

        var prev_noz = self.fb.no_depth_write
        if mesh.alpha_mode == 2 or alpha_byte < 245:
            self.fb.no_depth_write = True
        var has_tex = (
            mesh.texture >= 0
            and mesh.texture < len(textures)
            and textures[mesh.texture].width > 0
            and textures[mesh.texture].height > 0
            and len(textures[mesh.texture].pixels) >= textures[mesh.texture].width * textures[mesh.texture].height
        )
        var tw = textures[mesh.texture].width if has_tex else 0
        var th = textures[mesh.texture].height if has_tex else 0
        var use_gpu = self.fb.gpu.active
        var tex_offset = -1
        if use_gpu and has_tex:
            tex_offset = self.fb.gpu.texture_offset(Int(textures[mesh.texture].pixels.unsafe_ptr()), tw, th)

        var tri_count = mesh.triangle_count()
        var indexed = len(mesh.indices) >= tri_count * 3
        var clipped = Array[CameraVertex, 4](fill=CameraVertex())
        for ti in range(tri_count):
            var r0 = mesh.indices[ti * 3] if indexed else ti * 3
            var r1 = mesh.indices[ti * 3 + 1] if indexed else ti * 3 + 1
            var r2 = mesh.indices[ti * 3 + 2] if indexed else ti * 3 + 2
            if r0 < 0 or r1 < 0 or r2 < 0 or r0 >= vcount or r1 >= vcount or r2 >= vcount:
                continue
            if max(self.scratch_dists[r0], max(self.scratch_dists[r1], self.scratch_dists[r2])) < 0.005:
                continue
            # Swap (r1, r2) to compensate for mirror reflection determinant = -1
            var i0 = r0
            var i1 = r2
            var i2 = r1
            var fsum = Int(self.scratch_in_front[i0]) + Int(self.scratch_in_front[i1]) + Int(self.scratch_in_front[i2])
            if fsum == 0:
                continue
            if fsum == 3:
                var sa = self.scratch_screen[i0]
                var sb = self.scratch_screen[i1]
                var sc = self.scratch_screen[i2]
                if use_gpu:
                    self._submit_gpu_tri(sa, sb, sc, tex_offset, tw, th, alpha_byte, mesh.double_sided)
                elif has_tex:
                    self.fb.raster_triangle(sa, sb, sc, textures[mesh.texture].pixels, tw, th, alpha_byte, mesh.double_sided)
                else:
                    self.fb.raster_triangle(sa, sb, sc, self.empty_tex, 0, 0, alpha_byte, mesh.double_sided)
                continue
            var n_clip = _clip_near_tri(self.scratch_cam[i0], self.scratch_cam[i1], self.scratch_cam[i2], clipped)
            if n_clip < 3:
                continue
            var s0 = self.fb.project(clipped[0])
            for k in range(1, n_clip - 1):
                var s1 = self.fb.project(clipped[k])
                var s2 = self.fb.project(clipped[k + 1])
                if use_gpu:
                    self._submit_gpu_tri(s0, s1, s2, tex_offset, tw, th, alpha_byte, mesh.double_sided)
                elif has_tex:
                    self.fb.raster_triangle(s0, s1, s2, textures[mesh.texture].pixels, tw, th, alpha_byte, mesh.double_sided)
                else:
                    self.fb.raster_triangle(s0, s1, s2, self.empty_tex, 0, 0, alpha_byte, mesh.double_sided)
        self.fb.no_depth_write = prev_noz

    def draw_planar_reflection_model(
        mut self,
        mut model: NativeModel,
        world_mat: Mat4,
        plane_y: Float32,
        tint: UInt32 = 0xFFDCE8F8,
    ):
        """Renders a 3D mirror reflection of `model` beneath horizontal plane `y = plane_y`."""
        if len(model.meshes) == 0:
            return
        if len(model.nodes) == 0:
            for pass_blend in range(2):
                var want_blend = pass_blend == 1
                for mi in range(len(model.meshes)):
                    if (model.meshes[mi].alpha_mode == 2) == want_blend:
                        self.draw_planar_reflection_mesh(model.meshes[mi], model.textures, world_mat, plane_y, tint)
            return
        var globals = model.global_transforms()
        for pass_blend in range(2):
            var want_blend = pass_blend == 1
            for ni in range(len(model.nodes)):
                var node_world = world_mat.multiply(globals[ni])
                var count = len(model.nodes[ni].mesh_indices)
                if count > 0:
                    for k in range(count):
                        var mi = model.nodes[ni].mesh_indices[k]
                        if mi >= 0 and mi < len(model.meshes) and (model.meshes[mi].alpha_mode == 2) == want_blend:
                            self.draw_planar_reflection_mesh(model.meshes[mi], model.textures, node_world, plane_y, tint)
                elif model.nodes[ni].mesh >= 0 and model.nodes[ni].mesh < len(model.meshes):
                    var mi = model.nodes[ni].mesh
                    if (model.meshes[mi].alpha_mode == 2) == want_blend:
                        self.draw_planar_reflection_mesh(model.meshes[mi], model.textures, node_world, plane_y, tint)

    def draw_planar_shadow_mesh(
        mut self,
        mut mesh: NativeMesh,
        transform: Mat4,
        plane_point: Vec3,
        plane_normal: Vec3 = Vec3(0.0, 1.0, 0.0),
        shadow_color: UInt32 = 0xFF050A14,
        max_alpha: UInt8 = 138,
        max_height: Float32 = 70.0,
        bias: Float32 = 0.06,
    ):
        """Projects `mesh` along the active key light onto `(plane_point, plane_normal)`."""
        var vcount = mesh.vertex_count()
        if vcount < 3 or max_alpha == 0:
            return
        if mesh.alpha_mode == 2 and mesh.emissive_strength > 0.8:
            return
        var n = plane_normal.normalized()
        var light_dir = self.active_profile.key_direction.normalized()
        var n_dot_l = dot(n, light_dir)
        if n_dot_l > -0.12:
            light_dir = Vec3(light_dir.x - n.x * 0.75, light_dir.y - n.y * 0.75, light_dir.z - n.z * 0.75).normalized()
            n_dot_l = min(Float32(-0.18), dot(n, light_dir))
        var inv_neg_ndl = Float32(1.0) / (-n_dot_l)
        var plane_bias = n * max(bias, Float32(0.08))

        mesh.ensure_bounds()
        var world_center = transform.transform_point(mesh.bounds_center)
        var world_radius = mesh.bounds_radius * _mat4_max_scale(transform)
        var d_center = dot(world_center - plane_point, n)
        if d_center - world_radius > max_height or d_center + world_radius < -0.5:
            return
        var proj_center = world_center + light_dir * (max(Float32(0.0), d_center) * inv_neg_ndl) + plane_bias
        var proj_radius = world_radius * (Float32(1.0) + inv_neg_ndl) + Float32(0.25)
        if not self.fb.sphere_visible(proj_center, proj_radius):
            return

        var col = 0xFF000000 | (shadow_color & 0x00FFFFFF)
        var inv_max_h = Float32(1.0) / max(Float32(0.5), max_height)

        if len(self.scratch_cam) < vcount:
            self.scratch_cam.resize(vcount, CameraVertex())
            self.scratch_screen.resize(vcount, ScreenVertex())
            self.scratch_in_front.resize(vcount, 0)
        if len(self.scratch_world_pts) < vcount:
            self.scratch_world_pts.resize(vcount, Vec3())
            self.scratch_dists.resize(vcount, 0.0)

        var near = Float32(0.06)
        var any_in_front = False
        var pos_ptr = mesh.positions.unsafe_ptr()
        for vi in range(vcount):
            var p_off = vi * 3
            var lp = Vec3(pos_ptr[unsafe_offset=p_off], pos_ptr[unsafe_offset=p_off + 1], pos_ptr[unsafe_offset=p_off + 2])
            var wp = transform.transform_point(lp)
            self.scratch_world_pts[vi] = wp
            var d = dot(wp - plane_point, n)
            self.scratch_dists[vi] = d
            var ray_t = max(Float32(0.0), d) * inv_neg_ndl
            var p_proj = wp + light_dir * ray_t + plane_bias
            var rv = self.make_lit_vertex(
                p_proj,
                n,
                0.0,
                0.0,
                col,
                1.0,
                1.0,
                0.0,
                0.0,
                unlit=True,
                no_fog=False,
            )
            var cv = self.fb.camera_vertex(rv)
            self.scratch_cam[vi] = cv
            if cv.position.z >= near:
                self.scratch_in_front[vi] = 1
                self.scratch_screen[vi] = self.fb.project(cv)
                any_in_front = True
            else:
                self.scratch_in_front[vi] = 0

        if not any_in_front:
            return

        var prev_noz = self.fb.no_depth_write
        self.fb.no_depth_write = True
        var use_gpu = self.fb.gpu.active
        var tri_count = mesh.triangle_count()
        var indexed = len(mesh.indices) >= tri_count * 3
        var clipped = Array[CameraVertex, 4](fill=CameraVertex())
        for ti in range(tri_count):
            var i0 = mesh.indices[ti * 3] if indexed else ti * 3
            var i1 = mesh.indices[ti * 3 + 1] if indexed else ti * 3 + 1
            var i2 = mesh.indices[ti * 3 + 2] if indexed else ti * 3 + 2
            if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= vcount or i1 >= vcount or i2 >= vcount:
                continue
            var fsum = Int(self.scratch_in_front[i0]) + Int(self.scratch_in_front[i1]) + Int(self.scratch_in_front[i2])
            if fsum == 0:
                continue
            # Only light-facing front triangles cast planar shadows (avoids 2x backface & internal-bone stacking).
            var face_n = cross(self.scratch_world_pts[i1] - self.scratch_world_pts[i0], self.scratch_world_pts[i2] - self.scratch_world_pts[i0])
            if not mesh.double_sided and dot(face_n, light_dir) >= 0.0:
                continue
            var d0 = self.scratch_dists[i0]
            var d1 = self.scratch_dists[i1]
            var d2 = self.scratch_dists[i2]
            if max(d0, max(d1, d2)) < -0.5 or min(d0, min(d1, d2)) > max_height:
                continue
            var avg_d = max(Float32(0.0), (d0 + d1 + d2) * Float32(0.33333334))
            var atten = max(Float32(0.0), min(Float32(1.0), Float32(1.0) - avg_d * inv_max_h))
            var eff_a = UInt8(max(0, min(110, Int(Float32(max_alpha) * atten * 0.56))))
            if eff_a < 4:
                continue
            if fsum == 3:
                var sa = self.scratch_screen[i0]
                var sb = self.scratch_screen[i1]
                var sc = self.scratch_screen[i2]
                if use_gpu:
                    self._submit_gpu_tri(sa, sb, sc, -1, 0, 0, eff_a, True, min_area=0.25)
                else:
                    self.fb.raster_triangle(sa, sb, sc, self.empty_tex, 0, 0, eff_a, True)
                continue

            var n_clip = _clip_near_tri(self.scratch_cam[i0], self.scratch_cam[i1], self.scratch_cam[i2], clipped)
            if n_clip < 3:
                continue
            var s0 = self.fb.project(clipped[0])
            for k in range(1, n_clip - 1):
                var s1 = self.fb.project(clipped[k])
                var s2 = self.fb.project(clipped[k + 1])
                if use_gpu:
                    self._submit_gpu_tri(s0, s1, s2, -1, 0, 0, eff_a, True, min_area=0.25)
                else:
                    self.fb.raster_triangle(s0, s1, s2, self.empty_tex, 0, 0, eff_a, True)
        self.fb.no_depth_write = prev_noz

    def draw_planar_shadow_model(
        mut self,
        mut model: NativeModel,
        world_mat: Mat4,
        plane_point: Vec3,
        plane_normal: Vec3 = Vec3(0.0, 1.0, 0.0),
        shadow_color: UInt32 = 0xFF050A14,
        max_alpha: UInt8 = 138,
        max_height: Float32 = 70.0,
        bias: Float32 = 0.06,
    ):
        """Projects every articulated node of `model` onto a deck/ground shadow receiver plane."""
        if len(model.meshes) == 0 or max_alpha == 0:
            return
        if len(model.nodes) == 0:
            for mi in range(len(model.meshes)):
                self.draw_planar_shadow_mesh(
                    model.meshes[mi],
                    world_mat,
                    plane_point,
                    plane_normal=plane_normal,
                    shadow_color=shadow_color,
                    max_alpha=max_alpha,
                    max_height=max_height,
                    bias=bias,
                )
            return

        var globals = model.global_transforms()
        for ni in range(len(model.nodes)):
            var node_world = world_mat.multiply(globals[ni])
            var count = len(model.nodes[ni].mesh_indices)
            if count > 0:
                for k in range(count):
                    var mi = model.nodes[ni].mesh_indices[k]
                    if mi >= 0 and mi < len(model.meshes):
                        self.draw_planar_shadow_mesh(
                            model.meshes[mi],
                            node_world,
                            plane_point,
                            plane_normal=plane_normal,
                            shadow_color=shadow_color,
                            max_alpha=max_alpha,
                            max_height=max_height,
                            bias=bias,
                        )
            elif model.nodes[ni].mesh >= 0 and model.nodes[ni].mesh < len(model.meshes):
                self.draw_planar_shadow_mesh(
                    model.meshes[model.nodes[ni].mesh],
                    node_world,
                    plane_point,
                    plane_normal=plane_normal,
                    shadow_color=shadow_color,
                    max_alpha=max_alpha,
                    max_height=max_height,
                    bias=bias,
                )

    def draw_billboard(
        mut self,
        center: Vec3,
        half_size: Float32,
        fx_kind: Int,
        color: UInt32,
        alpha: UInt8,
        emissive: Float32 = 1.35,
        rotation: Float32 = 0.0,
        additive: Bool = False,
        soft_depth: Float32 = 0.0,
        erode: Float32 = 0.0,
        blackbody_heat: Float32 = -1.0,
        refract_strength: Float32 = 0.0,
        lit_smoke: Bool = False,
        preserve_smoke_base: Bool = False,
        no_fog: Bool = True,
    ):
        if half_size <= 0.001 or alpha == 0 or fx_kind < 0 or fx_kind >= len(self.fx.pixels):
            return
        var c_rot = cos(rotation)
        var s_rot = sin(rotation)
        var rx = (self.fb.right * c_rot + self.fb.up * s_rot) * half_size
        var ry = (self.fb.up * c_rot - self.fb.right * s_rot) * half_size
        var n = Vec3(-self.fb.forward.x, -self.fb.forward.y, -self.fb.forward.z)
        var col = 0xFF000000 | (color & 0x00FFFFFF)

        var p0 = center - rx + ry
        var p1 = center + rx + ry
        var p2 = center + rx - ry
        var p3 = center - rx - ry

        var v0 = self.make_lit_vertex(p0, n, 0.0, 0.0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=no_fog)
        var v1 = self.make_lit_vertex(p1, n, 1.0, 0.0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=no_fog)
        var v2 = self.make_lit_vertex(p2, n, 1.0, 1.0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=no_fog)
        var v3 = self.make_lit_vertex(p3, n, 0.0, 1.0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=no_fog)
        if lit_smoke:
            v0.light = self.sample_smoke_light(p0, Vec3(0.0, 0.85, 0.0), preserve_base=preserve_smoke_base)
            v1.light = self.sample_smoke_light(p1, Vec3(0.0, 0.85, 0.0), preserve_base=preserve_smoke_base)
            v2.light = self.sample_smoke_light(p2, Vec3(0.0, -0.45, 0.0), preserve_base=preserve_smoke_base)
            v3.light = self.sample_smoke_light(p3, Vec3(0.0, -0.45, 0.0), preserve_base=preserve_smoke_base)

        var prev_noz = self.fb.no_depth_write
        var prev_add = self.fb.additive_blend
        var prev_soft = self.fb.vfx_soft_inv_dist
        var prev_erode = self.fb.vfx_erode_threshold
        var prev_bb = self.fb.vfx_blackbody_heat
        var prev_ref = self.fb.vfx_refract_strength
        self.fb.no_depth_write = True
        self.fb.additive_blend = additive
        self.fb.vfx_soft_inv_dist = (Float32(1.0) / soft_depth) if soft_depth > Float32(0.001) else Float32(0.0)
        self.fb.vfx_erode_threshold = erode
        self.fb.vfx_blackbody_heat = blackbody_heat
        self.fb.vfx_refract_strength = refract_strength
        var sz = self.fx.sizes[fx_kind]
        self.fb.draw_triangle(v0, v1, v2, self.fx.pixels[fx_kind], sz, sz, alpha=alpha, double_sided=True)
        self.fb.draw_triangle(v0, v2, v3, self.fx.pixels[fx_kind], sz, sz, alpha=alpha, double_sided=True)
        self.fb.no_depth_write = prev_noz
        self.fb.additive_blend = prev_add
        self.fb.vfx_soft_inv_dist = prev_soft
        self.fb.vfx_erode_threshold = prev_erode
        self.fb.vfx_blackbody_heat = prev_bb
        self.fb.vfx_refract_strength = prev_ref

    def draw_pyro_explosion(
        mut self,
        center: Vec3,
        max_radius: Float32,
        age_raw: Float32,
        near_fade: Float32 = 1.0,
        seed: Int = 0,
        tint: UInt32 = 0xFFFFFFFF,
    ):
        """Renders a multi-lobe Planckian Blackbody pyroclastic explosion with cellular UV alpha erosion,
        inverse-Z soft particles, Henyey-Greenstein lit smoke, and a refractive supersonic shockwave ring."""
        if max_radius <= 0.05 or near_fade <= 0.01:
            return
        var age = max(Float32(0.0), min(Float32(1.0), age_raw))
        var life = Float32(1.0) - age
        var grow = Float32(0.38) + Float32(0.62) * sqrt(age)
        var rad = max_radius * grow
        var soft_d = max(Float32(1.2), rad * Float32(0.55))
        var base_rot = Float32(seed % 17) * Float32(0.37) + age * Float32(0.85)

        # 1. Billowing carbonaceous soot & cooling ember canopy (non-additive, Planckian + lit smoke + alpha erosion)
        var soot_erode = Float32(0.06) + age * age * Float32(0.76)
        var soot_heat = max(Float32(0.04), life * life * Float32(0.68))
        var soot_alpha = UInt8(max(0, min(215, Int(life * min(Float32(1.0), Float32(0.28) + age * Float32(2.6)) * Float32(218.0) * near_fade))))
        if soot_alpha > 4:
            self.draw_billboard(
                center + Vec3(0.0, age * max_radius * Float32(0.22), 0.0),
                rad * (Float32(1.34) + age * Float32(0.38)),
                FX_FIREBALL,
                tint,
                soot_alpha,
                emissive=Float32(0.18),
                rotation=base_rot * Float32(0.55),
                additive=False,
                soft_depth=soft_d,
                erode=soot_erode,
                blackbody_heat=soot_heat,
                lit_smoke=True,
                no_fog=False,
            )

        # 2. Multi-lobe turbulent Planckian plasma core (6500K ivory -> 3200K amber -> 1600K crimson -> soot)
        var core_heat = max(Float32(0.12), life * Float32(1.04))
        var core_erode = Float32(0.04) + age * Float32(0.72)
        var core_alpha = UInt8(max(0, min(250, Int((life * life * Float32(210.0) + life * Float32(48.0)) * near_fade))))
        if core_alpha > 4:
            for lobe in range(3):
                var l_ang = Float32(lobe) * Float32(2.0943951) + Float32(seed % 11) * Float32(0.51)
                var lobe_s = Float32(1.0) if lobe > 0 else Float32(0.0)
                var l_off = (self.fb.right * cos(l_ang) + self.fb.up * sin(l_ang)) * (rad * Float32(0.22) * lobe_s)
                var l_rad = rad * (Float32(1.12) if lobe == 0 else Float32(0.84))
                var l_heat = core_heat if lobe == 0 else min(Float32(1.10), core_heat * Float32(1.12))
                self.draw_billboard(
                    center + l_off,
                    l_rad,
                    FX_FIREBALL,
                    tint,
                    core_alpha if lobe == 0 else UInt8(Int(core_alpha) * 4 // 5),
                    emissive=Float32(0.55) + life * Float32(1.45),
                    rotation=base_rot + Float32(lobe) * Float32(1.15),
                    additive=life > Float32(0.42),
                    soft_depth=soft_d,
                    erode=core_erode,
                    blackbody_heat=l_heat,
                    refract_strength=life * Float32(4.5) if lobe == 0 else Float32(0.0),
                )

        # 3. Young incandescent detonation flash + Panavision anamorphic streak + refractive supersonic shock ring
        if life > Float32(0.38):
            var flash_f = (life - Float32(0.38)) / Float32(0.62)
            var flash_a = UInt8(max(0, min(245, Int(flash_f * Float32(242.0) * near_fade))))
            self.draw_billboard(
                center,
                rad * Float32(0.46),
                FX_SOFT,
                0xFFFFFAEA,
                flash_a,
                emissive=Float32(3.15),
                additive=True,
                soft_depth=soft_d,
            )
        if age < Float32(0.62):
            var sh_f = (Float32(0.62) - age) / Float32(0.62)
            var shock_a = UInt8(max(0, min(142, Int(sh_f * sh_f * Float32(142.0) * near_fade))))
            self.draw_billboard(
                center,
                max_radius * (Float32(0.52) + age * Float32(1.58)),
                FX_SHOCK_RING,
                0xFFFFD2A4,
                shock_a,
                emissive=Float32(1.85),
                additive=True,
                soft_depth=soft_d,
                refract_strength=sh_f * Float32(7.5),
            )
        if life > Float32(0.48):
            var fl_a = UInt8(min(205, Int((life - Float32(0.48)) * Float32(365.0) * near_fade)))
            self.draw_anamorphic_flare(
                center,
                max_radius * Float32(2.25),
                max_radius * Float32(0.24),
                0xFFFFD698,
                fl_a,
                emissive=2.65,
            )

    def draw_anamorphic_flare(
        mut self,
        center: Vec3,
        half_width: Float32,
        half_height: Float32,
        color: UInt32,
        alpha: UInt8,
        emissive: Float32 = 2.6,
        with_ghosts: Bool = False,
        additive: Bool = True,
    ):
        """Renders a Panavision-style horizontal anamorphic lens flare streak (+ optional chromatic ghost discs)."""
        if half_width <= 0.001 or half_height <= 0.001 or alpha == 0:
            return
        var rel = center - self.fb.camera
        var depth = dot(rel, self.fb.forward)
        if depth <= Float32(0.3):
            return
        var rx = self.fb.right * half_width
        var ry = self.fb.up * half_height
        var n = Vec3(-self.fb.forward.x, -self.fb.forward.y, -self.fb.forward.z)
        var col = 0xFF000000 | (color & 0x00FFFFFF)

        var p0 = center - rx + ry
        var p1 = center + rx + ry
        var p2 = center + rx - ry
        var p3 = center - rx - ry

        var v0 = self.make_lit_vertex(p0, n, 0.0, 0.0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var v1 = self.make_lit_vertex(p1, n, 1.0, 0.0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var v2 = self.make_lit_vertex(p2, n, 1.0, 1.0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var v3 = self.make_lit_vertex(p3, n, 0.0, 1.0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)

        var prev_noz = self.fb.no_depth_write
        var prev_add = self.fb.additive_blend
        self.fb.no_depth_write = True
        self.fb.additive_blend = additive
        var sz = self.fx.sizes[FX_ANAMORPHIC]
        self.fb.draw_triangle(v0, v1, v2, self.fx.pixels[FX_ANAMORPHIC], sz, sz, alpha=alpha, double_sided=True)
        self.fb.draw_triangle(v0, v2, v3, self.fx.pixels[FX_ANAMORPHIC], sz, sz, alpha=alpha, double_sided=True)
        self.fb.no_depth_write = prev_noz
        self.fb.additive_blend = prev_add

        if with_ghosts:
            var axis_center = self.fb.camera + self.fb.forward * depth
            var offset = center - axis_center
            var g1 = axis_center - offset * Float32(0.42)
            var g2 = axis_center - offset * Float32(0.88)
            var g3 = axis_center + offset * Float32(0.28)
            var ga = UInt8(max(6, Int(alpha) // 6))
            self.draw_billboard(g1, half_height * Float32(1.45), FX_SOFT, 0xFF5CE0FF, ga, emissive=1.45, additive=True)
            self.draw_billboard(g2, half_height * Float32(2.25), FX_SOFT, 0xFFFF9E64, UInt8(max(5, Int(alpha) // 7)), emissive=1.35, additive=True)
            self.draw_billboard(g3, half_height * Float32(0.95), FX_SOFT, 0xFF88B4FF, ga, emissive=1.55, additive=True)

    @staticmethod
    def _lerp_sky_argb(c0: UInt32, c1: UInt32, t_raw: Float32) -> UInt32:
        var t = max(Float32(0.0), min(Float32(1.0), t_raw))
        var ts = t * t * (Float32(3.0) - Float32(2.0) * t)
        var r0 = Float32((c0 >> 16) & 255)
        var g0 = Float32((c0 >> 8) & 255)
        var b0 = Float32(c0 & 255)
        var r1 = Float32((c1 >> 16) & 255)
        var g1 = Float32((c1 >> 8) & 255)
        var b1 = Float32(c1 & 255)
        var r = UInt32(max(0, min(255, Int(r0 + (r1 - r0) * ts + 0.5))))
        var g = UInt32(max(0, min(255, Int(g0 + (g1 - g0) * ts + 0.5))))
        var b = UInt32(max(0, min(255, Int(b0 + (b1 - b0) * ts + 0.5))))
        return 0xFF000000 | (r << 16) | (g << 8) | b

    @staticmethod
    def _sky_color_at_theta(
        theta: Float32,
        zenith_color: UInt32,
        mid_color: UInt32,
        horizon_color: UInt32,
        nadir_color: UInt32,
    ) -> UInt32:
        if theta <= Float32(0.96):
            return Self._lerp_sky_argb(zenith_color, mid_color, theta / Float32(0.96))
        if theta <= Float32(1.52):
            return Self._lerp_sky_argb(mid_color, horizon_color, (theta - Float32(0.96)) / Float32(0.56))
        if theta <= Float32(1.66):
            return 0xFF000000 | (horizon_color & 0x00FFFFFF)
        return Self._lerp_sky_argb(horizon_color, nadir_color, (theta - Float32(1.66)) / Float32(0.95))

    @staticmethod
    def _sky_vec_at_theta(
        theta: Float32,
        zenith_color: UInt32,
        mid_color: UInt32,
        horizon_color: UInt32,
        nadir_color: UInt32,
    ) -> Vec3:
        var c = Self._sky_color_at_theta(theta, zenith_color, mid_color, horizon_color, nadir_color)
        return Vec3(
            Float32((c >> 16) & 255) / Float32(255.0),
            Float32((c >> 8) & 255) / Float32(255.0),
            Float32(c & 255) / Float32(255.0),
        )

    def draw_sky_dome_gradient(
        mut self,
        zenith_color: UInt32,
        mid_color: UInt32,
        horizon_color: UInt32,
        nadir_color: UInt32,
        radius: Float32 = 4600.0,
    ):
        """Renders a Gouraud-interpolated 12-band Rayleigh/Mie atmospheric sky dome centered on the camera."""
        var cam = self.fb.camera
        var segs = 28
        var bands = 12
        var thetas = [
            Float32(0.01),
            Float32(0.22),
            Float32(0.44),
            Float32(0.66),
            Float32(0.86),
            Float32(1.04),
            Float32(1.20),
            Float32(1.36),
            Float32(1.51),
            Float32(1.66),
            Float32(1.92),
            Float32(2.28),
            Float32(2.72),
        ]
        var tau = Float32(6.2831853)
        for b in range(bands):
            var t0 = thetas[b]
            var t1 = thetas[b + 1]
            var st0 = sin(t0)
            var ct0 = cos(t0)
            var st1 = sin(t1)
            var ct1 = cos(t1)
            var l_top = Self._sky_vec_at_theta(t0, zenith_color, mid_color, horizon_color, nadir_color)
            var l_bot = Self._sky_vec_at_theta(t1, zenith_color, mid_color, horizon_color, nadir_color)
            for s in range(segs):
                var p0 = Float32(s) / Float32(segs) * tau
                var p1 = Float32(s + 1) / Float32(segs) * tau
                var u00 = Vec3(st0 * cos(p0), ct0, st0 * sin(p0))
                var u01 = Vec3(st0 * cos(p1), ct0, st0 * sin(p1))
                var u11 = Vec3(st1 * cos(p1), ct1, st1 * sin(p1))
                var u10 = Vec3(st1 * cos(p0), ct1, st1 * sin(p0))
                var v00 = self.make_lit_vertex(cam + u00 * radius, u00 * -1.0, 0.0, 0.0, 0xFFFFFFFF, 1.0, 1.0, 0.0, 0.0, unlit=True, no_fog=True)
                var v01 = self.make_lit_vertex(cam + u01 * radius, u01 * -1.0, 1.0, 0.0, 0xFFFFFFFF, 1.0, 1.0, 0.0, 0.0, unlit=True, no_fog=True)
                var v11 = self.make_lit_vertex(cam + u11 * radius, u11 * -1.0, 1.0, 1.0, 0xFFFFFFFF, 1.0, 1.0, 0.0, 0.0, unlit=True, no_fog=True)
                var v10 = self.make_lit_vertex(cam + u10 * radius, u10 * -1.0, 0.0, 1.0, 0xFFFFFFFF, 1.0, 1.0, 0.0, 0.0, unlit=True, no_fog=True)
                v00.light = l_top
                v01.light = l_top
                v11.light = l_bot
                v10.light = l_bot
                self.fb.draw_triangle(v00, v01, v11, self.empty_tex, 0, 0, double_sided=True)
                self.fb.draw_triangle(v00, v11, v10, self.empty_tex, 0, 0, double_sided=True)

    def draw_cinema_vignette(mut self, alpha: UInt8 = 96):
        """Subtle optical camera lens edge darkening across the viewport."""
        if alpha == 0:
            return
        var dist = Float32(0.45)
        var half_h = dist * (Float32(self.fb.height) * 0.5 / max(Float32(1.0), self.fb.focal)) * Float32(1.04)
        var half_w = half_h * (Float32(self.fb.width) / Float32(max(1, self.fb.height)))
        var center = self.fb.camera + self.fb.forward * dist
        var rx = self.fb.right * half_w
        var ry = self.fb.up * half_h
        var n = self.fb.forward * -1.0
        var col = UInt32(0xFF03050A)
        var v0 = self.make_lit_vertex(center - rx + ry, n, 0.0, 0.0, col, 1.0, 1.0, 0.0, 0.0, unlit=True, no_fog=True)
        var v1 = self.make_lit_vertex(center + rx + ry, n, 1.0, 0.0, col, 1.0, 1.0, 0.0, 0.0, unlit=True, no_fog=True)
        var v2 = self.make_lit_vertex(center + rx - ry, n, 1.0, 1.0, col, 1.0, 1.0, 0.0, 0.0, unlit=True, no_fog=True)
        var v3 = self.make_lit_vertex(center - rx - ry, n, 0.0, 1.0, col, 1.0, 1.0, 0.0, 0.0, unlit=True, no_fog=True)
        var prev_noz = self.fb.no_depth_write
        self.fb.no_depth_write = True
        var sz = self.fx.sizes[FX_VIGNETTE]
        self.fb.draw_triangle(v0, v1, v2, self.fx.pixels[FX_VIGNETTE], sz, sz, alpha=alpha, double_sided=True)
        self.fb.draw_triangle(v0, v2, v3, self.fx.pixels[FX_VIGNETTE], sz, sz, alpha=alpha, double_sided=True)
        self.fb.no_depth_write = prev_noz

    def draw_beam_3d(
        mut self,
        start: Vec3,
        end: Vec3,
        half_width: Float32,
        fx_kind: Int,
        color: UInt32,
        alpha: UInt8,
        emissive: Float32 = 1.65,
        u0: Float32 = 0.0,
        u1: Float32 = 1.0,
        v0: Float32 = 0.0,
        v1: Float32 = 1.0,
        additive: Bool = False,
        soft_depth: Float32 = 0.0,
        erode: Float32 = 0.0,
        refract_strength: Float32 = 0.0,
    ):
        if half_width <= 0.001 or alpha == 0 or fx_kind < 0 or fx_kind >= len(self.fx.pixels):
            return
        var axis = end - start
        if axis.length_squared() < 0.0001:
            return
        var dir_v = axis.normalized()
        var mid = (start + end) * 0.5
        var to_cam = (self.fb.camera - mid).normalized()
        var side = cross(dir_v, to_cam)
        if side.length_squared() < 0.0001:
            side = self.fb.up
        else:
            side = side.normalized()
        var offset = side * half_width
        var col = 0xFF000000 | (color & 0x00FFFFFF)

        var p0 = start + offset
        var p1 = end + offset
        var p2 = end - offset
        var p3 = start - offset

        var rv0 = self.make_lit_vertex(p0, to_cam, u0, v0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var rv1 = self.make_lit_vertex(p1, to_cam, u1, v0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var rv2 = self.make_lit_vertex(p2, to_cam, u1, v1, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var rv3 = self.make_lit_vertex(p3, to_cam, u0, v1, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)

        var prev_noz = self.fb.no_depth_write
        var prev_add = self.fb.additive_blend
        var prev_soft = self.fb.vfx_soft_inv_dist
        var prev_erode = self.fb.vfx_erode_threshold
        var prev_ref = self.fb.vfx_refract_strength
        self.fb.no_depth_write = True
        self.fb.additive_blend = additive
        self.fb.vfx_soft_inv_dist = (Float32(1.0) / soft_depth) if soft_depth > Float32(0.001) else Float32(0.0)
        self.fb.vfx_erode_threshold = erode
        self.fb.vfx_refract_strength = refract_strength
        var sz = self.fx.sizes[fx_kind]
        self.fb.draw_triangle(rv0, rv1, rv2, self.fx.pixels[fx_kind], sz, sz, alpha=alpha, double_sided=True)
        self.fb.draw_triangle(rv0, rv2, rv3, self.fx.pixels[fx_kind], sz, sz, alpha=alpha, double_sided=True)
        self.fb.no_depth_write = prev_noz
        self.fb.additive_blend = prev_add
        self.fb.vfx_soft_inv_dist = prev_soft
        self.fb.vfx_erode_threshold = prev_erode
        self.fb.vfx_refract_strength = prev_ref

    def draw_thruster_plume(
        mut self,
        nozzle: Vec3,
        exhaust_dir: Vec3,
        length: Float32,
        radius: Float32,
        core_color: UInt32,
        outer_color: UInt32,
        alpha: UInt8 = 235,
        emissive: Float32 = 2.15,
        additive: Bool = True,
    ):
        """Volumetric afterburner plume with TEX_FLAME outer cone, hot core, heat shimmer, and Mach shock diamonds."""
        if length <= 0.02 or radius <= 0.01 or alpha == 0:
            return
        var dir_n = exhaust_dir.normalized()
        var tip = nozzle + dir_n * length
        var mid = (nozzle + tip) * 0.5
        var to_cam = (self.fb.camera - mid).normalized()
        var side = cross(dir_n, to_cam)
        if side.length_squared() < 0.0001:
            side = self.fb.right
        else:
            side = side.normalized()
        var up_perp = cross(side, dir_n)
        if up_perp.length_squared() < 0.0001:
            up_perp = self.fb.up
        else:
            up_perp = up_perp.normalized()

        var soft_d = max(Float32(0.32), radius * Float32(1.35))
        var prev_noz = self.fb.no_depth_write
        var prev_add = self.fb.additive_blend
        var prev_soft = self.fb.vfx_soft_inv_dist
        var prev_ref = self.fb.vfx_refract_strength
        self.fb.no_depth_write = True
        self.fb.additive_blend = additive
        self.fb.vfx_soft_inv_dist = Float32(1.0) / soft_d

        var sz = self.fx.sizes[FX_FLAME]
        var outer_col = 0xFF000000 | (outer_color & 0x00FFFFFF)
        var core_col = 0xFF000000 | (core_color & 0x00FFFFFF)

        # Crossed dual-plane outer flame envelope + subtle exhaust heat-haze refraction
        self.fb.vfx_refract_strength = Float32(2.4) if radius > Float32(0.22) else Float32(0.0)
        var outer_alpha = UInt8(max(24, Int(alpha) * 58 // 100))
        for plane_i in range(2):
            var ax = side if plane_i == 0 else up_perp
            var o_off = ax * (radius * 1.44)
            var oa = self.make_lit_vertex(tip - o_off, to_cam, 0.0, 0.0, outer_col, 1.0, 1.0, 0.0, emissive * 0.78, unlit=True, no_fog=True)
            var ob = self.make_lit_vertex(tip + o_off, to_cam, 1.0, 0.0, outer_col, 1.0, 1.0, 0.0, emissive * 0.78, unlit=True, no_fog=True)
            var oc = self.make_lit_vertex(nozzle + o_off, to_cam, 1.0, 0.90, outer_col, 1.0, 1.0, 0.0, emissive * 0.92, unlit=True, no_fog=True)
            var od = self.make_lit_vertex(nozzle - o_off, to_cam, 0.0, 0.90, outer_col, 1.0, 1.0, 0.0, emissive * 0.92, unlit=True, no_fog=True)
            self.fb.draw_triangle(oa, ob, oc, self.fx.pixels[FX_FLAME], sz, sz, alpha=outer_alpha, double_sided=True)
            self.fb.draw_triangle(oa, oc, od, self.fx.pixels[FX_FLAME], sz, sz, alpha=outer_alpha, double_sided=True)
            self.fb.vfx_refract_strength = Float32(0.0)
        self.fb.vfx_refract_strength = prev_ref

        # Tapered white-hot inner plasma core (slender so Mach shock diamonds stand out crisply!)
        var c_tip = nozzle + dir_n * (length * 0.68)
        var c_off = side * (radius * 0.56)
        var core_alpha = UInt8(max(36, Int(alpha) * 78 // 100))
        var ca = self.make_lit_vertex(c_tip - c_off, to_cam, 0.0, 0.0, core_col, 1.0, 1.0, 0.0, emissive * 1.18, unlit=True, no_fog=True)
        var cb = self.make_lit_vertex(c_tip + c_off, to_cam, 1.0, 0.0, core_col, 1.0, 1.0, 0.0, emissive * 1.18, unlit=True, no_fog=True)
        var cc = self.make_lit_vertex(nozzle + c_off, to_cam, 1.0, 0.90, core_col, 1.0, 1.0, 0.0, emissive * 1.32, unlit=True, no_fog=True)
        var cd = self.make_lit_vertex(nozzle - c_off, to_cam, 0.0, 0.90, core_col, 1.0, 1.0, 0.0, emissive * 1.32, unlit=True, no_fog=True)
        self.fb.draw_triangle(ca, cb, cc, self.fx.pixels[FX_FLAME], sz, sz, alpha=core_alpha, double_sided=True)
        self.fb.draw_triangle(ca, cc, cd, self.fx.pixels[FX_FLAME], sz, sz, alpha=core_alpha, double_sided=True)

        self.fb.no_depth_write = prev_noz
        self.fb.additive_blend = prev_add
        self.fb.vfx_soft_inv_dist = prev_soft

        # Nozzle exit throat glow + 5 crisp supersonic Mach shock-diamond compression rings & nodes
        self.draw_billboard(
            nozzle + dir_n * (radius * 0.18),
            radius * 0.92,
            FX_SOFT,
            0xFFFFF4CC,
            UInt8(min(248, Int(alpha))),
            emissive * 1.40,
            additive=additive,
            soft_depth=soft_d,
        )
        for d in range(5):
            var frac = Float32(0.09) + Float32(d) * 0.145
            var d_pos = nozzle + dir_n * (length * frac)
            var d_rad = radius * (0.86 - Float32(d) * 0.125)
            var d_alpha = UInt8(max(32, Int(alpha) - d * 38))
            self.draw_billboard(
                d_pos,
                d_rad * 1.12,
                FX_SHOCK_RING,
                outer_color,
                UInt8(max(24, Int(d_alpha) * 4 // 5)),
                emissive * 1.28,
                additive=additive,
                soft_depth=soft_d,
            )
            self.draw_billboard(
                d_pos,
                d_rad * 0.62,
                FX_SOFT,
                core_color,
                d_alpha,
                emissive * 1.48,
                additive=additive,
                soft_depth=soft_d,
            )

    def draw_ribbon_3d(
        mut self,
        points: List[Vec3],
        head_width: Float32,
        tail_width: Float32,
        head_color: UInt32,
        tail_color: UInt32,
        head_alpha: UInt8,
        tail_alpha: UInt8 = 12,
        emissive: Float32 = 0.45,
        near_fade_dist: Float32 = 3.5,
        soft_depth: Float32 = 1.1,
        uv_scroll: Float32 = 0.0,
        uv_repeat: Float32 = 1.0,
        tail_erode: Float32 = 0.0,
        lit_smoke: Bool = False,
        additive: Bool = False,
        fx_kind: Int = FX_RIBBON,
    ):
        """Camera-facing volumetric strip from points[0] (newest/head) to points[n-1] (tail)
        with inverse-Z soft particles, optional Henyey-Greenstein smoke lighting, and progressive tail alpha erosion."""
        var n = len(points)
        if n < 2 or head_alpha == 0 or fx_kind < 0 or fx_kind >= len(self.fx.pixels):
            return
        var sz = self.fx.sizes[fx_kind]
        var hr = Float32((head_color >> 16) & 255)
        var hg = Float32((head_color >> 8) & 255)
        var hb = Float32(head_color & 255)
        var tr = Float32((tail_color >> 16) & 255)
        var tg = Float32((tail_color >> 8) & 255)
        var tb = Float32(tail_color & 255)

        var prev_noz = self.fb.no_depth_write
        var prev_add = self.fb.additive_blend
        var prev_soft = self.fb.vfx_soft_inv_dist
        var prev_erode = self.fb.vfx_erode_threshold
        self.fb.no_depth_write = True
        self.fb.additive_blend = additive
        self.fb.vfx_soft_inv_dist = (Float32(1.0) / soft_depth) if soft_depth > Float32(0.001) else Float32(0.0)
        for i in range(n - 1):
            var p0 = points[i]
            var p1 = points[i + 1]
            var seg = p1 - p0
            if seg.length_squared() < 0.0001:
                continue
            var d_cam0 = (p0 - self.fb.camera).length()
            var d_cam1 = (p1 - self.fb.camera).length()
            var min_d = min(d_cam0, d_cam1)
            var near_fade = Float32(1.0)
            if near_fade_dist > 0.01 and min_d < near_fade_dist:
                near_fade = max(Float32(0.0), (min_d - 0.65) / max(Float32(0.25), near_fade_dist - 0.65))
            if near_fade <= 0.02:
                continue
            var t0 = Float32(i) / Float32(max(1, n - 1))
            var t1 = Float32(i + 1) / Float32(max(1, n - 1))
            var w0 = head_width + (tail_width - head_width) * t0
            var w1 = head_width + (tail_width - head_width) * t1

            var dir0 = seg.normalized()
            var dir1 = (points[i + 2] - p1).normalized() if i + 2 < n and (points[i + 2] - p1).length_squared() > 0.0001 else dir0
            var cam0 = (self.fb.camera - p0).normalized()
            var cam1 = (self.fb.camera - p1).normalized()
            var side0 = cross(dir0, cam0)
            var side1 = cross(dir1, cam1)
            side0 = side0.normalized() if side0.length_squared() > 0.0001 else self.fb.up
            side1 = side1.normalized() if side1.length_squared() > 0.0001 else self.fb.up

            var c0 = 0xFF000000 | (UInt32(Int(hr + (tr - hr) * t0)) << 16) | (UInt32(Int(hg + (tg - hg) * t0)) << 8) | UInt32(Int(hb + (tb - hb) * t0))
            var c1 = 0xFF000000 | (UInt32(Int(hr + (tr - hr) * t1)) << 16) | (UInt32(Int(hg + (tg - hg) * t1)) << 8) | UInt32(Int(hb + (tb - hb) * t1))
            var em0 = emissive * (1.0 - t0 * 0.72)
            var em1 = emissive * (1.0 - t1 * 0.72)
            var t_mid = (t0 + t1) * Float32(0.5)
            var raw_a = (Float32(head_alpha) + Float32(Int(tail_alpha) - Int(head_alpha)) * t_mid) * near_fade
            var a_seg = UInt8(max(0, min(255, Int(raw_a))))
            if a_seg == 0:
                continue

            self.fb.vfx_erode_threshold = tail_erode * t_mid * t_mid if tail_erode > Float32(0.0) else Float32(0.0)
            var u0 = t0 * uv_repeat + uv_scroll
            var u1 = t1 * uv_repeat + uv_scroll
            var v0 = self.make_lit_vertex(p0 + side0 * w0, cam0, u0, 0.0, c0, 1.0, 1.0, 0.0, em0, unlit=True)
            var v1 = self.make_lit_vertex(p1 + side1 * w1, cam1, u1, 0.0, c1, 1.0, 1.0, 0.0, em1, unlit=True)
            var v2 = self.make_lit_vertex(p1 - side1 * w1, cam1, u1, 1.0, c1, 1.0, 1.0, 0.0, em1, unlit=True)
            var v3 = self.make_lit_vertex(p0 - side0 * w0, cam0, u0, 1.0, c0, 1.0, 1.0, 0.0, em0, unlit=True)
            if lit_smoke:
                var l0 = self.sample_smoke_light(p0, Vec3(0.0, 0.7, 0.0), preserve_base=True)
                var l1 = self.sample_smoke_light(p1, Vec3(0.0, 0.7, 0.0), preserve_base=True)
                v0.light = l0
                v3.light = l0
                v1.light = l1
                v2.light = l1
            self.fb.draw_triangle(v0, v1, v2, self.fx.pixels[fx_kind], sz, sz, alpha=a_seg, double_sided=True)
            self.fb.draw_triangle(v0, v2, v3, self.fx.pixels[fx_kind], sz, sz, alpha=a_seg, double_sided=True)
        self.fb.no_depth_write = prev_noz
        self.fb.additive_blend = prev_add
        self.fb.vfx_soft_inv_dist = prev_soft
        self.fb.vfx_erode_threshold = prev_erode

    def draw_colossal_beam_3d(
        mut self,
        start: Vec3,
        end: Vec3,
        radius: Float32,
        core_color: UInt32,
        sheath_color: UInt32,
        scroll: Float32 = 0.0,
        alpha: UInt8 = 245,
        emissive: Float32 = 3.2,
    ):
        """Capital-ship mega-particle cannon beam: counter-scrolling cellular Voronoi plasma sheaths,
        blazing core, twin helical corkscrew energy ribbons, high-voltage lightning arcs, and refractive shockwaves."""
        if radius <= 0.01 or alpha == 0:
            return
        var axis = end - start
        var beam_len = axis.length()
        if beam_len < 0.05:
            return
        var dir_v = axis * (1.0 / beam_len)
        var repeats = max(Float32(2.0), beam_len / max(Float32(1.0), radius * 5.2))
        var u0 = -scroll
        var u1 = repeats - scroll
        var soft_d = max(Float32(0.8), radius * Float32(0.9))

        # 1. Outer counter-scrolling cellular Voronoi plasma sheath
        self.draw_beam_3d(
            start,
            end,
            radius * 2.25,
            FX_VORONOI_BEAM,
            sheath_color,
            UInt8(Int(alpha) * 5 // 8),
            emissive=emissive * 0.82,
            u0=u0,
            u1=u1,
            additive=True,
            soft_depth=soft_d,
            refract_strength=Float32(3.2),
        )
        # 2. Inner faster Voronoi particle stream
        self.draw_beam_3d(
            start,
            end,
            radius * 1.24,
            FX_VORONOI_BEAM,
            sheath_color,
            UInt8(Int(alpha) * 7 // 8),
            emissive=emissive * 1.04,
            u0=scroll * Float32(0.65),
            u1=repeats * Float32(1.35) + scroll * Float32(0.65),
            additive=True,
            soft_depth=soft_d,
        )
        # 3. White-hot inner core
        self.draw_beam_3d(
            start,
            end,
            radius * 0.56,
            FX_BOLT,
            core_color,
            alpha,
            emissive=emissive * 1.38,
            additive=True,
            soft_depth=soft_d * Float32(0.5),
        )

        # 4. Twin helical corkscrew plasma ribbons spiraling around the beam axis
        var up_ref = Vec3(0.0, 1.0, 0.0) if abs(dir_v.y) < Float32(0.92) else Vec3(1.0, 0.0, 0.0)
        var b_right = cross(dir_v, up_ref).normalized()
        var b_up = cross(b_right, dir_v).normalized()
        var helix_segs = 14
        var turns = max(Float32(2.5), min(Float32(6.0), beam_len / max(Float32(8.0), radius * Float32(18.0))))
        var helix_pts = List[Vec3](capacity=helix_segs + 1)
        for h_idx in range(2):
            helix_pts.clear()
            var phase0 = Float32(h_idx) * Float32(3.14159265) - scroll * Float32(2.4)
            for s_i in range(helix_segs + 1):
                var ft = Float32(s_i) / Float32(helix_segs)
                var env_r = radius * (Float32(0.95) + Float32(0.45) * sin(ft * Float32(3.14159265)))
                var ang = phase0 + ft * turns * Float32(6.2831853)
                var radial = b_right * (cos(ang) * env_r) + b_up * (sin(ang) * env_r)
                helix_pts.append(start + dir_v * (beam_len * ft) + radial)
            self.draw_ribbon_3d(
                helix_pts,
                head_width=radius * Float32(0.28),
                tail_width=radius * Float32(0.22),
                head_color=core_color,
                tail_color=sheath_color,
                head_alpha=UInt8(Int(alpha) * 3 // 4),
                tail_alpha=UInt8(Int(alpha) * 2 // 3),
                emissive=emissive * Float32(1.15),
                near_fade_dist=1.5,
                soft_depth=soft_d,
                additive=True,
                fx_kind=FX_BOLT,
            )

        # 5. Stochastic high-voltage FX_LIGHTNING arcs at muzzle & terminal impact + refractive shock rings
        var arc_ang = scroll * Float32(5.7)
        var muzzle_arc_end = start + dir_v * min(beam_len * Float32(0.18), radius * Float32(5.5)) + (b_right * cos(arc_ang) + b_up * sin(arc_ang)) * (radius * Float32(2.2))
        var impact_arc_start = end - dir_v * min(beam_len * Float32(0.18), radius * Float32(5.5)) + (b_right * sin(arc_ang * Float32(1.3)) - b_up * cos(arc_ang * Float32(1.3))) * (radius * Float32(2.4))
        self.draw_beam_3d(start, muzzle_arc_end, radius * Float32(0.65), FX_LIGHTNING, core_color, UInt8(Int(alpha) * 4 // 5), emissive=emissive * Float32(1.25), additive=True, soft_depth=soft_d)
        self.draw_beam_3d(impact_arc_start, end, radius * Float32(0.72), FX_LIGHTNING, core_color, UInt8(Int(alpha) * 4 // 5), emissive=emissive * Float32(1.25), additive=True, soft_depth=soft_d)

        self.draw_billboard(start, radius * 3.4, FX_SOFT, sheath_color, UInt8(Int(alpha) * 3 // 4), emissive=emissive, additive=True, soft_depth=soft_d)
        self.draw_billboard(start, radius * 1.9, FX_SOFT, core_color, alpha, emissive=emissive * 1.4, additive=True, soft_depth=soft_d)
        self.draw_billboard(start, radius * 2.8, FX_SHOCK_RING, core_color, UInt8(Int(alpha) // 2), emissive=emissive, additive=True, soft_depth=soft_d, refract_strength=Float32(5.0))
        self.draw_billboard(end, radius * 3.8, FX_SOFT, sheath_color, UInt8(Int(alpha) * 3 // 4), emissive=emissive, additive=True, soft_depth=soft_d)
        self.draw_billboard(end, radius * 2.1, FX_SOFT, core_color, alpha, emissive=emissive * 1.4, additive=True, soft_depth=soft_d)
        self.draw_billboard(end, radius * 3.4, FX_SHOCK_RING, sheath_color, UInt8(Int(alpha) * 3 // 5), emissive=emissive * Float32(1.1), additive=True, soft_depth=soft_d, refract_strength=Float32(6.5))
        for k in range(3):
            var frac = Float32(((Float32(k) * 0.33 + scroll * 0.18) - Float32(Int(Float32(k) * 0.33 + scroll * 0.18))))
            var node_p = start + dir_v * (beam_len * max(Float32(0.08), min(Float32(0.92), frac)))
            self.draw_billboard(node_p, radius * 1.85, FX_SOFT, core_color, UInt8(Int(alpha) * 2 // 3), emissive=emissive * 1.15, additive=True, soft_depth=soft_d)

    def draw_cloud_deck(
        mut self,
        center: Vec3,
        altitude: Float32,
        span: Float32,
        count: Int,
        sunlit_color: UInt32 = 0xFFF8FAFF,
        shadow_color: UInt32 = 0xFF6B85AC,
        alpha: UInt8 = 185,
    ):
        """Volumetric FBM cumulus cloud deck sorted back-to-front around `center` for atmospheric flight."""
        if count <= 0 or span <= 1.0 or alpha == 0:
            return
        var half = span * 0.5
        var cell = max(Float32(20.0), span)
        var snap_x = Float32(Int(center.x / 64.0)) * 64.0
        var snap_z = Float32(Int(center.z / 64.0)) * 64.0
        var c_positions = List[Vec3](capacity=count)
        var c_radii = List[Float32](capacity=count)
        var c_alphas = List[UInt8](capacity=count)
        var c_rots = List[Float32](capacity=count)
        var c_dists = List[Float32](capacity=count)

        for i in range(count):
            var h1 = UInt32(i * 747796405 + 2891336453)
            var h2 = UInt32(i * 1597334677 + 3812015801)
            var h3 = UInt32(i * 2246822519 + 3266489917)
            var ox = (Float32(h1 & 0xFFFF) / 65535.0) * cell - half
            var oz = (Float32(h2 & 0xFFFF) / 65535.0) * cell - half
            var oy = ((Float32(h3 & 0xFF) / 255.0) - 0.5) * 16.0
            var wx = snap_x + ox
            var wz = snap_z + oz
            while wx - center.x > half:
                wx -= cell
            while wx - center.x < -half:
                wx += cell
            while wz - center.z > half:
                wz -= cell
            while wz - center.z < -half:
                wz += cell
            var base_r = Float32(15.0) + Float32((h3 >> 8) & 0xFF) / 255.0 * Float32(19.0)
            var c_pos = Vec3(wx, altitude + oy, wz)
            var d2 = (c_pos - self.fb.camera).length_squared()
            if d2 < 225.0:
                continue
            var fade = min(Float32(1.0), max(Float32(0.0), (half - max(abs(wx - center.x), abs(wz - center.z))) / (half * 0.35)))
            var near_f = min(Float32(1.0), max(Float32(0.0), (sqrt(d2) - 15.0) / 18.0))
            var eff_a = UInt8(max(0, min(255, Int(Float32(alpha) * fade * near_f))))
            if eff_a < 8:
                continue
            c_positions.append(c_pos)
            c_radii.append(base_r)
            c_alphas.append(eff_a)
            c_rots.append(((Float32((h3 >> 16) & 0xFF) / 255.0) - 0.5) * 0.32)
            c_dists.append(d2)

        # Sort back-to-front (largest distance first) for clean volumetric cloud layering
        var n_valid = len(c_positions)
        var order = List[Int](capacity=n_valid)
        for k in range(n_valid):
            order.append(k)
        for a in range(1, n_valid):
            var key_idx = order[a]
            var key_d = c_dists[key_idx]
            var b = a - 1
            while b >= 0 and c_dists[order[b]] < key_d:
                order[b + 1] = order[b]
                b -= 1
            order[b + 1] = key_idx

        for s_idx in range(n_valid):
            var idx = order[s_idx]
            var c_pos = c_positions[idx]
            var base_r = c_radii[idx]
            var eff_a = c_alphas[idx]
            var rot = c_rots[idx]
            var sh_a = UInt8(Int(eff_a) * 4 // 5)
            var cloud_soft = max(Float32(18.0), base_r * Float32(0.85))
            # Anvil shadow base + billowing FBM cumulus crown turrets with soft terrain/hull blending & HG silver-lining
            self.draw_billboard(c_pos + Vec3(0.0, -base_r * 0.24, 0.0), base_r * 1.24, FX_CUMULUS, shadow_color, sh_a, emissive=0.04, rotation=rot * 0.5, soft_depth=cloud_soft, lit_smoke=True, preserve_smoke_base=True)
            self.draw_billboard(c_pos + Vec3(-base_r * 0.32, base_r * 0.14, 0.0), base_r * 0.98, FX_CUMULUS, sunlit_color, eff_a, emissive=0.18, rotation=rot, soft_depth=cloud_soft, lit_smoke=True, preserve_smoke_base=True)
            self.draw_billboard(c_pos + Vec3(base_r * 0.34, base_r * 0.20, base_r * 0.14), base_r * 0.90, FX_CUMULUS, sunlit_color, eff_a, emissive=0.21, rotation=-rot * 0.8, soft_depth=cloud_soft, lit_smoke=True, preserve_smoke_base=True)

    def end_scene(mut self):
        """Complete 3D scene pass so 2D HUD widgets drawn afterward are not blurred by CPU bloom."""
        if self.scene_ended:
            return
        self.scene_ended = True
        if not self.fb.gpu.active:
            self.fb.apply_post(self.active_profile, self.active_quality, self.active_reduced)

    def present_headless(mut self) -> Bool:
        """Submit GPU passes (raster + bloom + smooth + HUD compose) and read back finished pixels."""
        if not self.scene_ended:
            self.end_scene()
        if self.fb.gpu.active:
            if self.fb.submit_gpu_frame():
                return self.fb.read_back_gpu()
            return False
        return True
