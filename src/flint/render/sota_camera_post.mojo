"""Area D (sota:camera): physically based camera post chain.

Passes (all compute, `ShaderDsl` -> SPIR-V, bindings of the AAA descriptor set):

Depth of field (replaces `PolygonalBokehDoF` when `SotaCameraPost.enable_dof`), at half resolution:
  1. `sota_dof_setup_shader`     full -> half res colour + signed thin-lens CoC (half-res px)
  2. `sota_dof_tile_shader`      8x8 half-px tiles: max near CoC / max far CoC
  3. `sota_dof_gather_shader`    scatter-as-gather: 48-tap ring kernel shaped by the aperture polygon
                                 (blade count, rotation, curvature) with cat's-eye clipping; separate
                                 far field (normalised, CoC clamped to the centre so background never
                                 smears over a sharper foreground) and near field (premultiplied colour +
                                 coverage alpha, kernel radius dilated from the 5x5 tile neighbourhood so
                                 a blurry foreground spreads over in-focus pixels behind it).
                                 Each tap is weighted by `1 / r^2` of its own CoC (energy of a disc of
                                 radius r spreads over pi r^2 pixels), so highlights form discs of the
                                 right brightness. (Jimenez, "Next-Generation Post Processing in Call of
                                 Duty: Advanced Warfare", SIGGRAPH 2014; Abadie, "A Life of a Bokeh",
                                 SIGGRAPH 2018.)
  4. `sota_dof_composite_shader` full res: sharp -> far by own CoC, then near field "over".

Lens and sensor (one pass at full res right before the display tonemap, plus a quarter-res flare chain):
  5. `sota_flare_bright_shader`  quarter-res exposure-aware bright pass
  6. `sota_flare_ghost_shader`   ghosts mirrored through the optical centre at fixed magnifications with
                                 per-channel dispersion, a thin-film (MgF2 quarter-wave) coating tint per
                                 ghost, and a halo (Chapman, "Pseudo Lens Flare", 2013; Hullin et al.,
                                 "Physically-Based Real-Time Lens Flare Rendering", SIGGRAPH 2011 for the
                                 coating model).
  7. `sota_flare_blur_shader`    aperture-polygon blur of the ghosts (ghosts are defocused images of the
                                 aperture, so they take its shape).
  7b. `sota_streak_shader`       optional anamorphic horizontal streak (exponential line kernel).
  8. `sota_lens_composite_shader` optical (f-stop dependent) vignetting, ghosts, anamorphic streak,
                                 lens dirt (procedural texture from `sota_camera_dirt.mojo`) lit by a wide
                                 average of the bright pass, then sensor noise: Poisson shot noise +
                                 read noise from ISO and the full-well capacity, in exposure-normalised
                                 signal units (applied last: it happens on the sensor).
  9. `sota_ca_shader`            lateral chromatic aberration (per-channel magnification), recorded right
                                 after DoF: the lens images each wavelength at its own magnification
                                 before any later scattering (bloom) or temporal smear (motion blur).

Compute VRS for the shading pass lives in `sota_camera_vrs.mojo`.

Physical order in the frame: TSR -> DoF -> lateral CA -> (NPR fx) -> bloom -> motion blur -> auto exposure
meter -> lens (vignette, ghosts, streak, dirt) -> sensor noise -> display (exposure, cos^4, tone curve).
"""
from std.collections import List
from std.math import atan2, cos, exp, floor, max, min, sin, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.camera.physical import PhysicalCameraModel, PHYS_EXPOSURE_MANUAL
from flint.platform.amdgpu import GpuBuffer
from flint.platform.vulkan import (
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    ACCESS_TRANSFER_READ,
    ACCESS_TRANSFER_WRITE,
    PIPELINE_BOTTOM,
    PIPELINE_COMPUTE,
    PIPELINE_TRANSFER,
    VulkanGpu,
)
from flint.render.spirv import MODEL_GL_COMPUTE
from flint.render.shader_dsl import (
    BufferRef,
    GpuF32,
    GpuU32,
    GpuVarF32,
    GpuVarVec3,
    GpuVec3,
    ShaderDsl,
    abs_f,
    clamp_f,
    exp2_f,
    cos_f,
    dot,
    floor_f,
    fract_f,
    max_f,
    min_f,
    min_u,
    select_f,
    select_u,
    select_vec3,
    sin_f,
    sqrt_f,
)
from flint.render.aaa_visbuffer import AAA_PUSH_WORDS, BIND_AAA_GLOBALS, BIND_AAA_HDR_OUT, BIND_AAA_POST_SCRATCH
from flint.render.sota_camera_dirt import (
    SOTA_DIRT_H,
    SOTA_DIRT_W,
    sota_dirt_offset,
    sota_lens_dirt_texture_cpu,
    sota_pack_dirt_u8,
    sota_upload_dirt,
)
from flint.render.sota_camera_vrs import (
    sota_vrs_classify_shader,
    sota_vrs_fill_shader,
    sota_vrs_rate_shader,
    sota_vrs_record_classify,
    sota_vrs_record_fill,
    sota_vrs_record_rate,
)

comptime SOTA_POST_BASE: Int = 1024            # == AAA_POST_BLOOM_BASE (post scratch words below are histogram/exposure)
comptime SOTA_POST_ADAPTED_EXP_WORD: Int = 64  # == AAA_POST_ADAPTED_EXP_WORD
comptime SOTA_DOF_RINGS: Int = 3               # 8 + 16 + 24 = 48 taps + centre
comptime SOTA_DOF_TILE: Int = 8                # half-res pixels per tile side
comptime SOTA_GHOSTS_MAX: Int = 8


# ============================================================================
# Settings + pipeline handles (one field on the renderer)
# ============================================================================

struct SotaCameraPost(Copyable, Movable):
    """Renderer settings for area D. Everything defaults off so existing frames are unchanged."""
    var camera: PhysicalCameraModel
    var drive_renderer: Bool        # camera sets ren.exposure / motion-blur shutter / legacy lens each frame
    var frame_dt_s: Float32         # frame interval used for the shutter fraction

    var enable_dof: Bool
    var dof_max_radius_px: Float32  # full-res px cap on CoC radius (<= 32)
    var dof_cat_eye: Float32        # 0 = off, ~0.3 = strong cat's-eye bokeh towards the frame edge

    var enable_flare: Bool
    var flare_intensity: Float32
    var flare_threshold: Float32    # exposed luminance where ghosts start
    var flare_ghosts: Int
    var flare_dispersion: Float32
    var flare_halo_radius: Float32
    var flare_halo_strength: Float32
    var coating_thickness_nm: Float32

    var enable_dirt: Bool
    var dirt_intensity: Float32
    var dirt_use_texture: Bool      # procedural dirt texture (`sota_camera_dirt.mojo`); False: inline specks
    var dirt_seed: UInt32
    var dirt_words: List[UInt32]    # packed 8-bit texture, generated on first use
    var dirt_words_seed: UInt32
    var dirt_uploaded_off: Int      # post-scratch word offset + buffer address of the uploaded copy
    var dirt_uploaded_addr: Int

    var enable_anamorphic_streak: Bool
    var anamorphic_streak_intensity: Float32
    var anamorphic_streak_length: Float32  # kernel reach as a fraction of the frame width

    var enable_chromatic_aberration: Bool
    var ca_strength: Float32        # lateral CA: relative magnification difference R vs B

    var enable_optical_vignette: Bool
    var optical_vignette_amount: Float32

    var enable_grain: Bool
    var grain_strength: Float32     # 1.0 = physical sensor noise for the camera's ISO

    var pipe_dof_setup: Int
    var pipe_dof_downsample: Int
    var pipe_dof_postfilter: Int
    var pipe_dof_tile: Int
    var pipe_dof_gather: Int
    var pipe_dof_composite: Int
    var pipe_flare_bright: Int
    var pipe_flare_ghost: Int
    var pipe_flare_blur: Int
    var pipe_streak: Int
    var pipe_lens_composite: Int
    var pipe_ca: Int
    # Compute VRS for the shading pass (`sota_camera_vrs.mojo`): rate map from the finished previous frame.
    var enable_vrs: Bool
    var vrs_threshold: Float32  # mean |error| of the coarse reconstruction in l/(1+l) display units
    var vrs_motion_px: Float32  # on-screen speed (output px / frame) at which the tolerance doubles
    var vrs_allow_4x4: Bool
    var vrs_rate_w: Int  # render size the current rate map was computed for (0: none yet)
    var vrs_rate_h: Int
    var pipe_vrs_rate: Int
    var pipe_vrs_classify: Int
    var pipe_vrs_fill: Int
    var frame_counter: UInt32

    def __init__(out self):
        self.camera = PhysicalCameraModel()
        self.drive_renderer = False
        self.frame_dt_s = Float32(1.0 / 60.0)
        self.enable_dof = False
        self.dof_max_radius_px = Float32(32.0)
        self.dof_cat_eye = Float32(0.0)
        self.enable_flare = False
        self.flare_intensity = Float32(1.0)  # 1 = physical coated-lens ghost energy; games often use 30-300
        self.flare_threshold = Float32(2.0)
        self.flare_ghosts = 6
        self.flare_dispersion = Float32(0.025)
        self.flare_halo_radius = Float32(0.42)
        self.flare_halo_strength = Float32(0.6)
        self.coating_thickness_nm = Float32(99.6)  # quarter wave of MgF2 (n = 1.38) at 550 nm
        self.enable_dirt = False
        self.dirt_intensity = Float32(0.25)
        self.dirt_use_texture = True
        self.dirt_seed = UInt32(0x5D17)
        self.dirt_words = List[UInt32]()
        self.dirt_words_seed = UInt32(0)
        self.dirt_uploaded_off = 0
        self.dirt_uploaded_addr = 0
        self.enable_anamorphic_streak = False
        self.anamorphic_streak_intensity = Float32(0.08)
        self.anamorphic_streak_length = Float32(0.25)
        self.enable_chromatic_aberration = False
        self.ca_strength = Float32(0.003)
        self.enable_optical_vignette = False
        self.optical_vignette_amount = Float32(1.0)
        self.enable_grain = False
        self.grain_strength = Float32(1.0)
        self.pipe_dof_setup = 0
        self.pipe_dof_downsample = 0
        self.pipe_dof_postfilter = 0
        self.pipe_dof_tile = 0
        self.pipe_dof_gather = 0
        self.pipe_dof_composite = 0
        self.pipe_flare_bright = 0
        self.pipe_flare_ghost = 0
        self.pipe_flare_blur = 0
        self.pipe_streak = 0
        self.pipe_lens_composite = 0
        self.pipe_ca = 0
        self.enable_vrs = False
        self.vrs_threshold = Float32(0.006)
        self.vrs_motion_px = Float32(8.0)
        self.vrs_allow_4x4 = True
        self.vrs_rate_w = 0
        self.vrs_rate_h = 0
        self.pipe_vrs_rate = 0
        self.pipe_vrs_classify = 0
        self.pipe_vrs_fill = 0
        self.frame_counter = 0

    def any_lens_effect(self) -> Bool:
        """Effects of the late lens + sensor pass (after bloom / motion blur / metering). Lateral CA is
        recorded separately and earlier (`sota_camera_record_ca`, right after DoF)."""
        return self.enable_flare or self.enable_dirt or self.enable_optical_vignette or self.enable_grain or self.enable_anamorphic_streak

    def create_pipelines(mut self, mut vk: VulkanGpu, pipe_layout: Int) -> Bool:
        self.pipe_dof_setup = vk.create_aaa_compute_pipeline(pipe_layout, sota_dof_setup_shader())
        self.pipe_dof_downsample = vk.create_aaa_compute_pipeline(pipe_layout, sota_dof_downsample_shader())
        self.pipe_dof_postfilter = vk.create_aaa_compute_pipeline(pipe_layout, sota_dof_postfilter_shader())
        self.pipe_dof_tile = vk.create_aaa_compute_pipeline(pipe_layout, sota_dof_tile_shader())
        self.pipe_dof_gather = vk.create_aaa_compute_pipeline(pipe_layout, sota_dof_gather_shader())
        self.pipe_dof_composite = vk.create_aaa_compute_pipeline(pipe_layout, sota_dof_composite_shader())
        self.pipe_flare_bright = vk.create_aaa_compute_pipeline(pipe_layout, sota_flare_bright_shader())
        self.pipe_flare_ghost = vk.create_aaa_compute_pipeline(pipe_layout, sota_flare_ghost_shader())
        self.pipe_flare_blur = vk.create_aaa_compute_pipeline(pipe_layout, sota_flare_blur_shader())
        self.pipe_streak = vk.create_aaa_compute_pipeline(pipe_layout, sota_streak_shader())
        self.pipe_lens_composite = vk.create_aaa_compute_pipeline(pipe_layout, sota_lens_composite_shader())
        self.pipe_ca = vk.create_aaa_compute_pipeline(pipe_layout, sota_ca_shader())
        self.pipe_vrs_rate = vk.create_aaa_compute_pipeline(pipe_layout, sota_vrs_rate_shader())
        self.pipe_vrs_classify = vk.create_aaa_compute_pipeline(pipe_layout, sota_vrs_classify_shader())
        self.pipe_vrs_fill = vk.create_aaa_compute_pipeline(pipe_layout, sota_vrs_fill_shader())
        return self.pipelines_ready()

    def pipelines_ready(self) -> Bool:
        return (
            self.pipe_dof_setup != 0
            and self.pipe_dof_downsample != 0
            and self.pipe_dof_postfilter != 0
            and self.pipe_dof_tile != 0
            and self.pipe_dof_gather != 0
            and self.pipe_dof_composite != 0
            and self.pipe_flare_bright != 0
            and self.pipe_flare_ghost != 0
            and self.pipe_flare_blur != 0
            and self.pipe_streak != 0
            and self.pipe_lens_composite != 0
            and self.pipe_ca != 0
        )

    def vrs_pipelines_ready(self) -> Bool:
        return self.pipe_vrs_rate != 0 and self.pipe_vrs_classify != 0 and self.pipe_vrs_fill != 0

    def vrs_active(self, render_w: Int, render_h: Int) -> Bool:
        """VRS shades this frame only with a rate map from the previous frame at the same render size."""
        return self.enable_vrs and self.vrs_pipelines_ready() and self.vrs_rate_w == render_w and self.vrs_rate_h == render_h

    def frame_sync(mut self, adapted_exposure: Float32, auto_exposure_on: Bool) -> Tuple[Float32, Float32]:
        """Exposure triangle for this frame. Returns `(base_exposure, motion_blur_shutter_fraction)`.
        Manual: exposure from EV100. Auto: the GPU meter's last adapted multiplier is the metered exposure;
        the body re-solves shutter / ISO (and aperture in shutter priority) from it, so motion blur and grain
        follow the light; the renderer's adapted multiplier keeps doing the actual scaling."""
        if auto_exposure_on and self.camera.exposure_mode != PHYS_EXPOSURE_MANUAL:
            var a = adapted_exposure if adapted_exposure > Float32(1.0e-6) else Float32(1.0)
            var metered = self.camera.ev100_for_exposure(a)
            var tri = self.camera.solve_exposure_triangle(metered)
            self.camera.apply_exposure_triangle(tri)
            return (Float32(1.0), self.camera.shutter_fraction(self.frame_dt_s))
        return (self.camera.exposure_scale(), self.camera.shutter_fraction(self.frame_dt_s))


# ============================================================================
# Scratch layout (words in BIND_AAA_POST_SCRATCH). Must stay below 1024 + 10 * max_pix (UI prims follow).
# ============================================================================

struct SotaDofLayout(Copyable, ImplicitlyCopyable, Movable):
    """Scratch layout of the DoF chain: half-res setup (4 w/px), quarter-res prefiltered level (4 w/px),
    8x8 tiles (2 w), gather output and post-filtered output (8 w/px each). Total ~5.3 * max_pix words."""
    var hw: Int
    var hh: Int
    var qw: Int
    var qh: Int
    var tx: Int
    var ty: Int
    var setup_off: Int
    var level1_off: Int
    var tile_off: Int
    var gather_off: Int
    var filtered_off: Int
    var end_off: Int

    def __init__(out self, out_w: Int, out_h: Int):
        self.hw = (out_w + 1) // 2
        self.hh = (out_h + 1) // 2
        self.qw = (self.hw + 1) // 2
        self.qh = (self.hh + 1) // 2
        self.tx = (self.hw + SOTA_DOF_TILE - 1) // SOTA_DOF_TILE
        self.ty = (self.hh + SOTA_DOF_TILE - 1) // SOTA_DOF_TILE
        self.setup_off = SOTA_POST_BASE
        self.level1_off = self.setup_off + self.hw * self.hh * 4
        self.tile_off = self.level1_off + self.qw * self.qh * 4
        self.gather_off = self.tile_off + self.tx * self.ty * 2 + 16
        self.filtered_off = self.gather_off + self.hw * self.hh * 8
        self.end_off = self.filtered_off + self.hw * self.hh * 8


def sota_lens_layout(out_w: Int, out_h: Int, max_pix: Int) -> Tuple[Int, Int, Int, Int, Int, Int, Int]:
    """`(quarter_w, quarter_h, hdr_copy_off, bright_off, ghost_off, ghost_blur_off, glow_off)`.
    The CA source copy (8 words/px) sits at the base; the quarter-res buffers go after the saved bloom mip
    the display pass reads at `1024 + 8 * max_pix` (max_pix / 4 words)."""
    var qw = (out_w + 3) // 4
    var qh = (out_h + 3) // 4
    var qc = qw * qh
    var copy_off = SOTA_POST_BASE
    var bright_off = SOTA_POST_BASE + max_pix * 8 + max_pix // 4 + 64
    var ghost_off = bright_off + qc * 4
    var blur_off = ghost_off + qc * 4
    var glow_off = blur_off + qc * 4
    return (qw, qh, copy_off, bright_off, ghost_off, blur_off, glow_off)


# ============================================================================
# CPU references
# ============================================================================

def eval_coc_px_cpu(k_inf_px: Float32, focus_m: Float32, z_m: Float32, max_coc: Float32) -> Float32:
    """Signed CoC radius (px) of the thin lens: `K * (1 - S / z)`, clamped."""
    var z = max(Float32(0.01), z_m)
    return max(-max_coc, min(max_coc, k_inf_px * (Float32(1.0) - focus_m / z)))


def aperture_poly_factor_cpu(theta: Float32, blades: Int, rotation_rad: Float32, curvature: Float32) -> Float32:
    """Radius of the iris polygon (inscribed circle at `cos(pi/n)`, vertices at 1.0) along `theta`,
    blended to the circle by `curvature`."""
    if blades < 3:
        return Float32(1.0)
    var sector = Float32(6.283185307) / Float32(blades)
    var a = theta - rotation_rad
    a = a - floor(a / sector) * sector - sector * Float32(0.5)
    var poly = cos(sector * Float32(0.5)) / max(cos(a), Float32(0.2))
    return poly + (Float32(1.0) - poly) * max(Float32(0.0), min(Float32(1.0), curvature))


def dof_ring_tap_cpu(ring: Int, j: Int) -> Tuple[Float32, Float32]:
    """`(r_norm, theta)` of tap `j` on ring `ring` (1-based): `8 * ring` taps, half-step stagger on odd rings."""
    var n = 8 * ring
    var stagger = Float32(0.5) if (ring & 1) == 1 else Float32(0.0)
    var theta = Float32(6.283185307) * (Float32(j) + stagger) / Float32(n)
    return (Float32(ring) / Float32(SOTA_DOF_RINGS), theta)


def ghost_magnification(i: Int) -> Float32:
    """Gather magnification of ghost `i` (negative = mirrored through the optical centre). A ghost is
    the source imaged at scale `1/m`; its irradiance scales by `m^2`."""
    var table: List[Float32] = [-1.0, -0.55, 0.62, -2.1, 1.45, -0.32, 2.6, -1.25]
    return table[i % SOTA_GHOSTS_MAX]


def thin_film_reflectance_cpu(thickness_nm: Float32, wavelength_nm: Float32) -> Float32:
    """Single-layer AR coating (MgF2 n=1.38 on n=1.52 glass) at normal incidence, Airy reflectance."""
    var n0 = Float32(1.0)
    var n1 = Float32(1.38)
    var n2 = Float32(1.52)
    var r01 = (n0 - n1) / (n0 + n1)
    var r12 = (n1 - n2) / (n1 + n2)
    var delta = Float32(4.0 * 3.14159265) * n1 * thickness_nm / wavelength_nm
    var c = cos(delta)
    var num = r01 * r01 + r12 * r12 + Float32(2.0) * r01 * r12 * c
    var den = Float32(1.0) + r01 * r01 * r12 * r12 + Float32(2.0) * r01 * r12 * c
    return num / den


def ghost_tint_cpu(i: Int, coating_nm: Float32) -> Vec3:
    """Fraction of the source reaching ghost `i`: two reflections off coated surfaces whose effective
    coating thickness differs per surface pair (`1 + 0.13 i`, and `x 1.07` for the second surface), giving
    the familiar magenta / green / amber ghosts at ~1e-4 of the source (`flare_intensity = 1` is physical)."""
    var d = coating_nm * (Float32(1.0) + Float32(0.13) * Float32(i))
    var d2 = d * Float32(1.07)
    var r = thin_film_reflectance_cpu(d, 640.0) * thin_film_reflectance_cpu(d2, 640.0)
    var g = thin_film_reflectance_cpu(d, 550.0) * thin_film_reflectance_cpu(d2, 550.0)
    var b = thin_film_reflectance_cpu(d, 460.0) * thin_film_reflectance_cpu(d2, 460.0)
    return Vec3(r, g, b)


def lateral_ca_scale_cpu(channel: Int, ca_strength: Float32) -> Float32:
    """Per-channel magnification about the optical centre (R magnified, B minified)."""
    if channel == 0:
        return Float32(1.0) + ca_strength
    if channel == 2:
        return Float32(1.0) - ca_strength
    return Float32(1.0)


def optical_vignette_cpu(ndc_x: Float32, ndc_y: Float32, aspect: Float32, strength: Float32) -> Float32:
    var x = ndc_x * aspect
    var r = sqrt(x * x + ndc_y * ndc_y) / sqrt(aspect * aspect + Float32(1.0))
    var t = max(Float32(0.0), min(Float32(1.0), (r - Float32(0.45)) / Float32(0.55)))
    var s = t * t * (Float32(3.0) - Float32(2.0) * t)
    return Float32(1.0) - strength * Float32(0.6) * s


def eval_dof_cpu(
    rgb: List[Vec3], depth_m: List[Float32], w: Int, h: Int, k_inf_px: Float32, focus_m: Float32, max_coc: Float32, blades: Int
) -> List[Vec3]:
    """Full-resolution CPU reference of the scatter-as-gather DoF (no half-res, no tiles: exhaustive
    neighbourhood within `max_coc`). Used to validate disc radius and energy conservation."""
    var coc = List[Float32](length=w * h, fill=0.0)
    for i in range(w * h):
        coc[i] = eval_coc_px_cpu(k_inf_px, focus_m, depth_m[i], max_coc)
    var out = List[Vec3](length=w * h, fill=Vec3(0.0, 0.0, 0.0))
    var rad = Int(max_coc) + 1
    for y in range(h):
        for x in range(w):
            var c = coc[y * w + x]
            var far_acc = Vec3(0.0, 0.0, 0.0)
            var far_w = Float32(0.0)
            var near_acc = Vec3(0.0, 0.0, 0.0)
            var near_w = Float32(0.0)
            var near_cnt = Float32(0.0)
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    var sx = max(0, min(w - 1, x + dx))
                    var sy = max(0, min(h - 1, y + dy))
                    var sc = coc[sy * w + sx]
                    var d = sqrt(Float32(dx * dx + dy * dy))
                    var theta = Float32(0.0)
                    if dx != 0 or dy != 0:
                        theta = atan2(Float32(dy), Float32(dx))
                    var poly = aperture_poly_factor_cpu(theta, blades, 0.0, 0.0)
                    var d_eff = d / poly
                    var col = rgb[sy * w + sx]
                    var eff = min(max(sc, Float32(0.0)), max(c, Float32(0.0)))
                    var wf = max(Float32(0.0), min(Float32(1.0), eff - d_eff + Float32(0.5))) / max(eff * eff, Float32(1.0))
                    if wf > Float32(0.0):
                        far_acc = far_acc + col * wf
                        far_w += wf
                    if sc < Float32(-0.5):
                        var r = -sc
                        var gate = max(Float32(0.0), min(Float32(1.0), (r - Float32(0.5)) * Float32(2.0)))
                        var wn = max(Float32(0.0), min(Float32(1.0), r - d_eff + Float32(0.5))) / max(r * r, Float32(1.0)) * gate
                        near_acc = near_acc + col * wn
                        near_w += wn
                    near_cnt += Float32(1.0)
            var far_col = far_acc * (Float32(1.0) / max(far_w, Float32(1.0e-6)))
            var sharp = rgb[y * w + x]
            var fb = max(Float32(0.0), min(Float32(1.0), c - Float32(0.5)))
            var res = sharp * (Float32(1.0) - fb) + far_col * fb
            if near_w > Float32(0.0):
                # coverage: sum of 1/r^2 weights over the pi r^2 footprint ~= pi when fully covered
                var alpha = max(Float32(0.0), min(Float32(1.0), near_w / Float32(3.14159265)))
                var near_col = near_acc * (Float32(1.0) / near_w)
                res = res * (Float32(1.0) - alpha) + near_col * alpha
            out[y * w + x] = res
    return out^


# ============================================================================
# Shader helpers
# ============================================================================

@always_inline
def _coc_dsl(z: GpuF32, k_inf: GpuF32, focus: GpuF32, max_coc: GpuF32) -> GpuF32:
    var zz = max_f(z, 0.01)
    var c = k_inf * (Float32(1.0) - focus / zz)
    return clamp_f(c, -max_coc, max_coc)


@always_inline
def _luma(v: GpuVec3) -> GpuF32:
    return v.x() * Float32(0.2126) + v.y() * Float32(0.7152) + v.z() * Float32(0.0722)


@always_inline
def _saturate(x: GpuF32) -> GpuF32:
    return clamp_f(x, 0.0, 1.0)


@always_inline
def _poly_factor_dsl(mut dsl: ShaderDsl, theta: Float32, blades: GpuF32, rot: GpuF32, curv: GpuF32) -> GpuF32:
    var sector = dsl.f32(6.283185307) / max_f(blades, 3.0)
    var a = dsl.f32(theta) - rot
    a = a - floor_f(a / sector) * sector - sector * Float32(0.5)
    var poly = cos_f(sector * Float32(0.5)) / max_f(cos_f(a), 0.2)
    return poly + (dsl.f32(1.0) - poly) * curv


@always_inline
def _pcg(v: GpuU32) -> GpuU32:
    var state = v * UInt32(747796405) + UInt32(2891336453)
    var word = ((state >> ((state >> UInt32(28)) + UInt32(4))) ^ state) * UInt32(277803737)
    return (word >> UInt32(22)) ^ word


@always_inline
def _u01(h: GpuU32) -> GpuF32:
    return (h >> UInt32(8)).to_f32() * Float32(1.0 / 16777216.0)


def _bilerp_post(
    mut dsl: ShaderDsl, b_post: BufferRef[1], base: GpuU32, stride: UInt32, w: GpuU32, h: GpuU32, fx_in: GpuF32, fy_in: GpuF32, comp_off: UInt32
) -> GpuVec3:
    """Bilinear fetch of 3 floats at `comp_off` from a `stride`-word image at `base` (texel-space coords)."""
    var fw = w.to_f32()
    var fh = h.to_f32()
    var fx = clamp_f(fx_in, dsl.f32(0.0), max_f(fw - Float32(1.0), 0.0))
    var fy = clamp_f(fy_in, dsl.f32(0.0), max_f(fh - Float32(1.0), 0.0))
    var x0 = min_u(w - UInt32(1), floor_f(fx).saturate_u32())
    var y0 = min_u(h - UInt32(1), floor_f(fy).saturate_u32())
    var x1 = min_u(w - UInt32(1), x0 + UInt32(1))
    var y1 = min_u(h - UInt32(1), y0 + UInt32(1))
    var tx = clamp_f(fx - x0.to_f32(), 0.0, 1.0)
    var ty = clamp_f(fy - y0.to_f32(), 0.0, 1.0)
    var c00 = b_post.load_vec3(base + (y0 * w + x0) * stride + comp_off)
    var c10 = b_post.load_vec3(base + (y0 * w + x1) * stride + comp_off)
    var c01 = b_post.load_vec3(base + (y1 * w + x0) * stride + comp_off)
    var c11 = b_post.load_vec3(base + (y1 * w + x1) * stride + comp_off)
    var one = dsl.f32(1.0)
    var top = c00 * (one - tx) + c10 * tx
    var bot = c01 * (one - tx) + c11 * tx
    return top * (one - ty) + bot * ty


def _bilerp_post_f(
    mut dsl: ShaderDsl, b_post: BufferRef[1], base: GpuU32, stride: UInt32, w: GpuU32, h: GpuU32, fx_in: GpuF32, fy_in: GpuF32, comp_off: UInt32
) -> GpuF32:
    var fw = w.to_f32()
    var fh = h.to_f32()
    var fx = clamp_f(fx_in, dsl.f32(0.0), max_f(fw - Float32(1.0), 0.0))
    var fy = clamp_f(fy_in, dsl.f32(0.0), max_f(fh - Float32(1.0), 0.0))
    var x0 = min_u(w - UInt32(1), floor_f(fx).saturate_u32())
    var y0 = min_u(h - UInt32(1), floor_f(fy).saturate_u32())
    var x1 = min_u(w - UInt32(1), x0 + UInt32(1))
    var y1 = min_u(h - UInt32(1), y0 + UInt32(1))
    var tx = clamp_f(fx - x0.to_f32(), 0.0, 1.0)
    var ty = clamp_f(fy - y0.to_f32(), 0.0, 1.0)
    var c00 = b_post.load_f32(base + (y0 * w + x0) * stride + comp_off)
    var c10 = b_post.load_f32(base + (y0 * w + x1) * stride + comp_off)
    var c01 = b_post.load_f32(base + (y1 * w + x0) * stride + comp_off)
    var c11 = b_post.load_f32(base + (y1 * w + x1) * stride + comp_off)
    var one = dsl.f32(1.0)
    var top = c00 * (one - tx) + c10 * tx
    var bot = c01 * (one - tx) + c11 * tx
    return top * (one - ty) + bot * ty


# ============================================================================
# 1. DoF setup: full -> half res colour + signed CoC (half-res px)
# ============================================================================

def sota_dof_setup_shader() -> List[UInt32]:
    """Push: 0 W, 1 H, 2 hw, 3 hh, 4 dst_off, 5 K (CoC radius at infinity, full px), 6 focus_m, 7 max_coc (full px)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var hx = dsl.global_id_x()
    var hy = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var hw = dsl.push_u32(2)
    var hh = dsl.push_u32(3)
    var dst = dsl.push_u32(4)
    var k_inf = dsl.push_f32(5)
    var focus = dsl.push_f32(6)
    var max_coc = dsl.push_f32(7)
    dsl.return_if(hx.ge(hw) | hy.ge(hh))

    var acc = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var c_min = dsl.var_f32_const(1.0e9)
    var c_sum = dsl.var_f32_const(0.0)
    for j in range(2):
        for i in range(2):
            var fx = min_u(w_u - UInt32(1), hx * UInt32(2) + UInt32(i))
            var fy = min_u(h_u - UInt32(1), hy * UInt32(2) + UInt32(j))
            var base = b_hdr.base_word(fy * w_u + fx)
            var rgb = b_hdr.load_vec3(base)
            var z = b_hdr.load_f32(base + UInt32(3))
            var c = _coc_dsl(z, k_inf, focus, max_coc)
            acc.set(acc.get() + rgb)
            c_min.set(min_f(c_min.get(), c))
            c_sum.set(c_sum.get() + c)
    var cm = c_min.get()
    var coc_full = select_f(cm.lt(-0.5), cm, c_sum.get() * Float32(0.25))
    var o = dst + (hy * hw + hx) * UInt32(4)
    b_post.store_vec3(o, acc.get() * Float32(0.25))
    b_post.store_f32(o + UInt32(3), coc_full * Float32(0.5))
    return dsl.finish()


# ============================================================================
# 2. DoF tiles: max near / far CoC per 8x8 half-res tile
# ============================================================================

def sota_dof_tile_shader() -> List[UInt32]:
    """One thread per tile. Push: 0 hw, 1 hh, 2 src_off, 3 tx, 4 ty, 5 tile_off."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var tx = dsl.global_id_x()
    var ty = dsl.global_id_y()
    var hw = dsl.push_u32(0)
    var hh = dsl.push_u32(1)
    var src = dsl.push_u32(2)
    var tiles_x = dsl.push_u32(3)
    var tiles_y = dsl.push_u32(4)
    var tile_off = dsl.push_u32(5)
    dsl.return_if(tx.ge(tiles_x) | ty.ge(tiles_y))
    var near_max = dsl.var_f32_const(0.0)
    var far_max = dsl.var_f32_const(0.0)
    for j in range(SOTA_DOF_TILE):
        for i in range(SOTA_DOF_TILE):
            var px = min_u(hw - UInt32(1), tx * UInt32(SOTA_DOF_TILE) + UInt32(i))
            var py = min_u(hh - UInt32(1), ty * UInt32(SOTA_DOF_TILE) + UInt32(j))
            var c = b_post.load_f32(src + (py * hw + px) * UInt32(4) + UInt32(3))
            near_max.set(max_f(near_max.get(), -c))
            far_max.set(max_f(far_max.get(), c))
    var o = tile_off + (ty * tiles_x + tx) * UInt32(2)
    b_post.store_f32(o, near_max.get())
    b_post.store_f32(o + UInt32(1), far_max.get())
    return dsl.finish()


# ============================================================================
# 3. DoF scatter-as-gather (half res)
# ============================================================================

def sota_dof_downsample_shader() -> List[UInt32]:
    """Half -> quarter res prefiltered level (colour box average, near-conservative CoC kept in half-px units).
    Push: 0 hw, 1 hh, 2 src_off, 3 qw, 4 qh, 5 dst_off."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var qx = dsl.global_id_x()
    var qy = dsl.global_id_y()
    var hw = dsl.push_u32(0)
    var hh = dsl.push_u32(1)
    var src = dsl.push_u32(2)
    var qw = dsl.push_u32(3)
    var qh = dsl.push_u32(4)
    var dst = dsl.push_u32(5)
    dsl.return_if(qx.ge(qw) | qy.ge(qh))
    var acc = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var c_min = dsl.var_f32_const(1.0e9)
    var c_sum = dsl.var_f32_const(0.0)
    for j in range(2):
        for i in range(2):
            var sx = min_u(hw - UInt32(1), qx * UInt32(2) + UInt32(i))
            var sy = min_u(hh - UInt32(1), qy * UInt32(2) + UInt32(j))
            var o = src + (sy * hw + sx) * UInt32(4)
            acc.set(acc.get() + b_post.load_vec3(o))
            var c = b_post.load_f32(o + UInt32(3))
            c_min.set(min_f(c_min.get(), c))
            c_sum.set(c_sum.get() + c)
    var cm = c_min.get()
    var oo = dst + (qy * qw + qx) * UInt32(4)
    b_post.store_vec3(oo, acc.get() * Float32(0.25))
    b_post.store_f32(oo + UInt32(3), select_f(cm.lt(-0.5), cm, c_sum.get() * Float32(0.25)))
    return dsl.finish()


def _dof_gather_taps(
    mut dsl: ShaderDsl,
    b_post: BufferRef[1],
    use_level1: Bool,
    src: GpuU32,
    l1: GpuU32,
    hw: GpuU32,
    hh: GpuU32,
    qw: GpuU32,
    qh: GpuU32,
    fpx: GpuF32,
    fpy: GpuF32,
    big_r: GpuF32,
    c_far: GpuF32,
    rot_c: GpuF32,
    rot_s: GpuF32,
    phi: GpuF32,
    jr: GpuF32,
    blades: GpuF32,
    rot: GpuF32,
    curv: GpuF32,
    cat: GpuF32,
    cat_x: GpuF32,
    cat_y: GpuF32,
    far_acc: GpuVarVec3,
    far_w: GpuVarF32,
    near_acc: GpuVarVec3,
    near_w: GpuVarF32,
):
    var fw1 = hw.to_f32() - Float32(1.0)
    var fh1 = hh.to_f32() - Float32(1.0)
    for ring in range(1, SOTA_DOF_RINGS + 1):
        for j in range(8 * ring):
            var tap = dof_ring_tap_cpu(ring, j)
            var theta = tap[1]
            # Per-pixel rotation `phi` and radial jitter `jr` (ring k spans ((k-1)/K, k/K]) turn the
            # residual tap structure into fine noise that the 3x3 post-filter removes.
            var r_norm = (dsl.f32(Float32(ring)) - jr) * Float32(1.0 / Float32(SOTA_DOF_RINGS))
            # Area each tap stands for (unit-disc units, sums to ~pi): ring k has 8k taps over an annulus of
            # area pi (2k-1) / K^2, so without this the inner rings are oversampled and a highlight's bokeh
            # peaks in the middle (smaller half-max disc, +16 % energy).
            var area = r_norm * Float32(6.283185307 / Float32(8 * ring * SOTA_DOF_RINGS))
            var poly = _poly_factor_dsl_rt(dsl, dsl.f32(theta) + phi, blades, rot, curv)
            var bx = r_norm * cos(theta)
            var by = r_norm * sin(theta)
            var ux = (bx * rot_c - by * rot_s) * poly
            var uy = (bx * rot_s + by * rot_c) * poly
            var cx = ux + cat_x
            var cy = uy + cat_y
            var cat_w = _saturate((dsl.f32(1.0) - sqrt_f(cx * cx + cy * cy)) * Float32(6.0) + Float32(1.0))
            cat_w = select_f(cat.gt(0.001), cat_w, dsl.f32(1.0))
            var ox = ux * big_r
            var oy = uy * big_r
            var d = r_norm * big_r
            var s_rgb: GpuVec3
            var s_c: GpuF32
            if use_level1:
                var lx = (fpx + ox) * Float32(0.5) - Float32(0.25)
                var ly = (fpy + oy) * Float32(0.5) - Float32(0.25)
                s_rgb = _bilerp_post(dsl, b_post, l1, UInt32(4), qw, qh, lx, ly, UInt32(0))
                s_c = _bilerp_post_f(dsl, b_post, l1, UInt32(4), qw, qh, lx, ly, UInt32(3))
            else:
                var sx = clamp_f(fpx + ox + Float32(0.5), dsl.f32(0.0), fw1).saturate_u32()
                var sy = clamp_f(fpy + oy + Float32(0.5), dsl.f32(0.0), fh1).saturate_u32()
                var s_off = src + (sy * hw + sx) * UInt32(4)
                s_rgb = b_post.load_vec3(s_off)
                s_c = b_post.load_f32(s_off + UInt32(3))
            # Far field: background never spreads further than the centre's own CoC.
            var eff = min_f(max_f(s_c, 0.0), c_far)
            var wf = _saturate(eff - d + Float32(0.5)) / max_f(eff * eff, 1.0) * cat_w * area
            far_acc.set(far_acc.get() + s_rgb * wf)
            far_w.set(far_w.get() + wf)
            # Near field: foreground spreads by its own CoC; density-normalised coverage.
            # Only real defocus belongs to the near layer: a CoC under 0.5 px is in focus (gate ramps to 1 at 1 px).
            var nr = max_f(-s_c, 0.0)
            var gate = _saturate((nr - Float32(0.5)) * Float32(2.0))
            var wn = _saturate(nr - d + Float32(0.5)) * big_r * big_r / max_f(nr * nr, 1.0) * gate * cat_w * area
            near_acc.set(near_acc.get() + s_rgb * wn)
            near_w.set(near_w.get() + wn)


@always_inline
def _poly_factor_dsl_rt(mut dsl: ShaderDsl, theta: GpuF32, blades: GpuF32, rot: GpuF32, curv: GpuF32) -> GpuF32:
    var sector = dsl.f32(6.283185307) / max_f(blades, 3.0)
    var a = theta - rot
    a = a - floor_f(a / sector) * sector - sector * Float32(0.5)
    var poly = cos_f(sector * Float32(0.5)) / max_f(cos_f(a), 0.2)
    return poly + (dsl.f32(1.0) - poly) * curv


def sota_dof_gather_shader() -> List[UInt32]:
    """Push: 0 hw, 1 hh, 2 src_off, 3 tile_off, 4 tiles_x, 5 tiles_y, 6 out_off, 7 max_r (half px),
    8 blades (f, >= 3; 64 = round), 9 blade rotation (rad), 10 cat's-eye amount, 11 blade curvature,
    12 level1_off, 13 qw, 14 qh. Taps come from the prefiltered quarter-res level once the kernel radius
    exceeds 6 half px (tap spacing > 2 px).
    Output per half pixel (8 words): far.rgb, centre CoC, near.rgb * alpha, alpha."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var hw = dsl.push_u32(0)
    var hh = dsl.push_u32(1)
    var src = dsl.push_u32(2)
    var tile_off = dsl.push_u32(3)
    var tiles_x = dsl.push_u32(4)
    var tiles_y = dsl.push_u32(5)
    var out_off = dsl.push_u32(6)
    var max_r = max_f(dsl.push_f32(7), 1.0)
    var blades = dsl.push_f32(8)
    var rot = dsl.push_f32(9)
    var cat = dsl.push_f32(10)
    var curv = clamp_f(dsl.push_f32(11), 0.0, 1.0)
    var l1 = dsl.push_u32(12)
    var qw = dsl.push_u32(13)
    var qh = dsl.push_u32(14)
    dsl.return_if(px.ge(hw) | py.ge(hh))

    var c_off = src + (py * hw + px) * UInt32(4)
    var c_rgb = b_post.load_vec3(c_off)
    var c = b_post.load_f32(c_off + UInt32(3))
    var c_far = max_f(c, 0.0)

    # Near-field reach: max near CoC over the 5x5 tile neighbourhood (16 half px reach).
    var tx_c = px // UInt32(SOTA_DOF_TILE)
    var ty_c = py // UInt32(SOTA_DOF_TILE)
    var near_reach = dsl.var_f32_const(0.0)
    for dj in range(-2, 3):
        for di in range(-2, 3):
            var tx_s = select_f(
                (tx_c.to_f32() + Float32(di)).lt(0.0), dsl.f32(0.0), min_f(tx_c.to_f32() + Float32(di), tiles_x.to_f32() - Float32(1.0))
            ).saturate_u32()
            var ty_s = select_f(
                (ty_c.to_f32() + Float32(dj)).lt(0.0), dsl.f32(0.0), min_f(ty_c.to_f32() + Float32(dj), tiles_y.to_f32() - Float32(1.0))
            ).saturate_u32()
            near_reach.set(max_f(near_reach.get(), b_post.load_f32(tile_off + (ty_s * tiles_x + tx_s) * UInt32(2))))
    # +0.5: the tap weights ramp over [CoC - 0.5, CoC + 0.5], so the kernel must reach CoC + 0.5.
    var big_r = min_f(max_f(abs_f(c), near_reach.get()), max_r) + Float32(0.5)

    var o = out_off + (py * hw + px) * UInt32(8)
    dsl.if_then(big_r.lt(1.0))
    b_post.store_vec3(o, c_rgb)
    b_post.store_f32(o + UInt32(3), c)
    b_post.store_vec3(o + UInt32(4), dsl.vec3_const(0.0, 0.0, 0.0))
    b_post.store_f32(o + UInt32(7), dsl.f32(0.0))
    dsl.else_branch()

    var fpx = px.to_f32()
    var fpy = py.to_f32()
    # Cat's eye: exit pupil clipped by the lens barrel; the clip circle slides with the image position.
    var ndc_x = (fpx + Float32(0.5)) / hw.to_f32() * Float32(2.0) - Float32(1.0)
    var ndc_y = (fpy + Float32(0.5)) / hh.to_f32() * Float32(2.0) - Float32(1.0)
    var cat_x = ndc_x * cat
    var cat_y = ndc_y * cat
    # Interleaved gradient noise (Jimenez 2014) -> per-pixel pattern rotation.
    var ign = fract_f(Float32(52.9829189) * fract_f(fpx * Float32(0.06711056) + fpy * Float32(0.00583715)))
    var phi = ign * Float32(6.283185307 / 8.0)
    var rot_c = cos_f(phi)
    var rot_s = sin_f(phi)
    var jr = fract_f(Float32(52.9829189) * fract_f((fpx + Float32(5.588238)) * Float32(0.00583715) + (fpy + Float32(17.31)) * Float32(0.06711056)))

    # The centre tap must come from the same (prefiltered) level as the ring taps, or a highlight keeps
    # an unfiltered hot dot in the middle of its bokeh disc.
    var c_l1 = _bilerp_post(dsl, b_post, l1, UInt32(4), qw, qh, fpx * Float32(0.5) - Float32(0.25), fpy * Float32(0.5) - Float32(0.25), UInt32(0))
    var c_src = select_vec3(big_r.gt(6.0), c_l1, c_rgb)
    # Ring 1 (radially jittered down to r = 0) already covers the kernel centre, so the centre tap only
    # needs a token weight: it keeps the weights non-zero when every ring tap is rejected. A real area
    # weight double-counts the centre and puts a hot spot in every highlight's bokeh disc.
    var a_c = Float32(1.0e-3)
    var w_c = dsl.f32(a_c) / max_f(c_far * c_far, 1.0)
    var far_acc = dsl.var_vec3(c_src * w_c)
    var far_w = dsl.var_f32(w_c)
    var near_r_c = max_f(-c, 0.0)
    var wn_c = big_r * big_r / max_f(near_r_c * near_r_c, 1.0) * _saturate((near_r_c - Float32(0.5)) * Float32(2.0)) * a_c
    var near_acc = dsl.var_vec3(c_src * wn_c)
    var near_w = dsl.var_f32(wn_c)
    dsl.if_then(big_r.gt(6.0))
    _dof_gather_taps(dsl, b_post, True, src, l1, hw, hh, qw, qh, fpx, fpy, big_r, c_far, rot_c, rot_s, phi, jr, blades, rot, curv, cat, cat_x, cat_y, far_acc, far_w, near_acc, near_w)
    dsl.else_branch()
    _dof_gather_taps(dsl, b_post, False, src, l1, hw, hh, qw, qh, fpx, fpy, big_r, c_far, rot_c, rot_s, phi, jr, blades, rot, curv, cat, cat_x, cat_y, far_acc, far_w, near_acc, near_w)
    dsl.end_if()

    var far_col = far_acc.get() / max_f(far_w.get(), 1.0e-12)
    var nw = near_w.get()
    # Coverage: the area-weighted near weights sum to ~pi over a fully covered kernel.
    var alpha = _saturate(nw * Float32(1.0 / 3.14159265))
    var near_pm = near_acc.get() * (alpha / max_f(nw, 1.0e-6))
    b_post.store_vec3(o, far_col)
    b_post.store_f32(o + UInt32(3), c)
    b_post.store_vec3(o + UInt32(4), near_pm)
    b_post.store_f32(o + UInt32(7), alpha)
    dsl.end_if()
    return dsl.finish()


def sota_dof_postfilter_shader() -> List[UInt32]:
    """3x3 tent over the gather output (half res). Far colour only mixes neighbours of similar CoC so
    in-focus edges stay crisp; the near layer (already soft) is filtered unconditionally.
    Push: 0 hw, 1 hh, 2 src_off, 3 dst_off."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var hw = dsl.push_u32(0)
    var hh = dsl.push_u32(1)
    var src = dsl.push_u32(2)
    var dst = dsl.push_u32(3)
    dsl.return_if(px.ge(hw) | py.ge(hh))
    var co = src + (py * hw + px) * UInt32(8)
    var c = b_post.load_f32(co + UInt32(3))
    var tol = max_f(abs_f(c) * Float32(0.5), 1.0)
    var far_acc = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var far_w = dsl.var_f32_const(0.0)
    var near_acc = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var a_acc = dsl.var_f32_const(0.0)
    for dj in range(-1, 2):
        for di in range(-1, 2):
            var wt = Float32((2 - abs(di)) * (2 - abs(dj))) / Float32(16.0)
            var sx = clamp_f(px.to_f32() + Float32(di), dsl.f32(0.0), hw.to_f32() - Float32(1.0)).saturate_u32()
            var sy = clamp_f(py.to_f32() + Float32(dj), dsl.f32(0.0), hh.to_f32() - Float32(1.0)).saturate_u32()
            var so = src + (sy * hw + sx) * UInt32(8)
            var sc = b_post.load_f32(so + UInt32(3))
            var ok = _saturate(dsl.f32(2.0) - abs_f(sc - c) / tol) * wt
            far_acc.set(far_acc.get() + b_post.load_vec3(so) * ok)
            far_w.set(far_w.get() + ok)
            near_acc.set(near_acc.get() + b_post.load_vec3(so + UInt32(4)) * wt)
            a_acc.set(a_acc.get() + b_post.load_f32(so + UInt32(7)) * wt)
    var oo = dst + (py * hw + px) * UInt32(8)
    b_post.store_vec3(oo, far_acc.get() / max_f(far_w.get(), 1.0e-6))
    b_post.store_f32(oo + UInt32(3), c)
    b_post.store_vec3(oo + UInt32(4), near_acc.get())
    b_post.store_f32(oo + UInt32(7), a_acc.get())
    return dsl.finish()


# ============================================================================
# 4. DoF composite (full res, in place on the HDR buffer)
# ============================================================================

def sota_dof_composite_shader() -> List[UInt32]:
    """Push: 0 W, 1 H, 2 hw, 3 hh, 4 gather_off, 5 K, 6 focus_m, 7 max_coc (full px)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var hw = dsl.push_u32(2)
    var hh = dsl.push_u32(3)
    var g_off = dsl.push_u32(4)
    var k_inf = dsl.push_f32(5)
    var focus = dsl.push_f32(6)
    var max_coc = dsl.push_f32(7)
    dsl.return_if(px.ge(w_u) | py.ge(h_u))

    var base = b_hdr.base_word(py * w_u + px)
    var sharp = b_hdr.load_vec3(base)
    var z = b_hdr.load_f32(base + UInt32(3))
    var c = _coc_dsl(z, k_inf, focus, max_coc)
    var hx = (px.to_f32() + Float32(0.5)) * Float32(0.5) - Float32(0.5)
    var hy = (py.to_f32() + Float32(0.5)) * Float32(0.5) - Float32(0.5)
    var far_col = _bilerp_post(dsl, b_post, g_off, UInt32(8), hw, hh, hx, hy, UInt32(0))
    var near_pm = _bilerp_post(dsl, b_post, g_off, UInt32(8), hw, hh, hx, hy, UInt32(4))
    var alpha = _bilerp_post_f(dsl, b_post, g_off, UInt32(8), hw, hh, hx, hy, UInt32(7))
    var fb = _saturate(c - Float32(0.5))
    var res = sharp * (dsl.f32(1.0) - fb) + far_col * fb
    res = res * (dsl.f32(1.0) - alpha) + near_pm
    b_hdr.store_vec3(base, res)
    return dsl.finish()


# ============================================================================
# 5. Flare bright pass (quarter res, exposure aware)
# ============================================================================

def sota_flare_bright_shader() -> List[UInt32]:
    """Push: 0 W, 1 H, 2 qw, 3 qh, 4 dst_off, 5 threshold (exposed luminance)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var b_glob = dsl.buffer[1](BIND_AAA_GLOBALS)
    var qx = dsl.global_id_x()
    var qy = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var qw = dsl.push_u32(2)
    var qh = dsl.push_u32(3)
    var dst = dsl.push_u32(4)
    var thr = max_f(dsl.push_f32(5), 0.0)
    dsl.return_if(qx.ge(qw) | qy.ge(qh))
    var expo = _total_exposure_dsl(dsl, b_glob, b_post)
    var acc = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    for j in range(4):
        for i in range(4):
            var fx = min_u(w_u - UInt32(1), qx * UInt32(4) + UInt32(i))
            var fy = min_u(h_u - UInt32(1), qy * UInt32(4) + UInt32(j))
            acc.set(acc.get() + b_hdr.load_vec3(b_hdr.base_word(fy * w_u + fx)))
    var col = acc.get() * Float32(1.0 / 16.0)
    var l = _luma(col) * expo
    var knee = max_f(thr * Float32(0.5), 1.0e-3)
    var soft = clamp_f(l - thr + knee, dsl.f32(0.0), knee * Float32(2.0))
    var contrib = max_f(soft * soft / (knee * Float32(4.0)), l - thr)
    var scale = contrib / max_f(l, 1.0e-4)
    b_post.store_vec3(dst + (qy * qw + qx) * UInt32(4), col * scale)
    return dsl.finish()


@always_inline
def _total_exposure_dsl(mut dsl: ShaderDsl, b_glob: BufferRef[1], b_post: BufferRef[1]) -> GpuF32:
    var base_exp = b_glob.load_f32(dsl.u32(47))
    var auto_on = b_glob.load_u32(dsl.u32(119)).ne(UInt32(0))
    var adapt = b_post.load_f32(dsl.u32(UInt32(SOTA_POST_ADAPTED_EXP_WORD)))
    return select_f(auto_on & adapt.gt(1.0e-5), base_exp * adapt, base_exp)


# ============================================================================
# 6. Ghosts + halo (quarter res)
# ============================================================================

def sota_flare_ghost_shader() -> List[UInt32]:
    """Push: 0 qw, 1 qh, 2 src_off, 3 dst_off, 4 n_ghosts, 5 dispersion, 6 halo_radius, 7 halo_strength,
    8 aspect (w/h), 9 coating_nm."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var qx = dsl.global_id_x()
    var qy = dsl.global_id_y()
    var qw = dsl.push_u32(0)
    var qh = dsl.push_u32(1)
    var src = dsl.push_u32(2)
    var dst = dsl.push_u32(3)
    var n_ghosts = dsl.push_u32(4)
    var disp = dsl.push_f32(5)
    var halo_r = dsl.push_f32(6)
    var halo_s = dsl.push_f32(7)
    var aspect = max_f(dsl.push_f32(8), 0.1)
    var coat = dsl.push_f32(9)
    dsl.return_if(qx.ge(qw) | qy.ge(qh))

    var fqw = qw.to_f32()
    var fqh = qh.to_f32()
    var u = (qx.to_f32() + Float32(0.5)) / fqw
    var v = (qy.to_f32() + Float32(0.5)) / fqh
    var pu = u - Float32(0.5)
    var pv = v - Float32(0.5)
    var acc = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    for gi in range(SOTA_GHOSTS_MAX):
        var m = ghost_magnification(gi)
        dsl.if_then(n_ghosts.ge(UInt32(gi + 1)))
        # Thin-film coating tint per ghost (Airy reflectance at 640 / 550 / 460 nm), see `ghost_tint_cpu`.
        var d_nm = coat * Float32(1.0 + 0.13 * Float32(gi))
        var tint = _thin_film_rgb_dsl(dsl, d_nm)
        var ch: List[Float32] = [-1.0, 0.0, 1.0]
        var rgb = List[GpuF32]()
        for k in range(3):
            var mk = dsl.f32(m) * (dsl.f32(1.0) + disp * ch[k] * (Float32(1.0) + Float32(0.3) * Float32(gi)))
            var su = pu * mk + Float32(0.5)
            var sv = pv * mk + Float32(0.5)
            var du = (su - Float32(0.5)) * Float32(2.0)
            var dv = (sv - Float32(0.5)) * Float32(2.0)
            var r2 = du * du + dv * dv
            var win = _saturate(dsl.f32(1.0) - r2)
            win = win * win
            var smp = _bilerp_post(dsl, b_post, src, UInt32(4), qw, qh, su * fqw - Float32(0.5), sv * fqh - Float32(0.5), UInt32(0))
            var comp = smp.x() if k == 0 else (smp.y() if k == 1 else smp.z())
            rgb.append(comp * win * mk * mk)
        acc.set(acc.get() + dsl.vec3(rgb[0] * tint.x(), rgb[1] * tint.y(), rgb[2] * tint.z()))
        dsl.end_if()

    # Halo: the bright source seen through the fixed ring of the aperture stop (aspect-correct circle).
    var ax = pu * aspect
    var len_p = max_f(sqrt_f(ax * ax + pv * pv), 1.0e-4)
    var hu = ax / len_p * halo_r / aspect + Float32(0.5)
    var hv = pv / len_p * halo_r + Float32(0.5)
    var hdu = (hu - Float32(0.5)) * Float32(2.0)
    var hdv = (hv - Float32(0.5)) * Float32(2.0)
    var h_win = _saturate(dsl.f32(1.0) - abs_f(len_p - halo_r) * Float32(5.0))
    var h_col = _bilerp_post(dsl, b_post, src, UInt32(4), qw, qh, hu * fqw - Float32(0.5), hv * fqh - Float32(0.5), UInt32(0))
    var h_edge = _saturate(dsl.f32(1.0) - (hdu * hdu + hdv * hdv) * Float32(0.25))
    acc.set(acc.get() + h_col * (h_win * h_edge * halo_s * Float32(0.15)) * dsl.vec3_const(0.85, 0.95, 1.1))
    b_post.store_vec3(dst + (qy * qw + qx) * UInt32(4), acc.get())
    return dsl.finish()


def _thin_film_rgb_dsl(mut dsl: ShaderDsl, d_nm: GpuF32) -> GpuVec3:
    var r01 = Float32((1.0 - 1.38) / (1.0 + 1.38))
    var r12 = Float32((1.38 - 1.52) / (1.38 + 1.52))
    var a = r01 * r01 + r12 * r12
    var b = Float32(2.0) * r01 * r12
    var e = Float32(1.0) + r01 * r01 * r12 * r12
    var out = List[GpuF32]()
    for lam in [Float32(640.0), Float32(550.0), Float32(460.0)]:
        var c1 = cos_f(d_nm * (Float32(4.0 * 3.14159265 * 1.38) / lam))
        var c2 = cos_f(d_nm * (Float32(1.07 * 4.0 * 3.14159265 * 1.38) / lam))
        var r1 = (dsl.f32(a) + c1 * b) / (dsl.f32(e) + c1 * b)
        var r2 = (dsl.f32(a) + c2 * b) / (dsl.f32(e) + c2 * b)
        out.append(r1 * r2)
    return dsl.vec3(out[0], out[1], out[2])


# ============================================================================
# 7. Aperture-shaped ghost blur (quarter res)
# ============================================================================

def sota_flare_blur_shader() -> List[UInt32]:
    """Push: 0 qw, 1 qh, 2 src_off, 3 dst_off, 4 radius (quarter px), 5 blades, 6 rotation, 7 curvature,
    8 flags (1 ghost blur, 2 dirt glow), 9 bright_off, 10 glow_off, 11 aspect (w/h).
    The dirt glow is a wide (12 % / 28 % of the frame height) average of the bright pass: the light
    that reaches the front element and scatters off the dirt on it."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var qx = dsl.global_id_x()
    var qy = dsl.global_id_y()
    var qw = dsl.push_u32(0)
    var qh = dsl.push_u32(1)
    var src = dsl.push_u32(2)
    var dst = dsl.push_u32(3)
    var rad = max_f(dsl.push_f32(4), 0.5)
    var blades = dsl.push_f32(5)
    var rot = dsl.push_f32(6)
    var curv = clamp_f(dsl.push_f32(7), 0.0, 1.0)
    var flags = dsl.push_u32(8)
    var bright_off = dsl.push_u32(9)
    var glow_off = dsl.push_u32(10)
    var aspect = max_f(dsl.push_f32(11), 0.1)
    dsl.return_if(qx.ge(qw) | qy.ge(qh))
    var fx = qx.to_f32()
    var fy = qy.to_f32()
    dsl.if_then((flags & UInt32(2)).ne(UInt32(0)))
    var glow = dsl.var_vec3(b_post.load_vec3(bright_off + (qy * qw + qx) * UInt32(4)))
    for k in range(8):
        var ang = Float32(k) * Float32(0.785398)
        var rad_uv = Float32(0.12) if (k & 1) == 0 else Float32(0.28)
        var gx = fx + rad_uv * cos(ang) / aspect * qw.to_f32()
        var gy = fy + rad_uv * sin(ang) * qh.to_f32()
        glow.set(glow.get() + _bilerp_post(dsl, b_post, bright_off, UInt32(4), qw, qh, gx, gy, UInt32(0)))
    b_post.store_vec3(glow_off + (qy * qw + qx) * UInt32(4), glow.get() * Float32(1.0 / 9.0))
    dsl.end_if()
    dsl.return_if((flags & UInt32(1)).eq(UInt32(0)))
    var acc = dsl.var_vec3(b_post.load_vec3(src + (qy * qw + qx) * UInt32(4)))
    var n = 1
    for ring in range(1, 3):
        for j in range(8 * ring):
            n += 1
            var theta = Float32(6.283185307) * (Float32(j) + Float32(0.5) * Float32(ring & 1)) / Float32(8 * ring)
            var poly = _poly_factor_dsl(dsl, theta, blades, rot, curv)
            var rr = rad * poly * (Float32(ring) / Float32(2.0))
            var s = _bilerp_post(dsl, b_post, src, UInt32(4), qw, qh, fx + rr * cos(theta), fy + rr * sin(theta), UInt32(0))
            acc.set(acc.get() + s)
    b_post.store_vec3(dst + (qy * qw + qx) * UInt32(4), acc.get() * Float32(1.0 / Float32(n)))
    return dsl.finish()


# ============================================================================
# 7b. Anamorphic streak (quarter res)
# ============================================================================

def sota_streak_shader() -> List[UInt32]:
    """The cylindrical front group of an anamorphic lens spreads every bright source into a long
    horizontal line. Push: 0 qw, 1 qh, 2 bright_off, 3 streak_off, 4 reach (quarter px).
    Exponential line kernel `exp(-4 |dx| / reach)`, normalised to unit sum, over the bright pass."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var qx = dsl.global_id_x()
    var qy = dsl.global_id_y()
    var qw = dsl.push_u32(0)
    var qh = dsl.push_u32(1)
    var bright_off = dsl.push_u32(2)
    var streak_off = dsl.push_u32(3)
    var reach = max_f(dsl.push_u32(4).to_f32(), 1.0)
    dsl.return_if(qx.ge(qw) | qy.ge(qh))
    var r_i = dsl.push_u32(4)
    var x_lo = select_u(qx.lt(r_i), dsl.u32(0), qx - r_i)
    var x_hi = min_u(qw - UInt32(1), qx + r_i)
    var acc = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var w_acc = dsl.var_f32_const(0.0)
    var row = bright_off + qy * qw * UInt32(4)
    var xi = dsl.var_u32(x_lo)
    var k = Float32(-4.0 * 1.442695) / reach  # exp(-4 d / reach) = exp2(k d)
    dsl.begin_loop()
    var x = xi.get()
    dsl.break_if(x.ge(x_hi + UInt32(1)))
    var d = abs_f(x.to_f32() - qx.to_f32())
    var wgt = exp2_f(k * d)
    acc.set(acc.get() + b_post.load_vec3(row + x * UInt32(4)) * wgt)
    w_acc.set(w_acc.get() + wgt)
    xi.set(x + UInt32(1))
    dsl.end_loop()
    # Normalise by the full (unclipped) kernel so the frame edge does not brighten the line.
    var w_full = dsl.f32(1.0) + (dsl.f32(1.0) - exp2_f(k * reach)) / max_f(dsl.f32(1.0) - exp2_f(k), 1.0e-6) * Float32(2.0) * exp2_f(k)
    b_post.store_vec3(streak_off + (qy * qw + qx) * UInt32(4), acc.get() / max_f(w_full, w_acc.get()))
    return dsl.finish()


# ============================================================================
# 8. Lens + sensor composite (full res, writes the HDR buffer)
# ============================================================================

def sota_lens_composite_shader() -> List[UInt32]:
    """Push: 0 W, 1 H, 2 anamorphic streak intensity, 3 flags (1 flare, 2 dirt, 8 vignette, 16 grain,
    32 anamorphic streak; lateral CA is its own earlier pass), 4 qw, 5 qh, 6 ghost_off, 7 glow_off (wide
    bright average), 8 flare_intensity, 9 dirt_intensity, 10 dirt texture offset (0: inline procedural
    fallback), 11 optical vignette strength, 12 grain photons per unit signal (Q), 13 read noise^2,
    14 grain strength, 15 frame counter."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var b_glob = dsl.buffer[1](BIND_AAA_GLOBALS)
    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var flags = dsl.push_u32(3)
    var qw = dsl.push_u32(4)
    var qh = dsl.push_u32(5)
    var ghost_off = dsl.push_u32(6)
    var bright_off = dsl.push_u32(7)
    var flare_i = dsl.push_f32(8)
    var dirt_i = dsl.push_f32(9)
    var dirt_off = dsl.push_u32(10)
    var streak_i = dsl.push_f32(2)
    var vig = dsl.push_f32(11)
    var grain_q = max_f(dsl.push_f32(12), 1.0)
    var read2 = dsl.push_f32(13)
    var grain_s = dsl.push_f32(14)
    var frame = dsl.push_u32(15)
    dsl.return_if(px.ge(w_u) | py.ge(h_u))

    var fw = w_u.to_f32()
    var fh = h_u.to_f32()
    var u = (px.to_f32() + Float32(0.5)) / fw
    var v = (py.to_f32() + Float32(0.5)) / fh
    var base = b_hdr.base_word(py * w_u + px)
    var col = dsl.var_vec3(b_hdr.load_vec3(base))

    # Optical (cat's-eye) vignetting, strength already scaled by the f-stop on the CPU.
    dsl.if_then((flags & UInt32(8)).ne(UInt32(0)))
    var aspect = fw / fh
    var nx = (u * Float32(2.0) - Float32(1.0)) * aspect
    var ny = v * Float32(2.0) - Float32(1.0)
    var rr = sqrt_f(nx * nx + ny * ny) / sqrt_f(aspect * aspect + Float32(1.0))
    var t = _saturate((rr - Float32(0.45)) * Float32(1.0 / 0.55))
    var sm = t * t * (dsl.f32(3.0) - t * Float32(2.0))
    col.set(col.get() * (dsl.f32(1.0) - vig * Float32(0.6) * sm))
    dsl.end_if()

    var qx_f = u * qw.to_f32() - Float32(0.5)
    var qy_f = v * qh.to_f32() - Float32(0.5)
    dsl.if_then((flags & UInt32(1)).ne(UInt32(0)))
    var gh = _bilerp_post(dsl, b_post, ghost_off, UInt32(4), qw, qh, qx_f, qy_f, UInt32(0))
    col.set(col.get() + gh * flare_i)
    dsl.end_if()

    # Anamorphic streak (`sota_streak_shader`, quarter res right after the glow buffer: push 6 is the blurred
    # ghost buffer, so the streak sits 8 qw qh words after it), tinted by the coatings.
    dsl.if_then((flags & UInt32(32)).ne(UInt32(0)))
    var stk = _bilerp_post(dsl, b_post, ghost_off + qw * qh * UInt32(8), UInt32(4), qw, qh, qx_f, qy_f, UInt32(0))
    col.set(col.get() + stk * dsl.vec3_const(0.45, 0.62, 1.0) * streak_i)
    dsl.end_if()

    # Lens dirt on the front element, lit by the wide-angle bright energy: the procedural dirt texture
    # (`sota_camera_dirt.mojo`) when uploaded, else inline smudges + specks.
    dsl.if_then((flags & UInt32(2)).ne(UInt32(0)))
    var mask = dsl.var_f32_const(0.0)
    dsl.if_then(dirt_off.ne(UInt32(0)))
    var dtx = clamp_f(u * Float32(SOTA_DIRT_W) - Float32(0.5), dsl.f32(0.0), dsl.f32(Float32(SOTA_DIRT_W - 1)))
    var dty = clamp_f(v * Float32(SOTA_DIRT_H) - Float32(0.5), dsl.f32(0.0), dsl.f32(Float32(SOTA_DIRT_H - 1)))
    var dx0 = min_u(dsl.u32(UInt32(SOTA_DIRT_W - 2)), floor_f(dtx).saturate_u32())
    var dy0 = min_u(dsl.u32(UInt32(SOTA_DIRT_H - 2)), floor_f(dty).saturate_u32())
    var dfx = clamp_f(dtx - dx0.to_f32(), 0.0, 1.0)
    var dfy = clamp_f(dty - dy0.to_f32(), 0.0, 1.0)
    var taps = List[GpuF32]()
    for j in range(2):
        for i in range(2):
            var ti = (dy0 + UInt32(j)) * UInt32(SOTA_DIRT_W) + dx0 + UInt32(i)
            var wd = b_post.load_u32(dirt_off + (ti >> UInt32(2)))
            taps.append(((wd >> ((ti & UInt32(3)) * UInt32(8))) & UInt32(255)).to_f32() * Float32(1.0 / 255.0))
    var one_d = dsl.f32(1.0)
    mask.set((taps[0] * (one_d - dfx) + taps[1] * dfx) * (one_d - dfy) + (taps[2] * (one_d - dfx) + taps[3] * dfx) * dfy)
    dsl.else_branch()
    for layer in range(2):
        var gs = Float32(7.0) if layer == 0 else Float32(19.0)
        var su = u * gs * (fw / fh)
        var sv = v * gs
        var cx = floor_f(su)
        var cy = floor_f(sv)
        var hseed = _pcg(cx.saturate_u32() * UInt32(73856093) ^ cy.saturate_u32() * UInt32(19349663) ^ UInt32(83492791 * (layer + 1)))
        var h1 = _pcg(hseed)
        var h2 = _pcg(h1)
        var ox = _u01(hseed) * Float32(0.6) + Float32(0.2)
        var oy = _u01(h1) * Float32(0.6) + Float32(0.2)
        var dx = su - cx - ox
        var dy = sv - cy - oy
        var dist = sqrt_f(dx * dx + dy * dy)
        var rad = Float32(0.38) if layer == 0 else Float32(0.12)
        var present = _u01(h2).lt(Float32(0.55) if layer == 0 else Float32(0.35))
        var blob = _saturate(dsl.f32(1.0) - dist / rad)
        blob = blob * blob * (dsl.f32(0.6) + _u01(h2) * Float32(0.8))
        mask.set(mask.get() + select_f(present, blob, dsl.f32(0.0)))
    dsl.end_if()
    var glow = _bilerp_post(dsl, b_post, bright_off, UInt32(4), qw, qh, qx_f, qy_f, UInt32(0))
    col.set(col.get() + glow * (mask.get() * dirt_i))
    dsl.end_if()

    # Sensor noise: Poisson shot + read noise for this ISO, in exposure-normalised signal units.
    dsl.if_then((flags & UInt32(16)).ne(UInt32(0)))
    var expo = max_f(_total_exposure_dsl(dsl, b_glob, b_post), 1.0e-6)
    var seed = _pcg(px * UInt32(1973) ^ py * UInt32(9277) ^ frame * UInt32(26699))
    var cur = col.get()
    var outc = List[GpuF32]()
    for k in range(3):
        var h_a = _pcg(seed ^ UInt32(0x9E3779B9 ^ (7919 * (k + 1))))
        var h_b = _pcg(h_a)
        var h_c = _pcg(h_b)
        var h_d = _pcg(h_c)
        var gauss = (_u01(h_a) + _u01(h_b) + _u01(h_c) + _u01(h_d) - Float32(2.0)) * Float32(1.7320508)
        var comp = cur.x() if k == 0 else (cur.y() if k == 1 else cur.z())
        var sig = max_f(comp * expo, 0.0)
        var sigma = sqrt_f(sig * grain_q + read2) / grain_q
        outc.append(max_f(comp + gauss * sigma * grain_s / expo, 0.0))
    col.set(dsl.vec3(outc[0], outc[1], outc[2]))
    dsl.end_if()

    b_hdr.store_vec3(base, col.get())
    return dsl.finish()


# ============================================================================
# 9. Lateral chromatic aberration (full res, before bloom / motion blur)
# ============================================================================

def _hdr_channel_bilerp(mut dsl: ShaderDsl, b_hdr: BufferRef[8], w: GpuU32, h: GpuU32, fx_in: GpuF32, fy_in: GpuF32, ch: UInt32) -> GpuF32:
    var fx = clamp_f(fx_in, dsl.f32(0.0), w.to_f32() - Float32(1.0))
    var fy = clamp_f(fy_in, dsl.f32(0.0), h.to_f32() - Float32(1.0))
    var x0 = min_u(w - UInt32(1), floor_f(fx).saturate_u32())
    var y0 = min_u(h - UInt32(1), floor_f(fy).saturate_u32())
    var x1 = min_u(w - UInt32(1), x0 + UInt32(1))
    var y1 = min_u(h - UInt32(1), y0 + UInt32(1))
    var tx = clamp_f(fx - x0.to_f32(), 0.0, 1.0)
    var ty = clamp_f(fy - y0.to_f32(), 0.0, 1.0)
    var c00 = b_hdr.load_f32(b_hdr.base_word(y0 * w + x0) + ch)
    var c10 = b_hdr.load_f32(b_hdr.base_word(y0 * w + x1) + ch)
    var c01 = b_hdr.load_f32(b_hdr.base_word(y1 * w + x0) + ch)
    var c11 = b_hdr.load_f32(b_hdr.base_word(y1 * w + x1) + ch)
    var one = dsl.f32(1.0)
    return (c00 * (one - tx) + c10 * tx) * (one - ty) + (c01 * (one - tx) + c11 * tx) * ty


def sota_ca_shader() -> List[UInt32]:
    """Push: 0 W, 1 H, 2 scratch_off (4 words/px), 3 ca_strength, 4 mode (0: resample HDR -> scratch with
    per-channel magnification about the optical centre, R magnified / B minified; 1: scratch rgb -> HDR)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](BIND_AAA_HDR_OUT)
    var b_post = dsl.buffer[1](BIND_AAA_POST_SCRATCH)
    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var off = dsl.push_u32(2)
    var ca = dsl.push_f32(3)
    var mode = dsl.push_u32(4)
    dsl.return_if(px.ge(w_u) | py.ge(h_u))
    var base = b_hdr.base_word(py * w_u + px)
    var so = off + (py * w_u + px) * UInt32(4)
    dsl.if_then(mode.eq(UInt32(0)))
    var fw = w_u.to_f32()
    var fh = h_u.to_f32()
    var u = (px.to_f32() + Float32(0.5)) / fw - Float32(0.5)
    var v = (py.to_f32() + Float32(0.5)) / fh - Float32(0.5)
    var kr = dsl.f32(1.0) + ca
    var kb = dsl.f32(1.0) - ca
    var r = _hdr_channel_bilerp(dsl, b_hdr, w_u, h_u, (u * kr + Float32(0.5)) * fw - Float32(0.5), (v * kr + Float32(0.5)) * fh - Float32(0.5), UInt32(0))
    var g = b_hdr.load_f32(base + UInt32(1))
    var b = _hdr_channel_bilerp(dsl, b_hdr, w_u, h_u, (u * kb + Float32(0.5)) * fw - Float32(0.5), (v * kb + Float32(0.5)) * fh - Float32(0.5), UInt32(2))
    b_post.store_vec3(so, dsl.vec3(r, g, b))
    dsl.else_branch()
    b_hdr.store_vec3(base, b_post.load_vec3(so))
    dsl.end_if()
    return dsl.finish()


# ============================================================================
# Frame recording (called from the sota:camera hooks in aaa_renderer.mojo)
# ============================================================================

def _barrier(mut vk: VulkanGpu, slot: Int):
    vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))


def _push(n: Int = 16) -> List[UInt32]:
    return List[UInt32](length=n, fill=0)


@always_inline
def _fb(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def sota_camera_record_dof(
    mut post: SotaCameraPost,
    mut vk: VulkanGpu,
    slot: Int,
    pipe_layout: Int,
    desc_set: Int,
    out_w: Int,
    out_h: Int,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
):
    """Records the 4 DoF dispatches (HDR buffer in, HDR buffer out)."""
    var q = next_q
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q, PIPELINE_BOTTOM)  # BOTTOM: start after prior work drains -> true pass duration
        next_q += 2
        prof_pairs.append((String("SotaBokehDoF"), q))
    var lay = SotaDofLayout(out_w, out_h)
    var hw = lay.hw
    var hh = lay.hh
    var cam = post.camera
    var max_coc = max(Float32(1.0), min(Float32(32.0), post.dof_max_radius_px))
    var k_inf = cam.coc_infinity_radius_px(out_h)
    var focus = cam.focus_distance_m
    var blades = Float32(cam.aperture_blades) if cam.aperture_blades >= 3 else Float32(64.0)
    var curv = cam.blade_curvature if cam.aperture_blades >= 3 else Float32(1.0)

    var p0 = _push()
    p0[0] = UInt32(out_w); p0[1] = UInt32(out_h); p0[2] = UInt32(hw); p0[3] = UInt32(hh)
    p0[4] = UInt32(lay.setup_off); p0[5] = _fb(k_inf); p0[6] = _fb(focus); p0[7] = _fb(max_coc)
    vk.dispatch_aaa(slot, post.pipe_dof_setup, pipe_layout, desc_set, p0, (hw + 7) // 8, (hh + 7) // 8)
    _barrier(vk, slot)

    var pd = _push()
    pd[0] = UInt32(hw); pd[1] = UInt32(hh); pd[2] = UInt32(lay.setup_off); pd[3] = UInt32(lay.qw); pd[4] = UInt32(lay.qh)
    pd[5] = UInt32(lay.level1_off)
    vk.dispatch_aaa(slot, post.pipe_dof_downsample, pipe_layout, desc_set, pd, (lay.qw + 7) // 8, (lay.qh + 7) // 8)
    var p1 = _push()
    p1[0] = UInt32(hw); p1[1] = UInt32(hh); p1[2] = UInt32(lay.setup_off); p1[3] = UInt32(lay.tx); p1[4] = UInt32(lay.ty)
    p1[5] = UInt32(lay.tile_off)
    vk.dispatch_aaa(slot, post.pipe_dof_tile, pipe_layout, desc_set, p1, (lay.tx + 7) // 8, (lay.ty + 7) // 8)
    _barrier(vk, slot)

    var p2 = _push()
    p2[0] = UInt32(hw); p2[1] = UInt32(hh); p2[2] = UInt32(lay.setup_off); p2[3] = UInt32(lay.tile_off)
    p2[4] = UInt32(lay.tx); p2[5] = UInt32(lay.ty); p2[6] = UInt32(lay.gather_off); p2[7] = _fb(max_coc * Float32(0.5))
    p2[8] = _fb(blades); p2[9] = _fb(cam.blade_rotation_deg * Float32(0.017453293)); p2[10] = _fb(post.dof_cat_eye); p2[11] = _fb(curv)
    p2[12] = UInt32(lay.level1_off); p2[13] = UInt32(lay.qw); p2[14] = UInt32(lay.qh)
    vk.dispatch_aaa(slot, post.pipe_dof_gather, pipe_layout, desc_set, p2, (hw + 7) // 8, (hh + 7) // 8)
    _barrier(vk, slot)

    var pf = _push()
    pf[0] = UInt32(hw); pf[1] = UInt32(hh); pf[2] = UInt32(lay.gather_off); pf[3] = UInt32(lay.filtered_off)
    vk.dispatch_aaa(slot, post.pipe_dof_postfilter, pipe_layout, desc_set, pf, (hw + 7) // 8, (hh + 7) // 8)
    _barrier(vk, slot)

    var p3 = _push()
    p3[0] = UInt32(out_w); p3[1] = UInt32(out_h); p3[2] = UInt32(hw); p3[3] = UInt32(hh)
    p3[4] = UInt32(lay.filtered_off); p3[5] = _fb(k_inf); p3[6] = _fb(focus); p3[7] = _fb(max_coc)
    vk.dispatch_aaa(slot, post.pipe_dof_composite, pipe_layout, desc_set, p3, (out_w + 7) // 8, (out_h + 7) // 8)
    _barrier(vk, slot)
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q + 1, PIPELINE_BOTTOM)


def sota_camera_record_lens(
    mut post: SotaCameraPost,
    mut vk: VulkanGpu,
    slot: Int,
    pipe_layout: Int,
    desc_set: Int,
    buf_hdr: GpuBuffer,
    buf_post: GpuBuffer,
    out_w: Int,
    out_h: Int,
    max_pix: Int,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
):
    """Records the flare chain (if needed) and the lens + sensor composite."""
    var q = next_q
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q, PIPELINE_BOTTOM)  # BOTTOM: start after prior work drains -> true pass duration
        next_q += 2
        prof_pairs.append((String("SotaLensSensor"), q))
    var lay = sota_lens_layout(out_w, out_h, max_pix)
    var qw = lay[0]
    var qh = lay[1]
    var cam = post.camera
    var need_bright = post.enable_flare or post.enable_dirt or post.enable_anamorphic_streak
    # Procedural dirt texture: generated once per seed, uploaded once per buffer / offset (it lives in a
    # persistent post-scratch region nothing else writes, so later frames just read it).
    var dirt_off = 0
    if post.enable_dirt and post.dirt_use_texture:
        dirt_off = sota_dirt_offset(out_w, out_h, max_pix)
        if dirt_off != 0:
            if len(post.dirt_words) == 0 or post.dirt_words_seed != post.dirt_seed:
                post.dirt_words = sota_pack_dirt_u8(sota_lens_dirt_texture_cpu(post.dirt_seed))
                post.dirt_words_seed = post.dirt_seed
                post.dirt_uploaded_off = 0
            if post.dirt_uploaded_off != dirt_off or post.dirt_uploaded_addr != buf_post.cpu_address:
                if sota_upload_dirt(post.dirt_words, buf_post, dirt_off):
                    post.dirt_uploaded_off = dirt_off
                    post.dirt_uploaded_addr = buf_post.cpu_address
                else:
                    dirt_off = 0
    _barrier(vk, slot)
    if need_bright:
        var pb = _push()
        pb[0] = UInt32(out_w); pb[1] = UInt32(out_h); pb[2] = UInt32(qw); pb[3] = UInt32(qh)
        pb[4] = UInt32(lay[3]); pb[5] = _fb(post.flare_threshold)
        vk.dispatch_aaa(slot, post.pipe_flare_bright, pipe_layout, desc_set, pb, (qw + 7) // 8, (qh + 7) // 8)
        _barrier(vk, slot)
    if post.enable_flare:
        var pg = _push()
        pg[0] = UInt32(qw); pg[1] = UInt32(qh); pg[2] = UInt32(lay[3]); pg[3] = UInt32(lay[4])
        pg[4] = UInt32(max(0, min(SOTA_GHOSTS_MAX, post.flare_ghosts))); pg[5] = _fb(post.flare_dispersion)
        pg[6] = _fb(post.flare_halo_radius); pg[7] = _fb(post.flare_halo_strength)
        pg[8] = _fb(Float32(out_w) / Float32(max(1, out_h))); pg[9] = _fb(post.coating_thickness_nm)
        vk.dispatch_aaa(slot, post.pipe_flare_ghost, pipe_layout, desc_set, pg, (qw + 7) // 8, (qh + 7) // 8)
        _barrier(vk, slot)
    if need_bright:
        # Ghost blur radius ~ the aperture image size: wider apertures -> bigger ghosts.
        var blades = Float32(cam.aperture_blades) if cam.aperture_blades >= 3 else Float32(64.0)
        var rad_q = max(Float32(1.0), min(Float32(6.0), Float32(8.0) / cam.fstop * Float32(qh) / Float32(180.0)))
        var pz = _push()
        pz[0] = UInt32(qw); pz[1] = UInt32(qh); pz[2] = UInt32(lay[4]); pz[3] = UInt32(lay[5])
        pz[4] = _fb(rad_q); pz[5] = _fb(blades); pz[6] = _fb(cam.blade_rotation_deg * Float32(0.017453293))
        pz[7] = _fb(cam.blade_curvature if cam.aperture_blades >= 3 else Float32(1.0))
        pz[8] = (UInt32(1) if post.enable_flare else UInt32(0)) | (UInt32(2) if post.enable_dirt else UInt32(0))
        pz[9] = UInt32(lay[3]); pz[10] = UInt32(lay[6]); pz[11] = _fb(Float32(out_w) / Float32(max(1, out_h)))
        vk.dispatch_aaa(slot, post.pipe_flare_blur, pipe_layout, desc_set, pz, (qw + 7) // 8, (qh + 7) // 8)
        _barrier(vk, slot)
    if post.enable_anamorphic_streak:
        var ps = _push()
        ps[0] = UInt32(qw); ps[1] = UInt32(qh); ps[2] = UInt32(lay[3]); ps[3] = UInt32(lay[6] + qw * qh * 4)
        ps[4] = UInt32(max(2, Int(Float32(qw) * max(Float32(0.02), min(Float32(1.0), post.anamorphic_streak_length)))))
        vk.dispatch_aaa(slot, post.pipe_streak, pipe_layout, desc_set, ps, (qw + 7) // 8, (qh + 7) // 8)
        _barrier(vk, slot)
    var flags = UInt32(0)
    if post.enable_flare:
        flags |= 1
    if post.enable_dirt:
        flags |= 2
    if post.enable_optical_vignette:
        flags |= 8
    if post.enable_grain:
        flags |= 16
    if post.enable_anamorphic_streak:
        flags |= 32
    var q_ph = cam.full_well_photons * Float32(100.0) / cam.iso
    var pl = _push()
    pl[0] = UInt32(out_w); pl[1] = UInt32(out_h); pl[2] = _fb(post.anamorphic_streak_intensity); pl[3] = flags
    pl[4] = UInt32(qw); pl[5] = UInt32(qh); pl[6] = UInt32(lay[5]); pl[7] = UInt32(lay[6])
    pl[8] = _fb(post.flare_intensity); pl[9] = _fb(post.dirt_intensity); pl[10] = UInt32(dirt_off)
    pl[11] = _fb(cam.optical_vignette_strength() * post.optical_vignette_amount)
    pl[12] = _fb(q_ph); pl[13] = _fb(cam.read_noise_photons * cam.read_noise_photons); pl[14] = _fb(post.grain_strength)
    pl[15] = post.frame_counter
    post.frame_counter += 1
    vk.dispatch_aaa(slot, post.pipe_lens_composite, pipe_layout, desc_set, pl, (out_w + 7) // 8, (out_h + 7) // 8)
    _barrier(vk, slot)
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q + 1, PIPELINE_BOTTOM)


def sota_camera_record_ca(
    mut post: SotaCameraPost,
    mut vk: VulkanGpu,
    slot: Int,
    pipe_layout: Int,
    desc_set: Int,
    out_w: Int,
    out_h: Int,
    max_pix: Int,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
):
    """Lateral chromatic aberration on the scene-referred image, right after DoF: the lens images each
    wavelength at its own magnification before any later scattering (bloom, glare) or the temporal
    integration of the shutter (motion blur), so those must see the already split image. Copies the HDR
    frame: resample into scratch (rgb), then write rgb back (no full 8-word HDR copy)."""
    var q = next_q
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q, PIPELINE_BOTTOM)
        next_q += 2
        prof_pairs.append((String("SotaChromaticAberration"), q))
    var lay = sota_lens_layout(out_w, out_h, max_pix)
    _barrier(vk, slot)
    for mode in range(2):
        var pc = _push()
        pc[0] = UInt32(out_w); pc[1] = UInt32(out_h); pc[2] = UInt32(lay[2]); pc[3] = _fb(post.ca_strength); pc[4] = UInt32(mode)
        vk.dispatch_aaa(slot, post.pipe_ca, pipe_layout, desc_set, pc, (out_w + 7) // 8, (out_h + 7) // 8)
        _barrier(vk, slot)
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q + 1, PIPELINE_BOTTOM)


def sota_camera_record_vrs_rate(
    mut post: SotaCameraPost,
    mut vk: VulkanGpu,
    slot: Int,
    pipe_layout: Int,
    desc_set: Int,
    out_w: Int,
    out_h: Int,
    render_w: Int,
    render_h: Int,
    max_pix: Int,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
):
    """End of the post chain (before lens noise): the rate map the next frame's shading uses."""
    if not (post.enable_vrs and post.vrs_pipelines_ready()):
        post.vrs_rate_w = 0
        post.vrs_rate_h = 0
        return
    sota_vrs_record_rate(
        vk, slot, post.pipe_vrs_rate, pipe_layout, desc_set, out_w, out_h, render_w, render_h, max_pix,
        post.vrs_threshold, post.vrs_motion_px, post.vrs_allow_4x4, use_prof, next_q, q_limit, prof_pairs,
    )
    post.vrs_rate_w = render_w
    post.vrs_rate_h = render_h


def sota_camera_record_vrs_classify(
    mut post: SotaCameraPost,
    mut vk: VulkanGpu,
    slot: Int,
    pipe_layout: Int,
    desc_set: Int,
    render_w: Int,
    render_h: Int,
    max_pix: Int,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
) -> Int:
    """Before shading: returns the shade-list offset for the shading pass (0 = VRS off this frame)."""
    if not post.vrs_active(render_w, render_h):
        return 0
    return sota_vrs_record_classify(
        vk, slot, post.pipe_vrs_classify, pipe_layout, desc_set, render_w, render_h, max_pix, use_prof, next_q, q_limit, prof_pairs
    )


def sota_camera_record_vrs_fill(
    mut post: SotaCameraPost,
    mut vk: VulkanGpu,
    slot: Int,
    pipe_layout: Int,
    desc_set: Int,
    render_w: Int,
    render_h: Int,
    max_pix: Int,
    demod_gi: Bool,
    demod_spec: Bool,
    spec_off: Int,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
):
    sota_vrs_record_fill(
        vk, slot, post.pipe_vrs_fill, pipe_layout, desc_set, render_w, render_h, max_pix, demod_gi, demod_spec, spec_off,
        use_prof, next_q, q_limit, prof_pairs,
    )
