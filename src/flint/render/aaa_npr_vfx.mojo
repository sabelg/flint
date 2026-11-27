"""Stylized Anime Mesh VFX & Screen-Space Impact/Shockwave Post-Processing (`aaa_npr_vfx.mojo`).

Implements two complementary anime/stylized visual-effects stages in the unified Vulkan AAA pipeline:
  1. **Eroding UV-Scrolling 3D Stylized Mesh VFX (`AaaStylizedVfxParams`)**:
     Drives opaque/alpha-cutout 3D ribbons, fire pillars, arcs, and energy bursts with dual-layer
     UV scrolling, FBM domain warping, sharp threshold alpha-erosion (`erosion` in `[0..1]`),
     3-band `core -> edge` stepped HDR ramp, and an outer hand-drawn dark anime ink border (`ink_rim`).
     Evaluated identically in `VisBuffer` alpha-cutout (`OpKill` silhouette holes) and NPR shading!
  2. **Screen-Space Chromatic-Dispersion Shockwaves & Manga Impact Frames (`aaa_npr_post_fx_shader`)**:
     Evaluates up to 4 simultaneous radial shockwave rings with independent R/G/B chromatic refraction
     plus optional high-contrast radial-streak anime impact frames (`AaaImpactFrameParams`).
"""
from std.math import abs, atan2, floor, isfinite, max, min, sin, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.render.spirv import (
    AaaSpirvModule,
    MODEL_FRAGMENT,
    MODEL_GL_COMPUTE,
    OP_LOAD,
    OP_STORE,
    STORAGE_INPUT,
    STORAGE_OUTPUT,
    emit,
)
from flint.render.aaa_visbuffer import (
    emit_alpha_cutoff,
    AAA_MAX_BINDLESS,
    AAA_PUSH_WORDS,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_POST_SCRATCH,
)

# `BIND_AAA_GLOBALS` companion layout (`641 .. 687` and `2240 .. 2751`)
comptime AAA_GLOBALS_VFX_COUNT_WORD: Int = 641
comptime AAA_GLOBALS_SHOCKWAVE_COUNT_WORD: Int = 642
comptime AAA_GLOBALS_IMPACT_PAPER_GAIN_WORD: Int = 643
comptime AAA_GLOBALS_IMPACT_FRAME_WORD: Int = 644   # 12 words: [644 .. 655]
comptime AAA_IMPACT_FRAME_WORDS: Int = 12

comptime AAA_GLOBALS_SHOCKWAVE_TABLE_WORD: Int = 656  # 4 * 8 = 32 words: [656 .. 687]
comptime AAA_MAX_SHOCKWAVES: Int = 4
comptime AAA_SHOCKWAVE_WORDS: Int = 8

comptime AAA_GLOBALS_VFX_TABLE_WORD: Int = 2496     # 32 * 16 = 512 words: [2496 .. 3007]
comptime AAA_MAX_VFX_MATERIALS: Int = 32
comptime AAA_VFX_ENTRY_WORDS: Int = 16


struct AaaStylizedVfxParams(Copyable, Movable):
    """Parameters for an alpha-eroded, UV-scrolling 3D anime VFX mesh (`material_id`)."""
    var material_id: Int
    var noise_tex: Int              # -1 = deterministic GPU 2-octave FBM noise; >=0 = bindless texture slot
    var uv_scroll_u: Float32        # Primary UV scroll speed X (cycles / sec)
    var uv_scroll_v: Float32        # Primary UV scroll speed Y (cycles / sec)
    var warp_scale: Float32         # Noise UV frequency scale (e.g. 4.0)
    var warp_strength: Float32      # Domain-warp amplitude applied before erosion test (0.0 .. 1.0)
    var erosion: Float32            # Dissolve cutoff in [0.0 .. 1.0]: 0 = full silhouette, 1 = fully eroded
    var edge_band_width: Float32    # Width above erosion cutoff painted with `edge_color` before `core_color`
    var core_color: Vec3            # Innermost hot HDR emissive color
    var edge_color: Vec3            # Mid/outer stepped HDR color band
    var ink_rim_width: Float32      # Narrow outer ink border width right at the erosion cutoff
    var ink_rim_darkness: Float32   # 0.0 = no dark ink border .. 1.0 = crisp charcoal anime ink edge

    def __init__(
        out self,
        material_id: Int = 0,
        uv_scroll_u: Float32 = 0.0,
        uv_scroll_v: Float32 = -2.4,
        warp_scale: Float32 = 4.5,
        warp_strength: Float32 = 0.28,
        erosion: Float32 = 0.35,
        edge_band_width: Float32 = 0.20,
        core_color: Vec3 = Vec3(7.5, 5.8, 2.4),
        edge_color: Vec3 = Vec3(3.8, 0.32, 0.05),
        ink_rim_width: Float32 = 0.045,
        ink_rim_darkness: Float32 = 0.88,
        noise_tex: Int = -1,
    ):
        self.material_id = material_id
        self.noise_tex = noise_tex
        self.uv_scroll_u = uv_scroll_u
        self.uv_scroll_v = uv_scroll_v
        self.warp_scale = max(Float32(0.25), warp_scale)
        self.warp_strength = max(Float32(0.0), min(Float32(2.0), warp_strength))
        self.erosion = max(Float32(0.0), min(Float32(1.0), erosion))
        self.edge_band_width = max(Float32(0.005), min(Float32(0.85), edge_band_width))
        self.core_color = core_color
        self.edge_color = edge_color
        self.ink_rim_width = max(Float32(0.0), min(Float32(0.35), ink_rim_width))
        self.ink_rim_darkness = max(Float32(0.0), min(Float32(1.0), ink_rim_darkness))

    @staticmethod
    def anime_flame(
        core_color: Vec3 = Vec3(8.5, 6.2, 2.2),
        edge_color: Vec3 = Vec3(4.2, 0.30, 0.04),
        erosion: Float32 = 0.34,
        scroll_speed: Float32 = 3.2,
    ) -> Self:
        return Self(
            material_id=0,
            uv_scroll_u=0.35,
            uv_scroll_v=-scroll_speed,
            warp_scale=5.0,
            warp_strength=0.36,
            erosion=erosion,
            edge_band_width=0.22,
            core_color=core_color,
            edge_color=edge_color,
            ink_rim_width=0.055,
            ink_rim_darkness=0.90,
        )

    @staticmethod
    def slash_arc(
        core_color: Vec3 = Vec3(9.0, 8.2, 6.5),
        edge_color: Vec3 = Vec3(4.5, 0.45, 0.12),
        erosion: Float32 = 0.28,
        scroll_speed: Float32 = 5.5,
    ) -> Self:
        return Self(
            material_id=0,
            uv_scroll_u=-scroll_speed,
            uv_scroll_v=0.25,
            warp_scale=6.5,
            warp_strength=0.22,
            erosion=erosion,
            edge_band_width=0.18,
            core_color=core_color,
            edge_color=edge_color,
            ink_rim_width=0.040,
            ink_rim_darkness=0.85,
        )

    @staticmethod
    def lightning_arc(
        core_color: Vec3 = Vec3(7.5, 9.2, 12.0),
        edge_color: Vec3 = Vec3(0.25, 1.85, 5.8),
        erosion: Float32 = 0.46,
        scroll_speed: Float32 = 7.5,
    ) -> Self:
        return Self(
            material_id=0,
            uv_scroll_u=0.8,
            uv_scroll_v=-scroll_speed,
            warp_scale=8.0,
            warp_strength=0.42,
            erosion=erosion,
            edge_band_width=0.15,
            core_color=core_color,
            edge_color=edge_color,
            ink_rim_width=0.035,
            ink_rim_darkness=0.78,
        )

    def append_words(self, mut out: List[UInt32]):
        var n_slot = UInt32(self.noise_tex & 0xFF) if (self.noise_tex >= 0 and self.noise_tex < 255) else UInt32(0xFF)
        out.append(UInt32(self.material_id & 0xFFFF))
        out.append(n_slot)
        out.append(bitcast[DType.uint32](self.uv_scroll_u))
        out.append(bitcast[DType.uint32](self.uv_scroll_v))
        out.append(bitcast[DType.uint32](self.warp_scale))
        out.append(bitcast[DType.uint32](self.warp_strength))
        out.append(bitcast[DType.uint32](self.erosion))
        out.append(bitcast[DType.uint32](self.edge_band_width))
        out.append(bitcast[DType.uint32](self.core_color.x))
        out.append(bitcast[DType.uint32](self.core_color.y))
        out.append(bitcast[DType.uint32](self.core_color.z))
        out.append(bitcast[DType.uint32](self.edge_color.x))
        out.append(bitcast[DType.uint32](self.edge_color.y))
        out.append(bitcast[DType.uint32](self.edge_color.z))
        out.append(bitcast[DType.uint32](self.ink_rim_width))
        out.append(bitcast[DType.uint32](self.ink_rim_darkness))


struct AaaShockwaveRing(Copyable, Movable):
    """Screen-space radial refractive shockwave with R/G/B chromatic dispersion."""
    var center_u: Float32               # Screen UV X in [0..1]
    var center_v: Float32               # Screen UV Y in [0..1]
    var radius_uv: Float32              # Ring radius in aspect-corrected height units
    var thickness_uv: Float32           # Ring half-width in aspect-corrected height units
    var refract_strength: Float32       # Radial UV refraction amplitude (e.g. 0.035)
    var chromatic_dispersion: Float32   # R/B split fraction in [0..1] (e.g. 0.55)
    var glow_intensity: Float32         # Additive HDR ring flash intensity
    var glow_color: Vec3                # Linear RGB tint of the ring flash (clamped to [0..1] chroma)

    def __init__(
        out self,
        center_u: Float32 = 0.5,
        center_v: Float32 = 0.5,
        radius_uv: Float32 = 0.18,
        thickness_uv: Float32 = 0.045,
        refract_strength: Float32 = 0.032,
        chromatic_dispersion: Float32 = 0.55,
        glow_intensity: Float32 = 1.2,
        glow_color: Vec3 = Vec3(1.0, 0.22, 0.30),
    ):
        self.center_u = center_u
        self.center_v = center_v
        self.radius_uv = max(Float32(0.002), radius_uv)
        self.thickness_uv = max(Float32(0.004), thickness_uv)
        self.refract_strength = max(Float32(-0.20), min(Float32(0.20), refract_strength))
        self.chromatic_dispersion = max(Float32(0.0), min(Float32(1.0), chromatic_dispersion))
        self.glow_intensity = max(Float32(0.0), glow_intensity)
        self.glow_color = Vec3(
            max(Float32(0.0), min(Float32(1.0), glow_color.x)),
            max(Float32(0.0), min(Float32(1.0), glow_color.y)),
            max(Float32(0.0), min(Float32(1.0), glow_color.z)),
        )

    def append_words(self, mut out: List[UInt32]):
        var r8 = UInt32(min(255, max(0, Int(self.glow_color.x * Float32(255.0) + Float32(0.5)))))
        var g8 = UInt32(min(255, max(0, Int(self.glow_color.y * Float32(255.0) + Float32(0.5)))))
        var b8 = UInt32(min(255, max(0, Int(self.glow_color.z * Float32(255.0) + Float32(0.5)))))
        var packed_rgb = (r8 << 16) | (g8 << 8) | b8
        out.append(bitcast[DType.uint32](self.center_u))
        out.append(bitcast[DType.uint32](self.center_v))
        out.append(bitcast[DType.uint32](self.radius_uv))
        out.append(bitcast[DType.uint32](self.thickness_uv))
        out.append(bitcast[DType.uint32](self.refract_strength))
        out.append(bitcast[DType.uint32](self.chromatic_dispersion))
        out.append(bitcast[DType.uint32](self.glow_intensity))
        out.append(packed_rgb)


struct AaaImpactFrameParams(Copyable, Movable):
    """High-contrast radial-streak anime **Manga Impact Frame** post-process parameters."""
    var blend: Float32              # 0.0 = inactive .. 1.0 = full impact frame
    var center_u: Float32           # Radial speed-line focal point U in [0..1]
    var center_v: Float32           # Radial speed-line focal point V in [0..1]
    var luma_threshold: Float32     # Scene luminance cutoff separating ink vs paper
    var streak_strength: Float32    # Angular speed-line modulation strength (0.0 .. 1.0)
    var accent_width: Float32       # Narrow contour band half-width painted in `accent_color`
    var invert_polarity: Bool       # True swaps black ink <-> white paper for 2-frame strobe flashes
    var paper_gain: Float32         # Linear HDR brightness of paper white
    var ink_color: Vec3             # Linear RGB of dark ink regions
    var accent_color: Vec3          # Linear HDR RGB of contour accent flash (e.g. crimson or gold)

    def __init__(
        out self,
        blend: Float32 = 0.0,
        center_u: Float32 = 0.5,
        center_v: Float32 = 0.5,
        luma_threshold: Float32 = 0.32,
        streak_strength: Float32 = 0.38,
        accent_width: Float32 = 0.055,
        invert_polarity: Bool = False,
        paper_gain: Float32 = 3.0,
        ink_color: Vec3 = Vec3(0.008, 0.008, 0.012),
        accent_color: Vec3 = Vec3(3.4, 0.08, 0.05),
    ):
        self.blend = max(Float32(0.0), min(Float32(1.0), blend))
        self.center_u = center_u
        self.center_v = center_v
        self.luma_threshold = max(Float32(0.02), min(Float32(4.0), luma_threshold))
        self.streak_strength = max(Float32(0.0), min(Float32(1.0), streak_strength))
        self.accent_width = max(Float32(0.0), min(Float32(0.5), accent_width))
        self.invert_polarity = invert_polarity
        self.paper_gain = max(Float32(0.25), paper_gain)
        self.ink_color = ink_color
        self.accent_color = accent_color


# ============================================================================
# Deterministic CPU Reference Evaluators (Matched by SPIR-V Shaders)
# ============================================================================

@always_inline
def _hash_cell_u8_cpu(ix: Int, iy: Int) -> Float32:
    var ux = UInt32(ix & 255)
    var uy = UInt32(iy & 255)
    var h = ux * UInt32(1664525) + uy * UInt32(1013904223)
    h = h ^ (h >> UInt32(16))
    h = h * UInt32(2246822519)
    h = h ^ (h >> UInt32(13))
    return Float32(h & UInt32(0xFFFF)) / Float32(65535.0)


def eval_value_noise_periodic_cpu(u: Float32, v: Float32) -> Float32:
    if not (isfinite(u) and isfinite(v)):
        return Float32(0.5)
    var u64 = Float64(u)
    var v64 = Float64(v)
    var up = (u64 / 256.0 - floor(u64 / 256.0)) * 256.0
    var vp = (v64 / 256.0 - floor(v64 / 256.0)) * 256.0
    if not isfinite(up) or up < 0.0 or up >= 256.0:
        up = 0.0
    if not isfinite(vp) or vp < 0.0 or vp >= 256.0:
        vp = 0.0
    var ix0 = Int(floor(up)) & 255
    var iy0 = Int(floor(vp)) & 255
    var ix1 = (ix0 + 1) & 255
    var iy1 = (iy0 + 1) & 255
    var fx = Float32(up - floor(up))
    var fy = Float32(vp - floor(vp))
    var sx = fx * fx * (Float32(3.0) - Float32(2.0) * fx)
    var sy = fy * fy * (Float32(3.0) - Float32(2.0) * fy)
    var n00 = _hash_cell_u8_cpu(ix0, iy0)
    var n10 = _hash_cell_u8_cpu(ix1, iy0)
    var n01 = _hash_cell_u8_cpu(ix0, iy1)
    var n11 = _hash_cell_u8_cpu(ix1, iy1)
    var nx0 = n00 + (n10 - n00) * sx
    var nx1 = n01 + (n11 - n01) * sx
    return nx0 + (nx1 - nx0) * sy


def eval_fbm2_cpu(u: Float32, v: Float32) -> Float32:
    var o0 = eval_value_noise_periodic_cpu(u, v)
    var o1 = eval_value_noise_periodic_cpu(u * Float32(2.03) + Float32(17.3), v * Float32(2.03) + Float32(31.7))
    return o0 * Float32(0.65) + o1 * Float32(0.35)


def eval_stylized_vfx_cpu(
    params: AaaStylizedVfxParams,
    uv_u: Float32,
    uv_v: Float32,
    time_sec: Float32,
    base_mask: Float32 = 1.0,
) -> Tuple[Bool, Vec3]:
    """Evaluates stylized 3D VFX erosion & 3-band `(ink_rim -> edge -> core)` HDR color on CPU.
    Returns `(visible, linear_hdr_rgb)`."""
    var su = uv_u + time_sec * params.uv_scroll_u
    var sv = uv_v + time_sec * params.uv_scroll_v
    var warp = eval_fbm2_cpu(su * params.warp_scale, sv * params.warp_scale)
    var wu = su + (warp - Float32(0.5)) * params.warp_strength
    var wv = sv + (warp - Float32(0.5)) * params.warp_strength
    var detail = eval_fbm2_cpu(wu * params.warp_scale * Float32(1.45) + Float32(7.1), wv * params.warp_scale * Float32(1.45) + Float32(13.9))
    # Taper near UV V borders (0 and 1) only via `base_mask`, modulated by animated FBM detail
    var field = max(Float32(0.0), min(Float32(1.0), base_mask * (Float32(0.35) + Float32(0.95) * detail)))
    var excess = field - params.erosion
    if excess < Float32(0.0):
        return (False, Vec3(0.0, 0.0, 0.0))

    var core_t = Float32(1.0) if excess >= (params.ink_rim_width + params.edge_band_width) else Float32(0.0)
    var band_rgb = params.edge_color * (Float32(1.0) - core_t) + params.core_color * core_t
    if params.ink_rim_width > Float32(1e-4) and excess < params.ink_rim_width:
        var keep = Float32(1.0) - params.ink_rim_darkness
        band_rgb = band_rgb * keep
    return (True, band_rgb)


def eval_shockwave_chromatic_offsets_cpu(
    ring: AaaShockwaveRing,
    uv_u: Float32,
    uv_v: Float32,
    aspect: Float32,
) -> Tuple[Tuple[Float32, Float32], Tuple[Float32, Float32], Tuple[Float32, Float32], Float32]:
    """Returns `((du_r, dv_r), (du_g, dv_g), (du_b, dv_b), ring_envelope)` for a single `AaaShockwaveRing`."""
    var dx = (uv_u - ring.center_u) * aspect
    var dy = uv_v - ring.center_v
    var r = sqrt(max(Float32(1e-10), dx * dx + dy * dy))
    var dist_from_ring = abs(r - ring.radius_uv)
    if dist_from_ring >= ring.thickness_uv:
        return ((Float32(0.0), Float32(0.0)), (Float32(0.0), Float32(0.0)), (Float32(0.0), Float32(0.0)), Float32(0.0))

    var t = Float32(1.0) - dist_from_ring / max(Float32(1e-5), ring.thickness_uv)
    var env = t * t * (Float32(3.0) - Float32(2.0) * t)
    # Signed wave profile: pushes outward outside crest, pulls inward inside crest
    var signed_wave = env * ((r - ring.radius_uv) / max(Float32(1e-5), ring.thickness_uv)) * Float32(-2.0)
    var dir_u = (dx / r) / max(Float32(0.05), aspect)
    var dir_v = dy / r
    var base_amp = ring.refract_strength * signed_wave
    var kr = Float32(1.0) + ring.chromatic_dispersion
    var kg = Float32(1.0)
    var kb = max(Float32(0.0), Float32(1.0) - ring.chromatic_dispersion)
    return (
        (dir_u * base_amp * kr, dir_v * base_amp * kr),
        (dir_u * base_amp * kg, dir_v * base_amp * kg),
        (dir_u * base_amp * kb, dir_v * base_amp * kb),
        env,
    )


def eval_impact_frame_cpu(
    params: AaaImpactFrameParams,
    hdr_rgb: Vec3,
    uv_u: Float32,
    uv_v: Float32,
    aspect: Float32,
) -> Vec3:
    """Evaluates high-contrast radial-streak anime impact frame on CPU."""
    if params.blend <= Float32(1e-4):
        return hdr_rgb
    var luma = hdr_rgb.x * Float32(0.2126) + hdr_rgb.y * Float32(0.7152) + hdr_rgb.z * Float32(0.0722)
    var dx = (uv_u - params.center_u) * aspect
    var dy = uv_v - params.center_v
    var r = sqrt(dx * dx + dy * dy)
    var ang = atan2(dy, dx)
    var streak = sin(ang * Float32(42.0)) * Float32(0.6) + sin(ang * Float32(97.0) + Float32(1.7)) * Float32(0.4)
    var radial_gate = min(Float32(1.0), r * Float32(2.2))
    var mod_luma = luma + streak * params.streak_strength * radial_gate * params.luma_threshold
    var diff = mod_luma - params.luma_threshold

    var paper = Vec3(params.paper_gain, params.paper_gain, params.paper_gain)
    var ink = params.ink_color
    if params.invert_polarity:
        var tmp = paper
        paper = ink
        ink = tmp

    var stylized: Vec3
    if abs(diff) < params.accent_width:
        stylized = params.accent_color
    elif diff >= Float32(0.0):
        stylized = paper
    else:
        stylized = ink
    return hdr_rgb * (Float32(1.0) - params.blend) + stylized * params.blend


# ============================================================================
# SPIR-V Helpers Shared by `VisBuffer` Alpha-Cutout & `aaa_npr_cel_shade`
# ============================================================================

def _emit_hash_cell_u8_spv(mut s: AaaSpirvModule, ix: UInt32, iy: UInt32) -> UInt32:
    var h0 = s.iadd(s.imul(ix, s.uint(1664525)), s.imul(iy, s.uint(1013904223)))
    var h1 = s.bit_xor(h0, s.shr(h0, s.uint(16)))
    var h2 = s.imul(h1, s.uint(2246822519))
    var h3 = s.bit_xor(h2, s.shr(h2, s.uint(13)))
    return s.fmul(s.float_from_uint(s.bit_and(h3, s.uint(0xFFFF))), s.float(1.0 / 65535.0))


def emit_value_noise_periodic_spv(mut s: AaaSpirvModule, u: UInt32, v: UInt32) -> UInt32:
    var up = s.fmul(s.fract(s.fmul(u, s.float(1.0 / 256.0))), s.float(256.0))
    var vp = s.fmul(s.fract(s.fmul(v, s.float(1.0 / 256.0))), s.float(256.0))
    var fl_u = s.ffloor(up)
    var fl_v = s.ffloor(vp)
    var ix0 = s.bit_and(s.uint_from_float(fl_u), s.uint(255))
    var iy0 = s.bit_and(s.uint_from_float(fl_v), s.uint(255))
    var ix1 = s.bit_and(s.iadd(ix0, s.uint(1)), s.uint(255))
    var iy1 = s.bit_and(s.iadd(iy0, s.uint(1)), s.uint(255))
    var fx = s.fsub(up, fl_u)
    var fy = s.fsub(vp, fl_v)
    var sx = s.fmul(fx, s.fmul(fx, s.fma(fx, s.float(-2.0), s.float(3.0))))
    var sy = s.fmul(fy, s.fmul(fy, s.fma(fy, s.float(-2.0), s.float(3.0))))
    var n00 = _emit_hash_cell_u8_spv(s, ix0, iy0)
    var n10 = _emit_hash_cell_u8_spv(s, ix1, iy0)
    var n01 = _emit_hash_cell_u8_spv(s, ix0, iy1)
    var n11 = _emit_hash_cell_u8_spv(s, ix1, iy1)
    var nx0 = s.fma(s.fsub(n10, n00), sx, n00)
    var nx1 = s.fma(s.fsub(n11, n01), sx, n01)
    return s.fma(s.fsub(nx1, nx0), sy, nx0)


def emit_fbm2_spv(mut s: AaaSpirvModule, u: UInt32, v: UInt32) -> UInt32:
    var o0 = emit_value_noise_periodic_spv(s, u, v)
    var u2 = s.fma(u, s.float(2.03), s.float(17.3))
    var v2 = s.fma(v, s.float(2.03), s.float(31.7))
    var o1 = emit_value_noise_periodic_spv(s, u2, v2)
    return s.fma(o0, s.float(0.65), s.fmul(o1, s.float(0.35)))


def emit_find_vfx_entry_spv(mut s: AaaSpirvModule, b_globals: UInt32, mat_id: UInt32) -> UInt32:
    """Returns word offset in `b_globals` of matching `AaaStylizedVfxParams`, or `0xFFFFFFFF` if none."""
    var n_vfx = s.umin(
        s.load_word(b_globals, s.uint(UInt32(AAA_GLOBALS_VFX_COUNT_WORD))),
        s.uint(UInt32(AAA_MAX_VFX_MATERIALS)),
    )
    var found = s.uvar(s.uint(0xFFFFFFFF))
    s.begin_if(s.ult(s.uint(0), n_vfx))
    var vi_v = s.uvar(s.uint(0))
    s.begin_loop()
    var vi = s.load_u(vi_v)
    s.break_if(s.uge(vi, n_vfx))
    var e_off = s.iadd(s.uint(UInt32(AAA_GLOBALS_VFX_TABLE_WORD)), s.imul(vi, s.uint(UInt32(AAA_VFX_ENTRY_WORDS))))
    s.begin_if(s.ieq(s.load_word(b_globals, e_off), mat_id))
    s.store(found, e_off)
    s.break_if(s.ieq(s.uint(1), s.uint(1)))
    s.end_if()
    s.store(vi_v, s.iadd(vi, s.uint(1)))
    s.end_loop()
    s.end_if()
    return s.load_u(found)


def emit_eval_vfx_excess_and_color_spv(
    mut s: AaaSpirvModule,
    b_globals: UInt32,
    bindless_arr: Tuple[UInt32, UInt32],
    e_base: UInt32,
    uv_u: UInt32,
    uv_v: UInt32,
    ddx_vec: UInt32,
    ddy_vec: UInt32,
    base_mask: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """SPIR-V twin of `eval_stylized_vfx_cpu`. Returns `(excess, hdr_r, hdr_g, hdr_b)` where `excess < 0` is eroded."""
    var time_s = s.load_float(b_globals, s.uint(123))
    var n_slot = s.bit_and(s.load_word(b_globals, s.iadd(e_base, s.uint(1))), s.uint(0xFF))
    var scr_u = s.load_float(b_globals, s.iadd(e_base, s.uint(2)))
    var scr_v = s.load_float(b_globals, s.iadd(e_base, s.uint(3)))
    var w_scale = s.load_float(b_globals, s.iadd(e_base, s.uint(4)))
    var w_str = s.load_float(b_globals, s.iadd(e_base, s.uint(5)))
    var erosion = s.load_float(b_globals, s.iadd(e_base, s.uint(6)))
    var edge_w = s.load_float(b_globals, s.iadd(e_base, s.uint(7)))
    var core_r = s.load_float(b_globals, s.iadd(e_base, s.uint(8)))
    var core_g = s.load_float(b_globals, s.iadd(e_base, s.uint(9)))
    var core_b = s.load_float(b_globals, s.iadd(e_base, s.uint(10)))
    var edge_r = s.load_float(b_globals, s.iadd(e_base, s.uint(11)))
    var edge_g = s.load_float(b_globals, s.iadd(e_base, s.uint(12)))
    var edge_b = s.load_float(b_globals, s.iadd(e_base, s.uint(13)))
    var ink_w = s.load_float(b_globals, s.iadd(e_base, s.uint(14)))
    var ink_dark = s.load_float(b_globals, s.iadd(e_base, s.uint(15)))

    var su = s.fma(time_s, scr_u, uv_u)
    var sv = s.fma(time_s, scr_v, uv_v)
    var warp = emit_fbm2_spv(s, s.fmul(su, w_scale), s.fmul(sv, w_scale))
    var w_off = s.fmul(s.fsub(warp, s.float(0.5)), w_str)
    var wu = s.fadd(su, w_off)
    var wv = s.fadd(sv, w_off)

    var detail_v = s.fvar(
        emit_fbm2_spv(
            s,
            s.fma(s.fmul(wu, w_scale), s.float(1.45), s.float(7.1)),
            s.fma(s.fmul(wv, w_scale), s.float(1.45), s.float(13.9)),
        )
    )
    s.begin_if(s.ult(n_slot, s.uint(0xFF)))
    var n_samp = s.sample_bindless_grad(bindless_arr, n_slot, s.vec2(wu, wv), ddx_vec, ddy_vec)
    s.store(detail_v, s.extract_f(n_samp, 0))
    s.end_if()

    var field = s.nclamp(
        s.fmul(base_mask, s.fma(s.load_f(detail_v), s.float(0.95), s.float(0.35))),
        s.float(0.0),
        s.float(1.0),
    )
    var excess = s.fsub(field, erosion)

    # Anti-aliased 1-pixel step between `edge_color` and `core_color`
    var core_cut = s.fadd(ink_w, edge_w)
    var core_t = s.nclamp(s.fmul(s.fsub(excess, core_cut), s.float(32.0)), s.float(0.0), s.float(1.0))
    var br = s.fma(s.fsub(core_r, edge_r), core_t, edge_r)
    var bg = s.fma(s.fsub(core_g, edge_g), core_t, edge_g)
    var bb = s.fma(s.fsub(core_b, edge_b), core_t, edge_b)

    var in_ink = s.logical_and(s.fgt(ink_w, s.float(1e-4)), s.flt(excess, ink_w))
    var ink_mul = s.select_f(in_ink, s.fsub(s.float(1.0), ink_dark), s.float(1.0))
    return (excess, s.fmul(br, ink_mul), s.fmul(bg, ink_mul), s.fmul(bb, ink_mul))


# ============================================================================
# Screen-Space Chromatic Shockwaves & Manga Impact Frame Compute Pass (`8x8`)
# ============================================================================

def _emit_sample_hdr_scratch_bilinear(
    mut s: AaaSpirvModule,
    b_post: UInt32,
    src_copy_off: UInt32,
    w_u: UInt32,
    h_u: UInt32,
    fw: UInt32,
    fh: UInt32,
    u_in: UInt32,
    v_in: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    var u = s.nclamp(u_in, s.float(0.0), s.float(0.9999))
    var v = s.nclamp(v_in, s.float(0.0), s.float(0.9999))
    var fx = s.nclamp(s.fsub(s.fmul(u, fw), s.float(0.5)), s.float(0.0), s.nmax(s.float(0.0), s.fsub(fw, s.float(1.0))))
    var fy = s.nclamp(s.fsub(s.fmul(v, fh), s.float(0.5)), s.float(0.0), s.nmax(s.float(0.0), s.fsub(fh, s.float(1.0))))
    var x0 = s.umin(s.isub(w_u, s.uint(1)), s.saturate_uint(s.ffloor(fx)))
    var y0 = s.umin(s.isub(h_u, s.uint(1)), s.saturate_uint(s.ffloor(fy)))
    var x1 = s.umin(s.isub(w_u, s.uint(1)), s.iadd(x0, s.uint(1)))
    var y1 = s.umin(s.isub(h_u, s.uint(1)), s.iadd(y0, s.uint(1)))
    var tx = s.nclamp(s.fsub(fx, s.float_from_uint(x0)), s.float(0.0), s.float(1.0))
    var ty = s.nclamp(s.fsub(fy, s.float_from_uint(y0)), s.float(0.0), s.float(1.0))
    var omx = s.fsub(s.float(1.0), tx)
    var omy = s.fsub(s.float(1.0), ty)
    var w00 = s.fmul(omx, omy)
    var w10 = s.fmul(tx, omy)
    var w01 = s.fmul(omx, ty)
    var w11 = s.fmul(tx, ty)

    var p00 = s.iadd(src_copy_off, s.imul(s.iadd(s.imul(y0, w_u), x0), s.uint(8)))
    var p10 = s.iadd(src_copy_off, s.imul(s.iadd(s.imul(y0, w_u), x1), s.uint(8)))
    var p01 = s.iadd(src_copy_off, s.imul(s.iadd(s.imul(y1, w_u), x0), s.uint(8)))
    var p11 = s.iadd(src_copy_off, s.imul(s.iadd(s.imul(y1, w_u), x1), s.uint(8)))

    var r = s.fma(s.load_float(b_post, p00), w00, s.fma(s.load_float(b_post, p10), w10, s.fma(s.load_float(b_post, p01), w01, s.fmul(s.load_float(b_post, p11), w11))))
    var g = s.fma(s.load_float(b_post, s.iadd(p00, s.uint(1))), w00, s.fma(s.load_float(b_post, s.iadd(p10, s.uint(1))), w10, s.fma(s.load_float(b_post, s.iadd(p01, s.uint(1))), w01, s.fmul(s.load_float(b_post, s.iadd(p11, s.uint(1))), w11))))
    var b = s.fma(s.load_float(b_post, s.iadd(p00, s.uint(2))), w00, s.fma(s.load_float(b_post, s.iadd(p10, s.uint(2))), w10, s.fma(s.load_float(b_post, s.iadd(p01, s.uint(2))), w01, s.fmul(s.load_float(b_post, s.iadd(p11, s.uint(2))), w11))))
    return (r, g, b)


def aaa_npr_post_fx_shader() -> List[UInt32]:
    """Screen-Space Chromatic-Dispersion Shockwaves + Manga Impact Frame Compute Pass (`group = 8 x 8`).
    Reads copied `buf_hdr_out` (`8` words/px) from `b_post[Push[2]]` (`AAA_POST_BLOOM_BASE`) and writes `b_hdr`.
    Push[0] = out_w, Push[1] = out_h, Push[2] = src_copy_word_offset.
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var src_copy_off = s.push_word(2)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var aspect = s.fdiv(fw, s.nmax(s.float(1.0), fh))
    var inv_asp = s.fdiv(s.float(1.0), s.nmax(s.float(0.05), aspect))
    var uv_u = s.fdiv(s.fadd(s.float_from_uint(px), s.float(0.5)), fw)
    var uv_v = s.fdiv(s.fadd(s.float_from_uint(py), s.float(0.5)), fh)

    var pix = s.iadd(s.imul(py, w_u), px)
    var src_pix = s.iadd(src_copy_off, s.imul(pix, s.uint(8)))
    var out_pix = s.imul(pix, s.uint(8))

    var cur_r = s.fvar(s.load_float(b_post, src_pix))
    var cur_g = s.fvar(s.load_float(b_post, s.iadd(src_pix, s.uint(1))))
    var cur_b = s.fvar(s.load_float(b_post, s.iadd(src_pix, s.uint(2))))

    # 1. Radial Chromatic-Dispersion Shockwave Rings (`0 .. 4`)
    var n_sw = s.umin(
        s.load_word(b_globals, s.uint(UInt32(AAA_GLOBALS_SHOCKWAVE_COUNT_WORD))),
        s.uint(UInt32(AAA_MAX_SHOCKWAVES)),
    )
    s.begin_if(s.ult(s.uint(0), n_sw))
    var du_r = s.fvar(s.float(0.0)); var dv_r = s.fvar(s.float(0.0))
    var du_g = s.fvar(s.float(0.0)); var dv_g = s.fvar(s.float(0.0))
    var du_b = s.fvar(s.float(0.0)); var dv_b = s.fvar(s.float(0.0))
    var glow_r = s.fvar(s.float(0.0))
    var glow_g = s.fvar(s.float(0.0))
    var glow_b = s.fvar(s.float(0.0))
    var any_hit = s.uvar(s.uint(0))

    var sw_i = s.uvar(s.uint(0))
    s.begin_loop()
    var si = s.load_u(sw_i)
    s.break_if(s.uge(si, n_sw))
    var sb = s.iadd(s.uint(UInt32(AAA_GLOBALS_SHOCKWAVE_TABLE_WORD)), s.imul(si, s.uint(UInt32(AAA_SHOCKWAVE_WORDS))))
    var cu = s.load_float(b_globals, sb)
    var cv = s.load_float(b_globals, s.iadd(sb, s.uint(1)))
    var rad = s.load_float(b_globals, s.iadd(sb, s.uint(2)))
    var thk = s.nmax(s.float(1e-4), s.load_float(b_globals, s.iadd(sb, s.uint(3))))
    var amp = s.load_float(b_globals, s.iadd(sb, s.uint(4)))
    var disp = s.load_float(b_globals, s.iadd(sb, s.uint(5)))
    var gl_int = s.load_float(b_globals, s.iadd(sb, s.uint(6)))
    var gl_pk = s.load_word(b_globals, s.iadd(sb, s.uint(7)))

    var dx = s.fmul(s.fsub(uv_u, cu), aspect)
    var dy = s.fsub(uv_v, cv)
    var r = s.fsqrt(s.nmax(s.float(1e-10), s.fma(dx, dx, s.fmul(dy, dy))))
    var d_ring = s.fabs(s.fsub(r, rad))
    s.begin_if(s.flt(d_ring, thk))
    s.store(any_hit, s.uint(1))
    var t = s.fsub(s.float(1.0), s.fdiv(d_ring, thk))
    var env = s.fmul(t, s.fmul(t, s.fma(t, s.float(-2.0), s.float(3.0))))
    var signed_w = s.fmul(env, s.fmul(s.fdiv(s.fsub(r, rad), thk), s.float(-2.0)))
    var inv_r = s.fdiv(s.float(1.0), r)
    var dir_u = s.fmul(s.fmul(dx, inv_r), inv_asp)
    var dir_v = s.fmul(dy, inv_r)
    var base_a = s.fmul(amp, signed_w)
    var kr = s.fadd(s.float(1.0), disp)
    var kb = s.nmax(s.float(0.0), s.fsub(s.float(1.0), disp))

    s.store(du_r, s.fma(dir_u, s.fmul(base_a, kr), s.load_f(du_r)))
    s.store(dv_r, s.fma(dir_v, s.fmul(base_a, kr), s.load_f(dv_r)))
    s.store(du_g, s.fma(dir_u, base_a, s.load_f(du_g)))
    s.store(dv_g, s.fma(dir_v, base_a, s.load_f(dv_g)))
    s.store(du_b, s.fma(dir_u, s.fmul(base_a, kb), s.load_f(du_b)))
    s.store(dv_b, s.fma(dir_v, s.fmul(base_a, kb), s.load_f(dv_b)))

    var inv255 = s.float(1.0 / 255.0)
    var gr = s.fmul(s.float_from_uint(s.bit_and(s.shr(gl_pk, s.uint(16)), s.uint(0xFF))), inv255)
    var gg = s.fmul(s.float_from_uint(s.bit_and(s.shr(gl_pk, s.uint(8)), s.uint(0xFF))), inv255)
    var gb = s.fmul(s.float_from_uint(s.bit_and(gl_pk, s.uint(0xFF))), inv255)
    var g_env = s.fmul(env, gl_int)
    s.store(glow_r, s.fma(gr, g_env, s.load_f(glow_r)))
    s.store(glow_g, s.fma(gg, g_env, s.load_f(glow_g)))
    s.store(glow_b, s.fma(gb, g_env, s.load_f(glow_b)))
    s.end_if()

    s.store(sw_i, s.iadd(si, s.uint(1)))
    s.end_loop()

    s.begin_if(s.ine(s.load_u(any_hit), s.uint(0)))
    var sr = _emit_sample_hdr_scratch_bilinear(s, b_post, src_copy_off, w_u, h_u, fw, fh, s.fadd(uv_u, s.load_f(du_r)), s.fadd(uv_v, s.load_f(dv_r)))
    var sg = _emit_sample_hdr_scratch_bilinear(s, b_post, src_copy_off, w_u, h_u, fw, fh, s.fadd(uv_u, s.load_f(du_g)), s.fadd(uv_v, s.load_f(dv_g)))
    var sb_c = _emit_sample_hdr_scratch_bilinear(s, b_post, src_copy_off, w_u, h_u, fw, fh, s.fadd(uv_u, s.load_f(du_b)), s.fadd(uv_v, s.load_f(dv_b)))
    s.store(cur_r, s.fadd(sr[0], s.load_f(glow_r)))
    s.store(cur_g, s.fadd(sg[1], s.load_f(glow_g)))
    s.store(cur_b, s.fadd(sb_c[2], s.load_f(glow_b)))
    s.end_if()
    s.end_if()

    # 2. High-Contrast Radial-Streak Anime **Manga Impact Frame**
    var imp_base = s.uint(UInt32(AAA_GLOBALS_IMPACT_FRAME_WORD))
    var imp_blend = s.load_float(b_globals, imp_base)
    s.begin_if(s.fgt(imp_blend, s.float(0.001)))
    var icu = s.load_float(b_globals, s.iadd(imp_base, s.uint(1)))
    var icv = s.load_float(b_globals, s.iadd(imp_base, s.uint(2)))
    var l_thresh = s.load_float(b_globals, s.iadd(imp_base, s.uint(3)))
    var str_amt = s.load_float(b_globals, s.iadd(imp_base, s.uint(4)))
    var inv_and_acc = s.load_float(b_globals, s.iadd(imp_base, s.uint(5)))
    var is_inv = s.flt(inv_and_acc, s.float(0.0))
    var acc_w = s.fabs(inv_and_acc)
    var ink_r = s.load_float(b_globals, s.iadd(imp_base, s.uint(6)))
    var ink_g = s.load_float(b_globals, s.iadd(imp_base, s.uint(7)))
    var ink_b = s.load_float(b_globals, s.iadd(imp_base, s.uint(8)))
    var acc_r = s.load_float(b_globals, s.iadd(imp_base, s.uint(9)))
    var acc_g = s.load_float(b_globals, s.iadd(imp_base, s.uint(10)))
    var acc_b = s.load_float(b_globals, s.iadd(imp_base, s.uint(11)))
    var paper_g = s.load_float(b_globals, s.uint(UInt32(AAA_GLOBALS_IMPACT_PAPER_GAIN_WORD)))

    var r_in = s.load_f(cur_r)
    var g_in = s.load_f(cur_g)
    var b_in = s.load_f(cur_b)
    var luma = s.fma(r_in, s.float(0.2126), s.fma(g_in, s.float(0.7152), s.fmul(b_in, s.float(0.0722))))
    var idx_x = s.fmul(s.fsub(uv_u, icu), aspect)
    var idx_y = s.fsub(uv_v, icv)
    var ir = s.fsqrt(s.fma(idx_x, idx_x, s.fmul(idx_y, idx_y)))
    var safe_x = s.select_f(s.logical_and(s.flt(s.fabs(idx_x), s.float(1e-6)), s.flt(s.fabs(idx_y), s.float(1e-6))), s.float(1e-6), idx_x)
    var ang = s.fatan2(idx_y, safe_x)
    var st0 = s.fsin(s.fmul(ang, s.float(42.0)))
    var st1 = s.fsin(s.fma(ang, s.float(97.0), s.float(1.7)))
    var streak = s.fma(st0, s.float(0.6), s.fmul(st1, s.float(0.4)))
    var r_gate = s.nmin(s.float(1.0), s.fmul(ir, s.float(2.2)))
    var mod_luma = s.fma(streak, s.fmul(str_amt, s.fmul(r_gate, l_thresh)), luma)
    var diff = s.fsub(mod_luma, l_thresh)

    var p_r = s.select_f(is_inv, ink_r, paper_g)
    var p_g = s.select_f(is_inv, ink_g, paper_g)
    var p_b = s.select_f(is_inv, ink_b, paper_g)
    var k_r = s.select_f(is_inv, paper_g, ink_r)
    var k_g = s.select_f(is_inv, paper_g, ink_g)
    var k_b = s.select_f(is_inv, paper_g, ink_b)

    var is_paper = s.fge(diff, s.float(0.0))
    var base_sr = s.select_f(is_paper, p_r, k_r)
    var base_sg = s.select_f(is_paper, p_g, k_g)
    var base_sb = s.select_f(is_paper, p_b, k_b)
    var is_acc = s.flt(s.fabs(diff), acc_w)
    var sty_r = s.select_f(is_acc, acc_r, base_sr)
    var sty_g = s.select_f(is_acc, acc_g, base_sg)
    var sty_b = s.select_f(is_acc, acc_b, base_sb)

    s.store(cur_r, s.fma(s.fsub(sty_r, r_in), imp_blend, r_in))
    s.store(cur_g, s.fma(s.fsub(sty_g, g_in), imp_blend, g_in))
    s.store(cur_b, s.fma(s.fsub(sty_b, b_in), imp_blend, b_in))
    s.end_if()

    s.store_float(b_hdr, out_pix, s.load_f(cur_r))
    s.store_float(b_hdr, s.iadd(out_pix, s.uint(1)), s.load_f(cur_g))
    s.store_float(b_hdr, s.iadd(out_pix, s.uint(2)), s.load_f(cur_b))
    return s.finish()


def visbuffer_alphamask_vfx_fragment_shader() -> List[UInt32]:
    """Bucket-2 Alpha-Cutout + Stylized Eroding VFX Fragment Shader (`early_depth = False`, `enable_bindless = True`).
    Identical to `visbuffer_alphamask_fragment_shader` when `n_vfx == 0`, and additionally discards (`OpKill`)
    dissolved texels (`excess < 0.0`) on `AaaStylizedVfxParams` meshes so background geometry & RT/VSM depth stay exact!
    """
    var s = AaaSpirvModule(
        MODEL_FRAGMENT,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
        early_depth=False,
    )
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var prim_id = s.primitive_id()
    var in_vis = s.add_stage_io(STORAGE_INPUT, s.type_uvec2, location=0, flat=True)
    var in_uv = s.add_stage_io(STORAGE_INPUT, s.type_vec2, location=1, flat=False)
    var in_talb = s.add_stage_io(STORAGE_INPUT, s.type_uint, location=2, flat=True)
    var out_col = s.add_stage_io(STORAGE_OUTPUT, s.type_uvec4, location=0, flat=False)

    var uv_val = s.fresh()
    emit(s.code, OP_LOAD, s.type_vec2, uv_val, in_uv[1])
    var uv_x = s.extract_f(uv_val, 0)
    var uv_y = s.extract_f(uv_val, 1)

    var ddx_u = s.dpdx(uv_x)
    var ddx_v = s.dpdx(uv_y)
    var ddy_u = s.dpdy(uv_x)
    var ddy_v = s.dpdy(uv_y)
    var ddx_vec = s.vec2(ddx_u, ddx_v)
    var ddy_vec = s.vec2(ddy_u, ddy_v)

    var gx2 = s.fma(ddx_u, ddx_u, s.fmul(ddx_v, ddx_v))
    var gy2 = s.fma(ddy_u, ddy_u, s.fmul(ddy_v, ddy_v))
    var px_foot = s.fmul(s.fsqrt(s.nmax(s.float(1e-8), s.nmax(gx2, gy2))), s.float(512.0))
    var mip_est = s.nclamp(s.log2(s.nmax(s.float(1.0), px_foot)), s.float(0.0), s.float(6.0))

    var packed = s.fresh()
    emit(s.code, OP_LOAD, s.type_uint, packed, in_talb[1])
    var t_alb = s.bit_and(packed, s.uint(0xFF))
    var raw_alpha_v = s.fvar(s.float(1.0))
    var alpha_v = s.fvar(s.float(1.0))
    s.begin_if(s.ult(t_alb, s.uint(0xFF)))
    var rgba = s.sample_bindless_grad(bindless_arr, t_alb, uv_val, ddx_vec, ddy_vec)
    var raw_a = s.extract_f(rgba, 3)
    s.store(raw_alpha_v, raw_a)
    s.store(alpha_v, s.fmul(raw_a, s.fma(mip_est, s.float(0.18), s.float(1.0))))
    s.end_if()

    var vis_val = s.fresh()
    emit(s.code, OP_LOAD, s.type_uvec2, vis_val, in_vis[1])
    var w0 = s.extract_u(vis_val, 0)
    var w1_base = s.extract_u(vis_val, 1)
    var final_mat = s.shr(w1_base, s.uint(16))

    var vfx_e = emit_find_vfx_entry_spv(s, b_globals, final_mat)
    s.begin_if(s.ine(vfx_e, s.uint(0xFFFFFFFF)))
    var vfx_res = emit_eval_vfx_excess_and_color_spv(
        s, b_globals, bindless_arr, vfx_e, uv_x, uv_y, ddx_vec, ddy_vec, s.load_f(raw_alpha_v)
    )
    s.store(alpha_v, s.select_f(s.fge(vfx_res[0], s.float(0.0)), s.float(1.0), s.float(0.0)))
    s.end_if()

    s.discard_if(s.flt(s.nmin(s.float(1.0), s.load_f(alpha_v)), emit_alpha_cutoff(s, packed)))

    var w1 = s.bit_or(w1_base, s.bit_and(prim_id, s.uint(0xFF)))
    emit(s.code, OP_STORE, out_col[1], s.uvec4(w0, w1, s.uint(0), s.uint(1)))
    return s.finish()

