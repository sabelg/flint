"""The frame's compute passes in SPIR-V, extending the baseline shaders in gpu_raster.mojo
with six modern Vulkan compute rendering systems:

1. True Linear Float32 HDR Framebuffer (PLANE_HDR_R/G/B) + Post-Compose Troy Sobotka AgX
   chromaticity inset, hue-preserving HDR shoulder, sub-integer interpolated tone LUT,
   and gamut-safe perceptual saturation.
2. 2-Cascade GPU Shadow Mapping (CSM, 1024x1024x2 via OpAtomicUMax) + 8-tap IGN-rotated
   Poisson disk soft PCF + normal-offset slope bias.
3. Temporal Anti-Aliasing (TAA) with 3x3 YCoCg neighborhood clamping, exact 3D camera
   motion-vector reprojection, ping-pong history planes, and 6-tap depth-weighted
   directional velocity motion blur (keeping HUD razor-sharp).
4. Automatic 5-level mipmap trilinear + up-to-4x major-axis anisotropic texture filtering
   from exact analytical plane-equation UV derivatives + tangent-free cotangent-frame
   per-texel micro-normal & ORM roughness perturbation.
5. Screen-Space Raymarched Reflections (12-step adaptive Hi-Z/depth SSR + bisection
   refinement in linear HDR) + Brian Karis analytical Split-Sum GGX Environment BRDF.
6. 10-step IGN-jittered Volumetric Fog & CSM-shadowed Henyey-Greenstein crepuscular
   light shafts + local point-light in-scattering.
"""
from flint.render.gpu_layout import (
    ADDITIVE_FLAG, ALPHA_ERODE_FLAG, BIND_BACKDROP, BIND_CELLS, BIND_CONSTANTS, BIND_HUD, BIND_LISTS,
    BIND_RECORDS, BIND_SHADOW_MAP, BIND_SHOWN, BIND_TARGET, BIND_TEXTURES, BIND_TILES, BIND_VOXELS,
    BLACKBODY_FLAG, BLOOM_FINAL_OFFSET, BLOOM_HIST_OFFSET, BLOOM_SAMPLES_X, BLOOM_SAMPLES_Y,
    CLUSTER_WORDS_PER_SLICE, CLUSTER_Z_SLICES, CONSTANT_BLOOM_DISTANCE, CONSTANT_CAMERA,
    CONSTANT_CLUSTER_LIGHTS, CONSTANT_CSM, CONSTANT_DECALS, CONSTANT_DITHER, CONSTANT_EFFECTS,
    CONSTANT_ENV_GROUND, CONSTANT_ENV_HORIZON, CONSTANT_ENV_SKY, CONSTANT_KEY, CONSTANT_LIGHTS,
    CONSTANT_PREV_CAMERA, CONSTANT_PROBES, CONSTANT_RADIANCE_CASCADES, CONSTANT_SATURATION,
    CONSTANT_SDF_OCCLUDERS, CONSTANT_SPHERES, CONSTANT_V10_POST, CONSTANT_V11_POST, CONSTANT_V7_POST,
    CONSTANT_V8_ATMOS, CONSTANT_V9_CLOUDS, CONSTANT_VOL_POST, CONSTANT_WORLD, CSM_CASCADE_WORDS,
    CSM_RECORD_DWORDS, CSM_RES, CSM_TILE_SIZE, DITHER_FLAG, EFFECT_AUTO_EXPOSURE, EFFECT_BILINEAR,
    EFFECT_CHROMATIC_REFRACT, EFFECT_CSM, EFFECT_DOF, EFFECT_DYN_REFLECT, EFFECT_DYRL_ANIME, EFFECT_FROXEL_VOL,
    EFFECT_GTAO_CONTACT, EFFECT_PHYSICAL_SKY, EFFECT_RADIANCE_GI, EFFECT_SDF_SHADOWS, EFFECT_SHADOWS,
    EFFECT_SSGI_BENT, EFFECT_SSR, EFFECT_TAA,
    EFFECT_VOL_CLOUDS, EFFECT_VOLUMETRIC, EFFECT_WEATHER_PRECIP, FROXEL_INTEGRATED_OFFSET, FROXEL_X, FROXEL_Y,
    FROXEL_Z, GPU_CULL_IN_STRIDE_WORDS, GPU_PARTICLE_STRIDE_WORDS, GROUP_HEIGHT, GROUP_WIDTH,
    MAX_IRRADIANCE_PROBES, MAX_RADIANCE_CASCADES, MAX_SDF_OCCLUDERS,
    MAX_SURFACE_DECALS, MIPMAPPED_FLAG,
    NORM_ORM_FLAG, NO_DEPTH_WRITE_FLAG, OBJECT_VELOCITY_FLAG, PBR_FLAG, PER_PIXEL_FLAG,
    RECORD_AMBIENT_OCCLUSION, RECORD_ANISO, RECORD_BLACKBODY_HEAT, RECORD_EMISSIVE,
    RECORD_ERODE_THRESHOLD, RECORD_FLAGS, RECORD_FOG, RECORD_INVERSE_Z, RECORD_LIGHT,
    RECORD_METALLIC, RECORD_NORMAL, RECORD_NORM_ORM_TEX, RECORD_REFRACT_STRENGTH, RECORD_ROUGHNESS,
    RECORD_SOFT_INV_DIST, RECORD_TEXTURE, RECORD_TEXTURE_FLOAT_HEIGHT, RECORD_TEXTURE_FLOAT_WIDTH,
    RECORD_TEXTURE_MAX_X, RECORD_TEXTURE_MAX_Y, RECORD_TEXTURE_WIDTH, RECORD_TINT, RECORD_U,
    RECORD_V, RECORD_VEL_X, RECORD_VEL_Y, RECORD_WEIGHT_A, RECORD_WEIGHT_B, REFRACT_FLAG,
    SHADOW_COARSE_STEPS, SHADOW_FINE_STEPS, SKY_LUT_H, SKY_LUT_OFFSET, SKY_LUT_W,
    SOFT_PARTICLE_FLAG, TEXTURED_FLAG, TILE_ENTRY_DWORDS, WATER_WAVE_FLAG,
)
from flint.render.spirv import SpirvModule

# Push constants, in words.
comptime RASTER_PUSH_ROW: Int = 0
comptime RASTER_PUSH_COLUMNS: Int = 1
comptime RASTER_PUSH_CLEAR: Int = 2
comptime RASTER_PUSH_WIDTH: Int = 3
comptime RASTER_PUSH_HEIGHT: Int = 4
comptime RASTER_PUSH_WORDS: Int = 5

comptime REDUCE_PUSH_ROW: Int = 0
comptime REDUCE_PUSH_CELLS_X: Int = 1
comptime REDUCE_PUSH_CELLS_Y: Int = 2
comptime REDUCE_PUSH_SCALE_X: Int = 3
comptime REDUCE_PUSH_SCALE_Y: Int = 4
comptime REDUCE_PUSH_LAST_X: Int = 5
comptime REDUCE_PUSH_LAST_Y: Int = 6
comptime REDUCE_PUSH_SOURCE: Int = 7
comptime REDUCE_PUSH_WORDS: Int = 8

comptime BLUR_PUSH_CELLS_X: Int = 0
comptime BLUR_PUSH_CELLS_Y: Int = 1
comptime BLUR_PUSH_INVERSE_STRENGTH: Int = 2
comptime BLUR_PUSH_SOURCE: Int = 3
comptime BLUR_PUSH_TARGET: Int = 4
comptime BLUR_PUSH_PACK_BYTES: Int = 5
comptime BLUR_PUSH_WORDS: Int = 6

comptime KAWASE_DOWN_SRC_W: Int = 0
comptime KAWASE_DOWN_SRC_H: Int = 1
comptime KAWASE_DOWN_DST_W: Int = 2
comptime KAWASE_DOWN_DST_H: Int = 3
comptime KAWASE_DOWN_SRC_OFF: Int = 4
comptime KAWASE_DOWN_DST_OFF: Int = 5
comptime KAWASE_DOWN_WORDS: Int = 6

comptime KAWASE_UP_LOW_W: Int = 0
comptime KAWASE_UP_LOW_H: Int = 1
comptime KAWASE_UP_HIGH_W: Int = 2
comptime KAWASE_UP_HIGH_H: Int = 3
comptime KAWASE_UP_LOW_OFF: Int = 4
comptime KAWASE_UP_HIGH_OFF: Int = 5
comptime KAWASE_UP_DST_OFF: Int = 6
comptime KAWASE_UP_INV_STRENGTH: Int = 7
comptime KAWASE_UP_WORDS: Int = 8

comptime COMPOSE_PUSH_ROW: Int = 0
comptime COMPOSE_PUSH_WIDTH: Int = 1
comptime COMPOSE_PUSH_HEIGHT: Int = 2
comptime COMPOSE_PUSH_CELLS_X: Int = 3
comptime COMPOSE_PUSH_SCALE_X: Int = 4
comptime COMPOSE_PUSH_SCALE_Y: Int = 5
comptime COMPOSE_PUSH_BLOOM: Int = 6
comptime COMPOSE_PUSH_CLEAR: Int = 7
comptime COMPOSE_PUSH_CELLS_Y: Int = 8
comptime COMPOSE_PUSH_PLANE_STRIDE: Int = 9
comptime COMPOSE_PUSH_WORDS: Int = 10

comptime SMOOTH_PUSH_ROW: Int = 0
comptime SMOOTH_PUSH_WIDTH: Int = 1
comptime SMOOTH_PUSH_HEIGHT: Int = 2
comptime SMOOTH_PUSH_TARGET: Int = 3
comptime SMOOTH_PUSH_WORDS: Int = 4

comptime CSM_CLEAR_PUSH_QUADS: Int = 0
comptime CSM_CLEAR_PUSH_WORDS: Int = 1

comptime CSM_RASTER_PUSH_RECORDS: Int = 0
comptime CSM_RASTER_PUSH_ITEMS: Int = 1
comptime CSM_RASTER_PUSH_COUNT: Int = 2
comptime CSM_RASTER_PUSH_GROUPS_X: Int = 3
comptime CSM_RASTER_PUSH_WORDS: Int = 4

comptime TAA_PUSH_ROW: Int = 0
comptime TAA_PUSH_WIDTH: Int = 1
comptime TAA_PUSH_HEIGHT: Int = 2
comptime TAA_PUSH_SCENE: Int = 3
comptime TAA_PUSH_PLANE_STRIDE: Int = 4
comptime TAA_PUSH_WORDS: Int = 5

comptime HIST_PUSH_ROW: Int = 0
comptime HIST_PUSH_WIDTH: Int = 1
comptime HIST_PUSH_HEIGHT: Int = 2
comptime HIST_PUSH_PLANE_STRIDE: Int = 3
comptime HIST_PUSH_WORDS: Int = 4
comptime HIST_GROUP_SIZE: Int = 16

comptime FROXEL_PUSH_READ_OFF: Int = 0
comptime FROXEL_PUSH_WRITE_OFF: Int = 1
comptime FROXEL_PUSH_INTEG_OFF: Int = 2
comptime FROXEL_PUSH_FRAME: Int = 3
comptime FROXEL_PUSH_WORDS: Int = 4

comptime HIZ_PUSH_ROW: Int = 0
comptime HIZ_PUSH_WIDTH: Int = 1
comptime HIZ_PUSH_HEIGHT: Int = 2
comptime HIZ_PUSH_TILES_X: Int = 3
comptime HIZ_PUSH_DEPTH_OFF: Int = 4
comptime HIZ_PUSH_OUT_OFF: Int = 5
comptime HIZ_PUSH_WORDS: Int = 6

comptime SKIN_PUSH_JOINTS_OFF: Int = 0
comptime SKIN_PUSH_IN_OFF: Int = 1
comptime SKIN_PUSH_OUT_OFF: Int = 2
comptime SKIN_PUSH_VERT_COUNT: Int = 3
comptime SKIN_PUSH_WORDS: Int = 4

comptime PARTICLE_PUSH_IN_OFF: Int = 0
comptime PARTICLE_PUSH_OUT_OFF: Int = 1
comptime PARTICLE_PUSH_COUNT: Int = 2
comptime PARTICLE_PUSH_DT: Int = 3
comptime PARTICLE_PUSH_GRAVITY: Int = 4
comptime PARTICLE_PUSH_DRAG: Int = 5
comptime PARTICLE_PUSH_DEPTH_OFF: Int = 6
comptime PARTICLE_PUSH_ROW_BOUNCE: Int = 7
comptime PARTICLE_PUSH_WORDS: Int = 8

comptime TSR_PUSH_SRC_OFF: Int = 0
comptime TSR_PUSH_DST_OFF: Int = 1
comptime TSR_PUSH_HIST_OFF: Int = 2
comptime TSR_PUSH_VEL_OFF: Int = 3
comptime TSR_PUSH_SRC_WH: Int = 4
comptime TSR_PUSH_DST_WH: Int = 5
comptime TSR_PUSH_ROW_BLEND: Int = 6
comptime TSR_PUSH_SHARP_JITTER: Int = 7
comptime TSR_PUSH_WORDS: Int = 8

comptime CULL_PUSH_IN_OFF: Int = 0
comptime CULL_PUSH_VIS_OFF: Int = 1
comptime CULL_PUSH_COUNT_OFF: Int = 2
comptime CULL_PUSH_INDIRECT_OFF: Int = 3
comptime CULL_PUSH_INSTANCE_COUNT: Int = 4
comptime CULL_PUSH_HIZ_OFF: Int = 5
comptime CULL_PUSH_HIZ_DIMS: Int = 6
comptime CULL_PUSH_SCREEN_WH: Int = 7
comptime CULL_PUSH_WORDS: Int = 8


struct Resources:
    """The ids the raster and post passes reach from everywhere."""
    var records: UInt32
    var constants: UInt32
    var textures: UInt32
    var voxels: UInt32
    var shadow_map: UInt32

    def __init__(out self, mut s: SpirvModule):
        self.records = s.buffer(BIND_RECORDS)
        self.constants = s.buffer(BIND_CONSTANTS)
        self.textures = s.buffer(BIND_TEXTURES)
        self.voxels = s.buffer(BIND_VOXELS)
        self.shadow_map = s.buffer(BIND_SHADOW_MAP)


struct Vec3Ids(Copyable, Movable):
    var x: UInt32
    var y: UInt32
    var z: UInt32

    def __init__(out self, x: UInt32, y: UInt32, z: UInt32):
        self.x = x
        self.y = y
        self.z = z


def divide_by_255(mut s: SpirvModule, value: UInt32) -> UInt32:
    """Exact floor(n / 255) for n below 65535, matching the CPU's integer division."""
    var high = s.shr(value, s.uint(8))
    return s.shr(s.iadd(s.iadd(value, high), s.uint(1)), s.uint(8))


def channel(mut s: SpirvModule, value: UInt32, shift: Int) -> UInt32:
    return s.bit_and(s.shr(value, s.uint(UInt32(shift))), s.uint(255))


def clamp_byte(mut s: SpirvModule, value: UInt32) -> UInt32:
    return s.smin(s.smax(value, s.uint(0)), s.uint(255))


def float_to_byte(mut s: SpirvModule, value: UInt32) -> UInt32:
    """Truncate toward zero and clamp to a byte, NaN giving 0."""
    return s.int_from_float(s.nclamp(value, s.float(0), s.float(255.0)))


def lerp_packed(mut s: SpirvModule, a: UInt32, b: UInt32, weight: UInt32) -> UInt32:
    """a * (256 - weight) / 256 + b * weight / 256 on all four bytes at once."""
    var mask = s.uint(0x00FF00FF)
    var eight = s.uint(8)
    var other = s.isub(s.uint(256), weight)
    var low = s.iadd(s.imul(s.bit_and(a, mask), other), s.imul(s.bit_and(b, mask), weight))
    low = s.bit_and(s.shr(low, eight), mask)
    var high_a = s.bit_and(s.shr(a, eight), mask)
    var high_b = s.bit_and(s.shr(b, eight), mask)
    var high = s.iadd(s.imul(high_a, other), s.imul(high_b, weight))
    high = s.bit_and(s.shr(high, eight), mask)
    return s.bit_or(s.shl(high, eight), low)


def absolute_int(mut s: SpirvModule, value: UInt32) -> UInt32:
    return s.smax(value, s.isub(s.uint(0), value))


def luma(mut s: SpirvModule, color: UInt32) -> UInt32:
    var red = s.imul(channel(s, color, 16), s.uint(54))
    var green = s.imul(channel(s, color, 8), s.uint(183))
    var blue = s.imul(channel(s, color, 0), s.uint(19))
    return s.shr(s.iadd(s.iadd(red, green), blue), s.uint(8))


def luma_float(mut s: SpirvModule, color: UInt32) -> UInt32:
    var r = s.float_from_uint(channel(s, color, 16))
    var g = s.float_from_uint(channel(s, color, 8))
    var b = s.float_from_uint(channel(s, color, 0))
    return s.fma(b, s.float(0.000447), s.fma(g, s.float(0.002302), s.fmul(r, s.float(0.001172))))


def field(mut s: SpirvModule, res: Resources, base: UInt32, dword: Int) -> UInt32:
    return s.load_word(res.records, s.iadd(base, s.uint(UInt32(dword))))


def plane(mut s: SpirvModule, res: Resources, base: UInt32, dword: Int, x: UInt32, y: UInt32) -> UInt32:
    var a = s.bitcast_float(field(s, res, base, dword))
    var b = s.bitcast_float(field(s, res, base, dword + 1))
    var c = s.bitcast_float(field(s, res, base, dword + 2))
    return s.fma(b, y, s.fma(a, x, c))


def constant_float(mut s: SpirvModule, res: Resources, byte_offset: Int) -> UInt32:
    return s.load_float(res.constants, s.uint(UInt32(byte_offset // 4)))


def constant_word(mut s: SpirvModule, res: Resources, byte_offset: Int) -> UInt32:
    return s.load_word(res.constants, s.uint(UInt32(byte_offset // 4)))


def constant_vector(mut s: SpirvModule, res: Resources, byte_offset: Int) -> Vec3Ids:
    var x = constant_float(s, res, byte_offset)
    var y = constant_float(s, res, byte_offset + 4)
    var z = constant_float(s, res, byte_offset + 8)
    return Vec3Ids(x, y, z)


def dot3(mut s: SpirvModule, a: Vec3Ids, b: Vec3Ids) -> UInt32:
    return s.fma(a.z, b.z, s.fma(a.y, b.y, s.fmul(a.x, b.x)))


def ign_noise(mut s: SpirvModule, x: UInt32, y: UInt32) -> UInt32:
    """Jorge Jimenez Interleaved Gradient Noise in [0, 1)."""
    var inner = s.fract(s.fma(x, s.float(0.06711056), s.fmul(y, s.float(0.00583715))))
    return s.fract(s.fmul(s.float(52.9829189), inner))


def _parabolic_sin(mut s: SpirvModule, phase: UInt32) -> UInt32:
    """Fast C1 parabolic sin(2*pi*phase) accurate to 99.8% using 5 float ops."""
    var f = s.fract(phase)
    var u = s.fma(f, s.float(2.0), s.float(-1.0))
    var abs_u = s.fabs(u)
    return s.fmul(s.fmul(s.float(4.0), u), s.fsub(s.float(1.0), abs_u))


def texel(mut s: SpirvModule, res: Resources, base: UInt32, x: UInt32, y: UInt32) -> UInt32:
    var width = field(s, res, base, RECORD_TEXTURE_WIDTH)
    var first = s.shr(field(s, res, base, RECORD_TEXTURE), s.uint(2))
    return s.load_word(res.textures, s.iadd(first, s.iadd(s.imul(y, width), x)))


def wrap(mut s: SpirvModule, value: UInt32, size: UInt32) -> UInt32:
    """An unsigned minimum with (value - size) folds value back into range."""
    return s.umin(s.isub(value, size), value)


def sample_bilinear(mut s: SpirvModule, res: Resources, base: UInt32, u: UInt32, v: UInt32) -> UInt32:
    """u and v are wrapped texel coordinates at mip 0."""
    var half = s.float(-0.5)
    var shifted_u = s.fadd(s.fadd(u, s.bitcast_float(field(s, res, base, RECORD_TEXTURE_FLOAT_WIDTH))), half)
    var shifted_v = s.fadd(s.fadd(v, s.bitcast_float(field(s, res, base, RECORD_TEXTURE_FLOAT_HEIGHT))), half)
    var fractions = s.float(256.0)
    var weight_x = s.saturate_uint(s.fmul(s.fract(shifted_u), fractions))
    var weight_y = s.saturate_uint(s.fmul(s.fract(shifted_v), fractions))
    var width = field(s, res, base, RECORD_TEXTURE_WIDTH)
    var height = s.iadd(field(s, res, base, RECORD_TEXTURE_MAX_Y), s.uint(1))
    var x0 = wrap(s, s.saturate_uint(shifted_u), width)
    var x1 = wrap(s, s.iadd(x0, s.uint(1)), width)
    var y0 = wrap(s, s.saturate_uint(shifted_v), height)
    var y1 = wrap(s, s.iadd(y0, s.uint(1)), height)
    var top = lerp_packed(s, texel(s, res, base, x0, y0), texel(s, res, base, x1, y0), weight_x)
    var bottom = lerp_packed(s, texel(s, res, base, x0, y1), texel(s, res, base, x1, y1), weight_x)
    return lerp_packed(s, top, bottom, weight_y)


def _mip_word_offset(mut s: SpirvModule, area0: UInt32, level: UInt32) -> UInt32:
    """Exact word offset of mip `level` (0..4) for 16-aligned textures."""
    var m1 = area0
    var m2 = s.iadd(m1, s.shr(area0, s.uint(2)))
    var m3 = s.iadd(m2, s.shr(area0, s.uint(4)))
    var m4 = s.iadd(m3, s.shr(area0, s.uint(6)))
    var r = s.select_u(s.ieq(level, s.uint(0)), s.uint(0), m1)
    r = s.select_u(s.ieq(level, s.uint(2)), m2, r)
    r = s.select_u(s.ieq(level, s.uint(3)), m3, r)
    r = s.select_u(s.uge(level, s.uint(4)), m4, r)
    return r


def _sample_mip_bilinear(
    mut s: SpirvModule,
    res: Resources,
    first0: UInt32,
    w0: UInt32,
    h0: UInt32,
    area0: UInt32,
    u_norm: UInt32,
    v_norm: UInt32,
    level: UInt32,
) -> UInt32:
    """Bilinear sample from mip `level` (0..4) at normalized [0,1) UVs."""
    var mw = s.umax(s.shr(w0, level), s.uint(1))
    var mh = s.umax(s.shr(h0, level), s.uint(1))
    var mwf = s.float_from_uint(mw)
    var mhf = s.float_from_uint(mh)
    var u_wr = s.fract(u_norm)
    var v_wr = s.fract(v_norm)
    var su = s.fadd(s.fma(u_wr, mwf, mwf), s.float(-0.5))
    var sv = s.fadd(s.fma(v_wr, mhf, mhf), s.float(-0.5))
    var wx = s.saturate_uint(s.fmul(s.fract(su), s.float(256.0)))
    var wy = s.saturate_uint(s.fmul(s.fract(sv), s.float(256.0)))
    var x0 = wrap(s, s.saturate_uint(su), mw)
    var x1 = wrap(s, s.iadd(x0, s.uint(1)), mw)
    var y0 = wrap(s, s.saturate_uint(sv), mh)
    var y1 = wrap(s, s.iadd(y0, s.uint(1)), mh)
    var base_m = s.iadd(first0, _mip_word_offset(s, area0, level))
    var r0 = s.iadd(base_m, s.imul(y0, mw))
    var r1 = s.iadd(base_m, s.imul(y1, mw))
    var c00 = s.load_word(res.textures, s.iadd(r0, x0))
    var c10 = s.load_word(res.textures, s.iadd(r0, x1))
    var c01 = s.load_word(res.textures, s.iadd(r1, x0))
    var c11 = s.load_word(res.textures, s.iadd(r1, x1))
    var top = lerp_packed(s, c00, c10, wx)
    var bot = lerp_packed(s, c01, c11, wx)
    return lerp_packed(s, top, bot, wy)


def _sample_trilinear(
    mut s: SpirvModule,
    res: Resources,
    first0: UInt32,
    w0: UInt32,
    h0: UInt32,
    area0: UInt32,
    u_norm: UInt32,
    v_norm: UInt32,
    mip_lo: UInt32,
    mip_hi: UInt32,
    lod_frac: UInt32,
) -> UInt32:
    var s_lo = _sample_mip_bilinear(s, res, first0, w0, h0, area0, u_norm, v_norm, mip_lo)
    var out_c = s.uvar(s_lo)
    s.begin_if(s.logical_and(s.ine(mip_lo, mip_hi), s.ine(lod_frac, s.uint(0))))
    var s_hi = _sample_mip_bilinear(s, res, first0, w0, h0, area0, u_norm, v_norm, mip_hi)
    s.store(out_c, lerp_packed(s, s_lo, s_hi, lod_frac))
    s.end_if()
    return s.load_u(out_c)


def sample_anisotropic_mipmapped(
    mut s: SpirvModule,
    res: Resources,
    base: UInt32,
    u_norm: UInt32,
    v_norm: UInt32,
    du_dx: UInt32,
    dv_dx: UInt32,
    du_dy: UInt32,
    dv_dy: UInt32,
    out_lod: UInt32,
) -> UInt32:
    """5-level trilinear + up-to-4x major-axis anisotropic filtering using exact plane derivatives."""
    var first0 = s.shr(field(s, res, base, RECORD_TEXTURE), s.uint(2))
    var w0 = field(s, res, base, RECORD_TEXTURE_WIDTH)
    var h0 = s.iadd(field(s, res, base, RECORD_TEXTURE_MAX_Y), s.uint(1))
    var area0 = s.imul(w0, h0)
    var wf = s.bitcast_float(field(s, res, base, RECORD_TEXTURE_FLOAT_WIDTH))
    var hf = s.bitcast_float(field(s, res, base, RECORD_TEXTURE_FLOAT_HEIGHT))

    var jx_u = s.fmul(du_dx, wf)
    var jx_v = s.fmul(dv_dx, hf)
    var jy_u = s.fmul(du_dy, wf)
    var jy_v = s.fmul(dv_dy, hf)
    var len_x2 = s.fma(jx_v, jx_v, s.fmul(jx_u, jx_u))
    var len_y2 = s.fma(jy_v, jy_v, s.fmul(jy_u, jy_u))
    var x_major = s.fge(len_x2, len_y2)
    var major2 = s.nmax(len_x2, len_y2)
    var minor2 = s.nmax(s.nmin(len_x2, len_y2), s.float(0.000001))
    var maj_du = s.select_f(x_major, du_dx, du_dy)
    var maj_dv = s.select_f(x_major, dv_dx, dv_dy)

    var aniso4 = s.fgt(major2, s.fmul(minor2, s.float(3.2)))
    var aniso2 = s.fgt(major2, s.fmul(minor2, s.float(1.5)))
    var eff_foot2 = s.select_f(
        aniso4,
        s.nmax(minor2, s.fmul(major2, s.float(0.0625))),
        s.select_f(aniso2, s.nmax(minor2, s.fmul(major2, s.float(0.25))), major2),
    )
    var lod_raw = s.fmul(s.float(0.5), s.log2(s.nmax(eff_foot2, s.float(1.0))))
    var lod = s.nclamp(lod_raw, s.float(0.0), s.float(4.0))
    s.store(out_lod, lod)
    var mip_lo = s.umin(s.saturate_uint(lod), s.uint(4))
    var mip_hi = s.umin(s.iadd(mip_lo, s.uint(1)), s.uint(4))
    var lod_frac = s.saturate_uint(s.fmul(s.fsub(lod, s.float_from_uint(mip_lo)), s.float(256.0)))

    var result = s.uvar(s.uint(0xFFFFFFFF))
    s.begin_if(aniso4)
    var c0 = _sample_trilinear(
        s, res, first0, w0, h0, area0,
        s.fma(maj_du, s.float(-0.375), u_norm), s.fma(maj_dv, s.float(-0.375), v_norm),
        mip_lo, mip_hi, lod_frac,
    )
    var c1 = _sample_trilinear(
        s, res, first0, w0, h0, area0,
        s.fma(maj_du, s.float(-0.125), u_norm), s.fma(maj_dv, s.float(-0.125), v_norm),
        mip_lo, mip_hi, lod_frac,
    )
    var c2 = _sample_trilinear(
        s, res, first0, w0, h0, area0,
        s.fma(maj_du, s.float(0.125), u_norm), s.fma(maj_dv, s.float(0.125), v_norm),
        mip_lo, mip_hi, lod_frac,
    )
    var c3 = _sample_trilinear(
        s, res, first0, w0, h0, area0,
        s.fma(maj_du, s.float(0.375), u_norm), s.fma(maj_dv, s.float(0.375), v_norm),
        mip_lo, mip_hi, lod_frac,
    )
    var pair0 = lerp_packed(s, c0, c1, s.uint(128))
    var pair1 = lerp_packed(s, c2, c3, s.uint(128))
    s.store(result, lerp_packed(s, pair0, pair1, s.uint(128)))
    s.begin_else()
    s.begin_if(aniso2)
    var b0 = _sample_trilinear(
        s, res, first0, w0, h0, area0,
        s.fma(maj_du, s.float(-0.25), u_norm), s.fma(maj_dv, s.float(-0.25), v_norm),
        mip_lo, mip_hi, lod_frac,
    )
    var b1 = _sample_trilinear(
        s, res, first0, w0, h0, area0,
        s.fma(maj_du, s.float(0.25), u_norm), s.fma(maj_dv, s.float(0.25), v_norm),
        mip_lo, mip_hi, lod_frac,
    )
    s.store(result, lerp_packed(s, b0, b1, s.uint(128)))
    s.begin_else()
    s.store(result, _sample_trilinear(s, res, first0, w0, h0, area0, u_norm, v_norm, mip_lo, mip_hi, lod_frac))
    s.end_if()
    s.end_if()
    return s.load_u(result)


def _sample_csm_pcf(
    mut s: SpirvModule,
    res: Resources,
    world: Vec3Ids,
    normal: Vec3Ids,
    to_key: Vec3Ids,
    px_x: UInt32,
    px_y: UInt32,
) -> UInt32:
    """2-Cascade Shadow Map lookup with normal-offset slope bias + 8-tap IGN-rotated Poisson PCF."""
    var csm_on = constant_word(s, res, CONSTANT_CSM + 72)
    var vis = s.fvar(s.float(1.0))
    s.begin_if(s.ine(csm_on, s.uint(0)))
    var csm_right = constant_vector(s, res, CONSTANT_CSM)
    var csm_up = constant_vector(s, res, CONSTANT_CSM + 12)
    var c0_center = constant_vector(s, res, CONSTANT_CSM + 24)
    var c0_inv_half = constant_float(s, res, CONSTANT_CSM + 36)
    var c0_z_bias = constant_float(s, res, CONSTANT_CSM + 40)
    var c0_z_scale = constant_float(s, res, CONSTANT_CSM + 44)
    var c1_center = constant_vector(s, res, CONSTANT_CSM + 48)
    var c1_inv_half = constant_float(s, res, CONSTANT_CSM + 60)
    var c1_z_bias = constant_float(s, res, CONSTANT_CSM + 64)
    var c1_z_scale = constant_float(s, res, CONSTANT_CSM + 68)

    var rel0 = Vec3Ids(
        s.fsub(world.x, c0_center.x),
        s.fsub(world.y, c0_center.y),
        s.fsub(world.z, c0_center.z),
    )
    var u0 = s.fmul(dot3(s, rel0, csm_right), c0_inv_half)
    var v0 = s.fmul(dot3(s, rel0, csm_up), c0_inv_half)
    var max_uv0 = s.nmax(s.fabs(u0), s.fabs(v0))
    var use_c0 = s.flt(max_uv0, s.float(0.92))

    var rel1 = Vec3Ids(
        s.fsub(world.x, c1_center.x),
        s.fsub(world.y, c1_center.y),
        s.fsub(world.z, c1_center.z),
    )
    var u1 = s.fmul(dot3(s, rel1, csm_right), c1_inv_half)
    var v1 = s.fmul(dot3(s, rel1, csm_up), c1_inv_half)
    var max_uv1 = s.nmax(s.fabs(u1), s.fabs(v1))
    var in_any = s.logical_or(use_c0, s.flt(max_uv1, s.float(0.96)))

    s.begin_if(in_any)
    var cx = s.select_f(use_c0, c0_center.x, c1_center.x)
    var cy = s.select_f(use_c0, c0_center.y, c1_center.y)
    var cz = s.select_f(use_c0, c0_center.z, c1_center.z)
    var inv_h = s.select_f(use_c0, c0_inv_half, c1_inv_half)
    var z_bias = s.select_f(use_c0, c0_z_bias, c1_z_bias)
    var z_scale = s.select_f(use_c0, c0_z_scale, c1_z_scale)
    var casc_base = s.select_u(use_c0, s.uint(0), s.uint(UInt32(CSM_CASCADE_WORDS)))

    var ndl = s.nclamp(dot3(s, normal, to_key), s.float(0.0), s.float(1.0))
    var slope = s.fsqrt(s.nmax(s.fsub(s.float(1.0), s.fmul(ndl, ndl)), s.float(0.0)))
    var tex_world = s.fdiv(s.float(2.0), s.fmul(s.float(Float32(CSM_RES)), s.nmax(inv_h, s.float(0.0001))))
    var n_off = s.fmul(tex_world, s.fma(slope, s.float(2.1), s.float(1.35)))
    var l_off = s.fmul(tex_world, s.float(1.65))
    var bx = s.fma(to_key.x, l_off, s.fma(normal.x, n_off, world.x))
    var by = s.fma(to_key.y, l_off, s.fma(normal.y, n_off, world.y))
    var bz = s.fma(to_key.z, l_off, s.fma(normal.z, n_off, world.z))
    var b_rel = Vec3Ids(s.fsub(bx, cx), s.fsub(by, cy), s.fsub(bz, cz))
    var b_pos = Vec3Ids(bx, by, bz)

    var su = s.fmul(s.fma(s.fmul(dot3(s, b_rel, csm_right), inv_h), s.float(0.5), s.float(0.5)), s.float(Float32(CSM_RES)))
    var sv = s.fmul(s.fsub(s.float(0.5), s.fmul(s.fmul(dot3(s, b_rel, csm_up), inv_h), s.float(0.5))), s.float(Float32(CSM_RES)))
    var z_rec = s.fma(s.fsub(dot3(s, b_pos, to_key), z_bias), z_scale, s.float(0.5))

    s.begin_if(s.logical_and(s.fgt(z_rec, s.float(0.002)), s.flt(z_rec, s.float(0.998))))
    var z_u = s.saturate_uint(s.fmul(z_rec, s.float(16777215.0)))
    var ign = ign_noise(s, px_x, px_y)
    var s_a = _parabolic_sin(s, ign)
    var c_a = _parabolic_sin(s, s.fadd(ign, s.float(0.25)))
    var rad = s.select_f(use_c0, s.float(1.65), s.float(1.35))
    var ca_r = s.fmul(c_a, rad)
    var sa_r = s.fmul(s_a, rad)

    var p_x = List[Float32]()
    var p_y = List[Float32]()
    p_x.append(-0.707)
    p_y.append(-0.354)
    p_x.append(0.354)
    p_y.append(-0.822)
    p_x.append(-0.245)
    p_y.append(0.866)
    p_x.append(0.854)
    p_y.append(0.275)
    p_x.append(-0.882)
    p_y.append(0.312)
    p_x.append(0.215)
    p_y.append(0.295)
    p_x.append(-0.185)
    p_y.append(-0.310)
    p_x.append(0.562)
    p_y.append(-0.218)

    var lit_sum = s.fvar(s.float(0.0))
    var max_csm = s.uint(UInt32(CSM_RES - 1))
    var res_u = s.uint(UInt32(CSM_RES))
    for tap in range(8):
        var ox = s.float(p_x[tap])
        var oy = s.float(p_y[tap])
        var rx = s.fsub(s.fmul(ox, ca_r), s.fmul(oy, sa_r))
        var ry = s.fadd(s.fmul(ox, sa_r), s.fmul(oy, ca_r))
        var tx = s.umin(s.saturate_uint(s.fadd(su, rx)), max_csm)
        var ty = s.umin(s.saturate_uint(s.fadd(sv, ry)), max_csm)
        var map_z = s.load_word(res.shadow_map, s.iadd(casc_base, s.iadd(s.imul(ty, res_u), tx)))
        var tap_lit = s.select_f(s.ult(z_u, map_z), s.float(0.0), s.float(0.125))
        s.store(lit_sum, s.fadd(s.load_f(lit_sum), tap_lit))

    var border_fade = s.nclamp(s.fmul(s.fsub(max_uv1, s.float(0.86)), s.float(10.0)), s.float(0.0), s.float(1.0))
    var eff_fade = s.select_f(use_c0, s.float(0.0), border_fade)
    var raw_vis = s.load_f(lit_sum)
    s.store(vis, s.fma(s.fsub(s.float(1.0), raw_vis), eff_fade, raw_vis))
    s.end_if()
    s.end_if()
    s.end_if()
    return s.load_f(vis)


def _sample_volumetric_shadow(mut s: SpirvModule, res: Resources, pos: Vec3Ids, to_key: Vec3Ids) -> UInt32:
    """Fast 1-tap CSM + analytical sphere visibility for volumetric light shafts."""
    var vis = s.fvar(s.float(1.0))
    var csm_on = constant_word(s, res, CONSTANT_CSM + 72)
    s.begin_if(s.ine(csm_on, s.uint(0)))
    var csm_right = constant_vector(s, res, CONSTANT_CSM)
    var csm_up = constant_vector(s, res, CONSTANT_CSM + 12)
    var c0_center = constant_vector(s, res, CONSTANT_CSM + 24)
    var c0_inv_half = constant_float(s, res, CONSTANT_CSM + 36)
    var c0_z_bias = constant_float(s, res, CONSTANT_CSM + 40)
    var c0_z_scale = constant_float(s, res, CONSTANT_CSM + 44)
    var c1_center = constant_vector(s, res, CONSTANT_CSM + 48)
    var c1_inv_half = constant_float(s, res, CONSTANT_CSM + 60)
    var c1_z_bias = constant_float(s, res, CONSTANT_CSM + 64)
    var c1_z_scale = constant_float(s, res, CONSTANT_CSM + 68)

    var rel0 = Vec3Ids(s.fsub(pos.x, c0_center.x), s.fsub(pos.y, c0_center.y), s.fsub(pos.z, c0_center.z))
    var u0 = s.fmul(dot3(s, rel0, csm_right), c0_inv_half)
    var v0 = s.fmul(dot3(s, rel0, csm_up), c0_inv_half)
    var use_c0 = s.flt(s.nmax(s.fabs(u0), s.fabs(v0)), s.float(0.93))
    var rel1 = Vec3Ids(s.fsub(pos.x, c1_center.x), s.fsub(pos.y, c1_center.y), s.fsub(pos.z, c1_center.z))
    var u1 = s.fmul(dot3(s, rel1, csm_right), c1_inv_half)
    var v1 = s.fmul(dot3(s, rel1, csm_up), c1_inv_half)
    var in_any = s.logical_or(use_c0, s.flt(s.nmax(s.fabs(u1), s.fabs(v1)), s.float(0.96)))

    s.begin_if(in_any)
    var u_s = s.select_f(use_c0, u0, u1)
    var v_s = s.select_f(use_c0, v0, v1)
    var z_b = s.select_f(use_c0, c0_z_bias, c1_z_bias)
    var z_sc = s.select_f(use_c0, c0_z_scale, c1_z_scale)
    var c_base = s.select_u(use_c0, s.uint(0), s.uint(UInt32(CSM_CASCADE_WORDS)))
    var tx = s.umin(
        s.saturate_uint(s.fmul(s.fma(u_s, s.float(0.5), s.float(0.5)), s.float(Float32(CSM_RES)))),
        s.uint(UInt32(CSM_RES - 1)),
    )
    var ty = s.umin(
        s.saturate_uint(s.fmul(s.fsub(s.float(0.5), s.fmul(v_s, s.float(0.5))), s.float(Float32(CSM_RES)))),
        s.uint(UInt32(CSM_RES - 1)),
    )
    var z_rec = s.fma(s.fsub(dot3(s, pos, to_key), z_b), z_sc, s.float(0.502))
    var z_u = s.saturate_uint(s.fmul(s.nclamp(z_rec, s.float(0.0), s.float(1.0)), s.float(16777215.0)))
    var map_z = s.load_word(res.shadow_map, s.iadd(c_base, s.iadd(s.imul(ty, s.uint(UInt32(CSM_RES))), tx)))
    s.begin_if(s.ult(z_u, map_z))
    s.store(vis, s.float(0.08))
    s.end_if()
    s.end_if()
    s.end_if()

    var sphere_count = constant_word(s, res, CONSTANT_WORLD + 12)
    var sp = s.uvar(s.uint(0))
    var one = s.uint(1)
    s.begin_loop()
    var idx = s.load_u(sp)
    s.break_if(s.uge(idx, sphere_count))
    s.store(sp, s.iadd(idx, one))
    var word = s.iadd(s.uint(UInt32(CONSTANT_SPHERES // 4)), s.shl(idx, s.uint(2)))
    var sx = s.fsub(s.load_float(res.constants, word), pos.x)
    var sy = s.fsub(s.load_float(res.constants, s.iadd(word, one)), pos.y)
    var sz = s.fsub(s.load_float(res.constants, s.iadd(word, s.uint(2))), pos.z)
    var r2 = s.load_float(res.constants, s.iadd(word, s.uint(3)))
    var c_vec = Vec3Ids(sx, sy, sz)
    var along = dot3(s, c_vec, to_key)
    var disc = s.fadd(s.fsub(s.fmul(along, along), dot3(s, c_vec, c_vec)), r2)
    s.begin_if(s.logical_and(s.flt(s.float(0.0), along), s.flt(s.float(0.0), disc)))
    s.store(vis, s.fmul(s.load_f(vis), s.float(0.15)))
    s.end_if()
    s.end_loop()
    return s.load_f(vis)


def trace_shadow(
    mut s: SpirvModule,
    res: Resources,
    world: Vec3Ids,
    normal: Vec3Ids,
    to_key: Vec3Ids,
    effects: UInt32,
    px_x: UInt32,
    px_y: UInt32,
) -> UInt32:
    """1 where the key light gets through, 0 where CSM, a sphere, or a voxel blocks it."""
    var one = s.uint(1)
    var shadow = s.fvar(s.float(1.0))

    s.begin_if(s.ine(s.bit_and(effects, s.uint(EFFECT_CSM)), s.uint(0)))
    s.store(shadow, _sample_csm_pcf(s, res, world, normal, to_key, px_x, px_y))
    s.end_if()

    var sphere_count = constant_word(s, res, CONSTANT_WORLD + 12)
    var sphere = s.uvar(s.uint(0))
    s.begin_loop()
    var index = s.load_u(sphere)
    s.break_if(s.uge(index, sphere_count))
    s.store(sphere, s.iadd(index, one))
    var word = s.iadd(s.uint(UInt32(CONSTANT_SPHERES // 4)), s.shl(index, s.uint(2)))
    var sphere_x = s.fsub(s.load_float(res.constants, word), world.x)
    var sphere_y = s.fsub(s.load_float(res.constants, s.iadd(word, one)), world.y)
    var sphere_z = s.fsub(s.load_float(res.constants, s.iadd(word, s.uint(2))), world.z)
    var radius_squared = s.load_float(res.constants, s.iadd(word, s.uint(3)))
    var center = Vec3Ids(sphere_x, sphere_y, sphere_z)
    var along = dot3(s, center, to_key)
    var discriminant = s.fadd(s.fsub(s.fmul(along, along), dot3(s, center, center)), radius_squared)
    var zero = s.float(0)
    s.begin_if(s.logical_and(s.flt(zero, along), s.flt(zero, discriminant)))
    var raw_pen = s.nclamp(
        s.fdiv(s.fmul(discriminant, s.float(4.0)), s.fadd(radius_squared, s.float(0.0001))),
        zero,
        s.float(1.0),
    )
    var dist_ratio = s.fdiv(s.fmul(along, along), s.fma(radius_squared, s.float(36.0), s.float(0.001)))
    var dist_atten = s.fdiv(s.float(1.0), s.fadd(s.float(1.0), dist_ratio))
    var penumbra = s.fmul(raw_pen, dist_atten)
    s.store(shadow, s.nmin(s.load_f(shadow), s.fsub(s.float(1.0), penumbra)))
    s.end_if()
    s.end_loop()

    # Start just off the surface, then walk toward the light half a voxel at a time.
    var half = s.float(0.5)
    var ray_x = s.fvar(s.fadd(world.x, s.fmul(normal.x, half)))
    var ray_y = s.fvar(s.fadd(world.y, s.fmul(normal.y, half)))
    var ray_z = s.fvar(s.fadd(world.z, s.fmul(normal.z, half)))
    var max_x = constant_word(s, res, CONSTANT_WORLD)
    var max_y = constant_word(s, res, CONSTANT_WORLD + 4)
    var max_z = constant_word(s, res, CONSTANT_WORLD + 8)
    var row = constant_word(s, res, CONSTANT_WORLD + 28)
    var step = s.uvar(s.uint(0))
    var fine = s.uint(UInt32(SHADOW_FINE_STEPS))
    var total = s.uint(UInt32(SHADOW_FINE_STEPS + SHADOW_COARSE_STEPS))
    s.begin_if(s.logical_and(s.fgt(s.load_f(shadow), s.float(0)), s.ine(row, s.uint(0))))
    s.begin_loop()
    var current = s.load_u(step)
    s.break_if(s.uge(current, total))
    s.store(step, s.iadd(current, one))
    s.begin_if(s.ult(current, fine))
    s.store(ray_x, s.fma(to_key.x, half, s.load_f(ray_x)))
    s.store(ray_y, s.fma(to_key.y, half, s.load_f(ray_y)))
    s.store(ray_z, s.fma(to_key.z, half, s.load_f(ray_z)))
    s.begin_else()
    s.store(ray_x, s.fadd(to_key.x, s.load_f(ray_x)))
    s.store(ray_y, s.fadd(to_key.y, s.load_f(ray_y)))
    s.store(ray_z, s.fadd(to_key.z, s.load_f(ray_z)))
    s.end_if()
    var cell_x = s.umin(s.saturate_uint(s.load_f(ray_x)), max_x)
    var cell_y = s.umin(s.saturate_uint(s.load_f(ray_y)), max_y)
    var cell_z = s.umin(s.saturate_uint(s.load_f(ray_z)), max_z)
    var address = s.iadd(s.imul(s.iadd(s.imul(cell_y, row), cell_z), row), cell_x)
    var loaded = s.load_word(res.voxels, s.shr(address, s.uint(2)))
    var bits = s.shl(s.bit_and(address, s.uint(3)), s.uint(3))
    var voxel = s.bit_and(s.shr(loaded, bits), s.uint(255))
    s.begin_if(s.ine(voxel, s.uint(0)))
    s.store(shadow, s.float(0))
    s.end_if()
    s.break_if(s.ine(voxel, s.uint(0)))
    s.end_loop()
    s.end_if()
    return s.load_f(shadow)


def light_float(mut s: SpirvModule, res: Resources, word: UInt32, member: Int) -> UInt32:
    return s.load_float(res.constants, s.iadd(word, s.uint(UInt32(member))))


def accumulate_ggx_specular(
    mut s: SpirvModule,
    normal: Vec3Ids,
    tan_dir: Vec3Ids,
    bit_dir: Vec3Ids,
    view_dir: Vec3Ids,
    light_dir: Vec3Ids,
    light_color: Vec3Ids,
    n_dot_v: UInt32,
    n_dot_l: UInt32,
    ax: UInt32,
    ay: UInt32,
    ms_comp: UInt32,
    k_smith: UInt32,
    one_minus_k: UInt32,
    f0_r: UInt32,
    f0_g: UInt32,
    f0_b: UInt32,
    spec_r: UInt32,
    spec_g: UInt32,
    spec_b: UInt32,
):
    """Elliptical Anisotropic GGX microfacet specular + Kulla-Conty multiscatter + Schlick Fresnel."""
    var hx = s.fadd(view_dir.x, light_dir.x)
    var hy = s.fadd(view_dir.y, light_dir.y)
    var hz = s.fadd(view_dir.z, light_dir.z)
    var h_raw = Vec3Ids(hx, hy, hz)
    var inv_h = s.inverse_sqrt(s.fadd(dot3(s, h_raw, h_raw), s.float(0.00001)))
    var half_dir = Vec3Ids(s.fmul(hx, inv_h), s.fmul(hy, inv_h), s.fmul(hz, inv_h))
    var n_dot_h = s.nmax(dot3(s, normal, half_dir), s.float(0.0))
    var t_dot_h = dot3(s, tan_dir, half_dir)
    var b_dot_h = dot3(s, bit_dir, half_dir)
    var v_dot_h = s.nclamp(dot3(s, view_dir, half_dir), s.float(0.0), s.float(1.0))

    var tx_h = s.fdiv(t_dot_h, ax)
    var by_h = s.fdiv(b_dot_h, ay)
    var ellip = s.fma(n_dot_h, n_dot_h, s.fma(by_h, by_h, s.fmul(tx_h, tx_h)))
    var ax_ay_pi = s.fmul(s.float(3.14159265), s.fmul(ax, ay))
    var d_ggx = s.fdiv(s.float(1.0), s.fadd(s.fmul(ax_ay_pi, s.fmul(ellip, ellip)), s.float(0.0001)))

    var gv = s.fma(n_dot_v, one_minus_k, k_smith)
    var gl = s.fma(n_dot_l, one_minus_k, k_smith)
    var vis = s.fdiv(s.float(0.25), s.fadd(s.fmul(gv, gl), s.float(0.0001)))
    var lobe = s.fmul(s.nmin(s.fmul(s.fmul(d_ggx, vis), n_dot_l), s.float(6.0)), ms_comp)

    var om = s.fsub(s.float(1.0), v_dot_h)
    var om2 = s.fmul(om, om)
    var om5 = s.fmul(s.fmul(om2, om2), om)
    var fr = s.fma(s.fsub(s.float(255.0), f0_r), om5, f0_r)
    var fg = s.fma(s.fsub(s.float(255.0), f0_g), om5, f0_g)
    var fb = s.fma(s.fsub(s.float(255.0), f0_b), om5, f0_b)

    s.store(spec_r, s.fma(s.fmul(lobe, light_color.x), fr, s.load_f(spec_r)))
    s.store(spec_g, s.fma(s.fmul(lobe, light_color.y), fg, s.load_f(spec_g)))
    s.store(spec_b, s.fma(s.fmul(lobe, light_color.z), fb, s.load_f(spec_b)))


def _encode_octahedral_byte(mut s: SpirvModule, val: UInt32) -> UInt32:
    return clamp_byte(s, s.int_from_float(s.fma(s.fma(val, s.float(0.5), s.float(0.5)), s.float(254.0), s.float(1.5))))


def _decode_octahedral_normal(mut s: SpirvModule, gbuf: UInt32) -> Vec3Ids:
    var ox_u = channel(s, gbuf, 0)
    var oz_u = channel(s, gbuf, 8)
    var ox0 = s.fma(s.fmul(s.fsub(s.float_from_uint(ox_u), s.float(1.0)), s.float(0.003937008)), s.float(2.0), s.float(-1.0))
    var oz0 = s.fma(s.fmul(s.fsub(s.float_from_uint(oz_u), s.float(1.0)), s.float(0.003937008)), s.float(2.0), s.float(-1.0))
    var ny = s.fsub(s.fsub(s.float(1.0), s.fabs(ox0)), s.fabs(oz0))
    var below = s.flt(ny, s.float(0.0))
    var sx = s.select_f(s.fge(ox0, s.float(0.0)), s.float(1.0), s.float(-1.0))
    var sz = s.select_f(s.fge(oz0, s.float(0.0)), s.float(1.0), s.float(-1.0))
    var nx_f = s.select_f(below, s.fmul(s.fsub(s.float(1.0), s.fabs(oz0)), sx), ox0)
    var nz_f = s.select_f(below, s.fmul(s.fsub(s.float(1.0), s.fabs(ox0)), sz), oz0)
    var raw = Vec3Ids(nx_f, ny, nz_f)
    var inv_l = s.inverse_sqrt(s.fadd(dot3(s, raw, raw), s.float(0.000001)))
    return Vec3Ids(s.fmul(nx_f, inv_l), s.fmul(ny, inv_l), s.fmul(nz_f, inv_l))


def _gyroid3d_unit(mut s: SpirvModule, px: UInt32, py: UInt32, pz: UInt32) -> UInt32:
    """Fast C1 3D gyroid field in [0, 1] using 6 parabolic sin/cos ops (zero lattice blowup)."""
    var q = s.float(0.25)
    var sx = _parabolic_sin(s, px)
    var sy = _parabolic_sin(s, py)
    var sz = _parabolic_sin(s, pz)
    var cx = _parabolic_sin(s, s.fadd(px, q))
    var cy = _parabolic_sin(s, s.fadd(py, q))
    var cz = _parabolic_sin(s, s.fadd(pz, q))
    var g = s.fma(sz, cy, s.fma(sy, cx, s.fmul(sx, cz)))
    return s.fma(g, s.float(0.16666667), s.float(0.5))


def _cloud_density_3d(
    mut s: SpirvModule,
    res: Resources,
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
) -> UInt32:
    """3-Octave Gyroid-Worley FBM cumulus/cumulonimbus cloud density in [0, 1]."""
    var base_y = constant_float(s, res, CONSTANT_V9_CLOUDS)
    var top_y = constant_float(s, res, CONSTANT_V9_CLOUDS + 4)
    var coverage = constant_float(s, res, CONSTANT_V9_CLOUDS + 8)
    var wind_x = constant_float(s, res, CONSTANT_V9_CLOUDS + 16)
    var wind_z = constant_float(s, res, CONSTANT_V9_CLOUDS + 20)
    var scale = s.fmul(constant_float(s, res, CONSTANT_V9_CLOUDS + 28), s.float(0.22))
    var v9_time = constant_float(s, res, CONSTANT_V9_CLOUDS + 44)
    var span_y = s.nmax(s.fsub(top_y, base_y), s.float(10.0))
    var h_t = s.nclamp(s.fdiv(s.fsub(wy, base_y), span_y), s.float(0.0), s.float(1.0))
    var bot_m = s.nclamp(s.fmul(h_t, s.float(4.2)), s.float(0.0), s.float(1.0))
    var top_m = s.nclamp(s.fmul(s.fsub(s.float(1.0), h_t), s.float(2.1)), s.float(0.0), s.float(1.0))
    var anvil = s.fmul(bot_m, top_m)

    var ax0 = s.fmul(s.fma(wind_x, v9_time, wx), scale)
    var ay0 = s.fmul(wy, s.fmul(scale, s.float(1.35)))
    var az0 = s.fmul(s.fma(wind_z, v9_time, wz), scale)
    var n0 = _gyroid3d_unit(
        s,
        s.fadd(ax0, s.float(0.11)),
        s.fadd(ay0, s.float(0.23)),
        s.fadd(az0, s.float(0.37)),
    )
    var n1 = _gyroid3d_unit(
        s,
        s.fma(ax0, s.float(2.17), s.float(0.41)),
        s.fma(ay0, s.float(2.17), s.float(0.19)),
        s.fma(az0, s.float(2.17), s.float(0.53)),
    )
    var worley1 = s.fsub(s.float(1.0), s.fabs(s.fma(n1, s.float(2.0), s.float(-1.0))))
    var n2 = _gyroid3d_unit(
        s,
        s.fma(ax0, s.float(4.43), s.float(0.67)),
        s.fma(ay0, s.float(4.43), s.float(0.31)),
        s.fma(az0, s.float(4.43), s.float(0.13)),
    )
    var fbm = s.fma(n2, s.float(0.16), s.fma(worley1, s.float(0.30), s.fmul(n0, s.float(0.54))))
    var thresh = s.fma(s.fsub(s.float(1.0), coverage), s.float(0.68), s.float(0.10))
    return s.nclamp(s.fmul(s.fsub(s.fmul(fbm, anvil), thresh), s.float(2.8)), s.float(0.0), s.float(1.0))


def per_pixel_light(
    mut s: SpirvModule,
    res: Resources,
    tiles: UInt32,
    tile_index: UInt32,
    base: UInt32,
    flags: UInt32,
    x: UInt32,
    y: UInt32,
    depth: UInt32,
    prev_nearest_iz: UInt32,
    ray: Vec3Ids,
    camera: Vec3Ids,
    cam_right: Vec3Ids,
    cam_up: Vec3Ids,
    inverse_focal: UInt32,
    to_key: Vec3Ids,
    key_color: Vec3Ids,
    effects: UInt32,
    tile_lights: UInt32,
    u_norm: UInt32,
    v_norm: UInt32,
    du_dx: UInt32,
    dv_dx: UInt32,
    du_dy: UInt32,
    dv_dy: UInt32,
    tex_lod: UInt32,
    base_r_var: UInt32,
    base_g_var: UInt32,
    base_b_var: UInt32,
    light_r: UInt32,
    light_g: UInt32,
    light_b: UInt32,
    spec_r: UInt32,
    spec_g: UInt32,
    spec_b: UInt32,
    out_ox: UInt32,
    out_oz: UInt32,
    out_rough: UInt32,
    out_refl: UInt32,
):
    """Key and 3D-clustered local/tube lights per pixel, with Cotangent Micro-Normal/ORM detail,
    Anisotropic Brushed-Metal GGX + Kulla-Conty Multiscattering, Karis Split-Sum IBL, 2-Cascade PCF,
    and Trochoidal Gerstner Ocean Jacobians + Wave-Crest SSS + Depth-Buffer Shoreline Surf Foam."""
    var world = Vec3Ids(s.fma(ray.x, depth, camera.x), s.fma(ray.y, depth, camera.y), s.fma(ray.z, depth, camera.z))
    var raw_x = plane(s, res, base, RECORD_NORMAL, x, y)
    var raw_y = plane(s, res, base, RECORD_NORMAL + 3, x, y)
    var raw_z = plane(s, res, base, RECORD_NORMAL + 6, x, y)
    var occ_var = s.fvar(plane(s, res, base, RECORD_AMBIENT_OCCLUSION, x, y))

    var rx_var = s.fvar(raw_x)
    var ry_var = s.fvar(raw_y)
    var rz_var = s.fvar(raw_z)
    var has_pbr = s.ine(s.bit_and(flags, s.uint(PBR_FLAG)), s.uint(0))
    var base_rough = s.nclamp(s.bitcast_float(field(s, res, base, RECORD_ROUGHNESS)), s.float(0.06), s.float(1.0))
    var rough_var = s.fvar(base_rough)
    var water_crest_var = s.fvar(s.float(0.0))
    var water_near_var = s.fvar(s.float(0.0))
    var decal_emit_r = s.fvar(s.float(0.0))
    var decal_emit_g = s.fvar(s.float(0.0))
    var decal_emit_b = s.fvar(s.float(0.0))

    var is_water = s.ine(s.bit_and(flags, s.uint(WATER_WAVE_FLAG)), s.uint(0))
    s.begin_if(is_water)
    var water_time = constant_float(s, res, CONSTANT_V7_POST)
    var ray_len = s.inverse_sqrt(s.fadd(dot3(s, ray, ray), s.float(0.0001)))
    var graz_sin = s.nmax(s.fmul(s.fabs(ray.y), ray_len), s.float(0.06))
    var px_foot = s.fdiv(s.fmul(depth, inverse_focal), graz_sin)
    var near_att = s.nclamp(s.fsub(s.float(1.0), s.fmul(px_foot, s.float(0.11))), s.float(0.0), s.float(1.0))
    var fine_att = s.nclamp(s.fsub(s.float(1.0), s.fmul(px_foot, s.float(0.62))), s.float(0.0), s.float(1.0))
    var lean_rough = s.fma(
        s.fsub(s.float(1.0), fine_att),
        s.float(0.075),
        s.fmul(s.fsub(s.float(1.0), near_att), s.float(0.055)),
    )
    s.store(rough_var, s.nclamp(s.fadd(base_rough, lean_rough), s.float(0.06), s.float(1.0)))
    var ph1 = s.fma(water_time, s.float(0.22), s.fma(world.x, s.float(0.021), s.fmul(world.z, s.float(0.014))))
    var ph2 = s.fma(water_time, s.float(-0.31), s.fma(world.z, s.float(0.043), s.fmul(world.x, s.float(-0.019))))
    var ph3 = s.fma(water_time, s.float(0.53), s.fma(world.x, s.float(0.118), s.fmul(world.z, s.float(0.089))))
    var ph4 = s.fma(water_time, s.float(-0.79), s.fma(world.z, s.float(0.340), s.fmul(world.x, s.float(-0.235))))
    var w1 = _parabolic_sin(s, ph1)
    var w2 = _parabolic_sin(s, ph2)
    var w3 = _parabolic_sin(s, ph3)
    var w4 = _parabolic_sin(s, ph4)
    var c1 = _parabolic_sin(s, s.fadd(ph1, s.float(0.25)))
    var c2 = _parabolic_sin(s, s.fadd(ph2, s.float(0.25)))
    var c3 = _parabolic_sin(s, s.fadd(ph3, s.float(0.25)))
    var c4 = _parabolic_sin(s, s.fadd(ph4, s.float(0.25)))
    var swell_x = s.fma(w1, s.float(0.108), s.fmul(w2, s.float(-0.072)))
    var chop_x = s.fma(w3, s.float(0.086), s.fmul(w4, s.float(-0.068)))
    var swell_z = s.fma(w1, s.float(0.072), s.fmul(w2, s.float(0.098)))
    var chop_z = s.fma(w3, s.float(0.068), s.fmul(w4, s.float(0.092)))
    s.store(rx_var, s.fadd(raw_x, s.fma(swell_x, near_att, s.fmul(chop_x, fine_att))))
    s.store(rz_var, s.fadd(raw_z, s.fma(swell_z, near_att, s.fmul(chop_z, fine_att))))

    # Trochoidal Gerstner wave-crest elevation & Jacobian whitecap determinant J = 1 - sum(Q_i * k_i * A_i * cos)
    var crest_sw = s.fma(c1, s.float(0.42), s.fmul(c2, s.float(0.30)))
    var crest_ch = s.fma(c3, s.float(0.18), s.fmul(c4, s.float(0.12)))
    var crest_h = s.nclamp(s.fma(crest_sw, near_att, s.fmul(crest_ch, fine_att)), s.float(-1.0), s.float(1.0))
    s.store(water_crest_var, s.nmax(crest_h, s.float(0.0)))
    s.store(water_near_var, near_att)

    var j_sw = s.fma(c1, s.float(0.34), s.fmul(c2, s.float(0.28)))
    var j_ch = s.fma(c3, s.float(0.24), s.fmul(c4, s.float(0.20)))
    var jacobian = s.fsub(s.float(1.0), s.fma(j_sw, near_att, s.fmul(j_ch, fine_att)))
    var whitecap = s.nclamp(s.fmul(s.fsub(s.float(0.52), jacobian), s.float(2.1)), s.float(0.0), s.float(0.72))

    # Depth-Buffer Shoreline Surf Foam against submerged island/reef geometry already in nearest_depth
    var shore_foam = s.fvar(s.float(0.0))
    s.begin_if(s.fgt(prev_nearest_iz, s.float(0.0002)))
    var sea_z = s.fdiv(s.float(1.0), prev_nearest_iz)
    var col_depth = s.fsub(sea_z, depth)
    s.begin_if(s.logical_and(s.fgt(col_depth, s.float(0.0)), s.flt(col_depth, s.float(2.8))))
    var shallows = s.nclamp(s.fsub(s.float(1.0), s.fmul(col_depth, s.float(0.357143))), s.float(0.0), s.float(1.0))
    var surf_ph = s.fma(w1, s.float(0.18), s.fsub(s.fmul(col_depth, s.float(1.15)), s.fmul(water_time, s.float(0.42))))
    var surf_band = s.fma(_parabolic_sin(s, surf_ph), s.float(0.5), s.float(0.5))
    s.store(shore_foam, s.fmul(s.fmul(shallows, shallows), s.fma(surf_band, s.float(0.62), s.float(0.38))))
    s.end_if()
    s.end_if()

    var foam = s.nclamp(s.nmax(whitecap, s.load_f(shore_foam)), s.float(0.0), s.float(0.86))
    s.begin_if(s.fgt(foam, s.float(0.005)))
    var cur_br = s.load_f(base_r_var)
    var cur_bg = s.load_f(base_g_var)
    var cur_bb = s.load_f(base_b_var)
    s.store(base_r_var, s.fma(s.fsub(s.float(216.0), cur_br), foam, cur_br))
    s.store(base_g_var, s.fma(s.fsub(s.float(224.0), cur_bg), foam, cur_bg))
    s.store(base_b_var, s.fma(s.fsub(s.float(226.0), cur_bb), foam, cur_bb))
    var cur_wr = s.load_f(rough_var)
    s.store(rough_var, s.fma(s.fsub(s.float(0.58), cur_wr), foam, cur_wr))
    s.end_if()
    s.end_if()

    var cot_tx = s.fvar(s.float(0.0))
    var cot_ty = s.fvar(s.float(0.0))
    var cot_tz = s.fvar(s.float(0.0))
    var has_cot = s.uvar(s.uint(0))

    # Cotangent-frame Normal+ORM Atlas (NORM_ORM_FLAG) + fallback albedo micro-normal perturbation
    var has_norm_orm = s.logical_and(
        has_pbr,
        s.logical_and(
            s.ine(s.bit_and(flags, s.uint(NORM_ORM_FLAG)), s.uint(0)),
            s.logical_not(is_water),
        ),
    )
    var is_mip_tex = s.logical_and(
        has_pbr,
        s.logical_and(
            s.ine(s.bit_and(flags, s.uint(MIPMAPPED_FLAG)), s.uint(0)),
            s.logical_not(is_water),
        ),
    )
    var do_mip_micro = s.logical_and(is_mip_tex, s.flt(tex_lod, s.float(2.8)))
    s.begin_if(s.logical_or(has_norm_orm, do_mip_micro))
    var n0_raw = Vec3Ids(s.load_f(rx_var), s.load_f(ry_var), s.load_f(rz_var))
    var inv_n0 = s.inverse_sqrt(s.fadd(dot3(s, n0_raw, n0_raw), s.float(0.000001)))
    var n0 = Vec3Ids(s.fmul(n0_raw.x, inv_n0), s.fmul(n0_raw.y, inv_n0), s.fmul(n0_raw.z, inv_n0))
    var px_scale = s.fmul(inverse_focal, depth)
    var neg_py = s.fsub(s.float(0.0), px_scale)
    var dpx = Vec3Ids(s.fmul(cam_right.x, px_scale), s.fmul(cam_right.y, px_scale), s.fmul(cam_right.z, px_scale))
    var dpy = Vec3Ids(s.fmul(cam_up.x, neg_py), s.fmul(cam_up.y, neg_py), s.fmul(cam_up.z, neg_py))
    var dot_nx = dot3(s, n0, dpx)
    var dot_ny = dot3(s, n0, dpy)
    var tx_v = Vec3Ids(
        s.fsub(dpx.x, s.fmul(n0.x, dot_nx)),
        s.fsub(dpx.y, s.fmul(n0.y, dot_nx)),
        s.fsub(dpx.z, s.fmul(n0.z, dot_nx)),
    )
    var ty_v = Vec3Ids(
        s.fsub(dpy.x, s.fmul(n0.x, dot_ny)),
        s.fsub(dpy.y, s.fmul(n0.y, dot_ny)),
        s.fsub(dpy.z, s.fmul(n0.z, dot_ny)),
    )
    var tan_raw = Vec3Ids(
        s.fsub(s.fmul(tx_v.x, dv_dy), s.fmul(ty_v.x, dv_dx)),
        s.fsub(s.fmul(tx_v.y, dv_dy), s.fmul(ty_v.y, dv_dx)),
        s.fsub(s.fmul(tx_v.z, dv_dy), s.fmul(ty_v.z, dv_dx)),
    )
    var bit_raw = Vec3Ids(
        s.fsub(s.fmul(ty_v.x, du_dx), s.fmul(tx_v.x, du_dy)),
        s.fsub(s.fmul(ty_v.y, du_dx), s.fmul(tx_v.y, du_dy)),
        s.fsub(s.fmul(ty_v.z, du_dx), s.fmul(tx_v.z, du_dy)),
    )
    var tan_len2 = dot3(s, tan_raw, tan_raw)
    var bit_len2 = dot3(s, bit_raw, bit_raw)
    s.begin_if(s.logical_and(s.fgt(tan_len2, s.float(1e-11)), s.fgt(bit_len2, s.float(1e-11))))
    var inv_t = s.inverse_sqrt(tan_len2)
    var inv_b = s.inverse_sqrt(bit_len2)
    var tan_u = Vec3Ids(s.fmul(tan_raw.x, inv_t), s.fmul(tan_raw.y, inv_t), s.fmul(tan_raw.z, inv_t))
    var bit_u = Vec3Ids(s.fmul(bit_raw.x, inv_b), s.fmul(bit_raw.y, inv_b), s.fmul(bit_raw.z, inv_b))
    s.store(cot_tx, tan_u.x)
    s.store(cot_ty, tan_u.y)
    s.store(cot_tz, tan_u.z)
    s.store(has_cot, s.uint(1))

    s.begin_if(has_norm_orm)
    var norm_meta = field(s, res, base, RECORD_NORM_ORM_TEX)
    var norm_first = s.bit_and(norm_meta, s.uint(0x0FFFFFFF))
    var norm_log2 = s.shr(norm_meta, s.uint(28))
    var norm_dim = s.shl(s.uint(1), norm_log2)
    var norm_area = s.imul(norm_dim, norm_dim)
    var norm_dim_f = s.float_from_uint(norm_dim)
    var grad_m = s.fmul(
        s.nmax(s.fadd(s.fabs(du_dx), s.fabs(dv_dx)), s.fadd(s.fabs(du_dy), s.fabs(dv_dy))),
        norm_dim_f,
    )
    var raw_nlod = s.umin(
        s.saturate_uint(s.fmul(s.log2(s.nmax(grad_m, s.float(1.0))), s.float(0.85))),
        s.umin(norm_log2, s.uint(3)),
    )
    var safe_nlod = s.select_u(s.uge(norm_log2, s.uint(4)), raw_nlod, s.uint(0))
    var norm_pk = _sample_mip_bilinear(s, res, norm_first, norm_dim, norm_dim, norm_area, u_norm, v_norm, safe_nlod)
    var nx_m = s.fmul(s.fsub(s.float_from_uint(channel(s, norm_pk, 16)), s.float(128.0)), s.float(0.007874016))
    var ny_m = s.fmul(s.fsub(s.float_from_uint(channel(s, norm_pk, 8)), s.float(128.0)), s.float(0.007874016))
    var orm_rough = s.nclamp(s.fmul(s.float_from_uint(channel(s, norm_pk, 0)), s.float(0.003921569)), s.float(0.06), s.float(1.0))
    var orm_ao = s.nclamp(s.fmul(s.float_from_uint(channel(s, norm_pk, 24)), s.float(0.003921569)), s.float(0.25), s.float(1.0))
    var n_gain = s.fmul(
        s.nclamp(s.fsub(s.float(1.0), s.fmul(s.float_from_uint(safe_nlod), s.float(0.24))), s.float(0.25), s.float(1.0)),
        s.float(0.72),
    )
    var gnu = s.fmul(nx_m, n_gain)
    var gnv = s.fmul(ny_m, n_gain)
    s.store(rx_var, s.fadd(s.load_f(rx_var), s.fma(tan_u.x, gnu, s.fmul(bit_u.x, gnv))))
    s.store(ry_var, s.fadd(s.load_f(ry_var), s.fma(tan_u.y, gnu, s.fmul(bit_u.y, gnv))))
    s.store(rz_var, s.fadd(s.load_f(rz_var), s.fma(tan_u.z, gnu, s.fmul(bit_u.z, gnv))))
    var cur_rv = s.load_f(rough_var)
    s.store(rough_var, s.nclamp(s.fma(s.fsub(orm_rough, cur_rv), s.float(0.55), cur_rv), s.float(0.06), s.float(1.0)))
    s.store(occ_var, s.fmul(s.load_f(occ_var), orm_ao))
    s.begin_else()
    var first0 = s.shr(field(s, res, base, RECORD_TEXTURE), s.uint(2))
    var w0 = field(s, res, base, RECORD_TEXTURE_WIDTH)
    var h0 = s.iadd(field(s, res, base, RECORD_TEXTURE_MAX_Y), s.uint(1))
    var area0 = s.imul(w0, h0)
    var det_mip = s.umin(s.saturate_uint(tex_lod), s.uint(2))
    var inv_mw = s.fdiv(s.float(1.0), s.float_from_uint(s.umax(s.shr(w0, det_mip), s.uint(1))))
    var inv_mh = s.fdiv(s.float(1.0), s.float_from_uint(s.umax(s.shr(h0, det_mip), s.uint(1))))
    var lu_p = luma_float(s, _sample_mip_bilinear(s, res, first0, w0, h0, area0, s.fadd(u_norm, inv_mw), v_norm, det_mip))
    var lu_n = luma_float(s, _sample_mip_bilinear(s, res, first0, w0, h0, area0, s.fsub(u_norm, inv_mw), v_norm, det_mip))
    var lv_p = luma_float(s, _sample_mip_bilinear(s, res, first0, w0, h0, area0, u_norm, s.fadd(v_norm, inv_mh), det_mip))
    var lv_n = luma_float(s, _sample_mip_bilinear(s, res, first0, w0, h0, area0, u_norm, s.fsub(v_norm, inv_mh), det_mip))
    var dh_u = s.fsub(lu_p, lu_n)
    var dh_v = s.fsub(lv_p, lv_n)
    var det_fade = s.fmul(s.nclamp(s.fsub(s.float(1.0), s.fmul(tex_lod, s.float(0.35))), s.float(0.0), s.float(1.0)), s.float(0.18))
    var bu = s.fmul(s.fmul(dh_u, inv_t), det_fade)
    var bv = s.fmul(s.fmul(dh_v, inv_b), det_fade)
    s.store(rx_var, s.fsub(s.load_f(rx_var), s.fma(tan_raw.x, bu, s.fmul(bit_raw.x, bv))))
    s.store(ry_var, s.fsub(s.load_f(ry_var), s.fma(tan_raw.y, bu, s.fmul(bit_raw.y, bv))))
    s.store(rz_var, s.fsub(s.load_f(rz_var), s.fma(tan_raw.z, bu, s.fmul(bit_raw.z, bv))))
    var seam_delta = s.nclamp(s.fmul(s.fadd(s.fabs(dh_u), s.fabs(dh_v)), s.float(0.22)), s.float(0.0), s.float(0.09))
    s.store(rough_var, s.nclamp(s.fadd(s.load_f(rough_var), seam_delta), s.float(0.06), s.float(1.0)))
    s.end_if()
    s.end_if()
    s.end_if()

    # Oriented Box Surface Decals (laser/plasma scorches, blast marks, carrier deck markings)
    var decal_cnt = s.umin(constant_word(s, res, CONSTANT_V8_ATMOS + 36), s.uint(UInt32(MAX_SURFACE_DECALS)))
    s.begin_if(s.logical_and(s.ine(decal_cnt, s.uint(0)), s.logical_not(is_water)))
    var dec_i = s.uvar(s.uint(0))
    s.begin_loop()
    var di = s.load_u(dec_i)
    s.break_if(s.uge(di, decal_cnt))
    s.store(dec_i, s.iadd(di, s.uint(1)))
    var dw_b = s.iadd(s.uint(UInt32(CONSTANT_DECALS // 4)), s.shl(di, s.uint(4)))
    var dr_x = s.fsub(world.x, light_float(s, res, dw_b, 0))
    var dr_y = s.fsub(world.y, light_float(s, res, dw_b, 1))
    var dr_z = s.fsub(world.z, light_float(s, res, dw_b, 2))
    var d_rel = Vec3Ids(dr_x, dr_y, dr_z)
    var ax_r = Vec3Ids(light_float(s, res, dw_b, 4), light_float(s, res, dw_b, 5), light_float(s, res, dw_b, 6))
    var ax_f = Vec3Ids(light_float(s, res, dw_b, 8), light_float(s, res, dw_b, 9), light_float(s, res, dw_b, 10))
    var ax_n = Vec3Ids(
        s.fsub(s.fmul(ax_r.y, ax_f.z), s.fmul(ax_r.z, ax_f.y)),
        s.fsub(s.fmul(ax_r.z, ax_f.x), s.fmul(ax_r.x, ax_f.z)),
        s.fsub(s.fmul(ax_r.x, ax_f.y), s.fmul(ax_r.y, ax_f.x)),
    )
    var du_d = s.fmul(dot3(s, d_rel, ax_r), light_float(s, res, dw_b, 3))
    var dv_d = s.fmul(dot3(s, d_rel, ax_f), light_float(s, res, dw_b, 7))
    var dn_d = s.fabs(s.fmul(dot3(s, d_rel, ax_n), light_float(s, res, dw_b, 11)))
    var r2_d = s.fma(dv_d, dv_d, s.fmul(du_d, du_d))
    s.begin_if(s.logical_and(s.flt(r2_d, s.float(1.0)), s.flt(dn_d, s.float(1.0))))
    var fall_d = s.fmul(s.fsub(s.float(1.0), r2_d), s.fsub(s.float(1.0), dn_d))
    var p15 = light_float(s, res, dw_b, 15)
    var dec_op = s.nclamp(s.fmul(s.fmul(s.fabs(p15), fall_d), s.float(1.35)), s.float(0.0), s.float(0.94))
    var dc_r = light_float(s, res, dw_b, 12)
    var dc_g = light_float(s, res, dw_b, 13)
    var dc_b = light_float(s, res, dw_b, 14)
    var cbr = s.load_f(base_r_var)
    var cbg = s.load_f(base_g_var)
    var cbb = s.load_f(base_b_var)
    s.store(base_r_var, s.fma(s.fsub(dc_r, cbr), dec_op, cbr))
    s.store(base_g_var, s.fma(s.fsub(dc_g, cbg), dec_op, cbg))
    s.store(base_b_var, s.fma(s.fsub(dc_b, cbb), dec_op, cbb))
    var crv = s.load_f(rough_var)
    s.store(rough_var, s.fma(s.fsub(s.float(0.80), crv), s.fmul(dec_op, s.float(0.65)), crv))
    s.begin_if(s.flt(p15, s.float(0.0)))
    var ring = s.nmax(s.fsub(s.float(1.0), s.fmul(s.fabs(s.fsub(r2_d, s.float(0.30))), s.float(3.2))), s.float(0.0))
    var ember = s.fmul(s.fmul(ring, s.fsub(s.float(1.0), dn_d)), s.nclamp(s.fsub(s.float(0.0), p15), s.float(0.0), s.float(2.0)))
    s.store(decal_emit_r, s.fma(ember, s.float(245.0), s.load_f(decal_emit_r)))
    s.store(decal_emit_g, s.fma(ember, s.float(118.0), s.load_f(decal_emit_g)))
    s.store(decal_emit_b, s.fma(ember, s.float(28.0), s.load_f(decal_emit_b)))
    s.end_if()
    s.end_if()
    s.end_loop()
    s.end_if()

    # V11 Wet-PBR Surfaces & Puddle Raindrop Ripples (EFFECT_WEATHER_PRECIP)
    var wx_pbr_on = s.logical_and(
        s.ine(s.bit_and(effects, s.uint(EFFECT_WEATHER_PRECIP)), s.uint(0)),
        s.logical_not(is_water),
    )
    s.begin_if(wx_pbr_on)
    var wet_amt = constant_float(s, res, CONSTANT_V11_POST + 16)
    s.begin_if(s.fgt(wet_amt, s.float(0.01)))
    var up_wet = s.fmul(s.nclamp(s.fma(s.load_f(ry_var), s.float(0.85), s.float(0.25)), s.float(0.0), s.float(1.0)), wet_amt)
    var met_wet = s.nclamp(s.bitcast_float(field(s, res, base, RECORD_METALLIC)), s.float(0.0), s.float(1.0))
    var dark_f = s.fsub(s.float(1.0), s.fmul(up_wet, s.fmul(s.fsub(s.float(1.0), met_wet), s.float(0.42))))
    s.store(base_r_var, s.fmul(s.load_f(base_r_var), dark_f))
    s.store(base_g_var, s.fmul(s.load_f(base_g_var), dark_f))
    s.store(base_b_var, s.fmul(s.load_f(base_b_var), dark_f))
    var cur_rw = s.load_f(rough_var)
    var wet_g = s.fmul(up_wet, s.float(0.62))
    s.store(rough_var, s.fma(s.fsub(s.float(0.075), cur_rw), wet_g, cur_rw))
    var wx_kind_p = constant_word(s, res, CONSTANT_V11_POST + 8)
    s.begin_if(s.logical_and(s.ieq(wx_kind_p, s.uint(1)), s.fgt(s.load_f(ry_var), s.float(0.65))))
    var w_t = constant_float(s, res, CONSTANT_V7_POST)
    var rip_ph = s.fsub(s.fma(world.x, s.float(1.45), s.fmul(world.z, s.float(1.32))), s.fmul(w_t, s.float(2.4)))
    var rip_s = s.fmul(_parabolic_sin(s, rip_ph), s.fmul(wet_amt, s.float(0.045)))
    var rip_c = s.fmul(_parabolic_sin(s, s.fadd(rip_ph, s.float(0.25))), s.fmul(wet_amt, s.float(0.045)))
    s.store(rx_var, s.fadd(s.load_f(rx_var), rip_s))
    s.store(rz_var, s.fadd(s.load_f(rz_var), rip_c))
    s.end_if()
    s.end_if()
    s.end_if()

    var occlusion = s.load_f(occ_var)
    var pert_x = s.load_f(rx_var)
    var pert_y = s.load_f(ry_var)
    var pert_z = s.load_f(rz_var)
    var raw = Vec3Ids(pert_x, pert_y, pert_z)
    var inverse_length = s.inverse_sqrt(s.fadd(dot3(s, raw, raw), s.float(0.000001)))
    var normal = Vec3Ids(s.fmul(inverse_length, pert_x), s.fmul(inverse_length, pert_y), s.fmul(inverse_length, pert_z))

    # Construct orthonormal (tan_dir, bit_dir) around perturbed normal for Anisotropic GGX
    var not_pole = s.flt(s.fabs(normal.y), s.float(0.92))
    var fb_tx = s.select_f(not_pole, normal.z, s.float(0.0))
    var fb_ty = s.select_f(not_pole, s.float(0.0), s.fsub(s.float(0.0), normal.z))
    var fb_tz = s.select_f(not_pole, s.fsub(s.float(0.0), normal.x), normal.y)
    var use_cot = s.ine(s.load_u(has_cot), s.uint(0))
    var t0_x = s.select_f(use_cot, s.load_f(cot_tx), fb_tx)
    var t0_y = s.select_f(use_cot, s.load_f(cot_ty), fb_ty)
    var t0_z = s.select_f(use_cot, s.load_f(cot_tz), fb_tz)
    var n_dot_t0 = dot3(s, normal, Vec3Ids(t0_x, t0_y, t0_z))
    var to_x = s.fsub(t0_x, s.fmul(normal.x, n_dot_t0))
    var to_y = s.fsub(t0_y, s.fmul(normal.y, n_dot_t0))
    var to_z = s.fsub(t0_z, s.fmul(normal.z, n_dot_t0))
    var inv_to = s.inverse_sqrt(s.fadd(dot3(s, Vec3Ids(to_x, to_y, to_z), Vec3Ids(to_x, to_y, to_z)), s.float(0.000001)))
    var tan_dir = Vec3Ids(s.fmul(to_x, inv_to), s.fmul(to_y, inv_to), s.fmul(to_z, inv_to))
    var bit_dir = Vec3Ids(
        s.fsub(s.fmul(normal.y, tan_dir.z), s.fmul(normal.z, tan_dir.y)),
        s.fsub(s.fmul(normal.z, tan_dir.x), s.fmul(normal.x, tan_dir.z)),
        s.fsub(s.fmul(normal.x, tan_dir.y), s.fmul(normal.y, tan_dir.x)),
    )

    # Encode octahedral normal for G-buffer SSR + GTAO
    var l1_norm = s.fadd(s.fadd(s.fabs(normal.x), s.fabs(normal.y)), s.fadd(s.fabs(normal.z), s.float(0.000001)))
    var inv_l1 = s.fdiv(s.float(1.0), l1_norm)
    var ox0 = s.fmul(normal.x, inv_l1)
    var oz0 = s.fmul(normal.z, inv_l1)
    var below_h = s.flt(normal.y, s.float(0.0))
    var sign_ox = s.select_f(s.fge(ox0, s.float(0.0)), s.float(1.0), s.float(-1.0))
    var sign_oz = s.select_f(s.fge(oz0, s.float(0.0)), s.float(1.0), s.float(-1.0))
    var ox_enc = s.select_f(below_h, s.fmul(s.fsub(s.float(1.0), s.fabs(oz0)), sign_ox), ox0)
    var oz_enc = s.select_f(below_h, s.fmul(s.fsub(s.float(1.0), s.fabs(ox0)), sign_oz), oz0)
    s.store(out_ox, _encode_octahedral_byte(s, ox_enc))
    s.store(out_oz, _encode_octahedral_byte(s, oz_enc))

    var roughness = s.load_f(rough_var)
    s.store(out_rough, float_to_byte(s, s.fmul(roughness, s.float(255.0))))
    var metallic = s.nclamp(s.bitcast_float(field(s, res, base, RECORD_METALLIC)), s.float(0.0), s.float(1.0))
    var kd = s.fvar(s.float(1.0))
    s.begin_if(has_pbr)
    s.store(kd, s.fsub(s.float(1.0), s.fmul(metallic, s.float(0.65))))
    s.end_if()
    var kd_val = s.load_f(kd)

    var base_r = s.load_f(base_r_var)
    var base_g = s.load_f(base_g_var)
    var base_b = s.load_f(base_b_var)

    var inv_ray = s.inverse_sqrt(s.fadd(dot3(s, ray, ray), s.float(0.000001)))
    var neg_inv_ray = s.fsub(s.float(0.0), inv_ray)
    var view_dir = Vec3Ids(s.fmul(ray.x, neg_inv_ray), s.fmul(ray.y, neg_inv_ray), s.fmul(ray.z, neg_inv_ray))
    var n_dot_v = s.nmax(dot3(s, normal, view_dir), s.float(0.001))
    var two_ndv = s.fmul(s.float(2.0), n_dot_v)
    var refl_dir = Vec3Ids(
        s.fsub(s.fmul(two_ndv, normal.x), view_dir.x),
        s.fsub(s.fmul(two_ndv, normal.y), view_dir.y),
        s.fsub(s.fmul(two_ndv, normal.z), view_dir.z),
    )

    var alpha_r = s.fmul(roughness, roughness)
    var rec_aniso = s.nclamp(s.bitcast_float(field(s, res, base, RECORD_ANISO)), s.float(0.0), s.float(0.85))
    var auto_aniso = s.select_f(
        s.logical_and(has_pbr, s.logical_not(is_water)),
        s.nclamp(s.fmul(s.fsub(metallic, s.float(0.35)), s.float(0.55)), s.float(0.0), s.float(0.36)),
        s.float(0.0),
    )
    var aniso = s.nmax(rec_aniso, auto_aniso)
    var ax = s.nmax(s.float(0.03), s.fmul(alpha_r, s.fma(aniso, s.float(0.85), s.float(1.0))))
    var ay = s.nmax(s.float(0.03), s.fmul(alpha_r, s.fsub(s.float(1.0), s.fmul(aniso, s.float(0.55)))))

    var r_plus_1 = s.fadd(roughness, s.float(1.0))
    var k_smith = s.fmul(s.fmul(r_plus_1, r_plus_1), s.float(0.125))
    var one_minus_k = s.fsub(s.float(1.0), k_smith)
    var dielec_f0 = s.float(11.5)
    var f0_r = s.fma(metallic, s.fsub(base_r, dielec_f0), dielec_f0)
    var f0_g = s.fma(metallic, s.fsub(base_g, dielec_f0), dielec_f0)
    var f0_b = s.fma(metallic, s.fsub(base_b, dielec_f0), dielec_f0)

    # Brian Karis analytical Split-Sum GGX Environment BRDF + Kulla-Conty Multiscattering Energy Compensation
    var kr_x = s.fma(roughness, s.float(-1.0), s.float(1.0))
    var kr_y = s.fma(roughness, s.float(-0.0275), s.float(0.0425))
    var kr_z = s.fma(roughness, s.float(-0.572), s.float(1.04))
    var kr_w = s.fma(roughness, s.float(0.022), s.float(-0.04))
    var exp_nv = s.exp2(s.fmul(s.float(-9.28), n_dot_v))
    var a004 = s.fma(s.nmin(s.fmul(kr_x, kr_x), exp_nv), kr_x, kr_y)
    var brdf_scale = s.nclamp(s.fma(a004, s.float(-1.04), kr_z), s.float(0.0), s.float(1.0))
    var brdf_bias = s.nclamp(s.fma(a004, s.float(1.04), kr_w), s.float(0.0), s.float(1.0))
    var dir_albedo = s.nclamp(s.fadd(brdf_scale, brdf_bias), s.float(0.18), s.float(1.0))
    var f0_norm = s.nclamp(s.fmul(s.fadd(s.fadd(f0_r, f0_g), f0_b), s.float(0.00130719)), s.float(0.0), s.float(1.0))
    var ms_loss = s.nclamp(s.fdiv(s.fsub(s.float(1.0), dir_albedo), dir_albedo), s.float(0.0), s.float(1.35))
    var ms_comp = s.fma(s.fmul(f0_norm, ms_loss), s.float(0.58), s.float(1.0))

    # Enclosed-Interior Ambient / Spherical-Harmonic Irradiance Probes (sky-occluding bounce)
    var probe_r = s.fvar(s.float(0.0))
    var probe_g = s.fvar(s.float(0.0))
    var probe_b = s.fvar(s.float(0.0))
    var sky_occ = s.fvar(s.float(0.0))
    var probe_cnt = s.umin(constant_word(s, res, CONSTANT_V8_ATMOS + 32), s.uint(UInt32(MAX_IRRADIANCE_PROBES)))
    s.begin_if(s.ine(probe_cnt, s.uint(0)))
    var pr_i = s.uvar(s.uint(0))
    s.begin_loop()
    var pi = s.load_u(pr_i)
    s.break_if(s.uge(pi, probe_cnt))
    s.store(pr_i, s.iadd(pi, s.uint(1)))
    var pw_b = s.iadd(s.uint(UInt32(CONSTANT_PROBES // 4)), s.shl(pi, s.uint(3)))
    var dp_x = s.fsub(light_float(s, res, pw_b, 0), world.x)
    var dp_y = s.fsub(light_float(s, res, pw_b, 1), world.y)
    var dp_z = s.fsub(light_float(s, res, pw_b, 2), world.z)
    var dp_v = Vec3Ids(dp_x, dp_y, dp_z)
    var dp2 = s.fadd(dot3(s, dp_v, dp_v), s.float(0.001))
    var tp0 = s.nmax(s.fsub(s.float(1.0), s.fmul(dp2, light_float(s, res, pw_b, 3))), s.float(0.0))
    var wp = s.fmul(tp0, tp0)
    s.begin_if(s.fgt(wp, s.float(0.0005)))
    var inv_dp = s.inverse_sqrt(dp2)
    var sh_dir = s.fma(s.nmax(s.fmul(dot3(s, normal, dp_v), inv_dp), s.float(0.0)), s.float(0.35), s.float(0.65))
    var wp_sh = s.fmul(wp, sh_dir)
    s.store(probe_r, s.fma(light_float(s, res, pw_b, 4), wp_sh, s.load_f(probe_r)))
    s.store(probe_g, s.fma(light_float(s, res, pw_b, 5), wp_sh, s.load_f(probe_g)))
    s.store(probe_b, s.fma(light_float(s, res, pw_b, 6), wp_sh, s.load_f(probe_b)))
    var p_occ = s.nclamp(light_float(s, res, pw_b, 7), s.float(0.0), s.float(0.95))
    s.store(sky_occ, s.nmax(s.load_f(sky_occ), s.fmul(wp, p_occ)))
    s.end_if()
    s.end_loop()
    s.end_if()
    var ext_vis = s.fsub(s.float(1.0), s.load_f(sky_occ))

    # Key light + 2-Cascade Poisson PCF / sphere / voxel shadows
    var facing = s.nmax(dot3(s, normal, to_key), s.float(0))
    var key_shadow = s.fvar(s.float(1.0))
    var key = s.fvar(s.fmul(s.fadd(s.float(0.78), s.fmul(s.float(0.22), occlusion)), facing))
    var traced = s.ine(s.bit_and(effects, s.uint(EFFECT_SHADOWS)), s.uint(0))
    s.begin_if(s.logical_and(traced, s.flt(s.float(0), s.load_f(key))))
    var shadow = trace_shadow(s, res, world, normal, to_key, effects, x, y)
    s.store(key_shadow, shadow)
    s.store(key, s.fmul(shadow, s.load_f(key)))
    s.end_if()

    # V9 Projected Volumetric Cloud Shadows sweeping across terrain, skyscrapers, hulls & ocean waves
    var cl_on_sh = s.ine(s.bit_and(effects, s.uint(EFFECT_VOL_CLOUDS)), s.uint(0))
    s.begin_if(s.logical_and(cl_on_sh, s.fgt(to_key.y, s.float(0.05))))
    var cl_base_sh = constant_float(s, res, CONSTANT_V9_CLOUDS)
    var cl_top_sh = constant_float(s, res, CONSTANT_V9_CLOUDS + 4)
    var cl_mid_y = s.fmul(s.fadd(cl_base_sh, cl_top_sh), s.float(0.5))
    s.begin_if(s.flt(world.y, cl_mid_y))
    var t_cl_sh = s.fdiv(s.fsub(cl_mid_y, world.y), s.nmax(to_key.y, s.float(0.08)))
    s.begin_if(s.logical_and(s.fgt(t_cl_sh, s.float(0.0)), s.flt(t_cl_sh, s.float(6500.0))))
    var cl_d_sh = _cloud_density_3d(
        s,
        res,
        s.fma(to_key.x, t_cl_sh, world.x),
        cl_mid_y,
        s.fma(to_key.z, t_cl_sh, world.z),
    )
    var cl_str = constant_float(s, res, CONSTANT_V9_CLOUDS + 24)
    var cl_vis = s.nclamp(s.fsub(s.float(1.0), s.fmul(cl_d_sh, cl_str)), s.float(0.22), s.float(1.0))
    s.store(key_shadow, s.fmul(s.load_f(key_shadow), cl_vis))
    s.store(key, s.fmul(s.load_f(key), cl_vis))
    s.end_if()
    s.end_if()
    s.end_if()

    # V11 World-Space 3D SDF Soft-Penumbra Capsule / Sphere Shadows (EFFECT_SDF_SHADOWS)
    var sdf_on = s.logical_and(
        s.ine(s.bit_and(effects, s.uint(EFFECT_SDF_SHADOWS)), s.uint(0)),
        s.fgt(facing, s.float(0.001)),
    )
    s.begin_if(sdf_on)
    var sdf_cnt = s.umin(constant_word(s, res, CONSTANT_V11_POST), s.uint(UInt32(MAX_SDF_OCCLUDERS)))
    s.begin_if(s.ine(sdf_cnt, s.uint(0)))
    var sdf_k = s.nmax(constant_float(s, res, CONSTANT_V11_POST + 4), s.float(1.5))
    var wb_x = s.fma(normal.x, s.float(0.18), world.x)
    var wb_y = s.fma(normal.y, s.float(0.18), world.y)
    var wb_z = s.fma(normal.z, s.float(0.18), world.z)
    var sdf_vis = s.fvar(s.float(1.0))
    var sdf_i = s.uvar(s.uint(0))
    s.begin_loop()
    var si_u = s.load_u(sdf_i)
    s.break_if(s.uge(si_u, sdf_cnt))
    s.store(sdf_i, s.iadd(si_u, s.uint(1)))
    var sb_w = s.iadd(s.uint(UInt32(CONSTANT_SDF_OCCLUDERS // 4)), s.shl(si_u, s.uint(3)))
    var oa_x = s.fsub(light_float(s, res, sb_w, 0), wb_x)
    var oa_y = s.fsub(light_float(s, res, sb_w, 1), wb_y)
    var oa_z = s.fsub(light_float(s, res, sb_w, 2), wb_z)
    var rad_s = light_float(s, res, sb_w, 3)
    var sg_x = light_float(s, res, sb_w, 4)
    var sg_y = light_float(s, res, sb_w, 5)
    var sg_z = light_float(s, res, sb_w, 6)
    var opac_s = s.nclamp(light_float(s, res, sb_w, 7), s.float(0.0), s.float(0.95))
    var oa_v = Vec3Ids(oa_x, oa_y, oa_z)
    var sg_v = Vec3Ids(sg_x, sg_y, sg_z)
    var s_len2 = dot3(s, sg_v, sg_v)
    var u_seg = s.fvar(s.float(0.0))
    s.begin_if(s.fgt(s_len2, s.float(0.0001)))
    var d_s = dot3(s, to_key, sg_v)
    var oa_d = dot3(s, oa_v, to_key)
    var oa_s = dot3(s, oa_v, sg_v)
    var den_s = s.nmax(s.fsub(s_len2, s.fmul(d_s, d_s)), s.float(0.0001))
    s.store(u_seg, s.nclamp(s.fdiv(s.fsub(s.fmul(oa_d, d_s), oa_s), den_s), s.float(0.0), s.float(1.0)))
    s.end_if()
    var u_f = s.load_f(u_seg)
    var rc_v = Vec3Ids(s.fma(sg_x, u_f, oa_x), s.fma(sg_y, u_f, oa_y), s.fma(sg_z, u_f, oa_z))
    var rc2 = dot3(s, rc_v, rc_v)
    var self_r2 = s.fmul(s.fmul(rad_s, rad_s), s.float(1.08))
    var t_ray = dot3(s, rc_v, to_key)
    s.begin_if(s.logical_and(s.fgt(rc2, self_r2), s.fgt(t_ray, s.float(0.10))))
    var perp = s.fsqrt(s.nmax(s.fsub(rc2, s.fmul(t_ray, t_ray)), s.float(0.0)))
    var h_clr = s.fsub(perp, rad_s)
    var pen_den = s.nmax(s.fma(t_ray, s.float(0.35), rad_s), s.float(0.45))
    var pen_v = s.nclamp(s.fma(s.fdiv(s.fmul(sdf_k, h_clr), pen_den), s.float(0.5), s.float(0.5)), s.float(0.0), s.float(1.0))
    var occ_v = s.fsub(s.float(1.0), s.fmul(opac_s, s.fsub(s.float(1.0), pen_v)))
    s.store(sdf_vis, s.nmin(s.load_f(sdf_vis), occ_v))
    s.end_if()
    s.end_loop()
    var sv_f = s.load_f(sdf_vis)
    s.store(key_shadow, s.fmul(s.load_f(key_shadow), sv_f))
    s.store(key, s.fmul(s.load_f(key), sv_f))
    s.end_if()
    s.end_if()
    var key_light = s.fmul(s.load_f(key), kd_val)
    var pr_kd = s.fmul(occlusion, kd_val)
    s.store(light_r, s.fma(s.load_f(probe_r), pr_kd, s.fmul(key_color.x, key_light)))
    s.store(light_g, s.fma(s.load_f(probe_g), pr_kd, s.fmul(key_color.y, key_light)))
    s.store(light_b, s.fma(s.load_f(probe_b), pr_kd, s.fmul(key_color.z, key_light)))
    s.store(spec_r, s.fadd(s.load_f(spec_r), s.load_f(decal_emit_r)))
    s.store(spec_g, s.fadd(s.load_f(spec_g), s.load_f(decal_emit_g)))
    s.store(spec_b, s.fadd(s.load_f(spec_b), s.load_f(decal_emit_b)))

    # Backlit Turquoise Subsurface Wave-Crest Scattering (SSS) on ocean swells
    s.begin_if(is_water)
    var back_align = s.nmax(s.fma(s.fsub(s.float(0.0), dot3(s, view_dir, to_key)), s.float(0.65), s.float(0.35)), s.float(0.0))
    var sss_w = s.fmul(s.fmul(s.load_f(water_crest_var), back_align), s.fmul(s.load_f(water_near_var), s.float(0.38)))
    s.store(light_r, s.fma(key_color.x, s.fmul(sss_w, s.float(0.08)), s.load_f(light_r)))
    s.store(light_g, s.fma(key_color.y, s.fmul(sss_w, s.float(0.44)), s.load_f(light_g)))
    s.store(light_b, s.fma(key_color.z, s.fmul(sss_w, s.float(0.42)), s.load_f(light_b)))
    s.end_if()

    # Key-light Dual-Lobe Clear-Coat + Anisotropic/Kulla-Conty GGX specular + Split-Sum Environment BRDF IBL
    s.begin_if(has_pbr)
    var n_dot_l_key = s.fmul(facing, s.load_f(key_shadow))
    s.begin_if(s.flt(s.float(0.0001), n_dot_l_key))
    accumulate_ggx_specular(
        s, normal, tan_dir, bit_dir, view_dir, to_key, key_color, n_dot_v, n_dot_l_key,
        ax, ay, ms_comp, k_smith, one_minus_k, f0_r, f0_g, f0_b, spec_r, spec_g, spec_b,
    )
    var cc_gain = s.fmul(s.float(0.36), s.nmax(s.fsub(s.float(1.0), s.fmul(roughness, s.float(0.72))), s.float(0.0)))
    var cc_col = Vec3Ids(s.fmul(key_color.x, cc_gain), s.fmul(key_color.y, cc_gain), s.fmul(key_color.z, cc_gain))
    var cc_f0 = s.float(10.2)
    accumulate_ggx_specular(
        s, normal, tan_dir, bit_dir, view_dir, to_key, cc_col, n_dot_v, n_dot_l_key,
        s.float(0.0922), s.float(0.0922), s.float(1.0), s.float(0.149), s.float(0.851), cc_f0, cc_f0, cc_f0, spec_r, spec_g, spec_b,
    )
    s.end_if()

    var env_sky = constant_vector(s, res, CONSTANT_ENV_SKY)
    var env_ground = constant_vector(s, res, CONSTANT_ENV_GROUND)
    var env_horizon = constant_vector(s, res, CONSTANT_ENV_HORIZON)
    var refl_y = refl_dir.y
    var up_t = s.nclamp(s.fmul(refl_y, s.float(1.35)), s.float(0.0), s.float(1.0))
    var dn_t = s.nclamp(s.fmul(s.fsub(s.float(0.0), refl_y), s.float(1.55)), s.float(0.0), s.float(1.0))
    var env_r0 = s.fma(dn_t, s.fsub(env_ground.x, env_horizon.x), s.fma(up_t, s.fsub(env_sky.x, env_horizon.x), env_horizon.x))
    var env_g0 = s.fma(dn_t, s.fsub(env_ground.y, env_horizon.y), s.fma(up_t, s.fsub(env_sky.y, env_horizon.y), env_horizon.y))
    var env_b0 = s.fma(dn_t, s.fsub(env_ground.z, env_horizon.z), s.fma(up_t, s.fsub(env_sky.z, env_horizon.z), env_horizon.z))

    var smoothness = s.fsub(s.float(1.0), s.fmul(roughness, s.float(0.80)))
    var sun_align = s.nmax(dot3(s, refl_dir, to_key), s.float(0.0))
    var sun_a2 = s.fmul(sun_align, sun_align)
    var sun_mie = s.fmul(sun_a2, sun_a2)
    var abs_ry = s.fabs(refl_y)
    var horiz_mask = s.nclamp(s.fsub(s.float(1.0), s.fmul(abs_ry, s.float(0.68))), s.float(0.0), s.float(1.0))
    var mie_w = s.fmul(s.fmul(sun_mie, horiz_mask), s.fmul(smoothness, s.float(0.28)))
    var env_r = s.fma(s.load_f(probe_r), s.float(0.42), s.fmul(s.fma(key_color.x, mie_w, env_r0), ext_vis))
    var env_g = s.fma(s.load_f(probe_g), s.float(0.42), s.fmul(s.fma(key_color.y, mie_w, env_g0), ext_vis))
    var env_b = s.fma(s.load_f(probe_b), s.float(0.42), s.fmul(s.fma(key_color.z, mie_w, env_b0), ext_vis))

    var max_f = s.fmul(smoothness, s.float(255.0))
    var f90_r = s.nmax(max_f, f0_r)
    var f90_g = s.nmax(max_f, f0_g)
    var f90_b = s.nmax(max_f, f0_b)
    var fe_r = s.fmul(s.fma(f0_r, brdf_scale, s.fmul(f90_r, brdf_bias)), ms_comp)
    var fe_g = s.fmul(s.fma(f0_g, brdf_scale, s.fmul(f90_g, brdf_bias)), ms_comp)
    var fe_b = s.fmul(s.fma(f0_b, brdf_scale, s.fmul(f90_b, brdf_bias)), ms_comp)

    var ao_ibl = s.fadd(s.float(0.50), s.fmul(s.float(0.50), occlusion))
    var ibl_w = s.fmul(smoothness, s.fmul(ao_ibl, s.float(0.84)))
    s.store(spec_r, s.fma(s.fmul(fe_r, env_r), ibl_w, s.load_f(spec_r)))
    s.store(spec_g, s.fma(s.fmul(fe_g, env_g), ibl_w, s.load_f(spec_g)))
    s.store(spec_b, s.fma(s.fmul(fe_b, env_b), ibl_w, s.load_f(spec_b)))

    # V11 Real-Time Dynamic Environment Reflection Lobe (EFFECT_DYN_REFLECT)
    s.begin_if(s.ine(s.bit_and(effects, s.uint(EFFECT_DYN_REFLECT)), s.uint(0)))
    var dyn_str = constant_float(s, res, CONSTANT_V11_POST + 28)
    s.begin_if(s.fgt(dyn_str, s.float(0.005)))
    var d_dir = constant_vector(s, res, CONSTANT_V11_POST + 32)
    var d_col = constant_vector(s, res, CONSTANT_V11_POST + 48)
    var d_align = s.nmax(dot3(s, refl_dir, d_dir), s.float(0.0))
    var d_a2 = s.fmul(d_align, d_align)
    var d_lobe = s.fmul(s.fma(s.fmul(d_a2, d_a2), smoothness, s.fmul(d_align, s.float(0.22))), s.fmul(dyn_str, ao_ibl))
    s.store(spec_r, s.fma(s.fmul(fe_r, d_col.x), d_lobe, s.load_f(spec_r)))
    s.store(spec_g, s.fma(s.fmul(fe_g, d_col.y), d_lobe, s.load_f(spec_g)))
    s.store(spec_b, s.fma(s.fmul(fe_b, d_col.z), d_lobe, s.load_f(spec_b)))
    s.end_if()
    s.end_if()

    # SSR reflectivity weight in [0..127] for glossy dielectric/metallic surfaces & ocean water
    var gloss = s.nclamp(s.fmul(s.fsub(s.float(0.58), roughness), s.float(2.1)), s.float(0.0), s.float(1.0))
    var fe_avg = s.nclamp(s.fmul(s.fadd(s.fadd(fe_r, fe_g), fe_b), s.float(0.00130719)), s.float(0.0), s.float(1.0))
    var water_boost = s.select_f(is_water, s.float(1.35), s.float(1.0))
    var ssr_f = s.nclamp(
        s.fmul(s.fmul(s.fmul(gloss, gloss), s.fma(fe_avg, s.float(0.72), s.float(0.28))), water_boost),
        s.float(0.0),
        s.float(1.0),
    )
    s.store(out_refl, s.umin(s.saturate_uint(s.fmul(ssr_f, s.float(127.0))), s.uint(127)))

    # Grazing Fresnel silhouette rim backscatter
    var nv_om = s.fsub(s.float(1.0), n_dot_v)
    var nv_om2 = s.fmul(nv_om, nv_om)
    var nv_om4 = s.fmul(nv_om2, nv_om2)
    var back_sun = s.fma(s.nmax(s.fsub(s.float(0.0), dot3(s, view_dir, to_key)), s.float(0.0)), s.float(0.45), s.float(0.55))
    var rim_w = s.fmul(s.fmul(nv_om4, back_sun), s.fmul(s.fmul(ao_ibl, ext_vis), s.float(0.14)))
    s.store(light_r, s.fma(key_color.x, rim_w, s.load_f(light_r)))
    s.store(light_g, s.fma(key_color.y, rim_w, s.load_f(light_g)))
    s.store(light_b, s.fma(key_color.z, rim_w, s.load_f(light_b)))
    s.end_if()

    # 3D Clustered Light Culling (128 Dynamic Point + Karis Tube Area Lights across 4 Z-Slices)
    var one = s.uint(1)
    var zero_u = s.uint(0)
    var total_tiles = s.load_word(tiles, s.iadd(s.imul(tile_index, s.uint(UInt32(TILE_ENTRY_DWORDS))), s.uint(3)))
    var cluster_count = constant_word(s, res, CONSTANT_V7_POST + 32)
    s.begin_if(s.ine(total_tiles, zero_u))
    s.begin_if(s.ine(cluster_count, zero_u))
    var z_sl = s.select_u(
        s.flt(depth, s.float(18.0)),
        s.uint(0),
        s.select_u(
            s.flt(depth, s.float(55.0)),
            s.uint(1),
            s.select_u(s.flt(depth, s.float(145.0)), s.uint(2), s.uint(3)),
        ),
    )
    var froxel_idx = s.iadd(s.imul(tile_index, s.uint(UInt32(CLUSTER_Z_SLICES))), z_sl)
    var mask_base = s.iadd(
        s.imul(total_tiles, s.uint(UInt32(TILE_ENTRY_DWORDS))),
        s.imul(froxel_idx, s.uint(UInt32(CLUSTER_WORDS_PER_SLICE))),
    )
    var w_iter = s.uvar(zero_u)
    s.begin_loop()
    var wi = s.load_u(w_iter)
    s.break_if(s.uge(wi, s.uint(UInt32(CLUSTER_WORDS_PER_SLICE))))
    s.store(w_iter, s.iadd(wi, one))
    var mask_var = s.uvar(s.load_word(tiles, s.iadd(mask_base, wi)))
    var bit_base = s.shl(wi, s.uint(5))
    s.begin_loop()
    var cur_m = s.load_u(mask_var)
    s.break_if(s.ieq(cur_m, zero_u))
    var bit_i = s.find_lsb(cur_m)
    s.store(mask_var, s.bit_and(cur_m, s.isub(cur_m, one)))
    var l_idx = s.iadd(bit_base, bit_i)
    s.continue_if(s.uge(l_idx, cluster_count))

    var cw = s.iadd(s.uint(UInt32(CONSTANT_CLUSTER_LIGHTS // 4)), s.imul(l_idx, s.uint(12)))
    var l0_x = s.fsub(light_float(s, res, cw, 0), world.x)
    var l0_y = s.fsub(light_float(s, res, cw, 1), world.y)
    var l0_z = s.fsub(light_float(s, res, cw, 2), world.z)
    var inv_r2 = light_float(s, res, cw, 3)
    var l_col_r = light_float(s, res, cw, 4)
    var l_col_g = light_float(s, res, cw, 5)
    var l_col_b = light_float(s, res, cw, 6)
    var seg_len2 = light_float(s, res, cw, 7)
    var sd_x = light_float(s, res, cw, 8)
    var sd_y = light_float(s, res, cw, 9)
    var sd_z = light_float(s, res, cw, 10)
    var seg_half = light_float(s, res, cw, 11)

    var l0_v = Vec3Ids(l0_x, l0_y, l0_z)
    var sd_v = Vec3Ids(sd_x, sd_y, sd_z)
    var is_tube = s.fgt(seg_len2, s.float(0.0001))
    var t_diff = s.select_f(
        is_tube,
        s.nclamp(s.fdiv(s.fsub(s.float(0.0), dot3(s, l0_v, sd_v)), s.nmax(seg_len2, s.float(0.0001))), s.float(0.0), s.float(1.0)),
        s.float(0.0),
    )
    var td_x = s.fma(sd_x, t_diff, l0_x)
    var td_y = s.fma(sd_y, t_diff, l0_y)
    var td_z = s.fma(sd_z, t_diff, l0_z)
    var td_v = Vec3Ids(td_x, td_y, td_z)
    var dist2 = s.fadd(dot3(s, td_v, td_v), s.float(0.0001))
    var fall0 = s.nmax(s.fsub(s.float(1.0), s.fmul(inv_r2, dist2)), s.float(0.0))
    var falloff = s.fmul(fall0, fall0)
    var inv_dist = s.inverse_sqrt(dist2)
    var raw_ndl = s.nmax(s.fmul(dot3(s, normal, td_v), inv_dist), s.float(0.0))
    var wrapped = s.fadd(s.fmul(raw_ndl, s.float(0.76)), s.float(0.24))
    var amount = s.fmul(s.fmul(s.fmul(falloff, wrapped), occlusion), kd_val)
    s.store(light_r, s.fma(l_col_r, amount, s.load_f(light_r)))
    s.store(light_g, s.fma(l_col_g, amount, s.load_f(light_g)))
    s.store(light_b, s.fma(l_col_b, amount, s.load_f(light_b)))

    # Karis Representative-Point Tube / Point GGX Specular
    s.begin_if(has_pbr)
    var ts_x = s.fvar(td_x)
    var ts_y = s.fvar(td_y)
    var ts_z = s.fvar(td_z)
    var sp_fall = s.fvar(falloff)
    var sp_inv_d = s.fvar(inv_dist)
    var tube_norm = s.fvar(s.float(1.0))
    s.begin_if(is_tube)
    var r_dot_sd = dot3(s, refl_dir, sd_v)
    var r_dot_l0 = dot3(s, refl_dir, l0_v)
    var l0_dot_sd = dot3(s, l0_v, sd_v)
    var denom_k = s.nmax(s.fsub(seg_len2, s.fmul(r_dot_sd, r_dot_sd)), s.float(0.0001))
    var t_spec = s.nclamp(s.fdiv(s.fsub(s.fmul(r_dot_l0, r_dot_sd), l0_dot_sd), denom_k), s.float(0.0), s.float(1.0))
    var sx_k = s.fma(sd_x, t_spec, l0_x)
    var sy_k = s.fma(sd_y, t_spec, l0_y)
    var sz_k = s.fma(sd_z, t_spec, l0_z)
    var sd2_k = s.fadd(dot3(s, Vec3Ids(sx_k, sy_k, sz_k), Vec3Ids(sx_k, sy_k, sz_k)), s.float(0.0001))
    var sinv_k = s.inverse_sqrt(sd2_k)
    var sf0 = s.nmax(s.fsub(s.float(1.0), s.fmul(inv_r2, sd2_k)), s.float(0.0))
    s.store(ts_x, sx_k)
    s.store(ts_y, sy_k)
    s.store(ts_z, sz_k)
    s.store(sp_inv_d, sinv_k)
    s.store(sp_fall, s.fmul(sf0, sf0))
    var a_pr = s.nclamp(s.fma(seg_half, s.fmul(sinv_k, s.float(0.5)), alpha_r), s.nmax(alpha_r, s.float(0.02)), s.float(1.0))
    var ratio_a = s.fdiv(s.nmax(alpha_r, s.float(0.02)), a_pr)
    s.store(tube_norm, s.fmul(ratio_a, ratio_a))
    s.end_if()

    var s_inv = s.load_f(sp_inv_d)
    var s_dir = Vec3Ids(s.fmul(s.load_f(ts_x), s_inv), s.fmul(s.load_f(ts_y), s_inv), s.fmul(s.load_f(ts_z), s_inv))
    var s_ndl = s.nmax(dot3(s, normal, s_dir), s.float(0.0))
    var s_f = s.fmul(s.load_f(sp_fall), s.load_f(tube_norm))
    s.begin_if(s.flt(s.float(0.0005), s.fmul(s_f, s_ndl)))
    var atten_col = Vec3Ids(s.fmul(l_col_r, s_f), s.fmul(l_col_g, s_f), s.fmul(l_col_b, s_f))
    accumulate_ggx_specular(
        s, normal, tan_dir, bit_dir, view_dir, s_dir, atten_col, n_dot_v, s_ndl,
        ax, ay, ms_comp, k_smith, one_minus_k, f0_r, f0_g, f0_b, spec_r, spec_g, spec_b,
    )
    s.end_if()
    s.end_if()
    s.end_loop()
    s.end_loop()
    s.end_if()
    s.begin_else()
    # Legacy 32-light fallback when cluster froxel masks are unpopulated
    var count = constant_word(s, res, CONSTANT_WORLD + 24)
    var light = s.uvar(s.uint(0))
    s.begin_loop()
    var index = s.load_u(light)
    s.break_if(s.uge(index, count))
    s.store(light, s.iadd(index, one))
    s.continue_if(s.ieq(s.bit_and(s.shr(tile_lights, index), one), s.uint(0)))
    var word = s.iadd(s.uint(UInt32(CONSTANT_LIGHTS // 4)), s.shl(index, s.uint(3)))
    var to_lx = s.fsub(light_float(s, res, word, 0), world.x)
    var to_ly = s.fsub(light_float(s, res, word, 1), world.y)
    var to_lz = s.fsub(light_float(s, res, word, 2), world.z)
    var to_light = Vec3Ids(to_lx, to_ly, to_lz)
    var distance_squared = s.fadd(dot3(s, to_light, to_light), s.float(0.0001))
    var falloff_l = s.nmax(s.fsub(s.float(1.0), s.fmul(light_float(s, res, word, 7), distance_squared)), s.float(0))
    falloff_l = s.fmul(falloff_l, falloff_l)
    var inverse_distance = s.inverse_sqrt(distance_squared)
    var raw_ndl_l = s.nmax(s.fmul(dot3(s, normal, to_light), inverse_distance), s.float(0))
    var wrapped_l = s.fadd(s.fmul(raw_ndl_l, s.float(0.76)), s.float(0.24))
    var amount_l = s.fmul(s.fmul(s.fmul(falloff_l, wrapped_l), occlusion), kd_val)
    var lc_r = light_float(s, res, word, 4)
    var lc_g = light_float(s, res, word, 5)
    var lc_b = light_float(s, res, word, 6)
    s.store(light_r, s.fma(lc_r, amount_l, s.load_f(light_r)))
    s.store(light_g, s.fma(lc_g, amount_l, s.load_f(light_g)))
    s.store(light_b, s.fma(lc_b, amount_l, s.load_f(light_b)))
    s.begin_if(s.logical_and(has_pbr, s.flt(s.float(0.001), s.fmul(falloff_l, raw_ndl_l))))
    var l_dir = Vec3Ids(s.fmul(to_lx, inverse_distance), s.fmul(to_ly, inverse_distance), s.fmul(to_lz, inverse_distance))
    var atten_col_l = Vec3Ids(s.fmul(lc_r, falloff_l), s.fmul(lc_g, falloff_l), s.fmul(lc_b, falloff_l))
    accumulate_ggx_specular(
        s, normal, tan_dir, bit_dir, view_dir, l_dir, atten_col_l, n_dot_v, raw_ndl_l,
        ax, ay, ms_comp, k_smith, one_minus_k, f0_r, f0_g, f0_b, spec_r, spec_g, spec_b,
    )
    s.end_if()
    s.end_loop()
    s.end_if()


def csm_clear_shader() -> List[UInt32]:
    """Clear the 2-Cascade 1024x1024 shadow depth buffer (4 words per thread)."""
    var s = SpirvModule(64, 1, CSM_CLEAR_PUSH_WORDS)
    var shadow_map = s.buffer(BIND_SHADOW_MAP)
    var tid = s.invocation_axis(0)
    var quads = s.push_word(CSM_CLEAR_PUSH_QUADS)
    s.return_if(s.uge(tid, quads))
    var base = s.shl(tid, s.uint(2))
    var zero = s.uint(0)
    s.store_word(shadow_map, base, zero)
    s.store_word(shadow_map, s.iadd(base, s.uint(1)), zero)
    s.store_word(shadow_map, s.iadd(base, s.uint(2)), zero)
    s.store_word(shadow_map, s.iadd(base, s.uint(3)), zero)
    return s.finish()


def csm_raster_shader() -> List[UInt32]:
    """Rasterize 16x16 shadow-map tiles into Cascade 0 / Cascade 1 using OpAtomicUMax."""
    var s = SpirvModule(CSM_TILE_SIZE, 8, CSM_RASTER_PUSH_WORDS)
    var shadow_map = s.buffer(BIND_SHADOW_MAP)
    var gx_id = s.invocation_axis(0)
    var gy_id = s.invocation_axis(1)
    var group_x = s.shr(gx_id, s.uint(4))
    var local_x = s.bit_and(gx_id, s.uint(15))
    var group_y = s.shr(gy_id, s.uint(3))
    var local_y = s.bit_and(gy_id, s.uint(7))
    var item_idx = s.iadd(s.imul(group_y, s.push_word(CSM_RASTER_PUSH_GROUPS_X)), group_x)
    s.return_if(s.uge(item_idx, s.push_word(CSM_RASTER_PUSH_COUNT)))

    var item_word = s.iadd(s.push_word(CSM_RASTER_PUSH_ITEMS), s.shl(item_idx, s.uint(1)))
    var rec_idx = s.load_word(shadow_map, item_word)
    var packed_tile = s.load_word(shadow_map, s.iadd(item_word, s.uint(1)))
    var tile_x = s.bit_and(packed_tile, s.uint(255))
    var tile_y = s.bit_and(s.shr(packed_tile, s.uint(8)), s.uint(255))
    var cascade = s.bit_and(s.shr(packed_tile, s.uint(16)), s.uint(1))

    var rec_base = s.iadd(
        s.push_word(CSM_RASTER_PUSH_RECORDS),
        s.imul(rec_idx, s.uint(UInt32(CSM_RECORD_DWORDS))),
    )
    var wa_a = s.load_float(shadow_map, rec_base)
    var wa_b = s.load_float(shadow_map, s.iadd(rec_base, s.uint(1)))
    var wa_c = s.load_float(shadow_map, s.iadd(rec_base, s.uint(2)))
    var wb_a = s.load_float(shadow_map, s.iadd(rec_base, s.uint(3)))
    var wb_b = s.load_float(shadow_map, s.iadd(rec_base, s.uint(4)))
    var wb_c = s.load_float(shadow_map, s.iadd(rec_base, s.uint(5)))
    var z_a = s.load_float(shadow_map, s.iadd(rec_base, s.uint(6)))
    var z_b = s.load_float(shadow_map, s.iadd(rec_base, s.uint(7)))
    var z_c = s.load_float(shadow_map, s.iadd(rec_base, s.uint(8)))

    var col_u = s.iadd(s.shl(tile_x, s.uint(4)), local_x)
    var row_base = s.iadd(s.shl(tile_y, s.uint(4)), local_y)
    var px = s.fadd(s.float_from_uint(col_u), s.float(0.5))
    var casc_off = s.imul(cascade, s.uint(UInt32(CSM_CASCADE_WORDS)))
    var edge_tol = s.float(-0.0002)

    for sub in range(2):
        var row_u = row_base if sub == 0 else s.iadd(row_base, s.uint(8))
        var py = s.fadd(s.float_from_uint(row_u), s.float(0.5))
        var wa = s.fma(wa_b, py, s.fma(wa_a, px, wa_c))
        var wb = s.fma(wb_b, py, s.fma(wb_a, px, wb_c))
        var wc = s.fsub(s.fsub(s.float(1.0), wa), wb)
        var inside = s.logical_and(
            s.logical_and(s.fge(wa, edge_tol), s.fge(wb, edge_tol)),
            s.fge(wc, edge_tol),
        )
        s.begin_if(inside)
        var z_sun = s.fma(z_b, py, s.fma(z_a, px, z_c))
        s.begin_if(s.logical_and(s.fgt(z_sun, s.float(0.0005)), s.flt(z_sun, s.float(0.9995))))
        var z_u = s.saturate_uint(s.fmul(z_sun, s.float(16777215.0)))
        var idx = s.iadd(casc_off, s.iadd(s.imul(row_u, s.uint(UInt32(CSM_RES))), col_u))
        _ = s.atomic_umax_word(shadow_map, idx, z_u)
        s.end_if()
        s.end_if()
    return s.finish()


def raster_shader() -> List[UInt32]:
    """Walk this pixel's tile list in order, accumulating True Linear Float32 HDR + G-buffer."""
    var s = SpirvModule(GROUP_WIDTH, GROUP_HEIGHT, RASTER_PUSH_WORDS)
    var res = Resources(s)
    var lists = s.buffer(BIND_LISTS)
    var tiles = s.buffer(BIND_TILES)
    var target = s.buffer(BIND_TARGET)
    var column = s.invocation_axis(0)
    var line = s.invocation_axis(1)
    var row_pixels = s.push_word(RASTER_PUSH_ROW)
    s.return_if(s.uge(column, s.push_word(RASTER_PUSH_WIDTH)))
    s.return_if(s.uge(line, s.push_word(RASTER_PUSH_HEIGHT)))
    var x_base = s.float_from_uint(column)
    var y_base = s.float_from_uint(line)
    var jitter_x = constant_float(s, res, CONSTANT_PREV_CAMERA + 60)
    var jitter_y = constant_float(s, res, CONSTANT_PREV_CAMERA + 64)
    var x = s.fadd(x_base, jitter_x)
    var y = s.fadd(y_base, jitter_y)
    var one = s.uint(1)

    var tile_index = s.iadd(s.shr(column, s.uint(5)), s.imul(s.shr(line, s.uint(5)), s.push_word(RASTER_PUSH_COLUMNS)))
    var tile_word = s.imul(tile_index, s.uint(UInt32(TILE_ENTRY_DWORDS)))
    var first = s.shr(s.load_word(tiles, tile_word), s.uint(2))
    var count = s.load_word(tiles, s.iadd(tile_word, one))
    var tile_lights = s.load_word(tiles, s.iadd(tile_word, s.uint(2)))
    var bloom_distance = s.shr(constant_word(s, res, CONSTANT_BLOOM_DISTANCE), s.uint(2))
    var effects = constant_word(s, res, CONSTANT_EFFECTS)

    var camera = constant_vector(s, res, CONSTANT_CAMERA)
    var right = constant_vector(s, res, CONSTANT_CAMERA + 12)
    var up = constant_vector(s, res, CONSTANT_CAMERA + 24)
    var forward = constant_vector(s, res, CONSTANT_CAMERA + 36)
    var half_width = constant_float(s, res, CONSTANT_CAMERA + 48)
    var half_height = constant_float(s, res, CONSTANT_CAMERA + 52)
    var inverse_focal = constant_float(s, res, CONSTANT_CAMERA + 56)
    var half = s.float(0.5)
    var view_x = s.fmul(s.fsub(s.fadd(half, x), half_width), inverse_focal)
    var view_y = s.fmul(s.fsub(half_height, s.fadd(half, y)), inverse_focal)
    var ray = Vec3Ids(
        s.fma(up.x, view_y, s.fma(right.x, view_x, forward.x)),
        s.fma(up.y, view_y, s.fma(right.y, view_x, forward.y)),
        s.fma(up.z, view_y, s.fma(right.z, view_x, forward.z)),
    )
    var to_key = constant_vector(s, res, CONSTANT_KEY)
    var key_color = constant_vector(s, res, CONSTANT_KEY + 12)

    var dither_x = constant_float(s, res, CONSTANT_DITHER)
    var dither_y = constant_float(s, res, CONSTANT_DITHER + 4)
    var dither_scale = constant_float(s, res, CONSTANT_DITHER + 8)
    var dither_offset = constant_float(s, res, CONSTANT_DITHER + 12)
    var dither_near = constant_float(s, res, CONSTANT_DITHER + 16)
    var dither_dx = s.fsub(x_base, dither_x)
    var dither_dy = s.fsub(y_base, dither_y)
    var dither_value = s.fma(s.fma(dither_dy, dither_dy, s.fmul(dither_dx, dither_dx)), dither_scale, dither_offset)
    var dither_v = s.bit_xor(column, line)
    var dither_level = s.bit_or(
        s.bit_or(s.shl(s.bit_and(dither_v, one), s.uint(3)), s.shl(s.bit_and(line, one), s.uint(2))),
        s.bit_or(s.bit_and(dither_v, s.uint(2)), s.bit_and(s.shr(line, one), one)),
    )
    var dither_below = s.flt(dither_value, s.float_from_uint(dither_level))

    var clear = s.push_word(RASTER_PUSH_CLEAR)
    var fog_r = s.float_from_uint(channel(s, clear, 16))
    var fog_g = s.float_from_uint(channel(s, clear, 8))
    var fog_b = s.float_from_uint(channel(s, clear, 0))
    var hdr_r = s.fvar(fog_r)
    var hdr_g = s.fvar(fog_g)
    var hdr_b = s.fvar(fog_b)
    var touched = s.uvar(s.uint(0))
    var nearest_depth = s.fvar(s.float_bits(0x3089705F))
    var bloom = s.uvar(s.uint(0))
    var edge = s.float_bits(0xB8D1B717)

    var gbuf_ox = s.uvar(s.uint(128))
    var gbuf_oz = s.uvar(s.uint(128))
    var gbuf_rough = s.uvar(s.uint(255))
    var gbuf_refl = s.uvar(s.uint(0))
    var surf_ox = s.uvar(s.uint(128))
    var surf_oz = s.uvar(s.uint(128))
    var surf_rough = s.uvar(s.uint(255))
    var surf_refl = s.uvar(s.uint(0))

    var light_r = s.fvar(s.float(0))
    var light_g = s.fvar(s.float(0))
    var light_b = s.fvar(s.float(0))
    var spec_r = s.fvar(s.float(0))
    var spec_g = s.fvar(s.float(0))
    var spec_b = s.fvar(s.float(0))
    var sampled = s.uvar(s.uint(0xFFFFFFFF))
    var uv_u = s.fvar(s.float(0.5))
    var uv_v = s.fvar(s.float(0.5))
    var du_dx_var = s.fvar(s.float(0.0))
    var dv_dx_var = s.fvar(s.float(0.0))
    var du_dy_var = s.fvar(s.float(0.0))
    var dv_dy_var = s.fvar(s.float(0.0))
    var tex_lod_var = s.fvar(s.float(0.0))
    var refract_packed = s.uvar(s.uint(0))
    var vel_packed = s.uvar(s.uint(0))
    var base_r = s.fvar(s.float(0))
    var base_g = s.fvar(s.float(0))
    var base_b = s.fvar(s.float(0))
    var emissive_var = s.fvar(s.float(0))
    var tex_alpha_var = s.uvar(s.uint(255))
    var eff_alpha_var = s.uvar(s.uint(255))

    var entry = s.uvar(s.uint(0))
    s.begin_loop()
    var index = s.load_u(entry)
    s.break_if(s.uge(index, count))
    s.store(entry, s.iadd(index, one))
    var base = s.shr(s.load_word(lists, s.iadd(first, index)), s.uint(2))

    var weight_a = plane(s, res, base, RECORD_WEIGHT_A, x, y)
    var weight_b = plane(s, res, base, RECORD_WEIGHT_B, x, y)
    var weight_c = s.fsub(s.fsub(s.float(1.0), weight_a), weight_b)
    var inside = s.logical_and(s.logical_and(s.fge(weight_a, edge), s.fge(weight_b, edge)), s.fge(weight_c, edge))
    s.continue_if(s.logical_not(inside))

    var iz_a = s.bitcast_float(field(s, res, base, RECORD_INVERSE_Z))
    var iz_b = s.bitcast_float(field(s, res, base, RECORD_INVERSE_Z + 1))
    var iz_c = s.bitcast_float(field(s, res, base, RECORD_INVERSE_Z + 2))
    var inverse_z = s.fma(iz_b, y, s.fma(iz_a, x, iz_c))
    s.continue_if(s.logical_not(s.fgt(inverse_z, s.load_f(nearest_depth))))

    var depth = s.fdiv(s.float(1.0), inverse_z)
    var flags = field(s, res, base, RECORD_FLAGS)
    var occluder = s.ine(s.bit_and(flags, s.uint(DITHER_FLAG)), s.uint(0))
    s.continue_if(s.logical_and(occluder, s.logical_and(s.fgt(inverse_z, dither_near), dither_below)))

    s.store(sampled, s.uint(0xFFFFFFFF))
    s.store(uv_u, s.float(0.5))
    s.store(uv_v, s.float(0.5))
    s.store(du_dx_var, s.float(0.0))
    s.store(dv_dx_var, s.float(0.0))
    s.store(du_dy_var, s.float(0.0))
    s.store(dv_dy_var, s.float(0.0))
    s.store(tex_lod_var, s.float(0.0))

    s.begin_if(s.ine(s.bit_and(flags, s.uint(TEXTURED_FLAG | NORM_ORM_FLAG)), s.uint(0)))
    var u_a = s.bitcast_float(field(s, res, base, RECORD_U))
    var u_b = s.bitcast_float(field(s, res, base, RECORD_U + 1))
    var u_c = s.bitcast_float(field(s, res, base, RECORD_U + 2))
    var v_a = s.bitcast_float(field(s, res, base, RECORD_V))
    var v_b = s.bitcast_float(field(s, res, base, RECORD_V + 1))
    var v_c = s.bitcast_float(field(s, res, base, RECORD_V + 2))
    var uz = s.fma(u_b, y, s.fma(u_a, x, u_c))
    var vz = s.fma(v_b, y, s.fma(v_a, x, v_c))
    var u_raw = s.fmul(depth, uz)
    var v_raw = s.fmul(depth, vz)
    var u_frac = s.fract(u_raw)
    var v_frac = s.fract(v_raw)
    s.store(uv_u, u_frac)
    s.store(uv_v, v_frac)

    var dudx = s.fmul(depth, s.fsub(u_a, s.fmul(u_raw, iz_a)))
    var dudy = s.fmul(depth, s.fsub(u_b, s.fmul(u_raw, iz_b)))
    var dvdx = s.fmul(depth, s.fsub(v_a, s.fmul(v_raw, iz_a)))
    var dvdy = s.fmul(depth, s.fsub(v_b, s.fmul(v_raw, iz_b)))
    s.store(du_dx_var, dudx)
    s.store(du_dy_var, dudy)
    s.store(dv_dx_var, dvdx)
    s.store(dv_dy_var, dvdy)

    s.begin_if(s.ine(s.bit_and(flags, s.uint(TEXTURED_FLAG)), s.uint(0)))
    s.begin_if(s.ine(s.bit_and(effects, s.uint(EFFECT_BILINEAR)), s.uint(0)))
    s.begin_if(s.ine(s.bit_and(flags, s.uint(MIPMAPPED_FLAG)), s.uint(0)))
    s.store(sampled, sample_anisotropic_mipmapped(s, res, base, u_frac, v_frac, dudx, dvdx, dudy, dvdy, tex_lod_var))
    s.begin_else()
    var texture_u = s.fmul(s.bitcast_float(field(s, res, base, RECORD_TEXTURE_FLOAT_WIDTH)), u_frac)
    var texture_v = s.fmul(s.bitcast_float(field(s, res, base, RECORD_TEXTURE_FLOAT_HEIGHT)), v_frac)
    s.store(sampled, sample_bilinear(s, res, base, texture_u, texture_v))
    s.end_if()
    s.begin_else()
    var tex_u_n = s.fmul(s.bitcast_float(field(s, res, base, RECORD_TEXTURE_FLOAT_WIDTH)), u_frac)
    var tex_v_n = s.fmul(s.bitcast_float(field(s, res, base, RECORD_TEXTURE_FLOAT_HEIGHT)), v_frac)
    var nearest_x = s.umin(s.saturate_uint(tex_u_n), field(s, res, base, RECORD_TEXTURE_MAX_X))
    var nearest_y = s.umin(s.saturate_uint(tex_v_n), field(s, res, base, RECORD_TEXTURE_MAX_Y))
    s.store(sampled, texel(s, res, base, nearest_x, nearest_y))
    s.end_if()
    s.end_if()
    s.end_if()

    var tint = field(s, res, base, RECORD_TINT)
    var pixel = s.load_u(sampled)
    for which in range(3):
        var shift = 16 - which * 8
        var product = s.imul(channel(s, tint, shift), channel(s, pixel, shift))
        var bval = s.float_from_uint(divide_by_255(s, product))
        s.store(base_r if which == 0 else (base_g if which == 1 else base_b), bval)

    s.store(emissive_var, s.fmul(s.float(1.25), plane(s, res, base, RECORD_EMISSIVE, x, y)))
    s.begin_if(s.ine(s.bit_and(flags, s.uint(BLACKBODY_FLAG)), s.uint(0)))
    var bb_heat = s.bitcast_float(field(s, res, base, RECORD_BLACKBODY_HEAT))
    var cell_n = s.fmul(s.float_from_uint(channel(s, pixel, 16)), s.float_bits(0x3B808081))
    var t_gain = s.fma(bb_heat, s.float(1.28), s.float(0.30))
    var t_cool = s.fmul(s.fsub(s.float(1.0), bb_heat), s.float(0.26))
    var t_kelvin = s.nclamp(s.fsub(s.fmul(cell_n, t_gain), t_cool), s.float(0.0), s.float(1.35))
    var r_th = s.fma(s.nclamp(s.fmul(t_kelvin, s.float(2.25)), s.float(0.0), s.float(1.0)), s.float(211.0), s.float(44.0))
    var g_th = s.fma(s.nclamp(s.fmul(s.fsub(t_kelvin, s.float(0.22)), s.float(1.42)), s.float(0.0), s.float(1.0)), s.float(212.0), s.float(36.0))
    var b_th = s.fma(s.nclamp(s.fmul(s.fsub(t_kelvin, s.float(0.62)), s.float(1.45)), s.float(0.0), s.float(1.0)), s.float(184.0), s.float(34.0))
    s.store(base_r, s.fmul(r_th, s.fmul(s.float_from_uint(channel(s, tint, 16)), s.float_bits(0x3B808081))))
    s.store(base_g, s.fmul(g_th, s.fmul(s.float_from_uint(channel(s, tint, 8)), s.float_bits(0x3B808081))))
    s.store(base_b, s.fmul(b_th, s.fmul(s.float_from_uint(channel(s, tint, 0)), s.float_bits(0x3B808081))))
    var bb_em = s.fmul(s.fmul(t_kelvin, t_kelvin), s.fmul(bb_heat, s.float(2.90)))
    s.store(emissive_var, s.fadd(s.load_f(emissive_var), bb_em))
    s.end_if()

    s.store(light_r, s.float(0))
    s.store(light_g, s.float(0))
    s.store(light_b, s.float(0))
    s.store(spec_r, s.float(0))
    s.store(spec_g, s.float(0))
    s.store(spec_b, s.float(0))
    s.store(surf_ox, s.uint(128))
    s.store(surf_oz, s.uint(128))
    s.store(surf_rough, s.uint(255))
    s.store(surf_refl, s.uint(0))
    s.begin_if(s.ine(s.bit_and(flags, s.uint(PER_PIXEL_FLAG)), s.uint(0)))
    per_pixel_light(
        s, res, tiles, tile_index, base, flags, x, y, depth, s.load_f(nearest_depth),
        ray, camera, right, up, inverse_focal,
        to_key, key_color, effects, tile_lights,
        s.load_f(uv_u), s.load_f(uv_v),
        s.load_f(du_dx_var), s.load_f(dv_dx_var), s.load_f(du_dy_var), s.load_f(dv_dy_var), s.load_f(tex_lod_var),
        base_r, base_g, base_b, light_r, light_g, light_b, spec_r, spec_g, spec_b,
        surf_ox, surf_oz, surf_rough, surf_refl,
    )
    s.end_if()

    var bases = List[UInt32]()
    bases.append(s.load_f(base_r))
    bases.append(s.load_f(base_g))
    bases.append(s.load_f(base_b))

    var emissive = s.load_f(emissive_var)
    var fog = plane(s, res, base, RECORD_FOG, x, y)
    var clearness = s.fsub(s.float(1.0), fog)

    # Linear scene-referred Float32 HDR radiance (preserved unclipped for compose-stage AgX + bloom)
    var raw_hdrs = List[UInt32]()
    for which in range(3):
        var base_color = bases[which]
        var fog_color = fog_r if which == 0 else (fog_g if which == 1 else fog_b)
        var pixel_light = s.load_f(light_r if which == 0 else (light_g if which == 1 else light_b))
        var pixel_spec = s.load_f(spec_r if which == 0 else (spec_g if which == 1 else spec_b))
        var total = s.fadd(s.fadd(plane(s, res, base, RECORD_LIGHT + which * 3, x, y), emissive), pixel_light)
        var lit = s.fmul(s.fma(base_color, total, pixel_spec), clearness)
        raw_hdrs.append(s.nmax(s.fadd(lit, s.fmul(fog_color, fog)), s.float(0.0)))

    # Effective alpha: triangle * tint * texture (+ optional cellular UV alpha erosion & inverse-Z soft particles).
    var scaled = s.imul(s.bit_and(flags, s.uint(255)), s.shr(tint, s.uint(24)))
    var triangle_alpha = s.shr(s.iadd(s.iadd(scaled, s.shr(scaled, s.uint(8))), one), s.uint(8))
    s.store(tex_alpha_var, s.shr(pixel, s.uint(24)))
    s.begin_if(s.ine(s.bit_and(flags, s.uint(ALPHA_ERODE_FLAG)), s.uint(0)))
    var erode = s.nclamp(s.bitcast_float(field(s, res, base, RECORD_ERODE_THRESHOLD)), s.float(0.0), s.float(0.96))
    var tex_af = s.fmul(s.float_from_uint(s.load_u(tex_alpha_var)), s.float_bits(0x3B808081))
    var inv_span = s.fdiv(s.float(1.0), s.nmax(s.fsub(s.float(1.0), erode), s.float(0.06)))
    var er_t = s.nclamp(s.fmul(s.fsub(tex_af, erode), inv_span), s.float(0.0), s.float(1.0))
    var er_s = s.fmul(s.fmul(er_t, er_t), s.fsub(s.float(3.0), s.fmul(s.float(2.0), er_t)))
    s.store(tex_alpha_var, float_to_byte(s, s.fmul(er_s, s.float(255.0))))
    s.end_if()

    s.store(eff_alpha_var, divide_by_255(s, s.imul(triangle_alpha, s.load_u(tex_alpha_var))))
    s.begin_if(s.ine(s.bit_and(flags, s.uint(SOFT_PARTICLE_FLAG)), s.uint(0)))
    var inv_soft = s.bitcast_float(field(s, res, base, RECORD_SOFT_INV_DIST))
    var scene_z = s.fdiv(s.float(1.0), s.nmax(s.load_f(nearest_depth), s.float_bits(0x3089705F)))
    var dz_t = s.nclamp(s.fmul(s.fsub(scene_z, depth), inv_soft), s.float(0.0), s.float(1.0))
    var near_t = s.nclamp(s.fmul(s.fsub(depth, s.float(0.22)), s.float(0.85)), s.float(0.0), s.float(1.0))
    var soft_t = s.fmul(dz_t, near_t)
    var soft_h = s.fmul(s.fmul(soft_t, soft_t), s.fsub(s.float(3.0), s.fmul(s.float(2.0), soft_t)))
    s.store(eff_alpha_var, float_to_byte(s, s.fmul(s.float_from_uint(s.load_u(eff_alpha_var)), soft_h)))
    s.end_if()

    var alpha = s.load_u(eff_alpha_var)
    s.continue_if(s.ieq(alpha, s.uint(0)))
    s.store(touched, one)

    var opacity = s.fmul(s.float_bits(0x3B808081), s.float_from_uint(alpha))
    s.begin_if(s.ine(s.bit_and(flags, s.uint(REFRACT_FLAG)), s.uint(0)))
    var r_amp = s.fmul(s.bitcast_float(field(s, res, base, RECORD_REFRACT_STRENGTH)), opacity)
    var du_px = s.nclamp(s.fmul(s.fsub(s.load_f(uv_u), s.float(0.5)), r_amp), s.float(-14.0), s.float(14.0))
    var dv_px = s.nclamp(s.fmul(s.fsub(s.load_f(uv_v), s.float(0.5)), r_amp), s.float(-14.0), s.float(14.0))
    var dx_b = clamp_byte(s, float_to_byte(s, s.fadd(du_px, s.float(128.5))))
    var dy_b = clamp_byte(s, float_to_byte(s, s.fadd(dv_px, s.float(128.5))))
    s.store(refract_packed, s.bit_or(s.uint(0x00010000), s.bit_or(s.shl(dy_b, s.uint(8)), dx_b)))
    s.end_if()

    var remaining = s.fsub(s.float(1.0), opacity)
    var is_additive = s.ine(s.bit_and(flags, s.uint(ADDITIVE_FLAG)), s.uint(0))
    var no_z_write = s.ine(s.bit_and(flags, s.uint(NO_DEPTH_WRITE_FLAG)), s.uint(0))
    var has_pbr_px = s.ine(s.bit_and(flags, s.uint(PBR_FLAG)), s.uint(0))

    s.begin_if(is_additive)
    for which in range(3):
        var hvar = hdr_r if which == 0 else (hdr_g if which == 1 else hdr_b)
        s.store(hvar, s.fma(raw_hdrs[which], opacity, s.load_f(hvar)))
    s.begin_else()
    s.begin_if(s.ieq(alpha, s.uint(255)))
    s.store(hdr_r, raw_hdrs[0])
    s.store(hdr_g, raw_hdrs[1])
    s.store(hdr_b, raw_hdrs[2])
    s.begin_else()
    for which in range(3):
        var hvar = hdr_r if which == 0 else (hdr_g if which == 1 else hdr_b)
        var sp_ch = s.load_f(spec_r if which == 0 else (spec_g if which == 1 else spec_b))
        var over = s.fmul(raw_hdrs[which], opacity)
        var under = s.fmul(s.load_f(hvar), remaining)
        var glass_glint = s.select_f(
            has_pbr_px,
            s.fmul(s.nmin(s.fmul(sp_ch, clearness), s.float(520.0)), s.fmul(remaining, s.float(0.88))),
            s.float(0.0),
        )
        s.store(hvar, s.fadd(s.fadd(over, under), glass_glint))
    s.end_if()
    s.end_if()

    s.begin_if(s.logical_and(s.ult(s.uint(127), alpha), s.logical_not(no_z_write)))
    s.store(nearest_depth, inverse_z)
    s.store(gbuf_ox, s.load_u(surf_ox))
    s.store(gbuf_oz, s.load_u(surf_oz))
    s.store(gbuf_rough, s.load_u(surf_rough))
    var eff_refl = float_to_byte(s, s.fmul(s.float_from_uint(s.load_u(surf_refl)), clearness))
    s.store(gbuf_refl, s.umin(eff_refl, s.uint(127)))
    s.begin_if(s.ine(s.bit_and(flags, s.uint(OBJECT_VELOCITY_FLAG)), s.uint(0)))
    var vx_oz = plane(s, res, base, RECORD_VEL_X, x, y)
    var vy_oz = plane(s, res, base, RECORD_VEL_Y, x, y)
    var dvx = s.nclamp(s.fmul(vx_oz, depth), s.float(-250.0), s.float(250.0))
    var dvy = s.nclamp(s.fmul(vy_oz, depth), s.float(-250.0), s.float(250.0))
    var qx = s.umax(s.umin(s.saturate_uint(s.fma(dvx, s.float(64.0), s.float(32768.5))), s.uint(65535)), one)
    var qy = s.umax(s.umin(s.saturate_uint(s.fma(dvy, s.float(64.0), s.float(32768.5))), s.uint(65535)), one)
    s.store(vel_packed, s.bit_or(qx, s.shl(qy, s.uint(16))))
    s.begin_else()
    s.store(vel_packed, s.uint(0))
    s.end_if()
    s.end_if()

    var spec_peak = s.nmax(s.load_f(spec_r), s.nmax(s.load_f(spec_g), s.load_f(spec_b)))
    var bright = s.logical_or(
        s.fge(emissive, s.float_bits(0x3E4CCCCD)),
        s.fge(spec_peak, s.float(135.0)),
    )
    s.begin_if(s.logical_and(bright, s.ult(s.uint(23), alpha)))
    var strength = s.fmul(opacity, s.fmul(s.float_bits(0x3F4CCCCD), emissive))
    var spec_op = s.select_f(has_pbr_px, s.float(1.0), opacity)
    var glow = s.uvar(s.uint(0))
    for which in range(3):
        var sp_c = s.load_f(spec_r if which == 0 else (spec_g if which == 1 else spec_b))
        var sp_bloom = s.fmul(spec_op, s.fmul(s.nmax(s.fsub(sp_c, s.float(122.0)), s.float(0.0)), s.float(0.66)))
        var value = float_to_byte(s, s.fadd(s.fmul(strength, bases[which]), sp_bloom))
        s.store(glow, s.bit_or(s.load_u(glow), s.shl(value, s.uint(UInt32(16 - which * 8)))))
    s.store(bloom, s.load_u(glow))
    s.end_if()
    s.end_loop()

    var word = s.iadd(s.imul(line, row_pixels), column)
    s.store_word(target, s.iadd(word, bloom_distance), s.load_u(bloom))
    s.store_word(target, s.iadd(word, s.imul(bloom_distance, s.uint(3))), s.bitcast_uint(s.load_f(nearest_depth)))
    s.store_word(target, s.iadd(word, s.imul(bloom_distance, s.uint(4))), s.load_u(refract_packed))
    s.store_word(target, s.iadd(word, s.imul(bloom_distance, s.uint(5))), s.bitcast_uint(s.load_f(hdr_r)))
    s.store_word(target, s.iadd(word, s.imul(bloom_distance, s.uint(6))), s.bitcast_uint(s.load_f(hdr_g)))
    s.store_word(target, s.iadd(word, s.imul(bloom_distance, s.uint(7))), s.bitcast_uint(s.load_f(hdr_b)))
    var gbuf_packed = s.bit_or(
        s.bit_or(s.load_u(gbuf_ox), s.shl(s.load_u(gbuf_oz), s.uint(8))),
        s.bit_or(
            s.shl(s.load_u(gbuf_rough), s.uint(16)),
            s.bit_or(s.shl(s.load_u(gbuf_refl), s.uint(24)), s.shl(s.load_u(touched), s.uint(31))),
        ),
    )
    s.store_word(target, s.iadd(word, s.imul(bloom_distance, s.uint(8))), gbuf_packed)
    s.store_word(target, s.iadd(word, s.imul(bloom_distance, s.uint(11))), s.load_u(vel_packed))
    return s.finish()


def histogram_shader() -> List[UInt32]:
    """64-bin log2 scene-HDR luminance histogram over touched surface pixels (striding 4x4)."""
    var s = SpirvModule(HIST_GROUP_SIZE, HIST_GROUP_SIZE, HIST_PUSH_WORDS)
    var target = s.buffer(BIND_TARGET)
    var cells = s.buffer(BIND_CELLS)
    var gx = s.invocation_axis(0)
    var gy = s.invocation_axis(1)
    var px = s.iadd(s.shl(gx, s.uint(2)), s.uint(1))
    var py = s.iadd(s.shl(gy, s.uint(2)), s.uint(1))
    s.return_if(s.uge(px, s.push_word(HIST_PUSH_WIDTH)))
    s.return_if(s.uge(py, s.push_word(HIST_PUSH_HEIGHT)))

    var row_w = s.push_word(HIST_PUSH_ROW)
    var stride = s.push_word(HIST_PUSH_PLANE_STRIDE)
    var idx = s.iadd(s.imul(py, row_w), px)
    var gbuf = s.load_word(target, s.iadd(s.imul(stride, s.uint(8)), idx))
    s.return_if(s.ieq(s.shr(gbuf, s.uint(31)), s.uint(0)))

    var hr = s.load_float(target, s.iadd(s.imul(stride, s.uint(5)), idx))
    var hg = s.load_float(target, s.iadd(s.imul(stride, s.uint(6)), idx))
    var hb = s.load_float(target, s.iadd(s.imul(stride, s.uint(7)), idx))
    var lum = s.fma(hb, s.float(0.16), s.fma(hg, s.float(0.58), s.fmul(hr, s.float(0.26))))
    var log_l = s.nclamp(s.log2(s.nmax(lum, s.float(1.0))), s.float(0.0), s.float(10.0))
    var bin_u = s.umin(s.saturate_uint(s.fmul(log_l, s.float(6.399))), s.uint(63))
    var hist_base = s.uint(UInt32(BLOOM_HIST_OFFSET // 4))
    _ = s.atomic_add_word(cells, s.iadd(hist_base, bin_u), s.uint(1))
    return s.finish()


def auto_exposure_shader() -> List[UInt32]:
    """12%..95% percentile-trimmed log2 luminance solver with temporal eye adaptation & bin reset."""
    var s = SpirvModule(1, 1, 0)
    var res = Resources(s)
    var cells = s.buffer(BIND_CELLS)
    var hist_base = s.uint(UInt32(BLOOM_HIST_OFFSET // 4))
    var zero_u = s.uint(0)
    var one_u = s.uint(1)

    var total_var = s.uvar(zero_u)
    var b_i = s.uvar(zero_u)
    s.begin_loop()
    var bi = s.load_u(b_i)
    s.break_if(s.uge(bi, s.uint(64)))
    s.store(b_i, s.iadd(bi, one_u))
    var c_b = s.load_word(cells, s.iadd(hist_base, bi))
    s.store(total_var, s.iadd(s.load_u(total_var), c_b))
    s.end_loop()

    var total = s.load_u(total_var)
    var speed = s.nclamp(constant_float(s, res, CONSTANT_V7_POST + 16), s.float(0.02), s.float(1.0))
    var exp_min = constant_float(s, res, CONSTANT_V7_POST + 20)
    var exp_max = constant_float(s, res, CONSTANT_V7_POST + 24)
    var tgt_override = constant_float(s, res, CONSTANT_V7_POST + 28)
    var target_exp = s.fvar(s.float(1.0))

    s.begin_if(s.uge(total, s.uint(16)))
    var total_f = s.float_from_uint(total)
    var lo_cut = s.fmul(total_f, s.float(0.12))
    var hi_cut = s.fmul(total_f, s.float(0.95))
    var accum_f = s.fvar(s.float(0.0))
    var w_sum = s.fvar(s.float(0.0))
    var k_sum = s.fvar(s.float(0.0))

    var b_j = s.uvar(zero_u)
    s.begin_loop()
    var bj = s.load_u(b_j)
    s.break_if(s.uge(bj, s.uint(64)))
    s.store(b_j, s.iadd(bj, one_u))
    var slot_j = s.iadd(hist_base, bj)
    var cnt_u = s.load_word(cells, slot_j)
    s.store_word(cells, slot_j, zero_u)
    var cnt_f = s.float_from_uint(cnt_u)
    var prev_a = s.load_f(accum_f)
    var next_a = s.fadd(prev_a, cnt_f)
    s.store(accum_f, next_a)
    var ov_lo = s.nmax(prev_a, lo_cut)
    var ov_hi = s.nmin(next_a, hi_cut)
    var keep = s.nmax(s.fsub(ov_hi, ov_lo), s.float(0.0))
    var bin_mid = s.fadd(s.float_from_uint(bj), s.float(0.5))
    s.store(w_sum, s.fma(bin_mid, keep, s.load_f(w_sum)))
    s.store(k_sum, s.fadd(s.load_f(k_sum), keep))
    s.end_loop()

    var mean_bin = s.fdiv(s.load_f(w_sum), s.nmax(s.load_f(k_sum), s.float(1.0)))
    var avg_luma = s.exp2(s.fmul(mean_bin, s.float(0.15625)))
    var use_ovr = s.fgt(tgt_override, s.float(0.1))
    var ovr_exp = s.fdiv(tgt_override, s.nmax(avg_luma, s.float(1.0)))
    var hi_exp = s.fdiv(s.float(96.0), s.nmax(avg_luma, s.float(1.0)))
    var lo_exp = s.fdiv(s.float(34.0), s.nmax(avg_luma, s.float(4.0)))
    var db_exp = s.select_f(
        s.fgt(avg_luma, s.float(96.0)),
        hi_exp,
        s.select_f(s.flt(avg_luma, s.float(34.0)), lo_exp, s.float(1.0)),
    )
    var raw_exp = s.select_f(use_ovr, ovr_exp, db_exp)
    s.store(target_exp, s.nclamp(raw_exp, exp_min, exp_max))
    s.begin_else()
    var b_c = s.uvar(zero_u)
    s.begin_loop()
    var bc = s.load_u(b_c)
    s.break_if(s.uge(bc, s.uint(64)))
    s.store(b_c, s.iadd(bc, one_u))
    s.store_word(cells, s.iadd(hist_base, bc), zero_u)
    s.end_loop()
    s.end_if()

    var exp_slot = s.iadd(hist_base, s.uint(64))
    var mag_slot = s.iadd(hist_base, s.uint(65))
    var magic = s.uint(0xA7E00001)
    var has_prev = s.ieq(s.load_word(cells, mag_slot), magic)
    var prev_exp = s.select_f(has_prev, s.load_float(cells, exp_slot), s.float(1.0))
    var next_exp = s.nclamp(
        s.fma(s.fsub(s.load_f(target_exp), prev_exp), speed, prev_exp),
        exp_min,
        exp_max,
    )
    s.store_word(cells, exp_slot, s.bitcast_uint(next_exp))
    s.store_word(cells, mag_slot, magic)
    return s.finish()


def bloom_reduce_shader() -> List[UInt32]:
    """Max-combine Linear Float32 HDR firefly-clamped highlights + emissive tags into bloom cells."""
    var s = SpirvModule(GROUP_WIDTH, GROUP_HEIGHT, REDUCE_PUSH_WORDS)
    var target = s.buffer(BIND_TARGET)
    var cells = s.buffer(BIND_CELLS)
    var column = s.invocation_axis(0)
    var line = s.invocation_axis(1)
    var cell_columns = s.push_word(REDUCE_PUSH_CELLS_X)
    s.return_if(s.uge(column, cell_columns))
    s.return_if(s.uge(line, s.push_word(REDUCE_PUSH_CELLS_Y)))
    var x = s.float_from_uint(column)
    var y = s.float_from_uint(line)
    var scale_x = s.push_float(REDUCE_PUSH_SCALE_X)
    var scale_y = s.push_float(REDUCE_PUSH_SCALE_Y)
    var last_x = s.push_word(REDUCE_PUSH_LAST_X)
    var last_y = s.push_word(REDUCE_PUSH_LAST_Y)
    var row_pixels = s.push_word(REDUCE_PUSH_ROW)
    var source = s.push_word(REDUCE_PUSH_SOURCE)
    var plane_r = s.imul(source, s.uint(5))
    var plane_g = s.imul(source, s.uint(6))
    var plane_b = s.imul(source, s.uint(7))
    var one = s.uint(1)
    var red = s.fvar(s.float(0.0))
    var green = s.fvar(s.float(0.0))
    var blue = s.fvar(s.float(0.0))
    var half = s.float(0.5)
    var outer = s.uvar(s.uint(0))
    s.begin_loop()
    var sample_y = s.load_u(outer)
    s.break_if(s.uge(sample_y, s.uint(UInt32(BLOOM_SAMPLES_Y))))
    s.store(outer, s.iadd(sample_y, one))
    var pixel_y = s.fadd(y, s.fmul(s.float_bits(0x3E2AAAAB), s.fadd(half, s.float_from_uint(sample_y))))
    var at_y = s.umin(s.saturate_uint(s.fmul(scale_y, pixel_y)), last_y)
    var inner = s.uvar(s.uint(0))
    s.begin_loop()
    var sample_x = s.load_u(inner)
    s.break_if(s.uge(sample_x, s.uint(UInt32(BLOOM_SAMPLES_X))))
    s.store(inner, s.iadd(sample_x, one))
    var pixel_x = s.fadd(x, s.fmul(s.float(0.125), s.fadd(half, s.float_from_uint(sample_x))))
    var at_x = s.umin(s.saturate_uint(s.fmul(scale_x, pixel_x)), last_x)
    var px = s.iadd(s.imul(at_y, row_pixels), at_x)
    var value = s.load_word(target, s.iadd(source, px))
    var hr = s.load_float(target, s.iadd(plane_r, px))
    var hg = s.load_float(target, s.iadd(plane_g, px))
    var hb = s.load_float(target, s.iadd(plane_b, px))
    var er = s.nmin(s.nmax(s.float_from_uint(channel(s, value, 16)), s.fmul(s.nmax(s.fsub(hr, s.float(210.0)), s.float(0.0)), s.float(0.38))), s.float(380.0))
    var eg = s.nmin(s.nmax(s.float_from_uint(channel(s, value, 8)), s.fmul(s.nmax(s.fsub(hg, s.float(210.0)), s.float(0.0)), s.float(0.38))), s.float(380.0))
    var eb = s.nmin(s.nmax(s.float_from_uint(channel(s, value, 0)), s.fmul(s.nmax(s.fsub(hb, s.float(210.0)), s.float(0.0)), s.float(0.38))), s.float(380.0))
    s.store(red, s.nmax(s.load_f(red), er))
    s.store(green, s.nmax(s.load_f(green), eg))
    s.store(blue, s.nmax(s.load_f(blue), eb))
    s.end_loop()
    s.end_loop()
    var cell = s.imul(s.iadd(s.imul(line, cell_columns), column), s.uint(3))
    s.store_word(cells, cell, s.bitcast_uint(s.load_f(red)))
    s.store_word(cells, s.iadd(cell, one), s.bitcast_uint(s.load_f(green)))
    s.store_word(cells, s.iadd(cell, s.uint(2)), s.bitcast_uint(s.load_f(blue)))
    return s.finish()


def bloom_blur_shader(horizontal: Bool) -> List[UInt32]:
    """One 5-tap [0.0625, 0.25, 0.375, 0.25, 0.0625] Gaussian axis of the bloom blur."""
    var s = SpirvModule(GROUP_WIDTH, GROUP_HEIGHT, BLUR_PUSH_WORDS)
    var cells = s.buffer(BIND_CELLS)
    var column = s.invocation_axis(0)
    var line = s.invocation_axis(1)
    var cell_columns = s.push_word(BLUR_PUSH_CELLS_X)
    var cell_rows = s.push_word(BLUR_PUSH_CELLS_Y)
    s.return_if(s.uge(column, cell_columns))
    s.return_if(s.uge(line, cell_rows))
    var source = s.push_word(BLUR_PUSH_SOURCE)
    var target = s.push_word(BLUR_PUSH_TARGET)
    var one = s.uint(1)
    var two = s.uint(2)
    var three = s.uint(3)
    var moving = column if horizontal else line
    var last = s.isub(cell_columns if horizontal else cell_rows, one)
    var before2 = s.smax(s.isub(moving, two), s.uint(0))
    var before = s.smax(s.isub(moving, one), s.uint(0))
    var after = s.smin(s.iadd(moving, one), last)
    var after2 = s.smin(s.iadd(moving, two), last)
    var here = s.iadd(s.imul(line, cell_columns), column)
    var before2_cell = s.iadd(s.imul(line, cell_columns), before2) if horizontal else s.iadd(s.imul(before2, cell_columns), column)
    var before_cell = s.iadd(s.imul(line, cell_columns), before) if horizontal else s.iadd(s.imul(before, cell_columns), column)
    var after_cell = s.iadd(s.imul(line, cell_columns), after) if horizontal else s.iadd(s.imul(after, cell_columns), column)
    var after2_cell = s.iadd(s.imul(line, cell_columns), after2) if horizontal else s.iadd(s.imul(after2, cell_columns), column)
    var inverse_strength = s.push_float(BLUR_PUSH_INVERSE_STRENGTH)
    var pack_bytes = s.ine(s.push_word(BLUR_PUSH_PACK_BYTES), s.uint(0))
    for which in range(3):
        var offset = s.uint(UInt32(which))
        var a0 = s.load_float(cells, s.iadd(source, s.iadd(s.imul(before2_cell, three), offset)))
        var a = s.load_float(cells, s.iadd(source, s.iadd(s.imul(before_cell, three), offset)))
        var b = s.load_float(cells, s.iadd(source, s.iadd(s.imul(here, three), offset)))
        var c = s.load_float(cells, s.iadd(source, s.iadd(s.imul(after_cell, three), offset)))
        var c2 = s.load_float(cells, s.iadd(source, s.iadd(s.imul(after2_cell, three), offset)))
        var inner = s.fma(s.float(0.25), s.fadd(a, c), s.fmul(s.float(0.375), b))
        var mixed = s.fma(s.float(0.0625), s.fadd(a0, c2), inner)
        if horizontal:
            var bits = s.bitcast_uint(mixed)
            s.store_word(cells, s.iadd(target, s.iadd(s.imul(here, three), offset)), bits)
        else:
            var out_f = s.select_f(pack_bytes, s.fmul(mixed, inverse_strength), mixed)
            s.store_word(cells, s.iadd(target, s.iadd(s.imul(here, three), offset)), s.bitcast_uint(out_f))
    return s.finish()


def kawase_down_shader() -> List[UInt32]:
    """5-tap Dual-Kawase pyramid downsample with subtle anamorphic horizontal spread."""
    var s = SpirvModule(GROUP_WIDTH, GROUP_HEIGHT, KAWASE_DOWN_WORDS)
    var cells = s.buffer(BIND_CELLS)
    var column = s.invocation_axis(0)
    var line = s.invocation_axis(1)
    var dst_w = s.push_word(KAWASE_DOWN_DST_W)
    var dst_h = s.push_word(KAWASE_DOWN_DST_H)
    s.return_if(s.uge(column, dst_w))
    s.return_if(s.uge(line, dst_h))
    var src_w = s.push_word(KAWASE_DOWN_SRC_W)
    var src_h = s.push_word(KAWASE_DOWN_SRC_H)
    var src_off = s.push_word(KAWASE_DOWN_SRC_OFF)
    var dst_off = s.push_word(KAWASE_DOWN_DST_OFF)
    var one = s.uint(1)
    var two = s.uint(2)
    var three = s.uint(3)
    var zero = s.uint(0)
    var max_sx = s.isub(src_w, one)
    var max_sy = s.isub(src_h, one)
    var cx = s.umin(s.iadd(s.shl(column, one), zero), max_sx)
    var cy = s.umin(s.iadd(s.shl(line, one), zero), max_sy)
    var lx = s.smax(s.isub(cx, two), zero)
    var rx = s.smin(s.iadd(cx, two), max_sx)
    var uy = s.smax(s.isub(cy, one), zero)
    var dy = s.smin(s.iadd(cy, one), max_sy)

    var c_mid = s.iadd(s.imul(cy, src_w), cx)
    var c_ul = s.iadd(s.imul(uy, src_w), lx)
    var c_ur = s.iadd(s.imul(uy, src_w), rx)
    var c_dl = s.iadd(s.imul(dy, src_w), lx)
    var c_dr = s.iadd(s.imul(dy, src_w), rx)
    var out_cell = s.iadd(s.imul(line, dst_w), column)
    for which in range(3):
        var ch = s.uint(UInt32(which))
        var v_mid = s.load_float(cells, s.iadd(src_off, s.iadd(s.imul(c_mid, three), ch)))
        var v_ul = s.load_float(cells, s.iadd(src_off, s.iadd(s.imul(c_ul, three), ch)))
        var v_ur = s.load_float(cells, s.iadd(src_off, s.iadd(s.imul(c_ur, three), ch)))
        var v_dl = s.load_float(cells, s.iadd(src_off, s.iadd(s.imul(c_dl, three), ch)))
        var v_dr = s.load_float(cells, s.iadd(src_off, s.iadd(s.imul(c_dr, three), ch)))
        var corners = s.fadd(s.fadd(v_ul, v_ur), s.fadd(v_dl, v_dr))
        var down = s.fma(corners, s.float(0.125), s.fmul(v_mid, s.float(0.5)))
        s.store_word(cells, s.iadd(dst_off, s.iadd(s.imul(out_cell, three), ch)), s.bitcast_uint(down))
    return s.finish()


def _sample_float_bilinear(
    mut s: SpirvModule,
    cells: UInt32,
    base_off: UInt32,
    width: UInt32,
    height: UInt32,
    u_coord: UInt32,
    v_coord: UInt32,
    ch: UInt32,
) -> UInt32:
    var one = s.uint(1)
    var three = s.uint(3)
    var max_x = s.isub(width, one)
    var max_y = s.isub(height, one)
    var uc = s.nmax(u_coord, s.float(0.0))
    var vc = s.nmax(v_coord, s.float(0.0))
    var x0 = s.umin(s.saturate_uint(uc), max_x)
    var x1 = s.umin(s.iadd(x0, one), max_x)
    var y0 = s.umin(s.saturate_uint(vc), max_y)
    var y1 = s.umin(s.iadd(y0, one), max_y)
    var tx = s.nclamp(s.fsub(uc, s.float_from_uint(x0)), s.float(0.0), s.float(1.0))
    var ty = s.nclamp(s.fsub(vc, s.float_from_uint(y0)), s.float(0.0), s.float(1.0))
    var r0 = s.imul(y0, width)
    var r1 = s.imul(y1, width)
    var s00 = s.load_float(cells, s.iadd(base_off, s.iadd(s.imul(s.iadd(r0, x0), three), ch)))
    var s10 = s.load_float(cells, s.iadd(base_off, s.iadd(s.imul(s.iadd(r0, x1), three), ch)))
    var s01 = s.load_float(cells, s.iadd(base_off, s.iadd(s.imul(s.iadd(r1, x0), three), ch)))
    var s11 = s.load_float(cells, s.iadd(base_off, s.iadd(s.imul(s.iadd(r1, x1), three), ch)))
    var top = s.fma(s.fsub(s10, s00), tx, s00)
    var bot = s.fma(s.fsub(s11, s01), tx, s01)
    return s.fma(s.fsub(bot, top), ty, top)


def kawase_up_shader() -> List[UInt32]:
    """Dual-Kawase progressive tent upsample + Linear Float32 HDR output in BLOOM_FINAL_OFFSET."""
    var s = SpirvModule(GROUP_WIDTH, GROUP_HEIGHT, KAWASE_UP_WORDS)
    var cells = s.buffer(BIND_CELLS)
    var column = s.invocation_axis(0)
    var line = s.invocation_axis(1)
    var high_w = s.push_word(KAWASE_UP_HIGH_W)
    var high_h = s.push_word(KAWASE_UP_HIGH_H)
    s.return_if(s.uge(column, high_w))
    s.return_if(s.uge(line, high_h))
    var low_w = s.push_word(KAWASE_UP_LOW_W)
    var low_h = s.push_word(KAWASE_UP_LOW_H)
    var low_off = s.push_word(KAWASE_UP_LOW_OFF)
    var high_off = s.push_word(KAWASE_UP_HIGH_OFF)
    var dst_off = s.push_word(KAWASE_UP_DST_OFF)
    var inv_strength = s.push_float(KAWASE_UP_INV_STRENGTH)
    var is_final = s.fgt(inv_strength, s.float(0.0001))

    var half = s.float(0.5)
    var sx = s.fdiv(s.float_from_uint(low_w), s.nmax(s.float_from_uint(high_w), s.float(1.0)))
    var sy = s.fdiv(s.float_from_uint(low_h), s.nmax(s.float_from_uint(high_h), s.float(1.0)))
    var u = s.fsub(s.fmul(sx, s.fadd(s.float_from_uint(column), half)), half)
    var v = s.fsub(s.fmul(sy, s.fadd(s.float_from_uint(line), half)), half)
    var here = s.iadd(s.imul(line, high_w), column)
    var three = s.uint(3)

    for which in range(3):
        var ch = s.uint(UInt32(which))
        var c0 = _sample_float_bilinear(s, cells, low_off, low_w, low_h, u, v, ch)
        var cl = _sample_float_bilinear(s, cells, low_off, low_w, low_h, s.fsub(u, s.float(1.15)), v, ch)
        var cr = _sample_float_bilinear(s, cells, low_off, low_w, low_h, s.fadd(u, s.float(1.15)), v, ch)
        var cu = _sample_float_bilinear(s, cells, low_off, low_w, low_h, u, s.fsub(v, s.float(0.95)), ch)
        var cd = _sample_float_bilinear(s, cells, low_off, low_w, low_h, u, s.fadd(v, s.float(0.95)), ch)
        var ring = s.fadd(s.fadd(cl, cr), s.fadd(cu, cd))
        var tent = s.fma(ring, s.float(0.125), s.fmul(c0, s.float(0.5)))
        var base_val = s.load_float(cells, s.iadd(high_off, s.iadd(s.imul(here, three), ch)))
        s.begin_if(is_final)
        var fused = s.fmul(s.fma(tent, s.float(0.66), s.fmul(base_val, s.float(0.68))), inv_strength)
        s.store_word(cells, s.iadd(dst_off, s.iadd(s.imul(here, three), ch)), s.bitcast_uint(fused))
        s.begin_else()
        var accum = s.fma(tent, s.float(0.74), base_val)
        s.store_word(cells, s.iadd(dst_off, s.iadd(s.imul(here, three), ch)), s.bitcast_uint(accum))
        s.end_if()
    return s.finish()


def _ssao_symmetric_pair(
    mut s: SpirvModule,
    target: UInt32,
    depth_plane: UInt32,
    row_w: UInt32,
    xp: UInt32,
    yp: UInt32,
    xn: UInt32,
    yn: UInt32,
    z_c: UInt32,
    gain: Float32,
) -> UInt32:
    """Affine-1/z symmetric concavity pair: 0.0 on any flat plane at any tilt, >0 in true crevices."""
    var zp = s.load_float(target, s.iadd(depth_plane, s.iadd(s.imul(yp, row_w), xp)))
    var zn = s.load_float(target, s.iadd(depth_plane, s.iadd(s.imul(yn, row_w), xn)))
    var dp = s.fsub(zp, z_c)
    var dn = s.fsub(zn, z_c)
    var max_d = s.nmax(s.fabs(dp), s.fabs(dn))
    var in_range = s.flt(max_d, s.fmul(z_c, s.float(0.14)))
    var avg_n = s.fmul(s.fadd(zp, zn), s.float(0.5))
    var concav = s.fdiv(s.fsub(avg_n, z_c), z_c)
    var occ = s.nclamp(s.fmul(s.fsub(concav, s.float(0.0014)), s.float(gain)), s.float(0.0), s.float(0.12))
    return s.select_f(in_range, occ, s.float(0.0))


def _lookup_tone_lut_interp(mut s: SpirvModule, res: Resources, mapped_hdr: UInt32, which: Int) -> UInt32:
    """Sub-integer linearly interpolated tone_lut lookup to eliminate 8-bit sky/gradient banding."""
    var clamped = s.nclamp(mapped_hdr, s.float(0.0), s.float(255.0))
    var idx0 = s.umin(s.saturate_uint(clamped), s.uint(255))
    var idx1 = s.umin(s.iadd(idx0, s.uint(1)), s.uint(255))
    var frac_t = s.fsub(clamped, s.float_from_uint(idx0))
    var off_ch = s.uint(UInt32(which * 256))
    var tb0 = s.iadd(idx0, off_ch)
    var tb1 = s.iadd(idx1, off_ch)
    var w0 = s.load_word(res.constants, s.shr(tb0, s.uint(2)))
    var w1 = s.load_word(res.constants, s.shr(tb1, s.uint(2)))
    var b0 = s.bit_and(s.shr(w0, s.shl(s.bit_and(tb0, s.uint(3)), s.uint(3))), s.uint(255))
    var b1 = s.bit_and(s.shr(w1, s.shl(s.bit_and(tb1, s.uint(3)), s.uint(3))), s.uint(255))
    var f0 = s.float_from_uint(b0)
    var f1 = s.float_from_uint(b1)
    return s.fma(s.fsub(f1, f0), frac_t, f0)


def _tonemap_agx_sdr(
    mut s: SpirvModule,
    res: Resources,
    hr_post: UInt32,
    hg_post: UInt32,
    hb_post: UInt32,
    ign: UInt32,
) -> UInt32:
    """Troy Sobotka cross-channel AgX chromaticity inset + hue-preserving HDR shoulder + Tone LUT + Saturation."""
    var zero_u = s.uint(0)
    var post_hdrs = List[UInt32]()
    post_hdrs.append(hr_post)
    post_hdrs.append(hg_post)
    post_hdrs.append(hb_post)

    var hdr_luma = s.fma(
        post_hdrs[2],
        s.float(0.16),
        s.fma(post_hdrs[1], s.float(0.58), s.fmul(post_hdrs[0], s.float(0.26))),
    )
    var peak0 = s.nmax(s.nmax(post_hdrs[0], post_hdrs[1]), s.nmax(post_hdrs[2], s.float(0.0001)))
    var hi_t = s.nclamp(s.fmul(s.fsub(peak0, s.float(172.0)), s.float(0.0048)), s.float(0.0), s.float(1.0))
    var agx_inset = s.fma(hi_t, s.float(0.30), s.float(0.045))
    var inset_hdrs = List[UInt32]()
    for which in range(3):
        inset_hdrs.append(s.fma(s.fsub(hdr_luma, post_hdrs[which]), agx_inset, post_hdrs[which]))

    var peak_in = s.nmax(s.nmax(inset_hdrs[0], inset_hdrs[1]), s.nmax(inset_hdrs[2], s.float(0.0001)))
    var excess = s.nmax(s.fsub(peak_in, s.float(175.0)), s.float(0.0))
    var mapped_peak = s.fadd(
        s.nmin(peak_in, s.float(175.0)),
        s.fdiv(s.fmul(excess, s.float(80.0)), s.fadd(excess, s.float(95.0))),
    )
    var hdr_scale = s.fdiv(mapped_peak, peak_in)
    var dither_ign = s.fmul(s.fsub(ign, s.float(0.5)), s.float(0.55))

    var shades_u = List[UInt32]()
    for which in range(3):
        var mapped_hdr = s.fmul(inset_hdrs[which], hdr_scale)
        var lut_f = _lookup_tone_lut_interp(s, res, mapped_hdr, which)
        shades_u.append(float_to_byte(s, s.fadd(lut_f, dither_ign)))

    var saturation = constant_word(s, res, CONSTANT_SATURATION)
    var red_v = s.uvar(shades_u[0])
    var green_v = s.uvar(shades_u[1])
    var blue_v = s.uvar(shades_u[2])
    s.begin_if(s.ine(saturation, zero_u))
    var weighted = s.iadd(s.iadd(s.imul(shades_u[0], s.uint(66)), s.imul(shades_u[1], s.uint(150))), s.imul(shades_u[2], s.uint(40)))
    var luminance = s.shr(weighted, s.uint(8))
    var gain = s.iadd(saturation, s.uint(64))
    for which in range(3):
        var spread = s.sar(s.imul(s.isub(shades_u[which], luminance), gain), s.uint(6))
        s.store(red_v if which == 0 else (green_v if which == 1 else blue_v), clamp_byte(s, s.iadd(luminance, spread)))
    s.end_if()

    return s.bit_or(
        s.uint(0xFF000000),
        s.bit_or(s.shl(s.load_u(red_v), s.uint(16)), s.bit_or(s.shl(s.load_u(green_v), s.uint(8)), s.load_u(blue_v))),
    )


def _sample_sky_lut_rgb(
    mut s: SpirvModule,
    res: Resources,
    cells: UInt32,
    ray_dir: Vec3Ids,
    to_key: Vec3Ids,
    key_color: Vec3Ids,
) -> Vec3Ids:
    """Sample the 128x64 Hillaire Sky-View LUT (4 floats/texel) + limb-guarded solar disc."""
    var rh2 = s.fma(ray_dir.z, ray_dir.z, s.fmul(ray_dir.x, ray_dir.x))
    var sh2 = s.fma(to_key.z, to_key.z, s.fmul(to_key.x, to_key.x))
    var h_dot = s.fma(ray_dir.z, to_key.z, s.fmul(ray_dir.x, to_key.x))
    var cos_az = s.nclamp(
        s.fmul(h_dot, s.inverse_sqrt(s.fadd(s.fmul(rh2, sh2), s.float(1e-8)))),
        s.float(-1.0),
        s.float(1.0),
    )
    var u_lut = s.fmul(s.fma(cos_az, s.float(0.5), s.float(0.5)), s.float(Float32(SKY_LUT_W - 1)))
    var abs_dy = s.fabs(ray_dir.y)
    var sgn_dy = s.select_f(s.fge(ray_dir.y, s.float(0.0)), s.float(1.0), s.float(-1.0))
    var v_lut = s.fmul(
        s.fma(s.fmul(sgn_dy, s.fsqrt(abs_dy)), s.float(0.5), s.float(0.5)),
        s.float(Float32(SKY_LUT_H - 1)),
    )

    var one = s.uint(1)
    var max_x = s.uint(UInt32(SKY_LUT_W - 1))
    var max_y = s.uint(UInt32(SKY_LUT_H - 1))
    var x0 = s.umin(s.saturate_uint(s.nmax(u_lut, s.float(0.0))), max_x)
    var x1 = s.umin(s.iadd(x0, one), max_x)
    var y0 = s.umin(s.saturate_uint(s.nmax(v_lut, s.float(0.0))), max_y)
    var y1 = s.umin(s.iadd(y0, one), max_y)
    var tx = s.nclamp(s.fsub(u_lut, s.float_from_uint(x0)), s.float(0.0), s.float(1.0))
    var ty = s.nclamp(s.fsub(v_lut, s.float_from_uint(y0)), s.float(0.0), s.float(1.0))
    var w_lut = s.uint(UInt32(SKY_LUT_W))
    var lut_base = s.uint(UInt32(SKY_LUT_OFFSET // 4))
    var i00 = s.iadd(lut_base, s.shl(s.iadd(s.imul(y0, w_lut), x0), s.uint(2)))
    var i10 = s.iadd(lut_base, s.shl(s.iadd(s.imul(y0, w_lut), x1), s.uint(2)))
    var i01 = s.iadd(lut_base, s.shl(s.iadd(s.imul(y1, w_lut), x0), s.uint(2)))
    var i11 = s.iadd(lut_base, s.shl(s.iadd(s.imul(y1, w_lut), x1), s.uint(2)))

    var out_chs = List[UInt32]()
    for ch in range(3):
        var c_u = s.uint(UInt32(ch))
        var s00 = s.load_float(cells, s.iadd(i00, c_u))
        var s10 = s.load_float(cells, s.iadd(i10, c_u))
        var s01 = s.load_float(cells, s.iadd(i01, c_u))
        var s11 = s.load_float(cells, s.iadd(i11, c_u))
        var top = s.fma(s.fsub(s10, s00), tx, s00)
        var bot = s.fma(s.fsub(s11, s01), tx, s01)
        out_chs.append(s.fma(s.fsub(bot, top), ty, top))

    # Limb-guarded solar disc & corona (transmittance stored in LUT alpha channel 3)
    var tr_sun = s.load_float(cells, s.iadd(i00, s.uint(3)))
    var cos_sun = dot3(s, ray_dir, to_key)
    var disc = s.nclamp(s.fmul(s.fsub(cos_sun, s.float(0.99935)), s.float(2200.0)), s.float(0.0), s.float(1.0))
    var sun_gain = s.fmul(s.fmul(disc, disc), s.fmul(tr_sun, s.float(320.0)))
    return Vec3Ids(
        s.fma(key_color.x, sun_gain, out_chs[0]),
        s.fma(key_color.y, sun_gain, out_chs[1]),
        s.fma(key_color.z, s.fmul(sun_gain, s.float(0.88)), out_chs[2]),
    )


def _sample_froxel_trilinear(
    mut s: SpirvModule,
    cells: UInt32,
    fu: UInt32,
    fv: UInt32,
    fz: UInt32,
    out_r: UInt32,
    out_g: UInt32,
    out_b: UInt32,
    out_tr: UInt32,
):
    """Trilinear lookup of cumulative (scat_r, scat_g, scat_b, trans) from FROXEL_INTEGRATED_OFFSET."""
    var one = s.uint(1)
    var max_x = s.uint(UInt32(FROXEL_X - 1))
    var max_y = s.uint(UInt32(FROXEL_Y - 1))
    var max_z = s.uint(UInt32(FROXEL_Z - 1))
    var x0 = s.umin(s.saturate_uint(s.nmax(fu, s.float(0.0))), max_x)
    var x1 = s.umin(s.iadd(x0, one), max_x)
    var y0 = s.umin(s.saturate_uint(s.nmax(fv, s.float(0.0))), max_y)
    var y1 = s.umin(s.iadd(y0, one), max_y)
    var z0 = s.umin(s.saturate_uint(s.nmax(fz, s.float(0.0))), max_z)
    var z1 = s.umin(s.iadd(z0, one), max_z)
    var tx = s.nclamp(s.fsub(fu, s.float_from_uint(x0)), s.float(0.0), s.float(1.0))
    var ty = s.nclamp(s.fsub(fv, s.float_from_uint(y0)), s.float(0.0), s.float(1.0))
    var tz = s.nclamp(s.fsub(fz, s.float_from_uint(z0)), s.float(0.0), s.float(1.0))

    var fx_u = s.uint(UInt32(FROXEL_X))
    var fy_u = s.uint(UInt32(FROXEL_Y))
    var base = s.uint(UInt32(FROXEL_INTEGRATED_OFFSET // 4))
    var z0_row = s.imul(z0, fy_u)
    var z1_row = s.imul(z1, fy_u)
    var i000 = s.iadd(base, s.shl(s.iadd(s.imul(s.iadd(z0_row, y0), fx_u), x0), s.uint(2)))
    var i100 = s.iadd(base, s.shl(s.iadd(s.imul(s.iadd(z0_row, y0), fx_u), x1), s.uint(2)))
    var i010 = s.iadd(base, s.shl(s.iadd(s.imul(s.iadd(z0_row, y1), fx_u), x0), s.uint(2)))
    var i110 = s.iadd(base, s.shl(s.iadd(s.imul(s.iadd(z0_row, y1), fx_u), x1), s.uint(2)))
    var i001 = s.iadd(base, s.shl(s.iadd(s.imul(s.iadd(z1_row, y0), fx_u), x0), s.uint(2)))
    var i101 = s.iadd(base, s.shl(s.iadd(s.imul(s.iadd(z1_row, y0), fx_u), x1), s.uint(2)))
    var i011 = s.iadd(base, s.shl(s.iadd(s.imul(s.iadd(z1_row, y1), fx_u), x0), s.uint(2)))
    var i111 = s.iadd(base, s.shl(s.iadd(s.imul(s.iadd(z1_row, y1), fx_u), x1), s.uint(2)))

    for ch in range(4):
        var cu = s.uint(UInt32(ch))
        var c000 = s.load_float(cells, s.iadd(i000, cu))
        var c100 = s.load_float(cells, s.iadd(i100, cu))
        var c010 = s.load_float(cells, s.iadd(i010, cu))
        var c110 = s.load_float(cells, s.iadd(i110, cu))
        var c001 = s.load_float(cells, s.iadd(i001, cu))
        var c101 = s.load_float(cells, s.iadd(i101, cu))
        var c011 = s.load_float(cells, s.iadd(i011, cu))
        var c111 = s.load_float(cells, s.iadd(i111, cu))
        var x00 = s.fma(s.fsub(c100, c000), tx, c000)
        var x10 = s.fma(s.fsub(c110, c010), tx, c010)
        var x01 = s.fma(s.fsub(c101, c001), tx, c001)
        var x11 = s.fma(s.fsub(c111, c011), tx, c011)
        var y0_v = s.fma(s.fsub(x10, x00), ty, x00)
        var y1_v = s.fma(s.fsub(x11, x01), ty, x01)
        var val = s.fma(s.fsub(y1_v, y0_v), tz, y0_v)
        s.store(out_r if ch == 0 else (out_g if ch == 1 else (out_b if ch == 2 else out_tr)), val)


def _integrate_volumetric_clouds(
    mut s: SpirvModule,
    res: Resources,
    camera: Vec3Ids,
    ray_dir: Vec3Ids,
    max_dist: UInt32,
    sky_key: Vec3Ids,
    sky_kc: Vec3Ids,
    ign: UInt32,
    out_r: UInt32,
    out_g: UInt32,
    out_b: UInt32,
    out_tr: UInt32,
):
    """6-step IGN-jittered 3D Gyroid-Worley cumulus ray-march with Nubis Beer-Powder self-shadowing
    and dual-lobe Henyey-Greenstein silver-lining forward scattering."""
    s.store(out_r, s.float(0.0))
    s.store(out_g, s.float(0.0))
    s.store(out_b, s.float(0.0))
    s.store(out_tr, s.float(1.0))

    var base_y = constant_float(s, res, CONSTANT_V9_CLOUDS)
    var top_y = constant_float(s, res, CONSTANT_V9_CLOUDS + 4)
    var dens_mul = constant_float(s, res, CONSTANT_V9_CLOUDS + 12)
    s.begin_if(s.fgt(dens_mul, s.float(0.0001)))
    var ry = ray_dir.y
    var abs_ry = s.fabs(ry)
    var safe_ry = s.select_f(
        s.fge(ry, s.float(0.0)),
        s.nmax(ry, s.float(0.015)),
        s.fsub(s.float(0.0), s.nmax(abs_ry, s.float(0.015))),
    )
    var t0_raw = s.fdiv(s.fsub(base_y, camera.y), safe_ry)
    var t1_raw = s.fdiv(s.fsub(top_y, camera.y), safe_ry)
    var in_slab = s.logical_and(s.fge(camera.y, base_y), s.fge(top_y, camera.y))
    var t_min = s.select_f(in_slab, s.float(2.0), s.nmax(s.nmin(t0_raw, t1_raw), s.float(2.0)))
    var t_max_raw = s.select_f(
        in_slab,
        s.select_f(s.fge(ry, s.float(0.0)), t1_raw, t0_raw),
        s.nmax(t0_raw, t1_raw),
    )
    var t_end = s.nmin(s.nmin(t_max_raw, max_dist), s.float(8500.0))
    s.begin_if(s.logical_and(s.fgt(t_end, t_min), s.fgt(s.fadd(abs_ry, s.select_f(in_slab, s.float(1.0), s.float(0.0))), s.float(0.025))))
    var span = s.nmin(s.fsub(t_end, t_min), s.float(2400.0))
    var dt = s.fmul(span, s.float(0.16666667))
    var cos_th = dot3(s, ray_dir, sky_key)
    var hg_fwd = s.fdiv(
        s.float(0.6156),
        s.nmax(
            s.fmul(
                s.fsub(s.float(1.3844), s.fmul(s.float(1.24), cos_th)),
                s.fsqrt(s.nmax(s.fsub(s.float(1.3844), s.fmul(s.float(1.24), cos_th)), s.float(0.04))),
            ),
            s.float(0.04),
        ),
    )
    var hg_back = s.fma(cos_th, s.float(-0.18), s.float(0.42))
    var phase = s.nmin(s.fma(hg_fwd, s.float(0.68), s.fmul(hg_back, s.float(0.45))), s.float(2.35))
    var sun_r = s.fmul(sky_kc.x, s.float(228.0))
    var sun_g = s.fmul(sky_kc.y, s.float(218.0))
    var sun_b = s.fmul(sky_kc.z, s.float(202.0))
    var inv_slab_h = s.fdiv(s.float(1.0), s.nmax(s.fsub(top_y, base_y), s.float(10.0)))

    var step_u = s.uvar(s.uint(0))
    s.begin_loop()
    var si = s.load_u(step_u)
    s.break_if(s.logical_or(s.uge(si, s.uint(6)), s.flt(s.load_f(out_tr), s.float(0.03))))
    s.store(step_u, s.iadd(si, s.uint(1)))
    var t_s = s.fma(s.fadd(s.float_from_uint(si), ign), dt, t_min)
    var px = s.fma(ray_dir.x, t_s, camera.x)
    var py = s.fma(ray_dir.y, t_s, camera.y)
    var pz = s.fma(ray_dir.z, t_s, camera.z)
    var d_loc = _cloud_density_3d(s, res, px, py, pz)
    s.begin_if(s.fgt(d_loc, s.float(0.002)))
    var h_amb = s.nclamp(s.fmul(s.fsub(py, base_y), inv_slab_h), s.float(0.0), s.float(1.0))
    var opt_sun = s.fmul(d_loc, s.fsub(s.float(2.6), s.fmul(h_amb, s.float(1.5))))
    var beer_sun = s.exp2(s.fsub(s.float(0.0), opt_sun))
    var powder = s.fsub(s.float(1.0), s.fmul(s.exp2(s.fsub(s.float(0.0), s.fmul(d_loc, s.float(3.4)))), s.float(0.42)))
    var lit_e = s.fmul(s.fmul(beer_sun, powder), phase)
    var amb_w = s.fma(h_amb, s.float(0.45), s.float(0.55))
    var c_r = s.fma(sun_r, lit_e, s.fmul(s.float(138.0), amb_w))
    var c_g = s.fma(sun_g, lit_e, s.fmul(s.float(144.0), amb_w))
    var c_b = s.fma(sun_b, lit_e, s.fmul(s.float(156.0), amb_w))
    var sig_dt = s.nmin(s.fmul(s.fmul(d_loc, dens_mul), dt), s.float(0.88))
    var tr_cur = s.load_f(out_tr)
    var w_int = s.fmul(tr_cur, sig_dt)
    s.store(out_r, s.fma(c_r, w_int, s.load_f(out_r)))
    s.store(out_g, s.fma(c_g, w_int, s.load_f(out_g)))
    s.store(out_b, s.fma(c_b, w_int, s.load_f(out_b)))
    s.store(out_tr, s.fmul(tr_cur, s.fsub(s.float(1.0), sig_dt)))
    s.end_if()
    s.end_loop()
    s.end_if()
    s.end_if()


def compose_shader(capture: Bool) -> List[UInt32]:
    """Linear Float32 HDR post-process: Refraction -> 12-step SSR -> 10-step Shadowed Volumetrics ->
    8-tap SSAO -> Linear HDR Dual-Kawase Bloom -> Troy Sobotka AgX + Tone LUT + Saturation -> PLANE_COLOR."""
    var s = SpirvModule(GROUP_WIDTH, GROUP_HEIGHT, COMPOSE_PUSH_WORDS)
    var res = Resources(s)
    var target = s.buffer(BIND_TARGET)
    var cells = s.buffer(BIND_CELLS)
    var column = s.invocation_axis(0)
    var line = s.invocation_axis(1)
    var width_w = s.push_word(COMPOSE_PUSH_WIDTH)
    var height_w = s.push_word(COMPOSE_PUSH_HEIGHT)
    var row_w = s.push_word(COMPOSE_PUSH_ROW)
    s.return_if(s.uge(column, width_w))
    s.return_if(s.uge(line, height_w))
    var word = s.iadd(s.imul(line, row_w), column)
    var plane_stride = s.push_word(COMPOSE_PUSH_PLANE_STRIDE)
    var clear_word = s.push_word(COMPOSE_PUSH_CLEAR)
    var one_u = s.uint(1)
    var zero_u = s.uint(0)
    var max_px_x = s.isub(width_w, one_u)
    var max_px_y = s.isub(height_w, one_u)
    var width_f = s.float_from_uint(width_w)
    var height_f = s.float_from_uint(height_w)
    var x_f = s.float_from_uint(column)
    var y_f = s.float_from_uint(line)
    var ign = ign_noise(s, x_f, y_f)

    # 1. Depth-guarded screen-space refraction + V9 Cauchy RGB Chromatic Dispersion & Exhaust Heat-Haze
    var effects = constant_word(s, res, CONSTANT_EFFECTS)
    var scene_word = s.uvar(word)
    var refract_px_r = s.uvar(word)
    var refract_px_b = s.uvar(word)
    var depth_plane = s.imul(plane_stride, s.uint(3))
    var ref_word = s.load_word(target, s.iadd(word, s.imul(plane_stride, s.uint(4))))
    s.begin_if(s.ine(s.shr(ref_word, s.uint(16)), zero_u))
    var dx_s = s.isub(s.bit_and(ref_word, s.uint(255)), s.uint(128))
    var dy_s = s.isub(s.bit_and(s.shr(ref_word, s.uint(8)), s.uint(255)), s.uint(128))
    var dx_f = s.fvar(s.float_from_int(dx_s))
    var dy_f = s.fvar(s.float_from_int(dy_s))
    var chrom_on = s.ine(s.bit_and(effects, s.uint(EFFECT_CHROMATIC_REFRACT)), zero_u)
    s.begin_if(chrom_on)
    var haze_amp = constant_float(s, res, CONSTANT_V9_CLOUDS + 40)
    var v9_t = constant_float(s, res, CONSTANT_V9_CLOUDS + 44)
    var rip_x = _parabolic_sin(s, s.fma(y_f, s.float(0.018), s.fmul(v9_t, s.float(2.4))))
    var rip_y = _parabolic_sin(s, s.fma(x_f, s.float(0.015), s.fmul(v9_t, s.float(1.9))))
    s.store(dx_f, s.fma(rip_x, haze_amp, s.load_f(dx_f)))
    s.store(dy_f, s.fma(rip_y, haze_amp, s.load_f(dy_f)))
    s.end_if()
    var gx_i = s.int_from_float(s.load_f(dx_f))
    var gy_i = s.int_from_float(s.load_f(dy_f))
    var rx = s.smin(s.smax(s.iadd(column, gx_i), zero_u), max_px_x)
    var ry = s.smin(s.smax(s.iadd(line, gy_i), zero_u), max_px_y)
    var cand_word = s.iadd(s.imul(ry, row_w), rx)
    var z_here = s.load_float(target, s.iadd(word, depth_plane))
    var z_guard = s.fmul(z_here, s.float(1.15))
    var z_cand = s.load_float(target, s.iadd(cand_word, depth_plane))
    s.begin_if(s.fge(z_guard, z_cand))
    s.store(scene_word, cand_word)
    s.store(refract_px_r, cand_word)
    s.store(refract_px_b, cand_word)
    s.end_if()
    s.begin_if(chrom_on)
    var disp_str = constant_float(s, res, CONSTANT_V9_CLOUDS + 36)
    var r_mul = s.fsub(s.float(1.0), disp_str)
    var b_mul = s.fadd(s.float(1.0), disp_str)
    var r_dx = s.int_from_float(s.fmul(s.load_f(dx_f), r_mul))
    var r_dy = s.int_from_float(s.fmul(s.load_f(dy_f), r_mul))
    var b_dx = s.int_from_float(s.fmul(s.load_f(dx_f), b_mul))
    var b_dy = s.int_from_float(s.fmul(s.load_f(dy_f), b_mul))
    var cand_r = s.iadd(
        s.imul(s.smin(s.smax(s.iadd(line, r_dy), zero_u), max_px_y), row_w),
        s.smin(s.smax(s.iadd(column, r_dx), zero_u), max_px_x),
    )
    var cand_b = s.iadd(
        s.imul(s.smin(s.smax(s.iadd(line, b_dy), zero_u), max_px_y), row_w),
        s.smin(s.smax(s.iadd(column, b_dx), zero_u), max_px_x),
    )
    s.begin_if(s.fge(z_guard, s.load_float(target, s.iadd(cand_r, depth_plane))))
    s.store(refract_px_r, cand_r)
    s.end_if()
    s.begin_if(s.fge(z_guard, s.load_float(target, s.iadd(cand_b, depth_plane))))
    s.store(refract_px_b, cand_b)
    s.end_if()
    s.end_if()
    s.end_if()

    var base_px = s.load_u(scene_word)
    var plane_r = s.imul(plane_stride, s.uint(5))
    var plane_g = s.imul(plane_stride, s.uint(6))
    var plane_b = s.imul(plane_stride, s.uint(7))
    var gbuf_plane = s.imul(plane_stride, s.uint(8))
    var gbuf = s.load_word(target, s.iadd(gbuf_plane, base_px))
    var touched = s.shr(gbuf, s.uint(31))

    # Sample Linear Float32 HDR Dual-Kawase bloom
    var bl_r = s.fvar(s.float(0.0))
    var bl_g = s.fvar(s.float(0.0))
    var bl_b = s.fvar(s.float(0.0))
    s.begin_if(s.ine(s.push_word(COMPOSE_PUSH_BLOOM), zero_u))
    var cell_columns = s.push_word(COMPOSE_PUSH_CELLS_X)
    var cell_rows = s.push_word(COMPOSE_PUSH_CELLS_Y)
    var half = s.float(0.5)
    var bu = s.fsub(s.fmul(s.push_float(COMPOSE_PUSH_SCALE_X), s.fadd(x_f, half)), half)
    var bv = s.fsub(s.fmul(s.push_float(COMPOSE_PUSH_SCALE_Y), s.fadd(y_f, half)), half)
    var base_off = s.uint(UInt32(BLOOM_FINAL_OFFSET // 4))
    s.store(bl_r, _sample_float_bilinear(s, cells, base_off, cell_columns, cell_rows, bu, bv, s.uint(0)))
    s.store(bl_g, _sample_float_bilinear(s, cells, base_off, cell_columns, cell_rows, bu, bv, s.uint(1)))
    s.store(bl_b, _sample_float_bilinear(s, cells, base_off, cell_columns, cell_rows, bu, bv, s.uint(2)))
    s.end_if()

    var out_pixel = s.uvar(s.bit_or(clear_word, s.uint(0xFF000000)))
    var vol_on = s.ine(s.bit_and(effects, s.uint(EFFECT_VOLUMETRIC)), zero_u)
    var phys_sky_on = s.ine(s.bit_and(effects, s.uint(EFFECT_PHYSICAL_SKY)), zero_u)
    var froxel_on = s.ine(s.bit_and(effects, s.uint(EFFECT_FROXEL_VOL)), zero_u)

    var camera = constant_vector(s, res, CONSTANT_CAMERA)
    var right = constant_vector(s, res, CONSTANT_CAMERA + 12)
    var up = constant_vector(s, res, CONSTANT_CAMERA + 24)
    var forward = constant_vector(s, res, CONSTANT_CAMERA + 36)
    var half_width = constant_float(s, res, CONSTANT_CAMERA + 48)
    var half_height = constant_float(s, res, CONSTANT_CAMERA + 52)
    var inverse_focal = constant_float(s, res, CONSTANT_CAMERA + 56)
    var focal = s.fdiv(s.float(1.0), s.nmax(inverse_focal, s.float(0.00001)))
    var vx_cam = s.fmul(s.fsub(s.fadd(x_f, s.float(0.5)), half_width), inverse_focal)
    var vy_cam = s.fmul(s.fsub(half_height, s.fadd(y_f, s.float(0.5))), inverse_focal)
    var ray = Vec3Ids(
        s.fma(up.x, vy_cam, s.fma(right.x, vx_cam, forward.x)),
        s.fma(up.y, vy_cam, s.fma(right.y, vx_cam, forward.y)),
        s.fma(up.z, vy_cam, s.fma(right.z, vx_cam, forward.z)),
    )

    var auto_exp = s.fvar(s.float(1.0))
    s.begin_if(s.ine(s.bit_and(effects, s.uint(EFFECT_AUTO_EXPOSURE)), zero_u))
    var hist_base = s.uint(UInt32(BLOOM_HIST_OFFSET // 4))
    var has_ae = s.ieq(s.load_word(cells, s.iadd(hist_base, s.uint(65))), s.uint(0xA7E00001))
    s.begin_if(has_ae)
    s.store(auto_exp, s.load_float(cells, s.iadd(hist_base, s.uint(64))))
    s.end_if()
    s.end_if()

    var vol_clouds_on = s.ine(s.bit_and(effects, s.uint(EFFECT_VOL_CLOUDS)), zero_u)
    var cl_out_r = s.fvar(s.float(0.0))
    var cl_out_g = s.fvar(s.float(0.0))
    var cl_out_b = s.fvar(s.float(0.0))
    var cl_out_tr = s.fvar(s.float(1.0))
    s.begin_if(vol_clouds_on)
    var ray_l_u = s.fsqrt(s.fadd(dot3(s, ray, ray), s.float(0.000001)))
    var inv_rl_u = s.fdiv(s.float(1.0), ray_l_u)
    var dir_cl_u = Vec3Ids(s.fmul(ray.x, inv_rl_u), s.fmul(ray.y, inv_rl_u), s.fmul(ray.z, inv_rl_u))
    var z_cl_u = s.load_float(target, s.iadd(depth_plane, base_px))
    var surf_d_u = s.fmul(s.fdiv(s.float(1.0), s.nmax(z_cl_u, s.float(0.00001))), ray_l_u)
    var max_d_u = s.select_f(s.ieq(touched, zero_u), s.float(9000.0), surf_d_u)
    var cl_key_u = constant_vector(s, res, CONSTANT_KEY)
    var cl_kc_u = constant_vector(s, res, CONSTANT_KEY + 12)
    _integrate_volumetric_clouds(s, res, camera, dir_cl_u, max_d_u, cl_key_u, cl_kc_u, ign, cl_out_r, cl_out_g, cl_out_b, cl_out_tr)
    s.end_if()

    var wx_precip_on = s.ine(s.bit_and(effects, s.uint(EFFECT_WEATHER_PRECIP)), zero_u)
    var wx_sky_e = s.fvar(s.float(0.0))
    s.begin_if(wx_precip_on)
    var wx_int0 = constant_float(s, res, CONSTANT_V11_POST + 12)
    s.begin_if(s.fgt(wx_int0, s.float(0.01)))
    var wx_kind0 = constant_word(s, res, CONSTANT_V11_POST + 8)
    var wx_time0 = constant_float(s, res, CONSTANT_V7_POST)
    var wx_wx0 = constant_float(s, res, CONSTANT_V11_POST + 20)
    var uv_x0 = s.fdiv(x_f, s.nmax(width_f, s.float(1.0)))
    var uv_y0 = s.fdiv(y_f, s.nmax(height_f, s.float(1.0)))
    var slant0 = s.nclamp(s.fmul(wx_wx0, s.float(0.012)), s.float(-0.35), s.float(0.35))
    var is_snow0 = s.ieq(wx_kind0, s.uint(2))
    var fall_spd0 = s.select_f(is_snow0, s.float(0.85), s.float(2.90))
    var freq_x0 = s.select_f(is_snow0, s.float(38.0), s.float(86.0))
    var freq_y0 = s.select_f(is_snow0, s.float(22.0), s.float(9.5))
    var u_wx0 = s.fmul(s.fma(uv_y0, slant0, uv_x0), freq_x0)
    var cell_wx0 = s.ffloor(u_wx0)
    var fu_wx0 = s.fsub(u_wx0, cell_wx0)
    var h_in0 = s.fmul(s.fadd(cell_wx0, s.float(17.31)), s.float(0.1031))
    var cell_h0 = s.fsub(h_in0, s.ffloor(h_in0))
    var v_wx0 = s.fma(s.fsub(uv_y0, s.fmul(wx_time0, fall_spd0)), freq_y0, s.fmul(cell_h0, s.float(13.7)))
    var fv_wx0 = s.fsub(v_wx0, s.ffloor(v_wx0))
    var dx_wx0 = s.fabs(s.fsub(fu_wx0, s.fma(cell_h0, s.float(0.5), s.float(0.25))))
    var streak_w0 = s.select_f(is_snow0, s.float(0.16), s.float(0.075))
    var mx_wx0 = s.nclamp(s.fmul(s.fsub(streak_w0, dx_wx0), s.float(16.0)), s.float(0.0), s.float(1.0))
    var my_wx0 = s.nclamp(s.fmul(s.fsub(s.float(0.42), fv_wx0), s.float(3.2)), s.float(0.0), s.float(1.0))
    s.store(wx_sky_e, s.fmul(s.fmul(mx_wx0, my_wx0), s.fmul(wx_int0, s.float(56.0))))
    s.end_if()
    s.end_if()

    s.begin_if(s.ieq(touched, zero_u))
    s.begin_if(s.logical_or(s.logical_or(phys_sky_on, vol_clouds_on), s.fgt(s.load_f(wx_sky_e), s.float(0.01))))
    var inv_sk_l = s.inverse_sqrt(s.fadd(dot3(s, ray, ray), s.float(0.000001)))
    var sky_dir = Vec3Ids(s.fmul(ray.x, inv_sk_l), s.fmul(ray.y, inv_sk_l), s.fmul(ray.z, inv_sk_l))
    var sky_key = constant_vector(s, res, CONSTANT_KEY)
    var sky_kc = constant_vector(s, res, CONSTANT_KEY + 12)
    var sky_r_v = s.fvar(s.float_from_uint(channel(s, clear_word, 16)))
    var sky_g_v = s.fvar(s.float_from_uint(channel(s, clear_word, 8)))
    var sky_b_v = s.fvar(s.float_from_uint(channel(s, clear_word, 0)))
    s.begin_if(phys_sky_on)
    var sky_hdr = _sample_sky_lut_rgb(s, res, cells, sky_dir, sky_key, sky_kc)
    s.store(sky_r_v, sky_hdr.x)
    s.store(sky_g_v, sky_hdr.y)
    s.store(sky_b_v, sky_hdr.z)
    s.end_if()
    s.begin_if(vol_clouds_on)
    var ctr = s.load_f(cl_out_tr)
    s.store(sky_r_v, s.fma(s.load_f(sky_r_v), ctr, s.load_f(cl_out_r)))
    s.store(sky_g_v, s.fma(s.load_f(sky_g_v), ctr, s.load_f(cl_out_g)))
    s.store(sky_b_v, s.fma(s.load_f(sky_b_v), ctr, s.load_f(cl_out_b)))
    s.end_if()
    var wse = s.load_f(wx_sky_e)
    s.store(sky_r_v, s.fma(wse, s.float(1.00), s.load_f(sky_r_v)))
    s.store(sky_g_v, s.fma(wse, s.float(0.97), s.load_f(sky_g_v)))
    s.store(sky_b_v, s.fma(wse, s.float(0.90), s.load_f(sky_b_v)))
    var ae_sky = s.load_f(auto_exp)
    var sk_r = s.fmul(s.fadd(s.load_f(sky_r_v), s.load_f(bl_r)), ae_sky)
    var sk_g = s.fmul(s.fadd(s.load_f(sky_g_v), s.load_f(bl_g)), ae_sky)
    var sk_b = s.fmul(s.fadd(s.load_f(sky_b_v), s.load_f(bl_b)), ae_sky)
    s.store(out_pixel, _tonemap_agx_sdr(s, res, sk_r, sk_g, sk_b, ign))
    s.begin_else()
    var cr = clamp_byte(s, s.iadd(channel(s, clear_word, 16), float_to_byte(s, s.load_f(bl_r))))
    var cg = clamp_byte(s, s.iadd(channel(s, clear_word, 8), float_to_byte(s, s.load_f(bl_g))))
    var cb = clamp_byte(s, s.iadd(channel(s, clear_word, 0), float_to_byte(s, s.load_f(bl_b))))
    s.store(out_pixel, s.bit_or(s.uint(0xFF000000), s.bit_or(s.shl(cr, s.uint(16)), s.bit_or(s.shl(cg, s.uint(8)), cb))))
    s.end_if()
    s.begin_else()

    var cur_r = s.fvar(s.load_float(target, s.iadd(plane_r, s.load_u(refract_px_r))))
    var cur_g = s.fvar(s.load_float(target, s.iadd(plane_g, base_px)))
    var cur_b = s.fvar(s.load_float(target, s.iadd(plane_b, s.load_u(refract_px_b))))
    var z_c = s.load_float(target, s.iadd(depth_plane, base_px))

    # 2. 12-Step Adaptive-Stride + Bisection Depth-Buffer Screen-Space Reflections (SSR) in Linear HDR
    var ssr_refl_u = s.bit_and(s.shr(gbuf, s.uint(24)), s.uint(127))
    var ssr_on = s.logical_and(
        s.ine(s.bit_and(effects, s.uint(EFFECT_SSR)), zero_u),
        s.logical_and(s.ult(s.uint(2), ssr_refl_u), s.fgt(z_c, s.float(0.002))),
    )
    s.begin_if(ssr_on)
    var ssr_strength = constant_float(s, res, CONSTANT_VOL_POST + 28)
    var surf_depth = s.fdiv(s.float(1.0), z_c)
    var world_p = Vec3Ids(
        s.fma(ray.x, surf_depth, camera.x),
        s.fma(ray.y, surf_depth, camera.y),
        s.fma(ray.z, surf_depth, camera.z),
    )
    var surf_n = _decode_octahedral_normal(s, gbuf)
    var inv_ray_len = s.inverse_sqrt(s.fadd(dot3(s, ray, ray), s.float(0.000001)))
    var inc_dir = Vec3Ids(s.fmul(ray.x, inv_ray_len), s.fmul(ray.y, inv_ray_len), s.fmul(ray.z, inv_ray_len))
    var two_dn = s.fmul(s.float(2.0), dot3(s, inc_dir, surf_n))
    var r_dir = Vec3Ids(
        s.fsub(inc_dir.x, s.fmul(two_dn, surf_n.x)),
        s.fsub(inc_dir.y, s.fmul(two_dn, surf_n.y)),
        s.fsub(inc_dir.z, s.fmul(two_dn, surf_n.z)),
    )
    s.begin_if(s.fgt(dot3(s, r_dir, forward), s.float(-0.20)))
    var step_dt = s.fvar(s.nmax(s.float(0.45), s.fmul(surf_depth, s.float(0.035))))
    var march_t = s.fvar(s.fmul(s.load_f(step_dt), s.fma(ign, s.float(0.55), s.float(0.75))))
    var prev_t = s.fvar(s.float(0.0))
    var hit_found = s.uvar(zero_u)
    var hit_px = s.uvar(base_px)
    var hit_edge = s.fvar(s.float(0.0))
    var ssr_i = s.uvar(zero_u)
    var stop_ssr = s.uvar(zero_u)

    s.begin_loop()
    var si = s.load_u(ssr_i)
    s.break_if(s.logical_or(s.uge(si, s.uint(12)), s.ine(s.load_u(stop_ssr), zero_u)))
    s.store(ssr_i, s.iadd(si, one_u))
    var t_cur = s.load_f(march_t)
    var qx = s.fma(r_dir.x, t_cur, world_p.x)
    var qy = s.fma(r_dir.y, t_cur, world_p.y)
    var qz = s.fma(r_dir.z, t_cur, world_p.z)
    var rel_q = Vec3Ids(s.fsub(qx, camera.x), s.fsub(qy, camera.y), s.fsub(qz, camera.z))
    var q_vz = dot3(s, rel_q, forward)
    s.begin_if(s.flt(q_vz, s.float(0.25)))
    s.store(stop_ssr, one_u)
    s.begin_else()
    var inv_qvz = s.fdiv(s.float(1.0), q_vz)
    var sx = s.fma(s.fmul(dot3(s, rel_q, right), focal), inv_qvz, half_width)
    var sy = s.fsub(half_height, s.fmul(s.fmul(dot3(s, rel_q, up), focal), inv_qvz))
    var in_scr = s.logical_and(
        s.logical_and(s.fge(sx, s.float(2.0)), s.flt(sx, s.fsub(width_f, s.float(2.0)))),
        s.logical_and(s.fge(sy, s.float(2.0)), s.flt(sy, s.fsub(height_f, s.float(2.0)))),
    )
    s.begin_if(s.logical_not(in_scr))
    s.store(stop_ssr, one_u)
    s.begin_else()
    var spx = s.iadd(s.imul(s.saturate_uint(sy), row_w), s.saturate_uint(sx))
    var iz_b = s.load_float(target, s.iadd(depth_plane, spx))
    var z_b = s.fdiv(s.float(1.0), s.nmax(iz_b, s.float(1e-8)))
    var diff = s.fsub(q_vz, z_b)
    var thick = s.nmax(s.float(1.10), s.fmul(t_cur, s.float(0.32)))
    var hit_cond = s.logical_and(
        s.fgt(iz_b, s.float(0.0016)),
        s.logical_and(s.fgt(diff, s.float(0.0)), s.flt(diff, thick)),
    )
    s.begin_if(hit_cond)
    # 1-step bisection refinement
    var t_mid = s.fmul(s.fadd(s.load_f(prev_t), t_cur), s.float(0.5))
    var mx = s.fma(r_dir.x, t_mid, world_p.x)
    var my = s.fma(r_dir.y, t_mid, world_p.y)
    var mz = s.fma(r_dir.z, t_mid, world_p.z)
    var rel_m = Vec3Ids(s.fsub(mx, camera.x), s.fsub(my, camera.y), s.fsub(mz, camera.z))
    var m_vz = s.nmax(dot3(s, rel_m, forward), s.float(0.25))
    var inv_mvz = s.fdiv(s.float(1.0), m_vz)
    var msx = s.nclamp(s.fma(s.fmul(dot3(s, rel_m, right), focal), inv_mvz, half_width), s.float(1.0), s.fsub(width_f, s.float(2.0)))
    var msy = s.nclamp(s.fsub(half_height, s.fmul(s.fmul(dot3(s, rel_m, up), focal), inv_mvz)), s.float(1.0), s.fsub(height_f, s.float(2.0)))
    var mpx = s.iadd(s.imul(s.saturate_uint(msy), row_w), s.saturate_uint(msx))
    var miz = s.load_float(target, s.iadd(depth_plane, mpx))
    var use_mid = s.fgt(m_vz, s.fdiv(s.float(1.0), s.nmax(miz, s.float(1e-8))))
    s.store(hit_px, s.select_u(use_mid, mpx, spx))
    var fx_n = s.fabs(s.fma(s.fdiv(sx, width_f), s.float(2.0), s.float(-1.0)))
    var fy_n = s.fabs(s.fma(s.fdiv(sy, height_f), s.float(2.0), s.float(-1.0)))
    var edge_v = s.nclamp(s.fmul(s.fsub(s.float(1.0), s.nmax(fx_n, fy_n)), s.float(4.5)), s.float(0.0), s.float(1.0))
    s.store(hit_edge, edge_v)
    s.store(hit_found, one_u)
    s.store(stop_ssr, one_u)
    s.begin_else()
    s.store(prev_t, t_cur)
    var dt_next = s.fmul(s.load_f(step_dt), s.float(1.34))
    s.store(step_dt, dt_next)
    s.store(march_t, s.fadd(t_cur, dt_next))
    s.end_if()
    s.end_if()
    s.end_if()
    s.end_loop()

    s.begin_if(s.ine(s.load_u(hit_found), zero_u))
    var hpx = s.load_u(hit_px)
    var hr_hit = s.nmin(s.load_float(target, s.iadd(plane_r, hpx)), s.float(340.0))
    var hg_hit = s.nmin(s.load_float(target, s.iadd(plane_g, hpx)), s.float(340.0))
    var hb_hit = s.nmin(s.load_float(target, s.iadd(plane_b, hpx)), s.float(340.0))
    var refl_f = s.fmul(s.float_from_uint(ssr_refl_u), s.float(0.007874))
    var w_ssr = s.nclamp(s.fmul(s.fmul(refl_f, s.load_f(hit_edge)), ssr_strength), s.float(0.0), s.float(0.65))
    var keep_base = s.fsub(s.float(1.0), s.fmul(w_ssr, s.float(0.45)))
    s.store(cur_r, s.fma(hr_hit, w_ssr, s.fmul(s.load_f(cur_r), keep_base)))
    s.store(cur_g, s.fma(hg_hit, w_ssr, s.fmul(s.load_f(cur_g), keep_base)))
    s.store(cur_b, s.fma(hb_hit, w_ssr, s.fmul(s.load_f(cur_b), keep_base)))
    s.end_if()
    s.end_if()
    s.end_if()

    # 3. Temporally Reprojected 3D Froxel Volumetrics (64x36x16) + 10-Step Per-Pixel Raymarch Fallback
    s.begin_if(vol_on)
    var vol_density = constant_float(s, res, CONSTANT_VOL_POST + 24)
    s.begin_if(s.fgt(vol_density, s.float(0.0005)))
    var ray_len = s.fsqrt(s.fadd(dot3(s, ray, ray), s.float(0.000001)))
    var inv_rl = s.fdiv(s.float(1.0), ray_len)
    var ray_dir = Vec3Ids(s.fmul(ray.x, inv_rl), s.fmul(ray.y, inv_rl), s.fmul(ray.z, inv_rl))
    var raw_z = s.fdiv(s.float(1.0), s.nmax(z_c, s.float(0.00001)))
    var trans = s.fvar(s.float(1.0))
    var scat_r = s.fvar(s.float(0.0))
    var scat_g = s.fvar(s.float(0.0))
    var scat_b = s.fvar(s.float(0.0))

    s.begin_if(froxel_on)
    var frox_max = s.nmax(constant_float(s, res, CONSTANT_V8_ATMOS + 24), s.float(10.0))
    var surf_w_dist = s.nmin(s.fmul(raw_z, ray_len), frox_max)
    var fu_c = s.nclamp(
        s.fsub(s.fmul(s.fdiv(s.fadd(x_f, s.float(0.5)), width_f), s.float(Float32(FROXEL_X))), s.float(0.5)),
        s.float(0.0),
        s.float(Float32(FROXEL_X - 1)),
    )
    var fv_c = s.nclamp(
        s.fsub(s.fmul(s.fdiv(s.fadd(y_f, s.float(0.5)), height_f), s.float(Float32(FROXEL_Y))), s.float(0.5)),
        s.float(0.0),
        s.float(Float32(FROXEL_Y - 1)),
    )
    var fz_norm = s.fsqrt(s.nclamp(s.fdiv(surf_w_dist, frox_max), s.float(0.0), s.float(1.0)))
    var fz_c = s.nclamp(
        s.fsub(s.fmul(fz_norm, s.float(Float32(FROXEL_Z))), s.float(0.5)),
        s.float(0.0),
        s.float(Float32(FROXEL_Z - 1)),
    )
    _sample_froxel_trilinear(s, cells, fu_c, fv_c, fz_c, scat_r, scat_g, scat_b, trans)
    s.begin_else()
    var vol_fog = constant_vector(s, res, CONSTANT_VOL_POST)
    var height_fall = constant_float(s, res, CONSTANT_VOL_POST + 20)
    var to_key = constant_vector(s, res, CONSTANT_KEY)
    var key_color = constant_vector(s, res, CONSTANT_KEY + 12)
    var max_dist = s.nmin(s.fmul(raw_z, ray_len), s.float(175.0))

    # Henyey-Greenstein forward scattering phase (g = 0.64)
    var cos_th = dot3(s, ray_dir, to_key)
    var hg_den = s.fsub(s.float(1.4096), s.fmul(s.float(1.28), cos_th))
    var hg_phase = s.nmin(
        s.fdiv(s.float(0.04698), s.nmax(s.fmul(hg_den, s.fsqrt(hg_den)), s.float(0.02))),
        s.float(1.55),
    )
    var sun_r = s.fmul(key_color.x, s.fmul(hg_phase, s.float(195.0)))
    var sun_g = s.fmul(key_color.y, s.fmul(hg_phase, s.float(190.0)))
    var sun_b = s.fmul(key_color.z, s.fmul(hg_phase, s.float(172.0)))

    var light_cnt = s.umin(constant_word(s, res, CONSTANT_WORLD + 24), s.uint(4))
    var vs = s.uvar(zero_u)

    s.begin_loop()
    var step_i = s.load_u(vs)
    s.break_if(s.uge(step_i, s.uint(10)))
    s.store(vs, s.iadd(step_i, one_u))
    var u_s = s.fmul(s.fadd(s.float_from_uint(step_i), ign), s.float(0.1))
    var dist_s = s.fmul(max_dist, s.fma(u_s, u_s, s.fmul(u_s, s.float(0.05))))
    var dt_s = s.fmul(max_dist, s.fma(u_s, s.float(0.19), s.float(0.005)))
    var px_s = Vec3Ids(
        s.fma(ray_dir.x, dist_s, camera.x),
        s.fma(ray_dir.y, dist_s, camera.y),
        s.fma(ray_dir.z, dist_s, camera.z),
    )
    var alt_h = s.nmax(s.fadd(px_s.y, s.float(6.0)), s.float(0.0))
    var rho = s.fmul(s.exp2(s.fsub(s.float(0.0), s.fmul(alt_h, height_fall))), s.fmul(vol_density, s.float(0.0036)))
    var ext = s.nmin(s.fmul(rho, dt_s), s.float(0.25))
    var vis_s = _sample_volumetric_shadow(s, res, px_s, to_key)

    var loc_r = s.fvar(s.float(0.0))
    var loc_g = s.fvar(s.float(0.0))
    var loc_b = s.fvar(s.float(0.0))
    var li = s.uvar(zero_u)
    s.begin_loop()
    var l_idx = s.load_u(li)
    s.break_if(s.uge(l_idx, light_cnt))
    s.store(li, s.iadd(l_idx, one_u))
    var lw = s.iadd(s.uint(UInt32(CONSTANT_LIGHTS // 4)), s.shl(l_idx, s.uint(3)))
    var dlx = s.fsub(light_float(s, res, lw, 0), px_s.x)
    var dly = s.fsub(light_float(s, res, lw, 1), px_s.y)
    var dlz = s.fsub(light_float(s, res, lw, 2), px_s.z)
    var d2 = s.fadd(dot3(s, Vec3Ids(dlx, dly, dlz), Vec3Ids(dlx, dly, dlz)), s.float(1.5))
    var fall = s.nmax(s.fsub(s.float(1.0), s.fmul(light_float(s, res, lw, 7), d2)), s.float(0.0))
    var l_att = s.fmul(s.fmul(fall, fall), s.float(460.0))
    s.store(loc_r, s.fma(light_float(s, res, lw, 4), l_att, s.load_f(loc_r)))
    s.store(loc_g, s.fma(light_float(s, res, lw, 5), l_att, s.load_f(loc_g)))
    s.store(loc_b, s.fma(light_float(s, res, lw, 6), l_att, s.load_f(loc_b)))
    s.end_loop()

    var w_step = s.fmul(s.load_f(trans), ext)
    var in_r = s.fadd(s.fma(vol_fog.x, s.float(0.62), s.fmul(sun_r, vis_s)), s.load_f(loc_r))
    var in_g = s.fadd(s.fma(vol_fog.y, s.float(0.62), s.fmul(sun_g, vis_s)), s.load_f(loc_g))
    var in_b = s.fadd(s.fma(vol_fog.z, s.float(0.62), s.fmul(sun_b, vis_s)), s.load_f(loc_b))
    s.store(scat_r, s.fma(in_r, w_step, s.load_f(scat_r)))
    s.store(scat_g, s.fma(in_g, w_step, s.load_f(scat_g)))
    s.store(scat_b, s.fma(in_b, w_step, s.load_f(scat_b)))
    s.store(trans, s.fmul(s.load_f(trans), s.fsub(s.float(1.0), ext)))
    s.end_loop()
    s.end_if()

    var tr_f = s.load_f(trans)
    s.store(cur_r, s.fma(s.load_f(cur_r), tr_f, s.load_f(scat_r)))
    s.store(cur_g, s.fma(s.load_f(cur_g), tr_f, s.load_f(scat_g)))
    s.store(cur_b, s.fma(s.load_f(cur_b), tr_f, s.load_f(scat_b)))
    s.end_if()
    s.end_if()

    # 3b. Altitude-aware Hillaire Aerial Perspective on distant terrain/hulls when EFFECT_PHYSICAL_SKY is enabled
    s.begin_if(phys_sky_on)
    var aer_str = constant_float(s, res, CONSTANT_V8_ATMOS + 40)
    s.begin_if(s.fgt(aer_str, s.float(0.001)))
    var surf_z_ap = s.fdiv(s.float(1.0), s.nmax(z_c, s.float(0.00001)))
    var alt_km_ap = constant_float(s, res, CONSTANT_V8_ATMOS)
    var dens_ap = s.exp2(s.fsub(s.float(0.0), s.fmul(s.nmax(alt_km_ap, s.float(0.0)), s.float(0.1803))))
    var aer_t = s.nclamp(
        s.fmul(s.fsub(s.float(1.0), s.exp2(s.fsub(s.float(0.0), s.fmul(surf_z_ap, s.float(0.0018))))), s.fmul(aer_str, dens_ap)),
        s.float(0.0),
        s.float(0.58),
    )
    s.begin_if(s.fgt(aer_t, s.float(0.004)))
    var inv_ap_l = s.inverse_sqrt(s.fadd(dot3(s, ray, ray), s.float(0.000001)))
    var ap_dir = Vec3Ids(s.fmul(ray.x, inv_ap_l), s.fmul(ray.y, inv_ap_l), s.fmul(ray.z, inv_ap_l))
    var ap_key = constant_vector(s, res, CONSTANT_KEY)
    var ap_kc = constant_vector(s, res, CONSTANT_KEY + 12)
    var ap_sky = _sample_sky_lut_rgb(s, res, cells, ap_dir, ap_key, ap_kc)
    var c_ap_r = s.load_f(cur_r)
    var c_ap_g = s.load_f(cur_g)
    var c_ap_b = s.load_f(cur_b)
    s.store(cur_r, s.fma(s.fsub(ap_sky.x, c_ap_r), aer_t, c_ap_r))
    s.store(cur_g, s.fma(s.fsub(ap_sky.y, c_ap_g), aer_t, c_ap_g))
    s.store(cur_b, s.fma(s.fsub(ap_sky.z, c_ap_b), aer_t, c_ap_b))
    s.end_if()
    s.end_if()
    s.end_if()

    # 3c. V9 Ray-Marched 3D Volumetric Cloudscapes in front of high terrain/distant hulls
    s.begin_if(vol_clouds_on)
    var ctr_g = s.load_f(cl_out_tr)
    s.store(cur_r, s.fma(s.load_f(cur_r), ctr_g, s.load_f(cl_out_r)))
    s.store(cur_g, s.fma(s.load_f(cur_g), ctr_g, s.load_f(cl_out_g)))
    s.store(cur_b, s.fma(s.load_f(cur_b), ctr_g, s.load_f(cl_out_b)))
    s.end_if()

    # 4. Horizon-Based GTAO + V9 Screen-Space Diffuse GI (SSGI) & Bent Normals + 8-Step Contact Shadows
    var ao_mult = s.fvar(s.float(1.0))
    s.begin_if(s.fgt(z_c, s.float(0.0016)))
    var use_gtao = s.ine(s.bit_and(effects, s.uint(EFFECT_GTAO_CONTACT)), zero_u)
    var ssgi_on = s.ine(s.bit_and(effects, s.uint(EFFECT_SSGI_BENT)), zero_u)
    s.begin_if(use_gtao)
    var surf_d = s.fdiv(s.float(1.0), z_c)
    var surf_n_g = _decode_octahedral_normal(s, gbuf)
    var n_rx = dot3(s, surf_n_g, right)
    var n_uy = dot3(s, surf_n_g, up)
    var n_fz = dot3(s, surf_n_g, forward)
    var cp_x = s.fmul(vx_cam, surf_d)
    var cp_y = s.fmul(vy_cam, surf_d)

    var rot_ph = s.fmul(ign, s.float(0.5))
    var s_g = _parabolic_sin(s, rot_ph)
    var c_g = _parabolic_sin(s, s.fadd(rot_ph, s.float(0.25)))
    var neg_s_g = s.fsub(s.float(0.0), s_g)
    var neg_c_g = s.fsub(s.float(0.0), c_g)

    var dir_xs = List[UInt32]()
    var dir_ys = List[UInt32]()
    dir_xs.append(c_g)
    dir_ys.append(s_g)
    dir_xs.append(neg_s_g)
    dir_ys.append(c_g)
    dir_xs.append(neg_c_g)
    dir_ys.append(neg_s_g)
    dir_xs.append(s_g)
    dir_ys.append(neg_c_g)

    var max_r2 = s.nmax(s.float(0.38), s.fmul(s.fmul(surf_d, surf_d), s.float(0.038)))
    var inv_max_r2 = s.fdiv(s.float(1.0), max_r2)
    var h_sum = s.fvar(s.float(0.0))
    var bent_x = s.fvar(n_rx)
    var bent_y = s.fvar(n_uy)
    var bent_z = s.fvar(n_fz)
    var ssgi_r = s.fvar(s.float(0.0))
    var ssgi_g = s.fvar(s.float(0.0))
    var ssgi_b = s.fvar(s.float(0.0))
    var px_cx_g = s.fadd(x_f, s.float(0.5))
    var px_cy_g = s.fadd(y_f, s.float(0.5))

    for d_idx in range(4):
        var max_h = s.fvar(s.float(0.09))
        for r_step in range(2):
            var rad_px = s.float(Float32(2.6) if r_step == 0 else Float32(6.8))
            var sx_g = s.nclamp(s.fma(dir_xs[d_idx], rad_px, px_cx_g), s.float(0.5), s.fsub(width_f, s.float(0.5)))
            var sy_g = s.nclamp(s.fma(dir_ys[d_idx], rad_px, px_cy_g), s.float(0.5), s.fsub(height_f, s.float(0.5)))
            var spx_g = s.iadd(s.imul(s.saturate_uint(sy_g), row_w), s.saturate_uint(sx_g))
            var iz_s = s.load_float(target, s.iadd(depth_plane, spx_g))
            s.begin_if(s.fgt(iz_s, s.float(0.0016)))
            var d_s = s.fdiv(s.float(1.0), iz_s)
            var vx_s = s.fmul(s.fsub(sx_g, half_width), inverse_focal)
            var vy_s = s.fmul(s.fsub(half_height, sy_g), inverse_focal)
            var dv_x = s.fsub(s.fmul(vx_s, d_s), cp_x)
            var dv_y = s.fsub(s.fmul(vy_s, d_s), cp_y)
            var dv_z = s.fsub(d_s, surf_d)
            var dist2 = s.fadd(s.fma(dv_z, dv_z, s.fma(dv_y, dv_y, s.fmul(dv_x, dv_x))), s.float(0.0001))
            var inv_d = s.inverse_sqrt(dist2)
            var dot_hn = s.fma(dv_z, n_fz, s.fma(dv_y, n_uy, s.fmul(dv_x, n_rx)))
            var sin_h = s.fmul(dot_hn, inv_d)
            var rng_w = s.nclamp(s.fsub(s.float(1.0), s.fmul(dist2, inv_max_r2)), s.float(0.0), s.float(1.0))
            var h_val = s.fmul(sin_h, rng_w)
            s.store(max_h, s.nmax(s.load_f(max_h), h_val))
            s.begin_if(s.logical_and(ssgi_on, s.fgt(h_val, s.float(0.10))))
            var push_b = s.fmul(h_val, s.fmul(inv_d, s.float(0.35)))
            s.store(bent_x, s.fsub(s.load_f(bent_x), s.fmul(dv_x, push_b)))
            s.store(bent_y, s.fsub(s.load_f(bent_y), s.fmul(dv_y, push_b)))
            s.store(bent_z, s.fsub(s.load_f(bent_z), s.fmul(dv_z, push_b)))
            var gi_w = s.fmul(s.nmin(h_val, s.float(0.85)), s.float(0.125))
            var nr = s.nmin(s.load_float(target, s.iadd(plane_r, spx_g)), s.float(240.0))
            var ng = s.nmin(s.load_float(target, s.iadd(plane_g, spx_g)), s.float(240.0))
            var nb = s.nmin(s.load_float(target, s.iadd(plane_b, spx_g)), s.float(240.0))
            s.store(ssgi_r, s.fma(nr, gi_w, s.load_f(ssgi_r)))
            s.store(ssgi_g, s.fma(ng, gi_w, s.load_f(ssgi_g)))
            s.store(ssgi_b, s.fma(nb, gi_w, s.load_f(ssgi_b)))
            s.end_if()
            s.end_if()
        s.store(h_sum, s.fadd(s.load_f(h_sum), s.fsub(s.load_f(max_h), s.float(0.09))))

    s.begin_if(ssgi_on)
    var bx_l = s.load_f(bent_x)
    var by_l = s.load_f(bent_y)
    var bz_l = s.load_f(bent_z)
    var inv_bl = s.inverse_sqrt(s.fadd(dot3(s, Vec3Ids(bx_l, by_l, bz_l), Vec3Ids(bx_l, by_l, bz_l)), s.float(0.0001)))
    var bent_world_y = s.fmul(
        s.fma(bz_l, forward.y, s.fma(by_l, up.y, s.fmul(bx_l, right.y))),
        inv_bl,
    )
    var bent_sky = s.fma(s.fsub(bent_world_y, surf_n_g.y), s.float(0.12), s.float(1.0))
    var ssgi_str = constant_float(s, res, CONSTANT_V9_CLOUDS + 32)
    s.store(cur_r, s.fma(s.load_f(ssgi_r), ssgi_str, s.fmul(s.load_f(cur_r), bent_sky)))
    s.store(cur_g, s.fma(s.load_f(ssgi_g), ssgi_str, s.fmul(s.load_f(cur_g), bent_sky)))
    s.store(cur_b, s.fma(s.load_f(ssgi_b), ssgi_str, s.fmul(s.load_f(cur_b), bent_sky)))
    s.end_if()

    var gtao_occ = s.nclamp(s.fmul(s.load_f(h_sum), s.float(0.115)), s.float(0.0), s.float(0.34))
    var cs_atten = s.fvar(s.float(1.0))

    # 8-Step Screen-Space Contact Shadow raymarch toward key sun for sub-CSM micro-crevices
    var to_key_cs = constant_vector(s, res, CONSTANT_KEY)
    var n_dot_k = dot3(s, surf_n_g, to_key_cs)
    var do_cs = s.logical_and(
        s.ine(s.bit_and(effects, s.uint(EFFECT_SHADOWS)), zero_u),
        s.logical_and(s.fgt(n_dot_k, s.float(0.04)), s.flt(surf_d, s.float(95.0))),
    )
    s.begin_if(do_cs)
    var lk_x = dot3(s, to_key_cs, right)
    var lk_y = dot3(s, to_key_cs, up)
    var lk_z = dot3(s, to_key_cs, forward)
    var step_base = s.fmul(s.float(0.135), s.nclamp(s.fmul(surf_d, s.float(0.055)), s.float(0.35), s.float(1.35)))
    var min_bias = s.fma(surf_d, s.float(0.0024), s.float(0.014))
    var cs_i = s.uvar(zero_u)
    var cs_hit = s.uvar(zero_u)
    s.begin_loop()
    var ci = s.load_u(cs_i)
    s.break_if(s.logical_or(s.uge(ci, s.uint(8)), s.ine(s.load_u(cs_hit), zero_u)))
    s.store(cs_i, s.iadd(ci, one_u))
    var t_cs = s.fmul(s.fadd(s.float_from_uint(ci), s.fma(ign, s.float(0.65), s.float(0.35))), step_base)
    var rz_c = s.fma(lk_z, t_cs, surf_d)
    s.begin_if(s.fgt(rz_c, s.float(0.20)))
    var rx_c = s.fma(lk_x, t_cs, cp_x)
    var ry_c = s.fma(lk_y, t_cs, cp_y)
    var inv_rzc = s.fdiv(s.float(1.0), rz_c)
    var sx_c = s.fma(s.fmul(rx_c, focal), inv_rzc, half_width)
    var sy_c = s.fsub(half_height, s.fmul(s.fmul(ry_c, focal), inv_rzc))
    var in_cs = s.logical_and(
        s.logical_and(s.fge(sx_c, s.float(1.0)), s.flt(sx_c, s.fsub(width_f, s.float(1.0)))),
        s.logical_and(s.fge(sy_c, s.float(1.0)), s.flt(sy_c, s.fsub(height_f, s.float(1.0)))),
    )
    s.begin_if(in_cs)
    var cpx = s.iadd(s.imul(s.saturate_uint(sy_c), row_w), s.saturate_uint(sx_c))
    var iz_cs = s.load_float(target, s.iadd(depth_plane, cpx))
    var bz_cs = s.fdiv(s.float(1.0), s.nmax(iz_cs, s.float(1e-8)))
    var dz_cs = s.fsub(rz_c, bz_cs)
    var thick_cs = s.fma(t_cs, s.float(0.28), s.float(0.045))
    s.begin_if(s.logical_and(s.fgt(iz_cs, s.float(0.0016)), s.logical_and(s.fgt(dz_cs, min_bias), s.flt(dz_cs, thick_cs))))
    s.store(cs_hit, one_u)
    s.end_if()
    s.end_if()
    s.end_if()
    s.end_loop()
    s.begin_if(s.ine(s.load_u(cs_hit), zero_u))
    s.store(cs_atten, s.float(0.70))
    s.end_if()
    s.end_if()

    s.store(ao_mult, s.fmul(s.fsub(s.float(1.0), gtao_occ), s.load_f(cs_atten)))
    s.begin_else()
    var two_u = s.uint(2)
    var four_u = s.uint(4)
    var five_u = s.uint(5)
    var xp2 = s.smin(s.iadd(column, two_u), max_px_x)
    var xn2 = s.smax(s.isub(column, two_u), zero_u)
    var yp2 = s.smin(s.iadd(line, two_u), max_px_y)
    var yn2 = s.smax(s.isub(line, two_u), zero_u)
    var xp5 = s.smin(s.iadd(column, five_u), max_px_x)
    var xn5 = s.smax(s.isub(column, five_u), zero_u)
    var yp4 = s.smin(s.iadd(line, four_u), max_px_y)
    var yn4 = s.smax(s.isub(line, four_u), zero_u)
    var o1 = _ssao_symmetric_pair(s, target, depth_plane, row_w, xp2, line, xn2, line, z_c, 13.5)
    var o2 = _ssao_symmetric_pair(s, target, depth_plane, row_w, column, yp2, column, yn2, z_c, 13.5)
    var o3 = _ssao_symmetric_pair(s, target, depth_plane, row_w, xp5, yp4, xn5, yn4, z_c, 9.0)
    var o4 = _ssao_symmetric_pair(s, target, depth_plane, row_w, xp5, yn4, xn5, yp4, z_c, 9.0)
    var occ_sum = s.nmin(s.fadd(s.fadd(o1, o2), s.fadd(o3, o4)), s.float(0.32))
    s.store(ao_mult, s.fsub(s.float(1.0), occ_sum))
    s.end_if()
    s.end_if()

    var ao_f = s.load_f(ao_mult)

    # 4b. V10 World-Space Radiance Cascades GI (EFFECT_RADIANCE_GI) — off-screen & multi-bounce light propagation
    var rad_gi_on = s.logical_and(
        s.ine(s.bit_and(effects, s.uint(EFFECT_RADIANCE_GI)), zero_u),
        s.fgt(z_c, s.float(0.0016)),
    )
    s.begin_if(rad_gi_on)
    var rgi_str = constant_float(s, res, CONSTANT_V10_POST)
    var rgi_cnt = s.umin(constant_word(s, res, CONSTANT_V10_POST + 4), s.uint(UInt32(MAX_RADIANCE_CASCADES)))
    s.begin_if(s.fgt(rgi_str, s.float(0.001)))
    var rgi_d = s.fdiv(s.float(1.0), z_c)
    var rgi_wp = Vec3Ids(
        s.fma(ray.x, rgi_d, camera.x),
        s.fma(ray.y, rgi_d, camera.y),
        s.fma(ray.z, rgi_d, camera.z),
    )
    var rgi_n = _decode_octahedral_normal(s, gbuf)
    var env_sky_gi = constant_vector(s, res, CONSTANT_ENV_SKY)
    var env_gnd_gi = constant_vector(s, res, CONSTANT_ENV_GROUND)
    var env_hor_gi = constant_vector(s, res, CONSTANT_ENV_HORIZON)
    var up_w = s.fma(rgi_n.y, s.float(0.5), s.float(0.5))
    var gnd_w = s.fsub(s.float(1.0), up_w)
    var hor_w = s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fabs(rgi_n.y)))
    var gi_acc_r = s.fvar(s.fmul(s.fma(env_hor_gi.x, s.fmul(hor_w, s.float(0.35)), s.fma(env_sky_gi.x, up_w, s.fmul(env_gnd_gi.x, gnd_w))), s.float(18.0)))
    var gi_acc_g = s.fvar(s.fmul(s.fma(env_hor_gi.y, s.fmul(hor_w, s.float(0.35)), s.fma(env_sky_gi.y, up_w, s.fmul(env_gnd_gi.y, gnd_w))), s.float(18.0)))
    var gi_acc_b = s.fvar(s.fmul(s.fma(env_hor_gi.z, s.fmul(hor_w, s.float(0.35)), s.fma(env_sky_gi.z, up_w, s.fmul(env_gnd_gi.z, gnd_w))), s.float(18.0)))
    var rc_i = s.uvar(zero_u)
    var rc_base0 = s.uint(UInt32(CONSTANT_RADIANCE_CASCADES // 4))
    s.begin_loop()
    var ri = s.load_u(rc_i)
    s.break_if(s.uge(ri, rgi_cnt))
    s.store(rc_i, s.iadd(ri, one_u))
    var r_off = s.iadd(rc_base0, s.imul(ri, s.uint(8)))
    var npx = s.load_float(res.constants, r_off)
    var npy = s.load_float(res.constants, s.iadd(r_off, s.uint(1)))
    var npz = s.load_float(res.constants, s.iadd(r_off, s.uint(2)))
    var inv_r2 = s.load_float(res.constants, s.iadd(r_off, s.uint(3)))
    var dx_rc = s.fsub(npx, rgi_wp.x)
    var dy_rc = s.fsub(npy, rgi_wp.y)
    var dz_rc = s.fsub(npz, rgi_wp.z)
    var d2_rc = s.fma(dz_rc, dz_rc, s.fma(dy_rc, dy_rc, s.fmul(dx_rc, dx_rc)))
    var t_rc = s.fsub(s.float(1.0), s.fmul(d2_rc, inv_r2))
    s.begin_if(s.fgt(t_rc, s.float(0.0)))
    var inv_dl = s.inverse_sqrt(s.fadd(d2_rc, s.float(0.25)))
    var ndl_rc = s.fma(
        s.fmul(s.fma(dz_rc, rgi_n.z, s.fma(dy_rc, rgi_n.y, s.fmul(dx_rc, rgi_n.x))), inv_dl),
        s.float(0.45),
        s.float(0.55),
    )
    var dir_bias = s.load_float(res.constants, s.iadd(r_off, s.uint(7)))
    var aniso_w = s.nmax(s.float(0.25), s.fma(rgi_n.y, dir_bias, s.float(1.0)))
    var w_rc = s.fmul(s.fmul(t_rc, t_rc), s.fmul(s.nmax(ndl_rc, s.float(0.12)), aniso_w))
    var er_rc = s.load_float(res.constants, s.iadd(r_off, s.uint(4)))
    var eg_rc = s.load_float(res.constants, s.iadd(r_off, s.uint(5)))
    var eb_rc = s.load_float(res.constants, s.iadd(r_off, s.uint(6)))
    s.store(gi_acc_r, s.fma(er_rc, w_rc, s.load_f(gi_acc_r)))
    s.store(gi_acc_g, s.fma(eg_rc, w_rc, s.load_f(gi_acc_g)))
    s.store(gi_acc_b, s.fma(eb_rc, w_rc, s.load_f(gi_acc_b)))
    s.end_if()
    s.end_loop()
    var rgi_w = s.fmul(rgi_str, s.fma(ao_f, s.float(0.75), s.float(0.25)))
    s.store(cur_r, s.fma(s.load_f(gi_acc_r), rgi_w, s.load_f(cur_r)))
    s.store(cur_g, s.fma(s.load_f(gi_acc_g), rgi_w, s.load_f(cur_g)))
    s.store(cur_b, s.fma(s.load_f(gi_acc_b), rgi_w, s.load_f(cur_b)))
    s.end_if()
    s.end_if()

    # 4c. V10 1984 DYRL 35mm Cel-Film / PBR Hybrid Mode (EFFECT_DYRL_ANIME)
    var dyrl_on = s.logical_and(
        s.ine(s.bit_and(effects, s.uint(EFFECT_DYRL_ANIME)), zero_u),
        s.fgt(z_c, s.float(0.0016)),
    )
    s.begin_if(dyrl_on)
    var ink_str = constant_float(s, res, CONSTANT_V10_POST + 8)
    var cel_bands = s.nmax(constant_float(s, res, CONSTANT_V10_POST + 12), s.float(2.0))
    var glint_str = constant_float(s, res, CONSTANT_V10_POST + 16)
    var hal_str = constant_float(s, res, CONSTANT_V10_POST + 20)

    # Smooth 3-band cel luminance quantization preserving Golden-Hour hue & white point
    var cr0 = s.load_f(cur_r)
    var cg0 = s.load_f(cur_g)
    var cb0 = s.load_f(cur_b)
    var lum_cel = s.fma(cb0, s.float(0.1529), s.fma(cg0, s.float(0.5882), s.fmul(cr0, s.float(0.2588))))
    var u_cel = s.fdiv(lum_cel, s.fadd(lum_cel, s.float(92.0)))
    var u_sc = s.fmul(u_cel, cel_bands)
    var band_fl = s.ffloor(u_sc)
    var band_fr = s.fsub(u_sc, band_fl)
    var soft_step = s.nclamp(s.fmul(s.fsub(band_fr, s.float(0.36)), s.float(3.4)), s.float(0.0), s.float(1.0))
    var q_u = s.fdiv(s.fadd(s.fadd(band_fl, soft_step), s.float(0.18)), s.fadd(cel_bands, s.float(0.18)))
    var tgt_lum = s.fdiv(s.fmul(s.float(92.0), q_u), s.nmax(s.fsub(s.float(1.0), s.nmin(q_u, s.float(0.90))), s.float(0.06)))
    var raw_cel_s = s.nclamp(s.fdiv(tgt_lum, s.nmax(lum_cel, s.float(1.0))), s.float(0.54), s.float(1.38))
    var cel_scale = s.fma(s.fsub(raw_cel_s, s.float(1.0)), s.float(0.72), s.float(1.0))
    var cr1 = s.fmul(cr0, cel_scale)
    var cg1 = s.fmul(cg0, cel_scale)
    var cb1 = s.fmul(cb0, cel_scale)

    # Sobel depth + octahedral G-buffer normal variable-weight charcoal-sepia ink outlines
    var xm1_d = s.smax(s.isub(column, one_u), zero_u)
    var xp1_d = s.smin(s.iadd(column, one_u), max_px_x)
    var ym1_d = s.smax(s.isub(line, one_u), zero_u)
    var yp1_d = s.smin(s.iadd(line, one_u), max_px_y)
    var idx_n = s.iadd(s.imul(ym1_d, row_w), column)
    var idx_s = s.iadd(s.imul(yp1_d, row_w), column)
    var idx_w = s.iadd(s.imul(line, row_w), xm1_d)
    var idx_e = s.iadd(s.imul(line, row_w), xp1_d)
    var zn_d = s.load_float(target, s.iadd(depth_plane, idx_n))
    var zs_d = s.load_float(target, s.iadd(depth_plane, idx_s))
    var zw_d = s.load_float(target, s.iadd(depth_plane, idx_w))
    var ze_d = s.load_float(target, s.iadd(depth_plane, idx_e))
    var dz_sum = s.fadd(
        s.fadd(s.fabs(s.fsub(zn_d, z_c)), s.fabs(s.fsub(zs_d, z_c))),
        s.fadd(s.fabs(s.fsub(zw_d, z_c)), s.fabs(s.fsub(ze_d, z_c))),
    )
    var dz_rel = s.fdiv(dz_sum, s.nmax(z_c, s.float(0.0025)))
    var nw_o = _decode_octahedral_normal(s, s.load_word(target, s.iadd(gbuf_plane, idx_w)))
    var ne_o = _decode_octahedral_normal(s, s.load_word(target, s.iadd(gbuf_plane, idx_e)))
    var nn_o = _decode_octahedral_normal(s, s.load_word(target, s.iadd(gbuf_plane, idx_n)))
    var ns_o = _decode_octahedral_normal(s, s.load_word(target, s.iadd(gbuf_plane, idx_s)))
    var near_fade = s.nclamp(s.fmul(z_c, s.float(22.0)), s.float(0.15), s.float(1.0))
    var dn_crease = s.fmul(
        s.fadd(s.fsub(s.float(1.0), dot3(s, ne_o, nw_o)), s.fsub(s.float(1.0), dot3(s, nn_o, ns_o))),
        near_fade,
    )
    var edge_raw = s.fma(
        s.nmax(s.fsub(dz_rel, s.float(0.085)), s.float(0.0)),
        s.float(3.8),
        s.fmul(s.nmax(s.fsub(dn_crease, s.float(0.24)), s.float(0.0)), s.float(0.95)),
    )
    var ink_w = s.nclamp(s.fmul(edge_raw, ink_str), s.float(0.0), s.float(0.84))
    var keep_ink = s.fsub(s.float(1.0), ink_w)
    var cr2 = s.fma(cr1, keep_ink, s.fmul(s.float(9.0), ink_w))
    var cg2 = s.fma(cg1, keep_ink, s.fmul(s.float(8.0), ink_w))
    var cb2 = s.fma(cb1, keep_ink, s.fmul(s.float(10.0), ink_w))

    # Haruhiko Mikimoto 4-point cross-star specular glints + warm 35mm Eastmancolor halation & grain
    var three_u = s.uint(3)
    var gl_w = s.iadd(s.imul(line, row_w), s.smax(s.isub(column, three_u), zero_u))
    var gl_e = s.iadd(s.imul(line, row_w), s.smin(s.iadd(column, three_u), max_px_x))
    var gl_n = s.iadd(s.imul(s.smax(s.isub(line, three_u), zero_u), row_w), column)
    var gl_s = s.iadd(s.imul(s.smin(s.iadd(line, three_u), max_px_y), row_w), column)
    var pk_h = s.fmul(s.fadd(s.load_float(target, s.iadd(plane_r, gl_w)), s.load_float(target, s.iadd(plane_r, gl_e))), s.float(0.5))
    var pk_v = s.fmul(s.fadd(s.load_float(target, s.iadd(plane_r, gl_n)), s.load_float(target, s.iadd(plane_r, gl_s))), s.float(0.5))
    var pk_max = s.nmax(pk_h, pk_v)
    var glint_e = s.fmul(s.nclamp(s.fmul(s.fsub(pk_max, s.float(235.0)), s.float(0.008)), s.float(0.0), s.float(1.0)), s.fmul(glint_str, s.float(68.0)))
    var grain = s.fmul(s.fsub(ign, s.float(0.5)), s.fmul(hal_str, s.float(7.5)))
    s.store(cur_r, s.nmax(s.float(0.0), s.fadd(s.fma(glint_e, s.float(1.00), cr2), grain)))
    s.store(cur_g, s.nmax(s.float(0.0), s.fadd(s.fma(glint_e, s.float(0.95), cg2), grain)))
    s.store(cur_b, s.nmax(s.float(0.0), s.fadd(s.fma(glint_e, s.float(0.82), cb2), s.fmul(grain, s.float(0.85)))))
    s.store(bl_r, s.fmul(s.load_f(bl_r), s.fma(hal_str, s.float(0.65), s.float(1.0))))
    s.store(bl_g, s.fmul(s.load_f(bl_g), s.fma(hal_str, s.float(0.38), s.float(1.0))))
    s.store(bl_b, s.fmul(s.load_f(bl_b), s.fma(hal_str, s.float(0.12), s.float(1.0))))
    s.end_if()

    # 4d. V11 Dynamic Weather Precipitation & Canopy Streaks (EFFECT_WEATHER_PRECIP)
    s.begin_if(s.fgt(s.load_f(wx_sky_e), s.float(0.01)))
    var depth_wx = s.fdiv(s.float(1.0), s.nmax(z_c, s.float(0.0001)))
    var dep_ok = s.select_f(s.fgt(depth_wx, s.float(5.5)), s.float(1.0), s.float(0.25))
    var str_e = s.fmul(s.load_f(wx_sky_e), dep_ok)
    s.store(cur_r, s.fma(str_e, s.float(1.00), s.load_f(cur_r)))
    s.store(cur_g, s.fma(str_e, s.float(0.97), s.load_f(cur_g)))
    s.store(cur_b, s.fma(str_e, s.float(0.90), s.load_f(cur_b)))
    s.end_if()

    var ae_f = s.load_f(auto_exp)
    var hr_post = s.fmul(s.fma(s.load_f(cur_r), ao_f, s.load_f(bl_r)), ae_f)
    var hg_post = s.fmul(s.fma(s.load_f(cur_g), ao_f, s.load_f(bl_g)), ae_f)
    var hb_post = s.fmul(s.fma(s.load_f(cur_b), ao_f, s.load_f(bl_b)), ae_f)
    s.store(out_pixel, _tonemap_agx_sdr(s, res, hr_post, hg_post, hb_post, ign))
    s.end_if()

    s.store_word(target, word, s.load_u(out_pixel))
    return s.finish()


def _rgb_to_ycocg(mut s: SpirvModule, col: UInt32) -> Vec3Ids:
    var r = s.float_from_uint(channel(s, col, 16))
    var g = s.float_from_uint(channel(s, col, 8))
    var b = s.float_from_uint(channel(s, col, 0))
    var co = s.fmul(s.fsub(r, b), s.float(0.5))
    var tmp = s.fadd(b, co)
    var cg = s.fmul(s.fsub(g, tmp), s.float(0.5))
    var y = s.fadd(tmp, cg)
    return Vec3Ids(y, co, cg)


def _sample_packed_bilinear_rgb(
    mut s: SpirvModule,
    buf: UInt32,
    plane_off: UInt32,
    row_w: UInt32,
    max_x: UInt32,
    max_y: UInt32,
    fx: UInt32,
    fy: UInt32,
) -> UInt32:
    var cx = s.nmax(fx, s.float(0.0))
    var cy = s.nmax(fy, s.float(0.0))
    var x0 = s.umin(s.saturate_uint(cx), max_x)
    var x1 = s.umin(s.iadd(x0, s.uint(1)), max_x)
    var y0 = s.umin(s.saturate_uint(cy), max_y)
    var y1 = s.umin(s.iadd(y0, s.uint(1)), max_y)
    var wx = s.umin(s.saturate_uint(s.fmul(s.nmax(s.fsub(cx, s.float_from_uint(x0)), s.float(0.0)), s.float(256.0))), s.uint(256))
    var wy = s.umin(s.saturate_uint(s.fmul(s.nmax(s.fsub(cy, s.float_from_uint(y0)), s.float(0.0)), s.float(256.0))), s.uint(256))
    var r0 = s.iadd(plane_off, s.imul(y0, row_w))
    var r1 = s.iadd(plane_off, s.imul(y1, row_w))
    var c00 = s.load_word(buf, s.iadd(r0, x0))
    var c10 = s.load_word(buf, s.iadd(r0, x1))
    var c01 = s.load_word(buf, s.iadd(r1, x0))
    var c11 = s.load_word(buf, s.iadd(r1, x1))
    return lerp_packed(s, lerp_packed(s, c00, c10, wx), lerp_packed(s, c01, c11, wx), wy)


def taa_compose_shader(capture: Bool) -> List[UInt32]:
    """Halo-Free CAS -> 3x3 YCoCg Neighborhood-Clamped TAA -> 6-Tap Depth-Weighted Velocity Motion Blur -> HUD."""
    var s = SpirvModule(GROUP_WIDTH, GROUP_HEIGHT, TAA_PUSH_WORDS)
    var res = Resources(s)
    var target = s.buffer(BIND_TARGET)
    var hud_image = s.buffer(BIND_HUD)
    var output = s.buffer(BIND_BACKDROP if capture else BIND_SHOWN)
    var column = s.invocation_axis(0)
    var line = s.invocation_axis(1)
    var width_w = s.push_word(TAA_PUSH_WIDTH)
    var height_w = s.push_word(TAA_PUSH_HEIGHT)
    var row_w = s.push_word(TAA_PUSH_ROW)
    s.return_if(s.uge(column, width_w))
    s.return_if(s.uge(line, height_w))

    var one_u = s.uint(1)
    var zero_u = s.uint(0)
    var max_px_x = s.isub(width_w, one_u)
    var max_px_y = s.isub(height_w, one_u)
    var word = s.iadd(s.imul(line, row_w), column)
    var scene_off = s.push_word(TAA_PUSH_SCENE)
    var plane_stride = s.push_word(TAA_PUSH_PLANE_STRIDE)
    var depth_plane = s.imul(plane_stride, s.uint(3))
    var width_f = s.float_from_uint(width_w)
    var height_f = s.float_from_uint(height_w)
    var x_f = s.float_from_uint(column)
    var y_f = s.float_from_uint(line)

    # V11 Zero-Copy Sub-Native GPU TSR Reconstruction when tsr_rw < width_w (1280x720 -> 1920x1080 in VRAM!)
    var tsr_rw_raw = constant_word(s, res, CONSTANT_V11_POST + 60)
    var tsr_rh_raw = constant_word(s, res, CONSTANT_V11_POST + 64)
    var subnative_tsr = s.logical_and(
        s.logical_and(s.ult(one_u, tsr_rw_raw), s.ult(one_u, tsr_rh_raw)),
        s.logical_or(s.ult(tsr_rw_raw, width_w), s.ult(tsr_rh_raw, height_w)),
    )
    var src_w_u = s.select_u(subnative_tsr, tsr_rw_raw, width_w)
    var src_h_u = s.select_u(subnative_tsr, tsr_rh_raw, height_w)
    var max_src_x = s.isub(src_w_u, one_u)
    var max_src_y = s.isub(src_h_u, one_u)
    var sc_tx = s.fdiv(s.float_from_uint(src_w_u), s.nmax(width_f, s.float(1.0)))
    var sc_ty = s.fdiv(s.float_from_uint(src_h_u), s.nmax(height_f, s.float(1.0)))
    var u_tsr = s.fsub(s.fmul(s.fadd(x_f, s.float(0.5)), sc_tx), s.float(0.5))
    var v_tsr = s.fsub(s.fmul(s.fadd(y_f, s.float(0.5)), sc_ty), s.float(0.5))
    var src_col_u = s.umin(s.saturate_uint(s.fadd(u_tsr, s.float(0.5))), max_src_x)
    var src_lin_u = s.umin(s.saturate_uint(s.fadd(v_tsr, s.float(0.5))), max_src_y)
    var src_word = s.iadd(s.imul(src_lin_u, row_w), src_col_u)

    # 1. Load 3x3 neighborhood for Halo-Free CAS / 5-Tap Catmull-Rom TSR + YCoCg TAA AABB
    var ym1 = s.smax(s.isub(src_lin_u, one_u), zero_u)
    var yp1 = s.smin(s.iadd(src_lin_u, one_u), max_src_y)
    var xm1 = s.smax(s.isub(src_col_u, one_u), zero_u)
    var xp1 = s.smin(s.iadd(src_col_u, one_u), max_src_x)

    var r_m1 = s.iadd(scene_off, s.imul(ym1, row_w))
    var r_0 = s.iadd(scene_off, s.imul(src_lin_u, row_w))
    var r_p1 = s.iadd(scene_off, s.imul(yp1, row_w))

    var px_nw = s.load_word(target, s.iadd(r_m1, xm1))
    var px_n0 = s.load_word(target, s.iadd(r_m1, src_col_u))
    var px_ne = s.load_word(target, s.iadd(r_m1, xp1))
    var px_w0 = s.load_word(target, s.iadd(r_0, xm1))
    var px_c0 = s.load_word(target, s.iadd(r_0, src_col_u))
    var px_e0 = s.load_word(target, s.iadd(r_0, xp1))
    var px_sw = s.load_word(target, s.iadd(r_p1, xm1))
    var px_s0 = s.load_word(target, s.iadd(r_p1, src_col_u))
    var px_se = s.load_word(target, s.iadd(r_p1, xp1))

    var px_c_v = s.uvar(px_c0)
    var px_n_v = s.uvar(px_n0)
    var px_s_v = s.uvar(px_s0)
    var px_w_v = s.uvar(px_w0)
    var px_e_v = s.uvar(px_e0)
    s.begin_if(subnative_tsr)
    s.store(px_c_v, _sample_packed_bilinear_rgb(s, target, scene_off, row_w, max_src_x, max_src_y, u_tsr, v_tsr))
    s.store(px_n_v, _sample_packed_bilinear_rgb(s, target, scene_off, row_w, max_src_x, max_src_y, u_tsr, s.fsub(v_tsr, s.float(1.0))))
    s.store(px_s_v, _sample_packed_bilinear_rgb(s, target, scene_off, row_w, max_src_x, max_src_y, u_tsr, s.fadd(v_tsr, s.float(1.0))))
    s.store(px_w_v, _sample_packed_bilinear_rgb(s, target, scene_off, row_w, max_src_x, max_src_y, s.fsub(u_tsr, s.float(1.0)), v_tsr))
    s.store(px_e_v, _sample_packed_bilinear_rgb(s, target, scene_off, row_w, max_src_x, max_src_y, s.fadd(u_tsr, s.float(1.0)), v_tsr))
    s.end_if()
    var px_c = s.load_u(px_c_v)
    var px_n = s.load_u(px_n_v)
    var px_s = s.load_u(px_s_v)
    var px_w = s.load_u(px_w_v)
    var px_e = s.load_u(px_e_v)

    var lc = luma(s, px_c)
    var ln = luma(s, px_n)
    var ls = luma(s, px_s)
    var lw = luma(s, px_w)
    var le = luma(s, px_e)
    var max_l = s.umax(s.umax(s.umax(s.umax(lc, ln), ls), lw), le)
    var min_l = s.umin(s.umin(s.umin(s.umin(lc, ln), ls), lw), le)
    var head = s.umin(min_l, s.isub(s.uint(255), max_l))
    var tsr_sharp = constant_float(s, res, CONSTANT_V10_POST + 24)
    var cas_gain_nat = s.nclamp(
        s.fmul(s.fdiv(s.float_from_uint(head), s.nmax(s.float_from_uint(max_l), s.float(24.0))), s.float(0.40)),
        s.float(0.0),
        s.float(0.26),
    )
    var cas_gain = s.select_f(subnative_tsr, s.fma(tsr_sharp, s.float(0.45), s.float(0.18)), cas_gain_nat)
    var cas_packed = s.uvar(s.uint(0xFF000000))
    for which in range(3):
        var sh = 16 - which * 8
        var cc_u = channel(s, px_c, sh)
        var cn_u = channel(s, px_n, sh)
        var cs_u = channel(s, px_s, sh)
        var cw_u = channel(s, px_w, sh)
        var ce_u = channel(s, px_e, sh)
        var lo_f = s.float_from_uint(s.umin(s.umin(s.umin(s.umin(cc_u, cn_u), cs_u), cw_u), ce_u))
        var hi_f = s.float_from_uint(s.umax(s.umax(s.umax(s.umax(cc_u, cn_u), cs_u), cw_u), ce_u))
        var cc_f = s.float_from_uint(cc_u)
        var avg4 = s.fmul(
            s.fadd(s.fadd(s.float_from_uint(cn_u), s.float_from_uint(cs_u)), s.fadd(s.float_from_uint(cw_u), s.float_from_uint(ce_u))),
            s.float(0.25),
        )
        var crisp = float_to_byte(s, s.nclamp(s.fma(s.fsub(cc_f, avg4), cas_gain, cc_f), lo_f, hi_f))
        s.store(cas_packed, s.bit_or(s.load_u(cas_packed), s.shl(crisp, s.uint(UInt32(sh)))))

    var resolved = s.uvar(s.load_u(cas_packed))

    # 2. Exact 3D Camera Reprojection Velocity Vector (mapped to native output resolution!)
    var z_c = s.load_float(target, s.iadd(depth_plane, src_word))
    var depth = s.fdiv(s.float(1.0), s.nmax(z_c, s.float(0.0001)))
    var camera = constant_vector(s, res, CONSTANT_CAMERA)
    var right = constant_vector(s, res, CONSTANT_CAMERA + 12)
    var up = constant_vector(s, res, CONSTANT_CAMERA + 24)
    var forward = constant_vector(s, res, CONSTANT_CAMERA + 36)
    var inv_res_sc = s.fdiv(width_f, s.nmax(s.float_from_uint(src_w_u), s.float(1.0)))
    var half_width = s.fmul(width_f, s.float(0.5))
    var half_height = s.fmul(height_f, s.float(0.5))
    var inverse_focal = s.fdiv(constant_float(s, res, CONSTANT_CAMERA + 56), inv_res_sc)

    var px_cx = s.fadd(x_f, s.float(0.5))
    var px_cy = s.fadd(y_f, s.float(0.5))
    var vx_c = s.fmul(s.fsub(px_cx, half_width), inverse_focal)
    var vy_c = s.fmul(s.fsub(half_height, px_cy), inverse_focal)
    var world_p = Vec3Ids(
        s.fma(s.fma(up.x, vy_c, s.fma(right.x, vx_c, forward.x)), depth, camera.x),
        s.fma(s.fma(up.y, vy_c, s.fma(right.y, vx_c, forward.y)), depth, camera.y),
        s.fma(s.fma(up.z, vy_c, s.fma(right.z, vx_c, forward.z)), depth, camera.z),
    )

    var prev_cam = constant_vector(s, res, CONSTANT_PREV_CAMERA)
    var prev_right = constant_vector(s, res, CONSTANT_PREV_CAMERA + 12)
    var prev_up = constant_vector(s, res, CONSTANT_PREV_CAMERA + 24)
    var prev_fwd = constant_vector(s, res, CONSTANT_PREV_CAMERA + 36)
    var prev_focal = constant_float(s, res, CONSTANT_PREV_CAMERA + 48)
    var taa_blend = constant_float(s, res, CONSTANT_PREV_CAMERA + 52)
    var mblur_scale = constant_float(s, res, CONSTANT_PREV_CAMERA + 56)
    var parity = constant_word(s, res, CONSTANT_PREV_CAMERA + 72)

    var rel_prev = Vec3Ids(
        s.fsub(world_p.x, prev_cam.x),
        s.fsub(world_p.y, prev_cam.y),
        s.fsub(world_p.z, prev_cam.z),
    )
    var prev_vz = s.nmax(dot3(s, rel_prev, prev_fwd), s.float(0.1))
    var inv_pvz = s.fdiv(s.float(1.0), prev_vz)
    var cam_prev_sx = s.fma(s.fmul(dot3(s, rel_prev, prev_right), prev_focal), inv_pvz, half_width)
    var cam_prev_sy = s.fsub(half_height, s.fmul(s.fmul(dot3(s, rel_prev, prev_up), prev_focal), inv_pvz))
    var cam_vel_x = s.fsub(px_cx, cam_prev_sx)
    var cam_vel_y = s.fsub(px_cy, cam_prev_sy)

    # Per-Object Motion Vector override (PLANE_VELOCITY = 11) scaled from render_w to native width_w
    var vel_plane = s.imul(plane_stride, s.uint(11))
    var vel_word = s.load_word(target, s.iadd(vel_plane, src_word))
    var has_obj_vel = s.ine(vel_word, zero_u)
    var obj_vx = s.fmul(s.fmul(s.fsub(s.float_from_uint(s.bit_and(vel_word, s.uint(65535))), s.float(32768.0)), s.float(0.015625)), inv_res_sc)
    var obj_vy = s.fmul(s.fmul(s.fsub(s.float_from_uint(s.shr(vel_word, s.uint(16))), s.float(32768.0)), s.float(0.015625)), inv_res_sc)
    var vel_x = s.select_f(has_obj_vel, obj_vx, cam_vel_x)
    var vel_y = s.select_f(has_obj_vel, obj_vy, cam_vel_y)
    var prev_sx = s.fsub(px_cx, vel_x)
    var prev_sy = s.fsub(px_cy, vel_y)

    # 3. Ping-Pong TAA Resolve with 3x3 YCoCg Rounded Neighborhood Clamping
    var effects = constant_word(s, res, CONSTANT_EFFECTS)
    var hist_read = s.imul(plane_stride, s.select_u(s.ieq(parity, zero_u), s.uint(9), s.uint(10)))
    var hist_write = s.imul(plane_stride, s.select_u(s.ieq(parity, zero_u), s.uint(10), s.uint(9)))
    var taa_valid = s.logical_and(
        s.logical_and(
            s.ine(s.bit_and(effects, s.uint(EFFECT_TAA)), zero_u),
            s.fgt(taa_blend, s.float(0.001)),
        ),
        s.logical_and(
            s.logical_and(s.fge(prev_sx, s.float(0.5)), s.flt(prev_sx, s.fsub(width_f, s.float(0.5)))),
            s.logical_and(s.fge(prev_sy, s.float(0.5)), s.flt(prev_sy, s.fsub(height_f, s.float(0.5)))),
        ),
    )
    s.begin_if(taa_valid)
    var hist_col = _sample_packed_bilinear_rgb(
        s, target, hist_read, row_w, max_px_x, max_px_y,
        s.fsub(prev_sx, s.float(0.5)), s.fsub(prev_sy, s.float(0.5)),
    )
    var yc_c = _rgb_to_ycocg(s, s.load_u(resolved))
    var yc_n = _rgb_to_ycocg(s, px_n)
    var yc_s = _rgb_to_ycocg(s, px_s)
    var yc_w = _rgb_to_ycocg(s, px_w)
    var yc_e = _rgb_to_ycocg(s, px_e)
    var yc_nw = _rgb_to_ycocg(s, px_nw)
    var yc_ne = _rgb_to_ycocg(s, px_ne)
    var yc_sw = _rgb_to_ycocg(s, px_sw)
    var yc_se = _rgb_to_ycocg(s, px_se)

    var min5_y = s.nmin(yc_c.x, s.nmin(s.nmin(yc_n.x, yc_s.x), s.nmin(yc_w.x, yc_e.x)))
    var max5_y = s.nmax(yc_c.x, s.nmax(s.nmax(yc_n.x, yc_s.x), s.nmax(yc_w.x, yc_e.x)))
    var min9_y = s.nmin(min5_y, s.nmin(s.nmin(yc_nw.x, yc_ne.x), s.nmin(yc_sw.x, yc_se.x)))
    var max9_y = s.nmax(max5_y, s.nmax(s.nmax(yc_nw.x, yc_ne.x), s.nmax(yc_sw.x, yc_se.x)))
    var lo_y = s.fmul(s.fadd(min5_y, min9_y), s.float(0.5))
    var hi_y = s.fmul(s.fadd(max5_y, max9_y), s.float(0.5))

    var min5_co = s.nmin(yc_c.y, s.nmin(s.nmin(yc_n.y, yc_s.y), s.nmin(yc_w.y, yc_e.y)))
    var max5_co = s.nmax(yc_c.y, s.nmax(s.nmax(yc_n.y, yc_s.y), s.nmax(yc_w.y, yc_e.y)))
    var min9_co = s.nmin(min5_co, s.nmin(s.nmin(yc_nw.y, yc_ne.y), s.nmin(yc_sw.y, yc_se.y)))
    var max9_co = s.nmax(max5_co, s.nmax(s.nmax(yc_nw.y, yc_ne.y), s.nmax(yc_sw.y, yc_se.y)))
    var lo_co = s.fmul(s.fadd(min5_co, min9_co), s.float(0.5))
    var hi_co = s.fmul(s.fadd(max5_co, max9_co), s.float(0.5))

    var min5_cg = s.nmin(yc_c.z, s.nmin(s.nmin(yc_n.z, yc_s.z), s.nmin(yc_w.z, yc_e.z)))
    var max5_cg = s.nmax(yc_c.z, s.nmax(s.nmax(yc_n.z, yc_s.z), s.nmax(yc_w.z, yc_e.z)))
    var min9_cg = s.nmin(min5_cg, s.nmin(s.nmin(yc_nw.z, yc_ne.z), s.nmin(yc_sw.z, yc_se.z)))
    var max9_cg = s.nmax(max5_cg, s.nmax(s.nmax(yc_nw.z, yc_ne.z), s.nmax(yc_sw.z, yc_se.z)))
    var lo_cg = s.fmul(s.fadd(min5_cg, min9_cg), s.float(0.5))
    var hi_cg = s.fmul(s.fadd(max5_cg, max9_cg), s.float(0.5))

    var yc_h = _rgb_to_ycocg(s, hist_col)
    var cl_y = s.nclamp(yc_h.x, lo_y, hi_y)
    var cl_co = s.nclamp(yc_h.y, lo_co, hi_co)
    var cl_cg = s.nclamp(yc_h.z, lo_cg, hi_cg)
    var tmp_h = s.fsub(cl_y, cl_cg)
    var hr_cl = s.nclamp(s.fadd(tmp_h, cl_co), s.float(0.0), s.float(255.0))
    var hg_cl = s.nclamp(s.fadd(cl_y, cl_cg), s.float(0.0), s.float(255.0))
    var hb_cl = s.nclamp(s.fsub(tmp_h, cl_co), s.float(0.0), s.float(255.0))

    var speed = s.fsqrt(s.fma(vel_y, vel_y, s.fmul(vel_x, vel_x)))
    var eff_blend = s.fdiv(taa_blend, s.fma(speed, s.float(0.09), s.float(1.0)))
    var cur_p = s.load_u(resolved)
    var tr = float_to_byte(s, s.fma(s.fsub(hr_cl, s.float_from_uint(channel(s, cur_p, 16))), eff_blend, s.float_from_uint(channel(s, cur_p, 16))))
    var tg = float_to_byte(s, s.fma(s.fsub(hg_cl, s.float_from_uint(channel(s, cur_p, 8))), eff_blend, s.float_from_uint(channel(s, cur_p, 8))))
    var tb = float_to_byte(s, s.fma(s.fsub(hb_cl, s.float_from_uint(channel(s, cur_p, 0))), eff_blend, s.float_from_uint(channel(s, cur_p, 0))))
    s.store(resolved, s.bit_or(s.uint(0xFF000000), s.bit_or(s.shl(tr, s.uint(16)), s.bit_or(s.shl(tg, s.uint(8)), tb))))
    s.end_if()

    var taa_px = s.load_u(resolved)
    if not capture:
        s.store_word(target, s.iadd(hist_write, word), taa_px)

    # 4. 6-Tap Depth-Weighted Directional Velocity Motion Blur (full strength on dynamic object vectors)
    var final_scene = s.uvar(taa_px)
    var ign_post = ign_noise(s, x_f, y_f)
    var nx_rad = s.fdiv(s.fsub(px_cx, half_width), s.nmax(half_width, s.float(1.0)))
    var ny_rad = s.fdiv(s.fsub(px_cy, half_height), s.nmax(half_height, s.float(1.0)))
    var cam_rad_w = s.fma(s.nclamp(s.fma(ny_rad, ny_rad, s.fmul(nx_rad, nx_rad)), s.float(0.0), s.float(1.0)), s.float(0.75), s.float(0.25))
    var rad_w = s.select_f(has_obj_vel, s.float(1.0), cam_rad_w)
    var bvx_raw = s.fmul(vel_x, s.fmul(mblur_scale, rad_w))
    var bvy_raw = s.fmul(vel_y, s.fmul(mblur_scale, rad_w))
    var blen = s.fsqrt(s.fma(bvy_raw, bvy_raw, s.fmul(bvx_raw, bvx_raw)))
    s.begin_if(s.logical_and(s.fgt(mblur_scale, s.float(0.01)), s.fgt(blen, s.float(1.35))))
    var cap_scale = s.fdiv(s.nmin(blen, s.float(16.0)), blen)
    var bvx = s.fmul(bvx_raw, cap_scale)
    var bvy = s.fmul(bvy_raw, cap_scale)
    var ign_mb = s.fmul(s.fsub(ign_post, s.float(0.5)), s.float(0.10))
    var sum_r = s.fvar(s.float_from_uint(channel(s, taa_px, 16)))
    var sum_g = s.fvar(s.float_from_uint(channel(s, taa_px, 8)))
    var sum_b = s.fvar(s.float_from_uint(channel(s, taa_px, 0)))
    var sum_w = s.fvar(s.float(1.0))

    var mb_offs = List[Float32]()
    mb_offs.append(-0.45)
    mb_offs.append(-0.27)
    mb_offs.append(-0.09)
    mb_offs.append(0.09)
    mb_offs.append(0.27)
    mb_offs.append(0.45)
    for tap in range(6):
        var t_mb = s.fadd(s.float(mb_offs[tap]), ign_mb)
        var sx_m = s.nclamp(s.fmul(s.fma(bvx, t_mb, x_f), sc_tx), s.float(0.0), s.float_from_uint(max_src_x))
        var sy_m = s.nclamp(s.fmul(s.fma(bvy, t_mb, y_f), sc_ty), s.float(0.0), s.float_from_uint(max_src_y))
        var m_idx = s.iadd(s.imul(s.saturate_uint(sy_m), row_w), s.saturate_uint(sx_m))
        var tap_iz = s.load_float(target, s.iadd(depth_plane, m_idx))
        var w_tap = s.select_f(s.flt(s.fmul(z_c, s.float(1.30)), tap_iz), s.float(0.15), s.float(0.85))
        var tap_c = s.load_word(target, s.iadd(scene_off, m_idx))
        s.store(sum_r, s.fma(s.float_from_uint(channel(s, tap_c, 16)), w_tap, s.load_f(sum_r)))
        s.store(sum_g, s.fma(s.float_from_uint(channel(s, tap_c, 8)), w_tap, s.load_f(sum_g)))
        s.store(sum_b, s.fma(s.float_from_uint(channel(s, tap_c, 0)), w_tap, s.load_f(sum_b)))
        s.store(sum_w, s.fadd(s.load_f(sum_w), w_tap))

    var inv_sw = s.fdiv(s.float(1.0), s.load_f(sum_w))
    var mbr = float_to_byte(s, s.fmul(s.load_f(sum_r), inv_sw))
    var mbg = float_to_byte(s, s.fmul(s.load_f(sum_g), inv_sw))
    var mbb = float_to_byte(s, s.fmul(s.load_f(sum_b), inv_sw))
    s.store(final_scene, s.bit_or(s.uint(0xFF000000), s.bit_or(s.shl(mbr, s.uint(16)), s.bit_or(s.shl(mbg, s.uint(8)), mbb))))
    s.end_if()

    # 5. Thin-Lens Circle-of-Confusion (CoC) 8-Tap Vogel Golden-Angle Bokeh Depth-of-Field (DoF)
    s.begin_if(s.ine(s.bit_and(effects, s.uint(EFFECT_DOF)), zero_u))
    var focus_d = constant_float(s, res, CONSTANT_V7_POST + 4)
    var aperture = constant_float(s, res, CONSTANT_V7_POST + 8)
    var max_coc = constant_float(s, res, CONSTANT_V7_POST + 12)
    var coc = s.nclamp(
        s.fdiv(s.fmul(aperture, s.fabs(s.fsub(depth, focus_d))), s.nmax(depth, s.float(0.5))),
        s.float(0.0),
        max_coc,
    )
    s.begin_if(s.fgt(coc, s.float(0.65)))
    var base_dof = s.load_u(final_scene)
    var dof_r = s.fvar(s.float_from_uint(channel(s, base_dof, 16)))
    var dof_g = s.fvar(s.float_from_uint(channel(s, base_dof, 8)))
    var dof_b = s.fvar(s.float_from_uint(channel(s, base_dof, 0)))
    var dof_w = s.fvar(s.float(1.0))
    var s_dof = _parabolic_sin(s, ign_post)
    var c_dof = _parabolic_sin(s, s.fadd(ign_post, s.float(0.25)))
    var c_coc = s.fmul(c_dof, coc)
    var s_coc = s.fmul(s_dof, coc)

    var vx_taps = List[Float32]()
    var vy_taps = List[Float32]()
    vx_taps.append(0.2500)
    vy_taps.append(0.0000)
    vx_taps.append(-0.3191)
    vy_taps.append(0.2923)
    vx_taps.append(0.0501)
    vy_taps.append(-0.5568)
    vx_taps.append(0.4150)
    vy_taps.append(0.5293)
    vx_taps.append(-0.7330)
    vy_taps.append(-0.1583)
    vx_taps.append(0.6657)
    vy_taps.append(-0.4945)
    vx_taps.append(-0.2224)
    vy_taps.append(0.8752)
    vx_taps.append(-0.3910)
    vy_taps.append(-0.8861)

    for dtap in range(8):
        var ox_d = s.float(vx_taps[dtap])
        var oy_d = s.float(vy_taps[dtap])
        var rx_d = s.fsub(s.fmul(ox_d, c_coc), s.fmul(oy_d, s_coc))
        var ry_d = s.fadd(s.fmul(ox_d, s_coc), s.fmul(oy_d, c_coc))
        var sx_d = s.nclamp(s.fmul(s.fadd(x_f, rx_d), sc_tx), s.float(0.0), s.float_from_uint(max_src_x))
        var sy_d = s.nclamp(s.fmul(s.fadd(y_f, ry_d), sc_ty), s.float(0.0), s.float_from_uint(max_src_y))
        var d_idx = s.iadd(s.imul(s.saturate_uint(sy_d), row_w), s.saturate_uint(sx_d))
        var t_iz = s.load_float(target, s.iadd(depth_plane, d_idx))
        var t_dep = s.fdiv(s.float(1.0), s.nmax(t_iz, s.float(0.0001)))
        var t_coc = s.nclamp(
            s.fdiv(s.fmul(aperture, s.fabs(s.fsub(t_dep, focus_d))), s.nmax(t_dep, s.float(0.5))),
            s.float(0.0),
            max_coc,
        )
        var fg_guard = s.select_f(
            s.flt(s.fadd(t_dep, s.float(0.5)), depth),
            s.nclamp(s.fmul(t_coc, s.float(0.5)), s.float(0.05), s.float(1.0)),
            s.float(1.0),
        )
        var tap_col = s.load_word(target, s.iadd(scene_off, d_idx))
        s.store(dof_r, s.fma(s.float_from_uint(channel(s, tap_col, 16)), fg_guard, s.load_f(dof_r)))
        s.store(dof_g, s.fma(s.float_from_uint(channel(s, tap_col, 8)), fg_guard, s.load_f(dof_g)))
        s.store(dof_b, s.fma(s.float_from_uint(channel(s, tap_col, 0)), fg_guard, s.load_f(dof_b)))
        s.store(dof_w, s.fadd(s.load_f(dof_w), fg_guard))

    var inv_dw = s.fdiv(s.float(1.0), s.load_f(dof_w))
    var dfr = float_to_byte(s, s.fmul(s.load_f(dof_r), inv_dw))
    var dfg = float_to_byte(s, s.fmul(s.load_f(dof_g), inv_dw))
    var dfb = float_to_byte(s, s.fmul(s.load_f(dof_b), inv_dw))
    s.store(final_scene, s.bit_or(s.uint(0xFF000000), s.bit_or(s.shl(dfr, s.uint(16)), s.bit_or(s.shl(dfg, s.uint(8)), dfb))))
    s.end_if()
    s.end_if()

    # 5. Premultiplied HUD Composite (completely unjittered & unblurred)
    var scene = s.load_u(final_scene)
    var hud = s.uint(0) if capture else s.load_word(hud_image, word)
    var through = s.isub(s.uint(255), s.shr(hud, s.uint(24)))
    var limit = s.uint(255)
    var result = s.uvar(s.uint(0xFF000000))
    for which in range(3):
        var shift = 16 - which * 8
        var value = divide_by_255(s, s.imul(through, channel(s, scene, shift)))
        value = s.smin(s.iadd(value, channel(s, hud, shift)), limit)
        s.store(result, s.bit_or(s.load_u(result), s.shl(value, s.uint(UInt32(shift)))))
    s.store_word(output, word, s.load_u(result))
    return s.finish()


def smooth_shader() -> List[UInt32]:
    """Soften high-contrast edges by blending across them, in the manner of FXAA."""
    var s = SpirvModule(GROUP_WIDTH, GROUP_HEIGHT, SMOOTH_PUSH_WORDS)
    var image = s.buffer(BIND_TARGET)
    var column = s.invocation_axis(0)
    var line = s.invocation_axis(1)
    var width = s.push_word(SMOOTH_PUSH_WIDTH)
    var height = s.push_word(SMOOTH_PUSH_HEIGHT)
    s.return_if(s.uge(column, width))
    s.return_if(s.uge(line, height))
    var row = s.push_word(SMOOTH_PUSH_ROW)
    var one = s.uint(1)
    var zero = s.uint(0)
    var above = s.smax(s.isub(line, one), zero)
    var below = s.smin(s.iadd(line, one), s.isub(height, one))
    var left = s.smax(s.isub(column, one), zero)
    var right = s.smin(s.iadd(column, one), s.isub(width, one))
    var word = s.iadd(s.imul(line, row), column)
    var center = s.load_word(image, word)
    var north = s.load_word(image, s.iadd(s.imul(above, row), column))
    var south = s.load_word(image, s.iadd(s.imul(below, row), column))
    var west = s.load_word(image, s.iadd(s.imul(line, row), left))
    var east = s.load_word(image, s.iadd(s.imul(line, row), right))
    var target = s.iadd(word, s.push_word(SMOOTH_PUSH_TARGET))
    var luma_center = luma(s, center)
    var luma_north = luma(s, north)
    var luma_south = luma(s, south)
    var luma_west = luma(s, west)
    var luma_east = luma(s, east)
    var highest = s.umax(s.umax(s.umax(s.umax(luma_center, luma_north), luma_south), luma_west), luma_east)
    var lowest = s.umin(s.umin(s.umin(s.umin(luma_center, luma_north), luma_south), luma_west), luma_east)
    var contrast = s.isub(highest, lowest)
    var threshold = s.umax(s.shr(highest, s.uint(3)), s.uint(16))
    var result = s.uvar(center)
    s.begin_if(s.ult(threshold, contrast))
    var vertical = s.iadd(luma_north, luma_south)
    var sideways = s.iadd(luma_west, luma_east)
    var twice = s.shl(luma_center, one)
    var change_vertical = absolute_int(s, s.isub(vertical, twice))
    var change_sideways = absolute_int(s, s.isub(sideways, twice))
    var blend_sideways = s.ult(change_vertical, change_sideways)
    var first = s.select_u(blend_sideways, west, north)
    var second = s.select_u(blend_sideways, east, south)
    var luma_first = s.select_u(blend_sideways, luma_west, luma_north)
    var luma_second = s.select_u(blend_sideways, luma_east, luma_south)
    var differ_first = absolute_int(s, s.isub(luma_first, luma_center))
    var differ_second = absolute_int(s, s.isub(luma_second, luma_center))
    var neighbour = s.select_u(s.ult(differ_first, differ_second), second, first)
    var average = s.shr(s.iadd(vertical, sideways), s.uint(2))
    var stands_out = s.float_from_uint(absolute_int(s, s.isub(average, luma_center)))
    var ratio = s.fmul(stands_out, s.fdiv(s.float(1.0), s.float_from_uint(contrast)))
    var amount = s.saturate_uint(s.fmul(ratio, s.float(96.0)))
    amount = s.umin(s.umax(amount, s.uint(48)), s.uint(128))
    s.store(result, lerp_packed(s, center, neighbour, amount))
    s.end_if()
    s.store_word(image, target, s.load_u(result))
    return s.finish()


def sky_lut_shader() -> List[UInt32]:
    """Hillaire 128x64 Sky-View LUT: spherical-shell Rayleigh + Cornette-Shanks Mie + Ozone Chappuis
    optical depth from sea level (0 km) to curved sunlit planetary limb & deep starlight orbit (500 km)."""
    var s = SpirvModule(8, 8, 1)
    var res = Resources(s)
    var cells = s.buffer(BIND_CELLS)
    var col = s.invocation_axis(0)
    var row = s.invocation_axis(1)
    var w_lut = s.uint(UInt32(SKY_LUT_W))
    var h_lut = s.uint(UInt32(SKY_LUT_H))
    s.return_if(s.uge(col, w_lut))
    s.return_if(s.uge(row, h_lut))

    var u_n = s.fdiv(s.float_from_uint(col), s.float(Float32(SKY_LUT_W - 1)))
    var v_n = s.fdiv(s.float_from_uint(row), s.float(Float32(SKY_LUT_H - 1)))
    var cos_az = s.fma(u_n, s.float(2.0), s.float(-1.0))
    var sin_az = s.fsqrt(s.nmax(s.fsub(s.float(1.0), s.fmul(cos_az, cos_az)), s.float(0.0)))
    var sv = s.fma(v_n, s.float(2.0), s.float(-1.0))
    var dir_y = s.nclamp(s.fmul(sv, s.fabs(sv)), s.float(-1.0), s.float(1.0))
    var horiz_r = s.fsqrt(s.nmax(s.fsub(s.float(1.0), s.fmul(dir_y, dir_y)), s.float(0.0)))

    var to_key = constant_vector(s, res, CONSTANT_KEY)
    var key_color = constant_vector(s, res, CONSTANT_KEY + 12)
    var env_sky = constant_vector(s, res, CONSTANT_ENV_SKY)
    var env_ground = constant_vector(s, res, CONSTANT_ENV_GROUND)
    var env_horizon = constant_vector(s, res, CONSTANT_ENV_HORIZON)

    var sun_h = s.fsqrt(s.nmax(s.fma(to_key.z, to_key.z, s.fmul(to_key.x, to_key.x)), s.float(0.0)))
    var sun_loc = Vec3Ids(sun_h, to_key.y, s.float(0.0))
    var ray_loc = Vec3Ids(s.fmul(horiz_r, cos_az), dir_y, s.fmul(horiz_r, sin_az))
    var mu = dot3(s, ray_loc, sun_loc)

    var alt_km = s.nmax(constant_float(s, res, CONSTANT_V8_ATMOS), s.float(0.02))
    var r_planet = s.nmax(constant_float(s, res, CONSTANT_V8_ATMOS + 4), s.float(100.0))
    var atmos_h = s.nmax(constant_float(s, res, CONSTANT_V8_ATMOS + 8), s.float(12.0))
    var rayleigh_m = s.nmax(constant_float(s, res, CONSTANT_V8_ATMOS + 12), s.float(0.0))
    var mie_m = s.nmax(constant_float(s, res, CONSTANT_V8_ATMOS + 16), s.float(0.0))
    var ozone_m = s.nmax(constant_float(s, res, CONSTANT_V8_ATMOS + 20), s.float(0.0))

    var r_cam = s.fadd(r_planet, alt_km)
    var r_top = s.fadd(r_planet, atmos_h)
    var b_dot = s.fmul(r_cam, dir_y)
    var c_top = s.fsub(s.fmul(r_cam, r_cam), s.fmul(r_top, r_top))
    var disc_top = s.fsub(s.fmul(b_dot, b_dot), c_top)
    var c_pl = s.fsub(s.fmul(r_cam, r_cam), s.fmul(r_planet, r_planet))
    var disc_pl = s.fsub(s.fmul(b_dot, b_dot), c_pl)
    var t_pl_near = s.fsub(s.fsub(s.float(0.0), b_dot), s.fsqrt(s.nmax(disc_pl, s.float(0.0))))
    var hits_planet = s.logical_and(s.fgt(disc_pl, s.float(0.0)), s.fgt(t_pl_near, s.float(0.0)))

    # Rayleigh + Cornette-Shanks Mie (g = 0.78) phase functions
    var phase_r = s.fmul(s.float(0.0596831), s.fma(mu, mu, s.float(1.0)))
    var g_den = s.fsub(s.float(1.6084), s.fmul(s.float(1.56), mu))
    var phase_m = s.nmin(
        s.fdiv(s.float(0.0454), s.nmax(s.fmul(g_den, s.fsqrt(g_den)), s.float(0.015))),
        s.float(2.4),
    )

    var rad_r = s.fvar(s.float(0.0))
    var rad_g = s.fvar(s.float(0.0))
    var rad_b = s.fvar(s.float(0.0))
    var tr_r = s.fvar(s.float(1.0))
    var tr_g = s.fvar(s.float(1.0))
    var tr_b = s.fvar(s.float(1.0))

    s.begin_if(s.fgt(disc_top, s.float(0.0)))
    var sq_top = s.fsqrt(disc_top)
    var t_start = s.nmax(s.fsub(s.fsub(s.float(0.0), b_dot), sq_top), s.float(0.0))
    var t_exit = s.nmax(s.fsub(sq_top, b_dot), s.float(0.0))
    var t_end = s.select_f(hits_planet, s.nmin(t_exit, t_pl_near), t_exit)
    var seg_len = s.nmax(s.fsub(t_end, t_start), s.float(0.0))
    var dt = s.fmul(seg_len, s.float(0.1666667))

    s.begin_if(s.fgt(seg_len, s.float(0.001)))
    var si_v = s.uvar(s.uint(0))
    s.begin_loop()
    var si = s.load_u(si_v)
    s.break_if(s.uge(si, s.uint(6)))
    s.store(si_v, s.iadd(si, s.uint(1)))
    var t_cur = s.fma(s.fadd(s.float_from_uint(si), s.float(0.5)), dt, t_start)
    var px = s.fmul(ray_loc.x, t_cur)
    var py = s.fma(ray_loc.y, t_cur, r_cam)
    var pz = s.fmul(ray_loc.z, t_cur)
    var p_vec = Vec3Ids(px, py, pz)
    var r_here = s.fsqrt(s.fadd(dot3(s, p_vec, p_vec), s.float(0.001)))
    var h_km = s.nmax(s.fsub(r_here, r_planet), s.float(0.0))

    var rho_r = s.fmul(s.exp2(s.fsub(s.float(0.0), s.fmul(h_km, s.float(0.180337)))), rayleigh_m)
    var rho_m = s.fmul(s.exp2(s.fsub(s.float(0.0), s.fmul(h_km, s.float(1.154156)))), mie_m)
    var rho_o = s.fmul(s.nmax(s.fsub(s.float(1.0), s.fmul(s.fabs(s.fsub(h_km, s.float(25.0))), s.float(0.066))), s.float(0.0)), ozone_m)

    # Calibrated golden-hour / 35mm film Rayleigh + Mie + Ozone Chappuis extinction (per km)
    var ext_r = s.fma(rho_o, s.float(0.0018), s.fma(rho_m, s.float(0.0048), s.fmul(rho_r, s.float(0.0068))))
    var ext_g = s.fma(rho_o, s.float(0.0030), s.fma(rho_m, s.float(0.0048), s.fmul(rho_r, s.float(0.0138))))
    var ext_b = s.fma(rho_o, s.float(0.0002), s.fma(rho_m, s.float(0.0048), s.fmul(rho_r, s.float(0.0245))))

    var step_tr_r = s.exp2(s.fsub(s.float(0.0), s.fmul(ext_r, s.fmul(dt, s.float(1.442695)))))
    var step_tr_g = s.exp2(s.fsub(s.float(0.0), s.fmul(ext_g, s.fmul(dt, s.float(1.442695)))))
    var step_tr_b = s.exp2(s.fsub(s.float(0.0), s.fmul(ext_b, s.fmul(dt, s.float(1.442695)))))

    var inv_rh = s.fdiv(s.float(1.0), r_here)
    var up_s = Vec3Ids(s.fmul(px, inv_rh), s.fmul(py, inv_rh), s.fmul(pz, inv_rh))
    var sun_zen = dot3(s, up_s, sun_loc)
    s.begin_if(s.fgt(sun_zen, s.float(-0.20)))
    var chap = s.fdiv(s.float(1.0), s.nmax(s.fadd(sun_zen, s.float(0.22)), s.float(0.05)))
    var st_r = s.exp2(s.fsub(s.float(0.0), s.fmul(s.fma(rho_r, s.float(0.078), s.fmul(rho_m, s.float(0.009))), chap)))
    var st_g = s.exp2(s.fsub(s.float(0.0), s.fmul(s.fma(rho_r, s.float(0.158), s.fma(rho_o, s.float(0.036), s.fmul(rho_m, s.float(0.009)))), chap)))
    var st_b = s.exp2(s.fsub(s.float(0.0), s.fmul(s.fma(rho_r, s.float(0.285), s.fmul(rho_m, s.float(0.009))), chap)))

    var sc_r = s.fma(rho_m, s.fmul( s.float(0.0044), phase_m), s.fmul(rho_r, s.fmul(s.float(0.0068), phase_r)))
    var sc_g = s.fma(rho_m, s.fmul( s.float(0.0044), phase_m), s.fmul(rho_r, s.fmul(s.float(0.0138), phase_r)))
    var sc_b = s.fma(rho_m, s.fmul( s.float(0.0044), phase_m), s.fmul(rho_r, s.fmul(s.float(0.0245), phase_r)))

    var sun_e = s.float(1380.0)
    var in_r = s.fma(env_horizon.x, s.fmul(rho_r, s.float(0.68)), s.fmul(s.fmul(sc_r, st_r), s.fmul(key_color.x, sun_e)))
    var in_g = s.fma(env_horizon.y, s.fmul(rho_r, s.float(0.68)), s.fmul(s.fmul(sc_g, st_g), s.fmul(key_color.y, sun_e)))
    var in_b = s.fma(env_horizon.z, s.fmul(rho_r, s.float(0.68)), s.fmul(s.fmul(sc_b, st_b), s.fmul(key_color.z, sun_e)))

    var int_r = s.fdiv(s.fsub(s.float(1.0), step_tr_r), s.nmax(ext_r, s.float(0.00001)))
    var int_g = s.fdiv(s.fsub(s.float(1.0), step_tr_g), s.nmax(ext_g, s.float(0.00001)))
    var int_b = s.fdiv(s.fsub(s.float(1.0), step_tr_b), s.nmax(ext_b, s.float(0.00001)))

    s.store(rad_r, s.fma(s.load_f(tr_r), s.fmul(in_r, int_r), s.load_f(rad_r)))
    s.store(rad_g, s.fma(s.load_f(tr_g), s.fmul(in_g, int_g), s.load_f(rad_g)))
    s.store(rad_b, s.fma(s.load_f(tr_b), s.fmul(in_b, int_b), s.load_f(rad_b)))
    s.end_if()

    s.store(tr_r, s.fmul(s.load_f(tr_r), step_tr_r))
    s.store(tr_g, s.fmul(s.load_f(tr_g), step_tr_g))
    s.store(tr_b, s.fmul(s.load_f(tr_b), step_tr_b))
    s.end_loop()
    s.end_if()
    s.end_if()

    # Curved sunlit planetary surface / ocean-cloud limb below orbital horizon vs deep-space starlight above
    s.begin_if(hits_planet)
    var hx_p = s.fmul(ray_loc.x, t_pl_near)
    var hy_p = s.fma(ray_loc.y, t_pl_near, r_cam)
    var hz_p = s.fmul(ray_loc.z, t_pl_near)
    var inv_rp = s.fdiv(s.float(1.0), r_planet)
    var n_pl = Vec3Ids(s.fmul(hx_p, inv_rp), s.fmul(hy_p, inv_rp), s.fmul(hz_p, inv_rp))
    var ndl_pl = s.nmax(dot3(s, n_pl, sun_loc), s.float(0.0))
    var surf_lit = s.fmul(s.fma(ndl_pl, s.float(1.35), s.float(0.10)), s.float(185.0))
    var pl_r = s.fmul(s.fma(env_horizon.x, s.float(0.32), s.fmul(env_ground.x, s.float(0.85))), surf_lit)
    var pl_g = s.fmul(s.fma(env_horizon.y, s.float(0.32), s.fmul(env_ground.y, s.float(0.92))), surf_lit)
    var pl_b = s.fmul(s.fma(env_horizon.z, s.float(0.32), s.fmul(env_ground.z, s.float(0.96))), surf_lit)
    s.store(rad_r, s.fma(pl_r, s.load_f(tr_r), s.load_f(rad_r)))
    s.store(rad_g, s.fma(pl_g, s.load_f(tr_g), s.load_f(rad_g)))
    s.store(rad_b, s.fma(pl_b, s.load_f(tr_b), s.load_f(rad_b)))
    s.store(tr_r, s.float(0.0))
    s.store(tr_g, s.float(0.0))
    s.store(tr_b, s.float(0.0))
    s.begin_else()
    var space_w = s.nclamp(s.fmul(alt_km, s.float(0.012)), s.float(0.15), s.float(1.0))
    var sp_scale = s.fmul(space_w, s.float(28.0))
    var sp_r = s.fmul(env_sky.x, sp_scale)
    var sp_g = s.fmul(env_sky.y, sp_scale)
    var sp_b = s.fmul(env_sky.z, sp_scale)
    s.store(rad_r, s.fma(sp_r, s.load_f(tr_r), s.load_f(rad_r)))
    s.store(rad_g, s.fma(sp_g, s.load_f(tr_g), s.load_f(rad_g)))
    s.store(rad_b, s.fma(sp_b, s.load_f(tr_b), s.load_f(rad_b)))
    s.end_if()

    var tr_avg = s.fmul(s.fadd(s.fadd(s.load_f(tr_r), s.load_f(tr_g)), s.load_f(tr_b)), s.float(0.3333333))
    var out_idx = s.iadd(
        s.uint(UInt32(SKY_LUT_OFFSET // 4)),
        s.shl(s.iadd(s.imul(row, w_lut), col), s.uint(2)),
    )
    s.store_word(cells, out_idx, s.bitcast_uint(s.load_f(rad_r)))
    s.store_word(cells, s.iadd(out_idx, s.uint(1)), s.bitcast_uint(s.load_f(rad_g)))
    s.store_word(cells, s.iadd(out_idx, s.uint(2)), s.bitcast_uint(s.load_f(rad_b)))
    s.store_word(cells, s.iadd(out_idx, s.uint(3)), s.bitcast_uint(tr_avg))
    return s.finish()


def froxel_integrate_shader() -> List[UInt32]:
    """64x36x16 Temporally Reprojected 3D Froxel Volumetric Scattering & Prefix Integration."""
    var s = SpirvModule(8, 4, FROXEL_PUSH_WORDS)
    var res = Resources(s)
    var cells = s.buffer(BIND_CELLS)
    var fx = s.invocation_axis(0)
    var fy = s.invocation_axis(1)
    var fx_max = s.uint(UInt32(FROXEL_X))
    var fy_max = s.uint(UInt32(FROXEL_Y))
    s.return_if(s.uge(fx, fx_max))
    s.return_if(s.uge(fy, fy_max))

    var read_off = s.push_word(FROXEL_PUSH_READ_OFF)
    var write_off = s.push_word(FROXEL_PUSH_WRITE_OFF)
    var integ_off = s.push_word(FROXEL_PUSH_INTEG_OFF)
    var frame_u = s.push_word(FROXEL_PUSH_FRAME)

    var camera = constant_vector(s, res, CONSTANT_CAMERA)
    var right = constant_vector(s, res, CONSTANT_CAMERA + 12)
    var up = constant_vector(s, res, CONSTANT_CAMERA + 24)
    var forward = constant_vector(s, res, CONSTANT_CAMERA + 36)
    var half_w = constant_float(s, res, CONSTANT_CAMERA + 48)
    var half_h = constant_float(s, res, CONSTANT_CAMERA + 52)
    var inv_focal = constant_float(s, res, CONSTANT_CAMERA + 56)

    var prev_cam = constant_vector(s, res, CONSTANT_PREV_CAMERA)
    var prev_right = constant_vector(s, res, CONSTANT_PREV_CAMERA + 12)
    var prev_up = constant_vector(s, res, CONSTANT_PREV_CAMERA + 24)
    var prev_fwd = constant_vector(s, res, CONSTANT_PREV_CAMERA + 36)
    var prev_focal = constant_float(s, res, CONSTANT_PREV_CAMERA + 48)

    var u_scr = s.fdiv(s.fadd(s.float_from_uint(fx), s.float(0.5)), s.float(Float32(FROXEL_X)))
    var v_scr = s.fdiv(s.fadd(s.float_from_uint(fy), s.float(0.5)), s.float(Float32(FROXEL_Y)))
    var px_x = s.fmul(u_scr, s.fmul(half_w, s.float(2.0)))
    var px_y = s.fmul(v_scr, s.fmul(half_h, s.float(2.0)))
    var vx = s.fmul(s.fsub(px_x, half_w), inv_focal)
    var vy = s.fmul(s.fsub(half_h, px_y), inv_focal)
    var ray = Vec3Ids(
        s.fma(up.x, vy, s.fma(right.x, vx, forward.x)),
        s.fma(up.y, vy, s.fma(right.y, vx, forward.y)),
        s.fma(up.z, vy, s.fma(right.z, vx, forward.z)),
    )
    var inv_rl = s.inverse_sqrt(s.fadd(dot3(s, ray, ray), s.float(0.000001)))
    var ray_dir = Vec3Ids(s.fmul(ray.x, inv_rl), s.fmul(ray.y, inv_rl), s.fmul(ray.z, inv_rl))

    var vol_fog = constant_vector(s, res, CONSTANT_VOL_POST)
    var height_fall = constant_float(s, res, CONSTANT_VOL_POST + 20)
    var vol_density = constant_float(s, res, CONSTANT_VOL_POST + 24)
    var to_key = constant_vector(s, res, CONSTANT_KEY)
    var key_color = constant_vector(s, res, CONSTANT_KEY + 12)
    var frox_max = s.nmax(constant_float(s, res, CONSTANT_V8_ATMOS + 24), s.float(10.0))
    var temp_blend = s.nclamp(constant_float(s, res, CONSTANT_V8_ATMOS + 28), s.float(0.0), s.float(0.90))

    var cos_th = dot3(s, ray_dir, to_key)
    var hg_den = s.fsub(s.float(1.4096), s.fmul(s.float(1.28), cos_th))
    var hg_phase = s.nmin(
        s.fdiv(s.float(0.04698), s.nmax(s.fmul(hg_den, s.fsqrt(hg_den)), s.float(0.02))),
        s.float(1.55),
    )
    var sun_r = s.fmul(key_color.x, s.fmul(hg_phase, s.float(195.0)))
    var sun_g = s.fmul(key_color.y, s.fmul(hg_phase, s.float(190.0)))
    var sun_b = s.fmul(key_color.z, s.fmul(hg_phase, s.float(172.0)))

    var light_cnt = s.umin(constant_word(s, res, CONSTANT_WORLD + 24), s.uint(8))
    var frame_phase = s.fmul(s.float_from_uint(s.bit_and(frame_u, s.uint(15))), s.float(0.618034))
    var ign_f = s.fract(s.fadd(ign_noise(s, s.float_from_uint(fx), s.float_from_uint(fy)), frame_phase))

    var cum_r = s.fvar(s.float(0.0))
    var cum_g = s.fvar(s.float(0.0))
    var cum_b = s.fvar(s.float(0.0))
    var cum_tr = s.fvar(s.float(1.0))

    var z_v = s.uvar(s.uint(0))
    s.begin_loop()
    var zi = s.load_u(z_v)
    s.break_if(s.uge(zi, s.uint(UInt32(FROXEL_Z))))
    s.store(z_v, s.iadd(zi, s.uint(1)))

    var zf0 = s.fmul(s.float_from_uint(zi), s.float(Float32(1.0 / Float64(FROXEL_Z))))
    var zf1 = s.fmul(s.float_from_uint(s.iadd(zi, s.uint(1))), s.float(Float32(1.0 / Float64(FROXEL_Z))))
    var d0 = s.fmul(frox_max, s.fmul(zf0, zf0))
    var d1 = s.fmul(frox_max, s.fmul(zf1, zf1))
    var dt = s.nmax(s.fsub(d1, d0), s.float(0.05))
    var d_jit = s.fma(s.fma(ign_f, s.float(0.6), s.float(0.2)), dt, d0)

    var pw = Vec3Ids(
        s.fma(ray_dir.x, d_jit, camera.x),
        s.fma(ray_dir.y, d_jit, camera.y),
        s.fma(ray_dir.z, d_jit, camera.z),
    )
    var alt_h = s.nmax(s.fadd(pw.y, s.float(6.0)), s.float(0.0))
    var rho_cur = s.fmul(s.exp2(s.fsub(s.float(0.0), s.fmul(alt_h, height_fall))), s.fmul(vol_density, s.float(0.0036)))
    var vis_s = _sample_volumetric_shadow(s, res, pw, to_key)

    var loc_r = s.fvar(s.float(0.0))
    var loc_g = s.fvar(s.float(0.0))
    var loc_b = s.fvar(s.float(0.0))
    var li = s.uvar(s.uint(0))
    s.begin_loop()
    var l_idx = s.load_u(li)
    s.break_if(s.uge(l_idx, light_cnt))
    s.store(li, s.iadd(l_idx, s.uint(1)))
    var lw = s.iadd(s.uint(UInt32(CONSTANT_LIGHTS // 4)), s.shl(l_idx, s.uint(3)))
    var dlx = s.fsub(light_float(s, res, lw, 0), pw.x)
    var dly = s.fsub(light_float(s, res, lw, 1), pw.y)
    var dlz = s.fsub(light_float(s, res, lw, 2), pw.z)
    var d2 = s.fadd(dot3(s, Vec3Ids(dlx, dly, dlz), Vec3Ids(dlx, dly, dlz)), s.float(1.5))
    var fall = s.nmax(s.fsub(s.float(1.0), s.fmul(light_float(s, res, lw, 7), d2)), s.float(0.0))
    var l_att = s.fmul(s.fmul(fall, fall), s.float(460.0))
    s.store(loc_r, s.fma(light_float(s, res, lw, 4), l_att, s.load_f(loc_r)))
    s.store(loc_g, s.fma(light_float(s, res, lw, 5), l_att, s.load_f(loc_g)))
    s.store(loc_b, s.fma(light_float(s, res, lw, 6), l_att, s.load_f(loc_b)))
    s.end_loop()

    var sr_cur = s.fmul(s.fadd(s.fma(vol_fog.x, s.float(0.62), s.fmul(sun_r, vis_s)), s.load_f(loc_r)), rho_cur)
    var sg_cur = s.fmul(s.fadd(s.fma(vol_fog.y, s.float(0.62), s.fmul(sun_g, vis_s)), s.load_f(loc_g)), rho_cur)
    var sb_cur = s.fmul(s.fadd(s.fma(vol_fog.z, s.float(0.62), s.fmul(sun_b, vis_s)), s.load_f(loc_b)), rho_cur)

    var sr_fin = s.fvar(sr_cur)
    var sg_fin = s.fvar(sg_cur)
    var sb_fin = s.fvar(sb_cur)
    var rho_fin = s.fvar(rho_cur)

    # Temporal 3D froxel reprojection into previous camera frustum
    var d_mid = s.fmul(s.fadd(d0, d1), s.float(0.5))
    var p_mid = Vec3Ids(
        s.fma(ray_dir.x, d_mid, s.fsub(camera.x, prev_cam.x)),
        s.fma(ray_dir.y, d_mid, s.fsub(camera.y, prev_cam.y)),
        s.fma(ray_dir.z, d_mid, s.fsub(camera.z, prev_cam.z)),
    )
    var prev_vz = dot3(s, p_mid, prev_fwd)
    s.begin_if(s.logical_and(s.fgt(temp_blend, s.float(0.001)), s.fgt(prev_vz, s.float(0.25))))
    var inv_pvz = s.fdiv(s.float(1.0), prev_vz)
    var prev_px = s.fma(s.fmul(dot3(s, p_mid, prev_right), prev_focal), inv_pvz, half_w)
    var prev_py = s.fsub(half_h, s.fmul(s.fmul(dot3(s, p_mid, prev_up), prev_focal), inv_pvz))
    var pu = s.fdiv(prev_px, s.nmax(s.fmul(half_w, s.float(2.0)), s.float(1.0)))
    var pv = s.fdiv(prev_py, s.nmax(s.fmul(half_h, s.float(2.0)), s.float(1.0)))
    var p_dist = s.fsqrt(s.fadd(dot3(s, p_mid, p_mid), s.float(0.001)))
    var pz_n = s.fsqrt(s.nclamp(s.fdiv(p_dist, frox_max), s.float(0.0), s.float(1.0)))
    var in_prev = s.logical_and(
        s.logical_and(s.fge(pu, s.float(0.0)), s.flt(pu, s.float(1.0))),
        s.logical_and(s.fge(pv, s.float(0.0)), s.flt(pv, s.float(1.0))),
    )
    s.begin_if(in_prev)
    var pfx = s.umin(s.saturate_uint(s.fmul(pu, s.float(Float32(FROXEL_X)))), s.uint(UInt32(FROXEL_X - 1)))
    var pfy = s.umin(s.saturate_uint(s.fmul(pv, s.float(Float32(FROXEL_Y)))), s.uint(UInt32(FROXEL_Y - 1)))
    var pfz = s.umin(s.saturate_uint(s.fmul(pz_n, s.float(Float32(FROXEL_Z)))), s.uint(UInt32(FROXEL_Z - 1)))
    var p_cell = s.iadd(
        read_off,
        s.shl(s.iadd(s.imul(s.iadd(s.imul(pfz, fy_max), pfy), fx_max), pfx), s.uint(2)),
    )
    var pr_sr = s.load_float(cells, p_cell)
    var pr_sg = s.load_float(cells, s.iadd(p_cell, s.uint(1)))
    var pr_sb = s.load_float(cells, s.iadd(p_cell, s.uint(2)))
    var pr_rh = s.load_float(cells, s.iadd(p_cell, s.uint(3)))
    s.store(sr_fin, s.fma(s.fsub(pr_sr, sr_cur), temp_blend, sr_cur))
    s.store(sg_fin, s.fma(s.fsub(pr_sg, sg_cur), temp_blend, sg_cur))
    s.store(sb_fin, s.fma(s.fsub(pr_sb, sb_cur), temp_blend, sb_cur))
    s.store(rho_fin, s.fma(s.fsub(pr_rh, rho_cur), temp_blend, rho_cur))
    s.end_if()
    s.end_if()

    var cell_rel = s.shl(s.iadd(s.imul(s.iadd(s.imul(zi, fy_max), fy), fx_max), fx), s.uint(2))
    var w_idx = s.iadd(write_off, cell_rel)
    var sr_b = s.load_f(sr_fin)
    var sg_b = s.load_f(sg_fin)
    var sb_b = s.load_f(sb_fin)
    var rh_b = s.load_f(rho_fin)
    s.store_word(cells, w_idx, s.bitcast_uint(sr_b))
    s.store_word(cells, s.iadd(w_idx, s.uint(1)), s.bitcast_uint(sg_b))
    s.store_word(cells, s.iadd(w_idx, s.uint(2)), s.bitcast_uint(sb_b))
    s.store_word(cells, s.iadd(w_idx, s.uint(3)), s.bitcast_uint(rh_b))

    var ext_sl = s.nmin(s.fmul(rh_b, dt), s.float(0.65))
    var tr_sl = s.exp2(s.fsub(s.float(0.0), s.fmul(ext_sl, s.float(1.442695))))
    var step_int = s.fdiv(s.fsub(s.float(1.0), tr_sl), s.nmax(rh_b, s.float(0.00001)))
    var tr_before = s.load_f(cum_tr)
    var w_int = s.fmul(tr_before, step_int)
    var nr = s.fma(sr_b, w_int, s.load_f(cum_r))
    var ng = s.fma(sg_b, w_int, s.load_f(cum_g))
    var nb = s.fma(sb_b, w_int, s.load_f(cum_b))
    var ntr = s.fmul(tr_before, tr_sl)
    s.store(cum_r, nr)
    s.store(cum_g, ng)
    s.store(cum_b, nb)
    s.store(cum_tr, ntr)

    var i_idx = s.iadd(integ_off, cell_rel)
    s.store_word(cells, i_idx, s.bitcast_uint(nr))
    s.store_word(cells, s.iadd(i_idx, s.uint(1)), s.bitcast_uint(ng))
    s.store_word(cells, s.iadd(i_idx, s.uint(2)), s.bitcast_uint(nb))
    s.store_word(cells, s.iadd(i_idx, s.uint(3)), s.bitcast_uint(ntr))
    s.end_loop()
    return s.finish()


def hiz_pyramid_shader() -> List[UInt32]:
    """Build 16x16 coarse Hierarchical-Z (min inverse_z = farthest depth in tile) into BIND_BACKDROP."""
    var s = SpirvModule(8, 4, HIZ_PUSH_WORDS)
    var target = s.buffer(BIND_TARGET)
    var backdrop = s.buffer(BIND_BACKDROP)
    var tx = s.invocation_axis(0)
    var ty = s.invocation_axis(1)
    var row_w = s.push_word(HIZ_PUSH_ROW)
    var width_w = s.push_word(HIZ_PUSH_WIDTH)
    var height_w = s.push_word(HIZ_PUSH_HEIGHT)
    var tiles_x = s.push_word(HIZ_PUSH_TILES_X)
    var depth_off = s.push_word(HIZ_PUSH_DEPTH_OFF)
    var out_off = s.push_word(HIZ_PUSH_OUT_OFF)

    var x_start = s.shl(tx, s.uint(4))
    var y_start = s.shl(ty, s.uint(4))
    s.return_if(s.uge(x_start, width_w))
    s.return_if(s.uge(y_start, height_w))

    var x_end = s.umin(s.iadd(x_start, s.uint(16)), width_w)
    var y_end = s.umin(s.iadd(y_start, s.uint(16)), height_w)
    var min_iz = s.fvar(s.float(1e9))

    var py_v = s.uvar(y_start)
    s.begin_loop()
    var py = s.load_u(py_v)
    s.break_if(s.uge(py, y_end))
    s.store(py_v, s.iadd(py, s.uint(1)))
    var row_base = s.iadd(depth_off, s.imul(py, row_w))
    var px_v = s.uvar(x_start)
    s.begin_loop()
    var px = s.load_u(px_v)
    s.break_if(s.uge(px, x_end))
    s.store(px_v, s.iadd(px, s.uint(1)))
    var iz = s.load_float(target, s.iadd(row_base, px))
    s.store(min_iz, s.nmin(s.load_f(min_iz), iz))
    s.end_loop()
    s.end_loop()

    var out_word = s.iadd(out_off, s.iadd(s.imul(ty, tiles_x), tx))
    s.store_word(backdrop, out_word, s.bitcast_uint(s.load_f(min_iz)))
    return s.finish()


def gpu_skin_shader() -> List[UInt32]:
    """GPU Compute 4-Bone Linear-Blend Skeletal Skinning in BIND_BACKDROP."""
    var s = SpirvModule(64, 1, SKIN_PUSH_WORDS)
    var backdrop = s.buffer(BIND_BACKDROP)
    var vid = s.invocation_axis(0)
    var vert_cnt = s.push_word(SKIN_PUSH_VERT_COUNT)
    s.return_if(s.uge(vid, vert_cnt))

    var joints_off = s.push_word(SKIN_PUSH_JOINTS_OFF)
    var in_off = s.push_word(SKIN_PUSH_IN_OFF)
    var out_off = s.push_word(SKIN_PUSH_OUT_OFF)

    var v_base = s.iadd(in_off, s.imul(vid, s.uint(12)))
    var px = s.load_float(backdrop, v_base)
    var py = s.load_float(backdrop, s.iadd(v_base, s.uint(1)))
    var pz = s.load_float(backdrop, s.iadd(v_base, s.uint(2)))
    var nx = s.load_float(backdrop, s.iadd(v_base, s.uint(3)))
    var ny = s.load_float(backdrop, s.iadd(v_base, s.uint(4)))
    var nz = s.load_float(backdrop, s.iadd(v_base, s.uint(5)))
    var j_pk = s.load_word(backdrop, s.iadd(v_base, s.uint(6)))

    var out_px = s.fvar(s.float(0.0))
    var out_py = s.fvar(s.float(0.0))
    var out_pz = s.fvar(s.float(0.0))
    var out_nx = s.fvar(s.float(0.0))
    var out_ny = s.fvar(s.float(0.0))
    var out_nz = s.fvar(s.float(0.0))

    for inf in range(4):
        var w_inf = s.load_float(backdrop, s.iadd(v_base, s.uint(UInt32(7 + inf))))
        s.begin_if(s.fgt(w_inf, s.float(0.00001)))
        var j_idx = s.bit_and(s.shr(j_pk, s.uint(UInt32(inf * 8))), s.uint(63))
        var m_base = s.iadd(joints_off, s.imul(j_idx, s.uint(12)))
        var r0x = s.load_float(backdrop, m_base)
        var r0y = s.load_float(backdrop, s.iadd(m_base, s.uint(1)))
        var r0z = s.load_float(backdrop, s.iadd(m_base, s.uint(2)))
        var r0w = s.load_float(backdrop, s.iadd(m_base, s.uint(3)))
        var r1x = s.load_float(backdrop, s.iadd(m_base, s.uint(4)))
        var r1y = s.load_float(backdrop, s.iadd(m_base, s.uint(5)))
        var r1z = s.load_float(backdrop, s.iadd(m_base, s.uint(6)))
        var r1w = s.load_float(backdrop, s.iadd(m_base, s.uint(7)))
        var r2x = s.load_float(backdrop, s.iadd(m_base, s.uint(8)))
        var r2y = s.load_float(backdrop, s.iadd(m_base, s.uint(9)))
        var r2z = s.load_float(backdrop, s.iadd(m_base, s.uint(10)))
        var r2w = s.load_float(backdrop, s.iadd(m_base, s.uint(11)))

        var tx = s.fadd(s.fma(r0z, pz, s.fma(r0y, py, s.fmul(r0x, px))), r0w)
        var ty = s.fadd(s.fma(r1z, pz, s.fma(r1y, py, s.fmul(r1x, px))), r1w)
        var tz = s.fadd(s.fma(r2z, pz, s.fma(r2y, py, s.fmul(r2x, px))), r2w)
        s.store(out_px, s.fma(tx, w_inf, s.load_f(out_px)))
        s.store(out_py, s.fma(ty, w_inf, s.load_f(out_py)))
        s.store(out_pz, s.fma(tz, w_inf, s.load_f(out_pz)))

        var tnx = s.fma(r0z, nz, s.fma(r0y, ny, s.fmul(r0x, nx)))
        var tny = s.fma(r1z, nz, s.fma(r1y, ny, s.fmul(r1x, nx)))
        var tnz = s.fma(r2z, nz, s.fma(r2y, ny, s.fmul(r2x, nx)))
        s.store(out_nx, s.fma(tnx, w_inf, s.load_f(out_nx)))
        s.store(out_ny, s.fma(tny, w_inf, s.load_f(out_ny)))
        s.store(out_nz, s.fma(tnz, w_inf, s.load_f(out_nz)))
        s.end_if()

    var fnx = s.load_f(out_nx)
    var fny = s.load_f(out_ny)
    var fnz = s.load_f(out_nz)
    var inv_nl = s.inverse_sqrt(s.fadd(dot3(s, Vec3Ids(fnx, fny, fnz), Vec3Ids(fnx, fny, fnz)), s.float(0.000001)))
    var dst = s.iadd(out_off, s.imul(vid, s.uint(6)))
    s.store_word(backdrop, dst, s.bitcast_uint(s.load_f(out_px)))
    s.store_word(backdrop, s.iadd(dst, s.uint(1)), s.bitcast_uint(s.load_f(out_py)))
    s.store_word(backdrop, s.iadd(dst, s.uint(2)), s.bitcast_uint(s.load_f(out_pz)))
    s.store_word(backdrop, s.iadd(dst, s.uint(3)), s.bitcast_uint(s.fmul(fnx, inv_nl)))
    s.store_word(backdrop, s.iadd(dst, s.uint(4)), s.bitcast_uint(s.fmul(fny, inv_nl)))
    s.store_word(backdrop, s.iadd(dst, s.uint(5)), s.bitcast_uint(s.fmul(fnz, inv_nl)))
    return s.finish()


def gpu_particles_shader() -> List[UInt32]:
    """V9 GPU Compute 65,536-Particle Ballistic + Curl-Turbulence + Depth-Buffer Soft-Bounce Solver."""
    var s = SpirvModule(64, 1, PARTICLE_PUSH_WORDS)
    var res = Resources(s)
    var backdrop = s.buffer(BIND_BACKDROP)
    var target = s.buffer(BIND_TARGET)
    var pid = s.invocation_axis(0)
    var count = s.push_word(PARTICLE_PUSH_COUNT)
    s.return_if(s.uge(pid, count))

    var in_off = s.push_word(PARTICLE_PUSH_IN_OFF)
    var out_off = s.push_word(PARTICLE_PUSH_OUT_OFF)
    var dt = s.push_float(PARTICLE_PUSH_DT)
    var gravity = s.push_float(PARTICLE_PUSH_GRAVITY)
    var drag = s.push_float(PARTICLE_PUSH_DRAG)
    var depth_off = s.push_word(PARTICLE_PUSH_DEPTH_OFF)
    var row_bounce = s.push_word(PARTICLE_PUSH_ROW_BOUNCE)
    var row_w = s.bit_and(row_bounce, s.uint(0xFFFF))
    var rest = s.fmul(s.float_from_uint(s.bit_and(s.shr(row_bounce, s.uint(16)), s.uint(255))), s.float(0.003921569))

    var stride_u = s.uint(UInt32(GPU_PARTICLE_STRIDE_WORDS))
    var src = s.iadd(in_off, s.imul(pid, stride_u))
    var dst = s.iadd(out_off, s.imul(pid, stride_u))

    var px0 = s.load_float(backdrop, src)
    var py0 = s.load_float(backdrop, s.iadd(src, s.uint(1)))
    var pz0 = s.load_float(backdrop, s.iadd(src, s.uint(2)))
    var life0 = s.load_float(backdrop, s.iadd(src, s.uint(3)))
    var vx0 = s.load_float(backdrop, s.iadd(src, s.uint(4)))
    var vy0 = s.load_float(backdrop, s.iadd(src, s.uint(5)))
    var vz0 = s.load_float(backdrop, s.iadd(src, s.uint(6)))
    var meta = s.load_word(backdrop, s.iadd(src, s.uint(7)))

    var px = s.fvar(px0)
    var py = s.fvar(py0)
    var pz = s.fvar(pz0)
    var life = s.fvar(s.float(0.0))
    var vx = s.fvar(vx0)
    var vy = s.fvar(vy0)
    var vz = s.fvar(vz0)

    s.begin_if(s.fgt(life0, s.float(0.0)))
    var life1 = s.nmax(s.fsub(life0, dt), s.float(0.0))
    s.store(life, life1)
    var damp = s.nclamp(s.fsub(s.float(1.0), s.fmul(drag, dt)), s.float(0.0), s.float(1.0))
    var wind_x = constant_float(s, res, CONSTANT_V9_CLOUDS + 16)
    var wind_z = constant_float(s, res, CONSTANT_V9_CLOUDS + 20)
    var curl_p = s.fma(pz0, s.float(0.041), s.fma(px0, s.float(0.053), s.fmul(life0, s.float(0.65))))
    var curl_x = _parabolic_sin(s, curl_p)
    var curl_z = _parabolic_sin(s, s.fadd(curl_p, s.float(0.25)))
    var nvx = s.fma(s.fma(wind_x, s.float(0.08), s.fmul(curl_x, s.float(1.25))), dt, s.fmul(vx0, damp))
    var nvy = s.fma(gravity, dt, s.fmul(vy0, damp))
    var nvz = s.fma(s.fma(wind_z, s.float(0.08), s.fmul(curl_z, s.float(1.25))), dt, s.fmul(vz0, damp))
    var npx = s.fma(nvx, dt, px0)
    var npy = s.fma(nvy, dt, py0)
    var npz = s.fma(nvz, dt, pz0)
    s.store(vx, nvx)
    s.store(vy, nvy)
    s.store(vz, nvz)
    s.store(px, npx)
    s.store(py, npy)
    s.store(pz, npz)

    # Screen-space PLANE_DEPTH soft-bounce collision against visible terrain/hulls
    s.begin_if(s.logical_and(s.ine(depth_off, s.uint(0)), s.ult(s.uint(0), row_w)))
    var camera = constant_vector(s, res, CONSTANT_CAMERA)
    var right = constant_vector(s, res, CONSTANT_CAMERA + 12)
    var up = constant_vector(s, res, CONSTANT_CAMERA + 24)
    var forward = constant_vector(s, res, CONSTANT_CAMERA + 36)
    var half_w = constant_float(s, res, CONSTANT_CAMERA + 48)
    var half_h = constant_float(s, res, CONSTANT_CAMERA + 52)
    var inv_focal = constant_float(s, res, CONSTANT_CAMERA + 56)
    var focal = s.fdiv(s.float(1.0), s.nmax(inv_focal, s.float(0.00001)))
    var rel = Vec3Ids(s.fsub(npx, camera.x), s.fsub(npy, camera.y), s.fsub(npz, camera.z))
    var vz_cam = dot3(s, rel, forward)
    s.begin_if(s.fgt(vz_cam, s.float(0.35)))
    var inv_vz = s.fdiv(s.float(1.0), vz_cam)
    var sx = s.fma(s.fmul(dot3(s, rel, right), focal), inv_vz, half_w)
    var sy = s.fsub(half_h, s.fmul(s.fmul(dot3(s, rel, up), focal), inv_vz))
    var max_x = s.fsub(s.fmul(half_w, s.float(2.0)), s.float(2.0))
    var max_y = s.fsub(s.fmul(half_h, s.float(2.0)), s.float(2.0))
    var in_scr = s.logical_and(
        s.logical_and(s.fge(sx, s.float(1.0)), s.flt(sx, max_x)),
        s.logical_and(s.fge(sy, s.float(1.0)), s.flt(sy, max_y)),
    )
    s.begin_if(in_scr)
    var spx = s.iadd(depth_off, s.iadd(s.imul(s.saturate_uint(sy), row_w), s.saturate_uint(sx)))
    var iz_b = s.load_float(target, spx)
    s.begin_if(s.fgt(iz_b, s.float(0.002)))
    var z_b = s.fdiv(s.float(1.0), iz_b)
    var pen = s.fsub(vz_cam, z_b)
    s.begin_if(s.logical_and(s.fgt(pen, s.float(0.0)), s.flt(pen, s.float(2.20))))
    var push_out = s.fadd(pen, s.float(0.05))
    s.store(px, s.fsub(npx, s.fmul(forward.x, push_out)))
    s.store(py, s.fsub(npy, s.fmul(forward.y, push_out)))
    s.store(pz, s.fsub(npz, s.fmul(forward.z, push_out)))
    var neg_rest = s.fsub(s.float(0.0), rest)
    s.store(vx, s.fmul(nvx, neg_rest))
    s.store(vy, s.fma(s.fabs(nvy), rest, s.float(0.45)))
    s.store(vz, s.fmul(nvz, neg_rest))
    s.end_if()
    s.end_if()
    s.end_if()
    s.end_if()
    s.end_if()
    s.end_if()

    s.store_word(backdrop, dst, s.bitcast_uint(s.load_f(px)))
    s.store_word(backdrop, s.iadd(dst, s.uint(1)), s.bitcast_uint(s.load_f(py)))
    s.store_word(backdrop, s.iadd(dst, s.uint(2)), s.bitcast_uint(s.load_f(pz)))
    s.store_word(backdrop, s.iadd(dst, s.uint(3)), s.bitcast_uint(s.load_f(life)))
    s.store_word(backdrop, s.iadd(dst, s.uint(4)), s.bitcast_uint(s.load_f(vx)))
    s.store_word(backdrop, s.iadd(dst, s.uint(5)), s.bitcast_uint(s.load_f(vy)))
    s.store_word(backdrop, s.iadd(dst, s.uint(6)), s.bitcast_uint(s.load_f(vz)))
    s.store_word(backdrop, s.iadd(dst, s.uint(7)), meta)
    return s.finish()


def tsr_upscale_shader() -> List[UInt32]:
    """V10 Temporal Super-Resolution Upscaler (5-Tap Bicubic Catmull-Rom + YCoCg History Clamp)."""
    var s = SpirvModule(GROUP_WIDTH, GROUP_HEIGHT, TSR_PUSH_WORDS)
    var backdrop = s.buffer(BIND_BACKDROP)
    var target = s.buffer(BIND_TARGET)
    var column = s.invocation_axis(0)
    var line = s.invocation_axis(1)

    var src_wh = s.push_word(TSR_PUSH_SRC_WH)
    var dst_wh = s.push_word(TSR_PUSH_DST_WH)
    var src_w = s.bit_and(src_wh, s.uint(0xFFFF))
    var src_h = s.shr(src_wh, s.uint(16))
    var dst_w = s.bit_and(dst_wh, s.uint(0xFFFF))
    var dst_h = s.shr(dst_wh, s.uint(16))
    s.return_if(s.uge(column, dst_w))
    s.return_if(s.uge(line, dst_h))

    var src_off = s.push_word(TSR_PUSH_SRC_OFF)
    var dst_off = s.push_word(TSR_PUSH_DST_OFF)
    var hist_off = s.push_word(TSR_PUSH_HIST_OFF)
    var vel_off = s.push_word(TSR_PUSH_VEL_OFF)
    var row_blend = s.push_word(TSR_PUSH_ROW_BLEND)
    var sharp_jit = s.push_word(TSR_PUSH_SHARP_JITTER)

    var row_w = s.bit_and(row_blend, s.uint(0xFFFF))
    var blend_f = s.fmul(s.float_from_uint(s.bit_and(s.shr(row_blend, s.uint(16)), s.uint(255))), s.float(0.003921569))
    var sharp_f = s.fmul(s.float_from_uint(s.bit_and(sharp_jit, s.uint(255))), s.float(0.007843137))
    var jit_x = s.fmul(s.fsub(s.float_from_uint(s.bit_and(s.shr(sharp_jit, s.uint(8)), s.uint(255))), s.float(128.0)), s.float(0.0078125))
    var jit_y = s.fmul(s.fsub(s.float_from_uint(s.bit_and(s.shr(sharp_jit, s.uint(16)), s.uint(255))), s.float(128.0)), s.float(0.0078125))

    var one_u = s.uint(1)
    var max_sx = s.isub(src_w, one_u)
    var max_sy = s.isub(src_h, one_u)
    var max_dx = s.isub(dst_w, one_u)
    var max_dy = s.isub(dst_h, one_u)

    var scale_x = s.fdiv(s.float_from_uint(src_w), s.nmax(s.float_from_uint(dst_w), s.float(1.0)))
    var scale_y = s.fdiv(s.float_from_uint(src_h), s.nmax(s.float_from_uint(dst_h), s.float(1.0)))
    var u_src = s.fsub(s.fmul(s.fadd(s.float_from_uint(column), s.fadd(s.float(0.5), jit_x)), scale_x), s.float(0.5))
    var v_src = s.fsub(s.fmul(s.fadd(s.float_from_uint(line), s.fadd(s.float(0.5), jit_y)), scale_y), s.float(0.5))

    # 5-Tap Catmull-Rom / Bicubic spatial reconstruction with anti-ringing neighborhood clamp
    var c_ctr = _sample_packed_bilinear_rgb(s, backdrop, src_off, src_w, max_sx, max_sy, u_src, v_src)
    var c_n = _sample_packed_bilinear_rgb(s, backdrop, src_off, src_w, max_sx, max_sy, u_src, s.fsub(v_src, s.float(1.0)))
    var c_s = _sample_packed_bilinear_rgb(s, backdrop, src_off, src_w, max_sx, max_sy, u_src, s.fadd(v_src, s.float(1.0)))
    var c_w = _sample_packed_bilinear_rgb(s, backdrop, src_off, src_w, max_sx, max_sy, s.fsub(u_src, s.float(1.0)), v_src)
    var c_e = _sample_packed_bilinear_rgb(s, backdrop, src_off, src_w, max_sx, max_sy, s.fadd(u_src, s.float(1.0)), v_src)

    var up_packed = s.uvar(s.uint(0xFF000000))
    for ch in range(3):
        var sh = 16 - ch * 8
        var cc = s.float_from_uint(channel(s, c_ctr, sh))
        var cn = s.float_from_uint(channel(s, c_n, sh))
        var cs = s.float_from_uint(channel(s, c_s, sh))
        var cw = s.float_from_uint(channel(s, c_w, sh))
        var ce = s.float_from_uint(channel(s, c_e, sh))
        var lo_c = s.nmin(cc, s.nmin(s.nmin(cn, cs), s.nmin(cw, ce)))
        var hi_c = s.nmax(cc, s.nmax(s.nmax(cn, cs), s.nmax(cw, ce)))
        var avg4 = s.fmul(s.fadd(s.fadd(cn, cs), s.fadd(cw, ce)), s.float(0.25))
        # Catmull-Rom negative lobe (-0.125 * 4 = -0.50) modulated by sharpness, anti-ringing clamped
        var cat_w = s.fma(sharp_f, s.float(0.45), s.float(0.18))
        var val = float_to_byte(s, s.nclamp(s.fma(s.fsub(cc, avg4), cat_w, cc), lo_c, hi_c))
        s.store(up_packed, s.bit_or(s.load_u(up_packed), s.shl(val, s.uint(UInt32(sh)))))

    var resolved = s.uvar(s.load_u(up_packed))

    # Optional native-resolution motion-compensated YCoCg history accumulation
    s.begin_if(s.logical_and(s.ine(hist_off, s.uint(0)), s.fgt(blend_f, s.float(0.005))))
    var dst_word = s.iadd(s.imul(line, row_w), column)
    var vx_px = s.fvar(s.float(0.0))
    var vy_px = s.fvar(s.float(0.0))
    s.begin_if(s.ine(vel_off, s.uint(0)))
    var vw = s.load_word(target, s.iadd(vel_off, dst_word))
    s.begin_if(s.ine(vw, s.uint(0)))
    s.store(vx_px, s.fmul(s.fsub(s.float_from_uint(s.bit_and(vw, s.uint(65535))), s.float(32768.0)), s.float(0.015625)))
    s.store(vy_px, s.fmul(s.fsub(s.float_from_uint(s.shr(vw, s.uint(16))), s.float(32768.0)), s.float(0.015625)))
    s.end_if()
    s.end_if()
    var prev_x = s.fsub(s.float_from_uint(column), s.load_f(vx_px))
    var prev_y = s.fsub(s.float_from_uint(line), s.load_f(vy_px))
    var in_hist = s.logical_and(
        s.logical_and(s.fge(prev_x, s.float(0.0)), s.flt(prev_x, s.float_from_uint(max_dx))),
        s.logical_and(s.fge(prev_y, s.float(0.0)), s.flt(prev_y, s.float_from_uint(max_dy))),
    )
    s.begin_if(in_hist)
    var h_col = _sample_packed_bilinear_rgb(s, target, hist_off, row_w, max_dx, max_dy, prev_x, prev_y)
    var yc_c = _rgb_to_ycocg(s, s.load_u(resolved))
    var yc_n = _rgb_to_ycocg(s, c_n)
    var yc_s = _rgb_to_ycocg(s, c_s)
    var yc_w = _rgb_to_ycocg(s, c_w)
    var yc_e = _rgb_to_ycocg(s, c_e)
    var lo_y = s.nmin(yc_c.x, s.nmin(s.nmin(yc_n.x, yc_s.x), s.nmin(yc_w.x, yc_e.x)))
    var hi_y = s.nmax(yc_c.x, s.nmax(s.nmax(yc_n.x, yc_s.x), s.nmax(yc_w.x, yc_e.x)))
    var lo_co = s.nmin(yc_c.y, s.nmin(s.nmin(yc_n.y, yc_s.y), s.nmin(yc_w.y, yc_e.y)))
    var hi_co = s.nmax(yc_c.y, s.nmax(s.nmax(yc_n.y, yc_s.y), s.nmax(yc_w.y, yc_e.y)))
    var lo_cg = s.nmin(yc_c.z, s.nmin(s.nmin(yc_n.z, yc_s.z), s.nmin(yc_w.z, yc_e.z)))
    var hi_cg = s.nmax(yc_c.z, s.nmax(s.nmax(yc_n.z, yc_s.z), s.nmax(yc_w.z, yc_e.z)))
    var yc_h = _rgb_to_ycocg(s, h_col)
    var cl_y = s.nclamp(yc_h.x, lo_y, hi_y)
    var cl_co = s.nclamp(yc_h.y, lo_co, hi_co)
    var cl_cg = s.nclamp(yc_h.z, lo_cg, hi_cg)
    var tmp_h = s.fsub(cl_y, cl_cg)
    var hr_cl = s.nclamp(s.fadd(tmp_h, cl_co), s.float(0.0), s.float(255.0))
    var hg_cl = s.nclamp(s.fadd(cl_y, cl_cg), s.float(0.0), s.float(255.0))
    var hb_cl = s.nclamp(s.fsub(tmp_h, cl_co), s.float(0.0), s.float(255.0))
    var spd = s.fsqrt(s.fma(s.load_f(vy_px), s.load_f(vy_px), s.fmul(s.load_f(vx_px), s.load_f(vx_px))))
    var eff_b = s.fdiv(blend_f, s.fma(spd, s.float(0.12), s.float(1.0)))
    var cur_p = s.load_u(resolved)
    var tr = float_to_byte(s, s.fma(s.fsub(hr_cl, s.float_from_uint(channel(s, cur_p, 16))), eff_b, s.float_from_uint(channel(s, cur_p, 16))))
    var tg = float_to_byte(s, s.fma(s.fsub(hg_cl, s.float_from_uint(channel(s, cur_p, 8))), eff_b, s.float_from_uint(channel(s, cur_p, 8))))
    var tb = float_to_byte(s, s.fma(s.fsub(hb_cl, s.float_from_uint(channel(s, cur_p, 0))), eff_b, s.float_from_uint(channel(s, cur_p, 0))))
    s.store(resolved, s.bit_or(s.uint(0xFF000000), s.bit_or(s.shl(tr, s.uint(16)), s.bit_or(s.shl(tg, s.uint(8)), tb))))
    s.end_if()
    s.end_if()

    var out_idx = s.iadd(dst_off, s.iadd(s.imul(line, dst_w), column))
    s.store_word(backdrop, out_idx, s.load_u(resolved))
    return s.finish()


def gpu_cull_shader() -> List[UInt32]:
    """V10 GPU-Driven Compute Frustum + 16x16 Hi-Z Occlusion Indirect Instance Culler."""
    var s = SpirvModule(64, 1, CULL_PUSH_WORDS)
    var res = Resources(s)
    var backdrop = s.buffer(BIND_BACKDROP)
    var iid = s.invocation_axis(0)
    var count = s.push_word(CULL_PUSH_INSTANCE_COUNT)
    s.return_if(s.uge(iid, count))

    var in_off = s.push_word(CULL_PUSH_IN_OFF)
    var vis_off = s.push_word(CULL_PUSH_VIS_OFF)
    var cnt_off = s.push_word(CULL_PUSH_COUNT_OFF)
    var ind_off = s.push_word(CULL_PUSH_INDIRECT_OFF)
    var hiz_off = s.push_word(CULL_PUSH_HIZ_OFF)
    var hiz_dims = s.push_word(CULL_PUSH_HIZ_DIMS)
    var scr_wh = s.push_word(CULL_PUSH_SCREEN_WH)

    var base = s.iadd(in_off, s.imul(iid, s.uint(UInt32(GPU_CULL_IN_STRIDE_WORDS))))
    var cx = s.load_float(backdrop, base)
    var cy = s.load_float(backdrop, s.iadd(base, s.uint(1)))
    var cz = s.load_float(backdrop, s.iadd(base, s.uint(2)))
    var rad = s.load_float(backdrop, s.iadd(base, s.uint(3)))
    var idx_cnt = s.load_word(backdrop, s.iadd(base, s.uint(4)))
    var first_idx = s.load_word(backdrop, s.iadd(base, s.uint(5)))
    var vtx_off = s.load_word(backdrop, s.iadd(base, s.uint(6)))
    var flags = s.load_word(backdrop, s.iadd(base, s.uint(7)))

    var camera = constant_vector(s, res, CONSTANT_CAMERA)
    var right = constant_vector(s, res, CONSTANT_CAMERA + 12)
    var up = constant_vector(s, res, CONSTANT_CAMERA + 24)
    var forward = constant_vector(s, res, CONSTANT_CAMERA + 36)
    var half_w = constant_float(s, res, CONSTANT_CAMERA + 48)
    var half_h = constant_float(s, res, CONSTANT_CAMERA + 52)
    var inv_focal = constant_float(s, res, CONSTANT_CAMERA + 56)
    var focal = s.fdiv(s.float(1.0), s.nmax(inv_focal, s.float(0.00001)))

    var rel = Vec3Ids(s.fsub(cx, camera.x), s.fsub(cy, camera.y), s.fsub(cz, camera.z))
    var vx = dot3(s, rel, right)
    var vy = dot3(s, rel, up)
    var vz = dot3(s, rel, forward)

    # Exact camera-space perspective frustum sphere test
    var tan_x = s.fmul(half_w, inv_focal)
    var tan_y = s.fmul(half_h, inv_focal)
    var sec_x = s.fsqrt(s.fma(tan_x, tan_x, s.float(1.0)))
    var sec_y = s.fsqrt(s.fma(tan_y, tan_y, s.float(1.0)))
    var in_z = s.fge(s.fadd(vz, rad), s.float(0.20))
    var in_x = s.fge(s.fma(vz, tan_x, s.fmul(rad, sec_x)), s.fabs(vx))
    var in_y = s.fge(s.fma(vz, tan_y, s.fmul(rad, sec_y)), s.fabs(vy))
    var visible = s.uvar(s.select_u(s.logical_and(in_z, s.logical_and(in_x, in_y)), s.uint(1), s.uint(0)))

    # Optional conservative 16x16 Hi-Z occlusion culling when (flags & 1) != 0 and hiz_off != 0
    var near_z = s.fsub(vz, rad)
    var do_hiz = s.logical_and(
        s.ine(s.load_u(visible), s.uint(0)),
        s.logical_and(
            s.logical_and(s.ine(hiz_off, s.uint(0)), s.ine(s.bit_and(flags, s.uint(1)), s.uint(0))),
            s.fgt(near_z, s.float(0.35)),
        ),
    )
    s.begin_if(do_hiz)
    var hiz_tx = s.bit_and(hiz_dims, s.uint(0xFFFF))
    var hiz_ty = s.shr(hiz_dims, s.uint(16))
    var scr_w = s.float_from_uint(s.bit_and(scr_wh, s.uint(0xFFFF)))
    var scr_h = s.float_from_uint(s.shr(scr_wh, s.uint(16)))
    var inv_vz = s.fdiv(s.float(1.0), s.nmax(vz, s.float(0.35)))
    var sc_x = s.fma(s.fmul(vx, focal), inv_vz, half_w)
    var sc_y = s.fsub(half_h, s.fmul(s.fmul(vy, focal), inv_vz))
    var reach = s.fmul(s.fdiv(s.fmul(rad, focal), near_z), s.float(1.12))
    var min_px = s.fsub(sc_x, reach)
    var max_px = s.fadd(s.fadd(sc_x, reach), s.float(1.0))
    var min_py = s.fsub(sc_y, reach)
    var max_py = s.fadd(s.fadd(sc_y, reach), s.float(1.0))
    var on_scr = s.logical_and(
        s.logical_and(s.fge(max_px, s.float(0.0)), s.flt(min_px, scr_w)),
        s.logical_and(s.fge(max_py, s.float(0.0)), s.flt(min_py, scr_h)),
    )
    s.begin_if(on_scr)
    var tx0 = s.shr(s.saturate_uint(s.nmax(min_px, s.float(0.0))), s.uint(4))
    var ty0 = s.shr(s.saturate_uint(s.nmax(min_py, s.float(0.0))), s.uint(4))
    var tx1 = s.umin(s.isub(hiz_tx, s.uint(1)), s.iadd(s.shr(s.saturate_uint(s.nmax(max_px, s.float(0.0))), s.uint(4)), s.uint(1)))
    var ty1 = s.umin(s.isub(hiz_ty, s.uint(1)), s.iadd(s.shr(s.saturate_uint(s.nmax(max_py, s.float(0.0))), s.uint(4)), s.uint(1)))
    var span_x = s.isub(tx1, tx0)
    var span_y = s.isub(ty1, ty0)
    # Guard tile loop to <= 16x16 tiles so huge close spheres never stall a wave
    s.begin_if(s.logical_and(s.ult(span_x, s.uint(16)), s.ult(span_y, s.uint(16))))
    var biased_iz = s.fmul(s.fdiv(s.float(1.0), near_z), s.float(1.03))
    var all_occ = s.uvar(s.uint(1))
    var ty_v = s.uvar(ty0)
    s.begin_loop()
    var cy_t = s.load_u(ty_v)
    s.break_if(s.logical_or(s.uge(cy_t, s.iadd(ty1, s.uint(1))), s.ieq(s.load_u(all_occ), s.uint(0))))
    s.store(ty_v, s.iadd(cy_t, s.uint(1)))
    var tx_v = s.uvar(tx0)
    s.begin_loop()
    var cx_t = s.load_u(tx_v)
    s.break_if(s.uge(cx_t, s.iadd(tx1, s.uint(1))))
    s.store(tx_v, s.iadd(cx_t, s.uint(1)))
    var tile_iz = s.load_float(backdrop, s.iadd(hiz_off, s.iadd(s.imul(cy_t, hiz_tx), cx_t)))
    s.begin_if(s.logical_not(s.fgt(tile_iz, biased_iz)))
    s.store(all_occ, s.uint(0))
    s.end_if()
    s.end_loop()
    s.end_loop()
    s.begin_if(s.ine(s.load_u(all_occ), s.uint(0)))
    s.store(visible, s.uint(0))
    s.end_if()
    s.end_if()
    s.end_if()
    s.end_if()

    var vis_final = s.load_u(visible)
    s.store_word(backdrop, s.iadd(vis_off, iid), vis_final)
    s.begin_if(s.ine(vis_final, s.uint(0)))
    var slot = s.atomic_add_word(backdrop, cnt_off, s.uint(1))
    var cmd_dst = s.iadd(ind_off, s.imul(slot, s.uint(5)))
    s.store_word(backdrop, cmd_dst, idx_cnt)
    s.store_word(backdrop, s.iadd(cmd_dst, s.uint(1)), s.uint(1))
    s.store_word(backdrop, s.iadd(cmd_dst, s.uint(2)), first_idx)
    s.store_word(backdrop, s.iadd(cmd_dst, s.uint(3)), vtx_off)
    s.store_word(backdrop, s.iadd(cmd_dst, s.uint(4)), iid)
    s.end_if()
    return s.finish()
