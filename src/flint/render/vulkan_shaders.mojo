"""The frame's compute passes in SPIR-V, one for each shader in gpu_raster.mojo.

Those are the reference for what each pass does and keep the comments on why.
These follow them step by step, down to the order of multiplies and the integer
rounding, so the two backends agree to within a level or two. A wave's exec mask
becomes an ordinary branch here, because each invocation is one pixel.

Conversions differ in one way that matters: the ISA's clamp out-of-range floats,
and SPIR-V leaves them undefined, so every conversion here is clamped first.
"""
from flint.render.gpu_layout import (
    BIND_BACKDROP, BIND_CELLS, BIND_CONSTANTS, BIND_HUD, BIND_LISTS, BIND_RECORDS, BIND_SHOWN,
    BIND_TARGET, BIND_TEXTURES, BIND_TILES, BIND_VOXELS, BLOOM_FINAL_OFFSET, BLOOM_SAMPLES_X,
    BLOOM_SAMPLES_Y, CONSTANT_BLOOM_DISTANCE, CONSTANT_CAMERA, CONSTANT_DITHER, CONSTANT_EFFECTS, CONSTANT_KEY,
    CONSTANT_LIGHTS, CONSTANT_SATURATION, CONSTANT_SPHERES, CONSTANT_WORLD, EFFECT_BILINEAR,
    DITHER_FLAG, EFFECT_SHADOWS, GROUP_HEIGHT, GROUP_WIDTH, PBR_FLAG, PER_PIXEL_FLAG, RECORD_AMBIENT_OCCLUSION,
    RECORD_EMISSIVE, RECORD_FLAGS, RECORD_FOG, RECORD_INVERSE_Z, RECORD_LIGHT, RECORD_METALLIC, RECORD_NORMAL,
    RECORD_ROUGHNESS, RECORD_TEXTURE, RECORD_TEXTURE_FLOAT_HEIGHT, RECORD_TEXTURE_FLOAT_WIDTH, RECORD_TEXTURE_MAX_X,
    RECORD_TEXTURE_MAX_Y, RECORD_TEXTURE_WIDTH, RECORD_TINT, RECORD_U, RECORD_V, RECORD_WEIGHT_A,
    RECORD_WEIGHT_B, SHADOW_COARSE_STEPS, SHADOW_FINE_STEPS, TEXTURED_FLAG, TILE_ENTRY_DWORDS,
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
comptime BLUR_PUSH_WORDS: Int = 5

comptime COMPOSE_PUSH_ROW: Int = 0
comptime COMPOSE_PUSH_WIDTH: Int = 1
comptime COMPOSE_PUSH_HEIGHT: Int = 2
comptime COMPOSE_PUSH_CELLS_X: Int = 3
comptime COMPOSE_PUSH_SCALE_X: Int = 4
comptime COMPOSE_PUSH_SCALE_Y: Int = 5
comptime COMPOSE_PUSH_BLOOM: Int = 6
comptime COMPOSE_PUSH_SCENE: Int = 7
comptime COMPOSE_PUSH_CELLS_Y: Int = 8
comptime COMPOSE_PUSH_WORDS: Int = 9

comptime SMOOTH_PUSH_ROW: Int = 0
comptime SMOOTH_PUSH_WIDTH: Int = 1
comptime SMOOTH_PUSH_HEIGHT: Int = 2
comptime SMOOTH_PUSH_TARGET: Int = 3
comptime SMOOTH_PUSH_WORDS: Int = 4


struct Resources:
    """The ids the raster pass reaches from everywhere."""
    var records: UInt32
    var constants: UInt32
    var textures: UInt32
    var voxels: UInt32

    def __init__(out self, mut s: SpirvModule):
        self.records = s.buffer(BIND_RECORDS)
        self.constants = s.buffer(BIND_CONSTANTS)
        self.textures = s.buffer(BIND_TEXTURES)
        self.voxels = s.buffer(BIND_VOXELS)


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


def texel(mut s: SpirvModule, res: Resources, base: UInt32, x: UInt32, y: UInt32) -> UInt32:
    var width = field(s, res, base, RECORD_TEXTURE_WIDTH)
    var first = s.shr(field(s, res, base, RECORD_TEXTURE), s.uint(2))
    return s.load_word(res.textures, s.iadd(first, s.iadd(s.imul(y, width), x)))


def wrap(mut s: SpirvModule, value: UInt32, size: UInt32) -> UInt32:
    """An unsigned minimum with (value - size) folds value back into range."""
    return s.umin(s.isub(value, size), value)


def sample_bilinear(mut s: SpirvModule, res: Resources, base: UInt32, u: UInt32, v: UInt32) -> UInt32:
    """u and v are wrapped texel coordinates."""
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


def trace_shadow(mut s: SpirvModule, res: Resources, world: Vec3Ids, normal: Vec3Ids, to_key: Vec3Ids) -> UInt32:
    """1 where the key light gets through, 0 where a sphere or voxel blocks it."""
    var one = s.uint(1)
    var shadow = s.fvar(s.float(1.0))
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
    var penumbra = s.nclamp(
        s.fdiv(s.fmul(discriminant, s.float(4.0)), s.fadd(radius_squared, s.float(0.0001))),
        zero,
        s.float(1.0),
    )
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
    # Past the fine steps a voxel-sized stride still finds walls and costs half as much.
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
    view_dir: Vec3Ids,
    light_dir: Vec3Ids,
    light_color: Vec3Ids,
    n_dot_v: UInt32,
    n_dot_l: UInt32,
    alpha2: UInt32,
    k_smith: UInt32,
    one_minus_k: UInt32,
    f0_r: UInt32,
    f0_g: UInt32,
    f0_b: UInt32,
    spec_r: UInt32,
    spec_g: UInt32,
    spec_b: UInt32,
):
    """Cook-Torrance GGX microfacet specular + Schlick Fresnel for one light direction."""
    var hx = s.fadd(view_dir.x, light_dir.x)
    var hy = s.fadd(view_dir.y, light_dir.y)
    var hz = s.fadd(view_dir.z, light_dir.z)
    var h_raw = Vec3Ids(hx, hy, hz)
    var inv_h = s.inverse_sqrt(s.fadd(dot3(s, h_raw, h_raw), s.float(0.00001)))
    var half_dir = Vec3Ids(s.fmul(hx, inv_h), s.fmul(hy, inv_h), s.fmul(hz, inv_h))
    var n_dot_h = s.nmax(dot3(s, normal, half_dir), s.float(0.0))
    var v_dot_h = s.nclamp(dot3(s, view_dir, half_dir), s.float(0.0), s.float(1.0))

    # Trowbridge-Reitz GGX D(h) = alpha^2 / (pi * ((N.H)^2 * (alpha^2 - 1) + 1)^2)
    var ndh2 = s.fmul(n_dot_h, n_dot_h)
    var denom_d = s.fma(ndh2, s.fsub(alpha2, s.float(1.0)), s.float(1.0))
    var d_ggx = s.fdiv(alpha2, s.fadd(s.fmul(s.float(3.14159265), s.fmul(denom_d, denom_d)), s.float(0.0001)))

    # Schlick-Smith visibility V = 1 / (4 * ((N.V)*(1-k) + k) * ((N.L)*(1-k) + k))
    var gv = s.fma(n_dot_v, one_minus_k, k_smith)
    var gl = s.fma(n_dot_l, one_minus_k, k_smith)
    var vis = s.fdiv(s.float(0.25), s.fadd(s.fmul(gv, gl), s.float(0.0001)))
    var lobe = s.nmin(s.fmul(s.fmul(d_ggx, vis), n_dot_l), s.float(6.0))

    # Schlick Fresnel in [0..255] byte scale: F = F0 + (255 - F0) * (1 - V.H)^5
    var om = s.fsub(s.float(1.0), v_dot_h)
    var om2 = s.fmul(om, om)
    var om5 = s.fmul(s.fmul(om2, om2), om)
    var fr = s.fma(s.fsub(s.float(255.0), f0_r), om5, f0_r)
    var fg = s.fma(s.fsub(s.float(255.0), f0_g), om5, f0_g)
    var fb = s.fma(s.fsub(s.float(255.0), f0_b), om5, f0_b)

    s.store(spec_r, s.fma(s.fmul(lobe, light_color.x), fr, s.load_f(spec_r)))
    s.store(spec_g, s.fma(s.fmul(lobe, light_color.y), fg, s.load_f(spec_g)))
    s.store(spec_b, s.fma(s.fmul(lobe, light_color.z), fb, s.load_f(spec_b)))


def per_pixel_light(
    mut s: SpirvModule,
    res: Resources,
    base: UInt32,
    flags: UInt32,
    x: UInt32,
    y: UInt32,
    depth: UInt32,
    ray: Vec3Ids,
    camera: Vec3Ids,
    to_key: Vec3Ids,
    key_color: Vec3Ids,
    effects: UInt32,
    tile_lights: UInt32,
    base_r: UInt32,
    base_g: UInt32,
    base_b: UInt32,
    light_r: UInt32,
    light_g: UInt32,
    light_b: UInt32,
    spec_r: UInt32,
    spec_g: UInt32,
    spec_b: UInt32,
):
    """Key and local lights per pixel, with Cook-Torrance GGX microfacet specular when PBR_FLAG is set."""
    var world = Vec3Ids(s.fma(ray.x, depth, camera.x), s.fma(ray.y, depth, camera.y), s.fma(ray.z, depth, camera.z))
    var raw_x = plane(s, res, base, RECORD_NORMAL, x, y)
    var raw_y = plane(s, res, base, RECORD_NORMAL + 3, x, y)
    var raw_z = plane(s, res, base, RECORD_NORMAL + 6, x, y)
    var occlusion = plane(s, res, base, RECORD_AMBIENT_OCCLUSION, x, y)
    var raw = Vec3Ids(raw_x, raw_y, raw_z)
    var inverse_length = s.inverse_sqrt(s.fadd(dot3(s, raw, raw), s.float(0.000001)))
    var normal = Vec3Ids(s.fmul(inverse_length, raw_x), s.fmul(inverse_length, raw_y), s.fmul(inverse_length, raw_z))

    var has_pbr = s.ine(s.bit_and(flags, s.uint(PBR_FLAG)), s.uint(0))
    var roughness = s.nclamp(s.bitcast_float(field(s, res, base, RECORD_ROUGHNESS)), s.float(0.07), s.float(1.0))
    var metallic = s.nclamp(s.bitcast_float(field(s, res, base, RECORD_METALLIC)), s.float(0.0), s.float(1.0))
    var kd = s.fvar(s.float(1.0))
    s.begin_if(has_pbr)
    s.store(kd, s.fsub(s.float(1.0), s.fmul(metallic, s.float(0.65))))
    s.end_if()
    var kd_val = s.load_f(kd)

    var inv_ray = s.inverse_sqrt(s.fadd(dot3(s, ray, ray), s.float(0.000001)))
    var neg_inv_ray = s.fsub(s.float(0.0), inv_ray)
    var view_dir = Vec3Ids(s.fmul(ray.x, neg_inv_ray), s.fmul(ray.y, neg_inv_ray), s.fmul(ray.z, neg_inv_ray))
    var n_dot_v = s.nmax(dot3(s, normal, view_dir), s.float(0.001))
    var alpha_r = s.fmul(roughness, roughness)
    var alpha2 = s.fmul(alpha_r, alpha_r)
    var r_plus_1 = s.fadd(roughness, s.float(1.0))
    var k_smith = s.fmul(s.fmul(r_plus_1, r_plus_1), s.float(0.125))
    var one_minus_k = s.fsub(s.float(1.0), k_smith)
    var dielec_f0 = s.float(11.5)  # 0.045 * 255
    var f0_r = s.fma(metallic, s.fsub(base_r, dielec_f0), dielec_f0)
    var f0_g = s.fma(metallic, s.fsub(base_g, dielec_f0), dielec_f0)
    var f0_b = s.fma(metallic, s.fsub(base_b, dielec_f0), dielec_f0)

    # Key light: 0.78 + 0.22 * ambient occlusion matches the vertex formula.
    var facing = s.nmax(dot3(s, normal, to_key), s.float(0))
    var key_shadow = s.fvar(s.float(1.0))
    var key = s.fvar(s.fmul(s.fadd(s.float(0.78), s.fmul(s.float(0.22), occlusion)), facing))
    var traced = s.ine(s.bit_and(effects, s.uint(EFFECT_SHADOWS)), s.uint(0))
    # Faces turned away from the key light are already dark and need no ray.
    s.begin_if(s.logical_and(traced, s.flt(s.float(0), s.load_f(key))))
    var shadow = trace_shadow(s, res, world, normal, to_key)
    s.store(key_shadow, shadow)
    s.store(key, s.fmul(shadow, s.load_f(key)))
    s.end_if()
    var key_light = s.fmul(s.load_f(key), kd_val)
    s.store(light_r, s.fmul(key_color.x, key_light))
    s.store(light_g, s.fmul(key_color.y, key_light))
    s.store(light_b, s.fmul(key_color.z, key_light))

    # Key-light GGX specular + subtle grazing Fresnel sheen
    s.begin_if(has_pbr)
    var n_dot_l_key = s.fmul(facing, s.load_f(key_shadow))
    s.begin_if(s.flt(s.float(0.0001), n_dot_l_key))
    accumulate_ggx_specular(
        s, normal, view_dir, to_key, key_color, n_dot_v, n_dot_l_key,
        alpha2, k_smith, one_minus_k, f0_r, f0_g, f0_b, spec_r, spec_g, spec_b,
    )
    s.end_if()
    # Grazing Fresnel atmospheric/canopy edge sheen
    var nv_om = s.fsub(s.float(1.0), n_dot_v)
    var nv_om2 = s.fmul(nv_om, nv_om)
    var nv_om4 = s.fmul(nv_om2, nv_om2)
    var gloss = s.fsub(s.float(1.0), s.fmul(roughness, s.float(0.82)))
    var sheen = s.fmul(s.fmul(nv_om4, gloss), s.fmul(occlusion, s.float(22.0)))
    s.store(spec_r, s.fma(sheen, key_color.x, s.load_f(spec_r)))
    s.store(spec_g, s.fma(sheen, key_color.y, s.load_f(spec_g)))
    s.store(spec_b, s.fma(sheen, s.fadd(key_color.z, s.float(0.25)), s.load_f(spec_b)))
    s.end_if()

    # Local lights: the CPU marked which ones can reach this tile.
    var count = constant_word(s, res, CONSTANT_WORLD + 24)
    var one = s.uint(1)
    var light = s.uvar(s.uint(0))
    s.begin_loop()
    var index = s.load_u(light)
    s.break_if(s.uge(index, count))
    s.store(light, s.iadd(index, one))
    s.continue_if(s.ieq(s.bit_and(s.shr(tile_lights, index), one), s.uint(0)))
    var word = s.iadd(s.uint(UInt32(CONSTANT_LIGHTS // 4)), s.shl(index, s.uint(3)))
    var to_x = s.fsub(light_float(s, res, word, 0), world.x)
    var to_y = s.fsub(light_float(s, res, word, 1), world.y)
    var to_z = s.fsub(light_float(s, res, word, 2), world.z)
    var to_light = Vec3Ids(to_x, to_y, to_z)
    # 0.0001 keeps a light on the surface finite.
    var distance_squared = s.fadd(dot3(s, to_light, to_light), s.float(0.0001))
    var falloff = s.nmax(s.fsub(s.float(1.0), s.fmul(light_float(s, res, word, 7), distance_squared)), s.float(0))
    falloff = s.fmul(falloff, falloff)
    var inverse_distance = s.inverse_sqrt(distance_squared)
    var raw_ndl = s.nmax(s.fmul(dot3(s, normal, to_light), inverse_distance), s.float(0))
    var wrapped = s.fadd(s.fmul(raw_ndl, s.float(0.76)), s.float(0.24))
    var amount = s.fmul(s.fmul(s.fmul(falloff, wrapped), occlusion), kd_val)
    var l_col_r = light_float(s, res, word, 4)
    var l_col_g = light_float(s, res, word, 5)
    var l_col_b = light_float(s, res, word, 6)
    s.store(light_r, s.fma(l_col_r, amount, s.load_f(light_r)))
    s.store(light_g, s.fma(l_col_g, amount, s.load_f(light_g)))
    s.store(light_b, s.fma(l_col_b, amount, s.load_f(light_b)))
    s.begin_if(s.logical_and(has_pbr, s.flt(s.float(0.001), s.fmul(falloff, raw_ndl))))
    var l_dir = Vec3Ids(s.fmul(to_x, inverse_distance), s.fmul(to_y, inverse_distance), s.fmul(to_z, inverse_distance))
    var atten_col = Vec3Ids(s.fmul(l_col_r, falloff), s.fmul(l_col_g, falloff), s.fmul(l_col_b, falloff))
    accumulate_ggx_specular(
        s, normal, view_dir, l_dir, atten_col, n_dot_v, raw_ndl,
        alpha2, k_smith, one_minus_k, f0_r, f0_g, f0_b, spec_r, spec_g, spec_b,
    )
    s.end_if()
    s.end_loop()


def raster_shader() -> List[UInt32]:
    """Walk this pixel's tile list in order, with colour and depth held in variables."""
    var s = SpirvModule(GROUP_WIDTH, GROUP_HEIGHT, RASTER_PUSH_WORDS)
    var res = Resources(s)
    var lists = s.buffer(BIND_LISTS)
    var tiles = s.buffer(BIND_TILES)
    var target = s.buffer(BIND_TARGET)
    var column = s.invocation_axis(0)
    var line = s.invocation_axis(1)
    var row_pixels = s.push_word(RASTER_PUSH_ROW)
    # Group rounding runs past the edges, and those pixels are never shown.
    s.return_if(s.uge(column, s.push_word(RASTER_PUSH_WIDTH)))
    s.return_if(s.uge(line, s.push_word(RASTER_PUSH_HEIGHT)))
    var x = s.float_from_uint(column)
    var y = s.float_from_uint(line)
    var one = s.uint(1)

    var tile_index = s.iadd(s.shr(column, s.uint(5)), s.imul(s.shr(line, s.uint(5)), s.push_word(RASTER_PUSH_COLUMNS)))
    var tile_word = s.imul(tile_index, s.uint(UInt32(TILE_ENTRY_DWORDS)))
    var first = s.shr(s.load_word(tiles, tile_word), s.uint(2))
    var count = s.load_word(tiles, s.iadd(tile_word, one))
    var tile_lights = s.load_word(tiles, s.iadd(tile_word, s.uint(2)))
    var saturation = constant_word(s, res, CONSTANT_SATURATION)
    var bloom_distance = s.shr(constant_word(s, res, CONSTANT_BLOOM_DISTANCE), s.uint(2))
    var effects = constant_word(s, res, CONSTANT_EFFECTS)

    # The view ray through this pixel and the key light.
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
    # Package 3.2: the dither disc and this pixel's ordered-dither level (render/dither.mojo).
    var dither_x = constant_float(s, res, CONSTANT_DITHER)
    var dither_y = constant_float(s, res, CONSTANT_DITHER + 4)
    var dither_scale = constant_float(s, res, CONSTANT_DITHER + 8)
    var dither_offset = constant_float(s, res, CONSTANT_DITHER + 12)
    var dither_near = constant_float(s, res, CONSTANT_DITHER + 16)
    var dither_dx = s.fsub(x, dither_x)
    var dither_dy = s.fsub(y, dither_y)
    var dither_value = s.fma(s.fma(dither_dy, dither_dy, s.fmul(dither_dx, dither_dx)), dither_scale, dither_offset)
    var dither_v = s.bit_xor(column, line)
    var dither_level = s.bit_or(
        s.bit_or(s.shl(s.bit_and(dither_v, one), s.uint(3)), s.shl(s.bit_and(line, one), s.uint(2))),
        s.bit_or(s.bit_and(dither_v, s.uint(2)), s.bit_and(s.shr(line, one), one)),
    )
    var dither_below = s.flt(dither_value, s.float_from_uint(dither_level))

    var clear = s.push_word(RASTER_PUSH_CLEAR)
    var color = s.uvar(clear)
    var nearest_depth = s.fvar(s.float_bits(0x3089705F))  # 1e-9: the CPU clears depth to 1e9.
    var bloom = s.uvar(s.uint(0))
    var edge = s.float_bits(0xB8D1B717)  # -0.0001: the CPU's edge tolerance.
    var fog_r = s.float_from_uint(channel(s, clear, 16))
    var fog_g = s.float_from_uint(channel(s, clear, 8))
    var fog_b = s.float_from_uint(channel(s, clear, 0))
    var light_r = s.fvar(s.float(0))
    var light_g = s.fvar(s.float(0))
    var light_b = s.fvar(s.float(0))
    var spec_r = s.fvar(s.float(0))
    var spec_g = s.fvar(s.float(0))
    var spec_b = s.fvar(s.float(0))
    var sampled = s.uvar(s.uint(0xFFFFFFFF))

    var entry = s.uvar(s.uint(0))
    s.begin_loop()
    var index = s.load_u(entry)
    s.break_if(s.uge(index, count))
    s.store(entry, s.iadd(index, one))
    var base = s.shr(s.load_word(lists, s.iadd(first, index)), s.uint(2))

    # Coverage from barycentric weights.
    var weight_a = plane(s, res, base, RECORD_WEIGHT_A, x, y)
    var weight_b = plane(s, res, base, RECORD_WEIGHT_B, x, y)
    var weight_c = s.fsub(s.fsub(s.float(1.0), weight_a), weight_b)
    var inside = s.logical_and(s.logical_and(s.fge(weight_a, edge), s.fge(weight_b, edge)), s.fge(weight_c, edge))
    s.continue_if(s.logical_not(inside))

    # Nearer means a larger inverse depth.
    var inverse_z = plane(s, res, base, RECORD_INVERSE_Z, x, y)
    s.continue_if(s.logical_not(s.fgt(inverse_z, s.load_f(nearest_depth))))

    # Texture sample, or opaque white.
    var depth = s.fdiv(s.float(1.0), inverse_z)
    var flags = field(s, res, base, RECORD_FLAGS)
    # Occluders in front of Maroon thin out around him (package 3.2).
    var occluder = s.ine(s.bit_and(flags, s.uint(DITHER_FLAG)), s.uint(0))
    s.continue_if(s.logical_and(occluder, s.logical_and(s.fgt(inverse_z, dither_near), dither_below)))
    s.store(sampled, s.uint(0xFFFFFFFF))
    s.begin_if(s.ine(s.bit_and(flags, s.uint(TEXTURED_FLAG)), s.uint(0)))
    var texture_u = s.fract(s.fmul(depth, plane(s, res, base, RECORD_U, x, y)))
    var texture_v = s.fract(s.fmul(depth, plane(s, res, base, RECORD_V, x, y)))
    texture_u = s.fmul(s.bitcast_float(field(s, res, base, RECORD_TEXTURE_FLOAT_WIDTH)), texture_u)
    texture_v = s.fmul(s.bitcast_float(field(s, res, base, RECORD_TEXTURE_FLOAT_HEIGHT)), texture_v)
    s.begin_if(s.ine(s.bit_and(effects, s.uint(EFFECT_BILINEAR)), s.uint(0)))
    s.store(sampled, sample_bilinear(s, res, base, texture_u, texture_v))
    s.begin_else()
    var nearest_x = s.umin(s.saturate_uint(texture_u), field(s, res, base, RECORD_TEXTURE_MAX_X))
    var nearest_y = s.umin(s.saturate_uint(texture_v), field(s, res, base, RECORD_TEXTURE_MAX_Y))
    s.store(sampled, texel(s, res, base, nearest_x, nearest_y))
    s.end_if()
    s.end_if()

    var tint = field(s, res, base, RECORD_TINT)
    var pixel = s.load_u(sampled)
    var bases = List[UInt32]()
    for which in range(3):
        var shift = 16 - which * 8
        var product = s.imul(channel(s, tint, shift), channel(s, pixel, shift))
        bases.append(s.float_from_uint(divide_by_255(s, product)))

    s.store(light_r, s.float(0))
    s.store(light_g, s.float(0))
    s.store(light_b, s.float(0))
    s.store(spec_r, s.float(0))
    s.store(spec_g, s.float(0))
    s.store(spec_b, s.float(0))
    s.begin_if(s.ine(s.bit_and(flags, s.uint(PER_PIXEL_FLAG)), s.uint(0)))
    per_pixel_light(
        s, res, base, flags, x, y, depth, ray, camera, to_key, key_color, effects, tile_lights,
        bases[0], bases[1], bases[2], light_r, light_g, light_b, spec_r, spec_g, spec_b,
    )
    s.end_if()

    # Shared per-pixel inputs: emissive term, fog, and its inverse.
    var emissive = s.fmul(s.float(1.25), plane(s, res, base, RECORD_EMISSIVE, x, y))
    var fog = plane(s, res, base, RECORD_FOG, x, y)
    var clearness = s.fsub(s.float(1.0), fog)

    # Texture * tint, lighting + PBR specular, fog, then the region's tone table.
    var shades = List[UInt32]()
    for which in range(3):
        var base_color = bases[which]
        var fog_color = fog_r if which == 0 else (fog_g if which == 1 else fog_b)
        var pixel_light = s.load_f(light_r if which == 0 else (light_g if which == 1 else light_b))
        var pixel_spec = s.load_f(spec_r if which == 0 else (spec_g if which == 1 else spec_b))
        var total = s.fadd(s.fadd(plane(s, res, base, RECORD_LIGHT + which * 3, x, y), emissive), pixel_light)
        var lit = s.fmul(s.fma(base_color, total, pixel_spec), clearness)
        var shaded = float_to_byte(s, s.fadd(lit, s.fmul(fog_color, fog)))
        var table_byte = s.iadd(shaded, s.uint(UInt32(which * 256)))
        var loaded = s.load_word(res.constants, s.shr(table_byte, s.uint(2)))
        var bits = s.shl(s.bit_and(table_byte, s.uint(3)), s.uint(3))
        shades.append(s.bit_and(s.shr(loaded, bits), s.uint(255)))

    # Saturation about Rec. 709-style luminance.
    var red = s.uvar(shades[0])
    var green = s.uvar(shades[1])
    var blue = s.uvar(shades[2])
    s.begin_if(s.ine(saturation, s.uint(0)))
    var weighted = s.iadd(s.iadd(s.imul(shades[0], s.uint(54)), s.imul(shades[1], s.uint(183))), s.imul(shades[2], s.uint(19)))
    var luminance = s.shr(weighted, s.uint(8))
    var gain = s.iadd(saturation, s.uint(64))
    for which in range(3):
        var spread = s.sar(s.imul(s.isub(shades[which], luminance), gain), s.uint(6))
        s.store(red if which == 0 else (green if which == 1 else blue), clamp_byte(s, s.iadd(luminance, spread)))
    s.end_if()

    # Effective alpha: triangle * tint * texture.
    var scaled = s.imul(s.bit_and(flags, s.uint(255)), s.shr(tint, s.uint(24)))
    var triangle_alpha = s.shr(s.iadd(s.iadd(scaled, s.shr(scaled, s.uint(8))), one), s.uint(8))
    var alpha = divide_by_255(s, s.imul(triangle_alpha, s.shr(pixel, s.uint(24))))
    s.continue_if(s.ieq(alpha, s.uint(0)))

    # Blend over the running colour, keeping fully opaque pixels exact.
    var opacity = s.fmul(s.float_bits(0x3B808081), s.float_from_uint(alpha))  # 1 / 255
    var remaining = s.fsub(s.float(1.0), opacity)
    var previous = s.load_u(color)
    s.begin_if(s.ine(alpha, s.uint(255)))
    for which in range(3):
        var variable = red if which == 0 else (green if which == 1 else blue)
        var over = s.fmul(s.float_from_uint(s.load_u(variable)), opacity)
        var under = s.fmul(s.float_from_uint(channel(s, previous, 16 - which * 8)), remaining)
        s.store(variable, float_to_byte(s, s.fadd(over, under)))
    s.end_if()
    var packed = s.bit_or(s.shl(s.load_u(red), s.uint(16)), s.shl(s.load_u(green), s.uint(8)))
    s.store(color, s.bit_or(packed, s.load_u(blue)))

    # Only mostly opaque pixels write depth, as on the CPU.
    s.begin_if(s.ult(s.uint(127), alpha))
    s.store(nearest_depth, inverse_z)
    s.end_if()

    # Bloom source: base colour * emissive * alpha, plus hot PBR specular sun glints (> 145).
    var spec_peak = s.nmax(s.load_f(spec_r), s.nmax(s.load_f(spec_g), s.load_f(spec_b)))
    var bright = s.logical_or(
        s.fge(emissive, s.float_bits(0x3E4CCCCD)),  # 0.2: the CPU's 0.16 floor, times 1.25.
        s.fge(spec_peak, s.float(145.0)),
    )
    s.begin_if(s.logical_and(bright, s.ult(s.uint(23), alpha)))
    var strength = s.fmul(opacity, s.fmul(s.float_bits(0x3F4CCCCD), emissive))  # 0.8 removes the gain.
    var glow = s.uvar(s.uint(0))
    for which in range(3):
        var sp_c = s.load_f(spec_r if which == 0 else (spec_g if which == 1 else spec_b))
        var sp_bloom = s.fmul(opacity, s.fmul(s.nmax(s.fsub(sp_c, s.float(130.0)), s.float(0.0)), s.float(0.55)))
        var value = float_to_byte(s, s.fadd(s.fmul(strength, bases[which]), sp_bloom))
        s.store(glow, s.bit_or(s.load_u(glow), s.shl(value, s.uint(UInt32(16 - which * 8)))))
    s.store(bloom, s.load_u(glow))
    s.end_if()
    s.end_loop()

    var word = s.iadd(s.imul(line, row_pixels), column)
    s.store_word(target, word, s.bit_or(s.load_u(color), s.uint(0xFF000000)))
    s.store_word(target, s.iadd(word, bloom_distance), s.load_u(bloom))
    # Package 2.7: the inverse depth, two planes on, for the soft particles.
    s.store_word(target, s.iadd(word, s.imul(bloom_distance, s.uint(3))), s.bitcast_uint(s.load_f(nearest_depth)))
    return s.finish()


def bloom_reduce_shader() -> List[UInt32]:
    """Max-combine a grid of bright samples into each low-resolution bloom cell."""
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
    var one = s.uint(1)
    var red = s.uvar(s.uint(0))
    var green = s.uvar(s.uint(0))
    var blue = s.uvar(s.uint(0))
    var half = s.float(0.5)
    var outer = s.uvar(s.uint(0))
    s.begin_loop()
    var sample_y = s.load_u(outer)
    s.break_if(s.uge(sample_y, s.uint(UInt32(BLOOM_SAMPLES_Y))))
    s.store(outer, s.iadd(sample_y, one))
    var pixel_y = s.fadd(y, s.fmul(s.float_bits(0x3E2AAAAB), s.fadd(half, s.float_from_uint(sample_y))))  # 1 / 6
    var at_y = s.umin(s.saturate_uint(s.fmul(scale_y, pixel_y)), last_y)
    var inner = s.uvar(s.uint(0))
    s.begin_loop()
    var sample_x = s.load_u(inner)
    s.break_if(s.uge(sample_x, s.uint(UInt32(BLOOM_SAMPLES_X))))
    s.store(inner, s.iadd(sample_x, one))
    var pixel_x = s.fadd(x, s.fmul(s.float(0.125), s.fadd(half, s.float_from_uint(sample_x))))
    var at_x = s.umin(s.saturate_uint(s.fmul(scale_x, pixel_x)), last_x)
    var value = s.load_word(target, s.iadd(source, s.iadd(s.imul(at_y, row_pixels), at_x)))
    s.store(red, s.umax(s.load_u(red), channel(s, value, 16)))
    s.store(green, s.umax(s.load_u(green), channel(s, value, 8)))
    s.store(blue, s.umax(s.load_u(blue), channel(s, value, 0)))
    s.end_loop()
    s.end_loop()
    var cell = s.imul(s.iadd(s.imul(line, cell_columns), column), s.uint(3))
    s.store_word(cells, cell, s.bitcast_uint(s.float_from_uint(s.load_u(red))))
    s.store_word(cells, s.iadd(cell, one), s.bitcast_uint(s.float_from_uint(s.load_u(green))))
    s.store_word(cells, s.iadd(cell, s.uint(2)), s.bitcast_uint(s.float_from_uint(s.load_u(blue))))
    return s.finish()


def bloom_blur_shader(horizontal: Bool) -> List[UInt32]:
    """One 5-tap [0.0625, 0.25, 0.375, 0.25, 0.0625] Gaussian axis of the bloom blur.

    The vertical pass is last, so it also applies the bloom strength and packs
    each cell into a byte per channel.
    """
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
    var packed = s.uvar(s.uint(0xFF000000))
    var inverse_strength = s.push_float(BLUR_PUSH_INVERSE_STRENGTH)
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
            var byte = s.float_from_uint(float_to_byte(s, mixed))
            var scaled = float_to_byte(s, s.fmul(inverse_strength, byte))
            s.store(packed, s.bit_or(s.load_u(packed), s.shl(scaled, s.uint(UInt32(16 - which * 8)))))
    if not horizontal:
        s.store_word(cells, s.iadd(target, here), s.load_u(packed))
    return s.finish()


def compose_shader(capture: Bool) -> List[UInt32]:
    """Scene + bilinear-upsampled bloom, then the premultiplied HUD over it, into the shown image.

    The capture variant writes the backdrop instead, with nothing over the scene.
    """
    var s = SpirvModule(GROUP_WIDTH, GROUP_HEIGHT, COMPOSE_PUSH_WORDS)
    var target = s.buffer(BIND_TARGET)
    var cells = s.buffer(BIND_CELLS)
    var hud_image = s.buffer(BIND_HUD)
    var output = s.buffer(BIND_BACKDROP if capture else BIND_SHOWN)
    var column = s.invocation_axis(0)
    var line = s.invocation_axis(1)
    s.return_if(s.uge(column, s.push_word(COMPOSE_PUSH_WIDTH)))
    s.return_if(s.uge(line, s.push_word(COMPOSE_PUSH_HEIGHT)))
    var word = s.iadd(s.imul(line, s.push_word(COMPOSE_PUSH_ROW)), column)
    var scene = s.load_word(target, s.iadd(s.push_word(COMPOSE_PUSH_SCENE), word))
    var hud = s.uint(0) if capture else s.load_word(hud_image, word)
    var bloom = s.uvar(s.uint(0))
    s.begin_if(s.ine(s.push_word(COMPOSE_PUSH_BLOOM), s.uint(0)))
    var cell_columns = s.push_word(COMPOSE_PUSH_CELLS_X)
    var cell_rows = s.push_word(COMPOSE_PUSH_CELLS_Y)
    var max_cx = s.isub(cell_columns, s.uint(1))
    var max_cy = s.isub(cell_rows, s.uint(1))
    var half = s.float(0.5)
    var u = s.nmax(s.fsub(s.fmul(s.push_float(COMPOSE_PUSH_SCALE_X), s.fadd(s.float_from_uint(column), half)), half), s.float(0.0))
    var v = s.nmax(s.fsub(s.fmul(s.push_float(COMPOSE_PUSH_SCALE_Y), s.fadd(s.float_from_uint(line), half)), half), s.float(0.0))
    var x0 = s.umin(s.saturate_uint(u), max_cx)
    var x1 = s.umin(s.iadd(x0, s.uint(1)), max_cx)
    var y0 = s.umin(s.saturate_uint(v), max_cy)
    var y1 = s.umin(s.iadd(y0, s.uint(1)), max_cy)
    var fx = s.umin(s.saturate_uint(s.fmul(s.nmax(s.fsub(u, s.float_from_uint(x0)), s.float(0.0)), s.float(256.0))), s.uint(256))
    var fy = s.umin(s.saturate_uint(s.fmul(s.nmax(s.fsub(v, s.float_from_uint(y0)), s.float(0.0)), s.float(256.0))), s.uint(256))
    var base_off = s.uint(UInt32(BLOOM_FINAL_OFFSET // 4))
    var row0 = s.iadd(base_off, s.imul(y0, cell_columns))
    var row1 = s.iadd(base_off, s.imul(y1, cell_columns))
    var c00 = s.load_word(cells, s.iadd(row0, x0))
    var c10 = s.load_word(cells, s.iadd(row0, x1))
    var c01 = s.load_word(cells, s.iadd(row1, x0))
    var c11 = s.load_word(cells, s.iadd(row1, x1))
    var top_mix = lerp_packed(s, c00, c10, fx)
    var bot_mix = lerp_packed(s, c01, c11, fx)
    s.store(bloom, lerp_packed(s, top_mix, bot_mix, fy))
    s.end_if()
    var glow = s.load_u(bloom)
    var through = s.isub(s.uint(255), s.shr(hud, s.uint(24)))
    var limit = s.uint(255)
    var result = s.uvar(s.uint(0xFF000000))
    for which in range(3):
        var shift = 16 - which * 8
        var value = s.smin(s.iadd(channel(s, scene, shift), channel(s, glow, shift)), limit)
        value = divide_by_255(s, s.imul(through, value))
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
    # Across which axis does brightness change fastest?
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
    # Stronger blending the more this pixel stands out from its neighbours.
    var average = s.shr(s.iadd(vertical, sideways), s.uint(2))
    var stands_out = s.float_from_uint(absolute_int(s, s.isub(average, luma_center)))
    var ratio = s.fmul(stands_out, s.fdiv(s.float(1.0), s.float_from_uint(contrast)))
    var amount = s.saturate_uint(s.fmul(ratio, s.float(96.0)))  # 96 of 256 at full strength
    amount = s.umin(s.umax(amount, s.uint(48)), s.uint(128))
    s.store(result, lerp_packed(s, center, neighbour, amount))
    s.end_if()
    s.store_word(image, target, s.load_u(result))
    return s.finish()
