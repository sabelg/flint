"""Area H (sota:atmo, round 3): night sky drawn over the physical sky.

Everything is radiometric and in the same units as the sun (`SotaAtmoParams.solar_irradiance`), so the stars, the
Milky Way and the airglow fade in by themselves when the sky's in-scattering drops after sunset; no day/night gate.

* **Star catalogue (procedural, deterministic).** The celestial sphere (equatorial frame) is split into a cube of
  `SOTA_NIGHT_CELLS^2` cells per face. Each cell holds at most one star. The probability follows the cumulative star
  counts `N(<m) = 4800 * 10^(0.45 (m - 6))` over the whole sky (about 4.8k stars to V = 6, 13.5k to the default limit
  7.0) and rises toward the galactic plane. The magnitude is drawn from the same counts, the colour from a Planck
  spectrum (2 900 .. 25 000 K) relative to the 5 778 K sun. Irradiance: `E = E_sun * 10^(-0.4 (m + 26.74))` (luma),
  so Sirius-class stars sit at about 1e-10 of the sun, like the real sky.
* **Rendering.** Each star is a Gaussian point-spread function whose width is `star_psf_px` pixels, normalised so the
  pixels integrate to `E * T(view)` at any resolution (energy-conserving, no shimmer under TSR jitter). A pixel looks
  at the 2 x 2 cells that bracket it; star positions stay inside the middle 70 % of their cell.
* **Sky rotation.** World -> equatorial is the transpose of the local-sidereal-time / latitude rotation of
  `SotaTimeOfDay`, so the stars turn about the celestial pole (Polaris-like stars stand still).
* **Diffuse night sky.** Airglow + zodiacal light + unresolved stars at 22.0 mag/arcsec^2 at the zenith, brightening
  toward the horizon with the van Rhijn factor (90 km layer); a Milky Way band (peak 20.6 mag/arcsec^2, galactic pole
  RA 192.86 deg / Dec 27.13 deg, brighter toward the galactic centre) with value-noise dust lanes.

CPU twins: `star_cell_cpu`, `night_background_cpu`, `star_radiance_cpu`. Globals `b_globals[SOTA_ATMO_G2 + 0 .. 20)`.
"""
from std.collections import List
from std.math import abs, cos, exp, floor, log, max, min, sin, sqrt
from flint.core.math import Vec3
from flint.render.shader_dsl import (
    BufferRef,
    GpuF32,
    GpuU32,
    GpuVec3,
    ShaderDsl,
    abs_f,
    exp2_f,
    floor_f,
    log2_f,
    max_f,
    min_f,
    min_u,
    saturate,
    select_f,
    select_u,
    sqrt_f,
)
from flint.render.sota_atmo_core import RGB, SotaTimeOfDay, _equ_to_world
from flint.render.sota_atmo_dsl import SOTA_ATMO_G2, _gexp

comptime SOTA_NIGHT_CELLS: Int = 256
comptime SOTA_NIGHT_W_ROT: Int = 0          # 9 words: rows of world -> equatorial
comptime SOTA_NIGHT_W_STARS: Int = 9        # star irradiance scale (0 = off)
comptime SOTA_NIGHT_W_MLIM: Int = 10        # limiting magnitude
comptime SOTA_NIGHT_W_BG: Int = 11          # airglow radiance at the zenith (sun units)
comptime SOTA_NIGHT_W_MW: Int = 12          # Milky Way peak radiance (sun units)
comptime SOTA_NIGHT_W_PSF: Int = 13         # PSF sigma in pixels
comptime SOTA_NIGHT_W_SEED: Int = 14
comptime SOTA_NIGHT_W_E0: Int = 15          # irradiance of a magnitude-0 star (sun units)
comptime SOTA_NIGHT_W_FLAGS: Int = 16       # bit 0: night sky on, bit 1: clouds on

comptime _ARCSEC2_SR: Float32 = 2.3504430539e-11
comptime _SUN_MAG: Float32 = -26.74
comptime _COUNT_SLOPE: Float32 = 0.45
comptime _GAL_SIGMA2: Float32 = 0.06        # galactic-latitude star density: exp(-sin(b)^2 / 0.06)
comptime _GAL_NORM: Float32 = 1.0 / 0.8603  # mean of (0.6 + 1.2 exp(...)) over the sphere
comptime _VAN_RHIJN_K: Float32 = 0.97197    # (R / (R + 90 km))^2 for R = 6371 km
comptime _MW_SIGMA_B: Float32 = 0.13        # band half-width (rad)


# ============================================================================
# Shared definitions (CPU)
# ============================================================================

@always_inline
def hash_u32(x: UInt32) -> UInt32:
    """PCG-RXS-M-XS 32-bit hash (identical in the shaders)."""
    var s = x * UInt32(747796405) + UInt32(2891336453)
    var w = ((s >> ((s >> UInt32(28)) + UInt32(4))) ^ s) * UInt32(277803737)
    return (w >> UInt32(22)) ^ w


@always_inline
def hash_unit(h: UInt32) -> Float32:
    return Float32(h >> UInt32(8)) * Float32(1.0 / 16777216.0)


def _equ_vec(ra_deg: Float64, dec_deg: Float64) -> Vec3:
    var ra = ra_deg * 0.017453292519943295
    var dec = dec_deg * 0.017453292519943295
    return Vec3(Float32(cos(dec) * cos(ra)), Float32(cos(dec) * sin(ra)), Float32(sin(dec)))


def galactic_pole_equ() -> Vec3:
    return _equ_vec(192.85948, 27.12825)


def galactic_centre_equ() -> Vec3:
    return _equ_vec(266.40510, -28.93617)


def world_to_equatorial_rows(tod: SotaTimeOfDay) -> Tuple[Vec3, Vec3, Vec3]:
    """Rows of the world -> equatorial rotation (`e = (r0.w, r1.w, r2.w)`): the columns of equatorial -> world,
    which is `_equ_to_world` applied to the three equatorial axes."""
    var lst = tod._lst_deg()
    var c0 = _equ_to_world(0.0, 0.0, lst, tod.latitude_deg, tod.north_azimuth_deg)
    var c1 = _equ_to_world(1.5707963267948966, 0.0, lst, tod.latitude_deg, tod.north_azimuth_deg)
    var c2 = _equ_to_world(0.0, 1.5707963267948966, lst, tod.latitude_deg, tod.north_azimuth_deg)
    return (c0, c1, c2)


def equ_face_st(e: Vec3) -> Tuple[Int, Float32, Float32]:
    """Cube face of an equatorial direction and its (s, t) in [-1, 1]."""
    var ax = abs(e.x)
    var ay = abs(e.y)
    var az = abs(e.z)
    if ax >= ay and ax >= az:
        return (0 if e.x >= Float32(0.0) else 1, e.y / ax, e.z / ax)
    if ay >= az:
        return (2 if e.y >= Float32(0.0) else 3, e.x / ay, e.z / ay)
    return (4 if e.z >= Float32(0.0) else 5, e.x / az, e.y / az)


def equ_from_face_st(face: Int, s: Float32, t: Float32) -> Vec3:
    if face == 0:
        return Vec3(1.0, s, t).normalized()
    if face == 1:
        return Vec3(-1.0, s, t).normalized()
    if face == 2:
        return Vec3(s, 1.0, t).normalized()
    if face == 3:
        return Vec3(s, -1.0, t).normalized()
    if face == 4:
        return Vec3(s, t, 1.0).normalized()
    return Vec3(s, t, -1.0).normalized()


def star_count_brighter(m: Float32) -> Float32:
    """Whole-sky number of stars brighter than V magnitude `m` (the counts the catalogue is drawn from)."""
    return Float32(4800.0) * Float32(exp(Float64(_COUNT_SLOPE * (m - Float32(6.0))) * 2.302585092994046))


def _planck_ratio(c2_over_lambda: Float32, t_k: Float32) -> Float32:
    return Float32((exp(Float64(c2_over_lambda / Float32(5778.0))) - 1.0) / (exp(Float64(c2_over_lambda / t_k)) - 1.0))


def star_colour(t_k: Float32) -> RGB:
    """Planck colour relative to the sun (5 778 K = white), normalised to luma 1."""
    var r = _planck_ratio(23587.0, t_k)
    var g = _planck_ratio(26160.0, t_k)
    var b = _planck_ratio(30942.0, t_k)
    var l = Float32(0.2126) * r + Float32(0.7152) * g + Float32(0.0722) * b
    return RGB(r / l, g / l, b / l, 0.0)


struct StarCell(Copyable, ImplicitlyCopyable, Movable):
    var present: Bool
    var dir_equ: Vec3
    var mag: Float32
    var temp_k: Float32

    def __init__(out self, present: Bool, dir_equ: Vec3, mag: Float32, temp_k: Float32):
        self.present = present
        self.dir_equ = dir_equ
        self.mag = mag
        self.temp_k = temp_k


def _cell_key(face: Int, cx: Int, cy: Int, seed: UInt32) -> UInt32:
    return UInt32((face * SOTA_NIGHT_CELLS + cy) * SOTA_NIGHT_CELLS + cx) ^ (seed * UInt32(0x9E3779B9))


def star_cell_cpu(face: Int, cx: Int, cy: Int, m_lim: Float32, seed: UInt32) -> StarCell:
    var k = _cell_key(face, cx, cy, seed)
    var h0 = hash_u32(k)
    var h1 = hash_u32(h0)
    var h2 = hash_u32(h1)
    var h3 = hash_u32(h2)
    var h4 = hash_u32(h3)
    var inv_n = Float32(2.0) / Float32(SOTA_NIGHT_CELLS)
    var s = (Float32(cx) + Float32(0.15) + Float32(0.7) * hash_unit(h2)) * inv_n - Float32(1.0)
    var t = (Float32(cy) + Float32(0.15) + Float32(0.7) * hash_unit(h3)) * inv_n - Float32(1.0)
    var e = equ_from_face_st(face, s, t)
    var sb = e.dot(galactic_pole_equ())
    var dens = (Float32(0.6) + Float32(1.2) * Float32(exp(Float64(-(sb * sb) / _GAL_SIGMA2)))) * _GAL_NORM
    # Per-cell probability = N(<m_lim) * density * (cell solid angle / 4 pi): the cube cells near a face corner cover
    # 3^1.5 = 5.2x less sky than the centre ones, so the plain 1 / (6 n^2) would crowd the corners.
    var q = Float32(1.0) + s * s + t * t
    var p = star_count_brighter(m_lim) * dens * (Float32(4.0) / Float32(SOTA_NIGHT_CELLS * SOTA_NIGHT_CELLS) / Float32(12.566370614)) / (q * sqrt(q))
    var u1 = max(hash_unit(h1), Float32(1e-7))
    var mag = m_lim + Float32(log(Float64(u1))) / (_COUNT_SLOPE * Float32(2.302585092994046))
    var u4 = hash_unit(h4)
    var temp = Float32(2900.0) + Float32(9100.0) * u4 * sqrt(u4)
    if u4 > Float32(0.96):
        temp = Float32(25000.0)
    return StarCell(hash_unit(h0) < p, e, mag, temp)


def star_radiance_cpu(cell: StarCell, d_equ: Vec3, e0: Float32, sigma_rad: Float32) -> RGB:
    """Radiance of one star's PSF along `d_equ` (before atmospheric extinction)."""
    if not cell.present:
        return RGB(0.0)
    var c = max(Float32(-1.0), min(Float32(1.0), d_equ.dot(cell.dir_equ)))
    var th2 = Float32(2.0) * (Float32(1.0) - c)
    var g = Float32(exp(Float64(-th2 / (Float32(2.0) * sigma_rad * sigma_rad)))) / (Float32(6.283185307) * sigma_rad * sigma_rad)
    var e = e0 * Float32(exp(Float64(-Float32(0.4) * cell.mag) * 2.302585092994046))
    var col = star_colour(cell.temp_k)
    return RGB(col[0] * e * g, col[1] * e * g, col[2] * e * g, 0.0)


def surface_brightness_radiance(e0: Float32, mag_arcsec2: Float32) -> Float32:
    """Radiance (sun units) of a surface brightness in mag/arcsec^2: a magnitude-`mu` source per arcsec^2."""
    return e0 * Float32(exp(Float64(-Float32(0.4) * mag_arcsec2) * 2.302585092994046)) / _ARCSEC2_SR


def _vnoise_hash(ix: Int, iy: Int, iz: Int) -> Float32:
    var k = UInt32((ix & 1023) | ((iy & 1023) << 10) | ((iz & 1023) << 20))
    return hash_unit(hash_u32(k ^ UInt32(0x51ED270B)))


def value_noise3_cpu(p: Vec3) -> Float32:
    """Trilinear value noise on the integer lattice (offset by 512 so the lattice index is non-negative)."""
    var px = p.x + Float32(512.0)
    var py = p.y + Float32(512.0)
    var pz = p.z + Float32(512.0)
    var fx = floor(px)
    var fy = floor(py)
    var fz = floor(pz)
    var ix = Int(fx)
    var iy = Int(fy)
    var iz = Int(fz)
    var tx = px - fx
    var ty = py - fy
    var tz = pz - fz
    tx = tx * tx * (Float32(3.0) - Float32(2.0) * tx)
    ty = ty * ty * (Float32(3.0) - Float32(2.0) * ty)
    tz = tz * tz * (Float32(3.0) - Float32(2.0) * tz)
    var acc = Float32(0.0)
    for k in range(8):
        var dx = k & 1
        var dy = (k >> 1) & 1
        var dz = (k >> 2) & 1
        var w = (tx if dx == 1 else Float32(1.0) - tx) * (ty if dy == 1 else Float32(1.0) - ty) * (tz if dz == 1 else Float32(1.0) - tz)
        acc += w * _vnoise_hash(ix + dx, iy + dy, iz + dz)
    return acc


def night_background_cpu(d_world: Vec3, d_equ: Vec3, bg_l: Float32, mw_l: Float32) -> RGB:
    """Airglow (van Rhijn) + Milky Way along `d_world` / `d_equ`, before extinction."""
    var cz = max(Float32(0.0), d_world.y)
    var vr = Float32(1.0) / sqrt(max(Float32(0.05), Float32(1.0) - _VAN_RHIJN_K * (Float32(1.0) - cz * cz)))
    var air = bg_l * vr
    var sb = d_equ.dot(galactic_pole_equ())
    var cg = d_equ.dot(galactic_centre_equ())
    var band = Float32(exp(Float64(-(sb * sb) / (Float32(2.0) * _MW_SIGMA_B * _MW_SIGMA_B))))
    var core = Float32(0.3) + Float32(0.7) * Float32(exp(Float64((cg - Float32(1.0)) * Float32(2.2))))
    var n = value_noise3_cpu(d_equ * Float32(7.0)) * Float32(0.65) + value_noise3_cpu(d_equ * Float32(17.0)) * Float32(0.35)
    var dust = Float32(0.35) + Float32(0.65) * min(Float32(1.0), max(Float32(0.0), (n - Float32(0.25)) * Float32(1.6)))
    var mw = mw_l * band * core * dust
    return RGB(air * Float32(0.92) + mw * Float32(1.04), air * Float32(1.04) + mw * Float32(0.99), air * Float32(0.86) + mw * Float32(0.97), 0.0)


def night_sky_radiance_cpu(
    d_world: Vec3, rows: Tuple[Vec3, Vec3, Vec3], star_scale: Float32, m_lim: Float32, bg_l: Float32, mw_l: Float32,
    psf_px: Float32, seed: UInt32, e0: Float32, pix_ang: Float32,
) -> RGB:
    """CPU twin of `emit_night_sky`: airglow + Milky Way + the stars of the 2 x 2 cells bracketing `d_world`."""
    var e = Vec3(d_world.dot(rows[0]), d_world.dot(rows[1]), d_world.dot(rows[2]))
    var acc = night_background_cpu(d_world, e, bg_l, mw_l)
    if star_scale <= Float32(0.0):
        return acc
    var fst = equ_face_st(e)
    var nf = Float32(SOTA_NIGHT_CELLS)
    var gx = (fst[1] + Float32(1.0)) * (Float32(0.5) * nf) - Float32(0.5)
    var gy = (fst[2] + Float32(1.0)) * (Float32(0.5) * nf) - Float32(0.5)
    var bx = Int(floor(max(gx, Float32(0.0))))
    var by = Int(floor(max(gy, Float32(0.0))))
    var sigma = max(pix_ang * psf_px, Float32(1e-6))
    for k in range(4):
        var cx = min(bx + (k & 1), SOTA_NIGHT_CELLS - 1)
        var cy = min(by + (k >> 1), SOTA_NIGHT_CELLS - 1)
        var cell = star_cell_cpu(fst[0], cx, cy, m_lim, seed)
        if not cell.present:
            continue
        var c = max(Float32(-1.0), min(Float32(1.0), e.dot(cell.dir_equ)))
        var th2 = Float32(2.0) * (Float32(1.0) - c)
        if Float32(exp(Float64(-th2 / (Float32(2.0) * sigma * sigma)))) <= Float32(1e-3):
            continue
        var r = star_radiance_cpu(cell, e, e0, sigma)
        for ch in range(3):
            acc[ch] = acc[ch] + r[ch] * star_scale
    return acc


# ============================================================================
# DSL
# ============================================================================

def _ghash(x: GpuU32) -> GpuU32:
    var s = x * UInt32(747796405) + UInt32(2891336453)
    var w = ((s >> ((s >> UInt32(28)) + UInt32(4))) ^ s) * UInt32(277803737)
    return (w >> UInt32(22)) ^ w


def _gunit(h: GpuU32) -> GpuF32:
    return (h >> UInt32(8)).to_f32() * Float32(1.0 / 16777216.0)


def _g_planck_ratio(c2l: Float32, t_k: GpuF32) -> GpuF32:
    var num = Float32(exp(Float64(c2l / Float32(5778.0))) - 1.0)
    return num / max_f(exp2_f(Float32(c2l * Float32(1.4426950408889634)) / t_k) - 1.0, 1e-20)


def _g_star_colour(t_k: GpuF32) -> GpuVec3:
    var r = _g_planck_ratio(23587.0, t_k)
    var g = _g_planck_ratio(26160.0, t_k)
    var b = _g_planck_ratio(30942.0, t_k)
    var l = r * 0.2126 + g * 0.7152 + b * 0.0722
    return GpuVec3(r / l, g / l, b / l)


struct _FaceSt(Copyable, Movable):
    var face: GpuU32
    var s: GpuF32
    var t: GpuF32

    def __init__(out self, mut dsl: ShaderDsl, e: GpuVec3):
        var ax = abs_f(e.x())
        var ay = abs_f(e.y())
        var az = abs_f(e.z())
        var use_x = ax.ge(ay) & ax.ge(az)
        var use_y = (~use_x) & ay.ge(az)
        var fx = select_u(e.x().ge(0.0), dsl.u32(0), dsl.u32(1))
        var fy = select_u(e.y().ge(0.0), dsl.u32(2), dsl.u32(3))
        var fz = select_u(e.z().ge(0.0), dsl.u32(4), dsl.u32(5))
        self.face = select_u(use_x, fx, select_u(use_y, fy, fz))
        var m = select_f(use_x, ax, select_f(use_y, ay, az))
        self.s = select_f(use_x, e.y(), e.x()) / m
        self.t = select_f(use_x | use_y, e.z(), e.y()) / m


def _g_equ_from_face_st(mut dsl: ShaderDsl, face: GpuU32, s: GpuF32, t: GpuF32) -> GpuVec3:
    var one = dsl.f32(1.0)
    var sgn = select_f((face & UInt32(1)).eq(UInt32(0)), one, -one)
    var is_x = face.lt(UInt32(2))
    var is_y = (~is_x) & face.lt(UInt32(4))
    var x = select_f(is_x, sgn, s)
    var y = select_f(is_x, s, select_f(is_y, sgn, t))
    var z = select_f(is_x | is_y, t, sgn)
    return GpuVec3(x, y, z).normalized()


def _g_vnoise_hash(ix: GpuU32, iy: GpuU32, iz: GpuU32) -> GpuF32:
    var k = (ix & UInt32(1023)) | ((iy & UInt32(1023)) << UInt32(10)) | ((iz & UInt32(1023)) << UInt32(20))
    return _gunit(_ghash(k ^ UInt32(0x51ED270B)))


def _g_value_noise3(p: GpuVec3) -> GpuF32:
    var px = p.x() + 512.0
    var py = p.y() + 512.0
    var pz = p.z() + 512.0
    var fx = floor_f(px)
    var fy = floor_f(py)
    var fz = floor_f(pz)
    var ix = fx.saturate_u32()
    var iy = fy.saturate_u32()
    var iz = fz.saturate_u32()
    var tx0 = px - fx
    var ty0 = py - fy
    var tz0 = pz - fz
    var tx = tx0 * tx0 * (3.0 - tx0 * 2.0)
    var ty = ty0 * ty0 * (3.0 - ty0 * 2.0)
    var tz = tz0 * tz0 * (3.0 - tz0 * 2.0)
    var acc = tx * 0.0
    for k in range(8):
        var dx = k & 1
        var dy = (k >> 1) & 1
        var dz = (k >> 2) & 1
        var wx = tx if dx == 1 else 1.0 - tx
        var wy = ty if dy == 1 else 1.0 - ty
        var wz = tz if dz == 1 else 1.0 - tz
        acc = acc + wx * wy * wz * _g_vnoise_hash(ix + UInt32(dx), iy + UInt32(dy), iz + UInt32(dz))
    return acc


def emit_night_sky(mut dsl: ShaderDsl, b_glob: BufferRef[1], d: GpuVec3, pix_ang: GpuF32) -> GpuVec3:
    """Stars + airglow + Milky Way radiance along the world direction `d` (before extinction / occlusion)."""
    var g = SOTA_ATMO_G2
    var r0 = b_glob.load_vec3(dsl.u32(UInt32(g + 0)))
    var r1 = b_glob.load_vec3(dsl.u32(UInt32(g + 3)))
    var r2 = b_glob.load_vec3(dsl.u32(UInt32(g + 6)))
    var star_scale = b_glob.load_f32(dsl.u32(UInt32(g + SOTA_NIGHT_W_STARS)))
    var m_lim = b_glob.load_f32(dsl.u32(UInt32(g + SOTA_NIGHT_W_MLIM)))
    var bg_l = b_glob.load_f32(dsl.u32(UInt32(g + SOTA_NIGHT_W_BG)))
    var mw_l = b_glob.load_f32(dsl.u32(UInt32(g + SOTA_NIGHT_W_MW)))
    var psf_px = b_glob.load_f32(dsl.u32(UInt32(g + SOTA_NIGHT_W_PSF)))
    var seed = b_glob.load_u32(dsl.u32(UInt32(g + SOTA_NIGHT_W_SEED)))
    var e0 = b_glob.load_f32(dsl.u32(UInt32(g + SOTA_NIGHT_W_E0)))
    var e = GpuVec3(d.dot(r0), d.dot(r1), d.dot(r2))
    var gp = galactic_pole_equ()
    var gc = galactic_centre_equ()
    var sb = e.x() * gp.x + e.y() * gp.y + e.z() * gp.z
    var cg = e.x() * gc.x + e.y() * gc.y + e.z() * gc.z

    # --- diffuse: airglow (van Rhijn) + Milky Way with dust lanes ---
    var cz = max_f(d.y(), 0.0)
    var vr = 1.0 / sqrt_f(max_f(1.0 - (1.0 - cz * cz) * _VAN_RHIJN_K, 0.05))
    var air = bg_l * vr
    var band = _gexp(-(sb * sb) * Float32(1.0 / (2.0 * _MW_SIGMA_B * _MW_SIGMA_B)))
    var core = _gexp((cg - 1.0) * 2.2) * 0.7 + 0.3
    var n = _g_value_noise3(e * 7.0) * 0.65 + _g_value_noise3(e * 17.0) * 0.35
    var dust = saturate((n - 0.25) * 1.6) * 0.65 + 0.35
    var mw = mw_l * band * core * dust
    var acc = dsl.var_vec3(GpuVec3(air * 0.92 + mw * 1.04, air * 1.04 + mw * 0.99, air * 0.86 + mw * 0.97))

    # --- stars: the 2 x 2 cells bracketing the pixel ---
    dsl.if_then(star_scale.gt(0.0))
    var fs = _FaceSt(dsl, e)
    var nf = Float32(SOTA_NIGHT_CELLS)
    var gx = (fs.s + 1.0) * (0.5 * nf) - 0.5
    var gy = (fs.t + 1.0) * (0.5 * nf) - 0.5
    var bx = floor_f(max_f(gx, 0.0)).saturate_u32()
    var by = floor_f(max_f(gy, 0.0)).saturate_u32()
    var sigma = max_f(pix_ang * psf_px, 1e-6)
    var inv2s2 = 1.0 / (sigma * sigma * 2.0)
    var norm = 1.0 / (sigma * sigma * Float32(6.283185307))
    var p_base = star_count_brighter_g(m_lim) * Float32(4.0 / Float32(SOTA_NIGHT_CELLS * SOTA_NIGHT_CELLS) / 12.566370614)
    var cmax = dsl.u32(UInt32(SOTA_NIGHT_CELLS - 1))
    var seed_mix = seed * UInt32(0x9E3779B9)
    var inv_n = Float32(2.0 / Float32(SOTA_NIGHT_CELLS))
    for k in range(4):
        var cx = min_u(bx + UInt32(k & 1), cmax)
        var cy = min_u(by + UInt32(k >> 1), cmax)
        var key = ((fs.face * UInt32(SOTA_NIGHT_CELLS) + cy) * UInt32(SOTA_NIGHT_CELLS) + cx) ^ seed_mix
        var h0 = _ghash(key)
        var h1 = _ghash(h0)
        var h2 = _ghash(h1)
        var h3 = _ghash(h2)
        var h4 = _ghash(h3)
        var s = (cx.to_f32() + 0.15 + _gunit(h2) * 0.7) * inv_n - 1.0
        var t = (cy.to_f32() + 0.15 + _gunit(h3) * 0.7) * inv_n - 1.0
        var se = _g_equ_from_face_st(dsl, fs.face, s, t)
        var ssb = se.x() * gp.x + se.y() * gp.y + se.z() * gp.z
        var dens = (_gexp(-(ssb * ssb) * Float32(1.0 / _GAL_SIGMA2)) * 1.2 + 0.6) * _GAL_NORM
        var qj = s * s + t * t + 1.0
        dsl.if_then(_gunit(h0).lt(p_base * dens / (qj * sqrt_f(qj))))
        var c = clamp_dot(e.dot(se))
        var th2 = (1.0 - c) * 2.0
        var gw = _gexp(-(th2 * inv2s2)) * norm
        dsl.if_then(gw.gt(1e-3 * norm))
        var u1 = max_f(_gunit(h1), 1e-7)
        # m = m_lim + log10(u1) / 0.45; E = e0 * 10^(-0.4 m)
        var mag = m_lim + log2_g(u1) * Float32(0.30102999566 / 0.45)
        var e_star = e0 * exp2_f(mag * Float32(-0.4 * 3.321928094887362))
        var u4 = _gunit(h4)
        var temp = select_f(u4.gt(0.96), dsl.f32(25000.0), u4 * sqrt_f(u4) * 9100.0 + 2900.0)
        acc.set(acc.get() + _g_star_colour(temp) * (e_star * gw * star_scale))
        dsl.end_if()
        dsl.end_if()
    dsl.end_if()
    return acc.get()


@always_inline
def clamp_dot(c: GpuF32) -> GpuF32:
    return min_f(max_f(c, -1.0), 1.0)


def log2_g(x: GpuF32) -> GpuF32:
    return log2_f(x)


def star_count_brighter_g(m: GpuF32) -> GpuF32:
    return exp2_f((m - 6.0) * Float32(0.45 * 3.321928094887362)) * 4800.0


# ============================================================================
# Settings
# ============================================================================

struct SotaNightSky(Copyable, Movable):
    """Night-sky cvars (`ren.sota_atmo.night`). On by default inside `sota_atmo` (it is invisible in daylight)."""
    var enabled: Bool
    var star_brightness: Float32   # multiplier on the catalogue irradiance
    var limiting_magnitude: Float32
    var airglow_mag_arcsec2: Float32
    var milky_way_mag_arcsec2: Float32
    var airglow_scale: Float32
    var milky_way_scale: Float32
    var star_psf_px: Float32
    var seed: UInt32

    def __init__(out self):
        self.enabled = True
        self.star_brightness = 1.0
        self.limiting_magnitude = 7.0
        self.airglow_mag_arcsec2 = 22.0
        self.milky_way_mag_arcsec2 = 20.6
        self.airglow_scale = 1.0
        self.milky_way_scale = 1.0
        self.star_psf_px = 0.75
        self.seed = 1

    def e0(self, sun_luma: Float32) -> Float32:
        """Irradiance of a magnitude-0 star in sun units."""
        return sun_luma * Float32(exp(Float64(Float32(0.4) * _SUN_MAG) * 2.302585092994046))
