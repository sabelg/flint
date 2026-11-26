"""Area H (sota:atmo): physically based atmosphere, CPU side.

Everything here is the CPU twin of a GPU pass in `sota_atmo_gpu.mojo` (same constants, same mappings, same
step layouts), plus the references the GPU work is checked against:

- `SotaAtmoParams`: Earth-like participating medium in km (Hillaire 2020 defaults): Rayleigh (exponential,
  H = 8 km), Mie (exponential, H = 1.2 km, Cornette-Shanks phase, g = 0.8), ozone (tent profile 10..40 km,
  absorption only), Lambertian ground.
- Transmittance LUT (256 x 64, Bruneton & Neyret 2008 parameterisation), multiple-scattering LUT (32 x 32,
  Hillaire 2020 section 5.5: second order with an isotropic phase, infinite orders via 1 / (1 - f_ms)),
  sky-view LUT (192 x 108, latitude warped about the true horizon of the observer, world azimuth so the
  moon can share it) and the aerial-perspective volume (32 x 32 x 32 froxels along view rays).
- `sota_atmo_mc_sky_radiance`: a brute-force Monte Carlo path tracer of the same medium (delta tracking,
  next-event estimation to the sun with a converged transmittance, Lambertian ground, all scattering orders).
  It is the ground truth for the multiple-scattering approximation.
- `SotaTimeOfDay`: sun and moon positions (Meeus low-precision series, about 0.01 deg for the sun and
  about 0.5 deg for the moon), moon phase, and the irradiance of each at the top of the atmosphere.
- Analytic exponential height fog (the tail beyond the froxel volume).

Sources: E. Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique", EGSR 2020;
E. Bruneton & F. Neyret, "Precomputed Atmospheric Scattering", EGSR 2008; W. Cornette & J. Shanks 1992;
J. Meeus, "Astronomical Algorithms", 2nd ed., ch. 25 / 47 (low-precision series); NOAA solar calculator
notes; E. Hillaire, "Physically Based and Unified Volumetric Rendering in Frostbite", SIGGRAPH 2015 (analytic
slab integration). No code from any SDK was used.
"""
from std.collections import List
from std.math import acos, asin, atan2, cos, exp, floor, log, max, min, sin, sqrt
from flint.core.math import Vec3

comptime RGB = SIMD[DType.float32, 4]

comptime SOTA_ATMO_TRANS_W: Int = 256
comptime SOTA_ATMO_TRANS_H: Int = 64
comptime SOTA_ATMO_MS_DIM: Int = 32
comptime SOTA_ATMO_SKY_W: Int = 192
comptime SOTA_ATMO_SKY_H: Int = 108
comptime SOTA_ATMO_AP_DIM: Int = 32

comptime SOTA_ATMO_TRANS_STEPS: Int = 40
comptime SOTA_ATMO_MS_DIRS: Int = 64
comptime SOTA_ATMO_MS_STEPS: Int = 20
comptime SOTA_ATMO_SKY_STEPS: Int = 32
comptime SOTA_ATMO_AP_SUBSTEPS: Int = 2

comptime _PI: Float32 = 3.14159265358979
comptime _INV_4PI: Float32 = 0.0795774715459


@always_inline
def rgb(r: Float32, g: Float32, b: Float32) -> RGB:
    return RGB(r, g, b, 0.0)


@always_inline
def rgb_exp(v: RGB) -> RGB:
    return RGB(exp(v[0]), exp(v[1]), exp(v[2]), 0.0)


@always_inline
def rgb_luma(v: RGB) -> Float32:
    return v[0] * 0.2126 + v[1] * 0.7152 + v[2] * 0.0722


@always_inline
def _clamp01(x: Float32) -> Float32:
    return max(Float32(0.0), min(Float32(1.0), x))


@always_inline
def _smoothstep(e0: Float32, e1: Float32, x: Float32) -> Float32:
    var t = _clamp01((x - e0) / (e1 - e0))
    return t * t * (Float32(3.0) - Float32(2.0) * t)


struct SotaAtmoParams(Copyable, ImplicitlyCopyable, Movable):
    """Medium in km and km^-1. `solar_irradiance` is the sun's irradiance at the top of the atmosphere in
    the engine's lighting units (4.5 with the AgX exposure defaults, like `BrunetonAtmosphereParams`)."""
    var r_bottom_km: Float32
    var r_top_km: Float32
    var rayleigh_scat: RGB
    var rayleigh_h_km: Float32
    var mie_scat: RGB
    var mie_ext: RGB
    var mie_h_km: Float32
    var mie_g: Float32
    var ozone_abs: RGB
    var ozone_center_km: Float32
    var ozone_half_width_km: Float32
    var ground_albedo: RGB
    var solar_irradiance: RGB
    var sun_angular_radius: Float32
    var moon_angular_radius: Float32
    var moon_albedo: Float32  # Lommel-Seeliger normalisation: full-moon disc radiance = 0.5 * moon_albedo * E
    var sun_limb_u: Float32   # linear limb darkening coefficient (visible band, ~0.6)

    def __init__(out self, turbidity: Float32 = 1.0, solar_intensity: Float32 = 4.5):
        self.r_bottom_km = 6360.0
        self.r_top_km = 6460.0
        self.rayleigh_scat = rgb(5.802e-3, 13.558e-3, 33.1e-3)
        self.rayleigh_h_km = 8.0
        var t = max(Float32(0.0), turbidity)
        self.mie_scat = rgb(3.996e-3, 3.996e-3, 3.996e-3) * t
        self.mie_ext = rgb(4.40e-3, 4.40e-3, 4.40e-3) * t
        self.mie_h_km = 1.2
        self.mie_g = 0.8
        self.ozone_abs = rgb(0.650e-3, 1.881e-3, 0.085e-3)
        self.ozone_center_km = 25.0
        self.ozone_half_width_km = 15.0
        self.ground_albedo = rgb(0.3, 0.3, 0.3)
        self.solar_irradiance = rgb(1.0, 0.965, 0.925) * solar_intensity
        self.sun_angular_radius = 0.004653   # 0.2666 deg
        self.moon_angular_radius = 0.004520  # 0.259 deg (mean)
        self.moon_albedo = 0.047             # full-moon disc / sun disc radiance ~ 1.6e-6 (2.5e3 vs 1.6e9 cd/m^2)
        self.sun_limb_u = 0.6

    def signature(self) -> Float64:
        """Changes whenever a parameter that the transmittance / multi-scattering LUTs depend on changes."""
        var s = Float64(self.r_bottom_km) * 1.3 + Float64(self.r_top_km) * 1.7 + Float64(self.rayleigh_h_km) * 3.1
        s += Float64(self.mie_h_km) * 5.3 + Float64(self.mie_g) * 7.9 + Float64(self.ozone_center_km) * 0.37
        s += Float64(self.ozone_half_width_km) * 0.53
        for c in range(3):
            s += Float64(self.rayleigh_scat[c]) * Float64(1000.0 + 17.0 * Float64(c))
            s += Float64(self.mie_scat[c]) * Float64(1300.0 + 19.0 * Float64(c))
            s += Float64(self.mie_ext[c]) * Float64(1700.0 + 23.0 * Float64(c))
            s += Float64(self.ozone_abs[c]) * Float64(2900.0 + 29.0 * Float64(c))
            s += Float64(self.ground_albedo[c]) * Float64(31.0 + 3.0 * Float64(c))
        return s


# ============================================================================
# Medium, phase functions, geometry
# ============================================================================

@always_inline
def sota_atmo_densities(h_km: Float32, p: SotaAtmoParams) -> Tuple[Float32, Float32, Float32]:
    var h = max(Float32(0.0), h_km)
    var rr = exp(-h / p.rayleigh_h_km)
    var rm = exp(-h / p.mie_h_km)
    var ro = max(Float32(0.0), Float32(1.0) - abs(h - p.ozone_center_km) / p.ozone_half_width_km)
    return (rr, rm, ro)


@always_inline
def sota_atmo_extinction(h_km: Float32, p: SotaAtmoParams) -> RGB:
    var d = sota_atmo_densities(h_km, p)
    return p.rayleigh_scat * d[0] + p.mie_ext * d[1] + p.ozone_abs * d[2]


@always_inline
def rayleigh_phase(mu: Float32) -> Float32:
    return Float32(3.0 / (16.0 * 3.14159265358979)) * (Float32(1.0) + mu * mu)


@always_inline
def cornette_shanks_phase(mu: Float32, g: Float32) -> Float32:
    var g2 = g * g
    var num = Float32(3.0 / (8.0 * 3.14159265358979)) * (Float32(1.0) - g2) * (Float32(1.0) + mu * mu)
    var base = max(Float32(1e-4), Float32(1.0) + g2 - Float32(2.0) * g * mu)
    return num / ((Float32(2.0) + g2) * base * sqrt(base))


@always_inline
def ray_top_dist(r: Float32, mu: Float32, r_top: Float32) -> Float32:
    var disc = r * r * (mu * mu - Float32(1.0)) + r_top * r_top
    return max(Float32(0.0), -r * mu + sqrt(max(Float32(0.0), disc)))


@always_inline
def ray_ground_dist(r: Float32, mu: Float32, r_bot: Float32) -> Float32:
    """Distance to the ground sphere, or -1 when the ray misses it."""
    if mu >= Float32(0.0):
        return -1.0
    var disc = r * r * (mu * mu - Float32(1.0)) + r_bot * r_bot
    if disc < Float32(0.0):
        return -1.0
    var d = -r * mu - sqrt(disc)
    return d if d > Float32(0.0) else Float32(-1.0)


@always_inline
def horizon_mu(r: Float32, r_bot: Float32) -> Float32:
    """Cosine of the zenith angle of the horizon seen from radius `r` (negative above the ground)."""
    var s = min(Float32(1.0), r_bot / max(r_bot, r))
    return -sqrt(max(Float32(0.0), Float32(1.0) - s * s))


@always_inline
def sun_visibility(r: Float32, mu_s: Float32, p: SotaAtmoParams) -> Float32:
    """Earth shadow of the sun disc (linear over the disc's angular diameter about the horizon)."""
    var mh = horizon_mu(r, p.r_bottom_km)
    var sin_h = sqrt(max(Float32(0.0), Float32(1.0) - mh * mh))
    var w = p.sun_angular_radius * sin_h
    return _clamp01((mu_s - mh + w) / max(Float32(1e-6), Float32(2.0) * w))


@always_inline
def quad_step_t(i: Int, n: Int, t_max: Float32) -> Float32:
    """Endpoint `i` of the quadratically spaced march (fine steps near the start of the ray)."""
    var u = Float32(i) / Float32(n)
    return t_max * u * u


def optical_depth(r: Float32, mu: Float32, t_max: Float32, steps: Int, p: SotaAtmoParams) -> RGB:
    var tau = RGB(0.0)
    for i in range(steps):
        var t0 = quad_step_t(i, steps, t_max)
        var t1 = quad_step_t(i + 1, steps, t_max)
        var t = Float32(0.5) * (t0 + t1)
        var ri = sqrt(max(Float32(0.0), r * r + t * t + Float32(2.0) * r * mu * t))
        tau += sota_atmo_extinction(ri - p.r_bottom_km, p) * (t1 - t0)
    return tau


# ============================================================================
# Transmittance LUT (Bruneton 2008 mapping: x_mu over the distance to the top, x_r = rho / H)
# ============================================================================

@always_inline
def trans_lut_uv(r_in: Float32, mu: Float32, p: SotaAtmoParams) -> Tuple[Float32, Float32]:
    var rb = p.r_bottom_km
    var rt = p.r_top_km
    var r = max(rb, min(rt, r_in))
    var hh = sqrt(rt * rt - rb * rb)
    var rho = sqrt(max(Float32(0.0), r * r - rb * rb))
    var d = ray_top_dist(r, mu, rt)
    var d_min = rt - r
    var d_max = rho + hh
    var x_mu = (d - d_min) / max(Float32(1e-6), d_max - d_min)
    var x_r = rho / hh
    return (x_mu, x_r)


@always_inline
def trans_lut_r_mu(x_mu: Float32, x_r: Float32, p: SotaAtmoParams) -> Tuple[Float32, Float32]:
    var rb = p.r_bottom_km
    var rt = p.r_top_km
    var hh = sqrt(rt * rt - rb * rb)
    var rho = hh * x_r
    var r = sqrt(rho * rho + rb * rb)
    var d_min = rt - r
    var d_max = rho + hh
    var d = d_min + x_mu * (d_max - d_min)
    var mu = Float32(1.0) if d <= Float32(1e-4) else (hh * hh - rho * rho - d * d) / (Float32(2.0) * r * d)
    return (r, max(Float32(-1.0), min(Float32(1.0), mu)))


@always_inline
def _texel_coord(x01: Float32, n: Int) -> Float32:
    """Unit coordinate -> continuous texel index with texel centres at the ends of the range."""
    return _clamp01(x01) * Float32(n - 1)


def bilerp_rgb(lut: List[RGB], w: Int, h: Int, fx_in: Float32, fy_in: Float32, wrap_x: Bool = False) -> RGB:
    """`fx`, `fy` in texel units (texel centres at integers)."""
    var fx = fx_in
    if wrap_x:
        fx = fx - floor(fx / Float32(w)) * Float32(w)
    else:
        fx = max(Float32(0.0), min(Float32(w - 1), fx))
    var fy = max(Float32(0.0), min(Float32(h - 1), fy_in))
    var x0 = min(w - 1, Int(fx))
    var y0 = min(h - 1, Int(fy))
    var x1 = (x0 + 1) % w if wrap_x else min(w - 1, x0 + 1)
    var y1 = min(h - 1, y0 + 1)
    var tx = fx - Float32(x0)
    var ty = fy - Float32(y0)
    var a = lut[y0 * w + x0] * (Float32(1.0) - tx) + lut[y0 * w + x1] * tx
    var b = lut[y1 * w + x0] * (Float32(1.0) - tx) + lut[y1 * w + x1] * tx
    return a * (Float32(1.0) - ty) + b * ty


def bake_transmittance_lut(p: SotaAtmoParams) -> List[RGB]:
    var lut = List[RGB](length=SOTA_ATMO_TRANS_W * SOTA_ATMO_TRANS_H, fill=RGB(1.0))
    for y in range(SOTA_ATMO_TRANS_H):
        for x in range(SOTA_ATMO_TRANS_W):
            var rm = trans_lut_r_mu(Float32(x) / Float32(SOTA_ATMO_TRANS_W - 1), Float32(y) / Float32(SOTA_ATMO_TRANS_H - 1), p)
            var tau = optical_depth(rm[0], rm[1], ray_top_dist(rm[0], rm[1], p.r_top_km), SOTA_ATMO_TRANS_STEPS, p)
            lut[y * SOTA_ATMO_TRANS_W + x] = rgb_exp(-tau)
    return lut^


def sample_transmittance(lut: List[RGB], r: Float32, mu: Float32, p: SotaAtmoParams) -> RGB:
    """Transmittance to the top of the atmosphere. Below the horizon the ray is evaluated at the horizon
    (callers multiply by `sun_visibility`)."""
    var m = max(mu, horizon_mu(r, p.r_bottom_km))
    var uv = trans_lut_uv(r, m, p)
    return bilerp_rgb(lut, SOTA_ATMO_TRANS_W, SOTA_ATMO_TRANS_H, _texel_coord(uv[0], SOTA_ATMO_TRANS_W), _texel_coord(uv[1], SOTA_ATMO_TRANS_H))


def sun_transmittance_exact(r: Float32, mu_s: Float32, p: SotaAtmoParams, steps: Int = 256) -> RGB:
    """Converged transmittance to the top (reference; no LUT)."""
    if ray_ground_dist(r, mu_s, p.r_bottom_km) > Float32(0.0):
        return RGB(0.0)
    return rgb_exp(-optical_depth(r, mu_s, ray_top_dist(r, mu_s, p.r_top_km), steps, p))


# ============================================================================
# Multiple-scattering LUT (Hillaire 2020): Psi_ms(r, mu_s) = L_2nd / (1 - f_ms), per unit sun irradiance
# ============================================================================

@always_inline
def ms_dir(i: Int, n: Int) -> Vec3:
    """Fibonacci sphere direction `i` of `n` (uniform over the sphere)."""
    var z = Float32(1.0) - (Float32(2 * i + 1) / Float32(n))
    var rr = sqrt(max(Float32(0.0), Float32(1.0) - z * z))
    var phi = Float32(i) * Float32(2.39996322972865)
    return Vec3(rr * cos(phi), z, rr * sin(phi))


def bake_multiscatter_lut(p: SotaAtmoParams, trans: List[RGB], dirs: Int = SOTA_ATMO_MS_DIRS, steps: Int = SOTA_ATMO_MS_STEPS) -> List[RGB]:
    var lut = List[RGB](length=SOTA_ATMO_MS_DIM * SOTA_ATMO_MS_DIM, fill=RGB(0.0))
    for y in range(SOTA_ATMO_MS_DIM):
        for x in range(SOTA_ATMO_MS_DIM):
            lut[y * SOTA_ATMO_MS_DIM + x] = multiscatter_texel(x, y, p, trans, dirs, steps)
    return lut^


def multiscatter_texel(x: Int, y: Int, p: SotaAtmoParams, trans: List[RGB], dirs: Int, steps: Int) -> RGB:
    var rb = p.r_bottom_km
    var rt = p.r_top_km
    var mu_s = (Float32(x) + Float32(0.5)) / Float32(SOTA_ATMO_MS_DIM) * Float32(2.0) - Float32(1.0)
    var r = rb + (Float32(y) + Float32(0.5)) / Float32(SOTA_ATMO_MS_DIM) * (rt - rb)
    var sun = Vec3(sqrt(max(Float32(0.0), Float32(1.0) - mu_s * mu_s)), mu_s, 0.0)
    var l2 = RGB(0.0)
    var fms = RGB(0.0)
    for d_i in range(dirs):
        var d = ms_dir(d_i, dirs)
        var tg = ray_ground_dist(r, d.y, rb)
        var t_max = tg if tg > Float32(0.0) else ray_top_dist(r, d.y, rt)
        var thr = RGB(1.0)
        var lum = RGB(0.0)
        var f = RGB(0.0)
        for s_i in range(steps):
            var t0 = quad_step_t(s_i, steps, t_max)
            var t1 = quad_step_t(s_i + 1, steps, t_max)
            var t = Float32(0.5) * (t0 + t1)
            var dt = t1 - t0
            var px = d.x * t
            var py = r + d.y * t
            var pz = d.z * t
            var ri = sqrt(px * px + py * py + pz * pz)
            var dens = sota_atmo_densities(ri - rb, p)
            var sig_s = p.rayleigh_scat * dens[0] + p.mie_scat * dens[1]
            var sig_t = p.rayleigh_scat * dens[0] + p.mie_ext * dens[1] + p.ozone_abs * dens[2]
            var tr_step = rgb_exp(-sig_t * dt)
            var mu_l = (px * sun.x + py * sun.y + pz * sun.z) / ri
            var t_sun = sample_transmittance(trans, ri, mu_l, p) * sun_visibility(ri, mu_l, p)
            var integ = RGB(0.0)
            for c in range(3):
                integ[c] = (Float32(1.0) - tr_step[c]) / max(Float32(1e-9), sig_t[c])
            lum += thr * sig_s * t_sun * _INV_4PI * integ
            f += thr * sig_s * integ
            thr *= tr_step
        if tg > Float32(0.0):
            var gx = d.x * tg
            var gy = r + d.y * tg
            var gz = d.z * tg
            var n_dot_l = max(Float32(0.0), (gx * sun.x + gy * sun.y + gz * sun.z) / rb)
            var t_sun_g = sample_transmittance(trans, rb, n_dot_l, p) * sun_visibility(rb, n_dot_l, p)
            lum += thr * t_sun_g * p.ground_albedo * (n_dot_l / _PI)
        l2 += lum
        fms += f
    l2 = l2 * (Float32(1.0) / Float32(dirs))
    fms = fms * (Float32(1.0) / Float32(dirs))
    var out = RGB(0.0)
    for c in range(3):
        out[c] = l2[c] / max(Float32(1e-3), Float32(1.0) - fms[c])
    return out


def sample_multiscatter(lut: List[RGB], r: Float32, mu_s: Float32, p: SotaAtmoParams) -> RGB:
    var u = _clamp01(Float32(0.5) + Float32(0.5) * mu_s)
    var v = _clamp01((r - p.r_bottom_km) / (p.r_top_km - p.r_bottom_km))
    return bilerp_rgb(lut, SOTA_ATMO_MS_DIM, SOTA_ATMO_MS_DIM, u * Float32(SOTA_ATMO_MS_DIM) - Float32(0.5), v * Float32(SOTA_ATMO_MS_DIM) - Float32(0.5))


# ============================================================================
# Light sources for a frame (sun + moon), single + multiple scattering along a segment
# ============================================================================

struct SotaAtmoLights(Copyable, ImplicitlyCopyable, Movable):
    """Directions are world space (Y up). Irradiances are at the top of the atmosphere."""
    var sun_dir: Vec3
    var sun_e: RGB
    var moon_dir: Vec3
    var moon_e: RGB

    def __init__(out self, sun_dir: Vec3, sun_e: RGB, moon_dir: Vec3 = Vec3(0.0, -1.0, 0.0), moon_e: RGB = RGB(0.0)):
        self.sun_dir = sun_dir.normalized()
        self.sun_e = sun_e
        self.moon_dir = moon_dir.normalized()
        self.moon_e = moon_e


struct SotaAtmoLuts(Movable):
    """CPU twins of the GPU LUTs (the transmittance and multi-scattering LUTs depend only on the medium)."""
    var params: SotaAtmoParams
    var trans: List[RGB]
    var ms: List[RGB]

    def __init__(out self, p: SotaAtmoParams):
        self.params = p
        self.trans = bake_transmittance_lut(p)
        self.ms = bake_multiscatter_lut(p, self.trans)


def _light_source_term(
    luts: SotaAtmoLuts, ri: Float32, mu_l: Float32, cos_view_l: Float32, sig_r: RGB, sig_m: RGB, e: RGB, with_ms: Bool
) -> RGB:
    var p = luts.params
    var t_l = sample_transmittance(luts.trans, ri, mu_l, p) * sun_visibility(ri, mu_l, p)
    var s = (sig_r * rayleigh_phase(cos_view_l) + sig_m * cornette_shanks_phase(cos_view_l, p.mie_g)) * t_l
    if with_ms:
        s += (sig_r + sig_m) * sample_multiscatter(luts.ms, ri, mu_l, p)
    return s * e


def integrate_segment(
    luts: SotaAtmoLuts, lights: SotaAtmoLights, r0: Float32, d: Vec3, t_max: Float32, steps: Int, with_ms: Bool = True
) -> Tuple[RGB, RGB]:
    """In-scattered radiance and transmittance along `[0, t_max]` from radius `r0` (planet frame: observer at
    `(0, r0, 0)`) in direction `d`. Quadratic steps, analytic per-step integration (Hillaire 2015)."""
    var p = luts.params
    var rb = p.r_bottom_km
    var thr = RGB(1.0)
    var l = RGB(0.0)
    var cvs = d.dot(lights.sun_dir)
    var cvm = d.dot(lights.moon_dir)
    var has_moon = (lights.moon_e[0] + lights.moon_e[1] + lights.moon_e[2]) > Float32(0.0)
    for i in range(steps):
        var t0 = quad_step_t(i, steps, t_max)
        var t1 = quad_step_t(i + 1, steps, t_max)
        var t = Float32(0.5) * (t0 + t1)
        var dt = t1 - t0
        var px = d.x * t
        var py = r0 + d.y * t
        var pz = d.z * t
        var ri = sqrt(px * px + py * py + pz * pz)
        var dens = sota_atmo_densities(ri - rb, p)
        var sig_r = p.rayleigh_scat * dens[0]
        var sig_m = p.mie_scat * dens[1]
        var sig_t = sig_r + p.mie_ext * dens[1] + p.ozone_abs * dens[2]
        var mu_s = (px * lights.sun_dir.x + py * lights.sun_dir.y + pz * lights.sun_dir.z) / ri
        var src = _light_source_term(luts, ri, mu_s, cvs, sig_r, sig_m, lights.sun_e, with_ms)
        if has_moon:
            var mu_m = (px * lights.moon_dir.x + py * lights.moon_dir.y + pz * lights.moon_dir.z) / ri
            src += _light_source_term(luts, ri, mu_m, cvm, sig_r, sig_m, lights.moon_e, with_ms)
        var tr_step = rgb_exp(-sig_t * dt)
        for c in range(3):
            l[c] += thr[c] * src[c] * (Float32(1.0) - tr_step[c]) / max(Float32(1e-9), sig_t[c])
        thr *= tr_step
    return (l, thr)


def sky_radiance(
    luts: SotaAtmoLuts, lights: SotaAtmoLights, alt_km: Float32, d_in: Vec3, steps: Int = SOTA_ATMO_SKY_STEPS, with_ms: Bool = True
) -> RGB:
    """Radiance of the sky (no sun / moon disc) seen from altitude `alt_km` along `d_in`, including the lit
    ground below the horizon. Exactly what one sky-view LUT texel stores."""
    var p = luts.params
    var d = d_in.normalized()
    var r0 = p.r_bottom_km + max(Float32(0.001), alt_km)
    r0 = min(r0, p.r_top_km - Float32(0.001))
    var tg = ray_ground_dist(r0, d.y, p.r_bottom_km)
    var t_max = tg if tg > Float32(0.0) else ray_top_dist(r0, d.y, p.r_top_km)
    var seg = integrate_segment(luts, lights, r0, d, t_max, steps, with_ms)
    var l = seg[0]
    if tg > Float32(0.0):
        var gx = d.x * tg
        var gy = r0 + d.y * tg
        var gz = d.z * tg
        var rb = p.r_bottom_km
        var ns = max(Float32(0.0), (gx * lights.sun_dir.x + gy * lights.sun_dir.y + gz * lights.sun_dir.z) / rb)
        var e_g = sample_transmittance(luts.trans, rb, ns, p) * sun_visibility(rb, ns, p) * lights.sun_e * (ns / _PI)
        var nm = max(Float32(0.0), (gx * lights.moon_dir.x + gy * lights.moon_dir.y + gz * lights.moon_dir.z) / rb)
        e_g += sample_transmittance(luts.trans, rb, nm, p) * sun_visibility(rb, nm, p) * lights.moon_e * (nm / _PI)
        l += seg[1] * e_g * p.ground_albedo
    return l


# ============================================================================
# Sky-view LUT mapping (latitude warped about the true horizon; world azimuth)
# ============================================================================

@always_inline
def sky_view_dir(u: Float32, v: Float32, r: Float32, p: SotaAtmoParams) -> Vec3:
    """Direction for LUT coordinates `u`, `v` in [0, 1]. `v = 0.5` is the true horizon of radius `r`."""
    var half_pi = Float32(1.5707963268)
    var el_h = -acos(min(Float32(1.0), p.r_bottom_km / r))  # dip of the horizon below the horizontal
    var el: Float32
    if v < Float32(0.5):
        var t = Float32(1.0) - Float32(2.0) * v
        el = el_h + (half_pi - el_h) * t * t
    else:
        var t = Float32(2.0) * v - Float32(1.0)
        el = el_h - (half_pi + el_h) * t * t
    var az = (u - Float32(0.5)) * Float32(6.28318530718)
    var ce = cos(el)
    return Vec3(ce * cos(az), sin(el), ce * sin(az))


@always_inline
def sky_view_uv(d: Vec3, r: Float32, p: SotaAtmoParams) -> Tuple[Float32, Float32]:
    var half_pi = Float32(1.5707963268)
    var el_h = -acos(min(Float32(1.0), p.r_bottom_km / r))
    var horiz = sqrt(max(Float32(0.0), d.x * d.x + d.z * d.z))
    var el = atan2(d.y, horiz)
    var v: Float32
    if el >= el_h:
        v = Float32(0.5) - Float32(0.5) * sqrt(_clamp01((el - el_h) / (half_pi - el_h)))
    else:
        v = Float32(0.5) + Float32(0.5) * sqrt(_clamp01((el_h - el) / (half_pi + el_h)))
    var u = atan2(d.z, d.x) * Float32(0.159154943) + Float32(0.5)
    return (u, v)


def bake_sky_view_lut(luts: SotaAtmoLuts, lights: SotaAtmoLights, alt_km: Float32, w: Int = SOTA_ATMO_SKY_W, h: Int = SOTA_ATMO_SKY_H) -> List[RGB]:
    var p = luts.params
    var r = min(p.r_top_km - Float32(0.001), p.r_bottom_km + max(Float32(0.001), alt_km))
    var lut = List[RGB](length=w * h, fill=RGB(0.0))
    for y in range(h):
        for x in range(w):
            var d = sky_view_dir((Float32(x) + Float32(0.5)) / Float32(w), (Float32(y) + Float32(0.5)) / Float32(h), r, p)
            lut[y * w + x] = sky_radiance(luts, lights, alt_km, d)
    return lut^


def sample_sky_view(lut: List[RGB], d: Vec3, alt_km: Float32, p: SotaAtmoParams, w: Int = SOTA_ATMO_SKY_W, h: Int = SOTA_ATMO_SKY_H) -> RGB:
    var r = min(p.r_top_km - Float32(0.001), p.r_bottom_km + max(Float32(0.001), alt_km))
    var uv = sky_view_uv(d.normalized(), r, p)
    return bilerp_rgb(lut, w, h, uv[0] * Float32(w) - Float32(0.5), uv[1] * Float32(h) - Float32(0.5), wrap_x=True)


# ============================================================================
# Sun and moon discs
# ============================================================================

def sun_disc_radiance(luts: SotaAtmoLuts, lights: SotaAtmoLights, alt_km: Float32, d: Vec3) -> RGB:
    """Radiance of the solar disc along `d` (0 off the disc): linear limb darkening normalised so the
    disc integrates to the irradiance, attenuated by the transmittance along `d`."""
    var p = luts.params
    var c = max(Float32(-1.0), min(Float32(1.0), d.normalized().dot(lights.sun_dir)))
    var ang = acos(c)
    if ang >= p.sun_angular_radius:
        return RGB(0.0)
    var r0 = p.r_bottom_km + max(Float32(0.001), alt_km)
    if ray_ground_dist(r0, d.y, p.r_bottom_km) > Float32(0.0):
        return RGB(0.0)
    var omega = Float32(2.0) * _PI * (Float32(1.0) - cos(p.sun_angular_radius))
    var s = ang / p.sun_angular_radius
    var mu_disc = sqrt(max(Float32(0.0), Float32(1.0) - s * s))
    var limb = (Float32(1.0) - p.sun_limb_u * (Float32(1.0) - mu_disc)) / (Float32(1.0) - p.sun_limb_u / Float32(3.0))
    return lights.sun_e * (limb / omega) * sample_transmittance(luts.trans, r0, d.normalized().y, p)


def moon_disc_radiance(luts: SotaAtmoLuts, sun_e: RGB, sun_dir: Vec3, moon_dir: Vec3, alt_km: Float32, d: Vec3) -> RGB:
    """Lommel-Seeliger moon: the sphere point under `d` is lit by the sun from `sun_dir` (the sun is far, so
    the direction from the moon to the sun equals the one from the observer)."""
    var p = luts.params
    var dn = d.normalized()
    var m = moon_dir.normalized()
    var c = max(Float32(-1.0), min(Float32(1.0), dn.dot(m)))
    var ang = acos(c)
    if ang >= p.moon_angular_radius:
        return RGB(0.0)
    # Sphere normal: offset on the disc + the component toward the observer.
    var s = sin(ang) / sin(p.moon_angular_radius)
    var off = dn - m * c
    var ol = off.length()
    var tang = off * (Float32(1.0) / ol) if ol > Float32(1e-9) else Vec3(0.0, 0.0, 0.0)
    var nz = sqrt(max(Float32(0.0), Float32(1.0) - s * s))
    var n = tang * s - m * nz  # facing the observer: -m
    var mu0 = max(Float32(0.0), n.dot(sun_dir.normalized()))
    var mu = max(Float32(1e-3), nz)
    var r0 = p.r_bottom_km + max(Float32(0.001), alt_km)
    if ray_ground_dist(r0, dn.y, p.r_bottom_km) > Float32(0.0):
        return RGB(0.0)
    var ls = p.moon_albedo * mu0 / (mu0 + mu)
    return sun_e * ls * sample_transmittance(luts.trans, r0, dn.y, p)


def moon_irradiance(sun_e: RGB, phase_angle_rad: Float32, p: SotaAtmoParams) -> RGB:
    """Irradiance of moonlight at the top of the atmosphere: the Lommel-Seeliger disc integrated over its
    solid angle (numerically, phase angle 0 = full moon)."""
    var sd = Vec3(sin(phase_angle_rad), 0.0, -cos(phase_angle_rad))  # sun as seen from the moon's frame
    var acc = Float32(0.0)
    var n = 24
    for j in range(n):
        for i in range(n):
            var x = (Float32(i) + Float32(0.5)) / Float32(n) * Float32(2.0) - Float32(1.0)
            var y = (Float32(j) + Float32(0.5)) / Float32(n) * Float32(2.0) - Float32(1.0)
            var rr = x * x + y * y
            if rr >= Float32(1.0):
                continue
            var nz = sqrt(Float32(1.0) - rr)
            var nrm = Vec3(x, y, -nz)
            var mu0 = max(Float32(0.0), nrm.dot(sd))
            acc += p.moon_albedo * mu0 / (mu0 + nz) * (Float32(4.0) / Float32(n * n))
    var r = p.moon_angular_radius
    return sun_e * (acc * r * r)  # disc area pi r^2 * mean radiance = sum * r^2 (unit disc area element 4/n^2)


# ============================================================================
# Aerial-perspective volume (32^3 froxels along view rays, quadratic slice distances)
# ============================================================================

@always_inline
def ap_slice_dist_km(s: Int, max_km: Float32) -> Float32:
    """Far distance of slice `s` (0-based); slice `s` covers `[d(s-1), d(s)]` with `d(-1) = 0`."""
    var u = Float32(s + 1) / Float32(SOTA_ATMO_AP_DIM)
    return max_km * u * u


@always_inline
def ap_slice_coord(dist_km: Float32, max_km: Float32) -> Float32:
    """Continuous slice coordinate: 0 at the far end of slice 0 ... N-1 at `max_km`; negative below slice 0."""
    return sqrt(max(Float32(0.0), dist_km / max_km)) * Float32(SOTA_ATMO_AP_DIM) - Float32(1.0)


def ap_column_cpu(luts: SotaAtmoLuts, lights: SotaAtmoLights, alt_km: Float32, d: Vec3, max_km: Float32) -> List[RGB]:
    """One froxel column: entry `2 s` = in-scatter, `2 s + 1` = transmittance from the eye to slice end `s`."""
    var p = luts.params
    var r0 = min(p.r_top_km - Float32(0.001), p.r_bottom_km + max(Float32(0.001), alt_km))
    var out = List[RGB](length=2 * SOTA_ATMO_AP_DIM, fill=RGB(0.0))
    var l = RGB(0.0)
    var thr = RGB(1.0)
    var t_prev = Float32(0.0)
    var dn = d.normalized()
    for s in range(SOTA_ATMO_AP_DIM):
        var t_end = ap_slice_dist_km(s, max_km)
        for k in range(SOTA_ATMO_AP_SUBSTEPS):
            var ta = t_prev + (t_end - t_prev) * Float32(k) / Float32(SOTA_ATMO_AP_SUBSTEPS)
            var tb = t_prev + (t_end - t_prev) * Float32(k + 1) / Float32(SOTA_ATMO_AP_SUBSTEPS)
            # Single-step segment [ta, tb] from the eye: reuse the segment integrator on the sub-interval.
            var t = Float32(0.5) * (ta + tb)
            var px = dn.x * t
            var py = r0 + dn.y * t
            var pz = dn.z * t
            var ri = sqrt(px * px + py * py + pz * pz)
            var dens = sota_atmo_densities(ri - p.r_bottom_km, p)
            var sig_r = p.rayleigh_scat * dens[0]
            var sig_m = p.mie_scat * dens[1]
            var sig_t = sig_r + p.mie_ext * dens[1] + p.ozone_abs * dens[2]
            var mu_s = (px * lights.sun_dir.x + py * lights.sun_dir.y + pz * lights.sun_dir.z) / ri
            var src = _light_source_term(luts, ri, mu_s, dn.dot(lights.sun_dir), sig_r, sig_m, lights.sun_e, True)
            var mu_m = (px * lights.moon_dir.x + py * lights.moon_dir.y + pz * lights.moon_dir.z) / ri
            src += _light_source_term(luts, ri, mu_m, dn.dot(lights.moon_dir), sig_r, sig_m, lights.moon_e, True)
            var tr_step = rgb_exp(-sig_t * (tb - ta))
            for c in range(3):
                l[c] += thr[c] * src[c] * (Float32(1.0) - tr_step[c]) / max(Float32(1e-9), sig_t[c])
            thr *= tr_step
        out[2 * s] = l
        out[2 * s + 1] = thr
        t_prev = t_end
    return out^


def apply_ap_cpu(column: List[RGB], color: RGB, dist_km: Float32, max_km: Float32) -> RGB:
    """`color * T + S` with the linear slice interpolation used by the GPU apply pass."""
    var sc = ap_slice_coord(dist_km, max_km)
    var s_val: RGB
    var t_val: RGB
    if sc <= Float32(0.0):
        var w = max(Float32(0.0), sc + Float32(1.0))
        s_val = column[0] * w
        t_val = RGB(1.0) * (Float32(1.0) - w) + column[1] * w
    else:
        var s0 = min(SOTA_ATMO_AP_DIM - 1, Int(sc))
        var s1 = min(SOTA_ATMO_AP_DIM - 1, s0 + 1)
        var f = min(Float32(1.0), sc - Float32(s0))
        s_val = column[2 * s0] * (Float32(1.0) - f) + column[2 * s1] * f
        t_val = column[2 * s0 + 1] * (Float32(1.0) - f) + column[2 * s1 + 1] * f
    return color * t_val + s_val


# ============================================================================
# Analytic exponential height fog (the tail beyond the froxel volume)
# ============================================================================

def height_fog_optical_depth(density: Float32, falloff: Float32, base_y: Float32, cam_y: Float32, dir_y: Float32, t0: Float32, t1: Float32) -> Float32:
    """Integral of `density * exp(-falloff * max(0, y - base_y))` over `t` in `[t0, t1]` along a ray from
    height `cam_y` with vertical direction component `dir_y` (the part of the ray below `base_y` has the
    constant base density)."""
    if t1 <= t0 or density <= Float32(0.0):
        return 0.0
    var k = max(Float32(1e-6), falloff)
    # Split at the crossing of base_y.
    var tc: Float32 = -1.0
    if abs(dir_y) > Float32(1e-6):
        tc = (base_y - cam_y) / dir_y
    var acc = Float32(0.0)
    var a = t0
    if tc > t0 and tc < t1:
        acc += _hf_piece(density, k, base_y, cam_y, dir_y, a, tc)
        a = tc
    acc += _hf_piece(density, k, base_y, cam_y, dir_y, a, t1)
    return acc


def _hf_piece(density: Float32, k: Float32, base_y: Float32, cam_y: Float32, dir_y: Float32, a: Float32, b: Float32) -> Float32:
    var ym = cam_y + dir_y * Float32(0.5) * (a + b)
    if ym <= base_y:
        return density * (b - a)
    var e0 = exp(-k * (cam_y + dir_y * a - base_y))
    var kd = k * dir_y
    if abs(kd * (b - a)) < Float32(1e-4):
        return density * e0 * (b - a)
    return density * e0 * (Float32(1.0) - exp(-kd * (b - a))) / kd


# ============================================================================
# Monte Carlo reference (all scattering orders, same medium and geometry)
# ============================================================================

struct _Rng(Copyable, Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed * UInt64(6364136223846793005) + UInt64(1442695040888963407)

    def next(mut self) -> Float32:
        self.s = self.s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var x = (self.s >> UInt64(40)) & UInt64(0xFFFFFF)
        return (Float32(Int(x)) + Float32(0.5)) * Float32(1.0 / 16777216.0)


def _uniform_sphere(mut rng: _Rng) -> Vec3:
    var z = Float32(1.0) - Float32(2.0) * rng.next()
    var rr = sqrt(max(Float32(0.0), Float32(1.0) - z * z))
    var phi = Float32(6.28318530718) * rng.next()
    return Vec3(rr * cos(phi), z, rr * sin(phi))


def _onb(n: Vec3) -> Tuple[Vec3, Vec3]:
    var a = Vec3(1.0, 0.0, 0.0) if abs(n.x) < Float32(0.9) else Vec3(0.0, 1.0, 0.0)
    var t = a.cross(n).normalized()
    return (t, n.cross(t))


def _sample_hg(mut rng: _Rng, wo: Vec3, g: Float32) -> Vec3:
    """Henyey-Greenstein direction about `wo` (forward = `wo`)."""
    var u = rng.next()
    var c: Float32
    if abs(g) < Float32(1e-3):
        c = Float32(1.0) - Float32(2.0) * u
    else:
        var sq = (Float32(1.0) - g * g) / (Float32(1.0) - g + Float32(2.0) * g * u)
        c = (Float32(1.0) + g * g - sq * sq) / (Float32(2.0) * g)
    c = max(Float32(-1.0), min(Float32(1.0), c))
    var st = sqrt(max(Float32(0.0), Float32(1.0) - c * c))
    var phi = Float32(6.28318530718) * rng.next()
    var b = _onb(wo)
    return (b[0] * (st * cos(phi)) + b[1] * (st * sin(phi)) + wo * c).normalized()


@always_inline
def _hg_pdf(c: Float32, g: Float32) -> Float32:
    var d = max(Float32(1e-6), Float32(1.0) + g * g - Float32(2.0) * g * c)
    return _INV_4PI * (Float32(1.0) - g * g) / (d * sqrt(d))


def _exact_sun_tr(p: SotaAtmoParams, pos: Vec3, sun: Vec3, channel: Int) -> Float32:
    var r = pos.length()
    var mu = pos.dot(sun) / r
    if ray_ground_dist(r, mu, p.r_bottom_km) > Float32(0.0):
        return 0.0
    var tau = optical_depth(r, mu, ray_top_dist(r, mu, p.r_top_km), 96, p)
    return exp(-tau[channel])


def sota_atmo_mc_sky_radiance(
    p: SotaAtmoParams, sun_dir_in: Vec3, sun_e: RGB, alt_km: Float32, view_in: Vec3, n_paths: Int, seed: UInt64 = 1
) -> RGB:
    """Brute-force path-traced sky radiance (all orders; per-channel delta tracking with a majorant equal to
    the sea-level extinction; next-event estimation to the sun with a converged transmittance; the sun is a
    point source, matching the LUT model). Ground: Lambertian with `p.ground_albedo`."""
    var sun = sun_dir_in.normalized()
    var out = RGB(0.0)
    var rb = p.r_bottom_km
    var rt = p.r_top_km
    var rng = _Rng(seed)
    for ch in range(3):
        var sig_max = p.rayleigh_scat[ch] + p.mie_ext[ch] + p.ozone_abs[ch]
        var acc = Float64(0.0)
        for _ in range(n_paths):
            var pos = Vec3(0.0, rb + max(Float32(0.001), alt_km), 0.0)
            var d = view_in.normalized()
            var w = Float32(1.0)
            var lval = Float32(0.0)
            for bounce in range(64):
                # Free flight to the next real collision (delta tracking) or the boundary.
                var r = pos.length()
                var mu = pos.dot(d) / r
                var tg = ray_ground_dist(r, mu, rb)
                var t_bound = tg if tg > Float32(0.0) else ray_top_dist(r, mu, rt)
                var t = Float32(0.0)
                var collided = False
                var sig_s = Float32(0.0)
                var sig_r = Float32(0.0)
                while True:
                    t -= log(max(Float32(1e-12), rng.next())) / sig_max
                    if t >= t_bound:
                        break
                    var q = pos + d * t
                    var dens = sota_atmo_densities(q.length() - rb, p)
                    var sr = p.rayleigh_scat[ch] * dens[0]
                    var sm = p.mie_scat[ch] * dens[1]
                    var st = sr + p.mie_ext[ch] * dens[1] + p.ozone_abs[ch] * dens[2]
                    if rng.next() * sig_max < st:
                        collided = True
                        sig_s = sr + sm
                        sig_r = sr
                        w *= (sr + sm) / st  # absorption as a weight
                        pos = q
                        break
                if not collided:
                    if tg > Float32(0.0):
                        # Ground hit: NEE + cosine bounce.
                        pos = pos + d * tg
                        var n = pos.normalized()
                        pos = n * (rb + Float32(1e-4))
                        var ndl = n.dot(sun)
                        if ndl > Float32(0.0):
                            lval += w * p.ground_albedo[ch] / _PI * ndl * sun_e[ch] * _exact_sun_tr(p, pos, sun, ch)
                        w *= p.ground_albedo[ch]
                        var b = _onb(n)
                        var u1 = rng.next()
                        var u2 = rng.next()
                        var rr = sqrt(u1)
                        var ph = Float32(6.28318530718) * u2
                        d = (b[0] * (rr * cos(ph)) + b[1] * (rr * sin(ph)) + n * sqrt(max(Float32(0.0), Float32(1.0) - u1))).normalized()
                    else:
                        break
                else:
                    # Volume collision: choose Rayleigh or Mie, NEE to the sun, sample the lobe.
                    var is_r = rng.next() * sig_s < sig_r
                    var c_sun = d.dot(sun)
                    var ph_sun = rayleigh_phase(c_sun) if is_r else cornette_shanks_phase(c_sun, p.mie_g)
                    lval += w * ph_sun * sun_e[ch] * _exact_sun_tr(p, pos, sun, ch)
                    if is_r:
                        var nd = _uniform_sphere(rng)
                        w *= rayleigh_phase(nd.dot(d)) / _INV_4PI
                        d = nd
                    else:
                        var nd = _sample_hg(rng, d, p.mie_g)
                        var c = nd.dot(d)
                        w *= cornette_shanks_phase(c, p.mie_g) / _hg_pdf(c, p.mie_g)
                        d = nd
                if bounce >= 3:
                    var q_rr = min(Float32(0.95), w)
                    if rng.next() >= q_rr:
                        break
                    w /= q_rr
            acc += Float64(lval)
        out[ch] = Float32(acc / Float64(n_paths))
    return out


# ============================================================================
# Time of day: sun and moon (Meeus low-precision series)
# ============================================================================

def julian_day(year: Int, month: Int, day: Int, hour_utc: Float64) -> Float64:
    var y = year
    var m = month
    if m <= 2:
        y -= 1
        m += 12
    var a = y // 100
    var b = 2 - a + a // 4
    return Float64(Int(365.25 * Float64(y + 4716))) + Float64(Int(30.6001 * Float64(m + 1))) + Float64(day) + Float64(b) - 1524.5 + hour_utc / 24.0


def _deg(x: Float64) -> Float64:
    return x * 0.017453292519943295


def _equ_to_world(ra: Float64, dec: Float64, lst_deg: Float64, lat_deg: Float64, north_az_deg: Float64) -> Vec3:
    """Equatorial (rad) -> local horizon -> world: +Y up, north = +Z rotated by `north_az_deg` about Y, east = +X."""
    var ha = _deg(lst_deg) - ra
    var lat = _deg(lat_deg)
    var sin_alt = sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(ha)
    var alt = asin(max(-1.0, min(1.0, sin_alt)))
    # Azimuth from north toward east.
    var az = atan2(-sin(ha) * cos(dec), cos(lat) * sin(dec) - sin(lat) * cos(dec) * cos(ha))
    az += _deg(north_az_deg)
    var ce = cos(alt)
    return Vec3(Float32(ce * sin(az)), Float32(sin(alt)), Float32(ce * cos(az)))


struct SotaTimeOfDay(Copyable, ImplicitlyCopyable, Movable):
    """Calendar date + UTC time + observer location -> sun / moon directions and moon phase.
    `north_azimuth_deg` rotates the compass in the world (0: north = +Z, east = +X)."""
    var year: Int
    var month: Int
    var day: Int
    var hour_utc: Float64
    var latitude_deg: Float64
    var longitude_deg: Float64
    var north_azimuth_deg: Float64

    def __init__(
        out self,
        year: Int = 2026,
        month: Int = 6,
        day: Int = 21,
        hour_utc: Float64 = 12.0,
        latitude_deg: Float64 = 45.0,
        longitude_deg: Float64 = 0.0,
        north_azimuth_deg: Float64 = 0.0,
    ):
        self.year = year
        self.month = month
        self.day = day
        self.hour_utc = hour_utc
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.north_azimuth_deg = north_azimuth_deg

    def _n(self) -> Float64:
        return julian_day(self.year, self.month, self.day, self.hour_utc) - 2451545.0

    def _lst_deg(self) -> Float64:
        var n = self._n()
        var gmst_h = 18.697374558 + 24.06570982441908 * n
        return gmst_h * 15.0 + self.longitude_deg

    def sun_equatorial(self) -> Tuple[Float64, Float64]:
        var n = self._n()
        var l = 280.460 + 0.9856474 * n
        var g = _deg(357.528 + 0.9856003 * n)
        var lam = _deg(l + 1.915 * sin(g) + 0.020 * sin(2.0 * g))
        var eps = _deg(23.439 - 0.0000004 * n)
        return (atan2(cos(eps) * sin(lam), cos(lam)), asin(sin(eps) * sin(lam)))

    def moon_equatorial(self) -> Tuple[Float64, Float64]:
        var n = self._n()
        var lp = 218.316 + 13.176396 * n
        var mp = _deg(134.963 + 13.064993 * n)
        var ms = _deg(357.529 + 0.98560028 * n)
        var d = _deg(297.850 + 12.190749 * n)
        var f = _deg(93.272 + 13.229350 * n)
        var lam = lp + 6.289 * sin(mp) + 1.274 * sin(2.0 * d - mp) + 0.658 * sin(2.0 * d) + 0.214 * sin(2.0 * mp) - 0.186 * sin(ms) - 0.114 * sin(2.0 * f)
        var beta = 5.128 * sin(f) + 0.280 * sin(mp + f) + 0.277 * sin(mp - f) + 0.173 * sin(2.0 * d - f)
        var lr = _deg(lam)
        var br = _deg(beta)
        var eps = _deg(23.439 - 0.0000004 * n)
        var ra = atan2(sin(lr) * cos(eps) - tan_f64(br) * sin(eps), cos(lr))
        var dec = asin(sin(br) * cos(eps) + cos(br) * sin(eps) * sin(lr))
        return (ra, dec)

    def sun_dir(self) -> Vec3:
        var e = self.sun_equatorial()
        return _equ_to_world(e[0], e[1], self._lst_deg(), self.latitude_deg, self.north_azimuth_deg)

    def moon_dir(self) -> Vec3:
        var e = self.moon_equatorial()
        return _equ_to_world(e[0], e[1], self._lst_deg(), self.latitude_deg, self.north_azimuth_deg)

    def moon_elongation_rad(self) -> Float32:
        """Geocentric angle between the sun and the moon (0 = new moon, pi = full moon)."""
        var s = self.sun_equatorial()
        var m = self.moon_equatorial()
        var c = sin(s[1]) * sin(m[1]) + cos(s[1]) * cos(m[1]) * cos(s[0] - m[0])
        return Float32(acos(max(-1.0, min(1.0, c))))

    def moon_illuminated_fraction(self) -> Float32:
        return (Float32(1.0) - cos(self.moon_elongation_rad())) * Float32(0.5)


def tan_f64(x: Float64) -> Float64:
    return sin(x) / cos(x)
