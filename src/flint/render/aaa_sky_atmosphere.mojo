"""Phase 3 (`V18`): Bruneton / Hillaire Multiple-Scattering Physical Sky & Aerial Perspective (`aaa_sky_atmosphere.mojo`).

Implements a 100% Pure-Mojo spherical-planet physical atmosphere:
  1. **Transmittance LUT (`256 x 64`)**: Integrates Rayleigh scattering (`H_R = 8.0 km`),
     Mie aerosol extinction (`H_M = 1.2 km`), and stratospheric Ozone Chappuis absorption
     (`25 km` tent profile, width `15 km`) to the top of the `60 km` atmosphere shell.
  2. **Multiple-Scattering LUT (`32 x 32`)**: Hillaire (2020) closed-form infinite-bounce
     isotropic phase series `Psi_ms = L_2nd / (1 - f_ms)` including planetary ground albedo bounce.
  3. **Sky-View LUT (`192 x 108`)**: Horizon-warped non-linear latitude/azimuth ray-marched LUT
     feeding `BC6H_UFLOAT` Split-Sum GGX Cubemap + Order-2 `SH9` diffuse irradiance (`bake_bruneton_sky_ibl`).
  4. **Aerial Perspective (`eval_aerial_perspective_cpu` & `emit_aerial_perspective_spv`)**:
     Distance-dependent RGB Beer-Lambert transmittance + Rayleigh / Cornette-Shanks Mie + Multiple-Scattering
     in-scattering along camera view rays for distant world geometry.
"""
from std.math import atan2, cos, exp, floor, isfinite, max, min, sin, sqrt
from flint.core.math import Vec3
from flint.assets.bc_compress import encode_bc6h_hdr
from flint.render.spirv import AaaSpirvModule
from flint.render.aaa_ibl import (
    BakedIblEnvironment,
    _importance_sample_ggx,
    _radical_inverse_vdc,
    cube_face_direction,
)

comptime ATMO_TRANS_W: Int = 256
comptime ATMO_TRANS_H: Int = 64
comptime ATMO_MS_DIM: Int = 32
comptime ATMO_SKY_W: Int = 192
comptime ATMO_SKY_H: Int = 108


struct BrunetonAtmosphereParams(Copyable, Movable):
    """Physical planetary & aerosol parameters (`km` & `km^-1` units)."""
    var r_bottom_km: Float32
    var r_top_km: Float32
    var observer_alt_km: Float32
    var sun_dir: Vec3
    var solar_irradiance: Vec3
    var sun_angular_radius: Float32
    var rayleigh_scat: Vec3       # km^-1 at sea level
    var rayleigh_scale_h: Float32 # km
    var mie_scat_base: Vec3       # km^-1 at sea level (turbidity = 1)
    var mie_ext_base: Vec3        # km^-1 at sea level
    var mie_scale_h: Float32      # km
    var mie_g: Float32
    var turbidity: Float32
    var ozone_abs: Vec3           # km^-1 peak at 25 km Chappuis band
    var ground_albedo: Vec3
    var aerial_km_per_unit: Float32
    var cloud_coverage: Float32
    var cloud_wind_offset: Vec3
    var star_intensity: Float32

    def __init__(
        out self,
        sun_dir: Vec3 = Vec3(0.35, 0.78, -0.52),
        turbidity: Float32 = 1.0,
        solar_intensity: Float32 = 4.5,
        aerial_km_per_unit: Float32 = 0.05,
        cloud_coverage: Float32 = 0.0,
        cloud_wind_offset: Vec3 = Vec3(0.0, 0.0, 0.0),
        star_intensity: Float32 = 0.0,
    ):
        self.r_bottom_km = Float32(6360.0)
        self.r_top_km = Float32(6420.0)
        self.observer_alt_km = Float32(0.20)
        self.sun_dir = sun_dir.normalized()
        self.solar_irradiance = Vec3(1.0, 0.965, 0.925) * solar_intensity
        self.sun_angular_radius = Float32(0.00465)
        self.rayleigh_scat = Vec3(0.005802, 0.013558, 0.033100)
        self.rayleigh_scale_h = Float32(8.0)
        self.mie_scat_base = Vec3(0.003996, 0.003996, 0.003996)
        self.mie_ext_base = Vec3(0.004440, 0.004440, 0.004440)
        self.mie_scale_h = Float32(1.2)
        self.mie_g = Float32(0.80)
        self.turbidity = max(Float32(0.2), turbidity)
        self.ozone_abs = Vec3(0.000650, 0.001881, 0.000085)
        self.ground_albedo = Vec3(0.18, 0.16, 0.14)
        self.aerial_km_per_unit = max(Float32(1e-4), aerial_km_per_unit)
        self.cloud_coverage = max(Float32(0.0), min(Float32(1.0), cloud_coverage))
        self.cloud_wind_offset = cloud_wind_offset
        self.star_intensity = max(Float32(0.0), star_intensity)


@always_inline
def _rayleigh_phase(mu: Float32) -> Float32:
    return Float32(3.0 / (16.0 * 3.14159265)) * (Float32(1.0) + mu * mu)


@always_inline
def _mie_cornette_shanks_phase(mu: Float32, g: Float32) -> Float32:
    var g2 = g * g
    var num = Float32(3.0 / (8.0 * 3.14159265)) * (Float32(1.0) - g2) * (Float32(1.0) + mu * mu)
    var den_base = max(Float32(1e-4), Float32(1.0) + g2 - Float32(2.0) * g * mu)
    var den = (Float32(2.0) + g2) * den_base * sqrt(den_base)
    return num / max(Float32(1e-6), den)


@always_inline
def _smoothstep_cpu(e0: Float32, e1: Float32, x: Float32) -> Float32:
    var t = max(Float32(0.0), min(Float32(1.0), (x - e0) / (e1 - e0)))
    return t * t * (Float32(3.0) - Float32(2.0) * t)


@always_inline
def _hg_dual_cloud_phase_cpu(cos_theta: Float32, g_fwd: Float32, g_back: Float32) -> Float32:
    var inv_4pi = Float32(0.07957747)
    var gf2 = g_fwd * g_fwd
    var df = max(Float32(0.01), Float32(1.0) + gf2 - Float32(2.0) * g_fwd * cos_theta)
    var pf = inv_4pi * (Float32(1.0) - gf2) / (df * sqrt(df))
    var gb2 = g_back * g_back
    var db = max(Float32(0.01), Float32(1.0) + gb2 - Float32(2.0) * g_back * cos_theta)
    var pb = inv_4pi * (Float32(1.0) - gb2) / (db * sqrt(db))
    return Float32(0.72) * pf + Float32(0.28) * pb


def eval_nubis_cloud_density_cpu(
    pos_km: Vec3,
    h_frac: Float32,
    coverage: Float32,
    wind_offset_m: Vec3,
) -> Float32:
    """Nubis-class 3D Perlin-Worley cumulus density with rounded anvil height profile and billow erosion."""
    if coverage <= Float32(0.01) or h_frac <= Float32(0.0) or h_frac >= Float32(1.0):
        return Float32(0.0)
    var px = pos_km.x + wind_offset_m.x * Float32(0.001)
    var py = pos_km.y + wind_offset_m.y * Float32(0.001)
    var pz = pos_km.z + wind_offset_m.z * Float32(0.001)

    # Macro Perlin-like harmonic deck + billowing 3D Worley-ridge domes
    var m1 = sin(px * Float32(0.41) + pz * Float32(0.29)) * cos(pz * Float32(0.37) - px * Float32(0.23))
    var m2 = sin(px * Float32(0.83) - pz * Float32(0.67) + Float32(1.3)) * cos(px * Float32(0.59) + pz * Float32(0.79))
    var base_shape = Float32(0.52) + Float32(0.30) * m1 + Float32(0.20) * m2

    # 3D Billow cauliflower lobes (`1 - |sin*cos|`)
    var b1 = Float32(1.0) - abs(sin(px * Float32(1.85) + py * Float32(1.35)) * cos(pz * Float32(1.95) - py * Float32(1.15)))
    var b2 = Float32(1.0) - abs(sin(px * Float32(3.90) - pz * Float32(3.40)) * cos(py * Float32(3.70) + px * Float32(2.60)))
    var billow = b1 * Float32(0.65) + b2 * Float32(0.35)

    var h_prof = _smoothstep_cpu(Float32(0.0), Float32(0.14), h_frac) * _smoothstep_cpu(Float32(1.0), Float32(0.54), h_frac)
    var thresh = Float32(1.0) - coverage * Float32(0.88)
    var raw = base_shape * h_prof - thresh
    var eroded = raw - (Float32(1.0) - billow) * Float32(0.24) * (Float32(0.35) + Float32(0.65) * h_frac)
    return max(Float32(0.0), eroded * Float32(4.2))


@always_inline
def _ray_sphere_top_dist(r: Float32, mu: Float32, r_top: Float32) -> Float32:
    """Distance from radius `r` along zenith cosine `mu` to outer sphere `r_top` (`>= 0`)."""
    var disc = r * r * (mu * mu - Float32(1.0)) + r_top * r_top
    return max(Float32(0.0), -r * mu + sqrt(max(Float32(0.0), disc)))


@always_inline
def _ray_ground_dist(r: Float32, mu: Float32, r_bot: Float32) -> Float32:
    """Returns positive distance to planet surface `r_bot` if ray intersects ground, else `-1.0`."""
    if mu >= Float32(0.0):
        return Float32(-1.0)
    var disc = r * r * (mu * mu - Float32(1.0)) + r_bot * r_bot
    if disc < Float32(0.0):
        return Float32(-1.0)
    var d = -r * mu - sqrt(disc)
    return d if d > Float32(0.0) else Float32(-1.0)


@always_inline
def _medium_extinction_and_scattering(
    alt_km: Float32,
    p: BrunetonAtmosphereParams,
) -> Tuple[Vec3, Vec3, Vec3]:
    """Returns `(sigma_rayleigh, sigma_mie_scat, sigma_total_extinction)` at `alt_km`."""
    var h = max(Float32(0.0), alt_km)
    var rho_r = exp(-h / p.rayleigh_scale_h)
    var rho_m = exp(-h / p.mie_scale_h) * p.turbidity
    var rho_o = max(Float32(0.0), Float32(1.0) - abs(h - Float32(25.0)) / Float32(15.0))

    var sig_r = p.rayleigh_scat * rho_r
    var sig_ms = p.mie_scat_base * rho_m
    var sig_me = p.mie_ext_base * rho_m
    var sig_o = p.ozone_abs * rho_o
    var sig_ext = sig_r + sig_me + sig_o
    return (sig_r, sig_ms, sig_ext)


@always_inline
def _bilerp_lut(lut: List[Vec3], w: Int, h: Int, u_in: Float32, v_in: Float32, wrap_u: Bool = False) -> Vec3:
    if w <= 0 or h <= 0 or len(lut) < w * h:
        return Vec3()
    var u = u_in if isfinite(u_in) else Float32(0.0)
    var v_safe = v_in if isfinite(v_in) else Float32(0.0)
    if wrap_u:
        u = u - floor(u)
        if u < Float32(0.0):
            u += Float32(1.0)
        if u >= Float32(1.0):
            u = Float32(0.0)
    else:
        u = max(Float32(0.0), min(Float32(1.0), u))
    var v = max(Float32(0.0), min(Float32(1.0), v_safe))

    var fx = u * Float32(w - 1)
    var fy = v * Float32(h - 1)
    var x0 = min(w - 1, max(0, Int(fx)))
    var y0 = min(h - 1, max(0, Int(fy)))
    var x1 = ((x0 + 1) % w) if wrap_u else min(w - 1, x0 + 1)
    var y1 = min(h - 1, y0 + 1)
    var tx = max(Float32(0.0), min(Float32(1.0), fx - Float32(x0)))
    var ty = max(Float32(0.0), min(Float32(1.0), fy - Float32(y0)))

    var c00 = lut[y0 * w + x0]
    var c10 = lut[y0 * w + x1]
    var c01 = lut[y1 * w + x0]
    var c11 = lut[y1 * w + x1]
    var top = c00 * (Float32(1.0) - tx) + c10 * tx
    var bot = c01 * (Float32(1.0) - tx) + c11 * tx
    return top * (Float32(1.0) - ty) + bot * ty


struct BrunetonSkyLuts(Movable):
    """Holds precomputed Bruneton/Hillaire Transmittance (`256x64`), Multiple-Scattering (`32x32`),
    and Sky-View (`192x108`) LUTs."""
    var params: BrunetonAtmosphereParams
    var transmittance_lut: List[Vec3]
    var multiscatter_lut: List[Vec3]
    var sky_view_lut: List[Vec3]

    def __init__(out self, params: BrunetonAtmosphereParams):
        self.params = params.copy()
        self.transmittance_lut = List[Vec3](length=ATMO_TRANS_W * ATMO_TRANS_H, fill=Vec3(1.0, 1.0, 1.0))
        self.multiscatter_lut = List[Vec3](length=ATMO_MS_DIM * ATMO_MS_DIM, fill=Vec3(0.0, 0.0, 0.0))
        self.sky_view_lut = List[Vec3](length=ATMO_SKY_W * ATMO_SKY_H, fill=Vec3(0.0, 0.0, 0.0))
        self._bake_transmittance()
        self._bake_multiscattering()
        self._bake_sky_view()

    @always_inline
    def sample_transmittance(self, r_km: Float32, mu: Float32) -> Vec3:
        if not (isfinite(r_km) and isfinite(mu)):
            return Vec3(0.0, 0.0, 0.0)
        var u = (mu + Float32(0.25)) / Float32(1.25)
        var h_norm = max(Float32(0.0), min(Float32(1.0), (r_km - self.params.r_bottom_km) / (self.params.r_top_km - self.params.r_bottom_km)))
        var v = sqrt(h_norm)
        return _bilerp_lut(self.transmittance_lut, ATMO_TRANS_W, ATMO_TRANS_H, u, v, wrap_u=False)

    @always_inline
    def sample_multiscatter(self, r_km: Float32, mu_s: Float32) -> Vec3:
        if not (isfinite(r_km) and isfinite(mu_s)):
            return Vec3(0.0, 0.0, 0.0)
        var u = Float32(0.5) + Float32(0.5) * mu_s
        var v = (r_km - self.params.r_bottom_km) / (self.params.r_top_km - self.params.r_bottom_km)
        return _bilerp_lut(self.multiscatter_lut, ATMO_MS_DIM, ATMO_MS_DIM, u, v, wrap_u=False)

    def sun_transmittance_at_observer(self) -> Vec3:
        var r_obs = self.params.r_bottom_km + self.params.observer_alt_km
        var mu_s = self.params.sun_dir.y
        if _ray_ground_dist(r_obs, mu_s, self.params.r_bottom_km) > Float32(0.0):
            return Vec3(0.0, 0.0, 0.0)
        return self.sample_transmittance(r_obs, mu_s)

    def _bake_transmittance(mut self):
        var r_b = self.params.r_bottom_km
        var r_t = self.params.r_top_km
        comptime STEPS: Int = 24

        for y in range(ATMO_TRANS_H):
            var v = Float32(y) / Float32(ATMO_TRANS_H - 1)
            var r = r_b + (v * v) * (r_t - r_b) + Float32(1e-3)
            for x in range(ATMO_TRANS_W):
                var u = Float32(x) / Float32(ATMO_TRANS_W - 1)
                var mu = -Float32(0.25) + u * Float32(1.25)
                var d_max = _ray_sphere_top_dist(r, mu, r_t)
                var dt = d_max / Float32(STEPS)
                var tau = Vec3(0.0, 0.0, 0.0)
                for s_i in range(STEPS):
                    var t = (Float32(s_i) + Float32(0.5)) * dt
                    var ri = sqrt(max(r_b * r_b, r * r + t * t + Float32(2.0) * r * mu * t))
                    var med = _medium_extinction_and_scattering(ri - r_b, self.params)
                    tau = tau + med[2] * dt
                self.transmittance_lut[y * ATMO_TRANS_W + x] = Vec3(
                    exp(-tau.x), exp(-tau.y), exp(-tau.z)
                )

    def _bake_multiscattering(mut self):
        """Hillaire (2020) Eq. 5-7: integrates 2nd-order in-scattered luminance `L_2nd` and
        isotropic transfer factor `f_ms` over 16 sphere directions, storing `Psi_ms = L_2nd / (1 - f_ms)`."""
        var r_b = self.params.r_bottom_km
        var r_t = self.params.r_top_km
        var inv_4pi = Float32(1.0 / (4.0 * 3.14159265))
        comptime DIRS: Int = 16
        comptime RAY_STEPS: Int = 12

        for y in range(ATMO_MS_DIM):
            var v = (Float32(y) + Float32(0.5)) / Float32(ATMO_MS_DIM)
            var r = r_b + v * (r_t - r_b) + Float32(0.02)
            for x in range(ATMO_MS_DIM):
                var u = (Float32(x) + Float32(0.5)) / Float32(ATMO_MS_DIM)
                var mu_s = u * Float32(2.0) - Float32(1.0)
                var sun_d = Vec3(sqrt(max(Float32(0.0), Float32(1.0) - mu_s * mu_s)), mu_s, Float32(0.0))

                var l_2nd = Vec3(0.0, 0.0, 0.0)
                var f_ms = Vec3(0.0, 0.0, 0.0)

                for d_i in range(DIRS):
                    # Stratified Fibonacci sphere direction
                    var z_d = Float32(1.0) - (Float32(2 * d_i + 1) / Float32(DIRS))
                    var rad_xy = sqrt(max(Float32(0.0), Float32(1.0) - z_d * z_d))
                    var phi = Float32(d_i) * Float32(2.39996323)
                    var dir = Vec3(rad_xy * cos(phi), z_d, rad_xy * sin(phi))
                    var mu = dir.y

                    var g_dist = _ray_ground_dist(r, mu, r_b)
                    var t_max = g_dist if g_dist > Float32(0.0) else _ray_sphere_top_dist(r, mu, r_t)
                    var dt = t_max / Float32(RAY_STEPS)

                    var throughput = Vec3(1.0, 1.0, 1.0)
                    var lum_ray = Vec3(0.0, 0.0, 0.0)
                    var f_ray = Vec3(0.0, 0.0, 0.0)

                    for s_i in range(RAY_STEPS):
                        var t = (Float32(s_i) + Float32(0.5)) * dt
                        var px = dir.x * t
                        var py = r + dir.y * t
                        var pz = dir.z * t
                        var ri = sqrt(px * px + py * py + pz * pz)
                        var med = _medium_extinction_and_scattering(ri - r_b, self.params)
                        var sig_scat = med[0] + med[1]
                        var sig_ext = med[2]

                        var tr_step = Vec3(exp(-sig_ext.x * dt), exp(-sig_ext.y * dt), exp(-sig_ext.z * dt))
                        var scat_int = Vec3(
                            (Float32(1.0) - tr_step.x) / max(Float32(1e-6), sig_ext.x),
                            (Float32(1.0) - tr_step.y) / max(Float32(1e-6), sig_ext.y),
                            (Float32(1.0) - tr_step.z) / max(Float32(1e-6), sig_ext.z),
                        )

                        # Local zenith normal at `(px, py, pz)` is `(px, py, pz) / ri`
                        var local_mu_s = (px * sun_d.x + py * sun_d.y + pz * sun_d.z) / max(Float32(1.0), ri)
                        var shadow = Float32(0.0) if _ray_ground_dist(ri, local_mu_s, r_b) > Float32(0.0) else Float32(1.0)
                        var t_sun = self.sample_transmittance(ri, local_mu_s) * shadow

                        # Isotropic phase 1/(4*pi) for multiple scattering energy conservation
                        var s_in = Vec3(
                            sig_scat.x * t_sun.x * inv_4pi,
                            sig_scat.y * t_sun.y * inv_4pi,
                            sig_scat.z * t_sun.z * inv_4pi,
                        )
                        lum_ray = lum_ray + Vec3(
                            throughput.x * s_in.x * scat_int.x,
                            throughput.y * s_in.y * scat_int.y,
                            throughput.z * s_in.z * scat_int.z,
                        )
                        f_ray = f_ray + Vec3(
                            throughput.x * sig_scat.x * scat_int.x,
                            throughput.y * sig_scat.y * scat_int.y,
                            throughput.z * sig_scat.z * scat_int.z,
                        )
                        throughput = Vec3(
                            throughput.x * tr_step.x,
                            throughput.y * tr_step.y,
                            throughput.z * tr_step.z,
                        )

                    # Planetary Lambertian ground bounce
                    if g_dist > Float32(0.0):
                        var gx = dir.x * g_dist
                        var gy = r + dir.y * g_dist
                        var gz = dir.z * g_dist
                        var gn_dot_s = max(Float32(0.0), (gx * sun_d.x + gy * sun_d.y + gz * sun_d.z) / r_b)
                        if gn_dot_s > Float32(0.0):
                            var t_sun_g = self.sample_transmittance(r_b + Float32(0.01), gn_dot_s)
                            var inv_pi = Float32(1.0 / 3.14159265)
                            lum_ray = lum_ray + Vec3(
                                throughput.x * t_sun_g.x * self.params.ground_albedo.x * gn_dot_s * inv_pi,
                                throughput.y * t_sun_g.y * self.params.ground_albedo.y * gn_dot_s * inv_pi,
                                throughput.z * t_sun_g.z * self.params.ground_albedo.z * gn_dot_s * inv_pi,
                            )

                    var d_omega = Float32(4.0 * 3.14159265) / Float32(DIRS)
                    l_2nd = l_2nd + lum_ray * (inv_4pi * d_omega)
                    f_ms = f_ms + f_ray * (inv_4pi * d_omega)

                self.multiscatter_lut[y * ATMO_MS_DIM + x] = Vec3(
                    l_2nd.x / max(Float32(0.05), Float32(1.0) - f_ms.x),
                    l_2nd.y / max(Float32(0.05), Float32(1.0) - f_ms.y),
                    l_2nd.z / max(Float32(0.05), Float32(1.0) - f_ms.z),
                )

    def _bake_sky_view(mut self):
        var r_b = self.params.r_bottom_km
        var r_t = self.params.r_top_km
        var r_obs = r_b + self.params.observer_alt_km
        var half_pi = Float32(1.57079632679)
        var two_pi = Float32(6.28318530718)
        comptime STEPS: Int = 24

        for y in range(ATMO_SKY_H):
            var v = (Float32(y) + Float32(0.5)) / Float32(ATMO_SKY_H)
            # Horizon-warped elevation: high texel density near horizon `v = 0.5`!
            var elev: Float32
            if v < Float32(0.5):
                var t = Float32(1.0) - Float32(2.0) * v
                elev = half_pi * (t * t)
            else:
                var t = Float32(2.0) * v - Float32(1.0)
                elev = -half_pi * (t * t)

            var sin_el = sin(elev)
            var cos_el = cos(elev)

            for x in range(ATMO_SKY_W):
                var u = (Float32(x) + Float32(0.5)) / Float32(ATMO_SKY_W)
                var azim = (u - Float32(0.5)) * two_pi
                var dir = Vec3(cos_el * cos(azim), sin_el, cos_el * sin(azim)).normalized()
                var mu = dir.y
                var cos_theta = dir.dot(self.params.sun_dir)
                var pr = _rayleigh_phase(cos_theta)
                var pm = _mie_cornette_shanks_phase(cos_theta, self.params.mie_g)

                var g_dist = _ray_ground_dist(r_obs, mu, r_b)
                var t_max = g_dist if g_dist > Float32(0.0) else _ray_sphere_top_dist(r_obs, mu, r_t)
                var dt = t_max / Float32(STEPS)

                var throughput = Vec3(1.0, 1.0, 1.0)
                var L = Vec3(0.0, 0.0, 0.0)

                for s_i in range(STEPS):
                    var t = (Float32(s_i) + Float32(0.5)) * dt
                    var px = dir.x * t
                    var py = r_obs + dir.y * t
                    var pz = dir.z * t
                    var ri = sqrt(px * px + py * py + pz * pz)
                    var med = _medium_extinction_and_scattering(ri - r_b, self.params)
                    var sig_r = med[0]
                    var sig_ms = med[1]
                    var sig_scat = sig_r + sig_ms
                    var sig_ext = med[2]

                    var tr_step = Vec3(exp(-sig_ext.x * dt), exp(-sig_ext.y * dt), exp(-sig_ext.z * dt))
                    var scat_int = Vec3(
                        (Float32(1.0) - tr_step.x) / max(Float32(1e-6), sig_ext.x),
                        (Float32(1.0) - tr_step.y) / max(Float32(1e-6), sig_ext.y),
                        (Float32(1.0) - tr_step.z) / max(Float32(1e-6), sig_ext.z),
                    )

                    var local_mu_s = (px * self.params.sun_dir.x + py * self.params.sun_dir.y + pz * self.params.sun_dir.z) / max(Float32(1.0), ri)
                    var shadow = Float32(0.0) if _ray_ground_dist(ri, local_mu_s, r_b) > Float32(0.0) else Float32(1.0)
                    var t_sun = self.sample_transmittance(ri, local_mu_s) * shadow
                    var psi_ms = self.sample_multiscatter(ri, local_mu_s)

                    var s_single = Vec3(
                        (sig_r.x * pr + sig_ms.x * pm) * t_sun.x + sig_scat.x * psi_ms.x,
                        (sig_r.y * pr + sig_ms.y * pm) * t_sun.y + sig_scat.y * psi_ms.y,
                        (sig_r.z * pr + sig_ms.z * pm) * t_sun.z + sig_scat.z * psi_ms.z,
                    )
                    L = L + Vec3(
                        throughput.x * s_single.x * scat_int.x,
                        throughput.y * s_single.y * scat_int.y,
                        throughput.z * s_single.z * scat_int.z,
                    )
                    throughput = Vec3(
                        throughput.x * tr_step.x,
                        throughput.y * tr_step.y,
                        throughput.z * tr_step.z,
                    )

                if g_dist > Float32(0.0):
                    var gx = dir.x * g_dist
                    var gy = r_obs + dir.y * g_dist
                    var gz = dir.z * g_dist
                    var gn_dot_s = max(Float32(0.0), (gx * self.params.sun_dir.x + gy * self.params.sun_dir.y + gz * self.params.sun_dir.z) / r_b)
                    var t_sun_g = self.sample_transmittance(r_b + Float32(0.01), gn_dot_s)
                    var psi_g = self.sample_multiscatter(r_b + Float32(0.01), self.params.sun_dir.y)
                    var inv_pi = Float32(1.0 / 3.14159265)
                    var g_irr = t_sun_g * (gn_dot_s * inv_pi) + psi_g
                    L = L + Vec3(
                        throughput.x * g_irr.x * self.params.ground_albedo.x,
                        throughput.y * g_irr.y * self.params.ground_albedo.y,
                        throughput.z * g_irr.z * self.params.ground_albedo.z,
                    )

                var sky_clear = Vec3(
                    L.x * self.params.solar_irradiance.x,
                    L.y * self.params.solar_irradiance.y,
                    L.z * self.params.solar_irradiance.z,
                )

                # Composite Nubis 3D Volumetric Cumulus + Stratospheric Cirrus into Sky-View LUT (`BC6H` IBL & SH9!)
                if self.params.cloud_coverage > Float32(0.01) and g_dist < Float32(0.0) and mu > Float32(0.015):
                    var r_cbot = r_b + Float32(1.65)
                    var r_ctop = r_b + Float32(4.15)
                    var t_c0 = _ray_sphere_top_dist(r_obs, mu, r_cbot)
                    var t_c1 = min(t_c0 + Float32(28.0), _ray_sphere_top_dist(r_obs, mu, r_ctop))
                    var dist_fade = exp(-t_c0 * Float32(0.024))
                    if t_c1 > t_c0 and dist_fade > Float32(0.02):
                        comptime C_STEPS: Int = 10
                        var c_dt = (t_c1 - t_c0) / Float32(C_STEPS)
                        var tr_c = Float32(1.0)
                        var lum_c = Vec3(0.0, 0.0, 0.0)
                        var p_oct0 = _hg_dual_cloud_phase_cpu(cos_theta, Float32(0.74), Float32(-0.28))
                        var p_oct1 = _hg_dual_cloud_phase_cpu(cos_theta, Float32(0.33), Float32(-0.12))
                        var r_mid_c = r_b + Float32(2.8)
                        var sun_tr_c = self.sample_transmittance(r_mid_c, max(Float32(-0.15), self.params.sun_dir.y))
                        var sun_col_c = Vec3(
                            self.params.solar_irradiance.x * sun_tr_c.x,
                            self.params.solar_irradiance.y * sun_tr_c.y,
                            self.params.solar_irradiance.z * sun_tr_c.z,
                        )
                        var amb_c = sky_clear * Float32(0.38) + Vec3(0.03, 0.035, 0.045) * max(Float32(0.05), Float32(0.4) + self.params.sun_dir.y)

                        for ci in range(C_STEPS):
                            if tr_c < Float32(0.02):
                                break
                            var tc = t_c0 + (Float32(ci) + Float32(0.5)) * c_dt
                            var cpos = Vec3(dir.x * tc, (r_obs - r_b) + dir.y * tc, dir.z * tc)
                            var r_cur = sqrt(dir.x * dir.x * tc * tc + (r_obs + dir.y * tc) * (r_obs + dir.y * tc) + dir.z * dir.z * tc * tc)
                            var h_fr = (r_cur - r_cbot) / Float32(2.50)
                            var dens = eval_nubis_cloud_density_cpu(cpos, h_fr, self.params.cloud_coverage, self.params.cloud_wind_offset) * dist_fade
                            if dens > Float32(0.002):
                                var spos = cpos + self.params.sun_dir * Float32(0.42)
                                var dens_s = eval_nubis_cloud_density_cpu(
                                    spos, h_fr + self.params.sun_dir.y * Float32(0.17), self.params.cloud_coverage, self.params.cloud_wind_offset
                                )
                                var tau_s = (dens * Float32(0.35) + dens_s * Float32(0.85) + max(Float32(0.0), Float32(1.0) - h_fr) * Float32(0.35)) * Float32(1.6)
                                var ms_s = exp(-tau_s) * p_oct0 + Float32(0.58) * exp(-Float32(0.48) * tau_s) * p_oct1
                                var powder = Float32(1.0) - Float32(0.62) * exp(-tau_s * Float32(2.4))
                                var s_lit = sun_col_c * (ms_s * powder * Float32(4.2)) + amb_c * (Float32(0.42) + Float32(0.58) * h_fr)
                                var step_tr = exp(-dens * c_dt * Float32(2.6))
                                var w_int = tr_c * (Float32(1.0) - step_tr)
                                lum_c = lum_c + s_lit * w_int
                                tr_c *= step_tr

                        # High-altitude 8.8 km Cirrus wisps
                        var t_cir = _ray_sphere_top_dist(r_obs, mu, r_b + Float32(8.8))
                        var cir_fade = exp(-t_cir * Float32(0.016))
                        var cx_p = dir.x * t_cir * Float32(0.18) + self.params.cloud_wind_offset.x * Float32(0.0004)
                        var cz_p = dir.z * t_cir * Float32(0.07) + self.params.cloud_wind_offset.z * Float32(0.0004)
                        var cir_n = sin(cx_p * Float32(2.3) + sin(cz_p * Float32(3.1))) * cos(cz_p * Float32(1.9) - cx_p * Float32(0.8))
                        var cir_a = max(Float32(0.0), cir_n - (Float32(0.58) - self.params.cloud_coverage * Float32(0.28))) * Float32(0.42) * cir_fade
                        if cir_a > Float32(0.002):
                            var tr_hi = self.sample_transmittance(r_b + Float32(8.8), max(Float32(-0.18), self.params.sun_dir.y))
                            var cir_col = Vec3(
                                self.params.solar_irradiance.x * tr_hi.x * Float32(0.32),
                                self.params.solar_irradiance.y * tr_hi.y * Float32(0.32),
                                self.params.solar_irradiance.z * tr_hi.z * Float32(0.32),
                            ) + sky_clear * Float32(0.45)
                            lum_c = lum_c + cir_col * (tr_c * min(Float32(0.35), cir_a))
                            tr_c *= (Float32(1.0) - min(Float32(0.35), cir_a))

                        sky_clear = sky_clear * tr_c + lum_c

                self.sky_view_lut[y * ATMO_SKY_W + x] = sky_clear

    def sample_sky(self, d_in: Vec3, include_sun_disk: Bool = False) -> Vec3:
        """Samples the `192 x 108` Horizon-Warped Sky-View LUT along world unit direction `d_in`."""
        var d = d_in.normalized()
        var half_pi = Float32(1.57079632679)
        var inv_2pi = Float32(0.15915494309)

        # asin(d.y) via atan2(d.y, horiz_len)
        var horiz = sqrt(max(Float32(0.0), d.x * d.x + d.z * d.z))
        var elev = atan2(d.y, horiz)
        var v: Float32
        if elev >= Float32(0.0):
            v = Float32(0.5) - Float32(0.5) * sqrt(min(Float32(1.0), elev / half_pi))
        else:
            v = Float32(0.5) + Float32(0.5) * sqrt(min(Float32(1.0), (-elev) / half_pi))
        var u = atan2(d.z, d.x) * inv_2pi + Float32(0.5)

        var sky = _bilerp_lut(self.sky_view_lut, ATMO_SKY_W, ATMO_SKY_H, u, v, wrap_u=True)
        if include_sun_disk and d.y > Float32(-0.02):
            var sdot = d.dot(self.params.sun_dir)
            var cos_edge = cos(self.params.sun_angular_radius * Float32(2.5))
            if sdot > cos_edge:
                var tr = self.sample_transmittance(self.params.r_bottom_km + self.params.observer_alt_km, d.y)
                var limb = min(Float32(1.0), (sdot - cos_edge) / max(Float32(1e-5), Float32(1.0) - cos_edge))
                sky = sky + Vec3(
                    tr.x * self.params.solar_irradiance.x * Float32(12.0) * limb,
                    tr.y * self.params.solar_irradiance.y * Float32(12.0) * limb,
                    tr.z * self.params.solar_irradiance.z * Float32(12.0) * limb,
                )
        return sky

    def compute_aerial_perspective_constants(self) -> Tuple[Vec3, Vec3, Vec3, Float32]:
        """Computes GPU aerial perspective sea-level coefficients `(sigma_ext_per_unit, rayleigh_albedo_E, mie_albedo_E, mie_g)`
        for `b_globals[104..114]`."""
        var r_obs = self.params.r_bottom_km + self.params.observer_alt_km
        var med = _medium_extinction_and_scattering(self.params.observer_alt_km, self.params)
        var sig_r = med[0]
        var sig_ms = med[1]
        var sig_ext = med[2]
        var sig_ext_unit = sig_ext * self.params.aerial_km_per_unit

        var t_sun = self.sample_transmittance(r_obs, self.params.sun_dir.y)
        var psi_ms = self.sample_multiscatter(r_obs, self.params.sun_dir.y)
        var e_sun = Vec3(
            self.params.solar_irradiance.x * t_sun.x,
            self.params.solar_irradiance.y * t_sun.y,
            self.params.solar_irradiance.z * t_sun.z,
        )
        var e_ms = Vec3(
            self.params.solar_irradiance.x * psi_ms.x,
            self.params.solar_irradiance.y * psi_ms.y,
            self.params.solar_irradiance.z * psi_ms.z,
        )

        # Asymptotic in-scatter = ((sigma_r * P_R + sigma_m * P_M) * E_sun + (sigma_r + sigma_m) * E_ms) / sigma_ext
        # We pre-divide by sigma_ext and fold E_ms into the isotropic/Rayleigh term!
        var r_term = Vec3(
            (sig_r.x * e_sun.x + (sig_r.x + sig_ms.x) * e_ms.x * Float32(4.0 * 3.14159265)) / max(Float32(1e-6), sig_ext.x),
            (sig_r.y * e_sun.y + (sig_r.y + sig_ms.y) * e_ms.y * Float32(4.0 * 3.14159265)) / max(Float32(1e-6), sig_ext.y),
            (sig_r.z * e_sun.z + (sig_r.z + sig_ms.z) * e_ms.z * Float32(4.0 * 3.14159265)) / max(Float32(1e-6), sig_ext.z),
        )
        var m_term = Vec3(
            (sig_ms.x * e_sun.x) / max(Float32(1e-6), sig_ext.x),
            (sig_ms.y * e_sun.y) / max(Float32(1e-6), sig_ext.y),
            (sig_ms.z * e_sun.z) / max(Float32(1e-6), sig_ext.z),
        )
        return (sig_ext_unit, r_term, m_term, self.params.mie_g)


def eval_aerial_perspective_cpu(
    surface_rgb: Vec3,
    view_dir_to_surface: Vec3,
    dist_units: Float32,
    sun_dir: Vec3,
    sig_ext_unit: Vec3,
    r_term: Vec3,
    m_term: Vec3,
    mie_g: Float32,
) -> Vec3:
    """CPU reference matching `emit_aerial_perspective_spv`: attenuates `surface_rgb` by RGB Beer-Lambert
    transmittance `T(d)` and adds Rayleigh + Cornette-Shanks Mie + Multiple-Scattering aerial perspective."""
    var d = max(Float32(0.0), dist_units)
    var tr = Vec3(exp(-sig_ext_unit.x * d), exp(-sig_ext_unit.y * d), exp(-sig_ext_unit.z * d))
    var mu = view_dir_to_surface.normalized().dot(sun_dir.normalized())
    var pr = _rayleigh_phase(mu)
    var pm = _mie_cornette_shanks_phase(mu, mie_g)
    var inscat = Vec3(
        (r_term.x * pr + m_term.x * pm) * (Float32(1.0) - tr.x),
        (r_term.y * pr + m_term.y * pm) * (Float32(1.0) - tr.y),
        (r_term.z * pr + m_term.z * pm) * (Float32(1.0) - tr.z),
    )
    return Vec3(
        surface_rgb.x * tr.x + inscat.x,
        surface_rgb.y * tr.y + inscat.y,
        surface_rgb.z * tr.z + inscat.z,
    )


def emit_aerial_perspective_spv(
    mut s: AaaSpirvModule,
    b_globals: UInt32,
    col_r: UInt32,
    col_g: UInt32,
    col_b: UInt32,
    ray_dx: UInt32,
    ray_dy: UInt32,
    ray_dz: UInt32,
    dist_u: UInt32,
    sun_x: UInt32,
    sun_y: UInt32,
    sun_z: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Applies RGB atmospheric transmittance `exp(-sigma_ext * d)` + Rayleigh / Cornette-Shanks Mie /
    Multiple-Scattering aerial perspective in SPIR-V using `b_globals[104..114]`."""
    var sx = s.load_float(b_globals, s.uint(104))
    var sy = s.load_float(b_globals, s.uint(105))
    var sz = s.load_float(b_globals, s.uint(106))
    var mie_g = s.load_float(b_globals, s.uint(107))
    var r_rx = s.load_float(b_globals, s.uint(108))
    var r_ry = s.load_float(b_globals, s.uint(109))
    var r_rz = s.load_float(b_globals, s.uint(110))
    var m_rx = s.load_float(b_globals, s.uint(112))
    var m_ry = s.load_float(b_globals, s.uint(113))
    var m_rz = s.load_float(b_globals, s.uint(114))

    var log2_e = s.float(-1.44269504)
    var tr_r = s.exp2(s.fmul(s.fmul(sx, dist_u), log2_e))
    var tr_g = s.exp2(s.fmul(s.fmul(sy, dist_u), log2_e))
    var tr_b = s.exp2(s.fmul(s.fmul(sz, dist_u), log2_e))

    var mu = s.nclamp(
        s.fma(ray_dx, sun_x, s.fma(ray_dy, sun_y, s.fmul(ray_dz, sun_z))),
        s.float(-1.0),
        s.float(1.0),
    )
    var mu2 = s.fmul(mu, mu)
    var pr = s.fmul(s.float(0.0596831036), s.fadd(s.float(1.0), mu2))

    var g2 = s.fmul(mie_g, mie_g)
    var cs_num = s.fmul(s.float(0.119366207), s.fmul(s.fsub(s.float(1.0), g2), s.fadd(s.float(1.0), mu2)))
    var cs_base = s.nmax(s.float(1e-4), s.fsub(s.fadd(s.float(1.0), g2), s.fmul(s.float(2.0), s.fmul(mie_g, mu))))
    var cs_den = s.fmul(s.fadd(s.float(2.0), g2), s.fmul(cs_base, s.fsqrt(cs_base)))
    var pm = s.fdiv(cs_num, s.nmax(s.float(1e-6), cs_den))

    var ins_r = s.fmul(s.fma(r_rx, pr, s.fmul(m_rx, pm)), s.fsub(s.float(1.0), tr_r))
    var ins_g = s.fmul(s.fma(r_ry, pr, s.fmul(m_ry, pm)), s.fsub(s.float(1.0), tr_g))
    var ins_b = s.fmul(s.fma(r_rz, pr, s.fmul(m_rz, pm)), s.fsub(s.float(1.0), tr_b))

    return (
        s.fma(col_r, tr_r, ins_r),
        s.fma(col_g, tr_g, ins_g),
        s.fma(col_b, tr_b, ins_b),
    )


def _emit_nubis_cloud_density_spv(
    mut s: AaaSpirvModule,
    px_in: UInt32,
    py_in: UInt32,
    pz_in: UInt32,
    h_frac: UInt32,
    coverage: UInt32,
    wx_off: UInt32,
    wy_off: UInt32,
    wz_off: UInt32,
) -> UInt32:
    """GPU SPIR-V Nubis 3D Perlin-Worley cumulus density matching `eval_nubis_cloud_density_cpu`."""
    var px = s.fma(wx_off, s.float(0.001), px_in)
    var py = s.fma(wy_off, s.float(0.001), py_in)
    var pz = s.fma(wz_off, s.float(0.001), pz_in)

    var m1 = s.fmul(
        s.fsin(s.fma(px, s.float(0.41), s.fmul(pz, s.float(0.29)))),
        s.fcos(s.fsub(s.fmul(pz, s.float(0.37)), s.fmul(px, s.float(0.23)))),
    )
    var m2 = s.fmul(
        s.fsin(s.fadd(s.fsub(s.fmul(px, s.float(0.83)), s.fmul(pz, s.float(0.67))), s.float(1.3))),
        s.fcos(s.fma(px, s.float(0.59), s.fmul(pz, s.float(0.79)))),
    )
    var base_sh = s.fma(m1, s.float(0.30), s.fma(m2, s.float(0.20), s.float(0.52)))

    var b1 = s.fsub(
        s.float(1.0),
        s.fabs(
            s.fmul(
                s.fsin(s.fma(px, s.float(1.85), s.fmul(py, s.float(1.35)))),
                s.fcos(s.fsub(s.fmul(pz, s.float(1.95)), s.fmul(py, s.float(1.15)))),
            )
        ),
    )
    var b2 = s.fsub(
        s.float(1.0),
        s.fabs(
            s.fmul(
                s.fsin(s.fsub(s.fmul(px, s.float(3.90)), s.fmul(pz, s.float(3.40)))),
                s.fcos(s.fma(py, s.float(3.70), s.fmul(px, s.float(2.60)))),
            )
        ),
    )
    var b3 = s.fsub(
        s.float(1.0),
        s.fabs(s.fsin(s.fma(px, s.float(7.70), s.fma(py, s.float(6.90), s.fmul(pz, s.float(7.30)))))),
    )
    var billow = s.fma(b1, s.float(0.52), s.fma(b2, s.float(0.30), s.fmul(b3, s.float(0.18))))

    var t_bot = s.nclamp(s.fmul(h_frac, s.float(1.0 / 0.14)), s.float(0.0), s.float(1.0))
    var s_bot = s.fmul(t_bot, s.fmul(t_bot, s.fma(s.float(-2.0), t_bot, s.float(3.0))))
    var t_top = s.nclamp(s.fmul(s.fsub(s.float(1.0), h_frac), s.float(1.0 / 0.46)), s.float(0.0), s.float(1.0))
    var s_top = s.fmul(t_top, s.fmul(t_top, s.fma(s.float(-2.0), t_top, s.float(3.0))))
    var h_prof = s.fmul(s_bot, s_top)

    var thresh = s.fma(coverage, s.float(-0.88), s.float(1.0))
    var raw = s.fsub(s.fmul(base_sh, h_prof), thresh)
    var erode = s.fmul(s.fsub(s.float(1.0), billow), s.fmul(s.float(0.24), s.fma(h_frac, s.float(0.65), s.float(0.35))))
    var in_band = s.logical_and(s.fgt(h_frac, s.float(0.0)), s.flt(h_frac, s.float(1.0)))
    return s.select_f(in_band, s.nmax(s.float(0.0), s.fmul(s.fsub(raw, erode), s.float(4.2))), s.float(0.0))


def emit_nubis_sky_clouds_and_celestial_spv(
    mut s: AaaSpirvModule,
    b_globals: UInt32,
    env_cube: Tuple[UInt32, UInt32],
    dx: UInt32,
    dy: UInt32,
    dz: UInt32,
    fh: UInt32,
    base_bg_r: UInt32,
    base_bg_g: UInt32,
    base_bg_b: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Full-Resolution `1280 x 720` Primary Sky Evaluator (`vis0 == 0`):
    - Zero-overhead passthrough when `cloud_coverage <= 0.01 && star_intensity <= 0.01`
    - Sub-pixel Airy-filtered Planckian blackbody starfield (`3800K..11500K`, cubed-sphere 3D coordinates)
    - Phase-lit Oren-Nayar cratered Moon disc (`Maria` + `Tycho/Copernicus` crater normals + Earthshine)
    - 12-step spherical-shell **Nubis 3D Perlin-Worley Volumetric Cumulus + 8.8 km Cirrus** with
      2-octave multiple scattering, HG silver-lining & Beer-Powder self-shadowing!"""
    var coverage = s.load_float(b_globals, s.uint(156))
    var star_int = s.load_float(b_globals, s.uint(160))

    var out_r = s.fvar(base_bg_r)
    var out_g = s.fvar(base_bg_g)
    var out_b = s.fvar(base_bg_b)

    var has_v3_sky = s.logical_or(s.fgt(coverage, s.float(0.01)), s.fgt(star_int, s.float(0.01)))
    s.begin_if(has_v3_sky)
    var zen_r = s.load_float(b_globals, s.uint(161))
    var zen_g = s.load_float(b_globals, s.uint(162))
    var zen_b = s.load_float(b_globals, s.uint(163))
    var hor_r = s.load_float(b_globals, s.uint(164))
    var hor_g = s.load_float(b_globals, s.uint(165))
    var hor_b = s.load_float(b_globals, s.uint(166))
    var has_clear_anchors = s.fgt(s.fadd(zen_r, s.fadd(zen_g, s.fadd(zen_b, hor_b))), s.float(1e-5))
    var is_deep_space = s.logical_and(s.logical_not(has_clear_anchors), s.fle(coverage, s.float(0.01)))

    s.begin_if(s.logical_or(is_deep_space, s.fgt(dy, s.float(-0.02))))
    var sun_x = s.load_float(b_globals, s.uint(48))
    var sun_y = s.load_float(b_globals, s.uint(49))
    var sun_z = s.load_float(b_globals, s.uint(50))
    var sun_ang_r = s.load_float(b_globals, s.uint(51))
    var sun_r = s.load_float(b_globals, s.uint(52))
    var sun_g = s.load_float(b_globals, s.uint(53))
    var sun_b = s.load_float(b_globals, s.uint(54))
    var ibl_int = s.load_float(b_globals, s.uint(55))
    var tan_y = s.nmax(s.float(0.1), s.load_float(b_globals, s.uint(43)))
    var time_s = s.load_float(b_globals, s.uint(123))

    # Clear-air smooth atmospheric gradient (`b_globals[161..166]` zenith/horizon clear sky + analytic sun disk)
    # Prevents double-clouding when `env_cube` also bakes Nubis clouds for water/car reflections!
    var env_low = s.sample_cube_lod(env_cube, s.vec3(dx, dy, dz), s.float(3.0))
    var mu_up = s.nclamp(dy, s.float(0.0), s.float(1.0))
    var h_blend = s.fpow(s.fsub(s.float(1.0), mu_up), s.float(2.6))
    var cos_sun = s.fma(dx, sun_x, s.fma(dy, sun_y, s.fmul(dz, sun_z)))
    var mie_glow = s.fmul(
        s.fpow(s.nclamp(s.fma(cos_sun, s.float(0.5), s.float(0.5)), s.float(0.0), s.float(1.0)), s.float(8.0)),
        s.fma(h_blend, s.float(0.35), s.float(0.06)),
    )
    var an_r = s.fma(s.fsub(hor_r, zen_r), h_blend, s.fma(sun_r, mie_glow, zen_r))
    var an_g = s.fma(s.fsub(hor_g, zen_g), h_blend, s.fma(sun_g, mie_glow, zen_g))
    var an_b = s.fma(s.fsub(hor_b, zen_b), h_blend, s.fma(sun_b, mie_glow, zen_b))

    var clr_r = s.fvar(s.select_f(has_clear_anchors, an_r, s.select_f(s.fgt(coverage, s.float(0.01)), s.fmul(s.extract_f(env_low, 0), ibl_int), base_bg_r)))
    var clr_g = s.fvar(s.select_f(has_clear_anchors, an_g, s.select_f(s.fgt(coverage, s.float(0.01)), s.fmul(s.extract_f(env_low, 1), ibl_int), base_bg_g)))
    var clr_b = s.fvar(s.select_f(has_clear_anchors, an_b, s.select_f(s.fgt(coverage, s.float(0.01)), s.fmul(s.extract_f(env_low, 2), ibl_int), base_bg_b)))

    # Crisp 1280x720 Solar Disk with limb darkening (occluded naturally by `tr_cloud` below!)
    var cos_edge = s.fcos(s.fmul(s.nmax(s.float(0.004), sun_ang_r), s.float(2.4)))
    s.begin_if(s.logical_and(s.logical_or(is_deep_space, s.fgt(dy, s.float(0.0))), s.fgt(cos_sun, cos_edge)))
    var limb = s.nclamp(s.fdiv(s.fsub(cos_sun, cos_edge), s.nmax(s.float(1e-5), s.fsub(s.float(1.0), cos_edge))), s.float(0.0), s.float(1.0))
    var d_boost = s.fmul(limb, s.float(10.0))
    s.store(clr_r, s.fma(sun_r, d_boost, s.load_f(clr_r)))
    s.store(clr_g, s.fma(sun_g, d_boost, s.load_f(clr_g)))
    s.store(clr_b, s.fma(sun_b, d_boost, s.load_f(clr_b)))
    s.end_if()

    # 1. Sub-Pixel Airy Blackbody Starfield (`3800K..11500K`) + Phase-Lit Oren-Nayar Cratered Moon (`star_int > 0.01`)
    s.begin_if(s.logical_and(s.fgt(star_int, s.float(0.01)), s.logical_or(is_deep_space, s.fgt(dy, s.float(0.01)))))
    var sky_lum = s.fma(s.load_f(clr_r), s.float(0.2126), s.fma(s.load_f(clr_g), s.float(0.7152), s.fmul(s.load_f(clr_b), s.float(0.0722))))
    var horizon_fade = s.select_f(
        is_deep_space,
        s.float(1.0),
        s.nclamp(s.fmul(dy, s.float(5.5)), s.float(0.0), s.float(1.0)),
    )
    var night_vis = s.fmul(
        star_int,
        s.fmul(
            s.nclamp(s.fsub(s.float(1.0), s.fmul(sky_lum, s.float(2.8))), s.float(0.0), s.float(1.0)),
            horizon_fade,
        ),
    )

    # Moon unit vector in upper sky (`~0.95 deg` angular radius = `0.0165 rad`)
    var m_rx = s.fma(sun_x, s.float(-0.62), s.float(0.29))
    var m_ry = s.nmax(s.float(0.42), s.fma(sun_y, s.float(-0.70), s.float(0.38)))
    var m_rz = s.fma(sun_z, s.float(-0.62), s.float(-0.34))
    var m_inv = s.inverse_sqrt(s.nmax(s.float(1e-6), s.fma(m_rx, m_rx, s.fma(m_ry, m_ry, s.fmul(m_rz, m_rz)))))
    var mx_d = s.fmul(m_rx, m_inv); var my_d = s.fmul(m_ry, m_inv); var mz_d = s.fmul(m_rz, m_inv)

    var cos_moon = s.fma(dx, mx_d, s.fma(dy, my_d, s.fmul(dz, mz_d)))
    var sin2_moon = s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fmul(cos_moon, cos_moon)))
    var r_moon = s.float(0.0165)
    var r_moon2 = s.fmul(r_moon, r_moon)
    var in_moon = s.logical_and(s.fgt(cos_moon, s.float(0.0)), s.flt(sin2_moon, r_moon2))

    # Cubed-sphere 3D starfield projection (zero polar singularity, 100% camera-rotation invariant!)
    var ax = s.fabs(dx); var ay = s.fabs(dy); var az = s.fabs(dz)
    var ma = s.nmax(ax, s.nmax(ay, az))
    var inv_ma = s.fdiv(s.float(1.0), s.nmax(s.float(1e-4), ma))
    var su_c = s.select_f(s.fge(ax, s.nmax(ay, az)), dz, dx)
    var sv_c = s.select_f(s.fge(ay, s.nmax(ax, az)), dz, dy)
    var face_s = s.select_f(s.fge(ax, s.nmax(ay, az)), s.float(13.0), s.select_f(s.fge(ay, s.nmax(ax, az)), s.float(29.0), s.float(47.0)))
    var cu_s = s.fma(s.fmul(su_c, inv_ma), s.float(0.5), s.float(0.5))
    var cv_s = s.fma(s.fmul(sv_c, inv_ma), s.float(0.5), s.float(0.5))

    var px_ang = s.fdiv(s.fmul(s.float(2.0), tan_y), s.nmax(s.float(256.0), fh))
    var star_r = s.fvar(s.float(0.0))
    var star_g = s.fvar(s.float(0.0))
    var star_b = s.fvar(s.float(0.0))

    s.begin_if(s.logical_and(s.fgt(night_vis, s.float(0.01)), s.logical_not(in_moon)))
    var grid_scales = [Float32(96.0), Float32(196.0)]
    var occ_thresh = [Float32(0.16), Float32(0.24)]
    var mag_scales = [Float32(1.45), Float32(0.55)]
    for tier in range(2):
        var gs = s.float(grid_scales[tier])
        var gu = s.fmul(cu_s, gs)
        var gv = s.fmul(cv_s, gs)
        var iu = s.ffloor(gu)
        var iv = s.ffloor(gv)
        var fu = s.fsub(gu, iu)
        var fv = s.fsub(gv, iv)

        var seed_f = s.fma(iu, s.float(127.1), s.fma(iv, s.float(311.7), s.fma(face_s, s.float(74.7), s.float(Float32(tier) * 53.3))))
        var h0 = s.fract(s.fmul(s.fsin(seed_f), s.float(43758.5453)))
        var h1 = s.fract(s.fmul(s.fsin(s.fadd(seed_f, s.float(17.13))), s.float(24634.6345)))
        var h2 = s.fract(s.fmul(s.fsin(s.fadd(seed_f, s.float(41.89))), s.float(31415.9265)))
        var h3 = s.fract(s.fmul(s.fsin(s.fadd(seed_f, s.float(83.27))), s.float(19283.7465)))

        s.begin_if(s.flt(h0, s.float(occ_thresh[tier])))
        var cx_st = s.fma(h1, s.float(0.56), s.float(0.22))
        var cy_st = s.fma(h2, s.float(0.56), s.float(0.22))
        var du_st = s.fdiv(s.fsub(fu, cx_st), gs)
        var dv_st = s.fdiv(s.fsub(fv, cy_st), gs)
        var ang2 = s.fma(du_st, du_st, s.fmul(dv_st, dv_st))
        # Sub-pixel Airy point-spread kernel `sigma_ang = 0.85 * px_ang` prevents shimmer on slow camera pan!
        var airy_sig2 = s.nmax(s.float(1e-7), s.fmul(px_ang, s.fmul(px_ang, s.float(0.95))))
        var airy = s.exp2(s.fmul(s.fdiv(ang2, airy_sig2), s.float(-1.442695)))
        var mag = s.fmul(s.fpow(h3, s.float(3.2)), s.float(mag_scales[tier]))
        var scint = s.fma(s.float(0.16), s.fsin(s.fma(time_s, s.fma(h1, s.float(3.5), s.float(2.0)), s.fmul(h2, s.float(6.283)))), s.float(0.88))
        var s_int = s.fmul(airy, s.fmul(mag, s.fmul(scint, night_vis)))

        # Planckian blackbody star chromaticity (`h1`: 0 = 3800K M/K orange-gold -> 0.5 = 5800K G cream -> 1.0 = 11500K B blue-white)
        var bb_r = s.nclamp(s.fma(s.float(-0.34), s.nmax(s.float(0.0), s.fsub(h1, s.float(0.45))), s.float(1.0)), s.float(0.66), s.float(1.0))
        var bb_g = s.nclamp(s.fma(h1, s.float(0.32), s.float(0.64)), s.float(0.64), s.float(0.95))
        var bb_b = s.nclamp(s.fma(h1, s.float(0.76), s.float(0.34)), s.float(0.36), s.float(1.0))
        s.store(star_r, s.fma(bb_r, s_int, s.load_f(star_r)))
        s.store(star_g, s.fma(bb_g, s_int, s.load_f(star_g)))
        s.store(star_b, s.fma(bb_b, s_int, s.load_f(star_b)))
        s.end_if()
    s.end_if()

    # Soft atmospheric lunar corona halo + Phase-Lit Oren-Nayar Cratered Moon Disc!
    var corona = s.fmul(
        s.exp2(s.fmul(s.nmax(s.float(0.0), s.fsub(s.fsqrt(sin2_moon), r_moon)), s.float(-75.0))),
        s.fmul(night_vis, s.float(0.14)),
    )
    s.store(star_r, s.fma(s.float(0.82), corona, s.load_f(star_r)))
    s.store(star_g, s.fma(s.float(0.88), corona, s.load_f(star_g)))
    s.store(star_b, s.fma(s.float(1.00), corona, s.load_f(star_b)))

    s.begin_if(s.logical_and(in_moon, s.fgt(night_vis, s.float(0.01))))
    # Orthonormal lunar disc tangent frame `(mr_x..z, mu_x..z)`
    var mr_rx = mz_d; var mr_rz = s.fneg(mx_d)
    var mr_inv = s.inverse_sqrt(s.nmax(s.float(1e-6), s.fma(mr_rx, mr_rx, s.fmul(mr_rz, mr_rz))))
    var mrx = s.fmul(mr_rx, mr_inv); var mrz = s.fmul(mr_rz, mr_inv)
    var mux = s.fmul(my_d, mrz)
    var muy = s.fsub(s.fmul(mz_d, mrx), s.fmul(mx_d, mrz))
    var muz = s.fneg(s.fmul(my_d, mrx))

    var inv_rm = s.fdiv(s.float(1.0), r_moon)
    var u_m = s.fmul(s.fma(dx, mrx, s.fmul(dz, mrz)), inv_rm)
    var v_m = s.fmul(s.fma(dx, mux, s.fma(dy, muy, s.fmul(dz, muz))), inv_rm)
    var z_m = s.fsqrt(s.nmax(s.float(0.001), s.fsub(s.float(1.0), s.fma(u_m, u_m, s.fmul(v_m, v_m)))))
    var edge_aa = s.nclamp(s.fmul(z_m, s.float(8.0)), s.float(0.0), s.float(1.0))

    # Multi-scale Lunar Maria (Procellarum / Tranquillitatis dark basaltic basins) + Cratered Highlands
    var maria_f = s.fma(
        s.fsin(s.fma(u_m, s.float(2.4), s.fma(v_m, s.float(1.7), s.float(0.6)))), s.float(0.55),
        s.fmul(s.fcos(s.fsub(s.fmul(u_m, s.float(4.3)), s.fmul(v_m, s.float(3.8)))), s.float(0.45)),
    )
    var maria_t = s.nclamp(s.fma(maria_f, s.float(1.8), s.float(0.35)), s.float(0.0), s.float(1.0))
    var crat_g = s.fma(u_m, s.float(9.5), s.float(12.0))
    var crat_h = s.fma(v_m, s.float(9.5), s.float(12.0))
    var ci_u = s.ffloor(crat_g); var ci_v = s.ffloor(crat_h)
    var cf_u = s.fsub(s.fsub(crat_g, ci_u), s.float(0.5))
    var cf_v = s.fsub(s.fsub(crat_h, ci_v), s.float(0.5))
    var cr_d = s.fsqrt(s.nmax(s.float(1e-5), s.fma(cf_u, cf_u, s.fmul(cf_v, cf_v))))
    var cr_seed = s.fract(s.fmul(s.fsin(s.fma(ci_u, s.float(127.1), s.fmul(ci_v, s.float(311.7)))), s.float(43758.5453)))
    var cr_rim = s.select_f(
        s.flt(cr_seed, s.float(0.55)),
        s.fmul(s.fsin(s.fmul(s.nclamp(s.fmul(cr_d, s.float(3.4)), s.float(0.0), s.float(3.14159)), s.float(2.0))), s.float(0.24)),
        s.float(0.0),
    )

    var nmx = s.fma(cf_u, cr_rim, u_m)
    var nmy = s.fma(cf_v, cr_rim, v_m)
    var nmz = z_m
    var nm_inv = s.inverse_sqrt(s.nmax(s.float(1e-5), s.fma(nmx, nmx, s.fma(nmy, nmy, s.fmul(nmz, nmz)))))
    nmx = s.fmul(nmx, nm_inv); nmy = s.fmul(nmy, nm_inv); nmz = s.fmul(nmz, nm_inv)

    # Waxing gibbous illumination (`sigma = 0.82` Oren-Nayar rough lunar regolith flattened limb!)
    var lx_m = s.float(0.64); var ly_m = s.float(0.32); var lz_m = s.float(0.698)
    var ndotl_m = s.nmax(s.float(0.0), s.fma(nmx, lx_m, s.fma(nmy, ly_m, s.fmul(nmz, lz_m))))
    var ndotv_m = s.nmax(s.float(0.05), nmz)
    # Oren-Nayar limb brightening `ndotl / mix(ndotv, 1.0, 0.35)` reproduces flat full-moon disc look!
    var oren_nayar = s.fdiv(ndotl_m, s.fma(ndotv_m, s.float(0.65), s.float(0.35)))
    var albedo_m = s.fma(s.float(-0.52), maria_t, s.fma(cr_rim, s.float(0.25), s.float(0.92)))
    var moon_lit = s.fma(oren_nayar, s.float(1.65), s.float(0.035))  # +0.035 Earthshine on dark crescent
    var m_rad = s.fmul(albedo_m, s.fmul(moon_lit, s.fmul(edge_aa, night_vis)))
    s.store(star_r, s.fma(m_rad, s.float(1.02), s.load_f(star_r)))
    s.store(star_g, s.fma(m_rad, s.float(0.99), s.load_f(star_g)))
    s.store(star_b, s.fma(m_rad, s.float(0.92), s.load_f(star_b)))
    s.end_if()

    s.store(clr_r, s.fadd(s.load_f(clr_r), s.load_f(star_r)))
    s.store(clr_g, s.fadd(s.load_f(clr_g), s.load_f(star_g)))
    s.store(clr_b, s.fadd(s.load_f(clr_b), s.load_f(star_b)))
    s.end_if()

    # 2. 12-Step Spherical-Shell Nubis 3D Volumetric Cumulus + 8.8 km Cirrus (`coverage > 0.01`)
    s.begin_if(s.logical_and(s.fgt(coverage, s.float(0.01)), s.fgt(dy, s.float(0.012))))
    var wx_off = s.load_float(b_globals, s.uint(157))
    var wy_off = s.load_float(b_globals, s.uint(158))
    var wz_off = s.load_float(b_globals, s.uint(159))

    var r_obs = s.float(6360.20)
    var r_cbot = s.float(6361.65)
    var r_ctop = s.float(6364.15)
    var r2_obs = s.fmul(r_obs, r_obs)
    var mu2_m1 = s.fsub(s.fmul(dy, dy), s.float(1.0))
    var neg_rmu = s.fneg(s.fmul(r_obs, dy))

    var disc_bot = s.fma(r2_obs, mu2_m1, s.fmul(r_cbot, r_cbot))
    var disc_top = s.fma(r2_obs, mu2_m1, s.fmul(r_ctop, r_ctop))
    var t_c0 = s.nmax(s.float(0.0), s.fadd(neg_rmu, s.fsqrt(s.nmax(s.float(0.0), disc_bot))))
    var t_c1_raw = s.nmax(s.float(0.0), s.fadd(neg_rmu, s.fsqrt(s.nmax(s.float(0.0), disc_top))))
    var t_c1 = s.nmin(s.fadd(t_c0, s.float(28.0)), t_c1_raw)
    var dist_fade = s.exp2(s.fmul(t_c0, s.float(-0.0346)))  # exp(-0.024 * t_c0)

    var tr_cloud = s.fvar(s.float(1.0))
    var cl_r = s.fvar(s.float(0.0))
    var cl_g = s.fvar(s.float(0.0))
    var cl_b = s.fvar(s.float(0.0))

    s.begin_if(s.logical_and(s.fgt(t_c1, t_c0), s.fgt(dist_fade, s.float(0.015))))
    var c_dt = s.fmul(s.fsub(t_c1, t_c0), s.float(1.0 / 12.0))

    # Dual-lobe HG phase for Octave 0 (`g=0.74, -0.28`) & Octave 1 (`g=0.33, -0.12`)
    var inv_4pi = s.float(0.07957747)
    var d0_f = s.nmax(s.float(0.01), s.fsub(s.float(1.5476), s.fmul(s.float(1.48), cos_sun)))
    var d0_b = s.nmax(s.float(0.01), s.fadd(s.float(1.0784), s.fmul(s.float(0.56), cos_sun)))
    var p_oct0 = s.fma(
        s.float(0.72), s.fdiv(s.fmul(inv_4pi, s.float(0.4524)), s.fmul(d0_f, s.fsqrt(d0_f))),
        s.fmul(s.float(0.28), s.fdiv(s.fmul(inv_4pi, s.float(0.9216)), s.fmul(d0_b, s.fsqrt(d0_b)))),
    )
    var d1_f = s.nmax(s.float(0.01), s.fsub(s.float(1.1089), s.fmul(s.float(0.66), cos_sun)))
    var p_oct1 = s.fmul(inv_4pi, s.fdiv(s.float(0.8911), s.fmul(d1_f, s.fsqrt(d1_f))))

    var amb_cr = s.fma(s.load_f(clr_r), s.float(0.42), s.float(0.015))
    var amb_cg = s.fma(s.load_f(clr_g), s.float(0.42), s.float(0.018))
    var amb_cb = s.fma(s.load_f(clr_b), s.float(0.45), s.float(0.025))

    var step_i = s.uvar(s.uint(0))
    s.begin_loop()
    var ci = s.load_u(step_i)
    s.break_if(s.logical_or(s.uge(ci, s.uint(12)), s.flt(s.load_f(tr_cloud), s.float(0.02))))
    var tc = s.fma(s.fadd(s.float_from_uint(ci), s.float(0.5)), c_dt, t_c0)
    var cpx = s.fmul(dx, tc)
    var cpy = s.fma(dy, tc, s.float(0.20))
    var cpz = s.fmul(dz, tc)
    var ry_sph = s.fma(dy, tc, r_obs)
    var r_cur = s.fsqrt(s.fma(cpx, cpx, s.fma(ry_sph, ry_sph, s.fmul(cpz, cpz))))
    var h_fr = s.fmul(s.fsub(r_cur, r_cbot), s.float(1.0 / 2.50))

    var dens = s.fmul(_emit_nubis_cloud_density_spv(s, cpx, cpy, cpz, h_fr, coverage, wx_off, wy_off, wz_off), dist_fade)
    s.begin_if(s.fgt(dens, s.float(0.002)))
    var spx = s.fma(sun_x, s.float(0.42), cpx)
    var spy = s.fma(sun_y, s.float(0.42), cpy)
    var spz = s.fma(sun_z, s.float(0.42), cpz)
    var sh_fr = s.fma(sun_y, s.float(0.17), h_fr)
    var dens_s = _emit_nubis_cloud_density_spv(s, spx, spy, spz, sh_fr, coverage, wx_off, wy_off, wz_off)
    var tau_s = s.fmul(
        s.fma(dens, s.float(0.35), s.fma(dens_s, s.float(0.85), s.fmul(s.nmax(s.float(0.0), s.fsub(s.float(1.0), h_fr)), s.float(0.35)))),
        s.float(1.6),
    )
    var neg_l2 = s.float(-1.442695)
    var ms_s = s.fma(
        s.exp2(s.fmul(tau_s, neg_l2)), p_oct0,
        s.fmul(s.float(0.58), s.fmul(s.exp2(s.fmul(tau_s, s.float(-0.6925))), p_oct1)),
    )
    var powder = s.fsub(s.float(1.0), s.fmul(s.float(0.62), s.exp2(s.fmul(tau_s, s.float(-3.4625)))))
    var sun_gain = s.fmul(ms_s, s.fmul(powder, s.float(4.2)))
    var amb_h = s.fma(h_fr, s.float(0.58), s.float(0.42))

    var sl_r = s.fma(sun_r, sun_gain, s.fmul(amb_cr, amb_h))
    var sl_g = s.fma(sun_g, sun_gain, s.fmul(amb_cg, amb_h))
    var sl_b = s.fma(sun_b, sun_gain, s.fmul(amb_cb, amb_h))

    var step_tr = s.exp2(s.fmul(s.fmul(dens, c_dt), s.float(-3.7510)))  # exp(-2.6 * dens * dt)
    var cur_tr = s.load_f(tr_cloud)
    var w_int = s.fmul(cur_tr, s.fsub(s.float(1.0), step_tr))
    s.store(cl_r, s.fma(sl_r, w_int, s.load_f(cl_r)))
    s.store(cl_g, s.fma(sl_g, w_int, s.load_f(cl_g)))
    s.store(cl_b, s.fma(sl_b, w_int, s.load_f(cl_b)))
    s.store(tr_cloud, s.fmul(cur_tr, step_tr))
    s.end_if()

    s.store(step_i, s.iadd(ci, s.uint(1)))
    s.end_loop()
    s.end_if()

    # High-altitude 8.8 km Cirrus wisps
    var r_cir = s.float(6368.80)
    var disc_cir = s.fma(r2_obs, mu2_m1, s.fmul(r_cir, r_cir))
    var t_cir = s.nmax(s.float(0.0), s.fadd(neg_rmu, s.fsqrt(s.nmax(s.float(0.0), disc_cir))))
    var cir_fade = s.exp2(s.fmul(t_cir, s.float(-0.0231)))
    var cx_p = s.fma(s.fmul(dx, t_cir), s.float(0.18), s.fmul(wx_off, s.float(0.0004)))
    var cz_p = s.fma(s.fmul(dz, t_cir), s.float(0.07), s.fmul(wz_off, s.float(0.0004)))
    var cir_n = s.fmul(
        s.fsin(s.fma(cx_p, s.float(2.3), s.fsin(s.fmul(cz_p, s.float(3.1))))),
        s.fcos(s.fsub(s.fmul(cz_p, s.float(1.9)), s.fmul(cx_p, s.float(0.8)))),
    )
    var cir_a = s.nmin(
        s.float(0.35),
        s.fmul(s.nmax(s.float(0.0), s.fsub(cir_n, s.fma(coverage, s.float(-0.28), s.float(0.58)))), s.fmul(s.float(0.42), cir_fade)),
    )
    s.begin_if(s.fgt(cir_a, s.float(0.002)))
    var ctr = s.load_f(tr_cloud)
    var cw_i = s.fmul(ctr, cir_a)
    s.store(cl_r, s.fma(s.fma(sun_r, s.float(0.38), s.fmul(s.load_f(clr_r), s.float(0.45))), cw_i, s.load_f(cl_r)))
    s.store(cl_g, s.fma(s.fma(sun_g, s.float(0.38), s.fmul(s.load_f(clr_g), s.float(0.45))), cw_i, s.load_f(cl_g)))
    s.store(cl_b, s.fma(s.fma(sun_b, s.float(0.38), s.fmul(s.load_f(clr_b), s.float(0.45))), cw_i, s.load_f(cl_b)))
    s.store(tr_cloud, s.fmul(ctr, s.fsub(s.float(1.0), cir_a)))
    s.end_if()

    var f_tr = s.load_f(tr_cloud)
    s.store(clr_r, s.fma(s.load_f(clr_r), f_tr, s.load_f(cl_r)))
    s.store(clr_g, s.fma(s.load_f(clr_g), f_tr, s.load_f(cl_g)))
    s.store(clr_b, s.fma(s.load_f(clr_b), f_tr, s.load_f(cl_b)))
    s.end_if()

    s.store(out_r, s.load_f(clr_r))
    s.store(out_g, s.load_f(clr_g))
    s.store(out_b, s.load_f(clr_b))
    s.end_if()
    s.end_if()

    return (s.load_f(out_r), s.load_f(out_g), s.load_f(out_b))


def bake_bruneton_sky_ibl(
    luts: BrunetonSkyLuts,
    base_dim: Int = 64,
    mip_levels: Int = 5,
    ggx_samples: Int = 24,
) -> BakedIblEnvironment:
    """Projects precomputed Bruneton/Hillaire `Sky-View` LUT into a `BC6H_UFLOAT`
    Split-Sum GGX cubemap + `SH9` diffuse irradiance!"""
    var res = BakedIblEnvironment()
    res.base_dim = base_dim
    res.mip_levels = mip_levels

    var sh_raw = List[Vec3](length=9, fill=Vec3(0.0, 0.0, 0.0))
    var weight_sum = Float32(0.0)
    var sh_dim = max(16, base_dim)

    for face in range(6):
        for y in range(sh_dim):
            var v = (Float32(y) + Float32(0.5)) / Float32(sh_dim)
            var ty = v * Float32(2.0) - Float32(1.0)
            for x in range(sh_dim):
                var u = (Float32(x) + Float32(0.5)) / Float32(sh_dim)
                var tx = u * Float32(2.0) - Float32(1.0)
                var tmp = Float32(1.0) + tx * tx + ty * ty
                var d_omega = Float32(4.0) / (sqrt(tmp) * tmp)
                var dir = cube_face_direction(face, u, v)
                var rad = luts.sample_sky(dir, include_sun_disk=False)

                sh_raw[0] = sh_raw[0] + rad * (Float32(0.282095) * d_omega)
                sh_raw[1] = sh_raw[1] + rad * (Float32(0.488603) * dir.y * d_omega)
                sh_raw[2] = sh_raw[2] + rad * (Float32(0.488603) * dir.z * d_omega)
                sh_raw[3] = sh_raw[3] + rad * (Float32(0.488603) * dir.x * d_omega)
                sh_raw[4] = sh_raw[4] + rad * (Float32(1.092548) * dir.x * dir.y * d_omega)
                sh_raw[5] = sh_raw[5] + rad * (Float32(1.092548) * dir.y * dir.z * d_omega)
                sh_raw[6] = sh_raw[6] + rad * (Float32(0.315392) * (Float32(3.0) * dir.z * dir.z - Float32(1.0)) * d_omega)
                sh_raw[7] = sh_raw[7] + rad * (Float32(1.092548) * dir.x * dir.z * d_omega)
                sh_raw[8] = sh_raw[8] + rad * (Float32(0.546274) * (dir.x * dir.x - dir.y * dir.y) * d_omega)
                weight_sum += d_omega

    var norm_4pi = Float32(12.5663706) / max(Float32(1e-6), weight_sum)
    for k in range(9):
        sh_raw[k] = sh_raw[k] * norm_4pi

    res.sh9[0] = sh_raw[0] * Float32(0.28209479)
    res.sh9[1] = sh_raw[1] * Float32(0.32573500)
    res.sh9[2] = sh_raw[2] * Float32(0.32573500)
    res.sh9[3] = sh_raw[3] * Float32(0.32573500)
    res.sh9[4] = sh_raw[4] * Float32(0.27313711)
    res.sh9[5] = sh_raw[5] * Float32(0.27313711)
    res.sh9[6] = sh_raw[6] * Float32(0.07884789)
    res.sh9[7] = sh_raw[7] * Float32(0.27313711)
    res.sh9[8] = sh_raw[8] * Float32(0.13656855)

    for mip in range(mip_levels):
        var m_dim = max(4, base_dim >> mip)
        var roughness = Float32(mip) / Float32(max(1, mip_levels - 1))
        for face in range(6):
            var face_hdr = List[Vec3](length=m_dim * m_dim, fill=Vec3(0.0, 0.0, 0.0))
            for y in range(m_dim):
                var v = (Float32(y) + Float32(0.5)) / Float32(m_dim)
                for x in range(m_dim):
                    var u = (Float32(x) + Float32(0.5)) / Float32(m_dim)
                    var r_dir = cube_face_direction(face, u, v)
                    if mip == 0:
                        face_hdr[y * m_dim + x] = luts.sample_sky(r_dir, include_sun_disk=True)
                    else:
                        var pref = Vec3(0.0, 0.0, 0.0)
                        var w_tot = Float32(0.0)
                        for s_i in range(ggx_samples):
                            var u1 = Float32(s_i) / Float32(ggx_samples)
                            var u2 = _radical_inverse_vdc(UInt32(s_i))
                            var h = _importance_sample_ggx(u1, u2, roughness, r_dir)
                            var l = (h * (Float32(2.0) * r_dir.dot(h)) - r_dir).normalized()
                            var ndotl = max(Float32(0.0), r_dir.dot(l))
                            if ndotl > Float32(0.0):
                                pref = pref + luts.sample_sky(l, include_sun_disk=False) * ndotl
                                w_tot += ndotl
                        face_hdr[y * m_dim + x] = pref * (Float32(1.0) / max(Float32(1e-5), w_tot))
            res.mip_face_bytes.append(encode_bc6h_hdr(face_hdr, m_dim, m_dim))

    return res^
