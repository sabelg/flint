"""Area E (`sota:E`): digital-human material models for the deferred `VisBuffer` shade pass.

Three opt-in material models ride inside the existing 24-word `AaaMaterial` record, so no SSBO layout,
binding or descriptor changes are needed:

  word 13 (`iridescence`) < 0 tags the model: `-1` skin, `-2` eye, `-3` hair. Words 14..22 (thin-film,
  sheen and subsurface slots, which these models never use) carry the model's parameters.

1. **Skin** — pre-integrated diffusion of the Christensen-Burley normalized diffusion profile
   (Christensen & Burley 2015, "Approximate Reflectance Profiles for Efficient Subsurface Scattering";
   ring integration after Penner 2011, "Pre-Integrated Skin Shading"). The 2-D Burley profile
   `R(r) = (e^{-r/d} + e^{-r/3d}) / (8 pi d r)` splits 1/4 : 3/4 between its two exponentials; each is
   projected to 1-D as a Laplace kernel of matching variance (scale `d/sqrt2`, `3d/sqrt2`) and convolved
   with `max(0, cos)` around a ring of the local curvature radius. A Laplace kernel of angular scale
   `beta` multiplies Fourier mode `w` of the clamped cosine by `1 / (1 + beta^2 w^2)`, so the shader
   evaluates the convolution in closed form as `max(0, cos)` plus the filtered change of its first
   `SOTA_E_SKIN_TERMS` cosine harmonics (Chebyshev recurrence, no trig, no LUT texture). The
   curvature comes from the analytic neighbour-ray normals the pass already builds for specular AA.
   Specular is the dual-lobe GGX used for skin in production (two roughnesses, mixed).
2. **Eye** — physically based cornea refraction (`IOR 1.376`) into an iris plane at a parabolic
   anterior-chamber depth (iris parallax), limbal darkening ring, pupil, a cornea-focused caustic on
   the side of the iris away from the light, refracted-light iris lighting, a wrapped sclera, and a
   smooth cornea GGX lobe. The eye's optical axis is recovered per pixel from the planar-projected UV
   frame (`axis = z N - (x T + y z B) / c_x`, exact for a sphere), so a rotated eye needs no per-frame
   material update.
3. **Hair** — Marschner R / TT / TRT lobes with the closed-form azimuthal terms from Karis 2016
   ("Physically Based Hair Shading in Unreal", SIGGRAPH course), plus the local back-scatter term of
   the dual-scattering approximation (Zinke et al. 2008, "Dual Scattering Approximation for Fast
   Multiple Scattering in Hair"). The forward/backward attenuations `a_f`, `a_b` are integrated on the
   CPU per material and packed as the back-scatter amplitude `A_b` (RGB).

Every model has a CPU reference (`*_cpu`) and a SPIR-V emitter that mirrors it term by term.
The pass only runs the models when `AAA_FLAG_SOTA_E_CHARACTERS` is set in `globals[99]`
(`FlintAaaVulkanRenderer.sota_e_character_shading`, default on); otherwise skin falls back to the
legacy wrap-SSS lobe and eyes / hair to the standard BSDF.
"""
from std.math import abs, atan2, cos, exp, log, max, min, sin, sqrt
from flint.core.math import Vec3
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.spirv import AaaSpirvModule
from flint.render.aaa_visbuffer import emit_vec3_cross, emit_vec3_dot, emit_vec3_normalize

comptime SOTA_E_MODEL_STANDARD: Int = 0
comptime SOTA_E_MODEL_SKIN: Int = 1
comptime SOTA_E_MODEL_EYE: Int = 2
comptime SOTA_E_MODEL_HAIR: Int = 3

# `globals[99]` render flag (high bit chosen to stay clear of other areas' flags).
comptime AAA_FLAG_SOTA_E_CHARACTERS: UInt32 = 1 << 20

comptime SOTA_E_SKIN_TERMS: Int = 8
comptime SOTA_E_CORNEA_IOR: Float32 = 1.376
comptime SOTA_E_PI: Float32 = 3.14159265
comptime SOTA_E_INV_PI: Float32 = 0.318309886
comptime SOTA_E_SQRT_HALF: Float32 = 0.70710678


# ---------------------------------------------------------------------------------------------
# Parameters & packing
# ---------------------------------------------------------------------------------------------
@fieldwise_init
struct SotaSkinParams(Copyable, Movable):
    var scatter_radius_m: Float32   # mean free path of the red channel in metres (skin ~ 0.004 m at scale 1)
    var mfp_rgb: Vec3               # relative per-channel mean free path (red scatters furthest)
    var lobe_mix: Float32           # weight of the second (rougher) specular lobe
    var lobe0_scale: Float32        # roughness multiplier of lobe 0
    var lobe1_scale: Float32        # roughness multiplier of lobe 1
    var curvature_scale: Float32    # artist multiplier on the measured curvature radius

    @staticmethod
    def human() -> Self:
        return Self(Float32(0.0045), Vec3(1.0, 0.37, 0.19), Float32(0.15), Float32(0.75), Float32(1.30), Float32(1.0))


@fieldwise_init
struct SotaEyeParams(Copyable, Movable):
    var iris_radius_uv: Float32     # iris radius in UV units around the pupil centre (0.5, 0.5)
    var iris_depth: Float32         # anterior chamber depth at the pupil, in eye radii
    var limbus_width_uv: Float32    # width of the dark limbal ring in UV units
    var uv_per_radius: Float32      # UV units per eye radius in the planar front projection
    var caustic_strength: Float32   # cornea-focused caustic gain on the far side of the iris
    var pupil_frac: Float32         # pupil radius as a fraction of the iris radius

    @staticmethod
    def human() -> Self:
        return Self(Float32(0.24), Float32(0.22), Float32(0.035), Float32(0.5), Float32(1.0), Float32(0.34))


@fieldwise_init
struct SotaHairParams(Copyable, Movable):
    var shift_rad: Float32          # cuticle tilt alpha (negative: R lobe shifts toward the root)
    var long_rough_rad: Float32     # longitudinal width beta of the R lobe (radians)
    var tangent_along_v: Bool       # strand runs along +V instead of +U in texture space
    var backscatter_density: Float32  # d_b density factor of the dual-scattering back-scatter term
    var ior: Float32                # hair fibre IOR (1.55)

    @staticmethod
    def human() -> Self:
        return Self(Float32(-0.06), Float32(0.18), False, Float32(0.7), Float32(1.55))


def sota_e_skin_material(base_color: Vec3, roughness: Float32, p: SotaSkinParams) -> AaaMaterial:
    var m = AaaMaterial(base_color=base_color, roughness=roughness, metallic=0.0)
    m.specular_factor = Float32(0.7)  # F0 ~ 0.028 (IOR 1.4)
    m.iridescence = -Float32(SOTA_E_MODEL_SKIN)
    m.iridescence_thickness_nm = max(Float32(1e-4), p.scatter_radius_m)
    m.iridescence_ior = max(Float32(0.0), min(Float32(1.0), p.lobe_mix))
    m.sheen_color = Vec3(max(Float32(0.05), p.lobe0_scale), max(Float32(0.05), p.lobe1_scale), max(Float32(0.05), p.curvature_scale))
    m.sheen_roughness = Float32(0.05)
    m.subsurface_color = p.mfp_rgb
    m.subsurface_amount = Float32(1.0)
    return m^


def sota_e_eye_material(iris_color: Vec3, sclera_color: Vec3, cornea_roughness: Float32, p: SotaEyeParams) -> AaaMaterial:
    var m = AaaMaterial(base_color=iris_color, roughness=cornea_roughness, metallic=0.0)
    m.specular_factor = Float32(0.62)  # F0 ~ 0.025 (cornea IOR 1.376)
    m.iridescence = -Float32(SOTA_E_MODEL_EYE)
    m.iridescence_thickness_nm = p.iris_radius_uv
    m.iridescence_ior = p.iris_depth
    m.sheen_color = Vec3(p.limbus_width_uv, p.uv_per_radius, p.caustic_strength)
    m.sheen_roughness = max(Float32(0.05), p.pupil_frac)
    m.subsurface_color = sclera_color
    m.subsurface_amount = Float32(1.0)
    return m^


def sota_e_hair_backscatter_amp(color: Vec3, p: SotaHairParams) -> Vec3:
    """Dual-scattering back-scatter amplitude `d_b * A_b` (RGB), Zinke 2008 eq. 6 with `a_f`, `a_b`
    integrated over azimuth from the same R/TT/TRT lobes the shader evaluates (at `theta_d = 0`)."""
    var out = Vec3(0.0, 0.0, 0.0)
    for c in range(3):
        var col = color.x if c == 0 else (color.y if c == 1 else color.z)
        var af = Float32(0.0)
        var ab = Float32(0.0)
        comptime NPHI = 256
        for k in range(NPHI):
            var phi = (Float32(k) + Float32(0.5)) / Float32(NPHI) * Float32(2.0) * SOTA_E_PI - SOTA_E_PI
            var a = _hair_azimuthal_energy_cpu(cos(phi), Float32(1.0), Float32(1.0), col, p.ior)
            var w = Float32(2.0) * SOTA_E_PI / Float32(NPHI)
            if abs(phi) > SOTA_E_PI * Float32(0.5):
                af += a * w
            else:
                ab += a * w
        af = min(Float32(0.95), af)
        ab = min(Float32(0.95), ab)
        var om = Float32(1.0) - af * af
        var a1 = ab * af * af / om
        var a3 = ab * ab * ab * af * af / (om * om * om)
        var amp = p.backscatter_density * (a1 + a3)
        if c == 0:
            out.x = amp
        elif c == 1:
            out.y = amp
        else:
            out.z = amp
    return out


def sota_e_hair_material(hair_color: Vec3, p: SotaHairParams) -> AaaMaterial:
    var m = AaaMaterial(base_color=hair_color, roughness=Float32(0.35), metallic=0.0)
    m.iridescence = -Float32(SOTA_E_MODEL_HAIR)
    m.iridescence_thickness_nm = p.shift_rad
    m.iridescence_ior = max(Float32(0.02), p.long_rough_rad)
    m.sheen_color = Vec3(Float32(1.0) if p.tangent_along_v else Float32(0.0), p.backscatter_density, p.ior)
    m.sheen_roughness = Float32(0.05)
    m.subsurface_color = sota_e_hair_backscatter_amp(hair_color, p)
    m.subsurface_amount = Float32(1.0)
    return m^


def sota_e_material_model(m: AaaMaterial) -> Int:
    if m.iridescence >= Float32(-0.5):
        return SOTA_E_MODEL_STANDARD
    var k = Int(-m.iridescence + Float32(0.5))
    if k >= SOTA_E_MODEL_SKIN and k <= SOTA_E_MODEL_HAIR:
        return k
    return SOTA_E_MODEL_STANDARD


# ---------------------------------------------------------------------------------------------
# CPU references
# ---------------------------------------------------------------------------------------------
def _sat(x: Float32) -> Float32:
    return max(Float32(0.0), min(Float32(1.0), x))


def _schlick(f0: Float32, cos_t: Float32) -> Float32:
    var om = Float32(1.0) - _sat(cos_t)
    var om2 = om * om
    return f0 + (Float32(1.0) - f0) * om2 * om2 * om


def _ggx_spec_cpu(ndoth: Float32, ndotv: Float32, ndotl: Float32, rough: Float32) -> Float32:
    """`D * Vis` of height-correlated GGX (matches `emit_direct_multilobe_bsdf`)."""
    var alpha = max(Float32(0.002), rough * rough)
    var a2 = alpha * alpha
    var dd = ndoth * ndoth * (a2 - Float32(1.0)) + Float32(1.0)
    var d = a2 / max(Float32(1e-6), SOTA_E_PI * dd * dd)
    var gv = ndotl * sqrt(ndotv * ndotv * (Float32(1.0) - a2) + a2)
    var gl = ndotv * sqrt(ndotl * ndotl * (Float32(1.0) - a2) + a2)
    return d * Float32(0.5) / max(Float32(1e-5), gv + gl)


def sota_e_burley_d(mfp: Float32, albedo: Float32) -> Float32:
    """Christensen-Burley shape parameter `d = mfp / s(A)` (searchlight fit `s = 1.85 - A + 7|A - 0.8|^3`)."""
    var a = _sat(albedo)
    var dv = abs(a - Float32(0.8))
    var sfit = Float32(1.85) - a + Float32(7.0) * dv * dv * dv
    return max(Float32(1e-5), mfp) / sfit


def _burley_ring_quadrature(cos_theta: Float32, radius: Float32, d: Float32, taps: Int) -> Float32:
    """Reference: stratified inverse-CDF quadrature of the Laplace-mixture ring convolution."""
    var ct = max(Float32(-1.0), min(Float32(1.0), cos_theta))
    var st = sqrt(max(Float32(0.0), Float32(1.0) - ct * ct))
    var inv_r = Float32(1.0) / max(Float32(1e-5), radius)
    var acc = Float32(0.0)
    for lobe in range(2):
        var b = d * SOTA_E_SQRT_HALF * (Float32(1.0) if lobe == 0 else Float32(3.0))
        var w = Float32(0.25) if lobe == 0 else Float32(0.75)
        var sum = Float32(0.0)
        for k in range(taps):
            var u = (Float32(k) + Float32(0.5)) / Float32(taps)
            var x = -b * log(Float32(1.0) - u) * inv_r
            var cx = cos(x)
            var sx = sin(x)
            sum += max(Float32(0.0), ct * cx - st * sx) + max(Float32(0.0), ct * cx + st * sx)
        acc += w * sum * Float32(0.5) / Float32(taps)
    return acc


def _burley_ring_fourier(cos_theta: Float32, radius: Float32, d: Float32, terms: Int) -> Float32:
    """Closed form: `max(0,cos)` = 1/pi + cos/2 + (2/pi) sum (-1)^(k+1) cos(2k t) / (4k^2 - 1); the Laplace
    kernel scales mode `w` by `1 / (1 + beta^2 w^2)`. Returned as `max(0, cos)` plus the filtered change of
    the first `terms` harmonics (error <= ~1 / (2 pi terms) at the kink, ~0 for a sharp profile)."""
    var c = max(Float32(-1.0), min(Float32(1.0), cos_theta))
    var c2 = Float32(2.0) * c * c - Float32(1.0)
    var beta1 = d * SOTA_E_SQRT_HALF / max(Float32(1e-5), radius)
    var out = max(Float32(0.0), c)
    for lobe in range(2):
        var beta = beta1 * (Float32(1.0) if lobe == 0 else Float32(3.0))
        var w = Float32(0.25) if lobe == 0 else Float32(0.75)
        var b2 = beta * beta
        var acc = Float32(0.5) * c * (Float32(1.0) / (Float32(1.0) + b2) - Float32(1.0))
        var t_prev = Float32(1.0)
        var t_cur = c2
        for k in range(1, terms + 1):
            var kk = Float32(4 * k * k)
            var sgn = Float32(1.0) if (k % 2) == 1 else Float32(-1.0)
            var coef = sgn * Float32(2.0) * SOTA_E_INV_PI / (kk - Float32(1.0))
            acc += coef * t_cur * (Float32(1.0) / (Float32(1.0) + kk * b2) - Float32(1.0))
            var t_next = Float32(2.0) * c2 * t_cur - t_prev
            t_prev = t_cur
            t_cur = t_next
        out += w * acc
    return max(Float32(0.0), out)


def sota_e_burley_preintegrated_ref(cos_theta: Float32, radius: Float32, d: Float32) -> Float32:
    """Converged (4096-tap quadrature) ring integral of the projected Burley profile convolved with `max(0, cos)`."""
    return _burley_ring_quadrature(cos_theta, radius, d, 4096)


def sota_e_burley_preintegrated_fast(cos_theta: Float32, radius: Float32, d: Float32) -> Float32:
    """The closed-form `SOTA_E_SKIN_TERMS`-harmonic evaluation the shader runs."""
    return _burley_ring_fourier(cos_theta, radius, d, SOTA_E_SKIN_TERMS)


def sota_e_skin_direct_cpu(
    n: Vec3, v: Vec3, l: Vec3, albedo: Vec3, f0: Float32, rough: Float32, curv_radius: Float32, p: SotaSkinParams
) -> Vec3:
    """Outgoing radiance per unit sun irradiance: pre-integrated Burley diffuse + dual-lobe GGX."""
    var h = (v + l).normalized()
    var ndotv = max(Float32(1e-4), n.dot(v))
    var raw_nl = n.dot(l)
    var ndotl = max(Float32(0.0), raw_nl)
    var ndoth = max(Float32(0.0), n.dot(h))
    var vdoth = max(Float32(0.0), v.dot(h))
    var f = _schlick(f0, vdoth)
    var s0 = _ggx_spec_cpu(ndoth, ndotv, ndotl, max(Float32(0.02), rough * p.lobe0_scale))
    var s1 = _ggx_spec_cpu(ndoth, ndotv, ndotl, min(Float32(1.0), rough * p.lobe1_scale))
    var spec = (s0 * (Float32(1.0) - p.lobe_mix) + s1 * p.lobe_mix) * f * ndotl
    var r = curv_radius * p.curvature_scale
    var kd = (Float32(1.0) - f) * SOTA_E_INV_PI
    var dr = sota_e_burley_preintegrated_fast(raw_nl, r, sota_e_burley_d(p.scatter_radius_m * p.mfp_rgb.x, albedo.x))
    var dg = sota_e_burley_preintegrated_fast(raw_nl, r, sota_e_burley_d(p.scatter_radius_m * p.mfp_rgb.y, albedo.y))
    var db = sota_e_burley_preintegrated_fast(raw_nl, r, sota_e_burley_d(p.scatter_radius_m * p.mfp_rgb.z, albedo.z))
    return Vec3(albedo.x * kd * dr + spec, albedo.y * kd * dg + spec, albedo.z * kd * db + spec)


def sota_e_refract_cpu(i: Vec3, n: Vec3, eta: Float32) -> Vec3:
    """GLSL `refract(I, N, eta)` (I points toward the surface)."""
    var ndi = n.dot(i)
    var k = Float32(1.0) - eta * eta * (Float32(1.0) - ndi * ndi)
    if k < Float32(0.0):
        return i - n * (Float32(2.0) * ndi)
    return i * eta - n * (eta * ndi + sqrt(k))


@fieldwise_init
struct SotaEyeFrame(Copyable, Movable):
    var axis: Vec3    # optical axis (outward)
    var ex: Vec3      # planar +x (along +U)
    var ey: Vec3      # planar +y (along +V)


def sota_e_eye_frame_cpu(n: Vec3, t: Vec3, b: Vec3, u: Float32, v: Float32, uv_per_radius: Float32) -> SotaEyeFrame:
    """Recovers the eye frame from the orthonormal shading frame `(N, T, B)` at UV `(u, v)` of a planar
    front projection (`T = dP/du` Gram-Schmidt'd against `N`, `B = +-N x T` toward `dP/dv`).

    With planar coords `x, y` (eye radii), `z = sqrt(1 - x^2 - y^2)`, `c_x = sqrt(z^2 + x^2)`,
    `c_y = sqrt(z^2 + y^2)`, a sphere has `A.N = z`, `A.T = -x / c_x` and `A.B = -y z / c_x`, so
    `A = z N - (x / c_x) T - (y z / c_x) B` exactly (`c_x^2 = 1 - y^2`); `Ex = (c_x T + x A) / z`, `Ey = +-A x Ex`."""
    var x = (u - Float32(0.5)) / uv_per_radius
    var y = (v - Float32(0.5)) / uv_per_radius
    var z = sqrt(max(Float32(0.02), Float32(1.0) - x * x - y * y))
    var cx = sqrt(z * z + x * x)
    var a = (n * z - t * (x / cx) - b * (y * z / cx)).normalized()
    var ex = ((t * cx + a * x) * (Float32(1.0) / z)).normalized()
    var hand = Float32(1.0) if b.dot(n.cross(t)) >= Float32(0.0) else Float32(-1.0)
    var ey = a.cross(ex) * hand
    return SotaEyeFrame(a, ex, ey)


@fieldwise_init
struct SotaEyeSurface(Copyable, Movable):
    var u: Float32          # refracted iris UV
    var v: Float32
    var iris_mask: Float32  # 1 inside the iris (incl. pupil), 0 on the sclera
    var limbus: Float32     # limbal darkening multiplier
    var pupil: Float32      # 1 inside the pupil


def sota_e_eye_surface_cpu(n: Vec3, view: Vec3, fr: SotaEyeFrame, u: Float32, v: Float32, p: SotaEyeParams) -> SotaEyeSurface:
    """Cornea refraction + parabolic anterior chamber -> refracted iris UV, iris mask, limbus, pupil."""
    var du = u - Float32(0.5)
    var dv = v - Float32(0.5)
    var r_uv = sqrt(du * du + dv * dv)
    var rr = r_uv / max(Float32(1e-4), p.iris_radius_uv)
    var h = p.iris_depth * max(Float32(0.0), Float32(1.0) - rr * rr)
    var rd = sota_e_refract_cpu(-view, n, Float32(1.0) / SOTA_E_CORNEA_IOR)
    var along = max(Float32(0.05), -rd.dot(fr.axis))
    var tlen = h / along
    var ou = rd.dot(fr.ex) * tlen * p.uv_per_radius
    var ov = rd.dot(fr.ey) * tlen * p.uv_per_radius
    var ru = u + ou
    var rv = v + ov
    var qu = ru - Float32(0.5)
    var qv = rv - Float32(0.5)
    var rq = sqrt(qu * qu + qv * qv)
    var edge0 = p.iris_radius_uv - p.limbus_width_uv
    var mask = Float32(1.0) - _smoothstep(edge0, p.iris_radius_uv, rq)
    var ld = (rq - (p.iris_radius_uv - Float32(0.5) * p.limbus_width_uv)) / max(Float32(1e-4), p.limbus_width_uv)
    var limbus = Float32(1.0) - Float32(0.65) * exp(-ld * ld * Float32(2.0))
    var pr = p.pupil_frac * p.iris_radius_uv
    var pupil = Float32(1.0) - _smoothstep(pr * Float32(0.92), pr * Float32(1.08), rq)
    return SotaEyeSurface(ru, rv, mask, limbus, pupil)


def _smoothstep(e0: Float32, e1: Float32, x: Float32) -> Float32:
    var t = _sat((x - e0) / max(Float32(1e-6), e1 - e0))
    return t * t * (Float32(3.0) - Float32(2.0) * t)


def sota_e_iris_procedural_cpu(iris_color: Vec3, qu: Float32, qv: Float32, p: SotaEyeParams) -> Vec3:
    """Procedural iris albedo (radial stroma fibres + darker collarette) used when no albedo texture is bound."""
    var rq = sqrt(qu * qu + qv * qv) / max(Float32(1e-4), p.iris_radius_uv)
    var ang = _atan2_cpu(qv, qu)
    var fib = Float32(0.5) + Float32(0.5) * sin(ang * Float32(41.0) + rq * Float32(9.0)) * sin(ang * Float32(13.0) - rq * Float32(5.0))
    var collar = Float32(1.0) - Float32(0.35) * exp(-((rq - Float32(0.55)) * (rq - Float32(0.55))) * Float32(60.0))
    var g = (Float32(0.72) + Float32(0.5) * fib) * collar
    return iris_color * g


def _atan2_cpu(y: Float32, x: Float32) -> Float32:
    return atan2(y, x)


def sota_e_eye_direct_cpu(
    n: Vec3, v: Vec3, l: Vec3, fr: SotaEyeFrame, surf: SotaEyeSurface, albedo: Vec3, f0: Float32, rough: Float32,
    p: SotaEyeParams,
) -> Vec3:
    var h = (v + l).normalized()
    var ndotv = max(Float32(1e-4), n.dot(v))
    var raw_nl = n.dot(l)
    var ndotl = max(Float32(0.0), raw_nl)
    var spec = _ggx_spec_cpu(max(Float32(0.0), n.dot(h)), ndotv, ndotl, rough) * _schlick(f0, max(Float32(0.0), v.dot(h))) * ndotl
    var lr = sota_e_refract_cpu(-l, n, Float32(1.0) / SOTA_E_CORNEA_IOR)
    var iris_nl = _sat(-lr.dot(fr.axis)) * (Float32(1.0) if raw_nl > Float32(0.0) else Float32(0.0))
    var lpx = l.dot(fr.ex)
    var lpy = l.dot(fr.ey)
    var lpl = sqrt(lpx * lpx + lpy * lpy) + Float32(1e-4)
    var qu = surf.u - Float32(0.5)
    var qv = surf.v - Float32(0.5)
    var ql = sqrt(qu * qu + qv * qv) + Float32(1e-4)
    var opp = _sat(-(qu * lpx + qv * lpy) / (ql * lpl))
    var graze = Float32(1.0) - _sat(l.dot(fr.axis))
    # Light entering the light-facing cornea focuses on the far iris: gate by the eye (not local) facing.
    var front = _sat((l.dot(fr.axis) + Float32(0.15)) * Float32(4.0))
    var caustic = p.caustic_strength * opp * graze * _sat(ql / max(Float32(1e-4), p.iris_radius_uv)) * front
    var t_in = (Float32(1.0) - _schlick(f0, ndotl)) * (Float32(1.0) - _schlick(f0, ndotv))
    var iris_k = (iris_nl + caustic) * t_in * SOTA_E_INV_PI
    var sclera_k = _sat((raw_nl + Float32(0.2)) / Float32(1.2)) * SOTA_E_INV_PI
    var k = sclera_k + (iris_k - sclera_k) * surf.iris_mask
    return Vec3(albedo.x * k + spec, albedo.y * k + spec, albedo.z * k + spec)


def _hair_fresnel(ior: Float32, cos_t: Float32) -> Float32:
    var f0 = (Float32(1.0) - ior) / (Float32(1.0) + ior)
    return _schlick(f0 * f0, cos_t)


def _hair_azimuthal_energy_cpu(cos_phi: Float32, cos_td: Float32, vdotl_half: Float32, col: Float32, ior: Float32) -> Float32:
    """Sum of `N_p * A_p` for one channel (R + TT + TRT), the azimuthal part of the Karis lobes."""
    var cphi_h = sqrt(_sat(Float32(0.5) + Float32(0.5) * cos_phi))
    var n_r = Float32(0.25) * cphi_h * _hair_fresnel(ior, vdotl_half)
    var a = Float32(1.0) / (Float32(1.19) / cos_td + Float32(0.36) * cos_td)
    var hh = (Float32(1.0) + a * (Float32(0.6) - Float32(0.8) * cos_phi)) * cphi_h
    var f_tt = _hair_fresnel(ior, cos_td * sqrt(_sat(Float32(1.0) - hh * hh)))
    var t_tt = _powf(max(Float32(1e-4), col), sqrt(_sat(Float32(1.0) - hh * hh * a * a)) / (Float32(2.0) * cos_td))
    var n_tt = exp(Float32(-3.65) * cos_phi - Float32(3.98))
    var f_trt = _hair_fresnel(ior, cos_td * Float32(0.5))
    var t_trt = _powf(max(Float32(1e-4), col), Float32(0.8) / cos_td)
    var n_trt = exp(Float32(17.0) * cos_phi - Float32(16.78))
    var om_tt = Float32(1.0) - f_tt
    var om_trt = Float32(1.0) - f_trt
    return n_r + n_tt * om_tt * om_tt * t_tt + n_trt * om_trt * om_trt * f_trt * t_trt


def _powf(b: Float32, e: Float32) -> Float32:
    return exp(e * log(max(Float32(1e-20), b)))


def _gauss(beta: Float32, x: Float32) -> Float32:
    return exp(-Float32(0.5) * x * x / (beta * beta)) / (beta * Float32(2.50662827))


def sota_e_hair_direct_cpu(t_in: Vec3, v: Vec3, l: Vec3, color: Vec3, back_amp: Vec3, p: SotaHairParams) -> Vec3:
    """Marschner R/TT/TRT (Karis closed forms) + dual-scattering back-scatter, times `cos(theta_i)`."""
    var t = t_in.normalized()
    var sin_i = max(Float32(-1.0), min(Float32(1.0), l.dot(t)))
    var sin_o = max(Float32(-1.0), min(Float32(1.0), v.dot(t)))
    var cos_i = sqrt(_sat(Float32(1.0) - sin_i * sin_i))
    var cos_o = sqrt(_sat(Float32(1.0) - sin_o * sin_o))
    var cos_td = sqrt(_sat(Float32(0.5) + Float32(0.5) * (cos_i * cos_o + sin_i * sin_o)))
    cos_td = max(Float32(0.05), cos_td)
    var lp = l - t * sin_i
    var vp = v - t * sin_o
    var cos_phi = lp.dot(vp) / sqrt(max(Float32(1e-8), lp.length_squared() * vp.length_squared()))
    cos_phi = max(Float32(-1.0), min(Float32(1.0), cos_phi))
    var xs = sin_i + sin_o
    var a = p.shift_rad
    var b = p.long_rough_rad
    var m_r = _gauss(b, xs - a)
    var m_tt = _gauss(b * Float32(0.5), xs + a * Float32(0.5))
    var m_trt = _gauss(b * Float32(2.0), xs + a * Float32(1.5))
    var vdotl_half = sqrt(_sat(Float32(0.5) + Float32(0.5) * v.dot(l)))
    var cphi_h = sqrt(_sat(Float32(0.5) + Float32(0.5) * cos_phi))
    var f_r = m_r * Float32(0.25) * cphi_h * _hair_fresnel(p.ior, vdotl_half)
    var aa = Float32(1.0) / (Float32(1.19) / cos_td + Float32(0.36) * cos_td)
    var hh = (Float32(1.0) + aa * (Float32(0.6) - Float32(0.8) * cos_phi)) * cphi_h
    var om_ftt = Float32(1.0) - _hair_fresnel(p.ior, cos_td * sqrt(_sat(Float32(1.0) - hh * hh)))
    var e_tt = sqrt(_sat(Float32(1.0) - hh * hh * aa * aa)) / (Float32(2.0) * cos_td)
    var n_tt = exp(Float32(-3.65) * cos_phi - Float32(3.98))
    var f_trt_f = _hair_fresnel(p.ior, cos_td * Float32(0.5))
    var om_trt = Float32(1.0) - f_trt_f
    var e_trt = Float32(0.8) / cos_td
    var n_trt = exp(Float32(17.0) * cos_phi - Float32(16.78))
    var g_back = Float32(2.0) * _gauss(b * Float32(2.0), xs - a) / (SOTA_E_PI * cos_td * cos_td)
    var out = Vec3(0.0, 0.0, 0.0)
    for c in range(3):
        var col = max(Float32(1e-4), color.x if c == 0 else (color.y if c == 1 else color.z))
        var amp = back_amp.x if c == 0 else (back_amp.y if c == 1 else back_amp.z)
        var tt = m_tt * n_tt * om_ftt * om_ftt * _powf(col, e_tt)
        var trt = m_trt * n_trt * om_trt * om_trt * f_trt_f * _powf(col, e_trt)
        var val = (f_r + tt + trt + amp * g_back) * cos_i
        if c == 0:
            out.x = val
        elif c == 1:
            out.y = val
        else:
            out.z = val
    return out


# ---------------------------------------------------------------------------------------------
# SPIR-V emitters
# ---------------------------------------------------------------------------------------------
@fieldwise_init
struct SotaECharDecode(Copyable, Movable):
    """SSA ids of the decoded model (all defined at function scope, usable anywhere after the decode)."""
    var model_f: UInt32     # 0 standard, 1 skin, 2 eye, 3 hair (as float)
    var enabled: UInt32     # bool: `AAA_FLAG_SOTA_E_CHARACTERS` set
    var tagged: UInt32      # bool: the material carries a model tag
    var is_skin: UInt32
    var is_eye: UInt32
    var is_hair: UInt32
    var p14: UInt32
    var p15: UInt32
    var p16: UInt32
    var p17: UInt32
    var p18: UInt32
    var p19: UInt32
    var p20: UInt32
    var p21: UInt32
    var p22: UInt32


def emit_sota_e_material_decode(
    mut s: AaaSpirvModule, b_materials: UInt32, m_base: UInt32, irid_w: UInt32, r_flags: UInt32
) -> SotaECharDecode:
    var tagged = s.flt(irid_w, s.float(-0.5))
    var model_f = s.select_f(tagged, s.ffloor(s.fadd(s.fneg(irid_w), s.float(0.5))), s.float(0.0))
    var enabled = s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_SOTA_E_CHARACTERS)), s.uint(0))
    var on = s.logical_and(enabled, tagged)
    var is_skin = s.logical_and(on, s.flt(s.fabs(s.fsub(model_f, s.float(1.0))), s.float(0.25)))
    var is_eye = s.logical_and(on, s.flt(s.fabs(s.fsub(model_f, s.float(2.0))), s.float(0.25)))
    var is_hair = s.logical_and(on, s.flt(s.fabs(s.fsub(model_f, s.float(3.0))), s.float(0.25)))
    return SotaECharDecode(
        model_f, enabled, tagged, is_skin, is_eye, is_hair,
        s.load_float(b_materials, s.iadd(m_base, s.uint(14))),
        s.load_float(b_materials, s.iadd(m_base, s.uint(15))),
        s.load_float(b_materials, s.iadd(m_base, s.uint(16))),
        s.load_float(b_materials, s.iadd(m_base, s.uint(17))),
        s.load_float(b_materials, s.iadd(m_base, s.uint(18))),
        s.load_float(b_materials, s.iadd(m_base, s.uint(19))),
        s.load_float(b_materials, s.iadd(m_base, s.uint(20))),
        s.load_float(b_materials, s.iadd(m_base, s.uint(21))),
        s.load_float(b_materials, s.iadd(m_base, s.uint(22))),
    )


def _e_sat(mut s: AaaSpirvModule, x: UInt32) -> UInt32:
    return s.nclamp(x, s.float(0.0), s.float(1.0))


def _e_schlick(mut s: AaaSpirvModule, f0: UInt32, cos_t: UInt32) -> UInt32:
    var om = s.fsub(s.float(1.0), _e_sat(s, cos_t))
    var om2 = s.fmul(om, om)
    return s.fma(s.fsub(s.float(1.0), f0), s.fmul(s.fmul(om2, om2), om), f0)


def _e_ggx(mut s: AaaSpirvModule, ndoth: UInt32, ndotv: UInt32, ndotl: UInt32, rough: UInt32) -> UInt32:
    var alpha = s.nmax(s.float(0.002), s.fmul(rough, rough))
    var a2 = s.fmul(alpha, alpha)
    var dd = s.fma(s.fmul(ndoth, ndoth), s.fsub(a2, s.float(1.0)), s.float(1.0))
    var d = s.fdiv(a2, s.nmax(s.float(1e-6), s.fmul(s.float(SOTA_E_PI), s.fmul(dd, dd))))
    var gv = s.fmul(ndotl, s.fsqrt(s.fma(s.fmul(ndotv, ndotv), s.fsub(s.float(1.0), a2), a2)))
    var gl = s.fmul(ndotv, s.fsqrt(s.fma(s.fmul(ndotl, ndotl), s.fsub(s.float(1.0), a2), a2)))
    return s.fmul(d, s.fdiv(s.float(0.5), s.nmax(s.float(1e-5), s.fadd(gv, gl))))


def _e_burley_d(mut s: AaaSpirvModule, mfp: UInt32, albedo: UInt32) -> UInt32:
    var a = _e_sat(s, albedo)
    var dv = s.fabs(s.fsub(a, s.float(0.8)))
    var sfit = s.fma(s.float(7.0), s.fmul(dv, s.fmul(dv, dv)), s.fsub(s.float(1.85), a))
    return s.fdiv(s.nmax(s.float(1e-5), mfp), sfit)


def _e_burley_ring(mut s: AaaSpirvModule, ct_raw: UInt32, inv_r: UInt32, d: UInt32) -> UInt32:
    """Mirrors `_burley_ring_fourier` with `SOTA_E_SKIN_TERMS` harmonics."""
    var c = s.nclamp(ct_raw, s.float(-1.0), s.float(1.0))
    var c2 = s.fma(s.fmul(s.float(2.0), c), c, s.float(-1.0))
    var beta1 = s.fmul(s.fmul(d, s.float(SOTA_E_SQRT_HALF)), inv_r)
    var b2s = [s.fmul(beta1, beta1), s.fmul(s.float(9.0), s.fmul(beta1, beta1))]
    var accs = List[UInt32]()
    for lobe in range(2):
        accs.append(s.fmul(s.fmul(s.float(0.5), c), s.fsub(s.fdiv(s.float(1.0), s.fadd(s.float(1.0), b2s[lobe])), s.float(1.0))))
    var t_prev = s.float(1.0)
    var t_cur = c2
    for k in range(1, SOTA_E_SKIN_TERMS + 1):
        var kk = Float32(4 * k * k)
        var sgn = Float32(1.0) if (k % 2) == 1 else Float32(-1.0)
        var coef = s.fmul(s.float(sgn * Float32(2.0) * SOTA_E_INV_PI / (kk - Float32(1.0))), t_cur)
        for lobe in range(2):
            var filt = s.fsub(s.fdiv(s.float(1.0), s.fma(s.float(kk), b2s[lobe], s.float(1.0))), s.float(1.0))
            accs[lobe] = s.fma(coef, filt, accs[lobe])
        var t_next = s.fsub(s.fmul(s.fmul(s.float(2.0), c2), t_cur), t_prev)
        t_prev = t_cur
        t_cur = t_next
    var out = s.fma(s.float(0.25), accs[0], s.nmax(s.float(0.0), c))
    out = s.fma(s.float(0.75), accs[1], out)
    return s.nmax(s.float(0.0), out)


def emit_sota_e_skin_curvature_radius(
    mut s: AaaSpirvModule,
    gnx: UInt32, gny: UInt32, gnz: UInt32,
    dxx: UInt32, dxy: UInt32, dxz: UInt32,
    dyx: UInt32, dyy: UInt32, dyz: UInt32,
    z_lin: UInt32, tan_x: UInt32, fw: UInt32, ndotv: UInt32,
) -> UInt32:
    """Curvature radius (m) from the neighbour-ray normals: `kappa = |dn| * N.V / pixel_footprint`."""
    var ax = s.fsub(dxx, gnx); var ay = s.fsub(dxy, gny); var az = s.fsub(dxz, gnz)
    var bx = s.fsub(dyx, gnx); var by = s.fsub(dyy, gny); var bz = s.fsub(dyz, gnz)
    var la = emit_vec3_dot(s, ax, ay, az, ax, ay, az)
    var lb = emit_vec3_dot(s, bx, by, bz, bx, by, bz)
    var dn = s.fsqrt(s.nmax(la, lb))
    var px_size = s.fdiv(s.fmul(s.fmul(s.float(2.0), tan_x), z_lin), fw)
    var kappa = s.fdiv(s.fmul(dn, s.nmax(s.float(0.2), ndotv)), s.nmax(s.float(1e-6), px_size))
    return s.nclamp(s.fdiv(s.float(1.0), s.nmax(s.float(1e-3), kappa)), s.float(0.002), s.float(4.0))


def emit_sota_e_skin_direct(
    mut s: AaaSpirvModule,
    dec: SotaECharDecode,
    nx: UInt32, ny: UInt32, nz: UInt32,
    vx: UInt32, vy: UInt32, vz: UInt32,
    lx: UInt32, ly: UInt32, lz: UInt32,
    alb_r: UInt32, alb_g: UInt32, alb_b: UInt32,
    f0: UInt32, rough: UInt32, curv_radius: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Mirrors `sota_e_skin_direct_cpu` (p14 = mfp_red, p15 = lobe mix, p16/p17 = lobe scales,
    p18 = curvature scale, p20..22 = relative mfp RGB)."""
    var h = emit_vec3_normalize(s, s.fadd(vx, lx), s.fadd(vy, ly), s.fadd(vz, lz))
    var ndotv = s.nmax(s.float(1e-4), emit_vec3_dot(s, nx, ny, nz, vx, vy, vz))
    var raw_nl = emit_vec3_dot(s, nx, ny, nz, lx, ly, lz)
    var ndotl = s.nmax(s.float(0.0), raw_nl)
    var ndoth = s.nmax(s.float(0.0), emit_vec3_dot(s, nx, ny, nz, h[0], h[1], h[2]))
    var vdoth = s.nmax(s.float(0.0), emit_vec3_dot(s, vx, vy, vz, h[0], h[1], h[2]))
    var f = _e_schlick(s, f0, vdoth)
    var s0 = _e_ggx(s, ndoth, ndotv, ndotl, s.nmax(s.float(0.02), s.fmul(rough, dec.p16)))
    var s1 = _e_ggx(s, ndoth, ndotv, ndotl, s.nmin(s.float(1.0), s.fmul(rough, dec.p17)))
    var spec = s.fmul(s.fmul(s.fma(s.fsub(s1, s0), dec.p15, s0), f), ndotl)
    var inv_r = s.fdiv(s.float(1.0), s.nmax(s.float(1e-5), s.fmul(curv_radius, dec.p18)))
    var kd = s.fmul(s.fsub(s.float(1.0), f), s.float(SOTA_E_INV_PI))
    var dr = _e_burley_ring(s, raw_nl, inv_r, _e_burley_d(s, s.fmul(dec.p14, dec.p20), alb_r))
    var dg = _e_burley_ring(s, raw_nl, inv_r, _e_burley_d(s, s.fmul(dec.p14, dec.p21), alb_g))
    var db = _e_burley_ring(s, raw_nl, inv_r, _e_burley_d(s, s.fmul(dec.p14, dec.p22), alb_b))
    return (
        s.fma(s.fmul(alb_r, kd), dr, spec),
        s.fma(s.fmul(alb_g, kd), dg, spec),
        s.fma(s.fmul(alb_b, kd), db, spec),
    )


def _e_refract(
    mut s: AaaSpirvModule, ix: UInt32, iy: UInt32, iz: UInt32, nx: UInt32, ny: UInt32, nz: UInt32, eta: Float32
) -> Tuple[UInt32, UInt32, UInt32]:
    var ndi = emit_vec3_dot(s, nx, ny, nz, ix, iy, iz)
    var e = s.float(eta)
    var k = s.fsub(s.float(1.0), s.fmul(s.float(eta * eta), s.fsub(s.float(1.0), s.fmul(ndi, ndi))))
    var km = s.fma(e, ndi, s.fsqrt(s.nmax(s.float(0.0), k)))
    return (
        s.fsub(s.fmul(ix, e), s.fmul(nx, km)),
        s.fsub(s.fmul(iy, e), s.fmul(ny, km)),
        s.fsub(s.fmul(iz, e), s.fmul(nz, km)),
    )


def _e_smoothstep(mut s: AaaSpirvModule, e0: UInt32, e1: UInt32, x: UInt32) -> UInt32:
    var t = _e_sat(s, s.fdiv(s.fsub(x, e0), s.nmax(s.float(1e-6), s.fsub(e1, e0))))
    return s.fmul(s.fmul(t, t), s.fsub(s.float(3.0), s.fmul(s.float(2.0), t)))


@fieldwise_init
struct SotaEEyeVars(Copyable, Movable):
    """Function-scope variables the eye surface stage fills for the direct-lighting stage."""
    var ax: UInt32
    var ay: UInt32
    var az: UInt32
    var exx: UInt32
    var exy: UInt32
    var exz: UInt32
    var eyx: UInt32
    var eyy: UInt32
    var eyz: UInt32
    var qu: UInt32
    var qv: UInt32
    var mask: UInt32


def emit_sota_e_eye_surface(
    mut s: AaaSpirvModule,
    dec: SotaECharDecode,
    bindless_arr: Tuple[UInt32, UInt32],
    t_alb: UInt32,
    uv_u: UInt32, uv_v: UInt32,
    ddx_vec: UInt32, ddy_vec: UInt32,
    gnx: UInt32, gny: UInt32, gnz: UInt32,
    tx: UInt32, ty: UInt32, tz: UInt32,
    bx: UInt32, by: UInt32, bz: UInt32,
    nx_v: UInt32, ny_v: UInt32, nz_v: UInt32,
    vx: UInt32, vy: UInt32, vz: UInt32,
    alb_r: UInt32, alb_g: UInt32, alb_b: UInt32,
    mat_alb_r: UInt32, mat_alb_g: UInt32, mat_alb_b: UInt32,
) -> SotaEEyeVars:
    """Eye stage 1 (inside `if is_eye`): exact frame recovery (see `sota_e_eye_frame_cpu`), cornea refraction, refracted iris albedo.
    Replaces the albedo variables; the shading normal stays the cornea normal. Params:
    p14 iris radius (uv), p15 chamber depth (radii), p16 limbus width (uv), p17 uv per radius,
    p18 caustic gain, p19 pupil fraction, p20..22 sclera colour; base colour = iris colour."""
    var ev = SotaEEyeVars(
        s.fvar(gnx), s.fvar(gny), s.fvar(gnz),
        s.fvar(tx), s.fvar(ty), s.fvar(tz),
        s.fvar(bx), s.fvar(by), s.fvar(bz),
        s.fvar(s.float(0.0)), s.fvar(s.float(0.0)), s.fvar(s.float(0.0)),
    )
    s.begin_if(dec.is_eye)
    var upr = s.nmax(s.float(1e-3), dec.p17)
    var x = s.fdiv(s.fsub(uv_u, s.float(0.5)), upr)
    var y = s.fdiv(s.fsub(uv_v, s.float(0.5)), upr)
    var z = s.fsqrt(s.nmax(s.float(0.02), s.fsub(s.float(1.0), s.fma(x, x, s.fmul(y, y)))))
    var cx = s.fsqrt(s.fma(z, z, s.fmul(x, x)))
    var a = emit_vec3_normalize(
        s,
        s.fsub(s.fsub(s.fmul(gnx, z), s.fmul(tx, s.fdiv(x, cx))), s.fmul(bx, s.fdiv(s.fmul(y, z), cx))),
        s.fsub(s.fsub(s.fmul(gny, z), s.fmul(ty, s.fdiv(x, cx))), s.fmul(by, s.fdiv(s.fmul(y, z), cx))),
        s.fsub(s.fsub(s.fmul(gnz, z), s.fmul(tz, s.fdiv(x, cx))), s.fmul(bz, s.fdiv(s.fmul(y, z), cx))),
    )
    var inv_z = s.fdiv(s.float(1.0), z)
    var ex = emit_vec3_normalize(
        s,
        s.fmul(s.fma(tx, cx, s.fmul(a[0], x)), inv_z),
        s.fmul(s.fma(ty, cx, s.fmul(a[1], x)), inv_z),
        s.fmul(s.fma(tz, cx, s.fmul(a[2], x)), inv_z),
    )
    var nxt = emit_vec3_cross(s, gnx, gny, gnz, tx, ty, tz)
    var hand = s.select_f(s.fge(emit_vec3_dot(s, bx, by, bz, nxt[0], nxt[1], nxt[2]), s.float(0.0)), s.float(1.0), s.float(-1.0))
    var axe = emit_vec3_cross(s, a[0], a[1], a[2], ex[0], ex[1], ex[2])
    var ey = (s.fmul(axe[0], hand), s.fmul(axe[1], hand), s.fmul(axe[2], hand))
    s.store(ev.ax, a[0]); s.store(ev.ay, a[1]); s.store(ev.az, a[2])
    s.store(ev.exx, ex[0]); s.store(ev.exy, ex[1]); s.store(ev.exz, ex[2])
    s.store(ev.eyx, ey[0]); s.store(ev.eyy, ey[1]); s.store(ev.eyz, ey[2])

    var nx = s.load_f(nx_v); var ny = s.load_f(ny_v); var nz = s.load_f(nz_v)
    var du = s.fsub(uv_u, s.float(0.5))
    var dv = s.fsub(uv_v, s.float(0.5))
    var r_uv = s.fsqrt(s.fma(du, du, s.fmul(dv, dv)))
    var iris_r = s.nmax(s.float(1e-4), dec.p14)
    var rr = s.fdiv(r_uv, iris_r)
    var hgt = s.fmul(dec.p15, s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fmul(rr, rr))))
    var rd = _e_refract(s, s.fneg(vx), s.fneg(vy), s.fneg(vz), nx, ny, nz, Float32(1.0) / SOTA_E_CORNEA_IOR)
    var along = s.nmax(s.float(0.05), s.fneg(emit_vec3_dot(s, rd[0], rd[1], rd[2], a[0], a[1], a[2])))
    var tl = s.fmul(s.fdiv(hgt, along), upr)
    var ru = s.fma(emit_vec3_dot(s, rd[0], rd[1], rd[2], ex[0], ex[1], ex[2]), tl, uv_u)
    var rv = s.fma(emit_vec3_dot(s, rd[0], rd[1], rd[2], ey[0], ey[1], ey[2]), tl, uv_v)
    var qu = s.fsub(ru, s.float(0.5))
    var qv = s.fsub(rv, s.float(0.5))
    var rq = s.fsqrt(s.fma(qu, qu, s.fmul(qv, qv)))
    var mask = s.fsub(s.float(1.0), _e_smoothstep(s, s.fsub(iris_r, dec.p16), iris_r, rq))
    var ld = s.fdiv(s.fsub(rq, s.fsub(iris_r, s.fmul(s.float(0.5), dec.p16))), s.nmax(s.float(1e-4), dec.p16))
    var limbus = s.fsub(s.float(1.0), s.fmul(s.float(0.65), s.exp2(s.fmul(s.fmul(ld, ld), s.float(-2.0 * 1.44269504)))))
    var pr = s.fmul(dec.p19, iris_r)
    var pupil = s.fsub(s.float(1.0), _e_smoothstep(s, s.fmul(pr, s.float(0.92)), s.fmul(pr, s.float(1.08)), rq))
    s.store(ev.qu, qu); s.store(ev.qv, qv); s.store(ev.mask, mask)

    # Iris albedo: bound albedo texture sampled at the refracted UV, else the procedural stroma.
    var ir_v = s.fvar(mat_alb_r); var ig_v = s.fvar(mat_alb_g); var ib_v = s.fvar(mat_alb_b)
    s.begin_if(s.ult(t_alb, s.uint(0xFF)))
    var tex = s.sample_bindless_grad(bindless_arr, t_alb, s.vec2(ru, rv), ddx_vec, ddy_vec)
    s.store(ir_v, s.fmul(mat_alb_r, s.extract_f(tex, 0)))
    s.store(ig_v, s.fmul(mat_alb_g, s.extract_f(tex, 1)))
    s.store(ib_v, s.fmul(mat_alb_b, s.extract_f(tex, 2)))
    s.begin_else()
    var rn = s.fdiv(rq, iris_r)
    var ang = s.fatan2(qv, qu)
    var fib = s.fma(
        s.fmul(s.fsin(s.fma(ang, s.float(41.0), s.fmul(rn, s.float(9.0)))), s.fsin(s.fsub(s.fmul(ang, s.float(13.0)), s.fmul(rn, s.float(5.0))))),
        s.float(0.5), s.float(0.5),
    )
    var dcol = s.fsub(rn, s.float(0.55))
    var collar = s.fsub(s.float(1.0), s.fmul(s.float(0.35), s.exp2(s.fmul(s.fmul(dcol, dcol), s.float(-60.0 * 1.44269504)))))
    var g = s.fmul(s.fma(fib, s.float(0.5), s.float(0.72)), collar)
    s.store(ir_v, s.fmul(mat_alb_r, g))
    s.store(ig_v, s.fmul(mat_alb_g, g))
    s.store(ib_v, s.fmul(mat_alb_b, g))
    s.end_if()
    var pupil_k = s.fsub(s.float(1.0), s.fmul(pupil, s.float(0.985)))
    var iris_k = s.fmul(pupil_k, limbus)
    var sclera_k = limbus
    var ir = s.fmul(s.load_f(ir_v), iris_k)
    var ig = s.fmul(s.load_f(ig_v), iris_k)
    var ib = s.fmul(s.load_f(ib_v), iris_k)
    var sr = s.fmul(dec.p20, sclera_k)
    var sg = s.fmul(dec.p21, sclera_k)
    var sb = s.fmul(dec.p22, sclera_k)
    s.store(alb_r, s.fma(s.fsub(ir, sr), mask, sr))
    s.store(alb_g, s.fma(s.fsub(ig, sg), mask, sg))
    s.store(alb_b, s.fma(s.fsub(ib, sb), mask, sb))
    s.end_if()
    return ev^


def emit_sota_e_eye_direct(
    mut s: AaaSpirvModule,
    dec: SotaECharDecode,
    ev: SotaEEyeVars,
    nx: UInt32, ny: UInt32, nz: UInt32,
    vx: UInt32, vy: UInt32, vz: UInt32,
    lx: UInt32, ly: UInt32, lz: UInt32,
    alb_r: UInt32, alb_g: UInt32, alb_b: UInt32,
    f0: UInt32, rough: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Mirrors `sota_e_eye_direct_cpu` (call inside `if is_eye`)."""
    var ax = s.load_f(ev.ax); var ay = s.load_f(ev.ay); var az = s.load_f(ev.az)
    var h = emit_vec3_normalize(s, s.fadd(vx, lx), s.fadd(vy, ly), s.fadd(vz, lz))
    var ndotv = s.nmax(s.float(1e-4), emit_vec3_dot(s, nx, ny, nz, vx, vy, vz))
    var raw_nl = emit_vec3_dot(s, nx, ny, nz, lx, ly, lz)
    var ndotl = s.nmax(s.float(0.0), raw_nl)
    var ndoth = s.nmax(s.float(0.0), emit_vec3_dot(s, nx, ny, nz, h[0], h[1], h[2]))
    var vdoth = s.nmax(s.float(0.0), emit_vec3_dot(s, vx, vy, vz, h[0], h[1], h[2]))
    var spec = s.fmul(s.fmul(_e_ggx(s, ndoth, ndotv, ndotl, rough), _e_schlick(s, f0, vdoth)), ndotl)
    var lr = _e_refract(s, s.fneg(lx), s.fneg(ly), s.fneg(lz), nx, ny, nz, Float32(1.0) / SOTA_E_CORNEA_IOR)
    var lit = s.select_f(s.fgt(raw_nl, s.float(0.0)), s.float(1.0), s.float(0.0))
    var iris_nl = s.fmul(_e_sat(s, s.fneg(emit_vec3_dot(s, lr[0], lr[1], lr[2], ax, ay, az))), lit)
    var lpx = emit_vec3_dot(s, lx, ly, lz, s.load_f(ev.exx), s.load_f(ev.exy), s.load_f(ev.exz))
    var lpy = emit_vec3_dot(s, lx, ly, lz, s.load_f(ev.eyx), s.load_f(ev.eyy), s.load_f(ev.eyz))
    var lpl = s.fadd(s.fsqrt(s.fma(lpx, lpx, s.fmul(lpy, lpy))), s.float(1e-4))
    var qu = s.load_f(ev.qu); var qv = s.load_f(ev.qv)
    var ql = s.fadd(s.fsqrt(s.fma(qu, qu, s.fmul(qv, qv))), s.float(1e-4))
    var opp = _e_sat(s, s.fdiv(s.fneg(s.fma(qu, lpx, s.fmul(qv, lpy))), s.fmul(ql, lpl)))
    var graze = s.fsub(s.float(1.0), _e_sat(s, emit_vec3_dot(s, lx, ly, lz, ax, ay, az)))
    var caustic = s.fmul(
        s.fmul(s.fmul(dec.p18, opp), s.fmul(graze, _e_sat(s, s.fdiv(ql, s.nmax(s.float(1e-4), dec.p14))))),
        _e_sat(s, s.fmul(s.fadd(emit_vec3_dot(s, lx, ly, lz, ax, ay, az), s.float(0.15)), s.float(4.0))),
    )
    var t_in = s.fmul(s.fsub(s.float(1.0), _e_schlick(s, f0, ndotl)), s.fsub(s.float(1.0), _e_schlick(s, f0, ndotv)))
    var iris_k = s.fmul(s.fmul(s.fadd(iris_nl, caustic), t_in), s.float(SOTA_E_INV_PI))
    var sclera_k = s.fmul(_e_sat(s, s.fdiv(s.fadd(raw_nl, s.float(0.2)), s.float(1.2))), s.float(SOTA_E_INV_PI))
    var k = s.fma(s.fsub(iris_k, sclera_k), s.load_f(ev.mask), sclera_k)
    return (s.fma(alb_r, k, spec), s.fma(alb_g, k, spec), s.fma(alb_b, k, spec))


def _e_hair_fresnel(mut s: AaaSpirvModule, ior: UInt32, cos_t: UInt32) -> UInt32:
    var r = s.fdiv(s.fsub(s.float(1.0), ior), s.fadd(s.float(1.0), ior))
    return _e_schlick(s, s.fmul(r, r), cos_t)


def _e_gauss(mut s: AaaSpirvModule, beta: UInt32, x: UInt32) -> UInt32:
    var q = s.fdiv(s.fmul(x, x), s.fmul(beta, beta))
    return s.fdiv(s.exp2(s.fmul(q, s.float(-0.5 * 1.44269504))), s.fmul(beta, s.float(2.50662827)))


def _e_powf(mut s: AaaSpirvModule, b: UInt32, e: UInt32) -> UInt32:
    return s.exp2(s.fmul(e, s.log2(s.nmax(s.float(1e-20), b))))


def emit_sota_e_hair_direct(
    mut s: AaaSpirvModule,
    dec: SotaECharDecode,
    tx_in: UInt32, ty_in: UInt32, tz_in: UInt32,
    vx: UInt32, vy: UInt32, vz: UInt32,
    lx: UInt32, ly: UInt32, lz: UInt32,
    col_r: UInt32, col_g: UInt32, col_b: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Mirrors `sota_e_hair_direct_cpu`. p14 = shift, p15 = beta, p18 = IOR, p20..22 = `d_b * A_b`."""
    var t = emit_vec3_normalize(s, tx_in, ty_in, tz_in)
    var sin_i = s.nclamp(emit_vec3_dot(s, lx, ly, lz, t[0], t[1], t[2]), s.float(-1.0), s.float(1.0))
    var sin_o = s.nclamp(emit_vec3_dot(s, vx, vy, vz, t[0], t[1], t[2]), s.float(-1.0), s.float(1.0))
    var cos_i = s.fsqrt(_e_sat(s, s.fsub(s.float(1.0), s.fmul(sin_i, sin_i))))
    var cos_o = s.fsqrt(_e_sat(s, s.fsub(s.float(1.0), s.fmul(sin_o, sin_o))))
    var cos_td = s.nmax(
        s.float(0.05),
        s.fsqrt(_e_sat(s, s.fma(s.float(0.5), s.fma(cos_i, cos_o, s.fmul(sin_i, sin_o)), s.float(0.5)))),
    )
    var lpx = s.fsub(lx, s.fmul(t[0], sin_i)); var lpy = s.fsub(ly, s.fmul(t[1], sin_i)); var lpz = s.fsub(lz, s.fmul(t[2], sin_i))
    var vpx = s.fsub(vx, s.fmul(t[0], sin_o)); var vpy = s.fsub(vy, s.fmul(t[1], sin_o)); var vpz = s.fsub(vz, s.fmul(t[2], sin_o))
    var l2 = emit_vec3_dot(s, lpx, lpy, lpz, lpx, lpy, lpz)
    var v2 = emit_vec3_dot(s, vpx, vpy, vpz, vpx, vpy, vpz)
    var cos_phi = s.nclamp(
        s.fdiv(emit_vec3_dot(s, lpx, lpy, lpz, vpx, vpy, vpz), s.fsqrt(s.nmax(s.float(1e-8), s.fmul(l2, v2)))),
        s.float(-1.0), s.float(1.0),
    )
    var xs = s.fadd(sin_i, sin_o)
    var a = dec.p14
    var b = s.nmax(s.float(0.02), dec.p15)
    var ior = s.nmax(s.float(1.01), dec.p18)
    var m_r = _e_gauss(s, b, s.fsub(xs, a))
    var m_tt = _e_gauss(s, s.fmul(b, s.float(0.5)), s.fma(a, s.float(0.5), xs))
    var m_trt = _e_gauss(s, s.fmul(b, s.float(2.0)), s.fma(a, s.float(1.5), xs))
    var vdotl_half = s.fsqrt(_e_sat(s, s.fma(s.float(0.5), emit_vec3_dot(s, vx, vy, vz, lx, ly, lz), s.float(0.5))))
    var cphi_h = s.fsqrt(_e_sat(s, s.fma(s.float(0.5), cos_phi, s.float(0.5))))
    var f_r = s.fmul(s.fmul(m_r, s.fmul(s.float(0.25), cphi_h)), _e_hair_fresnel(s, ior, vdotl_half))
    var aa = s.fdiv(s.float(1.0), s.fma(s.float(0.36), cos_td, s.fdiv(s.float(1.19), cos_td)))
    var hh = s.fmul(s.fma(aa, s.fsub(s.float(0.6), s.fmul(s.float(0.8), cos_phi)), s.float(1.0)), cphi_h)
    var om_ftt = s.fsub(s.float(1.0), _e_hair_fresnel(s, ior, s.fmul(cos_td, s.fsqrt(_e_sat(s, s.fsub(s.float(1.0), s.fmul(hh, hh)))))))
    var e_tt = s.fdiv(s.fsqrt(_e_sat(s, s.fsub(s.float(1.0), s.fmul(s.fmul(hh, hh), s.fmul(aa, aa))))), s.fmul(s.float(2.0), cos_td))
    var n_tt = s.exp2(s.fmul(s.fma(s.float(-3.65), cos_phi, s.float(-3.98)), s.float(1.44269504)))
    var f_trt = _e_hair_fresnel(s, ior, s.fmul(cos_td, s.float(0.5)))
    var om_trt = s.fsub(s.float(1.0), f_trt)
    var e_trt = s.fdiv(s.float(0.8), cos_td)
    var n_trt = s.exp2(s.fmul(s.fma(s.float(17.0), cos_phi, s.float(-16.78)), s.float(1.44269504)))
    var g_back = s.fdiv(s.fmul(s.float(2.0), _e_gauss(s, s.fmul(b, s.float(2.0)), s.fsub(xs, a))), s.fmul(s.float(SOTA_E_PI), s.fmul(cos_td, cos_td)))
    var k_tt = s.fmul(s.fmul(m_tt, n_tt), s.fmul(om_ftt, om_ftt))
    var k_trt = s.fmul(s.fmul(m_trt, n_trt), s.fmul(s.fmul(om_trt, om_trt), f_trt))
    var cols = [col_r, col_g, col_b]
    var amps = [dec.p20, dec.p21, dec.p22]
    var outs = List[UInt32]()
    for c in range(3):
        var col = s.nmax(s.float(1e-4), cols[c])
        var tt = s.fmul(k_tt, _e_powf(s, col, e_tt))
        var trt = s.fmul(k_trt, _e_powf(s, col, e_trt))
        var val = s.fmul(s.fadd(s.fadd(f_r, tt), s.fma(amps[c], g_back, trt)), cos_i)
        outs.append(val)
    return (outs[0], outs[1], outs[2])


@fieldwise_init
struct SotaEDirectInputs(Copyable, Movable):
    var nx: UInt32
    var ny: UInt32
    var nz: UInt32
    var vx: UInt32
    var vy: UInt32
    var vz: UInt32
    var lx: UInt32
    var ly: UInt32
    var lz: UInt32
    var alb_r: UInt32
    var alb_g: UInt32
    var alb_b: UInt32
    var f0: UInt32
    var rough: UInt32


def emit_sota_e_direct_override(
    mut s: AaaSpirvModule,
    dec: SotaECharDecode,
    ev: SotaEEyeVars,
    inp: SotaEDirectInputs,
    hair_tx: UInt32, hair_ty: UInt32, hair_tz: UInt32,
    curv_radius: UInt32,
    std_r: UInt32, std_g: UInt32, std_b: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Replaces the standard multi-lobe direct sun response for skin / eye / hair pixels."""
    var out_r = s.fvar(std_r)
    var out_g = s.fvar(std_g)
    var out_b = s.fvar(std_b)
    s.begin_if(dec.is_skin)
    var sk = emit_sota_e_skin_direct(
        s, dec, inp.nx, inp.ny, inp.nz, inp.vx, inp.vy, inp.vz, inp.lx, inp.ly, inp.lz,
        inp.alb_r, inp.alb_g, inp.alb_b, inp.f0, inp.rough, curv_radius,
    )
    s.store(out_r, sk[0]); s.store(out_g, sk[1]); s.store(out_b, sk[2])
    s.end_if()
    s.begin_if(dec.is_eye)
    var ey = emit_sota_e_eye_direct(
        s, dec, ev, inp.nx, inp.ny, inp.nz, inp.vx, inp.vy, inp.vz, inp.lx, inp.ly, inp.lz,
        inp.alb_r, inp.alb_g, inp.alb_b, inp.f0, inp.rough,
    )
    s.store(out_r, ey[0]); s.store(out_g, ey[1]); s.store(out_b, ey[2])
    s.end_if()
    s.begin_if(dec.is_hair)
    var hr = emit_sota_e_hair_direct(
        s, dec, hair_tx, hair_ty, hair_tz, inp.vx, inp.vy, inp.vz, inp.lx, inp.ly, inp.lz,
        inp.alb_r, inp.alb_g, inp.alb_b,
    )
    s.store(out_r, hr[0]); s.store(out_g, hr[1]); s.store(out_b, hr[2])
    s.end_if()
    return (s.load_f(out_r), s.load_f(out_g), s.load_f(out_b))
