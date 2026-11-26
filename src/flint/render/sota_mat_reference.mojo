"""Area I (`sota:mat`) CPU references: ground truth for the layered material model of `sota_mat_core.mojo`.

  1. `sota_mat_rw_albedo`: Heitz et al. 2016 stochastic random walk on the Smith (uniform-height) microsurface of
     an anisotropic GGX conductor. It is the true multiple-scattering albedo that Kulla-Conty approximates
     (F = 1 conserves energy exactly; the single-scattering order matches the LUT).
  2. `sota_mat_thin_film_spectral`: exact Airy reflectance of a thin film (s/p averaged, real indices) integrated
     against the Wyman-Sloan-Shirley 2013 multi-lobe fit of the CIE 1931 observer, 380..780 nm.
  3. `sota_mat_albedo_mc`: Monte-Carlo directional albedo of the layered BSDF (one-sample mixture sampling).
  4. `SotaMatScene.render`: a CPU path tracer of a row of spheres under an analytic sky (or a white furnace) and a
     directional sun, with the renderer's exact camera model, used for image diffs against the GPU.
  5. Twins of the legacy (v5) BSDF for the audit: `sota_mat_legacy_direct_cpu` (port of
     `aaa_bsdf.emit_direct_multilobe_bsdf`) and the legacy split-sum ambient weights.
"""
from std.collections import List
from std.math import abs, cos, exp, floor, log, max, min, sin, sqrt, tan
from flint.core.math import Vec3
from flint.render.aaa_ibl import SkyDomeParams, bake_split_sum_brdf_lut_rg16f
from flint.render.material_graph import f16_bits_to_f32
from flint.render.camera_basis import view_right
from flint.render.sota_mat_core import (
    SOTA_MAT_ALPHA_MIN,
    SOTA_MAT_INV_PI,
    SOTA_MAT_MU_MIN,
    SOTA_MAT_PI,
    SotaMatFrame,
    SotaMatLut,
    SotaMatParams,
    sota_mat_ambient_cpu,
    sota_mat_eval_cpu,
    sota_mat_frame_cpu,
    sota_mat_lambda_aniso,
    sota_mat_lobes_cpu,
    sota_mat_sample_vndf,
    sota_mat_view_phi,
    sota_mat_d_aniso,
)


# ---------------------------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------------------------
struct SotaRng(Copyable, Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        _ = self.next_u32()

    def next_u32(mut self) -> UInt32:
        var old = self.state
        self.state = old * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var xs = UInt32(((old >> 18) ^ old) >> 27)
        var rot = UInt32(old >> 59)
        return (xs >> rot) | (xs << ((~rot + 1) & 31))

    def next(mut self) -> Float32:
        return Float32(self.next_u32() >> 8) * Float32(1.0 / 16777216.0)


@always_inline
def _vm(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z)


@always_inline
def _lum(c: Vec3) -> Float32:
    return Float32(0.2126) * c.x + Float32(0.7152) * c.y + Float32(0.0722) * c.z


@always_inline
def _pow5(x: Float32) -> Float32:
    var x2 = x * x
    return x2 * x2 * x


# ---------------------------------------------------------------------------------------------
# 1. Heitz 2016 random walk (conductor, uniform height distribution)
# ---------------------------------------------------------------------------------------------
@always_inline
def _rw_lambda(w: Vec3, at: Float32, ab: Float32) -> Float32:
    if w.z > Float32(0.9999):
        return Float32(0.0)
    if w.z < Float32(-0.9999):
        return Float32(-1.0)
    var q = sqrt(Float32(1.0) + (at * at * w.x * w.x + ab * ab * w.y * w.y) / max(Float32(1e-12), w.z * w.z))
    if w.z > Float32(0.0):
        return Float32(0.5) * (q - Float32(1.0))
    return Float32(-0.5) * (q + Float32(1.0))


@always_inline
def _c1(h: Float32) -> Float32:
    return max(Float32(0.0), min(Float32(1.0), Float32(0.5) * (h + Float32(1.0))))


@always_inline
def _inv_c1(u: Float32) -> Float32:
    return max(Float32(-1.0), min(Float32(1.0), Float32(2.0) * u - Float32(1.0)))


comptime _RW_ESCAPE: Float32 = 1e30


def _rw_sample_height(wr: Vec3, hr: Float32, u: Float32, at: Float32, ab: Float32) -> Float32:
    if wr.z > Float32(0.9999):
        return _RW_ESCAPE
    if wr.z < Float32(-0.9999):
        return _inv_c1(u * _c1(hr))
    if abs(wr.z) < Float32(0.0001):
        return hr
    var lam = _rw_lambda(wr, at, ab)
    var g1 = Float32(0.0)
    if wr.z > Float32(0.0):
        var c = _c1(hr)
        g1 = exp(lam * log(max(Float32(1e-30), c)))
    if u > Float32(1.0) - g1:
        return _RW_ESCAPE
    var denom = exp(log(max(Float32(1e-30), Float32(1.0) - u)) / lam)
    return _inv_c1(_c1(hr) / max(Float32(1e-30), denom))


struct SotaRwResult(Copyable, Movable):
    var total: Vec3      # all scattering orders
    var single: Vec3     # first order only

    def __init__(out self):
        self.total = Vec3(0.0, 0.0, 0.0)
        self.single = Vec3(0.0, 0.0, 0.0)


def sota_mat_rw_albedo(mu: Float32, phi: Float32, at: Float32, ab: Float32, f0: Vec3, f90: Float32, paths: Int, seed: UInt64) -> SotaRwResult:
    """Directional albedo of a Schlick-Fresnel anisotropic GGX conductor by Heitz-2016 random walks
    (`paths` walks, at most 64 bounces each)."""
    var rng = SotaRng(seed)
    var st = sqrt(max(Float32(0.0), Float32(1.0) - mu * mu))
    var wi = Vec3(st * cos(phi), st * sin(phi), mu)
    var acc = Vec3(0.0, 0.0, 0.0)
    var acc1 = Vec3(0.0, 0.0, 0.0)
    for _ in range(paths):
        var wr = wi * Float32(-1.0)
        var hr = Float32(1.0) + _inv_c1(Float32(0.999))
        var e = Vec3(1.0, 1.0, 1.0)
        var order = 0
        var escaped = False
        for _ in range(64):
            hr = _rw_sample_height(wr, hr, rng.next(), at, ab)
            if hr == _RW_ESCAPE:
                escaped = True
                break
            order += 1
            var wo = wr * Float32(-1.0)  # direction the walk came from, pointing away
            var m = sota_mat_sample_vndf(wo.x, wo.y, wo.z, at, ab, rng.next(), rng.next())
            var c = max(Float32(0.0), wo.dot(m))
            var fc = _pow5(Float32(1.0) - c)
            e = _vm(e, f0 + (Vec3(f90, f90, f90) - f0) * fc)
            wr = m * (Float32(2.0) * c) - wo
        if escaped and order > 0:
            acc = acc + e
            if order == 1:
                acc1 = acc1 + e
    var r = SotaRwResult()
    var inv = Float32(1.0) / Float32(paths)
    r.total = acc * inv
    r.single = acc1 * inv
    return r^


# ---------------------------------------------------------------------------------------------
# 2. Spectral thin-film reference
# ---------------------------------------------------------------------------------------------
@always_inline
def _g(x: Float32, mu: Float32, s1: Float32, s2: Float32) -> Float32:
    var t = (x - mu) / (s1 if x < mu else s2)
    return exp(Float32(-0.5) * t * t)


def _cie_xyz(lam: Float32) -> Vec3:
    var x = Float32(1.056) * _g(lam, 599.8, 37.9, 31.0) + Float32(0.362) * _g(lam, 442.0, 16.0, 26.7) - Float32(0.065) * _g(lam, 501.1, 20.4, 26.2)
    var y = Float32(0.821) * _g(lam, 568.8, 46.9, 40.5) + Float32(0.286) * _g(lam, 530.9, 16.3, 31.1)
    var z = Float32(1.217) * _g(lam, 437.0, 11.8, 36.0) + Float32(0.681) * _g(lam, 459.0, 26.0, 13.8)
    return Vec3(x, y, z)


@always_inline
def _airy_r(r12: Float32, r23: Float32, cd: Float32) -> Float32:
    var num = r12 * r12 + r23 * r23 + Float32(2.0) * r12 * r23 * cd
    var den = Float32(1.0) + r12 * r12 * r23 * r23 + Float32(2.0) * r12 * r23 * cd
    return num / den


def sota_mat_thin_film_spectral(cos1: Float32, d_nm: Float32, n2: Float32, n3: Float32) -> Vec3:
    """Linear sRGB reflectance of air / film (n2, d_nm) / dielectric substrate (n3), unpolarised light."""
    var s2 = Float32(1.0) - cos1 * cos1
    var c2 = sqrt(max(Float32(0.0), Float32(1.0) - s2 / (n2 * n2)))
    var c3 = sqrt(max(Float32(0.0), Float32(1.0) - s2 / (n3 * n3)))
    var rs12 = (cos1 - n2 * c2) / (cos1 + n2 * c2)
    var rp12 = (n2 * cos1 - c2) / (n2 * cos1 + c2)
    var rs23 = (n2 * c2 - n3 * c3) / (n2 * c2 + n3 * c3)
    var rp23 = (n3 * c2 - n2 * c3) / (n3 * c2 + n2 * c3)
    var acc = Vec3(0.0, 0.0, 0.0)
    var norm = Vec3(0.0, 0.0, 0.0)
    var lam = Float32(380.0)
    while lam <= Float32(780.0):
        var delta = Float32(4.0) * SOTA_MAT_PI * n2 * d_nm * c2 / lam
        var cd = cos(delta)
        var rr = Float32(0.5) * (_airy_r(rs12, rs23, cd) + _airy_r(rp12, rp23, cd))
        var cmf = _cie_xyz(lam)
        acc = acc + cmf * rr
        norm = norm + cmf
        lam += Float32(1.0)
    var x = acc.x / norm.x
    var y = acc.y / norm.y
    var z = acc.z / norm.z
    return Vec3(
        Float32(3.2404542) * x - Float32(1.5371385) * y - Float32(0.4985314) * z,
        Float32(-0.9692660) * x + Float32(1.8760108) * y + Float32(0.0415560) * z,
        Float32(0.0556434) * x - Float32(0.2040259) * y + Float32(1.0572252) * z,
    )


# ---------------------------------------------------------------------------------------------
# 3. Sampling the layered BSDF (one-sample mixture) and its directional albedo
# ---------------------------------------------------------------------------------------------
struct SotaMatSample(Copyable, Movable):
    var dir: Vec3
    var weight: Vec3   # f |cos| / pdf_mix
    var ok: Bool

    def __init__(out self):
        self.dir = Vec3(0.0, 0.0, 1.0)
        self.weight = Vec3(0.0, 0.0, 0.0)
        self.ok = False


def _vndf_pdf(n: Vec3, fr: SotaMatFrame, v: Vec3, l: Vec3, at: Float32, ab: Float32) -> Float32:
    """Density over the sphere of `l = reflect(-v, m)` with `m` drawn from the visible normals of `v`."""
    var h = v + l
    var hl = h.length()
    if hl < Float32(1e-7):
        return Float32(0.0)
    h = h * (Float32(1.0) / hl)
    var hn = n.dot(h)
    if hn <= Float32(0.0):
        return Float32(0.0)
    var vn = max(SOTA_MAT_MU_MIN, n.dot(v))
    var lam = sota_mat_lambda_aniso(at, ab, v.dot(fr.t), v.dot(fr.b), vn)
    var d = sota_mat_d_aniso(at, ab, h.dot(fr.t), h.dot(fr.b), hn)
    return d / ((Float32(1.0) + lam) * Float32(4.0) * vn)


def _vndf_dir(n: Vec3, fr: SotaMatFrame, v: Vec3, at: Float32, ab: Float32, u1: Float32, u2: Float32) -> Vec3:
    var m_l = sota_mat_sample_vndf(v.dot(fr.t), v.dot(fr.b), max(SOTA_MAT_MU_MIN, n.dot(v)), at, ab, u1, u2)
    var m = fr.t * m_l.x + fr.b * m_l.y + n * m_l.z
    return m * (Float32(2.0) * v.dot(m)) - v


def sota_mat_sample_cpu(p: SotaMatParams, lut: SotaMatLut, n: Vec3, t_in: Vec3, b_in: Vec3, v: Vec3, mut rng: SotaRng) -> SotaMatSample:
    var ndotv = max(SOTA_MAT_MU_MIN, n.dot(v))
    var fr = sota_mat_frame_cpu(n, t_in, b_in, p.aniso_rot)
    var phi_v = sota_mat_view_phi(v.dot(fr.t), v.dot(fr.b))
    var o = sota_mat_lobes_cpu(p, lut, ndotv, phi_v)
    var amb = sota_mat_ambient_cpu(p, lut, ndotv, phi_v)
    var ac = max(SOTA_MAT_ALPHA_MIN, p.clearcoat_roughness * p.clearcoat_roughness)
    var w_c = amb.w_cc
    var w_s = _lum(amb.w_spec)
    var w_d = _lum(amb.w_diff) + Float32(0.05)
    var w_t = _lum(amb.w_trans)
    var tot = w_c + w_s + w_d + w_t
    var pc = w_c / tot
    var ps = w_s / tot
    var pd = w_d / tot
    var pt = w_t / tot
    var u = rng.next()
    var u1 = rng.next()
    var u2 = rng.next()
    var l: Vec3
    if u < pc:
        l = _vndf_dir(n, fr, v, ac, ac, u1, u2)
    elif u < pc + ps:
        l = _vndf_dir(n, fr, v, o.at, o.ab, u1, u2)
    elif u < pc + ps + pd:
        var rr = sqrt(u1)
        var ph = Float32(6.28318530718) * u2
        var lz = sqrt(max(Float32(0.0), Float32(1.0) - u1))
        l = fr.t * (rr * cos(ph)) + fr.b * (rr * sin(ph)) + n * lz
    else:
        var lr = _vndf_dir(n, fr, v, o.at, o.ab, u1, u2)
        l = lr - n * (Float32(2.0) * n.dot(lr))
    var nl = n.dot(l)
    var lm = l - n * (Float32(2.0) * nl)
    var pdf = pd * max(Float32(0.0), nl) * SOTA_MAT_INV_PI
    pdf += pc * _vndf_pdf(n, fr, v, l, ac, ac)
    pdf += ps * _vndf_pdf(n, fr, v, l, o.at, o.ab)
    pdf += pt * _vndf_pdf(n, fr, v, lm, o.at, o.ab)
    var r = SotaMatSample()
    r.dir = l
    if pdf <= Float32(1e-12):
        return r^
    var f = sota_mat_eval_cpu(p, lut, n, t_in, b_in, v, l)
    r.weight = f * (Float32(1.0) / pdf)
    r.ok = True
    return r^


def sota_mat_albedo_mc(p: SotaMatParams, lut: SotaMatLut, mu: Float32, samples: Int, seed: UInt64) -> Vec3:
    """Reflected + transmitted directional albedo at view cosine `mu` (normal +z, tangent +x)."""
    var rng = SotaRng(seed)
    var n = Vec3(0.0, 0.0, 1.0)
    var t = Vec3(1.0, 0.0, 0.0)
    var b = Vec3(0.0, 1.0, 0.0)
    var v = Vec3(sqrt(max(Float32(0.0), Float32(1.0) - mu * mu)) * Float32(0.8), sqrt(max(Float32(0.0), Float32(1.0) - mu * mu)) * Float32(0.6), mu)
    var acc = Vec3(0.0, 0.0, 0.0)
    for _ in range(samples):
        var smp = sota_mat_sample_cpu(p, lut, n, t, b, v, rng)
        if smp.ok:
            acc = acc + smp.weight
    return acc * (Float32(1.0) / Float32(samples))


# ---------------------------------------------------------------------------------------------
# 4. Legacy (v5) BSDF twins for the audit
# ---------------------------------------------------------------------------------------------
struct SotaLegacyLut(Copyable, Movable):
    """The renderer's split-sum RG16F LUT (64 x 64, perceptual roughness rows) with GPU-style bilinear sampling."""
    var a: List[Float32]
    var b: List[Float32]

    def __init__(out self):
        var bytes = bake_split_sum_brdf_lut_rg16f(64, 128)
        self.a = List[Float32](length=64 * 64, fill=Float32(0.0))
        self.b = List[Float32](length=64 * 64, fill=Float32(0.0))
        for i in range(64 * 64):
            var ha = UInt16(bytes[i * 4]) | (UInt16(bytes[i * 4 + 1]) << 8)
            var hb = UInt16(bytes[i * 4 + 2]) | (UInt16(bytes[i * 4 + 3]) << 8)
            self.a[i] = f16_bits_to_f32(ha)
            self.b[i] = f16_bits_to_f32(hb)

    def sample(self, ndotv: Float32, rough: Float32) -> Tuple[Float32, Float32]:
        var u = max(Float32(0.01), min(Float32(0.99), ndotv)) * Float32(64.0) - Float32(0.5)
        var v = max(Float32(0.01), min(Float32(0.99), rough)) * Float32(64.0) - Float32(0.5)
        var x0 = max(0, min(63, Int(floor(u))))
        var y0 = max(0, min(63, Int(floor(v))))
        var x1 = min(63, x0 + 1)
        var y1 = min(63, y0 + 1)
        var tx = max(Float32(0.0), min(Float32(1.0), u - floor(u)))
        var ty = max(Float32(0.0), min(Float32(1.0), v - floor(v)))
        var ra = (self.a[y0 * 64 + x0] * (Float32(1.0) - tx) + self.a[y0 * 64 + x1] * tx) * (Float32(1.0) - ty) + (
            self.a[y1 * 64 + x0] * (Float32(1.0) - tx) + self.a[y1 * 64 + x1] * tx
        ) * ty
        var rb = (self.b[y0 * 64 + x0] * (Float32(1.0) - tx) + self.b[y0 * 64 + x1] * tx) * (Float32(1.0) - ty) + (
            self.b[y1 * 64 + x0] * (Float32(1.0) - tx) + self.b[y1 * 64 + x1] * tx
        ) * ty
        return (ra, rb)


def sota_mat_legacy_direct_cpu(p: SotaMatParams, llut: SotaLegacyLut, n: Vec3, v: Vec3, l: Vec3) -> Vec3:
    """CPU port of `emit_direct_multilobe_bsdf` with the deferred pass's F0 / F90 / LUT inputs (no iridescence)."""
    var m = p.metallic
    var diel = Float32(0.04) * p.specular
    var f0 = Vec3(diel, diel, diel) * (Float32(1.0) - m) + p.albedo * m
    var f90 = p.specular * (Float32(1.0) - m) + m
    var ndotv_lut = max(Float32(1e-3), n.dot(v))
    var lab = llut.sample(max(Float32(0.01), min(Float32(0.99), ndotv_lut)), max(Float32(0.01), min(Float32(0.99), p.roughness)))
    var h = (v + l).normalized()
    var ndotv = max(Float32(1e-4), n.dot(v))
    var raw_nl = n.dot(l)
    var ndotl = max(Float32(0.0), raw_nl)
    var ndoth = max(Float32(0.0), n.dot(h))
    var vdoth = max(Float32(0.0), v.dot(h))
    var alpha = max(Float32(0.002), p.roughness * p.roughness)
    var a2 = alpha * alpha
    var dd = ndoth * ndoth * (a2 - Float32(1.0)) + Float32(1.0)
    var d = a2 / max(Float32(1e-6), SOTA_MAT_PI * dd * dd)
    var gv = ndotl * sqrt(ndotv * ndotv * (Float32(1.0) - a2) + a2)
    var gl = ndotv * sqrt(ndotl * ndotl * (Float32(1.0) - a2) + a2)
    var vis = Float32(0.5) / max(Float32(1e-5), gv + gl)
    var omv5 = _pow5(Float32(1.0) - vdoth)
    var f = f0 + (Vec3(f90, f90, f90) - f0) * omv5
    var e_ss = max(Float32(0.05), lab[0] + lab[1])
    var boost = Float32(1.0) / e_ss - Float32(1.0)
    var spec = Vec3(f.x * (Float32(1.0) + f0.x * boost), f.y * (Float32(1.0) + f0.y * boost), f.z * (Float32(1.0) + f0.z * boost)) * (d * vis)
    var kd = Vec3(Float32(1.0) - f.x, Float32(1.0) - f.y, Float32(1.0) - f.z) * (Float32(1.0) - m)
    var diff = _vm(kd, p.albedo) * SOTA_MAT_INV_PI
    var inv_sr = Float32(1.0) / max(Float32(0.05), p.sheen_roughness)
    var sin2 = max(Float32(1e-5), Float32(1.0) - ndoth * ndoth)
    var dch = (Float32(2.0) + inv_sr) * exp(inv_sr * Float32(0.5) * log(sin2)) * Float32(0.15915494)
    var sv = dch / max(Float32(0.25), Float32(4.0) * (ndotl + ndotv - ndotl * ndotv))
    var base = (diff + spec + p.sheen * sv) * ndotl
    var wrap = max(Float32(0.0), (raw_nl + Float32(0.45)) / Float32(1.45)) * SOTA_MAT_INV_PI
    base = base + p.sss * wrap
    var cc_a = max(Float32(0.002), p.clearcoat_roughness * p.clearcoat_roughness)
    var cc_a2 = cc_a * cc_a
    var cden = ndoth * ndoth * (cc_a2 - Float32(1.0)) + Float32(1.0)
    var cc_d = cc_a2 / max(Float32(1e-6), SOTA_MAT_PI * cden * cden)
    var cc_vis = Float32(0.25) / max(Float32(1e-4), vdoth * vdoth)
    var f_cc = Float32(0.04) + Float32(0.96) * omv5
    var cc_spec = p.clearcoat * cc_d * cc_vis * f_cc * ndotl
    var atten = Float32(1.0) - p.clearcoat * f_cc
    return base * atten + Vec3(cc_spec, cc_spec, cc_spec)


def sota_mat_legacy_ambient_total_cpu(p: SotaMatParams, llut: SotaLegacyLut, ndotv_in: Float32) -> Vec3:
    """White-furnace output of the legacy deferred composition (env = irradiance = 1, AO = 1):
    `dw + spec_out` incl. the clearcoat term, i.e. what `furnace_white_dome` measures. Direct light is 0."""
    var ndotv = max(Float32(1e-3), ndotv_in)
    var m = p.metallic
    var diel = Float32(0.04) * p.specular
    var f0 = Vec3(diel, diel, diel) * (Float32(1.0) - m) + p.albedo * m
    var f90 = p.specular * (Float32(1.0) - m) + m
    var lab = llut.sample(max(Float32(0.01), min(Float32(0.99), ndotv)), max(Float32(0.01), min(Float32(0.99), p.roughness)))
    var lcc = llut.sample(max(Float32(0.01), min(Float32(0.99), ndotv)), max(Float32(0.01), min(Float32(0.99), p.clearcoat_roughness)))
    var e_ss = max(Float32(0.02), min(Float32(1.0), lab[0] + lab[1]))
    var om = Float32(1.0) - e_ss
    var f0s = [f0.x, f0.y, f0.z]
    var alb = [p.albedo.x, p.albedo.y, p.albedo.z]
    var out = List[Float32]()
    var r_coat = p.clearcoat * (Float32(0.04) * lcc[0] + lcc[1])
    for k in range(3):
        var fss = f0s[k] * lab[0] + f90 * lab[1]
        var favg = f0s[k] + (f90 - f0s[k]) / Float32(21.0)
        var spec_tot = fss + fss * favg / max(Float32(1e-4), Float32(1.0) - favg * om) * om
        var kd = (Float32(1.0) - m) * max(Float32(0.0), Float32(1.0) - spec_tot)
        out.append(r_coat + (Float32(1.0) - r_coat) * (spec_tot + kd * alb[k]))
    return Vec3(out[0], out[1], out[2])


def sota_mat_hemisphere_albedo(fn_vals: List[Vec3], strata: Int) -> Vec3:
    """Mean of `pi * f|cos| / cos` samples taken on a cosine-stratified grid (helper for the legacy direct)."""
    var acc = Vec3(0.0, 0.0, 0.0)
    for k in range(len(fn_vals)):
        acc = acc + fn_vals[k]
    return acc * (SOTA_MAT_PI / Float32(strata * strata))


# ---------------------------------------------------------------------------------------------
# 5. CPU path tracer of a sphere scene with the renderer's camera
# ---------------------------------------------------------------------------------------------
struct SotaMatSphere(Copyable, Movable):
    var center: Vec3
    var radius: Float32
    var params: SotaMatParams

    def __init__(out self, center: Vec3, radius: Float32, var params: SotaMatParams):
        self.center = center
        self.radius = radius
        self.params = params^


struct SotaMatScene(Movable):
    """Spheres with the planar front-projection UV of `tests/test_flint_v5_sota_mat.mojo` (u along +x, v along
    -y, seen from -z): the mesh tangent is dP/du, the bitangent points along dP/dv. Lighting: `sky` (or a white
    furnace) plus a directional sun of radiance `sun_rad` (irradiance at normal incidence)."""
    var spheres: List[SotaMatSphere]
    var sky: SkyDomeParams
    var furnace: Bool
    var sun_dir: Vec3
    var sun_rad: Vec3
    var ibl_intensity: Float32
    var cam_pos: Vec3
    var cam_target: Vec3
    var fov_y_deg: Float32

    def __init__(out self):
        self.spheres = List[SotaMatSphere]()
        self.sky = SkyDomeParams()
        self.furnace = False
        self.sun_dir = Vec3(0.3, 0.8, -0.5).normalized()
        self.sun_rad = Vec3(0.0, 0.0, 0.0)
        self.ibl_intensity = Float32(1.0)
        self.cam_pos = Vec3(0.0, 0.0, -6.0)
        self.cam_target = Vec3(0.0, 0.0, 0.0)
        self.fov_y_deg = Float32(30.0)

    def env(self, d: Vec3) -> Vec3:
        if self.furnace:
            return Vec3(1.0, 1.0, 1.0)
        return self.sky.eval(d) * self.ibl_intensity

    def hit(self, ro: Vec3, rd: Vec3) -> Tuple[Int, Float32]:
        var best = -1
        var bt = Float32(1e30)
        for i in range(len(self.spheres)):
            var oc = ro - self.spheres[i].center
            var bq = oc.dot(rd)
            var cq = oc.dot(oc) - self.spheres[i].radius * self.spheres[i].radius
            var disc = bq * bq - cq
            if disc < Float32(0.0):
                continue
            var t = -bq - sqrt(disc)
            if t > Float32(1e-4) and t < bt:
                bt = t
                best = i
        return (best, bt)

    def primary_dir(self, px: Int, py: Int, w: Int, h: Int, jx: Float32, jy: Float32) -> Vec3:
        var fwd = (self.cam_target - self.cam_pos).normalized()
        var rgt = view_right(fwd)
        var up = fwd.cross(rgt).normalized()
        var tan_y = tan(self.fov_y_deg * Float32(3.14159265 / 180.0) * Float32(0.5))
        var tan_x = tan_y * Float32(w) / Float32(h)
        var ndc_x = (Float32(px) + jx) / Float32(w) * Float32(2.0) - Float32(1.0)
        var ndc_y = Float32(1.0) - (Float32(py) + jy) / Float32(h) * Float32(2.0)
        return (fwd + rgt * (ndc_x * tan_x) + up * (ndc_y * tan_y)).normalized()

    def frame_at(self, i: Int, p: Vec3) -> Tuple[Vec3, Vec3, Vec3]:
        var c = self.spheres[i].center
        var r = self.spheres[i].radius
        var n = ((p - c) * (Float32(1.0) / r)).normalized()
        var zf = max(Float32(0.05), -n.z)   # depth toward the camera (-z hemisphere)
        var dpdu = Vec3(1.0, 0.0, n.x / zf)
        var dpdv = Vec3(0.0, -1.0, -n.y / zf)
        var t = (dpdu - n * n.dot(dpdu)).normalized()
        var b = n.cross(t)
        if b.dot(dpdv) < Float32(0.0):
            b = b * Float32(-1.0)
        return (n, t, b)

    def render(self, w: Int, h: Int, spp: Int, bounces: Int, occlusion: Bool, lut: SotaMatLut, seed: UInt64) -> List[Vec3]:
        """`bounces` = 1 with `occlusion` off is the integral the GPU split-sum + unshadowed sun approximates:
        each surface sees the unoccluded environment and the sun. Pixel centres only (no AA)."""
        var img = List[Vec3](length=w * h, fill=Vec3(0.0, 0.0, 0.0))
        for py in range(h):
            for px in range(w):
                var rng = SotaRng(seed + UInt64(py * w + px) * UInt64(7919))
                var d0 = self.primary_dir(px, py, w, h, Float32(0.5), Float32(0.5))
                var h0 = self.hit(self.cam_pos, d0)
                if h0[0] < 0:
                    img[py * w + px] = self.env(d0)
                    continue
                var acc = Vec3(0.0, 0.0, 0.0)
                for _ in range(spp):
                    var ro = self.cam_pos
                    var rd = d0
                    var thr = Vec3(1.0, 1.0, 1.0)
                    var hi = h0[0]
                    var ht = h0[1]
                    var l_acc = Vec3(0.0, 0.0, 0.0)
                    for bounce in range(bounces):
                        var p = ro + rd * ht
                        var fr = self.frame_at(hi, p)
                        var n = fr[0]
                        var v = rd * Float32(-1.0)
                        if n.dot(v) < Float32(0.0):
                            break
                        var prm = self.spheres[hi].params.copy()
                        if self.sun_rad.x + self.sun_rad.y + self.sun_rad.z > Float32(0.0):
                            var fs = sota_mat_eval_cpu(prm, lut, n, fr[1], fr[2], v, self.sun_dir)
                            l_acc = l_acc + _vm(thr, _vm(fs, self.sun_rad))
                        var smp = sota_mat_sample_cpu(prm, lut, n, fr[1], fr[2], v, rng)
                        if not smp.ok:
                            break
                        thr = _vm(thr, smp.weight)
                        var nl = n.dot(smp.dir)
                        ro = p + n * (Float32(1e-3) if nl > Float32(0.0) else Float32(-1e-3))
                        rd = smp.dir
                        var nh = self.hit(ro, rd) if (occlusion and bounce + 1 < bounces) else (-1, Float32(0.0))
                        if nh[0] < 0:
                            l_acc = l_acc + _vm(thr, self.env(rd))
                            break
                        hi = nh[0]
                        ht = nh[1]
                    acc = acc + l_acc
                img[py * w + px] = acc * (Float32(1.0) / Float32(spp))
        return img^
