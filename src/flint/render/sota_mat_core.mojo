"""Area I (`sota:mat`): layered, energy-conserving material model for the deferred shade pass.

One model, two twins: every function here has a CPU version (used by the furnace tests, the CPU path tracer in
`sota_mat_reference.mojo` and the emitter probe) and a SPIR-V version emitted into `aaa_deferred_shade_shader`
through two `# --- sota:mat hook ---` blocks. Both read the same packed material words and the same LUT words,
so they agree to float rounding.

Layer stack (top to bottom), each lower layer scaled by one minus the directional albedo of the layers above it
(albedo scaling, exact in a white furnace):
  1. clearcoat:  isotropic GGX dielectric, F0 0.04, own roughness (`AaaMaterial.clearcoat*`).
  2. sheen:      Charlie NDF with the Estevez-Kulla 2017 fitted Lambda visibility, alpha = sheen_roughness^2,
                 base scaled by `1 - max(sheen) * E_sheen(mu_v)`.
  3. base:       anisotropic GGX (KHR_materials_anisotropy: alpha_t = mix(alpha, 1, k^2)), height-correlated
                 Smith, Schlick or Belcour-Barla 2017 thin-film Fresnel, Kulla-Conty 2017 multiple-scattering
                 lobe; dielectric part = reciprocal energy-conserving diffuse
                 `c/pi (1-E_s(mu_v))(1-E_s(mu_l))/(1-E_s_avg)` plus thin-walled transmission (the reflection
                 lobe mirrored through the surface, normalised to `(1-E_s(mu_v))`).
With every colour white, `coat + sheen + spec + diffuse + transmission` integrate to exactly 1 for any roughness,
anisotropy, coat, sheen or transmission weight.

Data:
  - 24 material words are unchanged (`aaa_bsdf.mojo`). Area I adds an 8-word extension per material
    (`SotaMatLayers`: anisotropy strength / rotation, transmission + tint) after the 1024 material slots of
    `buf_materials`, and a ~36k-word LUT block after that (2-D `A`, `B` split-sum Schlick albedos of GGX,
    `E_sheen`, 4-D anisotropic `A`, `B` over (mu, phi, r, k), and their cosine-weighted averages), baked on the CPU on first use.
  - cvar `r.sota_mat.layered` = `FlintAaaVulkanRenderer.sota_mat.enabled` (default off, env `FLINT_SOTA_MAT=1`)
    sets `AAA_FLAG_SOTA_MAT` in `globals[99]`. Off, the deferred pass computes exactly the legacy values.
    Character materials of area E (iridescence < 0) always keep their own models.

Sources: Kulla & Conty 2017 (Revisiting Physically Based Shading at Imageworks); Estevez & Kulla 2017
(Production Friendly Microfacet Sheen BRDF); Belcour & Barla 2017 (A Practical Extension to Microfacet Theory
for the Modeling of Varying Iridescence); Heitz 2014 (Understanding the Masking-Shadowing Function); Dupuy & Benyoub 2023
(Sampling Visible GGX Normals with Spherical Caps); Kelemen & Szirmay-Kalos 2001 (energy-conserving coupled
diffuse); Khronos KHR_materials_{clearcoat,sheen,anisotropy,iridescence,transmission} specs and the
glTF-Sample-Viewer (Apache-2.0) for the sheen Lambda fit constants and the iridescence sensitivity fit.
"""
from std.collections import List
from std.math import abs, atan2, cos, exp, floor, log, max, min, sin, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.platform.os import env_value, get_time
from flint.platform.vulkan import put32, put_float
from flint.render.aaa_bsdf import AAA_MATERIAL_WORDS
from flint.render.spirv import AaaSpirvModule
from flint.render.aaa_visbuffer import emit_vec3_cross, emit_vec3_dot, emit_vec3_normalize

# `globals[99]` render flag (high bit, clear of area E's 1 << 20).
comptime AAA_FLAG_SOTA_MAT: UInt32 = 1 << 24

comptime SOTA_MAT_PI: Float32 = 3.14159265
comptime SOTA_MAT_INV_PI: Float32 = 0.318309886
comptime SOTA_MAT_MAX_MATERIALS: Int = 1024
comptime SOTA_MAT_EXT_WORDS: Int = 8
comptime SOTA_MAT_EXT_BASE: Int = SOTA_MAT_MAX_MATERIALS * AAA_MATERIAL_WORDS
comptime SOTA_MAT_MAGIC: UInt32 = 0x544D4153  # 'SAMT'
comptime SOTA_MAT_LUT_N: Int = 32
comptime SOTA_MAT_LUT_BASE: Int = SOTA_MAT_EXT_BASE + SOTA_MAT_MAX_MATERIALS * SOTA_MAT_EXT_WORDS
comptime SOTA_MAT_LUT_A: Int = 0
comptime SOTA_MAT_LUT_B: Int = SOTA_MAT_LUT_N * SOTA_MAT_LUT_N
comptime SOTA_MAT_LUT_SH: Int = 2 * SOTA_MAT_LUT_N * SOTA_MAT_LUT_N
comptime SOTA_MAT_LUT_AAVG: Int = 3 * SOTA_MAT_LUT_N * SOTA_MAT_LUT_N
comptime SOTA_MAT_LUT_BAVG: Int = SOTA_MAT_LUT_AAVG + SOTA_MAT_LUT_N
# Anisotropic GGX albedo: 4-D (mu, view azimuth phi in [0, pi/2], roughness r, anisotropy k), node-aligned.
comptime SOTA_MAT_L4_MU: Int = 16
comptime SOTA_MAT_L4_PHI: Int = 8
comptime SOTA_MAT_L4_R: Int = 16
comptime SOTA_MAT_L4_K: Int = 8
comptime SOTA_MAT_L4_SIZE: Int = SOTA_MAT_L4_MU * SOTA_MAT_L4_PHI * SOTA_MAT_L4_R * SOTA_MAT_L4_K
comptime SOTA_MAT_LUT_A4: Int = SOTA_MAT_LUT_BAVG + SOTA_MAT_LUT_N
comptime SOTA_MAT_LUT_B4: Int = SOTA_MAT_LUT_A4 + SOTA_MAT_L4_SIZE
comptime SOTA_MAT_LUT_AAVG4: Int = SOTA_MAT_LUT_B4 + SOTA_MAT_L4_SIZE
comptime SOTA_MAT_LUT_BAVG4: Int = SOTA_MAT_LUT_AAVG4 + SOTA_MAT_L4_R * SOTA_MAT_L4_K
comptime SOTA_MAT_LUT_WORDS: Int = SOTA_MAT_LUT_BAVG4 + SOTA_MAT_L4_R * SOTA_MAT_L4_K
# Extra bytes appended to `buf_materials`: extension table + LUT block.
comptime SOTA_MAT_EXTRA_BYTES: Int = (SOTA_MAT_MAX_MATERIALS * SOTA_MAT_EXT_WORDS + SOTA_MAT_LUT_WORDS) * 4

comptime SOTA_MAT_ALPHA_MIN: Float32 = 0.002
comptime SOTA_MAT_SHEEN_R_MIN: Float32 = 0.07
comptime SOTA_MAT_MU_MIN: Float32 = 0.001


# ---------------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------------
struct SotaMatLayers(Copyable, Movable):
    """Per-material extension words (the layers `AaaMaterial` has no room for)."""
    var anisotropy: Float32           # KHR_materials_anisotropy strength, 0..1
    var anisotropy_rotation: Float32  # radians, rotates the stretch direction from the mesh tangent toward the bitangent
    var transmission: Float32         # thin-walled transmission weight of the dielectric base, 0..1
    var transmission_tint: Vec3       # colour of light passing through

    def __init__(out self):
        self.anisotropy = Float32(0.0)
        self.anisotropy_rotation = Float32(0.0)
        self.transmission = Float32(0.0)
        self.transmission_tint = Vec3(1.0, 1.0, 1.0)

    def append_words(self, mut out: List[UInt32]):
        out.append(SOTA_MAT_MAGIC)
        out.append(bitcast[DType.uint32](max(Float32(0.0), min(Float32(1.0), self.anisotropy))))
        out.append(bitcast[DType.uint32](self.anisotropy_rotation))
        out.append(bitcast[DType.uint32](max(Float32(0.0), min(Float32(1.0), self.transmission))))
        out.append(bitcast[DType.uint32](max(Float32(0.0), self.transmission_tint.x)))
        out.append(bitcast[DType.uint32](max(Float32(0.0), self.transmission_tint.y)))
        out.append(bitcast[DType.uint32](max(Float32(0.0), self.transmission_tint.z)))
        out.append(UInt32(0))


struct SotaMatParams(Copyable, Movable):
    """Decoded shading inputs, read from the packed words exactly as the GPU decode reads them."""
    var albedo: Vec3
    var specular: Float32
    var roughness: Float32
    var metallic: Float32
    var clearcoat: Float32
    var clearcoat_roughness: Float32
    var irid: Float32
    var irid_nm: Float32
    var irid_ior: Float32
    var sheen: Vec3
    var sheen_roughness: Float32
    var sss: Vec3
    var aniso: Float32
    var aniso_rot: Float32
    var transmission: Float32
    var tint: Vec3

    def __init__(out self):
        self.albedo = Vec3(0.8, 0.8, 0.8)
        self.specular = Float32(1.0)
        self.roughness = Float32(0.5)
        self.metallic = Float32(0.0)
        self.clearcoat = Float32(0.0)
        self.clearcoat_roughness = Float32(0.08)
        self.irid = Float32(0.0)
        self.irid_nm = Float32(380.0)
        self.irid_ior = Float32(1.65)
        self.sheen = Vec3(0.0, 0.0, 0.0)
        self.sheen_roughness = Float32(0.5)
        self.sss = Vec3(0.0, 0.0, 0.0)
        self.aniso = Float32(0.0)
        self.aniso_rot = Float32(0.0)
        self.transmission = Float32(0.0)
        self.tint = Vec3(1.0, 1.0, 1.0)


@always_inline
def _wf(words: List[UInt32], i: Int) -> Float32:
    return bitcast[DType.float32](words[i])


def sota_mat_params_from_words(words: List[UInt32], ext: List[UInt32]) -> SotaMatParams:
    """`words` = the 24 packed material words, `ext` = the 8 extension words (may be empty or unmagicked)."""
    var p = SotaMatParams()
    p.albedo = Vec3(_wf(words, 0), _wf(words, 1), _wf(words, 2))
    p.specular = abs(_wf(words, 3))
    p.roughness = _wf(words, 4)
    p.metallic = _wf(words, 5)
    p.clearcoat = _wf(words, 6)
    p.clearcoat_roughness = _wf(words, 7)
    p.irid = max(Float32(0.0), _wf(words, 13))
    p.irid_nm = _wf(words, 14)
    var ior = _wf(words, 15)
    p.irid_ior = Float32(1.5) if ior < Float32(0.0) else ior  # word 15 < 0 is area A's VT code
    p.sheen = Vec3(_wf(words, 16), _wf(words, 17), _wf(words, 18))
    p.sheen_roughness = _wf(words, 19)
    p.sss = Vec3(_wf(words, 20), _wf(words, 21), _wf(words, 22))
    if len(ext) >= SOTA_MAT_EXT_WORDS and ext[0] == SOTA_MAT_MAGIC:
        p.aniso = _wf(ext, 1)
        p.aniso_rot = _wf(ext, 2)
        p.transmission = _wf(ext, 3)
        p.tint = Vec3(_wf(ext, 4), _wf(ext, 5), _wf(ext, 6))
    else:
        p.tint = Vec3(0.0, 0.0, 0.0)
    return p^


# ---------------------------------------------------------------------------------------------
# Small math (CPU)
# ---------------------------------------------------------------------------------------------
@always_inline
def _vm(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z)


@always_inline
def _vlerp(a: Vec3, b: Vec3, t: Float32) -> Vec3:
    return a + (b - a) * t


@always_inline
def _vs(x: Float32) -> Vec3:
    return Vec3(x, x, x)


@always_inline
def _sat(x: Float32) -> Float32:
    return max(Float32(0.0), min(Float32(1.0), x))


@always_inline
def _powf(x: Float32, y: Float32) -> Float32:
    if x <= Float32(0.0):
        return Float32(0.0)
    return exp(y * log(x))


@always_inline
def _pow5(x: Float32) -> Float32:
    var x2 = x * x
    return x2 * x2 * x


def sota_mat_lambda_aniso(at: Float32, ab: Float32, wt: Float32, wb: Float32, wn: Float32) -> Float32:
    """Smith Lambda of anisotropic GGX for a direction with local components (t, b, n), n > 0."""
    var n2 = max(Float32(1e-12), wn * wn)
    return Float32(0.5) * (sqrt(Float32(1.0) + (at * at * wt * wt + ab * ab * wb * wb) / n2) - Float32(1.0))


def sota_mat_d_aniso(at: Float32, ab: Float32, ht: Float32, hb: Float32, hn: Float32) -> Float32:
    var q = ht * ht / (at * at) + hb * hb / (ab * ab) + hn * hn
    return Float32(1.0) / max(Float32(1e-20), SOTA_MAT_PI * at * ab * q * q)


def sota_mat_vis_aniso(
    at: Float32, ab: Float32, vt: Float32, vb: Float32, vn: Float32, lt: Float32, lb: Float32, ln: Float32
) -> Float32:
    """Height-correlated Smith G2 / (4 mu_v mu_l)."""
    var sv = sqrt(at * at * vt * vt + ab * ab * vb * vb + vn * vn)
    var sl = sqrt(at * at * lt * lt + ab * ab * lb * lb + ln * ln)
    return Float32(0.5) / max(Float32(1e-12), ln * sv + vn * sl)


# ---------------------------------------------------------------------------------------------
# Sheen (Charlie NDF + Estevez-Kulla Lambda fit)
# ---------------------------------------------------------------------------------------------
@always_inline
def _sheen_l(x: Float32, alpha: Float32) -> Float32:
    var t = (Float32(1.0) - alpha) * (Float32(1.0) - alpha)
    var a = Float32(21.5473) + (Float32(25.3245) - Float32(21.5473)) * t
    var b = Float32(3.82987) + (Float32(3.32435) - Float32(3.82987)) * t
    var c = Float32(0.19823) + (Float32(0.16801) - Float32(0.19823)) * t
    var d = Float32(-1.97760) + (Float32(-1.27393) - Float32(-1.97760)) * t
    var e = Float32(-4.32054) + (Float32(-4.85967) - Float32(-4.32054)) * t
    return a / (Float32(1.0) + b * _powf(x, c)) + d * x + e


def sota_mat_sheen_lambda(cos_t: Float32, alpha: Float32) -> Float32:
    if cos_t < Float32(0.5):
        return exp(_sheen_l(cos_t, alpha))
    return exp(Float32(2.0) * _sheen_l(Float32(0.5), alpha) - _sheen_l(Float32(1.0) - cos_t, alpha))


def sota_mat_sheen_dv(ndoth: Float32, ndotv: Float32, ndotl: Float32, sheen_rough: Float32) -> Float32:
    """`D_charlie * V_sheen` (the sheen BRDF without colour)."""
    var r = max(SOTA_MAT_SHEEN_R_MIN, sheen_rough)
    var alpha = r * r
    var inv_a = Float32(1.0) / alpha
    var sin2 = max(Float32(1e-5), Float32(1.0) - ndoth * ndoth)
    var d = (Float32(2.0) + inv_a) * _powf(sin2, inv_a * Float32(0.5)) * Float32(0.15915494)
    var den = (Float32(1.0) + sota_mat_sheen_lambda(ndotv, alpha) + sota_mat_sheen_lambda(ndotl, alpha)) * (
        Float32(4.0) * ndotv * ndotl
    )
    var v = min(Float32(1.0), Float32(1.0) / max(Float32(1e-6), den))
    return d * v


# ---------------------------------------------------------------------------------------------
# Thin-film iridescence (Belcour & Barla 2017, closed-form spectral integration)
# ---------------------------------------------------------------------------------------------
@always_inline
def _sq(x: Float32) -> Float32:
    return x * x


def _xyz_to_rgb(x: Float32, y: Float32, z: Float32) -> Vec3:
    return Vec3(
        Float32(3.2404542) * x - Float32(1.5371385) * y - Float32(0.4985314) * z,
        Float32(-0.9692660) * x + Float32(1.8760108) * y + Float32(0.0415560) * z,
        Float32(0.0556434) * x - Float32(0.2040259) * y + Float32(1.0572252) * z,
    )


def _eval_sensitivity(opd: Float32, shift: Vec3) -> Vec3:
    var phase = Float32(6.28318530718e-9) * opd
    var p2 = phase * phase
    var x = Float32(5.4856e-13) * sqrt(Float32(6.28318530718) * Float32(4.3278e9)) * cos(Float32(1.6810e6) * phase + shift.x) * exp(-p2 * Float32(4.3278e9))
    var y = Float32(4.4201e-13) * sqrt(Float32(6.28318530718) * Float32(9.3046e9)) * cos(Float32(1.7953e6) * phase + shift.y) * exp(-p2 * Float32(9.3046e9))
    var z = Float32(5.2481e-13) * sqrt(Float32(6.28318530718) * Float32(6.6121e9)) * cos(Float32(2.2084e6) * phase + shift.z) * exp(-p2 * Float32(6.6121e9))
    x += Float32(9.7470e-14) * sqrt(Float32(6.28318530718) * Float32(4.5282e9)) * cos(Float32(2.2399e6) * phase + shift.x) * exp(-p2 * Float32(4.5282e9))
    var k = Float32(1.0) / Float32(1.0685e-7)
    return _xyz_to_rgb(x * k, y * k, z * k)


@always_inline
def _f0_to_ior(f0: Float32) -> Float32:
    var s = sqrt(max(Float32(0.0), min(Float32(0.9999), f0)))
    return (Float32(1.0) + s) / (Float32(1.0) - s)


@always_inline
def _ior_to_f0(n_t: Float32, n_i: Float32) -> Float32:
    return _sq((n_t - n_i) / (n_t + n_i))


@always_inline
def _schlick1(f0: Float32, c: Float32) -> Float32:
    return f0 + (Float32(1.0) - f0) * _pow5(Float32(1.0) - _sat(c))


def sota_mat_iridescence_cpu(cos1: Float32, thickness_nm: Float32, film_ior: Float32, base_f0: Vec3) -> Vec3:
    """Belcour-Barla Airy reflectance of a film of `thickness_nm` / `film_ior` in air over a base of F0 `base_f0`."""
    var outside = Float32(1.0)
    var ts = _sat(thickness_nm / Float32(0.03))
    var eta2 = outside + (film_ior - outside) * (ts * ts * (Float32(3.0) - Float32(2.0) * ts))
    var sin2 = _sq(outside / eta2) * (Float32(1.0) - cos1 * cos1)
    var cos2sq = Float32(1.0) - sin2
    if cos2sq < Float32(0.0):
        return Vec3(1.0, 1.0, 1.0)
    var cos2 = sqrt(cos2sq)
    var r12 = _schlick1(_ior_to_f0(eta2, outside), cos1)
    var t121 = Float32(1.0) - r12
    var phi12 = SOTA_MAT_PI if eta2 < outside else Float32(0.0)
    var phi21 = SOTA_MAT_PI - phi12
    var bf = [base_f0.x, base_f0.y, base_f0.z]
    var out = List[Float32]()
    var phis = List[Float32]()
    var r123s = List[Float32]()
    var rss = List[Float32]()
    for k in range(3):
        var bior = _f0_to_ior(bf[k])
        var r23 = _schlick1(_ior_to_f0(bior, eta2), cos2)
        var phi23 = SOTA_MAT_PI if bior < eta2 else Float32(0.0)
        phis.append(phi21 + phi23)
        var r123 = max(Float32(1e-5), min(Float32(0.9999), r12 * r23))
        r123s.append(r123)
        var rs = t121 * t121 * r23 / (Float32(1.0) - r123)
        rss.append(rs)
        out.append(r12 + rs)
    var opd = Float32(2.0) * eta2 * thickness_nm * cos2
    var cm = [rss[0] - t121, rss[1] - t121, rss[2] - t121]
    for m in range(1, 3):
        for k in range(3):
            cm[k] = cm[k] * sqrt(r123s[k])
        var sm = _eval_sensitivity(Float32(m) * opd, Vec3(Float32(m) * phis[0], Float32(m) * phis[1], Float32(m) * phis[2]))
        out[0] += cm[0] * Float32(2.0) * sm.x
        out[1] += cm[1] * Float32(2.0) * sm.y
        out[2] += cm[2] * Float32(2.0) * sm.z
    return Vec3(max(Float32(0.0), out[0]), max(Float32(0.0), out[1]), max(Float32(0.0), out[2]))


# ---------------------------------------------------------------------------------------------
# LUTs
# ---------------------------------------------------------------------------------------------
@always_inline
def _rad_inv(bits_in: UInt32) -> Float32:
    var b = bits_in
    b = (b << 16) | (b >> 16)
    b = ((b & 0x55555555) << 1) | ((b & 0xAAAAAAAA) >> 1)
    b = ((b & 0x33333333) << 2) | ((b & 0xCCCCCCCC) >> 2)
    b = ((b & 0x0F0F0F0F) << 4) | ((b & 0xF0F0F0F0) >> 4)
    b = ((b & 0x00FF00FF) << 8) | ((b & 0xFF00FF00) >> 8)
    return Float32(b) * Float32(2.3283064365386963e-10)


def sota_mat_sample_vndf(vt: Float32, vb: Float32, vn: Float32, at: Float32, ab: Float32, u1: Float32, u2: Float32) -> Vec3:
    """Visible-normal sample of anisotropic GGX in the local (t, b, n) frame (Dupuy & Benyoub 2023 spherical
    caps; valid for any incident direction, including below the horizon). Returns the microfacet normal."""
    var wi = Vec3(at * vt, ab * vb, vn).normalized()
    var phi = Float32(6.28318530718) * u2
    var z = (Float32(1.0) - u1) * (Float32(1.0) + wi.z) - wi.z
    var st = sqrt(max(Float32(0.0), min(Float32(1.0), Float32(1.0) - z * z)))
    var h = Vec3(st * cos(phi) + wi.x, st * sin(phi) + wi.y, z + wi.z)
    return Vec3(at * h.x, ab * h.y, max(Float32(0.0), h.z)).normalized()


@always_inline
def _l4_index(m: Int, p: Int, r: Int, k: Int) -> Int:
    return ((k * SOTA_MAT_L4_R + r) * SOTA_MAT_L4_PHI + p) * SOTA_MAT_L4_MU + m


struct _L4Coords(Copyable, Movable):
    var m0: Int
    var p0: Int
    var r0: Int
    var k0: Int
    var tm: Float32
    var tp: Float32
    var tr: Float32
    var tk: Float32

    def __init__(out self):
        self.m0 = 0
        self.p0 = 0
        self.r0 = 0
        self.k0 = 0
        self.tm = Float32(0.0)
        self.tp = Float32(0.0)
        self.tr = Float32(0.0)
        self.tk = Float32(0.0)


@always_inline
def _l4_axis(x: Float32, n: Int) -> Tuple[Int, Float32]:
    var f = _sat(x) * Float32(n - 1)
    var f0 = min(floor(f), Float32(n - 2))
    return (Int(f0), f - f0)


def _l4_coords(mu: Float32, phi: Float32, r: Float32, k: Float32) -> _L4Coords:
    var c = _L4Coords()
    var a = _l4_axis(mu, SOTA_MAT_L4_MU)
    c.m0 = a[0]
    c.tm = a[1]
    a = _l4_axis(phi * Float32(0.636619772), SOTA_MAT_L4_PHI)
    c.p0 = a[0]
    c.tp = a[1]
    a = _l4_axis(r, SOTA_MAT_L4_R)
    c.r0 = a[0]
    c.tr = a[1]
    a = _l4_axis(k, SOTA_MAT_L4_K)
    c.k0 = a[0]
    c.tk = a[1]
    return c^


def sota_mat_view_phi(vt: Float32, vb: Float32) -> Float32:
    """View azimuth folded into [0, pi/2] (the anisotropic albedo is symmetric in both tangent axes)."""
    return atan2(abs(vb), abs(vt))


struct SotaMatLut(Copyable, Movable):
    """The LUT block, exactly as uploaded after the extension table (`SOTA_MAT_LUT_WORDS` floats)."""
    var data: List[Float32]

    def __init__(out self):
        self.data = List[Float32](length=SOTA_MAT_LUT_WORDS, fill=Float32(0.0))

    @staticmethod
    def node_mu(i: Int) -> Float32:
        return max(SOTA_MAT_MU_MIN, Float32(i) / Float32(SOTA_MAT_LUT_N - 1))

    @staticmethod
    def node_r(j: Int) -> Float32:
        return Float32(j) / Float32(SOTA_MAT_LUT_N - 1)

    @staticmethod
    def bake(strata: Int = 32, sheen_strata: Int = 48, aniso_strata: Int = 16) -> SotaMatLut:
        var lut = SotaMatLut()
        comptime N = SOTA_MAT_LUT_N
        for j in range(N):
            var r = SotaMatLut.node_r(j)
            var a = max(SOTA_MAT_ALPHA_MIN, r * r)
            for i in range(N):
                var mu = SotaMatLut.node_mu(i)
                var vt = sqrt(max(Float32(0.0), Float32(1.0) - mu * mu))
                var lam_v = sota_mat_lambda_aniso(a, a, vt, Float32(0.0), mu)
                var acc_a = Float64(0.0)
                var acc_b = Float64(0.0)
                for sy in range(strata):
                    for sx in range(strata):
                        var u1 = (Float32(sx) + Float32(0.5)) / Float32(strata)
                        var u2 = (Float32(sy) + _rad_inv(UInt32(sx * strata + sy + 1)) * Float32(0.999)) / Float32(strata)
                        var h = sota_mat_sample_vndf(vt, Float32(0.0), mu, a, a, u1, u2)
                        var vh = vt * h.x + mu * h.z
                        var lx = Float32(2.0) * vh * h.x - vt
                        var ly = Float32(2.0) * vh * h.y
                        var lz = Float32(2.0) * vh * h.z - mu
                        if lz <= Float32(0.0):
                            continue
                        var lam_l = sota_mat_lambda_aniso(a, a, lx, ly, lz)
                        var w = (Float32(1.0) + lam_v) / (Float32(1.0) + lam_v + lam_l)
                        var fc = _pow5(Float32(1.0) - _sat(vh))
                        acc_a += Float64((Float32(1.0) - fc) * w)
                        acc_b += Float64(fc * w)
                var inv = Float64(1.0) / Float64(strata * strata)
                lut.data[SOTA_MAT_LUT_A + j * N + i] = Float32(acc_a * inv)
                lut.data[SOTA_MAT_LUT_B + j * N + i] = Float32(acc_b * inv)
        # Sheen albedo: E = pi * mean(D V) under cosine-weighted l (stratified)
        for j in range(N):
            var rs = SotaMatLut.node_r(j)
            for i in range(N):
                var mu = SotaMatLut.node_mu(i)
                var vt = sqrt(max(Float32(0.0), Float32(1.0) - mu * mu))
                var acc = Float64(0.0)
                for sy in range(sheen_strata):
                    for sx in range(sheen_strata):
                        var u1 = (Float32(sx) + Float32(0.5)) / Float32(sheen_strata)
                        var u2 = (Float32(sy) + Float32(0.5)) / Float32(sheen_strata)
                        var rr = sqrt(u1)
                        var ph = Float32(6.28318530718) * u2
                        var lx = rr * cos(ph)
                        var ly = rr * sin(ph)
                        var lz = sqrt(max(Float32(1e-6), Float32(1.0) - u1))
                        var hx = lx + vt
                        var hy = ly
                        var hz = lz + mu
                        var hl = sqrt(hx * hx + hy * hy + hz * hz)
                        acc += Float64(sota_mat_sheen_dv(hz / hl, mu, lz, rs))
                lut.data[SOTA_MAT_LUT_SH + j * N + i] = Float32(acc / Float64(sheen_strata * sheen_strata)) * SOTA_MAT_PI
        # Cosine-weighted averages 2 * integral(E(mu) mu dmu) of the piecewise-linear tables (exact)
        for j in range(N):
            var sa = Float64(0.0)
            var sb = Float64(0.0)
            for i in range(N - 1):
                var m0 = Float64(Float32(i) / Float32(N - 1))
                var m1 = Float64(Float32(i + 1) / Float32(N - 1))
                var dm = m1 - m0
                var a0 = Float64(lut.data[SOTA_MAT_LUT_A + j * N + i])
                var a1 = Float64(lut.data[SOTA_MAT_LUT_A + j * N + i + 1])
                var b0 = Float64(lut.data[SOTA_MAT_LUT_B + j * N + i])
                var b1 = Float64(lut.data[SOTA_MAT_LUT_B + j * N + i + 1])
                sa += dm * (a0 * (2.0 * m0 + m1) + a1 * (m0 + 2.0 * m1)) / 6.0
                sb += dm * (b0 * (2.0 * m0 + m1) + b1 * (m0 + 2.0 * m1)) / 6.0
            lut.data[SOTA_MAT_LUT_AAVG + j] = Float32(2.0 * sa)
            lut.data[SOTA_MAT_LUT_BAVG + j] = Float32(2.0 * sb)
        lut._bake_aniso(aniso_strata)
        return lut^

    def _bake_aniso(mut self, strata: Int):
        for kk in range(SOTA_MAT_L4_K):
            var k = Float32(kk) / Float32(SOTA_MAT_L4_K - 1)
            for rr in range(SOTA_MAT_L4_R):
                var r = Float32(rr) / Float32(SOTA_MAT_L4_R - 1)
                var a = max(SOTA_MAT_ALPHA_MIN, r * r)
                var at = max(SOTA_MAT_ALPHA_MIN, a + (Float32(1.0) - a) * k * k)
                for pp in range(SOTA_MAT_L4_PHI):
                    var phi = Float32(pp) / Float32(SOTA_MAT_L4_PHI - 1) * Float32(1.57079633)
                    for ii in range(SOTA_MAT_L4_MU):
                        var mu = max(SOTA_MAT_MU_MIN, Float32(ii) / Float32(SOTA_MAT_L4_MU - 1))
                        var st = sqrt(max(Float32(0.0), Float32(1.0) - mu * mu))
                        var vt = st * cos(phi)
                        var vb = st * sin(phi)
                        var lam_v = sota_mat_lambda_aniso(at, a, vt, vb, mu)
                        var acc_a = Float64(0.0)
                        var acc_b = Float64(0.0)
                        for sy in range(strata):
                            for sx in range(strata):
                                var u1 = (Float32(sx) + Float32(0.5)) / Float32(strata)
                                var u2 = (Float32(sy) + _rad_inv(UInt32(sx * strata + sy + 1)) * Float32(0.999)) / Float32(strata)
                                var h = sota_mat_sample_vndf(vt, vb, mu, at, a, u1, u2)
                                var vh = vt * h.x + vb * h.y + mu * h.z
                                var lz = Float32(2.0) * vh * h.z - mu
                                if lz <= Float32(0.0):
                                    continue
                                var lx = Float32(2.0) * vh * h.x - vt
                                var ly = Float32(2.0) * vh * h.y - vb
                                var lam_l = sota_mat_lambda_aniso(at, a, lx, ly, lz)
                                var w = (Float32(1.0) + lam_v) / (Float32(1.0) + lam_v + lam_l)
                                var fc = _pow5(Float32(1.0) - _sat(vh))
                                acc_a += Float64((Float32(1.0) - fc) * w)
                                acc_b += Float64(fc * w)
                        var inv = Float64(1.0) / Float64(strata * strata)
                        var idx = _l4_index(ii, pp, rr, kk)
                        self.data[SOTA_MAT_LUT_A4 + idx] = Float32(acc_a * inv)
                        self.data[SOTA_MAT_LUT_B4 + idx] = Float32(acc_b * inv)
                # (1/pi) * hemisphere integral of the multilinear interpolant times cos, exact by symmetry:
                # 4/pi * int_0^{pi/2} int_0^1 E mu dmu dphi (trapezoid in phi, exact linear-times-mu in mu).
                var sa = Float64(0.0)
                var sb = Float64(0.0)
                var dphi = Float64(1.57079633) / Float64(SOTA_MAT_L4_PHI - 1)
                for pp in range(SOTA_MAT_L4_PHI):
                    var wphi = dphi * (0.5 if (pp == 0 or pp == SOTA_MAT_L4_PHI - 1) else 1.0)
                    var ia = Float64(0.0)
                    var ib = Float64(0.0)
                    for ii in range(SOTA_MAT_L4_MU - 1):
                        var m0 = Float64(Float32(ii) / Float32(SOTA_MAT_L4_MU - 1))
                        var m1 = Float64(Float32(ii + 1) / Float32(SOTA_MAT_L4_MU - 1))
                        var dm = m1 - m0
                        var a0 = Float64(self.data[SOTA_MAT_LUT_A4 + _l4_index(ii, pp, rr, kk)])
                        var a1 = Float64(self.data[SOTA_MAT_LUT_A4 + _l4_index(ii + 1, pp, rr, kk)])
                        var b0 = Float64(self.data[SOTA_MAT_LUT_B4 + _l4_index(ii, pp, rr, kk)])
                        var b1 = Float64(self.data[SOTA_MAT_LUT_B4 + _l4_index(ii + 1, pp, rr, kk)])
                        ia += dm * (a0 * (2.0 * m0 + m1) + a1 * (m0 + 2.0 * m1)) / 6.0
                        ib += dm * (b0 * (2.0 * m0 + m1) + b1 * (m0 + 2.0 * m1)) / 6.0
                    sa += wphi * ia
                    sb += wphi * ib
                self.data[SOTA_MAT_LUT_AAVG4 + kk * SOTA_MAT_L4_R + rr] = Float32(sa * 4.0 / 3.14159265358979)
                self.data[SOTA_MAT_LUT_BAVG4 + kk * SOTA_MAT_L4_R + rr] = Float32(sb * 4.0 / 3.14159265358979)

    def fetch4(self, table: Int, mu: Float32, phi: Float32, r: Float32, k: Float32) -> Float32:
        """Quadrilinear fetch of an anisotropic table, the same arithmetic as `_e_lut4`."""
        var c = _l4_coords(mu, phi, r, k)
        var acc = Float32(0.0)
        for corner in range(4):
            var ri = c.r0 + (corner & 1)
            var ki = c.k0 + (corner >> 1)
            var b = table + _l4_index(c.m0, c.p0, ri, ki)
            var v00 = self.data[b]
            var v10 = self.data[b + 1]
            var v01 = self.data[b + SOTA_MAT_L4_MU]
            var v11 = self.data[b + SOTA_MAT_L4_MU + 1]
            var top = v00 + (v10 - v00) * c.tm
            var bot = v01 + (v11 - v01) * c.tm
            var e = top + (bot - top) * c.tp
            var wr = c.tr if (corner & 1) == 1 else (Float32(1.0) - c.tr)
            var wk = c.tk if (corner >> 1) == 1 else (Float32(1.0) - c.tk)
            acc += e * (wr * wk)
        return acc

    def fetch_avg4(self, table: Int, r: Float32, k: Float32) -> Float32:
        var c = _l4_coords(Float32(0.0), Float32(0.0), r, k)
        var b = table + c.k0 * SOTA_MAT_L4_R + c.r0
        var v00 = self.data[b]
        var v10 = self.data[b + 1]
        var v01 = self.data[b + SOTA_MAT_L4_R]
        var v11 = self.data[b + SOTA_MAT_L4_R + 1]
        var top = v00 + (v10 - v00) * c.tr
        var bot = v01 + (v11 - v01) * c.tr
        return top + (bot - top) * c.tk

    def fetch2(self, table: Int, mu: Float32, r: Float32) -> Float32:
        """Bilinear fetch, node-aligned (`mu`, `r` in [0, 1]) — the same arithmetic as `_emit_lut2`."""
        var fx = _sat(mu) * Float32(SOTA_MAT_LUT_N - 1)
        var fy = _sat(r) * Float32(SOTA_MAT_LUT_N - 1)
        var x0f = min(floor(fx), Float32(SOTA_MAT_LUT_N - 2))
        var y0f = min(floor(fy), Float32(SOTA_MAT_LUT_N - 2))
        var tx = fx - x0f
        var ty = fy - y0f
        var x0 = Int(x0f)
        var y0 = Int(y0f)
        var b = table + y0 * SOTA_MAT_LUT_N + x0
        var v00 = self.data[b]
        var v10 = self.data[b + 1]
        var v01 = self.data[b + SOTA_MAT_LUT_N]
        var v11 = self.data[b + SOTA_MAT_LUT_N + 1]
        var top = v00 + (v10 - v00) * tx
        var bot = v01 + (v11 - v01) * tx
        return top + (bot - top) * ty

    def fetch1(self, table: Int, r: Float32) -> Float32:
        var fy = _sat(r) * Float32(SOTA_MAT_LUT_N - 1)
        var y0f = min(floor(fy), Float32(SOTA_MAT_LUT_N - 2))
        var ty = fy - y0f
        var y0 = Int(y0f)
        var v0 = self.data[table + y0]
        var v1 = self.data[table + y0 + 1]
        return v0 + (v1 - v0) * ty

    def write_to(self, cpu_address: Int):
        """Uploads the block into a host-visible `buf_materials` mapping."""
        if cpu_address == 0:
            return
        for k in range(SOTA_MAT_LUT_WORDS):
            put_float(cpu_address + (SOTA_MAT_LUT_BASE + k) * 4, self.data[k])


def sota_mat_write_layers(cpu_address: Int, mat_id: Int, layers: SotaMatLayers):
    if cpu_address == 0 or mat_id < 0 or mat_id >= SOTA_MAT_MAX_MATERIALS:
        return
    var w = List[UInt32]()
    layers.append_words(w)
    for k in range(SOTA_MAT_EXT_WORDS):
        put32(cpu_address + (SOTA_MAT_EXT_BASE + mat_id * SOTA_MAT_EXT_WORDS + k) * 4, w[k])


def sota_mat_clear_layers(cpu_address: Int):
    if cpu_address == 0:
        return
    for k in range(SOTA_MAT_MAX_MATERIALS * SOTA_MAT_EXT_WORDS):
        put32(cpu_address + (SOTA_MAT_EXT_BASE + k) * 4, UInt32(0))


# ---------------------------------------------------------------------------------------------
# The model (CPU)
# ---------------------------------------------------------------------------------------------
struct SotaMatLobes(Copyable, Movable):
    """Everything that depends on the material and `mu_v` only (one evaluation per pixel)."""
    var f0: Vec3
    var f90: Float32
    var at: Float32
    var ab: Float32
    var r_eff: Float32
    var e_v: Float32
    var e_avg: Float32
    var fi_v: Vec3      # thin-film Fresnel at mu_v (blended into Fresnel averages)
    var fms: Vec3
    var es_v: Vec3      # base specular albedo E_s(mu_v)
    var es_avg: Vec3
    var ecc_v: Float32
    var esh_v: Float32
    var att_cc: Float32
    var att_sh: Float32
    var a_avg: Float32
    var b_avg: Float32

    def __init__(out self):
        self.f0 = Vec3(0.04, 0.04, 0.04)
        self.f90 = Float32(1.0)
        self.at = Float32(0.25)
        self.ab = Float32(0.25)
        self.r_eff = Float32(0.5)
        self.e_v = Float32(1.0)
        self.e_avg = Float32(1.0)
        self.fi_v = Vec3(0.0, 0.0, 0.0)
        self.fms = Vec3(0.0, 0.0, 0.0)
        self.es_v = Vec3(0.0, 0.0, 0.0)
        self.es_avg = Vec3(0.0, 0.0, 0.0)
        self.ecc_v = Float32(0.0)
        self.esh_v = Float32(0.0)
        self.att_cc = Float32(1.0)
        self.att_sh = Float32(1.0)
        self.a_avg = Float32(0.0)
        self.b_avg = Float32(0.0)


def _spec_ab(p: SotaMatParams, lut: SotaMatLut, r_eff: Float32, mu: Float32, phi: Float32) -> Tuple[Float32, Float32]:
    if p.aniso > Float32(0.0):
        return (
            lut.fetch4(SOTA_MAT_LUT_A4, mu, phi, p.roughness, p.aniso),
            lut.fetch4(SOTA_MAT_LUT_B4, mu, phi, p.roughness, p.aniso),
        )
    return (lut.fetch2(SOTA_MAT_LUT_A, mu, r_eff), lut.fetch2(SOTA_MAT_LUT_B, mu, r_eff))


def _spec_avg(p: SotaMatParams, lut: SotaMatLut, r_eff: Float32) -> Tuple[Float32, Float32]:
    if p.aniso > Float32(0.0):
        return (lut.fetch_avg4(SOTA_MAT_LUT_AAVG4, p.roughness, p.aniso), lut.fetch_avg4(SOTA_MAT_LUT_BAVG4, p.roughness, p.aniso))
    return (lut.fetch1(SOTA_MAT_LUT_AAVG, r_eff), lut.fetch1(SOTA_MAT_LUT_BAVG, r_eff))


def sota_mat_lobes_cpu(p: SotaMatParams, lut: SotaMatLut, ndotv: Float32, phi_v: Float32 = 0.0) -> SotaMatLobes:
    var o = SotaMatLobes()
    var m = p.metallic
    var diel = Float32(0.04) * p.specular
    o.f0 = Vec3(diel, diel, diel) * (Float32(1.0) - m) + p.albedo * m
    o.f90 = p.specular * (Float32(1.0) - m) + m
    var alpha = max(SOTA_MAT_ALPHA_MIN, p.roughness * p.roughness)
    o.at = max(SOTA_MAT_ALPHA_MIN, alpha + (Float32(1.0) - alpha) * p.aniso * p.aniso)
    o.ab = alpha
    o.r_eff = sqrt(sqrt(o.at * o.ab))
    var abv = _spec_ab(p, lut, o.r_eff, ndotv, phi_v)
    var a_v = abv[0]
    var b_v = abv[1]
    o.e_v = a_v + b_v
    var avg = _spec_avg(p, lut, o.r_eff)
    o.a_avg = avg[0]
    o.b_avg = avg[1]
    o.e_avg = o.a_avg + o.b_avg
    var w_i = p.irid
    o.fi_v = Vec3(0.0, 0.0, 0.0)
    if w_i > Float32(0.0):
        o.fi_v = sota_mat_iridescence_cpu(ndotv, p.irid_nm, p.irid_ior, o.f0)
    var favg = _vlerp(o.f0 + (_vs(o.f90) - o.f0) * Float32(1.0 / 21.0), o.fi_v, w_i)
    var om_eavg = Float32(1.0) - o.e_avg
    o.fms = Vec3(
        favg.x * favg.x * o.e_avg / max(Float32(1e-4), Float32(1.0) - favg.x * om_eavg),
        favg.y * favg.y * o.e_avg / max(Float32(1e-4), Float32(1.0) - favg.y * om_eavg),
        favg.z * favg.z * o.e_avg / max(Float32(1e-4), Float32(1.0) - favg.z * om_eavg),
    )
    var ef_v = _vlerp(o.f0 * a_v + _vs(o.f90 * b_v), o.fi_v * o.e_v, w_i)
    o.es_v = ef_v + o.fms * (Float32(1.0) - o.e_v)
    var ef_avg = _vlerp(o.f0 * o.a_avg + _vs(o.f90 * o.b_avg), o.fi_v * o.e_avg, w_i)
    o.es_avg = ef_avg + o.fms * om_eavg
    var rc = p.clearcoat_roughness
    o.ecc_v = Float32(0.04) * lut.fetch2(SOTA_MAT_LUT_A, ndotv, rc) + lut.fetch2(SOTA_MAT_LUT_B, ndotv, rc)
    o.esh_v = lut.fetch2(SOTA_MAT_LUT_SH, ndotv, p.sheen_roughness)
    o.att_cc = Float32(1.0) - p.clearcoat * o.ecc_v
    var smax = max(p.sheen.x, max(p.sheen.y, p.sheen.z))
    o.att_sh = Float32(1.0) - smax * o.esh_v
    return o^


def _es_at(p: SotaMatParams, o: SotaMatLobes, lut: SotaMatLut, mu: Float32, phi: Float32) -> Vec3:
    var ab = _spec_ab(p, lut, o.r_eff, mu, phi)
    var a = ab[0]
    var b = ab[1]
    var e = a + b
    var ef = _vlerp(o.f0 * a + _vs(o.f90 * b), o.fi_v * e, p.irid)
    return ef + o.fms * (Float32(1.0) - e)


struct SotaMatFrame(Copyable, Movable):
    var t: Vec3
    var b: Vec3

    def __init__(out self, t: Vec3, b: Vec3):
        self.t = t
        self.b = b


def sota_mat_frame_cpu(n: Vec3, t_in: Vec3, b_in: Vec3, rot: Float32) -> SotaMatFrame:
    """Gram-Schmidt the mesh tangent against the shading normal, keep the bitangent handedness, then rotate."""
    var t0 = (t_in - n * n.dot(t_in)).normalized()
    var b0 = n.cross(t0)
    if b0.dot(b_in) < Float32(0.0):
        b0 = b0 * Float32(-1.0)
    var c = cos(rot)
    var s = sin(rot)
    return SotaMatFrame(t0 * c + b0 * s, b0 * c - t0 * s)


def _fresnel_v(p: SotaMatParams, o: SotaMatLobes, vh: Float32) -> Vec3:
    var fc = _pow5(Float32(1.0) - _sat(vh))
    var fs = o.f0 + (_vs(o.f90) - o.f0) * fc
    if p.irid > Float32(0.0):
        return _vlerp(fs, sota_mat_iridescence_cpu(vh, p.irid_nm, p.irid_ior, o.f0), p.irid)
    return fs


def sota_mat_eval_cpu(p: SotaMatParams, lut: SotaMatLut, n: Vec3, t_in: Vec3, b_in: Vec3, v: Vec3, l: Vec3) -> Vec3:
    """BSDF times |cos theta_l| (no wrap lobe): the integrand of the CPU path tracer and of the furnace tests."""
    var ndotv = max(SOTA_MAT_MU_MIN, n.dot(v))
    var fr = sota_mat_frame_cpu(n, t_in, b_in, p.aniso_rot)
    var vt = v.dot(fr.t)
    var vb = v.dot(fr.b)
    var o = sota_mat_lobes_cpu(p, lut, ndotv, sota_mat_view_phi(vt, vb))
    var nl = n.dot(l)
    var m = p.metallic
    if nl > Float32(0.0):
        var h = (v + l).normalized()
        var vh = max(Float32(0.0), v.dot(h))
        var ndoth = max(Float32(0.0), n.dot(h))
        var d = sota_mat_d_aniso(o.at, o.ab, h.dot(fr.t), h.dot(fr.b), ndoth)
        var vis = sota_mat_vis_aniso(o.at, o.ab, vt, vb, ndotv, l.dot(fr.t), l.dot(fr.b), nl)
        var f = _fresnel_v(p, o, vh)
        var f_ss = f * (d * vis)
        var phi_l = sota_mat_view_phi(l.dot(fr.t), l.dot(fr.b))
        var ab_l = _spec_ab(p, lut, o.r_eff, nl, phi_l)
        var e_l = ab_l[0] + ab_l[1]
        var es_l = _es_at(p, o, lut, nl, phi_l)
        var ms_k = (Float32(1.0) - o.e_v) * (Float32(1.0) - e_l) * SOTA_MAT_INV_PI / max(Float32(1e-4), Float32(1.0) - o.e_avg)
        var f_ms = o.fms * ms_k
        var kd = (Float32(1.0) - m) * (Float32(1.0) - p.transmission) * SOTA_MAT_INV_PI
        var f_d = Vec3(
            p.albedo.x * kd * (Float32(1.0) - o.es_v.x) * (Float32(1.0) - es_l.x) / max(Float32(1e-4), Float32(1.0) - o.es_avg.x),
            p.albedo.y * kd * (Float32(1.0) - o.es_v.y) * (Float32(1.0) - es_l.y) / max(Float32(1e-4), Float32(1.0) - o.es_avg.y),
            p.albedo.z * kd * (Float32(1.0) - o.es_v.z) * (Float32(1.0) - es_l.z) / max(Float32(1e-4), Float32(1.0) - o.es_avg.z),
        )
        var f_base = f_ss + f_ms + f_d
        var f_sh = p.sheen * sota_mat_sheen_dv(ndoth, ndotv, nl, p.sheen_roughness)
        var ac = max(SOTA_MAT_ALPHA_MIN, p.clearcoat_roughness * p.clearcoat_roughness)
        var d_cc = sota_mat_d_aniso(ac, ac, Float32(0.0), sqrt(max(Float32(0.0), Float32(1.0) - ndoth * ndoth)), ndoth)
        var vis_cc = Float32(0.5) / max(
            Float32(1e-12),
            nl * sqrt(ac * ac * (Float32(1.0) - ndotv * ndotv) + ndotv * ndotv)
            + ndotv * sqrt(ac * ac * (Float32(1.0) - nl * nl) + nl * nl),
        )
        var f_cc = p.clearcoat * d_cc * vis_cc * (Float32(0.04) + Float32(0.96) * _pow5(Float32(1.0) - vh))
        var total = _vs(f_cc) + (f_sh + f_base * o.att_sh) * o.att_cc
        return total * nl
    elif nl < Float32(0.0) and p.transmission > Float32(0.0) and m < Float32(1.0):
        var lm = l - n * (Float32(2.0) * nl)
        var mul = -nl
        var h = (v + lm).normalized()
        var ndoth = max(Float32(0.0), n.dot(h))
        var d = sota_mat_d_aniso(o.at, o.ab, h.dot(fr.t), h.dot(fr.b), ndoth)
        var vis = sota_mat_vis_aniso(o.at, o.ab, vt, vb, ndotv, lm.dot(fr.t), lm.dot(fr.b), mul)
        var k = (Float32(1.0) - m) * p.transmission * d * vis / max(Float32(1e-4), o.e_v) * o.att_cc * o.att_sh
        return _vm(p.tint, Vec3(Float32(1.0) - o.es_v.x, Float32(1.0) - o.es_v.y, Float32(1.0) - o.es_v.z)) * (k * mul)
    return Vec3(0.0, 0.0, 0.0)


def sota_mat_direct_cpu(p: SotaMatParams, lut: SotaMatLut, n: Vec3, t_in: Vec3, b_in: Vec3, v: Vec3, l: Vec3) -> Vec3:
    """What the GPU direct hook returns: `f |cos|` plus the legacy wrap lobe (subsurface colour words 20..22)."""
    var r = sota_mat_eval_cpu(p, lut, n, t_in, b_in, v, l)
    var ndotv = max(SOTA_MAT_MU_MIN, n.dot(v))
    var fr = sota_mat_frame_cpu(n, t_in, b_in, p.aniso_rot)
    var o = sota_mat_lobes_cpu(p, lut, ndotv, sota_mat_view_phi(v.dot(fr.t), v.dot(fr.b)))
    var wrap = max(Float32(0.0), (n.dot(l) + Float32(0.45)) / Float32(1.45)) * SOTA_MAT_INV_PI * o.att_cc * o.att_sh
    return r + p.sss * wrap


struct SotaMatAmbient(Copyable, Movable):
    """Split-sum weights: out = w_cc L_coat + w_spec L_spec + w_diff E_irr + w_trans L(-v) + w_em emission."""
    var w_cc: Float32
    var w_spec: Vec3
    var w_diff: Vec3
    var w_trans: Vec3
    var w_em: Float32

    def __init__(out self):
        self.w_cc = Float32(0.0)
        self.w_spec = Vec3(0.0, 0.0, 0.0)
        self.w_diff = Vec3(0.0, 0.0, 0.0)
        self.w_trans = Vec3(0.0, 0.0, 0.0)
        self.w_em = Float32(1.0)

    def total(self) -> Vec3:
        return _vs(self.w_cc) + self.w_spec + self.w_diff + self.w_trans


def sota_mat_ambient_cpu(p: SotaMatParams, lut: SotaMatLut, ndotv_in: Float32, phi_v: Float32 = 0.0) -> SotaMatAmbient:
    """`phi_v`: view azimuth from the rotated tangent, folded to [0, pi/2] (`sota_mat_view_phi`)."""
    var ndotv = max(SOTA_MAT_MU_MIN, ndotv_in)
    var o = sota_mat_lobes_cpu(p, lut, ndotv, phi_v)
    var r = SotaMatAmbient()
    var m = p.metallic
    var one_es = Vec3(Float32(1.0) - o.es_v.x, Float32(1.0) - o.es_v.y, Float32(1.0) - o.es_v.z)
    r.w_cc = p.clearcoat * o.ecc_v
    r.w_spec = o.es_v * (o.att_cc * o.att_sh)
    r.w_diff = (p.sheen * o.esh_v + _vm(p.albedo, one_es) * ((Float32(1.0) - m) * (Float32(1.0) - p.transmission) * o.att_sh)) * o.att_cc
    r.w_trans = _vm(p.tint, one_es) * ((Float32(1.0) - m) * p.transmission * o.att_cc * o.att_sh)
    r.w_em = o.att_cc
    return r^


# ---------------------------------------------------------------------------------------------
# SPIR-V twins
# ---------------------------------------------------------------------------------------------
@always_inline
def _e_sat(mut s: AaaSpirvModule, x: UInt32) -> UInt32:
    return s.nclamp(x, s.float(0.0), s.float(1.0))


@always_inline
def _e_lerp(mut s: AaaSpirvModule, a: UInt32, b: UInt32, t: UInt32) -> UInt32:
    return s.fma(s.fsub(b, a), t, a)


@always_inline
def _e_pow5(mut s: AaaSpirvModule, x: UInt32) -> UInt32:
    var x2 = s.fmul(x, x)
    return s.fmul(s.fmul(x2, x2), x)


def _e_powf(mut s: AaaSpirvModule, x: UInt32, y: UInt32) -> UInt32:
    """exp(y ln x) for x > 0 (0 otherwise), matching `_powf`."""
    var safe = s.nmax(x, s.float(1e-30))
    var r = s.exp2(s.fmul(y, s.log2(safe)))
    return s.select_f(s.fgt(x, s.float(0.0)), r, s.float(0.0))


def _e_lut2(mut s: AaaSpirvModule, b_mats: UInt32, table: Int, mu: UInt32, r: UInt32) -> UInt32:
    var nm1 = s.float(Float32(SOTA_MAT_LUT_N - 1))
    var fx = s.fmul(_e_sat(s, mu), nm1)
    var fy = s.fmul(_e_sat(s, r), nm1)
    var x0f = s.nmin(s.ffloor(fx), s.float(Float32(SOTA_MAT_LUT_N - 2)))
    var y0f = s.nmin(s.ffloor(fy), s.float(Float32(SOTA_MAT_LUT_N - 2)))
    var tx = s.fsub(fx, x0f)
    var ty = s.fsub(fy, y0f)
    var x0 = s.uint_from_float(x0f)
    var y0 = s.uint_from_float(y0f)
    var base = s.iadd(s.uint(UInt32(SOTA_MAT_LUT_BASE + table)), s.iadd(s.imul(y0, s.uint(UInt32(SOTA_MAT_LUT_N))), x0))
    var v00 = s.load_float(b_mats, base)
    var v10 = s.load_float(b_mats, s.iadd(base, s.uint(1)))
    var v01 = s.load_float(b_mats, s.iadd(base, s.uint(UInt32(SOTA_MAT_LUT_N))))
    var v11 = s.load_float(b_mats, s.iadd(base, s.uint(UInt32(SOTA_MAT_LUT_N + 1))))
    var top = _e_lerp(s, v00, v10, tx)
    var bot = _e_lerp(s, v01, v11, tx)
    return _e_lerp(s, top, bot, ty)


def _e_lut1(mut s: AaaSpirvModule, b_mats: UInt32, table: Int, r: UInt32) -> UInt32:
    var fy = s.fmul(_e_sat(s, r), s.float(Float32(SOTA_MAT_LUT_N - 1)))
    var y0f = s.nmin(s.ffloor(fy), s.float(Float32(SOTA_MAT_LUT_N - 2)))
    var ty = s.fsub(fy, y0f)
    var base = s.iadd(s.uint(UInt32(SOTA_MAT_LUT_BASE + table)), s.uint_from_float(y0f))
    var v0 = s.load_float(b_mats, base)
    var v1 = s.load_float(b_mats, s.iadd(base, s.uint(1)))
    return _e_lerp(s, v0, v1, ty)


def _e_axis(mut s: AaaSpirvModule, x: UInt32, n: Int) -> Tuple[UInt32, UInt32]:
    var f = s.fmul(_e_sat(s, x), s.float(Float32(n - 1)))
    var f0 = s.nmin(s.ffloor(f), s.float(Float32(n - 2)))
    return (s.uint_from_float(f0), s.fsub(f, f0))


def _e_lut4(mut s: AaaSpirvModule, b_mats: UInt32, table: Int, mu: UInt32, phi: UInt32, r: UInt32, k: UInt32) -> UInt32:
    """Quadrilinear twin of `SotaMatLut.fetch4`."""
    var am = _e_axis(s, mu, SOTA_MAT_L4_MU)
    var ap = _e_axis(s, s.fmul(phi, s.float(0.636619772)), SOTA_MAT_L4_PHI)
    var ar = _e_axis(s, r, SOTA_MAT_L4_R)
    var ak = _e_axis(s, k, SOTA_MAT_L4_K)
    var one = s.float(1.0)
    var acc = s.float(0.0)
    for corner in range(4):
        var ri = s.iadd(ar[0], s.uint(UInt32(corner & 1)))
        var ki = s.iadd(ak[0], s.uint(UInt32(corner >> 1)))
        var idx = s.iadd(
            s.imul(s.iadd(s.imul(s.iadd(s.imul(ki, s.uint(UInt32(SOTA_MAT_L4_R))), ri), s.uint(UInt32(SOTA_MAT_L4_PHI))), ap[0]), s.uint(UInt32(SOTA_MAT_L4_MU))),
            am[0],
        )
        var b = s.iadd(s.uint(UInt32(SOTA_MAT_LUT_BASE + table)), idx)
        var v00 = s.load_float(b_mats, b)
        var v10 = s.load_float(b_mats, s.iadd(b, s.uint(1)))
        var v01 = s.load_float(b_mats, s.iadd(b, s.uint(UInt32(SOTA_MAT_L4_MU))))
        var v11 = s.load_float(b_mats, s.iadd(b, s.uint(UInt32(SOTA_MAT_L4_MU + 1))))
        var top = _e_lerp(s, v00, v10, am[1])
        var bot = _e_lerp(s, v01, v11, am[1])
        var e = _e_lerp(s, top, bot, ap[1])
        var wr = ar[1] if (corner & 1) == 1 else s.fsub(one, ar[1])
        var wk = ak[1] if (corner >> 1) == 1 else s.fsub(one, ak[1])
        acc = s.fadd(acc, s.fmul(e, s.fmul(wr, wk)))
    return acc


def _e_lut_avg4(mut s: AaaSpirvModule, b_mats: UInt32, table: Int, r: UInt32, k: UInt32) -> UInt32:
    var ar = _e_axis(s, r, SOTA_MAT_L4_R)
    var ak = _e_axis(s, k, SOTA_MAT_L4_K)
    var b = s.iadd(s.uint(UInt32(SOTA_MAT_LUT_BASE + table)), s.iadd(s.imul(ak[0], s.uint(UInt32(SOTA_MAT_L4_R))), ar[0]))
    var v00 = s.load_float(b_mats, b)
    var v10 = s.load_float(b_mats, s.iadd(b, s.uint(1)))
    var v01 = s.load_float(b_mats, s.iadd(b, s.uint(UInt32(SOTA_MAT_L4_R))))
    var v11 = s.load_float(b_mats, s.iadd(b, s.uint(UInt32(SOTA_MAT_L4_R + 1))))
    var top = _e_lerp(s, v00, v10, ar[1])
    var bot = _e_lerp(s, v01, v11, ar[1])
    return _e_lerp(s, top, bot, ak[1])


def _e_spec_ab(
    mut s: AaaSpirvModule, b_mats: UInt32, is_aniso: UInt32, mu: UInt32, phi: UInt32, rough: UInt32, k: UInt32, r_eff: UInt32
) -> Tuple[UInt32, UInt32]:
    """Twin of `_spec_ab`: 4-D anisotropic tables when k > 0, the 2-D isotropic ones otherwise."""
    var av = s.fvar(s.float(0.0))
    var bv = s.fvar(s.float(0.0))
    s.begin_if(is_aniso)
    s.store(av, _e_lut4(s, b_mats, SOTA_MAT_LUT_A4, mu, phi, rough, k))
    s.store(bv, _e_lut4(s, b_mats, SOTA_MAT_LUT_B4, mu, phi, rough, k))
    s.begin_else()
    s.store(av, _e_lut2(s, b_mats, SOTA_MAT_LUT_A, mu, r_eff))
    s.store(bv, _e_lut2(s, b_mats, SOTA_MAT_LUT_B, mu, r_eff))
    s.end_if()
    return (s.load_f(av), s.load_f(bv))


def _e_spec_avg(mut s: AaaSpirvModule, b_mats: UInt32, is_aniso: UInt32, rough: UInt32, k: UInt32, r_eff: UInt32) -> Tuple[UInt32, UInt32]:
    var av = s.fvar(s.float(0.0))
    var bv = s.fvar(s.float(0.0))
    s.begin_if(is_aniso)
    s.store(av, _e_lut_avg4(s, b_mats, SOTA_MAT_LUT_AAVG4, rough, k))
    s.store(bv, _e_lut_avg4(s, b_mats, SOTA_MAT_LUT_BAVG4, rough, k))
    s.begin_else()
    s.store(av, _e_lut1(s, b_mats, SOTA_MAT_LUT_AAVG, r_eff))
    s.store(bv, _e_lut1(s, b_mats, SOTA_MAT_LUT_BAVG, r_eff))
    s.end_if()
    return (s.load_f(av), s.load_f(bv))


def _e_sheen_l(mut s: AaaSpirvModule, x: UInt32, alpha: UInt32) -> UInt32:
    var om = s.fsub(s.float(1.0), alpha)
    var t = s.fmul(om, om)
    var a = s.fma(s.float(Float32(25.3245) - Float32(21.5473)), t, s.float(21.5473))
    var b = s.fma(s.float(Float32(3.32435) - Float32(3.82987)), t, s.float(3.82987))
    var c = s.fma(s.float(Float32(0.16801) - Float32(0.19823)), t, s.float(0.19823))
    var d = s.fma(s.float(Float32(-1.27393) - Float32(-1.97760)), t, s.float(-1.97760))
    var e = s.fma(s.float(Float32(-4.85967) - Float32(-4.32054)), t, s.float(-4.32054))
    var px = _e_powf(s, x, c)
    return s.fadd(s.fdiv(a, s.fma(b, px, s.float(1.0))), s.fma(d, x, e))


def _e_exp(mut s: AaaSpirvModule, x: UInt32) -> UInt32:
    return s.exp2(s.fmul(x, s.float(1.44269504)))


def _e_sheen_lambda(mut s: AaaSpirvModule, c: UInt32, alpha: UInt32) -> UInt32:
    var lo = _e_exp(s, _e_sheen_l(s, c, alpha))
    var hi = _e_exp(s, s.fsub(s.fmul(s.float(2.0), _e_sheen_l(s, s.float(0.5), alpha)), _e_sheen_l(s, s.fsub(s.float(1.0), c), alpha)))
    return s.select_f(s.flt(c, s.float(0.5)), lo, hi)


def _e_sheen_dv(mut s: AaaSpirvModule, ndoth: UInt32, ndotv: UInt32, ndotl: UInt32, rough: UInt32) -> UInt32:
    var r = s.nmax(s.float(SOTA_MAT_SHEEN_R_MIN), rough)
    var alpha = s.fmul(r, r)
    var inv_a = s.fdiv(s.float(1.0), alpha)
    var sin2 = s.nmax(s.float(1e-5), s.fsub(s.float(1.0), s.fmul(ndoth, ndoth)))
    var d = s.fmul(s.fmul(s.fadd(s.float(2.0), inv_a), _e_powf(s, sin2, s.fmul(inv_a, s.float(0.5)))), s.float(0.15915494))
    var den = s.fmul(
        s.fadd(s.fadd(s.float(1.0), _e_sheen_lambda(s, ndotv, alpha)), _e_sheen_lambda(s, ndotl, alpha)),
        s.fmul(s.float(4.0), s.fmul(ndotv, ndotl)),
    )
    var v = s.nmin(s.float(1.0), s.fdiv(s.float(1.0), s.nmax(s.float(1e-6), den)))
    return s.fmul(d, v)


def _e_ior_to_f0(mut s: AaaSpirvModule, nt: UInt32, ni: UInt32) -> UInt32:
    var q = s.fdiv(s.fsub(nt, ni), s.fadd(nt, ni))
    return s.fmul(q, q)


def _e_schlick1(mut s: AaaSpirvModule, f0: UInt32, c: UInt32) -> UInt32:
    return s.fma(s.fsub(s.float(1.0), f0), _e_pow5(s, s.fsub(s.float(1.0), _e_sat(s, c))), f0)


def _e_sensitivity(mut s: AaaSpirvModule, opd: UInt32, sh_x: UInt32, sh_y: UInt32, sh_z: UInt32) -> Tuple[UInt32, UInt32, UInt32]:
    var phase = s.fmul(s.float(6.28318530718e-9), opd)
    var p2 = s.fmul(phase, phase)
    var kx = Float32(5.4856e-13) * sqrt(Float32(6.28318530718) * Float32(4.3278e9)) / Float32(1.0685e-7)
    var ky = Float32(4.4201e-13) * sqrt(Float32(6.28318530718) * Float32(9.3046e9)) / Float32(1.0685e-7)
    var kz = Float32(5.2481e-13) * sqrt(Float32(6.28318530718) * Float32(6.6121e9)) / Float32(1.0685e-7)
    var kx2 = Float32(9.7470e-14) * sqrt(Float32(6.28318530718) * Float32(4.5282e9)) / Float32(1.0685e-7)
    var x = s.fmul(s.fmul(s.float(kx), s.fcos(s.fma(s.float(1.6810e6), phase, sh_x))), _e_exp(s, s.fmul(p2, s.float(-4.3278e9))))
    var y = s.fmul(s.fmul(s.float(ky), s.fcos(s.fma(s.float(1.7953e6), phase, sh_y))), _e_exp(s, s.fmul(p2, s.float(-9.3046e9))))
    var z = s.fmul(s.fmul(s.float(kz), s.fcos(s.fma(s.float(2.2084e6), phase, sh_z))), _e_exp(s, s.fmul(p2, s.float(-6.6121e9))))
    x = s.fadd(x, s.fmul(s.fmul(s.float(kx2), s.fcos(s.fma(s.float(2.2399e6), phase, sh_x))), _e_exp(s, s.fmul(p2, s.float(-4.5282e9)))))
    var r = s.fadd(s.fmul(s.float(3.2404542), x), s.fadd(s.fmul(s.float(-1.5371385), y), s.fmul(s.float(-0.4985314), z)))
    var g = s.fadd(s.fmul(s.float(-0.9692660), x), s.fadd(s.fmul(s.float(1.8760108), y), s.fmul(s.float(0.0415560), z)))
    var b = s.fadd(s.fmul(s.float(0.0556434), x), s.fadd(s.fmul(s.float(-0.2040259), y), s.fmul(s.float(1.0572252), z)))
    return (r, g, b)


def emit_sota_mat_iridescence(
    mut s: AaaSpirvModule, cos1: UInt32, thickness_nm: UInt32, film_ior: UInt32, f0r: UInt32, f0g: UInt32, f0b: UInt32
) -> Tuple[UInt32, UInt32, UInt32]:
    """SPIR-V twin of `sota_mat_iridescence_cpu`."""
    var one = s.float(1.0)
    var ts = _e_sat(s, s.fdiv(thickness_nm, s.float(0.03)))
    var sm = s.fmul(s.fmul(ts, ts), s.fsub(s.float(3.0), s.fmul(s.float(2.0), ts)))
    var eta2 = s.fma(s.fsub(film_ior, one), sm, one)
    var ie = s.fdiv(one, eta2)
    var sin2 = s.fmul(s.fmul(ie, ie), s.fsub(one, s.fmul(cos1, cos1)))
    var cos2sq = s.fsub(one, sin2)
    var tir = s.flt(cos2sq, s.float(0.0))
    var cos2 = s.fsqrt(s.nmax(cos2sq, s.float(0.0)))
    var r12 = _e_schlick1(s, _e_ior_to_f0(s, eta2, one), cos1)
    var t121 = s.fsub(one, r12)
    var phi12 = s.select_f(s.flt(eta2, one), s.float(SOTA_MAT_PI), s.float(0.0))
    var phi21 = s.fsub(s.float(SOTA_MAT_PI), phi12)
    var opd = s.fmul(s.fmul(s.float(2.0), eta2), s.fmul(thickness_nm, cos2))
    var f0s = [f0r, f0g, f0b]
    var c0 = List[UInt32]()
    var cm = List[UInt32]()
    var rr = List[UInt32]()
    var ph = List[UInt32]()
    for k in range(3):
        var sf = s.fsqrt(s.nclamp(f0s[k], s.float(0.0), s.float(0.9999)))
        var bior = s.fdiv(s.fadd(one, sf), s.fsub(one, sf))
        var r23 = _e_schlick1(s, _e_ior_to_f0(s, bior, eta2), cos2)
        var phi23 = s.select_f(s.flt(bior, eta2), s.float(SOTA_MAT_PI), s.float(0.0))
        ph.append(s.fadd(phi21, phi23))
        var r123 = s.nclamp(s.fmul(r12, r23), s.float(1e-5), s.float(0.9999))
        rr.append(s.fsqrt(r123))
        var rs = s.fdiv(s.fmul(s.fmul(t121, t121), r23), s.fsub(one, r123))
        c0.append(s.fadd(r12, rs))
        cm.append(s.fsub(rs, t121))
    var acc = [c0[0], c0[1], c0[2]]
    for m in range(1, 3):
        for k in range(3):
            cm[k] = s.fmul(cm[k], rr[k])
        var fm = s.float(Float32(m))
        var sens = _e_sensitivity(s, s.fmul(fm, opd), s.fmul(fm, ph[0]), s.fmul(fm, ph[1]), s.fmul(fm, ph[2]))
        acc[0] = s.fadd(acc[0], s.fmul(s.fmul(cm[0], s.float(2.0)), sens[0]))
        acc[1] = s.fadd(acc[1], s.fmul(s.fmul(cm[1], s.float(2.0)), sens[1]))
        acc[2] = s.fadd(acc[2], s.fmul(s.fmul(cm[2], s.float(2.0)), sens[2]))
    return (
        s.select_f(tir, one, s.nmax(s.float(0.0), acc[0])),
        s.select_f(tir, one, s.nmax(s.float(0.0), acc[1])),
        s.select_f(tir, one, s.nmax(s.float(0.0), acc[2])),
    )


@fieldwise_init
struct SotaMatInputs(Copyable, Movable):
    """SSA ids the deferred pass hands to the direct hook."""
    var nx: UInt32
    var ny: UInt32
    var nz: UInt32
    var vx: UInt32
    var vy: UInt32
    var vz: UInt32
    var lx: UInt32
    var ly: UInt32
    var lz: UInt32
    var tx: UInt32
    var ty: UInt32
    var tz: UInt32
    var bx: UInt32
    var by: UInt32
    var bz: UInt32
    var alb_r: UInt32
    var alb_g: UInt32
    var alb_b: UInt32
    var metal: UInt32
    var spec: UInt32
    var rough: UInt32
    var cc: UInt32
    var cc_rough: UInt32
    var sh_r: UInt32
    var sh_g: UInt32
    var sh_b: UInt32
    var sh_rough: UInt32
    var sss_r: UInt32
    var sss_g: UInt32
    var sss_b: UInt32
    var irid_w: UInt32
    var irid_nm: UInt32
    var irid_ior: UInt32


@fieldwise_init
struct SotaMatShade(Copyable, Movable):
    """Function-scope variables written by the direct hook and read by the compose hook."""
    var on: UInt32
    var dir_r: UInt32
    var dir_g: UInt32
    var dir_b: UInt32
    var w_cc: UInt32
    var ws_r: UInt32
    var ws_g: UInt32
    var ws_b: UInt32
    var wd_r: UInt32
    var wd_g: UInt32
    var wd_b: UInt32
    var wt_r: UInt32
    var wt_g: UInt32
    var wt_b: UInt32
    var w_em: UInt32
    var r_eff: UInt32
    var aniso: UInt32
    var bx: UInt32   # rotated bitangent (the un-stretched direction), for the anisotropic bent reflection
    var by: UInt32
    var bz: UInt32


def emit_sota_mat_decode_on(mut s: AaaSpirvModule, r_flags: UInt32, e_tagged: UInt32) -> UInt32:
    return s.logical_and(s.ine(s.bit_and(r_flags, s.uint(AAA_FLAG_SOTA_MAT)), s.uint(0)), s.logical_not(e_tagged))


def emit_sota_mat_shade(
    mut s: AaaSpirvModule,
    b_mats: UInt32,
    mat_id: UInt32,
    on: UInt32,
    inp: SotaMatInputs,
    legacy_direct: Tuple[UInt32, UInt32, UInt32],
) -> SotaMatShade:
    """Direct sun response (BSDF * cos + legacy wrap) and the split-sum weights of the layered model.
    Everything runs inside `if on`; with the flag off the outputs keep the legacy direct value."""
    var out = SotaMatShade(
        on,
        s.fvar(legacy_direct[0]), s.fvar(legacy_direct[1]), s.fvar(legacy_direct[2]),
        s.fvar(s.float(0.0)),
        s.fvar(s.float(0.0)), s.fvar(s.float(0.0)), s.fvar(s.float(0.0)),
        s.fvar(s.float(0.0)), s.fvar(s.float(0.0)), s.fvar(s.float(0.0)),
        s.fvar(s.float(0.0)), s.fvar(s.float(0.0)), s.fvar(s.float(0.0)),
        s.fvar(s.float(1.0)),
        s.fvar(s.float(0.5)),
        s.fvar(s.float(0.0)),
        s.fvar(s.float(0.0)), s.fvar(s.float(0.0)), s.fvar(s.float(1.0)),
    )
    s.begin_if(on)
    var one = s.float(1.0)
    var zero = s.float(0.0)
    # Extension words
    var eb = s.iadd(s.uint(UInt32(SOTA_MAT_EXT_BASE)), s.imul(mat_id, s.uint(UInt32(SOTA_MAT_EXT_WORDS))))
    var has_ext = s.ieq(s.load_word(b_mats, eb), s.uint(SOTA_MAT_MAGIC))
    var aniso = s.select_f(has_ext, s.load_float(b_mats, s.iadd(eb, s.uint(1))), zero)
    var rot = s.select_f(has_ext, s.load_float(b_mats, s.iadd(eb, s.uint(2))), zero)
    var trans = s.select_f(has_ext, s.load_float(b_mats, s.iadd(eb, s.uint(3))), zero)
    var tint_r = s.select_f(has_ext, s.load_float(b_mats, s.iadd(eb, s.uint(4))), zero)
    var tint_g = s.select_f(has_ext, s.load_float(b_mats, s.iadd(eb, s.uint(5))), zero)
    var tint_b = s.select_f(has_ext, s.load_float(b_mats, s.iadd(eb, s.uint(6))), zero)
    var irid_w = s.nmax(zero, inp.irid_w)

    # Frame: Gram-Schmidt tangent, bitangent with the mesh handedness, then rotate
    var nx = inp.nx
    var ny = inp.ny
    var nz = inp.nz
    var ndt = emit_vec3_dot(s, nx, ny, nz, inp.tx, inp.ty, inp.tz)
    var t0 = emit_vec3_normalize(s, s.fsub(inp.tx, s.fmul(nx, ndt)), s.fsub(inp.ty, s.fmul(ny, ndt)), s.fsub(inp.tz, s.fmul(nz, ndt)))
    var b0 = emit_vec3_cross(s, nx, ny, nz, t0[0], t0[1], t0[2])
    var hand = s.select_f(s.flt(emit_vec3_dot(s, b0[0], b0[1], b0[2], inp.bx, inp.by, inp.bz), zero), s.float(-1.0), one)
    var b0x = s.fmul(b0[0], hand)
    var b0y = s.fmul(b0[1], hand)
    var b0z = s.fmul(b0[2], hand)
    var cr = s.fcos(rot)
    var sr = s.fsin(rot)
    var ftx = s.fma(t0[0], cr, s.fmul(b0x, sr))
    var fty = s.fma(t0[1], cr, s.fmul(b0y, sr))
    var ftz = s.fma(t0[2], cr, s.fmul(b0z, sr))
    var fbx = s.fsub(s.fmul(b0x, cr), s.fmul(t0[0], sr))
    var fby = s.fsub(s.fmul(b0y, cr), s.fmul(t0[1], sr))
    var fbz = s.fsub(s.fmul(b0z, cr), s.fmul(t0[2], sr))

    var vx = inp.vx
    var vy = inp.vy
    var vz = inp.vz
    var lx = inp.lx
    var ly = inp.ly
    var lz = inp.lz
    var ndotv = s.nmax(s.float(SOTA_MAT_MU_MIN), emit_vec3_dot(s, nx, ny, nz, vx, vy, vz))
    var vt = emit_vec3_dot(s, vx, vy, vz, ftx, fty, ftz)
    var vb = emit_vec3_dot(s, vx, vy, vz, fbx, fby, fbz)
    var phi_v = s.fatan2(s.fabs(vb), s.fabs(vt))
    var is_aniso = s.fgt(aniso, zero)

    # Lobes (`sota_mat_lobes_cpu`)
    var m = inp.metal
    var om_m = s.fsub(one, m)
    var diel = s.fmul(s.float(0.04), inp.spec)
    var f0r = s.fma(diel, om_m, s.fmul(inp.alb_r, m))
    var f0g = s.fma(diel, om_m, s.fmul(inp.alb_g, m))
    var f0b = s.fma(diel, om_m, s.fmul(inp.alb_b, m))
    var f90 = s.fma(inp.spec, om_m, m)
    var alpha = s.nmax(s.float(SOTA_MAT_ALPHA_MIN), s.fmul(inp.rough, inp.rough))
    var at = s.nmax(s.float(SOTA_MAT_ALPHA_MIN), s.fma(s.fsub(one, alpha), s.fmul(aniso, aniso), alpha))
    var ab = alpha
    var r_eff = s.fsqrt(s.fsqrt(s.fmul(at, ab)))
    var ab_v = _e_spec_ab(s, b_mats, is_aniso, ndotv, phi_v, inp.rough, aniso, r_eff)
    var a_v = ab_v[0]
    var b_v = ab_v[1]
    var e_v = s.fadd(a_v, b_v)
    var avg_ab = _e_spec_avg(s, b_mats, is_aniso, inp.rough, aniso, r_eff)
    var a_avg = avg_ab[0]
    var b_avg = avg_ab[1]
    var e_avg = s.fadd(a_avg, b_avg)
    var fiv_r = s.fvar(zero)
    var fiv_g = s.fvar(zero)
    var fiv_b = s.fvar(zero)
    var has_irid = s.fgt(irid_w, zero)
    s.begin_if(has_irid)
    var fiv = emit_sota_mat_iridescence(s, ndotv, inp.irid_nm, inp.irid_ior, f0r, f0g, f0b)
    s.store(fiv_r, fiv[0])
    s.store(fiv_g, fiv[1])
    s.store(fiv_b, fiv[2])
    s.end_if()
    var fi = [s.load_f(fiv_r), s.load_f(fiv_g), s.load_f(fiv_b)]
    var f0 = [f0r, f0g, f0b]
    var om_eavg = s.fsub(one, e_avg)
    var inv21 = s.float(Float32(1.0 / 21.0))
    var fms = List[UInt32]()
    var es_v = List[UInt32]()
    var es_avg = List[UInt32]()
    for k in range(3):
        var favg = _e_lerp(s, s.fma(s.fsub(f90, f0[k]), inv21, f0[k]), fi[k], irid_w)
        var fm = s.fdiv(s.fmul(s.fmul(favg, favg), e_avg), s.nmax(s.float(1e-4), s.fsub(one, s.fmul(favg, om_eavg))))
        fms.append(fm)
        var ef_v = _e_lerp(s, s.fma(f0[k], a_v, s.fmul(f90, b_v)), s.fmul(fi[k], e_v), irid_w)
        es_v.append(s.fma(fm, s.fsub(one, e_v), ef_v))
        var ef_avg = _e_lerp(s, s.fma(f0[k], a_avg, s.fmul(f90, b_avg)), s.fmul(fi[k], e_avg), irid_w)
        es_avg.append(s.fma(fm, om_eavg, ef_avg))
    var rc = inp.cc_rough
    var ecc_v = s.fma(s.float(0.04), _e_lut2(s, b_mats, SOTA_MAT_LUT_A, ndotv, rc), _e_lut2(s, b_mats, SOTA_MAT_LUT_B, ndotv, rc))
    var esh_v = _e_lut2(s, b_mats, SOTA_MAT_LUT_SH, ndotv, inp.sh_rough)
    var att_cc = s.fsub(one, s.fmul(inp.cc, ecc_v))
    var smax = s.nmax(inp.sh_r, s.nmax(inp.sh_g, inp.sh_b))
    var att_sh = s.fsub(one, s.fmul(smax, esh_v))
    var att = s.fmul(att_cc, att_sh)

    # Ambient weights (`sota_mat_ambient_cpu`)
    var alb = [inp.alb_r, inp.alb_g, inp.alb_b]
    var shc = [inp.sh_r, inp.sh_g, inp.sh_b]
    var tint = [tint_r, tint_g, tint_b]
    var kd_amb = s.fmul(s.fmul(om_m, s.fsub(one, trans)), att_sh)
    var kt_amb = s.fmul(s.fmul(om_m, trans), att)
    var ws_out = [out.ws_r, out.ws_g, out.ws_b]
    var wd_out = [out.wd_r, out.wd_g, out.wd_b]
    var wt_out = [out.wt_r, out.wt_g, out.wt_b]
    for k in range(3):
        var one_es = s.fsub(one, es_v[k])
        s.store(ws_out[k], s.fmul(es_v[k], att))
        s.store(wd_out[k], s.fmul(s.fma(shc[k], esh_v, s.fmul(s.fmul(alb[k], one_es), kd_amb)), att_cc))
        s.store(wt_out[k], s.fmul(s.fmul(tint[k], one_es), kt_amb))
    s.store(out.w_cc, s.fmul(inp.cc, ecc_v))
    s.store(out.w_em, att_cc)
    s.store(out.r_eff, r_eff)
    s.store(out.aniso, aniso)
    s.store(out.bx, fbx)
    s.store(out.by, fby)
    s.store(out.bz, fbz)

    # Direct: front side (`sota_mat_eval_cpu`, nl > 0)
    var nl_raw = emit_vec3_dot(s, nx, ny, nz, lx, ly, lz)
    var front = s.fgt(nl_raw, zero)
    var back = s.logical_and(s.flt(nl_raw, zero), s.logical_and(s.fgt(trans, zero), s.flt(m, one)))
    var res_r = s.fvar(zero)
    var res_g = s.fvar(zero)
    var res_b = s.fvar(zero)
    s.begin_if(front)
    var nl = nl_raw
    var h = emit_vec3_normalize(s, s.fadd(vx, lx), s.fadd(vy, ly), s.fadd(vz, lz))
    var vh = s.nmax(zero, emit_vec3_dot(s, vx, vy, vz, h[0], h[1], h[2]))
    var ndoth = s.nmax(zero, emit_vec3_dot(s, nx, ny, nz, h[0], h[1], h[2]))
    var d = _e_d_aniso(s, at, ab, emit_vec3_dot(s, h[0], h[1], h[2], ftx, fty, ftz), emit_vec3_dot(s, h[0], h[1], h[2], fbx, fby, fbz), ndoth)
    var vis = _e_vis_aniso(
        s, at, ab, vt, vb, ndotv,
        emit_vec3_dot(s, lx, ly, lz, ftx, fty, ftz), emit_vec3_dot(s, lx, ly, lz, fbx, fby, fbz), nl,
    )
    var fc = _e_pow5(s, s.fsub(one, _e_sat(s, vh)))
    var fs = List[UInt32]()
    for k in range(3):
        fs.append(s.fma(s.fsub(f90, f0[k]), fc, f0[k]))
    var fh_r = s.fvar(fs[0])
    var fh_g = s.fvar(fs[1])
    var fh_b = s.fvar(fs[2])
    s.begin_if(has_irid)
    var fih = emit_sota_mat_iridescence(s, vh, inp.irid_nm, inp.irid_ior, f0r, f0g, f0b)
    s.store(fh_r, _e_lerp(s, fs[0], fih[0], irid_w))
    s.store(fh_g, _e_lerp(s, fs[1], fih[1], irid_w))
    s.store(fh_b, _e_lerp(s, fs[2], fih[2], irid_w))
    s.end_if()
    var fh = [s.load_f(fh_r), s.load_f(fh_g), s.load_f(fh_b)]
    var dvis = s.fmul(d, vis)
    var phi_l = s.fatan2(
        s.fabs(emit_vec3_dot(s, lx, ly, lz, fbx, fby, fbz)), s.fabs(emit_vec3_dot(s, lx, ly, lz, ftx, fty, ftz))
    )
    var ab_l = _e_spec_ab(s, b_mats, is_aniso, nl, phi_l, inp.rough, aniso, r_eff)
    var a_l = ab_l[0]
    var b_l = ab_l[1]
    var e_l = s.fadd(a_l, b_l)
    var ms_k = s.fdiv(
        s.fmul(s.fmul(s.fsub(one, e_v), s.fsub(one, e_l)), s.float(SOTA_MAT_INV_PI)),
        s.nmax(s.float(1e-4), s.fsub(one, e_avg)),
    )
    var kd = s.fmul(s.fmul(om_m, s.fsub(one, trans)), s.float(SOTA_MAT_INV_PI))
    var sheen_dv = _e_sheen_dv(s, ndoth, ndotv, nl, inp.sh_rough)
    var ac = s.nmax(s.float(SOTA_MAT_ALPHA_MIN), s.fmul(rc, rc))
    var sin_h = s.fsqrt(s.nmax(zero, s.fsub(one, s.fmul(ndoth, ndoth))))
    var d_cc = _e_d_aniso(s, ac, ac, zero, sin_h, ndoth)
    var ac2 = s.fmul(ac, ac)
    var vis_cc = s.fdiv(
        s.float(0.5),
        s.nmax(
            s.float(1e-12),
            s.fadd(
                s.fmul(nl, s.fsqrt(s.fma(ac2, s.fsub(one, s.fmul(ndotv, ndotv)), s.fmul(ndotv, ndotv)))),
                s.fmul(ndotv, s.fsqrt(s.fma(ac2, s.fsub(one, s.fmul(nl, nl)), s.fmul(nl, nl)))),
            ),
        ),
    )
    var f_cc = s.fmul(s.fmul(inp.cc, s.fmul(d_cc, vis_cc)), s.fma(s.float(0.96), _e_pow5(s, s.fsub(one, vh)), s.float(0.04)))
    var res_out = [res_r, res_g, res_b]
    for k in range(3):
        var ef_l = _e_lerp(s, s.fma(f0[k], a_l, s.fmul(f90, b_l)), s.fmul(fi[k], e_l), irid_w)
        var es_l = s.fma(fms[k], s.fsub(one, e_l), ef_l)
        var f_ss = s.fmul(fh[k], dvis)
        var f_ms = s.fmul(fms[k], ms_k)
        var f_d = s.fdiv(
            s.fmul(s.fmul(s.fmul(alb[k], kd), s.fsub(one, es_v[k])), s.fsub(one, es_l)),
            s.nmax(s.float(1e-4), s.fsub(one, es_avg[k])),
        )
        var f_base = s.fadd(s.fadd(f_ss, f_ms), f_d)
        var f_sh = s.fmul(shc[k], sheen_dv)
        var tot = s.fma(s.fma(f_base, att_sh, f_sh), att_cc, f_cc)
        s.store(res_out[k], s.fmul(tot, nl))
    s.end_if()
    # Direct: thin-walled transmission (light from behind)
    s.begin_if(back)
    var mul = s.fneg(nl_raw)
    var two_nl = s.fmul(s.float(2.0), nl_raw)
    var lmx = s.fsub(lx, s.fmul(nx, two_nl))
    var lmy = s.fsub(ly, s.fmul(ny, two_nl))
    var lmz = s.fsub(lz, s.fmul(nz, two_nl))
    var hm = emit_vec3_normalize(s, s.fadd(vx, lmx), s.fadd(vy, lmy), s.fadd(vz, lmz))
    var ndoth_m = s.nmax(zero, emit_vec3_dot(s, nx, ny, nz, hm[0], hm[1], hm[2]))
    var d_m = _e_d_aniso(s, at, ab, emit_vec3_dot(s, hm[0], hm[1], hm[2], ftx, fty, ftz), emit_vec3_dot(s, hm[0], hm[1], hm[2], fbx, fby, fbz), ndoth_m)
    var vis_m = _e_vis_aniso(
        s, at, ab, vt, vb, ndotv,
        emit_vec3_dot(s, lmx, lmy, lmz, ftx, fty, ftz), emit_vec3_dot(s, lmx, lmy, lmz, fbx, fby, fbz), mul,
    )
    var kt = s.fmul(s.fdiv(s.fmul(s.fmul(om_m, trans), s.fmul(d_m, vis_m)), s.nmax(s.float(1e-4), e_v)), att)
    for k in range(3):
        s.store(res_out[k], s.fmul(s.fmul(tint[k], s.fsub(one, es_v[k])), s.fmul(kt, mul)))
    s.end_if()
    # Legacy wrap lobe (subsurface colour), under the coat and sheen
    var wrap = s.fmul(s.fmul(s.nmax(zero, s.fdiv(s.fadd(nl_raw, s.float(0.45)), s.float(1.45))), s.float(SOTA_MAT_INV_PI)), att)
    var sss = [inp.sss_r, inp.sss_g, inp.sss_b]
    var dir_out = [out.dir_r, out.dir_g, out.dir_b]
    for k in range(3):
        s.store(dir_out[k], s.fma(sss[k], wrap, s.load_f(res_out[k])))
    s.end_if()
    return out^


def _e_d_aniso(mut s: AaaSpirvModule, at: UInt32, ab: UInt32, ht: UInt32, hb: UInt32, hn: UInt32) -> UInt32:
    var q = s.fadd(s.fadd(s.fdiv(s.fmul(ht, ht), s.fmul(at, at)), s.fdiv(s.fmul(hb, hb), s.fmul(ab, ab))), s.fmul(hn, hn))
    return s.fdiv(s.float(1.0), s.nmax(s.float(1e-20), s.fmul(s.fmul(s.float(SOTA_MAT_PI), s.fmul(at, ab)), s.fmul(q, q))))


def _e_vis_aniso(
    mut s: AaaSpirvModule, at: UInt32, ab: UInt32, vt: UInt32, vb: UInt32, vn: UInt32, lt: UInt32, lb: UInt32, ln: UInt32
) -> UInt32:
    var at2 = s.fmul(at, at)
    var ab2 = s.fmul(ab, ab)
    var sv = s.fsqrt(s.fadd(s.fadd(s.fmul(at2, s.fmul(vt, vt)), s.fmul(ab2, s.fmul(vb, vb))), s.fmul(vn, vn)))
    var sl = s.fsqrt(s.fadd(s.fadd(s.fmul(at2, s.fmul(lt, lt)), s.fmul(ab2, s.fmul(lb, lb))), s.fmul(ln, ln)))
    return s.fdiv(s.float(0.5), s.nmax(s.float(1e-12), s.fadd(s.fmul(ln, sv), s.fmul(vn, sl))))


@fieldwise_init
struct SotaMatComposeIn(Copyable, Movable):
    """SSA ids / variables of the legacy composition that the compose hook reads or overwrites."""
    var nx: UInt32
    var ny: UInt32
    var nz: UInt32
    var vx: UInt32
    var vy: UInt32
    var vz: UInt32
    var rough: UInt32
    var refl_r: UInt32     # fvar: substrate reflection radiance (cubemap or RT)
    var refl_g: UInt32
    var refl_b: UInt32
    var cc_r: UInt32       # fvar: coat reflection radiance
    var cc_g: UInt32
    var cc_b: UInt32
    var env_sp: UInt32     # the legacy cubemap sample (vec4) along R at the base roughness
    var env_occ: UInt32
    var ibl_int: UInt32
    var max_mip: UInt32
    var is_furnace: UInt32
    var dir_r: UInt32      # fvar: direct light (sun + clustered)
    var dir_g: UInt32
    var dir_b: UInt32
    var em_r: UInt32
    var em_g: UInt32
    var em_b: UInt32
    var ao_r: UInt32       # multi-bounce diffuse AO
    var ao_g: UInt32
    var ao_b: UInt32
    var demod_spec: UInt32
    var crisp_r: UInt32    # fvars overwritten when on
    var crisp_g: UInt32
    var crisp_b: UInt32
    var dw_r: UInt32
    var dw_g: UInt32
    var dw_b: UInt32
    var mod_r: UInt32
    var mod_g: UInt32
    var mod_b: UInt32


def emit_sota_mat_compose(
    mut s: AaaSpirvModule,
    sh: SotaMatShade,
    c: SotaMatComposeIn,
    env_cube: Tuple[UInt32, UInt32],
    demod: Tuple[UInt32, UInt32, UInt32],
) -> Tuple[UInt32, UInt32, UInt32]:
    """Split-sum composition of the layered model; overwrites the legacy crisp / diffuse-weight / spec-modulation
    variables when on and returns the demodulated specular (legacy values when off)."""
    var dm_r = s.fvar(demod[0])
    var dm_g = s.fvar(demod[1])
    var dm_b = s.fvar(demod[2])
    s.begin_if(sh.on)
    var one = s.float(1.0)
    var zero = s.float(0.0)
    var r_eff = s.load_f(sh.r_eff)
    var aniso = s.load_f(sh.aniso)
    # Anisotropic bent reflection (KHR sample-viewer form) replaces the cubemap part of the base reflection
    var ref_r = s.fvar(s.load_f(c.refl_r))
    var ref_g = s.fvar(s.load_f(c.refl_g))
    var ref_b = s.fvar(s.load_f(c.refl_b))
    s.begin_if(s.logical_and(s.fgt(aniso, s.float(1e-3)), s.logical_not(c.is_furnace)))
    var bx = s.load_f(sh.bx)
    var by = s.load_f(sh.by)
    var bz = s.load_f(sh.bz)
    var c1 = emit_vec3_cross(s, bx, by, bz, c.vx, c.vy, c.vz)
    var bent = emit_vec3_normalize(s, c1[0], c1[1], c1[2])
    var bent2 = emit_vec3_cross(s, bent[0], bent[1], bent[2], bx, by, bz)
    var bn0 = emit_vec3_normalize(s, bent2[0], bent2[1], bent2[2])
    var q = s.fsub(one, s.fmul(aniso, s.fsub(one, c.rough)))
    var q2 = s.fmul(q, q)
    var a_mix = s.fmul(q2, q2)
    var bn = emit_vec3_normalize(s, _e_lerp(s, bn0[0], c.nx, a_mix), _e_lerp(s, bn0[1], c.ny, a_mix), _e_lerp(s, bn0[2], c.nz, a_mix))
    var two_bv = s.fmul(s.float(2.0), emit_vec3_dot(s, bn[0], bn[1], bn[2], c.vx, c.vy, c.vz))
    var env_a = s.sample_cube_lod(
        env_cube,
        s.vec3(s.fsub(s.fmul(bn[0], two_bv), c.vx), s.fsub(s.fmul(bn[1], two_bv), c.vy), s.fsub(s.fmul(bn[2], two_bv), c.vz)),
        s.fmul(r_eff, c.max_mip),
    )
    var kk = s.fmul(c.ibl_int, c.env_occ)
    s.store(ref_r, s.fma(s.fsub(s.extract_f(env_a, 0), s.extract_f(c.env_sp, 0)), kk, s.load_f(ref_r)))
    s.store(ref_g, s.fma(s.fsub(s.extract_f(env_a, 1), s.extract_f(c.env_sp, 1)), kk, s.load_f(ref_g)))
    s.store(ref_b, s.fma(s.fsub(s.extract_f(env_a, 2), s.extract_f(c.env_sp, 2)), kk, s.load_f(ref_b)))
    s.end_if()
    # Thin-walled transmission sees the environment straight through (-v)
    var tr_rad_r = s.fvar(one)
    var tr_rad_g = s.fvar(one)
    var tr_rad_b = s.fvar(one)
    var wt = [s.load_f(sh.wt_r), s.load_f(sh.wt_g), s.load_f(sh.wt_b)]
    var has_t = s.fgt(s.nmax(wt[0], s.nmax(wt[1], wt[2])), zero)
    s.begin_if(s.logical_and(has_t, s.logical_not(c.is_furnace)))
    var env_t = s.sample_cube_lod(env_cube, s.vec3(s.fneg(c.vx), s.fneg(c.vy), s.fneg(c.vz)), s.fmul(r_eff, c.max_mip))
    s.store(tr_rad_r, s.fmul(s.extract_f(env_t, 0), c.ibl_int))
    s.store(tr_rad_g, s.fmul(s.extract_f(env_t, 1), c.ibl_int))
    s.store(tr_rad_b, s.fmul(s.extract_f(env_t, 2), c.ibl_int))
    s.end_if()
    var w_cc = s.load_f(sh.w_cc)
    var w_em = s.load_f(sh.w_em)
    var ws = [s.load_f(sh.ws_r), s.load_f(sh.ws_g), s.load_f(sh.ws_b)]
    var wd = [s.load_f(sh.wd_r), s.load_f(sh.wd_g), s.load_f(sh.wd_b)]
    var refl = [s.load_f(ref_r), s.load_f(ref_g), s.load_f(ref_b)]
    var ccr = [s.load_f(c.cc_r), s.load_f(c.cc_g), s.load_f(c.cc_b)]
    var dirs = [s.load_f(c.dir_r), s.load_f(c.dir_g), s.load_f(c.dir_b)]
    var ems = [c.em_r, c.em_g, c.em_b]
    var aos = [c.ao_r, c.ao_g, c.ao_b]
    var trr = [s.load_f(tr_rad_r), s.load_f(tr_rad_g), s.load_f(tr_rad_b)]
    var crisp = [c.crisp_r, c.crisp_g, c.crisp_b]
    var dws = [c.dw_r, c.dw_g, c.dw_b]
    var mods = [c.mod_r, c.mod_g, c.mod_b]
    var dms = [dm_r, dm_g, dm_b]
    for k in range(3):
        var spec_out = s.fma(w_cc, ccr[k], s.fmul(ws[k], refl[k]))
        var modk = s.nmax(s.float(1e-4), s.fadd(w_cc, ws[k]))
        var non_spec = s.fadd(s.fma(ems[k], w_em, dirs[k]), s.fmul(wt[k], trr[k]))
        s.store(crisp[k], s.select_f(c.demod_spec, non_spec, s.fadd(non_spec, spec_out)))
        s.store(dws[k], s.fmul(wd[k], aos[k]))
        s.store(mods[k], modk)
        s.store(dms[k], s.fdiv(spec_out, modk))
    s.end_if()
    return (s.load_f(dm_r), s.load_f(dm_g), s.load_f(dm_b))


# ---------------------------------------------------------------------------------------------
# Renderer-side state
# ---------------------------------------------------------------------------------------------
struct SotaMatState(Copyable, Movable):
    """cvar `r.sota_mat.layered` (`enabled`, default off; env `FLINT_SOTA_MAT=1` turns it on). The LUT block is
    baked on the CPU and written into `buf_materials` the first frame the model is enabled."""
    var enabled: Bool
    var lut_uploaded: Bool
    var bake_s: Float64

    def __init__(out self):
        self.enabled = env_value("FLINT_SOTA_MAT") == "1"
        self.lut_uploaded = False
        self.bake_s = Float64(0.0)

    def prepare(mut self, cpu_address: Int):
        if self.lut_uploaded or cpu_address == 0:
            return
        var t0 = get_time()
        var lut = SotaMatLut.bake()
        lut.write_to(cpu_address)
        self.bake_s = get_time() - t0
        self.lut_uploaded = True
