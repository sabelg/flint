"""Milestone 3 (`V14`): Split-Sum `RG16F` BRDF LUT, `BC6H` Prefiltered GGX Cubemap & `SH9` Irradiance.

Implements:
  1. Hammersley 128-sample Split-Sum GGX BRDF Integration LUT (`VK_FORMAT_R16G16_SFLOAT`, `64x64`),
     computing exact `(A, B)` directional albedo integrals for Fdez-Aguera / Kulla-Conty
     multi-scatter energy compensation (`1.0000` White-Furnace conservation).
  2. Vulkan 1.3 Section 16.5.4 compliant 6-face Importance-Sampled GGX Prefiltered Environment
     Cubemap encoded directly into `VK_FORMAT_BC6H_UFLOAT` mip chains (`roughness = 0.0 .. 1.0`).
  3. Ramamoorthi-Hanrahan Order-2 `9`-Coefficient Spherical Harmonics (`SH9`) diffuse irradiance
     projector & SPIR-V evaluator (normalized so uniform `1.0` sky yields exact `1.0000` irradiance).
"""
from std.math import cos, max, min, sin, sqrt
from flint.core.math import Vec3
from flint.assets.bc_compress import encode_bc6h_hdr, f32_to_uf16_bits
from flint.assets.hdr_image import HdrImage
from flint.render.spirv import AaaSpirvModule


@always_inline
def _radical_inverse_vdc(bits_in: UInt32) -> Float32:
    var b = bits_in
    b = (b << 16) | (b >> 16)
    b = ((b & 0x55555555) << 1) | ((b & 0xAAAAAAAA) >> 1)
    b = ((b & 0x33333333) << 2) | ((b & 0xCCCCCCCC) >> 2)
    b = ((b & 0x0F0F0F0F) << 4) | ((b & 0xF0F0F0F0) >> 4)
    b = ((b & 0x00FF00FF) << 8) | ((b & 0xFF00FF00) >> 8)
    return Float32(b) * Float32(2.3283064365386963e-10)


@always_inline
def _importance_sample_ggx(u1: Float32, u2: Float32, roughness: Float32, n: Vec3) -> Vec3:
    var a = max(Float32(0.002), roughness * roughness)
    var phi = Float32(6.28318530718) * u1
    var cos_theta = sqrt((Float32(1.0) - u2) / max(Float32(1e-6), Float32(1.0) + (a * a - Float32(1.0)) * u2))
    var sin_theta = sqrt(max(Float32(0.0), Float32(1.0) - cos_theta * cos_theta))
    var hx = cos(phi) * sin_theta
    var hy = sin(phi) * sin_theta
    var hz = cos_theta

    var up = Vec3(1.0, 0.0, 0.0) if abs(n.z) > Float32(0.999) else Vec3(0.0, 0.0, 1.0)
    var tx = up.cross(n).normalized()
    var ty = n.cross(tx)
    return (tx * hx + ty * hy + n * hz).normalized()


def bake_split_sum_brdf_lut_rg16f(lut_dim: Int = 64, num_samples: Int = 128) -> List[UInt8]:
    """Bakes a `lut_dim x lut_dim` `VK_FORMAT_R16G16_SFLOAT` (`4` bytes/texel) Split-Sum BRDF LUT."""
    var out = List[UInt8](capacity=lut_dim * lut_dim * 4)
    var n = Vec3(0.0, 0.0, 1.0)

    for y in range(lut_dim):
        var roughness = (Float32(y) + Float32(0.5)) / Float32(lut_dim)
        var a = max(Float32(0.002), roughness * roughness)
        var a2 = a * a
        for x in range(lut_dim):
            var ndotv = max(Float32(1e-3), (Float32(x) + Float32(0.5)) / Float32(lut_dim))
            var vx = sqrt(max(Float32(0.0), Float32(1.0) - ndotv * ndotv))
            var v = Vec3(vx, Float32(0.0), ndotv)

            var acc_a = Float32(0.0)
            var acc_b = Float32(0.0)
            for i in range(num_samples):
                var u1 = Float32(i) / Float32(num_samples)
                var u2 = _radical_inverse_vdc(UInt32(i))
                var h = _importance_sample_ggx(u1, u2, roughness, n)
                var vdoth = max(Float32(0.0), v.dot(h))
                var l = (h * (Float32(2.0) * vdoth) - v).normalized()
                var ndotl = max(Float32(0.0), l.z)
                var ndoth = max(Float32(1e-5), h.z)

                if ndotl > Float32(0.0):
                    # Exact Height-Correlated Smith G2 importance-sampling visibility weight:
                    # G2(V, L) = 2 * NdotL * NdotV / (NdotV * sqrt(a2 + (1-a2)*NdotL^2) + NdotL * sqrt(a2 + (1-a2)*NdotV^2))
                    # Weight = G2 * VdotH / (NdotH * NdotV)
                    var gv = ndotl * sqrt(a2 + (Float32(1.0) - a2) * ndotv * ndotv)
                    var gl = ndotv * sqrt(a2 + (Float32(1.0) - a2) * ndotl * ndotl)
                    var g2 = (Float32(2.0) * ndotl * ndotv) / max(Float32(1e-6), gv + gl)
                    var g_vis = (g2 * vdoth) / max(Float32(1e-6), ndoth * ndotv)
                    var om = Float32(1.0) - vdoth
                    var fc = om * om * om * om * om
                    acc_a += (Float32(1.0) - fc) * g_vis
                    acc_b += fc * g_vis

            acc_a /= Float32(num_samples)
            acc_b /= Float32(num_samples)
            var ha = f32_to_uf16_bits(min(Float32(1.0), acc_a))
            var hb = f32_to_uf16_bits(min(Float32(1.0), acc_b))
            out.append(UInt8(ha & 0xFF))
            out.append(UInt8((ha >> 8) & 0xFF))
            out.append(UInt8(hb & 0xFF))
            out.append(UInt8((hb >> 8) & 0xFF))

    return out^


@always_inline
def cube_face_direction(face: Int, u_01: Float32, v_01: Float32) -> Vec3:
    """Vulkan Spec 16.5.4 exact cube-map face `(face 0..5, u, v)` -> unit world direction."""
    var s = u_01 * Float32(2.0) - Float32(1.0)
    var t = v_01 * Float32(2.0) - Float32(1.0)
    if face == 0:    # +X
        return Vec3(Float32(1.0), -t, -s).normalized()
    elif face == 1:  # -X
        return Vec3(Float32(-1.0), -t, s).normalized()
    elif face == 2:  # +Y
        return Vec3(s, Float32(1.0), t).normalized()
    elif face == 3:  # -Y
        return Vec3(s, Float32(-1.0), -t).normalized()
    elif face == 4:  # +Z
        return Vec3(s, -t, Float32(1.0)).normalized()
    else:            # -Z
        return Vec3(-s, -t, Float32(-1.0)).normalized()


struct SkyDomeParams(Copyable, Movable):
    var zenith_rgb: Vec3
    var horizon_rgb: Vec3
    var ground_rgb: Vec3
    var sun_dir: Vec3
    var sun_lobe_rgb: Vec3
    var uniform_furnace: Bool

    def __init__(out self, uniform_furnace: Bool = False):
        self.zenith_rgb = Vec3(0.24, 0.44, 0.82)
        self.horizon_rgb = Vec3(0.76, 0.70, 0.62)
        self.ground_rgb = Vec3(0.16, 0.14, 0.12)
        self.sun_dir = Vec3(0.35, 0.78, -0.52).normalized()
        self.sun_lobe_rgb = Vec3(6.5, 5.8, 4.8)
        self.uniform_furnace = uniform_furnace

    @always_inline
    def eval(self, d: Vec3) -> Vec3:
        if self.uniform_furnace:
            return Vec3(1.0, 1.0, 1.0)
        var up_t = max(Float32(0.0), d.y)
        var dn_t = max(Float32(0.0), -d.y)
        var sky = self.horizon_rgb * (Float32(1.0) - up_t) + self.zenith_rgb * up_t
        if d.y < Float32(0.0):
            var g_blend = min(Float32(1.0), dn_t * Float32(3.0))
            sky = self.horizon_rgb * (Float32(1.0) - g_blend) + self.ground_rgb * g_blend
        var sdot = max(Float32(0.0), d.dot(self.sun_dir))
        var s2 = sdot * sdot
        var s8 = s2 * s2 * s2 * s2
        var s32 = s8 * s8 * s8 * s8
        return sky + self.sun_lobe_rgb * s32


struct BakedIblEnvironment(Movable):
    """Holds `BC6H_UFLOAT` 6-face prefiltered GGX cubemap mip buffers + `SH9` diffuse irradiance."""
    var base_dim: Int
    var mip_levels: Int
    var mip_face_bytes: List[List[UInt8]]  # length = mip_levels * 6 (mip * 6 + face)
    var sh9: List[Vec3]                    # 9 Vec3 coefficients for normalized irradiance E(N)/pi

    def __init__(out self):
        self.base_dim = 0
        self.mip_levels = 0
        self.mip_face_bytes = List[List[UInt8]]()
        self.sh9 = List[Vec3](length=9, fill=Vec3(0.0, 0.0, 0.0))


def bake_bc6h_ibl_cubemap_and_sh9(
    sky: SkyDomeParams,
    base_dim: Int = 64,
    mip_levels: Int = 5,
    ggx_samples: Int = 32,
) -> BakedIblEnvironment:
    """Prefilters an HDR sky environment into a `BC6H_UFLOAT` GGX cubemap mip chain + exact `SH9` irradiance."""
    var res = BakedIblEnvironment()
    res.base_dim = base_dim
    res.mip_levels = mip_levels

    # 1. Project diffuse irradiance / pi onto Order-2 Spherical Harmonics (`SH9`) over all 6 cube faces
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
                var rad = sky.eval(dir)

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

    # Normalize solid angle to exact 4*pi and pre-multiply Ramamoorthi-Hanrahan cosine lobe A_l / pi
    # AND basis constants so shader evaluation is pure polynomials!
    var norm_4pi = Float32(12.5663706) / max(Float32(1e-6), weight_sum)
    for k in range(9):
        sh_raw[k] = sh_raw[k] * norm_4pi

    # Pre-scale by (A_l / pi) * Y_lm_const:
    # l=0: (pi / pi) * 0.282095 = 0.282095 -> note for uniform L=1, sh_raw[0] = 4*pi * 0.282095 = 3.5449077,
    #      and 3.5449077 * 0.282095 = 1.000000!
    # l=1: (2/3) * 0.488603 = 0.325735
    # l=2: (1/4) * {1.092548, 1.092548, 0.315392, 1.092548, 0.546274}
    res.sh9[0] = sh_raw[0] * Float32(0.28209479)
    res.sh9[1] = sh_raw[1] * Float32(0.32573500)
    res.sh9[2] = sh_raw[2] * Float32(0.32573500)
    res.sh9[3] = sh_raw[3] * Float32(0.32573500)
    res.sh9[4] = sh_raw[4] * Float32(0.27313711)
    res.sh9[5] = sh_raw[5] * Float32(0.27313711)
    res.sh9[6] = sh_raw[6] * Float32(0.07884789)
    res.sh9[7] = sh_raw[7] * Float32(0.27313711)
    res.sh9[8] = sh_raw[8] * Float32(0.13656855)

    # 2. Bake 6-face GGX prefiltered mip chain and block-compress every face/mip to `BC6H_UFLOAT`!
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
                    if mip == 0 or sky.uniform_furnace:
                        face_hdr[y * m_dim + x] = sky.eval(r_dir)
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
                                pref = pref + sky.eval(l) * ndotl
                                w_tot += ndotl
                        face_hdr[y * m_dim + x] = pref * (Float32(1.0) / max(Float32(1e-5), w_tot))
            res.mip_face_bytes.append(encode_bc6h_hdr(face_hdr, m_dim, m_dim))

    return res^


def bake_hdr_image_ibl_cubemap_and_sh9(
    hdr_img: HdrImage,
    base_dim: Int = 64,
    mip_levels: Int = 5,
    ggx_samples: Int = 32,
) -> BakedIblEnvironment:
    """Projects an equirectangular `.hdr`/`.exr` `HdrImage` panorama directly into a `BC6H_UFLOAT`
    6-face GGX prefiltered cubemap mip chain + exact Order-2 `SH9` diffuse irradiance!"""
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
                var rad = hdr_img.sample_equirect(dir)

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
                        face_hdr[y * m_dim + x] = hdr_img.sample_equirect(r_dir)
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
                                pref = pref + hdr_img.sample_equirect(l) * ndotl
                                w_tot += ndotl
                        face_hdr[y * m_dim + x] = pref * (Float32(1.0) / max(Float32(1e-5), w_tot))
            res.mip_face_bytes.append(encode_bc6h_hdr(face_hdr, m_dim, m_dim))

    return res^


def eval_sh9_cpu(sh9: List[Vec3], n: Vec3) -> Vec3:
    var out = sh9[0]
    out = out + sh9[1] * n.y + sh9[2] * n.z + sh9[3] * n.x
    out = out + sh9[4] * (n.x * n.y) + sh9[5] * (n.y * n.z)
    out = out + sh9[6] * (Float32(3.0) * n.z * n.z - Float32(1.0))
    out = out + sh9[7] * (n.x * n.z) + sh9[8] * (n.x * n.x - n.y * n.y)
    return Vec3(max(Float32(0.0), out.x), max(Float32(0.0), out.y), max(Float32(0.0), out.z))


def emit_eval_sh9_irradiance(
    mut s: AaaSpirvModule,
    b_globals: UInt32,
    sh_word_base: Int,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Evaluates pre-scaled `SH9` diffuse irradiance `E(N) / pi` in SPIR-V."""
    var b1 = ny
    var b2 = nz
    var b3 = nx
    var b4 = s.fmul(nx, ny)
    var b5 = s.fmul(ny, nz)
    var b6 = s.fma(s.float(3.0), s.fmul(nz, nz), s.float(-1.0))
    var b7 = s.fmul(nx, nz)
    var b8 = s.fsub(s.fmul(nx, nx), s.fmul(ny, ny))

    var basis = [s.float(1.0), b1, b2, b3, b4, b5, b6, b7, b8]
    var er = s.fvar(s.float(0.0))
    var eg = s.fvar(s.float(0.0))
    var eb = s.fvar(s.float(0.0))
    for k in range(9):
        var w_off = UInt32(sh_word_base + k * 3)
        var cr = s.load_float(b_globals, s.uint(w_off))
        var cg = s.load_float(b_globals, s.uint(w_off + 1))
        var cb = s.load_float(b_globals, s.uint(w_off + 2))
        var bk = basis[k]
        s.store(er, s.fma(cr, bk, s.load_f(er)))
        s.store(eg, s.fma(cg, bk, s.load_f(eg)))
        s.store(eb, s.fma(cb, bk, s.load_f(eb)))

    return (
        s.nmax(s.float(0.0), s.load_f(er)),
        s.nmax(s.float(0.0), s.load_f(eg)),
        s.nmax(s.float(0.0), s.load_f(eb)),
    )


def sample_heitz_bounded_vndf_cpu(
    u1: Float32,
    u2: Float32,
    roughness: Float32,
    v_local: Vec3,
) -> Tuple[Vec3, Float32]:
    """Samples visible normal H from Heitz 2018 Bounded VNDF distribution in local tangent space.
    Returns: (microfacet_normal_H, smith_g2_over_g1_weight).
    """
    var alpha = max(Float32(0.04), min(Float32(0.55), roughness * roughness))
    var v_h = Vec3(alpha * v_local.x, alpha * v_local.y, max(Float32(1e-4), v_local.z)).normalized()

    var lensq = v_h.x * v_h.x + v_h.y * v_h.y
    var t1 = Vec3(1.0, 0.0, 0.0)
    if lensq > Float32(1e-6):
        var inv_len = Float32(1.0) / sqrt(lensq)
        t1 = Vec3(-v_h.y * inv_len, v_h.x * inv_len, Float32(0.0))
    var t2 = v_h.cross(t1)

    var r = sqrt(u1)
    var phi = Float32(6.28318530718) * u2
    var p1 = r * cos(phi)
    var p2 = r * sin(phi)
    var s = Float32(0.5) * (Float32(1.0) + v_h.z)
    p2 = (Float32(1.0) - s) * sqrt(max(Float32(0.0), Float32(1.0) - p1 * p1)) + s * p2

    var p3 = sqrt(max(Float32(0.0), Float32(1.0) - p1 * p1 - p2 * p2))
    var n_h = (t1 * p1 + t2 * p2 + v_h * p3)
    var h = Vec3(alpha * n_h.x, alpha * n_h.y, max(Float32(0.0), n_h.z)).normalized()

    var vdoth = max(Float32(0.0), v_local.dot(h))
    var l = (h * (Float32(2.0) * vdoth) - v_local).normalized()

    var ndotv = max(Float32(1e-4), v_local.z)
    var ndotl = l.z
    if ndotl <= Float32(0.0):
        return (h, Float32(0.0))

    var a2 = alpha * alpha
    var g1_v = Float32(2.0) * ndotv / (ndotv + sqrt(a2 + (Float32(1.0) - a2) * ndotv * ndotv))
    var g2 = Float32(2.0) * ndotv * ndotl / (
        ndotl * sqrt(a2 + (Float32(1.0) - a2) * ndotv * ndotv) +
        ndotv * sqrt(a2 + (Float32(1.0) - a2) * ndotl * ndotl)
    )
    var weight = g2 / max(Float32(1e-4), g1_v)
    return (h, weight)

