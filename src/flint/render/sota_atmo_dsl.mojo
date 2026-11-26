"""Area H (sota:atmo): shared ShaderDsl helpers (medium, LUT fetches, phase functions, segment march).

Used by the sky / aerial-perspective passes (`sota_atmo_gpu`), the volumetric clouds (`sota_atmo_clouds`), the
night sky (`sota_atmo_night`) and the GPU IBL bake (`sota_atmo_ibl`). Each helper mirrors a CPU function of
`sota_atmo_core`. Constants per frame: `b_globals[SOTA_ATMO_G .. +64)` and `[SOTA_ATMO_G2 .. +64)`.
"""
from flint.render.shader_dsl import (
    BufferRef,
    GpuBool,
    GpuF32,
    GpuU32,
    GpuVec3,
    ShaderDsl,
    abs_f,
    clamp_f,
    exp2_f,
    floor_f,
    max_f,
    min_f,
    min_u,
    saturate,
    select_f,
    sqrt_f,
)
from flint.render.sota_atmo_core import (
    SOTA_ATMO_AP_DIM,
    SOTA_ATMO_MS_DIM,
    SOTA_ATMO_SKY_H,
    SOTA_ATMO_SKY_W,
    SOTA_ATMO_TRANS_H,
    SOTA_ATMO_TRANS_W,
)

comptime SOTA_ATMO_G: Int = 3840  # b_globals words [3840, 3904) (free in the v5 layout; E uses 3072..3099)
comptime SOTA_ATMO_G2: Int = 3904  # b_globals words [3904, 3968): round-3 block (night sky, clouds, reprojection)
comptime SOTA_ATMO_FLAG_SKY: UInt32 = 1
comptime SOTA_ATMO_FLAG_AP: UInt32 = 2
comptime SOTA_ATMO_FLAG_FOG: UInt32 = 4
comptime SOTA_ATMO_FLAG_SUN_DISC: UInt32 = 8
comptime SOTA_ATMO_FLAG_MOON_DISC: UInt32 = 16

comptime _LOG2E: Float32 = 1.44269504
comptime _PI: Float32 = 3.14159265358979


# ============================================================================
# DSL helpers (each mirrors a CPU function of sota_atmo_core)
# ============================================================================

@always_inline
def _gexp(x: GpuF32) -> GpuF32:
    return exp2_f(x * _LOG2E)


@always_inline
def _gexp3(v: GpuVec3) -> GpuVec3:
    return GpuVec3(_gexp(v.x()), _gexp(v.y()), _gexp(v.z()))


def _gatan2(mut dsl: ShaderDsl, y: GpuF32, x: GpuF32) -> GpuF32:
    return GpuF32(dsl.mod_addr, dsl.raw_module()[].fatan2(y.id, x.id))


def _gacos(mut dsl: ShaderDsl, c: GpuF32) -> GpuF32:
    var cc = clamp_f(c, -1.0, 1.0)
    return _gatan2(dsl, sqrt_f(max_f(1.0 - cc * cc, 0.0)), cc)


struct _Atmo(Copyable, Movable):
    var ray: GpuVec3
    var h_r: GpuF32
    var mie_s: GpuVec3
    var h_m: GpuF32
    var mie_e: GpuVec3
    var g: GpuF32
    var oz: GpuVec3
    var oz_c: GpuF32
    var oz_hw: GpuF32
    var rb: GpuF32
    var rt: GpuF32
    var kpu: GpuF32
    var albedo: GpuVec3
    var r_cam: GpuF32
    var sun_d: GpuVec3
    var sun_ar: GpuF32
    var sun_e: GpuVec3
    var ap_max: GpuF32
    var moon_d: GpuVec3
    var moon_ar: GpuF32
    var moon_e: GpuVec3
    var moon_alb: GpuF32

    def __init__(out self, mut dsl: ShaderDsl, b: BufferRef[1]):
        var w = SOTA_ATMO_G
        self.ray = b.load_vec3(dsl.u32(UInt32(w + 0)))
        self.h_r = b.load_f32(dsl.u32(UInt32(w + 3)))
        self.mie_s = b.load_vec3(dsl.u32(UInt32(w + 4)))
        self.h_m = b.load_f32(dsl.u32(UInt32(w + 7)))
        self.mie_e = b.load_vec3(dsl.u32(UInt32(w + 8)))
        self.g = b.load_f32(dsl.u32(UInt32(w + 11)))
        self.oz = b.load_vec3(dsl.u32(UInt32(w + 12)))
        self.oz_c = b.load_f32(dsl.u32(UInt32(w + 15)))
        self.oz_hw = b.load_f32(dsl.u32(UInt32(w + 16)))
        self.rb = b.load_f32(dsl.u32(UInt32(w + 17)))
        self.rt = b.load_f32(dsl.u32(UInt32(w + 18)))
        self.kpu = b.load_f32(dsl.u32(UInt32(w + 19)))
        self.albedo = b.load_vec3(dsl.u32(UInt32(w + 20)))
        self.r_cam = b.load_f32(dsl.u32(UInt32(w + 23)))
        self.sun_d = b.load_vec3(dsl.u32(UInt32(w + 24)))
        self.sun_ar = b.load_f32(dsl.u32(UInt32(w + 27)))
        self.sun_e = b.load_vec3(dsl.u32(UInt32(w + 28)))
        self.ap_max = b.load_f32(dsl.u32(UInt32(w + 31)))
        self.moon_d = b.load_vec3(dsl.u32(UInt32(w + 32)))
        self.moon_ar = b.load_f32(dsl.u32(UInt32(w + 35)))
        self.moon_e = b.load_vec3(dsl.u32(UInt32(w + 36)))
        self.moon_alb = b.load_f32(dsl.u32(UInt32(w + 39)))


struct _Medium(Copyable, Movable):
    var sig_r: GpuVec3
    var sig_m: GpuVec3
    var sig_t: GpuVec3

    def __init__(out self, a: _Atmo, h: GpuF32):
        var hc = max_f(h, 0.0)
        var rr = _gexp(-(hc / a.h_r))
        var rm = _gexp(-(hc / a.h_m))
        var ro = max_f(1.0 - abs_f(hc - a.oz_c) / a.oz_hw, 0.0)
        self.sig_r = a.ray * rr
        self.sig_m = a.mie_s * rm
        self.sig_t = self.sig_r + a.mie_e * rm + a.oz * ro


@always_inline
def _top_dist(r: GpuF32, mu: GpuF32, rt: GpuF32) -> GpuF32:
    var disc = r * r * (mu * mu - 1.0) + rt * rt
    return max_f(sqrt_f(max_f(disc, 0.0)) - r * mu, 0.0)


@always_inline
def _ground_dist(r: GpuF32, mu: GpuF32, rb: GpuF32) -> GpuF32:
    """Distance to the ground or -1."""
    var disc = r * r * (mu * mu - 1.0) + rb * rb
    var d = -(r * mu) - sqrt_f(max_f(disc, 0.0))
    return select_f(mu.lt(0.0) & disc.ge(0.0) & d.gt(0.0), d, d * 0.0 - 1.0)


@always_inline
def _horizon_mu(r: GpuF32, rb: GpuF32) -> GpuF32:
    var s = min_f(rb / max_f(r, rb), 1.0)
    return -sqrt_f(max_f(1.0 - s * s, 0.0))


@always_inline
def _sun_vis(r: GpuF32, mu: GpuF32, a: _Atmo) -> GpuF32:
    var mh = _horizon_mu(r, a.rb)
    var w = a.sun_ar * sqrt_f(max_f(1.0 - mh * mh, 0.0))
    return saturate((mu - mh + w) / max_f(w * 2.0, 1e-6))


@always_inline
def _rayleigh(mu: GpuF32) -> GpuF32:
    return (mu * mu + 1.0) * Float32(3.0 / (16.0 * 3.14159265358979))


@always_inline
def _cs(mu: GpuF32, g: GpuF32) -> GpuF32:
    var g2 = g * g
    var num = (1.0 - g2) * (mu * mu + 1.0) * Float32(3.0 / (8.0 * 3.14159265358979))
    var base = max_f(g2 + 1.0 - g * mu * 2.0, 1e-4)
    return num / ((g2 + 2.0) * base * sqrt_f(base))


def _bilerp_buf(
    mut dsl: ShaderDsl, b: BufferRef[1], off: GpuU32, w: Int, h: Int, fx_in: GpuF32, fy_in: GpuF32, wrap: Bool
) -> GpuVec3:
    """Bilinear fetch of a vec3 LUT stored as 4 words per texel (texel centres at integer coordinates)."""
    var fw = Float32(w)
    var fx: GpuF32
    if wrap:
        fx = fx_in - floor_f(fx_in / fw) * fw
    else:
        fx = clamp_f(fx_in, 0.0, Float32(w - 1))
    var fy = clamp_f(fy_in, 0.0, Float32(h - 1))
    var wm1 = dsl.u32(UInt32(w - 1))
    var x0 = min_u(floor_f(fx).saturate_u32(), wm1)
    var y0 = min_u(floor_f(fy).saturate_u32(), dsl.u32(UInt32(h - 1)))
    var x1: GpuU32
    if wrap:
        x1 = (x0 + UInt32(1)) % UInt32(w)
    else:
        x1 = min_u(x0 + UInt32(1), wm1)
    var y1 = min_u(y0 + UInt32(1), dsl.u32(UInt32(h - 1)))
    var tx = saturate(fx - x0.to_f32())
    var ty = saturate(fy - y0.to_f32())
    var c00 = b.load_vec3(off + (y0 * UInt32(w) + x0) * UInt32(4))
    var c10 = b.load_vec3(off + (y0 * UInt32(w) + x1) * UInt32(4))
    var c01 = b.load_vec3(off + (y1 * UInt32(w) + x0) * UInt32(4))
    var c11 = b.load_vec3(off + (y1 * UInt32(w) + x1) * UInt32(4))
    var one_tx = 1.0 - tx
    var top = c00 * one_tx + c10 * tx
    var bot = c01 * one_tx + c11 * tx
    return top * (1.0 - ty) + bot * ty


def _sample_trans(mut dsl: ShaderDsl, b: BufferRef[1], off: GpuU32, a: _Atmo, r_in: GpuF32, mu_in: GpuF32) -> GpuVec3:
    var r = clamp_f(r_in, a.rb, a.rt)
    var mu = max_f(mu_in, _horizon_mu(r, a.rb))
    var hh = sqrt_f(a.rt * a.rt - a.rb * a.rb)
    var rho = sqrt_f(max_f(r * r - a.rb * a.rb, 0.0))
    var d = _top_dist(r, mu, a.rt)
    var d_min = a.rt - r
    var d_max = rho + hh
    var x_mu = saturate((d - d_min) / max_f(d_max - d_min, 1e-6))
    var x_r = saturate(rho / hh)
    return _bilerp_buf(dsl, b, off, SOTA_ATMO_TRANS_W, SOTA_ATMO_TRANS_H, x_mu * Float32(SOTA_ATMO_TRANS_W - 1), x_r * Float32(SOTA_ATMO_TRANS_H - 1), False)


def _sample_ms(mut dsl: ShaderDsl, b: BufferRef[1], off: GpuU32, a: _Atmo, r: GpuF32, mu_s: GpuF32) -> GpuVec3:
    var u = saturate(mu_s * 0.5 + 0.5)
    var v = saturate((r - a.rb) / (a.rt - a.rb))
    var n = Float32(SOTA_ATMO_MS_DIM)
    return _bilerp_buf(dsl, b, off, SOTA_ATMO_MS_DIM, SOTA_ATMO_MS_DIM, u * n - 0.5, v * n - 0.5, False)


def _light_term(
    mut dsl: ShaderDsl, b: BufferRef[1], t_off: GpuU32, m_off: GpuU32, a: _Atmo,
    ri: GpuF32, mu_l: GpuF32, cos_vl: GpuF32, med: _Medium, e: GpuVec3,
) -> GpuVec3:
    var t_l = _sample_trans(dsl, b, t_off, a, ri, mu_l) * _sun_vis(ri, mu_l, a)
    var ph = med.sig_r * _rayleigh(cos_vl) + med.sig_m * _cs(cos_vl, a.g)
    var ms = _sample_ms(dsl, b, m_off, a, ri, mu_l)
    return (ph * t_l + (med.sig_r + med.sig_m) * ms) * e


def _step_integral(src: GpuVec3, sig_t: GpuVec3, tr_step: GpuVec3) -> GpuVec3:
    return GpuVec3(
        src.x() * (1.0 - tr_step.x()) / max_f(sig_t.x(), 1e-9),
        src.y() * (1.0 - tr_step.y()) / max_f(sig_t.y(), 1e-9),
        src.z() * (1.0 - tr_step.z()) / max_f(sig_t.z(), 1e-9),
    )


def _march_segment(
    mut dsl: ShaderDsl, b: BufferRef[1], t_off: GpuU32, m_off: GpuU32, a: _Atmo,
    r0: GpuF32, d: GpuVec3, t_max: GpuF32, steps: Int,
) -> Tuple[GpuVec3, GpuVec3]:
    """GPU twin of `integrate_segment` (sun + moon, quadratic steps)."""
    var l_v = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var thr_v = dsl.var_vec3(dsl.vec3_const(1.0, 1.0, 1.0))
    var cvs = d.dot(a.sun_d)
    var cvm = d.dot(a.moon_d)
    var has_moon = (a.moon_e.x() + a.moon_e.y() + a.moon_e.z()).gt(0.0)
    var i_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var i = i_v.get()
    dsl.break_if(i.ge(UInt32(steps)))
    var u0 = i.to_f32() * (1.0 / Float32(steps))
    var u1 = (i + UInt32(1)).to_f32() * (1.0 / Float32(steps))
    var t0 = t_max * u0 * u0
    var t1 = t_max * u1 * u1
    var t = (t0 + t1) * 0.5
    var dt = t1 - t0
    var px = d.x() * t
    var py = r0 + d.y() * t
    var pz = d.z() * t
    var ri = sqrt_f(px * px + py * py + pz * pz)
    var med = _Medium(a, ri - a.rb)
    var mu_s = (px * a.sun_d.x() + py * a.sun_d.y() + pz * a.sun_d.z()) / ri
    var src_v = dsl.var_vec3(_light_term(dsl, b, t_off, m_off, a, ri, mu_s, cvs, med, a.sun_e))
    dsl.if_then(has_moon)
    var mu_m = (px * a.moon_d.x() + py * a.moon_d.y() + pz * a.moon_d.z()) / ri
    src_v.set(src_v.get() + _light_term(dsl, b, t_off, m_off, a, ri, mu_m, cvm, med, a.moon_e))
    dsl.end_if()
    var tr_step = _gexp3(med.sig_t * (-dt))
    var thr = thr_v.get()
    l_v.set(l_v.get() + thr * _step_integral(src_v.get(), med.sig_t, tr_step))
    thr_v.set(thr * tr_step)
    i_v.set(i + UInt32(1))
    dsl.end_loop()
    return (l_v.get(), thr_v.get())


# ============================================================================
# Aerial-perspective fetch and analytic height fog
# ============================================================================

def _ap_fetch(mut dsl: ShaderDsl, b: BufferRef[1], ap_off: GpuU32, fx: GpuF32, fy: GpuF32, s: GpuU32) -> Tuple[GpuVec3, GpuVec3]:
    comptime N = SOTA_ATMO_AP_DIM
    var cx = clamp_f(fx, 0.0, Float32(N - 1))
    var cy = clamp_f(fy, 0.0, Float32(N - 1))
    var nm1 = dsl.u32(UInt32(N - 1))
    var x0 = min_u(floor_f(cx).saturate_u32(), nm1)
    var y0 = min_u(floor_f(cy).saturate_u32(), nm1)
    var x1 = min_u(x0 + UInt32(1), nm1)
    var y1 = min_u(y0 + UInt32(1), nm1)
    var tx = saturate(cx - x0.to_f32())
    var ty = saturate(cy - y0.to_f32())
    var base = ap_off + s * UInt32(N * N * 8)
    var i00 = base + (y0 * UInt32(N) + x0) * UInt32(8)
    var i10 = base + (y0 * UInt32(N) + x1) * UInt32(8)
    var i01 = base + (y1 * UInt32(N) + x0) * UInt32(8)
    var i11 = base + (y1 * UInt32(N) + x1) * UInt32(8)
    var w00 = (1.0 - tx) * (1.0 - ty)
    var w10 = tx * (1.0 - ty)
    var w01 = (1.0 - tx) * ty
    var w11 = tx * ty
    var sc = b.load_vec3(i00) * w00 + b.load_vec3(i10) * w10 + b.load_vec3(i01) * w01 + b.load_vec3(i11) * w11
    var tr = b.load_vec3(i00 + UInt32(4)) * w00 + b.load_vec3(i10 + UInt32(4)) * w10 + b.load_vec3(i01 + UInt32(4)) * w01 + b.load_vec3(i11 + UInt32(4)) * w11
    return (sc, tr)


def _hf_piece(dens: GpuF32, k: GpuF32, base_y: GpuF32, cam_y: GpuF32, dir_y: GpuF32, a: GpuF32, b: GpuF32) -> GpuF32:
    var ym = cam_y + dir_y * ((a + b) * 0.5)
    var e0 = _gexp(-(k * (cam_y + dir_y * a - base_y)))
    var kd = k * dir_y
    var len = max_f(b - a, 0.0)
    var small = abs_f(kd * len).lt(1e-4)
    var safe_kd = select_f(small, kd * 0.0 + 1.0, kd)
    var expo = select_f(small, dens * e0 * len, dens * e0 * (1.0 - _gexp(-(kd * len))) / safe_kd)
    return select_f(ym.gt(base_y), expo, dens * len)


def _height_fog_od(dens: GpuF32, k_in: GpuF32, base_y: GpuF32, cam_y: GpuF32, dir_y: GpuF32, t0: GpuF32, t1: GpuF32) -> GpuF32:
    """GPU twin of `height_fog_optical_depth` (split at the crossing of the base height)."""
    var k = max_f(k_in, 1e-6)
    var tc = (base_y - cam_y) / select_f(abs_f(dir_y).gt(1e-6), dir_y, dir_y * 0.0 + 1e-6)
    var split = tc.gt(t0) & t1.gt(tc) & abs_f(dir_y).gt(1e-6)
    var mid = select_f(split, tc, t0)
    var od = _hf_piece(dens, k, base_y, cam_y, dir_y, t0, mid) + _hf_piece(dens, k, base_y, cam_y, dir_y, mid, t1)
    return select_f(t1.gt(t0), od, dens * 0.0)



def _sky_view_radiance(mut dsl: ShaderDsl, b: BufferRef[1], s_off: GpuU32, a: _Atmo, d: GpuVec3) -> GpuVec3:
    """Sky-view LUT radiance along the unit world direction `d` (`sample_sky_view`)."""
    var half_pi = Float32(1.5707963268)
    var el_h = -_gacos(dsl, min_f(a.rb / a.r_cam, 1.0))
    var horiz = sqrt_f(max_f(d.x() * d.x() + d.z() * d.z(), 0.0))
    var el = _gatan2(dsl, d.y(), horiz)
    var v_up = 0.5 - sqrt_f(saturate((el - el_h) / (half_pi - el_h))) * 0.5
    var v_dn = sqrt_f(saturate((el_h - el) / (el_h + half_pi))) * 0.5 + 0.5
    var v = select_f(el.ge(el_h), v_up, v_dn)
    var u = _gatan2(dsl, d.z(), d.x()) * Float32(0.159154943) + 0.5
    return _bilerp_buf(dsl, b, s_off, SOTA_ATMO_SKY_W, SOTA_ATMO_SKY_H, u * Float32(SOTA_ATMO_SKY_W) - 0.5, v * Float32(SOTA_ATMO_SKY_H) - 0.5, True)
