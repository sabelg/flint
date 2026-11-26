"""Area H (sota:atmo): physically based sky, aerial perspective and height fog on the GPU.

Passes (compute, `ShaderDsl` -> SPIR-V, AAA descriptor set; LUTs live in an appended `BIND_AAA_HISTORY` region):

  1. `sota_atmo_transmittance_shader` 256 x 64   transmittance to the top of the atmosphere (Bruneton mapping)
  2. `sota_atmo_multiscatter_shader`  32 x 32    Hillaire 2020 multiple-scattering LUT (64 directions x 20 steps)
     (1 and 2 depend only on the medium; they run on the first frame and whenever `params` change)
  3. `sota_atmo_skyview_shader`       192 x 108  sky radiance for the camera altitude, sun + moon, every frame
  4. `sota_atmo_aerial_shader`        32 x 32 x 32 aerial-perspective froxels (in-scatter + RGB transmittance)
     along the camera frustum, quadratic slices up to `ap_max_km`, every frame
  5. `sota_atmo_apply_shader`         render res, right before the froxel composite:
       sky pixels    -> sky-view LUT + limb-darkened sun disc + Lommel-Seeliger moon disc (phase from geometry)
       surface px    -> `c * T + S` from the aerial-perspective volume
       both          -> analytic exponential height fog from the froxel far plane (or the eye) to the surface /
                        `fog_tail_max_dist`, lit by the primary light (Cornette-Shanks) and the SH9 sky; the froxel
                        volume then composites the near range on top, so the two fogs join without a seam.

CPU twins and references: `sota_atmo_core.mojo`. Constants per frame: `b_globals[SOTA_ATMO_G .. +64)`.
With `enabled = False` nothing here runs and the v5 sky / aerial perspective path is untouched.
"""
from std.collections import List
from std.ffi import external_call
from std.math import acos, cos, max, min, sin, sqrt
from std.memory import bitcast
from flint.core.math import Vec3
from flint.platform.vulkan import (
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    PIPELINE_BOTTOM,
    PIPELINE_COMPUTE,
    VulkanGpu,
    put32,
    put_float,
)
from flint.assets.bc_compress import encode_bc6h_hdr
from flint.render.aaa_ibl import BakedIblEnvironment, _importance_sample_ggx, _radical_inverse_vdc, cube_face_direction, emit_eval_sh9_irradiance
from flint.render.aaa_visbuffer import AAA_PUSH_WORDS, BIND_AAA_GLOBALS, BIND_AAA_HDR_OUT, BIND_AAA_HISTORY
from flint.render.spirv import MODEL_GL_COMPUTE
from flint.render.shader_dsl import (
    BufferRef,
    GpuBool,
    GpuF32,
    GpuU32,
    GpuVec3,
    ShaderDsl,
    abs_f,
    clamp_f,
    cos_f,
    exp2_f,
    floor_f,
    max_f,
    min_f,
    min_u,
    saturate,
    select_f,
    select_u,
    select_vec3,
    sin_f,
    sqrt_f,
)
from flint.render.sota_atmo_core import (
    RGB,
    SOTA_ATMO_AP_DIM,
    SOTA_ATMO_AP_SUBSTEPS,
    SOTA_ATMO_MS_DIM,
    SOTA_ATMO_MS_DIRS,
    SOTA_ATMO_MS_STEPS,
    SOTA_ATMO_SKY_H,
    SOTA_ATMO_SKY_STEPS,
    SOTA_ATMO_SKY_W,
    SOTA_ATMO_TRANS_H,
    SOTA_ATMO_TRANS_STEPS,
    SOTA_ATMO_TRANS_W,
    SotaAtmoLights,
    SotaAtmoLuts,
    SotaAtmoParams,
    SotaTimeOfDay,
    bake_sky_view_lut,
    moon_irradiance,
    rgb_luma,
    sample_sky_view,
    sample_transmittance,
    sun_visibility,
)

comptime SOTA_ATMO_G: Int = 3840  # b_globals words [3840, 3904) (free in the v5 layout; E uses 3072..3099)
comptime SOTA_ATMO_FLAG_SKY: UInt32 = 1
comptime SOTA_ATMO_FLAG_AP: UInt32 = 2
comptime SOTA_ATMO_FLAG_FOG: UInt32 = 4
comptime SOTA_ATMO_FLAG_SUN_DISC: UInt32 = 8
comptime SOTA_ATMO_FLAG_MOON_DISC: UInt32 = 16

comptime _LOG2E: Float32 = 1.44269504
comptime _PI: Float32 = 3.14159265358979


# ============================================================================
# Region layout (words in BIND_AAA_HISTORY, appended past every other history region)
# ============================================================================

@always_inline
def sota_atmo_trans_words() -> Int:
    return SOTA_ATMO_TRANS_W * SOTA_ATMO_TRANS_H * 4


@always_inline
def sota_atmo_ms_words() -> Int:
    return SOTA_ATMO_MS_DIM * SOTA_ATMO_MS_DIM * 4


@always_inline
def sota_atmo_sky_words() -> Int:
    return SOTA_ATMO_SKY_W * SOTA_ATMO_SKY_H * 4


@always_inline
def sota_atmo_ap_words() -> Int:
    return SOTA_ATMO_AP_DIM * SOTA_ATMO_AP_DIM * SOTA_ATMO_AP_DIM * 8


@always_inline
def sota_atmo_region_words() -> Int:
    return sota_atmo_trans_words() + sota_atmo_ms_words() + sota_atmo_sky_words() + sota_atmo_ap_words()


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
# 1. Transmittance LUT
# ============================================================================

def sota_atmo_transmittance_shader() -> List[UInt32]:
    """Push: 0 trans_off."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_glob = dsl.buffer[1](BIND_AAA_GLOBALS)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var x = dsl.global_id_x()
    var y = dsl.global_id_y()
    var t_off = dsl.push_u32(0)
    dsl.return_if(x.ge(UInt32(SOTA_ATMO_TRANS_W)) | y.ge(UInt32(SOTA_ATMO_TRANS_H)))
    var a = _Atmo(dsl, b_glob)
    var x_mu = x.to_f32() * (1.0 / Float32(SOTA_ATMO_TRANS_W - 1))
    var x_r = y.to_f32() * (1.0 / Float32(SOTA_ATMO_TRANS_H - 1))
    var hh = sqrt_f(a.rt * a.rt - a.rb * a.rb)
    var rho = hh * x_r
    var r = sqrt_f(rho * rho + a.rb * a.rb)
    var d_min = a.rt - r
    var d_max = rho + hh
    var d = d_min + x_mu * (d_max - d_min)
    var mu_raw = (hh * hh - rho * rho - d * d) / (r * max_f(d, 1e-4) * 2.0)
    var mu = select_f(d.gt(1e-4), clamp_f(mu_raw, -1.0, 1.0), dsl.f32(1.0))
    var t_max = _top_dist(r, mu, a.rt)
    var tau_v = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var i_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var i = i_v.get()
    dsl.break_if(i.ge(UInt32(SOTA_ATMO_TRANS_STEPS)))
    var u0 = i.to_f32() * (1.0 / Float32(SOTA_ATMO_TRANS_STEPS))
    var u1 = (i + UInt32(1)).to_f32() * (1.0 / Float32(SOTA_ATMO_TRANS_STEPS))
    var t0 = t_max * u0 * u0
    var t1 = t_max * u1 * u1
    var t = (t0 + t1) * 0.5
    var ri = sqrt_f(max_f(r * r + t * t + r * mu * t * 2.0, 0.0))
    var med = _Medium(a, ri - a.rb)
    tau_v.set(tau_v.get() + med.sig_t * (t1 - t0))
    i_v.set(i + UInt32(1))
    dsl.end_loop()
    b_hist.store_vec3(t_off + (y * UInt32(SOTA_ATMO_TRANS_W) + x) * UInt32(4), _gexp3(-tau_v.get()))
    return dsl.finish()


# ============================================================================
# 2. Multiple-scattering LUT
# ============================================================================

def sota_atmo_multiscatter_shader() -> List[UInt32]:
    """Push: 0 trans_off, 1 ms_off."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_glob = dsl.buffer[1](BIND_AAA_GLOBALS)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var x = dsl.global_id_x()
    var y = dsl.global_id_y()
    var t_off = dsl.push_u32(0)
    var m_off = dsl.push_u32(1)
    dsl.return_if(x.ge(UInt32(SOTA_ATMO_MS_DIM)) | y.ge(UInt32(SOTA_ATMO_MS_DIM)))
    var a = _Atmo(dsl, b_glob)
    var n = Float32(SOTA_ATMO_MS_DIM)
    var mu_s = (x.to_f32() + 0.5) * (2.0 / n) - 1.0
    var r = a.rb + (y.to_f32() + 0.5) * (1.0 / n) * (a.rt - a.rb)
    var sun = GpuVec3(sqrt_f(max_f(1.0 - mu_s * mu_s, 0.0)), mu_s, dsl.f32(0.0))
    var l2_v = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var f_v = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var di_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var di = di_v.get()
    dsl.break_if(di.ge(UInt32(SOTA_ATMO_MS_DIRS)))
    var z = 1.0 - (di.to_f32() * 2.0 + 1.0) * (1.0 / Float32(SOTA_ATMO_MS_DIRS))
    var rr = sqrt_f(max_f(1.0 - z * z, 0.0))
    var phi = di.to_f32() * Float32(2.39996322972865)
    var d = GpuVec3(rr * cos_f(phi), z, rr * sin_f(phi))
    var tg = _ground_dist(r, d.y(), a.rb)
    var hit = tg.gt(0.0)
    var t_max = select_f(hit, tg, _top_dist(r, d.y(), a.rt))
    var thr_v = dsl.var_vec3(dsl.vec3_const(1.0, 1.0, 1.0))
    var s_i = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var si = s_i.get()
    dsl.break_if(si.ge(UInt32(SOTA_ATMO_MS_STEPS)))
    var u0 = si.to_f32() * (1.0 / Float32(SOTA_ATMO_MS_STEPS))
    var u1 = (si + UInt32(1)).to_f32() * (1.0 / Float32(SOTA_ATMO_MS_STEPS))
    var t0 = t_max * u0 * u0
    var t1 = t_max * u1 * u1
    var t = (t0 + t1) * 0.5
    var dt = t1 - t0
    var px = d.x() * t
    var py = r + d.y() * t
    var pz = d.z() * t
    var ri = sqrt_f(px * px + py * py + pz * pz)
    var med = _Medium(a, ri - a.rb)
    var sig_s = med.sig_r + med.sig_m
    var tr_step = _gexp3(med.sig_t * (-dt))
    var mu_l = (px * sun.x() + py * sun.y() + pz * sun.z()) / ri
    var t_sun = _sample_trans(dsl, b_hist, t_off, a, ri, mu_l) * _sun_vis(ri, mu_l, a)
    var thr = thr_v.get()
    var integ_s = _step_integral(sig_s, med.sig_t, tr_step)
    l2_v.set(l2_v.get() + thr * t_sun * integ_s * Float32(0.0795774715459))
    f_v.set(f_v.get() + thr * integ_s)
    thr_v.set(thr * tr_step)
    s_i.set(si + UInt32(1))
    dsl.end_loop()
    dsl.if_then(hit)
    var gx = d.x() * tg
    var gy = r + d.y() * tg
    var gz = d.z() * tg
    var ndl = max_f((gx * sun.x() + gy * sun.y() + gz * sun.z()) / a.rb, 0.0)
    var t_g = _sample_trans(dsl, b_hist, t_off, a, a.rb, ndl) * _sun_vis(a.rb, ndl, a)
    l2_v.set(l2_v.get() + thr_v.get() * t_g * a.albedo * (ndl * Float32(1.0 / 3.14159265358979)))
    dsl.end_if()
    di_v.set(di + UInt32(1))
    dsl.end_loop()
    var inv_n = Float32(1.0 / Float32(SOTA_ATMO_MS_DIRS))
    var l2 = l2_v.get() * inv_n
    var f = f_v.get() * inv_n
    var psi = GpuVec3(
        l2.x() / max_f(1.0 - f.x(), 1e-3),
        l2.y() / max_f(1.0 - f.y(), 1e-3),
        l2.z() / max_f(1.0 - f.z(), 1e-3),
    )
    b_hist.store_vec3(m_off + (y * UInt32(SOTA_ATMO_MS_DIM) + x) * UInt32(4), psi)
    return dsl.finish()


# ============================================================================
# 3. Sky-view LUT
# ============================================================================

def _sky_view_dir(mut dsl: ShaderDsl, a: _Atmo, u: GpuF32, v: GpuF32) -> GpuVec3:
    var half_pi = Float32(1.5707963268)
    var el_h = -_gacos(dsl, min_f(a.rb / a.r_cam, 1.0))
    var t_up = 1.0 - v * 2.0
    var t_dn = v * 2.0 - 1.0
    var el = select_f(v.lt(0.5), el_h + (half_pi - el_h) * t_up * t_up, el_h - (el_h + half_pi) * t_dn * t_dn)
    var az = (u - 0.5) * Float32(6.28318530718)
    var ce = cos_f(el)
    return GpuVec3(ce * cos_f(az), sin_f(el), ce * sin_f(az))


def sota_atmo_skyview_shader() -> List[UInt32]:
    """Push: 0 trans_off, 1 ms_off, 2 sky_off."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=4, push_words=AAA_PUSH_WORDS)
    var b_glob = dsl.buffer[1](BIND_AAA_GLOBALS)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var x = dsl.global_id_x()
    var y = dsl.global_id_y()
    var t_off = dsl.push_u32(0)
    var m_off = dsl.push_u32(1)
    var s_off = dsl.push_u32(2)
    dsl.return_if(x.ge(UInt32(SOTA_ATMO_SKY_W)) | y.ge(UInt32(SOTA_ATMO_SKY_H)))
    var a = _Atmo(dsl, b_glob)
    var d = _sky_view_dir(dsl, a, (x.to_f32() + 0.5) * (1.0 / Float32(SOTA_ATMO_SKY_W)), (y.to_f32() + 0.5) * (1.0 / Float32(SOTA_ATMO_SKY_H)))
    var r0 = a.r_cam
    var tg = _ground_dist(r0, d.y(), a.rb)
    var hit = tg.gt(0.0)
    var t_max = select_f(hit, tg, _top_dist(r0, d.y(), a.rt))
    var seg = _march_segment(dsl, b_hist, t_off, m_off, a, r0, d, t_max, SOTA_ATMO_SKY_STEPS)
    var l_v = dsl.var_vec3(seg[0])
    dsl.if_then(hit)
    var gx = d.x() * tg
    var gy = r0 + d.y() * tg
    var gz = d.z() * tg
    var ns = max_f((gx * a.sun_d.x() + gy * a.sun_d.y() + gz * a.sun_d.z()) / a.rb, 0.0)
    var nm = max_f((gx * a.moon_d.x() + gy * a.moon_d.y() + gz * a.moon_d.z()) / a.rb, 0.0)
    var inv_pi = Float32(1.0 / 3.14159265358979)
    var e_g = _sample_trans(dsl, b_hist, t_off, a, a.rb, ns) * _sun_vis(a.rb, ns, a) * a.sun_e * (ns * inv_pi)
    e_g = e_g + _sample_trans(dsl, b_hist, t_off, a, a.rb, nm) * _sun_vis(a.rb, nm, a) * a.moon_e * (nm * inv_pi)
    l_v.set(l_v.get() + seg[1] * e_g * a.albedo)
    dsl.end_if()
    b_hist.store_vec3(s_off + (y * UInt32(SOTA_ATMO_SKY_W) + x) * UInt32(4), l_v.get())
    return dsl.finish()


# ============================================================================
# 4. Aerial-perspective volume
# ============================================================================

def sota_atmo_aerial_shader() -> List[UInt32]:
    """One thread per froxel column, cumulative march. Push: 0 trans_off, 1 ms_off, 2 ap_off."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_glob = dsl.buffer[1](BIND_AAA_GLOBALS)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var x = dsl.global_id_x()
    var y = dsl.global_id_y()
    var t_off = dsl.push_u32(0)
    var m_off = dsl.push_u32(1)
    var ap_off = dsl.push_u32(2)
    comptime N = SOTA_ATMO_AP_DIM
    dsl.return_if(x.ge(UInt32(N)) | y.ge(UInt32(N)))
    var a = _Atmo(dsl, b_glob)
    var fwd = b_glob.load_vec3(dsl.u32(36))
    var tan_x = b_glob.load_f32(dsl.u32(39))
    var rgt = b_glob.load_vec3(dsl.u32(40))
    var tan_y = b_glob.load_f32(dsl.u32(43))
    var up = b_glob.load_vec3(dsl.u32(44))
    var ndc_x = (x.to_f32() + 0.5) * (2.0 / Float32(N)) - 1.0
    var ndc_y = 1.0 - (y.to_f32() + 0.5) * (2.0 / Float32(N))
    var d = (fwd + rgt * (ndc_x * tan_x) + up * (ndc_y * tan_y)).normalized()
    var r0 = a.r_cam
    var cvs = d.dot(a.sun_d)
    var cvm = d.dot(a.moon_d)
    var has_moon = (a.moon_e.x() + a.moon_e.y() + a.moon_e.z()).gt(0.0)
    var l_v = dsl.var_vec3(dsl.vec3_const(0.0, 0.0, 0.0))
    var thr_v = dsl.var_vec3(dsl.vec3_const(1.0, 1.0, 1.0))
    var t_prev = dsl.var_f32_const(0.0)
    var s_v = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var s = s_v.get()
    dsl.break_if(s.ge(UInt32(N)))
    var us = (s + UInt32(1)).to_f32() * (1.0 / Float32(N))
    var t_end = a.ap_max * us * us
    var tp = t_prev.get()
    for k in range(SOTA_ATMO_AP_SUBSTEPS):
        var ta = tp + (t_end - tp) * (Float32(k) / Float32(SOTA_ATMO_AP_SUBSTEPS))
        var tb = tp + (t_end - tp) * (Float32(k + 1) / Float32(SOTA_ATMO_AP_SUBSTEPS))
        var t = (ta + tb) * 0.5
        var px = d.x() * t
        var py = r0 + d.y() * t
        var pz = d.z() * t
        var ri = max_f(sqrt_f(px * px + py * py + pz * pz), a.rb)
        var med = _Medium(a, ri - a.rb)
        var mu_s = (px * a.sun_d.x() + py * a.sun_d.y() + pz * a.sun_d.z()) / ri
        var src_v = dsl.var_vec3(_light_term(dsl, b_hist, t_off, m_off, a, ri, mu_s, cvs, med, a.sun_e))
        dsl.if_then(has_moon)
        var mu_m = (px * a.moon_d.x() + py * a.moon_d.y() + pz * a.moon_d.z()) / ri
        src_v.set(src_v.get() + _light_term(dsl, b_hist, t_off, m_off, a, ri, mu_m, cvm, med, a.moon_e))
        dsl.end_if()
        var tr_step = _gexp3(med.sig_t * (ta - tb))
        var thr = thr_v.get()
        l_v.set(l_v.get() + thr * _step_integral(src_v.get(), med.sig_t, tr_step))
        thr_v.set(thr * tr_step)
    var w0 = ap_off + ((s * UInt32(N) + y) * UInt32(N) + x) * UInt32(8)
    b_hist.store_vec3(w0, l_v.get())
    b_hist.store_vec3(w0 + UInt32(4), thr_v.get())
    t_prev.set(t_end)
    s_v.set(s + UInt32(1))
    dsl.end_loop()
    return dsl.finish()


# ============================================================================
# 5. Apply (sky / sun / moon, aerial perspective, height-fog tail)
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


def sota_atmo_apply_shader() -> List[UInt32]:
    """Push: 0 W, 1 H, 2 trans_off, 3 sky_off, 4 ap_off."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_glob = dsl.buffer[1](BIND_AAA_GLOBALS)
    var b_hdr = dsl.buffer[1](BIND_AAA_HDR_OUT)
    var b_hist = dsl.buffer[1](BIND_AAA_HISTORY)
    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var t_off = dsl.push_u32(2)
    var s_off = dsl.push_u32(3)
    var ap_off = dsl.push_u32(4)
    dsl.return_if(px.ge(w_u) | py.ge(h_u))
    var a = _Atmo(dsl, b_glob)
    var flags = b_glob.load_u32(dsl.u32(UInt32(SOTA_ATMO_G + 41)))
    var base = (py * w_u + px) * UInt32(8)
    var z = b_hdr.load_f32(base + UInt32(3))
    var is_sky = z.ge(1.0e5)
    var col_v = dsl.var_vec3(b_hdr.load_vec3(base))

    var fw = w_u.to_f32()
    var fh = h_u.to_f32()
    var cam = b_glob.load_vec3(dsl.u32(32))
    var fwd = b_glob.load_vec3(dsl.u32(36))
    var tan_x = b_glob.load_f32(dsl.u32(39))
    var rgt = b_glob.load_vec3(dsl.u32(40))
    var tan_y = b_glob.load_f32(dsl.u32(43))
    var up = b_glob.load_vec3(dsl.u32(44))
    var jit_x = b_glob.load_f32(dsl.u32(101))
    var jit_y = b_glob.load_f32(dsl.u32(102))
    var fx0 = px.to_f32() + 0.5 - jit_x * fw * 0.5
    var fy0 = py.to_f32() + 0.5 + jit_y * fh * 0.5
    var ndc_x = fx0 / fw * 2.0 - 1.0
    var ndc_y = 1.0 - fy0 / fh * 2.0
    var ray = fwd + rgt * (ndc_x * tan_x) + up * (ndc_y * tan_y)
    var sec = ray.length()
    var d = ray / sec
    var dist_w = z * sec

    # --- sky: LUT + sun disc + moon disc ---
    dsl.if_then(is_sky & (flags & SOTA_ATMO_FLAG_SKY).ne(UInt32(0)))
    var half_pi = Float32(1.5707963268)
    var el_h = -_gacos(dsl, min_f(a.rb / a.r_cam, 1.0))
    var horiz = sqrt_f(max_f(d.x() * d.x() + d.z() * d.z(), 0.0))
    var el = _gatan2(dsl, d.y(), horiz)
    var v_up = 0.5 - sqrt_f(saturate((el - el_h) / (half_pi - el_h))) * 0.5
    var v_dn = sqrt_f(saturate((el_h - el) / (el_h + half_pi))) * 0.5 + 0.5
    var v = select_f(el.ge(el_h), v_up, v_dn)
    var u = _gatan2(dsl, d.z(), d.x()) * Float32(0.159154943) + 0.5
    var sky = _bilerp_buf(dsl, b_hist, s_off, SOTA_ATMO_SKY_W, SOTA_ATMO_SKY_H, u * Float32(SOTA_ATMO_SKY_W) - 0.5, v * Float32(SOTA_ATMO_SKY_H) - 0.5, True)
    var sky_v = dsl.var_vec3(sky)
    var above = _ground_dist(a.r_cam, d.y(), a.rb).lt(0.0)
    var t_view = _sample_trans(dsl, b_hist, t_off, a, a.r_cam, d.y())
    var pix_ang = tan_y * 2.0 / fh
    # Sun disc (linear limb darkening normalised to the disc irradiance), 1-pixel analytic edge.
    dsl.if_then(above & (flags & SOTA_ATMO_FLAG_SUN_DISC).ne(UInt32(0)))
    var ang_s = _gacos(dsl, d.dot(a.sun_d))
    var cov_s = saturate((a.sun_ar - ang_s) / pix_ang + 0.5)
    dsl.if_then(cov_s.gt(0.0))
    var s_rel = min_f(ang_s / a.sun_ar, 1.0)
    var mu_disc = sqrt_f(max_f(1.0 - s_rel * s_rel, 0.0))
    var limb_u = b_glob.load_f32(dsl.u32(UInt32(SOTA_ATMO_G + 43)))
    var limb = (1.0 - limb_u * (1.0 - mu_disc)) / (1.0 - limb_u * Float32(1.0 / 3.0))
    var omega = (1.0 - cos_f(a.sun_ar)) * Float32(2.0 * 3.14159265358979)
    var disc_scale = b_glob.load_f32(dsl.u32(UInt32(SOTA_ATMO_G + 40)))
    sky_v.set(sky_v.get() + a.sun_e * t_view * (limb / omega * cov_s * disc_scale))
    dsl.end_if()
    dsl.end_if()
    # Moon disc (Lommel-Seeliger, lit from the sun direction: the phase follows the geometry).
    dsl.if_then(above & (flags & SOTA_ATMO_FLAG_MOON_DISC).ne(UInt32(0)))
    var c_m = d.dot(a.moon_d)
    var ang_m = _gacos(dsl, c_m)
    var cov_m = saturate((a.moon_ar - ang_m) / pix_ang + 0.5)
    dsl.if_then(cov_m.gt(0.0))
    var s_m = min_f(sin_f(ang_m) / sin_f(a.moon_ar), 1.0)
    var off = d - a.moon_d * c_m
    var ol = max_f(off.length(), 1e-9)
    var tang = off / ol
    var nz = sqrt_f(max_f(1.0 - s_m * s_m, 0.0))
    var nrm = tang * s_m - a.moon_d * nz
    var mu0 = max_f(nrm.dot(a.sun_d), 0.0)
    var ls = a.moon_alb * mu0 / (mu0 + max_f(nz, 1e-3))
    sky_v.set(sky_v.get() + a.sun_e * t_view * (ls * cov_m))  # lit by the sun's top-of-atmosphere irradiance
    dsl.end_if()
    dsl.end_if()
    col_v.set(sky_v.get())
    dsl.end_if()

    # --- aerial perspective on surfaces ---
    dsl.if_then((~is_sky) & (flags & SOTA_ATMO_FLAG_AP).ne(UInt32(0)))
    comptime N = SOTA_ATMO_AP_DIM
    var dist_km = dist_w * a.kpu
    var fxc = (ndc_x + 1.0) * (0.5 * Float32(N)) - 0.5
    var fyc = (1.0 - ndc_y) * (0.5 * Float32(N)) - 0.5
    var sc = sqrt_f(max_f(dist_km / a.ap_max, 0.0)) * Float32(N) - 1.0
    var s0f = clamp_f(floor_f(sc), 0.0, Float32(N - 1))
    var s0 = s0f.saturate_u32()
    var s1 = min_u(s0 + UInt32(1), dsl.u32(UInt32(N - 1)))
    var fs = saturate(sc - s0f)
    var f0 = _ap_fetch(dsl, b_hist, ap_off, fxc, fyc, s0)
    var f1 = _ap_fetch(dsl, b_hist, ap_off, fxc, fyc, s1)
    var below = sc.lt(0.0)
    var wb = saturate(sc + 1.0)
    var one3 = dsl.vec3_const(1.0, 1.0, 1.0)
    var s_val = select_vec3(below, f0[0] * wb, f0[0] * (1.0 - fs) + f1[0] * fs)
    var t_val = select_vec3(below, one3 * (1.0 - wb) + f0[1] * wb, f0[1] * (1.0 - fs) + f1[1] * fs)
    col_v.set(col_v.get() * t_val + s_val)
    dsl.end_if()

    # --- analytic exponential height-fog tail ---
    dsl.if_then((flags & SOTA_ATMO_FLAG_FOG).ne(UInt32(0)))
    var g0 = SOTA_ATMO_G + 45
    var f_dens = b_glob.load_f32(dsl.u32(UInt32(g0 + 0)))
    var f_start_z = b_glob.load_f32(dsl.u32(UInt32(g0 + 1)))
    var f_fall = b_glob.load_f32(dsl.u32(UInt32(g0 + 2)))
    var f_base = b_glob.load_f32(dsl.u32(UInt32(g0 + 3)))
    var f_alb = b_glob.load_vec3(dsl.u32(UInt32(g0 + 4)))
    var f_g = b_glob.load_f32(dsl.u32(UInt32(g0 + 7)))
    var f_max = b_glob.load_f32(dsl.u32(UInt32(g0 + 8)))
    var t0 = f_start_z * sec
    var t1 = select_f(is_sky, f_max, min_f(dist_w, f_max))
    var od = _height_fog_od(f_dens, f_fall, f_base, cam.y(), d.y(), t0, t1)
    dsl.if_then(od.gt(1e-6))
    var tr = _gexp(-od)
    var l_dir = b_glob.load_vec3(dsl.u32(48))
    var l_rad = b_glob.load_vec3(dsl.u32(52))
    var ibl_int = b_glob.load_f32(dsl.u32(55))
    var sh = emit_eval_sh9_irradiance(dsl.raw_module()[], b_glob.buf_id, 72, d.x().id, d.y().id, d.z().id)
    var amb = GpuVec3(GpuF32(dsl.mod_addr, sh[0]), GpuF32(dsl.mod_addr, sh[1]), GpuF32(dsl.mod_addr, sh[2])) * (ibl_int * 0.28)
    var ph = _cs(d.dot(l_dir), clamp_f(f_g, -0.92, 0.92))
    var l_in = (l_rad * ph + amb) * f_alb
    col_v.set(col_v.get() * tr + l_in * (1.0 - tr))
    dsl.end_if()
    dsl.end_if()

    b_hdr.store_vec3(base, col_v.get())
    return dsl.finish()


# ============================================================================
# Settings, CPU-side frame logic, recording
# ============================================================================

@always_inline
def _fb(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


@always_inline
def _put_f(addr: Int, word: Int, v: Float32):
    put_float(addr + word * 4, v)


@always_inline
def _put_u(addr: Int, word: Int, v: UInt32):
    put32(addr + word * 4, v)


def _put_rgb(addr: Int, word: Int, v: RGB):
    _put_f(addr, word, v[0])
    _put_f(addr, word + 1, v[1])
    _put_f(addr, word + 2, v[2])


def _put_v3(addr: Int, word: Int, v: Vec3):
    _put_f(addr, word, v.x)
    _put_f(addr, word + 1, v.y)
    _put_f(addr, word + 2, v.z)


struct SotaAtmosphere(Movable):
    """Area-H settings (all cvars; `enabled = False` keeps the v5 path bit for bit).

    Units: the world is `km_per_unit` km per unit (0.001: metres). The world origin sits at
    `origin_altitude_km` above sea level; the planet centre is under the camera.

    Light sources: with `use_time_of_day` the sun / moon follow `time_of_day` (date, UTC, latitude,
    longitude); otherwise `sun_dir` (and `moon_dir` when `moon_enabled`). With `drive_renderer_light`
    the renderer's directional light becomes the brighter of the sun and the moon after the atmosphere
    (direction, radiance = TOA irradiance x transmittance x earth shadow, angular radius)."""
    var enabled: Bool
    var params: SotaAtmoParams
    var km_per_unit: Float32
    var origin_altitude_km: Float32
    var ap_max_km: Float32
    var sky: Bool
    var aerial_perspective: Bool
    var sun_disc: Bool
    var moon_disc: Bool
    var sun_disc_scale: Float32
    var fog_tail: Bool
    var fog_tail_max_dist: Float32
    var height_fog_density: Float32   # standalone analytic height fog (per unit) when the froxel fog is off
    var use_time_of_day: Bool
    var time_of_day: SotaTimeOfDay
    var sun_dir: Vec3
    var moon_enabled: Bool
    var moon_dir: Vec3
    var drive_renderer_light: Bool
    var update_ibl: Bool
    var ibl_update_deg: Float32
    # state
    var pipe_trans: Int
    var pipe_ms: Int
    var pipe_sky: Int
    var pipe_ap: Int
    var pipe_apply: Int
    var region_base: Int
    var gpu_lut_signature: Float64
    var cpu_luts_signature: Float64
    var cpu_luts: List[SotaAtmoLuts]
    var ibl_sun: Vec3
    var ibl_moon: Vec3
    var ibl_valid: Bool
    var last_ibl_ms: Float64
    var lights: SotaAtmoLights
    var moon_phase_angle: Float32

    def __init__(out self):
        self.enabled = False
        self.params = SotaAtmoParams()
        self.km_per_unit = 0.001
        self.origin_altitude_km = 0.0
        self.ap_max_km = 32.0
        self.sky = True
        self.aerial_perspective = True
        self.sun_disc = True
        self.moon_disc = True
        self.sun_disc_scale = 1.0
        self.fog_tail = True
        self.fog_tail_max_dist = 6000.0
        self.height_fog_density = 0.0
        self.use_time_of_day = False
        self.time_of_day = SotaTimeOfDay()
        self.sun_dir = Vec3(0.35, 0.78, -0.52).normalized()
        self.moon_enabled = False
        self.moon_dir = Vec3(-0.3, 0.5, 0.8).normalized()
        self.drive_renderer_light = True
        self.update_ibl = True
        self.ibl_update_deg = 1.0
        self.pipe_trans = 0
        self.pipe_ms = 0
        self.pipe_sky = 0
        self.pipe_ap = 0
        self.pipe_apply = 0
        self.region_base = 0
        self.gpu_lut_signature = -1.0
        self.cpu_luts_signature = -1.0
        self.cpu_luts = List[SotaAtmoLuts]()
        self.ibl_sun = Vec3(0.0, 0.0, 0.0)
        self.ibl_moon = Vec3(0.0, 0.0, 0.0)
        self.ibl_valid = False
        self.last_ibl_ms = 0.0
        self.lights = SotaAtmoLights(Vec3(0.0, 1.0, 0.0), RGB(0.0))
        self.moon_phase_angle = 0.0

    def create_pipelines(mut self, mut vk: VulkanGpu, pipe_layout: Int) -> Bool:
        self.pipe_trans = vk.create_aaa_compute_pipeline(pipe_layout, sota_atmo_transmittance_shader())
        self.pipe_ms = vk.create_aaa_compute_pipeline(pipe_layout, sota_atmo_multiscatter_shader())
        self.pipe_sky = vk.create_aaa_compute_pipeline(pipe_layout, sota_atmo_skyview_shader())
        self.pipe_ap = vk.create_aaa_compute_pipeline(pipe_layout, sota_atmo_aerial_shader())
        self.pipe_apply = vk.create_aaa_compute_pipeline(pipe_layout, sota_atmo_apply_shader())
        return self.pipelines_ready()

    def destroy_pipelines(mut self, mut vk: VulkanGpu):
        for p in [self.pipe_trans, self.pipe_ms, self.pipe_sky, self.pipe_ap, self.pipe_apply]:
            if p != 0:
                vk.destroy_pipeline(p)
        self.pipe_trans = 0
        self.pipe_ms = 0
        self.pipe_sky = 0
        self.pipe_ap = 0
        self.pipe_apply = 0

    def pipelines_ready(self) -> Bool:
        return self.pipe_trans != 0 and self.pipe_ms != 0 and self.pipe_sky != 0 and self.pipe_ap != 0 and self.pipe_apply != 0

    def active(self) -> Bool:
        return self.enabled and self.pipelines_ready()

    def trans_off(self) -> Int:
        return self.region_base

    def ms_off(self) -> Int:
        return self.region_base + sota_atmo_trans_words()

    def sky_off(self) -> Int:
        return self.ms_off() + sota_atmo_ms_words()

    def ap_off(self) -> Int:
        return self.sky_off() + sota_atmo_sky_words()

    def replaces_legacy_ap(self) -> Bool:
        return self.active() and self.aerial_perspective

    def camera_altitude_km(self, cam_y: Float32) -> Float32:
        return max(Float32(0.001), self.origin_altitude_km + cam_y * self.km_per_unit)

    def ensure_luts(mut self):
        """CPU transmittance / multi-scattering LUTs for the current medium (about 70 ms, once per change)."""
        var sig = self.params.signature()
        if len(self.cpu_luts) == 0 or sig != self.cpu_luts_signature:
            self.cpu_luts.clear()
            self.cpu_luts.append(SotaAtmoLuts(self.params))
            self.cpu_luts_signature = sig
            self.ibl_valid = False

    def update_lights(mut self):
        var sun_e = self.params.solar_irradiance
        if self.use_time_of_day:
            var sd = self.time_of_day.sun_dir()
            var md = self.time_of_day.moon_dir()
            self.moon_phase_angle = Float32(3.14159265358979) - self.time_of_day.moon_elongation_rad()
            self.lights = SotaAtmoLights(sd, sun_e, md, moon_irradiance(sun_e, self.moon_phase_angle, self.params))
        else:
            var sd = self.sun_dir.normalized()
            if self.moon_enabled:
                var md = self.moon_dir.normalized()
                self.moon_phase_angle = acos(max(Float32(-1.0), min(Float32(1.0), -sd.dot(md))))
                self.lights = SotaAtmoLights(sd, sun_e, md, moon_irradiance(sun_e, self.moon_phase_angle, self.params))
            else:
                self.lights = SotaAtmoLights(sd, sun_e, Vec3(0.0, -1.0, 0.0), RGB(0.0))

    def primary_light(mut self, cam_y: Float32) -> Tuple[Vec3, Vec3, Float32]:
        """(direction, radiance at the camera, angular radius) of the brighter of sun and moon."""
        var p = self.params
        var r = p.r_bottom_km + self.camera_altitude_km(cam_y)
        var l = self.lights
        self.ensure_luts()
        var ts = sample_transmittance(self.cpu_luts[0].trans, r, l.sun_dir.y, p) * sun_visibility(r, l.sun_dir.y, p)
        var es = l.sun_e * ts
        var tm = sample_transmittance(self.cpu_luts[0].trans, r, l.moon_dir.y, p) * sun_visibility(r, l.moon_dir.y, p)
        var em = l.moon_e * tm
        if rgb_luma(es) >= rgb_luma(em):
            return (l.sun_dir, Vec3(es[0], es[1], es[2]), p.sun_angular_radius)
        return (l.moon_dir, Vec3(em[0], em[1], em[2]), p.moon_angular_radius)

    def needs_ibl_refresh(self) -> Bool:
        if not self.update_ibl:
            return False
        if not self.ibl_valid:
            return True
        var c = cos(self.ibl_update_deg * Float32(0.0174532925))
        return self.lights.sun_dir.dot(self.ibl_sun) < c or self.lights.moon_dir.dot(self.ibl_moon) < c

    def bake_ibl(mut self, cam_y: Float32, base_dim: Int, mip_levels: Int) -> BakedIblEnvironment:
        """SH9 + split-sum GGX cubemap of this sky (no discs) from a 64 x 36 CPU sky-view LUT."""
        var alt = self.camera_altitude_km(cam_y)
        var lights = self.lights
        var p = self.params
        self.ensure_luts()
        var lw = 64
        var lh = 36
        var sv = bake_sky_view_lut(self.cpu_luts[0], lights, alt, lw, lh)
        var res = BakedIblEnvironment()
        res.base_dim = base_dim
        res.mip_levels = mip_levels
        var sh_raw = List[Vec3](length=9, fill=Vec3(0.0, 0.0, 0.0))
        var weight_sum = Float32(0.0)
        var sh_dim = 16
        for face in range(6):
            for y in range(sh_dim):
                var ty = (Float32(y) + Float32(0.5)) / Float32(sh_dim) * Float32(2.0) - Float32(1.0)
                for x in range(sh_dim):
                    var tx = (Float32(x) + Float32(0.5)) / Float32(sh_dim) * Float32(2.0) - Float32(1.0)
                    var tmp = Float32(1.0) + tx * tx + ty * ty
                    var d_omega = Float32(4.0) / (sqrt(tmp) * tmp)
                    var dir = cube_face_direction(face, (Float32(x) + Float32(0.5)) / Float32(sh_dim), (Float32(y) + Float32(0.5)) / Float32(sh_dim))
                    var c = sample_sky_view(sv, dir, alt, p, lw, lh)
                    var rad = Vec3(c[0], c[1], c[2])
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
        var band = [Float32(0.28209479), 0.32573500, 0.32573500, 0.32573500, 0.27313711, 0.27313711, 0.07884789, 0.27313711, 0.13656855]
        for k in range(9):
            res.sh9[k] = sh_raw[k] * (norm_4pi * band[k])
        for mip in range(mip_levels):
            var m_dim = max(4, base_dim >> mip)
            var roughness = Float32(mip) / Float32(max(1, mip_levels - 1))
            var samples = 16
            for face in range(6):
                var face_hdr = List[Vec3](length=m_dim * m_dim, fill=Vec3(0.0, 0.0, 0.0))
                for y in range(m_dim):
                    var v = (Float32(y) + Float32(0.5)) / Float32(m_dim)
                    for x in range(m_dim):
                        var u = (Float32(x) + Float32(0.5)) / Float32(m_dim)
                        var r_dir = cube_face_direction(face, u, v)
                        if mip == 0:
                            var c = sample_sky_view(sv, r_dir, alt, p, lw, lh)
                            face_hdr[y * m_dim + x] = Vec3(c[0], c[1], c[2])
                        else:
                            var acc = Vec3(0.0, 0.0, 0.0)
                            var w_tot = Float32(0.0)
                            for s_i in range(samples):
                                var hv = _importance_sample_ggx(Float32(s_i) / Float32(samples), _radical_inverse_vdc(UInt32(s_i)), roughness, r_dir)
                                var l = (hv * (Float32(2.0) * r_dir.dot(hv)) - r_dir).normalized()
                                var ndl = max(Float32(0.0), r_dir.dot(l))
                                if ndl > Float32(0.0):
                                    var c = sample_sky_view(sv, l, alt, p, lw, lh)
                                    acc = acc + Vec3(c[0], c[1], c[2]) * ndl
                                    w_tot += ndl
                            face_hdr[y * m_dim + x] = acc * (Float32(1.0) / max(Float32(1e-5), w_tot))
                res.mip_face_bytes.append(encode_bc6h_hdr(face_hdr, m_dim, m_dim))
        self.ibl_sun = lights.sun_dir
        self.ibl_moon = lights.moon_dir
        self.ibl_valid = True
        return res^

    def write_globals(
        mut self,
        g_addr: Int,
        cam_y: Float32,
        legacy_sky_fx: Bool,
        froxel_on: Bool,
        fog_density: Float32,
        froxel_far_z: Float32,
        fog_falloff: Float32,
        fog_base_y: Float32,
        fog_albedo: Vec3,
        fog_g: Float32,
    ):
        """Per-frame constants (`b_globals[SOTA_ATMO_G ..]`). `legacy_sky_fx`: v5 clouds or stars are on, so
        sky pixels keep the v5 sky (only aerial perspective and fog run)."""
        var p = self.params
        var w = SOTA_ATMO_G
        _put_rgb(g_addr, w + 0, p.rayleigh_scat)
        _put_f(g_addr, w + 3, p.rayleigh_h_km)
        _put_rgb(g_addr, w + 4, p.mie_scat)
        _put_f(g_addr, w + 7, p.mie_h_km)
        _put_rgb(g_addr, w + 8, p.mie_ext)
        _put_f(g_addr, w + 11, p.mie_g)
        _put_rgb(g_addr, w + 12, p.ozone_abs)
        _put_f(g_addr, w + 15, p.ozone_center_km)
        _put_f(g_addr, w + 16, p.ozone_half_width_km)
        _put_f(g_addr, w + 17, p.r_bottom_km)
        _put_f(g_addr, w + 18, p.r_top_km)
        _put_f(g_addr, w + 19, self.km_per_unit)
        _put_rgb(g_addr, w + 20, p.ground_albedo)
        _put_f(g_addr, w + 23, min(p.r_top_km - Float32(0.001), p.r_bottom_km + self.camera_altitude_km(cam_y)))
        var l = self.lights
        _put_v3(g_addr, w + 24, l.sun_dir)
        _put_f(g_addr, w + 27, p.sun_angular_radius)
        _put_rgb(g_addr, w + 28, l.sun_e)
        _put_f(g_addr, w + 31, self.ap_max_km)
        _put_v3(g_addr, w + 32, l.moon_dir)
        _put_f(g_addr, w + 35, p.moon_angular_radius)
        _put_rgb(g_addr, w + 36, l.moon_e)
        _put_f(g_addr, w + 39, p.moon_albedo)
        _put_f(g_addr, w + 40, self.sun_disc_scale)
        var flags = UInt32(0)
        if self.sky and not legacy_sky_fx:
            flags |= SOTA_ATMO_FLAG_SKY
        if self.aerial_perspective:
            flags |= SOTA_ATMO_FLAG_AP
        if self.sun_disc:
            flags |= SOTA_ATMO_FLAG_SUN_DISC
        if self.moon_disc and (self.use_time_of_day or self.moon_enabled):
            flags |= SOTA_ATMO_FLAG_MOON_DISC
        var f_dens = Float32(0.0)
        var f_start = Float32(0.0)
        if self.fog_tail:
            if froxel_on and fog_density > Float32(0.0):
                f_dens = fog_density * Float32(0.92)  # mean of the froxel FBM modulation (0.92 +- 0.28 / 0.16)
                f_start = froxel_far_z
            elif self.height_fog_density > Float32(0.0):
                f_dens = self.height_fog_density
                f_start = 0.0
        if f_dens > Float32(0.0):
            flags |= SOTA_ATMO_FLAG_FOG
        _put_u(g_addr, w + 41, flags)
        _put_f(g_addr, w + 42, Float32(0.0))
        _put_f(g_addr, w + 43, p.sun_limb_u)
        _put_f(g_addr, w + 44, Float32(0.0))
        # fog tail block [45 .. 53]
        _put_f(g_addr, w + 45, f_dens)
        _put_f(g_addr, w + 46, f_start)
        _put_f(g_addr, w + 47, max(Float32(0.001), fog_falloff))
        _put_f(g_addr, w + 48, fog_base_y)
        _put_v3(g_addr, w + 49, fog_albedo)
        _put_f(g_addr, w + 52, fog_g)
        _put_f(g_addr, w + 53, self.fog_tail_max_dist)


def _barrier(mut vk: VulkanGpu, slot: Int):
    vk.memory_barrier2(slot, PIPELINE_COMPUTE, Int(ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE))


def sota_atmo_record_luts(
    mut atmo: SotaAtmosphere,
    mut vk: VulkanGpu,
    slot: Int,
    pipe_layout: Int,
    desc_set: Int,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
):
    var q = next_q
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q, PIPELINE_BOTTOM)
        next_q += 2
        prof_pairs.append((String("SotaAtmoLuts"), q))
    var push = List[UInt32](length=16, fill=0)
    push[0] = UInt32(atmo.trans_off())
    push[1] = UInt32(atmo.ms_off())
    push[2] = UInt32(atmo.sky_off())
    var sig = atmo.params.signature()
    if sig != atmo.gpu_lut_signature:
        vk.dispatch_aaa(slot, atmo.pipe_trans, pipe_layout, desc_set, push, SOTA_ATMO_TRANS_W // 8, SOTA_ATMO_TRANS_H // 8)
        _barrier(vk, slot)
        vk.dispatch_aaa(slot, atmo.pipe_ms, pipe_layout, desc_set, push, SOTA_ATMO_MS_DIM // 8, SOTA_ATMO_MS_DIM // 8)
        _barrier(vk, slot)
        atmo.gpu_lut_signature = sig
    vk.dispatch_aaa(slot, atmo.pipe_sky, pipe_layout, desc_set, push, SOTA_ATMO_SKY_W // 8, (SOTA_ATMO_SKY_H + 3) // 4)
    var push_ap = List[UInt32](length=16, fill=0)
    push_ap[0] = UInt32(atmo.trans_off())
    push_ap[1] = UInt32(atmo.ms_off())
    push_ap[2] = UInt32(atmo.ap_off())
    vk.dispatch_aaa(slot, atmo.pipe_ap, pipe_layout, desc_set, push_ap, SOTA_ATMO_AP_DIM // 8, SOTA_ATMO_AP_DIM // 8)
    _barrier(vk, slot)
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q + 1, PIPELINE_BOTTOM)


def sota_atmo_record_apply(
    mut atmo: SotaAtmosphere,
    mut vk: VulkanGpu,
    slot: Int,
    pipe_layout: Int,
    desc_set: Int,
    render_w: Int,
    render_h: Int,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
):
    var q = next_q
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q, PIPELINE_BOTTOM)
        next_q += 2
        prof_pairs.append((String("SotaAtmoApply"), q))
    var push = List[UInt32](length=16, fill=0)
    push[0] = UInt32(render_w)
    push[1] = UInt32(render_h)
    push[2] = UInt32(atmo.trans_off())
    push[3] = UInt32(atmo.sky_off())
    push[4] = UInt32(atmo.ap_off())
    vk.dispatch_aaa(slot, atmo.pipe_apply, pipe_layout, desc_set, push, (render_w + 7) // 8, (render_h + 7) // 8)
    _barrier(vk, slot)
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q + 1, PIPELINE_BOTTOM)
