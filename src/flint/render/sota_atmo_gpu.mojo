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
    GpuBuffer,
    VulkanGpu,
    VulkanImage,
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
from flint.render.sota_atmo_clouds import (
    CW_ALPHA,
    CW_DIV,
    CW_DWIND_X,
    CW_DWIND_Z,
    CW_FRAME,
    CW_HIST_VALID,
    CW_JITTER,
    CW_PREV_CAM,
    SOTA_CLOUD_BASE_N,
    SOTA_CLOUD_DETAIL_N,
    SOTA_CLOUD_WEATHER_N,
    SotaClouds,
    cloud_buffer_words,
    cloud_region_words,
    emit_cloud_composite,
    sota_cloud_weather_shader,
)
from flint.render.sota_atmo_ibl import SotaAtmoIblGpu, ibl_region_words
from flint.render.sota_atmo_night import (
    SOTA_NIGHT_W_BG,
    SOTA_NIGHT_W_E0,
    SOTA_NIGHT_W_FLAGS,
    SOTA_NIGHT_W_MLIM,
    SOTA_NIGHT_W_MW,
    SOTA_NIGHT_W_PSF,
    SOTA_NIGHT_W_SEED,
    SOTA_NIGHT_W_STARS,
    SotaNightSky,
    emit_night_sky,
    surface_brightness_radiance,
    world_to_equatorial_rows,
)
from flint.render.sota_atmo_dsl import (
    SOTA_ATMO_FLAG_AP,
    SOTA_ATMO_FLAG_FOG,
    SOTA_ATMO_FLAG_MOON_DISC,
    SOTA_ATMO_FLAG_SKY,
    SOTA_ATMO_FLAG_SUN_DISC,
    SOTA_ATMO_G,
    SOTA_ATMO_G2,
    _Atmo,
    _ap_fetch,
    _bilerp_buf,
    _cs,
    _gacos,
    _gatan2,
    _ground_dist,
    _height_fog_od,
    _march_segment,
    _sample_trans,
    _step_integral,
    _light_term,
    _Medium,
    _gexp,
    _gexp3,
    _top_dist,
    _sun_vis,
    _sky_view_radiance,
)


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
def sota_atmo_region_words(max_pix: Int = 0, env_dim: Int = 64, env_mips: Int = 5) -> Int:
    """LUTs + aerial-perspective volume (1.66 MB), the GPU IBL staging (SH9 + BC6H blocks, about 0.1 MB for a 64^2
    cube), the cloud noise volumes + weather map (1.7 MB) and two cloud images (`max_pix / 4` pixels each)."""
    var w = sota_atmo_trans_words() + sota_atmo_ms_words() + sota_atmo_sky_words() + sota_atmo_ap_words() + 4
    w += ibl_region_words(env_dim, env_mips)
    if max_pix > 0:
        w += cloud_region_words(max_pix)
    return w



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


def sota_atmo_apply_shader() -> List[UInt32]:
    """Push: 0 W, 1 H, 2 trans_off, 3 sky_off, 4 ap_off, 5 cloud image offset, 6 cloud w, 7 cloud h, 8 cloud div."""
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
    var cl_off = dsl.push_u32(5)
    var cl_w = dsl.push_u32(6)
    var cl_h = dsl.push_u32(7)
    var cl_div = dsl.push_u32(8)
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
    var sky = _sky_view_radiance(dsl, b_hist, s_off, a, d)
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
    var occ_v = dsl.var_f32_const(0.0)
    dsl.if_then(above & (flags & SOTA_ATMO_FLAG_MOON_DISC).ne(UInt32(0)))
    var c_m = d.dot(a.moon_d)
    var ang_m = _gacos(dsl, c_m)
    var cov_m = saturate((a.moon_ar - ang_m) / pix_ang + 0.5)
    occ_v.set(cov_m)
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
    # --- sota:atmo r3: night sky (stars, Milky Way, airglow) behind the moon, then the cloud layer in front ---
    var nflags = b_glob.load_u32(dsl.u32(UInt32(SOTA_ATMO_G2 + SOTA_NIGHT_W_FLAGS)))
    dsl.if_then(above & (nflags & UInt32(1)).ne(UInt32(0)))
    var night = emit_night_sky(dsl, b_glob, d, pix_ang)
    sky_v.set(sky_v.get() + night * t_view * (1.0 - occ_v.get()))
    dsl.end_if()
    dsl.if_then((nflags & UInt32(2)).ne(UInt32(0)))
    sky_v.set(emit_cloud_composite(dsl, b_hist, cl_off, cl_w, cl_h, cl_div, fx0, fy0, sky_v.get()))
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
    # --- round 3 ---
    var night: SotaNightSky
    var clouds: SotaClouds
    var ibl_gpu: Bool             # bake SH9 + the BC6H GGX cube on the GPU (no CPU work, no stall)
    var ibl_gpu_update_deg: Float32
    var ibl: SotaAtmoIblGpu
    var ibl_cube_pending: Bool
    var max_pix: Int
    var cloud_cw: Int
    var cloud_ch: Int
    var cloud_div_used: Int

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
        self.night = SotaNightSky()
        self.clouds = SotaClouds()
        self.ibl_gpu = True
        self.ibl_gpu_update_deg = 0.05
        self.ibl = SotaAtmoIblGpu()
        self.ibl_cube_pending = False
        self.max_pix = 0
        self.cloud_cw = 0
        self.cloud_ch = 0
        self.cloud_div_used = 2

    def create_pipelines(mut self, mut vk: VulkanGpu, pipe_layout: Int) -> Bool:
        self.pipe_trans = vk.create_aaa_compute_pipeline(pipe_layout, sota_atmo_transmittance_shader())
        self.pipe_ms = vk.create_aaa_compute_pipeline(pipe_layout, sota_atmo_multiscatter_shader())
        self.pipe_sky = vk.create_aaa_compute_pipeline(pipe_layout, sota_atmo_skyview_shader())
        self.pipe_ap = vk.create_aaa_compute_pipeline(pipe_layout, sota_atmo_aerial_shader())
        self.pipe_apply = vk.create_aaa_compute_pipeline(pipe_layout, sota_atmo_apply_shader())
        self.ibl.create(vk, pipe_layout)
        self.clouds.create(vk, pipe_layout)
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
        self.ibl.destroy(vk)
        self.clouds.destroy(vk)

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

    def configure_region(mut self, base: Int, max_pix: Int, env_dim: Int, env_mips: Int) -> Int:
        """Places the area-H region at word `base` of the history buffer; returns its size in words."""
        self.region_base = base
        self.max_pix = max_pix
        self.ibl.base_dim = env_dim
        self.ibl.mips = env_mips
        self.ibl.region_base = (self.ap_off() + sota_atmo_ap_words() + 3) & ~3
        return sota_atmo_region_words(max_pix, env_dim, env_mips)

    def cloud_base_off(self) -> Int:
        return self.ibl.region_base + ibl_region_words(self.ibl.base_dim, self.ibl.mips)

    def cloud_detail_off(self) -> Int:
        return self.cloud_base_off() + SOTA_CLOUD_BASE_N * SOTA_CLOUD_BASE_N * SOTA_CLOUD_BASE_N

    def cloud_weather_off(self) -> Int:
        return self.cloud_detail_off() + SOTA_CLOUD_DETAIL_N * SOTA_CLOUD_DETAIL_N * SOTA_CLOUD_DETAIL_N

    def cloud_image_off(self, parity: Int) -> Int:
        return self.cloud_weather_off() + SOTA_CLOUD_WEATHER_N * SOTA_CLOUD_WEATHER_N * 2 + (parity & 1) * cloud_buffer_words(self.max_pix)

    def cloud_cur_off(self) -> Int:
        return self.cloud_image_off(self.clouds.parity)

    def clouds_active(self) -> Bool:
        return self.clouds.enabled and self.clouds.ready() and self.max_pix > 0 and self.sky

    def gpu_ibl_active(self) -> Bool:
        return self.update_ibl and self.ibl_gpu and self.ibl.ready()

    def read_gpu_sh9(self, hist_addr: Int) -> List[Vec3]:
        """The SH9 the GPU wrote last (`b_globals[72..99)` of that frame), from the history region."""
        var out = List[Vec3](length=9, fill=Vec3(0.0, 0.0, 0.0))
        var a = hist_addr + self.ibl.sh_off() * 4
        var p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=a)
        for k in range(9):
            out[k] = Vec3(p[unsafe_offset=k * 3], p[unsafe_offset=k * 3 + 1], p[unsafe_offset=k * 3 + 2])
        return out^

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
        var deg = self.ibl_gpu_update_deg if self.gpu_ibl_active() else self.ibl_update_deg
        var c = cos(deg * Float32(0.0174532925))
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
        sky pixels keep the v5 sky (only aerial perspective and fog run). Round 3 uses `write_globals_r3`."""
        self._write_globals_r2(g_addr, cam_y, legacy_sky_fx, froxel_on, fog_density, froxel_far_z, fog_falloff, fog_base_y, fog_albedo, fog_g)

    def write_globals_r3(
        mut self,
        g_addr: Int,
        cam_pos: Vec3,
        cam_fwd: Vec3,
        cam_right: Vec3,
        cam_up: Vec3,
        tan_x: Float32,
        tan_y: Float32,
        render_w: Int,
        render_h: Int,
        scene_time_sec: Float32,
        legacy_cloud_coverage: Float32,
        legacy_star_intensity: Float32,
        froxel_on: Bool,
        fog_density: Float32,
        froxel_far_z: Float32,
        fog_falloff: Float32,
        fog_base_y: Float32,
        fog_albedo: Vec3,
        fog_g: Float32,
    ):
        """Round 3: the r2 block plus the night sky and cloud block (`b_globals[SOTA_ATMO_G2 .. +64)`). The legacy v5
        clouds keep the v5 sky only while the volumetric clouds are off; the legacy stars only while the night sky
        is off."""
        var clouds_on = self.clouds_active()
        var legacy = (legacy_cloud_coverage > Float32(0.01) and not clouds_on) or (legacy_star_intensity > Float32(0.01) and not self.night.enabled)
        self._write_globals_r2(g_addr, cam_pos.y, legacy, froxel_on, fog_density, froxel_far_z, fog_falloff, fog_base_y, fog_albedo, fog_g)
        var g = SOTA_ATMO_G2
        var rows = world_to_equatorial_rows(self.time_of_day)
        _put_v3(g_addr, g + 0, rows[0])
        _put_v3(g_addr, g + 3, rows[1])
        _put_v3(g_addr, g + 6, rows[2])
        var nt = self.night.copy()
        var e0 = nt.e0(rgb_luma(self.params.solar_irradiance))
        _put_f(g_addr, g + SOTA_NIGHT_W_STARS, nt.star_brightness)
        _put_f(g_addr, g + SOTA_NIGHT_W_MLIM, nt.limiting_magnitude)
        _put_f(g_addr, g + SOTA_NIGHT_W_BG, surface_brightness_radiance(e0, nt.airglow_mag_arcsec2) * nt.airglow_scale)
        _put_f(g_addr, g + SOTA_NIGHT_W_MW, surface_brightness_radiance(e0, nt.milky_way_mag_arcsec2) * nt.milky_way_scale)
        _put_f(g_addr, g + SOTA_NIGHT_W_PSF, max(Float32(0.5), nt.star_psf_px))
        _put_u(g_addr, g + SOTA_NIGHT_W_SEED, nt.seed)
        _put_f(g_addr, g + SOTA_NIGHT_W_E0, e0)
        var nflags = UInt32(0)
        var sky_on = self.sky and not legacy
        if nt.enabled and sky_on:
            nflags |= 1
        if clouds_on and sky_on:
            nflags |= 2
        _put_u(g_addr, g + SOTA_NIGHT_W_FLAGS, nflags)
        if not clouds_on:
            self.clouds.hist_valid = False
            return
        # --- clouds ---
        var cp = self.clouds.params
        var wx = -self.clouds.wind_mps.x * scene_time_sec * Float32(0.001)
        var wz = -self.clouds.wind_mps.z * scene_time_sec * Float32(0.001)
        _put_f(g_addr, g + 20, cp.bottom_km)
        _put_f(g_addr, g + 21, max(cp.bottom_km + Float32(0.05), cp.top_km))
        _put_f(g_addr, g + 22, max(Float32(0.0), min(Float32(1.0), cp.coverage)))
        _put_f(g_addr, g + 23, cp.density_per_km)
        _put_f(g_addr, g + 24, wx)
        _put_f(g_addr, g + 25, wz)
        _put_f(g_addr, g + 26, max(Float32(0.1), cp.tile_km))
        _put_f(g_addr, g + 27, max(Float32(0.01), cp.base_scale_km))
        _put_f(g_addr, g + 28, max(Float32(0.01), cp.detail_scale_km))
        _put_f(g_addr, g + 29, cp.detail_strength)
        _put_u(g_addr, g + 30, UInt32(max(1, cp.steps)))
        _put_u(g_addr, g + 31, UInt32(max(0, cp.light_steps)))
        _put_f(g_addr, g + 32, cp.light_step_km)
        _put_f(g_addr, g + 33, cp.max_dist_km)
        _put_f(g_addr, g + 34, cp.g_fwd)
        _put_f(g_addr, g + 35, cp.g_back)
        _put_f(g_addr, g + 36, cp.g_mix)
        _put_f(g_addr, g + 37, cp.ms_a)
        _put_f(g_addr, g + 38, cp.ms_b)
        _put_f(g_addr, g + 39, cp.ms_c)
        _put_f(g_addr, g + 40, cp.ambient)
        _put_f(g_addr, g + 41, cp.type_bias)
        _put_f(g_addr, g + CW_ALPHA, max(Float32(0.01), min(Float32(1.0), self.clouds.temporal_alpha)))
        _put_u(g_addr, g + CW_FRAME, UInt32(self.clouds.frame & 0xFFFF))
        _put_f(g_addr, g + CW_DWIND_X, (wx - self.clouds.last_wind_x_km) / self.km_per_unit)
        _put_f(g_addr, g + CW_DWIND_Z, (wz - self.clouds.last_wind_z_km) / self.km_per_unit)
        _put_u(g_addr, g + CW_JITTER, UInt32(1) if self.clouds.jitter else UInt32(0))
        _put_u(g_addr, g + 47, UInt32(0) if self.clouds.aerial_perspective else UInt32(1))  # CW_NO_AP
        # Resolution: raise the divisor until the cloud image fits its region.
        var dv = self.clouds.div()
        var cap = cloud_buffer_words(self.max_pix) // 4
        while dv < 8 and ((render_w + dv - 1) // dv) * ((render_h + dv - 1) // dv) > cap:
            dv *= 2
        var cw = (render_w + dv - 1) // dv
        var ch = (render_h + dv - 1) // dv
        var valid = self.clouds.hist_valid and cw == self.clouds.last_cw and ch == self.clouds.last_ch
        for k in range(14):
            _put_f(g_addr, g + CW_PREV_CAM + k, self.clouds.prev_cam[k])
        _put_u(g_addr, g + CW_HIST_VALID, UInt32(1) if valid else UInt32(0))
        _put_u(g_addr, g + CW_DIV, UInt32(dv))
        self.cloud_cw = cw
        self.cloud_ch = ch
        self.cloud_div_used = dv
        # The next frame's history: this frame's camera, wind and image size.
        var pc: List[Float32] = [cam_pos.x, cam_pos.y, cam_pos.z, cam_fwd.x, cam_fwd.y, cam_fwd.z, cam_right.x, cam_right.y,
                                 cam_right.z, cam_up.x, cam_up.y, cam_up.z, tan_x, tan_y]
        for k in range(14):
            self.clouds.prev_cam[k] = pc[k]
        self.clouds.last_wind_x_km = wx
        self.clouds.last_wind_z_km = wz
        self.clouds.last_cw = cw
        self.clouds.last_ch = ch

    def _write_globals_r2(
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
    push[5] = UInt32(atmo.cloud_cur_off()) if atmo.clouds_active() else UInt32(0)
    push[6] = UInt32(max(1, atmo.cloud_cw))
    push[7] = UInt32(max(1, atmo.cloud_ch))
    push[8] = UInt32(atmo.cloud_div_used)
    vk.dispatch_aaa(slot, atmo.pipe_apply, pipe_layout, desc_set, push, (render_w + 7) // 8, (render_h + 7) // 8)
    _barrier(vk, slot)
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q + 1, PIPELINE_BOTTOM)


def sota_atmo_record_ibl(
    mut atmo: SotaAtmosphere,
    mut vk: VulkanGpu,
    slot: Int,
    pipe_layout: Int,
    desc_set: Int,
    globals_buf: GpuBuffer,
    history_buf: GpuBuffer,
    mut env_cube: VulkanImage,
    use_prof: Bool,
    mut next_q: Int,
    q_limit: Int,
    mut prof_pairs: List[Tuple[String, Int]],
):
    """GPU IBL (after the sky-view LUT): SH9 every frame into `b_globals[72..99)`, the BC6H GGX cube when a light
    moved more than `ibl_gpu_update_deg` (flagged by `_sota_atmo_prepare`)."""
    if not atmo.gpu_ibl_active():
        return
    var q = next_q
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q, PIPELINE_BOTTOM)
        next_q += 2
        prof_pairs.append((String("SotaAtmoIbl"), q))
    atmo.ibl.record(vk, slot, pipe_layout, desc_set, atmo.sky_off(), globals_buf, history_buf, env_cube, atmo.ibl_cube_pending)
    atmo.ibl_cube_pending = False
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q + 1, PIPELINE_BOTTOM)


def sota_atmo_record_clouds(
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
    """Cloud noise (once), weather map (on seed change), then the march + reprojection into this frame's image.
    Runs after the HDR buffer holds this frame's depth (sky test) and before the apply pass."""
    if not atmo.clouds_active():
        return
    var q = next_q
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q, PIPELINE_BOTTOM)
        next_q += 2
        prof_pairs.append((String("SotaAtmoClouds"), q))
    if not atmo.clouds.noise_ready:
        var pn = List[UInt32](length=16, fill=0)
        pn[0] = 0
        pn[1] = UInt32(atmo.cloud_base_off())
        vk.dispatch_aaa(slot, atmo.clouds.pipe_noise, pipe_layout, desc_set, pn, SOTA_CLOUD_BASE_N // 8, SOTA_CLOUD_BASE_N // 8)
        var pd = List[UInt32](length=16, fill=0)
        pd[0] = 1
        pd[1] = UInt32(atmo.cloud_detail_off())
        vk.dispatch_aaa(slot, atmo.clouds.pipe_noise, pipe_layout, desc_set, pd, SOTA_CLOUD_DETAIL_N // 8, SOTA_CLOUD_DETAIL_N // 8)
        atmo.clouds.noise_ready = True
    if atmo.clouds.weather_seed_built != atmo.clouds.weather_seed or atmo.clouds.pipe_weather == 0:
        if atmo.clouds.pipe_weather != 0:
            vk.destroy_pipeline(atmo.clouds.pipe_weather)
        atmo.clouds.pipe_weather = vk.create_aaa_compute_pipeline(pipe_layout, sota_cloud_weather_shader(atmo.clouds.weather_seed))
        atmo.clouds.weather_seed_built = atmo.clouds.weather_seed
        atmo.clouds.weather_ready = False
    if not atmo.clouds.weather_ready and atmo.clouds.pipe_weather != 0:
        var pw = List[UInt32](length=16, fill=0)
        pw[0] = UInt32(atmo.cloud_weather_off())
        vk.dispatch_aaa(slot, atmo.clouds.pipe_weather, pipe_layout, desc_set, pw, SOTA_CLOUD_WEATHER_N // 8, SOTA_CLOUD_WEATHER_N // 8)
        atmo.clouds.weather_ready = True
    _barrier(vk, slot)
    atmo.clouds.parity = atmo.clouds.parity ^ 1
    var push = List[UInt32](length=16, fill=0)
    push[0] = UInt32(atmo.cloud_cw)
    push[1] = UInt32(atmo.cloud_ch)
    push[2] = UInt32(atmo.cloud_div_used)
    push[3] = UInt32(atmo.cloud_image_off(atmo.clouds.parity))
    push[4] = UInt32(atmo.cloud_image_off(atmo.clouds.parity ^ 1))
    push[5] = UInt32(atmo.trans_off())
    push[6] = UInt32(atmo.ap_off())
    push[7] = UInt32(atmo.cloud_base_off())
    push[8] = UInt32(atmo.cloud_detail_off())
    push[9] = UInt32(atmo.cloud_weather_off())
    push[10] = UInt32(render_w)
    push[11] = UInt32(render_h)
    push[12] = bitcast[DType.uint32](atmo.ap_max_km)
    vk.dispatch_aaa(slot, atmo.clouds.pipe_march, pipe_layout, desc_set, push, (atmo.cloud_cw + 7) // 8, (atmo.cloud_ch + 7) // 8)
    _barrier(vk, slot)
    atmo.clouds.frame += 1
    atmo.clouds.hist_valid = True
    if use_prof and q + 1 < q_limit:
        vk.write_timestamp(slot, q + 1, PIPELINE_BOTTOM)
