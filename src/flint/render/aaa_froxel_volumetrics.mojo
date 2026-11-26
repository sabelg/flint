"""Commercial Engine v2 (Pillar II): `160 x 90 x 64` 2-Pass Temporal Froxel Volumetric
Scattering Grid, `VSM` Sun God-Rays, Clustered Spot/`LTC` Cones & GPU Froxel-Lit 3D Rain (`aaa_froxel_volumetrics.mojo`).

Implements:
  1. `froxel_inject_and_reproject_shader` (`160 x 90 x 64` `FP16x4` camera-aligned exponential froxels,
     `10:1` XY-aligned with the `16 x 9` light clusters):
     - Height-exponential + wind-advected 3D FBM mist density
     - `1`-tap `VSM` sun god-rays + Cornette-Shanks forward Mie phase (`g = 0.72`)
     - Parent-cluster `Spot` cones (`Spinner` searchlights, headlights) & `LTC` neon tube/rect mist halos
     - `90%` 3D world-reprojected temporal filtering!
  2. `froxel_raymarch_integrate_shader` (`render_w x render_h`):
     - 64-slice Sébastien Hillaire analytical front-to-back transmittance/scattering integration
       composited smoothly over both opaque surfaces and sky background (`z >= 1e5`)
     - Optional stateless **GPU Froxel-Lit 3D Rain Streaks** stretched by `(v_rain - v_cam)`,
       depth-tested against `b_hdr[+3]` & VSM roof occlusion, and illuminated directly from the Froxel LUT!
"""
from std.math import exp, max, min, sqrt
from flint.core.math import Vec3
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_PUSH_WORDS,
    BIND_AAA_CLUSTERS,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    BIND_AAA_OIT_NODES,
    BIND_AAA_VSM_TABLE,
    BIND_AAA_VSM_TEX,
    emit_mat4_mul_point,
    emit_vec3_dot,
    emit_vec3_normalize,
)
from flint.render.aaa_vsm import emit_sample_vsm_depth
from flint.render.sota_shadow_vsm import emit_sota_shadow_cascade_uv  # --- sota:shadow hook ---
from flint.render.aaa_ibl import emit_eval_sh9_irradiance
from flint.render.aaa_clustered_lights import (
    AAA_CLUSTER_COUNTS_BASE,
    AAA_CLUSTER_INDICES_BASE,
    AAA_CLUSTER_LIGHTS_BASE,
    AAA_CLUSTER_LIGHT_WORDS,
    AAA_LIGHT_LTC_RECT,
    AAA_LIGHT_LTC_TUBE,
    AAA_LIGHT_SPOT,
    AAA_MAX_LIGHTS_PER_CLUSTER,
)
from flint.render.aaa_oit_volumetrics import AAA_OIT_REACTIVE_BASE_WORDS

comptime AAA_FROXEL_X: Int = 160
comptime AAA_FROXEL_Y: Int = 90
comptime AAA_FROXEL_Z: Int = 64
comptime AAA_FROXEL_SLICE_STRIDE: Int = AAA_FROXEL_X * AAA_FROXEL_Y  # 14,400
comptime AAA_FROXEL_TOTAL: Int = AAA_FROXEL_SLICE_STRIDE * AAA_FROXEL_Z  # 921,600
comptime AAA_FROXEL_GRID_WORDS: Int = AAA_FROXEL_TOTAL * 2  # 1,843,200 UInt32 words (7.03 MiB FP16x4)


@always_inline
def froxel_ping_word_offset(max_pix: Int) -> Int:
    return max_pix * 44


@always_inline
def froxel_pong_word_offset(max_pix: Int) -> Int:
    return max_pix * 44 + AAA_FROXEL_GRID_WORDS


@always_inline
def froxel_prefix_lut_word_offset(max_pix: Int) -> Int:
    """Word offset in `BIND_AAA_HISTORY` for the `160 x 90 x 64` `FP16x4` Prefix-Integrated
    `(L_accum.r, L_accum.g, L_accum.b, T_accum)` LUT (`+1,843,200` words = `+7.03 MiB`)."""
    return max_pix * 44 + AAA_FROXEL_GRID_WORDS * 2


def eval_cornette_shanks_phase_cpu(cos_theta: Float32, g: Float32) -> Float32:
    """Cornette-Shanks physical Mie scattering phase function normalized over `4*pi` steradians."""
    var gg = max(Float32(-0.92), min(Float32(0.92), g))
    var g2 = gg * gg
    var mu = max(Float32(-1.0), min(Float32(1.0), cos_theta))
    var num = Float32(3.0) * (Float32(1.0) - g2) * (Float32(1.0) + mu * mu)
    var den_b = max(Float32(0.02), Float32(1.0) + g2 - Float32(2.0) * gg * mu)
    var den = Float32(25.1327412) * (Float32(2.0) + g2) * den_b * sqrt(den_b)
    return num / max(Float32(1e-5), den)


def eval_hillaire_slice_integration_cpu(
    scatter_rgb: Vec3,
    sigma_ext: Float32,
    ds: Float32,
) -> Tuple[Vec3, Float32]:
    """Hillaire (SIGGRAPH 2015) analytical homogeneous slab integration returning `(S_integrated, T_slice)`."""
    var tau = max(Float32(0.0), sigma_ext * max(Float32(0.0), ds))
    var tr = exp(-tau)
    var integ = (Float32(1.0) - tr) / max(Float32(1e-5), sigma_ext) if sigma_ext > Float32(1e-5) else max(Float32(0.0), ds)
    return (scatter_rgb * integ, tr)


def _emit_cornette_shanks_spv(mut s: AaaSpirvModule, cos_th: UInt32, g_in: UInt32) -> UInt32:
    var g = s.nclamp(g_in, s.float(-0.92), s.float(0.92))
    var g2 = s.fmul(g, g)
    var mu = s.nclamp(cos_th, s.float(-1.0), s.float(1.0))
    var num = s.fmul(
        s.fmul(s.float(3.0), s.fsub(s.float(1.0), g2)),
        s.fma(mu, mu, s.float(1.0)),
    )
    var den_b = s.nmax(s.float(0.02), s.fsub(s.fadd(s.float(1.0), g2), s.fmul(s.float(2.0), s.fmul(g, mu))))
    var den = s.fmul(
        s.fmul(s.float(25.1327412), s.fadd(s.float(2.0), g2)),
        s.fmul(den_b, s.fsqrt(den_b)),
    )
    return s.fdiv(num, s.nmax(s.float(1e-5), den))


def froxel_inject_and_reproject_shader() -> List[UInt32]:
    """Pass 1 (`160 x 90 x 64`, `group = 8 x 6 x 1`, dispatched `20 x 15 x 64`):
    Injects height-exponential + wind-advected 3D FBM density, `1`-tap `VSM` sun god-rays,
    and parent-cluster `Spot` cones / `LTC` tube & billboard mist halos, with `90%` 3D temporal reprojection!
    Push[0] = froxel_hist_in_off, Push[1] = froxel_hist_out_off, Push[2] = frame_idx.
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=6, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_vsm = s.buffer(BIND_AAA_VSM_TABLE)
    var b_clusters = s.buffer(BIND_AAA_CLUSTERS)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var vsm_samp = s.sampled_image_2d(BIND_AAA_VSM_TEX)

    var fx = s.invocation_axis(0)
    var fy = s.invocation_axis(1)
    var fz = s.invocation_axis(2)
    var hist_in_off = s.push_word(0)
    var hist_out_off = s.push_word(1)
    var frame_idx = s.push_word(2)
    s.return_if(
        s.logical_or(
            s.uge(fx, s.uint(UInt32(AAA_FROXEL_X))),
            s.logical_or(s.uge(fy, s.uint(UInt32(AAA_FROXEL_Y))), s.uge(fz, s.uint(UInt32(AAA_FROXEL_Z)))),
        )
    )

    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
    var near_z = s.nmax(s.float(0.10), s.load_float(b_globals, s.uint(35)))
    var fwd_x = s.load_float(b_globals, s.uint(36))
    var fwd_y = s.load_float(b_globals, s.uint(37))
    var fwd_z = s.load_float(b_globals, s.uint(38))
    var tan_x = s.load_float(b_globals, s.uint(39))
    var rgt_x = s.load_float(b_globals, s.uint(40))
    var rgt_y = s.load_float(b_globals, s.uint(41))
    var rgt_z = s.load_float(b_globals, s.uint(42))
    var tan_y = s.load_float(b_globals, s.uint(43))
    var up_x = s.load_float(b_globals, s.uint(44))
    var up_y = s.load_float(b_globals, s.uint(45))
    var up_z = s.load_float(b_globals, s.uint(46))
    var sun_x = s.load_float(b_globals, s.uint(48))
    var sun_y = s.load_float(b_globals, s.uint(49))
    var sun_z = s.load_float(b_globals, s.uint(50))
    var sun_r = s.load_float(b_globals, s.uint(52))
    var sun_g = s.load_float(b_globals, s.uint(53))
    var sun_b = s.load_float(b_globals, s.uint(54))
    var ibl_int = s.load_float(b_globals, s.uint(55))
    var r_flags = s.load_word(b_globals, s.uint(99))

    var prev_cam_x = s.load_float(b_globals, s.uint(120))
    var prev_cam_y = s.load_float(b_globals, s.uint(121))
    var prev_cam_z = s.load_float(b_globals, s.uint(122))
    var time_s = s.load_float(b_globals, s.uint(123))
    var wind_x = s.load_float(b_globals, s.uint(128))
    var wind_y = s.load_float(b_globals, s.uint(129))
    var wind_z = s.load_float(b_globals, s.uint(130))
    var far_frox = s.nmax(s.fadd(near_z, s.float(5.0)), s.load_float(b_globals, s.uint(131)))
    var base_dens = s.nmax(s.float(0.0), s.load_float(b_globals, s.uint(132)))
    var h_falloff = s.nmax(s.float(0.0), s.load_float(b_globals, s.uint(133)))
    var mie_g = s.load_float(b_globals, s.uint(134))
    var base_y = s.load_float(b_globals, s.uint(135))
    var fog_ar = s.load_float(b_globals, s.uint(136))
    var fog_ag = s.load_float(b_globals, s.uint(137))
    var fog_ab = s.load_float(b_globals, s.uint(138))

    var log_ratio = s.log2(s.fdiv(far_frox, near_z))

    # Sub-slice temporal Halton/IGN jitter for artifact-free 64-slice god-ray shafts
    var f_fx = s.float_from_uint(fx)
    var f_fy = s.float_from_uint(fy)
    var f_fz = s.float_from_uint(fz)
    var f_fr = s.float_from_uint(s.bit_and(frame_idx, s.uint(31)))
    var ign_z = s.fract(s.fmul(s.float(52.9829189), s.fract(s.fma(f_fx, s.float(0.06711056), s.fma(f_fy, s.float(0.00583715), s.fmul(f_fr, s.float(0.61803399)))))))
    var jit_slice = s.fma(ign_z, s.float(0.75), s.float(0.125))
    var s_norm = s.fmul(s.fadd(f_fz, jit_slice), s.float(1.0 / 64.0))
    var z_view = s.fmul(near_z, s.exp2(s.fmul(log_ratio, s_norm)))

    var u_f = s.fmul(s.fadd(f_fx, s.float(0.5)), s.float(1.0 / 160.0))
    var v_f = s.fmul(s.fadd(f_fy, s.float(0.5)), s.float(1.0 / 90.0))
    var ndc_x = s.fma(u_f, s.float(2.0), s.float(-1.0))
    var ndc_y = s.fsub(s.float(1.0), s.fmul(v_f, s.float(2.0)))

    var vx_cam = s.fmul(s.fmul(ndc_x, tan_x), z_view)
    var vy_cam = s.fmul(s.fmul(ndc_y, tan_y), z_view)
    var wx = s.fma(rgt_x, vx_cam, s.fma(up_x, vy_cam, s.fma(fwd_x, z_view, cam_x)))
    var wy = s.fma(rgt_y, vx_cam, s.fma(up_y, vy_cam, s.fma(fwd_y, z_view, cam_y)))
    var wz = s.fma(rgt_z, vx_cam, s.fma(up_z, vy_cam, s.fma(fwd_z, z_view, cam_z)))

    var view_dir = emit_vec3_normalize(s, s.fsub(wx, cam_x), s.fsub(wy, cam_y), s.fsub(wz, cam_z))

    # 1. Height-Exponential + Wind-Advected 3D FBM Mist Density
    var rel_h = s.nmax(s.float(0.0), s.fsub(wy, base_y))
    var h_atten = s.exp2(s.fmul(s.fmul(rel_h, h_falloff), s.float(-1.44269504)))
    var px_w = s.fma(wind_x, time_s, wx)
    var py_w = s.fma(wind_y, time_s, wy)
    var pz_w = s.fma(wind_z, time_s, wz)
    var fb1 = s.fmul(s.fsin(s.fma(px_w, s.float(0.11), s.fmul(pz_w, s.float(0.09)))), s.fcos(s.fma(py_w, s.float(0.17), s.fmul(px_w, s.float(-0.07)))))
    var fb2 = s.fmul(s.fsin(s.fma(px_w, s.float(0.29), s.fmul(py_w, s.float(0.23)))), s.fcos(s.fma(pz_w, s.float(0.31), s.float(1.1))))
    var noise_m = s.nclamp(s.fma(fb1, s.float(0.28), s.fma(fb2, s.float(0.16), s.float(0.92))), s.float(0.35), s.float(1.45))
    # Foreground crispness ramp `smoothstep(1.5, 14.0, z_view)` keeps hero cars & coaster trains razor-sharp!
    var fg_t = s.nclamp(s.fmul(s.fsub(z_view, s.float(1.5)), s.float(1.0 / 12.5)), s.float(0.0), s.float(1.0))
    var fg_ramp = s.fmul(fg_t, s.fmul(fg_t, s.fma(s.float(-2.0), fg_t, s.float(3.0))))
    var sigma_ext = s.fmul(base_dens, s.fmul(h_atten, s.fmul(noise_m, fg_ramp)))

    # 2. Ambient SH9 Sky Mist + `1`-Tap `VSM` Sun God-Rays!
    var sh_amb = emit_eval_sh9_irradiance(s, b_globals, 72, view_dir[0], view_dir[1], view_dir[2])
    var amb_k = s.fmul(ibl_int, s.float(0.28))
    var lin_r = s.fvar(s.fmul(sh_amb[0], amb_k))
    var lin_g = s.fvar(s.fmul(sh_amb[1], amb_k))
    var lin_b = s.fvar(s.fmul(sh_amb[2], amb_k))

    var sun_vis = s.fvar(s.float(1.0))
    s.begin_if(s.ine(s.bit_and(r_flags, s.uint(1)), s.uint(0)))
    # --- sota:shadow hook --- cascade-aware + toroidal lookup (v5 used cascade 0's matrix on the whole table)
    var cuv = emit_sota_shadow_cascade_uv(s, b_globals, wx, wy, wz)
    var lu = cuv[2]
    var lv = cuv[3]
    var in_vsm = s.logical_and(
        s.logical_and(s.fgt(lu, s.float(0.002)), s.flt(lu, s.float(0.998))),
        s.logical_and(s.fgt(lv, s.float(0.002)), s.flt(lv, s.float(0.998))),
    )
    s.begin_if(s.logical_and(in_vsm, s.flt(cuv[4], s.float(0.999))))
    var vz_s = emit_sample_vsm_depth(s, b_vsm, vsm_samp, cuv[0], cuv[1], b_globals)
    s.store(sun_vis, s.select_f(s.flt(vz_s, s.fsub(cuv[4], s.float(0.0015))), s.float(0.0), s.float(1.0)))
    # --- end sota:shadow hook ---
    s.end_if()
    s.end_if()

    var cos_sun = emit_vec3_dot(s, view_dir[0], view_dir[1], view_dir[2], sun_x, sun_y, sun_z)
    var ph_sun = _emit_cornette_shanks_spv(s, cos_sun, mie_g)
    var sun_sc = s.fmul(s.load_f(sun_vis), ph_sun)
    s.store(lin_r, s.fma(sun_r, sun_sc, s.load_f(lin_r)))
    s.store(lin_g, s.fma(sun_g, sun_sc, s.load_f(lin_g)))
    s.store(lin_b, s.fma(sun_b, sun_sc, s.load_f(lin_b)))

    # 3. Parent-Cluster `Spot` Cones (`Spinner` searchlights / headlights) & `LTC` Neon Mist Halos!
    s.begin_if(s.ine(s.bit_and(r_flags, s.uint(32)), s.uint(0)))
    var total_cl = s.load_word(b_clusters, s.uint(0))
    s.begin_if(s.ine(total_cl, s.uint(0)))
    var near_cl = s.load_float(b_clusters, s.uint(1))
    var log_cl = s.load_float(b_clusters, s.uint(3))
    var cx_p = s.umin(s.uint(15), s.udiv(fx, s.uint(10)))
    var cy_p = s.umin(s.uint(8), s.udiv(fy, s.uint(10)))
    var zn_cl = s.nclamp(
        s.fdiv(s.log2(s.nmax(s.float(1.0), s.fdiv(z_view, near_cl))), s.nmax(s.float(1e-4), log_cl)),
        s.float(0.0),
        s.float(0.9999),
    )
    var cz_p = s.umin(s.uint(23), s.saturate_uint(s.fmul(zn_cl, s.float(24.0))))
    var cid = s.iadd(s.imul(cz_p, s.uint(144)), s.iadd(s.imul(cy_p, s.uint(16)), cx_p))

    var c_cnt = s.umin(s.load_word(b_clusters, s.iadd(s.uint(UInt32(AAA_CLUSTER_COUNTS_BASE)), cid)), s.uint(16))
    var l_list = s.iadd(s.uint(UInt32(AAA_CLUSTER_INDICES_BASE)), s.imul(cid, s.uint(UInt32(AAA_MAX_LIGHTS_PER_CLUSTER))))

    var ki_v = s.uvar(s.uint(0))
    s.begin_loop()
    var ki = s.load_u(ki_v)
    s.break_if(s.uge(ki, c_cnt))
    var li = s.load_word(b_clusters, s.iadd(l_list, ki))
    var l_off = s.iadd(s.uint(UInt32(AAA_CLUSTER_LIGHTS_BASE)), s.imul(li, s.uint(UInt32(AAA_CLUSTER_LIGHT_WORDS))))

    var lpx = s.load_float(b_clusters, l_off)
    var lpy = s.load_float(b_clusters, s.iadd(l_off, s.uint(1)))
    var lpz = s.load_float(b_clusters, s.iadd(l_off, s.uint(2)))
    var lrad = s.load_float(b_clusters, s.iadd(l_off, s.uint(3)))
    var lrr = s.load_float(b_clusters, s.iadd(l_off, s.uint(4)))
    var lrg = s.load_float(b_clusters, s.iadd(l_off, s.uint(5)))
    var lrb = s.load_float(b_clusters, s.iadd(l_off, s.uint(6)))
    var kind = s.bit_and(s.load_word(b_clusters, s.iadd(l_off, s.uint(7))), s.uint(0xFF))
    var aux_x = s.load_float(b_clusters, s.iadd(l_off, s.uint(8)))
    var aux_y = s.load_float(b_clusters, s.iadd(l_off, s.uint(9)))
    var aux_z = s.load_float(b_clusters, s.iadd(l_off, s.uint(10)))
    var p0_f = s.load_float(b_clusters, s.iadd(l_off, s.uint(11)))
    var avx_x = s.load_float(b_clusters, s.iadd(l_off, s.uint(12)))
    var avx_y = s.load_float(b_clusters, s.iadd(l_off, s.uint(13)))
    var avx_z = s.load_float(b_clusters, s.iadd(l_off, s.uint(14)))
    var p1_f = s.load_float(b_clusters, s.iadd(l_off, s.uint(15)))

    var tx_e = s.fvar(lpx); var ty_e = s.fvar(lpy); var tz_e = s.fvar(lpz)
    var emit_gain = s.fvar(s.float(1.0))

    s.begin_if(s.ieq(kind, s.uint(AAA_LIGHT_LTC_TUBE)))
    var cwx = s.fsub(wx, lpx); var cwy = s.fsub(wy, lpy); var cwz = s.fsub(wz, lpz)
    var ul2 = s.nmax(s.float(1e-5), emit_vec3_dot(s, aux_x, aux_y, aux_z, aux_x, aux_y, aux_z))
    var ts = s.nclamp(s.fdiv(emit_vec3_dot(s, cwx, cwy, cwz, aux_x, aux_y, aux_z), ul2), s.float(-1.0), s.float(1.0))
    s.store(tx_e, s.fma(aux_x, ts, lpx))
    s.store(ty_e, s.fma(aux_y, ts, lpy))
    s.store(tz_e, s.fma(aux_z, ts, lpz))
    s.store(emit_gain, s.nmax(s.float(0.35), s.fmul(s.fsqrt(ul2), p1_f)))
    s.end_if()

    s.begin_if(s.ieq(kind, s.uint(AAA_LIGHT_LTC_RECT)))
    var cwx2 = s.fsub(wx, lpx); var cwy2 = s.fsub(wy, lpy); var cwz2 = s.fsub(wz, lpz)
    var ul2_r = s.nmax(s.float(1e-5), emit_vec3_dot(s, aux_x, aux_y, aux_z, aux_x, aux_y, aux_z))
    var vl2_r = s.nmax(s.float(1e-5), emit_vec3_dot(s, avx_x, avx_y, avx_z, avx_x, avx_y, avx_z))
    var tu_r = s.nclamp(s.fdiv(emit_vec3_dot(s, cwx2, cwy2, cwz2, aux_x, aux_y, aux_z), ul2_r), s.float(-1.0), s.float(1.0))
    var tv_r = s.nclamp(s.fdiv(emit_vec3_dot(s, cwx2, cwy2, cwz2, avx_x, avx_y, avx_z), vl2_r), s.float(-1.0), s.float(1.0))
    s.store(tx_e, s.fma(aux_x, tu_r, s.fma(avx_x, tv_r, lpx)))
    s.store(ty_e, s.fma(aux_y, tu_r, s.fma(avx_y, tv_r, lpy)))
    s.store(tz_e, s.fma(aux_z, tu_r, s.fma(avx_z, tv_r, lpz)))
    s.store(emit_gain, s.nmin(s.float(8.0), s.fsqrt(s.fmul(ul2_r, vl2_r))))
    s.end_if()

    var dlx = s.fsub(s.load_f(tx_e), wx)
    var dly = s.fsub(s.load_f(ty_e), wy)
    var dlz = s.fsub(s.load_f(tz_e), wz)
    var d2 = s.nmax(s.float(0.08), emit_vec3_dot(s, dlx, dly, dlz, dlx, dly, dlz))
    var r2 = s.nmax(s.float(0.04), s.fmul(lrad, lrad))
    s.begin_if(s.flt(d2, r2))
    var inv_d = s.inverse_sqrt(d2)
    var lx = s.fmul(dlx, inv_d); var ly = s.fmul(dly, inv_d); var lz = s.fmul(dlz, inv_d)
    var q = s.fdiv(d2, r2)
    var q2 = s.nclamp(s.fsub(s.float(1.0), s.fmul(q, q)), s.float(0.0), s.float(1.0))
    var att = s.fvar(s.fdiv(s.fmul(q2, q2), s.fadd(d2, s.float(0.45))))

    s.begin_if(s.ieq(kind, s.uint(AAA_LIGHT_SPOT)))
    var cos_c = s.fneg(emit_vec3_dot(s, lx, ly, lz, aux_x, aux_y, aux_z))
    var sw = s.nclamp(s.fmul(s.fsub(cos_c, p0_f), p1_f), s.float(0.0), s.float(1.0))
    s.store(att, s.fmul(s.load_f(att), s.fmul(sw, sw)))
    s.end_if()

    var cos_loc = emit_vec3_dot(s, view_dir[0], view_dir[1], view_dir[2], lx, ly, lz)
    var ph_loc = _emit_cornette_shanks_spv(s, cos_loc, s.fmul(mie_g, s.float(0.75)))
    var w_loc = s.fmul(s.load_f(att), s.fmul(ph_loc, s.load_f(emit_gain)))
    s.store(lin_r, s.fma(lrr, w_loc, s.load_f(lin_r)))
    s.store(lin_g, s.fma(lrg, w_loc, s.load_f(lin_g)))
    s.store(lin_b, s.fma(lrb, w_loc, s.load_f(lin_b)))
    s.end_if()

    s.store(ki_v, s.iadd(ki, s.uint(1)))
    s.end_loop()
    s.end_if()
    s.end_if()

    # Pre-multiply in-scattering source `S_rgb = sigma_ext * fog_albedo * L_in`
    var cur_sr = s.fvar(s.fmul(sigma_ext, s.fmul(fog_ar, s.load_f(lin_r))))
    var cur_sg = s.fvar(s.fmul(sigma_ext, s.fmul(fog_ag, s.load_f(lin_g))))
    var cur_sb = s.fvar(s.fmul(sigma_ext, s.fmul(fog_ab, s.load_f(lin_b))))
    var cur_se = s.fvar(sigma_ext)

    # 4. `90%` 3D World-Space Temporal Reprojection from `hist_in_off`
    s.begin_if(s.ine(frame_idx, s.uint(0)))
    # Unjittered froxel center world pos for stable temporal lookup
    var s_mid = s.fmul(s.fadd(f_fz, s.float(0.5)), s.float(1.0 / 64.0))
    var z_mid = s.fmul(near_z, s.exp2(s.fmul(log_ratio, s_mid)))
    var vx_m = s.fmul(s.fmul(ndc_x, tan_x), z_mid)
    var vy_m = s.fmul(s.fmul(ndc_y, tan_y), z_mid)
    var wx_m = s.fma(rgt_x, vx_m, s.fma(up_x, vy_m, s.fma(fwd_x, z_mid, cam_x)))
    var wy_m = s.fma(rgt_y, vx_m, s.fma(up_y, vy_m, s.fma(fwd_y, z_mid, cam_y)))
    var wz_m = s.fma(rgt_z, vx_m, s.fma(up_z, vy_m, s.fma(fwd_z, z_mid, cam_z)))

    var p_clip = emit_mat4_mul_point(s, b_globals, s.uint(16), wx_m, wy_m, wz_m)
    var prev_vz = p_clip[3]
    s.begin_if(s.logical_and(s.fgt(prev_vz, near_z), s.flt(prev_vz, far_frox)))
    var inv_pw = s.fdiv(s.float(1.0), prev_vz)
    var pu = s.fma(s.fmul(p_clip[0], inv_pw), s.float(0.5), s.float(0.5))
    var pv = s.fma(s.fneg(s.fmul(p_clip[1], inv_pw)), s.float(0.5), s.float(0.5))
    var in_p = s.logical_and(
        s.logical_and(s.fge(pu, s.float(0.0)), s.flt(pu, s.float(1.0))),
        s.logical_and(s.fge(pv, s.float(0.0)), s.flt(pv, s.float(1.0))),
    )
    s.begin_if(in_p)
    var pdx = s.fsub(wx_m, prev_cam_x); var pdy = s.fsub(wy_m, prev_cam_y); var pdz = s.fsub(wz_m, prev_cam_z)
    _ = pdx; _ = pdy; _ = pdz
    var ps_f = s.fmul(s.nclamp(s.fdiv(s.log2(s.fdiv(prev_vz, near_z)), log_ratio), s.float(0.0), s.float(0.9999)), s.float(64.0))
    var pfx = s.umin(s.uint(159), s.saturate_uint(s.fmul(pu, s.float(160.0))))
    var pfy = s.umin(s.uint(89), s.saturate_uint(s.fmul(pv, s.float(90.0))))
    var pfz0 = s.umin(s.uint(63), s.saturate_uint(s.ffloor(ps_f)))
    var pfz1 = s.umin(s.uint(63), s.iadd(pfz0, s.uint(1)))
    var tz_p = s.nclamp(s.fsub(ps_f, s.float_from_uint(pfz0)), s.float(0.0), s.float(1.0))

    var idx_xy = s.iadd(s.imul(pfy, s.uint(160)), pfx)
    var hb0 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(pfz0, s.uint(UInt32(AAA_FROXEL_SLICE_STRIDE))), idx_xy), s.uint(2)))
    var hb1 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(pfz1, s.uint(UInt32(AAA_FROXEL_SLICE_STRIDE))), idx_xy), s.uint(2)))

    var a0 = s.unpack_f16x2(s.load_word(b_hist, hb0))
    var a1 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(hb0, s.uint(1))))
    var b0_h = s.unpack_f16x2(s.load_word(b_hist, hb1))
    var b1_h = s.unpack_f16x2(s.load_word(b_hist, s.iadd(hb1, s.uint(1))))

    var hr = s.fma(s.fsub(b0_h[0], a0[0]), tz_p, a0[0])
    var hg = s.fma(s.fsub(b0_h[1], a0[1]), tz_p, a0[1])
    var hb = s.fma(s.fsub(b1_h[0], a1[0]), tz_p, a1[0])
    var he = s.fma(s.fsub(b1_h[1], a1[1]), tz_p, a1[1])

    var alpha_t = s.float(0.10)
    var om_at = s.float(0.90)
    s.store(cur_sr, s.fma(s.load_f(cur_sr), alpha_t, s.fmul(hr, om_at)))
    s.store(cur_sg, s.fma(s.load_f(cur_sg), alpha_t, s.fmul(hg, om_at)))
    s.store(cur_sb, s.fma(s.load_f(cur_sb), alpha_t, s.fmul(hb, om_at)))
    s.store(cur_se, s.fma(s.load_f(cur_se), alpha_t, s.fmul(he, om_at)))
    s.end_if()
    s.end_if()
    s.end_if()

    var frox_idx = s.iadd(s.imul(fz, s.uint(UInt32(AAA_FROXEL_SLICE_STRIDE))), s.iadd(s.imul(fy, s.uint(160)), fx))
    var out_off = s.iadd(hist_out_off, s.imul(frox_idx, s.uint(2)))
    s.store_word(b_hist, out_off, s.pack_f16x2(s.load_f(cur_sr), s.load_f(cur_sg)))
    s.store_word(b_hist, s.iadd(out_off, s.uint(1)), s.pack_f16x2(s.load_f(cur_sb), s.load_f(cur_se)))
    return s.finish()


def emit_sample_froxel_prefix_lut_spv(
    mut s: AaaSpirvModule,
    b_hist: UInt32,
    prefix_lut_off: UInt32,
    fx_f: UInt32,
    fy_f: UInt32,
    z_limit: UInt32,
    near_z: UInt32,
    log_ratio: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """`O(1)` 8-Tap Trilinear Fetch from the `160 x 90 x 64` Froxel Prefix-Integrated LUT!
    Returns `(L_accum.r, L_accum.g, L_accum.b, T_accum)` with exact `(0, 0, 0, 1.0)` at `z_limit <= near_z`."""
    var fx0 = s.umin(s.uint(159), s.saturate_uint(s.ffloor(fx_f)))
    var fy0 = s.umin(s.uint(89), s.saturate_uint(s.ffloor(fy_f)))
    var fx1 = s.umin(s.uint(159), s.iadd(fx0, s.uint(1)))
    var fy1 = s.umin(s.uint(89), s.iadd(fy0, s.uint(1)))
    var tx = s.nclamp(s.fsub(fx_f, s.float_from_uint(fx0)), s.float(0.0), s.float(1.0))
    var ty = s.nclamp(s.fsub(fy_f, s.float_from_uint(fy0)), s.float(0.0), s.float(1.0))
    var w00 = s.fmul(s.fsub(s.float(1.0), tx), s.fsub(s.float(1.0), ty))
    var w10 = s.fmul(tx, s.fsub(s.float(1.0), ty))
    var w01 = s.fmul(s.fsub(s.float(1.0), tx), ty)
    var w11 = s.fmul(tx, ty)

    var xy00 = s.iadd(s.imul(fy0, s.uint(160)), fx0)
    var xy10 = s.iadd(s.imul(fy0, s.uint(160)), fx1)
    var xy01 = s.iadd(s.imul(fy1, s.uint(160)), fx0)
    var xy11 = s.iadd(s.imul(fy1, s.uint(160)), fx1)

    var u_z = s.nclamp(
        s.fmul(s.fdiv(s.log2(s.nmax(s.float(1.0), s.fdiv(z_limit, near_z))), s.nmax(s.float(1e-4), log_ratio)), s.float(64.0)),
        s.float(0.0),
        s.float(64.0),
    )
    var in_first_slice = s.fle(u_z, s.float(1.0))
    var u_z_m1 = s.nmax(s.float(0.0), s.fsub(u_z, s.float(1.0)))
    var sz0 = s.umin(s.uint(63), s.saturate_uint(s.ffloor(u_z_m1)))
    var sz1 = s.select_u(in_first_slice, s.uint(0), s.umin(s.uint(63), s.iadd(sz0, s.uint(1))))
    var tz = s.select_f(
        in_first_slice,
        u_z,
        s.nclamp(s.fsub(u_z_m1, s.float_from_uint(sz0)), s.float(0.0), s.float(1.0)),
    )

    var sl1_off = s.imul(sz1, s.uint(UInt32(AAA_FROXEL_SLICE_STRIDE)))
    var b00 = s.iadd(prefix_lut_off, s.imul(s.iadd(sl1_off, xy00), s.uint(2)))
    var b10 = s.iadd(prefix_lut_off, s.imul(s.iadd(sl1_off, xy10), s.uint(2)))
    var b01 = s.iadd(prefix_lut_off, s.imul(s.iadd(sl1_off, xy01), s.uint(2)))
    var b11 = s.iadd(prefix_lut_off, s.imul(s.iadd(sl1_off, xy11), s.uint(2)))

    var q00_a = s.unpack_f16x2(s.load_word(b_hist, b00))
    var q00_b = s.unpack_f16x2(s.load_word(b_hist, s.iadd(b00, s.uint(1))))
    var q10_a = s.unpack_f16x2(s.load_word(b_hist, b10))
    var q10_b = s.unpack_f16x2(s.load_word(b_hist, s.iadd(b10, s.uint(1))))
    var q01_a = s.unpack_f16x2(s.load_word(b_hist, b01))
    var q01_b = s.unpack_f16x2(s.load_word(b_hist, s.iadd(b01, s.uint(1))))
    var q11_a = s.unpack_f16x2(s.load_word(b_hist, b11))
    var q11_b = s.unpack_f16x2(s.load_word(b_hist, s.iadd(b11, s.uint(1))))

    var r1 = s.fma(q00_a[0], w00, s.fma(q10_a[0], w10, s.fma(q01_a[0], w01, s.fmul(q11_a[0], w11))))
    var g1 = s.fma(q00_a[1], w00, s.fma(q10_a[1], w10, s.fma(q01_a[1], w01, s.fmul(q11_a[1], w11))))
    var bl1 = s.fma(q00_b[0], w00, s.fma(q10_b[0], w10, s.fma(q01_b[0], w01, s.fmul(q11_b[0], w11))))
    var tr1 = s.fma(q00_b[1], w00, s.fma(q10_b[1], w10, s.fma(q01_b[1], w01, s.fmul(q11_b[1], w11))))

    var r0 = s.fvar(s.float(0.0))
    var g0 = s.fvar(s.float(0.0))
    var bl0 = s.fvar(s.float(0.0))
    var tr0 = s.fvar(s.float(1.0))

    s.begin_if(s.logical_not(in_first_slice))
    var sl0_off = s.imul(sz0, s.uint(UInt32(AAA_FROXEL_SLICE_STRIDE)))
    var a00 = s.iadd(prefix_lut_off, s.imul(s.iadd(sl0_off, xy00), s.uint(2)))
    var a10 = s.iadd(prefix_lut_off, s.imul(s.iadd(sl0_off, xy10), s.uint(2)))
    var a01 = s.iadd(prefix_lut_off, s.imul(s.iadd(sl0_off, xy01), s.uint(2)))
    var a11 = s.iadd(prefix_lut_off, s.imul(s.iadd(sl0_off, xy11), s.uint(2)))

    var p00_a = s.unpack_f16x2(s.load_word(b_hist, a00))
    var p00_b = s.unpack_f16x2(s.load_word(b_hist, s.iadd(a00, s.uint(1))))
    var p10_a = s.unpack_f16x2(s.load_word(b_hist, a10))
    var p10_b = s.unpack_f16x2(s.load_word(b_hist, s.iadd(a10, s.uint(1))))
    var p01_a = s.unpack_f16x2(s.load_word(b_hist, a01))
    var p01_b = s.unpack_f16x2(s.load_word(b_hist, s.iadd(a01, s.uint(1))))
    var p11_a = s.unpack_f16x2(s.load_word(b_hist, a11))
    var p11_b = s.unpack_f16x2(s.load_word(b_hist, s.iadd(a11, s.uint(1))))

    s.store(r0, s.fma(p00_a[0], w00, s.fma(p10_a[0], w10, s.fma(p01_a[0], w01, s.fmul(p11_a[0], w11)))))
    s.store(g0, s.fma(p00_a[1], w00, s.fma(p10_a[1], w10, s.fma(p01_a[1], w01, s.fmul(p11_a[1], w11)))))
    s.store(bl0, s.fma(p00_b[0], w00, s.fma(p10_b[0], w10, s.fma(p01_b[0], w01, s.fmul(p11_b[0], w11)))))
    s.store(tr0, s.fma(p00_b[1], w00, s.fma(p10_b[1], w10, s.fma(p01_b[1], w01, s.fmul(p11_b[1], w11)))))
    s.end_if()

    var om_tz = s.fsub(s.float(1.0), tz)
    return (
        s.fma(s.load_f(r0), om_tz, s.fmul(r1, tz)),
        s.fma(s.load_f(g0), om_tz, s.fmul(g1, tz)),
        s.fma(s.load_f(bl0), om_tz, s.fmul(bl1, tz)),
        s.nclamp(s.fma(s.load_f(tr0), om_tz, s.fmul(tr1, tz)), s.float(0.0), s.float(1.0)),
    )


def froxel_prefix_integrate_shader() -> List[UInt32]:
    """Pass 1.5 (`160 x 90`, `group = 8 x 6`, `20 x 15` workgroups = `14,400` threads!):
    Marches `sz = 0..63` ONCE per froxel column using Sébastien Hillaire analytical slab integration
    and writes cumulative `(L_accum.rgb, T_accum)` (`FP16x4`) into `prefix_lut_off`, reclaiming `-1.95 ms`!
    Push[0] = froxel_grid_off, Push[1] = prefix_lut_off.
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=6, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hist = s.buffer(BIND_AAA_HISTORY)

    var fx = s.invocation_axis(0)
    var fy = s.invocation_axis(1)
    var grid_off = s.push_word(0)
    var prefix_off = s.push_word(1)
    s.return_if(s.logical_or(s.uge(fx, s.uint(UInt32(AAA_FROXEL_X))), s.uge(fy, s.uint(UInt32(AAA_FROXEL_Y)))))

    var near_z = s.nmax(s.float(0.10), s.load_float(b_globals, s.uint(35)))
    var far_frox = s.nmax(s.fadd(near_z, s.float(5.0)), s.load_float(b_globals, s.uint(131)))
    var log_ratio = s.log2(s.fdiv(far_frox, near_z))
    var tan_x = s.load_float(b_globals, s.uint(39))
    var tan_y = s.load_float(b_globals, s.uint(43))

    var u_f = s.fmul(s.fadd(s.float_from_uint(fx), s.float(0.5)), s.float(1.0 / 160.0))
    var v_f = s.fmul(s.fadd(s.float_from_uint(fy), s.float(0.5)), s.float(1.0 / 90.0))
    var nx_s = s.fmul(s.fma(u_f, s.float(2.0), s.float(-1.0)), tan_x)
    var ny_s = s.fmul(s.fsub(s.float(1.0), s.fmul(v_f, s.float(2.0))), tan_y)
    var ray_sec = s.fsqrt(s.fadd(s.float(1.0), s.fma(nx_s, nx_s, s.fmul(ny_s, ny_s))))

    var col_idx = s.iadd(s.imul(fy, s.uint(160)), fx)
    var acc_r = s.fvar(s.float(0.0))
    var acc_g = s.fvar(s.float(0.0))
    var acc_b = s.fvar(s.float(0.0))
    var trans = s.fvar(s.float(1.0))
    var z_prev = s.fvar(near_z)
    var sz_v = s.uvar(s.uint(0))

    s.begin_loop()
    var sz = s.load_u(sz_v)
    s.break_if(s.uge(sz, s.uint(64)))

    var s1_norm = s.fmul(s.float_from_uint(s.iadd(sz, s.uint(1))), s.float(1.0 / 64.0))
    var z_next = s.fmul(near_z, s.exp2(s.fmul(log_ratio, s1_norm)))
    var ds = s.fmul(s.nmax(s.float(0.0), s.fsub(z_next, s.load_f(z_prev))), ray_sec)
    s.store(z_prev, z_next)

    var cell_idx = s.iadd(s.imul(sz, s.uint(UInt32(AAA_FROXEL_SLICE_STRIDE))), col_idx)
    var src_w = s.iadd(grid_off, s.imul(cell_idx, s.uint(2)))
    var p_a = s.unpack_f16x2(s.load_word(b_hist, src_w))
    var p_b = s.unpack_f16x2(s.load_word(b_hist, s.iadd(src_w, s.uint(1))))
    var sr = p_a[0]
    var sg = p_a[1]
    var sb = p_b[0]
    var sig = s.nmax(s.float(0.0), p_b[1])

    var tau = s.fmul(sig, ds)
    var tr_s = s.exp2(s.fmul(tau, s.float(-1.44269504)))
    var has_sig = s.fgt(sig, s.float(1e-5))
    var integ = s.select_f(has_sig, s.fdiv(s.fsub(s.float(1.0), tr_s), s.nmax(s.float(1e-5), sig)), ds)
    var cur_tr = s.load_f(trans)
    var w_slice = s.fmul(cur_tr, integ)

    var nr = s.fma(sr, w_slice, s.load_f(acc_r))
    var ng = s.fma(sg, w_slice, s.load_f(acc_g))
    var nb = s.fma(sb, w_slice, s.load_f(acc_b))
    var ntr = s.fmul(cur_tr, tr_s)
    s.store(acc_r, nr)
    s.store(acc_g, ng)
    s.store(acc_b, nb)
    s.store(trans, ntr)

    var dst_w = s.iadd(prefix_off, s.imul(cell_idx, s.uint(2)))
    s.store_word(b_hist, dst_w, s.pack_f16x2(nr, ng))
    s.store_word(b_hist, s.iadd(dst_w, s.uint(1)), s.pack_f16x2(nb, ntr))

    s.store(sz_v, s.iadd(sz, s.uint(1)))
    s.end_loop()
    return s.finish()


def froxel_raymarch_integrate_shader() -> List[UInt32]:
    """Pass 2 (`render_w x render_h`, `group = 8 x 8`):
    Performs **1 single 8-tap trilinear fetch** (`32x` fewer SSBO reads!) from `froxel_prefix_integrate_shader`'s
    `FP16x4` Prefix-LUT at `(u, v, slice_of(min(z_surf, far_frox)))`, composites over `b_hdr[0..2]`,
    and renders optional stateless **GPU Froxel-Lit 3D Rain Streaks**!
    Push[0] = render_w, Push[1] = render_h, Push[2] = prefix_lut_off, Push[3] = enable_rain_u32, Push[4] = froxel_grid_off.
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var b_oit = s.buffer(BIND_AAA_OIT_NODES)
    var b_vsm = s.buffer(BIND_AAA_VSM_TABLE)
    var vsm_samp = s.sampled_image_2d(BIND_AAA_VSM_TEX)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var prefix_lut_off = s.push_word(2)
    var do_rain = s.ine(s.push_word(3), s.uint(0))
    var grid_off = s.push_word(4)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

    var pix_idx = s.iadd(s.imul(py, w_u), px)
    var h_base = s.imul(pix_idx, s.uint(8))
    var z_surf = s.load_float(b_hdr, s.iadd(h_base, s.uint(3)))

    var near_z = s.nmax(s.float(0.10), s.load_float(b_globals, s.uint(35)))
    var far_frox = s.nmax(s.fadd(near_z, s.float(5.0)), s.load_float(b_globals, s.uint(131)))
    var log_ratio = s.log2(s.fdiv(far_frox, near_z))

    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var fpx = s.float_from_uint(px)
    var fpy = s.float_from_uint(py)
    var frame_idx = s.load_word(b_globals, s.uint(100))
    var f_fr = s.float_from_uint(s.bit_and(frame_idx, s.uint(15)))

    # Sub-froxel IGN dither eliminates `160x90` bilinear grid banding on smooth shafts
    var ign = s.fract(s.fmul(s.float(52.9829189), s.fract(s.fma(fpx, s.float(0.06711056), s.fma(fpy, s.float(0.00583715), s.fmul(f_fr, s.float(0.381966)))))))
    var dith = s.fsub(ign, s.float(0.5))

    var u_c = s.fdiv(s.fadd(fpx, s.float(0.5)), fw)
    var v_c = s.fdiv(s.fadd(fpy, s.float(0.5)), fh)
    var fx_f = s.nclamp(s.fma(u_c, s.float(160.0), s.fma(dith, s.float(0.5), s.float(-0.5))), s.float(0.0), s.float(159.0))
    var fy_f = s.nclamp(s.fma(v_c, s.float(90.0), s.fma(dith, s.float(0.5), s.float(-0.5))), s.float(0.0), s.float(89.0))
    var fx0 = s.umin(s.uint(159), s.saturate_uint(s.ffloor(fx_f)))
    var fy0 = s.umin(s.uint(89), s.saturate_uint(s.ffloor(fy_f)))
    var xy00 = s.iadd(s.imul(fy0, s.uint(160)), fx0)

    var tan_x = s.load_float(b_globals, s.uint(39))
    var tan_y = s.load_float(b_globals, s.uint(43))
    var nx_s = s.fmul(s.fma(u_c, s.float(2.0), s.float(-1.0)), tan_x)
    var ny_s = s.fmul(s.fsub(s.float(1.0), s.fmul(v_c, s.float(2.0))), tan_y)

    var z_limit = s.nmin(z_surf, far_frox)
    var pref = emit_sample_froxel_prefix_lut_spv(
        s, b_hist, prefix_lut_off, fx_f, fy_f, z_limit, near_z, log_ratio,
    )
    var fin_tr = pref[3]
    var hdr_r = s.fvar(s.fma(s.load_float(b_hdr, h_base), fin_tr, pref[0]))
    var hdr_g = s.fvar(s.fma(s.load_float(b_hdr, s.iadd(h_base, s.uint(1))), fin_tr, pref[1]))
    var hdr_b = s.fvar(s.fma(s.load_float(b_hdr, s.iadd(h_base, s.uint(2))), fin_tr, pref[2]))

    # Stateless Zero-CPU GPU Froxel-Lit 3D Rain Streaks (`do_rain && rain_intensity > 0.005`)
    var rain_int = s.load_float(b_globals, s.uint(125))
    s.begin_if(s.logical_and(do_rain, s.fgt(rain_int, s.float(0.005))))
    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
    var fwd_x = s.load_float(b_globals, s.uint(36))
    var fwd_y = s.load_float(b_globals, s.uint(37))
    var fwd_z = s.load_float(b_globals, s.uint(38))
    var rgt_x = s.load_float(b_globals, s.uint(40))
    var rgt_y = s.load_float(b_globals, s.uint(41))
    var rgt_z = s.load_float(b_globals, s.uint(42))
    var up_x = s.load_float(b_globals, s.uint(44))
    var up_y = s.load_float(b_globals, s.uint(45))
    var up_z = s.load_float(b_globals, s.uint(46))
    var time_s = s.load_float(b_globals, s.uint(123))
    var wind_x = s.load_float(b_globals, s.uint(128))
    var wind_z = s.load_float(b_globals, s.uint(130))
    var cvel_x = s.load_float(b_globals, s.uint(153))
    var cvel_y = s.load_float(b_globals, s.uint(154))
    var cvel_z = s.load_float(b_globals, s.uint(155))
    var ibl_int = s.load_float(b_globals, s.uint(55))
    var r_flags = s.load_word(b_globals, s.uint(99))

    var rd = emit_vec3_normalize(
        s,
        s.fma(rgt_x, nx_s, s.fma(up_x, ny_s, fwd_x)),
        s.fma(rgt_y, nx_s, s.fma(up_y, ny_s, fwd_y)),
        s.fma(rgt_z, nx_s, s.fma(up_z, ny_s, fwd_z)),
    )
    # Relative rain velocity `v_rel = v_rain - v_cam` stretches streaks into diagonal motion lines at speed!
    var vrx = s.fsub(wind_x, cvel_x)
    var vry = s.fsub(s.float(-11.5), cvel_y)
    var vrz = s.fsub(wind_z, cvel_z)
    var vr_dir = emit_vec3_normalize(s, vrx, vry, vrz)
    var vr_speed = s.fsqrt(s.nmax(s.float(1.0), emit_vec3_dot(s, vrx, vry, vrz, vrx, vry, vrz)))
    var streak_len = s.nclamp(s.fmul(vr_speed, s.float(0.022)), s.float(0.18), s.float(0.85))

    var horiz_len = s.fsqrt(s.nmax(s.float(1e-4), s.fma(rd[0], rd[0], s.fmul(rd[2], rd[2]))))
    var rain_alpha_sum = s.fvar(s.float(0.0))

    s.begin_if(s.fgt(horiz_len, s.float(0.08)))
    var shells = [Float32(3.4), Float32(7.8), Float32(15.5)]
    var cols_n = [Float32(96.0), Float32(144.0), Float32(210.0)]
    for sh_i in range(3):
        var r_cyl = s.float(shells[sh_i])
        var t_cyl = s.fdiv(r_cyl, horiz_len)
        var z_cyl = s.fmul(t_cyl, emit_vec3_dot(s, rd[0], rd[1], rd[2], fwd_x, fwd_y, fwd_z))
        s.begin_if(s.logical_and(s.fgt(z_cyl, near_z), s.flt(z_cyl, z_surf)))
        var px_c = s.fma(rd[0], t_cyl, cam_x)
        var py_c = s.fma(rd[1], t_cyl, cam_y)
        var pz_c = s.fma(rd[2], t_cyl, cam_z)

        # Azimuthal cell around camera + falling vertical cell
        var ang = s.fatan2(rd[2], rd[0])
        var u_ang = s.fmul(s.fma(ang, s.float(0.15915494), s.float(0.5)), s.float(cols_n[sh_i]))
        var col_id = s.saturate_uint(s.nmax(s.float(0.0), s.ffloor(u_ang)))
        var fu_a = s.fsub(u_ang, s.ffloor(u_ang))

        # Per-column random fall phase + sparse occupancy
        var h_col = s.bit_xor(s.imul(col_id, s.uint(73856093)), s.uint(UInt32(sh_i * 19349663 + 17)))
        var r_occ = s.fmul(s.float_from_uint(s.bit_and(h_col, s.uint(0xFF))), s.float(1.0 / 255.0))
        var r_off = s.fmul(s.float_from_uint(s.bit_and(s.shr(h_col, s.uint(8)), s.uint(0xFF))), s.float(1.0 / 255.0))
        var r_jit = s.fmul(s.float_from_uint(s.bit_and(s.shr(h_col, s.uint(16)), s.uint(0xFF))), s.float(1.0 / 255.0))

        var y_scroll = s.fsub(s.fadd(py_c, s.fmul(r_off, s.float(4.0))), s.fmul(time_s, s.float(-11.5)))
        var v_cell = s.fmul(y_scroll, s.float(0.65))
        var fv_y = s.fract(v_cell)

        # Thin capsule width across azimuth + elongated vertical streak `[0.15 .. 0.15 + streak_norm]`
        var dx_c = s.fabs(s.fsub(fu_a, s.fma(r_jit, s.float(0.5), s.float(0.25))))
        var dy_c = s.fabs(s.fsub(fv_y, s.float(0.5)))
        var half_len = s.fmul(streak_len, s.float(0.42))
        var in_drop = s.logical_and(
            s.flt(r_occ, s.fmul(rain_int, s.float(0.65))),
            s.logical_and(s.flt(dx_c, s.float(0.065)), s.flt(dy_c, half_len)),
        )
        s.begin_if(in_drop)
        # Check overhead VSM occlusion so rain stops cleanly underneath highway overpasses!
        var sky_open = s.fvar(s.float(1.0))
        s.begin_if(s.ine(s.bit_and(r_flags, s.uint(1)), s.uint(0)))
        var rcuv = emit_sota_shadow_cascade_uv(s, b_globals, px_c, py_c, pz_c)  # --- sota:shadow hook ---
        var rlu = rcuv[2]
        var rlv = rcuv[3]
        s.begin_if(s.logical_and(s.logical_and(s.fgt(rlu, s.float(0.01)), s.flt(rlu, s.float(0.99))), s.logical_and(s.fgt(rlv, s.float(0.01)), s.flt(rlv, s.float(0.99)))))
        var rvz = emit_sample_vsm_depth(s, b_vsm, vsm_samp, rcuv[0], rcuv[1], b_globals)
        s.store(sky_open, s.select_f(s.flt(rvz, s.fsub(rcuv[4], s.float(0.008))), s.float(0.15), s.float(1.0)))
        s.end_if()
        s.end_if()

        var core_x = s.fsub(s.float(1.0), s.fmul(dx_c, s.float(15.38)))
        var core_y = s.fsub(s.float(1.0), s.fdiv(dy_c, s.nmax(s.float(0.05), half_len)))
        var drop_a = s.fmul(s.fmul(core_x, core_y), s.fmul(rain_int, s.fmul(s.load_f(sky_open), s.float(0.24))))

        # Sample live Froxel Scattering LUT at `z_cyl` so neon billboards & headlights tint falling rain!
        var sl_f = s.fmul(s.nclamp(s.fdiv(s.log2(s.nmax(s.float(1.0), s.fdiv(z_cyl, near_z))), log_ratio), s.float(0.0), s.float(0.9999)), s.float(64.0))
        var sl_u = s.umin(s.uint(63), s.saturate_uint(sl_f))
        var ro_f = s.iadd(grid_off, s.imul(s.iadd(s.imul(sl_u, s.uint(UInt32(AAA_FROXEL_SLICE_STRIDE))), xy00), s.uint(2)))
        var rf0 = s.unpack_f16x2(s.load_word(b_hist, ro_f))
        var rf1 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(ro_f, s.uint(1))))
        var inv_rsig = s.fdiv(s.float(1.8), s.nmax(s.float(0.004), rf1[1]))
        var amb_drop = s.fma(ibl_int, s.float(0.18), s.float(0.04))
        var lr_d = s.fma(rf0[0], inv_rsig, amb_drop)
        var lg_d = s.fma(rf0[1], inv_rsig, amb_drop)
        var lb_d = s.fma(rf1[0], inv_rsig, s.fmul(amb_drop, s.float(1.12)))
        _ = vr_dir

        s.store(hdr_r, s.fma(lr_d, drop_a, s.load_f(hdr_r)))
        s.store(hdr_g, s.fma(lg_d, drop_a, s.load_f(hdr_g)))
        s.store(hdr_b, s.fma(lb_d, drop_a, s.load_f(hdr_b)))
        s.store(rain_alpha_sum, s.nmin(s.float(0.45), s.fadd(s.load_f(rain_alpha_sum), drop_a)))
        s.end_if()
        s.end_if()
    s.end_if()

    s.begin_if(s.fgt(s.load_f(rain_alpha_sum), s.float(0.01)))
    var r_slot = s.iadd(s.uint(UInt32(AAA_OIT_REACTIVE_BASE_WORDS)), pix_idx)
    var cur_rc = s.load_float(b_oit, r_slot)
    s.store_float(b_oit, r_slot, s.nmax(cur_rc, s.load_f(rain_alpha_sum)))
    s.end_if()
    s.end_if()

    s.store_float(b_hdr, h_base, s.load_f(hdr_r))
    s.store_float(b_hdr, s.iadd(h_base, s.uint(1)), s.load_f(hdr_g))
    s.store_float(b_hdr, s.iadd(h_base, s.uint(2)), s.load_f(hdr_b))
    return s.finish()
