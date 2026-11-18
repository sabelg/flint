"""Commercial Engine v2 (Pillar I): `NRD ReBLUR`-Lite Dual Virtual-Depth Spatiotemporal
Specular Denoiser (`aaa_spec_reblur.mojo`).

Implements:
  1. `spec_reblur_temporal_shader`: `FP16x2`-packed ping-pong specular temporal accumulation in
     `BIND_AAA_HISTORY[28*max_pix .. 44*max_pix)` combining **Virtual-Depth Planar Reprojection**
     `P_virt = C + D * (d_cam + t_hit)` with surface motion vectors and Karis HDR firefly suppression.
  2. `spec_reblur_atrous_shader`: 2-Pass **Anisotropic View-Stretched GGX À-Trous** spatial filter
     whose screen radius scales with `t_hit * alpha_ggx / sqrt(hist_len)` (`0.0` blur on mirrors
     `roughness < 0.05` and contact points `t_hit -> 0`; realistic vertically stretched wet-road
     neon reflections on `0.06 .. 0.55`!), remodulating `mod_rgb * spec_rgb` into `b_hdr[0..2]`
     while strictly preserving `(z_lin, vel_u, vel_v, nx, ny)` in `b_hdr[3..7]`.
"""
from std.math import cos, exp, max, min, sin, sqrt
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_PUSH_WORDS,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    emit_mat4_mul_point,
    emit_vec3_dot,
    emit_vec3_normalize,
)


@always_inline
def spec_reblur_raw_offset(max_pix: Int) -> Int:
    return max_pix * 28


@always_inline
def spec_reblur_hist_ping_offset(max_pix: Int) -> Int:
    return max_pix * 32


@always_inline
def spec_reblur_hist_pong_offset(max_pix: Int) -> Int:
    return max_pix * 36


@always_inline
def spec_reblur_scratch_offset(max_pix: Int) -> Int:
    return max_pix * 40


def eval_anisotropic_ggx_blur_radii_cpu(
    roughness: Float32,
    ndotv: Float32,
    t_hit: Float32,
    z_surf: Float32,
    tan_y: Float32,
    render_h: Int,
    hist_len: Float32 = 1.0,
) -> Tuple[Float32, Float32]:
    """CPU reference for ReBLUR world-space GGX lobe cone -> anisotropic `(r_minor_px, r_major_px)` radii.
    Guarantees `0.0` blur on mirrors (`roughness < 0.05`) and contact hits (`t_hit -> 0.0`)."""
    if roughness < Float32(0.05) or t_hit <= Float32(1e-4):
        return (Float32(0.0), Float32(0.0))
    var alpha = roughness * roughness
    var world_cone = t_hit * alpha
    var px_per_m = Float32(render_h) / max(Float32(0.05), Float32(2.0) * z_surf * max(Float32(0.1), tan_y))
    var base_r = min(Float32(6.0), (world_cone * px_per_m) / sqrt(max(Float32(1.0), hist_len)))
    var aniso_stretch = Float32(1.0) / max(Float32(0.22), min(Float32(1.0), ndotv))
    var r_minor = min(Float32(4.0), base_r)
    var r_major = min(Float32(8.0), base_r * min(Float32(3.5), aniso_stretch))
    return (r_minor, r_major)


def spec_reblur_temporal_shader() -> List[UInt32]:
    """Pass 1: Virtual-Depth + Surface-Velocity Specular Temporal Accumulation (`group = 8 x 8`).
    Push[0] = w_u, Push[1] = h_u, Push[2] = raw_in_off (`28*max_pix`),
    Push[3] = hist_in_off, Push[4] = hist_out_off, Push[5] = scratch_out_off (`40*max_pix`),
    Push[6] = frame_idx.
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var raw_in_off = s.push_word(2)
    var hist_in_off = s.push_word(3)
    var hist_out_off = s.push_word(4)
    var scratch_out_off = s.push_word(5)
    var frame_idx = s.push_word(6)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

    var pix_idx = s.iadd(s.imul(py, w_u), px)
    var h_base = s.imul(pix_idx, s.uint(8))
    var r_base = s.iadd(raw_in_off, s.imul(pix_idx, s.uint(4)))
    var ho_base = s.iadd(hist_out_off, s.imul(pix_idx, s.uint(4)))
    var sc_base = s.iadd(scratch_out_off, s.imul(pix_idx, s.uint(4)))

    var cur_z = s.load_float(b_hdr, s.iadd(h_base, s.uint(3)))
    var vel_u = s.load_float(b_hdr, s.iadd(h_base, s.uint(4)))
    var vel_v = s.load_float(b_hdr, s.iadd(h_base, s.uint(5)))
    var cur_nx = s.load_float(b_hdr, s.iadd(h_base, s.uint(6)))

    # Early sky pixel handling (`cur_z >= 1e5`)
    s.begin_if(s.fge(cur_z, s.float(1e5)))
    var z0_pk = s.pack_f16x2(s.float(0.0), s.float(0.0))
    var z1_pk = s.pack_f16x2(s.float(0.0), s.float(1.0))
    var z3_pk = s.pack_f16x2(s.float(1.0), s.float(32768.0))
    s.store_word(b_hist, ho_base, z0_pk)
    s.store_word(b_hist, s.iadd(ho_base, s.uint(1)), z1_pk)
    s.store_float(b_hist, s.iadd(ho_base, s.uint(2)), cur_z)
    s.store_word(b_hist, s.iadd(ho_base, s.uint(3)), z3_pk)
    s.store_word(b_hist, sc_base, z0_pk)
    s.store_word(b_hist, s.iadd(sc_base, s.uint(1)), z1_pk)
    s.store_float(b_hist, s.iadd(sc_base, s.uint(2)), cur_z)
    s.store_word(b_hist, s.iadd(sc_base, s.uint(3)), z3_pk)
    s.end_if()
    s.return_if(s.fge(cur_z, s.float(1e5)))

    var u0 = s.unpack_f16x2(s.load_word(b_hist, r_base))
    var u1 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(r_base, s.uint(1))))
    var u3 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(r_base, s.uint(3))))
    var cur_r = u0[0]
    var cur_g = u0[1]
    var cur_b = u1[0]
    var rough = u1[1]
    var t_hit = u3[1]

    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var cur_u = s.fdiv(s.fadd(s.float_from_uint(px), s.float(0.5)), fw)
    var cur_v = s.fdiv(s.fadd(s.float_from_uint(py), s.float(0.5)), fh)

    # 1. Surface motion reprojection UV
    var surf_u = s.fsub(cur_u, vel_u)
    var surf_v = s.fsub(cur_v, vel_v)

    # 2. Virtual-Depth Planar Reprojection: `P_virt = cam_pos + D * (d_cam + t_hit)`
    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
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

    var jit_x = s.load_float(b_globals, s.uint(101))
    var jit_y = s.load_float(b_globals, s.uint(102))
    var fx0 = s.fsub(s.fadd(s.float_from_uint(px), s.float(0.5)), s.fmul(jit_x, s.fmul(fw, s.float(0.5))))
    var fy0 = s.fadd(s.fadd(s.float_from_uint(py), s.float(0.5)), s.fmul(jit_y, s.fmul(fh, s.float(0.5))))
    var ndc_x = s.fsub(s.fmul(s.fdiv(fx0, fw), s.float(2.0)), s.float(1.0))
    var ndc_y = s.fsub(s.float(1.0), s.fmul(s.fdiv(fy0, fh), s.float(2.0)))
    var ray_d = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(ndc_x, tan_x), s.fma(up_x, s.fmul(ndc_y, tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(ndc_x, tan_x), s.fma(up_y, s.fmul(ndc_y, tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(ndc_x, tan_x), s.fma(up_z, s.fmul(ndc_y, tan_y), fwd_z)),
    )
    var cos_fwd = s.nmax(s.float(0.05), emit_vec3_dot(s, ray_d[0], ray_d[1], ray_d[2], fwd_x, fwd_y, fwd_z))
    var d_cam = s.fdiv(cur_z, cos_fwd)
    var z_virt = s.fadd(d_cam, s.nmin(t_hit, s.float(500.0)))
    var pvx = s.fma(ray_d[0], z_virt, cam_x)
    var pvy = s.fma(ray_d[1], z_virt, cam_y)
    var pvz = s.fma(ray_d[2], z_virt, cam_z)
    var v_clip = emit_mat4_mul_point(s, b_globals, s.uint(16), pvx, pvy, pvz)
    var v_inv_w = s.fdiv(s.float(1.0), s.nmax(s.float(1e-4), v_clip[3]))
    var virt_u = s.fma(s.fmul(v_clip[0], v_inv_w), s.float(0.5), s.float(0.5))
    var virt_v = s.fma(s.fneg(s.fmul(v_clip[1], v_inv_w)), s.float(0.5), s.float(0.5))

    # Blend virtual UV toward surface UV as roughness increases (`0.05 -> 0.45`)
    var virt_w = s.select_f(
        s.flt(t_hit, s.float(1000.0)),
        s.nclamp(s.fmul(s.fsub(s.float(0.45), rough), s.float(2.5)), s.float(0.0), s.float(1.0)),
        s.float(0.0),
    )
    var cand_u = s.fma(s.fsub(virt_u, surf_u), virt_w, surf_u)
    var cand_v = s.fma(s.fsub(virt_v, surf_v), virt_w, surf_v)

    # Compute local `3x3` neighborhood soft clamping bounds (expanded for glossy stochastic VNDF convergence!)
    var min_r = s.fvar(cur_r); var max_r = s.fvar(cur_r)
    var min_g = s.fvar(cur_g); var max_g = s.fvar(cur_g)
    var min_b = s.fvar(cur_b); var max_b = s.fvar(cur_b)
    for ky in range(-1, 2):
        for kx in range(-1, 2):
            if kx != 0 or ky != 0:
                var nx_u = px
                if kx < 0:
                    nx_u = s.select_u(s.uge(px, s.uint(1)), s.isub(px, s.uint(1)), s.uint(0))
                elif kx > 0:
                    nx_u = s.umin(s.isub(w_u, s.uint(1)), s.iadd(px, s.uint(1)))
                var ny_u = py
                if ky < 0:
                    ny_u = s.select_u(s.uge(py, s.uint(1)), s.isub(py, s.uint(1)), s.uint(0))
                elif ky > 0:
                    ny_u = s.umin(s.isub(h_u, s.uint(1)), s.iadd(py, s.uint(1)))
                var nb = s.iadd(raw_in_off, s.imul(s.iadd(s.imul(ny_u, w_u), nx_u), s.uint(4)))
                var nu0 = s.unpack_f16x2(s.load_word(b_hist, nb))
                var nu1 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(nb, s.uint(1))))
                s.store(min_r, s.nmin(s.load_f(min_r), nu0[0])); s.store(max_r, s.nmax(s.load_f(max_r), nu0[0]))
                s.store(min_g, s.nmin(s.load_f(min_g), nu0[1])); s.store(max_g, s.nmax(s.load_f(max_g), nu0[1]))
                s.store(min_b, s.nmin(s.load_f(min_b), nu1[0])); s.store(max_b, s.nmax(s.load_f(max_b), nu1[0]))

    var res_r = s.fvar(cur_r)
    var res_g = s.fvar(cur_g)
    var res_b = s.fvar(cur_b)
    var res_len = s.fvar(s.float(1.0))

    s.begin_if(s.ine(frame_idx, s.uint(0)))
    # Try virtual-depth candidate UV first; if reflector plane ended (`depth/normal jump`), fall back to surface UV!
    var use_u = s.fvar(cand_u)
    var use_v = s.fvar(cand_v)
    var cand_in = s.logical_and(
        s.logical_and(s.fge(cand_u, s.float(0.0)), s.flt(cand_u, s.float(1.0))),
        s.logical_and(s.fge(cand_v, s.float(0.0)), s.flt(cand_v, s.float(1.0))),
    )
    var valid_hist = s.uvar(s.uint(0))
    var pr_v = s.fvar(cur_r)
    var pg_v = s.fvar(cur_g)
    var pb_v = s.fvar(cur_b)
    var plen_v = s.fvar(s.float(0.0))

    for attempt in range(2):
        if attempt == 1:
            s.begin_if(s.ieq(s.load_u(valid_hist), s.uint(0)))
            s.store(use_u, surf_u)
            s.store(use_v, surf_v)
        var tu = s.load_f(use_u)
        var tv = s.load_f(use_v)
        var in_b = s.logical_and(
            s.logical_and(s.fge(tu, s.float(0.0)), s.flt(tu, s.float(1.0))),
            s.logical_and(s.fge(tv, s.float(0.0)), s.flt(tv, s.float(1.0))),
        )
        if attempt == 0:
            in_b = cand_in
        s.begin_if(in_b)
        var hfx = s.nclamp(s.fsub(s.fmul(tu, fw), s.float(0.5)), s.float(0.0), s.fsub(fw, s.float(1.0)))
        var hfy = s.nclamp(s.fsub(s.fmul(tv, fh), s.float(0.5)), s.float(0.0), s.fsub(fh, s.float(1.0)))
        var hx0 = s.umin(s.isub(w_u, s.uint(1)), s.saturate_uint(s.ffloor(hfx)))
        var hy0 = s.umin(s.isub(h_u, s.uint(1)), s.saturate_uint(s.ffloor(hfy)))
        var hx1 = s.umin(s.isub(w_u, s.uint(1)), s.iadd(hx0, s.uint(1)))
        var hy1 = s.umin(s.isub(h_u, s.uint(1)), s.iadd(hy0, s.uint(1)))
        var tx_f = s.nclamp(s.fsub(hfx, s.float_from_uint(hx0)), s.float(0.0), s.float(1.0))
        var ty_f = s.nclamp(s.fsub(hfy, s.float_from_uint(hy0)), s.float(0.0), s.float(1.0))
        var ox_f = s.fsub(s.float(1.0), tx_f)
        var oy_f = s.fsub(s.float(1.0), ty_f)
        var w00 = s.fmul(ox_f, oy_f); var w10 = s.fmul(tx_f, oy_f)
        var w01 = s.fmul(ox_f, ty_f); var w11 = s.fmul(tx_f, ty_f)

        var b00 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy0, w_u), hx0), s.uint(4)))
        var b10 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy0, w_u), hx1), s.uint(4)))
        var b01 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy1, w_u), hx0), s.uint(4)))
        var b11 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy1, w_u), hx1), s.uint(4)))

        var pz = s.load_float(b_hist, s.iadd(b00, s.uint(2)))
        var p3_00 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(b00, s.uint(3))))
        var pnx = p3_00[0]
        var dz_rel = s.fdiv(s.fabs(s.fsub(cur_z, pz)), s.nmax(s.float(0.12), s.fmul(cur_z, s.float(0.12))))
        var dnx = s.fabs(s.fsub(cur_nx, pnx))
        s.begin_if(s.logical_and(s.flt(dz_rel, s.float(0.14)), s.flt(dnx, s.float(0.24))))
        var c00_0 = s.unpack_f16x2(s.load_word(b_hist, b00))
        var c00_1 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(b00, s.uint(1))))
        var c10_0 = s.unpack_f16x2(s.load_word(b_hist, b10))
        var c10_1 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(b10, s.uint(1))))
        var c01_0 = s.unpack_f16x2(s.load_word(b_hist, b01))
        var c01_1 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(b01, s.uint(1))))
        var c11_0 = s.unpack_f16x2(s.load_word(b_hist, b11))
        var c11_1 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(b11, s.uint(1))))

        var hr = s.fma(c00_0[0], w00, s.fma(c10_0[0], w10, s.fma(c01_0[0], w01, s.fmul(c11_0[0], w11))))
        var hg = s.fma(c00_0[1], w00, s.fma(c10_0[1], w10, s.fma(c01_0[1], w01, s.fmul(c11_0[1], w11))))
        var hb = s.fma(c00_1[0], w00, s.fma(c10_1[0], w10, s.fma(c01_1[0], w01, s.fmul(c11_1[0], w11))))
        var hl = s.fma(c00_1[1], w00, s.fma(c10_1[1], w10, s.fma(c01_1[1], w01, s.fmul(c11_1[1], w11))))
        s.store(pr_v, hr); s.store(pg_v, hg); s.store(pb_v, hb); s.store(plen_v, hl)
        s.store(valid_hist, s.uint(1))
        s.end_if()
        s.end_if()
        if attempt == 1:
            s.end_if()

    s.begin_if(s.ine(s.load_u(valid_hist), s.uint(0)))
    # Generous roughness-scaled neighborhood clamp box so 1-SPP glossy VNDF converges smoothly without ghost trails
    var slack = s.fma(rough, s.float(1.6), s.float(0.08))
    var span_r = s.fma(s.fsub(s.load_f(max_r), s.load_f(min_r)), slack, s.float(0.15))
    var span_g = s.fma(s.fsub(s.load_f(max_g), s.load_f(min_g)), slack, s.float(0.15))
    var span_b = s.fma(s.fsub(s.load_f(max_b), s.load_f(min_b)), slack, s.float(0.15))
    var hr_cl = s.nclamp(s.load_f(pr_v), s.nmax(s.float(0.0), s.fsub(s.load_f(min_r), span_r)), s.fadd(s.load_f(max_r), span_r))
    var hg_cl = s.nclamp(s.load_f(pg_v), s.nmax(s.float(0.0), s.fsub(s.load_f(min_g), span_g)), s.fadd(s.load_f(max_g), span_g))
    var hb_cl = s.nclamp(s.load_f(pb_v), s.nmax(s.float(0.0), s.fsub(s.load_f(min_b), span_b)), s.fadd(s.load_f(max_b), span_b))

    var max_len = s.fma(s.nclamp(s.fmul(rough, s.float(2.5)), s.float(0.0), s.float(1.0)), s.float(14.0), s.float(6.0))
    var new_len = s.nmin(max_len, s.fadd(s.load_f(plen_v), s.float(1.0)))
    var alpha = s.fdiv(s.float(1.0), new_len)
    var om_a = s.fsub(s.float(1.0), alpha)

    # Karis HDR firefly-suppressed blend `c / (1 + luma * 0.15)`
    var l_cur = s.fma(cur_r, s.float(0.2126), s.fma(cur_g, s.float(0.7152), s.fmul(cur_b, s.float(0.0722))))
    var l_his = s.fma(hr_cl, s.float(0.2126), s.fma(hg_cl, s.float(0.7152), s.fmul(hb_cl, s.float(0.0722))))
    var kw_cur = s.fmul(alpha, s.fdiv(s.float(1.0), s.fma(l_cur, s.float(0.15), s.float(1.0))))
    var kw_his = s.fmul(om_a, s.fdiv(s.float(1.0), s.fma(l_his, s.float(0.15), s.float(1.0))))
    var inv_kw = s.fdiv(s.float(1.0), s.nmax(s.float(1e-5), s.fadd(kw_cur, kw_his)))

    s.store(res_r, s.fmul(s.fma(cur_r, kw_cur, s.fmul(hr_cl, kw_his)), inv_kw))
    s.store(res_g, s.fmul(s.fma(cur_g, kw_cur, s.fmul(hg_cl, kw_his)), inv_kw))
    s.store(res_b, s.fmul(s.fma(cur_b, kw_cur, s.fmul(hb_cl, kw_his)), inv_kw))
    s.store(res_len, new_len)
    s.end_if()
    s.end_if()

    var fin_r = s.load_f(res_r)
    var fin_g = s.load_f(res_g)
    var fin_b = s.load_f(res_b)
    var fin_len = s.load_f(res_len)

    # Store persistent temporal history (`hist_out_off`)
    s.store_word(b_hist, ho_base, s.pack_f16x2(fin_r, fin_g))
    s.store_word(b_hist, s.iadd(ho_base, s.uint(1)), s.pack_f16x2(fin_b, fin_len))
    s.store_float(b_hist, s.iadd(ho_base, s.uint(2)), cur_z)
    s.store_word(b_hist, s.iadd(ho_base, s.uint(3)), s.pack_f16x2(cur_nx, s.nmin(t_hit, s.float(32768.0))))

    # Store À-Trous input (`scratch_out_off`)
    s.store_word(b_hist, sc_base, s.pack_f16x2(fin_r, fin_g))
    s.store_word(b_hist, s.iadd(sc_base, s.uint(1)), s.pack_f16x2(fin_b, rough))
    s.store_float(b_hist, s.iadd(sc_base, s.uint(2)), cur_z)
    s.store_word(b_hist, s.iadd(sc_base, s.uint(3)), s.pack_f16x2(fin_len, s.nmin(t_hit, s.float(32768.0))))
    return s.finish()


def spec_reblur_atrous_shader() -> List[UInt32]:
    """Pass 2: Anisotropic View-Stretched GGX À-Trous Specular Filter & Remodulator (`group = 8 x 8`).
    Push[0] = w_u, Push[1] = h_u, Push[2] = step_u (`1`, `2`),
    Push[3] = src_off, Push[4] = dst_off, Push[5] = raw_mod_off (`28*max_pix`),
    Push[6] = is_final (`1` = remodulate `mod_rgb * spec_rgb` into `b_hdr[0..2]`),
    Push[7] = meta_off (`40*max_pix`).
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var step_f = s.float_from_uint(s.push_word(2))
    var src_off = s.push_word(3)
    var dst_off = s.push_word(4)
    var raw_mod_off = s.push_word(5)
    var is_final = s.ine(s.push_word(6), s.uint(0))
    var meta_off = s.push_word(7)
    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

    var pix_idx = s.iadd(s.imul(py, w_u), px)
    var h_base = s.imul(pix_idx, s.uint(8))
    var cz = s.load_float(b_hdr, s.iadd(h_base, s.uint(3)))
    s.begin_if(s.fge(cz, s.float(1e5)))
    s.begin_if(s.logical_not(is_final))
    var d_sky = s.iadd(dst_off, s.imul(pix_idx, s.uint(4)))
    var z0_sky = s.pack_f16x2(s.float(0.0), s.float(0.0))
    var z1_sky = s.pack_f16x2(s.float(0.0), s.float(1.0))
    var z3_sky = s.pack_f16x2(s.float(1.0), s.float(32768.0))
    s.store_word(b_hist, d_sky, z0_sky)
    s.store_word(b_hist, s.iadd(d_sky, s.uint(1)), z1_sky)
    s.store_float(b_hist, s.iadd(d_sky, s.uint(2)), cz)
    s.store_word(b_hist, s.iadd(d_sky, s.uint(3)), z3_sky)
    s.end_if()
    s.end_if()
    s.return_if(s.fge(cz, s.float(1e5)))

    var s_base = s.iadd(src_off, s.imul(pix_idx, s.uint(4)))
    var m_base = s.iadd(meta_off, s.imul(pix_idx, s.uint(4)))
    var c0 = s.unpack_f16x2(s.load_word(b_hist, s_base))
    var c1 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(s_base, s.uint(1))))
    var m3_word = s.load_word(b_hist, s.iadd(m_base, s.uint(3)))
    var m3 = s.unpack_f16x2(m3_word)

    var cr = c0[0]
    var cg = c0[1]
    var cb = c1[0]
    var rough = c1[1]
    var hist_len = s.nmax(s.float(1.0), m3[0])
    var t_hit = m3[1]

    var cnx = s.load_float(b_hdr, s.iadd(h_base, s.uint(6)))
    var cny = s.load_float(b_hdr, s.iadd(h_base, s.uint(7)))
    var cnz = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fma(cnx, cnx, s.fmul(cny, cny)))))

    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var tan_x = s.nmax(s.float(0.1), s.load_float(b_globals, s.uint(39)))
    var tan_y = s.nmax(s.float(0.1), s.load_float(b_globals, s.uint(43)))
    var fwd_x = s.load_float(b_globals, s.uint(36))
    var fwd_y = s.load_float(b_globals, s.uint(37))
    var fwd_z = s.load_float(b_globals, s.uint(38))
    var rgt_x = s.load_float(b_globals, s.uint(40))
    var rgt_y = s.load_float(b_globals, s.uint(41))
    var rgt_z = s.load_float(b_globals, s.uint(42))
    var up_x = s.load_float(b_globals, s.uint(44))
    var up_y = s.load_float(b_globals, s.uint(45))
    var up_z = s.load_float(b_globals, s.uint(46))
    var frame_u = s.load_word(b_globals, s.uint(100))

    var fpx = s.float_from_uint(px)
    var fpy = s.float_from_uint(py)
    var ndc_x = s.fma(s.fdiv(s.fadd(fpx, s.float(0.5)), fw), s.float(2.0), s.float(-1.0))
    var ndc_y = s.fsub(s.float(1.0), s.fmul(s.fdiv(s.fadd(fpy, s.float(0.5)), fh), s.float(2.0)))
    var rd = emit_vec3_normalize(
        s,
        s.fma(rgt_x, s.fmul(ndc_x, tan_x), s.fma(up_x, s.fmul(ndc_y, tan_y), fwd_x)),
        s.fma(rgt_y, s.fmul(ndc_x, tan_x), s.fma(up_y, s.fmul(ndc_y, tan_y), fwd_y)),
        s.fma(rgt_z, s.fmul(ndc_x, tan_x), s.fma(up_z, s.fmul(ndc_y, tan_y), fwd_z)),
    )
    var ndotv = s.nclamp(s.fneg(emit_vec3_dot(s, cnx, cny, cnz, rd[0], rd[1], rd[2])), s.float(0.15), s.float(1.0))

    # World-space GGX ray cone -> screen-space radius (`0` on mirrors `rough < 0.05` or contact `t_hit -> 0`!)
    var alpha = s.fmul(rough, rough)
    var px_per_m = s.fdiv(fh, s.nmax(s.float(0.05), s.fmul(s.float(2.0), s.fmul(cz, tan_y))))
    var cone_px = s.fdiv(s.fmul(s.fmul(s.nmin(t_hit, s.float(80.0)), alpha), px_per_m), s.fsqrt(hist_len))
    var r_base_px = s.select_f(
        s.flt(rough, s.float(0.05)),
        s.float(0.0),
        s.nclamp(cone_px, s.float(0.0), s.float(4.2)),
    )

    var out_r = s.fvar(cr)
    var out_g = s.fvar(cg)
    var out_b = s.fvar(cb)

    s.begin_if(s.fgt(r_base_px, s.float(0.20)))
    # Screen-projected reflection vector `R = reflect(rd, c_nrm)` blended with `c_nrm` for true grazing GGX elongation!
    var two_nv = s.fma(ndotv, s.float(2.0), s.float(0.25))
    var rax_x = s.fma(cnx, two_nv, rd[0])
    var rax_y = s.fma(cny, two_nv, rd[1])
    var rax_z = s.fma(cnz, two_nv, rd[2])
    var nsx = emit_vec3_dot(s, rax_x, rax_y, rax_z, rgt_x, rgt_y, rgt_z)
    var nsy = s.fneg(emit_vec3_dot(s, rax_x, rax_y, rax_z, up_x, up_y, up_z))
    var ns_l2 = s.fma(nsx, nsx, s.fmul(nsy, nsy))
    var has_ns = s.fgt(ns_l2, s.float(1e-4))
    var inv_ns = s.inverse_sqrt(s.nmax(s.float(1e-6), ns_l2))
    var ax_x = s.select_f(has_ns, s.fmul(nsx, inv_ns), s.float(0.0))
    var ax_y = s.select_f(has_ns, s.fmul(nsy, inv_ns), s.float(1.0))
    var tan_ax_x = s.fneg(ax_y)
    var tan_ax_y = ax_x

    var stretch = s.nmin(s.float(3.0), s.fdiv(s.float(1.0), s.nmax(s.float(0.25), ndotv)))
    var r_min = s.nmin(s.float(10.0), s.fmul(s.nmin(s.float(2.4), r_base_px), step_f))
    var r_maj = s.nmin(s.float(22.0), s.fmul(s.nmin(s.float(5.2), s.fmul(r_base_px, stretch)), step_f))

    # Per-pixel Interleaved Gradient Noise (IGN) rotation angle to break discrete À-Trous grid gaps
    var fr_mod = s.float_from_uint(s.bit_and(frame_u, s.uint(63)))
    var ign_in = s.fma(fpx, s.float(0.06711056), s.fma(fpy, s.float(0.00583715), s.fmul(fr_mod, s.float(0.38196601))))
    var ign = s.fract(s.fmul(s.float(52.9829189), s.fract(ign_in)))
    var rot_a = s.fmul(ign, s.float(6.283185307))
    var cos_a = s.fcos(rot_a)
    var sin_a = s.fsin(rot_a)

    var c_luma = s.fma(cr, s.float(0.2126), s.fma(cg, s.float(0.7152), s.fmul(cb, s.float(0.0722))))
    var sum_r = s.fvar(cr)
    var sum_g = s.fvar(cg)
    var sum_b = s.fvar(cb)
    var sum_w = s.fvar(s.float(1.0))

    # 8-Tap Rotated Vogel / Poisson Disk around exact center tap (`w_center = 1.0`)
    for tap_i in range(8):
        var rad_f = sqrt((Float32(tap_i) + Float32(0.5)) / Float32(8.0))
        var ang_f = Float32(tap_i) * Float32(2.39996323)
        var bc_f = cos(ang_f) * rad_f
        var bs_f = sin(ang_f) * rad_f
        var kw = s.float(exp(Float32(-0.85) * rad_f * rad_f))

        var du = s.fsub(s.fmul(s.float(bc_f), cos_a), s.fmul(s.float(bs_f), sin_a))
        var dv = s.fma(s.float(bc_f), sin_a, s.fmul(s.float(bs_f), cos_a))
        var ox = s.fma(tan_ax_x, s.fmul(du, r_min), s.fmul(ax_x, s.fmul(dv, r_maj)))
        var oy = s.fma(tan_ax_y, s.fmul(du, r_min), s.fmul(ax_y, s.fmul(dv, r_maj)))
        var sx_u = s.umin(s.isub(w_u, s.uint(1)), s.saturate_uint(s.nclamp(s.fadd(s.fadd(fpx, ox), s.float(0.5)), s.float(0.0), s.fsub(fw, s.float(1.0)))))
        var sy_u = s.umin(s.isub(h_u, s.uint(1)), s.saturate_uint(s.nclamp(s.fadd(s.fadd(fpy, oy), s.float(0.5)), s.float(0.0), s.fsub(fh, s.float(1.0)))))
        var n_pix = s.iadd(s.imul(sy_u, w_u), sx_u)
        var nh_b = s.imul(n_pix, s.uint(8))
        var ns_b = s.iadd(src_off, s.imul(n_pix, s.uint(4)))

        var nz_l = s.load_float(b_hdr, s.iadd(nh_b, s.uint(3)))
        var nnx = s.load_float(b_hdr, s.iadd(nh_b, s.uint(6)))
        var nny = s.load_float(b_hdr, s.iadd(nh_b, s.uint(7)))
        var nnz = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fma(nnx, nnx, s.fmul(nny, nny)))))

        var nc0 = s.unpack_f16x2(s.load_word(b_hist, ns_b))
        var nc1 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(ns_b, s.uint(1))))
        var nr = nc0[0]; var ng = nc0[1]; var nb = nc1[0]; var n_rough = nc1[1]

        var ndot = s.nmax(s.float(0.0), emit_vec3_dot(s, cnx, cny, cnz, nnx, nny, nnz))
        var n_exp = s.fma(s.fsub(s.float(1.0), rough), s.float(96.0), s.float(16.0))
        var w_n = s.fpow(ndot, n_exp)
        var dz = s.fdiv(s.fabs(s.fsub(cz, nz_l)), s.nmax(s.float(0.05), s.fmul(cz, s.float(0.07))))
        var drg = s.fmul(s.fabs(s.fsub(rough, n_rough)), s.float(8.0))
        var n_luma = s.fma(nr, s.float(0.2126), s.fma(ng, s.float(0.7152), s.fmul(nb, s.float(0.0722))))
        var dl = s.fdiv(s.fabs(s.fsub(c_luma, n_luma)), s.fma(s.nmax(c_luma, s.float(0.2)), s.fma(rough, s.float(3.0), s.float(0.35)), s.float(0.05)))
        var w_edge = s.exp2(s.fneg(s.fmul(s.fadd(dz, s.fadd(drg, dl)), s.float(1.442695))))
        var w_tot = s.fmul(kw, s.fmul(w_n, w_edge))

        s.store(sum_r, s.fma(nr, w_tot, s.load_f(sum_r)))
        s.store(sum_g, s.fma(ng, w_tot, s.load_f(sum_g)))
        s.store(sum_b, s.fma(nb, w_tot, s.load_f(sum_b)))
        s.store(sum_w, s.fadd(s.load_f(sum_w), w_tot))

    var inv_sw = s.fdiv(s.float(1.0), s.nmax(s.float(1e-5), s.load_f(sum_w)))
    s.store(out_r, s.fmul(s.load_f(sum_r), inv_sw))
    s.store(out_g, s.fmul(s.load_f(sum_g), inv_sw))
    s.store(out_b, s.fmul(s.load_f(sum_b), inv_sw))
    s.end_if()

    var fr = s.load_f(out_r)
    var fg = s.load_f(out_g)
    var fb = s.load_f(out_b)

    s.begin_if(is_final)
    var mod_b = s.iadd(raw_mod_off, s.imul(pix_idx, s.uint(4)))
    var md2 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(mod_b, s.uint(2))))
    var md3 = s.unpack_f16x2(s.load_word(b_hist, s.iadd(mod_b, s.uint(3))))
    var mr = md2[0]
    var mg = md2[1]
    var mb_c = md3[0]
    # Add remodulated specular into `b_hdr[0..2]` while strictly preserving `b_hdr[3..7]`!
    s.store_float(b_hdr, h_base, s.fma(mr, fr, s.load_float(b_hdr, h_base)))
    s.store_float(b_hdr, s.iadd(h_base, s.uint(1)), s.fma(mg, fg, s.load_float(b_hdr, s.iadd(h_base, s.uint(1)))))
    s.store_float(b_hdr, s.iadd(h_base, s.uint(2)), s.fma(mb_c, fb, s.load_float(b_hdr, s.iadd(h_base, s.uint(2)))))
    s.begin_else()
    var d_base = s.iadd(dst_off, s.imul(pix_idx, s.uint(4)))
    s.store_word(b_hist, d_base, s.pack_f16x2(fr, fg))
    s.store_word(b_hist, s.iadd(d_base, s.uint(1)), s.pack_f16x2(fb, rough))
    s.store_float(b_hist, s.iadd(d_base, s.uint(2)), cz)
    s.store_word(b_hist, s.iadd(d_base, s.uint(3)), m3_word)
    s.end_if()

    return s.finish()
