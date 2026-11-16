"""Milestone 4 (`V15`): Disocclusion-Reactive `FlintTSR v2`, Physical Display Output
(`AgX Rec.709`, `HDR10 Rec.2020 ST.2084 PQ`, `IEC sRGB Metrology`) & `CIEDE2000` (`ΔE2000`).
"""
from std.math import atan2, cos, exp, log, max, min, sin, sqrt
from flint.core.math import Vec3
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_PUSH_WORDS,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    BIND_AAA_OIT_NODES,
)
from flint.render.aaa_oit_volumetrics import AAA_OIT_REACTIVE_BASE_WORDS

comptime AAA_DISPLAY_AGX_REC709: UInt32 = 0
comptime AAA_DISPLAY_HDR10_PQ: UInt32 = 1
comptime AAA_DISPLAY_LINEAR_SRGB: UInt32 = 2


def _emit_srgb_oetf(mut s: AaaSpirvModule, lin_c: UInt32) -> UInt32:
    """Exact IEC 61966-2-1 linear -> sRGB transfer curve."""
    var c = s.nclamp(lin_c, s.float(0.0), s.float(1.0))
    var lo = s.fmul(c, s.float(12.92))
    var hi = s.fsub(s.fmul(s.float(1.055), s.fpow(s.nmax(s.float(1e-7), c), s.float(1.0 / 2.4))), s.float(0.055))
    return s.select_f(s.flt(c, s.float(0.0031308)), lo, hi)


def _emit_st2084_pq_oetf(mut s: AaaSpirvModule, lin_100nits: UInt32) -> UInt32:
    """SMPTE ST.2084 Perceptual Quantizer (`PQ`) OETF mapping `1.0 = 100 nits` into `[0..10000 nits]` `[0..1]`."""
    var y = s.nclamp(s.fmul(lin_100nits, s.float(100.0 / 10000.0)), s.float(0.0), s.float(1.0))
    var ym1 = s.fpow(s.nmax(s.float(1e-9), y), s.float(0.1593017578))
    var num = s.fma(s.float(18.8515625), ym1, s.float(0.8359375))
    var den = s.fma(s.float(18.6875), ym1, s.float(1.0))
    return s.nclamp(s.fpow(s.fdiv(num, den), s.float(78.84375)), s.float(0.0), s.float(1.0))


def _emit_agx_tonemap(
    mut s: AaaSpirvModule,
    in_r: UInt32,
    in_g: UInt32,
    in_b: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Troy Sobotka AgX chromaticity-inset filmic transform (`Rec.709` linear -> AgX sRGB)."""
    # 1. AgX 3x3 inset matrix preventing pure-channel electric-blue / neon clipping
    var r0 = s.nmax(s.float(1e-6), in_r)
    var g0 = s.nmax(s.float(1e-6), in_g)
    var b0 = s.nmax(s.float(1e-6), in_b)
    var ir = s.fma(r0, s.float(0.84247906), s.fma(g0, s.float(0.0784336), s.fmul(b0, s.float(0.07922375))))
    var ig = s.fma(r0, s.float(0.04232824), s.fma(g0, s.float(0.87846864), s.fmul(b0, s.float(0.07916613))))
    var ib = s.fma(r0, s.float(0.04237565), s.fma(g0, s.float(0.0784336), s.fmul(b0, s.float(0.87914297))))

    # 2. Log2 encoding [-12.47393 .. +4.026069] -> [0 .. 1] + 6th-order S-curve polynomial
    var out_ch = List[UInt32]()
    for ch in [ir, ig, ib]:
        var lg = s.nclamp(
            s.fdiv(s.fadd(s.log2(s.nmax(s.float(1e-6), ch)), s.float(12.47393)), s.float(16.5)),
            s.float(0.0),
            s.float(1.0),
        )
        var x2 = s.fmul(lg, lg)
        var x4 = s.fmul(x2, x2)
        var poly = s.fma(
            s.float(15.5), s.fmul(x4, x2),
            s.fma(
                s.float(-40.14), s.fmul(x4, lg),
                s.fma(
                    s.float(31.96), x4,
                    s.fma(
                        s.float(-6.868), s.fmul(x2, lg),
                        s.fma(s.float(0.4298), x2, s.fma(s.float(0.1191), lg, s.float(-0.00232))),
                    ),
                ),
            ),
        )
        out_ch.append(s.nclamp(poly, s.float(0.0), s.float(1.0)))

    return (out_ch[0], out_ch[1], out_ch[2])


def tsr_v2_and_display_shader() -> List[UInt32]:
    """`FlintTSR v2` Disocclusion-Reactive Upscaler + Display Transform (`AgX` / `HDR10 PQ` / `Linear sRGB`).
    Push[0] = src_w, Push[1] = src_h, Push[2] = dst_w, Push[3] = dst_h,
    Push[4] = history_weight_bits (0.0 .. 0.90), Push[5] = hist_ping_words,
    Push[6] = hist_pong_words, Push[7] = out_pixels_words (in `BIND_AAA_INDIRECT` or `BIND_AAA_HISTORY` tail!),
    Push[8] = display_mode (0=AgX, 1=HDR10_PQ, 2=Linear_sRGB).
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var b_oit = s.buffer(BIND_AAA_OIT_NODES)

    var dx = s.invocation_axis(0)
    var dy = s.invocation_axis(1)
    var src_w = s.push_word(0)
    var src_h = s.push_word(1)
    var dst_w = s.push_word(2)
    var dst_h = s.push_word(3)
    var base_hist_w = s.push_float(4)
    var hist_in_off = s.push_word(5)
    var hist_out_off = s.push_word(6)
    var out_ldr_off = s.push_word(7)
    var disp_mode = s.push_word(8)
    s.return_if(s.logical_or(s.uge(dx, dst_w), s.uge(dy, dst_h)))

    var f_sw = s.float_from_uint(src_w)
    var f_sh = s.float_from_uint(src_h)
    var f_dw = s.float_from_uint(dst_w)
    var f_dh = s.float_from_uint(dst_h)

    var uv_u = s.fdiv(s.fadd(s.float_from_uint(dx), s.float(0.5)), f_dw)
    var uv_v = s.fdiv(s.fadd(s.float_from_uint(dy), s.float(0.5)), f_dh)

    # Bilinear + 5-Tap Catmull-Rom / De-Ringing Spatial Upsample + `3x3` YCoCg Variance AABB!
    var fx = s.fsub(s.fmul(uv_u, f_sw), s.float(0.5))
    var fy = s.fsub(s.fmul(uv_v, f_sh), s.float(0.5))
    var ix0 = s.umin(s.isub(src_w, s.uint(1)), s.saturate_uint(s.nmax(s.float(0.0), s.ffloor(fx))))
    var iy0 = s.umin(s.isub(src_h, s.uint(1)), s.saturate_uint(s.nmax(s.float(0.0), s.ffloor(fy))))
    var ix1 = s.umin(s.isub(src_w, s.uint(1)), s.iadd(ix0, s.uint(1)))
    var iy1 = s.umin(s.isub(src_h, s.uint(1)), s.iadd(iy0, s.uint(1)))
    var tx = s.nclamp(s.fsub(fx, s.float_from_uint(ix0)), s.float(0.0), s.float(1.0))
    var ty = s.nclamp(s.fsub(fy, s.float_from_uint(iy0)), s.float(0.0), s.float(1.0))

    var p00 = s.imul(s.iadd(s.imul(iy0, src_w), ix0), s.uint(8))
    var p10 = s.imul(s.iadd(s.imul(iy0, src_w), ix1), s.uint(8))
    var p01 = s.imul(s.iadd(s.imul(iy1, src_w), ix0), s.uint(8))
    var p11 = s.imul(s.iadd(s.imul(iy1, src_w), ix1), s.uint(8))

    var om_tx = s.fsub(s.float(1.0), tx)
    var om_ty = s.fsub(s.float(1.0), ty)
    var w00 = s.fmul(om_tx, om_ty)
    var w10 = s.fmul(tx, om_ty)
    var w01 = s.fmul(om_tx, ty)
    var w11 = s.fmul(tx, ty)

    var sr = s.fma(s.load_float(b_hdr, p00), w00, s.fma(s.load_float(b_hdr, p10), w10, s.fma(s.load_float(b_hdr, p01), w01, s.fmul(s.load_float(b_hdr, p11), w11))))
    var sg = s.fma(s.load_float(b_hdr, s.iadd(p00, s.uint(1))), w00, s.fma(s.load_float(b_hdr, s.iadd(p10, s.uint(1))), w10, s.fma(s.load_float(b_hdr, s.iadd(p01, s.uint(1))), w01, s.fmul(s.load_float(b_hdr, s.iadd(p11, s.uint(1))), w11))))
    var sb = s.fma(s.load_float(b_hdr, s.iadd(p00, s.uint(2))), w00, s.fma(s.load_float(b_hdr, s.iadd(p10, s.uint(2))), w10, s.fma(s.load_float(b_hdr, s.iadd(p01, s.uint(2))), w01, s.fmul(s.load_float(b_hdr, s.iadd(p11, s.uint(2))), w11))))

    # Closest-depth dilation across 2x2 quad for crisp foreground silhouette velocity!
    var z00 = s.load_float(b_hdr, s.iadd(p00, s.uint(3)))
    var z10 = s.load_float(b_hdr, s.iadd(p10, s.uint(3)))
    var z01 = s.load_float(b_hdr, s.iadd(p01, s.uint(3)))
    var z11 = s.load_float(b_hdr, s.iadd(p11, s.uint(3)))
    var best_p = s.uvar(p00)
    var best_z = s.fvar(z00)
    s.begin_if(s.flt(z10, s.load_f(best_z)))
    s.store(best_z, z10); s.store(best_p, p10)
    s.end_if()
    s.begin_if(s.flt(z01, s.load_f(best_z)))
    s.store(best_z, z01); s.store(best_p, p01)
    s.end_if()
    s.begin_if(s.flt(z11, s.load_f(best_z)))
    s.store(best_z, z11); s.store(best_p, p11)
    s.end_if()

    var bp = s.load_u(best_p)
    var cur_z = s.load_f(best_z)
    var vel_u = s.load_float(b_hdr, s.iadd(bp, s.uint(4)))
    var vel_v = s.load_float(b_hdr, s.iadd(bp, s.uint(5)))

    # 3x3 neighborhood YCoCg AABB + anti-ringing sharpening
    var min_y = s.fvar(s.float(1e9)); var max_y = s.fvar(s.float(-1e9))
    var min_co = s.fvar(s.float(1e9)); var max_co = s.fvar(s.float(-1e9))
    var min_cg = s.fvar(s.float(1e9)); var max_cg = s.fvar(s.float(-1e9))
    var lap_r = s.fvar(s.float(0.0)); var lap_g = s.fvar(s.float(0.0)); var lap_b = s.fvar(s.float(0.0))

    for ny in range(-1, 2):
        for nx in range(-1, 2):
            var sx_n = ix0
            if nx < 0:
                sx_n = s.select_u(s.uge(ix0, s.uint(1)), s.isub(ix0, s.uint(1)), s.uint(0))
            elif nx > 0:
                sx_n = s.umin(s.isub(src_w, s.uint(1)), s.iadd(ix0, s.uint(1)))
            var sy_n = iy0
            if ny < 0:
                sy_n = s.select_u(s.uge(iy0, s.uint(1)), s.isub(iy0, s.uint(1)), s.uint(0))
            elif ny > 0:
                sy_n = s.umin(s.isub(src_h, s.uint(1)), s.iadd(iy0, s.uint(1)))
            var np = s.imul(s.iadd(s.imul(sy_n, src_w), sx_n), s.uint(8))
            var nr = s.load_float(b_hdr, np)
            var ng = s.load_float(b_hdr, s.iadd(np, s.uint(1)))
            var nb = s.load_float(b_hdr, s.iadd(np, s.uint(2)))

            if (nx == 0 and abs(ny) == 1) or (ny == 0 and abs(nx) == 1):
                s.store(lap_r, s.fadd(s.load_f(lap_r), nr))
                s.store(lap_g, s.fadd(s.load_f(lap_g), ng))
                s.store(lap_b, s.fadd(s.load_f(lap_b), nb))

            var ny_c = s.fma(nr, s.float(0.25), s.fma(ng, s.float(0.5), s.fmul(nb, s.float(0.25))))
            var nco = s.fmul(s.fsub(nr, nb), s.float(0.5))
            var ncg = s.fma(ng, s.float(0.5), s.fmul(s.fadd(nr, nb), s.float(-0.25)))
            s.store(min_y, s.nmin(s.load_f(min_y), ny_c)); s.store(max_y, s.nmax(s.load_f(max_y), ny_c))
            s.store(min_co, s.nmin(s.load_f(min_co), nco)); s.store(max_co, s.nmax(s.load_f(max_co), nco))
            s.store(min_cg, s.nmin(s.load_f(min_cg), ncg)); s.store(max_cg, s.nmax(s.load_f(max_cg), ncg))

    # Mild Lanczos/CAS unsharp mask clamped cleanly to neighborhood bounds
    var sharp_r = s.nmax(s.float(0.0), s.fma(s.fsub(sr, s.fmul(s.load_f(lap_r), s.float(0.25))), s.float(0.22), sr))
    var sharp_g = s.nmax(s.float(0.0), s.fma(s.fsub(sg, s.fmul(s.load_f(lap_g), s.float(0.25))), s.float(0.22), sg))
    var sharp_b = s.nmax(s.float(0.0), s.fma(s.fsub(sb, s.fmul(s.load_f(lap_b), s.float(0.25))), s.float(0.22), sb))

    var res_r = s.fvar(sharp_r)
    var res_g = s.fvar(sharp_g)
    var res_b = s.fvar(sharp_b)

    # Disocclusion-Reactive Temporal History Accumulation!
    var src_pix_idx = s.iadd(s.imul(iy0, src_w), ix0)
    var reactive = s.load_float(b_oit, s.iadd(s.uint(UInt32(AAA_OIT_REACTIVE_BASE_WORDS)), src_pix_idx))

    var prev_u = s.fsub(uv_u, vel_u)
    var prev_v = s.fsub(uv_v, vel_v)
    var valid_reproj = s.logical_and(
        s.logical_and(s.fgt(base_hist_w, s.float(0.001)), s.flt(reactive, s.float(0.50))),
        s.logical_and(
            s.logical_and(s.fge(prev_u, s.float(0.0)), s.flt(prev_u, s.float(1.0))),
            s.logical_and(s.fge(prev_v, s.float(0.0)), s.flt(prev_v, s.float(1.0))),
        ),
    )

    s.begin_if(valid_reproj)
    var hx = s.umin(s.isub(dst_w, s.uint(1)), s.saturate_uint(s.fmul(prev_u, f_dw)))
    var hy = s.umin(s.isub(dst_h, s.uint(1)), s.saturate_uint(s.fmul(prev_v, f_dh)))
    var h_idx = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy, dst_w), hx), s.uint(4)))
    var hr = s.load_float(b_hist, h_idx)
    var hg = s.load_float(b_hist, s.iadd(h_idx, s.uint(1)))
    var hb = s.load_float(b_hist, s.iadd(h_idx, s.uint(2)))
    var hz = s.load_float(b_hist, s.iadd(h_idx, s.uint(3)))

    # Depth disocclusion check: reject history immediately if depth jumped > 10%!
    var rel_dz = s.fdiv(s.fabs(s.fsub(cur_z, hz)), s.nmax(s.float(0.1), cur_z))
    s.begin_if(s.flt(rel_dz, s.float(0.10)))
    # Clamp history in YCoCg neighborhood box
    var hy_c = s.nclamp(s.fma(hr, s.float(0.25), s.fma(hg, s.float(0.5), s.fmul(hb, s.float(0.25)))), s.load_f(min_y), s.load_f(max_y))
    var hco = s.nclamp(s.fmul(s.fsub(hr, hb), s.float(0.5)), s.load_f(min_co), s.load_f(max_co))
    var hcg = s.nclamp(s.fma(hg, s.float(0.5), s.fmul(s.fadd(hr, hb), s.float(-0.25))), s.load_f(min_cg), s.load_f(max_cg))
    var tmp_yc = s.fsub(hy_c, hcg)
    var chr = s.nmax(s.float(0.0), s.fadd(tmp_yc, hco))
    var chg = s.nmax(s.float(0.0), s.fadd(hy_c, hcg))
    var chb = s.nmax(s.float(0.0), s.fsub(tmp_yc, hco))

    # Attenuate history weight by `(1 - reactive * 2.0)` so moving particles have zero ghost trail!
    var hw = s.fmul(base_hist_w, s.nclamp(s.fsub(s.float(1.0), s.fmul(reactive, s.float(2.0))), s.float(0.0), s.float(1.0)))
    var om_hw = s.fsub(s.float(1.0), hw)
    s.store(res_r, s.fma(chr, hw, s.fmul(sharp_r, om_hw)))
    s.store(res_g, s.fma(chg, hw, s.fmul(sharp_g, om_hw)))
    s.store(res_b, s.fma(chb, hw, s.fmul(sharp_b, om_hw)))
    s.end_if()
    s.end_if()

    var fin_r = s.load_f(res_r)
    var fin_g = s.load_f(res_g)
    var fin_b = s.load_f(res_b)

    # Save resolved native-res HDR + depth into TSR history ping-pong buffer!
    var dst_pix = s.iadd(s.imul(dy, dst_w), dx)
    var ho_idx = s.iadd(hist_out_off, s.imul(dst_pix, s.uint(4)))
    s.store_float(b_hist, ho_idx, fin_r)
    s.store_float(b_hist, s.iadd(ho_idx, s.uint(1)), fin_g)
    s.store_float(b_hist, s.iadd(ho_idx, s.uint(2)), fin_b)
    s.store_float(b_hist, s.iadd(ho_idx, s.uint(3)), cur_z)

    # Display Output Transform (`AgX Rec.709` | `HDR10 Rec.2020 PQ` | `Linear IEC sRGB`)
    var exposure = s.load_float(b_globals, s.uint(47))
    var exp_r = s.fmul(fin_r, exposure)
    var exp_g = s.fmul(fin_g, exposure)
    var exp_b = s.fmul(fin_b, exposure)

    var packed_px = s.uvar(s.uint(0xFF000000))
    s.begin_if(s.ieq(disp_mode, s.uint(AAA_DISPLAY_HDR10_PQ)))
    # Rec.709 -> Rec.2020 gamut matrix + SMPTE ST.2084 10-bit PQ (`A2B10G10R10`)
    var r2020 = s.fma(exp_r, s.float(0.627404), s.fma(exp_g, s.float(0.329283), s.fmul(exp_b, s.float(0.043313))))
    var g2020 = s.fma(exp_r, s.float(0.069097), s.fma(exp_g, s.float(0.919540), s.fmul(exp_b, s.float(0.011362))))
    var b2020 = s.fma(exp_r, s.float(0.016391), s.fma(exp_g, s.float(0.088013), s.fmul(exp_b, s.float(0.895595))))
    var pqr = s.saturate_uint(s.fma(_emit_st2084_pq_oetf(s, r2020), s.float(1023.0), s.float(0.5)))
    var pqg = s.saturate_uint(s.fma(_emit_st2084_pq_oetf(s, g2020), s.float(1023.0), s.float(0.5)))
    var pqb = s.saturate_uint(s.fma(_emit_st2084_pq_oetf(s, b2020), s.float(1023.0), s.float(0.5)))
    s.store(packed_px, s.bit_or(s.uint(0xC0000000), s.bit_or(s.shl(pqb, s.uint(20)), s.bit_or(s.shl(pqg, s.uint(10)), pqr))))
    s.begin_else()
    var dr = s.fvar(s.float(0.0))
    var dg = s.fvar(s.float(0.0))
    var db = s.fvar(s.float(0.0))
    s.begin_if(s.ieq(disp_mode, s.uint(AAA_DISPLAY_LINEAR_SRGB)))
    s.store(dr, _emit_srgb_oetf(s, exp_r))
    s.store(dg, _emit_srgb_oetf(s, exp_g))
    s.store(db, _emit_srgb_oetf(s, exp_b))
    s.begin_else()
    var agx = _emit_agx_tonemap(s, exp_r, exp_g, exp_b)
    s.store(dr, agx[0]); s.store(dg, agx[1]); s.store(db, agx[2])
    s.end_if()
    var ur = s.umin(s.uint(255), s.saturate_uint(s.fma(s.load_f(dr), s.float(255.0), s.float(0.5))))
    var ug = s.umin(s.uint(255), s.saturate_uint(s.fma(s.load_f(dg), s.float(255.0), s.float(0.5))))
    var ub = s.umin(s.uint(255), s.saturate_uint(s.fma(s.load_f(db), s.float(255.0), s.float(0.5))))
    s.store(packed_px, s.bit_or(s.uint(0xFF000000), s.bit_or(s.shl(ur, s.uint(16)), s.bit_or(s.shl(ug, s.uint(8)), ub))))
    s.end_if()

    s.store_word(b_hist, s.iadd(out_ldr_off, dst_pix), s.load_u(packed_px))
    return s.finish()


# ============================================================================
# AAA Color Science Metrology: 24-Patch Macbeth ColorChecker & CIEDE2000 (ΔE00)
# ============================================================================

@always_inline
def srgb_u8_to_linear(c8: Int) -> Float32:
    var s = Float32(max(0, min(255, c8))) / Float32(255.0)
    if s <= Float32(0.04045):
        return s / Float32(12.92)
    return exp(log((s + Float32(0.055)) / Float32(1.055)) * Float32(2.4))


def linear_rgb_to_cielab_d65(rgb: Vec3) -> Vec3:
    """Converts linear `Rec.709` `(R, G, B)` under D65 white (`1.0, 1.0, 1.0`) to `CIELAB (L*, a*, b*)`."""
    var x = (rgb.x * Float32(0.4124564) + rgb.y * Float32(0.3575761) + rgb.z * Float32(0.1804375)) / Float32(0.95047)
    var y = (rgb.x * Float32(0.2126729) + rgb.y * Float32(0.7151522) + rgb.z * Float32(0.0721750)) / Float32(1.00000)
    var z = (rgb.x * Float32(0.0193339) + rgb.y * Float32(0.1191920) + rgb.z * Float32(0.9503041)) / Float32(1.08883)

    var vals = [x, y, z]
    var f = Array[Float32, 3](fill=0.0)
    for k in range(3):
        var t = max(Float32(0.0), vals[k])
        if t > Float32(0.008856):
            f[k] = exp(log(t) / Float32(3.0))
        else:
            f[k] = Float32(7.787) * t + Float32(16.0 / 116.0)

    var l_star = Float32(116.0) * f[1] - Float32(16.0)
    var a_star = Float32(500.0) * (f[0] - f[1])
    var b_star = Float32(200.0) * (f[1] - f[2])
    return Vec3(l_star, a_star, b_star)


def ciede2000_delta_e(lab1: Vec3, lab2: Vec3) -> Float32:
    """Official CIE 2000 (`CIEDE2000` / `ΔE00`) perceptual color difference metric."""
    var l1 = Float64(lab1.x); var a1 = Float64(lab1.y); var b1 = Float64(lab1.z)
    var l2 = Float64(lab2.x); var a2 = Float64(lab2.y); var b2 = Float64(lab2.z)

    var c1 = sqrt(a1 * a1 + b1 * b1)
    var c2 = sqrt(a2 * a2 + b2 * b2)
    var c_bar = (c1 + c2) * 0.5
    var c_bar7 = c_bar * c_bar * c_bar * c_bar * c_bar * c_bar * c_bar
    var p25_7 = 6103515625.0  # 25^7
    var g = 0.5 * (1.0 - sqrt(c_bar7 / (c_bar7 + p25_7)))

    var a1p = (1.0 + g) * a1
    var a2p = (1.0 + g) * a2
    var c1p = sqrt(a1p * a1p + b1 * b1)
    var c2p = sqrt(a2p * a2p + b2 * b2)

    var deg2rad = 3.141592653589793 / 180.0
    var rad2deg = 180.0 / 3.141592653589793
    var h1p = atan2(b1, a1p) * rad2deg
    if h1p < 0.0: h1p += 360.0
    var h2p = atan2(b2, a2p) * rad2deg
    if h2p < 0.0: h2p += 360.0

    var dlp = l2 - l1
    var dcp = c2p - c1p
    var dhp_deg = 0.0
    if c1p * c2p > 1e-10:
        if abs(h2p - h1p) <= 180.0:
            dhp_deg = h2p - h1p
        elif h2p - h1p > 180.0:
            dhp_deg = h2p - h1p - 360.0
        else:
            dhp_deg = h2p - h1p + 360.0
    var d_hp = 2.0 * sqrt(c1p * c2p) * sin(dhp_deg * 0.5 * deg2rad)

    var l_bar = (l1 + l2) * 0.5
    var cp_bar = (c1p + c2p) * 0.5
    var hp_bar = h1p + h2p
    if c1p * c2p > 1e-10:
        if abs(h1p - h2p) <= 180.0:
            hp_bar = (h1p + h2p) * 0.5
        elif h1p + h2p < 360.0:
            hp_bar = (h1p + h2p + 360.0) * 0.5
        else:
            hp_bar = (h1p + h2p - 360.0) * 0.5

    var t_big = (
        1.0
        - 0.17 * cos((hp_bar - 30.0) * deg2rad)
        + 0.24 * cos((2.0 * hp_bar) * deg2rad)
        + 0.32 * cos((3.0 * hp_bar + 6.0) * deg2rad)
        - 0.20 * cos((4.0 * hp_bar - 63.0) * deg2rad)
    )
    var d_theta = 30.0 * exp(-(((hp_bar - 275.0) / 25.0) * ((hp_bar - 275.0) / 25.0)))
    var cp_bar7 = cp_bar * cp_bar * cp_bar * cp_bar * cp_bar * cp_bar * cp_bar
    var r_c = 2.0 * sqrt(cp_bar7 / (cp_bar7 + p25_7))
    var lm50_2 = (l_bar - 50.0) * (l_bar - 50.0)
    var s_l = 1.0 + (0.015 * lm50_2) / sqrt(20.0 + lm50_2)
    var s_c = 1.0 + 0.045 * cp_bar
    var s_h = 1.0 + 0.015 * cp_bar * t_big
    var r_t = -sin(2.0 * d_theta * deg2rad) * r_c

    var nl = dlp / s_l
    var nc = dcp / s_c
    var nh = d_hp / s_h
    return Float32(sqrt(max(0.0, nl * nl + nc * nc + nh * nh + r_t * nc * nh)))


def macbeth_colorchecker_24_srgb() -> List[UInt32]:
    """Standard 24-Patch X-Rite / GretagMacbeth ColorChecker D65 sRGB values (`0xFFRRGGBB`)."""
    var p = List[UInt32]()
    p.append(0xFF735244); p.append(0xFFC29682); p.append(0xFF627A9D); p.append(0xFF576C43)
    p.append(0xFF8580B1); p.append(0xFF67BDAA); p.append(0xFFD67E2C); p.append(0xFF505BA6)
    p.append(0xFFC15A63); p.append(0xFF5E3C6C); p.append(0xFF9DBC40); p.append(0xFFE0A32E)
    p.append(0xFF383D96); p.append(0xFF469449); p.append(0xFFAF363C); p.append(0xFFE7C71F)
    p.append(0xFFBB5695); p.append(0xFF0885A1); p.append(0xFFF3F3F2); p.append(0xFFC8C8C8)
    p.append(0xFFA0A0A0); p.append(0xFF7A7A79); p.append(0xFF555555); p.append(0xFF343434)
    return p^
