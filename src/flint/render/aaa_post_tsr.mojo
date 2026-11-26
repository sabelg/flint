"""Milestone 4 (`V15`) & Phase 4 (`V19`):
  - Physical Camera (`EV100`, 64-Bin Log-Luminance Histogram Auto-Exposure)
  - Energy-Conserving 5-Mip Dual-Kawase HDR Bloom (`ShaderDsl`)
  - Thin-Lens Circle-of-Confusion Polygonal Diaphragm Bokeh Depth-of-Field (`ShaderDsl`)
  - Natural `cos^4(theta)` Optical Vignette, Disocclusion-Reactive `FlintTSR v2`,
    Physical Display Output (`AgX Rec.709`, `HDR10 PQ`, `IEC sRGB`) & `CIEDE2000` (`ΔE2000`).
"""
from std.math import atan2, cos, exp, log, log2, max, min, sin, sqrt
from flint.core.math import Vec3
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.shader_dsl import (
    GpuF32,
    GpuU32,
    GpuVec3,
    ShaderDsl,
    abs_f,
    clamp_f,
    cos_f,
    dot,
    exp2_f,
    floor_f,
    fma_f,
    log2_f,
    max_f,
    min_f,
    min_u,
    select_f,
    sin_f,
)
from flint.render.aaa_visbuffer import (
    AAA_PUSH_WORDS,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    BIND_AAA_OIT_NODES,
    BIND_AAA_POST_SCRATCH,
)
from flint.render.aaa_oit_volumetrics import AAA_OIT_REACTIVE_BASE_WORDS
from flint.render.sota_camera_hdr import sota_hdr_emit_tonemap_hdr  # --- sota:camera hook ---

comptime AAA_DISPLAY_AGX_REC709: UInt32 = 0
comptime AAA_DISPLAY_HDR10_PQ: UInt32 = 1
comptime AAA_DISPLAY_LINEAR_SRGB: UInt32 = 2

# `BIND_AAA_POST_SCRATCH` word layout:
comptime AAA_POST_HIST_BINS: Int = 64
comptime AAA_POST_ADAPTED_EXP_WORD: Int = 64
comptime AAA_POST_AVG_LUMA_WORD: Int = 65
comptime AAA_POST_BLOOM_MIPS: Int = 5
comptime AAA_POST_BLOOM_BASE: Int = 1024


struct AaaPhysicalCamera(Copyable, Movable):
    """Physical cinema camera model (`EV100`, thin-lens Circle of Confusion, polygonal iris & optical vignette)."""
    var aperture_fstop: Float32      # e.g. f/1.8 .. f/16.0
    var shutter_seconds: Float32     # e.g. 1/125 s = 0.008
    var iso: Float32                 # e.g. 100.0
    var focal_length_mm: Float32     # e.g. 50.0 mm
    var sensor_height_mm: Float32    # Super35 / Full-Frame 24.0 mm
    var focus_distance_m: Float32    # Plane of sharp focus in meters
    var max_coc_px: Float32          # Maximum bokeh radius in pixels
    var diaphragm_blades: Int        # 6 = hexagonal, 8 = octagonal
    var vignette_strength: Float32   # 0.0 = off (metrology), 1.0 = natural cos^4(theta)

    def __init__(
        out self,
        aperture_fstop: Float32 = 8.0,
        shutter_seconds: Float32 = 1.0 / 125.0,
        iso: Float32 = 100.0,
        focal_length_mm: Float32 = 50.0,
        focus_distance_m: Float32 = 5.0,
        diaphragm_blades: Int = 6,
        vignette_strength: Float32 = 0.0,
    ):
        self.aperture_fstop = max(Float32(0.7), aperture_fstop)
        self.shutter_seconds = max(Float32(1e-5), shutter_seconds)
        self.iso = max(Float32(10.0), iso)
        self.focal_length_mm = max(Float32(8.0), focal_length_mm)
        self.sensor_height_mm = Float32(24.0)
        self.focus_distance_m = max(Float32(0.15), focus_distance_m)
        self.max_coc_px = Float32(8.0)
        self.diaphragm_blades = max(5, min(12, diaphragm_blades))
        self.vignette_strength = max(Float32(0.0), min(Float32(1.0), vignette_strength))

    def compute_ev100(self) -> Float32:
        """ISO 2720 standard exposure value `EV100 = log2((N^2 / t) * (100 / ISO))`."""
        var n2_t = (self.aperture_fstop * self.aperture_fstop) / self.shutter_seconds
        return log2(max(Float32(1e-6), n2_t * (Float32(100.0) / self.iso)))

    def compute_manual_exposure(self) -> Float32:
        """Lagarde / Sébastien Hillaire saturation-based exposure scale `1 / (1.2 * 2^EV100)`."""
        var ev100 = self.compute_ev100()
        return Float32(1.0) / max(Float32(1e-6), Float32(1.2) * exp(ev100 * Float32(0.69314718)))

    def compute_coc_scale_px(self, render_h: Int) -> Float32:
        """Thin-lens Circle-of-Confusion radius scale (pixels at infinity for focus plane `focus_distance_m`)."""
        var f_m = self.focal_length_mm * Float32(0.001)
        var sensor_m = self.sensor_height_mm * Float32(0.001)
        var denom = max(Float32(0.02), self.aperture_fstop * max(Float32(0.05), self.focus_distance_m - f_m))
        var coc_sensor_m = (f_m * f_m) / denom
        return (coc_sensor_m / sensor_m) * Float32(render_h) * Float32(0.5)


def bloom_mip_word_offset(max_pix: Int, mip: Int) -> Int:
    """Word offset inside `BIND_AAA_POST_SCRATCH` for Dual-Kawase bloom mip `0 .. 4` (`4` floats/texel)."""
    var off = AAA_POST_BLOOM_BASE
    var cur_cap = max(16, max_pix // 4)
    for _ in range(mip):
        off += cur_cap * 4
        cur_cap = max(16, cur_cap // 4)
    return off


def dof_scratch_word_offset(max_pix: Int) -> Int:
    return AAA_POST_BLOOM_BASE + max_pix * 2


@always_inline
def _radical_inverse_base(idx: Int, base: Int) -> Float32:
    var val = Float32(0.0)
    var inv_b = Float32(1.0) / Float32(base)
    var f = inv_b
    var n = idx
    while n > 0:
        val += Float32(n % base) * f
        n = n // base
        f *= inv_b
    return val


def halton_2_3_jitter_px(frame_index: Int) -> Tuple[Float32, Float32]:
    """True 16-phase radical-inverse Halton `(2, 3)` sub-pixel jitter sequence in `[-0.5, +0.5)` render pixels.
    Shifted by `+0.5 mod 1.0 - 0.5` so `phase == 0` (`frame_index == 0`) is exact `(0.0, 0.0)` while
    phases `0..15` cover the full 16-sample low-discrepancy sub-pixel unit square at all `tsr_scale` values!
    """
    var phase = frame_index & 15
    var rx = _radical_inverse_base(phase, 2) + Float32(0.5)
    if rx >= Float32(1.0):
        rx -= Float32(1.0)
    var ry = _radical_inverse_base(phase, 3) + Float32(0.5)
    if ry >= Float32(1.0):
        ry -= Float32(1.0)
    return (rx - Float32(0.5), ry - Float32(0.5))


# ============================================================================
# 1. GPU 64-Bin Log-Luminance Histogram & Temporal Auto-Exposure (`ShaderDsl`)
# ============================================================================

def aaa_luminance_histogram_shader(bind_hdr: Int, bind_post: Int) -> List[UInt32]:
    """Accumulates `64`-bin log2 scene luminance (`[-10 EV .. +14 EV]`) across `render_w x render_h`."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](bind_hdr)
    var b_post = dsl.buffer[1](bind_post)

    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    dsl.return_if(px.ge(w_u) | py.ge(h_u))

    var base = b_hdr.base_word(py * w_u + px)
    var rgb = b_hdr.load_vec3(base)
    var luma = dot(rgb, dsl.vec3_const(0.2126, 0.7152, 0.0722))
    var ev = log2_f(max_f(luma, 0.0009765625))  # >= -10.0 EV
    var norm = clamp_f((ev + Float32(10.0)) * Float32(1.0 / 24.0), 0.0, 1.0)
    var bin_u = min_u((norm * Float32(63.0) + Float32(0.5)).saturate_u32(), dsl.u32(63))
    _ = b_post.atomic_add(bin_u, dsl.u32(1))
    return dsl.finish()


def aaa_adapt_exposure_shader(bind_post: Int) -> List[UInt32]:
    """Outlier-trimmed (`5% .. 95%` = 95%-trimmed percentile) histogram reducer, bounded EV clamp
    & smooth asymmetric eye adaptation (`group = 1x1`).
    Push[0] = total_pixels, Push[1] = adapt_lerp_bits (`0.0 .. 1.0`), Push[2] = key_value_bits (default `0.18`),
    Push[3] = ev_comp_scale_bits (`2^ev_bias`), Push[4] = min_auto_exposure (`0` -> `0.02`),
    Push[5] = max_auto_exposure (`0` -> `32.0`)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=1, group_height=1, push_words=AAA_PUSH_WORDS)
    var b_post = dsl.buffer[1](bind_post)
    dsl.return_if(dsl.global_id_x().ne(0))

    var tot_px = max_f(dsl.push_u32(0).to_f32(), 1.0)
    var adapt_k = clamp_f(dsl.push_f32(1), 0.01, 1.0)
    var key_val = max_f(dsl.push_f32(2), 0.02)
    var ev_comp = max_f(dsl.push_f32(3), 0.01)
    var min_exp_in = dsl.push_f32(4)
    var max_exp_in = dsl.push_f32(5)
    var min_exp = select_f(min_exp_in.gt(0.0001), min_exp_in, dsl.f32(0.02))
    var max_exp = select_f(max_exp_in.gt(min_exp), max_exp_in, dsl.f32(32.0))

    var lo_cut = tot_px * Float32(0.05)
    var hi_cut = tot_px * Float32(0.95)
    var running = dsl.var_f32_const(0.0)
    var w_sum = dsl.var_f32_const(0.0)
    var ev_sum = dsl.var_f32_const(0.0)

    var bi = dsl.var_u32(dsl.u32(0))
    dsl.begin_loop()
    var b_cur = bi.get()
    dsl.break_if(b_cur.ge(UInt32(64)))

    var cnt_f = b_post.load_u32(b_cur).to_f32()
    b_post.store_u32(b_cur, dsl.u32(0))

    var r0 = running.get()
    var r1 = r0 + cnt_f
    running.set(r1)

    var seg_lo = max_f(r0, lo_cut)
    var seg_hi = min_f(r1, hi_cut)
    var w_bin = max_f(seg_hi - seg_lo, 0.0)
    var bin_ev = fma_f(b_cur.to_f32(), dsl.f32(24.0 / 63.0), dsl.f32(-10.0))
    ev_sum.set(fma_f(w_bin, bin_ev, ev_sum.get()))
    w_sum.set(w_sum.get() + w_bin)

    bi.set(b_cur + UInt32(1))
    dsl.end_loop()

    var mean_ev = ev_sum.get() / max_f(w_sum.get(), 1.0)
    var avg_luma = exp2_f(mean_ev)
    var target_exp = clamp_f((key_val / max_f(avg_luma, 0.001)) * ev_comp, min_exp, max_exp)

    var prev_exp = b_post.load_f32(dsl.u32(UInt32(AAA_POST_ADAPTED_EXP_WORD)))
    var has_prev = prev_exp.gt(0.0001)
    var next_exp = select_f(has_prev, prev_exp + (target_exp - prev_exp) * adapt_k, target_exp)

    b_post.store_f32(dsl.u32(UInt32(AAA_POST_ADAPTED_EXP_WORD)), next_exp)
    b_post.store_f32(dsl.u32(UInt32(AAA_POST_AVG_LUMA_WORD)), avg_luma)
    return dsl.finish()


# ============================================================================
# 2. Energy-Conserving 5-Mip Dual-Kawase HDR Bloom (`ShaderDsl`)
# ============================================================================

@always_inline
def _sample_bloom_bilinear_dsl(
    mut dsl: ShaderDsl,
    b_hdr_id: UInt32,
    b_post_id: UInt32,
    from_hdr: GpuU32,
    src_base: GpuU32,
    src_w: GpuU32,
    src_h: GpuU32,
    u_01: GpuF32,
    v_01: GpuF32,
) -> GpuVec3:
    from flint.render.shader_dsl import BufferRef, select_vec3
    var b_hdr = BufferRef[8](dsl.mod_addr, b_hdr_id)
    var b_post = BufferRef[1](dsl.mod_addr, b_post_id)

    var f_sw = src_w.to_f32()
    var f_sh = src_h.to_f32()
    var fx = clamp_f(u_01 * f_sw - Float32(0.5), dsl.f32(0.0), max_f(f_sw - Float32(1.0), 0.0))
    var fy = clamp_f(v_01 * f_sh - Float32(0.5), dsl.f32(0.0), max_f(f_sh - Float32(1.0), 0.0))
    var ix0 = min_u(src_w - UInt32(1), floor_f(fx).saturate_u32())
    var iy0 = min_u(src_h - UInt32(1), floor_f(fy).saturate_u32())
    var ix1 = min_u(src_w - UInt32(1), ix0 + UInt32(1))
    var iy1 = min_u(src_h - UInt32(1), iy0 + UInt32(1))
    var tx = clamp_f(fx - ix0.to_f32(), 0.0, 1.0)
    var ty = clamp_f(fy - iy0.to_f32(), 0.0, 1.0)

    var idx00 = iy0 * src_w + ix0
    var idx10 = iy0 * src_w + ix1
    var idx01 = iy1 * src_w + ix0
    var idx11 = iy1 * src_w + ix1

    var is_hdr = from_hdr.ne(0)
    var c00 = select_vec3(is_hdr, b_hdr.load_vec3(b_hdr.base_word(idx00)), b_post.load_vec3(src_base + idx00 * UInt32(4)))
    var c10 = select_vec3(is_hdr, b_hdr.load_vec3(b_hdr.base_word(idx10)), b_post.load_vec3(src_base + idx10 * UInt32(4)))
    var c01 = select_vec3(is_hdr, b_hdr.load_vec3(b_hdr.base_word(idx01)), b_post.load_vec3(src_base + idx01 * UInt32(4)))
    var c11 = select_vec3(is_hdr, b_hdr.load_vec3(b_hdr.base_word(idx11)), b_post.load_vec3(src_base + idx11 * UInt32(4)))

    var top = c00 * (dsl.f32(1.0) - tx) + c10 * tx
    var bot = c01 * (dsl.f32(1.0) - tx) + c11 * tx
    return top * (dsl.f32(1.0) - ty) + bot * ty


def aaa_bloom_downsample_shader(bind_hdr: Int, bind_post: Int) -> List[UInt32]:
    """5-Tap Energy-Conserving Dual-Kawase Downsample (`0.5 * center + 0.125 * 4 corners = 1.0000`!).
    Push[0]=src_w, Push[1]=src_h, Push[2]=dst_w, Push[3]=dst_h,
    Push[4]=from_hdr (1 for mip0, 0 for mip1..4), Push[5]=src_base_words, Push[6]=dst_base_words."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](bind_hdr)
    var b_post = dsl.buffer[1](bind_post)

    var dx = dsl.global_id_x()
    var dy = dsl.global_id_y()
    var src_w = dsl.push_u32(0)
    var src_h = dsl.push_u32(1)
    var dst_w = dsl.push_u32(2)
    var dst_h = dsl.push_u32(3)
    var from_hdr = dsl.push_u32(4)
    var src_base = dsl.push_u32(5)
    var dst_base = dsl.push_u32(6)
    var thresh = dsl.push_f32(7)
    dsl.return_if(dx.ge(dst_w) | dy.ge(dst_h))

    var inv_dw = dsl.f32(1.0) / dst_w.to_f32()
    var inv_dh = dsl.f32(1.0) / dst_h.to_f32()
    var u = (dx.to_f32() + Float32(0.5)) * inv_dw
    var v = (dy.to_f32() + Float32(0.5)) * inv_dh
    var du = dsl.f32(1.0) / max_f(src_w.to_f32(), 1.0)
    var dv = dsl.f32(1.0) / max_f(src_h.to_f32(), 1.0)

    var c_mid = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, from_hdr, src_base, src_w, src_h, u, v)
    var c_tl = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, from_hdr, src_base, src_w, src_h, u - du, v - dv)
    var c_tr = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, from_hdr, src_base, src_w, src_h, u + du, v - dv)
    var c_bl = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, from_hdr, src_base, src_w, src_h, u - du, v + dv)
    var c_br = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, from_hdr, src_base, src_w, src_h, u + du, v + dv)

    var down_raw = c_mid * Float32(0.5) + (c_tl + c_tr + c_bl + c_br) * Float32(0.125)
    var down_out = dsl.var_vec3(down_raw)
    dsl.if_then(from_hdr.ne(0) & thresh.gt(0.001))
    var br = max_f(down_raw.x(), max_f(down_raw.y(), down_raw.z()))
    var knee = thresh * Float32(0.5)
    var soft = clamp_f(br - thresh + knee, dsl.f32(0.0), knee * Float32(2.0))
    var contrib = max_f((soft * soft) / max_f(knee * Float32(4.0), 1e-4), br - thresh)
    down_out.set(down_raw * (contrib / max_f(br, 1e-4)))
    dsl.end_if()

    var out_off = dst_base + (dy * dst_w + dx) * UInt32(4)
    b_post.store_vec3(out_off, down_out.get())
    b_post.store_f32(out_off + UInt32(3), dsl.f32(1.0))
    return dsl.finish()


def aaa_bloom_upsample_shader(bind_hdr: Int, bind_post: Int) -> List[UInt32]:
    """8-Tap Energy-Conserving Dual-Kawase Tent Upsample (`4*(1/12) + 4*(2/12) = 1.0000`!).
    Push[0]=src_w, Push[1]=src_h, Push[2]=dst_w, Push[3]=dst_h,
    Push[4]=final_to_hdr (`1` = energy-conserving composite into `b_hdr`, `0` = blend into `dst_base`),
    Push[5]=src_base_words, Push[6]=dst_base_words, Push[7]=bloom_mix_bits (`0.0 .. 1.0`)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](bind_hdr)
    var b_post = dsl.buffer[1](bind_post)

    from flint.render.shader_dsl import select_vec3
    var dx = dsl.global_id_x()
    var dy = dsl.global_id_y()
    var src_w = dsl.push_u32(0)
    var src_h = dsl.push_u32(1)
    var dst_w = dsl.push_u32(2)
    var dst_h = dsl.push_u32(3)
    var to_hdr = dsl.push_u32(4).ne(0)
    var src_base = dsl.push_u32(5)
    var dst_base = dsl.push_u32(6)
    var b_mix = clamp_f(dsl.push_f32(7), 0.0, 1.0)
    var additive_hdr = dsl.push_u32(8).ne(0)
    var glare_str = clamp_f(dsl.push_f32(9), 0.0, 2.0)
    var mip_idx = dsl.push_u32(10)
    dsl.return_if(dx.ge(dst_w) | dy.ge(dst_h))

    var u = (dx.to_f32() + Float32(0.5)) / dst_w.to_f32()
    var v = (dy.to_f32() + Float32(0.5)) / dst_h.to_f32()
    var du = dsl.f32(1.0) / max_f(src_w.to_f32(), 1.0)
    var dv = dsl.f32(1.0) / max_f(src_h.to_f32(), 1.0)
    var zero_u = dsl.u32(0)

    var c_n = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u, v - dv * Float32(2.0))
    var c_s = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u, v + dv * Float32(2.0))
    var c_w = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u - du * Float32(2.0), v)
    var c_e = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u + du * Float32(2.0), v)
    var c_nw = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u - du, v - dv)
    var c_ne = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u + du, v - dv)
    var c_sw = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u - du, v + dv)
    var c_se = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u + du, v + dv)

    var tent_base = (c_n + c_s + c_w + c_e) * Float32(1.0 / 12.0) + (c_nw + c_ne + c_sw + c_se) * Float32(2.0 / 12.0)
    var tent_var = dsl.var_vec3(tent_base)

    # Cinema Anamorphic Horizontal Flare + 6-Blade Starburst Diffraction Streaks on Mip 2 & Mip 3
    dsl.if_then(glare_str.gt(0.001) & (mip_idx.eq(2) | mip_idx.eq(3)))
    var a_l1 = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u - du * Float32(4.5), v)
    var a_r1 = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u + du * Float32(4.5), v)
    var a_l2 = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u - du * Float32(9.0), v)
    var a_r2 = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u + du * Float32(9.0), v)
    var s_d1 = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u - du * Float32(3.1), v - dv * Float32(1.8))
    var s_d2 = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u + du * Float32(3.1), v + dv * Float32(1.8))
    var s_d3 = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u - du * Float32(3.1), v + dv * Float32(1.8))
    var s_d4 = _sample_bloom_bilinear_dsl(dsl, b_hdr.buf_id, b_post.buf_id, zero_u, src_base, src_w, src_h, u + du * Float32(3.1), v - dv * Float32(1.8))
    var anam = ((a_l1 + a_r1) * Float32(0.14) + (a_l2 + a_r2) * Float32(0.06)) * dsl.vec3_const(0.76, 0.90, 1.24)
    var star = (s_d1 + s_d2 + s_d3 + s_d4) * Float32(0.065) * dsl.vec3_const(1.10, 1.00, 0.90)
    tent_var.set(tent_base + (anam + star) * glare_str)
    dsl.end_if()

    var tent = tent_var.get()
    var pix_i = dy * dst_w + dx

    dsl.if_then(to_hdr)
    var h_off = b_hdr.base_word(pix_i)
    var cur_hdr = b_hdr.load_vec3(h_off)
    var hdr_comp = select_vec3(additive_hdr, cur_hdr + tent * b_mix, cur_hdr * (dsl.f32(1.0) - b_mix) + tent * b_mix)
    b_hdr.store_vec3(h_off, hdr_comp)
    dsl.else_branch()
    var d_off = dst_base + pix_i * UInt32(4)
    var finer_mip = b_post.load_vec3(d_off)
    b_post.store_vec3(d_off, finer_mip * Float32(0.5) + tent * Float32(0.5))
    dsl.end_if()
    return dsl.finish()


# ============================================================================
# 3. Thin-Lens Polygonal Diaphragm (`6..12`-Blade) Bokeh DoF (`ShaderDsl`)
# ============================================================================

def aaa_bokeh_dof_shader(bind_hdr: Int, bind_post: Int) -> List[UInt32]:
    """2-Subpass Thin-Lens Polygonal Bokeh Depth-of-Field (`ShaderDsl`)."""
    var dsl = ShaderDsl(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = dsl.buffer[8](bind_hdr)
    var b_post = dsl.buffer[1](bind_post)

    var px = dsl.global_id_x()
    var py = dsl.global_id_y()
    var w_u = dsl.push_u32(0)
    var h_u = dsl.push_u32(1)
    var mode = dsl.push_u32(2)
    var dof_base = dsl.push_u32(3)
    var focus_z = max_f(dsl.push_f32(4), 0.05)
    var coc_scale = dsl.push_f32(5)
    var max_coc = max_f(dsl.push_f32(6), 1.0)
    var blades_f = max_f(dsl.push_f32(7), 5.0)
    dsl.return_if(px.ge(w_u) | py.ge(h_u))

    var pix_i = py * w_u + px
    var h_off = b_hdr.base_word(pix_i)
    var d_off = dof_base + pix_i * UInt32(4)

    dsl.if_then(mode.eq(0))
    var c_rgb = b_hdr.load_vec3(h_off)
    var z_lin = max_f(b_hdr.load_f32(h_off + UInt32(3)), 0.05)
    var z_eff = min_f(z_lin, 100.0)
    var scoc = clamp_f((z_eff - focus_z) / z_eff * coc_scale, -max_coc, max_coc)
    b_post.store_vec3(d_off, c_rgb)
    b_post.store_f32(d_off + UInt32(3), scoc)
    dsl.else_branch()
    var center_rgb = b_post.load_vec3(d_off)
    var center_scoc = b_post.load_f32(d_off + UInt32(3))
    var center_coc = abs_f(center_scoc)

    var acc_rgb = dsl.var_vec3(center_rgb)
    var acc_w = dsl.var_f32_const(1.0)
    var fw = w_u.to_f32()
    var fh = h_u.to_f32()
    var fpx = px.to_f32()
    var fpy = py.to_f32()
    var sector_ang = dsl.f32(6.283185307) / blades_f
    var half_sec = sector_ang * Float32(0.5)
    var poly_num = cos_f(half_sec)

    comptime TAPS: Int = 24
    for ti in range(TAPS):
        var r_norm = sqrt(Float32(ti + 1) / Float32(TAPS))
        var theta = Float32(ti) * Float32(2.39996323)
        var d_th = dsl.f32(theta)
        var k_sec = floor_f(d_th / sector_ang)
        var local_ang = d_th - k_sec * sector_ang - half_sec
        var poly_r = poly_num / max_f(cos_f(local_ang), 0.35)
        var rad = max_f(center_coc, 1.5) * poly_r * r_norm
        var ox = cos_f(d_th) * rad
        var oy = sin_f(d_th) * rad

        var sx = min_u(w_u - UInt32(1), clamp_f(fpx + ox + Float32(0.5), dsl.f32(0.0), fw - Float32(1.0)).saturate_u32())
        var sy = min_u(h_u - UInt32(1), clamp_f(fpy + oy + Float32(0.5), dsl.f32(0.0), fh - Float32(1.0)).saturate_u32())
        var s_off = dof_base + (sy * w_u + sx) * UInt32(4)
        var s_rgb = b_post.load_vec3(s_off)
        var s_scoc = b_post.load_f32(s_off + UInt32(3))
        var s_abs = abs_f(s_scoc)

        # Prevent far-background blur (`s_scoc > 0.5`) from leaking onto closer in-focus foreground subjects!
        var bg_occluded = s_scoc.gt(0.5) & center_scoc.lt(s_scoc - Float32(0.5))
        var eff_tap_coc = select_f(bg_occluded, center_coc, s_abs)
        var use_coc = max_f(center_coc, eff_tap_coc)
        var w_tap = clamp_f(use_coc - rad + Float32(1.0), 0.0, 1.0) * select_f(use_coc.gt(0.5), dsl.f32(1.0), dsl.f32(0.0))
        acc_rgb.set(acc_rgb.get() + s_rgb * w_tap)
        acc_w.set(acc_w.get() + w_tap)

    b_hdr.store_vec3(h_off, acc_rgb.get() * (dsl.f32(1.0) / max_f(acc_w.get(), 1e-4)))
    dsl.end_if()
    return dsl.finish()


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
    """Troy Sobotka Full 3-Stage Cinema AgX Transform (`Purkinje Shadow Desat -> Inset -> +5.5 EV Log2 Sigmoid -> t0=0.018 Shadow Toe -> Outset -> sRGB`)."""
    var r_pos = s.nmax(s.float(1e-6), in_r)
    var g_pos = s.nmax(s.float(1e-6), in_g)
    var b_pos = s.nmax(s.float(1e-6), in_b)

    # 0. Purkinje physiological scotopic/mesopic dark-shadow desaturation (`y_lin < 0.035`)
    # Eliminates oversaturated milky indigo/violet night shadows while preserving midtones & highlights!
    var y_lin = s.fma(r_pos, s.float(0.2126), s.fma(g_pos, s.float(0.7152), s.fmul(b_pos, s.float(0.0722))))
    var purk_t = s.nclamp(s.fmul(y_lin, s.float(1.0 / 0.035)), s.float(0.0), s.float(1.0))
    var purk_sm = s.fmul(purk_t, s.fmul(purk_t, s.fma(s.float(-2.0), purk_t, s.float(3.0))))
    var purk_sat = s.fma(s.float(0.62), purk_sm, s.float(0.38))
    var r0 = s.nmax(s.float(1e-6), s.fma(s.fsub(r_pos, y_lin), purk_sat, y_lin))
    var g0 = s.nmax(s.float(1e-6), s.fma(s.fsub(g_pos, y_lin), purk_sat, y_lin))
    var b0 = s.nmax(s.float(1e-6), s.fma(s.fsub(b_pos, y_lin), purk_sat, y_lin))

    # 1. AgX 3x3 inset matrix preventing pure-channel electric-blue / neon clipping
    var ir = s.fma(r0, s.float(0.84247906), s.fma(g0, s.float(0.07843360), s.fmul(b0, s.float(0.07922375))))
    var ig = s.fma(r0, s.float(0.04232824), s.fma(g0, s.float(0.87846864), s.fmul(b0, s.float(0.07916613))))
    var ib = s.fma(r0, s.float(0.04237565), s.fma(g0, s.float(0.07843360), s.fmul(b0, s.float(0.87914297))))

    # 2. Extended Log2 encoding [-12.47393 .. +5.52607 EV] (`18.0 EV` span = `+5.5 EV` white shoulder!)
    #    + 6th-order S-curve + CDL Power (2.816) + Filmic Shadow Toe Crush (`t0 = 0.018`)
    var lin_ch = List[UInt32]()
    for ch in [ir, ig, ib]:
        var lg = s.nclamp(
            s.fdiv(s.fadd(s.log2(s.nmax(s.float(1e-6), ch)), s.float(12.47393)), s.float(18.0)),
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
        var sig = s.nclamp(poly, s.float(1e-6), s.float(1.0))
        var cdl_c = s.fpow(sig, s.float(2.816))
        # Smooth C2 rational toe: preserves dark gunmetal micro-gradient while holding deep cinema blacks
        var toe_num = s.fmul(cdl_c, s.fadd(cdl_c, s.float(0.0012)))
        var toe_den = s.fadd(cdl_c, s.float(0.0105))
        var toe_c = s.fmul(s.fdiv(toe_num, toe_den), s.float(1.0093))
        lin_ch.append(toe_c)

    # 3. Exact Inverse AgX 3x3 Outset Matrix (unit row sums preserve neutral D65 grey!) + Punchy Chroma Boost (1.24x)
    var lr = lin_ch[0]
    var lg_c = lin_ch[1]
    var lb = lin_ch[2]
    var or_c = s.fma(lr, s.float(1.1968790), s.fma(lg_c, s.float(-0.0980209), s.fmul(lb, s.float(-0.0988581))))
    var og_c = s.fma(lr, s.float(-0.0528969), s.fma(lg_c, s.float(1.1519031), s.fmul(lb, s.float(-0.0990062))))
    var ob_c = s.fma(lr, s.float(-0.0529716), s.fma(lg_c, s.float(-0.0980435), s.fmul(lb, s.float(1.1510151))))

    var luma = s.fma(or_c, s.float(0.2126), s.fma(og_c, s.float(0.7152), s.fmul(ob_c, s.float(0.0722))))
    var sat = s.fma(s.float(0.24), purk_sm, s.float(1.00))
    var sr = _emit_srgb_oetf(s, s.fma(s.fsub(or_c, luma), sat, luma))
    var sg = _emit_srgb_oetf(s, s.fma(s.fsub(og_c, luma), sat, luma))
    var sb = _emit_srgb_oetf(s, s.fma(s.fsub(ob_c, luma), sat, luma))
    return (sr, sg, sb)


def eval_agx_tonemap_cpu(rgb: Vec3) -> Vec3:
    """CPU reference for Cinema AgX (`Purkinje Desat -> Inset -> +5.5 EV Shoulder -> t0=0.018 Toe -> Outset -> sRGB`)."""
    var rp = max(Float32(1e-6), rgb.x)
    var gp = max(Float32(1e-6), rgb.y)
    var bp = max(Float32(1e-6), rgb.z)
    var y_lin = rp * Float32(0.2126) + gp * Float32(0.7152) + bp * Float32(0.0722)
    var pt = max(Float32(0.0), min(Float32(1.0), y_lin / Float32(0.035)))
    var psm = pt * pt * (Float32(3.0) - Float32(2.0) * pt)
    var psat = Float32(0.38) + Float32(0.62) * psm
    var r0 = max(Float32(1e-6), y_lin + (rp - y_lin) * psat)
    var g0 = max(Float32(1e-6), y_lin + (gp - y_lin) * psat)
    var b0 = max(Float32(1e-6), y_lin + (bp - y_lin) * psat)

    var ir = r0 * Float32(0.84247906) + g0 * Float32(0.07843360) + b0 * Float32(0.07922375)
    var ig = r0 * Float32(0.04232824) + g0 * Float32(0.87846864) + b0 * Float32(0.07916613)
    var ib = r0 * Float32(0.04237565) + g0 * Float32(0.07843360) + b0 * Float32(0.87914297)

    var lin = Array[Float32, 3](fill=0.0)
    var ins = [ir, ig, ib]
    for k in range(3):
        var lg = max(Float32(0.0), min(Float32(1.0), (log2(max(Float32(1e-6), ins[k])) + Float32(12.47393)) / Float32(18.0)))
        var x2 = lg * lg
        var x4 = x2 * x2
        var poly = (
            Float32(15.5) * x4 * x2
            - Float32(40.14) * x4 * lg
            + Float32(31.96) * x4
            - Float32(6.868) * x2 * lg
            + Float32(0.4298) * x2
            + Float32(0.1191) * lg
            - Float32(0.00232)
        )
        var sig = max(Float32(1e-6), min(Float32(1.0), poly))
        var cdl_c = exp(log(sig) * Float32(2.816))
        lin[k] = ((cdl_c * (cdl_c + Float32(0.0012))) / (cdl_c + Float32(0.0105))) * Float32(1.0093)

    var or_c = lin[0] * Float32(1.1968790) + lin[1] * Float32(-0.0980209) + lin[2] * Float32(-0.0988581)
    var og_c = lin[0] * Float32(-0.0528969) + lin[1] * Float32(1.1519031) + lin[2] * Float32(-0.0990062)
    var ob_c = lin[0] * Float32(-0.0529716) + lin[1] * Float32(-0.0980435) + lin[2] * Float32(1.1510151)
    var luma = or_c * Float32(0.2126) + og_c * Float32(0.7152) + ob_c * Float32(0.0722)
    var sat = Float32(1.00) + Float32(0.24) * psm
    var out = Array[Float32, 3](fill=0.0)
    var outs = [luma + (or_c - luma) * sat, luma + (og_c - luma) * sat, luma + (ob_c - luma) * sat]
    for k in range(3):
        var c = max(Float32(0.0), min(Float32(1.0), outs[k]))
        out[k] = c * Float32(12.92) if c < Float32(0.0031308) else (Float32(1.055) * exp(log(c) / Float32(2.4)) - Float32(0.055))
    return Vec3(out[0], out[1], out[2])



def _emit_display_grade(
    mut s: AaaSpirvModule, r: UInt32, g: UInt32, b: UInt32, sat_delta: UInt32, contrast_delta: UInt32
) -> Tuple[UInt32, UInt32, UInt32]:
    """GPU twin of `eval_display_grade_cpu`: saturation then contrast on display-encoded [0, 1] color.

    The deltas are (amount - 1), so an all-zero push block is the identity grade.
    """
    var sat = s.nmax(s.float(0.0), s.fadd(s.float(1.0), sat_delta))
    var con = s.nmax(s.float(0.0), s.fadd(s.float(1.0), contrast_delta))
    var luma = s.fma(r, s.float(0.2126), s.fma(g, s.float(0.7152), s.fmul(b, s.float(0.0722))))
    var out = List[UInt32]()
    for ch in [r, g, b]:
        var sc = s.fma(s.fsub(ch, luma), sat, luma)
        out.append(s.nclamp(s.fma(s.fsub(sc, s.float(0.5)), con, s.float(0.5)), s.float(0.0), s.float(1.0)))
    return (out[0], out[1], out[2])


def eval_display_grade_cpu(rgb: Vec3, saturation: Float32, contrast: Float32) -> Vec3:
    """The display grade on display-encoded [0, 1] color: saturation around Rec.709 luma, then
    contrast around mid-grey 0.5; 1.0 and 1.0 leave the color unchanged."""
    var sat = max(Float32(0.0), saturation)
    var con = max(Float32(0.0), contrast)
    var luma = rgb.x * Float32(0.2126) + rgb.y * Float32(0.7152) + rgb.z * Float32(0.0722)
    var ch = [rgb.x, rgb.y, rgb.z]
    for k in range(3):
        var sc = luma + (ch[k] - luma) * sat
        ch[k] = max(Float32(0.0), min(Float32(1.0), Float32(0.5) + (sc - Float32(0.5)) * con))
    return Vec3(ch[0], ch[1], ch[2])

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
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)

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
    var sharp_amt = s.nclamp(s.push_float(9), s.float(0.0), s.float(1.0))
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

    # 3x3 neighborhood YCoCg AABB + RCAS-clamped anti-ringing sharpening
    var min_y = s.fvar(s.float(1e9)); var max_y = s.fvar(s.float(-1e9))
    var min_co = s.fvar(s.float(1e9)); var max_co = s.fvar(s.float(-1e9))
    var min_cg = s.fvar(s.float(1e9)); var max_cg = s.fvar(s.float(-1e9))
    var min_r = s.fvar(s.float(1e9)); var max_r = s.fvar(s.float(-1e9))
    var min_g = s.fvar(s.float(1e9)); var max_g = s.fvar(s.float(-1e9))
    var min_b = s.fvar(s.float(1e9)); var max_b = s.fvar(s.float(-1e9))
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

            s.store(min_r, s.nmin(s.load_f(min_r), nr)); s.store(max_r, s.nmax(s.load_f(max_r), nr))
            s.store(min_g, s.nmin(s.load_f(min_g), ng)); s.store(max_g, s.nmax(s.load_f(max_g), ng))
            s.store(min_b, s.nmin(s.load_f(min_b), nb)); s.store(max_b, s.nmax(s.load_f(max_b), nb))

            var ny_c = s.fma(nr, s.float(0.25), s.fma(ng, s.float(0.5), s.fmul(nb, s.float(0.25))))
            var nco = s.fmul(s.fsub(nr, nb), s.float(0.5))
            var ncg = s.fma(ng, s.float(0.5), s.fmul(s.fadd(nr, nb), s.float(-0.25)))
            s.store(min_y, s.nmin(s.load_f(min_y), ny_c)); s.store(max_y, s.nmax(s.load_f(max_y), ny_c))
            s.store(min_co, s.nmin(s.load_f(min_co), nco)); s.store(max_co, s.nmax(s.load_f(max_co), nco))
            s.store(min_cg, s.nmin(s.load_f(min_cg), ncg)); s.store(max_cg, s.nmax(s.load_f(max_cg), ncg))

    # Contrast-adaptive unsharp mask strictly clamped to local 3x3 neighborhood `[min_rgb, max_rgb]` (zero ringing!)
    var raw_sr = s.fma(s.fsub(sr, s.fmul(s.load_f(lap_r), s.float(0.25))), sharp_amt, sr)
    var raw_sg = s.fma(s.fsub(sg, s.fmul(s.load_f(lap_g), s.float(0.25))), sharp_amt, sg)
    var raw_sb = s.fma(s.fsub(sb, s.fmul(s.load_f(lap_b), s.float(0.25))), sharp_amt, sb)
    var sharp_r = s.nmax(s.float(0.0), s.nclamp(raw_sr, s.load_f(min_r), s.load_f(max_r)))
    var sharp_g = s.nmax(s.float(0.0), s.nclamp(raw_sg, s.load_f(min_g), s.load_f(max_g)))
    var sharp_b = s.nmax(s.float(0.0), s.nclamp(raw_sb, s.load_f(min_b), s.load_f(max_b)))

    var res_r = s.fvar(sharp_r)
    var res_g = s.fvar(sharp_g)
    var res_b = s.fvar(sharp_b)

    # Disocclusion-Reactive & Velocity-Adaptive Sub-Pixel Bilinear Temporal History Accumulation!
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
    var hfx = s.fsub(s.fmul(prev_u, f_dw), s.float(0.5))
    var hfy = s.fsub(s.fmul(prev_v, f_dh), s.float(0.5))
    var hx0 = s.umin(s.isub(dst_w, s.uint(1)), s.saturate_uint(s.nmax(s.float(0.0), s.ffloor(hfx))))
    var hy0 = s.umin(s.isub(dst_h, s.uint(1)), s.saturate_uint(s.nmax(s.float(0.0), s.ffloor(hfy))))
    var hx1 = s.umin(s.isub(dst_w, s.uint(1)), s.iadd(hx0, s.uint(1)))
    var hy1 = s.umin(s.isub(dst_h, s.uint(1)), s.iadd(hy0, s.uint(1)))
    var htx = s.nclamp(s.fsub(hfx, s.float_from_uint(hx0)), s.float(0.0), s.float(1.0))
    var hty = s.nclamp(s.fsub(hfy, s.float_from_uint(hy0)), s.float(0.0), s.float(1.0))
    var hom_x = s.fsub(s.float(1.0), htx)
    var hom_y = s.fsub(s.float(1.0), hty)
    var hw00 = s.fmul(hom_x, hom_y)
    var hw10 = s.fmul(htx, hom_y)
    var hw01 = s.fmul(hom_x, hty)
    var hw11 = s.fmul(htx, hty)

    var hi00 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy0, dst_w), hx0), s.uint(4)))
    var hi10 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy0, dst_w), hx1), s.uint(4)))
    var hi01 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy1, dst_w), hx0), s.uint(4)))
    var hi11 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy1, dst_w), hx1), s.uint(4)))

    var hr = s.fma(s.load_float(b_hist, hi00), hw00, s.fma(s.load_float(b_hist, hi10), hw10, s.fma(s.load_float(b_hist, hi01), hw01, s.fmul(s.load_float(b_hist, hi11), hw11))))
    var hg = s.fma(s.load_float(b_hist, s.iadd(hi00, s.uint(1))), hw00, s.fma(s.load_float(b_hist, s.iadd(hi10, s.uint(1))), hw10, s.fma(s.load_float(b_hist, s.iadd(hi01, s.uint(1))), hw01, s.fmul(s.load_float(b_hist, s.iadd(hi11, s.uint(1))), hw11))))
    var hb = s.fma(s.load_float(b_hist, s.iadd(hi00, s.uint(2))), hw00, s.fma(s.load_float(b_hist, s.iadd(hi10, s.uint(2))), hw10, s.fma(s.load_float(b_hist, s.iadd(hi01, s.uint(2))), hw01, s.fmul(s.load_float(b_hist, s.iadd(hi11, s.uint(2))), hw11))))
    var hz = s.fma(s.load_float(b_hist, s.iadd(hi00, s.uint(3))), hw00, s.fma(s.load_float(b_hist, s.iadd(hi10, s.uint(3))), hw10, s.fma(s.load_float(b_hist, s.iadd(hi01, s.uint(3))), hw01, s.fmul(s.load_float(b_hist, s.iadd(hi11, s.uint(3))), hw11))))

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

    # Attenuate history weight by `(1 - reactive * 2.0)` AND sub-pixel velocity so high-speed driving has zero motion smear!
    var vpx_x = s.fmul(vel_u, f_dw)
    var vpx_y = s.fmul(vel_v, f_dh)
    var vel_px = s.fsqrt(s.fma(vpx_x, vpx_x, s.fmul(vpx_y, vpx_y)))
    var vel_atten = s.fdiv(s.float(1.0), s.fma(vel_px, s.float(0.38), s.float(1.0)))
    var react_atten = s.nclamp(s.fsub(s.float(1.0), s.fmul(reactive, s.float(2.0))), s.float(0.0), s.float(1.0))
    var hw = s.fmul(base_hist_w, s.fmul(react_atten, vel_atten))
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
    # Supports GPU 64-bin histogram auto-exposure (`b_globals[119] != 0`) + natural `cos^4(theta)` lens vignette (`b_globals[115]`)!
    var base_exp = s.load_float(b_globals, s.uint(47))
    var auto_exp_on = s.ine(s.load_word(b_globals, s.uint(119)), s.uint(0))
    var adapt_exp = s.load_float(b_post, s.uint(UInt32(AAA_POST_ADAPTED_EXP_WORD)))
    var eff_exp = s.select_f(auto_exp_on, s.fmul(base_exp, adapt_exp), base_exp)

    var vig_str = s.load_float(b_globals, s.uint(115))
    var tan_x_v = s.load_float(b_globals, s.uint(39))
    var tan_y_v = s.load_float(b_globals, s.uint(43))
    var nx_v = s.fmul(s.fma(uv_u, s.float(2.0), s.float(-1.0)), tan_x_v)
    var ny_v = s.fmul(s.fma(uv_v, s.float(2.0), s.float(-1.0)), tan_y_v)
    var cos2_th = s.fdiv(s.float(1.0), s.fadd(s.float(1.0), s.fma(nx_v, nx_v, s.fmul(ny_v, ny_v))))
    var cos4_th = s.fmul(cos2_th, cos2_th)
    var vig_mul = s.fma(s.fsub(cos4_th, s.float(1.0)), vig_str, s.float(1.0))
    var exposure = s.fmul(eff_exp, vig_mul)

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
# 4. `FlintTSR v3` Pre-Bloom Temporal Upscaler, `180°` Tile-Max Shutter Motion
#    Blur & Cinema Display Grading (`aaa_tsr_v3_resolve_shader`, etc.)
# ============================================================================

@always_inline
def _emit_hist_bilinear_tap(
    mut s: AaaSpirvModule,
    b_hist: UInt32,
    hist_in_off: UInt32,
    dst_w: UInt32,
    dst_h: UInt32,
    f_dw: UInt32,
    f_dh: UInt32,
    px_x: UInt32,
    px_y: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """Bilinear fetch on `b_hist[hist_in_off]` (`dst_w x dst_h`, 4 floats/px) at texel-space `(px_x, px_y)`."""
    var fx = s.nclamp(s.fsub(px_x, s.float(0.5)), s.float(0.0), s.nmax(s.float(0.0), s.fsub(f_dw, s.float(1.0))))
    var fy = s.nclamp(s.fsub(px_y, s.float(0.5)), s.float(0.0), s.nmax(s.float(0.0), s.fsub(f_dh, s.float(1.0))))
    var hx0 = s.umin(s.isub(dst_w, s.uint(1)), s.saturate_uint(s.ffloor(fx)))
    var hy0 = s.umin(s.isub(dst_h, s.uint(1)), s.saturate_uint(s.ffloor(fy)))
    var hx1 = s.umin(s.isub(dst_w, s.uint(1)), s.iadd(hx0, s.uint(1)))
    var hy1 = s.umin(s.isub(dst_h, s.uint(1)), s.iadd(hy0, s.uint(1)))
    var htx = s.nclamp(s.fsub(fx, s.float_from_uint(hx0)), s.float(0.0), s.float(1.0))
    var hty = s.nclamp(s.fsub(fy, s.float_from_uint(hy0)), s.float(0.0), s.float(1.0))
    var hom_x = s.fsub(s.float(1.0), htx)
    var hom_y = s.fsub(s.float(1.0), hty)
    var hw00 = s.fmul(hom_x, hom_y)
    var hw10 = s.fmul(htx, hom_y)
    var hw01 = s.fmul(hom_x, hty)
    var hw11 = s.fmul(htx, hty)

    var hi00 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy0, dst_w), hx0), s.uint(4)))
    var hi10 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy0, dst_w), hx1), s.uint(4)))
    var hi01 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy1, dst_w), hx0), s.uint(4)))
    var hi11 = s.iadd(hist_in_off, s.imul(s.iadd(s.imul(hy1, dst_w), hx1), s.uint(4)))

    var r = s.fma(s.load_float(b_hist, hi00), hw00, s.fma(s.load_float(b_hist, hi10), hw10, s.fma(s.load_float(b_hist, hi01), hw01, s.fmul(s.load_float(b_hist, hi11), hw11))))
    var g = s.fma(s.load_float(b_hist, s.iadd(hi00, s.uint(1))), hw00, s.fma(s.load_float(b_hist, s.iadd(hi10, s.uint(1))), hw10, s.fma(s.load_float(b_hist, s.iadd(hi01, s.uint(1))), hw01, s.fmul(s.load_float(b_hist, s.iadd(hi11, s.uint(1))), hw11))))
    var b = s.fma(s.load_float(b_hist, s.iadd(hi00, s.uint(2))), hw00, s.fma(s.load_float(b_hist, s.iadd(hi10, s.uint(2))), hw10, s.fma(s.load_float(b_hist, s.iadd(hi01, s.uint(2))), hw01, s.fmul(s.load_float(b_hist, s.iadd(hi11, s.uint(2))), hw11))))
    var z = s.fma(s.load_float(b_hist, s.iadd(hi00, s.uint(3))), hw00, s.fma(s.load_float(b_hist, s.iadd(hi10, s.uint(3))), hw10, s.fma(s.load_float(b_hist, s.iadd(hi01, s.uint(3))), hw01, s.fmul(s.load_float(b_hist, s.iadd(hi11, s.uint(3))), hw11))))
    return (r, g, b, z)


@always_inline
def _emit_catmull_rom_5tap_hist(
    mut s: AaaSpirvModule,
    b_hist: UInt32,
    hist_in_off: UInt32,
    dst_w: UInt32,
    dst_h: UInt32,
    f_dw: UInt32,
    f_dh: UInt32,
    prev_u: UInt32,
    prev_v: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """5-Tap Jimenez Catmull-Rom Bicubic History Fetch preventing recurrent bilinear history blur."""
    var sx = s.fmul(prev_u, f_dw)
    var sy = s.fmul(prev_v, f_dh)
    var tc1_x = s.fadd(s.ffloor(s.fsub(sx, s.float(0.5))), s.float(0.5))
    var tc1_y = s.fadd(s.ffloor(s.fsub(sy, s.float(0.5))), s.float(0.5))
    var fx = s.fsub(sx, tc1_x)
    var fy = s.fsub(sy, tc1_y)
    var fx2 = s.fmul(fx, fx)
    var fy2 = s.fmul(fy, fy)

    var w0_x = s.fmul(fx, s.fma(fx, s.fma(fx, s.float(-0.5), s.float(1.0)), s.float(-0.5)))
    var w1_x = s.fma(fx2, s.fma(fx, s.float(1.5), s.float(-2.5)), s.float(1.0))
    var w2_x = s.fmul(fx, s.fma(fx, s.fma(fx, s.float(-1.5), s.float(2.0)), s.float(0.5)))
    var w3_x = s.fmul(fx2, s.fma(fx, s.float(0.5), s.float(-0.5)))

    var w0_y = s.fmul(fy, s.fma(fy, s.fma(fy, s.float(-0.5), s.float(1.0)), s.float(-0.5)))
    var w1_y = s.fma(fy2, s.fma(fy, s.float(1.5), s.float(-2.5)), s.float(1.0))
    var w2_y = s.fmul(fy, s.fma(fy, s.fma(fy, s.float(-1.5), s.float(2.0)), s.float(0.5)))
    var w3_y = s.fmul(fy2, s.fma(fy, s.float(0.5), s.float(-0.5)))

    var w12_x = s.fadd(w1_x, w2_x)
    var w12_y = s.fadd(w1_y, w2_y)
    var tc12_x = s.fadd(tc1_x, s.fdiv(w2_x, s.nmax(s.float(1e-5), w12_x)))
    var tc12_y = s.fadd(tc1_y, s.fdiv(w2_y, s.nmax(s.float(1e-5), w12_y)))
    var tc0_x = s.fsub(tc1_x, s.float(1.0))
    var tc3_x = s.fadd(tc1_x, s.float(2.0))
    var tc0_y = s.fsub(tc1_y, s.float(1.0))
    var tc3_y = s.fadd(tc1_y, s.float(2.0))

    var t_n = _emit_hist_bilinear_tap(s, b_hist, hist_in_off, dst_w, dst_h, f_dw, f_dh, tc12_x, tc0_y)
    var t_w = _emit_hist_bilinear_tap(s, b_hist, hist_in_off, dst_w, dst_h, f_dw, f_dh, tc0_x, tc12_y)
    var t_c = _emit_hist_bilinear_tap(s, b_hist, hist_in_off, dst_w, dst_h, f_dw, f_dh, tc12_x, tc12_y)
    var t_e = _emit_hist_bilinear_tap(s, b_hist, hist_in_off, dst_w, dst_h, f_dw, f_dh, tc3_x, tc12_y)
    var t_s = _emit_hist_bilinear_tap(s, b_hist, hist_in_off, dst_w, dst_h, f_dw, f_dh, tc12_x, tc3_y)

    var wn = s.fmul(w12_x, w0_y)
    var ww = s.fmul(w0_x, w12_y)
    var wc = s.fmul(w12_x, w12_y)
    var we = s.fmul(w3_x, w12_y)
    var ws = s.fmul(w12_x, w3_y)
    var inv_tw = s.fdiv(s.float(1.0), s.nmax(s.float(1e-5), s.fadd(wn, s.fadd(ww, s.fadd(wc, s.fadd(we, ws))))))

    var r = s.nmax(s.float(0.0), s.fmul(s.fma(t_n[0], wn, s.fma(t_w[0], ww, s.fma(t_c[0], wc, s.fma(t_e[0], we, s.fmul(t_s[0], ws))))), inv_tw))
    var g = s.nmax(s.float(0.0), s.fmul(s.fma(t_n[1], wn, s.fma(t_w[1], ww, s.fma(t_c[1], wc, s.fma(t_e[1], we, s.fmul(t_s[1], ws))))), inv_tw))
    var b = s.nmax(s.float(0.0), s.fmul(s.fma(t_n[2], wn, s.fma(t_w[2], ww, s.fma(t_c[2], wc, s.fma(t_e[2], we, s.fmul(t_s[2], ws))))), inv_tw))
    return (r, g, b, t_c[3])


def aaa_tsr_v3_resolve_shader() -> List[UInt32]:
    """`FlintTSR v3` Pre-Bloom Temporal Super-Resolution & 16-Phase Halton Anti-Aliasing Resolve:
      - Exact camera sub-pixel unjittering + Blackman-Harris 3.3 sample confidence weight
      - 5-Tap Jimenez Catmull-Rom bicubic history fetch
      - Karis-weighted `3x3` YCoCg moment variance ray-clipping `[mu - gamma*sigma, mu + gamma*sigma]`
      - Closest-depth foreground velocity dilation
      - Writes resolved linear HDR + Z to `b_hist[hist_out_off]` AND 8-word HDR+Z+Vel+Norm to `b_post[dst_scratch_off]`!
    Push[0]=src_w, Push[1]=src_h, Push[2]=dst_w, Push[3]=dst_h, Push[4]=history_weight,
    Push[5]=hist_in_off, Push[6]=hist_out_off, Push[7]=dst_scratch_off, Push[8]=sharp_amt.
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var b_oit = s.buffer(BIND_AAA_OIT_NODES)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)

    var dx = s.invocation_axis(0)
    var dy = s.invocation_axis(1)
    var src_w = s.push_word(0)
    var src_h = s.push_word(1)
    var dst_w = s.push_word(2)
    var dst_h = s.push_word(3)
    var base_hist_w = s.push_float(4)
    var hist_in_off = s.push_word(5)
    var hist_out_off = s.push_word(6)
    var dst_scratch_off = s.push_word(7)
    var sharp_amt = s.nclamp(s.push_float(8), s.float(0.0), s.float(1.0))
    s.return_if(s.logical_or(s.uge(dx, dst_w), s.uge(dy, dst_h)))

    var f_sw = s.float_from_uint(src_w)
    var f_sh = s.float_from_uint(src_h)
    var f_dw = s.float_from_uint(dst_w)
    var f_dh = s.float_from_uint(dst_h)

    var uv_u = s.fdiv(s.fadd(s.float_from_uint(dx), s.float(0.5)), f_dw)
    var uv_v = s.fdiv(s.fadd(s.float_from_uint(dy), s.float(0.5)), f_dh)

    # Sub-pixel Halton camera jitter `(jx_px, jy_px)` in `b_globals[151..152]`
    var jx_px = s.load_float(b_globals, s.uint(151))
    var jy_px = s.load_float(b_globals, s.uint(152))

    # Unjittered render-space coordinate (`+jx_px, -jy_px` matches `visbuffer_vertex_shader` clip jitter!)
    var fx_raw = s.fsub(s.fadd(s.fmul(uv_u, f_sw), jx_px), s.float(0.5))
    var fy_raw = s.fsub(s.fsub(s.fmul(uv_v, f_sh), jy_px), s.float(0.5))
    var fx = s.nclamp(fx_raw, s.float(0.0), s.nmax(s.float(0.0), s.fsub(f_sw, s.float(1.0))))
    var fy = s.nclamp(fy_raw, s.float(0.0), s.nmax(s.float(0.0), s.fsub(f_sh, s.float(1.0))))
    var ix0 = s.umin(s.isub(src_w, s.uint(1)), s.saturate_uint(s.ffloor(fx)))
    var iy0 = s.umin(s.isub(src_h, s.uint(1)), s.saturate_uint(s.ffloor(fy)))
    var ix1 = s.umin(s.isub(src_w, s.uint(1)), s.iadd(ix0, s.uint(1)))
    var iy1 = s.umin(s.isub(src_h, s.uint(1)), s.iadd(iy0, s.uint(1)))
    var tx = s.nclamp(s.fsub(fx, s.float_from_uint(ix0)), s.float(0.0), s.float(1.0))
    var ty = s.nclamp(s.fsub(fy, s.float_from_uint(iy0)), s.float(0.0), s.float(1.0))

    # Blackman-Harris 3.3 sub-pixel sample proximity weight `exp(-2.29 * d^2)`
    var near_x = s.ffloor(s.fadd(fx_raw, s.float(0.5)))
    var near_y = s.ffloor(s.fadd(fy_raw, s.float(0.5)))
    var sdx = s.fsub(fx_raw, near_x)
    var sdy = s.fsub(fy_raw, near_y)
    var sub_d2 = s.nmin(s.float(1.0), s.fma(sdx, sdx, s.fmul(sdy, sdy)))
    var bh_weight = s.exp2(s.fmul(sub_d2, s.float(-2.29 * 1.442695)))

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

    # Closest-depth dilation across 3x3 render neighborhood for crisp foreground silhouette velocity!
    var ix_c = s.umin(s.isub(src_w, s.uint(1)), s.saturate_uint(s.nmax(s.float(0.0), near_x)))
    var iy_c = s.umin(s.isub(src_h, s.uint(1)), s.saturate_uint(s.nmax(s.float(0.0), near_y)))
    var p_center = s.imul(s.iadd(s.imul(iy_c, src_w), ix_c), s.uint(8))
    var best_p = s.uvar(p_center)
    var best_z = s.fvar(s.load_float(b_hdr, s.iadd(p_center, s.uint(3))))

    # Karis-weighted 3x3 YCoCg moments (m1, m2) + RGB anti-ringing bounds
    var sum_wk = s.fvar(s.float(0.0))
    var m1_y = s.fvar(s.float(0.0)); var m2_y = s.fvar(s.float(0.0))
    var m1_co = s.fvar(s.float(0.0)); var m2_co = s.fvar(s.float(0.0))
    var m1_cg = s.fvar(s.float(0.0)); var m2_cg = s.fvar(s.float(0.0))
    var min_r = s.fvar(s.float(1e9)); var max_r = s.fvar(s.float(-1e9))
    var min_g = s.fvar(s.float(1e9)); var max_g = s.fvar(s.float(-1e9))
    var min_b = s.fvar(s.float(1e9)); var max_b = s.fvar(s.float(-1e9))
    var lap_r = s.fvar(s.float(0.0)); var lap_g = s.fvar(s.float(0.0)); var lap_b = s.fvar(s.float(0.0))

    for ny in range(-1, 2):
        for nx in range(-1, 2):
            var sx_n = ix_c
            if nx < 0:
                sx_n = s.select_u(s.uge(ix_c, s.uint(1)), s.isub(ix_c, s.uint(1)), s.uint(0))
            elif nx > 0:
                sx_n = s.umin(s.isub(src_w, s.uint(1)), s.iadd(ix_c, s.uint(1)))
            var sy_n = iy_c
            if ny < 0:
                sy_n = s.select_u(s.uge(iy_c, s.uint(1)), s.isub(iy_c, s.uint(1)), s.uint(0))
            elif ny > 0:
                sy_n = s.umin(s.isub(src_h, s.uint(1)), s.iadd(iy_c, s.uint(1)))
            var np = s.imul(s.iadd(s.imul(sy_n, src_w), sx_n), s.uint(8))
            var nr = s.load_float(b_hdr, np)
            var ng = s.load_float(b_hdr, s.iadd(np, s.uint(1)))
            var nb = s.load_float(b_hdr, s.iadd(np, s.uint(2)))
            var nz = s.load_float(b_hdr, s.iadd(np, s.uint(3)))

            s.begin_if(s.flt(nz, s.load_f(best_z)))
            s.store(best_z, nz)
            s.store(best_p, np)
            s.end_if()

            if (nx == 0 and abs(ny) == 1) or (ny == 0 and abs(nx) == 1):
                s.store(lap_r, s.fadd(s.load_f(lap_r), nr))
                s.store(lap_g, s.fadd(s.load_f(lap_g), ng))
                s.store(lap_b, s.fadd(s.load_f(lap_b), nb))

            s.store(min_r, s.nmin(s.load_f(min_r), nr)); s.store(max_r, s.nmax(s.load_f(max_r), nr))
            s.store(min_g, s.nmin(s.load_f(min_g), ng)); s.store(max_g, s.nmax(s.load_f(max_g), ng))
            s.store(min_b, s.nmin(s.load_f(min_b), nb)); s.store(max_b, s.nmax(s.load_f(max_b), nb))

            var ny_c = s.fma(nr, s.float(0.25), s.fma(ng, s.float(0.5), s.fmul(nb, s.float(0.25))))
            var nco = s.fmul(s.fsub(nr, nb), s.float(0.5))
            var ncg = s.fma(ng, s.float(0.5), s.fmul(s.fadd(nr, nb), s.float(-0.25)))
            var wk = s.fdiv(s.float(1.0), s.fma(s.nmax(s.float(0.0), ny_c), s.float(0.25), s.float(1.0)))
            s.store(sum_wk, s.fadd(s.load_f(sum_wk), wk))
            s.store(m1_y, s.fma(ny_c, wk, s.load_f(m1_y)))
            s.store(m2_y, s.fma(s.fmul(ny_c, ny_c), wk, s.load_f(m2_y)))
            s.store(m1_co, s.fma(nco, wk, s.load_f(m1_co)))
            s.store(m2_co, s.fma(s.fmul(nco, nco), wk, s.load_f(m2_co)))
            s.store(m1_cg, s.fma(ncg, wk, s.load_f(m1_cg)))
            s.store(m2_cg, s.fma(s.fmul(ncg, ncg), wk, s.load_f(m2_cg)))

    var bp = s.load_u(best_p)
    var cur_z = s.load_f(best_z)
    var vel_u = s.load_float(b_hdr, s.iadd(bp, s.uint(4)))
    var vel_v = s.load_float(b_hdr, s.iadd(bp, s.uint(5)))
    var cur_nx = s.load_float(b_hdr, s.iadd(bp, s.uint(6)))
    var cur_ny = s.load_float(b_hdr, s.iadd(bp, s.uint(7)))

    var raw_sr = s.fma(s.fsub(sr, s.fmul(s.load_f(lap_r), s.float(0.25))), sharp_amt, sr)
    var raw_sg = s.fma(s.fsub(sg, s.fmul(s.load_f(lap_g), s.float(0.25))), sharp_amt, sg)
    var raw_sb = s.fma(s.fsub(sb, s.fmul(s.load_f(lap_b), s.float(0.25))), sharp_amt, sb)
    var sharp_r = s.nmax(s.float(0.0), s.nclamp(raw_sr, s.load_f(min_r), s.load_f(max_r)))
    var sharp_g = s.nmax(s.float(0.0), s.nclamp(raw_sg, s.load_f(min_g), s.load_f(max_g)))
    var sharp_b = s.nmax(s.float(0.0), s.nclamp(raw_sb, s.load_f(min_b), s.load_f(max_b)))

    var res_r = s.fvar(sharp_r)
    var res_g = s.fvar(sharp_g)
    var res_b = s.fvar(sharp_b)

    var src_pix_idx = s.iadd(s.imul(iy_c, src_w), ix_c)
    var reactive = s.load_float(b_oit, s.iadd(s.uint(UInt32(AAA_OIT_REACTIVE_BASE_WORDS)), src_pix_idx))

    var prev_u = s.fsub(uv_u, vel_u)
    var prev_v = s.fsub(uv_v, vel_v)
    var valid_reproj = s.logical_and(
        s.logical_and(s.fgt(base_hist_w, s.float(0.001)), s.flt(reactive, s.float(0.55))),
        s.logical_and(
            s.logical_and(s.fge(prev_u, s.float(0.0)), s.flt(prev_u, s.float(1.0))),
            s.logical_and(s.fge(prev_v, s.float(0.0)), s.flt(prev_v, s.float(1.0))),
        ),
    )

    s.begin_if(valid_reproj)
    var h_cat = _emit_catmull_rom_5tap_hist(s, b_hist, hist_in_off, dst_w, dst_h, f_dw, f_dh, prev_u, prev_v)
    var hr = h_cat[0]; var hg = h_cat[1]; var hb = h_cat[2]; var hz = h_cat[3]

    var rel_dz = s.fdiv(s.fabs(s.fsub(cur_z, hz)), s.nmax(s.float(0.1), cur_z))
    s.begin_if(s.flt(rel_dz, s.float(0.10)))
    var vpx_x = s.fmul(vel_u, f_dw)
    var vpx_y = s.fmul(vel_v, f_dh)
    var vel_px = s.fsqrt(s.fma(vpx_x, vpx_x, s.fmul(vpx_y, vpx_y)))

    # Karis YCoCg Moment Variance Ray-Clipping `[mu - gamma*sigma, mu + gamma*sigma]` toward `mu`
    var inv_swk = s.fdiv(s.float(1.0), s.nmax(s.float(1e-5), s.load_f(sum_wk)))
    var mu_y = s.fmul(s.load_f(m1_y), inv_swk)
    var mu_co = s.fmul(s.load_f(m1_co), inv_swk)
    var mu_cg = s.fmul(s.load_f(m1_cg), inv_swk)
    var sig_y = s.fsqrt(s.nmax(s.float(1e-7), s.fsub(s.fmul(s.load_f(m2_y), inv_swk), s.fmul(mu_y, mu_y))))
    var sig_co = s.fsqrt(s.nmax(s.float(1e-7), s.fsub(s.fmul(s.load_f(m2_co), inv_swk), s.fmul(mu_co, mu_co))))
    var sig_cg = s.fsqrt(s.nmax(s.float(1e-7), s.fsub(s.fmul(s.load_f(m2_cg), inv_swk), s.fmul(mu_cg, mu_cg))))

    var gamma = s.nclamp(s.fma(vel_px, s.float(-0.08), s.float(1.22)), s.float(0.85), s.float(1.25))
    var ey = s.nmax(s.float(1e-4), s.fmul(gamma, sig_y))
    var eco = s.nmax(s.float(1e-4), s.fmul(gamma, sig_co))
    var ecg = s.nmax(s.float(1e-4), s.fmul(gamma, sig_cg))

    var hy_raw = s.fma(hr, s.float(0.25), s.fma(hg, s.float(0.5), s.fmul(hb, s.float(0.25))))
    var hco_raw = s.fmul(s.fsub(hr, hb), s.float(0.5))
    var hcg_raw = s.fma(hg, s.float(0.5), s.fmul(s.fadd(hr, hb), s.float(-0.25)))
    var d_y = s.fsub(hy_raw, mu_y)
    var d_co = s.fsub(hco_raw, mu_co)
    var d_cg = s.fsub(hcg_raw, mu_cg)
    var max_rat = s.nmax(s.float(1.0), s.nmax(s.fdiv(s.fabs(d_y), ey), s.nmax(s.fdiv(s.fabs(d_co), eco), s.fdiv(s.fabs(d_cg), ecg))))
    var clip_s = s.fdiv(s.float(1.0), max_rat)
    var hy_c = s.fma(d_y, clip_s, mu_y)
    var hco = s.fma(d_co, clip_s, mu_co)
    var hcg = s.fma(d_cg, clip_s, mu_cg)

    var tmp_yc = s.fsub(hy_c, hcg)
    var chr = s.nmax(s.float(0.0), s.fadd(tmp_yc, hco))
    var chg = s.nmax(s.float(0.0), s.fadd(hy_c, hcg))
    var chb = s.nmax(s.float(0.0), s.fsub(tmp_yc, hco))

    var vel_atten = s.fdiv(s.float(1.0), s.fma(vel_px, s.float(0.32), s.float(1.0)))
    var react_atten = s.nclamp(s.fsub(s.float(1.0), s.fmul(reactive, s.float(1.85))), s.float(0.0), s.float(1.0))
    var raw_hw = s.fmul(base_hist_w, s.fmul(react_atten, vel_atten))
    var bh_curr_scale = s.fma(s.float(0.60), bh_weight, s.float(0.55))
    var curr_w = s.nclamp(s.fmul(s.fsub(s.float(1.0), raw_hw), bh_curr_scale), s.float(0.04), s.float(1.0))
    var hw = s.fsub(s.float(1.0), curr_w)
    s.store(res_r, s.fma(chr, hw, s.fmul(sharp_r, curr_w)))
    s.store(res_g, s.fma(chg, hw, s.fmul(sharp_g, curr_w)))
    s.store(res_b, s.fma(chb, hw, s.fmul(sharp_b, curr_w)))
    s.end_if()
    s.end_if()

    var fin_r = s.load_f(res_r)
    var fin_g = s.load_f(res_g)
    var fin_b = s.load_f(res_b)

    var dst_pix = s.iadd(s.imul(dy, dst_w), dx)
    var ho_idx = s.iadd(hist_out_off, s.imul(dst_pix, s.uint(4)))
    s.store_float(b_hist, ho_idx, fin_r)
    s.store_float(b_hist, s.iadd(ho_idx, s.uint(1)), fin_g)
    s.store_float(b_hist, s.iadd(ho_idx, s.uint(2)), fin_b)
    s.store_float(b_hist, s.iadd(ho_idx, s.uint(3)), cur_z)

    var post_out = s.iadd(dst_scratch_off, s.imul(dst_pix, s.uint(8)))
    s.store_float(b_post, post_out, fin_r)
    s.store_float(b_post, s.iadd(post_out, s.uint(1)), fin_g)
    s.store_float(b_post, s.iadd(post_out, s.uint(2)), fin_b)
    s.store_float(b_post, s.iadd(post_out, s.uint(3)), cur_z)
    s.store_float(b_post, s.iadd(post_out, s.uint(4)), vel_u)
    s.store_float(b_post, s.iadd(post_out, s.uint(5)), vel_v)
    s.store_float(b_post, s.iadd(post_out, s.uint(6)), cur_nx)
    s.store_float(b_post, s.iadd(post_out, s.uint(7)), cur_ny)
    return s.finish()


def aaa_motion_blur_shader() -> List[UInt32]:
    """Tile-Max `180°` Rotary-Shutter Camera & Per-Object Motion Blur (`group = 16 x 16`).
      - `16 x 16` LDS `atomic_umax_workgroup_u32` Tile-Max `< 0.75 px` uniform early-out
      - 9-Tap Jimenez Soft-Z Bilateral Gather preserving razor-sharp foreground player vehicles!
    Push[0]=out_w, Push[1]=out_h, Push[2]=src_copy_off (in `b_post`, 8 words/px),
    Push[3]=shutter_scale (`0.5` = `180°`), Push[4]=max_blur_px (`24.0`).
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=16, group_height=16, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)
    var lds_vmax = s.workgroup_array_u32(1)

    var lx = s.local_axis(0)
    var ly = s.local_axis(1)
    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var src_copy_off = s.push_word(2)
    var shutter_scale = s.nclamp(s.push_float(3), s.float(0.0), s.float(2.0))
    var max_blur_px = s.nmax(s.float(1.0), s.push_float(4))

    s.begin_if(s.logical_and(s.ieq(lx, s.uint(0)), s.ieq(ly, s.uint(0))))
    s.store_workgroup_u32(lds_vmax[0], lds_vmax[1], s.uint(0), s.uint(0))
    s.end_if()
    s.workgroup_barrier()

    var in_bounds = s.logical_and(s.ult(px, w_u), s.ult(py, h_u))
    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var c_vx = s.fvar(s.float(0.0))
    var c_vy = s.fvar(s.float(0.0))
    var c_vlen = s.fvar(s.float(0.0))
    var c_z_var = s.fvar(s.float(1000.0))
    var c_r_var = s.fvar(s.float(0.0))
    var c_g_var = s.fvar(s.float(0.0))
    var c_b_var = s.fvar(s.float(0.0))

    s.begin_if(in_bounds)
    var c_base = s.iadd(src_copy_off, s.imul(s.iadd(s.imul(py, w_u), px), s.uint(8)))
    s.store(c_r_var, s.load_float(b_post, c_base))
    s.store(c_g_var, s.load_float(b_post, s.iadd(c_base, s.uint(1))))
    s.store(c_b_var, s.load_float(b_post, s.iadd(c_base, s.uint(2))))
    s.store(c_z_var, s.load_float(b_post, s.iadd(c_base, s.uint(3))))
    var vu = s.load_float(b_post, s.iadd(c_base, s.uint(4)))
    var vv = s.load_float(b_post, s.iadd(c_base, s.uint(5)))
    var vx_raw = s.fmul(s.fmul(vu, fw), shutter_scale)
    var vy_raw = s.fmul(s.fmul(vv, fh), shutter_scale)
    var vl_raw = s.fsqrt(s.fma(vx_raw, vx_raw, s.fmul(vy_raw, vy_raw)))
    var vl_clamped = s.nmin(max_blur_px, vl_raw)
    var v_scale = s.fdiv(vl_clamped, s.nmax(s.float(1e-4), vl_raw))
    s.store(c_vx, s.fmul(vx_raw, v_scale))
    s.store(c_vy, s.fmul(vy_raw, v_scale))
    s.store(c_vlen, vl_clamped)
    _ = s.atomic_umax_workgroup_u32(lds_vmax[0], lds_vmax[1], s.uint(0), s.bitcast_uint(vl_clamped))
    s.end_if()
    s.workgroup_barrier()

    var tile_max_vel = s.bitcast_float(s.load_workgroup_u32(lds_vmax[0], lds_vmax[1], s.uint(0)))
    s.return_if(s.logical_or(s.logical_not(in_bounds), s.flt(tile_max_vel, s.float(0.75))))

    var dom_vx = s.fvar(s.load_f(c_vx))
    var dom_vy = s.fvar(s.load_f(c_vy))
    var dom_vl = s.fvar(s.load_f(c_vlen))
    var fpx = s.float_from_uint(px)
    var fpy = s.float_from_uint(py)

    # 4-way dominant neighbor velocity probe (`±4 px`) for smooth moving-silhouette spread
    for d_pair in [(-4, 0), (4, 0), (0, -4), (0, 4)]:
        var nx_u = s.umin(s.isub(w_u, s.uint(1)), s.saturate_uint(s.nmax(s.float(0.0), s.fadd(fpx, s.float(Float32(d_pair[0]))))))
        var ny_u = s.umin(s.isub(h_u, s.uint(1)), s.saturate_uint(s.nmax(s.float(0.0), s.fadd(fpy, s.float(Float32(d_pair[1]))))))
        var nb_off = s.iadd(src_copy_off, s.imul(s.iadd(s.imul(ny_u, w_u), nx_u), s.uint(8)))
        var nvu = s.load_float(b_post, s.iadd(nb_off, s.uint(4)))
        var nvv = s.load_float(b_post, s.iadd(nb_off, s.uint(5)))
        var nvx_r = s.fmul(s.fmul(nvu, fw), shutter_scale)
        var nvy_r = s.fmul(s.fmul(nvv, fh), shutter_scale)
        var nvl_r = s.fsqrt(s.fma(nvx_r, nvx_r, s.fmul(nvy_r, nvy_r)))
        var nvl_c = s.nmin(max_blur_px, nvl_r)
        s.begin_if(s.fgt(nvl_c, s.load_f(dom_vl)))
        var n_sc = s.fdiv(nvl_c, s.nmax(s.float(1e-4), nvl_r))
        s.store(dom_vl, nvl_c)
        s.store(dom_vx, s.fmul(nvx_r, n_sc))
        s.store(dom_vy, s.fmul(nvy_r, n_sc))
        s.end_if()

    var dv_len = s.load_f(dom_vl)
    s.return_if(s.flt(dv_len, s.float(0.75)))

    var dv_x = s.load_f(dom_vx)
    var dv_y = s.load_f(dom_vy)
    var center_vl = s.load_f(c_vlen)
    var center_z = s.nmax(s.float(0.05), s.load_f(c_z_var))
    var acc_r = s.fvar(s.load_f(c_r_var))
    var acc_g = s.fvar(s.load_f(c_g_var))
    var acc_b = s.fvar(s.load_f(c_b_var))
    var acc_w = s.fvar(s.float(1.0))

    # 9-Tap IGN-Stratified Bilinear Jimenez Soft-Z Gather (`t in -4..4`, zero discrete 9-copy ghost banding!)
    var frame_mb = s.float_from_uint(s.bit_and(s.load_word(b_globals, s.uint(100)), s.uint(63)))
    var ign_mb = s.fract(
        s.fmul(
            s.float(52.9829189),
            s.fract(s.fma(fpx, s.float(0.06711056), s.fma(fpy, s.float(0.00583715), s.fmul(frame_mb, s.float(0.38196601))))),
        )
    )
    var jit_mb = s.fmul(s.fsub(ign_mb, s.float(0.5)), s.float(0.85))
    var max_fx = s.nmax(s.float(0.0), s.fsub(fw, s.float(1.0)))
    var max_fy = s.nmax(s.float(0.0), s.fsub(fh, s.float(1.0)))

    for tap_i in [-4, -3, -2, -1, 1, 2, 3, 4]:
        var frac_v = s.fmul(s.fadd(s.float(Float32(tap_i)), jit_mb), s.float(0.11111111))
        var abs_frac_v = s.fabs(frac_v)
        var sx_f = s.nclamp(s.fma(dv_x, frac_v, fpx), s.float(0.0), max_fx)
        var sy_f = s.nclamp(s.fma(dv_y, frac_v, fpy), s.float(0.0), max_fy)

        # Bilinear RGB fetch at `(sx_f, sy_f)` + nearest depth/velocity Soft-Z classification
        var bx0 = s.umin(s.isub(w_u, s.uint(1)), s.saturate_uint(s.ffloor(sx_f)))
        var by0 = s.umin(s.isub(h_u, s.uint(1)), s.saturate_uint(s.ffloor(sy_f)))
        var bx1 = s.umin(s.isub(w_u, s.uint(1)), s.iadd(bx0, s.uint(1)))
        var by1 = s.umin(s.isub(h_u, s.uint(1)), s.iadd(by0, s.uint(1)))
        var btx = s.nclamp(s.fsub(sx_f, s.float_from_uint(bx0)), s.float(0.0), s.float(1.0))
        var bty = s.nclamp(s.fsub(sy_f, s.float_from_uint(by0)), s.float(0.0), s.float(1.0))
        var bomx = s.fsub(s.float(1.0), btx)
        var bomy = s.fsub(s.float(1.0), bty)
        var bw00 = s.fmul(bomx, bomy); var bw10 = s.fmul(btx, bomy)
        var bw01 = s.fmul(bomx, bty); var bw11 = s.fmul(btx, bty)

        var o00 = s.iadd(src_copy_off, s.imul(s.iadd(s.imul(by0, w_u), bx0), s.uint(8)))
        var o10 = s.iadd(src_copy_off, s.imul(s.iadd(s.imul(by0, w_u), bx1), s.uint(8)))
        var o01 = s.iadd(src_copy_off, s.imul(s.iadd(s.imul(by1, w_u), bx0), s.uint(8)))
        var o11 = s.iadd(src_copy_off, s.imul(s.iadd(s.imul(by1, w_u), bx1), s.uint(8)))

        var s_r = s.fma(s.load_float(b_post, o00), bw00, s.fma(s.load_float(b_post, o10), bw10, s.fma(s.load_float(b_post, o01), bw01, s.fmul(s.load_float(b_post, o11), bw11))))
        var s_g = s.fma(s.load_float(b_post, s.iadd(o00, s.uint(1))), bw00, s.fma(s.load_float(b_post, s.iadd(o10, s.uint(1))), bw10, s.fma(s.load_float(b_post, s.iadd(o01, s.uint(1))), bw01, s.fmul(s.load_float(b_post, s.iadd(o11, s.uint(1))), bw11))))
        var s_b = s.fma(s.load_float(b_post, s.iadd(o00, s.uint(2))), bw00, s.fma(s.load_float(b_post, s.iadd(o10, s.uint(2))), bw10, s.fma(s.load_float(b_post, s.iadd(o01, s.uint(2))), bw01, s.fmul(s.load_float(b_post, s.iadd(o11, s.uint(2))), bw11))))

        var sx_u = s.umin(s.isub(w_u, s.uint(1)), s.saturate_uint(s.fadd(sx_f, s.float(0.5))))
        var sy_u = s.umin(s.isub(h_u, s.uint(1)), s.saturate_uint(s.fadd(sy_f, s.float(0.5))))
        var s_off = s.iadd(src_copy_off, s.imul(s.iadd(s.imul(sy_u, w_u), sx_u), s.uint(8)))
        var s_z = s.nmax(s.float(0.05), s.load_float(b_post, s.iadd(s_off, s.uint(3))))
        var s_vu = s.load_float(b_post, s.iadd(s_off, s.uint(4)))
        var s_vv = s.load_float(b_post, s.iadd(s_off, s.uint(5)))
        var s_vx = s.fmul(s.fmul(s_vu, fw), shutter_scale)
        var s_vy = s.fmul(s.fmul(s_vv, fh), shutter_scale)
        var s_vl = s.nmin(max_blur_px, s.fsqrt(s.fma(s_vx, s_vx, s.fmul(s_vy, s_vy))))

        var dist_px = s.nmax(s.float(0.25), s.fmul(dv_len, abs_frac_v))
        var spread_c = s.nclamp(s.fadd(s.fsub(center_vl, dist_px), s.float(1.0)), s.float(0.0), s.float(1.0))
        var spread_s = s.nclamp(s.fadd(s.fsub(s_vl, dist_px), s.float(1.0)), s.float(0.0), s.float(1.0))

        var z_scale = s.fdiv(s.float(2.0), s.nmax(s.float(0.20), s.nmin(center_z, s.nmin(s_z, s.float(80.0)))))
        var fg = s.nclamp(s.fma(s.fsub(center_z, s_z), z_scale, s.float(1.0)), s.float(0.0), s.float(1.0))
        var bg = s.nclamp(s.fma(s.fsub(s_z, center_z), z_scale, s.float(1.0)), s.float(0.0), s.float(1.0))
        var same_z = s.nclamp(s.fsub(s.float(1.0), s.fmul(s.fabs(s.fsub(s_z, center_z)), z_scale)), s.float(0.0), s.float(1.0))

        var w_tap = s.nclamp(
            s.fma(fg, spread_s, s.fma(bg, spread_c, s.fmul(same_z, s.nmax(spread_c, spread_s)))),
            s.float(0.0),
            s.float(1.0),
        )
        s.store(acc_r, s.fma(s_r, w_tap, s.load_f(acc_r)))
        s.store(acc_g, s.fma(s_g, w_tap, s.load_f(acc_g)))
        s.store(acc_b, s.fma(s_b, w_tap, s.load_f(acc_b)))
        s.store(acc_w, s.fadd(s.load_f(acc_w), w_tap))

    var inv_w = s.fdiv(s.float(1.0), s.nmax(s.float(1e-4), s.load_f(acc_w)))
    var h_out = s.imul(s.iadd(s.imul(py, w_u), px), s.uint(8))
    s.store_float(b_hdr, h_out, s.fmul(s.load_f(acc_r), inv_w))
    s.store_float(b_hdr, s.iadd(h_out, s.uint(1)), s.fmul(s.load_f(acc_g), inv_w))
    s.store_float(b_hdr, s.iadd(h_out, s.uint(2)), s.fmul(s.load_f(acc_b), inv_w))
    return s.finish()


def aaa_display_tonemap_shader() -> List[UInt32]:
    """Final Cinema Display Output Pass (`group = 8 x 8` at native `output_w x output_h`):
      - Optional Cockpit Windshield Refractive Raindrops (`b_globals[126] > 0.001`)
      - Physical `EV100` / Bounded Auto-Exposure + `cos^4(theta)` Lens Vignette
      - `AgX Rec.709` (`+5.5 EV` shoulder + `t0=0.018` shadow toe + Purkinje desat) | `HDR10 PQ` | `Linear sRGB`
    Push[0]=out_w, Push[1]=out_h, Push[2]=out_ldr_off, Push[3]=display_mode,
    Push[4..7]=bloom mip, Push[8]=saturation - 1, Push[9]=contrast - 1 (display grade, SDR only).
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)

    var dx = s.invocation_axis(0)
    var dy = s.invocation_axis(1)
    var dst_w = s.push_word(0)
    var dst_h = s.push_word(1)
    var out_ldr_off = s.push_word(2)
    var disp_mode = s.push_word(3)
    var has_bloom_mip = s.ine(s.push_word(4), s.uint(0))
    var bm_w = s.push_word(5)
    var bm_h = s.push_word(6)
    var bm_off = s.push_word(7)
    var grade_sat = s.push_float(8)
    var grade_con = s.push_float(9)
    s.return_if(s.logical_or(s.uge(dx, dst_w), s.uge(dy, dst_h)))

    var f_dw = s.float_from_uint(dst_w)
    var f_dh = s.float_from_uint(dst_h)
    var uv_u = s.fdiv(s.fadd(s.float_from_uint(dx), s.float(0.5)), f_dw)
    var uv_v = s.fdiv(s.fadd(s.float_from_uint(dy), s.float(0.5)), f_dh)

    var dst_pix = s.iadd(s.imul(dy, dst_w), dx)
    var h_base = s.imul(dst_pix, s.uint(8))
    var base_r = s.load_float(b_hdr, h_base)
    var base_g = s.load_float(b_hdr, s.iadd(h_base, s.uint(1)))
    var base_b = s.load_float(b_hdr, s.iadd(h_base, s.uint(2)))
    var cur_z = s.load_float(b_hdr, s.iadd(h_base, s.uint(3)))
    var fin_r = s.fvar(base_r)
    var fin_g = s.fvar(base_g)
    var fin_b = s.fvar(base_b)

    # Cockpit POV Windshield Bokeh Raindrops (`b_globals[126]`), masked off exterior car hood (`z < 2.6m`)!
    var ws_rain_raw = s.load_float(b_globals, s.uint(126))
    var ht_m = s.nclamp(s.fmul(s.fsub(cur_z, s.float(2.6)), s.float(1.0 / 1.2)), s.float(0.0), s.float(1.0))
    var hood_mask = s.fmul(s.fmul(ht_m, ht_m), s.fma(ht_m, s.float(-2.0), s.float(3.0)))
    var ws_rain = s.fmul(ws_rain_raw, hood_mask)
    s.begin_if(s.fgt(ws_rain, s.float(0.001)))
    var time_s = s.load_float(b_globals, s.uint(123))
    var ref_dx = s.fvar(s.float(0.0))
    var ref_dy = s.fvar(s.float(0.0))
    var drop_top_hl = s.fvar(s.float(0.0))
    for layer_i in range(2):
        var gx_s = Float32(22.0) if layer_i == 0 else Float32(37.0)
        var gy_s = Float32(13.0) if layer_i == 0 else Float32(21.0)
        var spd = Float32(0.42) if layer_i == 0 else Float32(0.68)
        var su = s.fmul(uv_u, s.float(gx_s))
        var sv = s.fsub(s.fmul(uv_v, s.float(gy_s)), s.fmul(time_s, s.float(spd)))
        var cx_f = s.ffloor(su)
        var cy_f = s.ffloor(sv)
        var lx_d = s.fsub(s.fsub(su, cx_f), s.float(0.5))
        var ly_d = s.fsub(s.fsub(sv, cy_f), s.float(0.5))
        var h_seed = s.fadd(s.fmul(cx_f, s.float(127.1)), s.fma(cy_f, s.float(311.7), s.float(Float32(layer_i) * 74.7)))
        var rnd0 = s.fract(s.fmul(s.fsin(h_seed), s.float(43758.5453)))
        var rnd1 = s.fract(s.fmul(s.fsin(s.fadd(h_seed, s.float(19.31))), s.float(24634.6345)))
        var wob = s.fmul(s.fsin(s.fma(uv_v, s.float(18.0), s.fmul(rnd0, s.float(6.2831853)))), s.float(0.14))
        var dx_d = s.fsub(lx_d, s.fma(s.fsub(rnd1, s.float(0.5)), s.float(0.32), wob))
        var dy_d = s.fmul(ly_d, s.float(0.72))  # Teardrop vertical elongation
        var r2_d = s.fma(dx_d, dx_d, s.fmul(dy_d, dy_d))
        var drop_r = s.fmul(s.float(0.24), s.nclamp(ws_rain, s.float(0.2), s.float(1.0)))
        var drop_r2 = s.fmul(drop_r, drop_r)
        var active_d = s.logical_and(s.flt(rnd0, s.fmul(ws_rain, s.float(0.62))), s.flt(r2_d, drop_r2))
        s.begin_if(active_d)
        var nz_d = s.fsqrt(s.nmax(s.float(1e-4), s.fsub(drop_r2, r2_d)))
        var inv_dr = s.fdiv(s.float(1.0), s.nmax(s.float(1e-4), drop_r))
        var n_dx = s.fmul(dx_d, inv_dr)
        var n_dy = s.fmul(dy_d, inv_dr)
        var rim_d = s.fsub(s.float(1.0), s.fmul(nz_d, inv_dr))
        var top_cresc = s.fmul(s.nmax(s.float(0.0), s.fneg(n_dy)), s.fmul(rim_d, rim_d))
        s.store(ref_dx, s.fma(n_dx, s.float(-0.026), s.load_f(ref_dx)))
        s.store(ref_dy, s.fma(n_dy, s.float(-0.026), s.load_f(ref_dy)))
        s.store(drop_top_hl, s.fadd(top_cresc, s.load_f(drop_top_hl)))
        s.end_if()

    var rdx = s.load_f(ref_dx)
    var rdy = s.load_f(ref_dy)
    s.begin_if(s.fgt(s.fadd(s.fabs(rdx), s.fabs(rdy)), s.float(1e-5)))
    var ru = s.nclamp(s.fadd(uv_u, rdx), s.float(0.0), s.float(0.9999))
    var rv = s.nclamp(s.fadd(uv_v, rdy), s.float(0.0), s.float(0.9999))

    # Bilinear fetch of refracted `b_hdr`
    var hfx = s.nclamp(s.fsub(s.fmul(ru, f_dw), s.float(0.5)), s.float(0.0), s.nmax(s.float(0.0), s.fsub(f_dw, s.float(1.0))))
    var hfy = s.nclamp(s.fsub(s.fmul(rv, f_dh), s.float(0.5)), s.float(0.0), s.nmax(s.float(0.0), s.fsub(f_dh, s.float(1.0))))
    var hx0 = s.umin(s.isub(dst_w, s.uint(1)), s.saturate_uint(s.ffloor(hfx)))
    var hy0 = s.umin(s.isub(dst_h, s.uint(1)), s.saturate_uint(s.ffloor(hfy)))
    var hx1 = s.umin(s.isub(dst_w, s.uint(1)), s.iadd(hx0, s.uint(1)))
    var hy1 = s.umin(s.isub(dst_h, s.uint(1)), s.iadd(hy0, s.uint(1)))
    var htx = s.nclamp(s.fsub(hfx, s.float_from_uint(hx0)), s.float(0.0), s.float(1.0))
    var hty = s.nclamp(s.fsub(hfy, s.float_from_uint(hy0)), s.float(0.0), s.float(1.0))
    var homx = s.fsub(s.float(1.0), htx); var homy = s.fsub(s.float(1.0), hty)
    var hw00 = s.fmul(homx, homy); var hw10 = s.fmul(htx, homy)
    var hw01 = s.fmul(homx, hty); var hw11 = s.fmul(htx, hty)
    var p00 = s.imul(s.iadd(s.imul(hy0, dst_w), hx0), s.uint(8))
    var p10 = s.imul(s.iadd(s.imul(hy0, dst_w), hx1), s.uint(8))
    var p01 = s.imul(s.iadd(s.imul(hy1, dst_w), hx0), s.uint(8))
    var p11 = s.imul(s.iadd(s.imul(hy1, dst_w), hx1), s.uint(8))

    var rf_r = s.fvar(s.fma(s.load_float(b_hdr, p00), hw00, s.fma(s.load_float(b_hdr, p10), hw10, s.fma(s.load_float(b_hdr, p01), hw01, s.fmul(s.load_float(b_hdr, p11), hw11)))))
    var rf_g = s.fvar(s.fma(s.load_float(b_hdr, s.iadd(p00, s.uint(1))), hw00, s.fma(s.load_float(b_hdr, s.iadd(p10, s.uint(1))), hw10, s.fma(s.load_float(b_hdr, s.iadd(p01, s.uint(1))), hw01, s.fmul(s.load_float(b_hdr, s.iadd(p11, s.uint(1))), hw11)))))
    var rf_b = s.fvar(s.fma(s.load_float(b_hdr, s.iadd(p00, s.uint(2))), hw00, s.fma(s.load_float(b_hdr, s.iadd(p10, s.uint(2))), hw10, s.fma(s.load_float(b_hdr, s.iadd(p01, s.uint(2))), hw01, s.fmul(s.load_float(b_hdr, s.iadd(p11, s.uint(2))), hw11)))))

    # Blend with defocused `Bloom Mip 1` (`55%`) so droplets act as true wide-angle optical bokeh micro-lenses!
    s.begin_if(s.logical_and(has_bloom_mip, s.logical_and(s.ine(bm_w, s.uint(0)), s.ine(bm_h, s.uint(0)))))
    var f_bw = s.float_from_uint(bm_w)
    var f_bh = s.float_from_uint(bm_h)
    var bfx = s.nclamp(s.fsub(s.fmul(ru, f_bw), s.float(0.5)), s.float(0.0), s.nmax(s.float(0.0), s.fsub(f_bw, s.float(1.0))))
    var bfy = s.nclamp(s.fsub(s.fmul(rv, f_bh), s.float(0.5)), s.float(0.0), s.nmax(s.float(0.0), s.fsub(f_bh, s.float(1.0))))
    var bx0 = s.umin(s.isub(bm_w, s.uint(1)), s.saturate_uint(s.ffloor(bfx)))
    var by0 = s.umin(s.isub(bm_h, s.uint(1)), s.saturate_uint(s.ffloor(bfy)))
    var bx1 = s.umin(s.isub(bm_w, s.uint(1)), s.iadd(bx0, s.uint(1)))
    var by1 = s.umin(s.isub(bm_h, s.uint(1)), s.iadd(by0, s.uint(1)))
    var btx = s.nclamp(s.fsub(bfx, s.float_from_uint(bx0)), s.float(0.0), s.float(1.0))
    var bty = s.nclamp(s.fsub(bfy, s.float_from_uint(by0)), s.float(0.0), s.float(1.0))
    var bomx = s.fsub(s.float(1.0), btx); var bomy = s.fsub(s.float(1.0), bty)
    var bw00 = s.fmul(bomx, bomy); var bw10 = s.fmul(btx, bomy)
    var bw01 = s.fmul(bomx, bty); var bw11 = s.fmul(btx, bty)
    var bp00 = s.iadd(bm_off, s.imul(s.iadd(s.imul(by0, bm_w), bx0), s.uint(4)))
    var bp10 = s.iadd(bm_off, s.imul(s.iadd(s.imul(by0, bm_w), bx1), s.uint(4)))
    var bp01 = s.iadd(bm_off, s.imul(s.iadd(s.imul(by1, bm_w), bx0), s.uint(4)))
    var bp11 = s.iadd(bm_off, s.imul(s.iadd(s.imul(by1, bm_w), bx1), s.uint(4)))
    var bl_r = s.fma(s.load_float(b_post, bp00), bw00, s.fma(s.load_float(b_post, bp10), bw10, s.fma(s.load_float(b_post, bp01), bw01, s.fmul(s.load_float(b_post, bp11), bw11))))
    var bl_g = s.fma(s.load_float(b_post, s.iadd(bp00, s.uint(1))), bw00, s.fma(s.load_float(b_post, s.iadd(bp10, s.uint(1))), bw10, s.fma(s.load_float(b_post, s.iadd(bp01, s.uint(1))), bw01, s.fmul(s.load_float(b_post, s.iadd(bp11, s.uint(1))), bw11))))
    var bl_b = s.fma(s.load_float(b_post, s.iadd(bp00, s.uint(2))), bw00, s.fma(s.load_float(b_post, s.iadd(bp10, s.uint(2))), bw10, s.fma(s.load_float(b_post, s.iadd(bp01, s.uint(2))), bw01, s.fmul(s.load_float(b_post, s.iadd(bp11, s.uint(2))), bw11))))
    s.store(rf_r, s.fma(s.fsub(bl_r, s.load_f(rf_r)), s.float(0.55), s.load_f(rf_r)))
    s.store(rf_g, s.fma(s.fsub(bl_g, s.load_f(rf_g)), s.float(0.55), s.load_f(rf_g)))
    s.store(rf_b, s.fma(s.fsub(bl_b, s.load_f(rf_b)), s.float(0.55), s.load_f(rf_b)))
    s.end_if()

    var loc_lum = s.fma(base_r, s.float(0.2126), s.fma(base_g, s.float(0.7152), s.fmul(base_b, s.float(0.0722))))
    var d_sp = s.fmul(s.load_f(drop_top_hl), s.fma(s.nmin(s.float(4.0), loc_lum), s.float(0.30), s.float(0.030)))
    s.store(fin_r, s.fadd(s.nmax(s.load_f(rf_r), s.fmul(base_r, s.float(0.38))), d_sp))
    s.store(fin_g, s.fadd(s.nmax(s.load_f(rf_g), s.fmul(base_g, s.float(0.38))), d_sp))
    s.store(fin_b, s.fadd(s.nmax(s.load_f(rf_b), s.fmul(base_b, s.float(0.38))), s.fmul(d_sp, s.float(1.12))))
    s.end_if()
    s.end_if()

    var base_exp = s.load_float(b_globals, s.uint(47))
    var auto_exp_on = s.ine(s.load_word(b_globals, s.uint(119)), s.uint(0))
    var adapt_exp = s.load_float(b_post, s.uint(UInt32(AAA_POST_ADAPTED_EXP_WORD)))
    var eff_exp = s.select_f(auto_exp_on, s.fmul(base_exp, adapt_exp), base_exp)

    var vig_str = s.load_float(b_globals, s.uint(115))
    var tan_x_v = s.load_float(b_globals, s.uint(39))
    var tan_y_v = s.load_float(b_globals, s.uint(43))
    var nx_v = s.fmul(s.fma(uv_u, s.float(2.0), s.float(-1.0)), tan_x_v)
    var ny_v = s.fmul(s.fma(uv_v, s.float(2.0), s.float(-1.0)), tan_y_v)
    var cos2_th = s.fdiv(s.float(1.0), s.fadd(s.float(1.0), s.fma(nx_v, nx_v, s.fmul(ny_v, ny_v))))
    var cos4_th = s.fmul(cos2_th, cos2_th)
    var vig_mul = s.fma(s.fsub(cos4_th, s.float(1.0)), vig_str, s.float(1.0))
    var exposure = s.fmul(eff_exp, vig_mul)

    var exp_r = s.fmul(s.load_f(fin_r), exposure)
    var exp_g = s.fmul(s.load_f(fin_g), exposure)
    var exp_b = s.fmul(s.load_f(fin_b), exposure)

    var packed_px = s.uvar(s.uint(0xFF000000))
    s.begin_if(s.ieq(disp_mode, s.uint(AAA_DISPLAY_HDR10_PQ)))
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
    var graded = _emit_display_grade(s, s.load_f(dr), s.load_f(dg), s.load_f(db), grade_sat, grade_con)
    s.store(dr, graded[0]); s.store(dg, graded[1]); s.store(db, graded[2])
    var ur = s.umin(s.uint(255), s.saturate_uint(s.fma(s.load_f(dr), s.float(255.0), s.float(0.5))))
    var ug = s.umin(s.uint(255), s.saturate_uint(s.fma(s.load_f(dg), s.float(255.0), s.float(0.5))))
    var ub = s.umin(s.uint(255), s.saturate_uint(s.fma(s.load_f(db), s.float(255.0), s.float(0.5))))
    s.store(packed_px, s.bit_or(s.uint(0xFF000000), s.bit_or(s.shl(ur, s.uint(16)), s.bit_or(s.shl(ug, s.uint(8)), ub))))
    # --- sota:camera hook (HDR10 PQ / scRGB display mapping after AgX, Push[10..12]) ---
    sota_hdr_emit_tonemap_hdr(s, disp_mode, s.load_f(dr), s.load_f(dg), s.load_f(db), packed_px)
    # --- end sota:camera hook ---
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
