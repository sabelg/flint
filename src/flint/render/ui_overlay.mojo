"""Feature 15 (Flint v5): Tile-Binned GPU UI SDF Overlay Compute Pass (`ui_overlay.mojo`).

Features:
  - Exact `Flint 0xAARRGGBB` / `BGRA8` (`B=[0:7], G=[8:15], R=[16:23], A=[24:31]`) channel order
    with bit-exact SDR sRGB round-trip AND `A2B10G10R10` `HDR10 ST.2084 PQ` compositing at `203 nits`
    ITU-R BT.2408 UI paper-white (`disp_mode == 1`).
  - Chunked `256`-primitive outer tile loop in workgroup-uniform control flow: tiles with `>256`
    (up to `32,768+`) overlapping primitives never drop elements while preserving exact painter's order.
  - Full GPU analytical SDF evaluation across all `6` primitive kinds:
      `0` (`PRIM_SHAPE`): rounded-rect / circle SDF + border stroke + vertical gradient + soft drop shadow
      `1` (`PRIM_GLYPH`): bindless `R8_UNORM` SDF glyph + outer stroke outline + drop shadow
      `2` (`PRIM_NINE_SLICE`): piecewise 9-slice UV remapping sampling bindless UI atlas
      `3` (`PRIM_CONTROLLER_SYMBOL`): procedural GPU SDF controller button & icon glyphs
          (`A`, `B`, `X`, `Y`, `Cross`, `Circle`, `Triangle`, `Square`, `Play`, `Pause`, `Gizmo`)
      `4` (`PRIM_LINE`): analytical capsule segment & 12-chord quadratic Bézier curve SDF with AA feathering
      `5` (`PRIM_IMAGE`): textured rounded-rect quad sampling the bindless texture array (`tex_slot`)
"""
from std.collections import List
from std.math import max, min
from flint.ui.draw_list import (
    GpuUiPrim,
    PRIM_ARC,
    PRIM_CONTROLLER_SYMBOL,
    PRIM_GLYPH,
    PRIM_IMAGE,
    PRIM_LINE,
    PRIM_NINE_SLICE,
    PRIM_SHAPE,
    UI_FLAG_BORDER,
    UI_FLAG_CIRCLE,
    UI_FLAG_DASHED_FLOW,
    UI_FLAG_DROP_SHADOW,
    UI_FLAG_GLASS_SCRIM,
    UI_FLAG_HORIZ_GRADIENT,
    UI_FLAG_LINEAR_TEX,
    UI_FLAG_MSDF,
    UI_FLAG_MTSDF,
    UI_FLAG_OUTER_GLOW,
    UI_FLAG_PER_CORNER_RADII,
    UI_FLAG_QUAD_BEZIER,
    UI_FLAG_TOP_BEVEL,
    UI_FLAG_VERT_GRADIENT,
)
from flint.render.spirv import (
    AaaSpirvModule,
    IMAGE_FORMAT_UNKNOWN,
    MODEL_FRAGMENT,
    MODEL_GL_COMPUTE,
    OP_STORE,
    STORAGE_OUTPUT,
    emit,
)
from flint.render.aaa_visbuffer import (
    AAA_MAX_BINDLESS,
    AAA_PUSH_WORDS,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_HISTORY,
    BIND_AAA_POST_SCRATCH,
    BIND_AAA_SWAPCHAIN_IMG,
)

comptime AAA_POST_UI_PRIMS_BASE_OFFSET: Int = 1024
comptime TILE_PRIM_CAP: Int = 256  # prims composited per outer chunk (1 per invocation, zero drops across chunks)

comptime UI_FLAG_QUADRATIC_BEZIER: UInt32 = UI_FLAG_QUAD_BEZIER

# Procedural GPU SDF controller / HUD symbol IDs (`PRIM_CONTROLLER_SYMBOL`)
comptime UI_SYMBOL_A: Int = 0
comptime UI_SYMBOL_B: Int = 1
comptime UI_SYMBOL_X: Int = 2
comptime UI_SYMBOL_Y: Int = 3
comptime UI_SYMBOL_CROSS: Int = 4
comptime UI_SYMBOL_CIRCLE: Int = 5
comptime UI_SYMBOL_TRIANGLE: Int = 6
comptime UI_SYMBOL_SQUARE: Int = 7
comptime UI_SYMBOL_PLAY: Int = 8
comptime UI_SYMBOL_PAUSE: Int = 9
comptime UI_SYMBOL_GIZMO: Int = 10


def make_ui_line_prim(
    ax: Float32,
    ay: Float32,
    bx: Float32,
    by: Float32,
    thickness_px: Float32,
    rgba8: UInt32,
    feather_px: Float32 = 1.0,
    clip_x0: UInt16 = 0,
    clip_y0: UInt16 = 0,
    clip_x1: UInt16 = 0xFFFF,
    clip_y1: UInt16 = 0xFFFF,
) -> GpuUiPrim:
    """Builds a 64-byte `PRIM_LINE` capsule segment matching `UiDrawList.add_line` (`param0 = half_t`, `param2 = softness`)."""
    var half_t = max(Float32(0.5), thickness_px * Float32(0.5))
    var soft = max(Float32(0.5), feather_px)
    var pad = half_t + soft + Float32(1.0)
    var x0 = min(ax, bx) - pad
    var y0 = min(ay, by) - pad
    var x1 = max(ax, bx) + pad
    var y1 = max(ay, by) + pad
    var hdr: UInt32 = PRIM_LINE & 0xF
    return GpuUiPrim(
        x0, y0, x1, y1,
        ax, ay, bx, by,
        UInt32(clip_x0) | (UInt32(clip_y0) << 16),
        UInt32(clip_x1) | (UInt32(clip_y1) << 16),
        rgba8, 0,
        half_t, 0.0, soft, hdr,
    )


def make_ui_bezier_prim(
    ax: Float32,
    ay: Float32,
    cx: Float32,
    cy: Float32,
    bx: Float32,
    by: Float32,
    thickness_px: Float32,
    rgba8: UInt32,
    clip_x0: UInt16 = 0,
    clip_y0: UInt16 = 0,
    clip_x1: UInt16 = 0xFFFF,
    clip_y1: UInt16 = 0xFFFF,
) -> GpuUiPrim:
    """Builds a 64-byte `PRIM_LINE` quadratic Bézier curve matching `UiDrawList.add_quad_bezier` (`param0=cx, param1=cy, param2=half_t`)."""
    var half_t = max(Float32(0.5), thickness_px * Float32(0.5))
    var pad = half_t + Float32(2.0)
    var x0 = min(ax, min(cx, bx)) - pad
    var y0 = min(ay, min(cy, by)) - pad
    var x1 = max(ax, max(cx, bx)) + pad
    var y1 = max(ay, max(cy, by)) + pad
    var hdr: UInt32 = (PRIM_LINE & 0xF) | (UI_FLAG_QUADRATIC_BEZIER << 16)
    return GpuUiPrim(
        x0, y0, x1, y1,
        ax, ay, bx, by,
        UInt32(clip_x0) | (UInt32(clip_y0) << 16),
        UInt32(clip_x1) | (UInt32(clip_y1) << 16),
        rgba8, 0,
        cx, cy, half_t, hdr,
    )


def make_ui_symbol_prim(
    symbol_id: Int,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    fg_rgba8: UInt32,
    badge_rgba8: UInt32 = 0,
    stroke_scale: Float32 = 0.11,
    clip_x0: UInt16 = 0,
    clip_y0: UInt16 = 0,
    clip_x1: UInt16 = 0xFFFF,
    clip_y1: UInt16 = 0xFFFF,
) -> GpuUiPrim:
    """Builds a 64-byte `PRIM_CONTROLLER_SYMBOL` procedural GPU SDF icon."""
    var sym_u = UInt32(max(0, symbol_id) & 0xFFF)
    var hdr: UInt32 = (PRIM_CONTROLLER_SYMBOL & 0xF) | (sym_u << 4)
    return GpuUiPrim(
        x, y, x + w, y + h,
        0.0, 0.0, 1.0, 1.0,
        UInt32(clip_x0) | (UInt32(clip_y0) << 16),
        UInt32(clip_x1) | (UInt32(clip_y1) << 16),
        fg_rgba8, badge_rgba8,
        Float32(max(0, symbol_id)), stroke_scale, 1.0, hdr,
    )


def make_ui_image_prim(
    tex_slot: Int,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    u0: Float32 = 0.0,
    v0: Float32 = 0.0,
    u1: Float32 = 1.0,
    v1: Float32 = 1.0,
    tint_rgba8: UInt32 = 0xFFFFFFFF,
    corner_radius: Float32 = 0.0,
    border_width: Float32 = 0.0,
    border_rgba8: UInt32 = 0,
    is_linear_tex: Bool = False,
    clip_x0: UInt16 = 0,
    clip_y0: UInt16 = 0,
    clip_x1: UInt16 = 0xFFFF,
    clip_y1: UInt16 = 0xFFFF,
) -> GpuUiPrim:
    """Builds a 64-byte `PRIM_IMAGE` bindless textured rounded-rect primitive."""
    var flags: UInt32 = (UI_FLAG_BORDER if border_width > 0.0 else UInt32(0)) | (
        UI_FLAG_LINEAR_TEX if is_linear_tex else UInt32(0)
    )
    var hdr: UInt32 = (PRIM_IMAGE & 0xF) | (UInt32(max(0, tex_slot) & 0xFFF) << 4) | (flags << 16)
    return GpuUiPrim(
        x, y, x + w, y + h,
        u0, v0, u1, v1,
        UInt32(clip_x0) | (UInt32(clip_y0) << 16),
        UInt32(clip_x1) | (UInt32(clip_y1) << 16),
        tint_rgba8, border_rgba8,
        corner_radius, border_width, 1.0, hdr,
    )


@always_inline
def _emit_unpack_flint_argb(
    mut s: AaaSpirvModule, packed: UInt32
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """Unpacks Flint `0xAARRGGBB` (`BGRA8` in little-endian memory: `B=[0:7], G=[8:15], R=[16:23], A=[24:31]`)
    into normalized `(r, g, b, a)` floats in `[0.0, 1.0]`."""
    var inv255 = s.float(1.0 / 255.0)
    var b = s.fmul(s.float_from_uint(s.bit_and(packed, s.uint(0xFF))), inv255)
    var g = s.fmul(s.float_from_uint(s.bit_and(s.shr(packed, s.uint(8)), s.uint(0xFF))), inv255)
    var r = s.fmul(s.float_from_uint(s.bit_and(s.shr(packed, s.uint(16)), s.uint(0xFF))), inv255)
    var a = s.fmul(s.float_from_uint(s.bit_and(s.shr(packed, s.uint(24)), s.uint(0xFF))), inv255)
    return (r, g, b, a)


@always_inline
def _emit_srgb_eotf(mut s: AaaSpirvModule, c_srgb: UInt32) -> UInt32:
    """IEC 61966-2-1 sRGB -> linear EOTF."""
    var c = s.nclamp(c_srgb, s.float(0.0), s.float(1.0))
    var lo = s.fmul(c, s.float(1.0 / 12.92))
    var hi = s.fpow(s.nmax(s.float(1e-7), s.fmul(s.fadd(c, s.float(0.055)), s.float(1.0 / 1.055))), s.float(2.4))
    return s.select_f(s.flt(c, s.float(0.04045)), lo, hi)


@always_inline
def _emit_srgb_oetf(mut s: AaaSpirvModule, c_lin: UInt32) -> UInt32:
    """IEC 61966-2-1 linear -> sRGB OETF."""
    var c = s.nclamp(c_lin, s.float(0.0), s.float(1.0))
    var lo = s.fmul(c, s.float(12.92))
    var hi = s.fsub(s.fmul(s.float(1.055), s.fpow(s.nmax(s.float(1e-7), c), s.float(1.0 / 2.4))), s.float(0.055))
    return s.select_f(s.flt(c, s.float(0.0031308)), lo, hi)


@always_inline
def _emit_pq_eotf_100nits(mut s: AaaSpirvModule, pq_val: UInt32) -> UInt32:
    """SMPTE ST.2084 Inverse PQ EOTF mapping `[0..1]` -> linear luminance where `1.0 = 100 nits`."""
    var n = s.nclamp(pq_val, s.float(0.0), s.float(1.0))
    var np = s.fpow(s.nmax(s.float(1e-9), n), s.float(0.0126833135))
    var num = s.nmax(s.float(0.0), s.fsub(np, s.float(0.8359375)))
    var den = s.nmax(s.float(1e-6), s.fma(s.float(-18.6875), np, s.float(18.8515625)))
    var y_10k = s.fpow(s.nmax(s.float(1e-12), s.fdiv(num, den)), s.float(6.27739464))
    return s.fmul(y_10k, s.float(100.0))


@always_inline
def _emit_pq_oetf_100nits(mut s: AaaSpirvModule, lin_100nits: UInt32) -> UInt32:
    """SMPTE ST.2084 PQ OETF mapping `1.0 = 100 nits` (`2.03 = 203 nits` paper-white) -> `[0..1]`."""
    var y = s.nclamp(s.fmul(lin_100nits, s.float(100.0 / 10000.0)), s.float(0.0), s.float(1.0))
    var ym1 = s.fpow(s.nmax(s.float(1e-9), y), s.float(0.1593017578))
    var num = s.fma(s.float(18.8515625), ym1, s.float(0.8359375))
    var den = s.fma(s.float(18.6875), ym1, s.float(1.0))
    return s.nclamp(s.fpow(s.fdiv(num, den), s.float(78.84375)), s.float(0.0), s.float(1.0))


@always_inline
def _emit_ui_srgb_to_working(
    mut s: AaaSpirvModule,
    is_hdr10: UInt32,
    sr: UInt32,
    sg: UInt32,
    sb: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Converts `[0..1]` sRGB UI color into working composite space:
      - `disp_mode != 1`: stays in `[0..1]` sRGB (exact SDR UI blending & bit-exact `0xAARRGGBB` output)
      - `disp_mode == 1`: converts sRGB -> linear Rec.709 -> Rec.2020 @ `203 nits` paper-white (`2.03` in 100-nit units)."""
    var wr = s.fvar(sr)
    var wg = s.fvar(sg)
    var wb = s.fvar(sb)
    s.begin_if(is_hdr10)
    var lr = _emit_srgb_eotf(s, sr)
    var lg = _emit_srgb_eotf(s, sg)
    var lb = _emit_srgb_eotf(s, sb)
    var pw = s.float(2.03)  # ITU-R BT.2408 203 nits reference UI paper-white
    var r2020 = s.fmul(pw, s.fma(lr, s.float(0.627404), s.fma(lg, s.float(0.329283), s.fmul(lb, s.float(0.043313)))))
    var g2020 = s.fmul(pw, s.fma(lr, s.float(0.069097), s.fma(lg, s.float(0.919540), s.fmul(lb, s.float(0.011362)))))
    var b2020 = s.fmul(pw, s.fma(lr, s.float(0.016391), s.fma(lg, s.float(0.088013), s.fmul(lb, s.float(0.895595)))))
    s.store(wr, r2020)
    s.store(wg, g2020)
    s.store(wb, b2020)
    s.end_if()
    return (s.load_f(wr), s.load_f(wg), s.load_f(wb))


@always_inline
def _emit_blend_over(
    mut s: AaaSpirvModule,
    accum_r: UInt32,
    accum_g: UInt32,
    accum_b: UInt32,
    src_r: UInt32,
    src_g: UInt32,
    src_b: UInt32,
    src_a: UInt32,
):
    var eff_a = s.nclamp(src_a, s.float(0.0), s.float(1.0))
    var inv_a = s.fsub(s.float(1.0), eff_a)
    s.store(accum_r, s.fma(src_r, eff_a, s.fmul(s.load_f(accum_r), inv_a)))
    s.store(accum_g, s.fma(src_g, eff_a, s.fmul(s.load_f(accum_g), inv_a)))
    s.store(accum_b, s.fma(src_b, eff_a, s.fmul(s.load_f(accum_b), inv_a)))


@always_inline
def _emit_seg_dist(
    mut s: AaaSpirvModule,
    px: UInt32,
    py: UInt32,
    ax: UInt32,
    ay: UInt32,
    bx: UInt32,
    by: UInt32,
) -> UInt32:
    """Unsigned Euclidean distance from `(px, py)` to line segment `(ax, ay) -> (bx, by)`."""
    var pax = s.fsub(px, ax)
    var pay = s.fsub(py, ay)
    var bax = s.fsub(bx, ax)
    var bay = s.fsub(by, ay)
    var ba_dot = s.nmax(s.float(1e-6), s.fma(bax, bax, s.fmul(bay, bay)))
    var h = s.nclamp(s.fdiv(s.fma(pax, bax, s.fmul(pay, bay)), ba_dot), s.float(0.0), s.float(1.0))
    var dx = s.fsub(pax, s.fmul(bax, h))
    var dy = s.fsub(pay, s.fmul(bay, h))
    return s.fsqrt(s.fma(dx, dx, s.fmul(dy, dy)))


@always_inline
def _emit_msdf_median(mut s: AaaSpirvModule, r: UInt32, g: UInt32, b: UInt32) -> UInt32:
    """Chlumsky 3-channel MSDF median `max(min(r, g), min(max(r, g), b))`."""
    return s.nmax(s.nmin(r, g), s.nmin(s.nmax(r, g), b))


@always_inline
def _emit_rounded_box_sdf(
    mut s: AaaSpirvModule,
    px: UInt32,
    py: UInt32,
    cx: UInt32,
    cy: UInt32,
    w_half: UInt32,
    h_half: UInt32,
    radius: UInt32,
) -> UInt32:
    var r_lim = s.nclamp(radius, s.float(0.0), s.nmin(w_half, h_half))
    var qx = s.fsub(s.fabs(s.fsub(px, cx)), s.fsub(w_half, r_lim))
    var qy = s.fsub(s.fabs(s.fsub(py, cy)), s.fsub(h_half, r_lim))
    var ox = s.nmax(qx, s.float(0.0))
    var oy = s.nmax(qy, s.float(0.0))
    return s.fsub(
        s.fadd(s.fsqrt(s.fma(ox, ox, s.fmul(oy, oy))), s.nmin(s.nmax(qx, qy), s.float(0.0))),
        r_lim,
    )


@always_inline
def _emit_nine_slice_axis(
    mut s: AaaSpirvModule,
    local_p: UInt32,
    span: UInt32,
    b_lo_in: UInt32,
    b_hi_in: UInt32,
    uv_lo: UInt32,
    uv_hi: UInt32,
) -> UInt32:
    """Piecewise 3-zone 9-slice axis UV remapping (`[0..b_lo] -> [0..uv_lo]`, middle stretch, `[span-b_hi..span] -> [1-uv_hi..1]`)."""
    var b_sum = s.fadd(b_lo_in, b_hi_in)
    var shrink = s.select_f(s.fgt(b_sum, span), s.fdiv(span, s.nmax(s.float(1e-4), b_sum)), s.float(1.0))
    var b_lo = s.fmul(b_lo_in, shrink)
    var b_hi = s.fmul(b_hi_in, shrink)
    var hi_start = s.fsub(span, b_hi)

    var out_uv = s.fvar(s.float(0.5))
    s.begin_if(s.flt(local_p, b_lo))
    s.store(out_uv, s.fmul(s.fdiv(local_p, s.nmax(s.float(1e-4), b_lo)), uv_lo))
    s.begin_else()
    s.begin_if(s.fgt(local_p, hi_start))
    var t_hi = s.fdiv(s.fsub(local_p, hi_start), s.nmax(s.float(1e-4), b_hi))
    s.store(out_uv, s.fma(t_hi, uv_hi, s.fsub(s.float(1.0), uv_hi)))
    s.begin_else()
    var mid_span = s.nmax(s.float(1e-4), s.fsub(hi_start, b_lo))
    var t_mid = s.fdiv(s.fsub(local_p, b_lo), mid_span)
    var uv_mid = s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fadd(uv_lo, uv_hi)))
    s.store(out_uv, s.fma(t_mid, uv_mid, uv_lo))
    s.end_if()
    s.end_if()
    return s.nclamp(s.load_f(out_uv), s.float(0.0), s.float(1.0))


def aaa_ui_overlay_shader(
    direct_swapchain: Bool = False,
    swapchain_binding: Int = BIND_AAA_SWAPCHAIN_IMG,
) -> List[UInt32]:
    """Tile-Binned GPU UI SDF Overlay Compute Shader (16x16 workgroups, chunked 256-prim LDS loop).
    Push[0] = out_w, Push[1] = out_h, Push[2] = prim_count,
    Push[3] = prims_word_offset (in BIND_AAA_POST_SCRATCH),
    Push[4] = out_ldr_off (in BIND_AAA_HISTORY),
    Push[5] = display_mode (0=AgX BGRA8 sRGB, 1=HDR10 ST.2084 PQ A2B10G10R10 @ 203 nits, 2=Linear->sRGB BGRA8).
    """
    var s = AaaSpirvModule(
        MODEL_GL_COMPUTE,
        group_width=16,
        group_height=16,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
        enable_storage_image_without_format=direct_swapchain,
    )
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var swap_img = (UInt32(0), UInt32(0))
    if direct_swapchain:
        swap_img = s.storage_image_2d(swapchain_binding, format=IMAGE_FORMAT_UNKNOWN)
    _ = b_hist

    var gx = s.invocation_axis(0)
    var gy = s.invocation_axis(1)
    var out_w = s.push_word(0)
    var out_h = s.push_word(1)
    var prim_cnt = s.push_word(2)
    var prims_off = s.push_word(3)
    var out_ldr_off = s.push_word(4)
    var disp_mode = s.push_word(5)
    var ui_time_sec = s.push_float(6)
    var ui_global_flags = s.push_word(8)

    # Uniform early exit if draw list is empty
    s.return_if(s.ieq(prim_cnt, s.uint(0)))

    var is_hdr10 = s.ieq(disp_mode, s.uint(1))
    var glass_enabled = s.ieq(s.bit_and(ui_global_flags, s.uint(1)), s.uint(0))
    var in_screen = s.logical_and(s.ult(gx, out_w), s.ult(gy, out_h))

    var wx = s.workgroup_axis(0)
    var wy = s.workgroup_axis(1)
    var lx = s.local_axis(0)
    var ly = s.local_axis(1)
    var local_idx = s.iadd(s.imul(ly, s.uint(16)), lx)

    # 16x16 Tile bounds in screen pixels
    var tile_x0_u = s.imul(wx, s.uint(16))
    var tile_y0_u = s.imul(wy, s.uint(16))
    var tile_x0 = s.float_from_uint(tile_x0_u)
    var tile_y0 = s.float_from_uint(tile_y0_u)
    var tile_x1 = s.fadd(tile_x0, s.float(16.0))
    var tile_y1 = s.fadd(tile_y0, s.float(16.0))

    # Workgroup Shared Memory for chunked 256-prim binning, rank sort & race-free glass backdrop LDS
    var shared_cnt = s.workgroup_array_u32(1)
    var shared_prims = s.workgroup_array_u32(TILE_PRIM_CAP)
    var ordered_prims = s.workgroup_array_u32(TILE_PRIM_CAP)
    var bg_lds_r = s.workgroup_array_f32(256)
    var bg_lds_g = s.workgroup_array_f32(256)
    var bg_lds_b = s.workgroup_array_f32(256)

    var fgx = s.fadd(s.float_from_uint(gx), s.float(0.5))
    var fgy = s.fadd(s.float_from_uint(gy), s.float(0.5))
    var pix_idx = s.iadd(out_ldr_off, s.iadd(s.imul(gy, out_w), gx))

    var bg_loaded = s.uvar(s.uint(0))
    var accum_r = s.fvar(s.float(0.0))
    var accum_g = s.fvar(s.float(0.0))
    var accum_b = s.fvar(s.float(0.0))

    # Outer Chunked Loop: processes `prim_cnt` in 256-primitive draw-ordered slices (zero drops!)
    var chunk_base_var = s.uvar(s.uint(0))
    s.begin_loop()
    var chunk_base = s.load_u(chunk_base_var)
    s.break_if(s.uge(chunk_base, prim_cnt))

    s.begin_if(s.ieq(local_idx, s.uint(0)))
    s.store_workgroup_u32(shared_cnt[0], shared_cnt[1], s.uint(0), s.uint(0))
    s.end_if()
    s.workgroup_barrier()

    var cand_p = s.iadd(chunk_base, local_idx)
    s.begin_if(s.ult(cand_p, prim_cnt))
    var base_w = s.iadd(prims_off, s.imul(cand_p, s.uint(16)))
    var rx0 = s.load_float(b_post, base_w)
    var ry0 = s.load_float(b_post, s.iadd(base_w, s.uint(1)))
    var rx1 = s.load_float(b_post, s.iadd(base_w, s.uint(2)))
    var ry1 = s.load_float(b_post, s.iadd(base_w, s.uint(3)))
    var c_cxy0 = s.load_word(b_post, s.iadd(base_w, s.uint(8)))
    var c_cxy1 = s.load_word(b_post, s.iadd(base_w, s.uint(9)))
    var c_hdr = s.load_word(b_post, s.iadd(base_w, s.uint(15)))
    var c_kind = s.bit_and(c_hdr, s.uint(0xF))
    var c_flags = s.shr(c_hdr, s.uint(16))
    var c_p0 = s.load_float(b_post, s.iadd(base_w, s.uint(12)))
    var c_p1 = s.load_float(b_post, s.iadd(base_w, s.uint(13)))
    var c_p2 = s.load_float(b_post, s.iadd(base_w, s.uint(14)))

    # Compute conservative screen AABB accounting for lines, arcs, stroke outlines, shadows & outer glows
    var pad_v = s.fvar(s.float(1.5))
    s.begin_if(s.ieq(c_kind, s.uint(4)))
    s.store(pad_v, s.fadd(s.fmul(s.nmax(c_p0, s.float(1.0)), s.float(0.5)), s.float(2.5)))
    s.begin_if(s.ine(s.bit_and(c_flags, s.uint(UI_FLAG_DASHED_FLOW)), s.uint(0)))
    s.store(pad_v, s.fadd(s.load_f(pad_v), s.nmax(s.float(4.0), s.fmul(s.nmax(c_p0, c_p2), s.float(3.0)))))
    s.end_if()
    s.end_if()
    s.begin_if(s.ieq(c_kind, s.uint(PRIM_ARC)))
    s.store(pad_v, s.fadd(s.fmul(s.nmax(c_p1, s.float(1.0)), s.float(0.5)), s.fadd(s.nclamp(c_p2, s.float(0.0), s.float(28.0)), s.float(2.5))))
    s.end_if()
    var has_sh = s.logical_or(
        s.ine(s.bit_and(c_flags, s.uint(1)), s.uint(0)),
        s.logical_and(s.ieq(c_kind, s.uint(1)), s.fgt(c_p2, s.float(0.01))),
    )
    s.begin_if(has_sh)
    s.store(pad_v, s.fadd(s.load_f(pad_v), s.fadd(s.nmax(c_p2, s.float(6.0)), s.float(4.0))))
    s.end_if()
    s.begin_if(s.ine(s.bit_and(c_flags, s.uint(UI_FLAG_OUTER_GLOW)), s.uint(0)))
    s.store(pad_v, s.fadd(s.load_f(pad_v), s.fadd(s.nclamp(c_p2, s.float(2.0), s.float(28.0)), s.float(2.0))))
    s.end_if()
    s.begin_if(s.fgt(c_p1, s.float(0.01)))
    s.store(pad_v, s.fadd(s.load_f(pad_v), s.nmin(c_p1, s.float(16.0))))
    s.end_if()

    var eff_pad = s.load_f(pad_v)
    var ccx0_f = s.float_from_uint(s.bit_and(c_cxy0, s.uint(0xFFFF)))
    var ccy0_f = s.float_from_uint(s.shr(c_cxy0, s.uint(16)))
    var ccx1_f = s.float_from_uint(s.bit_and(c_cxy1, s.uint(0xFFFF)))
    var ccy1_f = s.float_from_uint(s.shr(c_cxy1, s.uint(16)))
    var ax0 = s.nmax(s.fsub(s.nmin(rx0, rx1), eff_pad), ccx0_f)
    var ay0 = s.nmax(s.fsub(s.nmin(ry0, ry1), eff_pad), ccy0_f)
    var ax1 = s.nmin(s.fadd(s.nmax(rx0, rx1), eff_pad), ccx1_f)
    var ay1 = s.nmin(s.fadd(s.nmax(ry0, ry1), eff_pad), ccy1_f)

    var outside = s.logical_or(
        s.logical_or(
            s.logical_or(s.fge(tile_x0, ax1), s.fge(ax0, tile_x1)),
            s.logical_or(s.fge(tile_y0, ay1), s.fge(ay0, tile_y1)),
        ),
        s.logical_or(
            s.logical_or(s.fge(ax0, ax1), s.fge(ay0, ay1)),
            s.ieq(c_kind, s.uint(7)),  # PRIM_POLY_DATA: vertex storage of the polygon before it
        ),
    )
    s.begin_if(s.logical_not(outside))
    var slot = s.atomic_add_workgroup_u32(shared_cnt[0], shared_cnt[1], s.uint(0), s.uint(1))
    s.store_workgroup_u32(shared_prims[0], shared_prims[1], slot, cand_p)
    s.end_if()
    s.end_if()
    s.workgroup_barrier()

    var chunk_hits = s.load_workgroup_u32(shared_cnt[0], shared_cnt[1], s.uint(0))
    s.begin_if(s.ine(chunk_hits, s.uint(0)))

    # Lazy background fetch on first chunk touching this tile (avoids touching untouched 3D scene tiles!)
    s.begin_if(s.logical_and(in_screen, s.ieq(s.load_u(bg_loaded), s.uint(0))))
    if direct_swapchain:
        var bg_v4 = s.image_read(swap_img, s.uvec2(gx, gy))
        var bg_x = s.extract_f(bg_v4, 0)
        var bg_y = s.extract_f(bg_v4, 1)
        var bg_z = s.extract_f(bg_v4, 2)
        s.begin_if(is_hdr10)
        s.store(accum_r, _emit_pq_eotf_100nits(s, bg_x))
        s.store(accum_g, _emit_pq_eotf_100nits(s, bg_y))
        s.store(accum_b, _emit_pq_eotf_100nits(s, bg_z))
        s.begin_else()
        s.store(accum_r, bg_x)
        s.store(accum_g, bg_y)
        s.store(accum_b, bg_z)
        s.end_if()
    else:
        var bg_raw = s.load_word(b_hist, pix_idx)
        s.begin_if(is_hdr10)
        var inv1023 = s.float(1.0 / 1023.0)
        var pq_r = s.fmul(s.float_from_uint(s.bit_and(bg_raw, s.uint(0x3FF))), inv1023)
        var pq_g = s.fmul(s.float_from_uint(s.bit_and(s.shr(bg_raw, s.uint(10)), s.uint(0x3FF))), inv1023)
        var pq_b = s.fmul(s.float_from_uint(s.bit_and(s.shr(bg_raw, s.uint(20)), s.uint(0x3FF))), inv1023)
        s.store(accum_r, _emit_pq_eotf_100nits(s, pq_r))
        s.store(accum_g, _emit_pq_eotf_100nits(s, pq_g))
        s.store(accum_b, _emit_pq_eotf_100nits(s, pq_b))
        s.begin_else()
        var bg_rgba = _emit_unpack_flint_argb(s, bg_raw)
        s.store(accum_r, bg_rgba[0])
        s.store(accum_g, bg_rgba[1])
        s.store(accum_b, bg_rgba[2])
        s.end_if()
    s.store_workgroup_f32(bg_lds_r[0], bg_lds_r[1], local_idx, s.load_f(accum_r))
    s.store_workgroup_f32(bg_lds_g[0], bg_lds_g[1], local_idx, s.load_f(accum_g))
    s.store_workgroup_f32(bg_lds_b[0], bg_lds_b[1], local_idx, s.load_f(accum_b))
    s.store(bg_loaded, s.uint(1))
    s.end_if()

    # Parallel rank-sort `shared_prims[0..chunk_hits-1]` back into exact painter's draw order
    s.begin_if(s.ult(local_idx, chunk_hits))
    var mine = s.load_workgroup_u32(shared_prims[0], shared_prims[1], local_idx)
    var rank_var = s.uvar(s.uint(0))
    var j_var = s.uvar(s.uint(0))
    s.begin_loop()
    var j = s.load_u(j_var)
    s.break_if(s.uge(j, chunk_hits))
    var other = s.load_workgroup_u32(shared_prims[0], shared_prims[1], j)
    s.begin_if(s.ult(other, mine))
    s.store(rank_var, s.iadd(s.load_u(rank_var), s.uint(1)))
    s.end_if()
    s.store(j_var, s.iadd(j, s.uint(1)))
    s.end_loop()
    s.store_workgroup_u32(ordered_prims[0], ordered_prims[1], s.load_u(rank_var), mine)
    s.end_if()
    s.workgroup_barrier()

    s.begin_if(in_screen)
    var i_var = s.uvar(s.uint(0))
    s.begin_loop()
    var i = s.load_u(i_var)
    s.break_if(s.uge(i, chunk_hits))

    var prim_idx = s.load_workgroup_u32(ordered_prims[0], ordered_prims[1], i)
    var pb = s.iadd(prims_off, s.imul(prim_idx, s.uint(16)))

    # Scissor clip rect check
    var cxy0 = s.load_word(b_post, s.iadd(pb, s.uint(8)))
    var cxy1 = s.load_word(b_post, s.iadd(pb, s.uint(9)))
    var cx0 = s.bit_and(cxy0, s.uint(0xFFFF))
    var cy0 = s.shr(cxy0, s.uint(16))
    var cx1 = s.bit_and(cxy1, s.uint(0xFFFF))
    var cy1 = s.shr(cxy1, s.uint(16))

    var in_clip = s.logical_and(
        s.logical_and(s.uge(gx, cx0), s.ult(gx, cx1)),
        s.logical_and(s.uge(gy, cy0), s.ult(gy, cy1)),
    )
    s.begin_if(in_clip)

    var x0 = s.load_float(b_post, pb)
    var y0 = s.load_float(b_post, s.iadd(pb, s.uint(1)))
    var x1 = s.load_float(b_post, s.iadd(pb, s.uint(2)))
    var y1 = s.load_float(b_post, s.iadd(pb, s.uint(3)))
    var hdr = s.load_word(b_post, s.iadd(pb, s.uint(15)))
    var kind = s.bit_and(hdr, s.uint(0xF))
    var tex_slot = s.bit_and(s.shr(hdr, s.uint(4)), s.uint(0xFFF))
    var flags = s.shr(hdr, s.uint(16))
    var param0 = s.load_float(b_post, s.iadd(pb, s.uint(12)))
    var param1 = s.load_float(b_post, s.iadd(pb, s.uint(13)))
    var param2 = s.load_float(b_post, s.iadd(pb, s.uint(14)))

    # Expanded pixel AABB check (supports shadow / glow / arc / stroke / thick-line / flow boxes)
    var box_pad = s.fvar(s.float(0.5))
    s.begin_if(s.ieq(kind, s.uint(4)))
    s.store(box_pad, s.fadd(s.fmul(s.nmax(param0, s.float(1.0)), s.float(0.5)), s.float(2.5)))
    s.begin_if(s.ine(s.bit_and(flags, s.uint(UI_FLAG_DASHED_FLOW)), s.uint(0)))
    s.store(box_pad, s.fadd(s.load_f(box_pad), s.nmax(s.float(4.0), s.fmul(s.nmax(param0, param2), s.float(3.0)))))
    s.end_if()
    s.end_if()
    s.begin_if(s.ieq(kind, s.uint(PRIM_ARC)))
    s.store(box_pad, s.fadd(s.fmul(s.nmax(param1, s.float(1.0)), s.float(0.5)), s.fadd(s.nclamp(param2, s.float(0.0), s.float(28.0)), s.float(2.5))))
    s.end_if()
    var has_drop_shadow = s.ine(s.bit_and(flags, s.uint(1)), s.uint(0))
    var has_outer_glow = s.ine(s.bit_and(flags, s.uint(UI_FLAG_OUTER_GLOW)), s.uint(0))
    s.begin_if(s.logical_or(has_drop_shadow, s.logical_and(s.ieq(kind, s.uint(1)), s.fgt(param2, s.float(0.01)))))
    s.store(box_pad, s.fadd(s.load_f(box_pad), s.fadd(s.nmax(param2, s.float(6.0)), s.float(4.0))))
    s.end_if()
    s.begin_if(has_outer_glow)
    s.store(box_pad, s.fadd(s.load_f(box_pad), s.fadd(s.nclamp(param2, s.float(2.0), s.float(28.0)), s.float(2.0))))
    s.end_if()
    s.begin_if(s.fgt(param1, s.float(0.01)))
    s.store(box_pad, s.fadd(s.load_f(box_pad), s.nmin(param1, s.float(16.0))))
    s.end_if()

    var bp_val = s.load_f(box_pad)
    var bx0 = s.fsub(s.nmin(x0, x1), bp_val)
    var by0 = s.fsub(s.nmin(y0, y1), bp_val)
    var bx1 = s.fadd(s.nmax(x0, x1), bp_val)
    var by1 = s.fadd(s.nmax(y0, y1), bp_val)

    var in_box = s.logical_and(
        s.logical_and(s.fge(fgx, bx0), s.fge(bx1, fgx)),
        s.logical_and(s.fge(fgy, by0), s.fge(by1, fgy)),
    )
    s.begin_if(in_box)

    var u0 = s.load_float(b_post, s.iadd(pb, s.uint(4)))
    var v0 = s.load_float(b_post, s.iadd(pb, s.uint(5)))
    var u1 = s.load_float(b_post, s.iadd(pb, s.uint(6)))
    var v1 = s.load_float(b_post, s.iadd(pb, s.uint(7)))

    var fill_raw = s.load_word(b_post, s.iadd(pb, s.uint(10)))
    var sec_raw = s.load_word(b_post, s.iadd(pb, s.uint(11)))
    var fill_rgba = _emit_unpack_flint_argb(s, fill_raw)
    var sec_rgba = _emit_unpack_flint_argb(s, sec_raw)
    var cr = fill_rgba[0]
    var cg = fill_rgba[1]
    var cb = fill_rgba[2]
    var ca = fill_rgba[3]
    var sr = sec_rgba[0]
    var sg = sec_rgba[1]
    var sb = sec_rgba[2]
    var sa = sec_rgba[3]

    # ------------------------------------------------------------------------
    # Kind 0 (`PRIM_SHAPE`): Rounded-Rect (uniform or 4-corner) / Circle SDF
    #   + Outer Drop Shadow + Outer Glow + LDS Glass Scrim + V/H/Diag Gradient
    #   + Border Stroke + 1px Top Specular Bevel
    # ------------------------------------------------------------------------
    s.begin_if(s.ieq(kind, s.uint(0)))
    var w_half = s.fmul(s.nmax(s.float(0.5), s.fsub(x1, x0)), s.float(0.5))
    var h_half = s.fmul(s.nmax(s.float(0.5), s.fsub(y1, y0)), s.float(0.5))
    var cx = s.fadd(x0, w_half)
    var cy = s.fadd(y0, h_half)
    var is_circle = s.ine(s.bit_and(flags, s.uint(4)), s.uint(0))
    var has_border = s.logical_and(s.ine(s.bit_and(flags, s.uint(2)), s.uint(0)), s.fgt(param1, s.float(0.01)))
    var has_vgrad = s.ine(s.bit_and(flags, s.uint(8)), s.uint(0))
    var has_hgrad = s.ine(s.bit_and(flags, s.uint(UI_FLAG_HORIZ_GRADIENT)), s.uint(0))
    var has_per_corner = s.ine(s.bit_and(flags, s.uint(UI_FLAG_PER_CORNER_RADII)), s.uint(0))
    var has_top_bevel = s.ine(s.bit_and(flags, s.uint(UI_FLAG_TOP_BEVEL)), s.uint(0))
    var has_glass_scrim = s.ine(s.bit_and(flags, s.uint(UI_FLAG_GLASS_SCRIM)), s.uint(0))

    # Resolve corner radius (uniform `param0` or IQ per-corner `(u0=TL, v0=TR, u1=BR, v1=BL)`)
    var r_top_sel = s.select_f(s.flt(fgx, cx), u0, v0)
    var r_bot_sel = s.select_f(s.flt(fgx, cx), v1, u1)
    var eff_r = s.select_f(has_per_corner, s.select_f(s.flt(fgy, cy), r_top_sel, r_bot_sel), param0)

    var dist_s = s.fvar(s.float(0.0))
    s.begin_if(is_circle)
    var dx = s.fsub(fgx, cx)
    var dy = s.fsub(fgy, cy)
    s.store(dist_s, s.fsub(s.fsqrt(s.fma(dx, dx, s.fmul(dy, dy))), param0))
    s.begin_else()
    s.store(dist_s, _emit_rounded_box_sdf(s, fgx, fgy, cx, cy, w_half, h_half, eff_r))
    s.end_if()

    var d_shape = s.load_f(dist_s)
    var cov_outer = s.nclamp(s.fsub(s.float(0.5), d_shape), s.float(0.0), s.float(1.0))

    # Optional soft drop shadow (`UI_FLAG_DROP_SHADOW`, outer-only for translucent fills)
    s.begin_if(has_drop_shadow)
    var sh_x = s.fsub(fgx, s.float(2.0))
    var sh_y = s.fsub(fgy, s.float(3.0))
    var dist_sh = s.fvar(s.float(0.0))
    s.begin_if(is_circle)
    var sdx = s.fsub(sh_x, cx)
    var sdy = s.fsub(sh_y, cy)
    s.store(dist_sh, s.fsub(s.fsqrt(s.fma(sdx, sdx, s.fmul(sdy, sdy))), param0))
    s.begin_else()
    var sh_r_top = s.select_f(s.flt(sh_x, cx), u0, v0)
    var sh_r_bot = s.select_f(s.flt(sh_x, cx), v1, u1)
    var sh_eff_r = s.select_f(has_per_corner, s.select_f(s.flt(sh_y, cy), sh_r_top, sh_r_bot), param0)
    s.store(dist_sh, _emit_rounded_box_sdf(s, sh_x, sh_y, cx, cy, w_half, h_half, sh_eff_r))
    s.end_if()
    var sh_soft = s.nmax(param2, s.float(5.0))
    var sh_t = s.nclamp(s.fsub(s.float(1.0), s.fdiv(s.nmax(s.load_f(dist_sh), s.float(0.0)), sh_soft)), s.float(0.0), s.float(1.0))
    var sh_outer_mask = s.select_f(s.flt(ca, s.float(0.995)), s.fsub(s.float(1.0), cov_outer), s.float(1.0))
    var sh_a = s.fmul(s.fmul(s.fmul(sh_t, sh_t), s.fmul(ca, s.float(0.45))), sh_outer_mask)
    _emit_blend_over(s, accum_r, accum_g, accum_b, s.float(0.0), s.float(0.0), s.float(0.0), sh_a)
    s.end_if()

    # Optional zero-offset analytical outer glow (`UI_FLAG_OUTER_GLOW`)
    s.begin_if(has_outer_glow)
    var g_soft = s.nclamp(param2, s.float(1.5), s.float(28.0))
    var g_u = s.fdiv(s.nmax(d_shape, s.float(0.0)), g_soft)
    var g_fall = s.fmul(s.exp2(s.fmul(s.float(-2.4), s.fmul(g_u, g_u))), s.fsub(s.float(1.0), cov_outer))
    var use_sec_glow = s.fgt(sa, s.float(0.001))
    var gr_c = s.select_f(use_sec_glow, sr, cr)
    var gg_c = s.select_f(use_sec_glow, sg, cg)
    var gb_c = s.select_f(use_sec_glow, sb, cb)
    var ga_c = s.select_f(use_sec_glow, sa, ca)
    var wg_col = _emit_ui_srgb_to_working(s, is_hdr10, gr_c, gg_c, gb_c)
    _emit_blend_over(s, accum_r, accum_g, accum_b, wg_col[0], wg_col[1], wg_col[2], s.fmul(ga_c, g_fall))
    s.end_if()

    # Optional 5-tap diamond Kawase frosted-glass scrim blur from race-free workgroup LDS (`UI_FLAG_GLASS_SCRIM`)
    s.begin_if(s.logical_and(s.logical_and(has_glass_scrim, glass_enabled), s.fgt(cov_outer, s.float(0.001))))
    var r_blur = s.umin(s.uint(4), s.umax(s.uint(1), s.saturate_uint(s.fadd(param2, s.float(0.5)))))
    var max_lx = s.umin(s.uint(15), s.isub(s.isub(out_w, s.uint(1)), tile_x0_u))
    var max_ly = s.umin(s.uint(15), s.isub(s.isub(out_h, s.uint(1)), tile_y0_u))
    var lx_lo = s.select_u(s.uge(lx, r_blur), s.isub(lx, r_blur), s.uint(0))
    var lx_hi = s.umin(max_lx, s.iadd(lx, r_blur))
    var ly_lo = s.select_u(s.uge(ly, r_blur), s.isub(ly, r_blur), s.uint(0))
    var ly_hi = s.umin(max_ly, s.iadd(ly, r_blur))
    var idx0 = local_idx
    var idx1 = s.iadd(s.imul(ly_lo, s.uint(16)), lx_lo)
    var idx2 = s.iadd(s.imul(ly_lo, s.uint(16)), lx_hi)
    var idx3 = s.iadd(s.imul(ly_hi, s.uint(16)), lx_lo)
    var idx4 = s.iadd(s.imul(ly_hi, s.uint(16)), lx_hi)
    var br0 = s.load_workgroup_f32(bg_lds_r[0], bg_lds_r[1], idx0)
    var bg0 = s.load_workgroup_f32(bg_lds_g[0], bg_lds_g[1], idx0)
    var bb0 = s.load_workgroup_f32(bg_lds_b[0], bg_lds_b[1], idx0)
    var br_d = s.fadd(
        s.fadd(s.load_workgroup_f32(bg_lds_r[0], bg_lds_r[1], idx1), s.load_workgroup_f32(bg_lds_r[0], bg_lds_r[1], idx2)),
        s.fadd(s.load_workgroup_f32(bg_lds_r[0], bg_lds_r[1], idx3), s.load_workgroup_f32(bg_lds_r[0], bg_lds_r[1], idx4)),
    )
    var bg_d = s.fadd(
        s.fadd(s.load_workgroup_f32(bg_lds_g[0], bg_lds_g[1], idx1), s.load_workgroup_f32(bg_lds_g[0], bg_lds_g[1], idx2)),
        s.fadd(s.load_workgroup_f32(bg_lds_g[0], bg_lds_g[1], idx3), s.load_workgroup_f32(bg_lds_g[0], bg_lds_g[1], idx4)),
    )
    var bb_d = s.fadd(
        s.fadd(s.load_workgroup_f32(bg_lds_b[0], bg_lds_b[1], idx1), s.load_workgroup_f32(bg_lds_b[0], bg_lds_b[1], idx2)),
        s.fadd(s.load_workgroup_f32(bg_lds_b[0], bg_lds_b[1], idx3), s.load_workgroup_f32(bg_lds_b[0], bg_lds_b[1], idx4)),
    )
    var blur_r = s.fma(br0, s.float(0.28), s.fmul(br_d, s.float(0.18)))
    var blur_g = s.fma(bg0, s.float(0.28), s.fmul(bg_d, s.float(0.18)))
    var blur_b = s.fma(bb0, s.float(0.28), s.fmul(bb_d, s.float(0.18)))
    _emit_blend_over(s, accum_r, accum_g, accum_b, blur_r, blur_g, blur_b, cov_outer)
    s.end_if()

    var fill_r_v = s.fvar(cr)
    var fill_g_v = s.fvar(cg)
    var fill_b_v = s.fvar(cb)
    var fill_a_v = s.fvar(ca)
    s.begin_if(s.logical_or(has_vgrad, has_hgrad))
    var ty = s.nclamp(s.fdiv(s.fsub(fgy, y0), s.nmax(s.float(1.0), s.fsub(y1, y0))), s.float(0.0), s.float(1.0))
    var tx = s.nclamp(s.fdiv(s.fsub(fgx, x0), s.nmax(s.float(1.0), s.fsub(x1, x0))), s.float(0.0), s.float(1.0))
    var t_grad = s.select_f(
        s.logical_and(has_vgrad, has_hgrad),
        s.fmul(s.fadd(tx, ty), s.float(0.5)),
        s.select_f(has_hgrad, tx, ty),
    )
    s.begin_if(s.logical_and(s.logical_not(has_border), s.fgt(sa, s.float(0.001))))
    s.store(fill_r_v, s.fma(s.fsub(sr, cr), t_grad, cr))
    s.store(fill_g_v, s.fma(s.fsub(sg, cg), t_grad, cg))
    s.store(fill_b_v, s.fma(s.fsub(sb, cb), t_grad, cb))
    s.store(fill_a_v, s.fma(s.fsub(sa, ca), t_grad, ca))
    s.begin_else()
    var shade_mul = s.fma(s.float(-0.32), t_grad, s.float(1.12))
    s.store(fill_r_v, s.nclamp(s.fmul(cr, shade_mul), s.float(0.0), s.float(1.0)))
    s.store(fill_g_v, s.nclamp(s.fmul(cg, shade_mul), s.float(0.0), s.float(1.0)))
    s.store(fill_b_v, s.nclamp(s.fmul(cb, shade_mul), s.float(0.0), s.float(1.0)))
    s.end_if()
    s.end_if()

    s.begin_if(has_border)
    var cov_inner = s.nclamp(s.fsub(s.float(0.5), s.fadd(d_shape, param1)), s.float(0.0), s.float(1.0))
    var wf = _emit_ui_srgb_to_working(s, is_hdr10, s.load_f(fill_r_v), s.load_f(fill_g_v), s.load_f(fill_b_v))
    _emit_blend_over(s, accum_r, accum_g, accum_b, wf[0], wf[1], wf[2], s.fmul(s.load_f(fill_a_v), cov_inner))
    var brd_cov = s.nmax(s.float(0.0), s.fsub(cov_outer, cov_inner))
    var wb = _emit_ui_srgb_to_working(s, is_hdr10, sr, sg, sb)
    _emit_blend_over(s, accum_r, accum_g, accum_b, wb[0], wb[1], wb[2], s.fmul(sa, brd_cov))
    s.begin_else()
    var wf2 = _emit_ui_srgb_to_working(s, is_hdr10, s.load_f(fill_r_v), s.load_f(fill_g_v), s.load_f(fill_b_v))
    _emit_blend_over(s, accum_r, accum_g, accum_b, wf2[0], wf2[1], wf2[2], s.fmul(s.load_f(fill_a_v), cov_outer))
    s.end_if()

    # Optional subtle 1px inner top specular highlight bevel (`UI_FLAG_TOP_BEVEL`)
    s.begin_if(has_top_bevel)
    var bev_inset = s.fadd(s.select_f(has_border, param1, s.float(0.0)), s.float(0.5))
    var inner_band = s.nclamp(s.fsub(s.float(0.5), s.fabs(s.fadd(d_shape, bev_inset))), s.float(0.0), s.float(1.0))
    var top_mask = s.nclamp(s.fsub(s.float(1.0), s.fdiv(s.fsub(fgy, y0), s.nmax(s.float(3.0), s.fadd(eff_r, s.float(2.0))))), s.float(0.0), s.float(1.0))
    var bev_a = s.fmul(s.float(0.09), s.fmul(inner_band, s.fmul(top_mask, s.fmul(cov_outer, ca))))
    var wbev = _emit_ui_srgb_to_working(s, is_hdr10, s.float(1.0), s.float(1.0), s.float(1.0))
    _emit_blend_over(s, accum_r, accum_g, accum_b, wbev[0], wbev[1], wbev[2], bev_a)
    s.end_if()
    s.end_if()

    # ------------------------------------------------------------------------
    # Kind 1 (`PRIM_GLYPH`): SDF / MSDF / MTSDF Font Glyph + Outer Stroke + Drop Shadow
    # ------------------------------------------------------------------------
    s.begin_if(s.ieq(kind, s.uint(1)))
    var gw = s.nmax(s.float(1.0), s.fsub(x1, x0))
    var gh = s.nmax(s.float(1.0), s.fsub(y1, y0))
    var du = s.fsub(u1, u0)
    var dv = s.fsub(v1, v0)
    var tu = s.fadd(u0, s.fmul(s.fdiv(s.fsub(fgx, x0), gw), du))
    var tv = s.fadd(v0, s.fmul(s.fdiv(s.fsub(fgy, y0), gh), dv))
    var is_msdf = s.ine(s.bit_and(flags, s.uint(UI_FLAG_MSDF)), s.uint(0))
    var is_mtsdf = s.ine(s.bit_and(flags, s.uint(UI_FLAG_MTSDF)), s.uint(0))

    var has_glyph_sh = s.logical_or(has_drop_shadow, s.fgt(param2, s.float(0.01)))
    s.begin_if(has_glyph_sh)
    var sh_off_px = s.nmax(param2, s.float(1.5))
    var stu = s.nclamp(s.fsub(tu, s.fmul(s.fdiv(sh_off_px, gw), du)), s.float(0.0), s.float(1.0))
    var stv = s.nclamp(s.fsub(tv, s.fmul(s.fdiv(sh_off_px, gh), dv)), s.float(0.0), s.float(1.0))
    var sh_samp = s.sample_bindless_lod(bindless_arr, tex_slot, s.vec2(stu, stv), s.float(0.0))
    var sh_r = s.extract_f(sh_samp, 0)
    var sh_med = _emit_msdf_median(s, sh_r, s.extract_f(sh_samp, 1), s.extract_f(sh_samp, 2))
    var sh_d_raw = s.select_f(is_mtsdf, s.extract_f(sh_samp, 3), s.select_f(is_msdf, sh_med, sh_r))
    var sh_sd = s.fmul(s.fsub(sh_d_raw, s.float(0.5)), param0)
    var sh_cov = s.nclamp(s.fadd(sh_sd, s.float(0.5)), s.float(0.0), s.float(1.0))
    _emit_blend_over(s, accum_r, accum_g, accum_b, s.float(0.0), s.float(0.0), s.float(0.0), s.fmul(ca, s.fmul(sh_cov, s.float(0.60))))
    s.end_if()

    var in_glyph_uv = s.logical_and(
        s.logical_and(s.fge(fgx, x0), s.fge(x1, fgx)),
        s.logical_and(s.fge(fgy, y0), s.fge(y1, fgy)),
    )
    s.begin_if(in_glyph_uv)
    var samp = s.sample_bindless_lod(bindless_arr, tex_slot, s.vec2(tu, tv), s.float(0.0))
    var g_r = s.extract_f(samp, 0)
    var g_med = _emit_msdf_median(s, g_r, s.extract_f(samp, 1), s.extract_f(samp, 2))
    var d_val = s.select_f(is_msdf, g_med, g_r)
    var sd = s.fmul(s.fsub(d_val, s.float(0.5)), param0)
    var cov = s.nclamp(s.fadd(sd, s.float(0.5)), s.float(0.0), s.float(1.0))
    s.begin_if(s.logical_and(s.fgt(param1, s.float(0.01)), s.fgt(sa, s.float(0.001))))
    var sd_stroke_base = s.select_f(is_mtsdf, s.fmul(s.fsub(s.extract_f(samp, 3), s.float(0.5)), param0), sd)
    var cov_out = s.nclamp(s.fadd(s.fadd(sd_stroke_base, param1), s.float(0.5)), s.float(0.0), s.float(1.0))
    var stroke_w = s.nmax(s.float(0.0), s.fsub(cov_out, cov))
    var wo = _emit_ui_srgb_to_working(s, is_hdr10, sr, sg, sb)
    _emit_blend_over(s, accum_r, accum_g, accum_b, wo[0], wo[1], wo[2], s.fmul(sa, stroke_w))
    s.end_if()
    var wg = _emit_ui_srgb_to_working(s, is_hdr10, cr, cg, cb)
    _emit_blend_over(s, accum_r, accum_g, accum_b, wg[0], wg[1], wg[2], s.fmul(ca, cov))
    s.end_if()
    s.end_if()

    # ------------------------------------------------------------------------
    # Kind 2 (`PRIM_NINE_SLICE`): Piecewise 9-Slice UV Remapping
    # ------------------------------------------------------------------------
    s.begin_if(s.ieq(kind, s.uint(2)))
    var nw = s.nmax(s.float(1.0), s.fsub(x1, x0))
    var nh = s.nmax(s.float(1.0), s.fsub(y1, y0))
    var nlx = s.nclamp(s.fsub(fgx, x0), s.float(0.0), nw)
    var nly = s.nclamp(s.fsub(fgy, y0), s.float(0.0), nh)

    var max_uv_b = s.nmax(s.nmax(u0, v0), s.nmax(u1, v1))
    var is_uv_frac = s.flt(max_uv_b, s.float(0.5001))
    var sc_ns = s.nmax(param2, s.float(0.25))
    var def_px = s.nmax(param0, s.float(16.0))
    var bl_px = s.select_f(is_uv_frac, s.fmul(u0, def_px), s.fmul(u0, sc_ns))
    var bt_px = s.select_f(is_uv_frac, s.fmul(v0, def_px), s.fmul(v0, sc_ns))
    var br_px = s.select_f(is_uv_frac, s.fmul(u1, def_px), s.fmul(u1, sc_ns))
    var bb_px = s.select_f(is_uv_frac, s.fmul(v1, def_px), s.fmul(v1, sc_ns))

    var has_custom_uv = s.logical_and(s.fgt(param0, s.float(0.02)), s.flt(param0, s.float(0.49)))
    var third_uv = s.select_f(has_custom_uv, param0, s.float(1.0 / 3.0))
    var uv_l = s.select_f(is_uv_frac, u0, third_uv)
    var uv_t = s.select_f(is_uv_frac, v0, third_uv)
    var uv_r = s.select_f(is_uv_frac, u1, third_uv)
    var uv_b = s.select_f(is_uv_frac, v1, third_uv)

    var ns_u = _emit_nine_slice_axis(s, nlx, nw, bl_px, br_px, uv_l, uv_r)
    var ns_v = _emit_nine_slice_axis(s, nly, nh, bt_px, bb_px, uv_t, uv_b)
    var ns_samp = s.sample_bindless_lod(bindless_arr, tex_slot, s.vec2(ns_u, ns_v), s.float(0.0))
    var is_lin_ns = s.ine(s.bit_and(flags, s.uint(32)), s.uint(0))
    var nsr_srgb = s.select_f(is_lin_ns, _emit_srgb_oetf(s, s.extract_f(ns_samp, 0)), s.extract_f(ns_samp, 0))
    var nsg_srgb = s.select_f(is_lin_ns, _emit_srgb_oetf(s, s.extract_f(ns_samp, 1)), s.extract_f(ns_samp, 1))
    var nsb_srgb = s.select_f(is_lin_ns, _emit_srgb_oetf(s, s.extract_f(ns_samp, 2)), s.extract_f(ns_samp, 2))
    var nsa = s.extract_f(ns_samp, 3)
    var wns = _emit_ui_srgb_to_working(
        s, is_hdr10, s.fmul(cr, nsr_srgb), s.fmul(cg, nsg_srgb), s.fmul(cb, nsb_srgb)
    )
    _emit_blend_over(s, accum_r, accum_g, accum_b, wns[0], wns[1], wns[2], s.fmul(ca, nsa))
    s.end_if()

    # ------------------------------------------------------------------------
    # Kind 3 (`PRIM_CONTROLLER_SYMBOL`): Procedural GPU SDF Controller / Icon Glyphs
    # ------------------------------------------------------------------------
    s.begin_if(s.ieq(kind, s.uint(3)))
    var sw = s.nmax(s.float(2.0), s.fsub(x1, x0))
    var sh = s.nmax(s.float(2.0), s.fsub(y1, y0))
    var scx = s.fma(sw, s.float(0.5), x0)
    var scy = s.fma(sh, s.float(0.5), y0)
    var r_half = s.fmul(s.nmin(sw, sh), s.float(0.5))
    var nx = s.fdiv(s.fsub(fgx, scx), r_half)
    var ny = s.fdiv(s.fsub(fgy, scy), r_half)
    var sym_id = tex_slot
    var st_w = s.select_f(
        s.logical_and(s.fgt(param1, s.float(0.01)), s.flt(param1, s.float(0.45))),
        param1,
        s.float(0.11),
    )

    # Optional circular button badge in `secondary_rgba8`
    s.begin_if(s.fgt(sa, s.float(0.001)))
    var r_len = s.fsqrt(s.fma(nx, nx, s.fmul(ny, ny)))
    var badge_d = s.fmul(s.fsub(r_len, s.float(0.92)), r_half)
    var badge_cov = s.nclamp(s.fsub(s.float(0.5), badge_d), s.float(0.0), s.float(1.0))
    var wbadge = _emit_ui_srgb_to_working(s, is_hdr10, sr, sg, sb)
    _emit_blend_over(s, accum_r, accum_g, accum_b, wbadge[0], wbadge[1], wbadge[2], s.fmul(sa, badge_cov))
    s.end_if()

    var d_sym = s.fvar(s.float(10.0))

    # 0: 'A'
    s.begin_if(s.ieq(sym_id, s.uint(0)))
    var da0 = _emit_seg_dist(s, nx, ny, s.float(-0.36), s.float(0.44), s.float(0.0), s.float(-0.44))
    var da1 = _emit_seg_dist(s, nx, ny, s.float(0.0), s.float(-0.44), s.float(0.36), s.float(0.44))
    var da2 = _emit_seg_dist(s, nx, ny, s.float(-0.22), s.float(0.14), s.float(0.22), s.float(0.14))
    s.store(d_sym, s.fsub(s.nmin(da0, s.nmin(da1, da2)), st_w))
    s.end_if()

    # 1: 'B'
    s.begin_if(s.ieq(sym_id, s.uint(1)))
    var db_sp = _emit_seg_dist(s, nx, ny, s.float(-0.26), s.float(-0.44), s.float(-0.26), s.float(0.44))
    var db_t = _emit_seg_dist(s, nx, ny, s.float(-0.26), s.float(-0.44), s.float(0.06), s.float(-0.44))
    var db_m = _emit_seg_dist(s, nx, ny, s.float(-0.26), s.float(0.0), s.float(0.08), s.float(0.0))
    var db_b = _emit_seg_dist(s, nx, ny, s.float(-0.26), s.float(0.44), s.float(0.08), s.float(0.44))
    var lob_u = s.fabs(s.fsub(s.fsqrt(s.fma(s.fsub(nx, s.float(0.06)), s.fsub(nx, s.float(0.06)), s.fmul(s.fadd(ny, s.float(0.22)), s.fadd(ny, s.float(0.22))))), s.float(0.22)))
    var lob_l = s.fabs(s.fsub(s.fsqrt(s.fma(s.fsub(nx, s.float(0.08)), s.fsub(nx, s.float(0.08)), s.fmul(s.fsub(ny, s.float(0.22)), s.fsub(ny, s.float(0.22))))), s.float(0.22)))
    var lob_arc = s.select_f(s.fge(nx, s.float(0.04)), s.nmin(lob_u, lob_l), s.float(10.0))
    s.store(d_sym, s.fsub(s.nmin(s.nmin(db_sp, lob_arc), s.nmin(db_t, s.nmin(db_m, db_b))), st_w))
    s.end_if()

    # 2 ('X') or 4 ('Cross'): Diagonal cross
    s.begin_if(s.logical_or(s.ieq(sym_id, s.uint(2)), s.ieq(sym_id, s.uint(4))))
    var dx0 = _emit_seg_dist(s, nx, ny, s.float(-0.38), s.float(-0.38), s.float(0.38), s.float(0.38))
    var dx1 = _emit_seg_dist(s, nx, ny, s.float(-0.38), s.float(0.38), s.float(0.38), s.float(-0.38))
    s.store(d_sym, s.fsub(s.nmin(dx0, dx1), st_w))
    s.end_if()

    # 3: 'Y'
    s.begin_if(s.ieq(sym_id, s.uint(3)))
    var dy0 = _emit_seg_dist(s, nx, ny, s.float(-0.36), s.float(-0.42), s.float(0.0), s.float(0.02))
    var dy1 = _emit_seg_dist(s, nx, ny, s.float(0.36), s.float(-0.42), s.float(0.0), s.float(0.02))
    var dy2 = _emit_seg_dist(s, nx, ny, s.float(0.0), s.float(0.02), s.float(0.0), s.float(0.44))
    s.store(d_sym, s.fsub(s.nmin(dy0, s.nmin(dy1, dy2)), st_w))
    s.end_if()

    # 5: 'Circle'
    s.begin_if(s.ieq(sym_id, s.uint(5)))
    var r_c = s.fsqrt(s.fma(nx, nx, s.fmul(ny, ny)))
    s.store(d_sym, s.fsub(s.fabs(s.fsub(r_c, s.float(0.42))), st_w))
    s.end_if()

    # 6: 'Triangle'
    s.begin_if(s.ieq(sym_id, s.uint(6)))
    var dt0 = _emit_seg_dist(s, nx, ny, s.float(0.0), s.float(-0.44), s.float(0.40), s.float(0.30))
    var dt1 = _emit_seg_dist(s, nx, ny, s.float(0.40), s.float(0.30), s.float(-0.40), s.float(0.30))
    var dt2 = _emit_seg_dist(s, nx, ny, s.float(-0.40), s.float(0.30), s.float(0.0), s.float(-0.44))
    s.store(d_sym, s.fsub(s.nmin(dt0, s.nmin(dt1, dt2)), st_w))
    s.end_if()

    # 7: 'Square'
    s.begin_if(s.ieq(sym_id, s.uint(7)))
    var sq_d = s.fabs(s.fsub(s.nmax(s.fabs(nx), s.fabs(ny)), s.float(0.38)))
    s.store(d_sym, s.fsub(sq_d, st_w))
    s.end_if()

    # 8: 'Play' (filled right-pointing triangle `(-0.30, -0.42), (-0.30, 0.42), (0.42, 0.0)`)
    s.begin_if(s.ieq(sym_id, s.uint(8)))
    var e_left = s.fsub(s.float(-0.30), nx)
    var e_top = s.fmul(s.fma(s.float(0.42), nx, s.fma(s.float(0.72), ny, s.float(-0.1764))), s.float(1.1997))
    var e_bot = s.fmul(s.fma(s.float(0.42), nx, s.fma(s.float(-0.72), ny, s.float(-0.1764))), s.float(1.1997))
    var in_play = s.nmax(e_left, s.nmax(e_top, e_bot))
    var dp0 = _emit_seg_dist(s, nx, ny, s.float(-0.30), s.float(-0.42), s.float(-0.30), s.float(0.42))
    var dp1 = _emit_seg_dist(s, nx, ny, s.float(-0.30), s.float(-0.42), s.float(0.42), s.float(0.0))
    var dp2 = _emit_seg_dist(s, nx, ny, s.float(-0.30), s.float(0.42), s.float(0.42), s.float(0.0))
    var edge_play = s.nmin(dp0, s.nmin(dp1, dp2))
    s.store(d_sym, s.select_f(s.flt(in_play, s.float(0.0)), s.fneg(edge_play), edge_play))
    s.end_if()

    # 9: 'Pause' (two vertical rounded bars)
    s.begin_if(s.ieq(sym_id, s.uint(9)))
    var bar_x = s.fabs(s.fsub(s.fabs(nx), s.float(0.22)))
    var bar_d = _emit_rounded_box_sdf(
        s, bar_x, s.fabs(ny), s.float(0.0), s.float(0.0), s.float(0.10), s.float(0.40), s.float(0.04)
    )
    s.store(d_sym, bar_d)
    s.end_if()

    # 10+: 'Gizmo' (crosshair ring + 4 cardinal ticks + center dot)
    s.begin_if(s.uge(sym_id, s.uint(10)))
    var rg = s.fsqrt(s.fma(nx, nx, s.fmul(ny, ny)))
    var d_ring = s.fsub(s.fabs(s.fsub(rg, s.float(0.36))), st_w)
    var d_h = _emit_seg_dist(s, s.fabs(nx), ny, s.float(0.22), s.float(0.0), s.float(0.56), s.float(0.0))
    var d_v = _emit_seg_dist(s, nx, s.fabs(ny), s.float(0.0), s.float(0.22), s.float(0.0), s.float(0.56))
    var d_dot = s.fsub(rg, st_w)
    s.store(d_sym, s.nmin(d_ring, s.nmin(d_dot, s.fsub(s.nmin(d_h, d_v), s.fmul(st_w, s.float(0.85))))))
    s.end_if()

    var cov_sym = s.nclamp(s.fsub(s.float(0.5), s.fmul(s.load_f(d_sym), r_half)), s.float(0.0), s.float(1.0))
    var wsym = _emit_ui_srgb_to_working(s, is_hdr10, cr, cg, cb)
    _emit_blend_over(s, accum_r, accum_g, accum_b, wsym[0], wsym[1], wsym[2], s.fmul(ca, cov_sym))
    s.end_if()

    # ------------------------------------------------------------------------
    # Kind 4 (`PRIM_LINE`): Analytical Capsule Segment & Quadratic Bézier SDF + Dashed Flow
    # ------------------------------------------------------------------------
    s.begin_if(s.ieq(kind, s.uint(4)))
    var is_bez = s.ine(s.bit_and(flags, s.uint(16)), s.uint(0))
    var is_flow = s.ine(s.bit_and(flags, s.uint(UI_FLAG_DASHED_FLOW)), s.uint(0))
    var uv_is_default = s.logical_and(
        s.logical_and(s.flt(s.fabs(u0), s.float(1e-5)), s.flt(s.fabs(v0), s.float(1e-5))),
        s.logical_and(s.flt(s.fabs(s.fsub(u1, s.float(1.0))), s.float(1e-5)), s.flt(s.fabs(s.fsub(v1, s.float(1.0))), s.float(1e-5))),
    )
    var half_thick = s.select_f(is_bez, s.nmax(param2, s.float(0.5)), s.nmax(param0, s.float(0.5)))
    var d_line = s.fvar(s.float(1e6))
    var s_along = s.fvar(s.float(0.0))

    s.begin_if(is_bez)
    var p0x = u0
    var p0y = v0
    var p1x = param0
    var p1y = param1
    var p2x = u1
    var p2y = v1

    var prev_bx = s.fvar(p0x)
    var prev_by = s.fvar(p0y)
    var cum_len = s.fvar(s.float(0.0))
    var seg_i = s.uvar(s.uint(1))
    s.begin_loop()
    var si = s.load_u(seg_i)
    s.break_if(s.uge(si, s.uint(13)))
    var t_b = s.fmul(s.float_from_uint(si), s.float(1.0 / 12.0))
    var omt = s.fsub(s.float(1.0), t_b)
    var w0_b = s.fmul(omt, omt)
    var w1_b = s.fmul(s.float(2.0), s.fmul(omt, t_b))
    var w2_b = s.fmul(t_b, t_b)
    var cur_bx = s.fma(w0_b, p0x, s.fma(w1_b, p1x, s.fmul(w2_b, p2x)))
    var cur_by = s.fma(w0_b, p0y, s.fma(w1_b, p1y, s.fmul(w2_b, p2y)))
    var pbx = s.load_f(prev_bx)
    var pby = s.load_f(prev_by)
    var d_ch = _emit_seg_dist(s, fgx, fgy, pbx, pby, cur_bx, cur_by)
    var ch_dx = s.fsub(cur_bx, pbx)
    var ch_dy = s.fsub(cur_by, pby)
    var ch_len = s.fsqrt(s.fma(ch_dx, ch_dx, s.fmul(ch_dy, ch_dy)))
    s.begin_if(s.flt(d_ch, s.load_f(d_line)))
    s.store(d_line, d_ch)
    s.begin_if(is_flow)
    var c_dot = s.nmax(s.float(1e-6), s.fma(ch_dx, ch_dx, s.fmul(ch_dy, ch_dy)))
    var c_h = s.nclamp(s.fdiv(s.fma(s.fsub(fgx, pbx), ch_dx, s.fmul(s.fsub(fgy, pby), ch_dy)), c_dot), s.float(0.0), s.float(1.0))
    s.store(s_along, s.fma(c_h, ch_len, s.load_f(cum_len)))
    s.end_if()
    s.end_if()
    s.store(cum_len, s.fadd(s.load_f(cum_len), ch_len))
    s.store(prev_bx, cur_bx)
    s.store(prev_by, cur_by)
    s.store(seg_i, s.iadd(si, s.uint(1)))
    s.end_loop()
    s.begin_else()
    var lax = s.select_f(uv_is_default, x0, u0)
    var lay = s.select_f(uv_is_default, y0, v0)
    var lbx = s.select_f(uv_is_default, x1, u1)
    var lby = s.select_f(uv_is_default, y1, v1)
    s.store(d_line, _emit_seg_dist(s, fgx, fgy, lax, lay, lbx, lby))
    s.begin_if(is_flow)
    var ldx = s.fsub(lbx, lax)
    var ldy = s.fsub(lby, lay)
    var l_sq = s.nmax(s.float(1e-6), s.fma(ldx, ldx, s.fmul(ldy, ldy)))
    var l_len = s.fsqrt(l_sq)
    var l_h = s.nclamp(s.fdiv(s.fma(s.fsub(fgx, lax), ldx, s.fmul(s.fsub(fgy, lay), ldy)), l_sq), s.float(0.0), s.float(1.0))
    s.store(s_along, s.fmul(l_h, l_len))
    s.end_if()
    s.end_if()

    var feather = s.select_f(is_bez, s.float(1.0), s.nmax(param2, s.float(0.5)))
    var sd_line = s.fsub(s.load_f(d_line), half_thick)
    var cov_line = s.nclamp(s.fsub(s.float(0.5), s.fdiv(sd_line, feather)), s.float(0.0), s.float(1.0))
    var wln = _emit_ui_srgb_to_working(s, is_hdr10, cr, cg, cb)
    s.begin_if(is_flow)
    var f_dash = s.nmax(s.float(2.0), s.float_from_uint(s.bit_and(sec_raw, s.uint(0xFF))))
    var f_gap = s.float_from_uint(s.bit_and(s.shr(sec_raw, s.uint(8)), s.uint(0xFF)))
    var f_spd = s.fmul(s.float_from_uint(s.bit_and(s.shr(sec_raw, s.uint(16)), s.uint(0xFF))), s.float(4.0))
    var f_glw = s.fmul(s.float_from_uint(s.bit_and(s.shr(sec_raw, s.uint(24)), s.uint(0xFF))), s.float(1.0 / 255.0))
    var f_per = s.nmax(s.float(2.0), s.fadd(f_dash, f_gap))
    var f_cyc = s.fdiv(s.fsub(s.load_f(s_along), s.fmul(ui_time_sec, f_spd)), f_per)
    var f_ph = s.fmul(s.fsub(f_cyc, s.ffloor(f_cyc)), f_per)
    var f_mask = s.select_f(
        s.fgt(f_gap, s.float(0.5)),
        s.nclamp(s.fadd(s.nmin(f_ph, s.fsub(f_dash, f_ph)), s.float(0.5)), s.float(0.0), s.float(1.0)),
        s.float(1.0),
    )
    s.begin_if(s.fgt(f_glw, s.float(0.01)))
    var lg_r = s.nmax(s.float(2.0), s.fmul(half_thick, s.float(2.5)))
    var lg_u = s.fdiv(s.nmax(sd_line, s.float(0.0)), lg_r)
    var lg_a = s.fmul(
        s.exp2(s.fmul(s.float(-1.8), s.fmul(lg_u, lg_u))),
        s.fmul(f_glw, s.fmul(ca, s.fma(f_mask, s.float(0.65), s.float(0.35)))),
    )
    _emit_blend_over(s, accum_r, accum_g, accum_b, wln[0], wln[1], wln[2], lg_a)
    s.end_if()
    _emit_blend_over(s, accum_r, accum_g, accum_b, wln[0], wln[1], wln[2], s.fmul(ca, s.fmul(cov_line, f_mask)))
    s.begin_else()
    _emit_blend_over(s, accum_r, accum_g, accum_b, wln[0], wln[1], wln[2], s.fmul(ca, cov_line))
    s.end_if()
    s.end_if()

    # ------------------------------------------------------------------------
    # Kind 8 (`PRIM_ARC`): Analytical SDF Ring / Rounded-Cap Arc / Progress / Spinner
    # ------------------------------------------------------------------------
    s.begin_if(s.ieq(kind, s.uint(PRIM_ARC)))
    var arc_cx = u0
    var arc_cy = v0
    var arc_start_raw = u1
    var arc_sweep = v1
    var arc_r = s.nmax(s.float(0.5), param0)
    var arc_half = s.fmul(s.nmax(s.float(0.5), param1), s.float(0.5))
    var adx = s.fsub(fgx, arc_cx)
    var ady = s.fsub(fgy, arc_cy)
    var dist_c = s.fsqrt(s.fma(adx, adx, s.fmul(ady, ady)))
    var d_arc_ring = s.fsub(s.fabs(s.fsub(dist_c, arc_r)), arc_half)

    var has_arc_track = s.logical_and(s.ine(s.bit_and(flags, s.uint(UI_FLAG_BORDER)), s.uint(0)), s.fgt(sa, s.float(0.001)))
    s.begin_if(has_arc_track)
    var cov_tr = s.nclamp(s.fsub(s.float(0.5), d_arc_ring), s.float(0.0), s.float(1.0))
    var wtr = _emit_ui_srgb_to_working(s, is_hdr10, sr, sg, sb)
    _emit_blend_over(s, accum_r, accum_g, accum_b, wtr[0], wtr[1], wtr[2], s.fmul(sa, cov_tr))
    s.end_if()

    var sw_abs = s.fabs(arc_sweep)
    s.begin_if(s.fgt(sw_abs, s.float(1e-4)))
    var has_arc_spin = s.ine(s.bit_and(flags, s.uint(UI_FLAG_DASHED_FLOW)), s.uint(0))
    var eff_start = s.fadd(arc_start_raw, s.select_f(has_arc_spin, s.fmul(ui_time_sec, s.float(3.0)), s.float(0.0)))
    var d_arc = s.fvar(d_arc_ring)
    s.begin_if(s.flt(sw_abs, s.float(6.27)))
    var st_ang = s.select_f(s.fge(arc_sweep, s.float(0.0)), eff_start, s.fadd(eff_start, arc_sweep))
    var ang = s.fatan2(ady, adx)
    var two_pi = s.float(6.2831853)
    var rel_raw = s.fsub(ang, st_ang)
    var rel = s.fsub(rel_raw, s.fmul(two_pi, s.ffloor(s.fdiv(rel_raw, two_pi))))
    var sx0 = s.fma(arc_r, s.fcos(st_ang), arc_cx)
    var sy0 = s.fma(arc_r, s.fsin(st_ang), arc_cy)
    var end_ang = s.fadd(st_ang, sw_abs)
    var sx1 = s.fma(arc_r, s.fcos(end_ang), arc_cx)
    var sy1 = s.fma(arc_r, s.fsin(end_ang), arc_cy)
    var d0x = s.fsub(fgx, sx0)
    var d0y = s.fsub(fgy, sy0)
    var d1x = s.fsub(fgx, sx1)
    var d1y = s.fsub(fgy, sy1)
    var dc0 = s.fsqrt(s.fma(d0x, d0x, s.fmul(d0y, d0y)))
    var dc1 = s.fsqrt(s.fma(d1x, d1x, s.fmul(d1y, d1y)))
    var d_cap = s.fsub(s.nmin(dc0, dc1), arc_half)
    s.store(d_arc, s.select_f(s.fle(rel, sw_abs), d_arc_ring, d_cap))
    s.end_if()
    var da_val = s.load_f(d_arc)
    var cov_arc = s.nclamp(s.fsub(s.float(0.5), da_val), s.float(0.0), s.float(1.0))
    var warc = _emit_ui_srgb_to_working(s, is_hdr10, cr, cg, cb)
    s.begin_if(s.logical_and(has_outer_glow, s.fgt(param2, s.float(0.01))))
    var ag_soft = s.nclamp(param2, s.float(1.5), s.float(28.0))
    var ag_u = s.fdiv(s.nmax(da_val, s.float(0.0)), ag_soft)
    var ag_a = s.fmul(s.exp2(s.fmul(s.float(-2.4), s.fmul(ag_u, ag_u))), s.fmul(s.fsub(s.float(1.0), cov_arc), s.fmul(ca, s.float(0.65))))
    _emit_blend_over(s, accum_r, accum_g, accum_b, warc[0], warc[1], warc[2], ag_a)
    s.end_if()
    _emit_blend_over(s, accum_r, accum_g, accum_b, warc[0], warc[1], warc[2], s.fmul(ca, cov_arc))
    s.end_if()
    s.end_if()

    # ------------------------------------------------------------------------
    # Kind 6 (`PRIM_POLYGON`): Nonzero-Winding Polygon (several outlines, holes, unions)
    # u0 = vertex count; the vertices follow in PRIM_POLY_DATA prims (7 per prim, words
    # 0..13), outlines closed and separated by a vertex with x > 1e37. Pass 1: the winding
    # number at the pixel. Pass 2: the distance to the nearest *boundary* edge (one whose
    # far side flips inside/outside), so internal edges of unions leave no seams.
    # ------------------------------------------------------------------------
    s.begin_if(s.ieq(kind, s.uint(6)))
    var pg_first = s.iadd(prim_idx, s.uint(1))
    var pg_avail = s.imul(s.isub(s.umax(prim_cnt, pg_first), pg_first), s.uint(7))
    var pg_n = s.umin(s.uint_from_float(s.nmax(u0, s.float(0.0))), pg_avail)
    var pg_base = s.iadd(prims_off, s.imul(pg_first, s.uint(16)))
    # Inside/outside and edge sides are judged at a point nudged off the pixel centre, so a
    # centre lying exactly on an edge (axis-aligned edges at .5) has a definite side.
    var pg_px = s.fadd(fgx, s.float(1.3e-3))
    var pg_py = s.fadd(fgy, s.float(0.7e-3))
    var pg_wn = s.fvar(s.float(0.0))
    var pg_dmin = s.fvar(s.float(1e6))
    var pg_pass = s.uvar(s.uint(0))
    s.begin_loop()
    var pass_i = s.load_u(pg_pass)
    s.break_if(s.uge(pass_i, s.uint(2)))
    var wn_final = s.load_f(pg_wn)
    var pg_ax = s.fvar(s.load_float(b_post, pg_base))
    var pg_ay = s.fvar(s.load_float(b_post, s.iadd(pg_base, s.uint(1))))
    var pg_k = s.uvar(s.uint(1))
    s.begin_loop()
    var vk = s.load_u(pg_k)
    s.break_if(s.uge(vk, pg_n))
    var vw = s.iadd(pg_base, s.iadd(s.imul(s.udiv(vk, s.uint(7)), s.uint(16)), s.imul(s.umod(vk, s.uint(7)), s.uint(2))))
    var ebx = s.load_float(b_post, vw)
    var eby = s.load_float(b_post, s.iadd(vw, s.uint(1)))
    var eax = s.load_f(pg_ax)
    var eay = s.load_f(pg_ay)
    s.begin_if(s.logical_and(s.flt(eax, s.float(1e37)), s.flt(ebx, s.float(1e37))))
    var is_left = s.fsub(s.fmul(s.fsub(ebx, eax), s.fsub(pg_py, eay)), s.fmul(s.fsub(pg_px, eax), s.fsub(eby, eay)))
    s.begin_if(s.ieq(pass_i, s.uint(0)))
    var e_up = s.logical_and(s.logical_and(s.fle(eay, pg_py), s.fgt(eby, pg_py)), s.fgt(is_left, s.float(0.0)))
    var e_dn = s.logical_and(s.logical_and(s.fgt(eay, pg_py), s.fle(eby, pg_py)), s.flt(is_left, s.float(0.0)))
    s.store(pg_wn, s.fadd(s.load_f(pg_wn), s.fsub(s.select_f(e_up, s.float(1.0), s.float(0.0)), s.select_f(e_dn, s.float(1.0), s.float(0.0)))))
    s.begin_else()
    # The winding one step across this edge: one less on its left side, one more on its right.
    var wn_other = s.fadd(wn_final, s.select_f(s.fgt(is_left, s.float(0.0)), s.float(-1.0), s.float(1.0)))
    var in_here = s.fgt(s.fabs(wn_final), s.float(0.5))
    var in_there = s.fgt(s.fabs(wn_other), s.float(0.5))
    s.begin_if(s.logical_or(s.logical_and(in_here, s.logical_not(in_there)), s.logical_and(in_there, s.logical_not(in_here))))
    s.store(pg_dmin, s.nmin(s.load_f(pg_dmin), _emit_seg_dist(s, fgx, fgy, eax, eay, ebx, eby)))
    s.end_if()
    s.end_if()
    s.end_if()
    s.store(pg_ax, ebx)
    s.store(pg_ay, eby)
    s.store(pg_k, s.iadd(vk, s.uint(1)))
    s.end_loop()
    s.store(pg_pass, s.iadd(pass_i, s.uint(1)))
    s.end_loop()
    var pg_inside = s.fgt(s.fabs(s.load_f(pg_wn)), s.float(0.5))
    var pg_sd = s.select_f(pg_inside, s.fneg(s.load_f(pg_dmin)), s.load_f(pg_dmin))
    var cov_pg = s.nclamp(s.fsub(s.float(0.5), pg_sd), s.float(0.0), s.float(1.0))
    var wpg = _emit_ui_srgb_to_working(s, is_hdr10, cr, cg, cb)
    _emit_blend_over(s, accum_r, accum_g, accum_b, wpg[0], wpg[1], wpg[2], s.fmul(ca, cov_pg))
    s.end_if()

    # ------------------------------------------------------------------------
    # Kind 5 (`PRIM_IMAGE`): Bindless Textured Rounded-Rect Quad
    # ------------------------------------------------------------------------
    s.begin_if(s.ieq(kind, s.uint(5)))
    var iw = s.nmax(s.float(1.0), s.fsub(x1, x0))
    var ih = s.nmax(s.float(1.0), s.fsub(y1, y0))
    var iw_half = s.fmul(iw, s.float(0.5))
    var ih_half = s.fmul(ih, s.float(0.5))
    var icx = s.fadd(x0, iw_half)
    var icy = s.fadd(y0, ih_half)
    var d_img = _emit_rounded_box_sdf(s, fgx, fgy, icx, icy, iw_half, ih_half, param0)
    var cov_img = s.nclamp(s.fsub(s.float(0.5), d_img), s.float(0.0), s.float(1.0))

    var itu = s.nclamp(s.fadd(u0, s.fmul(s.fdiv(s.fsub(fgx, x0), iw), s.fsub(u1, u0))), s.float(0.0), s.float(1.0))
    var itv = s.nclamp(s.fadd(v0, s.fmul(s.fdiv(s.fsub(fgy, y0), ih), s.fsub(v1, v0))), s.float(0.0), s.float(1.0))
    var img_samp = s.sample_bindless_lod(bindless_arr, tex_slot, s.vec2(itu, itv), s.float(0.0))
    var is_lin_img = s.ine(s.bit_and(flags, s.uint(32)), s.uint(0))
    var ir_srgb = s.select_f(is_lin_img, _emit_srgb_oetf(s, s.extract_f(img_samp, 0)), s.extract_f(img_samp, 0))
    var ig_srgb = s.select_f(is_lin_img, _emit_srgb_oetf(s, s.extract_f(img_samp, 1)), s.extract_f(img_samp, 1))
    var ib_srgb = s.select_f(is_lin_img, _emit_srgb_oetf(s, s.extract_f(img_samp, 2)), s.extract_f(img_samp, 2))
    var ia = s.extract_f(img_samp, 3)

    var has_img_border = s.logical_and(s.ine(s.bit_and(flags, s.uint(2)), s.uint(0)), s.fgt(param1, s.float(0.01)))
    s.begin_if(has_img_border)
    var cov_img_in = s.nclamp(s.fsub(s.float(0.5), s.fadd(d_img, param1)), s.float(0.0), s.float(1.0))
    var wim = _emit_ui_srgb_to_working(s, is_hdr10, s.fmul(cr, ir_srgb), s.fmul(cg, ig_srgb), s.fmul(cb, ib_srgb))
    _emit_blend_over(s, accum_r, accum_g, accum_b, wim[0], wim[1], wim[2], s.fmul(s.fmul(ca, ia), cov_img_in))
    var wb_im = _emit_ui_srgb_to_working(s, is_hdr10, sr, sg, sb)
    _emit_blend_over(s, accum_r, accum_g, accum_b, wb_im[0], wb_im[1], wb_im[2], s.fmul(sa, s.nmax(s.float(0.0), s.fsub(cov_img, cov_img_in))))
    s.begin_else()
    var wim2 = _emit_ui_srgb_to_working(s, is_hdr10, s.fmul(cr, ir_srgb), s.fmul(cg, ig_srgb), s.fmul(cb, ib_srgb))
    _emit_blend_over(s, accum_r, accum_g, accum_b, wim2[0], wim2[1], wim2[2], s.fmul(s.fmul(ca, ia), cov_img))
    s.end_if()
    s.end_if()

    s.end_if()  # in_box
    s.end_if()  # in_clip

    s.store(i_var, s.iadd(i, s.uint(1)))
    s.end_loop()
    s.end_if()  # in_screen

    s.workgroup_barrier()
    s.end_if()  # chunk_hits != 0

    s.store(chunk_base_var, s.iadd(chunk_base, s.uint(UInt32(TILE_PRIM_CAP))))
    s.end_loop()

    # Write final composited pixel back to swapchain image or `b_hist` if any UI chunk touched this tile
    s.begin_if(s.logical_and(in_screen, s.ine(s.load_u(bg_loaded), s.uint(0))))
    if direct_swapchain:
        var out_r = s.fvar(s.float(0.0))
        var out_g = s.fvar(s.float(0.0))
        var out_b = s.fvar(s.float(0.0))
        s.begin_if(is_hdr10)
        s.store(out_r, _emit_pq_oetf_100nits(s, s.load_f(accum_r)))
        s.store(out_g, _emit_pq_oetf_100nits(s, s.load_f(accum_g)))
        s.store(out_b, _emit_pq_oetf_100nits(s, s.load_f(accum_b)))
        s.begin_else()
        s.store(out_r, s.nclamp(s.load_f(accum_r), s.float(0.0), s.float(1.0)))
        s.store(out_g, s.nclamp(s.load_f(accum_g), s.float(0.0), s.float(1.0)))
        s.store(out_b, s.nclamp(s.load_f(accum_b), s.float(0.0), s.float(1.0)))
        s.end_if()
        s.image_write(swap_img, s.uvec2(gx, gy), s.vec4(s.load_f(out_r), s.load_f(out_g), s.load_f(out_b), s.float(1.0)))
    else:
        var packed_out = s.uvar(s.uint(0xFF000000))
        s.begin_if(is_hdr10)
        var pqr_o = s.umin(s.uint(1023), s.saturate_uint(s.fma(_emit_pq_oetf_100nits(s, s.load_f(accum_r)), s.float(1023.0), s.float(0.5))))
        var pqg_o = s.umin(s.uint(1023), s.saturate_uint(s.fma(_emit_pq_oetf_100nits(s, s.load_f(accum_g)), s.float(1023.0), s.float(0.5))))
        var pqb_o = s.umin(s.uint(1023), s.saturate_uint(s.fma(_emit_pq_oetf_100nits(s, s.load_f(accum_b)), s.float(1023.0), s.float(0.5))))
        s.store(
            packed_out,
            s.bit_or(s.uint(0xC0000000), s.bit_or(s.shl(pqb_o, s.uint(20)), s.bit_or(s.shl(pqg_o, s.uint(10)), pqr_o))),
        )
        s.begin_else()
        var final_r = s.umin(s.uint(255), s.saturate_uint(s.fma(s.nclamp(s.load_f(accum_r), s.float(0.0), s.float(1.0)), s.float(255.0), s.float(0.5))))
        var final_g = s.umin(s.uint(255), s.saturate_uint(s.fma(s.nclamp(s.load_f(accum_g), s.float(0.0), s.float(1.0)), s.float(255.0), s.float(0.5))))
        var final_b = s.umin(s.uint(255), s.saturate_uint(s.fma(s.nclamp(s.load_f(accum_b), s.float(0.0), s.float(1.0)), s.float(255.0), s.float(0.5))))
        s.store(
            packed_out,
            s.bit_or(
                s.uint(0xFF000000),
                s.bit_or(s.shl(final_r, s.uint(16)), s.bit_or(s.shl(final_g, s.uint(8)), final_b)),
            ),
        )
        s.end_if()
        s.store_word(b_hist, pix_idx, s.load_u(packed_out))
    s.end_if()

    return s.finish()


def aaa_ui_overlay_direct_shader(swapchain_binding: Int = BIND_AAA_SWAPCHAIN_IMG) -> List[UInt32]:
    """Direct-to-Swapchain GPU UI SDF Overlay Compute Pass (`storage image2D`)."""
    return aaa_ui_overlay_shader(direct_swapchain=True, swapchain_binding=swapchain_binding)


def aaa_ui_present_fragment_shader() -> List[UInt32]:
    """Fullscreen `VK_KHR_dynamic_rendering` alpha-blended UI SDF fragment pass for non-storage swapchains."""
    var s = AaaSpirvModule(MODEL_FRAGMENT, push_words=AAA_PUSH_WORDS, enable_bindless=True, early_depth=False)
    var b_post = s.buffer(BIND_AAA_POST_SCRATCH)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var out_col = s.add_stage_io(STORAGE_OUTPUT, s.type_vec4, location=0, flat=False)

    var fc = s.frag_coord()
    var fgx = s.extract_f(fc, 0)
    var fgy = s.extract_f(fc, 1)
    var gx = s.saturate_uint(s.ffloor(fgx))
    var gy = s.saturate_uint(s.ffloor(fgy))
    var prim_cnt = s.push_word(2)
    var prims_off = s.push_word(3)
    var is_srgb_rt = s.ine(s.push_word(10), s.uint(0))

    var acc_r = s.fvar(s.float(0.0))
    var acc_g = s.fvar(s.float(0.0))
    var acc_b = s.fvar(s.float(0.0))
    var acc_a = s.fvar(s.float(0.0))

    var i_v = s.uvar(s.uint(0))
    s.begin_loop()
    var i = s.load_u(i_v)
    s.break_if(s.uge(i, prim_cnt))
    var pb = s.iadd(prims_off, s.imul(i, s.uint(16)))
    var cxy0 = s.load_word(b_post, s.iadd(pb, s.uint(8)))
    var cxy1 = s.load_word(b_post, s.iadd(pb, s.uint(9)))
    var cx0 = s.bit_and(cxy0, s.uint(0xFFFF))
    var cy0 = s.shr(cxy0, s.uint(16))
    var cx1 = s.bit_and(cxy1, s.uint(0xFFFF))
    var cy1 = s.shr(cxy1, s.uint(16))
    var in_clip = s.logical_and(
        s.logical_and(s.uge(gx, cx0), s.ult(gx, cx1)),
        s.logical_and(s.uge(gy, cy0), s.ult(gy, cy1)),
    )
    s.begin_if(in_clip)
    var x0 = s.load_float(b_post, pb)
    var y0 = s.load_float(b_post, s.iadd(pb, s.uint(1)))
    var x1 = s.load_float(b_post, s.iadd(pb, s.uint(2)))
    var y1 = s.load_float(b_post, s.iadd(pb, s.uint(3)))
    var hdr = s.load_word(b_post, s.iadd(pb, s.uint(15)))
    var kind = s.bit_and(hdr, s.uint(0xF))
    var tex_slot = s.bit_and(s.shr(hdr, s.uint(4)), s.uint(0xFFF))
    var param0 = s.load_float(b_post, s.iadd(pb, s.uint(12)))
    var param1 = s.load_float(b_post, s.iadd(pb, s.uint(13)))
    var pad = s.nmax(s.float(1.0), s.nmax(param0, param1))
    var in_box = s.logical_and(
        s.logical_and(s.fge(fgx, s.fsub(s.nmin(x0, x1), pad)), s.fge(s.fadd(s.nmax(x0, x1), pad), fgx)),
        s.logical_and(s.fge(fgy, s.fsub(s.nmin(y0, y1), pad)), s.fge(s.fadd(s.nmax(y0, y1), pad), fgy)),
    )
    s.begin_if(in_box)
    var rgba = _emit_unpack_flint_argb(s, s.load_word(b_post, s.iadd(pb, s.uint(10))))
    var cov_v = s.fvar(s.float(0.0))
    var sr_v = s.fvar(rgba[0])
    var sg_v = s.fvar(rgba[1])
    var sb_v = s.fvar(rgba[2])
    var sa_v = s.fvar(rgba[3])
    s.begin_if(s.ieq(kind, s.uint(0)))
    var hw = s.fmul(s.nmax(s.float(0.5), s.fsub(x1, x0)), s.float(0.5))
    var hh = s.fmul(s.nmax(s.float(0.5), s.fsub(y1, y0)), s.float(0.5))
    var d_s = _emit_rounded_box_sdf(s, fgx, fgy, s.fadd(x0, hw), s.fadd(y0, hh), hw, hh, param0)
    s.store(cov_v, s.nclamp(s.fsub(s.float(0.5), d_s), s.float(0.0), s.float(1.0)))
    s.begin_else()
    s.begin_if(s.ieq(kind, s.uint(4)))
    var dl = _emit_seg_dist(s, fgx, fgy, x0, y0, x1, y1)
    s.store(cov_v, s.nclamp(s.fsub(s.fmul(s.nmax(param0, s.float(1.0)), s.float(0.5)), dl), s.float(0.0), s.float(1.0)))
    s.begin_else()
    var u0 = s.load_float(b_post, s.iadd(pb, s.uint(4)))
    var v0 = s.load_float(b_post, s.iadd(pb, s.uint(5)))
    var u1 = s.load_float(b_post, s.iadd(pb, s.uint(6)))
    var v1 = s.load_float(b_post, s.iadd(pb, s.uint(7)))
    var tu = s.nclamp(s.fadd(u0, s.fmul(s.fdiv(s.fsub(fgx, x0), s.nmax(s.float(0.5), s.fsub(x1, x0))), s.fsub(u1, u0))), s.float(0.0), s.float(1.0))
    var tv = s.nclamp(s.fadd(v0, s.fmul(s.fdiv(s.fsub(fgy, y0), s.nmax(s.float(0.5), s.fsub(y1, y0))), s.fsub(v1, v0))), s.float(0.0), s.float(1.0))
    var ts = s.sample_bindless_lod(bindless_arr, tex_slot, s.vec2(tu, tv), s.float(0.0))
    s.begin_if(s.ieq(kind, s.uint(1)))
    var dist_g = s.extract_f(ts, 0)
    var w_px = s.nmax(s.float(0.04), s.fmul(param0, s.float(0.35)))
    var t_sm = s.nclamp(s.fdiv(s.fsub(dist_g, s.fsub(s.float(0.5), w_px)), s.fmul(w_px, s.float(2.0))), s.float(0.0), s.float(1.0))
    s.store(cov_v, s.fmul(s.fmul(t_sm, t_sm), s.fsub(s.float(3.0), s.fmul(s.float(2.0), t_sm))))
    s.begin_else()
    s.store(sr_v, s.fmul(rgba[0], s.extract_f(ts, 0)))
    s.store(sg_v, s.fmul(rgba[1], s.extract_f(ts, 1)))
    s.store(sb_v, s.fmul(rgba[2], s.extract_f(ts, 2)))
    s.store(sa_v, s.fmul(rgba[3], s.extract_f(ts, 3)))
    s.store(cov_v, s.float(1.0))
    s.end_if()
    s.end_if()
    s.end_if()
    var eff_a = s.fmul(s.load_f(sa_v), s.load_f(cov_v))
    var oma = s.fsub(s.float(1.0), eff_a)
    s.store(acc_r, s.fma(s.load_f(acc_r), oma, s.fmul(s.load_f(sr_v), eff_a)))
    s.store(acc_g, s.fma(s.load_f(acc_g), oma, s.fmul(s.load_f(sg_v), eff_a)))
    s.store(acc_b, s.fma(s.load_f(acc_b), oma, s.fmul(s.load_f(sb_v), eff_a)))
    s.store(acc_a, s.fma(s.load_f(acc_a), oma, eff_a))
    s.end_if()
    s.end_if()
    s.store(i_v, s.iadd(i, s.uint(1)))
    s.end_loop()

    var fa = s.nclamp(s.load_f(acc_a), s.float(0.0), s.float(1.0))
    var inv_a = s.select_f(s.fgt(fa, s.float(1e-5)), s.fdiv(s.float(1.0), fa), s.float(0.0))
    var pr = s.nclamp(s.fmul(s.load_f(acc_r), inv_a), s.float(0.0), s.float(1.0))
    var pg = s.nclamp(s.fmul(s.load_f(acc_g), inv_a), s.float(0.0), s.float(1.0))
    var pb_o = s.nclamp(s.fmul(s.load_f(acc_b), inv_a), s.float(0.0), s.float(1.0))
    var fr = s.select_f(is_srgb_rt, s.fpow(s.nmax(pr, s.float(1e-6)), s.float(2.2)), pr)
    var fg = s.select_f(is_srgb_rt, s.fpow(s.nmax(pg, s.float(1e-6)), s.float(2.2)), pg)
    var fb = s.select_f(is_srgb_rt, s.fpow(s.nmax(pb_o, s.float(1e-6)), s.float(2.2)), pb_o)
    emit(s.code, OP_STORE, out_col[1], s.vec4(fr, fg, fb, fa))
    return s.finish()
