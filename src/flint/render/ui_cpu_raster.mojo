"""Tier-0 CPU SIMD SDF UI Rasterizer (`flint.render.ui_cpu_raster`).

Provides bit-accurate Tier-0 CPU parity with `flint.render.ui_overlay.aaa_ui_overlay_shader`
for deterministic headless rendering across all `GpuUiPrim` kinds & flags:
  - `PRIM_SHAPE`: rounded rect (uniform & 4-corner), circle, outer drop shadow, outer glow,
    5-tap 16x16 tile-clamped Kawase frosted-glass scrim, V/H/diagonal gradients, border, top bevel
  - `PRIM_GLYPH`: bilinear SDF / Chlumsky MSDF / MTSDF sampling + stroke outline + drop shadow
  - `PRIM_CONTROLLER_SYMBOL`: procedural SDF symbols `0..10` + optional circular badge
  - `PRIM_LINE`: capsule segment & 12-chord quadratic Bézier + animated dashed flow & glow
  - `PRIM_POLYGON` / `PRIM_POLY_DATA`: 2-pass nonzero-winding AA filled polygon
  - `PRIM_ARC`: analytical SDF ring / rounded-cap arc / progress track / rotating spinner + glow
"""
from std.collections import List
from std.math import abs, atan2, cos, exp2, floor, isfinite, max, min, sin, sqrt
from std.memory import bitcast
from flint.render.font import SdfFont
from flint.render.simd_canvas import SimdCanvas, _simd8_store_u32
from flint.ui.draw_list import (
    GpuUiPrim,
    POLY_VERTS_PER_PRIM,
    PRIM_ARC,
    PRIM_CONTROLLER_SYMBOL,
    PRIM_GLYPH,
    PRIM_LINE,
    PRIM_POLY_DATA,
    PRIM_POLYGON,
    PRIM_SHAPE,
    UI_FLAG_BORDER,
    UI_FLAG_CIRCLE,
    UI_FLAG_DASHED_FLOW,
    UI_FLAG_DROP_SHADOW,
    UI_FLAG_GLASS_SCRIM,
    UI_FLAG_HORIZ_GRADIENT,
    UI_FLAG_MSDF,
    UI_FLAG_MTSDF,
    UI_FLAG_OUTER_GLOW,
    UI_FLAG_PER_CORNER_RADII,
    UI_FLAG_QUAD_BEZIER,
    UI_FLAG_TOP_BEVEL,
    UI_FLAG_VERT_GRADIENT,
    UiDrawList,
)


@always_inline
def _clampf(v: Float32, lo: Float32, hi: Float32) -> Float32:
    return max(lo, min(hi, v))


@always_inline
def _unpack_argb_f32(c: UInt32) -> Tuple[Float32, Float32, Float32, Float32]:
    var inv = Float32(1.0 / 255.0)
    var b = Float32(c & UInt32(0xFF)) * inv
    var g = Float32((c >> 8) & UInt32(0xFF)) * inv
    var r = Float32((c >> 16) & UInt32(0xFF)) * inv
    var a = Float32((c >> 24) & UInt32(0xFF)) * inv
    return (r, g, b, a)


@always_inline
def _pack_argb_f32(r: Float32, g: Float32, b: Float32) -> UInt32:
    var ir = UInt32(max(0, min(255, Int(_clampf(r, 0.0, 1.0) * Float32(255.0) + Float32(0.5)))))
    var ig = UInt32(max(0, min(255, Int(_clampf(g, 0.0, 1.0) * Float32(255.0) + Float32(0.5)))))
    var ib = UInt32(max(0, min(255, Int(_clampf(b, 0.0, 1.0) * Float32(255.0) + Float32(0.5)))))
    return UInt32(0xFF000000) | (ir << 16) | (ig << 8) | ib


@always_inline
def _blend_pixel(mut pixels: List[UInt32], idx: Int, sr: Float32, sg: Float32, sb: Float32, sa_in: Float32):
    var a = _clampf(sa_in, 0.0, 1.0)
    if a <= Float32(1.0e-5):
        return
    if a >= Float32(0.9999):
        pixels[idx] = _pack_argb_f32(sr, sg, sb)
        return
    var dst = _unpack_argb_f32(pixels[idx])
    var inv = Float32(1.0) - a
    pixels[idx] = _pack_argb_f32(
        sr * a + dst[0] * inv,
        sg * a + dst[1] * inv,
        sb * a + dst[2] * inv,
    )


@always_inline
def _rounded_box_sdf(
    px: Float32,
    py: Float32,
    cx: Float32,
    cy: Float32,
    w_half: Float32,
    h_half: Float32,
    radius: Float32,
) -> Float32:
    var r_lim = _clampf(radius, 0.0, min(w_half, h_half))
    var qx = abs(px - cx) - (w_half - r_lim)
    var qy = abs(py - cy) - (h_half - r_lim)
    var ox = max(qx, Float32(0.0))
    var oy = max(qy, Float32(0.0))
    return sqrt(ox * ox + oy * oy) + min(max(qx, qy), Float32(0.0)) - r_lim


@always_inline
def _seg_dist(
    px: Float32,
    py: Float32,
    ax: Float32,
    ay: Float32,
    bx: Float32,
    by: Float32,
) -> Float32:
    var pax = px - ax
    var pay = py - ay
    var bax = bx - ax
    var bay = by - ay
    var ba_dot = max(Float32(1.0e-6), bax * bax + bay * bay)
    var h = _clampf((pax * bax + pay * bay) / ba_dot, 0.0, 1.0)
    var dx = pax - bax * h
    var dy = pay - bay * h
    return sqrt(dx * dx + dy * dy)


@always_inline
def _poly_vert(prims: List[GpuUiPrim], first_data_idx: Int, vk: Int) -> Tuple[Float32, Float32]:
    var di = first_data_idx + vk // POLY_VERTS_PER_PRIM
    var slot = vk % POLY_VERTS_PER_PRIM
    ref dp = prims[di]
    if slot == 0:
        return (dp.x0, dp.y0)
    if slot == 1:
        return (dp.x1, dp.y1)
    if slot == 2:
        return (dp.u0, dp.v0)
    if slot == 3:
        return (dp.u1, dp.v1)
    if slot == 4:
        return (bitcast[DType.float32](dp.clip_xy0), bitcast[DType.float32](dp.clip_xy1))
    if slot == 5:
        return (bitcast[DType.float32](dp.fill_rgba8), bitcast[DType.float32](dp.secondary_rgba8))
    return (dp.param0, dp.param1)


@always_inline
def _sample_font_norm(font: SdfFont, tu: Float32, tv: Float32, is_msdf: Bool, is_mtsdf: Bool, for_true_dist: Bool) -> Float32:
    var ax = tu * Float32(font.atlas_width)
    var ay = tv * Float32(font.atlas_height)
    if for_true_dist and is_mtsdf and font.channels >= 4:
        return font.sample_channel(ax, ay, 3) * Float32(1.0 / 255.0)
    if is_msdf and font.channels >= 3:
        return font.sample_msdf(ax, ay) * Float32(1.0 / 255.0)
    return font.sample(ax, ay) * Float32(1.0 / 255.0)


def _raster_shape_fast_opaque(
    mut canvas: SimdCanvas,
    ix0: Int,
    iy0: Int,
    ix1: Int,
    iy1: Int,
    color: UInt32,
):
    """8-wide SIMD fast path for opaque unrounded, unstyled, borderless rectangles."""
    var opaque = color | UInt32(0xFF000000)
    var col_vec = SIMD[DType.uint32, 8](opaque)
    var px_addr = Int(canvas.pixels.unsafe_ptr())
    for py in range(iy0, iy1):
        var row = py * canvas.width
        var px = ix0
        var vec_end = ix0 + ((ix1 - ix0) // 8) * 8
        while px < vec_end:
            _simd8_store_u32(px_addr, row + px, col_vec)
            px += 8
        while px < ix1:
            canvas.pixels[row + px] = opaque
            px += 1


def rasterize_ui_draw_list_cpu(
    mut canvas: SimdCanvas,
    ui: UiDrawList,
    fonts: List[SdfFont],
    ui_time_sec: Float32 = 0.0,
    ui_global_flags: UInt32 = 0,
):
    """Evaluates all `GpuUiPrim` commands in `ui` onto `canvas` with Vulkan GPU SDF parity."""
    var w = canvas.width
    var h = canvas.height
    if w <= 0 or h <= 0 or ui.count() == 0:
        return

    var glass_enabled = (ui_global_flags & UInt32(1)) == 0
    var has_any_glass = False
    if glass_enabled:
        for i in range(ui.count()):
            var hdr = ui.prims[i].header
            if (hdr & UInt32(0xF)) == PRIM_SHAPE and (((hdr >> 16) & UI_FLAG_GLASS_SCRIM) != 0):
                has_any_glass = True
                break

    # Snapshot untouched scene pixels once when any frosted-glass scrim is present
    # (matches GPU's workgroup LDS capture of the initial background before UI compositing).
    var bg_snap = List[UInt32]()
    if has_any_glass:
        bg_snap = canvas.pixels.copy()

    var prim_cnt = ui.count()
    for p_i in range(prim_cnt):
        ref pr = ui.prims[p_i]
        var kind = pr.header & UInt32(0xF)
        if kind == PRIM_POLY_DATA:
            continue
        if not (isfinite(pr.x0) and isfinite(pr.y0) and isfinite(pr.x1) and isfinite(pr.y1)):
            continue

        var tex_slot = Int((pr.header >> 4) & UInt32(0xFFF))
        var flags = pr.header >> 16
        var p0 = pr.param0 if isfinite(pr.param0) else Float32(0.0)
        var p1 = pr.param1 if isfinite(pr.param1) else Float32(0.0)
        var p2 = pr.param2 if isfinite(pr.param2) else Float32(0.0)

        var cx0 = Int(pr.clip_xy0 & UInt32(0xFFFF))
        var cy0 = Int(pr.clip_xy0 >> 16)
        var cx1 = Int(pr.clip_xy1 & UInt32(0xFFFF))
        var cy1 = Int(pr.clip_xy1 >> 16)
        var cl_x0 = max(0, min(w, cx0))
        var cl_y0 = max(0, min(h, cy0))
        var cl_x1 = max(0, min(w, cx1))
        var cl_y1 = max(0, min(h, cy1))
        if cl_x0 >= cl_x1 or cl_y0 >= cl_y1:
            continue

        var box_pad = Float32(0.5)
        if kind == PRIM_LINE:
            box_pad = max(p0, Float32(1.0)) * Float32(0.5) + Float32(2.5)
            if (flags & UI_FLAG_DASHED_FLOW) != 0:
                box_pad += max(Float32(4.0), max(p0, p2) * Float32(3.0))
        elif kind == PRIM_ARC:
            box_pad = max(p1, Float32(1.0)) * Float32(0.5) + _clampf(p2, 0.0, 28.0) + Float32(2.5)

        var has_drop_shadow = (flags & UI_FLAG_DROP_SHADOW) != 0
        var has_outer_glow = (flags & UI_FLAG_OUTER_GLOW) != 0
        if has_drop_shadow or (kind == PRIM_GLYPH and p2 > Float32(0.01)):
            box_pad += max(p2, Float32(6.0)) + Float32(4.0)
        if has_outer_glow:
            box_pad += _clampf(p2, 2.0, 28.0) + Float32(2.0)
        if p1 > Float32(0.01):
            box_pad += min(p1, Float32(16.0))

        var bx0 = min(pr.x0, pr.x1) - box_pad
        var by0 = min(pr.y0, pr.y1) - box_pad
        var bx1 = max(pr.x0, pr.x1) + box_pad
        var by1 = max(pr.y0, pr.y1) + box_pad

        var ix0 = max(cl_x0, Int(floor(max(Float32(-1.0e6), bx0))))
        var iy0 = max(cl_y0, Int(floor(max(Float32(-1.0e6), by0))))
        var ix1 = min(cl_x1, Int(floor(min(Float32(1.0e6), bx1))) + 1)
        var iy1 = min(cl_y1, Int(floor(min(Float32(1.0e6), by1))) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue

        var fill = _unpack_argb_f32(pr.fill_rgba8)
        var sec = _unpack_argb_f32(pr.secondary_rgba8)
        var cr = fill[0]
        var cg = fill[1]
        var cb = fill[2]
        var ca = fill[3]
        var sr = sec[0]
        var sg = sec[1]
        var sb = sec[2]
        var sa = sec[3]

        # ---- Kind 0: PRIM_SHAPE ----
        if kind == PRIM_SHAPE:
            var is_circle = (flags & UI_FLAG_CIRCLE) != 0
            var has_border = ((flags & UI_FLAG_BORDER) != 0) and (p1 > Float32(0.01))
            var has_vgrad = (flags & UI_FLAG_VERT_GRADIENT) != 0
            var has_hgrad = (flags & UI_FLAG_HORIZ_GRADIENT) != 0
            var has_per_corner = (flags & UI_FLAG_PER_CORNER_RADII) != 0
            var has_top_bevel = (flags & UI_FLAG_TOP_BEVEL) != 0
            var has_glass_scrim = (flags & UI_FLAG_GLASS_SCRIM) != 0

            # Fast 8-wide SIMD path for plain opaque sharp rects
            if (
                flags == 0
                and p0 <= Float32(0.01)
                and (pr.fill_rgba8 >> 24) == 0xFF
            ):
                var fx0 = max(cl_x0, Int(floor(pr.x0 + Float32(0.5))))
                var fy0 = max(cl_y0, Int(floor(pr.y0 + Float32(0.5))))
                var fx1 = min(cl_x1, Int(floor(pr.x1 + Float32(0.5))))
                var fy1 = min(cl_y1, Int(floor(pr.y1 + Float32(0.5))))
                if fx0 < fx1 and fy0 < fy1:
                    _raster_shape_fast_opaque(canvas, fx0, fy0, fx1, fy1, pr.fill_rgba8)
                continue

            var w_half = max(Float32(0.5), pr.x1 - pr.x0) * Float32(0.5)
            var h_half = max(Float32(0.5), pr.y1 - pr.y0) * Float32(0.5)
            var cx = pr.x0 + w_half
            var cy = pr.y0 + h_half
            var inv_w = Float32(1.0) / max(Float32(1.0), pr.x1 - pr.x0)
            var inv_h = Float32(1.0) / max(Float32(1.0), pr.y1 - pr.y0)
            var r_blur = max(1, min(4, Int(p2 + Float32(0.5))))

            for py in range(iy0, iy1):
                var fgy = Float32(py) + Float32(0.5)
                var row = py * w
                var tile_y0 = (py // 16) * 16
                var tile_y1 = min(h - 1, tile_y0 + 15)
                for px in range(ix0, ix1):
                    var fgx = Float32(px) + Float32(0.5)
                    var eff_r = p0
                    if has_per_corner:
                        var r_top = pr.u0 if fgx < cx else pr.v0
                        var r_bot = pr.v1 if fgx < cx else pr.u1
                        eff_r = r_top if fgy < cy else r_bot

                    var d_shape: Float32
                    if is_circle:
                        var dx = fgx - cx
                        var dy = fgy - cy
                        d_shape = sqrt(dx * dx + dy * dy) - p0
                    else:
                        d_shape = _rounded_box_sdf(fgx, fgy, cx, cy, w_half, h_half, eff_r)
                    var cov_outer = _clampf(Float32(0.5) - d_shape, 0.0, 1.0)
                    var pidx = row + px

                    if has_drop_shadow:
                        var sh_x = fgx - Float32(2.0)
                        var sh_y = fgy - Float32(3.0)
                        var dist_sh: Float32
                        if is_circle:
                            var sdx = sh_x - cx
                            var sdy = sh_y - cy
                            dist_sh = sqrt(sdx * sdx + sdy * sdy) - p0
                        else:
                            var sh_r = p0
                            if has_per_corner:
                                var srt = pr.u0 if sh_x < cx else pr.v0
                                var srb = pr.v1 if sh_x < cx else pr.u1
                                sh_r = srt if sh_y < cy else srb
                            dist_sh = _rounded_box_sdf(sh_x, sh_y, cx, cy, w_half, h_half, sh_r)
                        var sh_soft = max(p2, Float32(5.0))
                        var sh_t = _clampf(Float32(1.0) - max(dist_sh, Float32(0.0)) / sh_soft, 0.0, 1.0)
                        var sh_mask = (Float32(1.0) - cov_outer) if ca < Float32(0.995) else Float32(1.0)
                        var sh_a = sh_t * sh_t * ca * Float32(0.45) * sh_mask
                        _blend_pixel(canvas.pixels, pidx, 0.0, 0.0, 0.0, sh_a)

                    if has_outer_glow:
                        var g_soft = _clampf(p2, 1.5, 28.0)
                        var g_u = max(d_shape, Float32(0.0)) / g_soft
                        var g_fall = exp2(Float32(-2.4) * g_u * g_u) * (Float32(1.0) - cov_outer)
                        var use_sec = sa > Float32(0.001)
                        _blend_pixel(
                            canvas.pixels,
                            pidx,
                            sr if use_sec else cr,
                            sg if use_sec else cg,
                            sb if use_sec else cb,
                            (sa if use_sec else ca) * g_fall,
                        )

                    if has_glass_scrim and glass_enabled and cov_outer > Float32(0.001) and len(bg_snap) == w * h:
                        var tile_x0 = (px // 16) * 16
                        var tile_x1 = min(w - 1, tile_x0 + 15)
                        var x_lo = max(tile_x0, px - r_blur)
                        var x_hi = min(tile_x1, px + r_blur)
                        var y_lo = max(tile_y0, py - r_blur)
                        var y_hi = min(tile_y1, py + r_blur)
                        var c0 = _unpack_argb_f32(bg_snap[pidx])
                        var c1 = _unpack_argb_f32(bg_snap[y_lo * w + x_lo])
                        var c2 = _unpack_argb_f32(bg_snap[y_lo * w + x_hi])
                        var c3 = _unpack_argb_f32(bg_snap[y_hi * w + x_lo])
                        var c4 = _unpack_argb_f32(bg_snap[y_hi * w + x_hi])
                        var br = c0[0] * Float32(0.28) + (c1[0] + c2[0] + c3[0] + c4[0]) * Float32(0.18)
                        var bg = c0[1] * Float32(0.28) + (c1[1] + c2[1] + c3[1] + c4[1]) * Float32(0.18)
                        var bb = c0[2] * Float32(0.28) + (c1[2] + c2[2] + c3[2] + c4[2]) * Float32(0.18)
                        _blend_pixel(canvas.pixels, pidx, br, bg, bb, cov_outer)

                    if cov_outer <= Float32(0.0):
                        continue

                    var fr = cr
                    var fg = cg
                    var fb = cb
                    var fa = ca
                    if has_vgrad or has_hgrad:
                        var ty = _clampf((fgy - pr.y0) * inv_h, 0.0, 1.0)
                        var tx = _clampf((fgx - pr.x0) * inv_w, 0.0, 1.0)
                        var tg = (tx + ty) * Float32(0.5) if (has_vgrad and has_hgrad) else (tx if has_hgrad else ty)
                        if (not has_border) and sa > Float32(0.001):
                            fr = cr + (sr - cr) * tg
                            fg = cg + (sg - cg) * tg
                            fb = cb + (sb - cb) * tg
                            fa = ca + (sa - ca) * tg
                        else:
                            var sm = Float32(1.12) - Float32(0.32) * tg
                            fr = _clampf(cr * sm, 0.0, 1.0)
                            fg = _clampf(cg * sm, 0.0, 1.0)
                            fb = _clampf(cb * sm, 0.0, 1.0)

                    if has_border:
                        var cov_in = _clampf(Float32(0.5) - (d_shape + p1), 0.0, 1.0)
                        _blend_pixel(canvas.pixels, pidx, fr, fg, fb, fa * cov_in)
                        var brd_cov = max(Float32(0.0), cov_outer - cov_in)
                        _blend_pixel(canvas.pixels, pidx, sr, sg, sb, sa * brd_cov)
                    else:
                        _blend_pixel(canvas.pixels, pidx, fr, fg, fb, fa * cov_outer)

                    if has_top_bevel:
                        var bev_inset = (p1 if has_border else Float32(0.0)) + Float32(0.5)
                        var inner_band = _clampf(Float32(0.5) - abs(d_shape + bev_inset), 0.0, 1.0)
                        var top_mask = _clampf(Float32(1.0) - (fgy - pr.y0) / max(Float32(3.0), eff_r + Float32(2.0)), 0.0, 1.0)
                        var bev_a = Float32(0.09) * inner_band * top_mask * cov_outer * ca
                        _blend_pixel(canvas.pixels, pidx, 1.0, 1.0, 1.0, bev_a)

        # ---- Kind 1: PRIM_GLYPH ----
        elif kind == PRIM_GLYPH:
            if tex_slot < 0 or tex_slot >= len(fonts) or not fonts[tex_slot].valid:
                continue
            ref font = fonts[tex_slot]
            var gw = max(Float32(1.0), pr.x1 - pr.x0)
            var gh = max(Float32(1.0), pr.y1 - pr.y0)
            var du = pr.u1 - pr.u0
            var dv = pr.v1 - pr.v0
            var is_msdf = (flags & UI_FLAG_MSDF) != 0
            var is_mtsdf = (flags & UI_FLAG_MTSDF) != 0
            var has_glyph_sh = has_drop_shadow or p2 > Float32(0.01)
            var eff_sdf_scale = p0 if p0 >= Float32(0.5) else max(Float32(1.5), p0 * Float32(255.0))

            for py in range(iy0, iy1):
                var fgy = Float32(py) + Float32(0.5)
                var row = py * w
                var tv = pr.v0 + ((fgy - pr.y0) / gh) * dv
                for px in range(ix0, ix1):
                    var fgx = Float32(px) + Float32(0.5)
                    var tu = pr.u0 + ((fgx - pr.x0) / gw) * du
                    var pidx = row + px

                    if has_glyph_sh:
                        var sh_off = max(p2, Float32(1.5))
                        var stu = _clampf(tu - (sh_off / gw) * du, 0.0, 1.0)
                        var stv = _clampf(tv - (sh_off / gh) * dv, 0.0, 1.0)
                        var sh_d = _sample_font_norm(font, stu, stv, is_msdf, is_mtsdf, True)
                        var sh_cov = _clampf((sh_d - Float32(0.5)) * eff_sdf_scale + Float32(0.5), 0.0, 1.0)
                        _blend_pixel(canvas.pixels, pidx, 0.0, 0.0, 0.0, ca * sh_cov * Float32(0.60))

                    if fgx >= pr.x0 and fgx <= pr.x1 and fgy >= pr.y0 and fgy <= pr.y1:
                        var d_norm = _sample_font_norm(font, tu, tv, is_msdf, is_mtsdf, False)
                        var sd = (d_norm - Float32(0.5)) * eff_sdf_scale
                        var cov = _clampf(sd + Float32(0.5), 0.0, 1.0)
                        if p1 > Float32(0.01) and sa > Float32(0.001):
                            var sd_str = (
                                (_sample_font_norm(font, tu, tv, is_msdf, is_mtsdf, True) - Float32(0.5)) * eff_sdf_scale
                                if is_mtsdf
                                else sd
                            )
                            var cov_out = _clampf(sd_str + p1 + Float32(0.5), 0.0, 1.0)
                            _blend_pixel(canvas.pixels, pidx, sr, sg, sb, sa * max(Float32(0.0), cov_out - cov))
                        _blend_pixel(canvas.pixels, pidx, cr, cg, cb, ca * cov)

        # ---- Kind 3: PRIM_CONTROLLER_SYMBOL ----
        elif kind == PRIM_CONTROLLER_SYMBOL:
            var sw = max(Float32(2.0), pr.x1 - pr.x0)
            var sh = max(Float32(2.0), pr.y1 - pr.y0)
            var scx = pr.x0 + sw * Float32(0.5)
            var scy = pr.y0 + sh * Float32(0.5)
            var r_half = min(sw, sh) * Float32(0.5)
            var inv_rh = Float32(1.0) / r_half
            var st_w = p1 if (p1 > Float32(0.01) and p1 < Float32(0.45)) else Float32(0.11)
            for py in range(iy0, iy1):
                var fgy = Float32(py) + Float32(0.5)
                var ny = (fgy - scy) * inv_rh
                var row = py * w
                for px in range(ix0, ix1):
                    var fgx = Float32(px) + Float32(0.5)
                    var nx = (fgx - scx) * inv_rh
                    var pidx = row + px
                    if sa > Float32(0.001):
                        var r_len = sqrt(nx * nx + ny * ny)
                        var b_cov = _clampf(Float32(0.5) - (r_len - Float32(0.92)) * r_half, 0.0, 1.0)
                        _blend_pixel(canvas.pixels, pidx, sr, sg, sb, sa * b_cov)
                    var d_sym: Float32
                    if tex_slot == 0:
                        d_sym = min(
                            _seg_dist(nx, ny, -0.36, 0.44, 0.0, -0.44),
                            min(_seg_dist(nx, ny, 0.0, -0.44, 0.36, 0.44), _seg_dist(nx, ny, -0.22, 0.14, 0.22, 0.14)),
                        ) - st_w
                    elif tex_slot == 1:
                        var db = min(
                            _seg_dist(nx, ny, -0.26, -0.44, -0.26, 0.44),
                            min(
                                _seg_dist(nx, ny, -0.26, -0.44, 0.06, -0.44),
                                min(_seg_dist(nx, ny, -0.26, 0.0, 0.08, 0.0), _seg_dist(nx, ny, -0.26, 0.44, 0.08, 0.44)),
                            ),
                        )
                        var lob_u = abs(sqrt((nx - 0.06) * (nx - 0.06) + (ny + 0.22) * (ny + 0.22)) - Float32(0.22))
                        var lob_l = abs(sqrt((nx - 0.08) * (nx - 0.08) + (ny - 0.22) * (ny - 0.22)) - Float32(0.22))
                        var lob = min(lob_u, lob_l) if nx >= Float32(0.04) else Float32(10.0)
                        d_sym = min(db, lob) - st_w
                    elif tex_slot == 2 or tex_slot == 4:
                        d_sym = min(_seg_dist(nx, ny, -0.38, -0.38, 0.38, 0.38), _seg_dist(nx, ny, -0.38, 0.38, 0.38, -0.38)) - st_w
                    elif tex_slot == 3:
                        d_sym = min(
                            _seg_dist(nx, ny, -0.36, -0.42, 0.0, 0.02),
                            min(_seg_dist(nx, ny, 0.36, -0.42, 0.0, 0.02), _seg_dist(nx, ny, 0.0, 0.02, 0.0, 0.44)),
                        ) - st_w
                    elif tex_slot == 5:
                        d_sym = abs(sqrt(nx * nx + ny * ny) - Float32(0.42)) - st_w
                    elif tex_slot == 6:
                        d_sym = min(
                            _seg_dist(nx, ny, 0.0, -0.44, 0.40, 0.30),
                            min(_seg_dist(nx, ny, 0.40, 0.30, -0.40, 0.30), _seg_dist(nx, ny, -0.40, 0.30, 0.0, -0.44)),
                        ) - st_w
                    elif tex_slot == 7:
                        d_sym = abs(max(abs(nx), abs(ny)) - Float32(0.38)) - st_w
                    elif tex_slot == 8:
                        var e_l = Float32(-0.30) - nx
                        var e_t = (Float32(0.42) * nx + Float32(0.72) * ny - Float32(0.1764)) * Float32(1.1997)
                        var e_b = (Float32(0.42) * nx - Float32(0.72) * ny - Float32(0.1764)) * Float32(1.1997)
                        var inside = max(e_l, max(e_t, e_b)) < Float32(0.0)
                        var ed = min(
                            _seg_dist(nx, ny, -0.30, -0.42, -0.30, 0.42),
                            min(_seg_dist(nx, ny, -0.30, -0.42, 0.42, 0.0), _seg_dist(nx, ny, -0.30, 0.42, 0.42, 0.0)),
                        )
                        d_sym = -ed if inside else ed
                    elif tex_slot == 9:
                        d_sym = _rounded_box_sdf(abs(abs(nx) - Float32(0.22)), abs(ny), 0.0, 0.0, 0.10, 0.40, 0.04)
                    else:
                        var rg = sqrt(nx * nx + ny * ny)
                        var d_r = abs(rg - Float32(0.36)) - st_w
                        var d_h = _seg_dist(abs(nx), ny, 0.22, 0.0, 0.56, 0.0)
                        var d_v = _seg_dist(nx, abs(ny), 0.0, 0.22, 0.0, 0.56)
                        d_sym = min(d_r, min(rg - st_w, min(d_h, d_v) - st_w * Float32(0.85)))
                    var cov_sym = _clampf(Float32(0.5) - d_sym * r_half, 0.0, 1.0)
                    _blend_pixel(canvas.pixels, pidx, cr, cg, cb, ca * cov_sym)

        # ---- Kind 4: PRIM_LINE (Capsule + 12-chord Quadratic Bézier + Dashed Flow) ----
        elif kind == PRIM_LINE:
            var is_bez = (flags & UI_FLAG_QUAD_BEZIER) != 0
            var is_flow = (flags & UI_FLAG_DASHED_FLOW) != 0
            var uv_def = (
                abs(pr.u0) < Float32(1.0e-5)
                and abs(pr.v0) < Float32(1.0e-5)
                and abs(pr.u1 - Float32(1.0)) < Float32(1.0e-5)
                and abs(pr.v1 - Float32(1.0)) < Float32(1.0e-5)
            )
            var half_t = max(p2, Float32(0.5)) if is_bez else max(p0, Float32(0.5))
            var feather = Float32(1.0) if is_bez else max(p2, Float32(0.5))
            var lax = pr.x0 if uv_def else pr.u0
            var lay = pr.y0 if uv_def else pr.v0
            var lbx = pr.x1 if uv_def else pr.u1
            var lby = pr.y1 if uv_def else pr.v1
            if not (isfinite(lax) and isfinite(lay) and isfinite(lbx) and isfinite(lby)):
                continue

            var line_ix0 = ix0
            var line_iy0 = iy0
            var line_ix1 = ix1
            var line_iy1 = iy1
            if not is_bez:
                # Float64 Liang-Barsky clip against padded clip box so extreme endpoints (±1e12+) never overflow Float32
                var cpad = Float64(box_pad) + 2.0
                var xmin = Float64(cl_x0) - cpad
                var xmax = Float64(cl_x1) + cpad
                var ymin = Float64(cl_y0) - cpad
                var ymax = Float64(cl_y1) + cpad
                var u0_64 = Float64(lax)
                var v0_64 = Float64(lay)
                var dx64 = Float64(lbx) - u0_64
                var dy64 = Float64(lby) - v0_64
                var t0 = 0.0
                var t1 = 1.0
                var p_edges = [-dx64, dx64, -dy64, dy64]
                var q_edges = [u0_64 - xmin, xmax - u0_64, v0_64 - ymin, ymax - v0_64]
                var visible = True
                for ei in range(4):
                    var pe = p_edges[ei]
                    var qe = q_edges[ei]
                    if pe == 0.0:
                        if qe < 0.0:
                            visible = False
                            break
                    else:
                        var rv = qe / pe
                        if pe < 0.0:
                            if rv > t1:
                                visible = False
                                break
                            if rv > t0:
                                t0 = rv
                        else:
                            if rv < t0:
                                visible = False
                                break
                            if rv < t1:
                                t1 = rv
                if not visible or t0 > t1:
                    continue
                var cl_ax = u0_64 + t0 * dx64
                var cl_ay = v0_64 + t0 * dy64
                var cl_bx = u0_64 + t1 * dx64
                var cl_by = v0_64 + t1 * dy64
                lax = Float32(cl_ax)
                lay = Float32(cl_ay)
                lbx = Float32(cl_bx)
                lby = Float32(cl_by)
                line_ix0 = max(cl_x0, Int(floor(min(lax, lbx) - box_pad)))
                line_iy0 = max(cl_y0, Int(floor(min(lay, lby) - box_pad)))
                line_ix1 = min(cl_x1, Int(floor(max(lax, lbx) + box_pad)) + 1)
                line_iy1 = min(cl_y1, Int(floor(max(lay, lby) + box_pad)) + 1)
                if line_ix0 >= line_ix1 or line_iy0 >= line_iy1:
                    continue

            var f_dash = max(Float32(2.0), Float32(pr.secondary_rgba8 & UInt32(0xFF)))
            var f_gap = Float32((pr.secondary_rgba8 >> 8) & UInt32(0xFF))
            var f_spd = Float32((pr.secondary_rgba8 >> 16) & UInt32(0xFF)) * Float32(4.0)
            var f_glw = Float32((pr.secondary_rgba8 >> 24) & UInt32(0xFF)) * Float32(1.0 / 255.0)
            var f_per = max(Float32(2.0), f_dash + f_gap)
            var sample_off = Float32(0.0) if ((not is_bez) and (not is_flow) and half_t <= Float32(0.55)) else Float32(0.5)

            for py in range(line_iy0, line_iy1):
                var fgy = Float32(py) + sample_off
                var row = py * w
                for px in range(line_ix0, line_ix1):
                    var fgx = Float32(px) + sample_off
                    var d_min = Float32(1.0e6)
                    var s_along = Float32(0.0)
                    if is_bez:
                        var prev_x = pr.u0
                        var prev_y = pr.v0
                        var cum_len = Float32(0.0)
                        for si in range(1, 13):
                            var tb = Float32(si) * Float32(1.0 / 12.0)
                            var omt = Float32(1.0) - tb
                            var cur_x = omt * omt * pr.u0 + Float32(2.0) * omt * tb * p0 + tb * tb * pr.u1
                            var cur_y = omt * omt * pr.v0 + Float32(2.0) * omt * tb * p1 + tb * tb * pr.v1
                            var d_ch = _seg_dist(fgx, fgy, prev_x, prev_y, cur_x, cur_y)
                            var cdx = cur_x - prev_x
                            var cdy = cur_y - prev_y
                            var clen = sqrt(cdx * cdx + cdy * cdy)
                            if d_ch < d_min:
                                d_min = d_ch
                                if is_flow:
                                    var c_dot = max(Float32(1.0e-6), cdx * cdx + cdy * cdy)
                                    var ch = _clampf(((fgx - prev_x) * cdx + (fgy - prev_y) * cdy) / c_dot, 0.0, 1.0)
                                    s_along = cum_len + ch * clen
                            cum_len += clen
                            prev_x = cur_x
                            prev_y = cur_y
                    else:
                        d_min = _seg_dist(fgx, fgy, lax, lay, lbx, lby)
                        if is_flow:
                            var ldx = lbx - lax
                            var ldy = lby - lay
                            var l_sq = max(Float32(1.0e-6), ldx * ldx + ldy * ldy)
                            var lh = _clampf(((fgx - lax) * ldx + (fgy - lay) * ldy) / l_sq, 0.0, 1.0)
                            s_along = lh * sqrt(l_sq)

                    var sd_l = d_min - half_t
                    var cov_l = _clampf(Float32(0.5) - sd_l / feather, 0.0, 1.0)
                    var pidx = row + px
                    if is_flow:
                        var cyc = (s_along - ui_time_sec * f_spd) / f_per
                        var ph = (cyc - floor(cyc)) * f_per
                        var mask = _clampf(min(ph, f_dash - ph) + Float32(0.5), 0.0, 1.0) if f_gap > Float32(0.5) else Float32(1.0)
                        if f_glw > Float32(0.01):
                            var lg_r = max(Float32(2.0), half_t * Float32(2.5))
                            var lg_u = max(sd_l, Float32(0.0)) / lg_r
                            var lg_a = exp2(Float32(-1.8) * lg_u * lg_u) * f_glw * ca * (Float32(0.35) + Float32(0.65) * mask)
                            _blend_pixel(canvas.pixels, pidx, cr, cg, cb, lg_a)
                        _blend_pixel(canvas.pixels, pidx, cr, cg, cb, ca * cov_l * mask)
                    else:
                        _blend_pixel(canvas.pixels, pidx, cr, cg, cb, ca * cov_l)

        # ---- Kind 6: PRIM_POLYGON ----
        elif kind == PRIM_POLYGON:
            var first_data = p_i + 1
            var avail = max(0, prim_cnt - first_data) * POLY_VERTS_PER_PRIM
            var pg_n = min(Int(max(Float32(0.0), pr.u0)), avail)
            if pg_n < 4:
                continue
            for py in range(iy0, iy1):
                var fgy = Float32(py) + Float32(0.5)
                var pg_py = fgy + Float32(0.7e-3)
                var row = py * w
                for px in range(ix0, ix1):
                    var fgx = Float32(px) + Float32(0.5)
                    var pg_px = fgx + Float32(1.3e-3)
                    var wn = Float32(0.0)
                    var dmin = Float32(1.0e6)
                    for pass_i in range(2):
                        var wn_f = wn
                        var v0 = _poly_vert(ui.prims, first_data, 0)
                        var eax = v0[0]
                        var eay = v0[1]
                        for vk in range(1, pg_n):
                            var vb = _poly_vert(ui.prims, first_data, vk)
                            var ebx = vb[0]
                            var eby = vb[1]
                            if eax < Float32(1.0e37) and ebx < Float32(1.0e37):
                                var is_left = (ebx - eax) * (pg_py - eay) - (pg_px - eax) * (eby - eay)
                                if pass_i == 0:
                                    if eay <= pg_py and eby > pg_py and is_left > Float32(0.0):
                                        wn += Float32(1.0)
                                    elif eay > pg_py and eby <= pg_py and is_left < Float32(0.0):
                                        wn -= Float32(1.0)
                                else:
                                    var wn_other = wn_f + (Float32(-1.0) if is_left > Float32(0.0) else Float32(1.0))
                                    var in_h = abs(wn_f) > Float32(0.5)
                                    var in_t = abs(wn_other) > Float32(0.5)
                                    if in_h != in_t:
                                        dmin = min(dmin, _seg_dist(fgx, fgy, eax, eay, ebx, eby))
                            eax = ebx
                            eay = eby
                    var inside = abs(wn) > Float32(0.5)
                    var sd_pg = -dmin if inside else dmin
                    var cov_pg = _clampf(Float32(0.5) - sd_pg, 0.0, 1.0)
                    _blend_pixel(canvas.pixels, row + px, cr, cg, cb, ca * cov_pg)

        # ---- Kind 8: PRIM_ARC ----
        elif kind == PRIM_ARC:
            var arc_cx = pr.u0
            var arc_cy = pr.v0
            var arc_start_raw = pr.u1
            var arc_sweep = pr.v1
            var arc_r = max(Float32(0.5), p0)
            var arc_half = max(Float32(0.5), p1) * Float32(0.5)
            var has_track = ((flags & UI_FLAG_BORDER) != 0) and (sa > Float32(0.001))
            var has_spin = (flags & UI_FLAG_DASHED_FLOW) != 0
            var sw_abs = abs(arc_sweep)
            var eff_start = arc_start_raw + (ui_time_sec * Float32(3.0) if has_spin else Float32(0.0))
            var st_ang = eff_start if arc_sweep >= Float32(0.0) else eff_start + arc_sweep
            var end_ang = st_ang + sw_abs
            var sx0 = arc_cx + arc_r * cos(st_ang)
            var sy0 = arc_cy + arc_r * sin(st_ang)
            var sx1 = arc_cx + arc_r * cos(end_ang)
            var sy1 = arc_cy + arc_r * sin(end_ang)
            var two_pi = Float32(6.2831853)

            for py in range(iy0, iy1):
                var fgy = Float32(py) + Float32(0.5)
                var ady = fgy - arc_cy
                var row = py * w
                for px in range(ix0, ix1):
                    var fgx = Float32(px) + Float32(0.5)
                    var adx = fgx - arc_cx
                    var dist_c = sqrt(adx * adx + ady * ady)
                    var d_ring = abs(dist_c - arc_r) - arc_half
                    var pidx = row + px

                    if has_track:
                        var cov_tr = _clampf(Float32(0.5) - d_ring, 0.0, 1.0)
                        _blend_pixel(canvas.pixels, pidx, sr, sg, sb, sa * cov_tr)

                    if sw_abs > Float32(1.0e-4):
                        var d_arc = d_ring
                        if sw_abs < Float32(6.27):
                            var ang = atan2(ady, adx)
                            var rel_raw = ang - st_ang
                            var rel = rel_raw - two_pi * floor(rel_raw / two_pi)
                            if rel > sw_abs:
                                var dc0 = sqrt((fgx - sx0) * (fgx - sx0) + (fgy - sy0) * (fgy - sy0))
                                var dc1 = sqrt((fgx - sx1) * (fgx - sx1) + (fgy - sy1) * (fgy - sy1))
                                d_arc = min(dc0, dc1) - arc_half
                        var cov_arc = _clampf(Float32(0.5) - d_arc, 0.0, 1.0)
                        if has_outer_glow and p2 > Float32(0.01):
                            var ag_soft = _clampf(p2, 1.5, 28.0)
                            var ag_u = max(d_arc, Float32(0.0)) / ag_soft
                            var ag_a = exp2(Float32(-2.4) * ag_u * ag_u) * (Float32(1.0) - cov_arc) * ca * Float32(0.65)
                            _blend_pixel(canvas.pixels, pidx, cr, cg, cb, ag_a)
                        _blend_pixel(canvas.pixels, pidx, cr, cg, cb, ca * cov_arc)
