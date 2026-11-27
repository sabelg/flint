"""Flint v5 Gameplay HUD Overlay Helpers (`flint.ui.hud`).

Provides AAA minimal frosted-glass screen-space HUD primitives (`game_hud`, `hud_bar`)
that emit directly into `UiContext.draw_list` and work safely both inside and outside
an active `UiContext.begin_frame` / `end_frame` scope (preserving existing
`draw_list.prims` and never indexing an empty `box_stack`).
"""
from std.collections import List
from std.math import isfinite, max, min
from flint.ui.context import UiContext
from flint.ui.draw_list import UI_FLAG_BORDER, UI_FLAG_TOP_BEVEL
from flint.ui.theme import rgba8, rgba8_with_alpha


def _format_hud_int(val: Int) -> String:
    """Formats an integer with thousands separators (`1250` -> `1,250`)."""
    var neg = val < 0
    var mag = -val if neg else val
    var raw = String(mag)
    var n = raw.byte_length()
    if n <= 3:
        return ("-" + raw) if neg else raw
    var bytes = raw.as_bytes()
    var out = List[UInt8](capacity=n + (n - 1) // 3 + (1 if neg else 0))
    if neg:
        out.append(UInt8(45))  # '-'
    for i in range(n):
        if i > 0 and ((n - i) % 3) == 0:
            out.append(UInt8(44))  # ','
        out.append(bytes[i])
    return String(unsafe_from_utf8=out)


def hud_bar(
    mut ctx: UiContext,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    value: Int,
    max_value: Int,
    fill_rgba: UInt32 = 0xFF2DD4BF,
):
    """Draws a refined segmented/ticked progress bar with specular leading edge directly onto `ctx.draw_list`."""
    var safe_w = max(Float32(4.0), w) if isfinite(w) else Float32(180.0)
    var safe_h = max(Float32(4.0), h) if isfinite(h) else Float32(9.0)
    var safe_x = x if isfinite(x) else Float32(16.0)
    var safe_y = y if isfinite(y) else Float32(16.0)
    var safe_max = max(1, max_value)
    var safe_val = max(0, min(safe_max, value))
    var ratio = Float32(safe_val) / Float32(safe_max)
    var track_r = min(Float32(4.5), safe_h * Float32(0.5))

    # Recessed obsidian track with subtle hairline rim
    ctx.draw_list.add_recessed_well(
        safe_x,
        safe_y,
        safe_w,
        safe_h,
        rgba8(9, 12, 17, 198),
        rgba8(255, 255, 255, 28),
        track_r,
        Float32(1.0),
    )

    var inset = Float32(1.5) if safe_h >= Float32(7.0) else Float32(1.0)
    var inner_h = max(Float32(2.0), safe_h - inset * Float32(2.0))
    var inner_max_w = max(Float32(0.0), safe_w - inset * Float32(2.0))
    var inner_w = inner_max_w * ratio
    var fill_r = min(Float32(3.5), inner_h * Float32(0.5))

    if inner_w > Float32(0.5):
        var fx = safe_x + inset
        var fy = safe_y + inset
        # Subtle low-HP urgency glow when <= 25%
        if ratio <= Float32(0.25):
            ctx.draw_list.add_glow_rect(
                fx,
                fy,
                inner_w,
                inner_h,
                UInt32(0),
                rgba8_with_alpha(fill_rgba, 125),
                fill_r,
                Float32(4.0),
            )
        ctx.draw_list.add_rect_styled(
            fx,
            fy,
            inner_w,
            inner_h,
            fill_rgba,
            UInt32(0),
            fill_r,
            Float32(0.0),
            Float32(1.0),
            UI_FLAG_TOP_BEVEL,
        )
        # Crisp bright leading-edge specular tip
        if ratio > Float32(0.04) and ratio < Float32(0.99) and inner_w >= Float32(4.0):
            var tip_x = fx + inner_w - Float32(2.0)
            ctx.draw_list.add_rect(
                tip_x,
                fy + Float32(0.5),
                Float32(2.0),
                max(Float32(1.5), inner_h - Float32(1.0)),
                rgba8(255, 255, 255, 132),
                Float32(1.0),
            )

    # Subtle 25% / 50% / 75% quarter-interval notch ticks for instant readability
    if safe_w >= Float32(64.0) and safe_h >= Float32(6.0):
        var tick_col = rgba8(9, 12, 17, 108)
        for q in range(1, 4):
            var tx = safe_x + safe_w * (Float32(q) * Float32(0.25))
            ctx.draw_list.add_line(
                tx,
                safe_y + Float32(1.5),
                tx,
                safe_y + safe_h - Float32(1.5),
                Float32(1.0),
                tick_col,
            )


def _draw_hud_keycap(
    mut ctx: UiContext,
    key_txt: String,
    x: Float32,
    cy: Float32,
    size_px: Float32,
) -> Float32:
    """Draws a compact translucent keycap micro-badge and returns its total width."""
    var tw = ctx._measure_text(key_txt, size_px)
    var kw = max(Float32(20.0), tw + Float32(10.0))
    var kh = Float32(15.0)
    var ky = cy - kh * Float32(0.5)
    ctx.draw_list.add_rect_styled(
        x,
        ky,
        kw,
        kh,
        rgba8(30, 36, 46, 210),
        rgba8(255, 255, 255, 42),
        Float32(3.5),
        Float32(1.0),
        Float32(1.0),
        UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
    )
    var base_y = cy + size_px * Float32(0.36)
    ctx._draw_text(
        key_txt,
        x + (kw - tw) * Float32(0.5),
        base_y - size_px * Float32(0.82),
        size_px,
        rgba8(226, 232, 240, 230),
    )
    return kw


def game_hud(
    mut ctx: UiContext,
    hp: Int,
    max_hp: Int,
    score: Int = 0,
    vp_x: Float32 = 0.0,
    vp_y: Float32 = 0.0,
    vp_w: Float32 = 0.0,
    vp_h: Float32 = 0.0,
    show_controls_hint: Bool = False,
) -> Int:
    """Emits an AAA minimal frosted-glass gameplay HUD overlay (HP telemetry + right-aligned Score +
    refined segmented bar + optional Playtest controls micro-hint) and returns added primitive count.

    Safe to call both inside an active `UiContext` frame and post-`end_frame` / headless
    without clearing existing `ctx.draw_list.prims`.
    """
    var base_w = ctx.screen_w if (isfinite(ctx.screen_w) and ctx.screen_w > Float32(0.0)) else Float32(1280.0)
    var base_h = ctx.screen_h if (isfinite(ctx.screen_h) and ctx.screen_h > Float32(0.0)) else Float32(720.0)
    var use_vp = isfinite(vp_w) and isfinite(vp_h) and vp_w > Float32(0.0) and vp_h > Float32(0.0)
    var ox = (vp_x if isfinite(vp_x) else Float32(0.0)) if use_vp else Float32(0.0)
    var oy = (vp_y if isfinite(vp_y) else Float32(0.0)) if use_vp else Float32(0.0)
    var sw = vp_w if use_vp else base_w
    var sh = vp_h if use_vp else base_h

    var safe_max = max(1, max_hp)
    var safe_hp = max(0, min(safe_max, hp))
    var ratio = Float32(safe_hp) / Float32(safe_max)

    # Smooth semantic ramp matching Flint Obsidian tokens (emerald -> amber -> crimson)
    var bar_color = UInt32(0xFF2DD4BF)
    if ratio <= Float32(0.25):
        bar_color = UInt32(0xFFF06A72)
    elif ratio <= Float32(0.55):
        bar_color = UInt32(0xFFF5B74F)

    var start_prims = len(ctx.draw_list.prims)

    var pad = Float32(14.0)
    var panel_w = min(Float32(236.0), max(Float32(168.0), sw - Float32(2.0) * pad))
    var panel_h = Float32(52.0)
    var px = ox + pad
    var py = oy + pad

    # Soft ambient shadow + smoky frosted obsidian glass + neutral hairline + 2.5px left semantic bar
    ctx.draw_list.add_ambient_shadow(
        px,
        py,
        panel_w,
        panel_h,
        Float32(8.0),
        Float32(12.0),
        120,
        Float32(2.0),
    )
    ctx.draw_list.add_glass_panel(
        px,
        py,
        panel_w,
        panel_h,
        rgba8(18, 22, 29, 204),
        Float32(8.0),
        Float32(4.0),
        Float32(1.0),
        rgba8(255, 255, 255, 42),
        True,
        False,
    )
    ctx.draw_list.add_accent_bar(
        px + Float32(3.5),
        py + Float32(9.0),
        Float32(2.5),
        panel_h - Float32(18.0),
        bar_color,
        True,
    )

    # Shared typographic baseline for top telemetry row
    var base_y = py + Float32(24.0)
    var lbl_sz = Float32(9.5)
    var val_sz = Float32(15.5)
    var max_sz = Float32(10.5)
    var score_sz = Float32(13.5)

    var col_primary = rgba8(238, 242, 247, 255)
    var col_secondary = rgba8(169, 180, 196, 255)
    var col_muted = rgba8(132, 145, 163, 255)
    var col_ember = rgba8(245, 155, 76, 255)

    # Left: HP micro-label + prominent numeric readout + muted denominator
    var lx = px + Float32(13.0)
    var hp_tag = String("HP")
    ctx._draw_text(hp_tag, lx, base_y - lbl_sz * Float32(0.82), lbl_sz, col_muted)
    lx += ctx._measure_text(hp_tag, lbl_sz) + Float32(5.0)

    var hp_val = String(safe_hp)
    ctx._draw_text(hp_val, lx, base_y - val_sz * Float32(0.82), val_sz, col_primary)
    lx += ctx._measure_text(hp_val, val_sz) + Float32(4.0)

    var hp_den = "/" + String(safe_max)
    ctx._draw_text(hp_den, lx, base_y - max_sz * Float32(0.82), max_sz, col_secondary)

    # Right: flush right-aligned SCORE micro-label + Flint Ember formatted readout
    var score_val = _format_hud_int(score)
    var score_val_w = ctx._measure_text(score_val, score_sz)
    var score_tag = String("SCORE")
    var score_tag_w = ctx._measure_text(score_tag, lbl_sz)
    var rx = px + panel_w - Float32(12.0) - score_val_w
    ctx._draw_text(score_val, rx, base_y - score_sz * Float32(0.82), score_sz, col_ember)
    var tag_x = rx - score_tag_w - Float32(5.0)
    if tag_x > lx + Float32(6.0):
        ctx._draw_text(score_tag, tag_x, base_y - lbl_sz * Float32(0.82), lbl_sz, col_muted)

    # Refined 9px segmented HP bar
    hud_bar(
        ctx,
        px + Float32(13.0),
        py + Float32(33.0),
        panel_w - Float32(25.0),
        Float32(9.0),
        safe_hp,
        safe_max,
        bar_color,
    )

    # Optional ultra-subtle bottom-center Playtest controls micro-hint pill
    if show_controls_hint and sw >= Float32(360.0) and sh >= Float32(150.0):
        var k_sz = Float32(9.5)
        var t_sz = Float32(10.0)
        var lbl1 = String("Pause")
        var sep = String("·")
        var lbl2 = String("Eject")
        var k1_w = max(Float32(20.0), ctx._measure_text("ESC", k_sz) + Float32(10.0))
        var t1_w = ctx._measure_text(lbl1, t_sz)
        var sep_w = ctx._measure_text(sep, t_sz)
        var k2_w = max(Float32(20.0), ctx._measure_text("F8", k_sz) + Float32(10.0))
        var t2_w = ctx._measure_text(lbl2, t_sz)
        var hint_w = Float32(16.0) + k1_w + Float32(5.0) + t1_w + Float32(9.0) + sep_w + Float32(9.0) + k2_w + Float32(5.0) + t2_w
        var hint_h = Float32(23.0)
        var hx = ox + (sw - hint_w) * Float32(0.5)
        var hy = oy + sh - hint_h - Float32(12.0)
        var hcy = hy + hint_h * Float32(0.5)

        ctx.draw_list.add_ambient_shadow(hx, hy, hint_w, hint_h, Float32(11.5), Float32(8.0), 85, Float32(1.5))
        ctx.draw_list.add_glass_panel(
            hx,
            hy,
            hint_w,
            hint_h,
            rgba8(18, 22, 29, 180),
            Float32(11.5),
            Float32(3.5),
            Float32(1.0),
            rgba8(255, 255, 255, 32),
            True,
            False,
        )
        var cx = hx + Float32(8.0)
        cx += _draw_hud_keycap(ctx, "ESC", cx, hcy, k_sz) + Float32(5.0)
        var t_base = hcy + t_sz * Float32(0.34)
        ctx._draw_text(lbl1, cx, t_base - t_sz * Float32(0.82), t_sz, rgba8(169, 180, 196, 215))
        cx += t1_w + Float32(9.0)
        ctx._draw_text(sep, cx, t_base - t_sz * Float32(0.82), t_sz, rgba8(132, 145, 163, 180))
        cx += sep_w + Float32(9.0)
        cx += _draw_hud_keycap(ctx, "F8", cx, hcy, k_sz) + Float32(5.0)
        ctx._draw_text(lbl2, cx, t_base - t_sz * Float32(0.82), t_sz, rgba8(169, 180, 196, 215))

    return len(ctx.draw_list.prims) - start_prims
