"""Flint v5 Gameplay HUD Overlay Helpers (`flint.ui.hud`).

Provides screen-space HUD primitives (`game_hud`, `hud_bar`) that emit directly into
`UiContext.draw_list` and work safely both inside and outside an active
`UiContext.begin_frame` / `end_frame` scope (preserving existing `draw_list.prims`
and never indexing an empty `box_stack`).
"""
from std.math import isfinite, max, min
from flint.ui.context import UiContext, UiTheme
from flint.ui.draw_list import UI_FLAG_BORDER, UI_FLAG_DROP_SHADOW


def hud_bar(
    mut ctx: UiContext,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    value: Int,
    max_value: Int,
    fill_rgba: UInt32 = 0xFF42D674,
):
    """Draws a clamped progress bar directly onto `ctx.draw_list`."""
    var safe_w = max(Float32(4.0), w) if isfinite(w) else Float32(180.0)
    var safe_h = max(Float32(4.0), h) if isfinite(h) else Float32(16.0)
    var safe_x = x if isfinite(x) else Float32(16.0)
    var safe_y = y if isfinite(y) else Float32(16.0)
    var safe_max = max(1, max_value)
    var safe_val = max(0, min(safe_max, value))
    var ratio = Float32(safe_val) / Float32(safe_max)

    ctx.draw_list.add_rect(safe_x, safe_y, safe_w, safe_h, UInt32(0xCC12161E), Float32(4.0), Float32(1.0), UInt32(0xFF334155))
    var inner_w = max(Float32(0.0), (safe_w - Float32(4.0)) * ratio)
    if inner_w > Float32(0.5):
        ctx.draw_list.add_rect(safe_x + Float32(2.0), safe_y + Float32(2.0), inner_w, max(Float32(2.0), safe_h - Float32(4.0)), fill_rgba, Float32(3.0))


def game_hud(
    mut ctx: UiContext,
    hp: Int,
    max_hp: Int,
    score: Int = 0,
) -> Int:
    """Emits a gameplay HUD overlay (HP bar + HP/score readouts) and returns added primitive count.

    Safe to call both inside an active `UiContext` frame and post-`end_frame` / headless
    without clearing existing `ctx.draw_list.prims`.
    """
    var sw = ctx.screen_w if (isfinite(ctx.screen_w) and ctx.screen_w > Float32(0.0)) else Float32(1280.0)
    var sh = ctx.screen_h if (isfinite(ctx.screen_h) and ctx.screen_h > Float32(0.0)) else Float32(720.0)
    var th = UiTheme.game_hud()

    var safe_max = max(1, max_hp)
    var safe_hp = max(0, min(safe_max, hp))
    var ratio = Float32(safe_hp) / Float32(safe_max)
    var bar_color = UInt32(0xFF42D674)
    if ratio <= Float32(0.25):
        bar_color = UInt32(0xFFEF4444)
    elif ratio <= Float32(0.55):
        bar_color = UInt32(0xFFF59E0B)

    var start_prims = len(ctx.draw_list.prims)

    var pad = Float32(14.0)
    var panel_w = min(Float32(248.0), max(Float32(140.0), sw - Float32(2.0) * pad))
    var panel_h = min(Float32(66.0), max(Float32(48.0), sh - Float32(2.0) * pad))
    var px = pad
    var py = pad

    ctx.draw_list.add_rect_styled(
        px,
        py,
        panel_w,
        panel_h,
        th.bg_panel,
        th.border,
        th.corner_radius,
        Float32(1.0),
        Float32(1.0),
        UI_FLAG_BORDER | UI_FLAG_DROP_SHADOW,
    )

    var font_px = max(Float32(12.0), min(Float32(16.0), th.font_size))
    var hp_label = "HP " + String(safe_hp) + "/" + String(safe_max)
    ctx._draw_text(hp_label, px + Float32(10.0), py + Float32(7.0), font_px, th.text_primary)

    var score_label = "SCORE " + String(score)
    ctx._draw_text(score_label, px + panel_w * Float32(0.54), py + Float32(7.0), font_px, th.accent)

    hud_bar(
        ctx,
        px + Float32(10.0),
        py + Float32(32.0),
        panel_w - Float32(20.0),
        Float32(18.0),
        safe_hp,
        safe_max,
        bar_color,
    )

    return len(ctx.draw_list.prims) - start_prims
