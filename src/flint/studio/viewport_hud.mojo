"""Flint Studio Unified Header Bar & Floating Translucent Viewport HUD (`flint.studio.viewport_hud`).

Pure-Mojo 1.2 UI rendering & hit-testing for:
  - `StudioHeaderBar` (`40 px` unified chrome bar): brand mark + menu chevron, scene breadcrumb pill
    with `ICON_SCENE` & dirty dot, recessed `l0_canvas` animated Workspace Preset segmented control
    (`Edit`, `Agents`, `Code`, `Lookdev`, `Play`), `Ctrl+K` Command Palette pill with measured keycap,
    live Agent Status pill with harmonized status ring, Ghost `ICON_MOON`/`ICON_SUN` theme toggle,
    and Save / Build status chips with tiered responsive compact behavior.
  - `ViewportHud`: 4 floating smoky frosted-glass HUD clusters inside `SceneViewport` with 3-tier
    responsive density, exact per-pill hit-rect tracking (outside passes straight through to 3D),
    area-filled 60-frame GPU sparkline, clean PIE/Play-mode chrome, and interactive 3D depth-sorted
    `OrientationGizmo` (`±X/±Y/±Z` camera snap + matched `Persp/Ortho` & `SEL`/delta status chips).
"""
from std.collections import List
from std.math import isfinite, sin, sqrt
from flint.core.math import Vec3
from flint.render.font import SdfFont
from flint.text.utf8 import decode_utf8, encode_utf8
from flint.ui.animator import SpringConfig, UI_ANIM_INSTANT, UiAnimator
from flint.ui.draw_list import (
    UI_FLAG_BORDER,
    UI_FLAG_TOP_BEVEL,
    UI_FLAG_VERT_GRADIENT,
    UiDrawList,
)
from flint.ui.flexbox_dock import RectF
from flint.ui.icons import draw_studio_icon
from flint.ui.theme import (
    AGENT_STATE_IDLE,
    AGENT_STATE_WAITING_APPROVAL,
    ICON_CAMERA,
    ICON_COLLIDER,
    ICON_EMPTY,
    ICON_GRID,
    ICON_LOCAL,
    ICON_MOON,
    ICON_MOVE,
    ICON_PAUSE,
    ICON_PLAY,
    ICON_ROTATE,
    ICON_SCALE,
    ICON_SCENE,
    ICON_SEARCH,
    ICON_SNAP,
    ICON_SPARK,
    ICON_STEP,
    ICON_STOP,
    ICON_SUN,
    ICON_WORLD,
    StudioTheme,
    lerp_rgba8,
    mix_rgba8,
    rgba8,
    rgba8_scale_alpha,
    rgba8_with_alpha,
)

# Unified Header Bar geometry & Workspace Presets
comptime STUDIO_HEADER_BAR_H: Float32 = 40.0
comptime STUDIO_STATUS_BAR_H: Float32 = 22.0
"""Bottom status bar: scene file + unsaved mark, selection, tool state, last message (click = console)."""

comptime WORKSPACE_EDIT: Int = 0
comptime WORKSPACE_AGENT_STUDIO: Int = 1
comptime WORKSPACE_CODE_SCENE: Int = 2
comptime WORKSPACE_LOOKDEV: Int = 3
comptime WORKSPACE_PLAY: Int = 4
comptime WORKSPACE_COUNT: Int = 5

# Header Bar Actions
comptime HDR_ACT_NONE: Int = 0
comptime HDR_ACT_WORKSPACE: Int = 1
comptime HDR_ACT_COMMAND_PALETTE: Int = 2
comptime HDR_ACT_AGENT_TOGGLE: Int = 3
comptime HDR_ACT_THEME_CYCLE: Int = 4
comptime HDR_ACT_SAVE: Int = 5
comptime HDR_ACT_BUILD: Int = 6
comptime HDR_ACT_MENU: Int = 7
comptime HDR_ACT_WIN_MINIMIZE: Int = 8
comptime HDR_ACT_WIN_MAXIMIZE: Int = 9
comptime HDR_ACT_WIN_CLOSE: Int = 10
comptime HDR_ACT_WIN_DRAG_MOVE: Int = 11
comptime HDR_ACT_WORKSPACE_PICKER: Int = 12

# Viewport HUD Actions
comptime VHUD_ACT_NONE: Int = 0
comptime VHUD_ACT_TOOL_SELECT: Int = 1
comptime VHUD_ACT_TOOL_MOVE: Int = 2
comptime VHUD_ACT_TOOL_ROTATE: Int = 3
comptime VHUD_ACT_TOOL_SCALE: Int = 4
comptime VHUD_ACT_TOGGLE_SPACE: Int = 5
comptime VHUD_ACT_TOGGLE_SNAP: Int = 6
comptime VHUD_ACT_PIE_PLAY: Int = 7
comptime VHUD_ACT_PIE_PAUSE: Int = 8
comptime VHUD_ACT_PIE_STEP: Int = 9
comptime VHUD_ACT_PIE_STOP: Int = 10
comptime VHUD_ACT_PIE_EJECT: Int = 11
comptime VHUD_ACT_CAM_MODE: Int = 12
comptime VHUD_ACT_TOGGLE_NAVMESH: Int = 13
comptime VHUD_ACT_TOGGLE_COLLIDERS: Int = 14
comptime VHUD_ACT_TOGGLE_GRID: Int = 15
comptime VHUD_ACT_TOGGLE_PERF: Int = 16
comptime VHUD_ACT_SNAP_VIEW: Int = 17
comptime VHUD_ACT_TOGGLE_ORTHO: Int = 18

comptime VHUD_PIE_STOPPED: Int = 0
comptime VHUD_PIE_PLAYING: Int = 1
comptime VHUD_PIE_PAUSED: Int = 2

comptime VHUD_TOOL_SELECT: Int = -1
comptime VHUD_TOOL_MOVE: Int = 0
comptime VHUD_TOOL_ROTATE: Int = 1
comptime VHUD_TOOL_SCALE: Int = 2

comptime GPU_SPARKLINE_SAMPLES: Int = 60


def workspace_preset_label(preset: Int) -> String:
    if preset == WORKSPACE_AGENT_STUDIO:
        return "Agent Studio"
    if preset == WORKSPACE_CODE_SCENE:
        return "Code + Scene"
    if preset == WORKSPACE_LOOKDEV:
        return "Lookdev"
    if preset == WORKSPACE_PLAY:
        return "Play"
    return "Edit"


def agent_status_short_label(status_code: Int, pending_approvals: Int = 0) -> String:
    if pending_approvals > 0 or status_code == AGENT_STATE_WAITING_APPROVAL:
        return "Approve (" + String(max(1, pending_approvals)) + ")"
    if status_code == 1:
        return "Thinking"
    if status_code == 2:
        return "Planning"
    if status_code == 3:
        return "Editing"
    if status_code == 4:
        return "Coding"
    if status_code == 5:
        return "Building"
    if status_code == 6:
        return "Testing"
    if status_code == 8:
        return "Error"
    return "Agent"


def _hud_hash(seed: UInt64, tag: String) -> UInt64:
    var h = seed
    for b in tag.as_bytes():
        h ^= UInt64(b)
        h *= UInt64(0x100000001B3)
    return h if h != 0 else UInt64(1)


def _measure_text_hud(font: SdfFont, text: String, size_px: Float32) -> Float32:
    if font.valid:
        return font.measure(text, size_px)
    return Float32(len(decode_utf8(text))) * size_px * Float32(0.56)


def _truncate_hud_text(font: SdfFont, text: String, size_px: Float32, max_w: Float32) -> String:
    if max_w <= Float32(4.0):
        return ""
    if _measure_text_hud(font, text, size_px) <= max_w:
        return text
    var codes = decode_utf8(text)
    var ell = String("...")
    var ell_w = _measure_text_hud(font, ell, size_px)
    var keep = len(codes)
    while keep > 0:
        var sub = encode_utf8(codes, 0, keep)
        if _measure_text_hud(font, sub, size_px) + ell_w <= max_w:
            return sub + ell
        keep -= 1
    return ell


def _draw_hud_text(
    mut dl: UiDrawList,
    font_slot: Int,
    font: SdfFont,
    text: String,
    x: Float32,
    y: Float32,
    size_px: Float32,
    color: UInt32,
):
    if text.byte_length() == 0:
        return
    if font.valid:
        dl.add_text(font_slot, font, text, x, y + size_px * Float32(0.82), size_px, color)
    else:
        var n_cp = len(decode_utf8(text))
        if n_cp > 0:
            dl.add_rect(
                x,
                y + size_px * Float32(0.20),
                Float32(n_cp) * size_px * Float32(0.54),
                size_px * Float32(0.60),
                rgba8_scale_alpha(color, Float32(0.42)),
                Float32(2.0),
            )


def _draw_hud_text_cy(
    mut dl: UiDrawList,
    font_slot: Int,
    font: SdfFont,
    text: String,
    x: Float32,
    center_y: Float32,
    size_px: Float32,
    color: UInt32,
):
    """Draws text optically cap-height centered around `center_y`."""
    if text.byte_length() == 0:
        return
    if font.valid:
        var baseline = center_y + size_px * Float32(0.35)
        dl.add_text(font_slot, font, text, x, baseline, size_px, color)
    else:
        var n_cp = len(decode_utf8(text))
        if n_cp > 0:
            var bh = size_px * Float32(0.56)
            dl.add_rect(
                x,
                center_y - bh * Float32(0.5),
                Float32(n_cp) * size_px * Float32(0.54),
                bh,
                rgba8_scale_alpha(color, Float32(0.42)),
                Float32(2.0),
            )


def _draw_navmesh_icon(
    mut dl: UiDrawList,
    x: Float32,
    y: Float32,
    s: Float32,
    color: UInt32,
    stroke: Float32 = 1.35,
):
    """Vector NavMesh polygon + path-edge glyph replacing the raw circle dot."""
    var x0 = x + s * Float32(0.16)
    var y0 = y + s * Float32(0.68)
    var x1 = x + s * Float32(0.36)
    var y1 = y + s * Float32(0.20)
    var x2 = x + s * Float32(0.84)
    var y2 = y + s * Float32(0.30)
    var x3 = x + s * Float32(0.66)
    var y3 = y + s * Float32(0.82)
    # Outer nav quad + internal triangulation diagonal
    dl.add_line(x0, y0, x1, y1, stroke, color)
    dl.add_line(x1, y1, x2, y2, stroke, color)
    dl.add_line(x2, y2, x3, y3, stroke, color)
    dl.add_line(x3, y3, x0, y0, stroke, color)
    dl.add_line(x1, y1, x3, y3, max(Float32(1.0), stroke * Float32(0.8)), rgba8_scale_alpha(color, Float32(0.68)))
    # Subtle waypoint vertices
    var vr = max(Float32(1.4), s * Float32(0.11))
    dl.add_circle(x1, y1, vr, color)
    dl.add_circle(x3, y3, vr, color)


def _draw_eject_icon(
    mut dl: UiDrawList,
    x: Float32,
    y: Float32,
    s: Float32,
    color: UInt32,
):
    """Compact media/camera Eject triangle + bar glyph."""
    var tri = List[Float32](capacity=6)
    tri.append(x + s * Float32(0.50)); tri.append(y + s * Float32(0.18))
    tri.append(x + s * Float32(0.84)); tri.append(y + s * Float32(0.58))
    tri.append(x + s * Float32(0.16)); tri.append(y + s * Float32(0.58))
    var cnt = List[Int](capacity=1)
    cnt.append(3)
    _ = dl.add_polygon(tri, cnt, color)
    dl.add_rect(x + s * Float32(0.18), y + s * Float32(0.70), s * Float32(0.64), max(Float32(1.6), s * Float32(0.14)), color, Float32(0.8))


def _draw_hud_glass_pill(
    mut dl: UiDrawList,
    theme: StudioTheme,
    r: RectF,
    radius: Float32,
    alpha_scale: Float32,
    border_col: UInt32,
    border_w: Float32 = 1.0,
):
    """Emits a smoky frosted-glass viewport pill with ambient shadow + 1px dark outer contrast rim."""
    var sh_a = Int(Float32(120.0) * alpha_scale)
    if sh_a > 4:
        dl.add_ambient_shadow(r.x, r.y, r.w, r.h, radius, Float32(10.0), sh_a, Float32(2.0))
    # 1px dark outer contrast keyline so glass separates cleanly over bright clouds & speculars
    dl.add_rect(
        r.x - Float32(0.8),
        r.y - Float32(0.8),
        r.w + Float32(1.6),
        r.h + Float32(1.6),
        UInt32(0),
        radius + Float32(0.8),
        Float32(1.0),
        rgba8(6, 8, 12, Int(Float32(115.0) * alpha_scale)),
    )
    var glass_tint = rgba8_scale_alpha(rgba8_with_alpha(theme.l4_overlay, 212), alpha_scale)
    dl.add_glass_panel(
        r.x,
        r.y,
        r.w,
        r.h,
        glass_tint,
        radius,
        Float32(4.0),
        border_w,
        rgba8_scale_alpha(border_col, alpha_scale),
        True,
        False,
    )


def _fmt_f32_1dp(v: Float32) -> String:
    if not isfinite(v):
        return "0.0"
    var neg = v < Float32(0.0)
    var mag = -v if neg else v
    var scaled = Int(mag * Float32(10.0) + Float32(0.5))
    var whole = scaled // 10
    var frac = scaled % 10
    return ("-" if neg and scaled > 0 else "") + String(whole) + "." + String(frac)


def _fmt_f32_2dp(v: Float32) -> String:
    if not isfinite(v):
        return "0.00"
    var neg = v < Float32(0.0)
    var mag = -v if neg else v
    var scaled = Int(mag * Float32(100.0) + Float32(0.5))
    var whole = scaled // 100
    var frac = scaled % 100
    var fstr = ("0" + String(frac)) if frac < 10 else String(frac)
    return ("-" if neg and scaled > 0 else "") + String(whole) + "." + fstr


# ============================================================================
# 1. Unified Studio Header Bar (`40 px`)
# ============================================================================

@fieldwise_init
struct HeaderBarAction(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var int_val: Int

    def __init__(out self):
        self.kind = HDR_ACT_NONE
        self.int_val = 0


struct StudioHeaderBar(Copyable, Movable):
    """Unified `40 px` Studio header bar with recessed Workspace Preset segmented rail."""
    var rect: RectF
    var active_workspace: Int
    var hovered: Bool
    var last_hit_rect: RectF
    var _last_bar_click_sec: Float32

    def __init__(out self):
        self.rect = RectF(0.0, 0.0, 1280.0, STUDIO_HEADER_BAR_H)
        self.active_workspace = WORKSPACE_EDIT
        self.hovered = False
        self.last_hit_rect = RectF()
        self._last_bar_click_sec = Float32(-10.0)

    def contains_point(self, mx: Float32, my: Float32) -> Bool:
        return self.rect.contains(mx, my)

    def draw(
        mut self,
        mut dl: UiDrawList,
        font_slot: Int,
        font: SdfFont,
        theme: StudioTheme,
        mut anim: UiAnimator,
        screen_w: Float32,
        scene_title: String,
        dirty: Bool,
        active_workspace: Int,
        agent_status: Int,
        pending_approvals: Int,
        build_state: Int,  # 0=ok, 1=building, 2=error
        mouse_x: Float32,
        mouse_y: Float32,
        mouse_down: Bool,
        mouse_pressed: Bool,
        y_offset: Float32 = 0.0,
        show_window_controls: Bool = False,
        window_maximized: Bool = False,
        pie_active: Bool = False,
    ) -> HeaderBarAction:
        """Emits the unified Studio header bar and returns any triggered `HeaderBarAction`."""
        var action = HeaderBarAction()
        var sw = max(Float32(320.0), screen_w if isfinite(screen_w) else Float32(1280.0))
        self.rect = RectF(Float32(0.0), y_offset, sw, STUDIO_HEADER_BAR_H)
        self.active_workspace = max(0, min(WORKSPACE_COUNT - 1, active_workspace))
        self.hovered = self.rect.contains(mouse_x, mouse_y)
        self.last_hit_rect = self.rect

        # Subtle vertical depth gradient (`l1_shell` -> deeper obsidian) + crisp bottom seam
        var bar_bot = mix_rgba8(theme.l0_canvas, theme.l1_shell, 55)
        dl.add_rect_styled(
            self.rect.x,
            self.rect.y,
            self.rect.w,
            self.rect.h,
            theme.l1_shell,
            bar_bot,
            Float32(0.0),
            Float32(0.0),
            Float32(1.0),
            UI_FLAG_VERT_GRADIENT,
        )
        dl.add_line(
            self.rect.x,
            self.rect.bottom() - Float32(0.5),
            self.rect.right(),
            self.rect.bottom() - Float32(0.5),
            Float32(1.0),
            theme.border_control if theme.is_dark else theme.border_subtle,
        )

        var cy = self.rect.y + self.rect.h * Float32(0.5)
        var text_sz = Float32(12.0)

        # Tiered responsive breakpoints:
        #   - full: sw >= 900 (with win controls) or sw >= 780
        #   - mid_compact: 500 <= sw < full (e.g. 640px 2x HiDPI — keeps Search + Agent + Theme + Build icons!)
        #   - ultra_compact: sw < 500 (e.g. 320px narrow — keeps compact brand + workspace picker + Search)
        var full_thresh = Float32(900.0) if show_window_controls else Float32(780.0)
        var full_header = sw >= full_thresh
        var mid_compact = (not full_header) and (sw >= Float32(500.0))
        var compact_brand = sw < Float32(560.0)

        # ---- Left Zone: Brand + Menu Chevron + Scene Breadcrumb ----
        var lx = Float32(8.0)
        var brand_text_w = Float32(0.0) if compact_brand else _measure_text_hud(font, "Flint Studio", Float32(13.5))
        var brand_w = Float32(34.0) if compact_brand else (brand_text_w + Float32(48.0))
        var brand_r = RectF(lx, cy - Float32(14.0), brand_w, Float32(28.0))
        var brand_hov = brand_r.contains(mouse_x, mouse_y)
        if brand_hov:
            dl.add_rect_styled(
                brand_r.x,
                brand_r.y,
                brand_r.w,
                brand_r.h,
                theme.l3_hover,
                theme.border_subtle,
                theme.radius_sm,
                Float32(1.0),
                Float32(1.0),
                UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )
            if mouse_pressed:
                action = HeaderBarAction(HDR_ACT_MENU, 0)
        var mark_x = lx + Float32(4.0)
        dl.add_rect(mark_x, cy - Float32(10.5), Float32(21.0), Float32(21.0), theme.accent_primary, Float32(5.0))
        _ = dl.add_polygon(
            [
                mark_x + Float32(4.8), cy + Float32(5.5),
                mark_x + Float32(8.6), cy - Float32(5.5),
                mark_x + Float32(17.0), cy - Float32(6.5),
                mark_x + Float32(13.4), cy - Float32(0.8),
                mark_x + Float32(9.6), cy - Float32(0.8),
            ],
            [5],
            theme.text_on_accent,
        )
        if not compact_brand:
            _draw_hud_text_cy(dl, font_slot, font, "Flint Studio", lx + Float32(31.0), cy, Float32(13.5), theme.text_primary)
            # Subtle down-chevron signaling application menu
            var ch_x = brand_r.right() - Float32(10.0)
            var ch_col = theme.text_primary if brand_hov else theme.text_muted
            dl.add_line(ch_x - Float32(3.0), cy - Float32(1.2), ch_x, cy + Float32(1.8), Float32(1.25), ch_col)
            dl.add_line(ch_x, cy + Float32(1.8), ch_x + Float32(3.0), cy - Float32(1.2), Float32(1.25), ch_col)
        else:
            var ch_x = brand_r.right() - Float32(5.0)
            var ch_col = theme.text_primary if brand_hov else theme.text_muted
            dl.add_line(ch_x - Float32(2.2), cy - Float32(1.0), ch_x, cy + Float32(1.4), Float32(1.2), ch_col)
            dl.add_line(ch_x, cy + Float32(1.4), ch_x + Float32(2.2), cy - Float32(1.0), Float32(1.2), ch_col)

        lx += brand_w + Float32(6.0)
        var title_w = Float32(0.0)
        if full_header:
            dl.add_line(lx, cy - Float32(8.0), lx, cy + Float32(8.0), Float32(1.0), theme.border_subtle)
            lx += Float32(8.0)
            var max_title_w = min(Float32(162.0), sw * Float32(0.14))
            var disp_title = _truncate_hud_text(font, scene_title if scene_title else String("Untitled"), text_sz, max_title_w)
            var txt_w = _measure_text_hud(font, disp_title, text_sz)
            var sc_pill_w = Float32(24.0) + txt_w + (Float32(14.0) if dirty else Float32(8.0))
            var sc_r = RectF(lx, cy - Float32(12.0), sc_pill_w, Float32(24.0))
            var sc_hov = sc_r.contains(mouse_x, mouse_y)
            if sc_hov:
                dl.add_rect(sc_r.x, sc_r.y, sc_r.w, sc_r.h, theme.l3_hover, theme.radius_xs)
                if mouse_pressed:
                    action = HeaderBarAction(HDR_ACT_MENU, 0)
            draw_studio_icon(dl, ICON_SCENE, sc_r.x + Float32(5.0), cy - Float32(6.0), Float32(12.0), theme.text_muted, Float32(1.3))
            _draw_hud_text_cy(dl, font_slot, font, disp_title, sc_r.x + Float32(21.0), cy, text_sz, theme.text_primary if sc_hov else theme.text_secondary)
            if dirty:
                dl.add_circle(sc_r.right() - Float32(7.0), cy, Float32(2.8), theme.play_paused)
            title_w = sc_pill_w
        elif dirty and mid_compact:
            dl.add_circle(lx + Float32(2.0), cy, Float32(2.8), theme.play_paused)
            title_w = Float32(10.0)

        # ---- Right Zone (laid out right-to-left so center workspace pill never overlaps) ----
        var rx = self.rect.right() - Float32(8.0)

        # 0. Native Wayland Client-Side Window Controls (Minimize / Maximize / Close)
        if show_window_controls:
            var wb_w = Float32(26.0)
            var wb_h = Float32(22.0)
            var wb_y = cy - Float32(11.0)

            # Close (X)
            rx -= wb_w
            var cl_r = RectF(rx, wb_y, wb_w, wb_h)
            var cl_hov = cl_r.contains(mouse_x, mouse_y)
            var cl_bg = rgba8_with_alpha(theme.status_danger, 225 if (cl_hov and mouse_down) else (195 if cl_hov else 0))
            if cl_hov:
                dl.add_rect(cl_r.x, cl_r.y, cl_r.w, cl_r.h, cl_bg, theme.radius_xs)
            var cl_fg = theme.text_on_accent if cl_hov else theme.text_secondary
            var cx0 = cl_r.center_x()
            var cy0 = cl_r.center_y()
            dl.add_line(cx0 - Float32(4.0), cy0 - Float32(4.0), cx0 + Float32(4.0), cy0 + Float32(4.0), Float32(1.5), cl_fg)
            dl.add_line(cx0 + Float32(4.0), cy0 - Float32(4.0), cx0 - Float32(4.0), cy0 + Float32(4.0), Float32(1.5), cl_fg)
            if cl_hov and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_WIN_CLOSE, 0)
            rx -= Float32(3.0)

            # Maximize / Restore
            rx -= wb_w
            var mx_r = RectF(rx, wb_y, wb_w, wb_h)
            var mx_hov = mx_r.contains(mouse_x, mouse_y)
            if mx_hov:
                dl.add_rect(mx_r.x, mx_r.y, mx_r.w, mx_r.h, theme.l3_hover, theme.radius_xs)
            var mx_fg = theme.text_primary if mx_hov else theme.text_secondary
            var mx_cx = mx_r.center_x()
            var mx_cy = mx_r.center_y()
            if window_maximized:
                dl.add_rect(mx_cx - Float32(2.5), mx_cy - Float32(5.0), Float32(7.0), Float32(7.0), 0, Float32(1.0), Float32(1.3), mx_fg)
                dl.add_rect(mx_cx - Float32(5.0), mx_cy - Float32(2.5), Float32(7.0), Float32(7.0), theme.l1_shell, Float32(1.0), Float32(1.3), mx_fg)
            else:
                dl.add_rect(mx_cx - Float32(4.5), mx_cy - Float32(4.5), Float32(9.0), Float32(9.0), 0, Float32(1.5), Float32(1.4), mx_fg)
            if mx_hov and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_WIN_MAXIMIZE, 0)
            rx -= Float32(3.0)

            # Minimize (—)
            rx -= wb_w
            var mn_r = RectF(rx, wb_y, wb_w, wb_h)
            var mn_hov = mn_r.contains(mouse_x, mouse_y)
            if mn_hov:
                dl.add_rect(mn_r.x, mn_r.y, mn_r.w, mn_r.h, theme.l3_hover, theme.radius_xs)
            var mn_fg = theme.text_primary if mn_hov else theme.text_secondary
            dl.add_line(mn_r.center_x() - Float32(4.5), mn_r.center_y() + Float32(2.0), mn_r.center_x() + Float32(4.5), mn_r.center_y() + Float32(2.0), Float32(1.5), mn_fg)
            if mn_hov and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_WIN_MINIMIZE, 0)
            rx -= Float32(6.0)

            dl.add_line(rx, cy - Float32(8.0), rx, cy + Float32(8.0), Float32(1.0), theme.border_subtle)
            rx -= Float32(8.0)
        else:
            rx -= Float32(2.0)

        var ag_col = theme.agent_status_color(agent_status)
        var ag_busy = agent_status != AGENT_STATE_IDLE or pending_approvals > 0
        var bld_dot = theme.status_danger if build_state == 2 else (theme.status_warn if build_state == 1 else theme.status_ok)

        if full_header:
            # ---- Group 2: Build + Save + Ghost Theme Toggle ----
            # 1. Build status chip (primary elevated utility action)
            var bld_txt_w = _measure_text_hud(font, "Build", Float32(11.5))
            var bld_w = max(Float32(58.0), bld_txt_w + Float32(26.0))
            rx -= bld_w
            var bld_r = RectF(rx, cy - Float32(12.0), bld_w, Float32(24.0))
            var bld_hov = bld_r.contains(mouse_x, mouse_y)
            var bld_ht = anim.drive_widget(_hud_hash(UInt64(0x48445201), "build"), bld_hov, bld_hov and mouse_down, False)
            var bld_bg = lerp_rgba8(theme.l3_raised, theme.l3_hover, bld_ht[0])
            var bld_bdr = rgba8_with_alpha(bld_dot, 130) if build_state != 0 else (theme.border_strong if bld_hov else theme.border_control)
            dl.add_rect_styled(bld_r.x, bld_r.y, bld_r.w, bld_r.h, bld_bg, bld_bdr, theme.radius_sm, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL)
            dl.add_circle(bld_r.x + Float32(10.0), cy, Float32(3.2), bld_dot)
            _draw_hud_text_cy(dl, font_slot, font, "Build", bld_r.x + Float32(18.0), cy, Float32(11.5), theme.text_primary)
            if bld_hov and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_BUILD, 0)
            rx -= Float32(6.0)

            # 2. Save button chip (secondary; dimmed when PIE active)
            var save_txt_w = _measure_text_hud(font, "Save", Float32(11.5))
            var save_w = max(Float32(48.0), save_txt_w + Float32(20.0))
            rx -= save_w
            var save_r = RectF(rx, cy - Float32(12.0), save_w, Float32(24.0))
            var save_hov = (not pie_active) and save_r.contains(mouse_x, mouse_y)
            var save_ht = anim.drive_widget(_hud_hash(UInt64(0x48445202), "save"), save_hov, save_hov and mouse_down, False)
            var save_alpha = Float32(0.42) if pie_active else Float32(1.0)
            var save_bg = lerp_rgba8(theme.l2_card, theme.l3_hover, save_ht[0])
            var save_bdr = theme.play_paused if (dirty and not pie_active) else (theme.border_strong if save_hov else theme.border_control)
            dl.add_rect_styled(
                save_r.x,
                save_r.y,
                save_r.w,
                save_r.h,
                rgba8_scale_alpha(save_bg, save_alpha),
                rgba8_scale_alpha(save_bdr, save_alpha),
                theme.radius_sm,
                Float32(1.0),
                Float32(1.0),
                UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )
            var save_fg = theme.text_primary if (dirty or save_hov) else theme.text_secondary
            _draw_hud_text_cy(dl, font_slot, font, "Save", save_r.center_x() - save_txt_w * Float32(0.5), cy, Float32(11.5), rgba8_scale_alpha(save_fg, save_alpha))
            if save_hov and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_SAVE, 0)
            rx -= Float32(5.0)

            # 3. Ghost Theme toggle button (`ICON_MOON` / `ICON_SUN`, borderless until hover)
            var thm_w = Float32(26.0)
            rx -= thm_w
            var thm_r = RectF(rx, cy - Float32(12.0), thm_w, Float32(24.0))
            var thm_hov = thm_r.contains(mouse_x, mouse_y)
            var thm_ht = anim.drive_widget(_hud_hash(UInt64(0x48445203), "theme"), thm_hov, thm_hov and mouse_down, False)
            if thm_hov or thm_ht[0] > Float32(0.02):
                dl.add_rect_styled(
                    thm_r.x,
                    thm_r.y,
                    thm_r.w,
                    thm_r.h,
                    rgba8_scale_alpha(theme.l3_hover, thm_ht[0]),
                    rgba8_scale_alpha(theme.border_control, thm_ht[0]),
                    theme.radius_sm,
                    Float32(1.0),
                    Float32(1.0),
                    UI_FLAG_BORDER,
                )
            var thm_icon = ICON_MOON if theme.is_dark else ICON_SUN
            var thm_col = theme.text_primary if thm_hov else theme.text_secondary
            draw_studio_icon(dl, thm_icon, thm_r.center_x() - Float32(6.5), cy - Float32(6.5), Float32(13.0), thm_col, Float32(1.4))
            if thm_hov and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_THEME_CYCLE, 0)
            rx -= Float32(6.0)

            # Group Divider (16px hairline separating Actions from Navigate & AI)
            dl.add_line(rx, cy - Float32(8.0), rx, cy + Float32(8.0), Float32(1.0), theme.border_subtle)
            rx -= Float32(7.0)

            # ---- Group 1: Live Agent Status Pill + Search (Ctrl+K) ----
            # 4. Live Agent Status Pill (harmonized color + zero spinner/spark overlap)
            var ag_label = agent_status_short_label(agent_status, pending_approvals)
            var ag_txt_w = _measure_text_hud(font, ag_label, Float32(11.5))
            var ag_w = max(Float32(74.0), ag_txt_w + Float32(34.0))
            rx -= ag_w
            var ag_r = RectF(rx, cy - Float32(12.0), ag_w, Float32(24.0))
            var ag_hov = ag_r.contains(mouse_x, mouse_y)
            var ag_ht = anim.drive_widget(_hud_hash(UInt64(0x48445204), "agent"), ag_hov, ag_hov and mouse_down, False)
            # Match fill tint directly to `ag_col` (kills purple-under-amber clash on Approve!)
            var ag_idle_bg = rgba8_with_alpha(ag_col, 32) if ag_busy else theme.l2_card
            var ag_bg = lerp_rgba8(ag_idle_bg, theme.l3_hover, ag_ht[0] * Float32(0.75))
            var ag_bdr = rgba8_with_alpha(ag_col, 155) if ag_busy else (theme.border_strong if ag_hov else theme.border_subtle)
            dl.add_rect_styled(ag_r.x, ag_r.y, ag_r.w, ag_r.h, ag_bg, ag_bdr, Float32(12.0), Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL)
            var icon_cx = ag_r.x + Float32(13.0)
            if ag_busy:
                # Crisp spinner ring with small centered core dot (zero collision with ICON_SPARK!)
                var spin_start = Float32(0.0) if anim.mode == UI_ANIM_INSTANT else (anim.time_sec * Float32(3.6))
                dl.add_arc(icon_cx, cy, Float32(5.6), Float32(1.7), spin_start, Float32(4.2), ag_col, rgba8_with_alpha(ag_col, 55), Float32(0.0))
                dl.add_circle(icon_cx, cy, Float32(2.0), ag_col)
            else:
                draw_studio_icon(dl, ICON_SPARK, icon_cx - Float32(5.5), cy - Float32(5.5), Float32(11.0), ag_col, Float32(1.3))
            _draw_hud_text_cy(dl, font_slot, font, ag_label, ag_r.x + Float32(24.0), cy, Float32(11.5), theme.text_primary if (ag_busy or ag_hov) else theme.text_secondary)
            if ag_hov and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_AGENT_TOGGLE, 0)
            rx -= Float32(7.0)

            # 5. Ctrl+K Command Palette Search Pill (recessed well + measured keycap)
            var kc_txt = String("Ctrl+K")
            var kc_tw = _measure_text_hud(font, kc_txt, Float32(10.0))
            var kc_w = max(Float32(44.0), kc_tw + Float32(10.0))
            var pal_w = max(kc_w + Float32(82.0), Float32(142.0)) if sw >= Float32(1060.0) else (kc_w + Float32(36.0))
            rx -= pal_w
            var pal_r = RectF(rx, cy - Float32(12.0), pal_w, Float32(24.0))
            var pal_hov = pal_r.contains(mouse_x, mouse_y)
            var pal_ht = anim.drive_widget(_hud_hash(UInt64(0x48445205), "cmdpal"), pal_hov, pal_hov and mouse_down, False)
            dl.add_recessed_well(
                pal_r.x,
                pal_r.y,
                pal_r.w,
                pal_r.h,
                lerp_rgba8(theme.l0_canvas, theme.l2_card, pal_ht[0]),
                theme.border_strong if pal_hov else theme.border_control,
                theme.radius_sm,
                Float32(1.0),
            )
            draw_studio_icon(dl, ICON_SEARCH, pal_r.x + Float32(7.0), cy - Float32(5.5), Float32(11.5), theme.text_secondary if pal_hov else theme.text_muted, Float32(1.35))
            if pal_w >= kc_w + Float32(74.0):
                _draw_hud_text_cy(dl, font_slot, font, "Search", pal_r.x + Float32(23.0), cy, Float32(11.5), theme.text_secondary if pal_hov else theme.text_muted)
            var kc_x = pal_r.right() - kc_w - Float32(3.5)
            dl.add_rect_styled(kc_x, pal_r.y + Float32(3.0), kc_w, Float32(18.0), theme.l2_card, theme.border_control, Float32(3.5), Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL)
            _draw_hud_text_cy(dl, font_slot, font, kc_txt, kc_x + (kc_w - kc_tw) * Float32(0.5), pal_r.center_y(), Float32(10.0), theme.text_secondary)
            if pal_hov and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_COMMAND_PALETTE, 0)

        elif mid_compact:
            # Tiered mid-compact right utilities (500..780px, e.g. 640px 2x HiDPI):
            # Keeps Build dot, Ghost Theme icon, Agent status dot/icon, and Search button!
            var ic_sz = Float32(26.0)
            var ic_y = cy - Float32(13.0)

            # Build status icon chip
            rx -= ic_sz
            var mb_r = RectF(rx, ic_y, ic_sz, ic_sz)
            var mb_hov = mb_r.contains(mouse_x, mouse_y)
            dl.add_rect_styled(mb_r.x, mb_r.y, mb_r.w, mb_r.h, theme.l3_hover if mb_hov else theme.l2_card, rgba8_with_alpha(bld_dot, 130) if build_state != 0 else theme.border_control, theme.radius_sm, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL)
            dl.add_circle(mb_r.center_x(), cy, Float32(4.0), bld_dot)
            if mb_hov and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_BUILD, 0)
            rx -= Float32(4.0)

            # Theme icon chip (ghost)
            rx -= ic_sz
            var mt_r = RectF(rx, ic_y, ic_sz, ic_sz)
            var mt_hov = mt_r.contains(mouse_x, mouse_y)
            if mt_hov:
                dl.add_rect(mt_r.x, mt_r.y, mt_r.w, mt_r.h, theme.l3_hover, theme.radius_sm, Float32(1.0), theme.border_control)
            draw_studio_icon(dl, ICON_MOON if theme.is_dark else ICON_SUN, mt_r.center_x() - Float32(6.5), cy - Float32(6.5), Float32(13.0), theme.text_primary if mt_hov else theme.text_secondary, Float32(1.4))
            if mt_hov and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_THEME_CYCLE, 0)
            rx -= Float32(4.0)

            # Agent status icon chip (shows colored ring when busy/approval pending!)
            rx -= ic_sz
            var ma_r = RectF(rx, ic_y, ic_sz, ic_sz)
            var ma_hov = ma_r.contains(mouse_x, mouse_y)
            dl.add_rect_styled(ma_r.x, ma_r.y, ma_r.w, ma_r.h, rgba8_with_alpha(ag_col, 36) if ag_busy else (theme.l3_hover if ma_hov else theme.l2_card), rgba8_with_alpha(ag_col, 165) if ag_busy else theme.border_control, theme.radius_sm, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL)
            if ag_busy:
                dl.add_arc(ma_r.center_x(), cy, Float32(6.0), Float32(1.6), Float32(0.0), Float32(4.2), ag_col, rgba8_with_alpha(ag_col, 55), Float32(0.0))
                dl.add_circle(ma_r.center_x(), cy, Float32(2.0), ag_col)
            else:
                draw_studio_icon(dl, ICON_SPARK, ma_r.center_x() - Float32(5.5), cy - Float32(5.5), Float32(11.0), ag_col, Float32(1.3))
            if ma_hov and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_AGENT_TOGGLE, 0)
            rx -= Float32(4.0)

            # Search icon chip
            rx -= ic_sz
            var ms_r = RectF(rx, ic_y, ic_sz, ic_sz)
            var ms_hov = ms_r.contains(mouse_x, mouse_y)
            dl.add_recessed_well(ms_r.x, ms_r.y, ms_r.w, ms_r.h, theme.l3_hover if ms_hov else theme.l0_canvas, theme.border_strong if ms_hov else theme.border_control, theme.radius_sm, Float32(1.0))
            draw_studio_icon(dl, ICON_SEARCH, ms_r.center_x() - Float32(6.0), cy - Float32(6.0), Float32(12.0), theme.text_primary if ms_hov else theme.text_secondary, Float32(1.4))
            if ms_hov and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_COMMAND_PALETTE, 0)
            rx -= Float32(6.0)
        else:
            # Ultra-compact (< 500px, e.g. 320px logical): single Search button preserving workspace picker room
            var search = RectF(rx - Float32(28.0), cy - Float32(13.0), Float32(28.0), Float32(26.0))
            var hover = search.contains(mouse_x, mouse_y)
            dl.add_recessed_well(search.x, search.y, search.w, search.h, theme.l3_hover if hover else theme.l0_canvas, theme.border_control, theme.radius_sm, Float32(1.0))
            draw_studio_icon(dl, ICON_SEARCH, search.center_x() - Float32(6.5), cy - Float32(6.5), Float32(13.0), theme.text_primary, Float32(1.4))
            if hover and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_COMMAND_PALETTE, 0)
            rx = search.x - Float32(8.0)

        # ---- Center Zone: Recessed Workspace Preset Segmented Control Rail ----
        var pill_labels = List[String](capacity=WORKSPACE_COUNT)
        pill_labels.append("Edit")
        pill_labels.append("Agents")
        pill_labels.append("Code")
        pill_labels.append("Lookdev")
        pill_labels.append("Play")

        var pill_ws = List[Float32](capacity=WORKSPACE_COUNT)
        var tab_pad_x = Float32(16.0) if full_header else Float32(12.0)
        var total_pw = Float32(6.0)
        for pi in range(WORKSPACE_COUNT):
            var min_tw = Float32(42.0) if full_header else Float32(36.0)
            var w = max(min_tw, _measure_text_hud(font, pill_labels[pi], Float32(11.5)) + tab_pad_x)
            pill_ws.append(w)
            total_pw += w + Float32(2.0)

        var left_bound = lx + title_w + (Float32(14.0) if full_header else Float32(10.0))
        var right_bound = rx - Float32(10.0)
        if right_bound - left_bound >= total_pw:
            var ideal_x = (sw - total_pw) * Float32(0.5)
            var group_x = max(left_bound, min(right_bound - total_pw, ideal_x))
            var group_r = RectF(group_x, cy - Float32(14.0), total_pw, Float32(28.0))

            # Recessed `l0_canvas` segmented control rail
            dl.add_recessed_well(
                group_r.x,
                group_r.y,
                group_r.w,
                group_r.h,
                theme.l0_canvas,
                theme.border_subtle,
                theme.radius_sm,
                Float32(1.0),
            )

            # Compute tab rects & target for animated sliding active pill
            var tab_xs = List[Float32](capacity=WORKSPACE_COUNT)
            var cur_x = group_r.x + Float32(3.0)
            var tgt_x = cur_x
            var tgt_w = pill_ws[0]
            for pi in range(WORKSPACE_COUNT):
                tab_xs.append(cur_x)
                if pi == self.active_workspace:
                    tgt_x = cur_x
                    tgt_w = pill_ws[pi]
                cur_x += pill_ws[pi] + Float32(2.0)

            var ind = anim.drive_tab_pill(UInt64(0x484452575350), tgt_x, tgt_w)
            var active_accent = (
                theme.agent_active
                if self.active_workspace == WORKSPACE_AGENT_STUDIO
                else (theme.play_mode if self.active_workspace == WORKSPACE_PLAY else theme.accent_primary)
            )
            var pill_y = group_r.y + Float32(2.0)
            var pill_h = group_r.h - Float32(4.0)
            # Elevated active pill with top specular bevel + crisp inset accent underline clearing descenders
            dl.add_rect_styled(
                ind[0],
                pill_y,
                ind[1],
                pill_h,
                theme.l3_raised,
                rgba8_with_alpha(active_accent, 92),
                theme.radius_xs,
                Float32(1.0),
                Float32(1.0),
                UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )
            dl.add_rect(
                ind[0] + Float32(8.0),
                pill_y + pill_h - Float32(2.2),
                max(Float32(8.0), ind[1] - Float32(16.0)),
                Float32(2.0),
                active_accent,
                Float32(1.0),
            )

            for pi in range(WORKSPACE_COUNT):
                var tr = RectF(tab_xs[pi], pill_y, pill_ws[pi], pill_h)
                var thov = tr.contains(mouse_x, mouse_y)
                var is_act = pi == self.active_workspace
                if thov and not is_act:
                    dl.add_rect(tr.x, tr.y, tr.w, tr.h, rgba8_with_alpha(theme.l3_hover, 130), theme.radius_xs)
                var tw = _measure_text_hud(font, pill_labels[pi], Float32(11.5))
                _draw_hud_text_cy(
                    dl,
                    font_slot,
                    font,
                    pill_labels[pi],
                    tr.x + max(Float32(4.0), (tr.w - tw) * Float32(0.5)),
                    cy - Float32(0.5),
                    Float32(11.5),
                    theme.text_primary if (is_act or thov) else theme.text_secondary,
                )
                if thov and mouse_pressed:
                    action = HeaderBarAction(HDR_ACT_WORKSPACE, pi)

        elif right_bound - left_bound >= Float32(42.0):
            var picker = RectF(left_bound, cy - Float32(13.0), min(Float32(124.0), right_bound - left_bound), Float32(26.0))
            var over = picker.contains(mouse_x, mouse_y)
            dl.add_rect_styled(
                picker.x,
                picker.y,
                picker.w,
                picker.h,
                theme.l3_hover if over else theme.l2_card,
                theme.border_strong if over else theme.border_control,
                theme.radius_sm,
                Float32(1.0),
                Float32(1.0),
                UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )
            var label = _truncate_hud_text(font, pill_labels[self.active_workspace], Float32(11.5), picker.w - Float32(24.0))
            _draw_hud_text_cy(dl, font_slot, font, label, picker.x + Float32(8.0), cy, Float32(11.5), theme.text_primary)
            dl.add_line(picker.right() - Float32(12.0), cy - Float32(1.2), picker.right() - Float32(9.0), cy + Float32(1.8), Float32(1.3), theme.text_secondary)
            dl.add_line(picker.right() - Float32(9.0), cy + Float32(1.8), picker.right() - Float32(6.0), cy - Float32(1.2), Float32(1.3), theme.text_secondary)
            if over and mouse_pressed:
                action = HeaderBarAction(HDR_ACT_WORKSPACE_PICKER, 0)

        # Clicking empty draggable area of the HeaderBar initiates native Wayland move (or double-click maximize)
        if show_window_controls and action.kind == HDR_ACT_NONE and self.hovered and mouse_pressed:
            var dt_click = anim.time_sec - self._last_bar_click_sec
            if dt_click > Float32(0.0) and dt_click < Float32(0.38):
                action = HeaderBarAction(HDR_ACT_WIN_MAXIMIZE, 0)
                self._last_bar_click_sec = Float32(-10.0)
            else:
                self._last_bar_click_sec = anim.time_sec
                action = HeaderBarAction(HDR_ACT_WIN_DRAG_MOVE, 0)

        return action


# ============================================================================
# 2. Floating Translucent Viewport Overlay HUD (4 Clusters + 3D OrientationGizmo)
# ============================================================================

@fieldwise_init
struct ViewportHudAction(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var snap_dir: Vec3
    var snap_axis_index: Int  # 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z

    def __init__(out self):
        self.kind = VHUD_ACT_NONE
        self.snap_dir = Vec3()
        self.snap_axis_index = -1


struct _AxisProj(Copyable, ImplicitlyCopyable, Movable):
    var idx: Int
    var sx: Float32
    var sy: Float32
    var depth: Float32
    var positive: Bool
    var color: UInt32
    var label: String
    var snap_fwd: Vec3

    def __init__(
        out self,
        idx: Int,
        sx: Float32,
        sy: Float32,
        depth: Float32,
        positive: Bool,
        color: UInt32,
        label: String,
        snap_fwd: Vec3,
    ):
        self.idx = idx
        self.sx = sx
        self.sy = sy
        self.depth = depth
        self.positive = positive
        self.color = color
        self.label = label
        self.snap_fwd = snap_fwd


struct ViewportHud(Copyable, Movable):
    """4-cluster translucent floating Viewport HUD with 3-tier responsive compaction,
    exact pointer passthrough, clean Play-mode chrome, & 3D OrientationGizmo."""
    var gpu_ms_ring: List[Float32]
    var ring_cursor: Int
    var ring_count: Int
    var perf_expanded: Bool
    var hit_rects: List[RectF]
    var hovered_orientation_axis: Int
    var reserved_top_px: Float32

    def __init__(out self):
        self.gpu_ms_ring = List[Float32](length=GPU_SPARKLINE_SAMPLES, fill=Float32(0.0))
        self.ring_cursor = 0
        self.ring_count = 0
        self.perf_expanded = False
        self.hit_rects = List[RectF]()
        self.hovered_orientation_axis = -1
        self.reserved_top_px = Float32(48.0)

    def __init__(out self, *, copy: Self):
        self.gpu_ms_ring = copy.gpu_ms_ring.copy()
        self.ring_cursor = copy.ring_cursor
        self.ring_count = copy.ring_count
        self.perf_expanded = copy.perf_expanded
        self.hit_rects = copy.hit_rects.copy()
        self.hovered_orientation_axis = copy.hovered_orientation_axis
        self.reserved_top_px = copy.reserved_top_px

    def push_gpu_ms(mut self, gpu_ms: Float32):
        var v = max(Float32(0.0), gpu_ms) if isfinite(gpu_ms) else Float32(0.0)
        if len(self.gpu_ms_ring) != GPU_SPARKLINE_SAMPLES:
            self.gpu_ms_ring = List[Float32](length=GPU_SPARKLINE_SAMPLES, fill=Float32(0.0))
            self.ring_cursor = 0
            self.ring_count = 0
        self.gpu_ms_ring[self.ring_cursor] = v
        self.ring_cursor = (self.ring_cursor + 1) % GPU_SPARKLINE_SAMPLES
        if self.ring_count < GPU_SPARKLINE_SAMPLES:
            self.ring_count += 1

    def seed_baseline_sparkline(mut self, base_ms: Float32 = 4.2):
        """Seeds 60 deterministic subtle GPU frame-time samples (`base_ms +/- 0.25 ms`)
        so headless/gallery captures render a crisp 60-frame sparkline ending at `base_ms`."""
        var b = max(Float32(0.2), base_ms) if isfinite(base_ms) else Float32(4.2)
        self.gpu_ms_ring = List[Float32](length=GPU_SPARKLINE_SAMPLES, fill=b)
        self.ring_cursor = 0
        self.ring_count = 0
        for i in range(GPU_SPARKLINE_SAMPLES):
            if i == GPU_SPARKLINE_SAMPLES - 1:
                self.push_gpu_ms(b)
            else:
                var t = Float32(i) * Float32(0.34)
                var wave = sin(t) * Float32(0.16) + sin(t * Float32(2.3) + Float32(0.8)) * Float32(0.09)
                self.push_gpu_ms(max(Float32(0.1), b + wave))

    def latest_gpu_ms(self) -> Float32:
        if self.ring_count == 0 or len(self.gpu_ms_ring) == 0:
            return Float32(0.0)
        var idx = (self.ring_cursor + len(self.gpu_ms_ring) - 1) % len(self.gpu_ms_ring)
        return self.gpu_ms_ring[idx]

    def contains_point(self, mx: Float32, my: Float32) -> Bool:
        """Returns `True` only when `(mx, my)` hits a floating HUD pill — outside passes through to 3D."""
        for i in range(len(self.hit_rects)):
            if self.hit_rects[i].contains(mx, my):
                return True
        return False

    def _draw_tool_icon_btn(
        self,
        mut dl: UiDrawList,
        theme: StudioTheme,
        mut anim: UiAnimator,
        wid: UInt64,
        r: RectF,
        icon_id: Int,
        active: Bool,
        alpha_scale: Float32,
        mouse_x: Float32,
        mouse_y: Float32,
        mouse_down: Bool,
        mouse_pressed: Bool,
        active_tint: UInt32 = 0,
        disabled: Bool = False,
        draw_active_pill: Bool = False,
    ) -> Bool:
        var hov = (not disabled) and r.contains(mouse_x, mouse_y)
        var st = anim.drive_widget(wid, hov, hov and mouse_down, False)
        var icon_tint = active_tint if active_tint != 0 else theme.accent_primary
        var eff_a = alpha_scale * (Float32(0.36) if disabled else Float32(1.0))
        if draw_active_pill and active:
            dl.add_rect_styled(
                r.x,
                r.y,
                r.w,
                r.h,
                rgba8_scale_alpha(rgba8_with_alpha(icon_tint, 38), eff_a),
                rgba8_scale_alpha(rgba8_with_alpha(icon_tint, 128), eff_a),
                theme.radius_xs,
                Float32(1.0),
                Float32(1.0),
                UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )
        elif hov and not active:
            dl.add_rect(r.x, r.y, r.w, r.h, rgba8_scale_alpha(theme.l3_hover, eff_a * Float32(0.75) * st[0]), theme.radius_xs)
        var col = icon_tint if active else (theme.text_primary if hov else theme.text_secondary)
        var isz = min(Float32(13.5), r.h - Float32(8.0))
        draw_studio_icon(
            dl,
            icon_id,
            r.center_x() - isz * Float32(0.5),
            r.center_y() - isz * Float32(0.5),
            isz,
            rgba8_scale_alpha(col, eff_a),
            Float32(1.45),
        )
        return hov and mouse_pressed

    def draw(
        mut self,
        mut dl: UiDrawList,
        font_slot: Int,
        font: SdfFont,
        theme: StudioTheme,
        mut anim: UiAnimator,
        vp: RectF,
        gizmo_mode: Int,
        gizmo_local_space: Bool,
        snap_enabled: Bool,
        grid_snap_m: Float32,
        pie_mode: Int,
        pie_camera_ejected: Bool,
        cam_free_fly: Bool,
        cam_ortho: Bool,
        show_navmesh: Bool,
        show_colliders: Bool,
        show_grid: Bool,
        cam_right: Vec3,
        cam_up: Vec3,
        cam_fwd: Vec3,
        selected_count: Int,
        drag_delta_text: String,
        mouse_x: Float32,
        mouse_y: Float32,
        mouse_down: Bool,
        mouse_pressed: Bool,
        play_workspace: Bool = False,
    ) -> ViewportHudAction:
        """Emits the floating viewport HUD clusters + PIE viewport border and returns any action."""
        var action = ViewportHudAction()
        self.hit_rects.clear()
        self.hovered_orientation_axis = -1
        self.reserved_top_px = Float32(48.0)

        if vp.w < Float32(120.0) or vp.h < Float32(80.0):
            return action

        dl.push_clip(Int(vp.x), Int(vp.y), Int(vp.right()), Int(vp.bottom()))

        # ---- 0. Refined PIE Active Viewport Perimeter Border ----
        if pie_mode == VHUD_PIE_PLAYING:
            dl.add_glow_rect(
                vp.x + Float32(1.0),
                vp.y + Float32(1.0),
                max(Float32(2.0), vp.w - Float32(2.0)),
                max(Float32(2.0), vp.h - Float32(2.0)),
                UInt32(0),
                rgba8_with_alpha(theme.play_mode, 165),
                Float32(2.0),
                Float32(3.2),
                Float32(1.5),
                UI_FLAG_BORDER,
            )
        elif pie_mode == VHUD_PIE_PAUSED:
            dl.add_rect(
                vp.x + Float32(1.0),
                vp.y + Float32(1.0),
                max(Float32(2.0), vp.w - Float32(2.0)),
                max(Float32(2.0), vp.h - Float32(2.0)),
                UInt32(0),
                Float32(2.0),
                Float32(1.6),
                rgba8_with_alpha(theme.play_paused, 190),
            )

        var in_pie = pie_mode == VHUD_PIE_PLAYING
        var hide_edit_chrome = in_pie and (not pie_camera_ejected)

        # 3-Tier Responsive Density:
        #   tier 0 (Full, >= 610px): Edit, Agents, PIE, Lookdev, Play — 1 single row
        #   tier 1 (Compact, 420..610px): Code+Scene split (~448px) — 1 single row
        #   tier 2 (Narrow, < 420px): 365px narrow / 2x HiDPI — slim 26px wrapped rows
        var compact_tier = 0 if vp.w >= Float32(605.0) else (1 if vp.w >= Float32(420.0) else 2)
        var inset = Float32(10.0) if compact_tier == 0 else (Float32(8.0) if compact_tier == 1 else Float32(6.0))
        var pill_h = Float32(30.0) if compact_tier == 0 else (Float32(28.0) if compact_tier == 1 else Float32(24.0))
        var btn_sz = Float32(22.0) if compact_tier == 0 else Float32(20.0)
        var btn_pad_y = (pill_h - btn_sz) * Float32(0.5)
        var row_gap = Float32(5.0) if compact_tier < 2 else Float32(3.0)
        var max_top_bottom = vp.y + inset

        # ---- 1. Top-Left Cluster: Transform (Q/W/E/R) + Space + Snap (hidden in live non-ejected PIE) ----
        var tl_r = RectF(vp.x + inset, vp.y + inset, Float32(0.0), Float32(0.0))
        if not hide_edit_chrome:
            var sp_w = Float32(46.0) if compact_tier == 0 else (Float32(20.0) if compact_tier == 1 else Float32(48.0))
            var sn_txt = _fmt_f32_2dp(grid_snap_m) + "m"
            var sn_w = Float32(55.0) if compact_tier == 0 else (Float32(20.0) if compact_tier == 1 else Float32(56.0))
            var t_gap = Float32(1.5)
            var tl_w = Float32(4.0) + Float32(4.0) * btn_sz + Float32(3.0) * t_gap + Float32(8.0) + sp_w + Float32(3.0) + sn_w + Float32(4.0)
            tl_r = RectF(vp.x + inset, vp.y + inset, tl_w, pill_h)
            self.hit_rects.append(tl_r)
            max_top_bottom = max(max_top_bottom, tl_r.bottom())

            _draw_hud_glass_pill(dl, theme, tl_r, theme.radius_md, Float32(1.0), theme.border_strong)

            # Sliding active highlight behind Q/W/E/R
            var tool_slot = 0 if gizmo_mode == VHUD_TOOL_SELECT else (1 if gizmo_mode == VHUD_TOOL_MOVE else (2 if gizmo_mode == VHUD_TOOL_ROTATE else 3))
            var btn_y = tl_r.y + btn_pad_y
            var base_x = tl_r.x + Float32(4.0)
            var tool_tgt_x = base_x + Float32(tool_slot) * (btn_sz + t_gap)
            var tool_ind = anim.drive_tab_pill(UInt64(0x56485544544C), tool_tgt_x, btn_sz)
            dl.add_rect_styled(
                tool_ind[0],
                btn_y,
                tool_ind[1],
                btn_sz,
                theme.accent_soft,
                rgba8_with_alpha(theme.accent_primary, 165),
                theme.radius_xs,
                Float32(1.0),
                Float32(1.0),
                UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )

            var tool_icons = List[Int](capacity=4)
            tool_icons.append(ICON_EMPTY)
            tool_icons.append(ICON_MOVE)
            tool_icons.append(ICON_ROTATE)
            tool_icons.append(ICON_SCALE)
            var tool_acts = List[Int](capacity=4)
            tool_acts.append(VHUD_ACT_TOOL_SELECT)
            tool_acts.append(VHUD_ACT_TOOL_MOVE)
            tool_acts.append(VHUD_ACT_TOOL_ROTATE)
            tool_acts.append(VHUD_ACT_TOOL_SCALE)
            for ti in range(4):
                var br = RectF(base_x + Float32(ti) * (btn_sz + t_gap), btn_y, btn_sz, btn_sz)
                if self._draw_tool_icon_btn(
                    dl, theme, anim,
                    _hud_hash(UInt64(0x564855441000), String(ti)),
                    br, tool_icons[ti], ti == tool_slot, Float32(1.0),
                    mouse_x, mouse_y, mouse_down, mouse_pressed,
                ):
                    action.kind = tool_acts[ti]

            # Hairline divider + World/Local raised toggle chip
            var div1_x = base_x + Float32(4.0) * btn_sz + Float32(3.0) * t_gap + Float32(3.5)
            dl.add_line(div1_x, tl_r.y + Float32(6.0), div1_x, tl_r.bottom() - Float32(6.0), Float32(1.0), theme.border_subtle)

            var sp_r = RectF(div1_x + Float32(4.5), btn_y, sp_w, btn_sz)
            var sp_hov = sp_r.contains(mouse_x, mouse_y)
            var sp_bg = theme.accent_soft if gizmo_local_space else (theme.l3_hover if sp_hov else rgba8_with_alpha(theme.l2_card, 145))
            var sp_bdr = rgba8_with_alpha(theme.accent_primary, 140) if gizmo_local_space else (theme.border_control if sp_hov else theme.border_subtle)
            dl.add_rect_styled(sp_r.x, sp_r.y, sp_r.w, sp_r.h, sp_bg, sp_bdr, theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL)
            var sp_col = theme.accent_primary if gizmo_local_space else (theme.text_primary if sp_hov else theme.text_secondary)
            if compact_tier != 1:
                draw_studio_icon(dl, ICON_LOCAL if gizmo_local_space else ICON_WORLD, sp_r.x + Float32(4.0), sp_r.center_y() - Float32(5.5), Float32(11.5), sp_col, Float32(1.35))
                _draw_hud_text_cy(dl, font_slot, font, "LOC" if gizmo_local_space else "WLD", sp_r.x + Float32(18.0), sp_r.center_y(), Float32(10.0), theme.text_primary)
            else:
                draw_studio_icon(dl, ICON_LOCAL if gizmo_local_space else ICON_WORLD, sp_r.center_x() - Float32(6.0), sp_r.center_y() - Float32(6.0), Float32(12.0), sp_col, Float32(1.35))
            if sp_hov and mouse_pressed:
                action.kind = VHUD_ACT_TOGGLE_SPACE

            # Snap raised toggle chip with grid step readout
            var sn_r = RectF(sp_r.right() + Float32(3.0), btn_y, sn_w, btn_sz)
            var sn_hov = sn_r.contains(mouse_x, mouse_y)
            var sn_bg = theme.accent_soft if snap_enabled else (theme.l3_hover if sn_hov else rgba8_with_alpha(theme.l2_card, 145))
            var sn_bdr = rgba8_with_alpha(theme.accent_primary, 155) if snap_enabled else (theme.border_control if sn_hov else theme.border_subtle)
            dl.add_rect_styled(sn_r.x, sn_r.y, sn_r.w, sn_r.h, sn_bg, sn_bdr, theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL)
            var sn_col = theme.accent_primary if snap_enabled else (theme.text_secondary if sn_hov else theme.text_muted)
            if compact_tier != 1:
                draw_studio_icon(dl, ICON_SNAP, sn_r.x + Float32(4.0), sn_r.center_y() - Float32(5.5), Float32(11.5), sn_col, Float32(1.35))
                _draw_hud_text_cy(dl, font_slot, font, sn_txt, sn_r.x + Float32(17.5), sn_r.center_y(), Float32(10.0), theme.text_primary if snap_enabled else theme.text_muted)
            else:
                draw_studio_icon(dl, ICON_SNAP, sn_r.center_x() - Float32(6.0), sn_r.center_y() - Float32(6.0), Float32(12.0), sn_col, Float32(1.35))
            if sn_hov and mouse_pressed:
                action.kind = VHUD_ACT_TOGGLE_SNAP

        # ---- 2. Top-Center Cluster: Transport & PIE Pill ----
        var badge_txt = "LIVE" if pie_mode == VHUD_PIE_PLAYING else ("PAUSED" if pie_mode == VHUD_PIE_PAUSED else "EDIT")
        var badge_txt_w = _measure_text_hud(font, badge_txt, Float32(10.0))
        var badge_w = max(Float32(42.0) if compact_tier != 2 else Float32(45.0), badge_txt_w + Float32(21.0))
        var ej_w = Float32(54.0) if pie_mode != VHUD_PIE_STOPPED else Float32(0.0)
        var t_step = btn_sz + Float32(1.5)
        var tc_w = Float32(4.5) + Float32(4.0) * t_step + Float32(4.0) + badge_w + (Float32(4.5) + ej_w if ej_w > Float32(0.0) else Float32(0.0)) + Float32(4.5)

        var tc_ideal_x = vp.center_x() - tc_w * Float32(0.5)
        var tc_x = tc_ideal_x if hide_edit_chrome else max(tl_r.right() + Float32(6.0), tc_ideal_x)
        var tc_y = vp.y + inset
        if (not hide_edit_chrome) and (tc_x + tc_w > vp.right() - inset):
            tc_x = tl_r.x
            tc_y = tl_r.bottom() + row_gap
        var tc_r = RectF(tc_x, tc_y, tc_w, pill_h)
        if tc_r.right() <= vp.right() - inset + Float32(2.0):
            self.hit_rects.append(tc_r)
            max_top_bottom = max(max_top_bottom, tc_r.bottom())
            var tc_bdr = (
                rgba8_with_alpha(theme.play_mode, 195)
                if pie_mode == VHUD_PIE_PLAYING
                else (rgba8_with_alpha(theme.play_paused, 195) if pie_mode == VHUD_PIE_PAUSED else theme.border_strong)
            )
            _draw_hud_glass_pill(dl, theme, tc_r, theme.radius_md, Float32(1.0), tc_bdr, Float32(1.2) if pie_mode != VHUD_PIE_STOPPED else Float32(1.0))

            var transport_y = tc_r.y + btn_pad_y
            var tx = tc_r.x + Float32(4.5)
            var is_stopped = pie_mode == VHUD_PIE_STOPPED

            # Play (always actionable; soft active pill when LIVE)
            var r_play = RectF(tx, transport_y, btn_sz, btn_sz)
            if self._draw_tool_icon_btn(
                dl, theme, anim, UInt64(0x564855442001), r_play, ICON_PLAY,
                pie_mode == VHUD_PIE_PLAYING, Float32(1.0), mouse_x, mouse_y, mouse_down, mouse_pressed,
                theme.play_mode, False, True,
            ):
                action.kind = VHUD_ACT_PIE_PLAY
            tx += t_step

            # Pause (dimmed when stopped)
            var r_pause = RectF(tx, transport_y, btn_sz, btn_sz)
            if self._draw_tool_icon_btn(
                dl, theme, anim, UInt64(0x564855442002), r_pause, ICON_PAUSE,
                pie_mode == VHUD_PIE_PAUSED, Float32(1.0), mouse_x, mouse_y, mouse_down, mouse_pressed,
                theme.play_paused, is_stopped, True,
            ):
                action.kind = VHUD_ACT_PIE_PAUSE
            tx += t_step

            # Step (dimmed when stopped)
            var r_step = RectF(tx, transport_y, btn_sz, btn_sz)
            if self._draw_tool_icon_btn(
                dl, theme, anim, UInt64(0x564855442003), r_step, ICON_STEP,
                False, Float32(1.0), mouse_x, mouse_y, mouse_down, mouse_pressed,
                UInt32(0), is_stopped, False,
            ):
                action.kind = VHUD_ACT_PIE_STEP
            tx += t_step

            # Stop (dimmed when stopped, crimson when live/paused)
            var r_stop = RectF(tx, transport_y, btn_sz, btn_sz)
            if self._draw_tool_icon_btn(
                dl, theme, anim, UInt64(0x564855442004), r_stop, ICON_STOP,
                not is_stopped, Float32(1.0), mouse_x, mouse_y, mouse_down, mouse_pressed,
                theme.status_danger, is_stopped, False,
            ):
                action.kind = VHUD_ACT_PIE_STOP
            tx += t_step + Float32(4.0)

            # Read-only Mode status badge (recessed flat well — distinct from raised toggles!)
            var badge_col = theme.play_mode if pie_mode == VHUD_PIE_PLAYING else (theme.play_paused if pie_mode == VHUD_PIE_PAUSED else theme.text_muted)
            var badge_h = Float32(19.0)
            var badge_y = tc_r.center_y() - badge_h * Float32(0.5)
            var badge_bg = rgba8_with_alpha(badge_col, 26) if not is_stopped else rgba8_with_alpha(theme.l0_canvas, 170)
            var badge_bdr = rgba8_with_alpha(badge_col, 96) if not is_stopped else theme.border_subtle
            dl.add_recessed_well(tx, badge_y, badge_w, badge_h, badge_bg, badge_bdr, badge_h * Float32(0.5), Float32(1.0))
            if pie_mode == VHUD_PIE_PLAYING:
                # Subtle soft live pulse halo
                dl.add_circle(tx + Float32(9.0), tc_r.center_y(), Float32(5.0), rgba8_with_alpha(badge_col, 48))
            dl.add_circle(tx + Float32(9.0), tc_r.center_y(), Float32(2.7), badge_col)
            _draw_hud_text_cy(dl, font_slot, font, badge_txt, tx + Float32(15.5), tc_r.center_y(), Float32(10.0), theme.text_primary if not is_stopped else theme.text_secondary)

            if pie_mode != VHUD_PIE_STOPPED:
                var ej_r = RectF(tx + badge_w + Float32(5.0), transport_y, ej_w, btn_sz)
                var ej_hov = ej_r.contains(mouse_x, mouse_y)
                var ej_bg = theme.accent_soft if pie_camera_ejected else (theme.l3_hover if ej_hov else theme.l2_card)
                var ej_bdr = rgba8_with_alpha(theme.accent_primary, 160) if pie_camera_ejected else (theme.border_strong if ej_hov else theme.border_control)
                dl.add_rect_styled(ej_r.x, ej_r.y, ej_r.w, ej_r.h, ej_bg, ej_bdr, theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL)
                var ej_col = theme.accent_primary if pie_camera_ejected else theme.text_primary
                _draw_eject_icon(dl, ej_r.x + Float32(5.0), ej_r.center_y() - Float32(5.0), Float32(10.0), ej_col)
                _draw_hud_text_cy(dl, font_slot, font, "Eject", ej_r.x + Float32(18.0), ej_r.center_y(), Float32(10.5), ej_col)
                if ej_hov and mouse_pressed:
                    action.kind = VHUD_ACT_PIE_EJECT

        # ---- 3. Top-Right Cluster: Camera / Overlays / Area-Filled 60-Frame GPU Sparkline ----
        var show_spark_graph = (compact_tier != 1) or hide_edit_chrome
        var tr_w: Float32
        if hide_edit_chrome:
            # Clean minimal telemetry pill during live Play/PIE (4.2ms + crisp sparkline)
            tr_w = Float32(106.0)
        elif compact_tier == 1:
            # Compact 1-row tier (Code+Scene ~446px): Cam + Grid + numeric ms
            tr_w = Float32(104.0)
        elif compact_tier == 2:
            # Narrow tier (< 420px): Cam + Grid + Col + Nav + numeric ms + compact graph
            tr_w = Float32(176.0)
        else:
            # Full tier (>= 605px): Cam + Grid + Col + Nav + ms + area sparkline
            tr_w = Float32(198.0)

        var tr_r = RectF(vp.right() - tr_w - inset, vp.y + inset, tr_w, pill_h)
        var collides_tc = (tc_r.w > Float32(0.0)) and (tr_r.x < tc_r.right() + Float32(6.0))
        var collides_tl = (not hide_edit_chrome) and (tr_r.x < tl_r.right() + Float32(6.0))
        if collides_tc or collides_tl:
            tr_r.y = max(tl_r.bottom(), tc_r.bottom()) + row_gap
        if tr_r.x >= vp.x and tr_r.bottom() <= vp.bottom() - inset:
            self.hit_rects.append(tr_r)
            max_top_bottom = max(max_top_bottom, tr_r.bottom())
            var tr_hov = tr_r.contains(mouse_x, mouse_y)
            var tr_alpha = Float32(0.88) if (hide_edit_chrome and not tr_hov) else Float32(1.0)

            _draw_hud_glass_pill(dl, theme, tr_r, theme.radius_md, tr_alpha, theme.border_strong)

            var overlay_y = tr_r.y + btn_pad_y
            var cx = tr_r.x + Float32(4.0)

            if not hide_edit_chrome:
                # Camera Orbit/Fly button
                var cam_w = Float32(46.0) if compact_tier == 0 else (Float32(41.0) if compact_tier == 1 else Float32(42.0))
                var cam_r = RectF(cx, overlay_y, cam_w, btn_sz)
                var cam_hov = cam_r.contains(mouse_x, mouse_y)
                var cam_bg = theme.accent_soft if cam_free_fly else (theme.l3_hover if cam_hov else rgba8_with_alpha(theme.l2_card, 130))
                var cam_bdr = rgba8_with_alpha(theme.accent_primary, 135) if cam_free_fly else (theme.border_control if cam_hov else theme.border_subtle)
                dl.add_rect_styled(cam_r.x, cam_r.y, cam_r.w, cam_r.h, rgba8_scale_alpha(cam_bg, tr_alpha), rgba8_scale_alpha(cam_bdr, tr_alpha), theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL)
                var cam_col = theme.accent_primary if cam_free_fly else theme.text_primary
                draw_studio_icon(dl, ICON_CAMERA, cam_r.x + Float32(3.5), cam_r.center_y() - Float32(5.5), Float32(11.0), rgba8_scale_alpha(cam_col, tr_alpha), Float32(1.35))
                _draw_hud_text_cy(dl, font_slot, font, "FLY" if cam_free_fly else "ORB", cam_r.x + Float32(17.0), cam_r.center_y(), Float32(10.0), rgba8_scale_alpha(theme.text_primary, tr_alpha))
                if cam_hov and mouse_pressed:
                    action.kind = VHUD_ACT_CAM_MODE
                cx += cam_w + Float32(2.5)

                # Unified overlay toggle icon buttons (Grid, Collider, vector NavMesh)
                var ov_w = Float32(21.0) if compact_tier == 0 else Float32(19.0)
                var g_r = RectF(cx, overlay_y, ov_w, btn_sz)
                if self._draw_tool_icon_btn(dl, theme, anim, UInt64(0x564855443001), g_r, ICON_GRID, show_grid, tr_alpha, mouse_x, mouse_y, mouse_down, mouse_pressed, UInt32(0), False, True):
                    action.kind = VHUD_ACT_TOGGLE_GRID
                cx += ov_w + Float32(2.0)

                if compact_tier != 1:
                    var c_r = RectF(cx, overlay_y, ov_w, btn_sz)
                    if self._draw_tool_icon_btn(dl, theme, anim, UInt64(0x564855443002), c_r, ICON_COLLIDER, show_colliders, tr_alpha, mouse_x, mouse_y, mouse_down, mouse_pressed, UInt32(0), False, True):
                        action.kind = VHUD_ACT_TOGGLE_COLLIDERS
                    cx += ov_w + Float32(2.0)

                    # Vector NavMesh toggle with unified active pill treatment
                    var nav_r = RectF(cx, overlay_y, ov_w, btn_sz)
                    var nav_hov = nav_r.contains(mouse_x, mouse_y)
                    if show_navmesh:
                        dl.add_rect_styled(
                            nav_r.x, nav_r.y, nav_r.w, nav_r.h,
                            rgba8_scale_alpha(rgba8_with_alpha(theme.play_mode, 36), tr_alpha),
                            rgba8_scale_alpha(rgba8_with_alpha(theme.play_mode, 128), tr_alpha),
                            theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
                        )
                    elif nav_hov:
                        dl.add_rect(nav_r.x, nav_r.y, nav_r.w, nav_r.h, rgba8_scale_alpha(theme.l3_hover, tr_alpha * Float32(0.75)), theme.radius_xs)
                    var nav_col = theme.play_mode if show_navmesh else (theme.text_primary if nav_hov else theme.text_secondary)
                    _draw_navmesh_icon(dl, nav_r.center_x() - Float32(6.0), nav_r.center_y() - Float32(6.0), Float32(12.0), rgba8_scale_alpha(nav_col, tr_alpha))
                    if nav_hov and mouse_pressed:
                        action.kind = VHUD_ACT_TOGGLE_NAVMESH
                    cx += ov_w + Float32(3.0)

                # Hairline divider before perf telemetry
                dl.add_line(cx, tr_r.y + Float32(6.0), cx, tr_r.bottom() - Float32(6.0), Float32(1.0), rgba8_scale_alpha(theme.border_subtle, tr_alpha))
                cx += Float32(4.0)

            var perf_r = RectF(cx, overlay_y, max(Float32(34.0), tr_r.right() - cx - Float32(4.0)), btn_sz)
            var perf_hov = perf_r.contains(mouse_x, mouse_y)
            if perf_hov or self.perf_expanded:
                dl.add_rect(perf_r.x, perf_r.y, perf_r.w, perf_r.h, rgba8_scale_alpha(theme.l3_hover, tr_alpha * Float32(0.7)), theme.radius_xs)
            var cur_ms = self.latest_gpu_ms()
            var perf_col = theme.agent_flow if cur_ms < Float32(8.3) else (theme.status_warn if cur_ms < Float32(16.6) else theme.status_danger)
            var ms_str = _fmt_f32_1dp(cur_ms) + "ms"
            _draw_hud_text_cy(dl, font_slot, font, ms_str, perf_r.x + Float32(2.0), perf_r.center_y(), Float32(10.0), rgba8_scale_alpha(theme.text_primary, tr_alpha))

            # Area-filled 60-frame sparkline with subtle reference line & live head dot
            var n_pts = self.ring_count
            if show_spark_graph and perf_r.w >= Float32(58.0):
                var sp_x = perf_r.x + Float32(37.0)
                var sp_w = max(Float32(18.0), perf_r.right() - sp_x - Float32(2.0))
                var sp_y = perf_r.y + Float32(2.5)
                var sp_h = max(Float32(10.0), perf_r.h - Float32(5.0))
                dl.add_recessed_well(
                    sp_x,
                    sp_y,
                    sp_w,
                    sp_h,
                    rgba8_scale_alpha(theme.l0_canvas, tr_alpha * Float32(0.88)),
                    rgba8_scale_alpha(theme.border_subtle, tr_alpha),
                    Float32(3.5),
                    Float32(1.0),
                )
                # Subtle mid-band reference hairline
                var ref_y = sp_y + sp_h * Float32(0.5)
                dl.add_line(sp_x + Float32(2.0), ref_y, sp_x + sp_w - Float32(2.0), ref_y, Float32(0.8), rgba8_scale_alpha(theme.border_subtle, tr_alpha * Float32(0.85)))
                if n_pts >= 2:
                    var lo_ms = Float32(1.0e9)
                    var hi_ms = Float32(0.0)
                    for si in range(n_pts):
                        var idx = (self.ring_cursor - n_pts + si + GPU_SPARKLINE_SAMPLES) % GPU_SPARKLINE_SAMPLES
                        var sv = self.gpu_ms_ring[idx]
                        if sv < lo_ms:
                            lo_ms = sv
                        if sv > hi_ms:
                            hi_ms = sv
                    var span = max(Float32(0.45), hi_ms - lo_ms)
                    var band_lo = max(Float32(0.0), lo_ms - span * Float32(0.55))
                    var band_hi = max(band_lo + Float32(0.8), hi_ms + span * Float32(0.45))
                    var inv_span = Float32(1.0) / max(Float32(0.1), band_hi - band_lo)
                    var step_x = (sp_w - Float32(5.0)) / Float32(max(1, GPU_SPARKLINE_SAMPLES - 1))
                    var bot_y = sp_y + sp_h - Float32(1.5)
                    var col_fill = rgba8_scale_alpha(rgba8_with_alpha(perf_col, 42), tr_alpha)
                    var col_line = rgba8_scale_alpha(perf_col, tr_alpha)

                    # Translucent area fill under the sparkline curve (sampled at <= 16 buckets for lean prim budget)
                    var poly_xy = List[Float32](capacity=40)
                    var first_x = sp_x + Float32(2.0) + Float32(GPU_SPARKLINE_SAMPLES - n_pts) * step_x
                    poly_xy.append(first_x)
                    poly_xy.append(bot_y)
                    var stride = max(1, n_pts // 14)
                    var last_px = first_x
                    for si in range(0, n_pts, stride):
                        var idx = (self.ring_cursor - n_pts + si + GPU_SPARKLINE_SAMPLES) % GPU_SPARKLINE_SAMPLES
                        var norm = max(Float32(0.0), min(Float32(1.0), (self.gpu_ms_ring[idx] - band_lo) * inv_span))
                        var px = sp_x + Float32(2.0) + Float32(GPU_SPARKLINE_SAMPLES - n_pts + si) * step_x
                        var py = sp_y + sp_h - Float32(2.5) - norm * (sp_h - Float32(5.0))
                        poly_xy.append(px)
                        poly_xy.append(py)
                        last_px = px
                    # Always include exact final point
                    var fin_idx = (self.ring_cursor - 1 + GPU_SPARKLINE_SAMPLES) % GPU_SPARKLINE_SAMPLES
                    var fin_norm = max(Float32(0.0), min(Float32(1.0), (self.gpu_ms_ring[fin_idx] - band_lo) * inv_span))
                    var fin_px = sp_x + Float32(2.0) + Float32(GPU_SPARKLINE_SAMPLES - 1) * step_x
                    var fin_py = sp_y + sp_h - Float32(2.5) - fin_norm * (sp_h - Float32(5.0))
                    if fin_px > last_px + Float32(0.2):
                        poly_xy.append(fin_px)
                        poly_xy.append(fin_py)
                        last_px = fin_px
                    poly_xy.append(last_px)
                    poly_xy.append(bot_y)
                    var pcnt = List[Int](capacity=1)
                    pcnt.append(len(poly_xy) // 2)
                    _ = dl.add_polygon(poly_xy, pcnt, col_fill)

                    # Crisp analytical polyline + glowing head dot
                    var prev_px = Float32(0.0)
                    var prev_py = Float32(0.0)
                    for si in range(n_pts):
                        var idx = (self.ring_cursor - n_pts + si + GPU_SPARKLINE_SAMPLES) % GPU_SPARKLINE_SAMPLES
                        var norm = max(Float32(0.0), min(Float32(1.0), (self.gpu_ms_ring[idx] - band_lo) * inv_span))
                        var px = sp_x + Float32(2.0) + Float32(GPU_SPARKLINE_SAMPLES - n_pts + si) * step_x
                        var py = sp_y + sp_h - Float32(2.5) - norm * (sp_h - Float32(5.0))
                        if si > 0:
                            dl.add_line(prev_px, prev_py, px, py, Float32(1.35), col_line)
                        prev_px = px
                        prev_py = py
                    dl.add_circle(prev_px, prev_py, Float32(2.0), rgba8_scale_alpha(theme.text_primary, tr_alpha))

            if perf_hov and mouse_pressed:
                self.perf_expanded = not self.perf_expanded
                action.kind = VHUD_ACT_TOGGLE_PERF

            # Optional expanded perf breakdown tooltip
            if self.perf_expanded:
                var pop_r = RectF(tr_r.right() - Float32(168.0), tr_r.bottom() + Float32(6.0), Float32(168.0), Float32(52.0))
                self.hit_rects.append(pop_r)
                max_top_bottom = max(max_top_bottom, pop_r.bottom())
                _draw_hud_glass_pill(dl, theme, pop_r, theme.radius_sm, Float32(1.0), theme.border_strong)
                var fps = Float32(1000.0) / max(Float32(0.1), cur_ms) if cur_ms > Float32(0.01) else Float32(0.0)
                _draw_hud_text_cy(dl, font_slot, font, "GPU Frame: " + _fmt_f32_2dp(cur_ms) + " ms", pop_r.x + Float32(10.0), pop_r.y + Float32(16.0), Float32(11.0), theme.text_primary)
                _draw_hud_text_cy(dl, font_slot, font, "Est. FPS: " + _fmt_f32_1dp(fps) + " (" + String(n_pts) + "f)", pop_r.x + Float32(10.0), pop_r.y + Float32(36.0), Float32(10.5), theme.text_secondary)

        self.reserved_top_px = max(Float32(48.0), max_top_bottom - vp.y + Float32(8.0))

        # ---- 4. Bottom-Right Cluster: Interactive 3D OrientationGizmo + Matched Persp/SEL Chips ----
        # Exact center preserved for hit-test stability: (vp.right() - 48, vp.bottom() - 62), r = 27
        var giz_r_px = Float32(27.0)
        var giz_cx = vp.right() - Float32(48.0)
        var giz_cy = vp.bottom() - Float32(62.0)
        var disc_r = Float32(33.0)
        var chip_h = Float32(20.0)
        var chip_y = giz_cy + disc_r + Float32(1.0)  # vp.bottom() - 28 -> bottom = vp.bottom() - 8px safe margin!
        var giz_box = RectF(giz_cx - Float32(36.0), giz_cy - Float32(36.0), Float32(72.0), Float32(94.0))

        var hide_compass = play_workspace and hide_edit_chrome
        if not hide_compass:
            self.hit_rects.append(giz_box)
            var br_hov = giz_box.contains(mouse_x, mouse_y)
            var br_dim_tgt = Float32(0.0) if (not in_pie or br_hov) else Float32(1.0)
            var br_dim = anim.drive_aux(
                UInt64(0x564855440004),
                br_dim_tgt,
                SpringConfig.snappy(),
            )
            if anim.mode == UI_ANIM_INSTANT:
                br_dim = br_dim_tgt
            var br_alpha = Float32(1.0) - Float32(0.52) * br_dim

            # Smoky frosted backing disc with subtle dark outer contrast ring
            dl.add_circle(
                giz_cx,
                giz_cy,
                disc_r + Float32(0.8),
                UInt32(0),
                Float32(1.0),
                rgba8(6, 8, 12, Int(Float32(110.0) * br_alpha)),
            )
            dl.add_circle(
                giz_cx,
                giz_cy,
                disc_r,
                rgba8_scale_alpha(rgba8_with_alpha(theme.l4_overlay, 186), br_alpha),
                Float32(1.0),
                rgba8_scale_alpha(theme.border_strong if br_hov else theme.border_control, br_alpha),
            )

            # Project 6 axes into viewport 2D using camera orthonormal basis
            var rn = cam_right.normalized()
            var un = cam_up.normalized()
            var fn_v = cam_fwd.normalized()
            var raw_dirs = List[Vec3](capacity=6)
            raw_dirs.append(Vec3(1.0, 0.0, 0.0))
            raw_dirs.append(Vec3(-1.0, 0.0, 0.0))
            raw_dirs.append(Vec3(0.0, 1.0, 0.0))
            raw_dirs.append(Vec3(0.0, -1.0, 0.0))
            raw_dirs.append(Vec3(0.0, 0.0, 1.0))
            raw_dirs.append(Vec3(0.0, 0.0, -1.0))

            var raw_cols = List[UInt32](capacity=6)
            raw_cols.append(theme.axis_x)
            raw_cols.append(theme.axis_x)
            raw_cols.append(theme.axis_y)
            raw_cols.append(theme.axis_y)
            raw_cols.append(theme.axis_z)
            raw_cols.append(theme.axis_z)

            var raw_lbls = List[String](capacity=6)
            raw_lbls.append("X")
            raw_lbls.append("-X")
            raw_lbls.append("Y")
            raw_lbls.append("-Y")
            raw_lbls.append("Z")
            raw_lbls.append("-Z")

            var projs = List[_AxisProj](capacity=6)
            for ai in range(6):
                var d = raw_dirs[ai]
                var px = giz_cx + d.dot(rn) * giz_r_px
                var py = giz_cy - d.dot(un) * giz_r_px
                var depth = d.dot(fn_v)  # higher = farther into screen; lower = closer to camera
                var snap_fwd = Vec3(-d.x, -d.y, -d.z)
                projs.append(_AxisProj(ai, px, py, depth, (ai % 2) == 0, raw_cols[ai], raw_lbls[ai], snap_fwd))

            # Sort back-to-front (largest depth first, smallest depth last)
            for i in range(len(projs)):
                for j in range(i + 1, len(projs)):
                    if projs[j].depth > projs[i].depth:
                        var tmp = projs[i]
                        projs[i] = projs[j]
                        projs[j] = tmp

            # Hit-test front-to-back
            var hit_axis = -1
            var k = len(projs) - 1
            while k >= 0:
                var dx = mouse_x - projs[k].sx
                var dy = mouse_y - projs[k].sy
                var hit_r = Float32(9.5) if projs[k].positive else Float32(7.5)
                if dx * dx + dy * dy <= hit_r * hit_r:
                    hit_axis = projs[k].idx
                    break
                k -= 1
            self.hovered_orientation_axis = hit_axis

            # Subtle center origin hub dot
            dl.add_circle(giz_cx, giz_cy, Float32(2.2), rgba8_scale_alpha(theme.text_muted, br_alpha * Float32(0.75)))

            # Emit trimmed stalks & spheres back-to-front
            for i in range(len(projs)):
                var p = projs[i]
                var is_hov = p.idx == hit_axis
                var ht = anim.drive_widget(_hud_hash(UInt64(0x4F524947), String(p.idx)), is_hov, is_hov and mouse_down, False)
                var depth_dim = Float32(1.0) if p.depth <= Float32(0.0) else Float32(0.68)
                var col = rgba8_scale_alpha(p.color, br_alpha * depth_dim)
                var vx = p.sx - giz_cx
                var vy = p.sy - giz_cy
                var vlen = sqrt(vx * vx + vy * vy)
                if p.positive:
                    var sr = Float32(7.0) + ht[0] * Float32(1.6)
                    if vlen > sr + Float32(2.5):
                        var ux = vx / vlen
                        var uy = vy / vlen
                        dl.add_line(
                            giz_cx + ux * Float32(2.4),
                            giz_cy + uy * Float32(2.4),
                            p.sx - ux * (sr - Float32(0.5)),
                            p.sy - uy * (sr - Float32(0.5)),
                            Float32(2.0),
                            col,
                        )
                    dl.add_circle(
                        p.sx,
                        p.sy,
                        sr,
                        col,
                        Float32(1.4) if is_hov else Float32(0.0),
                        rgba8_scale_alpha(theme.text_primary, br_alpha),
                    )
                    var lbl_sz = Float32(9.2)
                    var lbl_w = _measure_text_hud(font, p.label, lbl_sz)
                    _draw_hud_text_cy(
                        dl,
                        font_slot,
                        font,
                        p.label,
                        p.sx - lbl_w * Float32(0.5),
                        p.sy,
                        lbl_sz,
                        rgba8_scale_alpha(theme.text_on_accent, br_alpha),
                    )
                else:
                    var sr = Float32(4.4) + ht[0] * Float32(2.2)
                    if vlen > sr + Float32(2.5):
                        var ux = vx / vlen
                        var uy = vy / vlen
                        dl.add_line(
                            giz_cx + ux * Float32(2.4),
                            giz_cy + uy * Float32(2.4),
                            p.sx - ux * sr,
                            p.sy - uy * sr,
                            Float32(1.0),
                            rgba8_scale_alpha(p.color, br_alpha * Float32(0.28)),
                        )
                    var fill_neg = rgba8_scale_alpha(p.color, br_alpha * (Float32(0.92) if is_hov else Float32(0.26)))
                    dl.add_circle(p.sx, p.sy, sr, fill_neg, Float32(1.15), col)
                    if is_hov:
                        var nsz = Float32(8.0)
                        var nlw = _measure_text_hud(font, p.label, nsz)
                        _draw_hud_text_cy(dl, font_slot, font, p.label, p.sx - nlw * Float32(0.5), p.sy, nsz, rgba8_scale_alpha(theme.text_on_accent, br_alpha))
                if is_hov and mouse_pressed:
                    action = ViewportHudAction(VHUD_ACT_SNAP_VIEW, p.snap_fwd, p.idx)

            # Persp / Ortho toggle pill beneath the triad (clean 8px bottom safe margin!)
            var po_txt = String("Ortho") if cam_ortho else String("Persp")
            var po_tw = _measure_text_hud(font, po_txt, Float32(10.0))
            var po_w = max(Float32(52.0), po_tw + Float32(16.0))
            var po_r = RectF(giz_cx - po_w * Float32(0.5), chip_y, po_w, chip_h)
            var po_hov = po_r.contains(mouse_x, mouse_y)
            _draw_hud_glass_pill(
                dl,
                theme,
                po_r,
                chip_h * Float32(0.5),
                br_alpha,
                theme.accent_primary if cam_ortho else (theme.border_strong if po_hov else theme.border_control),
            )
            _draw_hud_text_cy(
                dl,
                font_slot,
                font,
                po_txt,
                po_r.center_x() - po_tw * Float32(0.5),
                po_r.center_y(),
                Float32(10.0),
                rgba8_scale_alpha(theme.accent_primary if cam_ortho else theme.text_primary, br_alpha),
            )
            if po_hov and mouse_pressed:
                action.kind = VHUD_ACT_TOGGLE_ORTHO

            # Matched height & margin SEL / live gizmo delta badge left of Persp (hidden in live PIE)
            if not hide_edit_chrome:
                var is_delta = drag_delta_text.byte_length() > 0
                var info_txt = drag_delta_text if is_delta else ("SEL " + String(selected_count))
                var info_tw = _measure_text_hud(font, info_txt, Float32(10.0))
                var info_w = max(Float32(48.0), info_tw + Float32(16.0))
                var info_r = RectF(po_r.x - Float32(6.0) - info_w, chip_y, info_w, chip_h)
                if info_r.x > vp.x + inset:
                    self.hit_rects.append(info_r)
                    _draw_hud_glass_pill(
                        dl,
                        theme,
                        info_r,
                        chip_h * Float32(0.5),
                        br_alpha,
                        theme.accent_primary if is_delta else theme.border_subtle,
                    )
                    _draw_hud_text_cy(
                        dl,
                        font_slot,
                        font,
                        info_txt,
                        info_r.center_x() - info_tw * Float32(0.5),
                        info_r.center_y(),
                        Float32(10.0),
                        rgba8_scale_alpha(theme.accent_primary if is_delta else theme.text_secondary, br_alpha),
                    )

        dl.pop_clip()
        return action
