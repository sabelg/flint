"""Flint Studio Unified Header Bar & Floating Translucent Viewport HUD (`flint.studio.viewport_hud`).

Pure-Mojo 1.2 UI rendering & hit-testing for:
  - `StudioHeaderBar` (`32 px` slim unified chrome bar): brand mark, scene title + dirty dot,
    animated Workspace Preset switcher (`Edit`, `Agent Studio`, `Code + Scene`, `Lookdev`, `Play`),
    `Ctrl+K` Command Palette pill, live Agent Status pill with pulse ring, Theme toggle, and
    Save / Build status chips.
  - `ViewportHud`: 4 floating translucent frosted-glass HUD clusters inside `SceneViewport`
    with exact per-pill hit-rect tracking so clicks outside pass straight through to the 3D scene:
      1. Top-Left: Transform (`Q/W/E/R`) + Space (`World/Local`) + Snap pill
      2. Top-Center: Transport (`Play/Pause/Step/Stop` + `Eject`) + PIE emerald/amber border
      3. Top-Right: Camera (`Orbit/Fly`) + Overlays + 60-frame GPU sparkline pill
      4. Bottom-Right: Interactive 3D depth-sorted `OrientationGizmo` (`±X/±Y/±Z` camera snap + Persp/Ortho)
"""
from std.collections import List
from std.math import isfinite, sin
from flint.core.math import Vec3
from flint.render.font import SdfFont
from flint.text.utf8 import decode_utf8, encode_utf8
from flint.ui.animator import SpringConfig, UI_ANIM_INSTANT, UiAnimator
from flint.ui.draw_list import (
    UI_FLAG_BORDER,
    UI_FLAG_TOP_BEVEL,
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
    ICON_WORLD,
    StudioTheme,
    lerp_rgba8,
    rgba8_scale_alpha,
    rgba8_with_alpha,
)

# Unified Header Bar geometry & Workspace Presets
comptime STUDIO_HEADER_BAR_H: Float32 = 32.0

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
# 1. Slim Unified Studio Header Bar (`32 px`)
# ============================================================================

@fieldwise_init
struct HeaderBarAction(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var int_val: Int

    def __init__(out self):
        self.kind = HDR_ACT_NONE
        self.int_val = 0


struct StudioHeaderBar(Copyable, Movable):
    """Slim `32 px` unified Studio header bar with animated Workspace Preset switcher."""
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
    ) -> HeaderBarAction:
        """Emits the `32 px` Studio header bar and returns any triggered `HeaderBarAction`."""
        var action = HeaderBarAction()
        var sw = max(Float32(360.0), screen_w if isfinite(screen_w) else Float32(1280.0))
        self.rect = RectF(Float32(0.0), y_offset, sw, STUDIO_HEADER_BAR_H)
        self.active_workspace = max(0, min(WORKSPACE_COUNT - 1, active_workspace))
        self.hovered = self.rect.contains(mouse_x, mouse_y)
        self.last_hit_rect = self.rect

        # Background bar + hairline bottom border
        dl.add_rect(self.rect.x, self.rect.y, self.rect.w, self.rect.h, theme.l1_shell)
        dl.add_line(
            self.rect.x,
            self.rect.bottom() - Float32(0.5),
            self.rect.right(),
            self.rect.bottom() - Float32(0.5),
            Float32(1.0),
            theme.border_subtle,
        )

        var cy = self.rect.y + self.rect.h * Float32(0.5)
        var text_sz = Float32(12.0)

        # ---- Left Zone: Brand mark + Scene title + Dirty dot ----
        var lx = self.rect.x + Float32(10.0)
        var brand_r = RectF(lx, self.rect.y + Float32(4.0), Float32(68.0), Float32(24.0))
        var brand_hov = brand_r.contains(mouse_x, mouse_y)
        if brand_hov and mouse_pressed:
            action = HeaderBarAction(HDR_ACT_MENU, 0)
        if brand_hov:
            dl.add_rect(brand_r.x, brand_r.y, brand_r.w, brand_r.h, theme.l3_hover, theme.radius_xs)
        draw_studio_icon(dl, ICON_SCENE, lx + Float32(4.0), cy - Float32(7.0), Float32(14.0), theme.accent_primary, Float32(1.5))
        _draw_hud_text(dl, font_slot, font, "FLINT", lx + Float32(22.0), cy - Float32(6.5), text_sz, theme.text_primary)
        lx += Float32(74.0)

        # Vertical hairline divider
        dl.add_line(lx, self.rect.y + Float32(7.0), lx, self.rect.bottom() - Float32(7.0), Float32(1.0), theme.border_subtle)
        lx += Float32(8.0)

        var max_title_w = max(Float32(60.0), min(Float32(170.0), sw * Float32(0.16)))
        var disp_title = _truncate_hud_text(font, scene_title if scene_title.byte_length() > 0 else String("Untitled"), text_sz, max_title_w)
        _draw_hud_text(dl, font_slot, font, disp_title, lx, cy - Float32(6.5), text_sz, theme.text_secondary)
        var title_w = _measure_text_hud(font, disp_title, text_sz)
        if dirty:
            dl.add_circle(lx + title_w + Float32(8.0), cy, Float32(3.2), theme.play_paused)

        # ---- Right Zone (laid out right-to-left so center workspace pill never overlaps) ----
        var rx = self.rect.right() - Float32(8.0)

        # 0. Native Wayland Client-Side Window Controls (Minimize / Maximize / Close)
        if show_window_controls:
            var wb_w = Float32(26.0)
            var wb_h = Float32(22.0)
            var wb_y = self.rect.y + Float32(5.0)

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

            # Divider between window controls and Build chip
            dl.add_line(rx, self.rect.y + Float32(7.0), rx, self.rect.bottom() - Float32(7.0), Float32(1.0), theme.border_subtle)
            rx -= Float32(8.0)
        else:
            rx -= Float32(2.0)

        # 1. Build status chip
        var bld_w = Float32(58.0)
        rx -= bld_w
        var bld_r = RectF(rx, self.rect.y + Float32(4.0), bld_w, Float32(24.0))
        var bld_hov = bld_r.contains(mouse_x, mouse_y)
        var bld_ht = anim.drive_widget(_hud_hash(UInt64(0x48445201), "build"), bld_hov, bld_hov and mouse_down, False)
        var bld_bg = lerp_rgba8(theme.l2_card, theme.l3_hover, bld_ht[0])
        var bld_dot = theme.status_danger if build_state == 2 else (theme.status_warn if build_state == 1 else theme.status_ok)
        dl.add_rect_styled(bld_r.x, bld_r.y, bld_r.w, bld_r.h, bld_bg, theme.border_control, theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL)
        dl.add_circle(bld_r.x + Float32(10.0), cy, Float32(3.2), bld_dot)
        _draw_hud_text(dl, font_slot, font, "Build", bld_r.x + Float32(18.0), cy - Float32(6.0), Float32(11.5), theme.text_primary)
        if bld_hov and mouse_pressed:
            action = HeaderBarAction(HDR_ACT_BUILD, 0)
        rx -= Float32(6.0)

        # 2. Save button chip
        var save_w = Float32(52.0)
        rx -= save_w
        var save_r = RectF(rx, self.rect.y + Float32(4.0), save_w, Float32(24.0))
        var save_hov = save_r.contains(mouse_x, mouse_y)
        var save_ht = anim.drive_widget(_hud_hash(UInt64(0x48445202), "save"), save_hov, save_hov and mouse_down, False)
        var save_bg = lerp_rgba8(theme.l2_card, theme.l3_hover, save_ht[0])
        var save_bdr = theme.play_paused if dirty else theme.border_control
        dl.add_rect_styled(save_r.x, save_r.y, save_r.w, save_r.h, save_bg, save_bdr, theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL)
        _draw_hud_text(dl, font_slot, font, "Save", save_r.x + Float32(11.0), cy - Float32(6.0), Float32(11.5), theme.text_primary if dirty else theme.text_secondary)
        if save_hov and mouse_pressed:
            action = HeaderBarAction(HDR_ACT_SAVE, 0)
        rx -= Float32(6.0)

        # 3. Theme toggle button
        var thm_w = Float32(26.0)
        rx -= thm_w
        var thm_r = RectF(rx, self.rect.y + Float32(4.0), thm_w, Float32(24.0))
        var thm_hov = thm_r.contains(mouse_x, mouse_y)
        var thm_ht = anim.drive_widget(_hud_hash(UInt64(0x48445203), "theme"), thm_hov, thm_hov and mouse_down, False)
        dl.add_rect_styled(thm_r.x, thm_r.y, thm_r.w, thm_r.h, lerp_rgba8(theme.l2_card, theme.l3_hover, thm_ht[0]), theme.border_control, theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER)
        dl.add_circle(thm_r.center_x(), cy, Float32(5.0), theme.accent_primary if theme.is_dark else theme.play_paused, Float32(1.2), theme.border_strong)
        if thm_hov and mouse_pressed:
            action = HeaderBarAction(HDR_ACT_THEME_CYCLE, 0)
        rx -= Float32(8.0)

        # 4. Live Agent Status Pill with pulse ring
        var ag_label = agent_status_short_label(agent_status, pending_approvals)
        var ag_w = max(Float32(82.0), _measure_text_hud(font, ag_label, Float32(11.5)) + Float32(34.0))
        rx -= ag_w
        var ag_r = RectF(rx, self.rect.y + Float32(4.0), ag_w, Float32(24.0))
        var ag_hov = ag_r.contains(mouse_x, mouse_y)
        var ag_ht = anim.drive_widget(_hud_hash(UInt64(0x48445204), "agent"), ag_hov, ag_hov and mouse_down, False)
        var ag_col = theme.agent_status_color(agent_status)
        var ag_busy = agent_status != AGENT_STATE_IDLE or pending_approvals > 0
        var ag_bg = lerp_rgba8(theme.agent_soft if ag_busy else theme.l2_card, theme.l3_hover, ag_ht[0])
        var ag_bdr = rgba8_with_alpha(ag_col, 165) if ag_busy else theme.border_control
        dl.add_rect_styled(ag_r.x, ag_r.y, ag_r.w, ag_r.h, ag_bg, ag_bdr, Float32(12.0), Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL)
        var ring_cx = ag_r.x + Float32(13.0)
        if ag_busy:
            var spin_start = Float32(0.0) if anim.mode == UI_ANIM_INSTANT else (anim.time_sec * Float32(3.6))
            dl.add_arc(ring_cx, cy, Float32(6.5), Float32(1.8), spin_start, Float32(4.2), ag_col, rgba8_with_alpha(ag_col, 55), Float32(3.0))
        draw_studio_icon(dl, ICON_SPARK, ring_cx - Float32(5.5), cy - Float32(5.5), Float32(11.0), ag_col, Float32(1.3))
        _draw_hud_text(dl, font_slot, font, ag_label, ag_r.x + Float32(25.0), cy - Float32(6.0), Float32(11.5), theme.text_primary if ag_busy else theme.text_secondary)
        if ag_hov and mouse_pressed:
            action = HeaderBarAction(HDR_ACT_AGENT_TOGGLE, 0)
        rx -= Float32(8.0)

        # 5. Ctrl+K Command Palette Search Pill
        var pal_w = Float32(138.0) if sw >= Float32(1060.0) else Float32(96.0)
        rx -= pal_w
        var pal_r = RectF(rx, self.rect.y + Float32(4.0), pal_w, Float32(24.0))
        var pal_hov = pal_r.contains(mouse_x, mouse_y)
        var pal_ht = anim.drive_widget(_hud_hash(UInt64(0x48445205), "cmdpal"), pal_hov, pal_hov and mouse_down, False)
        dl.add_rect_styled(pal_r.x, pal_r.y, pal_r.w, pal_r.h, lerp_rgba8(theme.l0_canvas, theme.l3_raised, pal_ht[0]), theme.border_strong if pal_hov else theme.border_control, theme.radius_sm, Float32(1.0), Float32(1.0), UI_FLAG_BORDER)
        draw_studio_icon(dl, ICON_SEARCH, pal_r.x + Float32(7.0), cy - Float32(6.0), Float32(12.0), theme.text_muted, Float32(1.4))
        if pal_w >= Float32(120.0):
            _draw_hud_text(dl, font_slot, font, "Search", pal_r.x + Float32(23.0), cy - Float32(6.0), Float32(11.5), theme.text_muted)
        var kc_w = Float32(44.0)
        var kc_x = pal_r.right() - kc_w - Float32(4.0)
        dl.add_rect(kc_x, pal_r.y + Float32(3.0), kc_w, Float32(18.0), theme.l2_card, Float32(3.5), Float32(1.0), theme.border_subtle)
        _draw_hud_text(dl, font_slot, font, "Ctrl+K", kc_x + Float32(5.0), pal_r.y + Float32(6.0), Float32(10.5), theme.text_secondary)
        if pal_hov and mouse_pressed:
            action = HeaderBarAction(HDR_ACT_COMMAND_PALETTE, 0)

        # ---- Center Zone: Animated Workspace Preset Segmented Pill ----
        var pill_labels = List[String](capacity=WORKSPACE_COUNT)
        pill_labels.append("Edit")
        pill_labels.append("Agent Studio")
        pill_labels.append("Code + Scene")
        pill_labels.append("Lookdev")
        pill_labels.append("Play")

        var pill_ws = List[Float32](capacity=WORKSPACE_COUNT)
        var total_pw = Float32(4.0)
        for pi in range(WORKSPACE_COUNT):
            var w = max(Float32(44.0), _measure_text_hud(font, pill_labels[pi], Float32(11.5)) + Float32(18.0))
            pill_ws.append(w)
            total_pw += w + Float32(2.0)
        total_pw += Float32(2.0)

        var left_bound = lx + title_w + Float32(20.0)
        var right_bound = rx - Float32(12.0)
        if right_bound - left_bound >= total_pw:
            var ideal_x = (sw - total_pw) * Float32(0.5)
            var group_x = max(left_bound, min(right_bound - total_pw, ideal_x))
            var group_r = RectF(group_x, self.rect.y + Float32(3.0), total_pw, Float32(26.0))
            dl.add_rect(group_r.x, group_r.y, group_r.w, group_r.h, theme.l0_canvas, theme.radius_sm, Float32(1.0), theme.border_subtle)

            # Compute tab rects & target for sliding active pill
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
            dl.add_rect_styled(
                ind[0],
                group_r.y + Float32(2.0),
                ind[1],
                group_r.h - Float32(4.0),
                theme.l3_raised,
                theme.border_control,
                theme.radius_xs,
                Float32(1.0),
                Float32(1.0),
                UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )
            dl.add_rect(ind[0] + Float32(6.0), group_r.bottom() - Float32(3.5), max(Float32(4.0), ind[1] - Float32(12.0)), Float32(2.0), active_accent, Float32(1.0))

            for pi in range(WORKSPACE_COUNT):
                var tr = RectF(tab_xs[pi], group_r.y + Float32(2.0), pill_ws[pi], group_r.h - Float32(4.0))
                var thov = tr.contains(mouse_x, mouse_y)
                var is_act = pi == self.active_workspace
                if thov and not is_act:
                    dl.add_rect(tr.x, tr.y, tr.w, tr.h, rgba8_with_alpha(theme.l3_hover, 110), theme.radius_xs)
                var tw = _measure_text_hud(font, pill_labels[pi], Float32(11.5))
                _draw_hud_text(
                    dl,
                    font_slot,
                    font,
                    pill_labels[pi],
                    tr.x + max(Float32(4.0), (tr.w - tw) * Float32(0.5)),
                    cy - Float32(6.0),
                    Float32(11.5),
                    theme.text_primary if (is_act or thov) else theme.text_secondary,
                )
                if thov and mouse_pressed:
                    action = HeaderBarAction(HDR_ACT_WORKSPACE, pi)

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
    """4-cluster translucent floating Viewport HUD with exact pointer passthrough & 3D OrientationGizmo."""
    var gpu_ms_ring: List[Float32]
    var ring_cursor: Int
    var ring_count: Int
    var perf_expanded: Bool
    var hit_rects: List[RectF]
    var hovered_orientation_axis: Int

    def __init__(out self):
        self.gpu_ms_ring = List[Float32](length=GPU_SPARKLINE_SAMPLES, fill=Float32(0.0))
        self.ring_cursor = 0
        self.ring_count = 0
        self.perf_expanded = False
        self.hit_rects = List[RectF]()
        self.hovered_orientation_axis = -1

    def __init__(out self, *, copy: Self):
        self.gpu_ms_ring = copy.gpu_ms_ring.copy()
        self.ring_cursor = copy.ring_cursor
        self.ring_count = copy.ring_count
        self.perf_expanded = copy.perf_expanded
        self.hit_rects = copy.hit_rects.copy()
        self.hovered_orientation_axis = copy.hovered_orientation_axis

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
    ) -> Bool:
        var hov = r.contains(mouse_x, mouse_y)
        var st = anim.drive_widget(wid, hov, hov and mouse_down, False)
        var icon_tint = active_tint if active_tint != 0 else theme.accent_primary
        if hov and not active:
            dl.add_rect(r.x, r.y, r.w, r.h, rgba8_scale_alpha(theme.l3_hover, alpha_scale * Float32(0.65) * st[0]), theme.radius_xs)
        var col = icon_tint if active else (theme.text_primary if hov else theme.text_secondary)
        var isz = Float32(14.0)
        draw_studio_icon(
            dl,
            icon_id,
            r.center_x() - isz * Float32(0.5),
            r.center_y() - isz * Float32(0.5),
            isz,
            rgba8_scale_alpha(col, alpha_scale),
            Float32(1.5),
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
    ) -> ViewportHudAction:
        """Emits the 4 floating viewport HUD clusters + PIE viewport border and returns any action."""
        var action = ViewportHudAction()
        self.hit_rects.clear()
        self.hovered_orientation_axis = -1

        if vp.w < Float32(120.0) or vp.h < Float32(80.0):
            return action

        dl.push_clip(Int(vp.x), Int(vp.y), Int(vp.right()), Int(vp.bottom()))

        # ---- 0. PIE Active Viewport Perimeter Border ----
        if pie_mode == VHUD_PIE_PLAYING:
            dl.add_glow_rect(
                vp.x + Float32(1.0),
                vp.y + Float32(1.0),
                max(Float32(2.0), vp.w - Float32(2.0)),
                max(Float32(2.0), vp.h - Float32(2.0)),
                UInt32(0),
                theme.play_mode,
                Float32(2.0),
                Float32(4.0),
                Float32(2.2),
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
                Float32(2.0),
                theme.play_paused,
            )

        var in_pie = pie_mode == VHUD_PIE_PLAYING
        var inset = Float32(10.0)
        var pill_h = Float32(30.0)

        # ---- 1. Top-Left Cluster: Transform (Q/W/E/R) + Space + Snap ----
        var tl_w = Float32(246.0)
        var tl_r = RectF(vp.x + inset, vp.y + inset, tl_w, pill_h)
        self.hit_rects.append(tl_r)
        var tl_hov = tl_r.contains(mouse_x, mouse_y)
        var tl_dim_tgt = Float32(0.0) if (not in_pie or tl_hov) else Float32(1.0)
        var tl_dim = anim.drive_aux(
            UInt64(0x564855440001),
            tl_dim_tgt,
            SpringConfig.snappy(),
        )
        if anim.mode == UI_ANIM_INSTANT:
            tl_dim = tl_dim_tgt
        var tl_alpha = Float32(1.0) - Float32(0.58) * tl_dim

        dl.add_glass_panel(
            tl_r.x, tl_r.y, tl_r.w, tl_r.h,
            rgba8_scale_alpha(theme.l4_overlay, tl_alpha),
            theme.radius_md,
            Float32(3.0),
            Float32(1.0),
            rgba8_scale_alpha(theme.border_strong, tl_alpha),
            True,
            True,
        )

        # Sliding active highlight behind Q/W/E/R
        var tool_slot = 0 if gizmo_mode == VHUD_TOOL_SELECT else (1 if gizmo_mode == VHUD_TOOL_MOVE else (2 if gizmo_mode == VHUD_TOOL_ROTATE else 3))
        var btn_sz = Float32(24.0)
        var btn_y = tl_r.y + Float32(3.0)
        var base_x = tl_r.x + Float32(4.0)
        var tool_tgt_x = base_x + Float32(tool_slot) * (btn_sz + Float32(2.0))
        var tool_ind = anim.drive_tab_pill(UInt64(0x56485544544C), tool_tgt_x, btn_sz)
        dl.add_rect(
            tool_ind[0],
            btn_y,
            tool_ind[1],
            btn_sz,
            rgba8_scale_alpha(theme.accent_soft, tl_alpha),
            theme.radius_xs,
            Float32(1.0),
            rgba8_scale_alpha(theme.accent_primary, tl_alpha * Float32(0.75)),
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
            var br = RectF(base_x + Float32(ti) * (btn_sz + Float32(2.0)), btn_y, btn_sz, btn_sz)
            if self._draw_tool_icon_btn(
                dl, theme, anim,
                _hud_hash(UInt64(0x564855441000), String(ti)),
                br, tool_icons[ti], ti == tool_slot, tl_alpha,
                mouse_x, mouse_y, mouse_down, mouse_pressed,
            ):
                action.kind = tool_acts[ti]

        # Divider + World/Local chip
        var div1_x = base_x + Float32(4.0) * (btn_sz + Float32(2.0)) + Float32(3.0)
        dl.add_line(div1_x, tl_r.y + Float32(6.0), div1_x, tl_r.bottom() - Float32(6.0), Float32(1.0), rgba8_scale_alpha(theme.border_subtle, tl_alpha))

        var sp_r = RectF(div1_x + Float32(5.0), btn_y, Float32(58.0), btn_sz)
        var sp_hov = sp_r.contains(mouse_x, mouse_y)
        if sp_hov or gizmo_local_space:
            dl.add_rect(
                sp_r.x, sp_r.y, sp_r.w, sp_r.h,
                rgba8_scale_alpha(theme.accent_soft if gizmo_local_space else theme.l3_hover, tl_alpha * Float32(0.85)),
                theme.radius_xs,
            )
        var sp_col = rgba8_scale_alpha(theme.accent_primary if gizmo_local_space else theme.text_secondary, tl_alpha)
        draw_studio_icon(dl, ICON_LOCAL if gizmo_local_space else ICON_WORLD, sp_r.x + Float32(5.0), sp_r.center_y() - Float32(6.5), Float32(13.0), sp_col, Float32(1.4))
        _draw_hud_text(dl, font_slot, font, "LOC" if gizmo_local_space else "WLD", sp_r.x + Float32(22.0), sp_r.y + Float32(6.0), Float32(11.0), rgba8_scale_alpha(theme.text_primary, tl_alpha))
        if sp_hov and mouse_pressed:
            action.kind = VHUD_ACT_TOGGLE_SPACE

        # Snap toggle chip with grid step readout
        var sn_r = RectF(sp_r.right() + Float32(4.0), btn_y, Float32(66.0), btn_sz)
        var sn_hov = sn_r.contains(mouse_x, mouse_y)
        if snap_enabled or sn_hov:
            dl.add_rect(
                sn_r.x, sn_r.y, sn_r.w, sn_r.h,
                rgba8_scale_alpha(theme.accent_soft if snap_enabled else theme.l3_hover, tl_alpha * Float32(0.85)),
                theme.radius_xs,
                Float32(1.0) if snap_enabled else Float32(0.0),
                rgba8_scale_alpha(theme.accent_primary, tl_alpha * Float32(0.6)),
            )
        var sn_col = rgba8_scale_alpha(theme.accent_primary if snap_enabled else theme.text_muted, tl_alpha)
        draw_studio_icon(dl, ICON_SNAP, sn_r.x + Float32(5.0), sn_r.center_y() - Float32(6.5), Float32(13.0), sn_col, Float32(1.4))
        _draw_hud_text(dl, font_slot, font, _fmt_f32_2dp(grid_snap_m) + "m", sn_r.x + Float32(21.0), sn_r.y + Float32(6.0), Float32(10.5), rgba8_scale_alpha(theme.text_primary if snap_enabled else theme.text_muted, tl_alpha))
        if sn_hov and mouse_pressed:
            action.kind = VHUD_ACT_TOGGLE_SNAP

        # ---- 2. Top-Center Cluster: Transport & PIE Pill ----
        var tc_w = Float32(236.0) if pie_mode != VHUD_PIE_STOPPED else Float32(184.0)
        var tc_x = max(tl_r.right() + Float32(8.0), vp.center_x() - tc_w * Float32(0.5))
        var tc_r = RectF(tc_x, vp.y + inset, tc_w, pill_h)
        if tc_r.right() <= vp.right() - inset:
            self.hit_rects.append(tc_r)
            var tc_bdr = (
                theme.play_mode
                if pie_mode == VHUD_PIE_PLAYING
                else (theme.play_paused if pie_mode == VHUD_PIE_PAUSED else theme.border_strong)
            )
            dl.add_glass_panel(
                tc_r.x, tc_r.y, tc_r.w, tc_r.h,
                theme.l4_overlay,
                theme.radius_md,
                Float32(3.0),
                Float32(1.2) if pie_mode != VHUD_PIE_STOPPED else Float32(1.0),
                tc_bdr,
                True,
                True,
            )
            var tx = tc_r.x + Float32(5.0)
            # Play
            var r_play = RectF(tx, btn_y, btn_sz, btn_sz)
            if self._draw_tool_icon_btn(dl, theme, anim, UInt64(0x564855442001), r_play, ICON_PLAY, pie_mode == VHUD_PIE_PLAYING, Float32(1.0), mouse_x, mouse_y, mouse_down, mouse_pressed, theme.play_mode):
                action.kind = VHUD_ACT_PIE_PLAY
            tx += btn_sz + Float32(2.0)
            # Pause
            var r_pause = RectF(tx, btn_y, btn_sz, btn_sz)
            if self._draw_tool_icon_btn(dl, theme, anim, UInt64(0x564855442002), r_pause, ICON_PAUSE, pie_mode == VHUD_PIE_PAUSED, Float32(1.0), mouse_x, mouse_y, mouse_down, mouse_pressed, theme.play_paused):
                action.kind = VHUD_ACT_PIE_PAUSE
            tx += btn_sz + Float32(2.0)
            # Step
            var r_step = RectF(tx, btn_y, btn_sz, btn_sz)
            if self._draw_tool_icon_btn(dl, theme, anim, UInt64(0x564855442003), r_step, ICON_STEP, False, Float32(1.0), mouse_x, mouse_y, mouse_down, mouse_pressed):
                action.kind = VHUD_ACT_PIE_STEP
            tx += btn_sz + Float32(2.0)
            # Stop
            var r_stop = RectF(tx, btn_y, btn_sz, btn_sz)
            if self._draw_tool_icon_btn(dl, theme, anim, UInt64(0x564855442004), r_stop, ICON_STOP, pie_mode != VHUD_PIE_STOPPED, Float32(1.0), mouse_x, mouse_y, mouse_down, mouse_pressed, theme.status_danger):
                action.kind = VHUD_ACT_PIE_STOP
            tx += btn_sz + Float32(6.0)

            # Mode badge pill
            var badge_col = theme.play_mode if pie_mode == VHUD_PIE_PLAYING else (theme.play_paused if pie_mode == VHUD_PIE_PAUSED else theme.text_muted)
            var badge_txt = "LIVE" if pie_mode == VHUD_PIE_PLAYING else ("PAUSED" if pie_mode == VHUD_PIE_PAUSED else "EDIT")
            var badge_w = Float32(58.0)
            dl.add_rect(tx, tc_r.y + Float32(5.0), badge_w, Float32(20.0), rgba8_with_alpha(badge_col, 34), Float32(10.0), Float32(1.0), rgba8_with_alpha(badge_col, 110))
            dl.add_circle(tx + Float32(9.0), tc_r.center_y(), Float32(3.0), badge_col)
            _draw_hud_text(dl, font_slot, font, badge_txt, tx + Float32(16.0), tc_r.y + Float32(8.5), Float32(10.5), theme.text_primary if pie_mode != VHUD_PIE_STOPPED else theme.text_secondary)

            if pie_mode != VHUD_PIE_STOPPED:
                var ej_r = RectF(tx + badge_w + Float32(5.0), btn_y, Float32(50.0), btn_sz)
                var ej_hov = ej_r.contains(mouse_x, mouse_y)
                dl.add_rect(ej_r.x, ej_r.y, ej_r.w, ej_r.h, theme.accent_soft if pie_camera_ejected else (theme.l3_hover if ej_hov else theme.l2_card), theme.radius_xs)
                _draw_hud_text(dl, font_slot, font, "Eject", ej_r.x + Float32(10.0), ej_r.y + Float32(6.0), Float32(10.5), theme.accent_primary if pie_camera_ejected else theme.text_primary)
                if ej_hov and mouse_pressed:
                    action.kind = VHUD_ACT_PIE_EJECT

        # ---- 3. Top-Right Cluster: Camera / Overlays / 60-Frame GPU Sparkline ----
        var tr_w = Float32(236.0)
        var tr_r = RectF(vp.right() - tr_w - inset, vp.y + inset, tr_w, pill_h)
        if tr_r.x >= tc_r.right() + Float32(6.0):
            self.hit_rects.append(tr_r)
            var tr_hov = tr_r.contains(mouse_x, mouse_y)
            var tr_dim_tgt = Float32(0.0) if (not in_pie or tr_hov) else Float32(1.0)
            var tr_dim = anim.drive_aux(
                UInt64(0x564855440003),
                tr_dim_tgt,
                SpringConfig.snappy(),
            )
            if anim.mode == UI_ANIM_INSTANT:
                tr_dim = tr_dim_tgt
            var tr_alpha = Float32(1.0) - Float32(0.58) * tr_dim

            dl.add_glass_panel(
                tr_r.x, tr_r.y, tr_r.w, tr_r.h,
                rgba8_scale_alpha(theme.l4_overlay, tr_alpha),
                theme.radius_md,
                Float32(3.0),
                Float32(1.0),
                rgba8_scale_alpha(theme.border_strong, tr_alpha),
                True,
                True,
            )

            var cx = tr_r.x + Float32(5.0)
            # Camera Orbit/Fly button
            var cam_r = RectF(cx, btn_y, Float32(52.0), btn_sz)
            var cam_hov = cam_r.contains(mouse_x, mouse_y)
            if cam_hov:
                dl.add_rect(cam_r.x, cam_r.y, cam_r.w, cam_r.h, rgba8_scale_alpha(theme.l3_hover, tr_alpha), theme.radius_xs)
            draw_studio_icon(dl, ICON_CAMERA, cam_r.x + Float32(4.0), cam_r.center_y() - Float32(6.5), Float32(13.0), rgba8_scale_alpha(theme.text_primary, tr_alpha), Float32(1.4))
            _draw_hud_text(dl, font_slot, font, "FLY" if cam_free_fly else "ORB", cam_r.x + Float32(21.0), cam_r.y + Float32(6.0), Float32(10.5), rgba8_scale_alpha(theme.text_primary, tr_alpha))
            if cam_hov and mouse_pressed:
                action.kind = VHUD_ACT_CAM_MODE
            cx += Float32(54.0)

            # Grid / Collider / Nav overlay mini icons
            var g_r = RectF(cx, btn_y, Float32(22.0), btn_sz)
            if self._draw_tool_icon_btn(dl, theme, anim, UInt64(0x564855443001), g_r, ICON_GRID, show_grid, tr_alpha, mouse_x, mouse_y, mouse_down, mouse_pressed):
                action.kind = VHUD_ACT_TOGGLE_GRID
            cx += Float32(23.0)

            var c_r = RectF(cx, btn_y, Float32(22.0), btn_sz)
            if self._draw_tool_icon_btn(dl, theme, anim, UInt64(0x564855443002), c_r, ICON_COLLIDER, show_colliders, tr_alpha, mouse_x, mouse_y, mouse_down, mouse_pressed):
                action.kind = VHUD_ACT_TOGGLE_COLLIDERS
            cx += Float32(23.0)

            var nav_r = RectF(cx, btn_y, Float32(22.0), btn_sz)
            var nav_hov = nav_r.contains(mouse_x, mouse_y)
            if nav_hov:
                dl.add_rect(nav_r.x, nav_r.y, nav_r.w, nav_r.h, rgba8_scale_alpha(theme.l3_hover, tr_alpha), theme.radius_xs)
            dl.add_circle(nav_r.center_x(), nav_r.center_y(), Float32(4.5), rgba8_scale_alpha(theme.play_mode if show_navmesh else theme.text_muted, tr_alpha))
            if nav_hov and mouse_pressed:
                action.kind = VHUD_ACT_TOGGLE_NAVMESH
            cx += Float32(26.0)

            # Divider + 60-frame GPU ms sparkline
            dl.add_line(cx, tr_r.y + Float32(6.0), cx, tr_r.bottom() - Float32(6.0), Float32(1.0), rgba8_scale_alpha(theme.border_subtle, tr_alpha))
            cx += Float32(5.0)

            var perf_r = RectF(cx, btn_y, tr_r.right() - cx - Float32(5.0), btn_sz)
            var perf_hov = perf_r.contains(mouse_x, mouse_y)
            if perf_hov or self.perf_expanded:
                dl.add_rect(perf_r.x, perf_r.y, perf_r.w, perf_r.h, rgba8_scale_alpha(theme.l3_hover, tr_alpha * Float32(0.7)), theme.radius_xs)
            var cur_ms = self.latest_gpu_ms()
            var perf_col = theme.agent_flow if cur_ms < Float32(8.3) else (theme.status_warn if cur_ms < Float32(16.6) else theme.status_danger)
            _draw_hud_text(dl, font_slot, font, _fmt_f32_1dp(cur_ms) + "ms", perf_r.x + Float32(3.0), perf_r.y + Float32(6.0), Float32(10.5), rgba8_scale_alpha(theme.text_primary, tr_alpha))

            # Sparkline graph inside right half of perf_r (auto-scaled around active frame-time band)
            var sp_x = perf_r.x + Float32(42.0)
            var sp_w = max(Float32(16.0), perf_r.right() - sp_x - Float32(3.0))
            var sp_y = perf_r.y + Float32(3.0)
            var sp_h = perf_r.h - Float32(6.0)
            dl.add_rect(sp_x, sp_y, sp_w, sp_h, rgba8_scale_alpha(theme.l0_canvas, tr_alpha * Float32(0.85)), Float32(3.0), Float32(1.0), rgba8_scale_alpha(theme.border_subtle, tr_alpha))
            var n_pts = self.ring_count
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
                var step_x = (sp_w - Float32(4.0)) / Float32(max(1, GPU_SPARKLINE_SAMPLES - 1))
                var prev_px = Float32(0.0)
                var prev_py = Float32(0.0)
                var col_glow = rgba8_scale_alpha(rgba8_with_alpha(perf_col, 68), tr_alpha)
                var col_line = rgba8_scale_alpha(perf_col, tr_alpha)
                for si in range(n_pts):
                    var idx = (self.ring_cursor - n_pts + si + GPU_SPARKLINE_SAMPLES) % GPU_SPARKLINE_SAMPLES
                    var norm = max(Float32(0.0), min(Float32(1.0), (self.gpu_ms_ring[idx] - band_lo) * inv_span))
                    var px = sp_x + Float32(2.0) + Float32(GPU_SPARKLINE_SAMPLES - n_pts + si) * step_x
                    var py = sp_y + sp_h - Float32(2.0) - norm * (sp_h - Float32(4.0))
                    if si > 0:
                        dl.add_line(prev_px, prev_py, px, py, Float32(3.0), col_glow)
                        dl.add_line(prev_px, prev_py, px, py, Float32(1.4), col_line)
                    prev_px = px
                    prev_py = py
            if perf_hov and mouse_pressed:
                self.perf_expanded = not self.perf_expanded
                action.kind = VHUD_ACT_TOGGLE_PERF

            # Optional expanded perf breakdown tooltip
            if self.perf_expanded:
                var pop_r = RectF(tr_r.right() - Float32(168.0), tr_r.bottom() + Float32(6.0), Float32(168.0), Float32(52.0))
                self.hit_rects.append(pop_r)
                dl.add_glass_panel(pop_r.x, pop_r.y, pop_r.w, pop_r.h, theme.l4_overlay, theme.radius_sm, Float32(3.0), Float32(1.0), theme.border_strong, True, True)
                var fps = Float32(1000.0) / max(Float32(0.1), cur_ms) if cur_ms > Float32(0.01) else Float32(0.0)
                _draw_hud_text(dl, font_slot, font, "GPU Frame: " + _fmt_f32_2dp(cur_ms) + " ms", pop_r.x + Float32(10.0), pop_r.y + Float32(9.0), Float32(11.5), theme.text_primary)
                _draw_hud_text(dl, font_slot, font, "Est. FPS: " + _fmt_f32_1dp(fps) + "  (" + String(n_pts) + "f)", pop_r.x + Float32(10.0), pop_r.y + Float32(28.0), Float32(11.0), theme.text_secondary)

        # ---- 4. Bottom-Right Cluster: Interactive 3D OrientationGizmo + Persp/Ortho + Delta Badge ----
        var giz_r_px = Float32(27.0)
        var giz_cx = vp.right() - Float32(48.0)
        var giz_cy = vp.bottom() - Float32(62.0)
        var giz_box = RectF(giz_cx - Float32(38.0), giz_cy - Float32(38.0), Float32(76.0), Float32(102.0))
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
        var br_alpha = Float32(1.0) - Float32(0.60) * br_dim

        # Translucent backing disc
        dl.add_circle(
            giz_cx,
            giz_cy,
            Float32(36.0),
            rgba8_scale_alpha(theme.l4_overlay, br_alpha * Float32(0.72)),
            Float32(1.0),
            rgba8_scale_alpha(theme.border_strong if br_hov else theme.border_subtle, br_alpha),
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

        # Hit-test front-to-back (reverse of back-to-front)
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

        # Emit stalks & spheres back-to-front
        for i in range(len(projs)):
            var p = projs[i]
            var is_hov = p.idx == hit_axis
            var ht = anim.drive_widget(_hud_hash(UInt64(0x4F524947), String(p.idx)), is_hov, is_hov and mouse_down, False)
            var depth_dim = Float32(1.0) if p.depth <= Float32(0.0) else Float32(0.68)
            var col = rgba8_scale_alpha(p.color, br_alpha * depth_dim)
            if p.positive:
                dl.add_line(giz_cx, giz_cy, p.sx, p.sy, Float32(2.2), col)
                var sr = Float32(7.2) + ht[0] * Float32(1.8)
                dl.add_circle(
                    p.sx,
                    p.sy,
                    sr,
                    col,
                    Float32(1.4) if is_hov else Float32(0.0),
                    rgba8_scale_alpha(theme.text_primary, br_alpha),
                )
                _draw_hud_text(dl, font_slot, font, p.label, p.sx - Float32(3.5), p.sy - Float32(5.5), Float32(9.5), rgba8_scale_alpha(theme.text_on_accent, br_alpha))
            else:
                var sr = Float32(4.6) + ht[0] * Float32(1.6)
                var fill_neg = rgba8_scale_alpha(p.color, br_alpha * (Float32(0.85) if is_hov else Float32(0.30)))
                dl.add_circle(p.sx, p.sy, sr, fill_neg, Float32(1.2), col)
            if is_hov and mouse_pressed:
                action = ViewportHudAction(VHUD_ACT_SNAP_VIEW, p.snap_fwd, p.idx)

        # Persp / Ortho toggle pill beneath the triad
        var po_r = RectF(giz_cx - Float32(30.0), giz_cy + Float32(40.0), Float32(60.0), Float32(20.0))
        var po_hov = po_r.contains(mouse_x, mouse_y)
        dl.add_glass_panel(
            po_r.x, po_r.y, po_r.w, po_r.h,
            rgba8_scale_alpha(theme.l3_hover if po_hov else theme.l4_overlay, br_alpha),
            Float32(10.0),
            Float32(2.0),
            Float32(1.0),
            rgba8_scale_alpha(theme.accent_primary if cam_ortho else theme.border_control, br_alpha),
            False,
            False,
        )
        var po_txt = String("Ortho") if cam_ortho else String("Persp")
        var po_tw = _measure_text_hud(font, po_txt, Float32(10.5))
        _draw_hud_text(dl, font_slot, font, po_txt, po_r.center_x() - po_tw * Float32(0.5), po_r.y + Float32(4.5), Float32(10.5), rgba8_scale_alpha(theme.text_primary, br_alpha))
        if po_hov and mouse_pressed:
            action.kind = VHUD_ACT_TOGGLE_ORTHO

        # Live selection count / active gizmo delta badge to the left of OrientationGizmo
        var info_txt = drag_delta_text if drag_delta_text.byte_length() > 0 else ("SEL " + String(selected_count))
        var info_w = max(Float32(52.0), _measure_text_hud(font, info_txt, Float32(11.0)) + Float32(16.0))
        var info_r = RectF(giz_cx - Float32(44.0) - info_w, po_r.y - Float32(1.0), info_w, Float32(22.0))
        if info_r.x > vp.x + inset:
            self.hit_rects.append(info_r)
            var is_delta = drag_delta_text.byte_length() > 0
            dl.add_glass_panel(
                info_r.x, info_r.y, info_r.w, info_r.h,
                rgba8_scale_alpha(theme.l4_overlay, br_alpha),
                Float32(11.0),
                Float32(2.0),
                Float32(1.0),
                rgba8_scale_alpha(theme.accent_primary if is_delta else theme.border_subtle, br_alpha),
                False,
                False,
            )
            _draw_hud_text(
                dl, font_slot, font, info_txt,
                info_r.x + Float32(8.0), info_r.y + Float32(5.0),
                Float32(11.0),
                rgba8_scale_alpha(theme.accent_primary if is_delta else theme.text_secondary, br_alpha),
            )

        dl.pop_clip()
        return action
