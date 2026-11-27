"""Flint Studio RGBMocap Panel & 3D Capture Viewport Overlay (`flint.studio.mocap_panel`).

Pure-Mojo 1.2 Studio UI module matching the **Obsidian & Flint Ember** design system:
  - `MocapStudioState`: backend cache probe state, source/output paths, `ContactCleanupConfig`,
    live `MocapCleanupMetrics`, raw & cleaned `MocapTake`, deterministic demo loader,
    playback transport, and 3D capture overlay visibility toggles.
  - `MocapPanelResult`: typed action flags returned from `draw_mocap_panel`.
  - `draw_mocap_panel`: Obsidian `L1_shell` / `L2_card` / `L3_raised` right-dock panel
    (zero clipping at `300..360px` width, deterministic under `UI_ANIM_INSTANT`) featuring:
      1. Header & Backend / Model Manager Card (`GEM-X` | `RTMPose`, commercial-safe badge,
         live cache probe pill, local cache path / export hint)
      2. Source & Multi-Person Track Card (path input, `[Demo]` & `[Capture]` buttons,
         multi-person track pills `Person #1 94%` | `Person #2 78%`, frame/FPS mono telemetry)
      3. Motion Cleanup Card (`toyxyz_RGBMocap` parity: One Euro Filter + Min Cutoff / Beta,
         Foot Contact Lock + Lock Strength / Height, Palm Contact, Ground Align, and live
         Foot Slide Reduction telemetry bar `5.7 cm -> 0.6 cm  -90%`)
      4. Confidence & Contact Timeline Card (interactive scrubber over per-frame emerald/amber/red
         confidence heatbar, L/R foot contact lock interval strips, Play/Pause/Prev/Next,
         `Frame k / N` mono readout, and `Skel` | `Frustum` | `Contacts` overlay toggles)
      5. Sticky Export & Retarget Footer (`[Apply to Entity]`, `[BVH]`, `[GLB]`)
  - `emit_mocap_viewport_overlay`: analytical 3D SOMA skeleton (confidence-coded limbs & joints,
    contact highlights), 3D Camera Frustum wireframe with frosted `CAM 45°` pill, and ground-plane
    concentric emerald Foot Contact Lock Rings.
"""
from std.collections import List
from std.math import abs, cos, isfinite, max, min, sin, tan
from flint.anim.mocap_filter import (
    ContactCleanupConfig,
    MocapCleanupMetrics,
    apply_mocap_cleanup_to_take,
)
from flint.core.math import Mat4, Vec3
from flint.ml.rgbmocap import (
    MOCAP_BACKEND_GEMX,
    MOCAP_BACKEND_RTMPOSE_MBERT,
    MocapBackendInfo,
    MocapCameraPose,
    MocapFrame,
    MocapTake,
    SOMA77_CHEST,
    SOMA77_HEAD,
    SOMA77_HIPS,
    SOMA77_LEFT_ARM,
    SOMA77_LEFT_FOOT,
    SOMA77_LEFT_FOREARM,
    SOMA77_LEFT_HAND,
    SOMA77_LEFT_LEG,
    SOMA77_LEFT_SHIN,
    SOMA77_LEFT_SHOULDER,
    SOMA77_LEFT_TOE_BASE,
    SOMA77_NECK1,
    SOMA77_NECK2,
    SOMA77_RIGHT_ARM,
    SOMA77_RIGHT_FOOT,
    SOMA77_RIGHT_FOREARM,
    SOMA77_RIGHT_HAND,
    SOMA77_RIGHT_LEG,
    SOMA77_RIGHT_SHIN,
    SOMA77_RIGHT_SHOULDER,
    SOMA77_RIGHT_TOE_BASE,
    SOMA77_SPINE1,
    SOMA77_SPINE2,
    generate_demo_mocap_take,
    probe_mocap_backends,
)
from flint.render.font import SdfFont
from flint.ui.context import (
    BTN_PRIMARY,
    BTN_SECONDARY,
    UI_ALIGN_CENTER,
    UI_ALIGN_LEFT,
    UI_ALIGN_RIGHT,
    UI_FONT_BOLD,
    UI_FONT_MONO,
    UI_FONT_REGULAR,
    UiContext,
    _BoxScope,
    hash_ui_id,
    pack_rgba8,
)
from flint.ui.draw_list import (
    UI_FLAG_BORDER,
    UI_FLAG_TOP_BEVEL,
    UiDrawList,
)
from flint.ui.flexbox_dock import FLEX_COL, FLEX_ROW, FlexStyle, RectF
from flint.ui.icons import draw_studio_icon
from flint.ui.theme import (
    ICON_CAMERA,
    ICON_CHECK,
    ICON_CLOCK,
    ICON_FOLDER,
    ICON_PAUSE,
    ICON_PLAY,
    ICON_REFRESH,
    ICON_SPARK,
    RADIUS_MD,
    RADIUS_PILL,
    RADIUS_SM,
    RADIUS_XS,
    StudioTheme,
    lerp_rgba8,
    rgba8_scale_alpha,
    rgba8_with_alpha,
)


# ============================================================================
# 1. Retained Panel State & Action Result Structs
# ============================================================================


struct MocapStudioState(Copyable, Movable):
    """Retained state for the Flint Studio RGBMocap panel & 3D capture viewport overlay."""
    var backends: List[MocapBackendInfo]
    var selected_backend: Int
    var source_path: String
    var output_bvh_path: String
    var cleanup_cfg: ContactCleanupConfig
    var metrics: MocapCleanupMetrics
    var raw_take: MocapTake
    var clean_take: MocapTake
    var has_take: Bool
    var playing: Bool
    var current_frame: Int
    var playback_accum: Float32
    var show_frustum_overlay: Bool
    var show_contact_rings: Bool
    var show_skeleton_overlay: Bool
    var status_message: String
    var job_running: Bool
    var job_progress: Float32
    var scroll_y: Float32

    def __init__(out self):
        self.backends = List[MocapBackendInfo]()
        self.selected_backend = MOCAP_BACKEND_GEMX
        self.source_path = String("")
        self.output_bvh_path = String("exports/mocap_take.bvh")
        self.cleanup_cfg = ContactCleanupConfig()
        self.metrics = MocapCleanupMetrics()
        self.raw_take = MocapTake()
        self.clean_take = MocapTake()
        self.has_take = False
        self.playing = False
        self.current_frame = 0
        self.playback_accum = Float32(0.0)
        self.show_frustum_overlay = True
        self.show_contact_rings = True
        self.show_skeleton_overlay = True
        self.status_message = String("Ready")
        self.job_running = False
        self.job_progress = Float32(0.0)
        self.scroll_y = Float32(0.0)

    def init_default(mut self, cache_root: String = ""):
        """Probes commercial-safe model caches and loads a realistic demo take ready for inspection."""
        self.backends = probe_mocap_backends(cache_root)
        self.selected_backend = MOCAP_BACKEND_GEMX
        self.show_frustum_overlay = True
        self.show_contact_rings = True
        self.show_skeleton_overlay = True
        self.cleanup_cfg = ContactCleanupConfig()
        self.load_demo_take()

    def load_demo_take(mut self, num_frames: Int = 90, fps: Float32 = 30.0):
        """Loads the deterministic synthetic SOMA-77 walk+turn+wave demo take and applies cleanup."""
        self.raw_take = generate_demo_mocap_take(num_frames, fps)
        self.raw_take.backend_id = self.selected_backend
        self.source_path = String("demo://soma_walk_turn_wave.mp4")
        self.output_bvh_path = String("exports/mocap_demo_soma.bvh")
        self.has_take = self.raw_take.num_frames() > 0
        self.playing = False
        self.playback_accum = Float32(0.0)
        self.job_running = False
        self.job_progress = Float32(0.0)
        self.recompute_cleanup()
        # Pick a clean mid-walk stance frame (frame 24) so foot-contact lock rings are visible immediately
        if self.clean_take.num_frames() > 0:
            self.current_frame = min(23, self.clean_take.num_frames() - 1)
        else:
            self.current_frame = 0

    def recompute_cleanup(mut self):
        """Re-applies `cleanup_cfg` onto a fresh copy of `raw_take` and refreshes live telemetry."""
        if not self.has_take or self.raw_take.num_frames() == 0:
            self.clean_take = MocapTake()
            self.metrics = MocapCleanupMetrics()
            self.status_message = String("No capture take loaded")
            return
        self.clean_take = self.raw_take.copy()
        self.clean_take.backend_id = self.selected_backend
        self.metrics = apply_mocap_cleanup_to_take(self.clean_take, self.cleanup_cfg)
        self.clean_take.selected_track_id = self.raw_take.selected_track_id
        if self.clean_take.num_frames() > 0:
            self.current_frame = max(0, min(self.clean_take.num_frames() - 1, self.current_frame))
        var pct = _slide_reduction_pct(self.metrics.foot_slide_before_cm, self.metrics.foot_slide_after_cm)
        self.status_message = (
            "Cleanup ready ("
            + _fmt_f32_1dp(self.metrics.foot_slide_before_cm)
            + " -> "
            + _fmt_f32_1dp(self.metrics.foot_slide_after_cm)
            + " cm, -"
            + String(pct)
            + "%)"
        )

    def step_playback(mut self, dt: Float32):
        """Advances take playback deterministically at `clean_take.fps` when `playing` is enabled."""
        if not self.has_take:
            return
        var nf = self.clean_take.num_frames()
        if nf <= 0:
            self.current_frame = 0
            return
        self.current_frame = max(0, min(nf - 1, self.current_frame))
        if not self.playing or not isfinite(dt) or dt <= Float32(0.0):
            return
        var fps = self.clean_take.fps if (isfinite(self.clean_take.fps) and self.clean_take.fps > Float32(1.0)) else Float32(30.0)
        var frame_dt = Float64(1.0) / Float64(fps)
        var accum = Float64(self.playback_accum) + Float64(min(dt, Float32(0.25)))
        while accum + 1.0e-6 >= frame_dt:
            accum = max(0.0, accum - frame_dt)
            self.current_frame = (self.current_frame + 1) % nf
        self.playback_accum = Float32(accum)


struct MocapPanelResult(Copyable, Movable):
    """Crisp per-frame action flags emitted by `draw_mocap_panel`."""
    var run_capture_clicked: Bool
    var load_demo_clicked: Bool
    var apply_to_scene_clicked: Bool
    var export_bvh_clicked: Bool
    var export_glb_clicked: Bool
    var cleanup_changed: Bool
    var frame_scrubbed: Bool
    var track_changed: Bool

    def __init__(out self):
        self.run_capture_clicked = False
        self.load_demo_clicked = False
        self.apply_to_scene_clicked = False
        self.export_bvh_clicked = False
        self.export_glb_clicked = False
        self.cleanup_changed = False
        self.frame_scrubbed = False
        self.track_changed = False


# ============================================================================
# 2. Formatting, Color & Micro-Widget Helpers (Obsidian & Flint Ember)
# ============================================================================


def _fmt_f32_1dp(v: Float32) -> String:
    if not isfinite(v):
        return String("0.0")
    var neg = v < Float32(0.0)
    var mag = -v if neg else v
    var scaled = Int(mag * Float32(10.0) + Float32(0.5))
    if scaled == 0:
        neg = False
    var whole = scaled // 10
    var frac = scaled % 10
    return ("-" if neg else "") + String(whole) + "." + String(frac)


def _slide_reduction_pct(before_cm: Float32, after_cm: Float32) -> Int:
    if not isfinite(before_cm) or before_cm <= Float32(1.0e-4):
        return 0
    var a = after_cm if isfinite(after_cm) else before_cm
    var ratio = (before_cm - a) / before_cm
    return max(0, min(99, Int(ratio * Float32(100.0) + Float32(0.5))))


@always_inline
def _confidence_color(conf: Float32, theme: StudioTheme) -> UInt32:
    var c = conf if isfinite(conf) else Float32(0.0)
    if c >= Float32(0.80):
        return theme.status_ok
    if c >= Float32(0.62):
        return theme.status_warn
    return theme.status_danger


def _backend_info_for(state: MocapStudioState) -> MocapBackendInfo:
    for i in range(len(state.backends)):
        if state.backends[i].id == state.selected_backend:
            return state.backends[i].copy()
    if len(state.backends) > 0:
        return state.backends[0].copy()
    var fallback = MocapBackendInfo()
    fallback.id = state.selected_backend
    if state.selected_backend == MOCAP_BACKEND_RTMPOSE_MBERT:
        fallback.name = String("RTMPose + MotionBERT")
        fallback.license_summary = String("Apache-2.0 / MIT")
        fallback.cache_subdir = String("rtmpose_mbert")
        fallback.status_detail = String("~/.cache/flint/models/rtmpose_mbert")
    else:
        fallback.name = String("GEM-X / SOMA-X")
        fallback.license_summary = String("Apache-2.0 / Open")
        fallback.cache_subdir = String("gemx")
        fallback.status_detail = String("~/.cache/flint/models/gemx")
    fallback.commercial_safe = True
    fallback.weights_present = False
    return fallback^


def _push_local_box(
    mut ui: UiContext,
    id: String,
    content_rect: RectF,
    direction: Int = FLEX_COL,
    gap: Float32 = 4.0,
    pad: Float32 = 0.0,
):
    var st = FlexStyle.row(gap=gap, pad=pad) if direction == FLEX_ROW else FlexStyle.col(gap=gap, pad=pad)
    st.fixed_w = content_rect.w
    st.fixed_h = content_rect.h
    var wid = ui.widget_id(id)
    var f_idx = ui.flex.add_node(-1, st, content_rect.w, content_rect.h)
    ui.dock_flex_root_idxs.append(f_idx)
    ui.dock_flex_root_rects.append(content_rect)
    ui.box_stack.append(
        _BoxScope(
            wid,
            f_idx,
            content_rect,
            content_rect.x + pad,
            content_rect.y + pad,
            direction,
            gap,
            pad,
            pad,
            content_rect.x + pad,
            content_rect.y + pad,
        )
    )
    ui.id_stack.append(wid)


def _pop_local_box(mut ui: UiContext):
    if len(ui.box_stack) > 1:
        _ = ui.box_stack.pop()
        _ = ui.id_stack.pop()


def _begin_card_bg(mut ui: UiContext, x: Float32, y: Float32, w: Float32, theme: StudioTheme) -> Int:
    """Reserves an Obsidian `L2_card` surface primitive whose bottom `y1` is finalized dynamically."""
    var idx = len(ui.draw_list.prims)
    ui.draw_list.add_card_surface(
        x, y, w, Float32(12.0),
        theme.l2_card, theme.border_subtle,
        RADIUS_MD, Float32(1.0), True, False,
    )
    return idx


def _end_card_bg(mut ui: UiContext, prim_idx: Int, bottom_y: Float32):
    if prim_idx >= 0 and prim_idx < len(ui.draw_list.prims) and isfinite(bottom_y):
        var y0 = ui.draw_list.prims[prim_idx].y0
        ui.draw_list.prims[prim_idx].y1 = max(y0 + Float32(4.0), bottom_y)


def _draw_compact_pill_btn(
    mut ui: UiContext,
    wid: UInt64,
    r: RectF,
    label: String,
    theme: StudioTheme,
    active: Bool = False,
    accent: UInt32 = 0,
    icon_id: Int = -1,
    primary_fill: Bool = False,
) -> Bool:
    """Tactile Obsidian direct-rect button with spring hover/press, `:focus-visible`, and zero clipping."""
    ui._register_focusable(wid, allow_auto_focus=False)
    var blocked = ui._blocked_by_modal() or ui.is_disabled()
    var ptr_blocked = blocked or ui._is_pointer_blocked_by_popup(wid)
    var hov = (not ptr_blocked) and r.contains(ui.mouse_x, ui.mouse_y)
    if hov:
        ui.hovered_id = wid
        ui.hovered_rect = ui._clip_hovered_rect(r)
        if ui.mouse_pressed_l:
            ui.active_id = wid
            ui._set_focused(wid)
    var clicked = (not blocked) and (
        (hov and ui.mouse_released_l and ui.active_id == wid)
        or (ui.focused_id == wid and (ui.key_enter_pressed or ui.key_space_pressed or ui.pad_confirm))
    )
    var pressed = (not blocked) and ui.active_id == wid and ui.mouse_down_l
    var sp = ui.animator.drive_widget(wid, hov, pressed, (not blocked) and ui.show_focus_ring(wid))
    var h_t = sp[0]
    var p_t = sp[1]
    var f_t = sp[2]

    var acc = accent if accent != 0 else theme.accent_primary
    var bg: UInt32
    var brd: UInt32
    var fg: UInt32
    if primary_fill:
        var hov_c = lerp_rgba8(acc, pack_rgba8(255, 255, 255, 255), Float32(0.14))
        var prs_c = lerp_rgba8(acc, pack_rgba8(0, 0, 0, 255), Float32(0.16))
        bg = lerp_rgba8(lerp_rgba8(acc, hov_c, h_t), prs_c, p_t)
        brd = rgba8_with_alpha(acc, 225)
        fg = theme.text_on_accent
    elif active:
        bg = lerp_rgba8(lerp_rgba8(rgba8_scale_alpha(acc, Float32(0.18)), rgba8_scale_alpha(acc, Float32(0.28)), h_t), theme.l3_press, p_t)
        brd = rgba8_with_alpha(acc, 180)
        fg = theme.text_primary
    else:
        bg = lerp_rgba8(lerp_rgba8(theme.l3_raised, theme.l3_hover, h_t), theme.l3_press, p_t)
        brd = lerp_rgba8(theme.border_control, theme.border_strong, h_t)
        fg = lerp_rgba8(theme.text_secondary, theme.text_primary, max(h_t, Float32(0.65)))

    if f_t > Float32(0.02):
        ui.draw_list.add_glow_rect(
            r.x, r.y, r.w, r.h,
            bg, rgba8_scale_alpha(theme.focus_ring, f_t),
            RADIUS_SM, Float32(2.5) + Float32(3.0) * f_t, Float32(1.2), UI_FLAG_TOP_BEVEL,
        )
    else:
        ui.draw_list.add_card_surface(r.x, r.y, r.w, r.h, bg, brd, RADIUS_SM, Float32(1.0), True, False)

    var fsz = theme.font_caption if r.h < Float32(25.0) else theme.font_body
    var fstyle = UI_FONT_BOLD if (active or primary_fill) else UI_FONT_REGULAR
    var has_lbl = label.byte_length() > 0
    var ico_col = fg if primary_fill else (acc if active else fg)
    if icon_id >= 0 and has_lbl:
        var isz = max(Float32(10.5), min(Float32(13.5), r.h - Float32(10.0)))
        var tw = ui._measure_text_styled(label, fsz, fstyle)
        var cw = isz + Float32(5.0) + tw
        var sx = r.x + max(Float32(6.0), (r.w - cw) * Float32(0.5))
        draw_studio_icon(ui.draw_list, icon_id, sx, r.center_y() - isz * Float32(0.5), isz, ico_col, Float32(1.45))
        var tx = sx + isz + Float32(5.0)
        ui.draw_text_in_rect(
            RectF(tx, r.y, max(Float32(8.0), r.right() - tx - Float32(4.0)), r.h),
            label, fsz, fg, UI_ALIGN_LEFT, fstyle, Float32(0.0), True, False,
        )
    elif icon_id >= 0:
        var isz = max(Float32(11.0), min(Float32(14.5), r.h - Float32(9.0)))
        draw_studio_icon(ui.draw_list, icon_id, r.center_x() - isz * Float32(0.5), r.center_y() - isz * Float32(0.5), isz, ico_col, Float32(1.45))
    elif has_lbl:
        ui.draw_text_in_rect(r, label, fsz, fg, UI_ALIGN_CENTER, fstyle, Float32(4.0), True, False)
    ui.last_item_id = wid
    ui.last_item_rect = r
    return clicked


def _draw_status_badge(
    mut ui: UiContext,
    x: Float32,
    y: Float32,
    h: Float32,
    text: String,
    accent: UInt32,
    font_style: Int = UI_FONT_BOLD,
    dot: Bool = False,
) -> Float32:
    """Emits a crisp tinted semantic pill badge with optional leading status dot and returns its width."""
    var fsz = Float32(10.0) if h <= Float32(18.0) else ui.studio_theme.font_caption
    var tw = ui._measure_text_styled(text, fsz, font_style)
    var dot_w = Float32(9.0) if dot else Float32(0.0)
    var bw = max(h, tw + dot_w + Float32(12.0))
    var rad = h * Float32(0.5)
    ui.draw_list.add_rect_styled(
        x, y, bw, h,
        rgba8_scale_alpha(accent, Float32(0.16)),
        rgba8_with_alpha(accent, 96),
        rad, Float32(1.0), Float32(1.0),
        UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
    )
    if dot:
        ui.draw_list.add_circle(x + Float32(7.5), y + h * Float32(0.5), Float32(2.4), accent)
        ui.draw_text_in_rect(
            RectF(x + Float32(13.0), y, max(Float32(8.0), bw - Float32(18.0)), h),
            text, fsz, accent, UI_ALIGN_LEFT, font_style, Float32(0.0), True, False,
        )
    else:
        ui.draw_text_in_rect(RectF(x, y, bw, h), text, fsz, accent, UI_ALIGN_CENTER, font_style, Float32(4.0), True, False)
    return bw


def _draw_panel_scrollbar(
    mut ui: UiContext,
    wid: UInt64,
    area: RectF,
    content_h: Float32,
    mut scroll_y: Float32,
    theme: StudioTheme,
    ptr_blocked: Bool,
):
    var max_scroll = max(Float32(0.0), content_h - area.h)
    if max_scroll <= Float32(0.5):
        scroll_y = Float32(0.0)
        return
    scroll_y = max(Float32(0.0), min(max_scroll, scroll_y))
    var hit_w = Float32(10.0)
    var track_y = area.y + Float32(4.0)
    var track_h = max(Float32(16.0), area.h - Float32(8.0))
    var thumb_h = max(Float32(24.0), track_h * (area.h / max(Float32(1.0), content_h)))
    var travel = max(Float32(1.0), track_h - thumb_h)
    var hit_r = RectF(area.right() - hit_w, track_y, hit_w, track_h)
    var hov = (not ptr_blocked) and hit_r.contains(ui.mouse_x, ui.mouse_y)
    if hov and ui.mouse_pressed_l:
        ui.active_id = wid
    var drag = ui.active_id == wid and ui.mouse_down_l and (not ui._blocked_by_modal())
    if drag and isfinite(ui.mouse_y):
        var rel = max(Float32(0.0), min(Float32(1.0), (ui.mouse_y - track_y - thumb_h * Float32(0.5)) / travel))
        scroll_y = rel * max_scroll
    var sp = ui.animator.drive_widget(wid, hov, drag, False)
    var hot_t = max(sp[0], sp[1])
    var sb_w = Float32(4.0) + Float32(2.0) * hot_t
    var thumb_y = track_y + (scroll_y / max_scroll) * travel
    var thumb_x = area.right() - sb_w - Float32(2.0)
    if hot_t > Float32(0.02):
        ui.draw_list.add_rect(
            thumb_x, track_y, sb_w, track_h,
            rgba8_scale_alpha(theme.l2_card, hot_t * Float32(0.55)),
            sb_w * Float32(0.5),
        )
    var thumb_col = theme.accent_primary if drag else lerp_rgba8(theme.border_strong, theme.text_secondary, hot_t)
    ui.draw_list.add_rect(thumb_x, thumb_y, sb_w, thumb_h, thumb_col, sb_w * Float32(0.5))


def _draw_step_chevron(mut dl: UiDrawList, cx: Float32, cy: Float32, pointing_right: Bool, col: UInt32):
    var s = Float32(3.5)
    if pointing_right:
        dl.add_line(cx - s * Float32(0.6), cy - s, cx + s * Float32(0.6), cy, Float32(1.5), col)
        dl.add_line(cx + s * Float32(0.6), cy, cx - s * Float32(0.6), cy + s, Float32(1.5), col)
        dl.add_line(cx + s * Float32(0.9), cy - s, cx + s * Float32(0.9), cy + s, Float32(1.4), col)
    else:
        dl.add_line(cx + s * Float32(0.6), cy - s, cx - s * Float32(0.6), cy, Float32(1.5), col)
        dl.add_line(cx - s * Float32(0.6), cy, cx + s * Float32(0.6), cy + s, Float32(1.5), col)
        dl.add_line(cx - s * Float32(0.9), cy - s, cx - s * Float32(0.9), cy + s, Float32(1.4), col)


# ============================================================================
# 3. Flint Studio RGBMocap Right-Dock Panel (`draw_mocap_panel`)
# ============================================================================


def draw_mocap_panel(
    mut ui: UiContext,
    mut state: MocapStudioState,
    rect: RectF,
    theme: StudioTheme,
) -> MocapPanelResult:
    """Renders the Obsidian & Flint Ember RGBMocap panel inside `rect` with zero clipping
    at `300..360px` width and full determinism under `UI_ANIM_INSTANT`."""
    var res = MocapPanelResult()
    if rect.w < Float32(120.0) or rect.h < Float32(100.0):
        return res^

    var prev_v2 = ui.dock_style_v2
    ui.dock_style_v2 = True
    ui.set_studio_theme(theme.copy())

    var cx0 = Int(rect.x)
    var cy0 = Int(rect.y)
    var cx1 = Int(rect.right())
    var cy1 = Int(rect.bottom())
    ui.draw_list.push_clip(cx0, cy0, cx1, cy1)

    # L1 Obsidian panel shell
    ui.draw_list.add_rect(rect.x, rect.y, rect.w, rect.h, theme.l1_shell)

    var blocked = ui._blocked_by_modal()
    var ptr_blocked = blocked or ui._is_pointer_blocked_by_popup()

    var pad = Float32(6.0)
    var footer_h = Float32(38.0)
    var scroll_area = RectF(rect.x, rect.y, rect.w, max(Float32(48.0), rect.h - footer_h))
    var footer_rect = RectF(rect.x, scroll_area.bottom(), rect.w, max(Float32(0.0), rect.bottom() - scroll_area.bottom()))

    # Handle mouse-wheel scrolling over the card area
    var prev_scroll_max = ui._get_state(ui.widget_id("mocap_panel_scroll")).scroll_max_y
    if (not ptr_blocked) and scroll_area.contains(ui.mouse_x, ui.mouse_y) and ui.mouse_wheel != Float32(0.0):
        state.scroll_y = max(Float32(0.0), min(max(Float32(0.0), prev_scroll_max), state.scroll_y - ui.mouse_wheel * Float32(36.0)))
    state.scroll_y = max(Float32(0.0), min(max(Float32(0.0), prev_scroll_max), state.scroll_y))

    var sb_reserve = Float32(7.0) if prev_scroll_max > Float32(0.5) else Float32(0.0)
    var card_x = scroll_area.x + pad
    var card_w = max(Float32(120.0), scroll_area.w - pad * Float32(2.0) - sb_reserve)
    var inner_pad = Float32(8.0)
    var inner_x = card_x + inner_pad
    var inner_w = max(Float32(96.0), card_w - inner_pad * Float32(2.0))
    var card_gap = Float32(6.0)

    # Clip scrollable card viewport
    ui.draw_list.push_clip(Int(scroll_area.x), Int(scroll_area.y), Int(scroll_area.right()), Int(scroll_area.bottom()))

    # Push a scroll scope so `slider_float_v2` / `toggle_switch` / `text_input` place deterministically at cursor_y
    var scroll_wid = ui.widget_id("mocap_panel_scroll")
    var content_start_y = scroll_area.y + pad - state.scroll_y
    var scroll_box_rect = RectF(card_x, content_start_y, card_w, Float32(4000.0))
    _push_local_box(ui, "mocap_panel_root", scroll_box_rect, FLEX_COL, Float32(0.0), Float32(0.0))
    from flint.ui.context import _ScrollScope
    ui.scroll_stack.append(_ScrollScope(scroll_wid, -1, scroll_area, content_start_y, state.scroll_y, Float32(-1.0)))

    var cur_y = content_start_y
    var binfo = _backend_info_for(state)
    var use_unicode_arrow = ui.font.has_source or ui.font.glyph_for(0x2192) >= 0
    var use_unicode_emdash = ui.font.has_source or ui.font.glyph_for(0x2014) >= 0
    var arrow_str = String(" → ") if use_unicode_arrow else String(" -> ")
    var dash_str = String(" — ") if use_unicode_emdash else String(" - ")

    # ========================================================================
    # CARD 1: Header & Backend / Model Manager Card (`L2_card`)
    # ========================================================================
    var c1_bg = _begin_card_bg(ui, card_x, cur_y, card_w, theme)
    var y = cur_y + Float32(6.0)

    # Title row: 20x20 Ember icon badge + Bold "RGBMocap" + commercial-safe emerald badge
    var ib_sz = Float32(20.0)
    ui.draw_list.add_card_surface(
        inner_x, y, ib_sz, ib_sz,
        theme.accent_soft, rgba8_with_alpha(theme.accent_primary, 108),
        RADIUS_SM, Float32(1.0), True, False,
    )
    draw_studio_icon(
        ui.draw_list, ICON_CAMERA,
        inner_x + Float32(4.0), y + Float32(4.0), Float32(12.0),
        theme.accent_primary, Float32(1.45),
    )

    var lic_text = String("Apache-2.0 / Open") if binfo.commercial_safe else String("Non-Commercial")
    var lic_col = theme.status_ok if binfo.commercial_safe else theme.status_warn
    var lic_tw = ui._measure_text_styled(lic_text, Float32(10.0), UI_FONT_BOLD)
    var lic_w = max(Float32(72.0), lic_tw + Float32(21.0))
    if lic_w > inner_w * Float32(0.54):
        lic_text = String("Apache-2.0") if binfo.commercial_safe else String("Check Lic")
        lic_tw = ui._measure_text_styled(lic_text, Float32(10.0), UI_FONT_BOLD)
        lic_w = max(Float32(56.0), lic_tw + Float32(21.0))
    var lic_x = inner_x + inner_w - lic_w
    _ = _draw_status_badge(ui, lic_x, y + Float32(1.0), Float32(18.0), lic_text, lic_col, UI_FONT_BOLD, dot=True)

    var title_x = inner_x + ib_sz + Float32(7.0)
    var title_w = max(Float32(36.0), lic_x - title_x - Float32(6.0))
    ui.draw_text_in_rect(
        RectF(title_x, y, title_w, ib_sz),
        "RGBMocap", theme.font_title, theme.text_primary,
        UI_ALIGN_LEFT, UI_FONT_BOLD, Float32(0.0), True, False,
    )
    y += ib_sz + Float32(5.0)

    # Segmented backend selector (`GEM-X` | `RTMPose`) with spring-animated pill + Ember accent bar
    var seg_h = Float32(24.0)
    var seg_r = RectF(inner_x, y, inner_w, seg_h)
    var seg_wid = ui.widget_id("mocap_backend_seg")
    ui.draw_list.add_recessed_well(seg_r.x, seg_r.y, seg_r.w, seg_r.h, theme.l0_canvas, theme.border_subtle, RADIUS_SM)
    var seg_half_w = (seg_r.w - Float32(6.0)) * Float32(0.5)
    var sel_idx = 0 if state.selected_backend == MOCAP_BACKEND_GEMX else 1
    var target_px = seg_r.x + Float32(3.0) + Float32(sel_idx) * seg_half_w
    var glide = ui.animator.drive_tab_pill(seg_wid, target_px, seg_half_w)
    ui.draw_list.add_card_surface(
        glide[0], seg_r.y + Float32(2.5), glide[1], seg_r.h - Float32(5.0),
        theme.l3_raised, rgba8_with_alpha(theme.accent_primary, 130),
        RADIUS_XS, Float32(1.0), True, False,
    )
    ui.draw_list.add_accent_bar(
        glide[0] + Float32(10.0), seg_r.bottom() - Float32(3.0),
        max(Float32(8.0), glide[1] - Float32(20.0)), Float32(2.0),
        theme.accent_primary, True,
    )
    var seg_names = [String("GEM-X"), String("RTMPose")]
    var seg_ids = [MOCAP_BACKEND_GEMX, MOCAP_BACKEND_RTMPOSE_MBERT]
    for si in range(2):
        var sr = RectF(seg_r.x + Float32(3.0) + Float32(si) * seg_half_w, seg_r.y + Float32(2.0), seg_half_w, seg_r.h - Float32(4.0))
        var is_cur = sel_idx == si
        var hov = (not ptr_blocked) and sr.contains(ui.mouse_x, ui.mouse_y)
        if hov:
            ui.hovered_id = hash_ui_id(seg_wid, String(si))
            ui.hovered_rect = ui._clip_hovered_rect(sr)
            if ui.mouse_pressed_l and state.selected_backend != seg_ids[si]:
                state.selected_backend = seg_ids[si]
                binfo = _backend_info_for(state)
                state.raw_take.backend_id = state.selected_backend
                state.clean_take.backend_id = state.selected_backend
                state.status_message = String("Backend: ") + binfo.name
        ui.draw_text_in_rect(
            sr, seg_names[si], theme.font_body,
            theme.text_primary if (is_cur or hov) else theme.text_secondary,
            UI_ALIGN_CENTER, UI_FONT_BOLD if is_cur else UI_FONT_REGULAR,
            Float32(4.0), True, False,
        )
    y += seg_h + Float32(5.0)

    # Live cache probe pill (`Weights Ready` or `Cache Empty — Demo Active`) + Re-probe button
    var probe_h = Float32(18.0)
    var ref_w = Float32(20.0)
    var ref_r = RectF(inner_x + inner_w - ref_w, y, ref_w, probe_h)
    var probe_txt = String("Weights Ready") if binfo.weights_present else (String("Cache Empty") + dash_str + String("Demo Active"))
    var probe_col = theme.status_ok if binfo.weights_present else theme.status_warn
    var max_pill_w = max(Float32(64.0), inner_w - ref_w - Float32(6.0))
    var probe_disp = ui._truncate_text_ellipsis_styled(probe_txt, max(Float32(24.0), max_pill_w - Float32(20.0)), Float32(10.0), UI_FONT_BOLD)
    _ = _draw_status_badge(ui, inner_x, y, Float32(18.0), probe_disp, probe_col, UI_FONT_BOLD, dot=True)
    var ref_wid = ui.widget_id("mocap_reprobe")
    if _draw_compact_pill_btn(ui, ref_wid, ref_r, "", theme, False, theme.accent_primary, ICON_REFRESH, False):
        state.backends = probe_mocap_backends()
        binfo = _backend_info_for(state)
        state.status_message = String("Cache probed: ") + (String("ready") if binfo.weights_present else String("demo active"))
    if ui.hovered_id == ref_wid:
        ui.queue_tooltip_rect(ref_wid, ref_r, "Re-scan local model cache")
    y += probe_h + Float32(3.0)

    # Local cache path / export hint (mono, middle-ellipsis, tooltip on hover)
    var hint_str = binfo.status_detail if binfo.status_detail.byte_length() > 0 else (String("~/.cache/flint/models/") + binfo.cache_subdir)
    var hint_r = RectF(inner_x, y, inner_w, Float32(14.0))
    ui.draw_text_in_rect(
        hint_r, hint_str, Float32(10.0), theme.text_muted,
        UI_ALIGN_LEFT, UI_FONT_MONO, Float32(0.0), True, True,
    )
    if (not ptr_blocked) and hint_r.contains(ui.mouse_x, ui.mouse_y):
        ui.queue_tooltip_rect(ui.widget_id("mocap_cache_hint"), hint_r, hint_str)
    y += Float32(14.0) + Float32(6.0)

    _end_card_bg(ui, c1_bg, y)
    cur_y = y + card_gap

    # ========================================================================
    # CARD 2: Source & Multi-Person Track Card (`L2_card`)
    # ========================================================================
    var c2_bg = _begin_card_bg(ui, card_x, cur_y, card_w, theme)
    y = cur_y + Float32(6.0)

    # Header row + frame count & FPS mono telemetry
    var hdr2_h = Float32(17.0)
    draw_studio_icon(ui.draw_list, ICON_FOLDER, inner_x, y + Float32(2.0), Float32(13.0), theme.accent_primary, Float32(1.4))
    var nf_take = state.clean_take.num_frames() if state.has_take else 0
    var fps_i = Int(state.clean_take.fps + Float32(0.5)) if (state.has_take and isfinite(state.clean_take.fps) and state.clean_take.fps > Float32(0.0)) else 30
    var telem_str = (String(nf_take) + "f | " + String(fps_i) + " fps") if state.has_take else String("No take")
    var telem_w = ui._measure_text_styled(telem_str, Float32(10.5), UI_FONT_MONO) + Float32(4.0)
    ui.draw_text_in_rect(
        RectF(inner_x + Float32(18.0), y, max(Float32(40.0), inner_w - Float32(18.0) - telem_w - Float32(4.0)), hdr2_h),
        "Source & Subject", theme.font_section, theme.text_primary,
        UI_ALIGN_LEFT, UI_FONT_BOLD, Float32(0.0), True, False,
    )
    ui.draw_text_in_rect(
        RectF(inner_x + inner_w - telem_w, y, telem_w, hdr2_h),
        telem_str, Float32(10.5), theme.text_secondary,
        UI_ALIGN_RIGHT, UI_FONT_MONO, Float32(0.0), True, False,
    )
    y += hdr2_h + Float32(5.0)

    # Source video / folder text input via UiContext
    var top_i = len(ui.box_stack) - 1
    ui.box_stack[top_i].cursor_x = inner_x
    ui.box_stack[top_i].cursor_y = y
    _ = ui.text_input("mocap_source_path", state.source_path, inner_w, "Video file or image folder...")
    y = ui.last_item_rect.bottom() + Float32(5.0)

    # [Demo] (BTN_SECONDARY) & [Capture] (BTN_PRIMARY Flint Ember) row
    var act_h = Float32(24.0)
    var btn_gap = Float32(6.0)
    var demo_w = max(Float32(72.0), (inner_w - btn_gap) * Float32(0.40))
    var cap_w = max(Float32(72.0), inner_w - btn_gap - demo_w)
    var row_r = RectF(inner_x, y, inner_w, act_h)
    _push_local_box(ui, "mocap_src_actions", row_r, FLEX_ROW, btn_gap, Float32(0.0))
    if ui.button_v2("mocap_btn_demo", "Demo", BTN_SECONDARY, ICON_SPARK, demo_w, act_h):
        state.load_demo_take()
        res.load_demo_clicked = True
    var cap_lbl = String("Running...") if state.job_running else String("Capture")
    if ui.button_v2("mocap_btn_capture", cap_lbl, BTN_PRIMARY, ICON_PLAY, cap_w, act_h, state.job_running):
        res.run_capture_clicked = True
    _pop_local_box(ui)
    y += act_h + Float32(5.0)

    # Optional job progress bar when capture job is active
    if state.job_running:
        var p_01 = max(Float32(0.0), min(Float32(1.0), state.job_progress if isfinite(state.job_progress) else Float32(0.0)))
        ui.draw_list.add_rect(inner_x, y, inner_w, Float32(5.0), theme.l0_canvas, Float32(2.5), Float32(1.0), theme.border_subtle)
        if p_01 > Float32(0.01):
            ui.draw_list.add_rect(inner_x + Float32(1.0), y + Float32(1.0), max(Float32(3.0), (inner_w - Float32(2.0)) * p_01), Float32(3.0), theme.accent_primary, Float32(1.5))
        y += Float32(9.0)

    # Multi-Person Track Selector pills (`Person #1 94%` | `Person #2 78%`)
    var n_tracks = len(state.clean_take.tracks) if state.has_take else 0
    if n_tracks > 0:
        var shown_tracks = min(n_tracks, 3)
        var t_gap = Float32(5.0)
        var pill_h = Float32(22.0)
        var pill_w = max(Float32(60.0), (inner_w - t_gap * Float32(shown_tracks - 1)) / Float32(shown_tracks))
        for ti in range(shown_tracks):
            var tid = state.clean_take.tracks[ti].track_id
            var conf_pct = max(0, min(100, Int(state.clean_take.tracks[ti].mean_confidence * Float32(100.0) + Float32(0.5))))
            var is_sel = state.clean_take.selected_track_id == tid
            var tr = RectF(inner_x + Float32(ti) * (pill_w + t_gap), y, pill_w, pill_h)
            var twid = ui.widget_id("mocap_track_" + String(tid))
            var hov = (not ptr_blocked) and tr.contains(ui.mouse_x, ui.mouse_y)
            if hov:
                ui.hovered_id = twid
                ui.hovered_rect = ui._clip_hovered_rect(tr)
                if ui.mouse_pressed_l and not is_sel:
                    state.raw_take.selected_track_id = tid
                    state.clean_take.selected_track_id = tid
                    state.status_message = String("Selected Person #") + String(tid)
                    res.track_changed = True
            var t_bg = theme.l3_raised if is_sel else (theme.l3_hover if hov else theme.l0_canvas)
            var t_brd = rgba8_with_alpha(theme.accent_primary, 185) if is_sel else (theme.border_strong if hov else theme.border_subtle)
            ui.draw_list.add_card_surface(tr.x, tr.y, tr.w, tr.h, t_bg, t_brd, RADIUS_SM, Float32(1.0), is_sel, False)
            if is_sel:
                ui.draw_list.add_accent_bar(tr.x + Float32(2.5), tr.y + Float32(3.5), Float32(2.5), tr.h - Float32(7.0), theme.accent_primary, True)
            var p_lbl = String("Person #") + String(tid)
            var pct_lbl = String(conf_pct) + "%"
            var pct_col = _confidence_color(state.clean_take.tracks[ti].mean_confidence, theme)
            var pct_w = ui._measure_text_styled(pct_lbl, Float32(10.0), UI_FONT_MONO) + Float32(6.0)
            var lx = tr.x + (Float32(8.0) if is_sel else Float32(6.0))
            var lw = max(Float32(24.0), tr.right() - lx - pct_w - Float32(4.0))
            ui.draw_text_in_rect(
                RectF(lx, tr.y, lw, tr.h),
                p_lbl, theme.font_caption,
                theme.text_primary if is_sel else theme.text_secondary,
                UI_ALIGN_LEFT, UI_FONT_BOLD if is_sel else UI_FONT_REGULAR,
                Float32(0.0), True, False,
            )
            ui.draw_text_in_rect(
                RectF(tr.right() - pct_w - Float32(4.0), tr.y, pct_w, tr.h),
                pct_lbl, Float32(10.0), pct_col,
                UI_ALIGN_RIGHT, UI_FONT_MONO, Float32(0.0), False, False,
            )
        y += pill_h + Float32(6.0)
    else:
        ui.draw_text_in_rect(
            RectF(inner_x, y, inner_w, Float32(16.0)),
            "No subjects tracked - click Demo or Capture",
            theme.font_caption, theme.text_muted,
            UI_ALIGN_LEFT, UI_FONT_REGULAR, Float32(0.0), True, False,
        )
        y += Float32(20.0)

    _end_card_bg(ui, c2_bg, y)
    cur_y = y + card_gap

    # ========================================================================
    # CARD 3: Motion Cleanup Card (`L2_card` — `toyxyz_RGBMocap` parity)
    # ========================================================================
    var c3_bg = _begin_card_bg(ui, card_x, cur_y, card_w, theme)
    y = cur_y + Float32(6.0)

    var hdr3_h = Float32(18.0)
    var red_pct = _slide_reduction_pct(state.metrics.foot_slide_before_cm, state.metrics.foot_slide_after_cm)
    var red_txt = String("-") + String(red_pct) + "%"
    var red_tw = ui._measure_text_styled(red_txt, Float32(10.0), UI_FONT_MONO)
    var red_w = max(Float32(40.0), red_tw + Float32(14.0))
    var red_col = theme.status_ok if red_pct >= 50 else (theme.status_warn if red_pct > 0 else theme.text_muted)

    draw_studio_icon(ui.draw_list, ICON_SPARK, inner_x, y + Float32(2.5), Float32(13.0), theme.accent_primary, Float32(1.4))
    ui.draw_text_in_rect(
        RectF(inner_x + Float32(18.0), y, max(Float32(40.0), inner_w - Float32(18.0) - red_w - Float32(6.0)), hdr3_h),
        "Motion Cleanup", theme.font_section, theme.text_primary,
        UI_ALIGN_LEFT, UI_FONT_BOLD, Float32(0.0), True, False,
    )
    _ = _draw_status_badge(ui, inner_x + inner_w - red_w, y, Float32(18.0), red_txt, red_col, UI_FONT_MONO, dot=False)
    y += hdr3_h + Float32(5.0)

    # Live Foot Slide Reduction telemetry bar (`5.7 cm -> 0.6 cm  -90%`)
    var well_h = Float32(30.0)
    var well_r = RectF(inner_x, y, inner_w, well_h)
    ui.draw_list.add_recessed_well(well_r.x, well_r.y, well_r.w, well_r.h, theme.l0_canvas, theme.border_subtle, RADIUS_SM)
    var slide_str = (
        _fmt_f32_1dp(state.metrics.foot_slide_before_cm)
        + " cm"
        + arrow_str
        + _fmt_f32_1dp(state.metrics.foot_slide_after_cm)
        + " cm"
    )
    var lbl_fs_w = Float32(64.0)
    ui.draw_text_in_rect(
        RectF(well_r.x + Float32(8.0), well_r.y + Float32(2.0), lbl_fs_w, Float32(15.0)),
        "Foot Slide", Float32(10.5), theme.text_muted,
        UI_ALIGN_LEFT, UI_FONT_REGULAR, Float32(0.0), True, False,
    )
    ui.draw_text_in_rect(
        RectF(well_r.x + Float32(8.0) + lbl_fs_w, well_r.y + Float32(2.0), max(Float32(40.0), well_r.w - Float32(16.0) - lbl_fs_w), Float32(15.0)),
        slide_str, Float32(10.5), theme.text_primary,
        UI_ALIGN_RIGHT, UI_FONT_MONO, Float32(0.0), True, False,
    )
    # Visual before/after bar
    var bar_x = well_r.x + Float32(8.0)
    var bar_y = well_r.y + Float32(20.0)
    var bar_w = max(Float32(24.0), well_r.w - Float32(16.0))
    var bar_h = Float32(5.0)
    ui.draw_list.add_rect(bar_x, bar_y, bar_w, bar_h, theme.l2_card, Float32(2.5))
    var ref_max_cm = max(Float32(8.0), state.metrics.foot_slide_before_cm)
    var bef_frac = max(Float32(0.0), min(Float32(1.0), state.metrics.foot_slide_before_cm / ref_max_cm))
    var aft_frac = max(Float32(0.0), min(Float32(1.0), state.metrics.foot_slide_after_cm / ref_max_cm))
    if bef_frac > Float32(0.01):
        ui.draw_list.add_rect(bar_x, bar_y, max(Float32(4.0), bar_w * bef_frac), bar_h, rgba8_scale_alpha(theme.status_warn, Float32(0.35)), Float32(2.5))
    if aft_frac > Float32(0.005):
        ui.draw_list.add_rect(bar_x, bar_y, max(Float32(4.0), bar_w * aft_frac), bar_h, theme.status_ok, Float32(2.5))
    y += well_h + Float32(4.0)

    # Filter & Contact Cleanup controls (`toggle_switch` + `slider_float_v2`)
    top_i = len(ui.box_stack) - 1
    ui.box_stack[top_i].cursor_x = inner_x
    ui.box_stack[top_i].cursor_y = y
    var cleanup_dirty = False

    if ui.toggle_switch("mc_one_euro", "One Euro", state.cleanup_cfg.enable_one_euro, inner_w):
        cleanup_dirty = True
    ui.push_disabled(not state.cleanup_cfg.enable_one_euro)
    if ui.slider_float_v2("mc_min_cutoff", "Min Cutoff", state.cleanup_cfg.one_euro.min_cutoff, Float32(0.10), Float32(5.00), inner_w):
        cleanup_dirty = True
    if ui.slider_float_v2("mc_beta", "Beta", state.cleanup_cfg.one_euro.beta, Float32(0.00), Float32(0.50), inner_w):
        cleanup_dirty = True
    ui.pop_disabled(not state.cleanup_cfg.enable_one_euro)

    if ui.toggle_switch("mc_foot_lock", "Foot Lock", state.cleanup_cfg.enable_foot_lock, inner_w):
        cleanup_dirty = True
    ui.push_disabled(not state.cleanup_cfg.enable_foot_lock)
    if ui.slider_float_v2("mc_lock_strength", "Strength", state.cleanup_cfg.foot_lock_strength, Float32(0.00), Float32(1.00), inner_w):
        cleanup_dirty = True
    if ui.slider_float_v2("mc_foot_height", "Height (m)", state.cleanup_cfg.foot_height_thresh_m, Float32(0.02), Float32(0.25), inner_w):
        cleanup_dirty = True
    ui.pop_disabled(not state.cleanup_cfg.enable_foot_lock)

    if ui.toggle_switch("mc_palm_lock", "Palm Contact", state.cleanup_cfg.enable_palm_lock, inner_w):
        cleanup_dirty = True
    if ui.toggle_switch("mc_ground_align", "Ground Align", state.cleanup_cfg.enable_ground_align, inner_w):
        cleanup_dirty = True

    if cleanup_dirty:
        state.recompute_cleanup()
        res.cleanup_changed = True

    y = ui.box_stack[top_i].cursor_y + Float32(6.0)
    _end_card_bg(ui, c3_bg, y)
    cur_y = y + card_gap

    # ========================================================================
    # CARD 4: Confidence & Contact Timeline Card (`L2_card`)
    # ========================================================================
    var c4_bg = _begin_card_bg(ui, card_x, cur_y, card_w, theme)
    y = cur_y + Float32(8.0)

    var hdr4_h = Float32(18.0)
    var nf = state.clean_take.num_frames() if state.has_take else 0
    var disp_f = (max(0, min(nf - 1, state.current_frame)) + 1) if nf > 0 else 0
    var frame_txt = "Frame " + String(disp_f) + " / " + String(nf)
    var frame_tw = ui._measure_text_styled(frame_txt, Float32(10.5), UI_FONT_MONO) + Float32(4.0)

    draw_studio_icon(ui.draw_list, ICON_CLOCK, inner_x, y + Float32(2.5), Float32(13.0), theme.accent_primary, Float32(1.4))
    ui.draw_text_in_rect(
        RectF(inner_x + Float32(18.0), y, max(Float32(40.0), inner_w - Float32(18.0) - frame_tw - Float32(4.0)), hdr4_h),
        "Take Timeline", theme.font_section, theme.text_primary,
        UI_ALIGN_LEFT, UI_FONT_BOLD, Float32(0.0), True, False,
    )
    ui.draw_text_in_rect(
        RectF(inner_x + inner_w - frame_tw, y, frame_tw, hdr4_h),
        frame_txt, Float32(10.5), theme.text_primary,
        UI_ALIGN_RIGHT, UI_FONT_MONO, Float32(0.0), True, False,
    )
    y += hdr4_h + Float32(6.0)

    # Interactive Scrubber + Per-Frame Confidence Heatbar + L/R Foot Contact Strips
    var tl_h = Float32(54.0)
    var tl_r = RectF(inner_x, y, inner_w, tl_h)
    var tl_wid = ui.widget_id("mocap_timeline_scrub")
    var tl_hov = (not ptr_blocked) and tl_r.contains(ui.mouse_x, ui.mouse_y)
    if tl_hov:
        ui.hovered_id = tl_wid
        ui.hovered_rect = ui._clip_hovered_rect(tl_r)
        if ui.mouse_pressed_l:
            ui.active_id = tl_wid
            ui._set_focused(tl_wid)

    var gutter_w = Float32(14.0)
    var track_x = tl_r.x + gutter_w + Float32(4.0)
    var track_w = max(Float32(24.0), tl_r.w - gutter_w - Float32(10.0))

    var scrubbing = (not blocked) and ui.active_id == tl_wid and ui.mouse_down_l
    if scrubbing and nf > 1 and isfinite(ui.mouse_x):
        var rel = max(Float32(0.0), min(Float32(1.0), (ui.mouse_x - track_x) / track_w))
        var next_f = max(0, min(nf - 1, Int(rel * Float32(nf - 1) + Float32(0.5))))
        if next_f != state.current_frame:
            state.current_frame = next_f
            res.frame_scrubbed = True
        state.playing = False

    ui.draw_list.add_recessed_well(
        tl_r.x, tl_r.y, tl_r.w, tl_r.h,
        theme.l0_canvas,
        theme.accent_primary if scrubbing else (theme.border_strong if tl_hov else theme.border_subtle),
        RADIUS_SM, Float32(1.0), scrubbing, theme.accent_primary,
    )

    var heat_y = tl_r.y + Float32(5.0)
    var heat_h = Float32(24.0)
    var div_y = heat_y + heat_h + Float32(3.0)
    var strip_l_y = div_y + Float32(3.0)
    var strip_r_y = strip_l_y + Float32(9.0)
    var strip_h = Float32(5.5)

    # Left micro-labels for confidence & L/R foot contact strips
    ui.draw_text_in_rect(RectF(tl_r.x + Float32(3.0), heat_y, gutter_w, heat_h), "C", Float32(9.0), theme.text_muted, UI_ALIGN_CENTER, UI_FONT_MONO)
    ui.draw_text_in_rect(RectF(tl_r.x + Float32(3.0), strip_l_y - Float32(1.0), gutter_w, strip_h + Float32(2.0)), "L", Float32(9.0), theme.status_ok, UI_ALIGN_CENTER, UI_FONT_MONO)
    ui.draw_text_in_rect(RectF(tl_r.x + Float32(3.0), strip_r_y - Float32(1.0), gutter_w, strip_h + Float32(2.0)), "R", Float32(9.0), theme.status_ok, UI_ALIGN_CENTER, UI_FONT_MONO)
    ui.draw_list.add_line(tl_r.x + Float32(4.0), div_y, tl_r.right() - Float32(4.0), div_y, Float32(1.0), theme.border_subtle)

    # Contact strip background tracks
    ui.draw_list.add_rect(track_x, strip_l_y, track_w, strip_h, rgba8_scale_alpha(theme.l2_card, Float32(0.85)), Float32(2.5))
    ui.draw_list.add_rect(track_x, strip_r_y, track_w, strip_h, rgba8_scale_alpha(theme.l2_card, Float32(0.85)), Float32(2.5))

    if nf > 0:
        # Per-frame confidence heatbar micro-bars (emerald / amber / red)
        var step_x = track_w / Float32(nf)
        var bar_w_f = max(Float32(1.2), step_x - (Float32(0.8) if step_x > Float32(2.4) else Float32(0.25)))
        for fi in range(nf):
            var c_val = max(Float32(0.05), min(Float32(1.0), state.clean_take.frames[fi].mean_confidence))
            var bh = max(Float32(3.0), heat_h * c_val)
            var bx = track_x + Float32(fi) * step_x
            var bcol = _confidence_color(c_val, theme)
            var alpha = Float32(1.0) if fi == state.current_frame else Float32(0.78)
            ui.draw_list.add_rect(bx, heat_y + heat_h - bh, bar_w_f, bh, rgba8_scale_alpha(bcol, alpha), Float32(1.0))

        # Coalesced Left & Right foot contact lock interval bars (clean continuous pills!)
        var fi_l = 0
        while fi_l < nf:
            if state.clean_take.frames[fi_l].left_foot_contact:
                var s0 = fi_l
                while fi_l + 1 < nf and state.clean_take.frames[fi_l + 1].left_foot_contact:
                    fi_l += 1
                var x0 = track_x + Float32(s0) * step_x
                var w_seg = max(Float32(3.0), Float32(fi_l - s0 + 1) * step_x)
                ui.draw_list.add_rect(x0, strip_l_y, w_seg, strip_h, theme.status_ok, Float32(2.5))
            fi_l += 1

        var fi_r = 0
        while fi_r < nf:
            if state.clean_take.frames[fi_r].right_foot_contact:
                var s0 = fi_r
                while fi_r + 1 < nf and state.clean_take.frames[fi_r + 1].right_foot_contact:
                    fi_r += 1
                var x0 = track_x + Float32(s0) * step_x
                var w_seg = max(Float32(3.0), Float32(fi_r - s0 + 1) * step_x)
                ui.draw_list.add_rect(x0, strip_r_y, w_seg, strip_h, theme.status_ok, Float32(2.5))
            fi_r += 1

        # Flint Ember playhead line + soft-halo thumb knob
        var play_frac = Float32(state.current_frame) / Float32(max(1, nf - 1))
        var ph_x = track_x + play_frac * track_w
        ui.draw_list.add_line(ph_x, tl_r.y + Float32(2.0), ph_x, tl_r.bottom() - Float32(2.0), Float32(1.8), theme.accent_primary)
        var knob_y = tl_r.y + Float32(5.5)
        ui.draw_list.add_circle(ph_x, knob_y, Float32(6.5), rgba8_scale_alpha(theme.accent_primary, Float32(0.28)))
        ui.draw_list.add_circle(ph_x, knob_y, Float32(4.2), theme.accent_primary, Float32(1.0), theme.text_primary)

    y += tl_h + Float32(7.0)

    # Transport (Prev / Play-Pause / Next) + Overlay Toggles (Skel | Frustum | Contacts)
    var tr_h = Float32(24.0)
    var prev_r = RectF(inner_x, y, Float32(22.0), tr_h)
    var play_r = RectF(prev_r.right() + Float32(3.0), y, Float32(26.0), tr_h)
    var next_r = RectF(play_r.right() + Float32(3.0), y, Float32(22.0), tr_h)

    var prev_wid = ui.widget_id("mocap_step_prev")
    if _draw_compact_pill_btn(ui, prev_wid, prev_r, "", theme, False, theme.accent_primary, -1, False):
        if nf > 0:
            state.playing = False
            state.current_frame = (state.current_frame - 1 + nf) % nf
            res.frame_scrubbed = True
    _draw_step_chevron(ui.draw_list, prev_r.center_x(), prev_r.center_y(), False, theme.text_primary)

    var play_wid = ui.widget_id("mocap_play_pause")
    if _draw_compact_pill_btn(
        ui, play_wid, play_r, "", theme,
        state.playing, theme.accent_primary,
        ICON_PAUSE if state.playing else ICON_PLAY,
        state.playing,
    ):
        state.playing = not state.playing
        state.playback_accum = Float32(0.0)

    var next_wid = ui.widget_id("mocap_step_next")
    if _draw_compact_pill_btn(ui, next_wid, next_r, "", theme, False, theme.accent_primary, -1, False):
        if nf > 0:
            state.playing = False
            state.current_frame = (state.current_frame + 1) % nf
            res.frame_scrubbed = True
    _draw_step_chevron(ui.draw_list, next_r.center_x(), next_r.center_y(), True, theme.text_primary)

    # Right-aligned overlay toggles (`Skel` | `Frustum` | `Contacts`)
    var ov_x = next_r.right() + Float32(8.0)
    var ov_avail = max(Float32(90.0), inner_x + inner_w - ov_x)
    var ov_gap = Float32(4.0)
    var ov_w_skel = max(Float32(32.0), (ov_avail - ov_gap * Float32(2.0)) * Float32(0.28))
    var ov_w_frus = max(Float32(38.0), (ov_avail - ov_gap * Float32(2.0)) * Float32(0.35))
    var ov_w_cont = max(Float32(38.0), ov_avail - ov_gap * Float32(2.0) - ov_w_skel - ov_w_frus)

    var r_skel = RectF(ov_x, y, ov_w_skel, tr_h)
    var r_frus = RectF(r_skel.right() + ov_gap, y, ov_w_frus, tr_h)
    var r_cont = RectF(r_frus.right() + ov_gap, y, ov_w_cont, tr_h)

    if _draw_compact_pill_btn(ui, ui.widget_id("mocap_ov_skel"), r_skel, "Skel", theme, state.show_skeleton_overlay, theme.accent_primary):
        state.show_skeleton_overlay = not state.show_skeleton_overlay
    if _draw_compact_pill_btn(ui, ui.widget_id("mocap_ov_frustum"), r_frus, "Frustum", theme, state.show_frustum_overlay, theme.agent_flow):
        state.show_frustum_overlay = not state.show_frustum_overlay
    if _draw_compact_pill_btn(ui, ui.widget_id("mocap_ov_contacts"), r_cont, "Contacts", theme, state.show_contact_rings, theme.status_ok):
        state.show_contact_rings = not state.show_contact_rings

    y += tr_h + Float32(7.0)

    # Compact status telemetry footer with kbd shortcut pill
    var kbd_w = ui.kbd_pill("Space", inner_x, y, Float32(16.0))
    ui.draw_text_in_rect(
        RectF(inner_x + kbd_w + Float32(6.0), y, max(Float32(32.0), inner_w - kbd_w - Float32(6.0)), Float32(16.0)),
        state.status_message, Float32(10.0), theme.text_muted,
        UI_ALIGN_LEFT, UI_FONT_REGULAR, Float32(0.0), True, False,
    )
    y += Float32(16.0) + Float32(8.0)

    _end_card_bg(ui, c4_bg, y)
    cur_y = y + pad

    # Pop scroll scope & record exact content height for next frame's scrollbar
    if len(ui.scroll_stack) > 0:
        _ = ui.scroll_stack.pop()
    _pop_local_box(ui)

    var total_content_h = max(Float32(0.0), cur_y - content_start_y + pad)
    var scroll_st = ui._get_state(scroll_wid)
    scroll_st.scroll_max_y = max(Float32(0.0), total_content_h - scroll_area.h)
    scroll_st.scroll_y = state.scroll_y
    ui.state_map[scroll_wid] = scroll_st

    _draw_panel_scrollbar(ui, ui.widget_id("mocap_panel_sb"), scroll_area, total_content_h, state.scroll_y, theme, ptr_blocked)
    ui.draw_list.pop_clip()

    # ========================================================================
    # STICKY FOOTER: Export & Retarget (`[Apply to Entity]`, `[BVH]`, `[GLB]`)
    # ========================================================================
    ui.draw_list.add_rect(footer_rect.x, footer_rect.y, footer_rect.w, footer_rect.h, theme.l1_shell)
    ui.draw_list.add_line(footer_rect.x, footer_rect.y, footer_rect.right(), footer_rect.y, Float32(1.0), theme.border_subtle)

    var foot_inner_w = max(Float32(120.0), footer_rect.w - pad * Float32(2.0))
    var f_gap = Float32(5.0)
    var f_btn_h = Float32(26.0)
    var f_btn_y = footer_rect.y + max(Float32(4.0), (footer_rect.h - f_btn_h) * Float32(0.5))
    var bvh_w = max(Float32(48.0), min(Float32(66.0), foot_inner_w * Float32(0.21)))
    var glb_w = bvh_w
    var apply_w = max(Float32(88.0), foot_inner_w - bvh_w - glb_w - f_gap * Float32(2.0))

    var foot_row_r = RectF(footer_rect.x + pad, f_btn_y, foot_inner_w, f_btn_h)
    _push_local_box(ui, "mocap_export_footer", foot_row_r, FLEX_ROW, f_gap, Float32(0.0))
    var no_take = not state.has_take
    if ui.button_v2("mocap_btn_apply", "Apply to Entity", BTN_PRIMARY, ICON_CHECK, apply_w, f_btn_h, no_take):
        res.apply_to_scene_clicked = True
    if ui.button_v2("mocap_btn_bvh", "BVH", BTN_SECONDARY, -1, bvh_w, f_btn_h, no_take):
        res.export_bvh_clicked = True
    if ui.button_v2("mocap_btn_glb", "GLB", BTN_SECONDARY, -1, glb_w, f_btn_h, no_take):
        res.export_glb_clicked = True
    _pop_local_box(ui)

    ui.draw_list.pop_clip()
    ui.dock_style_v2 = prev_v2
    return res^


# ============================================================================
# 4. 3D Capture Viewport Overlay (`emit_mocap_viewport_overlay`)
# ============================================================================


def _project_pt_vp(view_proj: Mat4, world_pt: Vec3, vp: RectF) -> Tuple[Float32, Float32, Bool]:
    """Projects a 3D world point through `view_proj` into viewport pixel coordinates `(sx, sy, visible)`."""
    if not (isfinite(world_pt.x) and isfinite(world_pt.y) and isfinite(world_pt.z)):
        return (Float32(0.0), Float32(0.0), False)
    var cx = view_proj.at(0, 0) * world_pt.x + view_proj.at(0, 1) * world_pt.y + view_proj.at(0, 2) * world_pt.z + view_proj.at(0, 3)
    var cy = view_proj.at(1, 0) * world_pt.x + view_proj.at(1, 1) * world_pt.y + view_proj.at(1, 2) * world_pt.z + view_proj.at(1, 3)
    var cw = view_proj.at(3, 0) * world_pt.x + view_proj.at(3, 1) * world_pt.y + view_proj.at(3, 2) * world_pt.z + view_proj.at(3, 3)
    if not (isfinite(cx) and isfinite(cy) and isfinite(cw)) or cw <= Float32(1.0e-4):
        return (Float32(0.0), Float32(0.0), False)
    var inv_w = Float32(1.0) / cw
    var ndc_x = cx * inv_w
    var ndc_y = cy * inv_w
    var sx = vp.x + (ndc_x * Float32(0.5) + Float32(0.5)) * vp.w
    var sy = vp.y + (Float32(0.5) - ndc_y * Float32(0.5)) * vp.h
    if not (isfinite(sx) and isfinite(sy)):
        return (Float32(0.0), Float32(0.0), False)
    return (sx, sy, True)


def _draw_line_3d(
    mut dl: UiDrawList,
    view_proj: Mat4,
    vp: RectF,
    a: Vec3,
    b: Vec3,
    thickness: Float32,
    color: UInt32,
    halo_color: UInt32 = 0,
):
    var pa = _project_pt_vp(view_proj, a, vp)
    var pb = _project_pt_vp(view_proj, b, vp)
    if not (pa[2] and pb[2]):
        return
    if (halo_color >> 24) != 0:
        dl.add_line(pa[0], pa[1], pb[0], pb[1], thickness + Float32(2.0), halo_color)
    dl.add_line(pa[0], pa[1], pb[0], pb[1], thickness, color)


def _draw_ground_ring_3d(
    mut dl: UiDrawList,
    view_proj: Mat4,
    vp: RectF,
    center: Vec3,
    radius: Float32,
    thickness: Float32,
    color: UInt32,
    segments: Int = 20,
):
    var segs = max(8, segments)
    var two_pi = Float32(6.2831853)
    var prev = Vec3(center.x + radius, center.y, center.z)
    for i in range(1, segs + 1):
        var ang = two_pi * Float32(i) / Float32(segs)
        var cur = Vec3(center.x + cos(ang) * radius, center.y, center.z + sin(ang) * radius)
        _draw_line_3d(dl, view_proj, vp, prev, cur, thickness, color)
        prev = cur


def _is_primary_soma_joint(j: Int) -> Bool:
    return (
        j == SOMA77_HIPS
        or j == SOMA77_SPINE1
        or j == SOMA77_SPINE2
        or j == SOMA77_CHEST
        or j == SOMA77_NECK1
        or j == SOMA77_NECK2
        or j == SOMA77_HEAD
        or j == SOMA77_LEFT_SHOULDER
        or j == SOMA77_LEFT_ARM
        or j == SOMA77_LEFT_FOREARM
        or j == SOMA77_LEFT_HAND
        or j == SOMA77_RIGHT_SHOULDER
        or j == SOMA77_RIGHT_ARM
        or j == SOMA77_RIGHT_FOREARM
        or j == SOMA77_RIGHT_HAND
        or j == SOMA77_LEFT_LEG
        or j == SOMA77_LEFT_SHIN
        or j == SOMA77_LEFT_FOOT
        or j == SOMA77_LEFT_TOE_BASE
        or j == SOMA77_RIGHT_LEG
        or j == SOMA77_RIGHT_SHIN
        or j == SOMA77_RIGHT_FOOT
        or j == SOMA77_RIGHT_TOE_BASE
    )


def _draw_overlay_text(
    mut dl: UiDrawList,
    font: SdfFont,
    font_slot: Int,
    text: String,
    x: Float32,
    y_top: Float32,
    size_px: Float32,
    color: UInt32,
):
    if font.valid and text.byte_length() > 0:
        dl.add_text(font_slot, font, text, x, y_top + size_px * Float32(0.82), size_px, color)


def _measure_overlay_text(font: SdfFont, text: String, size_px: Float32) -> Float32:
    if font.valid:
        return font.measure(text, size_px)
    return Float32(text.byte_length()) * size_px * Float32(0.56)


def _draw_contact_lock_badge(
    mut dl: UiDrawList,
    font: SdfFont,
    font_slot: Int,
    view_proj: Mat4,
    vp: RectF,
    ground_pt: Vec3,
    label: String,
    accent: UInt32,
    theme: StudioTheme,
):
    var proj = _project_pt_vp(view_proj, ground_pt, vp)
    if not proj[2]:
        return
    var fsz = Float32(9.5)
    var tw = _measure_overlay_text(font, label, fsz)
    var pw = max(Float32(34.0), tw + Float32(12.0))
    var ph = Float32(15.0)
    var px = proj[0] - pw * Float32(0.5)
    var py = proj[1] + Float32(10.0)
    dl.add_glass_panel(px, py, pw, ph, theme.l4_overlay, ph * Float32(0.5), Float32(2.0), Float32(1.0), rgba8_with_alpha(accent, 185), True, False)
    _draw_overlay_text(dl, font, font_slot, label, px + Float32(6.0), py + Float32(2.0), fsz, accent)


def emit_mocap_viewport_overlay(
    mut dl: UiDrawList,
    font: SdfFont,
    font_slot: Int,
    state: MocapStudioState,
    vp: RectF,
    view_proj: Mat4,
    theme: StudioTheme,
):
    """Projects and draws the 3D RGBMocap capture overlay directly into `UiDrawList`:
      - Recovered 3D SOMA skeleton limbs & joints (color-coded by confidence + contact highlights)
      - 3D Camera Frustum wireframe + compact frosted `CAM 45°` badge
      - Ground-plane concentric emerald Foot Contact Lock Rings when L/R foot is locked
    """
    if not state.has_take or vp.w < Float32(16.0) or vp.h < Float32(16.0):
        return
    var nf = state.clean_take.num_frames()
    if nf <= 0:
        return
    var f_idx = max(0, min(nf - 1, state.current_frame))
    ref frame = state.clean_take.frames[f_idx]
    var nj = len(frame.joint_positions)
    var np = len(state.clean_take.parent_indices)
    var nc = len(frame.joint_confidences)

    dl.push_clip(Int(vp.x), Int(vp.y), Int(vp.right()), Int(vp.bottom()))

    var halo_dark = pack_rgba8(10, 13, 18, 175)
    var ground_y = state.cleanup_cfg.ground_y if state.cleanup_cfg.enable_ground_align else state.metrics.estimated_ground_y
    if not isfinite(ground_y):
        ground_y = Float32(0.0)

    var l_foot_idx = SOMA77_LEFT_FOOT if SOMA77_LEFT_FOOT < nj else state.clean_take.find_joint("LeftFoot")
    var r_foot_idx = SOMA77_RIGHT_FOOT if SOMA77_RIGHT_FOOT < nj else state.clean_take.find_joint("RightFoot")
    var l_hand_idx = SOMA77_LEFT_HAND if SOMA77_LEFT_HAND < nj else state.clean_take.find_joint("LeftHand")
    var r_hand_idx = SOMA77_RIGHT_HAND if SOMA77_RIGHT_HAND < nj else state.clean_take.find_joint("RightHand")

    # ------------------------------------------------------------------------
    # 1. Ground-Plane Foot (and Palm) Contact Lock Rings
    # ------------------------------------------------------------------------
    if state.show_contact_rings and nj > 0:
        if l_foot_idx >= 0 and l_foot_idx < nj:
            var lf = frame.joint_positions[l_foot_idx]
            var g_lf = Vec3(lf.x, ground_y + Float32(0.004), lf.z)
            if frame.left_foot_contact:
                _draw_ground_ring_3d(dl, view_proj, vp, g_lf, Float32(0.155), Float32(2.4), theme.status_ok, 20)
                _draw_ground_ring_3d(dl, view_proj, vp, g_lf, Float32(0.088), Float32(1.5), rgba8_with_alpha(theme.status_ok, 195), 16)
                _draw_line_3d(dl, view_proj, vp, Vec3(g_lf.x - Float32(0.19), g_lf.y, g_lf.z), Vec3(g_lf.x - Float32(0.13), g_lf.y, g_lf.z), Float32(1.6), theme.status_ok)
                _draw_line_3d(dl, view_proj, vp, Vec3(g_lf.x + Float32(0.13), g_lf.y, g_lf.z), Vec3(g_lf.x + Float32(0.19), g_lf.y, g_lf.z), Float32(1.6), theme.status_ok)
                _draw_contact_lock_badge(dl, font, font_slot, view_proj, vp, g_lf, "L LOCK", theme.status_ok, theme)
            else:
                _draw_ground_ring_3d(dl, view_proj, vp, g_lf, Float32(0.105), Float32(1.2), rgba8_scale_alpha(theme.text_muted, Float32(0.35)), 14)

        if r_foot_idx >= 0 and r_foot_idx < nj:
            var rf = frame.joint_positions[r_foot_idx]
            var g_rf = Vec3(rf.x, ground_y + Float32(0.004), rf.z)
            if frame.right_foot_contact:
                _draw_ground_ring_3d(dl, view_proj, vp, g_rf, Float32(0.155), Float32(2.4), theme.status_ok, 20)
                _draw_ground_ring_3d(dl, view_proj, vp, g_rf, Float32(0.088), Float32(1.5), rgba8_with_alpha(theme.status_ok, 195), 16)
                _draw_line_3d(dl, view_proj, vp, Vec3(g_rf.x - Float32(0.19), g_rf.y, g_rf.z), Vec3(g_rf.x - Float32(0.13), g_rf.y, g_rf.z), Float32(1.6), theme.status_ok)
                _draw_line_3d(dl, view_proj, vp, Vec3(g_rf.x + Float32(0.13), g_rf.y, g_rf.z), Vec3(g_rf.x + Float32(0.19), g_rf.y, g_rf.z), Float32(1.6), theme.status_ok)
                _draw_contact_lock_badge(dl, font, font_slot, view_proj, vp, g_rf, "R LOCK", theme.status_ok, theme)
            else:
                _draw_ground_ring_3d(dl, view_proj, vp, g_rf, Float32(0.105), Float32(1.2), rgba8_scale_alpha(theme.text_muted, Float32(0.35)), 14)

        if frame.left_palm_contact and l_hand_idx >= 0 and l_hand_idx < nj:
            var lh = frame.joint_positions[l_hand_idx]
            _draw_ground_ring_3d(dl, view_proj, vp, lh, Float32(0.09), Float32(1.8), theme.agent_flow, 14)
        if frame.right_palm_contact and r_hand_idx >= 0 and r_hand_idx < nj:
            var rh = frame.joint_positions[r_hand_idx]
            _draw_ground_ring_3d(dl, view_proj, vp, rh, Float32(0.09), Float32(1.8), theme.agent_flow, 14)

    # ------------------------------------------------------------------------
    # 2. Recovered 3D SOMA Skeleton Limbs & Confidence-Coded Joints
    # ------------------------------------------------------------------------
    if state.show_skeleton_overlay and nj > 0:
        var lim_p = min(nj, np)
        # First pass: bone limbs with subtle dark backing for crisp contrast over lit scenes
        for j in range(lim_p):
            var p = state.clean_take.parent_indices[j]
            if p < 0 or p >= nj:
                continue
            var primary = _is_primary_soma_joint(j) and _is_primary_soma_joint(p)
            var cj = frame.joint_confidences[j] if j < nc else frame.mean_confidence
            var cp = frame.joint_confidences[p] if p < nc else frame.mean_confidence
            var limb_col = _confidence_color(min(cj, cp), theme)
            if primary:
                _draw_line_3d(
                    dl, view_proj, vp,
                    frame.joint_positions[p], frame.joint_positions[j],
                    Float32(2.6), limb_col, halo_dark,
                )
            else:
                _draw_line_3d(
                    dl, view_proj, vp,
                    frame.joint_positions[p], frame.joint_positions[j],
                    Float32(1.15), rgba8_scale_alpha(limb_col, Float32(0.55)),
                )

        # Second pass: primary joint nodes (Hips Ember root, contact feet/palms glowing)
        for j in range(nj):
            if not _is_primary_soma_joint(j):
                continue
            var proj = _project_pt_vp(view_proj, frame.joint_positions[j], vp)
            if not proj[2]:
                continue
            var cj = frame.joint_confidences[j] if j < nc else frame.mean_confidence
            var jcol = _confidence_color(cj, theme)
            var rad = Float32(3.8)
            var is_contact_joint = (
                (j == l_foot_idx and frame.left_foot_contact)
                or (j == r_foot_idx and frame.right_foot_contact)
                or (j == l_hand_idx and frame.left_palm_contact)
                or (j == r_hand_idx and frame.right_palm_contact)
            )
            if j == SOMA77_HIPS:
                jcol = theme.accent_primary
                rad = Float32(5.2)
            elif j == SOMA77_HEAD:
                rad = Float32(5.0)
            elif is_contact_joint:
                jcol = theme.agent_flow if (j == l_hand_idx or j == r_hand_idx) else theme.status_ok
                rad = Float32(5.4)
                dl.add_circle(proj[0], proj[1], rad + Float32(3.0), rgba8_scale_alpha(jcol, Float32(0.32)))
            dl.add_circle(proj[0], proj[1], rad, jcol, Float32(1.3), halo_dark)
            if is_contact_joint or j == SOMA77_HIPS:
                dl.add_circle(proj[0], proj[1], Float32(1.8), theme.text_primary)

    # ------------------------------------------------------------------------
    # 3. 3D Camera Frustum Wireframe + Frosted `CAM 45°` Badge
    # ------------------------------------------------------------------------
    if state.show_frustum_overlay:
        var cam: MocapCameraPose = frame.camera
        var fwd = cam.forward.normalized()
        if fwd.length_squared() < Float32(1.0e-4):
            fwd = Vec3(0.0, 0.0, 1.0)
        var up_hint = cam.up.normalized()
        if up_hint.length_squared() < Float32(1.0e-4):
            up_hint = Vec3(0.0, 1.0, 0.0)
        var right = fwd.cross(up_hint).normalized()
        if right.length_squared() < Float32(1.0e-4):
            right = Vec3(1.0, 0.0, 0.0)
        var cam_up = right.cross(fwd).normalized()

        var fov_deg = max(Float32(15.0), min(Float32(120.0), cam.fov_y_deg if isfinite(cam.fov_y_deg) else Float32(45.0)))
        var aspect = max(Float32(0.5), min(Float32(3.0), cam.aspect if isfinite(cam.aspect) else Float32(16.0 / 9.0)))
        var tan_half = tan(fov_deg * Float32(0.01745329252) * Float32(0.5))

        var d_near = Float32(0.18)
        var d_far = Float32(0.95)
        var nh = d_near * tan_half
        var nw = nh * aspect
        var fh = d_far * tan_half
        var fw = fh * aspect

        var nc_pt = cam.pos + fwd * d_near
        var fc_pt = cam.pos + fwd * d_far

        var n_tl = nc_pt - right * nw + cam_up * nh
        var n_tr = nc_pt + right * nw + cam_up * nh
        var n_br = nc_pt + right * nw - cam_up * nh
        var n_bl = nc_pt - right * nw - cam_up * nh

        var f_tl = fc_pt - right * fw + cam_up * fh
        var f_tr = fc_pt + right * fw + cam_up * fh
        var f_br = fc_pt + right * fw - cam_up * fh
        var f_bl = fc_pt - right * fw - cam_up * fh

        var frust_col = theme.agent_flow
        var frust_dim = rgba8_scale_alpha(theme.agent_flow, Float32(0.72))

        # 4 rays from camera apex to far plane corners
        _draw_line_3d(dl, view_proj, vp, cam.pos, f_tl, Float32(1.6), frust_dim, halo_dark)
        _draw_line_3d(dl, view_proj, vp, cam.pos, f_tr, Float32(1.6), frust_dim, halo_dark)
        _draw_line_3d(dl, view_proj, vp, cam.pos, f_br, Float32(1.6), frust_dim, halo_dark)
        _draw_line_3d(dl, view_proj, vp, cam.pos, f_bl, Float32(1.6), frust_dim, halo_dark)

        # Near & far image-plane rectangles
        _draw_line_3d(dl, view_proj, vp, n_tl, n_tr, Float32(1.4), frust_dim)
        _draw_line_3d(dl, view_proj, vp, n_tr, n_br, Float32(1.4), frust_dim)
        _draw_line_3d(dl, view_proj, vp, n_br, n_bl, Float32(1.4), frust_dim)
        _draw_line_3d(dl, view_proj, vp, n_bl, n_tl, Float32(1.4), frust_dim)

        _draw_line_3d(dl, view_proj, vp, f_tl, f_tr, Float32(2.0), frust_col, halo_dark)
        _draw_line_3d(dl, view_proj, vp, f_tr, f_br, Float32(2.0), frust_col, halo_dark)
        _draw_line_3d(dl, view_proj, vp, f_br, f_bl, Float32(2.0), frust_col, halo_dark)
        _draw_line_3d(dl, view_proj, vp, f_bl, f_tl, Float32(2.0), frust_col, halo_dark)

        # Up-orientation chevron above far-plane top edge
        var top_l = fc_pt - right * (fw * Float32(0.28)) + cam_up * (fh + Float32(0.03))
        var top_r = fc_pt + right * (fw * Float32(0.28)) + cam_up * (fh + Float32(0.03))
        var top_tip = fc_pt + cam_up * (fh + Float32(0.16))
        _draw_line_3d(dl, view_proj, vp, top_l, top_tip, Float32(1.8), theme.accent_primary, halo_dark)
        _draw_line_3d(dl, view_proj, vp, top_tip, top_r, Float32(1.8), theme.accent_primary, halo_dark)

        # Camera apex dot + compact frosted `CAM 45°` badge (analytical vector degree ring — zero missing glyph!)
        var cam_proj = _project_pt_vp(view_proj, cam.pos, vp)
        if cam_proj[2]:
            dl.add_circle(cam_proj[0], cam_proj[1], Float32(4.2), frust_col, Float32(1.2), halo_dark)
            var fov_i = Int(fov_deg + Float32(0.5))
            var cam_lbl = String("CAM ") + String(fov_i)
            var fsz = Float32(10.5)
            var tw = _measure_overlay_text(font, cam_lbl, fsz)
            var deg_pad = Float32(6.0)
            var bw = max(Float32(64.0), tw + deg_pad + Float32(26.0))
            var bh = Float32(20.0)
            var bx = max(vp.x + Float32(6.0), min(vp.right() - bw - Float32(6.0), cam_proj[0] - bw * Float32(0.5)))
            var by = max(vp.y + Float32(6.0), min(vp.bottom() - bh - Float32(6.0), cam_proj[1] - bh - Float32(10.0)))
            dl.add_ambient_shadow(bx, by, bw, bh, bh * Float32(0.5), Float32(8.0), 110, Float32(2.0))
            dl.add_glass_panel(bx, by, bw, bh, theme.l4_overlay, bh * Float32(0.5), Float32(3.0), Float32(1.0), rgba8_with_alpha(frust_col, 175), True, False)
            draw_studio_icon(dl, ICON_CAMERA, bx + Float32(6.5), by + Float32(4.5), Float32(11.0), frust_col, Float32(1.4))
            var tx = bx + Float32(20.0)
            _draw_overlay_text(dl, font, font_slot, cam_lbl, tx, by + Float32(4.0), fsz, theme.text_primary)
            # Analytical SDF degree ring (`°`) right after the FOV digits
            dl.add_arc(tx + tw + Float32(3.0), by + Float32(6.2), Float32(1.75), Float32(1.0), Float32(0.0), Float32(6.2831853), theme.text_primary)

    dl.pop_clip()
