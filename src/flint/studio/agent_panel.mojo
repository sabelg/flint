"""Flint Studio Visual Agent Mission Control Panel (`agent_panel.mojo`).

Pure-Mojo 1.2 dockable UI panel for orchestrating & inspecting live in-Studio AI agents:
  1. 4 One-Click Role Preset Cards (`Scene Builder`, `Gameplay Coder`, `Lookdev Artist`, `QA Playtester`)
     with live status rings (`PRIM_ARC`), role icons, and step counts.
  2. Approval Policy Segmented Toggle (`Auto` / `Edits` / `Step`) with spring-animated `UiTabIndicator`.
  3. Interactive Prompt Bar + Run / Cancel button (`@entity`, `@selected`, `@problems` ready).
  4. Live Status Bar with `PRIM_ARC` spinner, collapsible Plan Checklist, `[Follow Camera]` toggle,
     and `[Theatre]` 3D overlay toggle.
  5. High-Contrast Approval Queue Banner with inline Myers unified diff snippet (`diff_view.mojo`)
     and clickable `[Approve]` / `[Allow Turn]` / `[Reject]` buttons.
  6. Scrollable Step-by-Step Timeline Cards with clickable Entity Chips (hover highlights entity in 3D,
     click focuses camera), File Diff Chips (jump to Code Drawer), collapsible diff/params drawer,
     `[Revert Step]`, and `[Revert Turn]`.
  7. Bottom Time-Travel Step Scrubber (`Step k / N` + clickable step ticks + `[Live]` return pill).

Emits typed `List[AgentPanelAction]` events so `studio_app.mojo` executes scene/file/ACP/MCP side effects
cleanly without circular imports.
"""
from std.collections import List
from std.math import isfinite, max, min
from flint.devtools.diff_view import (
    DIFF_ROW_NEW,
    DIFF_ROW_OLD,
    DiffSide,
    build_unified_diff,
    diff_sides,
)
from flint.platform.input import InputState
from flint.render.font import SdfFont
from flint.studio.agent_bus import (
    ACT_ASSET_PROCEDURAL,
    ACT_BUILD_HOTRELOAD,
    ACT_CODE_EDIT,
    ACT_ENTITY_DESPAWN,
    ACT_ENTITY_PATCH,
    ACT_ENTITY_REPARENT,
    ACT_ENTITY_SPAWN,
    ACT_ENTITY_TRANSFORM,
    ACT_FILE_READ,
    ACT_LIGHT_ENV,
    ACT_MATERIAL_STYLE,
    ACT_NAVMESH_BAKE,
    ACT_PIE_CONTROL,
    ACT_PLAN,
    ACT_TERMINAL,
    ACT_THOUGHT,
    ACT_VIEWPORT_CAPTURE,
    AGENT_STATUS_ERROR,
    AGENT_STATUS_IDLE,
    AGENT_STATUS_WAITING_APPROVAL,
    APPROVAL_AUTO,
    APPROVAL_EDITS,
    APPROVAL_STEP,
    ROLE_GAMEPLAY_CODER,
    ROLE_LOOKDEV_ARTIST,
    ROLE_QA_PLAYTESTER,
    ROLE_SCENE_BUILDER,
    STEP_AWAITING_APPROVAL,
    STEP_COMPLETED,
    STEP_FAILED,
    STEP_REJECTED,
    STEP_REVERTED,
    STEP_RUNNING,
    AgentStep,
    AgentTelemetryBus,
    agent_role_accent_rgba8,
    agent_status_label,
)
from flint.ui.animator import (
    SpringConfig,
    UI_ANIM_INSTANT,
    UI_ANIM_NORMAL,
    UiSpring,
    UiTabIndicator,
)
from flint.ui.draw_list import UiDrawList
from flint.ui.icons import draw_studio_chevron, draw_studio_icon
from flint.ui.theme import (
    ICON_CAMERA,
    ICON_CHECK,
    ICON_CLOSE,
    ICON_CODE,
    ICON_CUBE,
    ICON_DIFF,
    ICON_ERROR_CIRCLE,
    ICON_EYE,
    ICON_EYE_OFF,
    ICON_FILE,
    ICON_FOCUS_CAM,
    ICON_GRID,
    ICON_LIGHT_SPOT,
    ICON_MATERIAL,
    ICON_MOVE,
    ICON_PLAY,
    ICON_REVERT,
    ICON_SPARK,
    ICON_STOP,
    ICON_WARN,
    StudioTheme,
    rgba8_with_alpha,
)

# ============================================================================
# 1. Typed Panel Action Events Returned to Studio Controller
# ============================================================================

comptime PANEL_ACT_NONE: UInt8 = 0
comptime PANEL_ACT_SELECT_OR_SPAWN_ROLE: UInt8 = 1
comptime PANEL_ACT_SET_POLICY: UInt8 = 2
comptime PANEL_ACT_RUN_PROMPT: UInt8 = 3
comptime PANEL_ACT_CANCEL_TURN: UInt8 = 4
comptime PANEL_ACT_TOGGLE_FOLLOW_CAMERA: UInt8 = 5
comptime PANEL_ACT_TOGGLE_THEATRE: UInt8 = 6
comptime PANEL_ACT_APPROVE: UInt8 = 7
comptime PANEL_ACT_APPROVE_TURN: UInt8 = 8
comptime PANEL_ACT_REJECT: UInt8 = 9
comptime PANEL_ACT_FOCUS_ENTITY: UInt8 = 10
comptime PANEL_ACT_OPEN_FILE_DIFF: UInt8 = 11
comptime PANEL_ACT_REVERT_STEP: UInt8 = 12
comptime PANEL_ACT_REVERT_TURN: UInt8 = 13
comptime PANEL_ACT_TIME_TRAVEL_SCRUB: UInt8 = 14
comptime PANEL_ACT_TIME_TRAVEL_EXIT: UInt8 = 15

comptime _KEY_ENTER: Int = 257
comptime _KEY_BACKSPACE: Int = 259
comptime _KEY_ESCAPE: Int = 256


struct AgentPanelAction(Copyable, Movable):
    """Typed UI event emitted by `AgentMissionControlPanel.render_and_update`."""
    var kind: UInt8
    var agent_id: UInt32
    var role: UInt8
    var policy: UInt8
    var approval_id: UInt64
    var step_id: UInt64
    var turn_id: UInt32
    var entity_id: Int
    var step_index: Int
    var text: String

    def __init__(out self, kind: UInt8 = PANEL_ACT_NONE):
        self.kind = kind
        self.agent_id = 0
        self.role = ROLE_SCENE_BUILDER
        self.policy = APPROVAL_EDITS
        self.approval_id = 0
        self.step_id = 0
        self.turn_id = 0
        self.entity_id = -1
        self.step_index = -1
        self.text = String()


# ============================================================================
# 2. Layout, Typography & Icon Helpers
# ============================================================================

@always_inline
def _hit_rect(mx: Float32, my: Float32, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
    return mx >= x and mx <= x + w and my >= y and my <= y + h


@always_inline
def _role_icon(role: UInt8) -> Int:
    if role == ROLE_SCENE_BUILDER:
        return ICON_CUBE
    if role == ROLE_GAMEPLAY_CODER:
        return ICON_CODE
    if role == ROLE_LOOKDEV_ARTIST:
        return ICON_MATERIAL
    if role == ROLE_QA_PLAYTESTER:
        return ICON_PLAY
    return ICON_SPARK


@always_inline
def _action_icon(kind: UInt8) -> Int:
    if kind == ACT_ENTITY_SPAWN:
        return ICON_CUBE
    if kind == ACT_ENTITY_TRANSFORM or kind == ACT_ENTITY_REPARENT:
        return ICON_MOVE
    if kind == ACT_ENTITY_DESPAWN:
        return ICON_CLOSE
    if kind == ACT_MATERIAL_STYLE:
        return ICON_MATERIAL
    if kind == ACT_LIGHT_ENV:
        return ICON_LIGHT_SPOT
    if kind == ACT_NAVMESH_BAKE:
        return ICON_GRID
    if kind == ACT_PIE_CONTROL:
        return ICON_PLAY
    if kind == ACT_VIEWPORT_CAPTURE:
        return ICON_CAMERA
    if kind == ACT_CODE_EDIT:
        return ICON_DIFF
    if kind == ACT_FILE_READ:
        return ICON_FILE
    if kind == ACT_BUILD_HOTRELOAD or kind == ACT_TERMINAL:
        return ICON_CODE
    if kind == ACT_ASSET_PROCEDURAL or kind == ACT_PLAN:
        return ICON_SPARK
    return ICON_SPARK


def _truncate_text(text: String, max_chars: Int) -> String:
    var lim = max(4, max_chars)
    if text.byte_length() <= lim:
        return text
    var cut = max(1, lim - 3)
    var b = text.as_bytes()
    while cut > 0 and (b[cut] & 0xC0) == 0x80:
        cut -= 1
    return String(text[byte=0:cut]) + "..."


def _fmt_ms(ms: Float32) -> String:
    if not isfinite(ms) or ms <= Float32(0.0):
        return ""
    if ms < Float32(1000.0):
        return String(Int(ms + Float32(0.5))) + "ms"
    var tenths = Int(ms / Float32(100.0) + Float32(0.5))
    return String(tenths // 10) + "." + String(tenths % 10) + "s"


def _draw_text_safe(
    mut dl: UiDrawList,
    font_slot: Int,
    font: SdfFont,
    text: String,
    x: Float32,
    y: Float32,
    px: Float32,
    color: UInt32,
):
    if font.valid and text.byte_length() > 0:
        dl.add_text(font_slot, font, text, x, y + px * Float32(0.82), px, color)


def _draw_text_mid(
    mut dl: UiDrawList,
    font_slot: Int,
    font: SdfFont,
    text: String,
    x: Float32,
    rect_y: Float32,
    rect_h: Float32,
    px: Float32,
    color: UInt32,
):
    if font.valid and text.byte_length() > 0:
        var base_y = rect_y + (rect_h + px * Float32(0.66)) * Float32(0.5)
        dl.add_text(font_slot, font, text, x, base_y, px, color)


def _format_payload_preview(raw: String) -> String:
    """Turns compact JSON tool payloads into clean scannable key: value text."""
    var b = raw.as_bytes()
    var n = len(b)
    if n == 0:
        return String()
    var out = List[UInt8](capacity=n + 16)
    var in_bracket = 0
    for i in range(n):
        var c = b[i]
        if c == 123 or c == 125 or c == 34:  # '{', '}', '"'
            continue
        if c == 91:  # '['
            in_bracket += 1
            out.append(c)
        elif c == 93:  # ']'
            in_bracket = max(0, in_bracket - 1)
            out.append(c)
        elif c == 58:  # ':'
            out.append(58)
            if i + 1 < n and b[i + 1] != 32:
                out.append(32)
        elif c == 44 and in_bracket == 0:  # top-level ','
            out.append(32)
            out.append(32)
        else:
            out.append(c)
    return String(unsafe_from_utf8=out)


def _measure_safe(font: SdfFont, text: String, px: Float32) -> Float32:
    if font.valid:
        return font.measure(text, px)
    return Float32(text.byte_length()) * px * Float32(0.56)


def _trim_last_utf8_char(s: String) -> String:
    var n = s.byte_length()
    if n <= 0:
        return String()
    var b = s.as_bytes()
    var i = n - 1
    while i > 0 and (b[i] & 0xC0) == 0x80:
        i -= 1
    return String(s[byte=0:i])


def _append_printable_utf8(mut dst: String, chunk: String):
    var b = chunk.as_bytes()
    var out = List[UInt8](capacity=len(b))
    for i in range(len(b)):
        var c = b[i]
        if c >= 32 and c != 127:
            out.append(c)
    if len(out) > 0:
        dst += String(unsafe_from_utf8=out)


# ============================================================================
# 3. `AgentMissionControlPanel`: Retained Panel State & Renderer
# ============================================================================

struct AgentMissionControlPanel(Copyable, Movable):
    """Dockable Visual Agent Mission Control panel with micro-animated tabs, approvals & scrubber."""
    var prompt_text: String
    var prompt_focused: Bool
    var plan_expanded: Bool
    var scroll_target_y: Float32
    var scroll_spring: UiSpring
    var policy_indicator: UiTabIndicator
    var hovered_entity_id: Int
    var hovered_step_id: UInt64
    var anim_mode: Int
    var auto_follow_latest: Bool
    var _last_seen_step_count: Int
    var _scrubber_dragging: Bool
    var _caret_blink_sec: Float32

    def __init__(out self):
        self.prompt_text = String()
        self.prompt_focused = False
        self.plan_expanded = True
        self.scroll_target_y = Float32(0.0)
        self.scroll_spring = UiSpring.create(Float32(0.0), SpringConfig.scroll())
        self.policy_indicator = UiTabIndicator()
        self.hovered_entity_id = -1
        self.hovered_step_id = 0
        self.anim_mode = UI_ANIM_NORMAL
        self.auto_follow_latest = True
        self._last_seen_step_count = 0
        self._scrubber_dragging = False
        self._caret_blink_sec = Float32(0.0)

    def wants_keyboard(self) -> Bool:
        return self.prompt_focused

    def set_prompt_text(mut self, text: String):
        self.prompt_text = text

    def _find_role_session_idx(self, bus: AgentTelemetryBus, role: UInt8) -> Int:
        for i in range(len(bus.sessions)):
            if bus.sessions[i].role == role:
                return i
        return -1

    def _active_session_idx(self, bus: AgentTelemetryBus) -> Int:
        var si = bus.find_session_index(bus.selected_agent_id)
        if si >= 0:
            return si
        if len(bus.sessions) > 0:
            return 0
        return -1

    def render_and_update(
        mut self,
        mut dl: UiDrawList,
        mut bus: AgentTelemetryBus,
        theme: StudioTheme,
        font_slot: Int,
        font: SdfFont,
        panel_x: Float32,
        panel_y: Float32,
        panel_w: Float32,
        panel_h: Float32,
        input: InputState,
        dt: Float32,
        time_sec: Float32,
        bold_font_slot: Int = -1,
        bold_font: SdfFont = SdfFont(),
        mono_font_slot: Int = -1,
        mono_font: SdfFont = SdfFont(),
    ) -> List[AgentPanelAction]:
        """Renders the complete Agent Mission Control UI inside `(panel_x, panel_y, panel_w, panel_h)`
        and returns any triggered `AgentPanelAction`s for `studio_app.mojo` to execute."""
        var actions = List[AgentPanelAction]()
        if panel_w < Float32(120.0) or panel_h < Float32(140.0):
            return actions^

        var use_bold = bold_font_slot >= 0 and bold_font.valid
        var b_slot = bold_font_slot if use_bold else font_slot
        var use_mono = mono_font_slot >= 0 and mono_font.valid
        var m_slot = mono_font_slot if use_mono else font_slot

        var safe_dt = Float32(0.0) if (not isfinite(dt) or dt < Float32(0.0)) else min(dt, Float32(0.1))
        var t = time_sec if isfinite(time_sec) else Float32(0.0)
        self._caret_blink_sec += safe_dt
        self.hovered_entity_id = -1
        self.hovered_step_id = 0

        var mx = input.last_mouse_x
        var my = input.last_mouse_y
        var click = input.mouse_pressed(0)
        var down = input.mouse_down(0)
        if not down:
            self._scrubber_dragging = False

        var pad = Float32(8.0)
        var inner_x = panel_x + pad
        var inner_w = max(Float32(96.0), panel_w - pad * Float32(2.0))
        var y = panel_y + pad

        dl.push_clip(Int(panel_x), Int(panel_y), Int(panel_x + panel_w), Int(panel_y + panel_h))

        # Panel L1 obsidian shell backdrop
        dl.add_rect(panel_x, panel_y, panel_w, panel_h, theme.l1_shell, Float32(0.0), Float32(1.0), theme.border_subtle)

        var active_si = self._active_session_idx(bus)
        var active_agent_id: UInt32 = bus.sessions[active_si].agent_id if active_si >= 0 else UInt32(0)
        var active_role: UInt8 = bus.sessions[active_si].role if active_si >= 0 else ROLE_SCENE_BUILDER
        var active_policy: UInt8 = bus.sessions[active_si].approval_policy if active_si >= 0 else APPROVAL_EDITS
        var active_busy: Bool = bus.sessions[active_si].is_busy() if active_si >= 0 else False
        var active_accent: UInt32 = bus.sessions[active_si].accent_rgba8 if active_si >= 0 else theme.agent_active
        var pending_cnt = bus.pending_approval_count()

        # --------------------------------------------------------------------
        # SECTION 1: 4 One-Click Role Preset Identity Cards (2x2 matte grid)
        # --------------------------------------------------------------------
        var roles = [ROLE_SCENE_BUILDER, ROLE_GAMEPLAY_CODER, ROLE_LOOKDEV_ARTIST, ROLE_QA_PLAYTESTER]
        var role_labels = [String("Scene Builder"), String("Gameplay Coder"), String("LookDev Artist"), String("QA Playtester")]
        var card_gap = Float32(5.0)
        var card_w = (inner_w - card_gap) * Float32(0.5)
        var card_h = Float32(35.0)

        for ri in range(4):
            var col = ri & 1
            var row = ri >> 1
            var cx = inner_x + Float32(col) * (card_w + card_gap)
            var cy = y + Float32(row) * (card_h + card_gap)
            var role = roles[ri]
            var r_accent = agent_role_accent_rgba8(role)
            var r_si = self._find_role_session_idx(bus, role)
            var is_selected = (r_si >= 0 and bus.sessions[r_si].agent_id == active_agent_id) or (active_si < 0 and ri == 0)
            var is_busy = r_si >= 0 and bus.sessions[r_si].is_busy()
            var r_status = bus.sessions[r_si].status if r_si >= 0 else AGENT_STATUS_IDLE
            var step_cnt = len(bus.sessions[r_si].steps) if r_si >= 0 else 0
            var hov = _hit_rect(mx, my, cx, cy, card_w, card_h)

            if is_selected:
                dl.add_card_surface(
                    cx, cy, card_w, card_h,
                    theme.l3_raised,
                    rgba8_with_alpha(r_accent, 155),
                    theme.radius_sm,
                    Float32(1.1),
                    top_bevel=True,
                    drop_shadow=False,
                )
                dl.add_accent_bar(cx + Float32(3.0), cy + Float32(5.0), Float32(2.5), card_h - Float32(10.0), r_accent, pill=True)
            else:
                var bg = theme.l3_hover if hov else theme.l2_card
                var bdr = theme.border_strong if hov else theme.border_subtle
                dl.add_card_surface(cx, cy, card_w, card_h, bg, bdr, theme.radius_sm, Float32(1.0), top_bevel=True, drop_shadow=False)

            # Clean 20x20 tinted icon badge cleanly separated from PRIM_ARC spinner
            var ic_cx = cx + Float32(19.0)
            var ic_cy = cy + card_h * Float32(0.5)
            dl.add_circle(ic_cx, ic_cy, Float32(9.5), rgba8_with_alpha(r_accent, 34 if is_selected else 20))
            if is_busy:
                var ring_col = theme.status_warn if r_status == AGENT_STATUS_WAITING_APPROVAL else r_accent
                dl.add_arc(ic_cx, ic_cy, Float32(10.3), Float32(1.5), t * Float32(4.2) + Float32(ri), Float32(3.8), ring_col, rgba8_with_alpha(ring_col, 40), Float32(1.0), 0)
            draw_studio_icon(dl, _role_icon(role), ic_cx - Float32(5.25), ic_cy - Float32(5.25), Float32(10.5), r_accent, Float32(1.35))

            # Optional right-aligned step count badge pill
            var badge_reserve = Float32(6.0)
            if step_cnt > 0:
                var cnt_s = String(step_cnt)
                var cnt_w = _measure_safe(mono_font if use_mono else font, cnt_s, Float32(9.5))
                var pill_w = max(Float32(16.0), cnt_w + Float32(8.0))
                var pill_h = Float32(14.5)
                var pill_x = cx + card_w - pill_w - Float32(6.0)
                var pill_y = cy + Float32(4.5)
                dl.add_rect(
                    pill_x, pill_y, pill_w, pill_h,
                    rgba8_with_alpha(r_accent, 36 if is_selected else 20),
                    Float32(7.25), Float32(1.0),
                    rgba8_with_alpha(r_accent, 110 if is_selected else 52),
                )
                _draw_text_mid(
                    dl, m_slot, mono_font if use_mono else font, cnt_s,
                    pill_x + max(Float32(3.0), (pill_w - cnt_w) * Float32(0.5)),
                    pill_y, pill_h, Float32(9.5),
                    r_accent if is_selected else theme.text_secondary,
                )
                badge_reserve = pill_w + Float32(10.0)

            var max_lbl_chars = max(8, Int((card_w - Float32(37.0) - badge_reserve) / Float32(6.4)))
            _draw_text_safe(
                dl, b_slot if is_selected else font_slot,
                bold_font if (use_bold and is_selected) else font,
                _truncate_text(role_labels[ri], max_lbl_chars),
                cx + Float32(34.0), cy + Float32(4.5), Float32(11.5),
                theme.text_primary if is_selected else theme.text_secondary,
            )

            # Status dot + label (no raw ASCII pipe)
            var dot_col = theme.text_muted
            if r_status == AGENT_STATUS_WAITING_APPROVAL:
                dot_col = theme.status_warn
            elif r_status == AGENT_STATUS_ERROR:
                dot_col = theme.status_danger
            elif is_busy:
                dot_col = r_accent
            elif r_si >= 0 and step_cnt > 0:
                dot_col = theme.status_ok
            dl.add_circle(cx + Float32(37.0), cy + Float32(24.5), Float32(2.3), dot_col)
            var sub_txt = agent_status_label(r_status) if r_si >= 0 else String("Ready")
            var max_sub_chars = max(8, Int((card_w - Float32(46.0)) / Float32(5.8)))
            _draw_text_safe(
                dl, font_slot, font,
                _truncate_text(sub_txt, max_sub_chars),
                cx + Float32(43.0), cy + Float32(19.0), Float32(10.0),
                dot_col if (is_busy or r_status == AGENT_STATUS_ERROR) else theme.text_muted,
            )

            if click and hov:
                var act = AgentPanelAction(PANEL_ACT_SELECT_OR_SPAWN_ROLE)
                act.role = role
                act.agent_id = bus.sessions[r_si].agent_id if r_si >= 0 else UInt32(0)
                actions.append(act^)

        y += card_h * Float32(2.0) + card_gap + Float32(6.0)

        # --------------------------------------------------------------------
        # SECTION 2: Labeled Approval Policy Segmented Toggle (`Approval: Auto|Edits|Step`)
        # --------------------------------------------------------------------
        var seg_h = Float32(24.0)
        var pol_lbl_w = Float32(58.0)
        var seg_track_x = inner_x + pol_lbl_w
        var seg_track_w = max(Float32(120.0), inner_w - pol_lbl_w)
        var seg_w = seg_track_w / Float32(3.0)

        _draw_text_mid(dl, font_slot, font, "Approval", inner_x + Float32(2.0), y, seg_h, Float32(11.0), theme.text_muted)
        dl.add_rect(seg_track_x, y, seg_track_w, seg_h, theme.l0_canvas, theme.radius_sm, Float32(1.0), theme.border_subtle)

        var pol_idx = 0 if active_policy == APPROVAL_AUTO else (1 if active_policy == APPROVAL_EDITS else 2)
        var target_pill_x = seg_track_x + Float32(pol_idx) * seg_w + Float32(2.0)
        var target_pill_w = max(Float32(16.0), seg_w - Float32(4.0))
        self.policy_indicator.x.set_target(target_pill_x, snap_if_uninit=True)
        self.policy_indicator.w.set_target(target_pill_w, snap_if_uninit=True)
        _ = self.policy_indicator.step(safe_dt, self.anim_mode)

        dl.add_card_surface(
            self.policy_indicator.x.value,
            y + Float32(2.0),
            self.policy_indicator.w.value,
            seg_h - Float32(4.0),
            theme.l3_raised,
            rgba8_with_alpha(active_accent, 145),
            theme.radius_xs,
            Float32(1.0),
            top_bevel=True,
            drop_shadow=False,
        )

        var pol_names = [String("Auto"), String("Edits"), String("Step")]
        var pol_vals = [APPROVAL_AUTO, APPROVAL_EDITS, APPROVAL_STEP]
        for pi in range(3):
            var bx = seg_track_x + Float32(pi) * seg_w
            if pi > 0 and pi != pol_idx and (pi - 1) != pol_idx:
                dl.add_rect(bx, y + Float32(5.0), Float32(1.0), seg_h - Float32(10.0), theme.border_subtle)
            var hov = _hit_rect(mx, my, bx, y, seg_w, seg_h)
            var is_cur_p = pi == pol_idx
            var tw = _measure_safe(bold_font if (use_bold and is_cur_p) else font, pol_names[pi], Float32(11.0))
            var tx = bx + max(Float32(4.0), (seg_w - tw) * Float32(0.5))
            _draw_text_mid(
                dl, b_slot if is_cur_p else font_slot,
                bold_font if (use_bold and is_cur_p) else font,
                pol_names[pi], tx, y, seg_h, Float32(11.0),
                theme.text_primary if is_cur_p else (theme.text_primary if hov else theme.text_secondary),
            )
            if click and hov and pol_vals[pi] != active_policy:
                var act = AgentPanelAction(PANEL_ACT_SET_POLICY)
                act.agent_id = active_agent_id
                act.policy = pol_vals[pi]
                actions.append(act^)

        y += seg_h + Float32(6.0)

        # --------------------------------------------------------------------
        # SECTION 3: Prompt Bar + High-Contrast Run / Stop Button
        # --------------------------------------------------------------------
        var run_w = Float32(64.0)
        var prompt_h = Float32(28.0)
        var prompt_w = max(Float32(60.0), inner_w - run_w - Float32(6.0))
        var prompt_hov = _hit_rect(mx, my, inner_x, y, prompt_w, prompt_h)
        var run_x = inner_x + prompt_w + Float32(6.0)
        var run_hov = _hit_rect(mx, my, run_x, y, run_w, prompt_h)

        if click:
            if prompt_hov:
                self.prompt_focused = True
            elif not run_hov:
                self.prompt_focused = False

        if self.prompt_focused:
            if input.text_input_utf8.byte_length() > 0:
                _append_printable_utf8(self.prompt_text, input.text_input_utf8)
            if input.key_pressed(_KEY_BACKSPACE):
                self.prompt_text = _trim_last_utf8_char(self.prompt_text)
            if input.key_pressed(_KEY_ESCAPE):
                if active_busy:
                    var c_act = AgentPanelAction(PANEL_ACT_CANCEL_TURN)
                    c_act.agent_id = active_agent_id
                    actions.append(c_act^)
                self.prompt_focused = False
            elif input.key_pressed(_KEY_ENTER) and self.prompt_text.byte_length() > 0 and not active_busy:
                var r_act = AgentPanelAction(PANEL_ACT_RUN_PROMPT)
                r_act.agent_id = active_agent_id
                r_act.role = active_role
                r_act.text = self.prompt_text
                actions.append(r_act^)
                self.prompt_text = String()

        var p_bdr = theme.focus_ring if self.prompt_focused else (theme.border_strong if prompt_hov else theme.border_control)
        dl.add_recessed_well(
            inner_x, y, prompt_w, prompt_h,
            theme.l0_canvas, p_bdr,
            theme.radius_sm,
            Float32(1.2) if self.prompt_focused else Float32(1.0),
            self.prompt_focused, theme.focus_ring,
        )
        draw_studio_icon(
            dl, ICON_SPARK, inner_x + Float32(7.0), y + Float32(8.0), Float32(12.0),
            active_accent if self.prompt_focused else theme.text_muted, Float32(1.3),
        )
        var disp_prompt = self.prompt_text
        var p_text_x = inner_x + Float32(23.0)
        var max_p_chars = max(10, Int((prompt_w - Float32(32.0)) / Float32(6.5)))
        if disp_prompt.byte_length() == 0:
            var ph_str = "Queue follow-up or Esc to stop..." if active_busy else "Assign task (@selected, @problems)..."
            _draw_text_mid(dl, font_slot, font, _truncate_text(ph_str, max_p_chars), p_text_x, y, prompt_h, Float32(11.5), theme.text_muted)
        else:
            if disp_prompt.byte_length() > max_p_chars:
                var start_b = disp_prompt.byte_length() - max_p_chars
                var bs = disp_prompt.as_bytes()
                while start_b < len(bs) and (bs[start_b] & 0xC0) == 0x80:
                    start_b += 1
                var sliced = String(disp_prompt[byte=start_b:])
                disp_prompt = sliced^
            _draw_text_mid(dl, font_slot, font, disp_prompt, p_text_x, y, prompt_h, Float32(11.5), theme.text_primary)
            if self.prompt_focused and (Int(self._caret_blink_sec * Float32(2.0)) & 1) == 0:
                var cw = _measure_safe(font, disp_prompt, Float32(11.5))
                dl.add_rect(p_text_x + Float32(1.0) + min(prompt_w - Float32(30.0), cw), y + Float32(6.0), Float32(1.5), Float32(16.0), active_accent)

        # High-contrast Run / Stop button (WCAG AA/AAA verified)
        var btn_fill: UInt32
        var btn_bdr: UInt32
        var btn_fg: UInt32
        if active_busy:
            btn_fill = rgba8_with_alpha(theme.status_danger, 248 if run_hov else 224)
            btn_bdr = rgba8_with_alpha(theme.status_danger, 255)
            btn_fg = theme.text_primary
        else:
            btn_fill = rgba8_with_alpha(active_accent, 252 if run_hov else 232)
            btn_bdr = active_accent
            btn_fg = theme.text_on_accent
        dl.add_card_surface(run_x, y, run_w, prompt_h, btn_fill, btn_bdr, theme.radius_sm, Float32(1.0), top_bevel=True, drop_shadow=False)
        var btn_lbl = String("Stop") if active_busy else String("Run")
        var btn_tw = _measure_safe(bold_font if use_bold else font, btn_lbl, Float32(11.5))
        var btn_content_w = Float32(15.0) + btn_tw
        var btn_start_x = run_x + max(Float32(6.0), (run_w - btn_content_w) * Float32(0.5))
        draw_studio_icon(dl, ICON_STOP if active_busy else ICON_PLAY, btn_start_x, y + Float32(8.5), Float32(11.0), btn_fg, Float32(1.6))
        _draw_text_mid(dl, b_slot, bold_font if use_bold else font, btn_lbl, btn_start_x + Float32(15.0), y, prompt_h, Float32(11.5), btn_fg)
        if click and run_hov:
            if active_busy:
                var c_act = AgentPanelAction(PANEL_ACT_CANCEL_TURN)
                c_act.agent_id = active_agent_id
                actions.append(c_act^)
            elif self.prompt_text.byte_length() > 0:
                var r_act = AgentPanelAction(PANEL_ACT_RUN_PROMPT)
                r_act.agent_id = active_agent_id
                r_act.role = active_role
                r_act.text = self.prompt_text
                actions.append(r_act^)
                self.prompt_text = String()

        y += prompt_h + Float32(6.0)

        # --------------------------------------------------------------------
        # SECTION 4: Live Status Bar (`PRIM_ARC` spinner + unified [Follow] & [Theatre])
        # --------------------------------------------------------------------
        var status_h = Float32(26.0)
        dl.add_rect(inner_x, y, inner_w, status_h, theme.l2_card, theme.radius_sm, Float32(1.0), theme.border_subtle)
        var st_code = bus.sessions[active_si].status if active_si >= 0 else AGENT_STATUS_IDLE
        var sp_cx = inner_x + Float32(14.0)
        var sp_cy = y + status_h * Float32(0.5)
        if active_busy:
            var sp_col = theme.status_warn if st_code == AGENT_STATUS_WAITING_APPROVAL else active_accent
            dl.add_arc(sp_cx, sp_cy, Float32(6.5), Float32(1.7), t * Float32(4.6), Float32(4.0), sp_col, rgba8_with_alpha(sp_col, 48), Float32(1.5), 0)
        else:
            var idle_col = theme.status_danger if st_code == AGENT_STATUS_ERROR else theme.status_ok
            dl.add_progress_ring(sp_cx, sp_cy, Float32(6.0), Float32(1.6), Float32(1.0), idle_col, rgba8_with_alpha(idle_col, 40), Float32(0.0))

        # Suppress duplicate approval title here when Approval Banner is already rendered right below
        var st_lbl = agent_status_label(st_code)
        var st_col = theme.text_primary
        if pending_cnt > 0 and st_code == AGENT_STATUS_WAITING_APPROVAL:
            st_lbl = "Awaiting approval (" + String(pending_cnt) + ")"
            st_col = theme.status_warn
        elif active_si >= 0 and bus.sessions[active_si].active_callout.byte_length() > 0 and active_busy:
            st_lbl = bus.sessions[active_si].active_callout
        var st_max_chars = max(8, Int((inner_w - Float32(138.0)) / Float32(6.4)))
        _draw_text_mid(dl, font_slot, font, _truncate_text(st_lbl, st_max_chars), inner_x + Float32(27.0), y, status_h, Float32(11.5), st_col)

        # Right-aligned [Follow] & [Theatre] toggles (unified 22px height & active border treatment)
        var th_w = Float32(26.0)
        var th_x = inner_x + inner_w - th_w - Float32(3.0)
        var fl_w = Float32(72.0)
        var fl_x = th_x - fl_w - Float32(4.0)
        var t_y = y + Float32(2.0)
        var t_h = status_h - Float32(4.0)

        var fl_on = bus.follow_agent_camera and not bus.follow_paused_by_user
        var fl_hov = _hit_rect(mx, my, fl_x, t_y, fl_w, t_h)
        var fl_bg = rgba8_with_alpha(active_accent, 44) if fl_on else (theme.l3_hover if fl_hov else theme.l3_raised)
        var fl_bdr = rgba8_with_alpha(active_accent, 175) if fl_on else (theme.border_strong if fl_hov else theme.border_control)
        dl.add_card_surface(fl_x, t_y, fl_w, t_h, fl_bg, fl_bdr, theme.radius_xs, Float32(1.0), top_bevel=True, drop_shadow=False)
        draw_studio_icon(dl, ICON_FOCUS_CAM, fl_x + Float32(6.0), t_y + Float32(5.0), Float32(12.0), active_accent if fl_on else theme.text_secondary, Float32(1.4))
        var fl_txt = "Follow" if (not bus.follow_paused_by_user) else "Resume"
        _draw_text_mid(dl, font_slot, font, fl_txt, fl_x + Float32(21.0), t_y, t_h, Float32(10.5), theme.text_primary if fl_on else theme.text_secondary)
        if click and fl_hov:
            var act = AgentPanelAction(PANEL_ACT_TOGGLE_FOLLOW_CAMERA)
            act.agent_id = active_agent_id
            actions.append(act^)

        var th_on = bus.theatre_overlays_enabled
        var th_hov = _hit_rect(mx, my, th_x, t_y, th_w, t_h)
        var th_bg = rgba8_with_alpha(active_accent, 44) if th_on else (theme.l3_hover if th_hov else theme.l3_raised)
        var th_bdr = rgba8_with_alpha(active_accent, 175) if th_on else (theme.border_strong if th_hov else theme.border_control)
        dl.add_card_surface(th_x, t_y, th_w, t_h, th_bg, th_bdr, theme.radius_xs, Float32(1.0), top_bevel=True, drop_shadow=False)
        draw_studio_icon(
            dl,
            ICON_EYE if th_on else ICON_EYE_OFF,
            th_x + Float32(7.0), t_y + Float32(5.0), Float32(12.0),
            active_accent if th_on else theme.text_muted,
            Float32(1.4),
        )
        if click and th_hov:
            var act = AgentPanelAction(PANEL_ACT_TOGGLE_THEATRE)
            actions.append(act^)

        y += status_h + Float32(6.0)

        # Collapsible Plan Checklist (wrapped in unified L2 card with progress bar)
        if active_si >= 0 and len(bus.sessions[active_si].plan_entries) > 0:
            var p_cnt = len(bus.sessions[active_si].plan_entries)
            var done_cnt = 0
            for pli in range(p_cnt):
                if bus.sessions[active_si].plan_entries[pli].startswith("[completed]"):
                    done_cnt += 1
            var shown = min(4, p_cnt)
            var hdr_h = Float32(21.0)
            var plan_row_h = Float32(15.5)
            var body_plan_h = (Float32(shown) * plan_row_h + Float32(6.0)) if self.plan_expanded else Float32(0.0)
            var plan_card_h = hdr_h + body_plan_h
            var hdr_hov = _hit_rect(mx, my, inner_x, y, inner_w, hdr_h)

            dl.add_card_surface(inner_x, y, inner_w, plan_card_h, theme.l2_card, theme.border_subtle, theme.radius_sm, Float32(1.0), top_bevel=True, drop_shadow=False)
            if hdr_hov:
                dl.add_rect(inner_x + Float32(1.0), y + Float32(1.0), inner_w - Float32(2.0), hdr_h - Float32(1.0), theme.l3_hover, theme.radius_xs)
            draw_studio_chevron(dl, inner_x + Float32(7.0), y + Float32(4.5), Float32(12.0), theme.text_secondary, Float32(1.0) if self.plan_expanded else Float32(0.0), Float32(1.4))
            _draw_text_mid(dl, b_slot, bold_font if use_bold else font, "Plan", inner_x + Float32(22.0), y, hdr_h, Float32(11.0), theme.text_primary)

            # Right-aligned mini progress bar + done/total fraction
            var prog_txt = String(done_cnt) + "/" + String(p_cnt)
            var prog_tw = _measure_safe(mono_font if use_mono else font, prog_txt, Float32(10.0))
            var pbar_w = Float32(42.0)
            var pbar_x = inner_x + inner_w - pbar_w - Float32(8.0)
            var pbar_y = y + (hdr_h - Float32(4.0)) * Float32(0.5)
            _draw_text_mid(dl, m_slot, mono_font if use_mono else font, prog_txt, pbar_x - prog_tw - Float32(6.0), y, hdr_h, Float32(10.0), theme.text_muted)
            dl.add_rect(pbar_x, pbar_y, pbar_w, Float32(4.0), theme.l0_canvas, Float32(2.0))
            if done_cnt > 0:
                var fill_w = max(Float32(4.0), pbar_w * (Float32(done_cnt) / Float32(max(1, p_cnt))))
                dl.add_rect(pbar_x, pbar_y, fill_w, Float32(4.0), theme.status_ok, Float32(2.0))

            if click and hdr_hov:
                self.plan_expanded = not self.plan_expanded

            if self.plan_expanded:
                dl.add_rect(inner_x + Float32(1.0), y + hdr_h, inner_w - Float32(2.0), Float32(1.0), theme.border_subtle)
                var py_i = y + hdr_h + Float32(3.0)
                for pli in range(shown):
                    var entry = bus.sessions[active_si].plan_entries[pli]
                    var is_done = entry.startswith("[completed]")
                    var is_prog = entry.startswith("[in_progress]")
                    if is_done:
                        draw_studio_icon(dl, ICON_CHECK, inner_x + Float32(9.0), py_i + Float32(2.0), Float32(11.0), theme.status_ok, Float32(1.5))
                    elif is_prog:
                        dl.add_arc(inner_x + Float32(14.5), py_i + Float32(7.5), Float32(4.0), Float32(1.5), t * Float32(4.5), Float32(4.0), active_accent, rgba8_with_alpha(active_accent, 45), Float32(0.8), 0)
                    else:
                        dl.add_circle(inner_x + Float32(14.5), py_i + Float32(7.5), Float32(2.1), theme.text_muted)
                    var clean_entry = entry
                    var close_br = entry.find("] ")
                    if close_br >= 0 and close_br + 2 < entry.byte_length():
                        clean_entry = String(entry[byte=close_br + 2:])
                    var max_pc = max(12, Int((inner_w - Float32(34.0)) / Float32(6.1)))
                    _draw_text_mid(
                        dl, font_slot, font, _truncate_text(clean_entry, max_pc),
                        inner_x + Float32(25.0), py_i, plan_row_h, Float32(11.0),
                        theme.text_muted if is_done else (theme.text_primary if is_prog else theme.text_secondary),
                    )
                    py_i += plan_row_h
            y += plan_card_h + Float32(6.0)

        # --------------------------------------------------------------------
        # SECTION 5: Matte-Precision Approval Queue Card (with Inline Diff / Formatted Preview)
        # --------------------------------------------------------------------
        if pending_cnt > 0:
            var first_pa = -1
            for pi in range(len(bus.pending_approvals)):
                if not bus.pending_approvals[pi].resolved:
                    first_pa = pi
                    break
            if first_pa >= 0:
                ref pa = bus.pending_approvals[first_pa]
                var diff_lines = List[Tuple[Int, String]]()
                if pa.old_text.byte_length() > 0 or pa.new_text.byte_length() > 0:
                    var old_s = DiffSide.from_text(pa.old_text)
                    var new_s = DiffSide.from_text(pa.new_text)
                    var hunks = diff_sides(old_s, new_s)
                    var drows = build_unified_diff(old_s, new_s, hunks, context=1)
                    for di in range(min(4, len(drows))):
                        diff_lines.append((drows[di].kind, drows[di].text))
                elif pa.raw_payload_json.byte_length() > 0:
                    diff_lines.append((0, _format_payload_preview(pa.raw_payload_json)))

                var box_h = (Float32(len(diff_lines)) * Float32(15.5) + Float32(6.0)) if len(diff_lines) > 0 else Float32(16.0)
                var ab_h = Float32(24.0)
                var banner_h = Float32(26.0) + box_h + Float32(6.0) + ab_h + Float32(6.0)

                # Matte L2 card + subtle warm amber tint wash + 3px left amber accent bar (zero muddy outer glow)
                dl.add_card_surface(
                    inner_x, y, inner_w, banner_h,
                    theme.l2_card,
                    rgba8_with_alpha(theme.status_warn, 155),
                    theme.radius_md,
                    Float32(1.1),
                    top_bevel=True,
                    drop_shadow=False,
                )
                dl.add_rect(inner_x + Float32(1.0), y + Float32(1.0), inner_w - Float32(2.0), banner_h - Float32(2.0), rgba8_with_alpha(theme.status_warn, 16), theme.radius_md - Float32(1.0))
                dl.add_accent_bar(inner_x + Float32(3.0), y + Float32(6.0), Float32(3.0), banner_h - Float32(12.0), theme.status_warn, pill=True)

                dl.add_circle(inner_x + Float32(19.0), y + Float32(14.0), Float32(7.5), rgba8_with_alpha(theme.status_warn, 34))
                draw_studio_icon(dl, ICON_WARN, inner_x + Float32(13.5), y + Float32(8.5), Float32(11.0), theme.status_warn, Float32(1.4))
                var a_prefix = "Approval"
                if pending_cnt > 1:
                    a_prefix += " (" + String(pending_cnt) + ")"
                var a_pw = _measure_safe(bold_font if use_bold else font, a_prefix, Float32(11.5))
                _draw_text_mid(dl, b_slot, bold_font if use_bold else font, a_prefix, inner_x + Float32(31.0), y + Float32(2.0), Float32(24.0), Float32(11.5), theme.status_warn)
                dl.add_circle(inner_x + Float32(36.0) + a_pw, y + Float32(14.0), Float32(1.5), theme.text_muted)
                var title_x = inner_x + Float32(42.0) + a_pw
                var max_ac = max(10, Int((inner_x + inner_w - title_x - Float32(8.0)) / Float32(6.4)))
                _draw_text_mid(dl, b_slot, bold_font if use_bold else font, _truncate_text(pa.title, max_ac), title_x, y + Float32(2.0), Float32(24.0), Float32(11.5), theme.text_primary)

                var dy = y + Float32(26.0)
                if len(diff_lines) > 0:
                    dl.add_rect(inner_x + Float32(10.0), dy, inner_w - Float32(18.0), box_h, theme.l0_canvas, theme.radius_xs, Float32(1.0), theme.border_subtle)
                    var d_max_c = max(14, Int((inner_w - Float32(32.0)) / Float32(6.1)))
                    for di in range(len(diff_lines)):
                        var rk = diff_lines[di][0]
                        var rt = diff_lines[di][1]
                        var rcol = theme.text_secondary
                        var row_y = dy + Float32(3.0) + Float32(di) * Float32(15.5)
                        if rk == DIFF_ROW_NEW:
                            dl.add_rect(inner_x + Float32(11.0), row_y, inner_w - Float32(20.0), Float32(15.0), theme.diff_add_bg)
                            dl.add_rect(inner_x + Float32(11.0), row_y, Float32(2.0), Float32(15.0), theme.status_ok)
                            rcol = theme.diff_add_fg
                        elif rk == DIFF_ROW_OLD:
                            dl.add_rect(inner_x + Float32(11.0), row_y, inner_w - Float32(20.0), Float32(15.0), theme.diff_del_bg)
                            dl.add_rect(inner_x + Float32(11.0), row_y, Float32(2.0), Float32(15.0), theme.status_danger)
                            rcol = theme.diff_del_fg
                        _draw_text_mid(dl, m_slot, mono_font if use_mono else font, _truncate_text(rt, d_max_c), inner_x + Float32(16.0), row_y, Float32(15.0), Float32(10.5), rcol)
                    dy += box_h + Float32(6.0)
                else:
                    var max_sc = max(14, Int((inner_w - Float32(24.0)) / Float32(6.3)))
                    _draw_text_mid(dl, font_slot, font, _truncate_text(pa.summary, max_sc), inner_x + Float32(12.0), dy, Float32(16.0), Float32(11.0), theme.text_secondary)
                    dy += Float32(20.0)

                # [✓ Approve] [Allow Turn] [✕ Reject] high-contrast 24px action buttons
                var ab_w = (inner_w - Float32(28.0)) / Float32(3.0)
                var b0_x = inner_x + Float32(10.0)
                var b1_x = b0_x + ab_w + Float32(4.0)
                var b2_x = b1_x + ab_w + Float32(4.0)

                var h0 = _hit_rect(mx, my, b0_x, dy, ab_w, ab_h)
                var h1 = _hit_rect(mx, my, b1_x, dy, ab_w, ab_h)
                var h2 = _hit_rect(mx, my, b2_x, dy, ab_w, ab_h)

                # Solid high-contrast emerald primary Approve CTA (dark ink on bright emerald ~9.5:1)
                dl.add_card_surface(b0_x, dy, ab_w, ab_h, rgba8_with_alpha(theme.status_ok, 248 if h0 else 218), theme.status_ok, theme.radius_xs, Float32(1.0), top_bevel=True, drop_shadow=False)
                var w_app = _measure_safe(bold_font if use_bold else font, "Approve", Float32(11.0))
                var app_sx = b0_x + max(Float32(4.0), (ab_w - (w_app + Float32(14.0))) * Float32(0.5))
                draw_studio_icon(dl, ICON_CHECK, app_sx, dy + Float32(6.5), Float32(11.0), theme.text_on_accent, Float32(1.6))
                _draw_text_mid(dl, b_slot, bold_font if use_bold else font, "Approve", app_sx + Float32(14.0), dy, ab_h, Float32(11.0), theme.text_on_accent)

                # Neutral elevated secondary Allow Turn
                dl.add_card_surface(b1_x, dy, ab_w, ab_h, theme.l3_hover if h1 else theme.l3_raised, theme.border_strong if h1 else theme.border_control, theme.radius_xs, Float32(1.0), top_bevel=True, drop_shadow=False)
                var w_turn = _measure_safe(font, "Allow Turn", Float32(10.5))
                _draw_text_mid(dl, font_slot, font, "Allow Turn", b1_x + max(Float32(4.0), (ab_w - w_turn) * Float32(0.5)), dy, ab_h, Float32(10.5), theme.text_primary)

                # High-contrast Reject (crisp bright text + icon on tinted danger surface ~9:1)
                dl.add_card_surface(b2_x, dy, ab_w, ab_h, rgba8_with_alpha(theme.status_danger, 98 if h2 else 58), rgba8_with_alpha(theme.status_danger, 235 if h2 else 190), theme.radius_xs, Float32(1.0), top_bevel=True, drop_shadow=False)
                var w_rej = _measure_safe(font, "Reject", Float32(11.0))
                var rej_sx = b2_x + max(Float32(4.0), (ab_w - (w_rej + Float32(13.0))) * Float32(0.5))
                draw_studio_icon(dl, ICON_CLOSE, rej_sx, dy + Float32(6.5), Float32(10.5), theme.text_primary, Float32(1.5))
                _draw_text_mid(dl, font_slot, font, "Reject", rej_sx + Float32(13.0), dy, ab_h, Float32(11.0), theme.text_primary)

                if click and h0:
                    var act = AgentPanelAction(PANEL_ACT_APPROVE)
                    act.agent_id = pa.agent_id
                    act.approval_id = pa.approval_id
                    actions.append(act^)
                elif click and h1:
                    var act = AgentPanelAction(PANEL_ACT_APPROVE_TURN)
                    act.agent_id = pa.agent_id
                    act.approval_id = pa.approval_id
                    actions.append(act^)
                elif click and h2:
                    var act = AgentPanelAction(PANEL_ACT_REJECT)
                    act.agent_id = pa.agent_id
                    act.approval_id = pa.approval_id
                    actions.append(act^)

                y += banner_h + Float32(6.0)

        # --------------------------------------------------------------------
        # SECTION 6: Scrollable Step-by-Step Timeline Cards + Sleek Scrollbar
        # --------------------------------------------------------------------
        var scrubber_h = Float32(44.0)
        var scrubber_y = panel_y + panel_h - pad - scrubber_h
        var feed_y = y
        var feed_h = max(Float32(48.0), scrubber_y - feed_y - Float32(6.0))

        var feed_hov = _hit_rect(mx, my, inner_x, feed_y, inner_w, feed_h)
        if feed_hov and isfinite(input.wheel) and input.wheel != Float32(0.0):
            if input.wheel > Float32(0.0):
                self.auto_follow_latest = False
            self.scroll_target_y = max(Float32(0.0), self.scroll_target_y - input.wheel * Float32(36.0))

        dl.add_rect(inner_x, feed_y, inner_w, feed_h, theme.l0_canvas, theme.radius_md, Float32(1.0), theme.border_subtle)
        dl.push_clip(Int(inner_x + Float32(1.0)), Int(feed_y + Float32(1.0)), Int(inner_x + inner_w - Float32(1.0)), Int(feed_y + feed_h - Float32(1.0)))

        var step_count = len(bus.sessions[active_si].steps) if active_si >= 0 else 0

        if step_count == 0:
            self._last_seen_step_count = 0
            _draw_text_safe(
                dl, font_slot, font,
                "No steps yet - pick a role and assign a task.",
                inner_x + Float32(12.0), feed_y + Float32(16.0), Float32(11.5), theme.text_muted,
            )
        else:
            # First pass: estimate total height to clamp scroll smoothly
            var est_h = Float32(6.0)
            var last_turn: UInt32 = 0
            for si_step in range(step_count):
                ref st = bus.sessions[active_si].steps[si_step]
                if st.turn_id != last_turn and st.turn_id != 0:
                    est_h += Float32(22.0)
                    last_turn = st.turn_id
                est_h += self._step_card_height(st) + Float32(5.0)
            var max_scroll = max(Float32(0.0), est_h - feed_h)
            if step_count > self._last_seen_step_count and self.auto_follow_latest and not bus.time_travel_active:
                self.scroll_target_y = max_scroll
                if self.anim_mode == UI_ANIM_INSTANT:
                    self.scroll_spring.snap(max_scroll)
            self._last_seen_step_count = step_count
            if self.scroll_target_y >= max_scroll - Float32(8.0):
                self.auto_follow_latest = True
            self.scroll_target_y = max(Float32(0.0), min(max_scroll, self.scroll_target_y))
            self.scroll_spring.set_target(self.scroll_target_y, snap_if_uninit=True)
            _ = self.scroll_spring.step(safe_dt, self.anim_mode)

            var has_sb = max_scroll > Float32(1.0)
            var sb_reserve = Float32(8.0) if has_sb else Float32(0.0)
            var cy = feed_y + Float32(5.0) - self.scroll_spring.value
            var card_x = inner_x + Float32(6.0)
            var card_w_f = max(Float32(80.0), inner_w - Float32(12.0) - sb_reserve)
            var cur_turn: UInt32 = 0

            for si_step in range(step_count):
                var st_turn_id = bus.sessions[active_si].steps[si_step].turn_id
                if st_turn_id != cur_turn and st_turn_id != 0:
                    cur_turn = st_turn_id
                    if cy + Float32(22.0) >= feed_y and cy <= feed_y + feed_h:
                        _draw_text_safe(dl, font_slot, font, "Turn #" + String(Int(cur_turn)), card_x + Float32(4.0), cy + Float32(4.5), Float32(10.5), theme.text_muted)
                        var rtb_lbl = "Revert Turn"
                        var rtb_w = max(Float32(84.0), _measure_safe(font, rtb_lbl, Float32(10.0)) + Float32(24.0))
                        var rtb_x = card_x + card_w_f - rtb_w
                        var rtb_hov = feed_hov and _hit_rect(mx, my, rtb_x, cy + Float32(1.0), rtb_w, Float32(18.0))
                        dl.add_rect(rtb_x, cy + Float32(1.0), rtb_w, Float32(18.0), theme.l3_hover if rtb_hov else theme.l2_card, Float32(4.0), Float32(1.0), theme.border_subtle)
                        draw_studio_icon(dl, ICON_REVERT, rtb_x + Float32(5.0), cy + Float32(4.5), Float32(11.0), theme.text_secondary, Float32(1.3))
                        _draw_text_safe(dl, font_slot, font, rtb_lbl, rtb_x + Float32(19.0), cy + Float32(5.0), Float32(10.0), theme.text_secondary)
                        if click and rtb_hov:
                            var act = AgentPanelAction(PANEL_ACT_REVERT_TURN)
                            act.agent_id = active_agent_id
                            act.turn_id = cur_turn
                            actions.append(act^)
                    cy += Float32(22.0)

                var ch = self._step_card_height(bus.sessions[active_si].steps[si_step])
                if cy + ch >= feed_y and cy <= feed_y + feed_h:
                    self._render_step_card(
                        dl,
                        bus.sessions[active_si].steps[si_step],
                        active_accent,
                        si_step == bus.time_travel_step_idx and bus.time_travel_active,
                        theme,
                        font_slot,
                        font,
                        b_slot,
                        bold_font if use_bold else font,
                        m_slot,
                        mono_font if use_mono else font,
                        card_x,
                        cy,
                        card_w_f,
                        ch,
                        mx,
                        my,
                        click and feed_hov,
                        feed_hov,
                        t,
                        actions,
                    )
                cy += ch + Float32(5.0)

            # Sleek 4px vertical scrollbar thumb
            if has_sb:
                var track_h = max(Float32(16.0), feed_h - Float32(12.0))
                var thumb_h = max(Float32(18.0), track_h * (feed_h / max(feed_h, est_h)))
                var frac_s = self.scroll_spring.value / max(Float32(1.0), max_scroll)
                var sb_x = inner_x + inner_w - Float32(6.0)
                var sb_y = feed_y + Float32(6.0) + frac_s * max(Float32(0.0), track_h - thumb_h)
                dl.add_rect(sb_x, sb_y, Float32(3.5), thumb_h, theme.border_strong if feed_hov else theme.border_control, Float32(2.0))

        dl.pop_clip()

        # --------------------------------------------------------------------
        # SECTION 7: Bottom Time-Travel Step Scrubber (`Step k / N` + `● LIVE` / `[Return Live]`)
        # --------------------------------------------------------------------
        dl.add_card_surface(
            inner_x, scrubber_y, inner_w, scrubber_h,
            theme.l2_card,
            rgba8_with_alpha(theme.status_warn, 175) if bus.time_travel_active else theme.border_subtle,
            theme.radius_md, Float32(1.0), top_bevel=True, drop_shadow=False,
        )

        var shown_idx = (bus.time_travel_step_idx + 1) if bus.time_travel_active else step_count
        var sc_lbl = "Time-Travel"
        var sc_step = "Step " + String(shown_idx) + " / " + String(step_count)
        var sc_lw = _measure_safe(bold_font if use_bold else font, sc_lbl, Float32(10.5))
        _draw_text_mid(dl, b_slot, bold_font if use_bold else font, sc_lbl, inner_x + Float32(9.0), scrubber_y + Float32(2.0), Float32(20.0), Float32(10.5), theme.status_warn if bus.time_travel_active else theme.text_secondary)
        dl.add_circle(inner_x + Float32(14.0) + sc_lw, scrubber_y + Float32(12.0), Float32(1.4), theme.text_muted)
        _draw_text_mid(dl, m_slot, mono_font if use_mono else font, sc_step, inner_x + Float32(20.0) + sc_lw, scrubber_y + Float32(2.0), Float32(20.0), Float32(10.5), theme.text_primary if bus.time_travel_active else theme.text_muted)

        var live_clicked = False
        if bus.time_travel_active:
            var live_lbl = String("Return Live")
            var live_tw = _measure_safe(bold_font if use_bold else font, live_lbl, Float32(10.0))
            var live_w = max(Float32(72.0), live_tw + Float32(16.0))
            var live_x = inner_x + inner_w - live_w - Float32(6.0)
            var live_y = scrubber_y + Float32(4.0)
            var live_hov = _hit_rect(mx, my, live_x, live_y, live_w, Float32(17.0))
            dl.add_rect(live_x, live_y, live_w, Float32(17.0), rgba8_with_alpha(theme.status_ok, 240 if live_hov else 205), Float32(8.5), Float32(1.0), theme.status_ok)
            _draw_text_mid(dl, b_slot, bold_font if use_bold else font, live_lbl, live_x + max(Float32(4.0), (live_w - live_tw) * Float32(0.5)), live_y, Float32(17.0), Float32(10.0), theme.text_on_accent)
            if click and live_hov:
                var act = AgentPanelAction(PANEL_ACT_TIME_TRAVEL_EXIT)
                act.agent_id = active_agent_id
                actions.append(act^)
                self._scrubber_dragging = False
                live_clicked = True
        else:
            # Subtle emerald LIVE pill indicator when synced to live head
            var live_w = Float32(46.0)
            var live_x = inner_x + inner_w - live_w - Float32(7.0)
            var live_y = scrubber_y + Float32(4.0)
            dl.add_rect(live_x, live_y, live_w, Float32(16.0), rgba8_with_alpha(theme.status_ok, 28), Float32(8.0), Float32(1.0), rgba8_with_alpha(theme.status_ok, 95))
            dl.add_circle(live_x + Float32(9.0), live_y + Float32(8.0), Float32(2.6), theme.status_ok)
            _draw_text_mid(dl, b_slot, bold_font if use_bold else font, "LIVE", live_x + Float32(15.0), live_y, Float32(16.0), Float32(9.5), theme.status_ok)

        # Interactive tick track (thumb travel inset by 6px so it never clips the right border)
        var tr_x = inner_x + Float32(10.0)
        var tr_y = scrubber_y + Float32(28.0)
        var tr_w = max(Float32(32.0), inner_w - Float32(20.0))
        var tr_h = Float32(6.0)
        var thumb_inset = Float32(6.0)
        var usable_w = max(Float32(8.0), tr_w - thumb_inset * Float32(2.0))
        dl.add_rect(tr_x, tr_y, tr_w, tr_h, theme.l0_canvas, Float32(3.0), Float32(1.0), theme.border_subtle)

        if step_count > 0:
            var frac = Float32(shown_idx) / Float32(max(1, step_count))
            var fill_col = theme.status_warn if bus.time_travel_active else active_accent
            var thumb_x = tr_x + thumb_inset + usable_w * max(Float32(0.0), min(Float32(1.0), frac))
            dl.add_rect(tr_x, tr_y, max(Float32(6.0), thumb_x - tr_x), tr_h, fill_col, Float32(3.0))

            # Contrasting dark-rimmed step notch markers visible over any accent fill
            var tick_stride = max(1, (step_count + 63) // 64)
            var k = 0
            while k < step_count:
                var u = (Float32(k) + Float32(0.5)) / Float32(step_count)
                var tx = tr_x + thumb_inset + u * usable_w
                var tick_col = theme.text_primary
                var sk = bus.sessions[active_si].steps[k].kind
                if sk == ACT_CODE_EDIT:
                    tick_col = theme.agent_active
                elif sk == ACT_ENTITY_DESPAWN:
                    tick_col = theme.status_danger
                elif sk == ACT_ENTITY_SPAWN or sk == ACT_ENTITY_TRANSFORM:
                    tick_col = theme.text_primary
                dl.add_rect(tx - Float32(1.5), tr_y - Float32(1.5), Float32(3.0), tr_h + Float32(3.0), theme.l0_canvas, Float32(1.0))
                dl.add_rect(tx - Float32(0.8), tr_y - Float32(1.0), Float32(1.6), tr_h + Float32(2.0), tick_col, Float32(0.8))
                k += tick_stride

            # Tactile drag-scrub handle with soft outer halo
            var thumb_cy = tr_y + tr_h * Float32(0.5)
            dl.add_circle(thumb_x, thumb_cy, Float32(7.5), rgba8_with_alpha(fill_col, 48))
            dl.add_circle(thumb_x, thumb_cy, Float32(5.4), theme.text_primary, Float32(1.8), fill_col)

            var tr_hit = _hit_rect(mx, my, tr_x - Float32(4.0), tr_y - Float32(6.0), tr_w + Float32(8.0), tr_h + Float32(12.0))
            if click:
                self._scrubber_dragging = (not live_clicked) and tr_hit
            if self._scrubber_dragging and down and not live_clicked:
                var norm = max(Float32(0.0), min(Float32(0.9999), (mx - (tr_x + thumb_inset)) / usable_w))
                var tgt_idx = Int(norm * Float32(step_count))
                if tgt_idx != bus.time_travel_step_idx or not bus.time_travel_active:
                    var act = AgentPanelAction(PANEL_ACT_TIME_TRAVEL_SCRUB)
                    act.agent_id = active_agent_id
                    act.step_index = tgt_idx
                    actions.append(act^)

        dl.pop_clip()
        return actions^

    def _step_card_height(self, st: AgentStep) -> Float32:
        var h = Float32(40.0)
        if len(st.affected_entity_ids) > 0 or st.file_path.byte_length() > 0 or st.is_revertible():
            h += Float32(22.0)
        if st.expanded:
            if st.old_text.byte_length() > 0 or st.new_text.byte_length() > 0:
                h += Float32(66.0)
            elif st.terminal_tail.byte_length() > 0 or st.params_json.byte_length() > 0:
                h += Float32(42.0)
        return h

    def _render_step_card(
        mut self,
        mut dl: UiDrawList,
        mut st: AgentStep,
        accent: UInt32,
        is_scrub_selected: Bool,
        theme: StudioTheme,
        font_slot: Int,
        font: SdfFont,
        bold_slot: Int,
        bold_font: SdfFont,
        mono_slot: Int,
        mono_font: SdfFont,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        mx: Float32,
        my: Float32,
        click: Bool,
        allow_hover: Bool,
        t: Float32,
        mut actions: List[AgentPanelAction],
    ):
        var hov = allow_hover and _hit_rect(mx, my, x, y, w, h)
        if hov:
            self.hovered_step_id = st.step_id
        var bg = theme.l3_raised if is_scrub_selected else (theme.l3_hover if hov else theme.l2_card)
        var bdr = theme.status_warn if is_scrub_selected else (theme.border_strong if hov else theme.border_subtle)
        dl.add_card_surface(x, y, w, h, bg, bdr, theme.radius_sm, Float32(1.0), top_bevel=True, drop_shadow=False)

        # Inset left role/status accent bar (clears rounded corners cleanly)
        var bar_col = accent
        if st.status == STEP_FAILED or st.status == STEP_REJECTED:
            bar_col = theme.status_danger
        elif st.status == STEP_REVERTED:
            bar_col = theme.text_muted
        elif st.status == STEP_AWAITING_APPROVAL:
            bar_col = theme.status_warn
        dl.add_accent_bar(x + Float32(3.0), y + Float32(6.0), Float32(2.5), h - Float32(12.0), bar_col, pill=True)

        # Human-readable summary as primary headline; raw tool ID as crisp secondary mono tag
        var primary_txt = st.summary if st.summary.byte_length() > 0 else st.title
        var secondary_txt = st.title if st.summary.byte_length() > 0 else String("")
        draw_studio_icon(dl, _action_icon(st.kind), x + Float32(11.0), y + Float32(9.0), Float32(13.0), bar_col, Float32(1.4))
        var max_tc = max(12, Int((w - Float32(90.0)) / Float32(6.4)))
        _draw_text_safe(
            dl, bold_slot, bold_font,
            _truncate_text(primary_txt, max_tc),
            x + Float32(29.0), y + Float32(6.5), Float32(11.5),
            theme.text_muted if st.status == STEP_REVERTED else theme.text_primary,
        )
        if secondary_txt.byte_length() > 0:
            _draw_text_safe(
                dl, mono_slot, mono_font,
                _truncate_text(secondary_txt, max_tc + 6),
                x + Float32(29.0), y + Float32(21.5), Float32(10.0),
                theme.text_muted,
            )

        # Right status glyph / PRIM_ARC spinner + elapsed ms + expand chevron
        var ms_txt = _fmt_ms(st.elapsed_ms)
        if ms_txt.byte_length() > 0:
            var mw = _measure_safe(mono_font, ms_txt, Float32(10.0))
            _draw_text_safe(dl, mono_slot, mono_font, ms_txt, x + w - Float32(43.0) - mw, y + Float32(7.5), Float32(10.0), theme.text_muted)

        var st_cx = x + w - Float32(30.0)
        var st_cy = y + Float32(14.0)
        if st.status == STEP_RUNNING:
            dl.add_arc(st_cx, st_cy, Float32(5.5), Float32(1.6), t * Float32(5.0), Float32(3.8), accent, rgba8_with_alpha(accent, 45), Float32(1.5), 0)
        elif st.status == STEP_AWAITING_APPROVAL:
            dl.add_circle(st_cx, st_cy, Float32(6.5), rgba8_with_alpha(theme.status_warn, 32))
            draw_studio_icon(dl, ICON_WARN, st_cx - Float32(5.0), st_cy - Float32(5.0), Float32(10.0), theme.status_warn, Float32(1.4))
        elif st.status == STEP_COMPLETED:
            draw_studio_icon(dl, ICON_CHECK, st_cx - Float32(5.5), st_cy - Float32(5.5), Float32(11.0), theme.status_ok, Float32(1.5))
        elif st.status == STEP_FAILED or st.status == STEP_REJECTED:
            draw_studio_icon(dl, ICON_ERROR_CIRCLE, st_cx - Float32(5.5), st_cy - Float32(5.5), Float32(11.0), theme.status_danger, Float32(1.4))
        elif st.status == STEP_REVERTED:
            draw_studio_icon(dl, ICON_REVERT, st_cx - Float32(5.5), st_cy - Float32(5.5), Float32(11.0), theme.text_muted, Float32(1.4))

        var chev_x = x + w - Float32(18.0)
        var chev_y = y + Float32(8.0)
        draw_studio_chevron(dl, chev_x, chev_y, Float32(12.0), theme.text_secondary, Float32(1.0) if st.expanded else Float32(0.0), Float32(1.4))
        if click and _hit_rect(mx, my, x + Float32(6.0), y + Float32(2.0), w - Float32(12.0), Float32(34.0)):
            st.expanded = not st.expanded

        # Row 2: Clickable Entity Chips, File Diff Chips & [Revert Step]
        var chip_y = y + Float32(37.0)
        var chip_x = x + Float32(11.0)
        var chip_h = Float32(19.0)

        for ei in range(min(3, len(st.affected_entity_ids))):
            var eid = st.affected_entity_ids[ei]
            var lbl = "#" + String(eid)
            var cw = _measure_safe(mono_font, lbl, Float32(10.0)) + Float32(22.0)
            if chip_x + cw > x + w - Float32(72.0):
                break
            var chov = allow_hover and _hit_rect(mx, my, chip_x, chip_y, cw, chip_h)
            if chov:
                self.hovered_entity_id = eid
            dl.add_rect(chip_x, chip_y, cw, chip_h, rgba8_with_alpha(accent, 64 if chov else 30), Float32(9.5), Float32(1.0), rgba8_with_alpha(accent, 185 if chov else 120))
            draw_studio_icon(dl, ICON_CUBE, chip_x + Float32(5.0), chip_y + Float32(4.0), Float32(11.0), accent, Float32(1.3))
            _draw_text_mid(dl, mono_slot, mono_font, lbl, chip_x + Float32(18.0), chip_y, chip_h, Float32(10.0), theme.text_primary)
            if click and chov:
                var act = AgentPanelAction(PANEL_ACT_FOCUS_ENTITY)
                act.agent_id = st.agent_id
                act.step_id = st.step_id
                act.entity_id = eid
                actions.append(act^)
            chip_x += cw + Float32(5.0)

        if st.file_path.byte_length() > 0:
            var flbl = _truncate_text(st.file_path, 18)
            var fw = _measure_safe(mono_font, flbl, Float32(10.0)) + Float32(22.0)
            if chip_x + fw <= x + w - Float32(66.0):
                var fhov = allow_hover and _hit_rect(mx, my, chip_x, chip_y, fw, chip_h)
                dl.add_rect(chip_x, chip_y, fw, chip_h, theme.l3_hover if fhov else theme.l3_raised, Float32(9.5), Float32(1.0), theme.border_strong if fhov else theme.border_control)
                draw_studio_icon(dl, ICON_DIFF, chip_x + Float32(5.0), chip_y + Float32(4.0), Float32(11.0), theme.agent_active, Float32(1.3))
                _draw_text_mid(dl, mono_slot, mono_font, flbl, chip_x + Float32(18.0), chip_y, chip_h, Float32(10.0), theme.text_primary)
                if click and fhov:
                    var act = AgentPanelAction(PANEL_ACT_OPEN_FILE_DIFF)
                    act.agent_id = st.agent_id
                    act.step_id = st.step_id
                    act.text = st.file_path
                    actions.append(act^)

        if st.is_revertible():
            var rv_w = max(Float32(62.0), _measure_safe(font, "Revert", Float32(10.0)) + Float32(24.0))
            var rv_x = x + w - rv_w - Float32(8.0)
            var rhov = allow_hover and _hit_rect(mx, my, rv_x, chip_y, rv_w, chip_h)
            dl.add_rect(rv_x, chip_y, rv_w, chip_h, theme.l3_hover if rhov else theme.l3_raised, theme.radius_xs, Float32(1.0), theme.border_strong if rhov else theme.border_control)
            draw_studio_icon(dl, ICON_REVERT, rv_x + Float32(5.0), chip_y + Float32(4.0), Float32(11.0), theme.text_primary if rhov else theme.text_secondary, Float32(1.3))
            _draw_text_mid(dl, font_slot, font, "Revert", rv_x + Float32(19.0), chip_y, chip_h, Float32(10.0), theme.text_primary if rhov else theme.text_secondary)
            if click and rhov:
                var act = AgentPanelAction(PANEL_ACT_REVERT_STEP)
                act.agent_id = st.agent_id
                act.step_id = st.step_id
                actions.append(act^)

        # Row 3: Collapsible inline detail drawer (diff with row fills / terminal tail / formatted JSON)
        if st.expanded:
            var dy = y + Float32(64.0)
            var dw = w - Float32(18.0)
            var d_max_c = max(14, Int((dw - Float32(14.0)) / Float32(6.0)))
            if st.old_text.byte_length() > 0 or st.new_text.byte_length() > 0:
                var old_s = DiffSide.from_text(st.old_text)
                var new_s = DiffSide.from_text(st.new_text)
                var hunks = diff_sides(old_s, new_s)
                var drows = build_unified_diff(old_s, new_s, hunks, context=1)
                dl.add_rect(x + Float32(10.0), dy, dw, Float32(60.0), theme.l0_canvas, theme.radius_xs, Float32(1.0), theme.border_subtle)
                for di in range(min(4, len(drows))):
                    var rcol = theme.text_secondary
                    var r_y = dy + Float32(2.0) + Float32(di) * Float32(14.0)
                    if drows[di].kind == DIFF_ROW_NEW:
                        dl.add_rect(x + Float32(11.0), r_y, dw - Float32(2.0), Float32(13.5), theme.diff_add_bg)
                        dl.add_rect(x + Float32(11.0), r_y, Float32(2.0), Float32(13.5), theme.status_ok)
                        rcol = theme.diff_add_fg
                    elif drows[di].kind == DIFF_ROW_OLD:
                        dl.add_rect(x + Float32(11.0), r_y, dw - Float32(2.0), Float32(13.5), theme.diff_del_bg)
                        dl.add_rect(x + Float32(11.0), r_y, Float32(2.0), Float32(13.5), theme.status_danger)
                        rcol = theme.diff_del_fg
                    _draw_text_mid(dl, mono_slot, mono_font, _truncate_text(drows[di].text, d_max_c), x + Float32(16.0), r_y, Float32(13.5), Float32(10.0), rcol)
            elif st.terminal_tail.byte_length() > 0 or st.params_json.byte_length() > 0:
                dl.add_rect(x + Float32(10.0), dy, dw, Float32(36.0), theme.l0_canvas, theme.radius_xs, Float32(1.0), theme.border_subtle)
                var body = st.terminal_tail if st.terminal_tail.byte_length() > 0 else _format_payload_preview(st.params_json)
                _draw_text_safe(dl, mono_slot, mono_font, _truncate_text(body, d_max_c), x + Float32(14.0), dy + Float32(5.0), Float32(10.0), theme.text_secondary)
                if st.flip_score >= Float32(0.0):
                    _draw_text_safe(dl, mono_slot, mono_font, "FLIP=" + String(Float64(st.flip_score)), x + Float32(14.0), dy + Float32(20.0), Float32(10.0), theme.status_ok)
