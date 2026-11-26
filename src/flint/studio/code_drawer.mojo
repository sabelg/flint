"""Native GPU-Rendered Code, Diff & Problems Drawer for Flint Studio (`flint.studio.code_drawer`).

Renders directly inside Flint Studio's `DockSpace` via `UiDrawList` + `SdfFont`
(zero Vulkan swapchain detach, full parity between Vulkan GPU SDF UI and Tier-0
CPU SIMD rasterizer). Integrates:
- File tabs with dirty dot, file-type vector icon (`draw_studio_icon`), and close button
- Scope breadcrumb bar (`Struct > method`) via `index_mojo_source` / `scope_breadcrumbs`
- 14-class syntax-highlighted Mojo/TOML/JSON/C/Python/Markdown code (`MojoHighlighter`)
- Line numbers, Myers diff gutter bars (`GUTTER_*`), and inline diagnostic squiggles
- Unified and Side-by-Side Split diff modes (`DiffDocument`) with clickable per-hunk
  `[Accept]` / `[Reject]` buttons and agent proposal `[Accept All]` / `[Reject All]`
- Collapsible Problems strip (`ProblemStore`) with click-to-jump & `[Ask Agent]`
- Live `[Build & Hot-Reload]` status pill with `PRIM_ARC` progress spinner & error badge
"""
from std.collections import List
from std.math import max, min
from flint.devtools.diff_view import (
    DIFF_ROW_HEADER,
    DIFF_ROW_NEW,
    DIFF_ROW_NOTE,
    DIFF_ROW_OLD,
    DIFF_ROW_SAME,
    DIFF_ROW_SPLIT_CHG,
    DiffDocument,
    DiffRow,
    GUTTER_ADDED,
    GUTTER_CHANGED,
    GUTTER_DELETED_ABOVE,
    GUTTER_DELETED_BELOW,
    GUTTER_NONE,
    HUNK_ACCEPTED,
    HUNK_PENDING,
    HUNK_REJECTED,
)
from flint.devtools.mojo_highlight import (
    HighlightSpan,
    LANG_MOJO,
    MojoHighlighter,
    TOK_TEXT,
    lang_for_path,
)
from flint.devtools.problems import (
    DIAG_ERROR,
    DIAG_WARNING,
    Diagnostic,
    ProblemFilter,
    ProblemStore,
    SymbolEntry,
    SymbolIndex,
    index_mojo_source,
    scope_breadcrumbs,
)
from flint.platform.input import InputState
from flint.platform.os import read_bytes
from flint.render.font import SdfFont
from flint.ui.draw_list import UI_FLAG_TOP_BEVEL, UiDrawList
from flint.ui.icons import draw_studio_chevron, draw_studio_icon
from flint.ui.theme import (
    ICON_CHECK,
    ICON_CLOSE,
    ICON_CODE,
    ICON_DIFF,
    ICON_PLAY,
    ICON_REVERT,
    ICON_SPARK,
    ICON_WARN,
    StudioTheme,
    icon_for_path,
    rgba8_with_alpha,
)

# Drawer view modes
comptime CODE_MODE_SOURCE: Int = 0
comptime CODE_MODE_DIFF_UNIFIED: Int = 1
comptime CODE_MODE_DIFF_SPLIT: Int = 2

# Build & Hot-Reload pill states
comptime BUILD_STATUS_IDLE: Int = 0
comptime BUILD_STATUS_BUILDING: Int = 1
comptime BUILD_STATUS_OK: Int = 2
comptime BUILD_STATUS_ERROR: Int = 3

# Typed actions returned to `studio_app.mojo`
comptime CODE_ACT_NONE: Int = 0
comptime CODE_ACT_SELECT_TAB: Int = 1
comptime CODE_ACT_CLOSE_TAB: Int = 2
comptime CODE_ACT_BUILD_HOT_RELOAD: Int = 3
comptime CODE_ACT_ACCEPT_HUNK: Int = 4
comptime CODE_ACT_REJECT_HUNK: Int = 5
comptime CODE_ACT_ACCEPT_ALL_DIFF: Int = 6
comptime CODE_ACT_REJECT_ALL_DIFF: Int = 7
comptime CODE_ACT_JUMP_DIAGNOSTIC: Int = 8
comptime CODE_ACT_ASK_AGENT_FIX: Int = 9
comptime CODE_ACT_OPEN_EXTERNAL_IDE: Int = 10
comptime CODE_ACT_TOGGLE_PROBLEMS: Int = 11


def _basename(path: String) -> String:
    var b = path.as_bytes()
    var last = -1
    for i in range(len(b)):
        if b[i] == 47:
            last = i
    if last < 0:
        return path
    return String(path[byte=last + 1 : len(b)])


def _split_lines(text: String) -> List[String]:
    var out = List[String]()
    if text.byte_length() == 0:
        out.append("")
        return out^
    for raw in text.split("\n"):
        out.append(String(String(raw).rstrip("\r")))
    if text.endswith("\n") and len(out) > 1 and out[len(out) - 1].byte_length() == 0:
        _ = out.pop()
    return out^


@always_inline
def _hit_rect(mx: Float32, my: Float32, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
    return mx >= x and mx < x + w and my >= y and my < y + h


def _truncate_text(text: String, max_chars: Int) -> String:
    if max_chars <= 3:
        return ""
    if text.byte_length() <= max_chars:
        return text
    var b = text.as_bytes()
    var cut = min(len(b), max_chars - 2)
    while cut > 0 and (b[cut] & 0xC0) == 0x80:
        cut -= 1
    return String(text[byte=0:cut]) + ".."


@fieldwise_init
struct CodeDrawerAction(Copyable, Movable):
    """Typed user interaction event emitted by `CodeDrawer.draw_and_handle`."""
    var kind: Int
    var tab_index: Int
    var hunk_index: Int
    var path: String
    var line: Int        # 1-based line
    var col: Int         # 1-based col
    var message: String

    def __init__(out self, kind: Int = CODE_ACT_NONE):
        self.kind = kind
        self.tab_index = -1
        self.hunk_index = -1
        self.path = ""
        self.line = 1
        self.col = 1
        self.message = ""


struct CodeTab(Copyable, Movable):
    """Open file buffer inside `CodeDrawer` with cached highlight spans, outline & diff."""
    var path: String
    var title: String
    var lang: Int
    var dirty: Bool
    var text: String
    var base_text: String
    var lines: List[String]
    var spans: List[List[HighlightSpan]]
    var symbols: List[SymbolEntry]
    var diff_doc: DiffDocument
    var has_proposal: Bool
    var proposal_request_id: String
    var cursor_line: Int      # 0-based
    var scroll_line: Int      # 0-based top visible line

    def __init__(out self, path: String = "", text: String = "", base_text: String = ""):
        self.path = path
        self.title = _basename(path) if path.byte_length() > 0 else String("untitled.mojo")
        self.lang = lang_for_path(path, self.title)
        self.dirty = False
        self.text = text
        self.base_text = base_text if base_text.byte_length() > 0 else text
        self.lines = _split_lines(text)
        self.spans = List[List[HighlightSpan]]()
        self.symbols = index_mojo_source(text, path) if self.lang == LANG_MOJO else List[SymbolEntry]()
        self.diff_doc = DiffDocument(path, self.base_text, text)
        self.has_proposal = False
        self.proposal_request_id = ""
        self.cursor_line = 0
        self.scroll_line = 0

    def rebuild_caches(mut self, hl: MojoHighlighter):
        self.lines = _split_lines(self.text)
        self.spans = hl.highlight_lines(self.lines, self.lang)
        if self.lang == LANG_MOJO:
            self.symbols = index_mojo_source(self.text, self.path)
        else:
            self.symbols.clear()
        if not self.has_proposal:
            self.diff_doc = DiffDocument(self.path, self.base_text, self.text)


struct CodeDrawer(Movable):
    """Native GPU SDF Code, Diff & Problems drawer for Flint Studio."""
    var tabs: List[CodeTab]
    var active_tab: Int
    var view_mode: Int
    var problems_open: Bool
    var problems_height: Float32
    var problems_scroll: Int
    var build_status: Int
    var build_generation: Int
    var build_error_count: Int
    var build_warning_count: Int
    var build_label: String
    var highlighter: MojoHighlighter
    var problems: ProblemStore
    var symbols: SymbolIndex
    var has_keyboard_focus: Bool
    var last_hit_count: Int

    def __init__(out self):
        self.tabs = List[CodeTab]()
        self.active_tab = -1
        self.view_mode = CODE_MODE_SOURCE
        self.problems_open = True
        self.problems_height = Float32(118.0)
        self.problems_scroll = 0
        self.build_status = BUILD_STATUS_IDLE
        self.build_generation = 0
        self.build_error_count = 0
        self.build_warning_count = 0
        self.build_label = "Build & Reload"
        self.highlighter = MojoHighlighter()
        self.problems = ProblemStore()
        self.symbols = SymbolIndex()
        self.has_keyboard_focus = False
        self.last_hit_count = 0

    def find_tab(self, path: String) -> Int:
        for i in range(len(self.tabs)):
            if self.tabs[i].path == path:
                return i
        return -1

    def open_source(
        mut self,
        path: String,
        text: String,
        base_text: String = "",
        dirty: Bool = False,
        focus_line_1based: Int = 1,
    ) -> Int:
        """Opens or updates a tab from in-memory source text and focuses `focus_line_1based`."""
        var idx = self.find_tab(path)
        if idx < 0:
            var tab = CodeTab(path, text, base_text)
            tab.dirty = dirty
            tab.rebuild_caches(self.highlighter)
            self.tabs.append(tab^)
            idx = len(self.tabs) - 1
        else:
            self.tabs[idx].text = text
            if base_text.byte_length() > 0:
                self.tabs[idx].base_text = base_text
            self.tabs[idx].dirty = dirty
            self.tabs[idx].rebuild_caches(self.highlighter)
        if self.tabs[idx].lang == LANG_MOJO:
            self.symbols.index_source(path, text)
        self.active_tab = idx
        self.jump_to_line(focus_line_1based)
        return idx

    def open_file(mut self, path: String, focus_line_1based: Int = 1) -> Int:
        """Loads a UTF-8 file from disk into a tab and focuses `focus_line_1based`."""
        var raw = read_bytes(path)
        var text = String(unsafe_from_utf8=raw) if len(raw) > 0 else String("")
        return self.open_source(path, text, text, False, focus_line_1based)

    def close_tab(mut self, idx: Int):
        if idx < 0 or idx >= len(self.tabs):
            return
        _ = self.tabs.pop(idx)
        if len(self.tabs) == 0:
            self.active_tab = -1
        elif self.active_tab >= len(self.tabs):
            self.active_tab = len(self.tabs) - 1

    def jump_to_line(mut self, line_1based: Int):
        if self.active_tab < 0 or self.active_tab >= len(self.tabs):
            return
        ref tab = self.tabs[self.active_tab]
        var max_ln = max(0, len(tab.lines) - 1)
        var ln = max(0, min(max_ln, line_1based - 1))
        self.tabs[self.active_tab].cursor_line = ln
        if ln < self.tabs[self.active_tab].scroll_line:
            self.tabs[self.active_tab].scroll_line = ln
        elif ln > self.tabs[self.active_tab].scroll_line + 12:
            self.tabs[self.active_tab].scroll_line = max(0, ln - 4)

    def show_agent_proposal(
        mut self,
        path: String,
        old_text: String,
        new_text: String,
        request_id: String = "",
        split_mode: Bool = False,
    ) -> Int:
        """Opens `path` in Diff mode showing a pending agent `FileWriteProposal` with Accept/Reject hunks."""
        var idx = self.open_source(path, old_text, old_text, False, 1)
        self.tabs[idx].diff_doc = DiffDocument(path, old_text, new_text)
        self.tabs[idx].has_proposal = True
        self.tabs[idx].proposal_request_id = request_id
        self.tabs[idx].scroll_line = 0
        self.view_mode = CODE_MODE_DIFF_SPLIT if split_mode else CODE_MODE_DIFF_UNIFIED
        return idx

    def apply_hunk_decision(mut self, hunk_idx: Int, accept: Bool) -> Bool:
        if self.active_tab < 0 or self.active_tab >= len(self.tabs):
            return False
        var ok = (
            self.tabs[self.active_tab].diff_doc.accept_hunk(hunk_idx)
            if accept
            else self.tabs[self.active_tab].diff_doc.reject_hunk(hunk_idx)
        )
        if ok and self.tabs[self.active_tab].diff_doc.pending_count() == 0:
            self.tabs[self.active_tab].text = self.tabs[self.active_tab].diff_doc.materialized_text()
            self.tabs[self.active_tab].dirty = (
                self.tabs[self.active_tab].text != self.tabs[self.active_tab].base_text
            )
            self.tabs[self.active_tab].has_proposal = False
            self.tabs[self.active_tab].rebuild_caches(self.highlighter)
        return ok

    def apply_all_diff(mut self, accept: Bool):
        if self.active_tab < 0 or self.active_tab >= len(self.tabs):
            return
        if accept:
            self.tabs[self.active_tab].diff_doc.accept_all()
        else:
            self.tabs[self.active_tab].diff_doc.reject_all()
        self.tabs[self.active_tab].text = self.tabs[self.active_tab].diff_doc.materialized_text()
        self.tabs[self.active_tab].dirty = (
            self.tabs[self.active_tab].text != self.tabs[self.active_tab].base_text
        )
        self.tabs[self.active_tab].has_proposal = False
        self.tabs[self.active_tab].rebuild_caches(self.highlighter)
        self.view_mode = CODE_MODE_SOURCE

    def set_build_status(
        mut self, status: Int, generation: Int = 0, errors: Int = 0, warnings: Int = 0, label: String = ""
    ):
        self.build_status = status
        self.build_generation = generation
        self.build_error_count = errors
        self.build_warning_count = warnings
        if label.byte_length() > 0:
            self.build_label = label
        elif status == BUILD_STATUS_BUILDING:
            self.build_label = "Building .so…"
        elif status == BUILD_STATUS_OK:
            self.build_label = "Reloaded gen " + String(generation)
        elif status == BUILD_STATUS_ERROR:
            self.build_label = String(errors) + (" Error" if errors == 1 else " Errors")
        else:
            self.build_label = "Build & Reload"

    def _syntax_color(self, theme: StudioTheme, tok_kind: Int) -> UInt32:
        if tok_kind >= 0 and tok_kind < len(theme.syntax):
            return theme.syntax[tok_kind]
        return theme.text_primary

    def _draw_syntax_text(
        self,
        mut dl: UiDrawList,
        font_slot: Int,
        font: SdfFont,
        theme: StudioTheme,
        line: String,
        spans: List[HighlightSpan],
        x: Float32,
        baseline_y: Float32,
        font_px: Float32,
    ):
        var pen_x = x
        if len(spans) == 0:
            if line.byte_length() > 0:
                dl.add_text(font_slot, font, line, pen_x, baseline_y, font_px, theme.text_primary)
            return
        for si in range(len(spans)):
            ref sp = spans[si]
            if sp.end_byte <= sp.start_byte or sp.end_byte > line.byte_length():
                continue
            var seg = String(line[byte=sp.start_byte : sp.end_byte])
            var col = self._syntax_color(theme, sp.token_kind)
            dl.add_text(font_slot, font, seg, pen_x, baseline_y, font_px, col)
            pen_x += font.measure(seg, font_px)

    def _draw_squiggle(
        self,
        mut dl: UiDrawList,
        x0: Float32,
        y: Float32,
        width: Float32,
        color: UInt32,
    ):
        var w = max(Float32(10.0), width)
        var step = Float32(4.0)
        var amp = Float32(1.5)
        var cx = x0
        var up = True
        while cx < x0 + w:
            var nx = min(x0 + w, cx + step)
            var y0 = y - amp if up else y + amp
            var y1 = y + amp if up else y - amp
            dl.add_line(cx, y0, nx, y1, Float32(1.2), color)
            cx = nx
            up = not up

    def draw_and_handle(
        mut self,
        mut dl: UiDrawList,
        font_slot: Int,
        font: SdfFont,
        theme: StudioTheme,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        inp: InputState,
        time_sec: Float32 = 0.0,
    ) -> CodeDrawerAction:
        """Draws the complete GPU-rendered Code/Diff/Problems drawer and processes pointer/keyboard input."""
        var action = CodeDrawerAction(CODE_ACT_NONE)
        self.last_hit_count = 0
        if w < Float32(64.0) or h < Float32(64.0):
            return action^

        var mx = inp.last_mouse_x
        var my = inp.last_mouse_y
        var click = inp.mouse_pressed(0)
        var inside_drawer = _hit_rect(mx, my, x, y, w, h)
        if click:
            self.has_keyboard_focus = inside_drawer

        dl.push_clip(Int(x), Int(y), Int(x + w), Int(y + h))

        # Outer shell surface
        dl.add_rect(x, y, w, h, theme.l1_shell, Float32(0.0), Float32(1.0), theme.border_subtle)

        var tab_bar_h = Float32(30.0)
        var crumb_h = Float32(24.0)
        var prop_bar_h = Float32(0.0)
        var has_active_prop = (
            self.active_tab >= 0
            and self.active_tab < len(self.tabs)
            and self.tabs[self.active_tab].has_proposal
        )
        if has_active_prop:
            prop_bar_h = Float32(28.0)

        var prob_hdr_h = Float32(24.0)
        var prob_total_h = (
            min(self.problems_height, max(Float32(64.0), h * Float32(0.34)))
            if self.problems_open
            else prob_hdr_h
        )
        if prob_total_h > h - tab_bar_h - crumb_h - prop_bar_h - Float32(48.0):
            prob_total_h = prob_hdr_h

        # --------------------------------------------------------------------
        # 1. Top Tab Bar + View Mode Pills + [Build & Hot-Reload] Status Pill
        # --------------------------------------------------------------------
        dl.add_rect(x, y, w, tab_bar_h, theme.l0_canvas)
        dl.add_rect(x, y + tab_bar_h - Float32(1.0), w, Float32(1.0), theme.border_subtle)

        # Right-aligned Build & Hot-Reload pill + Mode pills
        var pill_h = Float32(22.0)
        var pill_y = y + Float32(4.0)
        var build_w = Float32(138.0)
        var build_x = x + w - build_w - Float32(8.0)
        var build_hov = _hit_rect(mx, my, build_x, pill_y, build_w, pill_h)
        self.last_hit_count += 1

        var build_bg = theme.l3_hover if build_hov else theme.l3_raised
        var build_accent = theme.accent_primary
        if self.build_status == BUILD_STATUS_BUILDING:
            build_bg = rgba8_with_alpha(theme.accent_primary, 42)
            build_accent = theme.accent_primary
        elif self.build_status == BUILD_STATUS_OK:
            build_bg = rgba8_with_alpha(theme.status_ok, 38)
            build_accent = theme.status_ok
        elif self.build_status == BUILD_STATUS_ERROR:
            build_bg = rgba8_with_alpha(theme.status_danger, 44)
            build_accent = theme.status_danger

        dl.add_rect_styled(
            build_x, pill_y, build_w, pill_h,
            build_bg, build_accent,
            theme.radius_sm, Float32(1.0), Float32(0.0), UI_FLAG_TOP_BEVEL,
        )
        if self.build_status == BUILD_STATUS_BUILDING:
            var spin_a = time_sec * Float32(5.2)
            dl.add_arc(
                build_x + Float32(13.0),
                pill_y + pill_h * Float32(0.5),
                Float32(5.5),
                Float32(1.8),
                spin_a,
                Float32(4.2),
                build_accent,
                rgba8_with_alpha(build_accent, 55),
            )
        elif self.build_status == BUILD_STATUS_ERROR:
            draw_studio_icon(dl, ICON_WARN, build_x + Float32(7.0), pill_y + Float32(4.0), Float32(13.0), build_accent)
        elif self.build_status == BUILD_STATUS_OK:
            draw_studio_icon(dl, ICON_CHECK, build_x + Float32(7.0), pill_y + Float32(4.0), Float32(13.0), build_accent)
        else:
            draw_studio_icon(dl, ICON_PLAY, build_x + Float32(7.0), pill_y + Float32(4.0), Float32(13.0), build_accent)
        dl.add_text(
            font_slot, font,
            _truncate_text(self.build_label, 16),
            build_x + Float32(25.0), pill_y + Float32(15.0),
            Float32(11.5), theme.text_primary,
        )
        if click and build_hov:
            action = CodeDrawerAction(CODE_ACT_BUILD_HOT_RELOAD)

        # Mode segmented pills (Code | Diff | Split) + External IDE button
        var ext_w = Float32(26.0)
        var ext_x = build_x - ext_w - Float32(6.0)
        var ext_hov = _hit_rect(mx, my, ext_x, pill_y, ext_w, pill_h)
        self.last_hit_count += 1
        dl.add_rect(
            ext_x, pill_y, ext_w, pill_h,
            theme.l3_hover if ext_hov else theme.l2_card,
            theme.radius_xs, Float32(1.0), theme.border_control,
        )
        draw_studio_icon(dl, ICON_CODE, ext_x + Float32(6.0), pill_y + Float32(4.0), Float32(13.0), theme.text_secondary)
        if click and ext_hov and action.kind == CODE_ACT_NONE:
            action = CodeDrawerAction(CODE_ACT_OPEN_EXTERNAL_IDE)

        var mode_labels = List[String]()
        mode_labels.append("Code")
        mode_labels.append("Diff")
        mode_labels.append("Split")
        var seg_w = Float32(44.0)
        var seg_total_w = seg_w * Float32(3.0)
        var seg_x0 = ext_x - seg_total_w - Float32(8.0)
        if seg_x0 > x + Float32(120.0):
            dl.add_rect(
                seg_x0, pill_y, seg_total_w, pill_h,
                theme.l2_card, theme.radius_xs, Float32(1.0), theme.border_subtle,
            )
            for mi in range(3):
                var sx = seg_x0 + Float32(mi) * seg_w
                var shov = _hit_rect(mx, my, sx, pill_y, seg_w, pill_h)
                var active_m = self.view_mode == mi
                self.last_hit_count += 1
                if active_m or shov:
                    dl.add_rect(
                        sx + Float32(1.0), pill_y + Float32(1.0),
                        seg_w - Float32(2.0), pill_h - Float32(2.0),
                        theme.accent_soft if active_m else theme.l3_hover,
                        theme.radius_xs,
                    )
                dl.add_text(
                    font_slot, font, mode_labels[mi],
                    sx + Float32(8.0), pill_y + Float32(15.0),
                    Float32(11.0),
                    theme.accent_primary if active_m else theme.text_secondary,
                )
                if click and shov and action.kind == CODE_ACT_NONE:
                    self.view_mode = mi

        # Left-aligned File Tabs
        var tabs_limit_x = seg_x0 - Float32(8.0) if seg_x0 > x + Float32(120.0) else ext_x - Float32(6.0)
        var tx = x + Float32(4.0)
        for ti in range(len(self.tabs)):
            var tw = Float32(146.0)
            if tx + Float32(60.0) > tabs_limit_x:
                break
            tw = min(tw, tabs_limit_x - tx)
            var is_act = ti == self.active_tab
            var thov = _hit_rect(mx, my, tx, y + Float32(2.0), tw, tab_bar_h - Float32(2.0))
            var close_x = tx + tw - Float32(19.0)
            var close_y = y + Float32(8.0)
            var chov = _hit_rect(mx, my, close_x, close_y, Float32(14.0), Float32(14.0))
            self.last_hit_count += 2

            dl.add_rect(
                tx, y + Float32(3.0), tw - Float32(2.0), tab_bar_h - Float32(3.0),
                theme.l1_shell if is_act else (theme.l2_card if thov else theme.l0_canvas),
                theme.radius_xs, Float32(1.0) if is_act else Float32(0.0), theme.border_subtle,
            )
            if is_act:
                dl.add_rect(tx + Float32(2.0), y + Float32(3.0), tw - Float32(6.0), Float32(2.0), theme.accent_primary)

            var ic = icon_for_path(self.tabs[ti].path)
            draw_studio_icon(
                dl, ic, tx + Float32(7.0), y + Float32(9.0), Float32(13.0),
                theme.accent_primary if is_act else theme.text_muted,
            )
            var label_x = tx + Float32(24.0)
            if self.tabs[ti].dirty:
                dl.add_circle(label_x + Float32(3.0), y + Float32(15.0), Float32(3.0), theme.status_warn)
                label_x += Float32(10.0)
            var max_ch = max(4, Int((close_x - label_x) / Float32(7.0)))
            dl.add_text(
                font_slot, font,
                _truncate_text(self.tabs[ti].title, max_ch),
                label_x, y + Float32(19.0), Float32(11.5),
                theme.text_primary if is_act else theme.text_secondary,
            )
            draw_studio_icon(
                dl, ICON_CLOSE, close_x + Float32(2.0), close_y + Float32(2.0), Float32(10.0),
                theme.text_primary if chov else theme.text_muted,
            )

            if click and chov and action.kind == CODE_ACT_NONE:
                action = CodeDrawerAction(CODE_ACT_CLOSE_TAB)
                action.tab_index = ti
                action.path = self.tabs[ti].path
            elif click and thov and action.kind == CODE_ACT_NONE:
                self.active_tab = ti
                action = CodeDrawerAction(CODE_ACT_SELECT_TAB)
                action.tab_index = ti
                action.path = self.tabs[ti].path
            tx += tw

        # --------------------------------------------------------------------
        # 2. Scope Breadcrumb Bar
        # --------------------------------------------------------------------
        var crumb_y = y + tab_bar_h
        dl.add_rect(x, crumb_y, w, crumb_h, theme.l1_shell)
        dl.add_rect(x, crumb_y + crumb_h - Float32(1.0), w, Float32(1.0), theme.border_subtle)
        if self.active_tab >= 0 and self.active_tab < len(self.tabs):
            ref tab = self.tabs[self.active_tab]
            var scope = scope_breadcrumbs(tab.symbols, tab.cursor_line)
            var crumb_txt = tab.path if tab.path.byte_length() > 0 else tab.title
            if scope.byte_length() > 0:
                crumb_txt += "  >  " + scope
            dl.add_text(
                font_slot, font,
                _truncate_text(crumb_txt, max(12, Int((w - Float32(140.0)) / Float32(7.0)))),
                x + Float32(10.0), crumb_y + Float32(16.0),
                Float32(11.0), theme.text_secondary,
            )
            var loc_txt = "Ln " + String(tab.cursor_line + 1) + "  (" + String(len(tab.symbols)) + " syms)"
            dl.add_text(
                font_slot, font, loc_txt,
                x + w - Float32(130.0), crumb_y + Float32(16.0),
                Float32(11.0), theme.text_muted,
            )
        else:
            dl.add_text(
                font_slot, font, "No file open - click a component or diagnostic to inspect source",
                x + Float32(10.0), crumb_y + Float32(16.0),
                Float32(11.0), theme.text_muted,
            )

        # --------------------------------------------------------------------
        # 3. Optional Agent Proposal Banner ([Accept All] / [Reject All])
        # --------------------------------------------------------------------
        var body_y = crumb_y + crumb_h
        if has_active_prop:
            dl.add_rect(x, body_y, w, prop_bar_h, theme.agent_soft)
            dl.add_rect(x, body_y + prop_bar_h - Float32(1.0), w, Float32(1.0), rgba8_with_alpha(theme.agent_active, 90))
            draw_studio_icon(dl, ICON_SPARK, x + Float32(10.0), body_y + Float32(7.0), Float32(14.0), theme.agent_active)
            var pending_hunks = self.tabs[self.active_tab].diff_doc.pending_count()
            var rej_w = Float32(76.0)
            var acc_w = Float32(78.0)
            var rej_x = x + w - rej_w - Float32(8.0)
            var acc_x = rej_x - acc_w - Float32(6.0)
            var prop_label = "Agent Proposed Edit (" + String(pending_hunks) + " hunks)"
            var prop_max_ch = max(8, Int((acc_x - (x + Float32(34.0))) / Float32(7.0)))
            dl.add_text(
                font_slot, font,
                _truncate_text(prop_label, prop_max_ch),
                x + Float32(28.0), body_y + Float32(18.0),
                Float32(11.5), theme.text_primary,
            )
            var by_btn = body_y + Float32(4.0)
            var bh_btn = Float32(20.0)
            var acc_hov = _hit_rect(mx, my, acc_x, by_btn, acc_w, bh_btn)
            var rej_hov = _hit_rect(mx, my, rej_x, by_btn, rej_w, bh_btn)
            self.last_hit_count += 2

            dl.add_rect(
                acc_x, by_btn, acc_w, bh_btn,
                rgba8_with_alpha(theme.status_ok, 70 if acc_hov else 46),
                theme.radius_xs, Float32(1.0), theme.status_ok,
            )
            dl.add_text(font_slot, font, "Accept All", acc_x + Float32(9.0), by_btn + Float32(14.0), Float32(11.0), theme.text_primary)

            dl.add_rect(
                rej_x, by_btn, rej_w, bh_btn,
                rgba8_with_alpha(theme.status_danger, 70 if rej_hov else 44),
                theme.radius_xs, Float32(1.0), theme.status_danger,
            )
            dl.add_text(font_slot, font, "Reject All", rej_x + Float32(10.0), by_btn + Float32(14.0), Float32(11.0), theme.text_primary)

            if click and acc_hov and action.kind == CODE_ACT_NONE:
                action = CodeDrawerAction(CODE_ACT_ACCEPT_ALL_DIFF)
                action.path = self.tabs[self.active_tab].path
                self.apply_all_diff(True)
            elif click and rej_hov and action.kind == CODE_ACT_NONE:
                action = CodeDrawerAction(CODE_ACT_REJECT_ALL_DIFF)
                action.path = self.tabs[self.active_tab].path
                self.apply_all_diff(False)

            body_y += prop_bar_h

        # --------------------------------------------------------------------
        # 4. Main Code or Diff Canvas
        # --------------------------------------------------------------------
        var prob_y = y + h - prob_total_h
        var body_h = max(Float32(0.0), prob_y - body_y)
        var line_h = Float32(18.0)
        var mono_px = theme.font_mono
        var visible_rows = max(1, Int(body_h / line_h))

        dl.push_clip(Int(x), Int(body_y), Int(x + w), Int(body_y + body_h))
        dl.add_rect(x, body_y, w, body_h, theme.l0_canvas)

        if self.active_tab >= 0 and self.active_tab < len(self.tabs) and body_h > Float32(16.0):
            # Scroll & keyboard navigation when mouse/focus in body
            if _hit_rect(mx, my, x, body_y, w, body_h) and inp.wheel != Float32(0.0):
                var delta = -3 if inp.wheel > Float32(0.0) else 3
                self.tabs[self.active_tab].scroll_line = max(
                    0, self.tabs[self.active_tab].scroll_line + delta
                )
            if self.has_keyboard_focus:
                if inp.key_pressed(265):  # Up
                    self.tabs[self.active_tab].cursor_line = max(0, self.tabs[self.active_tab].cursor_line - 1)
                elif inp.key_pressed(264):  # Down
                    self.tabs[self.active_tab].cursor_line = min(
                        max(0, len(self.tabs[self.active_tab].lines) - 1),
                        self.tabs[self.active_tab].cursor_line + 1,
                    )

            if self.view_mode == CODE_MODE_SOURCE:
                self._draw_source_canvas(
                    dl, font_slot, font, theme, x, body_y, w, body_h, line_h, mono_px, visible_rows, mx, my, click
                )
            else:
                var diff_act = self._draw_diff_canvas(
                    dl, font_slot, font, theme, x, body_y, w, body_h, line_h, mono_px, visible_rows, mx, my, click
                )
                if diff_act.kind != CODE_ACT_NONE and action.kind == CODE_ACT_NONE:
                    action = diff_act^

        dl.pop_clip()

        # --------------------------------------------------------------------
        # 5. Collapsible Problems Strip
        # --------------------------------------------------------------------
        dl.add_rect(x, prob_y, w, prob_total_h, theme.l1_shell)
        dl.add_rect(x, prob_y, w, Float32(1.0), theme.border_strong)

        var phdr_hov = _hit_rect(mx, my, x, prob_y, w, prob_hdr_h)
        self.last_hit_count += 1
        dl.add_rect(x, prob_y + Float32(1.0), w, prob_hdr_h - Float32(1.0), theme.l2_card if phdr_hov else theme.l1_shell)
        draw_studio_chevron(
            dl, x + Float32(8.0), prob_y + Float32(6.0), Float32(12.0), theme.text_secondary,
            Float32(1.0) if self.problems_open else Float32(0.0),
        )
        var errs = self.problems.error_count()
        var warns = self.problems.warning_count()
        dl.add_text(
            font_slot, font,
            "Problems   E:" + String(errs) + "  W:" + String(warns),
            x + Float32(26.0), prob_y + Float32(16.0),
            Float32(11.5), theme.status_danger if errs > 0 else theme.text_secondary,
        )
        if click and phdr_hov and action.kind == CODE_ACT_NONE:
            self.problems_open = not self.problems_open
            action = CodeDrawerAction(CODE_ACT_TOGGLE_PROBLEMS)

        if self.problems_open and prob_total_h > prob_hdr_h + Float32(4.0):
            var list_y = prob_y + prob_hdr_h
            var list_h = prob_total_h - prob_hdr_h
            dl.push_clip(Int(x), Int(list_y), Int(x + w), Int(list_y + list_h))
            var cur_path = self.tabs[self.active_tab].path if (self.active_tab >= 0 and self.active_tab < len(self.tabs)) else String("")
            var diags = self.problems.query(ProblemFilter(), cur_path)
            var row_h = Float32(20.0)
            var max_rows = max(1, Int(list_h / row_h))
            if len(diags) == 0:
                dl.add_text(
                    font_slot, font, "No compiler or LSP diagnostics.",
                    x + Float32(14.0), list_y + Float32(15.0),
                    Float32(11.0), theme.text_muted,
                )
            else:
                for di in range(min(max_rows, len(diags))):
                    ref d = diags[di]
                    var ry = list_y + Float32(di) * row_h
                    var rhov = _hit_rect(mx, my, x, ry, w, row_h)
                    var ask_w = Float32(72.0)
                    var ask_x = x + w - ask_w - Float32(8.0)
                    var ask_hov = _hit_rect(mx, my, ask_x, ry + Float32(2.0), ask_w, row_h - Float32(4.0))
                    self.last_hit_count += 2
                    if rhov:
                        dl.add_rect(x, ry, w, row_h, theme.l3_raised)
                    var sev_col = (
                        theme.status_danger
                        if d.severity == DIAG_ERROR
                        else (theme.status_warn if d.severity == DIAG_WARNING else theme.accent_primary)
                    )
                    dl.add_circle(x + Float32(12.0), ry + row_h * Float32(0.5), Float32(3.5), sev_col)
                    var loc = _basename(d.path) + ":" + String(d.line) + ":" + String(d.col) + "  " + d.message
                    var max_c = max(10, Int((ask_x - (x + Float32(26.0))) / Float32(6.8)))
                    dl.add_text(
                        font_slot, font, _truncate_text(loc, max_c),
                        x + Float32(24.0), ry + Float32(14.0),
                        Float32(11.0), theme.text_primary if d.severity == DIAG_ERROR else theme.text_secondary,
                    )
                    dl.add_rect(
                        ask_x, ry + Float32(2.0), ask_w, row_h - Float32(4.0),
                        theme.agent_soft if ask_hov else theme.l2_card,
                        theme.radius_xs, Float32(1.0),
                        theme.agent_active if ask_hov else theme.border_subtle,
                    )
                    dl.add_text(
                        font_slot, font, "Ask Agent",
                        ask_x + Float32(9.0), ry + Float32(13.5),
                        Float32(10.5), theme.agent_active if ask_hov else theme.text_secondary,
                    )
                    if click and ask_hov and action.kind == CODE_ACT_NONE:
                        action = CodeDrawerAction(CODE_ACT_ASK_AGENT_FIX)
                        action.path = d.path
                        action.line = d.line
                        action.col = d.col
                        action.message = d.message
                    elif click and rhov and action.kind == CODE_ACT_NONE:
                        action = CodeDrawerAction(CODE_ACT_JUMP_DIAGNOSTIC)
                        action.path = d.path
                        action.line = d.line
                        action.col = d.col
                        action.message = d.message
                        if d.path == cur_path:
                            self.jump_to_line(d.line)
            dl.pop_clip()

        dl.pop_clip()
        return action^

    def _draw_source_canvas(
        mut self,
        mut dl: UiDrawList,
        font_slot: Int,
        font: SdfFont,
        theme: StudioTheme,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        line_h: Float32,
        mono_px: Float32,
        visible_rows: Int,
        mx: Float32,
        my: Float32,
        click: Bool,
    ):
        ref tab = self.tabs[self.active_tab]
        var gutter_w = Float32(48.0)
        var marks = tab.diff_doc.gutter_marks()
        var file_diags = self.problems.for_file(tab.path)
        var n_lines = len(tab.lines)
        var top = max(0, min(max(0, n_lines - 1), tab.scroll_line))
        self.tabs[self.active_tab].scroll_line = top

        dl.add_rect(x, y, gutter_w, h, theme.l1_shell)
        dl.add_rect(x + gutter_w, y, Float32(1.0), h, theme.border_subtle)

        for vi in range(visible_rows):
            var ln = top + vi
            if ln >= n_lines:
                break
            var ry = y + Float32(vi) * line_h
            var is_cur = ln == tab.cursor_line
            if is_cur:
                dl.add_rect(x + gutter_w + Float32(1.0), ry, w - gutter_w - Float32(1.0), line_h, rgba8_with_alpha(theme.accent_primary, 24))

            if click and _hit_rect(mx, my, x, ry, w, line_h):
                self.tabs[self.active_tab].cursor_line = ln

            # Diff gutter mark
            if ln < len(marks):
                var mk = Int(marks[ln])
                if mk == GUTTER_ADDED:
                    dl.add_rect(x + gutter_w - Float32(3.0), ry + Float32(1.0), Float32(3.0), line_h - Float32(2.0), theme.status_ok)
                elif mk == GUTTER_CHANGED:
                    dl.add_rect(x + gutter_w - Float32(3.0), ry + Float32(1.0), Float32(3.0), line_h - Float32(2.0), theme.accent_primary)
                elif mk == GUTTER_DELETED_ABOVE or mk == GUTTER_DELETED_BELOW:
                    dl.add_rect(x + gutter_w - Float32(4.0), ry, Float32(4.0), Float32(3.0), theme.status_danger)

            # Line number
            dl.add_text(
                font_slot, font, String(ln + 1),
                x + Float32(8.0), ry + Float32(13.5),
                Float32(11.0), theme.text_primary if is_cur else theme.text_muted,
            )

            # Syntax-highlighted code line
            var code_x = x + gutter_w + Float32(10.0)
            var base_y = ry + Float32(13.5)
            if ln < len(tab.spans):
                self._draw_syntax_text(dl, font_slot, font, theme, tab.lines[ln], tab.spans[ln], code_x, base_y, mono_px)
            else:
                dl.add_text(font_slot, font, tab.lines[ln], code_x, base_y, mono_px, theme.text_primary)

            # Inline diagnostic squiggle & badge for diagnostics on this line
            for di in range(len(file_diags)):
                ref d = file_diags[di]
                if d.line == ln + 1:
                    var dcol = theme.status_danger if d.severity == DIAG_ERROR else theme.status_warn
                    var char_w = max(Float32(6.5), font.measure("m", mono_px))
                    var sq_x = code_x + Float32(max(0, d.col - 1)) * char_w
                    var sq_w = max(Float32(24.0), char_w * Float32(6.0))
                    self._draw_squiggle(dl, sq_x, ry + line_h - Float32(2.0), sq_w, dcol)
                    var badge = _truncate_text(d.message, 32)
                    var bx = max(code_x + font.measure(tab.lines[ln], mono_px) + Float32(16.0), x + w - Float32(220.0))
                    if bx + Float32(60.0) < x + w:
                        dl.add_rect(bx, ry + Float32(2.0), x + w - bx - Float32(6.0), line_h - Float32(4.0), rgba8_with_alpha(dcol, 34), theme.radius_xs)
                        dl.add_text(font_slot, font, badge, bx + Float32(6.0), ry + Float32(13.0), Float32(10.5), dcol)
                    break

    def _draw_diff_canvas(
        mut self,
        mut dl: UiDrawList,
        font_slot: Int,
        font: SdfFont,
        theme: StudioTheme,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        line_h: Float32,
        mono_px: Float32,
        visible_rows: Int,
        mx: Float32,
        my: Float32,
        click: Bool,
    ) -> CodeDrawerAction:
        var act = CodeDrawerAction(CODE_ACT_NONE)
        var is_split = self.view_mode == CODE_MODE_DIFF_SPLIT
        var rows = (
            self.tabs[self.active_tab].diff_doc.split_rows(3)
            if is_split
            else self.tabs[self.active_tab].diff_doc.unified_rows(3)
        )
        if len(rows) == 0:
            dl.add_text(
                font_slot, font, "No changes (file matches base).",
                x + Float32(16.0), y + Float32(24.0),
                Float32(12.0), theme.text_muted,
            )
            return act^

        var top = max(0, min(max(0, len(rows) - 1), self.tabs[self.active_tab].scroll_line))
        self.tabs[self.active_tab].scroll_line = top

        var old_bg = theme.diff_del_bg
        var new_bg = theme.diff_add_bg
        var hdr_bg = theme.diff_hunk_bg
        var half_w = w * Float32(0.5)

        if is_split:
            dl.add_rect(x + half_w, y, Float32(1.0), h, theme.border_subtle)

        for vi in range(visible_rows):
            var ri = top + vi
            if ri >= len(rows):
                break
            ref row = rows[ri]
            var ry = y + Float32(vi) * line_h
            var base_y = ry + Float32(13.5)

            if row.kind == DIFF_ROW_HEADER:
                dl.add_rect(x, ry, w, line_h, hdr_bg)
                draw_studio_icon(dl, ICON_DIFF, x + Float32(8.0), ry + Float32(3.0), Float32(12.0), theme.accent_primary)
                dl.add_text(font_slot, font, row.text, x + Float32(26.0), base_y, Float32(11.0), theme.accent_primary)

                # Clickable per-hunk [Accept] / [Reject] buttons
                var hi = row.hunk
                if hi >= 0 and hi < len(self.tabs[self.active_tab].diff_doc.decisions):
                    var dec = self.tabs[self.active_tab].diff_doc.decisions[hi]
                    var bw = Float32(58.0)
                    var rbx = x + w - bw - Float32(8.0)
                    var abx = rbx - bw - Float32(6.0)
                    var ahov = _hit_rect(mx, my, abx, ry + Float32(1.0), bw, line_h - Float32(2.0))
                    var rhov = _hit_rect(mx, my, rbx, ry + Float32(1.0), bw, line_h - Float32(2.0))
                    self.last_hit_count += 2

                    var acc_fill = (
                        rgba8_with_alpha(theme.status_ok, 92)
                        if dec == HUNK_ACCEPTED
                        else rgba8_with_alpha(theme.status_ok, 60 if ahov else 32)
                    )
                    var rej_fill = (
                        rgba8_with_alpha(theme.status_danger, 92)
                        if dec == HUNK_REJECTED
                        else rgba8_with_alpha(theme.status_danger, 60 if rhov else 32)
                    )
                    dl.add_rect(abx, ry + Float32(1.0), bw, line_h - Float32(2.0), acc_fill, theme.radius_xs, Float32(1.0), theme.status_ok)
                    dl.add_text(font_slot, font, "Accept", abx + Float32(9.0), ry + Float32(13.0), Float32(10.5), theme.text_primary)
                    dl.add_rect(rbx, ry + Float32(1.0), bw, line_h - Float32(2.0), rej_fill, theme.radius_xs, Float32(1.0), theme.status_danger)
                    dl.add_text(font_slot, font, "Reject", rbx + Float32(10.0), ry + Float32(13.0), Float32(10.5), theme.text_primary)

                    if click and ahov and act.kind == CODE_ACT_NONE:
                        act = CodeDrawerAction(CODE_ACT_ACCEPT_HUNK)
                        act.hunk_index = hi
                        act.path = self.tabs[self.active_tab].path
                        _ = self.apply_hunk_decision(hi, True)
                    elif click and rhov and act.kind == CODE_ACT_NONE:
                        act = CodeDrawerAction(CODE_ACT_REJECT_HUNK)
                        act.hunk_index = hi
                        act.path = self.tabs[self.active_tab].path
                        _ = self.apply_hunk_decision(hi, False)
                continue

            if not is_split:
                var bg = UInt32(0)
                var bar_col = UInt32(0)
                if row.kind == DIFF_ROW_OLD:
                    bg = old_bg
                    bar_col = theme.status_danger
                elif row.kind == DIFF_ROW_NEW:
                    bg = new_bg
                    bar_col = theme.status_ok
                if (bg >> 24) != 0:
                    dl.add_rect(x, ry, w, line_h, bg)
                    dl.add_rect(x, ry, Float32(3.0), line_h, bar_col)
                var ln_str = (
                    String(row.new_line + 1)
                    if row.new_line >= 0 and row.kind != DIFF_ROW_OLD
                    else (String(row.old_line + 1) if row.old_line >= 0 else String(""))
                )
                dl.add_text(font_slot, font, ln_str, x + Float32(8.0), base_y, Float32(10.5), theme.text_muted)
                var prefix = String(row.text[byte=0:1]) if row.text.byte_length() > 0 else String(" ")
                var body = String(row.text[byte=1 : row.text.byte_length()]) if row.text.byte_length() > 1 else String("")
                var pcol = (
                    theme.status_danger
                    if row.kind == DIFF_ROW_OLD
                    else (theme.status_ok if row.kind == DIFF_ROW_NEW else theme.text_muted)
                )
                dl.add_text(font_slot, font, prefix, x + Float32(42.0), base_y, mono_px, pcol)
                var hl_u = self.highlighter.highlight_line(body, 0, self.tabs[self.active_tab].lang)
                self._draw_syntax_text(dl, font_slot, font, theme, body, hl_u[0], x + Float32(56.0), base_y, mono_px)
            else:
                # Split side-by-side rendering
                var left_bg = old_bg if (row.kind == DIFF_ROW_OLD or row.kind == DIFF_ROW_SPLIT_CHG) else UInt32(0)
                var right_bg = new_bg if (row.kind == DIFF_ROW_NEW or row.kind == DIFF_ROW_SPLIT_CHG) else UInt32(0)
                if (left_bg >> 24) != 0:
                    dl.add_rect(x, ry, half_w, line_h, left_bg)
                    dl.add_rect(x, ry, Float32(3.0), line_h, theme.status_danger)
                if (right_bg >> 24) != 0:
                    dl.add_rect(x + half_w + Float32(1.0), ry, half_w - Float32(1.0), line_h, right_bg)
                    dl.add_rect(x + half_w + Float32(1.0), ry, Float32(3.0), line_h, theme.status_ok)

                var max_col_chars = max(8, Int((half_w - Float32(44.0)) / Float32(7.0)))
                if row.old_line >= 0:
                    dl.add_text(font_slot, font, String(row.old_line + 1), x + Float32(6.0), base_y, Float32(10.5), theme.text_muted)
                if row.left_text.byte_length() > 0:
                    var lt = _truncate_text(row.left_text, max_col_chars)
                    var hl_l = self.highlighter.highlight_line(lt, 0, self.tabs[self.active_tab].lang)
                    self._draw_syntax_text(dl, font_slot, font, theme, lt, hl_l[0], x + Float32(38.0), base_y, mono_px)

                if row.new_line >= 0:
                    dl.add_text(font_slot, font, String(row.new_line + 1), x + half_w + Float32(8.0), base_y, Float32(10.5), theme.text_muted)
                if row.right_text.byte_length() > 0:
                    var rt = _truncate_text(row.right_text, max_col_chars)
                    var hl_r = self.highlighter.highlight_line(rt, 0, self.tabs[self.active_tab].lang)
                    self._draw_syntax_text(dl, font_slot, font, theme, rt, hl_r[0], x + half_w + Float32(40.0), base_y, mono_px)

        return act^
