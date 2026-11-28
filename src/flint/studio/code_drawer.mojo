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
from flint.platform.os import file_stamp, get_time, is_symlink, read_bytes, real_path, write_file_durable
from flint.render.font import SdfFont
from flint.ui.draw_list import UI_FLAG_BORDER, UI_FLAG_TOP_BEVEL, UiDrawList
from flint.ui.icons import draw_studio_chevron, draw_studio_icon
from flint.ui.theme import (
    ICON_CHECK,
    ICON_CLOSE,
    ICON_CODE,
    ICON_DIFF,
    ICON_ERROR_CIRCLE,
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
comptime CODE_ACT_SAVE_TAB: Int = 12
"""A save was attempted (Ctrl+S or the unsaved-close banner); `message` is "" on success."""
comptime CODE_ACT_RELOAD_FROM_DISK: Int = 13
comptime CODE_ACT_KEEP_BUFFER: Int = 14

comptime CODE_TAB_WIDTH: Int = 4
"""Columns per tab stop when drawing `\\t` and placing diagnostic squiggles."""
comptime CODE_DISK_POLL_SEC: Float64 = 0.5


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


def text_hash64(text: String) -> UInt64:
    """FNV-1a 64 over the UTF-8 bytes (detects on-disk content changes)."""
    var h: UInt64 = 0xCBF29CE484222325
    var b = text.as_bytes()
    for i in range(len(b)):
        h = (h ^ UInt64(b[i])) * 0x100000001B3
    return h


def _bytes_to_text(raw: List[UInt8]) -> String:
    return String(unsafe_from_utf8=raw) if len(raw) > 0 else String("")


def byte_offset_for_col(line: String, col_1based: Int) -> Int:
    """Byte offset of 1-based compiler column `col_1based` (a byte column), clamped to the
    line and moved back onto a UTF-8 boundary."""
    var b = line.as_bytes()
    var off = max(0, min(len(b), col_1based - 1))
    while off > 0 and off < len(b) and (b[off] & 0xC0) == 0x80:
        off -= 1
    return off


@always_inline
def _hit_rect(mx: Float32, my: Float32, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
    return mx >= x and mx < x + w and my >= y and my < y + h


def _truncate_text(text: String, max_chars: Int) -> String:
    if max_chars <= 4:
        return ""
    if text.byte_length() <= max_chars:
        return text
    var b = text.as_bytes()
    var cut = min(len(b), max_chars - 3)
    while cut > 0 and (b[cut] & 0xC0) == 0x80:
        cut -= 1
    return String(text[byte=0:cut]) + "..."


def _measure_cd(font: SdfFont, text: String, px: Float32) -> Float32:
    if font.valid:
        return font.measure(text, px)
    return Float32(text.byte_length()) * px * Float32(0.56)


def _draw_cd_mid(
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


def _tab_stop_px(font: SdfFont, font_px: Float32) -> Float32:
    return max(Float32(1.0), font.measure(" ", font_px) * Float32(CODE_TAB_WIDTH))


def advance_text(font: SdfFont, font_px: Float32, x0: Float32, pen: Float32, seg: String) -> Float32:
    """Pen position after laying out `seg` starting at `pen`, where tabs snap to the next
    `CODE_TAB_WIDTH` stop measured from the line origin `x0` (matches drawing)."""
    var b = seg.as_bytes()
    var p = pen
    var start = 0
    for i in range(len(b)):
        if b[i] == 9:
            if i > start:
                p += font.measure(String(seg[byte=start:i]), font_px)
            var stop = _tab_stop_px(font, font_px)
            p = x0 + (Float32(Int((p - x0) / stop + Float32(1.0e-3))) + Float32(1.0)) * stop
            start = i + 1
    if start < len(b):
        p += font.measure(String(seg[byte=start : len(b)]), font_px)
    return p


def diag_squiggle_span(
    font: SdfFont, font_px: Float32, line: String, col_1based: Int, end_col_1based: Int = 0
) -> Tuple[Float32, Float32]:
    """(x offset from the code origin, width) of a diagnostic squiggle: the column is mapped to
    a byte offset and the tab-expanded prefix is measured; the squiggle covers `end_col` when
    given, else the identifier/word at the column (at least one glyph)."""
    var b = line.as_bytes()
    var start = byte_offset_for_col(line, col_1based)
    var end = start
    if end_col_1based > col_1based:
        end = max(start, byte_offset_for_col(line, end_col_1based))
    else:
        while end < len(b) and (
            (b[end] >= 48 and b[end] <= 57)
            or (b[end] >= 65 and b[end] <= 90)
            or (b[end] >= 97 and b[end] <= 122)
            or b[end] == 95
            or b[end] >= 128
        ):
            end += 1
        if end == start and end < len(b):
            end += 1
            while end < len(b) and (b[end] & 0xC0) == 0x80:
                end += 1
    var x0 = advance_text(font, font_px, Float32(0.0), Float32(0.0), String(line[byte=0:start]))
    var x1 = advance_text(font, font_px, Float32(0.0), x0, String(line[byte=start:end]))
    var glyph = max(Float32(6.0), font.measure("m", font_px))
    return (x0, max(glyph, x1 - x0))


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
    """Open file buffer inside `CodeDrawer` with cached highlight spans, outline & diff.

    Disk-backed tabs (`disk_path` set) remember the size, mtime and content hash of the
    version last read or saved; `CodeDrawer.poll_disk_changes` reloads clean tabs when the
    file changes and flags `disk_conflict` (keeping the buffer) when the tab is dirty.
    """
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
    var disk_path: String
    var disk_size: Int
    var disk_mtime: Int
    var disk_hash: UInt64
    var disk_conflict: Bool
    """The file changed on disk while this tab had unsaved edits (or a pending proposal)."""
    var disk_missing: Bool
    var disk_pending_text: String
    """The newer on-disk (or newly supplied) text a conflict is about."""
    var diff_old_spans: List[List[HighlightSpan]]
    var diff_new_spans: List[List[HighlightSpan]]
    var diff_spans_valid: Bool

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
        self.disk_path = ""
        self.disk_size = -1
        self.disk_mtime = 0
        self.disk_hash = 0
        self.disk_conflict = False
        self.disk_missing = False
        self.disk_pending_text = ""
        self.diff_old_spans = List[List[HighlightSpan]]()
        self.diff_new_spans = List[List[HighlightSpan]]()
        self.diff_spans_valid = False

    def rebuild_caches(mut self, hl: MojoHighlighter):
        self.lines = _split_lines(self.text)
        self.spans = hl.highlight_lines(self.lines, self.lang)
        if self.lang == LANG_MOJO:
            self.symbols = index_mojo_source(self.text, self.path)
        else:
            self.symbols.clear()
        if not self.has_proposal:
            self.diff_doc = DiffDocument(self.path, self.base_text, self.text)
        self.diff_spans_valid = False

    def ensure_diff_spans(mut self, hl: MojoHighlighter):
        """Highlights both diff sides as whole files so multi-line strings, comments and
        Markdown fences keep their state across diff rows."""
        if self.diff_spans_valid:
            return
        self.diff_old_spans = hl.highlight_lines(self.diff_doc.old_side.lines, self.lang)
        self.diff_new_spans = hl.highlight_lines(self.diff_doc.new_side.lines, self.lang)
        self.diff_spans_valid = True


struct CodeDrawer(Movable):
    """Native GPU SDF Code, Diff & Problems drawer for Flint Studio.

    Public buffer API (also used by the MCP agent): `open_file`, `open_source`,
    `materialized_text`, `tab_text`, `save_tab` / `save_path`, `close_tab`,
    `poll_disk_changes`, `reload_tab_from_disk`, `keep_tab_buffer`, `mirror_problem_store`.
    """
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
    var visible_rows: Int
    """Code rows visible in the last drawn body (used by `jump_to_line`)."""
    var pending_close_tab: Int
    """Tab whose close is waiting on the unsaved-changes banner (-1: none)."""
    var last_error: String
    var last_disk_poll: Float64
    var mirrored_source_hash: List[UInt64]
    """Per-`DIAG_SRC_*` fingerprint of the last mirrored external `ProblemStore`."""

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
        self.visible_rows = 12
        self.pending_close_tab = -1
        self.last_error = ""
        self.last_disk_poll = -1.0e9
        self.mirrored_source_hash = List[UInt64](length=4, fill=UInt64(0))

    def find_tab(self, path: String) -> Int:
        if path.byte_length() == 0:
            return -1
        for i in range(len(self.tabs)):
            if self.tabs[i].path == path:
                return i
        for i in range(len(self.tabs)):
            if self.tabs[i].disk_path == path:
                return i
        return -1

    def _valid(self, idx: Int) -> Bool:
        return idx >= 0 and idx < len(self.tabs)

    def _resolve_idx(self, idx: Int) -> Int:
        return self.active_tab if idx < 0 else idx

    def open_source(
        mut self,
        path: String,
        text: String,
        base_text: String = "",
        dirty: Bool = False,
        focus_line_1based: Int = 1,
        force: Bool = False,
    ) -> Int:
        """Opens or updates a tab from in-memory source text and focuses `focus_line_1based`.

        Re-opening a tab with unsaved edits never overwrites them: unless `force`, a different
        `text` is parked in `disk_pending_text` and the tab is flagged `disk_conflict`.
        """
        var idx = self.find_tab(path)
        if idx < 0:
            var tab = CodeTab(path, text, base_text)
            tab.dirty = dirty
            tab.rebuild_caches(self.highlighter)
            self.tabs.append(tab^)
            idx = len(self.tabs) - 1
        elif self.tabs[idx].text == text and not force:
            pass  # Same buffer: keep its base text, dirty flag and caches.
        elif self.tabs[idx].dirty and not force:
            self.tabs[idx].disk_conflict = True
            self.tabs[idx].disk_pending_text = text
        else:
            self.tabs[idx].text = text
            if base_text.byte_length() > 0:
                self.tabs[idx].base_text = base_text
            self.tabs[idx].dirty = dirty
            self.tabs[idx].disk_conflict = False
            self.tabs[idx].rebuild_caches(self.highlighter)
        if self.tabs[idx].lang == LANG_MOJO:
            self.symbols.index_source(path, self.tabs[idx].text)
        self.active_tab = idx
        self.jump_to_line(focus_line_1based)
        return idx

    def _record_disk(mut self, idx: Int, disk_path: String, text: String, size: Int, mtime: Int):
        self.tabs[idx].disk_path = disk_path
        self.tabs[idx].disk_size = size
        self.tabs[idx].disk_mtime = mtime
        self.tabs[idx].disk_hash = text_hash64(text)
        self.tabs[idx].disk_missing = False

    def open_file(mut self, path: String, focus_line_1based: Int = 1) -> Int:
        """Loads a UTF-8 file from disk into a tab and focuses `focus_line_1based`.

        An already-open tab is focused as-is (never re-read over unsaved edits). Returns -1
        and sets `last_error` when the file cannot be read.
        """
        var idx = self.find_tab(path)
        if idx >= 0 and self.tabs[idx].disk_path.byte_length() > 0:
            self.active_tab = idx
            self.jump_to_line(focus_line_1based)
            return idx
        var st = file_stamp(path)
        if st[0] < 0:
            self.last_error = "cannot read " + path
            return -1
        var text = _bytes_to_text(read_bytes(path))
        if st[0] > 0 and text.byte_length() == 0:
            self.last_error = "cannot read " + path
            return -1
        if idx >= 0:
            # A tab opened from memory under this path becomes disk-backed.
            var same = self.tabs[idx].text == text
            self._record_disk(idx, path, text, st[0], st[1])
            if not same and self.tabs[idx].dirty:
                self.tabs[idx].disk_conflict = True
                self.tabs[idx].disk_pending_text = text
            elif not same:
                self.tabs[idx].text = text
                self.tabs[idx].base_text = text
                self.tabs[idx].rebuild_caches(self.highlighter)
            self.active_tab = idx
            self.jump_to_line(focus_line_1based)
            return idx
        idx = self.open_source(path, text, text, False, focus_line_1based)
        self._record_disk(idx, path, text, st[0], st[1])
        return idx

    def set_tab_disk_path(mut self, idx: Int, disk_path: String):
        """Binds a tab opened from memory (e.g. a workspace-relative agent proposal) to the
        file it should be saved to and watched at."""
        if not self._valid(idx):
            return
        self.tabs[idx].disk_path = disk_path
        var st = file_stamp(disk_path)
        self.tabs[idx].disk_size = st[0]
        self.tabs[idx].disk_mtime = st[1]
        self.tabs[idx].disk_missing = st[0] < 0
        if st[0] >= 0:
            self.tabs[idx].disk_hash = text_hash64(_bytes_to_text(read_bytes(disk_path)))

    def materialized_text(self, idx: Int = -1, accept_pending: Bool = False) -> String:
        """Text the tab would hold after its diff review: pending proposal hunks resolved
        with the current decisions (`accept_pending` takes undecided hunks), else the buffer."""
        var i = self._resolve_idx(idx)
        if not self._valid(i):
            return String("")
        if self.tabs[i].has_proposal:
            return self.tabs[i].diff_doc.materialized_text(accept_pending)
        return self.tabs[i].text

    def tab_text(self, path: String) -> String:
        var i = self.find_tab(path)
        return self.tabs[i].text if i >= 0 else String("")

    def _save_target(self, idx: Int) -> String:
        if self.tabs[idx].disk_path.byte_length() > 0:
            return self.tabs[idx].disk_path
        if self.tabs[idx].path.startswith("/"):
            return self.tabs[idx].path
        return String("")

    def save_tab(mut self, idx: Int = -1, force: Bool = False) -> Bool:
        """Durably writes the tab buffer to its file (`write_file_durable`: temp + fsync +
        rename). Refuses (unless `force`) while the file changed on disk under unsaved edits."""
        var i = self._resolve_idx(idx)
        if not self._valid(i):
            self.last_error = "no tab to save"
            return False
        var target = self._save_target(i)
        if target.byte_length() == 0:
            self.last_error = "no file path for " + self.tabs[i].title
            return False
        if self.tabs[i].disk_conflict and not force:
            self.last_error = self.tabs[i].title + " changed on disk: reload or keep your version first"
            return False
        var write_to = target
        if is_symlink(target):
            var rp = real_path(target)
            if rp[0]:
                write_to = rp[1]
        var data = List[UInt8](self.tabs[i].text.as_bytes())
        if not write_file_durable(write_to, data):
            self.last_error = "could not write " + target
            return False
        var st = file_stamp(target)
        var saved_text = self.tabs[i].text
        self._record_disk(i, target, saved_text, st[0], st[1])
        self.tabs[i].base_text = self.tabs[i].text
        self.tabs[i].dirty = False
        self.tabs[i].disk_conflict = False
        self.tabs[i].disk_pending_text = ""
        self.tabs[i].rebuild_caches(self.highlighter)
        self.last_error = ""
        return True

    def save_path(mut self, path: String, force: Bool = False) -> Bool:
        var i = self.find_tab(path)
        if i < 0:
            self.last_error = "no open tab for " + path
            return False
        return self.save_tab(i, force)

    def reload_tab_from_disk(mut self, idx: Int = -1) -> Bool:
        """Discards the buffer and loads the current file (or the parked conflicting text)."""
        var i = self._resolve_idx(idx)
        if not self._valid(i):
            return False
        var text: String
        if self.tabs[i].disk_path.byte_length() > 0:
            var st = file_stamp(self.tabs[i].disk_path)
            if st[0] < 0:
                self.last_error = "cannot read " + self.tabs[i].disk_path
                return False
            text = _bytes_to_text(read_bytes(self.tabs[i].disk_path))
            var dp = self.tabs[i].disk_path
            self._record_disk(i, dp, text, st[0], st[1])
        elif self.tabs[i].disk_conflict:
            text = self.tabs[i].disk_pending_text
        else:
            return False
        self.tabs[i].text = text
        self.tabs[i].base_text = text
        self.tabs[i].dirty = False
        self.tabs[i].has_proposal = False
        self.tabs[i].disk_conflict = False
        self.tabs[i].disk_pending_text = ""
        self.tabs[i].rebuild_caches(self.highlighter)
        self.tabs[i].cursor_line = min(self.tabs[i].cursor_line, max(0, len(self.tabs[i].lines) - 1))
        return True

    def keep_tab_buffer(mut self, idx: Int = -1):
        """Resolves a disk conflict in favour of the buffer: the next save overwrites the file."""
        var i = self._resolve_idx(idx)
        if not self._valid(i):
            return
        self.tabs[i].disk_conflict = False
        self.tabs[i].disk_pending_text = ""

    def poll_disk_changes(mut self, now_sec: Float64 = -1.0, force: Bool = False) -> Int:
        """Stats every disk-backed tab (at most every `CODE_DISK_POLL_SEC`); clean tabs reload a
        changed file, dirty / proposal tabs get `disk_conflict`. Returns tabs reloaded or flagged."""
        var now = now_sec if now_sec >= 0.0 else get_time()
        if not force and now - self.last_disk_poll < CODE_DISK_POLL_SEC:
            return 0
        self.last_disk_poll = now
        var changed = 0
        for i in range(len(self.tabs)):
            if self.tabs[i].disk_path.byte_length() == 0:
                continue
            var st = file_stamp(self.tabs[i].disk_path)
            if st[0] < 0:
                self.tabs[i].disk_missing = True
                continue
            self.tabs[i].disk_missing = False
            if st[0] == self.tabs[i].disk_size and st[1] == self.tabs[i].disk_mtime:
                continue
            var text = _bytes_to_text(read_bytes(self.tabs[i].disk_path))
            var h = text_hash64(text)
            self.tabs[i].disk_size = st[0]
            self.tabs[i].disk_mtime = st[1]
            if h == self.tabs[i].disk_hash:
                continue  # touched, same content
            self.tabs[i].disk_hash = h
            if text == self.tabs[i].text:
                self.tabs[i].base_text = text
                self.tabs[i].dirty = False
                self.tabs[i].disk_conflict = False
                self.tabs[i].rebuild_caches(self.highlighter)
            elif self.tabs[i].dirty or self.tabs[i].has_proposal:
                self.tabs[i].disk_conflict = True
                self.tabs[i].disk_pending_text = text
            else:
                self.tabs[i].text = text
                self.tabs[i].base_text = text
                self.tabs[i].rebuild_caches(self.highlighter)
                self.tabs[i].cursor_line = min(self.tabs[i].cursor_line, max(0, len(self.tabs[i].lines) - 1))
            changed += 1
        return changed

    def close_tab(mut self, idx: Int, force: Bool = False):
        """Closes a tab. A dirty tab is only closed with `force`; otherwise it becomes
        `pending_close_tab` and the drawer asks Save / Discard / Cancel."""
        if idx < 0 or idx >= len(self.tabs):
            return
        if self.tabs[idx].dirty and not force:
            self.pending_close_tab = idx
            self.active_tab = idx
            return
        _ = self.tabs.pop(idx)
        if self.pending_close_tab == idx:
            self.pending_close_tab = -1
        elif self.pending_close_tab > idx:
            self.pending_close_tab -= 1
        if len(self.tabs) == 0:
            self.active_tab = -1
        elif idx < self.active_tab:
            # A tab left of the active one closed: keep showing the same file.
            self.active_tab -= 1
        elif self.active_tab >= len(self.tabs):
            self.active_tab = len(self.tabs) - 1

    def mirror_problem_store(mut self, store: ProblemStore):
        """Mirrors another `ProblemStore` (e.g. the MCP server's `diagnostics_store`) into
        `problems`, replacing a source only when that source changed there."""
        for src in range(4):
            var h: UInt64 = 0xCBF29CE484222325
            var n = 0
            for k in range(len(store.items)):
                ref d = store.items[k]
                if d.source != src:
                    continue
                n += 1
                var key = d.path + ":" + String(d.line) + ":" + String(d.col) + ":" + String(d.severity) + ":" + d.message
                h = (h ^ text_hash64(key)) * 0x100000001B3
            h = (h ^ UInt64(n)) * 0x100000001B3
            if h == self.mirrored_source_hash[src]:
                continue
            self.mirrored_source_hash[src] = h
            var diags = List[Diagnostic]()
            for k in range(len(store.items)):
                if store.items[k].source == src:
                    diags.append(store.items[k].copy())
            self.problems.replace_source(src, diags)

    def _ensure_cursor_visible(mut self):
        if not self._valid(self.active_tab):
            return
        var rows = max(1, self.visible_rows)
        var ln = self.tabs[self.active_tab].cursor_line
        if ln < self.tabs[self.active_tab].scroll_line:
            self.tabs[self.active_tab].scroll_line = ln
        elif ln >= self.tabs[self.active_tab].scroll_line + rows:
            self.tabs[self.active_tab].scroll_line = ln - rows + 1

    def jump_to_line(mut self, line_1based: Int):
        if self.active_tab < 0 or self.active_tab >= len(self.tabs):
            return
        ref tab = self.tabs[self.active_tab]
        var max_ln = max(0, len(tab.lines) - 1)
        var ln = max(0, min(max_ln, line_1based - 1))
        self.tabs[self.active_tab].cursor_line = ln
        var rows = max(1, self.visible_rows)
        var top = self.tabs[self.active_tab].scroll_line
        if ln < top or ln >= top + rows:
            # Off screen: centre-ish (a third from the top), like most editors.
            self.tabs[self.active_tab].scroll_line = max(0, ln - rows // 3)

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
        self.tabs[idx].diff_spans_valid = False
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
            self.build_label = "Building .so..."
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
        """Draws `line` coloured by `spans` (byte ranges; spans past a truncated line are
        clipped and any uncovered tail is drawn plain). `\t` advances to the next tab stop."""
        var pen_x = x
        var n = line.byte_length()
        var drawn_to = 0
        for si in range(len(spans)):
            ref sp = spans[si]
            var s0 = max(sp.start_byte, drawn_to)
            var s1 = min(sp.end_byte, n)
            if s1 <= s0:
                continue
            if s0 > drawn_to:
                pen_x = self._draw_segment(dl, font_slot, font, String(line[byte=drawn_to:s0]), x, pen_x, baseline_y, font_px, theme.text_primary)
            var seg = String(line[byte=s0:s1])
            pen_x = self._draw_segment(dl, font_slot, font, seg, x, pen_x, baseline_y, font_px, self._syntax_color(theme, sp.token_kind))
            drawn_to = s1
        if drawn_to < n:
            _ = self._draw_segment(dl, font_slot, font, String(line[byte=drawn_to:n]), x, pen_x, baseline_y, font_px, theme.text_primary)

    def _draw_segment(
        self,
        mut dl: UiDrawList,
        font_slot: Int,
        font: SdfFont,
        seg: String,
        x0: Float32,
        pen: Float32,
        baseline_y: Float32,
        font_px: Float32,
        color: UInt32,
    ) -> Float32:
        """Draws one same-colour run, splitting at tabs; returns the new pen x."""
        var b = seg.as_bytes()
        var p = pen
        var start = 0
        for i in range(len(b) + 1):
            if i < len(b) and b[i] != 9:
                continue
            if i > start:
                var piece = String(seg[byte=start:i])
                dl.add_text(font_slot, font, piece, p, baseline_y, font_px, color)
                p += font.measure(piece, font_px)
            if i < len(b):
                p = advance_text(font, font_px, x0, p, "\t")
            start = i + 1
        return p

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
        mono_font_slot: Int = -1,
        mono_font: SdfFont = SdfFont(),
    ) -> CodeDrawerAction:
        """Draws the complete GPU-rendered Code/Diff/Problems drawer and processes pointer/keyboard input."""
        var action = CodeDrawerAction(CODE_ACT_NONE)
        self.last_hit_count = 0
        if w < Float32(64.0) or h < Float32(64.0):
            return action^

        var use_mono = mono_font_slot >= 0 and mono_font.valid
        var code_slot = mono_font_slot if use_mono else font_slot
        var code_font_valid = mono_font.valid if use_mono else font.valid

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
        if not self._valid(self.pending_close_tab):
            self.pending_close_tab = -1
        var notice_h = Float32(0.0)
        if self.pending_close_tab >= 0 or (
            self._valid(self.active_tab) and self.tabs[self.active_tab].disk_conflict
        ):
            notice_h = Float32(28.0)

        var prob_hdr_h = Float32(24.0)
        var prob_total_h = (
            min(self.problems_height, max(Float32(64.0), h * Float32(0.34)))
            if self.problems_open
            else prob_hdr_h
        )
        if prob_total_h > h - tab_bar_h - crumb_h - prop_bar_h - notice_h - Float32(48.0):
            prob_total_h = prob_hdr_h

        # --------------------------------------------------------------------
        # 1. Top Tab Bar + External IDE Button + Dynamic [Build & Hot-Reload] Pill
        # --------------------------------------------------------------------
        dl.add_rect(x, y, w, tab_bar_h, theme.l0_canvas)
        dl.add_rect(x, y + tab_bar_h - Float32(1.0), w, Float32(1.0), theme.border_subtle)

        var pill_h = Float32(22.0)
        var pill_y = y + Float32(4.0)
        var build_disp = self.build_label
        if w < Float32(390.0):
            if self.build_status == BUILD_STATUS_OK:
                build_disp = "Gen " + String(self.build_generation)
            elif self.build_status == BUILD_STATUS_BUILDING:
                build_disp = "Building..."
            elif self.build_status == BUILD_STATUS_ERROR:
                build_disp = String(self.build_error_count) + " Err"
            else:
                build_disp = "Reload"
        var build_tw = _measure_cd(font, build_disp, Float32(11.0))
        var build_w = max(Float32(72.0), min(Float32(142.0), build_tw + Float32(32.0)))
        var build_x = x + w - build_w - Float32(8.0)
        var build_hov = _hit_rect(mx, my, build_x, pill_y, build_w, pill_h)
        self.last_hit_count += 1

        var build_bg = theme.l3_hover if build_hov else theme.l3_raised
        var build_accent = theme.accent_primary
        var build_bdr = theme.border_strong if build_hov else theme.border_control
        if self.build_status == BUILD_STATUS_BUILDING:
            build_bg = rgba8_with_alpha(theme.accent_primary, 42)
            build_accent = theme.accent_primary
            build_bdr = rgba8_with_alpha(theme.accent_primary, 175)
        elif self.build_status == BUILD_STATUS_OK:
            build_bg = rgba8_with_alpha(theme.status_ok, 48 if build_hov else 32)
            build_accent = theme.status_ok
            build_bdr = rgba8_with_alpha(theme.status_ok, 185 if build_hov else 135)
        elif self.build_status == BUILD_STATUS_ERROR:
            build_bg = rgba8_with_alpha(theme.status_danger, 56 if build_hov else 40)
            build_accent = theme.status_danger
            build_bdr = rgba8_with_alpha(theme.status_danger, 200 if build_hov else 155)

        dl.add_rect_styled(
            build_x, pill_y, build_w, pill_h,
            build_bg, build_bdr,
            theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
        )
        if self.build_status == BUILD_STATUS_BUILDING:
            var spin_a = time_sec * Float32(5.2)
            dl.add_arc(
                build_x + Float32(13.0),
                pill_y + pill_h * Float32(0.5),
                Float32(5.2),
                Float32(1.7),
                spin_a,
                Float32(4.2),
                build_accent,
                rgba8_with_alpha(build_accent, 55),
            )
        elif self.build_status == BUILD_STATUS_ERROR:
            draw_studio_icon(dl, ICON_ERROR_CIRCLE, build_x + Float32(7.0), pill_y + Float32(5.0), Float32(12.0), build_accent)
        elif self.build_status == BUILD_STATUS_OK:
            draw_studio_icon(dl, ICON_CHECK, build_x + Float32(7.0), pill_y + Float32(5.0), Float32(12.0), build_accent)
        else:
            draw_studio_icon(dl, ICON_PLAY, build_x + Float32(7.0), pill_y + Float32(5.0), Float32(12.0), build_accent)
        _draw_cd_mid(
            dl, font_slot, font,
            _truncate_text(build_disp, 18),
            build_x + Float32(23.0), pill_y, pill_h,
            Float32(11.0), theme.text_primary,
        )
        if click and build_hov:
            action = CodeDrawerAction(CODE_ACT_BUILD_HOT_RELOAD)

        # Subtle vertical divider + External TurboMojo IDE button (`</>`)
        var ext_w = Float32(26.0)
        var ext_x = build_x - ext_w - Float32(8.0)
        dl.add_rect(build_x - Float32(4.5), pill_y + Float32(3.0), Float32(1.0), pill_h - Float32(6.0), theme.border_subtle)
        var ext_hov = _hit_rect(mx, my, ext_x, pill_y, ext_w, pill_h)
        self.last_hit_count += 1
        dl.add_rect_styled(
            ext_x, pill_y, ext_w, pill_h,
            theme.l3_hover if ext_hov else theme.l2_card,
            theme.border_strong if ext_hov else theme.border_control,
            theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
        )
        draw_studio_icon(dl, ICON_CODE, ext_x + Float32(6.5), pill_y + Float32(4.5), Float32(13.0), theme.text_primary if ext_hov else theme.text_secondary)
        if click and ext_hov and action.kind == CODE_ACT_NONE:
            action = CodeDrawerAction(CODE_ACT_OPEN_EXTERNAL_IDE)

        # Left-aligned Dynamic-Width File Tabs (measured via font.measure so filenames never truncate prematurely!)
        var tabs_limit_x = ext_x - Float32(8.0)
        var tx = x + Float32(4.0)
        var n_tabs = len(self.tabs)
        var avail_tabs_w = max(Float32(64.0), tabs_limit_x - tx)
        var per_tab_cap = avail_tabs_w / Float32(max(1, n_tabs))
        for ti in range(n_tabs):
            var title_w = _measure_cd(font, self.tabs[ti].title, Float32(11.5))
            var extra_w = Float32(64.0) if self.tabs[ti].dirty else Float32(54.0)
            var desired_tw = max(Float32(96.0), min(Float32(236.0), title_w + extra_w))
            var tw = min(desired_tw, max(Float32(72.0), per_tab_cap))
            if tx + Float32(60.0) > tabs_limit_x:
                break
            tw = min(tw, tabs_limit_x - tx)
            var is_act = ti == self.active_tab
            var thov = _hit_rect(mx, my, tx, y + Float32(2.0), tw, tab_bar_h - Float32(2.0))
            var close_x = tx + tw - Float32(20.0)
            var close_y = y + Float32(7.0)
            var chov = _hit_rect(mx, my, close_x, close_y, Float32(16.0), Float32(16.0))
            self.last_hit_count += 2

            if is_act:
                dl.add_rect(
                    tx, y + Float32(2.0), tw - Float32(2.0), tab_bar_h - Float32(1.0),
                    theme.l1_shell, theme.radius_xs, Float32(1.0), theme.border_subtle,
                )
                # Seamlessly merge active tab bottom into l1_shell breadcrumb bar
                dl.add_rect(tx + Float32(1.0), y + tab_bar_h - Float32(2.0), tw - Float32(4.0), Float32(3.0), theme.l1_shell)
                dl.add_accent_bar(tx + Float32(4.0), y + Float32(2.0), tw - Float32(10.0), Float32(2.0), theme.accent_primary, pill=True)
            elif thov:
                dl.add_rect(
                    tx, y + Float32(3.0), tw - Float32(2.0), tab_bar_h - Float32(4.0),
                    theme.l2_card, theme.radius_xs,
                )

            var ic = icon_for_path(self.tabs[ti].path)
            draw_studio_icon(
                dl, ic, tx + Float32(8.0), y + Float32(9.0), Float32(13.0),
                theme.accent_primary if is_act else theme.text_muted,
            )
            var label_x = tx + Float32(25.0)
            if self.tabs[ti].dirty:
                dl.add_circle(label_x + Float32(3.0), y + Float32(15.5), Float32(2.8), theme.status_warn)
                label_x += Float32(9.0)
            var avail_lbl_w = max(Float32(16.0), close_x - label_x - Float32(4.0))
            var tab_disp = self.tabs[ti].title
            if title_w > avail_lbl_w:
                var max_ch = max(5, Int(avail_lbl_w / Float32(6.2)))
                tab_disp = _truncate_text(tab_disp, max_ch)
            _draw_cd_mid(
                dl, font_slot, font,
                tab_disp,
                label_x, y + Float32(3.0), tab_bar_h - Float32(3.0), Float32(11.5),
                theme.text_primary if is_act else theme.text_secondary,
            )
            if chov:
                dl.add_rect(close_x, close_y, Float32(16.0), Float32(16.0), theme.l3_hover, theme.radius_xs)
            draw_studio_icon(
                dl, ICON_CLOSE, close_x + Float32(3.0), close_y + Float32(3.0), Float32(10.0),
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
        # 2. Scope Breadcrumb Bar + ALWAYS-VISIBLE `Code | Diff | Split` Switcher
        # --------------------------------------------------------------------
        var crumb_y = y + tab_bar_h
        dl.add_rect(x, crumb_y, w, crumb_h, theme.l1_shell)
        dl.add_rect(x, crumb_y + crumb_h - Float32(1.0), w, Float32(1.0), theme.border_subtle)

        # Compact Code | Diff | Split segmented control pinned on right of Breadcrumb bar
        var mode_labels = List[String]()
        mode_labels.append("Code")
        mode_labels.append("Diff")
        mode_labels.append("Split")
        var seg_w = Float32(38.0)
        var seg_total_w = seg_w * Float32(3.0)
        var seg_h = Float32(18.0)
        var seg_y = crumb_y + Float32(3.0)
        var seg_x0 = x + w - seg_total_w - Float32(8.0)
        var right_reserve_x = seg_x0 - Float32(8.0)

        if seg_x0 > x + Float32(60.0):
            dl.add_rect(
                seg_x0, seg_y, seg_total_w, seg_h,
                theme.l0_canvas, theme.radius_xs, Float32(1.0), theme.border_subtle,
            )
            for mi in range(3):
                var sx = seg_x0 + Float32(mi) * seg_w
                var shov = _hit_rect(mx, my, sx, seg_y, seg_w, seg_h)
                var active_m = self.view_mode == mi
                self.last_hit_count += 1
                if active_m:
                    dl.add_rect_styled(
                        sx + Float32(1.0), seg_y + Float32(1.0),
                        seg_w - Float32(2.0), seg_h - Float32(2.0),
                        theme.l3_raised, rgba8_with_alpha(theme.accent_primary, 130),
                        theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
                    )
                elif shov:
                    dl.add_rect(
                        sx + Float32(1.0), seg_y + Float32(1.0),
                        seg_w - Float32(2.0), seg_h - Float32(2.0),
                        theme.l2_card, theme.radius_xs,
                    )
                var mtw = _measure_cd(font, mode_labels[mi], Float32(10.5))
                _draw_cd_mid(
                    dl, font_slot, font, mode_labels[mi],
                    sx + max(Float32(3.0), (seg_w - mtw) * Float32(0.5)),
                    seg_y, seg_h, Float32(10.5),
                    theme.accent_primary if active_m else (theme.text_primary if shov else theme.text_secondary),
                )
                if click and shov and action.kind == CODE_ACT_NONE:
                    self.view_mode = mi

        if self.active_tab >= 0 and self.active_tab < len(self.tabs):
            ref tab = self.tabs[self.active_tab]
            # Optional compact Ln readout to the left of the mode toggle when space permits
            if w >= Float32(430.0):
                var loc_txt = "Ln " + String(tab.cursor_line + 1)
                var loc_w = _measure_cd(mono_font if use_mono else font, loc_txt, Float32(10.5))
                var loc_x = right_reserve_x - loc_w - Float32(4.0)
                _draw_cd_mid(
                    dl, code_slot, mono_font if use_mono else font, loc_txt,
                    loc_x, crumb_y, crumb_h, Float32(10.5), theme.text_muted,
                )
                right_reserve_x = loc_x - Float32(8.0)

            var scope = scope_breadcrumbs(tab.symbols, tab.cursor_line)
            var crumb_txt = tab.path if tab.path.byte_length() > 0 else tab.title
            if scope.byte_length() > 0:
                crumb_txt += "  >  " + scope
            var max_crumb_ch = max(10, Int((right_reserve_x - (x + Float32(10.0))) / Float32(6.3)))
            _draw_cd_mid(
                dl, font_slot, font,
                _truncate_text(crumb_txt, max_crumb_ch),
                x + Float32(10.0), crumb_y, crumb_h,
                Float32(11.0), theme.text_secondary,
            )
        else:
            var max_empty_ch = max(12, Int((right_reserve_x - (x + Float32(10.0))) / Float32(6.3)))
            _draw_cd_mid(
                dl, font_slot, font,
                _truncate_text("No file open - click a component or diagnostic to inspect source", max_empty_ch),
                x + Float32(10.0), crumb_y, crumb_h,
                Float32(11.0), theme.text_muted,
            )

        # --------------------------------------------------------------------
        # 3. Optional Agent Proposal Banner (Dynamic-Width [Accept All] / [Reject All])
        # --------------------------------------------------------------------
        var body_y = crumb_y + crumb_h
        if has_active_prop:
            dl.add_rect(x, body_y, w, prop_bar_h, theme.agent_soft)
            dl.add_rect(x, body_y, Float32(3.0), prop_bar_h, theme.agent_active)
            dl.add_rect(x, body_y + prop_bar_h - Float32(1.0), w, Float32(1.0), rgba8_with_alpha(theme.agent_active, 95))
            draw_studio_icon(dl, ICON_SPARK, x + Float32(10.0), body_y + Float32(7.0), Float32(14.0), theme.agent_active)
            var pending_hunks = self.tabs[self.active_tab].diff_doc.pending_count()

            var acc_lbl = String("Accept All")
            var rej_lbl = String("Reject All")
            var acc_tw = _measure_cd(font, acc_lbl, Float32(10.5))
            var rej_tw = _measure_cd(font, rej_lbl, Float32(10.5))
            var acc_w = max(Float32(72.0), acc_tw + Float32(18.0))
            var rej_w = max(Float32(70.0), rej_tw + Float32(18.0))
            var rej_x = x + w - rej_w - Float32(8.0)
            var acc_x = rej_x - acc_w - Float32(6.0)

            var prop_label = (
                "Agent Edit (" + String(pending_hunks) + " hunks)"
                if w >= Float32(375.0)
                else (String(pending_hunks) + " pending hunks")
            )
            var prop_max_ch = max(8, Int((acc_x - (x + Float32(32.0))) / Float32(6.4)))
            _draw_cd_mid(
                dl, font_slot, font,
                _truncate_text(prop_label, prop_max_ch),
                x + Float32(28.0), body_y, prop_bar_h,
                Float32(11.0), theme.text_primary,
            )
            var by_btn = body_y + Float32(4.0)
            var bh_btn = Float32(20.0)
            var acc_hov = _hit_rect(mx, my, acc_x, by_btn, acc_w, bh_btn)
            var rej_hov = _hit_rect(mx, my, rej_x, by_btn, rej_w, bh_btn)
            self.last_hit_count += 2

            dl.add_rect_styled(
                acc_x, by_btn, acc_w, bh_btn,
                rgba8_with_alpha(theme.status_ok, 235 if acc_hov else 205),
                theme.status_ok,
                theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )
            _draw_cd_mid(
                dl, font_slot, font, acc_lbl,
                acc_x + max(Float32(4.0), (acc_w - acc_tw) * Float32(0.5)),
                by_btn, bh_btn, Float32(10.5), theme.text_on_accent,
            )

            dl.add_rect_styled(
                rej_x, by_btn, rej_w, bh_btn,
                rgba8_with_alpha(theme.status_danger, 95 if rej_hov else 52),
                rgba8_with_alpha(theme.status_danger, 235 if rej_hov else 190),
                theme.radius_xs, Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )
            _draw_cd_mid(
                dl, font_slot, font, rej_lbl,
                rej_x + max(Float32(4.0), (rej_w - rej_tw) * Float32(0.5)),
                by_btn, bh_btn, Float32(10.5), theme.text_primary,
            )

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
        # 3b. Unsaved-close / disk-conflict notice (Save | Discard | Cancel, Reload | Keep Mine)
        # --------------------------------------------------------------------
        if notice_h > Float32(0.0):
            var notice_act = self._draw_notice_bar(
                dl, font_slot, font, theme, x, body_y, w, notice_h, mx, my, click
            )
            if notice_act.kind != CODE_ACT_NONE and action.kind == CODE_ACT_NONE:
                action = notice_act^
            body_y += notice_h

        # --------------------------------------------------------------------
        # 4. Main Code or Diff Canvas
        # --------------------------------------------------------------------
        var prob_y = y + h - prob_total_h
        var body_h = max(Float32(0.0), prob_y - body_y)
        var line_h = Float32(18.0)
        var mono_px = theme.font_mono
        var visible_rows = max(1, Int(body_h / line_h))
        self.visible_rows = visible_rows

        # Ctrl+S saves the active tab whenever the drawer owns the keyboard.
        if (
            self.has_keyboard_focus
            and self._valid(self.active_tab)
            and (inp.key_down(341) or inp.key_down(345))
            and inp.key_pressed(83)
            and action.kind == CODE_ACT_NONE
        ):
            var saved = self.save_tab(self.active_tab)
            action = CodeDrawerAction(CODE_ACT_SAVE_TAB)
            action.tab_index = self.active_tab
            action.path = self._save_target(self.active_tab)
            action.message = String("") if saved else self.last_error

        dl.push_clip(Int(x), Int(body_y), Int(x + w), Int(body_y + body_h))
        dl.add_rect(x, body_y, w, body_h, theme.l0_canvas)

        if self.active_tab >= 0 and self.active_tab < len(self.tabs) and body_h > Float32(16.0):
            # Scroll & keyboard navigation when mouse/focus in body
            if _hit_rect(mx, my, x, body_y, w, body_h) and inp.wheel != Float32(0.0):
                var delta = -3 if inp.wheel > Float32(0.0) else 3
                self.tabs[self.active_tab].scroll_line = max(
                    0, self.tabs[self.active_tab].scroll_line + delta
                )
            if self.has_keyboard_focus and self.view_mode == CODE_MODE_SOURCE:
                var last_ln = max(0, len(self.tabs[self.active_tab].lines) - 1)
                var cur = self.tabs[self.active_tab].cursor_line
                var moved = True
                if inp.key_pressed(265):  # Up
                    cur -= 1
                elif inp.key_pressed(264):  # Down
                    cur += 1
                elif inp.key_pressed(266):  # Page Up
                    cur -= max(1, visible_rows - 1)
                elif inp.key_pressed(267):  # Page Down
                    cur += max(1, visible_rows - 1)
                else:
                    moved = False
                if moved:
                    self.tabs[self.active_tab].cursor_line = max(0, min(last_ln, cur))
                    self._ensure_cursor_visible()

            if self.view_mode == CODE_MODE_SOURCE:
                self._draw_source_canvas(
                    dl, code_slot, mono_font if use_mono else font, theme,
                    x, body_y, w, body_h, line_h, mono_px, visible_rows, mx, my, click,
                )
            else:
                var diff_act = self._draw_diff_canvas(
                    dl, font_slot, font, code_slot, mono_font if use_mono else font, theme,
                    x, body_y, w, body_h, line_h, mono_px, visible_rows, mx, my, click,
                )
                if diff_act.kind != CODE_ACT_NONE and action.kind == CODE_ACT_NONE:
                    action = diff_act^

        _ = code_font_valid
        dl.pop_clip()

        # --------------------------------------------------------------------
        # 5. Collapsible Problems Strip (Semantic Count Pills + Message-First Rows)
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
        _draw_cd_mid(
            dl, font_slot, font, "Problems",
            x + Float32(25.0), prob_y, prob_hdr_h, Float32(11.5), theme.text_primary,
        )

        # Semantic pill badges for Errors & Warnings (generous icon-to-digit gap + padding)
        var pb_x = x + Float32(25.0) + _measure_cd(font, "Problems", Float32(11.5)) + Float32(10.0)
        var pb_y = prob_y + Float32(4.0)
        var pb_h = Float32(16.0)

        var err_s = String(errs)
        var err_w = max(Float32(34.0), _measure_cd(mono_font if use_mono else font, err_s, Float32(10.0)) + Float32(27.0))
        var err_bg = rgba8_with_alpha(theme.status_danger, 44) if errs > 0 else theme.l2_card
        var err_bdr = rgba8_with_alpha(theme.status_danger, 150) if errs > 0 else theme.border_subtle
        var err_fg = theme.status_danger if errs > 0 else theme.text_muted
        dl.add_rect(pb_x, pb_y, err_w, pb_h, err_bg, Float32(8.0), Float32(1.0), err_bdr)
        draw_studio_icon(dl, ICON_ERROR_CIRCLE, pb_x + Float32(6.0), pb_y + Float32(3.0), Float32(10.0), err_fg, Float32(1.3))
        _draw_cd_mid(dl, code_slot, mono_font if use_mono else font, err_s, pb_x + Float32(20.0), pb_y, pb_h, Float32(10.0), err_fg)
        pb_x += err_w + Float32(6.0)

        var warn_s = String(warns)
        var warn_w = max(Float32(34.0), _measure_cd(mono_font if use_mono else font, warn_s, Float32(10.0)) + Float32(27.0))
        var warn_bg = rgba8_with_alpha(theme.status_warn, 40) if warns > 0 else theme.l2_card
        var warn_bdr = rgba8_with_alpha(theme.status_warn, 145) if warns > 0 else theme.border_subtle
        var warn_fg = theme.status_warn if warns > 0 else theme.text_muted
        dl.add_rect(pb_x, pb_y, warn_w, pb_h, warn_bg, Float32(8.0), Float32(1.0), warn_bdr)
        draw_studio_icon(dl, ICON_WARN, pb_x + Float32(6.0), pb_y + Float32(3.0), Float32(10.0), warn_fg, Float32(1.3))
        _draw_cd_mid(dl, code_slot, mono_font if use_mono else font, warn_s, pb_x + Float32(20.0), pb_y, pb_h, Float32(10.0), warn_fg)

        if click and phdr_hov and action.kind == CODE_ACT_NONE:
            self.problems_open = not self.problems_open
            action = CodeDrawerAction(CODE_ACT_TOGGLE_PROBLEMS)

        if self.problems_open and prob_total_h > prob_hdr_h + Float32(4.0):
            var list_y = prob_y + prob_hdr_h
            var list_h = prob_total_h - prob_hdr_h
            dl.push_clip(Int(x), Int(list_y), Int(x + w), Int(list_y + list_h))
            dl.add_rect(x, list_y, w, Float32(1.0), theme.border_subtle)
            var cur_path = self.tabs[self.active_tab].path if (self.active_tab >= 0 and self.active_tab < len(self.tabs)) else String("")
            var diags = self.problems.query(ProblemFilter(), cur_path)
            var row_h = Float32(22.0)
            var max_rows = max(1, Int(list_h / row_h))
            if len(diags) == 0:
                _draw_cd_mid(
                    dl, font_slot, font, "No compiler or LSP diagnostics.",
                    x + Float32(14.0), list_y + Float32(2.0), Float32(20.0),
                    Float32(11.0), theme.text_muted,
                )
            else:
                for di in range(min(max_rows, len(diags))):
                    ref d = diags[di]
                    var ry = list_y + Float32(1.0) + Float32(di) * row_h
                    var rhov = _hit_rect(mx, my, x, ry, w, row_h)
                    var ask_w = Float32(50.0)
                    var ask_x = x + w - ask_w - Float32(8.0)
                    var ask_y = ry + Float32(2.0)
                    var ask_h = row_h - Float32(4.0)
                    var ask_hov = _hit_rect(mx, my, ask_x, ask_y, ask_w, ask_h)
                    self.last_hit_count += 2
                    if rhov:
                        dl.add_rect(x, ry, w, row_h, theme.l2_card)
                    var is_err = d.severity == DIAG_ERROR
                    var sev_col = (
                        theme.status_danger
                        if is_err
                        else (theme.status_warn if d.severity == DIAG_WARNING else theme.accent_primary)
                    )
                    draw_studio_icon(
                        dl, ICON_ERROR_CIRCLE if is_err else ICON_WARN,
                        x + Float32(9.0), ry + Float32(5.5), Float32(11.0), sev_col, Float32(1.35),
                    )

                    # Compact right-aligned file:line tag so the diagnostic MESSAGE gets priority!
                    var loc_tag = (
                        (_basename(d.path) + ":" + String(d.line))
                        if w >= Float32(480.0)
                        else ("L" + String(d.line) + ":" + String(d.col))
                    )
                    var loc_w = _measure_cd(mono_font if use_mono else font, loc_tag, Float32(10.0))
                    var loc_x = ask_x - loc_w - Float32(8.0)
                    _draw_cd_mid(
                        dl, code_slot, mono_font if use_mono else font, loc_tag,
                        loc_x, ry, row_h, Float32(10.0), theme.text_muted,
                    )

                    var msg_x = x + Float32(25.0)
                    var max_c = max(10, Int((loc_x - msg_x - Float32(6.0)) / Float32(6.3)))
                    _draw_cd_mid(
                        dl, font_slot, font, _truncate_text(d.message, max_c),
                        msg_x, ry, row_h,
                        Float32(11.0), theme.text_primary if is_err else theme.text_secondary,
                    )

                    # Compact [✦ Fix] pill
                    dl.add_rect(
                        ask_x, ask_y, ask_w, ask_h,
                        rgba8_with_alpha(theme.agent_active, 56 if ask_hov else 26),
                        theme.radius_xs, Float32(1.0),
                        theme.agent_active if ask_hov else rgba8_with_alpha(theme.agent_active, 115),
                    )
                    draw_studio_icon(dl, ICON_SPARK, ask_x + Float32(5.0), ask_y + Float32(4.0), Float32(10.0), theme.agent_active, Float32(1.3))
                    _draw_cd_mid(
                        dl, font_slot, font, "Fix",
                        ask_x + Float32(18.0), ask_y, ask_h,
                        Float32(10.5), theme.text_primary if ask_hov else theme.agent_active,
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

    def _notice_button(
        self,
        mut dl: UiDrawList,
        font_slot: Int,
        font: SdfFont,
        theme: StudioTheme,
        label: String,
        bx: Float32,
        by: Float32,
        bw: Float32,
        accent: UInt32,
        mx: Float32,
        my: Float32,
    ) -> Bool:
        var hov = _hit_rect(mx, my, bx, by, bw, Float32(20.0))
        dl.add_rect(bx, by, bw, Float32(20.0), rgba8_with_alpha(accent, 70 if hov else 40), theme.radius_xs, Float32(1.0), accent)
        dl.add_text(font_slot, font, label, bx + Float32(8.0), by + Float32(14.0), Float32(11.0), theme.text_primary)
        return hov

    def _draw_notice_bar(
        mut self,
        mut dl: UiDrawList,
        font_slot: Int,
        font: SdfFont,
        theme: StudioTheme,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        mx: Float32,
        my: Float32,
        click: Bool,
    ) -> CodeDrawerAction:
        """Unsaved-close prompt (Save / Discard / Cancel) or disk-conflict prompt
        (Reload / Keep Mine) for the active tab."""
        var act = CodeDrawerAction(CODE_ACT_NONE)
        dl.add_rect(x, y, w, h, rgba8_with_alpha(theme.status_warn, 30))
        dl.add_rect(x, y + h - Float32(1.0), w, Float32(1.0), rgba8_with_alpha(theme.status_warn, 90))
        draw_studio_icon(dl, ICON_WARN, x + Float32(10.0), y + Float32(7.0), Float32(14.0), theme.status_warn)
        var by = y + Float32(4.0)
        self.last_hit_count += 3
        if self.pending_close_tab >= 0:
            var ti = self.pending_close_tab
            var cx = x + w - Float32(70.0)
            var dx = cx - Float32(78.0)
            var sx = dx - Float32(62.0)
            dl.add_text(
                font_slot, font,
                _truncate_text("Unsaved changes in " + self.tabs[ti].title, max(8, Int((sx - x - Float32(40.0)) / Float32(7.0)))),
                x + Float32(30.0), y + Float32(18.0), Float32(11.5), theme.text_primary,
            )
            var save_h = self._notice_button(dl, font_slot, font, theme, "Save", sx, by, Float32(56.0), theme.status_ok, mx, my)
            var disc_h = self._notice_button(dl, font_slot, font, theme, "Discard", dx, by, Float32(72.0), theme.status_danger, mx, my)
            var canc_h = self._notice_button(dl, font_slot, font, theme, "Cancel", cx, by, Float32(62.0), theme.border_control, mx, my)
            if click and save_h:
                var saved = self.save_tab(ti)
                act = CodeDrawerAction(CODE_ACT_SAVE_TAB)
                act.tab_index = ti
                act.path = self._save_target(ti)
                act.message = String("") if saved else self.last_error
                if saved:
                    self.close_tab(ti, force=True)
                else:
                    self.pending_close_tab = -1
            elif click and disc_h:
                act = CodeDrawerAction(CODE_ACT_CLOSE_TAB)
                act.tab_index = ti
                act.path = self.tabs[ti].path
                self.close_tab(ti, force=True)
            elif click and canc_h:
                self.pending_close_tab = -1
            return act^
        var ai = self.active_tab
        var kx = x + w - Float32(92.0)
        var rx = kx - Float32(72.0)
        dl.add_text(
            font_slot, font,
            _truncate_text(self.tabs[ai].title + " changed on disk (you have unsaved edits)", max(8, Int((rx - x - Float32(40.0)) / Float32(7.0)))),
            x + Float32(30.0), y + Float32(18.0), Float32(11.5), theme.text_primary,
        )
        var rel_h = self._notice_button(dl, font_slot, font, theme, "Reload", rx, by, Float32(66.0), theme.accent_primary, mx, my)
        var keep_h = self._notice_button(dl, font_slot, font, theme, "Keep Mine", kx, by, Float32(84.0), theme.status_ok, mx, my)
        if click and rel_h:
            act = CodeDrawerAction(CODE_ACT_RELOAD_FROM_DISK)
            act.tab_index = ai
            act.path = self.tabs[ai].path
            if not self.reload_tab_from_disk(ai):
                act.message = self.last_error
        elif click and keep_h:
            act = CodeDrawerAction(CODE_ACT_KEEP_BUFFER)
            act.tab_index = ai
            act.path = self.tabs[ai].path
            self.keep_tab_buffer(ai)
        return act^

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
        var gutter_w = Float32(46.0)
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
                dl.add_rect(x + gutter_w + Float32(1.0), ry, w - gutter_w - Float32(1.0), line_h, rgba8_with_alpha(theme.accent_primary, 22))

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

            # Right-aligned line number
            var ln_s = String(ln + 1)
            var ln_w = _measure_cd(font, ln_s, Float32(10.5))
            _draw_cd_mid(
                dl, font_slot, font, ln_s,
                x + max(Float32(4.0), gutter_w - Float32(9.0) - ln_w), ry, line_h,
                Float32(10.5), theme.text_primary if is_cur else theme.text_muted,
            )

            # Syntax-highlighted code line (4px+ right inset clip so glyphs never kiss right dock border)
            var code_x = x + gutter_w + Float32(10.0)
            var base_y = ry + (line_h + mono_px * Float32(0.66)) * Float32(0.5)
            dl.push_clip(Int(x + gutter_w + Float32(1.0)), Int(ry), Int(x + w - Float32(8.0)), Int(ry + line_h))
            if ln < len(tab.spans):
                self._draw_syntax_text(dl, font_slot, font, theme, tab.lines[ln], tab.spans[ln], code_x, base_y, mono_px)
            else:
                dl.add_text(font_slot, font, tab.lines[ln], code_x, base_y, mono_px, theme.text_primary)
            dl.pop_clip()

            # Inline diagnostic squiggle & badge for diagnostics on this line
            for di in range(len(file_diags)):
                ref d = file_diags[di]
                if d.line == ln + 1:
                    var dcol = theme.status_danger if d.severity == DIAG_ERROR else theme.status_warn
                    var span = diag_squiggle_span(font, mono_px, tab.lines[ln], d.col, d.end_col if d.end_line == d.line or d.end_line == 0 else 0)
                    self._draw_squiggle(dl, code_x + span[0], ry + line_h - Float32(2.0), span[1], dcol)
                    var badge = _truncate_text(d.message, 32)
                    var line_end_x = advance_text(font, mono_px, code_x, code_x, tab.lines[ln])
                    var bx = max(line_end_x + Float32(16.0), x + w - Float32(220.0))
                    if bx + Float32(60.0) < x + w:
                        dl.add_rect(bx, ry + Float32(2.0), x + w - bx - Float32(10.0), line_h - Float32(4.0), rgba8_with_alpha(dcol, 34), theme.radius_xs)
                        _draw_cd_mid(dl, font_slot, font, badge, bx + Float32(6.0), ry, line_h, Float32(10.0), dcol)
                    break

        # Sleek 4px vertical scrollbar thumb when file exceeds viewport
        if n_lines > visible_rows and h > Float32(24.0):
            var track_h = h - Float32(8.0)
            var thumb_h = max(Float32(20.0), track_h * (Float32(visible_rows) / Float32(max(1, n_lines))))
            var max_top = max(1, n_lines - visible_rows)
            var frac = Float32(top) / Float32(max_top)
            var sb_y = y + Float32(4.0) + min(Float32(1.0), frac) * max(Float32(0.0), track_h - thumb_h)
            dl.add_rect(x + w - Float32(6.0), sb_y, Float32(3.5), thumb_h, theme.border_control, Float32(2.0))

    def _draw_diff_canvas(
        mut self,
        mut dl: UiDrawList,
        ui_slot: Int,
        ui_font: SdfFont,
        code_slot: Int,
        code_font: SdfFont,
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
            _draw_cd_mid(
                dl, ui_slot, ui_font, "No changes (file matches base).",
                x + Float32(16.0), y + Float32(12.0), Float32(24.0),
                Float32(12.0), theme.text_muted,
            )
            return act^

        var top = max(0, min(max(0, len(rows) - 1), self.tabs[self.active_tab].scroll_line))
        self.tabs[self.active_tab].scroll_line = top
        self.tabs[self.active_tab].ensure_diff_spans(self.highlighter)
        var empty_spans = List[HighlightSpan]()

        var old_bg = theme.diff_del_bg
        var new_bg = theme.diff_add_bg
        var half_w = w * Float32(0.5)
        var gutter_w = Float32(44.0)
        var split_gutter_w = Float32(34.0)
        var start_y = y

        if not is_split:
            # Unified diff gutter surface & vertical hairline divider
            dl.add_rect(x, y, gutter_w, h, theme.l1_shell)
            dl.add_rect(x + gutter_w, y, Float32(1.0), h, theme.border_subtle)
        else:
            # Split diff Original / Proposed mini header + per-pane gutters
            var sh_h = Float32(20.0)
            dl.add_rect(x, y, w, sh_h, theme.l1_shell)
            dl.add_rect(x, y + sh_h - Float32(1.0), w, Float32(1.0), theme.border_subtle)
            _draw_cd_mid(dl, ui_slot, ui_font, "Original", x + Float32(10.0), y, sh_h, Float32(10.5), theme.diff_del_fg)
            _draw_cd_mid(dl, ui_slot, ui_font, "Proposed", x + half_w + Float32(10.0), y, sh_h, Float32(10.5), theme.diff_add_fg)
            start_y = y + sh_h
            var body_sub_h = max(Float32(0.0), h - sh_h)
            dl.add_rect(x, start_y, split_gutter_w, body_sub_h, theme.l1_shell)
            dl.add_rect(x + split_gutter_w, start_y, Float32(1.0), body_sub_h, theme.border_subtle)
            dl.add_rect(x + half_w + Float32(1.0), start_y, split_gutter_w, body_sub_h, theme.l1_shell)
            dl.add_rect(x + half_w + Float32(1.0) + split_gutter_w, start_y, Float32(1.0), body_sub_h, theme.border_subtle)
            dl.add_rect(x + half_w, y, Float32(1.0), h, theme.border_control)

        var cur_y = start_y
        var ri = top
        while ri < len(rows) and cur_y < y + h:
            ref row = rows[ri]
            ri += 1

            if row.kind == DIFF_ROW_HEADER:
                # Neutral elevated L2 hunk header (22px height, zero amber-on-blue clash; hunk-0 Accept covers (535,91) at w=640)
                var hunk_h = Float32(22.0)
                dl.add_rect(x, cur_y, w, hunk_h, theme.l2_card)
                dl.add_rect(x, cur_y, w, hunk_h, theme.diff_hunk_bg)
                dl.add_rect(x, cur_y, w, Float32(1.0), theme.border_subtle)
                dl.add_rect(x, cur_y + hunk_h - Float32(1.0), w, Float32(1.0), theme.border_subtle)
                draw_studio_icon(dl, ICON_DIFF, x + Float32(8.0), cur_y + Float32(5.0), Float32(12.0), theme.agent_active)
                _draw_cd_mid(dl, code_slot, code_font, row.text, x + Float32(25.0), cur_y, hunk_h, Float32(10.5), theme.text_secondary)

                # Clickable per-hunk [Accept] / [Reject] buttons
                var hi = row.hunk
                if hi >= 0 and hi < len(self.tabs[self.active_tab].diff_doc.decisions):
                    var dec = self.tabs[self.active_tab].diff_doc.decisions[hi]
                    var bw = Float32(58.0)
                    var bh = Float32(18.0)
                    var by = cur_y + Float32(2.0)
                    var rbx = x + w - bw - Float32(8.0)
                    var abx = rbx - bw - Float32(6.0)
                    var ahov = _hit_rect(mx, my, abx, by, bw, bh)
                    var rhov = _hit_rect(mx, my, rbx, by, bw, bh)
                    self.last_hit_count += 2

                    var acc_fill = (
                        rgba8_with_alpha(theme.status_ok, 110)
                        if dec == HUNK_ACCEPTED
                        else rgba8_with_alpha(theme.status_ok, 68 if ahov else 36)
                    )
                    var rej_fill = (
                        rgba8_with_alpha(theme.status_danger, 110)
                        if dec == HUNK_REJECTED
                        else rgba8_with_alpha(theme.status_danger, 68 if rhov else 36)
                    )
                    dl.add_rect(abx, by, bw, bh, acc_fill, theme.radius_xs, Float32(1.0), rgba8_with_alpha(theme.status_ok, 215 if (ahov or dec == HUNK_ACCEPTED) else 155))
                    var acc_w_t = _measure_cd(ui_font, "Accept", Float32(10.5))
                    _draw_cd_mid(dl, ui_slot, ui_font, "Accept", abx + max(Float32(4.0), (bw - acc_w_t) * Float32(0.5)), by, bh, Float32(10.5), theme.text_primary)
                    dl.add_rect(rbx, by, bw, bh, rej_fill, theme.radius_xs, Float32(1.0), rgba8_with_alpha(theme.status_danger, 215 if (rhov or dec == HUNK_REJECTED) else 155))
                    var rej_w_t = _measure_cd(ui_font, "Reject", Float32(10.5))
                    _draw_cd_mid(dl, ui_slot, ui_font, "Reject", rbx + max(Float32(4.0), (bw - rej_w_t) * Float32(0.5)), by, bh, Float32(10.5), theme.text_primary)

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
                cur_y += hunk_h
                continue

            var base_y = cur_y + (line_h + mono_px * Float32(0.66)) * Float32(0.5)
            if row.kind == DIFF_ROW_NOTE:
                # "\ No newline at end of file" style notes: muted, no gutter number or syntax colour.
                var note_x = (
                    x + gutter_w + Float32(18.0)
                    if not is_split
                    else (
                        x + split_gutter_w + Float32(6.0)
                        if row.left_text.byte_length() > 0
                        else x + half_w + split_gutter_w + Float32(7.0)
                    )
                )
                dl.push_clip(Int(x), Int(cur_y), Int(x + w - Float32(8.0)), Int(cur_y + line_h))
                _draw_cd_mid(dl, code_slot, code_font, row.text, note_x, cur_y, line_h, Float32(10.5), theme.text_muted)
                dl.pop_clip()
                cur_y += line_h
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
                    dl.add_rect(x, cur_y, w, line_h, bg)
                    dl.add_rect(x, cur_y, Float32(3.0), line_h, bar_col)
                var ln_str = (
                    String(row.new_line + 1)
                    if row.new_line >= 0 and row.kind != DIFF_ROW_OLD
                    else (String(row.old_line + 1) if row.old_line >= 0 else String(""))
                )
                if ln_str.byte_length() > 0:
                    var ln_w = _measure_cd(code_font, ln_str, Float32(10.0))
                    _draw_cd_mid(
                        dl, code_slot, code_font, ln_str,
                        x + max(Float32(5.0), gutter_w - Float32(7.0) - ln_w), cur_y, line_h,
                        Float32(10.0),
                        theme.diff_del_fg if row.kind == DIFF_ROW_OLD else (theme.diff_add_fg if row.kind == DIFF_ROW_NEW else theme.text_muted),
                    )
                var prefix = String(row.text[byte=0:1]) if row.text.byte_length() > 0 else String(" ")
                var body = String(row.text[byte=1 : row.text.byte_length()]) if row.text.byte_length() > 1 else String("")
                var pcol = (
                    theme.status_danger
                    if row.kind == DIFF_ROW_OLD
                    else (theme.status_ok if row.kind == DIFF_ROW_NEW else theme.text_muted)
                )
                dl.push_clip(Int(x + gutter_w + Float32(1.0)), Int(cur_y), Int(x + w - Float32(8.0)), Int(cur_y + line_h))
                dl.add_text(code_slot, code_font, prefix, x + gutter_w + Float32(6.0), base_y, mono_px, pcol)
                # Spans come from highlighting each side as a whole file (multi-line state).
                ref tab_u = self.tabs[self.active_tab]
                var body_x = x + gutter_w + Float32(18.0)
                if row.kind == DIFF_ROW_OLD and row.old_line >= 0 and row.old_line < len(tab_u.diff_old_spans):
                    self._draw_syntax_text(dl, code_slot, code_font, theme, body, tab_u.diff_old_spans[row.old_line], body_x, base_y, mono_px)
                elif row.kind != DIFF_ROW_OLD and row.new_line >= 0 and row.new_line < len(tab_u.diff_new_spans):
                    self._draw_syntax_text(dl, code_slot, code_font, theme, body, tab_u.diff_new_spans[row.new_line], body_x, base_y, mono_px)
                else:
                    self._draw_syntax_text(dl, code_slot, code_font, theme, body, empty_spans, body_x, base_y, mono_px)
                dl.pop_clip()
            else:
                # Split side-by-side rendering with strict per-pane clipping & right-aligned line numbers
                var left_bg = old_bg if (row.kind == DIFF_ROW_OLD or row.kind == DIFF_ROW_SPLIT_CHG) else UInt32(0)
                var right_bg = new_bg if (row.kind == DIFF_ROW_NEW or row.kind == DIFF_ROW_SPLIT_CHG) else UInt32(0)
                if (left_bg >> 24) != 0:
                    dl.add_rect(x, cur_y, half_w, line_h, left_bg)
                    dl.add_rect(x, cur_y, Float32(3.0), line_h, theme.status_danger)
                if (right_bg >> 24) != 0:
                    dl.add_rect(x + half_w + Float32(1.0), cur_y, half_w - Float32(1.0), line_h, right_bg)
                    dl.add_rect(x + half_w + Float32(1.0), cur_y, Float32(3.0), line_h, theme.status_ok)

                # Left pane (clipped cleanly to half_w)
                dl.push_clip(Int(x), Int(cur_y), Int(x + half_w - Float32(2.0)), Int(cur_y + line_h))
                if row.old_line >= 0:
                    var ol_s = String(row.old_line + 1)
                    var ol_w = _measure_cd(code_font, ol_s, Float32(10.0))
                    _draw_cd_mid(dl, code_slot, code_font, ol_s, x + max(Float32(4.0), split_gutter_w - Float32(5.0) - ol_w), cur_y, line_h, Float32(10.0), theme.text_muted)
                ref tab_s = self.tabs[self.active_tab]
                if row.left_text.byte_length() > 0:
                    var lx = x + split_gutter_w + Float32(6.0)
                    if row.old_line >= 0 and row.old_line < len(tab_s.diff_old_spans):
                        self._draw_syntax_text(dl, code_slot, code_font, theme, row.left_text, tab_s.diff_old_spans[row.old_line], lx, base_y, mono_px)
                    else:
                        self._draw_syntax_text(dl, code_slot, code_font, theme, row.left_text, empty_spans, lx, base_y, mono_px)
                dl.pop_clip()

                # Right pane (clipped cleanly to right half)
                dl.push_clip(Int(x + half_w + Float32(1.0)), Int(cur_y), Int(x + w - Float32(8.0)), Int(cur_y + line_h))
                if row.new_line >= 0:
                    var nl_s = String(row.new_line + 1)
                    var nl_w = _measure_cd(code_font, nl_s, Float32(10.0))
                    _draw_cd_mid(dl, code_slot, code_font, nl_s, x + half_w + Float32(1.0) + max(Float32(4.0), split_gutter_w - Float32(5.0) - nl_w), cur_y, line_h, Float32(10.0), theme.text_muted)
                if row.right_text.byte_length() > 0:
                    var rx = x + half_w + split_gutter_w + Float32(7.0)
                    if row.new_line >= 0 and row.new_line < len(tab_s.diff_new_spans):
                        self._draw_syntax_text(dl, code_slot, code_font, theme, row.right_text, tab_s.diff_new_spans[row.new_line], rx, base_y, mono_px)
                    else:
                        self._draw_syntax_text(dl, code_slot, code_font, theme, row.right_text, empty_spans, rx, base_y, mono_px)
                dl.pop_clip()

            cur_y += line_h

        # Sleek 4px vertical scrollbar thumb when diff exceeds viewport
        if len(rows) > visible_rows and h > Float32(24.0):
            var track_h = max(Float32(16.0), (y + h) - start_y - Float32(8.0))
            var thumb_h = max(Float32(20.0), track_h * (Float32(visible_rows) / Float32(max(1, len(rows)))))
            var max_top = max(1, len(rows) - visible_rows)
            var frac = Float32(top) / Float32(max_top)
            var sb_y = start_y + Float32(4.0) + min(Float32(1.0), frac) * max(Float32(0.0), track_h - thumb_h)
            dl.add_rect(x + w - Float32(6.0), sb_y, Float32(3.5), thumb_h, theme.border_control, Float32(2.0))

        return act^
