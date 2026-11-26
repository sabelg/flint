"""Universal Command Palette, Quick-Spawn & Non-Blocking Key-Chord HUD (`flint.studio.palette`).

Pure-Mojo 1.2 UI-agnostic interaction model for Flint Studio:
  - `StudioCommandPalette`: Single-field unified fuzzy + frecency search across
    Commands (`>`), Quick-Spawn (`+`), Scene Entities (`@`), and Project Assets (`#`),
    with PIE-mode safety gating, keyboard navigation (`Up/Down/PgUp/PgDn/Tab/Enter/Esc`),
    and matched highlight spans for rendering.
  - `KeyChordHud`: Non-focus-stealing chord hint state that stays quiet for fast typists
    (`200 ms` threshold) and reveals valid follow-up keys and toggle states on pause.
"""
from std.collections import List
from std.math import isfinite
from flint.render.font import SdfFont
from flint.studio.frecency import FrecencyStore, FuzzyMatch, FuzzySpan, fuzzy_score
from flint.text.utf8 import decode_utf8, encode_utf8
from flint.ui.animator import SpringConfig, UI_ANIM_INSTANT, UiAnimator
from flint.ui.draw_list import UI_FLAG_BORDER, UI_FLAG_TOP_BEVEL, UiDrawList
from flint.ui.flexbox_dock import RectF
from flint.ui.icons import draw_studio_icon
from flint.ui.theme import (
    ICON_CODE,
    ICON_CUBE,
    ICON_FILE,
    ICON_PLUS,
    ICON_SEARCH,
    ICON_SPARK,
    StudioTheme,
    lerp_rgba8,
    rgba8_scale_alpha,
    rgba8_with_alpha,
)

comptime PALETTE_ALL: Int = 0
comptime PALETTE_COMMANDS: Int = 1
comptime PALETTE_SPAWN: Int = 2
comptime PALETTE_ENTITIES: Int = 3
comptime PALETTE_ASSETS: Int = 4
comptime PALETTE_MODE_COUNT: Int = 5

comptime PALETTE_ITEM_COMMAND: Int = 1
comptime PALETTE_ITEM_SPAWN: Int = 2
comptime PALETTE_ITEM_ENTITY: Int = 3
comptime PALETTE_ITEM_ASSET: Int = 4
comptime PALETTE_ITEM_AGENT: Int = 5

comptime KEY_CHORD_DELAY_SEC: Float32 = 0.20


def palette_mode_label(mode: Int) -> String:
    if mode == PALETTE_COMMANDS:
        return "Commands"
    if mode == PALETTE_SPAWN:
        return "Quick Spawn"
    if mode == PALETTE_ENTITIES:
        return "Entities"
    if mode == PALETTE_ASSETS:
        return "Assets"
    return "All"


def palette_mode_prefix(mode: Int) -> String:
    if mode == PALETTE_COMMANDS:
        return ">"
    if mode == PALETTE_SPAWN:
        return "+"
    if mode == PALETTE_ENTITIES:
        return "@"
    if mode == PALETTE_ASSETS:
        return "#"
    return ""


def parse_palette_query(raw_query: String, fallback_mode: Int = PALETTE_ALL) -> Tuple[Int, String, Bool]:
    """Parses leading prefix (`>`, `+`, `@`, `#`) into `(mode, filter_text, had_prefix)`."""
    if raw_query.startswith(">"):
        return (PALETTE_COMMANDS, String(String(raw_query[byte=1:]).strip()), True)
    if raw_query.startswith("+"):
        return (PALETTE_SPAWN, String(String(raw_query[byte=1:]).strip()), True)
    if raw_query.startswith("@"):
        return (PALETTE_ENTITIES, String(String(raw_query[byte=1:]).strip()), True)
    if raw_query.startswith("#"):
        return (PALETTE_ASSETS, String(String(raw_query[byte=1:]).strip()), True)
    return (fallback_mode, String(raw_query.strip()), False)


struct PaletteItem(Copyable, Movable):
    """Single searchable entry registered into `StudioCommandPalette`."""
    var kind: Int
    var action_id: Int
    var title: String
    var subtitle: String
    var shortcut: String
    var icon_id: Int
    var frecency_key: String
    var mutates_scene: Bool
    var enabled: Bool
    var search_key: String

    def __init__(
        out self,
        kind: Int = PALETTE_ITEM_COMMAND,
        action_id: Int = 0,
        title: String = "",
        subtitle: String = "",
        shortcut: String = "",
        icon_id: Int = 0,
        frecency_key: String = "",
        mutates_scene: Bool = False,
        enabled: Bool = True,
        search_key: String = "",
    ):
        self.kind = kind
        self.action_id = action_id
        self.title = title
        self.subtitle = subtitle
        self.shortcut = shortcut
        self.icon_id = icon_id
        self.frecency_key = frecency_key if frecency_key.byte_length() > 0 else (String(kind) + ":" + title)
        self.mutates_scene = mutates_scene
        self.enabled = enabled
        self.search_key = search_key


struct PaletteResult(Copyable, Movable):
    """Ranked match row produced by `StudioCommandPalette.refilter()`."""
    var item_index: Int
    var score: Int
    var effective_enabled: Bool
    var disabled_reason: String
    var title_spans: List[FuzzySpan]
    var title_byte_spans: List[FuzzySpan]

    def __init__(
        out self,
        item_index: Int = -1,
        score: Int = 0,
        effective_enabled: Bool = True,
        disabled_reason: String = "",
    ):
        self.item_index = item_index
        self.score = score
        self.effective_enabled = effective_enabled
        self.disabled_reason = disabled_reason
        self.title_spans = List[FuzzySpan]()
        self.title_byte_spans = List[FuzzySpan]()

    def __init__(out self, *, copy: Self):
        self.item_index = copy.item_index
        self.score = copy.score
        self.effective_enabled = copy.effective_enabled
        self.disabled_reason = copy.disabled_reason
        self.title_spans = copy.title_spans.copy()
        self.title_byte_spans = copy.title_byte_spans.copy()


@fieldwise_init
struct PaletteActionResult(Copyable, Movable):
    """Outcome of a palette keyboard/mouse interaction."""
    var executed: Bool
    var closed: Bool
    var kind: Int
    var action_id: Int
    var title: String
    var frecency_key: String
    var secondary: Bool

    def __init__(out self):
        self.executed = False
        self.closed = False
        self.kind = 0
        self.action_id = -1
        self.title = ""
        self.frecency_key = ""
        self.secondary = False


def _item_matches_mode(item_kind: Int, mode: Int) -> Bool:
    if mode == PALETTE_ALL:
        return True
    if mode == PALETTE_COMMANDS:
        return item_kind == PALETTE_ITEM_COMMAND or item_kind == PALETTE_ITEM_AGENT
    if mode == PALETTE_SPAWN:
        return item_kind == PALETTE_ITEM_SPAWN
    if mode == PALETTE_ENTITIES:
        return item_kind == PALETTE_ITEM_ENTITY
    if mode == PALETTE_ASSETS:
        return item_kind == PALETTE_ITEM_ASSET
    return True


struct StudioCommandPalette(Copyable, Movable):
    """Fuzzy + frecency Command Palette & Quick-Spawn controller."""
    var is_open: Bool
    var mode: Int
    var base_mode: Int
    var raw_query: String
    var filter_query: String
    var pie_active: Bool
    var items: List[PaletteItem]
    var results: List[PaletteResult]
    var selected_idx: Int
    var scroll_top: Int
    var visible_rows: Int
    var frecency: FrecencyStore
    var now_sec: Int

    def __init__(out self, visible_rows: Int = 9):
        self.is_open = False
        self.mode = PALETTE_ALL
        self.base_mode = PALETTE_ALL
        self.raw_query = ""
        self.filter_query = ""
        self.pie_active = False
        self.items = List[PaletteItem]()
        self.results = List[PaletteResult]()
        self.selected_idx = 0
        self.scroll_top = 0
        self.visible_rows = max(1, visible_rows)
        self.frecency = FrecencyStore()
        self.now_sec = 0

    def __init__(out self, *, copy: Self):
        self.is_open = copy.is_open
        self.mode = copy.mode
        self.base_mode = copy.base_mode
        self.raw_query = copy.raw_query
        self.filter_query = copy.filter_query
        self.pie_active = copy.pie_active
        self.items = copy.items.copy()
        self.results = copy.results.copy()
        self.selected_idx = copy.selected_idx
        self.scroll_top = copy.scroll_top
        self.visible_rows = copy.visible_rows
        self.frecency = copy.frecency.copy()
        self.now_sec = copy.now_sec

    def open(
        mut self,
        mode: Int = PALETTE_ALL,
        initial_query: String = "",
        pie_active: Bool = False,
        now_sec: Int = 0,
    ):
        var clamped = max(PALETTE_ALL, min(PALETTE_MODE_COUNT - 1, mode))
        self.is_open = True
        self.base_mode = clamped
        self.mode = clamped
        self.pie_active = pie_active
        self.now_sec = max(0, now_sec)
        self.selected_idx = 0
        self.scroll_top = 0
        if initial_query.byte_length() > 0:
            self.set_query(initial_query)
        else:
            self.raw_query = palette_mode_prefix(clamped)
            self.filter_query = ""
            self.refilter()

    def close(mut self):
        self.is_open = False

    def set_pie_active(mut self, active: Bool):
        if self.pie_active != active:
            self.pie_active = active
            self.refilter()

    def set_now_sec(mut self, now_sec: Int):
        self.now_sec = max(0, now_sec)

    def clear_items(mut self):
        self.items.clear()
        self.results.clear()
        self.selected_idx = 0
        self.scroll_top = 0

    def add_item(mut self, var item: PaletteItem):
        self.items.append(item^)

    def set_query(mut self, query: String):
        self.raw_query = query
        var parsed = parse_palette_query(query, self.base_mode)
        self.mode = parsed[0]
        self.filter_query = parsed[1]
        self.refilter()

    def cycle_mode(mut self, reverse: Bool = False):
        """Cycles active filter tab (`Tab` / `Shift+Tab`) while preserving the filter text."""
        var next_m = (self.mode + PALETTE_MODE_COUNT + (-1 if reverse else 1)) % PALETTE_MODE_COUNT
        self.base_mode = next_m
        self.mode = next_m
        self.raw_query = palette_mode_prefix(next_m) + self.filter_query
        self.refilter()

    def refilter(mut self):
        """Re-scores all items for the active mode, query, PIE safety state, and frecency."""
        self.results.clear()
        var q = self.filter_query
        for i in range(len(self.items)):
            ref it = self.items[i]
            if not _item_matches_mode(it.kind, self.mode):
                continue

            var title_m = fuzzy_score(q, it.title)
            var matched = title_m.matched
            var base_score = title_m.score
            var use_title_spans = title_m.matched and q.byte_length() > 0

            if not matched:
                var fallback_text = (
                    it.search_key
                    if it.search_key.byte_length() > 0
                    else (it.title + " " + it.subtitle)
                )
                var fb_m = fuzzy_score(q, fallback_text)
                if fb_m.matched:
                    matched = True
                    # Slightly lower than direct title hits so title matches sort first
                    base_score = max(1, fb_m.score - 60)
                    use_title_spans = False

            if not matched:
                continue

            var pie_blocked = self.pie_active and it.mutates_scene
            var eff_enabled = it.enabled and (not pie_blocked)
            var reason = String("Stop PIE to edit") if pie_blocked else (String("Unavailable") if not it.enabled else String(""))

            var total_score = base_score
            if it.frecency_key.byte_length() > 0:
                total_score += self.frecency.score(it.frecency_key, self.now_sec)
                total_score += self.frecency.recency_bonus(it.frecency_key)
            if not eff_enabled:
                total_score -= 400

            var row = PaletteResult(i, total_score, eff_enabled, reason)
            if use_title_spans:
                row.title_spans = title_m.spans.copy()
                row.title_byte_spans = title_m.byte_spans.copy()

            # Stable descending insertion by score
            var pos = len(self.results)
            while pos > 0 and self.results[pos - 1].score < total_score:
                pos -= 1
            self.results.insert(pos, row^)

        self.selected_idx = 0
        self.scroll_top = 0
        for i in range(len(self.results)):
            if self.results[i].effective_enabled:
                self.selected_idx = i
                self._sync_scroll()
                break

    def _sync_scroll(mut self):
        var n = len(self.results)
        if n == 0:
            self.selected_idx = 0
            self.scroll_top = 0
            return
        self.selected_idx = max(0, min(n - 1, self.selected_idx))
        var max_top = max(0, n - self.visible_rows)
        if self.selected_idx < self.scroll_top:
            self.scroll_top = self.selected_idx
        elif self.selected_idx >= self.scroll_top + self.visible_rows:
            self.scroll_top = self.selected_idx - self.visible_rows + 1
        self.scroll_top = max(0, min(max_top, self.scroll_top))

    def move_selection(mut self, delta: Int):
        var n = len(self.results)
        if n == 0:
            self.selected_idx = 0
            self.scroll_top = 0
            return
        self.selected_idx = (self.selected_idx + delta) % n
        if self.selected_idx < 0:
            self.selected_idx += n
        self._sync_scroll()

    def page_selection(mut self, delta_pages: Int):
        var n = len(self.results)
        if n == 0:
            return
        var step = max(1, self.visible_rows - 1) * delta_pages
        self.selected_idx = max(0, min(n - 1, self.selected_idx + step))
        self._sync_scroll()

    def accept_index(mut self, result_idx: Int, secondary: Bool = False) -> PaletteActionResult:
        """Executes `results[result_idx]` if enabled, records frecency, and closes the palette."""
        var out = PaletteActionResult()
        if result_idx < 0 or result_idx >= len(self.results):
            return out^
        ref r = self.results[result_idx]
        if not r.effective_enabled or r.item_index < 0 or r.item_index >= len(self.items):
            return out^
        ref it = self.items[r.item_index]
        if it.frecency_key.byte_length() > 0:
            self.frecency.note(it.frecency_key, self.now_sec)
        out.executed = True
        out.closed = True
        out.kind = it.kind
        out.action_id = it.action_id
        out.title = it.title
        out.frecency_key = it.frecency_key
        out.secondary = secondary
        self.is_open = False
        return out^

    def handle_key(
        mut self,
        key_up: Bool = False,
        key_down: Bool = False,
        key_pgup: Bool = False,
        key_pgdn: Bool = False,
        key_tab: Bool = False,
        key_shift: Bool = False,
        key_enter: Bool = False,
        key_esc: Bool = False,
    ) -> PaletteActionResult:
        """Processes navigation & acceptance keys while the palette is open."""
        var out = PaletteActionResult()
        if not self.is_open:
            return out^
        if key_esc:
            self.close()
            out.closed = True
            return out^
        if key_tab:
            self.cycle_mode(reverse=key_shift)
            return out^
        if key_up:
            self.move_selection(-1)
        elif key_down:
            self.move_selection(1)
        elif key_pgup:
            self.page_selection(-1)
        elif key_pgdn:
            self.page_selection(1)
        if key_enter:
            return self.accept_index(self.selected_idx, secondary=key_shift)
        return out^

    def selected_result(self) -> Optional[PaletteResult]:
        if self.selected_idx >= 0 and self.selected_idx < len(self.results):
            return Optional[PaletteResult](self.results[self.selected_idx].copy())
        return None


# ============================================================================
# Non-Blocking Key-Chord Hint Overlay State (`KeyChordHud`)
# ============================================================================

struct KeyChordEntry(Copyable, Movable):
    """Single follow-up binding displayed in `KeyChordHud`."""
    var second_key: String
    var label: String
    var action_id: Int
    var toggle_state: Int  # -1 = not a toggle, 0 = off, 1 = on

    def __init__(
        out self,
        second_key: String = "",
        label: String = "",
        action_id: Int = 0,
        toggle_state: Int = -1,
    ):
        self.second_key = second_key
        self.label = label
        self.action_id = action_id
        self.toggle_state = toggle_state


struct KeyChordHud(Copyable, Movable):
    """200 ms delayed non-focus-stealing chord hint box state."""
    var active: Bool
    var visible: Bool
    var prefix_label: String
    var elapsed_sec: Float32
    var delay_sec: Float32
    var entries: List[KeyChordEntry]

    def __init__(out self, delay_sec: Float32 = KEY_CHORD_DELAY_SEC):
        self.active = False
        self.visible = False
        self.prefix_label = ""
        self.elapsed_sec = Float32(0.0)
        self.delay_sec = max(Float32(0.0), delay_sec)
        self.entries = List[KeyChordEntry]()

    def __init__(out self, *, copy: Self):
        self.active = copy.active
        self.visible = copy.visible
        self.prefix_label = copy.prefix_label
        self.elapsed_sec = copy.elapsed_sec
        self.delay_sec = copy.delay_sec
        self.entries = copy.entries.copy()

    def begin_chord(mut self, prefix_label: String, var entries: List[KeyChordEntry]):
        self.active = True
        self.visible = self.delay_sec <= Float32(0.0)
        self.prefix_label = prefix_label
        self.elapsed_sec = Float32(0.0)
        self.entries = entries^

    def cancel(mut self):
        self.active = False
        self.visible = False
        self.prefix_label = ""
        self.elapsed_sec = Float32(0.0)
        self.entries.clear()

    def tick(mut self, dt: Float32):
        if not self.active:
            return
        if dt > Float32(0.0):
            self.elapsed_sec += dt
        if self.elapsed_sec >= self.delay_sec:
            self.visible = True

    def resolve_key(mut self, second_key: String) -> Int:
        """Completes the chord if `second_key` matches (case-insensitive ASCII); cancels chord either way."""
        if not self.active:
            return -1
        var target = _ascii_lower_str(second_key)
        var found = -1
        for i in range(len(self.entries)):
            if _ascii_lower_str(self.entries[i].second_key) == target:
                found = self.entries[i].action_id
                break
        self.cancel()
        return found

    def formatted_entry(self, idx: Int) -> String:
        if idx < 0 or idx >= len(self.entries):
            return ""
        ref e = self.entries[idx]
        var suffix = String(": on") if e.toggle_state == 1 else (String(": off") if e.toggle_state == 0 else String(""))
        return e.second_key + "  " + e.label + suffix


def _ascii_lower_str(s: String) -> String:
    var src = s.as_bytes()
    var out = List[UInt8](capacity=len(src))
    for i in range(len(src)):
        var b = src[i]
        if b >= 65 and b <= 90:
            out.append(b + 32)
        else:
            out.append(b)
    return String(unsafe_from_utf8=out)


# ============================================================================
# GPU SDF Rendering (`StudioTheme` + `add_glass_panel` + `draw_studio_icon` + `UiAnimator`)
# ============================================================================


def _pal_measure_text(font: SdfFont, text: String, size_px: Float32) -> Float32:
    if font.valid:
        return font.measure(text, size_px)
    return Float32(len(decode_utf8(text))) * size_px * Float32(0.56)


def _pal_draw_text(
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
                rgba8_scale_alpha(color, Float32(0.45)),
                Float32(2.0),
            )


def _pal_truncate(font: SdfFont, text: String, size_px: Float32, max_w: Float32) -> String:
    if max_w <= Float32(6.0):
        return ""
    if _pal_measure_text(font, text, size_px) <= max_w:
        return text
    var codes = decode_utf8(text)
    var ell = String("...")
    var ell_w = _pal_measure_text(font, ell, size_px)
    var keep = len(codes)
    while keep > 0:
        var sub = encode_utf8(codes, 0, keep)
        if _pal_measure_text(font, sub, size_px) + ell_w <= max_w:
            return sub + ell
        keep -= 1
    return ell


def _pal_default_icon(item: PaletteItem) -> Int:
    if item.icon_id > 0:
        return item.icon_id
    if item.kind == PALETTE_ITEM_SPAWN:
        return ICON_PLUS
    if item.kind == PALETTE_ITEM_ENTITY:
        return ICON_CUBE
    if item.kind == PALETTE_ITEM_ASSET:
        return ICON_FILE
    if item.kind == PALETTE_ITEM_AGENT:
        return ICON_SPARK
    return ICON_CODE


def _pal_draw_highlighted_title(
    mut dl: UiDrawList,
    font_slot: Int,
    font: SdfFont,
    title: String,
    byte_spans: List[FuzzySpan],
    x: Float32,
    y: Float32,
    size_px: Float32,
    base_col: UInt32,
    hi_col: UInt32,
):
    if len(byte_spans) == 0:
        _pal_draw_text(dl, font_slot, font, title, x, y, size_px, base_col)
        return
    var n_bytes = title.byte_length()
    var pen = x
    var cursor = 0
    for si in range(len(byte_spans)):
        var s0 = max(cursor, min(n_bytes, byte_spans[si].start))
        var s1 = max(s0, min(n_bytes, byte_spans[si].end))
        if s0 > cursor:
            var seg = String(title[byte=cursor:s0])
            _pal_draw_text(dl, font_slot, font, seg, pen, y, size_px, base_col)
            pen += _pal_measure_text(font, seg, size_px)
        if s1 > s0:
            var hseg = String(title[byte=s0:s1])
            _pal_draw_text(dl, font_slot, font, hseg, pen, y, size_px, hi_col)
            pen += _pal_measure_text(font, hseg, size_px)
        cursor = s1
    if cursor < n_bytes:
        var tail = String(title[byte=cursor:n_bytes])
        _pal_draw_text(dl, font_slot, font, tail, pen, y, size_px, base_col)


def draw_command_palette(
    mut palette: StudioCommandPalette,
    mut dl: UiDrawList,
    font_slot: Int,
    font: SdfFont,
    theme: StudioTheme,
    mut anim: UiAnimator,
    screen_w: Float32,
    screen_h: Float32,
    mouse_x: Float32,
    mouse_y: Float32,
    mouse_down: Bool,
    mouse_pressed: Bool,
    key_shift: Bool = False,
) -> PaletteActionResult:
    """Renders the frosted-glass Command Palette modal with animated reveal, sliding mode tabs,
    matched-span highlighting, shortcut keycaps, and mouse click selection."""
    var action = PaletteActionResult()
    var target_open = Float32(1.0) if palette.is_open else Float32(0.0)
    var reveal = anim.drive_aux(UInt64(0x50414C4554544501), target_open, SpringConfig.snappy())
    if anim.mode == UI_ANIM_INSTANT:
        reveal = target_open
    if (not palette.is_open) and reveal <= Float32(0.01):
        return action^

    var sw = max(Float32(360.0), screen_w if isfinite(screen_w) else Float32(1280.0))
    var sh = max(Float32(240.0), screen_h if isfinite(screen_h) else Float32(720.0))
    var alpha = max(Float32(0.0), min(Float32(1.0), reveal))

    # Full-screen translucent scrim
    dl.add_rect(Float32(0.0), Float32(0.0), sw, sh, rgba8_scale_alpha(theme.l4_scrim, alpha))

    var pal_w = min(Float32(560.0), sw - Float32(40.0))
    var n_res = len(palette.results)
    var shown_rows = max(1, min(palette.visible_rows, n_res))
    var row_h = Float32(34.0)
    var header_h = Float32(44.0)
    var tabs_h = Float32(30.0)
    var footer_h = Float32(26.0)
    var pal_h = header_h + tabs_h + Float32(shown_rows) * row_h + footer_h + Float32(12.0)

    var lift_y = (Float32(1.0) - alpha) * Float32(-14.0)
    var px = (sw - pal_w) * Float32(0.5)
    var py = max(Float32(48.0), sh * Float32(0.14)) + lift_y
    var pal_r = RectF(px, py, pal_w, pal_h)

    if palette.is_open and mouse_pressed and (not pal_r.contains(mouse_x, mouse_y)):
        palette.close()
        action.closed = True
        return action^

    # Main frosted-glass palette container
    dl.add_glass_panel(
        pal_r.x, pal_r.y, pal_r.w, pal_r.h,
        rgba8_scale_alpha(theme.l4_overlay, alpha),
        theme.radius_lg,
        Float32(4.0),
        Float32(1.0),
        rgba8_scale_alpha(theme.border_strong, alpha),
        True,
        True,
    )

    # ---- Search Input Row ----
    var search_y = pal_r.y + Float32(10.0)
    draw_studio_icon(dl, ICON_SEARCH, pal_r.x + Float32(14.0), search_y + Float32(5.0), Float32(16.0), rgba8_scale_alpha(theme.accent_primary, alpha), Float32(1.6))
    var q_x = pal_r.x + Float32(38.0)
    if palette.raw_query.byte_length() > 0:
        _pal_draw_text(dl, font_slot, font, palette.raw_query, q_x, search_y + Float32(4.0), Float32(14.5), rgba8_scale_alpha(theme.text_primary, alpha))
        var qw = _pal_measure_text(font, palette.raw_query, Float32(14.5))
        var blink_on = (anim.mode == UI_ANIM_INSTANT) or ((Int(anim.time_sec * Float32(2.0)) % 2) == 0)
        if blink_on:
            dl.add_rect(q_x + qw + Float32(2.0), search_y + Float32(4.0), Float32(1.8), Float32(17.0), rgba8_scale_alpha(theme.accent_primary, alpha), Float32(0.8))
    else:
        _pal_draw_text(
            dl, font_slot, font,
            "Type command, +spawn, @entity, #asset...",
            q_x, search_y + Float32(4.5), Float32(13.5),
            rgba8_scale_alpha(theme.text_muted, alpha),
        )
    # ESC keycap pill
    var esc_r = RectF(pal_r.right() - Float32(44.0), search_y + Float32(3.0), Float32(32.0), Float32(20.0))
    dl.add_rect(esc_r.x, esc_r.y, esc_r.w, esc_r.h, rgba8_scale_alpha(theme.l2_card, alpha), Float32(4.0), Float32(1.0), rgba8_scale_alpha(theme.border_subtle, alpha))
    _pal_draw_text(dl, font_slot, font, "ESC", esc_r.x + Float32(6.0), esc_r.y + Float32(4.0), Float32(10.0), rgba8_scale_alpha(theme.text_muted, alpha))
    if palette.is_open and esc_r.contains(mouse_x, mouse_y) and mouse_pressed:
        palette.close()
        action.closed = True
        return action^

    dl.add_line(pal_r.x + Float32(1.0), pal_r.y + header_h, pal_r.right() - Float32(1.0), pal_r.y + header_h, Float32(1.0), rgba8_scale_alpha(theme.border_subtle, alpha))

    # ---- Mode Filter Tabs with Sliding Pill ----
    var tab_y = pal_r.y + header_h + Float32(4.0)
    var tab_x = pal_r.x + Float32(12.0)
    var tab_xs = List[Float32](capacity=PALETTE_MODE_COUNT)
    var tab_ws = List[Float32](capacity=PALETTE_MODE_COUNT)
    var tgt_tx = tab_x
    var tgt_tw = Float32(42.0)
    for mi in range(PALETTE_MODE_COUNT):
        var pref = palette_mode_prefix(mi)
        var lbl = (pref + " " + palette_mode_label(mi)) if pref.byte_length() > 0 else palette_mode_label(mi)
        var tw = max(Float32(40.0), _pal_measure_text(font, lbl, Float32(11.5)) + Float32(16.0))
        tab_xs.append(tab_x)
        tab_ws.append(tw)
        if mi == palette.mode:
            tgt_tx = tab_x
            tgt_tw = tw
        tab_x += tw + Float32(4.0)

    var ind = anim.drive_tab_pill(UInt64(0x50414C4D4F4445), tgt_tx, tgt_tw)
    dl.add_rect_styled(
        ind[0], tab_y, ind[1], Float32(22.0),
        rgba8_scale_alpha(theme.accent_soft, alpha),
        rgba8_scale_alpha(theme.accent_primary, alpha * Float32(0.75)),
        Float32(5.0), Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
    )
    for mi in range(PALETTE_MODE_COUNT):
        var pref = palette_mode_prefix(mi)
        var lbl = (pref + " " + palette_mode_label(mi)) if pref.byte_length() > 0 else palette_mode_label(mi)
        var tr = RectF(tab_xs[mi], tab_y, tab_ws[mi], Float32(22.0))
        var thov = tr.contains(mouse_x, mouse_y)
        var is_act = mi == palette.mode
        _pal_draw_text(
            dl, font_slot, font, lbl,
            tr.x + Float32(8.0), tr.y + Float32(4.5), Float32(11.5),
            rgba8_scale_alpha(theme.text_primary if (is_act or thov) else theme.text_secondary, alpha),
        )
        if palette.is_open and thov and mouse_pressed:
            palette.base_mode = mi
            palette.mode = mi
            palette.raw_query = palette_mode_prefix(mi) + palette.filter_query
            palette.refilter()

    # ---- Results List ----
    var list_y = pal_r.y + header_h + tabs_h + Float32(4.0)
    if n_res == 0:
        _pal_draw_text(
            dl, font_slot, font,
            "No matching commands, entities, or assets",
            pal_r.x + Float32(16.0), list_y + Float32(9.0), Float32(12.5),
            rgba8_scale_alpha(theme.text_muted, alpha),
        )
    else:
        var start_i = max(0, min(palette.scroll_top, max(0, n_res - shown_rows)))
        var end_i = min(n_res, start_i + shown_rows)
        for ri in range(start_i, end_i):
            ref res = palette.results[ri]
            if res.item_index < 0 or res.item_index >= len(palette.items):
                continue
            ref it = palette.items[res.item_index]
            var ry = list_y + Float32(ri - start_i) * row_h
            var rr = RectF(pal_r.x + Float32(8.0), ry, pal_r.w - Float32(16.0), row_h - Float32(2.0))
            var is_sel = ri == palette.selected_idx
            var rhov = palette.is_open and rr.contains(mouse_x, mouse_y)
            var r_anim = anim.drive_widget(UInt64(0x50414C524F5700) ^ UInt64(ri), rhov, rhov and mouse_down, False)
            if is_sel:
                dl.add_rect(rr.x, rr.y, rr.w, rr.h, rgba8_scale_alpha(theme.accent_soft, alpha), theme.radius_sm)
                dl.add_rect(rr.x, rr.y + Float32(5.0), Float32(2.5), rr.h - Float32(10.0), rgba8_scale_alpha(theme.accent_primary, alpha), Float32(1.2))
            elif rhov or r_anim[0] > Float32(0.01):
                dl.add_rect(rr.x, rr.y, rr.w, rr.h, rgba8_scale_alpha(theme.l3_hover, alpha * Float32(0.55) * r_anim[0]), theme.radius_sm)

            var icon_col = (
                theme.agent_active
                if it.kind == PALETTE_ITEM_AGENT
                else (theme.accent_primary if is_sel else theme.text_secondary)
            )
            if not res.effective_enabled:
                icon_col = theme.text_muted
            draw_studio_icon(dl, _pal_default_icon(it), rr.x + Float32(10.0), rr.center_y() - Float32(7.0), Float32(14.0), rgba8_scale_alpha(icon_col, alpha), Float32(1.4))

            # Right-side badge: shortcut keycap or disabled reason
            var right_edge = rr.right() - Float32(10.0)
            if not res.effective_enabled and res.disabled_reason.byte_length() > 0:
                var dr_w = _pal_measure_text(font, res.disabled_reason, Float32(10.5)) + Float32(12.0)
                var dr_x = right_edge - dr_w
                dl.add_rect(dr_x, rr.center_y() - Float32(9.0), dr_w, Float32(18.0), rgba8_scale_alpha(rgba8_with_alpha(theme.status_warn, 36), alpha), Float32(4.0))
                _pal_draw_text(dl, font_slot, font, res.disabled_reason, dr_x + Float32(6.0), rr.center_y() - Float32(5.5), Float32(10.5), rgba8_scale_alpha(theme.status_warn, alpha))
                right_edge = dr_x - Float32(8.0)
            elif it.shortcut.byte_length() > 0:
                var sc_w = _pal_measure_text(font, it.shortcut, Float32(10.5)) + Float32(12.0)
                var sc_x = right_edge - sc_w
                dl.add_rect(sc_x, rr.center_y() - Float32(9.0), sc_w, Float32(18.0), rgba8_scale_alpha(theme.l2_card, alpha), Float32(4.0), Float32(1.0), rgba8_scale_alpha(theme.border_subtle, alpha))
                _pal_draw_text(dl, font_slot, font, it.shortcut, sc_x + Float32(6.0), rr.center_y() - Float32(5.5), Float32(10.5), rgba8_scale_alpha(theme.text_secondary, alpha))
                right_edge = sc_x - Float32(8.0)

            var title_x = rr.x + Float32(32.0)
            var title_base_col = rgba8_scale_alpha(theme.text_primary if res.effective_enabled else theme.text_muted, alpha)
            var title_hi_col = rgba8_scale_alpha(theme.accent_primary, alpha)
            _pal_draw_highlighted_title(
                dl, font_slot, font, it.title, res.title_byte_spans,
                title_x, rr.center_y() - Float32(7.0), Float32(13.0),
                title_base_col, title_hi_col,
            )
            if it.subtitle.byte_length() > 0:
                var sub_x = title_x + _pal_measure_text(font, it.title, Float32(13.0)) + Float32(10.0)
                var avail_sub = right_edge - sub_x
                if avail_sub > Float32(32.0):
                    var sub_disp = _pal_truncate(font, it.subtitle, Float32(11.5), avail_sub)
                    _pal_draw_text(dl, font_slot, font, sub_disp, sub_x, rr.center_y() - Float32(6.0), Float32(11.5), rgba8_scale_alpha(theme.text_muted, alpha))

            if rhov:
                palette.selected_idx = ri
                if mouse_pressed:
                    return palette.accept_index(ri, secondary=key_shift)

    # ---- Footer Hint Bar ----
    var fy = pal_r.bottom() - footer_h
    dl.add_line(pal_r.x + Float32(1.0), fy, pal_r.right() - Float32(1.0), fy, Float32(1.0), rgba8_scale_alpha(theme.border_subtle, alpha))
    _pal_draw_text(
        dl, font_slot, font,
        "Up/Down Navigate   Tab Mode   Enter Run   Shift+Enter Focus",
        pal_r.x + Float32(14.0), fy + Float32(6.5), Float32(11.0),
        rgba8_scale_alpha(theme.text_muted, alpha),
    )
    return action^


def draw_key_chord_hud(
    hud: KeyChordHud,
    mut dl: UiDrawList,
    font_slot: Int,
    font: SdfFont,
    theme: StudioTheme,
    mut anim: UiAnimator,
    vp: RectF,
):
    """Renders the delayed non-focus-stealing `KeyChordHud` box in the bottom-left viewport corner."""
    var target_v = Float32(1.0) if (hud.active and hud.visible and len(hud.entries) > 0) else Float32(0.0)
    var alpha = anim.drive_aux(UInt64(0x4B455943484F5244), target_v, SpringConfig.snappy())
    if anim.mode == UI_ANIM_INSTANT:
        alpha = target_v
    if target_v <= Float32(0.0) and alpha <= Float32(0.01):
        return
    if len(hud.entries) == 0:
        return

    var n = len(hud.entries)
    var row_h = Float32(22.0)
    var box_w = Float32(248.0)
    var box_h = Float32(28.0) + Float32(n) * row_h + Float32(8.0)
    var bx = vp.x + Float32(12.0)
    var by = vp.bottom() - box_h - Float32(12.0) + (Float32(1.0) - alpha) * Float32(8.0)

    dl.add_glass_panel(
        bx, by, box_w, box_h,
        rgba8_scale_alpha(theme.l4_overlay, alpha),
        theme.radius_md, Float32(3.0), Float32(1.0),
        rgba8_scale_alpha(theme.border_strong, alpha), True, True,
    )
    _pal_draw_text(
        dl, font_slot, font,
        hud.prefix_label + " ...  (Esc cancels)",
        bx + Float32(10.0), by + Float32(7.0), Float32(11.5),
        rgba8_scale_alpha(theme.accent_primary, alpha),
    )
    dl.add_line(bx + Float32(1.0), by + Float32(26.0), bx + box_w - Float32(1.0), by + Float32(26.0), Float32(1.0), rgba8_scale_alpha(theme.border_subtle, alpha))
    for i in range(n):
        ref e = hud.entries[i]
        var ry = by + Float32(30.0) + Float32(i) * row_h
        dl.add_rect(bx + Float32(10.0), ry + Float32(2.0), Float32(26.0), Float32(18.0), rgba8_scale_alpha(theme.l2_card, alpha), Float32(4.0), Float32(1.0), rgba8_scale_alpha(theme.border_control, alpha))
        _pal_draw_text(dl, font_slot, font, e.second_key, bx + Float32(16.0), ry + Float32(5.0), Float32(11.0), rgba8_scale_alpha(theme.accent_primary, alpha))
        _pal_draw_text(dl, font_slot, font, e.label, bx + Float32(44.0), ry + Float32(4.5), Float32(11.5), rgba8_scale_alpha(theme.text_primary, alpha))
        if e.toggle_state == 0 or e.toggle_state == 1:
            var dot_col = theme.status_ok if e.toggle_state == 1 else theme.text_muted
            dl.add_circle(bx + box_w - Float32(16.0), ry + Float32(11.0), Float32(3.5), rgba8_scale_alpha(dot_col, alpha))
