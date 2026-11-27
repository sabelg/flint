"""Retain-Cached Immediate-Mode GPU UI Context, Themes & Studio Widgets (`flint.ui.context`).

Emits 100% GPU `UiDrawList` vector primitives (`0` CPU pixel loops!):
  - `UiTheme`: `dark_studio()` & `game_hud()` presets.
  - `begin_box` / `end_box` backed by retain-cached two-pass `FlexTree` layout.
  - `begin_dock_panel` / `end_dock_panel` with interactive tab bars & draggable splitters.
  - `begin_scroll_view` / `end_scroll_view` with O(1) offscreen row/subtree culling
    (100,000-item `tree_node` outliners execute in `< 0.1 ms`!).
  - `label`, `rich_label`, `button`, `checkbox`, `slider_float`, `drag_float3`,
    UTF-8 multi-byte-safe `text_input`, `tree_node` / `tree_node_indexed`,
    `curve_editor` (`add_quad_bezier` + `add_line`), `DragDrop`, and focus-trapping
    `begin_modal` / `end_modal` with `Tab` & Gamepad D-Pad navigation.
"""
from std.collections import List, Dict
from std.math import abs, floor, isfinite, isinf, isnan
from flint.core.math import Vec3
from flint.platform.input import (
    InputState,
    PAD_A,
    PAD_B,
    PAD_UP,
    PAD_DOWN,
    PAD_LEFT,
    PAD_RIGHT,
)
from flint.render.font import SdfFont
from flint.text.utf8 import decode_utf8, encode_utf8
from flint.text.bidi import BIDI_AUTO, BIDI_LTR
from flint.text.shaper import ShapedGlyph
from flint.text.rich_text import parse_rich, layout_rich, layout_rich_shaped, SPAN_BUTTON
from flint.ui.animator import (
    SpringConfig,
    UI_ANIM_INSTANT,
    UI_ANIM_NORMAL,
    UI_ANIM_REDUCED,
    UiAnimator,
    lerp_rgba8,
)
from flint.ui.button_glyphs import pad_button, face_color
from flint.ui.draw_list import (
    UiDrawList,
    GpuUiPrim,
    PRIM_GLYPH,
    UI_FLAG_BORDER,
    UI_FLAG_DROP_SHADOW,
    UI_FLAG_OUTER_GLOW,
    UI_FLAG_PER_CORNER_RADII,
    UI_FLAG_TOP_BEVEL,
    UI_FLAG_VERT_GRADIENT,
)
from flint.ui.icons import draw_studio_chevron, draw_studio_icon
from flint.ui.theme import (
    CARD_HEADER_H,
    mix_rgba8,
    INSPECTOR_LABEL_RATIO,
    INSPECTOR_ROW_H,
    RADIUS_MD,
    RADIUS_SM,
    StudioTheme,
    rgba8_scale_alpha,
    rgba8_with_alpha,
)
from flint.ui.flexbox_dock import (
    RectF,
    FlexStyle,
    FlexTree,
    FLEX_ROW,
    FLEX_COL,
    DockSpace,
    DOCK_SPLIT_H,
    DOCK_SPLIT_V,
    DOCK_TABS,
    DOCK_CENTRAL_VIEWPORT,
)

# Flint Keyboard Keycodes (matching platform/x11.mojo & ui/widgets.mojo)
comptime UI_KEY_SPACE: Int = 32
comptime UI_KEY_A: Int = 65
comptime UI_KEY_C: Int = 67
comptime UI_KEY_V: Int = 86
comptime UI_KEY_X: Int = 88
comptime UI_KEY_ESCAPE: Int = 256
comptime UI_KEY_ENTER: Int = 257
comptime UI_KEY_TAB: Int = 258
comptime UI_KEY_BACKSPACE: Int = 259
comptime UI_KEY_DELETE: Int = 261
comptime UI_KEY_RIGHT: Int = 262
comptime UI_KEY_LEFT: Int = 263
comptime UI_KEY_DOWN: Int = 264
comptime UI_KEY_UP: Int = 265
comptime UI_KEY_HOME: Int = 268
comptime UI_KEY_END: Int = 269
comptime UI_KEY_LEFT_SHIFT: Int = 340
comptime UI_KEY_LEFT_CTRL: Int = 341
comptime UI_KEY_RIGHT_SHIFT: Int = 344
comptime UI_KEY_RIGHT_CTRL: Int = 345


@always_inline
def pack_rgba8(r: Int, g: Int, b: Int, a: Int = 255) -> UInt32:
    """Packs `[0..255]` channels into `0xAARRGGBB` matching `aaa_ui_overlay_shader`."""
    var cr = UInt32(max(0, min(255, r)))
    var cg = UInt32(max(0, min(255, g)))
    var cb = UInt32(max(0, min(255, b)))
    var ca = UInt32(max(0, min(255, a)))
    return cb | (cg << 8) | (cr << 16) | (ca << 24)


def format_float2(v: Float32) -> String:
    """Formats a `Float32` with 2 fixed decimal places (`0.05` -> `"0.05"`, `-0.5` -> `"-0.50"`)."""
    if isnan(v):
        return "nan"
    if isinf(v):
        return "-inf" if v < Float32(0.0) else "inf"
    var neg = v < Float32(0.0)
    var mag = -v if neg else v
    if abs(Float64(mag)) >= 1.0e15:
        return ("-" if neg else "") + String(mag)
    var scaled = Int(mag * Float32(100.0) + Float32(0.5))
    if scaled == 0:
        neg = False
    var whole = scaled // 100
    var frac = scaled % 100
    var sign = "-" if neg else ""
    var fstr = ("0" + String(frac)) if frac < 10 else String(frac)
    return sign + String(whole) + "." + fstr


@always_inline
def utf8_drop_last_char(text: String) -> String:
    """Returns `text` without its last UTF-8 code point (Backspace), never splitting a multibyte sequence."""
    var n = text.byte_length()
    if n == 0:
        return String("")
    var bytes = text.as_bytes()
    var cut = n - 1
    # Step back over continuation bytes (10xxxxxx), at most 3 of them.
    while cut > 0 and n - cut < 4 and (Int(bytes[cut]) & 0xC0) == 0x80:
        cut -= 1
    return String(text[byte=0:cut])


def hash_ui_id(parent_seed: UInt64, text: String) -> UInt64:
    """FNV-1a 64-bit hierarchical widget ID hash (non-zero)."""
    var h = parent_seed if parent_seed != 0 else UInt64(0xCBF29CE484222325)
    for b in text.as_bytes():
        h ^= UInt64(b)
        h *= UInt64(0x100000001B3)
    return h if h != 0 else UInt64(1)


@fieldwise_init
struct UiTheme(Copyable, ImplicitlyCopyable, Movable):
    """Visual styling tokens for `UiContext`."""
    var bg_window: UInt32
    var bg_panel: UInt32
    var bg_header: UInt32
    var bg_widget: UInt32
    var bg_widget_hover: UInt32
    var bg_widget_active: UInt32
    var accent: UInt32
    var accent_dim: UInt32
    var border: UInt32
    var focus_ring: UInt32
    var text_primary: UInt32
    var text_muted: UInt32
    var modal_backdrop: UInt32
    var font_size: Float32
    var row_height: Float32
    var corner_radius: Float32
    var scrollbar_width: Float32

    @staticmethod
    def dark_studio() -> Self:
        return Self(
            pack_rgba8(20, 22, 27, 255),     # bg_window
            pack_rgba8(26, 29, 36, 255),     # bg_panel
            pack_rgba8(34, 38, 47, 255),     # bg_header
            pack_rgba8(40, 45, 56, 255),     # bg_widget
            pack_rgba8(54, 61, 76, 255),     # bg_widget_hover
            pack_rgba8(68, 78, 98, 255),     # bg_widget_active
            pack_rgba8(62, 142, 247, 255),   # accent
            pack_rgba8(42, 96, 172, 255),    # accent_dim
            pack_rgba8(58, 64, 78, 255),     # border
            pack_rgba8(98, 178, 255, 255),   # focus_ring
            pack_rgba8(232, 236, 242, 255),  # text_primary
            pack_rgba8(146, 155, 170, 255),  # text_muted
            pack_rgba8(8, 10, 14, 190),      # modal_backdrop
            15.0,                            # font_size
            24.0,                            # row_height
            4.0,                             # corner_radius
            10.0,                            # scrollbar_width
        )

    @staticmethod
    def game_hud() -> Self:
        return Self(
            pack_rgba8(12, 20, 28, 215),     # bg_window
            pack_rgba8(16, 28, 38, 225),     # bg_panel
            pack_rgba8(24, 42, 54, 240),     # bg_header
            pack_rgba8(28, 48, 62, 235),     # bg_widget
            pack_rgba8(40, 68, 86, 245),     # bg_widget_hover
            pack_rgba8(54, 92, 114, 255),    # bg_widget_active
            pack_rgba8(78, 226, 200, 255),   # accent
            pack_rgba8(46, 148, 132, 255),   # accent_dim
            pack_rgba8(68, 112, 128, 255),   # border
            pack_rgba8(255, 211, 112, 255),  # focus_ring
            pack_rgba8(240, 248, 246, 255),  # text_primary
            pack_rgba8(162, 194, 190, 255),  # text_muted
            pack_rgba8(4, 10, 16, 205),      # modal_backdrop
            18.0,                            # font_size
            30.0,                            # row_height
            6.0,                             # corner_radius
            10.0,                            # scrollbar_width
        )


@fieldwise_init
struct RetainedWidgetState(Copyable, ImplicitlyCopyable, Movable):
    """Per-widget retained state across frames."""
    var cursor_cp: Int       # Unicode codepoint cursor index (NEVER mid-UTF8 byte!)
    var select_anchor_cp: Int
    var scroll_y: Float32
    var scroll_max_y: Float32
    var tree_open: Bool
    var tree_initialized: Bool
    var drag_start_val: Float32
    var scroll_x: Float32
    var last_click_time: Float32
    var last_click_cp: Int
    var drag_selecting: Bool

    def __init__(out self):
        self.cursor_cp = 0
        self.select_anchor_cp = -1
        self.scroll_y = 0.0
        self.scroll_max_y = 0.0
        self.tree_open = False
        self.tree_initialized = False
        self.drag_start_val = 0.0
        self.scroll_x = 0.0
        self.last_click_time = -10.0
        self.last_click_cp = -1
        self.drag_selecting = False


@fieldwise_init
struct DragDropPayload(Copyable, ImplicitlyCopyable, Movable):
    var active: Bool
    var kind: String
    var int_value: Int
    var str_value: String
    var source_id: UInt64
    var delivered: Bool

    def __init__(out self):
        self.active = False
        self.kind = ""
        self.int_value = 0
        self.str_value = ""
        self.source_id = 0
        self.delivered = False


@fieldwise_init
struct _BoxScope(Copyable, ImplicitlyCopyable, Movable):
    var id_hash: UInt64
    var flex_idx: Int
    var rect: RectF
    var cursor_x: Float32
    var cursor_y: Float32
    var direction: Int
    var gap: Float32
    var pad_r: Float32
    var pad_b: Float32
    var max_x: Float32
    var max_y: Float32


comptime UI_DRAG_START_PX: Float32 = 4.0  # pointer travel from the press before a drag begins


struct _TextCacheEntry(Copyable, Movable):
    """One measured/shaped `(text, px)` pair, reused while it keeps being drawn."""
    var text: String
    var px: Float32
    var width: Float32
    var has_width: Bool
    var has_glyphs: Bool
    var glyphs: List[ShapedGlyph]
    var epoch: UInt64

    def __init__(out self, text: String, px: Float32, epoch: UInt64):
        self.text = text
        self.px = px
        self.width = 0.0
        self.has_width = False
        self.has_glyphs = False
        self.glyphs = List[ShapedGlyph]()
        self.epoch = epoch


@fieldwise_init
struct _ScrollScope(Copyable, ImplicitlyCopyable, Movable):
    var id_hash: UInt64
    var flex_idx: Int
    var view_rect: RectF
    var start_cursor_y: Float32
    var scroll_y: Float32
    var explicit_content_h: Float32


struct UiContext(Movable):
    """Retain-cached immediate-mode GPU UI & Studio Docking Context."""
    var draw_list: UiDrawList
    var font: SdfFont
    var display_font: SdfFont
    var font_slot: Int
    var theme: UiTheme
    # Retain-cached two-pass Flexbox layout
    var flex: FlexTree
    var cached_rects: Dict[UInt64, RectF]
    var cached_rect_epochs: Dict[UInt64, UInt64]
    var next_frame_ids: List[UInt64]
    var next_frame_flex_idxs: List[Int]
    var dock_flex_root_idxs: List[Int]
    var dock_flex_root_rects: List[RectF]
    var box_stack: List[_BoxScope]
    var id_stack: List[UInt64]
    var scroll_stack: List[_ScrollScope]
    # Retained per-widget state
    var state_map: Dict[UInt64, RetainedWidgetState]
    var state_epochs: Dict[UInt64, UInt64]
    var open_tree_rows: Dict[Int, Bool]  # Fast int-keyed open set for 100k virtualized outliners
    var open_tree_row_epochs: Dict[Int, UInt64]  # last frame each `open_tree_rows` entry was read (GC)
    var frame_epoch: UInt64
    var gc_stale_frames: UInt64
    # Pointer, keyboard & gamepad frame snapshot
    var screen_w: Float32
    var screen_h: Float32
    var dt: Float32
    var mouse_x: Float32
    var mouse_y: Float32
    var mouse_down_l: Bool
    var mouse_pressed_l: Bool
    var mouse_released_l: Bool
    var mouse_delta_x: Float32
    var mouse_delta_y: Float32
    var mouse_wheel: Float32
    var text_utf8: String
    var key_shift: Bool
    var key_ctrl: Bool
    var key_tab_pressed: Bool
    var key_enter_pressed: Bool
    var key_space_pressed: Bool
    var key_esc_pressed: Bool
    var key_bs_pressed: Bool
    var key_del_pressed: Bool
    var key_left_pressed: Bool
    var key_right_pressed: Bool
    var key_up_pressed: Bool
    var key_down_pressed: Bool
    var key_home_pressed: Bool
    var key_end_pressed: Bool
    var key_a_pressed: Bool
    var key_c_pressed: Bool
    var key_v_pressed: Bool
    var key_x_pressed: Bool
    var pad_nav_next: Bool
    var pad_nav_prev: Bool
    var pad_confirm: Bool
    var pad_cancel: Bool
    var pad_sony: Bool
    var from_pad: Bool
    # Clipboard, IME & Timing (M4b F26)
    var ui_time_sec: Float32
    var clipboard_text: String
    var clipboard_out: String
    var clipboard_out_dirty: Bool
    var ime_active: Bool
    var ime_preedit: String
    var ime_cursor_cp: Int
    var ime_sel_len: Int
    var ime_commit: String
    var ime_widget_id: UInt64
    var ime_widget_name: String
    var _prev_input_ime_active: Bool
    var _ime_suppressed: Bool
    var _ime_widget_seen: Bool
    var _cancelled_ime_preedit: String
    var _cancelled_ime_cursor_cp: Int
    var _cancelled_ime_sel_len: Int
    # Interaction & Modal Focus Trap
    var hovered_id: UInt64
    var active_id: UInt64
    var focused_id: UInt64
    var last_item_id: UInt64
    var last_item_rect: RectF
    var bg_focusables: List[UInt64]
    var modal_focusables: List[UInt64]
    var modal_depth: Int
    var had_modal_prev_frame: Bool
    var active_modal_id: UInt64
    var drag_drop: DragDropPayload
    var culled_items_this_frame: Int
    # Input capture & deferred popups (M0 F09)
    var _focused_text_id: UInt64
    var _focused_text_seen: Bool
    var _text_widget_ids: List[UInt64]
    var hovered_rect: RectF
    var _dock_splitter_active: Bool
    var _open_popup_id: UInt64
    var _popup_seen: Bool
    var _popup_kind: Int
    var _popup_rect: RectF
    var _popup_consumed_click: Bool
    var _popup_consumed_rect: RectF
    var _popup_combo_items: List[String]
    var _popup_combo_sel: Int
    var _popup_combo_hover: Int
    var _popup_color: Vec3
    var _in_frame: Bool
    var _keyboard_capture: Bool
    # Studio v2 Design System, Micro-Animation & CSS :focus-visible
    var studio_theme: StudioTheme
    var animator: UiAnimator
    var focus_visible: Bool
    var dock_style_v2: Bool
    # Text measure/shape cache: `(text, px)` -> width + shaped glyphs, kept while used
    var _tc_index: Dict[UInt64, Int]
    var _tc_entries: List[_TextCacheEntry]
    var _tc_font_key: UInt64
    var text_cache_hits: Int
    var text_cache_misses: Int
    var _ui_theme_synced: Bool
    # Press bookkeeping: drag sources and popup dismissal
    var _press_x: Float32
    var _press_y: Float32
    var _drag_pending: DragDropPayload
    var _popup_owner_rect: RectF
    var _popup_swallowed_press: Bool
    var _popup_combo_owner: UInt64
    var _wheel_consumed: Bool
    var _wheel_target: UInt64
    var _wheel_target_next: UInt64
    var _wheel_tentative: UInt64
    var _wheel_tentative_sy: Float32

    def __init__(out self, preset_hud: Bool = False):
        self.draw_list = UiDrawList()
        self.font = SdfFont()
        self.display_font = SdfFont()
        self.font_slot = 0
        self.theme = UiTheme.game_hud() if preset_hud else UiTheme.dark_studio()
        self.flex = FlexTree()
        self.cached_rects = Dict[UInt64, RectF]()
        self.cached_rect_epochs = Dict[UInt64, UInt64]()
        self.next_frame_ids = List[UInt64]()
        self.next_frame_flex_idxs = List[Int]()
        self.dock_flex_root_idxs = List[Int]()
        self.dock_flex_root_rects = List[RectF]()
        self.box_stack = List[_BoxScope]()
        self.id_stack = List[UInt64]()
        self.scroll_stack = List[_ScrollScope]()
        self.state_map = Dict[UInt64, RetainedWidgetState]()
        self.state_epochs = Dict[UInt64, UInt64]()
        self.open_tree_rows = Dict[Int, Bool]()
        self.open_tree_row_epochs = Dict[Int, UInt64]()
        self.frame_epoch = 0
        self.gc_stale_frames = 120
        self.screen_w = 1280.0
        self.screen_h = 720.0
        self.dt = 1.0 / 60.0
        self.mouse_x = -10000.0
        self.mouse_y = -10000.0
        self.mouse_down_l = False
        self.mouse_pressed_l = False
        self.mouse_released_l = False
        self.mouse_delta_x = 0.0
        self.mouse_delta_y = 0.0
        self.mouse_wheel = 0.0
        self.text_utf8 = ""
        self.key_shift = False
        self.key_ctrl = False
        self.key_tab_pressed = False
        self.key_enter_pressed = False
        self.key_space_pressed = False
        self.key_esc_pressed = False
        self.key_bs_pressed = False
        self.key_del_pressed = False
        self.key_left_pressed = False
        self.key_right_pressed = False
        self.key_up_pressed = False
        self.key_down_pressed = False
        self.key_home_pressed = False
        self.key_end_pressed = False
        self.key_a_pressed = False
        self.key_c_pressed = False
        self.key_v_pressed = False
        self.key_x_pressed = False
        self.pad_nav_next = False
        self.pad_nav_prev = False
        self.pad_confirm = False
        self.pad_cancel = False
        self.pad_sony = False
        self.from_pad = False
        self.ui_time_sec = 0.0
        self.clipboard_text = ""
        self.clipboard_out = ""
        self.clipboard_out_dirty = False
        self.ime_active = False
        self.ime_preedit = ""
        self.ime_cursor_cp = 0
        self.ime_sel_len = 0
        self.ime_commit = ""
        self.ime_widget_id = 0
        self.ime_widget_name = ""
        self._prev_input_ime_active = False
        self._ime_suppressed = False
        self._ime_widget_seen = False
        self._cancelled_ime_preedit = ""
        self._cancelled_ime_cursor_cp = 0
        self._cancelled_ime_sel_len = 0
        self.hovered_id = 0
        self.active_id = 0
        self.focused_id = 0
        self.last_item_id = 0
        self.last_item_rect = RectF()
        self.bg_focusables = List[UInt64]()
        self.modal_focusables = List[UInt64]()
        self.modal_depth = 0
        self.had_modal_prev_frame = False
        self.active_modal_id = 0
        self.drag_drop = DragDropPayload()
        self.culled_items_this_frame = 0
        self._focused_text_id = 0
        self._focused_text_seen = False
        self._text_widget_ids = List[UInt64]()
        self.hovered_rect = RectF()
        self._dock_splitter_active = False
        self._open_popup_id = 0
        self._popup_seen = False
        self._popup_kind = 0
        self._popup_rect = RectF()
        self._popup_consumed_click = False
        self._popup_consumed_rect = RectF()
        self._popup_combo_items = List[String]()
        self._popup_combo_sel = -1
        self._popup_combo_hover = -1
        self._popup_color = Vec3()
        self._in_frame = False
        self._keyboard_capture = False
        self.studio_theme = StudioTheme()
        self.animator = UiAnimator(UI_ANIM_NORMAL)
        self.focus_visible = False
        self.dock_style_v2 = False
        self._tc_index = Dict[UInt64, Int]()
        self._tc_entries = List[_TextCacheEntry]()
        self._tc_font_key = 0
        self.text_cache_hits = 0
        self.text_cache_misses = 0
        self._ui_theme_synced = False
        self._press_x = -10000.0
        self._press_y = -10000.0
        self._drag_pending = DragDropPayload()
        self._popup_owner_rect = RectF()
        self._popup_swallowed_press = False
        self._popup_combo_owner = 0
        self._wheel_consumed = False
        self._wheel_target = 0
        self._wheel_target_next = 0
        self._wheel_tentative = 0
        self._wheel_tentative_sy = 0.0

    def set_studio_theme(mut self, st: StudioTheme):
        """Switches the active Studio v2 design-system theme (syncs reduced-motion preference and
        derives the legacy `UiTheme` colors, so M0 widgets follow the theme). Re-applying the
        current theme is a cheap comparison, not a copy."""
        if st.reduced_motion and self.animator.mode == UI_ANIM_NORMAL:
            self.animator.set_mode(UI_ANIM_REDUCED)
        elif (not st.reduced_motion) and self.animator.mode == UI_ANIM_REDUCED:
            self.animator.set_mode(UI_ANIM_NORMAL)
        if self._ui_theme_synced and self.studio_theme.same_tokens(st):
            return
        self.studio_theme = st.copy()
        self.sync_ui_theme_from_studio()

    def sync_ui_theme_from_studio(mut self):
        """Derives the legacy `UiTheme` colors from `studio_theme` (metrics are left alone)."""
        ref st = self.studio_theme
        self.theme.bg_window = st.l0_canvas
        self.theme.bg_panel = st.l1_shell
        self.theme.bg_header = st.l2_card
        self.theme.bg_widget = st.l3_raised
        self.theme.bg_widget_hover = st.l3_hover
        self.theme.bg_widget_active = mix_rgba8(st.accent_primary, st.l3_hover, 28)
        self.theme.accent = st.accent_primary
        self.theme.accent_dim = mix_rgba8(st.accent_primary, st.l2_card, 50) if st.is_dark else mix_rgba8(st.accent_primary, st.l3_raised, 26)
        self.theme.border = st.border_strong
        self.theme.focus_ring = st.focus_ring
        self.theme.text_primary = st.text_primary
        self.theme.text_muted = st.text_muted
        self.theme.modal_backdrop = st.l4_scrim
        self._ui_theme_synced = True

    def set_anim_mode(mut self, mode: Int):
        """Sets micro-animation mode (`UI_ANIM_NORMAL`, `UI_ANIM_REDUCED`, or `UI_ANIM_INSTANT`)."""
        self.animator.set_mode(mode)

    @always_inline
    def is_animating(self) -> Bool:
        """Returns `True` while any micro-animation spring, toast, or trail requires redraw."""
        return self.animator.is_animating()

    @always_inline
    def show_focus_ring(self, wid: UInt64) -> Bool:
        """CSS `:focus-visible` query: `True` only when `wid` is focused via keyboard/gamepad navigation."""
        return (not self._blocked_by_modal()) and self.focused_id == wid and self.focus_visible

    # ---- Frame Lifecycle ----

    def begin_frame(
        mut self,
        input: InputState,
        screen_w: Float32 = 1280.0,
        screen_h: Float32 = 720.0,
        dt: Float32 = 1.0 / 60.0,
    ):
        self._in_frame = True
        self.frame_epoch += 1
        self.draw_list.clear()
        self.flex.clear()
        self.next_frame_ids.clear()
        self.next_frame_flex_idxs.clear()
        self.dock_flex_root_idxs.clear()
        self.dock_flex_root_rects.clear()
        self.box_stack.clear()
        self.id_stack.clear()
        self.scroll_stack.clear()
        self.bg_focusables.clear()
        self.modal_focusables.clear()
        self.modal_depth = 0
        self.hovered_id = 0
        self.last_item_id = 0
        self.culled_items_this_frame = 0
        self._ime_widget_seen = False
        self._focused_text_seen = False
        self._text_widget_ids.clear()
        self.hovered_rect = RectF()
        self._dock_splitter_active = False
        self._popup_seen = False
        self._popup_consumed_click = False
        self._popup_consumed_rect = RectF()
        self._popup_swallowed_press = False
        self._wheel_consumed = False
        self._wheel_target = self._wheel_target_next
        self._wheel_target_next = 0
        self._wheel_tentative = 0
        self._sweep_text_cache()

        var next_sw = max(Float32(1.0), screen_w if isfinite(screen_w) else Float32(1280.0))
        var next_sh = max(Float32(1.0), screen_h if isfinite(screen_h) else Float32(720.0))
        if next_sw != self.screen_w or next_sh != self.screen_h:
            self.cached_rects.clear()
            self.cached_rect_epochs.clear()
        self.screen_w = next_sw
        self.screen_h = next_sh
        if not isfinite(dt):
            self.dt = Float32(1.0 / 60.0)
        else:
            self.dt = max(Float32(1.0e-4), min(Float32(1.0), dt))
        if not isfinite(self.ui_time_sec):
            self.ui_time_sec = Float32(0.0)
        self.ui_time_sec += self.dt
        self.animator.begin_frame(self.dt)
        self.clipboard_out_dirty = False
        self.clipboard_out = ""

        if input.has_mouse_position:
            if isfinite(input.last_mouse_x):
                self.mouse_x = input.last_mouse_x
            if isfinite(input.last_mouse_y):
                self.mouse_y = input.last_mouse_y
        var prev_down = self.mouse_down_l
        self.mouse_down_l = input.mouse_down(0)
        self.mouse_pressed_l = input.mouse_pressed(0) or (self.mouse_down_l and not prev_down)
        self.mouse_released_l = (not self.mouse_down_l) and prev_down
        self.mouse_delta_x = input.mouse_delta_x() if isfinite(input.mouse_delta_x()) else Float32(0.0)
        self.mouse_delta_y = input.mouse_delta_y() if isfinite(input.mouse_delta_y()) else Float32(0.0)
        self.mouse_wheel = input.mouse_wheel() if isfinite(input.mouse_wheel()) else Float32(0.0)
        self.text_utf8 = input.text_input_utf8

        # A press outside an open popup (and outside the widget that owns it) only dismisses the
        # popup: it is swallowed, so it never also clicks whatever lies underneath.
        if (
            self.mouse_pressed_l
            and self._open_popup_id != 0
            and not self._popup_rect.contains(self.mouse_x, self.mouse_y)
            and not self._popup_owner_rect.contains(self.mouse_x, self.mouse_y)
        ):
            self._open_popup_id = 0
            self._popup_kind = 0
            self.mouse_pressed_l = False
            self._popup_swallowed_press = True

        # Drags start only from the item the press began on, after `UI_DRAG_START_PX` of travel.
        if self.mouse_pressed_l:
            self._press_x = self.mouse_x
            self._press_y = self.mouse_y
            self._drag_pending = DragDropPayload()
        elif not self.mouse_down_l:
            self._drag_pending = DragDropPayload()
        elif self._drag_pending.source_id != 0 and not self.drag_drop.active:
            var ddx = self.mouse_x - self._press_x
            var ddy = self.mouse_y - self._press_y
            if isfinite(ddx) and isfinite(ddy) and ddx * ddx + ddy * ddy >= UI_DRAG_START_PX * UI_DRAG_START_PX:
                self.drag_drop = self._drag_pending
                self.drag_drop.active = True
                self.drag_drop.delivered = False
                self._drag_pending = DragDropPayload()

        self.key_shift = input.key_down(UI_KEY_LEFT_SHIFT) or input.key_down(UI_KEY_RIGHT_SHIFT)
        self.key_ctrl = input.key_down(UI_KEY_LEFT_CTRL) or input.key_down(UI_KEY_RIGHT_CTRL)
        self.key_tab_pressed = input.key_pressed(UI_KEY_TAB)
        self.key_enter_pressed = input.key_pressed(UI_KEY_ENTER)
        self.key_space_pressed = input.key_pressed(UI_KEY_SPACE)
        self.key_esc_pressed = input.key_pressed(UI_KEY_ESCAPE)
        self.key_bs_pressed = input.key_pressed(UI_KEY_BACKSPACE)
        self.key_del_pressed = input.key_pressed(UI_KEY_DELETE)
        self.key_left_pressed = input.key_pressed(UI_KEY_LEFT)
        self.key_right_pressed = input.key_pressed(UI_KEY_RIGHT)
        self.key_up_pressed = input.key_pressed(UI_KEY_UP)
        self.key_down_pressed = input.key_pressed(UI_KEY_DOWN)
        self.key_home_pressed = input.key_pressed(UI_KEY_HOME)
        self.key_end_pressed = input.key_pressed(UI_KEY_END)
        self.key_a_pressed = input.key_pressed(UI_KEY_A)
        self.key_c_pressed = input.key_pressed(UI_KEY_C)
        self.key_v_pressed = input.key_pressed(UI_KEY_V)
        self.key_x_pressed = input.key_pressed(UI_KEY_X)

        if input.ime_commit.byte_length() > 0:
            self.ime_commit = input.ime_commit
            self.ime_active = False
            self.ime_preedit = ""
            self.ime_cursor_cp = 0
            self.ime_sel_len = 0
            self._ime_suppressed = False
        if input.ime_active:
            if self._ime_suppressed:
                if (
                    input.ime_preedit != self._cancelled_ime_preedit
                    or input.ime_cursor_cp != self._cancelled_ime_cursor_cp
                    or input.ime_sel_len != self._cancelled_ime_sel_len
                ):
                    self._ime_suppressed = False
            if not self._ime_suppressed:
                self.ime_active = True
                self.ime_preedit = input.ime_preedit
                self.ime_cursor_cp = input.ime_cursor_cp
                self.ime_sel_len = input.ime_sel_len
        else:
            self._ime_suppressed = False
            if self._prev_input_ime_active:
                self.ime_active = False
                self.ime_preedit = ""
                self.ime_cursor_cp = 0
                self.ime_sel_len = 0
                self.ime_widget_id = 0
                self.ime_widget_name = ""
        self._prev_input_ime_active = input.ime_active

        var pd_down = input.gamepad_pressed(0, PAD_DOWN) or input.gamepad_pressed(0, PAD_RIGHT)
        var pd_up = input.gamepad_pressed(0, PAD_UP) or input.gamepad_pressed(0, PAD_LEFT)
        self.pad_nav_next = pd_down
        self.pad_nav_prev = pd_up
        self.pad_confirm = input.gamepad_pressed(0, PAD_A)
        self.pad_cancel = input.gamepad_pressed(0, PAD_B)
        self.pad_sony = input.pad_is_sony()
        if pd_down or pd_up or self.pad_confirm or self.pad_cancel:
            self.from_pad = True
        elif input.any_key_pressed() or self.mouse_pressed_l:
            self.from_pad = False

        # CSS `:focus-visible`: show keyboard/gamepad focus ring only after keyboard/pad navigation
        if (
            self.key_tab_pressed
            or self.key_up_pressed
            or self.key_down_pressed
            or self.key_left_pressed
            or self.key_right_pressed
            or pd_down
            or pd_up
        ):
            self.focus_visible = True
        elif self.mouse_pressed_l:
            self.focus_visible = False

        # Push root full-screen container box
        var root_style = FlexStyle.col(gap=4.0, pad=0.0)
        root_style.fixed_w = self.screen_w
        root_style.fixed_h = self.screen_h
        var f_idx = self.flex.add_node(-1, root_style, self.screen_w, self.screen_h)
        var root_rect = RectF(0.0, 0.0, self.screen_w, self.screen_h)
        self.box_stack.append(_BoxScope(UInt64(1), f_idx, root_rect, 0.0, 0.0, FLEX_COL, 4.0, 0.0, 0.0, 0.0, 0.0))
        self.id_stack.append(UInt64(0xCBF29CE484222325))

    def end_frame(mut self):
        # Solve full FlexTree (+ detached DockSpace panel roots) and retain-cache all solved RectFs for next frame
        if len(self.flex.nodes) > 0:
            self.flex.solve(RectF(0.0, 0.0, self.screen_w, self.screen_h), 0)
            for d_i in range(len(self.dock_flex_root_idxs)):
                self.flex.solve(self.dock_flex_root_rects[d_i], self.dock_flex_root_idxs[d_i])
            for i in range(len(self.next_frame_ids)):
                var wid = self.next_frame_ids[i]
                var fi = self.next_frame_flex_idxs[i]
                if fi >= 0 and fi < len(self.flex.nodes):
                    self.cached_rects[wid] = self.flex.nodes[fi].rect
                    self.cached_rect_epochs[wid] = self.frame_epoch

        # Frame-epoch GC for unused `cached_rects` and `state_map` entries
        var sweep_stride = max(UInt64(1), min(UInt64(30), self.gc_stale_frames))
        if self.frame_epoch > self.gc_stale_frames and (self.frame_epoch % sweep_stride) == 0:
            var cutoff = self.frame_epoch - self.gc_stale_frames
            var stale_rects = List[UInt64]()
            for item in self.cached_rect_epochs.items():
                if item.value < cutoff:
                    stale_rects.append(item.key)
            for k in stale_rects:
                try:
                    _ = self.cached_rects.pop(k)
                except:
                    pass
                try:
                    _ = self.cached_rect_epochs.pop(k)
                except:
                    pass

            var stale_states = List[UInt64]()
            for item in self.state_epochs.items():
                if item.value < cutoff:
                    stale_states.append(item.key)
            for k in stale_states:
                try:
                    _ = self.state_map.pop(k)
                except:
                    pass
                try:
                    _ = self.state_epochs.pop(k)
                except:
                    pass

            var stale_rows = List[Int]()
            for item in self.open_tree_row_epochs.items():
                if item.value < cutoff:
                    stale_rows.append(item.key)
            for k in stale_rows:
                try:
                    _ = self.open_tree_rows.pop(k)
                except:
                    pass
                try:
                    _ = self.open_tree_row_epochs.pop(k)
                except:
                    pass

        # Focus navigation (strictly trapped inside modal_focusables whenever a modal was active this frame!)
        var modal_was_open = len(self.modal_focusables) > 0 or self.active_modal_id != 0
        self.had_modal_prev_frame = modal_was_open

        if modal_was_open and len(self.modal_focusables) > 0:
            # Ensure focus cannot remain on any background widget outside the modal
            var in_modal = False
            for i in range(len(self.modal_focusables)):
                if self.modal_focusables[i] == self.focused_id:
                    in_modal = True
                    break
            if not in_modal:
                self.focused_id = self.modal_focusables[0]

        if self.ime_active and self.ime_widget_id != 0 and (not self._ime_widget_seen or self.focused_id != self.ime_widget_id):
            self.cancel_ime()

        var step_next = (self.key_tab_pressed and not self.key_shift) or self.pad_nav_next
        var step_prev = (self.key_tab_pressed and self.key_shift) or self.pad_nav_prev
        if (step_next or step_prev) and not self.ime_active:
            ref ring = self.modal_focusables if modal_was_open else self.bg_focusables
            var n = len(ring)
            if n > 0:
                var cur_idx = -1
                for i in range(n):
                    if ring[i] == self.focused_id:
                        cur_idx = i
                        break
                if cur_idx < 0:
                    self.focused_id = ring[0] if step_next else ring[n - 1]
                elif step_next:
                    self.focused_id = ring[(cur_idx + 1) % n]
                else:
                    self.focused_id = ring[(cur_idx - 1 + n) % n]

        if self.mouse_pressed_l and self._focused_text_id != 0 and self.active_id != self._focused_text_id:
            if self.focused_id == self._focused_text_id:
                self.focused_id = 0
            self._focused_text_id = 0
        var prev_focused_text_id = self._focused_text_id
        var resolved_text_id = UInt64(0)
        if self.focused_id != 0:
            for ti in range(len(self._text_widget_ids)):
                if self._text_widget_ids[ti] == self.focused_id:
                    resolved_text_id = self.focused_id
                    break
        if prev_focused_text_id != 0 and resolved_text_id == 0 and self.focused_id == prev_focused_text_id:
            self.focused_id = 0
        self._focused_text_id = resolved_text_id

        if not self._popup_seen:
            self._open_popup_id = 0
            self._popup_kind = 0
        if self._open_popup_id != 0 and self._popup_kind != 0:
            self._draw_deferred_popup()

        # Render DragDrop floating badge if dragging
        if self.drag_drop.active:
            if not self.mouse_down_l:
                self.drag_drop = DragDropPayload()
            elif isfinite(self.mouse_x) and isfinite(self.mouse_y):
                var badge_txt = self.drag_drop.kind + ": " + self.drag_drop.str_value
                var bw = max(Float32(90.0), self._measure_text(badge_txt, self.theme.font_size) + Float32(18.0))
                var bh = self.theme.row_height
                var bx = self.mouse_x + Float32(12.0)
                var by = self.mouse_y + Float32(10.0)
                self.draw_list.add_rect_styled(
                    bx, by, bw, bh,
                    self.theme.bg_header,
                    self.theme.accent,
                    self.theme.corner_radius,
                    1.5,
                    1.0,
                    UI_FLAG_BORDER | UI_FLAG_DROP_SHADOW,
                )
                self._draw_text(badge_txt, bx + Float32(8.0), by + Float32(4.0), self.theme.font_size, self.theme.text_primary)

        if not self.mouse_down_l and self.active_id != 0:
            # Keep text_input focused via focused_id while releasing mouse-drag active_id
            self.active_id = 0

        if self.ime_active and self.ime_widget_id != 0 and self.focused_id != self.ime_widget_id:
            self.cancel_ime()
        self.ime_commit = ""
        self.active_modal_id = 0
        self._in_frame = False

    # ---- Input Capture Barrier & Deferred Popups ----

    def set_keyboard_capture(mut self, active: Bool):
        """Explicitly enables or disables keyboard capture (e.g., keybind rebind listening)."""
        self._keyboard_capture = active

    @always_inline
    def is_in_frame(self) -> Bool:
        return self._in_frame

    def wants_keyboard(self) -> Bool:
        """Returns `True` when UI captures keyboard input (focused editable text input,
        active IME, open modal dialog, open popup dropdown, or explicit keyboard capture).
        Ignores auto-focused buttons."""
        return (
            self._focused_text_id != 0
            or self.ime_active
            or self.modal_depth > 0
            or self.had_modal_prev_frame
            or self._open_popup_id != 0
            or self._keyboard_capture
        )

    def wants_mouse(self) -> Bool:
        """Returns `True` when UI captures mouse/pointer input."""
        return (
            self.hovered_id != 0
            or self.active_id != 0
            or self.modal_depth > 0
            or self.had_modal_prev_frame
            or self.drag_drop.active
            or self._open_popup_id != 0
            or self._popup_swallowed_press
            or self._dock_splitter_active
        )

    def wants_mouse_at(self, mx: Float32, my: Float32) -> Bool:
        """Frame-boundary-safe mouse capture query using current pointer `(mx, my)`."""
        if (
            self.active_id != 0
            or self.modal_depth > 0
            or self.had_modal_prev_frame
            or self.drag_drop.active
            or self._open_popup_id != 0
            or self._popup_swallowed_press
            or self._dock_splitter_active
        ):
            return True
        if self.hovered_id != 0:
            return self.hovered_rect.contains(mx, my)
        return False

    def _draw_deferred_popup(mut self):
        var r = self._popup_rect
        if r.w <= Float32(0.0) or r.h <= Float32(0.0):
            return
        self.draw_list.add_rect_styled(
            r.x, r.y, r.w, r.h,
            self.theme.bg_panel,
            self.theme.accent,
            self.theme.corner_radius,
            1.5,
            1.0,
            UI_FLAG_BORDER | UI_FLAG_DROP_SHADOW,
        )
        if self._popup_kind == 1:
            var row_h = self.theme.row_height
            for i in range(len(self._popup_combo_items)):
                var iy = r.y + Float32(2.0) + Float32(i) * row_h
                var ir = RectF(r.x + Float32(2.0), iy, max(Float32(1.0), r.w - Float32(4.0)), row_h)
                if i == self._popup_combo_sel:
                    self.draw_list.add_rect(ir.x, ir.y, ir.w, ir.h, self.theme.accent_dim, 2.0)
                elif i == self._popup_combo_hover:
                    self.draw_list.add_rect(ir.x, ir.y, ir.w, ir.h, self.theme.bg_widget_hover, 2.0)
                var item_txt = self._popup_combo_items[i]
                self._draw_text(
                    item_txt,
                    ir.x + Float32(6.0),
                    ir.y + Float32(3.0),
                    self.theme.font_size * Float32(0.92),
                    self.theme.text_primary,
                )
        elif self._popup_kind == 2:
            var row_h = self.theme.row_height
            var pad = Float32(6.0)
            var bar_w = max(Float32(12.0), r.w - pad * Float32(2.0))
            var ch_cols = Array[UInt32, 3](fill=0)
            ch_cols[0] = pack_rgba8(224, 82, 82, 255)
            ch_cols[1] = pack_rgba8(92, 198, 108, 255)
            ch_cols[2] = pack_rgba8(78, 148, 240, 255)
            var ch_vals = Array[Float32, 3](fill=0.0)
            ch_vals[0] = max(Float32(0.0), min(Float32(1.0), self._popup_color.x))
            ch_vals[1] = max(Float32(0.0), min(Float32(1.0), self._popup_color.y))
            ch_vals[2] = max(Float32(0.0), min(Float32(1.0), self._popup_color.z))
            for c in range(3):
                var by = r.y + pad + Float32(c) * row_h + Float32(2.0)
                var bh = max(Float32(6.0), row_h - Float32(6.0))
                self.draw_list.add_rect(r.x + pad, by, bar_w, bh, self.theme.bg_widget, 3.0, 1.0, self.theme.border)
                var fw = max(Float32(2.0), bar_w * ch_vals[c])
                self.draw_list.add_rect(r.x + pad, by, fw, bh, ch_cols[c], 3.0)

    # ---- Internal Layout & State Helpers ----

    @always_inline
    def _current_seed(self) -> UInt64:
        return self.id_stack[len(self.id_stack) - 1] if len(self.id_stack) > 0 else UInt64(0xCBF29CE484222325)

    @always_inline
    def widget_id(self, name: String) -> UInt64:
        return hash_ui_id(self._current_seed(), name)

    @always_inline
    def _blocked_by_modal(self) -> Bool:
        return self.had_modal_prev_frame and self.modal_depth == 0

    @always_inline
    def _pointer_in_scroll_clip(self) -> Bool:
        for i in range(len(self.scroll_stack)):
            if not self.scroll_stack[i].view_rect.contains(self.mouse_x, self.mouse_y):
                return False
        return True

    @always_inline
    def _clip_hovered_rect(self, r: RectF) -> RectF:
        var out = r
        for i in range(len(self.scroll_stack)):
            var vr = self.scroll_stack[i].view_rect
            var x0 = max(out.x, vr.x)
            var y0 = max(out.y, vr.y)
            var x1 = min(out.right(), vr.right())
            var y1 = min(out.bottom(), vr.bottom())
            out = RectF(x0, y0, max(Float32(0.0), x1 - x0), max(Float32(0.0), y1 - y0))
        return out

    @always_inline
    def _is_pointer_blocked_by_popup(self, wid: UInt64 = 0) -> Bool:
        if not self._pointer_in_scroll_clip():
            return True
        if (
            self._open_popup_id != 0
            and self._open_popup_id != wid
            and self._popup_rect.contains(self.mouse_x, self.mouse_y)
        ):
            return True
        if self._popup_consumed_click and self._popup_consumed_rect.contains(self.mouse_x, self.mouse_y):
            return True
        return False

    def _set_focused(mut self, wid: UInt64):
        if self.ime_active and self.ime_widget_id != 0 and self.ime_widget_id != wid:
            self.cancel_ime()
        self.focused_id = wid

    def _register_focusable(mut self, wid: UInt64, allow_auto_focus: Bool = True):
        if wid == self.ime_widget_id:
            self._ime_widget_seen = True
        if self.modal_depth > 0:
            self.modal_focusables.append(wid)
        else:
            self.bg_focusables.append(wid)
        if allow_auto_focus and self.focused_id == 0 and not self._blocked_by_modal():
            self.focused_id = wid

    def _get_state(mut self, wid: UInt64) -> RetainedWidgetState:
        self.state_epochs[wid] = self.frame_epoch
        var found = self.state_map.get(wid)
        if found:
            return found.value()
        var fresh = RetainedWidgetState()
        self.state_map[wid] = fresh
        return fresh

    # ---- Text measure/shape cache ----
    #
    # Widgets measure a string and then draw it, many times per frame, and mostly the same
    # strings every frame. Each `(text, px)` is shaped once and kept while it keeps being used
    # (entries idle for two frames are swept); a change of font clears the cache. Shaping is
    # deterministic in `(font, text, px)`, so cached results equal fresh ones.

    @always_inline
    def _uses_shaping(self) -> Bool:
        return self.font.valid and (self.font.has_source or len(self.font.fallbacks) > 0)

    def _text_font_key(self) -> UInt64:
        var h = self.font._shaper_fingerprint()
        h = (h ^ UInt64(self.font.atlas_width)) * UInt64(0x100000001B3)
        h = (h ^ UInt64(len(self.font.glyphs) if self.font.baked_count == 0 else self.font.baked_count)) * UInt64(0x100000001B3)
        if isfinite(self.font.line_height):
            h = (h ^ UInt64(Int(self.font.line_height * Float32(64.0)))) * UInt64(0x100000001B3)
        if isfinite(self.font.ascent):
            h = (h ^ UInt64(Int(self.font.ascent * Float32(64.0)))) * UInt64(0x100000001B3)
        return h

    def _sync_text_cache(mut self):
        self.font.sync_shaper()
        var fk = self._text_font_key()
        if fk != self._tc_font_key:
            self._tc_index.clear()
            self._tc_entries.clear()
            self._tc_font_key = fk

    def _sweep_text_cache(mut self):
        """Drops entries unused for two frames (compacting at most every 30 frames)."""
        if len(self._tc_entries) == 0:
            return
        if len(self._tc_entries) < 4096 and (self.frame_epoch % 30) != 0:
            return
        var cutoff = self.frame_epoch - min(self.frame_epoch, UInt64(2))
        var kept = List[_TextCacheEntry](capacity=len(self._tc_entries))
        var index = Dict[UInt64, Int]()
        for i in range(len(self._tc_entries)):
            if self._tc_entries[i].epoch >= cutoff:
                index[Self._text_key(self._tc_entries[i].text, self._tc_entries[i].px)] = len(kept)
                kept.append(self._tc_entries[i].copy())
        self._tc_entries = kept^
        self._tc_index = index^

    @staticmethod
    @always_inline
    def _text_key(text: String, px: Float32) -> UInt64:
        var pk = UInt64(Int(px * Float32(4096.0))) if isfinite(px) else UInt64(0x7FF)
        return hash_ui_id(pk * UInt64(0x9E3779B97F4A7C15) + UInt64(0x51ED), text)

    def _text_entry(mut self, text: String, size_px: Float32) -> Int:
        """Index of the cache entry for `(text, size_px)` with its width (and, for a shaping font,
        its shaped glyphs) filled in, or -1 on a hash collision (the caller then computes directly)."""
        self._sync_text_cache()
        var key = Self._text_key(text, size_px)
        var idx: Int
        var found = self._tc_index.get(key)
        if found:
            idx = found.value()
            if idx < 0 or idx >= len(self._tc_entries) or self._tc_entries[idx].px != size_px or self._tc_entries[idx].text != text:
                return -1
            self.text_cache_hits += 1
        else:
            idx = len(self._tc_entries)
            self._tc_entries.append(_TextCacheEntry(text, size_px, self.frame_epoch))
            self._tc_index[key] = idx
            self.text_cache_misses += 1
        self._tc_entries[idx].epoch = self.frame_epoch
        if self._uses_shaping():
            if not self._tc_entries[idx].has_glyphs:
                var shaped = self.font.shape_text(text, size_px, BIDI_AUTO)
                var total: Float32 = 0.0
                for sg in shaped:
                    total += sg.x_advance
                self._tc_entries[idx].glyphs = shaped^
                self._tc_entries[idx].has_glyphs = True
                self._tc_entries[idx].width = total
                self._tc_entries[idx].has_width = True
        elif not self._tc_entries[idx].has_width:
            self._tc_entries[idx].width = self.font.measure(text, size_px)
            self._tc_entries[idx].has_width = True
        return idx

    def _measure_text(mut self, text: String, size_px: Float32) -> Float32:
        if self.font.valid:
            if text.byte_length() == 0:
                return 0.0
            var idx = self._text_entry(text, size_px)
            if idx >= 0:
                return self._tc_entries[idx].width
            if self._uses_shaping():
                return self.font.measure_shaped(text, size_px)
            return self.font.measure(text, size_px)
        return Float32(len(decode_utf8(text))) * size_px * Float32(0.56)

    @staticmethod
    def _emit_shaped_glyphs(
        mut draw_list: UiDrawList,
        font_slot: Int,
        mut font: SdfFont,
        shaped: List[ShapedGlyph],
        x: Float32,
        baseline: Float32,
        size_px: Float32,
        color: UInt32,
    ):
        var scale = size_px / (font.bake_px if font.bake_px > 0.0 else Float32(1.0))
        var sdf_to_screen = font.spread * scale / Float32(127.0)
        var aw = Float32(font.atlas_width if font.atlas_width > 0 else 1)
        var ah = Float32(font.atlas_height if font.atlas_height > 0 else 1)
        var c = draw_list.current_clip()
        var clip0 = UInt32(c[0]) | (UInt32(c[1]) << 16)
        var clip1 = UInt32(c[2]) | (UInt32(c[3]) << 16)
        var flags: UInt32 = (UInt32(1 << 6) if font.channels >= 3 else UInt32(0)) | (UInt32(1 << 7) if font.channels >= 4 else UInt32(0))
        var hdr: UInt32 = (PRIM_GLYPH & 0xF) | (UInt32(font_slot & 0xFFF) << 4) | (flags << 16)
        var pen = x
        for sg in shaped:
            var g_idx = font.ensure_glyph_id(sg.glyph_id, sg.font_index, sg.codepoint)
            if g_idx < 0:
                g_idx = font.glyph_for(sg.codepoint)
            if g_idx >= 0 and g_idx < len(font.glyphs):
                ref g = font.glyphs[g_idx]
                if g.w > 0 and g.h > 0:
                    var gx0 = pen + g.xoff * scale + sg.x_offset
                    var gy0 = baseline + g.yoff * scale - sg.y_offset
                    var gx1 = gx0 + Float32(g.w) * scale
                    var gy1 = gy0 + Float32(g.h) * scale
                    var u0 = Float32(g.x) / aw
                    var v0 = Float32(g.y) / ah
                    var u1 = Float32(g.x + g.w) / aw
                    var v1 = Float32(g.y + g.h) / ah
                    draw_list.prims.append(GpuUiPrim(
                        gx0, gy0, gx1, gy1,
                        u0, v0, u1, v1,
                        clip0, clip1,
                        color, 0,
                        sdf_to_screen, 0.0, 0.0, hdr,
                    ))
            pen += sg.x_advance

    def _draw_text(mut self, text: String, x: Float32, y: Float32, size_px: Float32, color: UInt32):
        if self.font.valid:
            var baseline = y + size_px * Float32(0.82)
            if self.font.has_source or len(self.font.fallbacks) > 0:
                if text.byte_length() == 0:
                    return
                var idx = self._text_entry(text, size_px)
                if idx >= 0:
                    Self._emit_shaped_glyphs(self.draw_list, self.font_slot, self.font, self._tc_entries[idx].glyphs, x, baseline, size_px, color)
                else:
                    var shaped = self.font.shape_text(text, size_px, BIDI_AUTO)
                    Self._emit_shaped_glyphs(self.draw_list, self.font_slot, self.font, shaped, x, baseline, size_px, color)
            else:
                self.draw_list.add_text_mut(self.font_slot, self.font, text, x, baseline, size_px, color)
        else:
            # Headless fallback when no SdfFont atlas is loaded: emit compact glyph-box indicator prims
            var n_cp = len(decode_utf8(text))
            if n_cp > 0:
                var adv = size_px * Float32(0.55)
                self.draw_list.add_rect(x, y + size_px * Float32(0.18), Float32(n_cp) * adv, size_px * Float32(0.64), color & UInt32(0x66FFFFFF), 2.0)

    def _alloc_item_rect(mut self, wid: UInt64, pref_w: Float32, pref_h: Float32, grow: Float32 = 0.0) -> RectF:
        var safe_pw = max(Float32(0.0), pref_w) if isfinite(pref_w) else Float32(0.0)
        var safe_ph = max(Float32(0.0), pref_h) if isfinite(pref_h) else Float32(0.0)
        var safe_grow = max(Float32(0.0), grow) if isfinite(grow) else Float32(0.0)
        var top_i = len(self.box_stack) - 1
        var parent_flex = self.box_stack[top_i].flex_idx
        var st = FlexStyle()
        st.fixed_w = safe_pw
        st.fixed_h = safe_ph
        st.grow = safe_grow
        var f_idx = self.flex.add_node(parent_flex, st, safe_pw, safe_ph)
        if wid != 0:
            self.next_frame_ids.append(wid)
            self.next_frame_flex_idxs.append(f_idx)

        var scope = self.box_stack[top_i]
        var r: RectF
        var cached = self.cached_rects.get(wid)
        if wid != 0 and cached and len(self.scroll_stack) == 0:
            var cr = cached.value()
            r = RectF(
                cr.x if isfinite(cr.x) else Float32(0.0),
                cr.y if isfinite(cr.y) else Float32(0.0),
                max(Float32(0.0), cr.w) if isfinite(cr.w) else Float32(0.0),
                max(Float32(0.0), cr.h) if isfinite(cr.h) else Float32(0.0),
            )
        else:
            var cx = scope.cursor_x if isfinite(scope.cursor_x) else Float32(0.0)
            var cy = scope.cursor_y if isfinite(scope.cursor_y) else Float32(0.0)
            if scope.direction == FLEX_ROW:
                var h = safe_ph if safe_ph > Float32(0.0) else max(Float32(0.0), scope.rect.bottom() - scope.pad_b - cy)
                var w = safe_pw if safe_pw > Float32(0.0) else max(Float32(40.0), scope.rect.right() - scope.pad_r - cx)
                if not isfinite(h):
                    h = Float32(0.0)
                if not isfinite(w):
                    w = Float32(40.0)
                r = RectF(cx, cy, w, h)
                self.box_stack[top_i].cursor_x = cx + w + scope.gap
            else:
                var w = safe_pw if safe_pw > Float32(0.0) else max(Float32(0.0), scope.rect.right() - scope.pad_r - cx)
                if not isfinite(w):
                    w = Float32(0.0)
                r = RectF(cx, cy, w, safe_ph)
                self.box_stack[top_i].cursor_y = cy + safe_ph + scope.gap
        self.last_item_id = wid
        self.last_item_rect = r
        if r.right() > self.box_stack[top_i].max_x:
            self.box_stack[top_i].max_x = r.right()
        if r.bottom() > self.box_stack[top_i].max_y:
            self.box_stack[top_i].max_y = r.bottom()
        return r

    # ---- Containers, Flexbox, DockSpace & Virtualized ScrollView ----

    def begin_box(mut self, id: String, style: FlexStyle):
        var wid = self.widget_id(id)
        var top_i = len(self.box_stack) - 1
        var parent_flex = self.box_stack[top_i].flex_idx
        var safe_st = style
        safe_st.basis = style.basis if isfinite(style.basis) else Float32(-1.0)
        safe_st.grow = max(Float32(0.0), style.grow) if isfinite(style.grow) else Float32(0.0)
        safe_st.shrink = max(Float32(0.0), style.shrink) if isfinite(style.shrink) else Float32(1.0)
        safe_st.min_w = max(Float32(0.0), style.min_w) if isfinite(style.min_w) else Float32(0.0)
        safe_st.min_h = max(Float32(0.0), style.min_h) if isfinite(style.min_h) else Float32(0.0)
        safe_st.max_w = max(safe_st.min_w, style.max_w if isfinite(style.max_w) else Float32(1.0e9))
        safe_st.max_h = max(safe_st.min_h, style.max_h if isfinite(style.max_h) else Float32(1.0e9))
        safe_st.pad_l = max(Float32(0.0), style.pad_l) if isfinite(style.pad_l) else Float32(0.0)
        safe_st.pad_t = max(Float32(0.0), style.pad_t) if isfinite(style.pad_t) else Float32(0.0)
        safe_st.pad_r = max(Float32(0.0), style.pad_r) if isfinite(style.pad_r) else Float32(0.0)
        safe_st.pad_b = max(Float32(0.0), style.pad_b) if isfinite(style.pad_b) else Float32(0.0)
        safe_st.gap = max(Float32(0.0), style.gap) if isfinite(style.gap) else Float32(0.0)
        safe_st.fixed_w = style.fixed_w if isfinite(style.fixed_w) else Float32(0.0)
        safe_st.fixed_h = style.fixed_h if isfinite(style.fixed_h) else Float32(0.0)
        var pref_w = safe_st.fixed_w if safe_st.fixed_w > Float32(0.0) else Float32(0.0)
        var pref_h = safe_st.fixed_h if safe_st.fixed_h > Float32(0.0) else Float32(0.0)
        var f_idx = self.flex.add_node(parent_flex, safe_st, pref_w, pref_h)
        self.next_frame_ids.append(wid)
        self.next_frame_flex_idxs.append(f_idx)

        var parent_scope = self.box_stack[top_i]
        var box_rect: RectF
        var cached = self.cached_rects.get(wid)
        if cached and len(self.scroll_stack) == 0:
            var cr = cached.value()
            box_rect = RectF(
                cr.x if isfinite(cr.x) else Float32(0.0),
                cr.y if isfinite(cr.y) else Float32(0.0),
                max(Float32(0.0), cr.w) if isfinite(cr.w) else Float32(0.0),
                max(Float32(0.0), cr.h) if isfinite(cr.h) else Float32(0.0),
            )
        else:
            var cx = parent_scope.cursor_x if isfinite(parent_scope.cursor_x) else Float32(0.0)
            var cy = parent_scope.cursor_y if isfinite(parent_scope.cursor_y) else Float32(0.0)
            var w = safe_st.fixed_w if safe_st.fixed_w > Float32(0.0) else max(Float32(0.0), parent_scope.rect.right() - parent_scope.pad_r - cx)
            var h: Float32
            if len(self.scroll_stack) > 0:
                if cached:
                    var cr = cached.value()
                    if safe_st.fixed_w <= Float32(0.0) and isfinite(cr.w) and cr.w > Float32(0.0):
                        w = cr.w
                    h = safe_st.fixed_h if safe_st.fixed_h > Float32(0.0) else (max(Float32(0.0), cr.h) if isfinite(cr.h) else Float32(0.0))
                else:
                    h = safe_st.fixed_h if safe_st.fixed_h > Float32(0.0) else Float32(0.0)
            else:
                h = safe_st.fixed_h if safe_st.fixed_h > Float32(0.0) else max(Float32(0.0), parent_scope.rect.bottom() - parent_scope.pad_b - cy)
            box_rect = RectF(cx, cy, w if isfinite(w) else Float32(0.0), h if isfinite(h) else Float32(0.0))

        self.box_stack.append(
            _BoxScope(
                wid,
                f_idx,
                box_rect,
                box_rect.x + safe_st.pad_l,
                box_rect.y + safe_st.pad_t,
                safe_st.direction,
                safe_st.gap,
                safe_st.pad_r,
                safe_st.pad_b,
                box_rect.x + safe_st.pad_l,
                box_rect.y + safe_st.pad_t,
            )
        )
        self.id_stack.append(wid)

    def end_box(mut self):
        if len(self.box_stack) <= 1:
            return
        var finished = self.box_stack.pop()
        _ = self.id_stack.pop()
        var top_i = len(self.box_stack) - 1
        var fin_right = finished.rect.right()
        var fin_bottom = finished.rect.bottom()
        if len(self.scroll_stack) > 0 or finished.rect.w <= Float32(0.0):
            fin_right = max(fin_right, finished.max_x + finished.pad_r)
        if len(self.scroll_stack) > 0 or finished.rect.h <= Float32(0.0):
            fin_bottom = max(fin_bottom, finished.max_y + finished.pad_b)
        if self.box_stack[top_i].direction == FLEX_ROW:
            self.box_stack[top_i].cursor_x = max(self.box_stack[top_i].cursor_x, fin_right + self.box_stack[top_i].gap)
        else:
            self.box_stack[top_i].cursor_y = max(self.box_stack[top_i].cursor_y, fin_bottom + self.box_stack[top_i].gap)
        if fin_right > self.box_stack[top_i].max_x:
            self.box_stack[top_i].max_x = fin_right
        if fin_bottom > self.box_stack[top_i].max_y:
            self.box_stack[top_i].max_y = fin_bottom

    def update_dock_space(mut self, mut dock: DockSpace, root_rect: RectF):
        """Solves `dock`, handles interactive splitter dragging & tab clicks, and emits DockSpace chrome."""
        dock.solve_layout(root_rect)
        var dock_ptr_blocked = self._blocked_by_modal() or self._is_pointer_blocked_by_popup()
        if not dock_ptr_blocked:
            _ = dock.interact_splitters(self.mouse_x, self.mouse_y, self.mouse_down_l, self.mouse_pressed_l, root_rect)
        if dock.hovered_splitter_node >= 0 or dock.dragging_node >= 0:
            self._dock_splitter_active = True

        var use_v2 = dock.dock_style_v2 or self.dock_style_v2
        if use_v2:
            self._draw_dock_space_v2(dock, dock_ptr_blocked)
            return

        for i in range(len(dock.nodes)):
            ref dn = dock.nodes[i]
            var r = dn.rect
            if dn.kind == DOCK_TABS:
                if not dock_ptr_blocked and r.contains(self.mouse_x, self.mouse_y) and self.hovered_id == 0:
                    self.hovered_id = hash_ui_id(UInt64(0xD0C47AB5), String(i))
                    self.hovered_rect = r
                self.draw_list.add_rect(r.x, r.y, r.w, r.h, self.theme.bg_panel, 0.0, 1.0, self.theme.border)
                var safe_hh = max(Float32(0.0), dock.header_height) if isfinite(dock.header_height) else Float32(24.0)
                var hh = min(safe_hh, max(Float32(0.0), r.h))
                self.draw_list.add_rect(r.x, r.y, r.w, hh, self.theme.bg_header)
                var tx = r.x + Float32(4.0)
                for t in range(len(dn.tabs)):
                    var tname = dn.tabs[t]
                    var tw = max(Float32(64.0), self._measure_text(tname, self.theme.font_size * Float32(0.92)) + Float32(18.0))
                    var tab_r = RectF(tx, r.y + Float32(2.0), tw, hh - Float32(2.0))
                    var is_act = dn.active_tab == t
                    if not dock_ptr_blocked and self.mouse_pressed_l and tab_r.contains(self.mouse_x, self.mouse_y):
                        dock.nodes[i].active_tab = t
                        is_act = True
                    if is_act:
                        self.draw_list.add_rect(tab_r.x, tab_r.y, tab_r.w, tab_r.h, self.theme.bg_panel)
                        self.draw_list.add_rect(tab_r.x, tab_r.bottom() - Float32(2.0), tab_r.w, Float32(2.0), self.theme.accent)
                    self._draw_text(
                        tname,
                        tab_r.x + Float32(8.0),
                        tab_r.y + Float32(3.0),
                        self.theme.font_size * Float32(0.92),
                        self.theme.text_primary if is_act else self.theme.text_muted,
                    )
                    tx += tw + Float32(2.0)
            elif dn.kind == DOCK_CENTRAL_VIEWPORT:
                # Pass-through 3D viewport frame border only (keeps 3D render clear!)
                self.draw_list.add_rect(r.x, r.y, r.w, r.h, 0, 0.0, 1.0, self.theme.border)

        # Draw splitters on top
        for i in range(len(dock.nodes)):
            var k = dock.nodes[i].kind
            var r = dock.nodes[i].rect
            var hot = dock.dragging_node == i or dock.hovered_splitter_node == i
            var col = self.theme.accent if hot else self.theme.border
            if k == DOCK_SPLIT_H:
                var sx = dock.split_seam(i)
                self.draw_list.add_line(sx, r.y, sx, r.bottom(), Float32(2.0) if hot else Float32(1.0), col)
            elif k == DOCK_SPLIT_V:
                var sy = dock.split_seam(i)
                self.draw_list.add_line(r.x, sy, r.right(), sy, Float32(2.0) if hot else Float32(1.0), col)

    def _draw_dock_space_v2(mut self, mut dock: DockSpace, dock_ptr_blocked: Bool):
        """Obsidian v2 DockSpace chrome: animated sliding tab indicator pills, collapse chevrons, and glowing glass splitters."""
        for i in range(len(dock.nodes)):
            var d_kind = dock.nodes[i].kind
            var r = dock.nodes[i].rect
            if d_kind == DOCK_TABS:
                var node_wid = hash_ui_id(UInt64(0xD0C47AB5), String(i))
                var collapsed = dock.is_node_collapsed(i)
                var safe_hh = max(Float32(0.0), dock.header_height) if isfinite(dock.header_height) else Float32(28.0)
                var hh = min(safe_hh, max(Float32(0.0), r.h))
                var hit_r = RectF(r.x, r.y, r.w, hh) if collapsed else r
                if not dock_ptr_blocked and hit_r.contains(self.mouse_x, self.mouse_y) and self.hovered_id == 0:
                    self.hovered_id = node_wid
                    self.hovered_rect = hit_r

                # Panel shell + subtle hairline
                if not collapsed:
                    self.draw_list.add_rect(r.x, r.y, r.w, r.h, self.studio_theme.l1_shell, 0.0, 1.0, self.studio_theme.border_subtle)
                self.draw_list.add_rect_styled(
                    r.x, r.y, r.w, hh,
                    self.studio_theme.l0_canvas,
                    self.studio_theme.border_subtle,
                    0.0, 0.0, 1.0, UI_FLAG_TOP_BEVEL,
                )
                self.draw_list.add_line(r.x, r.y + hh, r.right(), r.y + hh, 1.0, self.studio_theme.border_subtle)

                # Right-aligned collapse chevron button (zero dict allocation while at default open rest)
                var chev_r = RectF(max(r.x, r.right() - hh), r.y, min(hh, r.w), hh)
                var chev_hov = (not dock_ptr_blocked) and chev_r.contains(self.mouse_x, self.mouse_y)
                var chev_wid = hash_ui_id(node_wid, "collapse")
                var chev_open_t = Float32(0.0) if collapsed else Float32(1.0)
                if chev_hov and self.mouse_pressed_l:
                    _ = self.animator.drive_expand(chev_wid, not collapsed)
                    collapsed = dock.toggle_node_collapsed(i)
                    chev_open_t = self.animator.drive_expand(chev_wid, not collapsed)
                elif collapsed or (chev_wid in self.animator.widgets):
                    chev_open_t = self.animator.drive_expand(chev_wid, not collapsed)
                var chev_col = self.studio_theme.text_primary if chev_hov else self.studio_theme.text_muted
                draw_studio_chevron(
                    self.draw_list,
                    chev_r.center_x() - Float32(6.0),
                    chev_r.center_y() - Float32(6.0),
                    Float32(12.0),
                    chev_col,
                    chev_open_t,
                    1.5,
                )

                # Tabs + animated sliding active-tab indicator pill
                var tx = r.x + Float32(6.0)
                var max_tx = chev_r.x - Float32(4.0)
                var act_x = tx
                var act_w = Float32(0.0)
                var tab_font_sz = self.studio_theme.font_section
                var n_tabs = len(dock.nodes[i].tabs) if max_tx > tx + Float32(24.0) else 0  # a collapsed side strip shows only its chevron
                for t in range(n_tabs):
                    var tname = dock.nodes[i].tabs[t]
                    var tw = max(Float32(58.0), self._measure_text(tname, tab_font_sz) + Float32(20.0))
                    if tx + tw > max_tx and max_tx > tx + Float32(24.0):
                        tw = max_tx - tx
                    var tab_r = RectF(tx, r.y + Float32(3.0), tw, max(Float32(4.0), hh - Float32(4.0)))
                    var is_act = dock.nodes[i].active_tab == t
                    var tab_hov = (not dock_ptr_blocked) and tab_r.contains(self.mouse_x, self.mouse_y)
                    if tab_hov and self.mouse_pressed_l:
                        dock.nodes[i].active_tab = t
                        is_act = True
                        if collapsed:
                            _ = dock.toggle_node_collapsed(i)
                            collapsed = False
                    var tab_wid = hash_ui_id(node_wid, tname)
                    var anim_h = self.animator.drive_widget(tab_wid, tab_hov, tab_hov and self.mouse_down_l, False)
                    if is_act:
                        act_x = tab_r.x
                        act_w = tab_r.w
                        self.draw_list.add_rounded_rect_ex(
                            tab_r.x, tab_r.y, tab_r.w, tab_r.h,
                            self.studio_theme.l2_card,
                            RADIUS_SM, RADIUS_SM, 0.0, 0.0,
                            1.0, self.studio_theme.border_subtle, 1.0, UI_FLAG_TOP_BEVEL,
                        )
                    elif anim_h[0] > Float32(0.01):
                        self.draw_list.add_rounded_rect_ex(
                            tab_r.x, tab_r.y, tab_r.w, tab_r.h,
                            rgba8_scale_alpha(self.studio_theme.l3_raised, anim_h[0] * Float32(0.65)),
                            RADIUS_SM, RADIUS_SM, 0.0, 0.0,
                        )
                    var disp_txt = self._truncate_text_ellipsis(tname, max(Float32(16.0), tab_r.w - Float32(14.0)), tab_font_sz)
                    var txt_col = (
                        self.studio_theme.text_primary
                        if is_act
                        else lerp_rgba8(self.studio_theme.text_secondary, self.studio_theme.text_primary, anim_h[0])
                    )
                    self._draw_text(
                        disp_txt,
                        tab_r.x + Float32(8.0),
                        tab_r.y + max(Float32(1.0), (tab_r.h - tab_font_sz) * Float32(0.5)),
                        tab_font_sz,
                        txt_col,
                    )
                    tx += tw + Float32(4.0)

                if act_w > Float32(1.0):
                    var pill = self.animator.drive_tab_pill(node_wid, act_x, act_w)
                    self.draw_list.add_glow_rect(
                        pill[0] + Float32(4.0),
                        r.y + hh - Float32(2.5),
                        max(Float32(4.0), pill[1] - Float32(8.0)),
                        Float32(2.5),
                        self.studio_theme.accent_primary,
                        rgba8_scale_alpha(self.studio_theme.accent_primary, 0.65),
                        1.25,
                        5.0,
                    )
            elif d_kind == DOCK_CENTRAL_VIEWPORT:
                self.draw_list.add_rect(r.x, r.y, r.w, r.h, 0, 0.0, 1.0, self.studio_theme.border_subtle)

        # Glass/Obsidian splitters with smooth hover highlight
        for i in range(len(dock.nodes)):
            var k = dock.nodes[i].kind
            var r = dock.nodes[i].rect
            if k != DOCK_SPLIT_H and k != DOCK_SPLIT_V:
                continue
            var hot = dock.dragging_node == i or dock.hovered_splitter_node == i
            var swid = hash_ui_id(UInt64(0x5B1177E2), String(i))
            var sa = self.animator.drive_widget(swid, hot, dock.dragging_node == i, False)
            var t_hot = max(sa[0], sa[1])
            var col = lerp_rgba8(self.studio_theme.border_subtle, self.studio_theme.accent_primary, t_hot)
            var thick = Float32(1.0) + Float32(1.5) * t_hot
            if k == DOCK_SPLIT_H:
                var sx = dock.split_seam(i)
                self.draw_list.add_line(sx, r.y, sx, r.bottom(), thick, col)
                if t_hot > Float32(0.05):
                    var gh = min(Float32(36.0), r.h * Float32(0.3))
                    self.draw_list.add_glow_rect(
                        sx - Float32(1.5),
                        r.center_y() - gh * Float32(0.5),
                        Float32(3.0),
                        gh,
                        self.studio_theme.accent_primary,
                        rgba8_scale_alpha(self.studio_theme.accent_primary, t_hot * Float32(0.7)),
                        1.5,
                        6.0,
                    )
            else:
                var sy = dock.split_seam(i)
                self.draw_list.add_line(r.x, sy, r.right(), sy, thick, col)
                if t_hot > Float32(0.05):
                    var gw = min(Float32(36.0), r.w * Float32(0.3))
                    self.draw_list.add_glow_rect(
                        r.center_x() - gw * Float32(0.5),
                        sy - Float32(1.5),
                        gw,
                        Float32(3.0),
                        self.studio_theme.accent_primary,
                        rgba8_scale_alpha(self.studio_theme.accent_primary, t_hot * Float32(0.7)),
                        1.5,
                        6.0,
                    )

    def begin_dock_panel(mut self, dock: DockSpace, panel_name: String) -> Bool:
        var found = dock.find_panel(panel_name)
        if found[0] < 0 or not found[2]:
            return False
        var raw_content = found[3]
        var content = RectF(
            raw_content.x if isfinite(raw_content.x) else Float32(0.0),
            raw_content.y if isfinite(raw_content.y) else Float32(0.0),
            max(Float32(0.0), raw_content.w) if isfinite(raw_content.w) else Float32(0.0),
            max(Float32(0.0), raw_content.h) if isfinite(raw_content.h) else Float32(0.0),
        )
        var wid = self.widget_id("dock:" + panel_name)
        var st = FlexStyle.col(gap=4.0, pad=6.0)
        st.fixed_w = content.w
        st.fixed_h = content.h
        var f_idx = self.flex.add_node(-1, st, content.w, content.h)
        self.dock_flex_root_idxs.append(f_idx)
        self.dock_flex_root_rects.append(content)
        self.box_stack.append(
            _BoxScope(
                wid,
                f_idx,
                content,
                content.x + Float32(6.0),
                content.y + Float32(6.0),
                FLEX_COL,
                4.0,
                6.0,
                6.0,
                content.x + Float32(6.0),
                content.y + Float32(6.0),
            )
        )
        self.id_stack.append(wid)
        self.draw_list.push_clip_f(content.x, content.y, content.right(), content.bottom())
        return True

    def end_dock_panel(mut self):
        if len(self.box_stack) > 1:
            _ = self.box_stack.pop()
            _ = self.id_stack.pop()
            self.draw_list.pop_clip()

    def begin_scroll_view(mut self, id: String, w: Float32, h: Float32, total_content_h: Float32 = -1.0):
        var wid = self.widget_id(id)
        var safe_w = max(Float32(0.0), w) if isfinite(w) else Float32(0.0)
        var safe_h = max(Float32(0.0), h) if isfinite(h) else Float32(0.0)
        var safe_content_h = total_content_h if isfinite(total_content_h) else Float32(-1.0)
        var raw_rect = self._alloc_item_rect(wid, safe_w, safe_h)
        var rx = raw_rect.x if isfinite(raw_rect.x) else Float32(0.0)
        var ry = raw_rect.y if isfinite(raw_rect.y) else Float32(0.0)
        var rw = max(Float32(0.0), raw_rect.w) if isfinite(raw_rect.w) else Float32(0.0)
        var rh = max(Float32(0.0), raw_rect.h) if isfinite(raw_rect.h) else Float32(0.0)
        var rect = RectF(rx, ry, rw, rh)
        self.last_item_rect = rect
        var st = self._get_state(wid)

        if not self._blocked_by_modal() and not self._is_pointer_blocked_by_popup(wid) and rect.contains(self.mouse_x, self.mouse_y):
            if self.hovered_id == 0:
                self.hovered_id = wid
                self.hovered_rect = self._clip_hovered_rect(rect)
            # The wheel goes to the innermost hovered scroll view, applied here (before layout) so
            # this frame's rects already reflect it. A nested view begins after its parent, so the
            # last hovered view begun is the innermost: last frame's pick (`_wheel_target`) takes
            # the wheel outright while it is still under the pointer; otherwise each hovered view
            # takes it tentatively and hands it on (restoring its offset) to a deeper one.
            self._wheel_target_next = wid
            if self.mouse_wheel != Float32(0.0) and isfinite(self.mouse_wheel) and not self._wheel_consumed:
                var definitive = wid == self._wheel_target
                var wait_for_target = False
                if not definitive and self._wheel_target != 0:
                    var tr = self.cached_rects.get(self._wheel_target)
                    wait_for_target = Bool(tr) and tr.value().contains(self.mouse_x, self.mouse_y)
                if definitive or not wait_for_target:
                    if self._wheel_tentative != 0 and self._wheel_tentative != wid:
                        var prev = self.state_map.get(self._wheel_tentative)
                        if prev:
                            var pst = prev.value().copy()
                            pst.scroll_y = self._wheel_tentative_sy
                            self.state_map[self._wheel_tentative] = pst^
                    var sy0 = st.scroll_y if isfinite(st.scroll_y) else Float32(0.0)
                    st.scroll_y = max(Float32(0.0), min(st.scroll_max_y, sy0 - self.mouse_wheel * Float32(36.0)))
                    self.state_map[wid] = st.copy()
                    if definitive:
                        self._wheel_consumed = True
                        self._wheel_tentative = 0
                    else:
                        self._wheel_tentative = wid
                        self._wheel_tentative_sy = sy0

        self.draw_list.add_rect(rect.x, rect.y, rect.w, rect.h, self.theme.bg_window, self.theme.corner_radius, 1.0, self.theme.border)
        self.draw_list.push_clip_f(rect.x, rect.y, rect.right(), rect.bottom())

        var inner_w = max(Float32(0.0), rect.w - self.theme.scrollbar_width - Float32(8.0))
        var box_st = FlexStyle.col(gap=2.0, pad=4.0)
        box_st.fixed_w = inner_w + Float32(8.0)
        var f_idx = self.flex.add_node(-1, box_st, inner_w + Float32(8.0), rect.h)

        var start_y = rect.y + Float32(4.0) - st.scroll_y
        self.box_stack.append(
            _BoxScope(
                wid,
                f_idx,
                RectF(rect.x + Float32(4.0), rect.y + Float32(4.0), inner_w, max(Float32(0.0), rect.h - Float32(8.0))),
                rect.x + Float32(4.0),
                start_y,
                FLEX_COL,
                2.0,
                Float32(4.0),
                Float32(4.0),
                rect.x + Float32(4.0),
                start_y,
            )
        )
        self.id_stack.append(wid)
        self.scroll_stack.append(_ScrollScope(wid, f_idx, rect, start_y, st.scroll_y, safe_content_h))

    def end_scroll_view(mut self):
        if len(self.scroll_stack) == 0 or len(self.box_stack) <= 1:
            return
        var sc = self.scroll_stack.pop()
        var inner_scope = self.box_stack.pop()
        _ = self.id_stack.pop()

        var span_y = inner_scope.cursor_y - sc.start_cursor_y
        var auto_h = max(Float32(0.0), span_y) if isfinite(span_y) else Float32(0.0)
        if sc.flex_idx >= 0:
            var root_w = max(Float32(8.0), sc.view_rect.w - self.theme.scrollbar_width)
            var root_h = max(sc.view_rect.h, auto_h + Float32(8.0))
            self.dock_flex_root_idxs.append(sc.flex_idx)
            self.dock_flex_root_rects.append(RectF(sc.view_rect.x, sc.start_cursor_y - Float32(4.0), root_w, root_h))
        # An explicit height is a hint, never a cap: rows laid out past it (gaps, wrapped
        # lines) must stay reachable, so the larger of the two wins.
        var measured_h = (
            max(sc.explicit_content_h, auto_h)
            if (isfinite(sc.explicit_content_h) and sc.explicit_content_h >= Float32(0.0))
            else auto_h
        )
        var view_h = sc.view_rect.h if isfinite(sc.view_rect.h) else Float32(0.0)
        var inner_view_h = max(Float32(0.0), view_h - Float32(8.0))
        var max_scroll = max(Float32(0.0), measured_h - inner_view_h)
        if measured_h <= Float32(0.0) or view_h <= Float32(8.0) or not isfinite(max_scroll):
            max_scroll = Float32(0.0)
        var st = self._get_state(sc.id_hash)
        st.scroll_max_y = max_scroll
        var cur_sy = st.scroll_y if isfinite(st.scroll_y) else Float32(0.0)
        if (
            not self._wheel_consumed
            and self._wheel_tentative == 0
            and self.mouse_wheel != Float32(0.0)
            and isfinite(self.mouse_wheel)
            and not self._blocked_by_modal()
            and not self._is_pointer_blocked_by_popup(sc.id_hash)
            and sc.view_rect.contains(self.mouse_x, self.mouse_y)
        ):
            self._wheel_consumed = True
            cur_sy = cur_sy - self.mouse_wheel * Float32(36.0)
        st.scroll_y = max(Float32(0.0), min(max_scroll, cur_sy))

        # Scrollbar track & draggable thumb
        if max_scroll > Float32(0.5):
            var sb_w = self.theme.scrollbar_width
            var track_x = sc.view_rect.right() - sb_w - Float32(2.0)
            var track_y = sc.view_rect.y + Float32(2.0)
            var track_h = max(Float32(12.0), sc.view_rect.h - Float32(4.0))
            var thumb_h = max(Float32(18.0), track_h * (sc.view_rect.h / max(Float32(1.0), measured_h)))
            var travel = max(Float32(1.0), track_h - thumb_h)
            var thumb_y = track_y + (st.scroll_y / max_scroll) * travel
            var track_rect = RectF(track_x, track_y, sb_w, track_h)

            if not self._blocked_by_modal() and not self._is_pointer_blocked_by_popup(sc.id_hash) and self.mouse_pressed_l and track_rect.contains(self.mouse_x, self.mouse_y):
                self.active_id = sc.id_hash
            if self.active_id == sc.id_hash and self.mouse_down_l and isfinite(self.mouse_y):
                var rel_t = max(Float32(0.0), min(Float32(1.0), (self.mouse_y - track_y - thumb_h * Float32(0.5)) / travel))
                if isfinite(rel_t):
                    st.scroll_y = rel_t * max_scroll
                    thumb_y = track_y + rel_t * travel

            self.draw_list.add_rect(track_x, track_y, sb_w, track_h, pack_rgba8(14, 16, 20, 180), sb_w * Float32(0.5))
            self.draw_list.add_rect(
                track_x + Float32(1.0),
                thumb_y,
                sb_w - Float32(2.0),
                thumb_h,
                self.theme.accent if self.active_id == sc.id_hash else self.theme.border,
                (sb_w - Float32(2.0)) * Float32(0.5),
            )

        self.state_map[sc.id_hash] = st
        self.draw_list.pop_clip()

    @always_inline
    def scroll_visible_range(self, total_items: Int, item_stride_y: Float32) -> Tuple[Int, Int]:
        """O(1) visible index range `[start_idx, end_idx)` for virtualized massive lists/trees."""
        if (
            total_items <= 0
            or len(self.scroll_stack) == 0
            or not isfinite(item_stride_y)
            or item_stride_y <= Float32(0.0)
        ):
            return (0, max(0, total_items))
        var sc = self.scroll_stack[len(self.scroll_stack) - 1]
        if not isfinite(sc.scroll_y) or not isfinite(sc.view_rect.h):
            return (0, total_items)
        var stride = Float64(item_stride_y)
        var sy = max(0.0, Float64(sc.scroll_y))
        var vh = max(0.0, Float64(sc.view_rect.h))
        var max_f = Float64(total_items)
        var first_f = sy / stride
        var first = total_items if (not isfinite(first_f) or first_f >= max_f) else max(0, Int(first_f))
        var vis_f = vh / stride
        var vis_cnt: Int
        if not isfinite(vis_f) or vis_f >= max_f:
            vis_cnt = total_items
        else:
            vis_cnt = max(0, Int(vis_f)) + 3
        var last = first + min(total_items - first, vis_cnt)
        return (first, last)

    # ---- Core Widgets ----

    def label(mut self, text: String, color: UInt32 = 0):
        var col = self.theme.text_primary if color == 0 else color
        var h = self.theme.row_height
        var top_i = len(self.box_stack) - 1
        if len(self.scroll_stack) > 0:
            var sc = self.scroll_stack[len(self.scroll_stack) - 1]
            var cy = self.box_stack[top_i].cursor_y
            if cy + h < sc.view_rect.y or cy > sc.view_rect.bottom():
                _ = self._alloc_item_rect(0, Float32(0.0), h)
                self.culled_items_this_frame += 1
                return
        var w = self._measure_text(text, self.theme.font_size) + Float32(6.0)
        var r = self._alloc_item_rect(0, w, h)
        self._draw_text(text, r.x + Float32(2.0), r.y + Float32(3.0), self.theme.font_size, col)

    def rich_label(mut self, markup: String, max_width: Float32 = 0.0):
        var safe_max_w = max_width if (isfinite(max_width) and max_width > Float32(0.0)) else Float32(0.0)
        var spans = parse_rich(markup)
        # Borrow the fonts (the display face falls back to the UI face) instead of copying an SdfFont.
        var has_display = self.display_font.valid
        if self.font.valid and (self.font.has_source or len(self.font.fallbacks) > 0):
            self.font.sync_shaper()
            if has_display:
                self.display_font.sync_shaper()
            var rls = layout_rich_shaped(spans, self.font, self.display_font, self.theme.font_size, 0, safe_max_w, 0, self.from_pad, self.pad_sony) if has_display else layout_rich_shaped(spans, self.font, self.font, self.theme.font_size, 0, safe_max_w, 0, self.from_pad, self.pad_sony)
            var h = max(self.theme.row_height, rls.height + Float32(4.0))
            var w = max(Float32(40.0), rls.width if safe_max_w <= Float32(0.0) else safe_max_w)
            var r = self._alloc_item_rect(0, w, h)
            for i in range(len(rls.runs)):
                ref run = rls.runs[i]
                var rx = r.x + run.x
                var ry = r.y + run.baseline - run.px * Float32(0.82)
                if run.kind == SPAN_BUTTON:
                    var btn = pad_button(run.action)
                    var fcol = UInt32(0xFF000000) | face_color(btn, self.pad_sony)
                    self.draw_list.add_controller_symbol(btn, rx, ry, run.width, run.px * Float32(1.1), fcol)
                else:
                    var col = self.theme.text_primary
                    if run.color >= 0:
                        var rr = (run.color >> 16) & 0xFF
                        var gg = (run.color >> 8) & 0xFF
                        var bb = run.color & 0xFF
                        col = pack_rgba8(rr, gg, bb, 255)
                    if run.face == 1 and self.display_font.valid:
                        Self._emit_shaped_glyphs(self.draw_list, self.font_slot, self.display_font, run.glyphs, rx, r.y + run.baseline, run.px, col)
                    else:
                        Self._emit_shaped_glyphs(self.draw_list, self.font_slot, self.font, run.glyphs, rx, r.y + run.baseline, run.px, col)
            return
        var rl = layout_rich(spans, self.font, self.display_font, self.theme.font_size, 0, safe_max_w, 0, self.from_pad, self.pad_sony) if has_display else layout_rich(spans, self.font, self.font, self.theme.font_size, 0, safe_max_w, 0, self.from_pad, self.pad_sony)
        var h = max(self.theme.row_height, rl.height + Float32(4.0))
        var w = max(Float32(40.0), rl.width if safe_max_w <= Float32(0.0) else safe_max_w)
        var r = self._alloc_item_rect(0, w, h)
        for i in range(len(rl.runs)):
            ref run = rl.runs[i]
            var rx = r.x + run.x
            var ry = r.y + run.baseline - run.px * Float32(0.82)
            if run.kind == SPAN_BUTTON:
                var btn = pad_button(run.action)
                var fcol = UInt32(0xFF000000) | face_color(btn, self.pad_sony)
                self.draw_list.add_controller_symbol(btn, rx, ry, run.width, run.px * Float32(1.1), fcol)
            else:
                var col = self.theme.text_primary
                if run.color >= 0:
                    var rr = (run.color >> 16) & 0xFF
                    var gg = (run.color >> 8) & 0xFF
                    var bb = run.color & 0xFF
                    col = pack_rgba8(rr, gg, bb, 255)
                self._draw_text(run.text, rx, ry, run.px, col)

    def button(mut self, id: String, label_text: String, w: Float32 = 0.0, h: Float32 = 0.0) -> Bool:
        var wid = self.widget_id(id)
        var bw = w if (isfinite(w) and w > Float32(0.0)) else max(Float32(76.0), self._measure_text(label_text, self.theme.font_size) + Float32(24.0))
        var bh = h if (isfinite(h) and h > Float32(0.0)) else self.theme.row_height + Float32(4.0)
        if not isfinite(bw):
            bw = Float32(76.0)
        if not isfinite(bh):
            bh = Float32(24.0)
        var r = self._alloc_item_rect(wid, bw, bh)
        self._register_focusable(wid)

        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        var hovered = (not ptr_blocked) and r.contains(self.mouse_x, self.mouse_y)
        if hovered:
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(r)
            if self.mouse_pressed_l:
                self.active_id = wid
                self._set_focused(wid)

        var clicked = False
        if not blocked:
            if hovered and self.mouse_released_l and self.active_id == wid:
                clicked = True
            elif self.focused_id == wid and (self.key_enter_pressed or self.key_space_pressed or self.pad_confirm):
                clicked = True

        var bg = self.theme.bg_widget
        if self.active_id == wid and self.mouse_down_l:
            bg = self.theme.bg_widget_active
        elif hovered:
            bg = self.theme.bg_widget_hover

        var focused = (not blocked) and self.focused_id == wid and ((not self.dock_style_v2) or self.focus_visible)
        var b_col = self.theme.focus_ring if focused else self.theme.border
        var b_w = Float32(2.0) if focused else Float32(1.0)
        self.draw_list.add_rect_styled(
            r.x, r.y, r.w, r.h,
            bg, b_col,
            self.theme.corner_radius,
            b_w, 1.0,
            UI_FLAG_BORDER | UI_FLAG_VERT_GRADIENT,
        )
        var tw = self._measure_text(label_text, self.theme.font_size)
        var tx = r.x + max(Float32(6.0), (r.w - tw) * Float32(0.5))
        var ty = r.y + max(Float32(2.0), (r.h - self.theme.font_size) * Float32(0.5))
        self._draw_text(label_text, tx, ty, self.theme.font_size, self.theme.text_primary)
        return clicked

    def checkbox(mut self, id: String, label_text: String, mut checked: Bool) -> Bool:
        var wid = self.widget_id(id)
        var box_sz = self.theme.row_height - Float32(4.0)
        var total_w = box_sz + Float32(8.0) + self._measure_text(label_text, self.theme.font_size) + Float32(8.0)
        var r = self._alloc_item_rect(wid, total_w, self.theme.row_height)
        self._register_focusable(wid)

        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        var hovered = (not ptr_blocked) and r.contains(self.mouse_x, self.mouse_y)
        var toggled = False
        if hovered:
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(r)
            if self.mouse_pressed_l:
                self._set_focused(wid)
                checked = not checked
                toggled = True
        if (not blocked) and self.focused_id == wid and (self.key_enter_pressed or self.key_space_pressed or self.pad_confirm):
            checked = not checked
            toggled = True

        var focused = (not blocked) and self.focused_id == wid and ((not self.dock_style_v2) or self.focus_visible)
        var bx = r.x + Float32(2.0)
        var by = r.y + Float32(2.0)
        self.draw_list.add_rect(
            bx, by, box_sz, box_sz,
            self.theme.accent if checked else (self.theme.bg_widget_hover if hovered else self.theme.bg_widget),
            3.0,
            Float32(2.0) if focused else Float32(1.0),
            self.theme.focus_ring if focused else self.theme.border,
        )
        if checked:
            self.draw_list.add_line(bx + box_sz * Float32(0.22), by + box_sz * Float32(0.52), bx + box_sz * Float32(0.44), by + box_sz * Float32(0.76), 2.2, pack_rgba8(255, 255, 255, 255))
            self.draw_list.add_line(bx + box_sz * Float32(0.44), by + box_sz * Float32(0.76), bx + box_sz * Float32(0.80), by + box_sz * Float32(0.24), 2.2, pack_rgba8(255, 255, 255, 255))

        self._draw_text(label_text, bx + box_sz + Float32(8.0), r.y + Float32(3.0), self.theme.font_size, self.theme.text_primary)
        return toggled

    def slider_float(
        mut self,
        id: String,
        label_text: String,
        mut value: Float32,
        min_val: Float32,
        max_val: Float32,
        w: Float32 = 200.0,
    ) -> Bool:
        var wid = self.widget_id(id)
        var safe_w = w if (isfinite(w) and w > Float32(0.0)) else Float32(200.0)
        var r = self._alloc_item_rect(wid, safe_w, self.theme.row_height)
        self._register_focusable(wid)

        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        var track_w = r.w * Float32(0.58)
        var track = RectF(r.x, r.y + Float32(3.0), track_w, r.h - Float32(6.0))
        var hovered = (not ptr_blocked) and track.contains(self.mouse_x, self.mouse_y)
        if hovered:
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(track)
            if self.mouse_pressed_l:
                self.active_id = wid
                self._set_focused(wid)

        var lo64 = Float64(min_val) if isfinite(min_val) else 0.0
        var hi64 = Float64(max_val) if isfinite(max_val) else (lo64 + 1.0)
        if hi64 < lo64:
            var tmp_b = lo64
            lo64 = hi64
            hi64 = tmp_b
        var span64 = max(1.0e-6, hi64 - lo64)
        var was_finite = isfinite(value)
        var val64 = Float64(value) if was_finite else lo64
        val64 = max(lo64, min(hi64, val64))
        if not was_finite:
            value = Float32(val64)

        var changed = False
        if (not blocked) and self.active_id == wid and self.mouse_down_l:
            if isfinite(self.mouse_x) and isfinite(track.x) and isfinite(track.w):
                var t64 = max(0.0, min(1.0, (Float64(self.mouse_x) - Float64(track.x)) / max(1.0, Float64(track.w))))
                var nv64 = max(lo64, min(hi64, lo64 + t64 * span64))
                var nv = Float32(nv64)
                if not isfinite(nv):
                    nv = Float32(hi64 if t64 >= 0.5 else lo64)
                if (not was_finite) or abs(Float64(nv) - Float64(value)) > 1.0e-6:
                    value = nv
                    changed = True
        elif (not blocked) and self.focused_id == wid:
            if self.key_left_pressed:
                var nv = Float32(max(lo64, val64 - span64 * 0.05))
                if not isfinite(nv):
                    nv = Float32(lo64)
                value = nv
                changed = True
            elif self.key_right_pressed:
                var nv = Float32(min(hi64, val64 + span64 * 0.05))
                if not isfinite(nv):
                    nv = Float32(hi64)
                value = nv
                changed = True

        var cur_v64 = Float64(value) if isfinite(value) else lo64
        var norm = Float32(max(0.0, min(1.0, (cur_v64 - lo64) / span64)))
        var focused = (not blocked) and self.focused_id == wid and ((not self.dock_style_v2) or self.focus_visible)
        self.draw_list.add_rect(
            track.x, track.y, track.w, track.h,
            self.theme.bg_widget,
            self.theme.corner_radius,
            Float32(2.0) if focused else Float32(1.0),
            self.theme.focus_ring if focused else self.theme.border,
        )
        if norm > Float32(0.005):
            self.draw_list.add_rect(track.x + Float32(2.0), track.y + Float32(2.0), max(Float32(2.0), (track.w - Float32(4.0)) * norm), track.h - Float32(4.0), self.theme.accent, self.theme.corner_radius - Float32(1.0))
        self.draw_list.add_circle(track.x + track.w * norm, track.center_y(), Float32(6.0), pack_rgba8(245, 248, 255, 255), 1.5, self.theme.accent)
        self._draw_text(label_text, track.right() + Float32(8.0), r.y + Float32(3.0), self.theme.font_size, self.theme.text_primary)
        return changed

    def drag_float3(
        mut self,
        id: String,
        label_text: String,
        mut v: Vec3,
        speed: Float32 = 0.05,
        min_val: Float32 = -1.0e6,
        max_val: Float32 = 1.0e6,
        w: Float32 = 260.0,
    ) -> Bool:
        var safe_w = w if (isfinite(w) and w > Float32(0.0)) else Float32(260.0)
        var r = self._alloc_item_rect(self.widget_id(id), safe_w, self.theme.row_height)
        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup()
        var comp_w = max(Float32(1.0), (r.w * Float32(0.68) - Float32(8.0)) / Float32(3.0))
        var lo64 = Float64(min_val) if isfinite(min_val) else -1.0e6
        var hi64 = Float64(max_val) if isfinite(max_val) else 1.0e6
        if hi64 < lo64:
            var tmp_b = lo64
            lo64 = hi64
            hi64 = tmp_b
        var sp64 = Float64(speed) if isfinite(speed) else 0.05
        var had_non_finite = (not isfinite(v.x)) or (not isfinite(v.y)) or (not isfinite(v.z))
        var vals = Array[Float32, 3](fill=0.0)
        vals[0] = v.x if isfinite(v.x) else Float32(max(lo64, min(hi64, 0.0)))
        vals[1] = v.y if isfinite(v.y) else Float32(max(lo64, min(hi64, 0.0)))
        vals[2] = v.z if isfinite(v.z) else Float32(max(lo64, min(hi64, 0.0)))
        var axis_cols = Array[UInt32, 3](fill=0)
        axis_cols[0] = pack_rgba8(224, 82, 82, 255)
        axis_cols[1] = pack_rgba8(92, 198, 108, 255)
        axis_cols[2] = pack_rgba8(78, 148, 240, 255)
        var changed = False

        for c in range(3):
            var cid = self.widget_id(id + "#" + String(c))
            self._register_focusable(cid, allow_auto_focus=False)
            var cr = RectF(r.x + Float32(c) * (comp_w + Float32(4.0)), r.y + Float32(2.0), comp_w, r.h - Float32(4.0))
            if not ptr_blocked and cr.contains(self.mouse_x, self.mouse_y):
                self.hovered_id = cid
                self.hovered_rect = self._clip_hovered_rect(cr)
                if self.mouse_pressed_l:
                    self.active_id = cid
                    self._set_focused(cid)
            if not blocked and self.active_id == cid and self.mouse_down_l and (not self.mouse_pressed_l) and isfinite(self.mouse_delta_x) and self.mouse_delta_x != Float32(0.0):
                var nv = Float32(max(lo64, min(hi64, Float64(vals[c]) + Float64(self.mouse_delta_x) * sp64)))
                if isfinite(nv):
                    vals[c] = nv
                    changed = True
            elif not blocked and self.focused_id == cid and (self.key_left_pressed or self.key_right_pressed):
                var step64 = sp64 if self.key_right_pressed else -sp64
                var nv = Float32(max(lo64, min(hi64, Float64(vals[c]) + step64)))
                if isfinite(nv) and nv != vals[c]:
                    vals[c] = nv
                    changed = True
            var comp_focus = self.show_focus_ring(cid)
            self.draw_list.add_rect(cr.x, cr.y, cr.w, cr.h, self.theme.bg_widget, 3.0, Float32(2.0) if comp_focus else Float32(1.0), self.theme.focus_ring if comp_focus else self.theme.border)
            self.draw_list.add_rect(cr.x, cr.y, 3.0, cr.h, axis_cols[c], 1.5)
            var txt = format_float2(vals[c])
            self._draw_text(txt, cr.x + Float32(6.0), cr.y + Float32(2.0), self.theme.font_size * Float32(0.88), self.theme.text_primary)

        if changed or had_non_finite:
            v.x = vals[0]; v.y = vals[1]; v.z = vals[2]
        self._draw_text(label_text, r.x + r.w * Float32(0.68) + Float32(6.0), r.y + Float32(3.0), self.theme.font_size, self.theme.text_primary)
        return changed

    def drag_float(
        mut self,
        id: String,
        label_text: String,
        mut value: Float32,
        speed: Float32 = 0.05,
        min_val: Float32 = -1.0e6,
        max_val: Float32 = 1.0e6,
        w: Float32 = 220.0,
    ) -> Bool:
        var wid = self.widget_id(id)
        var safe_w = w if (isfinite(w) and w > Float32(0.0)) else Float32(220.0)
        var r = self._alloc_item_rect(wid, safe_w, self.theme.row_height)
        self._register_focusable(wid)

        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        var field_r = RectF(r.x, r.y + Float32(2.0), max(Float32(20.0), r.w * Float32(0.58)), r.h - Float32(4.0))
        var hovered = (not ptr_blocked) and field_r.contains(self.mouse_x, self.mouse_y)
        if hovered:
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(field_r)
            if self.mouse_pressed_l:
                self.active_id = wid
                self._set_focused(wid)

        var lo64 = Float64(min_val) if isfinite(min_val) else -1.0e6
        var hi64 = Float64(max_val) if isfinite(max_val) else 1.0e6
        if hi64 < lo64:
            var tmp_b = lo64
            lo64 = hi64
            hi64 = tmp_b
        var sp64 = Float64(speed) if (isfinite(speed) and speed > Float32(0.0)) else 0.05
        var was_finite = isfinite(value)
        var val64 = Float64(value) if was_finite else lo64
        var clamped64 = max(lo64, min(hi64, val64))
        if (not was_finite) or abs(clamped64 - val64) > 1.0e-7:
            value = Float32(clamped64)

        var changed = False
        if (not blocked) and self.active_id == wid and self.mouse_down_l and (not self.mouse_pressed_l) and isfinite(self.mouse_delta_x) and self.mouse_delta_x != Float32(0.0):
            var scale = 0.2 if self.key_shift else 1.0
            var nv64 = max(lo64, min(hi64, clamped64 + Float64(self.mouse_delta_x) * sp64 * scale))
            var nv = Float32(nv64)
            if isfinite(nv) and abs(Float64(nv) - clamped64) > 1.0e-7:
                value = nv
                changed = True
        elif (not blocked) and self.focused_id == wid:
            if self.key_left_pressed:
                var nv = Float32(max(lo64, clamped64 - sp64))
                if isfinite(nv):
                    value = nv
                    changed = True
            elif self.key_right_pressed:
                var nv = Float32(min(hi64, clamped64 + sp64))
                if isfinite(nv):
                    value = nv
                    changed = True

        var focused = (not blocked) and self.focused_id == wid and ((not self.dock_style_v2) or self.focus_visible)
        var bg = self.theme.bg_widget_active if (self.active_id == wid and self.mouse_down_l) else (self.theme.bg_widget_hover if hovered else self.theme.bg_widget)
        self.draw_list.add_rect(
            field_r.x, field_r.y, field_r.w, field_r.h,
            bg, 3.0,
            Float32(2.0) if focused else Float32(1.0),
            self.theme.focus_ring if focused else self.theme.border,
        )
        self._draw_text(format_float2(value), field_r.x + Float32(6.0), field_r.y + Float32(2.0), self.theme.font_size * Float32(0.88), self.theme.text_primary)
        self._draw_text(label_text, field_r.right() + Float32(8.0), r.y + Float32(3.0), self.theme.font_size, self.theme.text_primary)
        return changed

    def combo_box(
        mut self,
        id: String,
        label_text: String,
        mut selected_idx: Int,
        items: List[String],
        w: Float32 = 220.0,
    ) -> Bool:
        var wid = self.widget_id(id)
        var safe_w = w if (isfinite(w) and w > Float32(0.0)) else Float32(220.0)
        var r = self._alloc_item_rect(wid, safe_w, self.theme.row_height)
        self._register_focusable(wid)

        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        var box_r = RectF(r.x, r.y + Float32(2.0), max(Float32(24.0), r.w * Float32(0.58)), r.h - Float32(4.0))
        var hovered = (not ptr_blocked) and box_r.contains(self.mouse_x, self.mouse_y)
        var just_opened_popup = False
        var toggled_by_click = False
        if hovered:
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(box_r)
            if self.mouse_pressed_l:
                self._set_focused(wid)
                toggled_by_click = True
                if len(items) > 0:
                    if self._open_popup_id == wid:
                        self._open_popup_id = 0
                        self._popup_kind = 0
                    else:
                        self._open_popup_id = wid
                        self._popup_kind = 1
                        just_opened_popup = True

        if (not blocked) and (not toggled_by_click) and self.focused_id == wid and (self.key_enter_pressed or self.key_space_pressed or self.pad_confirm):
            if len(items) > 0:
                if self._open_popup_id == wid:
                    self._open_popup_id = 0
                    self._popup_kind = 0
                else:
                    self._open_popup_id = wid
                    self._popup_kind = 1
                    just_opened_popup = True

        var changed = False
        if not blocked and self._open_popup_id == wid and len(items) > 0:
            self._popup_seen = True
            var row_h = self.theme.row_height
            var drop_h = Float32(len(items)) * row_h + Float32(4.0)
            var drop_y = box_r.bottom() + Float32(2.0)
            if drop_y + drop_h > self.screen_h and box_r.y - drop_h - Float32(2.0) >= Float32(0.0):
                drop_y = box_r.y - drop_h - Float32(2.0)
            var drop_r = RectF(box_r.x, drop_y, box_r.w, drop_h)

            var hover_row = -1
            if not just_opened_popup:
                if drop_r.contains(self.mouse_x, self.mouse_y):
                    self.hovered_id = wid
                    self.hovered_rect = drop_r
                    self._popup_consumed_click = True
                    self._popup_consumed_rect = drop_r
                    var rel_y = self.mouse_y - (drop_r.y + Float32(2.0))
                    var idx = Int(rel_y / max(Float32(1.0), row_h))
                    if idx >= 0 and idx < len(items):
                        hover_row = idx
                        if self.mouse_pressed_l:
                            if selected_idx != idx:
                                selected_idx = idx
                                changed = True
                            self._open_popup_id = 0
                            self._popup_kind = 0
                elif self.mouse_pressed_l and not hovered:
                    self._open_popup_id = 0
                    self._popup_kind = 0

                if self._open_popup_id == wid:
                    if self.key_esc_pressed:
                        self._open_popup_id = 0
                        self._popup_kind = 0
                    elif self.key_down_pressed:
                        selected_idx = (selected_idx + 1) % len(items) if selected_idx >= 0 else 0
                        changed = True
                    elif self.key_up_pressed:
                        selected_idx = (selected_idx - 1 + len(items)) % len(items) if selected_idx >= 0 else 0
                        changed = True
                    elif self.key_enter_pressed and not hovered:
                        self._open_popup_id = 0
                        self._popup_kind = 0

            if self._open_popup_id == wid:
                self._popup_kind = 1
                self._popup_rect = drop_r
                self._popup_owner_rect = box_r
                # Copy the item list only when it changed, not every frame the popup is open.
                var same_items = self._popup_combo_owner == wid and len(self._popup_combo_items) == len(items)
                if same_items:
                    for ii in range(len(items)):
                        if self._popup_combo_items[ii] != items[ii]:
                            same_items = False
                            break
                if not same_items:
                    self._popup_combo_items = items.copy()
                    self._popup_combo_owner = wid
                self._popup_combo_sel = selected_idx
                self._popup_combo_hover = hover_row

        var focused = (not blocked) and self.focused_id == wid
        self.draw_list.add_rect(
            box_r.x, box_r.y, box_r.w, box_r.h,
            self.theme.bg_widget_hover if hovered else self.theme.bg_widget,
            3.0,
            Float32(2.0) if focused else Float32(1.0),
            self.theme.focus_ring if focused else self.theme.border,
        )
        var preview = items[selected_idx] if (selected_idx >= 0 and selected_idx < len(items)) else String("")
        self._draw_text(preview, box_r.x + Float32(6.0), box_r.y + Float32(2.0), self.theme.font_size * Float32(0.88), self.theme.text_primary)
        var ax = box_r.right() - Float32(10.0)
        var ay = box_r.center_y()
        self.draw_list.add_line(ax - Float32(4.0), ay - Float32(2.0), ax, ay + Float32(2.5), 1.5, self.theme.text_muted)
        self.draw_list.add_line(ax, ay + Float32(2.5), ax + Float32(4.0), ay - Float32(2.0), 1.5, self.theme.text_muted)
        self._draw_text(label_text, box_r.right() + Float32(8.0), r.y + Float32(3.0), self.theme.font_size, self.theme.text_primary)
        return changed

    def color_edit3(
        mut self,
        id: String,
        label_text: String,
        mut color: Vec3,
        min_val: Float32 = 0.0,
        max_val: Float32 = 1.0,
        w: Float32 = 260.0,
    ) -> Bool:
        var wid = self.widget_id(id)
        var safe_w = w if (isfinite(w) and w > Float32(0.0)) else Float32(260.0)
        var r = self._alloc_item_rect(wid, safe_w, self.theme.row_height)
        self._register_focusable(wid)

        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        var lo64 = Float64(min_val) if isfinite(min_val) else 0.0
        var hi64 = Float64(max_val) if isfinite(max_val) else 1.0
        if hi64 < lo64:
            var tmp_b = lo64
            lo64 = hi64
            hi64 = tmp_b
        var span64 = max(1.0e-6, hi64 - lo64)

        var had_non_finite = (not isfinite(color.x)) or (not isfinite(color.y)) or (not isfinite(color.z))
        var vals = Array[Float32, 3](fill=0.0)
        vals[0] = Float32(max(lo64, min(hi64, Float64(color.x)))) if isfinite(color.x) else Float32(max(lo64, min(hi64, 0.0)))
        vals[1] = Float32(max(lo64, min(hi64, Float64(color.y)))) if isfinite(color.y) else Float32(max(lo64, min(hi64, 0.0)))
        vals[2] = Float32(max(lo64, min(hi64, Float64(color.z)))) if isfinite(color.z) else Float32(max(lo64, min(hi64, 0.0)))

        var sw_sz = max(Float32(12.0), r.h - Float32(4.0))
        var sw_r = RectF(r.x, r.y + Float32(2.0), sw_sz, sw_sz)
        var sw_hov = (not ptr_blocked) and sw_r.contains(self.mouse_x, self.mouse_y)
        if sw_hov:
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(sw_r)
            if self.mouse_pressed_l:
                self._set_focused(wid)
                if self._open_popup_id == wid:
                    self._open_popup_id = 0
                    self._popup_kind = 0
                else:
                    self._open_popup_id = wid
                    self._popup_kind = 2

        var field_start = sw_r.right() + Float32(4.0)
        var field_end = r.x + r.w * Float32(0.68)
        var comp_w = max(Float32(16.0), (field_end - field_start - Float32(8.0)) / Float32(3.0))
        var axis_cols = Array[UInt32, 3](fill=0)
        axis_cols[0] = pack_rgba8(224, 82, 82, 255)
        axis_cols[1] = pack_rgba8(92, 198, 108, 255)
        axis_cols[2] = pack_rgba8(78, 148, 240, 255)

        var changed = False
        for c in range(3):
            var cid = self.widget_id(id + "#c" + String(c))
            self._register_focusable(cid, allow_auto_focus=False)
            var cr = RectF(field_start + Float32(c) * (comp_w + Float32(4.0)), r.y + Float32(2.0), comp_w, r.h - Float32(4.0))
            if not ptr_blocked and cr.contains(self.mouse_x, self.mouse_y):
                self.hovered_id = cid
                self.hovered_rect = self._clip_hovered_rect(cr)
                if self.mouse_pressed_l:
                    self.active_id = cid
                    self._set_focused(cid)
            if not blocked and self.active_id == cid and self.mouse_down_l and (not self.mouse_pressed_l) and isfinite(self.mouse_delta_x) and self.mouse_delta_x != Float32(0.0):
                var nv = Float32(max(lo64, min(hi64, Float64(vals[c]) + Float64(self.mouse_delta_x) * 0.005)))
                if isfinite(nv) and abs(Float64(nv) - Float64(vals[c])) > 1.0e-7:
                    vals[c] = nv
                    changed = True
            elif not blocked and self.focused_id == cid and (self.key_left_pressed or self.key_right_pressed):
                var step64 = span64 * 0.01 if self.key_right_pressed else -span64 * 0.01
                var nv = Float32(max(lo64, min(hi64, Float64(vals[c]) + step64)))
                if isfinite(nv) and nv != vals[c]:
                    vals[c] = nv
                    changed = True
            var comp_focus = self.show_focus_ring(cid)
            self.draw_list.add_rect(cr.x, cr.y, cr.w, cr.h, self.theme.bg_widget, 3.0, Float32(2.0) if comp_focus else Float32(1.0), self.theme.focus_ring if comp_focus else self.theme.border)
            self.draw_list.add_rect(cr.x, cr.y, 3.0, cr.h, axis_cols[c], 1.5)
            self._draw_text(format_float2(vals[c]), cr.x + Float32(6.0), cr.y + Float32(2.0), self.theme.font_size * Float32(0.84), self.theme.text_primary)

        if not blocked and self._open_popup_id == wid:
            self._popup_seen = True
            var row_h = self.theme.row_height
            var pad = Float32(6.0)
            var pop_h = Float32(3.0) * row_h + pad * Float32(2.0)
            var pop_y = sw_r.bottom() + Float32(2.0)
            if pop_y + pop_h > self.screen_h and sw_r.y - pop_h - Float32(2.0) >= Float32(0.0):
                pop_y = sw_r.y - pop_h - Float32(2.0)
            var pop_r = RectF(sw_r.x, pop_y, Float32(180.0), pop_h)
            if pop_r.contains(self.mouse_x, self.mouse_y):
                self.hovered_id = wid
                self.hovered_rect = pop_r
                self._popup_consumed_click = True
                self._popup_consumed_rect = pop_r
            elif self.mouse_pressed_l and not sw_hov:
                self._open_popup_id = 0
                self._popup_kind = 0
            if self.key_esc_pressed:
                self._open_popup_id = 0
                self._popup_kind = 0
            if self._open_popup_id == wid:
                var bar_w = max(Float32(12.0), pop_r.w - pad * Float32(2.0))
                for c in range(3):
                    var pid = self.widget_id(id + "#pop" + String(c))
                    var br = RectF(pop_r.x + pad, pop_r.y + pad + Float32(c) * row_h + Float32(2.0), bar_w, max(Float32(6.0), row_h - Float32(6.0)))
                    if br.contains(self.mouse_x, self.mouse_y) and self.mouse_pressed_l:
                        self.active_id = pid
                    if self.active_id == pid and self.mouse_down_l and isfinite(self.mouse_x):
                        var t64 = max(0.0, min(1.0, (Float64(self.mouse_x) - Float64(br.x)) / Float64(bar_w)))
                        var nv = Float32(max(lo64, min(hi64, lo64 + t64 * span64)))
                        if isfinite(nv) and abs(Float64(nv) - Float64(vals[c])) > 1.0e-7:
                            vals[c] = nv
                            changed = True
                self._popup_kind = 2
                self._popup_rect = pop_r
                self._popup_owner_rect = sw_r
                self._popup_color = Vec3(
                    Float32((Float64(vals[0]) - lo64) / span64),
                    Float32((Float64(vals[1]) - lo64) / span64),
                    Float32((Float64(vals[2]) - lo64) / span64),
                )

        if changed or had_non_finite or color.x != vals[0] or color.y != vals[1] or color.z != vals[2]:
            color.x = vals[0]; color.y = vals[1]; color.z = vals[2]

        var r8 = Int(max(Float32(0.0), min(Float32(1.0), color.x)) * Float32(255.0) + Float32(0.5))
        var g8 = Int(max(Float32(0.0), min(Float32(1.0), color.y)) * Float32(255.0) + Float32(0.5))
        var b8 = Int(max(Float32(0.0), min(Float32(1.0), color.z)) * Float32(255.0) + Float32(0.5))
        self.draw_list.add_rect(sw_r.x, sw_r.y, sw_r.w, sw_r.h, pack_rgba8(r8, g8, b8, 255), 3.0, 1.0, self.theme.border)
        self._draw_text(label_text, field_end + Float32(6.0), r.y + Float32(3.0), self.theme.font_size, self.theme.text_primary)
        return changed

    # ---- Clipboard & IME API ----

    def set_clipboard(mut self, text: String):
        self.clipboard_text = text

    def get_clipboard(self) -> String:
        return self.clipboard_text

    def has_clipboard_out(self) -> Bool:
        return self.clipboard_out_dirty

    def take_clipboard_out(mut self) -> String:
        var s = self.clipboard_out
        self.clipboard_out = ""
        self.clipboard_out_dirty = False
        return s

    def wants_clipboard_paste(self, input: InputState) -> Bool:
        var ctrl = input.key_down(UI_KEY_LEFT_CTRL) or input.key_down(UI_KEY_RIGHT_CTRL)
        return ctrl and input.key_pressed(UI_KEY_V) and not self.ime_active

    def set_ime_preedit(mut self, preedit: String, cursor_cp: Int = -1, sel_len: Int = 0):
        self._ime_suppressed = False
        self.ime_preedit = preedit
        self.ime_active = preedit.byte_length() > 0
        if cursor_cp < 0:
            self.ime_cursor_cp = len(decode_utf8(preedit))
        else:
            self.ime_cursor_cp = cursor_cp
        self.ime_sel_len = max(0, sel_len)
        if not self.ime_active:
            self.ime_widget_id = 0
            self.ime_widget_name = ""

    def commit_ime(mut self, text: String):
        self._ime_suppressed = False
        self.ime_commit += text
        self.ime_active = False
        self.ime_preedit = ""
        self.ime_cursor_cp = 0
        self.ime_sel_len = 0

    def cancel_ime(mut self):
        if self._prev_input_ime_active and self.ime_preedit.byte_length() > 0:
            self._ime_suppressed = True
            self._cancelled_ime_preedit = self.ime_preedit
            self._cancelled_ime_cursor_cp = self.ime_cursor_cp
            self._cancelled_ime_sel_len = self.ime_sel_len
        self.ime_active = False
        self.ime_preedit = ""
        self.ime_cursor_cp = 0
        self.ime_sel_len = 0
        self.ime_commit = ""
        self.ime_widget_id = 0
        self.ime_widget_name = ""

    def active_ime_preedit(self, id: String) -> String:
        if not self.ime_active:
            return ""
        var wid = self.widget_id(id)
        if self.ime_widget_id == wid or self.ime_widget_name == id or self.focused_id == wid:
            return self.ime_preedit
        return ""

    # ---- UTF-8 Multi-Byte-Safe TextInput ----

    def text_cursor_codepoint(mut self, id: String) -> Int:
        var st = self._get_state(self.widget_id(id))
        return st.cursor_cp

    def text_scroll_x(mut self, id: String) -> Float32:
        var st = self._get_state(self.widget_id(id))
        return st.scroll_x

    def text_selection_range_cp(mut self, id: String) -> Tuple[Int, Int]:
        var st = self._get_state(self.widget_id(id))
        if st.select_anchor_cp < 0 or st.select_anchor_cp == st.cursor_cp:
            return (st.cursor_cp, st.cursor_cp)
        return (min(st.cursor_cp, st.select_anchor_cp), max(st.cursor_cp, st.select_anchor_cp))

    def text_selection_byte_range(mut self, id: String, text: String) -> Tuple[Int, Int]:
        var codes = decode_utf8(text)
        var rng = self.text_selection_range_cp(id)
        var lo = max(0, min(len(codes), rng[0]))
        var hi = max(0, min(len(codes), rng[1]))
        var b0 = encode_utf8(codes, 0, lo).byte_length()
        var b1 = encode_utf8(codes, 0, hi).byte_length()
        return (b0, b1)

    def text_cursor_byte_offset(mut self, id: String, text: String) -> Int:
        """Returns the exact UTF-8 byte boundary corresponding to `cursor_cp`."""
        var codes = decode_utf8(text)
        var st = self._get_state(self.widget_id(id))
        var cp = max(0, min(len(codes), st.cursor_cp))
        var prefix = encode_utf8(codes, 0, cp)
        return prefix.byte_length()

    def _hit_test_codepoint_x(mut self, codes: List[Int], rel_x: Float32, size_px: Float32) -> Int:
        """Caret index nearest `rel_x`. Boundaries are measured exactly like the caret and the
        selection (`_measure_text` of the prefix: shaped clusters, kerning and ligatures
        included), so a click lands where the caret is then drawn. Binary search: O(log n) measures."""
        var n = len(codes)
        if rel_x <= Float32(0.0) or n == 0 or not isfinite(rel_x):
            return 0
        if not self.font.valid:
            var adv = size_px * Float32(0.56)
            for i in range(n):
                if rel_x < (Float32(i) + Float32(0.5)) * adv:
                    return i
            return n
        var lo = 0
        var hi = n
        while lo < hi:
            var m = (lo + hi) // 2
            var w0 = self._measure_text(encode_utf8(codes, 0, m), size_px) if m > 0 else Float32(0.0)
            var w1 = self._measure_text(encode_utf8(codes, 0, m + 1), size_px)
            if rel_x < (w0 + w1) * Float32(0.5):
                hi = m
            else:
                lo = m + 1
        return lo

    @staticmethod
    def _is_word_cp(cp: Int) -> Bool:
        if (cp >= 48 and cp <= 57) or (cp >= 65 and cp <= 90) or (cp >= 97 and cp <= 122) or cp == 95:
            return True
        return cp >= 128 and cp != 0x3000 and cp != 0xA0

    @staticmethod
    def _word_bounds_at(codes: List[Int], cp: Int) -> Tuple[Int, Int]:
        var n = len(codes)
        if n == 0:
            return (0, 0)
        var idx = max(0, min(n - 1, cp))
        var is_w = UiContext._is_word_cp(codes[idx])
        var lo = idx
        while lo > 0 and UiContext._is_word_cp(codes[lo - 1]) == is_w:
            lo -= 1
        var hi = idx + 1
        while hi < n and UiContext._is_word_cp(codes[hi]) == is_w:
            hi += 1
        return (lo, hi)

    @staticmethod
    def _delete_codepoint_range(codes: List[Int], lo: Int, hi: Int) -> List[Int]:
        var out = List[Int](capacity=max(0, len(codes) - (hi - lo)))
        for i in range(lo):
            out.append(codes[i])
        for i in range(hi, len(codes)):
            out.append(codes[i])
        return out^

    @staticmethod
    def _insert_codepoints_replacing(codes: List[Int], lo: Int, hi: Int, ins: List[Int]) -> List[Int]:
        var out = List[Int](capacity=max(0, len(codes) - (hi - lo) + len(ins)))
        for i in range(lo):
            out.append(codes[i])
        for c in ins:
            out.append(c)
        for i in range(hi, len(codes)):
            out.append(codes[i])
        return out^

    @staticmethod
    def _filter_single_line_codes(text: String) -> List[Int]:
        var ins_codes = decode_utf8(text)
        var filtered = List[Int]()
        for c in ins_codes:
            if c >= 32 and c != 127:
                filtered.append(c)
        return filtered^

    def text_input(
        mut self,
        id: String,
        mut text: String,
        w: Float32 = 220.0,
        placeholder: String = "",
    ) -> Bool:
        """UTF-8 multi-byte-safe single-line text editor operating strictly on Unicode
        scalar codepoints (`decode_utf8` / `encode_utf8`) with clipboard (`Ctrl+C/V/X/A`),
        mouse-click/drag/double-click selection, horizontal scroll, and IME composition."""
        var wid = self.widget_id(id)
        var safe_w = w if (isfinite(w) and w > Float32(0.0)) else Float32(220.0)
        var r = self._alloc_item_rect(wid, safe_w, self.theme.row_height + Float32(4.0))
        self._register_focusable(wid, allow_auto_focus=False)
        self._text_widget_ids.append(wid)

        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        var hovered = (not ptr_blocked) and r.contains(self.mouse_x, self.mouse_y)
        if hovered:
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(r)
            if self.mouse_pressed_l:
                self._set_focused(wid)
                self.active_id = wid

        var codes = decode_utf8(text)
        var st = self._get_state(wid)
        st.cursor_cp = max(0, min(len(codes), st.cursor_cp))
        if st.select_anchor_cp >= 0:
            st.select_anchor_cp = max(0, min(len(codes), st.select_anchor_cp))

        var pad_x = r.x + Float32(8.0)
        var inner_w = max(Float32(1.0), r.w - Float32(16.0))
        var text_y = r.y + Float32(4.0)
        var font_size = self.theme.font_size

        var edited = False
        var focused = (not blocked) and self.focused_id == wid
        if focused:
            self._focused_text_id = wid
            self._focused_text_seen = True

        if not focused and self.ime_active and self.ime_widget_id == wid:
            self.cancel_ime()
        if focused and self.ime_active and self.ime_widget_id != wid:
            if self.ime_widget_id != 0:
                self.cancel_ime()
            else:
                self.ime_widget_id = wid
                self.ime_widget_name = id
                self._ime_widget_seen = True

        # Mouse click, double-click word select, and drag selection
        if not blocked:
            if hovered and self.mouse_pressed_l:
                if self.ime_active and self.ime_widget_id == wid:
                    self.cancel_ime()
                var hit_cp = self._hit_test_codepoint_x(codes, self.mouse_x - (pad_x - st.scroll_x), font_size)
                if (self.ui_time_sec - st.last_click_time) <= Float32(0.40) and abs(hit_cp - st.last_click_cp) <= 1 and len(codes) > 0:
                    var probe_cp = hit_cp - 1 if hit_cp >= len(codes) else hit_cp
                    var wb = Self._word_bounds_at(codes, probe_cp)
                    st.select_anchor_cp = wb[0]
                    st.cursor_cp = wb[1]
                    st.drag_selecting = False
                    st.last_click_time = Float32(-10.0)
                else:
                    if self.key_shift:
                        if st.select_anchor_cp < 0:
                            st.select_anchor_cp = st.cursor_cp
                        st.cursor_cp = hit_cp
                    else:
                        st.cursor_cp = hit_cp
                        st.select_anchor_cp = hit_cp
                    st.drag_selecting = True
                    st.last_click_time = self.ui_time_sec
                    st.last_click_cp = hit_cp
            elif self.active_id == wid and self.mouse_down_l and not self.mouse_pressed_l and st.drag_selecting:
                var hit_cp = self._hit_test_codepoint_x(codes, self.mouse_x - (pad_x - st.scroll_x), font_size)
                st.cursor_cp = hit_cp
            if not self.mouse_down_l:
                st.drag_selecting = False
                if st.select_anchor_cp == st.cursor_cp:
                    st.select_anchor_cp = -1

        if focused:
            if self.ime_commit.byte_length() > 0:
                if self.ime_widget_id == 0 or self.ime_widget_id == wid:
                    var ins = Self._filter_single_line_codes(self.ime_commit)
                    var lo = st.cursor_cp
                    var hi = st.cursor_cp
                    if st.select_anchor_cp >= 0 and st.select_anchor_cp != st.cursor_cp:
                        lo = min(st.cursor_cp, st.select_anchor_cp)
                        hi = max(st.cursor_cp, st.select_anchor_cp)
                    if len(ins) > 0 or lo != hi:
                        codes = Self._insert_codepoints_replacing(codes, lo, hi, ins)
                        st.cursor_cp = lo + len(ins)
                        st.select_anchor_cp = -1
                        edited = True
                    self.ime_commit = ""
                    if not self.ime_active or self.ime_preedit.byte_length() == 0:
                        self.ime_active = False
                        self.ime_preedit = ""
                        self.ime_cursor_cp = 0
                        self.ime_sel_len = 0
                        self.ime_widget_id = 0
                        self.ime_widget_name = ""
                    elif self.ime_widget_id == 0:
                        self.ime_widget_id = wid
                        self.ime_widget_name = id
                        self._ime_widget_seen = True

            if self.ime_active:
                if self.key_esc_pressed:
                    self.cancel_ime()
                elif self.key_enter_pressed:
                    var ins = Self._filter_single_line_codes(self.ime_preedit)
                    if len(ins) > 0:
                        var lo = st.cursor_cp
                        var hi = st.cursor_cp
                        if st.select_anchor_cp >= 0 and st.select_anchor_cp != st.cursor_cp:
                            lo = min(st.cursor_cp, st.select_anchor_cp)
                            hi = max(st.cursor_cp, st.select_anchor_cp)
                        codes = Self._insert_codepoints_replacing(codes, lo, hi, ins)
                        st.cursor_cp = lo + len(ins)
                        st.select_anchor_cp = -1
                        edited = True
                    self.cancel_ime()
            else:
                var has_sel = st.select_anchor_cp >= 0 and st.select_anchor_cp != st.cursor_cp
                var sel_lo = min(st.cursor_cp, st.select_anchor_cp) if has_sel else st.cursor_cp
                var sel_hi = max(st.cursor_cp, st.select_anchor_cp) if has_sel else st.cursor_cp

                # Ctrl+A: select all codepoints
                if self.key_ctrl and self.key_a_pressed:
                    st.select_anchor_cp = 0
                    st.cursor_cp = len(codes)

                # Ctrl+C: copy selection
                if self.key_ctrl and self.key_c_pressed and has_sel:
                    var slice = encode_utf8(codes, sel_lo, sel_hi)
                    self.clipboard_text = slice
                    self.clipboard_out = slice
                    self.clipboard_out_dirty = True

                # Ctrl+X: cut selection
                if self.key_ctrl and self.key_x_pressed and has_sel:
                    var slice = encode_utf8(codes, sel_lo, sel_hi)
                    self.clipboard_text = slice
                    self.clipboard_out = slice
                    self.clipboard_out_dirty = True
                    codes = Self._delete_codepoint_range(codes, sel_lo, sel_hi)
                    st.cursor_cp = sel_lo
                    st.select_anchor_cp = -1
                    edited = True

                # Ctrl+V: paste clipboard replacing selection
                if self.key_ctrl and self.key_v_pressed and self.clipboard_text.byte_length() > 0:
                    var ins = Self._filter_single_line_codes(self.clipboard_text)
                    if len(ins) > 0:
                        var lo = st.cursor_cp
                        var hi = st.cursor_cp
                        if st.select_anchor_cp >= 0 and st.select_anchor_cp != st.cursor_cp:
                            lo = min(st.cursor_cp, st.select_anchor_cp)
                            hi = max(st.cursor_cp, st.select_anchor_cp)
                        codes = Self._insert_codepoints_replacing(codes, lo, hi, ins)
                        st.cursor_cp = lo + len(ins)
                        st.select_anchor_cp = -1
                        edited = True

                # Navigation along codepoint boundaries
                if self.key_left_pressed:
                    if self.key_shift:
                        if st.select_anchor_cp < 0:
                            st.select_anchor_cp = st.cursor_cp
                        st.cursor_cp = max(0, st.cursor_cp - 1)
                    elif st.select_anchor_cp >= 0 and st.select_anchor_cp != st.cursor_cp:
                        st.cursor_cp = min(st.cursor_cp, st.select_anchor_cp)
                        st.select_anchor_cp = -1
                    else:
                        st.select_anchor_cp = -1
                        st.cursor_cp = max(0, st.cursor_cp - 1)

                if self.key_right_pressed:
                    if self.key_shift:
                        if st.select_anchor_cp < 0:
                            st.select_anchor_cp = st.cursor_cp
                        st.cursor_cp = min(len(codes), st.cursor_cp + 1)
                    elif st.select_anchor_cp >= 0 and st.select_anchor_cp != st.cursor_cp:
                        st.cursor_cp = max(st.cursor_cp, st.select_anchor_cp)
                        st.select_anchor_cp = -1
                    else:
                        st.select_anchor_cp = -1
                        st.cursor_cp = min(len(codes), st.cursor_cp + 1)

                if self.key_home_pressed:
                    if self.key_shift and st.select_anchor_cp < 0:
                        st.select_anchor_cp = st.cursor_cp
                    elif not self.key_shift:
                        st.select_anchor_cp = -1
                    st.cursor_cp = 0

                if self.key_end_pressed:
                    if self.key_shift and st.select_anchor_cp < 0:
                        st.select_anchor_cp = st.cursor_cp
                    elif not self.key_shift:
                        st.select_anchor_cp = -1
                    st.cursor_cp = len(codes)

                # Backspace (deletes selection or full multi-byte codepoint before cursor)
                if self.key_bs_pressed:
                    if st.select_anchor_cp >= 0 and st.select_anchor_cp != st.cursor_cp:
                        var lo = min(st.cursor_cp, st.select_anchor_cp)
                        var hi = max(st.cursor_cp, st.select_anchor_cp)
                        codes = Self._delete_codepoint_range(codes, lo, hi)
                        st.cursor_cp = lo
                        st.select_anchor_cp = -1
                        edited = True
                    elif st.cursor_cp > 0:
                        var del_at = st.cursor_cp - 1
                        _ = codes.pop(del_at)
                        st.cursor_cp = del_at
                        edited = True

                # Forward Delete (deletes selection or full multi-byte codepoint at cursor)
                if self.key_del_pressed:
                    if st.select_anchor_cp >= 0 and st.select_anchor_cp != st.cursor_cp:
                        var lo = min(st.cursor_cp, st.select_anchor_cp)
                        var hi = max(st.cursor_cp, st.select_anchor_cp)
                        codes = Self._delete_codepoint_range(codes, lo, hi)
                        st.cursor_cp = lo
                        st.select_anchor_cp = -1
                        edited = True
                    elif st.cursor_cp < len(codes):
                        _ = codes.pop(st.cursor_cp)
                        edited = True

                # Insert UTF-8 text from `input.text_input_utf8`
                if self.text_utf8.byte_length() > 0 and not self.key_ctrl:
                    var filtered = Self._filter_single_line_codes(self.text_utf8)
                    if len(filtered) > 0:
                        var lo = st.cursor_cp
                        var hi = st.cursor_cp
                        if st.select_anchor_cp >= 0 and st.select_anchor_cp != st.cursor_cp:
                            lo = min(st.cursor_cp, st.select_anchor_cp)
                            hi = max(st.cursor_cp, st.select_anchor_cp)
                        codes = Self._insert_codepoints_replacing(codes, lo, hi, filtered)
                        st.cursor_cp = lo + len(filtered)
                        st.select_anchor_cp = -1
                        edited = True

                if self.key_esc_pressed and self.modal_depth == 0:
                    self._focused_text_id = 0
                    self.focused_id = 0
                    focused = False

        if edited:
            text = encode_utf8(codes)

        # Horizontal scroll tracking to keep caret visible
        var show_ime = focused and self.ime_active and self.ime_preedit.byte_length() > 0
        var ime_codes = decode_utf8(self.ime_preedit) if show_ime else List[Int]()
        var ime_cur_cp = max(0, min(len(ime_codes), self.ime_cursor_cp)) if show_ime else 0
        var prefix_str = encode_utf8(codes, 0, st.cursor_cp)
        var prefix_w = self._measure_text(prefix_str, font_size)
        var ime_pre_cur_w = self._measure_text(encode_utf8(ime_codes, 0, ime_cur_cp), font_size) if show_ime else Float32(0.0)
        var caret_px = prefix_w + ime_pre_cur_w
        if caret_px < st.scroll_x:
            st.scroll_x = caret_px
        elif caret_px > st.scroll_x + inner_w:
            st.scroll_x = caret_px - inner_w
        var total_render_w = self._measure_text(text, font_size) + (self._measure_text(String(self.ime_preedit), font_size) if show_ime else Float32(0.0))
        var max_scroll_x = max(Float32(0.0), total_render_w - inner_w)
        st.scroll_x = max(Float32(0.0), min(max_scroll_x, st.scroll_x))

        self.state_map[wid] = st

        # Render input box, selection highlight, text/placeholder, IME preedit/underline, and caret
        self.draw_list.add_rect(
            r.x, r.y, r.w, r.h,
            self.theme.bg_window,
            self.theme.corner_radius,
            Float32(2.0) if focused else Float32(1.0),
            self.theme.focus_ring if focused else self.theme.border,
        )
        self.draw_list.push_clip_f(r.x + Float32(2.0), r.y + Float32(1.0), r.right() - Float32(2.0), r.bottom() - Float32(1.0))

        var draw_x = pad_x - st.scroll_x
        if len(codes) == 0 and placeholder.byte_length() > 0 and not focused and not show_ime:
            self._draw_text(placeholder, pad_x, text_y, font_size, self.theme.text_muted)
        elif show_ime:
            var preedit_txt = self.ime_preedit
            var preedit_w = self._measure_text(preedit_txt, font_size)
            var suffix_str = encode_utf8(codes, st.cursor_cp, len(codes))
            if st.cursor_cp > 0:
                self._draw_text(prefix_str, draw_x, text_y, font_size, self.theme.text_primary)
            var ime_x = draw_x + prefix_w
            if self.ime_sel_len > 0:
                var cl_end = min(len(ime_codes), ime_cur_cp + self.ime_sel_len)
                var cl_w = max(Float32(2.0), self._measure_text(encode_utf8(ime_codes, ime_cur_cp, cl_end), font_size))
                self.draw_list.add_rect(ime_x + ime_pre_cur_w, r.y + Float32(3.0), cl_w, r.h - Float32(6.0), self.theme.accent_dim, 2.0)
                self.draw_list.add_line(ime_x + ime_pre_cur_w, r.bottom() - Float32(4.0), ime_x + ime_pre_cur_w + cl_w, r.bottom() - Float32(4.0), 2.5, self.theme.focus_ring)
            self._draw_text(preedit_txt, ime_x, text_y, font_size, self.theme.text_primary)
            self.draw_list.add_line(ime_x, r.bottom() - Float32(4.0), ime_x + preedit_w, r.bottom() - Float32(4.0), 1.5, self.theme.accent)
            if st.cursor_cp < len(codes):
                self._draw_text(suffix_str, ime_x + preedit_w, text_y, font_size, self.theme.text_primary)
            var caret_x = ime_x + ime_pre_cur_w
            self.draw_list.add_line(caret_x, r.y + Float32(4.0), caret_x, r.bottom() - Float32(4.0), 1.6, self.theme.focus_ring)
        else:
            if focused and st.select_anchor_cp >= 0 and st.select_anchor_cp != st.cursor_cp:
                var lo = min(st.cursor_cp, st.select_anchor_cp)
                var hi = max(st.cursor_cp, st.select_anchor_cp)
                var x0 = draw_x + self._measure_text(encode_utf8(codes, 0, lo), font_size)
                var x1 = draw_x + self._measure_text(encode_utf8(codes, 0, hi), font_size)
                self.draw_list.add_rect(x0, r.y + Float32(3.0), max(Float32(2.0), x1 - x0), r.h - Float32(6.0), self.theme.accent_dim, 2.0)

            self._draw_text(text, draw_x, text_y, font_size, self.theme.text_primary)
            if focused:
                var caret_x = draw_x + prefix_w
                self.draw_list.add_line(caret_x, r.y + Float32(4.0), caret_x, r.bottom() - Float32(4.0), 1.6, self.theme.focus_ring)

        self.draw_list.pop_clip()
        return edited

    # ---- Tree Outliner (with < 0.1 ms 100k-item Offscreen Subtree Culling) ----

    @always_inline
    def tree_node_indexed(
        mut self,
        row_index: Int,
        label_text: StaticString = "Entity",
        depth: Int = 0,
        default_open: Bool = False,
    ) -> Bool:
        """Ultra-fast indexed tree row: culls offscreen rows in O(1) (`~0.3 ns`/row) before any
        string construction, hashing, or `UiDrawList` emission so 100k-item outliners run in `< 0.1 ms`."""
        var top_i = len(self.box_stack) - 1
        var row_h = self.theme.row_height
        var cy = self.box_stack[top_i].cursor_y
        if len(self.scroll_stack) > 0:
            var sc = self.scroll_stack[len(self.scroll_stack) - 1]
            if cy + row_h < sc.view_rect.y or cy > sc.view_rect.bottom():
                self.box_stack[top_i].cursor_y = cy + row_h + self.box_stack[top_i].gap
                self.culled_items_this_frame += 1
                var cached_open = self.open_tree_rows.get(row_index)
                if cached_open:
                    self.open_tree_row_epochs[row_index] = self.frame_epoch
                    return cached_open.value()
                return default_open

        var is_open = default_open
        var found = self.open_tree_rows.get(row_index)
        if found:
            is_open = found.value()
            self.open_tree_row_epochs[row_index] = self.frame_epoch

        var cx = self.box_stack[top_i].cursor_x + Float32(depth) * Float32(16.0)
        var rw = max(Float32(40.0), self.box_stack[top_i].rect.right() - cx)
        var r = RectF(cx, cy, rw, row_h)
        self.box_stack[top_i].cursor_y = cy + row_h + self.box_stack[top_i].gap

        if not self._blocked_by_modal() and not self._is_pointer_blocked_by_popup() and r.contains(self.mouse_x, self.mouse_y):
            self.hovered_id = UInt64(0x9E3779B97F4A7C15) ^ UInt64(row_index)
            self.hovered_rect = self._clip_hovered_rect(r)
            self.draw_list.add_rect(r.x, r.y, r.w, r.h, self.theme.bg_widget_hover, 3.0)
            if self.mouse_pressed_l:
                is_open = not is_open
                self.open_tree_rows[row_index] = is_open
                self.open_tree_row_epochs[row_index] = self.frame_epoch

        var ax = r.x + Float32(8.0)
        var ay = r.center_y()
        if is_open:
            self.draw_list.add_line(ax - Float32(4.0), ay - Float32(2.0), ax, ay + Float32(3.0), 1.6, self.theme.text_muted)
            self.draw_list.add_line(ax, ay + Float32(3.0), ax + Float32(4.0), ay - Float32(2.0), 1.6, self.theme.text_muted)
        else:
            self.draw_list.add_line(ax - Float32(2.0), ay - Float32(4.0), ax + Float32(3.0), ay, 1.6, self.theme.text_muted)
            self.draw_list.add_line(ax + Float32(3.0), ay, ax - Float32(2.0), ay + Float32(4.0), 1.6, self.theme.text_muted)

        self._draw_text(String(label_text) + " #" + String(row_index), r.x + Float32(18.0), r.y + Float32(3.0), self.theme.font_size, self.theme.text_primary)
        return is_open

    def clear_tree_rows(mut self):
        """Forgets every `tree_node_indexed` open/closed state (call when the indexed list changes)."""
        self.open_tree_rows.clear()
        self.open_tree_row_epochs.clear()

    def tree_node(mut self, id: String, label_text: String, default_open: Bool = False) -> Bool:
        """Collapsible hierarchical tree node; culls offscreen rows automatically inside `begin_scroll_view`."""
        var top_i = len(self.box_stack) - 1
        var row_h = self.theme.row_height
        var cy = self.box_stack[top_i].cursor_y
        var wid = self.widget_id(id)
        var st = self._get_state(wid)
        if not st.tree_initialized:
            st.tree_open = default_open
            st.tree_initialized = True
            self.state_map[wid] = st

        if len(self.scroll_stack) > 0:
            var sc = self.scroll_stack[len(self.scroll_stack) - 1]
            if cy + row_h < sc.view_rect.y or cy > sc.view_rect.bottom():
                _ = self._alloc_item_rect(wid, Float32(0.0), row_h)
                self.culled_items_this_frame += 1
                if st.tree_open:
                    self.box_stack[top_i].cursor_x += Float32(16.0)
                    self.id_stack.append(wid)
                return st.tree_open

        var r = self._alloc_item_rect(wid, 0.0, row_h)
        if not self._blocked_by_modal() and not self._is_pointer_blocked_by_popup(wid) and r.contains(self.mouse_x, self.mouse_y):
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(r)
            self.draw_list.add_rect(r.x, r.y, r.w, r.h, self.theme.bg_widget_hover, 3.0)
            if self.mouse_pressed_l:
                st.tree_open = not st.tree_open
                self.state_map[wid] = st

        var ax = r.x + Float32(8.0)
        var ay = r.center_y()
        if st.tree_open:
            self.draw_list.add_line(ax - Float32(4.0), ay - Float32(2.0), ax, ay + Float32(3.0), 1.6, self.theme.text_muted)
            self.draw_list.add_line(ax, ay + Float32(3.0), ax + Float32(4.0), ay - Float32(2.0), 1.6, self.theme.text_muted)
            self.box_stack[top_i].cursor_x += Float32(16.0)
            self.id_stack.append(wid)
        else:
            self.draw_list.add_line(ax - Float32(2.0), ay - Float32(4.0), ax + Float32(3.0), ay, 1.6, self.theme.text_muted)
            self.draw_list.add_line(ax + Float32(3.0), ay, ax - Float32(2.0), ay + Float32(4.0), 1.6, self.theme.text_muted)

        self._draw_text(label_text, r.x + Float32(20.0), r.y + Float32(3.0), self.theme.font_size, self.theme.text_primary)
        return st.tree_open

    def tree_pop(mut self):
        if len(self.id_stack) > 1:
            _ = self.id_stack.pop()
            var top_i = len(self.box_stack) - 1
            self.box_stack[top_i].cursor_x -= Float32(16.0)

    # ---- Quadratic Bezier Curve Editor ----

    def curve_editor(
        mut self,
        id: String,
        mut ctrl_x: Float32,
        mut ctrl_y: Float32,
        w: Float32 = 180.0,
        h: Float32 = 110.0,
    ) -> Bool:
        """Interactive normalized `[0, 1]^2` quadratic Bézier response curve editor
        emitting `add_line` grid/tangents and `add_quad_bezier` curve primitives."""
        var wid = self.widget_id(id)
        var r = self._alloc_item_rect(wid, w, h)
        var pad = Float32(8.0)
        var plot = r.inset(pad, pad, pad, pad)

        self.draw_list.add_rect(r.x, r.y, r.w, r.h, self.theme.bg_window, self.theme.corner_radius, 1.0, self.theme.border)
        # Quarter grid lines
        for g in range(1, 4):
            var t = Float32(g) * Float32(0.25)
            var gx = plot.x + plot.w * t
            var gy = plot.y + plot.h * t
            self.draw_list.add_line(gx, plot.y, gx, plot.bottom(), 1.0, pack_rgba8(45, 50, 62, 180))
            self.draw_list.add_line(plot.x, gy, plot.right(), gy, 1.0, pack_rgba8(45, 50, 62, 180))

        var p0x = plot.x
        var p0y = plot.bottom()
        var p1x = plot.right()
        var p1y = plot.y
        var had_non_finite = (not isfinite(ctrl_x)) or (not isfinite(ctrl_y))
        var tx = ctrl_x if isfinite(ctrl_x) else Float32(0.5)
        var ty = ctrl_y if isfinite(ctrl_y) else Float32(0.5)
        tx = max(Float32(0.0), min(Float32(1.0), tx))
        ty = max(Float32(0.0), min(Float32(1.0), ty))
        var cx = plot.x + tx * plot.w
        var cy = plot.bottom() - ty * plot.h
        if not isfinite(cx):
            cx = plot.x
        if not isfinite(cy):
            cy = plot.bottom()

        var changed = False
        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        if not ptr_blocked:
            if r.contains(self.mouse_x, self.mouse_y):
                self.hovered_id = wid
                self.hovered_rect = self._clip_hovered_rect(r)
            var dx = self.mouse_x - cx
            var dy = self.mouse_y - cy
            if isfinite(dx) and isfinite(dy):
                if self.mouse_pressed_l and (dx * dx + dy * dy <= Float32(100.0) or plot.contains(self.mouse_x, self.mouse_y)):
                    self.active_id = wid
        if (not blocked) and self.active_id == wid and self.mouse_down_l and plot.w > Float32(1.0) and plot.h > Float32(1.0):
            var nx = (self.mouse_x - plot.x) / plot.w
            var ny = (plot.bottom() - self.mouse_y) / plot.h
            if isfinite(nx) and isfinite(ny):
                tx = max(Float32(0.0), min(Float32(1.0), nx))
                ty = max(Float32(0.0), min(Float32(1.0), ny))
                cx = plot.x + tx * plot.w
                cy = plot.bottom() - ty * plot.h
                changed = True
        if changed or had_non_finite:
            ctrl_x = tx
            ctrl_y = ty

        self.draw_list.add_line(p0x, p0y, cx, cy, 1.0, self.theme.text_muted)
        self.draw_list.add_line(cx, cy, p1x, p1y, 1.0, self.theme.text_muted)
        self.draw_list.add_quad_bezier(p0x, p0y, cx, cy, p1x, p1y, 2.5, self.theme.accent)
        self.draw_list.add_circle(cx, cy, 5.0, self.theme.focus_ring, 1.5, pack_rgba8(255, 255, 255, 255))
        return changed

    # ---- Drag & Drop ----

    def begin_drag_source(mut self, payload_kind: String, int_val: Int = 0, str_val: String = "") -> Bool:
        """Starts a drag operation from the immediately preceding widget when dragged."""
        if self.last_item_id == 0:
            return False
        if self.drag_drop.active:
            return self.drag_drop.source_id == self.last_item_id
        if self._blocked_by_modal() or self._is_pointer_blocked_by_popup():
            return False
        # Arm on the press, over this item only. `begin_frame` starts the drag once the pointer has
        # travelled `UI_DRAG_START_PX` with the button held, so a plain click (or a press that
        # began elsewhere and slid over the item) never drags.
        if self.mouse_pressed_l and self.last_item_rect.contains(self.mouse_x, self.mouse_y):
            self._drag_pending = DragDropPayload(False, payload_kind, int_val, str_val, self.last_item_id, False)
            return False
        if self.mouse_down_l and self._drag_pending.source_id == self.last_item_id:
            var ddx = self.mouse_x - self._press_x
            var ddy = self.mouse_y - self._press_y
            if isfinite(ddx) and isfinite(ddy) and ddx * ddx + ddy * ddy >= UI_DRAG_START_PX * UI_DRAG_START_PX:
                self.drag_drop = DragDropPayload(True, payload_kind, int_val, str_val, self.last_item_id, False)
                self._drag_pending = DragDropPayload()
                return True
        return False

    def accept_drag_target_rect(mut self, payload_kind: String, r: RectF) -> Tuple[Bool, Int, String]:
        """Highlights `r` when hovered with matching `payload_kind` and delivers on mouse release."""
        if not self.drag_drop.active or self.drag_drop.kind != payload_kind:
            return (False, 0, "")
        if self._blocked_by_modal() or self._is_pointer_blocked_by_popup():
            return (False, 0, "")
        if r.contains(self.mouse_x, self.mouse_y):
            self.draw_list.add_rect(r.x, r.y, r.w, r.h, 0, self.theme.corner_radius, 2.0, self.theme.focus_ring)
            if self.mouse_released_l or not self.mouse_down_l:
                var iv = self.drag_drop.int_value
                var sv = self.drag_drop.str_value
                self.drag_drop = DragDropPayload()
                return (True, iv, sv)
        return (False, 0, "")

    def accept_drag_target(mut self, payload_kind: String) -> Tuple[Bool, Int, String]:
        """Highlights the preceding widget when hovered with matching `payload_kind` and delivers on mouse release."""
        var r = self.last_item_rect
        return self.accept_drag_target_rect(payload_kind, r)

    # ---- Focus-Trapping Modal Dialog ----

    def begin_modal(mut self, id: String, title: String, w: Float32, h: Float32, mut is_open: Bool) -> Bool:
        if not is_open:
            return False
        var wid = self.widget_id("modal:" + id)
        self.modal_depth += 1
        self.active_modal_id = wid
        self.had_modal_prev_frame = True  # Immediately trap background focus & input

        if (self.key_esc_pressed and not self.ime_active) or self.pad_cancel:
            is_open = False
            self.modal_depth -= 1
            if self.modal_depth == 0:
                self.active_modal_id = 0
                self.had_modal_prev_frame = False
            return False

        # Full-screen dim backdrop
        var safe_sw = max(Float32(0.0), self.screen_w) if isfinite(self.screen_w) else Float32(1280.0)
        var safe_sh = max(Float32(0.0), self.screen_h) if isfinite(self.screen_h) else Float32(720.0)
        var safe_w = max(Float32(0.0), w) if isfinite(w) else Float32(0.0)
        var safe_h = max(Float32(0.0), h) if isfinite(h) else Float32(0.0)
        self.draw_list.add_rect(0.0, 0.0, safe_sw, safe_sh, self.theme.modal_backdrop)

        var mx = max(Float32(0.0), (safe_sw - safe_w) * Float32(0.5))
        var my = max(Float32(0.0), (safe_sh - safe_h) * Float32(0.5))
        var mrect = RectF(mx, my, safe_w, safe_h)

        self.draw_list.add_rect_styled(
            mrect.x, mrect.y, mrect.w, mrect.h,
            self.theme.bg_panel,
            self.theme.focus_ring,
            self.theme.corner_radius + Float32(2.0),
            1.5,
            2.0,
            UI_FLAG_BORDER | UI_FLAG_DROP_SHADOW,
        )
        var hh = Float32(28.0)
        self.draw_list.add_rect(mrect.x, mrect.y, mrect.w, hh, self.theme.bg_header, self.theme.corner_radius + Float32(2.0))
        self._draw_text(title, mrect.x + Float32(12.0), mrect.y + Float32(5.0), self.theme.font_size, self.theme.text_primary)

        var body = RectF(mrect.x + Float32(12.0), mrect.y + hh + Float32(10.0), max(Float32(20.0), mrect.w - Float32(24.0)), max(Float32(20.0), mrect.h - hh - Float32(20.0)))
        var st = FlexStyle.col(gap=6.0, pad=0.0)
        st.fixed_w = body.w
        st.fixed_h = body.h
        var f_idx = self.flex.add_node(-1, st, body.w, body.h)
        self.dock_flex_root_idxs.append(f_idx)
        self.dock_flex_root_rects.append(body)
        self.box_stack.append(_BoxScope(wid, f_idx, body, body.x, body.y, FLEX_COL, 6.0, 0.0, 0.0, body.x, body.y))
        self.id_stack.append(wid)
        return True

    def end_modal(mut self):
        if self.modal_depth > 0:
            self.modal_depth -= 1
            if len(self.box_stack) > 1:
                _ = self.box_stack.pop()
                _ = self.id_stack.pop()

    # =========================================================================
    # Studio v2 Tactile Widgets (`StudioTheme` + `UiAnimator` + `:focus-visible`)
    # =========================================================================

    def _truncate_text_ellipsis(mut self, text: String, max_w: Float32, size_px: Float32) -> String:
        """Truncates UTF-8 text cleanly on codepoint boundaries with `'...'` to fit `max_w`."""
        if not isfinite(max_w) or max_w <= Float32(0.0):
            return ""
        if self._measure_text(text, size_px) <= max_w:
            return text
        var dots = String("...")
        var dots_w = self._measure_text(dots, size_px)
        if dots_w >= max_w:
            return dots
        var codes = decode_utf8(text)
        var n = len(codes)
        var lo = 0
        var hi = n
        while lo < hi:
            var mid = (lo + hi + 1) // 2
            var sub = List[Int](capacity=mid)
            for i in range(mid):
                sub.append(codes[i])
            if self._measure_text(encode_utf8(sub) + dots, size_px) <= max_w:
                lo = mid
            else:
                hi = mid - 1
        var out = List[Int](capacity=lo)
        for i in range(lo):
            out.append(codes[i])
        return encode_utf8(out) + dots

    def glass_pill(
        mut self,
        id: String,
        rect: RectF,
        corner_radius: Float32 = 8.0,
        blur_radius: Float32 = 3.0,
        drop_shadow: Bool = True,
    ) -> Bool:
        """Emits a floating translucent frosted-glass HUD pill (`L4_overlay` + 5-tap LDS blur + top bevel).
        Returns `True` when hovered (capturing pointer so clicks don't fall through to the 3D viewport)."""
        var wid = self.widget_id(id)
        var r = RectF(
            rect.x if isfinite(rect.x) else Float32(0.0),
            rect.y if isfinite(rect.y) else Float32(0.0),
            max(Float32(0.0), rect.w) if isfinite(rect.w) else Float32(0.0),
            max(Float32(0.0), rect.h) if isfinite(rect.h) else Float32(0.0),
        )
        self.last_item_id = wid
        self.last_item_rect = r
        var ptr_blocked = self._blocked_by_modal() or self._is_pointer_blocked_by_popup(wid)
        var hovered = (not ptr_blocked) and r.contains(self.mouse_x, self.mouse_y)
        if hovered:
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(r)
        self.draw_list.add_glass_panel(
            r.x, r.y, r.w, r.h,
            self.studio_theme.l4_overlay,
            max(Float32(0.0), corner_radius) if isfinite(corner_radius) else Float32(8.0),
            max(Float32(1.0), blur_radius) if isfinite(blur_radius) else Float32(3.0),
            1.0,
            self.studio_theme.border_strong,
            True,
            drop_shadow,
        )
        return hovered

    def icon_button(
        mut self,
        id: String,
        icon_id: Int,
        label_text: String = "",
        active: Bool = False,
        w: Float32 = 0.0,
        h: Float32 = 26.0,
        accent_override: UInt32 = 0,
    ) -> Bool:
        """Tactile Studio v2 icon/text button with spring hover lift, press inset & `:focus-visible` bloom."""
        var wid = self.widget_id(id)
        var bh = h if (isfinite(h) and h > Float32(0.0)) else Float32(26.0)
        var icon_sz = max(Float32(12.0), min(Float32(18.0), bh - Float32(10.0)))
        var font_sz = self.studio_theme.font_body
        var has_lbl = label_text.byte_length() > 0
        var auto_w = (
            icon_sz + Float32(14.0) + (self._measure_text(label_text, font_sz) + Float32(6.0) if has_lbl else Float32(0.0))
        )
        var bw = w if (isfinite(w) and w > Float32(0.0)) else max(bh, auto_w)
        var r = self._alloc_item_rect(wid, bw, bh)
        self._register_focusable(wid)

        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        var hovered = (not ptr_blocked) and r.contains(self.mouse_x, self.mouse_y)
        if hovered:
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(r)
            if self.mouse_pressed_l:
                self.active_id = wid
                self._set_focused(wid)

        var clicked = False
        if not blocked:
            if hovered and self.mouse_released_l and self.active_id == wid:
                clicked = True
            elif self.focused_id == wid and (self.key_enter_pressed or self.key_space_pressed or self.pad_confirm):
                clicked = True

        var pressed = (not blocked) and self.active_id == wid and self.mouse_down_l
        var f_vis = self.show_focus_ring(wid)
        var sp = self.animator.drive_widget(wid, hovered, pressed, f_vis)
        var h_t = sp[0]
        var p_t = sp[1]
        var f_t = sp[2]

        var acc = accent_override if accent_override != 0 else self.studio_theme.accent_primary
        var base_bg = rgba8_scale_alpha(acc, 0.22) if active else self.studio_theme.l3_raised
        var hov_bg = rgba8_scale_alpha(acc, 0.34) if active else self.studio_theme.l3_hover
        var bg = lerp_rgba8(lerp_rgba8(base_bg, hov_bg, h_t), self.studio_theme.l3_press, p_t)
        var brd = acc if active else lerp_rgba8(self.studio_theme.border_control, self.studio_theme.border_strong, h_t)

        # Spring visual offset (hit-testing strictly stays on logical rect `r`)
        var lift_y = Float32(-0.75) * h_t + Float32(0.75) * p_t
        var inset = Float32(0.5) * p_t
        var vr = RectF(r.x + inset, r.y + lift_y + inset, max(Float32(2.0), r.w - inset * Float32(2.0)), max(Float32(2.0), r.h - inset * Float32(2.0)))

        if f_t > Float32(0.02):
            self.draw_list.add_glow_rect(
                vr.x, vr.y, vr.w, vr.h,
                bg,
                rgba8_scale_alpha(self.studio_theme.focus_ring, f_t),
                RADIUS_SM,
                Float32(2.5) + Float32(4.0) * f_t,
                1.5,
                UI_FLAG_TOP_BEVEL,
            )
        else:
            self.draw_list.add_rect_styled(
                vr.x, vr.y, vr.w, vr.h,
                bg, brd,
                RADIUS_SM,
                1.0,
                1.0,
                UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )

        var fg = acc if active else lerp_rgba8(self.studio_theme.text_secondary, self.studio_theme.text_primary, max(h_t, Float32(0.5) if has_lbl else Float32(0.0)))
        if has_lbl:
            var tw = self._measure_text(label_text, font_sz)
            var total = icon_sz + Float32(6.0) + tw
            var sx = vr.x + max(Float32(6.0), (vr.w - total) * Float32(0.5))
            draw_studio_icon(self.draw_list, icon_id, sx, vr.center_y() - icon_sz * Float32(0.5), icon_sz, fg, 1.5)
            var avail_tw = max(Float32(8.0), vr.right() - (sx + icon_sz + Float32(6.0)) - Float32(4.0))
            var txt = self._truncate_text_ellipsis(label_text, avail_tw, font_sz)
            self._draw_text(txt, sx + icon_sz + Float32(6.0), vr.center_y() - font_sz * Float32(0.5), font_sz, self.studio_theme.text_primary if active else fg)
        else:
            draw_studio_icon(self.draw_list, icon_id, vr.center_x() - icon_sz * Float32(0.5), vr.center_y() - icon_sz * Float32(0.5), icon_sz, fg, 1.5)
        return clicked

    def collapsible_card(
        mut self,
        id: String,
        title: String,
        icon_id: Int = -1,
        default_open: Bool = True,
        w: Float32 = 0.0,
    ) -> Tuple[Bool, Float32]:
        """Obsidian `L2_card` collapsible section header with smooth spring chevron & expand factor `(is_open, expand_t)`."""
        var wid = self.widget_id(id)
        var top_i = len(self.box_stack) - 1
        var avail_w = max(Float32(60.0), self.box_stack[top_i].rect.w - self.box_stack[top_i].pad_r)
        var cw = w if (isfinite(w) and w > Float32(0.0)) else avail_w
        var r = self._alloc_item_rect(wid, cw, CARD_HEADER_H)
        self._register_focusable(wid)

        var st = self._get_state(wid)
        if not st.tree_initialized:
            st.tree_open = default_open
            st.tree_initialized = True
            self.state_map[wid] = st

        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        var hovered = (not ptr_blocked) and r.contains(self.mouse_x, self.mouse_y)
        if hovered:
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(r)
            if self.mouse_pressed_l:
                self._set_focused(wid)
                st.tree_open = not st.tree_open
                self.state_map[wid] = st
        elif (not blocked) and self.focused_id == wid and (self.key_enter_pressed or self.key_space_pressed or self.pad_confirm):
            st.tree_open = not st.tree_open
            self.state_map[wid] = st

        var sp = self.animator.drive_widget(wid, hovered, hovered and self.mouse_down_l, self.show_focus_ring(wid))
        var expand_t = self.animator.drive_expand(wid, st.tree_open)
        var bg = lerp_rgba8(self.studio_theme.l2_card, self.studio_theme.l3_raised, sp[0])
        var brd = self.studio_theme.focus_ring if sp[2] > Float32(0.05) else self.studio_theme.border_subtle

        self.draw_list.add_rounded_rect_ex(
            r.x, r.y, r.w, r.h,
            bg,
            RADIUS_MD, RADIUS_MD,
            Float32(0.0) if st.tree_open else RADIUS_MD,
            Float32(0.0) if st.tree_open else RADIUS_MD,
            1.0, brd, 1.0, UI_FLAG_TOP_BEVEL,
        )
        draw_studio_chevron(
            self.draw_list,
            r.x + Float32(8.0),
            r.center_y() - Float32(6.0),
            Float32(12.0),
            self.studio_theme.text_primary if st.tree_open else self.studio_theme.text_muted,
            expand_t,
            1.5,
        )
        var tx = r.x + Float32(24.0)
        if icon_id >= 0:
            draw_studio_icon(
                self.draw_list,
                icon_id,
                tx,
                r.center_y() - Float32(7.0),
                Float32(14.0),
                self.studio_theme.accent_primary if st.tree_open else self.studio_theme.text_secondary,
                1.4,
            )
            tx += Float32(20.0)
        var lbl = self._truncate_text_ellipsis(title, max(Float32(20.0), r.right() - tx - Float32(8.0)), self.studio_theme.font_section)
        self._draw_text(lbl, tx, r.center_y() - self.studio_theme.font_section * Float32(0.5), self.studio_theme.font_section, self.studio_theme.text_primary)
        return (st.tree_open, expand_t)

    def property_row_label(
        mut self,
        label_text: String,
        w: Float32 = 0.0,
        label_ratio: Float32 = INSPECTOR_LABEL_RATIO,
        id: String = "",
    ) -> Tuple[RectF, RectF]:
        """Allocates a Studio v2 left-label / right-control Inspector row with clean ellipsis truncation.
        Returns `(full_row_rect, right_control_rect)`. Pass the control's `id`: two rows that share
        a label (\"Position\" in two components) would otherwise share one retained row rect."""
        var wid = self.widget_id("prow:" + (id if id.byte_length() > 0 else label_text))
        var top_i = len(self.box_stack) - 1
        var avail_w = max(Float32(100.0), self.box_stack[top_i].rect.w - self.box_stack[top_i].pad_r)
        var rw = w if (isfinite(w) and w > Float32(0.0)) else avail_w
        var r = self._alloc_item_rect(wid, rw, INSPECTOR_ROW_H)
        var ratio = max(Float32(0.12), min(Float32(0.65), label_ratio if isfinite(label_ratio) else INSPECTOR_LABEL_RATIO))
        var lbl_w = max(Float32(24.0), r.w * ratio - Float32(4.0))
        var disp = self._truncate_text_ellipsis(label_text, lbl_w, self.studio_theme.font_body)
        self._draw_text(
            disp,
            r.x + Float32(2.0),
            r.center_y() - self.studio_theme.font_body * Float32(0.5),
            self.studio_theme.font_body,
            self.studio_theme.text_secondary,
        )
        var ctrl_x = r.x + r.w * ratio
        var ctrl_r = RectF(ctrl_x, r.y, max(Float32(24.0), r.right() - ctrl_x), r.h)
        return (r, ctrl_r)

    def drag_float3_v2(
        mut self,
        id: String,
        label_text: String,
        mut v: Vec3,
        speed: Float32 = 0.05,
        min_val: Float32 = -1.0e6,
        max_val: Float32 = 1.0e6,
        w: Float32 = 0.0,
        snap_step: Float32 = 0.25,
    ) -> Bool:
        """Left-label Studio v2 XYZ scrubber with color-coded X `#E85B5B` / Y `#4BC47D` / Z `#4E8FF7` capsules,
        `Shift = 0.1x` fine scrub, `Ctrl = snap_step` quantization, and spring hover/active bloom."""
        var top_i = len(self.box_stack) - 1
        var avail_w = max(Float32(100.0), self.box_stack[top_i].rect.w - self.box_stack[top_i].pad_r)
        var eff_w = w if (isfinite(w) and w > Float32(0.0)) else avail_w
        var lbl_ratio = min(INSPECTOR_LABEL_RATIO, Float32(44.0) / max(Float32(100.0), eff_w))
        var row_pair = self.property_row_label(label_text, w, label_ratio=lbl_ratio, id=id)
        var ctrl = row_pair[1]
        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup()
        var gap = Float32(3.0)
        var cap_w = max(Float32(28.0), (ctrl.w - gap * Float32(2.0)) / Float32(3.0))

        var lo64 = Float64(min_val) if isfinite(min_val) else -1.0e6
        var hi64 = Float64(max_val) if isfinite(max_val) else 1.0e6
        if hi64 < lo64:
            var tmp_b = lo64
            lo64 = hi64
            hi64 = tmp_b
        var sp64 = Float64(speed) if (isfinite(speed) and speed > Float32(0.0)) else 0.05
        var snap64 = Float64(snap_step) if (isfinite(snap_step) and snap_step > Float32(1.0e-5)) else 0.25
        var had_non_finite = (not isfinite(v.x)) or (not isfinite(v.y)) or (not isfinite(v.z))
        var vals = Array[Float32, 3](fill=0.0)
        vals[0] = v.x if isfinite(v.x) else Float32(max(lo64, min(hi64, 0.0)))
        vals[1] = v.y if isfinite(v.y) else Float32(max(lo64, min(hi64, 0.0)))
        vals[2] = v.z if isfinite(v.z) else Float32(max(lo64, min(hi64, 0.0)))

        var axis_cols = Array[UInt32, 3](fill=0)
        axis_cols[0] = pack_rgba8(232, 91, 91, 255)   # X #E85B5B
        axis_cols[1] = pack_rgba8(75, 196, 125, 255)  # Y #4BC47D
        axis_cols[2] = pack_rgba8(78, 143, 247, 255)  # Z #4E8FF7
        var axis_lbls = [String("X"), String("Y"), String("Z")]
        var changed = False

        for c in range(3):
            var cid = self.widget_id(id + "#v2_" + String(c))
            self._register_focusable(cid)
            var cr = RectF(ctrl.x + Float32(c) * (cap_w + gap), ctrl.y + Float32(1.0), cap_w, max(Float32(16.0), ctrl.h - Float32(2.0)))
            var hov = (not ptr_blocked) and cr.contains(self.mouse_x, self.mouse_y)
            if hov:
                self.hovered_id = cid
                self.hovered_rect = self._clip_hovered_rect(cr)
                if self.mouse_pressed_l:
                    self.active_id = cid
                    self._set_focused(cid)

            var is_drag = (not blocked) and self.active_id == cid and self.mouse_down_l
            if is_drag and (not self.mouse_pressed_l) and isfinite(self.mouse_delta_x) and self.mouse_delta_x != Float32(0.0):
                var fine = 0.1 if self.key_shift else 1.0
                var raw64 = Float64(vals[c]) + Float64(self.mouse_delta_x) * sp64 * fine
                if self.key_ctrl:
                    raw64 = floor(raw64 / snap64 + 0.5) * snap64
                var nv = Float32(max(lo64, min(hi64, raw64)))
                if isfinite(nv) and abs(Float64(nv) - Float64(vals[c])) > 1.0e-7:
                    vals[c] = nv
                    changed = True
            elif (not blocked) and self.focused_id == cid:
                var step_mul = 0.1 if self.key_shift else (snap64 / max(1.0e-6, sp64) if self.key_ctrl else 1.0)
                if self.key_left_pressed:
                    var nv = Float32(max(lo64, Float64(vals[c]) - sp64 * step_mul))
                    if isfinite(nv):
                        vals[c] = nv
                        changed = True
                elif self.key_right_pressed:
                    var nv = Float32(min(hi64, Float64(vals[c]) + sp64 * step_mul))
                    if isfinite(nv):
                        vals[c] = nv
                        changed = True

            var sp_a = self.animator.drive_widget(cid, hov, is_drag, self.show_focus_ring(cid))
            var bloom = max(sp_a[1], sp_a[2])
            var bg = lerp_rgba8(self.studio_theme.l0_canvas, self.studio_theme.l2_card, max(sp_a[0], sp_a[1]))
            var brd = lerp_rgba8(self.studio_theme.border_control, axis_cols[c], max(sp_a[0] * Float32(0.55), bloom))

            if bloom > Float32(0.02):
                self.draw_list.add_glow_rect(
                    cr.x, cr.y, cr.w, cr.h,
                    bg,
                    rgba8_scale_alpha(axis_cols[c], bloom * Float32(0.75)),
                    RADIUS_SM,
                    Float32(2.0) + Float32(4.0) * bloom,
                    1.2,
                )
            else:
                self.draw_list.add_rect_styled(
                    cr.x, cr.y, cr.w, cr.h,
                    bg, brd,
                    RADIUS_SM, 1.0, 1.0, UI_FLAG_BORDER,
                )

            # Axis color pill cap on left edge of capsule
            var tag_w = Float32(13.0)
            self.draw_list.add_rounded_rect_ex(
                cr.x + Float32(1.0), cr.y + Float32(1.0), tag_w, max(Float32(4.0), cr.h - Float32(2.0)),
                rgba8_scale_alpha(axis_cols[c], Float32(0.24) + Float32(0.26) * max(sp_a[0], sp_a[1])),
                RADIUS_SM - Float32(1.0), 0.0, 0.0, RADIUS_SM - Float32(1.0),
            )
            self._draw_text(
                axis_lbls[c],
                cr.x + Float32(3.5),
                cr.center_y() - self.studio_theme.font_caption * Float32(0.5),
                self.studio_theme.font_caption,
                axis_cols[c],
            )
            var val_txt = format_float2(vals[c])
            var disp_val = self._truncate_text_ellipsis(val_txt, max(Float32(10.0), cr.w - tag_w - Float32(4.0)), self.studio_theme.font_mono)
            self._draw_text(
                disp_val,
                cr.x + tag_w + Float32(3.0),
                cr.center_y() - self.studio_theme.font_mono * Float32(0.5),
                self.studio_theme.font_mono,
                self.studio_theme.text_primary,
            )

        if changed or had_non_finite:
            v.x = vals[0]; v.y = vals[1]; v.z = vals[2]
        return changed

    def slider_float_v2(
        mut self,
        id: String,
        label_text: String,
        mut value: Float32,
        min_val: Float32,
        max_val: Float32,
        w: Float32 = 0.0,
    ) -> Bool:
        """Left-label Studio v2 slider with recessed pill track, bounded spring-scaled thumb & right numeric readout."""
        var wid = self.widget_id(id)
        var row_pair = self.property_row_label(label_text, w, id=id)
        var ctrl = row_pair[1]
        self._register_focusable(wid)

        var readout_w = Float32(44.0)
        var track_w = max(Float32(36.0), ctrl.w - readout_w - Float32(6.0))
        var track = RectF(ctrl.x, ctrl.center_y() - Float32(3.0), track_w, Float32(6.0))
        var hit_r = RectF(ctrl.x, ctrl.y, track_w, ctrl.h)

        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        var hovered = (not ptr_blocked) and hit_r.contains(self.mouse_x, self.mouse_y)
        if hovered:
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(hit_r)
            if self.mouse_pressed_l:
                self.active_id = wid
                self._set_focused(wid)

        var lo64 = Float64(min_val) if isfinite(min_val) else 0.0
        var hi64 = Float64(max_val) if isfinite(max_val) else (lo64 + 1.0)
        if hi64 < lo64:
            var tmp_b = lo64
            lo64 = hi64
            hi64 = tmp_b
        var span64 = max(1.0e-6, hi64 - lo64)
        var was_finite = isfinite(value)
        var val64 = max(lo64, min(hi64, Float64(value) if was_finite else lo64))
        if not was_finite:
            value = Float32(val64)

        var changed = False
        var dragging = (not blocked) and self.active_id == wid and self.mouse_down_l
        if dragging and isfinite(self.mouse_x):
            var t64 = max(0.0, min(1.0, (Float64(self.mouse_x) - Float64(track.x)) / max(1.0, Float64(track.w))))
            var nv = Float32(max(lo64, min(hi64, lo64 + t64 * span64)))
            if isfinite(nv) and abs(Float64(nv) - Float64(value)) > 1.0e-6:
                value = nv
                changed = True
        elif (not blocked) and self.focused_id == wid:
            var step = span64 * (0.01 if self.key_shift else 0.05)
            if self.key_left_pressed:
                value = Float32(max(lo64, val64 - step))
                changed = True
            elif self.key_right_pressed:
                value = Float32(min(hi64, val64 + step))
                changed = True

        var cur_v64 = Float64(value) if isfinite(value) else lo64
        var norm = Float32(max(0.0, min(1.0, (cur_v64 - lo64) / span64)))
        var sp = self.animator.drive_widget(wid, hovered, dragging, self.show_focus_ring(wid))
        var hot_t = max(sp[0], sp[1])

        # Recessed track + glowing accent fill + bounded thumb (never clips past track edges!)
        self.draw_list.add_rect_styled(
            track.x, track.y, track.w, track.h,
            self.studio_theme.l0_canvas,
            self.studio_theme.border_control,
            3.0, 1.0, 1.0, UI_FLAG_BORDER,
        )
        var fill_w = track.w * norm
        if fill_w > Float32(1.0):
            self.draw_list.add_rect(track.x, track.y, fill_w, track.h, self.studio_theme.accent_primary, 3.0)

        var thumb_r = Float32(5.0) + Float32(1.5) * hot_t
        var knob_x = max(track.x + thumb_r, min(track.right() - thumb_r, track.x + track.w * norm))
        if max(sp[1], sp[2]) > Float32(0.02):
            self.draw_list.add_glow_rect(
                knob_x - thumb_r, track.center_y() - thumb_r,
                thumb_r * Float32(2.0), thumb_r * Float32(2.0),
                self.studio_theme.text_primary,
                rgba8_scale_alpha(self.studio_theme.focus_ring, max(sp[1], sp[2])),
                thumb_r,
                Float32(3.0) + Float32(3.0) * max(sp[1], sp[2]),
            )
        self.draw_list.add_circle(
            knob_x, track.center_y(), thumb_r,
            self.studio_theme.text_primary,
            1.5,
            self.studio_theme.accent_primary,
        )

        var val_txt = format_float2(value)
        self._draw_text(
            val_txt,
            track.right() + Float32(6.0),
            ctrl.center_y() - self.studio_theme.font_mono * Float32(0.5),
            self.studio_theme.font_mono,
            self.studio_theme.text_primary if hot_t > Float32(0.1) else self.studio_theme.text_secondary,
        )
        return changed

    def toggle_switch(
        mut self,
        id: String,
        label_text: String,
        mut checked: Bool,
        w: Float32 = 0.0,
    ) -> Bool:
        """Left-label Studio v2 pill toggle switch with spring-gliding thumb & `:focus-visible` bloom."""
        var wid = self.widget_id(id)
        var row_pair = self.property_row_label(label_text, w, id=id)
        var ctrl = row_pair[1]
        self._register_focusable(wid)

        var sw_w = Float32(34.0)
        var sw_h = Float32(18.0)
        var sw_r = RectF(ctrl.x, ctrl.center_y() - sw_h * Float32(0.5), sw_w, sw_h)

        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        var hovered = (not ptr_blocked) and sw_r.contains(self.mouse_x, self.mouse_y)
        var toggled = False
        if hovered:
            self.hovered_id = wid
            self.hovered_rect = self._clip_hovered_rect(sw_r)
            if self.mouse_pressed_l:
                self._set_focused(wid)
                checked = not checked
                toggled = True
        elif (not blocked) and self.focused_id == wid and (self.key_enter_pressed or self.key_space_pressed or self.pad_confirm):
            checked = not checked
            toggled = True

        var sp = self.animator.drive_widget(wid, hovered, hovered and self.mouse_down_l, self.show_focus_ring(wid))
        var slide_t = self.animator.drive_aux(wid, Float32(1.0) if checked else Float32(0.0), SpringConfig.snappy())
        var off_col = lerp_rgba8(self.studio_theme.l3_raised, self.studio_theme.l3_hover, sp[0])
        var track_col = lerp_rgba8(off_col, self.studio_theme.accent_primary, slide_t)

        if sp[2] > Float32(0.02):
            self.draw_list.add_glow_rect(
                sw_r.x, sw_r.y, sw_r.w, sw_r.h,
                track_col,
                rgba8_scale_alpha(self.studio_theme.focus_ring, sp[2]),
                sw_h * Float32(0.5),
                Float32(3.0) + Float32(3.0) * sp[2],
                1.0,
            )
        else:
            self.draw_list.add_rect_styled(
                sw_r.x, sw_r.y, sw_r.w, sw_r.h,
                track_col,
                self.studio_theme.border_control,
                sw_h * Float32(0.5), 1.0, 1.0, UI_FLAG_BORDER,
            )
        var thumb_r = Float32(6.5)
        var cx_lo = sw_r.x + Float32(2.5) + thumb_r
        var cx_hi = sw_r.right() - Float32(2.5) - thumb_r
        var thumb_cx = cx_lo + (cx_hi - cx_lo) * slide_t
        self.draw_list.add_circle(thumb_cx, sw_r.center_y(), thumb_r, self.studio_theme.text_primary)
        return toggled

    def tab_bar_animated(
        mut self,
        id: String,
        labels: List[String],
        mut active_idx: Int,
        w: Float32 = 0.0,
        h: Float32 = 28.0,
    ) -> Bool:
        """Horizontal tab strip with spring-gliding `UiTabIndicator` active pill. Returns `True` if `active_idx` changed."""
        var wid = self.widget_id(id)
        var top_i = len(self.box_stack) - 1
        var avail_w = max(Float32(80.0), self.box_stack[top_i].rect.w - self.box_stack[top_i].pad_r)
        var bw = w if (isfinite(w) and w > Float32(0.0)) else avail_w
        var bh = h if (isfinite(h) and h > Float32(0.0)) else Float32(28.0)
        var r = self._alloc_item_rect(wid, bw, bh)
        var n = len(labels)
        if n == 0:
            return False
        if active_idx < 0 or active_idx >= n:
            active_idx = 0

        self.draw_list.add_rect_styled(
            r.x, r.y, r.w, r.h,
            self.studio_theme.l0_canvas,
            self.studio_theme.border_subtle,
            RADIUS_SM, 1.0, 1.0, UI_FLAG_BORDER,
        )

        var blocked = self._blocked_by_modal()
        var ptr_blocked = blocked or self._is_pointer_blocked_by_popup(wid)
        var changed = False
        var tx = r.x + Float32(3.0)
        var act_x = tx
        var act_w = Float32(0.0)
        var font_sz = self.studio_theme.font_section

        for i in range(n):
            var tw = max(Float32(48.0), self._measure_text(labels[i], font_sz) + Float32(18.0))
            if tx + tw > r.right() - Float32(3.0):
                tw = max(Float32(24.0), r.right() - Float32(3.0) - tx)
            var tr = RectF(tx, r.y + Float32(3.0), tw, max(Float32(8.0), r.h - Float32(6.0)))
            var hov = (not ptr_blocked) and tr.contains(self.mouse_x, self.mouse_y)
            if hov:
                self.hovered_id = hash_ui_id(wid, String(i))
                self.hovered_rect = self._clip_hovered_rect(tr)
                if self.mouse_pressed_l and active_idx != i:
                    active_idx = i
                    changed = True
            if active_idx == i:
                act_x = tr.x
                act_w = tr.w
            tx += tw + Float32(3.0)

        # Glide active pill behind labels
        if act_w > Float32(1.0):
            var pill = self.animator.drive_tab_pill(wid, act_x, act_w)
            self.draw_list.add_rect_styled(
                pill[0], r.y + Float32(3.0), pill[1], max(Float32(8.0), r.h - Float32(6.0)),
                self.studio_theme.l3_raised,
                self.studio_theme.border_control,
                RADIUS_SM - Float32(1.0), 1.0, 1.0, UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )
            self.draw_list.add_glow_rect(
                pill[0] + Float32(6.0),
                r.bottom() - Float32(3.0),
                max(Float32(4.0), pill[1] - Float32(12.0)),
                Float32(2.0),
                self.studio_theme.accent_primary,
                rgba8_scale_alpha(self.studio_theme.accent_primary, 0.65),
                1.0, 5.0,
            )

        # Draw tab labels on top of sliding pill
        tx = r.x + Float32(3.0)
        for i in range(n):
            var tw = max(Float32(48.0), self._measure_text(labels[i], font_sz) + Float32(18.0))
            if tx + tw > r.right() - Float32(3.0):
                tw = max(Float32(24.0), r.right() - Float32(3.0) - tx)
            var tr = RectF(tx, r.y + Float32(3.0), tw, max(Float32(8.0), r.h - Float32(6.0)))
            var disp = self._truncate_text_ellipsis(labels[i], max(Float32(12.0), tr.w - Float32(10.0)), font_sz)
            var tw_m = self._measure_text(disp, font_sz)
            var col = self.studio_theme.text_primary if active_idx == i else self.studio_theme.text_secondary
            self._draw_text(
                disp,
                tr.x + max(Float32(4.0), (tr.w - tw_m) * Float32(0.5)),
                tr.center_y() - font_sz * Float32(0.5),
                font_sz,
                col,
            )
            tx += tw + Float32(3.0)
        return changed

    def badge_pill(
        mut self,
        text: String,
        bg_color: UInt32,
        fg_color: UInt32,
        x: Float32,
        y: Float32,
        h: Float32 = 20.0,
        glow: Bool = False,
    ) -> Float32:
        """Emits a compact pill status badge at `(x, y)` and returns its rendered width."""
        var bh = max(Float32(14.0), h if isfinite(h) else Float32(20.0))
        var font_sz = self.studio_theme.font_caption
        var tw = self._measure_text(text, font_sz)
        var bw = max(bh, tw + Float32(14.0))
        var rad = bh * Float32(0.5)
        if glow:
            self.draw_list.add_glow_rect(
                x, y, bw, bh, bg_color, rgba8_scale_alpha(fg_color, 0.45), rad, 6.0, 1.0, UI_FLAG_TOP_BEVEL
            )
        else:
            self.draw_list.add_rect_styled(
                x, y, bw, bh, bg_color, rgba8_with_alpha(fg_color, 64), rad, 1.0, 1.0, UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL
            )
        self._draw_text(text, x + (bw - tw) * Float32(0.5), y + (bh - font_sz) * Float32(0.5), font_sz, fg_color)
        return bw

    def sparkline(
        mut self,
        samples: List[Float32],
        rect: RectF,
        line_color: UInt32,
        bg_color: UInt32 = 0,
        min_val: Float32 = 0.0,
        max_val: Float32 = 0.0,
    ):
        """Renders a crisp GPU SDF polyline sparkline (e.g., 60-frame GPU ms trace) inside `rect`."""
        var r = RectF(
            rect.x if isfinite(rect.x) else Float32(0.0),
            rect.y if isfinite(rect.y) else Float32(0.0),
            max(Float32(0.0), rect.w) if isfinite(rect.w) else Float32(0.0),
            max(Float32(0.0), rect.h) if isfinite(rect.h) else Float32(0.0),
        )
        if r.w < Float32(4.0) or r.h < Float32(4.0):
            return
        if (bg_color >> 24) != 0:
            self.draw_list.add_rect_styled(
                r.x, r.y, r.w, r.h, bg_color, self.studio_theme.border_subtle, RADIUS_SM, 1.0, 1.0, UI_FLAG_BORDER
            )
        var n = len(samples)
        if n < 2:
            return
        var lo = min_val if isfinite(min_val) else Float32(0.0)
        var hi = max_val if isfinite(max_val) else Float32(0.0)
        if hi <= lo:
            lo = Float32(1.0e30)
            hi = Float32(-1.0e30)
            for i in range(n):
                if isfinite(samples[i]):
                    lo = min(lo, samples[i])
                    hi = max(hi, samples[i])
            if not isfinite(lo) or not isfinite(hi) or hi <= lo:
                lo = Float32(0.0)
                hi = Float32(1.0)
        var span = max(Float32(1.0e-5), hi - lo)
        var pad = Float32(2.0)
        var iw = max(Float32(2.0), r.w - pad * Float32(2.0))
        var ih = max(Float32(2.0), r.h - pad * Float32(2.0))
        var step_x = iw / Float32(n - 1)

        var prev_x = r.x + pad
        var s0 = samples[0] if isfinite(samples[0]) else lo
        var prev_y = r.bottom() - pad - max(Float32(0.0), min(Float32(1.0), (s0 - lo) / span)) * ih
        for i in range(1, n):
            var si = samples[i] if isfinite(samples[i]) else lo
            var cx = r.x + pad + Float32(i) * step_x
            var cy = r.bottom() - pad - max(Float32(0.0), min(Float32(1.0), (si - lo) / span)) * ih
            self.draw_list.add_line(prev_x, prev_y, cx, cy, 1.5, line_color)
            prev_x = cx
            prev_y = cy
        self.draw_list.add_circle(prev_x, prev_y, 2.2, line_color)
