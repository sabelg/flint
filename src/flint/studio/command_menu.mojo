"""A mouse and keyboard menu bar shared by Studio's scene and animation workspaces."""
from std.collections import List
from flint.platform.input import InputState
from flint.render.font import SdfFont
from flint.studio.ide_view import ui_color
from flint.text.utf8 import decode_utf8
from flint.ui.draw_list import UI_FLAG_BORDER, UI_FLAG_TOP_BEVEL, UiDrawList
from flint.ui.icons import draw_studio_icon
from flint.ui.theme import (
    ICON_CHECK,
    StudioTheme,
    lerp_rgba8,
    rgba8_scale_alpha,
    rgba8_with_alpha,
)


def _menu_measure_text(font: SdfFont, text: String, size_px: Float32) -> Float32:
    if font.valid:
        return font.measure(text, size_px)
    return Float32(len(decode_utf8(text))) * size_px * Float32(0.56)


def _menu_draw_text(
    mut ui: UiDrawList,
    slot: Int,
    font: SdfFont,
    text: String,
    x: Float32,
    center_y: Float32,
    size_px: Float32,
    color: UInt32,
):
    if text.byte_length() == 0:
        return
    if font.valid:
        var cap_ratio = (
            font.cap_height / font.bake_px
            if (font.bake_px > Float32(0.0) and font.cap_height > Float32(0.0))
            else Float32(0.72)
        )
        var baseline = center_y + Float32(0.5) * cap_ratio * size_px
        ui.add_text(slot, font, text, x, baseline, size_px, color)
    else:
        var n_cp = len(decode_utf8(text))
        if n_cp > 0:
            ui.add_rect(
                x,
                center_y - size_px * Float32(0.30),
                Float32(n_cp) * size_px * Float32(0.54),
                size_px * Float32(0.60),
                rgba8_scale_alpha(color, Float32(0.45)),
                Float32(2.0),
            )


def _split_shortcut_tokens(shortcut: String) -> List[String]:
    """Splits `"Ctrl+Shift+Z"` into `["Ctrl", "Shift", "Z"]` while preserving lone `"+"`."""
    var out = List[String]()
    var n = shortcut.byte_length()
    if n == 0:
        return out^
    var bytes = shortcut.as_bytes()
    var start = 0
    for i in range(n):
        if bytes[i] == 43 and i > start and i + 1 < n:  # '+'
            var tok = String(String(shortcut[byte=start:i]).strip())
            if tok.byte_length() > 0:
                out.append(tok)
            start = i + 1
    var tail = String(String(shortcut[byte=start:n]).strip())
    if tail.byte_length() > 0:
        out.append(tail)
    return out^


def _draw_menu_keycaps_rtl(
    mut ui: UiDrawList,
    slot: Int,
    font: SdfFont,
    theme: StudioTheme,
    shortcut: String,
    right_x: Float32,
    center_y: Float32,
    enabled: Bool,
) -> Float32:
    """Draws tactile split keycap chips right-to-left from `right_x` and returns the new left edge."""
    var tokens = _split_shortcut_tokens(shortcut)
    if len(tokens) == 0:
        return right_x
    var sz = Float32(10.5)
    var cap_h = Float32(18.0)
    var cap_y = center_y - cap_h * Float32(0.5)
    var pen = right_x
    var fg = theme.text_secondary if enabled else theme.text_muted
    var fill = theme.l2_card if enabled else rgba8_scale_alpha(theme.l2_card, Float32(0.65))
    var bdr = theme.border_control if enabled else theme.border_subtle
    var k = len(tokens) - 1
    while k >= 0:
        var tok = tokens[k]
        var tw = _menu_measure_text(font, tok, sz)
        var cw = max(Float32(18.0), tw + Float32(10.0))
        pen -= cw
        # Subtle 1px bottom keycap lip for physical depth
        ui.add_rect(pen, cap_y + Float32(1.0), cw, cap_h, rgba8_with_alpha(0x000000, 90), theme.radius_xs)
        ui.add_rect_styled(
            pen,
            cap_y,
            cw,
            cap_h,
            fill,
            bdr,
            theme.radius_xs,
            Float32(1.0),
            Float32(1.0),
            UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
        )
        _menu_draw_text(ui, slot, font, tok, pen + (cw - tw) * Float32(0.5), center_y, sz, fg)
        pen -= Float32(4.0)
        k -= 1
    return pen


struct MenuItem(Copyable, Movable):
    var command: Int
    var title: String
    var shortcut: String
    var enabled: Bool
    var separator_before: Bool
    var checked: Int  # -1 = none, 0 = unchecked toggle, 1 = checked toggle

    def __init__(
        out self,
        command: Int,
        title: String,
        shortcut: String = "",
        enabled: Bool = True,
        separator_before: Bool = False,
        checked: Int = -1,
    ):
        self.command = command
        self.title = title
        self.shortcut = shortcut
        self.enabled = enabled
        self.separator_before = separator_before
        self.checked = checked


@fieldwise_init
struct MenuGroup(Copyable, Movable):
    var title: String
    var width: Int
    var items: List[MenuItem]


struct CommandMenu(Movable):
    var software_colors: Bool
    var groups: List[MenuGroup]
    var group: Int
    var row: Int
    var first: Int
    var x: Int
    var y: Int
    var height: Int
    var screen_w: Int
    var screen_h: Int
    var last_x: Int
    var last_y: Int
    var swallow_release: Bool
    var dirty: Bool

    def __init__(out self, software_colors: Bool = False):
        self.software_colors = software_colors
        self.groups = []
        self.group = -1
        self.row = 0
        self.first = 0
        self.x = 0
        self.y = 0
        self.height = 30
        self.screen_w = 1024
        self.screen_h = 640
        self.last_x = -1
        self.last_y = -1
        self.swallow_release = False
        self.dirty = True

    def add_group(mut self, title: String, width: Int):
        self.groups.append(MenuGroup(title, width, List[MenuItem]()))

    def item(
        mut self,
        command: Int,
        title: String,
        shortcut: String = "",
        enabled: Bool = True,
        separator_before: Bool = False,
        checked: Int = -1,
    ):
        if not self.groups:
            return
        self.groups[len(self.groups) - 1].items.append(
            MenuItem(command, title, shortcut, enabled, separator_before, checked)
        )

    def set_item_checked(mut self, command: Int, checked: Bool):
        for gi in range(len(self.groups)):
            for ii in range(len(self.groups[gi].items)):
                if self.groups[gi].items[ii].command == command:
                    self.groups[gi].items[ii].checked = 1 if checked else 0
                    return

    def left(self, group: Int) -> Int:
        var x = self.x
        for i in range(min(max(0, group), len(self.groups))):
            x += self.groups[i].width
        return x

    def rows(self) -> Int:
        if self.group < 0 or self.group >= len(self.groups):
            return 0
        return min(
            len(self.groups[self.group].items),
            max(1, min(10, max(self.y, self.screen_h - self.y - self.height) // 32)),
        )

    def popup_x(self) -> Int:
        if self.group < 0 or self.group >= len(self.groups):
            return 0
        return max(0, min(self.screen_w - 280, self.left(self.group)))

    def popup_y(self) -> Int:
        var below = self.y + self.height
        var size = self.rows() * 32
        return below if below + size <= self.screen_h else max(0, self.y - size)

    def open(mut self, group: Int):
        if group < 0 or group >= len(self.groups):
            return
        self.group = group
        self.row = 0
        self.first = 0
        for i in range(len(self.groups[group].items)):
            if self.groups[group].items[i].enabled:
                self.row = i
                break
        self.dirty = True

    def input(mut self, inp: InputState, x: Int, y: Int) -> Int:
        """Return -1 to pass input through, zero when consumed, or the chosen command."""
        var moved = self.last_x != x or self.last_y != y
        self.last_x = x
        self.last_y = y
        var click = inp.mouse_pressed(0)
        if self.swallow_release and not click:
            if not inp.mouse_down(0):
                self.swallow_release = False
            return 0
        if not self.groups:
            return -1
        var heading = -1
        if self.y <= y < self.y + self.height:
            for i in range(len(self.groups)):
                if self.left(i) <= x < self.left(i + 1):
                    heading = i
                    break
        if inp.key_pressed(299) or (heading >= 0 and click):
            var next = (0 if self.group < 0 else self.group) if heading < 0 or not click else heading
            if self.group == next:
                self.group = -1
            else:
                self.open(next)
            self.swallow_release = click
            self.dirty = True
            return 0
        if self.group < 0 or self.group >= len(self.groups):
            self.group = -1
            return -1
        if inp.key_pressed(256):
            self.group = -1
            self.dirty = True
            return 0
        if moved and heading >= 0 and self.group != heading:
            self.open(heading)
        if inp.key_pressed(262) or inp.key_pressed(263):
            self.open(
                (self.group + len(self.groups) + (1 if inp.key_pressed(262) else -1)) % len(self.groups)
            )
        ref items = self.groups[self.group].items
        var n_rows = self.rows()
        if len(items) == 0:
            self.first = 0
            self.row = 0
        else:
            self.first = max(0, min(self.first, max(0, len(items) - n_rows)))
            self.row = max(0, min(self.row, len(items) - 1))
        var px = self.popup_x()
        var py = self.popup_y()
        var on_row = px <= x < px + 280 and py <= y < py + n_rows * 32
        if on_row and (moved or click) and len(items) > 0:
            self.row = min(len(items) - 1, self.first + (y - py) // 32)
            self.dirty = True
        if inp.wheel and on_row and len(items) > 0:
            self.first = max(
                0, min(len(items) - n_rows, self.first + (-1 if inp.wheel > 0 else 1))
            )
            self.row = min(len(items) - 1, self.first + (y - py) // 32)
            self.dirty = True
        if (inp.key_pressed(264) or inp.key_pressed(265)) and len(items) > 0:
            var dir = 1 if inp.key_pressed(264) else -1
            for _ in range(len(items)):
                self.row = (self.row + len(items) + dir) % len(items)
                if items[self.row].enabled:
                    break
            self.first = max(0, min(self.row, max(self.first, self.row - n_rows + 1)))
            self.dirty = True
        if click or inp.key_pressed(257):
            self.swallow_release = click
            var command = 0
            if (on_row or not click) and items:
                if not items[self.row].enabled:
                    return 0
                command = items[self.row].command
            self.group = -1
            self.dirty = True
            return command
        return 0

    def color(self, rgb: UInt32) -> UInt32:
        return ui_color(rgb) if self.software_colors else 0xFF000000 | rgb

    def draw(self, mut ui: UiDrawList, font: SdfFont, slot: Int):
        """Draws the menu bar and active dropdown using Obsidian theme (or software fallback)."""
        if not self.software_colors:
            self.draw_themed(ui, font, slot, StudioTheme())
            return
        for i in range(len(self.groups)):
            var left = self.left(i)
            var top = self.y
            var w = self.groups[i].width
            ui.add_rect(
                Float32(left),
                Float32(top),
                Float32(w - 2),
                Float32(self.height),
                self.color(UInt32(0x36515F if self.group == i else 0x273742)),
            )
            ui.push_clip(left, top, left + w - 3, top + self.height)
            ui.add_text(
                slot,
                font,
                self.groups[i].title,
                Float32(left + 9),
                Float32(top + self.height // 2 + 5),
                15,
                self.color(0xF1F5F4),
            )
            ui.pop_clip()
        if self.group < 0 or self.group >= len(self.groups):
            return
        var n = self.rows()
        if n <= 0:
            return
        var px = self.popup_x()
        var py = self.popup_y()
        var height = n * 32
        ui.add_rect(
            Float32(px),
            Float32(py),
            280,
            Float32(height),
            self.color(0x24333F),
            3,
            1,
            self.color(0x59717A),
        )
        ref items = self.groups[self.group].items
        var start = max(0, min(self.first, max(0, len(items) - n)))
        var end = min(len(items), start + n)
        for i in range(start, end):
            ref item = items[i]
            var y = py + 32 * (i - start)
            if i == self.row:
                ui.add_rect(Float32(px + 2), Float32(y + 1), 276, 30, self.color(0x3C5460))
            var right = px + 266 - item.shortcut.byte_length() * 9
            ui.push_clip(px + 4, y, right - 4, y + 32)
            ui.add_text(
                slot,
                font,
                item.title,
                Float32(px + 12),
                Float32(y + 21),
                15,
                self.color(UInt32(0xEBF2F5 if item.enabled else 0x8A9BA4)),
            )
            ui.pop_clip()
            ui.add_text(
                slot, font, item.shortcut, Float32(right), Float32(y + 21), 13, self.color(0xB4CCD0)
            )
        if len(items) > n:
            var track = height * n // len(items)
            ui.add_rect(
                Float32(px + 275),
                Float32(py + (height - track) * start // (len(items) - n)),
                3,
                Float32(track),
                self.color(0x91B9B7),
            )

    def draw_themed(
        self,
        mut ui: UiDrawList,
        font: SdfFont,
        slot: Int,
        theme: StudioTheme,
        mono_slot: Int = -1,
    ):
        """Renders the application menu strip and dropdown popover with `StudioTheme` frosted glass."""
        if len(self.groups) == 0:
            return
        var key_slot = mono_slot if mono_slot >= 0 else slot
        var total_w = self.left(len(self.groups)) - self.x
        var bx = Float32(self.x)
        var by = Float32(self.y)
        var bw = Float32(max(1, total_w))
        var bh = Float32(max(1, self.height))

        # Unified frosted-glass top menu strip anchored cleanly below STUDIO_HEADER_BAR_H
        ui.add_ambient_shadow(bx, by, bw, bh, theme.radius_sm, Float32(10.0), 115, Float32(2.0))
        ui.add_glass_panel(
            bx,
            by,
            bw,
            bh,
            theme.l4_overlay,
            theme.radius_sm,
            Float32(3.0),
            Float32(1.0),
            theme.border_strong,
            True,
            True,
        )

        var strip_cy = by + bh * Float32(0.5)
        var hdr_sz = Float32(12.5)
        for i in range(len(self.groups)):
            var left = self.left(i)
            var w = self.groups[i].width
            var is_act = self.group == i
            var is_hov = (
                self.last_x >= left
                and self.last_x < left + w
                and self.last_y >= self.y
                and self.last_y < self.y + self.height
            )
            var gx = Float32(left + 2)
            var gy = by + Float32(3.0)
            var gw = Float32(max(4, w - 4))
            var gh = max(Float32(4.0), bh - Float32(6.0))
            if is_act:
                ui.add_rect_styled(
                    gx,
                    gy,
                    gw,
                    gh,
                    lerp_rgba8(theme.l3_raised, theme.accent_primary, Float32(0.16)),
                    rgba8_with_alpha(theme.accent_primary, 110),
                    theme.radius_xs,
                    Float32(1.0),
                    Float32(1.0),
                    UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
                )
                ui.add_accent_bar(
                    gx + Float32(6.0),
                    gy + gh - Float32(2.0),
                    max(Float32(4.0), gw - Float32(12.0)),
                    Float32(2.0),
                    theme.accent_primary,
                    True,
                )
            elif is_hov:
                ui.add_rect(gx, gy, gw, gh, theme.l3_hover, theme.radius_xs)

            var tw = _menu_measure_text(font, self.groups[i].title, hdr_sz)
            var tx = Float32(left) + max(Float32(6.0), (Float32(w) - tw) * Float32(0.5))
            ui.push_clip(left + 2, self.y, left + w - 2, self.y + self.height)
            _menu_draw_text(
                ui,
                slot,
                font,
                self.groups[i].title,
                tx,
                strip_cy,
                hdr_sz,
                theme.text_primary if (is_act or is_hov) else theme.text_secondary,
            )
            ui.pop_clip()

        if self.group < 0 or self.group >= len(self.groups):
            return
        var n = self.rows()
        if n <= 0:
            return
        var px = self.popup_x()
        var py = self.popup_y()
        var pop_w = Float32(280.0)
        var pop_h = Float32(n * 32)

        # Ambient shadow + frosted glass popover
        ui.add_ambient_shadow(
            Float32(px), Float32(py), pop_w, pop_h, theme.radius_md, Float32(14.0), 150, Float32(4.0)
        )
        ui.add_glass_panel(
            Float32(px),
            Float32(py),
            pop_w,
            pop_h,
            theme.l4_overlay,
            theme.radius_md,
            Float32(4.0),
            Float32(1.0),
            theme.border_strong,
            True,
            True,
        )

        ref items = self.groups[self.group].items
        var start = max(0, min(self.first, max(0, len(items) - n)))
        var end = min(len(items), start + n)
        var has_toggles = False
        for ti in range(len(items)):
            if items[ti].checked >= 0:
                has_toggles = True
                break

        var item_sz = Float32(12.5)
        for i in range(start, end):
            ref item = items[i]
            var y = py + 32 * (i - start)
            var ry = Float32(y)
            var row_cy = ry + Float32(16.0)

            if item.separator_before and i > start:
                ui.add_line(
                    Float32(px) + Float32(10.0),
                    ry + Float32(0.5),
                    Float32(px) + pop_w - Float32(10.0),
                    ry + Float32(0.5),
                    Float32(1.0),
                    theme.border_subtle,
                )

            var is_sel = i == self.row
            if is_sel:
                var sel_x = Float32(px) + Float32(5.0)
                var sel_y = ry + Float32(2.0)
                var sel_w = pop_w - Float32(10.0)
                var sel_h = Float32(28.0)
                if item.enabled:
                    ui.add_rect_styled(
                        sel_x,
                        sel_y,
                        sel_w,
                        sel_h,
                        lerp_rgba8(theme.l3_raised, theme.accent_primary, Float32(0.14)),
                        rgba8_with_alpha(theme.accent_primary, 92),
                        theme.radius_xs,
                        Float32(1.0),
                        Float32(1.0),
                        UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
                    )
                    ui.add_accent_bar(
                        sel_x + Float32(2.0),
                        sel_y + Float32(5.0),
                        Float32(2.5),
                        sel_h - Float32(10.0),
                        theme.accent_primary,
                        True,
                    )
                else:
                    ui.add_rect(sel_x, sel_y, sel_w, sel_h, rgba8_scale_alpha(theme.l2_card, Float32(0.75)), theme.radius_xs)

            if has_toggles:
                if item.checked == 1:
                    draw_studio_icon(
                        ui,
                        ICON_CHECK,
                        Float32(px) + Float32(11.0),
                        row_cy - Float32(6.0),
                        Float32(12.0),
                        theme.accent_primary if item.enabled else theme.text_muted,
                        Float32(1.6),
                    )
                elif item.checked == 0:
                    ui.add_rect(
                        Float32(px) + Float32(13.0),
                        row_cy - Float32(4.0),
                        Float32(8.0),
                        Float32(8.0),
                        UInt32(0),
                        Float32(2.0),
                        Float32(1.0),
                        theme.border_control,
                    )

            var right_edge = Float32(px) + pop_w - Float32(12.0)
            if len(items) > n:
                right_edge -= Float32(6.0)
            if item.shortcut.byte_length() > 0:
                right_edge = _draw_menu_keycaps_rtl(
                    ui, key_slot, font, theme, item.shortcut, right_edge, row_cy, item.enabled
                ) - Float32(6.0)

            var title_x = Float32(px) + (Float32(29.0) if has_toggles else Float32(14.0))
            var fg = (
                theme.text_primary
                if (item.enabled and is_sel)
                else (theme.text_secondary if item.enabled else theme.text_muted)
            )
            ui.push_clip(Int(title_x), y, Int(max(title_x, right_edge)), y + 32)
            _menu_draw_text(ui, slot, font, item.title, title_x, row_cy, item_sz, fg)
            ui.pop_clip()

        if len(items) > n:
            var track = max(18, Int(pop_h) * n // len(items))
            var travel = max(1, Int(pop_h) - 10 - track)
            var thumb_y = Float32(py + 5) + Float32(travel * start // max(1, len(items) - n))
            ui.add_rect(
                Float32(px) + pop_w - Float32(6.0),
                thumb_y,
                Float32(3.0),
                Float32(track),
                rgba8_with_alpha(theme.text_muted, 115),
                Float32(1.5),
            )
