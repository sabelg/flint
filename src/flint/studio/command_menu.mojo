"""A mouse and keyboard menu bar shared by Studio's scene and animation workspaces."""
from flint.platform.input import InputState
from flint.render.font import SdfFont
from flint.studio.ide_view import ui_color
from flint.ui.draw_list import UiDrawList


@fieldwise_init
struct MenuItem(Copyable, Movable):
    var command: Int
    var title: String
    var shortcut: String
    var enabled: Bool


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

    def item(mut self, command: Int, title: String, shortcut: String = "", enabled: Bool = True):
        if not self.groups:
            return
        self.groups[len(self.groups) - 1].items.append(MenuItem(command, title, shortcut, enabled))

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
