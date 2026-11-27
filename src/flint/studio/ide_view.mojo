"""Native workspace toolbar and text canvas shared by Studio and TurboMojo."""
from std.math import isfinite
from flint.render.font import SdfFont
from flint.ui.draw_list import UiDrawList, PRIM_GLYPH
from flint.studio.ide_grid import IdeGrid, DEFAULT_BG

comptime HEADER = 40
comptime CW = 9
comptime CH = 18
comptime BACK = 4
comptime TAB_START = 426
comptime TAB_WIDTH = 146
comptime MOVES = 5


def tab_origin(moves: Bool) -> Int:
    return TAB_START + (84 if moves else 0)


def toolbar_start(scene: Bool) -> Int:
    return 304 if scene else 140


def toolbar_label(i: Int) -> String:
    var labels = ["Open", "Save", "Build", "Run"]
    if i < 0 or i >= len(labels):
        return ""
    return labels[i]


def toolbar_hit(x: Int, y: Int, width: Int, scene: Bool, moves: Bool = False) -> Int:
    if width <= 0 or x < 0 or x >= width or y < 0 or y >= HEADER:
        return -1
    if scene and width >= 166 and 0 <= x < 166 and 0 <= y < HEADER:
        return BACK
    if not scene and moves and width >= TAB_START + 72 and TAB_START <= x < TAB_START + 72 and 6 <= y < 34:
        return MOVES
    var offset = x - toolbar_start(scene)
    if y < 6 or y >= 34 or offset < 0 or offset >= 272:
        return -1
    var i = offset // 68
    if offset % 68 >= 64 or toolbar_start(scene) + i * 68 + 64 > width:
        return -1
    return i


def ui_color(rgb: UInt32) -> UInt32:
    return 0xFF000000 | ((rgb & 255) << 16) | (rgb & 0xFF00) | ((rgb >> 16) & 255)


def _measure_ide(font: SdfFont, text: String, px: Float32) -> Float32:
    if font.valid:
        return font.measure(text, px)
    return Float32(text.byte_length()) * px * Float32(0.56)


def draw_workspace(
    mut ui: UiDrawList,
    grid: IdeGrid,
    width: Int,
    height: Int,
    font: SdfFont,
    scene: Bool,
    hover: Int,
    status: String,
    tabs: List[String],
    active: Int,
    marks: List[String],
    moves: Bool,
):
    if width <= 0 or height <= 0:
        return
    var slot = 0
    # Flint Obsidian neutral graphite palette
    comptime OBS_L0: UInt32 = 0x0D1014
    comptime OBS_L1: UInt32 = 0x161A20
    comptime OBS_L2: UInt32 = 0x1E232B
    comptime OBS_L3: UInt32 = 0x282F3A
    comptime OBS_HOVER: UInt32 = 0x333C4A
    comptime OBS_BORDER: UInt32 = 0x2C3440
    comptime OBS_EMBER: UInt32 = 0xF59B4C
    comptime OBS_PLAY: UInt32 = 0x2DD4BF
    comptime OBS_PLAY_HOV: UInt32 = 0x45E0CC
    comptime OBS_INK_DARK: UInt32 = 0x0B0E14
    comptime OBS_TXT_PRI: UInt32 = 0xEEF2F7
    comptime OBS_TXT_SEC: UInt32 = 0xA9B4C4
    comptime OBS_TXT_MUT: UInt32 = 0x8491A3

    ui.add_rect(0, 0, Float32(width), Float32(height), ui_color(DEFAULT_BG))
    ui.add_rect(0, 0, Float32(width), Float32(HEADER), ui_color(OBS_L1))
    ui.add_rect(0, Float32(HEADER - 1), Float32(width), Float32(1), ui_color(OBS_BORDER))
    var origin = toolbar_start(scene)
    if scene and width >= 166:
        ui.add_rect(0, 0, 166, Float32(HEADER - 1), ui_color(OBS_HOVER if hover == BACK else OBS_L2))
        ui.add_rect(165, 0, 1, Float32(HEADER - 1), ui_color(OBS_BORDER))
        var back_lbl = "< Scene   Ctrl+F8"
        var back_w = _measure_ide(font, back_lbl, Float32(13.0))
        var back_x = max(Float32(8.0), (Float32(166.0) - back_w) * Float32(0.5))
        ui.add_text(slot, font, back_lbl, back_x, 24, 13, ui_color(OBS_TXT_PRI))
    var brand = max(Float32(8.0), Float32(origin - 124))
    ui.add_rect(brand, 12, 3, 16, ui_color(OBS_EMBER))
    ui.add_text(slot, font, "TurboMojo", brand + 11, 25, 16, ui_color(OBS_TXT_PRI))
    for i in range(4):
        var x = origin + i * 68
        if x + 64 > width:
            break
        var bg: UInt32
        var fg: UInt32
        if i == 3:
            bg = OBS_PLAY_HOV if hover == i else OBS_PLAY
            fg = OBS_INK_DARK
        else:
            bg = OBS_HOVER if hover == i else OBS_L3
            fg = OBS_TXT_PRI
        ui.add_rect(Float32(x), 6, 64, 28, ui_color(OBS_BORDER))
        ui.add_rect(Float32(x + 1), 7, 62, 26, ui_color(bg))
        var lbl = toolbar_label(i)
        var lw = _measure_ide(font, lbl, Float32(14.0))
        var lx = Float32(x) + max(Float32(4.0), (Float32(64.0) - lw) * Float32(0.5))
        ui.add_text(slot, font, lbl, lx, 25, 14, ui_color(fg))
    if not scene:
        if moves and width >= TAB_START + 72:
            ui.add_rect(TAB_START, 6, 72, 28, ui_color(OBS_BORDER))
            ui.add_rect(TAB_START + 1, 7, 70, 26, ui_color(OBS_HOVER if hover == MOVES else OBS_L3))
            var mw = _measure_ide(font, "Moves", Float32(14.0))
            var mx_t = Float32(TAB_START) + max(Float32(4.0), (Float32(72.0) - mw) * Float32(0.5))
            ui.add_text(slot, font, "Moves", mx_t, 25, 14, ui_color(OBS_TXT_PRI))
        for i in range(len(tabs)):
            var x = tab_origin(moves) + TAB_WIDTH * i
            if x + TAB_WIDTH > width:
                break
            if i == active:
                ui.add_rect(Float32(x), 4, TAB_WIDTH - 2, HEADER - 5, ui_color(OBS_L2))
                ui.add_rect(Float32(x), Float32(HEADER - 2), TAB_WIDTH - 2, 2, ui_color(OBS_EMBER))
            ui.push_clip(x + 8, 0, x + TAB_WIDTH - 8, HEADER)
            ui.add_text(
                slot,
                font,
                tabs[i],
                Float32(x + 10),
                25,
                13,
                ui_color(OBS_TXT_PRI if i == active else OBS_TXT_SEC),
            )
            ui.pop_clip()
    if status and width > origin + 280:
        ui.add_text(slot, font, status, Float32(origin + 280), 24, 12, ui_color(OBS_TXT_MUT))
    var has_content = False
    for y in range(grid.rows):
        var x = 0
        var top = Float32(HEADER + y * CH)
        while x < grid.cols:
            var bg = grid.cells[y * grid.cols + x].bg
            var start = x
            x += 1
            while x < grid.cols and grid.cells[y * grid.cols + x].bg == bg:
                x += 1
            if bg != DEFAULT_BG:
                has_content = True
                ui.add_rect(Float32(start * CW), top, Float32((x - start) * CW), CH, ui_color(bg))
        for col in range(grid.cols):
            var cell = grid.cells[y * grid.cols + col]
            var i = y * grid.cols + col
            if cell.code > 32 or (i < len(marks) and marks[i]):
                has_content = True
                var cluster = String(chr(cell.code))
                if i < len(marks):
                    cluster += marks[i]
                ui.add_text(
                    slot,
                    font,
                    cluster,
                    Float32(col * CW),
                    top + 14,
                    15,
                    ui_color(cell.fg),
                )
            if cell.underline:
                ui.add_rect(Float32(col * CW), top + 16, CW, 1, ui_color(cell.fg))
    if (not has_content) and status.byte_length() > 0 and width >= 340 and height >= HEADER + 110:
        var card_w = min(Float32(460.0), Float32(width - 48))
        var card_h = Float32(82.0)
        var card_x = max(Float32(0.0), (Float32(width) - card_w) * Float32(0.5))
        var card_y = max(Float32(HEADER + 12), Float32(HEADER) + (Float32(height - HEADER) - card_h) * Float32(0.5))
        ui.add_rect(card_x, card_y, card_w, card_h, ui_color(OBS_BORDER))
        ui.add_rect(card_x + 1, card_y + 1, card_w - 2, card_h - 2, ui_color(OBS_L2))
        ui.add_rect(card_x + 1, card_y + 1, 3, card_h - 2, ui_color(OBS_EMBER))
        ui.add_text(slot, font, "TurboMojo Terminal IDE", card_x + 18, card_y + 32, 15, ui_color(OBS_TXT_PRI))
        ui.add_text(slot, font, status, card_x + 18, card_y + 56, 13, ui_color(OBS_TXT_SEC))
    elif grid.cursor and grid.cols > 0 and grid.rows > 0:
        var cx = min(grid.cols - 1, max(0, grid.x))
        var cy = min(grid.rows - 1, max(0, grid.y))
        ui.add_rect(
            Float32(cx * CW),
            Float32(HEADER + cy * CH + 16),
            CW,
            2,
            ui_color(OBS_EMBER),
        )


def raster_workspace(
    mut pixels: List[UInt32],
    mut font: SdfFont,
    grid: IdeGrid,
    width: Int,
    height: Int,
    scene: Bool,
    hover: Int,
    status: String,
    tabs: List[String] = [],
    active: Int = -1,
    marks: List[String] = [],
    moves: Bool = False,
):
    if width <= 0 or height <= 0:
        pixels.clear()
        return
    if not font.valid:
        _ = font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 32, 5)
    for c in grid.cells:
        if c.code > 126:
            _ = font.ensure(c.code)
    for text in marks:
        if text:
            font.ensure_text(text)
    for tab in tabs:
        font.ensure_text(tab)
    var ui = UiDrawList()
    draw_workspace(ui, grid, width, height, font, scene, hover, status, tabs, active, marks, moves)
    raster_ui(pixels, font, ui, width, height)


def raster_ui(mut pixels: List[UInt32], font: SdfFont, ui: UiDrawList, width: Int, height: Int):
    if width <= 0 or height <= 0:
        pixels.clear()
        return
    pixels.resize(width * height, 0)
    for p in ui.prims:
        if not isfinite(p.x0) or not isfinite(p.y0) or not isfinite(p.x1) or not isfinite(p.y1):
            continue
        var rgb = ui_color(p.fill_rgba8)
        var glyph = (p.header & 15) == PRIM_GLYPH
        if glyph and (
            not isfinite(p.param0)
            or not isfinite(p.u0)
            or not isfinite(p.v0)
            or not isfinite(p.u1)
            or not isfinite(p.v1)
            or p.x1 <= p.x0
            or p.y1 <= p.y0
        ):
            continue
        var x0_i = Int(max(Float64(0.0), min(Float64(width), Float64(p.x0))))
        var y0_i = Int(max(Float64(0.0), min(Float64(height), Float64(p.y0))))
        var x1_i = Int(max(Float64(0.0), min(Float64(width), Float64(p.x1) + 0.5)))
        var y1_i = Int(max(Float64(0.0), min(Float64(height), Float64(p.y1) + 0.5)))
        var left = max(Int(p.clip_xy0 & 65535), x0_i)
        var right = min(Int(p.clip_xy1 & 65535), x1_i)
        var top = max(Int(p.clip_xy0 >> 16), y0_i)
        var bottom = min(Int(p.clip_xy1 >> 16), y1_i)
        if left >= right or top >= bottom:
            continue
        for y in range(top, bottom):
            for x in range(left, right):
                var index = y * width + x
                if not glyph:
                    pixels[index] = rgb
                    continue
                var u = (Float32(x) + 0.5 - p.x0) / (p.x1 - p.x0)
                var v = (Float32(y) + 0.5 - p.y0) / (p.y1 - p.y0)
                if u < 0 or v < 0 or u >= 1 or v >= 1:
                    continue
                var sdf = font.sample(
                    (p.u0 + u * (p.u1 - p.u0)) * Float32(font.atlas_width),
                    (p.v0 + v * (p.v1 - p.v0)) * Float32(font.atlas_height),
                )
                var a = UInt32(255 * min(Float32(1), max(Float32(0), 0.5 + (sdf - 128) * p.param0)))
                var old = pixels[index]
                var color: UInt32 = 0xFF000000
                comptime for shift in [0, 8, 16]:
                    color |= (
                        (
                            ((old >> UInt32(shift)) & 255) * (255 - a)
                            + ((rgb >> UInt32(shift)) & 255) * a
                        )
                        // 255
                    ) << UInt32(shift)
                pixels[index] = color
