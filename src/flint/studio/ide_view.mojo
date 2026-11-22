"""Native workspace toolbar and text canvas shared by Studio and TurboMojo."""
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
    ui.add_rect(0, 0, Float32(width), Float32(height), ui_color(DEFAULT_BG))
    ui.add_rect(0, 0, Float32(width), HEADER, ui_color(0x17212C))
    var origin = toolbar_start(scene)
    if scene:
        ui.add_rect(0, 0, 166, HEADER, ui_color(UInt32(0x324452 if hover == BACK else 0x253541)))
        ui.add_text(slot, font, "< Scene   Ctrl+F8", 10, 24, 14, ui_color(0xD1DFEA))
    var brand = Float32(origin - 124)
    ui.add_rect(brand, 12, 3, 16, ui_color(0xF79D5E))
    ui.add_text(slot, font, "TurboMojo", brand + 11, 25, 17, ui_color(0xF5F5F2))
    for i in range(4):
        var x = origin + i * 68
        if x + 64 > width:
            break
        var bg = 0xAADDC6 if i == 3 else (0x3F5566 if hover == i else 0x293845)
        ui.add_rect(Float32(x), 6, 64, 28, ui_color(UInt32(bg)))
        ui.add_text(
            slot,
            font,
            toolbar_label(i),
            Float32(x + 10),
            25,
            15,
            ui_color(UInt32(0x162721 if i == 3 else 0xE4ECF2)),
        )
    if not scene:
        if moves and width >= TAB_START + 72:
            ui.add_rect(TAB_START, 6, 72, 28, ui_color(UInt32(0x3F5566 if hover == MOVES else 0x293845)))
            ui.add_text(slot, font, "Moves", TAB_START + 12, 25, 15, ui_color(0xE4ECF2))
        for i in range(len(tabs)):
            var x = tab_origin(moves) + TAB_WIDTH * i
            if x + TAB_WIDTH > width:
                break
            if i == active:
                ui.add_rect(Float32(x), 3, TAB_WIDTH - 2, HEADER - 3, ui_color(0x263641))
                ui.add_rect(Float32(x), HEADER - 2, TAB_WIDTH - 2, 2, ui_color(0xF79D5E))
            ui.push_clip(x + 8, 0, x + TAB_WIDTH - 8, HEADER)
            ui.add_text(slot, font, tabs[i], Float32(x + 9), 25, 14, ui_color(0xD1DFEA))
            ui.pop_clip()
    if status and width > origin + 280:
        ui.add_text(slot, font, status, Float32(origin + 280), 24, 13, ui_color(0xBECBD5))
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
                ui.add_rect(Float32(start * CW), top, Float32((x - start) * CW), CH, ui_color(bg))
        for col in range(grid.cols):
            var cell = grid.cells[y * grid.cols + col]
            var i = y * grid.cols + col
            if cell.code > 32 or (i < len(marks) and marks[i]):
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
    if grid.cursor and grid.cols > 0 and grid.rows > 0:
        var cx = min(grid.cols - 1, max(0, grid.x))
        var cy = min(grid.rows - 1, max(0, grid.y))
        ui.add_rect(
            Float32(cx * CW),
            Float32(HEADER + cy * CH + 16),
            CW,
            2,
            0xFFFFFFFF,
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
        var rgb = ui_color(p.fill_rgba8)
        var glyph = (p.header & 15) == PRIM_GLYPH
        if glyph and (p.x1 <= p.x0 or p.y1 <= p.y0):
            continue
        var left = max(Int(p.clip_xy0 & 65535), Int(p.x0))
        var right = min(Int(p.clip_xy1 & 65535), min(width, Int(p.x1 + 0.5)))
        for y in range(
            max(Int(p.clip_xy0 >> 16), Int(p.y0)),
            min(Int(p.clip_xy1 >> 16), min(height, Int(p.y1 + 0.5))),
        ):
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
