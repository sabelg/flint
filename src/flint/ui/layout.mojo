"""Anchors and safe areas for the UI canvas (package 3.12).

The UI is laid out on a canvas 720 units tall; `ui_scale` maps canvas
units to pixels (1.0 at 1280x720 and 1280x800, 2.0 at 3440x1440). Menus keep
the centred 1280-wide design area, but the visible canvas is as wide (21:9)
or as tall (16:10) as the screen, so HUD elements are placed by anchor:
an element designed at a 16:9 position keeps its distance from the edge or
centre it is anchored to, and stays inside the safe area.

The safe area is the screen minus a margin (`safe_margin`, a fraction of
each side; 0 by default, up to 10% for televisions that crop the edge). A
`UiFrame` can also be built for part of the screen, the viewport of one
player, so a second player's HUD (decision 3, package 3.16) is the same
code with another frame.
"""

comptime DESIGN_WIDTH: Int = 1280
comptime DESIGN_HEIGHT: Int = 720

comptime ANCHOR_TOP_LEFT: Int = 0
comptime ANCHOR_TOP: Int = 1
comptime ANCHOR_TOP_RIGHT: Int = 2
comptime ANCHOR_LEFT: Int = 3
comptime ANCHOR_CENTER: Int = 4
comptime ANCHOR_RIGHT: Int = 5
comptime ANCHOR_BOTTOM_LEFT: Int = 6
comptime ANCHOR_BOTTOM: Int = 7
comptime ANCHOR_BOTTOM_RIGHT: Int = 8

comptime MAX_SAFE_MARGIN: Float32 = 0.10


@fieldwise_init
struct UiRect(Copyable, ImplicitlyCopyable, Movable):
    var x: Int
    var y: Int
    var w: Int
    var h: Int

    def right(self) -> Int:
        return self.x + self.w

    def bottom(self) -> Int:
        return self.y + self.h

    def contains(self, other: UiRect) -> Bool:
        return other.x >= self.x and other.y >= self.y and other.right() <= self.right() and other.bottom() <= self.bottom()

    def overlaps(self, other: UiRect) -> Bool:
        return self.x < other.right() and other.x < self.right() and self.y < other.bottom() and other.y < self.bottom()


struct UiFrame(Copyable, ImplicitlyCopyable, Movable):
    """The visible canvas and its safe area, in canvas units (the 16:9
    design area runs from 0 to 1280 and 0 to 720 inside it)."""
    var left: Int
    var top: Int
    var right: Int
    var bottom: Int
    var safe_left: Int
    var safe_top: Int
    var safe_right: Int
    var safe_bottom: Int
    var scale: Float32       # pixels per canvas unit
    var player: Int          # whose HUD this frame holds

    def __init__(out self):
        self.left = 0
        self.top = 0
        self.right = DESIGN_WIDTH
        self.bottom = DESIGN_HEIGHT
        self.safe_left = 0
        self.safe_top = 0
        self.safe_right = DESIGN_WIDTH
        self.safe_bottom = DESIGN_HEIGHT
        self.scale = 1
        self.player = 0

    def width(self) -> Int:
        return self.right - self.left

    def height(self) -> Int:
        return self.bottom - self.top

    def safe(self) -> UiRect:
        return UiRect(self.safe_left, self.safe_top, self.safe_right - self.safe_left, self.safe_bottom - self.safe_top)

    def shift_x(self, anchor: Int) -> Int:
        """How far an element designed on the 16:9 canvas moves sideways to
        keep its distance from its anchor's edge (or the centre)."""
        var column = anchor % 3
        if column == 0:
            return self.safe_left
        if column == 2:
            return self.safe_right - DESIGN_WIDTH
        return (self.safe_left + self.safe_right - DESIGN_WIDTH) // 2

    def shift_y(self, anchor: Int) -> Int:
        var row = anchor // 3
        if row == 0:
            return self.safe_top
        if row == 2:
            return self.safe_bottom - DESIGN_HEIGHT
        return (self.safe_top + self.safe_bottom - DESIGN_HEIGHT) // 2

    def x(self, anchor: Int, design_x: Int) -> Int:
        return design_x + self.shift_x(anchor)

    def y(self, anchor: Int, design_y: Int) -> Int:
        return design_y + self.shift_y(anchor)

    def place(self, anchor: Int, design: UiRect) -> UiRect:
        """A box designed at a 16:9 position, moved with its anchor."""
        return UiRect(self.x(anchor, design.x), self.y(anchor, design.y), design.w, design.h)

    def at(self, anchor: Int, offset_x: Int, offset_y: Int, w: Int, h: Int) -> UiRect:
        """A w x h box whose anchor-side corner sits `offset` inside the safe
        area's matching corner (or centred on the centre line)."""
        var column = anchor % 3
        var row = anchor // 3
        var x: Int
        if column == 0:
            x = self.safe_left + offset_x
        elif column == 2:
            x = self.safe_right - offset_x - w
        else:
            x = (self.safe_left + self.safe_right - w) // 2 + offset_x
        var y: Int
        if row == 0:
            y = self.safe_top + offset_y
        elif row == 2:
            y = self.safe_bottom - offset_y - h
        else:
            y = (self.safe_top + self.safe_bottom - h) // 2 + offset_y
        return UiRect(x, y, w, h)


def ui_scale_for(width: Int, height: Int) -> Float32:
    """Pixels per canvas unit: the 1280x720 design area fits the screen."""
    return max(Float32(0.25), min(Float32(width) / Float32(DESIGN_WIDTH), Float32(height) / Float32(DESIGN_HEIGHT)))


def ui_frame(width: Int, height: Int, ui_x: Int, ui_y: Int, scale: Float32, safe_margin: Float32 = 0, viewport_x: Int = 0, viewport_y: Int = 0, viewport_w: Int = -1, viewport_h: Int = -1, player: Int = 0) -> UiFrame:
    """The frame of a screen (or of one player's viewport inside it) whose
    design area starts at pixel (ui_x, ui_y)."""
    var frame = UiFrame()
    var vw = width if viewport_w < 0 else viewport_w
    var vh = height if viewport_h < 0 else viewport_h
    frame.scale = scale
    frame.player = player
    # Pixels to canvas units, rounded inward so nothing lands off screen.
    frame.left = canvas_ceil(Float32(viewport_x - ui_x) / scale)
    frame.top = canvas_ceil(Float32(viewport_y - ui_y) / scale)
    frame.right = canvas_floor(Float32(viewport_x + vw - ui_x) / scale)
    frame.bottom = canvas_floor(Float32(viewport_y + vh - ui_y) / scale)
    var margin = max(Float32(0), min(MAX_SAFE_MARGIN, safe_margin))
    var mx = Int(Float32(frame.width()) * margin + 0.5)
    var my = Int(Float32(frame.height()) * margin + 0.5)
    frame.safe_left = frame.left + mx
    frame.safe_top = frame.top + my
    frame.safe_right = frame.right - mx
    frame.safe_bottom = frame.bottom - my
    return frame


def canvas_floor(value: Float32) -> Int:
    var whole = Int(value)
    if Float32(whole) > value:
        whole -= 1
    return whole


def canvas_ceil(value: Float32) -> Int:
    var whole = Int(value)
    if Float32(whole) < value:
        whole += 1
    return whole
