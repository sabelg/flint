"""Nine-slice images for panels and focus frames (package 3.12).

A `NineSlice` is a small ARGB image with four borders. Drawn into a
rectangle of any size, the four corners keep their size (times the UI scale),
the edges stretch along one axis and the centre along both, so one 48 px
image makes every panel from a button to a full menu without blurring its
rounded corners or its lit top edge.

`slate_panel` and `focus_frame` build the game's two images procedurally, so
no texture file is needed. Pixels are straight (not premultiplied) alpha;
`SoftwareFramebuffer.draw_nine_slice` blends them like every other HUD call.
"""
from std.math import sqrt

comptime SLICE_SLATE: Int = 0
comptime SLICE_FOCUS: Int = 1


struct NineSlice(Copyable, Movable):
    var width: Int
    var height: Int
    var left: Int
    var top: Int
    var right: Int
    var bottom: Int
    var pixels: List[UInt32]   # ARGB, straight alpha

    def __init__(out self, width: Int, height: Int, left: Int, top: Int, right: Int, bottom: Int):
        self.width = width
        self.height = height
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.pixels = List[UInt32](length=width * height, fill=0)

    def at(self, x: Int, y: Int) -> UInt32:
        return self.pixels[max(0, min(self.height - 1, y)) * self.width + max(0, min(self.width - 1, x))]

    def set(mut self, x: Int, y: Int, color: UInt32):
        self.pixels[y * self.width + x] = color


def argb(alpha: Int, red: Int, green: Int, blue: Int) -> UInt32:
    return UInt32(max(0, min(255, alpha))) << 24 | UInt32(max(0, min(255, red))) << 16 | UInt32(max(0, min(255, green))) << 8 | UInt32(max(0, min(255, blue)))


def rounded_depth(x: Int, y: Int, width: Int, height: Int, radius: Float32) -> Float32:
    """Signed distance from pixel (x, y)'s centre in to the edge of a rounded
    rectangle filling the image: positive inside, negative outside."""
    var qx = abs(Float32(x) + 0.5 - Float32(width) * 0.5) - (Float32(width) * 0.5 - radius)
    var qy = abs(Float32(y) + 0.5 - Float32(height) * 0.5) - (Float32(height) * 0.5 - radius)
    var ox = max(qx, Float32(0))
    var oy = max(qy, Float32(0))
    var outside = sqrt(ox * ox + oy * oy) + min(max(qx, qy), Float32(0)) - radius
    return -outside


def rounded_coverage(x: Int, y: Int, width: Int, height: Int, radius: Float32) -> Float32:
    """Coverage (0-1) of pixel (x, y) with a one-pixel anti-aliased edge."""
    return max(Float32(0), min(Float32(1), rounded_depth(x, y, width, height, radius) + 0.5))


def mix(a: Int, b: Int, t: Float32) -> Int:
    return a + Int(Float32(b - a) * t + 0.5)


def slate_panel() -> NineSlice:
    """The carved slate tablet of `panel()` in render/interface.mojo: a dark
    body, a lighter upper face, a teal-lit top edge and a groove along the
    bottom, with rounded corners."""
    comptime SIZE = 48
    comptime BORDER = 16
    var image = NineSlice(SIZE, SIZE, BORDER, BORDER, BORDER, BORDER)
    for y in range(SIZE):
        for x in range(SIZE):
            var coverage = rounded_coverage(x, y, SIZE, SIZE, 11)
            if coverage <= 0:
                continue
            var red = 12
            var green = 24
            var blue = 36
            # The upper face catches light, fading out by the end of the border.
            if y < BORDER:
                var t = Float32(BORDER - y) / Float32(BORDER) * 0.32
                red = mix(red, 40, t)
                green = mix(green, 62, t)
                blue = mix(blue, 74, t)
            # Teal-lit top edge, inset from the corners.
            if (y == 2 or y == 3) and x >= 6 and x < SIZE - 6:
                var t = Float32(0.55) if y == 2 else Float32(0.3)
                red = mix(red, 98, t)
                green = mix(green, 178, t)
                blue = mix(blue, 170, t)
            # A deep groove along the bottom.
            if (y == SIZE - 5 or y == SIZE - 4) and x >= 8 and x < SIZE - 8:
                red = mix(red, 2, 0.6)
                green = mix(green, 6, 0.6)
                blue = mix(blue, 10, 0.6)
            image.set(x, y, argb(Int(coverage * 255.0 + 0.5), red, green, blue))
    return image^


def focus_frame() -> NineSlice:
    """A gold ring that glows inward, drawn over the focused widget. The
    centre is empty, so it stretches to any size for free."""
    comptime SIZE = 40
    comptime BORDER = 14
    var image = NineSlice(SIZE, SIZE, BORDER, BORDER, BORDER, BORDER)
    for y in range(SIZE):
        for x in range(SIZE):
            var coverage = rounded_coverage(x, y, SIZE, SIZE, 11)
            if coverage <= 0:
                continue
            var depth = rounded_depth(x, y, SIZE, SIZE, 11) + 0.5
            var alpha: Float32
            if depth <= 3:
                alpha = 1.0          # the ring itself
            else:
                alpha = max(Float32(0), 0.45 * (1.0 - (depth - 3.0) / 8.0))   # inner glow
            alpha *= coverage
            if alpha <= 0:
                continue
            image.set(x, y, argb(Int(alpha * 255.0 + 0.5), 255, 211 if depth <= 3 else 236, 112 if depth <= 3 else 150))
    return image^


def slice_map(destination: Int, source: Int, first: Int, last: Int, scale: Float32) -> List[Int]:
    """For each of `destination` screen pixels, the source pixel. The first
    and last borders (`first`, `last` source pixels) take `scale` screen
    pixels per source pixel, shrunk together if the rectangle is smaller
    than both; the middle stretches over the source's middle."""
    var map = List[Int](length=max(0, destination), fill=0)
    if destination <= 0:
        return map^
    var head = Int(Float32(first) * scale + 0.5)
    var tail = Int(Float32(last) * scale + 0.5)
    if head + tail > destination:
        var total = max(1, head + tail)
        head = destination * head // total
        tail = destination - head
    var middle = destination - head - tail
    var source_middle = max(1, source - first - last)
    for index in range(destination):
        if index < head:
            map[index] = min(first - 1, Int((Float32(index) + 0.5) * Float32(first) / Float32(head)))
        elif index >= destination - tail:
            var into = index - (destination - tail)
            map[index] = source - last + min(last - 1, Int((Float32(into) + 0.5) * Float32(last) / Float32(tail)))
        else:
            var into = index - head
            map[index] = first + min(source_middle - 1, Int((Float32(into) + 0.5) * Float32(source_middle) / Float32(middle)))
    return map^
