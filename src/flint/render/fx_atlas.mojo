"""The effects library's textures (package 2.13), made here from hashed
value noise (no image files), the same pixels every run.

One small ARGB texture per look, so a texture wraps on itself (beams scroll
along u). Colour is a grey shade the draw tints with its own colour; alpha is
the coverage. Indices are gameplay/fx_def.mojo's TEX_* and the extra looks
below.

    claw      three gouges side by side, dark in the cut, a light rim
    scorch    a blotch, darkest in the middle, soot fraying at the edge
    crack     cracks branching from the middle
    paw       a bear's print: the palm pad, five toes, claw points
    splat     a blob with drops round it
    bolt      across: a hot core and a glow (u runs along the beam)
    stripes   the glow with soft bands along u (scrolled for light beams)
    soft      a round soft disc (glows, billboards, CPU sprites)
    vignette  128 px: clear in the middle, opaque at the corners
    droplet   a water drop on the lens: a dark rim, a bright fleck
    ribbon    across: full in the middle, fading to both edges
    leaf      a pointed leaf with its vein
    flame     a flame tongue, tip up (the CPU path's fire)
"""
from std.math import sqrt, atan2, cos, sin
from flint.render.particle_atlas import value_noise

comptime TEX_VIGNETTE: Int = 8
comptime TEX_DROPLET: Int = 9
comptime TEX_RIBBON: Int = 10
comptime TEX_LEAF: Int = 11
comptime TEX_FLAME: Int = 12
comptime FX_TEXTURES: Int = 13
comptime FX_TEXTURE_SIZE: Int = 64
comptime VIGNETTE_SIZE: Int = 128


def argb(shade: Float32, alpha: Float32) -> UInt32:
    var s = UInt32(max(0, min(255, Int(shade * 255.0 + 0.5))))
    var a = UInt32(max(0, min(255, Int(alpha * 255.0 + 0.5))))
    return (a << 24) | (s << 16) | (s << 8) | s


def clamp01(value: Float32) -> Float32:
    return max(Float32(0), min(Float32(1), value))


def smooth(edge0: Float32, edge1: Float32, x: Float32) -> Float32:
    var t = clamp01((x - edge0) / (edge1 - edge0))
    return t * t * (3 - 2 * t)


def segment_distance(px: Float32, py: Float32, ax: Float32, ay: Float32, bx: Float32, by: Float32) -> Float32:
    var dx = bx - ax
    var dy = by - ay
    var t = clamp01(((px - ax) * dx + (py - ay) * dy) / max(Float32(1.0e-6), dx * dx + dy * dy))
    var cx = ax + dx * t - px
    var cy = ay + dy * t - py
    return sqrt(cx * cx + cy * cy)


def look(kind: Int, u: Float32, v: Float32) -> UInt32:
    """One texel of look `kind` at (u, v), each in -1..1."""
    var r = sqrt(u * u + v * v)
    if kind == 0:
        # Claw: three gouges slanting a little, each a tapered stroke.
        var best = Float32(1.0e9)
        var along = Float32(0)
        for k in range(3):
            var x = Float32(k - 1) * 0.42
            var d = segment_distance(u, v, x - 0.18, -0.82, x + 0.18, 0.82)
            if d < best:
                best = d
                along = clamp01((v + 0.82) / 1.64)
        var width = 0.13 * sin(along * 3.1415926) + 0.02
        var cut = 1.0 - smooth(width * 0.55, width, best)
        var rim = (1.0 - smooth(width, width * 1.7, best)) - cut
        var grain = value_noise(u * 9.0, v * 9.0, 3)
        return argb(0.25 + 0.75 * rim + 0.1 * grain, clamp01(cut + rim * 0.6) * (0.8 + 0.2 * grain))
    if kind == 1:
        # Scorch: a soot blotch that frays.
        var n = value_noise(u * 3.0 + 4.0, v * 3.0 + 1.0, 17) * 0.6 + value_noise(u * 7.0, v * 7.0, 19) * 0.4
        var edge = 1.0 - r * 1.05 + (n - 0.5) * 0.55
        var cover = clamp01(edge * 2.4)
        return argb(0.15 + 0.6 * r + 0.2 * n, cover * (0.55 + 0.45 * (1.0 - r)))
    if kind == 2:
        # Crack: eight wandering branches from the middle.
        var angle = atan2(v, u)
        var best = Float32(1)
        for k in range(8):
            var a = Float32(k) * 0.785398 + (value_noise(Float32(k) * 3.1, 0.5, 23) - 0.5) * 0.6
            var wander = (value_noise(r * 4.0, Float32(k) * 2.3, 29) - 0.5) * 0.5
            var d = abs(angle - a - wander)
            if d > 3.1415926:
                d = 6.2831853 - d
            var reach = 0.55 + 0.4 * value_noise(Float32(k) * 1.7, 4.2, 31)
            if r < reach:
                best = min(best, d * r)
        var line = 1.0 - smooth(0.012, 0.045, best)
        var core = 1.0 - smooth(0.0, 0.16, r)
        return argb(0.2, clamp01(line + core * 0.7))
    if kind == 3:
        # Paw: the palm pad low, five toes in an arc above, claw points.
        var cover = Float32(0)
        var pu = u / 0.62
        var pv = (v - 0.32) / 0.42
        if pu * pu + pv * pv < 1:
            cover = 1
        for k in range(5):
            var a = -0.95 + Float32(k) * 0.475
            var tx = sin(a) * 0.62
            var ty = -0.28 - cos(a) * 0.38
            var tr = 0.15 if k == 0 or k == 4 else Float32(0.17)
            var dx = u - tx
            var dy = v - ty
            if dx * dx + dy * dy < tr * tr:
                cover = 1
            var cx = u - sin(a) * 0.92
            var cy = v - (-0.28 - cos(a) * 0.66)
            if cx * cx + cy * cy < 0.05 * 0.05:
                cover = max(cover, Float32(0.8))
        var grain = value_noise(u * 8.0, v * 8.0, 37)
        return argb(0.4 + 0.3 * grain, cover * (0.75 + 0.25 * grain))
    if kind == 4:
        # Splat: a lumpy blob and drops round it.
        var n = value_noise(u * 2.5 + 7.0, v * 2.5, 41)
        var cover = 1.0 - smooth(0.5 + (n - 0.5) * 0.5, 0.56 + (n - 0.5) * 0.5, r)
        for k in range(6):
            var a = Float32(k) * 1.047 + 0.4
            var dx = u - cos(a) * 0.78
            var dy = v - sin(a) * 0.78
            if dx * dx + dy * dy < 0.012:
                cover = 1
        return argb(0.55 + 0.3 * n, cover)
    if kind == 5:
        # Bolt: across v, a white-hot core and a wider glow.
        var core = 1.0 - smooth(0.0, 0.22, abs(v))
        var glow = 1.0 - smooth(0.1, 1.0, abs(v))
        return argb(0.7 + 0.3 * core, clamp01(core + glow * 0.65))
    if kind == 6:
        # Stripes: the glow with soft bands along u.
        var glow = 1.0 - smooth(0.0, 1.0, abs(v))
        var band = 0.6 + 0.4 * cos(u * 3.1415926 * 2.0)
        return argb(0.85 + 0.15 * band, glow * band)
    if kind == 7:
        var soft = 1.0 - smooth(0.0, 1.0, r)
        return argb(1.0, soft * soft)
    if kind == TEX_VIGNETTE:
        # Clear inside an ellipse, rising to the corners.
        var e = sqrt(u * u * 0.9 + v * v * 1.1)
        return argb(1.0, smooth(0.62, 1.4, e))
    if kind == TEX_DROPLET:
        var rim = 1.0 - smooth(0.0, 0.25, abs(r - 0.72))
        var fleck_x = u + 0.3
        var fleck_y = v + 0.32
        var fleck = 1.0 - smooth(0.0, 0.18, sqrt(fleck_x * fleck_x + fleck_y * fleck_y))
        var inside = 1.0 - smooth(0.8, 0.9, r)
        return argb(0.55 + 0.45 * fleck - 0.3 * rim, clamp01(inside * 0.35 + rim * 0.5 + fleck * 0.6))
    if kind == TEX_RIBBON:
        var across = 1.0 - abs(v)
        return argb(1.0, across * across)
    if kind == TEX_LEAF:
        var half = 0.5 * cos(v * 1.5707963) * (1.0 - 0.2 * v)
        var cover = 1.0 - smooth(half - 0.06, half, abs(u))
        var vein = 1.0 - smooth(0.0, 0.05, abs(u))
        return argb(0.75 - 0.25 * vein, cover)
    # Flame: a teardrop, tip up (v = -1).
    var t = (v + 1.0) * 0.5
    var half = 0.62 * sqrt(clamp01(t)) * max(Float32(0), 1.0 - t * t * t) * 1.2
    var body = clamp01((half - abs(u)) / 0.2)
    var hot = clamp01((half * 0.45 - abs(u)) / 0.2) * clamp01(t * 2.0 - 0.6)
    return argb(0.75 + 0.25 * hot, body)


def fx_texture(kind: Int) -> List[UInt32]:
    var n = VIGNETTE_SIZE if kind == TEX_VIGNETTE else FX_TEXTURE_SIZE
    var pixels = List[UInt32](length=n * n, fill=0)
    for y in range(n):
        for x in range(n):
            var u = (Float32(x) + 0.5) / Float32(n) * 2.0 - 1.0
            var v = (Float32(y) + 0.5) / Float32(n) * 2.0 - 1.0
            pixels[y * n + x] = look(kind, u, v)
    return pixels^


def fx_texture_size(kind: Int) -> Int:
    return VIGNETTE_SIZE if kind == TEX_VIGNETTE else FX_TEXTURE_SIZE


struct FxTextures(Movable):
    """Every look, made once. The lists never move after this (the GPU keeps
    each by its address)."""
    var pixels: List[List[UInt32]]
    var sizes: List[Int]

    def __init__(out self):
        self.pixels = List[List[UInt32]]()
        self.sizes = List[Int]()
        for kind in range(FX_TEXTURES):
            self.pixels.append(fx_texture(kind))
            self.sizes.append(fx_texture_size(kind))
