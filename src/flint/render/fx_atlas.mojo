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
from std.math import atan2, cos, isfinite, sin, sqrt
from flint.render.particle_atlas import value_noise

comptime TEX_VIGNETTE: Int = 8
comptime TEX_DROPLET: Int = 9
comptime TEX_RIBBON: Int = 10
comptime TEX_LEAF: Int = 11
comptime TEX_FLAME: Int = 12
comptime TEX_SHOCK_RING: Int = 13
comptime TEX_ANAMORPHIC: Int = 14
comptime TEX_CUMULUS: Int = 15
comptime TEX_FIREBALL: Int = 16
comptime TEX_VORONOI_BEAM: Int = 17
comptime TEX_HEX_SHIELD: Int = 18
comptime TEX_VAPOR_CONE: Int = 19
comptime TEX_LIGHTNING: Int = 20
comptime FX_TEXTURES: Int = 21
comptime FX_TEXTURE_SIZE: Int = 64
comptime VIGNETTE_SIZE: Int = 128


def argb(shade: Float32, alpha: Float32) -> UInt32:
    var sf = clamp01(shade)
    var af = clamp01(alpha)
    var s = UInt32(max(0, min(255, Int(sf * 255.0 + 0.5))))
    var a = UInt32(max(0, min(255, Int(af * 255.0 + 0.5))))
    return (a << 24) | (s << 16) | (s << 8) | s


def clamp01(value: Float32) -> Float32:
    if not isfinite(value):
        return Float32(0.0)
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
        # Seamless periodic turbulent smoke trail along u, wispy curl-noise tendrils across v.
        var pu = u * 3.14159265
        var cu = cos(pu)
        var su = sin(pu)
        var n1 = value_noise(cu * 2.2 + 3.1, v * 3.8 + su * 1.4, 71)
        var n2 = value_noise(su * 4.4 + 1.7, v * 7.6 + cu * 2.1, 79)
        var warp_v = v + (n1 - 0.5) * 0.26 + (n2 - 0.5) * 0.14
        var core = clamp01(1.0 - abs(warp_v) * 1.08)
        var puff = 0.72 + 0.28 * n1
        var shade = clamp01(0.80 + 0.20 * core * puff - 0.08 * (1.0 - n2))
        return argb(shade, clamp01(core * core * puff * (1.0 - smooth(0.84, 1.0, abs(v)))))
    if kind == TEX_LEAF:
        var half = 0.5 * cos(v * 1.5707963) * (1.0 - 0.2 * v)
        var cover = 1.0 - smooth(half - 0.06, half, abs(u))
        var vein = 1.0 - smooth(0.0, 0.05, abs(u))
        return argb(0.75 - 0.25 * vein, cover)
    if kind == TEX_SHOCK_RING:
        # Supersonic spherical blast wave: smooth luminous plasma rim + inner expansion wake.
        var turb = (value_noise(u * 5.0 + 3.0, v * 5.0 + 2.0, 61) - 0.5) * 0.045
        var rr = r + turb
        var rim = 1.0 - smooth(0.0, 0.18, abs(rr - 0.74))
        var wake = (1.0 - smooth(0.38, 0.78, rr)) * smooth(0.16, 0.60, rr) * 0.36
        return argb(0.76 + 0.24 * rim, clamp01(rim * rim + wake))
    if kind == TEX_ANAMORPHIC:
        # Cinema Panavision horizontal lens streak + pinpoint diamond core.
        var h_taper = clamp01(1.0 - abs(u))
        var h_streak = (1.0 - smooth(0.0, 0.14, abs(v))) * h_taper * h_taper
        var h_wide = (1.0 - smooth(0.0, 0.36, abs(v))) * h_taper * h_taper * h_taper * 0.45
        var core = 1.0 - smooth(0.0, 0.22, r)
        return argb(0.82 + 0.18 * core, clamp01(h_streak + h_wide + core * 0.85))
    if kind == TEX_CUMULUS:
        # 4-octave FBM billowing cumulus cloud puff with sunlit crown and flat shadowed base.
        var f1 = value_noise(u * 2.4 + 5.2, v * 2.6 + 1.8, 101)
        var f2 = value_noise(u * 5.1 + 2.3, v * 5.3 + 4.7, 107)
        var f3 = value_noise(u * 10.4 + 7.1, v * 10.8 + 3.3, 113)
        var f4 = value_noise(u * 20.5 + 1.1, v * 20.5 + 8.9, 127)
        var fbm = f1 * 0.50 + f2 * 0.26 + f3 * 0.15 + f4 * 0.09
        # Cauliflower lobes: wider flat anvil base (v > 0.15), billowing turret domes on top (v < 0)
        var eu = u * (Float32(0.92) if v > 0.0 else Float32(1.06))
        var ev = (v - 0.06) * (Float32(1.32) if v > 0.18 else Float32(1.02))
        var er = sqrt(eu * eu + ev * ev)
        var density = clamp01((0.86 - er + (fbm - 0.46) * 0.48) * 2.15)
        density = density * (1.0 - smooth(0.78, 0.98, r))
        # Sunlit top silver-lining + self-occluded base
        var sun_top = clamp01(0.55 - v * 0.68 + (fbm - 0.5) * 0.36)
        var rim_silver = clamp01((1.0 - density) * 0.55) * clamp01(0.7 - v)
        var shade = clamp01(0.58 + 0.38 * sun_top + 0.16 * rim_silver)
        return argb(shade, clamp01(density * density * (1.35 - 0.35 * density)))
    if kind == TEX_FIREBALL:
        # 4-octave cellular pyroclastic fireball: dark outer soot lobes + incandescent magma crevices + hot core.
        var n1 = value_noise(u * 3.4 + 2.1, v * 3.4 + 6.4, 131)
        var n2 = value_noise(u * 7.1 + 8.3, v * 7.1 + 1.9, 137)
        var n3 = value_noise(u * 14.2 + 4.5, v * 14.2 + 5.1, 149)
        var n4 = value_noise(u * 26.0 + 1.7, v * 26.0 + 9.2, 157)
        var cell1 = 1.0 - abs(n1 - 0.5) * 2.0
        var cell2 = 1.0 - abs(n2 - 0.5) * 2.0
        var crevices = clamp01(cell1 * 0.58 + cell2 * 0.30 + n3 * 0.12)
        var turb_r = r + (n1 - 0.5) * 0.26 + (n2 - 0.5) * 0.13 + (n3 - 0.5) * 0.06
        var body = 1.0 - smooth(0.42, 0.92, turb_r)
        var core_hot = 1.0 - smooth(0.0, 0.58, turb_r)
        var heat_field = clamp01(0.14 + 0.52 * crevices * crevices + 0.56 * core_hot + (n4 - 0.5) * 0.08)
        var density = clamp01(body * (0.58 + 0.42 * (n1 * 0.6 + crevices * 0.4))) * (1.0 - smooth(0.82, 0.98, r))
        return argb(heat_field, density)
    if kind == TEX_VORONOI_BEAM:
        # Seamless periodic cellular mega-particle beam along u, white-hot core + churning Voronoi sheath across v.
        var pu = u * 3.14159265
        var cu = cos(pu)
        var su = sin(pu)
        var c1 = 1.0 - abs(value_noise(cu * 2.8 + 1.9, v * 4.6 + su * 2.1, 163) - 0.5) * 2.0
        var c2 = 1.0 - abs(value_noise(su * 5.4 + 4.2, v * 8.8 - cu * 3.2, 173) - 0.5) * 2.0
        var knots = clamp01(c1 * 0.62 + c2 * 0.38)
        var core = 1.0 - smooth(0.0, 0.26, abs(v))
        var sheath = (1.0 - smooth(0.12, 0.96, abs(v))) * (0.48 + 0.62 * knots * knots)
        var shade = clamp01(0.62 + 0.38 * core + 0.22 * knots)
        return argb(shade, clamp01(core + sheath))
    if kind == TEX_HEX_SHIELD:
        # Crisp axial hexagonal honeycomb energy lattice + grazing Fresnel rim ring.
        var sx = u * 5.2
        var sy = v * 5.2
        var qx = sx * 0.8660254 + sy * 0.5
        var qy = sy
        var rx_f = qx - Float32(Int(qx + 32.0)) + 32.0 - 0.5
        var ry_f = qy - Float32(Int(qy + 32.0)) + 32.0 - 0.5
        var rz_f = -rx_f - ry_f
        var hex_edge = max(abs(rx_f - ry_f), max(abs(ry_f - rz_f), abs(rz_f - rx_f)))
        var grid = smooth(0.62, 0.96, hex_edge)
        var disc = 1.0 - smooth(0.88, 0.99, r)
        var fresnel_rim = smooth(0.38, 0.88, r) * disc
        var pulse_n = value_noise(u * 4.0 + 2.0, v * 4.0 + 5.0, 181)
        var cover = clamp01((grid * (0.38 + 0.62 * fresnel_rim) + fresnel_rim * 0.58 + 0.14 * pulse_n) * disc)
        return argb(clamp01(0.72 + 0.28 * grid), cover)
    if kind == TEX_VAPOR_CONE:
        # Prandtl-Glauert transonic condensation collar: hollow around fuselage (r < 0.24), streaked Mach skirt.
        var ang = atan2(v, u)
        var s1 = value_noise(cos(ang * 5.0) * 2.6 + 3.0, sin(ang * 5.0) * 2.6 + r * 4.5, 191)
        var s2 = value_noise(cos(ang * 11.0) * 3.4 + 1.2, sin(ang * 11.0) * 3.4 + r * 8.5, 197)
        var streaks = 0.52 + 0.48 * (s1 * 0.65 + s2 * 0.35)
        var inner_cut = smooth(0.18, 0.38, r)
        var outer_fade = 1.0 - smooth(0.56 + (s1 - 0.5) * 0.22, 0.96, r)
        var collar_peak = 1.0 - smooth(0.0, 0.26, abs(r - 0.48))
        var cover = clamp01(inner_cut * outer_fade * streaks * (0.65 + 0.55 * collar_peak))
        return argb(clamp01(0.84 + 0.16 * collar_peak), cover)
    if kind == TEX_LIGHTNING:
        # Seamless jagged high-voltage plasma bolt + secondary fork along u.
        var pu = u * 3.14159265
        var cu = cos(pu)
        var su = sin(pu)
        var j1 = (value_noise(cu * 3.5 + 1.1, su * 3.5 + 4.3, 211) - 0.5) * 0.44
        var j2 = (value_noise(cu * 8.2 + 5.7, su * 8.2 + 2.9, 223) - 0.5) * 0.22
        var j3 = (value_noise(cu * 17.0 + 3.3, su * 17.0 + 7.1, 229) - 0.5) * 0.10
        var main_d = abs(v - (j1 + j2 + j3))
        var fork_d = abs(v - (j1 * 0.7 - j2 * 1.35 + 0.22 * sin(pu * 2.0)))
        var main_bolt = 1.0 - smooth(0.015, 0.095, main_d)
        var main_halo = (1.0 - smooth(0.04, 0.38, main_d)) * 0.48
        var fork_bolt = (1.0 - smooth(0.010, 0.065, fork_d)) * smooth(-0.2, 0.5, sin(pu)) * 0.72
        var cover = clamp01(main_bolt + main_halo + fork_bolt)
        return argb(clamp01(0.76 + 0.24 * main_bolt), cover)
    # Flame: a teardrop, tip up (v = -1).
    var t = (v + 1.0) * 0.5
    var half = 0.62 * sqrt(clamp01(t)) * max(Float32(0), 1.0 - t * t * t) * 1.2
    var body = clamp01((half - abs(u)) / 0.2)
    var hot = clamp01((half * 0.45 - abs(u)) / 0.2) * clamp01(t * 2.0 - 0.6)
    return argb(0.75 + 0.25 * hot, body)


def fx_texture_size(kind: Int) -> Int:
    if (
        kind == TEX_VIGNETTE
        or kind == TEX_SHOCK_RING
        or kind == TEX_ANAMORPHIC
        or kind == TEX_CUMULUS
        or kind == TEX_FIREBALL
        or kind == TEX_VORONOI_BEAM
        or kind == TEX_HEX_SHIELD
        or kind == TEX_VAPOR_CONE
        or kind == TEX_LIGHTNING
    ):
        return VIGNETTE_SIZE
    return FX_TEXTURE_SIZE


def fx_texture(kind: Int) -> List[UInt32]:
    var n = fx_texture_size(kind)
    var pixels = List[UInt32](length=n * n, fill=0)
    for y in range(n):
        for x in range(n):
            var u = (Float32(x) + 0.5) / Float32(n) * 2.0 - 1.0
            var v = (Float32(y) + 0.5) / Float32(n) * 2.0 - 1.0
            pixels[y * n + x] = look(kind, u, v)
    return pixels^


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
