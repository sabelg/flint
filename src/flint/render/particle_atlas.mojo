"""The particles' flipbook atlas (package 2.7): ATLAS_FRAMES frames of
ATLAS_FRAME x ATLAS_FRAME coverage bytes, made here from hashed value noise
(no image files), the same bytes every run.

- frames 0-7: a dust or smoke puff as it ages: a lumpy disc that loosens,
  thins and breaks into wisps.
- frames 8-11: an impact flash: a hot core with six rays that shorten.
- frames 12-15: flame tongues (package 2.13; they were spare crumbs): a
  teardrop with its tip up, licked sideways a little differently each frame.
"""
from std.math import sqrt, atan2, cos, floor, isfinite, isnan
from flint.render.particle_layout import ATLAS_FRAME, ATLAS_FRAMES, ATLAS_BYTES


def lattice(x: Int, y: Int, seed: Int) -> Float32:
    var h = UInt32((x * 374761393 + y * 668265263 + seed * 1442695041) & 0xFFFFFFFF)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return Float32(h & 0xFFFF) / 65535.0


def value_noise(x: Float32, y: Float32, seed: Int) -> Float32:
    if not isfinite(x) or not isfinite(y):
        return Float32(0.5)
    var xd = Float64(x)
    var yd = Float64(y)
    if abs(xd) > 1.0e12:
        xd = xd - floor(xd / 65536.0) * 65536.0
    if abs(yd) > 1.0e12:
        yd = yd - floor(yd / 65536.0) * 65536.0
    var fx_fl = floor(xd)
    var fy_fl = floor(yd)
    var ix = Int(fx_fl)
    var iy = Int(fy_fl)
    var fx = Float32(xd - fx_fl)
    var fy = Float32(yd - fy_fl)
    var sx = fx * fx * (3 - 2 * fx)
    var sy = fy * fy * (3 - 2 * fy)
    var a = lattice(ix, iy, seed)
    var b = lattice(ix + 1, iy, seed)
    var c = lattice(ix, iy + 1, seed)
    var d = lattice(ix + 1, iy + 1, seed)
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy


def to_byte(value: Float32) -> UInt8:
    if isnan(value) or value <= Float32(0.0):
        return UInt8(0)
    if value >= Float32(1.0):
        return UInt8(255)
    return UInt8(max(0, min(255, Int(value * 255.0 + 0.5))))


def particle_atlas() -> List[UInt8]:
    var bytes = List[UInt8](length=ATLAS_BYTES, fill=0)
    var n = ATLAS_FRAME
    for frame in range(ATLAS_FRAMES):
        for y in range(n):
            for x in range(n):
                var u = (Float32(x) + 0.5) / Float32(n) * 2.0 - 1.0
                var v = (Float32(y) + 0.5) / Float32(n) * 2.0 - 1.0
                var r = sqrt(u * u + v * v)
                var value: Float32
                if frame < 8:
                    var age = Float32(frame) / 7.0
                    var lumps = value_noise(u * 2.6 + 3.0, v * 2.6 + 5.0, 7 + frame // 3) * 0.6 + value_noise(u * 5.2, v * 5.2, 11) * 0.4
                    var edge = 1.0 - r * (0.9 + 0.35 * age) + (lumps - 0.5) * (0.5 + 0.6 * age)
                    var density = max(Float32(0), min(Float32(1), edge * 2.2))
                    value = density * (0.95 - 0.55 * age) * (0.75 + 0.5 * lumps)
                elif frame < 12:
                    var stage = Float32(frame - 8) / 3.0
                    var angle = atan2(v, u)
                    var c = max(Float32(0), cos(angle * 6.0))
                    var c2 = c * c
                    var rays = c2 * c2 * c2 * c2
                    var core = max(Float32(0), 1.0 - r * (1.6 + stage))
                    var reach = max(Float32(0), 1.0 - r / (1.0 - 0.45 * stage))
                    value = min(Float32(1), core * 1.4 + rays * reach * (1.0 - 0.6 * stage))
                else:
                    # A flame tongue (package 2.13): a teardrop with its tip
                    # at the top (v = -1), licked sideways by noise that
                    # differs from frame to frame, a hotter middle.
                    var t = (v + 1.0) * 0.5
                    var lick = (value_noise(t * 3.0 + Float32(frame) * 5.0, Float32(frame) * 1.7, 41 + frame) - 0.5) * 0.45 * (1.0 - t)
                    var half = 0.62 * sqrt(max(Float32(0), t)) * max(Float32(0), 1.0 - t * t * t) * 1.2
                    var off = abs(u - lick)
                    var body = max(Float32(0), min(Float32(1), (half - off) / 0.16))
                    var grain = value_noise(u * 4.0 + Float32(frame) * 3.0, v * 4.0 - Float32(frame) * 2.0, 53)
                    value = body * (0.65 + 0.35 * grain)
                bytes[frame * n * n + y * n + x] = to_byte(value)
    return bytes^
