"""The view basis turns smoothly as a camera pitches down to (almost) straight down, and stays
left-handed: looking along +z, +x is on the right."""
from std.math import cos, sin, abs
from std.testing import assert_true
from flint.core.math import Vec3
from flint.render.camera_basis import view_right, view_up


def fwd_at(yaw: Float32, pitch: Float32) -> Vec3:
    """Unit forward for a camera turned yaw about +y (0 looks along +z) and pitched down."""
    var c = cos(pitch)
    return Vec3(sin(yaw) * c, -sin(pitch), cos(yaw) * c)


def main() raises:
    # left-handed: looking along +z, right is +x and up is +y
    var r = view_right(Vec3(0.0, 0.0, 1.0))
    assert_true(r.x > Float32(0.999))
    assert_true(view_up(Vec3(0.0, 0.0, 1.0)).y > Float32(0.999))
    # pitching down from 60 to 89.99 degrees at any heading: no jumps (it used to flip at ~72)
    var yaws = List[Float32]()
    for k in range(8):
        yaws.append(Float32(k) * Float32(0.785398))
    var worst = Float32(0.0)
    for yaw in yaws:
        var prev_r = view_right(fwd_at(yaw, Float32(1.0472)))
        var prev_u = view_up(fwd_at(yaw, Float32(1.0472)))
        var steps = 3000
        for s in range(1, steps + 1):
            var pitch = Float32(1.0472) + (Float32(1.570621) - Float32(1.0472)) * Float32(s) / Float32(steps)
            var f = fwd_at(yaw, pitch)
            var rr = view_right(f)
            var uu = view_up(f)
            var dr = (rr - prev_r).length()
            var du = (uu - prev_u).length()
            worst = max(worst, max(dr, du))
            assert_true(dr < Float32(0.02) and du < Float32(0.02))
            assert_true(abs(rr.dot(f)) < Float32(1e-3) and abs(uu.dot(f)) < Float32(1e-3))
            assert_true(abs(rr.y) < Float32(1e-3))                   # no roll: right stays level
            prev_r = rr
            prev_u = uu
    # straight down still gives a usable, level basis
    var down = Vec3(0.0, -1.0, 0.0)
    assert_true(view_right(down).length() > Float32(0.99) and abs(view_right(down).y) < Float32(1e-6))
    assert_true(view_up(down).length() > Float32(0.99))
    print("camera basis: smooth to 89.99 degrees at 8 headings, worst step", worst)
