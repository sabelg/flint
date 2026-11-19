"""The view basis a camera builds from its forward direction.

Flint's view is left-handed: `right = up_ref x fwd`, `up = fwd x right`. The reference up is the
world's +y whenever the forward direction leans any measurable way off vertical (+y x fwd stays
well defined down to a horizontal lean of 1e-7), so a camera pitching through 72, 85, 89.99
degrees turns smoothly instead of snapping half a turn (as a fallback at |fwd.y| > 0.95 used
to). Only looking exactly straight up or down falls back to -z (down) or +z (up): the limit +y
gives as the camera tilts over from the +z side.
"""
from std.math import sqrt
from flint.core.math import Vec3

comptime VIEW_LEAN = Float32(1e-7)


def view_up_ref(fwd: Vec3) -> Vec3:
    """The reference up for a camera looking along unit fwd."""
    if sqrt(fwd.x * fwd.x + fwd.z * fwd.z) > VIEW_LEAN:
        return Vec3(0.0, 1.0, 0.0)
    return Vec3(0.0, 0.0, -1.0) if fwd.y < Float32(0.0) else Vec3(0.0, 0.0, 1.0)


def view_right(fwd: Vec3) -> Vec3:
    """The camera's right (screen +x) for unit fwd."""
    return view_up_ref(fwd).cross(fwd).normalized()


def view_up(fwd: Vec3) -> Vec3:
    """The camera's up (screen +y) for unit fwd."""
    return fwd.cross(view_right(fwd)).normalized()
