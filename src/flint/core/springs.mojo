"""Critically damped springs for smoothing values over time.

After Daniel Holden, "Spring-It-On: The Game Developer's Spring-Roll-Call".
Every spring here is parameterised by a halflife: the time in seconds it
takes to cover half the remaining distance to its goal. For a critically
damped response x(t) = (1 + y*t) * exp(-y*t) starting from rest, solving
(1 + u) * exp(-u) = 0.5 yields u = y * halflife = 1.67834699, so
damping = 2 * y = 3.356694 / halflife. Deterministic `det_exp` is used so
simulation and lockstep animation remain bit-exact across ISAs and frame rates.
"""
from std.math import floor
from flint.core.detmath import det_exp
from flint.core.math import Vec3


comptime LN2: Float32 = 0.69314718056
comptime CRIT_HALFLIFE_DAMPING: Float32 = 3.356694


@always_inline
def fast_negexp(x: Float32) -> Float32:
    """Rational Pade-like approximation to exp(-x) for x >= 0."""
    return Float32(1.0) / (Float32(1.0) + x + Float32(0.48) * x * x + Float32(0.235) * x * x * x)


@always_inline
def halflife_to_damping(halflife: Float32) -> Float32:
    """Exact critically damped coefficient: (1 + 1.678347) * exp(-1.678347) == 0.50000."""
    return CRIT_HALFLIFE_DAMPING / max(halflife, Float32(1.0e-5))


struct SpringFloat(Copyable, ImplicitlyCopyable, Movable):
    """A value chasing a goal with no overshoot."""
    var value: Float32
    var velocity: Float32

    def __init__(out self, value: Float32 = 0):
        self.value = value
        self.velocity = 0

    def update(mut self, goal: Float32, halflife: Float32, dt: Float32):
        if dt <= 0:
            return
        var y = halflife_to_damping(halflife) * Float32(0.5)
        var j0 = self.value - goal
        var j1 = self.velocity + j0 * y
        var eydt = det_exp(-y * dt)
        self.value = eydt * (j0 + j1 * dt) + goal
        self.velocity = eydt * (self.velocity - j1 * y * dt)


struct SpringVec3(Copyable, ImplicitlyCopyable, Movable):
    var value: Vec3
    var velocity: Vec3

    def __init__(out self, value: Vec3 = Vec3()):
        self.value = value
        self.velocity = Vec3()

    def update(mut self, goal: Vec3, halflife: Float32, dt: Float32):
        if dt <= 0:
            return
        var y = halflife_to_damping(halflife) * Float32(0.5)
        var j0 = self.value - goal
        var j1 = self.velocity + j0 * y
        var eydt = det_exp(-y * dt)
        self.value = (j0 + j1 * dt) * eydt + goal
        self.velocity = (self.velocity - j1 * (y * dt)) * eydt


@always_inline
def wrap_degrees(angle: Float32) -> Float32:
    """Branchless angle wrap into [-180, 180) degrees via floor."""
    return angle - Float32(360.0) * floor((angle + Float32(180.0)) * Float32(1.0 / 360.0))


struct SpringAngle(Copyable, ImplicitlyCopyable, Movable):
    """A heading in degrees that always turns the short way round."""
    var value: Float32
    var velocity: Float32

    def __init__(out self, value: Float32 = 0):
        self.value = value
        self.velocity = 0

    def update(mut self, goal: Float32, halflife: Float32, dt: Float32):
        if dt <= 0:
            return
        var y = halflife_to_damping(halflife) * Float32(0.5)
        var j0 = wrap_degrees(self.value - goal)
        var j1 = self.velocity + j0 * y
        var eydt = det_exp(-y * dt)
        self.value = wrap_degrees(eydt * (j0 + j1 * dt) + goal)
        self.velocity = eydt * (self.velocity - j1 * y * dt)


def decay_spring(mut offset: Vec3, mut velocity: Vec3, halflife: Float32, dt: Float32):
    """Moves a 3D offset and its velocity toward zero, as inertialization needs."""
    if dt <= 0:
        return
    var y = halflife_to_damping(halflife) * Float32(0.5)
    var j1 = velocity + offset * y
    var eydt = det_exp(-y * dt)
    offset = (offset + j1 * dt) * eydt
    velocity = (velocity - j1 * (y * dt)) * eydt


def decay_spring_scalar(mut offset: Float32, mut velocity: Float32, halflife: Float32, dt: Float32):
    """Moves a scalar offset and its velocity toward zero along a fixed axis."""
    if dt <= 0:
        return
    var y = halflife_to_damping(halflife) * Float32(0.5)
    var j1 = velocity + offset * y
    var eydt = det_exp(-y * dt)
    offset = (offset + j1 * dt) * eydt
    velocity = (velocity - j1 * (y * dt)) * eydt
