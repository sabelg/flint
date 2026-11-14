"""Critically damped springs for smoothing values over time.

After Daniel Holden, "Spring-It-On: The Game Developer's Spring-Roll-Call".
Every spring here is parameterised by a halflife: the time in seconds it
takes to cover half the remaining distance to its goal. The exact solution
is used, so results do not depend on the frame rate.
"""
from std.math import log
from flint.core.math import Vec3


comptime LN2: Float32 = 0.69314718056


def fast_negexp(x: Float32) -> Float32:
    """Approximates exp(-x) for x >= 0 without calling exp."""
    return 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)


def halflife_to_damping(halflife: Float32) -> Float32:
    return (4.0 * LN2) / max(halflife, Float32(0.00001))


struct SpringFloat(Copyable, ImplicitlyCopyable, Movable):
    """A value chasing a goal with no overshoot."""
    var value: Float32
    var velocity: Float32

    def __init__(out self, value: Float32 = 0):
        self.value = value
        self.velocity = 0

    def update(mut self, goal: Float32, halflife: Float32, dt: Float32):
        var y = halflife_to_damping(halflife) / 2.0
        var j0 = self.value - goal
        var j1 = self.velocity + j0 * y
        var eydt = fast_negexp(y * dt)
        self.value = eydt * (j0 + j1 * dt) + goal
        self.velocity = eydt * (self.velocity - j1 * y * dt)


struct SpringVec3(Copyable, ImplicitlyCopyable, Movable):
    var value: Vec3
    var velocity: Vec3

    def __init__(out self, value: Vec3 = Vec3()):
        self.value = value
        self.velocity = Vec3()

    def update(mut self, goal: Vec3, halflife: Float32, dt: Float32):
        var y = halflife_to_damping(halflife) / 2.0
        var j0 = self.value - goal
        var j1 = self.velocity + j0 * y
        var eydt = fast_negexp(y * dt)
        self.value = (j0 + j1 * dt) * eydt + goal
        self.velocity = (self.velocity - j1 * (y * dt)) * eydt


def wrap_degrees(angle: Float32) -> Float32:
    var result = angle
    while result > 180:
        result -= 360
    while result < -180:
        result += 360
    return result


struct SpringAngle(Copyable, ImplicitlyCopyable, Movable):
    """A heading in degrees that always turns the short way round."""
    var value: Float32
    var velocity: Float32

    def __init__(out self, value: Float32 = 0):
        self.value = value
        self.velocity = 0

    def update(mut self, goal: Float32, halflife: Float32, dt: Float32):
        var y = halflife_to_damping(halflife) / 2.0
        var j0 = wrap_degrees(self.value - goal)
        var j1 = self.velocity + j0 * y
        var eydt = fast_negexp(y * dt)
        self.value = wrap_degrees(eydt * (j0 + j1 * dt) + goal)
        self.velocity = eydt * (self.velocity - j1 * y * dt)


def decay_spring(mut offset: Vec3, mut velocity: Vec3, halflife: Float32, dt: Float32):
    """Moves an offset and its velocity toward zero, as inertialization needs."""
    var y = halflife_to_damping(halflife) / 2.0
    var j1 = velocity + offset * y
    var eydt = fast_negexp(y * dt)
    offset = (offset + j1 * dt) * eydt
    velocity = (velocity - j1 * (y * dt)) * eydt
