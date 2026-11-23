"""Critically damped springs for smoothing values over time.

After Daniel Holden, "Spring-It-On: The Game Developer's Spring-Roll-Call".
Every spring here is parameterised by a halflife: the time in seconds it
takes to cover half the remaining distance to its goal. For a critically
damped response x(t) = (1 + y*t) * exp(-y*t) starting from rest, solving
(1 + u) * exp(-u) = 0.5 yields u = y * halflife = 1.67834699, so
damping = 2 * y = 3.356694 / halflife. Deterministic `det_exp` is used so
simulation and lockstep animation remain bit-exact across ISAs and frame rates.
"""
from std.math import floor, isfinite, isnan
from flint.core.detmath import det_exp
from flint.core.math import Vec3


comptime LN2: Float32 = 0.69314718056
comptime CRIT_HALFLIFE_DAMPING: Float32 = 3.356694


@always_inline
def fast_negexp(x: Float32) -> Float32:
    """Rational Pade-like approximation to exp(-x) for x >= 0."""
    if not isfinite(x):
        return Float32(0.0) if (not isnan(x) and x > Float32(0.0)) else Float32(1.0)
    if x <= Float32(0.0):
        return Float32(1.0)
    if x >= Float32(1.0e12):
        return Float32(0.0)
    return Float32(1.0) / (Float32(1.0) + x + Float32(0.48) * x * x + Float32(0.235) * x * x * x)


@always_inline
def halflife_to_damping(halflife: Float32) -> Float32:
    """Exact critically damped coefficient: (1 + 1.678347) * exp(-1.678347) == 0.50000."""
    if not isfinite(halflife):
        return CRIT_HALFLIFE_DAMPING / Float32(1.0e-5)
    return CRIT_HALFLIFE_DAMPING / max(halflife, Float32(1.0e-5))


struct SpringFloat(Copyable, ImplicitlyCopyable, Movable):
    """A value chasing a goal with no overshoot."""
    var value: Float32
    var velocity: Float32

    def __init__(out self, value: Float32 = 0):
        self.value = value
        self.velocity = 0

    def update(mut self, goal: Float32, halflife: Float32, dt: Float32):
        if not isfinite(dt) or dt <= Float32(0.0) or not isfinite(goal) or not isfinite(halflife):
            return
        if not isfinite(self.value) or not isfinite(self.velocity):
            self.value = goal
            self.velocity = Float32(0.0)
            return
        var y = halflife_to_damping(halflife) * Float32(0.5)
        var eydt = det_exp(-y * dt)
        if not isfinite(eydt) or eydt == Float32(0.0):
            self.value = goal
            self.velocity = Float32(0.0)
            return
        var e64 = Float64(eydt)
        var y64 = Float64(y)
        var dt64 = Float64(dt)
        var goal64 = Float64(goal)
        var j0 = Float64(self.value) - goal64
        var j1 = Float64(self.velocity) + j0 * y64
        var next_val = Float32(e64 * (j0 + j1 * dt64) + goal64)
        var next_vel = Float32(e64 * (Float64(self.velocity) - j1 * (y64 * dt64)))
        if not isfinite(next_val) or not isfinite(next_vel):
            self.value = goal
            self.velocity = Float32(0.0)
            return
        self.value = next_val
        self.velocity = next_vel


struct SpringVec3(Copyable, ImplicitlyCopyable, Movable):
    var value: Vec3
    var velocity: Vec3

    def __init__(out self, value: Vec3 = Vec3()):
        self.value = value
        self.velocity = Vec3()

    def update(mut self, goal: Vec3, halflife: Float32, dt: Float32):
        if (
            not isfinite(dt)
            or dt <= Float32(0.0)
            or not isfinite(halflife)
            or not (isfinite(goal.x) and isfinite(goal.y) and isfinite(goal.z))
        ):
            return
        if (
            not (isfinite(self.value.x) and isfinite(self.value.y) and isfinite(self.value.z))
            or not (isfinite(self.velocity.x) and isfinite(self.velocity.y) and isfinite(self.velocity.z))
        ):
            self.value = goal
            self.velocity = Vec3()
            return
        var y = halflife_to_damping(halflife) * Float32(0.5)
        var eydt = det_exp(-y * dt)
        if not isfinite(eydt) or eydt == Float32(0.0):
            self.value = goal
            self.velocity = Vec3()
            return
        var e64 = Float64(eydt)
        var y64 = Float64(y)
        var dt64 = Float64(dt)

        var j0x = Float64(self.value.x) - Float64(goal.x)
        var j0y = Float64(self.value.y) - Float64(goal.y)
        var j0z = Float64(self.value.z) - Float64(goal.z)
        var j1x = Float64(self.velocity.x) + j0x * y64
        var j1y = Float64(self.velocity.y) + j0y * y64
        var j1z = Float64(self.velocity.z) + j0z * y64

        var nvx = Float32(e64 * (j0x + j1x * dt64) + Float64(goal.x))
        var nvy = Float32(e64 * (j0y + j1y * dt64) + Float64(goal.y))
        var nvz = Float32(e64 * (j0z + j1z * dt64) + Float64(goal.z))
        var nvelx = Float32(e64 * (Float64(self.velocity.x) - j1x * (y64 * dt64)))
        var nvely = Float32(e64 * (Float64(self.velocity.y) - j1y * (y64 * dt64)))
        var nvelz = Float32(e64 * (Float64(self.velocity.z) - j1z * (y64 * dt64)))
        if (
            not (isfinite(nvx) and isfinite(nvy) and isfinite(nvz))
            or not (isfinite(nvelx) and isfinite(nvely) and isfinite(nvelz))
        ):
            self.value = goal
            self.velocity = Vec3()
            return
        self.value = Vec3(nvx, nvy, nvz)
        self.velocity = Vec3(nvelx, nvely, nvelz)


@always_inline
def wrap_degrees(angle: Float32) -> Float32:
    """Branchless angle wrap into [-180, 180) degrees via floor."""
    if not isfinite(angle):
        return Float32(0.0)
    var a64 = Float64(angle)
    var wrapped = Float32(a64 - Float64(360.0) * floor((a64 + Float64(180.0)) / Float64(360.0)))
    return wrapped if isfinite(wrapped) else Float32(0.0)


struct SpringAngle(Copyable, ImplicitlyCopyable, Movable):
    """A heading in degrees that always turns the short way round."""
    var value: Float32
    var velocity: Float32

    def __init__(out self, value: Float32 = 0):
        self.value = value
        self.velocity = 0

    def update(mut self, goal: Float32, halflife: Float32, dt: Float32):
        if not isfinite(dt) or dt <= Float32(0.0) or not isfinite(goal) or not isfinite(halflife):
            return
        if not isfinite(self.value) or not isfinite(self.velocity):
            self.value = wrap_degrees(goal)
            self.velocity = Float32(0.0)
            return
        var y = halflife_to_damping(halflife) * Float32(0.5)
        var eydt = det_exp(-y * dt)
        if not isfinite(eydt) or eydt == Float32(0.0):
            self.value = wrap_degrees(goal)
            self.velocity = Float32(0.0)
            return
        var e64 = Float64(eydt)
        var y64 = Float64(y)
        var dt64 = Float64(dt)
        var j0 = Float64(wrap_degrees(self.value - goal))
        var j1 = Float64(self.velocity) + j0 * y64
        var next_val = wrap_degrees(Float32(e64 * (j0 + j1 * dt64) + Float64(goal)))
        var next_vel = Float32(e64 * (Float64(self.velocity) - j1 * (y64 * dt64)))
        if not isfinite(next_val) or not isfinite(next_vel):
            self.value = wrap_degrees(goal)
            self.velocity = Float32(0.0)
            return
        self.value = next_val
        self.velocity = next_vel


def decay_spring(mut offset: Vec3, mut velocity: Vec3, halflife: Float32, dt: Float32):
    """Moves a 3D offset and its velocity toward zero, as inertialization needs."""
    if not isfinite(dt) or dt <= Float32(0.0) or not isfinite(halflife):
        return
    if (
        not (isfinite(offset.x) and isfinite(offset.y) and isfinite(offset.z))
        or not (isfinite(velocity.x) and isfinite(velocity.y) and isfinite(velocity.z))
    ):
        offset = Vec3()
        velocity = Vec3()
        return
    var y = halflife_to_damping(halflife) * Float32(0.5)
    var eydt = det_exp(-y * dt)
    if not isfinite(eydt) or eydt == Float32(0.0):
        offset = Vec3()
        velocity = Vec3()
        return
    var e64 = Float64(eydt)
    var y64 = Float64(y)
    var dt64 = Float64(dt)
    var j1x = Float64(velocity.x) + Float64(offset.x) * y64
    var j1y = Float64(velocity.y) + Float64(offset.y) * y64
    var j1z = Float64(velocity.z) + Float64(offset.z) * y64
    var ox = Float32(e64 * (Float64(offset.x) + j1x * dt64))
    var oy = Float32(e64 * (Float64(offset.y) + j1y * dt64))
    var oz = Float32(e64 * (Float64(offset.z) + j1z * dt64))
    var vx = Float32(e64 * (Float64(velocity.x) - j1x * (y64 * dt64)))
    var vy = Float32(e64 * (Float64(velocity.y) - j1y * (y64 * dt64)))
    var vz = Float32(e64 * (Float64(velocity.z) - j1z * (y64 * dt64)))
    if (
        not (isfinite(ox) and isfinite(oy) and isfinite(oz))
        or not (isfinite(vx) and isfinite(vy) and isfinite(vz))
    ):
        offset = Vec3()
        velocity = Vec3()
        return
    offset = Vec3(ox, oy, oz)
    velocity = Vec3(vx, vy, vz)


def decay_spring_scalar(mut offset: Float32, mut velocity: Float32, halflife: Float32, dt: Float32):
    """Moves a scalar offset and its velocity toward zero along a fixed axis."""
    if not isfinite(dt) or dt <= Float32(0.0) or not isfinite(halflife):
        return
    if not isfinite(offset) or not isfinite(velocity):
        offset = Float32(0.0)
        velocity = Float32(0.0)
        return
    var y = halflife_to_damping(halflife) * Float32(0.5)
    var eydt = det_exp(-y * dt)
    if not isfinite(eydt) or eydt == Float32(0.0):
        offset = Float32(0.0)
        velocity = Float32(0.0)
        return
    var e64 = Float64(eydt)
    var y64 = Float64(y)
    var dt64 = Float64(dt)
    var j1 = Float64(velocity) + Float64(offset) * y64
    var next_off = Float32(e64 * (Float64(offset) + j1 * dt64))
    var next_vel = Float32(e64 * (Float64(velocity) - j1 * (y64 * dt64)))
    if not isfinite(next_off) or not isfinite(next_vel):
        offset = Float32(0.0)
        velocity = Float32(0.0)
        return
    offset = next_off
    velocity = next_vel
