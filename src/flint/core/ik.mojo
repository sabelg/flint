"""Inverse kinematics: analytic two-bone IK and FABRIK chains.

Two-bone IK follows Daniel Holden, "Simple Two Joint IK": the triangle
formed by the hip, knee and target is solved with the law of cosines, and
the result is applied as small rotations in each joint's local frame, so
the animated pose is kept wherever the target does not demand a change.

FABRIK follows Aristidou and Lasenby, "FABRIK: A fast, iterative solver for
the Inverse Kinematics problem" (Graphical Models, 2011): alternate
backward and forward passes along the chain, each one restoring bone
lengths, until the end is close enough to the target.
"""
from std.math import sqrt, sin, cos, acos
from flint.core.math import Vec3
from flint.core.transform import Quat


def dot(a: Vec3, b: Vec3) -> Float32:
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def clamp1(value: Float32) -> Float32:
    return max(Float32(-1), min(Float32(1), value))


def safe_normalized(value: Vec3, fallback: Vec3) -> Vec3:
    var length = value.length()
    if length < 0.000001:
        return fallback
    return value * (1.0 / length)


def angle_axis(angle: Float32, axis: Vec3) -> Quat:
    var half = angle * 0.5
    var s = sin(half)
    return Quat(axis.x * s, axis.y * s, axis.z * s, cos(half))


struct TwoBoneResult(Copyable, ImplicitlyCopyable, Movable):
    """Rotations to multiply onto the upper and middle joints' local rotations."""
    var upper: Quat
    var middle: Quat

    def __init__(out self, upper: Quat, middle: Quat):
        self.upper = upper
        self.middle = middle


def two_bone_ik(
    a: Vec3, b: Vec3, c: Vec3, target: Vec3,
    a_global: Quat, b_global: Quat, pole: Vec3,
) -> TwoBoneResult:
    """a, b, c: upper, middle and end joint positions. a_global, b_global:
    the upper and middle joints' global rotations. pole: the direction the
    middle joint should bend toward when the chain is straight."""
    var eps = Float32(0.001)
    var lab = (b - a).length()
    var lcb = (b - c).length()
    if lab < eps or lcb < eps:
        return TwoBoneResult(Quat(), Quat())
    var lat = max(eps, min((target - a).length(), lab + lcb - eps))
    var ac = safe_normalized(c - a, Vec3(0, -1, 0))
    var ab = safe_normalized(b - a, Vec3(0, -1, 0))
    var ba = -ab
    var bc = safe_normalized(c - b, Vec3(0, -1, 0))
    var at = safe_normalized(target - a, ac)
    var ac_ab_0 = acos(clamp1(dot(ac, ab)))
    var ba_bc_0 = acos(clamp1(dot(ba, bc)))
    var ac_at_0 = acos(clamp1(dot(ac, at)))
    var ac_ab_1 = acos(clamp1((lcb * lcb - lab * lab - lat * lat) / (-2.0 * lab * lat)))
    var ba_bc_1 = acos(clamp1((lat * lat - lab * lab - lcb * lcb) / (-2.0 * lab * lcb)))
    var bend = cross(ac, ab)
    if bend.length() < 0.0001:
        bend = cross(ac, pole)  # straight chain: bend toward the pole
    _ = ac_at_0
    var axis0 = safe_normalized(bend, Vec3(1, 0, 0))
    var a_inverse = a_global.conjugate()
    var b_inverse = b_global.conjugate()
    var r0 = angle_axis(ac_ab_1 - ac_ab_0, a_inverse.rotate(axis0))
    var r1 = angle_axis(ba_bc_1 - ba_bc_0, b_inverse.rotate(axis0))
    # Bend only: the chain now spans the target distance. aim_rotation then
    # swings it onto the target, measured on the bent pose, which is exact
    # (computing that swing from the unbent pose leaves a few cm of error).
    return TwoBoneResult(r0.normalized(), r1.normalized())


def aim_rotation(a: Vec3, c: Vec3, target: Vec3, a_global: Quat) -> Quat:
    """Local rotation for joint a that turns direction a->c onto a->target."""
    var from_direction = safe_normalized(c - a, Vec3(0, -1, 0))
    var to_direction = safe_normalized(target - a, from_direction)
    var axis = cross(from_direction, to_direction)
    var angle = acos(clamp1(dot(from_direction, to_direction)))
    if axis.length() < 0.000001 or angle < 0.000001:
        return Quat()
    return angle_axis(angle, a_global.conjugate().rotate(safe_normalized(axis, Vec3(1, 0, 0))))


def fabrik(mut points: List[Vec3], target: Vec3, iterations: Int = 12, tolerance: Float32 = 0.001) -> Bool:
    """Moves a joint chain (points[0] is fixed) so its end reaches target.
    Returns True when the end lands within tolerance."""
    var count = len(points)
    if count < 2:
        return False
    var lengths = List[Float32]()
    var total = Float32(0)
    for index in range(count - 1):
        var length = (points[index + 1] - points[index]).length()
        lengths.append(length)
        total += length
    var root = points[0]
    if (target - root).length() >= total:
        # Out of reach: straighten toward the target.
        var direction = safe_normalized(target - root, Vec3(0, 1, 0))
        for index in range(1, count):
            points[index] = points[index - 1] + direction * lengths[index - 1]
        return False
    for _ in range(iterations):
        if (points[count - 1] - target).length() <= tolerance:
            return True
        points[count - 1] = target
        for index in range(count - 2, -1, -1):
            var direction = safe_normalized(points[index] - points[index + 1], Vec3(0, 1, 0))
            points[index] = points[index + 1] + direction * lengths[index]
        points[0] = root
        for index in range(1, count):
            var direction = safe_normalized(points[index] - points[index - 1], Vec3(0, 1, 0))
            points[index] = points[index - 1] + direction * lengths[index - 1]
    return (points[count - 1] - target).length() <= tolerance
