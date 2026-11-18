"""Native quaternion and 4x4 transform math for glTF, physics, and rasterization.

`Quat` and `Mat4` are unified nominal aliases to `flint.core.math.Quat` and
`flint.core.math.Mat4` so every subsystem shares a single deterministic type.
"""
from flint.core.detmath import det_sin, det_cos
from flint.core.math import Vec3, Quat as MathQuat, Mat4 as MathMat4

comptime Quat = MathQuat
comptime Mat4 = MathMat4


def trs(translation: Vec3, rotation: Quat, scale: Vec3) -> Mat4:
    var result = Mat4()
    var q = rotation.normalized()
    var xx = q.x * q.x
    var yy = q.y * q.y
    var zz = q.z * q.z
    var xy = q.x * q.y
    var xz = q.x * q.z
    var yz = q.y * q.z
    var wx = q.w * q.x
    var wy = q.w * q.y
    var wz = q.w * q.z
    result.set(0, 0, (1.0 - 2.0 * (yy + zz)) * scale.x)
    result.set(0, 1, (2.0 * (xy - wz)) * scale.y)
    result.set(0, 2, (2.0 * (xz + wy)) * scale.z)
    result.set(1, 0, (2.0 * (xy + wz)) * scale.x)
    result.set(1, 1, (1.0 - 2.0 * (xx + zz)) * scale.y)
    result.set(1, 2, (2.0 * (yz - wx)) * scale.z)
    result.set(2, 0, (2.0 * (xz - wy)) * scale.x)
    result.set(2, 1, (2.0 * (yz + wx)) * scale.y)
    result.set(2, 2, (1.0 - 2.0 * (xx + yy)) * scale.z)
    result.set(0, 3, translation.x)
    result.set(1, 3, translation.y)
    result.set(2, 3, translation.z)
    return result^


def rotation_yaw(angle_degrees: Float32) -> Mat4:
    var angle = angle_degrees * Float32(0.01745329252)
    return trs(Vec3(), Quat(0, det_sin(angle * Float32(0.5)), 0, det_cos(angle * Float32(0.5))), Vec3(1, 1, 1))
