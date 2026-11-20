"""Core deterministic vector, quaternion, and 4x4 matrix math for Flint."""
from std.math import sqrt, abs, min, max
from flint.core.detmath import det_sin, det_cos, det_acos, det_atan2


struct Vec2(Copyable, ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32

    def __init__(out self, x: Float32 = 0, y: Float32 = 0):
        self.x = x
        self.y = y

    def __add__(self, other: Self) -> Self:
        return Self(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Self) -> Self:
        return Self(self.x - other.x, self.y - other.y)

    def __mul__(self, value: Float32) -> Self:
        return Self(self.x * value, self.y * value)

    def length_squared(self) -> Float32:
        return self.x * self.x + self.y * self.y

    def length(self) -> Float32:
        return sqrt(self.length_squared())


struct Vec3(Copyable, ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32

    def __init__(out self, x: Float32 = 0, y: Float32 = 0, z: Float32 = 0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: Self) -> Self:
        return Self(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Self) -> Self:
        return Self(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value: Float32) -> Self:
        return Self(self.x * value, self.y * value, self.z * value)

    def __neg__(self) -> Self:
        return Self(-self.x, -self.y, -self.z)

    def length_squared(self) -> Float32:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> Float32:
        return sqrt(self.length_squared())

    def normalized(self) -> Self:
        var size = self.length()
        if size < 0.00001:
            return Self()
        return self * (1.0 / size)

    def horizontal(self) -> Self:
        return Self(self.x, 0, self.z)

    def distance_to(self, other: Self) -> Float32:
        return (other - self).length()

    @always_inline
    def dot(self, other: Self) -> Float32:
        return self.x * other.x + self.y * other.y + self.z * other.z

    @always_inline
    def cross(self, other: Self) -> Self:
        return Self(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


struct Mat4(Copyable, Movable):
    var values: Array[Float32, 16]

    def __init__(out self):
        self.values = Array[Float32, 16](fill=0)
        self.values[0] = 1
        self.values[5] = 1
        self.values[10] = 1
        self.values[15] = 1

    def at(self, row: Int, column: Int) -> Float32:
        return self.values[column * 4 + row]

    def set(mut self, row: Int, column: Int, value: Float32):
        self.values[column * 4 + row] = value

    @always_inline
    def multiply(self, other: Self) -> Self:
        var result = Self()
        var c0 = SIMD[DType.float32, 4](
            self.values[0], self.values[1], self.values[2], self.values[3]
        )
        var c1 = SIMD[DType.float32, 4](
            self.values[4], self.values[5], self.values[6], self.values[7]
        )
        var c2 = SIMD[DType.float32, 4](
            self.values[8], self.values[9], self.values[10], self.values[11]
        )
        var c3 = SIMD[DType.float32, 4](
            self.values[12], self.values[13], self.values[14], self.values[15]
        )
        comptime for column in range(4):
            comptime b = column * 4
            var col_vec = (
                c0 * other.values[b]
                + c1 * other.values[b + 1]
                + c2 * other.values[b + 2]
                + c3 * other.values[b + 3]
            )
            result.values[b] = col_vec[0]
            result.values[b + 1] = col_vec[1]
            result.values[b + 2] = col_vec[2]
            result.values[b + 3] = col_vec[3]
        return result^

    def transform_point(self, value: Vec3) -> Vec3:
        var x = self.at(0, 0) * value.x + self.at(0, 1) * value.y + self.at(0, 2) * value.z + self.at(0, 3)
        var y = self.at(1, 0) * value.x + self.at(1, 1) * value.y + self.at(1, 2) * value.z + self.at(1, 3)
        var z = self.at(2, 0) * value.x + self.at(2, 1) * value.y + self.at(2, 2) * value.z + self.at(2, 3)
        var w = self.at(3, 0) * value.x + self.at(3, 1) * value.y + self.at(3, 2) * value.z + self.at(3, 3)
        if abs(w) > 0.000001 and abs(w - 1.0) > 0.000001:
            return Vec3(x / w, y / w, z / w)
        return Vec3(x, y, z)

    def transform_vector(self, value: Vec3) -> Vec3:
        return Vec3(
            self.at(0, 0) * value.x + self.at(0, 1) * value.y + self.at(0, 2) * value.z,
            self.at(1, 0) * value.x + self.at(1, 1) * value.y + self.at(1, 2) * value.z,
            self.at(2, 0) * value.x + self.at(2, 1) * value.y + self.at(2, 2) * value.z,
        )


struct Quat(Copyable, ImplicitlyCopyable, Movable):
    """Unified nominal quaternion type across core, physics, animation, and rendering.

    All transcendental operations use `flint.core.detmath` so simulation and
    animation code produce bit-identical results across platforms and worker counts.
    """
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32

    def __init__(out self, x: Float32 = 0, y: Float32 = 0, z: Float32 = 0, w: Float32 = 1):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    @staticmethod
    @always_inline
    def identity() -> Self:
        return Self(0, 0, 0, 1)

    @always_inline
    def length_squared(self) -> Float32:
        return self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

    @always_inline
    def length(self) -> Float32:
        return sqrt(self.length_squared())

    def normalized(self) -> Self:
        var size = self.length()
        if size < 0.000001:
            return Self()
        var inv = Float32(1.0) / size
        return Self(self.x * inv, self.y * inv, self.z * inv, self.w * inv)

    @always_inline
    def dot(self, other: Self) -> Float32:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    @always_inline
    def conjugate(self) -> Self:
        return Self(-self.x, -self.y, -self.z, self.w)

    def inverse(self) -> Self:
        var l2 = self.length_squared()
        if l2 < Float32(1.0e-12):
            return Self()
        var inv = Float32(1.0) / l2
        return Self(-self.x * inv, -self.y * inv, -self.z * inv, self.w * inv)

    @always_inline
    def multiply(self, other: Self) -> Self:
        return Self(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

    @always_inline
    def __mul__(self, other: Self) -> Self:
        return self.multiply(other)

    @always_inline
    def shortest(self) -> Self:
        if self.w < 0:
            return Self(-self.x, -self.y, -self.z, -self.w)
        return self

    def rotate(self, value: Vec3) -> Vec3:
        var q = self.normalized()
        var u = Vec3(q.x, q.y, q.z)
        var cross = Vec3(
            u.y * value.z - u.z * value.y,
            u.z * value.x - u.x * value.z,
            u.x * value.y - u.y * value.x,
        )
        return (
            value
            + cross * (Float32(2.0) * q.w)
            + Vec3(
                u.y * cross.z - u.z * cross.y,
                u.z * cross.x - u.x * cross.z,
                u.x * cross.y - u.y * cross.x,
            ) * Float32(2.0)
        )

    def nlerp(self, other: Self, amount: Float32) -> Self:
        var target = other
        if self.dot(other) < 0:
            target = Self(-other.x, -other.y, -other.z, -other.w)
        return Self(
            self.x + (target.x - self.x) * amount,
            self.y + (target.y - self.y) * amount,
            self.z + (target.z - self.z) * amount,
            self.w + (target.w - self.w) * amount,
        ).normalized()

    def slerp(self, other: Self, amount: Float32) -> Self:
        var target = other
        var cosine = self.dot(other)
        if cosine < 0:
            cosine = -cosine
            target = Self(-other.x, -other.y, -other.z, -other.w)
        if cosine > 0.9995:
            return Self(
                self.x + (target.x - self.x) * amount,
                self.y + (target.y - self.y) * amount,
                self.z + (target.z - self.z) * amount,
                self.w + (target.w - self.w) * amount,
            ).normalized()
        var theta = det_acos(max(Float32(-1), min(Float32(1), cosine)))
        var sine = det_sin(theta)
        if abs(sine) < 0.000001:
            return self
        var inv_sine = Float32(1.0) / sine
        var first = det_sin((Float32(1.0) - amount) * theta) * inv_sine
        var second = det_sin(amount * theta) * inv_sine
        return Self(
            self.x * first + target.x * second,
            self.y * first + target.y * second,
            self.z * first + target.z * second,
            self.w * first + target.w * second,
        ).normalized()

    def integrate(mut self, angular_velocity: Vec3, dt: Float32):
        var spin = Self(
            angular_velocity.x * dt * 0.5,
            angular_velocity.y * dt * 0.5,
            angular_velocity.z * dt * 0.5,
            0,
        )
        var q = self
        self.x = q.x + (spin.w * q.x + spin.x * q.w + spin.y * q.z - spin.z * q.y)
        self.y = q.y + (spin.w * q.y - spin.x * q.z + spin.y * q.w + spin.z * q.x)
        self.z = q.z + (spin.w * q.z + spin.x * q.y - spin.y * q.x + spin.z * q.w)
        self.w = q.w + (spin.w * q.w - spin.x * q.x - spin.y * q.y - spin.z * q.z)
        self = self.normalized()

    def axis_angle(self) -> Tuple[Vec3, Float32]:
        var q = self.normalized()
        var angle = 2.0 * det_acos(max(Float32(-1), min(Float32(1), q.w)))
        var scale = sqrt(max(Float32(0), 1.0 - q.w * q.w))
        if scale < 0.0001:
            return (Vec3(0, 1, 0), Float32(0))
        return (Vec3(q.x / scale, q.y / scale, q.z / scale), angle * 57.2957795)

    @staticmethod
    def from_axis_angle(axis: Vec3, angle_rad: Float32) -> Self:
        var l2 = axis.length_squared()
        if l2 < Float32(1.0e-12):
            return Self()
        var inv_l = Float32(1.0) / sqrt(l2)
        var half = angle_rad * Float32(0.5)
        var s = det_sin(half) * inv_l
        return Self(axis.x * s, axis.y * s, axis.z * s, det_cos(half)).normalized()

    @staticmethod
    def from_euler(pitch: Float32, yaw: Float32, roll: Float32) -> Self:
        """Intrinsic YXZ (yaw about Y, pitch about X, roll about Z) in radians."""
        var hp = pitch * Float32(0.5)
        var hy = yaw * Float32(0.5)
        var hr = roll * Float32(0.5)
        var sp = det_sin(hp); var cp = det_cos(hp)
        var sy = det_sin(hy); var cy = det_cos(hy)
        var sr = det_sin(hr); var cr = det_cos(hr)
        return Self(
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * cp * cr + sy * sp * sr,
        ).normalized()

    def to_scaled_angle(self) -> Vec3:
        """Rotation axis times angle in radians (the quaternion log, doubled)."""
        var q = self.normalized().shortest()
        var vec_len = sqrt(q.x * q.x + q.y * q.y + q.z * q.z)
        if vec_len < Float32(0.00001):
            return Vec3(q.x * Float32(2.0), q.y * Float32(2.0), q.z * Float32(2.0))
        var half = det_atan2(vec_len, max(Float32(0.0), q.w))
        var scale = Float32(2.0) * half / vec_len
        return Vec3(q.x * scale, q.y * scale, q.z * scale)

    @staticmethod
    def from_scaled_angle(value: Vec3) -> Self:
        var angle = value.length()
        if angle < Float32(0.00001):
            return Self(
                value.x * Float32(0.5),
                value.y * Float32(0.5),
                value.z * Float32(0.5),
                Float32(1.0),
            ).normalized()
        var half = angle * Float32(0.5)
        var scale = det_sin(half) / angle
        return Self(value.x * scale, value.y * scale, value.z * scale, det_cos(half)).normalized()

    def to_mat4(self) -> Mat4:
        var result = Mat4()
        var q = self.normalized()
        var xx = q.x * q.x
        var yy = q.y * q.y
        var zz = q.z * q.z
        var xy = q.x * q.y
        var xz = q.x * q.z
        var yz = q.y * q.z
        var wx = q.w * q.x
        var wy = q.w * q.y
        var wz = q.w * q.z
        result.set(0, 0, Float32(1.0) - Float32(2.0) * (yy + zz))
        result.set(0, 1, Float32(2.0) * (xy - wz))
        result.set(0, 2, Float32(2.0) * (xz + wy))
        result.set(1, 0, Float32(2.0) * (xy + wz))
        result.set(1, 1, Float32(1.0) - Float32(2.0) * (xx + zz))
        result.set(1, 2, Float32(2.0) * (yz - wx))
        result.set(2, 0, Float32(2.0) * (xz - wy))
        result.set(2, 1, Float32(2.0) * (yz + wx))
        result.set(2, 2, Float32(1.0) - Float32(2.0) * (xx + yy))
        return result^


struct CameraState(Copyable, ImplicitlyCopyable, Movable):
    var yaw: Float32
    var pitch: Float32
    var distance: Float32
    var shake: Float32

    def __init__(out self):
        self.yaw = 0.0
        self.pitch = -0.42
        self.distance = 10.0
        self.shake = 0.0

    def forward(self) -> Vec3:
        return Vec3(det_sin(self.yaw), 0, det_cos(self.yaw))

    def right(self) -> Vec3:
        return Vec3(det_cos(self.yaw), 0, -det_sin(self.yaw))
