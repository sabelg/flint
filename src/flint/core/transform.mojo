"""Native quaternion and 4x4 transform math for glTF and rasterization."""
from std.math import sqrt, sin, cos, acos
from flint.core.math import Vec3


struct Quat(Copyable, ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32

    def __init__(out self, x: Float32 = 0, y: Float32 = 0, z: Float32 = 0, w: Float32 = 1):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def normalized(self) -> Self:
        var length = sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)
        if length < 0.000001:
            return Self()
        return Self(self.x / length, self.y / length, self.z / length, self.w / length)

    def slerp(self, other: Self, amount: Float32) -> Self:
        var target = other
        var cosine = self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
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
        var theta = acos(max(Float32(-1), min(Float32(1), cosine)))
        var sine = sin(theta)
        if abs(sine) < 0.000001:
            return self
        var first = sin((1.0 - amount) * theta) / sine
        var second = sin(amount * theta) / sine
        return Self(self.x * first + target.x * second, self.y * first + target.y * second, self.z * first + target.z * second, self.w * first + target.w * second).normalized()

    def multiply(self, other: Self) -> Self:
        return Self(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

    def conjugate(self) -> Self:
        return Self(-self.x, -self.y, -self.z, self.w)

    def shortest(self) -> Self:
        if self.w < 0:
            return Self(-self.x, -self.y, -self.z, -self.w)
        return self

    def to_scaled_angle(self) -> Vec3:
        """Rotation axis times angle in radians (the quaternion log, doubled)."""
        var q = self.shortest()
        var length = sqrt(q.x * q.x + q.y * q.y + q.z * q.z)
        if length < 0.00001:
            return Vec3(q.x * 2.0, q.y * 2.0, q.z * 2.0)
        var half = acos(max(Float32(-1), min(Float32(1), q.w)))
        var scale = 2.0 * half / length
        return Vec3(q.x * scale, q.y * scale, q.z * scale)

    @staticmethod
    def from_scaled_angle(value: Vec3) -> Self:
        var angle = value.length()
        if angle < 0.00001:
            return Self(value.x * 0.5, value.y * 0.5, value.z * 0.5, 1.0).normalized()
        var half = angle * 0.5
        var scale = sin(half) / angle
        return Self(value.x * scale, value.y * scale, value.z * scale, cos(half))

    def rotate(self, value: Vec3) -> Vec3:
        var q = self.normalized()
        var u = Vec3(q.x, q.y, q.z)
        var cross = Vec3(u.y * value.z - u.z * value.y, u.z * value.x - u.x * value.z, u.x * value.y - u.y * value.x)
        return value + cross * (2.0 * q.w) + Vec3(u.y * cross.z - u.z * cross.y, u.z * cross.x - u.x * cross.z, u.x * cross.y - u.y * cross.x) * 2.0


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

    def multiply(self, other: Self) -> Self:
        var result = Self()
        for column in range(4):
            for row in range(4):
                var value = Float32(0)
                for index in range(4):
                    value += self.at(row, index) * other.at(index, column)
                result.set(row, column, value)
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
    var angle = angle_degrees * 0.01745329252
    return trs(Vec3(), Quat(0, sin(angle * 0.5), 0, cos(angle * 0.5)), Vec3(1, 1, 1))
