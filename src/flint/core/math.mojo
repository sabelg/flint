from std.math import sqrt
from flint.core.detmath import det_sin, det_cos, det_acos


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
        var size = sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)
        if size < 0.000001:
            return Self()
        return Self(self.x / size, self.y / size, self.z / size, self.w / size)

    def integrate(mut self, angular_velocity: Vec3, dt: Float32):
        var spin = Self(angular_velocity.x * dt * 0.5, angular_velocity.y * dt * 0.5, angular_velocity.z * dt * 0.5, 0)
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
