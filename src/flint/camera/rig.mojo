"""Cinemachine-Style Camera Rig (`flint.camera.rig`, v5).

Implements `CameraRig` across 4 modes:
- `CAM_MODE_FIRST_PERSON`: eye-attached yaw/pitch with clamped vertical look.
- `CAM_MODE_ORBIT_FOLLOW`: `SpringVec3` smoothed pivot follow + `PhysicsWorld.shapecast_sphere`
  `SpringArm` anti-clipping (instant/fast snap-in against walls, smooth critically-damped spring-out).
- `CAM_MODE_RAIL_DOLLY`: piecewise / Catmull-Rom spline dolly tracking `target_pos` with smoothed look-at.
- `CAM_MODE_FREE_FLY`: 6-DOF spectator/editor fly camera.
Plus Squirrel Eiserloh deterministic `trauma * trauma` 6-DOF gradient-noise camera shake.
"""
from std.collections import List
from std.math import sqrt, abs, min, max, floor
from flint.core.math import Vec3, Quat, Mat4
from flint.core.transform import trs
from flint.core.detmath import det_sin, det_cos, det_atan2, PI
from flint.core.detsort import mix64
from flint.core.springs import SpringFloat, SpringVec3
from flint.physics.filter import CollisionFilter, CAT_TERRAIN, CAT_STATIC
from flint.physics.world import PhysicsWorld
from flint.camera.physical import PhysicalCameraModel  # sota:camera hook

comptime CAM_MODE_FIRST_PERSON: UInt8 = 0
comptime CAM_MODE_ORBIT_FOLLOW: UInt8 = 1
comptime CAM_MODE_RAIL_DOLLY: UInt8 = 2
comptime CAM_MODE_FREE_FLY: UInt8 = 3

comptime _CAM_DEG_TO_RAD: Float32 = 0.01745329252


@always_inline
def _det_grad_1d(hash_seed: UInt64, i: Int64) -> Float32:
    var h = mix64(hash_seed ^ UInt64(i * 0x9E3779B97F4A7C15))
    var u = Float32(Int(h & 0xFFFF)) * (Float32(2.0) / Float32(65535.0)) - Float32(1.0)
    return u


@always_inline
def _det_perlin_1d(seed: UInt64, x: Float32) -> Float32:
    """Deterministic C2-continuous 1D quintic gradient/value noise in `[-1, 1]`."""
    var xi = floor(x)
    var i0 = Int64(xi)
    var t = x - xi
    var fade = t * t * t * (t * (t * Float32(6.0) - Float32(15.0)) + Float32(10.0))
    var v0 = _det_grad_1d(seed, i0)
    var v1 = _det_grad_1d(seed, i0 + 1)
    return v0 + (v1 - v0) * fade


def _look_rotation(forward: Vec3, world_up: Vec3 = Vec3(0.0, 1.0, 0.0)) -> Quat:
    """Constructs world camera orientation `Quat` whose `+Z` (or `-Z` view) aligns cleanly:
    convention in Flint (`core.math` / `anim.procedural`): `+X` right, `+Y` up, `+Z` forward."""
    var fwd = forward.normalized()
    if fwd.length_squared() < Float32(0.5):
        fwd = Vec3(0.0, 0.0, 1.0)
    var right = world_up.cross(fwd)
    if right.length_squared() < Float32(1.0e-6):
        right = Vec3(1.0, 0.0, 0.0).cross(fwd)
    right = right.normalized()
    var up = fwd.cross(right).normalized()

    var trace = right.x + up.y + fwd.z
    if trace > Float32(0.0):
        var s = sqrt(trace + Float32(1.0)) * Float32(2.0)
        return Quat((up.z - fwd.y) / s, (fwd.x - right.z) / s, (right.y - up.x) / s, Float32(0.25) * s).normalized()
    if right.x > up.y and right.x > fwd.z:
        var s = sqrt(max(Float32(1.0e-6), Float32(1.0) + right.x - up.y - fwd.z)) * Float32(2.0)
        return Quat(Float32(0.25) * s, (up.x + right.y) / s, (fwd.x + right.z) / s, (up.z - fwd.y) / s).normalized()
    if up.y > fwd.z:
        var s = sqrt(max(Float32(1.0e-6), Float32(1.0) + up.y - right.x - fwd.z)) * Float32(2.0)
        return Quat((up.x + right.y) / s, Float32(0.25) * s, (fwd.y + up.z) / s, (fwd.x - right.z) / s).normalized()
    var s = sqrt(max(Float32(1.0e-6), Float32(1.0) + fwd.z - right.x - up.y)) * Float32(2.0)
    return Quat((fwd.x + right.z) / s, (fwd.y + up.z) / s, Float32(0.25) * s, (right.y - up.x) / s).normalized()


@fieldwise_init
struct SpringArm(Copyable, ImplicitlyCopyable, Movable):
    """Sphere-cast anti-clipping camera boom with instant/fast snap-in & smooth spring-out."""
    var target_length: Float32
    var min_length: Float32
    var probe_radius: Float32
    var wall_margin: Float32
    var snap_in_halflife: Float32
    var spring_out_halflife: Float32
    var current_length: SpringFloat
    var is_occluded: Bool
    var hit_body_id: Int
    var collision_filter: CollisionFilter

    def __init__(
        out self,
        target_length: Float32 = 4.5,
        probe_radius: Float32 = 0.22,
        min_length: Float32 = 0.35,
    ):
        self.target_length = max(Float32(0.1), target_length)
        self.min_length = max(Float32(0.05), min_length)
        self.probe_radius = max(Float32(0.02), probe_radius)
        self.wall_margin = Float32(0.05)
        self.snap_in_halflife = Float32(0.01)
        self.spring_out_halflife = Float32(0.18)
        self.current_length = SpringFloat(target_length)
        self.is_occluded = False
        self.hit_body_id = -1
        self.collision_filter = CollisionFilter(CAT_STATIC, CAT_TERRAIN | CAT_STATIC, 0)

    def resolve(
        mut self,
        world: PhysicsWorld,
        pivot: Vec3,
        arm_dir_unit: Vec3,
        dt: Float32,
        ignore_body_id: Int = -1,
    ) -> Vec3:
        var desired = self.target_length
        var sh = world.shapecast_sphere(
            pivot,
            self.probe_radius,
            arm_dir_unit * desired,
            self.collision_filter,
            ignore_body_id,
        )
        var max_unobstructed = desired
        if sh.hit and sh.distance < desired:
            self.is_occluded = True
            self.hit_body_id = sh.body_id
            max_unobstructed = max(self.min_length, sh.distance - self.wall_margin)
        else:
            self.is_occluded = False
            self.hit_body_id = -1

        if max_unobstructed < self.current_length.value:
            # Fast snap-in: clamp strictly <= max_unobstructed so camera NEVER penetrates walls!
            if self.snap_in_halflife <= Float32(0.02):
                self.current_length.value = max_unobstructed
                self.current_length.velocity = Float32(0.0)
            else:
                self.current_length.update(max_unobstructed, self.snap_in_halflife, dt)
                self.current_length.value = min(self.current_length.value, max_unobstructed)
        else:
            # Smooth critically-damped spring-out back toward target boom length
            self.current_length.update(max_unobstructed, self.spring_out_halflife, dt)

        return pivot + arm_dir_unit * self.current_length.value


struct CameraRig(Copyable, Movable):
    var mode: UInt8
    var yaw_deg: Float32
    var pitch_deg: Float32
    var min_pitch_deg: Float32
    var max_pitch_deg: Float32
    var fov_deg: Float32

    var look_offset: Vec3
    var shoulder_offset: Vec3
    var first_person_eye_offset: Vec3
    var follow_halflife: Float32
    var pivot_spring: SpringVec3
    var pivot_initialized: Bool
    var spring_arm: SpringArm
    var ignore_body_id: Int

    # Rail Dolly spline state
    var rail_points: List[Vec3]
    var rail_auto_track: Bool
    var rail_param: SpringFloat
    var rail_halflife: Float32

    # Free-fly state
    var fly_pos: Vec3
    var fly_vel: Vec3

    # Deterministic 6-DOF trauma^2 camera shake
    var trauma: Float32
    var trauma_decay: Float32
    var shake_freq_hz: Float32
    var max_shake_offset: Vec3
    var max_shake_deg: Vec3
    var shake_time: Float32
    var shake_seed: UInt64

    # Final solved world camera pose
    var world_pos: Vec3
    var world_rot: Quat
    var unshaken_pos: Vec3
    var unshaken_rot: Quat

    # --- sota:camera hook (physical lens) ---
    var lens: PhysicalCameraModel  # focal length, f-stop, shutter, ISO, sensor, focus (renderer: ren.sota_cam.camera)
    # --- end sota:camera hook ---

    def __init__(out self, mode: UInt8 = CAM_MODE_ORBIT_FOLLOW):
        self.mode = mode
        self.yaw_deg = 0.0
        self.pitch_deg = 15.0
        self.min_pitch_deg = -85.0
        self.max_pitch_deg = 85.0
        self.fov_deg = 65.0
        self.look_offset = Vec3(0.0, 1.5, 0.0)
        self.shoulder_offset = Vec3(0.0, 0.0, 0.0)
        self.first_person_eye_offset = Vec3(0.0, 1.65, 0.0)
        self.follow_halflife = 0.08
        self.pivot_spring = SpringVec3()
        self.pivot_initialized = False
        self.spring_arm = SpringArm(4.5, 0.22, 0.35)
        self.ignore_body_id = -1

        self.rail_points = List[Vec3]()
        self.rail_auto_track = True
        self.rail_param = SpringFloat(0.0)
        self.rail_halflife = 0.15

        self.fly_pos = Vec3(0.0, 2.0, -5.0)
        self.fly_vel = Vec3()

        self.trauma = 0.0
        self.trauma_decay = 1.2
        self.shake_freq_hz = 18.0
        self.max_shake_offset = Vec3(0.18, 0.18, 0.10)
        self.max_shake_deg = Vec3(4.5, 4.5, 3.0)
        self.shake_time = 0.0
        self.shake_seed = 0x517CC1B727220A95

        self.world_pos = Vec3(0.0, 2.0, -4.5)
        self.world_rot = Quat()
        self.unshaken_pos = self.world_pos
        self.unshaken_rot = Quat()
        # --- sota:camera hook (physical lens) ---
        self.lens = PhysicalCameraModel()
        # --- end sota:camera hook ---

    # --- sota:camera hook (physical lens) ---
    def use_lens_fov(mut self):
        """Vertical FOV from the lens' focal length and sensor height."""
        self.fov_deg = self.lens.fov_y_deg()

    def focus_on(mut self, world_point: Vec3):
        """Pull focus to a world point (thin-lens focus distance along the optical axis)."""
        self.lens.focus_distance_m = max(Float32(0.05), (world_point - self.world_pos).dot(self.forward()))
    # --- end sota:camera hook ---

    def add_trauma(mut self, amount: Float32):
        self.trauma = max(Float32(0.0), min(Float32(1.0), self.trauma + amount))

    def rotate_orbit(mut self, delta_yaw_deg: Float32, delta_pitch_deg: Float32):
        self.yaw_deg += delta_yaw_deg
        self.yaw_deg -= Float32(360.0) * floor((self.yaw_deg + Float32(180.0)) * Float32(1.0 / 360.0))
        self.pitch_deg = max(self.min_pitch_deg, min(self.max_pitch_deg, self.pitch_deg + delta_pitch_deg))

    def set_rail_waypoints(mut self, pts: List[Vec3]):
        self.rail_points = pts.copy()

    def _eval_catmull_rom_rail(self, t_norm: Float32) -> Vec3:
        var n = len(self.rail_points)
        if n == 0:
            return Vec3()
        if n == 1:
            return self.rail_points[0]
        var t_clamped = max(Float32(0.0), min(Float32(1.0), t_norm))
        var scaled = t_clamped * Float32(n - 1)
        var seg = min(n - 2, Int(floor(scaled)))
        var u = scaled - Float32(seg)
        var p0 = self.rail_points[max(0, seg - 1)]
        var p1 = self.rail_points[seg]
        var p2 = self.rail_points[seg + 1]
        var p3 = self.rail_points[min(n - 1, seg + 2)]
        var u2 = u * u
        var u3 = u2 * u
        return (
            p1 * Float32(2.0)
            + (p2 - p0) * u
            + (p0 * Float32(2.0) - p1 * Float32(5.0) + p2 * Float32(4.0) - p3) * u2
            + (-p0 + p1 * Float32(3.0) - p2 * Float32(3.0) + p3) * u3
        ) * Float32(0.5)

    def _closest_rail_param(self, target: Vec3) -> Float32:
        var n = len(self.rail_points)
        if n < 2:
            return Float32(0.0)
        var best_t = Float32(0.0)
        var best_d2 = Float32(1.0e30)
        var inv_segs = Float32(1.0) / Float32(n - 1)
        for i in range(n - 1):
            var a = self.rail_points[i]
            var b = self.rail_points[i + 1]
            var ab = b - a
            var l2 = ab.length_squared()
            var u = max(Float32(0.0), min(Float32(1.0), (target - a).dot(ab) / l2)) if l2 > Float32(1.0e-8) else Float32(0.0)
            var d2 = (target - (a + ab * u)).length_squared()
            if d2 < best_d2:
                best_d2 = d2
                best_t = (Float32(i) + u) * inv_segs
        return best_t

    def update(
        mut self,
        world: PhysicsWorld,
        target_pos: Vec3,
        dt: Float32,
        fly_move_local: Vec3 = Vec3(),
        fly_speed: Float32 = 10.0,
    ):
        self.pitch_deg = max(self.min_pitch_deg, min(self.max_pitch_deg, self.pitch_deg))
        var yaw_r = self.yaw_deg * _CAM_DEG_TO_RAD
        var pitch_r = self.pitch_deg * _CAM_DEG_TO_RAD

        # Aim forward vector from yaw (+Z rotated toward +X about +Y) and pitch (positive pitch looks down from elevated orbit or up in FP)
        var cy = det_cos(yaw_r)
        var sy = det_sin(yaw_r)
        var cp = det_cos(pitch_r)
        var sp = det_sin(pitch_r)

        var base_pos: Vec3
        var base_rot: Quat

        if self.mode == CAM_MODE_FIRST_PERSON:
            var fp_fwd = Vec3(sy * cp, -sp, cy * cp).normalized()
            base_pos = target_pos + self.first_person_eye_offset
            base_rot = _look_rotation(fp_fwd)
        elif self.mode == CAM_MODE_FREE_FLY:
            var fly_fwd = Vec3(sy * cp, -sp, cy * cp).normalized()
            base_rot = _look_rotation(fly_fwd)
            var world_move = base_rot.rotate(fly_move_local) * fly_speed
            self.fly_vel = world_move
            self.fly_pos = self.fly_pos + self.fly_vel * max(Float32(0.0), dt)
            base_pos = self.fly_pos
        elif self.mode == CAM_MODE_RAIL_DOLLY:
            var look_tgt = target_pos + self.look_offset
            if not self.pivot_initialized:
                self.pivot_spring.value = look_tgt
                self.pivot_initialized = True
            else:
                self.pivot_spring.update(look_tgt, self.follow_halflife, dt)
            if self.rail_auto_track:
                var goal_t = self._closest_rail_param(target_pos)
                self.rail_param.update(goal_t, self.rail_halflife, dt)
            base_pos = self._eval_catmull_rom_rail(self.rail_param.value)
            base_rot = _look_rotation(self.pivot_spring.value - base_pos)
        else:
            # CAM_MODE_ORBIT_FOLLOW
            var goal_pivot = target_pos + self.look_offset
            if not self.pivot_initialized:
                self.pivot_spring.value = goal_pivot
                self.pivot_spring.velocity = Vec3()
                self.pivot_initialized = True
            else:
                self.pivot_spring.update(goal_pivot, self.follow_halflife, dt)

            var fwd = Vec3(sy * cp, -sp, cy * cp).normalized()
            base_rot = _look_rotation(fwd)
            var pivot = self.pivot_spring.value + base_rot.rotate(self.shoulder_offset)
            var arm_dir = -fwd
            base_pos = self.spring_arm.resolve(world, pivot, arm_dir, dt, self.ignore_body_id)

        self.unshaken_pos = base_pos
        self.unshaken_rot = base_rot

        # Apply deterministic `trauma * trauma` 6-DOF Perlin camera shake
        if self.trauma > Float32(1.0e-5) and dt >= Float32(0.0):
            self.shake_time += dt
            var shake = self.trauma * self.trauma
            var phase = self.shake_time * self.shake_freq_hz
            var ox = _det_perlin_1d(self.shake_seed ^ 0x1111, phase) * self.max_shake_offset.x * shake
            var oy = _det_perlin_1d(self.shake_seed ^ 0x2222, phase + Float32(13.7)) * self.max_shake_offset.y * shake
            var oz = _det_perlin_1d(self.shake_seed ^ 0x3333, phase + Float32(29.3)) * self.max_shake_offset.z * shake

            var ap = _det_perlin_1d(self.shake_seed ^ 0x4444, phase + Float32(47.1)) * self.max_shake_deg.x * _CAM_DEG_TO_RAD * shake
            var ay = _det_perlin_1d(self.shake_seed ^ 0x5555, phase + Float32(61.9)) * self.max_shake_deg.y * _CAM_DEG_TO_RAD * shake
            var ar = _det_perlin_1d(self.shake_seed ^ 0x6666, phase + Float32(83.5)) * self.max_shake_deg.z * _CAM_DEG_TO_RAD * shake

            var q_shake = Quat.from_euler(ap, ay, ar)
            self.world_pos = base_pos + base_rot.rotate(Vec3(ox, oy, oz))
            self.world_rot = (base_rot * q_shake).normalized()
            self.trauma = max(Float32(0.0), self.trauma - self.trauma_decay * dt)
        else:
            self.world_pos = base_pos
            self.world_rot = base_rot

    def forward(self) -> Vec3:
        return self.world_rot.rotate(Vec3(0.0, 0.0, 1.0)).normalized()

    def right(self) -> Vec3:
        return self.world_rot.rotate(Vec3(1.0, 0.0, 0.0)).normalized()

    def up(self) -> Vec3:
        return self.world_rot.rotate(Vec3(0.0, 1.0, 0.0)).normalized()

    def world_matrix(self) -> Mat4:
        return trs(self.world_pos, self.world_rot, Vec3(1.0, 1.0, 1.0))

    def view_matrix(self) -> Mat4:
        """Right-handed / Flint view matrix transforming world points into camera space."""
        var r = self.right()
        var u = self.up()
        var f = self.forward()
        var m = Mat4()
        m.set(0, 0, r.x); m.set(0, 1, r.y); m.set(0, 2, r.z); m.set(0, 3, -r.dot(self.world_pos))
        m.set(1, 0, u.x); m.set(1, 1, u.y); m.set(1, 2, u.z); m.set(1, 3, -u.dot(self.world_pos))
        m.set(2, 0, f.x); m.set(2, 1, f.y); m.set(2, 2, f.z); m.set(2, 3, -f.dot(self.world_pos))
        return m^
