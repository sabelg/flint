"""Virtual Cameras, Priority Blending, Lock-On Framing & Event-Driven Shake (`flint.camera.vcam`, v5).

Game-facing camera layer usable without Studio:
- `VirtualCamera`: a shot with `priority`, `enabled`, `blend_in_time` and one rig mode:
  `VCAM_FOLLOW` (orbit-follow whose yaw lazily tracks `follow_yaw_deg`, spring-arm occluders),
  `VCAM_ORBIT` (input-driven orbit via `rig.rotate_orbit`), `VCAM_FIXED` (static position, optional
  look-at), `VCAM_LOCK_ON` (frames `follow_target` + `lock_target` from behind the player with
  spring-arm occlusion) and `VCAM_FIRST_PERSON`. Orbit/follow modes reuse `CameraRig` + `SpringArm`.
- `CameraBrain`: picks the highest-priority enabled vcam (latest activation wins ties), blends
  position/rotation/FOV from the previous output with smoothstep easing over the new vcam's
  `blend_in_time`, and applies deterministic `trauma^2` Perlin shake fed by `add_trauma` or
  positional `shake_event(pos, strength, radius)` calls.
"""
from std.collections import List
from std.math import sqrt, min, max, isfinite, floor
from flint.core.math import Vec3, Quat, Mat4
from flint.core.transform import trs
from flint.core.detmath import det_sin, det_cos, det_atan2
from flint.physics.world import PhysicsWorld
from flint.camera.rig import (
    CameraRig,
    SpringArm,
    CAM_MODE_FIRST_PERSON,
    CAM_MODE_ORBIT_FOLLOW,
    _look_rotation,
    _det_perlin_1d,
)

comptime VCAM_FOLLOW: UInt8 = 0
comptime VCAM_ORBIT: UInt8 = 1
comptime VCAM_FIXED: UInt8 = 2
comptime VCAM_LOCK_ON: UInt8 = 3
comptime VCAM_FIRST_PERSON: UInt8 = 4

comptime _VC_DEG_TO_RAD: Float32 = 0.01745329252


struct VirtualCamera(Copyable, Movable):
    var id: Int
    var priority: Int
    var mode: UInt8
    var enabled: Bool
    var blend_in_time: Float32
    var fov_deg: Float32
    var rig: CameraRig
    var follow_target: Vec3
    var follow_yaw_deg: Float32  # target heading for VCAM_FOLLOW
    var yaw_follow_rate: Float32  # 1/s exponential yaw catch-up
    var lock_target: Vec3
    var has_lock_target: Bool
    var lock_distance: Float32
    var lock_pitch_deg: Float32
    var lock_framing_bias: Float32  # 0 = look at player pivot, 1 = look at lock target
    var fixed_pos: Vec3
    var fixed_look_at_target: Bool
    var fixed_rot: Quat
    var activation_serial: Int
    var pos: Vec3
    var rot: Quat

    def __init__(out self, mode: UInt8 = VCAM_FOLLOW, priority: Int = 0, blend_in_time: Float32 = 0.5):
        self.id = -1
        self.priority = priority
        self.mode = mode
        self.enabled = True
        self.blend_in_time = max(Float32(0.0), blend_in_time)
        self.fov_deg = 65.0
        self.rig = CameraRig(CAM_MODE_FIRST_PERSON if mode == VCAM_FIRST_PERSON else CAM_MODE_ORBIT_FOLLOW)
        self.rig.trauma_decay = 10.0  # shake lives in the brain
        self.follow_target = Vec3()
        self.follow_yaw_deg = 0.0
        self.yaw_follow_rate = 4.0
        self.lock_target = Vec3()
        self.has_lock_target = False
        self.lock_distance = 4.5
        self.lock_pitch_deg = 18.0
        self.lock_framing_bias = 0.4
        self.fixed_pos = Vec3(0.0, 3.0, -6.0)
        self.fixed_look_at_target = True
        self.fixed_rot = Quat()
        self.activation_serial = 0
        self.pos = Vec3()
        self.rot = Quat()

    @staticmethod
    def fixed(pos: Vec3, priority: Int = 0, look_at_target: Bool = True) -> VirtualCamera:
        var v = VirtualCamera(VCAM_FIXED, priority)
        v.fixed_pos = pos
        v.fixed_look_at_target = look_at_target
        return v^

    def solve(mut self, world: PhysicsWorld, dt: Float32):
        var safe_dt = Float32(0.0) if (not isfinite(dt) or dt < Float32(0.0)) else min(dt, Float32(1.0))
        if self.mode == VCAM_FIXED:
            self.pos = self.fixed_pos
            if self.fixed_look_at_target:
                var look = self.follow_target + self.rig.look_offset
                self.rot = _look_rotation(look - self.pos)
            else:
                self.rot = self.fixed_rot
            return
        if self.mode == VCAM_LOCK_ON and self.has_lock_target:
            var pivot = self.follow_target + self.rig.look_offset
            var to_t = self.lock_target - self.follow_target
            to_t.y = 0.0
            var lt = to_t.length()
            var dir = to_t * (Float32(1.0) / lt) if lt > Float32(1.0e-4) else Vec3(0.0, 0.0, 1.0)
            var pr = self.lock_pitch_deg * _VC_DEG_TO_RAD
            var arm = (-dir * det_cos(pr) + Vec3(0.0, det_sin(pr), 0.0)).normalized()
            self.rig.spring_arm.target_length = max(Float32(0.5), self.lock_distance)
            self.pos = self.rig.spring_arm.resolve(world, pivot, arm, safe_dt, self.rig.ignore_body_id)
            var bias = max(Float32(0.0), min(Float32(1.0), self.lock_framing_bias))
            var look = pivot + (self.lock_target + Vec3(0.0, self.rig.look_offset.y * Float32(0.6), 0.0) - pivot) * bias
            self.rot = _look_rotation(look - self.pos)
            # keep orbit yaw aligned so un-locking does not snap
            self.rig.yaw_deg = _yaw_of(dir)
            return
        if self.mode == VCAM_FOLLOW:
            var diff = self.follow_yaw_deg - self.rig.yaw_deg
            diff -= Float32(360.0) * floor((diff + Float32(180.0)) / Float32(360.0))
            var k = min(Float32(1.0), self.yaw_follow_rate * safe_dt)
            self.rig.rotate_orbit(diff * k, 0.0)
        self.rig.update(world, self.follow_target, safe_dt)
        self.pos = self.rig.unshaken_pos
        self.rot = self.rig.unshaken_rot


def _yaw_of(dir: Vec3) -> Float32:
    return det_atan2(dir.x, dir.z) * Float32(57.29577951)


@always_inline
def _smoothstep01(t: Float32) -> Float32:
    var x = max(Float32(0.0), min(Float32(1.0), t))
    return x * x * (Float32(3.0) - Float32(2.0) * x)


struct CameraBrain(Copyable, Movable):
    var vcams: List[VirtualCamera]
    var next_id: Int
    var active_id: Int
    var activation_counter: Int
    var blending: Bool
    var blend_t: Float32
    var blend_duration: Float32
    var blend_from_pos: Vec3
    var blend_from_rot: Quat
    var blend_from_fov: Float32
    var has_output: Bool

    # Event-driven trauma shake
    var trauma: Float32
    var trauma_decay: Float32
    var shake_freq_hz: Float32
    var max_shake_offset: Vec3
    var max_shake_deg: Vec3
    var shake_time: Float32
    var shake_seed: UInt64

    # Output pose
    var pos: Vec3
    var rot: Quat
    var fov_deg: Float32
    var unshaken_pos: Vec3
    var unshaken_rot: Quat

    def __init__(out self):
        self.vcams = List[VirtualCamera]()
        self.next_id = 1
        self.active_id = -1
        self.activation_counter = 0
        self.blending = False
        self.blend_t = 0.0
        self.blend_duration = 0.0
        self.blend_from_pos = Vec3()
        self.blend_from_rot = Quat()
        self.blend_from_fov = 65.0
        self.has_output = False
        self.trauma = 0.0
        self.trauma_decay = 1.2
        self.shake_freq_hz = 18.0
        self.max_shake_offset = Vec3(0.18, 0.18, 0.10)
        self.max_shake_deg = Vec3(4.5, 4.5, 3.0)
        self.shake_time = 0.0
        self.shake_seed = 0x2545F4914F6CDD1D
        self.pos = Vec3()
        self.rot = Quat()
        self.fov_deg = 65.0
        self.unshaken_pos = Vec3()
        self.unshaken_rot = Quat()

    def add_vcam(mut self, var vc: VirtualCamera) -> Int:
        vc.id = self.next_id
        self.next_id += 1
        self.activation_counter += 1
        vc.activation_serial = self.activation_counter
        self.vcams.append(vc^)
        return self.next_id - 1

    def index_of(self, id: Int) -> Int:
        for i in range(len(self.vcams)):
            if self.vcams[i].id == id:
                return i
        return -1

    def remove_vcam(mut self, id: Int):
        var i = self.index_of(id)
        if i >= 0:
            _ = self.vcams.pop(i)

    def set_priority(mut self, id: Int, priority: Int):
        var i = self.index_of(id)
        if i >= 0 and self.vcams[i].priority != priority:
            self.vcams[i].priority = priority
            self.activation_counter += 1
            self.vcams[i].activation_serial = self.activation_counter

    def set_enabled(mut self, id: Int, enabled: Bool):
        var i = self.index_of(id)
        if i >= 0 and self.vcams[i].enabled != enabled:
            self.vcams[i].enabled = enabled
            if enabled:
                self.activation_counter += 1
                self.vcams[i].activation_serial = self.activation_counter

    def set_follow_target(mut self, id: Int, target: Vec3, yaw_deg: Float32 = 0.0):
        var i = self.index_of(id)
        if i >= 0:
            self.vcams[i].follow_target = target
            self.vcams[i].follow_yaw_deg = yaw_deg

    def set_lock_target(mut self, id: Int, target: Vec3, has_target: Bool = True):
        var i = self.index_of(id)
        if i >= 0:
            self.vcams[i].lock_target = target
            self.vcams[i].has_lock_target = has_target

    def add_trauma(mut self, amount: Float32):
        if isfinite(amount):
            self.trauma = max(Float32(0.0), min(Float32(1.0), self.trauma + amount))

    def shake_event(mut self, pos: Vec3, strength: Float32, radius: Float32):
        """Positional impulse (explosion, landing): trauma falls off linearly to 0 at `radius`."""
        if not isfinite(strength) or radius <= Float32(0.0):
            return
        var d = (pos - self.unshaken_pos).length()
        if d >= radius:
            return
        self.add_trauma(strength * (Float32(1.0) - d / radius))

    def _select(self) -> Int:
        var best = -1
        for i in range(len(self.vcams)):
            if not self.vcams[i].enabled:
                continue
            if best < 0 or self.vcams[i].priority > self.vcams[best].priority or (
                self.vcams[i].priority == self.vcams[best].priority and self.vcams[i].activation_serial > self.vcams[best].activation_serial
            ):
                best = i
        return best

    def blend_weight(self) -> Float32:
        """Weight of the active vcam in the current output (1 when not blending)."""
        if not self.blending or self.blend_duration <= Float32(0.0):
            return 1.0
        return _smoothstep01(self.blend_t / self.blend_duration)

    def update(mut self, world: PhysicsWorld, dt: Float32):
        var safe_dt = Float32(0.0) if (not isfinite(dt) or dt < Float32(0.0)) else min(dt, Float32(1.0))
        for i in range(len(self.vcams)):
            if self.vcams[i].enabled:
                self.vcams[i].solve(world, safe_dt)
        var sel = self._select()
        if sel < 0:
            return
        var sel_id = self.vcams[sel].id
        if sel_id != self.active_id:
            if self.has_output and self.vcams[sel].blend_in_time > Float32(0.0):
                self.blending = True
                self.blend_t = 0.0
                self.blend_duration = self.vcams[sel].blend_in_time
                self.blend_from_pos = self.unshaken_pos
                self.blend_from_rot = self.unshaken_rot
                self.blend_from_fov = self.fov_deg
            else:
                self.blending = False
            self.active_id = sel_id

        var tgt_pos = self.vcams[sel].pos
        var tgt_rot = self.vcams[sel].rot
        var tgt_fov = self.vcams[sel].fov_deg
        if self.blending:
            self.blend_t += safe_dt
            var w = self.blend_weight()
            self.unshaken_pos = self.blend_from_pos + (tgt_pos - self.blend_from_pos) * w
            self.unshaken_rot = self.blend_from_rot.slerp(tgt_rot, w).normalized()
            self.fov_deg = self.blend_from_fov + (tgt_fov - self.blend_from_fov) * w
            if self.blend_t >= self.blend_duration:
                self.blending = False
        else:
            self.unshaken_pos = tgt_pos
            self.unshaken_rot = tgt_rot
            self.fov_deg = tgt_fov
        self.has_output = True

        # Deterministic trauma^2 shake (offset + rotation), decays linearly
        if self.trauma > Float32(1.0e-5) and safe_dt > Float32(0.0):
            self.shake_time += safe_dt
            if self.shake_time >= Float32(65536.0):
                self.shake_time -= Float32(65536.0)
            var shake = self.trauma * self.trauma
            var phase = self.shake_time * self.shake_freq_hz
            var ox = _det_perlin_1d(self.shake_seed ^ 0x1111, phase) * self.max_shake_offset.x * shake
            var oy = _det_perlin_1d(self.shake_seed ^ 0x2222, phase + Float32(13.7)) * self.max_shake_offset.y * shake
            var oz = _det_perlin_1d(self.shake_seed ^ 0x3333, phase + Float32(29.3)) * self.max_shake_offset.z * shake
            var ap = _det_perlin_1d(self.shake_seed ^ 0x4444, phase + Float32(47.1)) * self.max_shake_deg.x * _VC_DEG_TO_RAD * shake
            var ay = _det_perlin_1d(self.shake_seed ^ 0x5555, phase + Float32(61.9)) * self.max_shake_deg.y * _VC_DEG_TO_RAD * shake
            var ar = _det_perlin_1d(self.shake_seed ^ 0x6666, phase + Float32(83.5)) * self.max_shake_deg.z * _VC_DEG_TO_RAD * shake
            self.pos = self.unshaken_pos + self.unshaken_rot.rotate(Vec3(ox, oy, oz))
            self.rot = (self.unshaken_rot * Quat.from_euler(ap, ay, ar)).normalized()
            self.trauma = max(Float32(0.0), self.trauma - self.trauma_decay * safe_dt)
        else:
            self.pos = self.unshaken_pos
            self.rot = self.unshaken_rot

    def forward(self) -> Vec3:
        return self.rot.rotate(Vec3(0.0, 0.0, 1.0)).normalized()

    def world_matrix(self) -> Mat4:
        return trs(self.pos, self.rot, Vec3(1.0, 1.0, 1.0))
