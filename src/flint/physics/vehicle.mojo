"""Advanced Vehicle Dynamics (v4): 240 Hz suspension substepping, Ackermann steering,
anti-roll bars, 1.5-way LSD, bristle hill-hold, gyroscopic precession, and BVH8 terrain support.

Implements:
1. Multi-wheel `VehicleChassis` with rigid body inertial state, aerodynamic downforce & drag,
   Ackermann geometry, and roll-centre blended tire force application (`roll_center_blend = 0.65`).
2. Decoupled 240 Hz suspension substepping on both flat `y = 0` (`substep_suspension`) and
   arbitrary 3D triangle meshes (`substep_suspension_bvh` via `Bvh8MeshCollider.sweep_sphere`).
3. Stabilizing anti-roll bar coupling (`+arb` on compressed side, `-arb` on unloaded side)
   and correct negative lateral slip sign (`slip_angle = -det_atan2(v_lat, max(0.1, abs(v_long)))`).
4. Powertrain with dynamic `engine_rpm`, torque curve, per-wheel `cfg.brake_factor`,
   1.5-way Salisbury Limited-Slip Differential (LSD), and implicit chassis + wheel gyroscopics.
"""
from std.collections import List
from std.math import sqrt, max, min, abs, isfinite
from flint.core.math import Vec3, Quat
from flint.core.detmath import det_sin, det_cos, det_atan2
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.physics.tires import PacejkaTire, PacejkaParams
from flint.physics.bvh import Bvh8MeshCollider

comptime VEHICLE_BLOB_TAG: UInt32 = fourcc("VHCL")
comptime VEHICLE_BLOB_VERSION: Int = 1


@fieldwise_init
struct WheelConfig(Copyable, ImplicitlyCopyable, Movable):
    var offset: Vec3                 # Position relative to chassis CoM
    var radius: Float32              # Tire radius in meters (~0.33 m)
    var width: Float32               # Tire width in meters (~0.245 m)
    var spring_k: Float32            # Suspension spring rate (N/m)
    var damper_c_bump: Float32       # Compression damping (N*s/m)
    var damper_c_rebound: Float32    # Rebound damping (N*s/m)
    var suspension_rest_len: Float32 # Rest length in meters (~0.32 m)
    var max_travel: Float32          # Maximum allowable travel stroke (~0.20 m)
    var steered: Bool                # Does this wheel steer?
    var driven: Bool                 # Does this wheel receive drive torque?
    var brake_factor: Float32        # Proportion of brake torque

    def __init__(out self):
        self.offset = Vec3(0, 0, 0)
        self.radius = 0.33
        self.width = 0.245
        self.spring_k = 42000.0
        self.damper_c_bump = 2800.0
        self.damper_c_rebound = 4600.0
        self.suspension_rest_len = 0.32
        self.max_travel = 0.20
        self.steered = False
        self.driven = False
        self.brake_factor = 0.5


@fieldwise_init
struct WheelState(Copyable, ImplicitlyCopyable, Movable):
    var compression: Float32
    var compression_rate: Float32
    var steer_angle: Float32
    var angular_velocity: Float32
    var normal_load: Float32
    var slip_ratio: Float32
    var slip_angle: Float32
    var longitudinal_force: Float32
    var lateral_force: Float32
    var aligning_torque: Float32
    var grounded: Bool
    var contact_point: Vec3
    var contact_normal: Vec3
    var bristle_x: Float32
    var bristle_y: Float32

    def __init__(out self):
        self.compression = 0.0
        self.compression_rate = 0.0
        self.steer_angle = 0.0
        self.angular_velocity = 0.0
        self.normal_load = 0.0
        self.slip_ratio = 0.0
        self.slip_angle = 0.0
        self.longitudinal_force = 0.0
        self.lateral_force = 0.0
        self.aligning_torque = 0.0
        self.grounded = False
        self.contact_point = Vec3()
        self.contact_normal = Vec3(0, 1, 0)
        self.bristle_x = 0.0
        self.bristle_y = 0.0


@always_inline
def _clamp_vec3_f64(x: Float64, y: Float64, z: Float64, max_mag: Float64 = 1.0e35) -> Vec3:
    if not (isfinite(x) and isfinite(y) and isfinite(z)):
        return Vec3()
    var mag = sqrt(x * x + y * y + z * z)
    if not isfinite(mag):
        return Vec3()
    if mag > max_mag and mag > 0.0:
        var s = max_mag / mag
        return Vec3(Float32(x * s), Float32(y * s), Float32(z * s))
    return Vec3(Float32(x), Float32(y), Float32(z))


@always_inline
def _implicit_gyro_step(wb_in: Vec3, inertia: Vec3, dt: Float32) -> Vec3:
    """1-step Implicit Newton-Raphson solver for `I * (w^+ - w) + dt * (w^+ x (I * w^+)) = 0`."""
    var dt64 = Float64(dt)
    var ix = Float64(inertia.x)
    var iy = Float64(inertia.y)
    var iz = Float64(inertia.z)
    var wx = Float64(wb_in.x)
    var wy = Float64(wb_in.y)
    var wz = Float64(wb_in.z)
    var fx = dt64 * (iz - iy) * wy * wz
    var fy = dt64 * (ix - iz) * wz * wx
    var fz = dt64 * (iy - ix) * wx * wy

    var j00 = ix
    var j01 = dt64 * (iz - iy) * wz
    var j02 = dt64 * (iz - iy) * wy
    var j10 = dt64 * (ix - iz) * wz
    var j11 = iy
    var j12 = dt64 * (ix - iz) * wx
    var j20 = dt64 * (iy - ix) * wy
    var j21 = dt64 * (iy - ix) * wx
    var j22 = iz

    var c00 = j11 * j22 - j12 * j21
    var c01 = j12 * j20 - j10 * j22
    var c02 = j10 * j21 - j11 * j20
    var det = j00 * c00 + j01 * c01 + j02 * c02
    if not isfinite(det) or abs(det) <= 1.0e-18:
        return wb_in
    var inv_det = 1.0 / det
    var c10 = j02 * j21 - j01 * j22
    var c11 = j00 * j22 - j02 * j20
    var c12 = j01 * j20 - j00 * j21
    var c20 = j01 * j12 - j02 * j11
    var c21 = j02 * j10 - j00 * j12
    var c22 = j00 * j11 - j01 * j10
    var rx = wx - (c00 * fx + c10 * fy + c20 * fz) * inv_det
    var ry = wy - (c01 * fx + c11 * fy + c21 * fz) * inv_det
    var rz = wz - (c02 * fx + c12 * fy + c22 * fz) * inv_det
    if not (isfinite(rx) and isfinite(ry) and isfinite(rz)):
        return wb_in
    return _clamp_vec3_f64(rx, ry, rz, 3.4e38)


struct VehicleChassis(Copyable, Movable):
    var mass: Float32
    var inv_mass: Float32
    var inertia: Vec3
    var inv_inertia: Vec3
    var pos: Vec3
    var rot: Quat
    var linear_vel: Vec3
    var angular_vel: Vec3
    var wheels: List[WheelConfig]
    var wheel_states: List[WheelState]
    var tire_model: PacejkaTire
    var anti_roll_bar_front: Float32
    var anti_roll_bar_rear: Float32
    var roll_center_blend: Float32
    var drag_coeff: Float32
    var downforce_coeff: Float32
    var engine_rpm: Float32
    var idle_rpm: Float32
    var redline_rpm: Float32
    var current_gear: Int
    var gear_ratios: List[Float32]
    var final_drive: Float32
    var lsd_preload: Float32
    var lsd_power_lock: Float32
    var lsd_coast_lock: Float32
    var wheel_inertia: Float32
    var throttle_input: Float32
    var brake_input: Float32
    var steer_input: Float32

    def __init__(out self, mass: Float32 = 1450.0):
        self.mass = mass
        self.inv_mass = Float32(1.0) / max(Float32(1.0), mass)
        self.inertia = Vec3(2200.0, 2600.0, 750.0)
        self.inv_inertia = Vec3(Float32(1.0) / 2200.0, Float32(1.0) / 2600.0, Float32(1.0) / 750.0)
        self.pos = Vec3(0, 0.5, 0)
        self.rot = Quat()
        self.linear_vel = Vec3()
        self.angular_vel = Vec3()
        self.wheels = List[WheelConfig]()
        self.wheel_states = List[WheelState]()
        self.tire_model = PacejkaTire()
        self.anti_roll_bar_front = 14000.0
        self.anti_roll_bar_rear = 8000.0
        self.roll_center_blend = 0.65
        self.drag_coeff = 0.34
        self.downforce_coeff = 0.55
        self.engine_rpm = 850.0
        self.idle_rpm = 850.0
        self.redline_rpm = 7500.0
        self.current_gear = 1
        self.gear_ratios = List[Float32]()
        self.gear_ratios.append(-3.2)   # Reverse
        self.gear_ratios.append(3.6)    # 1st
        self.gear_ratios.append(2.1)    # 2nd
        self.gear_ratios.append(1.4)    # 3rd
        self.gear_ratios.append(1.0)    # 4th
        self.gear_ratios.append(0.8)    # 5th
        self.final_drive = 3.42
        self.lsd_preload = 120.0
        self.lsd_power_lock = 0.65
        self.lsd_coast_lock = 0.25
        self.wheel_inertia = 0.65
        self.throttle_input = 0.0
        self.brake_input = 0.0
        self.steer_input = 0.0

    def set_controls(mut self, throttle: Float32, brake: Float32, steer: Float32):
        self.throttle_input = throttle
        self.brake_input = brake
        self.steer_input = steer

    def setup_four_wheel_car(
        mut self,
        wheelbase: Float32 = 2.7,
        track_width: Float32 = 1.6,
        chassis_height: Float32 = 0.45,
    ):
        """Configures standard 4-wheel sports coupe chassis layout."""
        self.wheels.clear()
        self.wheel_states.clear()

        var half_wb = wheelbase * Float32(0.5)
        var half_tw = track_width * Float32(0.5)

        # 0: Front-Left (steered)
        var fl = WheelConfig()
        fl.offset = Vec3(-half_tw, -chassis_height, half_wb)
        fl.steered = True
        fl.driven = False
        fl.brake_factor = 0.65
        self.wheels.append(fl)
        self.wheel_states.append(WheelState())

        # 1: Front-Right (steered)
        var fr = WheelConfig()
        fr.offset = Vec3(half_tw, -chassis_height, half_wb)
        fr.steered = True
        fr.driven = False
        fr.brake_factor = 0.65
        self.wheels.append(fr)
        self.wheel_states.append(WheelState())

        # 2: Rear-Left (driven)
        var rl = WheelConfig()
        rl.offset = Vec3(-half_tw, -chassis_height, -half_wb)
        rl.steered = False
        rl.driven = True
        rl.brake_factor = 0.35
        self.wheels.append(rl)
        self.wheel_states.append(WheelState())

        # 3: Rear-Right (driven)
        var rr = WheelConfig()
        rr.offset = Vec3(half_tw, -chassis_height, -half_wb)
        rr.steered = False
        rr.driven = True
        rr.brake_factor = 0.35
        self.wheels.append(rr)
        self.wheel_states.append(WheelState())

    def _eval_engine_torque(self, rpm: Float32, throttle: Float32) -> Float32:
        """Evaluates naturally-aspirated/turbo sports-coupe torque curve with off-throttle engine braking."""
        var norm = max(Float32(0.0), min(Float32(1.0), (rpm - self.idle_rpm) / max(Float32(1.0), self.redline_rpm - self.idle_rpm)))
        # Parabolic torque curve: ~340 N*m at idle, ~485 N*m peak around 4600 RPM, ~400 N*m at redline
        var wide_open = Float32(340.0) + Float32(145.0) * (Float32(1.0) - (norm - Float32(0.56)) * (norm - Float32(0.56)) * Float32(2.6))
        if rpm >= self.redline_rpm - Float32(80.0):
            var cut = max(Float32(0.0), (self.redline_rpm - rpm) / Float32(80.0))
            wide_open *= cut
        var th = max(Float32(-1.0), min(Float32(1.0), throttle))
        if abs(th) > Float32(1.0e-4):
            return th * wide_open
        # Mild engine compression braking above idle
        return -Float32(35.0) * norm

    def _update_ackermann_steering(mut self, steer: Float32):
        var num_wheels = min(len(self.wheels), len(self.wheel_states))
        var max_steer_rad = Float32(0.55)
        var s_safe = steer if isfinite(steer) else Float32(0.0)
        var s_clamped = max(Float32(-1.0), min(Float32(1.0), s_safe))
        if abs(s_clamped) <= Float32(1.0e-5):
            for i in range(num_wheels):
                self.wheel_states[i].steer_angle = Float32(0.0)
            return

        var z_max = Float32(-1.0e30)
        var z_min = Float32(1.0e30)
        var x_max = Float32(0.0)
        for i in range(num_wheels):
            var off = self.wheels[i].offset
            z_max = max(z_max, off.z)
            z_min = min(z_min, off.z)
            if self.wheels[i].steered:
                x_max = max(x_max, abs(off.x))
        var wheelbase = max(Float32(0.5), z_max - z_min)
        var track = max(Float32(0.4), Float32(2.0) * x_max)

        var delta = abs(s_clamped) * max_steer_rad
        var tan_d = max(Float32(1.0e-5), det_sin(delta) / max(Float32(1.0e-5), det_cos(delta)))
        var turn_r = wheelbase / tan_d
        var d_in = det_atan2(wheelbase, max(Float32(0.25), turn_r - Float32(0.5) * track))
        var d_out = det_atan2(wheelbase, turn_r + Float32(0.5) * track)
        var sgn = Float32(1.0) if s_clamped > Float32(0.0) else Float32(-1.0)

        for i in range(num_wheels):
            if not self.wheels[i].steered:
                self.wheel_states[i].steer_angle = Float32(0.0)
            else:
                var ox = self.wheels[i].offset.x
                var is_inside = (s_clamped > Float32(0.0) and ox > Float32(0.0)) or (
                    s_clamped < Float32(0.0) and ox < Float32(0.0)
                )
                self.wheel_states[i].steer_angle = sgn * (d_in if is_inside else d_out)

    def substep_suspension(
        mut self,
        dt: Float32,
        throttle: Float32 = 0.0,
        brake: Float32 = 0.0,
        steer: Float32 = 0.0,
        substep_hz: Float32 = 240.0,
    ):
        """Runs decoupled high-frequency suspension substepping (240 Hz default) on flat `y = 0`."""
        if not isfinite(dt) or dt <= Float32(1.0e-7):
            return
        var th_in = throttle if (isfinite(throttle) and throttle != Float32(0.0)) else self.throttle_input
        var br_in = brake if (isfinite(brake) and brake != Float32(0.0)) else self.brake_input
        var st_in = steer if (isfinite(steer) and steer != Float32(0.0)) else self.steer_input
        var th = th_in if isfinite(th_in) else Float32(0.0)
        var br = br_in if isfinite(br_in) else Float32(0.0)
        var st = st_in if isfinite(st_in) else Float32(0.0)
        var safe_hz = max(Float32(60.0), min(Float32(2000.0), substep_hz)) if isfinite(substep_hz) else Float32(240.0)
        var safe_dt = min(dt, Float32(1.0))
        var sub_dt = Float32(1.0) / safe_hz
        var steps = min(2400, max(1, Int(safe_dt / sub_dt + Float32(0.5))))
        var actual_sub_dt = safe_dt / Float32(steps)

        for _ in range(steps):
            self._probe_flat_ground(actual_sub_dt)
            self._step_dynamics_core(actual_sub_dt, th, br, st)

    def substep_suspension_bvh(
        mut self,
        dt: Float32,
        bvh: Bvh8MeshCollider,
        throttle: Float32 = 0.0,
        brake: Float32 = 0.0,
        steer: Float32 = 0.0,
        substep_hz: Float32 = 240.0,
    ):
        """Runs 240 Hz suspension substepping against a `Bvh8MeshCollider` terrain/track."""
        if not isfinite(dt) or dt <= Float32(1.0e-7):
            return
        var th_in = throttle if (isfinite(throttle) and throttle != Float32(0.0)) else self.throttle_input
        var br_in = brake if (isfinite(brake) and brake != Float32(0.0)) else self.brake_input
        var st_in = steer if (isfinite(steer) and steer != Float32(0.0)) else self.steer_input
        var th = th_in if isfinite(th_in) else Float32(0.0)
        var br = br_in if isfinite(br_in) else Float32(0.0)
        var st = st_in if isfinite(st_in) else Float32(0.0)
        var safe_hz = max(Float32(60.0), min(Float32(2000.0), substep_hz)) if isfinite(substep_hz) else Float32(240.0)
        var safe_dt = min(dt, Float32(1.0))
        var sub_dt = Float32(1.0) / safe_hz
        var steps = min(2400, max(1, Int(safe_dt / sub_dt + Float32(0.5))))
        var actual_sub_dt = safe_dt / Float32(steps)

        for _ in range(steps):
            self._probe_bvh_ground(actual_sub_dt, bvh)
            self._step_dynamics_core(actual_sub_dt, th, br, st)

    def substep_suspension_bvh(
        mut self,
        dt: Float32,
        throttle: Float32,
        brake: Float32,
        steer: Float32,
        bvh: Bvh8MeshCollider,
        substep_hz: Float32 = 240.0,
    ):
        """Overload accepting `(dt, throttle, brake, steer, bvh)`."""
        self.substep_suspension_bvh(dt, bvh, throttle, brake, steer, substep_hz)

    def _probe_flat_ground(mut self, dt: Float32):
        var up = self.rot.rotate(Vec3(0, 1, 0)).normalized()
        var num_wheels = min(len(self.wheels), len(self.wheel_states))
        for i in range(num_wheels):
            var hub_r = self.rot.rotate(self.wheels[i].offset)
            var world_hub = self.pos + hub_r
            var rest_len = self.wheels[i].suspension_rest_len
            var r = self.wheels[i].radius
            var ground_y = Float32(0.0)

            var wheel_bottom = world_hub.y - up.y * rest_len - r
            if wheel_bottom < ground_y:
                var was_grounded = self.wheel_states[i].grounded
                var old_comp = self.wheel_states[i].compression
                var new_comp = min(self.wheels[i].max_travel, max(Float32(0.0), ground_y - wheel_bottom))
                if was_grounded:
                    var cr = (Float64(new_comp) - Float64(old_comp)) / Float64(dt)
                    self.wheel_states[i].compression_rate = Float32(max(-3.4e38, min(3.4e38, cr))) if isfinite(cr) else Float32(0.0)
                else:
                    var hv_y64 = Float64(self.linear_vel.y) + (
                        Float64(self.angular_vel.z) * Float64(hub_r.x)
                        - Float64(self.angular_vel.x) * Float64(hub_r.z)
                    )
                    self.wheel_states[i].compression_rate = Float32(max(-3.4e38, min(3.4e38, -hv_y64))) if isfinite(hv_y64) else Float32(0.0)
                self.wheel_states[i].grounded = True
                self.wheel_states[i].compression = new_comp
                self.wheel_states[i].contact_point = Vec3(world_hub.x, ground_y, world_hub.z)
                self.wheel_states[i].contact_normal = Vec3(0, 1, 0)
            else:
                self.wheel_states[i].grounded = False
                self.wheel_states[i].compression = Float32(0.0)
                self.wheel_states[i].compression_rate = Float32(0.0)
                self.wheel_states[i].bristle_x = Float32(0.0)
                self.wheel_states[i].bristle_y = Float32(0.0)

    def _probe_bvh_ground(mut self, dt: Float32, bvh: Bvh8MeshCollider):
        var up = self.rot.rotate(Vec3(0, 1, 0)).normalized()
        var down = -up
        var num_wheels = min(len(self.wheels), len(self.wheel_states))
        for i in range(num_wheels):
            var cfg = self.wheels[i]
            var hub_r = self.rot.rotate(cfg.offset)
            var world_hub = self.pos + hub_r
            var sweep_start = world_hub + up * cfg.max_travel
            var sweep_len = cfg.max_travel + cfg.suspension_rest_len
            var sh = bvh.sweep_sphere(sweep_start, down * sweep_len, cfg.radius, Float32(1.0))
            if sh.hit and sh.toi < Float32(1.0):
                var n = sh.normal
                if n.dot(up) < Float32(0.0):
                    n = -n
                var was_grounded = self.wheel_states[i].grounded
                var old_comp = self.wheel_states[i].compression
                var raw_comp = (Float32(1.0) - sh.toi) * sweep_len
                var new_comp = min(cfg.max_travel, max(Float32(0.0), raw_comp))
                if was_grounded:
                    var cr = (Float64(new_comp) - Float64(old_comp)) / Float64(dt)
                    self.wheel_states[i].compression_rate = Float32(max(-3.4e38, min(3.4e38, cr))) if isfinite(cr) else Float32(0.0)
                else:
                    var hv_x64 = Float64(self.linear_vel.x) + (Float64(self.angular_vel.y) * Float64(hub_r.z) - Float64(self.angular_vel.z) * Float64(hub_r.y))
                    var hv_y64 = Float64(self.linear_vel.y) + (Float64(self.angular_vel.z) * Float64(hub_r.x) - Float64(self.angular_vel.x) * Float64(hub_r.z))
                    var hv_z64 = Float64(self.linear_vel.z) + (Float64(self.angular_vel.x) * Float64(hub_r.y) - Float64(self.angular_vel.y) * Float64(hub_r.x))
                    var cr = -(hv_x64 * Float64(n.x) + hv_y64 * Float64(n.y) + hv_z64 * Float64(n.z))
                    self.wheel_states[i].compression_rate = Float32(max(-3.4e38, min(3.4e38, cr))) if isfinite(cr) else Float32(0.0)
                self.wheel_states[i].grounded = True
                self.wheel_states[i].compression = new_comp
                self.wheel_states[i].contact_point = sh.point
                self.wheel_states[i].contact_normal = n
            else:
                self.wheel_states[i].grounded = False
                self.wheel_states[i].compression = Float32(0.0)
                self.wheel_states[i].compression_rate = Float32(0.0)
                self.wheel_states[i].bristle_x = Float32(0.0)
                self.wheel_states[i].bristle_y = Float32(0.0)

    def _step_dynamics_core(
        mut self,
        dt: Float32,
        throttle: Float32,
        brake: Float32,
        steer: Float32,
    ):
        var unused_forces = List[Vec3]()
        var empty_ground_vels = List[Vec3]()
        self._step_dynamics_impl(
            dt,
            throttle,
            brake,
            steer,
            Float32(9.81),
            empty_ground_vels,
            unused_forces,
            True,
        )

    def step_world_substep(
        mut self,
        dt: Float32,
        throttle: Float32,
        brake: Float32,
        steer: Float32,
        gravity_y: Float32,
        ground_vels: List[Vec3],
        mut out_wheel_forces: List[Vec3],
        apply_gyro: Bool = False,
    ):
        """Advances one 240 Hz vehicle substep using externally probed `wheel_states`
        and moving contact surface `ground_vels`, emitting per-wheel world forces
        for two-way reaction impulses onto hit `PhysicsWorld` dynamic bodies."""
        self._step_dynamics_impl(
            dt,
            throttle,
            brake,
            steer,
            gravity_y,
            ground_vels,
            out_wheel_forces,
            apply_gyro,
        )

    def _step_dynamics_impl(
        mut self,
        dt_in: Float32,
        throttle_in: Float32,
        brake_in: Float32,
        steer_in: Float32,
        gravity_y_in: Float32,
        ground_vels: List[Vec3],
        mut out_wheel_forces: List[Vec3],
        apply_gyro: Bool,
    ):
        var num_wheels = len(self.wheels)
        out_wheel_forces.clear()
        for _ in range(num_wheels):
            out_wheel_forces.append(Vec3())
        if num_wheels < 4 or len(self.wheel_states) < num_wheels:
            return
        if not isfinite(dt_in) or dt_in <= Float32(1.0e-7):
            return
        var dt = min(dt_in, Float32(0.1))
        var throttle = max(Float32(0.0), min(Float32(1.0), throttle_in)) if isfinite(throttle_in) else Float32(0.0)
        var brake = max(Float32(0.0), min(Float32(1.0), brake_in)) if isfinite(brake_in) else Float32(0.0)
        var steer = max(Float32(-1.0), min(Float32(1.0), steer_in)) if isfinite(steer_in) else Float32(0.0)
        var gravity_y = gravity_y_in if isfinite(gravity_y_in) else Float32(9.81)
        if not (isfinite(self.pos.x) and isfinite(self.pos.y) and isfinite(self.pos.z)):
            self.pos = Vec3()
        if not (isfinite(self.linear_vel.x) and isfinite(self.linear_vel.y) and isfinite(self.linear_vel.z)):
            self.linear_vel = Vec3()
        if not (isfinite(self.angular_vel.x) and isfinite(self.angular_vel.y) and isfinite(self.angular_vel.z)):
            self.angular_vel = Vec3()
        if not (isfinite(self.rot.x) and isfinite(self.rot.y) and isfinite(self.rot.z) and isfinite(self.rot.w)):
            self.rot = Quat()
        if not isfinite(self.engine_rpm):
            self.engine_rpm = self.idle_rpm

        var fwd = self.rot.rotate(Vec3(0, 0, 1)).normalized()
        var up = self.rot.rotate(Vec3(0, 1, 0)).normalized()
        var right = self.rot.rotate(Vec3(1, 0, 0)).normalized()

        # 1. Aerodynamic drag & downforce (Float64 overflow-safe for extreme speeds)
        var lv_x64 = Float64(self.linear_vel.x)
        var lv_y64 = Float64(self.linear_vel.y)
        var lv_z64 = Float64(self.linear_vel.z)
        var fwd_speed64 = lv_x64 * Float64(fwd.x) + lv_y64 * Float64(fwd.y) + lv_z64 * Float64(fwd.z)
        var speed64 = sqrt(lv_x64 * lv_x64 + lv_y64 * lv_y64 + lv_z64 * lv_z64)
        var drag_k64 = 0.5 * 1.225 * Float64(max(Float32(0.0), self.drag_coeff)) * 2.2 * speed64
        var max_drag_k64 = 0.99 / max(1.0e-9, Float64(self.inv_mass) * Float64(dt))
        if drag_k64 > max_drag_k64:
            drag_k64 = max_drag_k64
        var f_drag = _clamp_vec3_f64(
            -lv_x64 * drag_k64,
            -lv_y64 * drag_k64,
            -lv_z64 * drag_k64,
            1.0e35,
        )
        var f_down_mag64 = min(1.0e35, 0.5 * 1.225 * Float64(max(Float32(0.0), self.downforce_coeff)) * 2.2 * fwd_speed64 * fwd_speed64)
        var f_down = _clamp_vec3_f64(
            -Float64(up.x) * f_down_mag64,
            -Float64(up.y) * f_down_mag64,
            -Float64(up.z) * f_down_mag64,
            1.0e35,
        )
        var f_grav = _clamp_vec3_f64(0.0, -Float64(gravity_y) * Float64(self.mass), 0.0, 1.0e35)

        var tf_x64 = Float64(f_grav.x) + Float64(f_drag.x) + Float64(f_down.x)
        var tf_y64 = Float64(f_grav.y) + Float64(f_drag.y) + Float64(f_down.y)
        var tf_z64 = Float64(f_grav.z) + Float64(f_drag.z) + Float64(f_down.z)
        var tt_x64 = 0.0
        var tt_y64 = 0.0
        var tt_z64 = 0.0

        # 2. Ackermann wheel steering geometry
        self._update_ackermann_steering(steer)

        # 3. Stabilizing anti-roll bar coupling: +arb on compressed side, -arb on unloaded side
        var arb_f = self.anti_roll_bar_front * (self.wheel_states[0].compression - self.wheel_states[1].compression)
        var arb_r = self.anti_roll_bar_rear * (self.wheel_states[2].compression - self.wheel_states[3].compression)

        # 4. Suspension normal loads & forces along contact normal
        for i in range(num_wheels):
            if not self.wheel_states[i].grounded:
                self.wheel_states[i].normal_load = Float32(0.0)
                continue

            var cfg = self.wheels[i]
            var comp = self.wheel_states[i].compression if isfinite(self.wheel_states[i].compression) else Float32(0.0)
            var crate = self.wheel_states[i].compression_rate if isfinite(self.wheel_states[i].compression_rate) else Float32(0.0)
            var f_spring64 = Float64(cfg.spring_k) * Float64(comp)
            var c_damp = cfg.damper_c_bump if crate >= Float32(0.0) else cfg.damper_c_rebound
            var f_damper64 = Float64(c_damp) * Float64(crate)

            var arb_force = Float32(0.0)
            if i == 0:
                arb_force = arb_f
            elif i == 1:
                arb_force = -arb_f
            elif i == 2:
                arb_force = arb_r
            elif i == 3:
                arb_force = -arb_r
            var arb64 = Float64(arb_force) if isfinite(arb_force) else 0.0

            var f_norm64 = f_spring64 + f_damper64 + arb64
            if not isfinite(f_norm64) or f_norm64 < 0.0:
                f_norm64 = 0.0
            elif f_norm64 > 1.0e35:
                f_norm64 = 1.0e35
            self.wheel_states[i].normal_load = Float32(f_norm64)

            var hub_r = self.rot.rotate(cfg.offset)
            var cn = self.wheel_states[i].contact_normal
            var susp_force = _clamp_vec3_f64(
                Float64(cn.x) * f_norm64,
                Float64(cn.y) * f_norm64,
                Float64(cn.z) * f_norm64,
                1.0e35,
            )
            out_wheel_forces[i] = _clamp_vec3_f64(
                Float64(out_wheel_forces[i].x) + Float64(susp_force.x),
                Float64(out_wheel_forces[i].y) + Float64(susp_force.y),
                Float64(out_wheel_forces[i].z) + Float64(susp_force.z),
                1.0e35,
            )
            tf_x64 += Float64(susp_force.x)
            tf_y64 += Float64(susp_force.y)
            tf_z64 += Float64(susp_force.z)
            tt_x64 += Float64(hub_r.y) * Float64(susp_force.z) - Float64(hub_r.z) * Float64(susp_force.y)
            tt_y64 += Float64(hub_r.z) * Float64(susp_force.x) - Float64(hub_r.x) * Float64(susp_force.z)
            tt_z64 += Float64(hub_r.x) * Float64(susp_force.y) - Float64(hub_r.y) * Float64(susp_force.x)

        # 5. Powertrain: engine RPM update, torque curve, and 1.5-way LSD torque split
        var total_ratio = self.final_drive
        if len(self.gear_ratios) > 0:
            var gear_idx = max(0, min(len(self.gear_ratios) - 1, self.current_gear))
            total_ratio = self.gear_ratios[gear_idx] * self.final_drive
        var driven_count = 0
        var driven_omega_sum64 = 0.0
        var left_driven = -1
        var right_driven = -1
        for i in range(num_wheels):
            if not isfinite(self.wheel_states[i].angular_velocity):
                self.wheel_states[i].angular_velocity = Float32(0.0)
            if self.wheels[i].driven:
                driven_count += 1
                driven_omega_sum64 += Float64(self.wheel_states[i].angular_velocity)
                if self.wheels[i].offset.x < Float32(0.0) and left_driven < 0:
                    left_driven = i
                elif self.wheels[i].offset.x >= Float32(0.0) and right_driven < 0:
                    right_driven = i

        var avg_omega64 = driven_omega_sum64 / Float64(max(1, driven_count))
        var wheel_rpm64 = avg_omega64 * 9.5492966
        var raw_target64 = abs(wheel_rpm64 * Float64(total_ratio)) + Float64(max(Float32(0.0), throttle)) * 650.0
        if not isfinite(raw_target64):
            raw_target64 = Float64(self.redline_rpm)
        var target_rpm = max(
            self.idle_rpm,
            min(
                self.redline_rpm,
                Float32(min(Float64(self.redline_rpm), raw_target64)),
            ),
        )
        self.engine_rpm += (target_rpm - self.engine_rpm) * min(Float32(1.0), dt * Float32(20.0))
        if not isfinite(self.engine_rpm):
            self.engine_rpm = self.idle_rpm

        var eng_torque = self._eval_engine_torque(self.engine_rpm, throttle)
        var axle_torque = eng_torque * total_ratio

        var wheel_drive_torque = Array[Float32, 8](fill=0.0)
        if left_driven >= 0 and right_driven >= 0 and left_driven < 8 and right_driven < 8:
            var lock_factor = self.lsd_power_lock if axle_torque >= Float32(0.0) else self.lsd_coast_lock
            var max_lock = self.lsd_preload + lock_factor * abs(axle_torque)
            var d_omega64 = Float64(self.wheel_states[left_driven].angular_velocity) - Float64(self.wheel_states[right_driven].angular_velocity)
            var t_lsd = max(-max_lock, min(max_lock, Float32(max(-Float64(max_lock), min(Float64(max_lock), 45.0 * d_omega64)))))
            wheel_drive_torque[left_driven] = Float32(0.5) * axle_torque - t_lsd
            wheel_drive_torque[right_driven] = Float32(0.5) * axle_torque + t_lsd
        elif driven_count > 0:
            var share = axle_torque / Float32(driven_count)
            for i in range(min(8, num_wheels)):
                if self.wheels[i].driven:
                    wheel_drive_torque[i] = share

        # 6. Tire traction via bristle-coupled Pacejka combined slip, per-wheel brake_factor & gyroscopics
        var brake_clamped = max(Float32(0.0), min(Float32(1.0), brake))
        var max_brake_torque = brake_clamped * Float32(4200.0)
        var w_inertia = max(Float32(0.1), self.wheel_inertia)

        for i in range(num_wheels):
            var cfg = self.wheels[i]
            if not self.wheel_states[i].grounded or self.wheel_states[i].normal_load <= Float32(1.0):
                self.wheel_states[i].longitudinal_force = Float32(0.0)
                self.wheel_states[i].lateral_force = Float32(0.0)
                self.wheel_states[i].aligning_torque = Float32(0.0)
                continue

            var n_contact = self.wheel_states[i].contact_normal
            var arm_x64 = Float64(self.wheel_states[i].contact_point.x) - Float64(self.pos.x)
            var arm_y64 = Float64(self.wheel_states[i].contact_point.y) - Float64(self.pos.y)
            var arm_z64 = Float64(self.wheel_states[i].contact_point.z) - Float64(self.pos.z)
            var ground_v = ground_vels[i] if i < len(ground_vels) else Vec3()
            var gv_x64 = Float64(ground_v.x) if isfinite(ground_v.x) else 0.0
            var gv_y64 = Float64(ground_v.y) if isfinite(ground_v.y) else 0.0
            var gv_z64 = Float64(ground_v.z) if isfinite(ground_v.z) else 0.0
            var pv_x64 = lv_x64 + (Float64(self.angular_vel.y) * arm_z64 - Float64(self.angular_vel.z) * arm_y64) - gv_x64
            var pv_y64 = lv_y64 + (Float64(self.angular_vel.z) * arm_x64 - Float64(self.angular_vel.x) * arm_z64) - gv_y64
            var pv_z64 = lv_z64 + (Float64(self.angular_vel.x) * arm_y64 - Float64(self.angular_vel.y) * arm_x64) - gv_z64

            var sa = self.wheel_states[i].steer_angle
            var c_sa = det_cos(sa)
            var s_sa = det_sin(sa)
            var steer_fwd = fwd * c_sa + right * s_sa
            var wheel_fwd = (steer_fwd - n_contact * steer_fwd.dot(n_contact)).normalized()
            if wheel_fwd.length_squared() < Float32(0.25):
                wheel_fwd = steer_fwd.normalized()
            var wheel_lat = n_contact.cross(wheel_fwd).normalized()

            var v_long64 = pv_x64 * Float64(wheel_fwd.x) + pv_y64 * Float64(wheel_fwd.y) + pv_z64 * Float64(wheel_fwd.z)
            var v_lat64 = pv_x64 * Float64(wheel_lat.x) + pv_y64 * Float64(wheel_lat.y) + pv_z64 * Float64(wheel_lat.z)
            var v_long = Float32(max(-3.4e38, min(3.4e38, v_long64))) if isfinite(v_long64) else Float32(0.0)
            var v_lat = Float32(max(-3.4e38, min(3.4e38, v_lat64))) if isfinite(v_lat64) else Float32(0.0)

            var r = cfg.radius
            var omega64 = Float64(self.wheel_states[i].angular_velocity)
            if cfg.driven and i < 8:
                omega64 += (Float64(wheel_drive_torque[i]) / Float64(w_inertia)) * Float64(dt)

            var wheel_brake_t = max_brake_torque * max(Float32(0.0), cfg.brake_factor)
            if brake_clamped > Float32(0.001) and abs(v_long) <= Float32(1.5):
                var pre_decel64 = (Float64(wheel_brake_t) / Float64(w_inertia)) * Float64(dt)
                if abs(omega64) <= pre_decel64:
                    omega64 = 0.0
                elif omega64 > 0.0:
                    omega64 -= pre_decel64
                else:
                    omega64 += pre_decel64

            var tire_v64 = omega64 * Float64(r)
            var tire_v = Float32(max(-3.4e38, min(3.4e38, tire_v64))) if isfinite(tire_v64) else Float32(0.0)
            var sr64 = (tire_v64 - v_long64) / max(0.1, abs(v_long64))
            var slip_ratio = Float32(max(-3.4e38, min(3.4e38, sr64))) if isfinite(sr64) else Float32(0.0)
            var slip_angle = -det_atan2(v_lat, max(Float32(0.1), abs(v_long)))

            self.wheel_states[i].slip_ratio = slip_ratio
            self.wheel_states[i].slip_angle = slip_angle

            var forces = self.tire_model.calc_forces_bristle(
                v_long,
                v_lat,
                tire_v,
                self.wheel_states[i].normal_load,
                dt,
                self.wheel_states[i].bristle_x,
                self.wheel_states[i].bristle_y,
            )
            var fx = forces[0] if isfinite(forces[0]) else Float32(0.0)
            var fy = forces[1] if isfinite(forces[1]) else Float32(0.0)
            var mz = forces[2] if isfinite(forces[2]) else Float32(0.0)

            var reaction_torque64 = -Float64(fx) * Float64(r)
            var new_omega64 = omega64 + (reaction_torque64 / Float64(w_inertia)) * Float64(dt)
            if brake_clamped > Float32(0.001):
                var decel64 = (Float64(wheel_brake_t) / Float64(w_inertia)) * Float64(dt)
                if abs(v_long64) > 1.5:
                    var abs_floor64 = (1.0 - 0.15 * Float64(brake_clamped)) * (abs(v_long64) / max(0.05, Float64(r)))
                    if abs(new_omega64) > abs_floor64:
                        if new_omega64 > 0.0:
                            new_omega64 = max(abs_floor64, new_omega64 - decel64)
                        else:
                            new_omega64 = min(-abs_floor64, new_omega64 + decel64)
                else:
                    if abs(new_omega64) <= decel64:
                        new_omega64 = 0.0
                    elif new_omega64 > 0.0:
                        new_omega64 -= decel64
                    else:
                        new_omega64 += decel64
            var new_omega = Float32(max(-3.4e38, min(3.4e38, new_omega64))) if isfinite(new_omega64) else Float32(0.0)
            self.wheel_states[i].angular_velocity = new_omega
            self.wheel_states[i].longitudinal_force = fx
            self.wheel_states[i].lateral_force = fy
            self.wheel_states[i].aligning_torque = mz

            var tf_wx64 = Float64(wheel_fwd.x) * Float64(fx) + Float64(wheel_lat.x) * Float64(fy)
            var tf_wy64 = Float64(wheel_fwd.y) * Float64(fx) + Float64(wheel_lat.y) * Float64(fy)
            var tf_wz64 = Float64(wheel_fwd.z) * Float64(fx) + Float64(wheel_lat.z) * Float64(fy)
            var tire_force = _clamp_vec3_f64(tf_wx64, tf_wy64, tf_wz64, 1.0e35)
            out_wheel_forces[i] = _clamp_vec3_f64(
                Float64(out_wheel_forces[i].x) + Float64(tire_force.x),
                Float64(out_wheel_forces[i].y) + Float64(tire_force.y),
                Float64(out_wheel_forces[i].z) + Float64(tire_force.z),
                1.0e35,
            )
            tf_x64 += Float64(tire_force.x)
            tf_y64 += Float64(tire_force.y)
            tf_z64 += Float64(tire_force.z)

            # Apply tire force at (contact_point - self.pos) with roll_center_blend
            var arm_dot_up64 = arm_x64 * Float64(up.x) + arm_y64 * Float64(up.y) + arm_z64 * Float64(up.z)
            var av_x64 = Float64(up.x) * arm_dot_up64
            var av_y64 = Float64(up.y) * arm_dot_up64
            var av_z64 = Float64(up.z) * arm_dot_up64
            var rcb64 = Float64(self.roll_center_blend)
            var fa_x64 = (arm_x64 - av_x64) + av_x64 * rcb64
            var fa_y64 = (arm_y64 - av_y64) + av_y64 * rcb64
            var fa_z64 = (arm_z64 - av_z64) + av_z64 * rcb64
            tt_x64 += (fa_y64 * Float64(tire_force.z) - fa_z64 * Float64(tire_force.y)) + Float64(n_contact.x) * Float64(mz)
            tt_y64 += (fa_z64 * Float64(tire_force.x) - fa_x64 * Float64(tire_force.z)) + Float64(n_contact.y) * Float64(mz)
            tt_z64 += (fa_x64 * Float64(tire_force.y) - fa_y64 * Float64(tire_force.x)) + Float64(n_contact.z) * Float64(mz)

            # Spinning wheel gyroscopic precession reaction: -omega_chassis x L_wheel
            var l_scale64 = Float64(w_inertia) * Float64(new_omega)
            var wl_x64 = -Float64(wheel_lat.x) * l_scale64
            var wl_y64 = -Float64(wheel_lat.y) * l_scale64
            var wl_z64 = -Float64(wheel_lat.z) * l_scale64
            tt_x64 -= Float64(self.angular_vel.y) * wl_z64 - Float64(self.angular_vel.z) * wl_y64
            tt_y64 -= Float64(self.angular_vel.z) * wl_x64 - Float64(self.angular_vel.x) * wl_z64
            tt_z64 -= Float64(self.angular_vel.x) * wl_y64 - Float64(self.angular_vel.y) * wl_x64

        # 7. Rigid body integration with implicit chassis gyroscopic torque & SO(3) attitude update
        var total_force = _clamp_vec3_f64(tf_x64, tf_y64, tf_z64, 1.0e35)
        var total_torque = _clamp_vec3_f64(tt_x64, tt_y64, tt_z64, 1.0e35)
        var inv_m_dt64 = Float64(self.inv_mass) * Float64(dt)
        self.linear_vel = _clamp_vec3_f64(
            lv_x64 + Float64(total_force.x) * inv_m_dt64,
            lv_y64 + Float64(total_force.y) * inv_m_dt64,
            lv_z64 + Float64(total_force.z) * inv_m_dt64,
            3.4e38,
        )
        self.pos = _clamp_vec3_f64(
            Float64(self.pos.x) + Float64(self.linear_vel.x) * Float64(dt),
            Float64(self.pos.y) + Float64(self.linear_vel.y) * Float64(dt),
            Float64(self.pos.z) + Float64(self.linear_vel.z) * Float64(dt),
            3.4e38,
        )

        var inv_rot = self.rot.conjugate()
        var wb_cur = inv_rot.rotate(self.angular_vel)
        var tau_local = inv_rot.rotate(total_torque)
        var wb_forced = _clamp_vec3_f64(
            Float64(wb_cur.x) + Float64(tau_local.x) * Float64(self.inv_inertia.x) * Float64(dt),
            Float64(wb_cur.y) + Float64(tau_local.y) * Float64(self.inv_inertia.y) * Float64(dt),
            Float64(wb_cur.z) + Float64(tau_local.z) * Float64(self.inv_inertia.z) * Float64(dt),
            3.4e38,
        )
        var wb_next = _implicit_gyro_step(wb_forced, self.inertia, dt) if apply_gyro else wb_forced
        var av_rot = self.rot.rotate(wb_next)
        self.angular_vel = _clamp_vec3_f64(Float64(av_rot.x), Float64(av_rot.y), Float64(av_rot.z), 3.4e38)

        var dw_x64 = Float64(self.angular_vel.x) * Float64(dt)
        var dw_y64 = Float64(self.angular_vel.y) * Float64(dt)
        var dw_z64 = Float64(self.angular_vel.z) * Float64(dt)
        var ang64 = sqrt(dw_x64 * dw_x64 + dw_y64 * dw_y64 + dw_z64 * dw_z64)
        var dq: Quat
        if not isfinite(ang64) or ang64 < 1.0e-12:
            dq = Quat()
        elif ang64 < 1.0e-5:
            dq = Quat(
                Float32(dw_x64 * 0.5),
                Float32(dw_y64 * 0.5),
                Float32(dw_z64 * 0.5),
                Float32(1.0),
            ).normalized()
        else:
            var half32 = Float32((ang64 * 0.5) % 6.28318530717958647692)
            var s = det_sin(half32)
            var inv_ang = 1.0 / ang64
            dq = Quat(
                Float32(dw_x64 * inv_ang) * s,
                Float32(dw_y64 * inv_ang) * s,
                Float32(dw_z64 * inv_ang) * s,
                det_cos(half32),
            ).normalized()
        var next_rot = (dq * self.rot).normalized()
        if isfinite(next_rot.x) and isfinite(next_rot.y) and isfinite(next_rot.z) and isfinite(next_rot.w):
            self.rot = next_rot

    def serialize(self, mut w: BlobWriter):
        """Bit-exact snapshot of chassis, powertrain, tire parameters, and wheel elastoplastic bristle state."""
        w.begin(VEHICLE_BLOB_TAG, VEHICLE_BLOB_VERSION)
        w.f32(self.mass)
        w.f32(self.inv_mass)
        w.vec3(self.inertia)
        w.vec3(self.inv_inertia)
        w.vec3(self.pos)
        w.quat(self.rot)
        w.vec3(self.linear_vel)
        w.vec3(self.angular_vel)
        w.f32(self.anti_roll_bar_front)
        w.f32(self.anti_roll_bar_rear)
        w.f32(self.roll_center_blend)
        w.f32(self.drag_coeff)
        w.f32(self.downforce_coeff)
        w.f32(self.engine_rpm)
        w.f32(self.idle_rpm)
        w.f32(self.redline_rpm)
        w.integer(self.current_gear)
        w.integer(len(self.gear_ratios))
        for i in range(len(self.gear_ratios)):
            w.f32(self.gear_ratios[i])
        w.f32(self.final_drive)
        w.f32(self.lsd_preload)
        w.f32(self.lsd_power_lock)
        w.f32(self.lsd_coast_lock)
        w.f32(self.wheel_inertia)
        w.f32(self.throttle_input)
        w.f32(self.brake_input)
        w.f32(self.steer_input)

        var tp = self.tire_model.params
        w.f32(tp.B)
        w.f32(tp.C)
        w.f32(tp.D)
        w.f32(tp.E)
        w.f32(tp.load_sensitivity)
        w.f32(tp.nominal_load)
        w.f32(tp.brush_stiffness)
        w.f32(tp.pneumatic_trail_0)
        w.f32(tp.kappa_peak)
        w.f32(tp.alpha_peak)
        w.f32(tp.relaxation_len_x)
        w.f32(tp.relaxation_len_y)

        w.integer(len(self.wheels))
        for i in range(len(self.wheels)):
            var wc = self.wheels[i]
            w.vec3(wc.offset)
            w.f32(wc.radius)
            w.f32(wc.width)
            w.f32(wc.spring_k)
            w.f32(wc.damper_c_bump)
            w.f32(wc.damper_c_rebound)
            w.f32(wc.suspension_rest_len)
            w.f32(wc.max_travel)
            w.boolean(wc.steered)
            w.boolean(wc.driven)
            w.f32(wc.brake_factor)

        w.integer(len(self.wheel_states))
        for i in range(len(self.wheel_states)):
            var ws = self.wheel_states[i]
            w.f32(ws.compression)
            w.f32(ws.compression_rate)
            w.f32(ws.steer_angle)
            w.f32(ws.angular_velocity)
            w.f32(ws.normal_load)
            w.f32(ws.slip_ratio)
            w.f32(ws.slip_angle)
            w.f32(ws.longitudinal_force)
            w.f32(ws.lateral_force)
            w.f32(ws.aligning_torque)
            w.boolean(ws.grounded)
            w.vec3(ws.contact_point)
            w.vec3(ws.contact_normal)
            w.f32(ws.bristle_x)
            w.f32(ws.bristle_y)
        w.end()

    def deserialize(mut self, mut r: BlobReader) raises:
        """Restores bit-exact `VehicleChassis` state from `BlobReader`."""
        _ = r.enter(VEHICLE_BLOB_TAG, VEHICLE_BLOB_VERSION)
        self.mass = r.f32()
        self.inv_mass = r.f32()
        self.inertia = r.vec3()
        self.inv_inertia = r.vec3()
        self.pos = r.vec3()
        self.rot = r.quat()
        self.linear_vel = r.vec3()
        self.angular_vel = r.vec3()
        self.anti_roll_bar_front = r.f32()
        self.anti_roll_bar_rear = r.f32()
        self.roll_center_blend = r.f32()
        self.drag_coeff = r.f32()
        self.downforce_coeff = r.f32()
        self.engine_rpm = r.f32()
        self.idle_rpm = r.f32()
        self.redline_rpm = r.f32()
        self.current_gear = r.integer()
        var n_gears = r.integer()
        if r.failed or n_gears < 1 or n_gears > 64:
            raise Error("vehicle blob: invalid gear count")
        self.gear_ratios.clear()
        for _ in range(n_gears):
            self.gear_ratios.append(r.f32())
        self.final_drive = r.f32()
        self.lsd_preload = r.f32()
        self.lsd_power_lock = r.f32()
        self.lsd_coast_lock = r.f32()
        self.wheel_inertia = r.f32()
        self.throttle_input = r.f32()
        self.brake_input = r.f32()
        self.steer_input = r.f32()

        var tp = PacejkaParams()
        tp.B = r.f32()
        tp.C = r.f32()
        tp.D = r.f32()
        tp.E = r.f32()
        tp.load_sensitivity = r.f32()
        tp.nominal_load = r.f32()
        tp.brush_stiffness = r.f32()
        tp.pneumatic_trail_0 = r.f32()
        tp.kappa_peak = r.f32()
        tp.alpha_peak = r.f32()
        tp.relaxation_len_x = r.f32()
        tp.relaxation_len_y = r.f32()
        self.tire_model = PacejkaTire(tp)

        var n_wheels = r.integer()
        if r.failed or n_wheels < 0 or n_wheels > 64:
            raise Error("vehicle blob: invalid wheel count")
        self.wheels.clear()
        for _ in range(n_wheels):
            var wc = WheelConfig()
            wc.offset = r.vec3()
            wc.radius = r.f32()
            wc.width = r.f32()
            wc.spring_k = r.f32()
            wc.damper_c_bump = r.f32()
            wc.damper_c_rebound = r.f32()
            wc.suspension_rest_len = r.f32()
            wc.max_travel = r.f32()
            wc.steered = r.boolean()
            wc.driven = r.boolean()
            wc.brake_factor = r.f32()
            self.wheels.append(wc)

        var n_states = r.integer()
        if r.failed or n_states < 0 or n_states > 64 or n_states != n_wheels:
            raise Error("vehicle blob: invalid wheel state count")
        self.wheel_states.clear()
        for _ in range(n_states):
            var ws = WheelState()
            ws.compression = r.f32()
            ws.compression_rate = r.f32()
            ws.steer_angle = r.f32()
            ws.angular_velocity = r.f32()
            ws.normal_load = r.f32()
            ws.slip_ratio = r.f32()
            ws.slip_angle = r.f32()
            ws.longitudinal_force = r.f32()
            ws.lateral_force = r.f32()
            ws.aligning_torque = r.f32()
            ws.grounded = r.boolean()
            ws.contact_point = r.vec3()
            ws.contact_normal = r.vec3()
            ws.bristle_x = r.f32()
            ws.bristle_y = r.f32()
            self.wheel_states.append(ws)
        r.leave()
