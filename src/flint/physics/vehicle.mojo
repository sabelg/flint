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
from std.math import sqrt, max, min, abs
from flint.core.math import Vec3, Quat
from flint.core.detmath import det_sin, det_cos, det_atan2
from flint.physics.tires import PacejkaTire
from flint.physics.bvh import Bvh8MeshCollider


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
def _implicit_gyro_step(wb_in: Vec3, inertia: Vec3, dt: Float32) -> Vec3:
    """1-step Implicit Newton-Raphson solver for `I * (w^+ - w) + dt * (w^+ x (I * w^+)) = 0`."""
    var ix = inertia.x
    var iy = inertia.y
    var iz = inertia.z
    var fx = dt * (iz - iy) * wb_in.y * wb_in.z
    var fy = dt * (ix - iz) * wb_in.z * wb_in.x
    var fz = dt * (iy - ix) * wb_in.x * wb_in.y

    var j00 = ix
    var j01 = dt * (iz - iy) * wb_in.z
    var j02 = dt * (iz - iy) * wb_in.y
    var j10 = dt * (ix - iz) * wb_in.z
    var j11 = iy
    var j12 = dt * (ix - iz) * wb_in.x
    var j20 = dt * (iy - ix) * wb_in.y
    var j21 = dt * (iy - ix) * wb_in.x
    var j22 = iz

    var c00 = j11 * j22 - j12 * j21
    var c01 = j12 * j20 - j10 * j22
    var c02 = j10 * j21 - j11 * j20
    var det = j00 * c00 + j01 * c01 + j02 * c02
    if abs(det) <= Float32(1.0e-18):
        return wb_in
    var inv_det = Float32(1.0) / det
    var c10 = j02 * j21 - j01 * j22
    var c11 = j00 * j22 - j02 * j20
    var c12 = j01 * j20 - j00 * j21
    var c20 = j01 * j12 - j02 * j11
    var c21 = j02 * j10 - j00 * j12
    var c22 = j00 * j11 - j01 * j10
    return Vec3(
        wb_in.x - (c00 * fx + c10 * fy + c20 * fz) * inv_det,
        wb_in.y - (c01 * fx + c11 * fy + c21 * fz) * inv_det,
        wb_in.z - (c02 * fx + c12 * fy + c22 * fz) * inv_det,
    )


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
        var num_wheels = len(self.wheels)
        var max_steer_rad = Float32(0.55)
        var s_clamped = max(Float32(-1.0), min(Float32(1.0), steer))
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
        if dt <= Float32(1.0e-7):
            return
        var th = throttle if throttle != Float32(0.0) else self.throttle_input
        var br = brake if brake != Float32(0.0) else self.brake_input
        var st = steer if steer != Float32(0.0) else self.steer_input
        var sub_dt = Float32(1.0) / max(Float32(60.0), substep_hz)
        var steps = max(1, Int(dt / sub_dt + Float32(0.5)))
        var actual_sub_dt = dt / Float32(steps)

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
        if dt <= Float32(1.0e-7):
            return
        var th = throttle if throttle != Float32(0.0) else self.throttle_input
        var br = brake if brake != Float32(0.0) else self.brake_input
        var st = steer if steer != Float32(0.0) else self.steer_input
        var sub_dt = Float32(1.0) / max(Float32(60.0), substep_hz)
        var steps = max(1, Int(dt / sub_dt + Float32(0.5)))
        var actual_sub_dt = dt / Float32(steps)

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
        var num_wheels = len(self.wheels)
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
                    self.wheel_states[i].compression_rate = (new_comp - old_comp) / dt
                else:
                    var hub_vel = self.linear_vel + self.angular_vel.cross(hub_r)
                    self.wheel_states[i].compression_rate = -hub_vel.y
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
        var num_wheels = len(self.wheels)
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
                    self.wheel_states[i].compression_rate = (new_comp - old_comp) / dt
                else:
                    var hub_vel = self.linear_vel + self.angular_vel.cross(hub_r)
                    self.wheel_states[i].compression_rate = -hub_vel.dot(n)
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
        var num_wheels = len(self.wheels)
        if num_wheels < 4:
            return

        var fwd = self.rot.rotate(Vec3(0, 0, 1)).normalized()
        var up = self.rot.rotate(Vec3(0, 1, 0)).normalized()
        var right = self.rot.rotate(Vec3(1, 0, 0)).normalized()

        # 1. Aerodynamic drag & downforce
        var fwd_speed = self.linear_vel.dot(fwd)
        var speed = self.linear_vel.length()
        var f_drag = -self.linear_vel * (Float32(0.5) * Float32(1.225) * self.drag_coeff * Float32(2.2) * speed)
        var f_down = -up * (Float32(0.5) * Float32(1.225) * self.downforce_coeff * Float32(2.2) * fwd_speed * fwd_speed)

        var total_force = Vec3(0, -9.81 * self.mass, 0) + f_drag + f_down
        var total_torque = Vec3(0, 0, 0)

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
            var comp = self.wheel_states[i].compression
            var crate = self.wheel_states[i].compression_rate
            var f_spring = cfg.spring_k * comp
            var c_damp = cfg.damper_c_bump if crate >= Float32(0.0) else cfg.damper_c_rebound
            var f_damper = c_damp * crate

            var arb_force = Float32(0.0)
            if i == 0:
                arb_force = arb_f
            elif i == 1:
                arb_force = -arb_f
            elif i == 2:
                arb_force = arb_r
            elif i == 3:
                arb_force = -arb_r

            var f_norm = max(Float32(0.0), f_spring + f_damper + arb_force)
            self.wheel_states[i].normal_load = f_norm

            var hub_r = self.rot.rotate(cfg.offset)
            var susp_force = self.wheel_states[i].contact_normal * f_norm
            total_force = total_force + susp_force
            total_torque = total_torque + hub_r.cross(susp_force)

        # 5. Powertrain: engine RPM update, torque curve, and 1.5-way LSD torque split
        var gear_idx = max(0, min(len(self.gear_ratios) - 1, self.current_gear))
        var total_ratio = self.gear_ratios[gear_idx] * self.final_drive
        var driven_count = 0
        var driven_omega_sum = Float32(0.0)
        var left_driven = -1
        var right_driven = -1
        for i in range(num_wheels):
            if self.wheels[i].driven:
                driven_count += 1
                driven_omega_sum += self.wheel_states[i].angular_velocity
                if self.wheels[i].offset.x < Float32(0.0) and left_driven < 0:
                    left_driven = i
                elif self.wheels[i].offset.x >= Float32(0.0) and right_driven < 0:
                    right_driven = i

        var avg_omega = driven_omega_sum / Float32(max(1, driven_count))
        var wheel_rpm = avg_omega * Float32(9.5492966)
        var target_rpm = max(
            self.idle_rpm,
            min(
                self.redline_rpm,
                abs(wheel_rpm * total_ratio) + max(Float32(0.0), throttle) * Float32(650.0),
            ),
        )
        self.engine_rpm += (target_rpm - self.engine_rpm) * min(Float32(1.0), dt * Float32(20.0))

        var eng_torque = self._eval_engine_torque(self.engine_rpm, throttle)
        var axle_torque = eng_torque * total_ratio

        var wheel_drive_torque = Array[Float32, 8](fill=0.0)
        if left_driven >= 0 and right_driven >= 0 and left_driven < 8 and right_driven < 8:
            var lock_factor = self.lsd_power_lock if axle_torque >= Float32(0.0) else self.lsd_coast_lock
            var max_lock = self.lsd_preload + lock_factor * abs(axle_torque)
            var d_omega = self.wheel_states[left_driven].angular_velocity - self.wheel_states[right_driven].angular_velocity
            var t_lsd = max(-max_lock, min(max_lock, Float32(45.0) * d_omega))
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
            var raw_arm = self.wheel_states[i].contact_point - self.pos
            var patch_vel = self.linear_vel + self.angular_vel.cross(raw_arm)

            var sa = self.wheel_states[i].steer_angle
            var c_sa = det_cos(sa)
            var s_sa = det_sin(sa)
            var steer_fwd = fwd * c_sa + right * s_sa
            var wheel_fwd = (steer_fwd - n_contact * steer_fwd.dot(n_contact)).normalized()
            if wheel_fwd.length_squared() < Float32(0.25):
                wheel_fwd = steer_fwd.normalized()
            var wheel_lat = n_contact.cross(wheel_fwd).normalized()

            var v_long = patch_vel.dot(wheel_fwd)
            var v_lat = patch_vel.dot(wheel_lat)

            var r = cfg.radius
            var omega = self.wheel_states[i].angular_velocity
            if cfg.driven and i < 8:
                omega += (wheel_drive_torque[i] / w_inertia) * dt

            var wheel_brake_t = max_brake_torque * max(Float32(0.0), cfg.brake_factor)
            if brake_clamped > Float32(0.001) and abs(v_long) <= Float32(1.5):
                var pre_decel = (wheel_brake_t / w_inertia) * dt
                if abs(omega) <= pre_decel:
                    omega = Float32(0.0)
                elif omega > Float32(0.0):
                    omega -= pre_decel
                else:
                    omega += pre_decel

            var tire_v = omega * r
            var slip_ratio = (tire_v - v_long) / max(Float32(0.1), abs(v_long))
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
            var fx = forces[0]
            var fy = forces[1]
            var mz = forces[2]

            var reaction_torque = -fx * r
            var new_omega = omega + (reaction_torque / w_inertia) * dt
            if brake_clamped > Float32(0.001):
                var decel = (wheel_brake_t / w_inertia) * dt
                if abs(v_long) > Float32(1.5):
                    var abs_floor = (Float32(1.0) - Float32(0.15) * brake_clamped) * (abs(v_long) / max(Float32(0.05), r))
                    if abs(new_omega) > abs_floor:
                        if new_omega > Float32(0.0):
                            new_omega = max(abs_floor, new_omega - decel)
                        else:
                            new_omega = min(-abs_floor, new_omega + decel)
                else:
                    if abs(new_omega) <= decel:
                        new_omega = Float32(0.0)
                    elif new_omega > Float32(0.0):
                        new_omega -= decel
                    else:
                        new_omega += decel
            self.wheel_states[i].angular_velocity = new_omega
            self.wheel_states[i].longitudinal_force = fx
            self.wheel_states[i].lateral_force = fy
            self.wheel_states[i].aligning_torque = mz

            var tire_force = wheel_fwd * fx + wheel_lat * fy
            total_force = total_force + tire_force

            # Apply tire force at (contact_point - self.pos) with roll_center_blend
            var arm_vert = up * raw_arm.dot(up)
            var force_arm = (raw_arm - arm_vert) + arm_vert * self.roll_center_blend
            total_torque = total_torque + force_arm.cross(tire_force) + n_contact * mz

            # Spinning wheel gyroscopic precession reaction: -omega_chassis x L_wheel
            var wheel_spin_l = -wheel_lat * (w_inertia * new_omega)
            total_torque = total_torque - self.angular_vel.cross(wheel_spin_l)

        # 7. Rigid body integration with implicit chassis gyroscopic torque & SO(3) attitude update
        self.linear_vel = self.linear_vel + total_force * (self.inv_mass * dt)
        self.pos = self.pos + self.linear_vel * dt

        var inv_rot = self.rot.conjugate()
        var wb_cur = inv_rot.rotate(self.angular_vel)
        var tau_local = inv_rot.rotate(total_torque)
        var wb_forced = Vec3(
            wb_cur.x + tau_local.x * self.inv_inertia.x * dt,
            wb_cur.y + tau_local.y * self.inv_inertia.y * dt,
            wb_cur.z + tau_local.z * self.inv_inertia.z * dt,
        )
        var wb_next = _implicit_gyro_step(wb_forced, self.inertia, dt)
        self.angular_vel = self.rot.rotate(wb_next)
        self.rot = (Quat.from_scaled_angle(self.angular_vel * dt) * self.rot).normalized()
