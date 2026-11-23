"""6-DOF Aerospace, Variable-Geometry Mecha Flight & SIMD8 ZEM-APN Missile Swarm (`aero6dof.mojo`, v4).

First-class flight dynamics & guidance module for *Robotech* / high-speed aerospace simulation:
1. `Aero6DofConfig`, `Aero6DofControls`, & `Aero6DofState`:
   - Continuous morphing mass `m(u, t) = lerp(m_0, m_1, u) + m_fuel(t)` and principal body
     inertia `I(u) = lerp(I_0, I_1, u)` across Fighter (`u = 0`) <-> GERWALK/Battroid (`u = 1`).
   - ISA scale-height atmosphere `rho(h) = rho_0 * exp(-h / H)` and Mach speed of sound `a(h)`.
   - Chebyshev / McCormick ground-effect lift cushion & induced-drag reduction `G_E(h_AGL, b)`.
   - Transonic Prandtl-Glauert compressibility & Lock wave-drag rise `C_{D,wave}(M)` across Mach `0.85..1.25`.
   - Full-360° `[-pi, +pi]` post-stall double-angle polar `C_L(alpha), C_D(alpha)` supporting
     high-alpha maneuvers (Pugachev's Cobra at `alpha = 90..110 deg`!) without singularity.
   - True Newtonian `a = F / m(t)`, off-CoM GERWALK twin vectoring hover-nozzle moments,
     2-step Implicit Newton-Raphson gyroscopic + TVC/RCS Euler solver
     `I(u) * w_dot + w x (I(u) * w) = tau`, and `detmath` SO(3) attitude integration.
2. `step_zem_apn_simd8`:
   - Pure 8-wide `SIMD[DType.float32, 8]` Zero-Effort-Miss Augmented Proportional Navigation
     (ZEM-APN) + decaying helical *Itano Circus* corkscrew weave kernel.
"""
from std.math import sqrt, abs, min, max, isfinite, isnan
from std.memory import bitcast
from flint.core.math import Vec3, Quat
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.core.detmath import (
    det_sin,
    det_cos,
    det_atan2,
    det_exp,
    set_sim_float_mode,
    PI,
)
from flint.physics.solver_bodies import LANES, F8, V8

comptime U8 = SIMD[DType.uint32, LANES]
comptime F64_8 = SIMD[DType.float64, LANES]
comptime F32_FINITE_MAX: Float32 = 3.4028234e38


@always_inline
def _aero_f32_safe(v: Float32, fallback: Float32 = 0.0) -> Float32:
    return v if isfinite(v) else fallback


@always_inline
def _aero_vec3_safe(v: Vec3) -> Vec3:
    return Vec3(
        v.x if isfinite(v.x) else Float32(0.0),
        v.y if isfinite(v.y) else Float32(0.0),
        v.z if isfinite(v.z) else Float32(0.0),
    )


@always_inline
def _aero_quat_safe(q: Quat) -> Quat:
    if isfinite(q.x) and isfinite(q.y) and isfinite(q.z) and isfinite(q.w):
        var n = q.normalized()
        if isfinite(n.x) and isfinite(n.y) and isfinite(n.z) and isfinite(n.w):
            return n
    return Quat()


@always_inline
def _aero_f8_isfinite(v: F8) -> SIMD[DType.bool, LANES]:
    return (bitcast[DType.uint32](v) & U8(0x7F800000)).ne(U8(0x7F800000))


@always_inline
def _aero_f8_safe(v: F8, fallback: F8 = F8(0.0)) -> F8:
    return _aero_f8_isfinite(v).select(v, fallback)


@always_inline
def _aero_v8_safe(v: V8, fallback: V8 = V8.zero()) -> V8:
    return V8(
        _aero_f8_safe(v.x, fallback.x),
        _aero_f8_safe(v.y, fallback.y),
        _aero_f8_safe(v.z, fallback.z),
    )


@always_inline
def _f64_to_f8_finite(v: F64_8) -> F8:
    var lim = F64_8(Float64(F32_FINITE_MAX))
    var is_nan = v.ne(v)
    var safe = is_nan.select(F64_8(0.0), max(-lim, min(lim, v)))
    return safe.cast[DType.float32]()


@always_inline
def _aero_f64_to_f32_finite(v: Float64, fallback: Float32 = 0.0) -> Float32:
    if not isfinite(v):
        return fallback
    var lim = Float64(F32_FINITE_MAX)
    if v > lim:
        return F32_FINITE_MAX
    if v < -lim:
        return -F32_FINITE_MAX
    return Float32(v)


@always_inline
def _aero_vec3_len_safe(v: Vec3) -> Float32:
    var x = Float64(_aero_f32_safe(v.x))
    var y = Float64(_aero_f32_safe(v.y))
    var z = Float64(_aero_f32_safe(v.z))
    return _aero_f64_to_f32_finite(sqrt(max(Float64(0.0), x * x + y * y + z * z)))

comptime AERO_CFG_TAG: UInt32 = fourcc("AROC")
comptime AERO_CTRL_TAG: UInt32 = fourcc("AROK")
comptime AERO_STATE_TAG: UInt32 = fourcc("AROS")
comptime AERO_BLOB_VERSION: Int = 1


comptime ISA_RHO0: Float32 = 1.225
comptime ISA_SCALE_HEIGHT_M: Float32 = 8500.0
comptime ISA_T0_K: Float32 = 288.15
comptime ISA_LAPSE_K_PER_M: Float32 = 0.0065
comptime ISA_T_STRATO_K: Float32 = 216.65


@always_inline
def eval_isa_atmosphere(altitude_m: Float32) -> Tuple[Float32, Float32]:
    """Returns `(air_density_kg_m3, speed_of_sound_m_s)` at geometric altitude `altitude_m`."""
    if not isfinite(altitude_m):
        if not isnan(altitude_m) and altitude_m > Float32(0.0):
            return (Float32(0.0), sqrt(Float32(1.4 * 287.058) * ISA_T_STRATO_K))
        return (ISA_RHO0, sqrt(Float32(1.4 * 287.058) * ISA_T0_K))
    var h = max(Float32(0.0), altitude_m)
    var rho = ISA_RHO0 * det_exp(-h / ISA_SCALE_HEIGHT_M)
    var temp_k = max(ISA_T_STRATO_K, ISA_T0_K - ISA_LAPSE_K_PER_M * h)
    var a_sound = sqrt(Float32(1.4 * 287.058) * temp_k)
    return (rho, a_sound)


@always_inline
def eval_ground_effect(h_agl: Float32, wingspan: Float32) -> Tuple[Float32, Float32]:
    """Chebyshev / McCormick ground-effect model.
    Returns `(ge_lift_factor, ge_induced_drag_factor)` for height-above-ground `h_agl` and span `wingspan`."""
    if not isfinite(h_agl):
        if not isnan(h_agl) and h_agl > Float32(0.0):
            return (Float32(1.0), Float32(1.0))
        return (Float32(1.35), Float32(0.0))
    var b = max(Float32(0.5), _aero_f32_safe(wingspan, Float32(10.0)))
    var hb = min(Float32(1.0e6), max(Float32(0.02), h_agl)) / b
    var hb16 = Float32(16.0) * hb * hb
    var ind_drag_mult = hb16 / (Float32(1.0) + hb16)
    var cushion = Float32(1.0) / (Float32(1.0) + hb16)
    # Shifted Chebyshev T_2 cushion near ground (hb < 1)
    var s = max(Float32(0.0), Float32(1.0) - hb)
    var t2_norm = Float32(0.5) * (Float32(2.0) * s * s - Float32(1.0) + Float32(1.0))
    var lift_mult = Float32(1.0) + Float32(0.28) * cushion + Float32(0.07) * t2_norm
    return (lift_mult, ind_drag_mult)


@always_inline
def eval_transonic_wave_drag(mach: Float32, cd_wave_peak: Float32 = 0.048) -> Tuple[Float32, Float32]:
    """Evaluates Prandtl-Glauert / Ackeret compressibility lift factor and Lock transonic
    wave-drag rise `C_{D,wave}(M)` across `Mach 0.85 .. 1.25`.
    Returns `(pg_lift_factor, cd_wave)`."""
    if not isfinite(mach):
        if not isnan(mach) and mach > Float32(0.0):
            return (Float32(0.0), Float32(0.0))
        return (Float32(1.0), Float32(0.0))
    var safe_peak = max(Float32(0.0), _aero_f32_safe(cd_wave_peak, Float32(0.048)))
    var m = min(Float32(1.0e6), max(Float32(0.0), mach))

    # 1. Compressibility lift-slope factor (smoothly capped across sonic M = 1.0)
    var pg_lift: Float32
    if m < Float32(0.85):
        pg_lift = Float32(1.0) / sqrt(max(Float32(0.2775), Float32(1.0) - m * m))
    elif m <= Float32(1.25):
        # Smooth transonic bridge from M=0.85 (1.898) through sonic peak (~1.95) to M=1.25 (1.333)
        var u = (m - Float32(0.85)) / Float32(0.40)
        var w = u * u * (Float32(3.0) - Float32(2.0) * u)
        var bump = Float32(0.12) * Float32(4.0) * u * (Float32(1.0) - u)
        pg_lift = (Float32(1.0) - w) * Float32(1.8983) + w * Float32(1.3333) + bump
    else:
        pg_lift = Float32(1.0) / sqrt(max(Float32(0.5625), m * m - Float32(1.0)))

    # 2. Transonic & supersonic wave drag C_{D,wave}(M)
    var cd_wave = Float32(0.0)
    if m > Float32(0.85) and m <= Float32(1.08):
        var t = (m - Float32(0.85)) / Float32(0.23)
        var s = t * t * (Float32(3.0) - Float32(2.0) * t)
        cd_wave = safe_peak * s
    elif m > Float32(1.08) and m <= Float32(1.25):
        var t = (m - Float32(1.08)) / Float32(0.17)
        var s = t * t * (Float32(3.0) - Float32(2.0) * t)
        cd_wave = safe_peak * (Float32(1.0) - Float32(0.22) * s)
    elif m > Float32(1.25):
        var sup_decay = Float32(0.75) / sqrt(max(Float32(0.5625), m * m - Float32(1.0)))
        cd_wave = safe_peak * Float32(0.78) * min(Float32(1.0), sup_decay)

    return (_aero_f32_safe(pg_lift, Float32(1.0)), _aero_f32_safe(cd_wave, Float32(0.0)))


@always_inline
def eval_post_stall_polar(
    alpha_rad: Float32,
    cl_alpha: Float32,
    cl0: Float32,
    cd0: Float32,
    stall_alpha: Float32,
    aspect_ratio: Float32,
    pg_mult: Float32,
    ge_lift: Float32,
    ge_ind_drag: Float32,
    cd_wave: Float32,
) -> Tuple[Float32, Float32, Float32]:
    """Full-360° `[-pi, +pi]` blended attached + post-stall double-angle Hoerner flat-plate polar.
    Returns `(C_L, C_D, C_m_alpha)` supporting Pugachev's Cobra (`alpha ~ 90..120 deg`)."""
    var safe_alpha = _aero_f32_safe(alpha_rad, Float32(0.0))
    var safe_stall = max(Float32(0.05), _aero_f32_safe(stall_alpha, Float32(0.30)))
    var safe_cl_alpha = _aero_f32_safe(cl_alpha, Float32(5.2))
    var safe_cl0 = _aero_f32_safe(cl0, Float32(0.08))
    var safe_cd0 = _aero_f32_safe(cd0, Float32(0.02))
    var safe_ar = max(Float32(0.5), _aero_f32_safe(aspect_ratio, Float32(5.0)))
    var safe_pg = _aero_f32_safe(pg_mult, Float32(1.0))
    var safe_ge_l = _aero_f32_safe(ge_lift, Float32(1.0))
    var safe_ge_d = _aero_f32_safe(ge_ind_drag, Float32(1.0))
    var safe_cd_w = _aero_f32_safe(cd_wave, Float32(0.0))

    var abs_a = abs(safe_alpha)
    var s_a = det_sin(safe_alpha)
    var c_a = det_cos(safe_alpha)
    var sin_2a = Float32(2.0) * s_a * c_a

    # Attached regime (clamped linear input so pre-blend arithmetic never overflows at large alpha)
    var a_lin = max(-safe_stall * Float32(1.6), min(safe_stall * Float32(1.6), safe_alpha))
    var cl_att = (safe_cl0 + safe_cl_alpha * safe_pg * a_lin) * safe_ge_l
    var k_ind = Float32(1.0) / max(Float32(1.5), PI * Float32(0.82) * safe_ar)
    var cd_att = safe_cd0 + safe_cd_w + k_ind * safe_ge_d * cl_att * cl_att

    # Post-stall double-angle vortex/flat-plate regime
    var cl_post = Float32(1.18) * sin_2a * safe_ge_l
    var cd_post = safe_cd0 + safe_cd_w + Float32(1.85) * s_a * s_a

    # Smooth C1 stall transition over [stall_alpha, stall_alpha + 0.22]
    var t_stall = max(Float32(0.0), min(Float32(1.0), (abs_a - safe_stall) / Float32(0.22)))
    var w_stall = t_stall * t_stall * (Float32(3.0) - Float32(2.0) * t_stall)

    var cl = (Float32(1.0) - w_stall) * cl_att + w_stall * cl_post
    var cd = (Float32(1.0) - w_stall) * cd_att + w_stall * cd_post
    # In right-handed (+X right, +Y up, +Z nose-forward), +tau_x pitches nose DOWN toward +Z,
    # restoring positive alpha > 0 back toward 0 (weathervane static longitudinal stability).
    var cm = Float32(0.65) * s_a
    return (_aero_f32_safe(cl, Float32(0.0)), _aero_f32_safe(cd, safe_cd0), _aero_f32_safe(cm, Float32(0.0)))


struct Aero6DofConfig(Copyable, ImplicitlyCopyable, Movable):
    """Variable-geometry aerospace / mecha configuration (`u = 0`: Fighter, `u = 1`: GERWALK/Battroid)."""
    var mass_0: Float32
    var mass_1: Float32
    var inertia_0: Vec3
    var inertia_1: Vec3
    var wingspan_0: Float32
    var wingspan_1: Float32
    var wing_area_0: Float32
    var wing_area_1: Float32
    var cl_alpha: Float32
    var cl0: Float32
    var cd0_0: Float32
    var cd0_1: Float32
    var cd_wave_peak: Float32
    var stall_alpha: Float32
    var max_afterburner_thrust: Float32
    var sfc_kg_per_n_s: Float32
    var tvc_arm_z: Float32
    var max_tvc_angle_rad: Float32
    var aero_ctrl_torque_gain: Vec3
    var aero_rate_damping: Vec3
    var rcs_max_torque: Vec3
    var gerwalk_nozzle_left: Vec3
    var gerwalk_nozzle_right: Vec3
    var max_hover_nozzle_thrust: Float32

    def __init__(out self):
        self.mass_0 = 13200.0
        self.mass_1 = 13200.0
        self.inertia_0 = Vec3(28000.0, 82000.0, 19000.0)
        self.inertia_1 = Vec3(48000.0, 54000.0, 39000.0)
        self.wingspan_0 = 14.78
        self.wingspan_1 = 8.40
        self.wing_area_0 = 28.5
        self.wing_area_1 = 18.0
        self.cl_alpha = 5.2
        self.cl0 = 0.08
        self.cd0_0 = 0.018
        self.cd0_1 = 0.14
        self.cd_wave_peak = 0.048
        self.stall_alpha = 0.30
        self.max_afterburner_thrust = 230000.0
        self.sfc_kg_per_n_s = 0.0
        self.tvc_arm_z = -5.8
        self.max_tvc_angle_rad = 0.38
        self.aero_ctrl_torque_gain = Vec3(18.0, 10.0, 14.0)
        self.aero_rate_damping = Vec3(42000.0, 38000.0, 28000.0)
        self.rcs_max_torque = Vec3(95000.0, 85000.0, 65000.0)
        self.gerwalk_nozzle_left = Vec3(-1.85, -0.95, 0.35)
        self.gerwalk_nozzle_right = Vec3(1.85, -0.95, 0.35)
        self.max_hover_nozzle_thrust = 115000.0

    def serialize(self, mut w: BlobWriter):
        w.begin(AERO_CFG_TAG, AERO_BLOB_VERSION)
        w.f32(self.mass_0)
        w.f32(self.mass_1)
        w.vec3(self.inertia_0)
        w.vec3(self.inertia_1)
        w.f32(self.wingspan_0)
        w.f32(self.wingspan_1)
        w.f32(self.wing_area_0)
        w.f32(self.wing_area_1)
        w.f32(self.cl_alpha)
        w.f32(self.cl0)
        w.f32(self.cd0_0)
        w.f32(self.cd0_1)
        w.f32(self.cd_wave_peak)
        w.f32(self.stall_alpha)
        w.f32(self.max_afterburner_thrust)
        w.f32(self.sfc_kg_per_n_s)
        w.f32(self.tvc_arm_z)
        w.f32(self.max_tvc_angle_rad)
        w.vec3(self.aero_ctrl_torque_gain)
        w.vec3(self.aero_rate_damping)
        w.vec3(self.rcs_max_torque)
        w.vec3(self.gerwalk_nozzle_left)
        w.vec3(self.gerwalk_nozzle_right)
        w.f32(self.max_hover_nozzle_thrust)
        w.end()

    def deserialize(mut self, mut r: BlobReader) raises:
        _ = r.enter(AERO_CFG_TAG, AERO_BLOB_VERSION)
        self.mass_0 = r.f32()
        self.mass_1 = r.f32()
        self.inertia_0 = r.vec3()
        self.inertia_1 = r.vec3()
        self.wingspan_0 = r.f32()
        self.wingspan_1 = r.f32()
        self.wing_area_0 = r.f32()
        self.wing_area_1 = r.f32()
        self.cl_alpha = r.f32()
        self.cl0 = r.f32()
        self.cd0_0 = r.f32()
        self.cd0_1 = r.f32()
        self.cd_wave_peak = r.f32()
        self.stall_alpha = r.f32()
        self.max_afterburner_thrust = r.f32()
        self.sfc_kg_per_n_s = r.f32()
        self.tvc_arm_z = r.f32()
        self.max_tvc_angle_rad = r.f32()
        self.aero_ctrl_torque_gain = r.vec3()
        self.aero_rate_damping = r.vec3()
        self.rcs_max_torque = r.vec3()
        self.gerwalk_nozzle_left = r.vec3()
        self.gerwalk_nozzle_right = r.vec3()
        self.max_hover_nozzle_thrust = r.f32()
        r.leave()


struct Aero6DofControls(Copyable, ImplicitlyCopyable, Movable):
    var throttle: Float32
    var pitch_cmd: Float32
    var yaw_cmd: Float32
    var roll_cmd: Float32
    var tvc_pitch: Float32
    var tvc_yaw: Float32
    var rcs_cmd: Vec3
    var morph_u_target: Float32
    var morph_rate: Float32
    var hover_left_throttle: Float32
    var hover_right_throttle: Float32
    var hover_tilt_rad: Float32
    var wind_vel: Vec3
    var ground_y: Float32
    var gravity: Float32

    def __init__(out self):
        self.throttle = 0.0
        self.pitch_cmd = 0.0
        self.yaw_cmd = 0.0
        self.roll_cmd = 0.0
        self.tvc_pitch = 0.0
        self.tvc_yaw = 0.0
        self.rcs_cmd = Vec3()
        self.morph_u_target = 0.0
        self.morph_rate = 2.0
        self.hover_left_throttle = 0.0
        self.hover_right_throttle = 0.0
        self.hover_tilt_rad = Float32(1.5707963)
        self.wind_vel = Vec3()
        self.ground_y = 0.0
        self.gravity = 9.81

    def serialize(self, mut w: BlobWriter):
        w.begin(AERO_CTRL_TAG, AERO_BLOB_VERSION)
        w.f32(self.throttle)
        w.f32(self.pitch_cmd)
        w.f32(self.yaw_cmd)
        w.f32(self.roll_cmd)
        w.f32(self.tvc_pitch)
        w.f32(self.tvc_yaw)
        w.vec3(self.rcs_cmd)
        w.f32(self.morph_u_target)
        w.f32(self.morph_rate)
        w.f32(self.hover_left_throttle)
        w.f32(self.hover_right_throttle)
        w.f32(self.hover_tilt_rad)
        w.vec3(self.wind_vel)
        w.f32(self.ground_y)
        w.f32(self.gravity)
        w.end()

    def deserialize(mut self, mut r: BlobReader) raises:
        _ = r.enter(AERO_CTRL_TAG, AERO_BLOB_VERSION)
        self.throttle = r.f32()
        self.pitch_cmd = r.f32()
        self.yaw_cmd = r.f32()
        self.roll_cmd = r.f32()
        self.tvc_pitch = r.f32()
        self.tvc_yaw = r.f32()
        self.rcs_cmd = r.vec3()
        self.morph_u_target = r.f32()
        self.morph_rate = r.f32()
        self.hover_left_throttle = r.f32()
        self.hover_right_throttle = r.f32()
        self.hover_tilt_rad = r.f32()
        self.wind_vel = r.vec3()
        self.ground_y = r.f32()
        self.gravity = r.f32()
        r.leave()


struct Aero6DofState(Copyable, ImplicitlyCopyable, Movable):
    var pos: Vec3
    var rot: Quat
    var linear_vel: Vec3
    var angular_vel: Vec3
    var morph_u: Float32
    var fuel_mass: Float32
    # Diagnostic telemetry
    var mach: Float32
    var alpha_rad: Float32
    var beta_rad: Float32
    var dynamic_pressure: Float32
    var cl: Float32
    var cd: Float32
    var cd_wave: Float32
    var ge_lift_factor: Float32
    var current_mass: Float32
    var current_inertia: Vec3
    var g_load: Float32

    def __init__(out self, pos: Vec3 = Vec3(0, 500, 0), vel: Vec3 = Vec3(0, 0, 120)):
        self.pos = pos
        self.rot = Quat()
        self.linear_vel = vel
        self.angular_vel = Vec3()
        self.morph_u = 0.0
        self.fuel_mass = 0.0
        self.mach = 0.0
        self.alpha_rad = 0.0
        self.beta_rad = 0.0
        self.dynamic_pressure = 0.0
        self.cl = 0.0
        self.cd = 0.0
        self.cd_wave = 0.0
        self.ge_lift_factor = 1.0
        self.current_mass = 13200.0
        self.current_inertia = Vec3(28000.0, 82000.0, 19000.0)
        self.g_load = 1.0

    def serialize(self, mut w: BlobWriter):
        w.begin(AERO_STATE_TAG, AERO_BLOB_VERSION)
        w.vec3(self.pos)
        w.quat(self.rot)
        w.vec3(self.linear_vel)
        w.vec3(self.angular_vel)
        w.f32(self.morph_u)
        w.f32(self.fuel_mass)
        w.f32(self.mach)
        w.f32(self.alpha_rad)
        w.f32(self.beta_rad)
        w.f32(self.dynamic_pressure)
        w.f32(self.cl)
        w.f32(self.cd)
        w.f32(self.cd_wave)
        w.f32(self.ge_lift_factor)
        w.f32(self.current_mass)
        w.vec3(self.current_inertia)
        w.f32(self.g_load)
        w.end()

    def deserialize(mut self, mut r: BlobReader) raises:
        _ = r.enter(AERO_STATE_TAG, AERO_BLOB_VERSION)
        self.pos = r.vec3()
        self.rot = r.quat()
        self.linear_vel = r.vec3()
        self.angular_vel = r.vec3()
        self.morph_u = r.f32()
        self.fuel_mass = r.f32()
        self.mach = r.f32()
        self.alpha_rad = r.f32()
        self.beta_rad = r.f32()
        self.dynamic_pressure = r.f32()
        self.cl = r.f32()
        self.cd = r.f32()
        self.cd_wave = r.f32()
        self.ge_lift_factor = r.f32()
        self.current_mass = r.f32()
        self.current_inertia = r.vec3()
        self.g_load = r.f32()
        r.leave()


@always_inline
def _solve_implicit_euler_rotation(wb0: Vec3, inertia: Vec3, tau_b: Vec3, h: Float32) -> Vec3:
    """2-iteration Implicit Newton-Raphson solver for coupled Euler rotational dynamics:
    `R(w^+) = I * (w^+ - w0) + h * (w^+ x (I * w^+) - tau_b) = 0`."""
    var wb_safe = _aero_vec3_safe(wb0)
    var tau_safe = _aero_vec3_safe(tau_b)
    var safe_h = min(Float32(1.0), max(Float32(0.0), _aero_f32_safe(h, Float32(0.0))))
    var ix = max(Float32(1.0e-3), _aero_f32_safe(inertia.x, Float32(1000.0)))
    var iy = max(Float32(1.0e-3), _aero_f32_safe(inertia.y, Float32(1000.0)))
    var iz = max(Float32(1.0e-3), _aero_f32_safe(inertia.z, Float32(1000.0)))

    var w = Vec3(
        wb_safe.x + (tau_safe.x / ix) * safe_h,
        wb_safe.y + (tau_safe.y / iy) * safe_h,
        wb_safe.z + (tau_safe.z / iz) * safe_h,
    )
    if not (isfinite(w.x) and isfinite(w.y) and isfinite(w.z)):
        return wb_safe

    comptime for _ in range(2):
        var rx = ix * (w.x - wb_safe.x) + safe_h * ((iz - iy) * w.y * w.z - tau_safe.x)
        var ry = iy * (w.y - wb_safe.y) + safe_h * ((ix - iz) * w.z * w.x - tau_safe.y)
        var rz = iz * (w.z - wb_safe.z) + safe_h * ((iy - ix) * w.x * w.y - tau_safe.z)

        var j00 = ix
        var j01 = safe_h * (iz - iy) * w.z
        var j02 = safe_h * (iz - iy) * w.y
        var j10 = safe_h * (ix - iz) * w.z
        var j11 = iy
        var j12 = safe_h * (ix - iz) * w.x
        var j20 = safe_h * (iy - ix) * w.y
        var j21 = safe_h * (iy - ix) * w.x
        var j22 = iz

        var c00 = j11 * j22 - j12 * j21
        var c01 = j12 * j20 - j10 * j22
        var c02 = j10 * j21 - j11 * j20
        var det = j00 * c00 + j01 * c01 + j02 * c02
        if not isfinite(det) or abs(det) <= Float32(1.0e-18):
            return w if (isfinite(w.x) and isfinite(w.y) and isfinite(w.z)) else wb_safe
        var inv_det = Float32(1.0) / det
        var c10 = j02 * j21 - j01 * j22
        var c11 = j00 * j22 - j02 * j20
        var c12 = j01 * j20 - j00 * j21
        var c20 = j01 * j12 - j02 * j11
        var c21 = j02 * j10 - j00 * j12
        var c22 = j00 * j11 - j01 * j10

        var w_next = Vec3(
            w.x - (c00 * rx + c10 * ry + c20 * rz) * inv_det,
            w.y - (c01 * rx + c11 * ry + c21 * rz) * inv_det,
            w.z - (c02 * rx + c12 * ry + c22 * rz) * inv_det,
        )
        if not (isfinite(w_next.x) and isfinite(w_next.y) and isfinite(w_next.z)):
            return wb_safe
        w = w_next
    return w


def eval_aero6dof_forces(
    mut state: Aero6DofState,
    cfg: Aero6DofConfig,
    ctrl: Aero6DofControls,
    h: Float32,
) -> Tuple[Vec3, Vec3]:
    """Evaluates aerodynamic, TVC, GERWALK hover-nozzle, and RCS forces & torques over step `h`.
    Updates `morph_u`, `fuel_mass`, `current_mass`, `current_inertia`, `g_load`, and telemetry
    without integrating `pos`, `rot`, `linear_vel`, or `angular_vel`.
    Returns `(f_non_grav_world, tau_total_body)`."""
    state.pos = _aero_vec3_safe(state.pos)
    state.linear_vel = _aero_vec3_safe(state.linear_vel)
    state.angular_vel = _aero_vec3_safe(state.angular_vel)
    state.rot = _aero_quat_safe(state.rot)
    if not isfinite(state.morph_u):
        state.morph_u = Float32(0.0)
    if not isfinite(state.fuel_mass):
        state.fuel_mass = Float32(0.0)
    var safe_h = min(Float32(1.0), max(Float32(0.0), _aero_f32_safe(h, Float32(0.0))))
    # 1. Morph blend u(t) and instantaneous mass & principal inertia I(u) = lerp(I_0, I_1, u)
    var u_tgt = max(Float32(0.0), min(Float32(1.0), _aero_f32_safe(ctrl.morph_u_target, state.morph_u)))
    var max_du = max(Float32(0.0), _aero_f32_safe(ctrl.morph_rate, Float32(0.0))) * safe_h
    var du = max(-max_du, min(max_du, u_tgt - state.morph_u))
    state.morph_u = max(Float32(0.0), min(Float32(1.0), state.morph_u + du))
    var u = state.morph_u
    var om_u = Float32(1.0) - u

    var m0 = max(Float32(1.0), _aero_f32_safe(cfg.mass_0, Float32(13200.0)))
    var m1 = max(Float32(1.0), _aero_f32_safe(cfg.mass_1, Float32(13200.0)))
    var dry_mass = om_u * m0 + u * m1
    var cur_mass = max(Float32(10.0), _aero_f32_safe(dry_mass + max(Float32(0.0), state.fuel_mass), m0))
    var cur_i = Vec3(
        max(Float32(1.0), om_u * _aero_f32_safe(cfg.inertia_0.x, Float32(28000.0)) + u * _aero_f32_safe(cfg.inertia_1.x, Float32(48000.0))),
        max(Float32(1.0), om_u * _aero_f32_safe(cfg.inertia_0.y, Float32(82000.0)) + u * _aero_f32_safe(cfg.inertia_1.y, Float32(54000.0))),
        max(Float32(1.0), om_u * _aero_f32_safe(cfg.inertia_0.z, Float32(19000.0)) + u * _aero_f32_safe(cfg.inertia_1.z, Float32(39000.0))),
    )
    var span = max(Float32(0.5), om_u * _aero_f32_safe(cfg.wingspan_0, Float32(14.78)) + u * _aero_f32_safe(cfg.wingspan_1, Float32(8.40)))
    var area = max(Float32(0.5), om_u * _aero_f32_safe(cfg.wing_area_0, Float32(28.5)) + u * _aero_f32_safe(cfg.wing_area_1, Float32(18.0)))
    var cd0 = max(Float32(0.0), om_u * _aero_f32_safe(cfg.cd0_0, Float32(0.018)) + u * _aero_f32_safe(cfg.cd0_1, Float32(0.14)))
    var aspect_ratio = _aero_f32_safe((span * span) / area, Float32(5.0))

    state.current_mass = cur_mass
    state.current_inertia = cur_i

    # 2. ISA Atmosphere & Ground Effect
    var isa = eval_isa_atmosphere(state.pos.y)
    var rho = isa[0]
    var a_sound = isa[1]
    var h_agl = _aero_f64_to_f32_finite(Float64(state.pos.y) - Float64(_aero_f32_safe(ctrl.ground_y)))
    var ge = eval_ground_effect(h_agl, span)
    var ge_lift = ge[0]
    var ge_ind_drag = ge[1]
    state.ge_lift_factor = ge_lift

    # 3. Body-frame airspeed, AoA alpha, sideslip beta, Mach, and dynamic pressure
    # Body frame convention: +X starboard-right, +Y dorsal-up, +Z nose-forward
    var inv_q = state.rot.conjugate()
    var wind = _aero_vec3_safe(ctrl.wind_vel)
    var v_air_w = Vec3(
        _aero_f64_to_f32_finite(Float64(state.linear_vel.x) - Float64(wind.x)),
        _aero_f64_to_f32_finite(Float64(state.linear_vel.y) - Float64(wind.y)),
        _aero_f64_to_f32_finite(Float64(state.linear_vel.z) - Float64(wind.z)),
    )
    var vb = _aero_vec3_safe(inv_q.rotate(v_air_w))
    var v_mag64 = sqrt(max(Float64(0.0), Float64(vb.x) * Float64(vb.x) + Float64(vb.y) * Float64(vb.y) + Float64(vb.z) * Float64(vb.z)))
    var mach = _aero_f64_to_f32_finite(v_mag64 / max(Float64(1.0), Float64(a_sound)))
    var q_dyn64 = Float64(0.5) * Float64(rho) * v_mag64 * v_mag64
    var q_dyn = _aero_f64_to_f32_finite(q_dyn64)
    state.mach = mach
    state.dynamic_pressure = q_dyn

    var f_aero_b = Vec3()
    var tau_aero_b = Vec3()
    var wb = _aero_vec3_safe(inv_q.rotate(state.angular_vel))

    if v_mag64 > Float64(0.05) and q_dyn64 > Float64(1.0e-5):
        var alpha = det_atan2(-vb.y, vb.z)
        var v_yz = _aero_f64_to_f32_finite(sqrt(max(Float64(0.0), Float64(vb.y) * Float64(vb.y) + Float64(vb.z) * Float64(vb.z))))
        var beta = det_atan2(-vb.x, max(Float32(1.0e-4), v_yz))
        state.alpha_rad = alpha
        state.beta_rad = beta

        var trans = eval_transonic_wave_drag(mach, cfg.cd_wave_peak)
        var pg_mult = trans[0]
        var cd_wave = trans[1]
        state.cd_wave = cd_wave

        var polar = eval_post_stall_polar(
            alpha, cfg.cl_alpha, cfg.cl0, cd0, cfg.stall_alpha,
            aspect_ratio, pg_mult, ge_lift, ge_ind_drag, cd_wave,
        )
        var cl = polar[0]
        var cd = polar[1]
        var cm_static = polar[2]
        state.cl = cl
        state.cd = cd

        # Wind axes in body space: drag opposes vb, lift perpendicular to vb in (Y, Z) symmetry plane
        var inv_vmag64 = Float64(1.0) / v_mag64
        var drag_dir_b = Vec3(
            Float32(-Float64(vb.x) * inv_vmag64),
            Float32(-Float64(vb.y) * inv_vmag64),
            Float32(-Float64(vb.z) * inv_vmag64),
        )
        var lift_dir_b = Vec3(0.0, det_cos(alpha), det_sin(alpha))
        var cy_side = -Float32(0.85) * det_sin(beta)

        var q_s64 = q_dyn64 * Float64(area)
        f_aero_b = Vec3(
            _aero_f64_to_f32_finite(Float64(drag_dir_b.x) * (q_s64 * Float64(cd)) + Float64(lift_dir_b.x) * (q_s64 * Float64(cl)) + q_s64 * Float64(cy_side)),
            _aero_f64_to_f32_finite(Float64(drag_dir_b.y) * (q_s64 * Float64(cd)) + Float64(lift_dir_b.y) * (q_s64 * Float64(cl))),
            _aero_f64_to_f32_finite(Float64(drag_dir_b.z) * (q_s64 * Float64(cd)) + Float64(lift_dir_b.z) * (q_s64 * Float64(cl))),
        )

        # Aerodynamic stability, control surfaces, and angular rate damping
        var chord = Float64(area) / max(Float64(0.5), Float64(span))
        var pitch_surf = max(Float32(-1.0), min(Float32(1.0), _aero_f32_safe(ctrl.pitch_cmd)))
        var yaw_surf = max(Float32(-1.0), min(Float32(1.0), _aero_f32_safe(ctrl.yaw_cmd)))
        var roll_surf = max(Float32(-1.0), min(Float32(1.0), _aero_f32_safe(ctrl.roll_cmd)))

        # Control surface authority fades smoothly post-stall while TVC/RCS remain 100% active
        var s_a_ctrl = det_sin(alpha)
        var surf_eff = Float64(1.0) / (Float64(1.0) + Float64(1.5) * Float64(s_a_ctrl) * Float64(s_a_ctrl))
        var damp_scale = min(Float64(2.5), Float64(rho) / Float64(ISA_RHO0)) * max(Float64(0.2), min(Float64(3.0), v_mag64 / Float64(100.0)))

        var gain_v = _aero_vec3_safe(cfg.aero_ctrl_torque_gain)
        var damp_v = _aero_vec3_safe(cfg.aero_rate_damping)
        var s_b = Float64(det_sin(beta))
        tau_aero_b = Vec3(
            _aero_f64_to_f32_finite(q_s64 * chord * Float64(cm_static) - q_s64 * Float64(gain_v.x) * surf_eff * Float64(pitch_surf) - Float64(damp_v.x) * damp_scale * Float64(wb.x)),
            _aero_f64_to_f32_finite(-q_s64 * Float64(span) * Float64(0.18) * s_b + q_s64 * Float64(gain_v.y) * surf_eff * Float64(yaw_surf) - Float64(damp_v.y) * damp_scale * Float64(wb.y)),
            _aero_f64_to_f32_finite(-q_s64 * Float64(span) * Float64(0.06) * s_b + q_s64 * Float64(gain_v.z) * surf_eff * Float64(roll_surf) - Float64(damp_v.z) * damp_scale * Float64(wb.z)),
        )
    else:
        state.alpha_rad = 0.0
        state.beta_rad = 0.0
        state.cl = 0.0
        state.cd = cd0
        state.cd_wave = 0.0

    # 4. Main Afterburner + Thrust Vectoring Control (TVC)
    var fuel_empty = (cfg.sfc_kg_per_n_s > Float32(0.0) and state.fuel_mass <= Float32(0.0)) or state.fuel_mass < Float32(0.0)
    var th_main = max(Float32(0.0), min(Float32(1.0), _aero_f32_safe(ctrl.throttle)))
    if fuel_empty:
        th_main = Float32(0.0)
    var main_thrust_mag = _aero_f32_safe(th_main * _aero_f32_safe(cfg.max_afterburner_thrust))

    # Deflect aft exhaust nozzle: positive pitch_cmd deflects thrust vector so aft tail pushes -Y, pitching nose +X up!
    var max_tvc = max(Float32(0.0), _aero_f32_safe(cfg.max_tvc_angle_rad, Float32(0.38)))
    var safe_tvc_p = _aero_f32_safe(ctrl.tvc_pitch)
    var safe_tvc_y = _aero_f32_safe(ctrl.tvc_yaw)
    var safe_pitch_cmd = _aero_f32_safe(ctrl.pitch_cmd)
    var safe_yaw_cmd = _aero_f32_safe(ctrl.yaw_cmd)
    var tvc_p = safe_tvc_p if safe_tvc_p != Float32(0.0) else safe_pitch_cmd * max_tvc
    var tvc_y = safe_tvc_y if safe_tvc_y != Float32(0.0) else safe_yaw_cmd * max_tvc
    tvc_p = max(-max_tvc, min(max_tvc, tvc_p))
    tvc_y = max(-max_tvc, min(max_tvc, tvc_y))

    var cp_t = det_cos(tvc_p)
    var sp_t = det_sin(tvc_p)
    var cy_t = det_cos(tvc_y)
    var sy_t = det_sin(tvc_y)
    # With r_tvc = (0, 0, tvc_arm_z < 0):
    # r_tvc x F_tvc = (0, 0, -|z|) x (Fx, Fy, Fz) = (+|z|*Fy, -|z|*Fx, 0).
    # Nose-up rotation (+Z -> +Y) is -X, so tvc_p > 0 pushes tail -Y (Fy < 0 => tau_x < 0, nose UP!).
    var f_tvc_b = _aero_vec3_safe(Vec3(
        -sy_t * cp_t * main_thrust_mag,
        -sp_t * main_thrust_mag,
        cy_t * cp_t * main_thrust_mag,
    ))
    var r_tvc = Vec3(0.0, 0.0, _aero_f32_safe(cfg.tvc_arm_z, Float32(-5.8)))
    var tau_tvc_b = _aero_vec3_safe(r_tvc.cross(f_tvc_b))

    # 5. Off-CoM GERWALK Twin Vectoring Hover Nozzles
    var th_hl = max(Float32(0.0), min(Float32(1.0), _aero_f32_safe(ctrl.hover_left_throttle)))
    var th_hr = max(Float32(0.0), min(Float32(1.0), _aero_f32_safe(ctrl.hover_right_throttle)))
    if fuel_empty:
        th_hl = Float32(0.0)
        th_hr = Float32(0.0)
    var hover_tilt = _aero_f32_safe(ctrl.hover_tilt_rad, Float32(1.5707963))
    var max_hover_thrust = max(Float32(0.0), _aero_f32_safe(cfg.max_hover_nozzle_thrust))
    var noz_dir_b = Vec3(0.0, det_sin(hover_tilt), det_cos(hover_tilt))
    var f_noz_l = _aero_vec3_safe(noz_dir_b * (th_hl * max_hover_thrust))
    var f_noz_r = _aero_vec3_safe(noz_dir_b * (th_hr * max_hover_thrust))
    var tau_noz_b = _aero_vec3_safe(
        _aero_vec3_safe(cfg.gerwalk_nozzle_left).cross(f_noz_l)
        + _aero_vec3_safe(cfg.gerwalk_nozzle_right).cross(f_noz_r)
    )

    # Fuel burn update (if configured)
    var safe_sfc = _aero_f32_safe(cfg.sfc_kg_per_n_s)
    if safe_sfc > Float32(0.0) and state.fuel_mass > Float32(0.0):
        var total_thrust_n = Float64(main_thrust_mag) + Float64(th_hl + th_hr) * Float64(max_hover_thrust)
        state.fuel_mass = max(Float32(0.0), _aero_f64_to_f32_finite(Float64(state.fuel_mass) - Float64(safe_sfc) * total_thrust_n * Float64(safe_h)))

    # 6. Reaction Control System (RCS) verniers (100% effective at 0 airspeed and in orbital vacuum)
    var rcs_in = _aero_vec3_safe(ctrl.rcs_cmd)
    var rcs_len2 = Float64(rcs_in.x) * Float64(rcs_in.x) + Float64(rcs_in.y) * Float64(rcs_in.y) + Float64(rcs_in.z) * Float64(rcs_in.z)
    if rcs_len2 == Float64(0.0) and q_dyn < Float32(250.0):
        rcs_in = Vec3(-safe_pitch_cmd, safe_yaw_cmd, _aero_f32_safe(ctrl.roll_cmd))
    var rcs_max_t = _aero_vec3_safe(cfg.rcs_max_torque)
    var tau_rcs_b = _aero_vec3_safe(Vec3(
        max(Float32(-1.0), min(Float32(1.0), rcs_in.x)) * rcs_max_t.x,
        max(Float32(-1.0), min(Float32(1.0), rcs_in.y)) * rcs_max_t.y,
        max(Float32(-1.0), min(Float32(1.0), rcs_in.z)) * rcs_max_t.z,
    ))

    var f_body_total = _aero_vec3_safe(f_aero_b + f_tvc_b + f_noz_l + f_noz_r)
    var f_non_grav_w = _aero_vec3_safe(state.rot.rotate(f_body_total))
    var f_len64 = sqrt(max(Float64(0.0), Float64(f_non_grav_w.x) * Float64(f_non_grav_w.x) + Float64(f_non_grav_w.y) * Float64(f_non_grav_w.y) + Float64(f_non_grav_w.z) * Float64(f_non_grav_w.z)))
    state.g_load = _aero_f64_to_f32_finite((f_len64 / Float64(cur_mass)) / Float64(9.81))
    var tau_total_b = _aero_vec3_safe(tau_aero_b + tau_tvc_b + tau_noz_b + tau_rcs_b)
    return (f_non_grav_w, tau_total_b)


def step_aero6dof(
    mut state: Aero6DofState,
    cfg: Aero6DofConfig,
    ctrl: Aero6DofControls,
    dt: Float32,
    substeps: Int = 4,
):
    """Advances a 6-DOF variable-geometry aerospace/mecha craft by `dt`."""
    set_sim_float_mode()
    if not (isfinite(state.pos.x) and isfinite(state.pos.y) and isfinite(state.pos.z)):
        state.pos = _aero_vec3_safe(state.pos)
    if not (isfinite(state.linear_vel.x) and isfinite(state.linear_vel.y) and isfinite(state.linear_vel.z)):
        state.linear_vel = _aero_vec3_safe(state.linear_vel)
    if not (isfinite(state.angular_vel.x) and isfinite(state.angular_vel.y) and isfinite(state.angular_vel.z)):
        state.angular_vel = _aero_vec3_safe(state.angular_vel)
    if not (isfinite(state.rot.x) and isfinite(state.rot.y) and isfinite(state.rot.z) and isfinite(state.rot.w)):
        state.rot = Quat()
    if not isfinite(state.morph_u):
        state.morph_u = Float32(0.0)
    if not isfinite(state.fuel_mass):
        state.fuel_mass = Float32(0.0)
    if not isfinite(state.mach):
        state.mach = Float32(0.0)
    if not isfinite(state.alpha_rad):
        state.alpha_rad = Float32(0.0)
    if not isfinite(state.beta_rad):
        state.beta_rad = Float32(0.0)
    if not isfinite(state.dynamic_pressure):
        state.dynamic_pressure = Float32(0.0)
    if not isfinite(state.cl):
        state.cl = Float32(0.0)
    if not isfinite(state.cd):
        state.cd = Float32(0.0)
    if not isfinite(state.cd_wave):
        state.cd_wave = Float32(0.0)
    if not isfinite(state.ge_lift_factor):
        state.ge_lift_factor = Float32(1.0)
    if not isfinite(state.current_mass):
        state.current_mass = max(Float32(10.0), _aero_f32_safe(cfg.mass_0, Float32(13200.0)))
    if not (isfinite(state.current_inertia.x) and isfinite(state.current_inertia.y) and isfinite(state.current_inertia.z)):
        state.current_inertia = Vec3(Float32(28000.0), Float32(82000.0), Float32(19000.0))
    if not isfinite(state.g_load):
        state.g_load = Float32(1.0)

    if not isfinite(dt) or dt <= Float32(1.0e-7):
        return
    var safe_dt = min(dt, Float32(1.0))
    var steps = max(1, min(64, substeps))
    var h = safe_dt / Float32(steps)
    var grav = _aero_f32_safe(ctrl.gravity, Float32(9.81))

    for _ in range(steps):
        var ft = eval_aero6dof_forces(state, cfg, ctrl, h)
        var f_non_grav_w = _aero_vec3_safe(ft[0])
        var tau_total_b = _aero_vec3_safe(ft[1])

        # 7. True Newtonian translational integration: a = F / m(t)
        var inv_m = Float32(1.0) / max(Float32(10.0), _aero_f32_safe(state.current_mass, Float32(1000.0)))
        var accel_w = f_non_grav_w * inv_m + Vec3(0.0, -grav, 0.0)
        var next_vel = state.linear_vel + accel_w * h
        if isfinite(next_vel.x) and isfinite(next_vel.y) and isfinite(next_vel.z):
            state.linear_vel = next_vel
        var next_pos = state.pos + state.linear_vel * h
        if isfinite(next_pos.x) and isfinite(next_pos.y) and isfinite(next_pos.z):
            state.pos = next_pos

        # 8. Implicit Newton-Raphson gyroscopic + external moment integration & detmath SO(3) attitude update
        var wb = state.rot.conjugate().rotate(state.angular_vel)
        var wb_next = _solve_implicit_euler_rotation(wb, state.current_inertia, tau_total_b, h)
        var next_ang = state.rot.rotate(wb_next)
        if isfinite(next_ang.x) and isfinite(next_ang.y) and isfinite(next_ang.z):
            state.angular_vel = next_ang
        var next_rot = (Quat.from_scaled_angle(state.angular_vel * h) * state.rot).normalized()
        state.rot = _aero_quat_safe(next_rot)


# =========================================================================
# 8-Wide SIMD ZEM-APN + Decaying Helical Itano Circus Missile Swarm Kernel
# =========================================================================


@fieldwise_init
struct MissileBatch8(Copyable, ImplicitlyCopyable, Movable):
    """8-wide SIMD Structure-of-Arrays packet of guided micro-missiles."""
    var pos: V8
    var vel: V8
    var swirl_phase: F8
    var active: F8
    var min_range: F8

    def __init__(out self):
        self.pos = V8.zero()
        self.vel = V8(F8(0.0), F8(0.0), F8(100.0))
        self.swirl_phase = F8(0.0, 0.785398, 1.570796, 2.356194, 3.141593, 3.926991, 4.712389, 5.497787)
        self.active = F8(1.0)
        self.min_range = F8(1.0e30)


def step_zem_apn_simd8(
    mut pos: V8,
    mut vel: V8,
    mut swirl_phase: F8,
    target_pos: V8,
    target_vel: V8,
    target_accel: V8,
    dt: Float32,
    active_mask: F8 = F8(1.0),
    nav_gain: Float32 = 4.0,
    max_accel: Float32 = 420.0,
    cruise_speed: Float32 = 420.0,
    swirl_amp: Float32 = 95.0,
    swirl_freq: Float32 = 11.0,
    terminal_homing_range: Float32 = 110.0,
    swirl_fade_span: Float32 = 340.0,
) -> F8:
    """8-wide `SIMD[DType.float32, 8]` Zero-Effort-Miss Augmented Proportional Navigation (ZEM-APN)
    + decaying helical *Itano Circus* swirl guidance step.
    Returns the updated 8-lane slant range `R` to target."""
    pos = _aero_v8_safe(pos)
    vel = _aero_v8_safe(vel)
    swirl_phase = _aero_f8_safe(swirl_phase)
    var t_pos = _aero_v8_safe(target_pos)
    var t_vel = _aero_v8_safe(target_vel)
    var t_acc = _aero_v8_safe(target_accel)

    if not isfinite(dt) or dt <= Float32(0.0):
        var dx0 = t_pos.x.cast[DType.float64]() - pos.x.cast[DType.float64]()
        var dy0 = t_pos.y.cast[DType.float64]() - pos.y.cast[DType.float64]()
        var dz0 = t_pos.z.cast[DType.float64]() - pos.z.cast[DType.float64]()
        return _f64_to_f8_finite(sqrt(max(F64_8(0.0), dx0 * dx0 + dy0 * dy0 + dz0 * dz0)))

    var safe_dt = min(dt, Float32(1.0))
    var h = F8(safe_dt)
    var h64 = F64_8(Float64(safe_dt))
    var mask = _aero_f8_safe(active_mask, F8(0.0))
    var safe_nav = max(Float32(0.0), _aero_f32_safe(nav_gain, Float32(4.0)))
    var safe_max_accel = max(Float32(0.0), _aero_f32_safe(max_accel, Float32(420.0)))
    var safe_cruise = max(Float32(0.0), _aero_f32_safe(cruise_speed, Float32(420.0)))
    var safe_swirl_amp = _aero_f32_safe(swirl_amp, Float32(95.0))
    var safe_swirl_freq = _aero_f32_safe(swirl_freq, Float32(11.0))
    var safe_term_range = max(Float32(0.0), _aero_f32_safe(terminal_homing_range, Float32(110.0)))
    var safe_fade_span = max(Float32(1.0), _aero_f32_safe(swirl_fade_span, Float32(340.0)))

    # Evaluate LOS range and unit vector in Float64 so ±1e30+ target separation never overflows r2
    var rx64 = t_pos.x.cast[DType.float64]() - pos.x.cast[DType.float64]()
    var ry64 = t_pos.y.cast[DType.float64]() - pos.y.cast[DType.float64]()
    var rz64 = t_pos.z.cast[DType.float64]() - pos.z.cast[DType.float64]()
    var r2_64 = rx64 * rx64 + ry64 * ry64 + rz64 * rz64
    var dist64 = sqrt(max(F64_8(1.0e-6), r2_64))
    var inv_dist64 = F64_8(1.0) / dist64
    var ux64 = rx64 * inv_dist64
    var uy64 = ry64 * inv_dist64
    var uz64 = rz64 * inv_dist64
    var u_los = V8(
        ux64.cast[DType.float32](),
        uy64.cast[DType.float32](),
        uz64.cast[DType.float32](),
    )

    # Closing velocity & time-to-go t_go in Float64
    var vrx64 = t_vel.x.cast[DType.float64]() - vel.x.cast[DType.float64]()
    var vry64 = t_vel.y.cast[DType.float64]() - vel.y.cast[DType.float64]()
    var vrz64 = t_vel.z.cast[DType.float64]() - vel.z.cast[DType.float64]()
    var vr_dot_u64 = vrx64 * ux64 + vry64 * uy64 + vrz64 * uz64
    var v_close64 = max(F64_8(40.0), F64_8(0.0) - vr_dot_u64)
    var t_go64 = max(F64_8(0.04), dist64 / v_close64)

    # Zero-Effort-Miss with target acceleration feed-forward: ZEM = r + v_rel * t_go + 0.5 * a_T * t_go^2
    var tgo2_64 = t_go64 * t_go64
    var half_tgo2_64 = F64_8(0.5) * tgo2_64
    var zx64 = rx64 + vrx64 * t_go64 + t_acc.x.cast[DType.float64]() * half_tgo2_64
    var zy64 = ry64 + vry64 * t_go64 + t_acc.y.cast[DType.float64]() * half_tgo2_64
    var zz64 = rz64 + vrz64 * t_go64 + t_acc.z.cast[DType.float64]() * half_tgo2_64
    var z_dot_u64 = zx64 * ux64 + zy64 * uy64 + zz64 * uz64
    var zpx64 = zx64 - ux64 * z_dot_u64
    var zpy64 = zy64 - uy64 * z_dot_u64
    var zpz64 = zz64 - uz64 * z_dot_u64
    var gain_over_tgo2 = F64_8(Float64(safe_nav)) / tgo2_64
    var aapnx64 = zpx64 * gain_over_tgo2
    var aapny64 = zpy64 * gain_over_tgo2
    var aapnz64 = zpz64 * gain_over_tgo2

    # Orthonormal basis (e1, e2) perpendicular to u_los for decaying helical Itano swirl
    var x_steep = abs(u_los.x).gt(F8(0.57735))
    var e1_raw = V8(
        x_steep.select(u_los.y, F8(0.0)),
        x_steep.select(F8(0.0) - u_los.x, u_los.z),
        x_steep.select(F8(0.0), F8(0.0) - u_los.y),
    )
    var e1 = e1_raw.scale(F8(1.0) / sqrt(max(F8(1.0e-12), e1_raw.dot(e1_raw))))
    var e2 = u_los.cross(e1)

    swirl_phase = _aero_f8_safe(swirl_phase + F8(safe_swirl_freq) * h)
    var s_swirl = det_sin(swirl_phase)
    var c_swirl = det_cos(swirl_phase)
    var fade_t64 = max(
        F64_8(0.0),
        min(F64_8(1.0), (dist64 - F64_8(Float64(safe_term_range))) / F64_8(Float64(safe_fade_span))),
    )
    var swirl_mag64 = F64_8(Float64(safe_swirl_amp)) * fade_t64 * fade_t64
    var c_sw_mag64 = c_swirl.cast[DType.float64]() * swirl_mag64
    var s_sw_mag64 = s_swirl.cast[DType.float64]() * swirl_mag64
    var aswirlx64 = e1.x.cast[DType.float64]() * c_sw_mag64 + e2.x.cast[DType.float64]() * s_sw_mag64
    var aswirly64 = e1.y.cast[DType.float64]() * c_sw_mag64 + e2.y.cast[DType.float64]() * s_sw_mag64
    var aswirlz64 = e1.z.cast[DType.float64]() * c_sw_mag64 + e2.z.cast[DType.float64]() * s_sw_mag64

    # Structural G-limit clamp on lateral guidance + swirl acceleration (in Float64 to avoid dot overflow)
    var alatx64 = aapnx64 + aswirlx64
    var alaty64 = aapny64 + aswirly64
    var alatz64 = aapnz64 + aswirlz64
    var alat2_64 = alatx64 * alatx64 + alaty64 * alaty64 + alatz64 * alatz64
    var alim64 = F64_8(Float64(safe_max_accel))
    var over = alat2_64.gt(alim64 * alim64)
    var scale_lat = over.select(alim64 / sqrt(max(alat2_64, F64_8(1.0e-12))), F64_8(1.0))
    alatx64 = alatx64 * scale_lat
    alaty64 = alaty64 * scale_lat
    alatz64 = alatz64 * scale_lat

    # Axial sustainer motor thrust toward cruise_speed along current heading (or LOS at launch)
    var vx64 = vel.x.cast[DType.float64]()
    var vy64 = vel.y.cast[DType.float64]()
    var vz64 = vel.z.cast[DType.float64]()
    var spd64 = sqrt(max(F64_8(1.0e-6), vx64 * vx64 + vy64 * vy64 + vz64 * vz64))
    var spd_gt1 = spd64.gt(F64_8(1.0))
    var fwdx64 = spd_gt1.select(vx64 / spd64, ux64)
    var fwdy64 = spd_gt1.select(vy64 / spd64, uy64)
    var fwdz64 = spd_gt1.select(vz64 / spd64, uz64)
    var a_ax_mag64 = max(F64_8(-80.0), min(F64_8(220.0), (F64_8(Float64(safe_cruise)) - spd64) * F64_8(4.5)))

    var atotx64 = alatx64 + fwdx64 * a_ax_mag64
    var atoty64 = alaty64 + fwdy64 * a_ax_mag64
    var atotz64 = alatz64 + fwdz64 * a_ax_mag64

    var v_next = V8(
        _f64_to_f8_finite(vx64 + atotx64 * h64),
        _f64_to_f8_finite(vy64 + atoty64 * h64),
        _f64_to_f8_finite(vz64 + atotz64 * h64),
    )
    var p_next = V8(
        _f64_to_f8_finite(pos.x.cast[DType.float64]() + v_next.x.cast[DType.float64]() * h64),
        _f64_to_f8_finite(pos.y.cast[DType.float64]() + v_next.y.cast[DType.float64]() * h64),
        _f64_to_f8_finite(pos.z.cast[DType.float64]() + v_next.z.cast[DType.float64]() * h64),
    )

    var live = mask.gt(F8(0.5))
    vel = V8(live.select(v_next.x, vel.x), live.select(v_next.y, vel.y), live.select(v_next.z, vel.z))
    pos = V8(live.select(p_next.x, pos.x), live.select(p_next.y, pos.y), live.select(p_next.z, pos.z))

    var rx_after64 = t_pos.x.cast[DType.float64]() - pos.x.cast[DType.float64]()
    var ry_after64 = t_pos.y.cast[DType.float64]() - pos.y.cast[DType.float64]()
    var rz_after64 = t_pos.z.cast[DType.float64]() - pos.z.cast[DType.float64]()
    return _f64_to_f8_finite(sqrt(max(F64_8(0.0), rx_after64 * rx_after64 + ry_after64 * ry_after64 + rz_after64 * rz_after64)))


def step_missile_batch8(
    mut batch: MissileBatch8,
    target_pos: V8,
    target_vel: V8,
    target_accel: V8,
    dt: Float32,
    nav_gain: Float32 = 4.0,
    max_accel: Float32 = 420.0,
    cruise_speed: Float32 = 420.0,
    swirl_amp: Float32 = 95.0,
    swirl_freq: Float32 = 11.0,
) -> F8:
    """Convenience wrapper updating a `MissileBatch8` packet and tracking minimum intercept range."""
    var d = step_zem_apn_simd8(
        batch.pos,
        batch.vel,
        batch.swirl_phase,
        target_pos,
        target_vel,
        target_accel,
        dt,
        batch.active,
        nav_gain,
        max_accel,
        cruise_speed,
        swirl_amp,
        swirl_freq,
    )
    batch.min_range = min(_aero_f8_safe(batch.min_range, F8(1.0e30)), d)
    return d
