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
from std.math import sqrt, floor, abs, min, max
from flint.core.math import Vec3, Quat
from flint.core.detmath import (
    det_sin,
    det_cos,
    det_atan2,
    det_exp,
    set_sim_float_mode,
    PI,
    TWO_OVER_PI,
    PIO2_1,
    PIO2_2,
    PIO2_3,
)
from flint.physics.solver_bodies import LANES, F8, I8, V8


comptime ISA_RHO0: Float32 = 1.225
comptime ISA_SCALE_HEIGHT_M: Float32 = 8500.0
comptime ISA_T0_K: Float32 = 288.15
comptime ISA_LAPSE_K_PER_M: Float32 = 0.0065
comptime ISA_T_STRATO_K: Float32 = 216.65


@always_inline
def eval_isa_atmosphere(altitude_m: Float32) -> Tuple[Float32, Float32]:
    """Returns `(air_density_kg_m3, speed_of_sound_m_s)` at geometric altitude `altitude_m`."""
    var h = max(Float32(0.0), altitude_m)
    var rho = ISA_RHO0 * det_exp(-h / ISA_SCALE_HEIGHT_M)
    var temp_k = max(ISA_T_STRATO_K, ISA_T0_K - ISA_LAPSE_K_PER_M * h)
    var a_sound = sqrt(Float32(1.4 * 287.058) * temp_k)
    return (rho, a_sound)


@always_inline
def eval_ground_effect(h_agl: Float32, wingspan: Float32) -> Tuple[Float32, Float32]:
    """Chebyshev / McCormick ground-effect model.
    Returns `(ge_lift_factor, ge_induced_drag_factor)` for height-above-ground `h_agl` and span `wingspan`."""
    var b = max(Float32(0.5), wingspan)
    var hb = max(Float32(0.02), h_agl) / b
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
    var m = max(Float32(0.0), mach)

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
        cd_wave = cd_wave_peak * s
    elif m > Float32(1.08) and m <= Float32(1.25):
        var t = (m - Float32(1.08)) / Float32(0.17)
        var s = t * t * (Float32(3.0) - Float32(2.0) * t)
        cd_wave = cd_wave_peak * (Float32(1.0) - Float32(0.22) * s)
    elif m > Float32(1.25):
        var sup_decay = Float32(0.75) / sqrt(max(Float32(0.5625), m * m - Float32(1.0)))
        cd_wave = cd_wave_peak * Float32(0.78) * min(Float32(1.0), sup_decay)

    return (pg_lift, cd_wave)


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
    var abs_a = abs(alpha_rad)
    var s_a = det_sin(alpha_rad)
    var c_a = det_cos(alpha_rad)
    var sin_2a = Float32(2.0) * s_a * c_a

    # Attached regime (clamped linear input so pre-blend arithmetic never overflows at large alpha)
    var a_lin = max(-stall_alpha * Float32(1.6), min(stall_alpha * Float32(1.6), alpha_rad))
    var cl_att = (cl0 + cl_alpha * pg_mult * a_lin) * ge_lift
    var k_ind = Float32(1.0) / max(Float32(1.5), PI * Float32(0.82) * aspect_ratio)
    var cd_att = cd0 + cd_wave + k_ind * ge_ind_drag * cl_att * cl_att

    # Post-stall double-angle vortex/flat-plate regime
    var cl_post = Float32(1.18) * sin_2a * ge_lift
    var cd_post = cd0 + cd_wave + Float32(1.85) * s_a * s_a

    # Smooth C1 stall transition over [stall_alpha, stall_alpha + 0.22]
    var t_stall = max(Float32(0.0), min(Float32(1.0), (abs_a - stall_alpha) / Float32(0.22)))
    var w_stall = t_stall * t_stall * (Float32(3.0) - Float32(2.0) * t_stall)

    var cl = (Float32(1.0) - w_stall) * cl_att + w_stall * cl_post
    var cd = (Float32(1.0) - w_stall) * cd_att + w_stall * cd_post
    # In right-handed (+X right, +Y up, +Z nose-forward), +tau_x pitches nose DOWN toward +Z,
    # restoring positive alpha > 0 back toward 0 (weathervane static longitudinal stability).
    var cm = Float32(0.65) * s_a
    return (cl, cd, cm)


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


@always_inline
def _solve_implicit_euler_rotation(wb0: Vec3, inertia: Vec3, tau_b: Vec3, h: Float32) -> Vec3:
    """2-iteration Implicit Newton-Raphson solver for coupled Euler rotational dynamics:
    `R(w^+) = I * (w^+ - w0) + h * (w^+ x (I * w^+) - tau_b) = 0`."""
    var ix = max(Float32(1.0e-3), inertia.x)
    var iy = max(Float32(1.0e-3), inertia.y)
    var iz = max(Float32(1.0e-3), inertia.z)

    var w = Vec3(
        wb0.x + (tau_b.x / ix) * h,
        wb0.y + (tau_b.y / iy) * h,
        wb0.z + (tau_b.z / iz) * h,
    )

    comptime for _ in range(2):
        var rx = ix * (w.x - wb0.x) + h * ((iz - iy) * w.y * w.z - tau_b.x)
        var ry = iy * (w.y - wb0.y) + h * ((ix - iz) * w.z * w.x - tau_b.y)
        var rz = iz * (w.z - wb0.z) + h * ((iy - ix) * w.x * w.y - tau_b.z)

        var j00 = ix
        var j01 = h * (iz - iy) * w.z
        var j02 = h * (iz - iy) * w.y
        var j10 = h * (ix - iz) * w.z
        var j11 = iy
        var j12 = h * (ix - iz) * w.x
        var j20 = h * (iy - ix) * w.y
        var j21 = h * (iy - ix) * w.x
        var j22 = iz

        var c00 = j11 * j22 - j12 * j21
        var c01 = j12 * j20 - j10 * j22
        var c02 = j10 * j21 - j11 * j20
        var det = j00 * c00 + j01 * c01 + j02 * c02
        if abs(det) <= Float32(1.0e-18):
            return w
        var inv_det = Float32(1.0) / det
        var c10 = j02 * j21 - j01 * j22
        var c11 = j00 * j22 - j02 * j20
        var c12 = j01 * j20 - j00 * j21
        var c20 = j01 * j12 - j02 * j11
        var c21 = j02 * j10 - j00 * j12
        var c22 = j00 * j11 - j01 * j10

        w = Vec3(
            w.x - (c00 * rx + c10 * ry + c20 * rz) * inv_det,
            w.y - (c01 * rx + c11 * ry + c21 * rz) * inv_det,
            w.z - (c02 * rx + c12 * ry + c22 * rz) * inv_det,
        )
    return w


def step_aero6dof(
    mut state: Aero6DofState,
    cfg: Aero6DofConfig,
    ctrl: Aero6DofControls,
    dt: Float32,
    substeps: Int = 4,
):
    """Advances a 6-DOF variable-geometry aerospace/mecha craft by `dt`."""
    set_sim_float_mode()
    if dt <= Float32(1.0e-7):
        return
    var steps = max(1, substeps)
    var h = dt / Float32(steps)

    for _ in range(steps):
        # 1. Morph blend u(t) and instantaneous mass & principal inertia I(u) = lerp(I_0, I_1, u)
        var u_tgt = max(Float32(0.0), min(Float32(1.0), ctrl.morph_u_target))
        var max_du = max(Float32(0.0), ctrl.morph_rate) * h
        var du = max(-max_du, min(max_du, u_tgt - state.morph_u))
        state.morph_u = max(Float32(0.0), min(Float32(1.0), state.morph_u + du))
        var u = state.morph_u
        var om_u = Float32(1.0) - u

        var dry_mass = om_u * cfg.mass_0 + u * cfg.mass_1
        var cur_mass = max(Float32(10.0), dry_mass + max(Float32(0.0), state.fuel_mass))
        var cur_i = Vec3(
            om_u * cfg.inertia_0.x + u * cfg.inertia_1.x,
            om_u * cfg.inertia_0.y + u * cfg.inertia_1.y,
            om_u * cfg.inertia_0.z + u * cfg.inertia_1.z,
        )
        var span = om_u * cfg.wingspan_0 + u * cfg.wingspan_1
        var area = max(Float32(0.5), om_u * cfg.wing_area_0 + u * cfg.wing_area_1)
        var cd0 = om_u * cfg.cd0_0 + u * cfg.cd0_1
        var aspect_ratio = (span * span) / area

        state.current_mass = cur_mass
        state.current_inertia = cur_i

        # 2. ISA Atmosphere & Ground Effect
        var isa = eval_isa_atmosphere(state.pos.y)
        var rho = isa[0]
        var a_sound = isa[1]
        var h_agl = state.pos.y - ctrl.ground_y
        var ge = eval_ground_effect(h_agl, span)
        var ge_lift = ge[0]
        var ge_ind_drag = ge[1]
        state.ge_lift_factor = ge_lift

        # 3. Body-frame airspeed, AoA alpha, sideslip beta, Mach, and dynamic pressure
        # Body frame convention: +X starboard-right, +Y dorsal-up, +Z nose-forward
        var inv_q = state.rot.conjugate()
        var v_air_w = state.linear_vel - ctrl.wind_vel
        var vb = inv_q.rotate(v_air_w)
        var v_mag = vb.length()
        var mach = v_mag / max(Float32(1.0), a_sound)
        var q_dyn = Float32(0.5) * rho * v_mag * v_mag
        state.mach = mach
        state.dynamic_pressure = q_dyn

        var f_aero_b = Vec3()
        var tau_aero_b = Vec3()
        var wb = inv_q.rotate(state.angular_vel)

        if v_mag > Float32(0.05) and q_dyn > Float32(1.0e-5):
            var alpha = det_atan2(-vb.y, vb.z)
            var v_yz = sqrt(vb.y * vb.y + vb.z * vb.z)
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
            var drag_dir_b = -vb * (Float32(1.0) / v_mag)
            var lift_dir_b = Vec3(0.0, det_cos(alpha), det_sin(alpha))
            var cy_side = -Float32(0.85) * det_sin(beta)

            var q_s = q_dyn * area
            f_aero_b = drag_dir_b * (q_s * cd) + lift_dir_b * (q_s * cl) + Vec3(q_s * cy_side, 0.0, 0.0)

            # Aerodynamic stability, control surfaces, and angular rate damping
            var chord = area / max(Float32(0.5), span)
            var pitch_surf = max(Float32(-1.0), min(Float32(1.0), ctrl.pitch_cmd))
            var yaw_surf = max(Float32(-1.0), min(Float32(1.0), ctrl.yaw_cmd))
            var roll_surf = max(Float32(-1.0), min(Float32(1.0), ctrl.roll_cmd))

            # Control surface authority fades smoothly post-stall while TVC/RCS remain 100% active
            var surf_eff = Float32(1.0) / (Float32(1.0) + Float32(1.5) * det_sin(alpha) * det_sin(alpha))
            var damp_scale = min(Float32(2.5), rho / ISA_RHO0) * max(Float32(0.2), min(Float32(3.0), v_mag / Float32(100.0)))

            tau_aero_b = Vec3(
                q_s * chord * cm_static - q_s * cfg.aero_ctrl_torque_gain.x * surf_eff * pitch_surf - cfg.aero_rate_damping.x * damp_scale * wb.x,
                -q_s * span * Float32(0.18) * det_sin(beta) + q_s * cfg.aero_ctrl_torque_gain.y * surf_eff * yaw_surf - cfg.aero_rate_damping.y * damp_scale * wb.y,
                -q_s * span * Float32(0.06) * det_sin(beta) + q_s * cfg.aero_ctrl_torque_gain.z * surf_eff * roll_surf - cfg.aero_rate_damping.z * damp_scale * wb.z,
            )
        else:
            state.alpha_rad = 0.0
            state.beta_rad = 0.0
            state.cl = 0.0
            state.cd = cd0
            state.cd_wave = 0.0

        # 4. Main Afterburner + Thrust Vectoring Control (TVC)
        var th_main = max(Float32(0.0), min(Float32(1.0), ctrl.throttle))
        if state.fuel_mass < Float32(0.0):
            th_main = Float32(0.0)
        var main_thrust_mag = th_main * cfg.max_afterburner_thrust

        # Deflect aft exhaust nozzle: positive pitch_cmd deflects thrust vector so aft tail pushes -Y, pitching nose +X up!
        var tvc_p = ctrl.tvc_pitch if ctrl.tvc_pitch != Float32(0.0) else ctrl.pitch_cmd * cfg.max_tvc_angle_rad
        var tvc_y = ctrl.tvc_yaw if ctrl.tvc_yaw != Float32(0.0) else ctrl.yaw_cmd * cfg.max_tvc_angle_rad
        tvc_p = max(-cfg.max_tvc_angle_rad, min(cfg.max_tvc_angle_rad, tvc_p))
        tvc_y = max(-cfg.max_tvc_angle_rad, min(cfg.max_tvc_angle_rad, tvc_y))

        var cp_t = det_cos(tvc_p)
        var sp_t = det_sin(tvc_p)
        var cy_t = det_cos(tvc_y)
        var sy_t = det_sin(tvc_y)
        # With r_tvc = (0, 0, tvc_arm_z < 0):
        # r_tvc x F_tvc = (0, 0, -|z|) x (Fx, Fy, Fz) = (+|z|*Fy, -|z|*Fx, 0).
        # Nose-up rotation (+Z -> +Y) is -X, so tvc_p > 0 pushes tail -Y (Fy < 0 => tau_x < 0, nose UP!).
        var f_tvc_b = Vec3(
            -sy_t * cp_t * main_thrust_mag,
            -sp_t * main_thrust_mag,
            cy_t * cp_t * main_thrust_mag,
        )
        var r_tvc = Vec3(0.0, 0.0, cfg.tvc_arm_z)
        var tau_tvc_b = r_tvc.cross(f_tvc_b)

        # 5. Off-CoM GERWALK Twin Vectoring Hover Nozzles
        var th_hl = max(Float32(0.0), min(Float32(1.0), ctrl.hover_left_throttle))
        var th_hr = max(Float32(0.0), min(Float32(1.0), ctrl.hover_right_throttle))
        var noz_dir_b = Vec3(0.0, det_sin(ctrl.hover_tilt_rad), det_cos(ctrl.hover_tilt_rad))
        var f_noz_l = noz_dir_b * (th_hl * cfg.max_hover_nozzle_thrust)
        var f_noz_r = noz_dir_b * (th_hr * cfg.max_hover_nozzle_thrust)
        var tau_noz_b = cfg.gerwalk_nozzle_left.cross(f_noz_l) + cfg.gerwalk_nozzle_right.cross(f_noz_r)

        # Fuel burn update (if configured)
        if cfg.sfc_kg_per_n_s > Float32(0.0) and state.fuel_mass > Float32(0.0):
            var total_thrust_n = main_thrust_mag + (th_hl + th_hr) * cfg.max_hover_nozzle_thrust
            state.fuel_mass = max(Float32(0.0), state.fuel_mass - cfg.sfc_kg_per_n_s * total_thrust_n * h)

        # 6. Reaction Control System (RCS) verniers (100% effective at 0 airspeed and in orbital vacuum)
        var rcs_in = ctrl.rcs_cmd
        if rcs_in.length_squared() == Float32(0.0) and q_dyn < Float32(250.0):
            rcs_in = Vec3(-ctrl.pitch_cmd, ctrl.yaw_cmd, ctrl.roll_cmd)
        var tau_rcs_b = Vec3(
            max(Float32(-1.0), min(Float32(1.0), rcs_in.x)) * cfg.rcs_max_torque.x,
            max(Float32(-1.0), min(Float32(1.0), rcs_in.y)) * cfg.rcs_max_torque.y,
            max(Float32(-1.0), min(Float32(1.0), rcs_in.z)) * cfg.rcs_max_torque.z,
        )

        # 7. True Newtonian translational integration: a = F / m(t)
        var f_body_total = f_aero_b + f_tvc_b + f_noz_l + f_noz_r
        var f_non_grav_w = state.rot.rotate(f_body_total)
        var accel_w = f_non_grav_w * (Float32(1.0) / cur_mass) + Vec3(0.0, -ctrl.gravity, 0.0)
        state.g_load = (f_non_grav_w.length() / cur_mass) / Float32(9.81)

        state.linear_vel = state.linear_vel + accel_w * h
        state.pos = state.pos + state.linear_vel * h

        # 8. Implicit Newton-Raphson gyroscopic + external moment integration & detmath SO(3) attitude update
        var tau_total_b = tau_aero_b + tau_tvc_b + tau_noz_b + tau_rcs_b
        var wb_next = _solve_implicit_euler_rotation(wb, cur_i, tau_total_b, h)
        state.angular_vel = state.rot.rotate(wb_next)
        state.rot = (Quat.from_scaled_angle(state.angular_vel * h) * state.rot).normalized()


# =========================================================================
# 8-Wide SIMD ZEM-APN + Decaying Helical Itano Circus Missile Swarm Kernel
# =========================================================================

@always_inline
def _det_sincos_f8(x: F8) -> Tuple[F8, F8]:
    """8-wide SIMD deterministic Cephes minimax `(sin, cos)` with Cody-Waite reduction (zero std.math.sin/cos)."""
    var j = floor(x * F8(TWO_OVER_PI) + F8(0.5))
    var r = ((x - j * F8(PIO2_1)) - j * F8(PIO2_2)) - j * F8(PIO2_3)
    var z = r * r
    var s_poly = ((F8(-1.9515295891e-4) * z + F8(8.3321608736e-3)) * z + F8(-1.6666654611e-1)) * z * r + r
    var c_poly = ((F8(2.443315711809948e-5) * z + F8(-1.388731625493765e-3)) * z + F8(4.166664568298827e-2)) * z * z - F8(0.5) * z + F8(1.0)

    var q = I8(j) & I8(3)
    var q0 = q.eq(I8(0))
    var q1 = q.eq(I8(1))
    var q2 = q.eq(I8(2))

    var sin_out = q0.select(s_poly, q1.select(c_poly, q2.select(F8(0.0) - s_poly, F8(0.0) - c_poly)))
    var cos_out = q0.select(c_poly, q1.select(F8(0.0) - s_poly, q2.select(F8(0.0) - c_poly, s_poly)))
    return (sin_out, cos_out)


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
    var h = F8(dt)
    var r_vec = target_pos - pos
    var r2 = r_vec.dot(r_vec)
    var dist = sqrt(max(F8(1.0e-6), r2))
    var u_los = r_vec.scale(F8(1.0) / dist)

    # Closing velocity & time-to-go t_go
    var v_rel = target_vel - vel
    var v_close = max(F8(40.0), F8(0.0) - v_rel.dot(u_los))
    var t_go = max(F8(0.04), dist / v_close)

    # Zero-Effort-Miss with target acceleration feed-forward: ZEM = r + v_rel * t_go + 0.5 * a_T * t_go^2
    var zem = r_vec + v_rel.scale(t_go) + target_accel.scale(F8(0.5) * t_go * t_go)
    var zem_perp = zem - u_los.scale(zem.dot(u_los))
    var a_apn = zem_perp.scale(F8(nav_gain) / (t_go * t_go))

    # Orthonormal basis (e1, e2) perpendicular to u_los for decaying helical Itano swirl
    var x_steep = abs(u_los.x).gt(F8(0.57735))
    var e1_raw = V8(
        x_steep.select(u_los.y, F8(0.0)),
        x_steep.select(F8(0.0) - u_los.x, u_los.z),
        x_steep.select(F8(0.0), F8(0.0) - u_los.y),
    )
    var e1 = e1_raw.scale(F8(1.0) / sqrt(max(F8(1.0e-12), e1_raw.dot(e1_raw))))
    var e2 = u_los.cross(e1)

    swirl_phase = swirl_phase + F8(swirl_freq) * h
    var sc = _det_sincos_f8(swirl_phase)
    var fade_t = max(
        F8(0.0),
        min(F8(1.0), (dist - F8(terminal_homing_range)) / F8(max(Float32(1.0), swirl_fade_span))),
    )
    var swirl_mag = F8(swirl_amp) * fade_t * fade_t
    var a_swirl = e1.scale(sc[1] * swirl_mag) + e2.scale(sc[0] * swirl_mag)

    # Structural G-limit clamp on lateral guidance + swirl acceleration
    var a_lat = a_apn + a_swirl
    var a_lat2 = a_lat.dot(a_lat)
    var a_lim = F8(max_accel)
    var over = a_lat2.gt(a_lim * a_lim)
    a_lat = a_lat.scale(over.select(a_lim / sqrt(max(a_lat2, F8(1.0e-12))), F8(1.0)))

    # Axial sustainer motor thrust toward cruise_speed along current heading (or LOS at launch)
    var spd = sqrt(max(F8(1.0e-6), vel.dot(vel)))
    var fwd = V8(
        spd.gt(F8(1.0)).select(vel.x / spd, u_los.x),
        spd.gt(F8(1.0)).select(vel.y / spd, u_los.y),
        spd.gt(F8(1.0)).select(vel.z / spd, u_los.z),
    )
    var a_axial = fwd.scale(max(F8(-80.0), min(F8(220.0), (F8(cruise_speed) - spd) * F8(4.5))))

    var live = active_mask.gt(F8(0.5))
    var a_total = a_lat + a_axial
    var v_next = vel + a_total.scale(h)
    var p_next = pos + v_next.scale(h)

    vel = V8(live.select(v_next.x, vel.x), live.select(v_next.y, vel.y), live.select(v_next.z, vel.z))
    pos = V8(live.select(p_next.x, pos.x), live.select(p_next.y, pos.y), live.select(p_next.z, pos.z))

    var r_after = target_pos - pos
    return sqrt(max(F8(0.0), r_after.dot(r_after)))


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
    batch.min_range = min(batch.min_range, d)
    return d
