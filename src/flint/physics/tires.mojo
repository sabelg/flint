"""Pacejka Magic Formula, C1-Smooth Combined-Slip Ellipse & Semi-Implicit Bristle Dynamics (v4).

Implements:
1. Deterministic (`flint.core.detmath`) Pacejka Magic Formula 94/02 for pure longitudinal
   force `Fx0(kappa)` and pure lateral force `Fy0(alpha)`.
2. C1-smooth normalized combined-slip friction ellipse:
   `sigma* = (kappa / kappa_p, tan(alpha) / alpha_p)`,
   `Fx = (sigma_x* / sigma*) * Fx0(sigma* * kappa_p)`,
   `Fy = (sigma_y* / sigma*) * Fy0(sigma* * alpha_p)` (zero derivative kink at `sigma* = 1`).
3. Semi-implicit elastoplastic bristle relaxation-length state `(bristle_x, bristle_y)` for
   zero-creep hill-hold on slopes and physically accurate transient turn-in response.
4. Dynamic normal load sensitivity `mu(Fz)` and pneumatic trail self-aligning torque `Mz`.
"""
from std.math import sqrt, max, min, abs
from flint.core.detmath import det_sin, det_cos, det_atan


@fieldwise_init
struct PacejkaParams(Copyable, ImplicitlyCopyable, Movable):
    var B: Float32                # Stiffness factor
    var C: Float32                # Shape factor (~1.3 - 1.65)
    var D: Float32                # Peak friction coefficient (~1.0 - 1.25)
    var E: Float32                # Curvature factor (~ -0.5 - 0.2)
    var load_sensitivity: Float32 # Friction drop under high normal load
    var nominal_load: Float32     # Reference normal load in Newtons (~4000 N)
    var brush_stiffness: Float32  # Low-speed brush stiffness (N/m)
    var pneumatic_trail_0: Float32 # Trail at zero slip angle (~0.032 m)
    var kappa_peak: Float32       # Peak longitudinal slip ratio (~0.14)
    var alpha_peak: Float32       # Peak lateral tan(alpha) (~0.12)
    var relaxation_len_x: Float32 # Longitudinal carcass relaxation length (m)
    var relaxation_len_y: Float32 # Lateral carcass relaxation length (m)

    def __init__(out self):
        self.B = 10.0
        self.C = 1.65
        self.D = 1.15
        self.E = -0.35
        self.load_sensitivity = 0.00004
        self.nominal_load = 4000.0
        self.brush_stiffness = 85000.0
        self.pneumatic_trail_0 = 0.032
        self.kappa_peak = 0.14
        self.alpha_peak = 0.12
        self.relaxation_len_x = 0.12
        self.relaxation_len_y = 0.15

    def __init__(
        out self,
        B: Float32,
        C: Float32,
        D: Float32,
        E: Float32,
        load_sensitivity: Float32,
        nominal_load: Float32,
        brush_stiffness: Float32,
        pneumatic_trail_0: Float32,
    ):
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.load_sensitivity = load_sensitivity
        self.nominal_load = nominal_load
        self.brush_stiffness = brush_stiffness
        self.pneumatic_trail_0 = pneumatic_trail_0
        self.kappa_peak = 0.14
        self.alpha_peak = 0.12
        self.relaxation_len_x = 0.12
        self.relaxation_len_y = 0.15


struct PacejkaTire(Copyable, ImplicitlyCopyable, Movable):
    var params: PacejkaParams

    def __init__(out self, params: PacejkaParams = PacejkaParams()):
        self.params = params

    @always_inline
    def _peak_force(self, fz: Float32) -> Float32:
        var delta_fz = max(Float32(0.0), fz - self.params.nominal_load)
        var mu = max(Float32(0.2), self.params.D * (Float32(1.0) - self.params.load_sensitivity * delta_fz))
        return mu * fz

    @always_inline
    def _pure_magic(self, peak_force: Float32, slip: Float32) -> Float32:
        var bs = self.params.B * slip
        var phi = bs - self.params.E * (bs - det_atan(bs))
        return peak_force * det_sin(self.params.C * det_atan(phi))

    def calc_forces(
        self,
        slip_ratio: Float32,
        slip_angle: Float32,
        normal_load: Float32,
        speed: Float32,
    ) -> Tuple[Float32, Float32, Float32]:
        """Calculates `(Fx, Fy, Mz)` using the C1-smooth normalized combined-slip friction ellipse:
        `sigma* = (kappa / kappa_p, tan(alpha) / alpha_p)`,
        `Fx = (sigma_x* / sigma*) * Fx0(sigma* * kappa_p)`,
        `Fy = (sigma_y* / sigma*) * Fy0(sigma* * alpha_p)`."""
        _ = speed
        var fz = max(Float32(0.0), normal_load)
        if fz <= Float32(1.0):
            return (Float32(0.0), Float32(0.0), Float32(0.0))

        var peak_force = self._peak_force(fz)
        var clamped_alpha = max(Float32(-1.53), min(Float32(1.53), slip_angle))
        var tan_alpha = det_sin(clamped_alpha) / max(Float32(1.0e-5), det_cos(clamped_alpha))

        var kp = max(Float32(1.0e-4), self.params.kappa_peak)
        var ap = max(Float32(1.0e-4), self.params.alpha_peak)
        var sx = slip_ratio / kp
        var sy = tan_alpha / ap
        var sigma = sqrt(sx * sx + sy * sy)

        var fx: Float32
        var fy: Float32
        if sigma > Float32(1.0e-6):
            var inv_sigma = Float32(1.0) / sigma
            var fx0 = self._pure_magic(peak_force, sigma * kp)
            var fy0 = self._pure_magic(peak_force, sigma * ap)
            fx = (sx * inv_sigma) * fx0
            fy = (sy * inv_sigma) * fy0
        else:
            var lin_k = peak_force * self.params.B * self.params.C
            fx = lin_k * slip_ratio
            fy = lin_k * tan_alpha

        var cos_trail = det_cos(Float32(1.4) * det_atan(Float32(7.5) * clamped_alpha))
        var trail = self.params.pneumatic_trail_0 * max(Float32(0.0), cos_trail)
        var mz = -trail * fy
        return (fx, fy, mz)

    def calc_forces_bristle(
        self,
        v_long: Float32,
        v_lat: Float32,
        tire_v: Float32,
        normal_load: Float32,
        dt: Float32,
        mut bristle_x: Float32,
        mut bristle_y: Float32,
    ) -> Tuple[Float32, Float32, Float32]:
        """Updates semi-implicit elastoplastic tire bristle deflection `(bristle_x, bristle_y)`
        and evaluates combined-slip `(Fx, Fy, Mz)` with zero-creep hill-hold stability."""
        var fz = max(Float32(0.0), normal_load)
        if fz <= Float32(1.0):
            bristle_x = Float32(0.0)
            bristle_y = Float32(0.0)
            return (Float32(0.0), Float32(0.0), Float32(0.0))

        var lx = max(Float32(0.02), self.params.relaxation_len_x)
        var ly = max(Float32(0.02), self.params.relaxation_len_y)
        var kp = max(Float32(1.0e-4), self.params.kappa_peak)
        var ap = max(Float32(1.0e-4), self.params.alpha_peak)

        var v_sx = tire_v - v_long
        var v_sy = -v_lat
        var roll_spd = abs(v_long)

        # Semi-implicit bristle advection/relaxation update (unconditionally stable for any dt & speed)
        var bx = (bristle_x + dt * v_sx) / (Float32(1.0) + dt * (roll_spd / lx))
        var by = (bristle_y + dt * v_sy) / (Float32(1.0) + dt * (roll_spd / ly))

        # Elastoplastic saturation cap: prevents unbounded windup at standstill while allowing full slip at speed
        var low_blend = min(Float32(1.0), roll_spd / Float32(1.5))
        var max_sigma = Float32(1.0) + low_blend * Float32(24.0)
        var sx_b = bx / (lx * kp)
        var sy_b = by / (ly * ap)
        var sig_b = sqrt(sx_b * sx_b + sy_b * sy_b)
        if sig_b > max_sigma:
            var scale = max_sigma / sig_b
            bx *= scale
            by *= scale

        bristle_x = bx
        bristle_y = by

        var kappa_eff = bx / lx
        var alpha_eff = det_atan(by / ly)
        var forces = self.calc_forces(kappa_eff, alpha_eff, fz, v_long)
        var fx = forces[0]
        var fy = forces[1]
        var mz = forces[2]

        # Critical bristle structural damping near standstill for dead-zero hill-hold
        var spd2d = sqrt(v_long * v_long + v_lat * v_lat)
        if spd2d < Float32(1.5):
            var damp_w = Float32(1.0) - spd2d / Float32(1.5)
            var load_scale = fz / max(Float32(500.0), self.params.nominal_load)
            var c_visc = Float32(3200.0) * load_scale * damp_w
            fx += c_visc * v_sx
            fy += c_visc * v_sy
            var peak = self._peak_force(fz)
            var mag2 = fx * fx + fy * fy
            if mag2 > peak * peak:
                var clamp_s = peak / sqrt(mag2)
                fx *= clamp_s
                fy *= clamp_s

        return (fx, fy, mz)
