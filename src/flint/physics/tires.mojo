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
from std.math import sqrt, max, min, abs, isfinite
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
        if not isfinite(normal_load) or normal_load <= Float32(1.0):
            return (Float32(0.0), Float32(0.0), Float32(0.0))
        var fz = min(Float32(1.0e12), normal_load)
        var safe_sr = max(Float32(-1.0e4), min(Float32(1.0e4), slip_ratio)) if isfinite(slip_ratio) else Float32(0.0)
        var safe_sa = slip_angle if isfinite(slip_angle) else Float32(0.0)

        var peak_force = self._peak_force(fz)
        var clamped_alpha = max(Float32(-1.53), min(Float32(1.53), safe_sa))
        var tan_alpha = det_sin(clamped_alpha) / max(Float32(1.0e-5), det_cos(clamped_alpha))

        var kp = max(Float32(1.0e-4), self.params.kappa_peak)
        var ap = max(Float32(1.0e-4), self.params.alpha_peak)
        var sx64 = Float64(safe_sr) / Float64(kp)
        var sy64 = Float64(tan_alpha) / Float64(ap)
        var sigma64 = sqrt(sx64 * sx64 + sy64 * sy64)

        var fx: Float32
        var fy: Float32
        if sigma64 > 1.0e-6 and isfinite(sigma64):
            var inv_sigma = Float32(1.0 / sigma64)
            var sigma_clamped = Float32(min(1.0e6, sigma64))
            var fx0 = self._pure_magic(peak_force, sigma_clamped * kp)
            var fy0 = self._pure_magic(peak_force, sigma_clamped * ap)
            fx = Float32(sx64 / sigma64) * fx0
            fy = Float32(sy64 / sigma64) * fy0
            _ = inv_sigma
        else:
            var lin_k = peak_force * self.params.B * self.params.C
            fx = lin_k * safe_sr
            fy = lin_k * tan_alpha

        var cos_trail = det_cos(Float32(1.4) * det_atan(Float32(7.5) * clamped_alpha))
        var trail = self.params.pneumatic_trail_0 * max(Float32(0.0), cos_trail)
        var mz = -trail * fy
        if not (isfinite(fx) and isfinite(fy) and isfinite(mz)):
            return (Float32(0.0), Float32(0.0), Float32(0.0))
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
        if not isfinite(bristle_x):
            bristle_x = Float32(0.0)
        if not isfinite(bristle_y):
            bristle_y = Float32(0.0)
        if not (
            isfinite(normal_load) and isfinite(dt) and dt > Float32(0.0)
            and isfinite(v_long) and isfinite(v_lat) and isfinite(tire_v)
        ):
            return (Float32(0.0), Float32(0.0), Float32(0.0))
        var fz = max(Float32(0.0), min(Float32(1.0e12), normal_load))
        if fz <= Float32(1.0):
            bristle_x = Float32(0.0)
            bristle_y = Float32(0.0)
            return (Float32(0.0), Float32(0.0), Float32(0.0))

        var safe_dt64 = min(0.5, Float64(dt))
        var lx = max(Float32(0.02), self.params.relaxation_len_x)
        var ly = max(Float32(0.02), self.params.relaxation_len_y)
        var kp = max(Float32(1.0e-4), self.params.kappa_peak)
        var ap = max(Float32(1.0e-4), self.params.alpha_peak)

        var v_sx64 = Float64(tire_v) - Float64(v_long)
        var v_sy64 = -Float64(v_lat)
        var roll_spd64 = abs(Float64(v_long))

        # Semi-implicit bristle advection/relaxation update in Float64 (overflow-safe)
        var bx64 = (Float64(bristle_x) + safe_dt64 * v_sx64) / (1.0 + safe_dt64 * (roll_spd64 / Float64(lx)))
        var by64 = (Float64(bristle_y) + safe_dt64 * v_sy64) / (1.0 + safe_dt64 * (roll_spd64 / Float64(ly)))

        # Elastoplastic saturation cap
        var low_blend = min(1.0, roll_spd64 / 1.5)
        var max_sigma = 1.0 + low_blend * 24.0
        var sx_b = bx64 / (Float64(lx) * Float64(kp))
        var sy_b = by64 / (Float64(ly) * Float64(ap))
        var sig_b = sqrt(sx_b * sx_b + sy_b * sy_b)
        if sig_b > max_sigma and sig_b > 0.0:
            var scale = max_sigma / sig_b
            bx64 *= scale
            by64 *= scale

        var bx = Float32(bx64) if isfinite(bx64) else Float32(0.0)
        var by = Float32(by64) if isfinite(by64) else Float32(0.0)
        bristle_x = bx
        bristle_y = by

        var kappa_eff = bx / lx
        var alpha_eff = det_atan(by / ly)
        var forces = self.calc_forces(kappa_eff, alpha_eff, fz, v_long)
        var fx = forces[0]
        var fy = forces[1]
        var mz = forces[2]

        # Critical bristle structural damping near standstill for dead-zero hill-hold
        var spd2d64 = sqrt(Float64(v_long) * Float64(v_long) + Float64(v_lat) * Float64(v_lat))
        if spd2d64 < 1.5:
            var damp_w = 1.0 - spd2d64 / 1.5
            var load_scale = Float64(fz) / Float64(max(Float32(500.0), self.params.nominal_load))
            var c_visc = 3200.0 * load_scale * damp_w
            var fx64 = Float64(fx) + c_visc * v_sx64
            var fy64 = Float64(fy) + c_visc * v_sy64
            var peak = Float64(self._peak_force(fz))
            var mag = sqrt(fx64 * fx64 + fy64 * fy64)
            if mag > peak and mag > 0.0:
                var clamp_s = peak / mag
                fx64 *= clamp_s
                fy64 *= clamp_s
            fx = Float32(fx64) if isfinite(fx64) else Float32(0.0)
            fy = Float32(fy64) if isfinite(fy64) else Float32(0.0)

        return (fx, fy, mz)
