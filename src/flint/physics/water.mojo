"""Water physics (v4): 8-wave Golden-Angle Beaufort spectrum matching `aaa_water.mojo`,
Archimedes submerged sphere/box buoyancy, metacentric righting torque, and quadratic drag.

Preserves flat-water helpers (`NO_WATER`, `WATER_DENSITY`, `WATER_DRAG`, `floats`,
`submerged_fraction`, `buoyant_acceleration`) for legacy voxel/field callers while
providing SI Beaufort wave kinematics and rigid-body hydrodynamics.
"""
from std.math import sqrt, max, min, isfinite
from flint.core.math import Vec3, Quat
from flint.core.detmath import det_sin, det_cos, PI

comptime NO_WATER: Float32 = -1000
comptime WATER_DENSITY: Float32 = 1.6
comptime WATER_DRAG: Float32 = 2.4
comptime RHO_SEAWATER_SI: Float32 = 1025.0
comptime RHO_FRESHWATER_SI: Float32 = 1000.0
# Strict upper bound on sum of the 8 Beaufort wave amplitudes (0.1594 m)
comptime MAX_BEAUFORT_WAVE_AMP: Float32 = 0.165


def floats(material_density: Float32) -> Bool:
    if not isfinite(material_density):
        return False
    return material_density > 0 and material_density < WATER_DENSITY


def submerged_fraction(bottom: Float32, height: Float32, water_level: Float32) -> Float32:
    if not (isfinite(bottom) and isfinite(height) and isfinite(water_level)):
        return Float32(0.0)
    if water_level <= NO_WATER + 1 or height <= 0:
        return Float32(0.0)
    var ratio = (Float64(water_level) - Float64(bottom)) / Float64(height)
    if not isfinite(ratio):
        return Float32(0.0)
    return Float32(max(0.0, min(1.0, ratio)))


def buoyant_acceleration(fraction: Float32, material_density: Float32, gravity: Float32) -> Float32:
    """Upward acceleration from displaced water (Archimedes)."""
    if not (isfinite(fraction) and isfinite(material_density) and isfinite(gravity)):
        return Float32(0.0)
    if fraction <= 0 or material_density <= 0:
        return Float32(0.0)
    var res = gravity * fraction * (WATER_DENSITY / material_density)
    return res if isfinite(res) else Float32(0.0)


def eval_beaufort_wave(
    wx: Float32,
    wz: Float32,
    time_s: Float32,
    water_y: Float32 = 0.0,
    shallow_damp: Float32 = 1.0,
) -> Tuple[Float32, Vec3, Vec3]:
    """Evaluates the 8-wave irrational Golden-Angle Beaufort spectrum matching `aaa_water.mojo`.

    Returns `(eta, normal, orbital_vel)`:
      - `eta`: instantaneous surface elevation (m) around `water_y`
      - `normal`: unit surface normal `normalize(-d_eta/dx, 1, -d_eta/dz)`
      - `orbital_vel`: Airy deep-water wave particle velocity `(vx, vy, vz)` (m/s)
    """
    if not (
        isfinite(wx)
        and isfinite(wz)
        and isfinite(time_s)
        and isfinite(water_y)
        and isfinite(shallow_damp)
    ):
        var safe_y = water_y if isfinite(water_y) else Float32(0.0)
        return (safe_y, Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 0.0))

    var k_arr = [
        Float32(0.42), Float32(0.71), Float32(1.18), Float32(1.95),
        Float32(3.15), Float32(4.85), Float32(6.90), Float32(9.20),
    ]
    var amp_arr = [
        Float32(0.052), Float32(0.038), Float32(0.026), Float32(0.017),
        Float32(0.011), Float32(0.0072), Float32(0.0048), Float32(0.0034),
    ]

    var crest_h = Float32(0.0)
    var dx_w = Float32(0.0)
    var dz_w = Float32(0.0)
    var orb_x = Float32(0.0)
    var orb_y = Float32(0.0)
    var orb_z = Float32(0.0)

    for wi in range(8):
        var kf = k_arr[wi]
        var a_eff = amp_arr[wi] * shallow_damp
        var ang = Float32(wi) * Float32(2.39996323) + Float32(0.37)
        var dir_x = det_cos(ang)
        var dir_z = det_sin(ang)
        var kx = dir_x * kf
        var kz = dir_z * kf
        var om = sqrt(Float32(9.81) * kf)
        var ph = wx * kx + wz * kz + time_s * om
        if not isfinite(ph):
            continue
        var s_ph = det_sin(ph)
        var c_ph = det_cos(ph)

        crest_h += a_eff * s_ph
        var cs = c_ph * a_eff
        dx_w += cs * kx
        dz_w += cs * kz

        # Deep-water Airy orbital velocity for eta = A * sin(kx*x + kz*z + om*t)
        var v_amp = a_eff * om
        orb_x -= dir_x * (v_amp * s_ph)
        orb_z -= dir_z * (v_amp * s_ph)
        orb_y += v_amp * c_ph

    var eta = water_y + crest_h
    var normal = Vec3(-dx_w, Float32(1.0), -dz_w).normalized()
    if not (
        isfinite(eta)
        and isfinite(normal.x) and isfinite(normal.y) and isfinite(normal.z)
        and isfinite(orb_x) and isfinite(orb_y) and isfinite(orb_z)
    ):
        return (water_y, Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 0.0))
    return (eta, normal, Vec3(orb_x, orb_y, orb_z))


@fieldwise_init
struct BuoyancyWrench(Copyable, ImplicitlyCopyable, Movable):
    var force: Vec3
    var torque: Vec3
    var submerged_volume: Float32
    var submerged_fraction: Float32
    var surface_eta: Float32

    def __init__(out self):
        self.force = Vec3()
        self.torque = Vec3()
        self.submerged_volume = 0.0
        self.submerged_fraction = 0.0
        self.surface_eta = 0.0


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


def eval_sphere_buoyancy_drag(
    center: Vec3,
    radius: Float32,
    linear_vel: Vec3,
    angular_vel: Vec3,
    time_s: Float32,
    water_y: Float32 = 0.0,
    fluid_density: Float32 = RHO_SEAWATER_SI,
    gravity: Float32 = 9.81,
    cd: Float32 = 0.47,
    use_waves: Bool = True,
) -> BuoyancyWrench:
    """Computes exact spherical-cap Archimedes buoyancy and quadratic drag in Beaufort waves."""
    if not (
        isfinite(center.x) and isfinite(center.y) and isfinite(center.z)
        and isfinite(radius) and isfinite(water_y)
    ):
        return BuoyancyWrench(Vec3(), Vec3(), Float32(0.0), Float32(0.0), Float32(0.0))
    var r = max(Float32(1.0e-4), min(Float32(1.0e6), radius))
    var max_surface = water_y + (MAX_BEAUFORT_WAVE_AMP if use_waves else Float32(0.0))
    if center.y - r >= max_surface:
        return BuoyancyWrench(Vec3(), Vec3(), Float32(0.0), Float32(0.0), water_y)
    var safe_t = time_s if isfinite(time_s) else Float32(0.0)
    var safe_rho = max(Float32(0.0), fluid_density) if isfinite(fluid_density) else RHO_SEAWATER_SI
    var safe_g = gravity if isfinite(gravity) else Float32(9.81)
    var safe_cd = max(Float32(0.0), cd) if isfinite(cd) else Float32(0.47)
    var safe_lv = Vec3(
        linear_vel.x if isfinite(linear_vel.x) else Float32(0.0),
        linear_vel.y if isfinite(linear_vel.y) else Float32(0.0),
        linear_vel.z if isfinite(linear_vel.z) else Float32(0.0),
    )
    var safe_av = Vec3(
        angular_vel.x if isfinite(angular_vel.x) else Float32(0.0),
        angular_vel.y if isfinite(angular_vel.y) else Float32(0.0),
        angular_vel.z if isfinite(angular_vel.z) else Float32(0.0),
    )
    var eta = water_y
    var orb_vel = Vec3()
    if use_waves:
        var w = eval_beaufort_wave(center.x, center.z, safe_t, water_y)
        eta = w[0]
        orb_vel = w[2]

    var h_sub = max(Float32(0.0), min(Float32(2.0) * r, eta - (center.y - r)))
    if h_sub <= Float32(0.0):
        return BuoyancyWrench(Vec3(), Vec3(), Float32(0.0), Float32(0.0), eta)

    # Exact Archimedes spherical cap volume: V = (pi * h^2 / 3) * (3R - h)
    var v_full = (Float32(4.0) / Float32(3.0)) * PI * r * r * r
    var v_sub = (PI * h_sub * h_sub / Float32(3.0)) * (Float32(3.0) * r - h_sub)
    var frac = max(Float32(0.0), min(Float32(1.0), v_sub / v_full))

    var f_buoy = Vec3(0.0, safe_rho * v_sub * safe_g, 0.0)
    var v_rel = safe_lv - orb_vel
    var vx64 = Float64(v_rel.x)
    var vy64 = Float64(v_rel.y)
    var vz64 = Float64(v_rel.z)
    var rel_spd64 = sqrt(vx64 * vx64 + vy64 * vy64 + vz64 * vz64)
    var area_wet64 = Float64(PI) * Float64(r) * Float64(r) * Float64(frac)
    var drag_k64 = 0.5 * Float64(safe_rho) * Float64(safe_cd) * area_wet64 * rel_spd64
    var f_drag = _clamp_vec3_f64(-vx64 * drag_k64, -vy64 * drag_k64, -vz64 * drag_k64)
    var tau_k64 = Float64(safe_rho) * Float64(r) * Float64(r) * Float64(v_sub) * 0.8
    var tau_drag = _clamp_vec3_f64(-Float64(safe_av.x) * tau_k64, -Float64(safe_av.y) * tau_k64, -Float64(safe_av.z) * tau_k64)

    return BuoyancyWrench(f_buoy + f_drag, tau_drag, v_sub, frac, eta)


def eval_box_buoyancy_drag(
    center: Vec3,
    rot: Quat,
    half_extents: Vec3,
    linear_vel: Vec3,
    angular_vel: Vec3,
    time_s: Float32,
    water_y: Float32 = 0.0,
    fluid_density: Float32 = RHO_SEAWATER_SI,
    gravity: Float32 = 9.81,
    cd: Float32 = 1.05,
    use_waves: Bool = True,
) -> BuoyancyWrench:
    """Evaluates 27-subcell Archimedes buoyancy, metacentric righting torque, and quadratic
    hydrodynamic drag on an oriented 3D box in Beaufort waves."""
    if not (
        isfinite(center.x) and isfinite(center.y) and isfinite(center.z)
        and isfinite(rot.x) and isfinite(rot.y) and isfinite(rot.z) and isfinite(rot.w)
        and isfinite(half_extents.x) and isfinite(half_extents.y) and isfinite(half_extents.z)
        and isfinite(water_y)
    ):
        return BuoyancyWrench(Vec3(), Vec3(), Float32(0.0), Float32(0.0), Float32(0.0))
    var hx = max(Float32(1.0e-4), min(Float32(1.0e5), half_extents.x))
    var hy = max(Float32(1.0e-4), min(Float32(1.0e5), half_extents.y))
    var hz = max(Float32(1.0e-4), min(Float32(1.0e5), half_extents.z))
    var safe_t = time_s if isfinite(time_s) else Float32(0.0)
    var safe_rho = max(Float32(0.0), fluid_density) if isfinite(fluid_density) else RHO_SEAWATER_SI
    var safe_g = gravity if isfinite(gravity) else Float32(9.81)
    var safe_cd = max(Float32(0.0), cd) if isfinite(cd) else Float32(1.05)
    var safe_lv = Vec3(
        linear_vel.x if isfinite(linear_vel.x) else Float32(0.0),
        linear_vel.y if isfinite(linear_vel.y) else Float32(0.0),
        linear_vel.z if isfinite(linear_vel.z) else Float32(0.0),
    )
    var safe_av = Vec3(
        angular_vel.x if isfinite(angular_vel.x) else Float32(0.0),
        angular_vel.y if isfinite(angular_vel.y) else Float32(0.0),
        angular_vel.z if isfinite(angular_vel.z) else Float32(0.0),
    )
    var v_full = Float32(8.0) * hx * hy * hz
    var v_cell = v_full * Float32(1.0 / 27.0)

    # Vertical span of the rotated box and of a 1/3 subcell
    var u0 = rot.rotate(Vec3(1, 0, 0))
    var u1 = rot.rotate(Vec3(0, 1, 0))
    var u2 = rot.rotate(Vec3(0, 0, 1))
    var box_hy = abs(u0.y) * hx + abs(u1.y) * hy + abs(u2.y) * hz
    var max_surface = water_y + (MAX_BEAUFORT_WAVE_AMP if use_waves else Float32(0.0))
    if center.y - box_hy >= max_surface:
        return BuoyancyWrench(Vec3(), Vec3(), Float32(0.0), Float32(0.0), water_y)

    var cell_hy = max(Float32(1.0e-4), box_hy * Float32(2.0 / 3.0))
    var cell_area = (Float32(4.0) * (hx * hy + hy * hz + hz * hx) * Float32(1.0 / 3.0)) * Float32(1.0 / 9.0)

    var center_eta = water_y
    if use_waves:
        center_eta = eval_beaufort_wave(center.x, center.z, safe_t, water_y)[0]

    var total_fx64 = 0.0
    var total_fy64 = 0.0
    var total_fz64 = 0.0
    var total_tx64 = 0.0
    var total_ty64 = 0.0
    var total_tz64 = 0.0
    var v_sub_sum = Float32(0.0)

    var coords = [Float32(-2.0 / 3.0), Float32(0.0), Float32(2.0 / 3.0)]
    for ix in range(3):
        var lx = coords[ix] * hx
        for iy in range(3):
            var ly = coords[iy] * hy
            for iz in range(3):
                var lz = coords[iz] * hz
                var r_arm = rot.rotate(Vec3(lx, ly, lz))
                var p_cell = center + r_arm
                var eta = water_y
                var orb = Vec3()
                if use_waves:
                    var w = eval_beaufort_wave(p_cell.x, p_cell.z, safe_t, water_y)
                    eta = w[0]
                    orb = w[2]

                var sub_f = max(
                    Float32(0.0),
                    min(Float32(1.0), (eta - (p_cell.y - Float32(0.5) * cell_hy)) / cell_hy),
                )
                if sub_f <= Float32(0.0):
                    continue

                var dv_vol = v_cell * sub_f
                v_sub_sum += dv_vol
                var fy_b = Float64(safe_rho) * Float64(dv_vol) * Float64(safe_g)

                var v_pt = safe_lv + safe_av.cross(r_arm) - orb
                var vpx64 = Float64(v_pt.x)
                var vpy64 = Float64(v_pt.y)
                var vpz64 = Float64(v_pt.z)
                var spd64 = sqrt(vpx64 * vpx64 + vpy64 * vpy64 + vpz64 * vpz64)
                var dk64 = 0.5 * Float64(safe_rho) * Float64(safe_cd) * Float64(cell_area) * Float64(sub_f) * spd64
                var fx_c = -vpx64 * dk64
                var fy_c = fy_b - vpy64 * dk64
                var fz_c = -vpz64 * dk64

                total_fx64 += fx_c
                total_fy64 += fy_c
                total_fz64 += fz_c
                total_tx64 += Float64(r_arm.y) * fz_c - Float64(r_arm.z) * fy_c
                total_ty64 += Float64(r_arm.z) * fx_c - Float64(r_arm.x) * fz_c
                total_tz64 += Float64(r_arm.x) * fy_c - Float64(r_arm.y) * fx_c

    var frac = max(Float32(0.0), min(Float32(1.0), v_sub_sum / v_full))
    if frac > Float32(0.0):
        var max_h = max(hx, max(hy, hz))
        var ang_k64 = Float64(safe_rho) * Float64(max_h) * Float64(max_h) * Float64(v_sub_sum) * 0.4
        total_tx64 -= Float64(safe_av.x) * ang_k64
        total_ty64 -= Float64(safe_av.y) * ang_k64
        total_tz64 -= Float64(safe_av.z) * ang_k64

    return BuoyancyWrench(
        _clamp_vec3_f64(total_fx64, total_fy64, total_fz64),
        _clamp_vec3_f64(total_tx64, total_ty64, total_tz64),
        v_sub_sum,
        frac,
        center_eta,
    )
