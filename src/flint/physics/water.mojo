"""Water physics (v4): 8-wave Golden-Angle Beaufort spectrum matching `aaa_water.mojo`,
Archimedes submerged sphere/box buoyancy, metacentric righting torque, and quadratic drag.

Preserves flat-water helpers (`NO_WATER`, `WATER_DENSITY`, `WATER_DRAG`, `floats`,
`submerged_fraction`, `buoyant_acceleration`) for legacy voxel/field callers while
providing SI Beaufort wave kinematics and rigid-body hydrodynamics.
"""
from std.math import sqrt, max, min
from flint.core.math import Vec3, Quat
from flint.core.detmath import det_sin, det_cos, PI

comptime NO_WATER: Float32 = -1000
comptime WATER_DENSITY: Float32 = 1.6
comptime WATER_DRAG: Float32 = 2.4
comptime RHO_SEAWATER_SI: Float32 = 1025.0
comptime RHO_FRESHWATER_SI: Float32 = 1000.0


def floats(material_density: Float32) -> Bool:
    return material_density > 0 and material_density < WATER_DENSITY


def submerged_fraction(bottom: Float32, height: Float32, water_level: Float32) -> Float32:
    if water_level <= NO_WATER + 1 or height <= 0:
        return 0
    return max(Float32(0), min(Float32(1), (water_level - bottom) / height))


def buoyant_acceleration(fraction: Float32, material_density: Float32, gravity: Float32) -> Float32:
    """Upward acceleration from displaced water (Archimedes)."""
    if fraction <= 0 or material_density <= 0:
        return 0
    return gravity * fraction * (WATER_DENSITY / material_density)


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
    var r = max(Float32(1.0e-4), radius)
    var eta = water_y
    var orb_vel = Vec3()
    if use_waves:
        var w = eval_beaufort_wave(center.x, center.z, time_s, water_y)
        eta = w[0]
        orb_vel = w[2]

    var h_sub = max(Float32(0.0), min(Float32(2.0) * r, eta - (center.y - r)))
    if h_sub <= Float32(0.0):
        return BuoyancyWrench(Vec3(), Vec3(), Float32(0.0), Float32(0.0), eta)

    # Exact Archimedes spherical cap volume: V = (pi * h^2 / 3) * (3R - h)
    var v_full = (Float32(4.0) / Float32(3.0)) * PI * r * r * r
    var v_sub = (PI * h_sub * h_sub / Float32(3.0)) * (Float32(3.0) * r - h_sub)
    var frac = max(Float32(0.0), min(Float32(1.0), v_sub / v_full))

    var f_buoy = Vec3(0.0, fluid_density * v_sub * gravity, 0.0)
    var v_rel = linear_vel - orb_vel
    var rel_spd = v_rel.length()
    var area_wet = PI * r * r * frac
    var f_drag = -v_rel * (Float32(0.5) * fluid_density * cd * area_wet * rel_spd)
    var tau_drag = -angular_vel * (fluid_density * r * r * v_sub * Float32(0.8))

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
    var hx = max(Float32(1.0e-4), half_extents.x)
    var hy = max(Float32(1.0e-4), half_extents.y)
    var hz = max(Float32(1.0e-4), half_extents.z)
    var v_full = Float32(8.0) * hx * hy * hz
    var v_cell = v_full * Float32(1.0 / 27.0)

    # Vertical span of a rotated 1/3 subcell
    var u0 = rot.rotate(Vec3(1, 0, 0))
    var u1 = rot.rotate(Vec3(0, 1, 0))
    var u2 = rot.rotate(Vec3(0, 0, 1))
    var cell_hy = max(
        Float32(1.0e-4),
        (abs(u0.y) * hx + abs(u1.y) * hy + abs(u2.y) * hz) * Float32(2.0 / 3.0),
    )
    var cell_area = (Float32(4.0) * (hx * hy + hy * hz + hz * hx) * Float32(1.0 / 3.0)) * Float32(1.0 / 9.0)

    var center_eta = water_y
    if use_waves:
        center_eta = eval_beaufort_wave(center.x, center.z, time_s, water_y)[0]

    var total_f = Vec3()
    var total_tau = Vec3()
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
                    var w = eval_beaufort_wave(p_cell.x, p_cell.z, time_s, water_y)
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
                var f_b = Vec3(0.0, fluid_density * dv_vol * gravity, 0.0)

                var v_pt = linear_vel + angular_vel.cross(r_arm) - orb
                var spd = v_pt.length()
                var f_d = -v_pt * (Float32(0.5) * fluid_density * cd * cell_area * sub_f * spd)

                var f_cell = f_b + f_d
                total_f = total_f + f_cell
                total_tau = total_tau + r_arm.cross(f_cell)

    var frac = max(Float32(0.0), min(Float32(1.0), v_sub_sum / v_full))
    if frac > Float32(0.0):
        var max_h = max(hx, max(hy, hz))
        total_tau = total_tau - angular_vel * (fluid_density * max_h * max_h * v_sub_sum * Float32(0.4))

    return BuoyancyWrench(total_f, total_tau, v_sub_sum, frac, center_eta)
