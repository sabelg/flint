"""Mass, centre of mass and inertia of a voxel body from its voxels (package 2.11).

Every voxel adds its material's density (thousandths: the material table's
mass_density, assets/deepsong/materials.json, through density_table.mojo) to
integer sums: the mass, the first moments and the second moments, with voxel
centres in doubled grid units (2x + 1) so every term is a whole number. Integer
sums are exact and independent of order, so a body's mass properties are the
same however its voxels were added or removed, and an edit to one voxel updates
the sums in O(1) (`add` / `remove`) instead of a pass over the grid.

From the sums: the centre of mass, the inertia tensor about it (each voxel a
unit cube: its own I = m/6 on every axis, plus the parallel-axis terms), and
the principal axes by a cyclic Jacobi eigen solve with a fixed number of
sweeps (deterministic: the same sums give the same bits). The body keeps its
grid frame as its orientation; the principal frame is a rotation relative to
it (`MassProperties.frame`), which the solver applies around the diagonal
inverse inertia.

Mass keeps the 2.1 scale: MASS_SCALE per thousandth, so a stone voxel (2500)
weighs 0.3, as the 2.1 sum of densities times 0.12 did.
"""
from std.math import sqrt
from flint.core.math import Vec3, Quat
from flint.physics.density_table import material_density_milli, DENSITY_MATERIALS

comptime MASS_SCALE: Float64 = 0.12 / 1000.0  # body mass per thousandth of density
comptime MIN_BODY_MASS: Float32 = 0.05
comptime JACOBI_SWEEPS: Int = 12


struct DensityTable(Copyable, Movable):
    """The density module as a 256-entry lookup (one per material byte)."""
    var milli: List[Int]

    def __init__(out self):
        self.milli = List[Int](length=256, fill=0)
        for material in range(DENSITY_MATERIALS):
            self.milli[material] = material_density_milli(material)

    @always_inline
    def get(self, material: UInt8) -> Int:
        return self.milli[Int(material)]


struct MassSums(Copyable, ImplicitlyCopyable, Movable, Equatable):
    """Integer moments of a voxel grid. Positions are doubled cell centres
    (u = 2x + 1), so first moments are in half voxels and second moments in
    quarter voxels squared."""
    var m: Int
    var voxels: Int
    var mx: Int
    var my: Int
    var mz: Int
    var mxx: Int
    var myy: Int
    var mzz: Int
    var mxy: Int
    var mxz: Int
    var myz: Int

    def __init__(out self):
        self.m = 0
        self.voxels = 0
        self.mx = 0
        self.my = 0
        self.mz = 0
        self.mxx = 0
        self.myy = 0
        self.mzz = 0
        self.mxy = 0
        self.mxz = 0
        self.myz = 0

    def __eq__(self, other: Self) -> Bool:
        return (
            self.m == other.m and self.voxels == other.voxels and self.mx == other.mx and self.my == other.my
            and self.mz == other.mz and self.mxx == other.mxx and self.myy == other.myy and self.mzz == other.mzz
            and self.mxy == other.mxy and self.mxz == other.mxz and self.myz == other.myz
        )

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def change(mut self, x: Int, y: Int, z: Int, density: Int, sign: Int):
        var u = 2 * x + 1
        var v = 2 * y + 1
        var w = 2 * z + 1
        var m = density * sign
        self.m += m
        self.voxels += sign
        self.mx += m * u
        self.my += m * v
        self.mz += m * w
        self.mxx += m * u * u
        self.myy += m * v * v
        self.mzz += m * w * w
        self.mxy += m * u * v
        self.mxz += m * u * w
        self.myz += m * v * w

    def add(mut self, x: Int, y: Int, z: Int, density: Int):
        self.change(x, y, z, density, 1)

    def remove(mut self, x: Int, y: Int, z: Int, density: Int):
        self.change(x, y, z, density, -1)


def grid_mass_sums(cells: List[UInt8], base: Int, size_x: Int, size_y: Int, size_z: Int, table: DensityTable) -> MassSums:
    """The sums over a grid stored y, z, x from `base` (a body's block of
    `PhysicsWorld.shape_cells`, or a PieceGrid's cells with base 0)."""
    var sums = MassSums()
    for y in range(size_y):
        for z in range(size_z):
            for x in range(size_x):
                var material = cells[base + (y * size_z + z) * size_x + x]
                if material != 0:
                    sums.add(x, y, z, table.get(material))
    return sums


struct MassProperties(Copyable, ImplicitlyCopyable, Movable):
    var mass: Float32
    var inverse_mass: Float32
    var center: Vec3            # centre of mass in grid units (corner at 0)
    var moments: Vec3           # principal moments of inertia
    var inverse_inertia: Vec3   # their inverses
    var frame: Quat             # principal axes, relative to the grid frame

    def __init__(out self):
        self.mass = 1
        self.inverse_mass = 1
        self.center = Vec3()
        self.moments = Vec3(1, 1, 1)
        self.inverse_inertia = Vec3(1, 1, 1)
        self.frame = Quat()


def jacobi_rotate(mut a: List[Float64], mut v: List[Float64], p: Int, q: Int):
    """One Jacobi rotation zeroing a[p][q] of the symmetric 3x3 `a` (row
    major), accumulated into the column eigenvectors `v`."""
    var apq = a[p * 3 + q]
    if apq == 0:
        return
    var app = a[p * 3 + p]
    var aqq = a[q * 3 + q]
    var theta = (aqq - app) / (2.0 * apq)
    var t = 1.0 / (abs(theta) + sqrt(theta * theta + 1.0))
    if theta < 0:
        t = -t
    var c = 1.0 / sqrt(t * t + 1.0)
    var s = t * c
    for k in range(3):
        var akp = a[k * 3 + p]
        var akq = a[k * 3 + q]
        a[k * 3 + p] = c * akp - s * akq
        a[k * 3 + q] = s * akp + c * akq
    for k in range(3):
        var apk = a[p * 3 + k]
        var aqk = a[q * 3 + k]
        a[p * 3 + k] = c * apk - s * aqk
        a[q * 3 + k] = s * apk + c * aqk
    for k in range(3):
        var vkp = v[k * 3 + p]
        var vkq = v[k * 3 + q]
        v[k * 3 + p] = c * vkp - s * vkq
        v[k * 3 + q] = s * vkp + c * vkq


def matrix_to_quat(m: List[Float64]) -> Quat:
    """A rotation matrix (row major, columns are the rotated axes) as a unit quaternion."""
    var trace = m[0] + m[4] + m[8]
    var w: Float64
    var x: Float64
    var y: Float64
    var z: Float64
    if trace > 0:
        var s = sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[7] - m[5]) / s
        y = (m[2] - m[6]) / s
        z = (m[3] - m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        var s = sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0
        w = (m[7] - m[5]) / s
        x = 0.25 * s
        y = (m[1] + m[3]) / s
        z = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        var s = sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0
        w = (m[2] - m[6]) / s
        x = (m[1] + m[3]) / s
        y = 0.25 * s
        z = (m[5] + m[7]) / s
    else:
        var s = sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0
        w = (m[3] - m[1]) / s
        x = (m[2] + m[6]) / s
        y = (m[5] + m[7]) / s
        z = 0.25 * s
    var length = sqrt(w * w + x * x + y * y + z * z)
    return Quat(Float32(x / length), Float32(y / length), Float32(z / length), Float32(w / length))


def inertia_tensor(sums: MassSums) -> List[Float64]:
    """The inertia tensor about the centre of mass, in thousandths of density
    times voxels squared (row major)."""
    var m = Float64(sums.m)
    var cx = Float64(sums.mx) / (2.0 * m)
    var cy = Float64(sums.my) / (2.0 * m)
    var cz = Float64(sums.mz) / (2.0 * m)
    var sxx = Float64(sums.mxx) * 0.25
    var syy = Float64(sums.myy) * 0.25
    var szz = Float64(sums.mzz) * 0.25
    var sxy = Float64(sums.mxy) * 0.25
    var sxz = Float64(sums.mxz) * 0.25
    var syz = Float64(sums.myz) * 0.25
    var own = m / 6.0  # each voxel is a unit cube
    var t = List[Float64](length=9, fill=0)
    t[0] = (syy + szz) - m * (cy * cy + cz * cz) + own
    t[4] = (sxx + szz) - m * (cx * cx + cz * cz) + own
    t[8] = (sxx + syy) - m * (cx * cx + cy * cy) + own
    t[1] = -(sxy - m * cx * cy)
    t[2] = -(sxz - m * cx * cz)
    t[5] = -(syz - m * cy * cz)
    t[3] = t[1]
    t[6] = t[2]
    t[7] = t[5]
    return t^


def mass_properties(sums: MassSums) -> MassProperties:
    """Mass, centre and principal inertia from the sums. An empty or weightless
    grid gets the smallest body mass at the grid's corner."""
    var result = MassProperties()
    if sums.m <= 0 or sums.voxels <= 0:
        result.mass = MIN_BODY_MASS
        result.inverse_mass = 1.0 / MIN_BODY_MASS
        return result
    var m = Float64(sums.m)
    result.mass = max(MIN_BODY_MASS, Float32(m * MASS_SCALE))
    result.inverse_mass = 1.0 / result.mass
    result.center = Vec3(Float32(Float64(sums.mx) / (2.0 * m)), Float32(Float64(sums.my) / (2.0 * m)), Float32(Float64(sums.mz) / (2.0 * m)))
    var a = inertia_tensor(sums)
    var v = List[Float64](length=9, fill=0)
    v[0] = 1
    v[4] = 1
    v[8] = 1
    for _ in range(JACOBI_SWEEPS):
        var off = abs(a[1]) + abs(a[2]) + abs(a[5])
        if off <= 1.0e-12 * (abs(a[0]) + abs(a[4]) + abs(a[8])):
            break
        jacobi_rotate(a, v, 0, 1)
        jacobi_rotate(a, v, 0, 2)
        jacobi_rotate(a, v, 1, 2)
    # A right-handed frame: flip the third axis of a reflection.
    var det = (
        v[0] * (v[4] * v[8] - v[5] * v[7])
        - v[1] * (v[3] * v[8] - v[5] * v[6])
        + v[2] * (v[3] * v[7] - v[4] * v[6])
    )
    if det < 0:
        v[2] = -v[2]
        v[5] = -v[5]
        v[8] = -v[8]
    result.frame = matrix_to_quat(v)
    var scale = MASS_SCALE
    var floor_moment = Float64(MIN_BODY_MASS) * 0.01
    var ix = max(floor_moment, a[0] * scale)
    var iy = max(floor_moment, a[4] * scale)
    var iz = max(floor_moment, a[8] * scale)
    result.moments = Vec3(Float32(ix), Float32(iy), Float32(iz))
    result.inverse_inertia = Vec3(Float32(1.0 / ix), Float32(1.0 / iy), Float32(1.0 / iz))
    return result
