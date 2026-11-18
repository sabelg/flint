"""The awake set in columns, and the solver's 8-wide arithmetic (v4).

`SolverBodies` holds the bodies the solver moves this tick (awake, active, not
quarantined), one column per field, in body-row order and padded with zero
rows to a multiple of LANES, followed by one more batch of zeros (`static_index`).

v4 Upgrades:
1. Multiplicative delta quaternion `(dqx, dqy, dqz, dqw)` initialized to `(0,0,0,1)`
   and integrated without trigonometric functions (`dq = normalize(dq + 0.5*(w*h)*dq)`),
   eliminating lever-arm stretching and `std.math.sin/cos` platform drift.
2. Unconditionally stable 1-step Implicit Newton-Raphson Gyroscopic Torque in `F8` SIMD:
   `I_b * (w_b^+ - w_b) + h * (w_b^+ x (I_b * w_b^+)) = 0`, preserving rotational
   energy and capturing the Dzhanibekov intermediate-axis flip without energy blowup.
"""
from std.math import sqrt, abs, max
from flint.core.math import Vec3, Quat

comptime LANES: Int = 8
comptime F8 = SIMD[DType.float32, LANES]
comptime I8 = SIMD[DType.int32, LANES]


@fieldwise_init
struct V8(Copyable, ImplicitlyCopyable, Movable):
    """Eight 3-vectors, one per lane."""
    var x: F8
    var y: F8
    var z: F8

    @staticmethod
    @always_inline
    def zero() -> Self:
        return Self(F8(0), F8(0), F8(0))

    @staticmethod
    @always_inline
    def splat(v: Vec3) -> Self:
        return Self(F8(v.x), F8(v.y), F8(v.z))

    @always_inline
    def __add__(self, o: Self) -> Self:
        return Self(self.x + o.x, self.y + o.y, self.z + o.z)

    @always_inline
    def __sub__(self, o: Self) -> Self:
        return Self(self.x - o.x, self.y - o.y, self.z - o.z)

    @always_inline
    def scale(self, s: F8) -> Self:
        return Self(self.x * s, self.y * s, self.z * s)

    @always_inline
    def dot(self, o: Self) -> F8:
        return self.x * o.x + self.y * o.y + self.z * o.z

    @always_inline
    def cross(self, o: Self) -> Self:
        return Self(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)


@fieldwise_init
struct S8(Copyable, ImplicitlyCopyable, Movable):
    """Eight symmetric 3x3 matrices (world inverse inertia tensors)."""
    var xx: F8
    var yy: F8
    var zz: F8
    var xy: F8
    var xz: F8
    var yz: F8

    @always_inline
    def apply(self, v: V8) -> V8:
        return V8(
            self.xx * v.x + self.xy * v.y + self.xz * v.z,
            self.xy * v.x + self.yy * v.y + self.yz * v.z,
            self.xz * v.x + self.yz * v.y + self.zz * v.z,
        )


@always_inline
def ld(values: List[Float32], at: Int) -> F8:
    return values.unsafe_ptr().unsafe_load[width=LANES](at)


@always_inline
def st(mut values: List[Float32], at: Int, value: F8):
    values.unsafe_ptr().unsafe_offset(at).unsafe_store(value)


@always_inline
def ld_i(values: List[Int32], at: Int) -> I8:
    return values.unsafe_ptr().unsafe_load[width=LANES](at)


@always_inline
def gather(values: List[Float32], index: I8) -> F8:
    var out = F8(0)
    comptime for lane in range(LANES):
        out[lane] = values[Int(index[lane])]
    return out


@always_inline
def scatter(mut values: List[Float32], index: I8, value: F8, limit: Int):
    """Writes the lanes whose index is a moving body (below `limit`)."""
    comptime for lane in range(LANES):
        var at = Int(index[lane])
        if at < limit:
            values[at] = value[lane]


struct Sym3(Copyable, ImplicitlyCopyable, Movable):
    """A symmetric 3x3 matrix, scalar (joints, inertia set-up)."""
    var xx: Float32
    var yy: Float32
    var zz: Float32
    var xy: Float32
    var xz: Float32
    var yz: Float32

    def __init__(out self, xx: Float32 = 0, yy: Float32 = 0, zz: Float32 = 0, xy: Float32 = 0, xz: Float32 = 0, yz: Float32 = 0):
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.xy = xy
        self.xz = xz
        self.yz = yz

    def apply(self, v: Vec3) -> Vec3:
        return Vec3(
            self.xx * v.x + self.xy * v.y + self.xz * v.z,
            self.xy * v.x + self.yy * v.y + self.yz * v.z,
            self.xz * v.x + self.yz * v.y + self.zz * v.z,
        )

    def scale(self, s: Float32) -> Self:
        return Self(self.xx * s, self.yy * s, self.zz * s, self.xy * s, self.xz * s, self.yz * s)

    def plus(self, o: Self) -> Self:
        return Self(self.xx + o.xx, self.yy + o.yy, self.zz + o.zz, self.xy + o.xy, self.xz + o.xz, self.yz + o.yz)

    def inverse(self) -> Self:
        """The inverse, or zero when the matrix is (nearly) singular."""
        var c00 = self.yy * self.zz - self.yz * self.yz
        var c01 = self.xz * self.yz - self.xy * self.zz
        var c02 = self.xy * self.yz - self.xz * self.yy
        var det = self.xx * c00 + self.xy * c01 + self.xz * c02
        if abs(det) < 1.0e-12:
            return Self()
        var inv = 1.0 / det
        var c11 = self.xx * self.zz - self.xz * self.xz
        var c12 = self.xy * self.xz - self.xx * self.yz
        var c22 = self.xx * self.yy - self.xy * self.xy
        return Self(c00 * inv, c11 * inv, c22 * inv, c01 * inv, c02 * inv, c12 * inv)


def skew_sandwich(r: Vec3, i: Sym3) -> Sym3:
    """-[r]x I [r]x: the rotational part of a point constraint's mass matrix,
    which is symmetric for a symmetric I."""
    var a00 = -r.z * i.xy + r.y * i.xz
    var a01 = -r.z * i.yy + r.y * i.yz
    var a02 = -r.z * i.yz + r.y * i.zz
    var a10 = r.z * i.xx - r.x * i.xz
    var a11 = r.z * i.xy - r.x * i.yz
    var a12 = r.z * i.xz - r.x * i.zz
    var a20 = -r.y * i.xx + r.x * i.xy
    var a21 = -r.y * i.xy + r.x * i.yy
    var m00 = a01 * r.z - a02 * r.y
    var m11 = -a10 * r.z + a12 * r.x
    var m22 = a20 * r.y - a21 * r.x
    var m01 = -a00 * r.z + a02 * r.x
    var m02 = a00 * r.y - a01 * r.x
    var m12 = a10 * r.y - a11 * r.x
    return Sym3(-m00, -m11, -m22, -m01, -m02, -m12)


def rotate_vec(q: Quat, v: Vec3) -> Vec3:
    var u = Vec3(q.x, q.y, q.z)
    var t = Vec3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x) * 2.0
    var c = Vec3(u.y * t.z - u.z * t.y, u.z * t.x - u.x * t.z, u.x * t.y - u.y * t.x)
    return v + t * q.w + c


def world_inverse_inertia(frame: Quat, local: Vec3) -> Sym3:
    """R diag(local) R^T for the principal frame R."""
    var ex = rotate_vec(frame, Vec3(1, 0, 0))
    var ey = rotate_vec(frame, Vec3(0, 1, 0))
    var ez = rotate_vec(frame, Vec3(0, 0, 1))
    return Sym3(
        ex.x * ex.x * local.x + ey.x * ey.x * local.y + ez.x * ez.x * local.z,
        ex.y * ex.y * local.x + ey.y * ey.y * local.y + ez.y * ez.y * local.z,
        ex.z * ex.z * local.x + ey.z * ey.z * local.y + ez.z * ez.z * local.z,
        ex.x * ex.y * local.x + ey.x * ey.y * local.y + ez.x * ez.y * local.z,
        ex.x * ex.z * local.x + ey.x * ey.z * local.y + ez.x * ez.z * local.z,
        ex.y * ex.z * local.x + ey.y * ey.z * local.y + ez.y * ez.z * local.z,
    )


@always_inline
def _quat_rotate_v8(qx: F8, qy: F8, qz: F8, qw: F8, v: V8) -> V8:
    var u = V8(qx, qy, qz)
    var t = u.cross(v).scale(F8(2.0))
    return v + t.scale(qw) + u.cross(t)


struct SolverBodies(Movable):
    var count: Int         # moving bodies
    var padded: Int        # count rounded up to LANES
    var static_index: Int  # the zero body
    var row: List[Int]     # solver index -> body row
    var vx: List[Float32]
    var vy: List[Float32]
    var vz: List[Float32]
    var wx: List[Float32]
    var wy: List[Float32]
    var wz: List[Float32]
    var dpx: List[Float32]
    var dpy: List[Float32]
    var dpz: List[Float32]
    # Multiplicative delta quaternion since the tick began: (0, 0, 0, 1) at start of tick
    var dqx: List[Float32]
    var dqy: List[Float32]
    var dqz: List[Float32]
    var dqw: List[Float32]
    var px: List[Float32]   # position at the start of the tick (the centre of mass)
    var py: List[Float32]
    var pz: List[Float32]
    var qx: List[Float32]   # orientation now
    var qy: List[Float32]
    var qz: List[Float32]
    var qw: List[Float32]
    var inv_mass: List[Float32]
    var ixx: List[Float32]
    var iyy: List[Float32]
    var izz: List[Float32]
    var ixy: List[Float32]
    var ixz: List[Float32]
    var iyz: List[Float32]
    # Optional principal body-frame moments of inertia (Ix, Iy, Iz) for implicit gyroscopics
    var ib_x: List[Float32]
    var ib_y: List[Float32]
    var ib_z: List[Float32]
    var lin_damp: List[Float32]
    var ang_damp: List[Float32]
    var gravity: List[Float32]   # m/s2 down; 0 on padding rows
    var ax: List[Float32]
    var ay: List[Float32]
    var az: List[Float32]
    var aax: List[Float32]
    var aay: List[Float32]
    var aaz: List[Float32]
    var tvx: List[Float32]
    var tvy: List[Float32]
    var tvz: List[Float32]
    var rate: List[Float32]
    var arate: List[Float32]
    var field: List[Float32]

    def __init__(out self):
        self.count = 0
        self.padded = 0
        self.static_index = 0
        self.row = List[Int]()
        self.vx = List[Float32]()
        self.vy = List[Float32]()
        self.vz = List[Float32]()
        self.wx = List[Float32]()
        self.wy = List[Float32]()
        self.wz = List[Float32]()
        self.dpx = List[Float32]()
        self.dpy = List[Float32]()
        self.dpz = List[Float32]()
        self.dqx = List[Float32]()
        self.dqy = List[Float32]()
        self.dqz = List[Float32]()
        self.dqw = List[Float32]()
        self.px = List[Float32]()
        self.py = List[Float32]()
        self.pz = List[Float32]()
        self.qx = List[Float32]()
        self.qy = List[Float32]()
        self.qz = List[Float32]()
        self.qw = List[Float32]()
        self.inv_mass = List[Float32]()
        self.ixx = List[Float32]()
        self.iyy = List[Float32]()
        self.izz = List[Float32]()
        self.ixy = List[Float32]()
        self.ixz = List[Float32]()
        self.iyz = List[Float32]()
        self.ib_x = List[Float32]()
        self.ib_y = List[Float32]()
        self.ib_z = List[Float32]()
        self.lin_damp = List[Float32]()
        self.ang_damp = List[Float32]()
        self.gravity = List[Float32]()
        self.ax = List[Float32]()
        self.ay = List[Float32]()
        self.az = List[Float32]()
        self.aax = List[Float32]()
        self.aay = List[Float32]()
        self.aaz = List[Float32]()
        self.tvx = List[Float32]()
        self.tvy = List[Float32]()
        self.tvz = List[Float32]()
        self.rate = List[Float32]()
        self.arate = List[Float32]()
        self.field = List[Float32]()

    def reset(mut self, count: Int):
        """Room for `count` bodies, every column zero (padding and the zero
        body included; orientation and delta-quaternion columns hold identity)."""
        self.count = count
        self.padded = ((count + LANES - 1) // LANES) * LANES
        self.static_index = self.padded
        var length = self.padded + LANES
        self.row = List[Int](length=count, fill=-1)
        self.vx = List[Float32](length=length, fill=0)
        self.vy = List[Float32](length=length, fill=0)
        self.vz = List[Float32](length=length, fill=0)
        self.wx = List[Float32](length=length, fill=0)
        self.wy = List[Float32](length=length, fill=0)
        self.wz = List[Float32](length=length, fill=0)
        self.dpx = List[Float32](length=length, fill=0)
        self.dpy = List[Float32](length=length, fill=0)
        self.dpz = List[Float32](length=length, fill=0)
        self.dqx = List[Float32](length=length, fill=0)
        self.dqy = List[Float32](length=length, fill=0)
        self.dqz = List[Float32](length=length, fill=0)
        self.dqw = List[Float32](length=length, fill=1)
        self.px = List[Float32](length=length, fill=0)
        self.py = List[Float32](length=length, fill=0)
        self.pz = List[Float32](length=length, fill=0)
        self.qx = List[Float32](length=length, fill=0)
        self.qy = List[Float32](length=length, fill=0)
        self.qz = List[Float32](length=length, fill=0)
        self.qw = List[Float32](length=length, fill=1)
        self.inv_mass = List[Float32](length=length, fill=0)
        self.ixx = List[Float32](length=length, fill=0)
        self.iyy = List[Float32](length=length, fill=0)
        self.izz = List[Float32](length=length, fill=0)
        self.ixy = List[Float32](length=length, fill=0)
        self.ixz = List[Float32](length=length, fill=0)
        self.iyz = List[Float32](length=length, fill=0)
        self.ib_x = List[Float32](length=length, fill=0)
        self.ib_y = List[Float32](length=length, fill=0)
        self.ib_z = List[Float32](length=length, fill=0)
        self.lin_damp = List[Float32](length=length, fill=0)
        self.ang_damp = List[Float32](length=length, fill=0)
        self.gravity = List[Float32](length=length, fill=0)
        self.ax = List[Float32](length=length, fill=0)
        self.ay = List[Float32](length=length, fill=0)
        self.az = List[Float32](length=length, fill=0)
        self.aax = List[Float32](length=length, fill=0)
        self.aay = List[Float32](length=length, fill=0)
        self.aaz = List[Float32](length=length, fill=0)
        self.tvx = List[Float32](length=length, fill=0)
        self.tvy = List[Float32](length=length, fill=0)
        self.tvz = List[Float32](length=length, fill=0)
        self.rate = List[Float32](length=length, fill=0)
        self.arate = List[Float32](length=length, fill=0)
        self.field = List[Float32](length=length, fill=0)

    def set_body_inertia(mut self, i: Int, moments: Vec3):
        """Sets principal body-frame inertia moments `(Ix, Iy, Iz)` and updates world inverse inertia."""
        if i < 0 or i >= self.count:
            return
        self.ib_x[i] = moments.x
        self.ib_y[i] = moments.y
        self.ib_z[i] = moments.z
        var inv_loc = Vec3(
            Float32(1.0) / moments.x if moments.x > Float32(1.0e-8) else Float32(0.0),
            Float32(1.0) / moments.y if moments.y > Float32(1.0e-8) else Float32(0.0),
            Float32(1.0) / moments.z if moments.z > Float32(1.0e-8) else Float32(0.0),
        )
        var wi = world_inverse_inertia(self.orientation(i), inv_loc)
        self.ixx[i] = wi.xx
        self.iyy[i] = wi.yy
        self.izz[i] = wi.zz
        self.ixy[i] = wi.xy
        self.ixz[i] = wi.xz
        self.iyz[i] = wi.yz

    def velocity(self, i: Int) -> Vec3:
        return Vec3(self.vx[i], self.vy[i], self.vz[i])

    def angular(self, i: Int) -> Vec3:
        return Vec3(self.wx[i], self.wy[i], self.wz[i])

    def set_velocity(mut self, i: Int, v: Vec3, w: Vec3):
        if i >= self.count:
            return
        self.vx[i] = v.x
        self.vy[i] = v.y
        self.vz[i] = v.z
        self.wx[i] = w.x
        self.wy[i] = w.y
        self.wz[i] = w.z

    def inertia(self, i: Int) -> Sym3:
        return Sym3(self.ixx[i], self.iyy[i], self.izz[i], self.ixy[i], self.ixz[i], self.iyz[i])

    def orientation(self, i: Int) -> Quat:
        return Quat(self.qx[i], self.qy[i], self.qz[i], self.qw[i])

    def delta_orientation(self, i: Int) -> Quat:
        return Quat(self.dqx[i], self.dqy[i], self.dqz[i], self.dqw[i])

    def center(self, i: Int) -> Vec3:
        """The centre of mass now."""
        return Vec3(self.px[i] + self.dpx[i], self.py[i] + self.dpy[i], self.pz[i] + self.dpz[i])

    def integrate_velocities(mut self, batch: Int, h: Float32, max_speed: Float32, max_turn: Float32):
        """Gravity, fields, damping, 1-step Implicit Newton-Raphson Gyroscopic Torque
        (`I_b*(w_b^+ - w_b) + h*(w_b^+ x I_b*w_b^+) = 0`), and speed clamps in `F8` SIMD."""
        var at = batch * LANES
        var v = V8(ld(self.vx, at), ld(self.vy, at), ld(self.vz, at))
        var w = V8(ld(self.wx, at), ld(self.wy, at), ld(self.wz, at))
        var hv = F8(h)
        v.y = v.y - ld(self.gravity, at) * hv
        var field = ld(self.field, at).gt(F8(0))
        if field.reduce_or():
            var fv = v + V8(ld(self.ax, at), ld(self.ay, at), ld(self.az, at)).scale(hv)
            var fw = w + V8(ld(self.aax, at), ld(self.aay, at), ld(self.aaz, at)).scale(hv)
            var target = V8(ld(self.tvx, at), ld(self.tvy, at), ld(self.tvz, at))
            var keep = max(F8(0), F8(1) - ld(self.rate, at) * hv)
            fv = target + (fv - target).scale(keep)
            fw = fw.scale(max(F8(0), F8(1) - ld(self.arate, at) * hv))
            v = V8(field.select(fv.x, v.x), field.select(fv.y, v.y), field.select(fv.z, v.z))
            w = V8(field.select(fw.x, w.x), field.select(fw.y, w.y), field.select(fw.z, w.z))
        v = v.scale(max(F8(0), F8(1) - ld(self.lin_damp, at) * hv))
        w = w.scale(max(F8(0), F8(1) - ld(self.ang_damp, at) * hv))

        # 1-step Implicit Newton-Raphson Gyroscopic Torque in body principal frame
        var ibx = ld(self.ib_x, at)
        var iby = ld(self.ib_y, at)
        var ibz = ld(self.ib_z, at)
        var has_ib = ibx.gt(F8(0)) & iby.gt(F8(0)) & ibz.gt(F8(0))

        # Also infer principal moments if caller set diagonal anisotropic ixx, iyy, izz directly
        var w_ixx = ld(self.ixx, at)
        var w_iyy = ld(self.iyy, at)
        var w_izz = ld(self.izz, at)
        var w_diag_only = (
            ld(self.ixy, at).eq(F8(0))
            & ld(self.ixz, at).eq(F8(0))
            & ld(self.iyz, at).eq(F8(0))
            & w_ixx.gt(F8(1.0e-8))
            & w_iyy.gt(F8(1.0e-8))
            & w_izz.gt(F8(1.0e-8))
            & (w_ixx.ne(w_iyy) | w_iyy.ne(w_izz))
        )
        var use_gyro = (has_ib | w_diag_only) & w.dot(w).gt(F8(1.0e-10))
        if use_gyro.reduce_or():
            var ix = has_ib.select(ibx, F8(1.0) / max(w_ixx, F8(1.0e-8)))
            var iy = has_ib.select(iby, F8(1.0) / max(w_iyy, F8(1.0e-8)))
            var iz = has_ib.select(ibz, F8(1.0) / max(w_izz, F8(1.0e-8)))

            var qx = has_ib.select(ld(self.qx, at), F8(0.0))
            var qy = has_ib.select(ld(self.qy, at), F8(0.0))
            var qz = has_ib.select(ld(self.qz, at), F8(0.0))
            var qw = has_ib.select(ld(self.qw, at), F8(1.0))

            var wb = _quat_rotate_v8(F8(0) - qx, F8(0) - qy, F8(0) - qz, qw, w)
            var fx = hv * (iz - iy) * wb.y * wb.z
            var fy = hv * (ix - iz) * wb.z * wb.x
            var fz = hv * (iy - ix) * wb.x * wb.y

            var j00 = ix
            var j01 = hv * (iz - iy) * wb.z
            var j02 = hv * (iz - iy) * wb.y
            var j10 = hv * (ix - iz) * wb.z
            var j11 = iy
            var j12 = hv * (ix - iz) * wb.x
            var j20 = hv * (iy - ix) * wb.y
            var j21 = hv * (iy - ix) * wb.x
            var j22 = iz

            var c00 = j11 * j22 - j12 * j21
            var c01 = j12 * j20 - j10 * j22
            var c02 = j10 * j21 - j11 * j20
            var det = j00 * c00 + j01 * c01 + j02 * c02
            var valid = use_gyro & det.gt(F8(1.0e-18))
            var inv_det = valid.select(F8(1.0) / max(det, F8(1.0e-18)), F8(0.0))

            var c10 = j02 * j21 - j01 * j22
            var c11 = j00 * j22 - j02 * j20
            var c12 = j01 * j20 - j00 * j21
            var c20 = j01 * j12 - j02 * j11
            var c21 = j02 * j10 - j00 * j12
            var c22 = j00 * j11 - j01 * j10

            var wb_next = V8(
                wb.x - (c00 * fx + c10 * fy + c20 * fz) * inv_det,
                wb.y - (c01 * fx + c11 * fy + c21 * fz) * inv_det,
                wb.z - (c02 * fx + c12 * fy + c22 * fz) * inv_det,
            )
            var w_gyro = _quat_rotate_v8(qx, qy, qz, qw, wb_next)
            w = V8(valid.select(w_gyro.x, w.x), valid.select(w_gyro.y, w.y), valid.select(w_gyro.z, w.z))

        var speed2 = v.dot(v)
        var limit = F8(max_speed)
        var fast = speed2.gt(limit * limit)
        v = v.scale(fast.select(limit / sqrt(max(speed2, F8(1.0e-12))), F8(1)))
        var spin2 = w.dot(w)
        var turn = F8(max_turn / h)
        var spinning = spin2.gt(turn * turn)
        w = w.scale(spinning.select(turn / sqrt(max(spin2, F8(1.0e-12))), F8(1)))
        st(self.vx, at, v.x)
        st(self.vy, at, v.y)
        st(self.vz, at, v.z)
        st(self.wx, at, w.x)
        st(self.wy, at, w.y)
        st(self.wz, at, w.z)

    def integrate_positions(mut self, batch: Int, h: Float32):
        var at = batch * LANES
        var hv = F8(h)
        var vx = ld(self.vx, at)
        var vy = ld(self.vy, at)
        var vz = ld(self.vz, at)
        var wx = ld(self.wx, at)
        var wy = ld(self.wy, at)
        var wz = ld(self.wz, at)
        st(self.dpx, at, ld(self.dpx, at) + vx * hv)
        st(self.dpy, at, ld(self.dpy, at) + vy * hv)
        st(self.dpz, at, ld(self.dpz, at) + vz * hv)

        var sx = wx * hv * F8(0.5)
        var sy = wy * hv * F8(0.5)
        var sz = wz * hv * F8(0.5)

        # Multiplicative delta quaternion since tick start: dq = normalize(dq + 0.5*(w*h)*dq)
        var dx = ld(self.dqx, at)
        var dy = ld(self.dqy, at)
        var dz = ld(self.dqz, at)
        var dw = ld(self.dqw, at)
        var ndx = dx + (sx * dw + sy * dz - sz * dy)
        var ndy = dy + (F8(0) - sx * dz + sy * dw + sz * dx)
        var ndz = dz + (sx * dy - sy * dx + sz * dw)
        var ndw = dw + (F8(0) - sx * dx - sy * dy - sz * dz)
        var dsize = sqrt(ndx * ndx + ndy * ndy + ndz * ndz + ndw * ndw)
        var dtiny = dsize.lt(F8(0.000001))
        var dsafe = dtiny.select(F8(1), dsize)
        st(self.dqx, at, dtiny.select(F8(0), ndx / dsafe))
        st(self.dqy, at, dtiny.select(F8(0), ndy / dsafe))
        st(self.dqz, at, dtiny.select(F8(0), ndz / dsafe))
        st(self.dqw, at, dtiny.select(F8(1), ndw / dsafe))

        # Orientation q: q = normalize(q + 0.5*(w*h)*q)
        var x = ld(self.qx, at)
        var y = ld(self.qy, at)
        var z = ld(self.qz, at)
        var q = ld(self.qw, at)
        var nx = x + (sx * q + sy * z - sz * y)
        var ny = y + (F8(0) - sx * z + sy * q + sz * x)
        var nz = z + (sx * y - sy * x + sz * q)
        var nw = q + (F8(0) - sx * x - sy * y - sz * z)
        var size = sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
        var tiny = size.lt(F8(0.000001))
        var safe = tiny.select(F8(1), size)
        var uqx = tiny.select(F8(0), nx / safe)
        var uqy = tiny.select(F8(0), ny / safe)
        var uqz = tiny.select(F8(0), nz / safe)
        var uqw = tiny.select(F8(1), nw / safe)
        st(self.qx, at, uqx)
        st(self.qy, at, uqy)
        st(self.qz, at, uqz)
        st(self.qw, at, uqw)

        # Keep world inverse inertia in sync on lanes with explicit principal body inertia
        var ibx = ld(self.ib_x, at)
        var iby = ld(self.ib_y, at)
        var ibz = ld(self.ib_z, at)
        var has_ib = ibx.gt(F8(0)) & iby.gt(F8(0)) & ibz.gt(F8(0))
        if has_ib.reduce_or():
            var inv_x = has_ib.select(F8(1.0) / max(ibx, F8(1.0e-8)), F8(0.0))
            var inv_y = has_ib.select(F8(1.0) / max(iby, F8(1.0e-8)), F8(0.0))
            var inv_z = has_ib.select(F8(1.0) / max(ibz, F8(1.0e-8)), F8(0.0))
            var ex = _quat_rotate_v8(uqx, uqy, uqz, uqw, V8(F8(1), F8(0), F8(0)))
            var ey = _quat_rotate_v8(uqx, uqy, uqz, uqw, V8(F8(0), F8(1), F8(0)))
            var ez = _quat_rotate_v8(uqx, uqy, uqz, uqw, V8(F8(0), F8(0), F8(1)))
            st(self.ixx, at, has_ib.select(ex.x * ex.x * inv_x + ey.x * ey.x * inv_y + ez.x * ez.x * inv_z, ld(self.ixx, at)))
            st(self.iyy, at, has_ib.select(ex.y * ex.y * inv_x + ey.y * ey.y * inv_y + ez.y * ez.y * inv_z, ld(self.iyy, at)))
            st(self.izz, at, has_ib.select(ex.z * ex.z * inv_x + ey.z * ey.y * F8(0) + ez.z * ez.z * inv_z + ey.z * ey.z * inv_y, ld(self.izz, at)))
            st(self.ixy, at, has_ib.select(ex.x * ex.y * inv_x + ey.x * ey.y * inv_y + ez.x * ez.y * inv_z, ld(self.ixy, at)))
            st(self.ixz, at, has_ib.select(ex.x * ex.z * inv_x + ey.x * ey.z * inv_y + ez.x * ez.z * inv_z, ld(self.ixz, at)))
            st(self.iyz, at, has_ib.select(ex.y * ex.z * inv_x + ey.y * ey.z * inv_y + ez.y * ez.z * inv_z, ld(self.iyz, at)))
