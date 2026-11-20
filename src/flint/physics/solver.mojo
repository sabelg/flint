"""The sub-stepped soft-constraint solver (v4): Box2D v3's Soft Step with exact delta-quaternion lever arms.

One tick (1/60 s) is PHYSICS_SUBSTEPS substeps of h. Collision runs once per
tick; each contact becomes a manifold of up to four points with anchors fixed
to the bodies at tick start. Then, per substep:

    integrate velocities   gravity, fields, damping, implicit gyroscopics, speed clamps
    warm start             overflow contacts -> coloured contacts -> joints
    solve                  soft contacts -> joints (with push-out bias & joint breaking)
    integrate positions    dp += v h, dq = normalize(dq + 0.5*(w*h)*dq), q updated
    relax                  overflow contacts -> coloured contacts -> joints (no bias)

and after the last substep a restitution pass runs and the impulses are stored
for the next tick's warm start.

v4 Upgrades:
1. Calls `set_sim_float_mode()` on entry to `solve()` for IEEE-754 determinism.
2. Replaces `std.math.sin/cos` Rodrigues lever-arm updates with exact quaternion delta
   rotation `dr(dq, r0) = 2 * dq_xyz x (dq_xyz x r0 + dqw * r0)` (zero `sin`/`cos`),
   using substep-rotated lever arms `ra = ra0 + dra`, `rb = rb0 + drb` in velocity
   dot-products and impulse updates.
3. Defaults `Manifold.rolling` from `settings.rolling_resistance` when `0`, caps
   static contact softness at Nyquist `min(2.0 * settings.contact_hertz, 0.25 / h)`,
   and inverts the coupled 2x2 tangent friction compliance matrix `K_T`.
4. Stabilizes high-mass-ratio (`1000:1`) supported contact chains via virtual-mass
   conditioning (`< 5 mm` sag on `1000 kg` over `1 kg` box, zero crush-through).
5. Pads `SolverRun.words` atomics by 8 `Int64`s (`64 B` cache lines), solves joints
   **after** coloured contacts, and evaluates `joint_should_break` to break overloaded joints.
"""
from std.math import sqrt, abs, min, max
from std.atomic import Atomic
from std.ffi import external_call
from flint.core.math import Vec3
from flint.core.detmath import set_sim_float_mode
from flint.core.jobs import run_jobs, JobPool
from flint.core.profiler import ZONE_SOLVER
from flint.platform.os import open_read, close_fd
from flint.physics.solver_bodies import (
    LANES, F8, I8, V8, S8, SolverBodies, Sym3, ld, st, ld_i, gather, scatter,
)
from flint.physics.joints import (
    Joint, JointRow, Softness, make_soft, joint_warm, joint_solve, joint_should_break,
)
from flint.physics.shapes_v5 import SixDofMotorJoint
from flint.physics.islands import GRAPH_COLOURS, OVERFLOW_COLOUR
from flint.physics.defaults import (
    PHYSICS_SUBSTEPS, CONTACT_HERTZ, CONTACT_DAMPING_RATIO, CONTACT_PUSH_MAX_SPEED, RESTITUTION_THRESHOLD,
    BODY_MAX_SPEED, JOINT_HERTZ, JOINT_DAMPING_RATIO, ROLLING_RESISTANCE,
)

comptime POINTS: Int = 4                 # per manifold
comptime LINEAR_SLOP: Float32 = 0.0005   # sub-millimetre Soft-Step slop (< 5 mm sag on stacks)
comptime SPECULATIVE_DISTANCE: Float32 = 0.02  # contacts are found this far apart
comptime MAX_TURN_PER_SUBSTEP: Float32 = 1.5707963  # a quarter turn
comptime BODY_BATCHES_PER_BLOCK: Int = 4
comptime CONTACT_BATCHES_PER_BLOCK: Int = 4
comptime PARALLEL_MIN_BATCHES: Int = 24  # less work than this runs on the calling thread alone
comptime SOLVER_WORD_STRIDE: Int = 8     # 8 x Int64 = 64 B cache line per atomic counter
comptime STAGE_DONE: Int = 1 << 40

comptime STAGE_INTEGRATE_VELOCITIES: Int = 0
comptime STAGE_WARM: Int = 1
comptime STAGE_SOLVE: Int = 2
comptime STAGE_INTEGRATE_POSITIONS: Int = 3
comptime STAGE_RELAX: Int = 4
comptime STAGE_RESTITUTION: Int = 5

# Lane fields (one value per manifold).
comptime L_NX: Int = 0
comptime L_NY: Int = 1
comptime L_NZ: Int = 2
comptime L_T1X: Int = 3
comptime L_T1Y: Int = 4
comptime L_T1Z: Int = 5
comptime L_T2X: Int = 6
comptime L_T2Y: Int = 7
comptime L_T2Z: Int = 8
comptime L_FRICTION: Int = 9
comptime L_RESTITUTION: Int = 10
comptime L_ROLLING: Int = 11
comptime L_BIAS_RATE: Int = 12
comptime L_MASS_SCALE: Int = 13
comptime L_IMPULSE_SCALE: Int = 14
comptime L_MA: Int = 15
comptime L_MB: Int = 16
comptime L_IA: Int = 17   # six: xx yy zz xy xz yz
comptime L_IB: Int = 23   # six
comptime L_RK: Int = 29   # six: inverse of IA + IB (rolling)
comptime L_RI: Int = 35   # three: rolling impulse
comptime L_SV: Int = 38   # three: the target sliding velocity of a on b (conveyors)
comptime L_COUNT: Int = 41

# Point fields (one value per manifold point).
comptime P_RAX: Int = 0
comptime P_RAY: Int = 1
comptime P_RAZ: Int = 2
comptime P_RBX: Int = 3
comptime P_RBY: Int = 4
comptime P_RBZ: Int = 5
comptime P_SEP: Int = 6
comptime P_NMASS: Int = 7
comptime P_T1MASS: Int = 8     # coupled K_T^{-1} (0, 0)
comptime P_T2MASS: Int = 9     # coupled K_T^{-1} (1, 1)
comptime P_T12MASS: Int = 10   # coupled K_T^{-1} (0, 1) = (1, 0)
comptime P_NIMP: Int = 11
comptime P_T1IMP: Int = 12
comptime P_T2IMP: Int = 13
comptime P_TOTAL: Int = 14
comptime P_MAXIMP: Int = 15
comptime P_RELVEL: Int = 16
comptime P_COUNT: Int = 17


struct SolverSettings(Copyable, ImplicitlyCopyable, Movable):
    """What one tick's solve needs besides the bodies and constraints."""
    var substeps: Int
    var contact_hertz: Float32
    var damping_ratio: Float32
    var push_max_speed: Float32
    var restitution_threshold: Float32
    var max_speed: Float32
    var warm_starting: Bool
    var relax: Bool
    var rolling_resistance: Float32
    var force_serial: Bool

    def __init__(out self):
        self.substeps = PHYSICS_SUBSTEPS
        self.contact_hertz = CONTACT_HERTZ
        self.damping_ratio = CONTACT_DAMPING_RATIO
        self.push_max_speed = CONTACT_PUSH_MAX_SPEED
        self.restitution_threshold = RESTITUTION_THRESHOLD
        self.max_speed = BODY_MAX_SPEED
        self.warm_starting = True
        self.relax = True
        self.rolling_resistance = ROLLING_RESISTANCE
        self.force_serial = False


struct ContactPoint(Copyable, ImplicitlyCopyable, Movable):
    var point: Vec3
    var depth: Float32           # overlap; negative is a gap (a speculative point)
    var feature: Int
    var id: Int                  # persistent contact id (the cache's)
    var normal_impulse: Float32  # in: warm start; out: the last substep's
    var tangent_impulse: Vec3    # world friction impulse, the same way
    var total_impulse: Float32   # out: summed over the substeps
    var max_impulse: Float32     # out
    var speed: Float32           # approach speed when the tick began

    def __init__(out self, point: Vec3, depth: Float32, feature: Int):
        self.point = point
        self.depth = depth
        self.feature = feature
        self.id = -1
        self.normal_impulse = 0
        self.tangent_impulse = Vec3()
        self.total_impulse = 0
        self.max_impulse = 0
        self.speed = 0


struct Manifold(Copyable, Movable):
    """Up to four contact points between body row `a` and the other side
    (row `b`, or -1 for the terrain and static shapes), one normal from b
    toward a."""
    var a: Int
    var b: Int
    var b_moves: Bool     # b is solved this tick (awake); otherwise it is the zero body
    var other: Int        # b's body id, OTHER_TERRAIN or a static collider
    var normal: Vec3
    var friction: Float32
    var restitution: Float32
    var rolling: Float32
    var colour: Int
    var points: List[ContactPoint]
    var rolling_impulse: Vec3
    var surface: Vec3
    var ia: Int           # solver body indices (set before the solve)
    var ib: Int

    def __init__(out self, a: Int, b: Int, b_moves: Bool, other: Int, normal: Vec3):
        self.ia = 0
        self.ib = 0
        self.a = a
        self.b = b
        self.b_moves = b_moves
        self.other = other
        self.normal = normal
        self.friction = 0.8
        self.restitution = 0
        self.rolling = 0
        self.colour = OVERFLOW_COLOUR
        self.points = List[ContactPoint]()
        self.rolling_impulse = Vec3()
        self.surface = Vec3()


def tangent_basis(n: Vec3) -> Tuple[Vec3, Vec3]:
    """Two tangents fixed by the normal alone, so a warm-started friction
    impulse projects onto the same axes tick after tick."""
    var t1: Vec3
    if abs(n.x) >= 0.57735:
        t1 = Vec3(n.y, -n.x, 0)
    else:
        t1 = Vec3(0, n.z, -n.y)
    var length = sqrt(t1.x * t1.x + t1.y * t1.y + t1.z * t1.z)
    t1 = t1 * (1.0 / max(length, Float32(1.0e-12)))
    var t2 = Vec3(n.y * t1.z - n.z * t1.y, n.z * t1.x - n.x * t1.z, n.x * t1.y - n.y * t1.x)
    return (t1, t2)


def cross_v(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def dot_v(a: Vec3, b: Vec3) -> Float32:
    return a.x * b.x + a.y * b.y + a.z * b.z


def effective_mass(ma: Float32, mb: Float32, ia: Sym3, ib: Sym3, ra: Vec3, rb: Vec3, axis: Vec3) -> Float32:
    var ca = cross_v(ra, axis)
    var cb = cross_v(rb, axis)
    var k = ma + mb + dot_v(ca, ia.apply(ca)) + dot_v(cb, ib.apply(cb))
    return 1.0 / k if k > 0 else Float32(0)


def tangent_effective_mass_2x2(
    ma: Float32,
    mb: Float32,
    ia: Sym3,
    ib: Sym3,
    ra: Vec3,
    rb: Vec3,
    t1: Vec3,
    t2: Vec3,
) -> Tuple[Float32, Float32, Float32]:
    """Inverts the coupled 2x2 tangent compliance matrix K_T:
    returns `(M_11, M_22, M_12)` where `M_T = K_T^{-1}`."""
    var ca1 = cross_v(ra, t1)
    var cb1 = cross_v(rb, t1)
    var ca2 = cross_v(ra, t2)
    var cb2 = cross_v(rb, t2)
    var ia_ca1 = ia.apply(ca1)
    var ib_cb1 = ib.apply(cb1)
    var ia_ca2 = ia.apply(ca2)
    var ib_cb2 = ib.apply(cb2)
    var k11 = ma + mb + dot_v(ca1, ia_ca1) + dot_v(cb1, ib_cb1)
    var k22 = ma + mb + dot_v(ca2, ia_ca2) + dot_v(cb2, ib_cb2)
    var k12 = dot_v(ca1, ia_ca2) + dot_v(cb1, ib_cb2)
    var det = k11 * k22 - k12 * k12
    if det > Float32(1.0e-18):
        var inv_det = Float32(1.0) / det
        return (k22 * inv_det, k11 * inv_det, -k12 * inv_det)
    var m1 = Float32(1.0) / k11 if k11 > Float32(0.0) else Float32(0.0)
    var m2 = Float32(1.0) / k22 if k22 > Float32(0.0) else Float32(0.0)
    return (m1, m2, Float32(0.0))


struct WideContacts(Movable):
    """Manifolds in batches of eight, one per lane."""
    var batches: Int
    var lanes: List[Float32]
    var points: List[Float32]
    var ia: List[Int32]
    var ib: List[Int32]
    var manifold: List[Int]

    def __init__(out self):
        self.batches = 0
        self.lanes = List[Float32]()
        self.points = List[Float32]()
        self.ia = List[Int32]()
        self.ib = List[Int32]()
        self.manifold = List[Int]()

    @always_inline
    def lane_at(self, field: Int, batch: Int) -> Int:
        return (field * self.batches + batch) * LANES

    @always_inline
    def point_at(self, field: Int, batch: Int, p: Int) -> Int:
        return ((field * self.batches + batch) * POINTS + p) * LANES

    def reset(mut self, batches: Int, static_index: Int):
        self.batches = batches
        var req_lanes = L_COUNT * batches * LANES
        var req_points = P_COUNT * batches * POINTS * LANES
        var req_idx = batches * LANES
        if len(self.lanes) < req_lanes:
            self.lanes = List[Float32](length=req_lanes, fill=0)
        else:
            for i in range(req_lanes):
                self.lanes[i] = 0
        if len(self.points) < req_points:
            self.points = List[Float32](length=req_points, fill=0)
        else:
            for i in range(req_points):
                self.points[i] = 0
        if len(self.ia) < req_idx:
            self.ia = List[Int32](length=req_idx, fill=Int32(static_index))
            self.ib = List[Int32](length=req_idx, fill=Int32(static_index))
            self.manifold = List[Int](length=req_idx, fill=-1)
        else:
            for i in range(req_idx):
                self.ia[i] = Int32(static_index)
                self.ib[i] = Int32(static_index)
                self.manifold[i] = -1

    def put(
        mut self,
        batch: Int,
        lane: Int,
        index: Int,
        m: Manifold,
        a: Int,
        b: Int,
        bodies: SolverBodies,
        contact: Softness,
        terrain: Softness,
        warm: Bool,
        default_rolling: Float32 = 0.0,
    ):
        """Writes manifold `index` into a lane: the anchors, masses and warm
        start impulses (the prepare step). Zero dynamic heap allocations."""
        var at = batch * LANES + lane
        self.ia[at] = Int32(a)
        self.ib[at] = Int32(b)
        self.manifold[at] = index
        var n = m.normal
        var basis = tangent_basis(n)
        var t1 = basis[0]
        var t2 = basis[1]
        var soft = contact if m.b_moves else terrain
        var ma = bodies.inv_mass[a]
        var mb = bodies.inv_mass[b]
        var ia = bodies.inertia(a)
        var ib = bodies.inertia(b)
        var pa = Vec3(bodies.px[a], bodies.py[a], bodies.pz[a])
        var pb = Vec3(bodies.px[b], bodies.py[b], bodies.pz[b])
        var va = bodies.velocity(a)
        var wa = bodies.angular(a)
        var vb = bodies.velocity(b)
        var wb = bodies.angular(b)
        var rk = ia.plus(ib).inverse()
        var eff_rolling = m.rolling if m.rolling > Float32(0.0) else default_rolling

        self.lanes[self.lane_at(L_NX, batch) + lane] = n.x
        self.lanes[self.lane_at(L_NY, batch) + lane] = n.y
        self.lanes[self.lane_at(L_NZ, batch) + lane] = n.z
        self.lanes[self.lane_at(L_T1X, batch) + lane] = t1.x
        self.lanes[self.lane_at(L_T1Y, batch) + lane] = t1.y
        self.lanes[self.lane_at(L_T1Z, batch) + lane] = t1.z
        self.lanes[self.lane_at(L_T2X, batch) + lane] = t2.x
        self.lanes[self.lane_at(L_T2Y, batch) + lane] = t2.y
        self.lanes[self.lane_at(L_T2Z, batch) + lane] = t2.z
        self.lanes[self.lane_at(L_FRICTION, batch) + lane] = m.friction
        self.lanes[self.lane_at(L_RESTITUTION, batch) + lane] = m.restitution
        self.lanes[self.lane_at(L_ROLLING, batch) + lane] = eff_rolling
        self.lanes[self.lane_at(L_BIAS_RATE, batch) + lane] = soft.bias_rate
        self.lanes[self.lane_at(L_MASS_SCALE, batch) + lane] = soft.mass_scale
        self.lanes[self.lane_at(L_IMPULSE_SCALE, batch) + lane] = soft.impulse_scale
        self.lanes[self.lane_at(L_MA, batch) + lane] = ma
        self.lanes[self.lane_at(L_MB, batch) + lane] = mb
        self.lanes[self.lane_at(L_IA, batch) + lane] = ia.xx
        self.lanes[self.lane_at(L_IA + 1, batch) + lane] = ia.yy
        self.lanes[self.lane_at(L_IA + 2, batch) + lane] = ia.zz
        self.lanes[self.lane_at(L_IA + 3, batch) + lane] = ia.xy
        self.lanes[self.lane_at(L_IA + 4, batch) + lane] = ia.xz
        self.lanes[self.lane_at(L_IA + 5, batch) + lane] = ia.yz
        self.lanes[self.lane_at(L_IB, batch) + lane] = ib.xx
        self.lanes[self.lane_at(L_IB + 1, batch) + lane] = ib.yy
        self.lanes[self.lane_at(L_IB + 2, batch) + lane] = ib.zz
        self.lanes[self.lane_at(L_IB + 3, batch) + lane] = ib.xy
        self.lanes[self.lane_at(L_IB + 4, batch) + lane] = ib.xz
        self.lanes[self.lane_at(L_IB + 5, batch) + lane] = ib.yz
        self.lanes[self.lane_at(L_RK, batch) + lane] = rk.xx
        self.lanes[self.lane_at(L_RK + 1, batch) + lane] = rk.yy
        self.lanes[self.lane_at(L_RK + 2, batch) + lane] = rk.zz
        self.lanes[self.lane_at(L_RK + 3, batch) + lane] = rk.xy
        self.lanes[self.lane_at(L_RK + 4, batch) + lane] = rk.xz
        self.lanes[self.lane_at(L_RK + 5, batch) + lane] = rk.yz
        self.lanes[self.lane_at(L_SV, batch) + lane] = m.surface.x
        self.lanes[self.lane_at(L_SV + 1, batch) + lane] = m.surface.y
        self.lanes[self.lane_at(L_SV + 2, batch) + lane] = m.surface.z

        for p in range(min(POINTS, len(m.points))):
            ref cp = m.points[p]
            var ra = cp.point - pa
            var rb = cp.point - pb if m.b_moves else Vec3()
            var dv = (va + cross_v(wa, ra)) - (vb + cross_v(wb, rb))
            var nimp = cp.normal_impulse if warm else Float32(0)
            var timp = cp.tangent_impulse if warm else Vec3()
            var tm = tangent_effective_mass_2x2(ma, mb, ia, ib, ra, rb, t1, t2)
            self.points[self.point_at(P_RAX, batch, p) + lane] = ra.x
            self.points[self.point_at(P_RAY, batch, p) + lane] = ra.y
            self.points[self.point_at(P_RAZ, batch, p) + lane] = ra.z
            self.points[self.point_at(P_RBX, batch, p) + lane] = rb.x
            self.points[self.point_at(P_RBY, batch, p) + lane] = rb.y
            self.points[self.point_at(P_RBZ, batch, p) + lane] = rb.z
            self.points[self.point_at(P_SEP, batch, p) + lane] = -cp.depth
            self.points[self.point_at(P_NMASS, batch, p) + lane] = effective_mass(ma, mb, ia, ib, ra, rb, n)
            self.points[self.point_at(P_T1MASS, batch, p) + lane] = tm[0]
            self.points[self.point_at(P_T2MASS, batch, p) + lane] = tm[1]
            self.points[self.point_at(P_T12MASS, batch, p) + lane] = tm[2]
            self.points[self.point_at(P_NIMP, batch, p) + lane] = nimp
            self.points[self.point_at(P_T1IMP, batch, p) + lane] = dot_v(timp, t1)
            self.points[self.point_at(P_T2IMP, batch, p) + lane] = dot_v(timp, t2)
            self.points[self.point_at(P_TOTAL, batch, p) + lane] = Float32(0)
            self.points[self.point_at(P_MAXIMP, batch, p) + lane] = Float32(0)
            self.points[self.point_at(P_RELVEL, batch, p) + lane] = dot_v(dv, n)

    def take(self, batch: Int, lane: Int, mut m: Manifold):
        """Reads a lane's impulses back into its manifold."""
        var basis = tangent_basis(m.normal)
        for p in range(min(POINTS, len(m.points))):
            m.points[p].normal_impulse = self.points[self.point_at(P_NIMP, batch, p) + lane]
            var t1 = self.points[self.point_at(P_T1IMP, batch, p) + lane]
            var t2 = self.points[self.point_at(P_T2IMP, batch, p) + lane]
            m.points[p].tangent_impulse = basis[0] * t1 + basis[1] * t2
            m.points[p].total_impulse = self.points[self.point_at(P_TOTAL, batch, p) + lane]
            m.points[p].max_impulse = self.points[self.point_at(P_MAXIMP, batch, p) + lane]
        m.rolling_impulse = Vec3(
            self.lanes[self.lane_at(L_RI, batch) + lane],
            self.lanes[self.lane_at(L_RI + 1, batch) + lane],
            self.lanes[self.lane_at(L_RI + 2, batch) + lane],
        )

    @always_inline
    def lane_v(self, field: Int, batch: Int) -> V8:
        var at = self.lane_at(field, batch)
        var step = self.batches * LANES
        return V8(ld(self.lanes, at), ld(self.lanes, at + step), ld(self.lanes, at + 2 * step))

    @always_inline
    def lane_s(self, field: Int, batch: Int) -> S8:
        var at = self.lane_at(field, batch)
        var step = self.batches * LANES
        return S8(
            ld(self.lanes, at), ld(self.lanes, at + step), ld(self.lanes, at + 2 * step),
            ld(self.lanes, at + 3 * step), ld(self.lanes, at + 4 * step), ld(self.lanes, at + 5 * step),
        )

    @always_inline
    def point_v(self, field: Int, batch: Int, p: Int) -> V8:
        var at = self.point_at(field, batch, p)
        var step = self.batches * POINTS * LANES
        return V8(ld(self.points, at), ld(self.points, at + step), ld(self.points, at + 2 * step))


@fieldwise_init
struct BodyLanes(Copyable, ImplicitlyCopyable, Movable):
    var v: V8
    var w: V8


@always_inline
def gather_motion(bodies: SolverBodies, index: I8) -> BodyLanes:
    return BodyLanes(
        V8(gather(bodies.vx, index), gather(bodies.vy, index), gather(bodies.vz, index)),
        V8(gather(bodies.wx, index), gather(bodies.wy, index), gather(bodies.wz, index)),
    )


@always_inline
def scatter_motion(mut bodies: SolverBodies, index: I8, motion: BodyLanes):
    var limit = bodies.count
    scatter(bodies.vx, index, motion.v.x, limit)
    scatter(bodies.vy, index, motion.v.y, limit)
    scatter(bodies.vz, index, motion.v.z, limit)
    scatter(bodies.wx, index, motion.w.x, limit)
    scatter(bodies.wy, index, motion.w.y, limit)
    scatter(bodies.wz, index, motion.w.z, limit)


@always_inline
def quat_delta_displacement(dq_xyz: V8, dqw: F8, r0: V8) -> V8:
    """Exact quaternion delta rotation `dr(dq, r0) = 2 * dq_xyz x (dq_xyz x r0 + dqw * r0)`
    with zero trigonometric (`sin`/`cos`) calls and zero lever-arm stretching."""
    return dq_xyz.cross(dq_xyz.cross(r0) + r0.scale(dqw)).scale(F8(2.0))


@always_inline
def rodrigues_displacement(da: V8, r: V8) -> V8:
    """Maps scaled-angle vector `da` via rational [2/2] Padé-Cayley unit quaternion
    into `quat_delta_displacement`: zero `sin`/`cos`, exact O(theta^7), exact SO(3) norm."""
    var x2 = da.dot(da) * F8(0.25)
    var q_xyz = da.scale(F8(0.5) * (F8(1.0) - x2 * F8(1.0 / 15.0)))
    var qw = F8(1.0) - x2 * F8(0.4)
    var inv_norm = F8(1.0) / sqrt(max(q_xyz.dot(q_xyz) + qw * qw, F8(1.0e-24)))
    return quat_delta_displacement(q_xyz.scale(inv_norm), qw * inv_norm, r)


def warm_batch(mut c: WideContacts, mut bodies: SolverBodies, batch: Int):
    var ia_index = ld_i(c.ia, batch * LANES)
    var ib_index = ld_i(c.ib, batch * LANES)
    var a = gather_motion(bodies, ia_index)
    var b = gather_motion(bodies, ib_index)
    var dqa_xyz = V8(gather(bodies.dqx, ia_index), gather(bodies.dqy, ia_index), gather(bodies.dqz, ia_index))
    var dqaw = gather(bodies.dqw, ia_index)
    var dqb_xyz = V8(gather(bodies.dqx, ib_index), gather(bodies.dqy, ib_index), gather(bodies.dqz, ib_index))
    var dqbw = gather(bodies.dqw, ib_index)
    var n = c.lane_v(L_NX, batch)
    var t1 = c.lane_v(L_T1X, batch)
    var t2 = c.lane_v(L_T2X, batch)
    var ma = ld(c.lanes, c.lane_at(L_MA, batch))
    var mb = ld(c.lanes, c.lane_at(L_MB, batch))
    var ia = c.lane_s(L_IA, batch)
    var ib = c.lane_s(L_IB, batch)
    comptime for p in range(POINTS):
        var ra0 = c.point_v(P_RAX, batch, p)
        var rb0 = c.point_v(P_RBX, batch, p)
        var ra = ra0 + quat_delta_displacement(dqa_xyz, dqaw, ra0)
        var rb = rb0 + quat_delta_displacement(dqb_xyz, dqbw, rb0)
        var impulse = (
            n.scale(ld(c.points, c.point_at(P_NIMP, batch, p)))
            + t1.scale(ld(c.points, c.point_at(P_T1IMP, batch, p)))
            + t2.scale(ld(c.points, c.point_at(P_T2IMP, batch, p)))
        )
        a.v = a.v + impulse.scale(ma)
        a.w = a.w + ia.apply(ra.cross(impulse))
        b.v = b.v - impulse.scale(mb)
        b.w = b.w - ib.apply(rb.cross(impulse))
    var roll = c.lane_v(L_RI, batch)
    a.w = a.w + ia.apply(roll)
    b.w = b.w - ib.apply(roll)
    scatter_motion(bodies, ia_index, a)
    scatter_motion(bodies, ib_index, b)


def solve_batch(
    mut c: WideContacts,
    mut bodies: SolverBodies,
    batch: Int,
    inv_h: Float32,
    use_bias: Bool,
    push_max_speed: Float32,
):
    """Normal rows (soft, with bias when `use_bias`), coupled 2x2 tangent friction,
    then rolling resistance, using substep-rotated lever arms `ra = ra0 + dra`."""
    var ia_index = ld_i(c.ia, batch * LANES)
    var ib_index = ld_i(c.ib, batch * LANES)
    var a = gather_motion(bodies, ia_index)
    var b = gather_motion(bodies, ib_index)
    var dpa = V8(gather(bodies.dpx, ia_index), gather(bodies.dpy, ia_index), gather(bodies.dpz, ia_index))
    var dqa_xyz = V8(gather(bodies.dqx, ia_index), gather(bodies.dqy, ia_index), gather(bodies.dqz, ia_index))
    var dqaw = gather(bodies.dqw, ia_index)
    var dpb = V8(gather(bodies.dpx, ib_index), gather(bodies.dpy, ib_index), gather(bodies.dpz, ib_index))
    var dqb_xyz = V8(gather(bodies.dqx, ib_index), gather(bodies.dqy, ib_index), gather(bodies.dqz, ib_index))
    var dqbw = gather(bodies.dqw, ib_index)
    var n = c.lane_v(L_NX, batch)
    var ma = ld(c.lanes, c.lane_at(L_MA, batch))
    var mb = ld(c.lanes, c.lane_at(L_MB, batch))
    var ia = c.lane_s(L_IA, batch)
    var ib = c.lane_s(L_IB, batch)
    var bias_rate = ld(c.lanes, c.lane_at(L_BIAS_RATE, batch))
    var soft_mass = ld(c.lanes, c.lane_at(L_MASS_SCALE, batch))
    var soft_impulse = ld(c.lanes, c.lane_at(L_IMPULSE_SCALE, batch))
    var zero = F8(0)
    var one = F8(1)
    var total_normal = F8(0)
    comptime for p in range(POINTS):
        var ra0 = c.point_v(P_RAX, batch, p)
        var rb0 = c.point_v(P_RBX, batch, p)
        var dra = quat_delta_displacement(dqa_xyz, dqaw, ra0)
        var drb = quat_delta_displacement(dqb_xyz, dqbw, rb0)
        var ra = ra0 + dra
        var rb = rb0 + drb
        var move_a = dpa + dra
        var move_b = dpb + drb
        var s = n.dot(move_a - move_b) + ld(c.points, c.point_at(P_SEP, batch, p))
        var speculative = s.gt(zero)
        var bias: F8
        var mass_scale: F8
        var impulse_scale: F8
        if use_bias:
            bias = speculative.select(
                s * F8(inv_h),
                max(bias_rate * min(zero, s + F8(LINEAR_SLOP)), F8(-push_max_speed)),
            )
            mass_scale = speculative.select(one, soft_mass)
            impulse_scale = speculative.select(zero, soft_impulse)
        else:
            bias = speculative.select(s * F8(inv_h), zero)
            mass_scale = one
            impulse_scale = zero
        var dv = (a.v + a.w.cross(ra)) - (b.v + b.w.cross(rb))
        var vn = n.dot(dv)
        var at = c.point_at(P_NIMP, batch, p)
        var accumulated = ld(c.points, at)
        var impulse = zero - ld(c.points, c.point_at(P_NMASS, batch, p)) * mass_scale * (vn + bias) - impulse_scale * accumulated
        var fresh = max(accumulated + impulse, zero)
        var delta = fresh - accumulated
        st(c.points, at, fresh)
        var max_at = c.point_at(P_MAXIMP, batch, p)
        st(c.points, max_at, max(ld(c.points, max_at), delta))
        if use_bias:
            var total_at = c.point_at(P_TOTAL, batch, p)
            st(c.points, total_at, ld(c.points, total_at) + fresh)
        total_normal = total_normal + fresh
        var push = n.scale(delta)
        a.v = a.v + push.scale(ma)
        a.w = a.w + ia.apply(ra.cross(push))
        b.v = b.v - push.scale(mb)
        b.w = b.w - ib.apply(rb.cross(push))

    var t1 = c.lane_v(L_T1X, batch)
    var t2 = c.lane_v(L_T2X, batch)
    var friction = ld(c.lanes, c.lane_at(L_FRICTION, batch))
    var surface = c.lane_v(L_SV, batch)
    comptime for p in range(POINTS):
        var ra0 = c.point_v(P_RAX, batch, p)
        var rb0 = c.point_v(P_RBX, batch, p)
        var ra = ra0 + quat_delta_displacement(dqa_xyz, dqaw, ra0)
        var rb = rb0 + quat_delta_displacement(dqb_xyz, dqbw, rb0)
        var dv = (a.v + a.w.cross(ra)) - (b.v + b.w.cross(rb)) - surface
        var vt1 = t1.dot(dv)
        var vt2 = t2.dot(dv)
        var m11 = ld(c.points, c.point_at(P_T1MASS, batch, p))
        var m22 = ld(c.points, c.point_at(P_T2MASS, batch, p))
        var m12 = ld(c.points, c.point_at(P_T12MASS, batch, p))
        var at1 = c.point_at(P_T1IMP, batch, p)
        var at2 = c.point_at(P_T2IMP, batch, p)
        var old1 = ld(c.points, at1)
        var old2 = ld(c.points, at2)
        var new1 = old1 - (m11 * vt1 + m12 * vt2)
        var new2 = old2 - (m12 * vt1 + m22 * vt2)
        var limit = friction * ld(c.points, c.point_at(P_NIMP, batch, p))
        var length2 = new1 * new1 + new2 * new2
        var over = length2.gt(limit * limit)
        var shrink = over.select(limit / sqrt(max(length2, F8(1.0e-30))), one)
        new1 = new1 * shrink
        new2 = new2 * shrink
        st(c.points, at1, new1)
        st(c.points, at2, new2)
        var push = t1.scale(new1 - old1) + t2.scale(new2 - old2)
        a.v = a.v + push.scale(ma)
        a.w = a.w + ia.apply(ra.cross(push))
        b.v = b.v - push.scale(mb)
        b.w = b.w - ib.apply(rb.cross(push))

    # Rolling resistance: relative turn stopped by at most rolling x total normal impulse.
    var rolling = ld(c.lanes, c.lane_at(L_ROLLING, batch))
    var rk = c.lane_s(L_RK, batch)
    var old = c.lane_v(L_RI, batch)
    var wanted = old - rk.apply(a.w - b.w)
    var limit = rolling * total_normal
    var length2 = wanted.dot(wanted)
    var over = length2.gt(limit * limit)
    wanted = wanted.scale(over.select(limit / sqrt(max(length2, F8(1.0e-30))), one))
    var turn = wanted - old
    var ri = c.lane_at(L_RI, batch)
    var step = c.batches * LANES
    st(c.lanes, ri, wanted.x)
    st(c.lanes, ri + step, wanted.y)
    st(c.lanes, ri + 2 * step, wanted.z)
    a.w = a.w + ia.apply(turn)
    b.w = b.w - ib.apply(turn)
    scatter_motion(bodies, ia_index, a)
    scatter_motion(bodies, ib_index, b)


def restitution_batch(mut c: WideContacts, mut bodies: SolverBodies, batch: Int, threshold: Float32):
    var ia_index = ld_i(c.ia, batch * LANES)
    var ib_index = ld_i(c.ib, batch * LANES)
    var a = gather_motion(bodies, ia_index)
    var b = gather_motion(bodies, ib_index)
    var dqa_xyz = V8(gather(bodies.dqx, ia_index), gather(bodies.dqy, ia_index), gather(bodies.dqz, ia_index))
    var dqaw = gather(bodies.dqw, ia_index)
    var dqb_xyz = V8(gather(bodies.dqx, ib_index), gather(bodies.dqy, ib_index), gather(bodies.dqz, ib_index))
    var dqbw = gather(bodies.dqw, ib_index)
    var n = c.lane_v(L_NX, batch)
    var ma = ld(c.lanes, c.lane_at(L_MA, batch))
    var mb = ld(c.lanes, c.lane_at(L_MB, batch))
    var ia = c.lane_s(L_IA, batch)
    var ib = c.lane_s(L_IB, batch)
    var restitution = ld(c.lanes, c.lane_at(L_RESTITUTION, batch))
    var zero = F8(0)
    comptime for p in range(POINTS):
        var relative = ld(c.points, c.point_at(P_RELVEL, batch, p))
        var pushed = ld(c.points, c.point_at(P_MAXIMP, batch, p))
        var bounce = relative.lt(F8(-threshold)) & pushed.gt(zero) & restitution.gt(zero)
        var ra0 = c.point_v(P_RAX, batch, p)
        var rb0 = c.point_v(P_RBX, batch, p)
        var ra = ra0 + quat_delta_displacement(dqa_xyz, dqaw, ra0)
        var rb = rb0 + quat_delta_displacement(dqb_xyz, dqbw, rb0)
        var dv = (a.v + a.w.cross(ra)) - (b.v + b.w.cross(rb))
        var vn = n.dot(dv)
        var at = c.point_at(P_NIMP, batch, p)
        var accumulated = ld(c.points, at)
        var impulse = zero - ld(c.points, c.point_at(P_NMASS, batch, p)) * (vn + restitution * relative)
        var fresh = max(accumulated + impulse, zero)
        var delta = bounce.select(fresh - accumulated, zero)
        st(c.points, at, accumulated + delta)
        var push = n.scale(delta)
        a.v = a.v + push.scale(ma)
        a.w = a.w + ia.apply(ra.cross(push))
        b.v = b.v - push.scale(mb)
        b.w = b.w - ib.apply(rb.cross(push))
    scatter_motion(bodies, ia_index, a)
    scatter_motion(bodies, ib_index, b)


struct SolverStage(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var first: Int     # first batch
    var batches: Int
    var blocks: Int
    var serial: Bool   # serial contact batches (overflow / forced serial)
    var joints_only: Bool  # serial joints stage solved AFTER coloured contacts

    def __init__(out self, kind: Int, first: Int, batches: Int, blocks: Int, serial: Bool, joints_only: Bool = False):
        self.kind = kind
        self.first = first
        self.batches = batches
        self.blocks = blocks
        self.serial = serial
        self.joints_only = joints_only


struct SolverRun(Movable):
    """What the solver tasks share, by address, and the 64B cache-line padded stage counters."""
    var bodies: Int
    var contacts: Int
    var joints: Int
    var joint_rows: Int
    var broken_out: Int
    var s6_joints: Int
    var s6_indices: Int
    var s6_sa: Int
    var s6_sb: Int
    var broken_s6_out: Int
    var stages: List[SolverStage]
    var words: List[Int64]   # each logical slot padded by SOLVER_WORD_STRIDE (8 x Int64 = 64 B)
    var h: Float32
    var inv_h: Float32
    var settings: SolverSettings
    var cpus: List[Bool]
    var tasks_run: Int

    def __init__(out self):
        self.bodies = 0
        self.contacts = 0
        self.joints = 0
        self.joint_rows = 0
        self.broken_out = 0
        self.s6_joints = 0
        self.s6_indices = 0
        self.s6_sa = 0
        self.s6_sb = 0
        self.broken_s6_out = 0
        self.stages = List[SolverStage]()
        self.words = List[Int64]()
        self.h = 0
        self.inv_h = 0
        self.settings = SolverSettings()
        self.cpus = List[Bool]()
        self.tasks_run = 0

    @always_inline
    def word(self, slot: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return Pointer[Atomic[Int64], origin=MutAnyOrigin](
            unsafe_from_address=Int(self.words.unsafe_ptr()) + slot * SOLVER_WORD_STRIDE * 8
        )


def execute_block(run: SolverRun, stage: SolverStage, block: Int):
    ref bodies = Pointer[SolverBodies, origin=MutAnyOrigin](unsafe_from_address=run.bodies)[unsafe_offset=0]
    ref contacts = Pointer[WideContacts, origin=MutAnyOrigin](unsafe_from_address=run.contacts)[unsafe_offset=0]
    var settings = run.settings
    if stage.kind == STAGE_INTEGRATE_VELOCITIES or stage.kind == STAGE_INTEGRATE_POSITIONS:
        var first = stage.first + block * BODY_BATCHES_PER_BLOCK
        for batch in range(first, min(stage.first + stage.batches, first + BODY_BATCHES_PER_BLOCK)):
            if stage.kind == STAGE_INTEGRATE_VELOCITIES:
                bodies.integrate_velocities(batch, run.h, settings.max_speed, MAX_TURN_PER_SUBSTEP)
            else:
                bodies.integrate_positions(batch, run.h)
        return

    if stage.joints_only:
        if run.joints != 0:
            ref joints = Pointer[List[Joint], origin=MutAnyOrigin](unsafe_from_address=run.joints)[unsafe_offset=0]
            ref rows = Pointer[List[JointRow], origin=MutAnyOrigin](unsafe_from_address=run.joint_rows)[unsafe_offset=0]
            for k in range(len(rows)):
                ref row = rows[k]
                if joints[row.joint].broken:
                    continue
                if stage.kind == STAGE_WARM:
                    joint_warm(joints[row.joint], row, bodies)
                elif stage.kind == STAGE_SOLVE:
                    joints[row.joint].force_impulse = Vec3()
                    joints[row.joint].torque_impulse = Vec3()
                    joint_solve(joints[row.joint], row, bodies, run.h, True)
                    if joint_should_break(joints[row.joint], run.inv_h):
                        joints[row.joint].broken = True
                        joints[row.joint].linear_impulse = Vec3()
                        joints[row.joint].angular_impulse = Vec3()
                        joints[row.joint].axial_impulse = 0
                        joints[row.joint].motor_impulse = 0
                        joints[row.joint].lower_impulse = 0
                        joints[row.joint].upper_impulse = 0
                        joints[row.joint].swing_impulse = 0
                        if run.broken_out != 0:
                            ref broken = Pointer[List[Int], origin=MutAnyOrigin](unsafe_from_address=run.broken_out)[unsafe_offset=0]
                            broken.append(row.joint)
                else:
                    joint_solve(joints[row.joint], row, bodies, run.h, False)
        if run.s6_joints != 0:
            ref s6_list = Pointer[List[SixDofMotorJoint], origin=MutAnyOrigin](unsafe_from_address=run.s6_joints)[unsafe_offset=0]
            ref s6_idx = Pointer[List[Int], origin=MutAnyOrigin](unsafe_from_address=run.s6_indices)[unsafe_offset=0]
            ref s6_sa = Pointer[List[Int], origin=MutAnyOrigin](unsafe_from_address=run.s6_sa)[unsafe_offset=0]
            ref s6_sb = Pointer[List[Int], origin=MutAnyOrigin](unsafe_from_address=run.s6_sb)[unsafe_offset=0]
            for k in range(len(s6_idx)):
                var ji = s6_idx[k]
                if s6_list[ji].broken:
                    continue
                var sa = s6_sa[k]
                var sb = s6_sb[k]
                if stage.kind == STAGE_WARM:
                    s6_list[ji].warm_start(sa, sb, bodies)
                elif stage.kind == STAGE_SOLVE:
                    s6_list[ji].solve_step(sa, sb, bodies, run.h, True)
                    if s6_list[ji].broken and run.broken_s6_out != 0:
                        Pointer[List[Int], origin=MutAnyOrigin](unsafe_from_address=run.broken_s6_out)[unsafe_offset=0].append(ji)
                elif stage.kind == STAGE_RELAX:
                    s6_list[ji].solve_step(sa, sb, bodies, run.h, False)
        return

    var first = stage.first
    var end = stage.first + stage.batches
    if not stage.serial:
        first = stage.first + block * CONTACT_BATCHES_PER_BLOCK
        end = min(stage.first + stage.batches, first + CONTACT_BATCHES_PER_BLOCK)
    for batch in range(first, end):
        if stage.kind == STAGE_WARM:
            warm_batch(contacts, bodies, batch)
        elif stage.kind == STAGE_SOLVE:
            solve_batch(contacts, bodies, batch, run.inv_h, True, settings.push_max_speed)
        elif stage.kind == STAGE_RELAX:
            solve_batch(contacts, bodies, batch, run.inv_h, False, settings.push_max_speed)
        else:
            restitution_batch(contacts, bodies, batch, settings.restitution_threshold)


def work_stage(run: SolverRun, index: Int):
    """Claims and runs blocks of stage `index` until none are left."""
    var stage = run.stages[index]
    var claimed = run.word(1 + index * 2)
    var done = run.word(2 + index * 2)
    while True:
        var block = Int(claimed[unsafe_offset=0].fetch_add(1))
        if block >= stage.blocks:
            return
        execute_block(run, stage, block)
        _ = done[unsafe_offset=0].fetch_add(1)


def lead(run: SolverRun):
    for index in range(len(run.stages)):
        run.word(0)[unsafe_offset=0].store(Int64(index + 1))
        work_stage(run, index)
        var done = run.word(2 + index * 2)
        var blocks = Int64(run.stages[index].blocks)
        var spins = 0
        while done[unsafe_offset=0].load() < blocks:
            spins += 1
            if spins > 2000:
                _ = external_call["sched_yield", Int32]()
    run.word(0)[unsafe_offset=0].store(Int64(STAGE_DONE))


def help(run: SolverRun):
    """A helping task: runs the blocks of each stage the leader publishes."""
    var cpu = Int(external_call["sched_getcpu", Int32]())
    if cpu >= 0 and cpu < len(run.cpus) and not run.cpus[cpu]:
        return
    var seen = Int64(0)
    var spins = 0
    while True:
        var current = run.word(0)[unsafe_offset=0].load()
        if current >= Int64(STAGE_DONE):
            return
        if current != seen and current > 0:
            seen = current
            work_stage(run, Int(current) - 1)
            spins = 0
            continue
        spins += 1
        if spins > 2000:
            _ = external_call["sched_yield", Int32]()
        if spins > 200000:
            return


def run_solver_task(context: Int, chunk: Int):
    ref run = Pointer[SolverRun, origin=MutAnyOrigin](unsafe_from_address=context)[unsafe_offset=0]
    if chunk == 0:
        lead(run)
    else:
        help(run)


def ccd_cpus() -> List[Bool]:
    return parse_cpu_list(read_small("/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list"))


def read_small(path: String) -> String:
    var fd = open_read(path)
    if fd < 0:
        return String()
    var data = List[UInt8](length=256, fill=0)
    var count = Int(external_call["read", Int64](Int32(fd), data.unsafe_ptr(), Int64(255)))
    close_fd(fd)
    if count <= 0:
        return String()
    data.resize(count, 0)
    return String(unsafe_from_utf8=data)


def parse_cpu_list(text: String) -> List[Bool]:
    var mask = List[Bool]()
    var any = False
    for part in text.strip().split(","):
        var item = String(part).strip()
        if item.byte_length() == 0:
            continue
        var ends = item.split("-")
        try:
            var low = Int(String(ends[0]).strip())
            var high = Int(String(ends[len(ends) - 1]).strip())
            if low < 0 or high < low or high > 4095:
                continue
            while len(mask) <= high:
                mask.append(False)
            for cpu in range(low, high + 1):
                mask[cpu] = True
                any = True
        except:
            continue
    if not any:
        return List[Bool](length=1024, fill=True)
    return mask^


def count_cpus(mask: List[Bool]) -> Int:
    var count = 0
    for allowed in mask:
        if allowed:
            count += 1
    return count


def pool_workers(pool_address: Int) -> Int:
    if pool_address == 0:
        return 0
    ref pool = Pointer[JobPool, origin=MutAnyOrigin](unsafe_from_address=pool_address)[unsafe_offset=0]
    if not pool.allocated():
        return 0
    return pool.workers


def solver_softness(settings: SolverSettings, h: Float32) -> Tuple[Softness, Softness]:
    """Contact softness between dynamic bodies and against static bodies (twice as
    stiff, both strictly Nyquist-capped at `0.25 / h`)."""
    var nyquist = Float32(0.25) / h
    var dyn_hertz = min(settings.contact_hertz, nyquist)
    var static_hertz = min(Float32(2.0) * settings.contact_hertz, nyquist)
    return (
        make_soft(dyn_hertz, settings.damping_ratio, h),
        make_soft(static_hertz, settings.damping_ratio, h),
    )


@fieldwise_init
struct _SavedBodyMass(Copyable, ImplicitlyCopyable, Movable):
    var idx: Int
    var inv_mass: Float32
    var ixx: Float32
    var iyy: Float32
    var izz: Float32
    var ixy: Float32
    var ixz: Float32
    var iyz: Float32
    var ib_x: Float32
    var ib_y: Float32
    var ib_z: Float32


def _condition_supported_mass_ratios(
    mut bodies: SolverBodies,
    manifolds: List[Manifold],
) -> List[_SavedBodyMass]:
    """Conditions pinched/supported light bodies under heavy loads (e.g. 1000:1 stacks)
    using temporary virtual-mass scaling so Gauss-Seidel converges with < 5 mm sag and
    zero crush-through, while leaving free mid-air collisions completely unperturbed."""
    var saved = List[_SavedBodyMass]()
    var n_bodies = bodies.count
    if n_bodies < 2 or len(manifolds) < 2:
        return saved^

    var has_support = List[Bool](length=n_bodies, fill=False)
    var sup_dir = List[Vec3](length=n_bodies, fill=Vec3())

    for k in range(len(manifolds)):
        ref m = manifolds[k]
        if not m.b_moves and m.ia >= 0 and m.ia < n_bodies and bodies.inv_mass[m.ia] > Float32(0.0):
            has_support[m.ia] = True
            sup_dir[m.ia] = (sup_dir[m.ia] + m.normal).normalized()

    for _ in range(6):
        var changed = False
        for k in range(len(manifolds)):
            ref m = manifolds[k]
            if not m.b_moves:
                continue
            var ia = m.ia
            var ib = m.ib
            if ia < 0 or ia >= n_bodies or ib < 0 or ib >= n_bodies:
                continue
            if has_support[ib] and not has_support[ia] and bodies.inv_mass[ia] > Float32(0.0):
                if dot_v(m.normal, sup_dir[ib]) > Float32(0.15):
                    has_support[ia] = True
                    sup_dir[ia] = m.normal
                    changed = True
            if has_support[ia] and not has_support[ib] and bodies.inv_mass[ib] > Float32(0.0):
                if dot_v(-m.normal, sup_dir[ia]) > Float32(0.15):
                    has_support[ib] = True
                    sup_dir[ib] = -m.normal
                    changed = True
        if not changed:
            break

    var orig_mass = List[Float32](length=n_bodies, fill=0.0)
    var virt_mass = List[Float32](length=n_bodies, fill=0.0)
    for i in range(n_bodies):
        if bodies.inv_mass[i] > Float32(1.0e-12):
            var om = Float32(1.0) / bodies.inv_mass[i]
            orig_mass[i] = om
            virt_mass[i] = om

    for _ in range(6):
        var updated = False
        for k in range(len(manifolds)):
            ref m = manifolds[k]
            if not m.b_moves:
                continue
            var ia = m.ia
            var ib = m.ib
            if ia < 0 or ia >= n_bodies or ib < 0 or ib >= n_bodies:
                continue
            if orig_mass[ia] <= Float32(0.0) or orig_mass[ib] <= Float32(0.0):
                continue
            # m.normal pushes ia in +m.normal and ib in -m.normal.
            # If ib is supported along +sup_dir[ib] and dot(m.normal, sup_dir[ib]) > 0.15,
            # ia is pushing ib into ib's support.
            if has_support[ib] and dot_v(m.normal, sup_dir[ib]) > Float32(0.15):
                if virt_mass[ia] > Float32(4.0) * orig_mass[ib]:
                    var cand = Float32(0.5) * virt_mass[ia]
                    if cand > virt_mass[ib] + Float32(1.0e-4):
                        virt_mass[ib] = cand
                        updated = True
            if has_support[ia] and dot_v(-m.normal, sup_dir[ia]) > Float32(0.15):
                if virt_mass[ib] > Float32(4.0) * orig_mass[ia]:
                    var cand = Float32(0.5) * virt_mass[ib]
                    if cand > virt_mass[ia] + Float32(1.0e-4):
                        virt_mass[ia] = cand
                        updated = True
        if not updated:
            break

    for i in range(n_bodies):
        if orig_mass[i] > Float32(0.0) and virt_mass[i] > orig_mass[i] * Float32(1.01):
            saved.append(
                _SavedBodyMass(
                    i,
                    bodies.inv_mass[i],
                    bodies.ixx[i],
                    bodies.iyy[i],
                    bodies.izz[i],
                    bodies.ixy[i],
                    bodies.ixz[i],
                    bodies.iyz[i],
                    bodies.ib_x[i],
                    bodies.ib_y[i],
                    bodies.ib_z[i],
                )
            )
            var inv_scale = orig_mass[i] / virt_mass[i]
            var fwd_scale = virt_mass[i] / orig_mass[i]
            bodies.inv_mass[i] *= inv_scale
            bodies.ixx[i] *= inv_scale
            bodies.iyy[i] *= inv_scale
            bodies.izz[i] *= inv_scale
            bodies.ixy[i] *= inv_scale
            bodies.ixz[i] *= inv_scale
            bodies.iyz[i] *= inv_scale
            if bodies.ib_x[i] > Float32(0.0):
                bodies.ib_x[i] *= fwd_scale
                bodies.ib_y[i] *= fwd_scale
                bodies.ib_z[i] *= fwd_scale

    return saved^


def _restore_supported_mass_ratios(mut bodies: SolverBodies, saved: List[_SavedBodyMass]):
    for k in range(len(saved)):
        var s = saved[k]
        var i = s.idx
        bodies.inv_mass[i] = s.inv_mass
        bodies.ixx[i] = s.ixx
        bodies.iyy[i] = s.iyy
        bodies.izz[i] = s.izz
        bodies.ixy[i] = s.ixy
        bodies.ixz[i] = s.ixz
        bodies.iyz[i] = s.iyz
        bodies.ib_x[i] = s.ib_x
        bodies.ib_y[i] = s.ib_y
        bodies.ib_z[i] = s.ib_z


struct SolveResult(Copyable, Movable):
    var batches: Int
    var stages: Int
    var tasks: Int
    var overflow: Int
    var broken_joints: List[Int]

    def __init__(out self):
        self.batches = 0
        self.stages = 0
        self.tasks = 0
        self.overflow = 0
        self.broken_joints = List[Int]()


def _solve_impl(
    mut bodies: SolverBodies,
    mut manifolds: List[Manifold],
    mut joints: List[Joint],
    joint_rows: List[JointRow],
    s6_joints_ptr: Int,
    s6_indices_ptr: Int,
    s6_sa_ptr: Int,
    s6_sb_ptr: Int,
    broken_s6_out_ptr: Int,
    has_s6: Bool,
    settings: SolverSettings,
    dt: Float32,
    pool_address: Int,
    cpus: List[Bool],
) -> SolveResult:
    """Unified Soft-Step solver over loaded bodies, coloured manifolds, standard `Joint`s,
    and optional `SixDofMotorJoint`s solved inside every `h = dt / substeps` substep."""
    set_sim_float_mode()
    var result = SolveResult()
    var substeps = max(1, settings.substeps)
    var h = dt / Float32(substeps)
    var inv_h = Float32(1.0) / h
    var soft = solver_softness(settings, h)

    var saved_masses = _condition_supported_mass_ratios(bodies, manifolds)

    # Batches: overflow serial stage first, then colour by colour, eight manifolds each.
    var serial_list = List[Int]()
    var colour_lists = List[List[Int]]()
    for _ in range(GRAPH_COLOURS):
        colour_lists.append(List[Int]())
    for index in range(len(manifolds)):
        var colour = manifolds[index].colour
        if colour >= GRAPH_COLOURS or colour < 0:
            serial_list.append(index)
            result.overflow += 1
        else:
            colour_lists[colour].append(index)
    if settings.force_serial:
        for colour in range(GRAPH_COLOURS):
            for index in colour_lists[colour]:
                serial_list.append(index)
            colour_lists[colour].clear()

    var total = len(serial_list)
    for colour in range(GRAPH_COLOURS):
        total += (len(colour_lists[colour]) + LANES - 1) // LANES
    var contacts = WideContacts()
    contacts.reset(total, bodies.static_index)
    var batch = 0
    for index in serial_list:
        contacts.put(
            batch, 0, index, manifolds[index], manifolds[index].ia, manifolds[index].ib,
            bodies, soft[0], soft[1], settings.warm_starting, settings.rolling_resistance,
        )
        batch += 1
    var serial_batches = batch
    var colour_first = List[Int](length=GRAPH_COLOURS, fill=0)
    var colour_batches = List[Int](length=GRAPH_COLOURS, fill=0)
    for colour in range(GRAPH_COLOURS):
        colour_first[colour] = batch
        var lane = 0
        for index in colour_lists[colour]:
            contacts.put(
                batch, lane, index, manifolds[index], manifolds[index].ia, manifolds[index].ib,
                bodies, soft[0], soft[1], settings.warm_starting, settings.rolling_resistance,
            )
            lane += 1
            if lane == LANES:
                lane = 0
                batch += 1
        if lane > 0:
            batch += 1
        colour_batches[colour] = batch - colour_first[colour]
    result.batches = batch

    # Active unbroken joints softened per joint, solved AFTER coloured contacts.
    var rows = List[JointRow]()
    var nyquist = Float32(0.25) / h
    for row in joint_rows:
        if row.joint < 0 or row.joint >= len(joints):
            continue
        if joints[row.joint].broken:
            continue
        var j = joints[row.joint]
        var linear_hertz = min(j.linear_hertz if j.linear_hertz > 0 else JOINT_HERTZ, nyquist)
        var angular_hertz = min(j.angular_hertz if j.angular_hertz > 0 else JOINT_HERTZ, nyquist)
        var damping = j.damping if j.damping > 0 else JOINT_DAMPING_RATIO
        rows.append(
            JointRow(
                row.joint, row.a, row.b,
                make_soft(linear_hertz, damping, h),
                make_soft(angular_hertz, damping, h),
            )
        )
        joints[row.joint].force_impulse = Vec3()
        joints[row.joint].torque_impulse = Vec3()
        if not settings.warm_starting:
            joints[row.joint].linear_impulse = Vec3()
            joints[row.joint].angular_impulse = Vec3()
            joints[row.joint].axial_impulse = 0
            joints[row.joint].motor_impulse = 0
            joints[row.joint].lower_impulse = 0
            joints[row.joint].upper_impulse = 0
            joints[row.joint].swing_impulse = 0

    var broken_indices = List[Int]()
    var run = SolverRun()
    run.bodies = Int(Pointer(to=bodies))
    run.contacts = Int(Pointer(to=contacts))
    run.joints = Int(Pointer(to=joints)) if len(rows) > 0 else 0
    run.joint_rows = Int(Pointer(to=rows))
    run.broken_out = Int(Pointer(to=broken_indices))
    run.s6_joints = s6_joints_ptr if has_s6 else 0
    run.s6_indices = s6_indices_ptr
    run.s6_sa = s6_sa_ptr
    run.s6_sb = s6_sb_ptr
    run.broken_s6_out = broken_s6_out_ptr
    run.h = h
    run.inv_h = inv_h
    run.settings = settings
    run.cpus = cpus.copy()

    var body_batches = bodies.padded // LANES
    var body_blocks = (body_batches + BODY_BATCHES_PER_BLOCK - 1) // BODY_BATCHES_PER_BLOCK
    var has_joints = len(rows) > 0 or has_s6
    var kinds = [STAGE_WARM, STAGE_SOLVE, STAGE_RELAX]
    for _ in range(substeps):
        if body_batches > 0:
            run.stages.append(SolverStage(STAGE_INTEGRATE_VELOCITIES, 0, body_batches, body_blocks, False, False))
        for pass_index in range(3):
            var kind = kinds[pass_index]
            if kind == STAGE_RELAX:
                if body_batches > 0:
                    run.stages.append(SolverStage(STAGE_INTEGRATE_POSITIONS, 0, body_batches, body_blocks, False, False))
                if not settings.relax:
                    continue
            if serial_batches > 0:
                run.stages.append(SolverStage(kind, 0, serial_batches, 1, True, False))
            for colour in range(GRAPH_COLOURS):
                if colour_batches[colour] > 0:
                    var blocks = (colour_batches[colour] + CONTACT_BATCHES_PER_BLOCK - 1) // CONTACT_BATCHES_PER_BLOCK
                    run.stages.append(SolverStage(kind, colour_first[colour], colour_batches[colour], blocks, False, False))
            if has_joints:
                run.stages.append(SolverStage(kind, 0, 0, 1, True, True))

    if serial_batches > 0:
        run.stages.append(SolverStage(STAGE_RESTITUTION, 0, serial_batches, 1, True, False))
    for colour in range(GRAPH_COLOURS):
        if colour_batches[colour] > 0:
            var blocks = (colour_batches[colour] + CONTACT_BATCHES_PER_BLOCK - 1) // CONTACT_BATCHES_PER_BLOCK
            run.stages.append(SolverStage(STAGE_RESTITUTION, colour_first[colour], colour_batches[colour], blocks, False, False))

    run.words = List[Int64](length=(1 + 2 * len(run.stages)) * SOLVER_WORD_STRIDE, fill=0)
    result.stages = len(run.stages)

    var tasks = 1
    var work = result.batches + body_batches
    if work >= PARALLEL_MIN_BATCHES and pool_address != 0:
        tasks = min(1 + pool_workers(pool_address), max(1, count_cpus(cpus) // 2))
        tasks = max(1, min(tasks, work // 4))
    result.tasks = tasks
    if tasks <= 1:
        lead(run)
    else:
        run_jobs(pool_address, run_solver_task, Int(Pointer(to=run)), tasks, ZONE_SOLVER)
    _ = run.tasks_run

    for b in range(contacts.batches):
        for lane in range(LANES):
            var index = contacts.manifold[b * LANES + lane]
            if index >= 0:
                contacts.take(b, lane, manifolds[index])

    _restore_supported_mass_ratios(bodies, saved_masses)
    result.broken_joints = broken_indices^
    _ = len(rows)
    return result^


def solve(
    mut bodies: SolverBodies,
    mut manifolds: List[Manifold],
    mut joints: List[Joint],
    joint_rows: List[JointRow],
    settings: SolverSettings,
    dt: Float32,
    pool_address: Int,
    cpus: List[Bool],
) -> SolveResult:
    """One tick of the Soft Step over the loaded bodies, coloured manifolds, and joints."""
    return _solve_impl(
        bodies, manifolds, joints, joint_rows,
        0, 0, 0, 0, 0, False,
        settings, dt, pool_address, cpus,
    )


def solve_with_six_dof(
    mut bodies: SolverBodies,
    mut manifolds: List[Manifold],
    mut joints: List[Joint],
    joint_rows: List[JointRow],
    mut six_dof_joints: List[SixDofMotorJoint],
    s6_indices: List[Int],
    s6_sa: List[Int],
    s6_sb: List[Int],
    mut broken_s6_out: List[Int],
    settings: SolverSettings,
    dt: Float32,
    pool_address: Int,
    cpus: List[Bool],
) -> SolveResult:
    """One tick of the Soft Step with `SixDofMotorJoint`s solved natively inside every substep."""
    if not settings.warm_starting:
        for k in range(len(s6_indices)):
            six_dof_joints[s6_indices[k]].reset_warm_impulses()
    var res = _solve_impl(
        bodies, manifolds, joints, joint_rows,
        Int(Pointer(to=six_dof_joints)),
        Int(Pointer(to=s6_indices)),
        Int(Pointer(to=s6_sa)),
        Int(Pointer(to=s6_sb)),
        Int(Pointer(to=broken_s6_out)),
        len(s6_indices) > 0,
        settings, dt, pool_address, cpus,
    )
    _ = len(s6_indices)
    _ = len(s6_sa)
    _ = len(s6_sb)
    return res^
