"""The sub-stepped soft-constraint solver (package 2.2): Box2D v3's Soft Step.

One tick (1/60 s) is PHYSICS_SUBSTEPS substeps of h. Collision runs once per
tick (rigid_body.mojo); each contact becomes a manifold of up to four points
with anchors fixed to the bodies. Then, per substep:

    integrate velocities   gravity, the fields (2.12), damping, speed clamps
    warm start             apply last substep's accumulated impulses
    solve                  soft contacts (and joints) with the push-out bias
    integrate positions    dp += v h, da += w h, the turn exactly
    relax                  the same constraints without the bias

and after the last substep a restitution pass and the impulses are stored for
the next tick's warm start (the contact cache keeps them by contact id).
A point's separation is updated each substep from the bodies' movement since
the tick began (dp, and da x r for the turn), so a contact found a little apart
(a *speculative* or predicted contact, s > 0) only lets the bodies close the
gap: v_n >= -s / h. Below zero the contact is soft: Box2D's mass and impulse
scales from a spring of CONTACT_HERTZ (twice that against the terrain, which
cannot move) and CONTACT_DAMPING_RATIO, pushing out at most
CONTACT_PUSH_MAX_SPEED. Overlap up to LINEAR_SLOP is left alone.

Friction is two tangent rows per point, clamped to a disc of radius
mu * normal impulse; a conveyor (package 2.14) gives a manifold a target
sliding velocity the rows drive toward instead of zero. Rolling resistance is one angular block per manifold on
the bodies' relative turn, clamped to ROLLING_RESISTANCE times the manifold's
normal impulse (Box2D v3.1). Restitution runs only for points that approached
faster than RESTITUTION_THRESHOLD and pushed this tick.

Parallel without locks: constraints are coloured (physics/islands.mojo), and
each colour's manifolds are packed eight to a batch, one per lane, so no two
lanes of a batch, nor two batches of a colour, touch the same moving body.
Every pass is a list of *stages*: a body stage (integration, blocks of
bodies), the serial stage (joints, then the overflow colour, one manifold per
batch, in order, on one worker) and one stage per colour (blocks of batches).
`run_solver` runs them on one long-running task per worker: the task that runs
chunk 0 leads, publishing each stage's index; every task claims the stage's
blocks with an atomic counter, and the leader waits for the stage's done
counter before the next. A task that starts late, or not at all, only means the
others do more, so the result never depends on the worker count (Box2D v3's
b2SolverStage, instead of about a hundred job-graph nodes per tick).

The forced serial path (debug cvar physics.serial_overflow) puts every manifold
in a batch of its own, in colour order, on the serial stage. Lanes never mix
and colours never share a moving body, so it gives the very same bits; the
solver test checks it.
"""
from std.math import sqrt
from std.atomic import Atomic
from std.ffi import external_call
from flint.core.math import Vec3, Quat
from flint.core.jobs import run_jobs, JobPool
from flint.core.profiler import ZONE_SOLVER
from flint.platform.os import open_read, close_fd
from flint.physics.solver_bodies import (
    LANES, F8, I8, V8, S8, SolverBodies, Sym3, ld, st, ld_i, gather, scatter,
)
from flint.physics.joints import Joint, JointRow, Softness, make_soft, joint_warm, joint_solve
from flint.physics.islands import GRAPH_COLOURS, OVERFLOW_COLOUR
from flint.physics.defaults import (
    PHYSICS_SUBSTEPS, CONTACT_HERTZ, CONTACT_DAMPING_RATIO, CONTACT_PUSH_MAX_SPEED, RESTITUTION_THRESHOLD,
    BODY_MAX_SPEED, JOINT_HERTZ, JOINT_DAMPING_RATIO, ROLLING_RESISTANCE,
)

comptime POINTS: Int = 4                 # per manifold
comptime LINEAR_SLOP: Float32 = 0.005    # overlap left alone (Box2D's b2_linearSlop)
comptime SPECULATIVE_DISTANCE: Float32 = 0.02  # contacts are found this far apart (4 x slop)
comptime MAX_TURN_PER_SUBSTEP: Float32 = 1.5707963  # a quarter turn
comptime BODY_BATCHES_PER_BLOCK: Int = 4
comptime CONTACT_BATCHES_PER_BLOCK: Int = 4
comptime PARALLEL_MIN_BATCHES: Int = 24  # less work than this runs on the calling thread alone
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
comptime L_SV: Int = 38   # three: the target sliding velocity of a on b (conveyors, package 2.14)
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
comptime P_T1MASS: Int = 8
comptime P_T2MASS: Int = 9
comptime P_NIMP: Int = 10
comptime P_T1IMP: Int = 11
comptime P_T2IMP: Int = 12
comptime P_TOTAL: Int = 13
comptime P_MAXIMP: Int = 14
comptime P_RELVEL: Int = 15
comptime P_COUNT: Int = 16


struct SolverSettings(Copyable, ImplicitlyCopyable, Movable):
    """What one tick's solve needs besides the bodies and constraints. The
    tunables give the defaults; tests may turn warm starting or rolling
    resistance off to measure what they do."""
    var substeps: Int
    var contact_hertz: Float32
    var damping_ratio: Float32
    var push_max_speed: Float32
    var restitution_threshold: Float32
    var max_speed: Float32
    var warm_starting: Bool
    var relax: Bool
    var rolling_resistance: Float32
    var force_serial: Bool  # the debug cvar: every manifold on the serial stage, in colour order

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
    (row `b`, or -1 for the terrain and static boxes), one normal from b
    toward a."""
    var a: Int
    var b: Int
    var b_moves: Bool     # b is solved this tick (awake); otherwise it is the zero body
    var other: Int        # b's body id, OTHER_TERRAIN or a static box (events.static_other)
    var normal: Vec3
    var friction: Float32
    var restitution: Float32
    var rolling: Float32
    var colour: Int
    var points: List[ContactPoint]
    var rolling_impulse: Vec3
    var surface: Vec3     # the sliding velocity of a on b friction drives toward (conveyors, 2.14); zero for most
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


struct WideContacts(Movable):
    """Manifolds in batches of eight, one per lane. `ia`/`ib` are solver body
    indices (the zero body for a side that does not move); `manifold` maps a
    lane back to its manifold (-1 for padding)."""
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
        self.lanes = List[Float32](length=L_COUNT * batches * LANES, fill=0)
        self.points = List[Float32](length=P_COUNT * batches * POINTS * LANES, fill=0)
        self.ia = List[Int32](length=batches * LANES, fill=Int32(static_index))
        self.ib = List[Int32](length=batches * LANES, fill=Int32(static_index))
        self.manifold = List[Int](length=batches * LANES, fill=-1)

    def put(mut self, batch: Int, lane: Int, index: Int, m: Manifold, a: Int, b: Int, bodies: SolverBodies, contact: Softness, terrain: Softness, warm: Bool):
        """Writes manifold `index` into a lane: the anchors, masses and warm
        start impulses (the prepare step)."""
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
        var values = [n.x, n.y, n.z, t1.x, t1.y, t1.z, t2.x, t2.y, t2.z, m.friction, m.restitution, m.rolling,
                      soft.bias_rate, soft.mass_scale, soft.impulse_scale, ma, mb,
                      ia.xx, ia.yy, ia.zz, ia.xy, ia.xz, ia.yz, ib.xx, ib.yy, ib.zz, ib.xy, ib.xz, ib.yz,
                      rk.xx, rk.yy, rk.zz, rk.xy, rk.xz, rk.yz]
        for field in range(len(values)):
            self.lanes[self.lane_at(field, batch) + lane] = values[field]
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
            var point_values = [ra.x, ra.y, ra.z, rb.x, rb.y, rb.z, -cp.depth,
                                effective_mass(ma, mb, ia, ib, ra, rb, n), effective_mass(ma, mb, ia, ib, ra, rb, t1),
                                effective_mass(ma, mb, ia, ib, ra, rb, t2), nimp, dot_v(timp, t1), dot_v(timp, t2),
                                Float32(0), Float32(0), dot_v(dv, n)]
            for field in range(len(point_values)):
                self.points[self.point_at(field, batch, p) + lane] = point_values[field]

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
        return S8(ld(self.lanes, at), ld(self.lanes, at + step), ld(self.lanes, at + 2 * step),
                  ld(self.lanes, at + 3 * step), ld(self.lanes, at + 4 * step), ld(self.lanes, at + 5 * step))

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


def warm_batch(mut c: WideContacts, mut bodies: SolverBodies, batch: Int):
    var ia_index = ld_i(c.ia, batch * LANES)
    var ib_index = ld_i(c.ib, batch * LANES)
    var a = gather_motion(bodies, ia_index)
    var b = gather_motion(bodies, ib_index)
    var n = c.lane_v(L_NX, batch)
    var t1 = c.lane_v(L_T1X, batch)
    var t2 = c.lane_v(L_T2X, batch)
    var ma = ld(c.lanes, c.lane_at(L_MA, batch))
    var mb = ld(c.lanes, c.lane_at(L_MB, batch))
    var ia = c.lane_s(L_IA, batch)
    var ib = c.lane_s(L_IB, batch)
    comptime for p in range(POINTS):
        var ra = c.point_v(P_RAX, batch, p)
        var rb = c.point_v(P_RBX, batch, p)
        var impulse = n.scale(ld(c.points, c.point_at(P_NIMP, batch, p))) + t1.scale(ld(c.points, c.point_at(P_T1IMP, batch, p))) + t2.scale(ld(c.points, c.point_at(P_T2IMP, batch, p)))
        a.v = a.v + impulse.scale(ma)
        a.w = a.w + ia.apply(ra.cross(impulse))
        b.v = b.v - impulse.scale(mb)
        b.w = b.w - ib.apply(rb.cross(impulse))
    var roll = c.lane_v(L_RI, batch)
    a.w = a.w + ia.apply(roll)
    b.w = b.w - ib.apply(roll)
    scatter_motion(bodies, ia_index, a)
    scatter_motion(bodies, ib_index, b)


def solve_batch(mut c: WideContacts, mut bodies: SolverBodies, batch: Int, inv_h: Float32, use_bias: Bool, push_max_speed: Float32):
    """The normal rows (soft, with the bias when `use_bias`), friction, then
    rolling resistance, for the batch's eight manifolds."""
    var ia_index = ld_i(c.ia, batch * LANES)
    var ib_index = ld_i(c.ib, batch * LANES)
    var a = gather_motion(bodies, ia_index)
    var b = gather_motion(bodies, ib_index)
    var dpa = V8(gather(bodies.dpx, ia_index), gather(bodies.dpy, ia_index), gather(bodies.dpz, ia_index))
    var daa = V8(gather(bodies.dax, ia_index), gather(bodies.day, ia_index), gather(bodies.daz, ia_index))
    var dpb = V8(gather(bodies.dpx, ib_index), gather(bodies.dpy, ib_index), gather(bodies.dpz, ib_index))
    var dab = V8(gather(bodies.dax, ib_index), gather(bodies.day, ib_index), gather(bodies.daz, ib_index))
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
        var ra = c.point_v(P_RAX, batch, p)
        var rb = c.point_v(P_RBX, batch, p)
        # The separation now, from how far the anchors moved since the tick began.
        var move_a = dpa + daa.cross(ra)
        var move_b = dpb + dab.cross(rb)
        var s = n.dot(move_a - move_b) + ld(c.points, c.point_at(P_SEP, batch, p))
        var speculative = s.gt(zero)
        var bias: F8
        var mass_scale: F8
        var impulse_scale: F8
        if use_bias:
            bias = speculative.select(s * F8(inv_h), max(bias_rate * min(zero, s + F8(LINEAR_SLOP)), F8(-push_max_speed)))
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
    # A conveyor (2.14) moves the zero of friction: a slides on b at `surface`.
    # Zero for most lanes, and x - 0 is x: every other contact keeps its bits.
    var surface = c.lane_v(L_SV, batch)
    comptime for p in range(POINTS):
        var ra = c.point_v(P_RAX, batch, p)
        var rb = c.point_v(P_RBX, batch, p)
        var dv = (a.v + a.w.cross(ra)) - (b.v + b.w.cross(rb)) - surface
        var at1 = c.point_at(P_T1IMP, batch, p)
        var at2 = c.point_at(P_T2IMP, batch, p)
        var old1 = ld(c.points, at1)
        var old2 = ld(c.points, at2)
        var new1 = old1 - ld(c.points, c.point_at(P_T1MASS, batch, p)) * t1.dot(dv)
        var new2 = old2 - ld(c.points, c.point_at(P_T2MASS, batch, p)) * t2.dot(dv)
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
    # Rolling resistance: the relative turn, stopped by at most rolling x the normal impulse.
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
        var ra = c.point_v(P_RAX, batch, p)
        var rb = c.point_v(P_RBX, batch, p)
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
    var serial: Bool   # the serial stage: joints, then every batch in order, one block

    def __init__(out self, kind: Int, first: Int, batches: Int, blocks: Int, serial: Bool):
        self.kind = kind
        self.first = first
        self.batches = batches
        self.blocks = blocks
        self.serial = serial


struct SolverRun(Movable):
    """What the solver tasks share, by address, and the stage counters."""
    var bodies: Int
    var contacts: Int
    var joints: Int
    var joint_rows: Int
    var stages: List[SolverStage]
    var words: List[Int64]   # [0]: the running stage + 1; then per stage: claimed, done
    var h: Float32
    var inv_h: Float32
    var settings: SolverSettings
    var cpus: List[Bool]     # which CPUs a helping task may run on (one CCD)
    var tasks_run: Int       # diagnostics, written by the leader

    def __init__(out self):
        self.bodies = 0
        self.contacts = 0
        self.joints = 0
        self.joint_rows = 0
        self.stages = List[SolverStage]()
        self.words = List[Int64]()
        self.h = 0
        self.inv_h = 0
        self.settings = SolverSettings()
        self.cpus = List[Bool]()
        self.tasks_run = 0

    @always_inline
    def word(self, index: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=Int(self.words.unsafe_ptr()) + index * 8)


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
    var first = stage.first
    var end = stage.first + stage.batches
    if stage.serial:
        if stage.kind != STAGE_RESTITUTION and run.joints != 0:
            ref joints = Pointer[List[Joint], origin=MutAnyOrigin](unsafe_from_address=run.joints)[unsafe_offset=0]
            ref rows = Pointer[List[JointRow], origin=MutAnyOrigin](unsafe_from_address=run.joint_rows)[unsafe_offset=0]
            for k in range(len(rows)):
                ref row = rows[k]
                if stage.kind == STAGE_WARM:
                    joint_warm(joints[row.joint], row, bodies)
                else:
                    joint_solve(joints[row.joint], row, bodies, run.h, stage.kind == STAGE_SOLVE)
    else:
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
    """A helping task: runs the blocks of each stage the leader publishes,
    until the leader is done. It only helps from a CPU of the physics CCD."""
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
            return  # the leader is busy elsewhere for long: leave it the work


def run_solver_task(context: Int, chunk: Int):
    ref run = Pointer[SolverRun, origin=MutAnyOrigin](unsafe_from_address=context)[unsafe_offset=0]
    if chunk == 0:
        lead(run)
    else:
        help(run)


def ccd_cpus() -> List[Bool]:
    """The CPUs sharing the L3 cache of CPU 0 (one CCD on a Ryzen), from
    /sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list; every CPU when
    that cannot be read."""
    return parse_cpu_list(read_small("/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list"))


def read_small(path: String) -> String:
    """A short sysfs file (whose reported size is a page, whatever it holds)."""
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
    """A kernel CPU list ("0-7,16-23") as a mask over CPU numbers. An empty
    or unreadable list allows every CPU (a mask of 1024 trues)."""
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
    """Contact softness between bodies, and against what does not move (twice
    as stiff), as Box2D v3: the stiffness is capped at a quarter of the
    substep rate."""
    var hertz = min(settings.contact_hertz, 0.25 / h)
    return (make_soft(hertz, settings.damping_ratio, h), make_soft(2.0 * hertz, settings.damping_ratio, h))


struct SolveResult(Copyable, ImplicitlyCopyable, Movable):
    var batches: Int
    var stages: Int
    var tasks: Int
    var overflow: Int

    def __init__(out self):
        self.batches = 0
        self.stages = 0
        self.tasks = 0
        self.overflow = 0


def solve(
    mut bodies: SolverBodies, mut manifolds: List[Manifold], mut joints: List[Joint], joint_rows: List[JointRow],
    settings: SolverSettings, dt: Float32, pool_address: Int, cpus: List[Bool],
) -> SolveResult:
    """One tick of the Soft Step over the loaded bodies, the coloured
    manifolds (`m.ia`/`m.ib` are solver body indices, the zero body for a
    side that does not move) and the joints."""
    var result = SolveResult()
    var substeps = max(1, settings.substeps)
    var h = dt / Float32(substeps)
    var inv_h = 1.0 / h
    var soft = solver_softness(settings, h)
    # Batches: the serial stage's (overflow, then everything when forced),
    # then colour by colour, eight manifolds each.
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
        contacts.put(batch, 0, index, manifolds[index], manifolds[index].ia, manifolds[index].ib, bodies, soft[0], soft[1], settings.warm_starting)
        batch += 1
    var serial_batches = batch
    var colour_first = List[Int](length=GRAPH_COLOURS, fill=0)
    var colour_batches = List[Int](length=GRAPH_COLOURS, fill=0)
    for colour in range(GRAPH_COLOURS):
        colour_first[colour] = batch
        var lane = 0
        for index in colour_lists[colour]:
            contacts.put(batch, lane, index, manifolds[index], manifolds[index].ia, manifolds[index].ib, bodies, soft[0], soft[1], settings.warm_starting)
            lane += 1
            if lane == LANES:
                lane = 0
                batch += 1
        if lane > 0:
            batch += 1
        colour_batches[colour] = batch - colour_first[colour]
    result.batches = batch
    # Joints start each tick from their stored impulses, softened per joint.
    var rows = List[JointRow]()
    for row in joint_rows:
        var j = joints[row.joint]
        # A joint with no stiffness of its own is as stiff as the tunables say
        # (Box2D's joint softness); a motor always has its own.
        var linear_hertz = min(j.linear_hertz if j.linear_hertz > 0 else JOINT_HERTZ, 0.25 / h)
        var angular_hertz = min(j.angular_hertz if j.angular_hertz > 0 else JOINT_HERTZ, 0.25 / h)
        var damping = j.damping if j.damping > 0 else JOINT_DAMPING_RATIO
        rows.append(JointRow(row.joint, row.a, row.b, make_soft(linear_hertz, damping, h), make_soft(angular_hertz, damping, h)))
        if not settings.warm_starting:
            joints[row.joint].linear_impulse = Vec3()
            joints[row.joint].angular_impulse = Vec3()
    # Stages.
    var run = SolverRun()
    run.bodies = Int(Pointer(to=bodies))
    run.contacts = Int(Pointer(to=contacts))
    run.joints = Int(Pointer(to=joints)) if len(rows) > 0 else 0
    run.joint_rows = Int(Pointer(to=rows))
    run.h = h
    run.inv_h = inv_h
    run.settings = settings
    run.cpus = cpus.copy()
    var body_batches = bodies.padded // LANES
    var body_blocks = (body_batches + BODY_BATCHES_PER_BLOCK - 1) // BODY_BATCHES_PER_BLOCK
    var has_serial = serial_batches > 0 or len(rows) > 0
    var kinds = [STAGE_WARM, STAGE_SOLVE, STAGE_RELAX]
    for _ in range(substeps):
        run.stages.append(SolverStage(STAGE_INTEGRATE_VELOCITIES, 0, body_batches, body_blocks, False))
        for pass_index in range(3):
            var kind = kinds[pass_index]
            if kind == STAGE_RELAX:
                run.stages.append(SolverStage(STAGE_INTEGRATE_POSITIONS, 0, body_batches, body_blocks, False))
                if not settings.relax:
                    continue
            if has_serial:
                run.stages.append(SolverStage(kind, 0, serial_batches, 1, True))
            for colour in range(GRAPH_COLOURS):
                if colour_batches[colour] > 0:
                    var blocks = (colour_batches[colour] + CONTACT_BATCHES_PER_BLOCK - 1) // CONTACT_BATCHES_PER_BLOCK
                    run.stages.append(SolverStage(kind, colour_first[colour], colour_batches[colour], blocks, False))
    if serial_batches > 0:
        run.stages.append(SolverStage(STAGE_RESTITUTION, 0, serial_batches, 1, True))
    for colour in range(GRAPH_COLOURS):
        if colour_batches[colour] > 0:
            var blocks = (colour_batches[colour] + CONTACT_BATCHES_PER_BLOCK - 1) // CONTACT_BATCHES_PER_BLOCK
            run.stages.append(SolverStage(STAGE_RESTITUTION, colour_first[colour], colour_batches[colour], blocks, False))
    run.words = List[Int64](length=1 + 2 * len(run.stages), fill=0)
    result.stages = len(run.stages)
    # Tasks: one per worker on the physics CCD, and the caller; small ticks
    # run on the caller alone (the same result, without waking anyone).
    var tasks = 1
    var work = result.batches + body_batches
    if work >= PARALLEL_MIN_BATCHES:
        tasks = min(1 + pool_workers(pool_address), max(1, count_cpus(cpus) // 2))
        tasks = max(1, min(tasks, work // 4))
    result.tasks = tasks
    if tasks <= 1:
        lead(run)
    else:
        run_jobs(pool_address, run_solver_task, Int(Pointer(to=run)), tasks, ZONE_SOLVER)
    _ = run.tasks_run
    # Impulses back into the manifolds.
    for b in range(contacts.batches):
        for lane in range(LANES):
            var index = contacts.manifold[b * LANES + lane]
            if index >= 0:
                contacts.take(b, lane, manifolds[index])
    _ = len(rows)
    return result
