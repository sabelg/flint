"""Soft joints: the breakable weld and the soft motor (package 2.2).

Both follow Box2D v3's joint solver: impulses accumulate over the substeps and
are warm-started; a soft constraint (`Softness`) pushes the error out at a
chosen stiffness and damping instead of all at once; the relax pass removes the
push's velocity.

Weld. Holds body B to body A (or to the world) at an anchor and at the turn
they had when it was made: a point constraint (3 rows) and an angular lock
(3 rows), each solved as a block with its 3x3 effective mass. A weld with no
stiffness of its own uses JOINT_HERTZ and JOINT_DAMPING_RATIO. It breaks when
its force (the linear impulse of a substep over the substep) passes
`break_force`, or its torque passes `break_torque`; a broken joint stays in the
table with its flag set (the flag is state), is never solved again, and
reports one joint-break event. The Finals keeps hinged pieces on such joints
between rigid debris and full fracture; 2.5 hangs pre-cut planks on them.

Motor. Drives body B's centre of mass toward a target pose (relative to A, or
in the world when A is the world) with a soft spring, but never with more
than `max_force` and `max_torque`: the accumulated impulses are clamped to
max * h each substep, as Box2D's motor joint does. 2.3 carries pieces with it:
a light piece follows the paw, a heavy one lags and sags.

Package 2.12 adds the remaining kinds, each with limits, a motor, a spring
and breaking, after Box2D v3's revolute, prismatic and distance joints
(scalar rows with speculative limits) carried to 3D:

    revolute     a hinge: B's anchor held to A's (3 rows), B's hinge axis
                 held along A's (2 angular rows); the angle about the axis
                 has limits, a motor (speed, max torque) and a spring
    prismatic    a slide: no turn relative to A (3 angular rows), the anchor
                 on A's axis line (2 rows); the travel along the axis has
                 limits, a motor (speed, max force) and a spring
    distance     the anchors kept `length` apart (rigid), or a spring toward
                 it, or free between min and max (`lower`, `upper`: a rope's
                 slack), with a motor along the line
    swing_twist  a ball joint (3 rows) with a cone limit on the swing of B's
                 twist axis away from A's, twist limits about it, and a
                 spring back to the made pose with at most `max_motor` torque
                 (a joint of a ragdoll; 3.11)

A broken joint of any kind stays in the table with its flag set. Breaking
compares the joint's whole force (every linear row's impulse this substep)
and torque with `break_force` and `break_torque`.
"""
from std.math import sqrt
from flint.core.math import Vec3, Quat
from flint.core.detmath import det_atan2, PI
from flint.core.serial import BlobWriter, BlobReader
from flint.physics.solver_bodies import SolverBodies, Sym3, skew_sandwich, rotate_vec

comptime JOINT_WELD: Int = 0
comptime JOINT_MOTOR: Int = 1
comptime JOINT_REVOLUTE: Int = 2
comptime JOINT_PRISMATIC: Int = 3
comptime JOINT_DISTANCE: Int = 4
comptime JOINT_SWING_TWIST: Int = 5
comptime JOINT_BLOB_VERSION: Int = 2  # SOLV v2: the 2.12 fields follow the 2.2 ones


def joint_kind_name(kind: Int) -> String:
    var names = ["weld", "motor", "revolute", "prismatic", "distance", "swing_twist"]
    return String(names[kind]) if kind >= 0 and kind < len(names) else "unknown"
comptime JOINT_WORLD: Int = 0  # body_a of a joint held by the world


struct Softness(Copyable, ImplicitlyCopyable, Movable):
    """Box2D v3's soft constraint coefficients for a spring of `hertz` and
    damping ratio `zeta`, stepped with substep `h` (Catto, "Solver2D", 2024)."""
    var bias_rate: Float32
    var mass_scale: Float32
    var impulse_scale: Float32

    def __init__(out self, bias_rate: Float32 = 0, mass_scale: Float32 = 1, impulse_scale: Float32 = 0):
        self.bias_rate = bias_rate
        self.mass_scale = mass_scale
        self.impulse_scale = impulse_scale


def make_soft(hertz: Float32, zeta: Float32, h: Float32) -> Softness:
    if hertz <= 0:
        return Softness(0, 1, 0)
    var omega = Float32(6.2831853) * hertz
    var a1 = 2.0 * zeta + h * omega
    var a2 = h * omega * a1
    var a3 = 1.0 / (1.0 + a2)
    return Softness(omega / a1, a2 * a3, a3)


def quat_conj(q: Quat) -> Quat:
    return Quat(-q.x, -q.y, -q.z, q.w)


def quat_product(a: Quat, b: Quat) -> Quat:
    """The rotation b, then a."""
    return Quat(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )


def rotation_error(actual: Quat, target: Quat) -> Vec3:
    """The small rotation (world axis times angle) from `target` to `actual`."""
    var e = quat_product(actual, quat_conj(target))
    var sign = Float32(1) if e.w >= 0 else Float32(-1)
    return Vec3(e.x, e.y, e.z) * (2.0 * sign)


struct Joint(Copyable, ImplicitlyCopyable, Movable):
    var id: Int
    var kind: Int
    var body_a: Int        # body id; JOINT_WORLD for the world
    var body_b: Int        # body id
    var local_a: Vec3      # weld: anchor in A's frame from its centre of mass (world point for the world); motor: target position of B
    var local_b: Vec3      # weld: anchor in B's frame from its centre of mass
    var reference: Quat    # weld: B's turn relative to A when made; motor: target turn of B relative to A
    var linear_hertz: Float32
    var angular_hertz: Float32
    var damping: Float32
    var break_force: Float32   # 0: unbreakable
    var break_torque: Float32
    var max_force: Float32     # motor
    var max_torque: Float32
    var linear_impulse: Vec3   # accumulated over the substeps (warm start)
    var angular_impulse: Vec3
    var broken: Bool
    # Package 2.12: axes, limits, motor and spring (all kinds but weld and
    # motor), and their accumulated row impulses (state, warm-started).
    var axis_a: Vec3       # hinge, slide or twist axis in A's frame (the world's for the world)
    var axis_b: Vec3       # the same axis in B's frame
    var lower: Float32     # revolute and twist: radians; prismatic: metres along the axis; distance: min length
    var upper: Float32
    var cone: Float32      # swing-twist: the swing limit in radians (0: free)
    var rest: Float32      # distance: rest length; revolute: spring angle; prismatic: spring travel
    var enable_limit: Bool
    var enable_motor: Bool
    var enable_spring: Bool
    var motor_speed: Float32   # rad/s or m/s
    var max_motor: Float32     # N m or N (swing-twist: the spring's torque cap; 0 none)
    var spring_hertz: Float32
    var spring_damping: Float32
    var axial_impulse: Float32 # spring, or a rigid distance
    var motor_impulse: Float32
    var lower_impulse: Float32
    var upper_impulse: Float32
    var swing_impulse: Float32
    # Scratch (not state): this substep's whole linear and angular impulse,
    # for the break check after the solve.
    var force_impulse: Vec3
    var torque_impulse: Vec3

    def __init__(out self):
        self.id = 0
        self.kind = JOINT_WELD
        self.body_a = JOINT_WORLD
        self.body_b = 0
        self.local_a = Vec3()
        self.local_b = Vec3()
        self.reference = Quat()
        self.linear_hertz = 0
        self.angular_hertz = 0
        self.damping = 0
        self.break_force = 0
        self.break_torque = 0
        self.max_force = 0
        self.max_torque = 0
        self.linear_impulse = Vec3()
        self.angular_impulse = Vec3()
        self.broken = False
        self.axis_a = Vec3(0, 1, 0)
        self.axis_b = Vec3(0, 1, 0)
        self.lower = 0
        self.upper = 0
        self.cone = 0
        self.rest = 0
        self.enable_limit = False
        self.enable_motor = False
        self.enable_spring = False
        self.motor_speed = 0
        self.max_motor = 0
        self.spring_hertz = 0
        self.spring_damping = 0
        self.axial_impulse = 0
        self.motor_impulse = 0
        self.lower_impulse = 0
        self.upper_impulse = 0
        self.swing_impulse = 0
        self.force_impulse = Vec3()
        self.torque_impulse = Vec3()

    def serialize(self, mut out: BlobWriter):
        out.integer(self.id)
        out.integer(self.kind)
        out.integer(self.body_a)
        out.integer(self.body_b)
        out.vec3(self.local_a)
        out.vec3(self.local_b)
        out.quat(self.reference)
        out.f32(self.linear_hertz)
        out.f32(self.angular_hertz)
        out.f32(self.damping)
        out.f32(self.break_force)
        out.f32(self.break_torque)
        out.f32(self.max_force)
        out.f32(self.max_torque)
        out.vec3(self.linear_impulse)
        out.vec3(self.angular_impulse)
        out.boolean(self.broken)
        out.vec3(self.axis_a)
        out.vec3(self.axis_b)
        out.f32(self.lower)
        out.f32(self.upper)
        out.f32(self.cone)
        out.f32(self.rest)
        out.boolean(self.enable_limit)
        out.boolean(self.enable_motor)
        out.boolean(self.enable_spring)
        out.f32(self.motor_speed)
        out.f32(self.max_motor)
        out.f32(self.spring_hertz)
        out.f32(self.spring_damping)
        out.f32(self.axial_impulse)
        out.f32(self.motor_impulse)
        out.f32(self.lower_impulse)
        out.f32(self.upper_impulse)
        out.f32(self.swing_impulse)

    @staticmethod
    def deserialize(mut input: BlobReader, version: Int = JOINT_BLOB_VERSION) raises -> Joint:
        var j = Joint()
        j.id = input.integer()
        j.kind = input.integer()
        j.body_a = input.integer()
        j.body_b = input.integer()
        j.local_a = input.vec3()
        j.local_b = input.vec3()
        j.reference = input.quat()
        j.linear_hertz = input.f32()
        j.angular_hertz = input.f32()
        j.damping = input.f32()
        j.break_force = input.f32()
        j.break_torque = input.f32()
        j.max_force = input.f32()
        j.max_torque = input.f32()
        j.linear_impulse = input.vec3()
        j.angular_impulse = input.vec3()
        j.broken = input.boolean()
        if version >= 2:
            j.axis_a = input.vec3()
            j.axis_b = input.vec3()
            j.lower = input.f32()
            j.upper = input.f32()
            j.cone = input.f32()
            j.rest = input.f32()
            j.enable_limit = input.boolean()
            j.enable_motor = input.boolean()
            j.enable_spring = input.boolean()
            j.motor_speed = input.f32()
            j.max_motor = input.f32()
            j.spring_hertz = input.f32()
            j.spring_damping = input.f32()
            j.axial_impulse = input.f32()
            j.motor_impulse = input.f32()
            j.lower_impulse = input.f32()
            j.upper_impulse = input.f32()
            j.swing_impulse = input.f32()
        if input.failed:
            raise Error("solver blob: truncated Joint")
        if j.kind < JOINT_WELD or j.kind > JOINT_SWING_TWIST:
            raise Error("solver blob: joint " + String(j.id) + " has unknown kind " + String(j.kind))
        return j


struct JointRow(Copyable, ImplicitlyCopyable, Movable):
    """A joint as the solver sees it this tick: its solver body indices (the
    zero body for the world or a body that does not move), its softness."""
    var joint: Int     # index into the joint table
    var a: Int
    var b: Int
    var linear: Softness
    var angular: Softness

    def __init__(out self, joint: Int, a: Int, b: Int, linear: Softness, angular: Softness):
        self.joint = joint
        self.a = a
        self.b = b
        self.linear = linear
        self.angular = angular


def pose_of(bodies: SolverBodies, index: Int) -> Tuple[Vec3, Quat]:
    """A solver body's centre and turn now (the world's for the zero body)."""
    if index >= bodies.count:
        return (Vec3(), Quat())
    return (bodies.center(index), bodies.orientation(index))


def joint_warm(mut joint: Joint, row: JointRow, mut bodies: SolverBodies):
    if joint.kind >= JOINT_REVOLUTE:
        joint_warm_2_12(joint, row, bodies)
        return
    var a = row.a
    var b = row.b
    var pa = pose_of(bodies, a)
    var pb = pose_of(bodies, b)
    var ra = arm_a(joint, pa[0], pa[1], pb[0])
    var rb = rotate_vec(pb[1], joint.local_b) if joint.kind == JOINT_WELD else Vec3()
    var p = joint.linear_impulse
    var l = joint.angular_impulse
    var ia = bodies.inertia(a)
    var ib = bodies.inertia(b)
    var ma = bodies.inv_mass[a]
    var mb = bodies.inv_mass[b]
    bodies.set_velocity(a, bodies.velocity(a) - p * ma, bodies.angular(a) - ia.apply(cross3(ra, p) + l))
    bodies.set_velocity(b, bodies.velocity(b) + p * mb, bodies.angular(b) + ib.apply(cross3(rb, p) + l))


def cross3(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def dot3(a: Vec3, b: Vec3) -> Float32:
    return a.x * b.x + a.y * b.y + a.z * b.z


def arm_a(joint: Joint, center_a: Vec3, turn_a: Quat, center_b: Vec3) -> Vec3:
    """A's lever to where the joint acts: the anchor, or for a motor B's
    centre (so A feels the motor's pull at B)."""
    if joint.kind != JOINT_MOTOR:
        return rotate_vec(turn_a, joint.local_a) if joint.body_a != JOINT_WORLD else Vec3()
    return center_b - center_a if joint.body_a != JOINT_WORLD else Vec3()


def clamp_length(value: Vec3, limit: Float32) -> Vec3:
    var length2 = dot3(value, value)
    if limit <= 0 or length2 <= limit * limit:
        return value
    return value * (limit / sqrt(length2))


def joint_solve(mut joint: Joint, row: JointRow, mut bodies: SolverBodies, h: Float32, use_bias: Bool):
    """One pass over one joint: the angular block, then the linear one."""
    if joint.kind >= JOINT_REVOLUTE:
        joint_solve_2_12(joint, row, bodies, h, use_bias)
        return
    var a = row.a
    var b = row.b
    var pose_a = pose_of(bodies, a)
    var pose_b = pose_of(bodies, b)
    var ma = bodies.inv_mass[a]
    var mb = bodies.inv_mass[b]
    var ia = bodies.inertia(a)
    var ib = bodies.inertia(b)
    var va = bodies.velocity(a)
    var wa = bodies.angular(a)
    var vb = bodies.velocity(b)
    var wb = bodies.angular(b)
    var motor = joint.kind == JOINT_MOTOR
    # Where B should be and how it should be turned.
    var target_turn = joint.reference
    var world_a = joint.body_a == JOINT_WORLD
    if not world_a:
        target_turn = quat_product(pose_a[1], joint.reference)
    # Angular block.
    var k_angular = ia.plus(ib).inverse()
    var bias = Vec3()
    var mass_scale = Float32(1)
    var impulse_scale = Float32(0)
    if use_bias or motor:
        var c = rotation_error(pose_b[1], target_turn)
        bias = c * row.angular.bias_rate
        mass_scale = row.angular.mass_scale
        impulse_scale = row.angular.impulse_scale
    var cdot = wb - wa
    var impulse = k_angular.apply(cdot + bias) * (-mass_scale) - joint.angular_impulse * impulse_scale
    if motor:
        var total = clamp_length(joint.angular_impulse + impulse, joint.max_torque * h)
        impulse = total - joint.angular_impulse
    joint.angular_impulse = joint.angular_impulse + impulse
    wa = wa - ia.apply(impulse)
    wb = wb + ib.apply(impulse)
    # Linear block.
    var ra = arm_a(joint, pose_a[0], pose_a[1], pose_b[0])
    var rb = rotate_vec(pose_b[1], joint.local_b) if not motor else Vec3()
    var k = Sym3(ma + mb, ma + mb, ma + mb).plus(skew_sandwich(ra, ia)).plus(skew_sandwich(rb, ib)).inverse()
    bias = Vec3()
    mass_scale = 1
    impulse_scale = 0
    if use_bias or motor:
        var target: Vec3
        if motor:
            target = joint.local_a if world_a else pose_a[0] + rotate_vec(pose_a[1], joint.local_a)
            target = target - rb
        else:
            target = (joint.local_a if world_a else pose_a[0] + ra)
        var c = (pose_b[0] + rb) - target
        bias = c * row.linear.bias_rate
        mass_scale = row.linear.mass_scale
        impulse_scale = row.linear.impulse_scale
    cdot = (vb + cross3(wb, rb)) - (va + cross3(wa, ra))
    impulse = k.apply(cdot + bias) * (-mass_scale) - joint.linear_impulse * impulse_scale
    if motor:
        var total = clamp_length(joint.linear_impulse + impulse, joint.max_force * h)
        impulse = total - joint.linear_impulse
    joint.linear_impulse = joint.linear_impulse + impulse
    if use_bias:
        if dot3(joint.linear_impulse, joint.linear_impulse) > dot3(joint.force_impulse, joint.force_impulse):
            joint.force_impulse = joint.linear_impulse
        if dot3(joint.angular_impulse, joint.angular_impulse) > dot3(joint.torque_impulse, joint.torque_impulse):
            joint.torque_impulse = joint.angular_impulse
    va = va - impulse * ma
    wa = wa - ia.apply(cross3(ra, impulse))
    vb = vb + impulse * mb
    wb = wb + ib.apply(cross3(rb, impulse))
    bodies.set_velocity(a, va, wa)
    bodies.set_velocity(b, vb, wb)


def joint_should_break(joint: Joint, inverse_h: Float32) -> Bool:
    """A joint (not a motor) whose peak solve force or torque passed its limit."""
    if joint.kind == JOINT_MOTOR or joint.broken:
        return False
    var lin_sq = dot3(joint.force_impulse, joint.force_impulse)
    var ang_sq = dot3(joint.torque_impulse, joint.torque_impulse)
    if joint.kind == JOINT_WELD:
        var cur_l = dot3(joint.linear_impulse, joint.linear_impulse)
        var cur_a = dot3(joint.angular_impulse, joint.angular_impulse)
        if cur_l > lin_sq:
            lin_sq = cur_l
        if cur_a > ang_sq:
            ang_sq = cur_a
    var force2 = lin_sq * inverse_h * inverse_h
    var torque2 = ang_sq * inverse_h * inverse_h
    if joint.break_force > 0 and force2 > joint.break_force * joint.break_force:
        return True
    if joint.break_torque > 0 and torque2 > joint.break_torque * joint.break_torque:
        return True
    return False


# Package 2.12: revolute, prismatic, distance and swing-twist joints.


def perp_basis(n: Vec3) -> Tuple[Vec3, Vec3]:
    """Two unit axes at right angles to unit n and to each other (a fixed
    choice: the same n gives the same pair)."""
    var t: Vec3
    if abs(n.x) < 0.57735:
        t = Vec3(0, n.z, -n.y) * (1.0 / sqrt(n.y * n.y + n.z * n.z))
    else:
        t = Vec3(n.z, 0, -n.x) * (1.0 / sqrt(n.x * n.x + n.z * n.z))
    return (t, cross3(n, t))


def unit_or(v: Vec3, fallback: Vec3) -> Vec3:
    var length2 = dot3(v, v)
    if length2 < 1.0e-12:
        return fallback
    return v * (1.0 / sqrt(length2))


def wrap_angle(angle: Float32) -> Float32:
    var a = angle
    if a > PI:
        a -= 2 * PI
    elif a < -PI:
        a += 2 * PI
    return a


def relative_turn(joint: Joint, turn_a: Quat, turn_b: Quat) -> Quat:
    """B's turn since the joint was made, in A's frame (identity then)."""
    var qa = turn_a if joint.body_a != JOINT_WORLD else Quat()
    return quat_product(quat_product(quat_conj(qa), turn_b), quat_conj(joint.reference))


def twist_angle(joint: Joint, turn_a: Quat, turn_b: Quat) -> Float32:
    """The angle B turned about the joint's axis since it was made (the
    twist part of the swing-twist split, in (-pi, pi])."""
    var r = relative_turn(joint, turn_a, turn_b)
    var along = r.x * joint.axis_a.x + r.y * joint.axis_a.y + r.z * joint.axis_a.z
    var w = r.w
    if w < 0:
        along = -along
        w = -w
    return wrap_angle(2.0 * det_atan2(along, w))


def world_axis_a(joint: Joint, turn_a: Quat) -> Vec3:
    return rotate_vec(turn_a, joint.axis_a) if joint.body_a != JOINT_WORLD else joint.axis_a


struct Motion(Copyable, ImplicitlyCopyable, Movable):
    """Two bodies' velocities and what moves them, while a joint is solved."""
    var va: Vec3
    var wa: Vec3
    var vb: Vec3
    var wb: Vec3
    var ma: Float32
    var mb: Float32
    var ia: Sym3
    var ib: Sym3

    def __init__(out self, bodies: SolverBodies, a: Int, b: Int):
        self.va = bodies.velocity(a)
        self.wa = bodies.angular(a)
        self.vb = bodies.velocity(b)
        self.wb = bodies.angular(b)
        self.ma = bodies.inv_mass[a]
        self.mb = bodies.inv_mass[b]
        self.ia = bodies.inertia(a)
        self.ib = bodies.inertia(b)

    def angular(mut self, impulse: Vec3):
        """An angular impulse on B (and its opposite on A)."""
        self.wa = self.wa - self.ia.apply(impulse)
        self.wb = self.wb + self.ib.apply(impulse)

    def linear(mut self, impulse: Vec3, ra: Vec3, rb: Vec3):
        """A linear impulse on B at rb (and its opposite on A at ra)."""
        self.va = self.va - impulse * self.ma
        self.wa = self.wa - self.ia.apply(cross3(ra, impulse))
        self.vb = self.vb + impulse * self.mb
        self.wb = self.wb + self.ib.apply(cross3(rb, impulse))

    def angular_mass(self, axis: Vec3) -> Float32:
        var k = dot3(axis, self.ia.plus(self.ib).apply(axis))
        return 1.0 / k if k > 0 else Float32(0)

    def linear_mass(self, axis: Vec3, ra: Vec3, rb: Vec3) -> Float32:
        var ca = cross3(ra, axis)
        var cb = cross3(rb, axis)
        var k = self.ma + self.mb + dot3(ca, self.ia.apply(ca)) + dot3(cb, self.ib.apply(cb))
        return 1.0 / k if k > 0 else Float32(0)

    def angular_cdot(self, axis: Vec3) -> Float32:
        return dot3(axis, self.wb - self.wa)

    def linear_cdot(self, axis: Vec3, ra: Vec3, rb: Vec3) -> Float32:
        return dot3(axis, (self.vb + cross3(self.wb, rb)) - (self.va + cross3(self.wa, ra)))

    def store(self, mut bodies: SolverBodies, a: Int, b: Int):
        bodies.set_velocity(a, self.va, self.wa)
        bodies.set_velocity(b, self.vb, self.wb)


struct Frame(Copyable, ImplicitlyCopyable, Movable):
    """Where a joint's anchors and axis are now."""
    var ra: Vec3        # A's lever to its anchor (zero for the world)
    var rb: Vec3
    var anchor_a: Vec3  # world
    var anchor_b: Vec3
    var axis: Vec3      # A's axis in the world
    var turn_a: Quat
    var turn_b: Quat

    def __init__(out self, joint: Joint, bodies: SolverBodies, a: Int, b: Int):
        var pa = pose_of(bodies, a)
        var pb = pose_of(bodies, b)
        self.turn_a = pa[1]
        self.turn_b = pb[1]
        var world_a = joint.body_a == JOINT_WORLD
        self.ra = rotate_vec(pa[1], joint.local_a) if not world_a else Vec3()
        self.anchor_a = joint.local_a if world_a else pa[0] + self.ra
        self.rb = rotate_vec(pb[1], joint.local_b)
        self.anchor_b = pb[0] + self.rb
        self.axis = world_axis_a(joint, pa[1])


def soft_terms(soft: Softness, use_bias: Bool) -> Tuple[Float32, Float32, Float32]:
    """(bias rate, mass scale, impulse scale) of a joint row: rigid with no
    push-out in the relax pass."""
    if use_bias:
        return (soft.bias_rate, soft.mass_scale, soft.impulse_scale)
    return (Float32(0), Float32(1), Float32(0))


def solve_point(mut joint: Joint, f: Frame, mut m: Motion, soft: Softness, use_bias: Bool):
    """B's anchor held to A's: a 3x3 block (the weld's linear block)."""
    var k = Sym3(m.ma + m.mb, m.ma + m.mb, m.ma + m.mb).plus(skew_sandwich(f.ra, m.ia)).plus(skew_sandwich(f.rb, m.ib)).inverse()
    var terms = soft_terms(soft, use_bias)
    var bias = (f.anchor_b - f.anchor_a) * terms[0]
    var cdot = (m.vb + cross3(m.wb, f.rb)) - (m.va + cross3(m.wa, f.ra))
    var impulse = k.apply(cdot + bias) * (-terms[1]) - joint.linear_impulse * terms[2]
    joint.linear_impulse = joint.linear_impulse + impulse
    m.linear(impulse, f.ra, f.rb)


def solve_turn_lock(mut joint: Joint, f: Frame, mut m: Motion, soft: Softness, use_bias: Bool):
    """No turn of B relative to A: a 3x3 angular block (the weld's)."""
    var target = joint.reference if joint.body_a == JOINT_WORLD else quat_product(f.turn_a, joint.reference)
    var k = m.ia.plus(m.ib).inverse()
    var terms = soft_terms(soft, use_bias)
    var bias = rotation_error(f.turn_b, target) * terms[0]
    var impulse = k.apply((m.wb - m.wa) + bias) * (-terms[1]) - joint.angular_impulse * terms[2]
    joint.angular_impulse = joint.angular_impulse + impulse
    m.angular(impulse)


def solve_axis_lock(mut joint: Joint, f: Frame, mut m: Motion, soft: Softness, use_bias: Bool):
    """B's axis held along A's: two angular rows at right angles to it,
    solved as a 2x2 block."""
    var b1 = rotate_vec(f.turn_b, joint.axis_b)
    var basis = perp_basis(f.axis)
    var t1 = basis[0]
    var t2 = basis[1]
    var c = cross3(f.axis, b1)
    var sum = m.ia.plus(m.ib)
    var k11 = dot3(t1, sum.apply(t1))
    var k12 = dot3(t1, sum.apply(t2))
    var k22 = dot3(t2, sum.apply(t2))
    var det = k11 * k22 - k12 * k12
    if abs(det) < 1.0e-12:
        return
    var inv = 1.0 / det
    var terms = soft_terms(soft, use_bias)
    var w = m.wb - m.wa
    var r1 = dot3(t1, w) + dot3(c, t1) * terms[0]
    var r2 = dot3(t2, w) + dot3(c, t2) * terms[0]
    var acc1 = dot3(joint.angular_impulse, t1)
    var acc2 = dot3(joint.angular_impulse, t2)
    var l1 = -terms[1] * (k22 * r1 - k12 * r2) * inv - terms[2] * acc1
    var l2 = -terms[1] * (k11 * r2 - k12 * r1) * inv - terms[2] * acc2
    joint.angular_impulse = t1 * (acc1 + l1) + t2 * (acc2 + l2)
    m.angular(t1 * l1 + t2 * l2)


def solve_perp_lock(mut joint: Joint, f: Frame, mut m: Motion, soft: Softness, use_bias: Bool):
    """A prismatic's anchor kept on A's axis line: two rows at right angles
    to the axis (A's lever reaches B's anchor)."""
    var d = f.anchor_b - f.anchor_a
    var ra = f.ra + d
    var basis = perp_basis(f.axis)
    var terms = soft_terms(soft, use_bias)
    var total = Vec3()
    for i in range(2):
        var t = basis[0] if i == 0 else basis[1]
        var k = m.linear_mass(t, ra, f.rb)
        var acc = dot3(joint.linear_impulse, t)
        var impulse = -k * terms[1] * (m.linear_cdot(t, ra, f.rb) + dot3(t, d) * terms[0]) - terms[2] * acc
        total = total + t * (acc + impulse)
        m.linear(t * impulse, ra, f.rb)
    joint.linear_impulse = total


struct AxisRow(Copyable, ImplicitlyCopyable, Movable):
    """One scalar degree of freedom of a joint: where it is (`value`), its
    direction and levers; `angular` rows turn, the others push."""
    var value: Float32
    var axis: Vec3
    var ra: Vec3
    var rb: Vec3
    var angular: Bool

    def __init__(out self, value: Float32, axis: Vec3, ra: Vec3, rb: Vec3, angular: Bool):
        self.value = value
        self.axis = axis
        self.ra = ra
        self.rb = rb
        self.angular = angular

    def mass(self, m: Motion) -> Float32:
        return m.angular_mass(self.axis) if self.angular else m.linear_mass(self.axis, self.ra, self.rb)

    def cdot(self, m: Motion) -> Float32:
        return m.angular_cdot(self.axis) if self.angular else m.linear_cdot(self.axis, self.ra, self.rb)

    def apply(self, mut m: Motion, impulse: Float32):
        if self.angular:
            m.angular(self.axis * impulse)
        else:
            m.linear(self.axis * impulse, self.ra, self.rb)


def solve_axis_rows(mut joint: Joint, row: AxisRow, mut m: Motion, soft: Softness, h: Float32, use_bias: Bool, rigid: Bool):
    """Spring, motor, limits (speculative, as Box2D v3), and for a rigid
    row the hold at `rest`: in that order, the stiffest last."""
    var k = row.mass(m)
    if k <= 0:
        return
    var inv_h = 1.0 / h
    if joint.enable_spring and joint.spring_hertz > 0:
        var spring = make_soft(joint.spring_hertz, joint.spring_damping, h)
        var c = row.value - joint.rest
        var impulse = -k * spring.mass_scale * (row.cdot(m) + c * spring.bias_rate) - spring.impulse_scale * joint.axial_impulse
        joint.axial_impulse += impulse
        row.apply(m, impulse)
    if joint.enable_motor:
        var impulse = -k * (row.cdot(m) - joint.motor_speed)
        var limit = joint.max_motor * h
        var total = max(-limit, min(limit, joint.motor_impulse + impulse))
        impulse = total - joint.motor_impulse
        joint.motor_impulse = total
        row.apply(m, impulse)
    if joint.enable_limit:
        # Lower: value - lower >= 0.
        var c = row.value - joint.lower
        var bias = Float32(0)
        var mass_scale = Float32(1)
        var impulse_scale = Float32(0)
        if c > 0:
            bias = c * inv_h
        elif use_bias:
            bias = c * soft.bias_rate
            mass_scale = soft.mass_scale
            impulse_scale = soft.impulse_scale
        var impulse = -k * mass_scale * (row.cdot(m) + bias) - impulse_scale * joint.lower_impulse
        var total = max(Float32(0), joint.lower_impulse + impulse)
        impulse = total - joint.lower_impulse
        joint.lower_impulse = total
        row.apply(m, impulse)
        # Upper: upper - value >= 0.
        c = joint.upper - row.value
        bias = 0
        mass_scale = 1
        impulse_scale = 0
        if c > 0:
            bias = c * inv_h
        elif use_bias:
            bias = c * soft.bias_rate
            mass_scale = soft.mass_scale
            impulse_scale = soft.impulse_scale
        impulse = -k * mass_scale * (-row.cdot(m) + bias) - impulse_scale * joint.upper_impulse
        total = max(Float32(0), joint.upper_impulse + impulse)
        impulse = total - joint.upper_impulse
        joint.upper_impulse = total
        row.apply(m, -impulse)
    if rigid:
        var terms = soft_terms(soft, use_bias)
        var c = row.value - joint.rest
        var impulse = -k * terms[1] * (row.cdot(m) + c * terms[0]) - terms[2] * joint.axial_impulse
        joint.axial_impulse += impulse
        row.apply(m, impulse)


def axial_total(joint: Joint) -> Float32:
    return joint.axial_impulse + joint.motor_impulse + joint.lower_impulse - joint.upper_impulse


def distance_row(joint: Joint, f: Frame) -> AxisRow:
    var d = f.anchor_b - f.anchor_a
    var length = sqrt(dot3(d, d))
    var u = d * (1.0 / length) if length > 1.0e-6 else Vec3(0, 1, 0)
    return AxisRow(length, u, f.ra, f.rb, False)


def prismatic_row(f: Frame) -> AxisRow:
    var d = f.anchor_b - f.anchor_a
    return AxisRow(dot3(f.axis, d), f.axis, f.ra + d, f.rb, False)


def swing_axis(joint: Joint, f: Frame) -> Tuple[Vec3, Float32]:
    """The axis B's twist axis swung about, away from A's, and how far."""
    var b1 = rotate_vec(f.turn_b, joint.axis_b)
    var c = cross3(f.axis, b1)
    var s = sqrt(dot3(c, c))
    var angle = det_atan2(s, dot3(f.axis, b1))
    return (unit_or(c, Vec3()), angle)


def joint_solve_2_12(mut joint: Joint, row: JointRow, mut bodies: SolverBodies, h: Float32, use_bias: Bool):
    var f = Frame(joint, bodies, row.a, row.b)
    var m = Motion(bodies, row.a, row.b)
    var kind = joint.kind
    var torque = Vec3()
    var force: Vec3
    if kind == JOINT_REVOLUTE:
        var angle = twist_angle(joint, f.turn_a, f.turn_b)
        solve_axis_rows(joint, AxisRow(angle, f.axis, Vec3(), Vec3(), True), m, row.angular, h, use_bias, False)
        solve_axis_lock(joint, f, m, row.angular, use_bias)
        solve_point(joint, f, m, row.linear, use_bias)
        torque = joint.angular_impulse + f.axis * axial_total(joint)
        force = joint.linear_impulse
    elif kind == JOINT_PRISMATIC:
        solve_axis_rows(joint, prismatic_row(f), m, row.linear, h, use_bias, False)
        solve_turn_lock(joint, f, m, row.angular, use_bias)
        solve_perp_lock(joint, f, m, row.linear, use_bias)
        torque = joint.angular_impulse
        force = joint.linear_impulse + f.axis * axial_total(joint)
    elif kind == JOINT_DISTANCE:
        var axis_row = distance_row(joint, f)
        var rigid = not joint.enable_limit and not (joint.enable_spring and joint.spring_hertz > 0)
        solve_axis_rows(joint, axis_row, m, row.linear, h, use_bias, rigid)
        force = axis_row.axis * axial_total(joint)
    else:
        # Swing-twist: the spring back to the made pose, the twist limits,
        # the swing cone, then the ball.
        if joint.enable_spring and joint.spring_hertz > 0:
            var spring = make_soft(joint.spring_hertz, joint.spring_damping, h)
            var target = joint.reference if joint.body_a == JOINT_WORLD else quat_product(f.turn_a, joint.reference)
            var k = m.ia.plus(m.ib).inverse()
            var c = rotation_error(f.turn_b, target)
            var impulse = k.apply((m.wb - m.wa) + c * spring.bias_rate) * (-spring.mass_scale) - joint.angular_impulse * spring.impulse_scale
            var total = joint.angular_impulse + impulse
            if joint.max_motor > 0:
                total = clamp_length(total, joint.max_motor * h)
            impulse = total - joint.angular_impulse
            joint.angular_impulse = total
            m.angular(impulse)
        if joint.enable_limit:
            var saved_spring = joint.enable_spring
            var saved_motor = joint.enable_motor
            joint.enable_spring = False
            joint.enable_motor = False
            var twist = twist_angle(joint, f.turn_a, f.turn_b)
            solve_axis_rows(joint, AxisRow(twist, f.axis, Vec3(), Vec3(), True), m, row.angular, h, use_bias, False)
            joint.enable_spring = saved_spring
            joint.enable_motor = saved_motor
        if joint.cone > 0:
            var swing = swing_axis(joint, f)
            var n = swing[0]
            if dot3(n, n) > 0:
                var k = m.angular_mass(n)
                var c = joint.cone - swing[1]
                var bias = Float32(0)
                var mass_scale = Float32(1)
                var impulse_scale = Float32(0)
                if c > 0:
                    bias = c / h
                elif use_bias:
                    bias = c * row.angular.bias_rate
                    mass_scale = row.angular.mass_scale
                    impulse_scale = row.angular.impulse_scale
                var impulse = -k * mass_scale * (-m.angular_cdot(n) + bias) - impulse_scale * joint.swing_impulse
                var total = max(Float32(0), joint.swing_impulse + impulse)
                impulse = total - joint.swing_impulse
                joint.swing_impulse = total
                m.angular(n * -impulse)
                torque = torque - n * joint.swing_impulse
        solve_point(joint, f, m, row.linear, use_bias)
        torque = torque + joint.angular_impulse + f.axis * (joint.lower_impulse - joint.upper_impulse)
        force = joint.linear_impulse
    if use_bias:
        if dot3(force, force) > dot3(joint.force_impulse, joint.force_impulse):
            joint.force_impulse = force
        if dot3(torque, torque) > dot3(joint.torque_impulse, joint.torque_impulse):
            joint.torque_impulse = torque
    m.store(bodies, row.a, row.b)


def joint_warm_2_12(mut joint: Joint, row: JointRow, mut bodies: SolverBodies):
    """Applies the stored row impulses along today's axes."""
    var f = Frame(joint, bodies, row.a, row.b)
    var m = Motion(bodies, row.a, row.b)
    var kind = joint.kind
    if kind == JOINT_REVOLUTE:
        m.linear(joint.linear_impulse, f.ra, f.rb)
        m.angular(joint.angular_impulse + f.axis * axial_total(joint))
    elif kind == JOINT_PRISMATIC:
        var r = prismatic_row(f)
        m.linear(joint.linear_impulse + f.axis * axial_total(joint), r.ra, f.rb)
        m.angular(joint.angular_impulse)
    elif kind == JOINT_DISTANCE:
        var r = distance_row(joint, f)
        m.linear(r.axis * axial_total(joint), f.ra, f.rb)
    else:
        m.linear(joint.linear_impulse, f.ra, f.rb)
        var swing = swing_axis(joint, f)
        m.angular(joint.angular_impulse + f.axis * (joint.lower_impulse - joint.upper_impulse) - swing[0] * joint.swing_impulse)
    m.store(bodies, row.a, row.b)


struct JointDef(Copyable, ImplicitlyCopyable, Movable):
    """What `PhysicsWorld.add_joint` makes: the kind, the two body rows
    (`slot_a` -1 for the world), a world anchor (and a second one on B for a
    distance joint; the same point otherwise), a world axis, and the
    limits, motor, spring and breaking. Angles in radians, lengths in metres;
    `length` < 0 takes the anchors' distance now."""
    var kind: Int
    var slot_a: Int
    var slot_b: Int
    var anchor: Vec3
    var anchor_b: Vec3
    var has_anchor_b: Bool
    var axis: Vec3
    var lower: Float32
    var upper: Float32
    var cone: Float32
    var length: Float32
    var rest: Float32
    var enable_limit: Bool
    var enable_motor: Bool
    var enable_spring: Bool
    var motor_speed: Float32
    var max_motor: Float32
    var spring_hertz: Float32
    var spring_damping: Float32
    var break_force: Float32
    var break_torque: Float32
    var hertz: Float32
    var damping: Float32

    def __init__(out self, kind: Int, slot_a: Int, slot_b: Int, anchor: Vec3, axis: Vec3 = Vec3(0, 1, 0)):
        self.kind = kind
        self.slot_a = slot_a
        self.slot_b = slot_b
        self.anchor = anchor
        self.anchor_b = anchor
        self.has_anchor_b = False
        self.axis = axis
        self.lower = 0
        self.upper = 0
        self.cone = 0
        self.length = -1
        self.rest = 0
        self.enable_limit = False
        self.enable_motor = False
        self.enable_spring = False
        self.motor_speed = 0
        self.max_motor = 0
        self.spring_hertz = 0
        self.spring_damping = 0
        self.break_force = 0
        self.break_torque = 0
        self.hertz = 0
        self.damping = 0

    def limits(mut self, lower: Float32, upper: Float32):
        self.lower = lower
        self.upper = upper
        self.enable_limit = True

    def motor(mut self, speed: Float32, max_motor: Float32):
        self.motor_speed = speed
        self.max_motor = max_motor
        self.enable_motor = True

    def spring(mut self, hertz: Float32, damping: Float32, rest: Float32 = 0):
        self.spring_hertz = hertz
        self.spring_damping = damping
        self.rest = rest
        self.enable_spring = True

    def second_anchor(mut self, at: Vec3):
        self.anchor_b = at
        self.has_anchor_b = True
