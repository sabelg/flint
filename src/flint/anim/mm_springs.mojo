"""Springs, quaternion helpers and spring inertialization for motion matching
(`flint.anim.mm_springs`).

A Mojo port of the spring and inertialization code from Daniel Holden's Motion-Matching
reference (https://github.com/orangeduck/Motion-Matching, MIT, see `licenses/README.md`),
following "Learned Motion Matching" (Holden, Kanoun, Perepichka, Popa; SIGGRAPH 2020) and
"Spring-It-On" (theorangeduck.com):

* `mm_simple_spring_*`: critically damped spring toward a goal (value + velocity).
* `mm_decay_spring_*`: the same spring toward zero, which is what an inertializer offset does.
* `mm_inertialize_transition_*` / `mm_inertialize_update_*`: on a switch the offset absorbs the
  difference between the pose being left and the pose being entered (positions and
  velocities), then decays, so the output stays continuous in value and velocity.
* `mm_spring_character_update`: the velocity spring that moves the simulation object
  ("Spring-It-On: controllers"), integrated exactly, so its trajectory can be predicted ahead.
* `mm_damp_adjustment_*`: exact damped fraction of a correction for one step.

Holden's damping is `4 ln2 / halflife`. Exponentials use `det_exp` and trigonometry the
`flint.core.detmath` routines, so identical inputs give bit-identical results.
"""
from std.math import sqrt
from flint.core.detmath import det_exp, det_atan2
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.anim.pose import det_quat_to_scaled_angle, det_quat_from_scaled_angle

comptime MM_LN2: Float32 = 0.69314718056
comptime MM_EPS: Float32 = 1.0e-5


# ----------------------------------------------------------------------------- quaternions

@always_inline
def mm_quat_abs(q: Quat) -> Quat:
    """The same rotation with w >= 0 (shortest path)."""
    return q.shortest()


@always_inline
def mm_quat_inv(q: Quat) -> Quat:
    return q.conjugate()


@always_inline
def mm_quat_mul(a: Quat, b: Quat) -> Quat:
    return a.multiply(b)


@always_inline
def mm_quat_inv_mul(a: Quat, b: Quat) -> Quat:
    """inv(a) * b."""
    return a.conjugate().multiply(b)


@always_inline
def mm_quat_mul_inv(a: Quat, b: Quat) -> Quat:
    """a * inv(b)."""
    return a.multiply(b.conjugate())


@always_inline
def mm_quat_mul_vec(q: Quat, v: Vec3) -> Vec3:
    return q.rotate(v)


@always_inline
def mm_quat_inv_mul_vec(q: Quat, v: Vec3) -> Vec3:
    return q.conjugate().rotate(v)


@always_inline
def mm_quat_to_scaled_angle_axis(q: Quat) -> Vec3:
    return det_quat_to_scaled_angle(q)


@always_inline
def mm_quat_from_scaled_angle_axis(v: Vec3) -> Quat:
    return det_quat_from_scaled_angle(v)


def mm_quat_from_xform_xy(x: Vec3, y: Vec3) -> Quat:
    """Rotation whose matrix has first column `x` and (re-orthogonalised) second column `y`
    (the two-axis rotation encoding the decompressor predicts)."""
    var c2 = x.cross(y)
    var l2 = c2.length()
    c2 = c2 * (Float32(1.0) / l2) if l2 > Float32(1.0e-12) else Vec3(0, 0, 1)
    var c1 = c2.cross(x)
    var l1 = c1.length()
    c1 = c1 * (Float32(1.0) / l1) if l1 > Float32(1.0e-12) else Vec3(0, 1, 0)
    var lx = x.length()
    var c0 = x * (Float32(1.0) / lx) if lx > Float32(1.0e-12) else Vec3(1, 0, 0)
    return mm_quat_from_matrix(c0, c1, c2)


def mm_quat_from_matrix(c0: Vec3, c1: Vec3, c2: Vec3) -> Quat:
    """Quaternion from a rotation matrix given by its columns (as `quat_from_xform`)."""
    var m00 = c0.x
    var m10 = c0.y
    var m20 = c0.z
    var m01 = c1.x
    var m11 = c1.y
    var m21 = c1.z
    var m02 = c2.x
    var m12 = c2.y
    var m22 = c2.z
    var q: Quat
    if m22 < 0:
        if m00 > m11:
            q = Quat(1 + m00 - m11 - m22, m01 + m10, m20 + m02, m21 - m12)
        else:
            q = Quat(m01 + m10, 1 - m00 + m11 - m22, m12 + m21, m02 - m20)
    else:
        if m00 < -m11:
            q = Quat(m20 + m02, m12 + m21, 1 - m00 - m11 + m22, m10 - m01)
        else:
            q = Quat(m21 - m12, m02 - m20, m10 - m01, 1 + m00 + m11 + m22)
    return q.normalized()


def mm_quat_to_xform_xy(q: Quat) -> Tuple[Vec3, Vec3]:
    """First two matrix columns of `q`."""
    var x2 = q.x + q.x
    var y2 = q.y + q.y
    var z2 = q.z + q.z
    var xx = q.x * x2
    var yy = q.y * y2
    var wx = q.w * x2
    var xy = q.x * y2
    var yz = q.y * z2
    var wy = q.w * y2
    var xz = q.x * z2
    var zz = q.z * z2
    var wz = q.w * z2
    return (Vec3(1 - (yy + zz), xy + wz, xz - wy), Vec3(xy - wz, 1 - (xx + zz), yz + wx))


@always_inline
def mm_yaw_quat(angle: Float32) -> Quat:
    return Quat.from_axis_angle(Vec3(0, 1, 0), angle)


def mm_quat_yaw(q: Quat) -> Float32:
    """Heading (radians) of `q`'s forward (+Z) axis around +Y."""
    var f = q.rotate(Vec3(0, 0, 1))
    if f.x * f.x + f.z * f.z < Float32(1.0e-12):
        return 0
    return det_atan2(f.x, f.z)


def mm_quat_between(a: Vec3, b: Vec3) -> Quat:
    """Shortest rotation taking direction `a` to `b` (as the reference `quat_between`)."""
    var c = a.cross(b)
    return Quat(c.x, c.y, c.z, sqrt(a.dot(a) * b.dot(b)) + a.dot(b)).normalized()


# ----------------------------------------------------------------------------- springs

@always_inline
def mm_halflife_to_damping(halflife: Float32) -> Float32:
    return (Float32(4.0) * MM_LN2) / (halflife + MM_EPS)


@always_inline
def mm_negexp(x: Float32) -> Float32:
    return det_exp(-x)


@always_inline
def mm_damp_fraction(halflife: Float32, dt: Float32) -> Float32:
    """Fraction of the way to the goal an exact damper covers in `dt`."""
    return Float32(1.0) - mm_negexp((MM_LN2 * dt) / (halflife + MM_EPS))


def mm_damp_adjustment_vec(g: Vec3, halflife: Float32, dt: Float32) -> Vec3:
    return g * mm_damp_fraction(halflife, dt)


def mm_damp_adjustment_quat(g: Quat, halflife: Float32, dt: Float32) -> Quat:
    return Quat().nlerp(g, mm_damp_fraction(halflife, dt))


def mm_simple_spring_scalar(mut x: Float32, mut v: Float32, goal: Float32, halflife: Float32, dt: Float32):
    var y = mm_halflife_to_damping(halflife) / 2
    var j0 = x - goal
    var j1 = v + j0 * y
    var eydt = mm_negexp(y * dt)
    x = eydt * (j0 + j1 * dt) + goal
    v = eydt * (v - j1 * y * dt)


def mm_simple_spring_vec(mut x: Vec3, mut v: Vec3, goal: Vec3, halflife: Float32, dt: Float32):
    var y = mm_halflife_to_damping(halflife) / 2
    var j0 = x - goal
    var j1 = v + j0 * y
    var eydt = mm_negexp(y * dt)
    x = (j0 + j1 * dt) * eydt + goal
    v = (v - j1 * (y * dt)) * eydt


def mm_simple_spring_quat(mut x: Quat, mut v: Vec3, goal: Quat, halflife: Float32, dt: Float32):
    var y = mm_halflife_to_damping(halflife) / 2
    var j0 = mm_quat_to_scaled_angle_axis(mm_quat_abs(mm_quat_mul_inv(x, goal)))
    var j1 = v + j0 * y
    var eydt = mm_negexp(y * dt)
    x = mm_quat_mul(mm_quat_from_scaled_angle_axis((j0 + j1 * dt) * eydt), goal).normalized()
    v = (v - j1 * (y * dt)) * eydt


def mm_decay_spring_vec(mut x: Vec3, mut v: Vec3, halflife: Float32, dt: Float32):
    var y = mm_halflife_to_damping(halflife) / 2
    var j1 = v + x * y
    var eydt = mm_negexp(y * dt)
    x = (x + j1 * dt) * eydt
    v = (v - j1 * (y * dt)) * eydt


def mm_decay_spring_quat(mut x: Quat, mut v: Vec3, halflife: Float32, dt: Float32):
    var y = mm_halflife_to_damping(halflife) / 2
    var j0 = mm_quat_to_scaled_angle_axis(x)
    var j1 = v + j0 * y
    var eydt = mm_negexp(y * dt)
    x = mm_quat_from_scaled_angle_axis((j0 + j1 * dt) * eydt)
    v = (v - j1 * (y * dt)) * eydt


def mm_spring_character_update(
    mut position: Vec3, mut velocity: Vec3, mut acceleration: Vec3, desired_velocity: Vec3, halflife: Float32, dt: Float32
):
    """Exact integration of a critically damped velocity spring (position, velocity and
    acceleration of the simulation object toward `desired_velocity`)."""
    var y = mm_halflife_to_damping(halflife) / 2
    var j0 = velocity - desired_velocity
    var j1 = acceleration + j0 * y
    var eydt = mm_negexp(y * dt)
    var position_prev = position
    var yy = y * y
    position = ((-j1) * (Float32(1.0) / yy) + (-j0 - j1 * dt) * (Float32(1.0) / y)) * eydt + j1 * (Float32(1.0) / yy) + j0 * (Float32(1.0) / y) + desired_velocity * dt + position_prev
    velocity = (j0 + j1 * dt) * eydt + desired_velocity
    acceleration = (acceleration - j1 * (y * dt)) * eydt


# ----------------------------------------------------------------------------- inertialization

def mm_inertialize_transition_vec(mut off_x: Vec3, mut off_v: Vec3, src_x: Vec3, src_v: Vec3, dst_x: Vec3, dst_v: Vec3):
    off_x = (src_x + off_x) - dst_x
    off_v = (src_v + off_v) - dst_v


def mm_inertialize_update_vec(
    mut out_x: Vec3, mut out_v: Vec3, mut off_x: Vec3, mut off_v: Vec3, in_x: Vec3, in_v: Vec3, halflife: Float32, dt: Float32
):
    mm_decay_spring_vec(off_x, off_v, halflife, dt)
    out_x = in_x + off_x
    out_v = in_v + off_v


def mm_inertialize_transition_quat(mut off_x: Quat, mut off_v: Vec3, src_x: Quat, src_v: Vec3, dst_x: Quat, dst_v: Vec3):
    off_x = mm_quat_abs(mm_quat_mul(mm_quat_mul(off_x, src_x), mm_quat_inv(dst_x))).normalized()
    off_v = (off_v + src_v) - dst_v


def mm_inertialize_update_quat(
    mut out_x: Quat, mut out_v: Vec3, mut off_x: Quat, mut off_v: Vec3, in_x: Quat, in_v: Vec3, halflife: Float32, dt: Float32
):
    mm_decay_spring_quat(off_x, off_v, halflife, dt)
    out_x = mm_quat_mul(off_x, in_x).normalized()
    out_v = off_v + mm_quat_mul_vec(off_x, in_v)


struct MmPose(Copyable, Movable):
    """A full pose in the motion-matching layout: bone 0 is the simulation (root) bone in
    world/animation space, every other bone is local to its parent. Velocities included."""
    var pos: List[Vec3]
    var vel: List[Vec3]
    var rot: List[Quat]
    var ang: List[Vec3]

    def __init__(out self, n: Int = 0):
        self.pos = List[Vec3](length=n, fill=Vec3())
        self.vel = List[Vec3](length=n, fill=Vec3())
        self.rot = List[Quat](length=n, fill=Quat())
        self.ang = List[Vec3](length=n, fill=Vec3())

    def count(self) -> Int:
        return len(self.pos)


struct MmInertializer(Copyable, Movable):
    """Offset-based spring inertializer for a full `MmPose` (Holden 2020, section 4 and the
    reference controller): the root is carried across the switch in world space through the
    transition source/destination frames, every other bone through a local offset."""
    var off_pos: List[Vec3]
    var off_vel: List[Vec3]
    var off_rot: List[Quat]
    var off_ang: List[Vec3]
    var src_pos: Vec3
    var src_rot: Quat
    var dst_pos: Vec3
    var dst_rot: Quat

    def __init__(out self, n: Int = 0):
        self.off_pos = List[Vec3](length=n, fill=Vec3())
        self.off_vel = List[Vec3](length=n, fill=Vec3())
        self.off_rot = List[Quat](length=n, fill=Quat())
        self.off_ang = List[Vec3](length=n, fill=Vec3())
        self.src_pos = Vec3()
        self.src_rot = Quat()
        self.dst_pos = Vec3()
        self.dst_rot = Quat()

    def reset(mut self, root_position: Vec3, root_rotation: Quat):
        """No offsets; the animation's root frame is placed at `root_position/rotation`."""
        for i in range(len(self.off_pos)):
            self.off_pos[i] = Vec3()
            self.off_vel[i] = Vec3()
            self.off_rot[i] = Quat()
            self.off_ang[i] = Vec3()
        self.src_pos = root_position
        self.src_rot = root_rotation
        self.dst_pos = Vec3()
        self.dst_rot = Quat()

    def largest_offset(self) -> Float32:
        """Largest local position offset (metres) over the non-root bones."""
        var m: Float32 = 0
        for i in range(1, len(self.off_pos)):
            m = max(m, self.off_pos[i].length())
        return m

    def transition(mut self, out_pose: MmPose, src: MmPose, dst: MmPose):
        """Switch from animation pose `src` to `dst` (both in their own animation spaces) while
        `out_pose` is what is currently shown."""
        self.dst_pos = out_pose.pos[0]
        self.dst_rot = out_pose.rot[0]
        self.src_pos = dst.pos[0]
        self.src_rot = dst.rot[0]
        var ws_vel = mm_quat_mul_vec(self.dst_rot, mm_quat_inv_mul_vec(self.src_rot, dst.vel[0]))
        var ws_ang = mm_quat_mul_vec(self.dst_rot, mm_quat_inv_mul_vec(self.src_rot, dst.ang[0]))
        mm_inertialize_transition_vec(self.off_pos[0], self.off_vel[0], out_pose.pos[0], out_pose.vel[0], out_pose.pos[0], ws_vel)
        mm_inertialize_transition_quat(self.off_rot[0], self.off_ang[0], out_pose.rot[0], out_pose.ang[0], out_pose.rot[0], ws_ang)
        for i in range(1, len(self.off_pos)):
            mm_inertialize_transition_vec(self.off_pos[i], self.off_vel[i], src.pos[i], src.vel[i], dst.pos[i], dst.vel[i])
            mm_inertialize_transition_quat(self.off_rot[i], self.off_ang[i], src.rot[i], src.ang[i], dst.rot[i], dst.ang[i])

    def update(mut self, mut out_pose: MmPose, input: MmPose, halflife: Float32, dt: Float32):
        """Advance the offsets by `dt` and write `input` (in its animation space) plus the
        offsets into `out_pose` (root in world space)."""
        var wp = mm_quat_mul_vec(self.dst_rot, mm_quat_inv_mul_vec(self.src_rot, input.pos[0] - self.src_pos)) + self.dst_pos
        var wv = mm_quat_mul_vec(self.dst_rot, mm_quat_inv_mul_vec(self.src_rot, input.vel[0]))
        var wr = mm_quat_mul(self.dst_rot, mm_quat_inv_mul(self.src_rot, input.rot[0])).normalized()
        var wa = mm_quat_mul_vec(self.dst_rot, mm_quat_inv_mul_vec(self.src_rot, input.ang[0]))
        mm_inertialize_update_vec(out_pose.pos[0], out_pose.vel[0], self.off_pos[0], self.off_vel[0], wp, wv, halflife, dt)
        mm_inertialize_update_quat(out_pose.rot[0], out_pose.ang[0], self.off_rot[0], self.off_ang[0], wr, wa, halflife, dt)
        for i in range(1, len(self.off_pos)):
            mm_inertialize_update_vec(out_pose.pos[i], out_pose.vel[i], self.off_pos[i], self.off_vel[i], input.pos[i], input.vel[i], halflife, dt)
            mm_inertialize_update_quat(out_pose.rot[i], out_pose.ang[i], self.off_rot[i], self.off_ang[i], input.rot[i], input.ang[i], halflife, dt)

    def root_adjust(mut self, mut position: Vec3, mut rotation: Quat, input_position: Vec3, input_rotation: Quat):
        """Move the shown root to `input_position/rotation` and shift the transition frames
        with it, so the next update continues from there (simulation-bone adjustment)."""
        var position_difference = input_position - position
        position = position_difference + position
        self.dst_pos = position_difference + self.dst_pos
        self.src_pos = self.src_pos + mm_quat_mul_vec(self.src_rot, mm_quat_inv_mul_vec(self.dst_rot, position - self.off_pos[0] - self.dst_pos))
        self.dst_pos = position
        self.off_pos[0] = Vec3()
        var rotation_difference = mm_quat_mul_inv(input_rotation, rotation).normalized()
        rotation = mm_quat_mul(rotation_difference, rotation).normalized()
        self.dst_rot = mm_quat_mul(rotation_difference, self.dst_rot).normalized()
