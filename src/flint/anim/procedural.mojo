"""Procedural passes after the clips (package 3.3), on a posed model in its
own space (+Z forward, +Y up): look-at, head stabilisation, breathing, lean,
hit recoil, orientation warping and motion warping.

Each works on joints through `NativeModel.rotate_node_model` (a model-space
turn about the joint's own pivot) or `offset_node`, and smooths what it
adds with springs (core/springs.mojo), so turning one on or off never pops.
Presentation only.

- Look-at: the head (and part of the spine) turn toward a point, clamped in
  yaw and pitch, weighted by a spring; the target's direction is measured
  from the body's forward, so the clip's own head motion stays on top.
- Head stabilisation: the head's rotation chases a smoothed copy of itself,
  so a gallop's bob reaches the head damped (birds and quadrupeds keep the
  head steady for the eyes).
- Breathing: a slow pitch of the spine and a counter-pitch of the head,
  faster and deeper after exertion.
- Lean: roll into turns from the heading's turn rate, pitch from forward
  acceleration, at the pelvis (foot IK puts the feet back).
- Hit recoil: an underdamped spring kicked away from the hit.
- Orientation warping: while the drawn heading lags the travel direction,
  the pelvis turns toward travel and the spine turns back, so the legs step
  where he goes.
- Motion warping: a clip's contact point (the hands at the contact time)
  is moved onto a target (the piece he tears up, a ledge's lip, a hug
  target) by an offset on the root that grows over the clip's approach.
"""
from std.math import sin, cos, atan2, sqrt, exp
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.springs import SpringFloat, SpringVec3, SpringAngle, wrap_degrees
from flint.core.ik import angle_axis, cross, dot, safe_normalized
from flint.assets.model import NativeModel
from flint.anim.pose import Pose

comptime DEG: Float32 = 0.01745329252


struct Placement(Copyable, ImplicitlyCopyable, Movable):
    """Where a model is drawn: its origin, scale and yaw (degrees)."""
    var position: Vec3
    var scale: Vec3
    var yaw: Float32

    def __init__(out self, position: Vec3 = Vec3(), scale: Vec3 = Vec3(1, 1, 1), yaw: Float32 = 0):
        self.position = position
        self.scale = scale
        self.yaw = yaw

    def to_world(self, local: Vec3) -> Vec3:
        return self.position + self.vector_to_world(local)

    def vector_to_world(self, local: Vec3) -> Vec3:
        var c = cos(self.yaw * DEG)
        var s = sin(self.yaw * DEG)
        return Vec3((local.x * c + local.z * s) * self.scale.x, local.y * self.scale.y, (-local.x * s + local.z * c) * self.scale.z)

    def to_model(self, point: Vec3) -> Vec3:
        return self.vector_to_model(point - self.position)

    def vector_to_model(self, vector: Vec3) -> Vec3:
        var c = cos(self.yaw * DEG)
        var s = sin(self.yaw * DEG)
        var x = vector.x / self.scale.x
        var y = vector.y / self.scale.y
        var z = vector.z / self.scale.z
        return Vec3(x * c - z * s, y, x * s + z * c)


def yaw_pitch(yaw_radians: Float32, pitch_radians: Float32) -> Quat:
    """Turn about +Y by yaw (+Z toward +X), then tip up by pitch (+Z toward +Y)."""
    var yaw = angle_axis(yaw_radians, Vec3(0, 1, 0))
    var pitch = angle_axis(-pitch_radians, Vec3(1, 0, 0))
    return yaw.multiply(pitch).normalized()


def rotation_between(a: Vec3, b: Vec3) -> Quat:
    """The shortest rotation taking direction a to direction b."""
    var u = safe_normalized(a, Vec3(0, 0, 1))
    var v = safe_normalized(b, Vec3(0, 0, 1))
    var c = dot(u, v)
    if c > 0.99999:
        return Quat()
    if c < -0.99999:
        var axis = cross(u, Vec3(1, 0, 0))
        if axis.length() < 0.001:
            axis = cross(u, Vec3(0, 1, 0))
        return angle_axis(3.14159265, safe_normalized(axis, Vec3(0, 1, 0)))
    var axis = cross(u, v)
    return Quat(axis.x, axis.y, axis.z, 1 + c).normalized()


def smooth01(x: Float32) -> Float32:
    var t = max(Float32(0), min(Float32(1), x))
    return t * t * (3 - 2 * t)


struct LookAt(Copyable, Movable):
    var weight: SpringFloat
    var yaw: SpringFloat
    var pitch: SpringFloat
    var applied_yaw: Float32    # degrees, after limits and weight (for tests and the inspector)
    var applied_pitch: Float32

    def __init__(out self):
        self.weight = SpringFloat(0)
        self.yaw = SpringFloat(0)
        self.pitch = SpringFloat(0)
        self.applied_yaw = 0
        self.applied_pitch = 0

    def reset(mut self):
        self = LookAt()

    def apply(
        mut self, mut model: NativeModel, head: Int, spine: Int, target: Vec3, has_target: Bool,
        max_yaw: Float32, max_pitch: Float32, spine_share: Float32, halflife: Float32, dt: Float32,
    ):
        """target in model space. max_yaw and max_pitch in degrees."""
        if head < 0:
            return
        var goal_yaw = Float32(0)
        var goal_pitch = Float32(0)
        var want = Float32(0)
        if has_target:
            var eye = model.global_position(head)
            var d = target - eye
            var flat = sqrt(d.x * d.x + d.z * d.z)
            if flat > 0.01 or abs(d.y) > 0.01:
                var yaw = atan2(d.x, d.z) / DEG
                # Past the limit (behind him) he does not wrench round; he lets go.
                if abs(yaw) <= max_yaw * 1.5:
                    goal_yaw = max(-max_yaw, min(max_yaw, yaw))
                    goal_pitch = max(-max_pitch, min(max_pitch, atan2(d.y, max(Float32(0.01), flat)) / DEG))
                    want = 1
        self.weight.update(want, halflife, dt)
        self.yaw.update(goal_yaw, halflife, dt)
        self.pitch.update(goal_pitch, halflife, dt)
        var w = max(Float32(0), min(Float32(1), self.weight.value))
        self.applied_yaw = self.yaw.value * w
        self.applied_pitch = self.pitch.value * w
        if w < 0.001:
            return
        var full = yaw_pitch(self.applied_yaw * DEG, self.applied_pitch * DEG)
        var part = Quat().slerp(full, max(Float32(0), min(Float32(1), spine_share)))
        if spine >= 0 and spine_share > 0:
            model.rotate_node_model(spine, part)
            model.rotate_node_model(head, full.multiply(part.conjugate()).normalized())
        else:
            model.rotate_node_model(head, full)


struct HeadStabilizer(Copyable, Movable):
    var smoothed: Quat
    var ready: Bool

    def __init__(out self):
        self.smoothed = Quat()
        self.ready = False

    def reset(mut self):
        self.ready = False

    def apply(mut self, mut model: NativeModel, head: Int, strength: Float32, halflife: Float32, dt: Float32):
        if head < 0:
            return
        var current = model.global_rotation(head)
        if not self.ready or dt <= 0:
            self.smoothed = current
            self.ready = True
            return
        var follow = 1.0 - exp(-0.69314718 * dt / max(Float32(0.001), halflife))
        self.smoothed = self.smoothed.slerp(current, follow).normalized()
        var k = max(Float32(0), min(Float32(1), strength))
        if k <= 0:
            return
        var target = current.slerp(self.smoothed, k)
        model.rotate_node_model(head, target.multiply(current.conjugate()).normalized())


struct Breathing(Copyable, Movable):
    var phase: Float32
    var exertion: SpringFloat
    var applied: Float32  # degrees of spine pitch this frame

    def __init__(out self):
        self.phase = 0
        self.exertion = SpringFloat(0)
        self.applied = 0

    def apply(
        mut self, mut model: NativeModel, spine: Int, head: Int, effort: Float32,
        rest_rate: Float32, hard_rate: Float32, depth: Float32, dt: Float32,
    ):
        """effort in [0, 1] now (running, fighting); breathing follows it
        slowly. Rates in breaths per second, depth in degrees."""
        self.exertion.update(max(Float32(0), min(Float32(1), effort)), 1.6, dt)
        var e = max(Float32(0), min(Float32(1), self.exertion.value))
        var rate = rest_rate + (hard_rate - rest_rate) * e
        self.phase += dt * rate
        self.phase -= Float32(Int(self.phase))
        self.applied = sin(self.phase * 6.2831853) * depth * (1.0 + e)
        if spine >= 0:
            model.rotate_node_model(spine, angle_axis(-self.applied * DEG, Vec3(1, 0, 0)))
        if head >= 0:
            model.rotate_node_model(head, angle_axis(self.applied * 0.6 * DEG, Vec3(1, 0, 0)))


struct Lean(Copyable, Movable):
    var roll: SpringFloat
    var pitch: SpringFloat
    var last_yaw: Float32
    var last_speed: Float32
    var ready: Bool

    def __init__(out self):
        self.roll = SpringFloat(0)
        self.pitch = SpringFloat(0)
        self.last_yaw = 0
        self.last_speed = 0
        self.ready = False

    def reset(mut self):
        self = Lean()

    def apply(
        mut self, mut model: NativeModel, pelvis: Int, yaw_degrees: Float32, speed: Float32,
        roll_per_turn: Float32, pitch_per_accel: Float32, max_degrees: Float32, dt: Float32,
    ):
        """Turn rate from the drawn heading (degrees a second); lean into
        it (roll_per_turn degrees of roll per 100 degrees a second, more at
        speed) and pitch with forward acceleration."""
        if dt <= 0:
            return
        if not self.ready:
            self.last_yaw = yaw_degrees
            self.last_speed = speed
            self.ready = True
        var turn = wrap_degrees(yaw_degrees - self.last_yaw) / dt
        var accel = (speed - self.last_speed) / dt
        self.last_yaw = yaw_degrees
        self.last_speed = speed
        var speed_share = min(Float32(1), speed / 6.0)
        var roll_goal = max(-max_degrees, min(max_degrees, -turn / 100.0 * roll_per_turn * speed_share))
        var pitch_goal = max(-max_degrees, min(max_degrees, accel * pitch_per_accel))
        self.roll.update(roll_goal, 0.12, dt)
        self.pitch.update(pitch_goal, 0.15, dt)
        if pelvis < 0:
            return
        var q = angle_axis(self.roll.value * DEG, Vec3(0, 0, 1)).multiply(angle_axis(self.pitch.value * DEG, Vec3(1, 0, 0)))
        model.rotate_node_model(pelvis, q.normalized())


struct Recoil(Copyable, Movable):
    """An underdamped angular spring (scaled-angle, model space)."""
    var offset: Vec3
    var velocity: Vec3

    def __init__(out self):
        self.offset = Vec3()
        self.velocity = Vec3()

    def reset(mut self):
        self = Recoil()

    def kick(mut self, away_model: Vec3, strength: Float32):
        """A hit pushing toward away_model (model space, horizontal): the
        upper body tips that way."""
        var flat = Vec3(away_model.x, 0, away_model.z)
        var direction = safe_normalized(flat, Vec3(0, 0, -1))
        # Tipping toward d is a turn about up x d.
        var axis = cross(Vec3(0, 1, 0), direction)
        self.velocity = self.velocity + axis * strength

    def apply(mut self, mut model: NativeModel, spine: Int, head: Int, stiffness: Float32, damping: Float32, dt: Float32):
        if dt <= 0:
            return
        # Semi-implicit Euler in a few sub-steps, stable for these constants.
        var steps = max(1, min(8, Int(dt / 0.004) + 1))
        var h = dt / Float32(steps)
        for _ in range(steps):
            self.velocity = self.velocity + (self.offset * (-stiffness) - self.velocity * damping) * h
            self.offset = self.offset + self.velocity * h
        if self.offset.length() < 0.00005 and self.velocity.length() < 0.0005:
            self.offset = Vec3()
            self.velocity = Vec3()
            return
        if spine >= 0:
            model.rotate_node_model(spine, Quat.from_scaled_angle(self.offset * 0.65))
        if head >= 0:
            model.rotate_node_model(head, Quat.from_scaled_angle(self.offset * 0.35))


struct OrientationWarp(Copyable, Movable):
    var angle: SpringFloat
    var applied: Float32

    def __init__(out self):
        self.angle = SpringFloat(0)
        self.applied = 0

    def reset(mut self):
        self = OrientationWarp()

    def apply(
        mut self, mut model: NativeModel, pelvis: Int, spine: Int, drawn_yaw: Float32, travel_yaw: Float32,
        moving: Bool, max_degrees: Float32, spine_back: Float32, dt: Float32,
    ):
        """Turns the pelvis (legs) toward travel by the heading's lag and
        the spine back by spine_back of it (the chest stays on the heading)."""
        var goal = Float32(0)
        if moving:
            goal = max(-max_degrees, min(max_degrees, wrap_degrees(travel_yaw - drawn_yaw)))
        self.angle.update(goal, 0.06, dt)
        self.applied = self.angle.value
        if abs(self.applied) < 0.01 or pelvis < 0:
            return
        model.rotate_node_model(pelvis, angle_axis(self.applied * DEG, Vec3(0, 1, 0)))
        if spine >= 0:
            model.rotate_node_model(spine, angle_axis(-self.applied * spine_back * DEG, Vec3(0, 1, 0)))


def pose_parents(model: NativeModel) -> List[Int]:
    var parents = List[Int](length=len(model.nodes), fill=-1)
    for index in range(len(model.nodes)):
        for child in model.nodes[index].children:
            if child >= 0 and child < len(parents):
                parents[child] = index
    return parents^


def pose_position(pose: Pose, parents: List[Int], node: Int) -> Vec3:
    """A node's model-space position in a pose (bones unscaled)."""
    var position = Vec3()
    var current = node
    var guard = 0
    while current >= 0 and current < pose.count() and guard < 128:
        position = pose.t[current] + pose.r[current].rotate(position)
        current = parents[current]
        guard += 1
    return position


struct MotionWarp(Copyable, Movable):
    """Moves a clip's contact onto a target over its approach.

    `begin` takes the authored contact point (model space, from the clip
    sampled at the contact time) and the target; while the clip runs,
    `apply` offsets the root by (target - contact) x smoothstep(time /
    contact time), at most max_distance. When the state ends the offset
    springs back to nothing."""
    var active: Bool
    var kind: Int
    var goal: Vec3
    var contact_time: Float32
    var shown: SpringVec3
    var progress: Float32

    def __init__(out self):
        self.active = False
        self.kind = 0
        self.goal = Vec3()
        self.contact_time = 0
        self.shown = SpringVec3()
        self.progress = 0

    def reset(mut self):
        self = MotionWarp()

    def begin(mut self, kind: Int, contact: Vec3, target: Vec3, contact_time: Float32, max_distance: Float32):
        var offset = target - contact
        var length = offset.length()
        if length > max_distance and length > 0:
            offset = offset * (max_distance / length)
        self.active = True
        self.kind = kind
        self.goal = offset
        self.contact_time = max(Float32(0.01), contact_time)
        self.progress = 0

    def retarget(mut self, contact: Vec3, target: Vec3, max_distance: Float32):
        """A moving target (a ledge he slides along): the goal follows."""
        var offset = target - contact
        var length = offset.length()
        if length > max_distance and length > 0:
            offset = offset * (max_distance / length)
        self.goal = offset

    def end(mut self):
        self.active = False

    def apply(mut self, mut model: NativeModel, root: Int, clip_time: Float32, dt: Float32, ease: Float32 = 0):
        """ease: 0 follows the warp exactly while it runs (a clip with an
        approach); a halflife eases into it instead (a held pose, such as
        hanging from a ledge, where the clip's time does not move)."""
        var want = Vec3()
        if self.active:
            self.progress = smooth01(clip_time / self.contact_time)
            want = self.goal * self.progress
            if ease > 0:
                self.shown.update(want, ease, dt)
            else:
                # While warping the offset follows exactly; the spring only
                # carries it back afterwards.
                self.shown.value = want
                self.shown.velocity = Vec3()
        else:
            self.progress = 0
            self.shown.update(want, 0.09, dt)
        if root >= 0 and self.shown.value.length() > 0.0001:
            model.offset_node(root, self.shown.value)
