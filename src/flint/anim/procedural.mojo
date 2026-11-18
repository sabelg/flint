"""Procedural passes after the clips (package 3.3 / v4), on a posed model or
`SkeletonInstance` (+Z forward, +Y up): look-at, head stabilisation, breathing,
lean, analytical hit recoil, orientation warping, motion warping, and stateful
terrain/deck `FootPlantSolver`.

Upgrades in Flint v4:
- 100% deterministic `flint.core.detmath` (`det_sin`, `det_cos`, `det_atan2`, `det_exp`, `det_acos`).
- `LookAt` transforms targets into the chest/head animated parent space so rotated,
  pitched, or rolled characters aim accurately.
- `Recoil` replaces multi-step Euler integration with the exact closed-form
  underdamped/critically-damped/overdamped analytical spring.
- `FootPlantSolver` provides stateful multi-leg world stance-phase foot locking
  (zero skate during `[plant, lift]`, critically damped inertial release during
  swing), surface-normal ankle pitch/roll alignment (clamped to `max_tilt_radians`),
  and critically damped `SpringFloat` pelvis lowering.
"""
from std.math import sqrt
from flint.core.detmath import det_sin, det_cos, det_atan2, det_exp, det_acos, PI
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.springs import SpringFloat, SpringVec3, wrap_degrees, decay_spring
from flint.core.ik import angle_axis, cross, dot, safe_normalized, solve_two_bone_pole
from flint.assets.model import NativeModel
from flint.anim.pose import Pose, SkeletonInstance, det_quat_from_scaled_angle

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
        var c = det_cos(self.yaw * DEG)
        var s = det_sin(self.yaw * DEG)
        return Vec3((local.x * c + local.z * s) * self.scale.x, local.y * self.scale.y, (-local.x * s + local.z * c) * self.scale.z)

    def to_model(self, point: Vec3) -> Vec3:
        return self.vector_to_model(point - self.position)

    def vector_to_model(self, vector: Vec3) -> Vec3:
        var c = det_cos(self.yaw * DEG)
        var s = det_sin(self.yaw * DEG)
        var x = vector.x / self.scale.x
        var y = vector.y / self.scale.y
        var z = vector.z / self.scale.z
        return Vec3(x * c - z * s, y, x * s + z * c)

    def world_yaw_quat(self) -> Quat:
        """World rotation of `Placement.yaw` matching `vector_to_world` (`+yaw` about +Y)."""
        return angle_axis(self.yaw * DEG, Vec3(0, 1, 0))


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
    if c > Float32(0.99999):
        return Quat()
    if c < Float32(-0.99999):
        var axis = cross(u, Vec3(1, 0, 0))
        if axis.length() < Float32(0.001):
            axis = cross(u, Vec3(0, 1, 0))
        return angle_axis(PI, safe_normalized(axis, Vec3(0, 1, 0)))
    var axis = cross(u, v)
    return Quat(axis.x, axis.y, axis.z, Float32(1.0) + c).normalized()


def smooth01(x: Float32) -> Float32:
    var t = max(Float32(0), min(Float32(1), x))
    return t * t * (Float32(3.0) - Float32(2.0) * t)


struct LookAt(Copyable, Movable):
    var weight: SpringFloat
    var yaw: SpringFloat
    var pitch: SpringFloat
    var applied_yaw: Float32    # degrees, after limits and weight
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
        """`target` in model space. Transforms `target - eye` into the chest/head
        parent space so rotated/turned characters aim accurately."""
        if head < 0 or head >= len(model.nodes):
            return
        model.ensure_parents()
        var ref_parent = -1
        if spine >= 0 and spine < len(model.parents) and model.parents[spine] >= 0:
            ref_parent = model.parents[spine]
        elif head < len(model.parents) and model.parents[head] >= 0:
            ref_parent = model.parents[head]
        var ref_rot = model.global_rotation(ref_parent) if ref_parent >= 0 else Quat()

        var goal_yaw = Float32(0)
        var goal_pitch = Float32(0)
        var want = Float32(0)
        if has_target:
            var eye = model.global_position(head)
            var d = ref_rot.conjugate().rotate(target - eye)
            var flat = sqrt(d.x * d.x + d.z * d.z)
            if flat > Float32(0.01) or abs(d.y) > Float32(0.01):
                var yaw = det_atan2(d.x, d.z) / DEG
                if abs(yaw) <= max_yaw * Float32(1.5):
                    goal_yaw = max(-max_yaw, min(max_yaw, yaw))
                    goal_pitch = max(-max_pitch, min(max_pitch, det_atan2(d.y, max(Float32(0.01), flat)) / DEG))
                    want = Float32(1.0)
        self.weight.update(want, halflife, dt)
        self.yaw.update(goal_yaw, halflife, dt)
        self.pitch.update(goal_pitch, halflife, dt)
        var w = max(Float32(0), min(Float32(1), self.weight.value))
        self.applied_yaw = self.yaw.value * w
        self.applied_pitch = self.pitch.value * w
        if w < Float32(0.001):
            return
        var full_local = yaw_pitch(self.applied_yaw * DEG, self.applied_pitch * DEG)
        var full = ref_rot.multiply(full_local).multiply(ref_rot.conjugate()).normalized()
        var part = Quat().slerp(full, max(Float32(0), min(Float32(1), spine_share)))
        if spine >= 0 and spine_share > 0:
            model.rotate_node_model(spine, part)
            model.rotate_node_model(head, full.multiply(part.conjugate()).normalized())
        else:
            model.rotate_node_model(head, full)

    def apply_to_skeleton(
        mut self, mut skel: SkeletonInstance, model: NativeModel, head: Int, spine: Int, target: Vec3,
        has_target: Bool, max_yaw: Float32, max_pitch: Float32, spine_share: Float32, halflife: Float32, dt: Float32,
    ):
        """Non-mutating `SkeletonInstance` overload for `LookAt`."""
        skel.ensure_topology(model)
        if head < 0 or head >= skel.node_count:
            return
        var ref_parent = -1
        if spine >= 0 and spine < len(skel.parents) and skel.parents[spine] >= 0:
            ref_parent = skel.parents[spine]
        elif head < len(skel.parents) and skel.parents[head] >= 0:
            ref_parent = skel.parents[head]
        var ref_rot = skel.global_rotation(ref_parent) if ref_parent >= 0 else Quat()

        var goal_yaw = Float32(0)
        var goal_pitch = Float32(0)
        var want = Float32(0)
        if has_target:
            var eye = skel.global_position(head)
            var d = ref_rot.conjugate().rotate(target - eye)
            var flat = sqrt(d.x * d.x + d.z * d.z)
            if flat > Float32(0.01) or abs(d.y) > Float32(0.01):
                var yaw = det_atan2(d.x, d.z) / DEG
                if abs(yaw) <= max_yaw * Float32(1.5):
                    goal_yaw = max(-max_yaw, min(max_yaw, yaw))
                    goal_pitch = max(-max_pitch, min(max_pitch, det_atan2(d.y, max(Float32(0.01), flat)) / DEG))
                    want = Float32(1.0)
        self.weight.update(want, halflife, dt)
        self.yaw.update(goal_yaw, halflife, dt)
        self.pitch.update(goal_pitch, halflife, dt)
        var w = max(Float32(0), min(Float32(1), self.weight.value))
        self.applied_yaw = self.yaw.value * w
        self.applied_pitch = self.pitch.value * w
        if w < Float32(0.001):
            return
        var full_local = yaw_pitch(self.applied_yaw * DEG, self.applied_pitch * DEG)
        var full = ref_rot.multiply(full_local).multiply(ref_rot.conjugate()).normalized()
        var part = Quat().slerp(full, max(Float32(0), min(Float32(1), spine_share)))
        if spine >= 0 and spine_share > 0:
            skel.rotate_node_model(spine, part, model)
            skel.rotate_node_model(head, full.multiply(part.conjugate()).normalized(), model)
        else:
            skel.rotate_node_model(head, full, model)


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
        var follow = Float32(1.0) - det_exp(Float32(-0.69314718) * dt / max(Float32(0.001), halflife))
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
        self.exertion.update(max(Float32(0), min(Float32(1), effort)), 1.6, dt)
        var e = max(Float32(0), min(Float32(1), self.exertion.value))
        var rate = rest_rate + (hard_rate - rest_rate) * e
        self.phase += dt * rate
        self.phase -= Float32(Int(self.phase))
        self.applied = det_sin(self.phase * Float32(6.2831853)) * depth * (Float32(1.0) + e)
        if spine >= 0:
            model.rotate_node_model(spine, angle_axis(-self.applied * DEG, Vec3(1, 0, 0)))
        if head >= 0:
            model.rotate_node_model(head, angle_axis(self.applied * Float32(0.6) * DEG, Vec3(1, 0, 0)))


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


def _step_analytical_spring3(
    mut offset: Vec3,
    mut velocity: Vec3,
    stiffness: Float32,
    damping: Float32,
    dt: Float32,
):
    """Exact closed-form 2nd-order harmonic oscillator `x'' + c x' + k x = 0`."""
    if dt <= Float32(0.0):
        return
    var k = max(Float32(1.0e-5), stiffness)
    var gamma = max(Float32(0.0), damping) * Float32(0.5)
    var disc = k - gamma * gamma
    var x0 = offset
    var v0 = velocity

    if disc > Float32(1.0e-6):
        # Underdamped closed form: exp(-gamma*dt) * (x0*cos(wd*dt) + ((v0 + gamma*x0)/wd)*sin(wd*dt))
        var wd = sqrt(disc)
        var env = det_exp(-gamma * dt)
        var c = det_cos(wd * dt)
        var s = det_sin(wd * dt)
        var b = (v0 + x0 * gamma) * (Float32(1.0) / wd)
        offset = (x0 * c + b * s) * env
        velocity = (v0 * c - (x0 * k + v0 * gamma) * (s / wd)) * env
    elif disc >= Float32(-1.0e-6):
        # Critically damped closed form
        var env = det_exp(-gamma * dt)
        var j1 = v0 + x0 * gamma
        offset = (x0 + j1 * dt) * env
        velocity = (v0 - j1 * (gamma * dt)) * env
    else:
        # Overdamped closed form
        var alpha = sqrt(-disc)
        var r1 = -gamma + alpha
        var r2 = -gamma - alpha
        var e1 = det_exp(r1 * dt)
        var e2 = det_exp(r2 * dt)
        var inv_2a = Float32(0.5) / alpha
        var c1 = (v0 - x0 * r2) * inv_2a
        var c2 = (x0 * r1 - v0) * inv_2a
        offset = c1 * e1 + c2 * e2
        velocity = c1 * (r1 * e1) + c2 * (r2 * e2)


struct Recoil(Copyable, Movable):
    """Exact closed-form analytical underdamped angular spring (scaled-angle, model space)."""
    var offset: Vec3
    var velocity: Vec3

    def __init__(out self):
        self.offset = Vec3()
        self.velocity = Vec3()

    def reset(mut self):
        self = Recoil()

    def kick(mut self, away_model: Vec3, strength: Float32):
        """A hit pushing toward `away_model` (model space, horizontal): tips upper body."""
        var flat = Vec3(away_model.x, 0, away_model.z)
        var direction = safe_normalized(flat, Vec3(0, 0, -1))
        var axis = cross(Vec3(0, 1, 0), direction)
        self.velocity = self.velocity + axis * strength

    def step(mut self, stiffness: Float32, damping: Float32, dt: Float32) -> Bool:
        """Advances analytical recoil spring by `dt`; returns False when settled."""
        if dt <= Float32(0.0):
            return self.offset.length() >= Float32(0.00005)
        _step_analytical_spring3(self.offset, self.velocity, stiffness, damping, dt)
        if self.offset.length() < Float32(0.00005) and self.velocity.length() < Float32(0.0005):
            self.offset = Vec3()
            self.velocity = Vec3()
            return False
        return True

    def apply(mut self, mut model: NativeModel, spine: Int, head: Int, stiffness: Float32, damping: Float32, dt: Float32):
        if not self.step(stiffness, damping, dt):
            return
        if spine >= 0:
            model.rotate_node_model(spine, det_quat_from_scaled_angle(self.offset * Float32(0.65)))
        if head >= 0:
            model.rotate_node_model(head, det_quat_from_scaled_angle(self.offset * Float32(0.35)))

    def apply_to_skeleton(
        mut self, mut skel: SkeletonInstance, model: NativeModel, spine: Int, head: Int,
        stiffness: Float32, damping: Float32, dt: Float32,
    ):
        if not self.step(stiffness, damping, dt):
            return
        if spine >= 0:
            skel.rotate_node_model(spine, det_quat_from_scaled_angle(self.offset * Float32(0.65)), model)
        if head >= 0:
            skel.rotate_node_model(head, det_quat_from_scaled_angle(self.offset * Float32(0.35)), model)


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
    """Moves a clip's contact onto a target over its approach."""
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
        var offset = target - contact
        var length = offset.length()
        if length > max_distance and length > 0:
            offset = offset * (max_distance / length)
        self.goal = offset

    def end(mut self):
        self.active = False

    def apply(mut self, mut model: NativeModel, root: Int, clip_time: Float32, dt: Float32, ease: Float32 = 0):
        var want = Vec3()
        if self.active:
            self.progress = smooth01(clip_time / self.contact_time)
            want = self.goal * self.progress
            if ease > 0:
                self.shown.update(want, ease, dt)
            else:
                self.shown.value = want
                self.shown.velocity = Vec3()
        else:
            self.progress = 0
            self.shown.update(want, 0.09, dt)
        if root >= 0 and self.shown.value.length() > 0.0001:
            model.offset_node(root, self.shown.value)


# ---------------------------------------------------------------------------
# Stateful Terrain & Warship Deck Foot-Placement Controller (`FootPlantSolver`)
# ---------------------------------------------------------------------------

struct FootPlantLeg(Copyable, ImplicitlyCopyable, Movable):
    """Per-leg state for stance world locking, inertial swing release, and slope tilt."""
    var hip: Int
    var knee: Int
    var ankle: Int
    var ankle_height: Float32
    var pole_local: Vec3
    var locked: Bool
    var locked_world_pos: Vec3
    var release_offset: Vec3
    var release_vel: Vec3
    var smoothed_tilt: Quat
    var stance_blend: SpringFloat

    def __init__(
        out self,
        hip: Int = -1,
        knee: Int = -1,
        ankle: Int = -1,
        ankle_height: Float32 = 0.10,
        pole_local: Vec3 = Vec3(0.0, 0.5, 1.0),
    ):
        self.hip = hip
        self.knee = knee
        self.ankle = ankle
        self.ankle_height = ankle_height
        self.pole_local = pole_local
        self.locked = False
        self.locked_world_pos = Vec3()
        self.release_offset = Vec3()
        self.release_vel = Vec3()
        self.smoothed_tilt = Quat()
        self.stance_blend = SpringFloat(0.0)


def _clamped_surface_tilt(normal_world: Vec3, max_tilt_rad: Float32) -> Quat:
    """World pitch/roll rotation taking `+Y` to `normal_world`, clamped to `max_tilt_rad`."""
    var up = Vec3(0.0, 1.0, 0.0)
    var n = safe_normalized(normal_world, up)
    if n.y < Float32(0.0):
        n = -n
    var cos_ang = max(Float32(-1.0), min(Float32(1.0), dot(up, n)))
    var ang = det_acos(cos_ang)
    if ang < Float32(1.0e-5):
        return Quat()
    var ax = safe_normalized(cross(up, n), Vec3(1.0, 0.0, 0.0))
    return angle_axis(min(ang, max_tilt_rad), ax).normalized()


struct FootPlantSolver(Copyable, Movable):
    """Stateful two-leg / multi-leg terrain & deck foot-placement solver:
    1. Stance-phase `[plant, lift]` world foot locking (zero foot sliding on slopes/decks).
    2. Critically damped inertial release (`decay_spring`) on lift -> swing transition.
    3. Critically damped `SpringFloat` pelvis lowering `min_f(y_hit_f - y_anim_f, 0)`.
    4. Exact hinge-plane pole-tracked two-bone IK (`solve_two_bone_pole`).
    5. Surface-normal ankle pitch/roll alignment clamped to `max_tilt_radians`.
    """
    var pelvis: Int
    var legs: List[FootPlantLeg]
    var pelvis_offset_y: SpringFloat
    var pelvis_halflife: Float32
    var release_halflife: Float32
    var tilt_halflife: Float32
    var max_tilt_radians: Float32
    var max_pelvis_drop: Float32

    def __init__(out self, pelvis: Int = 0):
        self.pelvis = pelvis
        self.legs = List[FootPlantLeg]()
        self.pelvis_offset_y = SpringFloat(0.0)
        self.pelvis_halflife = 0.08
        self.release_halflife = 0.07
        self.tilt_halflife = 0.06
        self.max_tilt_radians = 0.65
        self.max_pelvis_drop = 0.55

    def add_leg(
        mut self,
        hip: Int,
        knee: Int,
        ankle: Int,
        ankle_height: Float32 = 0.10,
        pole_local: Vec3 = Vec3(0.0, 0.5, 1.0),
    ):
        self.legs.append(FootPlantLeg(hip, knee, ankle, ankle_height, pole_local))

    def reset(mut self):
        self.pelvis_offset_y = SpringFloat(0.0)
        for i in range(len(self.legs)):
            self.legs[i].locked = False
            self.legs[i].release_offset = Vec3()
            self.legs[i].release_vel = Vec3()
            self.legs[i].smoothed_tilt = Quat()
            self.legs[i].stance_blend = SpringFloat(0.0)

    def foot_world_position(self, leg_index: Int, skel: SkeletonInstance, placement: Placement) -> Vec3:
        if leg_index < 0 or leg_index >= len(self.legs):
            return Vec3()
        return placement.to_world(skel.global_position(self.legs[leg_index].ankle))

    def apply_to_skeleton(
        mut self,
        mut skel: SkeletonInstance,
        model: NativeModel,
        placement: Placement,
        in_stance: List[Bool],
        hit_heights: List[Float32],
        hit_normals: List[Vec3],
        dt: Float32,
    ):
        """Solves pelvis drop, world stance foot lock, leg 2-bone IK, and surface-normal
        ankle alignment directly on `skel` without mutating `model.nodes`."""
        var leg_count = len(self.legs)
        if leg_count == 0:
            return
        skel.ensure_topology(model)
        skel.update_matrices(model)

        var scale_y = max(Float32(0.01), abs(placement.scale.y))
        var anim_ankle_world = List[Vec3](capacity=leg_count)
        var raw_pelvis_drop = Float32(0.0)

        # 1. Measure animated world ankle positions and required pelvis lowering
        for f in range(leg_count):
            var ankle_node = self.legs[f].ankle
            var p_anim = placement.to_world(skel.global_position(ankle_node))
            anim_ankle_world.append(p_anim)
            if f < len(hit_heights):
                var y_sole_anim = p_anim.y - self.legs[f].ankle_height * scale_y
                var diff_y = hit_heights[f] - y_sole_anim
                raw_pelvis_drop = min(raw_pelvis_drop, diff_y)

        raw_pelvis_drop = max(-self.max_pelvis_drop * scale_y, min(Float32(0.0), raw_pelvis_drop))
        self.pelvis_offset_y.update(raw_pelvis_drop, self.pelvis_halflife, dt)

        if self.pelvis >= 0 and self.pelvis < skel.node_count and abs(self.pelvis_offset_y.value) > Float32(1.0e-5):
            skel.offset_node(self.pelvis, Vec3(0.0, self.pelvis_offset_y.value / scale_y, 0.0), model)

        # 2. Per-leg world stance lock / swing inertial release + pole IK + normal alignment
        var world_yaw_q = placement.world_yaw_quat()
        var inv_world_yaw_q = world_yaw_q.conjugate()
        var tilt_alpha = Float32(1.0) - det_exp(Float32(-0.69314718) * max(Float32(0.0), dt) / max(Float32(0.005), self.tilt_halflife))

        for f in range(leg_count):
            var hip = self.legs[f].hip
            var knee = self.legs[f].knee
            var ankle = self.legs[f].ankle
            if hip < 0 or knee < 0 or ankle < 0:
                continue

            var stance = in_stance[f] if f < len(in_stance) else False
            var ankle_h_world = self.legs[f].ankle_height * scale_y
            var p_anim = anim_ankle_world[f]
            var y_hit = hit_heights[f] if f < len(hit_heights) else (p_anim.y - ankle_h_world)
            var n_hit = hit_normals[f] if f < len(hit_normals) else Vec3(0.0, 1.0, 0.0)

            var swing_goal = Vec3(p_anim.x, max(p_anim.y, y_hit + ankle_h_world), p_anim.z)
            var target_world: Vec3

            if stance:
                if not self.legs[f].locked:
                    # Plant lock: capture exact world XZ at plant instant, Y on surface
                    var plant_pos = swing_goal + self.legs[f].release_offset
                    self.legs[f].locked_world_pos = Vec3(plant_pos.x, y_hit + ankle_h_world, plant_pos.z)
                    self.legs[f].locked = True
                    self.legs[f].release_offset = Vec3()
                    self.legs[f].release_vel = Vec3()
                else:
                    # Keep world XZ strictly locked (zero foot skate!), track deck Y
                    self.legs[f].locked_world_pos.y = y_hit + ankle_h_world
                target_world = self.legs[f].locked_world_pos
            else:
                if self.legs[f].locked:
                    # Lift -> swing: seed inertial release offset from locked stance point
                    self.legs[f].locked = False
                    self.legs[f].release_offset = self.legs[f].locked_world_pos - swing_goal
                    self.legs[f].release_vel = Vec3()
                if dt > Float32(0.0):
                    decay_spring(self.legs[f].release_offset, self.legs[f].release_vel, self.release_halflife, dt)
                target_world = swing_goal + self.legs[f].release_offset

            self.legs[f].stance_blend.update(Float32(1.0) if stance else Float32(0.0), self.tilt_halflife, dt)

            # Preserve animated ankle orientation prior to 2-bone IK knee flexion
            var pre_ik_ankle_rot = skel.global_rotation(ankle)

            # Solve pole-tracked 2-bone IK in model space
            var target_model = placement.to_model(target_world)
            var hip_model = skel.global_position(hip)
            var pole_model = hip_model + self.legs[f].pole_local
            skel.solve_two_bone(hip, knee, ankle, target_model, pole_model, model, Float32(1.0))

            var ankle_parent = skel.parents[ankle]
            if ankle_parent >= 0:
                skel.pose.r[ankle] = skel.global_rotation(ankle_parent).conjugate().multiply(pre_ik_ankle_rot).normalized()
            else:
                skel.pose.r[ankle] = pre_ik_ankle_rot
            skel.update_matrices(model)

            # Surface-normal pitch/roll alignment on the ankle
            var want_tilt_world = _clamped_surface_tilt(n_hit, self.max_tilt_radians)
            var blended_tilt_world = Quat().slerp(
                want_tilt_world,
                max(Float32(0.0), min(Float32(1.0), self.legs[f].stance_blend.value)),
            )
            self.legs[f].smoothed_tilt = self.legs[f].smoothed_tilt.slerp(blended_tilt_world, tilt_alpha).normalized()
            var tilt_model = inv_world_yaw_q.multiply(self.legs[f].smoothed_tilt).multiply(world_yaw_q).normalized()
            skel.rotate_node_model(ankle, tilt_model, model)

    def apply(
        mut self,
        mut model: NativeModel,
        placement: Placement,
        in_stance: List[Bool],
        hit_heights: List[Float32],
        hit_normals: List[Vec3],
        dt: Float32,
    ):
        """Convenience wrapper applying `FootPlantSolver` onto `NativeModel.nodes`."""
        var skel = SkeletonInstance(len(model.nodes))
        skel.ensure_topology(model)
        skel.pose.read(model)
        skel.update_matrices(model)
        self.apply_to_skeleton(skel, model, placement, in_stance, hit_heights, hit_normals, dt)
        for i in range(min(len(model.nodes), skel.pose.count())):
            model.nodes[i].translation = skel.pose.t[i]
            model.nodes[i].rotation = skel.pose.r[i]
            model.nodes[i].scale = skel.pose.s[i]
        model.transforms_revision += 1
