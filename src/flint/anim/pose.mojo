"""Poses: every node's local translation, rotation and scale, sampled from a
clip, blended, masked, added and inertialized (package 3.3).

The animation runtime builds a pose from the state machine's clips and
layers, then writes it into the model's nodes once; IK and the procedural
passes work on the nodes after that. Presentation only: nothing here feeds
the simulation.

Additive layers follow the usual local-space form: the delta is the clip's
pose relative to a reference pose of the same clip (its first frame unless
the graph says otherwise), and the delta is applied on top of the base,
rotation after the base's own (base * delta), translation added.

Inertialization is the one in `NativeModel.animate` (Bollo, GDC 2018, with
Holden's springs), on a pose instead of the nodes, so a blended and layered
pose changes state without a crossfade buffer.
"""
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.springs import decay_spring
from flint.assets.model import NativeModel


struct Pose(Copyable, Movable):
    var t: List[Vec3]
    var r: List[Quat]
    var s: List[Vec3]

    def __init__(out self, count: Int = 0):
        self.t = List[Vec3](length=count, fill=Vec3())
        self.r = List[Quat](length=count, fill=Quat())
        self.s = List[Vec3](length=count, fill=Vec3(1, 1, 1))

    def count(self) -> Int:
        return len(self.t)

    def resize(mut self, count: Int):
        if len(self.t) != count:
            self = Pose(count)

    def rest(mut self, model: NativeModel):
        """The model's rest pose (each node's base transform)."""
        self.resize(len(model.nodes))
        for index in range(len(model.nodes)):
            self.t[index] = model.nodes[index].base_translation
            self.r[index] = model.nodes[index].base_rotation
            self.s[index] = model.nodes[index].base_scale

    def read(mut self, model: NativeModel):
        """The model's current node transforms."""
        self.resize(len(model.nodes))
        for index in range(len(model.nodes)):
            self.t[index] = model.nodes[index].translation
            self.r[index] = model.nodes[index].rotation
            self.s[index] = model.nodes[index].scale

    def assign(mut self, other: Pose):
        self.resize(other.count())
        for index in range(other.count()):
            self.t[index] = other.t[index]
            self.r[index] = other.r[index]
            self.s[index] = other.s[index]


def write_pose(mut model: NativeModel, pose: Pose):
    """Puts the pose into the model's nodes (one revision bump)."""
    var count = min(len(model.nodes), pose.count())
    for index in range(count):
        model.nodes[index].translation = pose.t[index]
        model.nodes[index].rotation = pose.r[index]
        model.nodes[index].scale = pose.s[index]
    model.transforms_revision += 1


def sample_pose(model: NativeModel, clip: Int, time_seconds: Float32, mut pose: Pose):
    """The clip at a time (clamped to its length) over the rest pose, keyed
    exactly as `NativeModel.sample_clip` keys it."""
    pose.rest(model)
    if clip < 0 or clip >= len(model.animations):
        return
    ref animation = model.animations[clip]
    var time = max(Float32(0), min(time_seconds, animation.duration))
    for channel in animation.channels:
        if channel.node < 0 or channel.node >= pose.count() or len(channel.times) == 0:
            continue
        var lower_bound = 0
        var upper_bound = len(channel.times) - 1
        while lower_bound + 1 < upper_bound:
            var middle = (lower_bound + upper_bound) // 2
            if channel.times[middle] < time:
                lower_bound = middle
            else:
                upper_bound = middle
        var lower = max(0, min(len(channel.times) - 1, lower_bound))
        var next = min(len(channel.times) - 1, lower + 1)
        var amount = Float32(0)
        if not channel.step and next != lower and channel.times[next] > channel.times[lower]:
            amount = max(Float32(0), min(Float32(1), (time - channel.times[lower]) / (channel.times[next] - channel.times[lower])))
        var components = 4 if channel.path == 1 else 3
        var first = lower * components
        var second = next * components
        if first + components > len(channel.values) or second + components > len(channel.values):
            continue
        if channel.path == 0:
            pose.t[channel.node] = Vec3(
                channel.values[first] + (channel.values[second] - channel.values[first]) * amount,
                channel.values[first + 1] + (channel.values[second + 1] - channel.values[first + 1]) * amount,
                channel.values[first + 2] + (channel.values[second + 2] - channel.values[first + 2]) * amount,
            )
        elif channel.path == 1:
            var a = Quat(channel.values[first], channel.values[first + 1], channel.values[first + 2], channel.values[first + 3]).normalized()
            var b = Quat(channel.values[second], channel.values[second + 1], channel.values[second + 2], channel.values[second + 3]).normalized()
            pose.r[channel.node] = a.slerp(b, amount)
        else:
            pose.s[channel.node] = Vec3(
                channel.values[first] + (channel.values[second] - channel.values[first]) * amount,
                channel.values[first + 1] + (channel.values[second + 1] - channel.values[first + 1]) * amount,
                channel.values[first + 2] + (channel.values[second + 2] - channel.values[first + 2]) * amount,
            )


def lerp3(a: Vec3, b: Vec3, w: Float32) -> Vec3:
    return a + (b - a) * w


def blend_into(mut base: Pose, other: Pose, weight: Float32):
    """base = base blended toward other by weight, every node."""
    if weight <= 0:
        return
    var count = min(base.count(), other.count())
    if weight >= 1:
        for index in range(count):
            base.t[index] = other.t[index]
            base.r[index] = other.r[index]
            base.s[index] = other.s[index]
        return
    for index in range(count):
        base.t[index] = lerp3(base.t[index], other.t[index], weight)
        base.r[index] = base.r[index].slerp(other.r[index], weight)
        base.s[index] = lerp3(base.s[index], other.s[index], weight)


def blend_masked(mut base: Pose, other: Pose, mask: List[Float32], weight: Float32):
    """An override layer: each node moves toward other by weight x its mask
    weight; nodes outside the mask keep the base."""
    if weight <= 0:
        return
    var count = min(min(base.count(), other.count()), len(mask))
    for index in range(count):
        var w = min(Float32(1), mask[index] * weight)
        if w <= 0:
            continue
        if w >= 1:
            base.t[index] = other.t[index]
            base.r[index] = other.r[index]
            base.s[index] = other.s[index]
            continue
        base.t[index] = lerp3(base.t[index], other.t[index], w)
        base.r[index] = base.r[index].slerp(other.r[index], w)
        base.s[index] = lerp3(base.s[index], other.s[index], w)


def add_additive(mut base: Pose, layer: Pose, reference: Pose, mask: List[Float32], weight: Float32):
    """An additive layer: layer's change from reference, times weight x the
    node's mask weight, on top of base (rotation: base * delta)."""
    if weight <= 0:
        return
    var count = min(min(base.count(), layer.count()), min(reference.count(), len(mask)))
    var identity = Quat()
    for index in range(count):
        var w = mask[index] * weight
        if w <= 0:
            continue
        var delta = reference.r[index].conjugate().multiply(layer.r[index]).normalized()
        base.r[index] = base.r[index].multiply(identity.slerp(delta, w)).normalized()
        base.t[index] = base.t[index] + (layer.t[index] - reference.t[index]) * w


def full_mask(count: Int) -> List[Float32]:
    return List[Float32](length=count, fill=1)


struct Inertializer(Copyable, Movable):
    """Pose memory that hides a state change: the jump between the last
    output and the new pose is an offset decayed by a critically damped
    spring (as `NativeModel.animate`, on poses)."""
    var has_pose: Bool
    var out_t: List[Vec3]
    var out_r: List[Quat]
    var out_vel_t: List[Vec3]
    var out_vel_r: List[Vec3]
    var off_t: List[Vec3]
    var vel_t: List[Vec3]
    var off_r: List[Vec3]
    var vel_r: List[Vec3]

    def __init__(out self):
        self.has_pose = False
        self.out_t = List[Vec3]()
        self.out_r = List[Quat]()
        self.out_vel_t = List[Vec3]()
        self.out_vel_r = List[Vec3]()
        self.off_t = List[Vec3]()
        self.vel_t = List[Vec3]()
        self.off_r = List[Vec3]()
        self.vel_r = List[Vec3]()

    def resize(mut self, count: Int):
        if len(self.out_t) == count:
            return
        self.out_t = List[Vec3](length=count, fill=Vec3())
        self.out_r = List[Quat](length=count, fill=Quat())
        self.out_vel_t = List[Vec3](length=count, fill=Vec3())
        self.out_vel_r = List[Vec3](length=count, fill=Vec3())
        self.off_t = List[Vec3](length=count, fill=Vec3())
        self.vel_t = List[Vec3](length=count, fill=Vec3())
        self.off_r = List[Vec3](length=count, fill=Vec3())
        self.vel_r = List[Vec3](length=count, fill=Vec3())
        self.has_pose = False

    def reset(mut self):
        """Forget the last output: the next pose is shown as it is."""
        self.has_pose = False

    def apply(mut self, mut pose: Pose, ahead: Pose, switching: Bool, halflife: Float32, dt: Float32):
        """pose: the new source pose, replaced by the output. ahead: the same
        source one step (dt) later, read only when switching, for its velocity."""
        var count = pose.count()
        self.resize(count)
        var switch_now = switching and self.has_pose and halflife > 0 and dt > 0 and ahead.count() == count
        if switch_now:
            var inverse_dt = 1.0 / dt
            for index in range(count):
                var src_t = pose.t[index]
                var src_r = pose.r[index]
                var src_vel_t = (ahead.t[index] - src_t) * inverse_dt
                var src_vel_r = ahead.r[index].multiply(src_r.conjugate()).to_scaled_angle() * inverse_dt
                self.off_t[index] = self.out_t[index] - src_t
                self.vel_t[index] = self.out_vel_t[index] - src_vel_t
                self.off_r[index] = self.out_r[index].multiply(src_r.conjugate()).to_scaled_angle()
                self.vel_r[index] = self.out_vel_r[index] - src_vel_r
        elif not self.has_pose:
            for index in range(count):
                self.off_t[index] = Vec3()
                self.vel_t[index] = Vec3()
                self.off_r[index] = Vec3()
                self.vel_r[index] = Vec3()
        for index in range(count):
            if dt > 0 and halflife > 0:
                decay_spring(self.off_t[index], self.vel_t[index], halflife, dt)
                decay_spring(self.off_r[index], self.vel_r[index], halflife, dt)
            var final_t = pose.t[index] + self.off_t[index]
            var final_r = Quat.from_scaled_angle(self.off_r[index]).multiply(pose.r[index]).normalized()
            if self.has_pose and dt > 0:
                self.out_vel_t[index] = (final_t - self.out_t[index]) * (1.0 / dt)
                self.out_vel_r[index] = final_r.multiply(self.out_r[index].conjugate()).to_scaled_angle() * (1.0 / dt)
            else:
                self.out_vel_t[index] = Vec3()
                self.out_vel_r[index] = Vec3()
            self.out_t[index] = final_t
            self.out_r[index] = final_r
            pose.t[index] = final_t
            pose.r[index] = final_r
        self.has_pose = True

    def largest_offset(self) -> Float32:
        """The largest remaining offset: metres of translation or radians of rotation."""
        var largest = Float32(0)
        for index in range(len(self.off_t)):
            largest = max(largest, max(self.off_t[index].length(), self.off_r[index].length()))
        return largest


def bone_mask(model: NativeModel, bones: List[String], weights: List[Float32], children: Bool) -> List[Float32]:
    """Per-node weights for a layer: each named bone gets its weight and,
    with children, every descendant without its own entry takes its
    nearest named ancestor's. Unknown names are skipped (the graph's
    loader reports them)."""
    var count = len(model.nodes)
    var mask = List[Float32](length=count, fill=0)
    var named = List[Bool](length=count, fill=False)
    for index in range(min(len(bones), len(weights))):
        var node = model.find_node(bones[index])
        if node >= 0:
            mask[node] = weights[index]
            named[node] = True
    if not children:
        return mask^
    var parents = List[Int](length=count, fill=-1)
    for index in range(count):
        for child in model.nodes[index].children:
            if child >= 0 and child < count:
                parents[child] = index
    for index in range(count):
        if named[index]:
            continue
        var current = parents[index]
        var guard = 0
        while current >= 0 and guard < 128:
            if named[current]:
                mask[index] = mask[current]
                break
            current = parents[current]
            guard += 1
    return mask^
