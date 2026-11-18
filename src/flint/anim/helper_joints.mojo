"""Helper joints (package 3.4 / v4): bones the engine poses itself, after IK.

Supports three rigging topologies:
1. `HELPER_HALF` (`1`): sibling half-angle crease joint (shoulders, hips, elbows, knees)
   sharing a parent with `source`, taking `amount` of `source`'s full turn.
2. `HELPER_TWIST` (`2`): sibling distal twist joint (`Forearm_Twist` beside `Hand`,
   both children of `Forearm`), taking `+amount` of `source`'s axial roll around
   the bone shaft and none of its bend.
3. `HELPER_PROXIMAL_TWIST` (`3`, or `HELPER_TWIST` when `helper` is a child of
   `source`, e.g. `UpperArm_Twist` parented to `UpperArm`): proximal counter-twist
   joint decomposing `source`'s local turn `inverse(source.rest) * source.rotation`
   along `UpperArm`'s bone shaft and applying `slerp(I, twist^*, amount) * helper.rest`
   so axial shoulder roll is canceled (`1 - amount` in world space) while 100% of
   shoulder swing is preserved!
"""
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.assets.model import NativeModel
from flint.assets.manifest import ManifestBone, HELPER_HALF, HELPER_TWIST
from flint.anim.pose import Pose

comptime HELPER_PROXIMAL_TWIST: Int = 3


struct HelperJoint(Copyable, ImplicitlyCopyable, Movable):
    var node: Int
    var source: Int
    var kind: Int
    var amount: Float32
    var axis: Vec3

    def __init__(out self):
        self.node = -1
        self.source = -1
        self.kind = HELPER_HALF
        self.amount = 0.5
        self.axis = Vec3(0, 1, 0)


def twist_part(turn: Quat, axis: Vec3) -> Quat:
    """The part of a rotation `turn` that rolls about unit vector `axis`."""
    var along = turn.x * axis.x + turn.y * axis.y + turn.z * axis.z
    var twist = Quat(axis.x * along, axis.y * along, axis.z * along, turn.w)
    if twist.x * twist.x + twist.y * twist.y + twist.z * twist.z + twist.w * twist.w < Float32(0.000001):
        return Quat()
    return twist.normalized()


def swing_twist_decompose(turn: Quat, axis: Vec3) -> Tuple[Quat, Quat]:
    """Decomposes `turn = swing * twist` where `twist` rotates purely around `axis`."""
    var twist = twist_part(turn, axis).shortest()
    var swing = turn.multiply(twist.conjugate()).normalized().shortest()
    return (swing, twist)


def helper_rotation(
    kind: Int,
    amount: Float32,
    axis: Vec3,
    source_rotation: Quat,
    source_rest: Quat,
    helper_rest: Quat,
) -> Quat:
    if kind == HELPER_PROXIMAL_TWIST:
        # Child of `source`: source_rotation = source_rest * swing * twist.
        # Setting helper = slerp(I, twist^*, amount) * helper_rest gives world
        # G_helper = source_rest * swing * twist^(1 - amount) * helper_rest!
        var local_turn = source_rest.conjugate().multiply(source_rotation).normalized().shortest()
        var twist = twist_part(local_turn, axis).shortest()
        var counter_twist = Quat().slerp(twist.conjugate().shortest(), amount)
        return counter_twist.multiply(helper_rest).normalized()

    var turn = source_rotation.multiply(source_rest.conjugate()).normalized().shortest()
    if kind == HELPER_TWIST:
        turn = twist_part(turn, axis).shortest()
    return Quat().slerp(turn, amount).multiply(helper_rest).normalized()


def _find_shaft_axis(model: NativeModel, bone_node: Int, helper_node: Int) -> Vec3:
    """Finds the bone shaft direction in `bone_node`'s local frame from its primary
    distal child (e.g. `Forearm` under `UpperArm`) or `helper_node`'s offset."""
    var best_dir = Vec3()
    var best_len = Float32(0.0)
    for child in model.nodes[bone_node].children:
        if child >= 0 and child < len(model.nodes) and child != helper_node:
            var v = model.nodes[child].base_translation
            var l = v.length()
            if l > best_len:
                best_len = l
                best_dir = v
    if best_len > Float32(0.00001):
        return best_dir * (Float32(1.0) / best_len)
    if helper_node >= 0 and helper_node < len(model.nodes):
        var hv = model.nodes[helper_node].base_translation
        if hv.length() > Float32(0.00001):
            return hv.normalized()
    return Vec3(0.0, 1.0, 0.0)


struct HelperJoints(Copyable, Movable):
    var handle: Int
    var joints: List[HelperJoint]
    var problems: List[String]

    def __init__(out self):
        self.handle = -1
        self.joints = List[HelperJoint]()
        self.problems = List[String]()

    def count(self) -> Int:
        return len(self.joints)

    def bind(mut self, model: NativeModel, handle: Int, skeleton: List[ManifestBone]):
        """Binds sibling half/distal-twist helpers AND child proximal counter-twist helpers."""
        if handle == self.handle:
            return
        self.handle = handle
        self.joints.clear()
        self.problems.clear()
        var parents = List[Int](length=len(model.nodes), fill=-1)
        for index in range(len(model.nodes)):
            for child in model.nodes[index].children:
                if child >= 0 and child < len(parents):
                    parents[child] = index
        for entry in skeleton:
            if entry.helper_kind == 0:
                continue
            var joint = HelperJoint()
            joint.node = model.find_node(entry.name)
            joint.source = model.find_node(entry.helper_source)
            joint.kind = entry.helper_kind
            joint.amount = entry.helper_amount
            if joint.node < 0 or joint.source < 0:
                self.problems.append("helper " + entry.name + ": no bone " + (entry.name if joint.node < 0 else entry.helper_source) + " in the model")
                continue
            if joint.node == joint.source:
                self.problems.append("helper " + entry.name + ": cannot drive itself")
                continue

            # Proximal counter-twist: helper is a direct child of its source bone (e.g. UpperArm_Twist -> UpperArm)
            if parents[joint.node] == joint.source and (joint.kind == HELPER_TWIST or joint.kind == HELPER_PROXIMAL_TWIST):
                joint.kind = HELPER_PROXIMAL_TWIST
                joint.axis = _find_shaft_axis(model, joint.source, joint.node)
                self.joints.append(joint)
                continue

            if parents[joint.node] != parents[joint.source]:
                self.problems.append("helper " + entry.name + ": it and its source " + entry.helper_source + " must share a parent (or be a child twist bone)")
                continue

            if joint.kind == HELPER_TWIST:
                var direction = model.nodes[joint.source].base_translation
                if direction.length() < Float32(0.00001):
                    self.problems.append("helper " + entry.name + ": its source " + entry.helper_source + " sits on its parent, so there is no bone to twist about")
                    continue
                joint.axis = direction.normalized()
            self.joints.append(joint)

    def apply(self, mut model: NativeModel):
        """Pose every helper from its source's rotation now."""
        if len(self.joints) == 0:
            return
        for joint in self.joints:
            if joint.node >= len(model.nodes) or joint.source >= len(model.nodes):
                continue
            model.nodes[joint.node].rotation = helper_rotation(
                joint.kind, joint.amount, joint.axis,
                model.nodes[joint.source].rotation, model.nodes[joint.source].base_rotation, model.nodes[joint.node].base_rotation,
            )
        model.transforms_revision += 1

    def apply_to_pose(self, mut pose: Pose, model: NativeModel):
        """Non-mutating `Pose` overload posing every helper joint without touching `model.nodes`."""
        for joint in self.joints:
            if joint.node < 0 or joint.source < 0 or joint.node >= pose.count() or joint.source >= pose.count():
                continue
            if joint.node >= len(model.nodes) or joint.source >= len(model.nodes):
                continue
            pose.r[joint.node] = helper_rotation(
                joint.kind, joint.amount, joint.axis,
                pose.r[joint.source], model.nodes[joint.source].base_rotation, model.nodes[joint.node].base_rotation,
            )
