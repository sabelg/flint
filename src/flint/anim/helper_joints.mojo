"""Helper joints (package 3.4): bones the engine poses itself, after IK.

Blending fixes most of what linear skinning breaks, but a joint still looks
better with a bone that does part of its neighbour's turn and takes the
vertices around the joint with it. Two kinds, both set on a pose bone in
Blender as the custom property `ds_helper` ({kind, source, amount}) and
exported into the manifest's skeleton (tools/blender/pipeline.py):

    half    a half-angle joint, for shoulders, hips, elbows and knees: the
            helper turns `amount` (0.5) of the way its source has turned from
            rest. Skin the crease and the outside of the joint to it.
    twist   a twist joint, for forearms: the helper takes `amount` of the
            source's turn about the source bone's own direction (the roll of
            the hand about the forearm) and none of its bend. Skin the middle
            of the forearm to it.

A helper and its source share a parent, so both rotations are in the same
frame: the elbow helper is a child of the upper arm beside the forearm, the
forearm twist a child of the forearm beside the hand. Each frame, after the
animation graph, the procedural passes and all IK, and before spring bones and
skinning:

    turn   = source.rotation x inverse(source.rest rotation)
    helper = slerp(identity, turn or its twist part, amount) x helper.rest rotation

Whatever a clip keyed on a helper is overwritten: the engine owns these
bones. Presentation only.
"""
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.assets.model import NativeModel
from flint.assets.manifest import ManifestBone, HELPER_HALF, HELPER_TWIST


struct HelperJoint(Copyable, ImplicitlyCopyable, Movable):
    var node: Int
    var source: Int
    var kind: Int
    var amount: Float32
    var axis: Vec3  # twist: the source bone's direction in the parent's frame

    def __init__(out self):
        self.node = -1
        self.source = -1
        self.kind = HELPER_HALF
        self.amount = 0.5
        self.axis = Vec3(0, 1, 0)


def twist_part(turn: Quat, axis: Vec3) -> Quat:
    """The part of a rotation that turns about `axis` (a unit vector)."""
    var along = turn.x * axis.x + turn.y * axis.y + turn.z * axis.z
    var twist = Quat(axis.x * along, axis.y * along, axis.z * along, turn.w)
    if twist.x * twist.x + twist.y * twist.y + twist.z * twist.z + twist.w * twist.w < 0.000001:
        return Quat()  # a half turn about another axis: no twist to speak of
    return twist.normalized()


def helper_rotation(kind: Int, amount: Float32, axis: Vec3, source_rotation: Quat, source_rest: Quat, helper_rest: Quat) -> Quat:
    var turn = source_rotation.multiply(source_rest.conjugate()).normalized().shortest()
    if kind == HELPER_TWIST:
        turn = twist_part(turn, axis).shortest()
    return Quat().slerp(turn, amount).multiply(helper_rest).normalized()


struct HelperJoints(Copyable, Movable):
    var handle: Int
    var joints: List[HelperJoint]
    var problems: List[String]  # manifest helpers this model cannot have, and why

    def __init__(out self):
        self.handle = -1
        self.joints = List[HelperJoint]()
        self.problems = List[String]()

    def count(self) -> Int:
        return len(self.joints)

    def bind(mut self, model: NativeModel, handle: Int, skeleton: List[ManifestBone]):
        """The helper joints of this model, from its manifest. A helper whose
        bones are missing or not siblings is left alone and named in `problems`."""
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
            if joint.node == joint.source or parents[joint.node] != parents[joint.source]:
                self.problems.append("helper " + entry.name + ": it and its source " + entry.helper_source + " must be children of the same bone")
                continue
            if joint.kind == HELPER_TWIST:
                var direction = model.nodes[joint.source].base_translation
                if direction.length() < 0.00001:
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
