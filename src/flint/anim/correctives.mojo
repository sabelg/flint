"""Corrective morphs (package 3.5): morph targets the engine drives from a
joint's swing, as Blender's drivers do on the control rig.

Maroon v6 has eight corrective shape keys (`corr_arm_down`, `corr_arm_up`,
`corr_arm_fwd`, `corr_elbow`, left and right): where linear blend skinning
folds the thick arm into the chest or the forearm into the upper arm, the
shape pushes the skin back out. In Blender each key's value is a driver on
its bone's rotation in SWING_TWIST_Y mode. The pipeline writes the drivers'
numbers into the mesh manifest (`correctives`: morph, bone, x, z) and each
frame, after the animation graph, the procedural passes, all IK and the
helper joints, and before skinning:

    turn   = inverse(bone.rest rotation) x bone.rotation     (in the bone's own axes)
    swing  = turn with its twist about the bone's length (Y) taken out
    weight = clamp(x * swing_x + z * swing_z, 0, 1)          (angles in radians)

where swing_x and swing_z are 2 asin of the swing quaternion's x and z, the
numbers Blender's ROT_X and ROT_Z channels give in that mode. The weight goes
to `NativeModel.set_morph_weight`; package 3.4's skinning applies the morph
before the bones, on the CPU and in the GPU pass. Only LOD0 has morph targets
(a decimated mesh has other vertices), so a lower level binds nothing.
Presentation only.
"""
from std.math import asin, atan2, sin, cos
from flint.core.transform import Quat
from flint.assets.model import NativeModel
from flint.assets.manifest import ManifestCorrective


struct Swing(Copyable, ImplicitlyCopyable, Movable):
    var x: Float32
    var z: Float32
    var twist: Float32

    def __init__(out self, x: Float32, z: Float32, twist: Float32):
        self.x = x
        self.z = z
        self.twist = twist


def swing_twist_y(rotation: Quat, rest: Quat) -> Swing:
    """The bone's turn from rest split into a twist about its own Y (its
    length) and the swing that follows it; the swing as Blender's driver
    channels report it."""
    var turn = rest.conjugate().multiply(rotation).normalized().shortest()
    var half = atan2(turn.y, turn.w)
    var swing = turn.multiply(Quat(0, -sin(half), 0, cos(half)))
    var x = max(Float32(-1), min(Float32(1), swing.x))
    var z = max(Float32(-1), min(Float32(1), swing.z))
    return Swing(2.0 * asin(x), 2.0 * asin(z), 2.0 * half)


def corrective_weight(x: Float32, z: Float32, rotation: Quat, rest: Quat) -> Float32:
    var swing = swing_twist_y(rotation, rest)
    return max(Float32(0), min(Float32(1), x * swing.x + z * swing.z))


struct Corrective(Copyable, ImplicitlyCopyable, Movable):
    var node: Int
    var morph: Int
    var x: Float32
    var z: Float32

    def __init__(out self):
        self.node = -1
        self.morph = -1
        self.x = 0
        self.z = 0


struct Correctives(Copyable, Movable):
    var handle: Int
    var rows: List[Corrective]
    var weights: List[Float32]  # what the last apply set, by row (tests, the inspector)

    def __init__(out self):
        self.handle = -1
        self.rows = List[Corrective]()
        self.weights = List[Float32]()

    def count(self) -> Int:
        return len(self.rows)

    def bind(mut self, model: NativeModel, handle: Int, table: List[ManifestCorrective]):
        """The rows this model can take: its bone and its morph target must
        both be in the model (lower levels of detail have no morphs)."""
        if handle == self.handle:
            return
        self.handle = handle
        self.rows.clear()
        self.weights.clear()
        for entry in table:
            var row = Corrective()
            row.node = model.find_node(entry.bone)
            for index in range(len(model.morph_names)):
                if model.morph_names[index] == entry.morph:
                    row.morph = index
            if row.node < 0 or row.morph < 0:
                continue
            row.x = entry.x
            row.z = entry.z
            self.rows.append(row)
            self.weights.append(0)

    def apply(mut self, mut model: NativeModel):
        for index in range(len(self.rows)):
            var row = self.rows[index]
            if row.node >= len(model.nodes):
                continue
            var weight = corrective_weight(row.x, row.z, model.nodes[row.node].rotation, model.nodes[row.node].base_rotation)
            self.weights[index] = weight
            model.set_morph_weight(row.morph, weight)
