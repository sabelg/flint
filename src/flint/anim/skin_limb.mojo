"""A procedural skinned limb (package 3.4), for the skinning tests, the
benchmark and the pictures: a capped cylinder along +Y, an upper arm (0 to 1),
a forearm (1 to 2) and a hand (2 to its end), weights blended smoothly across
the elbow and the wrist. With `helpers` it also has the two helper joints of
anim/helper_joints.mojo, skinned as an artist would: a half-angle joint on
the elbow and a twist joint in the middle of the forearm.

Vertex colours stripe it along its length, so a twist shows in a picture.
"""
from std.math import sin, cos, sqrt
from flint.core.math import Vec3
from flint.core.transform import Mat4
from flint.assets.model import NativeModel, NativeMesh, NativeNode, NativeSkin
from flint.assets.manifest import ManifestBone, HELPER_HALF, HELPER_TWIST
from flint.anim.skinning import compute_cor, SKIN_COR

comptime LIMB_ROOT: Int = 0
comptime LIMB_UPPER: Int = 1
comptime LIMB_FOREARM: Int = 2
comptime LIMB_HAND: Int = 3
comptime LIMB_ELBOW_HALF: Int = 4
comptime LIMB_TWIST: Int = 5
comptime LIMB_RADIUS: Float32 = 0.15
comptime LIMB_LENGTH: Float32 = 2.3


def smooth(value: Float32) -> Float32:
    var t = max(Float32(0), min(Float32(1), value))
    return t * t * (3.0 - 2.0 * t)


def limb_node(name: String, at: Vec3) -> NativeNode:
    var node = NativeNode()
    node.name = name
    node.base_translation = at
    node.translation = at
    return node^


def limb_weights(y: Float32, helpers: Bool) -> List[Float32]:
    """The share of each of the six bones at height y."""
    var out = List[Float32](length=6, fill=0)
    var elbow = smooth((y - 0.75) / 0.5)
    var wrist = smooth((y - 1.8) / 0.4)
    out[LIMB_UPPER] = 1.0 - elbow
    if not helpers:
        out[LIMB_FOREARM] = elbow * (1.0 - wrist)
        out[LIMB_HAND] = elbow * wrist
        return out^
    # Along the forearm: forearm at the elbow, the twist joint in the middle, the hand at the wrist.
    var to_twist = smooth((y - 1.2) / 0.4)
    var to_hand = smooth((y - 1.6) / 0.4)
    out[LIMB_FOREARM] = elbow * (1.0 - to_twist)
    out[LIMB_TWIST] = elbow * to_twist * (1.0 - to_hand)
    out[LIMB_HAND] = elbow * to_twist * to_hand
    # The half-angle joint takes the ring around the elbow from both neighbours.
    var half = max(Float32(0), 1.0 - abs(y - 1.0) / 0.3) * 0.7
    out[LIMB_UPPER] = out[LIMB_UPPER] * (1.0 - half)
    out[LIMB_FOREARM] = out[LIMB_FOREARM] * (1.0 - half)
    out[LIMB_TWIST] = out[LIMB_TWIST] * (1.0 - half)
    out[LIMB_ELBOW_HALF] = half
    return out^


def limb_vertex(mut mesh: NativeMesh, position: Vec3, normal: Vec3, stripe: Bool, helpers: Bool):
    mesh.positions.append(position.x)
    mesh.positions.append(position.y)
    mesh.positions.append(position.z)
    mesh.normals.append(normal.x)
    mesh.normals.append(normal.y)
    mesh.normals.append(normal.z)
    mesh.texcoords.append(0)
    mesh.texcoords.append(0)
    var shade = UInt8(255) if stripe else UInt8(150)
    mesh.colors.append(shade)
    mesh.colors.append(shade)
    mesh.colors.append(UInt8(255) if stripe else UInt8(190))
    mesh.colors.append(255)
    # The four heaviest bones, heaviest first.
    var shares = limb_weights(position.y, helpers)
    for _ in range(4):
        var best = 0
        for bone in range(1, 6):
            if shares[bone] > shares[best]:
                best = bone
        mesh.joints.append(UInt16(best))
        mesh.weights.append(shares[best])
        shares[best] = 0


def skin_limb(rings: Int = 47, sides: Int = 24, helpers: Bool = False) -> NativeModel:
    var model = NativeModel()
    model.nodes.append(limb_node("Root", Vec3()))
    model.nodes.append(limb_node("Upper", Vec3()))
    model.nodes.append(limb_node("Forearm", Vec3(0, 1, 0)))
    model.nodes.append(limb_node("Hand", Vec3(0, 1, 0)))
    model.nodes.append(limb_node("Elbow_Half", Vec3(0, 1, 0)))
    model.nodes.append(limb_node("Forearm_Twist", Vec3(0, 0.5, 0)))
    model.nodes[LIMB_ROOT].children.append(LIMB_UPPER)
    model.nodes[LIMB_UPPER].children.append(LIMB_FOREARM)
    model.nodes[LIMB_FOREARM].children.append(LIMB_HAND)
    model.nodes[LIMB_UPPER].children.append(LIMB_ELBOW_HALF)
    model.nodes[LIMB_FOREARM].children.append(LIMB_TWIST)
    model.roots.append(LIMB_ROOT)
    var rest: List[Float32] = [0.0, 0.0, 1.0, 2.0, 1.0, 1.5]  # each bone's height at rest
    var skin = NativeSkin()
    for joint in range(6):
        skin.joints.append(joint)
        var inverse = Mat4()
        inverse.set(1, 3, -rest[joint])
        for component in range(16):
            skin.inverse_bind.append(inverse.values[component])
    model.skins.append(skin^)
    var mesh = NativeMesh()
    mesh.skin = 0
    mesh.double_sided = True
    mesh.base_color = 0xFFF0E0C8

    for ring in range(rings):
        var y = LIMB_LENGTH * Float32(ring) / Float32(rings - 1)
        for side in range(sides):
            var angle = 6.2831853 * Float32(side) / Float32(sides)
            var normal = Vec3(cos(angle), 0, sin(angle))
            limb_vertex(mesh, Vec3(normal.x * LIMB_RADIUS, y, normal.z * LIMB_RADIUS), normal, (side * 8 // sides) % 2 == 0, helpers)
    var bottom = rings * sides
    limb_vertex(mesh, Vec3(0, 0, 0), Vec3(0, -1, 0), True, helpers)
    limb_vertex(mesh, Vec3(0, LIMB_LENGTH, 0), Vec3(0, 1, 0), True, helpers)
    for ring in range(rings - 1):
        for side in range(sides):
            var a = ring * sides + side
            var b = ring * sides + (side + 1) % sides
            var c = (ring + 1) * sides + side
            var d = (ring + 1) * sides + (side + 1) % sides
            for index in [a, c, b, b, c, d]:
                mesh.indices.append(index)
    for side in range(sides):
        var next = (side + 1) % sides
        for index in [bottom, side, next, bottom + 1, (rings - 1) * sides + next, (rings - 1) * sides + side]:
            mesh.indices.append(index)
    mesh.centers = compute_cor(mesh.positions, mesh.indices, mesh.joints, mesh.weights)
    model.meshes.append(mesh^)
    model.nodes[LIMB_ROOT].mesh = 0
    model.nodes[LIMB_ROOT].mesh_indices.append(0)
    model.skin_method = SKIN_COR
    model.compute_bounds()
    return model^


def limb_skeleton() -> List[ManifestBone]:
    """The limb's manifest skeleton: the two helper joints."""
    var bones = List[ManifestBone]()
    bones.append(ManifestBone("Root", "", Vec3()))
    bones.append(ManifestBone("Upper", "Root", Vec3()))
    bones.append(ManifestBone("Forearm", "Upper", Vec3(0, 1, 0)))
    bones.append(ManifestBone("Hand", "Forearm", Vec3(0, 2, 0)))
    var half = ManifestBone("Elbow_Half", "Upper", Vec3(0, 1, 0))
    half.helper_kind = HELPER_HALF
    half.helper_source = "Forearm"
    half.helper_amount = 0.5
    bones.append(half^)
    var twist = ManifestBone("Forearm_Twist", "Forearm", Vec3(0, 1.5, 0))
    twist.helper_kind = HELPER_TWIST
    twist.helper_source = "Hand"
    twist.helper_amount = 0.5
    bones.append(twist^)
    return bones^


def mesh_volume(skinned: List[Float32], indices: List[Int], stride: Int = 8) -> Float32:
    """The volume a closed mesh encloses (positions `stride` floats apart)."""
    var total = Float64(0)
    for triangle in range(len(indices) // 3):
        var a = indices[triangle * 3] * stride
        var b = indices[triangle * 3 + 1] * stride
        var c = indices[triangle * 3 + 2] * stride
        var ax = Float64(skinned[a])
        var ay = Float64(skinned[a + 1])
        var az = Float64(skinned[a + 2])
        var bx = Float64(skinned[b])
        var by = Float64(skinned[b + 1])
        var bz = Float64(skinned[b + 2])
        var cx = Float64(skinned[c])
        var cy = Float64(skinned[c + 1])
        var cz = Float64(skinned[c + 2])
        total += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
    return Float32(abs(total) / 6.0)


def ring_radius(skinned: List[Float32], ring: Int, sides: Int, stride: Int = 8) -> Float32:
    """The mean distance of a ring's vertices from their own centre."""
    var cx = Float32(0)
    var cy = Float32(0)
    var cz = Float32(0)
    for side in range(sides):
        var at = (ring * sides + side) * stride
        cx += skinned[at]
        cy += skinned[at + 1]
        cz += skinned[at + 2]
    cx = cx / Float32(sides)
    cy = cy / Float32(sides)
    cz = cz / Float32(sides)
    var total = Float32(0)
    for side in range(sides):
        var at = (ring * sides + side) * stride
        var dx = skinned[at] - cx
        var dy = skinned[at + 1] - cy
        var dz = skinned[at + 2] - cz
        total += sqrt(dx * dx + dy * dy + dz * dz)
    return total / Float32(sides)


def thinnest_ring(skinned: List[Float32], rings: Int, sides: Int, stride: Int = 8) -> Float32:
    var least = Float32(1000000)
    for ring in range(rings):
        least = min(least, ring_radius(skinned, ring, sides, stride))
    return least
