"""Bake a SOMA30 take onto a Y-up, unit-scale Mixamo humanoid skin."""
from flint.assets.model import NativeAnimation, NativeChannel, NativeModel
from flint.core.ik import shortest_arc_quat
from flint.core.math import Quat, Vec3
from flint.ml.motionbricks import (
    MbClip,
    MbQuat,
    MbVec3,
    MbXform,
    quat_from_basis,
    world_pose,
)


from flint.ml.take_edit import validate_take


def _vec(v: MbVec3) -> Vec3:
    return Vec3(v.x, v.y, v.z)


def _quat(q: MbQuat) -> Quat:
    return Quat(q.x, q.y, q.z, q.w)


def _basis(across: Vec3, up: Vec3) raises -> Quat:
    var x = across.normalized()
    var z = x.cross(up).normalized()
    if z.length_squared() < 0.9:
        raise Error(
            "Cannot determine rig axes from the hip and spine rest pose"
        )
    var y = z.cross(x)
    return _quat(
        quat_from_basis(
            MbVec3(x.x, x.y, x.z), MbVec3(y.x, y.y, y.z), MbVec3(z.x, z.y, z.z)
        )
    )


def _bone(model: NativeModel, name: String) raises -> Int:
    var matched = -1
    for skin in model.skins:
        for j in skin.joints:
            if j < 0 or j >= len(model.nodes):
                raise Error("Invalid skin joint")
            var parts = model.nodes[j].name.split(":")
            if parts[len(parts) - 1] == name:
                if matched >= 0 and matched != j:
                    raise Error("Ambiguous Mixamo bone " + name)
                matched = j
    if matched < 0:
        raise Error("Missing Mixamo skin bone " + name)
    return matched


def mapped_bone(
    model: NativeModel, soma_name: String, aliases: Dict[String, String]
) raises -> Int:
    var target = soma_name.copy()
    if soma_name in aliases:
        target = aliases[soma_name]
    elif soma_name == "Spine1":
        target = "Spine"
    elif soma_name == "Spine2":
        target = "Spine1"
    elif soma_name == "Chest":
        target = "Spine2"
    elif soma_name == "Neck1":
        target = "Neck"
    elif soma_name.endswith("Shin"):
        target = soma_name.replace("Shin", "Leg")
    elif soma_name.endswith("Leg"):
        target = soma_name.replace("Leg", "UpLeg")
    return _bone(model, target)


def retarget_soma(
    clip: MbClip,
    mut model: NativeModel,
    aliases: Dict[String, String] = Dict[String, String](),
    stage_origin: Bool = False,
) raises -> NativeAnimation:
    """Author local rotation tracks and one hip translation track using both rest poses.
    """
    validate_take(clip)
    model.ensure_parents()
    var src_names: List[String] = [
        "Hips",
        "Spine1",
        "Spine2",
        "Chest",
        "Neck1",
        "Head",
    ]
    # A guide supplies the direction to a child joint and preserves the source twist.
    var guides: List[Int] = [-1, 2, 3, 4, 5, -1]
    for s in [String("Left"), String("Right")]:
        var start = len(src_names)
        for name in [
            String("Shoulder"),
            String("Arm"),
            String("ForeArm"),
            String("Hand"),
            String("Leg"),
            String("Shin"),
            String("Foot"),
            String("ToeBase"),
        ]:
            src_names.append(s + name)
        for k in range(8):
            guides.append(-1 if k == 3 or k == 7 else start + k + 1)
    for name in aliases.keys():
        if name not in src_names:
            raise Error("Unknown SOMA source bone " + name)
    var src = List[Int]()
    var dst = List[Int]()
    var map_index = List[Int](length=len(model.nodes), fill=-1)
    for k in range(len(src_names)):
        var j = clip.skeleton.joint_index(src_names[k])
        if j < 0:
            raise Error("Missing SOMA joint " + src_names[k])
        src.append(j)
        var bone = mapped_bone(model, src_names[k], aliases)
        if bone in dst:
            raise Error("Duplicate destination bone for " + src_names[k])
        dst.append(bone)
        map_index[dst[k]] = k
    var sr = world_pose(
        clip.skeleton, clip.skeleton.reference, MbXform.identity()
    )
    var dr = List[Quat](length=len(model.nodes), fill=Quat())
    var dp = List[Vec3](length=len(model.nodes), fill=Vec3())
    for n in model.topo_order:
        var p = model.parents[n]
        var parent_rot = Quat() if p < 0 else dr[p]
        dr[n] = parent_rot.multiply(model.nodes[n].base_rotation).normalized()
        dp[n] = parent_rot.rotate(model.nodes[n].base_translation) + (
            Vec3() if p < 0 else dp[p]
        )
    for n in dst:
        var j = n
        var ancestors = 0
        while j >= 0:
            ancestors += 1
            if ancestors > len(model.nodes):
                raise Error("Cyclic target bone hierarchy")
            if (
                model.nodes[j].has_matrix
                or (model.nodes[j].base_scale - Vec3(1, 1, 1)).length() > 0.0001
            ):
                raise Error(
                    "Apply rig scale and convert matrix nodes to TRS before"
                    " retargeting"
                )
            j = model.parents[j]
    var hips = dst[0]
    var left = src_names.index("LeftLeg")
    var right = src_names.index("RightLeg")
    var axes = _basis(
        dp[dst[left]] - dp[dst[right]], dp[dst[4]] - dp[hips]
    ).multiply(
        _basis(
            _vec(sr[src[left]].p - sr[src[right]].p),
            _vec(sr[src[4]].p - sr[src[0]].p),
        ).conjugate()
    )
    var target_leg: Float32 = 0
    var source_leg: Float32 = 0
    for k in [left, right]:
        for i in range(k, k + 2):
            target_leg += (dp[dst[i + 1]] - dp[dst[i]]).length()
            source_leg += (sr[src[i + 1]].p - sr[src[i]].p).length()
    if min(source_leg, target_leg) < 0.0001:
        raise Error("Degenerate leg lengths in retarget rig")
    var scale = target_leg / source_leg
    var floor = min(sr[src[left + 3]].p.y, sr[src[right + 3]].p.y)
    var rest_height = sr[src[0]].p.y - floor
    var anchor = clip.frames[0][src[0]].p
    if stage_origin:
        anchor = MbVec3(0, 0, 0)
    var anim = NativeAnimation()
    anim.name = "Retargeted take"
    anim.duration = clip.duration()
    for k in range(len(src) + 1):
        var channel = NativeChannel()
        channel.node = hips if k == len(src) else dst[k]
        channel.path = 0 if k == len(src) else 1
        anim.channels.append(channel^)
    var world = dr.copy()
    for f in range(clip.num_frames()):
        var pose = world_pose(clip.skeleton, clip.frames[f], MbXform.identity())
        var t = Float32(f) / clip.sample_rate
        for n in model.topo_order:
            var p = model.parents[n]
            var parent_rot = Quat() if p < 0 else world[p]
            var k = map_index[n]
            if k < 0:
                world[n] = parent_rot.multiply(model.nodes[n].base_rotation)
                continue
            var j = src[k]
            var delta = (
                axes.multiply(_quat(pose[j].q))
                .multiply(_quat(sr[j].q).conjugate())
                .multiply(axes.conjugate())
            )
            var g = guides[k]
            if g >= 0:
                var actual = delta.rotate(dp[dst[g]] - dp[n])
                var desired = axes.rotate(_vec(pose[src[g]].p - pose[j].p))
                delta = shortest_arc_quat(actual, desired).multiply(delta)
            world[n] = delta.multiply(dr[n]).normalized()
            var local = parent_rot.conjugate().multiply(world[n]).normalized()
            ref ch = anim.channels[k]
            var i = len(ch.values) - 4
            if (
                i >= 0
                and local.x * ch.values[i]
                + local.y * ch.values[i + 1]
                + local.z * ch.values[i + 2]
                + local.w * ch.values[i + 3]
                < 0
            ):
                local = Quat(-local.x, -local.y, -local.z, -local.w)
            ch.times.append(t)
            ch.values.extend([local.x, local.y, local.z, local.w])
        var h = pose[src[0]].p
        var travel = (
            axes.rotate(Vec3(h.x - anchor.x, h.y - rest_height, h.z - anchor.z))
            * scale
        )
        var parent = model.parents[hips]
        var pos = model.nodes[hips].base_translation + (
            dr[parent].conjugate().rotate(travel) if parent >= 0 else travel
        )
        anim.channels[len(src)].times.append(t)
        anim.channels[len(src)].values.extend([pos.x, pos.y, pos.z])
    return anim^
