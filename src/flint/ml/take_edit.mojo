"""Non-destructive SOMA take editing and pose-based cycle selection."""
from std.math import isfinite
from flint.ml.motionbricks import (
    MbClip,
    MbQuat,
    MbVec3,
    MbXform,
    soma_joint_names,
    world_pose,
)


def validate_take(clip: MbClip) raises:
    ref sk = clip.skeleton
    if clip.num_frames() < 2 or sk.num_joints() != 30 or sk.names[0] != "Hips":
        raise Error("Expected a SOMA30 BVH with at least two frames")
    if not isfinite(clip.sample_rate) or clip.sample_rate <= 0:
        raise Error("Invalid take sample rate")
    if len(sk.parents) != 30 or len(sk.reference) != 30 or sk.parents[0] != -1:
        raise Error("Incomplete SOMA skeleton")
    for name in soma_joint_names():
        if sk.joint_index(name) < 0:
            raise Error("Missing SOMA joint " + name)
    for j in range(1, sk.num_joints()):
        if sk.parents[j] < 0 or sk.parents[j] >= j:
            raise Error("Invalid joint hierarchy")
    for frame in clip.frames:
        if len(frame) != sk.num_joints():
            raise Error("Incomplete SOMA pose")
        for j in range(len(frame)):
            var t = frame[j]
            if not isfinite(
                t.q.x + t.q.y + t.q.z + t.q.w + t.p.x + t.p.y + t.p.z
            ):
                raise Error("Take contains a non-finite transform")
            if j > 0 and (t.p - sk.reference[j].p).length() > 0.00001:
                raise Error("Take has animated joint offsets outside Hips")


def edit_take(
    clip: MbClip,
    first: Int,
    last: Int,
    mirror: Bool,
    in_place: Bool,
    reverse: Bool,
) raises -> MbClip:
    """Both frame bounds are inclusive. Keep the original skeleton and sample rate.
    """
    if first < 0 or last >= clip.num_frames() or first >= last:
        raise Error("Select at least two frames inside the take")
    var result = MbClip()
    result.skeleton = clip.skeleton.copy()
    result.sample_rate = clip.sample_rate
    var pairs = List[Int]()
    for name in clip.skeleton.names:
        var counterpart = name.copy()
        if name.startswith("Left"):
            counterpart = "Right" + String(name[byte=4:])
        elif name.startswith("Right"):
            counterpart = "Left" + String(name[byte=5:])
        var j = clip.skeleton.joint_index(counterpart)
        if j < 0:
            raise Error("Missing mirror joint " + counterpart)
        pairs.append(j)
    var anchor = clip.frames[last if reverse else first][0].p
    for step in range(last - first + 1):
        var f = last - step if reverse else first + step
        var pose = clip.frames[f].copy()
        if mirror:
            for j in range(len(pose)):
                var q = clip.frames[f][pairs[j]].q
                pose[j].q = MbQuat(q.x, -q.y, -q.z, q.w)
            pose[0].p.x = 2 * anchor.x - pose[0].p.x
        if in_place:
            pose[0].p.x = anchor.x
            pose[0].p.z = anchor.z
        result.frames.append(pose^)
    return result^


def _squared(v: MbVec3) -> Float32:
    return v.x * v.x + v.y * v.y + v.z * v.z


def find_cycle(clip: MbClip, first: Int, last: Int) raises -> Tuple[Int, Int]:
    """Find matching poses and velocities 0.5–2 seconds apart. Omit the repeated endpoint.
    """
    var min_gap = max(2, Int(clip.sample_rate * 0.5 + 0.5))
    var max_gap = min(last - first, max(min_gap, Int(clip.sample_rate * 2)))
    if first < 0 or last >= clip.num_frames() or last - first < min_gap:
        raise Error("Loop search needs at least half a second of motion")
    var joints = List[Int]()
    var names: List[String] = [
        "Hips",
        "Head",
        "LeftHand",
        "RightHand",
        "LeftFoot",
        "RightFoot",
        "LeftShin",
        "RightShin",
    ]
    for name in names:
        var j = clip.skeleton.joint_index(name)
        if j < 0:
            raise Error("Loop search needs " + name)
        joints.append(j)
    # Remove planar travel for the pose comparison. Keep hip height and heading.
    var features = List[List[MbVec3]]()
    var velocities = List[List[MbVec3]]()
    for f in range(first, last + 1):
        var world = world_pose(
            clip.skeleton, clip.frames[f], MbXform.identity()
        )
        var root = world[0].p
        var pts = List[MbVec3]()
        for j in joints:
            pts.append(world[j].p - MbVec3(root.x, 0, root.z))
        features.append(pts^)
    for f in range(len(features)):
        var prev = max(0, f - 1)
        var after = min(len(features) - 1, f + 1)
        var vel = List[MbVec3]()
        # Compare displacement over a tenth of a second to weight limb direction.
        var scale = clip.sample_rate * 0.1 / Float32(after - prev)
        for j in range(len(joints)):
            vel.append((features[after][j] - features[prev][j]) * scale)
        velocities.append(vel^)
    var best = Float64(1e100)
    var start = 0
    var stop = min_gap
    for a in range(0, len(features) - min_gap, max(1, len(features) // 512)):
        for b in range(a + min_gap, min(len(features), a + max_gap + 1)):
            var score: Float64 = 0
            for j in range(len(joints)):
                score += Float64(
                    _squared(features[a][j] - features[b][j])
                    + _squared(velocities[a][j] - velocities[b][j])
                )
            if score < best:
                best = score
                start = a
                stop = b
    return (first + start, first + stop - 1)
