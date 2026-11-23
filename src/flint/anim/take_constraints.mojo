"""World-space authoring constraints for SOMA takes. Distances are metres."""
from std.math import sqrt, atan2, isfinite
from flint.core.math import Quat, Vec3
from flint.core.ik import shortest_arc_quat
from flint.ml.motionbricks import (
    MbClip,
    MbVec3,
    MbQuat,
    MbXform,
    world_pose,
    quat_axis_angle,
    quat_slerp,
)


def vec(v: MbVec3) -> Vec3:
    return Vec3(v.x, v.y, v.z)


def mb(v: Vec3) -> MbVec3:
    return MbVec3(v.x, v.y, v.z)


def position(clip: MbClip, frame: Int, j: Int) -> Vec3:
    if frame < 0 or frame >= len(clip.frames):
        return Vec3()
    var nj = len(clip.frames[frame])
    if j < 0 or j >= nj or len(clip.skeleton.parents) < nj:
        return Vec3()
    return vec(
        world_pose(clip.skeleton, clip.frames[frame], MbXform.identity())[j].p
    )


def _turn(
    mut clip: MbClip,
    f: Int,
    j: Int,
    world: List[MbXform],
    before: Vec3,
    after: Vec3,
):
    if (
        f < 0
        or f >= len(clip.frames)
        or j < 0
        or j >= len(clip.frames[f])
        or j >= len(clip.skeleton.parents)
        or j >= len(world)
    ):
        return
    if min(before.length_squared(), after.length_squared()) < 1e-10:
        return
    var u = before.normalized()
    var v = after.normalized()
    var cross = u.cross(v)
    var cosine = u.dot(v)
    var d = shortest_arc_quat(before, after)
    if cosine > -0.9999:

        d = Quat(cross.x, cross.y, cross.z, 1 + cosine).normalized()
    var parent = clip.skeleton.parents[j]
    var pq = MbQuat.identity() if (parent < 0 or parent >= len(world)) else world[parent].q
    clip.frames[f][j].q = (
        pq.inverse() * MbQuat(d.x, d.y, d.z, d.w) * world[j].q
    ).normalized()


def aim_joint(mut clip: MbClip, f: Int, j: Int, child: Int, direction: Vec3):
    if not (isfinite(direction.x) and isfinite(direction.y) and isfinite(direction.z)):
        return
    if f < 0 or f >= len(clip.frames):
        return
    var nj = len(clip.frames[f])
    if j < 0 or j >= nj or child < 0 or child >= nj or len(clip.skeleton.parents) < nj:
        return
    var w = world_pose(clip.skeleton, clip.frames[f], MbXform.identity())
    if j >= len(w) or child >= len(w):
        return
    _turn(clip, f, j, w, vec(w[child].p - w[j].p), direction)


def pin_frame(mut clip: MbClip, f: Int, tip: Int, goal: Vec3):
    if not (isfinite(goal.x) and isfinite(goal.y) and isfinite(goal.z)):
        return
    if f < 0 or f >= len(clip.frames):
        return
    var nj = len(clip.frames[f])
    if tip < 0 or tip >= nj or tip >= len(clip.skeleton.parents) or len(clip.skeleton.parents) < nj:
        return
    var mid = clip.skeleton.parents[tip]
    if mid < 0 or mid >= tip or mid >= nj or mid >= len(clip.skeleton.parents):
        return
    var upper = clip.skeleton.parents[mid]
    if upper < 0 or upper >= mid or upper >= nj or upper >= len(clip.skeleton.parents):
        return
    var w = world_pose(clip.skeleton, clip.frames[f], MbXform.identity())
    if upper >= len(w) or mid >= len(w) or tip >= len(w):
        return
    var a = vec(w[upper].p)
    var b = vec(w[mid].p)
    var c = vec(w[tip].p)
    var diff = goal - a
    if not (isfinite(diff.x) and isfinite(diff.y) and isfinite(diff.z)):
        return
    var u = diff.normalized()
    if u.length_squared() < 0.5:
        u = (c - a).normalized()
    var ab = (b - a).length()
    var bc = (c - b).length()
    var dist = max(
        abs(ab - bc) + 0.00001, min(ab + bc - 0.00001, (goal - a).length())
    )
    if min(ab, bc) < 0.0001:
        return
    var pole = (b - a) - u * (b - a).dot(u)
    if pole.length_squared() < 1e-8:
        var axis = Vec3(0, 0, 1) if abs(u.z) < 0.9 else Vec3(1, 0, 0)
        pole = axis - u * u.dot(axis)
    var along = (ab * ab - bc * bc + dist * dist) / (2 * dist)
    var elbow = u * along + pole.normalized() * sqrt(
        max(Float32(0), ab * ab - along * along)
    )
    _turn(clip, f, upper, w, b - a, elbow)
    aim_joint(clip, f, mid, tip, goal - position(clip, f, mid))


def smooth(t: Float32) -> Float32:
    var v = max(Float32(0), min(Float32(1), t))
    return v * v * (3 - 2 * v)


def check_interval(clip: MbClip, start: Int, end: Int) raises:
    if start < 0 or end < start or end >= clip.num_frames():
        raise Error("Frame range outside take (zero-based, inclusive)")


def pin_joint(
    mut clip: MbClip,
    name: String,
    start: Int,
    end: Int,
    goal: Vec3,
    fade: Int,
    weight: Float32,
) raises -> Float32:
    check_interval(clip, start, end)
    if not (isfinite(goal.x) and isfinite(goal.y) and isfinite(goal.z)):
        raise Error("Pin goal must be finite")
    if (
        name not in ("LeftFoot", "RightFoot", "LeftHand", "RightHand")
        or fade < 0
        or not isfinite(weight)
        or weight < 0
        or weight > 1
    ):
        raise Error(
            "Pins need a hand or foot, a nonnegative fade and weight 0..1"
        )
    var j = clip.skeleton.joint_index(name)
    if j < 0 or j >= len(clip.skeleton.parents):
        raise Error("Unknown pin joint " + name)
    var mid = clip.skeleton.parents[j]
    if mid < 0 or mid >= j or mid >= len(clip.skeleton.parents):
        raise Error("Pin joint requires a 2-bone parent chain: " + name)
    var upper = clip.skeleton.parents[mid]
    if upper < 0 or upper >= mid or upper >= len(clip.skeleton.parents):
        raise Error("Pin joint requires a 2-bone parent chain: " + name)
    var peak: Float32 = 0
    for f in range(
        max(0, start - fade), min(clip.num_frames(), end + fade + 1)
    ):
        var outside = max(start - f, max(0, f - end))
        var blend = weight * smooth(1 - Float32(outside) / Float32(fade + 1))
        var p = position(clip, f, j)
        pin_frame(clip, f, j, p + (goal - p) * blend)
        if start <= f and f <= end:
            peak = max(peak, (position(clip, f, j) - goal).length())
    return peak


def clean_stances(
    mut clip: MbClip, speed: Float32, height: Float32, minimum: Int
) raises -> Int:
    if not isfinite(speed) or speed <= 0 or not isfinite(height) or height <= 0 or minimum < 2:
        raise Error("Invalid stance thresholds")
    if clip.num_frames() < 2 or not isfinite(clip.sample_rate) or clip.sample_rate <= 0:
        return 0
    var count = 0
    for name in [String("LeftFoot"), String("RightFoot")]:
        var j = clip.skeleton.joint_index(name)
        if j < 0 or j >= len(clip.skeleton.parents):
            continue
        var mid = clip.skeleton.parents[j]
        if mid < 0 or mid >= j or mid >= len(clip.skeleton.parents):
            continue
        var upper = clip.skeleton.parents[mid]
        if upper < 0 or upper >= mid or upper >= len(clip.skeleton.parents):
            continue
        var p = List[Vec3]()
        var floor = Float32(1e20)
        for f in range(clip.num_frames()):
            if j >= len(clip.frames[f]):
                p.clear()
                break
            var v = position(clip, f, j)
            p.append(v)
            floor = min(floor, v.y)
        if len(p) < minimum:
            continue
        var first = -1
        for f in range(len(p) + 1):
            var stance = False
            if f < len(p):
                var lo = max(0, f - 1)
                var hi = min(len(p) - 1, f + 1)
                stance = (
                    p[f].y <= floor + height
                    and (p[hi] - p[lo]).length()
                    * clip.sample_rate
                    / Float32(hi - lo)
                    <= speed
                )
            if stance and first < 0:
                first = f
            elif not stance and first >= 0:
                if f - first >= minimum:
                    _ = pin_joint(clip, name, first, f - 1, p[first], 2, 1)
                    count += 1
                first = -1
    return count


def warp_contact(
    mut clip: MbClip,
    joint: String,
    start: Int,
    end: Int,
    target: Vec3,
    limit: Float32,
) raises:
    check_interval(clip, start, end)
    var j = clip.skeleton.joint_index(joint)
    if j < 0 or start == end or not isfinite(limit) or limit <= 0:
        raise Error(
            "Warp needs a known joint, an approach interval and positive reach"
        )
    var offset = target - position(clip, end, j)
    if (
        not isfinite(offset.x)
        or not isfinite(offset.y)
        or not isfinite(offset.z)
        or offset.length() > limit
    ):
        raise Error("Warp target exceeds maximum correction")
    for f in range(start, clip.num_frames()):
        if len(clip.frames[f]) > 0:
            clip.frames[f][0].p = clip.frames[f][0].p + mb(offset) * smooth(
                Float32(f - start) / Float32(end - start)
            )


def follow_path(mut clip: MbClip, frames: List[Int], points: List[Vec3]) raises:
    if clip.num_frames() < 2:
        raise Error("Path needs a take with at least two frames")
    if (
        len(frames) < 2
        or len(frames) != len(points)
        or frames[0] != 0
        or frames[len(frames) - 1] != clip.num_frames() - 1
    ):
        raise Error("Path needs increasing frame markers covering the take")
    for i in range(len(points)):
        var pt = points[i]
        if not (isfinite(pt.x) and isfinite(pt.y) and isfinite(pt.z)):
            raise Error("Path points must be finite")
        if frames[i] < 0 or frames[i] >= clip.num_frames():
            raise Error("Invalid path frame order")
        if i > 0 and frames[i] <= frames[i - 1]:
            raise Error("Invalid path frame order")
    var up = MbVec3(0, 1, 0)
    var original = List[MbVec3]()
    for pose in clip.frames:
        if len(pose) == 0:
            raise Error("Path requires non-empty poses for every frame")
        original.append(pose[0].p)
    var headings = List[MbQuat]()
    for i in range(len(points) - 1):
        var a = frames[i]
        var b = frames[i + 1]
        if b <= a or b >= clip.num_frames():
            raise Error("Invalid path frame order")
        var old = original[b] - original[a]
        var d = points[i + 1] - points[i]
        var yaw = atan2(d.x, d.z) - atan2(old.x, old.z)
        headings.append(
            quat_axis_angle(up, yaw) if d.horizontal().length()
            > 0.0001 else MbQuat.identity()
        )
    for i in range(len(headings)):
        var a = frames[i]
        var b = frames[i + 1]
        var old = original[b] - original[a]
        var d = points[i + 1] - points[i]
        var blend_frames = min(4, (b - a) // 2)
        for f in range(a, b + Int(i == len(points) - 2)):
            var t = Float32(f - a) / Float32(b - a)
            var rot = headings[i]
            if blend_frames > 0 and i > 0 and f - a < blend_frames:
                rot = quat_slerp(
                    headings[i - 1],
                    rot,
                    0.5 + 0.5 * smooth(Float32(f - a) / Float32(blend_frames)),
                )
            elif (
                blend_frames > 0
                and i + 1 < len(headings)
                and b - f < blend_frames
            ):
                rot = quat_slerp(
                    rot,
                    headings[i + 1],
                    0.5 * smooth(1 - Float32(b - f) / Float32(blend_frames)),
                )
            var sway = original[f] - (original[a] + old * t)
            sway.y = 0
            var center = points[i] + d * t
            var p = rot.rotate(sway) + MbVec3(
                center.x, original[f].y + center.y, center.z
            )
            clip.frames[f][0].p = p
            clip.frames[f][0].q = (rot * clip.frames[f][0].q).normalized()
