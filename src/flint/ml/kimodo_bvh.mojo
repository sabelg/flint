"""Write Kimodo's local SOMA transforms as a centimetre, Z-Y-X BVH clip."""
from std.math import asin, atan2, isfinite
from flint.ml.motionbricks import MbClip, MbSkeleton, MbQuat
from flint.platform.os import write_file_durable


def _joint(
    s: MbSkeleton,
    j: Int,
    indent: String,
    mut text: String,
    mut traversal: List[Int],
):
    traversal.append(j)
    text += (
        indent
        + ("ROOT " if j == 0 else "JOINT ")
        + s.names[j]
        + "\n"
        + indent
        + "{\n"
    )
    var p = s.reference[j].p * 100
    text += (
        indent
        + "  OFFSET "
        + String(p.x)
        + " "
        + String(p.y)
        + " "
        + String(p.z)
        + "\n"
    )
    text += (
        indent
        + "  CHANNELS "
        + ("6 Xposition Yposition Zposition " if j == 0 else "3 ")
        + "Zrotation Yrotation Xrotation\n"
    )
    var leaf = True
    for child in range(j + 1, s.num_joints()):
        if s.parents[child] == j:
            leaf = False
            _joint(s, child, indent + "  ", text, traversal)
    if leaf:
        text += (
            indent
            + "  End Site\n"
            + indent
            + "  {\n"
            + indent
            + "    OFFSET 0 0 0\n"
            + indent
            + "  }\n"
        )
    text += indent + "}\n"


def write_kimodo_bvh(clip: MbClip, path: String) raises:
    var expected_joints = clip.skeleton.num_joints()
    if (
        expected_joints != 30
        or not clip.num_frames()
        or len(clip.skeleton.parents) != expected_joints
        or len(clip.skeleton.reference) != expected_joints
    ):
        raise Error("Expected a SOMA30 clip")
    if not isfinite(clip.sample_rate) or clip.sample_rate <= 0:
        raise Error("Invalid clip sample rate")
    var frame_time = 1.0 / Float64(clip.sample_rate)
    if not isfinite(frame_time) or frame_time <= 0.0:
        raise Error("Invalid clip sample rate")
    for pose in clip.frames:
        if len(pose) < expected_joints:
            raise Error("Incomplete pose frame in write_kimodo_bvh")
    var text = String("HIERARCHY\n")
    var traversal = List[Int]()
    _joint(clip.skeleton, 0, "", text, traversal)
    if len(traversal) != expected_joints:
        raise Error("Incomplete SOMA30 skeleton hierarchy")
    text += (
        "MOTION\nFrames: "
        + String(clip.num_frames())
        + "\nFrame Time: "
        + String(frame_time)
        + "\n"
    )
    for pose in clip.frames:
        var p = pose[0].p * 100
        text += String(p.x) + " " + String(p.y) + " " + String(p.z)
        for joint in traversal:
            var q = pose[joint].q
            var sy = max(
                Float32(-1), min(Float32(1), 2 * (q.w * q.y - q.z * q.x))
            )
            var x = atan2(
                2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y)
            )
            var z = atan2(
                2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
            )
            if abs(sy) >= 0.9999999:
                x = 0
                z = 2 * atan2(q.z, q.w)
            text += (
                " "
                + String(z * 57.2957795)
                + " "
                + String(asin(sy) * 57.2957795)
                + " "
                + String(x * 57.2957795)
            )
        text += "\n"
    var bytes = List[UInt8](text.as_bytes())
    if not write_file_durable(path, bytes):
        raise Error("Could not write " + path)
