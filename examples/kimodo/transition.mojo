"""Compose two SOMA takes with the native MotionBricks inbetweener."""
from flint.ml.motionbricks import MbClip, MbXform, MotionBricks
from flint.platform.os import env_now, file_size


def model_path() -> String:
    var path = env_now("FLINT_MOTIONBRICKS_MODEL")
    if not path:
        var checkout = env_now("MOTIONBRICKS_DIR")
        if checkout:
            path = (
                checkout
                + "/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx"
            )
        else:
            path = (
                env_now("HOME")
                + "/.cache/flint/motionbricks/mmm_v6_soma_fp32.onnx"
            )
    return path^


def join_takes(start: MbClip, target: MbClip) raises -> MbClip:
    if start.num_frames() < 4 or target.num_frames() < 4:
        raise Error("A transition needs at least four frames in each take")
    if (
        abs(start.sample_rate - 30) > 0.01
        or abs(target.sample_rate - 30) > 0.01
    ):
        raise Error("MotionBricks transitions need 30 Hz takes")
    var mapping = List[Int]()
    for j in range(start.skeleton.num_joints()):
        var k = target.skeleton.joint_index(start.skeleton.names[j])
        if k < 0:
            raise Error("Takes use different skeletons")
        var parent = start.skeleton.parents[j]
        if target.skeleton.parents[k] != (
            -1 if parent < 0 else mapping[parent]
        ):
            raise Error("Takes use different joint hierarchies")
        if (
            j
            and (
                start.skeleton.reference[j].p - target.skeleton.reference[k].p
            ).length()
            > 0.001
        ):
            raise Error("Takes use different bone lengths or rest poses")
        mapping.append(k)
    var path = model_path()
    if file_size(path) < 1000:
        raise Error("Set FLINT_MOTIONBRICKS_MODEL to the SOMA fp32 ONNX file")
    var context = List[List[MbXform]]()
    for f in range(start.num_frames() - 4, start.num_frames()):
        context.append(start.frames[f].copy())
    # Align floor position only. Keeping hip height and heading allows a crouch or turn.
    var delta = context[3][0].p - target.frames[0][0].p
    delta.y = 0
    var aligned = List[List[MbXform]]()
    for frame in target.frames:
        var pose = List[MbXform]()
        for k in mapping:
            pose.append(frame[k])
        pose[0].p = pose[0].p + delta
        aligned.append(pose^)
    var guide = List[List[MbXform]]()
    for f in range(4):
        guide.append(aligned[f].copy())
    var generator = MotionBricks.load(path, start.skeleton)
    var transition = generator.generate(context, guide)
    var result = start.copy()
    for f in range(1, transition.num_frames() - 1):
        result.frames.append(transition.poses[f].copy())
    for f in range(len(aligned)):
        result.frames.append(aligned[f].copy())
    return result^
