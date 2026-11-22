"""Regression tests for native Kimodo sampling, BVH export and real model compatibility."""
from std.testing import assert_equal, assert_true
from std.ffi import external_call
from flint.ml.kimodo import Kimodo, _normal_noise, _time, _alphas
from flint.ml.kimodo_text import KimodoTokenizer, KimodoTextEncoder
from flint.ml.kimodo_bvh import write_kimodo_bvh
from std.math import sin
from flint.ml.motionbricks import (
    MbClip,
    MbVec3,
    MbXform,
    load_bvh,
    quat_axis_angle,
    soma_joint_names,
)
from flint.ml.take_edit import edit_take, find_cycle, validate_take
from flint.ml.ops import U_ERF, U_SIGMOID, unary
from flint.ml.tensor import Tensor
from flint.platform.os import env_now, remove_file


def test_motion_contacts() raises:
    from flint.anim.take_constraints import (
        clean_stances,
        position,
        follow_path,
        warp_contact,
    )
    from flint.core.math import Vec3

    var clip = MbClip()
    clip.sample_rate = 30
    clip.skeleton.names = [
        "Hips",
        "LeftLeg",
        "LeftShin",
        "LeftFoot",
        "RightLeg",
        "RightShin",
        "RightFoot",
    ]
    clip.skeleton.parents = [-1, 0, 1, 2, 0, 4, 5]
    for offset in [
        MbVec3(0, 1, 0),
        MbVec3(0.12, 0, 0),
        MbVec3(0, -0.45, 0.12),
        MbVec3(0, -0.45, -0.12),
        MbVec3(-0.12, 0, 0),
        MbVec3(0, -0.45, 0.12),
        MbVec3(0, -0.45, -0.12),
    ]:
        var xf = MbXform.identity()
        xf.p = offset
        clip.skeleton.reference.append(xf)
    for f in range(15):
        var pose = clip.skeleton.reference.copy()
        pose[0].p.x = Float32(f) * 0.002
        clip.frames.append(pose^)
    var anchor = position(clip, 0, 3)
    assert_true((position(clip, 14, 3) - anchor).length() > 0.025)
    assert_equal(clean_stances(clip, 0.1, 0.02, 3), 2)
    for f in range(15):
        assert_true((position(clip, f, 3) - anchor).length() < 1e-5)
    follow_path(
        clip,
        [0, 7, 14],
        [Vec3(0, 0, 0), Vec3(0.4, 0.2, 0), Vec3(0.4, 0.1, 0.4)],
    )
    assert_true((position(clip, 7, 0) - Vec3(0.4, 1.2, 0)).length() < 1e-5)
    assert_true((position(clip, 14, 0) - Vec3(0.4, 1.1, 0.4)).length() < 1e-5)
    var target = position(clip, 10, 3) + Vec3(0.1, 0.03, 0.05)
    var start = position(clip, 2, 0)
    warp_contact(clip, "LeftFoot", 2, 10, target, 0.3)
    assert_true((position(clip, 10, 3) - target).length() < 1e-5)
    assert_true((position(clip, 2, 0) - start).length() < 1e-5)
    var rejected = False
    try:
        warp_contact(clip, "LeftFoot", 2, 10, target + Vec3(3, 0, 0), 0.1)
    except:
        rejected = True
    assert_true(rejected)


def test_take_edits() raises:
    var clip = MbClip()
    clip.sample_rate = 24
    clip.skeleton.names = soma_joint_names()
    for j in range(30):
        clip.skeleton.parents.append(-1 if j == 0 else 0)
        clip.skeleton.reference.append(MbXform.identity())
    var left = clip.skeleton.joint_index("LeftHand")
    var right = clip.skeleton.joint_index("RightHand")
    for f in range(34):
        var pose = clip.skeleton.reference.copy()
        var phase = sin(Float32(f % 18) * 0.34906585)
        pose[0].p = MbVec3(
            Float32(f) * 0.01, 1 + 0.1 * phase, Float32(f) * 0.02
        )
        pose[left].q = quat_axis_angle(MbVec3(0, 1, 0), 0.5 + 0.2 * phase)
        clip.frames.append(pose^)
    validate_take(clip)
    var cycle = find_cycle(clip, 0, 33)
    assert_equal(cycle[1] - cycle[0] + 1, 18)
    var edited = edit_take(clip, 3, 21, True, True, True)
    assert_equal(edited.num_frames(), 19)
    for f in range(19):
        var src = clip.frames[21 - f][0].p
        var root = edited.frames[f][0].p
        assert_equal(root.x, clip.frames[21][0].p.x)
        assert_equal(root.z, clip.frames[21][0].p.z)
        assert_equal(root.y, src.y)
        assert_equal(
            edited.frames[f][right].q.y, -clip.frames[21 - f][left].q.y
        )
    var once = edit_take(clip, 0, 33, True, False, False)
    var twice = edit_take(once, 0, 33, True, False, False)
    assert_true((twice.frames[33][0].p - clip.frames[33][0].p).length() < 1e-6)
    assert_equal(twice.frames[33][left].q.y, clip.frames[33][left].q.y)
    var path = (
        "/tmp/flint-24hz-" + String(external_call["getpid", Int32]()) + ".bvh"
    )
    write_kimodo_bvh(edited, path)
    var loaded = load_bvh(path)
    _ = remove_file(path)
    assert_equal(loaded.sample_rate, 24)
    assert_equal(loaded.num_frames(), 19)


def test_kimodo() raises:
    test_take_edits()
    test_motion_contacts()
    var noise = _normal_noise(5, 61)
    assert_equal(len(noise), 61)
    assert_equal(noise, _normal_noise(5, 61))
    assert_true(noise != _normal_noise(6, 61))
    assert_equal(_time(0, 25), 0)
    assert_equal(_time(24, 25), 999)
    assert_equal(_time(1, 3), 500)
    var a = _alphas()
    assert_true(abs(a[999] - Float32(2.4287e-9)) < 1e-12)
    var x = Tensor.floats([3], [-1, 0, 1])
    var e = unary(U_ERF, x)
    var s = unary(U_SIGMOID, x)
    assert_true(abs(e.fget(2) - 0.8427008) < 1e-6)
    assert_true(abs(s.fget(0) - 0.2689414) < 1e-6)
    var dir = env_now("FLINT_KIMODO_DIR")
    if not dir:
        return
    var tokenizer = KimodoTokenizer(dir + "/tokenizer.json")
    var ids = tokenizer.encode("A person walks forward.")
    var expected: List[Int] = [128000, 32, 1732, 23291, 4741, 13]
    assert_equal(ids, expected)
    var encoder = KimodoTextEncoder(
        dir + "/model_int4_b128_fp32act_s64.onnx_data"
    )
    var embedding = encoder.encode(ids)
    assert_equal(len(embedding), 4096)
    var k = Kimodo(dir)
    var clip = k.sample(embedding, 8, 3, 5)
    var file = (
        "/tmp/flint-kimodo-" + String(external_call["getpid", Int32]()) + ".bvh"
    )
    write_kimodo_bvh(clip, file)
    var imported = load_bvh(file)
    _ = remove_file(file)
    assert_equal(imported.num_frames(), 8)
    assert_equal(imported.skeleton.names, clip.skeleton.names)
    assert_equal(imported.skeleton.parents, clip.skeleton.parents)
    for i in range(8):
        for j in range(30):
            var a1 = clip.frames[i][j]
            var b = imported.frames[i][j]
            assert_true((a1.p - b.p).length() < 1e-5)
            assert_true(abs((a1.q.inverse() * b.q).w) > 0.99999)


def main() raises:
    test_kimodo()
    print("Kimodo checks passed")
