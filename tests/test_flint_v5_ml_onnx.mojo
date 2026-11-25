"""Flint v5 `flint.ml` suite: pure-Mojo ONNX interpreter, MAX GPU matrix products and the
MotionBricks SOMA runtime (`test_flint_v5_ml_onnx.mojo`).

Verifies:
  1. Broadcasting offsets and `binary` fast paths against element-by-element numpy rules.
  2. `tests/fixtures/onnx/tiny_ops.onnx` (MatMul, Add, Relu, LayerNormalization, Transpose, dilated
     Conv, Reshape, Gemm transB, Softmax, LeakyRelu, Gather, ReduceSum, Atan, Slice, Shape, Cast,
     Concat, TopK, ArgMax, Greater, Where, Unsqueeze) reproduces onnxruntime's outputs on the CPU,
     and again with the matrix products on a MAX GPU when one is present.
  3. `tiny_fused.onnx` (`make_tiny_fused.py`): the fusions `OnnxModel` plans — sparse-conv
     Gather -> Reshape -> MatMul(Transpose(Reshape(weight))) as one gather-GEMM on the GPU and
     Einsum -> Mul -> Softmax -> Einsum attention (CPU kernel and GPU kernel) — plus
     BatchNormalization, against onnxruntime.
  4. MotionBricks math: Warp-compatible quaternion slerp / basis conversion, and the endpoint
     correction lands the generated motion exactly on the last context and first target pose.
  5. With `MOTIONBRICKS_DIR` set to a MotionBricks checkout: the SOMA model generates a walk from
     its own BVH clip, choosing an allowed token count, with finite poses that start on the context.
"""
from std.math import abs, isnan, sqrt
from std.testing import assert_true, assert_equal
from flint.ml.onnx import OnnxModel
from flint.ml.tensor import Tensor, DT_F32, DT_I64, broadcast_offsets
from flint.ml.ops import binary, B_ADD, B_MUL
from flint.ml.motionbricks import (
    MotionBricks, MbConfig, MbMotion, MbQuat, MbVec3, MbXform, quat_slerp, quat_from_basis, quat_axis_angle,
    load_bvh, _correct_endpoints,
)
from flint.platform.os import read_bytes, env_now, file_stamp

comptime FIX = "tests/fixtures/onnx/"


def _f32(path: String) raises -> List[Float32]:
    var b = read_bytes(path)
    assert_true(len(b) > 0, "missing fixture " + path)
    var out = List[Float32](length=len(b) // 4, fill=0.0)
    var p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(b.unsafe_ptr()))
    for k in range(len(out)):
        out[k] = p.unsafe_offset(k).unsafe_load()
    return out^


def _i64(path: String) raises -> List[Int]:
    var b = read_bytes(path)
    assert_true(len(b) > 0, "missing fixture " + path)
    var out = List[Int](length=len(b) // 8, fill=0)
    var p = Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(b.unsafe_ptr()))
    for k in range(len(out)):
        out[k] = Int(p.unsafe_offset(k).unsafe_load())
    return out^


def test_broadcasting() raises:
    var offs = broadcast_offsets([3, 1], [2, 3, 4])
    assert_equal(len(offs), 24)
    assert_equal(offs[0], 0)
    assert_equal(offs[4], 1)
    assert_equal(offs[12], 0)
    var a = Tensor.floats([2, 3], [1, 2, 3, 4, 5, 6])
    var b = Tensor.floats([3], [10, 20, 30])
    var c = binary(B_ADD, a, b)
    assert_equal(c.f[4], 25.0)
    var d = binary(B_MUL, b, a)  # small operand first
    assert_equal(d.f[5], 180.0)
    var col = Tensor.floats([2, 1], [2, 3])
    var e = binary(B_MUL, a, col)
    assert_equal(e.f[3], 12.0)
    print("  broadcasting: ok")


def _run_tiny(gpu: Bool) raises -> Bool:
    var m = OnnxModel.load(FIX + "tiny_ops.onnx", 2)
    if gpu and not m.use_gpu():
        print("  tiny_ops on GPU: no MAX GPU device, skipped")
        return False
    m.set_input("x", Tensor.floats([1, 4, 8], _f32(FIX + "tiny_ops_x.f32")))
    m.run()
    var y = m.output("y")
    var want = _f32(FIX + "tiny_ops_y.f32")
    assert_equal(y.numel(), len(want))
    var worst: Float32 = 0
    for k in range(len(want)):
        worst = max(worst, abs(y.fget(k) - want[k]))
    assert_true(worst < 2e-5, "tiny_ops y differs from onnxruntime by " + String(worst))
    var idx = m.output("idx")
    var want_idx = _i64(FIX + "tiny_ops_idx.i64")
    assert_equal(idx.numel(), len(want_idx))
    for k in range(len(want_idx)):
        assert_equal(idx.i[k], want_idx[k])
    var tv = m.output("tv")
    var want_tv = _f32(FIX + "tiny_ops_tv.f32")
    for k in range(len(want_tv)):
        assert_true(abs(tv.fget(k) - want_tv[k]) < 2e-5)
    var shp = m.output("shpf")
    assert_equal(shp.fget(0), 2.0)
    assert_equal(shp.fget(1), 4.0)
    print("  tiny_ops on", m.gpu_name(), ": matches onnxruntime (max |dy| =", worst, ")")
    return True


def test_tiny_ops() raises:
    _ = _run_tiny(False)
    _ = _run_tiny(True)


def _qclose(a: MbQuat, b: MbQuat, tol: Float32) -> Bool:
    return 1.0 - abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w) < tol


def _run_fused(gpu: Bool) raises -> Bool:
    var m = OnnxModel.load(FIX + "tiny_fused.onnx", 2)
    if gpu and not m.use_gpu():
        print("  tiny_fused on GPU: no MAX GPU device, skipped")
        return False
    var attn = 0
    var gmm = 0
    for k in range(len(m.fuse)):
        if m.fuse[k] == 2:
            attn += 1
        elif m.fuse[k] == 3:
            gmm += 1
    assert_equal(attn, 1)
    assert_equal(gmm, 1)
    var x = _f32(FIX + "tiny_fused_x.f32")
    var nbr = _i64(FIX + "tiny_fused_nbr.i64")
    m.set_input("x", Tensor.floats([70, 8], x^))
    m.set_input("nbr", Tensor.ints(DT_I64, [70, 27], nbr^))
    m.run()
    var y = m.output("y")
    var want = _f32(FIX + "tiny_fused_y.f32")
    assert_equal(y.numel(), len(want))
    var worst: Float32 = 0
    for k in range(len(want)):
        worst = max(worst, abs(y.fget(k) - want[k]))
    assert_true(worst < 5e-5, "tiny_fused y differs from onnxruntime by " + String(worst))
    print("  tiny_fused (gather-GEMM, attention, BatchNorm) on", m.gpu_name(), ": max |dy| =", worst)
    return True


def test_fused_ops() raises:
    _ = _run_fused(False)
    _ = _run_fused(True)


def test_motionbricks_math() raises:
    var qa = quat_axis_angle(MbVec3(0, 1, 0), 0.4)
    var qb = quat_axis_angle(MbVec3(0, 1, 0), 1.2)
    assert_true(_qclose(quat_slerp(qa, qb, 0.5), quat_axis_angle(MbVec3(0, 1, 0), 0.8), 1e-6))
    # slerp takes the short way round even when the quaternions have opposite signs
    var neg = MbQuat(-qb.x, -qb.y, -qb.z, -qb.w)
    assert_true(_qclose(quat_slerp(qa, neg, 0.5), quat_axis_angle(MbVec3(0, 1, 0), 0.8), 1e-6))
    var q = quat_axis_angle(MbVec3(0.3, -0.8, 0.5), 2.7)
    var back = quat_from_basis(q.rotate(MbVec3(1, 0, 0)), q.rotate(MbVec3(0, 1, 0)), q.rotate(MbVec3(0, 0, 1)))
    assert_true(_qclose(q, back, 1e-6))
    # endpoint correction: first frame -> last context pose, last frame -> first target pose
    var m = MbMotion()
    for f in range(10):
        var pose = List[MbXform]()
        pose.append(MbXform(MbVec3(Float32(f) * 0.1, 1, 0), quat_axis_angle(MbVec3(0, 1, 0), Float32(f) * 0.05)))
        pose.append(MbXform(MbVec3(0, 0.3, 0), quat_axis_angle(MbVec3(1, 0, 0), Float32(f) * 0.1)))
        m.poses.append(pose^)
    var ctx = List[MbXform]()
    ctx.append(MbXform(MbVec3(-0.05, 1.02, 0.01), quat_axis_angle(MbVec3(0, 1, 0), -0.1)))
    ctx.append(MbXform(MbVec3(0, 0.3, 0), quat_axis_angle(MbVec3(1, 0, 0), 0.2)))
    var tgt = List[MbXform]()
    tgt.append(MbXform(MbVec3(1.0, 0.98, 0.03), quat_axis_angle(MbVec3(0, 1, 0), 0.5)))
    tgt.append(MbXform(MbVec3(0, 0.3, 0), quat_axis_angle(MbVec3(1, 0, 0), 0.7)))
    _correct_endpoints(m, ctx, tgt)
    for j in range(2):
        assert_true((m.poses[0][j].p - ctx[j].p).length() < 1e-5)
        assert_true(_qclose(m.poses[0][j].q, ctx[j].q, 1e-6))
        assert_true((m.poses[9][j].p - tgt[j].p).length() < 1e-5)
        assert_true(_qclose(m.poses[9][j].q, tgt[j].q, 1e-6))
    print("  motionbricks math + endpoint correction: ok")


def test_motionbricks_soma() raises:
    var root = env_now("MOTIONBRICKS_DIR")
    if root.byte_length() == 0:
        print("  SOMA generation: MOTIONBRICKS_DIR not set, skipped")
        return
    var onnx = root + "/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx"
    var bvh = root + "/assets/motions/soma/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh"
    if file_stamp(onnx)[0] < 1000 or file_stamp(bvh)[0] < 1000:
        print("  SOMA generation: model or clip missing under MOTIONBRICKS_DIR (git lfs pull?), skipped")
        return
    var clip = load_bvh(bvh)
    var mb = MotionBricks.load(onnx, clip.skeleton)
    var ctx = List[List[MbXform]]()
    for i in range(4):
        ctx.append(clip.sample(1.0 + Float32(i - 3) / 30.0))
    var tgt = List[List[MbXform]]()
    for i in range(4):
        tgt.append(clip.sample(1.8 + Float32(i) / 30.0))
    var cfg = MbConfig()
    var motion = mb.generate(ctx, tgt, cfg)
    assert_true(motion.tokens >= 8 and motion.tokens <= 14)
    assert_equal(motion.num_frames(), motion.tokens * 4 - 6)
    assert_true((motion.poses[0][0].p - ctx[3][0].p).length() < 1e-4)
    for f in range(motion.num_frames()):
        for j in range(clip.skeleton.num_joints()):
            assert_true(not isnan(motion.poses[f][j].p.x) and not isnan(motion.poses[f][j].q.w))
    var travelled = (motion.poses[motion.num_frames() - 1][0].p - motion.poses[0][0].p).length()
    assert_true(travelled > 0.3, "the walk should move the hips forward")
    print("  SOMA generation on", mb.device(), ":", motion.tokens, "tokens,", motion.num_frames(), "frames, hips moved", travelled, "m")


def main() raises:
    print("=== Flint v5 flint.ml ONNX + MAX GPU + MotionBricks Suite ===")
    test_broadcasting()
    test_tiny_ops()
    test_fused_ops()
    test_motionbricks_math()
    test_motionbricks_soma()
    print("=== ALL FLINT v5 ML TESTS PASSED ===")
