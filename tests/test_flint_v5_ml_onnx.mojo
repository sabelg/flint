"""Flint v5 `flint.ml` suite: pure-Mojo ONNX interpreter, MAX GPU matrix products and the
MotionBricks SOMA runtime (`test_flint_v5_ml_onnx.mojo`).

Verifies:
  1. Broadcasting offsets and `binary` fast paths against element-by-element numpy rules.
  2. `tests/fixtures/onnx/tiny_ops.onnx` (MatMul, Add, Relu, LayerNormalization, Transpose, dilated
     Conv, Reshape, Gemm transB, Softmax, LeakyRelu, Gather, ReduceSum, Atan, Slice, Shape, Cast,
     Concat, TopK, ArgMax, Greater, Where, Unsqueeze) reproduces onnxruntime's outputs on the CPU,
     and again with the matrix products on a MAX GPU when one is present.
  3. MotionBricks math: Warp-compatible quaternion slerp / basis conversion, and the endpoint
     correction lands the generated motion exactly on the last context and first target pose.
  4. With `MOTIONBRICKS_DIR` set to a MotionBricks checkout: the SOMA model generates a walk from
     its own BVH clip, choosing an allowed token count, with finite poses that start on the context.
"""
from test_kimodo import test_kimodo
from std.ffi import external_call
from std.math import abs, isfinite, isnan, sqrt
from std.testing import assert_true, assert_equal
from flint.core.jobs import JobPool
from flint.ml.gpu import GpuMatmul
from flint.ml.onnx import OnnxModel
from flint.ml.protobuf import PbReader
from flint.ml.tensor import Tensor, DT_F32, DT_I64, broadcast_offsets, f32_from_bits
from flint.ml.ops import (
    binary,
    where,
    concat,
    topk,
    scatter_nd,
    range_op,
    expand,
    tile,
    onehot,
    resize_nearest,
    gemm,
    softmax,
    layer_norm,
    B_ADD,
    B_EQ,
    B_MUL,
)
from flint.ml.motionbricks import (
    MotionBricks, MbConfig, MbMotion, MbQuat, MbVec3, MbXform, quat_slerp, quat_from_basis, quat_axis_angle,
    load_bvh, _correct_endpoints,
)
from flint.platform.os import read_bytes, write_bytes, remove_file, env_now, file_stamp

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


def _pb_varint(mut out: List[UInt8], var v: Int):
    while True:
        var b = UInt8(v & 0x7F)
        v >>= 7
        if v > 0:
            out.append(b | UInt8(0x80))
        else:
            out.append(b)
            break


def _pb_bytes(mut out: List[UInt8], field: Int, payload: List[UInt8]):
    _pb_varint(out, (field << 3) | 2)
    _pb_varint(out, len(payload))
    for b in payload:
        out.append(b)


def _pb_str(mut out: List[UInt8], field: Int, s: String):
    var bs = List[UInt8]()
    for b in s.as_bytes():
        bs.append(b)
    _pb_bytes(out, field, bs)


def test_t5_onnx_and_ops_negative() raises:
    # 1. [Crit-3] binary() mixed float/int dtypes must raise Error instead of OOB indexing
    var tf = Tensor.floats([2], [1.0, 2.0])
    var ti = Tensor.ints(DT_I64, [2], [3, 4])
    var caught_mixed = False
    try:
        _ = binary(B_ADD, tf, ti)
    except:
        caught_mixed = True
    assert_true(caught_mixed, "binary(B_ADD, float, int) must raise Error")

    var caught_mixed_rev = False
    try:
        _ = binary(B_MUL, ti, tf)
    except:
        caught_mixed_rev = True
    assert_true(caught_mixed_rev, "binary(B_MUL, int, float) must raise Error")

    # Comparison ops still compare across float and int cleanly
    var eq = binary(B_EQ, Tensor.floats([2], [3.0, 5.0]), ti)
    assert_equal(eq.i[0], 1)
    assert_equal(eq.i[1], 0)

    # where() mismatched branches raise Error
    var cond = Tensor.ints(DT_I64, [2], [1, 0])
    var caught_where = False
    try:
        _ = where(cond, tf, ti)
    except:
        caught_where = True
    assert_true(caught_where, "where() with mismatched float/int branches must raise Error")

    # 2. [Crit-1] ONNX node with empty/optional input 0 ("") must raise Error, not index -1
    var node_empty_in = List[UInt8]()
    _pb_str(node_empty_in, 1, "")  # input[0] = "" -> id -1
    _pb_str(node_empty_in, 2, "y")
    _pb_str(node_empty_in, 4, "Relu")
    var vi_in = List[UInt8]()
    _pb_str(vi_in, 1, "x")
    var vi_out = List[UInt8]()
    _pb_str(vi_out, 1, "y")
    var graph_a = List[UInt8]()
    _pb_bytes(graph_a, 1, node_empty_in)
    _pb_bytes(graph_a, 11, vi_in)
    _pb_bytes(graph_a, 12, vi_out)
    var model_a = List[UInt8]()
    _pb_bytes(model_a, 7, graph_a)

    var pid = Int(external_call["getpid", Int32]())
    var path_a = "/tmp/flint_t5_onnx_a_" + String(pid) + ".onnx"
    var path_b = "/tmp/flint_t5_onnx_b_" + String(pid) + ".onnx"
    assert_true(write_bytes(path_a, model_a))
    var m_a = OnnxModel.load(path_a, 1)
    _ = remove_file(path_a)
    m_a.set_input("x", Tensor.floats([2], [1.0, -1.0]))
    var caught_in0 = False
    try:
        m_a.run()
    except:
        caught_in0 = True
    assert_true(caught_in0, "ONNX node with optional/empty input 0 must raise Error")

    # 3. [Crit-2] ONNX truncated WIRE_LEN field-2 float attribute must raise Error, not OOB fl[0]
    var bad_attr = List[UInt8]()
    _pb_str(bad_attr, 1, "alpha")
    var short_f = List[UInt8]()
    short_f.append(UInt8(0))
    _pb_bytes(bad_attr, 2, short_f)  # field 2 (float) with WIRE_LEN length 1 (< 4 bytes)
    var node_bad_attr = List[UInt8]()
    _pb_str(node_bad_attr, 1, "x")
    _pb_str(node_bad_attr, 2, "y")
    _pb_str(node_bad_attr, 4, "LeakyRelu")
    _pb_bytes(node_bad_attr, 5, bad_attr)
    var graph_b = List[UInt8]()
    _pb_bytes(graph_b, 1, node_bad_attr)
    _pb_bytes(graph_b, 11, vi_in)
    _pb_bytes(graph_b, 12, vi_out)
    var model_b = List[UInt8]()
    _pb_bytes(model_b, 7, graph_b)

    assert_true(write_bytes(path_b, model_b))
    var caught_attr = False
    try:
        _ = OnnxModel.load(path_b, 1)
    except:
        caught_attr = True
    _ = remove_file(path_b)
    assert_true(caught_attr, "Truncated WIRE_LEN float attribute must raise Error")

    # 4. Round 2 PbReader.skip truncated WIRE_I64 / WIRE_I32
    var pb_i64: List[UInt8] = [UInt8(0x09)]  # field 1, WIRE_I64
    var pr64 = PbReader(Int(pb_i64.unsafe_ptr()), Int(pb_i64.unsafe_ptr()) + len(pb_i64))
    pr64.key()
    var caught_pb64 = False
    try:
        pr64.skip()
    except:
        caught_pb64 = True
    assert_true(caught_pb64, "PbReader.skip on truncated WIRE_I64 must raise Error")

    var pb_i32: List[UInt8] = [UInt8(0x0D)]  # field 1, WIRE_I32
    var pr32 = PbReader(Int(pb_i32.unsafe_ptr()), Int(pb_i32.unsafe_ptr()) + len(pb_i32))
    pr32.key()
    var caught_pb32 = False
    try:
        pr32.skip()
    except:
        caught_pb32 = True
    assert_true(caught_pb32, "PbReader.skip on truncated WIRE_I32 must raise Error")
    _ = len(pb_i64)
    _ = len(pb_i32)

    # 5. Round 2 ops.mojo bounds & dtype validations
    var caught_concat = False
    try:
        var xs = List[Tensor]()
        xs.append(ti.copy())
        xs.append(tf.copy())
        _ = concat(xs, 0)
    except:
        caught_concat = True
    assert_true(caught_concat, "concat mixed dtypes must raise Error")

    var caught_topk = False
    try:
        _ = topk(tf, 5, 0, True)
    except:
        caught_topk = True
    assert_true(caught_topk, "topk with k > dim must raise Error")

    var caught_scatter0 = False
    try:
        var idx0 = Tensor.ints(DT_I64, List[Int](), [0])
        _ = scatter_nd(tf, idx0, Tensor.floats([1], [9.0]))
    except:
        caught_scatter0 = True
    assert_true(caught_scatter0, "scatter_nd with 0-D indices must raise Error")

    var caught_range = False
    try:
        _ = range_op(Tensor.ints(DT_I64, [1], [0]), Tensor.ints(DT_I64, [1], [4]), Tensor.floats([1], [1.0]))
    except:
        caught_range = True
    assert_true(caught_range, "range_op with mixed dtypes must raise Error")

    # 6. Round 2 load_bvh malformed hierarchy errors
    var bvh_cases: List[String] = [
        String("HIERARCHY\nROOT"),
        String("HIERARCHY\nOFFSET 0 0 0"),
        String("HIERARCHY\n}"),
    ]
    for bi in range(len(bvh_cases)):
        var bp = "/tmp/flint_t5_bad_bvh_" + String(pid) + "_" + String(bi) + ".bvh"
        var bbytes = List[UInt8]()
        for b in bvh_cases[bi].as_bytes():
            bbytes.append(b)
        assert_true(write_bytes(bp, bbytes))
        var caught_bvh = False
        try:
            _ = load_bvh(bp)
        except:
            caught_bvh = True
        _ = remove_file(bp)
        assert_true(caught_bvh, "malformed BVH case " + String(bi) + " must raise Error")

    # 7. Round 2 OnnxModel 5 reproducers (empty output, Constant 0 outputs, Unsqueeze OOB, Reshape extra 0, ConstantOfShape empty value)
    # 7a. Empty graph output name
    var vi_empty = List[UInt8]()
    _pb_str(vi_empty, 1, "")
    var g_empty_out = List[UInt8]()
    _pb_bytes(g_empty_out, 11, vi_in)
    _pb_bytes(g_empty_out, 12, vi_empty)
    var m_empty_out = List[UInt8]()
    _pb_bytes(m_empty_out, 7, g_empty_out)
    var p_eo = "/tmp/flint_t5_eo_" + String(pid) + ".onnx"
    assert_true(write_bytes(p_eo, m_empty_out))
    var caught_eo = False
    try:
        _ = OnnxModel.load(p_eo, 1)
    except:
        caught_eo = True
    _ = remove_file(p_eo)
    assert_true(caught_eo, "ONNX graph output with empty name must raise Error")

    # 7b. Constant node with 0 outputs
    var t_val = List[UInt8]()
    _pb_varint(t_val, (1 << 3) | 0); _pb_varint(t_val, 1)  # dims = [1]
    _pb_varint(t_val, (2 << 3) | 0); _pb_varint(t_val, 1)  # DT_F32
    var f4: List[UInt8] = [UInt8(0), UInt8(0), UInt8(0x80), UInt8(0x3F)]
    _pb_bytes(t_val, 9, f4)
    var attr_val = List[UInt8]()
    _pb_str(attr_val, 1, "value")
    _pb_bytes(attr_val, 5, t_val)
    var n_const = List[UInt8]()
    _pb_str(n_const, 4, "Constant")
    _pb_bytes(n_const, 5, attr_val)
    var g_const = List[UInt8]()
    _pb_bytes(g_const, 1, n_const)
    var m_const = List[UInt8]()
    _pb_bytes(m_const, 7, g_const)
    var p_co = "/tmp/flint_t5_co_" + String(pid) + ".onnx"
    assert_true(write_bytes(p_co, m_const))
    var caught_co = False
    try:
        _ = OnnxModel.load(p_co, 1)
    except:
        caught_co = True
    _ = remove_file(p_co)
    assert_true(caught_co, "ONNX Constant with 0 outputs must raise Error")

    # Helper: TensorProto int64 1-D initializer helper
    var vi_ax = List[UInt8]()
    _pb_str(vi_ax, 1, "ax")

    # 7c. Unsqueeze with out-of-bounds axis 5
    var n_unsq = List[UInt8]()
    _pb_str(n_unsq, 1, "x")
    _pb_str(n_unsq, 1, "ax")
    _pb_str(n_unsq, 2, "y")
    _pb_str(n_unsq, 4, "Unsqueeze")
    var g_unsq = List[UInt8]()
    _pb_bytes(g_unsq, 1, n_unsq)
    _pb_bytes(g_unsq, 11, vi_in)
    _pb_bytes(g_unsq, 11, vi_ax)
    _pb_bytes(g_unsq, 12, vi_out)
    var m_unsq = List[UInt8]()
    _pb_bytes(m_unsq, 7, g_unsq)
    var p_unsq = "/tmp/flint_t5_unsq_" + String(pid) + ".onnx"
    assert_true(write_bytes(p_unsq, m_unsq))
    var mu = OnnxModel.load(p_unsq, 1)
    _ = remove_file(p_unsq)
    mu.set_input("x", Tensor.floats([2], [1.0, 2.0]))
    mu.set_input("ax", Tensor.ints(DT_I64, [1], [5]))
    var caught_unsq = False
    try:
        mu.run()
    except:
        caught_unsq = True
    assert_true(caught_unsq, "Unsqueeze with OOB axis must raise Error")

    # 7d. Reshape with target [2, 0] on 1-D [2]
    var n_rsh = List[UInt8]()
    _pb_str(n_rsh, 1, "x")
    _pb_str(n_rsh, 1, "ax")
    _pb_str(n_rsh, 2, "y")
    _pb_str(n_rsh, 4, "Reshape")
    var g_rsh = List[UInt8]()
    _pb_bytes(g_rsh, 1, n_rsh)
    _pb_bytes(g_rsh, 11, vi_in)
    _pb_bytes(g_rsh, 11, vi_ax)
    _pb_bytes(g_rsh, 12, vi_out)
    var m_rsh = List[UInt8]()
    _pb_bytes(m_rsh, 7, g_rsh)
    var p_rsh = "/tmp/flint_t5_rsh_" + String(pid) + ".onnx"
    assert_true(write_bytes(p_rsh, m_rsh))
    var mr = OnnxModel.load(p_rsh, 1)
    _ = remove_file(p_rsh)
    mr.set_input("x", Tensor.floats([2], [1.0, 2.0]))
    mr.set_input("ax", Tensor.ints(DT_I64, [2], [2, 0]))
    var caught_rsh = False
    try:
        mr.run()
    except:
        caught_rsh = True
    assert_true(caught_rsh, "Reshape with 0 dim past input rank must raise Error")

    # 7e. ConstantOfShape with empty integer value tensor
    var t_empty_i = List[UInt8]()
    _pb_varint(t_empty_i, (1 << 3) | 0); _pb_varint(t_empty_i, 0)  # dims = [0]
    _pb_varint(t_empty_i, (2 << 3) | 0); _pb_varint(t_empty_i, DT_I64)
    var attr_empty_val = List[UInt8]()
    _pb_str(attr_empty_val, 1, "value")
    _pb_bytes(attr_empty_val, 5, t_empty_i)
    var n_cos = List[UInt8]()
    _pb_str(n_cos, 1, "x")
    _pb_str(n_cos, 2, "y")
    _pb_str(n_cos, 4, "ConstantOfShape")
    _pb_bytes(n_cos, 5, attr_empty_val)
    var g_cos = List[UInt8]()
    _pb_bytes(g_cos, 1, n_cos)
    _pb_bytes(g_cos, 11, vi_in)
    _pb_bytes(g_cos, 12, vi_out)
    var m_cos = List[UInt8]()
    _pb_bytes(m_cos, 7, g_cos)
    var p_cos = "/tmp/flint_t5_cos_" + String(pid) + ".onnx"
    assert_true(write_bytes(p_cos, m_cos))
    var mcos = OnnxModel.load(p_cos, 1)
    _ = remove_file(p_cos)
    mcos.set_input("x", Tensor.ints(DT_I64, [1], [3]))
    var caught_cos = False
    try:
        mcos.run()
    except:
        caught_cos = True
    assert_true(caught_cos, "ConstantOfShape with empty value tensor must raise Error")

    # 7f. Squeeze invalid non-1 dimension, duplicate axis, OOB axis, and valid squeeze (axes & default)
    var n_sq = List[UInt8]()
    _pb_str(n_sq, 1, "x"); _pb_str(n_sq, 1, "ax"); _pb_str(n_sq, 2, "y"); _pb_str(n_sq, 4, "Squeeze")
    var g_sq = List[UInt8]()
    _pb_bytes(g_sq, 1, n_sq)
    _pb_bytes(g_sq, 11, vi_in); _pb_bytes(g_sq, 11, vi_ax); _pb_bytes(g_sq, 12, vi_out)
    var m_sq = _load_temp_onnx("/tmp/flint_t5_sq_" + String(pid) + ".onnx", g_sq)
    m_sq.set_input("x", Tensor.floats([2, 3], [1, 2, 3, 4, 5, 6]))
    m_sq.set_input("ax", Tensor.ints(DT_I64, [1], [0]))
    var caught_sq_non1 = False
    try:
        m_sq.run()
    except:
        caught_sq_non1 = True
    assert_true(caught_sq_non1, "Squeeze on non-1 dimension must raise Error instead of returning shape=[0]")

    m_sq.set_input("x", Tensor.floats([2, 1, 3], [1, 2, 3, 4, 5, 6]))
    m_sq.set_input("ax", Tensor.ints(DT_I64, [2], [1, 1]))
    var caught_sq_dup = False
    try:
        m_sq.run()
    except:
        caught_sq_dup = True
    assert_true(caught_sq_dup, "Squeeze with duplicate axis must raise Error")

    m_sq.set_input("ax", Tensor.ints(DT_I64, [1], [5]))
    var caught_sq_oob = False
    try:
        m_sq.run()
    except:
        caught_sq_oob = True
    assert_true(caught_sq_oob, "Squeeze with OOB axis must raise Error")

    m_sq.set_input("x", Tensor.floats([1, 2, 1, 3], [1, 2, 3, 4, 5, 6]))
    m_sq.set_input("ax", Tensor.ints(DT_I64, [2], [0, 2]))
    m_sq.run()
    var y_sq = m_sq.output("y")
    assert_equal(len(y_sq.shape), 2)
    assert_equal(y_sq.shape[0], 2)
    assert_equal(y_sq.shape[1], 3)
    assert_equal(y_sq.fget(5), Float32(6.0))

    var n_sq_def = List[UInt8]()
    _pb_str(n_sq_def, 1, "x"); _pb_str(n_sq_def, 2, "y"); _pb_str(n_sq_def, 4, "Squeeze")
    var g_sq_def = List[UInt8]()
    _pb_bytes(g_sq_def, 1, n_sq_def)
    _pb_bytes(g_sq_def, 11, vi_in); _pb_bytes(g_sq_def, 12, vi_out)
    var m_sq_def = _load_temp_onnx("/tmp/flint_t5_sqd_" + String(pid) + ".onnx", g_sq_def)
    m_sq_def.set_input("x", Tensor.floats([1, 2, 1, 3], [1, 2, 3, 4, 5, 6]))
    m_sq_def.run()
    var y_sq_def = m_sq_def.output("y")
    assert_equal(len(y_sq_def.shape), 2)
    assert_equal(y_sq_def.shape[0], 2)
    assert_equal(y_sq_def.shape[1], 3)
    assert_equal(y_sq_def.fget(0), Float32(1.0))

    # 7g. Resize with NaN / non-positive / +Inf scales must raise Error, and valid Resize must succeed
    var attr_rs_mode = List[UInt8](); _pb_str(attr_rs_mode, 1, "mode"); _pb_str(attr_rs_mode, 4, "nearest")
    var attr_rs_ctm = List[UInt8](); _pb_str(attr_rs_ctm, 1, "coordinate_transformation_mode"); _pb_str(attr_rs_ctm, 4, "asymmetric")
    var attr_rs_nm = List[UInt8](); _pb_str(attr_rs_nm, 1, "nearest_mode"); _pb_str(attr_rs_nm, 4, "floor")
    var vi_sc = List[UInt8](); _pb_str(vi_sc, 1, "sc")
    var n_rs = List[UInt8]()
    _pb_str(n_rs, 1, "x"); _pb_str(n_rs, 1, ""); _pb_str(n_rs, 1, "sc")
    _pb_str(n_rs, 2, "y"); _pb_str(n_rs, 4, "Resize")
    _pb_bytes(n_rs, 5, attr_rs_mode); _pb_bytes(n_rs, 5, attr_rs_ctm); _pb_bytes(n_rs, 5, attr_rs_nm)
    var g_rs = List[UInt8]()
    _pb_bytes(g_rs, 1, n_rs)
    _pb_bytes(g_rs, 11, vi_in); _pb_bytes(g_rs, 11, vi_sc); _pb_bytes(g_rs, 12, vi_out)
    var m_rs = _load_temp_onnx("/tmp/flint_t5_rs_" + String(pid) + ".onnx", g_rs)
    m_rs.set_input("x", Tensor.floats([2, 2], [Float32(1), 2, 3, 4]))

    var bad_scales: List[Float32] = [
        f32_from_bits(0x7FC00000),  # NaN
        Float32(0.0),               # zero
        Float32(-1.0),              # negative
        f32_from_bits(0x7F800000),  # +Inf
    ]
    for bs in bad_scales:
        m_rs.set_input("sc", Tensor.floats([2], [bs, Float32(1.0)]))
        var caught_rs = False
        try:
            m_rs.run()
        except:
            caught_rs = True
        assert_true(caught_rs, "Resize with invalid scale must raise Error")

    m_rs.set_input("sc", Tensor.floats([2], [Float32(2.0), Float32(2.0)]))
    m_rs.run()
    var y_rs = m_rs.output("y")
    assert_equal(y_rs.shape[0], 4)
    assert_equal(y_rs.shape[1], 4)
    assert_equal(y_rs.fget(0), Float32(1.0))
    assert_equal(y_rs.fget(15), Float32(4.0))

    # 7h. Round 13 B1-B3: Multi-axis & int64 Resize OOM, ops.{tile, expand, onehot, resize_nearest, binary} MAX_RANGE_ELEMS, unset Tensor() SIGTRAP guards
    # B1.1: 2D Resize with scales=[1e9, 1e9] (per-axis <= 1<<40, total > MAX_RANGE_ELEMS)
    m_rs.set_input("sc", Tensor.floats([2], [Float32(1e9), Float32(1e9)]))
    var caught_rs_2d = False
    try:
        m_rs.run()
    except:
        caught_rs_2d = True
    assert_true(caught_rs_2d, "Resize 2D scales=[1e9, 1e9] must raise Error")

    # B1.2: Resize with int64 sizes=[1<<30, 1<<30]
    var vi_sz = List[UInt8](); _pb_str(vi_sz, 1, "sz")
    var n_rs_sz = List[UInt8]()
    _pb_str(n_rs_sz, 1, "x"); _pb_str(n_rs_sz, 1, ""); _pb_str(n_rs_sz, 1, ""); _pb_str(n_rs_sz, 1, "sz")
    _pb_str(n_rs_sz, 2, "y"); _pb_str(n_rs_sz, 4, "Resize")
    _pb_bytes(n_rs_sz, 5, attr_rs_mode); _pb_bytes(n_rs_sz, 5, attr_rs_ctm); _pb_bytes(n_rs_sz, 5, attr_rs_nm)
    var g_rs_sz = List[UInt8]()
    _pb_bytes(g_rs_sz, 1, n_rs_sz)
    _pb_bytes(g_rs_sz, 11, vi_in); _pb_bytes(g_rs_sz, 11, vi_sz); _pb_bytes(g_rs_sz, 12, vi_out)
    var m_rs_sz = _load_temp_onnx("/tmp/flint_t5_rsz_" + String(pid) + ".onnx", g_rs_sz)
    m_rs_sz.set_input("x", Tensor.floats([2, 2], [Float32(1), 2, 3, 4]))
    m_rs_sz.set_input("sz", Tensor.ints(DT_I64, [2], [1 << 30, 1 << 30]))
    var caught_rs_int = False
    try:
        m_rs_sz.run()
    except:
        caught_rs_int = True
    assert_true(caught_rs_int, "Resize int64 sizes=[1<<30, 1<<30] must raise Error")

    m_rs_sz.set_input("sz", Tensor.ints(DT_I64, [2], [4, 4]))
    m_rs_sz.run()
    var y_rs_sz = m_rs_sz.output("y")
    assert_equal(y_rs_sz.shape[0], 4)
    assert_equal(y_rs_sz.shape[1], 4)

    var caught_rn_big = False
    try:
        _ = resize_nearest(Tensor.floats([2, 2], [Float32(1), 2, 3, 4]), [1 << 14, 1 << 14])
    except:
        caught_rn_big = True
    assert_true(caught_rn_big, "ops.resize_nearest exceeding MAX_RANGE_ELEMS must raise Error")

    # B2: ops.{tile, expand, onehot, binary} MAX_RANGE_ELEMS enforcement
    var caught_tile = False
    try:
        _ = tile(Tensor.floats([2, 2], [Float32(1), 2, 3, 4]), [1 << 30, 1 << 30])
    except:
        caught_tile = True
    assert_true(caught_tile, "ops.tile exceeding MAX_RANGE_ELEMS must raise Error")

    var caught_expand = False
    try:
        _ = expand(Tensor.floats([1, 1], [Float32(1.0)]), [1 << 30, 1 << 30])
    except:
        caught_expand = True
    assert_true(caught_expand, "ops.expand exceeding MAX_RANGE_ELEMS must raise Error")

    var caught_onehot = False
    try:
        var oh_idx = Tensor.ints(DT_I64, [1 << 13], List[Int](length=1 << 13, fill=0))
        var oh_vals = Tensor.floats([2], [Float32(0.0), Float32(1.0)])
        _ = onehot(oh_idx, 1 << 13, oh_vals)
    except:
        caught_onehot = True
    assert_true(caught_onehot, "ops.onehot exceeding MAX_RANGE_ELEMS must raise Error")

    var caught_bin_bcast = False
    try:
        var ba = Tensor.floats([1 << 13, 1], List[Float32](length=1 << 13, fill=Float32(1.0)))
        var bb = Tensor.floats([1, 1 << 13], List[Float32](length=1 << 13, fill=Float32(2.0)))
        _ = binary(B_ADD, ba, bb)
    except:
        caught_bin_bcast = True
    assert_true(caught_bin_bcast, "ops.binary broadcast exceeding MAX_RANGE_ELEMS must raise Error")

    # B3: Unset Tensor() auxiliary inputs in TopK, CumSum, OneHot, Resize, Clip must raise Error without SIGTRAP
    var n_tk_u = List[UInt8]()
    _pb_str(n_tk_u, 1, "x"); _pb_str(n_tk_u, 1, "ax"); _pb_str(n_tk_u, 2, "y"); _pb_str(n_tk_u, 4, "TopK")
    var g_tk_u = List[UInt8]()
    _pb_bytes(g_tk_u, 1, n_tk_u); _pb_bytes(g_tk_u, 11, vi_in); _pb_bytes(g_tk_u, 11, vi_ax); _pb_bytes(g_tk_u, 12, vi_out)
    var m_tk_u = _load_temp_onnx("/tmp/flint_t5_tku_" + String(pid) + ".onnx", g_tk_u)
    m_tk_u.set_input("x", Tensor.floats([3], [Float32(1), 2, 3]))
    var caught_tk_u = False
    try:
        m_tk_u.run()
    except:
        caught_tk_u = True
    assert_true(caught_tk_u, "TopK with unset k tensor must raise Error")

    var n_cs_u = List[UInt8]()
    _pb_str(n_cs_u, 1, "x"); _pb_str(n_cs_u, 1, "ax"); _pb_str(n_cs_u, 2, "y"); _pb_str(n_cs_u, 4, "CumSum")
    var g_cs_u = List[UInt8]()
    _pb_bytes(g_cs_u, 1, n_cs_u); _pb_bytes(g_cs_u, 11, vi_in); _pb_bytes(g_cs_u, 11, vi_ax); _pb_bytes(g_cs_u, 12, vi_out)
    var m_cs_u = _load_temp_onnx("/tmp/flint_t5_csu_" + String(pid) + ".onnx", g_cs_u)
    m_cs_u.set_input("x", Tensor.floats([3], [Float32(1), 2, 3]))
    var caught_cs_u = False
    try:
        m_cs_u.run()
    except:
        caught_cs_u = True
    assert_true(caught_cs_u, "CumSum with unset axis tensor must raise Error")

    var vi_vals = List[UInt8](); _pb_str(vi_vals, 1, "vals")
    var n_oh_u = List[UInt8]()
    _pb_str(n_oh_u, 1, "x"); _pb_str(n_oh_u, 1, "ax"); _pb_str(n_oh_u, 1, "vals"); _pb_str(n_oh_u, 2, "y"); _pb_str(n_oh_u, 4, "OneHot")
    var g_oh_u = List[UInt8]()
    _pb_bytes(g_oh_u, 1, n_oh_u)
    _pb_bytes(g_oh_u, 11, vi_in); _pb_bytes(g_oh_u, 11, vi_ax); _pb_bytes(g_oh_u, 11, vi_vals); _pb_bytes(g_oh_u, 12, vi_out)
    var m_oh_u = _load_temp_onnx("/tmp/flint_t5_ohu_" + String(pid) + ".onnx", g_oh_u)
    m_oh_u.set_input("x", Tensor.ints(DT_I64, [2], [0, 1]))
    m_oh_u.set_input("vals", Tensor.floats([2], [Float32(0), Float32(1)]))
    var caught_oh_u = False
    try:
        m_oh_u.run()
    except:
        caught_oh_u = True
    assert_true(caught_oh_u, "OneHot with unset depth tensor must raise Error")

    var m_rs_u = _load_temp_onnx("/tmp/flint_t5_rsu_" + String(pid) + ".onnx", g_rs)
    m_rs_u.set_input("x", Tensor.floats([2], [Float32(1), 2]))
    var caught_rs_u = False
    try:
        m_rs_u.run()
    except:
        caught_rs_u = True
    assert_true(caught_rs_u, "Resize with unset scales tensor must raise Error")

    var n_cl_u = List[UInt8]()
    _pb_str(n_cl_u, 1, "x"); _pb_str(n_cl_u, 1, "ax"); _pb_str(n_cl_u, 2, "y"); _pb_str(n_cl_u, 4, "Clip")
    var g_cl_u = List[UInt8]()
    _pb_bytes(g_cl_u, 1, n_cl_u); _pb_bytes(g_cl_u, 11, vi_in); _pb_bytes(g_cl_u, 11, vi_ax); _pb_bytes(g_cl_u, 12, vi_out)
    var m_cl_u = _load_temp_onnx("/tmp/flint_t5_clu_" + String(pid) + ".onnx", g_cl_u)
    m_cl_u.set_input("x", Tensor.floats([2], [Float32(1), 2]))
    var caught_cl_u = False
    try:
        m_cl_u.run()
    except:
        caught_cl_u = True
    assert_true(caught_cl_u, "Clip with unset min tensor must raise Error")

    var empty_t = Tensor()
    assert_equal(empty_t.get(0), Float64(0.0))
    assert_equal(empty_t.int_at(0), 0)
    assert_equal(Tensor.floats([2], [Float32(1), 2]).get(99), Float64(0.0))

    # B2b (R13b): ops.gemm & ONNX Gemm MAX_RANGE_ELEMS enforcement ([65536, 1] x [1, 65536] and [8192, 1] x [1, 8192])
    var jp_gm = JobPool()
    var ga_big = Tensor.floats([1 << 16, 1], List[Float32](length=1 << 16, fill=Float32(1.0)))
    var gb_big = Tensor.floats([1, 1 << 16], List[Float32](length=1 << 16, fill=Float32(1.0)))
    var caught_gemm_16 = False
    try:
        _ = gemm(jp_gm, ga_big, gb_big, Tensor(), False, Float32(1.0), Float32(1.0), False, False)
    except:
        caught_gemm_16 = True
    assert_true(caught_gemm_16, "ops.gemm [65536, 1] x [1, 65536] exceeding MAX_RANGE_ELEMS must raise Error")

    var ga_8k = Tensor.floats([8192, 1], List[Float32](length=8192, fill=Float32(1.0)))
    var gb_8k = Tensor.floats([1, 8192], List[Float32](length=8192, fill=Float32(1.0)))
    var caught_gemm_8k = False
    try:
        _ = gemm(jp_gm, ga_8k, gb_8k, Tensor(), False, Float32(1.0), Float32(1.0), False, False)
    except:
        caught_gemm_8k = True
    assert_true(caught_gemm_8k, "ops.gemm [8192, 1] x [1, 8192] exceeding MAX_RANGE_ELEMS must raise Error")

    var gemm_ok = gemm(
        jp_gm,
        Tensor.floats([2, 2], [Float32(1), 2, 3, 4]),
        Tensor.floats([2, 2], [Float32(1), 0, 0, 1]),
        Tensor(),
        False,
        Float32(1.0),
        Float32(1.0),
        False,
        False,
    )
    assert_equal(gemm_ok.shape[0], 2)
    assert_equal(gemm_ok.shape[1], 2)
    assert_equal(gemm_ok.fget(3), Float32(4.0))

    # ONNX Gemm node outer product > MAX_RANGE_ELEMS (with and without GPU)
    var vi_gb = List[UInt8](); _pb_str(vi_gb, 1, "b")
    var n_gm_big = List[UInt8]()
    _pb_str(n_gm_big, 1, "x"); _pb_str(n_gm_big, 1, "b"); _pb_str(n_gm_big, 2, "y"); _pb_str(n_gm_big, 4, "Gemm")
    var g_gm_big = List[UInt8]()
    _pb_bytes(g_gm_big, 1, n_gm_big)
    _pb_bytes(g_gm_big, 11, vi_in); _pb_bytes(g_gm_big, 11, vi_gb); _pb_bytes(g_gm_big, 12, vi_out)
    var m_gm_big = _load_temp_onnx("/tmp/flint_t5_gmbig_" + String(pid) + ".onnx", g_gm_big)
    m_gm_big.set_input("x", ga_8k^)
    m_gm_big.set_input("b", gb_8k^)
    var caught_onnx_gm = False
    try:
        m_gm_big.run()
    except:
        caught_onnx_gm = True
    assert_true(caught_onnx_gm, "ONNX Gemm [8192, 1] x [1, 8192] exceeding MAX_RANGE_ELEMS must raise Error")

    print("  t5 onnx & ops negative boundary checks: ok")


def _pb_f32_init(name: String, dims: List[Int], vals: List[Float32]) -> List[UInt8]:
    var t = List[UInt8]()
    for d in dims:
        _pb_varint(t, (1 << 3) | 0)
        _pb_varint(t, d)
    _pb_varint(t, (2 << 3) | 0)
    _pb_varint(t, DT_F32)
    _pb_str(t, 8, name)
    var raw = List[UInt8](length=len(vals) * 4, fill=0)
    var p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(raw.unsafe_ptr()))
    for k in range(len(vals)):
        p.unsafe_offset(k).unsafe_store[alignment=1](vals[k])
    _pb_bytes(t, 9, raw)
    return t^


def _load_temp_onnx(path: String, graph: List[UInt8]) raises -> OnnxModel:
    var model = List[UInt8]()
    _pb_bytes(model, 7, graph)
    assert_true(write_bytes(path, model))
    var m = OnnxModel.load(path, 1)
    _ = remove_file(path)
    return m^


def test_t5_gpu_product_and_overflow_guards() raises:
    var pid = Int(external_call["getpid", Int32]())

    # 1. PbReader.length_delimited signed 64-bit pointer+length overflow
    var pb_ovf: List[UInt8] = [
        UInt8(0x3A),
        UInt8(0x80), UInt8(0x80), UInt8(0x80), UInt8(0x80), UInt8(0x80), UInt8(0x80), UInt8(0x80), UInt8(0xC0), UInt8(0x7F),
    ]
    var pr_ovf = PbReader(Int(pb_ovf.unsafe_ptr()), Int(pb_ovf.unsafe_ptr()) + len(pb_ovf))
    pr_ovf.key()
    var caught_pb_ovf = False
    try:
        _ = pr_ovf.length_delimited()
    except:
        caught_pb_ovf = True
    _ = len(pb_ovf)
    assert_true(caught_pb_ovf, "PbReader.length_delimited signed overflow must raise Error")

    # 2. _parse_tensor byte-size multiplication overflow: dims=[2^62, 1], raw_data=b""
    var t_ovf = List[UInt8]()
    _pb_varint(t_ovf, (1 << 3) | 0); _pb_varint(t_ovf, 4611686018427387904)
    _pb_varint(t_ovf, (1 << 3) | 0); _pb_varint(t_ovf, 1)
    _pb_varint(t_ovf, (2 << 3) | 0); _pb_varint(t_ovf, DT_F32)
    _pb_str(t_ovf, 8, "w")
    _pb_bytes(t_ovf, 9, List[UInt8]())
    var g_ovf = List[UInt8]()
    _pb_bytes(g_ovf, 5, t_ovf)
    var p_tovf = "/tmp/flint_t5_tovf_" + String(pid) + ".onnx"
    var m_tovf = List[UInt8]()
    _pb_bytes(m_tovf, 7, g_ovf)
    assert_true(write_bytes(p_tovf, m_tovf))
    var caught_tovf = False
    try:
        _ = OnnxModel.load(p_tovf, 1)
    except:
        caught_tovf = True
    _ = remove_file(p_tovf)
    assert_true(caught_tovf, "_parse_tensor 2^62 * 4 byte-size overflow must raise Error")

    # 3. ConstantOfShape huge shape cap
    var vi_in = List[UInt8](); _pb_str(vi_in, 1, "x")
    var vi_out = List[UInt8](); _pb_str(vi_out, 1, "y")
    var n_cos = List[UInt8]()
    _pb_str(n_cos, 1, "x"); _pb_str(n_cos, 2, "y"); _pb_str(n_cos, 4, "ConstantOfShape")
    var g_cos_big = List[UInt8]()
    _pb_bytes(g_cos_big, 1, n_cos)
    _pb_bytes(g_cos_big, 11, vi_in)
    _pb_bytes(g_cos_big, 12, vi_out)
    var m_cos_big = _load_temp_onnx("/tmp/flint_t5_cosbig_" + String(pid) + ".onnx", g_cos_big)
    m_cos_big.set_input("x", Tensor.ints(DT_I64, [1], [100_000_000_000]))
    var caught_cos_big = False
    try:
        m_cos_big.run()
    except:
        caught_cos_big = True
    assert_true(caught_cos_big, "ConstantOfShape huge allocation must raise Error")

    # 4. GPU _gpu_product adversarial suite (always verifies CPU fallback / errors, and GPU path when available)
    # 4a. MatMul kk mismatch, kk==0, non-float x
    var n_mm = List[UInt8]()
    _pb_str(n_mm, 1, "x"); _pb_str(n_mm, 1, "w"); _pb_str(n_mm, 2, "y"); _pb_str(n_mm, 4, "MatMul")
    var g_mm = List[UInt8]()
    _pb_bytes(g_mm, 1, n_mm)
    _pb_bytes(g_mm, 5, _pb_f32_init("w", [1, 4], [Float32(1), 2, 3, 4]))
    _pb_bytes(g_mm, 11, vi_in); _pb_bytes(g_mm, 12, vi_out)
    var m_mm = _load_temp_onnx("/tmp/flint_t5_mm_" + String(pid) + ".onnx", g_mm)
    _ = m_mm.use_gpu()
    m_mm.set_input("x", Tensor.floats([2, 3], [Float32(1), 2, 3, 4, 5, 6]))
    var caught_mm_k = False
    try:
        m_mm.run()
    except:
        caught_mm_k = True
    assert_true(caught_mm_k, "MatMul inner dim mismatch must raise Error on GPU/CPU")

    m_mm.set_input("x", Tensor.ints(DT_I64, [1, 1], [1]))
    var caught_mm_i = False
    try:
        m_mm.run()
    except:
        caught_mm_i = True
    assert_true(caught_mm_i, "MatMul integer input must raise Error on GPU/CPU")

    var g_mm0 = List[UInt8]()
    _pb_bytes(g_mm0, 1, n_mm)
    _pb_bytes(g_mm0, 5, _pb_f32_init("w", [0, 2], List[Float32]()))
    _pb_bytes(g_mm0, 11, vi_in); _pb_bytes(g_mm0, 12, vi_out)
    var m_mm0 = _load_temp_onnx("/tmp/flint_t5_mm0_" + String(pid) + ".onnx", g_mm0)
    _ = m_mm0.use_gpu()
    m_mm0.set_input("x", Tensor.floats([2, 0], List[Float32]()))
    m_mm0.run()
    var y_mm0 = m_mm0.output("y")
    assert_equal(y_mm0.shape[0], 2)
    assert_equal(y_mm0.shape[1], 2)
    assert_equal(y_mm0.fget(0), Float32(0.0))

    # 4b. Gemm: 1D x, DT_I64 x, kk mismatch, [m,1] col bias when m==n, and weight-vs-bias key collision
    var n_gm = List[UInt8]()
    _pb_str(n_gm, 1, "x"); _pb_str(n_gm, 1, "w"); _pb_str(n_gm, 2, "y"); _pb_str(n_gm, 4, "Gemm")
    var g_gm = List[UInt8]()
    _pb_bytes(g_gm, 1, n_gm)
    _pb_bytes(g_gm, 5, _pb_f32_init("w", [4, 2], [Float32(1), 0, 0, 1, 1, 1, 0, 0]))
    _pb_bytes(g_gm, 11, vi_in); _pb_bytes(g_gm, 12, vi_out)
    var m_gm = _load_temp_onnx("/tmp/flint_t5_gm_" + String(pid) + ".onnx", g_gm)
    _ = m_gm.use_gpu()

    m_gm.set_input("x", Tensor.floats([4], [Float32(1), 2, 3, 4]))
    var caught_gm_1d = False
    try:
        m_gm.run()
    except:
        caught_gm_1d = True
    assert_true(caught_gm_1d, "Gemm 1D x must raise Error")

    m_gm.set_input("x", Tensor.ints(DT_I64, [2, 4], [1, 2, 3, 4, 5, 6, 7, 8]))
    var caught_gm_i = False
    try:
        m_gm.run()
    except:
        caught_gm_i = True
    assert_true(caught_gm_i, "Gemm integer x must raise Error")

    m_gm.set_input("x", Tensor.floats([2, 3], [Float32(1), 2, 3, 4, 5, 6]))
    var caught_gm_k = False
    try:
        m_gm.run()
    except:
        caught_gm_k = True
    assert_true(caught_gm_k, "Gemm kk mismatch must raise Error")

    # Gemm [m, 1] column bias when m == n == 2 (must broadcast down columns, not rows!)
    var n_gm_col = List[UInt8]()
    _pb_str(n_gm_col, 1, "x"); _pb_str(n_gm_col, 1, "w"); _pb_str(n_gm_col, 1, "b"); _pb_str(n_gm_col, 2, "y"); _pb_str(n_gm_col, 4, "Gemm")
    var g_gm_col = List[UInt8]()
    _pb_bytes(g_gm_col, 1, n_gm_col)
    _pb_bytes(g_gm_col, 5, _pb_f32_init("w", [2, 2], [Float32(1), 0, 0, 1]))
    _pb_bytes(g_gm_col, 5, _pb_f32_init("b", [2, 1], [Float32(10), 20]))
    _pb_bytes(g_gm_col, 11, vi_in); _pb_bytes(g_gm_col, 12, vi_out)
    var m_gm_col = _load_temp_onnx("/tmp/flint_t5_gmcol_" + String(pid) + ".onnx", g_gm_col)
    _ = m_gm_col.use_gpu()
    m_gm_col.set_input("x", Tensor.floats([2, 2], [Float32(1), 2, 3, 4]))
    m_gm_col.run()
    var y_gm_col = m_gm_col.output("y")
    assert_true(abs(y_gm_col.fget(0) - Float32(11.0)) < 1e-4)
    assert_true(abs(y_gm_col.fget(1) - Float32(12.0)) < 1e-4)
    assert_true(abs(y_gm_col.fget(2) - Float32(23.0)) < 1e-4)
    assert_true(abs(y_gm_col.fget(3) - Float32(24.0)) < 1e-4)

    # Gemm weight-id vs bias-id cache key isolation: node0 uses w(id=1) as weight, node1 uses id=1 as bias!
    var n_g0 = List[UInt8]()
    _pb_str(n_g0, 1, "x"); _pb_str(n_g0, 1, "shared"); _pb_str(n_g0, 2, "h"); _pb_str(n_g0, 4, "Gemm")
    var n_g1 = List[UInt8]()
    _pb_str(n_g1, 1, "h"); _pb_str(n_g1, 1, "w2"); _pb_str(n_g1, 1, "shared"); _pb_str(n_g1, 2, "y"); _pb_str(n_g1, 4, "Gemm")
    var g_key = List[UInt8]()
    _pb_bytes(g_key, 1, n_g0)
    _pb_bytes(g_key, 1, n_g1)
    _pb_bytes(g_key, 5, _pb_f32_init("shared", [1, 2], [Float32(2), 3]))
    _pb_bytes(g_key, 5, _pb_f32_init("w2", [2, 2], [Float32(1), 0, 0, 1]))
    _pb_bytes(g_key, 11, vi_in); _pb_bytes(g_key, 12, vi_out)
    var m_key = _load_temp_onnx("/tmp/flint_t5_key_" + String(pid) + ".onnx", g_key)
    _ = m_key.use_gpu()
    m_key.set_input("x", Tensor.floats([1, 1], [Float32(5)]))
    m_key.run()
    var y_key = m_key.output("y")
    assert_true(abs(y_key.fget(0) - Float32(12.0)) < 1e-4)  # 5*2 + 2
    assert_true(abs(y_key.fget(1) - Float32(18.0)) < 1e-4)  # 5*3 + 3

    # 4c. Conv: negative t_out, w.shape[1] != cin, bad bias shape
    var n_cv = List[UInt8]()
    _pb_str(n_cv, 1, "x"); _pb_str(n_cv, 1, "w"); _pb_str(n_cv, 1, "b"); _pb_str(n_cv, 2, "y"); _pb_str(n_cv, 4, "Conv")
    var g_cv = List[UInt8]()
    _pb_bytes(g_cv, 1, n_cv)
    _pb_bytes(g_cv, 5, _pb_f32_init("w", [2, 1, 5], List[Float32](length=10, fill=Float32(1.0))))
    _pb_bytes(g_cv, 5, _pb_f32_init("b", [1], [Float32(1.0)]))
    _pb_bytes(g_cv, 11, vi_in); _pb_bytes(g_cv, 12, vi_out)
    var m_cv = _load_temp_onnx("/tmp/flint_t5_cv_" + String(pid) + ".onnx", g_cv)
    _ = m_cv.use_gpu()

    m_cv.set_input("x", Tensor.floats([1, 1, 2], [Float32(1), 2]))  # t_out = -2
    var caught_cv_neg = False
    try:
        m_cv.run()
    except:
        caught_cv_neg = True
    assert_true(caught_cv_neg, "Conv negative t_out must raise Error")

    m_cv.set_input("x", Tensor.floats([1, 4, 8], List[Float32](length=32, fill=Float32(1.0))))  # cin=4 != w.shape[1]=1
    var caught_cv_cin = False
    try:
        m_cv.run()
    except:
        caught_cv_cin = True
    assert_true(caught_cv_cin, "Conv cin mismatch must raise Error")

    m_cv.set_input("x", Tensor.floats([1, 1, 8], List[Float32](length=8, fill=Float32(1.0))))  # b len 1 != cout 2
    var caught_cv_b = False
    try:
        m_cv.run()
    except:
        caught_cv_b = True
    assert_true(caught_cv_b, "Conv bad bias length must raise Error")

    # 4d. Direct GpuMatmul zero-dim & null-ptr guards when GPU is active
    if m_cv.gpu:
        var dummy = List[Float32](length=4, fill=Float32(1.0))
        var dptr = Int(dummy.unsafe_ptr())
        var w_zero_raised = False
        try:
            _ = m_cv.gpu.value().weight(999, dptr, 0, 2, False)
        except:
            w_zero_raised = True
        assert_true(w_zero_raised, "GpuMatmul.weight(rows=0) must raise Error")

        var w_ok = m_cv.gpu.value().weight(1000, dptr, 2, 2, False)
        var mm_zero_raised = False
        try:
            m_cv.gpu.value().matmul(dptr, w_ok, dptr, 0, 0, 0, 2, 2)
        except:
            mm_zero_raised = True
        assert_true(mm_zero_raised, "GpuMatmul.matmul(m=0) must raise Error")

        var mm_kzero_raised = False
        try:
            m_cv.gpu.value().matmul(dptr, w_ok, dptr, 0, 0, 2, 0, 2)
        except:
            mm_kzero_raised = True
        assert_true(mm_kzero_raised, "GpuMatmul.matmul(k=0) must raise Error")
        _ = len(dummy)

    # 4e. Round 18: softmax(+Inf/-Inf) & layer_norm(eps < 0 / NaN)
    var pinf_f = f32_from_bits(0x7F800000)
    var ninf_f = f32_from_bits(0xFF800000)
    var nan_f = f32_from_bits(0x7FC00000)
    var sm_inf = softmax(Tensor.floats([2], [pinf_f, Float32(1.0)]), 0)
    assert_true(abs(sm_inf.fget(0) - Float32(1.0)) < 1e-5)
    assert_true(abs(sm_inf.fget(1) - Float32(0.0)) < 1e-5)
    var sm_ninf = softmax(Tensor.floats([2], [ninf_f, ninf_f]), 0)
    assert_true(abs(sm_ninf.fget(0) - Float32(0.5)) < 1e-5)
    var ln_bad_raised = False
    try:
        _ = layer_norm(Tensor.floats([2], [1.0, 2.0]), Tensor.floats([2], [1.0, 1.0]), Tensor(), False, Float32(-1.0))
    except:
        ln_bad_raised = True
    assert_true(ln_bad_raised, "layer_norm with negative eps must raise Error")
    var ln_nan_raised = False
    try:
        _ = layer_norm(Tensor.floats([2], [1.0, 2.0]), Tensor.floats([2], [1.0, 1.0]), Tensor(), False, nan_f)
    except:
        ln_nan_raised = True
    assert_true(ln_nan_raised, "layer_norm with NaN eps must raise Error")

    var ln_in_nan_raised = False
    try:
        _ = layer_norm(Tensor.floats([2], [nan_f, 1.0]), Tensor.floats([2], [1.0, 1.0]), Tensor(), False, Float32(1.0e-5))
    except:
        ln_in_nan_raised = True
    assert_true(ln_in_nan_raised, "layer_norm with NaN input must raise Error")

    var ln_in_inf_raised = False
    try:
        _ = layer_norm(Tensor.floats([2], [pinf_f, 1.0]), Tensor.floats([2], [1.0, 1.0]), Tensor(), False, Float32(1.0e-5))
    except:
        ln_in_inf_raised = True
    assert_true(ln_in_inf_raised, "layer_norm with +Inf input must raise Error")

    var ln_scale_nan_raised = False
    try:
        _ = layer_norm(Tensor.floats([2], [1.0, 2.0]), Tensor.floats([2], [nan_f, 1.0]), Tensor(), False, Float32(1.0e-5))
    except:
        ln_scale_nan_raised = True
    assert_true(ln_scale_nan_raised, "layer_norm with NaN scale must raise Error")

    var ln_bias_inf_raised = False
    try:
        _ = layer_norm(Tensor.floats([2], [1.0, 2.0]), Tensor.floats([2], [1.0, 1.0]), Tensor.floats([2], [ninf_f, 0.0]), True, Float32(1.0e-5))
    except:
        ln_bias_inf_raised = True
    assert_true(ln_bias_inf_raised, "layer_norm with -Inf bias must raise Error")

    var ln_ext = layer_norm(
        Tensor.floats([2], [Float32(3.4e38), Float32(3.4e38)]),
        Tensor.floats([2], [Float32(1.0), Float32(1.0)]),
        Tensor.floats([2], [Float32(0.0), Float32(0.0)]),
        True,
        Float32(1.0e-5),
    )
    assert_true(isfinite(ln_ext.fget(0)) and isfinite(ln_ext.fget(1)), "layer_norm([3.4e38, 3.4e38]) must be finite")
    assert_true(abs(ln_ext.fget(0)) < 1e-5 and abs(ln_ext.fget(1)) < 1e-5, "layer_norm([3.4e38, 3.4e38]) must be [0, 0]")

    var ln_spread = layer_norm(
        Tensor.floats([2], [Float32(-3.4e38), Float32(3.4e38)]),
        Tensor.floats([2], [Float32(1.0), Float32(1.0)]),
        Tensor(),
        False,
        Float32(1.0e-5),
    )
    assert_true(abs(ln_spread.fget(0) + Float32(1.0)) < 1e-4, "layer_norm([-3.4e38, 3.4e38])[0] must be -1")
    assert_true(abs(ln_spread.fget(1) - Float32(1.0)) < 1e-4, "layer_norm([-3.4e38, 3.4e38])[1] must be +1")

    print("  t5 gpu _gpu_product & overflow guards: ok")


def test_t5_onnx_split_num_outputs_uneven() raises:
    var pid = Int(external_call["getpid", Int32]())
    var attr_no = List[UInt8]()
    _pb_str(attr_no, 1, "num_outputs")
    _pb_varint(attr_no, (3 << 3) | 0)
    _pb_varint(attr_no, 4)

    var node_split = List[UInt8]()
    _pb_str(node_split, 1, "x")
    _pb_str(node_split, 2, "y0")
    _pb_str(node_split, 2, "y1")
    _pb_str(node_split, 2, "y2")
    _pb_str(node_split, 2, "y3")
    _pb_str(node_split, 4, "Split")
    _pb_bytes(node_split, 5, attr_no)

    var vi_in = List[UInt8]()
    _pb_str(vi_in, 1, "x")
    var g = List[UInt8]()
    _pb_bytes(g, 1, node_split)
    _pb_bytes(g, 11, vi_in)
    for out_name in ["y0", "y1", "y2", "y3"]:
        var vi_o = List[UInt8]()
        _pb_str(vi_o, 1, String(out_name))
        _pb_bytes(g, 12, vi_o)

    var model_bytes = List[UInt8]()
    _pb_bytes(model_bytes, 7, g)

    var path_s = "/tmp/flint_t5_split_" + String(pid) + ".onnx"
    assert_true(write_bytes(path_s, model_bytes))
    var m = OnnxModel.load(path_s, 1)
    _ = remove_file(path_s)
    m.set_input("x", Tensor.floats([5], [1.0, 2.0, 3.0, 4.0, 5.0]))
    m.run()

    var y0 = m.output("y0")
    var y1 = m.output("y1")
    var y2 = m.output("y2")
    var y3 = m.output("y3")
    assert_equal(y0.numel(), 2)
    assert_equal(y1.numel(), 1)
    assert_equal(y2.numel(), 1)
    assert_equal(y3.numel(), 1)
    assert_equal(y0.fget(0), Float32(1.0))
    assert_equal(y0.fget(1), Float32(2.0))
    assert_equal(y1.fget(0), Float32(3.0))
    assert_equal(y2.fget(0), Float32(4.0))
    assert_equal(y3.fget(0), Float32(5.0))
    print("  t5 onnx Split(num_outputs=4) on [5] -> [2, 1, 1, 1]: ok")


def main() raises:
    print("=== Flint v5 flint.ml ONNX + MAX GPU + MotionBricks Suite ===")
    test_broadcasting()
    test_tiny_ops()
    test_kimodo()
    test_motionbricks_math()
    test_motionbricks_soma()
    test_t5_onnx_and_ops_negative()
    test_t5_gpu_product_and_overflow_guards()
    test_t5_onnx_split_num_outputs_uneven()
    print("=== ALL FLINT v5 ML TESTS PASSED ===")
