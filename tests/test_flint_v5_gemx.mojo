"""Flint v5 GEM-X suite (`test_flint_v5_gemx.mojo`).

Verifies, without model files:
  1. `bf16_round` (the ViTPose rotary round trip): ties to even, NaN kept.
  2. OpenCV-exact crops: `affine_from_points` matches `cv2.getAffineTransform` and
     `vitpose_crop` / `pose_crop` match `cv2.warpAffine(INTER_LINEAR)` checksums on a pattern
     image (values from OpenCV 5.1).
  3. Boxes: `square_crop`, the 5-frame smoothing, `keypoint_box`; the native77 flip table.
  4. Heatmap decoding: a peak with a stronger right / weaker upper neighbour lands a quarter pixel
     off, mapped through the UDP box convention.
  5. Motion plumbing: the 120-frame attention window, the scipy Gaussian kernel, 6-D rotations,
     the static-camera world rollout and the SOMA tables.
  6. `gemx_clip` on the neutral SOMA-X skeleton: a rest prediction is the T-pose, a posed one
     matches SOMA-X forward kinematics (joint positions from py-soma-x 0.3.3), and `write_bvh` /
     `load_bvh` round-trip it; with `MOTIONBRICKS_DIR`, a MotionBricks BVH skeleton's T-pose too.
  7. With `GEMX_DIR` (graphs from `tools/gemx/export.py`), `GEMX_REF` (dumps of
     `tools/gemx/reference.py`) and `GEMX_FRAMES` (the PNG frames), parity with PyTorch stage by
     stage and end to end, on the GPU when present (`GEMX_CPU=1` forces the CPU).
"""
from std.math import abs, sqrt
from std.time import perf_counter_ns
from std.testing import assert_true, assert_equal
from flint.ml.tensor import bf16_round, f32_from_bits
from flint.ml.gemx import (
    GxFrame, affine_from_points, vitpose_crop, pose_crop, square_crop, boxes_to_crops,
    keypoint_box, soma77_flip, decode_heatmaps, motion_attention_bias, gaussian_kernel,
    rotation_6d_to_matrix, GemxMotion, decode_motion, static_cam_angvel, soma78_joint_names,
    soma78_parents, soma78_t_pose_orient, gemx_clip, soma77_skeleton, write_bvh, Gemx, load_frames,
    motion_observations, GX_MOTION_DIM, GX_JOINTS,
)
from flint.ml.motionbricks import MbSkeleton, MbXform, MbVec3, MbQuat, world_pose, load_bvh
from flint.platform.os import env_value, read_bytes


def _pattern() -> GxFrame:
    var f = GxFrame(37, 29)
    for y in range(29):
        for x in range(37):
            for c in range(3):
                f.rgb[(y * 37 + x) * 3 + c] = UInt8((x * 7 + y * 13 + c * 50) % 256)
    return f^


def _sums(v: List[UInt8]) -> Tuple[Int, Int]:
    var s = 0
    var w = 0
    for i in range(len(v)):
        s += Int(v[i])
        w += Int(v[i]) * (i % 251)
    return (s, w)


def test_bf16() raises:
    assert_equal(bf16_round(1.0), 1.0)
    # 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties to the even mantissa (1.0)
    assert_equal(bf16_round(1.00390625), 1.0)
    assert_equal(bf16_round(1.01171875), 1.015625)   # 1 + 3 * 2^-8 rounds up to even
    var nan = Float32(0) / Float32(0)
    var r = bf16_round(nan)
    assert_true(r != r, "NaN stays NaN")
    print("  bf16 round: ok")


def test_crops() raises:
    var img = _pattern()
    var half = Float64(Float32(21.7)) / 2
    var src: List[Float32] = [Float32(Float64(Float32(17.3)) - half), Float32(Float64(Float32(12.9)) - half),
           Float32(Float64(Float32(17.3)) + half), Float32(Float64(Float32(12.9)) - half),
           Float32(17.3), Float32(12.9)]
    var m = affine_from_points(src, [0.0, 0.0, 255.0, 0.0, 127.5, 127.5])
    var want: List[Float64] = [11.751151660579852, 0.0, -75.7949147626142, 0.0, 11.751151660579852, -24.089851938771478]
    for k in range(6):
        assert_true(abs(m[k] - want[k]) < 1e-9, "getAffineTransform")
    var a = _sums(vitpose_crop(img, 17.3, 12.9, 21.7))
    assert_equal(a[0], 19179493)
    assert_equal(a[1], 2396299353)
    var b = _sums(pose_crop(img, 17.3, 12.9, 21.7))
    assert_equal(b[0], 25434930)
    assert_equal(b[1], 3180141878)
    var c = _sums(vitpose_crop(img, 30.25, 4.5, 40.0))
    assert_equal(c[0], 8284580)
    assert_equal(c[1], 1035386776)
    var d = _sums(pose_crop(img, 30.25, 4.5, 40.0))
    assert_equal(d[0], 10378608)
    assert_equal(d[1], 1296954154)
    print("  OpenCV-exact crops: ok")


def test_boxes() raises:
    var q = square_crop(10, 20, 40, 100)          # h 80 > w / 0.75 = 40
    assert_equal(q[0], 25.0)
    assert_equal(q[1], 60.0)
    assert_equal(q[2], Float32(80) * Float32(1.2))
    # 6 frames: smoothing applies; a constant box stays put, clamped to the image
    var xyxy = List[Float32]()
    for _ in range(6):
        for v in [Float32(-5), 10, 50, 90]:
            xyxy.append(v)
    var xys = boxes_to_crops(xyxy, 64, 64)
    # clamped to x 0..50, y 10..63: centre (25, 36.5), side 1.2 * max(53, 50 / 0.75)
    assert_true(abs(xys[0] - 25.0) < 1e-5 and abs(xys[1] - 36.5) < 1e-5, "box centre")
    assert_true(abs(xys[2] - 80.0) < 1e-3, "box size")
    var kp = List[Float32](length=GX_JOINTS * 3, fill=0.0)
    for j in range(5):
        kp[3 * j] = Float32(10 + j * 10)
        kp[3 * j + 1] = Float32(20 + j * 5)
        kp[3 * j + 2] = 0.9
    var nb = keypoint_box(kp, 0, 100, 100)
    assert_equal(len(nb), 4)
    assert_true(abs(nb[0] - 4.0) < 1e-5 and abs(nb[3] - 43.0) < 1e-5, "padded keypoint box")
    kp[2] = 0.1
    kp[5] = 0.1
    assert_equal(len(keypoint_box(kp, 0, 100, 100)), 0)
    var f = soma77_flip()
    assert_equal(f[9], 10)
    assert_equal(f[11], 39)
    assert_equal(f[66], 38)
    assert_equal(f[67], 72)
    assert_equal(f[0], 0)
    for k in range(GX_JOINTS):
        assert_equal(f[f[k]], k)
    print("  boxes / flip table: ok")


def test_heatmaps() raises:
    comptime HM = 77 * 64 * 48
    var heat = List[Float32](length=2 * HM, fill=0.0)
    # joint 0 peak at (x 20, y 30) in the original; right neighbour stronger, lower one weaker
    heat[30 * 48 + 20] = 1.0
    heat[30 * 48 + 21] = 0.5
    heat[29 * 48 + 20] = 0.3
    # the flipped map holds the mirrored peak for joint 0 (which flips to itself)
    heat[HM + 30 * 48 + (47 - 20)] = 1.0
    var kp = List[Float32](length=77 * 3, fill=0.0)
    decode_heatmaps(heat, 0, HM, 100, 200, 64, kp, 0)
    var sx = Float32(64) * 24 / 32 / 200
    var sy = Float32(64) / 200
    var want_x = Float32(20.25) / 47 * (sx * 200) + 100 - sx * 100
    var want_y = Float32(29.75) / 63 * (sy * 200) + 200 - sy * 100
    assert_true(abs(kp[0] - want_x) < 1e-5 and abs(kp[1] - want_y) < 1e-5, "quarter-pixel peak")
    assert_true(abs(kp[2] - 1.0) < 1e-6, "confidence = averaged peak")
    print("  heatmap decode: ok")


def test_motion_math() raises:
    var b = motion_attention_bias(130)
    assert_equal(b[0], 0.0)
    assert_equal(b[119], 0.0)
    assert_true(b[120] < -1e29, "frame 0 sees the first 120 frames")
    assert_true(b[129 * 130 + 9] < -1e29 and b[129 * 130 + 10] == 0.0, "the last frame sees the last 120")
    assert_equal(len(motion_attention_bias(40)), 1600)
    var k = gaussian_kernel(3.0, 12)
    assert_equal(len(k), 25)
    assert_true(abs(k[0] - 4.461134449229576e-05) < 1e-10 and abs(k[12] - 0.13298453390598297) < 1e-8, "scipy kernel")
    var d6: List[Float32] = [2, 0, 0, 1, 3, 0]
    var r = List[Float32](length=9, fill=0.0)
    rotation_6d_to_matrix(d6, 0, r, 0)
    assert_true(abs(r[0] - 1) < 1e-6 and abs(r[4] - 1) < 1e-6 and abs(r[8] - 1) < 1e-6, "6-D identity")
    # a still, upright clip: identity body, standing in front of a static camera
    var t_n = 8
    var m = GemxMotion()
    m.frames = t_n
    m.width = 640
    m.height = 480
    for _ in range(t_n):
        for v in [Float32(320), 240, 200]:
            m.bbx_xys.append(v)
    var mean = List[Float32](length=GX_MOTION_DIM, fill=0.0)
    var std = List[Float32](length=GX_MOTION_DIM, fill=1.0)
    var raw = List[Float32](length=t_n * GX_MOTION_DIM, fill=0.0)
    var cam = List[Float32]()
    for t in range(t_n):
        var o = t * GX_MOTION_DIM
        for j in range(76):
            raw[o + 6 * j] = 1
            raw[o + 6 * j + 4] = 1
        # camera orientation: 180 degrees about x (camera y down)
        raw[o + 640] = 1
        raw[o + 644] = -1
        # gravity view orientation, the same
        raw[o + 646] = 1
        raw[o + 650] = -1
        raw[o + 652 + 2] = 0.1      # 0.1 m per frame along the body's local z
        for v in [Float32(1), 0.5, -0.2]:
            cam.append(v)
    decode_motion(m, raw, cam, mean, std, static_cam_angvel(t_n))
    assert_true(abs(m.incam_transl[2] - 2 * motion_focal_ref(640, 480) / 200) < 1e-3, "depth from s and box size")
    assert_true(abs(m.world_rot[4] - 1) < 1e-5 and abs(m.world_rot[8] + 1) < 1e-5, "world: y up")
    assert_true(abs(m.world_transl[3 * 7 + 2] + 0.7) < 1e-4, "integrated walk")
    assert_equal(len(m.scale), 57)
    assert_equal(m.scale[1], 1.0)
    var names = soma78_joint_names()
    var parents = soma78_parents()
    assert_equal(len(names), 78)
    assert_equal(len(soma78_t_pose_orient()), 312)
    assert_equal(names[1], "Hips")
    assert_equal(names[13], "LeftArm")
    assert_equal(parents[13], 12)
    print("  motion math / tables: ok")


def motion_focal_ref(w: Int, h: Int) -> Float32:
    return Float32(sqrt(Float64(w * w + h * h)))


def _set_rot(mut m: GemxMotion, joint: String, r: List[Float32]):
    var names = soma78_joint_names()
    for s in range(2, 78):
        if names[s] == joint:
            for k in range(9):
                m.body_rot[(s - 2) * 9 + k] = r[k]


def test_clip() raises:
    var sk = soma77_skeleton()
    var m = GemxMotion()
    m.frames = 2
    for _ in range(2):
        for _ in range(76):
            for v in [Float32(1), 0, 0, 0, 1, 0, 0, 0, 1]:
                m.body_rot.append(v)
        for v in [Float32(1), 0, 0, 0, 1, 0, 0, 0, 1]:
            m.world_rot.append(v)
            m.incam_rot.append(v)
        for v in [Float32(0.5), 0, 0]:
            m.world_transl.append(v)
            m.incam_transl.append(v)
    var clip = gemx_clip(m, sk)
    var w = world_pose(clip.skeleton, clip.frames[1], MbXform.identity())
    var lh = w[sk.joint_index("LeftHand")].p
    var rh = w[sk.joint_index("RightHand")].p
    var head = w[sk.joint_index("Head")].p
    var hips = w[sk.joint_index("Hips")].p
    assert_true(abs(lh.y - rh.y) < 1e-4 and lh.x - rh.x > 0.5, "T-pose arms")
    assert_true(head.y > hips.y + 0.3, "head above hips")
    assert_true(abs(hips.x - 0.5) < 1e-5, "root translation")
    var path = env_value("TMPDIR")
    if path == "":
        path = "/tmp"
    path += "/flint_gemx_test.bvh"
    write_bvh(clip, path)
    var back = load_bvh(path)
    assert_equal(back.num_frames(), 2)
    assert_equal(back.skeleton.num_joints(), sk.num_joints())
    var w2 = world_pose(back.skeleton, back.frames[1], MbXform.identity())
    for j in range(sk.num_joints()):
        assert_true((w2[j].p - w[j].p).length() < 1e-3, "BVH round trip")
    # SOMA-X forward kinematics of a posed neutral body (py-soma-x 0.3.3, camera space)
    var pm = GemxMotion()
    pm.frames = 1
    for _ in range(76):
        for v in [Float32(1), 0, 0, 0, 1, 0, 0, 0, 1]:
            pm.body_rot.append(v)
    _set_rot(pm, "Spine2", [0.955373752, 0.0148754159, 0.295024941, 0.0148754159, 0.995041528, -0.0983416469, -0.295024941, 0.0983416469, 0.95041528])
    _set_rot(pm, "LeftArm", [0.696706709, 0.717356091, 0.0, -0.717356091, 0.696706709, 0.0, 0.0, 0.0, 1.0])
    _set_rot(pm, "RightForeArm", [1.0, 0.0, 0.0, 0.0, 0.877582562, -0.479425539, 0.0, 0.479425539, 0.877582562])
    _set_rot(pm, "LeftLeg", [1.0, 0.0, 0.0, 0.0, 0.825335615, 0.564642473, 0.0, -0.564642473, 0.825335615])
    _set_rot(pm, "Neck1", [0.901655595, -0.193399683, 0.386799367, 0.193399683, 0.980331119, 0.0393377619, -0.386799367, 0.0393377619, 0.921324476])
    pm.incam_rot = [0.916477126, -0.135841448, -0.376320047, 0.0572316851, 0.975434449, -0.212725574, 0.395972488, 0.173420693, 0.901737796]
    pm.incam_transl = [0.3, -0.1, 2.0]
    var pc = gemx_clip(pm, sk, world=False)
    var pw = world_pose(pc.skeleton, pc.frames[0], MbXform.identity())
    var fk_err: Float32 = 0
    fk_err = max(fk_err, (pw[sk.joint_index("Head")].p - MbVec3(0.199633986, 0.445592374, 2.16902542)).length())
    fk_err = max(fk_err, (pw[sk.joint_index("LeftHand")].p - MbVec3(0.819471538, 0.0206941813, 2.0497427)).length())
    fk_err = max(fk_err, (pw[sk.joint_index("RightHand")].p - MbVec3(-0.442531258, 0.21550557, 2.00910926)).length())
    fk_err = max(fk_err, (pw[sk.joint_index("LeftFoot")].p - MbVec3(0.325270355, -0.919638753, 2.34441662)).length())
    fk_err = max(fk_err, (pw[sk.joint_index("RightToeEnd")].p - MbVec3(0.247999579, -1.07450843, 1.976915)).length())
    print("    SOMA-X forward kinematics: max joint error " + String(fk_err) + " m")
    assert_true(fk_err < 1e-4, "joint positions match SOMA-X")
    var mb = env_value("MOTIONBRICKS_DIR")
    if mb != "":
        # a MotionBricks SOMA BVH skeleton: the rest prediction is its T-pose
        var real = load_bvh(mb + "/assets/motions/soma/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh")
        var rc = gemx_clip(m, real.skeleton)
        var rw = world_pose(rc.skeleton, rc.frames[0], MbXform.identity())
        var l = rw[real.skeleton.joint_index("LeftHand")].p
        var r = rw[real.skeleton.joint_index("RightHand")].p
        var hd = rw[real.skeleton.joint_index("Head")].p
        var hp = rw[real.skeleton.joint_index("Hips")].p
        var lf = rw[real.skeleton.joint_index("LeftFoot")].p
        print("    MotionBricks skeleton T-pose: hands " + String(l.x) + "," + String(l.y) + " / " + String(r.x) + "," + String(r.y)
              + ", head y " + String(hd.y) + ", hips y " + String(hp.y) + ", left foot y " + String(lf.y))
        assert_true(abs(l.y - r.y) < 0.02 and l.x - r.x > 1.2, "MotionBricks T-pose arms")
        assert_true(hd.y > hp.y + 0.4 and lf.y < hp.y - 0.7, "MotionBricks T-pose upright")
    print("  clip / BVH: ok")


# ---------------------------------------------------------------- reference parity


def read_f32(path: String) raises -> List[Float32]:
    var b = read_bytes(path)
    if len(b) == 0:
        raise Error("cannot read " + path)
    var n = len(b) // 4
    var out = List[Float32](capacity=n)
    for i in range(n):
        var u = UInt32(b[4 * i]) | (UInt32(b[4 * i + 1]) << 8) | (UInt32(b[4 * i + 2]) << 16) | (UInt32(b[4 * i + 3]) << 24)
        out.append(f32_from_bits(u))
    return out^


def max_diff(a: List[Float32], b: List[Float32], n: Int = -1) -> Float64:
    var m = 0.0
    var k = min(len(a), len(b)) if n < 0 else n
    for i in range(k):
        m = max(m, abs(Float64(a[i]) - Float64(b[i])))
    return m


def max_abs(a: List[Float32]) -> Float64:
    var m = 0.0
    for v in a:
        m = max(m, abs(Float64(v)))
    return m


def report(name: String, a: List[Float32], want: List[Float32], n: Int = -1) -> Float64:
    var d = max_diff(a, want, n)
    print("    " + name + ": max |diff| " + String(d) + " (max |ref| " + String(max_abs(want)) + ")")
    return d


def test_reference() raises:
    var dir = env_value("GEMX_DIR")
    var rdir = env_value("GEMX_REF")
    var frames_dir = env_value("GEMX_FRAMES")
    if dir == "" or rdir == "" or frames_dir == "":
        print("  reference parity: skipped (set GEMX_DIR, GEMX_REF and GEMX_FRAMES)")
        return
    var xyxy = read_f32(rdir + "/boxes_xyxy.f32")
    var t_n = len(xyxy) // 4
    var t0 = perf_counter_ns()
    var frames = load_frames(frames_dir, t_n)
    print("    " + String(t_n) + " frames decoded in " + String(Float64(perf_counter_ns() - t0) / 1e6) + " ms")
    var w = frames[0].width
    var h = frames[0].height
    var ref_xys = read_f32(rdir + "/bbx_xys.f32")
    var xys = boxes_to_crops(xyxy, w, h)
    assert_true(report("box crops (cx, cy, size)", xys, ref_xys) < 1e-3, "box post-processing")
    # crops: the first frames' network inputs, bit for bit
    var ref_crops = read_f32(rdir + "/vitpose_crops.f32")
    var k_img = len(ref_crops) // (3 * 256 * 192)
    var mismatch = 0
    for i in range(k_img):
        var crop = vitpose_crop(frames[i], ref_xys[3 * i], ref_xys[3 * i + 1], ref_xys[3 * i + 2])
        for c in range(3):
            for y in range(256):
                for x in range(192):
                    var mean = SIMD[DType.float32, 4](0.485, 0.456, 0.406, 0)
                    var sd = SIMD[DType.float32, 4](0.229, 0.224, 0.225, 1)
                    var v = (Float32(crop[(y * 192 + x) * 3 + c]) / 255.0 - mean[c]) / sd[c]
                    if v != ref_crops[((i * 3 + c) * 256 + y) * 192 + x]:
                        mismatch += 1
    print("    ViTPose crops: " + String(mismatch) + " of " + String(len(ref_crops)) + " values differ")
    assert_equal(mismatch, 0)
    var cpu = env_value("GEMX_CPU") == "1"
    t0 = perf_counter_ns()
    var g = Gemx.load(dir, not cpu)
    print("    models loaded in " + String(Float64(perf_counter_ns() - t0) / 1e6) + " ms (GPU: " + String(g.gpu) + ")")
    # stage 1: keypoints of every frame in the reference crops
    var ref_kp = read_f32(rdir + "/kp2d.f32")
    t0 = perf_counter_ns()
    var kp = g.keypoints(frames, ref_xys)
    var ms_kp = Float64(perf_counter_ns() - t0) / 1e6
    var d_kp = report("native77 keypoints (px, conf)", kp, ref_kp)
    var moved = 0
    for i in range(t_n * GX_JOINTS):
        var dx = abs(kp[3 * i] - ref_kp[3 * i]) + abs(kp[3 * i + 1] - ref_kp[3 * i + 1])
        if dx > 1e-3:
            moved += 1
    print("    keypoints off by a heatmap step (bf16 ties): " + String(moved) + " of " + String(t_n * GX_JOINTS))
    # stage 2: pose tokens
    var ref_tok = read_f32(rdir + "/pose_tokens.f32")
    t0 = perf_counter_ns()
    var tok = g.pose_tokens(frames, ref_xys)
    var ms_tok = Float64(perf_counter_ns() - t0) / 1e6
    var d_tok = report("Body Pose tokens", tok, ref_tok)
    # stage 3: Motion on the reference inputs (the rotation parity claim)
    var ref_x = read_f32(rdir + "/motion_x.f32")
    var ref_rot = read_f32(rdir + "/body_pose_R.f32")
    t0 = perf_counter_ns()
    var m = g.motion(ref_kp, ref_xys, ref_tok, t_n, w, h)
    var ms_motion = Float64(perf_counter_ns() - t0) / 1e6
    var d_x = report("Motion output (normalized)", m.motion, ref_x)
    var d_rot = report("body joint rotations (matrices)", m.body_rot, ref_rot)
    var d_or = report("camera Hips rotation", m.incam_rot, read_f32(rdir + "/incam_orient_R.f32"))
    var d_tr = report("camera translation (m)", m.incam_transl, read_f32(rdir + "/incam_transl.f32"))
    # GEM-X's own world rollout goes through float32 axis-angle near 180 degrees (~1e-2 error);
    # Flint is checked against the float64 rollout of the same network output
    _ = report("world Hips rotation vs float32 GEM-X", m.world_rot, read_f32(rdir + "/world_orient_R.f32"))
    var d_wor = report("world Hips rotation vs float64 rollout", m.world_rot, read_f32(rdir + "/world_orient_R_f64.f32"))
    var d_wtr = report("world translation vs float64 rollout (m)", m.world_transl, read_f32(rdir + "/world_transl_f64.f32"))
    _ = report("identity coefficients", m.identity, read_f32(rdir + "/identity.f32"))
    _ = report("scale parameters", m.scale, read_f32(rdir + "/scale.f32"))
    assert_true(d_rot < 1e-4 and d_or < 1e-4 and d_wor < 1e-4, "joint rotations within 1e-4")
    assert_true(d_tr < 1e-3 and d_wtr < 1e-3, "translations within 1 mm")
    # end to end: Flint keypoints and tokens into Motion
    var e = g.motion(kp, ref_xys, tok, t_n, w, h)
    _ = report("end to end: body joint rotations", e.body_rot, ref_rot)
    _ = report("end to end: camera translation (m)", e.incam_transl, read_f32(rdir + "/incam_transl.f32"))
    _ = report("end to end: world translation (m)", e.world_transl, read_f32(rdir + "/world_transl_f64.f32"))
    # the world clip stands upright: head above hips in every frame
    var clip = gemx_clip(e)
    var upright = 0
    for f in range(clip.num_frames()):
        var wp = world_pose(clip.skeleton, clip.frames[f], MbXform.identity())
        if wp[clip.skeleton.joint_index("Head")].p.y > wp[0].p.y + 0.3:
            upright += 1
    print("    world clip upright in " + String(upright) + " of " + String(clip.num_frames()) + " frames")
    assert_equal(upright, clip.num_frames())
    print("    timings: keypoints " + String(ms_kp) + " ms, pose tokens " + String(ms_tok) + " ms, motion "
          + String(ms_motion) + " ms for " + String(t_n) + " frames (crops " + String(Float64(g.ns_crop) / 1e6) + " ms)")
    # bf16 ties may move a keypoint by a heatmap step; nearly all must match exactly
    assert_true(moved * 100 < t_n * GX_JOINTS and d_kp < 20.0, "keypoint parity")
    assert_true(d_tok < 0.05 and d_x < 1e-3, "per-stage parity")
    print("  reference parity: ok")


def main() raises:
    print("=== Flint v5 GEM-X Suite ===")
    test_bf16()
    test_crops()
    test_boxes()
    test_heatmaps()
    test_motion_math()
    test_clip()
    test_reference()
    print("=== ALL FLINT v5 GEM-X TESTS PASSED ===")
