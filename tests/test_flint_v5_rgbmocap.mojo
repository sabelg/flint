"""Verification Suite for Flint v5 Pure-Mojo RGBMocap Pipeline & Motion Cleanup (`tests/test_flint_v5_rgbmocap.mojo`).

Verifies:
  1. `probe_mocap_backends`: commercial-safe GEM-X & RTMPose+MotionBERT metadata,
     non-throwing empty/partial/populated cache probing.
  2. `track_person_boxes` & `bbox_iou`: multi-person greedy IoU tracking, occlusion gap
     bridging (`<= max_missed_frames`), primary track selection, and post-gap track split.
  3. `generate_demo_mocap_take` + `export_mocap_take_bvh` + `mocap_take_to_native_animation`:
     deterministic SOMA-77 take invariants, valid ZYX-Euler BVH export, and `NativeAnimation` channels.
  4. `OneEuroScalar` / `OneEuroVec3` / `OneEuroQuat`: jitter reduction, quaternion double-cover
     hemisphere-flip continuity, NaN/Inf/`dt <= 0` safety, and `reset()`.
  5. `estimate_ground_plane_y`, `detect_contact_intervals`, `lock_contact_positions`,
     `solve_two_bone_leg_cleanup` (exact thigh & shin bone length preservation), and
     `apply_mocap_cleanup_to_take` (`foot_slide_after_cm < 1.5 cm`, `>75%` slide reduction,
     jitter reduction, and lockstep determinism).
"""

from std.collections import List
from std.ffi import external_call
from std.math import abs, isfinite, sin, cos
from std.memory import bitcast
from std.testing import assert_equal, assert_false, assert_true

from flint.anim.mocap_filter import (
    ContactCleanupConfig,
    ContactInterval,
    OneEuroConfig,
    OneEuroQuat,
    OneEuroScalar,
    OneEuroVec3,
    apply_mocap_cleanup_to_take,
    detect_contact_intervals,
    estimate_ground_plane_y,
    filter_quat_series,
    filter_vec3_series,
    lock_contact_positions,
    measure_foot_slide_cm,
    measure_series_jitter,
    solve_two_bone_leg_cleanup,
)
from flint.anim.soma_retarget import retarget_soma
from flint.assets.model import NativeModel, NativeNode, NativeSkin
from flint.core.math import Quat, Vec3
from flint.ml.rgbmocap import (
    MOCAP_BACKEND_GEMX,
    MOCAP_BACKEND_RTMPOSE_MBERT,
    MocapBBox,
    SOMA77_LEFT_FOOT,
    SOMA77_RIGHT_FOOT,
    bbox_iou,
    export_mocap_take_bvh,
    generate_demo_mocap_take,
    mocap_take_to_mb_clip,
    mocap_take_to_native_animation,
    probe_mocap_backends,
    track_person_boxes,
)
from flint.platform.input import InputState
from flint.platform.os import file_size, make_dirs, read_bytes, write_bytes
from flint.render.font import SdfFont
from flint.render.render_world import make_rh_look_at, make_rh_reverse_z_perspective
from flint.studio.mocap_panel import (
    MocapStudioState,
    draw_mocap_panel,
    emit_mocap_viewport_overlay,
)
from flint.ui.animator import UI_ANIM_INSTANT
from flint.ui.context import UiContext, hash_ui_id
from flint.ui.draw_list import UiDrawList
from flint.ui.flexbox_dock import RectF
from flint.ui.theme import StudioTheme


def _rm_rf_tmp(dir_path: String):
    if not dir_path.startswith("/tmp/flint-rgbmocap-"):
        return
    var cmd = "rm -rf " + dir_path
    _ = external_call["system", Int32](cmd.as_c_string_span())


def _write_dummy_file(path: String, text: String = "ONNX_DUMMY_WEIGHTS") raises:
    var sb = text.as_bytes()
    var out = List[UInt8](capacity=len(sb))
    for i in range(len(sb)):
        out.append(sb[i])
    assert_true(write_bytes(path, out))


# ---------------------------------------------------------------------------
# 1. Commercial-Safe Backend Discovery (`probe_mocap_backends`)
# ---------------------------------------------------------------------------


def test_probe_mocap_backends(tmp_root: String) raises:
    print("  [1/7] probe_mocap_backends (empty, partial & populated commercial-safe caches)...")
    var empty_cache = tmp_root + "/empty_cache"
    _ = make_dirs(empty_cache)

    var b_empty = probe_mocap_backends(empty_cache)
    assert_equal(len(b_empty), 2)
    assert_equal(b_empty[0].id, MOCAP_BACKEND_GEMX)
    assert_equal(b_empty[1].id, MOCAP_BACKEND_RTMPOSE_MBERT)
    assert_true(b_empty[0].commercial_safe and b_empty[1].commercial_safe)
    assert_true(b_empty[0].license_summary.find("Apache-2.0") >= 0)
    assert_true(b_empty[1].license_summary.find("MIT") >= 0)
    assert_false(b_empty[0].weights_present)
    assert_false(b_empty[1].weights_present)
    assert_true(b_empty[0].status_detail.find("Cache empty") >= 0)
    assert_true(b_empty[1].status_detail.find("Cache empty") >= 0)

    # Partial GEM-X weights (1/3 files present) -> still not ready, reports Partial
    var partial_cache = tmp_root + "/partial_cache"
    _ = make_dirs(partial_cache + "/gemx")
    _write_dummy_file(partial_cache + "/gemx/gemx_vitpose.onnx")
    var b_part = probe_mocap_backends(partial_cache)
    assert_false(b_part[0].weights_present)
    assert_true(b_part[0].status_detail.find("Partial weights (1/3)") >= 0)
    assert_false(b_part[1].weights_present)

    # Populated cache for both commercial-safe backends -> both report Weights Ready
    var full_cache = tmp_root + "/full_cache"
    _ = make_dirs(full_cache + "/gemx")
    _ = make_dirs(full_cache + "/rtmpose_mbert")
    _write_dummy_file(full_cache + "/gemx/gemx_vitpose.onnx")
    _write_dummy_file(full_cache + "/gemx/gemx_pose.onnx")
    _write_dummy_file(full_cache + "/gemx/gemx_motion.onnx")
    _write_dummy_file(full_cache + "/rtmpose_mbert/rtmpose_body.onnx")
    _write_dummy_file(full_cache + "/rtmpose_mbert/motionbert_dstformer.onnx")

    var b_full = probe_mocap_backends(full_cache)
    assert_true(b_full[0].weights_present)
    assert_true(b_full[1].weights_present)
    assert_true(b_full[0].status_detail.find("Weights Ready") >= 0)
    assert_true(b_full[1].status_detail.find("Weights Ready") >= 0)

    # Missing directory never throws
    var b_missing = probe_mocap_backends(tmp_root + "/does_not_exist_12345")
    assert_equal(len(b_missing), 2)
    assert_false(b_missing[0].weights_present)
    assert_false(b_missing[1].weights_present)


# ---------------------------------------------------------------------------
# 2. Multi-Person IoU Tracking (`bbox_iou`, `track_person_boxes`)
# ---------------------------------------------------------------------------


def test_track_person_boxes() raises:
    print("  [2/7] track_person_boxes (IoU, occlusion bridging, primary selection & gap split)...")
    var box_a = MocapBBox(100.0, 50.0, 200.0, 250.0, 0.95)
    var box_same = MocapBBox(100.0, 50.0, 200.0, 250.0, 0.95)
    var box_far = MocapBBox(500.0, 50.0, 600.0, 250.0, 0.85)
    var box_half = MocapBBox(150.0, 50.0, 250.0, 250.0, 0.90)  # 50x200 inter / 150x200 union = 1/3
    assert_true(abs(bbox_iou(box_a, box_same) - Float32(1.0)) < Float32(1.0e-5))
    assert_equal(bbox_iou(box_a, box_far), Float32(0.0))
    assert_true(abs(bbox_iou(box_a, box_half) - Float32(1.0 / 3.0)) < Float32(1.0e-4))

    # Build 20-frame multi-person sequence:
    # - Person A (high conf 0.94): visible on frames 0..19 EXCEPT occluded on frames 8..10 (3-frame gap <= 5)
    # - Person B (lower conf 0.78): visible on frames 0..14 (shorter span)
    # - Person C (passerby 0.82): visible on frames 2..4, vanishes for 9 frames (> max_missed=5),
    #   reappears at same spot on frames 14..16 -> must split into a new track ID!
    var dets = List[List[MocapBBox]](capacity=20)
    for f in range(20):
        var frame_dets = List[MocapBBox]()
        if f < 8 or f > 10:
            var ax = Float32(220.0) + Float32(f) * Float32(4.0)
            frame_dets.append(MocapBBox(ax, 80.0, ax + 110.0, 380.0, 0.94))
        if f < 15:
            var bx = Float32(580.0) - Float32(f) * Float32(3.0)
            frame_dets.append(MocapBBox(bx, 110.0, bx + 90.0, 360.0, 0.78))
        if (f >= 2 and f <= 4) or (f >= 14 and f <= 16):
            frame_dets.append(MocapBBox(900.0, 120.0, 980.0, 340.0, 0.82))
        dets.append(frame_dets^)

    var tracks = track_person_boxes(dets, iou_threshold=0.30, max_missed_frames=5)
    assert_equal(len(tracks), 4)

    # Track 1 (Person A): 17 frames bridged cleanly across 3-frame occlusion, selected as primary active
    assert_equal(tracks[0].track_id, 1)
    assert_equal(tracks[0].length(), 17)
    assert_true(tracks[0].active)
    assert_true(abs(tracks[0].mean_confidence - Float32(0.94)) < Float32(1.0e-3))
    assert_equal(tracks[0].label(), "Person #1")

    # Track 2 (Person B): 15 frames, inactive
    assert_equal(tracks[1].track_id, 2)
    assert_equal(tracks[1].length(), 15)
    assert_false(tracks[1].active)

    # Tracks 3 & 4 (Person C split across 9-frame gap > max_missed_frames=5): 3 frames each
    assert_equal(tracks[2].length(), 3)
    assert_equal(tracks[3].length(), 3)
    assert_false(tracks[2].active)
    assert_false(tracks[3].active)


# ---------------------------------------------------------------------------
# 3. Demo Take, BVH Export & NativeAnimation Conversion
# ---------------------------------------------------------------------------


def test_demo_take_bvh_and_native_animation(tmp_root: String) raises:
    print("  [3/7] generate_demo_mocap_take + export_mocap_take_bvh + mocap_take_to_native_animation...")
    var take = generate_demo_mocap_take(90, Float32(30.0))
    assert_equal(take.num_frames(), 90)
    assert_equal(take.num_joints(), 77)
    assert_equal(take.backend_id, MOCAP_BACKEND_GEMX)
    assert_true(abs(take.duration_sec() - Float32(89.0 / 30.0)) < Float32(1.0e-4))
    assert_equal(len(take.tracks), 2)
    assert_true(take.tracks[0].active and not take.tracks[1].active)
    assert_true(take.foot_slide_before_cm > Float32(1.5))

    var min_conf = Float32(1.0)
    var max_conf = Float32(0.0)
    for f in range(take.num_frames()):
        ref fr = take.frames[f]
        assert_equal(len(fr.joint_positions), 77)
        assert_equal(len(fr.joint_rotations), 77)
        assert_equal(len(fr.joint_confidences), 77)
        assert_true(fr.mean_confidence >= Float32(0.0) and fr.mean_confidence <= Float32(1.0))
        if fr.mean_confidence < min_conf:
            min_conf = fr.mean_confidence
        if fr.mean_confidence > max_conf:
            max_conf = fr.mean_confidence
        assert_true(isfinite(fr.root_pos.x) and isfinite(fr.root_pos.y) and isfinite(fr.root_pos.z))
        assert_true(isfinite(fr.camera.pos.x) and isfinite(fr.camera.pos.y) and isfinite(fr.camera.pos.z))
    # Verify realistic per-frame confidence variation & moving camera frustum
    assert_true(max_conf - min_conf > Float32(0.15))
    var cam_move = (take.frames[89].camera.pos - take.frames[0].camera.pos).length()
    assert_true(cam_move > Float32(0.05))

    # BVH export round-trip structure check
    var bvh_path = tmp_root + "/exports/demo_walk.bvh"
    export_mocap_take_bvh(take, bvh_path)
    var bvh_sz = file_size(bvh_path)
    assert_true(bvh_sz > 10000)
    var bvh_raw = read_bytes(bvh_path)
    var bvh_txt = String(unsafe_from_utf8=bvh_raw)
    assert_true(bvh_txt.startswith("HIERARCHY"))
    assert_true(bvh_txt.find("ROOT ") >= 0)
    assert_true(bvh_txt.find("MOTION") >= 0)
    assert_true(bvh_txt.find("Frames: 90") >= 0)
    assert_true(bvh_txt.find("Frame Time:") >= 0)

    # Count non-empty numeric frame rows after "Frame Time:"
    var ft_pos = bvh_txt.find("Frame Time:")
    var nl_pos = bvh_txt.find("\n", ft_pos)
    assert_true(nl_pos > ft_pos)
    var motion_lines = 0
    var cur = nl_pos + 1
    var nbytes = bvh_txt.byte_length()
    while cur < nbytes:
        var next_nl = bvh_txt.find("\n", cur)
        if next_nl < 0:
            next_nl = nbytes
        if next_nl > cur + 10:
            motion_lines += 1
        cur = next_nl + 1
    assert_equal(motion_lines, 90)

    # NativeAnimation conversion check
    var anim = mocap_take_to_native_animation(take)
    assert_true(anim.name.byte_length() > 0)
    assert_true(anim.duration > Float32(2.5))
    assert_true(len(anim.channels) >= 78)  # root translation + 77 joint rotations
    assert_equal(len(anim.channels[0].times), 90)


# ---------------------------------------------------------------------------
# 4. One Euro Filters (`OneEuroScalar`, `OneEuroVec3`, `OneEuroQuat`)
# ---------------------------------------------------------------------------


def test_one_euro_filters_and_resilience() raises:
    print("  [4/7] OneEuroScalar / OneEuroVec3 / OneEuroQuat (jitter, hemisphere flip & NaN/Inf)...")
    var nan_f = bitcast[DType.float32](UInt32(0x7FC00000))
    var pos_inf = bitcast[DType.float32](UInt32(0x7F800000))
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var dt = Float32(1.0 / 30.0)

    # A. OneEuroScalar jitter reduction + NaN/Inf/dt<=0 resilience + reset()
    var sc = OneEuroScalar(OneEuroConfig(min_cutoff=1.0, beta=0.02, d_cutoff=1.0))
    assert_equal(sc.filter(nan_f, dt), Float32(0.0))  # uninitialized NaN -> 0.0
    var s0 = sc.filter(Float32(1.5), dt)
    assert_true(abs(s0 - Float32(1.5)) < Float32(1.0e-5))
    assert_true(isfinite(sc.filter(nan_f, dt)))
    assert_true(isfinite(sc.filter(pos_inf, dt)))
    assert_true(isfinite(sc.filter(neg_inf, dt)))
    assert_true(isfinite(sc.filter(Float32(2.0), Float32(0.0))))
    assert_true(isfinite(sc.filter(Float32(2.0), Float32(-0.1))))
    sc.reset()
    assert_false(sc.initialized)

    # B. OneEuroVec3 series jitter reduction
    var noisy_v3 = List[Vec3](capacity=60)
    for i in range(60):
        var t = Float32(i) * dt
        var base = Vec3(t * Float32(0.5), Float32(1.0) + Float32(0.1) * sin(t * Float32(2.0)), Float32(0.0))
        var sign = Float32(1.0) if (i % 2 == 0) else Float32(-1.0)
        var jitter = Vec3(sign * Float32(0.025), -sign * Float32(0.020), sign * Float32(0.018))
        noisy_v3.append(base + jitter)
    var jit_before = measure_series_jitter(noisy_v3)
    var smooth_v3 = filter_vec3_series(noisy_v3, Float32(30.0), OneEuroConfig(1.0, 0.02, 1.0))
    var jit_after = measure_series_jitter(smooth_v3)
    assert_equal(len(smooth_v3), 60)
    assert_true(jit_after < jit_before * Float32(0.45))

    var v3_flt = OneEuroVec3()
    _ = v3_flt.filter(Vec3(1.0, 2.0, 3.0), dt)
    var v3_nan = v3_flt.filter(Vec3(nan_f, pos_inf, neg_inf), dt)
    assert_true(isfinite(v3_nan.x) and isfinite(v3_nan.y) and isfinite(v3_nan.z))

    # C. OneEuroQuat double-cover hemisphere flip safety (-q == q) + NaN safety
    var q_series = List[Quat](capacity=40)
    for i in range(40):
        var ang = Float32(i) * Float32(0.02)
        var q = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), ang).normalized()
        # Flip sign every other frame to stress hemisphere continuity!
        if i % 2 == 1:
            q = Quat(-q.x, -q.y, -q.z, -q.w)
        q_series.append(q)
    var q_out = filter_quat_series(q_series, Float32(30.0), OneEuroConfig(1.2, 0.05, 1.0))
    assert_equal(len(q_out), 40)
    for i in range(1, 40):
        var qi = q_out[i]
        var qp = q_out[i - 1]
        var norm = (qi.x * qi.x + qi.y * qi.y + qi.z * qi.z + qi.w * qi.w)
        assert_true(abs(norm - Float32(1.0)) < Float32(1.0e-3))
        # Consecutive outputs must stay in same hemisphere (dot > 0.99, zero 180-deg flip!)
        assert_true(abs(qi.dot(qp)) > Float32(0.99))

    var q_flt = OneEuroQuat()
    _ = q_flt.filter(Quat.identity(), dt)
    var q_bad = q_flt.filter(Quat(nan_f, pos_inf, 0.0, 1.0), dt)
    assert_true(isfinite(q_bad.x) and isfinite(q_bad.y) and isfinite(q_bad.z) and isfinite(q_bad.w))
    q_flt.reset()
    assert_false(q_flt.initialized)


# ---------------------------------------------------------------------------
# 5. Ground Plane, Contact Detection, 2-Bone IK & Full Cleanup Pipeline
# ---------------------------------------------------------------------------


def test_ground_contacts_ik_and_full_cleanup(tmp_root: String) raises:
    print("  [5/7] Ground plane, contact locking, exact 2-bone IK & full MocapTake cleanup...")

    # A. estimate_ground_plane_y rejects single-frame negative outlier dip
    var lf = List[Vec3](capacity=40)
    var rf = List[Vec3](capacity=40)
    for f in range(40):
        var ly = Float32(0.35) if f < 20 else Float32(0.55)
        var ry = Float32(0.55) if f < 20 else Float32(0.35)
        if f == 5:
            ly = Float32(-1.20)  # outlier spike
        lf.append(Vec3(0.0, ly, 0.0))
        rf.append(Vec3(0.2, ry, 0.0))
    var est_y = estimate_ground_plane_y(lf, rf)
    assert_true(abs(est_y - Float32(0.35)) < Float32(0.03))

    # B. detect_contact_intervals + gap bridging + lock_contact_positions
    var drift_foot = List[Vec3](capacity=30)
    for f in range(30):
        if f >= 5 and f <= 20:
            # Stance with 0.005 m/frame horizontal slide (~7.5 cm total) + 1-frame spike at f=12
            var sx = Float32(f - 5) * Float32(0.005)
            var sy = Float32(0.02) if f != 12 else Float32(0.14)
            drift_foot.append(Vec3(sx, sy, Float32(0.0)))
        else:
            drift_foot.append(Vec3(Float32(f) * Float32(0.03), Float32(0.25), Float32(0.0)))
    var ivs = detect_contact_intervals(
        drift_foot, Float32(30.0), Float32(0.0), Float32(0.08), Float32(0.25), 3, True, False
    )
    assert_equal(len(ivs), 1)
    assert_true(ivs[0].start_frame <= 6 and ivs[0].end_frame >= 19)
    var slide_pre = measure_foot_slide_cm(drift_foot, ivs)
    assert_true(slide_pre > Float32(5.0))
    var locked_foot = lock_contact_positions(drift_foot, ivs, Float32(0.0), Float32(0.95), 3, True)
    var slide_post = measure_foot_slide_cm(locked_foot, ivs)
    assert_true(slide_post < Float32(1.0))

    # C. solve_two_bone_leg_cleanup exact thigh & shin length preservation
    var hip = Vec3(0.0, 0.92, 0.0)
    var knee = Vec3(0.0, 0.50, 0.08)
    var ankle = Vec3(0.0, 0.08, 0.0)
    var orig_l1 = (knee - hip).length()
    var orig_l2 = (ankle - knee).length()

    # Reachable target
    var tgt_ok = Vec3(0.06, 0.10, 0.05)
    var ik_ok = solve_two_bone_leg_cleanup(hip, knee, ankle, tgt_ok)
    assert_true(abs((ik_ok[0] - hip).length() - orig_l1) < Float32(1.0e-4))
    assert_true(abs((ik_ok[1] - ik_ok[0]).length() - orig_l2) < Float32(1.0e-4))
    assert_true((ik_ok[1] - tgt_ok).length() < Float32(1.0e-3))

    # Over-extended unreachable target -> clamped cleanly, exact bone lengths preserved
    var tgt_far = Vec3(0.0, -2.5, 0.0)
    var ik_far = solve_two_bone_leg_cleanup(hip, knee, ankle, tgt_far)
    assert_true(abs((ik_far[0] - hip).length() - orig_l1) < Float32(1.0e-4))
    assert_true(abs((ik_far[1] - ik_far[0]).length() - orig_l2) < Float32(1.0e-4))

    # NaN target -> safe fallback
    var nan_f = bitcast[DType.float32](UInt32(0x7FC00000))
    var ik_nan = solve_two_bone_leg_cleanup(hip, knee, ankle, Vec3(nan_f, 0.0, 0.0))
    assert_true((ik_nan[0] - knee).length() < Float32(1.0e-5))
    assert_true((ik_nan[1] - ankle).length() < Float32(1.0e-5))

    # D. Full apply_mocap_cleanup_to_take on deterministic demo take
    var take_a = generate_demo_mocap_take(90, Float32(30.0))
    var take_b = generate_demo_mocap_take(90, Float32(30.0))
    var cfg = ContactCleanupConfig()
    var m_a = apply_mocap_cleanup_to_take(take_a, cfg)
    var m_b = apply_mocap_cleanup_to_take(take_b, cfg)

    assert_true(m_a.foot_contacts_detected >= 2)
    assert_true(m_a.foot_slide_before_cm > Float32(1.5))
    assert_true(m_a.foot_slide_after_cm < Float32(1.5))
    assert_true(m_a.foot_slide_after_cm < m_a.foot_slide_before_cm * Float32(0.25))
    assert_true(m_a.mean_jitter_after < m_a.mean_jitter_before)
    assert_equal(take_a.foot_slide_before_cm, m_a.foot_slide_before_cm)
    assert_equal(take_a.foot_slide_after_cm, m_a.foot_slide_after_cm)

    # Bit-for-bit lockstep determinism across two runs
    assert_equal(m_a.foot_slide_before_cm, m_b.foot_slide_before_cm)
    assert_equal(m_a.foot_slide_after_cm, m_b.foot_slide_after_cm)
    assert_equal(m_a.mean_jitter_after, m_b.mean_jitter_after)
    assert_equal(m_a.foot_contacts_detected, m_b.foot_contacts_detected)
    for f in range(90):
        var pa = take_a.frames[f].joint_positions[SOMA77_LEFT_FOOT]
        var pb = take_b.frames[f].joint_positions[SOMA77_LEFT_FOOT]
        assert_equal(pa.x, pb.x)
        assert_equal(pa.y, pb.y)
        assert_equal(pa.z, pb.z)
        var ra = take_a.frames[f].joint_positions[SOMA77_RIGHT_FOOT]
        var rb = take_b.frames[f].joint_positions[SOMA77_RIGHT_FOOT]
        assert_equal(ra.x, rb.x)
        assert_equal(ra.y, rb.y)
        assert_equal(ra.z, rb.z)

    # Cleaned take exports cleanly to BVH and NativeAnimation
    var clean_bvh = tmp_root + "/exports/demo_clean.bvh"
    export_mocap_take_bvh(take_a, clean_bvh)
    assert_true(file_size(clean_bvh) > 10000)
    var clean_anim = mocap_take_to_native_animation(take_a, name="DemoClean")
    assert_equal(clean_anim.name, "DemoClean")
    assert_true(len(clean_anim.channels) >= 78)


# ---------------------------------------------------------------------------
# 6. Headless MoCap Studio Panel & 3D Viewport Overlay (`mocap_panel.mojo`)
# ---------------------------------------------------------------------------


def test_mocap_panel_and_viewport_overlay(tmp_root: String) raises:
    print("  [6/7] draw_mocap_panel + emit_mocap_viewport_overlay (UI_ANIM_INSTANT & 0-prim guard)...")
    var state = MocapStudioState()
    state.init_default(tmp_root + "/empty_ui_cache")
    assert_true(state.has_take)
    assert_equal(len(state.backends), 2)
    assert_true(state.metrics.foot_slide_after_cm < Float32(1.5))
    assert_true(state.metrics.foot_slide_after_cm < state.metrics.foot_slide_before_cm * Float32(0.25))

    var ui = UiContext()
    ui.dock_style_v2 = True
    ui.set_anim_mode(UI_ANIM_INSTANT)
    var theme = StudioTheme.flint_obsidian()
    ui.set_studio_theme(theme.copy())
    var inp = InputState()
    var panel_rect = RectF(940.0, 32.0, 340.0, 688.0)

    var prim_counts = List[Int]()
    for _ in range(2):
        inp.begin_frame()
        inp.move_mouse(10.0, 10.0)
        ui.begin_frame(inp, 1280.0, 720.0)
        _ = draw_mocap_panel(ui, state, panel_rect, theme)
        ui.end_frame()
        prim_counts.append(ui.draw_list.count())
        for pi in range(ui.draw_list.count()):
            ref pr = ui.draw_list.prims[pi]
            assert_true(isfinite(pr.x0) and isfinite(pr.y0) and isfinite(pr.x1) and isfinite(pr.y1))

    assert_true(prim_counts[0] > 100)
    assert_equal(prim_counts[0], prim_counts[1])

    # Action button click hits: [Apply to Entity] and [BVH] in sticky footer
    var foot_id = hash_ui_id(UInt64(0xCBF29CE484222325), "mocap_export_footer")
    var apply_r = ui.cached_rects[hash_ui_id(foot_id, "mocap_btn_apply")]
    var bvh_r = ui.cached_rects[hash_ui_id(foot_id, "mocap_btn_bvh")]
    var saw_apply = False
    for down in [True, False]:
        inp.begin_frame()
        inp.move_mouse(apply_r.center_x(), apply_r.center_y())
        inp.set_mouse(0, down)
        ui.begin_frame(inp, 1280.0, 720.0)
        var r_app = draw_mocap_panel(ui, state, panel_rect, theme)
        if r_app.apply_to_scene_clicked:
            saw_apply = True
        ui.end_frame()
    assert_true(saw_apply)

    var saw_bvh = False
    for down in [True, False]:
        inp.begin_frame()
        inp.move_mouse(bvh_r.center_x(), bvh_r.center_y())
        inp.set_mouse(0, down)
        ui.begin_frame(inp, 1280.0, 720.0)
        var r_bvh = draw_mocap_panel(ui, state, panel_rect, theme)
        if r_bvh.export_bvh_clicked:
            saw_bvh = True
        ui.end_frame()
    assert_true(saw_bvh)

    # Playback stepping advances frame only while playing
    var f0 = state.current_frame
    state.playing = False
    state.step_playback(Float32(0.2))
    assert_equal(state.current_frame, f0)
    state.playing = True
    state.step_playback(Float32(0.2))
    assert_true(state.current_frame != f0)
    state.playing = False
    state.current_frame = f0

    # 3D viewport overlay: skeleton + camera frustum + ground contact rings
    var vp = RectF(200.0, 32.0, 740.0, 500.0)
    var view_proj = make_rh_reverse_z_perspective(
        Float32(55.0 * 3.14159265 / 180.0), Float32(740.0 / 500.0), Float32(0.1), Float32(200.0)
    ).multiply(
        make_rh_look_at(Vec3(0.0, 2.2, 4.5), Vec3(0.0, 0.9, 0.0), Vec3(0.0, 1.0, 0.0))
    )
    var dl_on = UiDrawList()
    var font = SdfFont()
    emit_mocap_viewport_overlay(dl_on, font, 0, state, vp, view_proj, theme)
    assert_true(dl_on.count() > 50)
    for pi in range(dl_on.count()):
        ref pr = dl_on.prims[pi]
        assert_true(isfinite(pr.x0) and isfinite(pr.y0) and isfinite(pr.x1) and isfinite(pr.y1))

    # Strict 0-prim guard when all 3 overlay toggles are disabled
    state.show_frustum_overlay = False
    state.show_contact_rings = False
    state.show_skeleton_overlay = False
    var dl_off = UiDrawList()
    emit_mocap_viewport_overlay(dl_off, font, 0, state, vp, view_proj, theme)
    assert_equal(dl_off.count(), 0)


# ---------------------------------------------------------------------------
# 7. SOMA-77 -> Mixamo Humanoid Retargeting Regression (`soma_retarget.mojo`)
# ---------------------------------------------------------------------------


def _add_bone(
    mut model: NativeModel,
    mut skin: NativeSkin,
    parent: Int,
    name: String,
    offset: Vec3,
) -> Int:
    var idx = len(model.nodes)
    var node = NativeNode()
    node.name = "mixamorig:" + name
    node.base_translation = offset
    node.translation = offset
    node.base_rotation = Quat.identity()
    node.rotation = Quat.identity()
    node.base_scale = Vec3(1.0, 1.0, 1.0)
    node.scale = Vec3(1.0, 1.0, 1.0)
    node.has_matrix = False
    model.nodes.append(node^)
    if parent >= 0:
        model.nodes[parent].children.append(idx)
    skin.joints.append(idx)
    return idx


def _make_synthetic_mixamo_humanoid() -> NativeModel:
    """Build a self-contained Y-up, unit-scale Mixamo humanoid T-pose rig (zero external files)."""
    var model = NativeModel()
    var skin = NativeSkin()

    var hips = _add_bone(model, skin, -1, "Hips", Vec3(0.0, 0.95, 0.0))
    var spine = _add_bone(model, skin, hips, "Spine", Vec3(0.0, 0.12, 0.0))
    var spine1 = _add_bone(model, skin, spine, "Spine1", Vec3(0.0, 0.14, 0.0))
    var spine2 = _add_bone(model, skin, spine1, "Spine2", Vec3(0.0, 0.15, 0.0))
    var neck = _add_bone(model, skin, spine2, "Neck", Vec3(0.0, 0.12, 0.0))
    _ = _add_bone(model, skin, neck, "Head", Vec3(0.0, 0.16, 0.0))

    # Left & Right arms (T-pose along +X / -X)
    var l_sh = _add_bone(model, skin, spine2, "LeftShoulder", Vec3(0.06, 0.08, 0.0))
    var l_arm = _add_bone(model, skin, l_sh, "LeftArm", Vec3(0.12, 0.0, 0.0))
    var l_fa = _add_bone(model, skin, l_arm, "LeftForeArm", Vec3(0.28, 0.0, 0.0))
    _ = _add_bone(model, skin, l_fa, "LeftHand", Vec3(0.25, 0.0, 0.0))

    var r_sh = _add_bone(model, skin, spine2, "RightShoulder", Vec3(-0.06, 0.08, 0.0))
    var r_arm = _add_bone(model, skin, r_sh, "RightArm", Vec3(-0.12, 0.0, 0.0))
    var r_fa = _add_bone(model, skin, r_arm, "RightForeArm", Vec3(-0.28, 0.0, 0.0))
    _ = _add_bone(model, skin, r_fa, "RightHand", Vec3(-0.25, 0.0, 0.0))

    # Left & Right legs (Y-down from Hips to ground)
    var l_up = _add_bone(model, skin, hips, "LeftUpLeg", Vec3(0.10, -0.04, 0.0))
    var l_leg = _add_bone(model, skin, l_up, "LeftLeg", Vec3(0.0, -0.43, 0.0))
    var l_ft = _add_bone(model, skin, l_leg, "LeftFoot", Vec3(0.0, -0.42, 0.0))
    _ = _add_bone(model, skin, l_ft, "LeftToeBase", Vec3(0.0, -0.06, 0.14))

    var r_up = _add_bone(model, skin, hips, "RightUpLeg", Vec3(-0.10, -0.04, 0.0))
    var r_leg = _add_bone(model, skin, r_up, "RightLeg", Vec3(0.0, -0.43, 0.0))
    var r_ft = _add_bone(model, skin, r_leg, "RightFoot", Vec3(0.0, -0.42, 0.0))
    _ = _add_bone(model, skin, r_ft, "RightToeBase", Vec3(0.0, -0.06, 0.14))

    model.skins.append(skin^)
    return model^


def test_soma77_retarget_regression() raises:
    print("  [7/7] retarget_soma on 77-joint SOMA-77 clip (synthetic Mixamo humanoid regression)...")
    var take = generate_demo_mocap_take(12, Float32(30.0))
    var clip = mocap_take_to_mb_clip(take)
    assert_equal(clip.skeleton.num_joints(), 77)
    assert_equal(clip.num_frames(), 12)

    var model = _make_synthetic_mixamo_humanoid()
    var anim = retarget_soma(clip, model)

    # 22 mapped humanoid rotation channels + 1 Hips translation channel = 23
    assert_equal(len(anim.channels), 23)
    assert_true(anim.duration > Float32(0.3))

    var saw_hip_trans = False
    for ci in range(len(anim.channels)):
        ref ch = anim.channels[ci]
        assert_equal(len(ch.times), 12)
        if ch.path == 0:
            saw_hip_trans = True
            assert_equal(len(ch.values), 36)
            for f in range(12):
                var tx = ch.values[3 * f]
                var ty = ch.values[3 * f + 1]
                var tz = ch.values[3 * f + 2]
                assert_true(isfinite(tx) and isfinite(ty) and isfinite(tz))
                # Standing humanoid hips should stay at reasonable Y height (0.3..1.6 m)
                assert_true(ty > Float32(0.3) and ty < Float32(1.6))
        else:
            assert_equal(ch.path, 1)
            assert_equal(len(ch.values), 48)
            for f in range(12):
                var qx = ch.values[4 * f]
                var qy = ch.values[4 * f + 1]
                var qz = ch.values[4 * f + 2]
                var qw = ch.values[4 * f + 3]
                assert_true(isfinite(qx) and isfinite(qy) and isfinite(qz) and isfinite(qw))
                var n2 = qx * qx + qy * qy + qz * qz + qw * qw
                assert_true(abs(n2 - Float32(1.0)) < Float32(1.0e-3))
    assert_true(saw_hip_trans)

    # Negative regression: 1-frame SOMA-77 clip must raise Error
    var bad_clip = clip.copy()
    while len(bad_clip.frames) > 1:
        _ = bad_clip.frames.pop()
    var caught_short = False
    try:
        _ = retarget_soma(bad_clip, model)
    except:
        caught_short = True
    assert_true(caught_short)


def main() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp_root = "/tmp/flint-rgbmocap-" + String(pid)
    _rm_rf_tmp(tmp_root)
    _ = make_dirs(tmp_root)
    try:
        print("============================================================")
        print(" FLINT v5 RGBMOCAP PIPELINE & MOTION CLEANUP SUITE")
        print("============================================================")
        test_probe_mocap_backends(tmp_root)
        test_track_person_boxes()
        test_demo_take_bvh_and_native_animation(tmp_root)
        test_one_euro_filters_and_resilience()
        test_ground_contacts_ik_and_full_cleanup(tmp_root)
        test_mocap_panel_and_viewport_overlay(tmp_root)
        test_soma77_retarget_regression()
        print("[PASS] All 7 Flint v5 RGBMocap verification suites PASSED!")
    finally:
        _rm_rf_tmp(tmp_root)
