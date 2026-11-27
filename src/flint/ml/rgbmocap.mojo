"""Commercial-safe, backend-agnostic RGBMocap pipeline in pure Mojo (`flint.ml.rgbmocap`).

Inspired by `toyxyz_RGBMocap` (MIT), this module provides a unified monocular video-to-3D-motion
pipeline with zero C/C++ dependencies and commercial-safe model backends:

  - `MOCAP_BACKEND_GEMX` (0): NVIDIA GEM-X / SOMA-X (Apache-2.0 code, NVIDIA Open Model weights;
    gravity-view world rollout on the 77-joint SOMA-X humanoid skeleton).
  - `MOCAP_BACKEND_RTMPOSE_MBERT` (1): RTMPose + MotionBERT (Apache-2.0 / MIT; fixed-camera 2D->3D).

All model weights live outside the repository (probed non-destructively via
`probe_mocap_backends`). `generate_demo_mocap_take` synthesizes a deterministic 77-joint SOMA-X
walk + turn + wave take with realistic micro-jitter, ~6.5 cm stance foot slide, two tracked
subjects, per-frame confidence dips, and a moving camera frustum so Flint Studio's MoCap panel,
motion cleanup filters, and BVH/NativeAnimation exporters work immediately with zero checkpoints.
"""

from std.math import sqrt, abs, min, max
from flint.core.math import Vec3, Quat
from flint.core.detmath import det_sin, det_cos
from flint.platform.os import env_value, file_size, is_directory, make_dirs, parent_directory
from flint.assets.model import NativeAnimation
from flint.ml.motionbricks import MbSkeleton, MbClip, MbXform, MbVec3, MbQuat, world_pose
from flint.ml.gemx import (
    GxFrame,
    Gemx,
    GemxMotion,
    GX_JOINTS,
    soma77_skeleton,
    soma78_parents,
    soma78_t_pose_orient,
    gemx_clip,
    write_bvh,
    clip_to_native_animation,
)

# ---------------------------------------------------------------- backend ids & SOMA-77 indices

comptime MOCAP_BACKEND_GEMX: Int = 0
comptime MOCAP_BACKEND_RTMPOSE_MBERT: Int = 1

comptime SOMA77_HIPS: Int = 0
comptime SOMA77_SPINE1: Int = 1
comptime SOMA77_SPINE2: Int = 2
comptime SOMA77_CHEST: Int = 3
comptime SOMA77_NECK1: Int = 4
comptime SOMA77_NECK2: Int = 5
comptime SOMA77_HEAD: Int = 6
comptime SOMA77_LEFT_SHOULDER: Int = 11
comptime SOMA77_LEFT_ARM: Int = 12
comptime SOMA77_LEFT_FOREARM: Int = 13
comptime SOMA77_LEFT_HAND: Int = 14
comptime SOMA77_RIGHT_SHOULDER: Int = 39
comptime SOMA77_RIGHT_ARM: Int = 40
comptime SOMA77_RIGHT_FOREARM: Int = 41
comptime SOMA77_RIGHT_HAND: Int = 42
comptime SOMA77_LEFT_LEG: Int = 67
comptime SOMA77_LEFT_SHIN: Int = 68
comptime SOMA77_LEFT_FOOT: Int = 69
comptime SOMA77_LEFT_TOE_BASE: Int = 70
comptime SOMA77_RIGHT_LEG: Int = 72
comptime SOMA77_RIGHT_SHIN: Int = 73
comptime SOMA77_RIGHT_FOOT: Int = 74
comptime SOMA77_RIGHT_TOE_BASE: Int = 75


# ---------------------------------------------------------------- backend discovery


@fieldwise_init
struct MocapBackendInfo(Copyable, Movable):
    """Metadata and local weight availability for a commercial-safe RGBMocap backend."""
    var id: Int
    var name: String
    var license_summary: String
    var commercial_safe: Bool
    var cache_subdir: String
    var required_files: List[String]
    var weights_present: Bool
    var status_detail: String

    def __init__(out self):
        self.id = MOCAP_BACKEND_GEMX
        self.name = String()
        self.license_summary = String()
        self.commercial_safe = True
        self.cache_subdir = String()
        self.required_files = List[String]()
        self.weights_present = False
        self.status_detail = String()


def _join_path(base: String, sub: String) -> String:
    if base.byte_length() == 0:
        return sub
    if sub.byte_length() == 0:
        return base
    if base.endswith("/"):
        return base + sub
    return base + "/" + sub


def _count_present_files(dir_path: String, files: List[String]) -> Int:
    if dir_path.byte_length() == 0 or not is_directory(dir_path, follow_symlinks=True):
        return 0
    var present = 0
    for i in range(len(files)):
        if file_size(_join_path(dir_path, files[i])) > 0:
            present += 1
    return present


def _probe_candidate_dirs(
    candidates: List[String],
    files: List[String],
    default_hint: String,
    export_hint: String,
) -> Tuple[Bool, String]:
    var total = len(files)
    var best_present = 0
    var best_dir = default_hint
    for i in range(len(candidates)):
        var d = candidates[i]
        if d.byte_length() == 0:
            continue
        var present = _count_present_files(d, files)
        if present == total and total > 0:
            return (True, "Weights Ready (" + d + ")")
        if present > best_present:
            best_present = present
            best_dir = d
    if best_present > 0:
        return (
            False,
            "Partial weights ("
            + String(best_present)
            + "/"
            + String(total)
            + ") in "
            + best_dir
            + " — "
            + export_hint,
        )
    return (
        False,
        "Cache empty (" + best_dir + ") — " + export_hint,
    )


def probe_mocap_backends(cache_root: String = "") -> List[MocapBackendInfo]:
    """Probe local model caches for commercial-safe RGBMocap backends without throwing.

    Search order when `cache_root` is empty:
      1. `$FLINT_MODEL_CACHE/<subdir>`
      2. `$HOME/.cache/flint/models/<subdir>`
      3. `$HOME/.cache/gemx` and `$GEMX_DIR` (for GEM-X)
    """
    var env_cache = env_value("FLINT_MODEL_CACHE")
    var home = env_value("HOME")
    var gemx_dir_env = env_value("GEMX_DIR")
    var default_root = String("~/.cache/flint/models")
    if cache_root.byte_length() > 0:
        default_root = cache_root
    elif env_cache.byte_length() > 0:
        default_root = env_cache
    elif home.byte_length() > 0:
        default_root = _join_path(home, ".cache/flint/models")

    # Backend 0: GEM-X / SOMA-X
    var gemx_files: List[String] = [
        "gemx_vitpose.onnx",
        "gemx_pose.onnx",
        "gemx_motion.onnx",
    ]
    var gemx_dirs = List[String]()
    if cache_root.byte_length() > 0:
        gemx_dirs.append(_join_path(cache_root, "gemx"))
        gemx_dirs.append(cache_root)
    else:
        if env_cache.byte_length() > 0:
            gemx_dirs.append(_join_path(env_cache, "gemx"))
        if home.byte_length() > 0:
            gemx_dirs.append(_join_path(home, ".cache/flint/models/gemx"))
            gemx_dirs.append(_join_path(home, ".cache/gemx"))
        if gemx_dir_env.byte_length() > 0:
            gemx_dirs.append(gemx_dir_env)
    var gemx_default = _join_path(default_root, "gemx")
    var gemx_probe = _probe_candidate_dirs(
        gemx_dirs,
        gemx_files,
        gemx_default,
        "export via tools/gemx/export.py or preview Demo",
    )
    var gemx_info = MocapBackendInfo(
        MOCAP_BACKEND_GEMX,
        "GEM-X / SOMA-X",
        "Apache-2.0 / NVIDIA Open Model",
        True,
        "gemx",
        gemx_files^,
        gemx_probe[0],
        gemx_probe[1],
    )

    # Backend 1: RTMPose + MotionBERT
    var rtm_files: List[String] = [
        "rtmpose_body.onnx",
        "motionbert_dstformer.onnx",
    ]
    var rtm_dirs = List[String]()
    if cache_root.byte_length() > 0:
        rtm_dirs.append(_join_path(cache_root, "rtmpose_mbert"))
        rtm_dirs.append(cache_root)
    else:
        if env_cache.byte_length() > 0:
            rtm_dirs.append(_join_path(env_cache, "rtmpose_mbert"))
        if home.byte_length() > 0:
            rtm_dirs.append(_join_path(home, ".cache/flint/models/rtmpose_mbert"))
    var rtm_default = _join_path(default_root, "rtmpose_mbert")
    var rtm_probe = _probe_candidate_dirs(
        rtm_dirs,
        rtm_files,
        rtm_default,
        "place ONNX weights in cache or preview Demo",
    )
    var rtm_info = MocapBackendInfo(
        MOCAP_BACKEND_RTMPOSE_MBERT,
        "RTMPose + MotionBERT",
        "Apache-2.0 / MIT",
        True,
        "rtmpose_mbert",
        rtm_files^,
        rtm_probe[0],
        rtm_probe[1],
    )

    var out = List[MocapBackendInfo](capacity=2)
    out.append(gemx_info^)
    out.append(rtm_info^)
    return out^


# ---------------------------------------------------------------- multi-person tracking


@fieldwise_init
struct MocapBBox(Copyable, ImplicitlyCopyable, Movable):
    """Axis-aligned 2D person bounding box in image pixel coordinates."""
    var x0: Float32
    var y0: Float32
    var x1: Float32
    var y1: Float32
    var confidence: Float32

    def __init__(out self):
        self.x0 = 0.0
        self.y0 = 0.0
        self.x1 = 0.0
        self.y1 = 0.0
        self.confidence = 0.0

    def width(self) -> Float32:
        return max(Float32(0.0), self.x1 - self.x0)

    def height(self) -> Float32:
        return max(Float32(0.0), self.y1 - self.y0)

    def area(self) -> Float32:
        return self.width() * self.height()

    def center_x(self) -> Float32:
        return (self.x0 + self.x1) * Float32(0.5)

    def center_y(self) -> Float32:
        return (self.y0 + self.y1) * Float32(0.5)


@fieldwise_init
struct MocapTrack(Copyable, Movable):
    """A tracked person identity across video frames."""
    var track_id: Int
    var frames: List[Int]
    var boxes: List[MocapBBox]
    var mean_confidence: Float32
    var active: Bool

    def __init__(out self, track_id: Int = 1):
        self.track_id = track_id
        self.frames = List[Int]()
        self.boxes = List[MocapBBox]()
        self.mean_confidence = 0.0
        self.active = False

    def length(self) -> Int:
        return len(self.frames)

    def label(self) -> String:
        return "Person #" + String(self.track_id)


def bbox_iou(a: MocapBBox, b: MocapBBox) -> Float32:
    """Intersection-over-Union of two axis-aligned boxes in [0, 1]."""
    var ix0 = max(a.x0, b.x0)
    var iy0 = max(a.y0, b.y0)
    var ix1 = min(a.x1, b.x1)
    var iy1 = min(a.y1, b.y1)
    var iw = max(Float32(0.0), ix1 - ix0)
    var ih = max(Float32(0.0), iy1 - iy0)
    var inter = iw * ih
    if inter <= Float32(0.0):
        return Float32(0.0)
    var denom = a.area() + b.area() - inter
    if denom <= Float32(1e-6):
        return Float32(0.0)
    return inter / denom


def track_person_boxes(
    detections_per_frame: List[List[MocapBBox]],
    iou_threshold: Float32 = 0.30,
    max_missed_frames: Int = 8,
) -> List[MocapTrack]:
    """Pure-Mojo greedy IoU multi-person tracker across frames.

    Links per-frame detections into persistent `MocapTrack` identities with occlusion tolerance
    (`max_missed_frames`). Marks the primary track (longest span, breaking ties by mean confidence)
    as `active = True`.
    """
    var tracks = List[MocapTrack]()
    var last_frame = List[Int]()
    var last_box = List[MocapBBox]()
    var next_id = 1

    for f in range(len(detections_per_frame)):
        var n_det = len(detections_per_frame[f])
        if n_det == 0:
            continue
        var det_matched = List[Bool](length=n_det, fill=False)
        var n_trk = len(tracks)
        var trk_matched = List[Bool](length=n_trk, fill=False)

        # Greedy global max-IoU assignment among recently active tracks
        while True:
            var best_iou = iou_threshold
            var best_t = -1
            var best_d = -1
            for t in range(n_trk):
                if trk_matched[t]:
                    continue
                if f - last_frame[t] > max_missed_frames:
                    continue
                for d in range(n_det):
                    if det_matched[d]:
                        continue
                    var iou = bbox_iou(last_box[t], detections_per_frame[f][d])
                    if iou > best_iou:
                        best_iou = iou
                        best_t = t
                        best_d = d
            if best_t < 0 or best_d < 0:
                break
            var box = detections_per_frame[f][best_d]
            tracks[best_t].frames.append(f)
            tracks[best_t].boxes.append(box)
            last_frame[best_t] = f
            last_box[best_t] = box
            trk_matched[best_t] = True
            det_matched[best_d] = True

        # Spawn new tracks for unmatched detections
        for d in range(n_det):
            if det_matched[d]:
                continue
            var box = detections_per_frame[f][d]
            var trk = MocapTrack(next_id)
            next_id += 1
            trk.frames.append(f)
            trk.boxes.append(box)
            tracks.append(trk^)
            last_frame.append(f)
            last_box.append(box)

    # Finalize mean confidence and pick primary active track
    var primary_idx = -1
    var primary_score = Float32(-1.0)
    for t in range(len(tracks)):
        var count = len(tracks[t].boxes)
        var conf_sum = Float32(0.0)
        for k in range(count):
            conf_sum += tracks[t].boxes[k].confidence
        tracks[t].mean_confidence = conf_sum / Float32(count) if count > 0 else Float32(0.0)
        tracks[t].active = False
        var score = Float32(count) * (Float32(0.5) + Float32(0.5) * tracks[t].mean_confidence)
        if score > primary_score:
            primary_score = score
            primary_idx = t
    if primary_idx >= 0:
        tracks[primary_idx].active = True
    return tracks^


# ---------------------------------------------------------------- take data model


@fieldwise_init
struct MocapCameraPose(Copyable, ImplicitlyCopyable, Movable):
    """Per-frame world-space camera pose and intrinsics for 3D frustum visualization."""
    var pos: Vec3
    var forward: Vec3
    var up: Vec3
    var fov_y_deg: Float32
    var aspect: Float32

    def __init__(out self):
        self.pos = Vec3(0.0, 1.45, 3.6)
        self.forward = Vec3(0.0, -0.08, -1.0).normalized()
        self.up = Vec3(0.0, 1.0, 0.0)
        self.fov_y_deg = 45.0
        self.aspect = Float32(16.0) / Float32(9.0)


@fieldwise_init
struct MocapFrame(Copyable, Movable):
    """One captured 3D skeleton frame (world positions + local joint rotations + contacts)."""
    var frame_idx: Int
    var timestamp_sec: Float32
    var root_pos: Vec3
    var joint_positions: List[Vec3]
    var joint_rotations: List[Quat]
    var joint_confidences: List[Float32]
    var mean_confidence: Float32
    var left_foot_contact: Bool
    var right_foot_contact: Bool
    var left_palm_contact: Bool
    var right_palm_contact: Bool
    var camera: MocapCameraPose

    def __init__(out self, frame_idx: Int = 0, timestamp_sec: Float32 = 0.0):
        self.frame_idx = frame_idx
        self.timestamp_sec = timestamp_sec
        self.root_pos = Vec3()
        self.joint_positions = List[Vec3]()
        self.joint_rotations = List[Quat]()
        self.joint_confidences = List[Float32]()
        self.mean_confidence = 1.0
        self.left_foot_contact = False
        self.right_foot_contact = False
        self.left_palm_contact = False
        self.right_palm_contact = False
        self.camera = MocapCameraPose()


@fieldwise_init
struct MocapTake(Copyable, Movable):
    """Complete monocular RGB motion capture take ready for cleanup, preview, and export."""
    var name: String
    var backend_id: Int
    var fps: Float32
    var source_path: String
    var joint_names: List[String]
    var parent_indices: List[Int]
    var frames: List[MocapFrame]
    var tracks: List[MocapTrack]
    var selected_track_id: Int
    var foot_slide_before_cm: Float32
    var foot_slide_after_cm: Float32

    def __init__(out self):
        self.name = String()
        self.backend_id = MOCAP_BACKEND_GEMX
        self.fps = 30.0
        self.source_path = String()
        self.joint_names = List[String]()
        self.parent_indices = List[Int]()
        self.frames = List[MocapFrame]()
        self.tracks = List[MocapTrack]()
        self.selected_track_id = 1
        self.foot_slide_before_cm = 0.0
        self.foot_slide_after_cm = 0.0

    def num_frames(self) -> Int:
        return len(self.frames)

    def num_joints(self) -> Int:
        return len(self.joint_names)

    def duration_sec(self) -> Float32:
        if len(self.frames) <= 1 or self.fps <= Float32(0.0):
            return Float32(0.0)
        return Float32(len(self.frames) - 1) / self.fps

    def find_joint(self, name: String) -> Int:
        for i in range(len(self.joint_names)):
            if self.joint_names[i] == name:
                return i
        return -1


# ---------------------------------------------------------------- conversion helpers


def _mb_to_vec3(v: MbVec3) -> Vec3:
    return Vec3(v.x, v.y, v.z)


def _vec3_to_mb(v: Vec3) -> MbVec3:
    return MbVec3(v.x, v.y, v.z)


def _mb_to_quat(q: MbQuat) -> Quat:
    return Quat(q.x, q.y, q.z, q.w)


def _quat_to_mb(q: Quat) -> MbQuat:
    return MbQuat(q.x, q.y, q.z, q.w)


def _estimate_contact_foot_slide_cm(frames: List[MocapFrame], left_idx: Int, right_idx: Int) -> Float32:
    """Measure average horizontal travel (cm) during flagged foot-contact spans."""
    var n = len(frames)
    if n < 2 or left_idx < 0 or right_idx < 0:
        return Float32(0.0)
    var total_slide_m = Float32(0.0)
    var spans = 0
    for side in range(2):
        var j_idx = left_idx if side == 0 else right_idx
        var f = 0
        while f < n:
            var in_contact = frames[f].left_foot_contact if side == 0 else frames[f].right_foot_contact
            if not in_contact or j_idx >= len(frames[f].joint_positions):
                f += 1
                continue
            var start = f
            var travel = Float32(0.0)
            while f + 1 < n:
                var next_c = frames[f + 1].left_foot_contact if side == 0 else frames[f + 1].right_foot_contact
                if not next_c or j_idx >= len(frames[f + 1].joint_positions):
                    break
                var p0 = frames[f].joint_positions[j_idx]
                var p1 = frames[f + 1].joint_positions[j_idx]
                var dx = p1.x - p0.x
                var dz = p1.z - p0.z
                travel += sqrt(dx * dx + dz * dz)
                f += 1
            if f - start >= 2:
                total_slide_m += travel
                spans += 1
            f += 1
    if spans == 0:
        return Float32(0.0)
    return (total_slide_m / Float32(spans)) * Float32(100.0)


def _gemx_camera_pose(m: GemxMotion, frame_idx: Int, width: Int, height: Int) -> MocapCameraPose:
    """Recover an approximate world-space camera pose from GEM-X world/in-camera translations."""
    var cam = MocapCameraPose()
    cam.aspect = Float32(width) / Float32(max(1, height))
    cam.fov_y_deg = 45.0
    var base = 3 * frame_idx
    if base + 2 < len(m.world_transl) and base + 2 < len(m.incam_transl):
        var wt = Vec3(m.world_transl[base], m.world_transl[base + 1] + Float32(0.95), m.world_transl[base + 2])
        var depth = max(Float32(1.5), abs(m.incam_transl[base + 2]))
        var lateral = -m.incam_transl[base]
        cam.pos = Vec3(wt.x + lateral, max(Float32(0.6), wt.y + Float32(0.45)), wt.z + depth)
        cam.forward = (wt - cam.pos).normalized()
        cam.up = Vec3(0.0, 1.0, 0.0)
    return cam


# ---------------------------------------------------------------- GEM-X bridge & export


def run_gemx_image_sequence(
    frames_rgb: List[GxFrame],
    width: Int,
    height: Int,
    fps: Float32,
    model_dir: String,
    track_box: MocapBBox = MocapBBox(),
    use_gpu: Bool = True,
) raises -> MocapTake:
    """Run real GEM-X ONNX inference over RGB frames and return a `MocapTake` on SOMA-77."""
    if len(frames_rgb) == 0:
        raise Error("rgbmocap: run_gemx_image_sequence received 0 frames")
    var actual_fps = fps if fps > Float32(0.0) else Float32(30.0)
    var gemx = Gemx.load(model_dir, gpu=use_gpu)
    var init_box = track_box
    if init_box.width() <= Float32(1.0) or init_box.height() <= Float32(1.0):
        init_box = MocapBBox(
            Float32(width) * Float32(0.22),
            Float32(height) * Float32(0.08),
            Float32(width) * Float32(0.78),
            Float32(height) * Float32(0.94),
            Float32(1.0),
        )
    var xyxy = gemx.track(frames_rgb, init_box.x0, init_box.y0, init_box.x1, init_box.y1)
    var motion = gemx.run(frames_rgb, xyxy)
    var skel = soma77_skeleton()
    var clip = gemx_clip(motion, skel, world=True, sample_rate=actual_fps)

    var take = MocapTake()
    take.name = "GEM-X Capture"
    take.backend_id = MOCAP_BACKEND_GEMX
    take.fps = actual_fps
    take.source_path = model_dir
    take.joint_names = skel.names.copy()
    take.parent_indices = skel.parents.copy()
    take.selected_track_id = 1

    # Record tracked boxes as Track #1
    var trk = MocapTrack(1)
    trk.active = True
    var n_frames = clip.num_frames()
    var nj = skel.num_joints()
    var static_cols = len(motion.static_conf) // n_frames if n_frames > 0 else 0
    var track_conf_sum = Float32(0.0)

    for f in range(n_frames):
        var world_xf = world_pose(skel, clip.frames[f], MbXform.identity())
        var mf = MocapFrame(f, Float32(f) / actual_fps)
        mf.root_pos = _mb_to_vec3(clip.frames[f][0].p)
        var conf_sum = Float32(0.0)
        for j in range(nj):
            mf.joint_positions.append(_mb_to_vec3(world_xf[j].p))
            mf.joint_rotations.append(_mb_to_quat(clip.frames[f][j].q))
            var kp_idx = (f * GX_JOINTS + j) * 3 + 2
            var c = motion.kp2d[kp_idx] if kp_idx < len(motion.kp2d) else Float32(0.9)
            c = min(Float32(1.0), max(Float32(0.0), c))
            mf.joint_confidences.append(c)
            conf_sum += c
        mf.mean_confidence = conf_sum / Float32(nj) if nj > 0 else Float32(0.0)
        track_conf_sum += mf.mean_confidence
        if static_cols >= 4:
            var s0 = f * static_cols
            mf.left_foot_contact = motion.static_conf[s0] > Float32(0.0) or motion.static_conf[s0 + 1] > Float32(0.0)
            mf.right_foot_contact = motion.static_conf[s0 + 2] > Float32(0.0) or motion.static_conf[s0 + 3] > Float32(0.0)
        if static_cols >= 6:
            var s0 = f * static_cols
            mf.left_palm_contact = motion.static_conf[s0 + 4] > Float32(0.0)
            mf.right_palm_contact = motion.static_conf[s0 + 5] > Float32(0.0)
        mf.camera = _gemx_camera_pose(motion, f, width, height)
        take.frames.append(mf^)

        if 4 * f + 3 < len(xyxy):
            trk.frames.append(f)
            trk.boxes.append(
                MocapBBox(
                    xyxy[4 * f],
                    xyxy[4 * f + 1],
                    xyxy[4 * f + 2],
                    xyxy[4 * f + 3],
                    take.frames[f].mean_confidence,
                )
            )

    trk.mean_confidence = track_conf_sum / Float32(n_frames) if n_frames > 0 else Float32(0.0)
    take.tracks.append(trk^)
    take.foot_slide_before_cm = _estimate_contact_foot_slide_cm(
        take.frames, SOMA77_LEFT_FOOT, SOMA77_RIGHT_FOOT
    )
    take.foot_slide_after_cm = take.foot_slide_before_cm
    return take^


def mocap_take_to_mb_clip(take: MocapTake) -> MbClip:
    """Convert a `MocapTake` into a SOMA `MbClip` suitable for BVH or `NativeAnimation` export."""
    var skel: MbSkeleton
    var nj = take.num_joints()
    if nj == GX_JOINTS:
        skel = soma77_skeleton()
    else:
        skel = MbSkeleton()
        skel.names = take.joint_names.copy()
        skel.parents = take.parent_indices.copy()
        for j in range(nj):
            var x = MbXform.identity()
            if len(take.frames) > 0 and j < len(take.frames[0].joint_positions):
                var p = take.frames[0].joint_positions[j]
                var par = take.parent_indices[j] if j < len(take.parent_indices) else -1
                if par >= 0 and par < len(take.frames[0].joint_positions):
                    var pp = take.frames[0].joint_positions[par]
                    x.p = MbVec3(p.x - pp.x, p.y - pp.y, p.z - pp.z)
                else:
                    x.p = _vec3_to_mb(p)
            skel.reference.append(x)

    var clip = MbClip()
    clip.skeleton = skel.copy()
    clip.sample_rate = take.fps if take.fps > Float32(0.0) else Float32(30.0)
    for f in range(len(take.frames)):
        var pose = List[MbXform](capacity=nj)
        for j in range(nj):
            var x = skel.reference[j]
            if j == 0:
                x.p = _vec3_to_mb(take.frames[f].root_pos)
            if j < len(take.frames[f].joint_rotations):
                x.q = _quat_to_mb(take.frames[f].joint_rotations[j]).normalized()
            pose.append(x)
        clip.frames.append(pose^)
    return clip^


def export_mocap_take_bvh(take: MocapTake, out_path: String) raises:
    """Export `take` as a centimetre ZYX-Euler BVH file readable by Blender, Maya, and Flint."""
    if take.num_frames() == 0 or take.num_joints() == 0:
        raise Error("rgbmocap: cannot export empty MocapTake to BVH")
    var parent = parent_directory(out_path)
    if parent.byte_length() > 0:
        _ = make_dirs(parent)
    var clip = mocap_take_to_mb_clip(take)
    write_bvh(clip, out_path)


def mocap_take_to_native_animation(
    take: MocapTake,
    node_of_joint: List[Int] = List[Int](),
    name: String = "",
) -> NativeAnimation:
    """Convert `take` into a `NativeAnimation` for Flint's runtime animation & Scene retargeting."""
    var clip = mocap_take_to_mb_clip(take)
    var mapping = List[Int]()
    if len(node_of_joint) > 0:
        mapping = node_of_joint.copy()
    else:
        for j in range(clip.skeleton.num_joints()):
            mapping.append(j)
    var anim_name = name if name.byte_length() > 0 else (take.name if take.name.byte_length() > 0 else String("rgbmocap"))
    return clip_to_native_animation(clip, mapping, anim_name)


# ---------------------------------------------------------------- synthetic demo take


def _soma_local_quat(orient: List[MbQuat], parents: List[Int], soma_idx: Int, delta_world: Quat) -> MbQuat:
    """Convert a T-pose-relative world delta rotation into SOMA-X's T-pose-oriented local frame."""
    var dq = _quat_to_mb(delta_world).normalized()
    if soma_idx == 1:
        return (dq * orient[1]).normalized()
    return (orient[parents[soma_idx]].inverse() * dq * orient[soma_idx]).normalized()


def _det_jitter(frame_idx: Int, channel: Int) -> Float32:
    """Deterministic zero-mean high-frequency tremor in [-1, 1] for One Euro Filter testing."""
    var a = Float32(frame_idx) * Float32(1.73) + Float32(channel) * Float32(2.41)
    var b = Float32(frame_idx) * Float32(2.89) + Float32(channel) * Float32(0.97)
    return Float32(0.6) * det_sin(a) + Float32(0.4) * det_cos(b)


def _clamp01(x: Float32) -> Float32:
    return min(Float32(1.0), max(Float32(0.0), x))


def generate_demo_mocap_take(num_frames: Int = 90, fps: Float32 = 30.0) -> MocapTake:
    """Generate a deterministic realistic 77-joint SOMA-X walking + turning + wave MocapTake.

    Includes:
      - Realistic micro-jitter on positions and rotations (for One Euro Filter verification)
      - Mild planted-foot slide (~6.5 cm) during stance phases (for Foot Contact Lock verification)
      - 2 tracked subjects (`Person #1` primary ~94% conf, `Person #2` background ~78% conf)
      - Per-frame confidence dips (emerald / amber / red heatbar coverage)
      - Smoothly moving 3D camera frustum tracking the subject
    """
    var n_frames = max(1, num_frames)
    var actual_fps = fps if fps > Float32(0.0) else Float32(30.0)
    var skel = soma77_skeleton()
    var parents78 = soma78_parents()
    var oq = soma78_t_pose_orient()
    var orient = List[MbQuat](capacity=78)
    for j in range(78):
        orient.append(MbQuat(oq[4 * j], oq[4 * j + 1], oq[4 * j + 2], oq[4 * j + 3]).normalized())

    var take = MocapTake()
    take.name = "Demo Walk & Wave (SOMA-77)"
    take.backend_id = MOCAP_BACKEND_GEMX
    take.fps = actual_fps
    take.source_path = "demo://walk_turn_wave_90f.mp4"
    take.joint_names = skel.names.copy()
    take.parent_indices = skel.parents.copy()
    take.selected_track_id = 1

    # Synthetic 2-person detections fed through the real greedy IoU tracker
    var dets_per_frame = List[List[MocapBBox]](capacity=n_frames)
    comptime TWO_PI: Float32 = 6.283185307179586

    for f in range(n_frames):
        var t = Float32(f) / actual_fps
        var u = Float32(f) / Float32(max(1, n_frames - 1))

        # Walking trajectory along -Z with gentle 35-deg right turn
        var yaw = Float32(0.61) * (u * u)  # 0 -> ~35 deg
        var dist = Float32(1.75) * u
        var root_x = Float32(0.32) * (u * u) + Float32(0.025) * det_sin(TWO_PI * t * Float32(0.85))
        var root_z = Float32(0.85) - dist
        var gait_phase = TWO_PI * t * Float32(0.85)  # ~0.85 full strides/sec -> ~0.58s stance
        var bob = Float32(0.028) * abs(det_sin(gait_phase))
        var jitter_x = Float32(0.008) * _det_jitter(f, 1)
        var jitter_y = Float32(0.006) * _det_jitter(f, 2)
        var jitter_z = Float32(0.008) * _det_jitter(f, 3)

        # Build local pose in SOMA-77 frame
        var nj = skel.num_joints()
        var local_xf = List[MbXform](capacity=nj)
        for j in range(nj):
            local_xf.append(skel.reference[j])

        var root_pos = Vec3(
            root_x + jitter_x,
            skel.reference[0].p.y - Float32(0.025) + bob + jitter_y,
            root_z + jitter_z,
        )
        local_xf[0].p = _vec3_to_mb(root_pos)

        # Root + spine + head rotations (with subtle jitter)
        var hip_roll = Float32(0.04) * det_sin(gait_phase) + Float32(0.015) * _det_jitter(f, 4)
        var hip_q = Quat.from_euler(Float32(0.03), yaw, hip_roll)
        local_xf[SOMA77_HIPS].q = _soma_local_quat(orient, parents78, SOMA77_HIPS + 1, hip_q)

        var spine_twist = Float32(-0.08) * det_sin(gait_phase)
        local_xf[SOMA77_SPINE1].q = _soma_local_quat(
            orient, parents78, SOMA77_SPINE1 + 1, Quat.from_euler(Float32(0.02), spine_twist * Float32(0.5), Float32(0.0))
        )
        local_xf[SOMA77_CHEST].q = _soma_local_quat(
            orient, parents78, SOMA77_CHEST + 1, Quat.from_euler(Float32(0.02), spine_twist, Float32(0.0))
        )
        var head_look = Float32(-0.18) * det_sin(u * Float32(3.14159265)) + Float32(0.02) * _det_jitter(f, 5)
        local_xf[SOMA77_HEAD].q = _soma_local_quat(
            orient, parents78, SOMA77_HEAD + 1, Quat.from_euler(Float32(-0.04), head_look, Float32(0.0))
        )

        # Legs: alternating walk gait (left swing when sin > 0, right swing when sin < 0)
        var l_swing = det_sin(gait_phase)
        var r_swing = -l_swing
        var l_hip_pitch = Float32(0.34) * l_swing + Float32(0.02) * _det_jitter(f, 6)
        var r_hip_pitch = Float32(0.34) * r_swing + Float32(0.02) * _det_jitter(f, 7)
        var l_knee_bend = Float32(0.08) + Float32(0.52) * max(Float32(0.0), l_swing)
        var r_knee_bend = Float32(0.08) + Float32(0.52) * max(Float32(0.0), r_swing)
        var l_ankle_pitch = Float32(-0.15) * l_swing
        var r_ankle_pitch = Float32(-0.15) * r_swing

        local_xf[SOMA77_LEFT_LEG].q = _soma_local_quat(
            orient, parents78, SOMA77_LEFT_LEG + 1, Quat.from_euler(l_hip_pitch, Float32(0.0), Float32(0.0))
        )
        local_xf[SOMA77_LEFT_SHIN].q = _soma_local_quat(
            orient, parents78, SOMA77_LEFT_SHIN + 1, Quat.from_euler(-l_knee_bend, Float32(0.0), Float32(0.0))
        )
        local_xf[SOMA77_LEFT_FOOT].q = _soma_local_quat(
            orient, parents78, SOMA77_LEFT_FOOT + 1, Quat.from_euler(l_ankle_pitch, Float32(0.0), Float32(0.0))
        )
        local_xf[SOMA77_RIGHT_LEG].q = _soma_local_quat(
            orient, parents78, SOMA77_RIGHT_LEG + 1, Quat.from_euler(r_hip_pitch, Float32(0.0), Float32(0.0))
        )
        local_xf[SOMA77_RIGHT_SHIN].q = _soma_local_quat(
            orient, parents78, SOMA77_RIGHT_SHIN + 1, Quat.from_euler(-r_knee_bend, Float32(0.0), Float32(0.0))
        )
        local_xf[SOMA77_RIGHT_FOOT].q = _soma_local_quat(
            orient, parents78, SOMA77_RIGHT_FOOT + 1, Quat.from_euler(r_ankle_pitch, Float32(0.0), Float32(0.0))
        )

        # Left arm counter-swing (lowered from T-pose ~68 deg)
        var l_arm_pitch = Float32(-0.26) * l_swing + Float32(0.02) * _det_jitter(f, 8)
        local_xf[SOMA77_LEFT_ARM].q = _soma_local_quat(
            orient, parents78, SOMA77_LEFT_ARM + 1, Quat.from_euler(l_arm_pitch, Float32(0.0), Float32(-1.18))
        )
        local_xf[SOMA77_LEFT_FOREARM].q = _soma_local_quat(
            orient, parents78, SOMA77_LEFT_FOREARM + 1, Quat.from_euler(Float32(0.28), Float32(0.0), Float32(0.0))
        )

        # Right arm: natural swing, then raises into a wave during u in [0.35, 0.85]
        var wave_env = Float32(0.0)
        if u > Float32(0.32) and u < Float32(0.88):
            var wu = (u - Float32(0.32)) / Float32(0.56)
            wave_env = det_sin(wu * Float32(3.14159265))
        var r_arm_roll = Float32(1.18) * (Float32(1.0) - wave_env) + Float32(-0.55) * wave_env
        var r_arm_pitch = Float32(-0.26) * r_swing * (Float32(1.0) - wave_env) + Float32(0.35) * wave_env
        var r_forearm_wave = Float32(0.28) + wave_env * (Float32(0.95) + Float32(0.35) * det_sin(TWO_PI * t * Float32(2.6)))
        local_xf[SOMA77_RIGHT_ARM].q = _soma_local_quat(
            orient, parents78, SOMA77_RIGHT_ARM + 1, Quat.from_euler(r_arm_pitch, Float32(0.0), r_arm_roll)
        )
        local_xf[SOMA77_RIGHT_FOREARM].q = _soma_local_quat(
            orient, parents78, SOMA77_RIGHT_FOREARM + 1, Quat.from_euler(r_forearm_wave, Float32(0.0), Float32(0.0))
        )

        # Forward kinematics on SOMA-77
        var world_xf = world_pose(skel, local_xf, MbXform.identity())

        # Contact flags & realistic planted-foot stance with ~6.5 cm intentional slide
        # Stance is strictly sin(phase) <= -0.05 (phase mod 2pi in [pi+0.05, 2pi-0.05]),
        # so stride step index transitions cleanly at mid-swing (never mid-stance).
        var l_contact = l_swing <= Float32(-0.05)
        var r_contact = r_swing <= Float32(-0.05)
        var cycles_l = gait_phase / TWO_PI
        var l_step_idx = Int(cycles_l)
        var l_cycle_frac = cycles_l - Float32(l_step_idx)  # [0, 1); stance is [0.5, 1.0)
        var l_stance_u = _clamp01((l_cycle_frac - Float32(0.5)) * Float32(2.0))

        var cycles_r = (gait_phase + Float32(3.14159265)) / TWO_PI
        var r_step_idx = Int(cycles_r)
        var r_cycle_frac = cycles_r - Float32(r_step_idx)
        var r_stance_u = _clamp01((r_cycle_frac - Float32(0.5)) * Float32(2.0))

        var mf = MocapFrame(f, t)
        mf.root_pos = root_pos
        mf.left_foot_contact = l_contact
        mf.right_foot_contact = r_contact
        mf.left_palm_contact = False
        mf.right_palm_contact = False

        # Per-frame confidence curve with realistic occlusion dips at ~33% and ~73%
        var dip1 = max(Float32(0.0), Float32(1.0) - abs(u - Float32(0.34)) / Float32(0.07))
        var dip2 = max(Float32(0.0), Float32(1.0) - abs(u - Float32(0.73)) / Float32(0.06))
        var frame_base_conf = Float32(0.94) - Float32(0.30) * dip1 - Float32(0.45) * dip2

        var conf_sum = Float32(0.0)
        for j in range(nj):
            var p = _mb_to_vec3(world_xf[j].p)
            # Subtle per-joint micro-jitter (~6 mm)
            var jx = Float32(0.006) * _det_jitter(f, j * 3 + 11)
            var jy = Float32(0.005) * _det_jitter(f, j * 3 + 12)
            var jz = Float32(0.006) * _det_jitter(f, j * 3 + 13)
            p = Vec3(p.x + jx, max(Float32(0.015), p.y + jy), p.z + jz)

            # Shape left/right foot & toe trajectories so stance feet sit near ground with ~6.5 cm slide
            if j == SOMA77_LEFT_FOOT or j == SOMA77_LEFT_TOE_BASE:
                var step_u = min(Float32(1.0), (Float32(l_step_idx) + Float32(0.75)) / Float32(2.6))
                var anchor_z = Float32(0.85) - Float32(1.75) * step_u
                var anchor_x = Float32(0.11) + Float32(0.32) * (step_u * step_u)
                if l_contact:
                    var slide_z = Float32(-0.068) * l_stance_u
                    var slide_x = Float32(0.016) * l_stance_u
                    var toe_off = Float32(-0.12) if j == SOMA77_LEFT_TOE_BASE else Float32(0.0)
                    var foot_y = Float32(0.022) if j == SOMA77_LEFT_TOE_BASE else Float32(0.048)
                    p = Vec3(
                        anchor_x + slide_x + jx * Float32(0.12),
                        foot_y + abs(jy) * Float32(0.25),
                        anchor_z + slide_z + toe_off + jz * Float32(0.12),
                    )
                else:
                    var lift = Float32(0.14) * max(Float32(0.0), l_swing)
                    p = Vec3(p.x, Float32(0.06) + lift, p.z)
            elif j == SOMA77_RIGHT_FOOT or j == SOMA77_RIGHT_TOE_BASE:
                var step_u = min(Float32(1.0), (Float32(r_step_idx) + Float32(0.25)) / Float32(2.6))
                var anchor_z = Float32(0.85) - Float32(1.75) * step_u
                var anchor_x = Float32(-0.11) + Float32(0.32) * (step_u * step_u)
                if r_contact:
                    var slide_z = Float32(-0.068) * r_stance_u
                    var slide_x = Float32(-0.016) * r_stance_u
                    var toe_off = Float32(-0.12) if j == SOMA77_RIGHT_TOE_BASE else Float32(0.0)
                    var foot_y = Float32(0.022) if j == SOMA77_RIGHT_TOE_BASE else Float32(0.048)
                    p = Vec3(
                        anchor_x + slide_x + jx * Float32(0.12),
                        foot_y + abs(jy) * Float32(0.25),
                        anchor_z + slide_z + toe_off + jz * Float32(0.12),
                    )
                else:
                    var lift = Float32(0.14) * max(Float32(0.0), r_swing)
                    p = Vec3(p.x, Float32(0.06) + lift, p.z)

            mf.joint_positions.append(p)
            mf.joint_rotations.append(_mb_to_quat(local_xf[j].q))
            var j_conf = _clamp01(frame_base_conf + Float32(0.04) * _det_jitter(f, j + 50))
            mf.joint_confidences.append(j_conf)
            conf_sum += j_conf

        mf.mean_confidence = conf_sum / Float32(nj) if nj > 0 else Float32(0.0)

        # Moving camera frustum tracking the walker
        var cam_angle = Float32(-0.22) + Float32(0.48) * u
        var cam_pos = Vec3(
            root_pos.x + Float32(3.1) * det_sin(cam_angle),
            Float32(1.52) + Float32(0.08) * det_sin(u * Float32(3.14159265)),
            root_pos.z + Float32(3.1) * det_cos(cam_angle),
        )
        var look_target = Vec3(root_pos.x, Float32(1.02), root_pos.z)
        mf.camera = MocapCameraPose(
            cam_pos,
            (look_target - cam_pos).normalized(),
            Vec3(0.0, 1.0, 0.0),
            Float32(45.0),
            Float32(16.0) / Float32(9.0),
        )
        take.frames.append(mf^)

        # Frame detections for 2-person tracking (1280x720 virtual video frame)
        var f_dets = List[MocapBBox](capacity=2)
        var p1_cx = Float32(560.0) + Float32(140.0) * u
        var p1_conf = _clamp01(Float32(0.95) - Float32(0.04) * dip1)
        f_dets.append(MocapBBox(p1_cx - Float32(115.0), Float32(120.0), p1_cx + Float32(115.0), Float32(640.0), p1_conf))

        var bg_start = n_frames // 6
        var bg_end = (n_frames * 5) // 6
        if f >= bg_start and f <= bg_end:
            var bu = Float32(f - bg_start) / Float32(max(1, bg_end - bg_start))
            var p2_cx = Float32(1040.0) - Float32(160.0) * bu
            var p2_conf = _clamp01(Float32(0.78) + Float32(0.05) * det_sin(bu * TWO_PI))
            f_dets.append(MocapBBox(p2_cx - Float32(68.0), Float32(210.0), p2_cx + Float32(68.0), Float32(560.0), p2_conf))
        dets_per_frame.append(f_dets^)

    take.tracks = track_person_boxes(dets_per_frame, iou_threshold=Float32(0.30), max_missed_frames=8)
    take.foot_slide_before_cm = _estimate_contact_foot_slide_cm(
        take.frames, SOMA77_LEFT_FOOT, SOMA77_RIGHT_FOOT
    )
    take.foot_slide_after_cm = take.foot_slide_before_cm
    return take^
