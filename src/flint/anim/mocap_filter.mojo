"""Pure-Mojo deterministic motion filtering and FootMR-style contact cleanup.

Commercial-safe, zero-dependency port of RGBMocap / Casiez et al. (2012)
One Euro filtering, robust ground-plane estimation, hysteresis foot/palm
contact detection, cosine-faded anchor locking, and exact planar 2-bone IK
limb cleanup.

All trigonometric operations use `flint.core.detmath` so motion cleanup is
100% lockstep-deterministic across platforms under `--fp-mode contract=off`.
"""
from std.math import isfinite, sqrt
from flint.core.detmath import det_acos, det_cos, PI
from flint.core.ik import dot, ortho_fallback, safe_normalized, shortest_arc_quat
from flint.core.math import Quat, Vec3
from flint.ml.rgbmocap import (
    MocapTake,
    SOMA77_LEFT_ARM,
    SOMA77_LEFT_FOOT,
    SOMA77_LEFT_FOREARM,
    SOMA77_LEFT_HAND,
    SOMA77_LEFT_LEG,
    SOMA77_LEFT_SHIN,
    SOMA77_RIGHT_ARM,
    SOMA77_RIGHT_FOOT,
    SOMA77_RIGHT_FOREARM,
    SOMA77_RIGHT_HAND,
    SOMA77_RIGHT_LEG,
    SOMA77_RIGHT_SHIN,
)

comptime TWO_PI: Float32 = Float32(6.28318530717958647692)


@always_inline
def _vec3_finite(v: Vec3) -> Bool:
    return isfinite(v.x) and isfinite(v.y) and isfinite(v.z)


@always_inline
def _quat_finite(q: Quat) -> Bool:
    return isfinite(q.x) and isfinite(q.y) and isfinite(q.z) and isfinite(q.w)


@always_inline
def _one_euro_alpha(cutoff: Float32, dt: Float32) -> Float32:
    """Exact One Euro smoothing factor `alpha = 1 / (1 + tau / dt)` with `tau = 1 / (2*pi*fc)`."""
    if not isfinite(dt) or dt <= Float32(1.0e-6):
        return Float32(1.0)
    var fc = max(Float32(1.0e-4), cutoff) if isfinite(cutoff) else Float32(1.0)
    var x = TWO_PI * fc * dt
    return x / (Float32(1.0) + x)


def _heapsort_f32(mut vals: List[Float32]):
    """Deterministic in-place O(N log N) heapsort for finite Float32 values."""
    var n = len(vals)
    if n < 2:
        return
    # Build max-heap
    var start = (n - 2) // 2
    while start >= 0:
        var root = start
        while root * 2 + 1 < n:
            var child = root * 2 + 1
            var swap_idx = root
            if vals[swap_idx] < vals[child]:
                swap_idx = child
            if child + 1 < n and vals[swap_idx] < vals[child + 1]:
                swap_idx = child + 1
            if swap_idx == root:
                break
            var tmp = vals[root]
            vals[root] = vals[swap_idx]
            vals[swap_idx] = tmp
            root = swap_idx
        start -= 1

    # Extract elements
    var end = n - 1
    while end > 0:
        var tmp = vals[end]
        vals[end] = vals[0]
        vals[0] = tmp
        end -= 1
        var root = 0
        while root * 2 + 1 <= end:
            var child = root * 2 + 1
            var swap_idx = root
            if vals[swap_idx] < vals[child]:
                swap_idx = child
            if child + 1 <= end and vals[swap_idx] < vals[child + 1]:
                swap_idx = child + 1
            if swap_idx == root:
                break
            var tmp2 = vals[root]
            vals[root] = vals[swap_idx]
            vals[swap_idx] = tmp2
            root = swap_idx


# ---------------------------------------------------------------------------
# 1. One Euro Filter (Casiez et al. 2012)
# ---------------------------------------------------------------------------


struct OneEuroConfig(Copyable, ImplicitlyCopyable, Movable):
    """Adaptive low-pass filter parameters (low jitter at rest, low lag when fast)."""

    var min_cutoff: Float32
    var beta: Float32
    var d_cutoff: Float32

    def __init__(
        out self,
        min_cutoff: Float32 = 1.0,
        beta: Float32 = 0.05,
        d_cutoff: Float32 = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff


struct OneEuroScalar(Copyable, ImplicitlyCopyable, Movable):
    """1D One Euro filter for scalar motion channels."""

    var cfg: OneEuroConfig
    var x_prev: Float32
    var dx_prev: Float32
    var initialized: Bool

    def __init__(out self, cfg: OneEuroConfig = OneEuroConfig()):
        self.cfg = cfg
        self.x_prev = Float32(0.0)
        self.dx_prev = Float32(0.0)
        self.initialized = False

    def reset(mut self):
        self.x_prev = Float32(0.0)
        self.dx_prev = Float32(0.0)
        self.initialized = False

    def filter(mut self, x: Float32, dt: Float32) -> Float32:
        if not isfinite(x):
            return self.x_prev if self.initialized else Float32(0.0)
        if not self.initialized or not isfinite(dt) or dt <= Float32(1.0e-6):
            self.x_prev = x
            self.dx_prev = Float32(0.0)
            self.initialized = True
            return x
        var dx = (x - self.x_prev) / dt
        var a_d = _one_euro_alpha(self.cfg.d_cutoff, dt)
        var dx_hat = a_d * dx + (Float32(1.0) - a_d) * self.dx_prev
        var min_fc = max(Float32(1.0e-4), self.cfg.min_cutoff) if isfinite(self.cfg.min_cutoff) else Float32(1.0)
        var beta = max(Float32(0.0), self.cfg.beta) if isfinite(self.cfg.beta) else Float32(0.05)
        var cutoff = min_fc + beta * abs(dx_hat)
        var a = _one_euro_alpha(cutoff, dt)
        var x_hat = a * x + (Float32(1.0) - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


struct OneEuroVec3(Copyable, ImplicitlyCopyable, Movable):
    """3D One Euro filter smoothing root and joint world positions per axis."""

    var fx: OneEuroScalar
    var fy: OneEuroScalar
    var fz: OneEuroScalar

    def __init__(out self, cfg: OneEuroConfig = OneEuroConfig()):
        self.fx = OneEuroScalar(cfg)
        self.fy = OneEuroScalar(cfg)
        self.fz = OneEuroScalar(cfg)

    def reset(mut self):
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()

    def filter(mut self, v: Vec3, dt: Float32) -> Vec3:
        return Vec3(
            self.fx.filter(v.x, dt),
            self.fy.filter(v.y, dt),
            self.fz.filter(v.z, dt),
        )


struct OneEuroQuat(Copyable, ImplicitlyCopyable, Movable):
    """Hemisphere-aligned, angular-speed adaptive SLERP One Euro filter for joint rotations."""

    var cfg: OneEuroConfig
    var q_prev: Quat
    var d_angle_prev: Float32
    var initialized: Bool

    def __init__(out self, cfg: OneEuroConfig = OneEuroConfig()):
        self.cfg = cfg
        self.q_prev = Quat.identity()
        self.d_angle_prev = Float32(0.0)
        self.initialized = False

    def reset(mut self):
        self.q_prev = Quat.identity()
        self.d_angle_prev = Float32(0.0)
        self.initialized = False

    def filter(mut self, q_in: Quat, dt: Float32) -> Quat:
        if not _quat_finite(q_in):
            return self.q_prev if self.initialized else Quat.identity()
        var q = q_in.normalized()
        if not self.initialized or not isfinite(dt) or dt <= Float32(1.0e-6):
            self.q_prev = q
            self.d_angle_prev = Float32(0.0)
            self.initialized = True
            return q
        # Enforce shortest-arc hemisphere continuity (prevent double-cover sign flips)
        var d = self.q_prev.dot(q)
        if d < Float32(0.0):
            q = Quat(-q.x, -q.y, -q.z, -q.w)
            d = -d
        var cos_half = max(Float32(-1.0), min(Float32(1.0), d))
        var angle = Float32(2.0) * det_acos(cos_half)
        var ang_speed = angle / dt
        var a_d = _one_euro_alpha(self.cfg.d_cutoff, dt)
        var d_hat = a_d * ang_speed + (Float32(1.0) - a_d) * self.d_angle_prev
        var min_fc = max(Float32(1.0e-4), self.cfg.min_cutoff) if isfinite(self.cfg.min_cutoff) else Float32(1.0)
        var beta = max(Float32(0.0), self.cfg.beta) if isfinite(self.cfg.beta) else Float32(0.05)
        var cutoff = min_fc + beta * abs(d_hat)
        var a = _one_euro_alpha(cutoff, dt)
        var q_hat = self.q_prev.slerp(q, a).normalized()
        self.q_prev = q_hat
        self.d_angle_prev = d_hat
        return q_hat


def filter_vec3_series(
    series: List[Vec3], fps: Float32, cfg: OneEuroConfig
) -> List[Vec3]:
    """Applies One Euro smoothing to a time series of 3D positions."""
    var n = len(series)
    var out = List[Vec3](capacity=n)
    if n == 0:
        return out^
    var safe_fps = fps if (isfinite(fps) and fps > Float32(1.0e-4)) else Float32(30.0)
    var dt = Float32(1.0) / safe_fps
    var flt = OneEuroVec3(cfg)
    for i in range(n):
        out.append(flt.filter(series[i], dt))
    return out^


def filter_quat_series(
    series: List[Quat], fps: Float32, cfg: OneEuroConfig
) -> List[Quat]:
    """Applies hemisphere-checked adaptive SLERP One Euro smoothing to a quaternion series."""
    var n = len(series)
    var out = List[Quat](capacity=n)
    if n == 0:
        return out^
    var safe_fps = fps if (isfinite(fps) and fps > Float32(1.0e-4)) else Float32(30.0)
    var dt = Float32(1.0) / safe_fps
    var flt = OneEuroQuat(cfg)
    for i in range(n):
        out.append(flt.filter(series[i], dt))
    return out^


# ---------------------------------------------------------------------------
# 2. Ground Plane Estimation & Foot/Palm Contact Cleanup (FootMR-style)
# ---------------------------------------------------------------------------


struct ContactCleanupConfig(Copyable, ImplicitlyCopyable, Movable):
    """Configuration for One Euro smoothing, ground alignment, and foot/palm contact locking."""

    var enable_one_euro: Bool
    var one_euro: OneEuroConfig
    var enable_foot_lock: Bool
    var foot_height_thresh_m: Float32
    var foot_speed_thresh_mps: Float32
    var foot_lock_strength: Float32
    var contact_fade_frames: Int
    var min_contact_frames: Int
    var enable_palm_lock: Bool
    var palm_speed_thresh_mps: Float32
    var palm_lock_strength: Float32
    var enable_ground_align: Bool
    var ground_y: Float32
    var auto_estimate_ground: Bool

    def __init__(
        out self,
        enable_one_euro: Bool = True,
        one_euro: OneEuroConfig = OneEuroConfig(),
        enable_foot_lock: Bool = True,
        foot_height_thresh_m: Float32 = 0.08,
        foot_speed_thresh_mps: Float32 = 0.22,
        foot_lock_strength: Float32 = 0.85,
        contact_fade_frames: Int = 3,
        min_contact_frames: Int = 3,
        enable_palm_lock: Bool = False,
        palm_speed_thresh_mps: Float32 = 0.14,
        palm_lock_strength: Float32 = 0.75,
        enable_ground_align: Bool = True,
        ground_y: Float32 = 0.0,
        auto_estimate_ground: Bool = True,
    ):
        self.enable_one_euro = enable_one_euro
        self.one_euro = one_euro
        self.enable_foot_lock = enable_foot_lock
        self.foot_height_thresh_m = foot_height_thresh_m
        self.foot_speed_thresh_mps = foot_speed_thresh_mps
        self.foot_lock_strength = foot_lock_strength
        self.contact_fade_frames = contact_fade_frames
        self.min_contact_frames = min_contact_frames
        self.enable_palm_lock = enable_palm_lock
        self.palm_speed_thresh_mps = palm_speed_thresh_mps
        self.palm_lock_strength = palm_lock_strength
        self.enable_ground_align = enable_ground_align
        self.ground_y = ground_y
        self.auto_estimate_ground = auto_estimate_ground


struct ContactInterval(Copyable, ImplicitlyCopyable, Movable):
    """Detected planted foot or palm contact interval `[start_frame, end_frame]` (inclusive)."""

    var start_frame: Int
    var end_frame: Int
    var anchor_pos: Vec3
    var is_left: Bool
    var is_palm: Bool

    def __init__(
        out self,
        start_frame: Int = 0,
        end_frame: Int = 0,
        anchor_pos: Vec3 = Vec3(),
        is_left: Bool = True,
        is_palm: Bool = False,
    ):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.anchor_pos = anchor_pos
        self.is_left = is_left
        self.is_palm = is_palm


struct MocapCleanupMetrics(Copyable, ImplicitlyCopyable, Movable):
    """Before/after telemetry from motion filtering and contact cleanup."""

    var estimated_ground_y: Float32
    var foot_contacts_detected: Int
    var palm_contacts_detected: Int
    var foot_slide_before_cm: Float32
    var foot_slide_after_cm: Float32
    var mean_jitter_before: Float32
    var mean_jitter_after: Float32

    def __init__(
        out self,
        estimated_ground_y: Float32 = 0.0,
        foot_contacts_detected: Int = 0,
        palm_contacts_detected: Int = 0,
        foot_slide_before_cm: Float32 = 0.0,
        foot_slide_after_cm: Float32 = 0.0,
        mean_jitter_before: Float32 = 0.0,
        mean_jitter_after: Float32 = 0.0,
    ):
        self.estimated_ground_y = estimated_ground_y
        self.foot_contacts_detected = foot_contacts_detected
        self.palm_contacts_detected = palm_contacts_detected
        self.foot_slide_before_cm = foot_slide_before_cm
        self.foot_slide_after_cm = foot_slide_after_cm
        self.mean_jitter_before = mean_jitter_before
        self.mean_jitter_after = mean_jitter_after


def estimate_ground_plane_y(
    left_foot: List[Vec3], right_foot: List[Vec3]
) -> Float32:
    """Estimates the robust stance floor height Y from left and right foot tracks.

    Averages the 5th-15th percentile of combined finite foot Y samples so single-frame
    outlier dips are rejected while stance height is recovered accurately.
    """
    var ys = List[Float32](capacity=len(left_foot) + len(right_foot))
    for i in range(len(left_foot)):
        var y = left_foot[i].y
        if isfinite(y):
            ys.append(y)
    for i in range(len(right_foot)):
        var y = right_foot[i].y
        if isfinite(y):
            ys.append(y)
    var n = len(ys)
    if n == 0:
        return Float32(0.0)
    _heapsort_f32(ys)
    if n <= 4:
        return ys[0]
    var lo = (n * 5) // 100
    var hi = max(lo + 1, (n * 15) // 100)
    if hi > n:
        hi = n
    var total = Float32(0.0)
    for i in range(lo, hi):
        total += ys[i]
    return total / Float32(hi - lo)


def detect_contact_intervals(
    positions: List[Vec3],
    fps: Float32,
    ground_y: Float32,
    height_thresh: Float32,
    speed_thresh: Float32,
    min_frames: Int,
    is_left: Bool,
    is_palm: Bool = False,
) -> List[ContactInterval]:
    """Detects planted contact intervals using hysteresis, short-gap bridging, and min-duration filtering.

    For feet (`is_palm=False`), requires both proximity to `ground_y` (`y <= ground_y + height_thresh`)
    and low speed (`<= speed_thresh`). For palms (`is_palm=True`), if `height_thresh <= 0`, contact
    is detected by speed alone so elevated wall/table/prop contacts are supported.
    """
    var intervals = List[ContactInterval]()
    var n = len(positions)
    var min_len = max(1, min_frames)
    if n < min_len:
        return intervals^

    var safe_fps = fps if (isfinite(fps) and fps > Float32(1.0e-4)) else Float32(30.0)
    var safe_ground = ground_y if isfinite(ground_y) else Float32(0.0)
    var safe_h = height_thresh if isfinite(height_thresh) else Float32(0.08)
    var safe_v = max(Float32(1.0e-4), speed_thresh) if isfinite(speed_thresh) else Float32(0.22)
    var check_height = (not is_palm) or (safe_h > Float32(0.0))

    # 1. Per-frame speed via central finite difference
    var speeds = List[Float32](length=n, fill=Float32(0.0))
    if n >= 2:
        for f in range(n):
            var lo = max(0, f - 1)
            var hi = min(n - 1, f + 1)
            var dp = positions[hi] - positions[lo]
            if _vec3_finite(dp) and hi > lo:
                speeds[f] = dp.length() * safe_fps / Float32(hi - lo)
            else:
                speeds[f] = Float32(1.0e6)

    # 2. Hysteresis contact classification (enter strict, exit 1.35x relaxed)
    var enter_h = safe_h
    var exit_h = safe_h * Float32(1.35)
    var enter_v = safe_v
    var exit_v = safe_v * Float32(1.35)

    var mask = List[Bool](length=n, fill=False)
    var active = False
    for f in range(n):
        var p = positions[f]
        if not _vec3_finite(p):
            active = False
            mask[f] = False
            continue
        var spd = speeds[f]
        var h_ok: Bool
        var v_ok: Bool
        if active:
            h_ok = (not check_height) or (p.y <= safe_ground + exit_h)
            v_ok = spd <= exit_v
        else:
            h_ok = (not check_height) or (p.y <= safe_ground + enter_h)
            v_ok = spd <= enter_v
        active = h_ok and v_ok
        mask[f] = active

    # 3. Short-gap bridging (up to 3 frames — absorbs 1-frame tracking spikes in central-diff speed)
    comptime MAX_BRIDGE_GAP: Int = 3
    var f = 0
    while f < n:
        if mask[f]:
            f += 1
            continue
        var gap_start = f
        while f < n and not mask[f]:
            f += 1
        var gap_end = f  # exclusive
        var gap_len = gap_end - gap_start
        if (
            gap_start > 0
            and gap_end < n
            and gap_len <= MAX_BRIDGE_GAP
            and mask[gap_start - 1]
            and mask[gap_end]
        ):
            var span_dist = (positions[gap_end] - positions[gap_start - 1]).length()
            var max_span = max(
                Float32(0.05),
                safe_v * Float32(gap_len + 1) / safe_fps * Float32(1.5),
            )
            if isfinite(span_dist) and span_dist <= max_span:
                for g in range(gap_start, gap_end):
                    mask[g] = True

    # 4. Extract runs >= min_len and compute speed-weighted anchor positions
    var run_start = -1
    for i in range(n + 1):
        var in_c = mask[i] if i < n else False
        if in_c and run_start < 0:
            run_start = i
        elif (not in_c) and run_start >= 0:
            var run_end = i - 1
            if run_end - run_start + 1 >= min_len:
                var sum_pos = Vec3()
                var sum_w = Float32(0.0)
                for k in range(run_start, run_end + 1):
                    var pk = positions[k]
                    if _vec3_finite(pk):
                        var w = Float32(1.0) / (Float32(0.05) + max(Float32(0.0), speeds[k]))
                        sum_pos = sum_pos + pk * w
                        sum_w += w
                var anchor = (
                    sum_pos * (Float32(1.0) / sum_w)
                    if sum_w > Float32(1.0e-6)
                    else positions[run_start]
                )
                intervals.append(
                    ContactInterval(run_start, run_end, anchor, is_left, is_palm)
                )
            run_start = -1

    return intervals^


def measure_foot_slide_cm(
    positions: List[Vec3], intervals: List[ContactInterval]
) -> Float32:
    """Measures mean horizontal (XZ) sliding travel per planted contact interval in centimeters."""
    var n = len(positions)
    var num_iv = len(intervals)
    if n < 2 or num_iv == 0:
        return Float32(0.0)

    var total_travel_m = Float32(0.0)
    var counted = 0
    for idx in range(num_iv):
        var s = max(0, intervals[idx].start_frame)
        var e = min(n - 1, intervals[idx].end_frame)
        if e <= s:
            continue
        var travel_m = Float32(0.0)
        for f in range(s, e):
            var p0 = positions[f]
            var p1 = positions[f + 1]
            if _vec3_finite(p0) and _vec3_finite(p1):
                var dx = p1.x - p0.x
                var dz = p1.z - p0.z
                travel_m += sqrt(dx * dx + dz * dz)
        total_travel_m += travel_m
        counted += 1

    if counted == 0:
        return Float32(0.0)
    return (total_travel_m / Float32(counted)) * Float32(100.0)


def measure_series_jitter(positions: List[Vec3]) -> Float32:
    """Measures mean 2nd-difference magnitude `|p[f+1] - 2*p[f] + p[f-1]|` across interior frames."""
    var n = len(positions)
    if n < 3:
        return Float32(0.0)
    var total = Float32(0.0)
    var counted = 0
    for f in range(1, n - 1):
        var p0 = positions[f - 1]
        var p1 = positions[f]
        var p2 = positions[f + 1]
        if _vec3_finite(p0) and _vec3_finite(p1) and _vec3_finite(p2):
            var acc = (p2 - p1) - (p1 - p0)
            total += acc.length()
            counted += 1
    if counted == 0:
        return Float32(0.0)
    return total / Float32(counted)


def lock_contact_positions(
    positions: List[Vec3],
    intervals: List[ContactInterval],
    ground_y: Float32,
    strength: Float32,
    fade_frames: Int,
    clamp_y_to_ground: Bool = True,
) -> List[Vec3]:
    """Applies smooth cosine fade-in/out anchor locking over detected contact intervals."""
    var n = len(positions)
    var out = List[Vec3](capacity=n)
    for i in range(n):
        out.append(positions[i])

    var safe_strength = (
        max(Float32(0.0), min(Float32(1.0), strength))
        if isfinite(strength)
        else Float32(0.0)
    )
    if n == 0 or len(intervals) == 0 or safe_strength <= Float32(0.0):
        return out^

    var safe_ground = ground_y if isfinite(ground_y) else Float32(0.0)
    var fade = max(0, fade_frames)
    var delta_sum = List[Vec3](length=n, fill=Vec3())
    var weight_sum = List[Float32](length=n, fill=Float32(0.0))

    for idx in range(len(intervals)):
        var iv = intervals[idx]
        var s = max(0, iv.start_frame)
        var e = min(n - 1, iv.end_frame)
        if e < s:
            continue
        var target = Vec3(
            iv.anchor_pos.x,
            safe_ground if clamp_y_to_ground else iv.anchor_pos.y,
            iv.anchor_pos.z,
        )
        if not _vec3_finite(target):
            continue
        var win_lo = max(0, s - fade)
        var win_hi = min(n - 1, e + fade)
        for f in range(win_lo, win_hi + 1):
            var p = positions[f]
            if not _vec3_finite(p):
                continue
            var w = Float32(1.0)
            if f < s:
                var dist = s - f
                var u = Float32(dist) / Float32(fade + 1)
                w = Float32(0.5) * (Float32(1.0) + det_cos(PI * u))
            elif f > e:
                var dist = f - e
                var u = Float32(dist) / Float32(fade + 1)
                w = Float32(0.5) * (Float32(1.0) + det_cos(PI * u))
            var eff_w = safe_strength * w
            delta_sum[f] = delta_sum[f] + (target - p) * eff_w
            weight_sum[f] += eff_w

    for f in range(n):
        var ws = weight_sum[f]
        if ws > Float32(1.0e-6):
            var d = delta_sum[f]
            if ws > Float32(1.0):
                d = d * (Float32(1.0) / ws)
            out[f] = positions[f] + d

    return out^


def solve_two_bone_leg_cleanup(
    hip: Vec3, knee: Vec3, ankle: Vec3, target_ankle: Vec3
) -> Tuple[Vec3, Vec3]:
    """Exact planar law-of-cosines 2-bone IK preserving thigh and shin lengths.

    Moves `ankle` toward `target_ankle` (clamped strictly to `[|l1 - l2| + eps, l1 + l2 - eps]`)
    and solves the updated `knee` in the original bend plane so locking a foot or palm
    never stretches the upper or lower limb bones.

    Returns `(new_knee, new_ankle)`.
    """
    if not (
        _vec3_finite(hip)
        and _vec3_finite(knee)
        and _vec3_finite(ankle)
        and _vec3_finite(target_ankle)
    ):
        return (knee, ankle)

    var v_thigh = knee - hip
    var v_shin = ankle - knee
    var l1 = v_thigh.length()
    var l2 = v_shin.length()
    var eps = Float32(1.0e-4)
    if l1 < eps or l2 < eps:
        return (knee, ankle)

    var orig_hip_to_ankle = ankle - hip
    var orig_dir = safe_normalized(orig_hip_to_ankle, Vec3(0.0, -1.0, 0.0))
    var to_target = target_ankle - hip
    var d_raw = to_target.length()
    var u = safe_normalized(to_target, orig_dir)

    var d_min = abs(l1 - l2) + eps
    var d_max = (l1 + l2) - eps
    var d: Float32
    if d_max <= d_min:
        d = Float32(0.5) * (l1 + l2)
    else:
        d = max(d_min, min(d_max, d_raw))
    var new_ankle = hip + u * d

    # Preserve original anatomical knee/elbow bend half-plane orthogonal to `u`
    var pole = v_thigh - u * dot(v_thigh, u)
    if pole.length_squared() < Float32(1.0e-8):
        var orig_perp = v_thigh - orig_dir * dot(v_thigh, orig_dir)
        if orig_perp.length_squared() >= Float32(1.0e-8):
            pole = orig_perp - u * dot(orig_perp, u)
    if pole.length_squared() < Float32(1.0e-8):
        var fwd = Vec3(0.0, 0.0, 1.0)
        pole = fwd - u * dot(fwd, u)
    var pole_u = safe_normalized(pole, ortho_fallback(u))

    # Law of cosines: l2^2 = l1^2 + d^2 - 2*l1*d*cos(alpha) => along = l1*cos(alpha)
    var along = (l1 * l1 + d * d - l2 * l2) / (Float32(2.0) * d)
    along = max(-l1, min(l1, along))
    var perp = sqrt(max(Float32(0.0), l1 * l1 - along * along))
    var new_knee = hip + u * along + pole_u * perp

    if not (_vec3_finite(new_knee) and _vec3_finite(new_ankle)):
        return (knee, ankle)
    return (new_knee, new_ankle)


# ---------------------------------------------------------------------------
# 3. Full MocapTake Cleanup Pipeline (`flint.ml.rgbmocap` bridge)
# ---------------------------------------------------------------------------


def _resolve_take_joint(
    take: MocapTake, name_a: String, name_b: String, soma77_fallback: Int
) -> Int:
    var idx = take.find_joint(name_a)
    if idx >= 0:
        return idx
    if name_b.byte_length() > 0:
        idx = take.find_joint(name_b)
        if idx >= 0:
            return idx
    if take.num_joints() == 77 and soma77_fallback >= 0 and soma77_fallback < 77:
        return soma77_fallback
    return -1


def _resolve_two_bone_chain(
    take: MocapTake,
    tip_idx: Int,
    mid_fallback: Int,
    upper_fallback: Int,
) -> Tuple[Int, Int]:
    """Resolves `(upper, mid)` parent indices for a 2-bone limb ending at `tip_idx`."""
    var nj = take.num_joints()
    var mid = -1
    var upper = -1
    if tip_idx >= 0 and tip_idx < len(take.parent_indices):
        mid = take.parent_indices[tip_idx]
    if mid >= 0 and mid < len(take.parent_indices):
        upper = take.parent_indices[mid]
    if mid < 0 or mid >= nj:
        mid = mid_fallback if (mid_fallback >= 0 and mid_fallback < nj) else -1
    if upper < 0 or upper >= nj:
        upper = upper_fallback if (upper_fallback >= 0 and upper_fallback < nj) else -1
    return (upper, mid)


def _extract_joint_track(take: MocapTake, joint_idx: Int) -> List[Vec3]:
    var n = len(take.frames)
    var out = List[Vec3](capacity=n)
    if joint_idx < 0:
        return out^
    for f in range(n):
        if joint_idx < len(take.frames[f].joint_positions):
            out.append(take.frames[f].joint_positions[joint_idx])
        else:
            out.append(Vec3())
    return out^


def _is_descendant(parents: List[Int], joint: Int, ancestor: Int) -> Bool:
    if joint == ancestor or joint < 0 or ancestor < 0:
        return False
    var cur = joint
    var guard = 0
    while cur >= 0 and cur < len(parents) and guard < 128:
        var p = parents[cur]
        if p == ancestor:
            return True
        if p == cur:
            return False
        cur = p
        guard += 1
    return False


def _compute_world_quats(
    parents: List[Int], local_quats: List[Quat]
) -> List[Quat]:
    var nj = len(local_quats)
    var world = List[Quat](length=nj, fill=Quat.identity())
    var computed = List[Bool](length=nj, fill=False)
    for j in range(nj):
        if computed[j]:
            continue
        # Walk up to nearest computed ancestor or root
        var chain = List[Int]()
        var cur = j
        var guard = 0
        while cur >= 0 and cur < nj and (not computed[cur]) and guard < 128:
            chain.append(cur)
            var p = parents[cur] if cur < len(parents) else -1
            if p == cur:
                break
            cur = p
            guard += 1
        var acc = world[cur] if (cur >= 0 and cur < nj and computed[cur]) else Quat.identity()
        for k in range(len(chain) - 1, -1, -1):
            var idx = chain[k]
            var lq = local_quats[idx].normalized() if _quat_finite(local_quats[idx]) else Quat.identity()
            acc = acc.multiply(lq).normalized()
            world[idx] = acc
            computed[idx] = True
    return world^


def _apply_limb_ik_frame(
    mut take: MocapTake,
    frame_idx: Int,
    upper_idx: Int,
    mid_idx: Int,
    tip_idx: Int,
    target_tip: Vec3,
    descendants: List[Int],
):
    """Applies 2-bone positional IK + local rotation update + child offset for one limb."""
    var nj_pos = len(take.frames[frame_idx].joint_positions)
    if (
        upper_idx < 0
        or mid_idx < 0
        or tip_idx < 0
        or upper_idx >= nj_pos
        or mid_idx >= nj_pos
        or tip_idx >= nj_pos
    ):
        return

    var hip = take.frames[frame_idx].joint_positions[upper_idx]
    var knee = take.frames[frame_idx].joint_positions[mid_idx]
    var ankle = take.frames[frame_idx].joint_positions[tip_idx]
    if (target_tip - ankle).length_squared() <= Float32(1.0e-10):
        return

    var solved = solve_two_bone_leg_cleanup(hip, knee, ankle, target_tip)
    var new_knee = solved[0]
    var new_ankle = solved[1]
    var tip_delta = new_ankle - ankle

    take.frames[frame_idx].joint_positions[mid_idx] = new_knee
    take.frames[frame_idx].joint_positions[tip_idx] = new_ankle
    for d in range(len(descendants)):
        var dj = descendants[d]
        if dj >= 0 and dj < nj_pos:
            take.frames[frame_idx].joint_positions[dj] = (
                take.frames[frame_idx].joint_positions[dj] + tip_delta
            )

    # Keep local joint rotations consistent for BVH & NativeAnimation export
    var nj_rot = len(take.frames[frame_idx].joint_rotations)
    if upper_idx < nj_rot and mid_idx < nj_rot:
        var old_thigh = knee - hip
        var new_thigh = new_knee - hip
        var old_shin = ankle - knee
        var new_shin = new_ankle - new_knee
        if (
            old_thigh.length_squared() > Float32(1.0e-8)
            and new_thigh.length_squared() > Float32(1.0e-8)
            and old_shin.length_squared() > Float32(1.0e-8)
            and new_shin.length_squared() > Float32(1.0e-8)
        ):
            var wq = _compute_world_quats(
                take.parent_indices, take.frames[frame_idx].joint_rotations
            )
            var p_upper = (
                take.parent_indices[upper_idx]
                if upper_idx < len(take.parent_indices)
                else -1
            )
            var w_parent = (
                wq[p_upper] if (p_upper >= 0 and p_upper < len(wq)) else Quat.identity()
            )
            var dq_upper = shortest_arc_quat(old_thigh, new_thigh)
            var w_upper_new = dq_upper.multiply(wq[upper_idx]).normalized()
            take.frames[frame_idx].joint_rotations[upper_idx] = (
                w_parent.conjugate().multiply(w_upper_new).normalized()
            )

            var mid_shin = dq_upper.rotate(old_shin)
            var dq_mid = shortest_arc_quat(mid_shin, new_shin)
            var w_mid_mid = dq_upper.multiply(wq[mid_idx]).normalized()
            var w_mid_new = dq_mid.multiply(w_mid_mid).normalized()
            take.frames[frame_idx].joint_rotations[mid_idx] = (
                w_upper_new.conjugate().multiply(w_mid_new).normalized()
            )


def _measure_take_mean_jitter(take: MocapTake) -> Float32:
    var n = len(take.frames)
    var nj = take.num_joints()
    if n < 3 or nj == 0:
        return Float32(0.0)
    var sum_j = Float32(0.0)
    var counted = 0
    for j in range(nj):
        var track = _extract_joint_track(take, j)
        if len(track) >= 3:
            sum_j += measure_series_jitter(track)
            counted += 1
    if counted == 0:
        return Float32(0.0)
    return sum_j / Float32(counted)


def _combined_foot_slide_cm(
    left_pos: List[Vec3],
    left_ivs: List[ContactInterval],
    right_pos: List[Vec3],
    right_ivs: List[ContactInterval],
) -> Float32:
    var nl = len(left_ivs)
    var nr = len(right_ivs)
    var total_n = nl + nr
    if total_n == 0:
        return Float32(0.0)
    var sl = measure_foot_slide_cm(left_pos, left_ivs) if nl > 0 else Float32(0.0)
    var sr = measure_foot_slide_cm(right_pos, right_ivs) if nr > 0 else Float32(0.0)
    return (sl * Float32(nl) + sr * Float32(nr)) / Float32(total_n)


def _filter_vec3_zero_phase(
    series: List[Vec3], fps: Float32, cfg: OneEuroConfig
) -> List[Vec3]:
    """Zero-phase (forward + backward) One Euro pass for offline take cleanup (zero phase lag)."""
    var n = len(series)
    if n < 2:
        return filter_vec3_series(series, fps, cfg)
    var fwd = filter_vec3_series(series, fps, cfg)
    var rev_in = List[Vec3](capacity=n)
    for i in range(n - 1, -1, -1):
        rev_in.append(series[i])
    var rev_out = filter_vec3_series(rev_in, fps, cfg)
    var out = List[Vec3](capacity=n)
    for i in range(n):
        out.append((fwd[i] + rev_out[n - 1 - i]) * Float32(0.5))
    return out^


def _filter_quat_zero_phase(
    series: List[Quat], fps: Float32, cfg: OneEuroConfig
) -> List[Quat]:
    """Zero-phase (forward + backward) hemisphere-aligned SLERP One Euro pass."""
    var n = len(series)
    if n < 2:
        return filter_quat_series(series, fps, cfg)
    var fwd = filter_quat_series(series, fps, cfg)
    var rev_in = List[Quat](capacity=n)
    for i in range(n - 1, -1, -1):
        rev_in.append(series[i])
    var rev_out = filter_quat_series(rev_in, fps, cfg)
    var out = List[Quat](capacity=n)
    for i in range(n):
        out.append(fwd[i].slerp(rev_out[n - 1 - i], Float32(0.5)).normalized())
    return out^


def _required_pelvis_drop(
    hip: Vec3, knee: Vec3, ankle: Vec3, target_ankle: Vec3
) -> Float32:
    """Computes minimum downward pelvis shift (>= 0) so `target_ankle` is within leg reach."""
    var l1 = (knee - hip).length()
    var l2 = (ankle - knee).length()
    var max_reach = l1 + l2 - Float32(0.002)
    if max_reach <= Float32(0.05):
        return Float32(0.0)
    var diff = target_ankle - hip
    var dist = diff.length()
    if dist <= max_reach:
        return Float32(0.0)
    var horiz_sq = diff.x * diff.x + diff.z * diff.z
    var reach_sq = max_reach * max_reach
    if horiz_sq >= reach_sq:
        return min(Float32(0.14), max(Float32(0.0), hip.y - target_ankle.y))
    var max_dy = sqrt(reach_sq - horiz_sq)
    var cur_dy = hip.y - target_ankle.y
    if cur_dy <= max_dy:
        return Float32(0.0)
    return min(Float32(0.14), cur_dy - max_dy)


def _build_lag_free_contact_track(
    raw_pos: List[Vec3],
    smooth_pos: List[Vec3],
    intervals: List[ContactInterval],
    fps: Float32,
    cfg: OneEuroConfig,
    use_one_euro: Bool,
    ground_y: Float32,
    strength: Float32,
    fade_frames: Int,
    clamp_y_to_ground: Bool,
) -> List[Vec3]:
    """Combines smooth swing motion with lag-free, contact-locked stance intervals.

    Preventing One Euro smoothing from smearing fast swing deceleration into a newly
    planted stance interval ensures heel-strike stays pinned at the true anchor while
    transitions remain C1-smooth via cosine fade windows.
    """
    var n = len(raw_pos)
    var out = List[Vec3](capacity=n)
    for i in range(n):
        out.append(smooth_pos[i] if i < len(smooth_pos) else raw_pos[i])
    if n == 0 or len(intervals) == 0:
        return out^

    var fade = max(0, fade_frames)
    var base = List[Vec3](capacity=n)
    for i in range(n):
        base.append(out[i])

    var in_stance = List[Bool](length=n, fill=False)
    for idx in range(len(intervals)):
        var s = max(0, intervals[idx].start_frame)
        var e = min(n - 1, intervals[idx].end_frame)
        if e < s:
            continue
        var seg = List[Vec3](capacity=e - s + 1)
        for f in range(s, e + 1):
            in_stance[f] = True
            seg.append(raw_pos[f])
        var seg_clean = _filter_vec3_zero_phase(seg, fps, cfg) if use_one_euro else seg^
        for f in range(s, e + 1):
            base[f] = seg_clean[f - s]

    # Smoothly cosine-blend fade windows around each stance interval into the smooth swing
    if fade > 0:
        for idx in range(len(intervals)):
            var s = max(0, intervals[idx].start_frame)
            var e = min(n - 1, intervals[idx].end_frame)
            if e < s:
                continue
            for d in range(1, fade + 1):
                var f_pre = s - d
                if f_pre >= 0 and not in_stance[f_pre]:
                    var u = Float32(d) / Float32(fade + 1)
                    var w = Float32(0.5) * (Float32(1.0) + det_cos(PI * u))
                    base[f_pre] = out[f_pre] + (base[s] - out[f_pre]) * w * Float32(0.5)
                var f_post = e + d
                if f_post < n and not in_stance[f_post]:
                    var u = Float32(d) / Float32(fade + 1)
                    var w = Float32(0.5) * (Float32(1.0) + det_cos(PI * u))
                    base[f_post] = out[f_post] + (base[e] - out[f_post]) * w * Float32(0.5)

    return lock_contact_positions(
        base, intervals, ground_y, strength, fade, clamp_y_to_ground
    )


def apply_mocap_cleanup_to_take(
    mut take: MocapTake, cfg: ContactCleanupConfig
) -> MocapCleanupMetrics:
    """Applies One Euro smoothing, Foot/Palm contact locking (with 2-bone IK), and ground alignment.

    Mutates `take` in place (positions, rotations, contact flags, and slide telemetry)
    and returns full before/after `MocapCleanupMetrics`.
    """
    var metrics = MocapCleanupMetrics()
    var n_frames = len(take.frames)
    var nj = take.num_joints()
    if n_frames == 0 or nj == 0:
        return metrics

    var fps = take.fps if (isfinite(take.fps) and take.fps > Float32(1.0e-4)) else Float32(30.0)

    # Resolve foot & palm chains (works for SOMA-77 and standard named skeletons)
    var l_foot_idx = _resolve_take_joint(take, "LeftFoot", "left_ankle", SOMA77_LEFT_FOOT)
    var r_foot_idx = _resolve_take_joint(take, "RightFoot", "right_ankle", SOMA77_RIGHT_FOOT)
    var l_hand_idx = _resolve_take_joint(take, "LeftHand", "left_wrist", SOMA77_LEFT_HAND)
    var r_hand_idx = _resolve_take_joint(take, "RightHand", "right_wrist", SOMA77_RIGHT_HAND)

    var l_leg_chain = _resolve_two_bone_chain(take, l_foot_idx, SOMA77_LEFT_SHIN, SOMA77_LEFT_LEG)
    var r_leg_chain = _resolve_two_bone_chain(take, r_foot_idx, SOMA77_RIGHT_SHIN, SOMA77_RIGHT_LEG)
    var l_arm_chain = _resolve_two_bone_chain(take, l_hand_idx, SOMA77_LEFT_FOREARM, SOMA77_LEFT_ARM)
    var r_arm_chain = _resolve_two_bone_chain(take, r_hand_idx, SOMA77_RIGHT_FOREARM, SOMA77_RIGHT_ARM)

    # Precompute descendant joint indices for feet (toes) and hands (fingers)
    var l_foot_desc = List[Int]()
    var r_foot_desc = List[Int]()
    var l_hand_desc = List[Int]()
    var r_hand_desc = List[Int]()
    for j in range(nj):
        if _is_descendant(take.parent_indices, j, l_foot_idx):
            l_foot_desc.append(j)
        if _is_descendant(take.parent_indices, j, r_foot_idx):
            r_foot_desc.append(j)
        if _is_descendant(take.parent_indices, j, l_hand_idx):
            l_hand_desc.append(j)
        if _is_descendant(take.parent_indices, j, r_hand_idx):
            r_hand_desc.append(j)

    # 0. Capture before metrics and detect unlagged stance/palm contacts on raw tracks
    metrics.mean_jitter_before = _measure_take_mean_jitter(take)
    var l_foot_raw = _extract_joint_track(take, l_foot_idx)
    var r_foot_raw = _extract_joint_track(take, r_foot_idx)
    var l_hand_raw = _extract_joint_track(take, l_hand_idx)
    var r_hand_raw = _extract_joint_track(take, r_hand_idx)
    var raw_est_ground = estimate_ground_plane_y(l_foot_raw, r_foot_raw)
    var raw_ground = raw_est_ground if cfg.auto_estimate_ground else cfg.ground_y
    metrics.estimated_ground_y = raw_est_ground

    var ref_l_ivs = detect_contact_intervals(
        l_foot_raw,
        fps,
        raw_ground,
        cfg.foot_height_thresh_m,
        cfg.foot_speed_thresh_mps,
        cfg.min_contact_frames,
        True,
        False,
    )
    var ref_r_ivs = detect_contact_intervals(
        r_foot_raw,
        fps,
        raw_ground,
        cfg.foot_height_thresh_m,
        cfg.foot_speed_thresh_mps,
        cfg.min_contact_frames,
        False,
        False,
    )

    # 1. Zero-phase One Euro Filtering (root + all joint positions + hemisphere-checked rotations)
    if cfg.enable_one_euro:
        var root_raw = List[Vec3](capacity=n_frames)
        for f in range(n_frames):
            root_raw.append(take.frames[f].root_pos)
        var root_smooth = _filter_vec3_zero_phase(root_raw, fps, cfg.one_euro)
        for f in range(n_frames):
            take.frames[f].root_pos = root_smooth[f]

        for j in range(nj):
            var pos_series = List[Vec3](capacity=n_frames)
            var has_pos = True
            for f in range(n_frames):
                if j < len(take.frames[f].joint_positions):
                    pos_series.append(take.frames[f].joint_positions[j])
                else:
                    has_pos = False
                    break
            if has_pos:
                var pos_smooth = _filter_vec3_zero_phase(pos_series, fps, cfg.one_euro)
                for f in range(n_frames):
                    take.frames[f].joint_positions[j] = pos_smooth[f]

            var rot_series = List[Quat](capacity=n_frames)
            var has_rot = True
            for f in range(n_frames):
                if j < len(take.frames[f].joint_rotations):
                    rot_series.append(take.frames[f].joint_rotations[j])
                else:
                    has_rot = False
                    break
            if has_rot:
                var rot_smooth = _filter_quat_zero_phase(rot_series, fps, cfg.one_euro)
                for f in range(n_frames):
                    take.frames[f].joint_rotations[j] = rot_smooth[f]

    # 2. Foot Contact Detection & Anti-Sliding Lock + 2-Bone Leg IK
    var l_foot_cur = _extract_joint_track(take, l_foot_idx)
    var r_foot_cur = _extract_joint_track(take, r_foot_idx)
    var cur_est_ground = raw_est_ground if len(ref_l_ivs) + len(ref_r_ivs) > 0 else estimate_ground_plane_y(l_foot_cur, r_foot_cur)
    var cur_ground = cur_est_ground if cfg.auto_estimate_ground else cfg.ground_y
    metrics.estimated_ground_y = cur_est_ground

    # Prefer unlagged raw contact intervals so One Euro swing lag never truncates heel-strike;
    # fall back to filtered-track detection if raw tracks were too noisy to detect any stance.
    var active_l_ivs: List[ContactInterval]
    var active_r_ivs: List[ContactInterval]
    if len(ref_l_ivs) + len(ref_r_ivs) > 0:
        active_l_ivs = ref_l_ivs.copy()
        active_r_ivs = ref_r_ivs.copy()
    else:
        active_l_ivs = detect_contact_intervals(
            l_foot_cur,
            fps,
            cur_ground,
            cfg.foot_height_thresh_m,
            cfg.foot_speed_thresh_mps,
            cfg.min_contact_frames,
            True,
            False,
        )
        active_r_ivs = detect_contact_intervals(
            r_foot_cur,
            fps,
            cur_ground,
            cfg.foot_height_thresh_m,
            cfg.foot_speed_thresh_mps,
            cfg.min_contact_frames,
            False,
            False,
        )

    metrics.foot_slide_before_cm = _combined_foot_slide_cm(
        l_foot_raw, active_l_ivs, r_foot_raw, active_r_ivs
    )

    if cfg.enable_foot_lock and l_foot_idx >= 0 and r_foot_idx >= 0:
        metrics.foot_contacts_detected = len(active_l_ivs) + len(active_r_ivs)
        var l_locked = _build_lag_free_contact_track(
            l_foot_raw,
            l_foot_cur,
            active_l_ivs,
            fps,
            cfg.one_euro,
            cfg.enable_one_euro,
            cur_ground,
            cfg.foot_lock_strength,
            cfg.contact_fade_frames,
            True,
        )
        var r_locked = _build_lag_free_contact_track(
            r_foot_raw,
            r_foot_cur,
            active_r_ivs,
            fps,
            cfg.one_euro,
            cfg.enable_one_euro,
            cur_ground,
            cfg.foot_lock_strength,
            cfg.contact_fade_frames,
            True,
        )

        # Refresh per-frame contact flags from detected intervals
        for f in range(n_frames):
            take.frames[f].left_foot_contact = False
            take.frames[f].right_foot_contact = False
        for i in range(len(active_l_ivs)):
            var s = max(0, active_l_ivs[i].start_frame)
            var e = min(n_frames - 1, active_l_ivs[i].end_frame)
            for f in range(s, e + 1):
                take.frames[f].left_foot_contact = True
        for i in range(len(active_r_ivs)):
            var s = max(0, active_r_ivs[i].start_frame)
            var e = min(n_frames - 1, active_r_ivs[i].end_frame)
            for f in range(s, e + 1):
                take.frames[f].right_foot_contact = True

        # Lower pelvis minimally when a planted foot target exceeds full leg extension,
        # smoothed smoothly into/out of contact transitions, then propagate locked ankles
        # cleanly through hip->knee->ankle 2-bone IK.
        var pelvis_drops = List[Float32](length=n_frames, fill=Float32(0.0))
        for f in range(n_frames):
            var nj_pos = len(take.frames[f].joint_positions)
            var drop = Float32(0.0)
            if (
                take.frames[f].left_foot_contact
                and l_leg_chain[0] >= 0
                and l_leg_chain[1] >= 0
                and l_foot_idx >= 0
                and l_leg_chain[0] < nj_pos
                and l_leg_chain[1] < nj_pos
                and l_foot_idx < nj_pos
            ):
                drop = max(
                    drop,
                    _required_pelvis_drop(
                        take.frames[f].joint_positions[l_leg_chain[0]],
                        take.frames[f].joint_positions[l_leg_chain[1]],
                        take.frames[f].joint_positions[l_foot_idx],
                        l_locked[f],
                    ),
                )
            if (
                take.frames[f].right_foot_contact
                and r_leg_chain[0] >= 0
                and r_leg_chain[1] >= 0
                and r_foot_idx >= 0
                and r_leg_chain[0] < nj_pos
                and r_leg_chain[1] < nj_pos
                and r_foot_idx < nj_pos
            ):
                drop = max(
                    drop,
                    _required_pelvis_drop(
                        take.frames[f].joint_positions[r_leg_chain[0]],
                        take.frames[f].joint_positions[r_leg_chain[1]],
                        take.frames[f].joint_positions[r_foot_idx],
                        r_locked[f],
                    ),
                )
            pelvis_drops[f] = drop

        # Smooth pelvis-drop envelope without ever dropping below required reach
        if n_frames >= 3:
            for _ in range(2):
                var next_drops = pelvis_drops.copy()
                for f in range(n_frames):
                    var prev_d = pelvis_drops[f - 1] if f > 0 else pelvis_drops[f]
                    var next_d = pelvis_drops[f + 1] if f + 1 < n_frames else pelvis_drops[f]
                    var env = Float32(0.25) * prev_d + Float32(0.5) * pelvis_drops[f] + Float32(0.25) * next_d
                    next_drops[f] = max(pelvis_drops[f], env)
                pelvis_drops = next_drops^

        for f in range(n_frames):
            var nj_pos = len(take.frames[f].joint_positions)
            var drop = pelvis_drops[f]
            if drop > Float32(1.0e-5):
                take.frames[f].root_pos.y -= drop
                for j in range(nj_pos):
                    take.frames[f].joint_positions[j].y -= drop

            _apply_limb_ik_frame(
                take,
                f,
                l_leg_chain[0],
                l_leg_chain[1],
                l_foot_idx,
                l_locked[f],
                l_foot_desc,
            )
            _apply_limb_ik_frame(
                take,
                f,
                r_leg_chain[0],
                r_leg_chain[1],
                r_foot_idx,
                r_locked[f],
                r_foot_desc,
            )

    # 3. Optional Palm Contact Detection & Lock (speed-only so elevated props/walls work)
    if cfg.enable_palm_lock and l_hand_idx >= 0 and r_hand_idx >= 0:
        var l_hand_cur = _extract_joint_track(take, l_hand_idx)
        var r_hand_cur = _extract_joint_track(take, r_hand_idx)
        var l_palm_ivs = detect_contact_intervals(
            l_hand_raw,
            fps,
            cur_ground,
            Float32(0.0),
            cfg.palm_speed_thresh_mps,
            cfg.min_contact_frames,
            True,
            True,
        )
        var r_palm_ivs = detect_contact_intervals(
            r_hand_raw,
            fps,
            cur_ground,
            Float32(0.0),
            cfg.palm_speed_thresh_mps,
            cfg.min_contact_frames,
            False,
            True,
        )
        if len(l_palm_ivs) + len(r_palm_ivs) == 0:
            l_palm_ivs = detect_contact_intervals(
                l_hand_cur,
                fps,
                cur_ground,
                Float32(0.0),
                cfg.palm_speed_thresh_mps,
                cfg.min_contact_frames,
                True,
                True,
            )
            r_palm_ivs = detect_contact_intervals(
                r_hand_cur,
                fps,
                cur_ground,
                Float32(0.0),
                cfg.palm_speed_thresh_mps,
                cfg.min_contact_frames,
                False,
                True,
            )
        metrics.palm_contacts_detected = len(l_palm_ivs) + len(r_palm_ivs)

        var l_hand_locked = _build_lag_free_contact_track(
            l_hand_raw,
            l_hand_cur,
            l_palm_ivs,
            fps,
            cfg.one_euro,
            cfg.enable_one_euro,
            cur_ground,
            cfg.palm_lock_strength,
            cfg.contact_fade_frames,
            False,
        )
        var r_hand_locked = _build_lag_free_contact_track(
            r_hand_raw,
            r_hand_cur,
            r_palm_ivs,
            fps,
            cfg.one_euro,
            cfg.enable_one_euro,
            cur_ground,
            cfg.palm_lock_strength,
            cfg.contact_fade_frames,
            False,
        )

        for f in range(n_frames):
            take.frames[f].left_palm_contact = False
            take.frames[f].right_palm_contact = False
        for i in range(len(l_palm_ivs)):
            var s = max(0, l_palm_ivs[i].start_frame)
            var e = min(n_frames - 1, l_palm_ivs[i].end_frame)
            for f in range(s, e + 1):
                take.frames[f].left_palm_contact = True
        for i in range(len(r_palm_ivs)):
            var s = max(0, r_palm_ivs[i].start_frame)
            var e = min(n_frames - 1, r_palm_ivs[i].end_frame)
            for f in range(s, e + 1):
                take.frames[f].right_palm_contact = True

        for f in range(n_frames):
            _apply_limb_ik_frame(
                take,
                f,
                l_arm_chain[0],
                l_arm_chain[1],
                l_hand_idx,
                l_hand_locked[f],
                l_hand_desc,
            )
            _apply_limb_ik_frame(
                take,
                f,
                r_arm_chain[0],
                r_arm_chain[1],
                r_hand_idx,
                r_hand_locked[f],
                r_hand_desc,
            )

    # 4. Ground Plane Vertical Alignment
    if cfg.enable_ground_align:
        var target_ground = cfg.ground_y if isfinite(cfg.ground_y) else Float32(0.0)
        var y_shift = target_ground - cur_est_ground
        if isfinite(y_shift) and abs(y_shift) > Float32(1.0e-5):
            for f in range(n_frames):
                take.frames[f].root_pos.y += y_shift
                take.frames[f].camera.pos.y += y_shift
                for j in range(len(take.frames[f].joint_positions)):
                    take.frames[f].joint_positions[j].y += y_shift

    # 5. Capture after metrics & mirror slide telemetry onto `take`
    var l_foot_final = _extract_joint_track(take, l_foot_idx)
    var r_foot_final = _extract_joint_track(take, r_foot_idx)
    metrics.foot_slide_after_cm = _combined_foot_slide_cm(
        l_foot_final, active_l_ivs, r_foot_final, active_r_ivs
    )
    metrics.mean_jitter_after = _measure_take_mean_jitter(take)
    take.foot_slide_before_cm = metrics.foot_slide_before_cm
    take.foot_slide_after_cm = metrics.foot_slide_after_cm
    return metrics
