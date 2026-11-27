"""3D Z-Score Normalized + SIMD8 SoA + 64-Frame AABB Chunked Motion Matching (`flint.anim.motion_matching`).

Upgrades in Flint v4 (100% backward-compatible with `NeonDrift` & v3 callers):
1. Extended `MotionFeature` with 3D `hip_vel`, `ang_vel`, and discrete `stance_tag`
   (`0=ground`, `1=hover/gerwalk`, `2=air/space`, `3=melee`, `-1=any`).
2. `finalize_and_normalize`: computes per-channel mean `mu_d` & std `sigma_d`
   (z-score standardization), encodes scalar trajectory facing angles as unit 2D
   `(cos, sin)` pairs (eliminating `floor` wraps and +-pi discontinuities), packs
   transposed 8-wide `SIMD[DType.float32, 8]` SoA feature columns, and builds
   64-frame AABB `(min, max)` chunk headers for exact branch-and-bound pruning.
3. Stateful `query_inertial` preventing same-frame micro-freeze (`exclude_window`)
   and ping-pong clip thrashing (`continuing_bias`).
4. `bake_clip_features` automatic indexer extracting local-frame trajectory, foot,
   and root/hip linear & angular velocity features from `NativeModel` animations.
"""
from std.collections import List
from std.math import sqrt, min, max, abs, floor, isfinite
from flint.core.detmath import det_sin, det_cos, det_atan2, PI
from flint.core.math import Vec3
from flint.assets.model import NativeModel
from flint.anim.pose import Pose, SkeletonInstance, sample_pose, det_quat_to_scaled_angle

comptime STANCE_ANY: Int = -1
comptime STANCE_GROUND: Int = 0
comptime STANCE_HOVER_GERWALK: Int = 1
comptime STANCE_AIR_SPACE: Int = 2
comptime STANCE_MELEE: Int = 3

comptime MM_LANES: Int = 8
comptime MM_CHUNK_SIZE: Int = 64
comptime MM_DIM: Int = 33
comptime MM_F8 = SIMD[DType.float32, MM_LANES]


@always_inline
def _wrap_angle_diff(a: Float32, b: Float32) -> Float32:
    comptime TAU_F32: Float32 = 6.28318530717958647692
    var d = a - b
    return abs(d - floor((d + PI) / TAU_F32) * TAU_F32)


@always_inline
def _wrap_signed_angle(a: Float32) -> Float32:
    comptime TAU_F32: Float32 = 6.28318530717958647692
    return a - floor((a + PI) / TAU_F32) * TAU_F32


struct MotionFeature(Copyable, ImplicitlyCopyable, Movable):
    var traj_pos_200: Vec3
    var traj_pos_400: Vec3
    var traj_pos_800: Vec3
    var traj_facing_200: Float32
    var traj_facing_400: Float32
    var traj_facing_800: Float32
    var left_foot_pos: Vec3
    var left_foot_vel: Vec3
    var right_foot_pos: Vec3
    var right_foot_vel: Vec3
    var hip_vel: Vec3
    var ang_vel: Vec3
    var stance_tag: Int

    def __init__(out self):
        self.traj_pos_200 = Vec3(0, 0, 0.2)
        self.traj_pos_400 = Vec3(0, 0, 0.4)
        self.traj_pos_800 = Vec3(0, 0, 0.8)
        self.traj_facing_200 = 0.0
        self.traj_facing_400 = 0.0
        self.traj_facing_800 = 0.0
        self.left_foot_pos = Vec3(-0.15, 0.0, 0.0)
        self.left_foot_vel = Vec3()
        self.right_foot_pos = Vec3(0.15, 0.0, 0.0)
        self.right_foot_vel = Vec3()
        self.hip_vel = Vec3()
        self.ang_vel = Vec3()
        self.stance_tag = STANCE_GROUND


@always_inline
def _safe_mm_f32(v: Float32) -> Float32:
    return v if isfinite(v) else Float32(0.0)


@always_inline
def _safe_mm_vec3(v: Vec3) -> Vec3:
    return Vec3(_safe_mm_f32(v.x), _safe_mm_f32(v.y), _safe_mm_f32(v.z))


@always_inline
def _extract_raw_channels(f: MotionFeature, mut out: Array[Float32, MM_DIM]):
    """Flattens a `MotionFeature` into 33 continuous channels, converting facing
    angles into unit 2D `(cos, sin)` vectors."""
    out[0] = _safe_mm_f32(f.traj_pos_200.x)
    out[1] = _safe_mm_f32(f.traj_pos_200.y)
    out[2] = _safe_mm_f32(f.traj_pos_200.z)
    out[3] = _safe_mm_f32(f.traj_pos_400.x)
    out[4] = _safe_mm_f32(f.traj_pos_400.y)
    out[5] = _safe_mm_f32(f.traj_pos_400.z)
    out[6] = _safe_mm_f32(f.traj_pos_800.x)
    out[7] = _safe_mm_f32(f.traj_pos_800.y)
    out[8] = _safe_mm_f32(f.traj_pos_800.z)
    var f200 = _safe_mm_f32(f.traj_facing_200)
    var f400 = _safe_mm_f32(f.traj_facing_400)
    var f800 = _safe_mm_f32(f.traj_facing_800)
    out[9] = det_cos(f200)
    out[10] = det_sin(f200)
    out[11] = det_cos(f400)
    out[12] = det_sin(f400)
    out[13] = det_cos(f800)
    out[14] = det_sin(f800)
    out[15] = _safe_mm_f32(f.left_foot_pos.x)
    out[16] = _safe_mm_f32(f.left_foot_pos.y)
    out[17] = _safe_mm_f32(f.left_foot_pos.z)
    out[18] = _safe_mm_f32(f.left_foot_vel.x)
    out[19] = _safe_mm_f32(f.left_foot_vel.y)
    out[20] = _safe_mm_f32(f.left_foot_vel.z)
    out[21] = _safe_mm_f32(f.right_foot_pos.x)
    out[22] = _safe_mm_f32(f.right_foot_pos.y)
    out[23] = _safe_mm_f32(f.right_foot_pos.z)
    out[24] = _safe_mm_f32(f.right_foot_vel.x)
    out[25] = _safe_mm_f32(f.right_foot_vel.y)
    out[26] = _safe_mm_f32(f.right_foot_vel.z)
    out[27] = _safe_mm_f32(f.hip_vel.x)
    out[28] = _safe_mm_f32(f.hip_vel.y)
    out[29] = _safe_mm_f32(f.hip_vel.z)
    out[30] = _safe_mm_f32(f.ang_vel.x)
    out[31] = _safe_mm_f32(f.ang_vel.y)
    out[32] = _safe_mm_f32(f.ang_vel.z)


struct MotionFrame(Copyable, ImplicitlyCopyable, Movable):
    var clip_index: Int
    var frame_index: Int
    var time_seconds: Float32
    var feature: MotionFeature

    def __init__(out self, clip: Int = 0, frame: Int = 0, time_s: Float32 = 0.0, feature: MotionFeature = MotionFeature()):
        self.clip_index = clip
        self.frame_index = frame
        self.time_seconds = time_s
        self.feature = feature


struct MotionDatabase(Copyable, Movable):
    var frames: List[MotionFrame]
    var weight_pos: Float32
    var weight_facing: Float32
    var weight_feet: Float32
    var weight_vel: Float32
    var weight_ang_vel: Float32
    var stance_mismatch_cost: Float32

    # Finalized Z-Score + SIMD8 SoA + 64-Frame AABB acceleration structures
    var is_finalized: Bool
    var padded_count: Int
    var means: List[Float32]
    var stds: List[Float32]
    var scales: List[Float32]
    var soa_features: List[Float32]
    var chunk_mins: List[Float32]
    var chunk_maxs: List[Float32]
    var chunk_starts: List[Int]
    var chunk_ends: List[Int]
    var chunk_stance_masks: List[UInt32]

    def __init__(out self):
        self.frames = List[MotionFrame]()
        self.weight_pos = 1.0
        self.weight_facing = 0.65
        self.weight_feet = 0.85
        self.weight_vel = 0.75
        self.weight_ang_vel = 0.50
        self.stance_mismatch_cost = 100.0
        self.is_finalized = False
        self.padded_count = 0
        self.means = List[Float32]()
        self.stds = List[Float32]()
        self.scales = List[Float32]()
        self.soa_features = List[Float32]()
        self.chunk_mins = List[Float32]()
        self.chunk_maxs = List[Float32]()
        self.chunk_starts = List[Int]()
        self.chunk_ends = List[Int]()
        self.chunk_stance_masks = List[UInt32]()

    def add_frame(mut self, clip: Int, frame: Int, time_s: Float32, feature: MotionFeature):
        var clean = feature
        clean.traj_pos_200 = _safe_mm_vec3(clean.traj_pos_200)
        clean.traj_pos_400 = _safe_mm_vec3(clean.traj_pos_400)
        clean.traj_pos_800 = _safe_mm_vec3(clean.traj_pos_800)
        clean.traj_facing_200 = _safe_mm_f32(clean.traj_facing_200)
        clean.traj_facing_400 = _safe_mm_f32(clean.traj_facing_400)
        clean.traj_facing_800 = _safe_mm_f32(clean.traj_facing_800)
        clean.left_foot_pos = _safe_mm_vec3(clean.left_foot_pos)
        clean.left_foot_vel = _safe_mm_vec3(clean.left_foot_vel)
        clean.right_foot_pos = _safe_mm_vec3(clean.right_foot_pos)
        clean.right_foot_vel = _safe_mm_vec3(clean.right_foot_vel)
        clean.hip_vel = _safe_mm_vec3(clean.hip_vel)
        clean.ang_vel = _safe_mm_vec3(clean.ang_vel)
        self.frames.append(MotionFrame(clip, frame, _safe_mm_f32(time_s), clean))
        self.is_finalized = False

    def count(self) -> Int:
        return len(self.frames)

    def chunk_count(self) -> Int:
        return len(self.chunk_starts)

    def _channel_weights(self, mut w_out: Array[Float32, MM_DIM]):
        for d in range(9):
            w_out[d] = self.weight_pos
        for d in range(9, 15):
            w_out[d] = self.weight_facing
        for d in range(15, 18):
            w_out[d] = self.weight_feet
        for d in range(18, 21):
            w_out[d] = self.weight_feet * Float32(0.1)
        for d in range(21, 24):
            w_out[d] = self.weight_feet
        for d in range(24, 27):
            w_out[d] = self.weight_feet * Float32(0.1)
        for d in range(27, 30):
            w_out[d] = self.weight_vel
        for d in range(30, 33):
            w_out[d] = self.weight_ang_vel

    def finalize_and_normalize(mut self):
        """Computes per-channel mean `mu_d` & std `sigma_d` (z-score standardization),
        converts facing angles to unit 2D `(cos, sin)` pairs, packs transposed 8-wide
        `SIMD[DType.float32, 8]` SoA columns, and builds 64-frame AABB chunk headers."""
        var n = len(self.frames)
        if n == 0:
            self.is_finalized = False
            self.padded_count = 0
            self.chunk_starts.clear()
            self.chunk_ends.clear()
            return

        self.means = List[Float32](length=MM_DIM, fill=0.0)
        self.stds = List[Float32](length=MM_DIM, fill=1.0)
        self.scales = List[Float32](length=MM_DIM, fill=1.0)

        var raw_flat = List[Float32](length=n * MM_DIM, fill=0.0)
        var tmp = Array[Float32, MM_DIM](fill=0.0)
        var means64 = Array[Float64, MM_DIM](fill=0.0)

        for i in range(n):
            _extract_raw_channels(self.frames[i].feature, tmp)
            var base = i * MM_DIM
            for d in range(MM_DIM):
                var v = tmp[d] if isfinite(tmp[d]) else Float32(0.0)
                raw_flat[base + d] = v
                means64[d] += Float64(v)

        var inv_n64 = 1.0 / Float64(n)
        for d in range(MM_DIM):
            means64[d] *= inv_n64
            self.means[d] = Float32(means64[d]) if isfinite(means64[d]) else Float32(0.0)

        var vars_acc64 = Array[Float64, MM_DIM](fill=0.0)
        for i in range(n):
            var base = i * MM_DIM
            for d in range(MM_DIM):
                var diff64 = Float64(raw_flat[base + d]) - means64[d]
                vars_acc64[d] += diff64 * diff64
        var stds64 = Array[Float64, MM_DIM](fill=1.0)
        for d in range(MM_DIM):
            var s64 = sqrt(vars_acc64[d] * inv_n64)
            stds64[d] = s64 if isfinite(s64) else 1.0
            self.stds[d] = Float32(stds64[d])

        # Pool 3D vector group standard deviations so flat axes (e.g. y=0 on flat ground)
        # share their 3D feature group's physical scale instead of amplifying noise.
        var vec_groups: Array[Int, 9] = [0, 3, 6, 15, 18, 21, 24, 27, 30]
        for g_idx in range(9):
            var d0 = vec_groups[g_idx]
            var rms64 = sqrt(
                (stds64[d0] * stds64[d0] + stds64[d0 + 1] * stds64[d0 + 1] + stds64[d0 + 2] * stds64[d0 + 2])
                * (1.0 / 3.0)
            )
            var safe_s64 = max(0.10, rms64 if isfinite(rms64) else 0.10)
            stds64[d0] = max(stds64[d0], safe_s64)
            stds64[d0 + 1] = max(stds64[d0 + 1], safe_s64)
            stds64[d0 + 2] = max(stds64[d0 + 2], safe_s64)
            self.stds[d0] = Float32(stds64[d0])
            self.stds[d0 + 1] = Float32(stds64[d0 + 1])
            self.stds[d0 + 2] = Float32(stds64[d0 + 2])

        # Unit 2D `(cos, sin)` pairs are already dimensionless on S^1; floor std at 0.35
        for p_idx in range(3):
            var dc = 9 + p_idx * 2
            var pair_rms64 = sqrt((stds64[dc] * stds64[dc] + stds64[dc + 1] * stds64[dc + 1]) * 0.5)
            var safe_cs64 = max(0.35, pair_rms64 if isfinite(pair_rms64) else 0.35)
            stds64[dc] = max(stds64[dc], safe_cs64)
            stds64[dc + 1] = max(stds64[dc + 1], safe_cs64)
            self.stds[dc] = Float32(stds64[dc])
            self.stds[dc + 1] = Float32(stds64[dc + 1])

        var w_chan = Array[Float32, MM_DIM](fill=1.0)
        self._channel_weights(w_chan)
        for d in range(MM_DIM):
            var sc64 = sqrt(max(0.0, Float64(w_chan[d]))) / max(0.05, stds64[d])
            self.scales[d] = Float32(sc64) if isfinite(sc64) else Float32(1.0)

        # Pack transposed 8-wide SIMD SoA layout: `[num_groups][MM_DIM][8]`
        var num_groups = (n + MM_LANES - 1) // MM_LANES
        self.padded_count = num_groups * MM_LANES
        self.soa_features = List[Float32](length=num_groups * MM_DIM * MM_LANES, fill=0.0)

        for g in range(num_groups):
            var g_frame0 = g * MM_LANES
            var g_soa_base = g * MM_DIM * MM_LANES
            for lane in range(MM_LANES):
                var frame_i = g_frame0 + lane
                if frame_i < n:
                    var raw_base = frame_i * MM_DIM
                    for d in range(MM_DIM):
                        var z = (raw_flat[raw_base + d] - self.means[d]) * self.scales[d]
                        self.soa_features[g_soa_base + d * MM_LANES + lane] = z
                else:
                    # Replicate last valid frame in tail lanes (masked out during reduction)
                    var raw_base = (n - 1) * MM_DIM
                    for d in range(MM_DIM):
                        var z = (raw_flat[raw_base + d] - self.means[d]) * self.scales[d]
                        self.soa_features[g_soa_base + d * MM_LANES + lane] = z

        # Build 64-frame AABB `(min, max)` chunk headers + stance bitmasks
        var num_chunks = (n + MM_CHUNK_SIZE - 1) // MM_CHUNK_SIZE
        self.chunk_mins = List[Float32](length=num_chunks * MM_DIM, fill=Float32(1.0e30))
        self.chunk_maxs = List[Float32](length=num_chunks * MM_DIM, fill=Float32(-1.0e30))
        self.chunk_starts = List[Int](capacity=num_chunks)
        self.chunk_ends = List[Int](capacity=num_chunks)
        self.chunk_stance_masks = List[UInt32](length=num_chunks, fill=0)

        for c in range(num_chunks):
            var c_start = c * MM_CHUNK_SIZE
            var c_end = min(n, c_start + MM_CHUNK_SIZE)
            self.chunk_starts.append(c_start)
            self.chunk_ends.append(c_end)
            var c_base = c * MM_DIM
            var mask = UInt32(0)
            for i in range(c_start, c_end):
                var tag = self.frames[i].feature.stance_tag
                if tag >= 0 and tag < 32:
                    mask |= UInt32(1) << UInt32(tag)
                var g = i // MM_LANES
                var lane = i & (MM_LANES - 1)
                var g_soa_base = g * MM_DIM * MM_LANES
                for d in range(MM_DIM):
                    var val = self.soa_features[g_soa_base + d * MM_LANES + lane]
                    if val < self.chunk_mins[c_base + d]:
                        self.chunk_mins[c_base + d] = val
                    if val > self.chunk_maxs[c_base + d]:
                        self.chunk_maxs[c_base + d] = val
            self.chunk_stance_masks[c] = mask

        self.is_finalized = True

    @always_inline
    def _normalize_query(self, query: MotionFeature, mut q_norm: Array[Float32, MM_DIM]):
        var raw = Array[Float32, MM_DIM](fill=0.0)
        _extract_raw_channels(query, raw)
        for d in range(MM_DIM):
            q_norm[d] = (raw[d] - self.means[d]) * self.scales[d]

    @always_inline
    def _chunk_lower_bound(self, chunk_idx: Int, q_norm: Array[Float32, MM_DIM], query_stance: Int) -> Float32:
        var lb = Float32(0.0)
        if query_stance >= 0 and query_stance < 32:
            var bit = UInt32(1) << UInt32(query_stance)
            if (self.chunk_stance_masks[chunk_idx] & bit) == UInt32(0):
                lb += self.stance_mismatch_cost
        var c_base = chunk_idx * MM_DIM
        var min_ptr = self.chunk_mins.unsafe_ptr()
        var max_ptr = self.chunk_maxs.unsafe_ptr()
        for d in range(MM_DIM):
            var qv = q_norm[d]
            var lo = min_ptr[unsafe_offset=c_base + d]
            var hi = max_ptr[unsafe_offset=c_base + d]
            var below = lo - qv
            var above = qv - hi
            var excess = max(Float32(0.0), max(below, above))
            lb += excess * excess
        return lb

    def raw_cost_terms(self, f: MotionFeature, query: MotionFeature) -> Array[Float32, 6]:
        var d_p200 = (f.traj_pos_200 - query.traj_pos_200).length_squared()
        var d_p400 = (f.traj_pos_400 - query.traj_pos_400).length_squared()
        var d_p800 = (f.traj_pos_800 - query.traj_pos_800).length_squared()
        var pos_cost = (d_p200 + d_p400 + d_p800) * self.weight_pos

        var d_f200 = _wrap_angle_diff(f.traj_facing_200, query.traj_facing_200)
        var d_f400 = _wrap_angle_diff(f.traj_facing_400, query.traj_facing_400)
        var d_f800 = _wrap_angle_diff(f.traj_facing_800, query.traj_facing_800)
        var facing_cost = (d_f200 * d_f200 + d_f400 * d_f400 + d_f800 * d_f800) * self.weight_facing

        var d_lf = (f.left_foot_pos - query.left_foot_pos).length_squared()
        var d_rf = (f.right_foot_pos - query.right_foot_pos).length_squared()
        var d_lfv = (f.left_foot_vel - query.left_foot_vel).length_squared()
        var d_rfv = (f.right_foot_vel - query.right_foot_vel).length_squared()
        var feet_cost = (d_lf + d_rf + (d_lfv + d_rfv) * Float32(0.1)) * self.weight_feet

        var vel_cost = (f.hip_vel - query.hip_vel).length_squared() * self.weight_vel
        var ang_cost = (f.ang_vel - query.ang_vel).length_squared() * self.weight_ang_vel
        var stance_cost = Float32(0.0)
        if query.stance_tag >= 0 and f.stance_tag != query.stance_tag:
            stance_cost = self.stance_mismatch_cost
        return [pos_cost, facing_cost, feet_cost, vel_cost, ang_cost, stance_cost]

    def _raw_frame_cost(self, f: MotionFeature, query: MotionFeature) -> Float32:
        var terms = self.raw_cost_terms(f, query)
        return terms[0] + terms[1] + terms[2] + terms[3] + terms[4] + terms[5]

    def _eval_chunk_simd8(
        self,
        chunk_idx: Int,
        q_norm: Array[Float32, MM_DIM],
        query_stance: Int,
        cur_clip: Int,
        cur_time: Float32,
        cont_idx: Int,
        continuing_bias: Float32,
        exclude_window: Float32,
        mut best_idx: Int,
        mut best_cost: Float32,
    ):
        var c_start = self.chunk_starts[chunk_idx]
        var c_end = self.chunk_ends[chunk_idx]
        var g_start = c_start // MM_LANES
        var g_end = (c_end + MM_LANES - 1) // MM_LANES
        var soa_ptr = self.soa_features.unsafe_ptr()

        for g in range(g_start, g_end):
            var g_soa_base = g * MM_DIM * MM_LANES
            var acc = MM_F8(0.0)
            for d in range(MM_DIM):
                var col = soa_ptr.unsafe_load[width=MM_LANES](g_soa_base + d * MM_LANES)
                var diff = col - MM_F8(q_norm[d])
                acc += diff * diff

            var frame0 = g * MM_LANES
            comptime for lane in range(MM_LANES):
                var idx = frame0 + lane
                if idx >= c_start and idx < c_end:
                    if not self.excludes_frame(idx, cur_clip, cur_time, cont_idx, exclude_window):
                        var cost = acc[lane]
                        if query_stance >= 0 and self.frames[idx].feature.stance_tag != query_stance:
                            cost += self.stance_mismatch_cost
                        if idx == cont_idx:
                            cost -= continuing_bias
                        if cost < best_cost:
                            best_cost = cost
                            best_idx = idx

    def count_pruned_chunks(self, query: MotionFeature) -> Int:
        """Returns how many 64-frame chunks AABB branch-and-bound prunes for `query`."""
        if not self.is_finalized or len(self.chunk_starts) <= 1:
            return 0
        var q_norm = Array[Float32, MM_DIM](fill=0.0)
        self._normalize_query(query, q_norm)

        # Find chunk with smallest lower bound first so upper bound tightens immediately
        var num_chunks = len(self.chunk_starts)
        var seed_chunk = 0
        var min_lb = self._chunk_lower_bound(0, q_norm, query.stance_tag)
        for c in range(1, num_chunks):
            var lb = self._chunk_lower_bound(c, q_norm, query.stance_tag)
            if lb < min_lb:
                min_lb = lb
                seed_chunk = c

        var best_idx = 0
        var best_cost = Float32(1.0e30)
        self._eval_chunk_simd8(seed_chunk, q_norm, query.stance_tag, -1, 0.0, -1, 0.0, 0.0, best_idx, best_cost)

        var pruned = 0
        for c in range(num_chunks):
            if c == seed_chunk:
                continue
            var lb = self._chunk_lower_bound(c, q_norm, query.stance_tag)
            if lb >= best_cost:
                pruned += 1
            else:
                self._eval_chunk_simd8(c, q_norm, query.stance_tag, -1, 0.0, -1, 0.0, 0.0, best_idx, best_cost)
        return pruned

    def find_best_match(self, query: MotionFeature) -> Tuple[Int, Float32, Float32]:
        """Finds closest feature match in database. Returns `(clip_index, time_seconds, cost)`."""
        var n = len(self.frames)
        if n == 0:
            return (-1, Float32(0.0), Float32(1.0e30))

        if self.is_finalized and len(self.chunk_starts) > 0:
            var q_norm = Array[Float32, MM_DIM](fill=0.0)
            self._normalize_query(query, q_norm)

            var num_chunks = len(self.chunk_starts)
            var seed_chunk = 0
            var min_lb = self._chunk_lower_bound(0, q_norm, query.stance_tag)
            for c in range(1, num_chunks):
                var lb = self._chunk_lower_bound(c, q_norm, query.stance_tag)
                if lb < min_lb:
                    min_lb = lb
                    seed_chunk = c

            var best_idx = 0
            var best_cost = Float32(1.0e30)
            self._eval_chunk_simd8(seed_chunk, q_norm, query.stance_tag, -1, 0.0, -1, 0.0, 0.0, best_idx, best_cost)

            for c in range(num_chunks):
                if c == seed_chunk:
                    continue
                if self._chunk_lower_bound(c, q_norm, query.stance_tag) >= best_cost:
                    continue
                self._eval_chunk_simd8(c, q_norm, query.stance_tag, -1, 0.0, -1, 0.0, 0.0, best_idx, best_cost)

            return (self.frames[best_idx].clip_index, self.frames[best_idx].time_seconds, best_cost)

        var best_clip = self.frames[0].clip_index
        var best_time = self.frames[0].time_seconds
        var best_cost = Float32(1.0e30)
        for i in range(n):
            var total_cost = self._raw_frame_cost(self.frames[i].feature, query)
            if total_cost < best_cost:
                best_cost = total_cost
                best_clip = self.frames[i].clip_index
                best_time = self.frames[i].time_seconds
        return (best_clip, best_time, best_cost)

    def continuation_frame(self, cur_clip: Int, cur_time: Float32, exclude_window: Float32) -> Int:
        var cont_idx = -1
        if cur_clip >= 0:
            var nearest_dt = max(Float32(0.05), exclude_window)
            for i in range(len(self.frames)):
                if self.frames[i].clip_index == cur_clip:
                    var dt_abs = abs(self.frames[i].time_seconds - cur_time)
                    if dt_abs <= nearest_dt:
                        nearest_dt = dt_abs
                        cont_idx = i
        return cont_idx

    def excludes_frame(self, idx: Int, cur_clip: Int, cur_time: Float32, cont_idx: Int, window: Float32) -> Bool:
        return (
            cur_clip >= 0 and window > 0 and idx != cont_idx and self.frames[idx].clip_index == cur_clip
            and abs(self.frames[idx].time_seconds - cur_time) < window
        )

    def query_inertial(
        self,
        query: MotionFeature,
        cur_clip: Int,
        cur_time: Float32,
        continuing_bias: Float32 = 0.08,
        exclude_window: Float32 = 0.12,
    ) -> Int:
        """Stateful motion-matching frame index search:
        - Identifies the natural continuation frame `cont_idx` nearest `(cur_clip, cur_time)`
          and discounts its cost by `continuing_bias` to prevent ping-pong clip thrashing.
        - Excludes all other neighbouring frames in `cur_clip` within `exclude_window`
          seconds of `cur_time` so playback never micro-loops or freezes in place.
        - Accelerates evaluation via SIMD8 SoA + 64-frame AABB branch-and-bound when finalized.
        """
        var n = len(self.frames)
        if n == 0:
            return -1

        var cont_idx = self.continuation_frame(cur_clip, cur_time, exclude_window)

        if self.is_finalized and len(self.chunk_starts) > 0:
            var q_norm = Array[Float32, MM_DIM](fill=0.0)
            self._normalize_query(query, q_norm)

            var num_chunks = len(self.chunk_starts)
            var seed_chunk = (cont_idx // MM_CHUNK_SIZE) if cont_idx >= 0 else 0
            var best_idx = -1
            var best_cost = Float32(1.0e30)

            self._eval_chunk_simd8(
                seed_chunk, q_norm, query.stance_tag,
                cur_clip, cur_time, cont_idx, continuing_bias, exclude_window,
                best_idx, best_cost,
            )

            var bias_allowance = max(Float32(0.0), continuing_bias) if cont_idx >= 0 else Float32(0.0)
            for c in range(num_chunks):
                if c == seed_chunk:
                    continue
                if self._chunk_lower_bound(c, q_norm, query.stance_tag) - bias_allowance >= best_cost:
                    continue
                self._eval_chunk_simd8(
                    c, q_norm, query.stance_tag,
                    cur_clip, cur_time, cont_idx, continuing_bias, exclude_window,
                    best_idx, best_cost,
                )

            if best_idx >= 0:
                return best_idx
            return cont_idx if cont_idx >= 0 else 0

        var best_idx = -1
        var best_cost = Float32(1.0e30)
        for i in range(n):
            if self.excludes_frame(i, cur_clip, cur_time, cont_idx, exclude_window):
                continue
            var cost = self._raw_frame_cost(self.frames[i].feature, query)
            if i == cont_idx:
                cost -= continuing_bias
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        if best_idx >= 0:
            return best_idx
        return cont_idx if cont_idx >= 0 else 0

    def bake_clip_features(
        mut self,
        model: NativeModel,
        clip_index: Int,
        left_foot_node: Int,
        right_foot_node: Int,
        hip_node: Int = 0,
        sample_fps: Float32 = 30.0,
        loop: Bool = True,
        stance_tag: Int = STANCE_GROUND,
    ):
        """Automatically samples `model.animations[clip_index]` at `sample_fps` and
        appends root-heading-relative `MotionFrame` entries (`traj_pos/facing` at
        `+200/+400/+800ms`, foot positions/velocities, `hip_vel`, `ang_vel`, `stance_tag`)."""
        if clip_index < 0 or clip_index >= len(model.animations) or len(model.nodes) == 0:
            return
        var dur = model.animations[clip_index].duration
        if not isfinite(dur) or dur < Float32(0.0) or not isfinite(sample_fps):
            return
        comptime MAX_BAKE_STEPS: Int = 16384
        var fps = max(Float32(1.0), sample_fps)
        var raw_steps_f64 = Float64(dur) * Float64(fps) if dur > Float32(0.0) else 1.0
        var steps: Int
        var dt: Float32
        var inv_dt: Float32
        if isfinite(raw_steps_f64) and raw_steps_f64 > Float64(MAX_BAKE_STEPS):
            steps = MAX_BAKE_STEPS
            dt = dur / Float32(steps)
            inv_dt = Float32(steps) / dur
        else:
            steps = max(1, Int(raw_steps_f64)) if dur > Float32(0.0) else 1
            dt = Float32(1.0) / fps
            inv_dt = fps

        var skel_cur = SkeletonInstance(len(model.nodes))
        var skel_fut = SkeletonInstance(len(model.nodes))
        var pose_cur = Pose(len(model.nodes))
        var pose_fut = Pose(len(model.nodes))
        var pose_0 = Pose(len(model.nodes))
        var pose_end = Pose(len(model.nodes))

        var hip = hip_node if (hip_node >= 0 and hip_node < len(model.nodes)) else 0
        sample_pose(model, clip_index, Float32(0.0), pose_0)
        sample_pose(model, clip_index, dur, pose_end)
        skel_cur.apply_pose(pose_0, model)
        skel_fut.apply_pose(pose_end, model)
        var cycle_delta = skel_fut.global_position(hip) - skel_cur.global_position(hip)

        var horizons: Array[Float32, 3] = [Float32(0.2), Float32(0.4), Float32(0.8)]

        for k in range(steps):
            var t0 = min(dur, Float32(k) * dt)
            sample_pose(model, clip_index, t0, pose_cur)
            skel_cur.apply_pose(pose_cur, model)

            var p_hip0 = skel_cur.global_position(hip)
            var q_hip0 = skel_cur.global_rotation(hip)
            var fwd0 = q_hip0.rotate(Vec3(0.0, 0.0, 1.0))
            var yaw0 = det_atan2(fwd0.x, fwd0.z) if (fwd0.x * fwd0.x + fwd0.z * fwd0.z > Float32(1.0e-8)) else Float32(0.0)
            var cy = det_cos(-yaw0)
            var sy = det_sin(-yaw0)

            var feat = MotionFeature()
            feat.stance_tag = stance_tag

            # Local foot positions at t0
            if left_foot_node >= 0 and left_foot_node < skel_cur.node_count:
                var dl = skel_cur.global_position(left_foot_node) - p_hip0
                feat.left_foot_pos = Vec3(dl.x * cy + dl.z * sy, dl.y, -dl.x * sy + dl.z * cy)
            if right_foot_node >= 0 and right_foot_node < skel_cur.node_count:
                var dr = skel_cur.global_position(right_foot_node) - p_hip0
                feat.right_foot_pos = Vec3(dr.x * cy + dr.z * sy, dr.y, -dr.x * sy + dr.z * cy)

            # Finite-difference velocities over +dt
            var t_vel = t0 + dt
            var vel_wraps = 0
            if dur > Float32(1.0e-4):
                if loop:
                    var d64 = Float64(dur)
                    var q_v = floor(Float64(t_vel) / d64)
                    if isfinite(q_v) and q_v >= 0.0:
                        t_vel = Float32(Float64(t_vel) - q_v * d64)
                        if t_vel >= dur:
                            t_vel = Float32(0.0)
                        vel_wraps = Int(min(1.0e9, q_v))
                else:
                    t_vel = min(dur, t_vel)
            else:
                t_vel = Float32(0.0)

            sample_pose(model, clip_index, t_vel, pose_fut)
            skel_fut.apply_pose(pose_fut, model)
            var wrap_shift_v = cycle_delta * Float32(vel_wraps)
            var p_hip1 = skel_fut.global_position(hip) + wrap_shift_v
            var q_hip1 = skel_fut.global_rotation(hip)

            var v_hip_w = (p_hip1 - p_hip0) * inv_dt
            feat.hip_vel = Vec3(v_hip_w.x * cy + v_hip_w.z * sy, v_hip_w.y, -v_hip_w.x * sy + v_hip_w.z * cy)

            var w_hip_w = det_quat_to_scaled_angle(q_hip1.multiply(q_hip0.conjugate())) * inv_dt
            feat.ang_vel = Vec3(w_hip_w.x * cy + w_hip_w.z * sy, w_hip_w.y, -w_hip_w.x * sy + w_hip_w.z * cy)

            if left_foot_node >= 0 and left_foot_node < skel_fut.node_count:
                var pl1 = skel_fut.global_position(left_foot_node) + wrap_shift_v - p_hip1
                var lf1 = Vec3(pl1.x * cy + pl1.z * sy, pl1.y, -pl1.x * sy + pl1.z * cy)
                feat.left_foot_vel = (lf1 - feat.left_foot_pos) * inv_dt
            if right_foot_node >= 0 and right_foot_node < skel_fut.node_count:
                var pr1 = skel_fut.global_position(right_foot_node) + wrap_shift_v - p_hip1
                var rf1 = Vec3(pr1.x * cy + pr1.z * sy, pr1.y, -pr1.x * sy + pr1.z * cy)
                feat.right_foot_vel = (rf1 - feat.right_foot_pos) * inv_dt

            # Future trajectory horizons +200ms, +400ms, +800ms
            for h_idx in range(3):
                var th = t0 + horizons[h_idx]
                var h_wraps = 0
                var extra_t = Float32(0.0)
                if dur > Float32(1.0e-4):
                    if loop:
                        var d64 = Float64(dur)
                        var q_h = floor(Float64(th) / d64)
                        if isfinite(q_h) and q_h >= 0.0:
                            th = Float32(Float64(th) - q_h * d64)
                            if th >= dur:
                                th = Float32(0.0)
                            h_wraps = Int(min(1.0e9, q_h))
                    else:
                        if th > dur:
                            extra_t = th - dur
                            th = dur
                else:
                    th = Float32(0.0)

                sample_pose(model, clip_index, th, pose_fut)
                skel_fut.apply_pose(pose_fut, model)
                var p_h = skel_fut.global_position(hip) + cycle_delta * Float32(h_wraps) + v_hip_w * extra_t
                var dp = p_h - p_hip0
                var local_p = Vec3(dp.x * cy + dp.z * sy, dp.y, -dp.x * sy + dp.z * cy)

                var q_h = skel_fut.global_rotation(hip)
                var fwd_h = q_h.rotate(Vec3(0.0, 0.0, 1.0))
                var yaw_h = det_atan2(fwd_h.x, fwd_h.z) if (fwd_h.x * fwd_h.x + fwd_h.z * fwd_h.z > Float32(1.0e-8)) else yaw0
                var rel_yaw = _wrap_signed_angle(yaw_h - yaw0)

                if h_idx == 0:
                    feat.traj_pos_200 = local_p
                    feat.traj_facing_200 = rel_yaw
                elif h_idx == 1:
                    feat.traj_pos_400 = local_p
                    feat.traj_facing_400 = rel_yaw
                else:
                    feat.traj_pos_800 = local_p
                    feat.traj_facing_800 = rel_yaw

            self.add_frame(clip_index, k, t0, feat)
