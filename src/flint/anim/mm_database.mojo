"""Motion-matching animation database, features and search (`flint.anim.mm_database`).

The database layout, matching features and search follow "Learned Motion Matching" (Holden,
Kanoun, Perepichka, Popa; SIGGRAPH 2020) and its MIT reference implementation
(https://github.com/orangeduck/Motion-Matching, `database.h` and `generate_database.py`):

* Every frame stores, per bone, the local position, velocity, rotation and angular velocity.
  Bone 0 is the *simulation bone*: the hips projected on the ground, facing the smoothed hip
  heading, in animation (world) space; every other bone is local to its parent. Frames are
  grouped into ranges (one per source clip, mirrored copies included).
* Matching features (27 floats): left/right foot position and velocity and hip velocity in the
  simulation-bone frame, plus the future trajectory (positions and facing directions in the
  ground plane, 20/40/60 frames ahead). Each group is normalised by its mean and by its mean
  per-dimension standard deviation divided by the group weight.
* Search: a brute-force scan (`search_brute`) and Holden's two-level AABB tree (boxes over 16
  and 64 consecutive frames, `search`). Both visit frames in the same order, accumulate costs
  in the same order and only accept strictly better costs, so they return the same frame.

`MmDatabaseBuilder` turns retargeted clips (`flint.anim.retarget.RetargetClip`) into a
database: optional left/right mirroring, simulation-bone extraction, central-difference
velocities, and foot contacts from toe speed with a majority filter.

`save` / `load` use the reference `database.bin` layout (quaternions stored w, x, y, z) with the
bone names appended, so the offline trainer (`tools/learned_mm/train.py`) reads the same file;
`save_features` writes the reference `features.bin` layout.
"""
from std.math import sqrt, abs
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.detmath import f32_bits, f32_from_bits, det_exp
from flint.platform.os import read_bytes, write_bytes
from flint.anim.retarget import RetargetSkeleton, RetargetClip
from flint.anim.mm_springs import (
    MmPose, mm_quat_abs, mm_quat_inv, mm_quat_mul, mm_quat_mul_inv, mm_quat_inv_mul_vec, mm_quat_mul_vec,
    mm_quat_to_scaled_angle_axis, mm_quat_between,
)

comptime MM_BOUND_SM: Int = 16
comptime MM_BOUND_LR: Int = 64
comptime MM_NFEATURES: Int = 27
comptime MM_FPS: Float32 = 60.0
comptime MM_BIG: Float32 = 3.0e38


# ----------------------------------------------------------------------------- byte IO

struct MmBytes(Movable):
    var data: List[UInt8]
    var at: Int

    def __init__(out self):
        self.data = List[UInt8]()
        self.at = 0

    def __init__(out self, var data: List[UInt8]):
        self.data = data^
        self.at = 0

    def u32(mut self, v: Int):
        var x = UInt32(v)
        for k in range(4):
            self.data.append(UInt8((x >> UInt32(8 * k)) & 255))

    def f32(mut self, v: Float32):
        self.u32(Int(f32_bits(v)))

    def u8(mut self, v: Int):
        self.data.append(UInt8(v))

    def read_u32(mut self) raises -> Int:
        if self.at + 4 > len(self.data):
            raise Error("mm: truncated file")
        var x = UInt32(0)
        for k in range(4):
            x |= UInt32(self.data[self.at + k]) << UInt32(8 * k)
        self.at += 4
        return Int(x)

    def read_i32(mut self) raises -> Int:
        var u = self.read_u32()
        return u - (1 << 32) if u >= (1 << 31) else u

    def read_f32(mut self) raises -> Float32:
        return f32_from_bits(UInt32(self.read_u32()))

    def read_u8(mut self) raises -> Int:
        if self.at >= len(self.data):
            raise Error("mm: truncated file")
        self.at += 1
        return Int(self.data[self.at - 1])

    def remaining(self) -> Int:
        return len(self.data) - self.at


# ----------------------------------------------------------------------------- weights

@fieldwise_init
struct MmFeatureWeights(Copyable, ImplicitlyCopyable, Movable):
    var foot_position: Float32
    var foot_velocity: Float32
    var hip_velocity: Float32
    var trajectory_positions: Float32
    var trajectory_directions: Float32

    @staticmethod
    def default() -> Self:
        """The reference controller's weights."""
        return Self(0.75, 1.0, 1.0, 1.0, 1.5)


# ----------------------------------------------------------------------------- kinematics

def mm_fk_velocity(
    pos: List[Vec3], vel: List[Vec3], rot: List[Quat], ang: List[Vec3], base: Int, parents: List[Int], bone: Int
) -> Tuple[Vec3, Vec3, Quat, Vec3]:
    """Global position, velocity, rotation and angular velocity of `bone` for the frame whose
    bone 0 sits at index `base` of the flat per-bone arrays."""
    var chain = List[Int]()
    var b = bone
    while b >= 0:
        chain.append(b)
        b = parents[b]
    var j0 = chain[len(chain) - 1]
    var gp = pos[base + j0]
    var gv = vel[base + j0]
    var gr = rot[base + j0]
    var ga = ang[base + j0]
    var k = len(chain) - 2
    while k >= 0:
        var j = chain[k]
        var lp = gr.rotate(pos[base + j])
        gv = gv + gr.rotate(vel[base + j]) + ga.cross(lp)
        gp = gp + lp
        ga = gr.rotate(ang[base + j]) + ga
        gr = gr.multiply(rot[base + j]).normalized()
        k -= 1
    return (gp, gv, gr, ga)


def mm_fk_full(pose: MmPose, parents: List[Int], mut gpos: List[Vec3], mut grot: List[Quat]):
    """Global transforms of every bone of `pose` (parents precede children)."""
    var n = pose.count()
    if len(gpos) != n:
        gpos = List[Vec3](length=n, fill=Vec3())
        grot = List[Quat](length=n, fill=Quat())
    for i in range(n):
        var p = parents[i]
        if p < 0:
            gpos[i] = pose.pos[i]
            grot[i] = pose.rot[i]
        else:
            gpos[i] = grot[p].rotate(pose.pos[i]) + gpos[p]
            grot[i] = grot[p].multiply(pose.rot[i]).normalized()


# ----------------------------------------------------------------------------- database

struct MmDatabase(Copyable, Movable):
    var nframes: Int
    var nbones: Int
    var names: List[String]
    var parents: List[Int]
    var pos: List[Vec3]          # [frame * nbones + bone]
    var vel: List[Vec3]
    var rot: List[Quat]
    var ang: List[Vec3]
    var range_starts: List[Int]
    var range_stops: List[Int]
    var ncontacts: Int
    var contacts: List[Bool]     # [frame * ncontacts + c]
    var nfeatures: Int
    var features: List[Float32]  # normalised, [frame * nfeatures + j]
    var features_offset: List[Float32]
    var features_scale: List[Float32]
    var bound_sm_min: List[Float32]
    var bound_sm_max: List[Float32]
    var bound_lr_min: List[Float32]
    var bound_lr_max: List[Float32]
    var bone_left_foot: Int
    var bone_right_foot: Int
    var bone_hips: Int
    var bone_left_toe: Int
    var bone_right_toe: Int

    def __init__(out self):
        self.nframes = 0
        self.nbones = 0
        self.names = List[String]()
        self.parents = List[Int]()
        self.pos = List[Vec3]()
        self.vel = List[Vec3]()
        self.rot = List[Quat]()
        self.ang = List[Vec3]()
        self.range_starts = List[Int]()
        self.range_stops = List[Int]()
        self.ncontacts = 2
        self.contacts = List[Bool]()
        self.nfeatures = 0
        self.features = List[Float32]()
        self.features_offset = List[Float32]()
        self.features_scale = List[Float32]()
        self.bound_sm_min = List[Float32]()
        self.bound_sm_max = List[Float32]()
        self.bound_lr_min = List[Float32]()
        self.bound_lr_max = List[Float32]()
        self.bone_left_foot = -1
        self.bone_right_foot = -1
        self.bone_hips = -1
        self.bone_left_toe = -1
        self.bone_right_toe = -1

    def nranges(self) -> Int:
        return len(self.range_starts)

    def bone_index(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    def resolve_bones(mut self):
        """Feature / contact bones by name (stick biped or SOMA-style names)."""
        self.bone_hips = self._first(["Hips", "Pelvis"])
        self.bone_left_foot = self._first(["LeftFoot", "LeftAnkle", "LeftFootMod"])
        self.bone_right_foot = self._first(["RightFoot", "RightAnkle", "RightFootMod"])
        self.bone_left_toe = self._first(["LeftToes", "LeftToe", "LeftToeBase"])
        self.bone_right_toe = self._first(["RightToes", "RightToe", "RightToeBase"])
        if self.bone_hips < 0:
            self.bone_hips = 1 if self.nbones > 1 else 0

    def _first(self, names: List[String]) -> Int:
        for n in names:
            var i = self.bone_index(n)
            if i >= 0:
                return i
        return -1

    @always_inline
    def frame_base(self, frame: Int) -> Int:
        return frame * self.nbones

    def pose_at(self, frame: Int, mut out: MmPose):
        var n = self.nbones
        if out.count() != n:
            out = MmPose(n)
        var b = frame * n
        for i in range(n):
            out.pos[i] = self.pos[b + i]
            out.vel[i] = self.vel[b + i]
            out.rot[i] = self.rot[b + i]
            out.ang[i] = self.ang[b + i]

    def contact(self, frame: Int, c: Int) -> Bool:
        return self.contacts[frame * self.ncontacts + c]

    def range_of(self, frame: Int) -> Int:
        for r in range(len(self.range_starts)):
            if frame >= self.range_starts[r] and frame < self.range_stops[r]:
                return r
        return -1

    def trajectory_index_clamp(self, frame: Int, offset: Int) -> Int:
        var r = self.range_of(frame)
        if r < 0:
            return frame
        return min(max(frame + offset, self.range_starts[r]), self.range_stops[r] - 1)

    @always_inline
    def feature(self, frame: Int, j: Int) -> Float32:
        return self.features[frame * self.nfeatures + j]

    # ------------------------------------------------------------------ features

    def _set_raw(mut self, frame: Int, j: Int, v: Float32):
        self.features[frame * self.nfeatures + j] = v

    def _bone_position_feature(mut self, mut offset: Int, bone: Int, weight: Float32):
        for i in range(self.nframes):
            var b = i * self.nbones
            var g = mm_fk_velocity(self.pos, self.vel, self.rot, self.ang, b, self.parents, bone)
            var p = mm_quat_inv_mul_vec(self.rot[b], g[0] - self.pos[b])
            self._set_raw(i, offset + 0, p.x)
            self._set_raw(i, offset + 1, p.y)
            self._set_raw(i, offset + 2, p.z)
        self._normalize_feature(offset, 3, weight)
        offset += 3

    def _bone_velocity_feature(mut self, mut offset: Int, bone: Int, weight: Float32):
        for i in range(self.nframes):
            var b = i * self.nbones
            var g = mm_fk_velocity(self.pos, self.vel, self.rot, self.ang, b, self.parents, bone)
            var v = mm_quat_inv_mul_vec(self.rot[b], g[1])
            self._set_raw(i, offset + 0, v.x)
            self._set_raw(i, offset + 1, v.y)
            self._set_raw(i, offset + 2, v.z)
        self._normalize_feature(offset, 3, weight)
        offset += 3

    def _trajectory_position_feature(mut self, mut offset: Int, weight: Float32):
        for i in range(self.nframes):
            var b = i * self.nbones
            var k = 0
            for step in [20, 40, 60]:
                var t = self.trajectory_index_clamp(i, step)
                var p = mm_quat_inv_mul_vec(self.rot[b], self.pos[t * self.nbones] - self.pos[b])
                self._set_raw(i, offset + k, p.x)
                self._set_raw(i, offset + k + 1, p.z)
                k += 2
        self._normalize_feature(offset, 6, weight)
        offset += 6

    def _trajectory_direction_feature(mut self, mut offset: Int, weight: Float32):
        for i in range(self.nframes):
            var b = i * self.nbones
            var k = 0
            for step in [20, 40, 60]:
                var t = self.trajectory_index_clamp(i, step)
                var d = mm_quat_inv_mul_vec(self.rot[b], self.rot[t * self.nbones].rotate(Vec3(0, 0, 1)))
                self._set_raw(i, offset + k, d.x)
                self._set_raw(i, offset + k + 1, d.z)
                k += 2
        self._normalize_feature(offset, 6, weight)
        offset += 6

    def _normalize_feature(mut self, offset: Int, size: Int, weight: Float32):
        var n = self.nframes
        var nf = self.nfeatures
        for j in range(size):
            var m: Float32 = 0
            for i in range(n):
                m += self.features[i * nf + offset + j] / Float32(n)
            self.features_offset[offset + j] = m
        var std: Float32 = 0
        for j in range(size):
            var v: Float32 = 0
            for i in range(n):
                var d = self.features[i * nf + offset + j] - self.features_offset[offset + j]
                v += d * d / Float32(n)
            std += sqrt(v) / Float32(size)
        std = max(std, Float32(1.0e-6))
        for j in range(size):
            self.features_scale[offset + j] = std / weight
        for i in range(n):
            for j in range(size):
                var k = i * nf + offset + j
                self.features[k] = (self.features[k] - self.features_offset[offset + j]) / self.features_scale[offset + j]

    def build_matching_features(mut self, w: MmFeatureWeights = MmFeatureWeights.default()):
        """Computes the normalised 27-float feature of every frame and the AABB bounds."""
        self.resolve_bones()
        self.nfeatures = MM_NFEATURES
        self.features = List[Float32](length=self.nframes * self.nfeatures, fill=0)
        self.features_offset = List[Float32](length=self.nfeatures, fill=0)
        self.features_scale = List[Float32](length=self.nfeatures, fill=1)
        var offset = 0
        self._bone_position_feature(offset, self.bone_left_foot, w.foot_position)
        self._bone_position_feature(offset, self.bone_right_foot, w.foot_position)
        self._bone_velocity_feature(offset, self.bone_left_foot, w.foot_velocity)
        self._bone_velocity_feature(offset, self.bone_right_foot, w.foot_velocity)
        self._bone_velocity_feature(offset, self.bone_hips, w.hip_velocity)
        self._trajectory_position_feature(offset, w.trajectory_positions)
        self._trajectory_direction_feature(offset, w.trajectory_directions)
        self.build_bounds()

    def build_bounds(mut self):
        var nf = self.nfeatures
        var nsm = (self.nframes + MM_BOUND_SM - 1) // MM_BOUND_SM
        var nlr = (self.nframes + MM_BOUND_LR - 1) // MM_BOUND_LR
        self.bound_sm_min = List[Float32](length=nsm * nf, fill=MM_BIG)
        self.bound_sm_max = List[Float32](length=nsm * nf, fill=-MM_BIG)
        self.bound_lr_min = List[Float32](length=nlr * nf, fill=MM_BIG)
        self.bound_lr_max = List[Float32](length=nlr * nf, fill=-MM_BIG)
        for i in range(self.nframes):
            var ism = (i // MM_BOUND_SM) * nf
            var ilr = (i // MM_BOUND_LR) * nf
            for j in range(nf):
                var v = self.features[i * nf + j]
                self.bound_sm_min[ism + j] = min(self.bound_sm_min[ism + j], v)
                self.bound_sm_max[ism + j] = max(self.bound_sm_max[ism + j], v)
                self.bound_lr_min[ilr + j] = min(self.bound_lr_min[ilr + j], v)
                self.bound_lr_max[ilr + j] = max(self.bound_lr_max[ilr + j], v)

    def normalize_query(self, query: List[Float32]) -> List[Float32]:
        var q = List[Float32](length=self.nfeatures, fill=0)
        for j in range(self.nfeatures):
            q[j] = (query[j] - self.features_offset[j]) / self.features_scale[j]
        return q^

    def denormalized_feature(self, frame: Int, j: Int) -> Float32:
        return self.features[frame * self.nfeatures + j] * self.features_scale[j] + self.features_offset[j]

    # ------------------------------------------------------------------ search

    def _frame_cost(self, q: List[Float32], i: Int, start: Float32, best: Float32) -> Float32:
        var c = start
        var nf = self.nfeatures
        var base = i * nf
        for j in range(nf):
            var d = q[j] - self.features[base + j]
            c += d * d
            if c >= best:
                break
        return c

    def search_brute(
        self, q: List[Float32], mut best_index: Int, mut best_cost: Float32,
        transition_cost: Float32 = 0, ignore_range_end: Int = 20, ignore_surrounding: Int = 20,
    ):
        """Brute-force nearest frame to the normalised query `q`. On entry `best_index` is the
        playing frame (or -1), which competes with its own cost."""
        var curr = best_index
        if best_index != -1:
            best_cost = self._frame_cost(q, best_index, 0, MM_BIG)
        for r in range(len(self.range_starts)):
            var stop = self.range_stops[r] - ignore_range_end
            for i in range(self.range_starts[r], stop):
                if curr != -1 and abs(i - curr) < ignore_surrounding:
                    continue
                var c = self._frame_cost(q, i, transition_cost, best_cost)
                if c < best_cost:
                    best_index = i
                    best_cost = c

    @always_inline
    def _box_cost(self, q: List[Float32], lo: List[Float32], hi: List[Float32], box: Int, start: Float32, best: Float32) -> Float32:
        var c = start
        var nf = self.nfeatures
        var base = box * nf
        for j in range(nf):
            var v = q[j]
            var cl = min(max(v, lo[base + j]), hi[base + j])
            c += (v - cl) * (v - cl)
            if c >= best:
                break
        return c

    def search(
        self, q: List[Float32], mut best_index: Int, mut best_cost: Float32,
        transition_cost: Float32 = 0, ignore_range_end: Int = 20, ignore_surrounding: Int = 20,
    ):
        """Holden's AABB-accelerated search: same result as `search_brute`."""
        var curr = best_index
        if best_index != -1:
            best_cost = self._frame_cost(q, best_index, 0, MM_BIG)
        for r in range(len(self.range_starts)):
            var i = self.range_starts[r]
            var range_end = self.range_stops[r] - ignore_range_end
            while i < range_end:
                var i_lr = i // MM_BOUND_LR
                var i_lr_next = (i_lr + 1) * MM_BOUND_LR
                if self._box_cost(q, self.bound_lr_min, self.bound_lr_max, i_lr, transition_cost, best_cost) >= best_cost:
                    i = i_lr_next
                    continue
                while i < i_lr_next and i < range_end:
                    var i_sm = i // MM_BOUND_SM
                    var i_sm_next = (i_sm + 1) * MM_BOUND_SM
                    if self._box_cost(q, self.bound_sm_min, self.bound_sm_max, i_sm, transition_cost, best_cost) >= best_cost:
                        i = i_sm_next
                        continue
                    while i < i_sm_next and i < range_end:
                        if curr != -1 and abs(i - curr) < ignore_surrounding:
                            i += 1
                            continue
                        var c = self._frame_cost(q, i, transition_cost, best_cost)
                        if c < best_cost:
                            best_index = i
                            best_cost = c
                        i += 1

    # ------------------------------------------------------------------ files

    def save(self, path: String) raises:
        var w = MmBytes()
        var nf = self.nframes
        var nb = self.nbones
        w.u32(nf); w.u32(nb)
        for v in self.pos:
            w.f32(v.x); w.f32(v.y); w.f32(v.z)
        w.u32(nf); w.u32(nb)
        for v in self.vel:
            w.f32(v.x); w.f32(v.y); w.f32(v.z)
        w.u32(nf); w.u32(nb)
        for q in self.rot:
            w.f32(q.w); w.f32(q.x); w.f32(q.y); w.f32(q.z)
        w.u32(nf); w.u32(nb)
        for v in self.ang:
            w.f32(v.x); w.f32(v.y); w.f32(v.z)
        w.u32(nb)
        for p in self.parents:
            w.u32(p if p >= 0 else (1 << 32) - 1)
        w.u32(len(self.range_starts))
        for s in self.range_starts:
            w.u32(s)
        w.u32(len(self.range_stops))
        for s in self.range_stops:
            w.u32(s)
        w.u32(nf); w.u32(self.ncontacts)
        for c in self.contacts:
            w.u8(1 if c else 0)
        # Flint extension: bone names
        w.u32(len(self.names))
        for n in self.names:
            var bytes = n.as_bytes()
            w.u32(len(bytes))
            for b in bytes:
                w.u8(Int(b))
        if not write_bytes(path, w.data):
            raise Error("mm: cannot write " + path)

    @staticmethod
    def load(path: String) raises -> MmDatabase:
        var bytes = read_bytes(path)
        if len(bytes) == 0:
            raise Error("mm: cannot read " + path)
        var r = MmBytes(bytes^)
        var db = MmDatabase()
        db.nframes = r.read_u32()
        db.nbones = r.read_u32()
        var n = db.nframes * db.nbones
        db.pos = List[Vec3](capacity=n)
        for _ in range(n):
            var x = r.read_f32(); var y = r.read_f32(); var z = r.read_f32()
            db.pos.append(Vec3(x, y, z))
        _ = r.read_u32(); _ = r.read_u32()
        db.vel = List[Vec3](capacity=n)
        for _ in range(n):
            var x = r.read_f32(); var y = r.read_f32(); var z = r.read_f32()
            db.vel.append(Vec3(x, y, z))
        _ = r.read_u32(); _ = r.read_u32()
        db.rot = List[Quat](capacity=n)
        for _ in range(n):
            var qw = r.read_f32(); var qx = r.read_f32(); var qy = r.read_f32(); var qz = r.read_f32()
            db.rot.append(Quat(qx, qy, qz, qw))
        _ = r.read_u32(); _ = r.read_u32()
        db.ang = List[Vec3](capacity=n)
        for _ in range(n):
            var x = r.read_f32(); var y = r.read_f32(); var z = r.read_f32()
            db.ang.append(Vec3(x, y, z))
        var nb = r.read_u32()
        for _ in range(nb):
            db.parents.append(r.read_i32())
        var nr = r.read_u32()
        for _ in range(nr):
            db.range_starts.append(r.read_u32())
        nr = r.read_u32()
        for _ in range(nr):
            db.range_stops.append(r.read_u32())
        _ = r.read_u32()
        db.ncontacts = r.read_u32()
        for _ in range(db.nframes * db.ncontacts):
            db.contacts.append(r.read_u8() != 0)
        if r.remaining() >= 4:
            var nn = r.read_u32()
            for _ in range(nn):
                var l = r.read_u32()
                var s = String()
                for _ in range(l):
                    s += chr(r.read_u8())
                db.names.append(s^)
        while len(db.names) < db.nbones:
            db.names.append("bone_" + String(len(db.names)))
        db.resolve_bones()
        return db^

    def save_features(self, path: String) raises:
        var w = MmBytes()
        w.u32(self.nframes); w.u32(self.nfeatures)
        for v in self.features:
            w.f32(v)
        w.u32(self.nfeatures)
        for v in self.features_offset:
            w.f32(v)
        w.u32(self.nfeatures)
        for v in self.features_scale:
            w.f32(v)
        if not write_bytes(path, w.data):
            raise Error("mm: cannot write " + path)


# ----------------------------------------------------------------------------- builder

def _smooth_vec(src: List[Vec3], window: Int) -> List[Vec3]:
    """Gaussian smoothing (sigma = window / 6) with renormalised edges."""
    var n = len(src)
    var half = window // 2
    var sigma = Float32(window) / 6
    var w = List[Float32](length=2 * half + 1, fill=0)
    for k in range(-half, half + 1):
        w[k + half] = det_exp(-Float32(k * k) / (2 * sigma * sigma))
    var out = List[Vec3](length=n, fill=Vec3())
    for i in range(n):
        var acc = Vec3()
        var ws: Float32 = 0
        for k in range(-half, half + 1):
            var j = i + k
            if j < 0 or j >= n:
                continue
            acc = acc + src[j] * w[k + half]
            ws += w[k + half]
        out[i] = acc * (Float32(1) / ws)
    return out^


def mm_mirror_index(names: List[String]) -> List[Int]:
    """For every joint, its left/right counterpart (itself when unpaired)."""
    var out = List[Int](length=len(names), fill=-1)
    for i in range(len(names)):
        var n = names[i]
        var other = n
        if n.startswith("Left"):
            other = "Right" + String(n[byte=4:])
        elif n.startswith("Right"):
            other = "Left" + String(n[byte=5:])
        out[i] = i
        for j in range(len(names)):
            if names[j] == other:
                out[i] = j
    return out^


struct MmDatabaseBuilder(Movable):
    """Accumulates retargeted clips into an `MmDatabase` (bone 0 = simulation bone, bone j+1 =
    skeleton joint j). The skeleton must have identity rest rotations and mirror-symmetric
    rest offsets for `mirror=True` (the Flint stick biped does)."""
    var skeleton: RetargetSkeleton
    var db: MmDatabase
    var mirror_of: List[Int]
    var position_joint: Int   # projected on the ground for the simulation bone position
    var direction_joint: Int  # its +Z heading gives the simulation bone direction
    var position_window: Int
    var direction_window: Int
    var contact_speed: Float32

    def __init__(out self, var skeleton: RetargetSkeleton):
        self.mirror_of = mm_mirror_index(skeleton.names)
        self.position_joint = max(0, skeleton.index_of("Chest"))
        self.direction_joint = max(0, skeleton.index_of("Hips"))
        self.position_window = 31
        self.direction_window = 61
        self.contact_speed = 0.15
        self.db = MmDatabase()
        self.db.nbones = skeleton.count() + 1
        self.db.names.append("Simulation")
        self.db.parents.append(-1)
        for j in range(skeleton.count()):
            self.db.names.append(skeleton.names[j])
            self.db.parents.append(skeleton.parents[j] + 1)
        self.skeleton = skeleton^
        self.db.resolve_bones()

    def add_clip(mut self, clip: RetargetClip, mirror: Bool = False, start: Int = 0, stop: Int = -1):
        """Appends frames [start, stop) of `clip` (resampled to 60 Hz) as one range."""
        var sk = self.skeleton.copy()
        var nj = sk.count()
        var s_end = clip.frame_count() if stop < 0 else min(stop, clip.frame_count())
        if s_end - start < 8:
            return
        # resample to 60 fps
        var src_rate = clip.sample_rate
        var dur = Float32(s_end - 1 - start) / src_rate
        var nf = Int(dur * MM_FPS) + 1
        var lrot = List[List[Quat]]()
        var root = List[Vec3]()
        for f in range(nf):
            var t = Float32(f) / MM_FPS * src_rate + Float32(start)
            var a = min(Int(t), s_end - 1)
            var b = min(a + 1, s_end - 1)
            var u = t - Float32(a)
            var rs = List[Quat](capacity=nj)
            for j in range(nj):
                rs.append(clip.rotations[a][j].slerp(clip.rotations[b][j], u))
            lrot.append(rs^)
            root.append(clip.root[a] + (clip.root[b] - clip.root[a]) * u)
        # global transforms (and mirrored)
        var gpos = List[List[Vec3]]()
        var grot = List[List[Quat]]()
        for f in range(nf):
            var gp = List[Vec3]()
            var gr = List[Quat]()
            sk.world(lrot[f], root[f], gp, gr)
            if mirror:
                var mp = List[Vec3](length=nj, fill=Vec3())
                var mr = List[Quat](length=nj, fill=Quat())
                for j in range(nj):
                    var o = self.mirror_of[j]
                    mp[j] = Vec3(-gp[o].x, gp[o].y, gp[o].z)
                    mr[j] = Quat(gr[o].x, -gr[o].y, -gr[o].z, gr[o].w)
                gp = mp^
                gr = mr^
            gpos.append(gp^)
            grot.append(gr^)
        # local rotations from the (possibly mirrored) globals, with sign continuity
        var rot = List[List[Quat]]()
        for f in range(nf):
            var rs = List[Quat](capacity=nj)
            for j in range(nj):
                var p = sk.parents[j]
                var q = grot[f][j] if p < 0 else grot[f][p].conjugate().multiply(grot[f][j]).normalized()
                if f > 0 and q.dot(rot[f - 1][j]) < 0:
                    q = Quat(-q.x, -q.y, -q.z, -q.w)
                rs.append(q)
            rot.append(rs^)
        # simulation bone
        var sim_p = List[Vec3](capacity=nf)
        var sim_d = List[Vec3](capacity=nf)
        for f in range(nf):
            var pp = gpos[f][self.position_joint]
            sim_p.append(Vec3(pp.x, 0, pp.z))
            var d = grot[f][self.direction_joint].rotate(Vec3(0, 0, 1))
            d = Vec3(d.x, 0, d.z)
            var l = d.length()
            sim_d.append(d * (Float32(1) / l) if l > Float32(1.0e-6) else Vec3(0, 0, 1))
        sim_p = _smooth_vec(sim_p, self.position_window)
        sim_d = _smooth_vec(sim_d, self.direction_window)
        var nb = nj + 1
        var pos = List[Vec3](length=nf * nb, fill=Vec3())
        var qr = List[Quat](length=nf * nb, fill=Quat())
        for f in range(nf):
            var d = sim_d[f]
            var l = d.length()
            d = d * (Float32(1) / l) if l > Float32(1.0e-6) else Vec3(0, 0, 1)
            var sr = mm_quat_between(Vec3(0, 0, 1), d)
            if f > 0 and sr.dot(qr[(f - 1) * nb]) < 0:
                sr = Quat(-sr.x, -sr.y, -sr.z, -sr.w)
            pos[f * nb] = sim_p[f]
            qr[f * nb] = sr
            for j in range(nj):
                var p = sk.parents[j]
                if p < 0:
                    pos[f * nb + j + 1] = mm_quat_inv_mul_vec(sr, gpos[f][j] - sim_p[f])
                    qr[f * nb + j + 1] = mm_quat_inv(sr).multiply(rot[f][j]).normalized()
                else:
                    pos[f * nb + j + 1] = sk.rest_t[j]
                    qr[f * nb + j + 1] = rot[f][j]
        # velocities (central differences, linear extrapolation at the ends)
        var vel = List[Vec3](length=nf * nb, fill=Vec3())
        var ang = List[Vec3](length=nf * nb, fill=Vec3())
        for f in range(1, nf - 1):
            for j in range(nb):
                var a = (f - 1) * nb + j
                var c = f * nb + j
                var b = (f + 1) * nb + j
                vel[c] = ((pos[b] - pos[c]) * MM_FPS + (pos[c] - pos[a]) * MM_FPS) * Float32(0.5)
                var w1 = mm_quat_to_scaled_angle_axis(mm_quat_abs(mm_quat_mul_inv(qr[b], qr[c])))
                var w0 = mm_quat_to_scaled_angle_axis(mm_quat_abs(mm_quat_mul_inv(qr[c], qr[a])))
                ang[c] = (w1 * MM_FPS + w0 * MM_FPS) * Float32(0.5)
        for j in range(nb):
            vel[j] = vel[nb + j] - (vel[3 * nb + j] - vel[2 * nb + j])
            ang[j] = ang[nb + j] - (ang[3 * nb + j] - ang[2 * nb + j])
            var l = (nf - 1) * nb + j
            vel[l] = vel[l - nb] + (vel[l - nb] - vel[l - 2 * nb])
            ang[l] = ang[l - nb] + (ang[l - nb] - ang[l - 2 * nb])
        # contacts: toe speed under the threshold, then a 6-frame majority filter
        var toes = [self.db.bone_left_toe, self.db.bone_right_toe]
        var raw = List[Bool](length=nf * 2, fill=False)
        for f in range(nf):
            for c in range(2):
                if toes[c] < 0:
                    continue
                var g = mm_fk_velocity(pos, vel, qr, ang, f * nb, self.db.parents, toes[c])
                raw[f * 2 + c] = g[1].length() < self.contact_speed
        var first = self.db.nframes
        for f in range(nf):
            for c in range(2):
                var cnt = 0
                for k in range(-3, 3):
                    var g = min(max(f + k, 0), nf - 1)
                    if raw[g * 2 + c]:
                        cnt += 1
                self.db.contacts.append(cnt >= 3)
        for i in range(nf * nb):
            self.db.pos.append(pos[i])
            self.db.vel.append(vel[i])
            self.db.rot.append(qr[i])
            self.db.ang.append(ang[i])
        self.db.nframes += nf
        self.db.range_starts.append(first)
        self.db.range_stops.append(first + nf)

    def finish(mut self, w: MmFeatureWeights = MmFeatureWeights.default()) -> MmDatabase:
        self.db.build_matching_features(w)
        return self.db.copy()
