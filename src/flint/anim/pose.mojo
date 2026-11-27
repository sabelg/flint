"""Poses: every node's local translation, rotation and scale, sampled from a
clip, blended, masked, added and inertialized (package 3.3 / v4).

Upgrades in Flint v4:
- glTF 2.0 `INTERP_STEP` (`<=` binary search reaching final keyframe) and exact
  `INTERP_CUBICSPLINE` Hermite spline evaluation (`[in_tan, vertex, out_tan]`).
- Zero-allocation topological `SkeletonInstance` (`parents` + `topo_order` cached
  once on bind/resize, flat linear FK in `update_matrices`) plus
  `prev_world_matrices` / `snapshot_prev_matrices` for exact motion vectors and
  non-mutating `Pose` / `SkeletonInstance` helpers.
- Bollo (GDC 2018) scalar-axis clamped rotational `Inertializer` (monotonic angle
  decay with zero pi-overshoot), scale inertialization, fast `is_settled` early-out
  paying zero per-bone trig when settled, and `reparent_socket_inertial` for
  seamless world-space weapon/hardpoint re-parenting.
"""
from std.math import isfinite, sqrt
from flint.core.detmath import det_sin, det_cos, det_acos
from flint.core.math import Vec3
from flint.core.serial import BlobReader, BlobWriter
from flint.core.transform import Quat, Mat4, trs
from flint.core.springs import decay_spring, decay_spring_scalar
from flint.core.ik import solve_two_bone_pole
from flint.assets.model import (
    NativeModel, INTERP_STEP, INTERP_CUBICSPLINE,
)


@always_inline
def det_quat_to_scaled_angle(quat: Quat) -> Vec3:
    """Deterministic quaternion logarithm doubled (axis * angle in [0, pi])."""
    var q = quat.normalized().shortest()
    var length = sqrt(q.x * q.x + q.y * q.y + q.z * q.z)
    if length < Float32(0.00001):
        return Vec3(q.x * Float32(2.0), q.y * Float32(2.0), q.z * Float32(2.0))
    var half = det_acos(max(Float32(-1.0), min(Float32(1.0), q.w)))
    var scale = Float32(2.0) * half / length
    return Vec3(q.x * scale, q.y * scale, q.z * scale)


@always_inline
def det_quat_from_scaled_angle(value: Vec3) -> Quat:
    """Deterministic quaternion exponential from scaled-angle vector."""
    var angle = value.length()
    if angle < Float32(0.00001):
        return Quat(
            value.x * Float32(0.5),
            value.y * Float32(0.5),
            value.z * Float32(0.5),
            Float32(1.0),
        ).normalized()
    var half = angle * Float32(0.5)
    var scale = det_sin(half) / angle
    return Quat(value.x * scale, value.y * scale, value.z * scale, det_cos(half))


def invert_affine_mat4(m: Mat4) -> Mat4:
    """Exact inverse of an affine 4x4 matrix `[A, t; 0, 1]` via 3x3 cofactor."""
    var a00 = m.at(0, 0)
    var a01 = m.at(0, 1)
    var a02 = m.at(0, 2)
    var a10 = m.at(1, 0)
    var a11 = m.at(1, 1)
    var a12 = m.at(1, 2)
    var a20 = m.at(2, 0)
    var a21 = m.at(2, 1)
    var a22 = m.at(2, 2)

    var c00 = a11 * a22 - a12 * a21
    var c01 = a12 * a20 - a10 * a22
    var c02 = a10 * a21 - a11 * a20
    var det = a00 * c00 + a01 * c01 + a02 * c02
    if abs(det) < Float32(1.0e-12):
        return Mat4()
    var inv_det = Float32(1.0) / det

    var c10 = a02 * a21 - a01 * a22
    var c11 = a00 * a22 - a02 * a20
    var c12 = a01 * a20 - a00 * a21
    var c20 = a01 * a12 - a02 * a11
    var c21 = a02 * a10 - a00 * a12
    var c22 = a00 * a11 - a01 * a10

    var out = Mat4()
    out.set(0, 0, c00 * inv_det)
    out.set(0, 1, c10 * inv_det)
    out.set(0, 2, c20 * inv_det)
    out.set(1, 0, c01 * inv_det)
    out.set(1, 1, c11 * inv_det)
    out.set(1, 2, c21 * inv_det)
    out.set(2, 0, c02 * inv_det)
    out.set(2, 1, c12 * inv_det)
    out.set(2, 2, c22 * inv_det)

    var tx = m.at(0, 3)
    var ty = m.at(1, 3)
    var tz = m.at(2, 3)
    out.set(0, 3, -(out.at(0, 0) * tx + out.at(0, 1) * ty + out.at(0, 2) * tz))
    out.set(1, 3, -(out.at(1, 0) * tx + out.at(1, 1) * ty + out.at(1, 2) * tz))
    out.set(2, 3, -(out.at(2, 0) * tx + out.at(2, 1) * ty + out.at(2, 2) * tz))
    return out^


def decompose_trs(m: Mat4) -> Tuple[Vec3, Quat, Vec3]:
    """Decomposes an affine TRS matrix into `(translation, rotation, scale)`."""
    var t = Vec3(m.at(0, 3), m.at(1, 3), m.at(2, 3))
    var c0 = Vec3(m.at(0, 0), m.at(1, 0), m.at(2, 0))
    var c1 = Vec3(m.at(0, 1), m.at(1, 1), m.at(2, 1))
    var c2 = Vec3(m.at(0, 2), m.at(1, 2), m.at(2, 2))
    var sx = c0.length()
    var sy = c1.length()
    var sz = c2.length()
    if sx < Float32(1.0e-6) or sy < Float32(1.0e-6) or sz < Float32(1.0e-6):
        return (t, Quat(), Vec3(sx, sy, sz))
    if c0.dot(c1.cross(c2)) < Float32(0.0):
        sx = -sx
    var r0 = c0 * (Float32(1.0) / sx)
    var r1 = c1 * (Float32(1.0) / sy)
    var r2 = c2 * (Float32(1.0) / sz)

    var trace = r0.x + r1.y + r2.z
    var q: Quat
    if trace > Float32(0.0):
        var s = sqrt(trace + Float32(1.0)) * Float32(2.0)
        q = Quat((r1.z - r2.y) / s, (r2.x - r0.z) / s, (r0.y - r1.x) / s, Float32(0.25) * s).normalized()
    elif r0.x > r1.y and r0.x > r2.z:
        var s = sqrt(max(Float32(1.0e-6), Float32(1.0) + r0.x - r1.y - r2.z)) * Float32(2.0)
        q = Quat(Float32(0.25) * s, (r0.y + r1.x) / s, (r2.x + r0.z) / s, (r1.z - r2.y) / s).normalized()
    elif r1.y > r2.z:
        var s = sqrt(max(Float32(1.0e-6), Float32(1.0) + r1.y - r0.x - r2.z)) * Float32(2.0)
        q = Quat((r0.y + r1.x) / s, Float32(0.25) * s, (r1.z + r2.y) / s, (r2.x - r0.z) / s).normalized()
    else:
        var s = sqrt(max(Float32(1.0e-6), Float32(1.0) + r2.z - r0.x - r1.y)) * Float32(2.0)
        q = Quat((r2.x + r0.z) / s, (r1.z + r2.y) / s, Float32(0.25) * s, (r0.y - r1.x) / s).normalized()
    return (t, q, Vec3(sx, sy, sz))


struct Pose(Copyable, Movable):
    var morphs: Dict[Int, List[Float32]]
    var t: List[Vec3]
    var r: List[Quat]
    var s: List[Vec3]

    def __init__(out self, count: Int = 0):
        self.morphs = Dict[Int, List[Float32]]()
        self.t = List[Vec3](length=count, fill=Vec3())
        self.r = List[Quat](length=count, fill=Quat())
        self.s = List[Vec3](length=count, fill=Vec3(1, 1, 1))

    def count(self) -> Int:
        return len(self.t)

    def resize(mut self, count: Int):
        if len(self.t) != count:
            self = Pose(count)

    def rest(mut self, model: NativeModel):
        """The model's rest pose (each node's base transform)."""
        self.resize(len(model.nodes))
        self.morphs.clear()
        for index in range(len(model.nodes)):
            if len(model.nodes[index].base_morph_weights) > 0:
                self.morphs[index] = model.nodes[index].base_morph_weights.copy()
            self.t[index] = model.nodes[index].base_translation
            self.r[index] = model.nodes[index].base_rotation
            self.s[index] = model.nodes[index].base_scale

    def read(mut self, model: NativeModel):
        """The model's current node transforms."""
        self.resize(len(model.nodes))
        self.morphs.clear()
        for index in range(len(model.nodes)):
            if len(model.nodes[index].morph_weights) > 0:
                self.morphs[index] = model.nodes[index].morph_weights.copy()
            self.t[index] = model.nodes[index].translation
            self.r[index] = model.nodes[index].rotation
            self.s[index] = model.nodes[index].scale

    def assign(mut self, other: Pose):
        self.resize(other.count())
        if len(other.morphs) > 0:
            self.morphs = other.morphs.copy()
        else:
            self.morphs.clear()
        for index in range(other.count()):
            self.t[index] = other.t[index]
            self.r[index] = other.r[index]
            self.s[index] = other.s[index]


def _topology_hash(model: NativeModel) -> UInt64:
    """Zero-allocation FNV-1a signature over node hierarchy edges."""
    var h = UInt64(14695981039346656037)
    var n = len(model.nodes)
    h = (h ^ UInt64(n)) * UInt64(1099511628211)
    for r in model.roots:
        h = (h ^ UInt64(r + 257)) * UInt64(1099511628211)
    for i in range(n):
        h = (h ^ UInt64(i + 1)) * UInt64(1099511628211)
        for ch in model.nodes[i].children:
            h = (h ^ UInt64(ch + 513)) * UInt64(1099511628211)
    return h


struct SkeletonInstance(Copyable, Movable):
    """Decoupled runtime skeleton instance: holds per-instance node transforms,
    topological cache, current & previous-frame world matrices without mutating
    the shared `NativeModel` asset."""
    var pose: Pose
    var world_matrices: List[Mat4]
    var prev_world_matrices: List[Mat4]
    var prev_morphs: Dict[Int, List[Float32]]
    var has_prev: Bool
    var node_count: Int
    var parents: List[Int]
    var topo_order: List[Int]
    var topo_signature: UInt64
    var socket_delta_t: Vec3
    var socket_delta_r: Quat

    def __init__(out self, count: Int = 0):
        self.pose = Pose(count)
        self.world_matrices = List[Mat4](length=count, fill=Mat4())
        self.prev_morphs = Dict[Int, List[Float32]]()
        self.prev_world_matrices = List[Mat4](length=count, fill=Mat4())
        self.has_prev = False
        self.node_count = count
        self.parents = List[Int](length=count, fill=-1)
        self.topo_order = List[Int](capacity=count)
        for i in range(count):
            self.topo_order.append(i)
        self.topo_signature = 0
        self.socket_delta_t = Vec3()
        self.socket_delta_r = Quat()

    def resize(mut self, count: Int):
        if self.node_count == count and len(self.world_matrices) == count and len(self.prev_world_matrices) == count:
            return
        self.node_count = count
        self.pose.resize(count)
        self.world_matrices = List[Mat4](length=count, fill=Mat4())
        self.prev_world_matrices = List[Mat4](length=count, fill=Mat4())
        self.has_prev = False
        self.parents = List[Int](length=count, fill=-1)
        self.topo_order = List[Int](capacity=count)
        for i in range(count):
            self.topo_order.append(i)
        self.topo_signature = 0

    def ensure_topology(mut self, model: NativeModel):
        """Builds `parents` and parent-before-child `topo_order` once per rig."""
        var n = len(model.nodes)
        var sig = _topology_hash(model)
        if self.node_count == n and self.topo_signature == sig and len(self.topo_order) == n and len(self.parents) == n:
            return
        self.resize(n)
        self.topo_signature = sig
        for i in range(n):
            self.parents[i] = -1
        for i in range(n):
            for ch in model.nodes[i].children:
                if ch >= 0 and ch < n and ch != i:
                    self.parents[ch] = i

        self.topo_order.clear()
        var visited = List[Bool](length=n, fill=False)
        for root in model.roots:
            if root >= 0 and root < n and self.parents[root] < 0 and not visited[root]:
                visited[root] = True
                self.topo_order.append(root)
        for i in range(n):
            if self.parents[i] < 0 and not visited[i]:
                visited[i] = True
                self.topo_order.append(i)
        var cursor = 0
        while cursor < len(self.topo_order):
            var u = self.topo_order[cursor]
            cursor += 1
            for ch in model.nodes[u].children:
                if ch >= 0 and ch < n and not visited[ch]:
                    visited[ch] = True
                    self.topo_order.append(ch)
        for i in range(n):
            if not visited[i]:
                visited[i] = True
                self.topo_order.append(i)

    def snapshot_prev_matrices(mut self):
        """Captures current `world_matrices` into `prev_world_matrices` with zero allocations."""
        if len(self.pose.morphs) > 0:
            self.prev_morphs = self.pose.morphs.copy()
        else:
            self.prev_morphs.clear()
        var n = len(self.world_matrices)
        if len(self.prev_world_matrices) != n:
            self.prev_world_matrices = List[Mat4](length=n, fill=Mat4())
        for i in range(n):
            self.prev_world_matrices[i] = self.world_matrices[i].copy()
        self.has_prev = True

    def reset_to_rest(mut self, model: NativeModel):
        self.ensure_topology(model)
        self.pose.rest(model)
        self.has_prev = False
        self.update_matrices(model)

    def apply_pose(mut self, new_pose: Pose, model: NativeModel):
        self.ensure_topology(model)
        self.pose.assign(new_pose)
        self.update_matrices(model)

    def update_matrices(mut self, model: NativeModel):
        """Zero-allocation, non-recursive topological forward kinematics."""
        self.ensure_topology(model)
        if self.pose.count() != self.node_count:
            self.pose.resize(self.node_count)
        for idx in range(self.node_count):
            var node = self.topo_order[idx]
            var local = trs(self.pose.t[node], self.pose.r[node], self.pose.s[node])
            var p = self.parents[node]
            if p >= 0:
                self.world_matrices[node] = self.world_matrices[p].multiply(local)
            else:
                self.world_matrices[node] = local^

    def global_position(self, node: Int) -> Vec3:
        """Model-space joint position read from `world_matrices` (or pose chain)."""
        if node < 0 or node >= self.node_count:
            return Vec3()
        if node < len(self.world_matrices):
            return Vec3(
                self.world_matrices[node].at(0, 3),
                self.world_matrices[node].at(1, 3),
                self.world_matrices[node].at(2, 3),
            )
        var pos = Vec3()
        var cur = node
        var guard = 0
        while cur >= 0 and guard < 128:
            pos = self.pose.t[cur] + self.pose.r[cur].rotate(pos)
            cur = self.parents[cur] if cur < len(self.parents) else -1
            guard += 1
        return pos

    def global_rotation(self, node: Int) -> Quat:
        """Model-space joint rotation computed directly from `self.pose` + `self.parents`."""
        if node < 0 or node >= self.pose.count():
            return Quat()
        var rot = Quat()
        var cur = node
        var guard = 0
        while cur >= 0 and guard < 128:
            rot = self.pose.r[cur].multiply(rot)
            cur = self.parents[cur] if cur < len(self.parents) else -1
            guard += 1
        return rot.normalized()

    def offset_node(mut self, node: Int, model_delta: Vec3, model: NativeModel):
        """Moves a joint in `self.pose` by a model-space offset without mutating `model`."""
        self.ensure_topology(model)
        if node < 0 or node >= self.node_count:
            return
        var parent = self.parents[node]
        var local = model_delta
        if parent >= 0:
            local = self.global_rotation(parent).conjugate().rotate(model_delta)
        self.pose.t[node] = self.pose.t[node] + local
        self.update_matrices(model)

    def rotate_node_model(mut self, node: Int, rotation: Quat, model: NativeModel):
        """Turns a joint in `self.pose` by a model-space rotation without mutating `model`."""
        self.ensure_topology(model)
        if node < 0 or node >= self.node_count:
            return
        var parent = self.parents[node]
        var parent_rot = self.global_rotation(parent) if parent >= 0 else Quat()
        var local_delta = parent_rot.conjugate().multiply(rotation).multiply(parent_rot)
        self.pose.r[node] = local_delta.multiply(self.pose.r[node]).normalized()
        self.update_matrices(model)

    def solve_two_bone(
        mut self,
        upper: Int,
        middle: Int,
        end: Int,
        target: Vec3,
        pole_pos: Vec3,
        model: NativeModel,
        weight: Float32 = 1.0,
    ):
        """Solves pole-tracked two-bone IK directly on `self.pose` without mutating `model`."""
        self.ensure_topology(model)
        if upper < 0 or middle < 0 or end < 0 or weight <= Float32(0.0):
            return
        var a = self.global_position(upper)
        var b = self.global_position(middle)
        var c = self.global_position(end)
        var goal = c + (target - c) * min(Float32(1.0), weight)
        var res = solve_two_bone_pole(
            a, b, c, goal, pole_pos,
            self.global_rotation(upper),
            self.global_rotation(middle),
        )
        self.pose.r[upper] = self.pose.r[upper].multiply(res[0]).normalized()
        self.pose.r[middle] = self.pose.r[middle].multiply(res[1]).normalized()
        self.update_matrices(model)


def write_pose(mut model: NativeModel, pose: Pose):
    """Puts the pose into the model's nodes (one revision bump)."""
    var count = min(len(model.nodes), pose.count())
    for index in range(count):
        model.nodes[index].translation = pose.t[index]
        model.nodes[index].rotation = pose.r[index]
        model.nodes[index].scale = pose.s[index]
    for pair in pose.morphs.items():
        if pair.key >= 0 and pair.key < count:
            model.nodes[pair.key].morph_weights = pair.value.copy()
    model.transforms_revision += 1


def sample_pose(model: NativeModel, clip: Int, time_seconds: Float32, mut pose: Pose):
    """Samples `clip` at `time_seconds` over the rest pose, supporting glTF 2.0
    `LINEAR`, `STEP` (`<=` binary search reaching final keyframe), and exact
    `CUBICSPLINE` Hermite tangents (`[in_tan, vertex, out_tan]`)."""
    pose.rest(model)
    if clip < 0 or clip >= len(model.animations):
        return
    ref animation = model.animations[clip]
    var dur = animation.duration if (isfinite(animation.duration) and animation.duration >= Float32(0.0)) else Float32(0.0)
    var safe_t = time_seconds if isfinite(time_seconds) else Float32(0.0)
    var time = max(Float32(0), min(dur, safe_t))
    for channel in animation.channels:
        var key_count = len(channel.times)
        if channel.node < 0 or channel.node >= pose.count() or key_count == 0:
            continue
        var lower = 0
        var next = 0
        if key_count > 1:
            if time <= channel.times[0]:
                lower = 0
                next = 0
            elif time >= channel.times[key_count - 1]:
                lower = key_count - 1
                next = lower
            else:
                var lower_bound = 0
                var upper_bound = key_count - 1
                while lower_bound + 1 < upper_bound:
                    var middle = (lower_bound + upper_bound) // 2
                    if channel.times[middle] <= time:
                        lower_bound = middle
                    else:
                        upper_bound = middle
                lower = lower_bound
                next = upper_bound

        var is_step = channel.step or channel.interp_mode == INTERP_STEP
        var is_cubic = (not is_step) and channel.interp_mode == INTERP_CUBICSPLINE
        var dt_seg = channel.times[next] - channel.times[lower]
        var amount = Float32(0)
        if not is_step and next != lower and dt_seg > Float32(0):
            amount = max(Float32(0), min(Float32(1), (time - channel.times[lower]) / dt_seg))

        if channel.path == 3:
            pose.morphs[channel.node] = channel.sample_weights(lower, next, amount, dt_seg)
            continue
        var components = channel.components()
        if is_cubic:
            var stride = components * 3
            var v0_at = lower * stride + components
            var out_tan0_at = lower * stride + components * 2
            var in_tan1_at = next * stride
            var v1_at = next * stride + components
            if v0_at + components > len(channel.values) or v1_at + components > len(channel.values):
                continue
            if next == lower or dt_seg <= Float32(0):
                if channel.path == 0:
                    pose.t[channel.node] = Vec3(channel.values[v0_at], channel.values[v0_at + 1], channel.values[v0_at + 2])
                elif channel.path == 1:
                    pose.r[channel.node] = Quat(channel.values[v0_at], channel.values[v0_at + 1], channel.values[v0_at + 2], channel.values[v0_at + 3]).normalized()
                else:
                    pose.s[channel.node] = Vec3(channel.values[v0_at], channel.values[v0_at + 1], channel.values[v0_at + 2])
            else:
                var s2 = amount * amount
                var s3 = s2 * amount
                var h00 = Float32(2.0) * s3 - Float32(3.0) * s2 + Float32(1.0)
                var h10 = (s3 - Float32(2.0) * s2 + amount) * dt_seg
                var h01 = Float32(-2.0) * s3 + Float32(3.0) * s2
                var h11 = (s3 - s2) * dt_seg
                if channel.path == 0:
                    pose.t[channel.node] = Vec3(
                        h00 * channel.values[v0_at] + h10 * channel.values[out_tan0_at] + h01 * channel.values[v1_at] + h11 * channel.values[in_tan1_at],
                        h00 * channel.values[v0_at + 1] + h10 * channel.values[out_tan0_at + 1] + h01 * channel.values[v1_at + 1] + h11 * channel.values[in_tan1_at + 1],
                        h00 * channel.values[v0_at + 2] + h10 * channel.values[out_tan0_at + 2] + h01 * channel.values[v1_at + 2] + h11 * channel.values[in_tan1_at + 2],
                    )
                elif channel.path == 1:
                    pose.r[channel.node] = Quat(
                        h00 * channel.values[v0_at] + h10 * channel.values[out_tan0_at] + h01 * channel.values[v1_at] + h11 * channel.values[in_tan1_at],
                        h00 * channel.values[v0_at + 1] + h10 * channel.values[out_tan0_at + 1] + h01 * channel.values[v1_at + 1] + h11 * channel.values[in_tan1_at + 1],
                        h00 * channel.values[v0_at + 2] + h10 * channel.values[out_tan0_at + 2] + h01 * channel.values[v1_at + 2] + h11 * channel.values[in_tan1_at + 2],
                        h00 * channel.values[v0_at + 3] + h10 * channel.values[out_tan0_at + 3] + h01 * channel.values[v1_at + 3] + h11 * channel.values[in_tan1_at + 3],
                    ).normalized()
                else:
                    pose.s[channel.node] = Vec3(
                        h00 * channel.values[v0_at] + h10 * channel.values[out_tan0_at] + h01 * channel.values[v1_at] + h11 * channel.values[in_tan1_at],
                        h00 * channel.values[v0_at + 1] + h10 * channel.values[out_tan0_at + 1] + h01 * channel.values[v1_at + 1] + h11 * channel.values[in_tan1_at + 1],
                        h00 * channel.values[v0_at + 2] + h10 * channel.values[out_tan0_at + 2] + h01 * channel.values[v1_at + 2] + h11 * channel.values[in_tan1_at + 2],
                    )
            continue

        var first = lower * components
        var second = next * components
        if first + components > len(channel.values) or second + components > len(channel.values):
            continue
        if channel.path == 0:
            pose.t[channel.node] = Vec3(
                channel.values[first] + (channel.values[second] - channel.values[first]) * amount,
                channel.values[first + 1] + (channel.values[second + 1] - channel.values[first + 1]) * amount,
                channel.values[first + 2] + (channel.values[second + 2] - channel.values[first + 2]) * amount,
            )
        elif channel.path == 1:
            var a = Quat(channel.values[first], channel.values[first + 1], channel.values[first + 2], channel.values[first + 3]).normalized()
            var b = Quat(channel.values[second], channel.values[second + 1], channel.values[second + 2], channel.values[second + 3]).normalized()
            pose.r[channel.node] = a.slerp(b, amount)
        else:
            pose.s[channel.node] = Vec3(
                channel.values[first] + (channel.values[second] - channel.values[first]) * amount,
                channel.values[first + 1] + (channel.values[second + 1] - channel.values[first + 1]) * amount,
                channel.values[first + 2] + (channel.values[second + 2] - channel.values[first + 2]) * amount,
            )


def lerp3(a: Vec3, b: Vec3, w: Float32) -> Vec3:
    return a + (b - a) * w


def _blend_morph_weights(mut base: Pose, other: Pose, index: Int, weight: Float32):
    if index not in base.morphs or index not in other.morphs:
        return
    try:
        ref dst = base.morphs[index]
        ref src = other.morphs[index]
        for j in range(min(len(dst), len(src))):
            dst[j] += weight * (src[j] - dst[j])
    except:
        return


def blend_into(mut base: Pose, other: Pose, weight: Float32):
    """Blends base pose toward other by weight, across every node."""
    if not isfinite(weight) or weight <= 0:
        return
    var count = min(base.count(), other.count())
    for index in other.morphs.keys():
        _blend_morph_weights(base, other, index, min(Float32(1), weight))
    if weight >= 1:
        for index in range(count):
            base.t[index] = other.t[index]
            base.r[index] = other.r[index]
            base.s[index] = other.s[index]
        return
    for index in range(count):
        base.t[index] = lerp3(base.t[index], other.t[index], weight)
        base.r[index] = base.r[index].slerp(other.r[index], weight)
        base.s[index] = lerp3(base.s[index], other.s[index], weight)


def blend_masked(mut base: Pose, other: Pose, mask: List[Float32], weight: Float32):
    """An override layer: each node moves toward other by weight x its mask
    weight; nodes outside the mask keep the base."""
    if not isfinite(weight) or weight <= 0:
        return
    var count = min(min(base.count(), other.count()), len(mask))
    for index in range(count):
        var mw = mask[index]
        if not isfinite(mw):
            continue
        var w = min(Float32(1), mw * weight)
        if not isfinite(w) or w <= 0:
            continue
        if len(other.morphs) > 0:
            _blend_morph_weights(base, other, index, w)
        if w >= 1:
            base.t[index] = other.t[index]
            base.r[index] = other.r[index]
            base.s[index] = other.s[index]
            continue
        base.t[index] = lerp3(base.t[index], other.t[index], w)
        base.r[index] = base.r[index].slerp(other.r[index], w)
        base.s[index] = lerp3(base.s[index], other.s[index], w)


def add_additive(mut base: Pose, layer: Pose, reference: Pose, mask: List[Float32], weight: Float32):
    """An additive layer: layer's change from reference, times weight x the
    node's mask weight, on top of base (rotation: base * delta, translation
    added, scale multiplied by lerp(1, layer.s / ref.s, mw))."""
    if not isfinite(weight) or weight <= 0:
        return
    var count = min(min(base.count(), layer.count()), min(reference.count(), len(mask)))
    var identity = Quat()
    for index in range(count):
        var mw = mask[index]
        if not isfinite(mw):
            continue
        var w = mw * weight
        if not isfinite(w) or w <= 0:
            continue
        if index in layer.morphs and index in reference.morphs and index in base.morphs:
            try:
                ref dst = base.morphs[index]
                ref src = layer.morphs[index]
                ref rest = reference.morphs[index]
                for j in range(min(len(dst), min(len(src), len(rest)))):
                    dst[j] += w * (src[j] - rest[j])
            except:
                pass
        var delta = reference.r[index].conjugate().multiply(layer.r[index]).normalized()
        base.r[index] = base.r[index].multiply(identity.slerp(delta, w)).normalized()
        base.t[index] = base.t[index] + (layer.t[index] - reference.t[index]) * w
        var rx = layer.s[index].x / reference.s[index].x if abs(reference.s[index].x) > Float32(1.0e-5) else Float32(1.0)
        var ry = layer.s[index].y / reference.s[index].y if abs(reference.s[index].y) > Float32(1.0e-5) else Float32(1.0)
        var rz = layer.s[index].z / reference.s[index].z if abs(reference.s[index].z) > Float32(1.0e-5) else Float32(1.0)
        base.s[index] = Vec3(
            base.s[index].x * (Float32(1.0) + (rx - Float32(1.0)) * w),
            base.s[index].y * (Float32(1.0) + (ry - Float32(1.0)) * w),
            base.s[index].z * (Float32(1.0) + (rz - Float32(1.0)) * w),
        )


def full_mask(count: Int) -> List[Float32]:
    return List[Float32](length=count, fill=1)


struct Inertializer(Copyable, Movable):
    """Bollo (GDC 2018) scalar-axis clamped rotational + translational + scale
    inertializer with `is_settled` fast early-out.

    On transition, extracts scalar rotation angle `theta_0 >= 0` and unit axis
    `u = theta_vec / theta_0`, clamping initial angular velocity along `u` to
    `dtheta_0 = min(0, omega_diff . u)` and decaying `theta(t) >= 0` monotonically
    so `||theta(t)||` never overshoots `pi` or oscillates past zero.
    When `max_active_offset < 1e-5`, sets `is_settled = True` and pays zero
    per-bone trigonometric calls until the next state transition.
    """
    var has_pose: Bool
    var is_settled: Bool
    var has_prev_settled: Bool
    var last_dt: Float32
    var out_t: List[Vec3]
    var out_r: List[Quat]
    var out_s: List[Vec3]
    var prev_t: List[Vec3]
    var prev_r: List[Quat]
    var prev_s: List[Vec3]
    var out_vel_t: List[Vec3]
    var out_vel_r: List[Vec3]
    var out_vel_s: List[Vec3]
    var off_t: List[Vec3]
    var vel_t: List[Vec3]
    var off_r: List[Vec3]
    var vel_r: List[Vec3]
    var rot_axis: List[Vec3]
    var rot_ang: List[Float32]
    var rot_vel: List[Float32]
    var off_s: List[Vec3]
    var vel_s: List[Vec3]

    def __init__(out self):
        self.has_pose = False
        self.is_settled = True
        self.has_prev_settled = False
        self.last_dt = Float32(0.0166667)
        self.out_t = List[Vec3]()
        self.out_r = List[Quat]()
        self.out_s = List[Vec3]()
        self.prev_t = List[Vec3]()
        self.prev_r = List[Quat]()
        self.prev_s = List[Vec3]()
        self.out_vel_t = List[Vec3]()
        self.out_vel_r = List[Vec3]()
        self.out_vel_s = List[Vec3]()
        self.off_t = List[Vec3]()
        self.vel_t = List[Vec3]()
        self.off_r = List[Vec3]()
        self.vel_r = List[Vec3]()
        self.rot_axis = List[Vec3]()
        self.rot_ang = List[Float32]()
        self.rot_vel = List[Float32]()
        self.off_s = List[Vec3]()
        self.vel_s = List[Vec3]()

    def write_history(self, mut writer: BlobWriter):
        writer.boolean(self.has_pose)
        writer.boolean(self.is_settled)
        writer.boolean(self.has_prev_settled)
        writer.f32(self.last_dt)
        writer.integer(len(self.out_t))
        for i in range(len(self.out_t)):
            writer.vec3(self.out_t[i])
            writer.quat(self.out_r[i])
            writer.vec3(self.out_s[i])
            writer.vec3(self.prev_t[i])
            writer.quat(self.prev_r[i])
            writer.vec3(self.prev_s[i])
            writer.vec3(self.out_vel_t[i])
            writer.vec3(self.out_vel_r[i])
            writer.vec3(self.out_vel_s[i])
            writer.vec3(self.off_t[i])
            writer.vec3(self.vel_t[i])
            writer.vec3(self.off_r[i])
            writer.vec3(self.vel_r[i])
            writer.vec3(self.rot_axis[i])
            writer.f32(self.rot_ang[i])
            writer.f32(self.rot_vel[i])
            writer.vec3(self.off_s[i])
            writer.vec3(self.vel_s[i])

    def read_history(mut self, mut reader: BlobReader) raises:
        var has_pose = reader.boolean()
        var settled = reader.boolean()
        var previous_settled = reader.boolean()
        var dt = reader.f32()
        var count = reader.integer()
        # Each joint stores fourteen vectors, two quaternions and two scalars.
        if reader.failed or count < 0 or count > 65536 or count > (reader.limit() - reader.offset) // 208:
            raise Error("Invalid inertial pose history length")
        self.resize(count)
        self.has_pose = has_pose
        self.is_settled = settled
        self.has_prev_settled = previous_settled
        self.last_dt = dt
        for i in range(count):
            self.out_t[i] = reader.vec3()
            self.out_r[i] = reader.quat()
            self.out_s[i] = reader.vec3()
            self.prev_t[i] = reader.vec3()
            self.prev_r[i] = reader.quat()
            self.prev_s[i] = reader.vec3()
            self.out_vel_t[i] = reader.vec3()
            self.out_vel_r[i] = reader.vec3()
            self.out_vel_s[i] = reader.vec3()
            self.off_t[i] = reader.vec3()
            self.vel_t[i] = reader.vec3()
            self.off_r[i] = reader.vec3()
            self.vel_r[i] = reader.vec3()
            self.rot_axis[i] = reader.vec3()
            self.rot_ang[i] = reader.f32()
            self.rot_vel[i] = reader.f32()
            self.off_s[i] = reader.vec3()
            self.vel_s[i] = reader.vec3()

    def resize(mut self, count: Int):
        if len(self.out_t) == count:
            return
        self.out_t = List[Vec3](length=count, fill=Vec3())
        self.out_r = List[Quat](length=count, fill=Quat())
        self.out_s = List[Vec3](length=count, fill=Vec3(1, 1, 1))
        self.prev_t = List[Vec3](length=count, fill=Vec3())
        self.prev_r = List[Quat](length=count, fill=Quat())
        self.prev_s = List[Vec3](length=count, fill=Vec3(1, 1, 1))
        self.out_vel_t = List[Vec3](length=count, fill=Vec3())
        self.out_vel_r = List[Vec3](length=count, fill=Vec3())
        self.out_vel_s = List[Vec3](length=count, fill=Vec3())
        self.off_t = List[Vec3](length=count, fill=Vec3())
        self.vel_t = List[Vec3](length=count, fill=Vec3())
        self.off_r = List[Vec3](length=count, fill=Vec3())
        self.vel_r = List[Vec3](length=count, fill=Vec3())
        self.rot_axis = List[Vec3](length=count, fill=Vec3(0, 1, 0))
        self.rot_ang = List[Float32](length=count, fill=0.0)
        self.rot_vel = List[Float32](length=count, fill=0.0)
        self.off_s = List[Vec3](length=count, fill=Vec3())
        self.vel_s = List[Vec3](length=count, fill=Vec3())
        self.has_pose = False
        self.is_settled = True
        self.has_prev_settled = False

    def reset(mut self):
        """Forget the last output: the next pose is shown as it is."""
        self.has_pose = False
        self.is_settled = True
        self.has_prev_settled = False
        for i in range(len(self.off_t)):
            self.off_t[i] = Vec3()
            self.vel_t[i] = Vec3()
            self.off_r[i] = Vec3()
            self.vel_r[i] = Vec3()
            self.rot_ang[i] = Float32(0.0)
            self.rot_vel[i] = Float32(0.0)
            self.off_s[i] = Vec3()
            self.vel_s[i] = Vec3()

    def apply(mut self, mut pose: Pose, ahead: Pose, switching: Bool, halflife: Float32, dt: Float32):
        """Applies Bollo scalar-axis clamped inertialization to `pose` (`t`, `r`, `s`)."""
        var count = pose.count()
        self.resize(count)
        var switch_now = switching and self.has_pose and halflife > Float32(0.0) and dt > Float32(0.0) and ahead.count() == count

        # Fast settled early-out: zero per-bone trig!
        if self.has_pose and not switch_now and self.is_settled:
            if dt > Float32(0.0):
                self.last_dt = dt
                self.has_prev_settled = True
                for index in range(count):
                    self.prev_t[index] = self.out_t[index]
                    self.prev_r[index] = self.out_r[index]
                    self.prev_s[index] = self.out_s[index]
                    self.out_t[index] = pose.t[index]
                    self.out_r[index] = pose.r[index]
                    self.out_s[index] = pose.s[index]
            else:
                for index in range(count):
                    self.out_t[index] = pose.t[index]
                    self.out_r[index] = pose.r[index]
                    self.out_s[index] = pose.s[index]
            return

        if switch_now:
            # Reconstruct output velocities on-demand if we were previously in settled fast-path
            if self.is_settled and self.has_prev_settled and self.last_dt > Float32(0.0):
                var inv_last = Float32(1.0) / self.last_dt
                for index in range(count):
                    self.out_vel_t[index] = (self.out_t[index] - self.prev_t[index]) * inv_last
                    self.out_vel_r[index] = det_quat_to_scaled_angle(
                        self.out_r[index].multiply(self.prev_r[index].conjugate())
                    ) * inv_last
                    self.out_vel_s[index] = (self.out_s[index] - self.prev_s[index]) * inv_last

            var inverse_dt = Float32(1.0) / dt
            for index in range(count):
                var src_t = pose.t[index]
                var src_r = pose.r[index]
                var src_s = pose.s[index]
                var src_vel_t = (ahead.t[index] - src_t) * inverse_dt
                var src_vel_r = det_quat_to_scaled_angle(ahead.r[index].multiply(src_r.conjugate())) * inverse_dt
                var src_vel_s = (ahead.s[index] - src_s) * inverse_dt

                self.off_t[index] = self.out_t[index] - src_t
                self.vel_t[index] = self.out_vel_t[index] - src_vel_t
                self.off_s[index] = self.out_s[index] - src_s
                self.vel_s[index] = self.out_vel_s[index] - src_vel_s

                var theta_vec = det_quat_to_scaled_angle(self.out_r[index].multiply(src_r.conjugate()))
                var theta_0 = theta_vec.length()
                var omega_diff = self.out_vel_r[index] - src_vel_r
                if theta_0 > Float32(1.0e-6):
                    var u = theta_vec * (Float32(1.0) / theta_0)
                    var dtheta_0 = min(Float32(0.0), omega_diff.dot(u))
                    self.rot_axis[index] = u
                    self.rot_ang[index] = theta_0
                    self.rot_vel[index] = dtheta_0
                    self.off_r[index] = theta_vec
                    self.vel_r[index] = u * dtheta_0
                else:
                    self.rot_axis[index] = Vec3(0, 1, 0)
                    self.rot_ang[index] = Float32(0.0)
                    self.rot_vel[index] = Float32(0.0)
                    self.off_r[index] = Vec3()
                    self.vel_r[index] = Vec3()
            self.is_settled = False
        elif not self.has_pose:
            for index in range(count):
                self.off_t[index] = Vec3()
                self.vel_t[index] = Vec3()
                self.off_r[index] = Vec3()
                self.vel_r[index] = Vec3()
                self.rot_axis[index] = Vec3(0, 1, 0)
                self.rot_ang[index] = Float32(0.0)
                self.rot_vel[index] = Float32(0.0)
                self.off_s[index] = Vec3()
                self.vel_s[index] = Vec3()
                self.out_vel_t[index] = Vec3()
                self.out_vel_r[index] = Vec3()
                self.out_vel_s[index] = Vec3()
                self.out_t[index] = pose.t[index]
                self.out_r[index] = pose.r[index]
                self.out_s[index] = pose.s[index]
                self.prev_t[index] = pose.t[index]
                self.prev_r[index] = pose.r[index]
                self.prev_s[index] = pose.s[index]
            self.has_pose = True
            self.is_settled = True
            self.has_prev_settled = False
            if dt > Float32(0.0):
                self.last_dt = dt
            return

        var max_active_offset = Float32(0.0)
        for index in range(count):
            if dt > Float32(0.0) and halflife > Float32(0.0):
                decay_spring(self.off_t[index], self.vel_t[index], halflife, dt)
                decay_spring(self.off_s[index], self.vel_s[index], halflife, dt)
                decay_spring_scalar(self.rot_ang[index], self.rot_vel[index], halflife, dt)
                # Bollo non-negativity clamp prevents scalar angle from crossing 0
                if self.rot_ang[index] <= Float32(0.0):
                    self.rot_ang[index] = Float32(0.0)
                    self.rot_vel[index] = Float32(0.0)
                self.off_r[index] = self.rot_axis[index] * self.rot_ang[index]
                self.vel_r[index] = self.rot_axis[index] * self.rot_vel[index]

            var off_t_len = self.off_t[index].length()
            var off_s_len = self.off_s[index].length()
            var ang_mag = self.rot_ang[index]
            max_active_offset = max(max_active_offset, max(off_t_len, max(ang_mag, off_s_len)))

            var final_t = pose.t[index] + self.off_t[index]
            var final_r = pose.r[index]
            if ang_mag > Float32(1.0e-7):
                final_r = det_quat_from_scaled_angle(self.off_r[index]).multiply(pose.r[index]).normalized()
            var final_s = pose.s[index] + self.off_s[index]

            if self.has_pose and dt > Float32(0.0):
                var inv_dt = Float32(1.0) / dt
                self.out_vel_t[index] = (final_t - self.out_t[index]) * inv_dt
                self.out_vel_r[index] = det_quat_to_scaled_angle(final_r.multiply(self.out_r[index].conjugate())) * inv_dt
                self.out_vel_s[index] = (final_s - self.out_s[index]) * inv_dt
            else:
                self.out_vel_t[index] = Vec3()
                self.out_vel_r[index] = Vec3()
                self.out_vel_s[index] = Vec3()

            self.prev_t[index] = self.out_t[index]
            self.prev_r[index] = self.out_r[index]
            self.prev_s[index] = self.out_s[index]
            self.out_t[index] = final_t
            self.out_r[index] = final_r
            self.out_s[index] = final_s
            pose.t[index] = final_t
            pose.r[index] = final_r
            pose.s[index] = final_s

        if dt > Float32(0.0):
            self.last_dt = dt
            self.has_prev_settled = True
        if max_active_offset < Float32(1.0e-5):
            for index in range(count):
                self.off_t[index] = Vec3()
                self.vel_t[index] = Vec3()
                self.off_r[index] = Vec3()
                self.vel_r[index] = Vec3()
                self.rot_ang[index] = Float32(0.0)
                self.rot_vel[index] = Float32(0.0)
                self.off_s[index] = Vec3()
                self.vel_s[index] = Vec3()
            self.is_settled = True
        self.has_pose = True

    def largest_offset(self) -> Float32:
        """The largest remaining offset: metres of translation, radians of rotation, or scale delta."""
        if self.is_settled:
            return Float32(0.0)
        var largest = Float32(0.0)
        for index in range(len(self.off_t)):
            largest = max(largest, max(self.off_t[index].length(), max(self.rot_ang[index], self.off_s[index].length())))
        return largest


def reparent_socket_inertial(
    mut skel: SkeletonInstance,
    child_node: Int,
    new_parent_world: Mat4,
    local_offset: Mat4,
):
    """Computes exact world-space continuity when re-parenting an articulated socket
    or weapon (`child_node`, e.g. VF-1 gunpod switching from ventral hardpoint to hand)
    under `new_parent_world` toward target `local_offset`.

    Sets `skel.pose` at `child_node` to `inv(new_parent_world) * old_world` so
    `new_parent_world * trs(pose[child_node]) == old_world` with zero world pop,
    and records the continuity delta `(socket_delta_t, socket_delta_r)` relative
    to `local_offset`.
    """
    if child_node < 0 or child_node >= skel.node_count or child_node >= len(skel.world_matrices):
        return
    var w_old = skel.world_matrices[child_node].copy()
    var inv_parent = invert_affine_mat4(new_parent_world)
    var l_cont = inv_parent.multiply(w_old)
    var cont_trs = decompose_trs(l_cont)
    var tgt_trs = decompose_trs(local_offset)

    skel.socket_delta_t = cont_trs[0] - tgt_trs[0]
    skel.socket_delta_r = cont_trs[1].multiply(tgt_trs[1].conjugate()).normalized()

    skel.pose.t[child_node] = cont_trs[0]
    skel.pose.r[child_node] = cont_trs[1]
    skel.pose.s[child_node] = cont_trs[2]
    skel.world_matrices[child_node] = new_parent_world.multiply(trs(cont_trs[0], cont_trs[1], cont_trs[2]))


def reparent_socket_inertial(
    mut skel: SkeletonInstance,
    mut inert: Inertializer,
    child_node: Int,
    new_parent_world: Mat4,
    local_offset: Mat4,
):
    """Re-parents `child_node` under `new_parent_world` and seeds `inert` so subsequent
    `inert.apply` calls smoothly decay `child_node` from its old world transform onto
    `new_parent_world * local_offset`."""
    reparent_socket_inertial(skel, child_node, new_parent_world, local_offset)
    if child_node < 0 or child_node >= skel.node_count:
        return
    inert.resize(skel.node_count)
    var tgt_trs = decompose_trs(local_offset)
    var cont_t = skel.pose.t[child_node]
    var cont_r = skel.pose.r[child_node]
    var cont_s = skel.pose.s[child_node]

    if not inert.has_pose:
        for i in range(skel.node_count):
            inert.out_t[i] = skel.pose.t[i]
            inert.out_r[i] = skel.pose.r[i]
            inert.out_s[i] = skel.pose.s[i]
            inert.prev_t[i] = skel.pose.t[i]
            inert.prev_r[i] = skel.pose.r[i]
            inert.prev_s[i] = skel.pose.s[i]
        inert.has_pose = True

    inert.off_t[child_node] = cont_t - tgt_trs[0]
    inert.vel_t[child_node] = Vec3()
    inert.off_s[child_node] = cont_s - tgt_trs[2]
    inert.vel_s[child_node] = Vec3()

    var theta_vec = det_quat_to_scaled_angle(cont_r.multiply(tgt_trs[1].conjugate()))
    var theta_0 = theta_vec.length()
    if theta_0 > Float32(1.0e-6):
        inert.rot_axis[child_node] = theta_vec * (Float32(1.0) / theta_0)
        inert.rot_ang[child_node] = theta_0
    else:
        inert.rot_axis[child_node] = Vec3(0, 1, 0)
        inert.rot_ang[child_node] = Float32(0.0)
    inert.rot_vel[child_node] = Float32(0.0)
    inert.off_r[child_node] = theta_vec
    inert.vel_r[child_node] = Vec3()
    inert.out_t[child_node] = cont_t
    inert.out_r[child_node] = cont_r
    inert.out_s[child_node] = cont_s
    inert.is_settled = False


def bone_mask(model: NativeModel, bones: List[String], weights: List[Float32], children: Bool) -> List[Float32]:
    """Per-node weights for a layer: each named bone gets its weight and,
    with children, every descendant without its own entry takes its
    nearest named ancestor's."""
    var count = len(model.nodes)
    var mask = List[Float32](length=count, fill=0)
    var named = List[Bool](length=count, fill=False)
    for index in range(min(len(bones), len(weights))):
        var node = model.find_node(bones[index])
        if node >= 0:
            mask[node] = weights[index]
            named[node] = True
    if not children:
        return mask^
    var parents = List[Int](length=count, fill=-1)
    for index in range(count):
        for child in model.nodes[index].children:
            if child >= 0 and child < count:
                parents[child] = index
    for index in range(count):
        if named[index]:
            continue
        var current = parents[index]
        var guard = 0
        while current >= 0 and guard < 128:
            if named[current]:
                mask[index] = mask[current]
                break
            current = parents[current]
            guard += 1
    return mask^
