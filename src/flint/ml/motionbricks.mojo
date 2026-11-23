"""MotionBricks motion inbetweening (SOMA body model) in pure Mojo on `flint.ml.onnx`.

A port of NVIDIA's MotionBricks runtime (https://github.com/NVlabs/motionbricks, Apache-2.0,
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES): its SOMA pose contract, the encoding of
context and target poses, the decoding of the generated motion and the endpoint correction.
The model itself (`mmm_v6_soma_fp32.onnx`, NVIDIA Open Model Agreement) is not part of Flint;
`MotionBricks.load()` reads it from a MotionBricks checkout or download.

Each generation fills in the motion between four context poses (the character's most recent
poses, 30 Hz) and one to four target poses, producing 0.6 to 1.9 s of motion at 30 Hz plus four
foot-contact curves. Chaining generations, each starting from the last four poses played, makes
continuous motion; the application chooses the targets.

Poses are local joint transforms (`MbXform`) of a `MbSkeleton` whose root (index 0) is the
SOMA `Hips` joint, in the model's space: Y up, metres.
"""

from std.math import sqrt, atan2, sin, cos, floor, isfinite, isnan
from flint.ml.onnx import OnnxModel
from flint.ml.tensor import Tensor, DT_F32, DT_I32
from flint.platform.os import read_bytes

comptime MB_SAMPLE_RATE: Float32 = 30.0
comptime MB_POSES_PER_TOKEN: Int = 4
comptime MB_CONTEXT_AND_TARGET: Int = 8
comptime MB_MIN_TOKENS: Int = 6
comptime MB_MAX_TOKENS: Int = 16
comptime MB_BOUNDARY_FRAMES: Int = 3
comptime MB_MAX_FRAMES: Int = MB_MAX_TOKENS * MB_POSES_PER_TOKEN - 2 * MB_BOUNDARY_FRAMES
comptime MB_FOOT_CONTACTS: Int = 4
comptime MB_POSE_VALUES: Int = 267


def soma_joint_names() -> List[String]:
    """The 30 joints of the SOMA model, in model order."""
    return [
        "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
        "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
    ]


# ------------------------------------------------------------------------------------- math
# Warp-compatible float32 quaternion/transform math, so results match the reference runtime.

@fieldwise_init
struct MbVec3(Copyable, ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32

    def __add__(self, o: Self) -> Self:
        return Self(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: Self) -> Self:
        return Self(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s: Float32) -> Self:
        return Self(self.x * s, self.y * s, self.z * s)

    def length(self) -> Float32:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> Self:
        var l = self.length()
        if l > 0:
            return self * (1.0 / l)
        return Self(0, 0, 0)


def _cross(a: MbVec3, b: MbVec3) -> MbVec3:
    return MbVec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def _lerp3(a: MbVec3, b: MbVec3, t: Float32) -> MbVec3:
    if not isfinite(t):
        return a
    return a + (b - a) * t


@fieldwise_init
struct MbQuat(Copyable, ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32

    @staticmethod
    def identity() -> Self:
        return Self(0, 0, 0, 1)

    def __mul__(self, o: Self) -> Self:
        return Self(
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
        )

    def inverse(self) -> Self:
        return Self(-self.x, -self.y, -self.z, self.w)

    def normalized(self) -> Self:
        var l = sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)
        if l > 0:
            var s = 1.0 / l
            return Self(self.x * s, self.y * s, self.z * s, self.w * s)
        return Self(0, 0, 0, 0)

    def rotate(self, v: MbVec3) -> MbVec3:
        var u = MbVec3(self.x, self.y, self.z)
        var c = _cross(u, v)
        return v * (2.0 * self.w * self.w - 1.0) + c * (2.0 * self.w) + u * (2.0 * (u.x * v.x + u.y * v.y + u.z * v.z))

    def rotate_inv(self, v: MbVec3) -> MbVec3:
        return self.inverse().rotate(v)


def quat_axis_angle(axis: MbVec3, angle: Float32) -> MbQuat:
    var s = sin(angle * 0.5)
    var n = axis.normalized()
    return MbQuat(n.x * s, n.y * s, n.z * s, cos(angle * 0.5))


def _mb_quat_finite(q: MbQuat) -> Bool:
    return isfinite(q.x) and isfinite(q.y) and isfinite(q.z) and isfinite(q.w)


def quat_slerp(q0: MbQuat, q1: MbQuat, t: Float32) -> MbQuat:
    """Warp's quat_slerp: q0 * exp(t * log(q0^-1 q1)), the short way round."""
    var sq0 = q0.normalized() if _mb_quat_finite(q0) else MbQuat.identity()
    if sq0.x == 0 and sq0.y == 0 and sq0.z == 0 and sq0.w == 0:
        sq0 = MbQuat.identity()
    var sq1 = q1.normalized() if _mb_quat_finite(q1) else sq0
    if sq1.x == 0 and sq1.y == 0 and sq1.z == 0 and sq1.w == 0:
        sq1 = sq0
    if not isfinite(t):
        return sq0
    var d = sq0.inverse() * sq1
    var v = MbVec3(d.x, d.y, d.z)
    var axis = v.normalized()
    if d.w < 0:
        axis = axis * -1.0
    var angle = 2.0 * atan2(v.length(), abs(d.w))
    var res = (sq0 * quat_axis_angle(axis, t * angle)).normalized()
    if not _mb_quat_finite(res) or (res.x == 0 and res.y == 0 and res.z == 0 and res.w == 0):
        return sq0
    return res


def quat_from_basis(x: MbVec3, y: MbVec3, z: MbVec3) -> MbQuat:
    """Warp's quat_from_matrix for the matrix with columns x, y, z."""
    var m00 = x.x
    var m10 = x.y
    var m20 = x.z
    var m01 = y.x
    var m11 = y.y
    var m21 = y.z
    var m02 = z.x
    var m12 = z.y
    var m22 = z.z
    var tr = m00 + m11 + m22
    var qx: Float32
    var qy: Float32
    var qz: Float32
    var qw: Float32
    if tr >= 0:
        var h = sqrt(tr + 1.0)
        qw = 0.5 * h
        h = 0.5 / h
        qx = (m21 - m12) * h
        qy = (m02 - m20) * h
        qz = (m10 - m01) * h
    else:
        var md = 0
        if m11 > m00:
            md = 1
        if m22 > (m11 if md == 1 else m00):
            md = 2
        if md == 0:
            var h = sqrt((m00 - (m11 + m22)) + 1.0)
            qx = 0.5 * h
            h = 0.5 / h
            qy = (m01 + m10) * h
            qz = (m20 + m02) * h
            qw = (m21 - m12) * h
        elif md == 1:
            var h = sqrt((m11 - (m22 + m00)) + 1.0)
            qy = 0.5 * h
            h = 0.5 / h
            qz = (m12 + m21) * h
            qx = (m01 + m10) * h
            qw = (m02 - m20) * h
        else:
            var h = sqrt((m22 - (m00 + m11)) + 1.0)
            qz = 0.5 * h
            h = 0.5 / h
            qx = (m20 + m02) * h
            qy = (m12 + m21) * h
            qw = (m10 - m01) * h
    return MbQuat(qx, qy, qz, qw).normalized()


@fieldwise_init
struct MbXform(Copyable, ImplicitlyCopyable, Movable):
    """A rigid transform: rotate by q, then translate by p."""
    var p: MbVec3
    var q: MbQuat

    @staticmethod
    def identity() -> Self:
        return Self(MbVec3(0, 0, 0), MbQuat.identity())

    def __mul__(self, o: Self) -> Self:
        return Self(self.p + self.q.rotate(o.p), self.q * o.q)

    def inverse(self) -> Self:
        var qi = self.q.inverse()
        return Self(qi.rotate(self.p) * -1.0, qi)


def blend_xform(a: MbXform, b: MbXform, t: Float32) -> MbXform:
    return MbXform(_lerp3(a.p, b.p, t), quat_slerp(a.q, b.q, t))


# --------------------------------------------------------------------------------- skeleton

struct MbSkeleton(Copyable, Movable):
    var names: List[String]
    var parents: List[Int]
    var reference: List[MbXform]

    def __init__(out self):
        self.names = List[String]()
        self.parents = List[Int]()
        self.reference = List[MbXform]()

    def num_joints(self) -> Int:
        return len(self.names)

    def joint_index(self, name: String) -> Int:
        for k in range(len(self.names)):
            if self.names[k] == name:
                return k
        return -1


def world_pose(skel: MbSkeleton, local: List[MbXform], root: MbXform) -> List[MbXform]:
    """World transforms of one pose (parents precede their children)."""
    var out = List[MbXform](capacity=len(local))
    for j in range(len(local)):
        var p = skel.parents[j] if j < len(skel.parents) else -1
        var parent_xf = root if (p < 0 or p >= len(out)) else out[p]
        out.append(parent_xf * local[j])
    return out^


struct MbClip(Copyable, Movable):
    """An animation clip: local transforms per frame, at `sample_rate` Hz."""
    var skeleton: MbSkeleton
    var sample_rate: Float32
    var frames: List[List[MbXform]]

    def __init__(out self):
        self.skeleton = MbSkeleton()
        self.sample_rate = 30
        self.frames = List[List[MbXform]]()

    def num_frames(self) -> Int:
        return len(self.frames)

    def duration(self) -> Float32:
        if len(self.frames) <= 1 or not isfinite(self.sample_rate) or self.sample_rate <= 0.0:
            return Float32(0)
        return Float32(len(self.frames) - 1) / self.sample_rate

    def sample(self, time: Float32, root_offset: MbXform = MbXform.identity()) -> List[MbXform]:
        """The pose at `time` (clamped), blending the two nearest frames; root moved by root_offset."""
        var nj = self.skeleton.num_joints()
        if len(self.frames) == 0:
            var ref_out = List[MbXform](capacity=nj)
            for j in range(nj):
                var x = self.skeleton.reference[j] if j < len(self.skeleton.reference) else MbXform.identity()
                if j == 0:
                    x = root_offset * x
                ref_out.append(x)
            return ref_out^
        var last = len(self.frames) - 1
        var sr = self.sample_rate if (isfinite(self.sample_rate) and self.sample_rate > 0.0) else Float32(0.0)
        var fr: Float32 = 0.0
        if not isnan(time) and time > Float32(0.0):
            if not isfinite(time):
                fr = Float32(last)
            else:
                fr = time * sr
                if not isfinite(fr) or fr >= Float32(last):
                    fr = Float32(last)
        var f = min(max(Int(floor(fr)), 0), last)
        var blend: Float32 = 0
        if f < last:
            blend = min(max(fr - Float32(f), Float32(0)), Float32(1))
        var g = min(f + 1, last)
        var n_f = len(self.frames[f])
        var n_g = len(self.frames[g])
        var out = List[MbXform](capacity=nj)
        for j in range(nj):
            var xf = self.frames[f][j] if j < n_f else (self.skeleton.reference[j] if j < len(self.skeleton.reference) else MbXform.identity())
            var xg = self.frames[g][j] if j < n_g else (self.skeleton.reference[j] if j < len(self.skeleton.reference) else MbXform.identity())
            var x = blend_xform(xf, xg, blend)
            if j == 0:
                x = root_offset * x
            out.append(x)
        return out^


def _euler_quat(angles: List[Float32], order: List[Int]) -> MbQuat:
    """BVH rotation channels in degrees, composed in the listed order (as MotionBricks does)."""
    var q = MbQuat.identity()
    for k in range(len(order)):
        var axis = MbVec3(Float32(1 if order[k] == 0 else 0), Float32(1 if order[k] == 1 else 0), Float32(1 if order[k] == 2 else 0))
        q = q * quat_axis_angle(axis, angles[k] * 0.017453292519943295).normalized()
    return q.normalized()


def _tokens(text: String) -> List[String]:
    var out = List[String]()
    var bytes = text.as_bytes()
    var start = -1
    for k in range(len(bytes)):
        var c = bytes[k]
        var space = c == 32 or c == 9 or c == 10 or c == 13
        if space:
            if start >= 0:
                out.append(String(text[byte=start:k]))
                start = -1
        elif start < 0:
            start = k
    if start >= 0:
        out.append(String(text[byte=start:len(bytes)]))
    return out^


def load_bvh(path: String, fold_root: Bool = True) raises -> MbClip:
    """A BVH clip: centimetres become metres, namespaces ("a:b:Hips") are dropped. With
    `fold_root`, a top joint named "Root" is folded into its single child, so that Hips is
    joint 0 as the SOMA model needs."""
    var data = read_bytes(path)
    if len(data) == 0:
        raise Error("BVH: cannot read " + path)
    var tok = _tokens(String(unsafe_from_utf8=data))
    if len(tok) == 0:
        raise Error("BVH: empty file " + path)
    var clip = MbClip()
    var channels = List[List[Int]]()   # per joint: channel codes 0..2 position xyz, 3..5 rotation xyz
    var stack = List[Int]()
    var in_end = False
    var i = 0
    while i < len(tok):
        var t = tok[i]
        if t == "MOTION":
            i += 1
            break
        if t == "ROOT" or t == "JOINT":
            if i + 1 >= len(tok):
                raise Error("BVH: unexpected EOF after " + t)
            var name = tok[i + 1]
            var parts = name.split(":")
            clip.skeleton.names.append(String(parts[len(parts) - 1]))
            clip.skeleton.parents.append(stack[len(stack) - 1] if len(stack) > 0 else -1)
            clip.skeleton.reference.append(MbXform.identity())
            channels.append(List[Int]())
            stack.append(len(clip.skeleton.names) - 1)
            i += 2
        elif t == "End":
            if i + 1 >= len(tok):
                raise Error("BVH: unexpected EOF after End")
            in_end = True
            i += 2
        elif t == "OFFSET":
            if i + 3 >= len(tok):
                raise Error("BVH: unexpected EOF in OFFSET")
            if not in_end:
                if len(stack) == 0:
                    raise Error("BVH: OFFSET outside joint block")
                var j = stack[len(stack) - 1]
                clip.skeleton.reference[j].p = MbVec3(
                    Float32(atof(tok[i + 1])) * 0.01, Float32(atof(tok[i + 2])) * 0.01, Float32(atof(tok[i + 3])) * 0.01)
            i += 4
        elif t == "CHANNELS":
            if i + 1 >= len(tok):
                raise Error("BVH: unexpected EOF after CHANNELS")
            if len(stack) == 0:
                raise Error("BVH: CHANNELS outside joint block")
            var n = Int(atol(tok[i + 1]))
            if n < 0 or i + 1 + n >= len(tok):
                raise Error("BVH: truncated CHANNELS list")
            var j = stack[len(stack) - 1]
            for c in range(n):
                var ch = tok[i + 2 + c].lower()
                var axis = 0 if ch.startswith("x") else (1 if ch.startswith("y") else 2)
                channels[j].append(axis if ch.endswith("position") else 3 + axis)
            i += 2 + n
        elif t == "}":
            if in_end:
                in_end = False
            else:
                if len(stack) == 0:
                    raise Error("BVH: unmatched '}'")
                _ = stack.pop()
            i += 1
        else:
            i += 1
    if len(stack) != 0:
        raise Error("BVH: unclosed joint block")
    var frames = 0
    while i < len(tok):
        if tok[i] == "Frames:":
            if i + 1 >= len(tok):
                raise Error("BVH: unexpected EOF after Frames:")
            frames = Int(atol(tok[i + 1]))
            if frames < 0:
                raise Error("BVH: negative frame count")
            i += 2
        elif tok[i] == "Frame" and i + 2 < len(tok) and tok[i + 1] == "Time:":
            var ft = atof(tok[i + 2])
            if not isfinite(ft) or ft <= 0.0:
                raise Error("BVH: invalid Frame Time")
            var inv = 1.0 / ft
            if not isfinite(inv) or inv <= 0.0 or inv > 1_000_000.0:
                raise Error("BVH: Frame Time out of supported range")
            var rate = Float32(Int(inv * 100.0 + 0.5)) / 100.0
            if not isfinite(rate) or rate <= 0.0:
                raise Error("BVH: invalid sample rate from Frame Time")
            clip.sample_rate = rate
            i += 3
            break
        else:
            i += 1
    var nj = clip.skeleton.num_joints()
    var per_frame = 0
    for j in range(nj):
        per_frame += len(channels[j])
    var rem_tok = len(tok) - i
    if (
        frames < 0
        or rem_tok < 0
        or (per_frame > 0 and frames > rem_tok // per_frame)
        or (per_frame == 0 and frames > 1_000_000)
    ):
        raise Error("BVH: " + path + " has fewer motion values than its header says")
    for _ in range(frames):
        var pose = List[MbXform](capacity=nj)
        for j in range(nj):
            var pos = clip.skeleton.reference[j].p
            var has_pos = False
            var angles = List[Float32]()
            var order = List[Int]()
            var px: Float32 = 0
            var py: Float32 = 0
            var pz: Float32 = 0
            for c in channels[j]:
                var v = Float32(atof(tok[i]))
                i += 1
                if c < 3:
                    has_pos = True
                    if c == 0:
                        px = v
                    elif c == 1:
                        py = v
                    else:
                        pz = v
                else:
                    angles.append(v)
                    order.append(c - 3)
            if has_pos:
                pos = MbVec3(px * 0.01, py * 0.01, pz * 0.01)
            var rot = _euler_quat(angles, order) if len(order) > 0 else clip.skeleton.reference[j].q
            pose.append(MbXform(pos, rot))
        clip.frames.append(pose^)
    if fold_root and nj > 1 and clip.skeleton.names[0] == "Root" and clip.skeleton.parents[1] == 0:
        clip = _fold_root(clip)
    return clip^


def _fold_root(clip: MbClip) -> MbClip:
    var out = MbClip()
    out.sample_rate = clip.sample_rate
    var nj = clip.skeleton.num_joints()
    for j in range(1, nj):
        out.skeleton.names.append(clip.skeleton.names[j])
        var p = clip.skeleton.parents[j]
        out.skeleton.parents.append(-1 if p <= 0 else p - 1)
        out.skeleton.reference.append(clip.skeleton.reference[0] * clip.skeleton.reference[j] if p == 0 else clip.skeleton.reference[j])
    for f in range(len(clip.frames)):
        var pose = List[MbXform](capacity=nj - 1)
        for j in range(1, nj):
            if clip.skeleton.parents[j] == 0:
                pose.append(clip.frames[f][0] * clip.frames[f][j])
            else:
                pose.append(clip.frames[f][j])
        out.frames.append(pose^)
    return out^


def retarget_clip(clip: MbClip, skel: MbSkeleton) -> MbClip:
    """The clip on another skeleton, matching joints by name; joints the clip lacks keep the
    skeleton's reference transforms."""
    var out = MbClip()
    out.skeleton = skel.copy()
    out.sample_rate = clip.sample_rate
    var src = List[Int]()
    for j in range(skel.num_joints()):
        src.append(clip.skeleton.joint_index(skel.names[j]))
    for f in range(len(clip.frames)):
        var pose = List[MbXform](capacity=skel.num_joints())
        for j in range(skel.num_joints()):
            pose.append(clip.frames[f][src[j]] if src[j] >= 0 else skel.reference[j])
        out.frames.append(pose^)
    return out^


# ------------------------------------------------------------------------------- generation

def _heading(right_hip: MbVec3, left_hip: MbVec3) -> Float32:
    var across = MbVec3(right_hip.x - left_hip.x, 0, right_hip.z - left_hip.z)
    var l = across.length()
    if l <= 1e-6:
        return 0
    across = across * (1.0 / l)
    var forward = _cross(MbVec3(0, 1, 0), across)
    return atan2(forward.x, forward.z)


def _read_rotation(m: Tensor, frame: Int, offset: Int) -> MbQuat:
    var w = m.shape[2]
    var b = frame * w + offset
    var xr = MbVec3(m.fget(b), m.fget(b + 1), m.fget(b + 2))
    var yr = MbVec3(m.fget(b + 3), m.fget(b + 4), m.fget(b + 5))
    var xa = xr.normalized()
    var za = _cross(xa, yr).normalized()
    var ya = _cross(za, xa)
    return quat_from_basis(xa, ya, za)


struct MbConfig(Copyable, Movable):
    """Generation options: which token counts (6..16, 4 frames each) the model may choose,
    whether to correct the endpoints onto the context and target, and the heading (x, z) the
    last context pose is turned to face in model space."""
    var allowed_tokens: List[Int]
    var correct_endpoints: Bool
    var heading_x: Float32
    var heading_z: Float32

    def __init__(out self):
        self.allowed_tokens = [8, 9, 10, 11, 12, 13, 14]
        self.correct_endpoints = True
        self.heading_x = 1
        self.heading_z = 0

    def token_bits(self) -> Int:
        var bits = 0
        for t in self.allowed_tokens:
            bits |= 1 << (t - MB_MIN_TOKENS)
        return bits


struct MbMotion(Copyable, Movable):
    """A generated motion: `poses[frame][joint]` local transforms at 30 Hz, and per frame the
    contacts of LeftFoot, LeftToeBase, RightFoot, RightToeBase in [0, 1]."""
    var poses: List[List[MbXform]]
    var contacts: List[SIMD[DType.float32, 4]]
    var tokens: Int

    def __init__(out self):
        self.poses = List[List[MbXform]]()
        self.contacts = List[SIMD[DType.float32, 4]]()
        self.tokens = 0

    def num_frames(self) -> Int:
        return len(self.poses)


struct MotionBricks(Movable):
    var model: OnnxModel
    var skeleton: MbSkeleton
    var model_to_pose: List[Int]
    var pose_to_model: List[Int]
    var right_leg: Int
    var left_leg: Int
    var model_to_runtime: MbXform
    var last_align: MbXform

    def __init__(out self, var model: OnnxModel, skeleton: MbSkeleton) raises:
        self.model = model^
        self.skeleton = skeleton.copy()
        self.model_to_pose = List[Int]()
        self.pose_to_model = List[Int](length=skeleton.num_joints(), fill=-1)
        self.model_to_runtime = MbXform.identity()
        self.last_align = MbXform.identity()
        var names = soma_joint_names()
        var missing = String()
        for n in names:
            var j = skeleton.joint_index(n)
            if j < 0:
                missing += " " + n
            self.model_to_pose.append(j)
        if missing.byte_length() > 0:
            raise Error("MotionBricks: the skeleton lacks model joints:" + missing)
        if self.model_to_pose[0] != 0:
            raise Error("MotionBricks: the skeleton's root (joint 0) must be Hips")
        for mj in range(len(names)):
            var pj = self.model_to_pose[mj]
            var par = skeleton.parents[pj]
            # model joints outside the connected hierarchy keep their reference transforms
            if par == -1 or self.pose_to_model[par] >= 0:
                self.pose_to_model[pj] = mj
        self.right_leg = skeleton.joint_index("RightLeg")
        self.left_leg = skeleton.joint_index("LeftLeg")

    @staticmethod
    def load(onnx_path: String, skeleton: MbSkeleton, gpu: Bool = True) raises -> MotionBricks:
        """Load the SOMA model (`mmm_v6_soma_fp32.onnx`); with `gpu`, run its matrix products on
        a MAX GPU when there is one."""
        var m = OnnxModel.load(onnx_path)
        if gpu:
            _ = m.use_gpu()
        return MotionBricks(m^, skeleton)

    def device(self) raises -> String:
        return self.model.gpu_name()

    def _encode(mut self, context: List[List[MbXform]], target: List[List[MbXform]], cfg: MbConfig) raises:
        var to_model = self.model_to_runtime.inverse()
        var world = List[List[MbXform]]()
        for f in range(MB_CONTEXT_AND_TARGET):
            if f < MB_POSES_PER_TOKEN:
                world.append(world_pose(self.skeleton, context[f], to_model))
            else:
                world.append(world_pose(self.skeleton, target[min(f - MB_POSES_PER_TOKEN, len(target) - 1)], to_model))
        var lc = MB_POSES_PER_TOKEN - 1
        var hips = world[lc][0].p
        var cur_angle = _heading(world[lc][self.right_leg].p, world[lc][self.left_leg].p)
        var want = atan2(cfg.heading_x, cfg.heading_z)
        var align = MbXform(MbVec3(hips.x, 0, hips.z), quat_axis_angle(MbVec3(0, 1, 0), cur_angle - want))
        self.last_align = align
        var nmj = len(self.model_to_pose)
        var g = List[Float32](length=MB_CONTEXT_AND_TARGET * 5, fill=0.0)
        var l = List[Float32](length=MB_CONTEXT_AND_TARGET * 4, fill=0.0)
        var p = List[Float32](length=MB_CONTEXT_AND_TARGET * MB_POSE_VALUES, fill=0.0)
        var qinv = align.q.inverse()
        for f in range(MB_CONTEXT_AND_TARGET):
            var nf = f if f == MB_POSES_PER_TOKEN - 1 else min(f + 1, MB_CONTEXT_AND_TARGET - 1)
            var ch = align.q.rotate_inv(world[f][0].p - align.p)
            var nh = align.q.rotate_inv(world[nf][0].p - align.p)
            var cr = align.q.rotate_inv(world[f][self.right_leg].p - align.p)
            var cl = align.q.rotate_inv(world[f][self.left_leg].p - align.p)
            var nr = align.q.rotate_inv(world[nf][self.right_leg].p - align.p)
            var nl = align.q.rotate_inv(world[nf][self.left_leg].p - align.p)
            var chd = _heading(cr, cl)
            var nhd = _heading(nr, nl)
            var dh = atan2(sin(nhd - chd), cos(nhd - chd))
            g[f * 5 + 0] = ch.x
            g[f * 5 + 1] = ch.y
            g[f * 5 + 2] = ch.z
            g[f * 5 + 3] = cos(chd)
            g[f * 5 + 4] = sin(chd)
            l[f * 4 + 0] = dh * MB_SAMPLE_RATE
            l[f * 4 + 1] = (nh.x - ch.x) * MB_SAMPLE_RATE
            l[f * 4 + 2] = (nh.z - ch.z) * MB_SAMPLE_RATE
            l[f * 4 + 3] = ch.y
            var root = MbVec3(ch.x, 0, ch.z)
            for mj in range(nmj):
                var pj = self.model_to_pose[mj]
                var jp = align.q.rotate_inv(world[f][pj].p - align.p)
                var jq = qinv * world[f][pj].q
                var base = f * MB_POSE_VALUES
                if mj > 0:
                    var c = jp - root
                    var po = base + (mj - 1) * 3
                    p[po] = c.x
                    p[po + 1] = c.y
                    p[po + 2] = c.z
                var ro = base + (nmj - 1) * 3 + mj * 6
                var xa = jq.rotate(MbVec3(1, 0, 0))
                var ya = jq.rotate(MbVec3(0, 1, 0))
                p[ro] = xa.x
                p[ro + 1] = xa.y
                p[ro + 2] = xa.z
                p[ro + 3] = ya.x
                p[ro + 4] = ya.y
                p[ro + 5] = ya.z
        var nt = len(target)
        var hg = List[Int](length=MB_CONTEXT_AND_TARGET, fill=0)
        var hl = List[Int](length=MB_CONTEXT_AND_TARGET, fill=0)
        for idx in range(MB_CONTEXT_AND_TARGET):
            var v = 1 if idx < MB_POSES_PER_TOKEN + nt else 0
            hg[idx] = v
            hl[idx] = 0 if (idx == MB_POSES_PER_TOKEN - 1 or idx == MB_POSES_PER_TOKEN + nt - 1) else v
        var bits = cfg.token_bits()
        var allowed = List[Int](length=MB_MAX_TOKENS - MB_MIN_TOKENS + 1, fill=0)
        for k in range(len(allowed)):
            allowed[k] = (bits >> k) & 1
        self.model.set_input("global_root_values", Tensor.floats([1, 8, 5], g^))
        self.model.set_input("local_root_values", Tensor.floats([1, 8, 4], l^))
        self.model.set_input("local_poses", Tensor.floats([1, 8, MB_POSE_VALUES], p^))
        self.model.set_input("has_global_root_values", Tensor.ints(DT_I32, [1, 8], hg.copy()))
        self.model.set_input("has_local_poses", Tensor.ints(DT_I32, [1, 8], hg^))
        self.model.set_input("has_local_root_values", Tensor.ints(DT_I32, [1, 8], hl^))
        self.model.set_input("allowed_pred_num_tokens", Tensor.ints(DT_I32, [1, len(allowed)], allowed^))

    def _decode(self, motions: Tensor, tokens: Int) -> MbMotion:
        var out = MbMotion()
        out.tokens = tokens
        var raw = min(max(tokens * MB_POSES_PER_TOKEN, 0), motions.shape[1])
        var count = max(0, raw - 2 * MB_BOUNDARY_FRAMES)
        var w = motions.shape[2]
        var nmj = len(self.model_to_pose)
        var rot0 = 5 + (nmj - 1) * 3
        var nj = self.skeleton.num_joints()
        for frame in range(count):
            var rf = MB_BOUNDARY_FRAMES + frame
            var fc = SIMD[DType.float32, 4](0)
            for c in range(MB_FOOT_CONTACTS):
                fc[c] = min(max(motions.fget(rf * w + w - MB_FOOT_CONTACTS + c), 0), 1)
            out.contacts.append(fc)
            var pose = List[MbXform](capacity=nj)
            var hp = MbVec3(motions.fget(rf * w), motions.fget(rf * w + 1), motions.fget(rf * w + 2))
            var hips = self.last_align * MbXform(hp, _read_rotation(motions, rf, rot0))
            pose.append(self.model_to_runtime * hips)
            for j in range(1, nj):
                var mj = self.pose_to_model[j]
                if mj < 0:
                    pose.append(self.skeleton.reference[j])
                    continue
                var pr = _read_rotation(motions, rf, rot0 + mj * 6)
                var parent_mj = self.pose_to_model[self.skeleton.parents[j]]
                var par = _read_rotation(motions, rf, rot0 + parent_mj * 6)
                pose.append(MbXform(self.skeleton.reference[j].p, (par.inverse() * pr).normalized()))
            out.poses.append(pose^)
        return out^

    def generate(mut self, context: List[List[MbXform]], target: List[List[MbXform]],
                 cfg: MbConfig = MbConfig()) raises -> MbMotion:
        """Generate the motion from the four `context` poses (oldest first, ending at the current
        pose) to the 1-4 `target` poses (starting at the pose to reach)."""
        if len(context) != MB_POSES_PER_TOKEN:
            raise Error("MotionBricks: the context needs 4 poses")
        if len(target) < 1 or len(target) > MB_POSES_PER_TOKEN:
            raise Error("MotionBricks: the target needs 1 to 4 poses")
        for t in cfg.allowed_tokens:
            if t < MB_MIN_TOKENS or t > MB_MAX_TOKENS:
                raise Error("MotionBricks: token counts must be 6 to 16")
        self._encode(context, target, cfg)
        self.model.run()
        var motions = self.model.output("pred_global_motions")
        var tokens = self.model.output("pred_num_tokens").int_at(0)
        var allowed = False
        for t in cfg.allowed_tokens:
            if t == tokens:
                allowed = True
        if not allowed:
            raise Error("MotionBricks: the model chose " + String(tokens) + " tokens, outside the allowed set")
        var out = self._decode(motions, tokens)
        if cfg.correct_endpoints:
            _correct_endpoints(out, context[MB_POSES_PER_TOKEN - 1], target[0])
        return out^


def _correct_endpoints(mut m: MbMotion, last_context: List[MbXform], first_target: List[MbXform]):
    """Fade a correction from the last context pose in, and toward the first target pose out,
    so the motion starts and ends exactly on them."""
    var n = m.num_frames()
    if n < 2:
        return
    var nj = len(last_context)
    var start = List[MbXform](capacity=nj)
    var end = List[MbXform](capacity=nj)
    for j in range(nj):
        if j == 0:
            start.append(_root_delta(m.poses[0][0], last_context[0]))
            end.append(_root_delta(m.poses[n - 1][0], first_target[0]))
        else:
            start.append(m.poses[0][j].inverse() * last_context[j])
            end.append(m.poses[n - 1][j].inverse() * first_target[j])
    for f in range(n):
        var ew = Float32(f) / Float32(n - 1)
        var sw = 1.0 - ew
        for j in range(nj):
            var x = m.poses[f][j]
            if j == 0:
                var q = (x.q * quat_slerp(MbQuat.identity(), start[0].q, sw)) * quat_slerp(MbQuat.identity(), end[0].q, ew)
                m.poses[f][0] = MbXform(x.p + start[0].p * sw + end[0].p * ew, q.normalized())
            else:
                var c = x * blend_xform(MbXform.identity(), start[j], sw)
                m.poses[f][j] = c * blend_xform(MbXform.identity(), end[j], ew)


def _root_delta(generated: MbXform, guide: MbXform) -> MbXform:
    var rot = generated.q.normalized().inverse() * guide.q.normalized()
    return MbXform(guide.p - generated.p, rot.normalized())
