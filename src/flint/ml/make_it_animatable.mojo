"""Make-It-Animatable v1 (`flint.ml.make_it_animatable`): fast humanoid auto-rigging onto a fixed
52-joint Mixamo skeleton, with skin weights and a pose-to-rest (T-pose) normalisation.

A port of the inference pipeline of Make-It-Animatable (Guo et al., CVPR 2025;
https://github.com/jasongzy/Make-It-Animatable, code MIT, weights Apache-2.0; see
licenses/make-it-animatable-LICENSE.txt). Only v1 is ported; v2 depends on the Hunyuan3D 2.1
ShapeVAE, whose licence is territory-restricted.

The five networks run on `flint.ml.onnx` (GPU when present) from graphs converted by
`tools/make_it_animatable/export.py` (kept outside the repo):
`mia_coarse.onnx`, `mia_joints.onnx`, `mia_pose.onnx`, `mia_bw_trunk.onnx`, `mia_bw_head.onnx`.
Everything around them is here, following upstream `app.py`:

  1. 32768 surface samples, bounding-box normalised; coarse joints from them;
  2. the "Hips frame": the pelvis plane (Hips, RightUpLeg, LeftUpLeg) turned to face +Z with
     the right-to-left axis on +X (Y up, glTF / Mixamo facing); 16384 uniform samples plus 8192
     in a box around each hand, normalised by the largest |coordinate|;
  3. skin weights for every vertex, joints (head + tail per bone, kinematic-causal decoder) and
     per-joint global rotations that take the input pose to the rest pose (ortho6d);
  4. upstream's skin-weight clean-up (`bw_post_process`: head above the head joint, mutually
     exclusive limbs / fingers, 4 influences), the rest transforms (`pose_rot_to_global`) and
     linear blend skinning into the rest pose.

Farthest point sampling starts at the first point (upstream's torch_cluster call starts at a
random one); `tools/make_it_animatable/shims/torch_cluster` does the same, so the latent points
match the PyTorch reference exactly.

`rig_native_model` takes a loaded glTF scene, rigs all its meshes together and returns a skinned
copy whose 52 joints are named `mixamorig:<Joint>`. With `rest=True` (the default) the mesh is
baked into the predicted T-pose and the skeleton is bound there, upright, facing +Z, feet on
y = 0, in the input's units: the layout Mixamo clips and `retarget_clip` expect. With
`rest=False` the mesh keeps its input pose and placement.

`retarget_clip` maps a MotionBricks / GEM-X SOMA clip (`flint.ml.motionbricks.MbClip`, e.g. from
`load_bvh` or `gemx_clip`) or a Mixamo-named BVH onto such a rig as a `NativeAnimation`
(`mia_soma_name` is the joint correspondence).
"""

from std.math import sqrt, atan2, sin, cos
from std.time import perf_counter_ns
from flint.core.math import Vec3
from flint.core.transform import Mat4
from flint.assets.model import NativeModel, NativeMesh, NativeNode, NativeSkin, NativeAnimation, NativeChannel
from flint.ml.tensor import Tensor
from flint.ml.onnx import OnnxModel
from flint.ml.unirig import UrRng
from flint.ml.motionbricks import MbClip, MbXform, MbQuat, world_pose

comptime MIA_JOINTS: Int = 52
comptime MIA_POINTS: Int = 32768
comptime MIA_LATENTS: Int = 512
comptime MIA_HANDS_RATIO: Float64 = 0.5
comptime MIA_HAND_BOX: Float64 = 0.15     # upstream attn_rel_radius
comptime MIA_INFLUENCES: Int = 4
comptime MIA_BW_CHUNK: Int = 32768
comptime MIA_PREFIX = "mixamorig:"


# ------------------------------------------------------------------------------- skeleton

def mia_joint_names() -> List[String]:
    """The 52 joints in network order (upstream MIXAMO_JOINTS without the "mixamorig:" prefix)."""
    return [
        "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
        "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
        "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
        "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
        "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
        "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
        "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
        "RightHandRing1", "RightHandRing2", "RightHandRing3",
        "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
        "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
        "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    ]


def _strip_prefix(name: String) -> String:
    if name.startswith(MIA_PREFIX):
        return String(name[byte=MIA_PREFIX.byte_length():])
    return name


def mia_joint_index(name: String) -> Int:
    """Index of a joint by name, with or without the "mixamorig:" prefix; -1 if unknown."""
    var n = _strip_prefix(name)
    var names = mia_joint_names()
    for j in range(len(names)):
        if names[j] == n:
            return j
    return -1


def mia_parents() -> List[Int]:
    """Parent of each joint (-1 for Hips): the Mixamo hierarchy."""
    var names = mia_joint_names()
    var out = List[Int](length=MIA_JOINTS, fill=-1)
    for j in range(MIA_JOINTS):
        ref n = names[j]
        var p = String()
        if n == "Hips":
            continue
        if n == "Spine":
            p = "Hips"
        elif n == "Spine1":
            p = "Spine"
        elif n == "Spine2":
            p = "Spine1"
        elif n == "Neck" or n.endswith("Shoulder"):
            p = "Spine2"
        elif n == "Head":
            p = "Neck"
        elif n.endswith("UpLeg"):
            p = "Hips"
        else:
            var side = "Left" if n.startswith("Left") else "Right"
            var rest = String(n[byte=side.byte_length():])
            if rest == "Arm":
                p = side + "Shoulder"
            elif rest == "ForeArm":
                p = side + "Arm"
            elif rest == "Hand":
                p = side + "ForeArm"
            elif rest == "Leg":
                p = side + "UpLeg"
            elif rest == "Foot":
                p = side + "Leg"
            elif rest == "ToeBase":
                p = side + "Foot"
            elif rest.startswith("Hand"):
                var digit = Int(rest.as_bytes()[rest.byte_length() - 1]) - 48
                if digit == 1:
                    p = side + "Hand"
                else:
                    p = String(n[byte=0:n.byte_length() - 1]) + String(digit - 1)
        if p.byte_length() > 0:
            out[j] = mia_joint_index(p)
    return out^


def mia_soma_name(name: String) -> String:
    """The SOMA joint (MotionBricks / GEM-X skeletons, `soma_joint_names`, `soma78_joint_names`)
    playing the same role as a Mixamo joint. SOMA has three spine joints and two neck joints
    where Mixamo has Spine/Spine1/Spine2 and Neck, and calls the thigh "Leg" and the shin "Shin"."""
    var n = _strip_prefix(name)
    if n == "Spine":
        return "Spine1"
    if n == "Spine1":
        return "Spine2"
    if n == "Spine2":
        return "Chest"
    if n == "Neck":
        return "Neck1"
    for side in ["Left", "Right"]:
        if n == side + "UpLeg":
            return side + "Leg"
        if n == side + "Leg":
            return side + "Shin"
    return n


def mia_depths() -> List[Int]:
    var par = mia_parents()
    var d = List[Int](length=MIA_JOINTS, fill=0)
    for j in range(MIA_JOINTS):
        var p = par[j]
        while p >= 0:
            d[j] += 1
            p = par[p]
    return d^


# ------------------------------------------------------------------------------- sampling

def _tri_area(v: List[Float32], a: Int, b: Int, c: Int) -> Float64:
    var e1x = Float64(v[b * 3]) - Float64(v[a * 3])
    var e1y = Float64(v[b * 3 + 1]) - Float64(v[a * 3 + 1])
    var e1z = Float64(v[b * 3 + 2]) - Float64(v[a * 3 + 2])
    var e2x = Float64(v[c * 3]) - Float64(v[a * 3])
    var e2y = Float64(v[c * 3 + 1]) - Float64(v[a * 3 + 1])
    var e2z = Float64(v[c * 3 + 2]) - Float64(v[a * 3 + 2])
    var nx = e1y * e2z - e1z * e2y
    var ny = e1z * e2x - e1x * e2z
    var nz = e1x * e2y - e1y * e2x
    return 0.5 * sqrt(nx * nx + ny * ny + nz * nz)


def _sample_triangles(tv: List[Float32], tf: List[Int], n: Int, mut rng: UrRng, mut out: List[Float32]):
    """`n` area-uniform points on a triangle soup (trimesh.sample.sample_surface's scheme)."""
    var nf = len(tf) // 3
    if nf == 0 or n <= 0:
        return
    var cum = List[Float64](length=nf, fill=0.0)
    var acc: Float64 = 0
    for t in range(nf):
        acc += _tri_area(tv, tf[t * 3], tf[t * 3 + 1], tf[t * 3 + 2])
        cum[t] = acc
    if acc <= 0:
        return
    for _ in range(n):
        var pick = rng.uniform() * acc
        var lo = 0
        var hi = nf - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if cum[mid] < pick:
                lo = mid + 1
            else:
                hi = mid
        var r0 = rng.uniform()
        var r1 = rng.uniform()
        if r0 + r1 > 1.0:
            r0 = 1.0 - r0
            r1 = 1.0 - r1
        var a = tf[lo * 3]
        var b = tf[lo * 3 + 1]
        var c = tf[lo * 3 + 2]
        for k in range(3):
            var pa = Float64(tv[a * 3 + k])
            out.append(Float32(pa + r0 * (Float64(tv[b * 3 + k]) - pa) + r1 * (Float64(tv[c * 3 + k]) - pa)))


def clip_to_box(v: List[Float32], f: List[Int], center: List[Float64], half: Float64,
                mut cv: List[Float32], mut cf: List[Int]):
    """The part of a triangle mesh inside an axis-aligned cube (upstream: trimesh slice_plane by
    the six faces of a box of side 2 * half), as a triangle soup."""
    cv = List[Float32]()
    cf = List[Int]()
    var nf = len(f) // 3
    for t in range(nf):
        var poly = List[Float64]()
        var inside = True
        var outside = False
        for k in range(3):
            var vi = f[t * 3 + k]
            for c in range(3):
                var x = Float64(v[vi * 3 + c])
                poly.append(x)
                if abs(x - center[c]) > half:
                    inside = False
        # quick reject: all three on the far side of one plane
        for c in range(3):
            var lo_all = True
            var hi_all = True
            for k in range(3):
                var x = poly[k * 3 + c] - center[c]
                if x >= -half:
                    lo_all = False
                if x <= half:
                    hi_all = False
            if lo_all or hi_all:
                outside = True
        if outside:
            continue
        if not inside:
            # Sutherland-Hodgman against the six planes
            for c in range(3):
                for s in range(2):
                    var sign: Float64 = 1.0 if s == 0 else -1.0
                    var lim = center[c] + sign * half
                    var np = len(poly) // 3
                    if np == 0:
                        break
                    var res = List[Float64]()
                    for i in range(np):
                        var j = (i + 1) % np
                        var di = sign * (lim - poly[i * 3 + c])
                        var dj = sign * (lim - poly[j * 3 + c])
                        if di >= 0:
                            for e in range(3):
                                res.append(poly[i * 3 + e])
                        if (di >= 0) != (dj >= 0):
                            var u = di / (di - dj)
                            for e in range(3):
                                res.append(poly[i * 3 + e] + u * (poly[j * 3 + e] - poly[i * 3 + e]))
                    poly = res^
        var np = len(poly) // 3
        if np < 3:
            continue
        var base = len(cv) // 3
        for e in range(np * 3):
            cv.append(Float32(poly[e]))
        for i in range(1, np - 1):
            cf.append(base)
            cf.append(base + i)
            cf.append(base + i + 1)


def sample_points(v: List[Float32], f: List[Int], n: Int, mut rng: UrRng) -> List[Float32]:
    var out = List[Float32](capacity=n * 3)
    _sample_triangles(v, f, n, rng, out)
    return out^


def sample_points_hands(v: List[Float32], f: List[Int], n: Int, hands: List[Float64], mut rng: UrRng) -> List[Float32]:
    """Upstream sample_mesh(attn_ratio=0.5, attn_centers=hands): n/2 uniform points, then n/4 in
    a box of side 0.15 * (max - min of all vertex coordinates) around each hand centre (flat
    xyz, LeftHand first), topped up with uniform points when a box holds no surface."""
    var n_attn = Int(Float64(n) * MIA_HANDS_RATIO)
    var out = sample_points(v, f, n - n_attn, rng)
    var nc = len(hands) // 3
    var lo: Float64 = 1e300
    var hi: Float64 = -1e300
    for x in v:
        lo = min(lo, Float64(x))
        hi = max(hi, Float64(x))
    var side = MIA_HAND_BOX * (hi - lo)
    for c in range(nc):
        var k = n_attn // nc if c < nc - 1 else n_attn - (n_attn // nc) * (nc - 1)
        var cv = List[Float32]()
        var cf = List[Int]()
        clip_to_box(v, f, [hands[c * 3], hands[c * 3 + 1], hands[c * 3 + 2]], side / 2, cv, cf)
        _sample_triangles(cv, cf, k, rng, out)
    if len(out) // 3 < n:
        _sample_triangles(v, f, n - len(out) // 3, rng, out)
    return out^


def normalize_points(pts: List[Float32], recenter: Bool) -> Tuple[Float32, List[Float32]]:
    """Upstream get_normalize_transform(keep_ratio=True) as (scale s, translation t): p -> p s + t.
    recenter: the bounding box centre goes to the origin, its longest side to 2; otherwise only
    scaled so the largest |coordinate| along the widest axis is 1."""
    var mn = List[Float32](length=3, fill=Float32(1e30))
    var mx = List[Float32](length=3, fill=Float32(-1e30))
    for i in range(len(pts) // 3):
        for c in range(3):
            mn[c] = min(mn[c], pts[i * 3 + c])
            mx[c] = max(mx[c], pts[i * 3 + c])
    if not recenter:
        for c in range(3):
            mx[c] = max(abs(mx[c]), abs(mn[c]))
            mn[c] = -mx[c]
    var ext = max(max(mx[0] - mn[0], mx[1] - mn[1]), mx[2] - mn[2])
    var s = Float32(2.0) / ext
    var t = List[Float32](length=3, fill=0.0)
    if recenter:
        for c in range(3):
            t[c] = -(mx[c] + mn[c]) / Float32(2.0) * s
    return (s, t^)


def apply_scale_shift(pts: List[Float32], s: Float32, t: List[Float32]) -> List[Float32]:
    var out = List[Float32](length=len(pts), fill=0.0)
    for i in range(len(pts) // 3):
        for c in range(3):
            out[i * 3 + c] = pts[i * 3 + c] * s + t[c]
    return out^


def farthest_points(pts: List[Float32], start: Int, n: Int, m: Int) -> List[Int]:
    """Farthest point sampling of `m` of the points start..start+n-1 (flat xyz), beginning with
    the first: float32 squared distances summed x + y + z, ties to the lowest index (exactly the
    shim's torch_cluster.fps)."""
    var sel = List[Int](capacity=m)
    var dist = List[Float32](length=n, fill=Float32(3.0e38))
    var cur = 0
    for _ in range(m):
        sel.append(start + cur)
        var cx = pts[(start + cur) * 3]
        var cy = pts[(start + cur) * 3 + 1]
        var cz = pts[(start + cur) * 3 + 2]
        var best: Float32 = -1
        var arg = 0
        for i in range(n):
            var b = (start + i) * 3
            var dx = pts[b] - cx
            var dy = pts[b + 1] - cy
            var dz = pts[b + 2] - cz
            var d = dx * dx + dy * dy + dz * dz
            if d < dist[i]:
                dist[i] = d
            if dist[i] > best:
                best = dist[i]
                arg = i
        cur = arg
    return sel^


def latent_indices(pts: List[Float32], hands_ratio: Float64) -> List[Int]:
    """PCAE.fps: the 512 latent points, split between the uniform and the hand samples."""
    var n = len(pts) // 3
    var n_h = Int(Float64(n) * hands_ratio)
    var l_h = Int(Float64(MIA_LATENTS) * hands_ratio)
    var out = farthest_points(pts, 0, n - n_h, MIA_LATENTS - l_h)
    if n_h > 0:
        for i in farthest_points(pts, n - n_h, n_h, l_h):
            out.append(i)
    return out^


def gather_points(pts: List[Float32], idx: List[Int]) -> List[Float32]:
    var out = List[Float32](capacity=len(idx) * 3)
    for i in idx:
        for c in range(3):
            out.append(pts[i * 3 + c])
    return out^


# ------------------------------------------------------------------------------- 3D math (row-major 4x4, Float64)

def _ident() -> List[Float64]:
    var m = List[Float64](length=16, fill=0.0)
    m[0] = 1
    m[5] = 1
    m[10] = 1
    m[15] = 1
    return m^


def mat_mul(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var o = List[Float64](length=16, fill=0.0)
    for i in range(4):
        for j in range(4):
            var s: Float64 = 0
            for k in range(4):
                s += a[i * 4 + k] * b[k * 4 + j]
            o[i * 4 + j] = s
    return o^


def mat_point(m: List[Float64], x: Float64, y: Float64, z: Float64) -> Tuple[Float64, Float64, Float64]:
    return (m[0] * x + m[1] * y + m[2] * z + m[3],
            m[4] * x + m[5] * y + m[6] * z + m[7],
            m[8] * x + m[9] * y + m[10] * z + m[11])


def hips_transform(hips: List[Float64], rup: List[Float64], lup: List[Float64]) -> List[Float64]:
    """Upstream get_hips_transform: moves Hips to the origin and turns the pelvis plane normal
    cross(RightUpLeg - Hips, LeftUpLeg - Hips) to +Z, then spins about Z so RightUpLeg -> LeftUpLeg
    points along +X."""
    var a = [rup[0] - hips[0], rup[1] - hips[1], rup[2] - hips[2]]
    var b = [lup[0] - hips[0], lup[1] - hips[1], lup[2] - hips[2]]
    var nx = a[1] * b[2] - a[2] * b[1]
    var ny = a[2] * b[0] - a[0] * b[2]
    var nz = a[0] * b[1] - a[1] * b[0]
    var nl = max(sqrt(nx * nx + ny * ny + nz * nz), 1e-12)
    nx /= nl
    ny /= nl
    nz /= nl
    # Rodrigues: rotate n onto (0, 0, 1)
    var cx = ny       # n x z = (ny, -nx, 0)
    var cy = -nx
    var dot = nz
    var cn2 = cx * cx + cy * cy
    var r = List[Float64](length=9, fill=0.0)
    if cn2 < 1e-24:
        r[0] = 1
        r[4] = 1 if dot > 0 else -1
        r[8] = 1 if dot > 0 else -1
    else:
        var sk = [0.0, 0.0, cy, 0.0, 0.0, -cx, -cy, cx, 0.0]
        var k = (1 - dot) / cn2
        for i in range(3):
            for j in range(3):
                var s2: Float64 = 0
                for e in range(3):
                    s2 += sk[i * 3 + e] * sk[e * 3 + j]
                r[i * 3 + j] = (1.0 if i == j else 0.0) + sk[i * 3 + j] + s2 * k
    var m = _ident()
    for i in range(3):
        for j in range(3):
            m[i * 4 + j] = r[i * 3 + j]
        m[i * 4 + 3] = -(r[i * 3] * hips[0] + r[i * 3 + 1] * hips[1] + r[i * 3 + 2] * hips[2])
    var tx = lup[0] - rup[0]
    var ty = lup[1] - rup[1]
    var tz = lup[2] - rup[2]
    var ang = atan2(r[3] * tx + r[4] * ty + r[5] * tz, r[0] * tx + r[1] * ty + r[2] * tz)
    var r2 = _ident()
    r2[0] = cos(-ang)
    r2[1] = -sin(-ang)
    r2[4] = sin(-ang)
    r2[5] = cos(-ang)
    return mat_mul(r2, m)


def ortho6d_matrix(o: List[Float32], j: Int) -> List[Float64]:
    """Upstream ortho6d_to_matrix of row j of a [*, 6] array: 3x3 row-major, columns x, y, z."""
    var xr = [Float64(o[j * 6]), Float64(o[j * 6 + 1]), Float64(o[j * 6 + 2])]
    var yr = [Float64(o[j * 6 + 3]), Float64(o[j * 6 + 4]), Float64(o[j * 6 + 5])]
    var xl = max(sqrt(xr[0] * xr[0] + xr[1] * xr[1] + xr[2] * xr[2]), 1e-12)
    var x = [xr[0] / xl, xr[1] / xl, xr[2] / xl]
    var z = [x[1] * yr[2] - x[2] * yr[1], x[2] * yr[0] - x[0] * yr[2], x[0] * yr[1] - x[1] * yr[0]]
    var zl = max(sqrt(z[0] * z[0] + z[1] * z[1] + z[2] * z[2]), 1e-12)
    for c in range(3):
        z[c] /= zl
    var y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    var m = List[Float64](length=9, fill=0.0)
    for r in range(3):
        m[r * 3] = x[r]
        m[r * 3 + 1] = y[r]
        m[r * 3 + 2] = z[r]
    return m^


def rest_transforms(pose6: List[Float32], joints: List[Float32]) -> List[Float64]:
    """Upstream pose_rot_to_global (then pose[0] = I): per joint a row-major 4x4 taking the input
    pose to the rest pose; `joints` is [52, 6] (heads used)."""
    var par = mia_parents()
    var g = List[Float64](length=MIA_JOINTS * 16, fill=0.0)
    for j in range(MIA_JOINTS):
        var r = ortho6d_matrix(pose6, j)
        var jx = Float64(joints[j * 6])
        var jy = Float64(joints[j * 6 + 1])
        var jz = Float64(joints[j * 6 + 2])
        var m = _ident()
        for a in range(3):
            for b in range(3):
                m[a * 4 + b] = r[a * 3 + b]
            m[a * 4 + 3] = [jx, jy, jz][a] - (r[a * 3] * jx + r[a * 3 + 1] * jy + r[a * 3 + 2] * jz)
        var p = par[j]
        if p >= 0:
            var pm = List[Float64](g[p * 16:(p + 1) * 16])
            var posed = mat_point(pm, jx, jy, jz)
            m[3] += posed[0] - jx
            m[7] += posed[1] - jy
            m[11] += posed[2] - jz
        for e in range(16):
            g[j * 16 + e] = m[e]
    var id = _ident()
    for e in range(16):
        g[e] = id[e]
    return g^


# ------------------------------------------------------------------------------- skin-weight clean-up

def _has_any(name: String, parts: List[String]) -> Bool:
    for p in parts:
        if p in name:
            return True
    return False


def _conflict(mut bw: List[Float32], dominant: List[Int], names: List[String], a: List[String], b: List[String]):
    """Rows whose dominant joint's name contains any of `a`: zero the columns containing any of `b`."""
    var in_a = List[Bool](length=MIA_JOINTS, fill=False)
    var cols = List[Int]()
    for j in range(MIA_JOINTS):
        in_a[j] = _has_any(names[j], a)
        if _has_any(names[j], b):
            cols.append(j)
    if len(cols) == 0:
        return
    for i in range(len(dominant)):
        if in_a[dominant[i]]:
            for c in cols:
                bw[i * MIA_JOINTS + c] = 0


def clean_weights(bw_in: List[Float32], verts_y: List[Float32], head_tail_y: Float32) -> List[Float32]:
    """Upstream bw_post_process (no extra bones, fingers kept): vertices above the head joint's
    tail go to Head, a vertex dominated by one limb / finger loses the weights of the mutually
    exclusive ones, rows renormalised, then only weights >= the 4th largest kept."""
    var J = MIA_JOINTS
    var nv = len(bw_in) // J
    var names = List[String]()
    for n in mia_joint_names():
        names.append(MIA_PREFIX + n)
    var bw = bw_in.copy()
    var head = mia_joint_index("Head")
    for i in range(nv):
        if verts_y[i] >= head_tail_y:
            bw[i * J + head] = 1e5
    var dominant = List[Int](length=nv, fill=0)
    for i in range(nv):
        var best = bw[i * J]
        for j in range(1, J):
            if bw[i * J + j] > best:
                best = bw[i * J + j]
                dominant[i] = j
    var fingers: List[String] = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    for hs in range(2):
        var hand = String("Left") if hs == 0 else String("Right")
        var other = String("Right") if hs == 0 else String("Left")
        _conflict(bw, dominant, names, [hand], [other])
        for fi in range(5):
            var others = List[String]()
            for fo in range(5):
                if fo != fi:
                    others.append(hand + "Hand" + fingers[fo])
            # upstream: dominant contains hand and finger; columns contain hand and another finger
            var in_a = List[Bool](length=J, fill=False)
            var cols = List[Int]()
            for j in range(J):
                in_a[j] = hand in names[j] and fingers[fi] in names[j]
                var hit = False
                if hand in names[j]:
                    for fo in range(5):
                        if fo != fi and fingers[fo] in names[j]:
                            hit = True
                if hit:
                    cols.append(j)
            for i in range(nv):
                if in_a[dominant[i]]:
                    for c in cols:
                        bw[i * J + c] = 0
    var sets = List[List[List[String]]]()
    sets.append([["Neck", "Head", "RabbitEar"], ["LeftLeg", "LeftFoot", "LeftToeBase"],
                 ["RightLeg", "RightFoot", "RightToeBase"],
                 ["LeftArm", "LeftForeArm", "LeftHand"], ["RightArm", "RightForeArm", "RightHand"]])
    sets.append([["Hips", "Spine", "Neck", "Head", "LeftShoulder", "RightShoulder", "RightUpLeg", "RightLeg",
                  "RightFoot", "RightToeBase", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "RabbitEar",
                  "FoxTail"],
                 ["RightForeArm", "RightHand", "LeftForeArm", "LeftHand"]])
    sets.append([["LRabbitEar"], ["RRabbitEar"]])
    sets.append([["FoxTail"], ["Neck", "Head", "RabbitEar"], ["RightFoot", "RightToeBase", "LeftFoot", "LeftToeBase"]])
    # (upstream lists every finger joint inside the arm parts; the "LeftHand" / "RightHand"
    # substrings above match exactly those names, and "Spine" matches Spine1 / Spine2)
    for s in range(len(sets)):
        for p in range(len(sets[s])):
            var others = List[String]()
            for q in range(len(sets[s])):
                if q != p:
                    for x in sets[s][q]:
                        others.append(x)
            _conflict(bw, dominant, names, sets[s][p], others)
    for i in range(nv):
        var tot: Float32 = 0
        for j in range(J):
            tot += bw[i * J + j]
        for j in range(J):
            bw[i * J + j] = bw[i * J + j] / (tot + Float32(1e-10))
        # 4th largest value
        var top = List[Float32](length=MIA_INFLUENCES, fill=Float32(-1))
        for j in range(J):
            var w = bw[i * J + j]
            var slot = MIA_INFLUENCES
            while slot > 0 and top[slot - 1] < w:
                slot -= 1
            if slot < MIA_INFLUENCES:
                var k = MIA_INFLUENCES - 1
                while k > slot:
                    top[k] = top[k - 1]
                    k -= 1
                top[slot] = w
        var thr = top[MIA_INFLUENCES - 1]
        for j in range(J):
            if bw[i * J + j] < thr:
                bw[i * J + j] = 0
    return bw^


def skin_to_rest(v: List[Float32], bw: List[Float32], pose: List[Float64]) -> List[Float32]:
    """Linear blend skinning of points (flat xyz) with [V, 52] weights and the rest transforms.
    Each row is divided by its sum, as Blender's armature deform does with upstream's (not
    renormalised) top-4 weights."""
    var nv = len(v) // 3
    var out = List[Float32](length=nv * 3, fill=0.0)
    for i in range(nv):
        var x = Float64(v[i * 3])
        var y = Float64(v[i * 3 + 1])
        var z = Float64(v[i * 3 + 2])
        var acc = List[Float64](length=3, fill=0.0)
        var tot: Float64 = 0
        for j in range(MIA_JOINTS):
            tot += Float64(bw[i * MIA_JOINTS + j])
        if tot <= 0:
            for r in range(3):
                out[i * 3 + r] = v[i * 3 + r]
            continue
        for j in range(MIA_JOINTS):
            var w = Float64(bw[i * MIA_JOINTS + j]) / tot
            if w == 0:
                continue
            var b = j * 16
            for r in range(3):
                acc[r] += w * (pose[b + r * 4] * x + pose[b + r * 4 + 1] * y + pose[b + r * 4 + 2] * z + pose[b + r * 4 + 3])
        for r in range(3):
            out[i * 3 + r] = Float32(acc[r])
    return out^


# ------------------------------------------------------------------------------- the networks

struct MiaRig(Movable):
    """Everything one run predicts, in the network frame (`to_net` maps the input frame there:
    uniform scale `scale`, a rotation and a shift) unless noted.
    joints [52, 6] (head xyz, tail xyz), weights [V, 52] (cleaned, <= 4 per row), weights_raw
    [V, 52], pose6 [52, 6], rest [52 x 16] row-major input-pose -> rest-pose transforms,
    rest_joints [52, 3], rest_verts [V, 3]."""
    var joints: List[Float32]
    var coarse: List[Float32]
    var weights: List[Float32]
    var weights_raw: List[Float32]
    var pose6: List[Float32]
    var rest: List[Float64]
    var rest_joints: List[Float32]
    var rest_verts: List[Float32]
    var to_net: List[Float64]
    var scale: Float64
    var seconds_networks: Float64

    def __init__(out self):
        self.joints = List[Float32]()
        self.coarse = List[Float32]()
        self.weights = List[Float32]()
        self.weights_raw = List[Float32]()
        self.pose6 = List[Float32]()
        self.rest = List[Float64]()
        self.rest_joints = List[Float32]()
        self.rest_verts = List[Float32]()
        self.to_net = _ident()
        self.scale = 1
        self.seconds_networks = 0


struct MakeItAnimatable(Movable):
    var coarse_net: OnnxModel
    var joints_net: OnnxModel
    var pose_net: OnnxModel
    var bw_trunk: OnnxModel
    var bw_head: OnnxModel
    var gpu: Bool
    var verbose: Bool

    def __init__(out self, var c: OnnxModel, var j: OnnxModel, var p: OnnxModel, var t: OnnxModel, var h: OnnxModel,
                 gpu: Bool):
        self.coarse_net = c^
        self.joints_net = j^
        self.pose_net = p^
        self.bw_trunk = t^
        self.bw_head = h^
        self.gpu = gpu
        self.verbose = False

    @staticmethod
    def load(dir: String, gpu: Bool = True) raises -> MakeItAnimatable:
        var c = OnnxModel.load(dir + "/mia_coarse.onnx")
        var j = OnnxModel.load(dir + "/mia_joints.onnx")
        var p = OnnxModel.load(dir + "/mia_pose.onnx")
        var t = OnnxModel.load(dir + "/mia_bw_trunk.onnx")
        var h = OnnxModel.load(dir + "/mia_bw_head.onnx")
        var on = False
        if gpu:
            on = c.use_gpu()
            if on:
                _ = j.use_gpu()
                _ = p.use_gpu()
                _ = t.use_gpu()
                _ = h.use_gpu()
        return MakeItAnimatable(c^, j^, p^, t^, h^, on)

    def device(self) raises -> String:
        return self.coarse_net.gpu_name() if self.gpu else String("CPU")

    def _log(self, what: String, t0: Int):
        if self.verbose:
            print("[make-it-animatable]", what, Float64(perf_counter_ns() - t0) / 1e9, "s")

    @staticmethod
    def _inputs(mut m: OnnxModel, pts: List[Float32], idx: List[Int]) raises:
        var n = len(pts) // 3
        m.set_input("pc", Tensor.floats([1, n, 3], pts.copy()))
        m.set_input("latent_pc", Tensor.floats([1, len(idx), 3], gather_points(pts, idx)))

    def coarse_joints(mut self, pts1: List[Float32], idx: List[Int]) raises -> List[Float32]:
        """[52, 6] joints from stage-1 points (bounding-box normalised) and their 512 FPS picks."""
        Self._inputs(self.coarse_net, pts1, idx)
        self.coarse_net.run()
        return self.coarse_net.output("joints").owned_floats()

    def fine_joints(mut self, pts2: List[Float32], idx: List[Int]) raises -> List[Float32]:
        Self._inputs(self.joints_net, pts2, idx)
        self.joints_net.run()
        return self.joints_net.output("joints").owned_floats()

    def rest_pose(mut self, pts2: List[Float32], idx: List[Int], joints: List[Float32]) raises -> List[Float32]:
        Self._inputs(self.pose_net, pts2, idx)
        self.pose_net.set_input("joints", Tensor.floats([1, MIA_JOINTS, 6], joints.copy()))
        self.pose_net.run()
        return self.pose_net.output("pose").owned_floats()

    def skin_weights(mut self, pts2: List[Float32], idx: List[Int], queries: List[Float32]) raises -> List[Float32]:
        """Softmax skin weights [V, 52] for query points (flat xyz, network frame)."""
        Self._inputs(self.bw_trunk, pts2, idx)
        self.bw_trunk.run()
        var lat = self.bw_trunk.output("latents")
        var nq = len(queries) // 3
        var out = List[Float32](capacity=nq * MIA_JOINTS)
        var s = 0
        while s < nq:
            var e = min(nq, s + MIA_BW_CHUNK)
            self.bw_head.set_input("latents", lat.copy())
            self.bw_head.set_input("queries", Tensor.floats([1, e - s, 3], List[Float32](queries[s * 3:e * 3])))
            self.bw_head.run()
            for x in self.bw_head.output("bw").owned_floats():
                out.append(x)
            s = e
        return out^

    def rig(mut self, v: List[Float32], f: List[Int], seed: UInt64 = 0) raises -> MiaRig:
        """The whole upstream pipeline on one Y-up triangle mesh (flat xyz, flat indices)."""
        var nv = len(v) // 3
        if nv == 0 or len(f) < 3:
            raise Error("Make-It-Animatable: empty mesh")
        var r = MiaRig()
        var rng = UrRng(seed * 0x9E3779B97F4A7C15 + 7)
        var t0 = perf_counter_ns()
        var t_net = 0
        var pts = sample_points(v, f, MIA_POINTS, rng)
        var n1 = normalize_points(pts, True)
        var pts1 = apply_scale_shift(pts, n1[0], n1[1])
        var tn = perf_counter_ns()
        r.coarse = self.coarse_joints(pts1, latent_indices(pts1, 0.0))
        t_net += perf_counter_ns() - tn
        self._log("coarse joints", t0)
        var hi = mia_joint_index("Hips") * 6
        var ri = mia_joint_index("RightUpLeg") * 6
        var li = mia_joint_index("LeftUpLeg") * 6
        var H = hips_transform([Float64(r.coarse[hi]), Float64(r.coarse[hi + 1]), Float64(r.coarse[hi + 2])],
                               [Float64(r.coarse[ri]), Float64(r.coarse[ri + 1]), Float64(r.coarse[ri + 2])],
                               [Float64(r.coarse[li]), Float64(r.coarse[li + 1]), Float64(r.coarse[li + 2])])
        var N1 = _ident()
        for c in range(3):
            N1[c * 5] = Float64(n1[0])
            N1[c * 4 + 3] = Float64(n1[1][c])
        var G = mat_mul(H, N1)
        var vh = List[Float32](length=nv * 3, fill=0.0)
        for i in range(nv):
            var p = mat_point(G, Float64(v[i * 3]), Float64(v[i * 3 + 1]), Float64(v[i * 3 + 2]))
            vh[i * 3] = Float32(p[0])
            vh[i * 3 + 1] = Float32(p[1])
            vh[i * 3 + 2] = Float32(p[2])
        var hands = List[Float64]()
        for name in ["LeftHand", "RightHand"]:
            var k = mia_joint_index(name) * 6
            var p = mat_point(H, Float64(r.coarse[k + 3]), Float64(r.coarse[k + 4]), Float64(r.coarse[k + 5]))
            hands.append(p[0])
            hands.append(p[1])
            hands.append(p[2])
        t0 = perf_counter_ns()
        var raw2 = sample_points_hands(vh, f, MIA_POINTS, hands, rng)
        var n2 = normalize_points(raw2, False)
        var pts2 = apply_scale_shift(raw2, n2[0], n2[1])
        var qv = apply_scale_shift(vh, n2[0], n2[1])
        var idx2 = latent_indices(pts2, MIA_HANDS_RATIO)
        self._log("hands re-sampling + FPS", t0)
        var N2 = _ident()
        for c in range(3):
            N2[c * 5] = Float64(n2[0])
        r.to_net = mat_mul(N2, G)
        r.scale = Float64(n1[0]) * Float64(n2[0])
        t0 = perf_counter_ns()
        tn = perf_counter_ns()
        r.weights_raw = self.skin_weights(pts2, idx2, qv)
        self._log("skin weights (" + String(nv) + " vertices)", t0)
        t0 = perf_counter_ns()
        r.joints = self.fine_joints(pts2, idx2)
        r.pose6 = self.rest_pose(pts2, idx2, r.joints)
        t_net += perf_counter_ns() - tn
        self._log("joints + rest pose", t0)
        r.seconds_networks = Float64(t_net) / 1e9
        t0 = perf_counter_ns()
        var ys = List[Float32](length=nv, fill=0.0)
        for i in range(nv):
            ys[i] = qv[i * 3 + 1]
        r.weights = clean_weights(r.weights_raw, ys, r.joints[mia_joint_index("Head") * 6 + 4])
        r.rest = rest_transforms(r.pose6, r.joints)
        r.rest_verts = skin_to_rest(qv, r.weights, r.rest)
        var heads = List[Float32](length=MIA_JOINTS * 3, fill=0.0)
        for j in range(MIA_JOINTS):
            for c in range(3):
                heads[j * 3 + c] = r.joints[j * 6 + c]
        r.rest_joints = List[Float32](length=MIA_JOINTS * 3, fill=0.0)
        for j in range(MIA_JOINTS):
            var m = List[Float64](r.rest[j * 16:(j + 1) * 16])
            var p = mat_point(m, Float64(heads[j * 3]), Float64(heads[j * 3 + 1]), Float64(heads[j * 3 + 2]))
            r.rest_joints[j * 3] = Float32(p[0])
            r.rest_joints[j * 3 + 1] = Float32(p[1])
            r.rest_joints[j * 3 + 2] = Float32(p[2])
        self._log("weight clean-up + rest pose", t0)
        return r^


# ------------------------------------------------------------------------------- glTF in / out

def top_influences(weights: List[Float32], mut joints: List[UInt16], mut out_w: List[Float32]):
    """[V, 52] weights -> the 4 largest per vertex, renormalised (a vertex with none: Hips)."""
    var J = MIA_JOINTS
    var nv = len(weights) // J
    joints = List[UInt16](length=nv * MIA_INFLUENCES, fill=0)
    out_w = List[Float32](length=nv * MIA_INFLUENCES, fill=0.0)
    for i in range(nv):
        var best = List[Int](length=MIA_INFLUENCES, fill=-1)
        for j in range(J):
            var wj = weights[i * J + j]
            if wj <= 0:
                continue
            var slot = MIA_INFLUENCES
            while slot > 0 and (best[slot - 1] < 0 or weights[i * J + best[slot - 1]] < wj):
                slot -= 1
            if slot < MIA_INFLUENCES:
                var k = MIA_INFLUENCES - 1
                while k > slot:
                    best[k] = best[k - 1]
                    k -= 1
                best[slot] = j
        var tot: Float32 = 0
        for s in range(MIA_INFLUENCES):
            if best[s] >= 0:
                tot += weights[i * J + best[s]]
        if tot <= 0:
            out_w[i * MIA_INFLUENCES] = 1.0
            continue
        for s in range(MIA_INFLUENCES):
            if best[s] >= 0:
                joints[i * MIA_INFLUENCES + s] = UInt16(best[s])
                out_w[i * MIA_INFLUENCES + s] = weights[i * J + best[s]] / tot


def _unit(v: Vec3) -> Vec3:
    var l = sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if l <= 1e-20:
        return Vec3(0, 1, 0)
    return Vec3(v.x / l, v.y / l, v.z / l)


def _invert_similarity(m: List[Float64], scale: Float64) -> List[Float64]:
    """Inverse of p -> s R p + t (row-major 4x4)."""
    var o = _ident()
    var s2 = scale * scale
    for i in range(3):
        for j in range(3):
            o[i * 4 + j] = m[j * 4 + i] / s2
    for i in range(3):
        o[i * 4 + 3] = -(o[i * 4] * m[3] + o[i * 4 + 1] * m[7] + o[i * 4 + 2] * m[11])
    return o^


def rig_native_model(mut mia: MakeItAnimatable, mut src: NativeModel, rest: Bool = True, seed: UInt64 = 0) raises -> NativeModel:
    """Rigs every mesh instance of a glTF scene together (baked to world space, one node each)
    and returns a skinned copy: 52 joint nodes `mixamorig:<Joint>` (translations only), one skin,
    4 influences per vertex. rest=True: mesh and skeleton in the predicted rest (T) pose, facing
    +Z, Hips above the origin, feet on y = 0, input units; rest=False: input pose and placement."""
    var inst_mesh = List[Int]()
    var inst_node = List[Int]()
    var inst_xf = List[Mat4]()
    if len(src.nodes) == 0:
        for mi in range(len(src.meshes)):
            inst_mesh.append(mi)
            inst_node.append(-1)
            inst_xf.append(Mat4())
    else:
        var world = src.global_transforms()
        for n in range(len(src.nodes)):
            var ms = src.nodes[n].mesh_indices.copy()
            if len(ms) == 0 and src.nodes[n].mesh >= 0:
                ms.append(src.nodes[n].mesh)
            for mi in ms:
                if mi >= 0 and mi < len(src.meshes):
                    inst_mesh.append(mi)
                    inst_node.append(n)
                    inst_xf.append(world[n].copy())
    var out = NativeModel()
    out.textures = src.textures.copy()
    var v = List[Float32]()
    var f = List[Int]()
    var base = List[Int]()
    for k in range(len(inst_mesh)):
        ref m = src.meshes[inst_mesh[k]]
        var nv = m.vertex_count()
        base.append(len(v) // 3)
        var mesh = m.copy()
        for i in range(nv):
            var p = inst_xf[k].transform_point(Vec3(m.positions[i * 3], m.positions[i * 3 + 1], m.positions[i * 3 + 2]))
            mesh.positions[i * 3] = p.x
            mesh.positions[i * 3 + 1] = p.y
            mesh.positions[i * 3 + 2] = p.z
            v.append(p.x)
            v.append(p.y)
            v.append(p.z)
            if i * 3 + 2 < len(m.normals):
                var nn = _unit(inst_xf[k].transform_vector(Vec3(m.normals[i * 3], m.normals[i * 3 + 1], m.normals[i * 3 + 2])))
                mesh.normals[i * 3] = nn.x
                mesh.normals[i * 3 + 1] = nn.y
                mesh.normals[i * 3 + 2] = nn.z
        # morph targets would not follow the rest-pose bake; drop them there
        if rest:
            mesh.morph_positions = List[Float32]()
            mesh.morph_normals = List[Float32]()
        else:
            for d in range(len(mesh.morph_positions) // 3):
                var dp = inst_xf[k].transform_vector(Vec3(mesh.morph_positions[d * 3], mesh.morph_positions[d * 3 + 1], mesh.morph_positions[d * 3 + 2]))
                mesh.morph_positions[d * 3] = dp.x
                mesh.morph_positions[d * 3 + 1] = dp.y
                mesh.morph_positions[d * 3 + 2] = dp.z
            for d in range(len(mesh.morph_normals) // 3):
                var dn = inst_xf[k].transform_vector(Vec3(mesh.morph_normals[d * 3], mesh.morph_normals[d * 3 + 1], mesh.morph_normals[d * 3 + 2]))
                mesh.morph_normals[d * 3] = dn.x
                mesh.morph_normals[d * 3 + 1] = dn.y
                mesh.morph_normals[d * 3 + 2] = dn.z
        if len(m.indices) > 0:
            for idx in m.indices:
                f.append(base[k] + idx)
        else:
            for idx in range(nv - nv % 3):
                f.append(base[k] + idx)
        mesh.centers = List[Float32]()
        mesh.bounds_verts = 0
        out.meshes.append(mesh^)
    if len(v) == 0:
        raise Error("Make-It-Animatable: the model has no mesh vertices")

    var r = mia.rig(v, f, seed)
    var jt = List[UInt16]()
    var jw = List[Float32]()
    top_influences(r.weights, jt, jw)

    # final joint positions and (rest mode) vertex positions / normals
    var jpos = List[Vec3]()
    if rest:
        var inv_s = 1.0 / r.scale
        var lo_y: Float64 = 1e300
        var ntot = len(r.rest_verts) // 3
        for i in range(ntot):
            lo_y = min(lo_y, Float64(r.rest_verts[i * 3 + 1]))
        var ox = Float64(r.rest_joints[0])
        var oz = Float64(r.rest_joints[2])
        for j in range(MIA_JOINTS):
            jpos.append(Vec3(Float32((Float64(r.rest_joints[j * 3]) - ox) * inv_s),
                             Float32((Float64(r.rest_joints[j * 3 + 1]) - lo_y) * inv_s),
                             Float32((Float64(r.rest_joints[j * 3 + 2]) - oz) * inv_s)))
        # rotation of each vertex's blended rest transform, for the normals
        var rot_net = List[Float64](length=9, fill=0.0)
        for i in range(3):
            for j in range(3):
                rot_net[i * 3 + j] = r.to_net[i * 4 + j] / r.scale
        for k in range(len(out.meshes)):
            var nv = out.meshes[k].vertex_count()
            var b = base[k]
            var has_n = len(out.meshes[k].normals) >= nv * 3
            for i in range(nv):
                var g = b + i
                out.meshes[k].positions[i * 3] = Float32((Float64(r.rest_verts[g * 3]) - ox) * inv_s)
                out.meshes[k].positions[i * 3 + 1] = Float32((Float64(r.rest_verts[g * 3 + 1]) - lo_y) * inv_s)
                out.meshes[k].positions[i * 3 + 2] = Float32((Float64(r.rest_verts[g * 3 + 2]) - oz) * inv_s)
                if has_n:
                    var n0 = Float64(out.meshes[k].normals[i * 3])
                    var n1 = Float64(out.meshes[k].normals[i * 3 + 1])
                    var n2 = Float64(out.meshes[k].normals[i * 3 + 2])
                    var nn = List[Float64](length=3, fill=0.0)
                    for e in range(3):
                        nn[e] = rot_net[e * 3] * n0 + rot_net[e * 3 + 1] * n1 + rot_net[e * 3 + 2] * n2
                    var acc = List[Float64](length=3, fill=0.0)
                    for j in range(MIA_JOINTS):
                        var w = Float64(r.weights[g * MIA_JOINTS + j])
                        if w == 0:
                            continue
                        for e in range(3):
                            acc[e] += w * (r.rest[j * 16 + e * 4] * nn[0] + r.rest[j * 16 + e * 4 + 1] * nn[1]
                                           + r.rest[j * 16 + e * 4 + 2] * nn[2])
                    var un = _unit(Vec3(Float32(acc[0]), Float32(acc[1]), Float32(acc[2])))
                    out.meshes[k].normals[i * 3] = un.x
                    out.meshes[k].normals[i * 3 + 1] = un.y
                    out.meshes[k].normals[i * 3 + 2] = un.z
    else:
        var back = _invert_similarity(r.to_net, r.scale)
        for j in range(MIA_JOINTS):
            var p = mat_point(back, Float64(r.joints[j * 6]), Float64(r.joints[j * 6 + 1]), Float64(r.joints[j * 6 + 2]))
            jpos.append(Vec3(Float32(p[0]), Float32(p[1]), Float32(p[2])))

    for k in range(len(inst_mesh)):
        var node = NativeNode()
        node.name = src.nodes[inst_node[k]].name if inst_node[k] >= 0 else "mesh_" + String(k)
        node.mesh = k
        out.nodes.append(node^)
        out.roots.append(k)
        var nv = out.meshes[k].vertex_count()
        var b = base[k]
        out.meshes[k].joints = List[UInt16](length=nv * MIA_INFLUENCES, fill=0)
        out.meshes[k].weights = List[Float32](length=nv * MIA_INFLUENCES, fill=0.0)
        for e in range(nv * MIA_INFLUENCES):
            out.meshes[k].joints[e] = jt[b * MIA_INFLUENCES + e]
            out.meshes[k].weights[e] = jw[b * MIA_INFLUENCES + e]
        out.meshes[k].skin = 0
    add_skeleton(out, jpos)
    out.max_influences = MIA_INFLUENCES
    return out^


def add_skeleton(mut out: NativeModel, jpos: List[Vec3]):
    """Appends the 52 `mixamorig:` joint nodes (identity rotations, bound at `jpos`) and a skin."""
    var names = mia_joint_names()
    var par = mia_parents()
    var first = len(out.nodes)
    var skin = NativeSkin()
    for j in range(MIA_JOINTS):
        var node = NativeNode()
        node.name = MIA_PREFIX + names[j]
        var p = par[j]
        var wp = jpos[j]
        node.translation = Vec3(wp.x - jpos[p].x, wp.y - jpos[p].y, wp.z - jpos[p].z) if p >= 0 else wp
        node.base_translation = node.translation
        out.nodes.append(node^)
        skin.joints.append(first + j)
        var ib = Mat4()
        for c in range(16):
            skin.inverse_bind.append(ib.values[c])
        skin.inverse_bind[j * 16 + 12] = -wp.x
        skin.inverse_bind[j * 16 + 13] = -wp.y
        skin.inverse_bind[j * 16 + 14] = -wp.z
    for j in range(MIA_JOINTS):
        if par[j] >= 0:
            out.nodes[first + par[j]].children.append(first + j)
        else:
            out.roots.append(first + j)
    out.skins.append(skin^)


# ------------------------------------------------------------------------------- retargeting

def retarget_clip(clip: MbClip, model: NativeModel, name: String = "retargeted") raises -> NativeAnimation:
    """A clip on a rig made by `rig_native_model(rest=True)` (joint nodes found by their
    `mixamorig:` names). The source is a SOMA clip (MotionBricks BVH, `gemx_clip`; joints mapped
    with `mia_soma_name`) or a Mixamo-named one (BVH exported from Mixamo; names used as they
    are), whose skeleton reference pose is a T-pose facing +Z. Each mapped joint takes the
    source joint's world rotation relative to its reference (the rig's joint frames are
    world-aligned in the T-pose), unmapped joints follow their parent, and the Hips path is
    scaled by the ratio of the two hip heights."""
    var names = mia_joint_names()
    var par = mia_parents()
    var node = List[Int](length=MIA_JOINTS, fill=-1)
    for n in range(len(model.nodes)):
        var j = mia_joint_index(model.nodes[n].name)
        if j >= 0 and model.nodes[n].name.startswith(MIA_PREFIX):
            node[j] = n
    for j in range(MIA_JOINTS):
        if node[j] < 0:
            raise Error("retarget_clip: the model has no node " + MIA_PREFIX + names[j])
    ref sk = clip.skeleton
    var mixamo = sk.joint_index("LeftUpLeg") >= 0 or sk.joint_index(MIA_PREFIX + "LeftUpLeg") >= 0
    var src = List[Int](length=MIA_JOINTS, fill=-1)
    for j in range(MIA_JOINTS):
        if mixamo:
            src[j] = sk.joint_index(names[j])
            if src[j] < 0:
                src[j] = sk.joint_index(MIA_PREFIX + names[j])
        else:
            src[j] = sk.joint_index(mia_soma_name(names[j]))
    if src[0] < 0:
        raise Error("retarget_clip: the clip has no Hips joint")
    var ref_world = world_pose(sk, sk.reference, MbXform.identity())
    # rig rest: Hips world position = its node translation (the root)
    var rest_hips = model.nodes[node[0]].base_translation
    var src_hips = ref_world[src[0]].p
    var ratio: Float32 = 1.0
    if abs(src_hips.y) > 1e-6:
        ratio = rest_hips.y / src_hips.y
    var anim = NativeAnimation()
    anim.name = name
    var nf = clip.num_frames()
    var times = List[Float32](capacity=nf)
    for fi in range(nf):
        times.append(Float32(fi) / clip.sample_rate)
    var rot_vals = List[List[Float32]]()
    for _ in range(MIA_JOINTS):
        rot_vals.append(List[Float32](capacity=nf * 4))
    var pos_vals = List[Float32](capacity=nf * 3)
    for fi in range(nf):
        var w = world_pose(sk, clip.frames[fi], MbXform.identity())
        var g = List[MbQuat](capacity=MIA_JOINTS)
        for j in range(MIA_JOINTS):
            var s = src[j]
            var q: MbQuat
            if s >= 0:
                q = (w[s].q * ref_world[s].q.inverse()).normalized()
            elif par[j] >= 0:
                q = g[par[j]]
            else:
                q = MbQuat.identity()
            g.append(q)
            var local = q if par[j] < 0 else (g[par[j]].inverse() * q).normalized()
            rot_vals[j].append(local.x)
            rot_vals[j].append(local.y)
            rot_vals[j].append(local.z)
            rot_vals[j].append(local.w)
        var d = w[src[0]].p - src_hips
        pos_vals.append(rest_hips.x + d.x * ratio)
        pos_vals.append(rest_hips.y + d.y * ratio)
        pos_vals.append(rest_hips.z + d.z * ratio)
    var ch = NativeChannel()
    ch.node = node[0]
    ch.path = 0
    ch.times = times.copy()
    ch.values = pos_vals^
    anim.channels.append(ch^)
    for j in range(MIA_JOINTS):
        var c = NativeChannel()
        c.node = node[j]
        c.path = 1
        c.times = times.copy()
        c.values = rot_vals[j].copy()
        anim.channels.append(c^)
    anim.duration = clip.duration()
    return anim^
