"""Deterministic inverse kinematics: analytic two-bone IK, exact hinge-plane
pole-vector two-bone IK, zero-allocation & cone-constrained FABRIK, mecha
closed-loop 4-bar Freudenstein linkage, and 3-bone digitigrade walker IK.

All trigonometric and exponential operations use `flint.core.detmath` so IK
joint poses and combat hitboxes are 100% cross-ISA lockstep-deterministic.
"""
from std.math import isfinite, sqrt
from flint.core.detmath import det_sin, det_cos, det_acos, det_atan2, det_exp, PI
from flint.core.math import Vec3
from flint.core.transform import Quat


@always_inline
def dot(a: Vec3, b: Vec3) -> Float32:
    return a.x * b.x + a.y * b.y + a.z * b.z


@always_inline
def cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


@always_inline
def clamp1(value: Float32) -> Float32:
    if not isfinite(value):
        return Float32(0.0)
    return max(Float32(-1.0), min(Float32(1.0), value))


@always_inline
def _vec3_finite(v: Vec3) -> Bool:
    return isfinite(v.x) and isfinite(v.y) and isfinite(v.z)


@always_inline
def _quat_finite(q: Quat) -> Bool:
    return isfinite(q.x) and isfinite(q.y) and isfinite(q.z) and isfinite(q.w)


@always_inline
def safe_normalized(value: Vec3, fallback: Vec3) -> Vec3:
    if not _vec3_finite(value):
        return fallback
    var length = value.length()
    if not isfinite(length) or length < Float32(0.000001):
        return fallback
    var res = value * (Float32(1.0) / length)
    if not _vec3_finite(res):
        return fallback
    return res


@always_inline
def ortho_fallback(n: Vec3) -> Vec3:
    """Deterministic unit vector orthogonal to unit vector `n`."""
    if abs(n.x) < Float32(0.57735):
        return safe_normalized(Vec3(0.0, n.z, -n.y), Vec3(0.0, 1.0, 0.0))
    return safe_normalized(Vec3(n.z, 0.0, -n.x), Vec3(1.0, 0.0, 0.0))


@always_inline
def angle_axis(angle: Float32, axis: Vec3) -> Quat:
    var half = angle * Float32(0.5)
    var s = det_sin(half)
    return Quat(axis.x * s, axis.y * s, axis.z * s, det_cos(half))


@always_inline
def shortest_arc_quat(from_dir: Vec3, to_dir: Vec3) -> Quat:
    """World-space shortest-arc unit quaternion rotating `from_dir` onto `to_dir`."""
    var u = safe_normalized(from_dir, Vec3(0.0, -1.0, 0.0))
    var v = safe_normalized(to_dir, u)
    var d = clamp1(dot(u, v))
    if d > Float32(0.999995):
        return Quat()
    if d < Float32(-0.9999):
        return angle_axis(PI, ortho_fallback(u))
    var ax = safe_normalized(cross(u, v), ortho_fallback(u))
    return angle_axis(det_acos(d), ax).normalized()


@always_inline
def _soft_clamp_distance(d_target: Float32, lab: Float32, lcb: Float32, eps: Float32) -> Float32:
    """Smooth C1 exponential soft-compression near inner fold `|lab - lcb|` and
    soft-extension near outer reach `lab + lcb` so targets close to the hip or
    past full reach never kink or snap."""
    var max_reach = lab + lcb
    var d_max = max_reach - eps
    var d_min = abs(lab - lcb) + eps
    if d_max <= d_min:
        return d_min
    var span = d_max - d_min
    var d_soft = max(eps, min(max_reach * Float32(0.04), span * Float32(0.45)))
    if d_target > d_max - d_soft:
        return d_max - d_soft * det_exp(-(d_target - (d_max - d_soft)) / d_soft)
    if d_target < d_min + d_soft:
        var excess = (d_min + d_soft) - d_target
        return d_min + d_soft * det_exp(-excess / d_soft)
    return d_target


struct TwoBoneResult(Copyable, ImplicitlyCopyable, Movable):
    """Rotations to multiply onto the upper and middle joints' local rotations."""
    var upper: Quat
    var middle: Quat

    def __init__(out self, upper: Quat, middle: Quat):
        self.upper = upper
        self.middle = middle


def two_bone_ik(
    a: Vec3, b: Vec3, c: Vec3, target: Vec3,
    a_global: Quat, b_global: Quat, pole: Vec3,
) -> TwoBoneResult:
    """Two-bone IK solver: a, b, c are upper, middle and end joint positions.
    a_global, b_global: the upper and middle joints' global rotations.
    pole: direction the middle joint should bend toward when straight."""
    if not (
        _vec3_finite(a)
        and _vec3_finite(b)
        and _vec3_finite(c)
        and _vec3_finite(target)
        and _vec3_finite(pole)
        and _quat_finite(a_global)
        and _quat_finite(b_global)
    ):
        return TwoBoneResult(Quat(), Quat())
    var eps = Float32(0.001)
    var lab = (b - a).length()
    var lcb = (b - c).length()
    if lab < eps or lcb < eps:
        return TwoBoneResult(Quat(), Quat())
    var d_target = (target - a).length()
    var lat = _soft_clamp_distance(d_target, lab, lcb, eps)

    var ac = safe_normalized(c - a, Vec3(0, -1, 0))
    var ab = safe_normalized(b - a, Vec3(0, -1, 0))
    var ba = -ab
    var bc = safe_normalized(c - b, Vec3(0, -1, 0))
    var ac_ab_0 = det_acos(clamp1(dot(ac, ab)))
    var ba_bc_0 = det_acos(clamp1(dot(ba, bc)))
    var ac_ab_1 = det_acos(clamp1((lcb * lcb - lab * lab - lat * lat) / (Float32(-2.0) * lab * lat)))
    var ba_bc_1 = det_acos(clamp1((lat * lat - lab * lab - lcb * lcb) / (Float32(-2.0) * lab * lcb)))

    var bend = cross(ac, ab)
    if bend.length() < Float32(0.0001):
        var pole_dir = safe_normalized(pole, ortho_fallback(ac))
        bend = cross(ac, pole_dir)
        if bend.length() < Float32(0.0001):
            bend = ortho_fallback(ac)
    var axis0 = safe_normalized(bend, Vec3(1, 0, 0))
    var a_inverse = a_global.conjugate()
    var b_inverse = b_global.conjugate()
    var r0 = angle_axis(ac_ab_1 - ac_ab_0, a_inverse.rotate(axis0)).normalized()
    var r1 = angle_axis(ba_bc_1 - ba_bc_0, b_inverse.rotate(axis0)).normalized()
    if not (_quat_finite(r0) and _quat_finite(r1)):
        return TwoBoneResult(Quat(), Quat())
    return TwoBoneResult(r0, r1)


def aim_rotation(a: Vec3, c: Vec3, target: Vec3, a_global: Quat) -> Quat:
    """Local rotation for joint a that turns direction a->c onto a->target."""
    if not (_vec3_finite(a) and _vec3_finite(c) and _vec3_finite(target) and _quat_finite(a_global)):
        return Quat()
    var from_direction = safe_normalized(c - a, Vec3(0, -1, 0))
    var to_direction = safe_normalized(target - a, from_direction)
    var d = clamp1(dot(from_direction, to_direction))
    if d < Float32(-0.9999):
        var ortho = ortho_fallback(from_direction)
        return angle_axis(PI, a_global.conjugate().rotate(ortho)).normalized()
    var axis = cross(from_direction, to_direction)
    var angle = det_acos(d)
    if axis.length() < Float32(0.000001) or angle < Float32(0.000001):
        return Quat()
    return angle_axis(angle, a_global.conjugate().rotate(safe_normalized(axis, Vec3(1, 0, 0)))).normalized()


def solve_two_bone_pole(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    target: Vec3,
    pole_pos: Vec3,
    gr_a: Quat,
    gr_b: Quat,
) -> Tuple[Quat, Quat]:
    """Complete two-bone IK with exact world pole-position tracking.

    Folds:
    1. Law-of-cosines flexion (`two_bone_ik` with smooth inner/outer reach),
    2. Exact swing of the bent chain onto `u_at = normalize(target - a)`,
    3. Exact hinge-plane pole twist around `u_at` so the middle joint (knee/elbow)
       tracks `pole_pos` smoothly across all bend angles without end-effector drift.

    Returns `(dq_upper, dq_middle)` to post-multiply onto the local rotations of
    joints `a` and `b` (`r_a = r_a.multiply(dq_upper)`, `r_b = r_b.multiply(dq_middle)`).
    """
    if not (
        _vec3_finite(a)
        and _vec3_finite(b)
        and _vec3_finite(c)
        and _vec3_finite(target)
        and _vec3_finite(pole_pos)
        and _quat_finite(gr_a)
        and _quat_finite(gr_b)
    ):
        return (Quat(), Quat())
    var eps = Float32(0.001)
    var lab = (b - a).length()
    var lcb = (b - c).length()
    if lab < eps or lcb < eps:
        return (Quat(), Quat())

    var ga = gr_a.normalized()
    var gb = gr_b.normalized()
    var pole_hint = safe_normalized(pole_pos - a, Vec3(0, 0, 1))
    var flex = two_bone_ik(a, b, c, target, ga, gb, pole_hint)

    # Evaluate bent joint positions in world/model space before swing + pole twist:
    # ga_1 = ga * flex.upper, gb_1 = ga_1 * (ga^* * gb) * flex.middle
    var v_ab_loc = ga.conjugate().rotate(b - a)
    var v_bc_loc = gb.conjugate().rotate(c - b)
    var r_b_loc = ga.conjugate().multiply(gb)

    var ga_1 = ga.multiply(flex.upper).normalized()
    var gb_1 = ga_1.multiply(r_b_loc).multiply(flex.middle).normalized()
    var b_1 = a + ga_1.rotate(v_ab_loc)
    var c_1 = b_1 + gb_1.rotate(v_bc_loc)

    # Swing bent a -> c_1 onto u_at = normalize(target - a)
    var from_ac = safe_normalized(c_1 - a, Vec3(0, -1, 0))
    var u_at = safe_normalized(target - a, from_ac)
    var w_aim = shortest_arc_quat(from_ac, u_at)
    var b_2 = a + w_aim.rotate(b_1 - a)

    # Project current middle offset and pole vector orthogonal to hinge axis u_at
    var cur_vec = b_2 - a
    var pole_vec = pole_pos - a
    var p_cur = cur_vec - u_at * dot(cur_vec, u_at)
    var p_pole = pole_vec - u_at * dot(pole_vec, u_at)

    var w_twist = Quat()
    var len_cur = p_cur.length()
    var len_pole = p_pole.length()
    if len_cur > Float32(0.00005) and len_pole > Float32(0.00005):
        var e_cur = p_cur * (Float32(1.0) / len_cur)
        var e_pole = p_pole * (Float32(1.0) / len_pole)
        var sin_phi = dot(cross(e_cur, e_pole), u_at)
        var cos_phi = clamp1(dot(e_cur, e_pole))
        var phi = det_atan2(sin_phi, cos_phi)
        w_twist = angle_axis(phi, u_at)

    # Final global rotation of joint a: ga_final = w_twist * w_aim * ga_1
    var ga_final = w_twist.multiply(w_aim).multiply(ga_1).normalized()
    var dq_upper = ga.conjugate().multiply(ga_final).normalized()
    if not (_quat_finite(dq_upper) and _quat_finite(flex.middle)):
        return (Quat(), Quat())
    return (dq_upper, flex.middle)


def fabrik(
    mut points: List[Vec3],
    target: Vec3,
    mut lengths_scratch: List[Float32],
    iterations: Int = 12,
    tolerance: Float32 = 0.001,
) -> Bool:
    """Zero-allocation FABRIK solver using caller-owned `lengths_scratch`."""
    var count = len(points)
    if count < 2 or not _vec3_finite(target):
        return False
    for index in range(count):
        if not _vec3_finite(points[index]):
            return False
    var tol = tolerance if (isfinite(tolerance) and tolerance >= Float32(0.0)) else Float32(0.001)
    var segs = count - 1
    if len(lengths_scratch) < segs:
        lengths_scratch.resize(segs, Float32(0.0))
    var total = Float32(0.0)
    for index in range(segs):
        var length = (points[index + 1] - points[index]).length()
        lengths_scratch[index] = length
        total += length
    if not isfinite(total):
        return False
    var root = points[0]
    var target_dist = (target - root).length()
    if not isfinite(target_dist) or target_dist >= total:
        var direction = safe_normalized(target - root, Vec3(0, 1, 0))
        for index in range(1, count):
            points[index] = points[index - 1] + direction * lengths_scratch[index - 1]
        return False
    for _ in range(iterations):
        if (points[count - 1] - target).length() <= tol:
            return True
        points[count - 1] = target
        for index in range(count - 2, -1, -1):
            var direction = safe_normalized(points[index] - points[index + 1], Vec3(0, 1, 0))
            points[index] = points[index + 1] + direction * lengths_scratch[index]
        points[0] = root
        for index in range(1, count):
            var direction = safe_normalized(points[index] - points[index - 1], Vec3(0, 1, 0))
            points[index] = points[index - 1] + direction * lengths_scratch[index - 1]
    return (points[count - 1] - target).length() <= tol


def fabrik(mut points: List[Vec3], target: Vec3, iterations: Int = 12, tolerance: Float32 = 0.001) -> Bool:
    """Moves a joint chain (`points[0]` is fixed) so its end reaches `target`.
    Uses stack storage for chains up to 32 segments (zero heap allocations)."""
    var count = len(points)
    if count < 2 or not _vec3_finite(target):
        return False
    for index in range(count):
        if not _vec3_finite(points[index]):
            return False
    var tol = tolerance if (isfinite(tolerance) and tolerance >= Float32(0.0)) else Float32(0.001)
    var segs = count - 1
    if segs <= 32:
        var stack_lengths = Array[Float32, 32](fill=0.0)
        var total = Float32(0.0)
        for index in range(segs):
            var length = (points[index + 1] - points[index]).length()
            stack_lengths[index] = length
            total += length
        if not isfinite(total):
            return False
        var root = points[0]
        var target_dist = (target - root).length()
        if not isfinite(target_dist) or target_dist >= total:
            var direction = safe_normalized(target - root, Vec3(0, 1, 0))
            for index in range(1, count):
                points[index] = points[index - 1] + direction * stack_lengths[index - 1]
            return False
        for _ in range(iterations):
            if (points[count - 1] - target).length() <= tol:
                return True
            points[count - 1] = target
            for index in range(count - 2, -1, -1):
                var direction = safe_normalized(points[index] - points[index + 1], Vec3(0, 1, 0))
                points[index] = points[index + 1] + direction * stack_lengths[index]
            points[0] = root
            for index in range(1, count):
                var direction = safe_normalized(points[index] - points[index - 1], Vec3(0, 1, 0))
                points[index] = points[index - 1] + direction * stack_lengths[index - 1]
        return (points[count - 1] - target).length() <= tol
    var heap_lengths = List[Float32](length=segs, fill=0.0)
    return fabrik(points, target, heap_lengths, iterations, tol)


def clamp_cone_direction(dir: Vec3, ref_dir: Vec3, max_angle_rad: Float32) -> Vec3:
    """Clamps unit vector `dir` to lie within `max_angle_rad` cone around `ref_dir`."""
    var ref_u = safe_normalized(ref_dir, Vec3(0, 1, 0))
    var u = safe_normalized(dir, ref_u)
    if not isfinite(max_angle_rad):
        return ref_u
    if max_angle_rad >= PI - Float32(1.0e-4):
        return u
    if max_angle_rad <= Float32(1.0e-6):
        return ref_u
    var c = clamp1(dot(u, ref_u))
    var ang = det_acos(c)
    if ang <= max_angle_rad:
        return u
    var perp = u - ref_u * c
    var perp_u = safe_normalized(perp, ortho_fallback(ref_u))
    var res = ref_u * det_cos(max_angle_rad) + perp_u * det_sin(max_angle_rad)
    return safe_normalized(res, ref_u)


def fabrik_constrained(
    mut points: List[Vec3],
    rest_points: List[Vec3],
    target: Vec3,
    cone_limits_rad: List[Float32],
    mut out_rotations: List[Quat],
    mut lengths_scratch: List[Float32],
    iterations: Int = 12,
    tolerance: Float32 = 0.001,
) -> Bool:
    """Constrained FABRIK solver with per-joint conical limits (`cone_limits_rad`
    in radians) and minimal-twist global joint `Quat` reconstruction in `out_rotations`."""
    var count = len(points)
    if count < 2 or len(rest_points) < count or not _vec3_finite(target):
        return False
    for index in range(count):
        if not _vec3_finite(points[index]) or not _vec3_finite(rest_points[index]):
            return False
    var segs = count - 1
    if len(lengths_scratch) < segs:
        lengths_scratch.resize(segs, Float32(0.0))
    if len(out_rotations) != count:
        out_rotations.resize(count, Quat())

    for index in range(segs):
        lengths_scratch[index] = max(Float32(1.0e-5), (rest_points[index + 1] - rest_points[index]).length())

    var root = points[0]
    var base_dir0 = safe_normalized(rest_points[1] - rest_points[0], Vec3(0, 1, 0))

    for _ in range(iterations):
        if (points[count - 1] - target).length() <= tolerance:
            break
        # Backward reaching pass (end-effector -> root)
        points[count - 1] = target
        for index in range(count - 2, -1, -1):
            var back_dir = safe_normalized(points[index] - points[index + 1], -base_dir0)
            points[index] = points[index + 1] + back_dir * lengths_scratch[index]

        # Forward reaching pass (root -> end-effector) with conical limits
        points[0] = root
        var prev_q = Quat()
        for index in range(segs):
            var rest_dir = safe_normalized(rest_points[index + 1] - rest_points[index], base_dir0)
            var ref_dir = base_dir0 if index == 0 else prev_q.rotate(rest_dir)
            var raw_dir = safe_normalized(points[index + 1] - points[index], ref_dir)
            var max_cone = cone_limits_rad[index] if index < len(cone_limits_rad) else PI
            var clamped_dir = clamp_cone_direction(raw_dir, ref_dir, max_cone)
            points[index + 1] = points[index] + clamped_dir * lengths_scratch[index]
            var delta_q = shortest_arc_quat(ref_dir, clamped_dir)
            prev_q = delta_q.multiply(prev_q).normalized()

    # Reconstruct minimal-twist global joint rotations mapping rest bone vectors to solved bone vectors
    var transport_q = Quat()
    for index in range(segs):
        var rest_dir = safe_normalized(rest_points[index + 1] - rest_points[index], base_dir0)
        var cur_dir = safe_normalized(points[index + 1] - points[index], rest_dir)
        var transported_rest = transport_q.rotate(rest_dir)
        var swing_q = shortest_arc_quat(transported_rest, cur_dir)
        transport_q = swing_q.multiply(transport_q).normalized()
        out_rotations[index] = transport_q
    out_rotations[count - 1] = out_rotations[segs - 1]
    return (points[count - 1] - target).length() <= tolerance


def fabrik_constrained(
    mut points: List[Vec3],
    target: Vec3,
    cone_limits_rad: List[Float32],
    mut out_rotations: List[Quat],
    iterations: Int = 12,
    tolerance: Float32 = 0.001,
) -> Bool:
    """Convenience overload using initial `points` as the rest chain."""
    var rest_copy = points.copy()
    var lengths_scratch = List[Float32](length=max(1, len(points) - 1), fill=0.0)
    return fabrik_constrained(points, rest_copy, target, cone_limits_rad, out_rotations, lengths_scratch, iterations, tolerance)


def solve_four_bar_linkage(
    ground_a: Vec3,
    ground_d: Vec3,
    len_crank_ab: Float32,
    len_coupler_bc: Float32,
    len_rocker_cd: Float32,
    crank_angle: Float32,
    plane_normal: Vec3,
    sign_branch: Float32 = 1.0,
) -> Tuple[Vec3, Vec3, Float32, Float32]:
    """Exact analytic 3D circle-circle / Freudenstein closed-loop 4-bar linkage solver.

    Solves closed loop A -> B -> C -> D in the plane orthogonal to `plane_normal`,
    with fixed pivots `ground_a` (A) and `ground_d` (D), input crank `len_crank_ab`
    driven at `crank_angle` (radians from A->D in the linkage plane), coupler
    `len_coupler_bc`, and output rocker `len_rocker_cd`.
    `sign_branch` (+1.0 or -1.0) selects between the open and crossed dyad branches.

    Returns `(joint_b, joint_c, coupler_angle_rad, rocker_angle_rad)`.
    """
    if not (
        _vec3_finite(ground_a)
        and _vec3_finite(ground_d)
        and _vec3_finite(plane_normal)
        and isfinite(len_crank_ab)
        and isfinite(len_coupler_bc)
        and isfinite(len_rocker_cd)
        and isfinite(crank_angle)
    ):
        var safe_a = ground_a if _vec3_finite(ground_a) else Vec3()
        return (safe_a, safe_a, Float32(0.0), Float32(0.0))

    var n = safe_normalized(plane_normal, Vec3(0, 0, 1))
    var ad_raw = ground_d - ground_a
    var ad_plane = ad_raw - n * dot(ad_raw, n)
    var ex = safe_normalized(ad_plane, ortho_fallback(n))
    var ey = safe_normalized(cross(n, ex), ortho_fallback(ex))

    var lab = max(Float32(1.0e-5), len_crank_ab)
    var r1 = max(Float32(1.0e-5), len_coupler_bc)
    var r2 = max(Float32(1.0e-5), len_rocker_cd)

    var joint_b = ground_a + (ex * det_cos(crank_angle) + ey * det_sin(crank_angle)) * lab

    # Project D onto the linkage plane passing through A so circle-circle stays planar
    var d_plane = ground_a + ad_plane
    var bd = d_plane - joint_b
    var dist = bd.length()
    var u = safe_normalized(bd, ex)
    var v = cross(n, u)
    var branch = Float32(1.0) if (not isfinite(sign_branch) or sign_branch >= Float32(0.0)) else Float32(-1.0)

    var joint_c: Vec3
    if not isfinite(dist):
        return (ground_a, ground_a, Float32(0.0), Float32(0.0))
    if dist >= r1 + r2 - Float32(1.0e-6):
        # Fully stretched limit
        joint_c = joint_b + u * r1
    elif dist <= abs(r1 - r2) + Float32(1.0e-6):
        # Fully folded limit
        var fold_sign = Float32(1.0) if r1 >= r2 else Float32(-1.0)
        joint_c = joint_b + u * (r1 * fold_sign)
    else:
        var a_proj = (r1 * r1 - r2 * r2 + dist * dist) / (Float32(2.0) * dist)
        var h_sq = max(Float32(0.0), r1 * r1 - a_proj * a_proj)
        var h = sqrt(h_sq)
        joint_c = joint_b + u * a_proj + v * (branch * h)

    var bc = joint_c - joint_b
    var dc = joint_c - d_plane
    var coupler_angle = det_atan2(dot(bc, ey), dot(bc, ex))
    var rocker_angle = det_atan2(dot(dc, ey), dot(dc, ex))
    if not (_vec3_finite(joint_b) and _vec3_finite(joint_c) and isfinite(coupler_angle) and isfinite(rocker_angle)):
        return (ground_a, ground_a, Float32(0.0), Float32(0.0))
    return (joint_b, joint_c, coupler_angle, rocker_angle)


def three_bone_digitigrade_ik(
    hip: Vec3,
    knee1: Vec3,
    ankle_rev: Vec3,
    foot: Vec3,
    target: Vec3,
    forward_dir: Vec3,
    crouch_bias: Float32 = 0.5,
) -> Tuple[Quat, Quat, Quat]:
    """Analytic 3-segment digitigrade / reverse-joint walker IK solver (`hip -> knee1 -> ankle_rev -> foot`).

    Distributes flexion across femur (`L1`), reverse-tibia (`L2`), and tarsus strut (`L3`)
    so `knee1` projects forward (`+forward_dir`), `ankle_rev` projects backward (`-forward_dir`),
    and `foot` reaches `target` exactly.

    Returns hierarchical local rotations `(q_hip, q_knee1, q_ankle_rev)` relative to the
    supplied rest joint positions (`world_hip = q_hip`, `world_knee = q_hip * q_knee1`,
    `world_ankle = q_hip * q_knee1 * q_ankle_rev`).
    """
    if not (
        _vec3_finite(hip)
        and _vec3_finite(knee1)
        and _vec3_finite(ankle_rev)
        and _vec3_finite(foot)
        and _vec3_finite(target)
        and _vec3_finite(forward_dir)
    ):
        return (Quat(), Quat(), Quat())
    var eps = Float32(0.001)
    var v1_rest = knee1 - hip
    var v2_rest = ankle_rev - knee1
    var v3_rest = foot - ankle_rev
    var l1 = v1_rest.length()
    var l2 = v2_rest.length()
    var l3 = v3_rest.length()
    if l1 < eps or l2 < eps or l3 < eps:
        return (Quat(), Quat(), Quat())

    var to_target = target - hip
    var d_raw = to_target.length()
    var max_reach = l1 + l2 + l3 - eps
    # Minimum 3-segment triangle inequality bound
    var longest = max(l1, max(l2, l3))
    var min_reach = max(eps * Float32(2.0), Float32(2.0) * longest - (l1 + l2 + l3) + eps * Float32(2.0))
    var d_soft = max(eps, (l1 + l2 + l3) * Float32(0.03))

    var d_eff = d_raw
    if max_reach <= min_reach:
        d_eff = min_reach
    elif d_raw > max_reach - d_soft:
        d_eff = max_reach - d_soft * det_exp(-(d_raw - (max_reach - d_soft)) / d_soft)
    elif d_raw < min_reach + d_soft:
        d_eff = min_reach + d_soft * det_exp(-((min_reach + d_soft) - d_raw) / d_soft)

    var u_d = safe_normalized(to_target, Vec3(0, -1, 0))
    var fwd_in = safe_normalized(forward_dir, Vec3(0, 0, 1))
    var fwd_perp = fwd_in - u_d * dot(fwd_in, u_d)
    var e_fwd = safe_normalized(fwd_perp, ortho_fallback(u_d))
    var n_side = safe_normalized(cross(u_d, e_fwd), Vec3(1, 0, 0))

    # Feasible virtual span S = |ankle_rev' - hip| satisfying both triangles:
    # (l1, l2, S) and (S, l3, d_eff)
    var s_low = max(abs(l1 - l2) + eps, abs(d_eff - l3) + eps)
    var s_high = min(l1 + l2 - eps, d_eff + l3 - eps)
    var bias = max(Float32(0.05), min(Float32(0.95), crouch_bias))
    var span_s: Float32
    if s_high <= s_low:
        span_s = Float32(0.5) * (s_low + s_high)
    else:
        # Higher crouch_bias compresses upper femur-tibia span S toward s_low
        span_s = s_high - bias * (s_high - s_low)

    # 1. Solve virtual triangle (hip, ankle_rev', target_eff) -> ankle_rev' bends backward (-e_fwd)
    var cos_alpha = clamp1((span_s * span_s + d_eff * d_eff - l3 * l3) / max(Float32(1.0e-6), Float32(2.0) * span_s * d_eff))
    var alpha = det_acos(cos_alpha)
    var ankle_solved = hip + (u_d * det_cos(alpha) - e_fwd * det_sin(alpha)) * span_s
    var target_eff = hip + u_d * d_eff

    # 2. Solve upper 2-bone triangle (hip, knee1', ankle_solved) -> knee1' bends forward (+e_ha_fwd)
    var u_ha = safe_normalized(ankle_solved - hip, u_d)
    var e_ha_fwd = safe_normalized(cross(n_side, u_ha), e_fwd)
    if dot(e_ha_fwd, e_fwd) < Float32(0.0):
        e_ha_fwd = -e_ha_fwd
    var cos_beta = clamp1((l1 * l1 + span_s * span_s - l2 * l2) / max(Float32(1.0e-6), Float32(2.0) * l1 * span_s))
    var beta = det_acos(cos_beta)
    var knee_solved = hip + (u_ha * det_cos(beta) + e_ha_fwd * det_sin(beta)) * l1

    # 3. Reconstruct world rotations W0, W1, W2 with sagittal-plane twist alignment, then local hierarchy
    var v1_goal = knee_solved - hip
    var v2_goal = ankle_solved - knee_solved
    var v3_goal = target_eff - ankle_solved

    # Align initial leg sagittal plane normal onto n_side first so minimal-arc rotations stay twist-free
    var rest_plane_n = cross(v1_rest, v2_rest)
    if rest_plane_n.length() < Float32(1.0e-4):
        rest_plane_n = cross(v1_rest, fwd_in)
    var w_plane = Quat()
    if rest_plane_n.length() >= Float32(1.0e-4):
        var rn = rest_plane_n.normalized()
        if dot(rn, n_side) < Float32(0.0):
            rn = -rn
        w_plane = shortest_arc_quat(rn, n_side)

    var w0 = shortest_arc_quat(w_plane.rotate(v1_rest), v1_goal).multiply(w_plane).normalized()
    var w1 = shortest_arc_quat(w0.rotate(v2_rest), v2_goal).multiply(w0).normalized()
    var w2 = shortest_arc_quat(w1.rotate(v3_rest), v3_goal).multiply(w1).normalized()

    var q_hip = w0
    var q_knee1 = w0.conjugate().multiply(w1).normalized()
    var q_ankle = w1.conjugate().multiply(w2).normalized()
    return (q_hip, q_knee1, q_ankle)
