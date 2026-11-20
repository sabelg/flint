"""Server-Side Lag Compensation via Temporal Hitbox Pose Ring (`flint.net.lag_comp`).

Implements:
- `HitboxPose`: sphere, AABB, OBB, and Y-axis-aligned capsule hitboxes
- `LagCompHit`: unified raycast / overlap hit record with clamped rewind timestamp
- `HitboxSnapshotRing`: fixed-capacity monotonic timestamp ring with bounded `max_rewind_ms`,
  linear/slerp bracketing interpolation across `(entity_id, hitbox_id)` pairs, and exact
  `rewind_and_raycast`, `rewind_and_overlap` (sphere), and `rewind_and_overlap_aabb`
- `LagCompHistory`: tick-indexed convenience wrapper over `HitboxSnapshotRing`
  (`record_tick`, `raycast_at_tick`, `overlap_at_tick`, `overlap_aabb_at_tick`).
"""
from std.collections import List
from std.math import abs, max, min, sqrt
from flint.core.math import Quat, Vec3

comptime HITBOX_SHAPE_SPHERE: Int = 0
comptime HITBOX_SHAPE_AABB: Int = 1
comptime HITBOX_SHAPE_OBB: Int = 2
comptime HITBOX_SHAPE_CAPSULE: Int = 3


@always_inline
def _is_finite32(v: Float32) -> Bool:
    return (v - v) == Float32(0.0)


@always_inline
def _vec3_finite(v: Vec3) -> Bool:
    return _is_finite32(v.x) and _is_finite32(v.y) and _is_finite32(v.z)


@always_inline
def _quat_finite(q: Quat) -> Bool:
    return (
        _is_finite32(q.x)
        and _is_finite32(q.y)
        and _is_finite32(q.z)
        and _is_finite32(q.w)
    )


struct HitboxPose(Copyable, ImplicitlyCopyable, Movable):
    """Single entity sub-hitbox pose recorded in a server lag-compensation snapshot."""
    var entity_id: Int
    var hitbox_id: Int
    var shape: Int
    var center: Vec3
    var rot: Quat
    var half_extents: Vec3
    var radius: Float32
    var half_height: Float32
    var active: Bool

    def __init__(
        out self,
        entity_id: Int = -1,
        hitbox_id: Int = 0,
        shape: Int = HITBOX_SHAPE_SPHERE,
        center: Vec3 = Vec3(),
        rot: Quat = Quat.identity(),
        half_extents: Vec3 = Vec3(0.5, 0.5, 0.5),
        radius: Float32 = 0.5,
        half_height: Float32 = 0.5,
        active: Bool = True,
    ):
        var geom_ok = (
            _vec3_finite(center)
            and _quat_finite(rot)
            and _vec3_finite(half_extents)
            and _is_finite32(radius)
            and _is_finite32(half_height)
        )
        var qm = max(
            max(abs(rot.x), abs(rot.y)),
            max(abs(rot.z), abs(rot.w)),
        )
        var norm_rot = Quat.identity()
        var rot_ok = _quat_finite(rot)
        if rot_ok and qm > Float32(0.0):
            var sx = rot.x / qm
            var sy = rot.y / qm
            var sz = rot.z / qm
            var sw = rot.w / qm
            var slen = sqrt(sx * sx + sy * sy + sz * sz + sw * sw)
            if _is_finite32(slen) and slen >= Float32(1.0e-6):
                var inv = Float32(1.0) / slen
                norm_rot = Quat(sx * inv, sy * inv, sz * inv, sw * inv)
            else:
                rot_ok = False
        self.entity_id = entity_id
        self.hitbox_id = hitbox_id
        self.shape = shape
        self.center = center
        self.rot = norm_rot
        self.half_extents = Vec3(
            max(Float32(1.0e-4), abs(half_extents.x)),
            max(Float32(1.0e-4), abs(half_extents.y)),
            max(Float32(1.0e-4), abs(half_extents.z)),
        )
        self.radius = max(Float32(1.0e-4), abs(radius))
        self.half_height = max(Float32(0.0), abs(half_height))
        self.active = active and geom_ok and rot_ok and _quat_finite(self.rot)

    @staticmethod
    def sphere(
        entity_id: Int,
        center: Vec3,
        radius: Float32,
        hitbox_id: Int = 0,
    ) -> Self:
        return Self(
            entity_id=entity_id,
            hitbox_id=hitbox_id,
            shape=HITBOX_SHAPE_SPHERE,
            center=center,
            radius=radius,
        )

    @staticmethod
    def aabb(
        entity_id: Int,
        center: Vec3,
        half_extents: Vec3,
        hitbox_id: Int = 0,
    ) -> Self:
        return Self(
            entity_id=entity_id,
            hitbox_id=hitbox_id,
            shape=HITBOX_SHAPE_AABB,
            center=center,
            half_extents=half_extents,
        )

    @staticmethod
    def obb(
        entity_id: Int,
        center: Vec3,
        rot: Quat,
        half_extents: Vec3,
        hitbox_id: Int = 0,
    ) -> Self:
        return Self(
            entity_id=entity_id,
            hitbox_id=hitbox_id,
            shape=HITBOX_SHAPE_OBB,
            center=center,
            rot=rot,
            half_extents=half_extents,
        )

    @staticmethod
    def capsule(
        entity_id: Int,
        center: Vec3,
        rot: Quat,
        radius: Float32,
        half_height: Float32,
        hitbox_id: Int = 0,
    ) -> Self:
        return Self(
            entity_id=entity_id,
            hitbox_id=hitbox_id,
            shape=HITBOX_SHAPE_CAPSULE,
            center=center,
            rot=rot,
            radius=radius,
            half_height=half_height,
        )


@fieldwise_init
struct LagCompHit(Copyable, ImplicitlyCopyable, Movable):
    """Result of a lag-compensated raycast or overlap query."""
    var hit: Bool
    var entity_id: Int
    var hitbox_id: Int
    var distance: Float32
    var fraction: Float32
    var point: Vec3
    var normal: Vec3
    var rewind_time_ms: Float64
    var was_clamped: Bool

    def __init__(out self):
        self.hit = False
        self.entity_id = -1
        self.hitbox_id = -1
        self.distance = Float32(1.0e30)
        self.fraction = Float32(1.0)
        self.point = Vec3()
        self.normal = Vec3(0.0, 1.0, 0.0)
        self.rewind_time_ms = 0.0
        self.was_clamped = False


struct _HitboxSnapshotSlot(Movable):
    var valid: Bool
    var tick: Int
    var time_ms: Float64
    var poses: List[HitboxPose]

    def __init__(out self):
        self.valid = False
        self.tick = -1
        self.time_ms = 0.0
        self.poses = List[HitboxPose]()


@always_inline
def _pose_is_finite(pose: HitboxPose) -> Bool:
    if not pose.active:
        return False
    if not _vec3_finite(pose.center) or not _quat_finite(pose.rot):
        return False
    if pose.shape == HITBOX_SHAPE_SPHERE:
        return _is_finite32(pose.radius) and pose.radius > Float32(0.0)
    if pose.shape == HITBOX_SHAPE_CAPSULE:
        return (
            _is_finite32(pose.radius)
            and pose.radius > Float32(0.0)
            and _is_finite32(pose.half_height)
            and pose.half_height >= Float32(0.0)
        )
    return (
        _vec3_finite(pose.half_extents)
        and pose.half_extents.x > Float32(0.0)
        and pose.half_extents.y > Float32(0.0)
        and pose.half_extents.z > Float32(0.0)
    )


@always_inline
def _clamp_f32(v: Float32, lo: Float32, hi: Float32) -> Float32:
    return max(lo, min(hi, v))


@always_inline
def _clamp_vec3(v: Vec3, lo: Vec3, hi: Vec3) -> Vec3:
    return Vec3(
        _clamp_f32(v.x, lo.x, hi.x),
        _clamp_f32(v.y, lo.y, hi.y),
        _clamp_f32(v.z, lo.z, hi.z),
    )


@always_inline
def _interpolate_pose(p0: HitboxPose, p1: HitboxPose, alpha: Float32) -> HitboxPose:
    var inv = Float32(1.0) - alpha
    return HitboxPose(
        entity_id=p0.entity_id,
        hitbox_id=p0.hitbox_id,
        shape=p0.shape,
        center=p0.center * inv + p1.center * alpha,
        rot=p0.rot.slerp(p1.rot, alpha),
        half_extents=p0.half_extents * inv + p1.half_extents * alpha,
        radius=p0.radius * inv + p1.radius * alpha,
        half_height=p0.half_height * inv + p1.half_height * alpha,
        active=True,
    )


@always_inline
def _hit_precedes(a: LagCompHit, b: LagCompHit) -> Bool:
    if a.entity_id != b.entity_id:
        return a.entity_id < b.entity_id
    return a.hitbox_id < b.hitbox_id


def _sort_hits_deterministic(mut hits: List[LagCompHit]):
    var n = len(hits)
    for i in range(1, n):
        var key = hits[i]
        var j = i - 1
        while j >= 0 and _hit_precedes(key, hits[j]):
            hits[j + 1] = hits[j]
            j -= 1
        hits[j + 1] = key


def _raycast_sphere(
    origin: Vec3,
    ndir: Vec3,
    max_dist: Float32,
    center: Vec3,
    radius: Float32,
) -> Tuple[Bool, Float32, Vec3, Vec3]:
    var oc = origin - center
    var c = oc.length_squared() - radius * radius
    if c <= Float32(0.0):
        var d2 = oc.length_squared()
        var n = oc.normalized() if d2 > Float32(1.0e-10) else -ndir
        return (True, Float32(0.0), origin, n)
    var b = oc.dot(ndir)
    if b > Float32(0.0):
        return (False, Float32(0.0), Vec3(), Vec3())
    var disc = b * b - c
    if disc < Float32(0.0):
        return (False, Float32(0.0), Vec3(), Vec3())
    var t = -b - sqrt(disc)
    if t < Float32(0.0):
        t = Float32(0.0)
    if t > max_dist:
        return (False, Float32(0.0), Vec3(), Vec3())
    var pt = origin + ndir * t
    var n = (pt - center).normalized()
    if n.length_squared() <= Float32(1.0e-10):
        n = -ndir
    return (True, t, pt, n)


def _inside_box_normal(p: Vec3, h: Vec3) -> Vec3:
    var dx = h.x - abs(p.x)
    var dy = h.y - abs(p.y)
    var dz = h.z - abs(p.z)
    if dx <= dy and dx <= dz:
        return Vec3(Float32(1.0) if p.x >= Float32(0.0) else Float32(-1.0), 0.0, 0.0)
    if dy <= dz:
        return Vec3(0.0, Float32(1.0) if p.y >= Float32(0.0) else Float32(-1.0), 0.0)
    return Vec3(0.0, 0.0, Float32(1.0) if p.z >= Float32(0.0) else Float32(-1.0))


def _raycast_local_aabb(
    orig: Vec3,
    dir: Vec3,
    h: Vec3,
    max_dist: Float32,
) -> Tuple[Bool, Float32, Vec3, Vec3]:
    if abs(orig.x) <= h.x and abs(orig.y) <= h.y and abs(orig.z) <= h.z:
        return (True, Float32(0.0), orig, _inside_box_normal(orig, h))

    var tmin = Float32(0.0)
    var tmax = max_dist
    var hit_axis = -1
    var hit_sign = Float32(1.0)

    # X slab
    if abs(dir.x) < Float32(1.0e-8):
        if orig.x < -h.x or orig.x > h.x:
            return (False, Float32(0.0), Vec3(), Vec3())
    else:
        var inv = Float32(1.0) / dir.x
        var t1 = (-h.x - orig.x) * inv
        var t2 = (h.x - orig.x) * inv
        var s = Float32(-1.0)
        if t1 > t2:
            var tmp = t1
            t1 = t2
            t2 = tmp
            s = Float32(1.0)
        if t1 > tmin:
            tmin = t1
            hit_axis = 0
            hit_sign = s
        if t2 < tmax:
            tmax = t2
        if tmin > tmax:
            return (False, Float32(0.0), Vec3(), Vec3())

    # Y slab
    if abs(dir.y) < Float32(1.0e-8):
        if orig.y < -h.y or orig.y > h.y:
            return (False, Float32(0.0), Vec3(), Vec3())
    else:
        var inv = Float32(1.0) / dir.y
        var t1 = (-h.y - orig.y) * inv
        var t2 = (h.y - orig.y) * inv
        var s = Float32(-1.0)
        if t1 > t2:
            var tmp = t1
            t1 = t2
            t2 = tmp
            s = Float32(1.0)
        if t1 > tmin:
            tmin = t1
            hit_axis = 1
            hit_sign = s
        if t2 < tmax:
            tmax = t2
        if tmin > tmax:
            return (False, Float32(0.0), Vec3(), Vec3())

    # Z slab
    if abs(dir.z) < Float32(1.0e-8):
        if orig.z < -h.z or orig.z > h.z:
            return (False, Float32(0.0), Vec3(), Vec3())
    else:
        var inv = Float32(1.0) / dir.z
        var t1 = (-h.z - orig.z) * inv
        var t2 = (h.z - orig.z) * inv
        var s = Float32(-1.0)
        if t1 > t2:
            var tmp = t1
            t1 = t2
            t2 = tmp
            s = Float32(1.0)
        if t1 > tmin:
            tmin = t1
            hit_axis = 2
            hit_sign = s
        if t2 < tmax:
            tmax = t2
        if tmin > tmax:
            return (False, Float32(0.0), Vec3(), Vec3())

    if tmin < Float32(0.0) or tmin > max_dist:
        return (False, Float32(0.0), Vec3(), Vec3())

    var pt = orig + dir * tmin
    var n: Vec3
    if hit_axis == 0:
        n = Vec3(hit_sign, 0.0, 0.0)
    elif hit_axis == 1:
        n = Vec3(0.0, hit_sign, 0.0)
    elif hit_axis == 2:
        n = Vec3(0.0, 0.0, hit_sign)
    else:
        n = _inside_box_normal(pt, h)
    return (True, tmin, pt, n)


def _raycast_capsule(
    origin: Vec3,
    ndir: Vec3,
    max_dist: Float32,
    pose: HitboxPose,
) -> Tuple[Bool, Float32, Vec3, Vec3]:
    if pose.half_height <= Float32(1.0e-6):
        return _raycast_sphere(origin, ndir, max_dist, pose.center, pose.radius)

    var axis = pose.rot.rotate(Vec3(0.0, pose.half_height, 0.0))
    var a = pose.center - axis
    var b = pose.center + axis
    var ab = b - a
    var ab_len2 = ab.length_squared()
    if ab_len2 <= Float32(1.0e-10):
        return _raycast_sphere(origin, ndir, max_dist, pose.center, pose.radius)

    # Check if origin starts inside or on the capsule
    var u0 = _clamp_f32((origin - a).dot(ab) / ab_len2, Float32(0.0), Float32(1.0))
    var q0 = a + ab * u0
    var diff0 = origin - q0
    var diff0_sq = diff0.length_squared()
    if diff0_sq <= pose.radius * pose.radius:
        var n0 = diff0.normalized() if diff0_sq > Float32(1.0e-10) else -ndir
        return (True, Float32(0.0), origin, n0)

    var best_hit = False
    var best_t = max_dist + Float32(1.0)
    var best_pt = Vec3()
    var best_n = Vec3(0.0, 1.0, 0.0)

    # 1. Finite cylinder body
    var inv_len = Float32(1.0) / sqrt(ab_len2)
    var v = ab * inv_len
    var ao = origin - a
    var d_perp = ndir - v * ndir.dot(v)
    var ao_perp = ao - v * ao.dot(v)
    var qa = d_perp.length_squared()
    if qa > Float32(1.0e-10):
        var qb = ao_perp.dot(d_perp)
        var qc = ao_perp.length_squared() - pose.radius * pose.radius
        var disc = qb * qb - qa * qc
        if disc >= Float32(0.0):
            var t_cyl = (-qb - sqrt(disc)) / qa
            if t_cyl >= Float32(0.0) and t_cyl <= max_dist:
                var hp = origin + ndir * t_cyl
                var proj = (hp - a).dot(ab)
                if proj >= Float32(0.0) and proj <= ab_len2:
                    best_hit = True
                    best_t = t_cyl
                    best_pt = hp
                    var q_seg = a + ab * (proj / ab_len2)
                    best_n = (hp - q_seg).normalized()

    # 2. Endcap spheres at a and b
    var cap_a = _raycast_sphere(origin, ndir, max_dist, a, pose.radius)
    if cap_a[0] and (not best_hit or cap_a[1] < best_t):
        best_hit = True
        best_t = cap_a[1]
        best_pt = cap_a[2]
        best_n = cap_a[3]

    var cap_b = _raycast_sphere(origin, ndir, max_dist, b, pose.radius)
    if cap_b[0] and (not best_hit or cap_b[1] < best_t):
        best_hit = True
        best_t = cap_b[1]
        best_pt = cap_b[2]
        best_n = cap_b[3]

    return (best_hit, best_t, best_pt, best_n)


def _raycast_pose(
    origin: Vec3,
    ndir: Vec3,
    max_dist: Float32,
    pose: HitboxPose,
) -> Tuple[Bool, Float32, Vec3, Vec3]:
    if pose.shape == HITBOX_SHAPE_SPHERE:
        return _raycast_sphere(origin, ndir, max_dist, pose.center, pose.radius)
    if pose.shape == HITBOX_SHAPE_AABB:
        var res = _raycast_local_aabb(
            origin - pose.center, ndir, pose.half_extents, max_dist
        )
        if not res[0]:
            return res
        return (True, res[1], res[2] + pose.center, res[3])
    if pose.shape == HITBOX_SHAPE_OBB:
        var inv_r = pose.rot.conjugate()
        var l_orig = inv_r.rotate(origin - pose.center)
        var l_dir = inv_r.rotate(ndir)
        var res = _raycast_local_aabb(
            l_orig, l_dir, pose.half_extents, max_dist
        )
        if not res[0]:
            return res
        var world_pt = origin + ndir * res[1]
        var world_n = pose.rot.rotate(res[3]).normalized()
        return (True, res[1], world_pt, world_n)
    return _raycast_capsule(origin, ndir, max_dist, pose)


def _sphere_overlap_pose(
    center: Vec3,
    radius: Float32,
    pose: HitboxPose,
) -> Tuple[Bool, Float32, Vec3, Vec3]:
    if pose.shape == HITBOX_SHAPE_SPHERE:
        var delta = center - pose.center
        var d2 = delta.length_squared()
        var sum_r = radius + pose.radius
        if d2 > sum_r * sum_r:
            return (False, Float32(0.0), Vec3(), Vec3())
        var d = sqrt(d2)
        var n = (delta * (Float32(1.0) / d)) if d > Float32(1.0e-6) else Vec3(0.0, 1.0, 0.0)
        var dist_to_surf = max(Float32(0.0), d - pose.radius)
        var witness = pose.center + n * min(pose.radius, d)
        return (True, dist_to_surf, witness, n)

    if pose.shape == HITBOX_SHAPE_AABB or pose.shape == HITBOX_SHAPE_OBB:
        var is_obb = pose.shape == HITBOX_SHAPE_OBB
        var local_c = (
            pose.rot.conjugate().rotate(center - pose.center)
            if is_obb
            else (center - pose.center)
        )
        var h = pose.half_extents
        var clamped = _clamp_vec3(local_c, -h, h)
        var diff = local_c - clamped
        var d2 = diff.length_squared()
        if d2 > radius * radius:
            return (False, Float32(0.0), Vec3(), Vec3())
        var dist = sqrt(d2)
        var local_n = (
            diff * (Float32(1.0) / dist)
            if dist > Float32(1.0e-6)
            else _inside_box_normal(local_c, h)
        )
        var world_pt = (
            pose.center + pose.rot.rotate(clamped)
            if is_obb
            else (pose.center + clamped)
        )
        var world_n = (
            pose.rot.rotate(local_n).normalized() if is_obb else local_n
        )
        return (True, dist, world_pt, world_n)

    # Capsule
    var axis = pose.rot.rotate(Vec3(0.0, pose.half_height, 0.0))
    var a = pose.center - axis
    var b = pose.center + axis
    var ab = b - a
    var ab_len2 = ab.length_squared()
    var u = (
        _clamp_f32((center - a).dot(ab) / ab_len2, Float32(0.0), Float32(1.0))
        if ab_len2 > Float32(1.0e-10)
        else Float32(0.0)
    )
    var q = a + ab * u
    var delta = center - q
    var d2 = delta.length_squared()
    var sum_r = radius + pose.radius
    if d2 > sum_r * sum_r:
        return (False, Float32(0.0), Vec3(), Vec3())
    var d = sqrt(d2)
    var n = (delta * (Float32(1.0) / d)) if d > Float32(1.0e-6) else Vec3(0.0, 1.0, 0.0)
    var dist_to_surf = max(Float32(0.0), d - pose.radius)
    var witness = q + n * min(pose.radius, d)
    return (True, dist_to_surf, witness, n)


def _aabb_overlap_obb_sat(
    box_center: Vec3,
    box_half: Vec3,
    obb: HitboxPose,
) -> Bool:
    var u0 = obb.rot.rotate(Vec3(1.0, 0.0, 0.0))
    var u1 = obb.rot.rotate(Vec3(0.0, 1.0, 0.0))
    var u2 = obb.rot.rotate(Vec3(0.0, 0.0, 1.0))

    # Rotation matrix R_ij = e_i . u_j (since e_i are standard basis, R_0j = u_j.x, etc.)
    var r00 = u0.x
    var r01 = u1.x
    var r02 = u2.x
    var r10 = u0.y
    var r11 = u1.y
    var r12 = u2.y
    var r20 = u0.z
    var r21 = u1.z
    var r22 = u2.z

    comptime EPS: Float32 = Float32(1.0e-5)
    var ar00 = abs(r00) + EPS
    var ar01 = abs(r01) + EPS
    var ar02 = abs(r02) + EPS
    var ar10 = abs(r10) + EPS
    var ar11 = abs(r11) + EPS
    var ar12 = abs(r12) + EPS
    var ar20 = abs(r20) + EPS
    var ar21 = abs(r21) + EPS
    var ar22 = abs(r22) + EPS

    var t = obb.center - box_center
    var a0 = box_half.x
    var a1 = box_half.y
    var a2 = box_half.z
    var b0 = obb.half_extents.x
    var b1 = obb.half_extents.y
    var b2 = obb.half_extents.z

    # 3 AABB face axes (e0, e1, e2)
    if abs(t.x) > a0 + b0 * ar00 + b1 * ar01 + b2 * ar02:
        return False
    if abs(t.y) > a1 + b0 * ar10 + b1 * ar11 + b2 * ar12:
        return False
    if abs(t.z) > a2 + b0 * ar20 + b1 * ar21 + b2 * ar22:
        return False

    # 3 OBB face axes (u0, u1, u2)
    if abs(t.x * r00 + t.y * r10 + t.z * r20) > a0 * ar00 + a1 * ar10 + a2 * ar20 + b0:
        return False
    if abs(t.x * r01 + t.y * r11 + t.z * r21) > a0 * ar01 + a1 * ar11 + a2 * ar21 + b1:
        return False
    if abs(t.x * r02 + t.y * r12 + t.z * r22) > a0 * ar02 + a1 * ar12 + a2 * ar22 + b2:
        return False

    # 9 cross-product axes (e_i x u_j)
    if abs(t.z * r10 - t.y * r20) > a1 * ar20 + a2 * ar10 + b1 * ar02 + b2 * ar01:
        return False
    if abs(t.z * r11 - t.y * r21) > a1 * ar21 + a2 * ar11 + b0 * ar02 + b2 * ar00:
        return False
    if abs(t.z * r12 - t.y * r22) > a1 * ar22 + a2 * ar12 + b0 * ar01 + b1 * ar00:
        return False

    if abs(t.x * r20 - t.z * r00) > a0 * ar20 + a2 * ar00 + b1 * ar12 + b2 * ar11:
        return False
    if abs(t.x * r21 - t.z * r01) > a0 * ar21 + a2 * ar01 + b0 * ar12 + b2 * ar10:
        return False
    if abs(t.x * r22 - t.z * r02) > a0 * ar22 + a2 * ar02 + b0 * ar11 + b1 * ar10:
        return False

    if abs(t.y * r00 - t.x * r10) > a0 * ar10 + a1 * ar00 + b1 * ar22 + b2 * ar21:
        return False
    if abs(t.y * r01 - t.x * r11) > a0 * ar11 + a1 * ar01 + b0 * ar22 + b2 * ar20:
        return False
    if abs(t.y * r02 - t.x * r12) > a0 * ar12 + a1 * ar02 + b0 * ar21 + b1 * ar20:
        return False

    return True


def _seg_aabb_dist_sq_at(
    a: Vec3, d: Vec3, t: Float32, lo: Vec3, hi: Vec3
) -> Float32:
    var p = a + d * t
    var c = _clamp_vec3(p, lo, hi)
    return (p - c).length_squared()


def _aabb_overlap_capsule(
    lo: Vec3,
    hi: Vec3,
    pose: HitboxPose,
) -> Tuple[Bool, Float32, Vec3, Vec3]:
    var axis = pose.rot.rotate(Vec3(0.0, pose.half_height, 0.0))
    var a = pose.center - axis
    var b = pose.center + axis
    var d = b - a

    # Collect sorted breakpoints in [0, 1] where segment crosses AABB slab planes
    var ts = List[Float32](capacity=10)
    ts.append(Float32(0.0))
    ts.append(Float32(1.0))

    if abs(d.x) > Float32(1.0e-7):
        var t0 = (lo.x - a.x) / d.x
        var t1 = (hi.x - a.x) / d.x
        if t0 > Float32(0.0) and t0 < Float32(1.0):
            ts.append(t0)
        if t1 > Float32(0.0) and t1 < Float32(1.0):
            ts.append(t1)
    if abs(d.y) > Float32(1.0e-7):
        var t0 = (lo.y - a.y) / d.y
        var t1 = (hi.y - a.y) / d.y
        if t0 > Float32(0.0) and t0 < Float32(1.0):
            ts.append(t0)
        if t1 > Float32(0.0) and t1 < Float32(1.0):
            ts.append(t1)
    if abs(d.z) > Float32(1.0e-7):
        var t0 = (lo.z - a.z) / d.z
        var t1 = (hi.z - a.z) / d.z
        if t0 > Float32(0.0) and t0 < Float32(1.0):
            ts.append(t0)
        if t1 > Float32(0.0) and t1 < Float32(1.0):
            ts.append(t1)

    # Insertion sort breakpoints
    for i in range(1, len(ts)):
        var key = ts[i]
        var j = i - 1
        while j >= 0 and ts[j] > key:
            ts[j + 1] = ts[j]
            j -= 1
        ts[j + 1] = key

    var best_t = Float32(0.0)
    var best_d2 = _seg_aabb_dist_sq_at(a, d, best_t, lo, hi)
    for i in range(len(ts)):
        var cand_t = ts[i]
        var d2 = _seg_aabb_dist_sq_at(a, d, cand_t, lo, hi)
        if d2 < best_d2:
            best_d2 = d2
            best_t = cand_t

    # Exact sub-interval analytical critical points of convex piecewise-quadratic distance
    for i in range(len(ts) - 1):
        var t_l = ts[i]
        var t_r = ts[i + 1]
        if t_r - t_l <= Float32(1.0e-6):
            continue
        var p_mid = a + d * ((t_l + t_r) * Float32(0.5))
        var qa = Float32(0.0)
        var qb = Float32(0.0)
        if p_mid.x < lo.x:
            qa += d.x * d.x
            qb += (a.x - lo.x) * d.x
        elif p_mid.x > hi.x:
            qa += d.x * d.x
            qb += (a.x - hi.x) * d.x
        if p_mid.y < lo.y:
            qa += d.y * d.y
            qb += (a.y - lo.y) * d.y
        elif p_mid.y > hi.y:
            qa += d.y * d.y
            qb += (a.y - hi.y) * d.y
        if p_mid.z < lo.z:
            qa += d.z * d.z
            qb += (a.z - lo.z) * d.z
        elif p_mid.z > hi.z:
            qa += d.z * d.z
            qb += (a.z - hi.z) * d.z
        if qa > Float32(1.0e-10):
            var t_crit = -qb / qa
            if t_crit > t_l and t_crit < t_r:
                var d2 = _seg_aabb_dist_sq_at(a, d, t_crit, lo, hi)
                if d2 < best_d2:
                    best_d2 = d2
                    best_t = t_crit

    if best_d2 > pose.radius * pose.radius:
        return (False, Float32(0.0), Vec3(), Vec3())

    var seg_pt = a + d * best_t
    var box_pt = _clamp_vec3(seg_pt, lo, hi)
    var delta = box_pt - seg_pt
    var dist = sqrt(best_d2)
    var n = (
        delta * (Float32(1.0) / dist)
        if dist > Float32(1.0e-6)
        else Vec3(0.0, 1.0, 0.0)
    )
    return (True, max(Float32(0.0), dist - pose.radius), box_pt, n)


def _aabb_overlap_pose(
    lo: Vec3,
    hi: Vec3,
    box_center: Vec3,
    box_half: Vec3,
    pose: HitboxPose,
) -> Tuple[Bool, Float32, Vec3, Vec3]:
    if pose.shape == HITBOX_SHAPE_SPHERE:
        var clamped = _clamp_vec3(pose.center, lo, hi)
        var delta = clamped - pose.center
        var d2 = delta.length_squared()
        if d2 > pose.radius * pose.radius:
            return (False, Float32(0.0), Vec3(), Vec3())
        var d = sqrt(d2)
        var n = (
            delta * (Float32(1.0) / d)
            if d > Float32(1.0e-6)
            else Vec3(0.0, 1.0, 0.0)
        )
        return (True, d, clamped, n)

    if pose.shape == HITBOX_SHAPE_AABB:
        var p_lo = pose.center - pose.half_extents
        var p_hi = pose.center + pose.half_extents
        if (
            hi.x < p_lo.x
            or lo.x > p_hi.x
            or hi.y < p_lo.y
            or lo.y > p_hi.y
            or hi.z < p_lo.z
            or lo.z > p_hi.z
        ):
            return (False, Float32(0.0), Vec3(), Vec3())
        var ov_lo = Vec3(max(lo.x, p_lo.x), max(lo.y, p_lo.y), max(lo.z, p_lo.z))
        var ov_hi = Vec3(min(hi.x, p_hi.x), min(hi.y, p_hi.y), min(hi.z, p_hi.z))
        var witness = (ov_lo + ov_hi) * Float32(0.5)
        var delta = box_center - pose.center
        var n = (
            delta.normalized()
            if delta.length_squared() > Float32(1.0e-10)
            else Vec3(0.0, 1.0, 0.0)
        )
        return (True, Float32(0.0), witness, n)

    if pose.shape == HITBOX_SHAPE_OBB:
        if not _aabb_overlap_obb_sat(box_center, box_half, pose):
            return (False, Float32(0.0), Vec3(), Vec3())
        var inv_r = pose.rot.conjugate()
        var l_c = inv_r.rotate(box_center - pose.center)
        var clamped = _clamp_vec3(l_c, -pose.half_extents, pose.half_extents)
        var witness = pose.center + pose.rot.rotate(clamped)
        var delta = box_center - pose.center
        var n = (
            delta.normalized()
            if delta.length_squared() > Float32(1.0e-10)
            else Vec3(0.0, 1.0, 0.0)
        )
        return (True, Float32(0.0), witness, n)

    return _aabb_overlap_capsule(lo, hi, pose)


struct HitboxSnapshotRing(Movable):
    """Fixed-capacity server hitbox history ring with bounded temporal rewind and interpolation."""
    var capacity: Int
    var max_rewind_ms: Float64
    var slots: List[_HitboxSnapshotSlot]
    var count: Int
    var write_idx: Int
    var latest_tick: Int
    var latest_time_ms: Float64

    def __init__(
        out self,
        capacity: Int = 64,
        max_rewind_ms: Float64 = 250.0,
    ):
        self.capacity = max(1, capacity)
        self.max_rewind_ms = max(0.0, max_rewind_ms)
        self.slots = List[_HitboxSnapshotSlot](capacity=self.capacity)
        for _ in range(self.capacity):
            self.slots.append(_HitboxSnapshotSlot())
        self.count = 0
        self.write_idx = 0
        self.latest_tick = -1
        self.latest_time_ms = 0.0

    def clear(mut self):
        for i in range(self.capacity):
            self.slots[i].valid = False
            self.slots[i].tick = -1
            self.slots[i].time_ms = 0.0
            self.slots[i].poses.clear()
        self.count = 0
        self.write_idx = 0
        self.latest_tick = -1
        self.latest_time_ms = 0.0

    @always_inline
    def snapshot_count(self) -> Int:
        return self.count

    def oldest_time_ms(self) -> Float64:
        if self.count == 0:
            return 0.0
        var oldest_idx = (self.write_idx - self.count + self.capacity) % self.capacity
        return self.slots[oldest_idx].time_ms

    def newest_time_ms(self) -> Float64:
        if self.count == 0:
            return 0.0
        var newest_idx = (self.write_idx - 1 + self.capacity) % self.capacity
        return self.slots[newest_idx].time_ms

    def record_snapshot(
        mut self, tick: Int, time_ms: Float64, poses: List[HitboxPose]
    ):
        """Records a monotonic server hitbox snapshot at `(tick, time_ms)`."""
        if (time_ms - time_ms) != 0.0:
            return
        if self.count > 0 and time_ms < self.latest_time_ms:
            return

        if self.count > 0 and time_ms == self.latest_time_ms:
            var newest_idx = (self.write_idx - 1 + self.capacity) % self.capacity
            self.slots[newest_idx].tick = tick
            self.slots[newest_idx].poses.clear()
            for i in range(len(poses)):
                self.slots[newest_idx].poses.append(poses[i])
            self.latest_tick = tick
            return

        var slot = self.write_idx
        self.slots[slot].valid = True
        self.slots[slot].tick = tick
        self.slots[slot].time_ms = time_ms
        self.slots[slot].poses.clear()
        for i in range(len(poses)):
            self.slots[slot].poses.append(poses[i])

        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1
        self.latest_tick = tick
        self.latest_time_ms = time_ms

    def _resolve_window(
        self, target_time_ms: Float64
    ) -> Tuple[Bool, Int, Int, Float32, Float64, Bool]:
        if self.count == 0:
            return (False, 0, 0, Float32(0.0), 0.0, False)

        var oldest_idx = (self.write_idx - self.count + self.capacity) % self.capacity
        var newest_idx = (self.write_idx - 1 + self.capacity) % self.capacity
        var oldest_t = self.slots[oldest_idx].time_ms
        var newest_t = self.slots[newest_idx].time_ms

        var clamped = False
        var t = target_time_ms
        if t != t:
            t = newest_t
            clamped = True

        var min_allowed = newest_t - self.max_rewind_ms
        if t < min_allowed:
            t = min_allowed
            clamped = True
        if t > newest_t:
            t = newest_t
            clamped = True
        if t < oldest_t:
            t = oldest_t
            clamped = True

        if self.count == 1 or t <= oldest_t:
            return (True, oldest_idx, oldest_idx, Float32(0.0), oldest_t, clamped)
        if t >= newest_t:
            return (True, newest_idx, newest_idx, Float32(0.0), newest_t, clamped)

        var idx0 = oldest_idx
        var idx1 = newest_idx
        for k in range(self.count - 1):
            var a = (oldest_idx + k) % self.capacity
            var b = (oldest_idx + k + 1) % self.capacity
            if self.slots[a].time_ms <= t and t <= self.slots[b].time_ms:
                idx0 = a
                idx1 = b
                break

        var dt = self.slots[idx1].time_ms - self.slots[idx0].time_ms
        var alpha = Float32(0.0)
        if dt > 1.0e-9:
            alpha = _clamp_f32(
                Float32((t - self.slots[idx0].time_ms) / dt),
                Float32(0.0),
                Float32(1.0),
            )
        return (True, idx0, idx1, alpha, t, clamped)

    def _find_matching_pose(
        self, idx: Int, hint_i: Int, entity_id: Int, hitbox_id: Int
    ) -> Int:
        var n = len(self.slots[idx].poses)
        if hint_i >= 0 and hint_i < n:
            var ph = self.slots[idx].poses[hint_i]
            if ph.entity_id == entity_id and ph.hitbox_id == hitbox_id:
                return hint_i
        for j in range(n):
            var pj = self.slots[idx].poses[j]
            if pj.entity_id == entity_id and pj.hitbox_id == hitbox_id:
                return j
        return -1

    def _gather_poses_at(
        self, idx0: Int, idx1: Int, alpha: Float32
    ) -> List[HitboxPose]:
        var out = List[HitboxPose]()
        if idx0 == idx1 or alpha <= Float32(1.0e-6):
            for i in range(len(self.slots[idx0].poses)):
                var p = self.slots[idx0].poses[i]
                if _pose_is_finite(p):
                    out.append(p)
            return out^

        if alpha >= Float32(1.0 - 1.0e-6):
            for i in range(len(self.slots[idx1].poses)):
                var p = self.slots[idx1].poses[i]
                if _pose_is_finite(p):
                    out.append(p)
            return out^

        var n0 = len(self.slots[idx0].poses)
        var n1 = len(self.slots[idx1].poses)
        for i in range(n0):
            var p0 = self.slots[idx0].poses[i]
            if not _pose_is_finite(p0):
                continue
            var m1 = self._find_matching_pose(idx1, i, p0.entity_id, p0.hitbox_id)
            if m1 >= 0:
                var p1 = self.slots[idx1].poses[m1]
                if _pose_is_finite(p1) and p1.shape == p0.shape:
                    var ip = _interpolate_pose(p0, p1, alpha)
                    if _pose_is_finite(ip):
                        out.append(ip)
                elif alpha < Float32(0.5):
                    out.append(p0)
            elif alpha < Float32(0.5):
                out.append(p0)

        if alpha >= Float32(0.5):
            for j in range(n1):
                var p1 = self.slots[idx1].poses[j]
                if not _pose_is_finite(p1):
                    continue
                var m0 = self._find_matching_pose(idx0, j, p1.entity_id, p1.hitbox_id)
                if m0 < 0 or not _pose_is_finite(self.slots[idx0].poses[m0]):
                    out.append(p1)
                elif self.slots[idx0].poses[m0].shape != p1.shape:
                    out.append(p1)
        return out^

    def rewind_and_raycast(
        self,
        target_time_ms: Float64,
        origin: Vec3,
        dir: Vec3,
        max_dist: Float32 = 1000.0,
        ignore_entity_id: Int = -1,
    ) -> LagCompHit:
        """Rewinds hitboxes to clamped `target_time_ms` and returns closest raycast hit."""
        var win = self._resolve_window(target_time_ms)
        if not win[0]:
            return LagCompHit()

        var best = LagCompHit()
        best.rewind_time_ms = win[4]
        best.was_clamped = win[5]

        var d2 = dir.length_squared()
        if (
            not _vec3_finite(origin)
            or not _vec3_finite(dir)
            or not _is_finite32(d2)
            or not (d2 > Float32(1.0e-12))
            or not _is_finite32(max_dist)
            or not (max_dist > Float32(0.0))
        ):
            return best

        var ndir = dir * (Float32(1.0) / sqrt(d2))
        if not _vec3_finite(ndir):
            return best
        var poses = self._gather_poses_at(win[1], win[2], win[3])
        var inv_max = Float32(1.0) / max_dist

        for i in range(len(poses)):
            var pose = poses[i]
            if not _pose_is_finite(pose) or pose.entity_id == ignore_entity_id:
                continue
            var res = _raycast_pose(origin, ndir, max_dist, pose)
            if not res[0] or not _is_finite32(res[1]):
                continue
            var dist = res[1]
            var better = not best.hit or dist < best.distance
            if not better and dist == best.distance:
                better = (
                    pose.entity_id < best.entity_id
                    or (
                        pose.entity_id == best.entity_id
                        and pose.hitbox_id < best.hitbox_id
                    )
                )
            if better:
                best.hit = True
                best.entity_id = pose.entity_id
                best.hitbox_id = pose.hitbox_id
                best.distance = dist
                best.fraction = _clamp_f32(dist * inv_max, Float32(0.0), Float32(1.0))
                best.point = res[2]
                best.normal = res[3]

        return best

    def rewind_and_overlap(
        self,
        target_time_ms: Float64,
        center: Vec3,
        radius: Float32,
        ignore_entity_id: Int = -1,
    ) -> List[LagCompHit]:
        """Rewinds hitboxes to clamped `target_time_ms` and returns all sphere-overlapping poses."""
        var hits = List[LagCompHit]()
        if (
            not _vec3_finite(center)
            or not _is_finite32(radius)
            or not (radius >= Float32(0.0))
        ):
            return hits^
        var win = self._resolve_window(target_time_ms)
        if not win[0]:
            return hits^

        var poses = self._gather_poses_at(win[1], win[2], win[3])
        for i in range(len(poses)):
            var pose = poses[i]
            if not _pose_is_finite(pose) or pose.entity_id == ignore_entity_id:
                continue
            var res = _sphere_overlap_pose(center, radius, pose)
            if not res[0] or not _is_finite32(res[1]):
                continue
            var h = LagCompHit()
            h.hit = True
            h.entity_id = pose.entity_id
            h.hitbox_id = pose.hitbox_id
            h.distance = res[1]
            h.fraction = Float32(0.0)
            h.point = res[2]
            h.normal = res[3]
            h.rewind_time_ms = win[4]
            h.was_clamped = win[5]
            hits.append(h)

        _sort_hits_deterministic(hits)
        return hits^

    def rewind_and_overlap_aabb(
        self,
        target_time_ms: Float64,
        box_min: Vec3,
        box_max: Vec3,
        ignore_entity_id: Int = -1,
    ) -> List[LagCompHit]:
        """Rewinds hitboxes to clamped `target_time_ms` and returns all AABB-overlapping poses."""
        var hits = List[LagCompHit]()
        if not _vec3_finite(box_min) or not _vec3_finite(box_max):
            return hits^
        var win = self._resolve_window(target_time_ms)
        if not win[0]:
            return hits^

        var lo = Vec3(
            min(box_min.x, box_max.x),
            min(box_min.y, box_max.y),
            min(box_min.z, box_max.z),
        )
        var hi = Vec3(
            max(box_min.x, box_max.x),
            max(box_min.y, box_max.y),
            max(box_min.z, box_max.z),
        )
        var box_center = (lo + hi) * Float32(0.5)
        var box_half = (hi - lo) * Float32(0.5)
        if not _vec3_finite(box_center) or not _vec3_finite(box_half):
            return hits^

        var poses = self._gather_poses_at(win[1], win[2], win[3])
        for i in range(len(poses)):
            var pose = poses[i]
            if not _pose_is_finite(pose) or pose.entity_id == ignore_entity_id:
                continue
            var res = _aabb_overlap_pose(lo, hi, box_center, box_half, pose)
            if not res[0] or not _is_finite32(res[1]):
                continue
            var h = LagCompHit()
            h.hit = True
            h.entity_id = pose.entity_id
            h.hitbox_id = pose.hitbox_id
            h.distance = res[1]
            h.fraction = Float32(0.0)
            h.point = res[2]
            h.normal = res[3]
            h.rewind_time_ms = win[4]
            h.was_clamped = win[5]
            hits.append(h)

        _sort_hits_deterministic(hits)
        return hits^


struct LagCompHistory(Movable):
    """Tick-indexed server lag-compensation history built over `HitboxSnapshotRing`."""
    var max_entities: Int
    var max_history_ticks: Int
    var max_rewind_ticks: Int
    var ring: HitboxSnapshotRing

    def __init__(
        out self,
        max_entities: Int = 64,
        max_history_ticks: Int = 64,
        max_rewind_ticks: Int = -1,
    ):
        self.max_entities = max(1, max_entities)
        self.max_history_ticks = max(1, max_history_ticks)
        self.max_rewind_ticks = (
            self.max_history_ticks
            if max_rewind_ticks < 0
            else max(0, max_rewind_ticks)
        )
        self.ring = HitboxSnapshotRing(
            capacity=self.max_history_ticks,
            max_rewind_ms=Float64(self.max_rewind_ticks),
        )

    def clear(mut self):
        self.ring.clear()

    @always_inline
    def tick_count(self) -> Int:
        return self.ring.snapshot_count()

    def record_tick(mut self, tick: Int, poses: List[HitboxPose]):
        self.ring.record_snapshot(tick, Float64(tick), poses)

    def raycast_at_tick(
        self,
        view_tick: Int,
        origin: Vec3,
        direction: Vec3,
        max_distance: Float32 = 1000.0,
        ignore_entity_id: Int = -1,
    ) -> LagCompHit:
        return self.ring.rewind_and_raycast(
            Float64(view_tick),
            origin,
            direction,
            max_distance,
            ignore_entity_id,
        )

    def overlap_at_tick(
        self,
        view_tick: Int,
        center: Vec3,
        radius: Float32,
        ignore_entity_id: Int = -1,
    ) -> List[LagCompHit]:
        return self.ring.rewind_and_overlap(
            Float64(view_tick), center, radius, ignore_entity_id
        )

    def overlap_aabb_at_tick(
        self,
        view_tick: Int,
        box_min: Vec3,
        box_max: Vec3,
        ignore_entity_id: Int = -1,
    ) -> List[LagCompHit]:
        return self.ring.rewind_and_overlap_aabb(
            Float64(view_tick), box_min, box_max, ignore_entity_id
        )
