"""Discrete Narrowphase: SAT, GJK-EPA, and multi-point contact manifolds (v4).

Implements:
1. ContactPoint and ContactManifold (up to 4 contact points per pair, normal A -> B).
2. Sphere-Sphere, Sphere-Capsule, Sphere-Box, Capsule-Box narrowphase.
3. Capsule-Capsule narrowphase with 2-point clipped manifolds for parallel capsules.
4. Box-Box 15-axis SAT with Sutherland-Hodgman face clipping and deterministic
   topological feature IDs `(ref_face << 16) | (edge_in << 8) | edge_out`.
5. Convex-Convex GJK + EPA + Reference/Incident polygon identification +
   Sutherland-Hodgman clipping + 4-point manifold reduction (zero 1-point rocking).
"""
from std.collections import List
from std.math import sqrt, min, max, abs
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.physics.solver_bodies import Sym3


struct ContactPoint(Copyable, ImplicitlyCopyable, Movable):
    var point: Vec3        # Contact point in world space
    var normal: Vec3       # Outward normal pointing from Body A to Body B
    var depth: Float32     # Penetration depth (positive when overlapping)
    var id: UInt32

    def __init__(out self):
        self.point = Vec3()
        self.normal = Vec3(0, 1, 0)
        self.depth = 0.0
        self.id = 0

    def __init__(out self, point: Vec3, normal: Vec3, depth: Float32, id: UInt32 = 0):
        self.point = point
        self.normal = normal
        self.depth = depth
        self.id = id


struct ContactManifold(Copyable, ImplicitlyCopyable, Movable):
    var count: Int
    var p0: ContactPoint
    var p1: ContactPoint
    var p2: ContactPoint
    var p3: ContactPoint
    var normal: Vec3
    var body_a: Int
    var body_b: Int
    var has_collision: Bool

    def __init__(out self, body_a: Int = 0, body_b: Int = 1):
        self.count = 0
        self.p0 = ContactPoint()
        self.p1 = ContactPoint()
        self.p2 = ContactPoint()
        self.p3 = ContactPoint()
        self.normal = Vec3(0, 1, 0)
        self.body_a = body_a
        self.body_b = body_b
        self.has_collision = False

    def add_point(mut self, point: Vec3, normal: Vec3, depth: Float32, id: UInt32 = 0):
        if self.count == 0:
            self.p0 = ContactPoint(point, normal, depth, id)
            self.normal = normal
            self.count = 1
            self.has_collision = True
        elif self.count == 1:
            self.p1 = ContactPoint(point, normal, depth, id)
            self.count = 2
        elif self.count == 2:
            self.p2 = ContactPoint(point, normal, depth, id)
            self.count = 3
        elif self.count == 3:
            self.p3 = ContactPoint(point, normal, depth, id)
            self.count = 4

    def get_point(self, index: Int) -> ContactPoint:
        if index == 0: return self.p0
        if index == 1: return self.p1
        if index == 2: return self.p2
        return self.p3


def reduce_manifold_candidates(mut m: ContactManifold, mut cands: List[ContactPoint]):
    """Sorts candidates descending by depth ( keeping p0 deepest ) and reduces >4 points
    to 4 maximally spread contact points for stable 3D stacking."""
    var n_cands = len(cands)
    if n_cands == 0:
        return
    for i in range(1, n_cands):
        var key = cands[i]
        var j = i - 1
        while j >= 0 and cands[j].depth < key.depth:
            cands[j + 1] = cands[j]
            j -= 1
        cands[j + 1] = key

    if n_cands <= 4:
        for k in range(n_cands):
            m.add_point(cands[k].point, cands[k].normal, cands[k].depth, cands[k].id)
        return

    # 1. p0 = deepest point
    var idx0 = 0
    # 2. p1 = furthest from p0
    var idx1 = 1
    var best_d2 = Float32(-1.0)
    for i in range(1, n_cands):
        var d2 = (cands[i].point - cands[idx0].point).length_squared()
        if d2 > best_d2:
            best_d2 = d2
            idx1 = i
    # 3. p2 = point maximizing triangle area with (p0, p1)
    var idx2 = -1
    var best_area = Float32(-1.0)
    var e01 = cands[idx1].point - cands[idx0].point
    for i in range(1, n_cands):
        if i == idx1:
            continue
        var area2 = e01.cross(cands[i].point - cands[idx0].point).length_squared()
        if area2 > best_area:
            best_area = area2
            idx2 = i
    if idx2 < 0:
        idx2 = 2 if idx1 != 2 else 1

    # 4. p3 = point maximizing minimum distance to (p0, p1, p2) / opposite side of quad
    var idx3 = -1
    var best_spread = Float32(-1.0)
    var tri_n = e01.cross(cands[idx2].point - cands[idx0].point)
    for i in range(1, n_cands):
        if i == idx1 or i == idx2:
            continue
        # Check signed side opposite to idx2 across (p0, p1), or general spread
        var cross_i = e01.cross(cands[i].point - cands[idx0].point)
        var opp_bonus = Float32(2.0) if cross_i.dot(tri_n) < Float32(0.0) else Float32(1.0)
        var d0 = (cands[i].point - cands[idx0].point).length_squared()
        var d1 = (cands[i].point - cands[idx1].point).length_squared()
        var d2 = (cands[i].point - cands[idx2].point).length_squared()
        var min_d = min(d0, min(d1, d2)) * opp_bonus
        if min_d > best_spread:
            best_spread = min_d
            idx3 = i
    if idx3 < 0:
        for i in range(1, n_cands):
            if i != idx1 and i != idx2:
                idx3 = i
                break

    m.add_point(cands[idx0].point, cands[idx0].normal, cands[idx0].depth, cands[idx0].id)
    m.add_point(cands[idx1].point, cands[idx1].normal, cands[idx1].depth, cands[idx1].id)
    m.add_point(cands[idx2].point, cands[idx2].normal, cands[idx2].depth, cands[idx2].id)
    if idx3 >= 0:
        m.add_point(cands[idx3].point, cands[idx3].normal, cands[idx3].depth, cands[idx3].id)


# =========================================================================
# 1. Sphere - Sphere, Sphere - Capsule, Sphere - Box, Capsule - Box
# =========================================================================

def collide_sphere_sphere(
    pos_a: Vec3,
    r_a: Float32,
    pos_b: Vec3,
    r_b: Float32,
    body_a: Int = 0,
    body_b: Int = 1,
    speculative_dist: Float32 = 0.0,
) -> ContactManifold:
    var m = ContactManifold(body_a, body_b)
    var diff = pos_b - pos_a
    var dist2 = diff.length_squared()
    var total_r = r_a + r_b
    var max_d = total_r + speculative_dist
    if dist2 <= max_d * max_d:
        var dist = sqrt(dist2)
        var normal = diff * (Float32(1.0) / dist) if dist > Float32(1.0e-6) else Vec3(0, 1, 0)
        var depth = total_r - dist
        var contact_pt = pos_a + normal * (r_a - depth * Float32(0.5))
        m.add_point(contact_pt, normal, depth, 0)
    return m^


def collide_sphere_capsule(
    s_center: Vec3,
    s_radius: Float32,
    cap_a: Vec3,
    cap_b: Vec3,
    cap_radius: Float32,
    body_a: Int = 0,
    body_b: Int = 1,
    speculative_dist: Float32 = 0.0,
) -> ContactManifold:
    var m = ContactManifold(body_a, body_b)
    var ab = cap_b - cap_a
    var l2 = ab.length_squared()
    var t = Float32(0.0)
    if l2 > 1.0e-8:
        t = max(Float32(0.0), min(Float32(1.0), (s_center - cap_a).dot(ab) / l2))
    var seg_pt = cap_a + ab * t

    var diff = seg_pt - s_center
    var dist2 = diff.length_squared()
    var total_r = s_radius + cap_radius
    var max_d = total_r + speculative_dist

    if dist2 <= max_d * max_d:
        var dist = sqrt(dist2)
        var normal = diff * (Float32(1.0) / dist) if dist > 1.0e-6 else Vec3(0, 1, 0)
        var depth = total_r - dist
        var contact_pt = s_center + normal * (s_radius - depth * Float32(0.5))
        m.add_point(contact_pt, normal, depth)
    return m^


@always_inline
def _sphere_local_aabb_contact(
    lp: Vec3,
    s_radius: Float32,
    b_pos: Vec3,
    b_rot: Quat,
    b_half: Vec3,
    id: UInt32,
    mut m: ContactManifold,
    speculative_dist: Float32 = 0.0,
):
    var cx = max(-b_half.x, min(b_half.x, lp.x))
    var cy = max(-b_half.y, min(b_half.y, lp.y))
    var cz = max(-b_half.z, min(b_half.z, lp.z))
    var cl = Vec3(cx, cy, cz)
    var diff = cl - lp  # A (sphere) -> B (box)
    var d2 = diff.length_squared()
    if d2 > Float32(1.0e-8):
        var max_d = s_radius + speculative_dist
        if d2 <= max_d * max_d:
            var dist = sqrt(d2)
            var ln = diff * (Float32(1.0) / dist)
            var wn = b_rot.rotate(ln).normalized()
            var depth = s_radius - dist
            var w_cl = b_pos + b_rot.rotate(cl)
            m.add_point(w_cl - wn * (depth * Float32(0.5)), wn, depth, id)
    else:
        # Sphere center inside Box B: push out to nearest box face (normal points A -> B)
        var dx = b_half.x - abs(lp.x)
        var dy = b_half.y - abs(lp.y)
        var dz = b_half.z - abs(lp.z)
        var surf_l: Vec3
        var ln: Vec3
        var min_d: Float32
        if dx <= dy and dx <= dz:
            min_d = dx
            var sx = Float32(1.0) if lp.x >= Float32(0.0) else Float32(-1.0)
            surf_l = Vec3(sx * b_half.x, lp.y, lp.z)
            ln = Vec3(-sx, 0.0, 0.0)
        elif dy <= dx and dy <= dz:
            min_d = dy
            var sy = Float32(1.0) if lp.y >= Float32(0.0) else Float32(-1.0)
            surf_l = Vec3(lp.x, sy * b_half.y, lp.z)
            ln = Vec3(0.0, -sy, 0.0)
        else:
            min_d = dz
            var sz = Float32(1.0) if lp.z >= Float32(0.0) else Float32(-1.0)
            surf_l = Vec3(lp.x, lp.y, sz * b_half.z)
            ln = Vec3(0.0, 0.0, -sz)
        var wn = b_rot.rotate(ln).normalized()
        var depth = s_radius + min_d
        var w_surf = b_pos + b_rot.rotate(surf_l)
        m.add_point(w_surf + wn * (depth * Float32(0.5)), wn, depth, id)


def collide_sphere_box(
    s_pos: Vec3,
    s_radius: Float32,
    b_pos: Vec3,
    b_rot: Quat,
    b_half: Vec3,
    body_a: Int = 0,
    body_b: Int = 1,
    speculative_dist: Float32 = 0.0,
) -> ContactManifold:
    var m = ContactManifold(body_a, body_b)
    var lp = b_rot.conjugate().rotate(s_pos - b_pos)
    _sphere_local_aabb_contact(lp, s_radius, b_pos, b_rot, b_half, 0, m, speculative_dist)
    return m^


# =========================================================================
# 2. Capsule - Capsule & Capsule - Box Narrowphase
# =========================================================================

def closest_points_segments(
    p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3,
) -> Tuple[Vec3, Vec3]:
    """Finds closest points between two 3D line segments p1-q1 and p2-q2."""
    var d1 = q1 - p1
    var d2 = q2 - p2
    var r = p1 - p2
    var a = d1.length_squared()
    var e = d2.length_squared()
    var f = d2.dot(r)

    var s: Float32
    var t: Float32

    if a <= 1.0e-8 and e <= 1.0e-8:
        return (p1, p2)
    if a <= 1.0e-8:
        s = 0.0
        t = max(Float32(0.0), min(Float32(1.0), f / e))
    else:
        var c = d1.dot(r)
        if e <= 1.0e-8:
            t = 0.0
            s = max(Float32(0.0), min(Float32(1.0), -c / a))
        else:
            var b = d1.dot(d2)
            var denom = a * e - b * b
            if denom > 1.0e-8:
                s = max(Float32(0.0), min(Float32(1.0), (b * f - c * e) / denom))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = max(Float32(0.0), min(Float32(1.0), -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(Float32(0.0), min(Float32(1.0), (b - c) / a))

    var c1 = p1 + d1 * s
    var c2 = p2 + d2 * t
    return (c1, c2)


def collide_capsule_capsule(
    c1_a: Vec3,
    c1_b: Vec3,
    r1: Float32,
    c2_a: Vec3,
    c2_b: Vec3,
    r2: Float32,
    body_a: Int = 0,
    body_b: Int = 1,
    speculative_dist: Float32 = 0.0,
) -> ContactManifold:
    var m = ContactManifold(body_a, body_b)
    var total_r = r1 + r2
    var max_d = total_r + speculative_dist
    var d1 = c1_b - c1_a
    var d2 = c2_b - c2_a
    var l1_sq = d1.length_squared()
    var l2_sq = d2.length_squared()

    # Check nearly parallel capsules (|d1_hat . d2_hat| > 0.999) for 2-point clipped manifold
    if l1_sq > Float32(1.0e-8) and l2_sq > Float32(1.0e-8):
        var l1 = sqrt(l1_sq)
        var l2 = sqrt(l2_sq)
        var u1 = d1 * (Float32(1.0) / l1)
        var u2 = d2 * (Float32(1.0) / l2)
        if abs(u1.dot(u2)) > Float32(0.999):
            var sa = (c2_a - c1_a).dot(u1)
            var sb = (c2_b - c1_a).dot(u1)
            var s_lo = max(Float32(0.0), min(sa, sb))
            var s_hi = min(l1, max(sa, sb))
            if s_hi - s_lo > Float32(1.0e-4):
                var s_vals = [s_lo, s_hi]
                for k in range(2):
                    var p1 = c1_a + u1 * s_vals[k]
                    var t_proj = max(Float32(0.0), min(Float32(1.0), (p1 - c2_a).dot(d2) / l2_sq))
                    var p2 = c2_a + d2 * t_proj
                    var diff = p2 - p1
                    var dist2 = diff.length_squared()
                    if dist2 <= max_d * max_d:
                        var dist = sqrt(dist2)
                        var normal = diff * (Float32(1.0) / dist) if dist > Float32(1.0e-6) else Vec3(0, 1, 0)
                        var depth = total_r - dist
                        var contact_pt = p1 + normal * (r1 - depth * Float32(0.5))
                        m.add_point(contact_pt, normal, depth, UInt32(k))
                if m.count > 0:
                    return m^

    var pts = closest_points_segments(c1_a, c1_b, c2_a, c2_b)
    var pt1 = pts[0]
    var pt2 = pts[1]
    var diff = pt2 - pt1
    var dist2 = diff.length_squared()

    if dist2 <= max_d * max_d:
        var dist = sqrt(dist2)
        var normal = diff * (Float32(1.0) / dist) if dist > 1.0e-6 else Vec3(0, 1, 0)
        var depth = total_r - dist
        var contact_pt = pt1 + normal * (r1 - depth * Float32(0.5))
        m.add_point(contact_pt, normal, depth, 0)
    return m^


@always_inline
def _local_aabb_signed_dist(p: Vec3, h: Vec3) -> Float32:
    var cx = max(-h.x, min(h.x, p.x))
    var cy = max(-h.y, min(h.y, p.y))
    var cz = max(-h.z, min(h.z, p.z))
    var d2 = (p - Vec3(cx, cy, cz)).length_squared()
    if d2 > Float32(1.0e-10):
        return sqrt(d2)
    var dx = h.x - abs(p.x)
    var dy = h.y - abs(p.y)
    var dz = h.z - abs(p.z)
    return -min(dx, min(dy, dz))


def collide_capsule_box(
    cap_a: Vec3,
    cap_b: Vec3,
    cap_radius: Float32,
    b_pos: Vec3,
    b_rot: Quat,
    b_half: Vec3,
    body_a: Int = 0,
    body_b: Int = 1,
) -> ContactManifold:
    var m = ContactManifold(body_a, body_b)
    var inv_q = b_rot.conjugate()
    var la = inv_q.rotate(cap_a - b_pos)
    var lb = inv_q.rotate(cap_b - b_pos)
    var ld = lb - la
    var l_len2 = ld.length_squared()

    if l_len2 < Float32(1.0e-8):
        _sphere_local_aabb_contact(la, cap_radius, b_pos, b_rot, b_half, 0, m)
        return m^

    # Check if capsule lies parallel to a box face: clip [0, 1] to box tangential bounds for 2-point support
    var l_len = sqrt(l_len2)
    var lu = ld * (Float32(1.0) / l_len)
    for face_ax in range(3):
        var ax_comp = abs(lu.x) if face_ax == 0 else (abs(lu.y) if face_ax == 1 else abs(lu.z))
        var mid_p = (la + lb) * Float32(0.5)
        var mid_coord = mid_p.x if face_ax == 0 else (mid_p.y if face_ax == 1 else mid_p.z)
        var f_half = b_half.x if face_ax == 0 else (b_half.y if face_ax == 1 else b_half.z)
        if ax_comp < Float32(0.05) and abs(mid_coord) >= f_half - cap_radius and abs(mid_coord) <= f_half + cap_radius:
            # Clip [0, 1] against the other two slab intervals
            var t0 = Float32(0.0)
            var t1 = Float32(1.0)
            var valid = True
            for oa in range(3):
                if oa == face_ax:
                    continue
                var p0_c = la.x if oa == 0 else (la.y if oa == 1 else la.z)
                var d_c = ld.x if oa == 0 else (ld.y if oa == 1 else ld.z)
                var h_c = b_half.x if oa == 0 else (b_half.y if oa == 1 else b_half.z)
                if abs(d_c) < Float32(1.0e-6):
                    if abs(p0_c) > h_c:
                        valid = False
                        break
                else:
                    var ta = (-h_c - p0_c) / d_c
                    var tb = (h_c - p0_c) / d_c
                    if ta > tb:
                        var tmp = ta; ta = tb; tb = tmp
                    t0 = max(t0, ta)
                    t1 = min(t1, tb)
                    if t0 > t1:
                        valid = False
                        break
            if valid and (t1 - t0) * l_len > Float32(1.0e-3):
                _sphere_local_aabb_contact(la + ld * t0, cap_radius, b_pos, b_rot, b_half, 0, m)
                _sphere_local_aabb_contact(la + ld * t1, cap_radius, b_pos, b_rot, b_half, 1, m)
                if m.count > 0:
                    return m^

    # General capsule-box: evaluate exact candidate parameters on segment [la, lb]
    var best_t = Float32(0.0)
    var best_sd = _local_aabb_signed_dist(la, b_half)
    var d_end = _local_aabb_signed_dist(lb, b_half)
    if d_end < best_sd:
        best_sd = d_end
        best_t = Float32(1.0)

    # Slab boundary crossings + interior midpoint
    var t_cands = List[Float32](capacity=24)
    t_cands.append(Float32(0.5))
    for ax in range(3):
        var p0_c = la.x if ax == 0 else (la.y if ax == 1 else la.z)
        var d_c = ld.x if ax == 0 else (ld.y if ax == 1 else ld.z)
        var h_c = b_half.x if ax == 0 else (b_half.y if ax == 1 else b_half.z)
        if abs(d_c) > Float32(1.0e-6):
            var ta = (-h_c - p0_c) / d_c
            var tb = (h_c - p0_c) / d_c
            if ta > Float32(0.0) and ta < Float32(1.0):
                t_cands.append(ta)
            if tb > Float32(0.0) and tb < Float32(1.0):
                t_cands.append(tb)
            var tm = max(Float32(0.0), min(Float32(1.0), (ta + tb) * Float32(0.5)))
            t_cands.append(tm)

    # 12 box edges closest points
    for s1 in range(2):
        var v1 = -Float32(1.0) if s1 == 0 else Float32(1.0)
        for s2 in range(2):
            var v2 = -Float32(1.0) if s2 == 0 else Float32(1.0)
            var ex0 = Vec3(-b_half.x, v1 * b_half.y, v2 * b_half.z)
            var ex1 = Vec3(b_half.x, v1 * b_half.y, v2 * b_half.z)
            var cpx = closest_points_segments(la, lb, ex0, ex1)
            t_cands.append(max(Float32(0.0), min(Float32(1.0), (cpx[0] - la).dot(ld) / l_len2)))

            var ey0 = Vec3(v1 * b_half.x, -b_half.y, v2 * b_half.z)
            var ey1 = Vec3(v1 * b_half.x, b_half.y, v2 * b_half.z)
            var cpy = closest_points_segments(la, lb, ey0, ey1)
            t_cands.append(max(Float32(0.0), min(Float32(1.0), (cpy[0] - la).dot(ld) / l_len2)))

            var ez0 = Vec3(v1 * b_half.x, v2 * b_half.y, -b_half.z)
            var ez1 = Vec3(v1 * b_half.x, v2 * b_half.y, b_half.z)
            var cpz = closest_points_segments(la, lb, ez0, ez1)
            t_cands.append(max(Float32(0.0), min(Float32(1.0), (cpz[0] - la).dot(ld) / l_len2)))

    for i in range(len(t_cands)):
        var tc = t_cands[i]
        var sd = _local_aabb_signed_dist(la + ld * tc, b_half)
        if sd < best_sd:
            best_sd = sd
            best_t = tc

    _sphere_local_aabb_contact(la + ld * best_t, cap_radius, b_pos, b_rot, b_half, 0, m)
    return m^


# =========================================================================
# 3. Box - Box Narrowphase (15-Axis SAT + Topological Sutherland-Hodgman)
# =========================================================================

def get_box_axes(rot: Quat) -> Tuple[Vec3, Vec3, Vec3]:
    var u0 = rot.rotate(Vec3(1, 0, 0)).normalized()
    var u1 = rot.rotate(Vec3(0, 1, 0)).normalized()
    var u2 = rot.rotate(Vec3(0, 0, 1)).normalized()
    return (u0, u1, u2)


@fieldwise_init
struct _TopoClipVert(Copyable, ImplicitlyCopyable, Movable):
    var p: Vec3
    var edge_in: UInt8
    var edge_out: UInt8


def _clip_poly_against_plane(poly: List[Vec3], plane_pt: Vec3, plane_n: Vec3) -> List[Vec3]:
    """Sutherland-Hodgman clip keeping points in half-space (p - plane_pt).dot(plane_n) <= 0."""
    var out = List[Vec3]()
    var n = len(poly)
    if n == 0:
        return out^
    for i in range(n):
        var a = poly[i]
        var b = poly[(i + 1) % n]
        var da = (a - plane_pt).dot(plane_n)
        var db = (b - plane_pt).dot(plane_n)
        if da <= 0.0:
            out.append(a)
            if db > 0.0:
                var t = da / (da - db)
                out.append(a + (b - a) * t)
        elif db <= 0.0:
            var t = da / (da - db)
            out.append(a + (b - a) * t)
    return out^


def _clip_poly_topo(
    poly: List[_TopoClipVert], plane_pt: Vec3, plane_n: Vec3, clip_edge_id: UInt8,
) -> List[_TopoClipVert]:
    """Topological Sutherland-Hodgman clip tracking incoming/outgoing feature edge IDs."""
    var out = List[_TopoClipVert]()
    var n = len(poly)
    if n == 0:
        return out^
    for i in range(n):
        var a = poly[i]
        var b = poly[(i + 1) % n]
        var da = (a.p - plane_pt).dot(plane_n)
        var db = (b.p - plane_pt).dot(plane_n)
        if da <= 0.0:
            out.append(a)
            if db > 0.0:
                var t = da / (da - db)
                out.append(_TopoClipVert(a.p + (b.p - a.p) * t, a.edge_out, clip_edge_id))
        elif db <= 0.0:
            var t = da / (da - db)
            out.append(_TopoClipVert(a.p + (b.p - a.p) * t, clip_edge_id, b.edge_in))
    return out^


def _box_face_quad(pos: Vec3, axes: Tuple[Vec3, Vec3, Vec3], half: Vec3, dir: Vec3) -> List[Vec3]:
    """Returns 4 vertices of the box face whose outward normal has maximum dot with `dir`."""
    var u0 = axes[0]; var u1 = axes[1]; var u2 = axes[2]
    var dots = [abs(u0.dot(dir)), abs(u1.dot(dir)), abs(u2.dot(dir))]
    var best_k = 0
    if dots[1] > dots[0]:
        best_k = 1
    if dots[2] > dots[best_k]:
        best_k = 2

    var n_ax: Vec3; var n_h: Float32
    var t0: Vec3; var h0: Float32
    var t1: Vec3; var h1: Float32
    if best_k == 0:
        var s = Float32(1.0) if u0.dot(dir) >= 0.0 else Float32(-1.0)
        n_ax = u0 * s; n_h = half.x
        t0 = u1; h0 = half.y
        t1 = u2; h1 = half.z
    elif best_k == 1:
        var s = Float32(1.0) if u1.dot(dir) >= 0.0 else Float32(-1.0)
        n_ax = u1 * s; n_h = half.y
        t0 = u0; h0 = half.x
        t1 = u2; h1 = half.z
    else:
        var s = Float32(1.0) if u2.dot(dir) >= 0.0 else Float32(-1.0)
        n_ax = u2 * s; n_h = half.z
        t0 = u0; h0 = half.x
        t1 = u1; h1 = half.y

    var c = pos + n_ax * n_h
    var quad = List[Vec3](capacity=4)
    quad.append(c + t0 * h0 + t1 * h1)
    quad.append(c - t0 * h0 + t1 * h1)
    quad.append(c - t0 * h0 - t1 * h1)
    quad.append(c + t0 * h0 - t1 * h1)
    return quad^


def _box_face_quad_topo(
    pos: Vec3, axes: Tuple[Vec3, Vec3, Vec3], half: Vec3, dir: Vec3,
) -> List[_TopoClipVert]:
    """Returns 4 vertices of the incident box face tagged with deterministic incident edge IDs."""
    var u0 = axes[0]; var u1 = axes[1]; var u2 = axes[2]
    var dots = [abs(u0.dot(dir)), abs(u1.dot(dir)), abs(u2.dot(dir))]
    var best_k = 0
    if dots[1] > dots[0]:
        best_k = 1
    if dots[2] > dots[best_k]:
        best_k = 2

    var n_ax: Vec3; var n_h: Float32
    var t0: Vec3; var h0: Float32
    var t1: Vec3; var h1: Float32
    var sign_bit: UInt8
    if best_k == 0:
        var pos_s = u0.dot(dir) >= 0.0
        var s = Float32(1.0) if pos_s else Float32(-1.0)
        sign_bit = UInt8(0) if pos_s else UInt8(1)
        n_ax = u0 * s; n_h = half.x
        t0 = u1; h0 = half.y
        t1 = u2; h1 = half.z
    elif best_k == 1:
        var pos_s = u1.dot(dir) >= 0.0
        var s = Float32(1.0) if pos_s else Float32(-1.0)
        sign_bit = UInt8(0) if pos_s else UInt8(1)
        n_ax = u1 * s; n_h = half.y
        t0 = u0; h0 = half.x
        t1 = u2; h1 = half.z
    else:
        var pos_s = u2.dot(dir) >= 0.0
        var s = Float32(1.0) if pos_s else Float32(-1.0)
        sign_bit = UInt8(0) if pos_s else UInt8(1)
        n_ax = u2 * s; n_h = half.z
        t0 = u0; h0 = half.x
        t1 = u1; h1 = half.y

    var base_e = UInt8((best_k * 2 + Int(sign_bit)) * 4 + 16)
    var c = pos + n_ax * n_h
    var quad = List[_TopoClipVert](capacity=4)
    quad.append(_TopoClipVert(c + t0 * h0 + t1 * h1, base_e + 3, base_e + 0))
    quad.append(_TopoClipVert(c - t0 * h0 + t1 * h1, base_e + 0, base_e + 1))
    quad.append(_TopoClipVert(c - t0 * h0 - t1 * h1, base_e + 1, base_e + 2))
    quad.append(_TopoClipVert(c + t0 * h0 - t1 * h1, base_e + 2, base_e + 3))
    return quad^


def collide_box_box(
    pos_a: Vec3,
    rot_a: Quat,
    half_a: Vec3,
    pos_b: Vec3,
    rot_b: Quat,
    half_b: Vec3,
    body_a: Int = 0,
    body_b: Int = 1,
) -> ContactManifold:
    var m = ContactManifold(body_a, body_b)
    var a_axes = get_box_axes(rot_a)
    var b_axes = get_box_axes(rot_b)

    var u0 = a_axes[0]; var u1 = a_axes[1]; var u2 = a_axes[2]
    var v0 = b_axes[0]; var v1 = b_axes[1]; var v2 = b_axes[2]

    var dp = pos_b - pos_a

    var c00 = u0.dot(v0); var c01 = u0.dot(v1); var c02 = u0.dot(v2)
    var c10 = u1.dot(v0); var c11 = u1.dot(v1); var c12 = u1.dot(v2)
    var c20 = u2.dot(v0); var c21 = u2.dot(v1); var c22 = u2.dot(v2)

    var min_overlap = Float32(1.0e30)
    var best_normal = Vec3(0, 1, 0)
    var best_axis_idx = 0

    # 1. Test 3 face axes of Box A (0..2)
    var d_u0 = abs(dp.dot(u0))
    var r_u0 = half_a.x + (half_b.x * abs(c00) + half_b.y * abs(c01) + half_b.z * abs(c02))
    var o_u0 = r_u0 - d_u0
    if o_u0 <= 0.0: return m^
    if o_u0 < min_overlap:
        min_overlap = o_u0
        best_normal = u0 if dp.dot(u0) >= 0.0 else -u0
        best_axis_idx = 0

    var d_u1 = abs(dp.dot(u1))
    var r_u1 = half_a.y + (half_b.x * abs(c10) + half_b.y * abs(c11) + half_b.z * abs(c12))
    var o_u1 = r_u1 - d_u1
    if o_u1 <= 0.0: return m^
    if o_u1 < min_overlap:
        min_overlap = o_u1
        best_normal = u1 if dp.dot(u1) >= 0.0 else -u1
        best_axis_idx = 1

    var d_u2 = abs(dp.dot(u2))
    var r_u2 = half_a.z + (half_b.x * abs(c20) + half_b.y * abs(c21) + half_b.z * abs(c22))
    var o_u2 = r_u2 - d_u2
    if o_u2 <= 0.0: return m^
    if o_u2 < min_overlap:
        min_overlap = o_u2
        best_normal = u2 if dp.dot(u2) >= 0.0 else -u2
        best_axis_idx = 2

    # 2. Test 3 face axes of Box B (3..5)
    var d_v0 = abs(dp.dot(v0))
    var r_v0 = (half_a.x * abs(c00) + half_a.y * abs(c10) + half_a.z * abs(c20)) + half_b.x
    var o_v0 = r_v0 - d_v0
    if o_v0 <= 0.0: return m^
    if o_v0 < min_overlap:
        min_overlap = o_v0
        best_normal = v0 if dp.dot(v0) >= 0.0 else -v0
        best_axis_idx = 3

    var d_v1 = abs(dp.dot(v1))
    var r_v1 = (half_a.x * abs(c01) + half_a.y * abs(c11) + half_a.z * abs(c21)) + half_b.y
    var o_v1 = r_v1 - d_v1
    if o_v1 <= 0.0: return m^
    if o_v1 < min_overlap:
        min_overlap = o_v1
        best_normal = v1 if dp.dot(v1) >= 0.0 else -v1
        best_axis_idx = 4

    var d_v2 = abs(dp.dot(v2))
    var r_v2 = (half_a.x * abs(c02) + half_a.y * abs(c12) + half_a.z * abs(c22)) + half_b.z
    var o_v2 = r_v2 - d_v2
    if o_v2 <= 0.0: return m^
    if o_v2 < min_overlap:
        min_overlap = o_v2
        best_normal = v2 if dp.dot(v2) >= 0.0 else -v2
        best_axis_idx = 5

    # 3. Test 9 edge-edge cross axes u_i x v_j (6..14)
    var u_list = [u0, u1, u2]
    var v_list = [v0, v1, v2]
    var best_edge_i = 0
    var best_edge_j = 0
    for ei in range(3):
        for ej in range(3):
            var raw_axis = u_list[ei].cross(v_list[ej])
            var len2 = raw_axis.length_squared()
            if len2 <= Float32(1.0e-6):
                continue
            var axis = raw_axis * (Float32(1.0) / sqrt(len2))
            var d_e = abs(dp.dot(axis))
            var ra = half_a.x * abs(u0.dot(axis)) + half_a.y * abs(u1.dot(axis)) + half_a.z * abs(u2.dot(axis))
            var rb = half_b.x * abs(v0.dot(axis)) + half_b.y * abs(v1.dot(axis)) + half_b.z * abs(v2.dot(axis))
            var ov = (ra + rb) - d_e
            if ov <= Float32(0.0):
                return m^
            if ov < min_overlap * Float32(0.995):
                min_overlap = ov
                best_normal = axis if dp.dot(axis) >= 0.0 else -axis
                best_axis_idx = 6 + ei * 3 + ej
                best_edge_i = ei
                best_edge_j = ej

    # 4. Generate contact manifold with deterministic topological feature IDs
    if best_axis_idx < 6:
        var ref_is_a = best_axis_idx < 3
        var ref_pos = pos_a if ref_is_a else pos_b
        var ref_axes = a_axes if ref_is_a else b_axes
        var ref_half = half_a if ref_is_a else half_b
        var ref_n = best_normal if ref_is_a else -best_normal

        var inc_pos = pos_b if ref_is_a else pos_a
        var inc_axes = b_axes if ref_is_a else a_axes
        var inc_half = half_b if ref_is_a else half_a

        var local_idx = best_axis_idx if ref_is_a else (best_axis_idx - 3)
        var ref_axis_vec = ref_axes[0] if local_idx == 0 else (ref_axes[1] if local_idx == 1 else ref_axes[2])
        var ref_sign_id = 0 if ref_n.dot(ref_axis_vec) >= 0.0 else 1
        var ref_face_id = UInt32((0 if ref_is_a else 6) + local_idx * 2 + ref_sign_id + 1)

        var ref_face_h = ref_half.x if local_idx == 0 else (ref_half.y if local_idx == 1 else ref_half.z)
        var ref_face_c = ref_pos + ref_n * ref_face_h

        var t0 = ref_axes[1] if local_idx == 0 else ref_axes[0]
        var h0 = ref_half.y if local_idx == 0 else ref_half.x
        var t1 = ref_axes[2] if local_idx != 2 else ref_axes[1]
        var h1 = ref_half.z if local_idx != 2 else ref_half.y

        var poly = _box_face_quad_topo(inc_pos, inc_axes, inc_half, -ref_n)
        poly = _clip_poly_topo(poly, ref_face_c + t0 * h0, t0, UInt8(1))
        poly = _clip_poly_topo(poly, ref_face_c - t0 * h0, -t0, UInt8(2))
        poly = _clip_poly_topo(poly, ref_face_c + t1 * h1, t1, UInt8(3))
        poly = _clip_poly_topo(poly, ref_face_c - t1 * h1, -t1, UInt8(4))

        var cands = List[ContactPoint](capacity=len(poly))
        for k in range(len(poly)):
            var cv = poly[k]
            var pen = -(cv.p - ref_face_c).dot(ref_n)
            if pen > Float32(-1.0e-4):
                var depth = max(Float32(0.0), pen)
                var contact_pt = cv.p + ref_n * (depth * Float32(0.5))
                var feat_id = (ref_face_id << 16) | (UInt32(cv.edge_in) << 8) | UInt32(cv.edge_out)
                cands.append(ContactPoint(contact_pt, best_normal, depth, feat_id))

        reduce_manifold_candidates(m, cands)

        if m.count == 0:
            var sa_off = (
                u0 * (half_a.x if u0.dot(best_normal) >= 0.0 else -half_a.x)
                + u1 * (half_a.y if u1.dot(best_normal) >= 0.0 else -half_a.y)
                + u2 * (half_a.z if u2.dot(best_normal) >= 0.0 else -half_a.z)
            )
            var c_pt = pos_a + sa_off - best_normal * (min_overlap * Float32(0.5))
            m.add_point(c_pt, best_normal, min_overlap, ref_face_id << 16)
    else:
        var ha_arr = [half_a.x, half_a.y, half_a.z]
        var hb_arr = [half_b.x, half_b.y, half_b.z]
        var pa_mid = pos_a
        for k in range(3):
            if k != best_edge_i:
                var s = ha_arr[k] if u_list[k].dot(best_normal) >= 0.0 else -ha_arr[k]
                pa_mid = pa_mid + u_list[k] * s
        var pb_mid = pos_b
        for k in range(3):
            if k != best_edge_j:
                var s = hb_arr[k] if v_list[k].dot(-best_normal) >= 0.0 else -hb_arr[k]
                pb_mid = pb_mid + v_list[k] * s
        var ea_dir = u_list[best_edge_i] * ha_arr[best_edge_i]
        var eb_dir = v_list[best_edge_j] * hb_arr[best_edge_j]
        var cps = closest_points_segments(pa_mid - ea_dir, pa_mid + ea_dir, pb_mid - eb_dir, pb_mid + eb_dir)
        var edge_feat = (UInt32(13 + best_axis_idx - 6) << 16) | (UInt32(best_edge_i + 1) << 8) | UInt32(best_edge_j + 1)
        m.add_point((cps[0] + cps[1]) * Float32(0.5), best_normal, min_overlap, edge_feat)

    return m^


# =========================================================================
# 4. Convex Hull Narrowphase (GJK + EPA + Clipped 4-Point Manifold)
# =========================================================================

struct _QhFace(Copyable, Movable):
    var i0: Int
    var i1: Int
    var i2: Int
    var n: Vec3
    var offset: Float32
    var alive: Bool
    var furthest_pt: Int
    var furthest_dist: Float32
    var outside: List[Int]

    def __init__(out self, i0: Int, i1: Int, i2: Int, n: Vec3, offset: Float32):
        self.i0 = i0
        self.i1 = i1
        self.i2 = i2
        self.n = n
        self.offset = offset
        self.alive = True
        self.furthest_pt = -1
        self.furthest_dist = Float32(-1.0)
        self.outside = List[Int]()

    def assign_point(mut self, pt_idx: Int, dist: Float32):
        self.outside.append(pt_idx)
        if dist > self.furthest_dist:
            self.furthest_dist = dist
            self.furthest_pt = pt_idx


def _make_qh_face(pts: List[Vec3], i0: Int, i1: Int, i2: Int, interior: Vec3) -> _QhFace:
    var a = pts[i0]
    var b = pts[i1]
    var c = pts[i2]
    var raw_n = (b - a).cross(c - a)
    var l2 = raw_n.length_squared()
    var n = raw_n * (Float32(1.0) / sqrt(l2)) if l2 > Float32(1.0e-16) else Vec3(0, 1, 0)
    var b_idx = i1
    var c_idx = i2
    if n.dot(a - interior) < Float32(0.0):
        n = -n
        b_idx = i2
        c_idx = i1
    return _QhFace(i0, b_idx, c_idx, n, n.dot(a))


def _quickhull_core(pts: List[Vec3], weld_eps: Float32) -> Tuple[List[Vec3], List[Int32]]:
    """Builds outward-oriented 3D convex hull `(hull_vertices, tri_indices)` from welded points."""
    var n_pts = len(pts)
    var out_v = List[Vec3]()
    var out_idx = List[Int32]()
    if n_pts == 0:
        return (out_v^, out_idx^)
    if n_pts < 4:
        for i in range(n_pts):
            out_v.append(pts[i])
        return (out_v^, out_idx^)

    # Find bounding box diagonal & 6 axis extrema for stable initial simplex
    var min_idx = [0, 0, 0]
    var max_idx = [0, 0, 0]
    var lo = pts[0]
    var hi = pts[0]
    for i in range(1, n_pts):
        var p = pts[i]
        if p.x < lo.x: lo.x = p.x; min_idx[0] = i
        if p.x > hi.x: hi.x = p.x; max_idx[0] = i
        if p.y < lo.y: lo.y = p.y; min_idx[1] = i
        if p.y > hi.y: hi.y = p.y; max_idx[1] = i
        if p.z < lo.z: lo.z = p.z; min_idx[2] = i
        if p.z > hi.z: hi.z = p.z; max_idx[2] = i

    var diag = (hi - lo).length()
    var eps = max(weld_eps, diag * Float32(1.0e-5))

    var extrema = [min_idx[0], max_idx[0], min_idx[1], max_idx[1], min_idx[2], max_idx[2]]
    var i0 = extrema[0]
    var i1 = extrema[1]
    var best_seg2 = Float32(-1.0)
    for a_i in range(6):
        for b_i in range(a_i + 1, 6):
            var d2 = (pts[extrema[b_i]] - pts[extrema[a_i]]).length_squared()
            if d2 > best_seg2:
                best_seg2 = d2
                i0 = extrema[a_i]
                i1 = extrema[b_i]
    if best_seg2 <= eps * eps:
        out_v.append(pts[0])
        return (out_v^, out_idx^)

    # Third vertex maximizing distance to line (i0, i1)
    var line_d = (pts[i1] - pts[i0]).normalized()
    var i2 = -1
    var best_line_d2 = Float32(0.0)
    for i in range(n_pts):
        if i == i0 or i == i1:
            continue
        var d2 = (pts[i] - pts[i0]).cross(line_d).length_squared()
        if d2 > best_line_d2:
            best_line_d2 = d2
            i2 = i
    if i2 < 0 or best_line_d2 <= eps * eps:
        out_v.append(pts[i0])
        out_v.append(pts[i1])
        return (out_v^, out_idx^)

    # Fourth vertex maximizing distance to plane (i0, i1, i2)
    var plane_n = (pts[i1] - pts[i0]).cross(pts[i2] - pts[i0]).normalized()
    var i3 = -1
    var best_plane_d = Float32(0.0)
    for i in range(n_pts):
        if i == i0 or i == i1 or i == i2:
            continue
        var dp = abs((pts[i] - pts[i0]).dot(plane_n))
        if dp > best_plane_d:
            best_plane_d = dp
            i3 = i
    if i3 < 0 or best_plane_d <= eps:
        # Coplanar point cloud: return unique boundary vertices without 3D volume
        for i in range(n_pts):
            out_v.append(pts[i])
        return (out_v^, out_idx^)

    var interior = (pts[i0] + pts[i1] + pts[i2] + pts[i3]) * Float32(0.25)
    var faces = List[_QhFace]()
    faces.append(_make_qh_face(pts, i0, i1, i2, interior))
    faces.append(_make_qh_face(pts, i0, i2, i3, interior))
    faces.append(_make_qh_face(pts, i0, i3, i1, interior))
    faces.append(_make_qh_face(pts, i1, i3, i2, interior))

    var on_hull = List[Bool](length=n_pts, fill=False)
    on_hull[i0] = True
    on_hull[i1] = True
    on_hull[i2] = True
    on_hull[i3] = True

    for p_i in range(n_pts):
        if on_hull[p_i]:
            continue
        var best_f = -1
        var max_sd = eps
        for f_i in range(4):
            var sd = faces[f_i].n.dot(pts[p_i]) - faces[f_i].offset
            if sd > max_sd:
                max_sd = sd
                best_f = f_i
        if best_f >= 0:
            faces[best_f].assign_point(p_i, max_sd)

    # Horizon-ridge expansion loop
    var max_iters = n_pts * 2 + 8
    for _ in range(max_iters):
        var best_f = -1
        var best_d = eps
        for f_i in range(len(faces)):
            if faces[f_i].alive and faces[f_i].furthest_pt >= 0 and faces[f_i].furthest_dist > best_d:
                best_d = faces[f_i].furthest_dist
                best_f = f_i
        if best_f < 0:
            break

        var eye_idx = faces[best_f].furthest_pt
        if on_hull[eye_idx]:
            faces[best_f].furthest_pt = -1
            faces[best_f].furthest_dist = Float32(-1.0)
            continue
        on_hull[eye_idx] = True
        var eye = pts[eye_idx]

        # Collect visible conflict faces and horizon ridge edges
        var horizon_a = List[Int]()
        var horizon_b = List[Int]()
        var orphans = List[Int]()
        var vis_eps = max(Float32(1.0e-6), eps * Float32(0.05))

        for f_i in range(len(faces)):
            if not faces[f_i].alive:
                continue
            var sd = faces[f_i].n.dot(eye) - faces[f_i].offset
            if f_i == best_f or sd > vis_eps:
                faces[f_i].alive = False
                for oi in range(len(faces[f_i].outside)):
                    var op = faces[f_i].outside[oi]
                    if not on_hull[op]:
                        orphans.append(op)
                var fe = [
                    (faces[f_i].i0, faces[f_i].i1),
                    (faces[f_i].i1, faces[f_i].i2),
                    (faces[f_i].i2, faces[f_i].i0),
                ]
                for ei in range(3):
                    var ea = fe[ei][0]
                    var eb = fe[ei][1]
                    var cancel = -1
                    for hk in range(len(horizon_a)):
                        if horizon_a[hk] == eb and horizon_b[hk] == ea:
                            cancel = hk
                            break
                    if cancel >= 0:
                        horizon_a[cancel] = -1
                        horizon_b[cancel] = -1
                    else:
                        horizon_a.append(ea)
                        horizon_b.append(eb)

        var new_face_start = len(faces)
        for hk in range(len(horizon_a)):
            if horizon_a[hk] >= 0:
                faces.append(_make_qh_face(pts, horizon_a[hk], horizon_b[hk], eye_idx, interior))
        var new_face_end = len(faces)
        if new_face_end == new_face_start:
            break

        # Partition orphan points across newly constructed cone faces
        for oi in range(len(orphans)):
            var op = orphans[oi]
            if on_hull[op]:
                continue
            var target_f = -1
            var max_sd = eps
            for nf in range(new_face_start, new_face_end):
                var sd = faces[nf].n.dot(pts[op]) - faces[nf].offset
                if sd > max_sd:
                    max_sd = sd
                    target_f = nf
            if target_f >= 0:
                faces[target_f].assign_point(op, max_sd)

    # Remap alive face vertex indices into compact hull vertex array
    var remap = List[Int](length=n_pts, fill=-1)
    for f_i in range(len(faces)):
        if not faces[f_i].alive:
            continue
        var tri_v = [faces[f_i].i0, faces[f_i].i1, faces[f_i].i2]
        for k in range(3):
            var orig = tri_v[k]
            if remap[orig] < 0:
                remap[orig] = len(out_v)
                out_v.append(pts[orig])
        var r0 = remap[faces[f_i].i0]
        var r1 = remap[faces[f_i].i1]
        var r2 = remap[faces[f_i].i2]
        if r0 != r1 and r1 != r2 and r2 != r0:
            out_idx.append(Int32(r0))
            out_idx.append(Int32(r1))
            out_idx.append(Int32(r2))

    return (out_v^, out_idx^)


struct ConvexHull(Copyable, Movable):
    """Convex polyhedron defined by vertex cloud and optional outward CCW triangle faces."""
    var vertices: List[Vec3]
    var indices: List[Int32]

    def __init__(out self):
        self.vertices = List[Vec3]()
        self.indices = List[Int32]()

    def __init__(out self, vertices: List[Vec3]):
        self.vertices = vertices.copy()
        self.indices = List[Int32]()

    def __init__(out self, vertices: List[Vec3], indices: List[Int32]):
        self.vertices = vertices.copy()
        self.indices = indices.copy()

    def support(self, dir: Vec3, pos: Vec3, rot: Quat) -> Vec3:
        """Finds supporting vertex in direction dir (world space)."""
        var inv_rot = rot.conjugate()
        var local_dir = inv_rot.rotate(dir)
        var best_dot = Float32(-1.0e30)
        var best_v = Vec3()
        for i in range(len(self.vertices)):
            var d = self.vertices[i].dot(local_dir)
            if d > best_dot:
                best_dot = d
                best_v = self.vertices[i]
        return pos + rot.rotate(best_v)

    @staticmethod
    def build_quickhull(
        points: List[Vec3],
        max_verts: Int = 64,
        weld_eps: Float32 = 1.0e-4,
    ) -> ConvexHull:
        """3D Quickhull with spatial vertex welding, horizon-ridge face expansion,
        and deterministic farthest-point vertex capping (`<= max_verts`)."""
        var welded = List[Vec3]()
        var eps2 = max(Float32(0.0), weld_eps) * max(Float32(0.0), weld_eps)
        for i in range(len(points)):
            var p = points[i]
            var dup = False
            for j in range(len(welded)):
                if (p - welded[j]).length_squared() <= eps2:
                    dup = True
                    break
            if not dup:
                welded.append(p)

        var qh = _quickhull_core(welded, max(Float32(1.0e-6), weld_eps))
        var hull_v = qh[0].copy()
        var hull_i = qh[1].copy()

        var cap = max(4, max_verts)
        if len(hull_v) > cap:
            # Farthest-point sampling seeded with 6-axis extrema to preserve bounding volume
            var m_count = len(hull_v)
            var picked = List[Bool](length=m_count, fill=False)
            var capped = List[Vec3](capacity=cap)
            var min_d2 = List[Float32](length=m_count, fill=Float32(1.0e30))

            # Seed 0: minimum X (deterministic)
            var seed0 = 0
            for i in range(1, m_count):
                if hull_v[i].x < hull_v[seed0].x:
                    seed0 = i
            picked[seed0] = True
            capped.append(hull_v[seed0])
            for i in range(m_count):
                min_d2[i] = (hull_v[i] - hull_v[seed0]).length_squared()

            # Add remaining 5 axis extrema first (if distinct and cap permits)
            var axes = [
                Vec3(1, 0, 0), Vec3(0, -1, 0), Vec3(0, 1, 0), Vec3(0, 0, -1), Vec3(0, 0, 1),
            ]
            for ax_i in range(5):
                if len(capped) >= cap:
                    break
                var best_ext = -1
                var best_proj = Float32(-1.0e30)
                for i in range(m_count):
                    var proj = hull_v[i].dot(axes[ax_i])
                    if proj > best_proj:
                        best_proj = proj
                        best_ext = i
                if best_ext >= 0 and not picked[best_ext]:
                    picked[best_ext] = True
                    var chosen = hull_v[best_ext]
                    capped.append(chosen)
                    for i in range(m_count):
                        var d2 = (hull_v[i] - chosen).length_squared()
                        if d2 < min_d2[i]:
                            min_d2[i] = d2

            while len(capped) < cap:
                var best_k = -1
                var best_dist2 = Float32(-1.0)
                for i in range(m_count):
                    if not picked[i] and min_d2[i] > best_dist2:
                        best_dist2 = min_d2[i]
                        best_k = i
                if best_k < 0:
                    break
                picked[best_k] = True
                var chosen = hull_v[best_k]
                capped.append(chosen)
                for i in range(m_count):
                    var d2 = (hull_v[i] - chosen).length_squared()
                    if d2 < min_d2[i]:
                        min_d2[i] = d2

            var qh_capped = _quickhull_core(capped, max(Float32(1.0e-6), weld_eps))
            return ConvexHull(qh_capped[0], qh_capped[1])

        return ConvexHull(hull_v, hull_i)

    def compute_polyhedral_mass_properties(
        self, density: Float32,
    ) -> Tuple[Float32, Vec3, Sym3]:
        """Computes exact `(mass, center_of_mass, inertia_tensor_about_com)` for uniform `density`
        via Mirtich/Eberly signed-tetrahedron integration."""
        var rho = max(Float32(1.0e-6), density)
        if len(self.vertices) < 4:
            return (Float32(0.0), Vec3(), Sym3())

        var verts: List[Vec3]
        var tris: List[Int32]
        if len(self.indices) >= 12:
            verts = self.vertices.copy()
            tris = self.indices.copy()
        else:
            var built = ConvexHull.build_quickhull(
                self.vertices, max_verts=max(64, len(self.vertices)), weld_eps=Float32(1.0e-5)
            )
            verts = built.vertices.copy()
            tris = built.indices.copy()

        var n_tris = len(tris) // 3
        if n_tris < 4 or len(verts) < 4:
            return (Float32(0.0), Vec3(), Sym3())

        # Reference centroid to avoid floating-point cancellation far from origin
        var p_ref = Vec3()
        for i in range(len(verts)):
            p_ref = p_ref + verts[i]
        p_ref = p_ref * (Float32(1.0) / Float32(len(verts)))

        # Pass 1: exact signed volume and Center of Mass
        var det_sum = Float32(0.0)
        var com_accum = Vec3()
        for t in range(n_tris):
            var a = verts[Int(tris[t * 3 + 0])] - p_ref
            var b = verts[Int(tris[t * 3 + 1])] - p_ref
            var c = verts[Int(tris[t * 3 + 2])] - p_ref
            var det = a.dot(b.cross(c))
            det_sum += det
            com_accum = com_accum + (a + b + c) * det

        var vol = det_sum * Float32(1.0 / 6.0)
        if vol <= Float32(1.0e-12):
            if vol < Float32(-1.0e-12):
                vol = -vol
                det_sum = -det_sum
                com_accum = -com_accum
            else:
                return (Float32(0.0), p_ref, Sym3())

        var com = p_ref + com_accum * (Float32(1.0) / (Float32(4.0) * det_sum))
        var mass = rho * vol

        # Pass 2: exact canonical tetrahedron second-moment covariance matrix about CoM
        var cxx = Float32(0.0)
        var cyy = Float32(0.0)
        var czz = Float32(0.0)
        var cxy = Float32(0.0)
        var cxz = Float32(0.0)
        var cyz = Float32(0.0)

        var sign_fix = Float32(1.0) if (
            (verts[Int(tris[0])] - com).dot(
                (verts[Int(tris[1])] - com).cross(verts[Int(tris[2])] - com)
            ) >= Float32(0.0)
        ) else Float32(-1.0)

        for t in range(n_tris):
            var a = verts[Int(tris[t * 3 + 0])] - com
            var b = verts[Int(tris[t * 3 + 1])] - com
            var c = verts[Int(tris[t * 3 + 2])] - com
            var det = a.dot(b.cross(c)) * sign_fix
            var w = det * Float32(1.0 / 120.0)

            var sx = a.x + b.x + c.x
            var sy = a.y + b.y + c.y
            var sz = a.z + b.z + c.z

            cxx += w * (a.x * a.x + b.x * b.x + c.x * c.x + sx * sx)
            cyy += w * (a.y * a.y + b.y * b.y + c.y * c.y + sy * sy)
            czz += w * (a.z * a.z + b.z * b.z + c.z * c.z + sz * sz)

            cxy += w * (Float32(2.0) * (a.x * a.y + b.x * b.y + c.x * c.y) + a.x * b.y + b.x * a.y + a.x * c.y + c.x * a.y + b.x * c.y + c.x * b.y)
            cxz += w * (Float32(2.0) * (a.x * a.z + b.x * b.z + c.x * c.z) + a.x * b.z + b.x * a.z + a.x * c.z + c.x * a.z + b.x * c.z + c.x * b.z)
            cyz += w * (Float32(2.0) * (a.y * a.z + b.y * b.z + c.y * c.z) + a.y * b.z + b.y * a.z + a.y * c.z + c.y * a.z + b.y * c.z + c.y * b.z)

        var ixx = rho * (cyy + czz)
        var iyy = rho * (cxx + czz)
        var izz = rho * (cxx + cyy)
        var ixy = -rho * cxy
        var ixz = -rho * cxz
        var iyz = -rho * cyz
        return (mass, com, Sym3(ixx, iyy, izz, ixy, ixz, iyz))


@fieldwise_init
struct _MinkowskiPt(Copyable, ImplicitlyCopyable, Movable):
    var v: Vec3
    var sa: Vec3
    var sb: Vec3


@fieldwise_init
struct _EpaTri(Copyable, ImplicitlyCopyable, Movable):
    var i0: Int
    var i1: Int
    var i2: Int
    var n: Vec3
    var dist: Float32
    var alive: Bool


def _support_minkowski_pt(
    hull_a: ConvexHull, pos_a: Vec3, rot_a: Quat,
    hull_b: ConvexHull, pos_b: Vec3, rot_b: Quat,
    dir: Vec3,
) -> _MinkowskiPt:
    var sa = hull_a.support(dir, pos_a, rot_a)
    var sb = hull_b.support(-dir, pos_b, rot_b)
    return _MinkowskiPt(sa - sb, sa, sb)


def support_minkowski(
    hull_a: ConvexHull, pos_a: Vec3, rot_a: Quat,
    hull_b: ConvexHull, pos_b: Vec3, rot_b: Quat,
    dir: Vec3,
) -> Vec3:
    return hull_a.support(dir, pos_a, rot_a) - hull_b.support(-dir, pos_b, rot_b)


def _make_epa_tri(verts: List[_MinkowskiPt], i0: Int, i1: Int, i2: Int, interior: Vec3) -> _EpaTri:
    var a = verts[i0].v
    var b = verts[i1].v
    var c = verts[i2].v
    var raw_n = (b - a).cross(c - a)
    var len2 = raw_n.length_squared()
    var n = raw_n * (Float32(1.0) / sqrt(len2)) if len2 > Float32(1.0e-12) else Vec3(0, 1, 0)
    var idx1 = i1
    var idx2 = i2
    if n.dot(a - interior) < Float32(0.0):
        n = -n
        idx1 = i2
        idx2 = i1
    var dist = n.dot(a)
    if dist < Float32(0.0):
        dist = Float32(0.0)
    return _EpaTri(i0, idx1, idx2, n, dist, True)


def _barycentric_on_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Tuple[Float32, Float32, Float32]:
    var v0 = b - a
    var v1 = c - a
    var v2 = p - a
    var d00 = v0.dot(v0)
    var d01 = v0.dot(v1)
    var d11 = v1.dot(v1)
    var d20 = v2.dot(v0)
    var d21 = v2.dot(v1)
    var denom = d00 * d11 - d01 * d01
    if abs(denom) < Float32(1.0e-10):
        return (Float32(1.0 / 3.0), Float32(1.0 / 3.0), Float32(1.0 / 3.0))
    var v = (d11 * d20 - d01 * d21) / denom
    var w = (d00 * d21 - d01 * d20) / denom
    var u = Float32(1.0) - v - w
    return (u, v, w)


def _same_dir_perp(ab: Vec3) -> Vec3:
    var cand = ab.cross(Vec3(1, 0, 0))
    if cand.length_squared() < Float32(1.0e-6):
        cand = ab.cross(Vec3(0, 1, 0))
    return cand


def _plane_basis(n: Vec3) -> Tuple[Vec3, Vec3]:
    var t1: Vec3
    if abs(n.x) >= Float32(0.57735):
        t1 = Vec3(n.y, -n.x, 0.0)
    else:
        t1 = Vec3(0.0, n.z, -n.y)
    t1 = t1.normalized()
    var t2 = n.cross(t1).normalized()
    return (t1, t2)


def _support_feature_polygon(
    hull: ConvexHull, pos: Vec3, rot: Quat, world_dir: Vec3,
) -> List[Vec3]:
    """Extracts the world-space supporting vertex, edge, or CCW convex face polygon of `hull` in `world_dir`."""
    var local_dir = rot.conjugate().rotate(world_dir)
    var max_d = Float32(-1.0e30)
    var min_d = Float32(1.0e30)
    for i in range(len(hull.vertices)):
        var d = hull.vertices[i].dot(local_dir)
        if d > max_d:
            max_d = d
        if d < min_d:
            min_d = d
    var tol = max(Float32(1.5e-3), (max_d - min_d) * Float32(2.0e-3))

    var raw = List[Vec3]()
    for i in range(len(hull.vertices)):
        if max_d - hull.vertices[i].dot(local_dir) <= tol:
            var wp = pos + rot.rotate(hull.vertices[i])
            var dup = False
            for j in range(len(raw)):
                if (wp - raw[j]).length_squared() < Float32(1.0e-8):
                    dup = True
                    break
            if not dup:
                raw.append(wp)

    if len(raw) <= 2:
        return raw^

    # 2D Andrew's Monotone Chain convex hull in the tangent plane (t1, t2) oriented CCW around world_dir
    var basis = _plane_basis(world_dir)
    var t1 = basis[0]
    var t2 = basis[1]
    var n_pts = len(raw)
    var ux = List[Float32](capacity=n_pts)
    var uy = List[Float32](capacity=n_pts)
    var order = List[Int](capacity=n_pts)
    for i in range(n_pts):
        ux.append(raw[i].dot(t1))
        uy.append(raw[i].dot(t2))
        order.append(i)

    for i in range(1, n_pts):
        var key = order[i]
        var j = i - 1
        while j >= 0 and (
            ux[order[j]] > ux[key] or (ux[order[j]] == ux[key] and uy[order[j]] > uy[key])
        ):
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key

    var lower = List[Int]()
    for idx in range(n_pts):
        var p = order[idx]
        while len(lower) >= 2:
            var a = lower[len(lower) - 2]
            var b = lower[len(lower) - 1]
            var cross_z = (ux[b] - ux[a]) * (uy[p] - uy[a]) - (uy[b] - uy[a]) * (ux[p] - ux[a])
            if cross_z <= Float32(1.0e-7):
                _ = lower.pop()
            else:
                break
        lower.append(p)

    var upper = List[Int]()
    var r_idx = n_pts - 1
    while r_idx >= 0:
        var p = order[r_idx]
        while len(upper) >= 2:
            var a = upper[len(upper) - 2]
            var b = upper[len(upper) - 1]
            var cross_z = (ux[b] - ux[a]) * (uy[p] - uy[a]) - (uy[b] - uy[a]) * (ux[p] - ux[a])
            if cross_z <= Float32(1.0e-7):
                _ = upper.pop()
            else:
                break
        upper.append(p)
        r_idx -= 1

    var hull2d = List[Vec3]()
    if len(lower) + len(upper) - 2 < 3:
        # Collinear points: keep only endpoints
        hull2d.append(raw[order[0]])
        hull2d.append(raw[order[n_pts - 1]])
        return hull2d^

    for k in range(len(lower) - 1):
        hull2d.append(raw[lower[k]])
    for k in range(len(upper) - 1):
        hull2d.append(raw[upper[k]])
    return hull2d^


def collide_convex_convex(
    hull_a: ConvexHull, pos_a: Vec3, rot_a: Quat,
    hull_b: ConvexHull, pos_b: Vec3, rot_b: Quat,
    body_a: Int = 0, body_b: Int = 1,
) -> ContactManifold:
    var m = ContactManifold(body_a, body_b)
    if len(hull_a.vertices) == 0 or len(hull_b.vertices) == 0:
        return m^

    # 1. GJK Intersection test
    var dir = Vec3(0, 1, 0)
    var diff = pos_b - pos_a
    if diff.length_squared() > 1.0e-6:
        dir = diff.normalized()

    var a = _support_minkowski_pt(hull_a, pos_a, rot_a, hull_b, pos_b, rot_b, dir)
    if a.v.dot(dir) < 0.0:
        return m^

    var simplex = List[_MinkowskiPt]()
    simplex.append(a)
    dir = -a.v

    var enclosed = False
    for _ in range(48):
        if dir.length_squared() < 1.0e-10:
            dir = Vec3(0, 1, 0)
        var p = _support_minkowski_pt(hull_a, pos_a, rot_a, hull_b, pos_b, rot_b, dir)
        if p.v.dot(dir) < 0.0:
            return m^
        simplex.append(p)

        var slen = len(simplex)
        if slen == 2:
            var sa_pt = simplex[1]
            var sb_pt = simplex[0]
            var ab = sb_pt.v - sa_pt.v
            var ao = -sa_pt.v
            if ab.dot(ao) > 0.0:
                dir = ab.cross(ao).cross(ab)
                if dir.length_squared() < Float32(1.0e-10):
                    dir = _same_dir_perp(ab)
            else:
                simplex[0] = sa_pt
                _ = simplex.pop()
                dir = ao
        elif slen == 3:
            var sa_pt = simplex[2]
            var sb_pt = simplex[1]
            var sc_pt = simplex[0]
            var ab = sb_pt.v - sa_pt.v
            var ac = sc_pt.v - sa_pt.v
            var ao = -sa_pt.v
            var abc = ab.cross(ac)
            if abc.cross(ac).dot(ao) > 0.0:
                if ac.dot(ao) > 0.0:
                    simplex[0] = sc_pt
                    simplex[1] = sa_pt
                    _ = simplex.pop()
                    dir = ac.cross(ao).cross(ac)
                elif ab.dot(ao) > 0.0:
                    simplex[0] = sb_pt
                    simplex[1] = sa_pt
                    _ = simplex.pop()
                    dir = ab.cross(ao).cross(ab)
                else:
                    simplex[0] = sa_pt
                    _ = simplex.pop()
                    _ = simplex.pop()
                    dir = ao
            elif ab.cross(abc).dot(ao) > 0.0:
                if ab.dot(ao) > 0.0:
                    simplex[0] = sb_pt
                    simplex[1] = sa_pt
                    _ = simplex.pop()
                    dir = ab.cross(ao).cross(ab)
                else:
                    simplex[0] = sa_pt
                    _ = simplex.pop()
                    _ = simplex.pop()
                    dir = ao
            else:
                if abc.dot(ao) > 0.0:
                    dir = abc
                else:
                    simplex[0] = sb_pt
                    simplex[1] = sc_pt
                    dir = -abc
        elif slen == 4:
            var sa_pt = simplex[3]
            var sb_pt = simplex[2]
            var sc_pt = simplex[1]
            var sd_pt = simplex[0]
            var ab = sb_pt.v - sa_pt.v
            var ac = sc_pt.v - sa_pt.v
            var ad = sd_pt.v - sa_pt.v
            var ao = -sa_pt.v

            var n_abc = ab.cross(ac)
            if n_abc.dot(ad) > 0.0:
                n_abc = -n_abc
            var n_acd = ac.cross(ad)
            if n_acd.dot(ab) > 0.0:
                n_acd = -n_acd
            var n_adb = ad.cross(ab)
            if n_adb.dot(ac) > 0.0:
                n_adb = -n_adb

            if n_abc.dot(ao) > 0.0:
                simplex[0] = sc_pt
                simplex[1] = sb_pt
                simplex[2] = sa_pt
                _ = simplex.pop()
                dir = n_abc
            elif n_acd.dot(ao) > 0.0:
                simplex[0] = sd_pt
                simplex[1] = sc_pt
                simplex[2] = sa_pt
                _ = simplex.pop()
                dir = n_acd
            elif n_adb.dot(ao) > 0.0:
                simplex[0] = sb_pt
                simplex[1] = sd_pt
                simplex[2] = sa_pt
                _ = simplex.pop()
                dir = n_adb
            else:
                enclosed = True
                break

    if not enclosed:
        return m^

    # 2. EPA (Expanding Polytope Algorithm) for true penetration depth & separating normal
    var verts = List[_MinkowskiPt]()
    for i in range(4):
        verts.append(simplex[i])

    var interior = (verts[0].v + verts[1].v + verts[2].v + verts[3].v) * Float32(0.25)
    var faces = List[_EpaTri]()
    faces.append(_make_epa_tri(verts, 0, 1, 2, interior))
    faces.append(_make_epa_tri(verts, 0, 2, 3, interior))
    faces.append(_make_epa_tri(verts, 0, 3, 1, interior))
    faces.append(_make_epa_tri(verts, 1, 3, 2, interior))

    var best_face_idx = 0
    for _ in range(32):
        best_face_idx = -1
        var min_dist = Float32(1.0e30)
        for f_i in range(len(faces)):
            if faces[f_i].alive and faces[f_i].dist < min_dist:
                min_dist = faces[f_i].dist
                best_face_idx = f_i
        if best_face_idx < 0:
            break

        var best_n = faces[best_face_idx].n
        var best_d = faces[best_face_idx].dist
        var sp = _support_minkowski_pt(hull_a, pos_a, rot_a, hull_b, pos_b, rot_b, best_n)
        var d_new = best_n.dot(sp.v)
        if d_new - best_d < Float32(1.0e-4):
            break

        var edge_a = List[Int]()
        var edge_b = List[Int]()
        for f_i in range(len(faces)):
            if not faces[f_i].alive:
                continue
            var f = faces[f_i]
            if f.n.dot(sp.v - verts[f.i0].v) > Float32(1.0e-6):
                faces[f_i].alive = False
                var tri_edges = [(f.i0, f.i1), (f.i1, f.i2), (f.i2, f.i0)]
                for ei in range(3):
                    var ea = tri_edges[ei][0]
                    var eb = tri_edges[ei][1]
                    var rev_idx = -1
                    for k in range(len(edge_a)):
                        if edge_a[k] == eb and edge_b[k] == ea:
                            rev_idx = k
                            break
                    if rev_idx >= 0:
                        edge_a[rev_idx] = -1
                        edge_b[rev_idx] = -1
                    else:
                        edge_a.append(ea)
                        edge_b.append(eb)

        var new_v_idx = len(verts)
        verts.append(sp)
        var added = 0
        for k in range(len(edge_a)):
            if edge_a[k] >= 0:
                faces.append(_make_epa_tri(verts, edge_a[k], edge_b[k], new_v_idx, interior))
                added += 1
        if added == 0:
            break

    if best_face_idx < 0:
        return m^

    var bf = faces[best_face_idx]
    var normal = bf.n
    var depth = bf.dist
    var proj_p = normal * depth
    var bcoords = _barycentric_on_tri(proj_p, verts[bf.i0].v, verts[bf.i1].v, verts[bf.i2].v)
    var pt_a = verts[bf.i0].sa * bcoords[0] + verts[bf.i1].sa * bcoords[1] + verts[bf.i2].sa * bcoords[2]
    var pt_b = verts[bf.i0].sb * bcoords[0] + verts[bf.i1].sb * bcoords[1] + verts[bf.i2].sb * bcoords[2]
    var epa_pt = (pt_a + pt_b) * Float32(0.5)

    # 3. Reference / Incident Feature Identification + Sutherland-Hodgman Clipping + 4-Point Reduction
    var feat_a = _support_feature_polygon(hull_a, pos_a, rot_a, normal)
    var feat_b = _support_feature_polygon(hull_b, pos_b, rot_b, -normal)

    var na = len(feat_a)
    var nb = len(feat_b)

    # Polygon face (>= 3) vs Polygon face (>= 3) or Edge (2)
    if (na >= 3 and nb >= 2) or (nb >= 3 and na >= 2):
        var ref_is_a = True
        if na < 3:
            ref_is_a = False
        elif nb >= 3:
            var na_plane = (feat_a[1] - feat_a[0]).cross(feat_a[2] - feat_a[0]).normalized()
            var nb_plane = (feat_b[1] - feat_b[0]).cross(feat_b[2] - feat_b[0]).normalized()
            if abs(nb_plane.dot(normal)) > abs(na_plane.dot(normal)) + Float32(1.0e-3):
                ref_is_a = False

        ref ref_poly = feat_a if ref_is_a else feat_b
        ref inc_src = feat_b if ref_is_a else feat_a
        var ref_n = normal if ref_is_a else -normal
        # Compute exact geometric normal of reference polygon oriented along ref_n
        var rp0 = ref_poly[0]
        var rp1 = ref_poly[1]
        var rp2 = ref_poly[2]
        var geom_n = (rp1 - rp0).cross(rp2 - rp0).normalized()
        if geom_n.dot(ref_n) < Float32(0.0):
            geom_n = -geom_n
        if geom_n.length_squared() < Float32(0.5):
            geom_n = ref_n

        var ref_origin = ref_poly[0]
        var clipped = List[_TopoClipVert](capacity=len(inc_src))
        for i in range(len(inc_src)):
            var prev_e = UInt8((i + len(inc_src) - 1) % len(inc_src) + 1)
            var next_e = UInt8(i + 1)
            clipped.append(_TopoClipVert(inc_src[i], prev_e, next_e))

        var ref_len = len(ref_poly)
        for e_i in range(ref_len):
            var va = ref_poly[e_i]
            var vb = ref_poly[(e_i + 1) % ref_len]
            # Outward edge plane normal for CCW polygon around ref_n: (vb - va) x geom_n
            var side_n = (vb - va).cross(geom_n)
            var sl2 = side_n.length_squared()
            if sl2 > Float32(1.0e-10):
                side_n = side_n * (Float32(1.0) / sqrt(sl2))
                clipped = _clip_poly_topo(clipped, va, side_n, UInt8(e_i + 32))
                if len(clipped) == 0:
                    break

        var cands = List[ContactPoint](capacity=len(clipped))
        var ref_face_tag = UInt32(1 if ref_is_a else 2)
        for k in range(len(clipped)):
            var cv = clipped[k]
            var pen = -(cv.p - ref_origin).dot(geom_n)
            if pen > Float32(-1.0e-4):
                var d_pt = max(Float32(0.0), pen)
                var c_pt = cv.p + geom_n * (d_pt * Float32(0.5))
                var fid = (ref_face_tag << 16) | (UInt32(cv.edge_in) << 8) | UInt32(cv.edge_out)
                cands.append(ContactPoint(c_pt, normal, d_pt, fid))

        reduce_manifold_candidates(m, cands)
        if m.count > 0:
            return m^
    elif na == 2 and nb == 2:
        # Edge vs Edge: check parallel edges for 2-point segment clip, else closest points
        var ea = feat_a[1] - feat_a[0]
        var eb = feat_b[1] - feat_b[0]
        var la = ea.length()
        var lb = eb.length()
        if la > Float32(1.0e-5) and lb > Float32(1.0e-5):
            var ua = ea * (Float32(1.0) / la)
            var ub = eb * (Float32(1.0) / lb)
            if abs(ua.dot(ub)) > Float32(0.995):
                var s0 = (feat_b[0] - feat_a[0]).dot(ua)
                var s1 = (feat_b[1] - feat_a[0]).dot(ua)
                var slo = max(Float32(0.0), min(s0, s1))
                var shi = min(la, max(s0, s1))
                if shi - slo > Float32(1.0e-4):
                    var p_lo = feat_a[0] + ua * slo - normal * (depth * Float32(0.5))
                    var p_hi = feat_a[0] + ua * shi - normal * (depth * Float32(0.5))
                    m.add_point(p_lo, normal, depth, 1)
                    m.add_point(p_hi, normal, depth, 2)
                    return m^
            var cps = closest_points_segments(feat_a[0], feat_a[1], feat_b[0], feat_b[1])
            m.add_point((cps[0] + cps[1]) * Float32(0.5), normal, depth, 3)
            return m^

    m.add_point(epa_pt, normal, depth, 0)
    return m^
