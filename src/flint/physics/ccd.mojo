"""Continuous Collision Detection (CCD) primitives for high-speed projectiles & OBBs.

Provides exact time-of-impact (`toi` in `[0, 1]`), world contact point, and unit
surface normal for:
  - Swept sphere vs. moving sphere (relative quadratic ray-sphere)
  - Swept sphere vs. Oriented Bounding Box (OBB-local inflated Slab ray-box)
  - Closest point & signed distance to an OBB surface (for deck grounding & bounces)
"""
from std.collections import List
from std.math import sqrt, min, max, abs
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.physics.narrowphase import ConvexHull, collide_convex_convex


@fieldwise_init
struct CcdHit(Copyable, ImplicitlyCopyable, Movable):
    var hit: Bool
    var toi: Float32
    var point: Vec3
    var normal: Vec3

    def __init__(out self):
        self.hit = False
        self.toi = Float32(1.0)
        self.point = Vec3()
        self.normal = Vec3(0.0, 1.0, 0.0)


@always_inline
def _safe_norm(v: Vec3, fallback: Vec3 = Vec3(0.0, 1.0, 0.0)) -> Vec3:
    var l2 = v.length_squared()
    if l2 < Float32(1.0e-10):
        return fallback
    return v * (Float32(1.0) / sqrt(l2))


def swept_sphere_moving_sphere(
    p0: Vec3,
    p1: Vec3,
    r_proj: Float32,
    c0: Vec3,
    c1: Vec3,
    r_target: Float32,
) -> CcdHit:
    """Tests a sphere of radius `r_proj` sweeping `p0 -> p1` against a sphere `r_target` moving `c0 -> c1`."""
    var s = p0 - c0
    var rad = r_proj + r_target
    var rad2 = rad * rad
    var s2 = s.length_squared()
    if s2 <= rad2:
        var n0 = _safe_norm(s)
        return CcdHit(True, Float32(0.0), c0 + n0 * r_target, n0)

    var d = (p1 - p0) - (c1 - c0)
    var a = d.length_squared()
    if a < Float32(1.0e-8):
        return CcdHit()

    var b = s.dot(d)
    if b >= Float32(0.0):
        return CcdHit()

    var c = s2 - rad2
    var disc = b * b - a * c
    if disc < Float32(0.0):
        return CcdHit()

    var t = (-b - sqrt(disc)) / a
    if t < Float32(0.0) or t > Float32(1.0):
        return CcdHit()

    var p_hit = p0 + (p1 - p0) * t
    var c_hit = c0 + (c1 - c0) * t
    var n = _safe_norm(p_hit - c_hit)
    return CcdHit(True, t, c_hit + n * r_target, n)


@always_inline
def _ray_sphere(p: Vec3, d: Vec3, c: Vec3, r: Float32) -> Float32:
    var oc = p - c
    var a = d.length_squared()
    if a < Float32(1.0e-12):
        return Float32(-1.0)
    var b = oc.dot(d)
    var cc = oc.length_squared() - r * r
    var disc = b * b - a * cc
    if disc < Float32(0.0):
        return Float32(-1.0)
    var s = sqrt(disc)
    var t0 = (-b - s) / a
    if t0 >= Float32(0.0):
        return t0
    var t1 = (-b + s) / a
    if t1 >= Float32(0.0):
        return t1
    return Float32(-1.0)


@always_inline
def _ray_cylinder_2d(p0_u: Float32, p0_v: Float32, d_u: Float32, d_v: Float32, r: Float32) -> Float32:
    var a = d_u * d_u + d_v * d_v
    if a < Float32(1.0e-12):
        return Float32(-1.0)
    var b = p0_u * d_u + p0_v * d_v
    var c = p0_u * p0_u + p0_v * p0_v - r * r
    var disc = b * b - a * c
    if disc < Float32(0.0):
        return Float32(-1.0)
    var s = sqrt(disc)
    var t0 = (-b - s) / a
    if t0 >= Float32(0.0):
        return t0
    var t1 = (-b + s) / a
    if t1 >= Float32(0.0):
        return t1
    return Float32(-1.0)


def swept_sphere_obb(
    p0: Vec3,
    p1: Vec3,
    r_proj: Float32,
    obb_center: Vec3,
    obb_rot: Quat,
    obb_half: Vec3,
) -> CcdHit:
    """Tests a sphere of radius `r_proj` sweeping `p0 -> p1` against an Oriented Bounding Box
    using exact rounded Minkowski sum (faces, cylindrical edge capsules, and corner spheres).
    """
    var inv_q = obb_rot.conjugate()
    var lp0 = inv_q.rotate(p0 - obb_center)
    var lp1 = inv_q.rotate(p1 - obb_center)

    # Initial check: sphere already inside or penetrating Minkowski sum at t = 0
    var clamped_p0 = Vec3(
        max(-obb_half.x, min(obb_half.x, lp0.x)),
        max(-obb_half.y, min(obb_half.y, lp0.y)),
        max(-obb_half.z, min(obb_half.z, lp0.z)),
    )
    var diff_p0 = lp0 - clamped_p0
    var d0_sq = diff_p0.length_squared()
    if d0_sq <= r_proj * r_proj:
        var ln = _safe_norm(diff_p0)
        if d0_sq < Float32(1.0e-8):
            var dx = obb_half.x - abs(lp0.x)
            var dy = obb_half.y - abs(lp0.y)
            var dz = obb_half.z - abs(lp0.z)
            if dx <= dy and dx <= dz:
                ln = Vec3(Float32(1.0) if lp0.x >= Float32(0.0) else Float32(-1.0), 0.0, 0.0)
            elif dy <= dx and dy <= dz:
                ln = Vec3(0.0, Float32(1.0) if lp0.y >= Float32(0.0) else Float32(-1.0), 0.0)
            else:
                ln = Vec3(0.0, 0.0, Float32(1.0) if lp0.z >= Float32(0.0) else Float32(-1.0))
        var wn0 = _safe_norm(obb_rot.rotate(ln))
        return CcdHit(True, Float32(0.0), p0 - wn0 * r_proj, wn0)

    var ext = obb_half + Vec3(r_proj, r_proj, r_proj)
    var ld = lp1 - lp0

    # Quick slab early-out against bounding box of Minkowski sum
    var t_box_min = Float32(0.0)
    var t_box_max = Float32(1.0)
    if abs(ld.x) < Float32(1.0e-7):
        if abs(lp0.x) > ext.x:
            return CcdHit()
    else:
        var inv_x = Float32(1.0) / ld.x
        var tx1 = (-ext.x - lp0.x) * inv_x
        var tx2 = (ext.x - lp0.x) * inv_x
        if tx1 > tx2:
            var tmp = tx1
            tx1 = tx2
            tx2 = tmp
        if tx1 > t_box_min:
            t_box_min = tx1
        if tx2 < t_box_max:
            t_box_max = tx2
        if t_box_min > t_box_max:
            return CcdHit()

    if abs(ld.y) < Float32(1.0e-7):
        if abs(lp0.y) > ext.y:
            return CcdHit()
    else:
        var inv_y = Float32(1.0) / ld.y
        var ty1 = (-ext.y - lp0.y) * inv_y
        var ty2 = (ext.y - lp0.y) * inv_y
        if ty1 > ty2:
            var tmp = ty1
            ty1 = ty2
            ty2 = tmp
        if ty1 > t_box_min:
            t_box_min = ty1
        if ty2 < t_box_max:
            t_box_max = ty2
        if t_box_min > t_box_max:
            return CcdHit()

    if abs(ld.z) < Float32(1.0e-7):
        if abs(lp0.z) > ext.z:
            return CcdHit()
    else:
        var inv_z = Float32(1.0) / ld.z
        var tz1 = (-ext.z - lp0.z) * inv_z
        var tz2 = (ext.z - lp0.z) * inv_z
        if tz1 > tz2:
            var tmp = tz1
            tz1 = tz2
            tz2 = tmp
        if tz1 > t_box_min:
            t_box_min = tz1
        if tz2 < t_box_max:
            t_box_max = tz2
        if t_box_min > t_box_max:
            return CcdHit()

    # Exact Minkowski feature tests: 6 planar faces, 12 cylindrical capsules, 8 corner spheres
    var best_t = Float32(1.0)
    var hit_found = False

    # 1. Planar faces
    if abs(ld.x) >= Float32(1.0e-7):
        var t_px = (obb_half.x + r_proj - lp0.x) / ld.x
        if t_px >= Float32(0.0) and t_px <= best_t:
            var hit = lp0 + ld * t_px
            if abs(hit.y) <= obb_half.y and abs(hit.z) <= obb_half.z:
                best_t = t_px
                hit_found = True
        var t_nx = (-obb_half.x - r_proj - lp0.x) / ld.x
        if t_nx >= Float32(0.0) and t_nx <= best_t:
            var hit = lp0 + ld * t_nx
            if abs(hit.y) <= obb_half.y and abs(hit.z) <= obb_half.z:
                best_t = t_nx
                hit_found = True

    if abs(ld.y) >= Float32(1.0e-7):
        var t_py = (obb_half.y + r_proj - lp0.y) / ld.y
        if t_py >= Float32(0.0) and t_py <= best_t:
            var hit = lp0 + ld * t_py
            if abs(hit.x) <= obb_half.x and abs(hit.z) <= obb_half.z:
                best_t = t_py
                hit_found = True
        var t_ny = (-obb_half.y - r_proj - lp0.y) / ld.y
        if t_ny >= Float32(0.0) and t_ny <= best_t:
            var hit = lp0 + ld * t_ny
            if abs(hit.x) <= obb_half.x and abs(hit.z) <= obb_half.z:
                best_t = t_ny
                hit_found = True

    if abs(ld.z) >= Float32(1.0e-7):
        var t_pz = (obb_half.z + r_proj - lp0.z) / ld.z
        if t_pz >= Float32(0.0) and t_pz <= best_t:
            var hit = lp0 + ld * t_pz
            if abs(hit.x) <= obb_half.x and abs(hit.y) <= obb_half.y:
                best_t = t_pz
                hit_found = True
        var t_nz = (-obb_half.z - r_proj - lp0.z) / ld.z
        if t_nz >= Float32(0.0) and t_nz <= best_t:
            var hit = lp0 + ld * t_nz
            if abs(hit.x) <= obb_half.x and abs(hit.y) <= obb_half.y:
                best_t = t_nz
                hit_found = True

    # 2. 12 Cylindrical edge capsules
    # 4 Edges along X
    for ey_i in range(2):
        var ey = -obb_half.y if ey_i == 0 else obb_half.y
        for ez_i in range(2):
            var ez = -obb_half.z if ez_i == 0 else obb_half.z
            var t_edge = _ray_cylinder_2d(lp0.y - ey, lp0.z - ez, ld.y, ld.z, r_proj)
            if t_edge >= Float32(0.0) and t_edge <= best_t:
                var hx = lp0.x + ld.x * t_edge
                if abs(hx) <= obb_half.x:
                    best_t = t_edge
                    hit_found = True

    # 4 Edges along Y
    for ex_i in range(2):
        var ex = -obb_half.x if ex_i == 0 else obb_half.x
        for ez_i in range(2):
            var ez = -obb_half.z if ez_i == 0 else obb_half.z
            var t_edge = _ray_cylinder_2d(lp0.x - ex, lp0.z - ez, ld.x, ld.z, r_proj)
            if t_edge >= Float32(0.0) and t_edge <= best_t:
                var hy = lp0.y + ld.y * t_edge
                if abs(hy) <= obb_half.y:
                    best_t = t_edge
                    hit_found = True

    # 4 Edges along Z
    for ex_i in range(2):
        var ex = -obb_half.x if ex_i == 0 else obb_half.x
        for ey_i in range(2):
            var ey = -obb_half.y if ey_i == 0 else obb_half.y
            var t_edge = _ray_cylinder_2d(lp0.x - ex, lp0.y - ey, ld.x, ld.y, r_proj)
            if t_edge >= Float32(0.0) and t_edge <= best_t:
                var hz = lp0.z + ld.z * t_edge
                if abs(hz) <= obb_half.z:
                    best_t = t_edge
                    hit_found = True

    # 3. 8 Corner spheres
    for ex_i in range(2):
        var ex = -obb_half.x if ex_i == 0 else obb_half.x
        for ey_i in range(2):
            var ey = -obb_half.y if ey_i == 0 else obb_half.y
            for ez_i in range(2):
                var ez = -obb_half.z if ez_i == 0 else obb_half.z
                var corner = Vec3(ex, ey, ez)
                var t_c = _ray_sphere(lp0, ld, corner, r_proj)
                if t_c >= Float32(0.0) and t_c <= best_t:
                    best_t = t_c
                    hit_found = True

    if not hit_found:
        return CcdHit()

    var hit_local = lp0 + ld * best_t
    var clamped = Vec3(
        max(-obb_half.x, min(obb_half.x, hit_local.x)),
        max(-obb_half.y, min(obb_half.y, hit_local.y)),
        max(-obb_half.z, min(obb_half.z, hit_local.z)),
    )
    var normal_local = _safe_norm(hit_local - clamped)
    var wn = _safe_norm(obb_rot.rotate(normal_local))
    var p_contact = p0 + (p1 - p0) * best_t - wn * r_proj
    return CcdHit(True, best_t, p_contact, wn)


def closest_obb_surface(
    p: Vec3,
    obb_center: Vec3,
    obb_rot: Quat,
    obb_half: Vec3,
) -> Tuple[Vec3, Vec3, Float32]:
    """Returns `(closest_world_pt, outward_world_normal, signed_distance)` for `p` against an OBB."""
    var lp = obb_rot.conjugate().rotate(p - obb_center)
    var cx = max(-obb_half.x, min(obb_half.x, lp.x))
    var cy = max(-obb_half.y, min(obb_half.y, lp.y))
    var cz = max(-obb_half.z, min(obb_half.z, lp.z))
    var clamped = Vec3(cx, cy, cz)
    var diff = lp - clamped
    var d2 = diff.length_squared()
    if d2 > Float32(1.0e-8):
        var dist = sqrt(d2)
        var wn = _safe_norm(obb_rot.rotate(diff * (Float32(1.0) / dist)))
        return (obb_center + obb_rot.rotate(clamped), wn, dist)

    # Inside OBB: push to nearest face
    var dx = obb_half.x - abs(lp.x)
    var dy = obb_half.y - abs(lp.y)
    var dz = obb_half.z - abs(lp.z)
    var surf_l: Vec3
    var ln: Vec3
    var min_d: Float32
    if dx <= dy and dx <= dz:
        min_d = dx
        var sx = Float32(1.0) if lp.x >= Float32(0.0) else Float32(-1.0)
        surf_l = Vec3(sx * obb_half.x, lp.y, lp.z)
        ln = Vec3(sx, 0.0, 0.0)
    elif dy <= dx and dy <= dz:
        min_d = dy
        var sy = Float32(1.0) if lp.y >= Float32(0.0) else Float32(-1.0)
        surf_l = Vec3(lp.x, sy * obb_half.y, lp.z)
        ln = Vec3(0.0, sy, 0.0)
    else:
        min_d = dz
        var sz = Float32(1.0) if lp.z >= Float32(0.0) else Float32(-1.0)
        surf_l = Vec3(lp.x, lp.y, sz * obb_half.z)
        ln = Vec3(0.0, 0.0, sz)
    return (obb_center + obb_rot.rotate(surf_l), _safe_norm(obb_rot.rotate(ln)), -min_d)


@fieldwise_init
struct _GjkVert(Copyable, ImplicitlyCopyable, Movable):
    var v: Vec3   # sb - sa (point in B - A)
    var sa: Vec3
    var sb: Vec3


def _closest_origin_tri(
    a: _GjkVert, b: _GjkVert, c: _GjkVert,
) -> Tuple[Vec3, Vec3, Vec3, Int, _GjkVert, _GjkVert, _GjkVert]:
    """Ericson's exact Voronoi closest point on triangle (a, b, c) to origin (0, 0, 0).
    Returns (v, sa, sb, kept_count, k0, k1, k2).
    """
    var ab = b.v - a.v
    var ac = c.v - a.v
    var ap = -a.v

    var d1 = ab.dot(ap)
    var d2 = ac.dot(ap)
    if d1 <= Float32(0.0) and d2 <= Float32(0.0):
        return (a.v, a.sa, a.sb, 1, a, a, a)

    var bp = -b.v
    var d3 = ab.dot(bp)
    var d4 = ac.dot(bp)
    if d3 >= Float32(0.0) and d4 <= d3:
        return (b.v, b.sa, b.sb, 1, b, b, b)

    var vc = d1 * d4 - d3 * d2
    if vc <= Float32(0.0) and d1 >= Float32(0.0) and d3 <= Float32(0.0):
        var w_ab = d1 / (d1 - d3)
        var om = Float32(1.0) - w_ab
        return (a.v * om + b.v * w_ab, a.sa * om + b.sa * w_ab, a.sb * om + b.sb * w_ab, 2, a, b, a)

    var cp = -c.v
    var d5 = ab.dot(cp)
    var d6 = ac.dot(cp)
    if d6 >= Float32(0.0) and d5 <= d6:
        return (c.v, c.sa, c.sb, 1, c, c, c)

    var vb = d5 * d2 - d1 * d6
    if vb <= Float32(0.0) and d2 >= Float32(0.0) and d6 <= Float32(0.0):
        var w_ac = d2 / (d2 - d6)
        var om = Float32(1.0) - w_ac
        return (a.v * om + c.v * w_ac, a.sa * om + c.sa * w_ac, a.sb * om + c.sb * w_ac, 2, a, c, a)

    var va = d3 * d6 - d5 * d4
    if va <= Float32(0.0) and (d4 - d3) >= Float32(0.0) and (d5 - d6) >= Float32(0.0):
        var w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        var om = Float32(1.0) - w_bc
        return (b.v * om + c.v * w_bc, b.sa * om + c.sa * w_bc, b.sb * om + c.sb * w_bc, 2, b, c, b)

    var denom = Float32(1.0) / max(Float32(1.0e-12), va + vb + vc)
    var wb = vb * denom
    var wc = vc * denom
    var wa = Float32(1.0) - wb - wc
    return (
        a.v * wa + b.v * wb + c.v * wc,
        a.sa * wa + b.sa * wb + c.sa * wc,
        a.sb * wa + b.sb * wb + c.sb * wc,
        3, a, b, c,
    )


def _reduce_gjk_simplex(mut s: List[_GjkVert]) -> Tuple[Vec3, Vec3, Vec3, Bool]:
    """Reduces simplex `s` (1..4 pts in B - A) to minimal Voronoi feature closest to origin.
    Returns (v, ca, cb, enclosed).
    """
    var n = len(s)
    if n == 1:
        return (s[0].v, s[0].sa, s[0].sb, False)
    if n == 2:
        var a = s[0]
        var b = s[1]
        var ab = b.v - a.v
        var l2 = ab.length_squared()
        if l2 < Float32(1.0e-12):
            _ = s.pop()
            return (a.v, a.sa, a.sb, False)
        var t = (-a.v).dot(ab) / l2
        if t <= Float32(0.0):
            s.clear()
            s.append(a)
            return (a.v, a.sa, a.sb, False)
        if t >= Float32(1.0):
            s.clear()
            s.append(b)
            return (b.v, b.sa, b.sb, False)
        var om = Float32(1.0) - t
        return (a.v * om + b.v * t, a.sa * om + b.sa * t, a.sb * om + b.sb * t, False)
    if n == 3:
        var r = _closest_origin_tri(s[0], s[1], s[2])
        s.clear()
        s.append(r[4])
        if r[3] >= 2:
            s.append(r[5])
        if r[3] >= 3:
            s.append(r[6])
        return (r[0], r[1], r[2], False)

    # n == 4: check if origin is strictly inside tetrahedron (s[0], s[1], s[2], s[3])
    var v0 = s[0].v; var v1 = s[1].v; var v2 = s[2].v; var v3 = s[3].v
    var faces = [
        (0, 1, 2, 3),
        (0, 2, 3, 1),
        (0, 3, 1, 2),
        (1, 3, 2, 0),
    ]
    var outside_any = False
    var best_d2 = Float32(1.0e38)
    var best_v = v0
    var best_sa = s[0].sa
    var best_sb = s[0].sb
    var best_cnt = 1
    var bk0 = s[0]; var bk1 = s[0]; var bk2 = s[0]
    var pts = [v0, v1, v2, v3]

    for fi in range(4):
        var i0 = faces[fi][0]; var i1 = faces[fi][1]; var i2 = faces[fi][2]; var i_opp = faces[fi][3]
        var face_n = (pts[i1] - pts[i0]).cross(pts[i2] - pts[i0])
        if face_n.dot(pts[i_opp] - pts[i0]) > Float32(0.0):
            face_n = -face_n
        # Origin relative to face: (-pts[i0]).dot(face_n)
        if (-pts[i0]).dot(face_n) >= Float32(-1.0e-7):
            outside_any = True
            var tr = _closest_origin_tri(s[i0], s[i1], s[i2])
            var d2 = tr[0].length_squared()
            if d2 < best_d2:
                best_d2 = d2
                best_v = tr[0]
                best_sa = tr[1]
                best_sb = tr[2]
                best_cnt = tr[3]
                bk0 = tr[4]; bk1 = tr[5]; bk2 = tr[6]

    if not outside_any:
        return (Vec3(), s[3].sa, s[3].sb, True)

    s.clear()
    s.append(bk0)
    if best_cnt >= 2:
        s.append(bk1)
    if best_cnt >= 3:
        s.append(bk2)
    return (best_v, best_sa, best_sb, False)


def gjk_closest_points(
    hull_a: ConvexHull, pos_a: Vec3, rot_a: Quat,
    hull_b: ConvexHull, pos_b: Vec3, rot_b: Quat,
) -> Tuple[Bool, Vec3, Vec3, Float32]:
    """Computes closest points `(overlapping, pt_a, pt_b, distance)` between two ConvexHulls."""
    if len(hull_a.vertices) == 0 or len(hull_b.vertices) == 0:
        return (False, pos_a, pos_b, Float32(1.0e30))

    var dir = pos_b - pos_a
    if dir.length_squared() < Float32(1.0e-8):
        dir = Vec3(1.0, 0.0, 0.0)
    var sa0 = hull_a.support(dir, pos_a, rot_a)
    var sb0 = hull_b.support(-dir, pos_b, rot_b)
    var simplex = List[_GjkVert](capacity=4)
    simplex.append(_GjkVert(sb0 - sa0, sa0, sb0))

    var v = simplex[0].v
    var ca = sa0
    var cb = sb0

    for _ in range(32):
        var d2 = v.length_squared()
        if d2 <= Float32(1.0e-10):
            return (True, ca, cb, Float32(0.0))

        # Support of C = B - A in direction -v (toward origin): max_{b in B, a in A} (b - a) . (-v)
        # = support_B(-v) - support_A(+v)
        var sa = hull_a.support(v, pos_a, rot_a)
        var sb = hull_b.support(-v, pos_b, rot_b)
        var w = sb - sa

        # Optimality gap: v . (v - w) <= tol
        var gap = d2 - v.dot(w)
        if gap <= Float32(1.0e-6) * max(Float32(1.0), d2):
            return (False, ca, cb, sqrt(d2))

        var duplicate = False
        for k in range(len(simplex)):
            if (simplex[k].v - w).length_squared() < Float32(1.0e-10):
                duplicate = True
                break
        if duplicate:
            return (False, ca, cb, sqrt(d2))

        simplex.append(_GjkVert(w, sa, sb))
        var red = _reduce_gjk_simplex(simplex)
        if red[3]:
            return (True, red[1], red[2], Float32(0.0))
        v = red[0]
        ca = red[1]
        cb = red[2]

    var dist = v.length()
    return (dist <= Float32(1.0e-5), ca, cb, dist)


def swept_convex_cast(
    hull_a: ConvexHull,
    pos_a0: Vec3,
    pos_a1: Vec3,
    rot_a: Quat,
    hull_b: ConvexHull,
    pos_b0: Vec3,
    pos_b1: Vec3,
    rot_b: Quat,
) -> CcdHit:
    """Linear GJK Raycast / Conservative Advancement for moving ConvexHull `hull_a` (`pos_a0 -> pos_a1`)
    against moving or static ConvexHull `hull_b` (`pos_b0 -> pos_b1`).
    Returns `CcdHit` with `toi` in `[0, 1]`, world contact point, and outward normal from `hull_b` toward `hull_a`.
    """
    if len(hull_a.vertices) == 0 or len(hull_b.vertices) == 0:
        return CcdHit()

    # 1. Check initial overlap at t = 0
    var m0 = collide_convex_convex(hull_a, pos_a0, rot_a, hull_b, pos_b0, rot_b)
    if m0.has_collision:
        return CcdHit(True, Float32(0.0), m0.p0.point, _safe_norm(-m0.p0.normal))

    var dpa = pos_a1 - pos_a0
    var dpb = pos_b1 - pos_b0
    var rel_move = dpa - dpb
    if rel_move.length_squared() < Float32(1.0e-10):
        return CcdHit()

    var lam = Float32(0.0)
    var hit_normal = _safe_norm(-rel_move)
    var hit_point = pos_a0

    for _ in range(32):
        var cur_pa = pos_a0 + dpa * lam
        var cur_pb = pos_b0 + dpb * lam
        var gjk = gjk_closest_points(hull_a, cur_pa, rot_a, hull_b, cur_pb, rot_b)
        var overlapping = gjk[0]
        var ca = gjk[1]
        var cb = gjk[2]
        var dist = gjk[3]

        if overlapping or dist <= Float32(2.0e-4):
            hit_point = (ca + cb) * Float32(0.5)
            return CcdHit(True, lam, hit_point, hit_normal)

        var n_ab = (cb - ca) * (Float32(1.0) / dist)  # unit vector from A(t) to B(t)
        var v_close = rel_move.dot(n_ab)
        if v_close <= Float32(1.0e-7):
            return CcdHit()  # Separating or parallel to separating plane

        var step = dist / v_close
        lam += step
        if lam > Float32(1.0):
            return CcdHit()

        hit_normal = -n_ab  # Outward from target B toward projectile A
        hit_point = ca + dpa * step

    return CcdHit()
