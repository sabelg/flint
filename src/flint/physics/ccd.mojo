"""Continuous Collision Detection (CCD) primitives for high-speed projectiles & OBBs.

Provides exact time-of-impact (`toi` in `[0, 1]`), world contact point, and unit
surface normal for:
  - Swept sphere vs. moving sphere (relative quadratic ray-sphere)
  - Swept sphere vs. Oriented Bounding Box (OBB-local inflated Slab ray-box)
  - Closest point & signed distance to an OBB surface (for deck grounding & bounces)
"""
from std.math import sqrt
from flint.core.math import Vec3
from flint.core.transform import Quat


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


def swept_sphere_obb(
    p0: Vec3,
    p1: Vec3,
    r_proj: Float32,
    obb_center: Vec3,
    obb_rot: Quat,
    obb_half: Vec3,
) -> CcdHit:
    """Tests a sphere of radius `r_proj` sweeping `p0 -> p1` against an Oriented Bounding Box."""
    var inv_q = obb_rot.conjugate()
    var lp0 = inv_q.rotate(p0 - obb_center)
    var lp1 = inv_q.rotate(p1 - obb_center)
    var ext = obb_half + Vec3(r_proj, r_proj, r_proj)

    if abs(lp0.x) <= ext.x and abs(lp0.y) <= ext.y and abs(lp0.z) <= ext.z:
        var dx = ext.x - abs(lp0.x)
        var dy = ext.y - abs(lp0.y)
        var dz = ext.z - abs(lp0.z)
        var ln = Vec3(Float32(1.0) if lp0.x >= Float32(0.0) else Float32(-1.0), 0.0, 0.0)
        if dy <= dx and dy <= dz:
            ln = Vec3(0.0, Float32(1.0) if lp0.y >= Float32(0.0) else Float32(-1.0), 0.0)
        elif dz <= dx and dz <= dy:
            ln = Vec3(0.0, 0.0, Float32(1.0) if lp0.z >= Float32(0.0) else Float32(-1.0))
        var wn0 = _safe_norm(obb_rot.rotate(ln))
        return CcdHit(True, Float32(0.0), p0 - wn0 * r_proj, wn0)

    var ld = lp1 - lp0
    var t_min = Float32(0.0)
    var t_max = Float32(1.0)
    var hit_ln = Vec3(0.0, 1.0, 0.0)

    # X slab
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
        if tx1 > t_min:
            t_min = tx1
            hit_ln = Vec3(Float32(-1.0) if ld.x > Float32(0.0) else Float32(1.0), 0.0, 0.0)
        if tx2 < t_max:
            t_max = tx2
        if t_min > t_max:
            return CcdHit()

    # Y slab
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
        if ty1 > t_min:
            t_min = ty1
            hit_ln = Vec3(0.0, Float32(-1.0) if ld.y > Float32(0.0) else Float32(1.0), 0.0)
        if ty2 < t_max:
            t_max = ty2
        if t_min > t_max:
            return CcdHit()

    # Z slab
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
        if tz1 > t_min:
            t_min = tz1
            hit_ln = Vec3(0.0, 0.0, Float32(-1.0) if ld.z > Float32(0.0) else Float32(1.0))
        if tz2 < t_max:
            t_max = tz2
        if t_min > t_max:
            return CcdHit()

    var wn = _safe_norm(obb_rot.rotate(hit_ln))
    var p_contact = p0 + (p1 - p0) * t_min - wn * r_proj
    return CcdHit(True, t_min, p_contact, wn)


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
    var surf_l = lp
    var ln = Vec3(0.0, 1.0, 0.0)
    var min_d = dy
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
