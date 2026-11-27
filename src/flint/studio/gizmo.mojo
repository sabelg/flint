"""Flint Studio Analytical 3D Viewport Gizmos & GPU `UiDrawList` Renderer (`gizmo.mojo`).

Implements:
  - Constant-screen-pixel-size `gizmo_world_scale` (invariant across camera distances `1m .. 1000m`)
  - `ViewportRay`, `ray_axis_closest_param` (skew-line closest approach), and `ray_plane_hit`
  - `gizmo_rotation_delta_rad` with automatic `< 0.15` grazing-angle (`~81.4°..90°`, e.g. `88°`)
    view-plane tangent fallback so edge-on rotation rings never blow up or flip!
  - Grid & angular snapping (`snap_translation`, `snap_angle`, `snap_angle_deg`)
  - Screen-space handle hit-testing (`pick_gizmo_handle`) & crisp GPU `UiDrawList` rendering
    (`emit_gizmo_draw_list`) of X/Y/Z arrows, XY/XZ/YZ dual-axis planes, rotation rings & scale cubes.
"""
from std.collections import List
from std.math import abs, atan2, cos, floor, isfinite, max, min, sin, sqrt, tan
from flint.core.math import Quat, Vec3
from flint.ui.draw_list import UiDrawList
from flint.ui.theme import lerp_rgba8, rgba8_scale_alpha, rgba8_with_alpha

comptime GIZMO_MODE_SELECT: Int = -1
comptime GIZMO_MODE_TRANSLATE: Int = 0
comptime GIZMO_MODE_ROTATE: Int = 1
comptime GIZMO_MODE_SCALE: Int = 2

comptime GIZMO_SPACE_WORLD: Int = 0
comptime GIZMO_SPACE_LOCAL: Int = 1

comptime GIZMO_AXIS_NONE: Int = 0
comptime GIZMO_AXIS_X: Int = 1
comptime GIZMO_AXIS_Y: Int = 2
comptime GIZMO_AXIS_Z: Int = 3
comptime GIZMO_PLANE_XY: Int = 4
comptime GIZMO_PLANE_XZ: Int = 5
comptime GIZMO_PLANE_YZ: Int = 6
comptime GIZMO_SCALE_UNIFORM: Int = 7

# Flint Obsidian harmonized default axis palette (overridable from StudioTheme)
comptime GIZMO_COLOR_X: UInt32 = 0xFFE85A66
comptime GIZMO_COLOR_Y: UInt32 = 0xFF4FC478
comptime GIZMO_COLOR_Z: UInt32 = 0xFF4C96F0
comptime GIZMO_COLOR_ACTIVE: UInt32 = 0xFFF5B74F
comptime GIZMO_COLOR_CENTER: UInt32 = 0xFFEEF2F7
comptime GIZMO_COLOR_HALO: UInt32 = 0x74080B10
comptime GIZMO_COLOR_CORE: UInt32 = 0xDC0D1014
comptime GIZMO_PLANE_GRAZING_MIN: Float32 = 0.18
comptime GIZMO_PLANE_S0: Float32 = 0.24
comptime GIZMO_PLANE_S1: Float32 = 0.44


@fieldwise_init
struct ViewportRay(Copyable, ImplicitlyCopyable, Movable):
    """World-space ray originating from a 2D viewport pixel `(origin, unit dir)`."""
    var origin: Vec3
    var dir: Vec3

    def __init__(out self):
        self.origin = Vec3()
        self.dir = Vec3(0.0, 0.0, 1.0)

    @always_inline
    def at(self, t: Float32) -> Vec3:
        return self.origin + self.dir * t

    @staticmethod
    def from_screen(
        px: Float32,
        py: Float32,
        vp_x: Float32,
        vp_y: Float32,
        vp_w: Float32,
        vp_h: Float32,
        cam_pos: Vec3,
        cam_fwd: Vec3,
        cam_right: Vec3,
        cam_up: Vec3,
        fov_y_deg: Float32,
    ) -> Self:
        var w = max(Float32(1.0), vp_w)
        var h = max(Float32(1.0), vp_h)
        var aspect = w / h
        var tan_y = tan(fov_y_deg * Float32(0.01745329252) * Float32(0.5))
        var tan_x = tan_y * aspect
        var ndc_x = ((px - vp_x) / w) * Float32(2.0) - Float32(1.0)
        var ndc_y = Float32(1.0) - ((py - vp_y) / h) * Float32(2.0)
        var d = (
            cam_fwd.normalized()
            + cam_right.normalized() * (ndc_x * tan_x)
            + cam_up.normalized() * (ndc_y * tan_y)
        ).normalized()
        return Self(cam_pos, d)


def project_world_to_viewport(
    world_pt: Vec3,
    vp_x: Float32,
    vp_y: Float32,
    vp_w: Float32,
    vp_h: Float32,
    cam_pos: Vec3,
    cam_fwd: Vec3,
    cam_right: Vec3,
    cam_up: Vec3,
    fov_y_deg: Float32,
    near_z: Float32 = 0.05,
) -> Tuple[Float32, Float32, Float32, Bool]:
    """Projects `world_pt` to viewport screen coordinates `(sx, sy, z_view, in_front)`."""
    if not (
        isfinite(world_pt.x) and isfinite(world_pt.y) and isfinite(world_pt.z)
        and isfinite(cam_pos.x) and isfinite(cam_pos.y) and isfinite(cam_pos.z)
        and isfinite(cam_fwd.x) and isfinite(cam_fwd.y) and isfinite(cam_fwd.z)
        and isfinite(cam_right.x) and isfinite(cam_right.y) and isfinite(cam_right.z)
        and isfinite(cam_up.x) and isfinite(cam_up.y) and isfinite(cam_up.z)
        and isfinite(vp_x) and isfinite(vp_y) and isfinite(vp_w) and isfinite(vp_h)
        and isfinite(fov_y_deg) and isfinite(near_z)
    ):
        return (0.0, 0.0, 0.0, False)
    var rel = world_pt - cam_pos
    var fwd = cam_fwd.normalized()
    var z_v = rel.dot(fwd)
    if not isfinite(z_v) or z_v < near_z:
        return (0.0, 0.0, z_v if isfinite(z_v) else Float32(0.0), False)
    var w = max(Float32(1.0), vp_w)
    var h = max(Float32(1.0), vp_h)
    var aspect = w / h
    var tan_y = max(Float32(1.0e-4), tan(fov_y_deg * Float32(0.01745329252) * Float32(0.5)))
    var tan_x = max(Float32(1.0e-4), tan_y * aspect)
    var ndc_x = rel.dot(cam_right.normalized()) / (z_v * tan_x)
    var ndc_y = rel.dot(cam_up.normalized()) / (z_v * tan_y)
    var sx = vp_x + (ndc_x * Float32(0.5) + Float32(0.5)) * w
    var sy = vp_y + (Float32(0.5) - ndc_y * Float32(0.5)) * h
    if not (isfinite(sx) and isfinite(sy)):
        return (0.0, 0.0, z_v, False)
    return (sx, sy, z_v, True)


def gizmo_world_scale(
    gizmo_origin: Vec3,
    cam_pos: Vec3,
    cam_fwd: Vec3,
    fov_y_deg: Float32,
    viewport_h: Int,
    target_px: Float32 = 96.0,
) -> Float32:
    """Computes world-space handle length so the gizmo projects to a constant `target_px` screen pixels."""
    if not (
        isfinite(gizmo_origin.x) and isfinite(gizmo_origin.y) and isfinite(gizmo_origin.z)
        and isfinite(cam_pos.x) and isfinite(cam_pos.y) and isfinite(cam_pos.z)
        and isfinite(cam_fwd.x) and isfinite(cam_fwd.y) and isfinite(cam_fwd.z)
        and isfinite(fov_y_deg) and isfinite(target_px)
    ):
        return Float32(1.0)
    var fwd = cam_fwd.normalized()
    var rel = gizmo_origin - cam_pos
    var z_depth = rel.dot(fwd)
    if z_depth <= Float32(0.05):
        z_depth = max(Float32(0.05), rel.length())
    var tan_half = tan(fov_y_deg * Float32(0.01745329252) * Float32(0.5))
    var world_per_px = (Float32(2.0) * z_depth * tan_half) / Float32(max(1, viewport_h))
    var res = max(Float32(1.0e-4), world_per_px * target_px)
    return res if isfinite(res) else Float32(1.0)


def ray_axis_closest_param(
    ray: ViewportRay,
    axis_origin: Vec3,
    axis_dir: Vec3,
) -> Tuple[Bool, Float32, Float32]:
    """Analytical skew-line closest parameter `t` along `axis_origin + t * unit(axis_dir)` to `ray`.
    Returns `(valid, axis_param_t, closest_distance)`."""
    if not (
        isfinite(ray.origin.x) and isfinite(ray.origin.y) and isfinite(ray.origin.z)
        and isfinite(ray.dir.x) and isfinite(ray.dir.y) and isfinite(ray.dir.z)
        and isfinite(axis_origin.x) and isfinite(axis_origin.y) and isfinite(axis_origin.z)
        and isfinite(axis_dir.x) and isfinite(axis_dir.y) and isfinite(axis_dir.z)
    ):
        return (False, Float32(0.0), Float32(0.0))
    var d_len = ray.dir.length()
    if not isfinite(d_len) or d_len <= Float32(1.0e-7):
        return (False, Float32(0.0), Float32(0.0))
    var d = ray.dir * (Float32(1.0) / d_len)
    var u_len = axis_dir.length()
    if not isfinite(u_len) or u_len <= Float32(1.0e-7):
        return (False, Float32(0.0), Float32(0.0))
    var u = axis_dir * (Float32(1.0) / u_len)
    var w0 = ray.origin - axis_origin
    var b = d.dot(u)
    var denom = Float32(1.0) - b * b
    if not isfinite(denom) or denom < Float32(1.0e-6):
        return (False, Float32(0.0), Float32(0.0))
    var d_w0 = d.dot(w0)
    var u_w0 = u.dot(w0)
    var s = (b * u_w0 - d_w0) / denom
    var t = (u_w0 - b * d_w0) / denom
    if not (isfinite(s) and isfinite(t)) or s < Float32(0.0):
        return (False, Float32(0.0), Float32(0.0))
    var dist = (ray.at(s) - (axis_origin + u * t)).length()
    if not isfinite(dist):
        return (False, Float32(0.0), Float32(0.0))
    return (True, t, dist)


def ray_plane_hit(
    ray: ViewportRay,
    plane_origin: Vec3,
    plane_normal: Vec3,
) -> Tuple[Bool, Vec3, Float32]:
    """Intersects `ray` with `(X - plane_origin) . unit(plane_normal) = 0`.
    Returns `(hit, world_hit_point, ray_t)`."""
    if not (
        isfinite(ray.origin.x) and isfinite(ray.origin.y) and isfinite(ray.origin.z)
        and isfinite(ray.dir.x) and isfinite(ray.dir.y) and isfinite(ray.dir.z)
        and isfinite(plane_origin.x) and isfinite(plane_origin.y) and isfinite(plane_origin.z)
        and isfinite(plane_normal.x) and isfinite(plane_normal.y) and isfinite(plane_normal.z)
    ):
        return (False, Vec3(), Float32(0.0))
    var n_len = plane_normal.length()
    if not isfinite(n_len) or n_len <= Float32(1.0e-7):
        return (False, plane_origin, Float32(0.0))
    var n = plane_normal * (Float32(1.0) / n_len)
    var d_len = ray.dir.length()
    if not isfinite(d_len) or d_len <= Float32(1.0e-7):
        return (False, plane_origin, Float32(0.0))
    var d = ray.dir * (Float32(1.0) / d_len)
    var denom = d.dot(n)
    if not isfinite(denom) or abs(denom) < Float32(1.0e-6):
        return (False, plane_origin, Float32(0.0))
    var t = (plane_origin - ray.origin).dot(n) / denom
    if not isfinite(t) or t < Float32(0.0):
        return (False, plane_origin, t if isfinite(t) else Float32(0.0))
    var hit_pt = ray.at(t)
    if not (isfinite(hit_pt.x) and isfinite(hit_pt.y) and isfinite(hit_pt.z)):
        return (False, plane_origin, Float32(0.0))
    return (True, hit_pt, t)


def gizmo_rotation_delta_rad(
    start_ray: ViewportRay,
    cur_ray: ViewportRay,
    pivot: Vec3,
    ring_axis: Vec3,
    cam_fwd: Vec3,
    ring_world_radius: Float32 = 1.0,
) -> Float32:
    """Computes signed rotation delta in radians around `ring_axis`.
    Automatically switches to a view-plane tangent solver when grazing incidence `|axis . view_dir| < 0.15`
    (e.g. `88°` edge-on view) so rotation never suffers near-parallel plane-intersection blowup!"""
    if not (
        isfinite(start_ray.origin.x) and isfinite(start_ray.origin.y) and isfinite(start_ray.origin.z)
        and isfinite(start_ray.dir.x) and isfinite(start_ray.dir.y) and isfinite(start_ray.dir.z)
        and isfinite(cur_ray.origin.x) and isfinite(cur_ray.origin.y) and isfinite(cur_ray.origin.z)
        and isfinite(cur_ray.dir.x) and isfinite(cur_ray.dir.y) and isfinite(cur_ray.dir.z)
        and isfinite(pivot.x) and isfinite(pivot.y) and isfinite(pivot.z)
        and isfinite(ring_axis.x) and isfinite(ring_axis.y) and isfinite(ring_axis.z)
        and isfinite(cam_fwd.x) and isfinite(cam_fwd.y) and isfinite(cam_fwd.z)
        and isfinite(ring_world_radius)
    ):
        return Float32(0.0)
    var ax_len = ring_axis.length()
    if not isfinite(ax_len) or ax_len <= Float32(1.0e-7):
        return Float32(0.0)
    var axis = ring_axis * (Float32(1.0) / ax_len)
    var cf_len = cam_fwd.length()
    if not isfinite(cf_len) or cf_len <= Float32(1.0e-7):
        return Float32(0.0)
    var view_n = cam_fwd * (Float32(1.0) / cf_len)
    var to_pivot = pivot - start_ray.origin
    var tp_len2 = to_pivot.length_squared()
    var view_ray = to_pivot.normalized() if (isfinite(tp_len2) and tp_len2 > Float32(1.0e-8)) else view_n

    var grazing_cos = abs(axis.dot(view_ray))
    if not isfinite(grazing_cos):
        return Float32(0.0)
    if grazing_cos >= Float32(0.15):
        var h0 = ray_plane_hit(start_ray, pivot, axis)
        var h1 = ray_plane_hit(cur_ray, pivot, axis)
        if h0[0] and h1[0]:
            var v0 = h0[1] - pivot
            var v1 = h1[1] - pivot
            v0 = v0 - axis * v0.dot(axis)
            v1 = v1 - axis * v1.dot(axis)
            if v0.length_squared() > Float32(1.0e-8) and v1.length_squared() > Float32(1.0e-8):
                var u0 = v0.normalized()
                var u1 = v1.normalized()
                var sin_a = axis.dot(u0.cross(u1))
                var cos_a = max(Float32(-1.0), min(Float32(1.0), u0.dot(u1)))
                if isfinite(sin_a) and isfinite(cos_a):
                    var ang = atan2(sin_a, cos_a)
                    if isfinite(ang):
                        return ang
                return Float32(0.0)

    # Automatic `< 0.15` grazing-angle view-plane tangent fallback!
    var vp0 = ray_plane_hit(start_ray, pivot, view_n)
    var vp1 = ray_plane_hit(cur_ray, pivot, view_n)
    if not vp0[0] or not vp1[0]:
        return Float32(0.0)
    var raw_tan = axis.cross(view_n)
    var tangent = raw_tan.normalized() if raw_tan.length_squared() > Float32(1.0e-8) else Vec3(1.0, 0.0, 0.0)
    var eff_r = max(Float32(0.05), ring_world_radius)
    var disp = (vp1[1] - vp0[1]).dot(tangent)
    if not isfinite(disp):
        return Float32(0.0)
    var d_rad = disp / eff_r
    if not isfinite(d_rad):
        return Float32(0.0)
    return max(Float32(-6.2831853), min(Float32(6.2831853), d_rad))


def _sanitize_gizmo_rot(q: Quat) -> Quat:
    if (
        isfinite(q.x) and isfinite(q.y) and isfinite(q.z) and isfinite(q.w)
        and q.length_squared() > Float32(1.0e-8)
    ):
        return q.normalized()
    return Quat.identity()


def gizmo_scale_drag(
    start_ray: ViewportRay,
    cur_ray: ViewportRay,
    pivot: Vec3,
    handle: Int,
    gizmo_radius: Float32,
    cam_fwd: Vec3,
    cam_right: Vec3,
    start_scale: Vec3,
    snap_step: Float32 = 0.0,
    gizmo_rot: Quat = Quat.identity(),
) -> Vec3:
    """Solves 1D per-axis or uniform 3D scale drag relative to `start_scale`."""
    var safe_start = Vec3(
        start_scale.x if isfinite(start_scale.x) else Float32(1.0),
        start_scale.y if isfinite(start_scale.y) else Float32(1.0),
        start_scale.z if isfinite(start_scale.z) else Float32(1.0),
    )
    if not (
        isfinite(pivot.x) and isfinite(pivot.y) and isfinite(pivot.z)
        and isfinite(cam_fwd.x) and isfinite(cam_fwd.y) and isfinite(cam_fwd.z)
        and isfinite(cam_right.x) and isfinite(cam_right.y) and isfinite(cam_right.z)
    ):
        return safe_start
    var gr = gizmo_radius if isfinite(gizmo_radius) else Float32(1.0)
    var eff_r = max(Float32(0.05), gr)
    var safe_snap = snap_step if (isfinite(snap_step) and snap_step > Float32(1.0e-6)) else Float32(0.0)
    if handle == GIZMO_SCALE_UNIFORM:
        var vp0 = ray_plane_hit(start_ray, pivot, cam_fwd)
        var vp1 = ray_plane_hit(cur_ray, pivot, cam_fwd)
        if not vp0[0] or not vp1[0]:
            return safe_start
        var cr_len = cam_right.length()
        if not isfinite(cr_len) or cr_len <= Float32(1.0e-7):
            return safe_start
        var cr = cam_right * (Float32(1.0) / cr_len)
        var delta_u = (vp1[1] - vp0[1]).dot(cr) / eff_r
        if not isfinite(delta_u):
            return safe_start
        var factor = max(Float32(0.05), Float32(1.0) + delta_u)
        if safe_snap > Float32(0.0):
            factor = max(safe_snap, snap_scalar(factor, safe_snap))
        var res_u = safe_start * factor
        if not (isfinite(res_u.x) and isfinite(res_u.y) and isfinite(res_u.z)):
            return safe_start
        return res_u

    var rot = _sanitize_gizmo_rot(gizmo_rot)
    var ax = rot.rotate(Vec3(1.0, 0.0, 0.0))
    if handle == GIZMO_AXIS_Y:
        ax = rot.rotate(Vec3(0.0, 1.0, 0.0))
    elif handle == GIZMO_AXIS_Z:
        ax = rot.rotate(Vec3(0.0, 0.0, 1.0))
    var p0 = ray_axis_closest_param(start_ray, pivot, ax)
    var p1 = ray_axis_closest_param(cur_ray, pivot, ax)
    if not p0[0] or not p1[0]:
        return safe_start
    var ratio_raw = Float32(1.0) + (p1[1] - p0[1]) / eff_r
    if not isfinite(ratio_raw):
        return safe_start
    var ratio = max(Float32(0.05), ratio_raw)
    var out = safe_start
    if handle == GIZMO_AXIS_X:
        var sx = safe_start.x * ratio
        out.x = max(Float32(0.01), sx) if isfinite(sx) else safe_start.x
        if safe_snap > Float32(0.0):
            out.x = max(safe_snap, snap_scalar(out.x, safe_snap))
    elif handle == GIZMO_AXIS_Y:
        var sy = safe_start.y * ratio
        out.y = max(Float32(0.01), sy) if isfinite(sy) else safe_start.y
        if safe_snap > Float32(0.0):
            out.y = max(safe_snap, snap_scalar(out.y, safe_snap))
    elif handle == GIZMO_AXIS_Z:
        var sz = safe_start.z * ratio
        out.z = max(Float32(0.01), sz) if isfinite(sz) else safe_start.z
        if safe_snap > Float32(0.0):
            out.z = max(safe_snap, snap_scalar(out.z, safe_snap))
    return out


@always_inline
def snap_scalar(val: Float32, step: Float32) -> Float32:
    if not isfinite(val):
        return Float32(0.0)
    if not isfinite(step) or step <= Float32(1.0e-6):
        return val
    var q = Float64(val) / Float64(step)
    if not isfinite(q):
        return val
    var snapped = Float32(floor(q + 0.5) * Float64(step))
    return snapped if isfinite(snapped) else val


@always_inline
def snap_translation(pos: Vec3, grid_step: Float32) -> Vec3:
    if not isfinite(grid_step) or grid_step <= Float32(1.0e-6):
        return pos
    return Vec3(
        snap_scalar(pos.x, grid_step),
        snap_scalar(pos.y, grid_step),
        snap_scalar(pos.z, grid_step),
    )


@always_inline
def snap_angle(angle_rad: Float32, step_rad: Float32) -> Float32:
    if not isfinite(step_rad) or step_rad <= Float32(1.0e-6):
        return angle_rad
    return snap_scalar(angle_rad, step_rad)


@always_inline
def snap_angle_deg(angle_rad: Float32, step_deg: Float32 = 15.0) -> Float32:
    return snap_angle(angle_rad, step_deg * Float32(0.01745329252))


def _point_seg_dist_px(px: Float32, py: Float32, ax: Float32, ay: Float32, bx: Float32, by: Float32) -> Float32:
    if not (isfinite(px) and isfinite(py) and isfinite(ax) and isfinite(ay) and isfinite(bx) and isfinite(by)):
        return Float32(1.0e30)
    var vx = Float64(bx) - Float64(ax)
    var vy = Float64(by) - Float64(ay)
    var len2 = vx * vx + vy * vy
    if not isfinite(len2) or len2 <= 1.0e-4:
        var dx0 = Float64(px) - Float64(ax)
        var dy0 = Float64(py) - Float64(ay)
        var d0 = sqrt(dx0 * dx0 + dy0 * dy0)
        return Float32(d0) if isfinite(d0) else Float32(1.0e30)
    var t = max(0.0, min(1.0, ((Float64(px) - Float64(ax)) * vx + (Float64(py) - Float64(ay)) * vy) / len2))
    var cx = Float64(ax) + vx * t
    var cy = Float64(ay) + vy * t
    var dx = Float64(px) - cx
    var dy = Float64(py) - cy
    var d = sqrt(dx * dx + dy * dy)
    return Float32(d) if isfinite(d) else Float32(1.0e30)


def _gizmo_view_dir(pivot: Vec3, cam_pos: Vec3, cam_fwd: Vec3) -> Vec3:
    var to_piv = pivot - cam_pos
    var tp2 = to_piv.length_squared()
    if isfinite(tp2) and tp2 > Float32(1.0e-8):
        return to_piv.normalized()
    var cf2 = cam_fwd.length_squared()
    if isfinite(cf2) and cf2 > Float32(1.0e-8):
        return cam_fwd.normalized()
    return Vec3(0.0, 0.0, 1.0)


def _point_in_quad_px(
    px: Float32,
    py: Float32,
    x0: Float32,
    y0: Float32,
    x1: Float32,
    y1: Float32,
    x2: Float32,
    y2: Float32,
    x3: Float32,
    y3: Float32,
) -> Bool:
    var c0 = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
    var c1 = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    var c2 = (x3 - x2) * (py - y2) - (y3 - y2) * (px - x2)
    var c3 = (x0 - x3) * (py - y3) - (y0 - y3) * (px - x3)
    var all_pos = c0 >= Float32(0.0) and c1 >= Float32(0.0) and c2 >= Float32(0.0) and c3 >= Float32(0.0)
    var all_neg = c0 <= Float32(0.0) and c1 <= Float32(0.0) and c2 <= Float32(0.0) and c3 <= Float32(0.0)
    return all_pos or all_neg


def _emit_gizmo_arrow_cone(
    mut ui: UiDrawList,
    tip_x: Float32,
    tip_y: Float32,
    ux: Float32,
    uy: Float32,
    head_len: Float32,
    half_w: Float32,
    color: UInt32,
    halo_alpha_scale: Float32 = 1.0,
):
    """Emits a razor-sharp 4-vertex machined cone arrowhead (`PRIM_POLYGON`) with dark contrast halo."""
    var px = -uy
    var py = ux
    var notch = head_len * Float32(0.15)

    # Subtle expanded dark halo behind arrowhead for high-contrast separation over bright speculars
    var h_pad = Float32(1.4)
    var h_tip_x = tip_x + ux * h_pad
    var h_tip_y = tip_y + uy * h_pad
    var h_len = head_len + h_pad * Float32(1.4)
    var h_hw = half_w + h_pad
    var h_xy = List[Float32](capacity=8)
    h_xy.append(h_tip_x)
    h_xy.append(h_tip_y)
    h_xy.append(tip_x - ux * h_len + px * h_hw)
    h_xy.append(tip_y - uy * h_len + py * h_hw)
    h_xy.append(tip_x - ux * (h_len - notch))
    h_xy.append(tip_y - uy * (h_len - notch))
    h_xy.append(tip_x - ux * h_len - px * h_hw)
    h_xy.append(tip_y - uy * h_len - py * h_hw)
    var counts = List[Int](capacity=1)
    counts.append(4)
    _ = ui.add_polygon(h_xy, counts, rgba8_scale_alpha(GIZMO_COLOR_HALO, halo_alpha_scale))

    var xy = List[Float32](capacity=8)
    xy.append(tip_x)
    xy.append(tip_y)
    xy.append(tip_x - ux * head_len + px * half_w)
    xy.append(tip_y - uy * head_len + py * half_w)
    xy.append(tip_x - ux * (head_len - notch))
    xy.append(tip_y - uy * (head_len - notch))
    xy.append(tip_x - ux * head_len - px * half_w)
    xy.append(tip_y - uy * head_len - py * half_w)
    _ = ui.add_polygon(xy, counts, color)


def pick_gizmo_handle(
    mouse_x: Float32,
    mouse_y: Float32,
    pivot: Vec3,
    mode: Int,
    vp_x: Float32,
    vp_y: Float32,
    vp_w: Float32,
    vp_h: Float32,
    cam_pos: Vec3,
    cam_fwd: Vec3,
    cam_right: Vec3,
    cam_up: Vec3,
    fov_y_deg: Float32,
    target_px: Float32 = 96.0,
    gizmo_rot: Quat = Quat.identity(),
) -> Int:
    """Screen-space handle picker returning `GIZMO_AXIS_*`, `GIZMO_PLANE_*`, or `GIZMO_SCALE_UNIFORM`."""
    if mode == GIZMO_MODE_SELECT:
        return GIZMO_AXIS_NONE
    if not (
        isfinite(mouse_x) and isfinite(mouse_y)
        and isfinite(vp_x) and isfinite(vp_y) and isfinite(vp_w) and isfinite(vp_h)
        and isfinite(target_px)
    ):
        return GIZMO_AXIS_NONE
    var c_proj = project_world_to_viewport(pivot, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
    if not c_proj[3]:
        return GIZMO_AXIS_NONE
    var vph_i = Int(max(Float32(1.0), min(Float32(16384.0), vp_h)))
    var wscale = gizmo_world_scale(pivot, cam_pos, cam_fwd, fov_y_deg, vph_i, target_px)
    var rot = _sanitize_gizmo_rot(gizmo_rot)
    var bx = rot.rotate(Vec3(1.0, 0.0, 0.0))
    var by = rot.rotate(Vec3(0.0, 1.0, 0.0))
    var bz = rot.rotate(Vec3(0.0, 0.0, 1.0))
    var view_dir = _gizmo_view_dir(pivot, cam_pos, cam_fwd)

    if mode == GIZMO_MODE_SCALE:
        var dc = _point_seg_dist_px(mouse_x, mouse_y, c_proj[0], c_proj[1], c_proj[0], c_proj[1])
        if dc <= Float32(11.0):
            return GIZMO_SCALE_UNIFORM

    if mode == GIZMO_MODE_TRANSLATE or mode == GIZMO_MODE_SCALE:
        if mode == GIZMO_MODE_TRANSLATE:
            var pl_u = [bx, bx, by]
            var pl_v = [by, bz, bz]
            var pl_n = [bz, by, bx]
            var pl_ids = [GIZMO_PLANE_XY, GIZMO_PLANE_XZ, GIZMO_PLANE_YZ]
            var s0 = wscale * GIZMO_PLANE_S0
            var s1 = wscale * GIZMO_PLANE_S1
            var best_plane = GIZMO_AXIS_NONE
            var best_plane_z = Float32(1.0e30)
            for pi in range(3):
                var grazing = abs(pl_n[pi].dot(view_dir))
                if grazing < GIZMO_PLANE_GRAZING_MIN:
                    continue
                var q0 = project_world_to_viewport(pivot + pl_u[pi] * s0 + pl_v[pi] * s0, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                var q1 = project_world_to_viewport(pivot + pl_u[pi] * s1 + pl_v[pi] * s0, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                var q2 = project_world_to_viewport(pivot + pl_u[pi] * s1 + pl_v[pi] * s1, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                var q3 = project_world_to_viewport(pivot + pl_u[pi] * s0 + pl_v[pi] * s1, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                if q0[3] and q1[3] and q2[3] and q3[3]:
                    var hit_q = _point_in_quad_px(mouse_x, mouse_y, q0[0], q0[1], q1[0], q1[1], q2[0], q2[1], q3[0], q3[1])
                    var mid_x = (q0[0] + q1[0] + q2[0] + q3[0]) * Float32(0.25)
                    var mid_y = (q0[1] + q1[1] + q2[1] + q3[1]) * Float32(0.25)
                    var near_mid = _point_seg_dist_px(mouse_x, mouse_y, mid_x, mid_y, mid_x, mid_y) <= Float32(9.0)
                    if hit_q or near_mid:
                        var avg_z = (q0[2] + q1[2] + q2[2] + q3[2]) * Float32(0.25)
                        if avg_z < best_plane_z:
                            best_plane_z = avg_z
                            best_plane = pl_ids[pi]
            if best_plane != GIZMO_AXIS_NONE:
                return best_plane

        var axes = [bx, by, bz]
        var ids = [GIZMO_AXIS_X, GIZMO_AXIS_Y, GIZMO_AXIS_Z]
        var best_id = GIZMO_AXIS_NONE
        var best_d = Float32(12.0)
        for a in range(3):
            if abs(axes[a].dot(view_dir)) > Float32(0.985):
                continue
            var tip = project_world_to_viewport(pivot + axes[a] * wscale, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            if tip[3]:
                var d = _point_seg_dist_px(mouse_x, mouse_y, c_proj[0], c_proj[1], tip[0], tip[1])
                if d < best_d:
                    best_d = d
                    best_id = ids[a]
        return best_id

    # Rotate mode: prioritize front-hemisphere projected 3D rings
    var ring_u = [by, bx, bx]
    var ring_v = [bz, bz, by]
    var r_ids = [GIZMO_AXIS_X, GIZMO_AXIS_Y, GIZMO_AXIS_Z]
    var best_r = GIZMO_AXIS_NONE
    var best_rd = Float32(11.0)
    comptime SEG: Int = 32
    for ri in range(3):
        for s in range(SEG):
            var a0 = Float32(s) * Float32(6.2831853 / Float32(SEG))
            var a1 = Float32(s + 1) * Float32(6.2831853 / Float32(SEG))
            var p0 = pivot + (ring_u[ri] * cos(a0) + ring_v[ri] * sin(a0)) * wscale
            var p1 = pivot + (ring_u[ri] * cos(a1) + ring_v[ri] * sin(a1)) * wscale
            var mid = (p0 + p1) * Float32(0.5)
            # Bias slightly toward front hemisphere while still allowing grazing grabs
            var is_back = (mid - pivot).dot(view_dir) > wscale * Float32(0.12)
            var s0 = project_world_to_viewport(p0, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            var s1 = project_world_to_viewport(p1, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            if s0[3] and s1[3]:
                var d = _point_seg_dist_px(mouse_x, mouse_y, s0[0], s0[1], s1[0], s1[1])
                var eff_d = d + (Float32(3.5) if is_back else Float32(0.0))
                if eff_d < best_rd:
                    best_rd = eff_d
                    best_r = r_ids[ri]
    return best_r


def emit_gizmo_draw_list(
    mut ui: UiDrawList,
    pivot: Vec3,
    mode: Int,
    active_handle: Int,
    vp_x: Float32,
    vp_y: Float32,
    vp_w: Float32,
    vp_h: Float32,
    cam_pos: Vec3,
    cam_fwd: Vec3,
    cam_right: Vec3,
    cam_up: Vec3,
    fov_y_deg: Float32,
    target_px: Float32 = 96.0,
    gizmo_rot: Quat = Quat.identity(),
    hovered_handle: Int = GIZMO_AXIS_NONE,
    axis_x_color: UInt32 = GIZMO_COLOR_X,
    axis_y_color: UInt32 = GIZMO_COLOR_Y,
    axis_z_color: UInt32 = GIZMO_COLOR_Z,
    active_color: UInt32 = GIZMO_COLOR_ACTIVE,
    drag_angle_rad: Float32 = 0.0,
):
    """Emits razor-sharp analytical GPU SDF 3D gizmos (depth-sorted `PRIM_LINE` shafts + contrast halos,
    tapered `PRIM_POLYGON` cone arrowheads, true 3D perspective plane quads with grazing-angle fade,
    hemisphere-cued rotation rings, scale cubes, and ringed center reticle)."""
    if mode == GIZMO_MODE_SELECT:
        return
    if not (isfinite(vp_x) and isfinite(vp_y) and isfinite(vp_w) and isfinite(vp_h) and isfinite(target_px)):
        return
    var c = project_world_to_viewport(pivot, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
    if not c[3]:
        return
    var cx0 = Int(max(Float32(-1.0e6), min(Float32(1.0e6), vp_x)))
    var cy0 = Int(max(Float32(-1.0e6), min(Float32(1.0e6), vp_y)))
    var cx1 = Int(max(Float32(-1.0e6), min(Float32(1.0e6), vp_x + vp_w)))
    var cy1 = Int(max(Float32(-1.0e6), min(Float32(1.0e6), vp_y + vp_h)))
    var vph_i = max(1, min(16384, cy1 - cy0))
    ui.push_clip(cx0, cy0, cx1, cy1)

    var wscale = gizmo_world_scale(pivot, cam_pos, cam_fwd, fov_y_deg, vph_i, target_px)
    var rot = _sanitize_gizmo_rot(gizmo_rot)
    var bx = rot.rotate(Vec3(1.0, 0.0, 0.0))
    var by = rot.rotate(Vec3(0.0, 1.0, 0.0))
    var bz = rot.rotate(Vec3(0.0, 0.0, 1.0))
    var view_dir = _gizmo_view_dir(pivot, cam_pos, cam_fwd)

    var axes = [bx, by, bz]
    var cols = [axis_x_color, axis_y_color, axis_z_color]
    var h_ids = [GIZMO_AXIS_X, GIZMO_AXIS_Y, GIZMO_AXIS_Z]

    if mode == GIZMO_MODE_TRANSLATE or mode == GIZMO_MODE_SCALE:
        # 1. Dual-axis 3D perspective-foreshortened planar handles (Translate mode)
        if mode == GIZMO_MODE_TRANSLATE:
            var pl_u = [bx, bx, by]
            var pl_v = [by, bz, bz]
            var pl_n = [bz, by, bx]
            var pl_cols = [axis_z_color, axis_y_color, axis_x_color]
            var pl_ids = [GIZMO_PLANE_XY, GIZMO_PLANE_XZ, GIZMO_PLANE_YZ]
            var s0 = wscale * GIZMO_PLANE_S0
            var s1 = wscale * GIZMO_PLANE_S1
            var quad_counts = List[Int](capacity=1)
            quad_counts.append(4)
            for pi in range(3):
                var grazing = abs(pl_n[pi].dot(view_dir))
                if grazing < GIZMO_PLANE_GRAZING_MIN:
                    continue
                var vis_t = min(Float32(1.0), (grazing - GIZMO_PLANE_GRAZING_MIN) / Float32(0.18))
                var q0 = project_world_to_viewport(pivot + pl_u[pi] * s0 + pl_v[pi] * s0, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                var q1 = project_world_to_viewport(pivot + pl_u[pi] * s1 + pl_v[pi] * s0, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                var q2 = project_world_to_viewport(pivot + pl_u[pi] * s1 + pl_v[pi] * s1, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                var q3 = project_world_to_viewport(pivot + pl_u[pi] * s0 + pl_v[pi] * s1, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                if q0[3] and q1[3] and q2[3] and q3[3]:
                    var is_act = active_handle == pl_ids[pi]
                    var is_hov = hovered_handle == pl_ids[pi]
                    var base_c = active_color if is_act else (lerp_rgba8(pl_cols[pi], UInt32(0xFFFFFFFF), Float32(0.28)) if is_hov else pl_cols[pi])
                    var fill_a = Float32(0.54) if is_act else (Float32(0.42) if is_hov else Float32(0.26))
                    var edge_a = Float32(0.96) if (is_act or is_hov) else Float32(0.84)
                    var edge_t = Float32(1.9) if (is_act or is_hov) else Float32(1.4)
                    var q_xy = List[Float32](capacity=8)
                    q_xy.append(q0[0]); q_xy.append(q0[1])
                    q_xy.append(q1[0]); q_xy.append(q1[1])
                    q_xy.append(q2[0]); q_xy.append(q2[1])
                    q_xy.append(q3[0]); q_xy.append(q3[1])
                    _ = ui.add_polygon(q_xy, quad_counts, rgba8_scale_alpha(base_c, fill_a * vis_t))
                    var ec = rgba8_scale_alpha(base_c, edge_a * vis_t)
                    ui.add_line(q0[0], q0[1], q1[0], q1[1], edge_t, ec)
                    ui.add_line(q1[0], q1[1], q2[0], q2[1], edge_t, ec)
                    ui.add_line(q2[0], q2[1], q3[0], q3[1], edge_t, ec)
                    ui.add_line(q3[0], q3[1], q0[0], q0[1], edge_t, ec)

        # 2. Project & depth-sort X/Y/Z axes back-to-front (largest z_view first)
        var order = [0, 1, 2]
        var tip_xs = [Float32(0.0), Float32(0.0), Float32(0.0)]
        var tip_ys = [Float32(0.0), Float32(0.0), Float32(0.0)]
        var tip_zs = [Float32(0.0), Float32(0.0), Float32(0.0)]
        var tip_ok = [False, False, False]
        for a in range(3):
            var tip = project_world_to_viewport(pivot + axes[a] * wscale, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            tip_xs[a] = tip[0]
            tip_ys[a] = tip[1]
            tip_zs[a] = tip[2]
            tip_ok[a] = tip[3]
        for i in range(3):
            for j in range(i + 1, 3):
                if tip_zs[order[j]] > tip_zs[order[i]]:
                    var tmp = order[i]
                    order[i] = order[j]
                    order[j] = tmp

        # 3. Emit depth-sorted analytical shafts + sharp cone arrowheads / machined scale cubes
        for oi in range(3):
            var a = order[oi]
            if not tip_ok[a]:
                continue
            var align = abs(axes[a].dot(view_dir))
            if align > Float32(0.985):
                continue
            var fade = Float32(1.0) if align < Float32(0.90) else max(Float32(0.25), (Float32(0.985) - align) / Float32(0.085))
            var is_act = active_handle == h_ids[a]
            var is_hov = hovered_handle == h_ids[a]
            var col_raw = active_color if is_act else (lerp_rgba8(cols[a], UInt32(0xFFFFFFFF), Float32(0.32)) if is_hov else cols[a])
            var col = rgba8_scale_alpha(col_raw, fade)
            var tx = tip_xs[a]
            var ty = tip_ys[a]
            var dx = tx - c[0]
            var dy = ty - c[1]
            var slen = sqrt(dx * dx + dy * dy)
            if slen <= Float32(3.0):
                continue
            var ux = dx / slen
            var uy = dy / slen

            # Subtle infinite axis guide rail during active drag
            if is_act:
                var rail_col = rgba8_with_alpha(active_color, 72)
                ui.add_line(c[0] - ux * Float32(420.0), c[1] - uy * Float32(420.0), c[0] + ux * Float32(420.0), c[1] + uy * Float32(420.0), Float32(1.2), rail_col)

            var head_len = Float32(16.5) if (is_act or is_hov) else Float32(14.5)
            var half_w = Float32(5.6) if (is_act or is_hov) else Float32(4.8)
            var start_off = min(slen * Float32(0.35), Float32(6.8))
            var end_trim = min(slen - start_off, (head_len * Float32(0.82)) if mode == GIZMO_MODE_TRANSLATE else Float32(5.0))
            var sx0 = c[0] + ux * start_off
            var sy0 = c[1] + uy * start_off
            var sx1 = tx - ux * end_trim
            var sy1 = ty - uy * end_trim
            var shaft_t = Float32(3.4) if (is_act or is_hov) else Float32(2.7)

            # Analytical SDF shaft with dark contrast backing halo
            ui.add_line(sx0, sy0, sx1, sy1, shaft_t + Float32(2.2), rgba8_scale_alpha(GIZMO_COLOR_HALO, fade))
            ui.add_line(sx0, sy0, sx1, sy1, shaft_t, col)

            if mode == GIZMO_MODE_TRANSLATE:
                _emit_gizmo_arrow_cone(ui, tx, ty, ux, uy, head_len, half_w, col, fade)
            else:
                var csz = Float32(11.0) if (is_act or is_hov) else Float32(9.6)
                var ch = csz * Float32(0.5)
                ui.add_rect(tx - ch - Float32(1.0), ty - ch - Float32(1.0), csz + Float32(2.0), csz + Float32(2.0), rgba8_scale_alpha(GIZMO_COLOR_HALO, fade), Float32(2.8))
                ui.add_rect(tx - ch, ty - ch, csz, csz, col, Float32(2.0), Float32(0.8), rgba8_scale_alpha(UInt32(0x40FFFFFF), fade))

        # 4. Precision Center Pivot Reticle
        var ctr_act = active_handle == GIZMO_SCALE_UNIFORM
        var ctr_hov = hovered_handle == GIZMO_SCALE_UNIFORM
        var center_col = active_color if ctr_act else (UInt32(0xFFFFFFFF) if ctr_hov else GIZMO_COLOR_CENTER)
        if mode == GIZMO_MODE_SCALE:
            var usz = Float32(13.0) if (ctr_act or ctr_hov) else Float32(11.6)
            var uh = usz * Float32(0.5)
            ui.add_rect(c[0] - uh - Float32(1.0), c[1] - uh - Float32(1.0), usz + Float32(2.0), usz + Float32(2.0), GIZMO_COLOR_HALO, Float32(3.2))
            ui.add_rect(c[0] - uh, c[1] - uh, usz, usz, GIZMO_COLOR_CORE, Float32(2.4), Float32(1.8), center_col)
        else:
            # Subtle outer screen-space free-move ring + dark-cored pivot ring
            ui.add_arc(c[0], c[1], Float32(11.5), Float32(1.2), Float32(0.0), Float32(6.2831853), UInt32(0x4CEEF2F7))
            ui.add_circle(c[0], c[1], Float32(5.6), GIZMO_COLOR_HALO)
            ui.add_circle(c[0], c[1], Float32(4.8), GIZMO_COLOR_CORE, Float32(1.8), center_col)
    else:
        # 3D Rotation Rings with Back/Front Hemisphere Depth Cueing
        # Subtle screen-space outer horizon silhouette ring
        ui.add_arc(c[0], c[1], target_px * Float32(1.07), Float32(2.4), Float32(0.0), Float32(6.2831853), UInt32(0x36080B10))
        ui.add_arc(c[0], c[1], target_px * Float32(1.07), Float32(1.2), Float32(0.0), Float32(6.2831853), UInt32(0x52DCE4F0))

        var ring_u = [by, bx, bx]
        var ring_v = [bz, bz, by]
        comptime RSEG: Int = 40
        var step_a = Float32(6.2831853 / Float32(RSEG))

        # Optional translucent angle sector wedge when actively dragging a rotation ring
        if isfinite(drag_angle_rad) and abs(drag_angle_rad) > Float32(0.01):
            var act_ri = 0 if active_handle == GIZMO_AXIS_X else (1 if active_handle == GIZMO_AXIS_Y else (2 if active_handle == GIZMO_AXIS_Z else -1))
            if act_ri >= 0:
                var wedge_steps = max(4, min(24, Int(abs(drag_angle_rad) * Float32(6.0)) + 2))
                var w_xy = List[Float32](capacity=(wedge_steps + 2) * 2)
                w_xy.append(c[0])
                w_xy.append(c[1])
                var ok_wedge = True
                for wi in range(wedge_steps + 1):
                    var ang = drag_angle_rad * (Float32(wi) / Float32(wedge_steps))
                    var wp = pivot + (ring_u[act_ri] * cos(ang) + ring_v[act_ri] * sin(ang)) * wscale
                    var sp = project_world_to_viewport(wp, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                    if not sp[3]:
                        ok_wedge = False
                        break
                    w_xy.append(sp[0])
                    w_xy.append(sp[1])
                if ok_wedge and len(w_xy) >= 6:
                    var wc = List[Int](capacity=1)
                    wc.append(len(w_xy) // 2)
                    _ = ui.add_polygon(w_xy, wc, rgba8_with_alpha(active_color, 58))

        # Pass 1: Back hemisphere (receding arcs, softly ghosted)
        for ri in range(3):
            var is_act = active_handle == h_ids[ri]
            var is_hov = hovered_handle == h_ids[ri]
            var base_c = active_color if is_act else (lerp_rgba8(cols[ri], UInt32(0xFFFFFFFF), Float32(0.28)) if is_hov else cols[ri])
            var back_c = rgba8_scale_alpha(base_c, Float32(0.36) if is_act else Float32(0.24))
            for s in range(RSEG):
                var a0 = Float32(s) * step_a
                var a1 = Float32(s + 1) * step_a
                var p0 = pivot + (ring_u[ri] * cos(a0) + ring_v[ri] * sin(a0)) * wscale
                var p1 = pivot + (ring_u[ri] * cos(a1) + ring_v[ri] * sin(a1)) * wscale
                var mid = (p0 + p1) * Float32(0.5)
                if (mid - pivot).dot(view_dir) <= wscale * Float32(0.04):
                    continue
                var s0 = project_world_to_viewport(p0, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                var s1 = project_world_to_viewport(p1, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                if s0[3] and s1[3]:
                    ui.add_line(s0[0], s0[1], s1[0], s1[1], Float32(1.3), back_c)

        # Pass 2: Front hemisphere (facing arcs, crisp analytical SDF stroke + dark contrast halo)
        for ri in range(3):
            var is_act = active_handle == h_ids[ri]
            var is_hov = hovered_handle == h_ids[ri]
            var rcol = active_color if is_act else (lerp_rgba8(cols[ri], UInt32(0xFFFFFFFF), Float32(0.32)) if is_hov else cols[ri])
            var rt = Float32(3.3) if (is_act or is_hov) else Float32(2.5)
            for s in range(RSEG):
                var a0 = Float32(s) * step_a
                var a1 = Float32(s + 1) * step_a
                var p0 = pivot + (ring_u[ri] * cos(a0) + ring_v[ri] * sin(a0)) * wscale
                var p1 = pivot + (ring_u[ri] * cos(a1) + ring_v[ri] * sin(a1)) * wscale
                var mid = (p0 + p1) * Float32(0.5)
                if (mid - pivot).dot(view_dir) > wscale * Float32(0.04):
                    continue
                var s0 = project_world_to_viewport(p0, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                var s1 = project_world_to_viewport(p1, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                if s0[3] and s1[3]:
                    ui.add_line(s0[0], s0[1], s1[0], s1[1], rt + Float32(2.0), GIZMO_COLOR_HALO)
                    ui.add_line(s0[0], s0[1], s1[0], s1[1], rt, rcol)

        # Center pivot reticle
        ui.add_circle(c[0], c[1], Float32(4.6), GIZMO_COLOR_HALO)
        ui.add_circle(c[0], c[1], Float32(3.8), GIZMO_COLOR_CORE, Float32(1.6), GIZMO_COLOR_CENTER)

    ui.pop_clip()
