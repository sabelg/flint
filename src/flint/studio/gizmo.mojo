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
from std.math import abs, atan2, cos, floor, isfinite, max, min, sin, sqrt, tan
from flint.core.math import Quat, Vec3
from flint.ui.draw_list import UiDrawList

comptime GIZMO_TARGET_PX: Float32 = 96.0
"""Gizmo handle length in UI points (multiply by the UI scale for physical pixels)."""
comptime GIZMO_MODE_TRANSLATE: Int = 0
comptime GIZMO_MODE_ROTATE: Int = 1
comptime GIZMO_MODE_SCALE: Int = 2
comptime GIZMO_MODE_SELECT: Int = 3
"""Select tool (Q): no gizmo is drawn or picked; clicks only select."""

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
comptime GIZMO_AXIS_VIEW: Int = 8
"""Rotate mode's outer screen ring: turns the selection about the camera's view axis."""
comptime GIZMO_VIEW_RING_SCALE: Float32 = 1.06

comptime GIZMO_COLOR_X: UInt32 = 0xFFE84848
comptime GIZMO_COLOR_Y: UInt32 = 0xFF3DDC68
comptime GIZMO_COLOR_Z: UInt32 = 0xFF428BF5
comptime GIZMO_COLOR_ACTIVE: UInt32 = 0xFFFFD54F
comptime GIZMO_COLOR_CENTER: UInt32 = 0xFFF2F5FA


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
    # Edge-on ring: it projects to a line, so the grab point on screen is the x coordinate along
    # e1 = axis x view_n; take the ring point on the half nearer the camera (the one users grab).
    # The drag tangent there is axis x r (using axis x view_n alone turned the ring against the cursor).
    var eff_r = max(Float32(0.05), ring_world_radius)
    var e1_raw = axis.cross(view_n)
    var tangent = Vec3(1.0, 0.0, 0.0)
    if e1_raw.length_squared() > Float32(1.0e-8):
        var e1 = e1_raw.normalized()
        var e2_raw = view_n - axis * view_n.dot(axis)
        var e2 = e2_raw.normalized() if e2_raw.length_squared() > Float32(1.0e-8) else view_n
        var gx = max(-eff_r, min(eff_r, (vp0[1] - pivot).dot(e1)))
        var gz = sqrt(max(Float32(0.0), eff_r * eff_r - gx * gx))
        var r = e1 * gx - e2 * gz
        var t_raw = axis.cross(r)
        tangent = t_raw.normalized() if t_raw.length_squared() > Float32(1.0e-10) else e1
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
def snap_constrained_delta(
    pivot: Vec3,
    delta: Vec3,
    handle: Int,
    grid_step: Float32,
    gizmo_rot: Quat = Quat.identity(),
) -> Vec3:
    """Grid-snaps a translate-gizmo drag along only the handle's constrained gizmo axes.

    An axis handle snaps one component and a plane handle two; the free components stay exactly where
    they were (snapping the whole pivot used to pull Y/Z onto the grid during an X drag). A gizmo axis
    that is world-aligned snaps the absolute world coordinate; a rotated (local) axis snaps the offset."""
    if not isfinite(grid_step) or grid_step <= Float32(1.0e-6):
        return delta
    var rot = _sanitize_gizmo_rot(gizmo_rot)
    var bx = rot.rotate(Vec3(1.0, 0.0, 0.0))
    var by = rot.rotate(Vec3(0.0, 1.0, 0.0))
    var bz = rot.rotate(Vec3(0.0, 0.0, 1.0))
    var axes = List[Vec3]()
    if handle == GIZMO_AXIS_X:
        axes.append(bx)
    elif handle == GIZMO_AXIS_Y:
        axes.append(by)
    elif handle == GIZMO_AXIS_Z:
        axes.append(bz)
    elif handle == GIZMO_PLANE_XY:
        axes.append(bx)
        axes.append(by)
    elif handle == GIZMO_PLANE_XZ:
        axes.append(bx)
        axes.append(bz)
    elif handle == GIZMO_PLANE_YZ:
        axes.append(by)
        axes.append(bz)
    else:
        return snap_translation(pivot + delta, grid_step) - pivot
    var out = delta
    for i in range(len(axes)):
        var a = axes[i]
        var t = delta.dot(a)
        var world_aligned = max(abs(a.x), max(abs(a.y), abs(a.z))) > Float32(0.9999)
        var snapped: Float32
        if world_aligned:
            var p = pivot.dot(a)
            snapped = snap_scalar(p + t, grid_step) - p
        else:
            snapped = snap_scalar(t, grid_step)
        out = out + a * (snapped - t)
    return out


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


def _emit_thick_line_ui(
    mut ui: UiDrawList,
    x0: Float32,
    y0: Float32,
    x1: Float32,
    y1: Float32,
    thickness: Float32,
    color: UInt32,
):
    if not (isfinite(x0) and isfinite(y0) and isfinite(x1) and isfinite(y1) and isfinite(thickness)):
        return
    var dx = x1 - x0
    var dy = y1 - y0
    var dist64 = sqrt(Float64(dx) * Float64(dx) + Float64(dy) * Float64(dy))
    if not isfinite(dist64):
        return
    var rad = max(Float32(1.2), min(Float32(256.0), thickness * Float32(0.5)))
    var denom64 = Float64(max(Float32(2.2), rad * Float32(0.95)))
    var steps = max(1, min(64, Int(min(64.0, dist64 / denom64)) + 1))
    for i in range(steps + 1):
        var t = Float32(i) / Float32(steps)
        var px = x0 + dx * t
        var py = y0 + dy * t
        if isfinite(px) and isfinite(py):
            ui.add_circle(px, py, rad, color)


comptime GIZMO_AXIS_MIN_VIEW_DEG: Float32 = 6.0
"""Axis/plane handles closer than this to edge-on (axis along the view ray, plane seen side-on) are hidden
and unpickable: dragging them divides by ~sin^2(angle), so a pixel of mouse motion moved objects metres."""


def _gizmo_view_dir(pivot: Vec3, cam_pos: Vec3, cam_fwd: Vec3) -> Vec3:
    var to_p = pivot - cam_pos
    var l2 = to_p.length_squared()
    if isfinite(l2) and l2 > Float32(1.0e-10):
        return to_p.normalized()
    return cam_fwd.normalized()


def gizmo_axis_usable(axis: Vec3, view_dir: Vec3) -> Bool:
    """True when a translate/scale axis handle is far enough from the view ray to drag stably."""
    var c = abs(axis.dot(view_dir))
    return isfinite(c) and c < cos(GIZMO_AXIS_MIN_VIEW_DEG * Float32(0.017453292))


def gizmo_plane_usable(normal: Vec3, view_dir: Vec3) -> Bool:
    """True when a dual-axis plane handle is not seen (nearly) edge-on."""
    var c = abs(normal.dot(view_dir))
    return isfinite(c) and c > sin(GIZMO_AXIS_MIN_VIEW_DEG * Float32(0.017453292))


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
    px_scale: Float32 = 1.0,
) -> Int:
    """Screen-space handle picker returning `GIZMO_AXIS_*`, `GIZMO_PLANE_*`, `GIZMO_SCALE_UNIFORM` or
    `GIZMO_AXIS_VIEW`. `px_scale` (physical pixels per UI point) scales every hit tolerance, so handles
    are as easy to grab at 200% UI scale as at 100%. Rotate rings only accept hits on the half facing
    the camera: the far half is behind the selection and grabbing it turns the object the wrong way."""
    var tol = Float32(11.0) * (px_scale if (isfinite(px_scale) and px_scale > Float32(0.0)) else Float32(1.0))
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

    var vdir = _gizmo_view_dir(pivot, cam_pos, cam_fwd)

    if mode == GIZMO_MODE_SCALE:
        var dc = _point_seg_dist_px(mouse_x, mouse_y, c_proj[0], c_proj[1], c_proj[0], c_proj[1])
        if dc <= tol:
            return GIZMO_SCALE_UNIFORM

    if mode == GIZMO_MODE_TRANSLATE or mode == GIZMO_MODE_SCALE:
        if mode == GIZMO_MODE_TRANSLATE:
            var p_xy = project_world_to_viewport(pivot + (bx + by) * (wscale * Float32(0.28)), vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            var p_xz = project_world_to_viewport(pivot + (bx + bz) * (wscale * Float32(0.28)), vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            var p_yz = project_world_to_viewport(pivot + (by + bz) * (wscale * Float32(0.28)), vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            if p_xy[3] and gizmo_plane_usable(bz, vdir) and _point_seg_dist_px(mouse_x, mouse_y, p_xy[0], p_xy[1], p_xy[0], p_xy[1]) <= tol:
                return GIZMO_PLANE_XY
            if p_xz[3] and gizmo_plane_usable(by, vdir) and _point_seg_dist_px(mouse_x, mouse_y, p_xz[0], p_xz[1], p_xz[0], p_xz[1]) <= tol:
                return GIZMO_PLANE_XZ
            if p_yz[3] and gizmo_plane_usable(bx, vdir) and _point_seg_dist_px(mouse_x, mouse_y, p_yz[0], p_yz[1], p_yz[0], p_yz[1]) <= tol:
                return GIZMO_PLANE_YZ

        var axes = [bx, by, bz]
        var ids = [GIZMO_AXIS_X, GIZMO_AXIS_Y, GIZMO_AXIS_Z]
        var best_id = GIZMO_AXIS_NONE
        var best_d = tol + Float32(1.0)
        for a in range(3):
            if not gizmo_axis_usable(axes[a], vdir):
                continue
            var tip = project_world_to_viewport(pivot + axes[a] * wscale, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            if tip[3]:
                var d = _point_seg_dist_px(mouse_x, mouse_y, c_proj[0], c_proj[1], tip[0], tip[1])
                if d < best_d:
                    best_d = d
                    best_id = ids[a]
        return best_id

    # Rotate mode: test distance to projected 3D rings
    var ring_u = [by, bx, bx]
    var ring_v = [bz, bz, by]
    var r_ids = [GIZMO_AXIS_X, GIZMO_AXIS_Y, GIZMO_AXIS_Z]
    var best_r = GIZMO_AXIS_NONE
    var best_rd = tol
    var to_cam = cam_pos - pivot
    comptime SEG: Int = 32
    for ri in range(3):
        for s in range(SEG):
            var a0 = Float32(s) * Float32(6.2831853 / Float32(SEG))
            var a1 = Float32(s + 1) * Float32(6.2831853 / Float32(SEG))
            var p0 = pivot + (ring_u[ri] * cos(a0) + ring_v[ri] * sin(a0)) * wscale
            var p1 = pivot + (ring_u[ri] * cos(a1) + ring_v[ri] * sin(a1)) * wscale
            if ((p0 + p1) * Float32(0.5) - pivot).dot(to_cam) < -Float32(0.05) * wscale * to_cam.length():
                continue  # far half of the ring
            var s0 = project_world_to_viewport(p0, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            var s1 = project_world_to_viewport(p1, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            if s0[3] and s1[3]:
                var d = _point_seg_dist_px(mouse_x, mouse_y, s0[0], s0[1], s1[0], s1[1])
                if d < best_rd:
                    best_rd = d
                    best_r = r_ids[ri]
    if best_r == GIZMO_AXIS_NONE:
        var view_r = target_px * GIZMO_VIEW_RING_SCALE
        var dxv = mouse_x - c_proj[0]
        var dyv = mouse_y - c_proj[1]
        if abs(sqrt(dxv * dxv + dyv * dyv) - view_r) <= tol:
            return GIZMO_AXIS_VIEW
    return best_r


def gizmo_uniform_scale_screen(
    start_x: Float32, start_y: Float32, cur_x: Float32, cur_y: Float32, center_x: Float32, center_y: Float32
) -> Float32:
    """Uniform scale factor from a drag on the centre handle: how much farther the mouse is from the
    pivot on screen than where the drag started. Moving in any direction away from the pivot grows the
    selection and moving toward it shrinks it (the old solver only read the camera-right component)."""
    var d0 = sqrt((start_x - center_x) * (start_x - center_x) + (start_y - center_y) * (start_y - center_y))
    var d1 = sqrt((cur_x - center_x) * (cur_x - center_x) + (cur_y - center_y) * (cur_y - center_y))
    if not (isfinite(d0) and isfinite(d1)):
        return Float32(1.0)
    # A press right on the pivot has no lever arm: measure from a small minimum radius instead.
    var base = max(Float32(12.0), d0)
    var f = Float32(1.0) + (d1 - d0) / base
    return max(Float32(0.05), min(Float32(100.0), f)) if isfinite(f) else Float32(1.0)


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
):
    """Emits crisp analytical 3D gizmo primitives (X/Y/Z arrows, XY/XZ/YZ dual-axis planes,
    3D rotation rings, or scale cubes) into `UiDrawList` clipped to the 3D viewport."""
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
    var axes = [bx, by, bz]
    var cols = [GIZMO_COLOR_X, GIZMO_COLOR_Y, GIZMO_COLOR_Z]
    var h_ids = [GIZMO_AXIS_X, GIZMO_AXIS_Y, GIZMO_AXIS_Z]

    if mode == GIZMO_MODE_TRANSLATE or mode == GIZMO_MODE_SCALE:
        if mode == GIZMO_MODE_TRANSLATE:
            # Dual-axis planar handles (XY, XZ, YZ)
            var pl_pts = [
                pivot + (bx + by) * (wscale * Float32(0.28)),
                pivot + (bx + bz) * (wscale * Float32(0.28)),
                pivot + (by + bz) * (wscale * Float32(0.28)),
            ]
            var pl_cols = [UInt32(0xAA428BF5), UInt32(0xAA3DDC68), UInt32(0xAAE84848)]
            var pl_borders = [GIZMO_COLOR_Z, GIZMO_COLOR_Y, GIZMO_COLOR_X]
            var pl_ids = [GIZMO_PLANE_XY, GIZMO_PLANE_XZ, GIZMO_PLANE_YZ]
            var pl_normals = [bz, by, bx]
            var vdir_pl = _gizmo_view_dir(pivot, cam_pos, cam_fwd)
            for pi in range(3):
                if not gizmo_plane_usable(pl_normals[pi], vdir_pl):
                    continue
                var pp = project_world_to_viewport(pl_pts[pi], vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                if pp[3]:
                    var fc = GIZMO_COLOR_ACTIVE if active_handle == pl_ids[pi] else pl_cols[pi]
                    ui.add_rect(pp[0] - Float32(7.0), pp[1] - Float32(7.0), Float32(14.0), Float32(14.0), fc, Float32(2.0), Float32(1.5), pl_borders[pi])

        var vdir_ax = _gizmo_view_dir(pivot, cam_pos, cam_fwd)
        for a in range(3):
            if not gizmo_axis_usable(axes[a], vdir_ax):
                continue  # matches `pick_gizmo_handle`: no handle is shown that cannot be dragged
            var tip = project_world_to_viewport(pivot + axes[a] * wscale, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            if not tip[3]:
                continue
            var col = GIZMO_COLOR_ACTIVE if active_handle == h_ids[a] else cols[a]
            _emit_thick_line_ui(ui, c[0], c[1], tip[0], tip[1], Float32(3.6), col)
            if mode == GIZMO_MODE_TRANSLATE:
                # Tapered arrowhead along screen-projected shaft direction
                var dx = tip[0] - c[0]
                var dy = tip[1] - c[1]
                var slen = sqrt(dx * dx + dy * dy)
                if slen > Float32(2.0):
                    var ux = dx / slen
                    var uy = dy / slen
                    for k in range(5):
                        var back = Float32(k) * Float32(2.4)
                        var rad = max(Float32(1.8), Float32(6.5) - Float32(k) * Float32(1.1))
                        ui.add_circle(tip[0] - ux * back, tip[1] - uy * back, rad, col)
            else:
                # Scale cube tip
                ui.add_rect(tip[0] - Float32(5.5), tip[1] - Float32(5.5), Float32(11.0), Float32(11.0), col, Float32(2.0))

        var center_col = GIZMO_COLOR_ACTIVE if active_handle == GIZMO_SCALE_UNIFORM else GIZMO_COLOR_CENTER
        if mode == GIZMO_MODE_SCALE:
            ui.add_rect(c[0] - Float32(6.5), c[1] - Float32(6.5), Float32(13.0), Float32(13.0), center_col, Float32(2.5))
        else:
            ui.add_circle(c[0], c[1], Float32(5.5), center_col)
    else:
        # 3D Rotation Rings (X=YZ plane, Y=XZ plane, Z=XY plane) + subtle outer screen-space horizon ring
        var view_col = GIZMO_COLOR_ACTIVE if active_handle == GIZMO_AXIS_VIEW else UInt32(0x66DCE4F0)
        ui.add_circle(c[0], c[1], target_px * GIZMO_VIEW_RING_SCALE, UInt32(0x14FFFFFF), Float32(1.5), view_col)
        var to_cam_e = cam_pos - pivot
        var ring_u = [by, bx, bx]
        var ring_v = [bz, bz, by]
        comptime RSEG: Int = 40
        for ri in range(3):
            var rcol = GIZMO_COLOR_ACTIVE if active_handle == h_ids[ri] else cols[ri]
            for s in range(RSEG):
                var a0 = Float32(s) * Float32(6.2831853 / Float32(RSEG))
                var a1 = Float32(s + 1) * Float32(6.2831853 / Float32(RSEG))
                var p0 = pivot + (ring_u[ri] * cos(a0) + ring_v[ri] * sin(a0)) * wscale
                var p1 = pivot + (ring_u[ri] * cos(a1) + ring_v[ri] * sin(a1)) * wscale
                var s0 = project_world_to_viewport(p0, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                var s1 = project_world_to_viewport(p1, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                if s0[3] and s1[3]:
                    var back = ((p0 + p1) * Float32(0.5) - pivot).dot(to_cam_e) < -Float32(0.05) * wscale * to_cam_e.length()
                    # The far half is drawn thin and faded: it cannot be grabbed (see `pick_gizmo_handle`).
                    var seg_col = (rcol & UInt32(0x00FFFFFF)) | UInt32(0x55000000) if back else rcol
                    _emit_thick_line_ui(ui, s0[0], s0[1], s1[0], s1[1], Float32(1.4) if back else Float32(3.0), seg_col)
        ui.add_circle(c[0], c[1], Float32(4.5), GIZMO_COLOR_CENTER)

    ui.pop_clip()
