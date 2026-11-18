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
from std.math import abs, atan2, cos, floor, max, min, sin, sqrt, tan
from flint.core.math import Vec3
from flint.ui.draw_list import UiDrawList

comptime GIZMO_MODE_TRANSLATE: Int = 0
comptime GIZMO_MODE_ROTATE: Int = 1
comptime GIZMO_MODE_SCALE: Int = 2

comptime GIZMO_AXIS_NONE: Int = 0
comptime GIZMO_AXIS_X: Int = 1
comptime GIZMO_AXIS_Y: Int = 2
comptime GIZMO_AXIS_Z: Int = 3
comptime GIZMO_PLANE_XY: Int = 4
comptime GIZMO_PLANE_XZ: Int = 5
comptime GIZMO_PLANE_YZ: Int = 6
comptime GIZMO_SCALE_UNIFORM: Int = 7

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
    var rel = world_pt - cam_pos
    var fwd = cam_fwd.normalized()
    var z_v = rel.dot(fwd)
    if z_v < near_z:
        return (0.0, 0.0, z_v, False)
    var w = max(Float32(1.0), vp_w)
    var h = max(Float32(1.0), vp_h)
    var aspect = w / h
    var tan_y = max(Float32(1.0e-4), tan(fov_y_deg * Float32(0.01745329252) * Float32(0.5)))
    var tan_x = max(Float32(1.0e-4), tan_y * aspect)
    var ndc_x = rel.dot(cam_right.normalized()) / (z_v * tan_x)
    var ndc_y = rel.dot(cam_up.normalized()) / (z_v * tan_y)
    var sx = vp_x + (ndc_x * Float32(0.5) + Float32(0.5)) * w
    var sy = vp_y + (Float32(0.5) - ndc_y * Float32(0.5)) * h
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
    var fwd = cam_fwd.normalized()
    var rel = gizmo_origin - cam_pos
    var z_depth = rel.dot(fwd)
    if z_depth <= Float32(0.05):
        z_depth = max(Float32(0.05), rel.length())
    var tan_half = tan(fov_y_deg * Float32(0.01745329252) * Float32(0.5))
    var world_per_px = (Float32(2.0) * z_depth * tan_half) / Float32(max(1, viewport_h))
    return max(Float32(1.0e-4), world_per_px * target_px)


def ray_axis_closest_param(
    ray: ViewportRay,
    axis_origin: Vec3,
    axis_dir: Vec3,
) -> Tuple[Bool, Float32, Float32]:
    """Analytical skew-line closest parameter `t` along `axis_origin + t * unit(axis_dir)` to `ray`.
    Returns `(valid, axis_param_t, closest_distance)`."""
    var d = ray.dir.normalized()
    var u_len = axis_dir.length()
    if u_len <= Float32(1.0e-7):
        return (False, Float32(0.0), Float32(1.0e30))
    var u = axis_dir * (Float32(1.0) / u_len)
    var w0 = ray.origin - axis_origin
    var b = d.dot(u)
    var denom = Float32(1.0) - b * b
    if denom < Float32(1.0e-6):
        return (False, Float32(0.0), Float32(1.0e30))
    var d_w0 = d.dot(w0)
    var u_w0 = u.dot(w0)
    var s = (b * u_w0 - d_w0) / denom
    var t = (u_w0 - b * d_w0) / denom
    if s < Float32(0.0):
        return (False, t, Float32(1.0e30))
    var dist = (ray.at(s) - (axis_origin + u * t)).length()
    return (True, t, dist)


def ray_plane_hit(
    ray: ViewportRay,
    plane_origin: Vec3,
    plane_normal: Vec3,
) -> Tuple[Bool, Vec3, Float32]:
    """Intersects `ray` with `(X - plane_origin) . unit(plane_normal) = 0`.
    Returns `(hit, world_hit_point, ray_t)`."""
    var n_len = plane_normal.length()
    if n_len <= Float32(1.0e-7):
        return (False, plane_origin, Float32(0.0))
    var n = plane_normal * (Float32(1.0) / n_len)
    var d = ray.dir.normalized()
    var denom = d.dot(n)
    if abs(denom) < Float32(1.0e-6):
        return (False, plane_origin, Float32(0.0))
    var t = (plane_origin - ray.origin).dot(n) / denom
    if t < Float32(0.0):
        return (False, plane_origin, t)
    return (True, ray.at(t), t)


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
    var ax_len = ring_axis.length()
    if ax_len <= Float32(1.0e-7):
        return Float32(0.0)
    var axis = ring_axis * (Float32(1.0) / ax_len)
    var view_n = cam_fwd.normalized()
    var to_pivot = pivot - start_ray.origin
    var view_ray = to_pivot.normalized() if to_pivot.length_squared() > Float32(1.0e-8) else view_n

    var grazing_cos = abs(axis.dot(view_ray))
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
                return atan2(sin_a, cos_a)

    # Automatic `< 0.15` grazing-angle view-plane tangent fallback!
    var vp0 = ray_plane_hit(start_ray, pivot, view_n)
    var vp1 = ray_plane_hit(cur_ray, pivot, view_n)
    if not vp0[0] or not vp1[0]:
        return Float32(0.0)
    var raw_tan = axis.cross(view_n)
    var tangent = raw_tan.normalized() if raw_tan.length_squared() > Float32(1.0e-8) else Vec3(1.0, 0.0, 0.0)
    var eff_r = max(Float32(0.05), ring_world_radius)
    var disp = (vp1[1] - vp0[1]).dot(tangent)
    return max(Float32(-6.2831853), min(Float32(6.2831853), disp / eff_r))


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
) -> Vec3:
    """Solves 1D per-axis or uniform 3D scale drag relative to `start_scale`."""
    var eff_r = max(Float32(0.05), gizmo_radius)
    if handle == GIZMO_SCALE_UNIFORM:
        var vp0 = ray_plane_hit(start_ray, pivot, cam_fwd)
        var vp1 = ray_plane_hit(cur_ray, pivot, cam_fwd)
        if not vp0[0] or not vp1[0]:
            return start_scale
        var delta_u = (vp1[1] - vp0[1]).dot(cam_right.normalized()) / eff_r
        var factor = max(Float32(0.05), Float32(1.0) + delta_u)
        if snap_step > Float32(0.0):
            factor = max(snap_step, snap_scalar(factor, snap_step))
        return start_scale * factor

    var ax = Vec3(1.0, 0.0, 0.0)
    if handle == GIZMO_AXIS_Y:
        ax = Vec3(0.0, 1.0, 0.0)
    elif handle == GIZMO_AXIS_Z:
        ax = Vec3(0.0, 0.0, 1.0)
    var p0 = ray_axis_closest_param(start_ray, pivot, ax)
    var p1 = ray_axis_closest_param(cur_ray, pivot, ax)
    if not p0[0] or not p1[0]:
        return start_scale
    var ratio = max(Float32(0.05), Float32(1.0) + (p1[1] - p0[1]) / eff_r)
    var out = start_scale
    if handle == GIZMO_AXIS_X:
        out.x = max(Float32(0.01), start_scale.x * ratio)
        if snap_step > Float32(0.0):
            out.x = max(snap_step, snap_scalar(out.x, snap_step))
    elif handle == GIZMO_AXIS_Y:
        out.y = max(Float32(0.01), start_scale.y * ratio)
        if snap_step > Float32(0.0):
            out.y = max(snap_step, snap_scalar(out.y, snap_step))
    elif handle == GIZMO_AXIS_Z:
        out.z = max(Float32(0.01), start_scale.z * ratio)
        if snap_step > Float32(0.0):
            out.z = max(snap_step, snap_scalar(out.z, snap_step))
    return out


@always_inline
def snap_scalar(val: Float32, step: Float32) -> Float32:
    if step <= Float32(1.0e-6):
        return val
    return floor(val / step + Float32(0.5)) * step


@always_inline
def snap_translation(pos: Vec3, grid_step: Float32) -> Vec3:
    if grid_step <= Float32(1.0e-6):
        return pos
    return Vec3(
        snap_scalar(pos.x, grid_step),
        snap_scalar(pos.y, grid_step),
        snap_scalar(pos.z, grid_step),
    )


@always_inline
def snap_angle(angle_rad: Float32, step_rad: Float32) -> Float32:
    if step_rad <= Float32(1.0e-6):
        return angle_rad
    return snap_scalar(angle_rad, step_rad)


@always_inline
def snap_angle_deg(angle_rad: Float32, step_deg: Float32 = 15.0) -> Float32:
    return snap_angle(angle_rad, step_deg * Float32(0.01745329252))


def _point_seg_dist_px(px: Float32, py: Float32, ax: Float32, ay: Float32, bx: Float32, by: Float32) -> Float32:
    var vx = bx - ax
    var vy = by - ay
    var len2 = vx * vx + vy * vy
    if len2 <= Float32(1.0e-4):
        return sqrt((px - ax) * (px - ax) + (py - ay) * (py - ay))
    var t = max(Float32(0.0), min(Float32(1.0), ((px - ax) * vx + (py - ay) * vy) / len2))
    var cx = ax + vx * t
    var cy = ay + vy * t
    return sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy))


def _emit_thick_line_ui(
    mut ui: UiDrawList,
    x0: Float32,
    y0: Float32,
    x1: Float32,
    y1: Float32,
    thickness: Float32,
    color: UInt32,
):
    var dx = x1 - x0
    var dy = y1 - y0
    var dist = sqrt(dx * dx + dy * dy)
    var rad = max(Float32(1.2), thickness * Float32(0.5))
    var steps = max(1, min(64, Int(dist / max(Float32(2.2), rad * Float32(0.95))) + 1))
    for i in range(steps + 1):
        var t = Float32(i) / Float32(steps)
        ui.add_circle(x0 + dx * t, y0 + dy * t, rad, color)


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
) -> Int:
    """Screen-space handle picker returning `GIZMO_AXIS_*`, `GIZMO_PLANE_*`, or `GIZMO_SCALE_UNIFORM`."""
    var c_proj = project_world_to_viewport(pivot, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
    if not c_proj[3]:
        return GIZMO_AXIS_NONE
    var wscale = gizmo_world_scale(pivot, cam_pos, cam_fwd, fov_y_deg, Int(vp_h), target_px)

    if mode == GIZMO_MODE_SCALE:
        var dc = sqrt((mouse_x - c_proj[0]) * (mouse_x - c_proj[0]) + (mouse_y - c_proj[1]) * (mouse_y - c_proj[1]))
        if dc <= Float32(11.0):
            return GIZMO_SCALE_UNIFORM

    if mode == GIZMO_MODE_TRANSLATE or mode == GIZMO_MODE_SCALE:
        if mode == GIZMO_MODE_TRANSLATE:
            var p_xy = project_world_to_viewport(pivot + Vec3(wscale * 0.28, wscale * 0.28, 0.0), vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            var p_xz = project_world_to_viewport(pivot + Vec3(wscale * 0.28, 0.0, wscale * 0.28), vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            var p_yz = project_world_to_viewport(pivot + Vec3(0.0, wscale * 0.28, wscale * 0.28), vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            if p_xy[3] and sqrt((mouse_x - p_xy[0]) * (mouse_x - p_xy[0]) + (mouse_y - p_xy[1]) * (mouse_y - p_xy[1])) <= Float32(11.0):
                return GIZMO_PLANE_XY
            if p_xz[3] and sqrt((mouse_x - p_xz[0]) * (mouse_x - p_xz[0]) + (mouse_y - p_xz[1]) * (mouse_y - p_xz[1])) <= Float32(11.0):
                return GIZMO_PLANE_XZ
            if p_yz[3] and sqrt((mouse_x - p_yz[0]) * (mouse_x - p_yz[0]) + (mouse_y - p_yz[1]) * (mouse_y - p_yz[1])) <= Float32(11.0):
                return GIZMO_PLANE_YZ

        var axes = [Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0)]
        var ids = [GIZMO_AXIS_X, GIZMO_AXIS_Y, GIZMO_AXIS_Z]
        var best_id = GIZMO_AXIS_NONE
        var best_d = Float32(12.0)
        for a in range(3):
            var tip = project_world_to_viewport(pivot + axes[a] * wscale, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            if tip[3]:
                var d = _point_seg_dist_px(mouse_x, mouse_y, c_proj[0], c_proj[1], tip[0], tip[1])
                if d < best_d:
                    best_d = d
                    best_id = ids[a]
        return best_id

    # Rotate mode: test distance to projected 3D rings
    var ring_u = [Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)]
    var ring_v = [Vec3(0.0, 0.0, 1.0), Vec3(0.0, 0.0, 1.0), Vec3(0.0, 1.0, 0.0)]
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
            var s0 = project_world_to_viewport(p0, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            var s1 = project_world_to_viewport(p1, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
            if s0[3] and s1[3]:
                var d = _point_seg_dist_px(mouse_x, mouse_y, s0[0], s0[1], s1[0], s1[1])
                if d < best_rd:
                    best_rd = d
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
):
    """Emits crisp analytical 3D gizmo primitives (X/Y/Z arrows, XY/XZ/YZ dual-axis planes,
    3D rotation rings, or scale cubes) into `UiDrawList` clipped to the 3D viewport."""
    var c = project_world_to_viewport(pivot, vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
    if not c[3]:
        return
    ui.push_clip(Int(vp_x), Int(vp_y), Int(vp_x + vp_w), Int(vp_y + vp_h))

    var wscale = gizmo_world_scale(pivot, cam_pos, cam_fwd, fov_y_deg, Int(vp_h), target_px)
    var axes = [Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0)]
    var cols = [GIZMO_COLOR_X, GIZMO_COLOR_Y, GIZMO_COLOR_Z]
    var h_ids = [GIZMO_AXIS_X, GIZMO_AXIS_Y, GIZMO_AXIS_Z]

    if mode == GIZMO_MODE_TRANSLATE or mode == GIZMO_MODE_SCALE:
        if mode == GIZMO_MODE_TRANSLATE:
            # Dual-axis planar handles (XY, XZ, YZ)
            var pl_pts = [
                pivot + Vec3(wscale * Float32(0.28), wscale * Float32(0.28), 0.0),
                pivot + Vec3(wscale * Float32(0.28), 0.0, wscale * Float32(0.28)),
                pivot + Vec3(0.0, wscale * Float32(0.28), wscale * Float32(0.28)),
            ]
            var pl_cols = [UInt32(0xAA428BF5), UInt32(0xAA3DDC68), UInt32(0xAAE84848)]
            var pl_borders = [GIZMO_COLOR_Z, GIZMO_COLOR_Y, GIZMO_COLOR_X]
            var pl_ids = [GIZMO_PLANE_XY, GIZMO_PLANE_XZ, GIZMO_PLANE_YZ]
            for pi in range(3):
                var pp = project_world_to_viewport(pl_pts[pi], vp_x, vp_y, vp_w, vp_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg)
                if pp[3]:
                    var fc = GIZMO_COLOR_ACTIVE if active_handle == pl_ids[pi] else pl_cols[pi]
                    ui.add_rect(pp[0] - Float32(7.0), pp[1] - Float32(7.0), Float32(14.0), Float32(14.0), fc, Float32(2.0), Float32(1.5), pl_borders[pi])

        for a in range(3):
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
        ui.add_circle(c[0], c[1], target_px * Float32(1.06), UInt32(0x14FFFFFF), Float32(1.5), UInt32(0x66DCE4F0))
        var ring_u = [Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)]
        var ring_v = [Vec3(0.0, 0.0, 1.0), Vec3(0.0, 0.0, 1.0), Vec3(0.0, 1.0, 0.0)]
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
                    _emit_thick_line_ui(ui, s0[0], s0[1], s1[0], s1[1], Float32(3.0), rcol)
        ui.add_circle(c[0], c[1], Float32(4.5), GIZMO_COLOR_CENTER)

    ui.pop_clip()
