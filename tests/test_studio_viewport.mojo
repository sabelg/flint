"""Phase 1 tests for the unified Studio viewport coordinate model (`tests/test_studio_viewport.mojo`).

The renderer projects over the whole window; the SceneViewport panel is only the clip and input
region. These tests pin that every Studio consumer agrees with the renderer's projection:
  - Rays from the render-target centre go straight down the camera's forward axis, wherever the
    panel sits, and logical / physical rays agree at any UI scale.
  - `hud_view_proj` (NDC spanning the panel) lands a world point on the same pixel as the full
    render-target projection.
  - The gizmo pick (full target) hits the handle where the gizmo is drawn.
"""
from std.math import abs
from std.testing import assert_true

from flint.core.math import Vec3
from flint.render.render_world import make_rh_look_at, make_rh_reverse_z_perspective
from flint.studio import GIZMO_AXIS_NONE, GIZMO_AXIS_X, GIZMO_MODE_TRANSLATE, pick_gizmo_handle, project_world_to_viewport
from flint.studio.viewport_transform import ViewportTransform
from flint.ui.flexbox_dock import RectF


def _near(a: Float32, b: Float32, eps: Float32 = 1.0e-3) -> Bool:
    return abs(a - b) <= eps


def _cam() -> Tuple[Vec3, Vec3, Vec3, Vec3]:
    var pos = Vec3(0.0, 2.0, 6.0)
    var fwd = (Vec3(0.0, 0.0, 0.0) - pos).normalized()
    var right = fwd.cross(Vec3(0.0, 1.0, 0.0)).normalized()
    var up = right.cross(fwd).normalized()
    return (pos, fwd, right, up)


def test_center_ray_is_forward() raises:
    var c = _cam()
    # Panel docked to the right half: its own centre is NOT the camera centre.
    var vxf = ViewportTransform.from_logical(RectF(640.0, 40.0, 640.0, 600.0), 1280.0, 720.0, 2.0)
    var r = vxf.ray_logical(640.0, 360.0, c[0], c[1], c[2], c[3], 55.0)
    assert_true(r.dir.dot(c[1]) > Float32(0.9999), "render-target centre must look down the forward axis")
    var rp = vxf.ray_phys(1280.0, 720.0, c[0], c[1], c[2], c[3], 55.0)
    assert_true(rp.dir.dot(r.dir) > Float32(0.99999), "logical and physical rays agree at 200% UI scale")
    var off = vxf.ray_logical(960.0, 340.0, c[0], c[1], c[2], c[3], 55.0)
    assert_true(off.dir.dot(c[1]) < Float32(0.99), "panel centre is off-axis; the old panel projection was wrong")


def test_hud_matches_render_projection() raises:
    var c = _cam()
    var panel = RectF(300.0, 60.0, 700.0, 500.0)
    var vxf = ViewportTransform.from_logical(panel, 1280.0, 720.0, 1.0)
    var p = Vec3(0.7, 0.4, -0.3)
    # Reference: the renderer-style full-target projection.
    var full = project_world_to_viewport(p, 0.0, 0.0, 1280.0, 720.0, c[0], c[1], c[2], c[3], 55.0)
    assert_true(full[3])
    # The HUD path: clip = M * p, then map NDC into the panel rect.
    var m = vxf.hud_view_proj(c[0], c[1], 55.0, 0.1, 1000.0)
    var ndc = m.transform_point(p)
    var ndc_x = ndc.x
    var ndc_y = ndc.y
    var sx = panel.x + (ndc_x * 0.5 + 0.5) * panel.w
    var sy = panel.y + (0.5 - ndc_y * 0.5) * panel.h
    assert_true(_near(sx, full[0], 0.05) and _near(sy, full[1], 0.05), "HUD overlay must land on the rendered pixel")
    # Sanity: the reference agrees with the look-at/perspective pair the renderer uses.
    var view_m = make_rh_look_at(c[0], c[0] + c[1], Vec3(0.0, 1.0, 0.0))
    var proj_m = make_rh_reverse_z_perspective(55.0 * 3.14159265 / 180.0, 1280.0 / 720.0, 0.1, 1000.0)
    var c2 = proj_m.multiply(view_m).transform_point(p)
    var rx = (c2.x * 0.5 + 0.5) * 1280.0
    assert_true(_near(rx, full[0], 0.05), "gizmo projection matches the renderer camera")


def test_gizmo_pick_on_drawn_handle() raises:
    var c = _cam()
    var vxf = ViewportTransform.from_logical(RectF(640.0, 40.0, 640.0, 600.0), 1280.0, 720.0, 1.0)
    var pivot = Vec3(1.5, 0.0, 0.0)
    var ctr = project_world_to_viewport(pivot, 0.0, 0.0, vxf.proj_w(), vxf.proj_h(), c[0], c[1], c[2], c[3], 55.0)
    assert_true(ctr[3])
    # Click on the X arrow shaft, beyond the plane handles (they sit at 0.28 of the arrow length).
    var tipw = project_world_to_viewport(pivot + Vec3(0.7, 0.0, 0.0), 0.0, 0.0, vxf.proj_w(), vxf.proj_h(), c[0], c[1], c[2], c[3], 55.0)
    var h = pick_gizmo_handle(
        tipw[0], tipw[1], pivot, GIZMO_MODE_TRANSLATE, 0.0, 0.0, vxf.proj_w(), vxf.proj_h(), c[0], c[1], c[2], c[3], 55.0
    )
    assert_true(h == GIZMO_AXIS_X, "pick with the render-target rect hits the drawn X arrow")
    var miss = pick_gizmo_handle(
        ctr[0] - 300.0, ctr[1] + 200.0, pivot, GIZMO_MODE_TRANSLATE, 0.0, 0.0, vxf.proj_w(), vxf.proj_h(), c[0], c[1], c[2], c[3], 55.0
    )
    assert_true(miss == GIZMO_AXIS_NONE)


def main() raises:
    test_center_ray_is_forward()
    print("[PASS] test_center_ray_is_forward")
    test_hud_matches_render_projection()
    print("[PASS] test_hud_matches_render_projection")
    test_gizmo_pick_on_drawn_handle()
    print("[PASS] test_gizmo_pick_on_drawn_handle")
    print("[PASS] test_studio_viewport: viewport coordinates agree with the renderer")
