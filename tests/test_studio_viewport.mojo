"""Tests for the unified Studio viewport coordinate model and the viewport/gizmo/camera fixes
(`tests/test_studio_viewport.mojo`).

The renderer projects over the whole window; the SceneViewport panel is only the clip and input
region. These tests pin that every Studio consumer agrees with the renderer's projection:
  - Rays from the render-target centre go straight down the camera's forward axis, wherever the
    panel sits, and logical / physical rays agree at any UI scale.
  - `hud_view_proj` (NDC spanning the panel) lands a world point on the same pixel as the full
    render-target projection, built from the renderer's basis (`right = up x fwd`), so overlays are
    not mirrored.
  - The AAA renderer's camera basis equals the editor's `CameraRig` basis on steep views too.
  - The gizmo pick (full target) hits the handle where the gizmo is drawn; handle draw sizes follow
    the UI scale.
  - Gizmo drags keep the last valid delta on a degenerate frame, clamp runaway parameters, keep the
    sign of negative scale, snap the resulting scale, and a press without motion records nothing.
  - Camera: Alt+RMB dollies (not fly), fractional wheel deltas dolly, pan tracks the cursor, and
    Frame Selection orbits around the selection centre.
  - Picking: marquee skips hidden entities; a click on a locked entity keeps the selection; the
    software raycast handles negative scale.
"""
from std.math import abs, tan
from std.testing import assert_equal, assert_true

from flint.core.math import Quat, Vec3
from flint.platform.input import InputState
from flint.render.aaa_renderer import aaa_camera_basis
from flint.render.render_world import make_rh_reverse_z_perspective
from flint.studio import (
    FlintStudioState,
    GIZMO_AXIS_NONE,
    GIZMO_AXIS_X,
    GIZMO_MODE_SCALE,
    GIZMO_MODE_TRANSLATE,
    pick_gizmo_handle,
    project_world_to_viewport,
)
from flint.studio.gizmo import (
    GIZMO_TARGET_PX,
    emit_gizmo_draw_list,
    gizmo_drag_param_limit,
    gizmo_world_scale,
)
from flint.studio.studio_app import (
    studio_pan_world_per_px,
    studio_signed_recip,
    studio_signed_scale,
    studio_wheel_dolly_factor,
)
from flint.studio.viewport_transform import ViewportTransform, hud_view_matrix
from flint.ui.draw_list import UiDrawList
from flint.ui.flexbox_dock import RectF


def _near(a: Float32, b: Float32, eps: Float32 = 1.0e-3) -> Bool:
    return abs(a - b) <= eps


def _cam() -> Tuple[Vec3, Vec3, Vec3, Vec3]:
    """A camera with the renderer / `CameraRig` basis: right = up x fwd, up = fwd x right."""
    var pos = Vec3(0.0, 2.0, 6.0)
    var fwd = (Vec3(0.0, 0.0, 0.0) - pos).normalized()
    var right = Vec3(0.0, 1.0, 0.0).cross(fwd).normalized()
    var up = fwd.cross(right).normalized()
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
    var pts = [Vec3(0.7, 0.4, -0.3), Vec3(-1.2, 0.1, 0.5), Vec3(0.0, 1.5, 0.0)]
    var m = vxf.hud_view_proj(c[0], c[1], c[2], c[3], 55.0, 0.1, 1000.0)
    for i in range(len(pts)):
        var p = pts[i]
        # Reference: the renderer-style full-target projection.
        var full = project_world_to_viewport(p, 0.0, 0.0, 1280.0, 720.0, c[0], c[1], c[2], c[3], 55.0)
        assert_true(full[3])
        # The HUD path: clip = M * p, then map NDC into the panel rect.
        var ndc = m.transform_point(p)
        var sx = panel.x + (ndc.x * 0.5 + 0.5) * panel.w
        var sy = panel.y + (0.5 - ndc.y * 0.5) * panel.h
        assert_true(_near(sx, full[0], 0.05) and _near(sy, full[1], 0.05), "HUD overlay must land on the rendered pixel")
    # Not mirrored: a point to the camera's right lands right of the screen centre.
    var right_pt = c[0] + c[1] * Float32(5.0) + c[2] * Float32(1.0)
    var view_m = hud_view_matrix(c[0], c[1], c[2], c[3])
    var proj_m = make_rh_reverse_z_perspective(55.0 * 3.14159265 / 180.0, 1280.0 / 720.0, 0.1, 1000.0)
    var c2 = proj_m.multiply(view_m).transform_point(right_pt)
    assert_true(c2.x > Float32(0.05), "a point on cam_right projects right of centre (no mirroring)")


def test_render_basis_matches_editor_on_steep_views() raises:
    var studio = FlintStudioState()
    var pitches = [Float32(20.0), Float32(60.0), Float32(72.0), Float32(75.0), Float32(80.0), Float32(85.0), Float32(-80.0)]
    for i in range(len(pitches)):
        studio.camera_rig.yaw_deg = Float32(37.0)
        studio.camera_rig.pitch_deg = pitches[i]
        studio.update_camera(Float32(0.0))
        var b = aaa_camera_basis(studio.cam_fwd)
        assert_true(b[0].dot(studio.cam_right) > Float32(0.9999), "render right = editor right at pitch " + String(pitches[i]))
        assert_true(b[1].dot(studio.cam_up) > Float32(0.9999), "render up = editor up at pitch " + String(pitches[i]))
    # Normal pitch: bit-identical to the old renderer rule (world Y up reference).
    var fwd = Vec3(0.52, -0.26, 0.81).normalized()
    var b2 = aaa_camera_basis(fwd)
    var r_old = Vec3(0.0, 1.0, 0.0).cross(fwd).normalized()
    var u_old = fwd.cross(r_old).normalized()
    assert_true(b2[0].x == r_old.x and b2[0].y == r_old.y and b2[0].z == r_old.z, "right unchanged at normal pitch")
    assert_true(b2[1].x == u_old.x and b2[1].y == u_old.y and b2[1].z == u_old.z, "up unchanged at normal pitch")
    # Straight down still yields an orthonormal basis.
    var bd = aaa_camera_basis(Vec3(0.0, -1.0, 0.0))
    assert_true(_near(bd[0].length(), 1.0) and _near(bd[1].length(), 1.0) and abs(bd[0].dot(bd[1])) < Float32(1.0e-4))


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


def test_gizmo_draw_sizes_follow_ui_scale() raises:
    var c = _cam()
    var u1 = UiDrawList()
    var u2 = UiDrawList()
    emit_gizmo_draw_list(u1, Vec3(), GIZMO_MODE_TRANSLATE, GIZMO_AXIS_NONE, 0.0, 0.0, 1280.0, 720.0, c[0], c[1], c[2], c[3], 55.0, target_px=96.0)
    emit_gizmo_draw_list(
        u2, Vec3(), GIZMO_MODE_TRANSLATE, GIZMO_AXIS_NONE, 0.0, 0.0, 1280.0, 720.0, c[0], c[1], c[2], c[3], 55.0,
        target_px=96.0, px_scale=2.0,
    )
    assert_true(u1.count() > 0 and u2.count() > 0)
    # The last primitive is the centre dot: its radius doubles at 200%.
    var a = u1.prims[u1.count() - 1]
    var b = u2.prims[u2.count() - 1]
    assert_true(_near(b.x1 - b.x0, Float32(2.0) * (a.x1 - a.x0), 0.01), "centre dot scales with the UI scale")


def _gizmo_studio() -> FlintStudioState:
    var studio = FlintStudioState()
    studio.viewport_xf = ViewportTransform.from_logical(RectF(0.0, 0.0, 1280.0, 720.0), 1280.0, 720.0, 1.0)
    return studio^


def _x_grab(studio: FlintStudioState) -> Tuple[Float32, Float32, Float32]:
    """Screen point on the X arm of the selection's gizmo, plus the gizmo's world scale."""
    var piv = studio.selection_gizmo_pivot()
    var vx = studio.viewport_xf
    var ws = gizmo_world_scale(piv[1], studio.cam_pos, studio.cam_fwd, studio.fov_y_deg, Int(vx.proj_phys_h()), GIZMO_TARGET_PX)
    var g = project_world_to_viewport(
        piv[1] + Vec3(1.0, 0.0, 0.0) * (ws * Float32(0.8)), Float32(0.0), Float32(0.0), vx.proj_phys_w(), vx.proj_phys_h(),
        studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg,
    )
    return (g[0], g[1], ws)


def _x_vanishing_point(studio: FlintStudioState) -> Tuple[Float32, Float32]:
    """The screen point whose camera ray is parallel to the world X axis."""
    var ax = Vec3(1.0, 0.0, 0.0)
    if ax.dot(studio.cam_fwd) < Float32(0.0):
        ax = Vec3(-1.0, 0.0, 0.0)
    var a = ax.dot(studio.cam_fwd)
    var vx = studio.viewport_xf
    var tan_y = tan(studio.fov_y_deg * Float32(3.14159265 / 360.0))
    var tan_x = tan_y * (vx.proj_phys_w() / vx.proj_phys_h())
    var ndc_x = ax.dot(studio.cam_right) / (a * tan_x)
    var ndc_y = ax.dot(studio.cam_up) / (a * tan_y)
    return ((ndc_x + Float32(1.0)) * Float32(0.5) * vx.proj_phys_w(), (Float32(1.0) - ndc_y) * Float32(0.5) * vx.proj_phys_h())


def test_drag_keeps_last_delta_and_clamps() raises:
    var studio = _gizmo_studio()
    var eid = studio.spawn_entity("Dragged")
    studio._sync_ecs_and_physics()
    studio.select_entity(eid)
    studio.gizmo_mode = GIZMO_MODE_TRANSLATE
    studio.gizmo_local_space = False
    var start = studio.entities[eid].translation
    var g = _x_grab(studio)
    assert_true(studio.gizmo_try_begin(g[0], g[1]))
    _ = studio.gizmo_drag_update(g[0] + Float32(80.0), g[1])
    var moved = studio.entities[eid].translation
    assert_true(abs(moved.x - start.x) > Float32(1.0e-3), "the drag moved the entity")
    # A frame whose ray runs parallel to the axis has no solution: the object stays where it was.
    var vp = _x_vanishing_point(studio)
    _ = studio.gizmo_drag_update(vp[0], vp[1])
    assert_true(abs(studio.entities[eid].translation.x - moved.x) < Float32(1.0e-4), "a degenerate frame keeps the last delta")
    # Grazing the axis (a few pixels off the vanishing point) cannot throw the object to infinity.
    var lim = gizmo_drag_param_limit(g[2])
    for k in range(1, 6):
        _ = studio.gizmo_drag_update(vp[0] + Float32(k), vp[1] + Float32(k))
        var dx = abs(studio.entities[eid].translation.x - start.x)
        assert_true(dx <= Float32(2.0) * lim + Float32(1.0e-2), "the drag parameter is clamped")
    studio.gizmo_drag_end()


def test_press_without_motion_records_nothing() raises:
    var studio = _gizmo_studio()
    var eid = studio.spawn_entity("Still")
    studio._sync_ecs_and_physics()
    studio.select_entity(eid)
    studio.gizmo_mode = GIZMO_MODE_TRANSLATE
    studio.gizmo_local_space = False
    # Make an undoable edit and undo it, so there is something on the redo stack.
    _ = studio.set_entity_transform(eid, Vec3(0.5, 0.0, 0.0), Quat.identity(), Vec3(1.0, 1.0, 1.0))
    assert_true(studio.undo())
    studio.mark_dirty(False)
    var undo_n = len(studio.undo_stack)
    var redo_n = len(studio.redo_stack)
    assert_true(redo_n > 0)
    var g = _x_grab(studio)
    assert_true(studio.gizmo_try_begin(g[0], g[1]))
    assert_true(not studio.gizmo_drag_update(g[0], g[1]), "no motion: nothing applied")
    _ = studio.gizmo_drag_update(g[0], g[1])
    studio.gizmo_drag_end()
    assert_equal(len(studio.undo_stack), undo_n, "a click on a handle records no undo entry")
    assert_equal(len(studio.redo_stack), redo_n, "a click on a handle keeps redo")
    assert_true(not studio.dirty, "a click on a handle does not dirty the scene")
    # Esc on a motionless drag does not record the start pose either.
    assert_true(studio.gizmo_try_begin(g[0], g[1]))
    studio.gizmo_drag_cancel()
    assert_equal(len(studio.undo_stack), undo_n)
    assert_equal(len(studio.redo_stack), redo_n)


def test_scale_keeps_sign_and_snaps_result() raises:
    assert_true(studio_signed_scale(Float32(-0.001), 1.0) == Float32(-0.01), "negative scale keeps its sign")
    assert_true(studio_signed_scale(Float32(0.001), 1.0) == Float32(0.01))
    assert_true(studio_signed_scale(Float32(-1.43), 1.0, 0.25) == Float32(-1.5))
    var studio = _gizmo_studio()
    var eid = studio.spawn_entity("Mirror")
    studio._sync_ecs_and_physics()
    _ = studio.set_entity_transform(eid, Vec3(), Quat.identity(), Vec3(-2.0, 1.3, 1.0))
    studio.select_entity(eid)
    var snaps = studio.capture_selected_transforms()
    var piv = studio.selection_gizmo_pivot()
    assert_true(studio.apply_gizmo_drag_selected(snaps, piv[1], Vec3(), Quat.identity(), Vec3(0.5, 1.0, 1.0), GIZMO_MODE_SCALE))
    assert_true(_near(studio.entities[eid].scale.x, -1.0), "scaling a mirrored object keeps it mirrored")
    # Snap applies to the resulting local scale, and only on the scaled axis.
    snaps = studio.capture_selected_transforms()
    assert_true(
        studio.apply_gizmo_drag_selected(
            snaps, piv[1], Vec3(), Quat.identity(), Vec3(1.0, 1.1, 1.0), GIZMO_MODE_SCALE, scale_snap=0.25
        )
    )
    assert_true(_near(studio.entities[eid].scale.y, 1.5), "1.3 * 1.1 = 1.43 snaps to 1.5")
    assert_true(_near(studio.entities[eid].scale.x, -1.0), "an unscaled axis is not re-snapped")


def test_signed_reciprocal_for_raycast() raises:
    assert_true(_near(studio_signed_recip(-2.0), -0.5))
    assert_true(_near(studio_signed_recip(4.0), 0.25))
    assert_true(_near(studio_signed_recip(-1.0e-6), -1.0e4, 1.0))
    assert_true(_near(studio_signed_recip(0.0), 1.0e4, 1.0))


def test_camera_dolly_and_pan() raises:
    assert_true(_near(studio_wheel_dolly_factor(1.0), 0.88, 1.0e-5))
    assert_true(_near(studio_wheel_dolly_factor(-1.0), 1.0 / 0.88, 1.0e-4))
    var half = studio_wheel_dolly_factor(0.5)
    assert_true(half < Float32(1.0) and half > Float32(0.88), "a fractional notch dollies part-way")
    assert_true(_near(studio_wheel_dolly_factor(0.1), 0.98730, 1.0e-4))
    var k = studio_pan_world_per_px(10.0, 60.0, 720.0)
    assert_true(_near(k, Float32(10.0) * Float32(2.0) * tan(Float32(3.14159265 / 6.0)) / Float32(720.0), 1.0e-6))
    assert_true(_near(studio_pan_world_per_px(10.0, 60.0, 1440.0), k * Float32(0.5), 1.0e-6), "pan factor follows target height")

    # Alt+RMB dollies; it does not enter fly mode.
    var studio = _gizmo_studio()
    studio.viewport_hovered = True
    var arm0 = studio.camera_rig.spring_arm.target_length
    var inp = InputState()
    inp.move_mouse(640.0, 360.0)
    inp.set_logical_key(342, True)
    inp.set_mouse(1, True)
    inp.add_mouse_delta(0.0, 40.0)
    assert_true(studio.process_editor_input(inp, Float32(1.0 / 60.0)))
    assert_true(not studio._rmb_fly_active, "Alt+RMB is not fly mode")
    assert_true(studio.camera_rig.spring_arm.target_length > arm0 * Float32(1.1), "Alt+RMB drag dollies")


def test_frame_selection_orbits_selection_centre() raises:
    var studio = FlintStudioState()
    # Panel docked off-centre (right part of the window).
    studio.viewport_xf = ViewportTransform.from_logical(RectF(500.0, 40.0, 780.0, 640.0), 1280.0, 720.0, 1.0)
    var eid = studio.spawn_entity("Framed")
    studio._sync_ecs_and_physics()
    _ = studio.set_entity_transform(eid, Vec3(3.0, 0.5, 2.0), Quat.identity(), Vec3(1.0, 1.0, 1.0))
    studio.select_entity(eid)
    studio.frame_selection()
    var b = studio.selection_bounds()
    var centre = (b[1] + b[2]) * Float32(0.5)
    assert_true((studio.cam_target - centre).length() < Float32(1.0e-3), "the orbit pivot is the selection centre")
    var vx = studio.viewport_xf
    var pcx = vx.panel.x + vx.panel.w * Float32(0.5)
    var pcy = vx.panel.y + vx.panel.h * Float32(0.5)
    var s0 = project_world_to_viewport(centre, 0.0, 0.0, vx.proj_w(), vx.proj_h(), studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg)
    assert_true(s0[3] and _near(s0[0], pcx, 1.0) and _near(s0[1], pcy, 1.0), "framed selection sits at the panel centre")
    var d0 = (studio.cam_pos - centre).length()
    studio.update_camera(Float32(0.0), Float32(40.0), Float32(10.0))
    var d1 = (studio.cam_pos - centre).length()
    assert_true(_near(d0, d1, 1.0e-2), "orbiting keeps the distance to the selection centre")
    var s1 = project_world_to_viewport(centre, 0.0, 0.0, vx.proj_w(), vx.proj_h(), studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg)
    assert_true(s1[3] and _near(s1[0], pcx, 1.0) and _near(s1[1], pcy, 1.0), "orbiting keeps the selection under the panel centre")


def test_marquee_skips_hidden_and_locked_click_keeps_selection() raises:
    var studio = _gizmo_studio()
    var a = studio.spawn_entity("A")
    var b = studio.spawn_entity("B")
    var c = studio.spawn_entity("C")
    studio._sync_ecs_and_physics()
    var vb = studio.entities[b].copy()
    vb.visible = False
    assert_true(studio.commit_node_edit(b, vb^, "Hide"))
    var hits = studio.entities_in_screen_rect(0.0, 0.0, 1280.0, 720.0)
    var has_a = False
    var has_b = False
    for i in range(len(hits)):
        has_a = has_a or hits[i] == a
        has_b = has_b or hits[i] == b
    assert_true(has_a, "a visible entity is marquee-selectable")
    assert_true(not has_b, "a hidden entity is skipped by the marquee")

    assert_true(studio.set_entity_locked(c, True))
    studio.select_entity(a)
    assert_true(not studio.apply_viewport_click_pick(True, c, False), "a locked hit does not select")
    assert_equal(studio.selected_entity_id, a, "a click on a locked entity keeps the selection")
    assert_true(studio.apply_viewport_click_pick(False, -1, True) == False)
    assert_equal(studio.selected_entity_id, a, "an additive miss keeps the selection")
    _ = studio.apply_viewport_click_pick(False, -1, False)
    assert_equal(studio.selected_count(), 0, "a plain miss clears the selection")


def main() raises:
    test_center_ray_is_forward()
    print("[PASS] test_center_ray_is_forward")
    test_hud_matches_render_projection()
    print("[PASS] test_hud_matches_render_projection")
    test_render_basis_matches_editor_on_steep_views()
    print("[PASS] test_render_basis_matches_editor_on_steep_views")
    test_gizmo_pick_on_drawn_handle()
    print("[PASS] test_gizmo_pick_on_drawn_handle")
    test_gizmo_draw_sizes_follow_ui_scale()
    print("[PASS] test_gizmo_draw_sizes_follow_ui_scale")
    test_drag_keeps_last_delta_and_clamps()
    print("[PASS] test_drag_keeps_last_delta_and_clamps")
    test_press_without_motion_records_nothing()
    print("[PASS] test_press_without_motion_records_nothing")
    test_scale_keeps_sign_and_snaps_result()
    print("[PASS] test_scale_keeps_sign_and_snaps_result")
    test_signed_reciprocal_for_raycast()
    print("[PASS] test_signed_reciprocal_for_raycast")
    test_camera_dolly_and_pan()
    print("[PASS] test_camera_dolly_and_pan")
    test_frame_selection_orbits_selection_centre()
    print("[PASS] test_frame_selection_orbits_selection_centre")
    test_marquee_skips_hidden_and_locked_click_keeps_selection()
    print("[PASS] test_marquee_skips_hidden_and_locked_click_keeps_selection")
    print("[PASS] test_studio_viewport: viewport, gizmo, picking and camera agree with the renderer")
