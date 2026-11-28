"""`DebugLineBatch.emit_to_ui_draw_list` must project world points onto the same pixels as the AAA
renderer (`tests/test_debug_line_handedness.mojo`).

The renderer builds its view from `aaa_camera_basis` (`right = Y x fwd`, left-handed `+Z` forward,
`_build_camera_view_col_major` + `make_reverse_z_perspective`). The debug-line path used
`make_rh_look_at` (`right = fwd x up`), which is the negated right axis: lines drawn there were
mirrored horizontally about the screen centre. This test projects off-centre points through both
paths on several camera poses and requires the same screen x and y.
"""
from std.math import abs
from std.testing import assert_true

from flint.core.math import Vec3
from flint.render.aaa_renderer import _build_camera_view_col_major, aaa_camera_basis
from flint.render.aaa_visbuffer import make_reverse_z_perspective
from flint.render.render_world import DebugLineBatch
from flint.ui.draw_list import UiDrawList


def _renderer_screen(
    eye: Vec3, target: Vec3, p: Vec3, fov_deg: Float32, w: Int, h: Int
) -> Tuple[Float32, Float32]:
    """Projects `p` exactly as `AaaRenderer.begin_frame` + the globals upload do (column-major VP),
    then maps NDC to pixels with `+Y` up (`y_px = (1 - ndc_y) / 2 * H`, the HUD convention)."""
    var fwd = (target - eye).normalized()
    var basis = aaa_camera_basis(fwd)
    var view = _build_camera_view_col_major(eye, fwd, basis[0], basis[1])
    var fov_rad = fov_deg * Float32(3.14159265 / 180.0)
    var proj = make_reverse_z_perspective(fov_rad, Float32(w) / Float32(h), Float32(0.1))
    # view-space
    var vx = view[0] * p.x + view[4] * p.y + view[8] * p.z + view[12]
    var vy = view[1] * p.x + view[5] * p.y + view[9] * p.z + view[13]
    var vz = view[2] * p.x + view[6] * p.y + view[10] * p.z + view[14]
    var cx = proj[0] * vx
    var cy = proj[5] * vy
    var cw = proj[11] * vz
    var ndc_x = cx / cw
    var ndc_y = cy / cw
    return (
        (ndc_x + Float32(1.0)) * Float32(0.5) * Float32(w),
        (Float32(1.0) - ndc_y) * Float32(0.5) * Float32(h),
    )


def _debug_line_screen(
    eye: Vec3, target: Vec3, p: Vec3, fov_deg: Float32, w: Int, h: Int
) raises -> Tuple[Float32, Float32]:
    var batch = DebugLineBatch()
    # A short segment starting at `p`; endpoint 0 of the emitted PRIM_LINE is `p`.
    batch.add_line(p, p + (target - eye).normalized() * Float32(0.001), 0xFFFFFFFF, 2.0)
    var ui = UiDrawList()
    var n = batch.emit_to_ui_draw_list(ui, eye, target, fov_deg, Float32(0.1), w, h, depth_cue=False)
    assert_true(n == 1, "debug segment in front of the camera is emitted")
    return (ui.prims[0].u0, ui.prims[0].v0)


def _check(eye: Vec3, target: Vec3, p: Vec3, label: String) raises:
    var w = 1280
    var h = 720
    var r = _renderer_screen(eye, target, p, 55.0, w, h)
    var d = _debug_line_screen(eye, target, p, 55.0, w, h)
    var cx = Float32(w) * Float32(0.5)
    print(
        "  ", label, ": renderer x=", r[0], " y=", r[1], " | debug-line x=", d[0], " y=", d[1],
        " | mirrored-about-centre x would be ", Float32(2.0) * cx - r[0],
    )
    assert_true(abs(r[0] - cx) > Float32(20.0), "test point is off-centre horizontally: " + label)
    assert_true(abs(d[0] - r[0]) < Float32(0.05), "debug-line x matches the renderer (not mirrored): " + label)
    assert_true(abs(d[1] - r[1]) < Float32(0.05), "debug-line y matches the renderer: " + label)


def main() raises:
    _check(Vec3(0.0, 2.0, 6.0), Vec3(0.0, 0.5, 0.0), Vec3(1.5, 0.5, 0.0), "looking -Z, point at +X")
    _check(Vec3(0.0, 1.5, -5.0), Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.3, 0.0), "looking +Z, point at +X")
    _check(Vec3(7.0, 4.0, 3.0), Vec3(0.0, 1.0, 0.0), Vec3(0.5, 1.2, -1.5), "oblique pose")
    _check(Vec3(0.2, 9.0, 0.3), Vec3(0.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.5), "steep top-down pose")
    print("[PASS] test_debug_line_handedness: debug lines land on the renderer's pixels")
