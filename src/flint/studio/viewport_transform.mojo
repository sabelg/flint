"""One coordinate model for the Studio 3D viewport (`flint.studio.viewport_transform`).

The scene renderer draws into the whole render target (the window), with the window's aspect ratio
and centre, and the dock panels are painted on top; the "SceneViewport" panel is the hole the user
looks through. Every tool that turns a mouse position into a ray, or a world point into a screen
point, therefore has to project with the *render target*, not with the panel rect: the panel is only
a clip and input region. Before this module each call site picked its own rect (panel or target)
and its own pixel space (logical UI points or physical pixels), so the gizmo, agent overlays, drop
placement and GPU picks disagreed whenever panels were docked or the UI scale was not 100%.

`ViewportTransform` carries both rects in logical UI points plus the UI scale, and hands out the
numbers each consumer needs:
  - `proj_*` / `proj_phys_*`: the projection rect (render target) in logical / physical pixels.
  - `panel` / `panel_phys()`: the clip / hit-test rect in logical / physical pixels.
  - `ray_logical`, `ray_phys`: camera rays from a mouse position.
  - `hud_view_proj`: a view-projection whose NDC spans the panel, for overlays that lay themselves
    out inside the panel rect but must land on the rendered pixels.
"""
from std.math import isfinite, max
from flint.core.math import Mat4, Vec3
from flint.render.render_world import make_rh_reverse_z_perspective
from flint.studio.gizmo import ViewportRay
from flint.ui.flexbox_dock import RectF


def hud_view_matrix(cam_pos: Vec3, cam_fwd: Vec3, cam_right: Vec3, cam_up: Vec3) -> Mat4:
    """RH `-Z`-forward view matrix (for `make_rh_reverse_z_perspective`) whose `+X` is the renderer's
    `cam_right` and `+Y` its `cam_up`, so projected points land where the renderer draws them."""
    var r = cam_right.normalized()
    var u = cam_up.normalized()
    var b = -cam_fwd.normalized()
    var m = Mat4()
    m.set(0, 0, r.x); m.set(0, 1, r.y); m.set(0, 2, r.z); m.set(0, 3, -r.dot(cam_pos))
    m.set(1, 0, u.x); m.set(1, 1, u.y); m.set(1, 2, u.z); m.set(1, 3, -u.dot(cam_pos))
    m.set(2, 0, b.x); m.set(2, 1, b.y); m.set(2, 2, b.z); m.set(2, 3, -b.dot(cam_pos))
    m.set(3, 0, Float32(0.0)); m.set(3, 1, Float32(0.0)); m.set(3, 2, Float32(0.0)); m.set(3, 3, Float32(1.0))
    return m^


@fieldwise_init
struct ViewportTransform(Copyable, ImplicitlyCopyable, Movable):
    var panel: RectF
    """The SceneViewport panel in logical UI points (clip + input region)."""
    var target_w: Float32
    """Render-target width in logical UI points (the renderer draws the whole window)."""
    var target_h: Float32
    var ui_scale: Float32
    """Physical pixels per logical UI point."""

    @staticmethod
    def from_logical(panel: RectF, screen_w_logical: Float32, screen_h_logical: Float32, ui_scale: Float32) -> Self:
        var s = ui_scale if (isfinite(ui_scale) and ui_scale > Float32(0.0)) else Float32(1.0)
        return Self(panel, max(Float32(1.0), screen_w_logical), max(Float32(1.0), screen_h_logical), s)

    @staticmethod
    def from_physical(panel_phys: RectF, screen_w_phys: Int, screen_h_phys: Int, ui_scale: Float32) -> Self:
        var s = ui_scale if (isfinite(ui_scale) and ui_scale > Float32(0.0)) else Float32(1.0)
        var p = RectF(panel_phys.x / s, panel_phys.y / s, panel_phys.w / s, panel_phys.h / s)
        return Self(p, Float32(max(1, screen_w_phys)) / s, Float32(max(1, screen_h_phys)) / s, s)

    # ---- Projection rect (what the renderer's camera covers) ----
    @always_inline
    def proj_x(self) -> Float32:
        return Float32(0.0)

    @always_inline
    def proj_y(self) -> Float32:
        return Float32(0.0)

    @always_inline
    def proj_w(self) -> Float32:
        return self.target_w

    @always_inline
    def proj_h(self) -> Float32:
        return self.target_h

    @always_inline
    def proj_phys_w(self) -> Float32:
        return self.target_w * self.ui_scale

    @always_inline
    def proj_phys_h(self) -> Float32:
        return self.target_h * self.ui_scale

    def panel_phys(self) -> RectF:
        var s = self.ui_scale
        return RectF(self.panel.x * s, self.panel.y * s, self.panel.w * s, self.panel.h * s)

    # ---- Rays ----
    def ray_logical(
        self, x: Float32, y: Float32, cam_pos: Vec3, cam_fwd: Vec3, cam_right: Vec3, cam_up: Vec3, fov_y_deg: Float32
    ) -> ViewportRay:
        return ViewportRay.from_screen(
            x, y, Float32(0.0), Float32(0.0), self.target_w, self.target_h, cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg
        )

    def ray_phys(
        self, x: Float32, y: Float32, cam_pos: Vec3, cam_fwd: Vec3, cam_right: Vec3, cam_up: Vec3, fov_y_deg: Float32
    ) -> ViewportRay:
        return ViewportRay.from_screen(
            x, y, Float32(0.0), Float32(0.0), self.proj_phys_w(), self.proj_phys_h(),
            cam_pos, cam_fwd, cam_right, cam_up, fov_y_deg,
        )

    # ---- Overlays laid out in the panel ----
    def hud_view_proj(
        self,
        cam_pos: Vec3,
        cam_fwd: Vec3,
        cam_right: Vec3,
        cam_up: Vec3,
        fov_y_deg: Float32,
        near_z: Float32,
        far_z: Float32,
    ) -> Mat4:
        """View-projection with the renderer's (render-target) aspect, re-mapped so NDC -1..1 spans the
        panel rect. Overlays that map NDC into `panel` then land exactly on the rendered pixels.

        The view is built from the renderer's own basis (`cam_right = up x fwd`, as in the AAA renderer
        and `CameraRig`), not from `make_rh_look_at` (whose right is `fwd x up`, which mirrors overlays)."""
        var fov_rad = fov_y_deg * Float32(3.14159265358979 / 180.0)
        var view_m = hud_view_matrix(cam_pos, cam_fwd, cam_right, cam_up)
        var proj_m = make_rh_reverse_z_perspective(fov_rad, self.target_w / self.target_h, near_z, far_z)
        var pw = max(Float32(1.0), self.panel.w)
        var ph = max(Float32(1.0), self.panel.h)
        # Screen x_px = (ndc+1)/2 * W; panel ndc' = 2(x_px - px)/pw - 1  =>  ndc' = ndc*W/pw + (W - 2px)/pw - 1.
        # Screen y_px = (1-ndc)/2 * H (y down); panel ndc'_y = 1 - 2(y_px - py)/ph => ndc'_y = ndc*H/ph + 1 - (H - 2py)/ph.
        var remap = Mat4()
        remap.set(0, 0, self.target_w / pw)
        remap.set(0, 3, (self.target_w - Float32(2.0) * self.panel.x) / pw - Float32(1.0))
        remap.set(1, 1, self.target_h / ph)
        remap.set(1, 3, Float32(1.0) - (self.target_h - Float32(2.0) * self.panel.y) / ph)
        return remap.multiply(proj_m.multiply(view_m))
