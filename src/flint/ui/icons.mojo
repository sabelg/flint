"""Resolution-Independent Vector/SDF Studio Icons (`flint.ui.icons`).

Emits crisp, optically balanced icons for every `UiIconId` (`ICON_*` in `flint.ui.theme`)
directly into `UiDrawList` using stable analytical primitives (`add_rect`, `add_circle`,
`add_line`, `add_quad_bezier`, `add_polygon`, `add_arc`) with optical stroke scaling and
orthogonal pixel-grid snapping so icons render identically across GPU compute overlay and
Tier-0 CPU SIMD rasterizer at any scale (`12px`..`48px`).
"""
from std.collections import List
from std.math import cos, floor, isfinite, sin
from flint.ui.draw_list import UiDrawList
from flint.ui.theme import (
    ICON_AUDIO,
    ICON_BELL,
    ICON_CAMERA,
    ICON_CHECK,
    ICON_CHEVRON,
    ICON_CLOCK,
    ICON_CLOSE,
    ICON_CODE,
    ICON_COLLIDER,
    ICON_CUBE,
    ICON_DIFF,
    ICON_EMPTY,
    ICON_ERROR_CIRCLE,
    ICON_EYE,
    ICON_EYE_OFF,
    ICON_FILE,
    ICON_FOCUS_CAM,
    ICON_FOLDER,
    ICON_FOLIAGE,
    ICON_GRID,
    ICON_LIGHT_POINT,
    ICON_LIGHT_SPOT,
    ICON_LOCAL,
    ICON_LOCK,
    ICON_MATERIAL,
    ICON_MOON,
    ICON_MORE_V,
    ICON_MOVE,
    ICON_PAUSE,
    ICON_PLAY,
    ICON_PLUS,
    ICON_PREFAB,
    ICON_REFRESH,
    ICON_REVERT,
    ICON_ROTATE,
    ICON_SCALE,
    ICON_SCENE,
    ICON_SEARCH,
    ICON_SNAP,
    ICON_SPARK,
    ICON_STEP,
    ICON_STOP,
    ICON_SUN,
    ICON_WARN,
    ICON_WORLD,
    rgba8_scale_alpha,
)


@always_inline
def _px(origin: Float32, size: Float32, norm: Float32) -> Float32:
    """Maps normalized `[0.0 .. 1.0]` icon coordinate to screen pixels."""
    return origin + norm * size


@always_inline
def _snap_axis(v: Float32, stroke: Float32) -> Float32:
    """Snaps an orthogonal stroke coordinate to pixel center (odd) or pixel edge (even) for razor-sharp AA."""
    if stroke < Float32(1.75):
        return floor(v) + Float32(0.5)
    return floor(v + Float32(0.5))


@always_inline
def _optical_stroke(size: Float32, stroke_px: Float32) -> Float32:
    """Computes optically uniform stroke width across `12px..48px` (scales gracefully above 18px)."""
    var base: Float32
    if not isfinite(stroke_px) or stroke_px <= Float32(0.0):
        base = size * Float32(0.105)
    else:
        base = stroke_px * max(Float32(1.0), size / Float32(18.0))
    return max(Float32(1.1), min(size * Float32(0.22), base))


@always_inline
def _add_seg(
    mut dl: UiDrawList,
    x: Float32,
    y: Float32,
    s: Float32,
    ax: Float32,
    ay: Float32,
    bx: Float32,
    by: Float32,
    t: Float32,
    col: UInt32,
):
    dl.add_line(_px(x, s, ax), _px(y, s, ay), _px(x, s, bx), _px(y, s, by), t, col)


@always_inline
def _add_hseg(
    mut dl: UiDrawList,
    x: Float32,
    y: Float32,
    s: Float32,
    ax: Float32,
    bx: Float32,
    ny: Float32,
    t: Float32,
    col: UInt32,
):
    var py = _snap_axis(_px(y, s, ny), t)
    dl.add_line(_px(x, s, ax), py, _px(x, s, bx), py, t, col)


@always_inline
def _add_vseg(
    mut dl: UiDrawList,
    x: Float32,
    y: Float32,
    s: Float32,
    nx: Float32,
    ay: Float32,
    by: Float32,
    t: Float32,
    col: UInt32,
):
    var px = _snap_axis(_px(x, s, nx), t)
    dl.add_line(px, _px(y, s, ay), px, _px(y, s, by), t, col)


@always_inline
def _add_bez(
    mut dl: UiDrawList,
    x: Float32,
    y: Float32,
    s: Float32,
    ax: Float32,
    ay: Float32,
    cx: Float32,
    cy: Float32,
    bx: Float32,
    by: Float32,
    t: Float32,
    col: UInt32,
):
    dl.add_quad_bezier(
        _px(x, s, ax),
        _px(y, s, ay),
        _px(x, s, cx),
        _px(y, s, cy),
        _px(x, s, bx),
        _px(y, s, by),
        t,
        col,
    )


def _add_poly_norm(
    mut dl: UiDrawList,
    x: Float32,
    y: Float32,
    s: Float32,
    pts_norm: List[Float32],
    col: UInt32,
):
    var n_pts = len(pts_norm) // 2
    if n_pts < 3:
        return
    var xy = List[Float32](capacity=n_pts * 2)
    for i in range(n_pts):
        xy.append(_px(x, s, pts_norm[2 * i]))
        xy.append(_px(y, s, pts_norm[2 * i + 1]))
    var counts = List[Int](capacity=1)
    counts.append(n_pts)
    _ = dl.add_polygon(xy, counts, col)


def _add_tri_norm(
    mut dl: UiDrawList,
    x: Float32,
    y: Float32,
    s: Float32,
    x0: Float32,
    y0: Float32,
    x1: Float32,
    y1: Float32,
    x2: Float32,
    y2: Float32,
    col: UInt32,
):
    var pts = List[Float32](capacity=6)
    pts.append(x0); pts.append(y0)
    pts.append(x1); pts.append(y1)
    pts.append(x2); pts.append(y2)
    _add_poly_norm(dl, x, y, s, pts, col)


def _draw_circle_outline(
    mut dl: UiDrawList,
    cx: Float32,
    cy: Float32,
    radius: Float32,
    stroke: Float32,
    col: UInt32,
):
    """Draws a crisp analytical SDF ring via a single `PRIM_ARC` primitive."""
    var r = max(Float32(1.0), radius)
    dl.add_arc(cx, cy, r, max(Float32(1.0), stroke), Float32(0.0), Float32(6.2831853), col)


def draw_studio_chevron(
    mut dl: UiDrawList,
    x: Float32,
    y: Float32,
    size: Float32,
    color_rgba8: UInt32,
    open_t: Float32 = 0.0,
    stroke_px: Float32 = 1.5,
):
    """Draws a chevron rotating smoothly from right-pointing (`open_t = 0.0`) to down-pointing (`open_t = 1.0`)."""
    if not isfinite(size) or size <= Float32(1.0):
        return
    var t = max(Float32(0.0), min(Float32(1.0), open_t)) if isfinite(open_t) else Float32(0.0)
    var angle = t * Float32(1.5707963)
    var ca = cos(angle)
    var sa = sin(angle)
    var cx = x + size * Float32(0.5)
    var cy = y + size * Float32(0.5)
    var arm = size * Float32(0.23)
    var tip = size * Float32(0.15)
    var st = _optical_stroke(size, stroke_px)

    # Local right-pointing chevron vertices: (-tip, -arm) -> (+tip, 0) -> (-tip, +arm)
    var p0x = cx + (-tip * ca - (-arm) * sa)
    var p0y = cy + (-tip * sa + (-arm) * ca)
    var p1x = cx + (tip * ca)
    var p1y = cy + (tip * sa)
    var p2x = cx + (-tip * ca - arm * sa)
    var p2y = cy + (-tip * sa + arm * ca)
    dl.add_line(p0x, p0y, p1x, p1y, st, color_rgba8)
    dl.add_line(p1x, p1y, p2x, p2y, st, color_rgba8)


def draw_studio_icon(
    mut dl: UiDrawList,
    icon_id: Int,
    x: Float32,
    y: Float32,
    size: Float32,
    color_rgba8: UInt32,
    stroke_px: Float32 = 1.5,
):
    """Emits a resolution-independent vector/SDF Studio icon inside `(x, y, size, size)`."""
    if not isfinite(x) or not isfinite(y) or not isfinite(size) or size <= Float32(1.0):
        return
    var st = _optical_stroke(size, stroke_px)
    var st_thin = max(Float32(1.0), st * Float32(0.80))
    var col = color_rgba8
    var col_sub = rgba8_scale_alpha(col, Float32(0.76))
    var s = size

    # ---- Transport (0..3) ----
    if icon_id == ICON_PLAY:
        _add_tri_norm(dl, x, y, s, 0.28, 0.18, 0.82, 0.50, 0.28, 0.82, col)
        return

    if icon_id == ICON_PAUSE:
        var bw = max(Float32(2.0), s * Float32(0.22))
        var bh = s * Float32(0.64)
        var rr = min(Float32(2.0), bw * Float32(0.35))
        dl.add_rect(_px(x, s, 0.22), _px(y, s, 0.18), bw, bh, col, rr)
        dl.add_rect(_px(x, s, 0.56), _px(y, s, 0.18), bw, bh, col, rr)
        return

    if icon_id == ICON_STEP:
        _add_tri_norm(dl, x, y, s, 0.18, 0.20, 0.62, 0.50, 0.18, 0.80, col)
        var bw = max(Float32(2.0), s * Float32(0.14))
        dl.add_rect(_px(x, s, 0.70), _px(y, s, 0.20), bw, s * Float32(0.60), col, Float32(1.2))
        return

    if icon_id == ICON_STOP:
        dl.add_rect(_px(x, s, 0.22), _px(y, s, 0.22), s * Float32(0.56), s * Float32(0.56), col, max(Float32(1.5), s * Float32(0.12)))
        return

    # ---- Gizmo & Viewport (4..11) ----
    if icon_id == ICON_MOVE:
        # Clean pixel-snapped cross shafts + 4 sharp filled dart arrowheads (zero clumping at 13px)
        _add_vseg(dl, x, y, s, 0.50, 0.24, 0.76, st * Float32(0.95), col)
        _add_hseg(dl, x, y, s, 0.24, 0.76, 0.50, st * Float32(0.95), col)
        _add_tri_norm(dl, x, y, s, 0.50, 0.10, 0.64, 0.27, 0.36, 0.27, col)  # Up
        _add_tri_norm(dl, x, y, s, 0.50, 0.90, 0.64, 0.73, 0.36, 0.73, col)  # Down
        _add_tri_norm(dl, x, y, s, 0.10, 0.50, 0.27, 0.36, 0.27, 0.64, col)  # Left
        _add_tri_norm(dl, x, y, s, 0.90, 0.50, 0.73, 0.36, 0.73, 0.64, col)  # Right
        return

    if icon_id == ICON_ROTATE:
        # Analytical 260-degree SDF arc anchored cleanly to a tangent-aligned filled arrowhead at top
        var cx = _px(x, s, 0.50)
        var cy = _px(y, s, 0.53)
        var r = s * Float32(0.31)
        dl.add_arc(cx, cy, r, st, Float32(0.12), Float32(4.55), col)
        # Arc ends near top (0.50, 0.22) traveling clockwise (+x)
        _add_tri_norm(dl, x, y, s, 0.67, 0.22, 0.45, 0.11, 0.46, 0.33, col)
        return

    if icon_id == ICON_SCALE:
        # Canonical anchor square (bottom-left) + expanding diagonal arrow to top-right corner bracket
        dl.add_rect(
            _px(x, s, 0.15), _px(y, s, 0.55), s * Float32(0.30), s * Float32(0.30),
            rgba8_scale_alpha(col, Float32(0.24)), Float32(1.5), st, col,
        )
        _add_seg(dl, x, y, s, 0.47, 0.53, 0.80, 0.20, st, col)
        _add_hseg(dl, x, y, s, 0.55, 0.82, 0.18, st, col)
        _add_vseg(dl, x, y, s, 0.82, 0.18, 0.45, st, col)
        return

    if icon_id == ICON_WORLD:
        # Airy globe: crisp outer ring + subtle equator + balanced meridian ellipse
        var cx = _px(x, s, 0.50)
        var cy = _px(y, s, 0.50)
        _draw_circle_outline(dl, cx, cy, s * Float32(0.36), st, col)
        _add_hseg(dl, x, y, s, 0.16, 0.84, 0.50, st_thin, col_sub)
        _add_bez(dl, x, y, s, 0.50, 0.15, 0.32, 0.50, 0.50, 0.85, st_thin, col_sub)
        _add_bez(dl, x, y, s, 0.50, 0.15, 0.68, 0.50, 0.50, 0.85, st_thin, col_sub)
        return

    if icon_id == ICON_LOCAL:
        _add_vseg(dl, x, y, s, 0.42, 0.16, 0.58, st, col)
        _add_hseg(dl, x, y, s, 0.42, 0.84, 0.58, st, col)
        _add_seg(dl, x, y, s, 0.42, 0.58, 0.18, 0.82, st, col)
        dl.add_circle(_px(x, s, 0.42), _px(y, s, 0.58), max(Float32(1.6), s * Float32(0.095)), col)
        return

    if icon_id == ICON_SNAP:
        # Clean U-magnet with semicircular analytical base and flush pole tips (zero protruding ears)
        var leg_st = st * Float32(1.15)
        var cx = _px(x, s, 0.50)
        var cy = _px(y, s, 0.54)
        var r = s * Float32(0.25)
        dl.add_arc(cx, cy, r, leg_st, Float32(0.0), Float32(3.1415926), col)
        _add_vseg(dl, x, y, s, 0.25, 0.34, 0.54, leg_st, col)
        _add_vseg(dl, x, y, s, 0.75, 0.34, 0.54, leg_st, col)
        # Flush pole caps separated by a crisp gap
        _add_vseg(dl, x, y, s, 0.25, 0.18, 0.24, leg_st, col_sub)
        _add_vseg(dl, x, y, s, 0.75, 0.18, 0.24, leg_st, col_sub)
        return

    if icon_id == ICON_CAMERA:
        # Crisp camera body + balanced frustum cone (no muddied inner ring at small sizes)
        dl.add_rect(_px(x, s, 0.12), _px(y, s, 0.26), s * Float32(0.48), s * Float32(0.48), 0, Float32(2.0), st, col)
        if s >= Float32(16.0):
            dl.add_circle(_px(x, s, 0.36), _px(y, s, 0.50), max(Float32(1.3), s * Float32(0.085)), col_sub)
        var cone = List[Float32](capacity=8)
        cone.append(0.62); cone.append(0.42)
        cone.append(0.88); cone.append(0.26)
        cone.append(0.88); cone.append(0.74)
        cone.append(0.62); cone.append(0.58)
        _add_poly_norm(dl, x, y, s, cone, rgba8_scale_alpha(col, Float32(0.88)))
        return

    if icon_id == ICON_GRID:
        # Pixel-snapped 3x3 grid frame
        var x0 = _snap_axis(_px(x, s, 0.15), st)
        var y0 = _snap_axis(_px(y, s, 0.15), st)
        var x1 = _snap_axis(_px(x, s, 0.85), st)
        var y1 = _snap_axis(_px(y, s, 0.85), st)
        dl.add_rect(x0, y0, max(Float32(2.0), x1 - x0), max(Float32(2.0), y1 - y0), 0, Float32(1.8), st, col)
        _add_vseg(dl, x, y, s, 0.39, 0.17, 0.83, st_thin, col_sub)
        _add_vseg(dl, x, y, s, 0.61, 0.17, 0.83, st_thin, col_sub)
        _add_hseg(dl, x, y, s, 0.17, 0.83, 0.39, st_thin, col_sub)
        _add_hseg(dl, x, y, s, 0.17, 0.83, 0.61, st_thin, col_sub)
        return

    # ---- Entities & Components (12..19) ----
    if icon_id == ICON_CUBE:
        # Crisp isometric cube silhouette + softer interior Y edges (eliminates center overdraw knot)
        _add_seg(dl, x, y, s, 0.50, 0.12, 0.84, 0.30, st, col)
        _add_seg(dl, x, y, s, 0.84, 0.30, 0.84, 0.70, st, col)
        _add_seg(dl, x, y, s, 0.84, 0.70, 0.50, 0.88, st, col)
        _add_seg(dl, x, y, s, 0.50, 0.88, 0.16, 0.70, st, col)
        _add_seg(dl, x, y, s, 0.16, 0.70, 0.16, 0.30, st, col)
        _add_seg(dl, x, y, s, 0.16, 0.30, 0.50, 0.12, st, col)
        var inner_col = rgba8_scale_alpha(col, Float32(0.70))
        _add_seg(dl, x, y, s, 0.19, 0.32, 0.50, 0.48, st_thin, inner_col)
        _add_seg(dl, x, y, s, 0.81, 0.32, 0.50, 0.48, st_thin, inner_col)
        _add_seg(dl, x, y, s, 0.50, 0.48, 0.50, 0.85, st_thin, inner_col)
        return

    if icon_id == ICON_LIGHT_POINT:
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.50), max(Float32(2.0), s * Float32(0.16)), col)
        _add_vseg(dl, x, y, s, 0.50, 0.12, 0.24, st, col)
        _add_vseg(dl, x, y, s, 0.50, 0.76, 0.88, st, col)
        _add_hseg(dl, x, y, s, 0.12, 0.24, 0.50, st, col)
        _add_hseg(dl, x, y, s, 0.76, 0.88, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.23, 0.23, 0.31, 0.31, st, col)
        _add_seg(dl, x, y, s, 0.69, 0.69, 0.77, 0.77, st, col)
        _add_seg(dl, x, y, s, 0.77, 0.23, 0.69, 0.31, st, col)
        _add_seg(dl, x, y, s, 0.31, 0.69, 0.23, 0.77, st, col)
        return

    if icon_id == ICON_LIGHT_SPOT:
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.24), max(Float32(2.0), s * Float32(0.11)), col)
        _add_seg(dl, x, y, s, 0.42, 0.30, 0.18, 0.76, st, col)
        _add_seg(dl, x, y, s, 0.58, 0.30, 0.82, 0.76, st, col)
        _add_bez(dl, x, y, s, 0.18, 0.76, 0.50, 0.92, 0.82, 0.76, st, col)
        _add_bez(dl, x, y, s, 0.18, 0.76, 0.50, 0.64, 0.82, 0.76, st_thin, col_sub)
        return

    if icon_id == ICON_AUDIO:
        var spk = List[Float32](capacity=12)
        spk.append(0.14); spk.append(0.38)
        spk.append(0.32); spk.append(0.38)
        spk.append(0.52); spk.append(0.20)
        spk.append(0.52); spk.append(0.80)
        spk.append(0.32); spk.append(0.62)
        spk.append(0.14); spk.append(0.62)
        _add_poly_norm(dl, x, y, s, spk, col)
        _add_bez(dl, x, y, s, 0.64, 0.36, 0.74, 0.50, 0.64, 0.64, st, col)
        _add_bez(dl, x, y, s, 0.74, 0.24, 0.90, 0.50, 0.74, 0.76, st, col)
        return

    if icon_id == ICON_FOLIAGE:
        _add_bez(dl, x, y, s, 0.20, 0.80, 0.20, 0.22, 0.82, 0.18, st, col)
        _add_bez(dl, x, y, s, 0.82, 0.18, 0.80, 0.78, 0.20, 0.80, st, col)
        _add_seg(dl, x, y, s, 0.18, 0.82, 0.66, 0.34, st * Float32(0.85), col_sub)
        return

    if icon_id == ICON_EMPTY:
        # Pixel-snapped corner brackets + center origin dot
        _add_vseg(dl, x, y, s, 0.18, 0.18, 0.34, st, col)
        _add_hseg(dl, x, y, s, 0.18, 0.34, 0.18, st, col)
        _add_hseg(dl, x, y, s, 0.66, 0.82, 0.18, st, col)
        _add_vseg(dl, x, y, s, 0.82, 0.18, 0.34, st, col)
        _add_vseg(dl, x, y, s, 0.82, 0.66, 0.82, st, col)
        _add_hseg(dl, x, y, s, 0.66, 0.82, 0.82, st, col)
        _add_hseg(dl, x, y, s, 0.18, 0.34, 0.82, st, col)
        _add_vseg(dl, x, y, s, 0.18, 0.66, 0.82, st, col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.50), max(Float32(1.5), s * Float32(0.10)), col)
        return

    if icon_id == ICON_PREFAB:
        _add_seg(dl, x, y, s, 0.50, 0.12, 0.86, 0.32, st, col)
        _add_seg(dl, x, y, s, 0.86, 0.32, 0.86, 0.68, st, col)
        _add_seg(dl, x, y, s, 0.86, 0.68, 0.50, 0.88, st, col)
        _add_seg(dl, x, y, s, 0.50, 0.88, 0.14, 0.68, st, col)
        _add_seg(dl, x, y, s, 0.14, 0.68, 0.14, 0.32, st, col)
        _add_seg(dl, x, y, s, 0.14, 0.32, 0.50, 0.12, st, col)
        var core = List[Float32](capacity=8)
        core.append(0.50); core.append(0.32)
        core.append(0.68); core.append(0.50)
        core.append(0.50); core.append(0.68)
        core.append(0.32); core.append(0.50)
        _add_poly_norm(dl, x, y, s, core, col)
        return

    if icon_id == ICON_COLLIDER:
        # Balanced 3D physics capsule wireframe with gently curved equatorial hemisphere rings
        var cr = s * Float32(0.26)
        dl.add_rect(_px(x, s, 0.24), _px(y, s, 0.12), s * Float32(0.52), s * Float32(0.76), 0, cr, st, col)
        _add_bez(dl, x, y, s, 0.26, 0.38, 0.50, 0.45, 0.74, 0.38, st_thin, col_sub)
        _add_bez(dl, x, y, s, 0.26, 0.62, 0.50, 0.69, 0.74, 0.62, st_thin, col_sub)
        return

    # ---- Chrome & Actions (20..27) ----
    if icon_id == ICON_EYE:
        _add_bez(dl, x, y, s, 0.12, 0.50, 0.50, 0.14, 0.88, 0.50, st, col)
        _add_bez(dl, x, y, s, 0.12, 0.50, 0.50, 0.86, 0.88, 0.50, st, col)
        dl.add_arc(_px(x, s, 0.50), _px(y, s, 0.50), max(Float32(1.8), s * Float32(0.14)), max(Float32(1.0), st * Float32(0.9)), Float32(0.0), Float32(6.2831853), col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.50), max(Float32(1.0), s * Float32(0.055)), col)
        return

    if icon_id == ICON_EYE_OFF:
        _add_bez(dl, x, y, s, 0.14, 0.50, 0.50, 0.18, 0.86, 0.50, st * Float32(0.9), col_sub)
        _add_bez(dl, x, y, s, 0.14, 0.50, 0.50, 0.82, 0.86, 0.50, st * Float32(0.9), col_sub)
        _add_seg(dl, x, y, s, 0.18, 0.82, 0.82, 0.18, st * Float32(1.1), col)
        return

    if icon_id == ICON_LOCK:
        dl.add_rect(_px(x, s, 0.22), _px(y, s, 0.46), s * Float32(0.56), s * Float32(0.40), 0, Float32(2.0), st, col)
        _add_vseg(dl, x, y, s, 0.34, 0.32, 0.46, st, col)
        _add_vseg(dl, x, y, s, 0.66, 0.32, 0.46, st, col)
        _add_bez(dl, x, y, s, 0.34, 0.32, 0.34, 0.14, 0.50, 0.14, st, col)
        _add_bez(dl, x, y, s, 0.50, 0.14, 0.66, 0.14, 0.66, 0.32, st, col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.66), max(Float32(1.2), s * Float32(0.06)), col)
        return

    if icon_id == ICON_SEARCH:
        _draw_circle_outline(dl, _px(x, s, 0.44), _px(y, s, 0.44), s * Float32(0.26), st, col)
        _add_seg(dl, x, y, s, 0.63, 0.63, 0.86, 0.86, st * Float32(1.15), col)
        return

    if icon_id == ICON_PLUS:
        _add_vseg(dl, x, y, s, 0.50, 0.20, 0.80, st, col)
        _add_hseg(dl, x, y, s, 0.20, 0.80, 0.50, st, col)
        return

    if icon_id == ICON_CHEVRON:
        draw_studio_chevron(dl, x, y, s, col, Float32(0.0), st)
        return

    if icon_id == ICON_MORE_V:
        var r = max(Float32(1.3), s * Float32(0.085))
        var cx = _snap_axis(_px(x, s, 0.50), Float32(1.0))
        dl.add_circle(cx, _px(y, s, 0.22), r, col)
        dl.add_circle(cx, _px(y, s, 0.50), r, col)
        dl.add_circle(cx, _px(y, s, 0.78), r, col)
        return

    if icon_id == ICON_CLOSE:
        _add_seg(dl, x, y, s, 0.24, 0.24, 0.76, 0.76, st, col)
        _add_seg(dl, x, y, s, 0.76, 0.24, 0.24, 0.76, st, col)
        return

    # ---- Agent Theatre & Diff Controls (28..33) ----
    if icon_id == ICON_SPARK:
        # Sharp primary 4-point star + crisp secondary top-right mini-star (zero cluttered arc noise)
        var star = List[Float32](capacity=16)
        star.append(0.43); star.append(0.14)
        star.append(0.52); star.append(0.42)
        star.append(0.79); star.append(0.51)
        star.append(0.52); star.append(0.60)
        star.append(0.43); star.append(0.88)
        star.append(0.34); star.append(0.60)
        star.append(0.08); star.append(0.51)
        star.append(0.34); star.append(0.42)
        _add_poly_norm(dl, x, y, s, star, col)
        var mini = List[Float32](capacity=16)
        mini.append(0.79); mini.append(0.10)
        mini.append(0.83); mini.append(0.20)
        mini.append(0.92); mini.append(0.24)
        mini.append(0.83); mini.append(0.28)
        mini.append(0.79); mini.append(0.38)
        mini.append(0.75); mini.append(0.28)
        mini.append(0.66); mini.append(0.24)
        mini.append(0.75); mini.append(0.20)
        _add_poly_norm(dl, x, y, s, mini, col_sub)
        return

    if icon_id == ICON_CHECK:
        _add_seg(dl, x, y, s, 0.18, 0.52, 0.40, 0.76, st * Float32(1.15), col)
        _add_seg(dl, x, y, s, 0.40, 0.76, 0.84, 0.26, st * Float32(1.15), col)
        return

    if icon_id == ICON_WARN:
        _add_seg(dl, x, y, s, 0.50, 0.14, 0.88, 0.84, st, col)
        _add_seg(dl, x, y, s, 0.88, 0.84, 0.12, 0.84, st, col)
        _add_seg(dl, x, y, s, 0.12, 0.84, 0.50, 0.14, st, col)
        _add_vseg(dl, x, y, s, 0.50, 0.38, 0.58, st, col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.72), max(Float32(1.2), s * Float32(0.06)), col)
        return

    if icon_id == ICON_REVERT:
        _add_hseg(dl, x, y, s, 0.20, 0.58, 0.36, st, col)
        _add_seg(dl, x, y, s, 0.34, 0.22, 0.20, 0.36, st, col)
        _add_seg(dl, x, y, s, 0.34, 0.50, 0.20, 0.36, st, col)
        dl.add_arc(_px(x, s, 0.58), _px(y, s, 0.58), s * Float32(0.22), st, Float32(-1.5707963), Float32(3.1415926), col)
        _add_hseg(dl, x, y, s, 0.30, 0.58, 0.80, st, col)
        return

    if icon_id == ICON_DIFF:
        dl.add_rect(_px(x, s, 0.14), _px(y, s, 0.18), s * Float32(0.72), s * Float32(0.64), 0, Float32(2.0), st, col)
        _add_vseg(dl, x, y, s, 0.50, 0.20, 0.80, st_thin, col_sub)
        # Left '+'
        _add_vseg(dl, x, y, s, 0.32, 0.38, 0.62, st * Float32(0.9), col)
        _add_hseg(dl, x, y, s, 0.22, 0.42, 0.50, st * Float32(0.9), col)
        # Right '-'
        _add_hseg(dl, x, y, s, 0.58, 0.78, 0.50, st * Float32(0.9), col)
        return

    if icon_id == ICON_FOCUS_CAM:
        _add_vseg(dl, x, y, s, 0.16, 0.16, 0.34, st, col)
        _add_hseg(dl, x, y, s, 0.16, 0.34, 0.16, st, col)
        _add_hseg(dl, x, y, s, 0.66, 0.84, 0.16, st, col)
        _add_vseg(dl, x, y, s, 0.84, 0.16, 0.34, st, col)
        _add_vseg(dl, x, y, s, 0.84, 0.66, 0.84, st, col)
        _add_hseg(dl, x, y, s, 0.66, 0.84, 0.84, st, col)
        _add_hseg(dl, x, y, s, 0.16, 0.34, 0.84, st, col)
        _add_vseg(dl, x, y, s, 0.16, 0.66, 0.84, st, col)
        _draw_circle_outline(dl, _px(x, s, 0.50), _px(y, s, 0.50), s * Float32(0.16), st * Float32(0.9), col)
        return

    # ---- Asset & File Browser Icons (34..38) ----
    if icon_id == ICON_FOLDER:
        # Rounded folder body + clean top-left tab matching Obsidian RADIUS_SM language
        var rad = max(Float32(1.6), s * Float32(0.13))
        dl.add_rounded_rect_ex(
            _px(x, s, 0.12), _px(y, s, 0.20), s * Float32(0.34), s * Float32(0.14),
            rgba8_scale_alpha(col, Float32(0.20)),
            rad, rad, Float32(0.0), Float32(0.0),
            st, col,
        )
        dl.add_rounded_rect_ex(
            _px(x, s, 0.12), _px(y, s, 0.32), s * Float32(0.76), s * Float32(0.50),
            rgba8_scale_alpha(col, Float32(0.12)),
            Float32(0.0), rad, rad, rad,
            st, col,
        )
        return

    if icon_id == ICON_CODE:
        _add_seg(dl, x, y, s, 0.34, 0.28, 0.14, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.14, 0.50, 0.34, 0.72, st, col)
        _add_seg(dl, x, y, s, 0.66, 0.28, 0.86, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.86, 0.50, 0.66, 0.72, st, col)
        _add_seg(dl, x, y, s, 0.56, 0.24, 0.44, 0.76, st * Float32(0.9), col_sub)
        return

    if icon_id == ICON_MATERIAL:
        # Clean PBR shader ball: outer sphere ring + crisp specular highlight arc + subtle core shading
        var cx = _px(x, s, 0.50)
        var cy = _px(y, s, 0.50)
        dl.add_circle(cx, cy, s * Float32(0.36), rgba8_scale_alpha(col, Float32(0.12)))
        _draw_circle_outline(dl, cx, cy, s * Float32(0.36), st, col)
        # Top-left specular highlight arc
        dl.add_arc(cx, cy, s * Float32(0.22), st * Float32(0.95), Float32(3.55), Float32(1.30), col)
        # Lower-right subtle bounce-light / shadow core crescent
        dl.add_arc(cx, cy, s * Float32(0.22), st_thin, Float32(0.45), Float32(1.25), rgba8_scale_alpha(col, Float32(0.45)))
        return

    if icon_id == ICON_SCENE:
        dl.add_rect(_px(x, s, 0.14), _px(y, s, 0.32), s * Float32(0.72), s * Float32(0.52), 0, Float32(2.0), st, col)
        _add_seg(dl, x, y, s, 0.16, 0.32, 0.84, 0.18, st, col)
        _add_seg(dl, x, y, s, 0.34, 0.28, 0.42, 0.32, st, col)
        _add_seg(dl, x, y, s, 0.58, 0.23, 0.66, 0.32, st, col)
        return

    # ---- Additional Curved HUD & Utility Icons (39..44) ----
    if icon_id == ICON_REFRESH:
        var cx = _px(x, s, 0.50)
        var cy = _px(y, s, 0.50)
        var r = s * Float32(0.32)
        dl.add_arc(cx, cy, r, st, Float32(-2.65), Float32(2.25), col)
        dl.add_arc(cx, cy, r, st, Float32(0.49), Float32(2.25), col)
        _add_seg(dl, x, y, s, 0.68, 0.22, 0.80, 0.38, st, col)
        _add_seg(dl, x, y, s, 0.82, 0.24, 0.80, 0.38, st, col)
        _add_seg(dl, x, y, s, 0.32, 0.78, 0.20, 0.62, st, col)
        _add_seg(dl, x, y, s, 0.18, 0.76, 0.20, 0.62, st, col)
        return

    if icon_id == ICON_CLOCK:
        _draw_circle_outline(dl, _px(x, s, 0.50), _px(y, s, 0.50), s * Float32(0.36), st, col)
        _add_vseg(dl, x, y, s, 0.50, 0.28, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.50, 0.50, 0.68, 0.56, st, col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.50), max(Float32(1.2), s * Float32(0.06)), col)
        return

    if icon_id == ICON_BELL:
        dl.add_arc(_px(x, s, 0.50), _px(y, s, 0.42), s * Float32(0.24), st, Float32(-3.1415926), Float32(3.1415926), col)
        _add_seg(dl, x, y, s, 0.26, 0.42, 0.22, 0.70, st, col)
        _add_seg(dl, x, y, s, 0.74, 0.42, 0.78, 0.70, st, col)
        _add_hseg(dl, x, y, s, 0.16, 0.84, 0.70, st, col)
        dl.add_arc(_px(x, s, 0.50), _px(y, s, 0.73), s * Float32(0.11), st, Float32(0.15), Float32(2.84), col)
        return

    if icon_id == ICON_SUN:
        _draw_circle_outline(dl, _px(x, s, 0.50), _px(y, s, 0.50), s * Float32(0.18), st, col)
        _add_vseg(dl, x, y, s, 0.50, 0.12, 0.22, st, col)
        _add_vseg(dl, x, y, s, 0.50, 0.78, 0.88, st, col)
        _add_hseg(dl, x, y, s, 0.12, 0.22, 0.50, st, col)
        _add_hseg(dl, x, y, s, 0.78, 0.88, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.23, 0.23, 0.30, 0.30, st, col)
        _add_seg(dl, x, y, s, 0.70, 0.70, 0.77, 0.77, st, col)
        _add_seg(dl, x, y, s, 0.77, 0.23, 0.70, 0.30, st, col)
        _add_seg(dl, x, y, s, 0.30, 0.70, 0.23, 0.77, st, col)
        return

    if icon_id == ICON_MOON:
        dl.add_arc(_px(x, s, 0.50), _px(y, s, 0.50), s * Float32(0.34), st, Float32(0.35), Float32(4.20), col)
        dl.add_arc(_px(x, s, 0.64), _px(y, s, 0.38), s * Float32(0.26), st, Float32(0.85), Float32(2.35), col)
        return

    if icon_id == ICON_ERROR_CIRCLE:
        # Distinct severity icon (circle + exclamation) so error toasts never collide with dismiss 'X'
        _draw_circle_outline(dl, _px(x, s, 0.50), _px(y, s, 0.50), s * Float32(0.37), st, col)
        _add_vseg(dl, x, y, s, 0.50, 0.27, 0.56, st * Float32(1.15), col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.71), max(Float32(1.3), s * Float32(0.068)), col)
        return

    # Default / ICON_FILE: document with folded dog-ear corner
    _add_hseg(dl, x, y, s, 0.22, 0.58, 0.14, st, col)
    _add_seg(dl, x, y, s, 0.58, 0.14, 0.78, 0.34, st, col)
    _add_vseg(dl, x, y, s, 0.78, 0.34, 0.86, st, col)
    _add_hseg(dl, x, y, s, 0.22, 0.78, 0.86, st, col)
    _add_vseg(dl, x, y, s, 0.22, 0.14, 0.86, st, col)
    _add_vseg(dl, x, y, s, 0.58, 0.14, 0.34, st_thin, col_sub)
    _add_hseg(dl, x, y, s, 0.58, 0.78, 0.34, st_thin, col_sub)
