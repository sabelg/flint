"""Resolution-Independent Vector/SDF Studio Icons (`flint.ui.icons`).

Emits crisp, geometrically balanced icons for every `UiIconId` (`ICON_*` in `flint.ui.theme`)
directly into `UiDrawList` using stable analytical primitives (`add_rect`, `add_circle`,
`add_line`, `add_quad_bezier`, `add_polygon`) so icons render identically across GPU
compute overlay and Tier-0 CPU SIMD rasterizer at any scale (`12px`..`48px`).
"""
from std.collections import List
from std.math import cos, isfinite, sin
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
)


@always_inline
def _px(origin: Float32, size: Float32, norm: Float32) -> Float32:
    """Maps normalized `[0.0 .. 1.0]` icon coordinate to screen pixels."""
    return origin + norm * size


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
    var arm = size * Float32(0.24)
    var tip = size * Float32(0.16)
    var st = max(Float32(1.0), stroke_px if isfinite(stroke_px) else Float32(1.5))

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
    var st = max(Float32(1.0), min(size * Float32(0.22), stroke_px if isfinite(stroke_px) else Float32(1.5)))
    var col = color_rgba8
    var s = size

    # ---- Transport (0..3) ----
    if icon_id == ICON_PLAY:
        var pts = List[Float32](capacity=6)
        pts.append(0.28); pts.append(0.18)
        pts.append(0.82); pts.append(0.50)
        pts.append(0.28); pts.append(0.82)
        _add_poly_norm(dl, x, y, s, pts, col)
        return

    if icon_id == ICON_PAUSE:
        var bw = max(Float32(2.0), s * Float32(0.22))
        var bh = s * Float32(0.64)
        var rr = min(Float32(2.0), bw * Float32(0.35))
        dl.add_rect(_px(x, s, 0.22), _px(y, s, 0.18), bw, bh, col, rr)
        dl.add_rect(_px(x, s, 0.56), _px(y, s, 0.18), bw, bh, col, rr)
        return

    if icon_id == ICON_STEP:
        var pts = List[Float32](capacity=6)
        pts.append(0.18); pts.append(0.20)
        pts.append(0.62); pts.append(0.50)
        pts.append(0.18); pts.append(0.80)
        _add_poly_norm(dl, x, y, s, pts, col)
        var bw = max(Float32(2.0), s * Float32(0.14))
        dl.add_rect(_px(x, s, 0.70), _px(y, s, 0.20), bw, s * Float32(0.60), col, Float32(1.2))
        return

    if icon_id == ICON_STOP:
        dl.add_rect(_px(x, s, 0.22), _px(y, s, 0.22), s * Float32(0.56), s * Float32(0.56), col, max(Float32(1.5), s * Float32(0.12)))
        return

    # ---- Gizmo & Viewport (4..11) ----
    if icon_id == ICON_MOVE:
        _add_seg(dl, x, y, s, 0.50, 0.14, 0.50, 0.86, st, col)
        _add_seg(dl, x, y, s, 0.14, 0.50, 0.86, 0.50, st, col)
        # Arrowheads
        _add_seg(dl, x, y, s, 0.38, 0.26, 0.50, 0.14, st, col)
        _add_seg(dl, x, y, s, 0.62, 0.26, 0.50, 0.14, st, col)
        _add_seg(dl, x, y, s, 0.38, 0.74, 0.50, 0.86, st, col)
        _add_seg(dl, x, y, s, 0.62, 0.74, 0.50, 0.86, st, col)
        _add_seg(dl, x, y, s, 0.26, 0.38, 0.14, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.26, 0.62, 0.14, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.74, 0.38, 0.86, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.74, 0.62, 0.86, 0.50, st, col)
        return

    if icon_id == ICON_ROTATE:
        var cx = _px(x, s, 0.50)
        var cy = _px(y, s, 0.52)
        var r = s * Float32(0.32)
        # Single analytical 270-degree SDF arc (`0` -> `3*PI/2`) + arrowhead at top pointing right
        dl.add_arc(cx, cy, r, st, Float32(0.0), Float32(4.712389), col)
        _add_seg(dl, x, y, s, 0.44, 0.11, 0.58, 0.20, st, col)
        _add_seg(dl, x, y, s, 0.44, 0.29, 0.58, 0.20, st, col)
        return

    if icon_id == ICON_SCALE:
        # Outer frame corner + diagonal scale arrow
        _add_seg(dl, x, y, s, 0.20, 0.80, 0.80, 0.20, st, col)
        _add_seg(dl, x, y, s, 0.56, 0.20, 0.80, 0.20, st, col)
        _add_seg(dl, x, y, s, 0.80, 0.20, 0.80, 0.44, st, col)
        _add_seg(dl, x, y, s, 0.20, 0.56, 0.20, 0.80, st, col)
        _add_seg(dl, x, y, s, 0.20, 0.80, 0.44, 0.80, st, col)
        dl.add_rect(_px(x, s, 0.16), _px(y, s, 0.16), s * Float32(0.24), s * Float32(0.24), 0, Float32(1.5), st, col)
        return

    if icon_id == ICON_WORLD:
        var cx = _px(x, s, 0.50)
        var cy = _px(y, s, 0.50)
        _draw_circle_outline(dl, cx, cy, s * Float32(0.36), st, col)
        _add_seg(dl, x, y, s, 0.14, 0.50, 0.86, 0.50, st * Float32(0.85), col)
        _add_bez(dl, x, y, s, 0.50, 0.14, 0.28, 0.50, 0.50, 0.86, st * Float32(0.85), col)
        _add_bez(dl, x, y, s, 0.50, 0.14, 0.72, 0.50, 0.50, 0.86, st * Float32(0.85), col)
        return

    if icon_id == ICON_LOCAL:
        _add_seg(dl, x, y, s, 0.42, 0.58, 0.42, 0.16, st, col)
        _add_seg(dl, x, y, s, 0.42, 0.58, 0.84, 0.58, st, col)
        _add_seg(dl, x, y, s, 0.42, 0.58, 0.18, 0.82, st, col)
        dl.add_circle(_px(x, s, 0.42), _px(y, s, 0.58), max(Float32(1.5), s * Float32(0.09)), col)
        return

    if icon_id == ICON_SNAP:
        # U-shaped magnet
        _add_seg(dl, x, y, s, 0.24, 0.20, 0.24, 0.56, st * Float32(1.15), col)
        _add_seg(dl, x, y, s, 0.76, 0.20, 0.76, 0.56, st * Float32(1.15), col)
        _add_bez(dl, x, y, s, 0.24, 0.56, 0.24, 0.86, 0.50, 0.86, st * Float32(1.15), col)
        _add_bez(dl, x, y, s, 0.50, 0.86, 0.76, 0.86, 0.76, 0.56, st * Float32(1.15), col)
        _add_seg(dl, x, y, s, 0.18, 0.34, 0.30, 0.34, st, col)
        _add_seg(dl, x, y, s, 0.70, 0.34, 0.82, 0.34, st, col)
        return

    if icon_id == ICON_CAMERA:
        dl.add_rect(_px(x, s, 0.12), _px(y, s, 0.28), s * Float32(0.50), s * Float32(0.44), 0, Float32(2.0), st, col)
        dl.add_arc(_px(x, s, 0.37), _px(y, s, 0.50), max(Float32(1.5), s * Float32(0.11)), max(Float32(1.0), st * Float32(0.85)), Float32(0.0), Float32(6.2831853), col)
        var cone = List[Float32](capacity=8)
        cone.append(0.64); cone.append(0.42)
        cone.append(0.88); cone.append(0.28)
        cone.append(0.88); cone.append(0.72)
        cone.append(0.64); cone.append(0.58)
        _add_poly_norm(dl, x, y, s, cone, col)
        return

    if icon_id == ICON_GRID:
        dl.add_rect(_px(x, s, 0.15), _px(y, s, 0.15), s * Float32(0.70), s * Float32(0.70), 0, Float32(2.0), st, col)
        _add_seg(dl, x, y, s, 0.39, 0.16, 0.39, 0.84, st * Float32(0.85), col)
        _add_seg(dl, x, y, s, 0.61, 0.16, 0.61, 0.84, st * Float32(0.85), col)
        _add_seg(dl, x, y, s, 0.16, 0.39, 0.84, 0.39, st * Float32(0.85), col)
        _add_seg(dl, x, y, s, 0.16, 0.61, 0.84, 0.61, st * Float32(0.85), col)
        return

    # ---- Entities & Components (12..19) ----
    if icon_id == ICON_CUBE:
        # Isometric cube outline + internal Y edges
        _add_seg(dl, x, y, s, 0.50, 0.12, 0.84, 0.30, st, col)
        _add_seg(dl, x, y, s, 0.84, 0.30, 0.84, 0.70, st, col)
        _add_seg(dl, x, y, s, 0.84, 0.70, 0.50, 0.88, st, col)
        _add_seg(dl, x, y, s, 0.50, 0.88, 0.16, 0.70, st, col)
        _add_seg(dl, x, y, s, 0.16, 0.70, 0.16, 0.30, st, col)
        _add_seg(dl, x, y, s, 0.16, 0.30, 0.50, 0.12, st, col)
        _add_seg(dl, x, y, s, 0.16, 0.30, 0.50, 0.48, st * Float32(0.9), col)
        _add_seg(dl, x, y, s, 0.84, 0.30, 0.50, 0.48, st * Float32(0.9), col)
        _add_seg(dl, x, y, s, 0.50, 0.48, 0.50, 0.88, st * Float32(0.9), col)
        return

    if icon_id == ICON_LIGHT_POINT:
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.50), max(Float32(2.0), s * Float32(0.16)), col)
        _add_seg(dl, x, y, s, 0.50, 0.12, 0.50, 0.24, st, col)
        _add_seg(dl, x, y, s, 0.50, 0.76, 0.50, 0.88, st, col)
        _add_seg(dl, x, y, s, 0.12, 0.50, 0.24, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.76, 0.50, 0.88, 0.50, st, col)
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
        _add_bez(dl, x, y, s, 0.18, 0.76, 0.50, 0.64, 0.82, 0.76, st * Float32(0.8), col)
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
        _add_seg(dl, x, y, s, 0.18, 0.82, 0.66, 0.34, st * Float32(0.9), col)
        return

    if icon_id == ICON_EMPTY:
        # Corner brackets + center dot
        _add_seg(dl, x, y, s, 0.18, 0.34, 0.18, 0.18, st, col)
        _add_seg(dl, x, y, s, 0.18, 0.18, 0.34, 0.18, st, col)
        _add_seg(dl, x, y, s, 0.66, 0.18, 0.82, 0.18, st, col)
        _add_seg(dl, x, y, s, 0.82, 0.18, 0.82, 0.34, st, col)
        _add_seg(dl, x, y, s, 0.82, 0.66, 0.82, 0.82, st, col)
        _add_seg(dl, x, y, s, 0.82, 0.82, 0.66, 0.82, st, col)
        _add_seg(dl, x, y, s, 0.34, 0.82, 0.18, 0.82, st, col)
        _add_seg(dl, x, y, s, 0.18, 0.82, 0.18, 0.66, st, col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.50), max(Float32(1.5), s * Float32(0.10)), col)
        return

    if icon_id == ICON_PREFAB:
        # Outer isometric diamond + inner filled core diamond
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
        # Capsule physics collider wireframe
        var cr = s * Float32(0.24)
        dl.add_rect(_px(x, s, 0.26), _px(y, s, 0.14), s * Float32(0.48), s * Float32(0.72), 0, cr, st, col)
        _add_seg(dl, x, y, s, 0.28, 0.38, 0.72, 0.38, st * Float32(0.8), col)
        _add_seg(dl, x, y, s, 0.28, 0.62, 0.72, 0.62, st * Float32(0.8), col)
        return

    # ---- Chrome & Actions (20..27) ----
    if icon_id == ICON_EYE:
        _add_bez(dl, x, y, s, 0.12, 0.50, 0.50, 0.14, 0.88, 0.50, st, col)
        _add_bez(dl, x, y, s, 0.12, 0.50, 0.50, 0.86, 0.88, 0.50, st, col)
        dl.add_arc(_px(x, s, 0.50), _px(y, s, 0.50), max(Float32(1.8), s * Float32(0.14)), max(Float32(1.0), st * Float32(0.9)), Float32(0.0), Float32(6.2831853), col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.50), max(Float32(1.0), s * Float32(0.055)), col)
        return

    if icon_id == ICON_EYE_OFF:
        _add_bez(dl, x, y, s, 0.14, 0.50, 0.50, 0.18, 0.86, 0.50, st * Float32(0.9), col)
        _add_bez(dl, x, y, s, 0.14, 0.50, 0.50, 0.82, 0.86, 0.50, st * Float32(0.9), col)
        _add_seg(dl, x, y, s, 0.18, 0.82, 0.82, 0.18, st * Float32(1.1), col)
        return

    if icon_id == ICON_LOCK:
        dl.add_rect(_px(x, s, 0.22), _px(y, s, 0.46), s * Float32(0.56), s * Float32(0.40), 0, Float32(2.0), st, col)
        _add_seg(dl, x, y, s, 0.34, 0.46, 0.34, 0.32, st, col)
        _add_seg(dl, x, y, s, 0.66, 0.46, 0.66, 0.32, st, col)
        _add_bez(dl, x, y, s, 0.34, 0.32, 0.34, 0.14, 0.50, 0.14, st, col)
        _add_bez(dl, x, y, s, 0.50, 0.14, 0.66, 0.14, 0.66, 0.32, st, col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.66), max(Float32(1.2), s * Float32(0.06)), col)
        return

    if icon_id == ICON_SEARCH:
        _draw_circle_outline(dl, _px(x, s, 0.44), _px(y, s, 0.44), s * Float32(0.26), st, col)
        _add_seg(dl, x, y, s, 0.63, 0.63, 0.86, 0.86, st * Float32(1.15), col)
        return

    if icon_id == ICON_PLUS:
        _add_seg(dl, x, y, s, 0.50, 0.20, 0.50, 0.80, st, col)
        _add_seg(dl, x, y, s, 0.20, 0.50, 0.80, 0.50, st, col)
        return

    if icon_id == ICON_CHEVRON:
        draw_studio_chevron(dl, x, y, s, col, Float32(0.0), st)
        return

    if icon_id == ICON_MORE_V:
        var r = max(Float32(1.3), s * Float32(0.085))
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.22), r, col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.50), r, col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.78), r, col)
        return

    if icon_id == ICON_CLOSE:
        _add_seg(dl, x, y, s, 0.24, 0.24, 0.76, 0.76, st, col)
        _add_seg(dl, x, y, s, 0.76, 0.24, 0.24, 0.76, st, col)
        return

    # ---- Agent Theatre & Diff Controls (28..33) ----
    if icon_id == ICON_SPARK:
        # 4-point AI star polygon + analytical orbital arc glint
        var star = List[Float32](capacity=16)
        star.append(0.46); star.append(0.12)
        star.append(0.55); star.append(0.39)
        star.append(0.82); star.append(0.48)
        star.append(0.55); star.append(0.57)
        star.append(0.46); star.append(0.84)
        star.append(0.37); star.append(0.57)
        star.append(0.10); star.append(0.48)
        star.append(0.37); star.append(0.39)
        _add_poly_norm(dl, x, y, s, star, col)
        dl.add_arc(_px(x, s, 0.46), _px(y, s, 0.48), s * Float32(0.38), max(Float32(1.0), st * Float32(0.75)), Float32(-1.15), Float32(0.95), col)
        dl.add_circle(_px(x, s, 0.82), _px(y, s, 0.20), max(Float32(1.2), s * Float32(0.065)), col)
        return

    if icon_id == ICON_CHECK:
        _add_seg(dl, x, y, s, 0.18, 0.52, 0.40, 0.76, st * Float32(1.15), col)
        _add_seg(dl, x, y, s, 0.40, 0.76, 0.84, 0.26, st * Float32(1.15), col)
        return

    if icon_id == ICON_WARN:
        _add_seg(dl, x, y, s, 0.50, 0.14, 0.88, 0.84, st, col)
        _add_seg(dl, x, y, s, 0.88, 0.84, 0.12, 0.84, st, col)
        _add_seg(dl, x, y, s, 0.12, 0.84, 0.50, 0.14, st, col)
        _add_seg(dl, x, y, s, 0.50, 0.38, 0.50, 0.58, st, col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.72), max(Float32(1.2), s * Float32(0.06)), col)
        return

    if icon_id == ICON_REVERT:
        _add_seg(dl, x, y, s, 0.20, 0.36, 0.58, 0.36, st, col)
        _add_seg(dl, x, y, s, 0.34, 0.22, 0.20, 0.36, st, col)
        _add_seg(dl, x, y, s, 0.34, 0.50, 0.20, 0.36, st, col)
        dl.add_arc(_px(x, s, 0.58), _px(y, s, 0.58), s * Float32(0.22), st, Float32(-1.5707963), Float32(3.1415926), col)
        _add_seg(dl, x, y, s, 0.58, 0.80, 0.30, 0.80, st, col)
        return

    if icon_id == ICON_DIFF:
        dl.add_rect(_px(x, s, 0.14), _px(y, s, 0.18), s * Float32(0.72), s * Float32(0.64), 0, Float32(2.0), st, col)
        _add_seg(dl, x, y, s, 0.50, 0.20, 0.50, 0.80, st * Float32(0.85), col)
        # Left '+'
        _add_seg(dl, x, y, s, 0.32, 0.38, 0.32, 0.62, st * Float32(0.9), col)
        _add_seg(dl, x, y, s, 0.22, 0.50, 0.42, 0.50, st * Float32(0.9), col)
        # Right '-'
        _add_seg(dl, x, y, s, 0.58, 0.50, 0.78, 0.50, st * Float32(0.9), col)
        return

    if icon_id == ICON_FOCUS_CAM:
        _add_seg(dl, x, y, s, 0.16, 0.34, 0.16, 0.16, st, col)
        _add_seg(dl, x, y, s, 0.16, 0.16, 0.34, 0.16, st, col)
        _add_seg(dl, x, y, s, 0.66, 0.16, 0.84, 0.16, st, col)
        _add_seg(dl, x, y, s, 0.84, 0.16, 0.84, 0.34, st, col)
        _add_seg(dl, x, y, s, 0.84, 0.66, 0.84, 0.84, st, col)
        _add_seg(dl, x, y, s, 0.84, 0.84, 0.66, 0.84, st, col)
        _add_seg(dl, x, y, s, 0.34, 0.84, 0.16, 0.84, st, col)
        _add_seg(dl, x, y, s, 0.16, 0.84, 0.16, 0.66, st, col)
        _draw_circle_outline(dl, _px(x, s, 0.50), _px(y, s, 0.50), s * Float32(0.16), st * Float32(0.9), col)
        return

    # ---- Asset & File Browser Icons (34..38) ----
    if icon_id == ICON_FOLDER:
        _add_seg(dl, x, y, s, 0.14, 0.24, 0.42, 0.24, st, col)
        _add_seg(dl, x, y, s, 0.42, 0.24, 0.52, 0.34, st, col)
        _add_seg(dl, x, y, s, 0.52, 0.34, 0.86, 0.34, st, col)
        _add_seg(dl, x, y, s, 0.86, 0.34, 0.86, 0.80, st, col)
        _add_seg(dl, x, y, s, 0.86, 0.80, 0.14, 0.80, st, col)
        _add_seg(dl, x, y, s, 0.14, 0.80, 0.14, 0.24, st, col)
        return

    if icon_id == ICON_CODE:
        _add_seg(dl, x, y, s, 0.34, 0.28, 0.14, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.14, 0.50, 0.34, 0.72, st, col)
        _add_seg(dl, x, y, s, 0.66, 0.28, 0.86, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.86, 0.50, 0.66, 0.72, st, col)
        _add_seg(dl, x, y, s, 0.56, 0.24, 0.44, 0.76, st * Float32(0.9), col)
        return

    if icon_id == ICON_MATERIAL:
        _draw_circle_outline(dl, _px(x, s, 0.50), _px(y, s, 0.50), s * Float32(0.36), st, col)
        _add_bez(dl, x, y, s, 0.32, 0.42, 0.34, 0.26, 0.52, 0.24, st, col)
        dl.add_circle(_px(x, s, 0.62), _px(y, s, 0.62), max(Float32(1.4), s * Float32(0.08)), col)
        return

    if icon_id == ICON_SCENE:
        dl.add_rect(_px(x, s, 0.14), _px(y, s, 0.32), s * Float32(0.72), s * Float32(0.52), 0, Float32(2.0), st, col)
        _add_seg(dl, x, y, s, 0.16, 0.32, 0.84, 0.18, st, col)
        _add_seg(dl, x, y, s, 0.34, 0.28, 0.42, 0.32, st, col)
        _add_seg(dl, x, y, s, 0.58, 0.23, 0.66, 0.32, st, col)
        return

    # ---- Additional Curved HUD & Utility Icons (39..43) ----
    if icon_id == ICON_REFRESH:
        var cx = _px(x, s, 0.50)
        var cy = _px(y, s, 0.50)
        var r = s * Float32(0.32)
        # Dual opposing analytical SDF arcs with arrowheads
        dl.add_arc(cx, cy, r, st, Float32(-2.65), Float32(2.25), col)
        dl.add_arc(cx, cy, r, st, Float32(0.49), Float32(2.25), col)
        _add_seg(dl, x, y, s, 0.68, 0.22, 0.80, 0.38, st, col)
        _add_seg(dl, x, y, s, 0.82, 0.24, 0.80, 0.38, st, col)
        _add_seg(dl, x, y, s, 0.32, 0.78, 0.20, 0.62, st, col)
        _add_seg(dl, x, y, s, 0.18, 0.76, 0.20, 0.62, st, col)
        return

    if icon_id == ICON_CLOCK:
        _draw_circle_outline(dl, _px(x, s, 0.50), _px(y, s, 0.50), s * Float32(0.36), st, col)
        _add_seg(dl, x, y, s, 0.50, 0.50, 0.50, 0.28, st, col)
        _add_seg(dl, x, y, s, 0.50, 0.50, 0.68, 0.56, st, col)
        dl.add_circle(_px(x, s, 0.50), _px(y, s, 0.50), max(Float32(1.2), s * Float32(0.06)), col)
        return

    if icon_id == ICON_BELL:
        dl.add_arc(_px(x, s, 0.50), _px(y, s, 0.42), s * Float32(0.24), st, Float32(-3.1415926), Float32(3.1415926), col)
        _add_seg(dl, x, y, s, 0.26, 0.42, 0.22, 0.70, st, col)
        _add_seg(dl, x, y, s, 0.74, 0.42, 0.78, 0.70, st, col)
        _add_seg(dl, x, y, s, 0.16, 0.70, 0.84, 0.70, st, col)
        dl.add_arc(_px(x, s, 0.50), _px(y, s, 0.73), s * Float32(0.11), st, Float32(0.15), Float32(2.84), col)
        return

    if icon_id == ICON_SUN:
        _draw_circle_outline(dl, _px(x, s, 0.50), _px(y, s, 0.50), s * Float32(0.18), st, col)
        _add_seg(dl, x, y, s, 0.50, 0.12, 0.50, 0.22, st, col)
        _add_seg(dl, x, y, s, 0.50, 0.78, 0.50, 0.88, st, col)
        _add_seg(dl, x, y, s, 0.12, 0.50, 0.22, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.78, 0.50, 0.88, 0.50, st, col)
        _add_seg(dl, x, y, s, 0.23, 0.23, 0.30, 0.30, st, col)
        _add_seg(dl, x, y, s, 0.70, 0.70, 0.77, 0.77, st, col)
        _add_seg(dl, x, y, s, 0.77, 0.23, 0.70, 0.30, st, col)
        _add_seg(dl, x, y, s, 0.30, 0.70, 0.23, 0.77, st, col)
        return

    if icon_id == ICON_MOON:
        # Crescent moon built from two analytical SDF arcs
        dl.add_arc(_px(x, s, 0.50), _px(y, s, 0.50), s * Float32(0.34), st, Float32(0.35), Float32(4.20), col)
        dl.add_arc(_px(x, s, 0.64), _px(y, s, 0.38), s * Float32(0.26), st, Float32(0.85), Float32(2.35), col)
        return

    # Default / ICON_FILE: document with folded dog-ear corner
    _add_seg(dl, x, y, s, 0.22, 0.14, 0.58, 0.14, st, col)
    _add_seg(dl, x, y, s, 0.58, 0.14, 0.78, 0.34, st, col)
    _add_seg(dl, x, y, s, 0.78, 0.34, 0.78, 0.86, st, col)
    _add_seg(dl, x, y, s, 0.78, 0.86, 0.22, 0.86, st, col)
    _add_seg(dl, x, y, s, 0.22, 0.86, 0.22, 0.14, st, col)
    _add_seg(dl, x, y, s, 0.58, 0.14, 0.58, 0.34, st * Float32(0.85), col)
    _add_seg(dl, x, y, s, 0.58, 0.34, 0.78, 0.34, st * Float32(0.85), col)
