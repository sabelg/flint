"""Flint Studio In-Viewport 3D Visual Agent Theatre (`agent_theatre_3d.mojo`).

Renders live, micro-animated AI agent presence directly inside Flint Studio's 3D viewport
and HUD overlay without extra Vulkan passes or CPU readback:
  1. `emit_agent_viewport_3d_overlays(bus, lines, time_sec)`:
     - Pulsing 3D corner-bracket halos around active target entities
     - Pre-commit ghost placement / transform / despawn wireframes (`AgentGhostPreview`)
     - Fading footprint rings & 3D breadcrumb arcs on recently modified entities
  2. `emit_agent_viewport_hud_overlays(bus, dl, view_proj, vp_x, vp_y, vp_w, vp_h, time_sec)`:
     - Translucent frosted-glass Agent Presence pills (`add_glass_panel` + `PRIM_ARC` spinner + SDF role icons)
     - GPU-animated flowing quadratic Bezier tether beams (`add_bezier_flow`) with flowing luminous beads
     - Analytical `PRIM_ARC` target landing progress reticles & clamped off-screen directional chevrons
     - Collision-avoided world-space frosted-glass step callout pills with inline `[Y]`/`[N]` approval buttons

Zero-overhead idle contract: when `not bus.has_active_visuals()`, emits 0 segments and 0 UI prims.
"""
from std.collections import List
from std.math import abs, cos, floor, isfinite, max, min, pi, sin, sqrt
from flint.core.math import Mat4, Quat, Vec3
from flint.render.font import SdfFont
from flint.render.render_world import DebugLineBatch
from flint.studio.agent_bus import (
    AGENT_STATUS_WAITING_APPROVAL,
    GHOST_DESPAWN,
    GHOST_MATERIAL,
    GHOST_NONE,
    GHOST_SPAWN,
    GHOST_TRANSFORM,
    ROLE_GAMEPLAY_CODER,
    ROLE_LOOKDEV_ARTIST,
    ROLE_QA_PLAYTESTER,
    ROLE_SCENE_BUILDER,
    TRAIL_ARC_DURATION_SEC,
    AgentTelemetryBus,
    agent_status_label,
)
from flint.ui.context import pack_rgba8
from flint.ui.draw_list import UiDrawList
from flint.ui.icons import draw_studio_icon
from flint.ui.theme import (
    ICON_CHECK,
    ICON_CLOSE,
    ICON_CODE,
    ICON_CUBE,
    ICON_MATERIAL,
    ICON_PLAY,
    ICON_REVERT,
    ICON_SPARK,
    ICON_WARN,
)

comptime HIT_KIND_CALLOUT_FOCUS: UInt8 = 1
comptime HIT_KIND_APPROVE: UInt8 = 2
comptime HIT_KIND_REJECT: UInt8 = 3
comptime HIT_KIND_PRESENCE: UInt8 = 4


@fieldwise_init
struct TheatrePillHit(Copyable, ImplicitlyCopyable, Movable):
    """Clickable hit region emitted by 3D viewport agent HUD pills."""
    var kind: UInt8
    var agent_id: UInt32
    var approval_id: UInt64
    var entity_id: Int
    var x: Float32
    var y: Float32
    var w: Float32
    var h: Float32

    def contains(self, px: Float32, py: Float32) -> Bool:
        return px >= self.x and px <= self.x + self.w and py >= self.y and py <= self.y + self.h


# ============================================================================
# 1. Color, Icon & Projection Helpers
# ============================================================================

@always_inline
def _with_alpha(rgba8: UInt32, alpha_01: Float32) -> UInt32:
    var a = Float32(0.0) if not isfinite(alpha_01) else max(Float32(0.0), min(Float32(1.0), alpha_01))
    var a8 = UInt32(Int(a * Float32(255.0) + Float32(0.5)))
    return (rgba8 & UInt32(0x00FFFFFF)) | (a8 << 24)


@always_inline
def _fract(x: Float32) -> Float32:
    if not isfinite(x):
        return Float32(0.0)
    return x - floor(x)


@always_inline
def _role_icon_id(role: UInt8) -> Int:
    if role == ROLE_SCENE_BUILDER:
        return ICON_CUBE
    if role == ROLE_GAMEPLAY_CODER:
        return ICON_CODE
    if role == ROLE_LOOKDEV_ARTIST:
        return ICON_MATERIAL
    if role == ROLE_QA_PLAYTESTER:
        return ICON_PLAY
    return ICON_SPARK


@always_inline
def _draw_theatre_text(
    mut dl: UiDrawList,
    font_slot: Int,
    font: SdfFont,
    text: String,
    x: Float32,
    y_top: Float32,
    size_px: Float32,
    color: UInt32,
):
    if font.valid and text.byte_length() > 0:
        dl.add_text(font_slot, font, text, x, y_top + size_px * Float32(0.82), size_px, color)


def _project_world_to_vp(
    view_proj: Mat4,
    world_pt: Vec3,
    vp_x: Float32,
    vp_y: Float32,
    vp_w: Float32,
    vp_h: Float32,
) -> Tuple[Float32, Float32, Bool, Float32, Float32]:
    """Projects `world_pt` via `view_proj` into `[vp_x..vp_x+vp_w, vp_y..vp_y+vp_h]`.
    Returns `(sx, sy, on_screen, clamped_dir_x, clamped_dir_y)`.
    When behind the camera or outside the viewport, `(sx, sy)` is cleanly clamped onto
    the viewport border margin and `(clamped_dir_x, clamped_dir_y)` gives the unit pointer direction."""
    var w = max(Float32(1.0), vp_w)
    var h = max(Float32(1.0), vp_h)
    var cx_vp = vp_x + w * Float32(0.5)
    var cy_vp = vp_y + h * Float32(0.5)

    if not (isfinite(world_pt.x) and isfinite(world_pt.y) and isfinite(world_pt.z)):
        return (cx_vp, cy_vp, False, Float32(0.0), Float32(-1.0))

    var cx = view_proj.at(0, 0) * world_pt.x + view_proj.at(0, 1) * world_pt.y + view_proj.at(0, 2) * world_pt.z + view_proj.at(0, 3)
    var cy = view_proj.at(1, 0) * world_pt.x + view_proj.at(1, 1) * world_pt.y + view_proj.at(1, 2) * world_pt.z + view_proj.at(1, 3)
    var cw = view_proj.at(3, 0) * world_pt.x + view_proj.at(3, 1) * world_pt.y + view_proj.at(3, 2) * world_pt.z + view_proj.at(3, 3)

    if not (isfinite(cx) and isfinite(cy) and isfinite(cw)):
        return (cx_vp, cy_vp, False, Float32(0.0), Float32(-1.0))

    var in_front = cw > Float32(1.0e-4)
    var denom = cw if in_front else -max(Float32(1.0e-4), abs(cw))
    var inv_w = Float32(1.0) / denom
    var ndc_x = cx * inv_w
    var ndc_y = cy * inv_w
    if not in_front:
        ndc_x = -ndc_x
        ndc_y = -ndc_y

    var raw_sx = vp_x + (ndc_x * Float32(0.5) + Float32(0.5)) * w
    var raw_sy = vp_y + (Float32(0.5) - ndc_y * Float32(0.5)) * h

    var margin = Float32(18.0)
    var min_x = vp_x + margin
    var max_x = vp_x + w - margin
    var min_y = vp_y + margin
    var max_y = vp_y + h - margin

    var on_screen = (
        in_front
        and raw_sx >= min_x
        and raw_sx <= max_x
        and raw_sy >= min_y
        and raw_sy <= max_y
    )
    if on_screen:
        return (raw_sx, raw_sy, True, Float32(0.0), Float32(0.0))

    var dx = raw_sx - cx_vp
    var dy = raw_sy - cy_vp
    var dlen = sqrt(dx * dx + dy * dy)
    if not isfinite(dlen) or dlen < Float32(1.0e-4):
        return (cx_vp, min_y, False, Float32(0.0), Float32(-1.0))
    var ux = dx / dlen
    var uy = dy / dlen
    var half_w = max(Float32(4.0), w * Float32(0.5) - margin)
    var half_h = max(Float32(4.0), h * Float32(0.5) - margin)
    var tx = half_w / max(Float32(1.0e-4), abs(ux))
    var ty = half_h / max(Float32(1.0e-4), abs(uy))
    var t_edge = min(tx, ty)
    return (cx_vp + ux * t_edge, cy_vp + uy * t_edge, False, ux, uy)


# ============================================================================
# 2. 3D Wireframe Primitives -> `DebugLineBatch`
# ============================================================================

def _add_corner_brackets(
    mut lines: DebugLineBatch,
    center: Vec3,
    half_extents: Vec3,
    rotation: Quat,
    bracket_frac: Float32,
    rgba8: UInt32,
    thickness_px: Float32,
    xray: Bool = True,
) -> Int:
    """Emits 8 crisp L-shaped 3D corner brackets (24 segments) around an oriented box."""
    var frac = max(Float32(0.12), min(Float32(0.45), bracket_frac))
    var ax = rotation.rotate(Vec3(half_extents.x, 0.0, 0.0))
    var ay = rotation.rotate(Vec3(0.0, half_extents.y, 0.0))
    var az = rotation.rotate(Vec3(0.0, 0.0, half_extents.z))
    var bx = ax * (Float32(2.0) * frac)
    var by = ay * (Float32(2.0) * frac)
    var bz = az * (Float32(2.0) * frac)

    var signs = [-Float32(1.0), Float32(1.0)]
    var added = 0
    for ix in range(2):
        var sx = signs[ix]
        for iy in range(2):
            var sy = signs[iy]
            for iz in range(2):
                var sz = signs[iz]
                var corner = center + ax * sx + ay * sy + az * sz
                lines.add_line(corner, corner - bx * sx, rgba8, thickness_px, xray)
                lines.add_line(corner, corner - by * sy, rgba8, thickness_px, xray)
                lines.add_line(corner, corner - bz * sz, rgba8, thickness_px, xray)
                added += 3
    return added


def _add_ground_ring(
    mut lines: DebugLineBatch,
    center: Vec3,
    radius: Float32,
    y_offset: Float32,
    rgba8: UInt32,
    segments: Int,
    thickness_px: Float32,
    dashed: Bool = False,
    phase_rad: Float32 = 0.0,
    xray: Bool = False,
) -> Int:
    var segs = max(8, segments)
    var r = max(Float32(0.15), radius)
    var step = Float32(2.0) * Float32(pi) / Float32(segs)
    var base = Vec3(center.x, center.y + y_offset, center.z)
    var added = 0
    for i in range(segs):
        if dashed and (i & 1) == 1:
            continue
        var a0 = phase_rad + Float32(i) * step
        var a1 = phase_rad + Float32(i + 1) * step
        var p0 = base + Vec3(cos(a0) * r, Float32(0.0), sin(a0) * r)
        var p1 = base + Vec3(cos(a1) * r, Float32(0.0), sin(a1) * r)
        lines.add_line(p0, p1, rgba8, thickness_px, xray)
        added += 1
    return added


def _add_3d_arc(
    mut lines: DebugLineBatch,
    p0: Vec3,
    p1: Vec3,
    rgba8: UInt32,
    segments: Int = 14,
    thickness_px: Float32 = 2.0,
    dashed: Bool = False,
    xray: Bool = True,
) -> Int:
    var delta = p1 - p0
    var dist = delta.length()
    if not isfinite(dist) or dist < Float32(0.04):
        return 0
    var segs = max(4, segments)
    var lift = min(Float32(3.0), max(Float32(0.35), dist * Float32(0.28)))
    var mid = (p0 + p1) * Float32(0.5) + Vec3(0.0, lift, 0.0)
    var added = 0
    for i in range(segs):
        if dashed and (i & 1) == 1:
            continue
        var t0 = Float32(i) / Float32(segs)
        var t1 = Float32(i + 1) / Float32(segs)
        var om0 = Float32(1.0) - t0
        var om1 = Float32(1.0) - t1
        var a = p0 * (om0 * om0) + mid * (Float32(2.0) * om0 * t0) + p1 * (t0 * t0)
        var b = p0 * (om1 * om1) + mid * (Float32(2.0) * om1 * t1) + p1 * (t1 * t1)
        lines.add_line(a, b, rgba8, thickness_px, xray)
        added += 1
    return added


def emit_agent_viewport_3d_overlays(
    bus: AgentTelemetryBus,
    mut lines: DebugLineBatch,
    time_sec: Float32,
) -> Int:
    """Emits 3D world-space Visual Agent Theatre overlays into `DebugLineBatch`:
    - Pulsing corner-bracket halos around active target entities
    - Ghost wireframes & motion arcs for pending approval previews
    - Recent-edit fading footprint rings & 3D breadcrumb trails."""
    if not bus.has_active_visuals():
        return 0

    var t = time_sec if isfinite(time_sec) else Float32(0.0)
    var pulse = Float32(0.5) + Float32(0.5) * sin(t * Float32(4.6))
    var fast_pulse = Float32(0.5) + Float32(0.5) * sin(t * Float32(6.5))
    var emitted = 0

    # Layer A: Active agent target halos (corner brackets + rotating ground halo)
    for si in range(len(bus.sessions)):
        ref sess = bus.sessions[si]
        if not sess.has_active_world_pos:
            continue
        if not sess.is_busy():
            continue
        var scale_mul = Float32(1.04) + Float32(0.05) * pulse
        var he = sess.active_half_extents * scale_mul
        var bracket_col = _with_alpha(sess.accent_rgba8, Float32(0.72) + Float32(0.28) * pulse)
        var ring_col = _with_alpha(sess.accent_rgba8, Float32(0.42) + Float32(0.25) * pulse)
        emitted += _add_corner_brackets(
            lines,
            sess.active_world_pos,
            he,
            sess.active_rotation,
            Float32(0.28),
            bracket_col,
            Float32(2.4),
            xray=True,
        )
        var rad = max(he.x, he.z) * Float32(1.32)
        emitted += _add_ground_ring(
            lines,
            sess.active_world_pos,
            rad,
            -he.y,
            ring_col,
            24,
            Float32(1.75),
            dashed=True,
            phase_rad=t * Float32(0.9),
            xray=False,
        )

    # Layer A2: Hovered entity chip highlight from Mission Control UI
    if bus.has_hovered_entity:
        var h_col = pack_rgba8(120, 210, 255, 235)
        emitted += _add_corner_brackets(
            lines,
            bus.hovered_entity_pos,
            bus.hovered_entity_half_extents * Float32(1.08),
            bus.hovered_entity_rot,
            Float32(0.32),
            h_col,
            Float32(2.2),
            xray=True,
        )

    # Layer B: Pre-commit Ghost Wireframe Previews (waiting for user approval)
    for pi in range(len(bus.pending_approvals)):
        ref pa = bus.pending_approvals[pi]
        if pa.resolved or pa.ghost.kind == GHOST_NONE:
            continue
        ref g = pa.ghost
        var g_alpha = Float32(0.62) + Float32(0.32) * fast_pulse
        if g.kind == GHOST_DESPAWN:
            var danger = _with_alpha(pack_rgba8(248, 113, 113, 255), g_alpha)
            emitted += _add_corner_brackets(
                lines, g.target_pos, g.half_extents * Float32(1.06), g.target_rot, Float32(0.34), danger, Float32(2.6), xray=True
            )
            var ax = g.target_rot.rotate(Vec3(g.half_extents.x, 0.0, 0.0))
            var ay = g.target_rot.rotate(Vec3(0.0, g.half_extents.y, 0.0))
            var az = g.target_rot.rotate(Vec3(0.0, 0.0, g.half_extents.z))
            lines.add_line(g.target_pos - ax + ay - az, g.target_pos + ax + ay + az, danger, Float32(2.0), xray=True)
            lines.add_line(g.target_pos + ax + ay - az, g.target_pos - ax + ay + az, danger, Float32(2.0), xray=True)
            emitted += 2
        elif g.kind == GHOST_MATERIAL:
            var mcol = _with_alpha(g.accent_rgba8, g_alpha)
            var r = max(g.half_extents.x, g.half_extents.z) * Float32(1.35)
            emitted += _add_ground_ring(lines, g.target_pos, r, Float32(0.0), mcol, 24, Float32(2.0), dashed=True, phase_rad=t * Float32(1.8), xray=True)
            emitted += _add_ground_ring(lines, g.target_pos, r * Float32(0.82), g.half_extents.y * Float32(0.5), mcol, 20, Float32(1.6), dashed=True, phase_rad=-t * Float32(1.4), xray=True)
        else:
            var gcol = _with_alpha(g.accent_rgba8, g_alpha)
            emitted += _add_corner_brackets(
                lines, g.target_pos, g.half_extents, g.target_rot, Float32(0.32), gcol, Float32(2.2), xray=True
            )
            var axis_len = max(Float32(0.25), min(g.half_extents.x, min(g.half_extents.y, g.half_extents.z)) * Float32(0.85))
            lines.add_line(g.target_pos, g.target_pos + g.target_rot.rotate(Vec3(axis_len, 0.0, 0.0)), _with_alpha(0xFFE55B64, g_alpha), Float32(1.75), xray=True)
            lines.add_line(g.target_pos, g.target_pos + g.target_rot.rotate(Vec3(0.0, axis_len, 0.0)), _with_alpha(0xFF52C77A, g_alpha), Float32(1.75), xray=True)
            lines.add_line(g.target_pos, g.target_pos + g.target_rot.rotate(Vec3(0.0, 0.0, axis_len)), _with_alpha(0xFF4E9AF1, g_alpha), Float32(1.75), xray=True)
            emitted += 3
            var drop_r = max(g.half_extents.x, g.half_extents.z) * Float32(1.25)
            emitted += _add_ground_ring(lines, g.target_pos, drop_r, -g.half_extents.y, _with_alpha(g.accent_rgba8, g_alpha * Float32(0.65)), 20, Float32(1.6), dashed=True, phase_rad=t * Float32(1.2), xray=False)
            if g.kind == GHOST_TRANSFORM and g.has_from_pos:
                emitted += _add_3d_arc(lines, g.from_pos, g.target_pos, gcol, 16, Float32(2.2), dashed=True, xray=True)

    # Layer C: Recent-edit fade-out footprint rings & breadcrumb arcs
    for ti in range(len(bus.trails)):
        ref mk = bus.trails[ti]
        if mk.heat <= Float32(0.01):
            continue
        var fade = mk.heat * mk.heat
        var tick_col = _with_alpha(mk.accent_rgba8, fade * Float32(0.55))
        emitted += _add_corner_brackets(
            lines, mk.world_pos, mk.half_extents * Float32(1.02), mk.rotation, Float32(0.18), tick_col, Float32(1.4), xray=False
        )
        var ring_r = max(mk.half_extents.x, mk.half_extents.z) * (Float32(1.15) + (Float32(1.0) - mk.heat) * Float32(0.35))
        emitted += _add_ground_ring(lines, mk.world_pos, ring_r, -mk.half_extents.y, _with_alpha(mk.accent_rgba8, fade * Float32(0.45)), 18, Float32(1.3), dashed=False, xray=False)
        if mk.has_prev_pos:
            var arc_age = bus.theatre_time_sec - mk.last_touched_sec
            if arc_age < TRAIL_ARC_DURATION_SEC:
                var arc_fade = max(Float32(0.0), Float32(1.0) - arc_age / TRAIL_ARC_DURATION_SEC)
                emitted += _add_3d_arc(lines, mk.prev_world_pos, mk.world_pos, _with_alpha(mk.accent_rgba8, arc_fade * Float32(0.65)), 12, Float32(1.6), dashed=False, xray=True)

    return emitted


# ============================================================================
# 3. Viewport HUD Overlay (Glass Pills, GPU Bezier Flow, PRIM_ARC Reticles)
# ============================================================================

def _emit_offscreen_chevron(
    mut dl: UiDrawList,
    cx: Float32,
    cy: Float32,
    ux: Float32,
    uy: Float32,
    rgba8: UInt32,
):
    var px = -uy
    var py = ux
    var tip_x = cx + ux * Float32(8.0)
    var tip_y = cy + uy * Float32(8.0)
    var left_x = cx - ux * Float32(6.0) + px * Float32(6.5)
    var left_y = cy - uy * Float32(6.0) + py * Float32(6.5)
    var right_x = cx - ux * Float32(6.0) - px * Float32(6.5)
    var right_y = cy - uy * Float32(6.0) - py * Float32(6.5)
    var xy = List[Float32](capacity=6)
    xy.append(tip_x); xy.append(tip_y)
    xy.append(left_x); xy.append(left_y)
    xy.append(right_x); xy.append(right_y)
    var counts = List[Int](capacity=1)
    counts.append(3)
    _ = dl.add_polygon(xy, counts, rgba8)


def _emit_bezier_beam_with_beads(
    mut dl: UiDrawList,
    x0: Float32,
    y0: Float32,
    x1: Float32,
    y1: Float32,
    accent_rgba8: UInt32,
    time_sec: Float32,
):
    """Emits a GPU-animated flowing dashed Bezier tether (`add_bezier_flow`) with soft glow
    and 3 luminous energy beads travelling along `B(u)`."""
    var dx = x1 - x0
    var dy = y1 - y0
    var dist = sqrt(dx * dx + dy * dy)
    if not isfinite(dist) or dist < Float32(12.0):
        return
    var arc_lift = min(Float32(90.0), max(Float32(22.0), dist * Float32(0.18)))
    var cx = (x0 + x1) * Float32(0.5)
    var cy = min(y0, y1) - arc_lift

    # Subtle continuous guide filament + GPU-animated flowing glowing dashes (`UI_FLAG_DASHED_FLOW`)
    dl.add_quad_bezier(x0, y0, cx, cy, x1, y1, Float32(1.0), _with_alpha(accent_rgba8, Float32(0.24)), Float32(1.0))
    dl.add_bezier_flow(
        x0, y0, cx, cy, x1, y1,
        Float32(2.4),
        _with_alpha(accent_rgba8, Float32(0.92)),
        dash_px=12,
        gap_px=8,
        speed_q4=36,
        glow_alpha=155,
    )

    # 3 luminous pulse beads travelling along B(u)
    var base_phase = _fract(time_sec * Float32(0.65))
    for k in range(3):
        var u = _fract(base_phase + Float32(k) * Float32(0.333333))
        var om = Float32(1.0) - u
        var bx = om * om * x0 + Float32(2.0) * om * u * cx + u * u * x1
        var by = om * om * y0 + Float32(2.0) * om * u * cy + u * u * y1
        var edge_fade = min(Float32(1.0), min(u * Float32(5.0), (Float32(1.0) - u) * Float32(5.0)))
        dl.add_circle(bx, by, Float32(4.6), _with_alpha(accent_rgba8, Float32(0.26) * edge_fade))
        dl.add_circle(bx, by, Float32(2.2), _with_alpha(0xFFFFFFFF, Float32(0.95) * edge_fade))


def _resolve_pill_y(
    desired_y: Float32,
    pill_x: Float32,
    pill_w: Float32,
    pill_h: Float32,
    vp_y: Float32,
    vp_h: Float32,
    mut placed: List[Tuple[Float32, Float32, Float32, Float32]],
) -> Float32:
    """Simple vertical collision avoidance so multiple concurrent agent pills never overlap."""
    var y = max(vp_y + Float32(10.0), min(vp_y + vp_h - pill_h - Float32(10.0), desired_y))
    for _ in range(8):
        var collided = False
        for i in range(len(placed)):
            var r = placed[i]
            var ox = (pill_x < r[0] + r[2] + Float32(6.0)) and (pill_x + pill_w + Float32(6.0) > r[0])
            var oy = (y < r[1] + r[3] + Float32(4.0)) and (y + pill_h + Float32(4.0) > r[1])
            if ox and oy:
                var up_y = r[1] - pill_h - Float32(6.0)
                if up_y >= vp_y + Float32(10.0):
                    y = up_y
                else:
                    y = min(vp_y + vp_h - pill_h - Float32(10.0), r[1] + r[3] + Float32(6.0))
                collided = True
                break
        if not collided:
            break
    placed.append((pill_x, y, pill_w, pill_h))
    return y


def _emit_hud_core(
    bus: AgentTelemetryBus,
    mut dl: UiDrawList,
    has_font: Bool,
    font_slot: Int,
    font: SdfFont,
    view_proj: Mat4,
    vp_x: Float32,
    vp_y: Float32,
    vp_w: Float32,
    vp_h: Float32,
    time_sec: Float32,
    mut out_hits: List[TheatrePillHit],
) -> Int:
    if not bus.has_active_visuals() or vp_w < Float32(40.0) or vp_h < Float32(40.0):
        return 0

    var before_prims = dl.count()
    var t = time_sec if isfinite(time_sec) else Float32(0.0)
    var pulse = Float32(0.5) + Float32(0.5) * sin(t * Float32(5.0))
    var glass_bg = pack_rgba8(18, 22, 31, 214)
    var text_pri = pack_rgba8(237, 241, 247, 255)
    var placed_pills = List[Tuple[Float32, Float32, Float32, Float32]]()

    dl.push_clip(Int(vp_x), Int(vp_y), Int(vp_x + vp_w), Int(vp_y + vp_h))

    # Time-Travel amber rim & top-center frosted-glass indicator pill
    if bus.time_travel_active:
        var amber = pack_rgba8(245, 183, 79, 235)
        dl.add_rect(vp_x + Float32(1.0), vp_y + Float32(1.0), vp_w - Float32(2.0), vp_h - Float32(2.0), 0, Float32(2.0), Float32(2.0), amber)
        var tt_txt = "TIME-TRAVEL PREVIEW  |  Step #" + String(bus.time_travel_step_idx + 1)
        var tt_w = (font.measure(tt_txt, Float32(12.0)) + Float32(42.0)) if (has_font and font.valid) else Float32(252.0)
        var tt_x = vp_x + (vp_w - tt_w) * Float32(0.5)
        var tt_y = vp_y + Float32(10.0)
        dl.add_glass_panel(
            tt_x, tt_y, tt_w, Float32(26.0),
            pack_rgba8(34, 24, 12, 224),
            Float32(13.0), Float32(3.5), Float32(1.2), amber,
            top_bevel=True, drop_shadow=True,
        )
        draw_studio_icon(dl, ICON_REVERT, tt_x + Float32(9.0), tt_y + Float32(6.0), Float32(14.0), amber, Float32(1.5))
        if has_font and font.valid:
            _draw_theatre_text(dl, font_slot, font, tt_txt, tt_x + Float32(28.0), tt_y + Float32(7.0), Float32(12.0), amber)

    # Render each active agent's Presence Pill, Bezier Flow tether, PRIM_ARC reticle & callout pill
    var presence_slot = 0
    for si in range(len(bus.sessions)):
        ref sess = bus.sessions[si]
        if not sess.is_busy() and not sess.has_active_world_pos:
            continue

        var accent = sess.accent_rgba8
        var pending_idx = bus.first_pending_for_agent(sess.agent_id)
        var has_approval = pending_idx >= 0
        var role_icon = _role_icon_id(sess.role)

        # 1. Floating Frosted-Glass Agent Presence Anchor Pill (clears Top-Left Transform HUD)
        var pres_label = sess.name + " | " + agent_status_label(sess.status)
        var pres_w = (font.measure(pres_label, Float32(12.0)) + Float32(46.0)) if (has_font and font.valid) else Float32(204.0)
        pres_w = min(pres_w, vp_w * Float32(0.48))
        var pres_h = Float32(28.0)
        var pres_x = vp_x + Float32(12.0)
        var pres_y = vp_y + Float32(54.0) + Float32(presence_slot) * Float32(34.0)
        presence_slot += 1

        dl.add_glass_panel(
            pres_x, pres_y, pres_w, pres_h,
            glass_bg,
            Float32(14.0), Float32(3.5), Float32(1.2), _with_alpha(accent, Float32(0.72)),
            top_bevel=True, drop_shadow=True,
        )
        # Animated PRIM_ARC status spinner around role icon
        var spin_cx = pres_x + Float32(15.0)
        var spin_cy = pres_y + pres_h * Float32(0.5)
        if sess.status == AGENT_STATUS_WAITING_APPROVAL:
            var warn_col = pack_rgba8(245, 183, 79, 255)
            dl.add_progress_ring(spin_cx, spin_cy, Float32(8.5), Float32(1.8), Float32(1.0), warn_col, _with_alpha(warn_col, Float32(0.25)), Float32(3.5))
            draw_studio_icon(dl, ICON_WARN, spin_cx - Float32(5.5), spin_cy - Float32(5.5), Float32(11.0), warn_col, Float32(1.4))
        else:
            dl.add_arc(
                spin_cx, spin_cy, Float32(8.5), Float32(1.8),
                t * Float32(4.4) + Float32(si), Float32(4.1),
                accent, _with_alpha(accent, Float32(0.22)), Float32(3.0), 0,
            )
            draw_studio_icon(dl, role_icon, spin_cx - Float32(5.5), spin_cy - Float32(5.5), Float32(11.0), accent, Float32(1.3))
        if has_font and font.valid:
            _draw_theatre_text(dl, font_slot, font, pres_label, pres_x + Float32(30.0), pres_y + Float32(8.0), Float32(12.0), text_pri)
        out_hits.append(TheatrePillHit(HIT_KIND_PRESENCE, sess.agent_id, 0, sess.active_entity_id, pres_x, pres_y, pres_w, pres_h))

        # 2. Target projection + GPU Bezier Flow Tether + PRIM_ARC Target Reticle + World Callout
        if not sess.has_active_world_pos:
            continue
        var proj = _project_world_to_vp(view_proj, sess.active_world_pos, vp_x, vp_y, vp_w, vp_h)
        var tx = proj[0]
        var ty = proj[1]
        var on_screen = proj[2]

        _emit_bezier_beam_with_beads(
            dl,
            pres_x + pres_w - Float32(2.0),
            pres_y + pres_h * Float32(0.5),
            tx,
            ty,
            accent,
            t + Float32(si) * Float32(0.37),
        )

        if on_screen:
            # Analytical PRIM_ARC progress ring landing reticle
            var ret_r = Float32(10.5) + Float32(1.8) * pulse
            var ret_prog = max(Float32(0.22), min(Float32(1.0), sess.active_progress))
            dl.add_progress_ring(tx, ty, ret_r, Float32(2.0), ret_prog, accent, _with_alpha(accent, Float32(0.22)), Float32(4.0))
            dl.add_circle(tx, ty, Float32(2.8), accent)
        else:
            _emit_offscreen_chevron(dl, tx, ty, proj[3], proj[4], accent)

        # Anchor world-space frosted-glass step callout above target AABB top
        var top_world = sess.active_world_pos + Vec3(0.0, sess.active_half_extents.y + Float32(0.25), 0.0)
        var top_proj = _project_world_to_vp(view_proj, top_world, vp_x, vp_y, vp_w, vp_h)
        var raw_callout = sess.active_callout if sess.active_callout.byte_length() > 0 else agent_status_label(sess.status)
        if has_approval and raw_callout.startswith("Approve: "):
            var trimmed_c = String(raw_callout[byte=9:])
            raw_callout = trimmed_c^
        var callout_text = sess.name + ": " + raw_callout
        var approval_extra_w = Float32(70.0) if has_approval else Float32(0.0)
        var text_w = font.measure(callout_text, Float32(11.5)) if (has_font and font.valid) else Float32(min(220, max(96, callout_text.byte_length() * 6)))
        var pill_w = min(vp_w - Float32(24.0), text_w + Float32(42.0) + approval_extra_w)
        var pill_h = Float32(28.0)
        var anchor_x = top_proj[0] if top_proj[2] else tx
        var anchor_y = (top_proj[1] - pill_h - Float32(10.0)) if top_proj[2] else (ty - pill_h - Float32(10.0))
        var pill_x = max(vp_x + Float32(10.0), min(vp_x + vp_w - pill_w - Float32(10.0), anchor_x - pill_w * Float32(0.5)))
        var pill_y = _resolve_pill_y(anchor_y, pill_x, pill_w, pill_h, vp_y, vp_h, placed_pills)

        var border_col = pack_rgba8(245, 183, 79, 225) if sess.status == AGENT_STATUS_WAITING_APPROVAL else _with_alpha(accent, Float32(0.75))
        dl.add_glass_panel(
            pill_x, pill_y, pill_w, pill_h,
            glass_bg,
            Float32(9.0), Float32(3.0), Float32(1.2), border_col,
            top_bevel=True, drop_shadow=True,
        )
        # Left role / warning SDF icon
        var badge_icon = ICON_WARN if sess.status == AGENT_STATUS_WAITING_APPROVAL else role_icon
        var badge_col = pack_rgba8(245, 183, 79, 255) if sess.status == AGENT_STATUS_WAITING_APPROVAL else accent
        draw_studio_icon(dl, badge_icon, pill_x + Float32(8.0), pill_y + Float32(7.0), Float32(14.0), badge_col, Float32(1.4))
        if has_font and font.valid:
            _draw_theatre_text(dl, font_slot, font, callout_text, pill_x + Float32(27.0), pill_y + Float32(8.0), Float32(11.5), text_pri)

        # Slim bottom progress bar on active step callout
        var prog = max(Float32(0.0), min(Float32(1.0), sess.active_progress))
        if prog > Float32(0.01):
            var bar_w = max(Float32(6.0), (pill_w - Float32(14.0)) * prog)
            dl.add_rect(pill_x + Float32(7.0), pill_y + pill_h - Float32(3.0), bar_w, Float32(2.0), accent, Float32(1.0))

        out_hits.append(TheatrePillHit(HIT_KIND_CALLOUT_FOCUS, sess.agent_id, 0, sess.active_entity_id, pill_x, pill_y, pill_w - approval_extra_w, pill_h))

        # Inline clickable [Y] [N] approval micro-pills directly inside 3D viewport
        if has_approval:
            var aid = bus.pending_approvals[pending_idx].approval_id
            var btn_y = pill_y + Float32(4.0)
            var btn_h = Float32(20.0)
            var btn_w = Float32(25.0)
            var y_x = pill_x + pill_w - Float32(58.0)
            var n_x = pill_x + pill_w - Float32(29.0)
            var ok_col = pack_rgba8(52, 211, 153, 235)
            var no_col = pack_rgba8(248, 113, 113, 235)
            dl.add_rect(y_x, btn_y, btn_w, btn_h, pack_rgba8(18, 58, 44, 235), Float32(6.0), Float32(1.0), ok_col)
            dl.add_rect(n_x, btn_y, btn_w, btn_h, pack_rgba8(66, 20, 24, 235), Float32(6.0), Float32(1.0), no_col)
            draw_studio_icon(dl, ICON_CHECK, y_x + Float32(6.5), btn_y + Float32(4.0), Float32(12.0), ok_col, Float32(1.6))
            draw_studio_icon(dl, ICON_CLOSE, n_x + Float32(6.5), btn_y + Float32(4.0), Float32(12.0), no_col, Float32(1.6))
            out_hits.append(TheatrePillHit(HIT_KIND_APPROVE, sess.agent_id, aid, sess.active_entity_id, y_x, btn_y, btn_w, btn_h))
            out_hits.append(TheatrePillHit(HIT_KIND_REJECT, sess.agent_id, aid, sess.active_entity_id, n_x, btn_y, btn_w, btn_h))

    dl.pop_clip()
    return dl.count() - before_prims


def emit_agent_viewport_hud_overlays(
    bus: AgentTelemetryBus,
    mut dl: UiDrawList,
    view_proj: Mat4,
    vp_x: Float32,
    vp_y: Float32,
    vp_w: Float32,
    vp_h: Float32,
    time_sec: Float32,
) -> Int:
    """Projects active agent targets cleanly inside `[vp_x, vp_y, vp_w, vp_h]` and emits
    frosted-glass presence pills, GPU-animated Bezier flow beams, PRIM_ARC landing reticles,
    offscreen chevrons, and world-space step callout cards."""
    var dummy_font = SdfFont()
    var hits = List[TheatrePillHit]()
    return _emit_hud_core(bus, dl, False, 0, dummy_font, view_proj, vp_x, vp_y, vp_w, vp_h, time_sec, hits)


def emit_agent_viewport_hud_overlays_with_font(
    bus: AgentTelemetryBus,
    mut dl: UiDrawList,
    font_slot: Int,
    font: SdfFont,
    view_proj: Mat4,
    vp_x: Float32,
    vp_y: Float32,
    vp_w: Float32,
    vp_h: Float32,
    time_sec: Float32,
    mut out_hits: List[TheatrePillHit],
) -> Int:
    """Full text + interactive hit-region variant of `emit_agent_viewport_hud_overlays`."""
    return _emit_hud_core(bus, dl, True, font_slot, font, view_proj, vp_x, vp_y, vp_w, vp_h, time_sec, out_hits)
