"""Chamfered Extrusions, Lathed 5-Spoke Wheel/Brake Assemblies & RMF Spline Lofts (`loft_bevel.mojo`).

Implements:
  1. `extrude_profile_beveled`: 2D polygon extrusion with smooth circular corner fillets and
     2-step specular cap chamfers + cavity/ground AO baking.
  2. `build_lathed_wheel_assembly`: Crowned tire tread + bulging sidewalls, deep-dish alloy barrel +
     `5` sculpted split-Y spokes + recessed ventilated brake rotor disc, and 6-piston caliper (`~420` tris).
  3. `sweep_profile_rmf` & `sweep_profile_rmf_banked`: Wang Double-Reflection Rotation-Minimizing Frame
     (`RMF`) parallel-transport spline loft + `make_chamfered_girder_profile`, `make_curb_profile`,
     and `sweep_catenary_cable`.
"""
from std.math import acos, cos, isfinite, max, min, sin, sqrt
from flint.core.detmath import det_atan2
from flint.core.math import Vec2, Vec3
from flint.assets.model import NativeMesh
from flint.assets.mesh_builder import MeshBuilder

comptime _PI: Float32 = 3.14159265
comptime _TAU: Float32 = 6.28318530


@always_inline
def _norm2d(v: Vec2) -> Vec2:
    var l = v.length()
    if l < Float32(1e-6):
        return Vec2(1.0, 0.0)
    return v * (Float32(1.0) / l)


@always_inline
def _signed_area_2d(poly: List[Vec2]) -> Float32:
    var n = len(poly)
    var area2 = Float32(0.0)
    for i in range(n):
        var a = poly[i]
        var b = poly[(i + 1) % n]
        area2 += a.x * b.y - b.x * a.y
    return area2 * Float32(0.5)


def _fillet_polygon_2d(poly_in: List[Vec2], radius_in: Float32, steps: Int) -> Tuple[List[Vec2], List[Vec2]]:
    """Rounds polygon corners in 2D and returns `(filleted_points_ccw, outward_2d_normals)`."""
    var n_raw = len(poly_in)
    var pts = List[Vec2]()
    var nrms = List[Vec2]()
    if n_raw < 3:
        return (pts^, nrms^)
    for i in range(n_raw):
        if not (isfinite(poly_in[i].x) and isfinite(poly_in[i].y)):
            return (pts^, nrms^)
    var radius = radius_in if isfinite(radius_in) else Float32(0.0)

    # Ensure CCW winding in `(x, y)` so outward 2D normal of edge `d = (dx, dy)` is `(dy, -dx)`
    var poly = List[Vec2](capacity=n_raw)
    if _signed_area_2d(poly_in) >= Float32(0.0):
        for i in range(n_raw):
            poly.append(poly_in[i])
    else:
        for i in range(n_raw):
            poly.append(poly_in[n_raw - 1 - i])

    var arc_steps = max(1, min(8, steps))
    for i in range(n_raw):
        var p_prev = poly[(i + n_raw - 1) % n_raw]
        var p_cur = poly[i]
        var p_next = poly[(i + 1) % n_raw]

        var e_in = p_cur - p_prev
        var e_out = p_next - p_cur
        var len_in = e_in.length()
        var len_out = e_out.length()
        if len_in < Float32(1e-5) or len_out < Float32(1e-5):
            continue

        var d_in = e_in * (Float32(1.0) / len_in)
        var d_out = e_out * (Float32(1.0) / len_out)
        var n_in = Vec2(d_in.y, -d_in.x)
        var n_out = Vec2(d_out.y, -d_out.x)

        if radius <= Float32(1e-5):
            pts.append(p_cur)
            nrms.append(_norm2d(n_in + n_out))
            continue

        # Clamp tangent trim distance so adjacent fillets never overlap
        var u_prev = Vec2(-d_in.x, -d_in.y)
        var u_next = d_out
        var dot_u = max(Float32(-0.995), min(Float32(0.995), u_prev.x * u_next.x + u_prev.y * u_next.y))
        var half_ang = acos(dot_u) * Float32(0.5)
        var sin_h = sin(half_ang)
        var cos_h = cos(half_ang)
        if sin_h < Float32(0.05) or cos_h < Float32(0.02):
            pts.append(p_cur)
            nrms.append(_norm2d(n_in + n_out))
            continue

        var max_trim = min(len_in, len_out) * Float32(0.38)
        var want_trim = radius * (cos_h / sin_h)
        var trim = min(max_trim, want_trim)
        var a_start = p_cur + u_prev * trim
        var a_end = p_cur + u_next * trim

        # Quadratic Bezier / circular arc blend between `(a_start, n_in)` and `(a_end, n_out)`
        for s in range(arc_steps + 1):
            var t = Float32(s) / Float32(arc_steps)
            var omt = Float32(1.0) - t
            # Rational-like circular weight keeps arc bulged nicely around corner
            var p_arc = a_start * (omt * omt) + p_cur * (Float32(2.0) * omt * t) + a_end * (t * t)
            var n_arc = _norm2d(n_in * omt + n_out * t)
            pts.append(p_arc)
            nrms.append(n_arc)

    return (pts^, nrms^)


def extrude_profile_beveled(
    polygon_2d: List[Vec2],
    height: Float32,
    corner_fillet: Float32 = 0.04,
    cap_chamfer: Float32 = 0.03,
    fillet_steps: Int = 3,
    base_color: UInt32 = 0xFFFFFFFF,
    roughness: Float32 = 0.40,
    metallic: Float32 = 0.15,
) -> NativeMesh:
    """Extrudes a 2D `(x, z)` polygon vertically from `y = 0` to `y = height` with smooth corner fillets
    and 2-step specular cap chamfers on both top and bottom caps."""
    if not isfinite(height) or height <= Float32(1e-5) or len(polygon_2d) < 3:
        return NativeMesh()
    for i in range(len(polygon_2d)):
        if not (isfinite(polygon_2d[i].x) and isfinite(polygon_2d[i].y)):
            return NativeMesh()
    var safe_fillet = corner_fillet if isfinite(corner_fillet) else Float32(0.0)
    var safe_chamfer = cap_chamfer if isfinite(cap_chamfer) else Float32(0.0)
    var safe_rough = roughness if isfinite(roughness) else Float32(0.40)
    var safe_metal = metallic if isfinite(metallic) else Float32(0.15)
    var filleted = _fillet_polygon_2d(polygon_2d, safe_fillet, fillet_steps)
    ref prof = filleted[0]
    ref nrm2 = filleted[1]
    var m = len(prof)
    if m < 3:
        return NativeMesh()

    # Compute centroid & conservative max inward inset sochamfer never inverts thin profiles
    var cen = Vec2(0.0, 0.0)
    var min_x = prof[0].x; var max_x = prof[0].x
    var min_z = prof[0].y; var max_z = prof[0].y
    for i in range(m):
        cen = cen + prof[i]
        if prof[i].x < min_x: min_x = prof[i].x
        if prof[i].x > max_x: max_x = prof[i].x
        if prof[i].y < min_z: min_z = prof[i].y
        if prof[i].y > max_z: max_z = prof[i].y
    cen = cen * (Float32(1.0) / Float32(m))
    var min_span = max(Float32(0.002), min(max_x - min_x, max_z - min_z))
    var ch = max(Float32(0.0), min(safe_chamfer, min(height * Float32(0.32), min_span * Float32(0.22))))

    # 6 vertical stations for 2-step specular cap chamfers:
    # 0: bottom cap rim (`y = 0, inset = ch`)
    # 1: bottom chamfer mid (`y = 0.35*ch, inset = 0.35*ch`)
    # 2: bottom wall (`y = ch, inset = 0`)
    # 3: top wall (`y = height - ch, inset = 0`)
    # 4: top chamfer mid (`y = height - 0.35*ch, inset = 0.35*ch`)
    # 5: top cap rim (`y = height, inset = ch`)
    var y_levels: List[Float32] = [
        Float32(0.0),
        ch * Float32(0.35),
        ch,
        height - ch,
        height - ch * Float32(0.35),
        height,
    ]
    var insets: List[Float32] = [
        ch,
        ch * Float32(0.35),
        Float32(0.0),
        Float32(0.0),
        ch * Float32(0.35),
        ch,
    ]
    var ny_tilt: List[Float32] = [
        Float32(-0.86),
        Float32(-0.55),
        Float32(-0.08),
        Float32(0.08),
        Float32(0.55),
        Float32(0.86),
    ]
    var ao_ring: List[UInt8] = [
        UInt8(125),
        UInt8(165),
        UInt8(205),
        UInt8(245),
        UInt8(252),
        UInt8(255),
    ]

    var perimeter = Float32(0.0)
    var arc_u = List[Float32](capacity=m + 1)
    arc_u.append(Float32(0.0))
    for i in range(m):
        var nxt = (i + 1) % m
        perimeter += (prof[nxt] - prof[i]).length()
        arc_u.append(perimeter)
    var inv_perim = Float32(1.0) / max(Float32(1e-4), perimeter)

    var mb = MeshBuilder(base_color=base_color, roughness=safe_rough, metallic=safe_metal)
    var stride = m + 1
    var ring_base = mb.mesh.vertex_count()

    for r in range(6):
        var yl = y_levels[r]
        var ins = insets[r]
        var nyt = ny_tilt[r]
        var h_rad = sqrt(max(Float32(0.04), Float32(1.0) - nyt * nyt))
        var tv = yl / max(Float32(1e-4), height)
        var ao_b = ao_ring[r]

        for i in range(stride):
            var pi = i % m
            var p2 = prof[pi] - nrm2[pi] * ins
            var n3 = Vec3(nrm2[pi].x * h_rad, nyt, nrm2[pi].y * h_rad).normalized()
            # Note: in `(X, Y, Z)` right-handed coords, polygon in `(X, Z)` with CCW `(x, z)` has `X x Z = -Y`,
            # so let's rely on `MeshBuilder.add_tri` which auto-orients triangles to match `n3`!
            _ = mb.add_vertex(Vec3(p2.x, yl, p2.y), n3, arc_u[i] * inv_perim, tv, ao=ao_b)

    for r in range(5):
        for i in range(m):
            var i0 = ring_base + r * stride + i
            var i1 = i0 + 1
            var i2 = i0 + stride + 1
            var i3 = i0 + stride
            mb.add_quad(i0, i1, i2, i3)

    # Flat specular-isolated bottom cap (`y = 0`) & top cap (`y = height`)
    var inv_span = Float32(1.0) / max(Float32(1e-4), max(max_x - min_x, max_z - min_z))
    var bot_c = mb.add_vertex(
        Vec3(cen.x, 0.0, cen.y),
        Vec3(0.0, -1.0, 0.0),
        (cen.x - min_x) * inv_span,
        (cen.y - min_z) * inv_span,
        ao=UInt8(105),
    )
    var bot_ring = List[Int](capacity=m)
    for i in range(m):
        var p2 = prof[i] - nrm2[i] * ch
        bot_ring.append(
            mb.add_vertex(
                Vec3(p2.x, 0.0, p2.y),
                Vec3(0.0, -1.0, 0.0),
                (p2.x - min_x) * inv_span,
                (p2.y - min_z) * inv_span,
                ao=UInt8(120),
            )
        )
    for i in range(m):
        mb.add_tri(bot_c, bot_ring[(i + 1) % m], bot_ring[i])

    var top_c = mb.add_vertex(
        Vec3(cen.x, height, cen.y),
        Vec3(0.0, 1.0, 0.0),
        (cen.x - min_x) * inv_span,
        (cen.y - min_z) * inv_span,
        ao=UInt8(255),
    )
    var top_ring = List[Int](capacity=m)
    for i in range(m):
        var p2 = prof[i] - nrm2[i] * ch
        top_ring.append(
            mb.add_vertex(
                Vec3(p2.x, height, p2.y),
                Vec3(0.0, 1.0, 0.0),
                (p2.x - min_x) * inv_span,
                (p2.y - min_z) * inv_span,
                ao=UInt8(255),
            )
        )
    for i in range(m):
        mb.add_tri(top_c, top_ring[i], top_ring[(i + 1) % m])

    var out = mb^.finish()
    out.ensure_bounds()
    return out^


# ============================================================================
# 2. Lathed 5-Spoke Alloy Wheel, Crowned Tire, Drilled Rotor & 6-Piston Caliper
# ============================================================================

def _add_lathe_x_ring_strip(
    mut mb: MeshBuilder,
    x_stat: List[Float32],
    r_stat: List[Float32],
    nx_stat: List[Float32],
    nr_stat: List[Float32],
    ao_stat: List[UInt8],
    segs: Int,
    color: UInt32 = 0xFFFFFFFF,
    ground_contact_ao: Bool = False,
):
    """Lathes a multi-station profile around the `+X` wheel axle (`YZ` plane)."""
    var n_st = len(x_stat)
    if n_st < 2 or segs < 3:
        return
    var base_v = mb.mesh.vertex_count()
    var stride = segs + 1

    for i in range(n_st):
        var xv = x_stat[i]
        var rv = r_stat[i]
        var nx = nx_stat[i]
        var nr = nr_stat[i]
        var base_ao = Int(ao_stat[i])
        var tv = Float32(i) / Float32(n_st - 1)

        for s in range(stride):
            var tu = Float32(s) / Float32(segs)
            var phi = tu * _TAU
            var cp = cos(phi)
            var sp = sin(phi)
            # `y = -cp * rv` so `phi = 0` is bottom ground contact (`y = -rv`)!
            var py = -cp * rv
            var pz = sp * rv
            var n = Vec3(nx, -cp * nr, sp * nr).normalized()
            var v_ao = base_ao
            if ground_contact_ao and cp > Float32(0.35):
                var g_dark = Int((cp - Float32(0.35)) * Float32(95.0))
                v_ao = max(72, base_ao - g_dark)
            _ = mb.add_vertex(Vec3(xv, py, pz), n, tu, tv, color=color, ao=UInt8(v_ao))

    for i in range(n_st - 1):
        for s in range(segs):
            var i0 = base_v + i * stride + s
            var i1 = i0 + 1
            var i2 = i0 + stride + 1
            var i3 = i0 + stride
            mb.add_quad(i0, i1, i2, i3)


def build_lathed_wheel_assembly(
    radius: Float32 = 0.35,
    width: Float32 = 0.28,
    spoke_count: Int = 5,
) -> Tuple[NativeMesh, NativeMesh, NativeMesh]:
    """Builds a 3-part automotive wheel & brake assembly (`~420` tris total, axle along `X`, outer face `+X`):
    - `0`: `tire_mesh` (`180` tris): Crowned slick/tread band + bulging pneumatic sidewalls + inner/outer bead lips.
    - `1`: `alloy_rim_and_rotor_mesh` (`~184` tris): Deep-dish alloy barrel + `spoke_count` (`5`) sculpted split-Y
           twin spokes + center lug hub + recessed ventilated brake rotor disc.
    - `2`: `brake_caliper_mesh` (`~56` tris): Sculpted 6-piston Brembo caliper straddling the inner rotor.
    """
    var r_in = radius if isfinite(radius) else Float32(0.35)
    var w_in = width if isfinite(width) else Float32(0.28)
    var r_tire = max(Float32(0.12), r_in)
    var hw = max(Float32(0.06), w_in * Float32(0.5))
    var r_rim = r_tire * Float32(0.68)
    var r_barrel_in = r_rim * Float32(0.84)
    var r_hub = r_rim * Float32(0.26)
    var r_rotor = r_rim * Float32(0.76)
    var spokes = max(3, min(10, spoke_count))

    # -------------------------------------------------------------------------
    # Part 0: Crowned Tire (`18` radial segments x `5` strips = `180` tris)
    # -------------------------------------------------------------------------
    var mb_tire = MeshBuilder(base_color=0xFF1A1D22, roughness=0.72, metallic=0.04)
    var tx: List[Float32] = [
        -hw * Float32(0.88),
        -hw * Float32(1.03),
        -hw * Float32(0.82),
         hw * Float32(0.82),
         hw * Float32(1.03),
         hw * Float32(0.88),
    ]
    var tr: List[Float32] = [
        r_rim,
        r_tire * Float32(0.84),
        r_tire,
        r_tire,
        r_tire * Float32(0.84),
        r_rim,
    ]
    var tnx: List[Float32] = [
        Float32(-0.92),
        Float32(-0.96),
        Float32(-0.28),
        Float32(0.28),
        Float32(0.96),
        Float32(0.92),
    ]
    var tnr: List[Float32] = [
        Float32(-0.35),
        Float32(0.28),
        Float32(0.96),
        Float32(0.96),
        Float32(0.28),
        Float32(-0.35),
    ]
    var tao: List[UInt8] = [
        UInt8(105),
        UInt8(185),
        UInt8(245),
        UInt8(250),
        UInt8(230),
        UInt8(165),
    ]
    _add_lathe_x_ring_strip(mb_tire, tx, tr, tnx, tnr, tao, segs=20, ground_contact_ao=True)
    var tire_mesh = mb_tire^.finish()
    tire_mesh.ensure_bounds()

    # -------------------------------------------------------------------------
    # Part 1: Deep-Dish Alloy Rim + Split-Y Spokes + Ventilated Brake Rotor
    # -------------------------------------------------------------------------
    var mb_rim = MeshBuilder(base_color=0xFFD8DEE6, roughness=0.20, metallic=0.88)
    mb_rim.mesh.clearcoat = 0.65
    mb_rim.mesh.clearcoat_roughness = 0.06

    # Stepped concave alloy barrel (`18` segments x `2` strips = `72` tris)
    var bx: List[Float32] = [
        hw * Float32(0.92),
        hw * Float32(0.72),
        -hw * Float32(0.75),
    ]
    var br: List[Float32] = [
        r_rim * Float32(1.01),
        r_barrel_in,
        r_barrel_in * Float32(0.96),
    ]
    var bnx: List[Float32] = [
        Float32(0.95),
        Float32(0.42),
        Float32(-0.25),
    ]
    var bnr: List[Float32] = [
        Float32(0.30),
        Float32(-0.90),
        Float32(-0.96),
    ]
    var bao: List[UInt8] = [
        UInt8(248),
        UInt8(170),
        UInt8(88),
    ]
    _add_lathe_x_ring_strip(mb_rim, bx, br, bnx, bnr, bao, segs=18, color=0xFFE2E7EE)

    # Center concave lug-nut hub (`14` tris)
    var hub_x_out = hw * Float32(0.56)
    var hub_c = mb_rim.add_vertex(Vec3(hub_x_out - hw * Float32(0.08), 0.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.5, 0.5, color=0xFFB8C2CE, ao=UInt8(155))
    var hub_ring = List[Int](capacity=14)
    for s in range(14):
        var a = Float32(s) / Float32(14.0) * _TAU
        var hy = cos(a) * r_hub
        var hz = sin(a) * r_hub
        hub_ring.append(
            mb_rim.add_vertex(
                Vec3(hub_x_out, hy, hz),
                Vec3(0.88, cos(a) * Float32(0.46), sin(a) * Float32(0.46)).normalized(),
                0.5 + cos(a) * 0.2,
                0.5 + sin(a) * 0.2,
                color=0xFFDFE5EC,
                ao=UInt8(225),
            )
        )
    for s in range(14):
        mb_rim.add_tri(hub_c, hub_ring[s], hub_ring[(s + 1) % 14])

    # Sculpted 3D closed Split-Y Spokes (`spokes` * `2` branches * `6` tris = `60` tris for 5 spokes!)
    for sp_i in range(spokes):
        var base_a = (Float32(sp_i) / Float32(spokes)) * _TAU
        for branch in range(2):
            var sign_b = Float32(-1.0) if branch == 0 else Float32(1.0)
            var a_in = base_a + sign_b * Float32(0.055)
            var a_out = base_a + sign_b * Float32(0.235)

            var dir_in = Vec3(0.0, cos(a_in), sin(a_in))
            var side_in = Vec3(0.0, -sin(a_in), cos(a_in))
            var dir_out = Vec3(0.0, cos(a_out), sin(a_out))
            var side_out = Vec3(0.0, -sin(a_out), cos(a_out))

            var w_in = r_rim * Float32(0.058)
            var w_out = r_rim * Float32(0.044)
            var p_root = Vec3(hub_x_out, 0.0, 0.0) + dir_in * (r_hub * Float32(0.88))
            var p_tip = Vec3(hw * Float32(0.84), 0.0, 0.0) + dir_out * (r_rim * Float32(0.95))

            # Closed triangular-prism spoke (`3` quad sides = `6` tris/branch!)
            var v_r_l = mb_rim.add_vertex(p_root - side_in * w_in - Vec3(hw * Float32(0.09), 0.0, 0.0), (Vec3(0.45, 0.0, 0.0) - side_in * Float32(0.88)).normalized(), 0.2, 0.1, color=0xFFCFD6DF, ao=UInt8(185))
            var v_r_c = mb_rim.add_vertex(p_root + Vec3(hw * Float32(0.04), 0.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.5, 0.1, color=0xFFF1F5F9, ao=UInt8(248))
            var v_r_r = mb_rim.add_vertex(p_root + side_in * w_in - Vec3(hw * Float32(0.09), 0.0, 0.0), (Vec3(0.45, 0.0, 0.0) + side_in * Float32(0.88)).normalized(), 0.8, 0.1, color=0xFFCFD6DF, ao=UInt8(185))

            var v_t_l = mb_rim.add_vertex(p_tip - side_out * w_out - Vec3(hw * Float32(0.08), 0.0, 0.0), (Vec3(0.45, 0.0, 0.0) - side_out * Float32(0.88)).normalized(), 0.2, 0.9, color=0xFFCFD6DF, ao=UInt8(210))
            var v_t_c = mb_rim.add_vertex(p_tip + Vec3(hw * Float32(0.03), 0.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.5, 0.9, color=0xFFF1F5F9, ao=UInt8(255))
            var v_t_r = mb_rim.add_vertex(p_tip + side_out * w_out - Vec3(hw * Float32(0.08), 0.0, 0.0), (Vec3(0.45, 0.0, 0.0) + side_out * Float32(0.88)).normalized(), 0.8, 0.9, color=0xFFCFD6DF, ao=UInt8(210))

            var v_r_bl = mb_rim.add_vertex(p_root - side_in * w_in - Vec3(hw * Float32(0.09), 0.0, 0.0), Vec3(-1.0, 0.0, 0.0), 0.2, 0.1, color=0xFF9AA3AE, ao=UInt8(130))
            var v_r_br = mb_rim.add_vertex(p_root + side_in * w_in - Vec3(hw * Float32(0.09), 0.0, 0.0), Vec3(-1.0, 0.0, 0.0), 0.8, 0.1, color=0xFF9AA3AE, ao=UInt8(130))
            var v_t_bl = mb_rim.add_vertex(p_tip - side_out * w_out - Vec3(hw * Float32(0.08), 0.0, 0.0), Vec3(-1.0, 0.0, 0.0), 0.2, 0.9, color=0xFF9AA3AE, ao=UInt8(140))
            var v_t_br = mb_rim.add_vertex(p_tip + side_out * w_out - Vec3(hw * Float32(0.08), 0.0, 0.0), Vec3(-1.0, 0.0, 0.0), 0.8, 0.9, color=0xFF9AA3AE, ao=UInt8(140))

            mb_rim.add_quad(v_r_l, v_t_l, v_t_c, v_r_c)
            mb_rim.add_quad(v_r_c, v_t_c, v_t_r, v_r_r)
            mb_rim.add_quad(v_r_br, v_t_br, v_t_bl, v_r_bl)

    # Recessed inner ventilated iron/carbon brake rotor disc (`18` segments annulus = `36` tris)
    var x_rot = hw * Float32(0.20)
    var r_rot_in = r_hub * Float32(1.15)
    var rot_base = mb_rim.mesh.vertex_count()
    for s in range(19):
        var a = (Float32(s) / Float32(18.0)) * _TAU
        var cy = cos(a); var sz = sin(a)
        _ = mb_rim.add_vertex(Vec3(x_rot, cy * r_rot_in, sz * r_rot_in), Vec3(1.0, 0.0, 0.0), 0.3, Float32(s) / Float32(18.0), color=0xFF7B848F, ao=UInt8(120))
        _ = mb_rim.add_vertex(Vec3(x_rot, cy * r_rotor, sz * r_rotor), Vec3(1.0, 0.0, 0.0), 0.9, Float32(s) / Float32(18.0), color=0xFF9AA3AE, ao=UInt8(150))
    for s in range(18):
        var b0 = rot_base + s * 2
        mb_rim.add_quad(b0, b0 + 1, b0 + 3, b0 + 2)

    var rim_mesh = mb_rim^.finish()
    rim_mesh.ensure_bounds()

    # -------------------------------------------------------------------------
    # Part 2: 6-Piston Curved Brembo Brake Caliper (`~52` tris)
    # -------------------------------------------------------------------------
    var mb_cal = MeshBuilder(base_color=0xFFE02424, roughness=0.26, metallic=0.35)
    mb_cal.mesh.clearcoat = 0.85
    mb_cal.mesh.clearcoat_roughness = 0.05

    # Curved 4-station arc straddling `r_rotor` across `theta = 0.35 .. 1.35` rad (`rear-upper` quadrant)
    var cal_r_in = r_rotor * Float32(0.76)
    var cal_r_out = r_rotor * Float32(1.09)
    var cal_x0 = x_rot - hw * Float32(0.18)
    var cal_x1 = x_rot + hw * Float32(0.24)
    var cal_rings = List[List[Int]]()

    for st_i in range(4):
        var t_a = Float32(st_i) / Float32(3.0)
        var ang = Float32(0.32) + t_a * Float32(1.04)
        var cy = cos(ang)
        var sz = -sin(ang)
        var rad_dir = Vec3(0.0, cy, sz)
        var tan_dir = Vec3(0.0, sin(ang), cos(ang))
        var taper = Float32(0.90) if (st_i == 0 or st_i == 3) else Float32(1.0)
        var r_lo = cal_r_in + (Float32(1.0) - taper) * r_rotor * Float32(0.05)
        var r_hi = cal_r_out - (Float32(1.0) - taper) * r_rotor * Float32(0.04)

        var ring = List[Int](capacity=4)
        ring.append(mb_cal.add_vertex(Vec3(cal_x1, 0.0, 0.0) + rad_dir * r_lo, (Vec3(0.8, 0.0, 0.0) - rad_dir * Float32(0.6)).normalized(), t_a, 0.0, ao=UInt8(175)))
        ring.append(mb_cal.add_vertex(Vec3(cal_x1, 0.0, 0.0) + rad_dir * r_hi, (Vec3(0.7, 0.0, 0.0) + rad_dir * Float32(0.7)).normalized(), t_a, 0.33, ao=UInt8(235)))
        ring.append(mb_cal.add_vertex(Vec3(cal_x0, 0.0, 0.0) + rad_dir * r_hi, (Vec3(-0.6, 0.0, 0.0) + rad_dir * Float32(0.8)).normalized(), t_a, 0.66, ao=UInt8(210)))
        ring.append(mb_cal.add_vertex(Vec3(cal_x0, 0.0, 0.0) + rad_dir * r_lo, (Vec3(-0.8, 0.0, 0.0) - rad_dir * Float32(0.6)).normalized(), t_a, 1.0, ao=UInt8(140)))
        _ = tan_dir
        cal_rings.append(ring^)

    for st_i in range(3):
        for k in range(4):
            var kn = (k + 1) % 4
            mb_cal.add_quad(cal_rings[st_i][k], cal_rings[st_i + 1][k], cal_rings[st_i + 1][kn], cal_rings[st_i][kn])

    # Cap both ends of caliper housing (`4` tris)
    mb_cal.add_quad(cal_rings[0][0], cal_rings[0][3], cal_rings[0][2], cal_rings[0][1])
    mb_cal.add_quad(cal_rings[3][0], cal_rings[3][1], cal_rings[3][2], cal_rings[3][3])

    var cal_mesh = mb_cal^.finish()
    cal_mesh.ensure_bounds()

    return (tire_mesh^, rim_mesh^, cal_mesh^)


# ============================================================================
# 3. Wang Double-Reflection Rotation-Minimizing Frame (`RMF`) Spline Loft
# ============================================================================

def compute_rmf_frames(
    path_pts: List[Vec3],
    closed_loop: Bool = False,
    up_hint: Vec3 = Vec3(0.0, 1.0, 0.0),
) -> Tuple[List[Vec3], List[Vec3], List[Vec3]]:
    """Computes twist-free Parallel-Transport frames `(rights, ups, tangents)` along `path_pts`
    via Wang et al. (2008) Double-Reflection RMF (zero gimbal-flip even through vertical coaster loops)."""
    var n = len(path_pts)
    var rights = List[Vec3](length=n, fill=Vec3(1.0, 0.0, 0.0))
    var ups = List[Vec3](length=n, fill=Vec3(0.0, 1.0, 0.0))
    var tangs = List[Vec3](length=n, fill=Vec3(0.0, 0.0, 1.0))
    if n < 2:
        return (rights^, ups^, tangs^)
    for i in range(n):
        if not (isfinite(path_pts[i].x) and isfinite(path_pts[i].y) and isfinite(path_pts[i].z)):
            return (rights^, ups^, tangs^)
    var safe_up_hint = up_hint if (isfinite(up_hint.x) and isfinite(up_hint.y) and isfinite(up_hint.z)) else Vec3(0.0, 1.0, 0.0)

    for i in range(n):
        var t_dir: Vec3
        if closed_loop:
            t_dir = path_pts[(i + 1) % n] - path_pts[(i + n - 1) % n]
        elif i == 0:
            t_dir = path_pts[1] - path_pts[0]
        elif i == n - 1:
            t_dir = path_pts[n - 1] - path_pts[n - 2]
        else:
            t_dir = path_pts[i + 1] - path_pts[i - 1]
        if t_dir.length_squared() < Float32(1e-10):
            t_dir = tangs[max(0, i - 1)]
        tangs[i] = t_dir.normalized()

    # Initial frame at station 0
    var t0 = tangs[0]
    var ref_up = safe_up_hint.normalized()
    if ref_up.length_squared() < Float32(0.1) or abs(t0.dot(ref_up)) > Float32(0.95):
        ref_up = Vec3(0.0, 0.0, 1.0) if abs(t0.y) > Float32(0.9) else Vec3(0.0, 1.0, 0.0)
    var r0 = ref_up.cross(t0).normalized()
    var u0 = t0.cross(r0).normalized()
    rights[0] = r0
    ups[0] = u0

    # Double-reflection parallel transport `i -> i + 1`
    for i in range(n - 1):
        var v1 = path_pts[i + 1] - path_pts[i]
        var c1 = v1.length_squared()
        var ti = tangs[i]
        var ti1 = tangs[i + 1]
        var ri = rights[i]
        if c1 < Float32(1e-12):
            rights[i + 1] = ri
            ups[i + 1] = ups[i]
            continue
        var r_l = ri - v1 * ((Float32(2.0) / c1) * v1.dot(ri))
        var t_l = ti - v1 * ((Float32(2.0) / c1) * v1.dot(ti))
        var v2 = ti1 - t_l
        var c2 = v2.length_squared()
        var ri1 = r_l - v2 * ((Float32(2.0) / c2) * v2.dot(r_l)) if c2 > Float32(1e-12) else r_l
        var ui1 = ti1.cross(ri1).normalized()
        ri1 = ui1.cross(ti1).normalized()
        rights[i + 1] = ri1
        ups[i + 1] = ui1

    # Distribute closed-loop holonomy twist error uniformly around the loop to eliminate seam discontinuity
    if closed_loop and n >= 3:
        var v1 = path_pts[0] - path_pts[n - 1]
        var c1 = v1.length_squared()
        var ti = tangs[n - 1]
        var ti1 = tangs[0]
        var ri = rights[n - 1]
        var r_end: Vec3
        if c1 < Float32(1e-12):
            r_end = ri
        else:
            var r_l = ri - v1 * ((Float32(2.0) / c1) * v1.dot(ri))
            var t_l = ti - v1 * ((Float32(2.0) / c1) * v1.dot(ti))
            var v2 = ti1 - t_l
            var c2 = v2.length_squared()
            var r_tmp = r_l - v2 * ((Float32(2.0) / c2) * v2.dot(r_l)) if c2 > Float32(1e-12) else r_l
            var u_tmp = ti1.cross(r_tmp).normalized()
            r_end = u_tmp.cross(ti1).normalized()
        var cos_a = r_end.dot(rights[0])
        var sin_a = r_end.dot(ups[0])
        var twist_err = det_atan2(sin_a, cos_a)
        if abs(twist_err) > Float32(1e-6):
            var inv_n = Float32(1.0) / Float32(n)
            for i in range(1, n):
                var ang = -twist_err * (Float32(i) * inv_n)
                var ca = cos(ang)
                var sa = sin(ang)
                var ri_new = (rights[i] * ca + ups[i] * sa).normalized()
                var ui_new = tangs[i].cross(ri_new).normalized()
                ri_new = ui_new.cross(tangs[i]).normalized()
                rights[i] = ri_new
                ups[i] = ui_new

    return (rights^, ups^, tangs^)


def sweep_profile_rmf_banked(
    path_pts: List[Vec3],
    profile_2d: List[Vec2],
    bank_angles_rad: List[Float32],
    closed_loop: Bool = False,
    closed_profile: Bool = True,
    up_hint: Vec3 = Vec3(0.0, 1.0, 0.0),
    uv_scale_u: Float32 = 1.0,
    uv_scale_v: Float32 = 1.0,
    cap_ends: Bool = True,
    base_color: UInt32 = 0xFFFFFFFF,
) -> NativeMesh:
    """Lofts a 2D cross-section `(x=right, y=up)` along `path_pts` using twist-free RMF frames + banking."""
    var n_pts = len(path_pts)
    var m_prof = len(profile_2d)
    if n_pts < 2 or m_prof < 2:
        return NativeMesh()
    for i in range(n_pts):
        if not (isfinite(path_pts[i].x) and isfinite(path_pts[i].y) and isfinite(path_pts[i].z)):
            return NativeMesh()
    for j in range(m_prof):
        if not (isfinite(profile_2d[j].x) and isfinite(profile_2d[j].y)):
            return NativeMesh()
    var safe_up_hint = up_hint if (isfinite(up_hint.x) and isfinite(up_hint.y) and isfinite(up_hint.z)) else Vec3(0.0, 1.0, 0.0)
    var safe_uv_u = uv_scale_u if isfinite(uv_scale_u) else Float32(1.0)
    var safe_uv_v = uv_scale_v if isfinite(uv_scale_v) else Float32(1.0)

    # Compute 2D profile outward normals & arc-length UVs + concave pocket AO
    var prof_n2 = List[Vec2](length=m_prof, fill=Vec2(0.0, 1.0))
    var prof_u = List[Float32](capacity=m_prof + 1)
    var prof_ao = List[UInt8](length=m_prof, fill=UInt8(245))
    prof_u.append(Float32(0.0))

    var ccw_sign = Float32(1.0)
    if closed_profile and m_prof >= 3:
        var area2_64 = 0.0
        for j in range(m_prof):
            var a = profile_2d[j]
            var b = profile_2d[(j + 1) % m_prof]
            area2_64 += Float64(a.x) * Float64(b.y) - Float64(b.x) * Float64(a.y)
        if area2_64 < 0.0:
            ccw_sign = Float32(-1.0)

    var cen_x64 = 0.0
    var cen_y64 = 0.0
    for j in range(m_prof):
        cen_x64 += Float64(profile_2d[j].x)
        cen_y64 += Float64(profile_2d[j].y)
    cen_x64 /= Float64(m_prof)
    cen_y64 /= Float64(m_prof)
    var p_cen = Vec2(Float32(cen_x64), Float32(cen_y64))
    var max_pr64 = 1.0e-4
    for j in range(m_prof):
        var dx = Float64(profile_2d[j].x) - cen_x64
        var dy = Float64(profile_2d[j].y) - cen_y64
        var r_j64 = sqrt(dx * dx + dy * dy)
        if r_j64 > max_pr64:
            max_pr64 = r_j64

    var acc_u64 = 0.0
    for j in range(m_prof):
        var j_prev = (j + m_prof - 1) % m_prof if closed_profile else max(0, j - 1)
        var j_next = (j + 1) % m_prof if closed_profile else min(m_prof - 1, j + 1)
        var eax64 = Float64(profile_2d[j].x) - Float64(profile_2d[j_prev].x)
        var eay64 = Float64(profile_2d[j].y) - Float64(profile_2d[j_prev].y)
        var eal64 = sqrt(eax64 * eax64 + eay64 * eay64)
        var e_a = Vec2(Float32(eax64 / eal64), Float32(eay64 / eal64)) if eal64 >= 1.0e-6 else Vec2(1.0, 0.0)
        var ebx64 = Float64(profile_2d[j_next].x) - Float64(profile_2d[j].x)
        var eby64 = Float64(profile_2d[j_next].y) - Float64(profile_2d[j].y)
        var ebl64 = sqrt(ebx64 * ebx64 + eby64 * eby64)
        var e_b = Vec2(Float32(ebx64 / ebl64), Float32(eby64 / ebl64)) if ebl64 >= 1.0e-6 else Vec2(1.0, 0.0)
        var na = Vec2(e_a.y, -e_a.x) * ccw_sign
        var nb = Vec2(e_b.y, -e_b.x) * ccw_sign
        prof_n2[j] = _norm2d(na + nb)

        # Recessed web cavities (closer to profile centroid & facing inward/downward) receive natural cavity AO
        var dx_c = Float64(profile_2d[j].x) - cen_x64
        var dy_c = Float64(profile_2d[j].y) - cen_y64
        var rad_ratio64 = sqrt(dx_c * dx_c + dy_c * dy_c) / max_pr64
        var ao_f = Float32(0.58) + Float32(0.42) * min(Float32(1.0), Float32(rad_ratio64) if isfinite(rad_ratio64) else Float32(1.0))
        if prof_n2[j].y < Float32(-0.2):
            ao_f *= Float32(0.88)
        var safe_ao = max(Float32(0.0), min(Float32(1.0), ao_f)) if isfinite(ao_f) else Float32(0.75)
        prof_ao[j] = UInt8(max(95, min(255, Int(safe_ao * Float32(255.0) + Float32(0.5)))))

        acc_u64 += ebl64
        prof_u.append(Float32(min(3.4e38, acc_u64)))

    var inv_perim64 = Float64(safe_uv_u) / max(1.0e-4, acc_u64)
    var frames = compute_rmf_frames(path_pts, closed_loop=closed_loop, up_hint=safe_up_hint)
    ref rights = frames[0]
    ref ups = frames[1]
    ref tangs = frames[2]

    var ring_count = n_pts + (1 if closed_loop else 0)
    var v_cols = m_prof + (1 if closed_profile else 0)
    var mb = MeshBuilder(base_color=base_color, roughness=0.38, metallic=0.25)
    var base_v = mb.mesh.vertex_count()

    var path_dist64 = 0.0
    for s in range(ring_count):
        var si = s % n_pts
        if s > 0:
            var sip = (s - 1) % n_pts
            var pdx = Float64(path_pts[si].x) - Float64(path_pts[sip].x)
            var pdy = Float64(path_pts[si].y) - Float64(path_pts[sip].y)
            var pdz = Float64(path_pts[si].z) - Float64(path_pts[sip].z)
            path_dist64 += sqrt(pdx * pdx + pdy * pdy + pdz * pdz)
        var r_vec = rights[si]
        var u_vec = ups[si]
        if si < len(bank_angles_rad):
            var ba = bank_angles_rad[si]
            if isfinite(ba) and abs(ba) > Float32(1e-6):
                var cb = cos(ba)
                var sb = sin(ba)
                var nr = r_vec * cb + u_vec * sb
                var nu = u_vec * cb - r_vec * sb
                r_vec = nr
                u_vec = nu

        var origin = path_pts[si]
        var tv64 = path_dist64 * Float64(safe_uv_v)
        var tv = Float32(min(3.4e38, max(-3.4e38, tv64))) if isfinite(tv64) else Float32(0.0)

        for j in range(v_cols):
            var pj = j % m_prof
            var lp = profile_2d[pj]
            var ln = prof_n2[pj]
            var wp = origin + r_vec * lp.x + u_vec * lp.y
            var wn = (r_vec * ln.x + u_vec * ln.y).normalized()
            var tu = Float32(Float64(prof_u[j]) * inv_perim64)
            _ = mb.add_vertex(wp, wn, tu, tv, ao=prof_ao[pj])

    var seg_u = m_prof if closed_profile else (m_prof - 1)
    for s in range(ring_count - 1):
        for j in range(seg_u):
            var i0 = base_v + s * v_cols + j
            var i1 = i0 + 1
            var i2 = i0 + v_cols + 1
            var i3 = i0 + v_cols
            mb.add_quad(i0, i1, i2, i3)

    if cap_ends and not closed_loop and closed_profile and m_prof >= 3:
        # Start cap (`s = 0`, normal `-tangs[0]`) & end cap (`s = n_pts - 1`, normal `+tangs[n_pts - 1]`)
        for end_k in range(2):
            var si = 0 if end_k == 0 else (n_pts - 1)
            var cap_n = (-tangs[0]) if end_k == 0 else tangs[n_pts - 1]
            var r_vec = rights[si]
            var u_vec = ups[si]
            if si < len(bank_angles_rad):
                var ba = bank_angles_rad[si]
                if isfinite(ba):
                    var cb = cos(ba); var sb = sin(ba)
                    var nr = r_vec * cb + u_vec * sb
                    var nu = u_vec * cb - r_vec * sb
                    r_vec = nr; u_vec = nu
            var c_pos = path_pts[si] + r_vec * p_cen.x + u_vec * p_cen.y
            var c_v = mb.add_vertex(c_pos, cap_n, 0.5, 0.5, ao=UInt8(190))
            var cap_ring = List[Int](capacity=m_prof)
            for j in range(m_prof):
                var lp = profile_2d[j]
                var cap_u = Float32(0.5 + Float64(lp.x) / (max_pr64 * 2.0))
                var cap_v = Float32(0.5 + Float64(lp.y) / (max_pr64 * 2.0))
                cap_ring.append(
                    mb.add_vertex(
                        path_pts[si] + r_vec * lp.x + u_vec * lp.y,
                        cap_n,
                        cap_u,
                        cap_v,
                        ao=UInt8(210),
                    )
                )
            for j in range(m_prof):
                var jn = (j + 1) % m_prof
                if end_k == 0:
                    mb.add_tri(c_v, cap_ring[jn], cap_ring[j])
                else:
                    mb.add_tri(c_v, cap_ring[j], cap_ring[jn])

    var out = mb^.finish()
    out.ensure_bounds()
    return out^


def sweep_profile_rmf(
    path_pts: List[Vec3],
    profile_2d: List[Vec2],
    closed_loop: Bool = False,
    closed_profile: Bool = True,
    up_hint: Vec3 = Vec3(0.0, 1.0, 0.0),
    uv_scale_u: Float32 = 1.0,
    uv_scale_v: Float32 = 1.0,
    cap_ends: Bool = True,
    base_color: UInt32 = 0xFFFFFFFF,
) -> NativeMesh:
    """Twist-free Rotation-Minimizing Parallel-Transport spline loft along `path_pts`."""
    var empty_bank = List[Float32]()
    return sweep_profile_rmf_banked(
        path_pts,
        profile_2d,
        empty_bank,
        closed_loop=closed_loop,
        closed_profile=closed_profile,
        up_hint=up_hint,
        uv_scale_u=uv_scale_u,
        uv_scale_v=uv_scale_v,
        cap_ends=cap_ends,
        base_color=base_color,
    )


def make_chamfered_girder_profile(
    width: Float32 = 0.56,
    height: Float32 = 0.64,
    web_width: Float32 = 0.24,
    flange_thick: Float32 = 0.14,
    chamfer: Float32 = 0.035,
) -> List[Vec2]:
    """Builds a 16-vertex CCW chamfered I-beam / box-girder track spine cross-section."""
    var hw = max(Float32(0.08), width * Float32(0.5))
    var hh = max(Float32(0.10), height * Float32(0.5))
    var hweb = max(Float32(0.03), min(hw * Float32(0.75), web_width * Float32(0.5)))
    var ft = max(Float32(0.03), min(hh * Float32(0.45), flange_thick))
    var ch = max(Float32(0.005), min(chamfer, ft * Float32(0.4)))

    var pts = List[Vec2](capacity=16)
    # CCW starting bottom-right flange -> right recessed web -> top-right flange -> top-left -> left web -> bottom-left
    pts.append(Vec2(hw - ch, -hh))
    pts.append(Vec2(hw, -hh + ch))
    pts.append(Vec2(hw, -hh + ft))
    pts.append(Vec2(hweb, -hh + ft + ch))
    pts.append(Vec2(hweb, hh - ft - ch))
    pts.append(Vec2(hw, hh - ft))
    pts.append(Vec2(hw, hh - ch))
    pts.append(Vec2(hw - ch, hh))
    pts.append(Vec2(-hw + ch, hh))
    pts.append(Vec2(-hw, hh - ch))
    pts.append(Vec2(-hw, hh - ft))
    pts.append(Vec2(-hweb, hh - ft - ch))
    pts.append(Vec2(-hweb, -hh + ft + ch))
    pts.append(Vec2(-hw, -hh + ft))
    pts.append(Vec2(-hw, -hh + ch))
    pts.append(Vec2(-hw + ch, -hh))
    return pts^


def make_curb_profile(
    width: Float32 = 0.36,
    height: Float32 = 0.18,
    bevel: Float32 = 0.04,
) -> List[Vec2]:
    """Builds a chamfered granite street curb + gutter apron 2D profile (`CCW`)."""
    var hw = max(Float32(0.08), width * Float32(0.5))
    var h = max(Float32(0.06), height)
    var bv = max(Float32(0.01), min(bevel, min(hw * Float32(0.4), h * Float32(0.45))))

    var pts = List[Vec2](capacity=7)
    pts.append(Vec2(-hw, 0.0))
    pts.append(Vec2(hw, 0.0))
    pts.append(Vec2(hw, h * Float32(0.24)))
    pts.append(Vec2(hw * Float32(0.32), h * Float32(0.28)))
    pts.append(Vec2(hw * Float32(0.24), h - bv))
    pts.append(Vec2(hw * Float32(0.24) - bv, h))
    pts.append(Vec2(-hw, h))
    return pts^


def sweep_catenary_cable(
    p0: Vec3,
    p1: Vec3,
    sag: Float32 = 0.35,
    radius: Float32 = 0.018,
    segments: Int = 16,
    radial_sides: Int = 6,
) -> NativeMesh:
    """Sweeps a hanging catenary cable between `p0` and `p1` with `sag` vertical drop."""
    if not (
        isfinite(p0.x) and isfinite(p0.y) and isfinite(p0.z)
        and isfinite(p1.x) and isfinite(p1.y) and isfinite(p1.z)
    ):
        return NativeMesh()
    var safe_sag = sag if isfinite(sag) else Float32(0.0)
    var safe_rad = radius if isfinite(radius) else Float32(0.018)
    var steps = max(3, segments)
    var sides = max(4, radial_sides)
    var path = List[Vec3](capacity=steps + 1)
    for i in range(steps + 1):
        var t = Float32(i) / Float32(steps)
        var base_p = p0 * (Float32(1.0) - t) + p1 * t
        var drop = Float32(4.0) * safe_sag * t * (Float32(1.0) - t)
        path.append(Vec3(base_p.x, base_p.y - drop, base_p.z))

    var circle = List[Vec2](capacity=sides)
    var r = max(Float32(0.003), safe_rad)
    for k in range(sides):
        var a = (Float32(k) / Float32(sides)) * _TAU
        circle.append(Vec2(cos(a) * r, sin(a) * r))

    var mesh = sweep_profile_rmf(
        path,
        circle,
        closed_loop=False,
        closed_profile=True,
        uv_scale_u=1.0,
        uv_scale_v=1.5,
        cap_ends=True,
        base_color=0xFF212529,
    )
    mesh.roughness = 0.42
    mesh.metallic = 0.65
    return mesh^
