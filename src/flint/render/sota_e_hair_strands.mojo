"""Area E (`sota:E`): strand hair geometry with a card fallback.

Strand path: guide strands are grown from a scalp cap (deterministic hash seeding, gravity droop and
curl), then each strand polyline is expanded into a thin two-sided ribbon whose `+U` texture axis runs
root to tip. The cooked vertex tangent is `dP/dU`, i.e. the strand direction, which the hair material
model (`sota_e_characters`) shades with Marschner R/TT/TRT + dual-scattering back-scatter. Ribbons are
rasterised by the regular VisBuffer pass, so strands get shadows, TSR and GI like any mesh.

Card fallback: the same growth produces a few wide cards with an alpha-tested strand texture
(`sota_e_hair_card_texture`), for distant or budget-limited characters. `sota_e_pick_hair_lod` picks
the representation from the projected head height and a strand budget.
"""
from std.math import cos, max, min, sin, sqrt
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh
from flint.core.math import Vec3

comptime SOTA_E_HAIR_LOD_STRANDS: Int = 0
comptime SOTA_E_HAIR_LOD_CARDS: Int = 1


@fieldwise_init
struct SotaHairGroom(Copyable, Movable):
    var scalp_center: Vec3
    var scalp_radius: Float32
    var cap_angle: Float32       # polar extent of the scalp cap from +Y (radians)
    var strand_count: Int
    var segments: Int
    var length: Float32
    var width: Float32           # strand ribbon width (m)
    var droop: Float32           # gravity bend toward -Y over the strand
    var curl: Float32            # curl amplitude (m)
    var seed: UInt32

    @staticmethod
    def short_bob(center: Vec3, radius: Float32) -> Self:
        return Self(center, radius, Float32(1.35), 2400, 8, radius * Float32(1.15), Float32(0.0016), Float32(0.9), radius * Float32(0.03), UInt32(17))


def _hash01(x: UInt32) -> Float32:
    var h = x * UInt32(747796405) + UInt32(2891336453)
    h = ((h >> ((h >> UInt32(28)) + UInt32(4))) ^ h) * UInt32(277803737)
    h = (h >> UInt32(22)) ^ h
    return Float32(h & UInt32(0xFFFFFF)) / Float32(16777216.0)


def _ortho(n: Vec3) -> Vec3:
    var helper = Vec3(1.0, 0.0, 0.0) if abs(n.x) < Float32(0.9) else Vec3(0.0, 0.0, 1.0)
    return helper.cross(n).normalized()


def sota_e_strand_points(g: SotaHairGroom, idx: Int) -> List[Vec3]:
    """Root-to-tip polyline (`segments + 1` points) of strand `idx`."""
    var u1 = _hash01(g.seed * UInt32(9781) + UInt32(idx) * UInt32(2) + UInt32(1))
    var u2 = _hash01(g.seed * UInt32(6271) + UInt32(idx) * UInt32(2) + UInt32(2))
    var u3 = _hash01(g.seed * UInt32(3301) + UInt32(idx) * UInt32(3) + UInt32(5))
    # Uniform on the cap: cos(theta) uniform in [cos(cap), 1].
    var ct = Float32(1.0) - u1 * (Float32(1.0) - cos(g.cap_angle))
    var st = sqrt(max(Float32(0.0), Float32(1.0) - ct * ct))
    var ph = u2 * Float32(6.2831853)
    var n = Vec3(st * cos(ph), ct, st * sin(ph))
    var root = g.scalp_center + n * g.scalp_radius
    var side = _ortho(n)
    var pts = List[Vec3]()
    var p = root
    var dir = n
    var seg_len = g.length * (Float32(0.8) + Float32(0.4) * u3) / Float32(g.segments)
    pts.append(p)
    for k in range(g.segments):
        var tk = Float32(k + 1) / Float32(g.segments)
        # Bend toward gravity, keep strands outside the scalp sphere.
        dir = (dir + Vec3(0.0, -g.droop * (Float32(0.35) + Float32(0.5) * tk), 0.0)).normalized()
        var curl_off = side * (g.curl * sin(tk * Float32(9.0) + u3 * Float32(6.28)))
        var np = p + dir * seg_len + curl_off * Float32(0.25)
        var rel = np - g.scalp_center
        var rl = sqrt(rel.length_squared())
        var min_r = g.scalp_radius * (Float32(1.0) + Float32(0.02) * tk + Float32(0.005))
        if rl < min_r:
            np = g.scalp_center + rel * (min_r / max(Float32(1e-5), rl))
        p = np
        pts.append(p)
    return pts^


def _append_ribbon(mut mb: MeshBuilder, pts: List[Vec3], center: Vec3, width_root: Float32, width_tip: Float32, v0: Float32, v1: Float32):
    var n = len(pts)
    var base = List[Int]()
    for k in range(n):
        var t = (pts[min(n - 1, k + 1)] - pts[max(0, k - 1)]).normalized()
        var outward = (pts[k] - center).normalized()
        var side = t.cross(outward).normalized()
        var face = side.cross(t).normalized()
        var tk = Float32(k) / Float32(n - 1)
        var w = width_root + (width_tip - width_root) * tk
        var a = mb.add_vertex(pts[k] - side * (w * Float32(0.5)), face, tk, v0)
        var b = mb.add_vertex(pts[k] + side * (w * Float32(0.5)), face, tk, v1)
        base.append(a)
        _ = b
    for k in range(n - 1):
        var a0 = base[k]
        var b0 = a0 + 1
        var a1 = base[k + 1]
        var b1 = a1 + 1
        # Two-sided: both windings (front-face culling keeps exactly one per view).
        mb.mesh.indices.append(a0); mb.mesh.indices.append(b0); mb.mesh.indices.append(b1)
        mb.mesh.indices.append(a0); mb.mesh.indices.append(b1); mb.mesh.indices.append(a1)
        mb.mesh.indices.append(a0); mb.mesh.indices.append(b1); mb.mesh.indices.append(b0)
        mb.mesh.indices.append(a0); mb.mesh.indices.append(a1); mb.mesh.indices.append(b1)


def build_hair_strand_ribbons(g: SotaHairGroom) -> NativeMesh:
    """Strand path: one tapered two-sided ribbon per strand (`+U` = root to tip)."""
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.35, metallic=0.0)
    for i in range(g.strand_count):
        var pts = sota_e_strand_points(g, i)
        _append_ribbon(mb, pts, g.scalp_center, g.width, g.width * Float32(0.35), Float32(0.0), Float32(1.0))
    return mb^.finish()


def build_hair_cards(g: SotaHairGroom, card_count: Int, card_width: Float32) -> NativeMesh:
    """Card fallback: `card_count` wide ribbons (alpha-tested strand texture along `V`)."""
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.35, metallic=0.0)
    var stride = max(1, g.strand_count // max(1, card_count))
    for c in range(card_count):
        var pts = sota_e_strand_points(g, c * stride)
        _append_ribbon(mb, pts, g.scalp_center, card_width, card_width * Float32(0.6), Float32(0.0), Float32(1.0))
    return mb^.finish()


def sota_e_hair_card_texture(dim: Int, strands_across: Int, seed: UInt32) -> List[UInt32]:
    """ARGB8 card texture: white strands with alpha coverage (`alpha < 0.5` is cut). U = along the strand."""
    var px = List[UInt32](length=dim * dim, fill=0)
    for y in range(dim):
        for x in range(dim):
            var fv = (Float32(y) + Float32(0.5)) / Float32(dim)
            var fu = (Float32(x) + Float32(0.5)) / Float32(dim)
            var lane = fv * Float32(strands_across)
            var li = Int(lane)
            var jitter = _hash01(seed + UInt32(li) * UInt32(31)) * Float32(0.3)
            var d = abs((lane - Float32(li)) - Float32(0.5) - jitter + Float32(0.15))
            var taper = Float32(1.0) - fu * Float32(0.7)
            var cover = Float32(1.0) - min(Float32(1.0), d / (Float32(0.6) * taper))
            var tip_cut = _hash01(seed * UInt32(7) + UInt32(li) * UInt32(13)) * Float32(0.25)
            if fu > Float32(1.0) - tip_cut:
                cover = Float32(0.0)
            var a8 = UInt32(max(Float32(0.0), min(Float32(255.0), cover * Float32(255.0))))
            px[y * dim + x] = (a8 << 24) | UInt32(0xFFFFFF)
    return px^


def sota_e_pick_hair_lod(projected_head_px: Float32, strand_count: Int, strand_budget: Int) -> Int:
    """Strands when the head covers enough pixels that a strand is ~>= 0.35 px wide and the budget
    allows it; cards otherwise."""
    if strand_count > strand_budget:
        return SOTA_E_HAIR_LOD_CARDS
    if projected_head_px < Float32(160.0):
        return SOTA_E_HAIR_LOD_CARDS
    return SOTA_E_HAIR_LOD_STRANDS
