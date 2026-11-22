"""Signed distance fields from TrueType outlines (package 3.12).

Replaces `tools/bake_font.py` (Pillow, FreeType, SciPy). A glyph's
quadratic contours are flattened to line segments at the bake scale; every
texel of its cell gets the exact distance from its centre to the nearest
segment, signed by the nonzero winding of the outline (inside positive).
This is the distance the old baker approximated with a Euclidean distance
transform of a 4x supersampled bitmap, without the supersampling error.
The edge sits at 128 and 127 units are `spread` bake pixels (Green,
SIGGRAPH 2007), the convention `render/font.mojo` draws with.

The same `bake_glyph` serves the offline baker (`tools/bake_font.mojo`,
which writes the `.dsf` files) and the game's dynamic glyph cache (a glyph
missing from the baked atlas is baked from the font file on first use).
"""
from std.math import abs, sqrt, floor, ceil, sin, cos, acos
from flint.text.ttf import TrueTypeFont, GlyphOutline

comptime DSF_MAGIC: Int = 0x31465344  # "DSF1"
comptime CURVE_TOLERANCE: Float32 = 0.05   # flattening error, bake px
comptime MAX_BAKED_GLYPH_DIM: Int = 2048
comptime MAX_BAKED_GLYPH_BYTES: Int = 4 * 1024 * 1024

comptime EDGE_R: Int = 1
comptime EDGE_G: Int = 2
comptime EDGE_B: Int = 4
comptime EDGE_Y: Int = EDGE_R | EDGE_G  # 3
comptime EDGE_M: Int = EDGE_R | EDGE_B  # 5
comptime EDGE_C: Int = EDGE_G | EDGE_B  # 6
comptime EDGE_W: Int = EDGE_R | EDGE_G | EDGE_B  # 7


struct BakedGlyph(Copyable, Movable):
    var code: Int
    var glyph: Int
    var x: Int        # atlas position, set by the packer
    var y: Int
    var w: Int
    var h: Int
    var xoff: Float32   # cell top-left relative to the pen on the baseline, bake px
    var yoff: Float32
    var advance: Float32
    var channels: Int
    var pixels: List[UInt8]

    def __init__(out self):
        self.code = 0
        self.glyph = 0
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.xoff = 0
        self.yoff = 0
        self.advance = 0
        self.channels = 1
        self.pixels = List[UInt8]()


struct Segments(Movable):
    """Line segments in bake pixels, y down: (ax, ay, bx, by) per segment."""
    var values: List[Float32]

    def __init__(out self):
        self.values = List[Float32]()

    def add(mut self, ax: Float32, ay: Float32, bx: Float32, by: Float32):
        if ax == bx and ay == by:
            return
        self.values.append(ax)
        self.values.append(ay)
        self.values.append(bx)
        self.values.append(by)

    def count(self) -> Int:
        return len(self.values) // 4


def add_quad(mut segments: Segments, ax: Float32, ay: Float32, cx: Float32, cy: Float32, bx: Float32, by: Float32):
    # Error of a quadratic flattened into n pieces is |a - 2c + b| / (8 n^2).
    var ex = ax - 2 * cx + bx
    var ey = ay - 2 * cy + by
    var deviation = sqrt(ex * ex + ey * ey)
    var pieces = max(1, min(32, Int(ceil(sqrt(deviation / (8 * CURVE_TOLERANCE))))))
    var px = ax
    var py = ay
    for step in range(1, pieces + 1):
        var t = Float32(step) / Float32(pieces)
        var u = 1 - t
        var x = u * u * ax + 2 * u * t * cx + t * t * bx
        var y = u * u * ay + 2 * u * t * cy + t * t * by
        segments.add(px, py, x, y)
        px = x
        py = y


def add_cubic(
    mut segments: Segments,
    ax: Float32,
    ay: Float32,
    c1x: Float32,
    c1y: Float32,
    c2x: Float32,
    c2y: Float32,
    bx: Float32,
    by: Float32,
):
    # Wang's bound: max deviation of second difference controls flattening step count.
    var ex1 = ax - 2.0 * c1x + c2x
    var ey1 = ay - 2.0 * c1y + c2y
    var ex2 = c1x - 2.0 * c2x + bx
    var ey2 = c1y - 2.0 * c2y + by
    var dev = max(sqrt(ex1 * ex1 + ey1 * ey1), sqrt(ex2 * ex2 + ey2 * ey2))
    var pieces = max(1, min(64, Int(ceil(sqrt(Float32(0.75) * dev / CURVE_TOLERANCE)))))
    var px = ax
    var py = ay
    for step in range(1, pieces + 1):
        var t = Float32(step) / Float32(pieces)
        var u = Float32(1.0) - t
        var u2 = u * u
        var t2 = t * t
        var w0 = u2 * u
        var w1 = Float32(3.0) * u2 * t
        var w2 = Float32(3.0) * u * t2
        var w3 = t2 * t
        var x = w0 * ax + w1 * c1x + w2 * c2x + w3 * bx
        var y = w0 * ay + w1 * c1y + w2 * c2y + w3 * by
        segments.add(px, py, x, y)
        px = x
        py = y


def flatten(outline: GlyphOutline, scale: Float32) -> Segments:
    """Contours to segments in bake px (y flipped down). Supports TrueType
    quadratic contours (`is_cubic=False`) and CFF cubic contours (`is_cubic=True`)."""
    var segments = Segments()
    var n_pts = min(len(outline.xs), min(len(outline.ys), len(outline.on)))
    var start = 0
    for contour in range(len(outline.ends)):
        var end = outline.ends[contour]
        if start < 0 or end < start or start >= n_pts or end >= n_pts:
            if end >= start:
                start = end + 1
            continue
        var count = end - start + 1
        if count < 2:
            start = end + 1
            continue
        var xs = List[Float32]()
        var ys = List[Float32]()
        var on = List[Bool]()
        for p in range(start, end + 1):
            xs.append(outline.xs[p] * scale)
            ys.append(-outline.ys[p] * scale)
            on.append(outline.on[p])
        if outline.is_cubic:
            var first = -1
            for p in range(count):
                if on[p]:
                    first = p
                    break
            if first >= 0:
                var k = 0
                while k < count:
                    var i = (first + k) % count
                    var p1 = (first + k + 1) % count
                    if on[p1]:
                        segments.add(xs[i], ys[i], xs[p1], ys[p1])
                        k += 1
                    elif k + 3 <= count:
                        var c1 = p1
                        var c2 = (first + k + 2) % count
                        var end_pt = (first + k + 3) % count
                        if not on[c2] and on[end_pt]:
                            add_cubic(segments, xs[i], ys[i], xs[c1], ys[c1], xs[c2], ys[c2], xs[end_pt], ys[end_pt])
                            k += 3
                        elif on[c2]:
                            add_quad(segments, xs[i], ys[i], xs[c1], ys[c1], xs[c2], ys[c2])
                            k += 2
                        else:
                            k += 1
                    elif k + 2 <= count:
                        var c1 = p1
                        var end_pt = (first + k + 2) % count
                        if on[end_pt]:
                            add_quad(segments, xs[i], ys[i], xs[c1], ys[c1], xs[end_pt], ys[end_pt])
                            k += 2
                        else:
                            k += 1
                    else:
                        k += 1
            start = end + 1
            continue
        # Quadratic TrueType contour: start at an on-curve point (or the midpoint of two off ones).
        var first = -1
        for p in range(count):
            if on[p]:
                first = p
                break
        var sx: Float32
        var sy: Float32
        if first < 0:
            sx = (xs[0] + xs[count - 1]) * 0.5
            sy = (ys[0] + ys[count - 1]) * 0.5
            # Every point is off the curve: on-curve points sit between them.
            var cur_x = sx
            var cur_y = sy
            for k in range(count):
                var n = (k + 1) % count
                var mx = (xs[k] + xs[n]) * 0.5
                var my = (ys[k] + ys[n]) * 0.5
                add_quad(segments, cur_x, cur_y, xs[k], ys[k], mx, my)
                cur_x = mx
                cur_y = my
            start = end + 1
            continue
        sx = xs[first]
        sy = ys[first]
        var cur_x = sx
        var cur_y = sy
        var has_control = False
        var cx: Float32 = 0
        var cy: Float32 = 0
        for k in range(1, count + 1):
            var p = (first + k) % count
            var x = xs[p]
            var y = ys[p]
            if on[p]:
                if has_control:
                    add_quad(segments, cur_x, cur_y, cx, cy, x, y)
                else:
                    segments.add(cur_x, cur_y, x, y)
                cur_x = x
                cur_y = y
                has_control = False
            else:
                if has_control:
                    var mx = (cx + x) * 0.5
                    var my = (cy + y) * 0.5
                    add_quad(segments, cur_x, cur_y, cx, cy, mx, my)
                    cur_x = mx
                    cur_y = my
                cx = x
                cy = y
                has_control = True
        if has_control:
            add_quad(segments, cur_x, cur_y, cx, cy, sx, sy)
        start = end + 1
    return segments^


def signed_distance(segments: Segments, px: Float32, py: Float32) -> Float32:
    """Distance from (px, py) to the outline, positive inside (nonzero rule)."""
    var best: Float32 = 1e30
    var winding = 0
    var v = segments.values.unsafe_ptr()
    for index in range(segments.count()):
        var ax = v[unsafe_offset=index * 4]
        var ay = v[unsafe_offset=index * 4 + 1]
        var bx = v[unsafe_offset=index * 4 + 2]
        var by = v[unsafe_offset=index * 4 + 3]
        var dx = bx - ax
        var dy = by - ay
        var wx = px - ax
        var wy = py - ay
        var t = (wx * dx + wy * dy) / (dx * dx + dy * dy)
        t = max(Float32(0), min(Float32(1), t))
        var ox = wx - dx * t
        var oy = wy - dy * t
        best = min(best, ox * ox + oy * oy)
        # Crossings of the ray to +x; half-open in y so shared ends count once.
        if (ay <= py) != (by <= py):
            var cross_x = ax + (py - ay) * dx / dy
            if cross_x > px:
                winding += 1 if by > ay else -1
    var distance = sqrt(best)
    return distance if winding != 0 else -distance


@always_inline
def msdf_median(r: Float32, g: Float32, b: Float32) -> Float32:
    return max(min(r, g), min(max(r, g), b))


def _solve_quadratic_roots(a: Float32, b: Float32, c: Float32) -> List[Float32]:
    var roots = List[Float32]()
    if abs(a) < Float32(1.0e-8):
        if abs(b) > Float32(1.0e-8):
            roots.append(-c / b)
        return roots^
    var disc = b * b - Float32(4.0) * a * c
    if disc >= Float32(0.0):
        var s = sqrt(disc)
        roots.append((-b + s) / (Float32(2.0) * a))
        if disc > Float32(1.0e-8):
            roots.append((-b - s) / (Float32(2.0) * a))
    return roots^


def _solve_cubic_real(a: Float32, b: Float32, c: Float32, d: Float32) -> List[Float32]:
    if abs(a) < Float32(1.0e-8):
        return _solve_quadratic_roots(b, c, d)
    var inv_a = Float32(1.0) / a
    var bn = b * inv_a
    var cn = c * inv_a
    var dn = d * inv_a
    var bn3 = bn / Float32(3.0)
    var p = cn - bn * bn3
    var q = Float32(2.0) * bn3 * bn3 * bn3 - bn3 * cn + dn
    var half_q = q * Float32(0.5)
    var p3 = p / Float32(3.0)
    var disc = half_q * half_q + p3 * p3 * p3
    var roots = List[Float32]()
    if disc > Float32(1.0e-10):
        var sd = sqrt(disc)
        var u_arg = -half_q + sd
        var v_arg = -half_q - sd
        var u = (abs(u_arg) ** Float32(1.0 / 3.0)) * (Float32(1.0) if u_arg >= 0.0 else Float32(-1.0))
        var v = (abs(v_arg) ** Float32(1.0 / 3.0)) * (Float32(1.0) if v_arg >= 0.0 else Float32(-1.0))
        roots.append(u + v - bn3)
    elif p >= Float32(0.0):
        var t_single = (abs(q) ** Float32(1.0 / 3.0)) * (Float32(-1.0) if q >= 0.0 else Float32(1.0))
        roots.append(t_single - bn3)
    else:
        var r = Float32(2.0) * sqrt(max(Float32(0.0), -p3))
        var denom = max(Float32(1.0e-12), sqrt(max(Float32(0.0), -p3 * p3 * p3)))
        var cos_arg = max(Float32(-1.0), min(Float32(1.0), -half_q / denom))
        var phi = acos(cos_arg) / Float32(3.0)
        var two_pi_3 = Float32(2.0943951023931953)
        roots.append(r * cos(phi) - bn3)
        roots.append(r * cos(phi - two_pi_3) - bn3)
        roots.append(r * cos(phi + two_pi_3) - bn3)
    return roots^


@fieldwise_init
struct EdgeSegment(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int      # 0 = line, 1 = quad, 2 = cubic
    var color: Int     # bitmask of EDGE_R, EDGE_G, EDGE_B
    var p0x: Float32
    var p0y: Float32
    var c1x: Float32
    var c1y: Float32
    var c2x: Float32
    var c2y: Float32
    var p1x: Float32
    var p1y: Float32

    @staticmethod
    def line(ax: Float32, ay: Float32, bx: Float32, by: Float32, color: Int = EDGE_W) -> EdgeSegment:
        return EdgeSegment(0, color, ax, ay, ax, ay, bx, by, bx, by)

    @staticmethod
    def quad(ax: Float32, ay: Float32, cx: Float32, cy: Float32, bx: Float32, by: Float32, color: Int = EDGE_W) -> EdgeSegment:
        return EdgeSegment(1, color, ax, ay, cx, cy, cx, cy, bx, by)

    @staticmethod
    def cubic(
        ax: Float32,
        ay: Float32,
        c1x: Float32,
        c1y: Float32,
        c2x: Float32,
        c2y: Float32,
        bx: Float32,
        by: Float32,
        color: Int = EDGE_W,
    ) -> EdgeSegment:
        return EdgeSegment(2, color, ax, ay, c1x, c1y, c2x, c2y, bx, by)

    def point(self, t: Float32) -> Tuple[Float32, Float32]:
        var u = Float32(1.0) - t
        if self.kind == 0:
            return (self.p0x * u + self.p1x * t, self.p0y * u + self.p1y * t)
        if self.kind == 1:
            var w0 = u * u
            var w1 = Float32(2.0) * u * t
            var w2 = t * t
            return (w0 * self.p0x + w1 * self.c1x + w2 * self.p1x, w0 * self.p0y + w1 * self.c1y + w2 * self.p1y)
        var u2 = u * u
        var t2 = t * t
        var w0 = u2 * u
        var w1 = Float32(3.0) * u2 * t
        var w2 = Float32(3.0) * u * t2
        var w3 = t2 * t
        return (
            w0 * self.p0x + w1 * self.c1x + w2 * self.c2x + w3 * self.p1x,
            w0 * self.p0y + w1 * self.c1y + w2 * self.c2y + w3 * self.p1y,
        )

    def direction(self, t: Float32) -> Tuple[Float32, Float32]:
        if self.kind == 0:
            return (self.p1x - self.p0x, self.p1y - self.p0y)
        var u = Float32(1.0) - t
        if self.kind == 1:
            var dx = Float32(2.0) * u * (self.c1x - self.p0x) + Float32(2.0) * t * (self.p1x - self.c1x)
            var dy = Float32(2.0) * u * (self.c1y - self.p0y) + Float32(2.0) * t * (self.p1y - self.c1y)
            if dx * dx + dy * dy < Float32(1.0e-10):
                return (self.p1x - self.p0x, self.p1y - self.p0y)
            return (dx, dy)
        var dx = Float32(3.0) * u * u * (self.c1x - self.p0x) + Float32(6.0) * u * t * (self.c2x - self.c1x) + Float32(3.0) * t * t * (self.p1x - self.c2x)
        var dy = Float32(3.0) * u * u * (self.c1y - self.p0y) + Float32(6.0) * u * t * (self.c2y - self.c1y) + Float32(3.0) * t * t * (self.p1y - self.c2y)
        if dx * dx + dy * dy < Float32(1.0e-10):
            if t < Float32(0.5):
                dx = self.c2x - self.p0x
                dy = self.c2y - self.p0y
            else:
                dx = self.p1x - self.c1x
                dy = self.p1y - self.c1y
            if dx * dx + dy * dy < Float32(1.0e-10):
                return (self.p1x - self.p0x, self.p1y - self.p0y)
        return (dx, dy)

    def reverse(self) -> EdgeSegment:
        if self.kind == 0:
            return EdgeSegment.line(self.p1x, self.p1y, self.p0x, self.p0y, self.color)
        if self.kind == 1:
            return EdgeSegment.quad(self.p1x, self.p1y, self.c1x, self.c1y, self.p0x, self.p0y, self.color)
        return EdgeSegment.cubic(self.p1x, self.p1y, self.c2x, self.c2y, self.c1x, self.c1y, self.p0x, self.p0y, self.color)

    def split_at(self, t: Float32) -> Tuple[EdgeSegment, EdgeSegment]:
        if self.kind == 0:
            var m = self.point(t)
            return (
                EdgeSegment.line(self.p0x, self.p0y, m[0], m[1], self.color),
                EdgeSegment.line(m[0], m[1], self.p1x, self.p1y, self.color),
            )
        var u = Float32(1.0) - t
        if self.kind == 1:
            var q0x = self.p0x * u + self.c1x * t
            var q0y = self.p0y * u + self.c1y * t
            var q1x = self.c1x * u + self.p1x * t
            var q1y = self.c1y * u + self.p1y * t
            var mx = q0x * u + q1x * t
            var my = q0y * u + q1y * t
            return (
                EdgeSegment.quad(self.p0x, self.p0y, q0x, q0y, mx, my, self.color),
                EdgeSegment.quad(mx, my, q1x, q1y, self.p1x, self.p1y, self.color),
            )
        var a0x = self.p0x * u + self.c1x * t
        var a0y = self.p0y * u + self.c1y * t
        var a1x = self.c1x * u + self.c2x * t
        var a1y = self.c1y * u + self.c2y * t
        var a2x = self.c2x * u + self.p1x * t
        var a2y = self.c2y * u + self.p1y * t
        var b0x = a0x * u + a1x * t
        var b0y = a0y * u + a1y * t
        var b1x = a1x * u + a2x * t
        var b1y = a1y * u + a2y * t
        var mx = b0x * u + b1x * t
        var my = b0y * u + b1y * t
        return (
            EdgeSegment.cubic(self.p0x, self.p0y, a0x, a0y, b0x, b0y, mx, my, self.color),
            EdgeSegment.cubic(mx, my, b1x, b1y, a2x, a2y, self.p1x, self.p1y, self.color),
        )

    def signed_distance_param(self, px: Float32, py: Float32) -> Tuple[Float32, Float32, Float32]:
        """Returns (signed_distance, orthogonality, raw_param_t)."""
        if self.kind == 0:
            var abx = self.p1x - self.p0x
            var aby = self.p1y - self.p0y
            var apx = px - self.p0x
            var apy = py - self.p0y
            var ab_len2 = abx * abx + aby * aby
            if ab_len2 < Float32(1.0e-12):
                var d0 = sqrt(apx * apx + apy * apy)
                return (-d0, Float32(0.0), Float32(0.0))
            var t_raw = (apx * abx + apy * aby) / ab_len2
            var t = max(Float32(0.0), min(Float32(1.0), t_raw))
            var qx = self.p0x + abx * t
            var qy = self.p0y + aby * t
            var dx = px - qx
            var dy = py - qy
            var dist = sqrt(dx * dx + dy * dy)
            var cross_s = abx * apy - aby * apx
            var signed_d = dist if cross_s >= Float32(0.0) else -dist
            var ab_len = sqrt(ab_len2)
            var ortho = Float32(1.0)
            if dist > Float32(1.0e-6):
                ortho = abs((abx / ab_len) * (dy / dist) - (aby / ab_len) * (dx / dist))
            return (signed_d, ortho, t_raw)

        if self.kind == 1:
            var qax = self.p0x - px
            var qay = self.p0y - py
            var qbx = self.c1x - self.p0x
            var qby = self.c1y - self.p0y
            var qcx = self.p0x - Float32(2.0) * self.c1x + self.p1x
            var qcy = self.p0y - Float32(2.0) * self.c1y + self.p1y
            var a_coef = qcx * qcx + qcy * qcy
            var b_coef = Float32(3.0) * (qbx * qcx + qby * qcy)
            var c_coef = Float32(2.0) * (qbx * qbx + qby * qby) + (qax * qcx + qay * qcy)
            var d_coef = qax * qbx + qay * qby
            var roots = _solve_cubic_real(a_coef, b_coef, c_coef, d_coef)
            var best_d2 = (self.p0x - px) * (self.p0x - px) + (self.p0y - py) * (self.p0y - py)
            var best_t: Float32 = 0.0
            var end_d2 = (self.p1x - px) * (self.p1x - px) + (self.p1y - py) * (self.p1y - py)
            if end_d2 < best_d2:
                best_d2 = end_d2
                best_t = 1.0
            for rt in roots:
                if rt > Float32(0.0) and rt < Float32(1.0):
                    var pt = self.point(rt)
                    var dd = (pt[0] - px) * (pt[0] - px) + (pt[1] - py) * (pt[1] - py)
                    if dd < best_d2:
                        best_d2 = dd
                        best_t = rt
            var best_pt = self.point(best_t)
            var dir_v = self.direction(best_t)
            var dir_len = sqrt(dir_v[0] * dir_v[0] + dir_v[1] * dir_v[1])
            var vx = px - best_pt[0]
            var vy = py - best_pt[1]
            var dist = sqrt(best_d2)
            var cross_s = dir_v[0] * vy - dir_v[1] * vx
            var signed_d = dist if cross_s >= Float32(0.0) else -dist
            var ortho = Float32(1.0)
            if dist > Float32(1.0e-6) and dir_len > Float32(1.0e-6):
                ortho = abs((dir_v[0] / dir_len) * (vy / dist) - (dir_v[1] / dir_len) * (vx / dist))
            var t_raw = best_t
            if best_t == Float32(0.0) and dir_len > Float32(1.0e-6):
                var d0 = self.direction(0.0)
                var dl0 = d0[0] * d0[0] + d0[1] * d0[1]
                if dl0 > Float32(1.0e-10) and ((px - self.p0x) * d0[0] + (py - self.p0y) * d0[1]) < Float32(0.0):
                    t_raw = -Float32(0.1)
            elif best_t == Float32(1.0) and dir_len > Float32(1.0e-6):
                var d1 = self.direction(1.0)
                var dl1 = d1[0] * d1[0] + d1[1] * d1[1]
                if dl1 > Float32(1.0e-10) and ((px - self.p1x) * d1[0] + (py - self.p1y) * d1[1]) > Float32(0.0):
                    t_raw = Float32(1.1)
            return (signed_d, ortho, t_raw)

        # Cubic Bezier: 5-sample coarse search + 4-step Newton-Raphson refinement
        var best_t: Float32 = 0.0
        var best_d2: Float32 = 1.0e30
        for s in range(5):
            var ts = Float32(s) * Float32(0.25)
            var pt = self.point(ts)
            var dd = (pt[0] - px) * (pt[0] - px) + (pt[1] - py) * (pt[1] - py)
            if dd < best_d2:
                best_d2 = dd
                best_t = ts
        var t = best_t
        for _ in range(4):
            var pt = self.point(t)
            var u = Float32(1.0) - t
            var d1x = Float32(3.0) * u * u * (self.c1x - self.p0x) + Float32(6.0) * u * t * (self.c2x - self.c1x) + Float32(3.0) * t * t * (self.p1x - self.c2x)
            var d1y = Float32(3.0) * u * u * (self.c1y - self.p0y) + Float32(6.0) * u * t * (self.c2y - self.c1y) + Float32(3.0) * t * t * (self.p1y - self.c2y)
            var d2x = Float32(6.0) * u * (self.c2x - Float32(2.0) * self.c1x + self.p0x) + Float32(6.0) * t * (self.p1x - Float32(2.0) * self.c2x + self.c1x)
            var d2y = Float32(6.0) * u * (self.c2y - Float32(2.0) * self.c1y + self.p0y) + Float32(6.0) * t * (self.p1y - Float32(2.0) * self.c2y + self.c1y)
            var rx = pt[0] - px
            var ry = pt[1] - py
            var f_val = rx * d1x + ry * d1y
            var df_val = d1x * d1x + d1y * d1y + rx * d2x + ry * d2y
            if abs(df_val) > Float32(1.0e-8):
                var t_next = max(Float32(0.0), min(Float32(1.0), t - f_val / df_val))
                var pt_next = self.point(t_next)
                var dd_next = (pt_next[0] - px) * (pt_next[0] - px) + (pt_next[1] - py) * (pt_next[1] - py)
                if dd_next <= best_d2:
                    t = t_next
                    best_d2 = dd_next
        var p0_d2 = (self.p0x - px) * (self.p0x - px) + (self.p0y - py) * (self.p0y - py)
        if p0_d2 < best_d2:
            best_d2 = p0_d2
            t = 0.0
        var p1_d2 = (self.p1x - px) * (self.p1x - px) + (self.p1y - py) * (self.p1y - py)
        if p1_d2 < best_d2:
            best_d2 = p1_d2
            t = 1.0
        var best_pt = self.point(t)
        var dir_v = self.direction(t)
        var dir_len = sqrt(dir_v[0] * dir_v[0] + dir_v[1] * dir_v[1])
        var vx = px - best_pt[0]
        var vy = py - best_pt[1]
        var dist = sqrt(best_d2)
        var cross_s = dir_v[0] * vy - dir_v[1] * vx
        var signed_d = dist if cross_s >= Float32(0.0) else -dist
        var ortho = Float32(1.0)
        if dist > Float32(1.0e-6) and dir_len > Float32(1.0e-6):
            ortho = abs((dir_v[0] / dir_len) * (vy / dist) - (dir_v[1] / dir_len) * (vx / dist))
        var t_raw = t
        if t == Float32(0.0):
            var d0 = self.direction(0.0)
            if (px - self.p0x) * d0[0] + (py - self.p0y) * d0[1] < Float32(0.0):
                t_raw = -Float32(0.1)
        elif t == Float32(1.0):
            var d1 = self.direction(1.0)
            if (px - self.p1x) * d1[0] + (py - self.p1y) * d1[1] > Float32(0.0):
                t_raw = Float32(1.1)
        return (signed_d, ortho, t_raw)

    def to_pseudo_distance(self, px: Float32, py: Float32, d: Float32, t_raw: Float32) -> Float32:
        if t_raw < Float32(0.0):
            var dir_v = self.direction(0.0)
            var dlen = sqrt(dir_v[0] * dir_v[0] + dir_v[1] * dir_v[1])
            if dlen > Float32(1.0e-6):
                var nx = dir_v[0] / dlen
                var ny = dir_v[1] / dlen
                var aqx = px - self.p0x
                var aqy = py - self.p0y
                var ts = aqx * nx + aqy * ny
                if ts < Float32(0.0):
                    var pd = nx * aqy - ny * aqx
                    if abs(pd) <= abs(d):
                        return pd
        elif t_raw > Float32(1.0):
            var dir_v = self.direction(1.0)
            var dlen = sqrt(dir_v[0] * dir_v[0] + dir_v[1] * dir_v[1])
            if dlen > Float32(1.0e-6):
                var nx = dir_v[0] / dlen
                var ny = dir_v[1] / dlen
                var bqx = px - self.p1x
                var bqy = py - self.p1y
                var ts = bqx * nx + bqy * ny
                if ts > Float32(0.0):
                    var pd = nx * bqy - ny * bqx
                    if abs(pd) <= abs(d):
                        return pd
        return d


struct EdgeContour(Copyable, Movable):
    var edges: List[EdgeSegment]

    def __init__(out self):
        self.edges = List[EdgeSegment]()

    def signed_area(self) -> Float32:
        var area2: Float32 = 0.0
        for e in self.edges:
            if e.kind == 0:
                area2 += e.p0x * e.p1y - e.p1x * e.p0y
            else:
                var prev = e.point(0.0)
                for s in range(1, 5):
                    var cur = e.point(Float32(s) * Float32(0.25))
                    area2 += prev[0] * cur[1] - cur[0] * prev[1]
                    prev = cur
        return area2 * Float32(0.5)

    def reverse(mut self):
        var rev = List[EdgeSegment](capacity=len(self.edges))
        var i = len(self.edges) - 1
        while i >= 0:
            rev.append(self.edges[i].reverse())
            i -= 1
        self.edges = rev^


def _edge_non_degenerate(e: EdgeSegment) -> Bool:
    var dx = e.p1x - e.p0x
    var dy = e.p1y - e.p0y
    if e.kind == 0:
        return dx * dx + dy * dy > Float32(1.0e-8)
    var d1x = e.c1x - e.p0x
    var d1y = e.c1y - e.p0y
    var d2x = e.p1x - e.c2x
    var d2y = e.p1y - e.c2y
    return (dx * dx + dy * dy + d1x * d1x + d1y * d1y + d2x * d2x + d2y * d2y) > Float32(1.0e-8)


def extract_edge_contours(outline: GlyphOutline, scale: Float32) -> List[EdgeContour]:
    """Extracts exact Line/Quad/Cubic EdgeContours in y-down bake pixels and
    normalizes winding so TTF (CW y-up) and CFF (CCW y-up) share positive interior sign."""
    var contours = List[EdgeContour]()
    var n_pts = min(len(outline.xs), min(len(outline.ys), len(outline.on)))
    var start = 0
    for c_idx in range(len(outline.ends)):
        var end = outline.ends[c_idx]
        if start < 0 or end < start or start >= n_pts or end >= n_pts:
            if end >= start:
                start = end + 1
            continue
        var count = end - start + 1
        if count < 2:
            start = end + 1
            continue
        var xs = List[Float32]()
        var ys = List[Float32]()
        var on = List[Bool]()
        for p in range(start, end + 1):
            xs.append(outline.xs[p] * scale)
            ys.append(-outline.ys[p] * scale)
            on.append(outline.on[p])
        var ec = EdgeContour()
        if outline.is_cubic:
            var first = -1
            for p in range(count):
                if on[p]:
                    first = p
                    break
            if first >= 0:
                var k = 0
                while k < count:
                    var i = (first + k) % count
                    var p1 = (first + k + 1) % count
                    if on[p1]:
                        var seg = EdgeSegment.line(xs[i], ys[i], xs[p1], ys[p1])
                        if _edge_non_degenerate(seg):
                            ec.edges.append(seg)
                        k += 1
                    elif k + 3 <= count:
                        var c1 = p1
                        var c2 = (first + k + 2) % count
                        var end_pt = (first + k + 3) % count
                        if not on[c2] and on[end_pt]:
                            var seg = EdgeSegment.cubic(xs[i], ys[i], xs[c1], ys[c1], xs[c2], ys[c2], xs[end_pt], ys[end_pt])
                            if _edge_non_degenerate(seg):
                                ec.edges.append(seg)
                            k += 3
                        elif on[c2]:
                            var seg = EdgeSegment.quad(xs[i], ys[i], xs[c1], ys[c1], xs[c2], ys[c2])
                            if _edge_non_degenerate(seg):
                                ec.edges.append(seg)
                            k += 2
                        else:
                            k += 1
                    elif k + 2 <= count:
                        var c1 = p1
                        var end_pt = (first + k + 2) % count
                        if on[end_pt]:
                            var seg = EdgeSegment.quad(xs[i], ys[i], xs[c1], ys[c1], xs[end_pt], ys[end_pt])
                            if _edge_non_degenerate(seg):
                                ec.edges.append(seg)
                            k += 2
                        else:
                            k += 1
                    else:
                        k += 1
        else:
            var first = -1
            for p in range(count):
                if on[p]:
                    first = p
                    break
            if first < 0:
                var sx = (xs[0] + xs[count - 1]) * Float32(0.5)
                var sy = (ys[0] + ys[count - 1]) * Float32(0.5)
                var cur_x = sx
                var cur_y = sy
                for k in range(count):
                    var n = (k + 1) % count
                    var mx = (xs[k] + xs[n]) * Float32(0.5)
                    var my = (ys[k] + ys[n]) * Float32(0.5)
                    var seg = EdgeSegment.quad(cur_x, cur_y, xs[k], ys[k], mx, my)
                    if _edge_non_degenerate(seg):
                        ec.edges.append(seg)
                    cur_x = mx
                    cur_y = my
            else:
                var sx = xs[first]
                var sy = ys[first]
                var cur_x = sx
                var cur_y = sy
                var has_control = False
                var cx: Float32 = 0
                var cy: Float32 = 0
                for k in range(1, count + 1):
                    var p = (first + k) % count
                    var x = xs[p]
                    var y = ys[p]
                    if on[p]:
                        var seg = EdgeSegment.quad(cur_x, cur_y, cx, cy, x, y) if has_control else EdgeSegment.line(cur_x, cur_y, x, y)
                        if _edge_non_degenerate(seg):
                            ec.edges.append(seg)
                        cur_x = x
                        cur_y = y
                        has_control = False
                    else:
                        if has_control:
                            var mx = (cx + x) * Float32(0.5)
                            var my = (cy + y) * Float32(0.5)
                            var seg = EdgeSegment.quad(cur_x, cur_y, cx, cy, mx, my)
                            if _edge_non_degenerate(seg):
                                ec.edges.append(seg)
                            cur_x = mx
                            cur_y = my
                        cx = x
                        cy = y
                        has_control = True
                if has_control:
                    var seg = EdgeSegment.quad(cur_x, cur_y, cx, cy, sx, sy)
                    if _edge_non_degenerate(seg):
                        ec.edges.append(seg)
        if len(ec.edges) > 0:
            contours.append(ec^)
        start = end + 1

    # Normalize overall winding so outer contours have positive signed area in y-down
    var total_area: Float32 = 0.0
    for c in contours:
        total_area += c.signed_area()
    if total_area < Float32(0.0):
        for i in range(len(contours)):
            contours[i].reverse()
    return contours^


def _is_corner(a: EdgeSegment, b: EdgeSegment, sin_thr: Float32, cos_thr: Float32) -> Bool:
    var d1 = a.direction(1.0)
    var d2 = b.direction(0.0)
    var l1 = sqrt(d1[0] * d1[0] + d1[1] * d1[1])
    var l2 = sqrt(d2[0] * d2[0] + d2[1] * d2[1])
    if l1 < Float32(1.0e-6) or l2 < Float32(1.0e-6):
        return True
    var n1x = d1[0] / l1; var n1y = d1[1] / l1
    var n2x = d2[0] / l2; var n2y = d2[1] / l2
    var cr = abs(n1x * n2y - n1y * n2x)
    var dt = n1x * n2x + n1y * n2y
    return dt <= Float32(0.0) or cr > sin_thr or dt < cos_thr


def color_edges(mut contours: List[EdgeContour], angle_threshold: Float32 = 3.0):
    """Chlumsky edge-coloring: assigns multi-channel colors (M/Y/C/W) so every
    sharp corner lies between two edges with distinct non-overlapping channels."""
    var sin_thr = sin(angle_threshold)
    var cos_thr = cos(angle_threshold)
    var palette = [EDGE_M, EDGE_Y, EDGE_C]

    for c_idx in range(len(contours)):
        var n = len(contours[c_idx].edges)
        if n == 0:
            continue
        var corners = List[Int]()
        for i in range(n):
            var prev = contours[c_idx].edges[(i + n - 1) % n]
            var cur = contours[c_idx].edges[i]
            if _is_corner(prev, cur, sin_thr, cos_thr):
                corners.append(i)
        if len(corners) == 0:
            for i in range(n):
                contours[c_idx].edges[i].color = EDGE_W
        elif len(corners) == 1:
            var c0 = corners[0]
            if n == 1:
                var e0 = contours[c_idx].edges[0]
                var s1 = e0.split_at(Float32(1.0 / 3.0))
                var s2 = s1[1].split_at(Float32(0.5))
                var p0 = s1[0]; p0.color = EDGE_C
                var p1 = s2[0]; p1.color = EDGE_W
                var p2 = s2[1]; p2.color = EDGE_M
                contours[c_idx].edges = [p0, p1, p2]
            else:
                for i in range(n):
                    contours[c_idx].edges[i].color = EDGE_W
                contours[c_idx].edges[c0].color = EDGE_M
                contours[c_idx].edges[(c0 + n - 1) % n].color = EDGE_Y
        else:
            var m = len(corners)
            var rot = 0
            var seg_lens = List[Float32](length=m, fill=0.0)
            for k in range(m):
                var idx = corners[k]
                var end_e = corners[(k + 1) % m]
                var slen: Float32 = 0.0
                while True:
                    var ed = contours[c_idx].edges[idx]
                    var dx = ed.p1x - ed.p0x
                    var dy = ed.p1y - ed.p0y
                    slen += sqrt(dx * dx + dy * dy)
                    idx = (idx + 1) % n
                    if idx == end_e:
                        break
                seg_lens[k] = slen
            if m == 4 and seg_lens[1] + seg_lens[3] > seg_lens[0] + seg_lens[2]:
                rot = 1
            var logical_colors = List[Int](length=m, fill=EDGE_M)
            var color_idx = c_idx % 3
            for j in range(m):
                var k = (j + rot) % m
                var col = palette[color_idx % 3]
                if j == m - 1 and (color_idx % 3) == (c_idx % 3):
                    col = palette[(color_idx + 1) % 3]
                logical_colors[k] = col
                color_idx += 1
            if m >= 5:
                var i1 = 0
                var i2 = 1
                if seg_lens[i2] > seg_lens[i1]:
                    i1 = 1
                    i2 = 0
                for k in range(2, m):
                    if seg_lens[k] > seg_lens[i1]:
                        i2 = i1
                        i1 = k
                    elif seg_lens[k] > seg_lens[i2]:
                        i2 = k
                if logical_colors[i1] == logical_colors[i2]:
                    var c1 = logical_colors[i1]
                    var left = logical_colors[(i2 - 1 + m) % m]
                    var right = logical_colors[(i2 + 1) % m]
                    var alt = 0
                    for p in palette:
                        if p != c1 and p != left and p != right:
                            alt = p
                            break
                    if alt != 0:
                        logical_colors[i2] = alt
                    else:
                        logical_colors[i2] = right
                        var n2 = (i2 + 1) % m
                        var n3 = (i2 + 2) % m
                        for p in palette:
                            if p != right and p != logical_colors[n3]:
                                logical_colors[n2] = p
                                break
            for k in range(m):
                var col = logical_colors[k]
                var idx = corners[k]
                var end_e = corners[(k + 1) % m]
                while True:
                    contours[c_idx].edges[idx].color = col
                    idx = (idx + 1) % n
                    if idx == end_e:
                        break


def _is_msdf_artifact_pair(
    r0: Float32, g0: Float32, b0: Float32,
    r1: Float32, g1: Float32, b1: Float32,
    min_dev: Float32,
    thr: Float32,
) -> Tuple[Bool, Float32, Float32]:
    """Checks if bilinear interpolation between two texels produces a genuine
    MSDF channel-jump artifact (violating distance-field Lipschitz continuity).
    Returns `(is_artifact, t_eval, median_at_t)`."""
    var dr = abs(r1 - r0)
    var dg = abs(g1 - g0)
    var db = abs(b1 - b0)
    if dr <= min_dev and dg <= min_dev and db <= min_dev:
        return (False, Float32(0.5), Float32(0.0))
    var m0 = msdf_median(r0, g0, b0)
    var m1 = msdf_median(r1, g1, b1)
    var mm = msdf_median(Float32(0.5) * (r0 + r1), Float32(0.5) * (g0 + g1), Float32(0.5) * (b0 + b1))
    var lin_m = Float32(0.5) * (m0 + m1)
    if abs(mm - lin_m) > thr and ((m0 < Float32(128.0)) == (m1 < Float32(128.0))) and ((mm < Float32(128.0)) != (m0 < Float32(128.0))):
        return (True, Float32(0.5), mm)
    for pair in range(3):
        var a0: Float32; var b0v: Float32; var c0: Float32
        var a1: Float32; var b1v: Float32; var c1: Float32
        if pair == 0:
            a0 = r0; b0v = g0; c0 = b0; a1 = r1; b1v = g1; c1 = b1
        elif pair == 1:
            a0 = g0; b0v = b0; c0 = r0; a1 = g1; b1v = b1; c1 = r1
        else:
            a0 = r0; b0v = b0; c0 = g0; a1 = r1; b1v = b1; c1 = g1
        if (a0 - b0v) * (a1 - b1v) < Float32(0.0):
            var da = a1 - a0
            var dbv = b1v - b0v
            if abs(da) > min_dev and abs(dbv) > min_dev and da * dbv < Float32(0.0):
                var denom = da - dbv
                if abs(denom) > Float32(1.0e-5):
                    var t = (b0v - a0) / denom
                    if t > Float32(0.0) and t < Float32(1.0):
                        var mt = msdf_median(a0 + t * da, b0v + t * dbv, c0 + t * (c1 - c0))
                        var lin_t = m0 + t * (m1 - m0)
                        if abs(mt - lin_t) > thr and ((m0 < Float32(128.0)) == (m1 < Float32(128.0))) and ((mt < Float32(128.0)) != (m0 < Float32(128.0))):
                            return (True, t, mt)
    return (False, Float32(0.5), mm)


def _msdf_error_correct_impl(
    mut pixels: List[UInt8],
    w: Int,
    h: Int,
    channels: Int,
    spread: Float32,
    segments: Segments,
    gx0: Float32,
    gy0: Float32,
    verify_shape: Bool,
):
    if w <= 1 or h <= 1 or channels < 3:
        return
    var to_units = Float32(127.0) / max(Float32(1.0), spread)
    var thr = Float32(1.0 / 3.0) * to_units
    var min_dev_ortho = Float32(1.111111) * to_units
    var min_dev_diag = Float32(1.111111) * Float32(1.41421356) * to_units
    var med_vals = List[UInt8](length=w * h, fill=128)
    for idx in range(w * h):
        var base = idx * channels
        var m = msdf_median(Float32(pixels[base]), Float32(pixels[base + 1]), Float32(pixels[base + 2]))
        med_vals[idx] = UInt8(max(0, min(255, Int(floor(m + Float32(0.5))))))
    var suspect = List[Bool](length=w * h, fill=False)
    for y in range(h):
        for x in range(w):
            var i0 = (y * w + x) * channels
            var r0 = Float32(pixels[i0]); var g0 = Float32(pixels[i0 + 1]); var b0 = Float32(pixels[i0 + 2])
            if x + 1 < w:
                var i1 = (y * w + x + 1) * channels
                var r1 = Float32(pixels[i1]); var g1 = Float32(pixels[i1 + 1]); var b1 = Float32(pixels[i1 + 2])
                var chk = _is_msdf_artifact_pair(r0, g0, b0, r1, g1, b1, min_dev_ortho, thr)
                if chk[0]:
                    var is_err = True
                    if verify_shape and segments.count() > 0:
                        var px = gx0 + Float32(x) + Float32(0.5) + chk[1]
                        var py = gy0 + Float32(y) + Float32(0.5)
                        var td = signed_distance(segments, px, py)
                        is_err = (chk[2] >= Float32(128.0)) != (td >= Float32(0.0))
                    if is_err:
                        suspect[y * w + x] = True
                        suspect[y * w + x + 1] = True
            if y + 1 < h:
                var i2 = ((y + 1) * w + x) * channels
                var r2 = Float32(pixels[i2]); var g2 = Float32(pixels[i2 + 1]); var b2 = Float32(pixels[i2 + 2])
                var chk = _is_msdf_artifact_pair(r0, g0, b0, r2, g2, b2, min_dev_ortho, thr)
                if chk[0]:
                    var is_err = True
                    if verify_shape and segments.count() > 0:
                        var px = gx0 + Float32(x) + Float32(0.5)
                        var py = gy0 + Float32(y) + Float32(0.5) + chk[1]
                        var td = signed_distance(segments, px, py)
                        is_err = (chk[2] >= Float32(128.0)) != (td >= Float32(0.0))
                    if is_err:
                        suspect[y * w + x] = True
                        suspect[(y + 1) * w + x] = True
            if x + 1 < w and y + 1 < h:
                var i3 = ((y + 1) * w + x + 1) * channels
                var r3 = Float32(pixels[i3]); var g3 = Float32(pixels[i3 + 1]); var b3 = Float32(pixels[i3 + 2])
                var chk1 = _is_msdf_artifact_pair(r0, g0, b0, r3, g3, b3, min_dev_diag, thr)
                if chk1[0]:
                    var is_err = True
                    if verify_shape and segments.count() > 0:
                        var px = gx0 + Float32(x) + Float32(0.5) + chk1[1]
                        var py = gy0 + Float32(y) + Float32(0.5) + chk1[1]
                        var td = signed_distance(segments, px, py)
                        is_err = (chk1[2] >= Float32(128.0)) != (td >= Float32(0.0))
                    if is_err:
                        suspect[y * w + x] = True
                        suspect[(y + 1) * w + x + 1] = True
                var i_tr = (y * w + x + 1) * channels
                var i_bl = ((y + 1) * w + x) * channels
                var r_tr = Float32(pixels[i_tr]); var g_tr = Float32(pixels[i_tr + 1]); var b_tr = Float32(pixels[i_tr + 2])
                var r_bl = Float32(pixels[i_bl]); var g_bl = Float32(pixels[i_bl + 1]); var b_bl = Float32(pixels[i_bl + 2])
                var chk2 = _is_msdf_artifact_pair(r_tr, g_tr, b_tr, r_bl, g_bl, b_bl, min_dev_diag, thr)
                if chk2[0]:
                    var is_err = True
                    if verify_shape and segments.count() > 0:
                        var px = gx0 + Float32(x + 1) + Float32(0.5) - chk2[1]
                        var py = gy0 + Float32(y) + Float32(0.5) + chk2[1]
                        var td = signed_distance(segments, px, py)
                        is_err = (chk2[2] >= Float32(128.0)) != (td >= Float32(0.0))
                    if is_err:
                        suspect[y * w + x + 1] = True
                        suspect[(y + 1) * w + x] = True
    for idx in range(w * h):
        if suspect[idx]:
            var m = med_vals[idx]
            var base = idx * channels
            pixels[base] = m
            pixels[base + 1] = m
            pixels[base + 2] = m


def msdf_error_correct(mut pixels: List[UInt8], w: Int, h: Int, channels: Int, spread: Float32):
    """Detects and corrects bilinear channel-crossing interpolation artifacts
    while keeping alpha (channel 3 in MTSDF) and sub-pixel thin/acute features intact."""
    _msdf_error_correct_impl(pixels, w, h, channels, spread, Segments(), Float32(0.0), Float32(0.0), False)


def _bake_glyph_msdf_internal(
    font: TrueTypeFont,
    glyph_id: Int,
    bake_px: Int,
    spread: Int,
    code: Int = 0,
    metric_scale: Float32 = 1.0,
    channels: Int = 3,
) -> BakedGlyph:
    var baked = BakedGlyph()
    baked.code = code
    baked.glyph = glyph_id
    baked.channels = channels
    if bake_px <= 0 or spread < 0 or metric_scale <= Float32(0.0):
        return baked^
    var scale = (Float32(bake_px) / Float32(max(1, font.units_per_em))) * metric_scale
    if code != 0 and invisible(code):
        return baked^
    var outline = font.outline(baked.glyph)
    baked.advance = outline.advance * scale
    if outline.point_count() == 0 or len(outline.ends) == 0:
        return baked^
    var box = outline.bounds()
    var fx0 = box[0] * scale
    var fy0 = -box[3] * scale
    var fx1 = box[2] * scale
    var fy1 = -box[1] * scale
    if not (
        abs(fx0) <= Float32(1.0e6)
        and abs(fy0) <= Float32(1.0e6)
        and abs(fx1) <= Float32(1.0e6)
        and abs(fy1) <= Float32(1.0e6)
    ):
        return baked^
    var pad = spread + 1
    var gx0 = Int(floor(fx0)) - pad
    var gy0 = Int(floor(fy0)) - pad
    var gx1 = Int(ceil(fx1)) + pad
    var gy1 = Int(ceil(fy1)) + pad
    var bw = gx1 - gx0
    var bh = gy1 - gy0
    if (
        bw <= 0
        or bh <= 0
        or bw > MAX_BAKED_GLYPH_DIM
        or bh > MAX_BAKED_GLYPH_DIM
        or bw * bh * channels > MAX_BAKED_GLYPH_BYTES
    ):
        return baked^
    var contours = extract_edge_contours(outline, scale)
    if len(contours) == 0:
        return baked^
    color_edges(contours)
    var segments = flatten(outline, scale)
    baked.w = bw
    baked.h = bh
    baked.xoff = Float32(gx0)
    baked.yoff = Float32(gy0)
    baked.pixels = List[UInt8](length=baked.w * baked.h * channels, fill=0)
    var to_units = Float32(127.0) / Float32(max(1, spread))

    for j in range(baked.h):
        var py = Float32(gy0 + j) + Float32(0.5)
        for i in range(baked.w):
            var px = Float32(gx0 + i) + Float32(0.5)
            var best_rd: Float32 = -1.0e30
            var best_ro: Float32 = -1.0
            var best_rt: Float32 = 0.5
            var best_re = EdgeSegment.line(0, 0, 0, 0)
            var best_gd: Float32 = -1.0e30
            var best_go: Float32 = -1.0
            var best_gt: Float32 = 0.5
            var best_ge = EdgeSegment.line(0, 0, 0, 0)
            var best_bd: Float32 = -1.0e30
            var best_bo: Float32 = -1.0
            var best_bt: Float32 = 0.5
            var best_be = EdgeSegment.line(0, 0, 0, 0)

            for c_idx in range(len(contours)):
                for e_idx in range(len(contours[c_idx].edges)):
                    var edge = contours[c_idx].edges[e_idx]
                    var res = edge.signed_distance_param(px, py)
                    var d = res[0]
                    var ortho = res[1]
                    var t_raw = res[2]
                    var ad = abs(d)
                    if (edge.color & EDGE_R) != 0:
                        var cur_ad = abs(best_rd)
                        if ad < cur_ad - Float32(1.0e-5) or (abs(ad - cur_ad) <= Float32(1.0e-5) and ortho > best_ro):
                            best_rd = d; best_ro = ortho; best_rt = t_raw; best_re = edge
                    if (edge.color & EDGE_G) != 0:
                        var cur_ad = abs(best_gd)
                        if ad < cur_ad - Float32(1.0e-5) or (abs(ad - cur_ad) <= Float32(1.0e-5) and ortho > best_go):
                            best_gd = d; best_go = ortho; best_gt = t_raw; best_ge = edge
                    if (edge.color & EDGE_B) != 0:
                        var cur_ad = abs(best_bd)
                        if ad < cur_ad - Float32(1.0e-5) or (abs(ad - cur_ad) <= Float32(1.0e-5) and ortho > best_bo):
                            best_bd = d; best_bo = ortho; best_bt = t_raw; best_be = edge

            var pr = best_re.to_pseudo_distance(px, py, best_rd, best_rt) if best_ro >= Float32(0.0) else Float32(-1.0e6)
            var pg = best_ge.to_pseudo_distance(px, py, best_gd, best_gt) if best_go >= Float32(0.0) else Float32(-1.0e6)
            var pb = best_be.to_pseudo_distance(px, py, best_bd, best_bt) if best_bo >= Float32(0.0) else Float32(-1.0e6)

            var out_idx = (j * baked.w + i) * channels
            baked.pixels[out_idx] = UInt8(max(0, min(255, Int(floor(Float32(128.0) + pr * to_units + Float32(0.5))))))
            baked.pixels[out_idx + 1] = UInt8(max(0, min(255, Int(floor(Float32(128.0) + pg * to_units + Float32(0.5))))))
            baked.pixels[out_idx + 2] = UInt8(max(0, min(255, Int(floor(Float32(128.0) + pb * to_units + Float32(0.5))))))
            if channels == 4:
                var td = signed_distance(segments, px, py) if segments.count() > 0 else msdf_median(pr, pg, pb)
                baked.pixels[out_idx + 3] = UInt8(max(0, min(255, Int(floor(Float32(128.0) + td * to_units + Float32(0.5))))))

    _msdf_error_correct_impl(baked.pixels, baked.w, baked.h, channels, Float32(spread), segments, Float32(gx0), Float32(gy0), True)
    return baked^


def bake_glyph_id(
    font: TrueTypeFont,
    glyph_id: Int,
    bake_px: Int,
    spread: Int,
    code: Int = 0,
    metric_scale: Float32 = 1.0,
    channels: Int = 1,
) -> BakedGlyph:
    """Bake one glyph by glyph ID (supporting post-GSUB glyphs without cmap entries
    and fallback font metric scaling) in 1-channel SDF or 3/4-channel MSDF/MTSDF."""
    if channels >= 3:
        return _bake_glyph_msdf_internal(font, glyph_id, bake_px, spread, code, metric_scale, 4 if channels >= 4 else 3)
    var baked = BakedGlyph()
    baked.code = code
    baked.glyph = glyph_id
    baked.channels = 1
    if bake_px <= 0 or spread < 0 or metric_scale <= Float32(0.0):
        return baked^
    var scale = (Float32(bake_px) / Float32(max(1, font.units_per_em))) * metric_scale
    if code != 0 and invisible(code):
        return baked^
    var outline = font.outline(baked.glyph)
    baked.advance = outline.advance * scale
    if outline.point_count() == 0 or len(outline.ends) == 0:
        return baked^
    var box = outline.bounds()
    var fx0 = box[0] * scale
    var fy0 = -box[3] * scale
    var fx1 = box[2] * scale
    var fy1 = -box[1] * scale
    if not (
        abs(fx0) <= Float32(1.0e6)
        and abs(fy0) <= Float32(1.0e6)
        and abs(fx1) <= Float32(1.0e6)
        and abs(fy1) <= Float32(1.0e6)
    ):
        return baked^
    var pad = spread + 1
    var gx0 = Int(floor(fx0)) - pad
    var gy0 = Int(floor(fy0)) - pad
    var gx1 = Int(ceil(fx1)) + pad
    var gy1 = Int(ceil(fy1)) + pad
    var bw = gx1 - gx0
    var bh = gy1 - gy0
    if (
        bw <= 0
        or bh <= 0
        or bw > MAX_BAKED_GLYPH_DIM
        or bh > MAX_BAKED_GLYPH_DIM
        or bw * bh > MAX_BAKED_GLYPH_BYTES
    ):
        return baked^
    var segments = flatten(outline, scale)
    if segments.count() == 0:
        return baked^
    baked.w = bw
    baked.h = bh
    baked.xoff = Float32(gx0)
    baked.yoff = Float32(gy0)
    baked.pixels = List[UInt8](length=baked.w * baked.h, fill=0)
    var to_units = Float32(127) / Float32(max(1, spread))
    for j in range(baked.h):
        var py = Float32(gy0 + j) + 0.5
        for i in range(baked.w):
            var px = Float32(gx0 + i) + 0.5
            var d = signed_distance(segments, px, py)
            var value = 128.0 + d * to_units
            baked.pixels[j * baked.w + i] = UInt8(max(0, min(255, Int(floor(value + 0.5)))))
    return baked^


def bake_glyph(font: TrueTypeFont, code: Int, bake_px: Int, spread: Int) -> BakedGlyph:
    """One glyph's distance cell, metrics in bake px. A glyph with no ink
    (a space) has an empty cell and only its advance."""
    var gid = font.glyph_id(code)
    if gid == 0 and code == 0xA0:
        gid = font.glyph_id(32)
    return bake_glyph_id(font, gid, bake_px, spread, code, 1.0, 1)


def bake_glyph_msdf(font: TrueTypeFont, code: Int, bake_px: Int, spread: Int) -> BakedGlyph:
    """Bake one glyph as a 3-channel RGB Multi-channel Signed Distance Field."""
    var gid = font.glyph_id(code)
    if gid == 0 and code == 0xA0:
        gid = font.glyph_id(32)
    return bake_glyph_id(font, gid, bake_px, spread, code, 1.0, 3)


def bake_glyph_mtsdf(font: TrueTypeFont, code: Int, bake_px: Int, spread: Int) -> BakedGlyph:
    """Bake one glyph as a 4-channel RGBA Multi-channel + True Signed Distance Field."""
    var gid = font.glyph_id(code)
    if gid == 0 and code == 0xA0:
        gid = font.glyph_id(32)
    return bake_glyph_id(font, gid, bake_px, spread, code, 1.0, 4)


def invisible(code: Int) -> Bool:
    """Default-ignorable characters the UI may meet: soft hyphen, zero
    width space and joiners, word joiner, byte order mark."""
    return code == 0xAD or (code >= 0x200B and code <= 0x200D) or code == 0x2060 or code == 0xFEFF


def default_charset() -> List[Int]:
    """The codepoints baked ahead: ASCII, Latin-1 (Spanish and the western
    European languages) and the punctuation, arrows and symbols the UI uses.
    Anything else is baked on first use by the dynamic cache."""
    var codes = List[Int]()
    for code in range(32, 127):
        codes.append(code)
    for code in range(0xA0, 0x100):
        codes.append(code)
    for code in [0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2026,
                 0x2190, 0x2191, 0x2192, 0x2193, 0x25B6, 0x25C0, 0x2605]:
        codes.append(code)
    return codes^


struct ShelfPacker(Copyable, Movable):
    """Rows of cells left to right, a new row when one is full; one texel
    of gap. Used for the baked atlas and for the dynamic cache's page."""
    var width: Int
    var height: Int
    var top: Int       # y where packing starts (the dynamic page begins below the baked atlas)
    var x: Int
    var y: Int
    var shelf: Int

    def __init__(out self, width: Int, height: Int, top: Int = 0):
        self.width = width
        self.height = height
        self.top = top
        self.x = 0
        self.y = top
        self.shelf = 0

    def reset(mut self):
        self.x = 0
        self.y = self.top
        self.shelf = 0

    def place(mut self, w: Int, h: Int) -> Tuple[Int, Int]:
        """Top-left of a w x h cell, or (-1, -1) when the area is full."""
        if w > self.width:
            return (-1, -1)
        if self.x + w > self.width:
            self.x = 0
            self.y += self.shelf + 1
            self.shelf = 0
        if self.y + h > self.height:
            return (-1, -1)
        var spot = (self.x, self.y)
        self.x += w + 1
        self.shelf = max(self.shelf, h)
        return spot

    def used_height(self) -> Int:
        return self.y + self.shelf + 1


struct BakedFont(Movable):
    var atlas_width: Int
    var atlas_height: Int
    var bake_px: Int
    var spread: Int
    var channels: Int
    var ascent: Float32
    var descent: Float32
    var glyphs: List[BakedGlyph]
    var kern_left: List[Int]
    var kern_right: List[Int]
    var kern_amount: List[Float32]
    var atlas: List[UInt8]

    def __init__(out self):
        self.atlas_width = 0
        self.atlas_height = 0
        self.bake_px = 0
        self.spread = 0
        self.channels = 1
        self.ascent = 0
        self.descent = 0
        self.glyphs = List[BakedGlyph]()
        self.kern_left = List[Int]()
        self.kern_right = List[Int]()
        self.kern_amount = List[Float32]()
        self.atlas = List[UInt8]()


def bake_font(font: TrueTypeFont, codes: List[Int], bake_px: Int, spread: Int, atlas_width: Int = 1024, channels: Int = 1) raises -> BakedFont:
    """Bake every listed codepoint the font has into one atlas (cells
    shelf-packed tallest first, height rounded up to a power of two), with
    the kerning pairs between them of a quarter pixel or more."""
    var ch = 4 if channels >= 4 else (3 if channels >= 3 else 1)
    var baked = BakedFont()
    baked.bake_px = bake_px
    baked.spread = spread
    baked.channels = ch
    var scale = Float32(bake_px) / Float32(max(1, font.units_per_em))
    baked.ascent = ceil(Float32(font.ascender) * scale)
    baked.descent = ceil(Float32(-font.descender) * scale)
    for code in codes:
        if ch == 4:
            baked.glyphs.append(bake_glyph_mtsdf(font, code, bake_px, spread))
        elif ch == 3:
            baked.glyphs.append(bake_glyph_msdf(font, code, bake_px, spread))
        else:
            baked.glyphs.append(bake_glyph(font, code, bake_px, spread))
        if baked.glyphs[len(baked.glyphs) - 1].glyph == 0 and not invisible(code) and code != 32:
            _ = baked.glyphs.pop()
    # Tallest first (stable for equal heights, so the layout is reproducible).
    var order = List[Int]()
    for index in range(len(baked.glyphs)):
        order.append(index)
    for i in range(1, len(order)):
        var v = order[i]
        var j = i - 1
        while j >= 0 and baked.glyphs[order[j]].h < baked.glyphs[v].h:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = v
    var packer = ShelfPacker(atlas_width, 8192)
    for index in order:
        if baked.glyphs[index].w == 0:
            continue
        var spot = packer.place(baked.glyphs[index].w, baked.glyphs[index].h)
        if spot[0] < 0:
            raise Error("atlas too small")
        baked.glyphs[index].x = spot[0]
        baked.glyphs[index].y = spot[1]
    var height = 64
    while height < packer.used_height():
        height *= 2
    baked.atlas_width = atlas_width
    baked.atlas_height = height
    baked.atlas = List[UInt8](length=atlas_width * height * ch, fill=0)
    for glyph in baked.glyphs:
        var g_ch = max(1, glyph.channels)
        for j in range(glyph.h):
            for i in range(glyph.w):
                var dst_base = ((glyph.y + j) * atlas_width + glyph.x + i) * ch
                var src_base = (j * glyph.w + i) * g_ch
                for c in range(ch):
                    baked.atlas[dst_base + c] = glyph.pixels[src_base + min(c, g_ch - 1)]
    # Kerning between the baked characters below U+2000, like the old baker.
    var ids = List[Int]()
    var kern_codes = List[Int]()
    for glyph in baked.glyphs:
        if glyph.code > 32 and glyph.code < 0x2000 and not invisible(glyph.code):
            kern_codes.append(glyph.code)
            ids.append(glyph.glyph)
    for a in range(len(ids)):
        for b in range(len(ids)):
            var amount = font.kerning(ids[a], ids[b]) * scale
            if abs(amount) >= 0.25:
                baked.kern_left.append(kern_codes[a])
                baked.kern_right.append(kern_codes[b])
                baked.kern_amount.append(amount)
    return baked^


def put_i32(mut out: List[UInt8], value: Int):
    var v = value & 0xFFFFFFFF
    out.append(UInt8(v & 255))
    out.append(UInt8((v >> 8) & 255))
    out.append(UInt8((v >> 16) & 255))
    out.append(UInt8((v >> 24) & 255))


def fixed64(value: Float32) -> Int:
    return Int(floor(value * 64.0 + 0.5))


def dsf_bytes(baked: BakedFont) -> List[UInt8]:
    """The `.dsf` layout `render/font.mojo` loads (little-endian):
    v1 (`channels == 1`): 11 int32 header, then glyphs, kerning, 1-byte atlas.
    v2 (`channels > 1`): 12 int32 header (with `channels` at byte 44), then
    glyphs, kerning, `channels`-byte interleaved atlas."""
    var ch = max(1, baked.channels)
    var version = 2 if ch > 1 else 1
    var out = List[UInt8]()
    put_i32(out, DSF_MAGIC)
    put_i32(out, version)
    put_i32(out, baked.atlas_width)
    put_i32(out, baked.atlas_height)
    put_i32(out, baked.bake_px)
    put_i32(out, baked.spread)
    put_i32(out, fixed64(baked.ascent))
    put_i32(out, fixed64(baked.descent))
    put_i32(out, fixed64(baked.ascent + baked.descent))
    put_i32(out, len(baked.glyphs))
    put_i32(out, len(baked.kern_left))
    if version == 2:
        put_i32(out, ch)
    for glyph in baked.glyphs:
        put_i32(out, glyph.code)
        put_i32(out, glyph.x)
        put_i32(out, glyph.y)
        put_i32(out, glyph.w)
        put_i32(out, glyph.h)
        put_i32(out, fixed64(glyph.xoff))
        put_i32(out, fixed64(glyph.yoff))
        put_i32(out, fixed64(glyph.advance))
    for index in range(len(baked.kern_left)):
        put_i32(out, baked.kern_left[index])
        put_i32(out, baked.kern_right[index])
        put_i32(out, fixed64(baked.kern_amount[index]))
    out.extend(baked.atlas.copy())
    return out^
