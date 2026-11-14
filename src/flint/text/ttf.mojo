"""Our own TrueType / OpenType font reader (package 3.12).

Reads the tables a UI needs straight from the file bytes, with no library:

- the table directory, `head`, `hhea`, `maxp`, `OS/2` (cap and x height),
- `cmap` formats 4 and 12 (Unicode BMP and full range),
- `hmtx`, `loca` and `glyf`: simple glyphs (on and off curve points,
  repeated flags, short and long coordinates) and composite glyphs
  (offsets, scales, 2x2 transforms, USE_MY_METRICS), resolved recursively,
- kerning from `GPOS` pair adjustment (lookup type 2, formats 1 and 2,
  through type 9 extensions, every lookup of the `kern` feature) and from
  the old `kern` table (format 0),
- variable fonts: `fvar` axes, `avar` maps, `gvar` glyph deltas (shared and
  private point numbers, packed deltas, intermediate regions, the IUP
  interpolation of untouched points, phantom points for advances) and the
  `GDEF` item variation store for GPOS device deltas, so a weight such as
  Atkinson Hyperlegible Next SemiBold (wght 600) is an instance we compute.

CFF outlines (`OTTO` fonts) are refused with a message: both shipped faces
are glyf fonts. Everything is in font units; `text/sdf_bake.mojo` scales.

References: the OpenType specification 1.9 (Microsoft), chapters cmap,
glyf, gvar, avar, fvar, GPOS, GDEF and "OpenType font variations overview".
"""
from std.collections import Dict
from flint.platform.os import read_bytes


comptime TAG_OTTO: Int = 0x4F54544F
comptime MAX_COMPONENT_DEPTH: Int = 8


def tag_value(tag: String) -> Int:
    var bytes = tag.as_bytes()
    var value = 0
    for index in range(4):
        value = (value << 8) | (Int(bytes[index]) if index < len(bytes) else 32)
    return value


struct GlyphOutline(Copyable, Movable):
    """Points of one glyph in font units, y up. `ends` holds the index of
    the last point of each contour. `advance` and `lsb` are the
    horizontal metrics after variations (from the phantom points)."""
    var xs: List[Float32]
    var ys: List[Float32]
    var on: List[Bool]
    var ends: List[Int]
    var advance: Float32
    var lsb: Float32

    def __init__(out self):
        self.xs = List[Float32]()
        self.ys = List[Float32]()
        self.on = List[Bool]()
        self.ends = List[Int]()
        self.advance = 0
        self.lsb = 0

    def point_count(self) -> Int:
        return len(self.xs)

    def bounds(self) -> Tuple[Float32, Float32, Float32, Float32]:
        """(x_min, y_min, x_max, y_max) over all points (the control
        points bound a quadratic curve, so this holds the ink)."""
        if len(self.xs) == 0:
            return (Float32(0), Float32(0), Float32(0), Float32(0))
        var x0 = self.xs[0]
        var y0 = self.ys[0]
        var x1 = x0
        var y1 = y0
        for index in range(len(self.xs)):
            x0 = min(x0, self.xs[index])
            x1 = max(x1, self.xs[index])
            y0 = min(y0, self.ys[index])
            y1 = max(y1, self.ys[index])
        return (x0, y0, x1, y1)


struct VariationAxis(Copyable, ImplicitlyCopyable, Movable):
    var tag: Int
    var minimum: Float32
    var default: Float32
    var maximum: Float32

    def __init__(out self, tag: Int, minimum: Float32, default: Float32, maximum: Float32):
        self.tag = tag
        self.minimum = minimum
        self.default = default
        self.maximum = maximum


struct TrueTypeFont(Copyable, Movable):
    var data: List[UInt8]
    var valid: Bool
    var error: String
    var tables: Dict[Int, Int]       # tag -> offset
    var lengths: Dict[Int, Int]
    var units_per_em: Int
    var index_to_loc: Int
    var glyph_count: Int
    var ascender: Int
    var descender: Int
    var line_gap: Int
    var cap_height: Int
    var x_height: Int
    var metric_count: Int
    var cmap_offset: Int
    var cmap_format: Int
    var axes: List[VariationAxis]
    var coords: List[Float32]        # normalized, after avar, rounded to F2Dot14
    var kern_lookups: List[Int]      # GPOS lookup offsets (absolute) of the kern feature

    def __init__(out self):
        self.data = List[UInt8]()
        self.valid = False
        self.error = String()
        self.tables = Dict[Int, Int]()
        self.lengths = Dict[Int, Int]()
        self.units_per_em = 1000
        self.index_to_loc = 0
        self.glyph_count = 0
        self.ascender = 0
        self.descender = 0
        self.line_gap = 0
        self.cap_height = 0
        self.x_height = 0
        self.metric_count = 0
        self.cmap_offset = -1
        self.cmap_format = 0
        self.axes = List[VariationAxis]()
        self.coords = List[Float32]()
        self.kern_lookups = List[Int]()

    # ---- bytes ----

    def u8(self, offset: Int) -> Int:
        if offset < 0 or offset >= len(self.data):
            return 0
        return Int(self.data[offset])

    def u16(self, offset: Int) -> Int:
        return (self.u8(offset) << 8) | self.u8(offset + 1)

    def i16(self, offset: Int) -> Int:
        var value = self.u16(offset)
        return value - 65536 if value >= 32768 else value

    def u32(self, offset: Int) -> Int:
        return (self.u16(offset) << 16) | self.u16(offset + 2)

    def i32(self, offset: Int) -> Int:
        var value = self.u32(offset)
        return value - 0x100000000 if value >= 0x80000000 else value

    def f2dot14(self, offset: Int) -> Float32:
        return Float32(self.i16(offset)) / 16384.0

    def fixed(self, offset: Int) -> Float32:
        return Float32(Float64(self.i32(offset)) / 65536.0)

    def table(self, tag: String) -> Int:
        var found = self.tables.get(tag_value(tag))
        if found:
            return found.value()
        return -1

    # ---- loading ----

    def load(mut self, path: String) -> Bool:
        return self.load_bytes(read_bytes(path))

    def load_bytes(mut self, var data: List[UInt8]) -> Bool:
        self.data = data^
        self.valid = False
        self.tables = Dict[Int, Int]()
        self.lengths = Dict[Int, Int]()
        if len(self.data) < 12:
            self.error = "not a font file (too short)"
            return False
        var version = self.u32(0)
        if version == TAG_OTTO:
            self.error = "CFF outlines (OTTO) are not supported; use a TrueType-outline font"
            return False
        if version != 0x00010000 and version != tag_value("true"):
            self.error = "not a TrueType font (version " + String(version) + ")"
            return False
        var count = self.u16(4)
        if 12 + count * 16 > len(self.data):
            self.error = "table directory runs past the end of the file"
            return False
        for index in range(count):
            var record = 12 + index * 16
            var offset = self.u32(record + 8)
            var length = self.u32(record + 12)
            if offset + length > len(self.data):
                self.error = "a table runs past the end of the file"
                return False
            self.tables[self.u32(record)] = offset
            self.lengths[self.u32(record)] = length
        for name in ["head", "hhea", "maxp", "cmap", "hmtx", "loca", "glyf"]:
            if self.table(name) < 0:
                self.error = "missing table " + name
                return False
        var head = self.table("head")
        self.units_per_em = self.u16(head + 18)
        self.index_to_loc = self.i16(head + 50)
        if self.units_per_em < 16 or self.units_per_em > 16384:
            self.error = "bad unitsPerEm"
            return False
        self.glyph_count = self.u16(self.table("maxp") + 4)
        var hhea = self.table("hhea")
        self.ascender = self.i16(hhea + 4)
        self.descender = self.i16(hhea + 6)
        self.line_gap = self.i16(hhea + 8)
        self.metric_count = self.u16(hhea + 34)
        if self.metric_count < 1:
            self.error = "no horizontal metrics"
            return False
        var os2 = self.table("OS/2")
        self.cap_height = Int(Float32(self.units_per_em) * 0.7)
        self.x_height = Int(Float32(self.units_per_em) * 0.5)
        if os2 >= 0 and self.u16(os2) >= 2:
            self.x_height = self.i16(os2 + 86)
            self.cap_height = self.i16(os2 + 88)
        if not self.pick_cmap():
            self.error = "no Unicode cmap (format 4 or 12)"
            return False
        self.read_axes()
        self.find_kern_lookups()
        self.valid = True
        return True

    def pick_cmap(mut self) -> Bool:
        var cmap = self.table("cmap")
        var count = self.u16(cmap + 2)
        var best = -1
        var best_score = 0
        for index in range(count):
            var record = cmap + 4 + index * 8
            var platform = self.u16(record)
            var encoding = self.u16(record + 2)
            var sub = cmap + self.u32(record + 4)
            var format = self.u16(sub)
            var score = 0
            if format == 12 and (platform == 3 and encoding == 10 or platform == 0):
                score = 3
            elif format == 4 and (platform == 3 and encoding == 1 or platform == 0):
                score = 2
            if score > best_score:
                best_score = score
                best = sub
        if best < 0:
            return False
        self.cmap_offset = best
        self.cmap_format = self.u16(best)
        return True

    def glyph_id(self, code: Int) -> Int:
        """The glyph for a codepoint, 0 (.notdef) when the font has none."""
        if not self.valid or code < 0:
            return 0
        var sub = self.cmap_offset
        if self.cmap_format == 12:
            var groups = self.u32(sub + 12)
            var low = 0
            var high = groups - 1
            while low <= high:
                var middle = (low + high) // 2
                var group = sub + 16 + middle * 12
                var start = self.u32(group)
                var end = self.u32(group + 4)
                if code < start:
                    high = middle - 1
                elif code > end:
                    low = middle + 1
                else:
                    return self.u32(group + 8) + (code - start)
            return 0
        if code > 0xFFFF:
            return 0
        var segments = self.u16(sub + 6) // 2
        var ends = sub + 14
        var starts = ends + segments * 2 + 2
        var deltas = starts + segments * 2
        var ranges = deltas + segments * 2
        var low = 0
        var high = segments - 1
        while low <= high:
            var middle = (low + high) // 2
            var end = self.u16(ends + middle * 2)
            var start = self.u16(starts + middle * 2)
            if code > end:
                low = middle + 1
            elif code < start:
                high = middle - 1
            else:
                var delta = self.i16(deltas + middle * 2)
                var range_offset = self.u16(ranges + middle * 2)
                if range_offset == 0:
                    return (code + delta) & 0xFFFF
                var at = ranges + middle * 2 + range_offset + (code - start) * 2
                var glyph = self.u16(at)
                if glyph == 0:
                    return 0
                return (glyph + delta) & 0xFFFF
        return 0

    def has_glyph(self, code: Int) -> Bool:
        return self.glyph_id(code) != 0

    # ---- metrics ----

    def default_advance(self, glyph: Int) -> Int:
        var hmtx = self.table("hmtx")
        var index = min(glyph, self.metric_count - 1)
        return self.u16(hmtx + index * 4)

    def default_lsb(self, glyph: Int) -> Int:
        var hmtx = self.table("hmtx")
        if glyph < self.metric_count:
            return self.i16(hmtx + glyph * 4 + 2)
        return self.i16(hmtx + self.metric_count * 4 + (glyph - self.metric_count) * 2)

    def advance(self, glyph: Int) -> Float32:
        """Advance width in font units at the current instance."""
        if len(self.coords) == 0:
            return Float32(self.default_advance(glyph))
        return self.outline(glyph).advance

    # ---- variations ----

    def read_axes(mut self):
        self.axes = List[VariationAxis]()
        self.coords = List[Float32]()
        var fvar = self.table("fvar")
        if fvar < 0:
            return
        var axes_offset = fvar + self.u16(fvar + 4)
        var count = self.u16(fvar + 8)
        var size = self.u16(fvar + 10)
        for index in range(count):
            var at = axes_offset + index * size
            self.axes.append(VariationAxis(self.u32(at), self.fixed(at + 4), self.fixed(at + 8), self.fixed(at + 12)))
        for _ in range(count):
            self.coords.append(0)

    def is_variable(self) -> Bool:
        return len(self.axes) > 0 and self.table("gvar") >= 0

    def set_variation(mut self, tag: String, value: Float32) -> Bool:
        """Set one axis in user units (wght 600). Normalizes, applies avar
        and rounds to F2Dot14 as the specification orders."""
        var wanted = tag_value(tag)
        for index in range(len(self.axes)):
            var axis = self.axes[index]
            if axis.tag != wanted:
                continue
            var v = max(axis.minimum, min(axis.maximum, value))
            var n: Float32 = 0
            if v < axis.default:
                n = -(axis.default - v) / max(Float32(1e-6), axis.default - axis.minimum)
            elif v > axis.default:
                n = (v - axis.default) / max(Float32(1e-6), axis.maximum - axis.default)
            n = self.avar_map(index, n)
            self.coords[index] = Float32(Int(round_half_up(n * 16384.0))) / 16384.0
            return True
        return False

    def avar_map(self, axis: Int, value: Float32) -> Float32:
        var avar = self.table("avar")
        if avar < 0:
            return value
        var count = self.u16(avar + 6)
        var at = avar + 8
        for index in range(count):
            var pairs = self.u16(at)
            if index == axis and pairs > 0:
                var previous_from = self.f2dot14(at + 2)
                var previous_to = self.f2dot14(at + 4)
                if value <= previous_from:
                    return previous_to
                for pair in range(1, pairs):
                    var from_value = self.f2dot14(at + 2 + pair * 4)
                    var to_value = self.f2dot14(at + 4 + pair * 4)
                    if value <= from_value:
                        if from_value == previous_from:
                            return to_value
                        return previous_to + (to_value - previous_to) * (value - previous_from) / (from_value - previous_from)
                    previous_from = from_value
                    previous_to = to_value
                return previous_to
            at += 2 + pairs * 4
        return value

    def region_scalar(self, peaks: List[Float32], starts: List[Float32], ends: List[Float32], intermediate: Bool) -> Float32:
        """How much of one tuple's deltas apply at the current instance."""
        var scalar: Float32 = 1
        for axis in range(len(self.coords)):
            var peak = peaks[axis]
            if peak == 0:
                continue
            var coord = self.coords[axis]
            if coord == peak:
                continue
            if intermediate:
                var start = starts[axis]
                var end = ends[axis]
                if start > peak or peak > end or (start < 0 and end > 0):
                    continue
                if coord < start or coord > end:
                    return 0
                if coord < peak:
                    scalar *= (coord - start) / (peak - start)
                else:
                    scalar *= (end - coord) / (end - peak)
            else:
                if coord == 0 or coord < min(Float32(0), peak) or coord > max(Float32(0), peak):
                    return 0
                scalar *= coord / peak
        return scalar

    # ---- outlines ----

    def glyph_range(self, glyph: Int) -> Tuple[Int, Int]:
        var loca = self.table("loca")
        var glyf = self.table("glyf")
        if glyph < 0 or glyph >= self.glyph_count:
            return (0, 0)
        if self.index_to_loc == 0:
            return (glyf + self.u16(loca + glyph * 2) * 2, glyf + self.u16(loca + glyph * 2 + 2) * 2)
        return (glyf + self.u32(loca + glyph * 4), glyf + self.u32(loca + glyph * 4 + 4))

    def outline(self, glyph: Int) -> GlyphOutline:
        return self.outline_at(glyph, 0)

    def outline_at(self, glyph: Int, depth: Int) -> GlyphOutline:
        var result = GlyphOutline()
        var span = self.glyph_range(glyph)
        var start = span[0]
        var end = span[1]
        var x_min = 0
        if end - start >= 10:
            x_min = self.i16(start + 2)
        # Phantom points (horizontal pair; the vertical pair is not used).
        var lsb = self.default_lsb(glyph)
        var advance = self.default_advance(glyph)
        var phantom_x0 = Float32(x_min - lsb)
        var phantom_x1 = phantom_x0 + Float32(advance)
        if end - start < 10:
            # An empty glyph (a space): only the phantom points vary.
            var px: List[Float32] = [phantom_x0, phantom_x1, Float32(0), Float32(0)]
            var py: List[Float32] = [Float32(0), Float32(0), Float32(0), Float32(0)]
            var none = List[Int]()
            self.apply_gvar(glyph, px, py, none, False)
            result.advance = px[1] - px[0]
            result.lsb = -px[0]
            return result^
        var contours = self.i16(start)
        if contours >= 0:
            self.read_simple(start, contours, result)
            var n = len(result.xs)
            result.xs.append(phantom_x0)
            result.ys.append(0)
            result.xs.append(phantom_x1)
            result.ys.append(0)
            result.xs.append(0)
            result.ys.append(0)
            result.xs.append(0)
            result.ys.append(0)
            self.apply_gvar(glyph, result.xs, result.ys, result.ends, True)
            result.advance = result.xs[n + 1] - result.xs[n]
            var shift = result.xs[n]
            for index in range(n):
                result.xs[index] -= shift
            result.lsb = -shift
            result.xs.resize(n, 0)
            result.ys.resize(n, 0)
            return result^
        if depth >= MAX_COMPONENT_DEPTH:
            return result^
        # Composite: each component's offset is a point for gvar, then the phantoms.
        var at = start + 10
        var flags_list = List[Int]()
        var components = List[Int]()
        var dx = List[Float32]()
        var dy = List[Float32]()
        var matrices = List[Float32]()
        var use_metrics = -1
        while True:
            var flags = self.u16(at)
            var component = self.u16(at + 2)
            at += 4
            var arg1: Int
            var arg2: Int
            if flags & 0x0001 != 0:
                arg1 = self.i16(at)
                arg2 = self.i16(at + 2)
                at += 4
            else:
                arg1 = self.u8(at)
                arg2 = self.u8(at + 1)
                if arg1 >= 128:
                    arg1 -= 256
                if arg2 >= 128:
                    arg2 -= 256
                at += 2
            var a: Float32 = 1
            var b: Float32 = 0
            var c: Float32 = 0
            var d: Float32 = 1
            if flags & 0x0008 != 0:
                a = self.f2dot14(at)
                d = a
                at += 2
            elif flags & 0x0040 != 0:
                a = self.f2dot14(at)
                d = self.f2dot14(at + 2)
                at += 4
            elif flags & 0x0080 != 0:
                a = self.f2dot14(at)
                b = self.f2dot14(at + 2)
                c = self.f2dot14(at + 4)
                d = self.f2dot14(at + 6)
                at += 8
            if flags & 0x0002 == 0:
                # Point matching: rare in UI faces; placed at the origin.
                arg1 = 0
                arg2 = 0
            if flags & 0x0200 != 0:
                use_metrics = len(components)
            flags_list.append(flags)
            components.append(component)
            dx.append(Float32(arg1))
            dy.append(Float32(arg2))
            matrices.append(a)
            matrices.append(b)
            matrices.append(c)
            matrices.append(d)
            if flags & 0x0020 == 0:
                break
        var count = len(components)
        dx.append(phantom_x0)
        dy.append(0)
        dx.append(phantom_x1)
        dy.append(0)
        dx.append(0)
        dy.append(0)
        dx.append(0)
        dy.append(0)
        var no_contours = List[Int]()
        self.apply_gvar(glyph, dx, dy, no_contours, False)
        var shift = dx[count]
        result.advance = dx[count + 1] - dx[count]
        result.lsb = -shift
        for index in range(count):
            var part = self.outline_at(components[index], depth + 1)
            if index == use_metrics:
                result.advance = part.advance
            var a = matrices[index * 4]
            var b = matrices[index * 4 + 1]
            var c = matrices[index * 4 + 2]
            var d = matrices[index * 4 + 3]
            var base = len(result.xs)
            for p in range(len(part.xs)):
                var x = part.xs[p]
                var y = part.ys[p]
                result.xs.append(a * x + c * y + dx[index] - shift)
                result.ys.append(b * x + d * y + dy[index])
                result.on.append(part.on[p])
            for e in range(len(part.ends)):
                result.ends.append(part.ends[e] + base)
        return result^

    def read_simple(self, start: Int, contours: Int, mut result: GlyphOutline):
        var at = start + 10
        var last = -1
        for _ in range(contours):
            last = self.u16(at)
            result.ends.append(last)
            at += 2
        var count = last + 1
        var instructions = self.u16(at)
        at += 2 + instructions
        var flags = List[Int]()
        while len(flags) < count:
            var flag = self.u8(at)
            at += 1
            flags.append(flag)
            if flag & 0x08 != 0:
                var repeat = self.u8(at)
                at += 1
                for _ in range(repeat):
                    if len(flags) < count:
                        flags.append(flag)
        var x = 0
        for index in range(count):
            var flag = flags[index]
            if flag & 0x02 != 0:
                var step = self.u8(at)
                at += 1
                x += step if flag & 0x10 != 0 else -step
            elif flag & 0x10 == 0:
                x += self.i16(at)
                at += 2
            result.xs.append(Float32(x))
            result.on.append(flag & 0x01 != 0)
        var y = 0
        for index in range(count):
            var flag = flags[index]
            if flag & 0x04 != 0:
                var step = self.u8(at)
                at += 1
                y += step if flag & 0x20 != 0 else -step
            elif flag & 0x20 == 0:
                y += self.i16(at)
                at += 2
            result.ys.append(Float32(y))

    def packed_points(self, mut at: Int, total: Int) -> List[Int]:
        """Packed point numbers; an empty list means every point."""
        var points = List[Int]()
        var count = self.u8(at)
        at += 1
        if count == 0:
            return points^
        if count & 0x80 != 0:
            count = ((count & 0x7F) << 8) | self.u8(at)
            at += 1
        var point = 0
        while len(points) < count:
            var control = self.u8(at)
            at += 1
            var run = (control & 0x7F) + 1
            for _ in range(run):
                if len(points) >= count:
                    break
                if control & 0x80 != 0:
                    point += self.u16(at)
                    at += 2
                else:
                    point += self.u8(at)
                    at += 1
                points.append(point)
        _ = total
        return points^

    def packed_deltas(self, mut at: Int, count: Int) -> List[Float32]:
        var deltas = List[Float32]()
        while len(deltas) < count:
            var control = self.u8(at)
            at += 1
            var run = (control & 0x3F) + 1
            for _ in range(run):
                if len(deltas) >= count:
                    break
                if control & 0x80 != 0:
                    deltas.append(0)
                elif control & 0x40 != 0:
                    deltas.append(Float32(self.i16(at)))
                    at += 2
                else:
                    var b = self.u8(at)
                    deltas.append(Float32(b - 256 if b >= 128 else b))
                    at += 1
        return deltas^

    def apply_gvar(self, glyph: Int, mut xs: List[Float32], mut ys: List[Float32], ends: List[Int], interpolate: Bool):
        """Add this instance's deltas to the glyph's points (the last four
        are the phantom points). `interpolate`: untouched points of simple
        glyphs follow their touched neighbours (IUP)."""
        if len(self.coords) == 0:
            return
        var gvar = self.table("gvar")
        if gvar < 0:
            return
        var axis_count = self.u16(gvar + 4)
        var shared_count = self.u16(gvar + 6)
        var shared_offset = gvar + self.u32(gvar + 8)
        var glyphs = self.u16(gvar + 12)
        var long_offsets = self.u16(gvar + 14) & 1 != 0
        var array = gvar + self.u32(gvar + 16)
        if glyph >= glyphs or axis_count != len(self.coords):
            return
        var begin: Int
        var finish: Int
        if long_offsets:
            begin = self.u32(gvar + 20 + glyph * 4)
            finish = self.u32(gvar + 24 + glyph * 4)
        else:
            begin = self.u16(gvar + 20 + glyph * 2) * 2
            finish = self.u16(gvar + 22 + glyph * 2) * 2
        if finish <= begin:
            return
        var base = array + begin
        var tuple_word = self.u16(base)
        var tuple_count = tuple_word & 0x0FFF
        var data = base + self.u16(base + 2)
        var header = base + 4
        var total = len(xs)
        var shared_points = List[Int]()
        var have_shared = False
        if tuple_word & 0x8000 != 0:
            shared_points = self.packed_points(data, total)
            have_shared = True
        # Original positions for IUP: deltas interpolate against them.
        var original_x = xs.copy()
        var original_y = ys.copy()
        for _ in range(tuple_count):
            var size = self.u16(header)
            var index = self.u16(header + 2)
            header += 4
            var peaks = List[Float32]()
            if index & 0x8000 != 0:
                for axis in range(axis_count):
                    peaks.append(self.f2dot14(header + axis * 2))
                header += axis_count * 2
            else:
                var shared = index & 0x0FFF
                if shared >= shared_count:
                    return
                for axis in range(axis_count):
                    peaks.append(self.f2dot14(shared_offset + (shared * axis_count + axis) * 2))
            var starts = List[Float32]()
            var stops = List[Float32]()
            var intermediate = index & 0x4000 != 0
            if intermediate:
                for axis in range(axis_count):
                    starts.append(self.f2dot14(header + axis * 2))
                    stops.append(self.f2dot14(header + (axis_count + axis) * 2))
                header += axis_count * 4
            var next_data = data + size
            var scalar = self.region_scalar(peaks, starts, stops, intermediate)
            if scalar == 0:
                data = next_data
                continue
            var at = data
            var points: List[Int]
            if index & 0x2000 != 0:
                points = self.packed_points(at, total)
            elif have_shared:
                points = shared_points.copy()
            else:
                points = List[Int]()
            var all_points = len(points) == 0
            var count = total if all_points else len(points)
            var delta_x = self.packed_deltas(at, count)
            var delta_y = self.packed_deltas(at, count)
            data = next_data
            if all_points:
                for p in range(total):
                    xs[p] += delta_x[p] * scalar
                    ys[p] += delta_y[p] * scalar
                continue
            var touched = List[Bool](length=total, fill=False)
            var tx = List[Float32](length=total, fill=0)
            var ty = List[Float32](length=total, fill=0)
            for k in range(len(points)):
                var p = points[k]
                if p < total:
                    touched[p] = True
                    tx[p] += delta_x[k]
                    ty[p] += delta_y[k]
            if interpolate:
                interpolate_untouched(original_x, touched, tx, ends)
                interpolate_untouched(original_y, touched, ty, ends)
            for p in range(total):
                xs[p] += tx[p] * scalar
                ys[p] += ty[p] * scalar

    # ---- kerning ----

    def find_kern_lookups(mut self):
        self.kern_lookups = List[Int]()
        var gpos = self.table("GPOS")
        if gpos < 0:
            return
        var features = gpos + self.u16(gpos + 6)
        var lookups = gpos + self.u16(gpos + 8)
        var feature_count = self.u16(features)
        var wanted = List[Int]()
        for index in range(feature_count):
            var record = features + 2 + index * 6
            if self.u32(record) != tag_value("kern"):
                continue
            var feature = features + self.u16(record + 4)
            var count = self.u16(feature + 2)
            for k in range(count):
                var lookup = self.u16(feature + 4 + k * 2)
                var seen = False
                for w in wanted:
                    if w == lookup:
                        seen = True
                if not seen:
                    wanted.append(lookup)
        sort_ints(wanted)
        var lookup_count = self.u16(lookups)
        for lookup in wanted:
            if lookup < lookup_count:
                self.kern_lookups.append(lookups + self.u16(lookups + 2 + lookup * 2))

    def coverage_index(self, coverage: Int, glyph: Int) -> Int:
        var format = self.u16(coverage)
        if format == 1:
            var count = self.u16(coverage + 2)
            var low = 0
            var high = count - 1
            while low <= high:
                var middle = (low + high) // 2
                var value = self.u16(coverage + 4 + middle * 2)
                if value < glyph:
                    low = middle + 1
                elif value > glyph:
                    high = middle - 1
                else:
                    return middle
            return -1
        if format == 2:
            var count = self.u16(coverage + 2)
            var low = 0
            var high = count - 1
            while low <= high:
                var middle = (low + high) // 2
                var record = coverage + 4 + middle * 6
                var first = self.u16(record)
                var last = self.u16(record + 2)
                if glyph < first:
                    high = middle - 1
                elif glyph > last:
                    low = middle + 1
                else:
                    return self.u16(record + 4) + glyph - first
            return -1
        return -1

    def class_of(self, table: Int, glyph: Int) -> Int:
        var format = self.u16(table)
        if format == 1:
            var first = self.u16(table + 2)
            var count = self.u16(table + 4)
            if glyph >= first and glyph < first + count:
                return self.u16(table + 6 + (glyph - first) * 2)
            return 0
        if format == 2:
            var count = self.u16(table + 2)
            var low = 0
            var high = count - 1
            while low <= high:
                var middle = (low + high) // 2
                var record = table + 4 + middle * 6
                if glyph < self.u16(record):
                    high = middle - 1
                elif glyph > self.u16(record + 2):
                    low = middle + 1
                else:
                    return self.u16(record + 4)
            return 0
        return 0

    def value_x_advance(self, record: Int, format: Int, subtable: Int) -> Float32:
        """XAdvance of a ValueRecord plus its variation delta, if any."""
        if format & 0x0004 == 0:
            return 0
        var at = record
        if format & 0x0001 != 0:
            at += 2
        if format & 0x0002 != 0:
            at += 2
        var value = Float32(self.i16(at))
        at += 2
        if format & 0x0008 != 0:
            at += 2
        if format & 0x0010 != 0:
            at += 2
        if format & 0x0020 != 0:
            at += 2
        if format & 0x0040 != 0:
            var device = self.u16(at)
            if device != 0 and len(self.coords) > 0:
                var table = subtable + device
                if self.u16(table + 4) == 0x8000:
                    value += self.item_delta(self.u16(table), self.u16(table + 2))
        return value

    def item_delta(self, outer: Int, inner: Int) -> Float32:
        """A delta from GDEF's item variation store at this instance."""
        var gdef = self.table("GDEF")
        if gdef < 0 or self.u16(gdef + 2) < 3:
            return 0
        var store_offset = self.u32(gdef + 14)
        if store_offset == 0:
            return 0
        var store = gdef + store_offset
        var regions = store + self.u32(store + 2)
        var data_count = self.u16(store + 6)
        if outer >= data_count:
            return 0
        var data = store + self.u32(store + 8 + outer * 4)
        var items = self.u16(data)
        var word_word = self.u16(data + 2)
        var long_words = word_word & 0x8000 != 0
        var word_count = word_word & 0x7FFF
        var region_count = self.u16(data + 4)
        if inner >= items:
            return 0
        var word_size = 4 if long_words else 2
        var small_size = 2 if long_words else 1
        var row_size = word_count * word_size + (region_count - word_count) * small_size
        var row = data + 6 + region_count * 2 + inner * row_size
        var axis_count = self.u16(regions)
        var total: Float32 = 0
        var at = row
        for r in range(region_count):
            var delta: Int
            if r < word_count:
                delta = self.i32(at) if long_words else self.i16(at)
                at += word_size
            else:
                if long_words:
                    delta = self.i16(at)
                else:
                    var b = self.u8(at)
                    delta = b - 256 if b >= 128 else b
                at += small_size
            var region = self.u16(data + 6 + r * 2)
            var peaks = List[Float32]()
            var starts = List[Float32]()
            var stops = List[Float32]()
            for axis in range(axis_count):
                var axis_at = regions + 4 + (region * axis_count + axis) * 6
                starts.append(self.f2dot14(axis_at))
                peaks.append(self.f2dot14(axis_at + 2))
                stops.append(self.f2dot14(axis_at + 4))
            total += Float32(delta) * self.region_scalar(peaks, starts, stops, True)
        return total

    def pair_kern(self, subtable: Int, left: Int, right: Int) -> Tuple[Bool, Float32]:
        var format = self.u16(subtable)
        var covered = self.coverage_index(subtable + self.u16(subtable + 2), left)
        if covered < 0:
            return (False, Float32(0))
        var format1 = self.u16(subtable + 4)
        var format2 = self.u16(subtable + 6)
        var size1 = bit_count(format1) * 2
        var size2 = bit_count(format2) * 2
        if format == 1:
            var sets = self.u16(subtable + 8)
            if covered >= sets:
                return (False, Float32(0))
            var pair_set = subtable + self.u16(subtable + 10 + covered * 2)
            var count = self.u16(pair_set)
            var record_size = 2 + size1 + size2
            var low = 0
            var high = count - 1
            while low <= high:
                var middle = (low + high) // 2
                var record = pair_set + 2 + middle * record_size
                var second = self.u16(record)
                if second < right:
                    low = middle + 1
                elif second > right:
                    high = middle - 1
                else:
                    return (True, self.value_x_advance(record + 2, format1, subtable))
            return (False, Float32(0))
        if format == 2:
            var class1 = self.class_of(subtable + self.u16(subtable + 8), left)
            var class2 = self.class_of(subtable + self.u16(subtable + 10), right)
            var class1_count = self.u16(subtable + 12)
            var class2_count = self.u16(subtable + 14)
            if class1 >= class1_count or class2 >= class2_count:
                return (False, Float32(0))
            var record = subtable + 16 + (class1 * class2_count + class2) * (size1 + size2)
            return (True, self.value_x_advance(record, format1, subtable))
        return (False, Float32(0))

    def kerning(self, left: Int, right: Int) -> Float32:
        """Pair adjustment between two glyphs in font units (negative pulls
        them together): GPOS `kern` lookups summed in lookup order, the
        first subtable that covers the pair inside each; else `kern`."""
        var total: Float32 = 0
        var found = False
        for lookup in self.kern_lookups:
            var kind = self.u16(lookup)
            var count = self.u16(lookup + 4)
            for index in range(count):
                var subtable = lookup + self.u16(lookup + 6 + index * 2)
                var sub_kind = kind
                if kind == 9:
                    sub_kind = self.u16(subtable + 2)
                    subtable = subtable + self.u32(subtable + 4)
                if sub_kind != 2:
                    continue
                var hit = self.pair_kern(subtable, left, right)
                if hit[0]:
                    total += hit[1]
                    found = True
                    break
        if found or len(self.kern_lookups) > 0:
            return total
        return self.old_kern(left, right)

    def old_kern(self, left: Int, right: Int) -> Float32:
        var kern = self.table("kern")
        if kern < 0 or self.u16(kern) != 0:
            return 0
        var tables = self.u16(kern + 2)
        var at = kern + 4
        var total: Float32 = 0
        for _ in range(tables):
            var length = self.u16(at + 2)
            var coverage = self.u16(at + 4)
            if coverage >> 8 == 0 and coverage & 1 != 0:
                var pairs = self.u16(at + 6)
                var key = (left << 16) | right
                var low = 0
                var high = pairs - 1
                while low <= high:
                    var middle = (low + high) // 2
                    var record = at + 14 + middle * 6
                    var value = self.u32(record)
                    if value < key:
                        low = middle + 1
                    elif value > key:
                        high = middle - 1
                    else:
                        total += Float32(self.i16(record + 4))
                        break
            at += length
        return total


def bit_count(value: Int) -> Int:
    var count = 0
    var v = value
    while v != 0:
        count += v & 1
        v >>= 1
    return count


def round_half_up(value: Float32) -> Float32:
    var whole = Float32(Int(value))
    if value >= 0:
        return whole + (Float32(1) if value - whole >= 0.5 else Float32(0))
    return whole - (Float32(1) if whole - value > 0.5 else Float32(0))


def sort_ints(mut values: List[Int]):
    for i in range(1, len(values)):
        var v = values[i]
        var j = i - 1
        while j >= 0 and values[j] > v:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = v


def interpolate_untouched(original: List[Float32], touched: List[Bool], mut deltas: List[Float32], ends: List[Int]):
    """IUP along one axis, contour by contour (gvar, "Inferred deltas for
    un-referenced point numbers")."""
    var start = 0
    for contour in range(len(ends)):
        var end = ends[contour]
        if end >= len(original) - 4 or end < start:
            break
        var first_touched = -1
        for p in range(start, end + 1):
            if touched[p]:
                first_touched = p
                break
        if first_touched < 0:
            start = end + 1
            continue
        var span = end - start + 1
        var p = first_touched
        for _ in range(span):
            # Find the next touched point after p (cyclically).
            var q = p
            var gap = 0
            while True:
                q = start + (q - start + 1) % span
                gap += 1
                if touched[q] or gap > span:
                    break
            if gap > 1:
                var u = start + (p - start + 1) % span
                while u != q:
                    deltas[u] = infer_delta(original[u], original[p], original[q], deltas[p], deltas[q])
                    u = start + (u - start + 1) % span
            p = q
            if p == first_touched:
                break
        start = end + 1


def infer_delta(value: Float32, a: Float32, b: Float32, da: Float32, db: Float32) -> Float32:
    var low = a
    var high = b
    var dl = da
    var dh = db
    if low > high:
        low = b
        high = a
        dl = db
        dh = da
    if value <= low:
        return dl
    if value >= high:
        return dh
    if high == low:
        return dl if da == db else Float32(0)
    return dl + (dh - dl) * (value - low) / (high - low)
