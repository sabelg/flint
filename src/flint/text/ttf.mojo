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

- CFF/OpenType (`OTTO`) outlines with Type 2 charstrings, `GDEF` glyph classes
  and mark filtering sets, and `GSUB` substitutions (LookupTypes 1-7).
Everything is in font units; `text/sdf_bake.mojo` scales.

References: the OpenType specification 1.9 (Microsoft), chapters cmap,
glyf, gvar, avar, fvar, GPOS, GDEF and "OpenType font variations overview".
"""
from std.collections import Dict
from flint.platform.os import read_bytes


comptime TAG_OTTO: Int = 0x4F54544F
comptime MAX_COMPONENT_DEPTH: Int = 8

# OpenType GSUB Feature Bitmasks (used by TextShaper / TrueTypeFont.apply_gsub)
comptime FEAT_LOCL: UInt32 = 1 << 0
comptime FEAT_CCMP: UInt32 = 1 << 1
comptime FEAT_ISOL: UInt32 = 1 << 2
comptime FEAT_INIT: UInt32 = 1 << 3
comptime FEAT_MEDI: UInt32 = 1 << 4
comptime FEAT_FINA: UInt32 = 1 << 5
comptime FEAT_RLIG: UInt32 = 1 << 6
comptime FEAT_LIGA: UInt32 = 1 << 7
comptime FEAT_CLIG: UInt32 = 1 << 8
comptime FEAT_CALT: UInt32 = 1 << 9
comptime FEAT_DEFAULT: UInt32 = FEAT_LOCL | FEAT_CCMP | FEAT_RLIG | FEAT_LIGA | FEAT_CLIG | FEAT_CALT

# GDEF GlyphClassDef values & LookupFlag bits
comptime GDEF_CLASS_BASE: Int = 1
comptime GDEF_CLASS_LIGATURE: Int = 2
comptime GDEF_CLASS_MARK: Int = 3
comptime GDEF_CLASS_COMPONENT: Int = 4
comptime LOOKUP_FLAG_IGNORE_BASE: Int = 0x0002
comptime LOOKUP_FLAG_IGNORE_LIGATURES: Int = 0x0004
comptime LOOKUP_FLAG_IGNORE_MARKS: Int = 0x0008
comptime LOOKUP_FLAG_USE_MARK_FILTERING_SET: Int = 0x0010
comptime LOOKUP_FLAG_MARK_ATTACHMENT_MASK: Int = 0xFF00


@fieldwise_init
struct GsubSequenceLookup(Copyable, ImplicitlyCopyable, Movable):
    var sequence_index: Int
    var lookup_index: Int


struct GsubGlyph(Copyable, ImplicitlyCopyable, Movable):
    var glyph_id: Int
    var cluster: Int
    var mask: UInt32
    var alt_choice: Int

    def __init__(out self, glyph_id: Int, cluster: Int = 0, mask: UInt32 = FEAT_DEFAULT, alt_choice: Int = 0):
        self.glyph_id = glyph_id
        self.cluster = cluster
        self.mask = mask
        self.alt_choice = alt_choice


@fieldwise_init
struct CffSlice(Copyable, ImplicitlyCopyable, Movable):
    var offset: Int
    var length: Int


struct CffPrivateDict(Copyable, Movable):
    var default_width: Float32
    var nominal_width: Float32
    var local_subrs: List[CffSlice]

    def __init__(out self):
        self.default_width = 0.0
        self.nominal_width = 0.0
        self.local_subrs = List[CffSlice]()


struct _T2State(Movable):
    var stack: List[Float32]
    var trans: List[Float32]
    var cur_x: Float32
    var cur_y: Float32
    var stem_count: Int
    var have_width: Bool
    var width: Float32
    var contour_open: Bool
    var contour_start: Int
    var done: Bool

    def __init__(out self, default_width: Float32):
        self.stack = List[Float32]()
        self.trans = List[Float32](length=32, fill=0.0)
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.stem_count = 0
        self.have_width = False
        self.width = default_width
        self.contour_open = False
        self.contour_start = 0
        self.done = False


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
    var is_cubic: Bool

    def __init__(out self):
        self.xs = List[Float32]()
        self.ys = List[Float32]()
        self.on = List[Bool]()
        self.ends = List[Int]()
        self.advance = 0
        self.lsb = 0
        self.is_cubic = False

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
    # CFF / OTF (OTTO) state
    var is_cff: Bool
    var cff_base: Int
    var cff_end: Int
    var cff_charstrings: List[CffSlice]
    var cff_gsubrs: List[CffSlice]
    var cff_private: CffPrivateDict
    var cff_fd_privates: List[CffPrivateDict]
    var cff_fd_select_off: Int
    var cff_encoding_off: Int
    var cff_charset_off: Int
    var cff_units_per_em: Int
    var has_hmtx: Bool
    # GDEF / GSUB cached table offsets
    var gdef_offset: Int
    var gsub_offset: Int

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
        self.is_cff = False
        self.cff_base = -1
        self.cff_end = -1
        self.cff_charstrings = List[CffSlice]()
        self.cff_gsubrs = List[CffSlice]()
        self.cff_private = CffPrivateDict()
        self.cff_fd_privates = List[CffPrivateDict]()
        self.cff_fd_select_off = -1
        self.cff_encoding_off = -1
        self.cff_charset_off = -1
        self.cff_units_per_em = 1000
        self.has_hmtx = False
        self.gdef_offset = -1
        self.gsub_offset = -1

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

    def table_length(self, tag: String) -> Int:
        var found = self.lengths.get(tag_value(tag))
        if found:
            return found.value()
        return 0

    # ---- loading ----

    def load(mut self, path: String) -> Bool:
        return self.load_bytes(read_bytes(path))

    def load_bytes(mut self, var data: List[UInt8]) -> Bool:
        self.data = data^
        self.valid = False
        self.tables = Dict[Int, Int]()
        self.lengths = Dict[Int, Int]()
        self.cmap_offset = -1
        self.cmap_format = 0
        self.is_cff = False
        self.cff_base = -1
        self.cff_end = -1
        self.cff_charstrings = List[CffSlice]()
        self.cff_gsubrs = List[CffSlice]()
        self.cff_private = CffPrivateDict()
        self.cff_fd_privates = List[CffPrivateDict]()
        self.cff_fd_select_off = -1
        self.cff_encoding_off = -1
        self.cff_charset_off = -1
        self.cff_units_per_em = 1000
        self.has_hmtx = False
        self.gdef_offset = -1
        self.gsub_offset = -1
        if len(self.data) < 12:
            self.error = "not a font file (too short)"
            return False
        var version = self.u32(0)
        if version != TAG_OTTO and version != 0x00010000 and version != tag_value("true"):
            self.error = "not a TrueType/OpenType font (version " + String(version) + ")"
            return False
        self.is_cff = (version == TAG_OTTO)
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
        if self.is_cff:
            if self.table("CFF ") < 0:
                self.error = "missing table CFF "
                return False
            if not self.parse_cff():
                return False
            var head = self.table("head")
            if head >= 0 and self.table_length("head") >= 52:
                self.units_per_em = self.u16(head + 18)
                self.index_to_loc = self.i16(head + 50)
            else:
                self.units_per_em = self.cff_units_per_em
                self.index_to_loc = 0
            if self.units_per_em < 16 or self.units_per_em > 16384:
                self.error = "bad unitsPerEm"
                return False
            var maxp = self.table("maxp")
            if maxp >= 0 and self.table_length("maxp") >= 6:
                self.glyph_count = self.u16(maxp + 4)
            else:
                self.glyph_count = len(self.cff_charstrings)
            var hhea = self.table("hhea")
            if hhea >= 0 and self.table_length("hhea") >= 36:
                self.ascender = self.i16(hhea + 4)
                self.descender = self.i16(hhea + 6)
                self.line_gap = self.i16(hhea + 8)
                self.metric_count = self.u16(hhea + 34)
            else:
                self.ascender = Int(Float32(self.units_per_em) * 0.8)
                self.descender = -Int(Float32(self.units_per_em) * 0.2)
                self.line_gap = 0
                self.metric_count = max(1, self.glyph_count)
            self.has_hmtx = self.table("hmtx") >= 0
            var os2 = self.table("OS/2")
            self.cap_height = Int(Float32(self.units_per_em) * 0.7)
            self.x_height = Int(Float32(self.units_per_em) * 0.5)
            if os2 >= 0 and self.u16(os2) >= 2:
                self.x_height = self.i16(os2 + 86)
                self.cap_height = self.i16(os2 + 88)
            if self.table("cmap") >= 0:
                if not self.pick_cmap():
                    self.error = "no Unicode cmap (format 4 or 12)"
                    return False
        else:
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
            self.has_hmtx = True
            var os2 = self.table("OS/2")
            self.cap_height = Int(Float32(self.units_per_em) * 0.7)
            self.x_height = Int(Float32(self.units_per_em) * 0.5)
            if os2 >= 0 and self.u16(os2) >= 2:
                self.x_height = self.i16(os2 + 86)
                self.cap_height = self.i16(os2 + 88)
            if not self.pick_cmap():
                self.error = "no Unicode cmap (format 4 or 12)"
                return False
        self.gdef_offset = self.table("GDEF")
        self.gsub_offset = self.table("GSUB")
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
        if self.cmap_offset < 0:
            if self.is_cff and self.cff_encoding_off > 0 and code >= 0 and code <= 255:
                var enc_fmt = self.u8(self.cff_encoding_off) & 0x7F
                if enc_fmt == 0:
                    var n_codes = self.u8(self.cff_encoding_off + 1)
                    for gid in range(1, n_codes + 1):
                        if self.u8(self.cff_encoding_off + 1 + gid) == code:
                            return gid
                elif enc_fmt == 1:
                    var n_ranges = self.u8(self.cff_encoding_off + 1)
                    var gid = 1
                    for r in range(n_ranges):
                        var first = self.u8(self.cff_encoding_off + 2 + r * 2)
                        var n_left = self.u8(self.cff_encoding_off + 3 + r * 2)
                        if code >= first and code <= first + n_left:
                            return gid + (code - first)
                        gid += n_left + 1
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
        if not self.has_hmtx or self.metric_count < 1:
            return 0
        var hmtx = self.table("hmtx")
        if hmtx < 0:
            return 0
        var index = min(max(0, glyph), self.metric_count - 1)
        return self.u16(hmtx + index * 4)

    def default_lsb(self, glyph: Int) -> Int:
        if not self.has_hmtx or self.metric_count < 1:
            return 0
        var hmtx = self.table("hmtx")
        if hmtx < 0:
            return 0
        if glyph < self.metric_count:
            return self.i16(hmtx + glyph * 4 + 2)
        return self.i16(hmtx + self.metric_count * 4 + (glyph - self.metric_count) * 2)

    def advance(self, glyph: Int) -> Float32:
        """Advance width in font units at the current instance."""
        if self.is_cff and not self.has_hmtx:
            return self.outline(glyph).advance
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

    # ---- CFF / Type2 CharString ----

    def cff_uint(self, at: Int, off_size: Int) -> Int:
        var v = 0
        for i in range(off_size):
            v = (v << 8) | self.u8(at + i)
        return v

    def cff_read_index(self, at: Int) -> Tuple[Bool, List[CffSlice], Int]:
        var slices = List[CffSlice]()
        if at < 0 or at + 2 > self.cff_end:
            return (False, slices^, at)
        var count = self.u16(at)
        if count == 0:
            return (True, slices^, at + 2)
        if at + 3 > self.cff_end:
            return (False, slices^, at)
        var off_size = self.u8(at + 2)
        if off_size < 1 or off_size > 4:
            return (False, slices^, at)
        var off_arr = at + 3
        var data_base = off_arr + (count + 1) * off_size - 1
        if data_base + 1 > self.cff_end:
            return (False, slices^, at)
        for i in range(count):
            var o0 = self.cff_uint(off_arr + i * off_size, off_size)
            var o1 = self.cff_uint(off_arr + (i + 1) * off_size, off_size)
            if o0 < 1 or o1 < o0 or data_base + o1 > self.cff_end:
                return (False, List[CffSlice](), at)
            slices.append(CffSlice(data_base + o0, o1 - o0))
        var last_off = self.cff_uint(off_arr + count * off_size, off_size)
        var next_at = data_base + last_off
        if next_at > self.cff_end:
            return (False, List[CffSlice](), at)
        return (True, slices^, next_at)

    def cff_decode_real(self, mut at: Int, end: Int) -> Float32:
        var s = String()
        var done = False
        while at < end and not done:
            var b = self.u8(at)
            at += 1
            for half in range(2):
                var nib = (b >> 4) if half == 0 else (b & 0xF)
                if nib <= 9:
                    s += String(nib)
                elif nib == 0xA:
                    s += "."
                elif nib == 0xB:
                    s += "e"
                elif nib == 0xC:
                    s += "e-"
                elif nib == 0xE:
                    s += "-"
                elif nib == 0xF:
                    done = True
                    break
        if s.byte_length() == 0:
            return 0.0
        try:
            return Float32(Float64(s))
        except:
            return 0.0

    def cff_parse_dict(
        mut self,
        start: Int,
        length: Int,
        mut priv: CffPrivateDict,
        is_top: Bool,
        mut charstrings_off: Int,
        mut encoding_off: Int,
        mut charset_off: Int,
        mut priv_size: Int,
        mut priv_off: Int,
        mut fd_array_off: Int,
        mut fd_select_off: Int,
        mut subrs_rel_off: Int,
    ):
        var end = min(self.cff_end, start + length)
        var at = start
        var ops = List[Float32]()
        while at < end:
            var b0 = self.u8(at)
            if b0 == 28:
                if at + 3 > end:
                    break
                ops.append(Float32(self.i16(at + 1)))
                at += 3
            elif b0 == 29:
                if at + 5 > end:
                    break
                ops.append(Float32(self.i32(at + 1)))
                at += 5
            elif b0 == 30:
                at += 1
                ops.append(self.cff_decode_real(at, end))
            elif b0 >= 32 and b0 <= 246:
                ops.append(Float32(b0 - 139))
                at += 1
            elif b0 >= 247 and b0 <= 250:
                if at + 2 > end:
                    break
                var b1 = self.u8(at + 1)
                ops.append(Float32((b0 - 247) * 256 + b1 + 108))
                at += 2
            elif b0 >= 251 and b0 <= 254:
                if at + 2 > end:
                    break
                var b1 = self.u8(at + 1)
                ops.append(Float32(-(b0 - 251) * 256 - b1 - 108))
                at += 2
            elif b0 == 12:
                if at + 2 > end:
                    break
                var op2 = self.u8(at + 1)
                at += 2
                if is_top:
                    if op2 == 7 and len(ops) >= 4:
                        var sy = abs(ops[3])
                        if sy > Float32(1.0e-7):
                            self.cff_units_per_em = max(16, min(16384, Int(round_half_up(Float32(1.0) / sy))))
                    elif op2 == 36 and len(ops) >= 1:
                        fd_array_off = Int(ops[0])
                    elif op2 == 37 and len(ops) >= 1:
                        fd_select_off = Int(ops[0])
                ops.clear()
            elif b0 <= 21:
                at += 1
                if is_top:
                    if b0 == 15 and len(ops) >= 1:
                        charset_off = Int(ops[0])
                    elif b0 == 16 and len(ops) >= 1:
                        encoding_off = Int(ops[0])
                    elif b0 == 17 and len(ops) >= 1:
                        charstrings_off = Int(ops[0])
                    elif b0 == 18 and len(ops) >= 2:
                        priv_size = Int(ops[0])
                        priv_off = Int(ops[1])
                else:
                    if b0 == 19 and len(ops) >= 1:
                        subrs_rel_off = Int(ops[0])
                    elif b0 == 20 and len(ops) >= 1:
                        priv.default_width = ops[0]
                    elif b0 == 21 and len(ops) >= 1:
                        priv.nominal_width = ops[0]
                ops.clear()
            else:
                at += 1
                ops.clear()

    def parse_cff(mut self) -> Bool:
        var cff = self.table("CFF ")
        var cff_len = self.table_length("CFF ")
        if cff < 0 or cff_len < 4 or cff + cff_len > len(self.data):
            self.error = "invalid CFF table"
            return False
        self.cff_base = cff
        self.cff_end = cff + cff_len
        var major = self.u8(cff)
        var hdr_size = self.u8(cff + 2)
        if major != 1 or hdr_size < 4 or cff + hdr_size > self.cff_end:
            self.error = "unsupported CFF header"
            return False
        var pos = cff + hdr_size
        # Allow minimal CFF v1.0 header-only containers (e.g. 8-byte CFF stub)
        if pos == self.cff_end:
            return True
        var name_res = self.cff_read_index(pos)
        if not name_res[0]:
            self.error = "invalid CFF Name INDEX"
            return False
        if len(name_res[1]) == 0:
            return True
        pos = name_res[2]
        var top_res = self.cff_read_index(pos)
        if not top_res[0] or len(top_res[1]) == 0:
            self.error = "invalid CFF Top DICT INDEX"
            return False
        pos = top_res[2]
        var str_res = self.cff_read_index(pos)
        if not str_res[0]:
            self.error = "invalid CFF String INDEX"
            return False
        pos = str_res[2]
        var gsubr_res = self.cff_read_index(pos)
        if not gsubr_res[0]:
            self.error = "invalid CFF Global Subr INDEX"
            return False
        self.cff_gsubrs = gsubr_res[1].copy()

        var top_slice = top_res[1][0]
        var charstrings_off = -1
        var encoding_off = -1
        var charset_off = -1
        var priv_size = 0
        var priv_off = 0
        var fd_array_off = -1
        var fd_select_off = -1
        var unused_subr = -1
        var dummy_priv = CffPrivateDict()
        self.cff_parse_dict(
            top_slice.offset,
            top_slice.length,
            dummy_priv,
            True,
            charstrings_off,
            encoding_off,
            charset_off,
            priv_size,
            priv_off,
            fd_array_off,
            fd_select_off,
            unused_subr,
        )
        if encoding_off > 1 and cff + encoding_off < self.cff_end:
            self.cff_encoding_off = cff + encoding_off
        if charset_off > 0 and cff + charset_off < self.cff_end:
            self.cff_charset_off = cff + charset_off
        if charstrings_off > 0:
            var cs_res = self.cff_read_index(cff + charstrings_off)
            if not cs_res[0]:
                self.error = "invalid CFF CharStrings INDEX"
                return False
            self.cff_charstrings = cs_res[1].copy()
        if priv_size > 0 and priv_off > 0 and cff + priv_off + priv_size <= self.cff_end:
            var subr_rel = -1
            var u1 = 0; var u2 = 0; var u3 = 0; var u4 = 0; var u5 = 0; var u6 = 0; var u7 = 0
            var priv = CffPrivateDict()
            self.cff_parse_dict(
                cff + priv_off,
                priv_size,
                priv,
                False,
                u1, u2, u3, u4, u5, u6, u7, subr_rel,
            )
            if subr_rel > 0 and cff + priv_off + subr_rel < self.cff_end:
                var ls_res = self.cff_read_index(cff + priv_off + subr_rel)
                if ls_res[0]:
                    priv.local_subrs = ls_res[1].copy()
            self.cff_private = priv^
        if fd_array_off > 0 and fd_select_off > 0 and cff + fd_array_off < self.cff_end:
            self.cff_fd_select_off = cff + fd_select_off
            var fd_res = self.cff_read_index(cff + fd_array_off)
            if fd_res[0]:
                for fd_slice in fd_res[1]:
                    var fd_priv = CffPrivateDict()
                    var f_psize = 0
                    var f_poff = 0
                    var u1 = 0; var u2 = 0; var u3 = 0; var u4 = 0; var u5 = 0; var u6 = 0
                    self.cff_parse_dict(
                        fd_slice.offset,
                        fd_slice.length,
                        fd_priv,
                        True,
                        u1, u2, u3, f_psize, f_poff, u4, u5, u6,
                    )
                    if f_psize > 0 and f_poff > 0 and cff + f_poff + f_psize <= self.cff_end:
                        var s_rel = -1
                        var v1 = 0; var v2 = 0; var v3 = 0; var v4 = 0; var v5 = 0; var v6 = 0; var v7 = 0
                        self.cff_parse_dict(
                            cff + f_poff,
                            f_psize,
                            fd_priv,
                            False,
                            v1, v2, v3, v4, v5, v6, v7, s_rel,
                        )
                        if s_rel > 0 and cff + f_poff + s_rel < self.cff_end:
                            var fls_res = self.cff_read_index(cff + f_poff + s_rel)
                            if fls_res[0]:
                                fd_priv.local_subrs = fls_res[1].copy()
                    self.cff_fd_privates.append(fd_priv^)
        return True

    def cff_fd_for_glyph(self, glyph: Int) -> Int:
        if self.cff_fd_select_off < 0 or self.cff_fd_select_off >= self.cff_end:
            return 0
        var off = self.cff_fd_select_off
        var fmt = self.u8(off)
        if fmt == 0:
            if off + 1 + glyph < self.cff_end:
                return self.u8(off + 1 + glyph)
            return 0
        if fmt == 3:
            var n_ranges = self.u16(off + 1)
            var low = 0
            var high = n_ranges - 1
            while low <= high:
                var mid = (low + high) // 2
                var rec = off + 3 + mid * 3
                var first = self.u16(rec)
                var next_first = self.u16(rec + 3)
                if glyph < first:
                    high = mid - 1
                elif glyph >= next_first:
                    low = mid + 1
                else:
                    return self.u8(rec + 2)
        return 0

    @staticmethod
    def cff_subr_bias(count: Int) -> Int:
        if count < 1240:
            return 107
        if count < 33900:
            return 1131
        return 32768

    @staticmethod
    def _t2_close_contour(mut state: _T2State, mut out: GlyphOutline):
        if not state.contour_open:
            return
        var n = len(out.xs)
        if n > state.contour_start + 1:
            out.ends.append(n - 1)
        elif n == state.contour_start + 1:
            _ = out.xs.pop()
            _ = out.ys.pop()
            _ = out.on.pop()
        state.contour_open = False

    @staticmethod
    def _t2_moveto(mut state: _T2State, mut out: GlyphOutline, dx: Float32, dy: Float32):
        TrueTypeFont._t2_close_contour(state, out)
        state.cur_x += dx
        state.cur_y += dy
        state.contour_start = len(out.xs)
        state.contour_open = True
        out.xs.append(state.cur_x)
        out.ys.append(state.cur_y)
        out.on.append(True)

    @staticmethod
    def _t2_lineto(mut state: _T2State, mut out: GlyphOutline, dx: Float32, dy: Float32):
        state.cur_x += dx
        state.cur_y += dy
        out.xs.append(state.cur_x)
        out.ys.append(state.cur_y)
        out.on.append(True)

    @staticmethod
    def _t2_curveto(
        mut state: _T2State,
        mut out: GlyphOutline,
        dx1: Float32,
        dy1: Float32,
        dx2: Float32,
        dy2: Float32,
        dx3: Float32,
        dy3: Float32,
    ):
        var c1x = state.cur_x + dx1
        var c1y = state.cur_y + dy1
        var c2x = c1x + dx2
        var c2y = c1y + dy2
        var p1x = c2x + dx3
        var p1y = c2y + dy3
        out.xs.append(c1x); out.ys.append(c1y); out.on.append(False)
        out.xs.append(c2x); out.ys.append(c2y); out.on.append(False)
        out.xs.append(p1x); out.ys.append(p1y); out.on.append(True)
        state.cur_x = p1x
        state.cur_y = p1y

    def _exec_t2_slice(
        self,
        slice: CffSlice,
        priv: CffPrivateDict,
        mut state: _T2State,
        mut out: GlyphOutline,
        depth: Int,
    ):
        if depth > 16 or state.done:
            return
        var pc = slice.offset
        var end = min(self.cff_end, slice.offset + slice.length)
        while pc < end and not state.done:
            var b0 = self.u8(pc)
            if b0 == 28:
                if pc + 3 > end:
                    break
                state.stack.append(Float32(self.i16(pc + 1)))
                pc += 3
            elif b0 >= 32 and b0 <= 246:
                state.stack.append(Float32(b0 - 139))
                pc += 1
            elif b0 >= 247 and b0 <= 250:
                if pc + 2 > end:
                    break
                var b1 = self.u8(pc + 1)
                state.stack.append(Float32((b0 - 247) * 256 + b1 + 108))
                pc += 2
            elif b0 >= 251 and b0 <= 254:
                if pc + 2 > end:
                    break
                var b1 = self.u8(pc + 1)
                state.stack.append(Float32(-(b0 - 251) * 256 - b1 - 108))
                pc += 2
            elif b0 == 255:
                if pc + 5 > end:
                    break
                state.stack.append(Float32(Float64(self.i32(pc + 1)) / 65536.0))
                pc += 5
            elif b0 == 12:
                if pc + 2 > end:
                    break
                var op2 = self.u8(pc + 1)
                pc += 2
                var n = len(state.stack)
                if op2 == 34 and n >= 7:  # hflex
                    var base = n - 7
                    var dx1 = state.stack[base]; var dx2 = state.stack[base + 1]; var dy2 = state.stack[base + 2]
                    var dx3 = state.stack[base + 3]; var dx4 = state.stack[base + 4]; var dx5 = state.stack[base + 5]
                    var dx6 = state.stack[base + 6]
                    Self._t2_curveto(state, out, dx1, 0.0, dx2, dy2, dx3, 0.0)
                    Self._t2_curveto(state, out, dx4, 0.0, dx5, -dy2, dx6, 0.0)
                    state.stack.clear()
                elif op2 == 35 and n >= 13:  # flex
                    var base = n - 13
                    Self._t2_curveto(
                        state, out,
                        state.stack[base], state.stack[base + 1],
                        state.stack[base + 2], state.stack[base + 3],
                        state.stack[base + 4], state.stack[base + 5],
                    )
                    Self._t2_curveto(
                        state, out,
                        state.stack[base + 6], state.stack[base + 7],
                        state.stack[base + 8], state.stack[base + 9],
                        state.stack[base + 10], state.stack[base + 11],
                    )
                    state.stack.clear()
                elif op2 == 36 and n >= 9:  # hflex1
                    var base = n - 9
                    var dx1 = state.stack[base]; var dy1 = state.stack[base + 1]
                    var dx2 = state.stack[base + 2]; var dy2 = state.stack[base + 3]
                    var dx3 = state.stack[base + 4]; var dx4 = state.stack[base + 5]
                    var dx5 = state.stack[base + 6]; var dy5 = state.stack[base + 7]
                    var dx6 = state.stack[base + 8]
                    Self._t2_curveto(state, out, dx1, dy1, dx2, dy2, dx3, 0.0)
                    Self._t2_curveto(state, out, dx4, 0.0, dx5, dy5, dx6, -(dy1 + dy2 + dy5))
                    state.stack.clear()
                elif op2 == 37 and n >= 11:  # flex1
                    var base = n - 11
                    var dx1 = state.stack[base]; var dy1 = state.stack[base + 1]
                    var dx2 = state.stack[base + 2]; var dy2 = state.stack[base + 3]
                    var dx3 = state.stack[base + 4]; var dy3 = state.stack[base + 5]
                    var dx4 = state.stack[base + 6]; var dy4 = state.stack[base + 7]
                    var dx5 = state.stack[base + 8]; var dy5 = state.stack[base + 9]
                    var d6 = state.stack[base + 10]
                    var sum_dx = dx1 + dx2 + dx3 + dx4 + dx5
                    var sum_dy = dy1 + dy2 + dy3 + dy4 + dy5
                    Self._t2_curveto(state, out, dx1, dy1, dx2, dy2, dx3, dy3)
                    if abs(sum_dx) > abs(sum_dy):
                        Self._t2_curveto(state, out, dx4, dy4, dx5, dy5, d6, -sum_dy)
                    else:
                        Self._t2_curveto(state, out, dx4, dy4, dx5, dy5, -sum_dx, d6)
                    state.stack.clear()
                elif op2 == 3 and n >= 2:  # and
                    var b = state.stack.pop(); var a = state.stack.pop()
                    state.stack.append(Float32(1.0) if (a != 0.0 and b != 0.0) else Float32(0.0))
                elif op2 == 4 and n >= 2:  # or
                    var b = state.stack.pop(); var a = state.stack.pop()
                    state.stack.append(Float32(1.0) if (a != 0.0 or b != 0.0) else Float32(0.0))
                elif op2 == 5 and n >= 1:  # not
                    var a = state.stack.pop()
                    state.stack.append(Float32(1.0) if a == 0.0 else Float32(0.0))
                elif op2 == 9 and n >= 1:  # abs
                    var a = state.stack.pop()
                    state.stack.append(abs(a))
                elif op2 == 10 and n >= 2:  # add
                    var b = state.stack.pop(); var a = state.stack.pop()
                    state.stack.append(a + b)
                elif op2 == 11 and n >= 2:  # sub
                    var b = state.stack.pop(); var a = state.stack.pop()
                    state.stack.append(a - b)
                elif op2 == 12 and n >= 2:  # div
                    var b = state.stack.pop(); var a = state.stack.pop()
                    state.stack.append(a / b if b != 0.0 else Float32(0.0))
                elif op2 == 14 and n >= 1:  # neg
                    var a = state.stack.pop()
                    state.stack.append(-a)
                elif op2 == 15 and n >= 2:  # eq
                    var b = state.stack.pop(); var a = state.stack.pop()
                    state.stack.append(Float32(1.0) if a == b else Float32(0.0))
                elif op2 == 18 and n >= 1:  # drop
                    _ = state.stack.pop()
                elif op2 == 20 and n >= 2:  # put
                    var idx = Int(state.stack.pop()); var val = state.stack.pop()
                    if idx >= 0 and idx < len(state.trans):
                        state.trans[idx] = val
                elif op2 == 21 and n >= 1:  # get
                    var idx = Int(state.stack.pop())
                    state.stack.append(state.trans[idx] if (idx >= 0 and idx < len(state.trans)) else Float32(0.0))
                elif op2 == 22 and n >= 4:  # ifelse
                    var v2 = state.stack.pop(); var v1 = state.stack.pop()
                    var s2 = state.stack.pop(); var s1 = state.stack.pop()
                    state.stack.append(s1 if v1 <= v2 else s2)
                elif op2 == 24 and n >= 2:  # mul
                    var b = state.stack.pop(); var a = state.stack.pop()
                    state.stack.append(a * b)
                elif op2 == 26 and n >= 1:  # sqrt
                    var a = state.stack.pop()
                    state.stack.append(a ** 0.5 if a > 0.0 else Float32(0.0))
                elif op2 == 27 and n >= 1:  # dup
                    state.stack.append(state.stack[n - 1])
                elif op2 == 28 and n >= 2:  # exch
                    var a = state.stack[n - 1]; var b = state.stack[n - 2]
                    state.stack[n - 1] = b; state.stack[n - 2] = a
                elif op2 == 29 and n >= 1:  # index
                    var i = Int(state.stack.pop())
                    var m = len(state.stack)
                    if m > 0:
                        var pick = max(0, min(m - 1, m - 1 - max(0, i)))
                        state.stack.append(state.stack[pick])
                elif op2 == 30 and n >= 2:  # roll
                    var j = Int(state.stack.pop())
                    var k = Int(state.stack.pop())
                    var m = len(state.stack)
                    if k > 0 and k <= m:
                        var shift = ((j % k) + k) % k
                        if shift != 0:
                            var tmp = List[Float32](capacity=k)
                            var start_idx = m - k
                            for idx in range(k):
                                tmp.append(state.stack[start_idx + (idx + k - shift) % k])
                            for idx in range(k):
                                state.stack[start_idx + idx] = tmp[idx]
                else:
                    state.stack.clear()
            else:
                pc += 1
                if b0 == 1 or b0 == 3 or b0 == 18 or b0 == 23:  # hstem, vstem, hstemhm, vstemhm
                    if not state.have_width:
                        if (len(state.stack) & 1) == 1:
                            state.width = priv.nominal_width + state.stack[0]
                            _ = state.stack.pop(0)
                        state.have_width = True
                    state.stem_count += len(state.stack) // 2
                    state.stack.clear()
                elif b0 == 19 or b0 == 20:  # hintmask, cntrmask
                    if not state.have_width:
                        if (len(state.stack) & 1) == 1:
                            state.width = priv.nominal_width + state.stack[0]
                            _ = state.stack.pop(0)
                        state.have_width = True
                    state.stem_count += len(state.stack) // 2
                    state.stack.clear()
                    var mask_bytes = (state.stem_count + 7) // 8
                    pc = min(end, pc + mask_bytes)
                elif b0 == 10:  # callsubr
                    if len(state.stack) >= 1 and len(priv.local_subrs) > 0:
                        var bias = Self.cff_subr_bias(len(priv.local_subrs))
                        var idx = Int(state.stack.pop()) + bias
                        if idx >= 0 and idx < len(priv.local_subrs):
                            self._exec_t2_slice(priv.local_subrs[idx], priv, state, out, depth + 1)
                elif b0 == 29:  # callgsubr
                    if len(state.stack) >= 1 and len(self.cff_gsubrs) > 0:
                        var bias = Self.cff_subr_bias(len(self.cff_gsubrs))
                        var idx = Int(state.stack.pop()) + bias
                        if idx >= 0 and idx < len(self.cff_gsubrs):
                            self._exec_t2_slice(self.cff_gsubrs[idx], priv, state, out, depth + 1)
                elif b0 == 11:  # return
                    return
                elif b0 == 14:  # endchar
                    if not state.have_width:
                        if len(state.stack) > 0:
                            state.width = priv.nominal_width + state.stack[0]
                        state.have_width = True
                    state.stack.clear()
                    state.done = True
                    return
                elif b0 == 21:  # rmoveto
                    if not state.have_width:
                        if len(state.stack) > 2:
                            state.width = priv.nominal_width + state.stack[0]
                            _ = state.stack.pop(0)
                        state.have_width = True
                    if len(state.stack) >= 2:
                        Self._t2_moveto(state, out, state.stack[0], state.stack[1])
                    state.stack.clear()
                elif b0 == 22:  # hmoveto
                    if not state.have_width:
                        if len(state.stack) > 1:
                            state.width = priv.nominal_width + state.stack[0]
                            _ = state.stack.pop(0)
                        state.have_width = True
                    if len(state.stack) >= 1:
                        Self._t2_moveto(state, out, state.stack[0], 0.0)
                    state.stack.clear()
                elif b0 == 4:  # vmoveto
                    if not state.have_width:
                        if len(state.stack) > 1:
                            state.width = priv.nominal_width + state.stack[0]
                            _ = state.stack.pop(0)
                        state.have_width = True
                    if len(state.stack) >= 1:
                        Self._t2_moveto(state, out, 0.0, state.stack[0])
                    state.stack.clear()
                elif b0 == 5:  # rlineto
                    var i = 0
                    while i + 1 < len(state.stack):
                        Self._t2_lineto(state, out, state.stack[i], state.stack[i + 1])
                        i += 2
                    state.stack.clear()
                elif b0 == 6 or b0 == 7:  # hlineto / vlineto
                    var horiz = (b0 == 6)
                    for i in range(len(state.stack)):
                        if horiz:
                            Self._t2_lineto(state, out, state.stack[i], 0.0)
                        else:
                            Self._t2_lineto(state, out, 0.0, state.stack[i])
                        horiz = not horiz
                    state.stack.clear()
                elif b0 == 8:  # rrcurveto
                    var i = 0
                    while i + 5 < len(state.stack):
                        Self._t2_curveto(
                            state, out,
                            state.stack[i], state.stack[i + 1],
                            state.stack[i + 2], state.stack[i + 3],
                            state.stack[i + 4], state.stack[i + 5],
                        )
                        i += 6
                    state.stack.clear()
                elif b0 == 24:  # rcurveline
                    var n = len(state.stack)
                    var i = 0
                    while i + 7 < n:
                        Self._t2_curveto(
                            state, out,
                            state.stack[i], state.stack[i + 1],
                            state.stack[i + 2], state.stack[i + 3],
                            state.stack[i + 4], state.stack[i + 5],
                        )
                        i += 6
                    if i + 1 < n:
                        Self._t2_lineto(state, out, state.stack[i], state.stack[i + 1])
                    state.stack.clear()
                elif b0 == 25:  # rlinecurve
                    var n = len(state.stack)
                    var i = 0
                    while i + 7 < n:
                        Self._t2_lineto(state, out, state.stack[i], state.stack[i + 1])
                        i += 2
                    if i + 5 < n:
                        Self._t2_curveto(
                            state, out,
                            state.stack[i], state.stack[i + 1],
                            state.stack[i + 2], state.stack[i + 3],
                            state.stack[i + 4], state.stack[i + 5],
                        )
                    state.stack.clear()
                elif b0 == 26:  # vvcurveto
                    var n = len(state.stack)
                    var i = 0
                    var dx1: Float32 = 0.0
                    if (n & 3) == 1 and n >= 1:
                        dx1 = state.stack[0]
                        i = 1
                    while i + 3 < n:
                        Self._t2_curveto(
                            state, out,
                            dx1, state.stack[i],
                            state.stack[i + 1], state.stack[i + 2],
                            0.0, state.stack[i + 3],
                        )
                        dx1 = 0.0
                        i += 4
                    state.stack.clear()
                elif b0 == 27:  # hhcurveto
                    var n = len(state.stack)
                    var i = 0
                    var dy1: Float32 = 0.0
                    if (n & 3) == 1 and n >= 1:
                        dy1 = state.stack[0]
                        i = 1
                    while i + 3 < n:
                        Self._t2_curveto(
                            state, out,
                            state.stack[i], dy1,
                            state.stack[i + 1], state.stack[i + 2],
                            state.stack[i + 3], 0.0,
                        )
                        dy1 = 0.0
                        i += 4
                    state.stack.clear()
                elif b0 == 30 or b0 == 31:  # vhcurveto / hvcurveto
                    var vert = (b0 == 30)
                    var n = len(state.stack)
                    var i = 0
                    while i + 3 < n:
                        var a = state.stack[i]
                        var b = state.stack[i + 1]
                        var c = state.stack[i + 2]
                        var d = state.stack[i + 3]
                        var extra: Float32 = 0.0
                        if i + 5 == n:
                            extra = state.stack[i + 4]
                            i += 5
                        else:
                            i += 4
                        if vert:
                            Self._t2_curveto(state, out, 0.0, a, b, c, d, extra)
                        else:
                            Self._t2_curveto(state, out, a, 0.0, b, c, extra, d)
                        vert = not vert
                    state.stack.clear()
                else:
                    state.stack.clear()

    def cff_outline(self, glyph: Int) -> GlyphOutline:
        var result = GlyphOutline()
        result.is_cubic = True
        if glyph < 0 or glyph >= len(self.cff_charstrings):
            return result^
        var fd = 0
        var use_fd = len(self.cff_fd_privates) > 0 and self.cff_fd_select_off >= 0
        if use_fd:
            fd = self.cff_fd_for_glyph(glyph)
            if fd < 0 or fd >= len(self.cff_fd_privates):
                fd = 0
        var priv = self.cff_fd_privates[fd].copy() if use_fd else self.cff_private.copy()
        var state = _T2State(priv.default_width)
        self._exec_t2_slice(self.cff_charstrings[glyph], priv, state, result, 0)
        Self._t2_close_contour(state, result)
        if self.has_hmtx:
            result.advance = Float32(self.default_advance(glyph))
            result.lsb = Float32(self.default_lsb(glyph))
        else:
            result.advance = state.width
            result.lsb = result.bounds()[0]
        return result^

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
        if self.is_cff:
            return self.cff_outline(glyph)
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

    # ---- GDEF glyph classification & lookup filtering ----

    def gdef_glyph_class(self, glyph: Int) -> Int:
        if self.gdef_offset < 0:
            return 0
        var off = self.u16(self.gdef_offset + 4)
        if off == 0:
            return 0
        return self.class_of(self.gdef_offset + off, glyph)

    def gdef_mark_attach_class(self, glyph: Int) -> Int:
        if self.gdef_offset < 0:
            return 0
        var off = self.u16(self.gdef_offset + 10)
        if off == 0:
            return 0
        return self.class_of(self.gdef_offset + off, glyph)

    def gdef_mark_glyph_set_contains(self, set_index: Int, glyph: Int) -> Bool:
        if self.gdef_offset < 0 or self.u16(self.gdef_offset + 2) < 2:
            return False
        var mgs_off = self.u16(self.gdef_offset + 12)
        if mgs_off == 0:
            return False
        var mgs = self.gdef_offset + mgs_off
        var count = self.u16(mgs + 2)
        if set_index < 0 or set_index >= count:
            return False
        var cov_off = self.u32(mgs + 4 + set_index * 4)
        if cov_off == 0:
            return False
        return self.coverage_index(mgs + cov_off, glyph) >= 0

    def should_ignore_glyph(self, glyph: Int, lookup_flag: Int, mark_filtering_set: Int = -1) -> Bool:
        if (lookup_flag & (LOOKUP_FLAG_IGNORE_BASE | LOOKUP_FLAG_IGNORE_LIGATURES | LOOKUP_FLAG_IGNORE_MARKS | LOOKUP_FLAG_USE_MARK_FILTERING_SET | LOOKUP_FLAG_MARK_ATTACHMENT_MASK)) == 0:
            return False
        var cls = self.gdef_glyph_class(glyph)
        if (lookup_flag & LOOKUP_FLAG_IGNORE_BASE) != 0 and cls == GDEF_CLASS_BASE:
            return True
        if (lookup_flag & LOOKUP_FLAG_IGNORE_LIGATURES) != 0 and cls == GDEF_CLASS_LIGATURE:
            return True
        if (lookup_flag & LOOKUP_FLAG_IGNORE_MARKS) != 0 and cls == GDEF_CLASS_MARK:
            return True
        if cls == GDEF_CLASS_MARK:
            if (lookup_flag & LOOKUP_FLAG_USE_MARK_FILTERING_SET) != 0 and mark_filtering_set >= 0:
                if not self.gdef_mark_glyph_set_contains(mark_filtering_set, glyph):
                    return True
            var mtype = (lookup_flag & LOOKUP_FLAG_MARK_ATTACHMENT_MASK) >> 8
            if mtype != 0 and self.gdef_mark_attach_class(glyph) != mtype:
                return True
        return False

    # ---- GSUB ----

    def has_gsub(self) -> Bool:
        return self.gsub_offset >= 0

    @staticmethod
    def _gsub_feature_mask(tag_val: Int) -> UInt32:
        if tag_val == tag_value("locl"):
            return FEAT_LOCL
        if tag_val == tag_value("ccmp"):
            return FEAT_CCMP
        if tag_val == tag_value("isol"):
            return FEAT_ISOL
        if tag_val == tag_value("init"):
            return FEAT_INIT
        if tag_val == tag_value("medi"):
            return FEAT_MEDI
        if tag_val == tag_value("fina"):
            return FEAT_FINA
        if tag_val == tag_value("rlig"):
            return FEAT_RLIG
        if tag_val == tag_value("liga"):
            return FEAT_LIGA
        if tag_val == tag_value("clig"):
            return FEAT_CLIG
        if tag_val == tag_value("calt"):
            return FEAT_CALT
        return 0

    def _find_gsub_script(self, script_list: Int, wanted_tag: Int) -> Int:
        var count = self.u16(script_list)
        for i in range(count):
            var rec = script_list + 2 + i * 6
            if self.u32(rec) == wanted_tag:
                return script_list + self.u16(rec + 4)
        return -1

    def find_gsub_lookups(self, script_tag: String, lang_tag: String = "") -> List[Tuple[Int, UInt32]]:
        var out = List[Tuple[Int, UInt32]]()
        if self.gsub_offset < 0:
            return out^
        var gsub = self.gsub_offset
        var script_list = gsub + self.u16(gsub + 4)
        var feature_list = gsub + self.u16(gsub + 6)

        var script_off = self._find_gsub_script(script_list, tag_value(script_tag))
        if script_off < 0:
            script_off = self._find_gsub_script(script_list, tag_value("DFLT"))
        if script_off < 0 and (script_tag == "latn" or script_tag == "DFLT"):
            script_off = self._find_gsub_script(script_list, tag_value("latn"))
        if script_off < 0:
            return out^

        var lang_off = -1
        var def_lang = self.u16(script_off)
        if lang_tag.byte_length() > 0:
            var ltag_val = tag_value(lang_tag)
            var lcount = self.u16(script_off + 2)
            for i in range(lcount):
                var rec = script_off + 4 + i * 6
                if self.u32(rec) == ltag_val:
                    lang_off = script_off + self.u16(rec + 4)
                    break
        if lang_off < 0 and def_lang != 0:
            lang_off = script_off + def_lang
        if lang_off < 0:
            return out^

        var feat_indices = List[Int]()
        var req_feat = self.u16(lang_off + 2)
        if req_feat != 0xFFFF:
            feat_indices.append(req_feat)
        var fcount = self.u16(lang_off + 4)
        for i in range(fcount):
            feat_indices.append(self.u16(lang_off + 6 + i * 2))

        var total_features = self.u16(feature_list)
        var lookup_masks = Dict[Int, UInt32]()
        var lookup_ids = List[Int]()
        for fidx in feat_indices:
            if fidx < 0 or fidx >= total_features:
                continue
            var rec = feature_list + 2 + fidx * 6
            var mask = Self._gsub_feature_mask(self.u32(rec))
            if mask == 0:
                continue
            var feat = feature_list + self.u16(rec + 4)
            var lcnt = self.u16(feat + 2)
            for k in range(lcnt):
                var lidx = self.u16(feat + 4 + k * 2)
                var prev = lookup_masks.get(lidx)
                if prev:
                    lookup_masks[lidx] = prev.value() | mask
                else:
                    lookup_masks[lidx] = mask
                    lookup_ids.append(lidx)
        sort_ints(lookup_ids)
        for lidx in lookup_ids:
            var m = lookup_masks.get(lidx)
            out.append((lidx, m.value() if m else UInt32(0)))
        return out^

    def apply_gsub(self, mut glyphs: List[GsubGlyph], script_tag: String, lang_tag: String = ""):
        if self.gsub_offset < 0 or len(glyphs) == 0:
            return
        var lookups = self.find_gsub_lookups(script_tag, lang_tag)
        if len(lookups) == 0:
            return
        var lookup_list = self.gsub_offset + self.u16(self.gsub_offset + 8)
        var lookup_count = self.u16(lookup_list)
        for entry in lookups:
            var lidx = entry[0]
            var fmask = entry[1]
            if lidx >= 0 and lidx < lookup_count:
                var l_off = lookup_list + self.u16(lookup_list + 2 + lidx * 2)
                self._apply_gsub_lookup(glyphs, l_off, fmask, lookup_list, lookup_count, 0)

    def _next_non_ignored(self, glyphs: List[GsubGlyph], start: Int, lflag: Int, mfs: Int) -> Int:
        var p = start
        while p < len(glyphs):
            if not self.should_ignore_glyph(glyphs[p].glyph_id, lflag, mfs):
                return p
            p += 1
        return -1

    def _prev_non_ignored(self, glyphs: List[GsubGlyph], start: Int, lflag: Int, mfs: Int) -> Int:
        var p = start
        while p >= 0:
            if not self.should_ignore_glyph(glyphs[p].glyph_id, lflag, mfs):
                return p
            p -= 1
        return -1

    def _apply_gsub_lookup(
        self,
        mut glyphs: List[GsubGlyph],
        lookup_off: Int,
        feat_mask: UInt32,
        lookup_list: Int,
        lookup_count: Int,
        depth: Int,
    ):
        if depth >= 8 or lookup_off < 0 or lookup_off + 6 > len(self.data):
            return
        var ltype = self.u16(lookup_off)
        var lflag = self.u16(lookup_off + 2)
        var sub_count = self.u16(lookup_off + 4)
        var mfs = self.u16(lookup_off + 6 + sub_count * 2) if (lflag & LOOKUP_FLAG_USE_MARK_FILTERING_SET) != 0 else -1
        var i = 0
        while i < len(glyphs):
            if (glyphs[i].mask & feat_mask) == 0 or self.should_ignore_glyph(glyphs[i].glyph_id, lflag, mfs):
                i += 1
                continue
            var matched = False
            var next_i = i + 1
            for s_idx in range(sub_count):
                var sub = lookup_off + self.u16(lookup_off + 6 + s_idx * 2)
                var sub_type = ltype
                if sub_type == 7:
                    if self.u16(sub) != 1:
                        continue
                    sub_type = self.u16(sub + 2)
                    sub = sub + self.u32(sub + 4)
                var res = self._apply_gsub_subtable(
                    glyphs, i, sub_type, sub, lflag, mfs, lookup_list, lookup_count, depth
                )
                if res[0]:
                    matched = True
                    next_i = res[1]
                    break
            if matched:
                i = max(0, next_i)
            else:
                i += 1

    def _apply_gsub_at(
        self,
        mut glyphs: List[GsubGlyph],
        pos: Int,
        lookup_idx: Int,
        lookup_list: Int,
        lookup_count: Int,
        depth: Int,
    ) -> Bool:
        if depth >= 8 or lookup_idx < 0 or lookup_idx >= lookup_count or pos < 0 or pos >= len(glyphs):
            return False
        var lookup_off = lookup_list + self.u16(lookup_list + 2 + lookup_idx * 2)
        var ltype = self.u16(lookup_off)
        var lflag = self.u16(lookup_off + 2)
        var sub_count = self.u16(lookup_off + 4)
        var mfs = self.u16(lookup_off + 6 + sub_count * 2) if (lflag & LOOKUP_FLAG_USE_MARK_FILTERING_SET) != 0 else -1
        if self.should_ignore_glyph(glyphs[pos].glyph_id, lflag, mfs):
            return False
        for s_idx in range(sub_count):
            var sub = lookup_off + self.u16(lookup_off + 6 + s_idx * 2)
            var sub_type = ltype
            if sub_type == 7:
                if self.u16(sub) != 1:
                    continue
                sub_type = self.u16(sub + 2)
                sub = sub + self.u32(sub + 4)
            var res = self._apply_gsub_subtable(
                glyphs, pos, sub_type, sub, lflag, mfs, lookup_list, lookup_count, depth + 1
            )
            if res[0]:
                return True
        return False

    def _apply_seq_lookups(
        self,
        mut glyphs: List[GsubGlyph],
        mut match_pos: List[Int],
        seq_records: List[GsubSequenceLookup],
        lookup_list: Int,
        lookup_count: Int,
        depth: Int,
    ):
        for rec in seq_records:
            if rec.sequence_index < 0 or rec.sequence_index >= len(match_pos):
                continue
            var p = match_pos[rec.sequence_index]
            var old_len = len(glyphs)
            if self._apply_gsub_at(glyphs, p, rec.lookup_index, lookup_list, lookup_count, depth):
                var delta = len(glyphs) - old_len
                if delta != 0:
                    for k in range(rec.sequence_index + 1, len(match_pos)):
                        match_pos[k] += delta

    def _read_seq_lookups(self, at: Int, count: Int) -> List[GsubSequenceLookup]:
        var recs = List[GsubSequenceLookup](capacity=count)
        for k in range(count):
            recs.append(GsubSequenceLookup(self.u16(at + k * 4), self.u16(at + k * 4 + 2)))
        return recs^

    def _apply_gsub_subtable(
        self,
        mut glyphs: List[GsubGlyph],
        i: Int,
        sub_type: Int,
        sub: Int,
        lflag: Int,
        mfs: Int,
        lookup_list: Int,
        lookup_count: Int,
        depth: Int,
    ) -> Tuple[Bool, Int]:
        if sub < 0 or sub + 4 > len(self.data) or i < 0 or i >= len(glyphs):
            return (False, i + 1)
        var fmt = self.u16(sub)
        var gid = glyphs[i].glyph_id

        if sub_type == 1:  # SingleSubst
            var cov_idx = self.coverage_index(sub + self.u16(sub + 2), gid)
            if cov_idx < 0:
                return (False, i + 1)
            if fmt == 1:
                glyphs[i].glyph_id = (gid + self.i16(sub + 4)) & 0xFFFF
                return (True, i + 1)
            if fmt == 2:
                var cnt = self.u16(sub + 4)
                if cov_idx < cnt:
                    glyphs[i].glyph_id = self.u16(sub + 6 + cov_idx * 2)
                    return (True, i + 1)
            return (False, i + 1)

        if sub_type == 2:  # MultipleSubst
            if fmt != 1:
                return (False, i + 1)
            var cov_idx = self.coverage_index(sub + self.u16(sub + 2), gid)
            var seq_cnt = self.u16(sub + 4)
            if cov_idx < 0 or cov_idx >= seq_cnt:
                return (False, i + 1)
            var seq = sub + self.u16(sub + 6 + cov_idx * 2)
            var gcnt = self.u16(seq)
            var orig = glyphs[i]
            var next_glyphs = List[GsubGlyph](capacity=len(glyphs) - 1 + gcnt)
            for k in range(i):
                next_glyphs.append(glyphs[k])
            for k in range(gcnt):
                next_glyphs.append(GsubGlyph(self.u16(seq + 2 + k * 2), orig.cluster, orig.mask, orig.alt_choice))
            for k in range(i + 1, len(glyphs)):
                next_glyphs.append(glyphs[k])
            glyphs = next_glyphs^
            return (True, i + gcnt)

        if sub_type == 3:  # AlternateSubst
            if fmt != 1:
                return (False, i + 1)
            var cov_idx = self.coverage_index(sub + self.u16(sub + 2), gid)
            var aset_cnt = self.u16(sub + 4)
            if cov_idx < 0 or cov_idx >= aset_cnt:
                return (False, i + 1)
            var aset = sub + self.u16(sub + 6 + cov_idx * 2)
            var acnt = self.u16(aset)
            if acnt <= 0:
                return (False, i + 1)
            var choice = max(0, min(acnt - 1, glyphs[i].alt_choice))
            glyphs[i].glyph_id = self.u16(aset + 2 + choice * 2)
            return (True, i + 1)

        if sub_type == 4:  # LigatureSubst
            if fmt != 1:
                return (False, i + 1)
            var cov_idx = self.coverage_index(sub + self.u16(sub + 2), gid)
            var lset_cnt = self.u16(sub + 4)
            if cov_idx < 0 or cov_idx >= lset_cnt:
                return (False, i + 1)
            var lset = sub + self.u16(sub + 6 + cov_idx * 2)
            var lig_cnt = self.u16(lset)
            for li in range(lig_cnt):
                var lig = lset + self.u16(lset + 2 + li * 2)
                var lig_gid = self.u16(lig)
                var comp_cnt = self.u16(lig + 2)
                if comp_cnt < 1:
                    continue
                var comp_pos = List[Int](capacity=comp_cnt - 1)
                var cur = i
                var ok = True
                for c in range(comp_cnt - 1):
                    var nxt = self._next_non_ignored(glyphs, cur + 1, lflag, mfs)
                    if nxt < 0 or glyphs[nxt].glyph_id != self.u16(lig + 4 + c * 2):
                        ok = False
                        break
                    comp_pos.append(nxt)
                    cur = nxt
                if ok:
                    glyphs[i].glyph_id = lig_gid
                    var k = len(comp_pos) - 1
                    while k >= 0:
                        _ = glyphs.pop(comp_pos[k])
                        k -= 1
                    return (True, i + 1)
            return (False, i + 1)

        if sub_type == 5:  # ContextualSubst
            if fmt == 1:
                var cov_idx = self.coverage_index(sub + self.u16(sub + 2), gid)
                var rset_cnt = self.u16(sub + 4)
                if cov_idx < 0 or cov_idx >= rset_cnt:
                    return (False, i + 1)
                var rset = sub + self.u16(sub + 6 + cov_idx * 2)
                var rcnt = self.u16(rset)
                for ri in range(rcnt):
                    var rule = rset + self.u16(rset + 2 + ri * 2)
                    var gcnt = self.u16(rule)
                    var scnt = self.u16(rule + 2)
                    var match_pos = List[Int](capacity=max(1, gcnt))
                    match_pos.append(i)
                    var cur = i
                    var ok = True
                    for k in range(gcnt - 1):
                        var nxt = self._next_non_ignored(glyphs, cur + 1, lflag, mfs)
                        if nxt < 0 or glyphs[nxt].glyph_id != self.u16(rule + 4 + k * 2):
                            ok = False
                            break
                        match_pos.append(nxt)
                        cur = nxt
                    if ok:
                        var recs = self._read_seq_lookups(rule + 4 + max(0, gcnt - 1) * 2, scnt)
                        self._apply_seq_lookups(glyphs, match_pos, recs, lookup_list, lookup_count, depth)
                        return (True, max(i + 1, match_pos[len(match_pos) - 1] + 1))
                return (False, i + 1)
            if fmt == 2:
                var cov_idx = self.coverage_index(sub + self.u16(sub + 2), gid)
                if cov_idx < 0:
                    return (False, i + 1)
                var cdef = sub + self.u16(sub + 4)
                var cset_cnt = self.u16(sub + 6)
                var cls0 = self.class_of(cdef, gid)
                if cls0 < 0 or cls0 >= cset_cnt:
                    return (False, i + 1)
                var cset_rel = self.u16(sub + 8 + cls0 * 2)
                if cset_rel == 0:
                    return (False, i + 1)
                var cset = sub + cset_rel
                var rcnt = self.u16(cset)
                for ri in range(rcnt):
                    var rule = cset + self.u16(cset + 2 + ri * 2)
                    var gcnt = self.u16(rule)
                    var scnt = self.u16(rule + 2)
                    var match_pos = List[Int](capacity=max(1, gcnt))
                    match_pos.append(i)
                    var cur = i
                    var ok = True
                    for k in range(gcnt - 1):
                        var nxt = self._next_non_ignored(glyphs, cur + 1, lflag, mfs)
                        if nxt < 0 or self.class_of(cdef, glyphs[nxt].glyph_id) != self.u16(rule + 4 + k * 2):
                            ok = False
                            break
                        match_pos.append(nxt)
                        cur = nxt
                    if ok:
                        var recs = self._read_seq_lookups(rule + 4 + max(0, gcnt - 1) * 2, scnt)
                        self._apply_seq_lookups(glyphs, match_pos, recs, lookup_list, lookup_count, depth)
                        return (True, max(i + 1, match_pos[len(match_pos) - 1] + 1))
                return (False, i + 1)
            if fmt == 3:
                var gcnt = self.u16(sub + 2)
                var scnt = self.u16(sub + 4)
                if gcnt <= 0:
                    return (False, i + 1)
                if self.coverage_index(sub + self.u16(sub + 6), gid) < 0:
                    return (False, i + 1)
                var match_pos = List[Int](capacity=gcnt)
                match_pos.append(i)
                var cur = i
                var ok = True
                for k in range(1, gcnt):
                    var nxt = self._next_non_ignored(glyphs, cur + 1, lflag, mfs)
                    if nxt < 0 or self.coverage_index(sub + self.u16(sub + 6 + k * 2), glyphs[nxt].glyph_id) < 0:
                        ok = False
                        break
                    match_pos.append(nxt)
                    cur = nxt
                if ok:
                    var recs = self._read_seq_lookups(sub + 6 + gcnt * 2, scnt)
                    self._apply_seq_lookups(glyphs, match_pos, recs, lookup_list, lookup_count, depth)
                    return (True, max(i + 1, match_pos[len(match_pos) - 1] + 1))
                return (False, i + 1)
            return (False, i + 1)

        if sub_type == 6:  # ChainingContextualSubst
            if fmt == 1:
                var cov_idx = self.coverage_index(sub + self.u16(sub + 2), gid)
                var rset_cnt = self.u16(sub + 4)
                if cov_idx < 0 or cov_idx >= rset_cnt:
                    return (False, i + 1)
                var rset = sub + self.u16(sub + 6 + cov_idx * 2)
                var rcnt = self.u16(rset)
                for ri in range(rcnt):
                    var rule = rset + self.u16(rset + 2 + ri * 2)
                    var nb = self.u16(rule)
                    var at = rule + 2
                    var ok = True
                    var bcur = i
                    for k in range(nb):
                        var prev = self._prev_non_ignored(glyphs, bcur - 1, lflag, mfs)
                        if prev < 0 or glyphs[prev].glyph_id != self.u16(at + k * 2):
                            ok = False
                            break
                        bcur = prev
                    at += nb * 2
                    var ni = self.u16(at)
                    at += 2
                    var match_pos = List[Int](capacity=max(1, ni))
                    match_pos.append(i)
                    var fcur = i
                    if ok:
                        for k in range(ni - 1):
                            var nxt = self._next_non_ignored(glyphs, fcur + 1, lflag, mfs)
                            if nxt < 0 or glyphs[nxt].glyph_id != self.u16(at + k * 2):
                                ok = False
                                break
                            match_pos.append(nxt)
                            fcur = nxt
                    at += max(0, ni - 1) * 2
                    var nl = self.u16(at)
                    at += 2
                    if ok:
                        var lcur = fcur
                        for k in range(nl):
                            var nxt = self._next_non_ignored(glyphs, lcur + 1, lflag, mfs)
                            if nxt < 0 or glyphs[nxt].glyph_id != self.u16(at + k * 2):
                                ok = False
                                break
                            lcur = nxt
                    at += nl * 2
                    if ok:
                        var scnt = self.u16(at)
                        var recs = self._read_seq_lookups(at + 2, scnt)
                        self._apply_seq_lookups(glyphs, match_pos, recs, lookup_list, lookup_count, depth)
                        return (True, max(i + 1, match_pos[len(match_pos) - 1] + 1))
                return (False, i + 1)
            if fmt == 2:
                var cov_idx = self.coverage_index(sub + self.u16(sub + 2), gid)
                if cov_idx < 0:
                    return (False, i + 1)
                var back_cdef = sub + self.u16(sub + 4)
                var in_cdef = sub + self.u16(sub + 6)
                var look_cdef = sub + self.u16(sub + 8)
                var cset_cnt = self.u16(sub + 10)
                var cls0 = self.class_of(in_cdef, gid)
                if cls0 < 0 or cls0 >= cset_cnt:
                    return (False, i + 1)
                var cset_rel = self.u16(sub + 12 + cls0 * 2)
                if cset_rel == 0:
                    return (False, i + 1)
                var cset = sub + cset_rel
                var rcnt = self.u16(cset)
                for ri in range(rcnt):
                    var rule = cset + self.u16(cset + 2 + ri * 2)
                    var nb = self.u16(rule)
                    var at = rule + 2
                    var ok = True
                    var bcur = i
                    for k in range(nb):
                        var prev = self._prev_non_ignored(glyphs, bcur - 1, lflag, mfs)
                        if prev < 0 or self.class_of(back_cdef, glyphs[prev].glyph_id) != self.u16(at + k * 2):
                            ok = False
                            break
                        bcur = prev
                    at += nb * 2
                    var ni = self.u16(at)
                    at += 2
                    var match_pos = List[Int](capacity=max(1, ni))
                    match_pos.append(i)
                    var fcur = i
                    if ok:
                        for k in range(ni - 1):
                            var nxt = self._next_non_ignored(glyphs, fcur + 1, lflag, mfs)
                            if nxt < 0 or self.class_of(in_cdef, glyphs[nxt].glyph_id) != self.u16(at + k * 2):
                                ok = False
                                break
                            match_pos.append(nxt)
                            fcur = nxt
                    at += max(0, ni - 1) * 2
                    var nl = self.u16(at)
                    at += 2
                    if ok:
                        var lcur = fcur
                        for k in range(nl):
                            var nxt = self._next_non_ignored(glyphs, lcur + 1, lflag, mfs)
                            if nxt < 0 or self.class_of(look_cdef, glyphs[nxt].glyph_id) != self.u16(at + k * 2):
                                ok = False
                                break
                            lcur = nxt
                    at += nl * 2
                    if ok:
                        var scnt = self.u16(at)
                        var recs = self._read_seq_lookups(at + 2, scnt)
                        self._apply_seq_lookups(glyphs, match_pos, recs, lookup_list, lookup_count, depth)
                        return (True, max(i + 1, match_pos[len(match_pos) - 1] + 1))
                return (False, i + 1)
            if fmt == 3:
                var nb = self.u16(sub + 2)
                var at = sub + 4
                var ok = True
                var bcur = i
                for k in range(nb):
                    var prev = self._prev_non_ignored(glyphs, bcur - 1, lflag, mfs)
                    if prev < 0 or self.coverage_index(sub + self.u16(at + k * 2), glyphs[prev].glyph_id) < 0:
                        ok = False
                        break
                    bcur = prev
                at += nb * 2
                var ni = self.u16(at)
                at += 2
                if ni <= 0:
                    return (False, i + 1)
                var match_pos = List[Int](capacity=ni)
                match_pos.append(i)
                var fcur = i
                if ok:
                    if self.coverage_index(sub + self.u16(at), gid) < 0:
                        ok = False
                    else:
                        for k in range(1, ni):
                            var nxt = self._next_non_ignored(glyphs, fcur + 1, lflag, mfs)
                            if nxt < 0 or self.coverage_index(sub + self.u16(at + k * 2), glyphs[nxt].glyph_id) < 0:
                                ok = False
                                break
                            match_pos.append(nxt)
                            fcur = nxt
                at += ni * 2
                var nl = self.u16(at)
                at += 2
                if ok:
                    var lcur = fcur
                    for k in range(nl):
                        var nxt = self._next_non_ignored(glyphs, lcur + 1, lflag, mfs)
                        if nxt < 0 or self.coverage_index(sub + self.u16(at + k * 2), glyphs[nxt].glyph_id) < 0:
                            ok = False
                            break
                        lcur = nxt
                at += nl * 2
                if ok:
                    var scnt = self.u16(at)
                    var recs = self._read_seq_lookups(at + 2, scnt)
                    self._apply_seq_lookups(glyphs, match_pos, recs, lookup_list, lookup_count, depth)
                    return (True, max(i + 1, match_pos[len(match_pos) - 1] + 1))
                return (False, i + 1)
        return (False, i + 1)


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
