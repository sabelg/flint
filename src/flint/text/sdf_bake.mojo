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
from std.math import sqrt, floor, ceil
from flint.text.ttf import TrueTypeFont, GlyphOutline

comptime DSF_MAGIC: Int = 0x31465344  # "DSF1"
comptime CURVE_TOLERANCE: Float32 = 0.05   # flattening error, bake px


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


def flatten(outline: GlyphOutline, scale: Float32) -> Segments:
    """Contours to segments in bake px (y flipped down). Two off-curve
    points in a row imply the on-curve point between them."""
    var segments = Segments()
    var start = 0
    for contour in range(len(outline.ends)):
        var end = outline.ends[contour]
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
        # Start at an on-curve point (or the midpoint of two off ones).
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


def bake_glyph(font: TrueTypeFont, code: Int, bake_px: Int, spread: Int) -> BakedGlyph:
    """One glyph's distance cell, metrics in bake px. A glyph with no ink
    (a space) has an empty cell and only its advance."""
    var baked = BakedGlyph()
    baked.code = code
    baked.glyph = font.glyph_id(code)
    var scale = Float32(bake_px) / Float32(font.units_per_em)
    if invisible(code):
        return baked^      # a soft hyphen: no ink, no advance (shown only at a break)
    if baked.glyph == 0 and code == 0xA0:
        baked.glyph = font.glyph_id(32)   # a no-break space looks like a space
    var outline = font.outline(baked.glyph)
    baked.advance = outline.advance * scale
    if outline.point_count() == 0 or len(outline.ends) == 0:
        return baked^
    var segments = flatten(outline, scale)
    if segments.count() == 0:
        return baked^
    var box = outline.bounds()
    var pad = spread + 1
    var gx0 = Int(floor(box[0] * scale)) - pad
    var gy0 = Int(floor(-box[3] * scale)) - pad
    var gx1 = Int(ceil(box[2] * scale)) + pad
    var gy1 = Int(ceil(-box[1] * scale)) + pad
    baked.w = gx1 - gx0
    baked.h = gy1 - gy0
    baked.xoff = Float32(gx0)
    baked.yoff = Float32(gy0)
    baked.pixels = List[UInt8](length=baked.w * baked.h, fill=0)
    var to_units = Float32(127) / Float32(spread)
    for j in range(baked.h):
        var py = Float32(gy0 + j) + 0.5
        for i in range(baked.w):
            var px = Float32(gx0 + i) + 0.5
            var d = signed_distance(segments, px, py)
            var value = 128.0 + d * to_units
            baked.pixels[j * baked.w + i] = UInt8(max(0, min(255, Int(floor(value + 0.5)))))
    return baked^


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
        self.ascent = 0
        self.descent = 0
        self.glyphs = List[BakedGlyph]()
        self.kern_left = List[Int]()
        self.kern_right = List[Int]()
        self.kern_amount = List[Float32]()
        self.atlas = List[UInt8]()


def bake_font(font: TrueTypeFont, codes: List[Int], bake_px: Int, spread: Int, atlas_width: Int = 1024) raises -> BakedFont:
    """Bake every listed codepoint the font has into one atlas (cells
    shelf-packed tallest first, height rounded up to a power of two), with
    the kerning pairs between them of a quarter pixel or more."""
    var baked = BakedFont()
    baked.bake_px = bake_px
    baked.spread = spread
    var scale = Float32(bake_px) / Float32(font.units_per_em)
    baked.ascent = ceil(Float32(font.ascender) * scale)
    baked.descent = ceil(Float32(-font.descender) * scale)
    for code in codes:
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
    baked.atlas = List[UInt8](length=atlas_width * height, fill=0)
    for glyph in baked.glyphs:
        for j in range(glyph.h):
            for i in range(glyph.w):
                baked.atlas[(glyph.y + j) * atlas_width + glyph.x + i] = glyph.pixels[j * glyph.w + i]
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
    11 int32 header (magic, version 1, atlas w, h, bake px, spread,
    ascent*64, descent*64, line height*64, glyph count, kern count), then
    int32[8] per glyph (code, x, y, w, h, xoff*64, yoff*64, advance*64),
    int32[3] per kerning pair (left, right, amount*64), then the atlas."""
    var out = List[UInt8]()
    put_i32(out, DSF_MAGIC)
    put_i32(out, 1)
    put_i32(out, baked.atlas_width)
    put_i32(out, baked.atlas_height)
    put_i32(out, baked.bake_px)
    put_i32(out, baked.spread)
    put_i32(out, fixed64(baked.ascent))
    put_i32(out, fixed64(baked.descent))
    put_i32(out, fixed64(baked.ascent + baked.descent))
    put_i32(out, len(baked.glyphs))
    put_i32(out, len(baked.kern_left))
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
