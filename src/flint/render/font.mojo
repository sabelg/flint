"""Signed-distance-field fonts for the HUD and menus.

`tools/bake_font.mojo` (package 3.12; it replaced `tools/bake_font.py`)
bakes a TrueType face with our own reader into a `.dsf` file: one 8-bit
distance atlas (edge at 128, 127 units = `spread` bake pixels), per-glyph
boxes and metrics, and kerning pairs. Drawing samples the atlas bilinearly
and turns distance into coverage, so one atlas serves every size and an
outline or a soft shadow costs nothing extra (Green, "Improved
Alpha-Tested Magnification for Vector Textures and Special Effects",
SIGGRAPH 2007).

Dynamic glyph cache (package 3.12): a font can keep its TrueType source
(`attach_source`). A character the baked atlas lacks is then baked from
the outline on first use (`ensure`) into a page of rows below the baked
atlas, packed in shelves. When the page is full it is flushed whole and
refilled (the next lines bake what they need again), so memory stays fixed
and the result never depends on what was drawn before. Measuring never
bakes: a missing glyph is measured from the source's advance.

This module only loads and measures; `Framebuffer` does the drawing.
"""
from std.collections import Dict
from flint.platform.os import read_bytes, write_bytes
from flint.text.utf8 import decode_utf8 as text_decode_utf8
from flint.text.ttf import TrueTypeFont
from flint.text.sdf_bake import bake_glyph, bake_font, dsf_bytes, ShelfPacker, invisible

comptime DSF_MAGIC: Int = 0x31465344
comptime GLYPH_TABLE: Int = 0x2700  # codepoints below this are looked up directly
comptime FALLBACK_CODEPOINT: Int = 63  # '?'
comptime DYNAMIC_ROWS: Int = 256       # atlas rows of the dynamic page


struct Glyph(Copyable, ImplicitlyCopyable, Movable):
    var x: Int
    var y: Int
    var w: Int
    var h: Int
    var xoff: Float32   # cell top-left relative to the pen on the baseline, bake px
    var yoff: Float32
    var advance: Float32

    def __init__(out self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.xoff = 0
        self.yoff = 0
        self.advance = 0


def read_i32(data: List[UInt8], offset: Int) -> Int:
    var value = Int(data[offset]) | (Int(data[offset + 1]) << 8) | (Int(data[offset + 2]) << 16) | (Int(data[offset + 3]) << 24)
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def decode_utf8(text: String) -> List[Int]:
    """Codepoints of a UTF-8 string; malformed bytes become U+FFFD
    (strict rules in text/utf8.mojo, package 3.12)."""
    return text_decode_utf8(text)


struct SdfFont(Copyable, Movable):
    var valid: Bool
    var atlas_width: Int
    var atlas_height: Int
    var bake_px: Float32
    var spread: Float32
    var ascent: Float32
    var descent: Float32
    var line_height: Float32
    var cap_height: Float32
    var pad: Float32
    var lookup: List[Int]        # codepoint -> glyph index, or -1
    var glyphs: List[Glyph]
    var kerning: Dict[Int, Float32]
    var atlas: List[UInt8]
    # Package 3.12: the dynamic glyph cache.
    var source: TrueTypeFont
    var has_source: Bool
    var baked_count: Int         # glyphs from the .dsf; later ones are dynamic
    var baked_height: Int        # atlas rows from the .dsf; the dynamic page follows
    var extra: Dict[Int, Int]    # codepoints at or above GLYPH_TABLE -> glyph index
    var dynamic_codes: List[Int]
    var packer: ShelfPacker
    var flushes: Int
    var dynamic_baked: Int

    def __init__(out self):
        self.valid = False
        self.atlas_width = 0
        self.atlas_height = 0
        self.bake_px = 48
        self.spread = 6
        self.ascent = 0
        self.descent = 0
        self.line_height = 0
        self.cap_height = 0
        self.pad = 7
        self.lookup = List[Int]()
        self.glyphs = List[Glyph]()
        self.kerning = Dict[Int, Float32]()
        self.atlas = List[UInt8]()
        self.source = TrueTypeFont()
        self.has_source = False
        self.baked_count = 0
        self.baked_height = 0
        self.extra = Dict[Int, Int]()
        self.dynamic_codes = List[Int]()
        self.packer = ShelfPacker(0, 0)
        self.flushes = 0
        self.dynamic_baked = 0

    def load(mut self, path: String) -> Bool:
        return self.load_bytes(read_bytes(path))

    def load_bytes(mut self, data: List[UInt8]) -> Bool:
        self.valid = False
        if len(data) < 44 or read_i32(data, 0) != DSF_MAGIC or read_i32(data, 4) != 1:
            return False
        self.atlas_width = read_i32(data, 8)
        self.atlas_height = read_i32(data, 12)
        self.bake_px = Float32(read_i32(data, 16))
        self.spread = Float32(read_i32(data, 20))
        self.pad = self.spread + 1
        self.ascent = Float32(read_i32(data, 24)) / 64.0
        self.descent = Float32(read_i32(data, 28)) / 64.0
        self.line_height = Float32(read_i32(data, 32)) / 64.0
        var glyph_count = read_i32(data, 36)
        var kern_count = read_i32(data, 40)
        var atlas_start = 44 + glyph_count * 32 + kern_count * 12
        if glyph_count <= 0 or kern_count < 0 or self.atlas_width <= 0 or self.atlas_height <= 0:
            return False
        if len(data) != atlas_start + self.atlas_width * self.atlas_height:
            return False
        self.lookup = List[Int](length=GLYPH_TABLE, fill=-1)
        self.glyphs = List[Glyph]()
        var offset = 44
        for _ in range(glyph_count):
            var code = read_i32(data, offset)
            var glyph = Glyph()
            glyph.x = read_i32(data, offset + 4)
            glyph.y = read_i32(data, offset + 8)
            glyph.w = read_i32(data, offset + 12)
            glyph.h = read_i32(data, offset + 16)
            glyph.xoff = Float32(read_i32(data, offset + 20)) / 64.0
            glyph.yoff = Float32(read_i32(data, offset + 24)) / 64.0
            glyph.advance = Float32(read_i32(data, offset + 28)) / 64.0
            if glyph.x < 0 or glyph.y < 0 or glyph.x + glyph.w > self.atlas_width or glyph.y + glyph.h > self.atlas_height:
                return False
            if code >= 0 and code < GLYPH_TABLE:
                self.lookup[code] = len(self.glyphs)
            self.glyphs.append(glyph)
            offset += 32
        self.kerning = Dict[Int, Float32]()
        for _ in range(kern_count):
            var key = read_i32(data, offset) * 65536 + read_i32(data, offset + 4)
            self.kerning[key] = Float32(read_i32(data, offset + 8)) / 64.0
            offset += 12
        self.atlas = List[UInt8](length=self.atlas_width * self.atlas_height, fill=0)
        for index in range(len(self.atlas)):
            self.atlas[index] = data[atlas_start + index]
        # Cap height from 'H': the top of its ink is the cell top plus the padding.
        var h_index = self.glyph_index(72)
        if h_index >= 0:
            self.cap_height = -(self.glyphs[h_index].yoff + self.pad)
        else:
            self.cap_height = self.ascent * 0.7
        self.valid = True
        return True

    def load_ttf(
        mut self,
        path: String,
        bake_px: Int = 44,
        spread: Int = 6,
        weight: Float32 = 0,
        cache_dsf: String = "",
    ) -> Bool:
        """Load a TrueType (.ttf) font directly, baking an anti-aliased SDF atlas
        for ASCII 32..126 (with optional .dsf cache) and attaching the outline source
        for on-demand Unicode glyphs."""
        if cache_dsf.byte_length() > 0 and self.load(cache_dsf):
            _ = self.attach_source(path, weight)
            return True
        var font = TrueTypeFont()
        if not font.load(path):
            return False
        if weight > 0:
            _ = font.set_variation("wght", weight)
        var codes = List[Int]()
        for c in range(32, 127):
            codes.append(c)
        try:
            var baked = bake_font(font, codes, bake_px, spread, 512)
            var bytes = dsf_bytes(baked)
            if cache_dsf.byte_length() > 0:
                _ = write_bytes(cache_dsf, bytes)
            if not self.load_bytes(bytes):
                return False
        except:
            return False
        self.source = font^
        self.has_source = True
        self.baked_count = len(self.glyphs)
        self.baked_height = self.atlas_height
        self.atlas_height += DYNAMIC_ROWS
        self.atlas.resize(self.atlas_width * self.atlas_height, 0)
        self.packer = ShelfPacker(self.atlas_width, self.atlas_height, self.baked_height + 1)
        return True

    def glyph_index(self, code: Int) -> Int:
        if code >= 0 and code < GLYPH_TABLE:
            return self.lookup[code]
        var found = self.extra.get(code)
        if found:
            return found.value()
        return -1

    def glyph_for(self, code: Int) -> Int:
        var index = self.glyph_index(code)
        if index < 0:
            index = self.glyph_index(FALLBACK_CODEPOINT)
        return index

    def kern(self, left: Int, right: Int) -> Float32:
        var key = left * 65536 + right
        var found = self.kerning.get(key)
        if found:
            return found.value()
        if self.has_source and (self.is_dynamic(left) or self.is_dynamic(right)):
            # A pair with a dynamic glyph: ask the font itself.
            var a = self.source.glyph_id(left)
            var b = self.source.glyph_id(right)
            if a != 0 and b != 0:
                return self.source.kerning(a, b) * self.bake_px / Float32(self.source.units_per_em)
        return 0

    def is_dynamic(self, code: Int) -> Bool:
        var index = self.glyph_index(code)
        return index < 0 or index >= self.baked_count

    def advance_of(self, code: Int) -> Float32:
        """Advance in bake px: the baked glyph's, else the source's (a glyph
        not baked yet), else the fallback's."""
        var index = self.glyph_index(code)
        if index >= 0:
            return self.glyphs[index].advance
        if self.has_source:
            if invisible(code):
                return 0
            var id = self.source.glyph_id(code)
            if id != 0:
                return self.source.advance(id) * self.bake_px / Float32(self.source.units_per_em)
        var fallback = self.glyph_index(FALLBACK_CODEPOINT)
        return self.glyphs[fallback].advance if fallback >= 0 else Float32(0)

    def measure(self, text: String, px: Float32) -> Float32:
        """Advance width of one line at `px` pixels per em."""
        if not self.valid:
            return 0
        var scale = px / self.bake_px
        var codes = decode_utf8(text)
        var width: Float32 = 0
        for index in range(len(codes)):
            width += self.advance_of(codes[index])
            if index + 1 < len(codes):
                width += self.kern(codes[index], codes[index + 1])
        return width * scale

    # ---- the dynamic glyph cache ----

    def attach_source(mut self, path: String, weight: Float32 = 0) -> Bool:
        """Keep the font file this atlas was baked from (same bake size and
        weight) so missing characters can be baked on first use, and add the
        dynamic page below the baked atlas."""
        if not self.valid:
            return False
        var font = TrueTypeFont()
        if not font.load(path):
            return False
        if weight > 0:
            _ = font.set_variation("wght", weight)
        self.source = font^
        self.has_source = True
        self.baked_count = len(self.glyphs)
        self.baked_height = self.atlas_height
        self.atlas_height += DYNAMIC_ROWS
        self.atlas.resize(self.atlas_width * self.atlas_height, 0)
        self.packer = ShelfPacker(self.atlas_width, self.atlas_height, self.baked_height + 1)
        return True

    def ensure(mut self, code: Int) -> Bool:
        """Make `code` drawable: True if it is (or is now) in the atlas."""
        if self.glyph_index(code) >= 0:
            return True
        if not self.has_source or code < 0:
            return False
        if not invisible(code) and not self.source.has_glyph(code):
            return False
        var baked = bake_glyph(self.source, code, Int(self.bake_px), Int(self.spread))
        var spot = (0, 0)
        if baked.w > 0:
            spot = self.packer.place(baked.w, baked.h)
            if spot[0] < 0:
                self.flush()
                spot = self.packer.place(baked.w, baked.h)
                if spot[0] < 0:
                    return False
            for j in range(baked.h):
                for i in range(baked.w):
                    self.atlas[(spot[1] + j) * self.atlas_width + spot[0] + i] = baked.pixels[j * baked.w + i]
        var glyph = Glyph()
        glyph.x = spot[0]
        glyph.y = spot[1]
        glyph.w = baked.w
        glyph.h = baked.h
        glyph.xoff = baked.xoff
        glyph.yoff = baked.yoff
        glyph.advance = baked.advance
        self.glyphs.append(glyph)
        if code < GLYPH_TABLE:
            self.lookup[code] = len(self.glyphs) - 1
        else:
            self.extra[code] = len(self.glyphs) - 1
        self.dynamic_codes.append(code)
        self.dynamic_baked += 1
        return True

    def ensure_text(mut self, text: String):
        if not self.has_source:
            return
        for code in decode_utf8(text):
            if code >= 0 and code < GLYPH_TABLE and self.lookup[code] >= 0:
                continue
            _ = self.ensure(code)

    def flush(mut self):
        """Empty the dynamic page: forget its glyphs and clear its rows."""
        for code in self.dynamic_codes:
            if code < GLYPH_TABLE:
                self.lookup[code] = -1
            else:
                _ = self.extra.pop(code, -1)
        self.dynamic_codes.clear()
        self.glyphs.resize(self.baked_count, Glyph())
        for index in range(self.baked_height * self.atlas_width, len(self.atlas)):
            self.atlas[index] = 0
        self.packer.reset()
        self.flushes += 1

    def sample(self, x: Float32, y: Float32) -> Float32:
        """Bilinear distance at atlas coordinates (texel centres at +0.5)."""
        var fx = x - 0.5
        var fy = y - 0.5
        var x0 = Int(fx) if fx >= 0 else Int(fx) - 1
        var y0 = Int(fy) if fy >= 0 else Int(fy) - 1
        var tx = fx - Float32(x0)
        var ty = fy - Float32(y0)
        var a = self.texel(x0, y0)
        var b = self.texel(x0 + 1, y0)
        var c = self.texel(x0, y0 + 1)
        var d = self.texel(x0 + 1, y0 + 1)
        var top = a + (b - a) * tx
        var bottom = c + (d - c) * tx
        return top + (bottom - top) * ty

    def texel(self, x: Int, y: Int) -> Float32:
        if x < 0 or y < 0 or x >= self.atlas_width or y >= self.atlas_height:
            return 0
        return Float32(self.atlas[y * self.atlas_width + x])


# Original 5x7 bitmap glyphs, packed in five-bit rows from top to bottom.
def bitmap_glyph(code: Int) -> UInt64:
    if code >= 97 and code <= 122:
        return bitmap_glyph(code - 32)
    match code:
        case 48: return 0x3A33AE62E
        case 49: return 0x3884210C4
        case 50: return 0x7C444422E
        case 51: return 0x3E107420F
        case 52: return 0x211F4A988
        case 53: return 0x3E107843F
        case 54: return 0x3A317842E
        case 55: return 0x08422221F
        case 56: return 0x3A317462E
        case 57: return 0x3A10F462E
        case 65: return 0x4631FC62E
        case 66: return 0x3E317C62F
        case 67: return 0x3A210862E
        case 68: return 0x3E318C62F
        case 69: return 0x7C217843F
        case 70: return 0x04217843F
        case 71: return 0x7A31E862E
        case 72: return 0x4631FC631
        case 73: return 0x38842108E
        case 74: return 0x19294211C
        case 75: return 0x452519531
        case 76: return 0x7C2108421
        case 77: return 0x4631AD771
        case 78: return 0x4631CD671
        case 79: return 0x3A318C62E
        case 80: return 0x04217C62F
        case 81: return 0x59358C62E
        case 82: return 0x45257C62F
        case 83: return 0x3E107043E
        case 84: return 0x10842109F
        case 85: return 0x3A318C631
        case 86: return 0x11518C631
        case 87: return 0x2AB5AC631
        case 88: return 0x462A22A31
        case 89: return 0x108422A31
        case 90: return 0x7C222221F
        case 46: return 0x318000000
        case 44: return 0x088C00000
        case 58: return 0x018C03180
        case 59: return 0x088C03180
        case 45: return 0x0000F8000
        case 95: return 0x7C0000000
        case 47: return 0x044222110
        case 92: return 0x410820841
        case 124: return 0x108421084
        case 33: return 0x100421084
        case 63: return 0x10044422E
        case 40: return 0x208210888
        case 41: return 0x088842082
        case 43: return 0x0084F9080
        case 61: return 0x001F07C00
        case 37: return 0x674222173
        case 38: return 0x593511526
        case 42: return 0x009575480
        case 62: return 0x044441041
        case 60: return 0x410411110
        case 35: return 0x15F55F54A
        case 36: return 0x11F4717C4
        case 39: return 0x000001084
        case 91: return 0x38421084E
        case 93: return 0x39084210E
        case _: return 0x1FFFFFF
