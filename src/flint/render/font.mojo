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
from flint.text.sdf_bake import (
    bake_glyph,
    bake_glyph_id,
    bake_font,
    dsf_bytes,
    msdf_median,
    ShelfPacker,
    invisible,
)
from flint.text.bidi import BIDI_AUTO
from flint.text.shaper import ShapedGlyph, FontFallbackChain, TextShaper

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
    var channels: Int
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
    var dirty_y0: Int
    var dirty_y1: Int
    var atlas_version: UInt64
    # M4b F26: fallback fonts & glyph-ID cache for GSUB/shaped glyphs
    var fallbacks: List[TrueTypeFont]
    var fallback_scales: List[Float32]
    var gid_extra: Dict[Int, Int]
    var dynamic_gids: List[Int]
    var baked_cp_map: Dict[Int, Int]

    def __init__(out self):
        self.valid = False
        self.atlas_width = 0
        self.atlas_height = 0
        self.channels = 1
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
        self.dirty_y0 = 0
        self.dirty_y1 = 0
        self.atlas_version = 0
        self.fallbacks = List[TrueTypeFont]()
        self.fallback_scales = List[Float32]()
        self.gid_extra = Dict[Int, Int]()
        self.dynamic_gids = List[Int]()
        self.baked_cp_map = Dict[Int, Int]()

    def load(mut self, path: String) -> Bool:
        return self.load_bytes(read_bytes(path))

    def load_bytes(mut self, data: List[UInt8]) -> Bool:
        self.valid = False
        if len(data) < 44 or read_i32(data, 0) != DSF_MAGIC:
            return False
        var version = read_i32(data, 4)
        if version != 1 and version != 2:
            return False
        var hdr_bytes = 48 if version == 2 else 44
        if len(data) < hdr_bytes:
            return False
        var atlas_w = read_i32(data, 8)
        var atlas_h = read_i32(data, 12)
        var bake_px_i = read_i32(data, 16)
        var spread_i = read_i32(data, 20)
        var ascent_i = read_i32(data, 24)
        var descent_i = read_i32(data, 28)
        var line_height_i = read_i32(data, 32)
        var glyph_count = read_i32(data, 36)
        var kern_count = read_i32(data, 40)
        var ch = read_i32(data, 44) if version == 2 else 1
        if atlas_w <= 0 or atlas_h <= 0 or (ch != 1 and ch != 3 and ch != 4):
            return False
        if glyph_count <= 0 or kern_count < 0 or bake_px_i <= 0 or spread_i <= 0:
            return False
        var bake_px_f = Float32(bake_px_i)
        var spread_f = Float32(spread_i)
        if not (bake_px_f > Float32(0.0)) or not (spread_f > Float32(0.0)):
            return False

        # Overflow-free bounds checks against `len(data)`
        var rem = len(data) - hdr_bytes
        if rem < 0:
            return False
        if glyph_count > rem // 32:
            return False
        rem -= glyph_count * 32
        if kern_count > rem // 12:
            return False
        rem -= kern_count * 12
        if atlas_w > rem // ch:
            return False
        var row_bytes = atlas_w * ch
        if atlas_h > rem // row_bytes:
            return False
        if atlas_h * row_bytes != rem:
            return False

        self.atlas_width = atlas_w
        self.atlas_height = atlas_h
        self.bake_px = bake_px_f
        self.spread = spread_f
        self.pad = self.spread + 1
        self.ascent = Float32(ascent_i) / 64.0
        self.descent = Float32(descent_i) / 64.0
        self.line_height = Float32(line_height_i) / 64.0
        self.channels = ch
        var atlas_start = hdr_bytes + glyph_count * 32 + kern_count * 12

        self.lookup = List[Int](length=GLYPH_TABLE, fill=-1)
        self.glyphs = List[Glyph]()
        self.extra = Dict[Int, Int]()
        self.dynamic_codes = List[Int]()
        self.gid_extra = Dict[Int, Int]()
        self.dynamic_gids = List[Int]()
        self.baked_cp_map = Dict[Int, Int]()
        var offset = hdr_bytes
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
            if (
                glyph.x < 0
                or glyph.y < 0
                or glyph.w < 0
                or glyph.h < 0
                or glyph.x + glyph.w > self.atlas_width
                or glyph.y + glyph.h > self.atlas_height
            ):
                return False
            var g_idx = len(self.glyphs)
            if code >= 0 and code < GLYPH_TABLE:
                self.lookup[code] = g_idx
                self.baked_cp_map[code] = g_idx + 1
            elif code >= GLYPH_TABLE:
                self.extra[code] = g_idx
                self.baked_cp_map[code] = g_idx + 1
            self.glyphs.append(glyph)
            offset += 32
        self.kerning = Dict[Int, Float32]()
        for _ in range(kern_count):
            var key = read_i32(data, offset) * 65536 + read_i32(data, offset + 4)
            self.kerning[key] = Float32(read_i32(data, offset + 8)) / 64.0
            offset += 12
        var atlas_bytes = self.atlas_width * self.atlas_height * self.channels
        self.atlas = List[UInt8](length=atlas_bytes, fill=0)
        for index in range(atlas_bytes):
            self.atlas[index] = data[atlas_start + index]
        # Cap height from 'H': the top of its ink is the cell top plus the padding.
        var h_index = self.glyph_index(72)
        if h_index >= 0:
            self.cap_height = -(self.glyphs[h_index].yoff + self.pad)
        else:
            self.cap_height = self.ascent * 0.7
        self.baked_count = len(self.glyphs)
        self.baked_height = 0
        self.dirty_y0 = 0
        self.dirty_y1 = self.atlas_height
        self.atlas_version = 1
        self.valid = True
        return True

    def load_ttf(
        mut self,
        path: String,
        bake_px: Int = 44,
        spread: Int = 6,
        weight: Float32 = 0,
        cache_dsf: String = "",
        channels: Int = 1,
    ) -> Bool:
        """Load a TrueType/OpenType (.ttf/.otf) font directly, baking an SDF/MSDF/MTSDF
        atlas for ASCII 32..126 (with optional .dsf cache) and attaching the outline
        source for on-demand Unicode glyphs."""
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
            var baked = bake_font(font, codes, bake_px, spread, 512, channels)
            var bytes = dsf_bytes(baked)
            if cache_dsf.byte_length() > 0:
                _ = write_bytes(cache_dsf, bytes)
            if not self.load_bytes(bytes):
                return False
        except:
            return False
        self.source = font^
        self.has_source = True
        self._ensure_dynamic_page()
        return True

    def load_otf(
        mut self,
        path: String,
        bake_px: Int = 44,
        spread: Int = 6,
        weight: Float32 = 0,
        cache_dsf: String = "",
        channels: Int = 1,
    ) -> Bool:
        return self.load_ttf(path, bake_px, spread, weight, cache_dsf, channels)

    def load_msdf(
        mut self,
        path: String,
        bake_px: Int = 44,
        spread: Int = 6,
        weight: Float32 = 0,
        cache_dsf: String = "",
    ) -> Bool:
        return self.load_ttf(path, bake_px, spread, weight, cache_dsf, 3)

    def load_mtsdf(
        mut self,
        path: String,
        bake_px: Int = 44,
        spread: Int = 6,
        weight: Float32 = 0,
        cache_dsf: String = "",
    ) -> Bool:
        return self.load_ttf(path, bake_px, spread, weight, cache_dsf, 4)

    def _ensure_dynamic_page(mut self):
        if not self.valid or self.baked_height > 0 or self.atlas_width <= 0:
            return
        var ch = max(1, self.channels)
        self.baked_count = len(self.glyphs)
        self.baked_height = self.atlas_height
        self.atlas_height += DYNAMIC_ROWS
        self.atlas.resize(self.atlas_width * self.atlas_height * ch, 0)
        self.packer = ShelfPacker(self.atlas_width, self.atlas_height, self.baked_height + 1)
        self.dirty_y0 = 0
        self.dirty_y1 = self.atlas_height
        self.atlas_version += 1

    def glyph_index(self, code: Int) -> Int:
        if code >= 0 and code < len(self.lookup):
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

    def resolve_codepoint(self, code: Int) -> Tuple[Int, Int]:
        if code < 0:
            return (0, 0)
        if self.has_source and self.source.valid:
            var gid = self.source.glyph_id(code)
            if gid != 0:
                return (0, gid)
        elif self.valid:
            var idx = self.glyph_index(code)
            if idx >= 0 and idx < self.baked_count:
                return (0, idx + 1)
        for i in range(len(self.fallbacks)):
            if self.fallbacks[i].valid:
                var gid = self.fallbacks[i].glyph_id(code)
                if gid != 0:
                    return (i + 1, gid)
        return (0, 0)

    def add_fallback_font(mut self, var font: TrueTypeFont, match_ascent: Bool = False) -> Bool:
        if not font.valid:
            return False
        var mscale: Float32 = 1.0
        if match_ascent:
            var p_asc: Float32 = 0.0
            if self.has_source and self.source.valid:
                p_asc = Float32(max(1, self.source.ascender)) / Float32(max(1, self.source.units_per_em))
            elif self.valid and self.bake_px > Float32(0.0):
                p_asc = self.ascent / self.bake_px
            if p_asc > Float32(0.0):
                var f_asc = Float32(max(1, font.ascender)) / Float32(max(1, font.units_per_em))
                if f_asc > Float32(1.0e-5):
                    mscale = p_asc / f_asc
        self.fallbacks.append(font^)
        self.fallback_scales.append(mscale)
        if self.valid:
            self._ensure_dynamic_page()
        return True

    def add_fallback_bytes(mut self, var data: List[UInt8], weight: Float32 = 0, match_ascent: Bool = False) -> Bool:
        var font = TrueTypeFont()
        if not font.load_bytes(data^):
            return False
        if weight > 0:
            _ = font.set_variation("wght", weight)
        return self.add_fallback_font(font^, match_ascent)

    def add_fallback(mut self, path: String, weight: Float32 = 0, match_ascent: Bool = False) -> Bool:
        return self.add_fallback_bytes(read_bytes(path), weight, match_ascent)

    def clear_fallbacks(mut self):
        self.fallbacks.clear()
        self.fallback_scales.clear()
        if self.valid and self.baked_height > 0:
            self.flush()

    def fallback_chain(self) -> FontFallbackChain:
        var chain = FontFallbackChain()
        if self.has_source:
            chain.set_primary(self.source.copy())
        else:
            chain.set_primary(TrueTypeFont())
            if self.valid:
                var advs = List[Float32](capacity=self.baked_count)
                for i in range(self.baked_count):
                    advs.append(self.glyphs[i].advance)
                chain.set_baked_primary(
                    self.bake_px,
                    self.baked_cp_map.copy(),
                    advs^,
                    self.kerning.copy(),
                )
        for i in range(len(self.fallbacks)):
            chain.add_fallback_scaled(self.fallbacks[i].copy(), self.fallback_scales[i])
        return chain^

    def kern(self, left: Int, right: Int) -> Float32:
        var key = left * 65536 + right
        var found = self.kerning.get(key)
        if found:
            return found.value()
        if (self.has_source or len(self.fallbacks) > 0) and (self.is_dynamic(left) or self.is_dynamic(right)):
            var ra = self.resolve_codepoint(left)
            var rb = self.resolve_codepoint(right)
            if ra[1] != 0 and rb[1] != 0 and ra[0] == rb[0]:
                var fi = ra[0]
                if fi == 0 and self.has_source:
                    return self.source.kerning(ra[1], rb[1]) * self.bake_px / Float32(max(1, self.source.units_per_em))
                if fi > 0 and fi - 1 < len(self.fallbacks):
                    var fb_idx = fi - 1
                    return self.fallbacks[fb_idx].kerning(ra[1], rb[1]) * self.bake_px * self.fallback_scales[fb_idx] / Float32(max(1, self.fallbacks[fb_idx].units_per_em))
        return 0

    def is_dynamic(self, code: Int) -> Bool:
        var index = self.glyph_index(code)
        return index < 0 or index >= self.baked_count

    def advance_of(self, code: Int) -> Float32:
        """Advance in bake px: the baked glyph's, else the source/fallback's,
        else the '?' fallback."""
        var index = self.glyph_index(code)
        if index >= 0:
            return self.glyphs[index].advance
        if self.has_source or len(self.fallbacks) > 0:
            if invisible(code):
                return 0
            var res = self.resolve_codepoint(code)
            if res[1] != 0:
                if res[0] == 0 and self.has_source:
                    return self.source.advance(res[1]) * self.bake_px / Float32(max(1, self.source.units_per_em))
                if res[0] > 0 and res[0] - 1 < len(self.fallbacks):
                    var fb_idx = res[0] - 1
                    return self.fallbacks[fb_idx].advance(res[1]) * self.bake_px * self.fallback_scales[fb_idx] / Float32(max(1, self.fallbacks[fb_idx].units_per_em))
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
        self._ensure_dynamic_page()
        return True

    def ensure_glyph_id(mut self, glyph_id: Int, font_index: Int = 0, code: Int = 0) -> Int:
        """Bake and cache a specific `(font_index, glyph_id)` (including GSUB
        ligatures/contextual forms and fallback font glyphs) into the dynamic page."""
        if glyph_id < 0 or not self.valid:
            return -1
        var gid_key = (font_index << 20) | (glyph_id & 0xFFFFF)
        var cached = self.gid_extra.get(gid_key)
        if cached:
            return cached.value()
        if code > 0:
            var existing_cp = self.glyph_index(code)
            if existing_cp >= 0:
                var rc = self.resolve_codepoint(code)
                if rc[0] == font_index and rc[1] == glyph_id:
                    self.gid_extra[gid_key] = existing_cp
                    return existing_cp
        if font_index == 0 and not self.has_source:
            if code > 0:
                var existing = self.glyph_index(code)
                if existing >= 0 and existing < self.baked_count:
                    return existing
            if glyph_id > 0 and glyph_id - 1 < self.baked_count:
                return glyph_id - 1
            return self.glyph_for(code) if code > 0 else -1
        var mscale: Float32 = 1.0
        var has_font = False
        if font_index == 0 and self.has_source and self.source.valid:
            has_font = True
        elif font_index > 0 and font_index - 1 < len(self.fallbacks) and self.fallbacks[font_index - 1].valid:
            has_font = True
            mscale = self.fallback_scales[font_index - 1]
        if not has_font:
            return -1

        self._ensure_dynamic_page()
        var ch = max(1, self.channels)
        var baked = bake_glyph_id(self.source, glyph_id, Int(self.bake_px), Int(self.spread), code, mscale, ch) if font_index == 0 else bake_glyph_id(self.fallbacks[font_index - 1], glyph_id, Int(self.bake_px), Int(self.spread), code, mscale, ch)
        var spot = (0, 0)
        if baked.w > 0:
            spot = self.packer.place(baked.w, baked.h)
            if spot[0] < 0:
                self.flush()
                spot = self.packer.place(baked.w, baked.h)
                if spot[0] < 0:
                    return -1
            var g_ch = max(1, baked.channels)
            for j in range(baked.h):
                for i in range(baked.w):
                    var dst_base = ((spot[1] + j) * self.atlas_width + spot[0] + i) * ch
                    var src_base = (j * baked.w + i) * g_ch
                    for c in range(ch):
                        self.atlas[dst_base + c] = baked.pixels[src_base + min(c, g_ch - 1)]
        var glyph = Glyph()
        glyph.x = spot[0]
        glyph.y = spot[1]
        glyph.w = baked.w
        glyph.h = baked.h
        glyph.xoff = baked.xoff
        glyph.yoff = baked.yoff
        glyph.advance = baked.advance
        self.glyphs.append(glyph)
        var g_idx = len(self.glyphs) - 1
        self.gid_extra[gid_key] = g_idx
        self.dynamic_gids.append(gid_key)
        if code > 0 and self.glyph_index(code) < 0:
            var rc = self.resolve_codepoint(code)
            if rc[0] == font_index and rc[1] == glyph_id:
                if code < GLYPH_TABLE:
                    self.lookup[code] = g_idx
                else:
                    self.extra[code] = g_idx
                self.dynamic_codes.append(code)
        self.dynamic_baked += 1
        if spot[0] >= 0 and baked.w > 0:
            self.dirty_y0 = min(self.dirty_y0, spot[1])
            self.dirty_y1 = max(self.dirty_y1, spot[1] + baked.h)
            self.atlas_version += 1
        return g_idx

    def ensure(mut self, code: Int) -> Bool:
        """Make `code` drawable: True if it is (or is now) in the atlas."""
        if self.glyph_index(code) >= 0:
            return True
        if code < 0 or not self.valid:
            return False
        if not self.has_source and len(self.fallbacks) == 0:
            return False
        if invisible(code):
            var fi = 0 if (self.has_source and self.source.valid) else 1
            return self.ensure_glyph_id(0, fi, code) >= 0
        var res = self.resolve_codepoint(code)
        if res[1] == 0:
            return False
        return self.ensure_glyph_id(res[1], res[0], code) >= 0

    def ensure_text(mut self, text: String):
        if not self.has_source and len(self.fallbacks) == 0:
            return
        for code in decode_utf8(text):
            if code >= 0 and code < GLYPH_TABLE and self.lookup[code] >= 0:
                continue
            _ = self.ensure(code)

    def shape_text(
        self,
        text: String,
        px: Float32,
        base_dir: Int = BIDI_AUTO,
        alt_choice: Int = 0,
    ) -> List[ShapedGlyph]:
        if not self.valid and not self.has_source and len(self.fallbacks) == 0:
            return List[ShapedGlyph]()
        var shaper = TextShaper(self.fallback_chain(), "", alt_choice)
        return shaper.shape_text(text, px, base_dir)

    def shape_and_ensure(
        mut self,
        text: String,
        px: Float32,
        base_dir: Int = BIDI_AUTO,
        alt_choice: Int = 0,
    ) -> List[Tuple[ShapedGlyph, Int]]:
        var out = List[Tuple[ShapedGlyph, Int]]()
        var shaped = self.shape_text(text, px, base_dir, alt_choice)
        for sg in shaped:
            var idx = self.ensure_glyph_id(sg.glyph_id, sg.font_index, sg.codepoint)
            if idx < 0:
                idx = self.glyph_for(sg.codepoint)
            out.append((sg, idx))
        return out^

    def measure_shaped(
        self,
        text: String,
        px: Float32,
        base_dir: Int = BIDI_AUTO,
    ) -> Float32:
        if not self.valid:
            return 0.0
        if not self.has_source and len(self.fallbacks) == 0:
            return self.measure(text, px)
        var shaped = self.shape_text(text, px, base_dir)
        var total: Float32 = 0.0
        for sg in shaped:
            total += sg.x_advance
        return total

    def flush(mut self):
        """Empty the dynamic page: forget its glyphs and clear its rows."""
        for code in self.dynamic_codes:
            if code < GLYPH_TABLE:
                self.lookup[code] = -1
            else:
                _ = self.extra.pop(code, -1)
        self.dynamic_codes.clear()
        for gid_key in self.dynamic_gids:
            _ = self.gid_extra.pop(gid_key, -1)
        self.dynamic_gids.clear()
        self.glyphs.resize(self.baked_count, Glyph())
        var ch = max(1, self.channels)
        for index in range(self.baked_height * self.atlas_width * ch, len(self.atlas)):
            self.atlas[index] = 0
        self.packer.reset()
        self.flushes += 1
        self.dirty_y0 = min(self.dirty_y0, self.baked_height)
        self.dirty_y1 = max(self.dirty_y1, self.atlas_height)
        self.atlas_version += 1

    def is_dirty(self) -> Bool:
        return self.dirty_y1 > self.dirty_y0 and self.atlas_width > 0

    def mark_clean(mut self):
        self.dirty_y0 = self.atlas_height
        self.dirty_y1 = 0

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

    def sample_channel(self, x: Float32, y: Float32, channel: Int) -> Float32:
        var fx = x - 0.5
        var fy = y - 0.5
        var x0 = Int(fx) if fx >= 0 else Int(fx) - 1
        var y0 = Int(fy) if fy >= 0 else Int(fy) - 1
        var tx = fx - Float32(x0)
        var ty = fy - Float32(y0)
        var a = self.texel(x0, y0, channel)
        var b = self.texel(x0 + 1, y0, channel)
        var c = self.texel(x0, y0 + 1, channel)
        var d = self.texel(x0 + 1, y0 + 1, channel)
        var top = a + (b - a) * tx
        var bottom = c + (d - c) * tx
        return top + (bottom - top) * ty

    def sample_msdf(self, x: Float32, y: Float32) -> Float32:
        """Bilinear RGB sample followed by Chlumsky median(r, g, b)."""
        if self.channels < 3:
            return self.sample(x, y)
        var r = self.sample_channel(x, y, 0)
        var g = self.sample_channel(x, y, 1)
        var b = self.sample_channel(x, y, 2)
        return msdf_median(r, g, b)

    def sample_true_distance(self, x: Float32, y: Float32) -> Float32:
        """Bilinear true signed distance (alpha channel on MTSDF, else MSDF/SDF)."""
        if self.channels >= 4:
            return self.sample_channel(x, y, 3)
        if self.channels == 3:
            return self.sample_msdf(x, y)
        return self.sample(x, y)

    def texel(self, x: Int, y: Int, channel: Int = 0) -> Float32:
        if x < 0 or y < 0 or x >= self.atlas_width or y >= self.atlas_height:
            return 0
        var ch = max(1, self.channels)
        var c = max(0, min(ch - 1, channel))
        return Float32(self.atlas[(y * self.atlas_width + x) * ch + c])


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
        case 34: return 0x00000014A
        case 64: return 0x3A17ADEDE
        case 91: return 0x38421084E
        case 93: return 0x39084210E
        case 94: return 0x000004544
        case 96: return 0x000000082
        case 123: return 0x30841908C
        case 125: return 0x1884C1086
        case 126: return 0x00006D800
        case _: return 0x1FFFFFF
