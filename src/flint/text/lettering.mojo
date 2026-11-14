"""Lettering for the world (package 3.24): Vantor's "V", SILVER QUEEN on the
crates, SILVER QUEEN PUMP HOUSE on the pipe.

Each word is set from our own font (text/ttf.mojo: the outlines, advances
and pair kerning of the display face) and baked as coverage with the exact
distance baker of 3.12 (text/sdf_bake.mojo), one small picture a word: the
glyphs' distance cells laid along the baseline, the nearest outline winning
where two cells overlap, then the distance turned into a soft edge one
pixel wide. Nothing is painted by hand and the same font gives the same
bytes.

tools/bake_lettering.mojo writes the words to
assets/deepsong/fonts/lettering.dsl; render/story_props.mojo reads the file
and draws a word as a decal quad on a piece's side.

    file   "DSLT" u32 version u32 count, then for each word:
           u16 name bytes, the name (UTF-8), u16 width, u16 height,
           width x height coverage bytes, rows from the top
"""
from std.math import floor, ceil
from flint.text.ttf import TrueTypeFont
from flint.text.sdf_bake import bake_glyph, BakedGlyph

comptime LETTERING_VERSION: Int = 1
comptime LETTERING_PX: Int = 48        # the em, in pixels
comptime LETTERING_SPREAD: Int = 4
comptime LETTERING_MARGIN: Int = 3     # clear pixels round a word (a decal's edge never shows)
comptime LETTERING_FILE = "assets/deepsong/fonts/lettering.dsl"
comptime LETTERING_FONT = "assets/deepsong/fonts/LilitaOne-Regular.ttf"


struct Word(Copyable, Movable):
    var name: String
    var width: Int
    var height: Int
    var coverage: List[UInt8]

    def __init__(out self, name: String = String()):
        self.name = name
        self.width = 0
        self.height = 0
        self.coverage = List[UInt8]()

    def ink(self) -> Int:
        """Pixels more than half covered."""
        var n = 0
        for value in self.coverage:
            if value >= 128:
                n += 1
        return n

    def ink_box(self) -> Tuple[Int, Int, Int, Int]:
        """x0, y0, x1, y1 (half-open) of the pixels more than half covered."""
        var x0 = self.width
        var y0 = self.height
        var x1 = 0
        var y1 = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.coverage[y * self.width + x] >= 128:
                    x0 = min(x0, x)
                    y0 = min(y0, y)
                    x1 = max(x1, x + 1)
                    y1 = max(y1, y + 1)
        return (x0, y0, x1, y1)

    def argb(self) -> List[UInt32]:
        """White paint, the coverage as alpha: what the rasterizers draw."""
        var out = List[UInt32](length=self.width * self.height, fill=0)
        for i in range(self.width * self.height):
            out[i] = (UInt32(self.coverage[i]) << 24) | 0x00FFFFFF
        return out^


def lettering_words() -> List[String]:
    return ["V", "SILVER QUEEN", "SILVER QUEEN PUMP HOUSE"]


def bake_word(font: TrueTypeFont, text: String, px: Int = LETTERING_PX, spread: Int = LETTERING_SPREAD) -> Word:
    """One word as coverage. ASCII only (the three words are)."""
    var word = Word(text)
    var glyphs = List[BakedGlyph]()
    var pens = List[Float32]()
    var pen: Float32 = 0
    var scale = Float32(px) / Float32(font.units_per_em)
    var previous = 0
    var x0: Float32 = 1.0e9
    var y0: Float32 = 1.0e9
    var x1: Float32 = -1.0e9
    var y1: Float32 = -1.0e9
    for byte in text.as_bytes():
        var g = bake_glyph(font, Int(byte), px, spread)
        if previous != 0 and g.glyph != 0:
            pen += font.kerning(previous, g.glyph) * scale
        # Whole pixels, so every glyph's cell lands on the picture's grid.
        var at = floor(pen + 0.5)
        if g.w > 0:
            x0 = min(x0, at + g.xoff)
            y0 = min(y0, g.yoff)
            x1 = max(x1, at + g.xoff + Float32(g.w))
            y1 = max(y1, g.yoff + Float32(g.h))
        pens.append(at)
        pen += g.advance
        previous = g.glyph
        glyphs.append(g^)
    if x1 <= x0:
        return word^
    var left = Int(x0) - LETTERING_MARGIN + spread   # the cells already carry spread + 1 of padding
    var top = Int(y0) - LETTERING_MARGIN + spread
    word.width = Int(x1) + LETTERING_MARGIN - spread - left
    word.height = Int(y1) + LETTERING_MARGIN - spread - top
    var field = List[UInt8](length=word.width * word.height, fill=0)
    for k in range(len(glyphs)):
        ref g = glyphs[k]
        var gx = Int(pens[k] + g.xoff) - left
        var gy = Int(g.yoff) - top
        for j in range(g.h):
            var y = gy + j
            if y < 0 or y >= word.height:
                continue
            for i in range(g.w):
                var x = gx + i
                if x < 0 or x >= word.width:
                    continue
                var value = g.pixels[j * g.w + i]
                if value > field[y * word.width + x]:
                    field[y * word.width + x] = value
    # 128 is the outline, 127 / spread a pixel: full paint half a pixel inside.
    word.coverage = List[UInt8](length=word.width * word.height, fill=0)
    var per_pixel = Float32(127) / Float32(spread)
    for i in range(word.width * word.height):
        var d = (Float32(field[i]) - 128.0) / per_pixel
        var a = max(Float32(0), min(Float32(1), d + 0.5))
        word.coverage[i] = UInt8(Int(a * 255.0 + 0.5))
    return word^


def bake_lettering(font: TrueTypeFont) -> List[Word]:
    var out = List[Word]()
    for text in lettering_words():
        out.append(bake_word(font, text))
    return out^


def put16(mut out: List[UInt8], value: Int):
    out.append(UInt8(value & 0xFF))
    out.append(UInt8((value >> 8) & 0xFF))


def put32(mut out: List[UInt8], value: Int):
    put16(out, value & 0xFFFF)
    put16(out, (value >> 16) & 0xFFFF)


def lettering_bytes(words: List[Word]) -> List[UInt8]:
    var out = List[UInt8]()
    for byte in "DSLT".as_bytes():
        out.append(byte)
    put32(out, LETTERING_VERSION)
    put32(out, len(words))
    for word in words:
        var name = word.name.as_bytes()
        put16(out, len(name))
        for byte in name:
            out.append(byte)
        put16(out, word.width)
        put16(out, word.height)
        for value in word.coverage:
            out.append(value)
    return out^


def get16(data: List[UInt8], at: Int) -> Int:
    return Int(data[at]) | (Int(data[at + 1]) << 8)


def parse_lettering(data: List[UInt8]) -> List[Word]:
    """The words of a lettering file; none if it is not one (the game then
    draws no lettering)."""
    var out = List[Word]()
    if len(data) < 12 or data[0] != 0x44 or data[1] != 0x53 or data[2] != 0x4C or data[3] != 0x54:
        return out^
    if (get16(data, 4) | (get16(data, 6) << 16)) != LETTERING_VERSION:
        return out^
    var count = get16(data, 8) | (get16(data, 10) << 16)
    var at = 12
    for _ in range(count):
        if at + 2 > len(data):
            return List[Word]()
        var n = get16(data, at)
        at += 2
        if at + n + 4 > len(data):
            return List[Word]()
        var word = Word(String(unsafe_from_utf8=data[at:at + n]))
        at += n
        word.width = get16(data, at)
        word.height = get16(data, at + 2)
        at += 4
        var size = word.width * word.height
        if size <= 0 or size > 4096 * 256 or at + size > len(data):
            return List[Word]()
        word.coverage = List[UInt8](length=size, fill=0)
        for i in range(size):
            word.coverage[i] = data[at + i]
        at += size
        out.append(word^)
    return out^
