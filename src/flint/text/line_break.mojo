"""Line breaking (package 3.12).

A simplified Unicode line-breaking algorithm (UAX #14) that covers what the
game's English and Spanish strings need and keeps Chinese, Japanese and
Korean cheap to add later (decision 9):

- a mandatory break after '\\n' (the newline stays on the line it ends);
- a break opportunity after a run of spaces; trailing spaces hang past the
  edge and are not counted in a line's width;
- a break after a hyphen, an en dash or an em dash;
- never a break before closing punctuation (`. , ; : ! ? ) ] } » … ’ ”` and
  the CJK closers `、 。 」 』 ） ！ ？`) or after opening punctuation
  (`( [ { « ¿ ¡ ‘ “` and `「 『 （`), so "¡Hola!" stays whole;
- U+00A0 (no-break space) and U+2060 (word joiner) glue their neighbours;
- between two CJK characters (kana, ideographs, Hangul syllables) a break
  is allowed anywhere, as CJK text has no spaces;
- a word wider than the whole line is broken by character.

Lines tile the text: every codepoint belongs to exactly one line, in order.
Rejected: the full UAX #14 pair table (40+ classes, about 1,800 table
entries) is more than two Latin languages need; it can replace
`break_allowed` without changing callers.
"""
from std.math import isfinite
from flint.render.font import SdfFont
from flint.text.utf8 import decode_utf8
from flint.text.bidi import (
    BIDI_AUTO,
    resolve_bidi,
    reorder_line,
    compute_arabic_forms,
)
from flint.text.shaper import (
    FontFallbackChain,
    TextShaper,
    ShapedGlyph,
    ShapedLine,
    is_bidi_format_code,
)


@fieldwise_init
struct TextLine(Copyable, ImplicitlyCopyable, Movable):
    var start: Int      # first codepoint index
    var end: Int        # one past the last (includes hanging spaces and '\n')
    var width: Float32  # without hanging spaces, in the units of the advances


def is_space(code: Int) -> Bool:
    return code == 0x20 or code == 0x09 or code == 0x0D or code == 0x3000


def is_bidi_control(code: Int) -> Bool:
    return is_bidi_format_code(code)


def is_glue(code: Int) -> Bool:
    return code == 0xA0 or code == 0x2060 or code == 0x202F


def is_closing(code: Int) -> Bool:
    if code == 0x2E or code == 0x2C or code == 0x3B or code == 0x3A or code == 0x21 or code == 0x3F:
        return True
    if code == 0x29 or code == 0x5D or code == 0x7D or code == 0xBB or code == 0x2026 or code == 0x2019 or code == 0x201D:
        return True
    if code == 0x060C or code == 0x061B or code == 0x061F:
        return True
    return code == 0x3001 or code == 0x3002 or code == 0x300D or code == 0x300F or code == 0xFF09 or code == 0xFF01 or code == 0xFF1F


def is_opening(code: Int) -> Bool:
    if code == 0x28 or code == 0x5B or code == 0x7B or code == 0xAB or code == 0xBF or code == 0xA1 or code == 0x2018 or code == 0x201C:
        return True
    return code == 0x300C or code == 0x300E or code == 0xFF08


def is_dash(code: Int) -> Bool:
    return code == 0x2D or code == 0x2010 or code == 0x2013 or code == 0x2014


def is_cjk(code: Int) -> Bool:
    return (code >= 0x3040 and code <= 0x30FF) or (code >= 0x3400 and code <= 0x9FFF) or (code >= 0xAC00 and code <= 0xD7AF) or (code >= 0xF900 and code <= 0xFAFF)


def break_allowed(before: Int, after: Int) -> Bool:
    """May a line end between `before` and `after`? ('\\n' is handled by
    the caller as a mandatory break.)"""
    if is_glue(before) or is_glue(after):
        return False
    if is_space(after):
        return False          # spaces stay with the word before them
    if is_closing(after) or is_opening(before):
        return False
    if is_space(before):
        return True
    if is_dash(before):
        return not is_dash(after)
    if is_cjk(before) or is_cjk(after):
        return True
    return False


def run_width(font: SdfFont, codes: List[Int], start: Int, end: Int, px: Float32) -> Float32:
    """Advance width of `codes[start:end]` at `px` per em, kerned inside the run."""
    if not font.valid or end <= start or not isfinite(px) or px <= Float32(0.0) or not isfinite(font.bake_px) or font.bake_px <= Float32(0.0):
        return 0
    var scale = px / font.bake_px
    if not isfinite(scale):
        return 0
    var width: Float32 = 0
    for index in range(start, end):
        width += font.advance_of(codes[index])   # unbaked glyphs from the source (dynamic cache)
        if index + 1 < end:
            width += font.kern(codes[index], codes[index + 1])
    var total = width * scale
    return total if isfinite(total) else Float32(0)


def item_widths(font: SdfFont, codes: List[Int], px: Float32, mut advance: List[Float32], mut kern_before: List[Float32]):
    """Per-codepoint advance and the kerning with the codepoint before it."""
    var scale = Float32(0)
    if font.valid and isfinite(px) and px > Float32(0.0) and isfinite(font.bake_px) and font.bake_px > Float32(0.0):
        var s = px / font.bake_px
        if isfinite(s):
            scale = s
    for index in range(len(codes)):
        var a: Float32 = 0
        var k: Float32 = 0
        if scale != Float32(0) and font.valid and codes[index] != 0x0A and codes[index] != 0x0D and not is_bidi_control(codes[index]):
            var raw_a = font.advance_of(codes[index]) * scale
            if isfinite(raw_a):
                a = raw_a
            if index > 0 and not is_bidi_control(codes[index - 1]):
                var raw_k = font.kern(codes[index - 1], codes[index]) * scale
                if isfinite(raw_k):
                    k = raw_k
        advance.append(a)
        kern_before.append(k)


@always_inline
def _safe_add_f32_lb(a: Float32, b: Float32) -> Float32:
    var sa = a if isfinite(a) else Float32(0.0)
    var sb = b if isfinite(b) else Float32(0.0)
    var s = sa + sb
    if isfinite(s):
        return s
    return Float32(3.402823e38) if s > Float32(0.0) else Float32(-3.402823e38)


def break_items(codes: List[Int], advance: List[Float32], kern_before: List[Float32], max_width: Float32) -> List[TextLine]:
    """Greedy breaking of items with known widths. `kern_before[i]` is
    dropped when item i starts a line. `max_width` <= 0 means no limit."""
    var lines = List[TextLine]()
    var count = len(codes)
    var start = 0
    while start < count:
        var width: Float32 = 0      # up to and including the last item
        var visible: Float32 = 0    # without trailing spaces
        var last_break = -1         # index where the next line may start
        var last_break_visible: Float32 = 0
        var index = start
        var ended = False
        while index < count:
            var code = codes[index]
            if code == 0x0A:
                lines.append(TextLine(start, index + 1, visible))
                start = index + 1
                ended = True
                break
            if index > start and break_allowed(codes[index - 1], code):
                last_break = index
                last_break_visible = visible
            var step = _safe_add_f32_lb(advance[index], (kern_before[index] if index > start else Float32(0)))
            var next_width = _safe_add_f32_lb(width, step)
            if not is_space(code) and max_width > 0 and next_width > max_width and index > start:
                if last_break > start:
                    lines.append(TextLine(start, last_break, last_break_visible))
                    start = last_break
                else:
                    # One word longer than the line: break it by character.
                    lines.append(TextLine(start, index, visible))
                    start = index
                ended = True
                break
            width = next_width
            if not is_space(code):
                visible = width
            index += 1
        if not ended:
            lines.append(TextLine(start, count, visible))
            start = count
    return lines^


def wrap(font: SdfFont, codes: List[Int], px: Float32, max_width: Float32) -> List[TextLine]:
    """Lines of `codes` at `px` per em no wider than `max_width` (same units
    as px; <= 0 for no limit), using the font's advances and kerning."""
    var advance = List[Float32]()
    var kern_before = List[Float32]()
    item_widths(font, codes, px, advance, kern_before)
    return break_items(codes, advance, kern_before, max_width)


def wrap_shaped(
    chain: FontFallbackChain,
    text: String,
    px: Float32,
    max_width: Float32,
    base_dir: Int = BIDI_AUTO,
    lang_tag: String = "",
) -> List[ShapedLine]:
    """Break `text` into lines using GSUB/kern-shaped cluster widths, then
    visually reorder each wrapped line via UAX #9 L1/L2 and shape runs."""
    var out = List[ShapedLine]()
    var codes = decode_utf8(text)
    var n = len(codes)
    if n == 0:
        return out^

    var resolved = resolve_bidi(codes, base_dir)
    var arabic_forms = compute_arabic_forms(codes)
    var shaper = TextShaper(chain.copy(), lang_tag)

    var adv = List[Float32](length=n, fill=0.0)
    var kern_b = List[Float32](length=n, fill=0.0)
    var safe_px = px if (isfinite(px) and px > Float32(0.0)) else Float32(0.0)
    var safe_max_w = max_width if (isfinite(max_width) and max_width > Float32(0.0)) else Float32(0.0)
    var idx = 0
    while idx < n:
        var lvl = resolved.levels[idx]
        var s = idx
        while idx < n and resolved.levels[idx] == lvl:
            idx += 1
        var sruns = shaper.shape_run_slice(codes, arabic_forms, s, idx, lvl, safe_px)
        for sr in sruns:
            for g in sr.glyphs:
                if g.cluster >= 0 and g.cluster < n:
                    adv[g.cluster] = _safe_add_f32_lb(adv[g.cluster], g.x_advance)

    var lines = break_items(codes, adv, kern_b, safe_max_w)
    for tl in lines:
        var bidi_runs = reorder_line(resolved, tl.start, tl.end)
        var sline = ShapedLine(tl.start, tl.end, tl.width)
        var line_w: Float32 = 0.0
        for br in bidi_runs:
            var sruns = shaper.shape_run_slice(codes, arabic_forms, br.start, br.end, br.level, safe_px)
            for sr in sruns:
                for g in sr.glyphs:
                    sline.glyphs.append(g)
                    line_w = _safe_add_f32_lb(line_w, g.x_advance)
        # Keep break_items visible width unless zero (e.g., when shaped width differs from trailing spaces)
        if sline.width == 0.0 and line_w > 0.0:
            sline.width = line_w
        out.append(sline^)
    return out^
