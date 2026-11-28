"""Rich text: tags in UI strings, and their layout on lines (package 3.12).

Tags (ASCII, case-sensitive):

    [c=RRGGBB] ... [/c]   colour (hex), nests
    [d] ... [/d]          the display face (Lilita One); otherwise the UI face
    [s=150] ... [/s]      size in percent of the line's size (25-400), nests
    [btn=jump]            an inline button glyph for an action
                          (ui/button_glyphs.mojo), following the last device
    [[                    a literal '['

Anything else in brackets (an unknown tag, a bad colour, an unknown action,
a closing tag with nothing open, a bracket never closed) stays as literal
text, so a typo shows on screen instead of eating the line. Strings come
from tables translators edit; failing loudly but harmlessly is the point.

Layout places spans on lines with text/line_break.mojo: tags never split a
word (the breaker sees one stream of codepoints across spans), kerning
applies only inside a span, and a button is one unbreakable box that
behaves like a letter.
"""
from std.collections import Dict
from std.math import isfinite
from flint.render.font import SdfFont
from flint.text.utf8 import decode_utf8, encode_utf8
from flint.text.line_break import TextLine, break_items
from flint.text.bidi import (
    BIDI_AUTO,
    resolve_bidi,
    reorder_line,
    bidi_mirror,
    compute_arabic_forms,
)
from flint.text.shaper import ShapedGlyph, ShapedRun, TextShaper, FontFallbackChain
from flint.ui.button_glyphs import ACTION_NONE, action_id, glyph_shape, glyph_label, glyph_width, SHAPE_KEYCAP

comptime SPAN_TEXT: Int = 0
comptime SPAN_BUTTON: Int = 1
comptime OBJECT_CODE: Int = 0xFFFC   # stands in for a button in the breaker
comptime BUTTON_HEIGHT: Float32 = 1.15   # em
comptime BUTTON_LABEL: Float32 = 0.74    # label size, em of the line: 9 px capitals at 1280x800 (package 3.12 legibility rule)


@fieldwise_init
struct RichSpan(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var text: String
    var color: Int      # 0xRRGGBB, or -1 for the line's colour
    var face: Int       # -1 the line's face, 0 UI, 1 display
    var scale: Int      # percent of the line's size
    var action: Int     # for SPAN_BUTTON


struct PlacedRun(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var text: String
    var x: Float32          # from the layout's left edge
    var line: Int
    var baseline: Float32   # from the layout's top
    var width: Float32
    var px: Float32         # this run's em
    var color: Int
    var face: Int           # resolved: 0 UI, 1 display
    var action: Int
    var button_label: String
    var level: Int
    var rtl: Bool

    def __init__(
        out self,
        kind: Int,
        text: String,
        x: Float32,
        line: Int,
        baseline: Float32,
        width: Float32,
        px: Float32,
        color: Int,
        face: Int,
        action: Int,
        level: Int = 0,
        rtl: Bool = False,
        button_label: String = "",
    ):
        self.kind = kind
        self.text = text
        self.x = x
        self.line = line
        self.baseline = baseline
        self.width = width
        self.px = px
        self.color = color
        self.face = face
        self.action = action
        self.button_label = button_label
        self.level = level
        self.rtl = rtl


struct RichLayout(Copyable, Movable):
    var runs: List[PlacedRun]
    var lines: Int
    var width: Float32
    var height: Float32
    var base_level: Int

    def __init__(out self):
        self.runs = List[PlacedRun]()
        self.lines = 0
        self.width = 0
        self.height = 0
        self.base_level = 0


def hex_value(bytes: List[UInt8]) -> Int:
    """0xRRGGBB from six hex digits, or -1."""
    if len(bytes) != 6:
        return -1
    var value = 0
    for byte in bytes:
        var b = Int(byte)
        var digit: Int
        if b >= 48 and b <= 57:
            digit = b - 48
        elif b >= 97 and b <= 102:
            digit = b - 87
        elif b >= 65 and b <= 70:
            digit = b - 55
        else:
            return -1
        value = value * 16 + digit
    return value


def decimal_value(bytes: List[UInt8]) -> Int:
    if len(bytes) == 0 or len(bytes) > 4:
        return -1
    var value = 0
    for byte in bytes:
        var b = Int(byte)
        if b < 48 or b > 57:
            return -1
        value = value * 10 + b - 48
    return value


def starts_with(bytes: List[UInt8], prefix: String) -> Bool:
    var p = prefix.as_bytes()
    if len(bytes) < len(p):
        return False
    for index in range(len(p)):
        if bytes[index] != p[index]:
            return False
    return True


def tail(bytes: List[UInt8], start: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for index in range(start, len(bytes)):
        out.append(bytes[index])
    return out^


def same(bytes: List[UInt8], word: String) -> Bool:
    return len(bytes) == word.byte_length() and starts_with(bytes, word)


struct RichParser(Movable):
    var spans: List[RichSpan]
    var pending: List[UInt8]
    var colors: List[Int]
    var scales: List[Int]
    var display_depth: Int

    def __init__(out self):
        self.spans = List[RichSpan]()
        self.pending = List[UInt8]()
        self.colors = List[Int]()
        self.scales = List[Int]()
        self.display_depth = 0

    def color(self) -> Int:
        return self.colors[len(self.colors) - 1] if len(self.colors) > 0 else -1

    def scale(self) -> Int:
        return self.scales[len(self.scales) - 1] if len(self.scales) > 0 else 100

    def face(self) -> Int:
        return 1 if self.display_depth > 0 else -1

    def flush(mut self):
        if len(self.pending) == 0:
            return
        self.spans.append(RichSpan(SPAN_TEXT, String(unsafe_from_utf8=self.pending), self.color(), self.face(), self.scale(), ACTION_NONE))
        self.pending = List[UInt8]()

    def apply(mut self, tag: List[UInt8], prompts: Dict[String, String]) -> Bool:
        """Apply one tag's body (between the brackets); False if it is not a tag."""
        if starts_with(tag, "c="):
            var value = hex_value(tail(tag, 2))
            if value < 0:
                return False
            self.flush()
            self.colors.append(value)
            return True
        if same(tag, "/c"):
            if len(self.colors) == 0:
                return False
            self.flush()
            _ = self.colors.pop()
            return True
        if same(tag, "d"):
            self.flush()
            self.display_depth += 1
            return True
        if same(tag, "/d"):
            if self.display_depth == 0:
                return False
            self.flush()
            self.display_depth -= 1
            return True
        if starts_with(tag, "s="):
            var value = decimal_value(tail(tag, 2))
            if value < 25 or value > 400:
                return False
            self.flush()
            self.scales.append(value)
            return True
        if same(tag, "/s"):
            if len(self.scales) == 0:
                return False
            self.flush()
            _ = self.scales.pop()
            return True
        if starts_with(tag, "btn="):
            var name = String(unsafe_from_utf8=tail(tag, 4))
            var action = action_id(name)
            var prompt = prompts.get(name)
            if action == ACTION_NONE and not prompt:
                return False
            self.flush()
            self.spans.append(RichSpan(SPAN_BUTTON, prompt.value() if prompt else String(), self.color(), self.face(), self.scale(), action))
            return True
        return False


def parse_rich(text: String, prompts: Dict[String, String] = Dict[String, String]()) -> List[RichSpan]:
    """Spans of a tagged string (see the module notes)."""
    var parser = RichParser()
    var bytes = text.as_bytes()
    var count = len(bytes)
    var index = 0
    while index < count:
        var byte = bytes[index]
        if byte != 91:  # '['
            parser.pending.append(byte)
            index += 1
            continue
        if index + 1 < count and bytes[index + 1] == 91:
            parser.pending.append(91)
            index += 2
            continue
        var close = -1
        for probe in range(index + 1, count):
            if bytes[probe] == 93:  # ']'
                close = probe
                break
            if bytes[probe] == 91:
                break
        if close < 0:
            parser.pending.append(byte)
            index += 1
            continue
        var body = List[UInt8]()
        for k in range(index + 1, close):
            body.append(bytes[k])
        if parser.apply(body, prompts):
            index = close + 1
        else:
            parser.pending.append(byte)
            index += 1
    parser.flush()
    var spans = parser.spans.copy()
    return spans^


def plain_text(spans: List[RichSpan]) -> String:
    """The spans' text without tags (buttons left out)."""
    var out = String()
    for span in spans:
        if span.kind == SPAN_TEXT:
            out += span.text
    return out^


def face_font_metric(ui: SdfFont, display: SdfFont, face: Int) -> Tuple[Float32, Float32, Float32]:
    """(line height, cap height, bake px) of a face."""
    if face == 1:
        return (display.line_height, display.cap_height, display.bake_px)
    return (ui.line_height, ui.cap_height, ui.bake_px)


@always_inline
def _safe_add_f32(a: Float32, b: Float32) -> Float32:
    var sa = a if isfinite(a) else Float32(0.0)
    var sb = b if isfinite(b) else Float32(0.0)
    var s = sa + sb
    if isfinite(s):
        return s
    return Float32(3.402823e38) if s > Float32(0.0) else Float32(-3.402823e38)


def _safe_font_scale(px: Float32, bake_px: Float32) -> Float32:
    if not isfinite(px) or not isfinite(bake_px) or bake_px <= Float32(0.0):
        return Float32(0.0)
    var scale = px / bake_px
    if not isfinite(scale):
        return Float32(0.0)
    return scale


def _safe_span_px(px: Float32, scale_pct: Int) -> Float32:
    if not isfinite(px):
        return Float32(0.0)
    var span_px = px * Float32(scale_pct) / Float32(100.0)
    if not isfinite(span_px):
        return Float32(0.0)
    return span_px


def _visual_run_text(codes: List[Int], start: Int, end: Int, rtl: Bool) -> String:
    if not rtl:
        return encode_utf8(codes, start, end)
    var rev = List[Int](capacity=end - start)
    var i = end - 1
    while i >= start:
        rev.append(bidi_mirror(codes[i]))
        i -= 1
    return encode_utf8(rev)


def layout_rich(
    spans: List[RichSpan], ui: SdfFont, display: SdfFont, px: Float32, base_face: Int,
    max_width: Float32, align: Int, pad: Bool, sony: Bool, keys: List[String] = List[String](),
    base_dir: Int = BIDI_AUTO,
) -> RichLayout:
    """Place `spans` on lines no wider than `max_width` (<= 0: no limit) at
    `px` per em. align 0 left, 1 centre, 2 right inside the widest line (or
    `max_width` when given). Reorders mixed/RTL lines via UAX #9 BiDi."""
    var codes = List[Int]()
    var owner = List[Int]()
    var advance = List[Float32]()
    var kern_before = List[Float32]()
    for s in range(len(spans)):
        var span = spans[s]
        var face = span.face if span.face >= 0 else base_face
        var span_px = _safe_span_px(px, span.scale)
        if span.kind == SPAN_BUTTON:
            var h = span_px * BUTTON_HEIGHT
            if not isfinite(h):
                h = Float32(0.0)
            var label = span.text if span.text.byte_length() else glyph_label(span.action, pad, sony, keys)
            var label_px = span_px * BUTTON_LABEL
            if not isfinite(label_px):
                label_px = Float32(0.0)
            var label_width = run_width_text(ui, label, label_px)
            var adv = glyph_width(SHAPE_KEYCAP if span.text.byte_length() else glyph_shape(span.action, pad), label_width, h)
            if not isfinite(adv):
                adv = Float32(0.0)
            codes.append(OBJECT_CODE)
            owner.append(s)
            advance.append(adv)
            kern_before.append(0)
            continue
        var font_scale = _safe_font_scale(span_px, display.bake_px if face == 1 else ui.bake_px)
        var span_codes = decode_utf8(span.text)
        for i in range(len(span_codes)):
            var code = span_codes[i]
            codes.append(code)
            owner.append(s)
            var a: Float32 = 0
            var k: Float32 = 0
            if code != 0x0A and font_scale > Float32(0.0):
                var raw_a = (display.advance_of(code) if face == 1 else ui.advance_of(code)) * font_scale
                if isfinite(raw_a):
                    a = raw_a
                if i > 0:
                    var raw_k = (display.kern(span_codes[i - 1], code) if face == 1 else ui.kern(span_codes[i - 1], code)) * font_scale
                    if isfinite(raw_k):
                        k = raw_k
            advance.append(a)
            kern_before.append(k)
    var safe_max_width = max_width if isfinite(max_width) else Float32(0.0)
    var resolved = resolve_bidi(codes, base_dir)
    var lines = break_items(codes, advance, kern_before, safe_max_width)
    var layout = RichLayout()
    layout.lines = len(lines)
    layout.base_level = resolved.base_level
    for line in lines:
        if isfinite(line.width):
            layout.width = max(layout.width, line.width)
    var box = safe_max_width if safe_max_width > 0 else layout.width
    var top: Float32 = 0
    for l in range(len(lines)):
        var line = lines[l]
        var line_h: Float32 = 0
        var cap: Float32 = 0
        for index in range(line.start, line.end):
            var span = spans[owner[index]]
            var face = span.face if span.face >= 0 else base_face
            var metric = face_font_metric(ui, display, face)
            var span_px = _safe_span_px(px, span.scale)
            var mscale = _safe_font_scale(span_px, metric[2])
            if isfinite(metric[0]):
                var lh = metric[0] * mscale
                if isfinite(lh):
                    line_h = max(line_h, lh)
            if isfinite(metric[1]):
                var ch = metric[1] * mscale
                if isfinite(ch):
                    cap = max(cap, ch)
        if line.end == line.start or line_h == 0:
            var metric = face_font_metric(ui, display, base_face)
            var mscale = _safe_font_scale(px, metric[2])
            if isfinite(metric[0]):
                var lh = metric[0] * mscale
                if isfinite(lh):
                    line_h = lh
            if isfinite(metric[1]):
                var ch = metric[1] * mscale
                if isfinite(ch):
                    cap = ch
        var baseline = _safe_add_f32(top, cap)
        var x: Float32 = 0
        var line_w = line.width if isfinite(line.width) else Float32(0)
        if align == 1:
            x = (box - line_w) * 0.5
        elif align == 2:
            x = box - line_w
        if not isfinite(x):
            x = Float32(0)

        var bidi_runs = reorder_line(resolved, line.start, line.end)
        for br in bidi_runs:
            var rtl = br.is_rtl()
            # Partition [br.start, br.end) into contiguous span-owner segments
            var seg_starts = List[Int]()
            var seg_ends = List[Int]()
            var p = br.start
            while p < br.end:
                var s_id = owner[p]
                var s_start = p
                if spans[s_id].kind == SPAN_BUTTON:
                    p += 1
                else:
                    while p < br.end and owner[p] == s_id and spans[owner[p]].kind != SPAN_BUTTON:
                        p += 1
                seg_starts.append(s_start)
                seg_ends.append(p)

            var seg_count = len(seg_starts)
            for k in range(seg_count):
                var seg_i = (seg_count - 1 - k) if rtl else k
                var rs = seg_starts[seg_i]
                var re = seg_ends[seg_i]
                var s = owner[rs]
                var span = spans[s]
                var face = span.face if span.face >= 0 else base_face
                var span_px = _safe_span_px(px, span.scale)
                if span.kind == SPAN_BUTTON:
                    layout.runs.append(PlacedRun(SPAN_BUTTON, String(), x, l, baseline, advance[rs], span_px, span.color, face, span.action, br.level, rtl, span.text))
                    x = _safe_add_f32(x, advance[rs])
                    continue
                var vis_end = re
                while vis_end > rs and (codes[vis_end - 1] == 0x0A or codes[vis_end - 1] == 0x0D or codes[vis_end - 1] == 0x20):
                    vis_end -= 1
                var vis_w: Float32 = 0
                for idx in range(rs, vis_end):
                    var step_w = _safe_add_f32(advance[idx], (kern_before[idx] if idx > line.start and idx > rs else Float32(0)))
                    vis_w = _safe_add_f32(vis_w, step_w)
                var trail_w: Float32 = 0
                for idx in range(vis_end, re):
                    var step_tw = _safe_add_f32(advance[idx], (kern_before[idx] if idx > line.start and idx > rs else Float32(0)))
                    trail_w = _safe_add_f32(trail_w, step_tw)
                var total_seg_w = _safe_add_f32(vis_w, trail_w)
                if rtl:
                    if re < line.end:
                        x = _safe_add_f32(x, trail_w)
                    if vis_end > rs:
                        var vtxt = _visual_run_text(codes, rs, vis_end, True)
                        layout.runs.append(PlacedRun(SPAN_TEXT, vtxt, x, l, baseline, vis_w, span_px, span.color, face, ACTION_NONE, br.level, True))
                    x = _safe_add_f32(x, vis_w)
                else:
                    if vis_end > rs:
                        var vtxt = _visual_run_text(codes, rs, vis_end, False)
                        layout.runs.append(PlacedRun(SPAN_TEXT, vtxt, x, l, baseline, total_seg_w, span_px, span.color, face, ACTION_NONE, br.level, False))
                    x = _safe_add_f32(x, total_seg_w)
        top = _safe_add_f32(top, line_h)
    layout.height = top
    return layout^


struct PlacedShapedRun(Copyable, Movable):
    var kind: Int
    var glyphs: List[ShapedGlyph]
    var x: Float32
    var line: Int
    var baseline: Float32
    var width: Float32
    var px: Float32
    var color: Int
    var face: Int
    var action: Int
    var button_label: String
    var level: Int
    var rtl: Bool

    def __init__(
        out self,
        kind: Int,
        var glyphs: List[ShapedGlyph],
        x: Float32,
        line: Int,
        baseline: Float32,
        width: Float32,
        px: Float32,
        color: Int,
        face: Int,
        action: Int,
        level: Int = 0,
        rtl: Bool = False,
        button_label: String = "",
    ):
        self.kind = kind
        self.glyphs = glyphs^
        self.x = x
        self.line = line
        self.baseline = baseline
        self.width = width
        self.px = px
        self.color = color
        self.face = face
        self.action = action
        self.button_label = button_label
        self.level = level
        self.rtl = rtl


struct RichShapedLayout(Copyable, Movable):
    var runs: List[PlacedShapedRun]
    var lines: Int
    var width: Float32
    var height: Float32
    var base_level: Int

    def __init__(out self):
        self.runs = List[PlacedShapedRun]()
        self.lines = 0
        self.width = 0.0
        self.height = 0.0
        self.base_level = 0


def layout_rich_shaped(
    spans: List[RichSpan],
    ui: SdfFont,
    display: SdfFont,
    px: Float32,
    base_face: Int,
    max_width: Float32,
    align: Int,
    pad: Bool,
    sony: Bool,
    keys: List[String] = List[String](),
    base_dir: Int = BIDI_AUTO,
) -> RichShapedLayout:
    """Full GSUB + BiDi + fallback-chain shaped rich text layout without
    double-reversing RTL text."""
    var codes = List[Int]()
    var owner = List[Int]()
    var advance = List[Float32]()
    var kern_before = List[Float32]()
    # Borrow each font's cached shaper when it is current; build a private one only if not.
    var disp_uses_ui = not (display.valid or display.has_source or len(display.fallbacks) > 0)
    var ui_cached = ui.shaper_current()
    var disp_cached = ui_cached if disp_uses_ui else display.shaper_current()
    var ui_shaper = TextShaper() if ui_cached else TextShaper(ui.fallback_chain())
    var disp_shaper = TextShaper()
    if not disp_cached:
        if disp_uses_ui:
            disp_shaper = TextShaper(ui.fallback_chain())
        else:
            disp_shaper = TextShaper(display.fallback_chain())

    for s in range(len(spans)):
        var span = spans[s]
        var face = span.face if span.face >= 0 else base_face
        var span_px = _safe_span_px(px, span.scale)
        if span.kind == SPAN_BUTTON:
            var h = span_px * BUTTON_HEIGHT
            if not isfinite(h):
                h = Float32(0.0)
            var label = span.text if span.text.byte_length() else glyph_label(span.action, pad, sony, keys)
            var label_px = span_px * BUTTON_LABEL
            if not isfinite(label_px):
                label_px = Float32(0.0)
            var label_width = run_width_text(ui, label, label_px)
            var adv = glyph_width(SHAPE_KEYCAP if span.text.byte_length() else glyph_shape(span.action, pad), label_width, h)
            if not isfinite(adv):
                adv = Float32(0.0)
            codes.append(OBJECT_CODE)
            owner.append(s)
            advance.append(adv)
            kern_before.append(0.0)
            continue
        var font_scale = _safe_font_scale(span_px, display.bake_px if face == 1 else ui.bake_px)
        var span_codes = decode_utf8(span.text)
        for i in range(len(span_codes)):
            var code = span_codes[i]
            codes.append(code)
            owner.append(s)
            var a: Float32 = 0.0
            var k: Float32 = 0.0
            if code != 0x0A and code != 0x0D and font_scale > Float32(0.0):
                var raw_a = (display.advance_of(code) if face == 1 else ui.advance_of(code)) * font_scale
                if isfinite(raw_a):
                    a = raw_a
                if i > 0:
                    var raw_k = (display.kern(span_codes[i - 1], code) if face == 1 else ui.kern(span_codes[i - 1], code)) * font_scale
                    if isfinite(raw_k):
                        k = raw_k
            advance.append(a)
            kern_before.append(k)

    var safe_max_width = max_width if isfinite(max_width) else Float32(0.0)
    var resolved = resolve_bidi(codes, base_dir)
    var arabic_forms = compute_arabic_forms(codes)
    var lines = break_items(codes, advance, kern_before, safe_max_width)
    var layout = RichShapedLayout()
    layout.lines = len(lines)
    layout.base_level = resolved.base_level
    for line in lines:
        if isfinite(line.width):
            layout.width = max(layout.width, line.width)
    var box = safe_max_width if safe_max_width > 0 else layout.width
    var top: Float32 = 0.0
    for l in range(len(lines)):
        var line = lines[l]
        var line_h: Float32 = 0.0
        var cap: Float32 = 0.0
        for index in range(line.start, line.end):
            var span = spans[owner[index]]
            var face = span.face if span.face >= 0 else base_face
            var metric = face_font_metric(ui, display, face)
            var span_px = _safe_span_px(px, span.scale)
            var mscale = _safe_font_scale(span_px, metric[2])
            if isfinite(metric[0]):
                var lh = metric[0] * mscale
                if isfinite(lh):
                    line_h = max(line_h, lh)
            if isfinite(metric[1]):
                var ch = metric[1] * mscale
                if isfinite(ch):
                    cap = max(cap, ch)
        if line.end == line.start or line_h == 0.0:
            var metric = face_font_metric(ui, display, base_face)
            var mscale = _safe_font_scale(px, metric[2])
            if isfinite(metric[0]):
                var lh = metric[0] * mscale
                if isfinite(lh):
                    line_h = lh
            if isfinite(metric[1]):
                var ch = metric[1] * mscale
                if isfinite(ch):
                    cap = ch
        var baseline = _safe_add_f32(top, cap)
        var x: Float32 = 0.0
        var line_w = line.width if isfinite(line.width) else Float32(0.0)
        if align == 1:
            x = (box - line_w) * Float32(0.5)
        elif align == 2:
            x = box - line_w
        if not isfinite(x):
            x = Float32(0.0)

        var bidi_runs = reorder_line(resolved, line.start, line.end)
        for br in bidi_runs:
            var rtl = br.is_rtl()
            var seg_starts = List[Int]()
            var seg_ends = List[Int]()
            var p = br.start
            while p < br.end:
                var s_id = owner[p]
                var s_start = p
                if spans[s_id].kind == SPAN_BUTTON:
                    p += 1
                else:
                    while p < br.end and owner[p] == s_id and spans[owner[p]].kind != SPAN_BUTTON:
                        p += 1
                seg_starts.append(s_start)
                seg_ends.append(p)

            var seg_count = len(seg_starts)
            for k in range(seg_count):
                var seg_i = (seg_count - 1 - k) if rtl else k
                var rs = seg_starts[seg_i]
                var re = seg_ends[seg_i]
                var s = owner[rs]
                var span = spans[s]
                var face = span.face if span.face >= 0 else base_face
                var span_px = _safe_span_px(px, span.scale)
                if span.kind == SPAN_BUTTON:
                    layout.runs.append(PlacedShapedRun(SPAN_BUTTON, List[ShapedGlyph](), x, l, baseline, advance[rs], span_px, span.color, face, span.action, br.level, rtl, span.text))
                    x = _safe_add_f32(x, advance[rs])
                    continue
                var vis_end = re
                while vis_end > rs and (codes[vis_end - 1] == 0x0A or codes[vis_end - 1] == 0x0D or codes[vis_end - 1] == 0x20):
                    vis_end -= 1
                var sglyphs = List[ShapedGlyph]()
                var shaped_w: Float32 = 0.0
                if vis_end > rs:
                    var sruns: List[ShapedRun]
                    if face == 1:
                        if not disp_cached:
                            sruns = disp_shaper.shape_run_slice(codes, arabic_forms, rs, vis_end, br.level, span_px)
                        elif disp_uses_ui:
                            sruns = ui._shaper.shape_run_slice(codes, arabic_forms, rs, vis_end, br.level, span_px)
                        else:
                            sruns = display._shaper.shape_run_slice(codes, arabic_forms, rs, vis_end, br.level, span_px)
                    elif ui_cached:
                        sruns = ui._shaper.shape_run_slice(codes, arabic_forms, rs, vis_end, br.level, span_px)
                    else:
                        sruns = ui_shaper.shape_run_slice(codes, arabic_forms, rs, vis_end, br.level, span_px)
                    for sr in sruns:
                        for g in sr.glyphs:
                            sglyphs.append(g)
                            shaped_w = _safe_add_f32(shaped_w, g.x_advance)
                var trailing_space_w: Float32 = 0.0
                for idx in range(vis_end, re):
                    trailing_space_w = _safe_add_f32(trailing_space_w, advance[idx])
                if rtl:
                    if re < line.end:
                        x = _safe_add_f32(x, trailing_space_w)
                    if len(sglyphs) > 0:
                        layout.runs.append(PlacedShapedRun(SPAN_TEXT, sglyphs^, x, l, baseline, shaped_w, span_px, span.color, face, ACTION_NONE, br.level, True))
                    x = _safe_add_f32(x, shaped_w)
                else:
                    if len(sglyphs) > 0:
                        layout.runs.append(PlacedShapedRun(SPAN_TEXT, sglyphs^, x, l, baseline, shaped_w, span_px, span.color, face, ACTION_NONE, br.level, False))
                    x = _safe_add_f32(x, _safe_add_f32(shaped_w, trailing_space_w))
        top = _safe_add_f32(top, line_h)
    layout.height = top
    return layout^


def run_width_text(font: SdfFont, text: String, px: Float32) -> Float32:
    return font.measure(text, px)
