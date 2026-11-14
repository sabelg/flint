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
from flint.render.font import SdfFont
from flint.text.utf8 import decode_utf8, encode_utf8
from flint.text.line_break import TextLine, break_items
from flint.ui.button_glyphs import ACTION_NONE, action_id, glyph_shape, glyph_label, glyph_width

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


@fieldwise_init
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


struct RichLayout(Copyable, Movable):
    var runs: List[PlacedRun]
    var lines: Int
    var width: Float32
    var height: Float32

    def __init__(out self):
        self.runs = List[PlacedRun]()
        self.lines = 0
        self.width = 0
        self.height = 0


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

    def apply(mut self, tag: List[UInt8]) -> Bool:
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
            var action = action_id(String(unsafe_from_utf8=tail(tag, 4)))
            if action == ACTION_NONE:
                return False
            self.flush()
            self.spans.append(RichSpan(SPAN_BUTTON, String(), self.color(), self.face(), self.scale(), action))
            return True
        return False


def parse_rich(text: String) -> List[RichSpan]:
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
        for probe in range(index + 1, min(count, index + 24)):
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
        if parser.apply(body):
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


def layout_rich(
    spans: List[RichSpan], ui: SdfFont, display: SdfFont, px: Float32, base_face: Int,
    max_width: Float32, align: Int, pad: Bool, sony: Bool, keys: List[String] = List[String](),
) -> RichLayout:
    """Place `spans` on lines no wider than `max_width` (<= 0: no limit) at
    `px` per em. align 0 left, 1 centre, 2 right inside the widest line (or
    `max_width` when given). Units are those of `px`."""
    var codes = List[Int]()
    var owner = List[Int]()
    var advance = List[Float32]()
    var kern_before = List[Float32]()
    for s in range(len(spans)):
        var span = spans[s]
        var face = span.face if span.face >= 0 else base_face
        var span_px = px * Float32(span.scale) / 100.0
        if span.kind == SPAN_BUTTON:
            var h = span_px * BUTTON_HEIGHT
            var label = glyph_label(span.action, pad, sony, keys)
            var label_width = run_width_text(ui, label, span_px * BUTTON_LABEL)
            codes.append(OBJECT_CODE)
            owner.append(s)
            advance.append(glyph_width(glyph_shape(span.action, pad), label_width, h))
            kern_before.append(0)
            continue
        var font_scale = span_px / (display.bake_px if face == 1 else ui.bake_px)
        var span_codes = decode_utf8(span.text)
        for i in range(len(span_codes)):
            var code = span_codes[i]
            codes.append(code)
            owner.append(s)
            var a: Float32 = 0
            var k: Float32 = 0
            if code != 0x0A:
                a = (display.advance_of(code) if face == 1 else ui.advance_of(code)) * font_scale
                if i > 0:
                    k = (display.kern(span_codes[i - 1], code) if face == 1 else ui.kern(span_codes[i - 1], code)) * font_scale
            advance.append(a)
            kern_before.append(k)
    var lines = break_items(codes, advance, kern_before, max_width)
    var layout = RichLayout()
    layout.lines = len(lines)
    for line in lines:
        layout.width = max(layout.width, line.width)
    var box = max_width if max_width > 0 else layout.width
    var top: Float32 = 0
    for l in range(len(lines)):
        var line = lines[l]
        # The tallest span on the line sets its height and baseline.
        var line_h: Float32 = 0
        var cap: Float32 = 0
        for index in range(line.start, line.end):
            var span = spans[owner[index]]
            var face = span.face if span.face >= 0 else base_face
            var metric = face_font_metric(ui, display, face)
            var span_px = px * Float32(span.scale) / 100.0
            line_h = max(line_h, metric[0] * span_px / metric[2])
            cap = max(cap, metric[1] * span_px / metric[2])
        if line.end == line.start or line_h == 0:
            var metric = face_font_metric(ui, display, base_face)
            line_h = metric[0] * px / metric[2]
            cap = metric[1] * px / metric[2]
        var baseline = top + cap
        var x: Float32 = 0
        if align == 1:
            x = (box - line.width) * 0.5
        elif align == 2:
            x = box - line.width
        var index = line.start
        while index < line.end:
            var s = owner[index]
            var span = spans[s]
            var face = span.face if span.face >= 0 else base_face
            var span_px = px * Float32(span.scale) / 100.0
            if span.kind == SPAN_BUTTON:
                layout.runs.append(PlacedRun(SPAN_BUTTON, String(), x, l, baseline, advance[index], span_px, span.color, face, span.action))
                x += advance[index]
                index += 1
                continue
            var run_start = index
            var run_x = x
            while index < line.end and owner[index] == s:
                x += advance[index] + (kern_before[index] if index > line.start and index > run_start else Float32(0))
                index += 1
            # Leave the newline and hanging spaces out of the drawn text.
            var run_end = index
            while run_end > run_start and (codes[run_end - 1] == 0x0A or codes[run_end - 1] == 0x20):
                run_end -= 1
            if run_end > run_start:
                layout.runs.append(PlacedRun(SPAN_TEXT, encode_utf8(codes, run_start, run_end), run_x, l, baseline, x - run_x, span_px, span.color, face, ACTION_NONE))
        top += line_h
    layout.height = top
    return layout^


def run_width_text(font: SdfFont, text: String, px: Float32) -> Float32:
    return font.measure(text, px)
