"""Incremental decoder for the xterm cell updates emitted by TurboMojo.

Rows live in a ring buffer: a line feed on the bottom row advances `top` and clears
one row instead of shifting the whole grid. Read cells with `cell(row, col)` /
`index(row, col)`. Writers that fill `cells` linearly (TurboMojo's GUI) never scroll,
so `top` stays 0 for them and linear indices keep working.

Wide (East Asian / emoji) code points take two cells: the lead cell holds the code and
the cell after it holds `WIDE_CONTINUATION`. Zero-width combining marks, ZWJ and
variation selectors (U+FE0F) attach to the previous cell (`mark(row, col)`).
"""
from flint.studio.wcwidth_tables import WIDE, WIDE_N, ZERO_WIDTH, ZERO_WIDTH_N

comptime DEFAULT_FG: UInt32 = 0xE1E5EB
comptime DEFAULT_BG: UInt32 = 0x111822
comptime WIDE_CONTINUATION: Int = 0
"""`Cell.code` of the right half of a double-width character (never drawn)."""
comptime TAB_STOP: Int = 8

# Decoder states
comptime _ST_GROUND: Int = 0
comptime _ST_ESC: Int = 1
comptime _ST_CSI: Int = 2
comptime _ST_STRING: Int = 3
comptime _ST_STRING_ESC: Int = 4
comptime _ST_ESC_INTERMEDIATE: Int = 5


@fieldwise_init
struct Cell(TrivialRegisterPassable):
    var code: Int
    var fg: UInt32
    var bg: UInt32
    var underline: Bool


def palette(n: Int) -> UInt32:
    var basic: Array[UInt32, 16] = [
        0x101820,
        0xB34747,
        0x55AA68,
        0xC3A749,
        0x527CBA,
        0xAB74AB,
        0x58ACAE,
        0xBABABA,
        0x707070,
        0xF07878,
        0x80DC90,
        0xEFDA84,
        0x90AFFF,
        0xDF98DF,
        0x8DE1DF,
        0xFFFFFF,
    ]
    if n < 16:
        return basic[max(0, n)]
    if n >= 232:
        return UInt32(8 + 10 * (min(n, 255) - 232)) * 0x010101
    var levels: Array[Int, 6] = [0, 95, 135, 175, 215, 255]
    var k = n - 16
    return UInt32((levels[k // 36] << 16) | (levels[k // 6 % 6] << 8) | levels[k % 6])


def _in_ranges[N: Int](table: Array[Int32, N], pairs: Int, cp: Int) -> Bool:
    var lo = 0
    var hi = pairs - 1
    while lo <= hi:
        var mid = (lo + hi) // 2
        if cp < Int(table[2 * mid]):
            hi = mid - 1
        elif cp > Int(table[2 * mid + 1]):
            lo = mid + 1
        else:
            return True
    return False


def char_width(cp: Int) -> Int:
    """wcwidth: -1 for C0/C1 controls and DEL, 0 for zero-width code points (Mn/Me/Cf, ZWJ,
    variation selectors, Hangul medial/final jamo), 2 for East Asian Wide/Fullwidth and emoji
    presentation, else 1. Tables: `flint.studio.wcwidth_tables` (tools/gen_wcwidth.py)."""
    if cp < 0x20 or (cp >= 0x7F and cp < 0xA0):
        return -1
    if cp < 0x300:
        return 1
    var zero = materialize[ZERO_WIDTH]()
    if _in_ranges(zero, ZERO_WIDTH_N, cp):
        return 0
    if cp < 0x1100:
        return 1
    var wide = materialize[WIDE]()
    if _in_ranges(wide, WIDE_N, cp):
        return 2
    return 1


struct IdeGrid(Movable):
    var cells: List[Cell]
    """Physical storage; logical row `r` is physical row `(top + r) % rows`."""
    var marks: List[String]
    """Combining marks per physical cell (empty until the first mark arrives)."""
    var cols: Int
    var rows: Int
    var top: Int
    var x: Int
    var y: Int
    var cursor: Bool
    var mouse: Bool
    var fg: UInt32
    var bg: UInt32
    var underline: Bool
    var reverse: Bool
    var state: Int  # ground, escape, CSI, control string, string escape, escape intermediate
    var params: List[Int]
    var private_mode: Bool
    var prefix: Int
    """CSI private marker byte (`?`, `>`, `<`, `=`) or 0."""
    var intermediate: Bool
    """A CSI intermediate byte (0x20-0x2F) was seen."""
    var utf: Int
    var remaining: Int

    def __init__(out self):
        self.cols = 0
        self.rows = 0
        self.top = 0
        self.cells = List[Cell]()
        self.marks = List[String]()
        self.x = 0
        self.y = 0
        self.cursor = True
        self.mouse = False
        self.fg = DEFAULT_FG
        self.bg = DEFAULT_BG
        self.underline = False
        self.reverse = False
        self.state = _ST_GROUND
        self.params = List[Int]()
        self.private_mode = False
        self.prefix = 0
        self.intermediate = False
        self.utf = 0
        self.remaining = 0

    def resize(mut self, cols: Int, rows: Int):
        var c = max(0, cols)
        var r = max(0, rows)
        if c == 0 or r == 0:
            self.cols = 0
            self.rows = 0
            self.cells = List[Cell]()
        else:
            self.cols = c
            self.rows = r
            self.cells = List[Cell](length=c * r, fill=Cell(32, DEFAULT_FG, DEFAULT_BG, False))
        self.marks = List[String]()
        self.top = 0
        self.x = 0
        self.y = 0

    @always_inline
    def index(self, row: Int, col: Int) -> Int:
        """Physical `cells` index of logical (`row`, `col`)."""
        return ((self.top + row) % self.rows) * self.cols + col

    def cell(self, row: Int, col: Int) -> Cell:
        if row < 0 or col < 0 or row >= self.rows or col >= self.cols:
            return Cell(32, DEFAULT_FG, DEFAULT_BG, False)
        return self.cells[self.index(row, col)]

    def mark(self, row: Int, col: Int) -> String:
        """Combining marks attached to logical (`row`, `col`)."""
        if len(self.marks) == 0 or row < 0 or col < 0 or row >= self.rows or col >= self.cols:
            return String()
        return self.marks[self.index(row, col)]

    def row_text(self, row: Int) -> String:
        """Logical row as UTF-8 (wide continuations skipped, marks kept); for tests and copy."""
        var s = String()
        for c in range(self.cols):
            var code = self.cell(row, c).code
            if code == WIDE_CONTINUATION:
                continue
            s += String(chr(code))
            s += self.mark(row, c)
        return s^

    def _blank(mut self, phys: Int):
        self.cells[phys] = Cell(32, self.fg, self.bg, False)
        if len(self.marks) > 0:
            self.marks[phys] = String()

    def clear_range(mut self, start: Int, end: Int):
        """Clears logical linear positions [start, end) (`row * cols + col`)."""
        if self.cols <= 0 or self.rows <= 0:
            return
        for i in range(max(0, start), min(len(self.cells), end)):
            self._blank(self.index(i // self.cols, i % self.cols))

    def newline(mut self):
        if self.cols <= 0 or self.rows <= 0:
            return
        if self.y + 1 < self.rows:
            self.y += 1
        else:
            # Ring scroll: the old top row becomes the new (cleared) bottom row.
            self.top = (self.top + 1) % self.rows
            var base = self.index(self.rows - 1, 0)
            for c in range(self.cols):
                self._blank(base + c)

    def _store(mut self, col: Int, code: Int):
        var pos = self.index(self.y, col)
        # Overwriting either half of a wide character blanks the other half.
        var old = self.cells[pos].code
        if old == WIDE_CONTINUATION and col > 0:
            var lead = self.index(self.y, col - 1)
            self.cells[lead] = Cell(32, self.cells[lead].fg, self.cells[lead].bg, False)
        elif col + 1 < self.cols and self.cells[self.index(self.y, col + 1)].code == WIDE_CONTINUATION and code != WIDE_CONTINUATION:
            var tail = self.index(self.y, col + 1)
            self.cells[tail] = Cell(32, self.cells[tail].fg, self.cells[tail].bg, False)
        self.cells[pos] = Cell(
            code,
            self.bg if self.reverse else self.fg,
            self.fg if self.reverse else self.bg,
            self.underline,
        )
        if len(self.marks) > 0:
            self.marks[pos] = String()

    def put(mut self, code: Int):
        """Writes one code point at the cursor, honouring its display width."""
        if self.cols <= 0 or self.rows <= 0:
            return
        var w = char_width(code)
        if w < 0:
            return  # C0/C1 control that reached the printable path (e.g. UTF-8 encoded C1)
        if w == 0:
            self._attach_mark(code)
            return
        if w == 2 and self.cols < 2:
            w = 1
        if self.x >= self.cols or (w == 2 and self.x + 1 >= self.cols):
            self.x = 0
            self.newline()
        self._store(self.x, code)
        if w == 2:
            self._store(self.x + 1, WIDE_CONTINUATION)
        self.x += w

    def _attach_mark(mut self, code: Int):
        if self.x <= 0:
            return
        var col = min(self.x, self.cols) - 1
        if col > 0 and self.cells[self.index(self.y, col)].code == WIDE_CONTINUATION:
            col -= 1
        if len(self.marks) == 0:
            self.marks = List[String](length=len(self.cells), fill=String())
        self.marks[self.index(self.y, col)] += String(chr(code))

    def tab(mut self):
        if self.cols <= 0:
            return
        self.x = min(self.cols - 1, (min(self.x, self.cols - 1) // TAB_STOP + 1) * TAB_STOP)

    def sgr(mut self):
        var i = 0
        while i < len(self.params):
            var p = self.params[i]
            if p == 0:
                self.fg = DEFAULT_FG
                self.bg = DEFAULT_BG
                self.reverse = False
                self.underline = False
            elif p == 4 or p == 24:
                self.underline = p == 4
            elif p == 7 or p == 27:
                self.reverse = p == 7
            elif p == 39:
                self.fg = DEFAULT_FG
            elif p == 49:
                self.bg = DEFAULT_BG
            elif 30 <= p <= 37 or 90 <= p <= 97:
                self.fg = palette(p - 30 if p < 90 else p - 82)
            elif 40 <= p <= 47 or 100 <= p <= 107:
                self.bg = palette(p - 40 if p < 100 else p - 92)
            elif (p == 38 or p == 48 or p == 58) and i + 2 < len(self.params):
                var color: UInt32
                if self.params[i + 1] == 2 and i + 4 < len(self.params):
                    color = UInt32(
                        (min(255, self.params[i + 2]) << 16)
                        | (min(255, self.params[i + 3]) << 8)
                        | min(255, self.params[i + 4])
                    )
                    i += 4
                elif self.params[i + 1] == 5:
                    color = palette(self.params[i + 2])
                    i += 2
                else:
                    i += 1
                    continue
                if p == 38:
                    self.fg = color
                elif p == 48:
                    self.bg = color
            i += 1

    def csi(mut self, b: Int):
        if len(self.params) == 0:
            return
        # `CSI > 4;2 m` (modifyOtherKeys), `CSI = ... c`, `CSI < ...` and anything with an
        # intermediate byte (`CSI 2 SP q`, `CSI ! p`) are not cursor/SGR commands.
        if self.intermediate or (self.prefix != 0 and self.prefix != 63):
            return
        var p = self.params[0]
        var n = max(1, p)
        if self.private_mode:
            if b == 104 or b == 108:
                for mode in self.params:
                    if mode == 25:
                        self.cursor = b == 104
                    elif mode == 1006:
                        self.mouse = b == 104
            return
        if b == 109:
            self.sgr()
            return
        if self.cols <= 0 or self.rows <= 0:
            self.x = 0
            self.y = 0
            return
        var pos = self.y * self.cols + min(self.x, self.cols - 1)
        if b == 72 or b == 102:
            self.y = min(self.rows - 1, n - 1)
            self.x = min(self.cols - 1, max(0, self.params[1] - 1)) if len(self.params) > 1 else 0
        elif b == 65:
            self.y = max(0, self.y - n)
        elif b == 66:
            self.y = min(self.rows - 1, self.y + n)
        elif b == 67:
            self.x = min(self.cols - 1, self.x + n)
        elif b == 68:
            self.x = max(0, min(self.x, self.cols) - n)
        elif b == 71:
            self.x = min(self.cols - 1, n - 1)
        elif b == 74:
            self.clear_range(pos if p == 0 else 0, pos + 1 if p == 1 else len(self.cells))
        elif b == 75:
            self.clear_range(
                pos if p == 0 else self.y * self.cols,
                pos + 1 if p == 1 else (self.y + 1) * self.cols,
            )

    def feed(mut self, b: Int):
        if self.state >= _ST_STRING and self.state <= _ST_STRING_ESC:
            if b == 7 or (self.state == _ST_STRING_ESC and b == 92):
                self.state = _ST_GROUND
            else:
                self.state = _ST_STRING_ESC if b == 27 else _ST_STRING
        elif self.state == _ST_CSI:
            var i = len(self.params) - 1
            if 48 <= b <= 57:
                self.params[i] = min(65535, self.params[i] * 10 + b - 48)
            elif b == 59 or b == 58:
                if len(self.params) < 32:
                    self.params.append(0)
            elif 60 <= b <= 63:
                if self.prefix == 0:
                    self.prefix = b
                self.private_mode = self.prefix == 63
            elif 32 <= b <= 47:
                self.intermediate = True
            elif 64 <= b <= 126:
                self.csi(b)
                self.state = _ST_GROUND
            elif b == 27:
                self.state = _ST_ESC
            elif b == 24 or b == 26:
                self.state = _ST_GROUND
        elif self.state == _ST_ESC_INTERMEDIATE:
            # `ESC ( B`, `ESC # 8`, `ESC SP F`, ...: swallow intermediates, then the final byte.
            if b == 27:
                self.state = _ST_ESC
            elif 32 <= b <= 47:
                pass
            elif 48 <= b <= 126 or b == 24 or b == 26:
                self.state = _ST_GROUND
        elif self.state == _ST_ESC:
            self.state = _ST_GROUND
            if b == 91:
                self.state = _ST_CSI
                self.params.clear()
                self.params.append(0)
                self.private_mode = False
                self.prefix = 0
                self.intermediate = False
            elif b == 93 or b == 80 or b == 95 or b == 94:
                self.state = _ST_STRING
            elif 32 <= b <= 47:
                self.state = _ST_ESC_INTERMEDIATE
            elif b == 27:
                self.state = _ST_ESC
        elif self.remaining and 128 <= b <= 191:
            self.utf = (self.utf << 6) | (b & 63)
            self.remaining -= 1
            if self.remaining == 0:
                self.put(
                    self.utf if self.utf <= 0x10FFFF
                    and not 0xD800 <= self.utf <= 0xDFFF else 0xFFFD
                )
        else:
            self.remaining = 0
            if b == 27:
                self.state = _ST_ESC
            elif b == 13:
                self.x = 0
            elif b == 10 or b == 11 or b == 12:
                self.newline()
            elif b == 8:
                self.x = max(0, min(self.x, self.cols) - 1)
            elif b == 9:
                self.tab()
            elif 32 <= b <= 126:
                self.put(b)
            elif 194 <= b <= 244:
                self.remaining = 1 if b < 224 else (2 if b < 240 else 3)
                self.utf = b & (127 >> (self.remaining + 1))
