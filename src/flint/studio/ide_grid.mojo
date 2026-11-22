"""Incremental decoder for the xterm cell updates emitted by TurboMojo."""
comptime DEFAULT_FG: UInt32 = 0xE1E5EB
comptime DEFAULT_BG: UInt32 = 0x111822


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


struct IdeGrid(Movable):
    var cells: List[Cell]
    var cols: Int
    var rows: Int
    var x: Int
    var y: Int
    var cursor: Bool
    var mouse: Bool
    var fg: UInt32
    var bg: UInt32
    var underline: Bool
    var reverse: Bool
    var state: Int  # ground, escape, CSI, control string, string escape
    var params: List[Int]
    var private_mode: Bool
    var utf: Int
    var remaining: Int

    def __init__(out self):
        self.cols = 0
        self.rows = 0
        self.cells = List[Cell]()
        self.x = 0
        self.y = 0
        self.cursor = True
        self.mouse = False
        self.fg = DEFAULT_FG
        self.bg = DEFAULT_BG
        self.underline = False
        self.reverse = False
        self.state = 0
        self.params = List[Int]()
        self.private_mode = False
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
        self.x = 0
        self.y = 0

    def clear_range(mut self, start: Int, end: Int):
        for i in range(max(0, start), min(len(self.cells), end)):
            self.cells[i] = Cell(32, self.fg, self.bg, False)

    def newline(mut self):
        if self.cols <= 0 or self.rows <= 0:
            return
        if self.y + 1 < self.rows:
            self.y += 1
        else:
            var end = len(self.cells) - self.cols
            for i in range(end):
                self.cells[i] = self.cells[i + self.cols]
            self.clear_range(end, len(self.cells))

    def put(mut self, code: Int):
        if self.cols <= 0 or self.rows <= 0:
            return
        if self.x >= self.cols:
            self.x = 0
            self.newline()
        var pos = self.y * self.cols + self.x
        if 0 <= pos < len(self.cells):
            self.cells[pos] = Cell(
                code,
                self.bg if self.reverse else self.fg,
                self.fg if self.reverse else self.bg,
                self.underline,
            )
        self.x += 1

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
            self.x = max(0, self.x - n)
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
        if self.state >= 3:
            if b == 7 or (self.state == 4 and b == 92):
                self.state = 0
            else:
                self.state = 4 if b == 27 else 3
        elif self.state == 2:
            var i = len(self.params) - 1
            if 48 <= b <= 57:
                self.params[i] = min(65535, self.params[i] * 10 + b - 48)
            elif b == 59 or b == 58:
                if len(self.params) < 32:
                    self.params.append(0)
            elif b == 63:
                self.private_mode = True
            elif 64 <= b <= 126:
                self.csi(b)
                self.state = 0
            elif b == 27:
                self.state = 1
        elif self.state == 1:
            self.state = 0
            if b == 91:
                self.state = 2
                self.params.clear()
                self.params.append(0)
                self.private_mode = False
            elif b == 93 or b == 80 or b == 95 or b == 94:
                self.state = 3
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
                self.state = 1
            elif b == 13:
                self.x = 0
            elif b == 10:
                self.newline()
            elif b == 8:
                self.x = max(0, self.x - 1)
            elif 32 <= b <= 126:
                self.put(b)
            elif 194 <= b <= 244:
                self.remaining = 1 if b < 224 else (2 if b < 240 else 3)
                self.utf = b & (127 >> (self.remaining + 1))
