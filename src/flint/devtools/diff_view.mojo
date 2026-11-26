"""Pure-Mojo 1.2 line diff, unified/split diff model & hunk accept/reject (`flint.devtools.diff_view`).

Implements:
- Myers O(ND) hash-based line diff with common prefix/suffix stripping (`diff_line_hashes`, `diff_sides`).
- Per-line gutter change marks (`GUTTER_ADDED`, `GUTTER_CHANGED`, `GUTTER_DELETED_ABOVE/BELOW`).
- Unified and side-by-side split `DiffRow` builders + standard unified diff parser (`parse_unified_diff`).
- Interactive hunk-level accept/reject model (`DiffDocument`) for reviewing agent code edits.
"""
from std.collections import List
from std.math import max, min

comptime GUTTER_NONE: Int = 0
comptime GUTTER_ADDED: Int = 1
comptime GUTTER_CHANGED: Int = 2
comptime GUTTER_DELETED_ABOVE: Int = 3
comptime GUTTER_DELETED_BELOW: Int = 4

comptime DIFF_ROW_HEADER: Int = 0
comptime DIFF_ROW_SAME: Int = 1
comptime DIFF_ROW_OLD: Int = 2
comptime DIFF_ROW_NEW: Int = 3
comptime DIFF_ROW_NOTE: Int = 4
comptime DIFF_ROW_SPLIT_CHG: Int = 5

comptime HUNK_PENDING: Int = 0
comptime HUNK_ACCEPTED: Int = 1
comptime HUNK_REJECTED: Int = 2

comptime DEFAULT_DIFF_CONTEXT: Int = 3


@fieldwise_init
struct DiffHunk(Copyable, Movable, Equatable):
    """Old lines [a, a + la) replaced by new lines [b, b + lb) (0-based)."""
    var a: Int
    var la: Int
    var b: Int
    var lb: Int

    def __eq__(self, other: Self) -> Bool:
        return (
            self.a == other.a
            and self.la == other.la
            and self.b == other.b
            and self.lb == other.lb
        )


@fieldwise_init
struct DiffRow(Copyable, Movable):
    """Renderable row for unified or side-by-side split diff views."""
    var kind: Int
    var text: String        # unified representation: "@@ ...", " line", "-line", "+line"
    var hunk: Int           # hunk index (-1 for notes/unpaired)
    var old_line: Int       # 0-based line in old side (-1 if none)
    var new_line: Int       # 0-based line in new side (-1 if none)
    var left_text: String   # split view left column text
    var right_text: String  # split view right column text


@fieldwise_init
struct DiffSide(Copyable, Movable):
    """Normalized line representation of one side of a file diff."""
    var lines: List[String]
    var trailing_eol: Bool

    @staticmethod
    def from_text(text: String) -> DiffSide:
        var lines = List[String]()
        if text.byte_length() == 0:
            return DiffSide(lines^, True)
        for raw in text.split("\n"):
            lines.append(String(String(raw).rstrip("\r")))
        var eol = text.endswith("\n")
        if eol and len(lines) > 0 and lines[len(lines) - 1].byte_length() == 0:
            _ = lines.pop()
        return DiffSide(lines^, eol or len(lines) == 0)

    def to_text(self) -> String:
        var s = String("")
        for k in range(len(self.lines)):
            if k > 0:
                s += "\n"
            s += self.lines[k]
        if self.trailing_eol and len(self.lines) > 0:
            s += "\n"
        return s^

    def hashes(self) -> List[UInt64]:
        var out = List[UInt64](capacity=len(self.lines))
        for i in range(len(self.lines)):
            out.append(line_hash64(self.lines[i]))
        return out^


def line_hash64(line: String) -> UInt64:
    """FNV-1a 64-bit hash over UTF-8 bytes, ignoring a trailing CR."""
    var b = line.as_bytes()
    var n = len(b)
    if n > 0 and b[n - 1] == 13:
        n -= 1
    var h: UInt64 = 0xCBF29CE484222325
    for i in range(n):
        h = (h ^ UInt64(b[i])) * 0x100000001B3
    return h


def diff_line_hashes(old: List[UInt64], new: List[UInt64], max_d: Int = 1000) -> List[DiffHunk]:
    """Myers O(ND) diff over 64-bit line hashes with prefix/suffix trimming."""
    var n = len(old)
    var m = len(new)
    var pre = 0
    while pre < n and pre < m and old[pre] == new[pre]:
        pre += 1
    var suf = 0
    while suf < n - pre and suf < m - pre and old[n - 1 - suf] == new[m - 1 - suf]:
        suf += 1
    var out = List[DiffHunk]()
    var a0 = pre
    var a1 = n - suf
    var b0 = pre
    var b1 = m - suf
    if a0 == a1 and b0 == b1:
        return out^
    if a0 == a1 or b0 == b1:
        out.append(DiffHunk(a0, a1 - a0, b0, b1 - b0))
        return out^

    var N = a1 - a0
    var M = b1 - b0
    var limit = min(N + M, max_d)
    var off = limit + 1
    var v = List[Int](length=2 * limit + 3, fill=0)
    var trace = List[List[Int]]()
    var found = -1
    for d in range(limit + 1):
        var keep = List[Int](length=2 * d + 1, fill=0)
        for k in range(-d, d + 1):
            keep[k + d] = v[k + off]
        trace.append(keep^)
        for k in range(-d, d + 1, 2):
            var x: Int
            if k == -d or (k != d and v[k - 1 + off] < v[k + 1 + off]):
                x = v[k + 1 + off]
            else:
                x = v[k - 1 + off] + 1
            var y = x - k
            while x < N and y < M and old[a0 + x] == new[b0 + y]:
                x += 1
                y += 1
            v[k + off] = x
            if x >= N and y >= M:
                found = d
                break
        if found >= 0:
            break

    if found < 0:
        out.append(DiffHunk(a0, N, b0, M))
        return out^

    var pairs_x = List[Int]()
    var pairs_y = List[Int]()
    var lens = List[Int]()
    var x = N
    var y = M
    var d = found
    while d > 0:
        var k = x - y
        ref prev = trace[d]
        var down: Bool
        if k == -d:
            down = True
        elif k == d:
            down = False
        else:
            down = prev[k - 1 + d] < prev[k + 1 + d]
        var pk = k + 1 if down else k - 1
        var px = prev[pk + d]
        var py = px - pk
        var sx = px if down else px + 1
        if x > sx:
            pairs_x.append(sx)
            pairs_y.append(sx - k)
            lens.append(x - sx)
        x = px
        y = py
        d -= 1
    if x > 0:
        pairs_x.append(0)
        pairs_y.append(0)
        lens.append(x)

    var cx = 0
    var cy = 0
    var j = len(lens) - 1
    while j >= -1:
        var mx = N
        var my = M
        var ml = 0
        if j >= 0:
            mx = pairs_x[j]
            my = pairs_y[j]
            ml = lens[j]
        if mx > cx or my > cy:
            out.append(DiffHunk(a0 + cx, mx - cx, b0 + cy, my - cy))
        cx = mx + ml
        cy = my + ml
        j -= 1
    return out^


def diff_sides(old: DiffSide, new: DiffSide) -> List[DiffHunk]:
    """Computes `DiffHunk`s transforming `old` into `new`."""
    var h = diff_line_hashes(old.hashes(), new.hashes())
    if (
        len(h) == 0
        and old.trailing_eol != new.trailing_eol
        and len(old.lines) > 0
        and len(new.lines) > 0
    ):
        h.append(DiffHunk(len(old.lines) - 1, 1, len(new.lines) - 1, 1))
    return h^


def compute_gutter_marks(hunks: List[DiffHunk], new_line_count: Int) -> List[UInt8]:
    """Produces a per-line gutter mark (`GUTTER_*`) for the new file."""
    var marks = List[UInt8](length=max(0, new_line_count), fill=UInt8(GUTTER_NONE))
    for i in range(len(hunks)):
        ref h = hunks[i]
        if h.lb > 0:
            var mark = UInt8(GUTTER_ADDED if h.la == 0 else GUTTER_CHANGED)
            for r in range(h.b, min(h.b + h.lb, new_line_count)):
                marks[r] = mark
        elif new_line_count > 0:
            if h.b < new_line_count:
                if marks[h.b] == UInt8(GUTTER_NONE):
                    marks[h.b] = UInt8(GUTTER_DELETED_ABOVE)
            elif marks[new_line_count - 1] == UInt8(GUTTER_NONE):
                marks[new_line_count - 1] = UInt8(GUTTER_DELETED_BELOW)
    return marks^


def build_unified_diff(
    old: DiffSide, new: DiffSide, hunks: List[DiffHunk], context: Int = DEFAULT_DIFF_CONTEXT
) -> List[DiffRow]:
    """Builds unified diff rows (`@@ ... @@`, context, `-old`, `+new`)."""
    var rows = List[DiffRow]()
    var shown_to = 0
    for hi in range(len(hunks)):
        var h = hunks[hi].copy()
        var c0 = max(max(0, h.a - context), shown_to)
        var c1 = min(len(old.lines), h.a + h.la + context)
        if hi + 1 < len(hunks):
            c1 = min(c1, hunks[hi + 1].a)
        var before = h.a - c0
        var after = c1 - (h.a + h.la)
        shown_to = c1
        var ostart = c0
        var nstart = h.b - before
        var olen = before + h.la + after
        var nlen = before + h.lb + after
        var hdr = (
            "@@ -"
            + String(ostart + 1 if olen > 0 else ostart)
            + ","
            + String(olen)
            + " +"
            + String(nstart + 1 if nlen > 0 else nstart)
            + ","
            + String(nlen)
            + " @@"
        )
        rows.append(DiffRow(DIFF_ROW_HEADER, hdr, hi, ostart, max(0, h.b), hdr, hdr))
        for k in range(before):
            var txt = old.lines[c0 + k]
            rows.append(DiffRow(DIFF_ROW_SAME, " " + txt, hi, c0 + k, nstart + k, txt, txt))
        for k in range(h.la):
            var txt = old.lines[h.a + k]
            rows.append(DiffRow(DIFF_ROW_OLD, "-" + txt, hi, h.a + k, h.b, txt, ""))
        for k in range(h.lb):
            var txt = new.lines[h.b + k]
            rows.append(DiffRow(DIFF_ROW_NEW, "+" + txt, hi, h.a, h.b + k, "", txt))
        for k in range(after):
            var txt = old.lines[h.a + h.la + k]
            rows.append(
                DiffRow(DIFF_ROW_SAME, " " + txt, hi, h.a + h.la + k, h.b + h.lb + k, txt, txt)
            )
    return rows^


def build_split_diff(
    old: DiffSide, new: DiffSide, hunks: List[DiffHunk], context: Int = DEFAULT_DIFF_CONTEXT
) -> List[DiffRow]:
    """Builds side-by-side split diff rows with paired left/right texts and line indices."""
    var rows = List[DiffRow]()
    var shown_to = 0
    for hi in range(len(hunks)):
        var h = hunks[hi].copy()
        var c0 = max(max(0, h.a - context), shown_to)
        var c1 = min(len(old.lines), h.a + h.la + context)
        if hi + 1 < len(hunks):
            c1 = min(c1, hunks[hi + 1].a)
        var before = h.a - c0
        var after = c1 - (h.a + h.la)
        shown_to = c1
        var ostart = c0
        var nstart = h.b - before
        var olen = before + h.la + after
        var nlen = before + h.lb + after
        var hdr = (
            "@@ -"
            + String(ostart + 1 if olen > 0 else ostart)
            + ","
            + String(olen)
            + " +"
            + String(nstart + 1 if nlen > 0 else nstart)
            + ","
            + String(nlen)
            + " @@"
        )
        rows.append(DiffRow(DIFF_ROW_HEADER, hdr, hi, ostart, max(0, h.b), hdr, hdr))
        for k in range(before):
            var ltxt = old.lines[c0 + k]
            var rtxt = new.lines[nstart + k]
            rows.append(DiffRow(DIFF_ROW_SAME, " " + ltxt, hi, c0 + k, nstart + k, ltxt, rtxt))
        var pairs = max(h.la, h.lb)
        for k in range(pairs):
            var has_l = k < h.la
            var has_r = k < h.lb
            if has_l and has_r:
                var ltxt = old.lines[h.a + k]
                var rtxt = new.lines[h.b + k]
                rows.append(
                    DiffRow(DIFF_ROW_SPLIT_CHG, ltxt + " | " + rtxt, hi, h.a + k, h.b + k, ltxt, rtxt)
                )
            elif has_l:
                var ltxt = old.lines[h.a + k]
                rows.append(DiffRow(DIFF_ROW_OLD, "-" + ltxt, hi, h.a + k, -1, ltxt, ""))
            else:
                var rtxt = new.lines[h.b + k]
                rows.append(DiffRow(DIFF_ROW_NEW, "+" + rtxt, hi, -1, h.b + k, "", rtxt))
        for k in range(after):
            var ltxt = old.lines[h.a + h.la + k]
            var rtxt = new.lines[h.b + h.lb + k]
            rows.append(
                DiffRow(
                    DIFF_ROW_SAME, " " + ltxt, hi, h.a + h.la + k, h.b + h.lb + k, ltxt, rtxt
                )
            )
    return rows^


def _parse_int_prefix(s: String) -> Int:
    var b = s.as_bytes()
    var v = 0
    var digits = 0
    for i in range(len(b)):
        if b[i] >= 48 and b[i] <= 57:
            v = v * 10 + Int(b[i] - 48)
            digits += 1
        else:
            break
    return v if digits > 0 else 1


def parse_unified_diff(patch_text: String) -> List[DiffRow]:
    """Parses a standard unified diff string (`@@ -a,la +b,lb @@`) into `DiffRow`s."""
    var rows = List[DiffRow]()
    var cur_hunk = -1
    var old_ln = 0
    var new_ln = 0
    for raw in patch_text.split("\n"):
        var l = String(String(raw).rstrip("\r"))
        if l.startswith("@@ "):
            cur_hunk += 1
            var dash = l.find("-")
            var plus = l.find("+")
            if dash >= 0:
                old_ln = max(0, _parse_int_prefix(String(l[byte=dash + 1 : l.byte_length()])) - 1)
            if plus >= 0:
                new_ln = max(0, _parse_int_prefix(String(l[byte=plus + 1 : l.byte_length()])) - 1)
            rows.append(DiffRow(DIFF_ROW_HEADER, l, cur_hunk, old_ln, new_ln, l, l))
        elif l.startswith("---") or l.startswith("+++") or l.startswith("diff ") or l.startswith("index "):
            rows.append(DiffRow(DIFF_ROW_NOTE, l, -1, -1, -1, l, ""))
        elif cur_hunk >= 0 and l.startswith("-"):
            var body = String(l[byte=1 : l.byte_length()])
            rows.append(DiffRow(DIFF_ROW_OLD, l, cur_hunk, old_ln, new_ln, body, ""))
            old_ln += 1
        elif cur_hunk >= 0 and l.startswith("+"):
            var body = String(l[byte=1 : l.byte_length()])
            rows.append(DiffRow(DIFF_ROW_NEW, l, cur_hunk, old_ln, new_ln, "", body))
            new_ln += 1
        elif cur_hunk >= 0 and l.startswith(" "):
            var body = String(l[byte=1 : l.byte_length()])
            rows.append(DiffRow(DIFF_ROW_SAME, l, cur_hunk, old_ln, new_ln, body, body))
            old_ln += 1
            new_ln += 1
    return rows^


def apply_hunk(old: DiffSide, new: DiffSide, h: DiffHunk) -> DiffSide:
    """Returns `old` with a single hunk `h` from `new` applied."""
    var lines = List[String]()
    for k in range(min(h.a, len(old.lines))):
        lines.append(old.lines[k])
    for k in range(h.b, min(h.b + h.lb, len(new.lines))):
        lines.append(new.lines[k])
    for k in range(min(h.a + h.la, len(old.lines)), len(old.lines)):
        lines.append(old.lines[k])
    var eol = new.trailing_eol if h.a + h.la >= len(old.lines) else old.trailing_eol
    return DiffSide(lines^, eol)


def revert_hunk(old: DiffSide, new: DiffSide, h: DiffHunk) -> DiffSide:
    """Returns `new` with a single hunk `h` reverted back to `old`."""
    var lines = List[String]()
    for k in range(min(h.b, len(new.lines))):
        lines.append(new.lines[k])
    for k in range(h.a, min(h.a + h.la, len(old.lines))):
        lines.append(old.lines[k])
    for k in range(min(h.b + h.lb, len(new.lines)), len(new.lines)):
        lines.append(new.lines[k])
    var eol = old.trailing_eol if h.b + h.lb >= len(new.lines) else new.trailing_eol
    return DiffSide(lines^, eol)


def hunk_at_new_line(hunks: List[DiffHunk], line: Int) -> Int:
    """Returns the hunk index touching 0-based `line` in the new file, or -1."""
    for k in range(len(hunks)):
        ref h = hunks[k]
        if h.lb > 0 and line >= h.b and line < h.b + h.lb:
            return k
        if h.lb == 0 and (line == h.b or (line == h.b - 1 and k == len(hunks) - 1)):
            return k
    return -1


struct DiffDocument(Copyable, Movable):
    """Interactive diff document tracking per-hunk accept/reject decisions."""
    var path: String
    var old_side: DiffSide
    var new_side: DiffSide
    var hunks: List[DiffHunk]
    var decisions: List[Int]

    def __init__(out self, path: String = "", old_text: String = "", new_text: String = ""):
        self.path = path
        self.old_side = DiffSide.from_text(old_text)
        self.new_side = DiffSide.from_text(new_text)
        self.hunks = diff_sides(self.old_side, self.new_side)
        self.decisions = List[Int](length=len(self.hunks), fill=HUNK_PENDING)

    def hunk_count(self) -> Int:
        return len(self.hunks)

    def set_decision(mut self, hunk_idx: Int, decision: Int) -> Bool:
        if hunk_idx < 0 or hunk_idx >= len(self.decisions):
            return False
        self.decisions[hunk_idx] = decision
        return True

    def accept_hunk(mut self, hunk_idx: Int) -> Bool:
        return self.set_decision(hunk_idx, HUNK_ACCEPTED)

    def reject_hunk(mut self, hunk_idx: Int) -> Bool:
        return self.set_decision(hunk_idx, HUNK_REJECTED)

    def accept_all(mut self):
        for i in range(len(self.decisions)):
            self.decisions[i] = HUNK_ACCEPTED

    def reject_all(mut self):
        for i in range(len(self.decisions)):
            self.decisions[i] = HUNK_REJECTED

    def pending_count(self) -> Int:
        var n = 0
        for i in range(len(self.decisions)):
            if self.decisions[i] == HUNK_PENDING:
                n += 1
        return n

    def unified_rows(self, context: Int = DEFAULT_DIFF_CONTEXT) -> List[DiffRow]:
        return build_unified_diff(self.old_side, self.new_side, self.hunks, context)

    def split_rows(self, context: Int = DEFAULT_DIFF_CONTEXT) -> List[DiffRow]:
        return build_split_diff(self.old_side, self.new_side, self.hunks, context)

    def gutter_marks(self) -> List[UInt8]:
        return compute_gutter_marks(self.hunks, len(self.new_side.lines))

    def materialized_text(self, accept_pending: Bool = False) -> String:
        """Constructs resulting file text applying accepted (and optionally pending) hunks."""
        var out_lines = List[String]()
        var old_pos = 0
        var last_applied_end = False
        for i in range(len(self.hunks)):
            ref h = self.hunks[i]
            while old_pos < h.a and old_pos < len(self.old_side.lines):
                out_lines.append(self.old_side.lines[old_pos])
                old_pos += 1
            var take_new = self.decisions[i] == HUNK_ACCEPTED or (
                accept_pending and self.decisions[i] == HUNK_PENDING
            )
            if take_new:
                for k in range(h.b, min(h.b + h.lb, len(self.new_side.lines))):
                    out_lines.append(self.new_side.lines[k])
                if h.a + h.la >= len(self.old_side.lines):
                    last_applied_end = True
            else:
                for k in range(h.a, min(h.a + h.la, len(self.old_side.lines))):
                    out_lines.append(self.old_side.lines[k])
            old_pos = h.a + h.la
        while old_pos < len(self.old_side.lines):
            out_lines.append(self.old_side.lines[old_pos])
            old_pos += 1
        var eol = self.new_side.trailing_eol if last_applied_end else self.old_side.trailing_eol
        return DiffSide(out_lines^, eol).to_text()
