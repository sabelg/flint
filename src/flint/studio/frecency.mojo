"""Studio Frecency Store & Fuzzy Matcher (`flint.studio.frecency`).

Pure-Mojo 1.2 foundation for Flint Studio's universal Command Palette & Quick-Spawn:
  - `FrecencyStore`: Tracks how often and how recently commands, assets, or entities
    were accessed using 4h / 24h / 7d / 30d decay buckets, with safe line-based
    state persistence.
  - `fuzzy_score(query, candidate) -> FuzzyMatch`: Case-insensitive UTF-8 subsequence
    & boundary-aware fuzzy scorer returning both a ranking score and coalesced
    codepoint/byte match spans for crisp UI highlight rendering.
"""
from std.collections import List
from flint.platform.os import file_size, make_dirs, parent_directory, read_bytes, write_bytes
from flint.text.utf8 import decode_utf8

comptime FRECENCY_MAX_ENTRIES: Int = 500
comptime FRECENCY_4H_SEC: Int = 4 * 3600
comptime FRECENCY_24H_SEC: Int = 24 * 3600
comptime FRECENCY_7D_SEC: Int = 7 * 24 * 3600
comptime FRECENCY_30D_SEC: Int = 30 * 24 * 3600


@always_inline
def frecency_decay_weight(age_sec: Int) -> Int:
    """Recency bucket weight: 100 (<4h), 70 (<24h), 50 (<7d), 30 (<30d), 10 (older)."""
    if age_sec < FRECENCY_4H_SEC:
        return 100
    if age_sec < FRECENCY_24H_SEC:
        return 70
    if age_sec < FRECENCY_7D_SEC:
        return 50
    if age_sec < FRECENCY_30D_SEC:
        return 30
    return 10


@fieldwise_init
struct FrecencyEntry(Copyable, Movable):
    """Single key access record in `FrecencyStore`."""
    var key: String
    var count: Int
    var last_epoch_sec: Int

    def __init__(out self):
        self.key = ""
        self.count = 0
        self.last_epoch_sec = 0


def _sanitize_key(raw: String) -> String:
    """Strips control characters (CR/LF/NUL) so encoded lines stay one record per line."""
    var src = raw.as_bytes()
    var out = List[UInt8](capacity=len(src))
    for i in range(len(src)):
        var b = src[i]
        if b == 10 or b == 13 or b == 0:
            continue
        out.append(b)
    return String(String(unsafe_from_utf8=out).strip())


def _parse_nonneg_int(s: String) -> Tuple[Bool, Int]:
    var bytes = s.as_bytes()
    if len(bytes) == 0:
        return (False, 0)
    var val: Int = 0
    for i in range(len(bytes)):
        var b = Int(bytes[i])
        if b < 48 or b > 57:
            return (False, 0)
        if val > 922337203685477580:
            return (False, 0)
        val = val * 10 + (b - 48)
    return (True, val)


struct FrecencyStore(Copyable, Movable):
    """Bounded MRU-ordered frecency table for ranking commands, files, assets & entities."""
    var entries: List[FrecencyEntry]
    var max_entries: Int

    def __init__(out self, max_entries: Int = FRECENCY_MAX_ENTRIES):
        self.entries = List[FrecencyEntry]()
        self.max_entries = max(1, max_entries)

    def __init__(out self, *, copy: Self):
        self.entries = copy.entries.copy()
        self.max_entries = copy.max_entries

    def clear(mut self):
        self.entries.clear()

    def count(self) -> Int:
        return len(self.entries)

    def find(self, key: String) -> Int:
        for i in range(len(self.entries)):
            if self.entries[i].key == key:
                return i
        return -1

    def note(mut self, key: String, now_sec: Int):
        """Records an access of `key` at `now_sec` (Unix epoch seconds) and moves it to front."""
        var clean = _sanitize_key(key)
        if clean.byte_length() == 0:
            return
        var ts = max(0, now_sec)
        var idx = self.find(clean)
        var next_count = 1
        if idx >= 0:
            next_count = self.entries[idx].count + 1
            _ = self.entries.pop(idx)
        self.entries.insert(0, FrecencyEntry(clean, next_count, ts))
        while len(self.entries) > self.max_entries:
            _ = self.entries.pop()

    def forget(mut self, key: String):
        var idx = self.find(key)
        if idx >= 0:
            _ = self.entries.pop(idx)

    def score(self, key: String, now_sec: Int) -> Int:
        """Returns 0 for unseen keys, up to 1000 for frequently and recently used keys."""
        var idx = self.find(key)
        if idx < 0:
            return 0
        var age = max(0, now_sec - self.entries[idx].last_epoch_sec)
        return min(self.entries[idx].count, 10) * frecency_decay_weight(age)

    def recency_bonus(self, key: String, top_k: Int = 20, step: Int = 20) -> Int:
        """Additional rank boost for the `top_k` most recently used items."""
        var idx = self.find(key)
        if idx < 0 or idx >= top_k:
            return 0
        return (top_k - idx) * max(1, step)

    def encode(self) -> String:
        """Serializes entries as `"count last_epoch_sec key\\n"` lines."""
        var out = String("")
        for i in range(len(self.entries)):
            ref e = self.entries[i]
            out += String(e.count) + " " + String(e.last_epoch_sec) + " " + e.key + "\n"
        return out

    def decode(mut self, text: String, require_abs_path: Bool = False):
        """Loads entries written by `encode()`, skipping malformed or duplicate lines safely."""
        self.entries.clear()
        for raw_slice in text.split("\n"):
            var line = String(raw_slice)
            if line.byte_length() > 0 and line.as_bytes()[line.byte_length() - 1] == 13:
                var trimmed_cr = String(line[byte=0 : line.byte_length() - 1])
                line = trimmed_cr^
            var sp1 = line.find(" ")
            if sp1 <= 0:
                continue
            var rest = String(line[byte=sp1 + 1:])
            var sp2 = rest.find(" ")
            if sp2 <= 0 or sp2 + 1 >= rest.byte_length():
                continue
            var p_cnt = _parse_nonneg_int(String(line[byte=0:sp1]))
            var p_ts = _parse_nonneg_int(String(rest[byte=0:sp2]))
            if not p_cnt[0] or not p_ts[0] or p_cnt[1] <= 0:
                continue
            var key = _sanitize_key(String(rest[byte=sp2 + 1:]))
            if key.byte_length() == 0:
                continue
            if require_abs_path and not key.startswith("/"):
                continue
            if self.find(key) >= 0:
                continue
            if len(self.entries) >= self.max_entries:
                break
            self.entries.append(FrecencyEntry(key, p_cnt[1], p_ts[1]))

    def save_file(self, path: String) -> Bool:
        """Atomically writes the encoded frecency table to `path`, creating parent directories."""
        if path.byte_length() == 0:
            return False
        var pdir = parent_directory(path)
        if pdir.byte_length() > 0 and pdir != ".":
            _ = make_dirs(pdir)
        var text = self.encode()
        var sb = text.as_bytes()
        var raw = List[UInt8](capacity=len(sb))
        for i in range(len(sb)):
            raw.append(sb[i])
        return write_bytes(path, raw)

    def load_file(mut self, path: String, require_abs_path: Bool = False) -> Bool:
        """Reads and decodes a persisted frecency table from `path`."""
        if path.byte_length() == 0 or file_size(path) <= 0:
            return False
        var raw = read_bytes(path)
        if len(raw) == 0:
            return False
        self.decode(String(unsafe_from_utf8=raw), require_abs_path=require_abs_path)
        return True


# ============================================================================
# UTF-8 Boundary-Aware Fuzzy Matcher with Highlight Spans
# ============================================================================

@fieldwise_init
struct FuzzySpan(Copyable, ImplicitlyCopyable, Movable):
    """Half-open `[start, end)` match range (either codepoints or UTF-8 bytes)."""
    var start: Int
    var end: Int

    def __init__(out self):
        self.start = 0
        self.end = 0


struct FuzzyMatch(Copyable, Movable):
    """Result of `fuzzy_score(query, candidate)` with score and highlight spans."""
    var matched: Bool
    var score: Int
    var char_indices: List[Int]
    var spans: List[FuzzySpan]
    var byte_spans: List[FuzzySpan]

    def __init__(out self, matched: Bool = False, score: Int = -1):
        self.matched = matched
        self.score = score
        self.char_indices = List[Int]()
        self.spans = List[FuzzySpan]()
        self.byte_spans = List[FuzzySpan]()

    def __init__(out self, *, copy: Self):
        self.matched = copy.matched
        self.score = copy.score
        self.char_indices = copy.char_indices.copy()
        self.spans = copy.spans.copy()
        self.byte_spans = copy.byte_spans.copy()


@always_inline
def _ascii_lower_cp(cp: Int) -> Int:
    if cp >= 65 and cp <= 90:
        return cp + 32
    return cp


@always_inline
def _is_word_sep(cp: Int) -> Bool:
    return (
        cp == 32   # ' '
        or cp == 47  # '/'
        or cp == 95  # '_'
        or cp == 45  # '-'
        or cp == 46  # '.'
        or cp == 58  # ':'
        or cp == 92  # '\\'
    )


def _decode_with_byte_offsets(text: String) -> Tuple[List[Int], List[Int]]:
    """Decodes UTF-8 codepoints and records starting byte offset for each codepoint + end sentinel."""
    var codes = List[Int]()
    var offsets = List[Int]()
    var bytes = text.as_bytes()
    var i = 0
    var n = len(bytes)
    while i < n:
        offsets.append(i)
        var lead = Int(bytes[i])
        if lead < 0x80:
            codes.append(lead)
            i += 1
            continue
        var extra: Int
        var code: Int
        var minimum: Int
        if lead >= 0xC2 and lead < 0xE0:
            extra = 1
            code = lead & 0x1F
            minimum = 0x80
        elif lead >= 0xE0 and lead < 0xF0:
            extra = 2
            code = lead & 0x0F
            minimum = 0x800
        elif lead >= 0xF0 and lead < 0xF5:
            extra = 3
            code = lead & 0x07
            minimum = 0x10000
        else:
            codes.append(0xFFFD)
            i += 1
            continue
        var ok = True
        for step in range(1, extra + 1):
            if i + step >= n:
                ok = False
                break
            var nx = Int(bytes[i + step])
            if (nx & 0xC0) != 0x80:
                ok = False
                break
            code = (code << 6) | (nx & 0x3F)
        if ok and (code < minimum or code > 0x10FFFF or (code >= 0xD800 and code <= 0xDFFF)):
            ok = False
        if not ok:
            codes.append(0xFFFD)
            i += 1
        else:
            codes.append(code)
            i += extra + 1
    offsets.append(n)
    return (codes^, offsets^)


def _build_spans(indices: List[Int], byte_offsets: List[Int]) -> Tuple[List[FuzzySpan], List[FuzzySpan]]:
    """Coalesces sorted codepoint match indices into contiguous codepoint and byte spans."""
    var cp_spans = List[FuzzySpan]()
    var b_spans = List[FuzzySpan]()
    var n = len(indices)
    if n == 0:
        return (cp_spans^, b_spans^)
    var run_start = indices[0]
    var run_prev = indices[0]
    for i in range(1, n):
        var cur = indices[i]
        if cur == run_prev + 1:
            run_prev = cur
        else:
            cp_spans.append(FuzzySpan(run_start, run_prev + 1))
            if run_start >= 0 and run_prev + 1 < len(byte_offsets):
                b_spans.append(FuzzySpan(byte_offsets[run_start], byte_offsets[run_prev + 1]))
            run_start = cur
            run_prev = cur
    cp_spans.append(FuzzySpan(run_start, run_prev + 1))
    if run_start >= 0 and run_prev + 1 < len(byte_offsets):
        b_spans.append(FuzzySpan(byte_offsets[run_start], byte_offsets[run_prev + 1]))
    return (cp_spans^, b_spans^)


def fuzzy_score(query: String, candidate: String) -> FuzzyMatch:
    """Matches `query` against `candidate` (case-insensitive UTF-8).

    Scoring tiers:
      - Empty query: `matched=True, score=1`
      - Exact match (or exact match before `"  "`): `2200`
      - Contiguous substring match: `1000 - start_pos + boundary_bonus` (up to `1050`)
      - Subsequence match with word/path/camelCase bonuses: `[2, 900]`
      - No match: `matched=False, score=-1`
    """
    var q_raw = decode_utf8(query)
    if len(q_raw) == 0:
        return FuzzyMatch(matched=True, score=1)

    var cand_dec = _decode_with_byte_offsets(candidate)
    var c_raw = cand_dec[0].copy()
    var c_offs = cand_dec[1].copy()
    var nc = len(c_raw)
    var nq = len(q_raw)
    if nc < nq:
        return FuzzyMatch(matched=False, score=-1)

    var c_low = List[Int](capacity=nc)
    for i in range(nc):
        c_low.append(_ascii_lower_cp(c_raw[i]))
    var q_low = List[Int](capacity=nq)
    for i in range(nq):
        q_low.append(_ascii_lower_cp(q_raw[i]))

    # 1. Exact match or exact prefix before double-space ("Name  path/to/file")
    var exact_prefix = True
    for i in range(nq):
        if c_low[i] != q_low[i]:
            exact_prefix = False
            break
    if exact_prefix and (nc == nq or (nc >= nq + 2 and c_low[nq] == 32 and c_low[nq + 1] == 32)):
        var res = FuzzyMatch(matched=True, score=2200)
        for i in range(nq):
            res.char_indices.append(i)
        var built = _build_spans(res.char_indices, c_offs)
        res.spans = built[0].copy()
        res.byte_spans = built[1].copy()
        return res^

    # 2. Best contiguous substring match (preferring start / word-boundary hits)
    var best_sub_pos = -1
    var best_sub_score = -1
    for start in range(nc - nq + 1):
        var ok = True
        for k in range(nq):
            if c_low[start + k] != q_low[k]:
                ok = False
                break
        if ok:
            var bonus = 0
            if start == 0:
                bonus = 50
            elif _is_word_sep(c_raw[start - 1]):
                bonus = 40
            elif (
                c_raw[start] >= 65
                and c_raw[start] <= 90
                and c_raw[start - 1] >= 97
                and c_raw[start - 1] <= 122
            ):
                bonus = 35
            var sc = 1000 - start + bonus
            if sc > best_sub_score:
                best_sub_score = sc
                best_sub_pos = start
    if best_sub_pos >= 0:
        var res = FuzzyMatch(matched=True, score=max(2, best_sub_score))
        for k in range(nq):
            res.char_indices.append(best_sub_pos + k)
        var built = _build_spans(res.char_indices, c_offs)
        res.spans = built[0].copy()
        res.byte_spans = built[1].copy()
        return res^

    # 3. Subsequence match with word/path/camelCase boundary & adjacency scoring
    var matched_idx = List[Int](capacity=nq)
    var qi = 0
    var score = 500
    var last = -1
    for i in range(nc):
        if qi < nq and c_low[i] == q_low[qi]:
            if i == 0 or _is_word_sep(c_raw[i - 1]):
                score += 20
            elif c_raw[i] >= 65 and c_raw[i] <= 90 and c_raw[i - 1] >= 97 and c_raw[i - 1] <= 122:
                score += 20
            if last >= 0:
                if i == last + 1:
                    score += 10
                else:
                    score -= i - last
            matched_idx.append(i)
            last = i
            qi += 1
    if qi < nq:
        return FuzzyMatch(matched=False, score=-1)

    var res = FuzzyMatch(matched=True, score=min(900, max(score, 2)))
    res.char_indices = matched_idx^
    var built = _build_spans(res.char_indices, c_offs)
    res.spans = built[0].copy()
    res.byte_spans = built[1].copy()
    return res^
