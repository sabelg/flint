"""Grammar-constrained decoding (`flint.ml.llm.constrain`).

A `JsonGrammar` is a sequence of byte segments: fixed literals, a choice between fixed strings
(an enum) and free JSON string content (no quote, backslash or control bytes, at most `max_len`
bytes). `JsonGrammar.object_of_fields` builds `{"k1": "v1", "k2": "v2"}` from field specs, the
shape an NPC answers in: `{"action": "<one of the BT actions>", "target": "<one of the
targets>", "say": "<free text>"}`.

`GrammarMatcher` tracks the set of grammar positions consistent with the bytes emitted so far
(an enum branches per option; a string may end at any byte). `mask(tok)` marks every vocabulary
token whose bytes can be consumed from the current positions, so sampling from the masked
logits can only ever produce a sentence of the grammar; the end-of-turn tokens are allowed only
once the grammar is complete. Tokens made only of plain string bytes take a fast path inside a
string field (their length decides), so a mask over a 150k vocabulary costs a few milliseconds.
"""

from flint.ml.llm.tokenizer import BpeTokenizer

comptime SEG_LIT: Int = 0
comptime SEG_ENUM: Int = 1
comptime SEG_STR: Int = 2
comptime _NONE: Int = 0xFFFF


@always_inline
def _pack(seg: Int, opt: Int, off: Int) -> Int:
    return (seg << 40) | (opt << 20) | off


@always_inline
def _seg(s: Int) -> Int:
    return s >> 40


@always_inline
def _opt(s: Int) -> Int:
    return (s >> 20) & 0xFFFFF


@always_inline
def _off(s: Int) -> Int:
    return s & 0xFFFFF


@always_inline
def json_string_byte(b: UInt8) -> Bool:
    return b >= 32 and b != 34 and b != 92


def _bytes(s: String) -> List[UInt8]:
    return List[UInt8](s.as_bytes())


struct JsonGrammar(Copyable, Movable):
    var kinds: List[Int]
    var lits: List[List[UInt8]]
    var options: List[List[List[UInt8]]]
    var max_len: List[Int]

    def __init__(out self):
        self.kinds = List[Int]()
        self.lits = List[List[UInt8]]()
        self.options = List[List[List[UInt8]]]()
        self.max_len = List[Int]()

    def segments(self) -> Int:
        return len(self.kinds)

    def literal(mut self, s: String):
        if len(self.kinds) > 0 and self.kinds[len(self.kinds) - 1] == SEG_LIT:
            for b in s.as_bytes():
                self.lits[len(self.lits) - 1].append(b)
            return
        self.kinds.append(SEG_LIT)
        self.lits.append(_bytes(s))
        self.options.append(List[List[UInt8]]())
        self.max_len.append(0)

    def choice(mut self, opts: List[String]) raises:
        var o = List[List[UInt8]]()
        for s in opts:
            if s.byte_length() == 0:
                raise Error("JsonGrammar: empty enum option")
            for b in s.as_bytes():
                if not json_string_byte(b):
                    raise Error("JsonGrammar: enum option '" + s + "' needs JSON escaping")
            o.append(_bytes(s))
        self.kinds.append(SEG_ENUM)
        self.lits.append(List[UInt8]())
        self.options.append(o^)
        self.max_len.append(0)

    def string(mut self, max_len: Int):
        """Free string content; the caller follows it with a literal starting with a quote."""
        self.kinds.append(SEG_STR)
        self.lits.append(List[UInt8]())
        self.options.append(List[List[UInt8]]())
        self.max_len.append(max_len)

    @staticmethod
    def object_of_fields(names: List[String], enums: List[List[String]], max_len: List[Int]) raises -> JsonGrammar:
        """`{"n0": "v0", "n1": "v1", ...}`: field i is an enum of `enums[i]` when that list is
        non-empty, else a free string of at most `max_len[i]` bytes."""
        var g = JsonGrammar()
        g.literal("{")
        for i in range(len(names)):
            g.literal(("" if i == 0 else ", ") + "\"" + names[i] + "\": \"")
            if len(enums[i]) > 0:
                g.choice(enums[i])
            else:
                g.string(max_len[i])
            g.literal("\"")
        g.literal("}")
        return g^


struct GrammarMatcher(Movable):
    var g: JsonGrammar
    var states: List[Int]
    var emitted: List[UInt8]
    var plain: List[Bool]       # per token: non-empty, only JSON string bytes
    var special: List[Bool]     # per token: an added / control token

    def __init__(out self, var g: JsonGrammar):
        self.g = g^
        self.states = List[Int]()
        self.emitted = List[UInt8]()
        self.plain = List[Bool]()
        self.special = List[Bool]()
        self.reset()

    def reset(mut self):
        var start = List[Int]()
        self._close(_pack(0, _NONE, 0), start)
        self.states = start^
        self.emitted = List[UInt8]()

    def _close(self, s: Int, mut out: List[Int]):
        var seg = _seg(s)
        if seg >= len(self.g.kinds):
            _add(out, _pack(len(self.g.kinds), 0, 0))
            return
        var kind = self.g.kinds[seg]
        if kind == SEG_LIT:
            if _off(s) >= len(self.g.lits[seg]):
                self._close(_pack(seg + 1, _NONE, 0), out)
            else:
                _add(out, _pack(seg, 0, _off(s)))
            return
        if kind == SEG_ENUM:
            var opt = _opt(s)
            if opt == _NONE:
                for o in range(len(self.g.options[seg])):
                    self._close(_pack(seg, o, 0), out)
                return
            if _off(s) >= len(self.g.options[seg][opt]):
                self._close(_pack(seg + 1, _NONE, 0), out)
            else:
                _add(out, s)
            return
        _add(out, _pack(seg, 0, _off(s)))
        self._close(_pack(seg + 1, _NONE, 0), out)

    def _step(self, s: Int, b: UInt8, mut out: List[Int]):
        var seg = _seg(s)
        if seg >= len(self.g.kinds):
            return
        var kind = self.g.kinds[seg]
        var off = _off(s)
        if kind == SEG_LIT:
            if self.g.lits[seg][off] == b:
                self._close(_pack(seg, 0, off + 1), out)
        elif kind == SEG_ENUM:
            var opt = _opt(s)
            if self.g.options[seg][opt][off] == b:
                self._close(_pack(seg, opt, off + 1), out)
        else:
            if json_string_byte(b) and off < self.g.max_len[seg]:
                self._close(_pack(seg, 0, off + 1), out)

    def _advance(self, cur: List[Int], b: UInt8, mut out: List[Int]):
        out.clear()
        for s in cur:
            self._step(s, b, out)

    def accepts(self, bytes: List[UInt8]) -> Bool:
        """Whether `bytes` can follow what was emitted."""
        var cur = self.states.copy()
        var nxt = List[Int]()
        for b in bytes:
            self._advance(cur, b, nxt)
            if len(nxt) == 0:
                return False
            swap(cur, nxt)
        return True

    def feed(mut self, bytes: List[UInt8]) raises:
        var nxt = List[Int]()
        for b in bytes:
            self._advance(self.states, b, nxt)
            if len(nxt) == 0:
                raise Error("GrammarMatcher: bytes outside the grammar")
            self.states = nxt.copy()
            self.emitted.append(b)

    def complete(self) -> Bool:
        """The grammar can end here."""
        for s in self.states:
            if _seg(s) >= len(self.g.kinds):
                return True
        return False

    def done(self) -> Bool:
        """Complete and nothing more can follow."""
        return len(self.states) == 1 and self.complete()

    def text(self) -> String:
        return String(unsafe_from_utf8=self.emitted.copy())

    def _prepare(mut self, tok: BpeTokenizer):
        if len(self.plain) == tok.vocab_size:
            return
        self.plain = List[Bool](length=tok.vocab_size, fill=False)
        self.special = List[Bool](length=tok.vocab_size, fill=False)
        for a in tok.added_ids:
            self.special[a] = True
        for t in range(tok.vocab_size):
            ref bs = tok.id_bytes[t]
            if len(bs) == 0:
                self.special[t] = True
                continue
            var ok = True
            for b in bs:
                if not json_string_byte(b):
                    ok = False
                    break
            self.plain[t] = ok and not self.special[t]

    def mask(mut self, tok: BpeTokenizer, end_ids: List[Int]) -> List[Bool]:
        """Allowed[token] for the next step; `end_ids` (end of turn) only when complete."""
        self._prepare(tok)
        var V = tok.vocab_size
        var allowed = List[Bool](length=V, fill=False)
        if self.complete():
            for e in end_ids:
                if e >= 0 and e < V:
                    allowed[e] = True
        # first-byte filter
        var first = List[Bool](length=256, fill=False)
        var nxt = List[Int]()
        for b in range(256):
            self._advance(self.states, UInt8(b), nxt)
            first[b] = len(nxt) > 0
        # a string field: plain tokens are allowed when they fit
        var str_room = -1
        for s in self.states:
            var seg = _seg(s)
            if seg < len(self.g.kinds) and self.g.kinds[seg] == SEG_STR:
                str_room = max(str_room, self.g.max_len[seg] - _off(s))
        var cur = List[Int]()
        var n2 = List[Int]()
        for t in range(V):
            if self.special[t]:
                continue
            ref bs = tok.id_bytes[t]
            if not first[Int(bs[0])]:
                continue
            if self.plain[t] and str_room >= 0:
                allowed[t] = len(bs) <= str_room
                continue
            cur.clear()
            for s in self.states:
                cur.append(s)
            var ok = True
            for b in bs:
                self._advance(cur, b, n2)
                if len(n2) == 0:
                    ok = False
                    break
                swap(cur, n2)
            allowed[t] = ok
        return allowed^


def _add(mut out: List[Int], s: Int):
    for x in out:
        if x == s:
            return
    out.append(s)
