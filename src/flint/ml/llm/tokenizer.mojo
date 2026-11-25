"""Byte-level BPE tokenizer read from a Hugging Face `tokenizer.json` (`flint.ml.llm.tokenizer`).

Covers the GPT-2 style byte-level BPE used by Qwen2/Qwen3 and Llama 3 / SmolLM3:

  1. Added tokens (`<|im_start|>`, `<think>`, ...) are split out first (longest match), unless
     `allow_special=False`, which treats their text as plain text (player input cannot inject
     control tokens).
  2. Pre-tokenization reproduces the split regex
     `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,D}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`
     with hand-written matchers (D = 1 for Qwen, 3 for Llama 3; read from the regex).
     Unicode letter / number classes are exact for ASCII and Latin-1 and table-driven for the
     common punctuation, symbol, digit and combining-mark blocks; other code points count as
     letters. Input is assumed to be NFC already (the Qwen normalizer).
  3. Each pre-token's UTF-8 bytes start as one symbol per byte and merge by rank (`merges`).

Decoding concatenates each id's raw bytes (byte-level alphabet mapped back to bytes).
"""

from std.collections import Dict
from flint.assets.json import parse_json, JsonDocument
from flint.platform.os import read_bytes

comptime _PAIR_SHIFT: Int = 18     # vocab ids < 262144


def _bytes_to_unicode() -> List[Int]:
    """GPT-2's byte -> code point table for the byte-level alphabet."""
    var table = List[Int](length=256, fill=-1)
    for b in range(33, 127):
        table[b] = b
    for b in range(161, 173):
        table[b] = b
    for b in range(174, 256):
        table[b] = b
    var n = 0
    for b in range(256):
        if table[b] < 0:
            table[b] = 256 + n
            n += 1
    return table^


def utf8_decode(data: List[UInt8], i: Int) -> Tuple[Int, Int]:
    """(code point, byte length) at byte i; invalid bytes decode as themselves with length 1."""
    var b0 = Int(data[i])
    if b0 < 0x80:
        return (b0, 1)
    var n = len(data)
    if b0 >= 0xC0 and b0 < 0xE0 and i + 1 < n:
        return (((b0 & 0x1F) << 6) | (Int(data[i + 1]) & 0x3F), 2)
    if b0 >= 0xE0 and b0 < 0xF0 and i + 2 < n:
        return (((b0 & 0x0F) << 12) | ((Int(data[i + 1]) & 0x3F) << 6) | (Int(data[i + 2]) & 0x3F), 3)
    if b0 >= 0xF0 and i + 3 < n:
        return (((b0 & 0x07) << 18) | ((Int(data[i + 1]) & 0x3F) << 12) | ((Int(data[i + 2]) & 0x3F) << 6)
                | (Int(data[i + 3]) & 0x3F), 4)
    return (b0, 1)


def utf8_append(mut out: List[UInt8], cp: Int):
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


# ---------------------------------------------------------------- character classes

comptime CC_LETTER: Int = 0
comptime CC_NUMBER: Int = 1
comptime CC_SPACE: Int = 2
comptime CC_OTHER: Int = 3


def is_space(cp: Int) -> Bool:
    if cp == 32 or (cp >= 9 and cp <= 13):
        return True
    if cp < 0x85:
        return False
    return (cp == 0x85 or cp == 0xA0 or cp == 0x1680 or (cp >= 0x2000 and cp <= 0x200A) or cp == 0x2028
            or cp == 0x2029 or cp == 0x202F or cp == 0x205F or cp == 0x3000)


def char_class(cp: Int) -> Int:
    """\\p{L} / \\p{N} / \\s / other for the pre-tokenizer."""
    if cp < 128:
        if (cp >= 65 and cp <= 90) or (cp >= 97 and cp <= 122):
            return CC_LETTER
        if cp >= 48 and cp <= 57:
            return CC_NUMBER
        if cp == 32 or (cp >= 9 and cp <= 13):
            return CC_SPACE
        return CC_OTHER
    if is_space(cp):
        return CC_SPACE
    if cp < 0x100:
        if cp == 0xAA or cp == 0xB5 or cp == 0xBA:
            return CC_LETTER
        if cp == 0xB2 or cp == 0xB3 or cp == 0xB9 or (cp >= 0xBC and cp <= 0xBE):
            return CC_NUMBER
        if cp < 0xC0 or cp == 0xD7 or cp == 0xF7:
            return CC_OTHER
        return CC_LETTER
    # combining marks (\p{M}) and modifier symbols
    if (cp >= 0x2C2 and cp <= 0x2C5) or (cp >= 0x2D2 and cp <= 0x2DF) or (cp >= 0x2E5 and cp <= 0x2EB) \
            or cp == 0x2ED or (cp >= 0x2EF and cp <= 0x36F) or (cp >= 0x483 and cp <= 0x489) \
            or (cp >= 0x591 and cp <= 0x5C7) or (cp >= 0x610 and cp <= 0x61A) or (cp >= 0x64B and cp <= 0x65F) \
            or cp == 0x670 or (cp >= 0x6D6 and cp <= 0x6ED) or (cp >= 0x900 and cp <= 0x903) \
            or (cp >= 0x93A and cp <= 0x93C) or (cp >= 0x93E and cp <= 0x94F) or (cp >= 0x951 and cp <= 0x957) \
            or (cp >= 0x962 and cp <= 0x963) or (cp >= 0xE31 and cp <= 0xE3A and cp != 0xE32 and cp != 0xE33) \
            or (cp >= 0xE47 and cp <= 0xE4E) or (cp >= 0x1AB0 and cp <= 0x1AFF) or (cp >= 0x1DC0 and cp <= 0x1DFF) \
            or (cp >= 0x20D0 and cp <= 0x20FF) or (cp >= 0x3099 and cp <= 0x309C) or (cp >= 0xFE00 and cp <= 0xFE0F) \
            or (cp >= 0xFE20 and cp <= 0xFE2F) or (cp >= 0xE0100 and cp <= 0xE01EF):
        return CC_OTHER
    # decimal digits of other scripts and number forms
    if (cp >= 0x660 and cp <= 0x669) or (cp >= 0x6F0 and cp <= 0x6F9) or (cp >= 0x966 and cp <= 0x96F) \
            or (cp >= 0x9E6 and cp <= 0x9EF) or (cp >= 0xE50 and cp <= 0xE59) or cp == 0x2070 \
            or (cp >= 0x2074 and cp <= 0x2079) or (cp >= 0x2080 and cp <= 0x2089) or (cp >= 0x2150 and cp <= 0x2182) \
            or (cp >= 0x2185 and cp <= 0x2189) or (cp >= 0x2460 and cp <= 0x249B) or (cp >= 0x24EA and cp <= 0x24FF) \
            or (cp >= 0x2776 and cp <= 0x2793) or cp == 0x3007 or (cp >= 0x3021 and cp <= 0x3029) \
            or (cp >= 0xFF10 and cp <= 0xFF19):
        return CC_NUMBER
    # punctuation and symbol blocks
    if (cp >= 0x2000 and cp <= 0x2BFF) or (cp >= 0x2E00 and cp <= 0x2E7F) \
            or (cp >= 0x3000 and cp <= 0x303F and cp != 0x3005 and cp != 0x3006) or (cp >= 0x3200 and cp <= 0x33FF) \
            or (cp >= 0xFE10 and cp <= 0xFE1F) or (cp >= 0xFE30 and cp <= 0xFE6F) or (cp >= 0xFF00 and cp <= 0xFF0F) \
            or (cp >= 0xFF1A and cp <= 0xFF20) or (cp >= 0xFF3B and cp <= 0xFF40) or (cp >= 0xFF5B and cp <= 0xFF65) \
            or (cp >= 0xFFE0 and cp <= 0xFFFF) or (cp >= 0x1F000 and cp <= 0x1FBFF) or cp == 0x60C or cp == 0x61B \
            or cp == 0x61F or (cp >= 0x66A and cp <= 0x66D) or cp == 0x6D4 or cp == 0x964 or cp == 0x965 \
            or cp == 0x5BE or cp == 0x5C0 or cp == 0x5C3 or cp == 0x5C6 or cp == 0x5F3 or cp == 0x5F4 \
            or cp == 0x37E or cp == 0x387 or (cp >= 0x55A and cp <= 0x55F) or cp == 0x589 or cp == 0xE3F \
            or cp == 0xE4F or cp == 0xE5A or cp == 0xE5B \
            or (cp >= 0x200B and cp <= 0x200F):
        return CC_OTHER
    return CC_LETTER


@fieldwise_init
struct _Cps(Copyable, Movable):
    """Code points of a text segment with each one's byte offset (plus an end sentinel)."""
    var cp: List[Int]
    var cls: List[Int]
    var off: List[Int]


def _split_cps(data: List[UInt8], start: Int, end: Int) -> _Cps:
    var cps = _Cps(List[Int](), List[Int](), List[Int]())
    var i = start
    while i < end:
        var d = utf8_decode(data, i)
        cps.cp.append(d[0])
        cps.cls.append(char_class(d[0]))
        cps.off.append(i)
        i += d[1]
    cps.off.append(end)
    return cps^


def _is_crlf(cp: Int) -> Bool:
    return cp == 10 or cp == 13


def _match_at(c: _Cps, i: Int, digits: Int) -> Int:
    """End (code point index, exclusive) of the pre-token starting at i."""
    var n = len(c.cp)
    var cp = c.cp[i]
    # 1. contractions
    if cp == 39 and i + 1 < n:
        var a = c.cp[i + 1] | 32
        if a == 115 or a == 116 or a == 109 or a == 100:     # s t m d
            return i + 2
        if i + 2 < n:
            var b = c.cp[i + 2] | 32
            if (a == 114 and b == 101) or (a == 118 and b == 101) or (a == 108 and b == 108):   # re ve ll
                return i + 3
    # 2. [^\r\n\p{L}\p{N}]?\p{L}+
    var j = i
    if c.cls[j] != CC_LETTER and c.cls[j] != CC_NUMBER and not _is_crlf(cp):
        j += 1
    if j < n and c.cls[j] == CC_LETTER:
        while j < n and c.cls[j] == CC_LETTER:
            j += 1
        return j
    # 3. \p{N}{1,digits}
    if c.cls[i] == CC_NUMBER:
        j = i
        while j < n and c.cls[j] == CC_NUMBER and j - i < digits:
            j += 1
        return j
    # 4. ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    j = i
    if cp == 32:
        j += 1
    if j < n and c.cls[j] == CC_OTHER:
        while j < n and c.cls[j] == CC_OTHER:
            j += 1
        while j < n and _is_crlf(c.cp[j]):
            j += 1
        return j
    # 5. \s*[\r\n]+   6. \s+(?!\S)   7. \s+
    if c.cls[i] == CC_SPACE:
        var e = i
        while e < n and c.cls[e] == CC_SPACE:
            e += 1
        var last_nl = -1
        for k in range(i, e):
            if _is_crlf(c.cp[k]):
                last_nl = k
        if last_nl >= 0:
            return last_nl + 1
        if e == n:
            return e
        if e - i >= 2:
            return e - 1
        return e
    return i + 1


struct BpeTokenizer(Movable):
    var vocab: Dict[String, Int]          # byte-level string -> id
    var id_bytes: List[List[UInt8]]       # id -> raw bytes
    var byte_id: List[Int]                # raw byte -> single-byte token id
    var merges: Dict[Int, Int]            # (a << 18 | b) -> (rank << 18 | merged)
    var added: List[String]               # added token contents
    var added_ids: List[Int]
    var added_special: List[Bool]
    var digits: Int
    var vocab_size: Int
    var cache: Dict[String, List[Int]]

    def __init__(out self):
        self.vocab = Dict[String, Int]()
        self.id_bytes = List[List[UInt8]]()
        self.byte_id = List[Int](length=256, fill=-1)
        self.merges = Dict[Int, Int]()
        self.added = List[String]()
        self.added_ids = List[Int]()
        self.added_special = List[Bool]()
        self.digits = 1
        self.vocab_size = 0
        self.cache = Dict[String, List[Int]]()

    @staticmethod
    def load(path: String) raises -> BpeTokenizer:
        var data = read_bytes(path)
        if len(data) == 0:
            raise Error("tokenizer: cannot read " + path)
        return BpeTokenizer.from_json(data)

    @staticmethod
    def from_json(data: List[UInt8]) raises -> BpeTokenizer:
        var doc = parse_json(data)
        if not doc.valid:
            raise Error("tokenizer: invalid JSON")
        var tok = BpeTokenizer()
        var root = doc.root
        var model = doc.object_get(root, "model")
        if doc.text(doc.object_get(model, "type")) != "BPE":
            raise Error("tokenizer: only BPE models are supported")
        # the digit run length of the split regex: \p{N}{1,3} (Llama 3) or \p{N} (Qwen)
        var pre = doc.object_get(root, "pre_tokenizer")
        tok.digits = 1
        if _find_regex(doc, pre).find("\\p{N}{1,3}") >= 0:
            tok.digits = 3
        var bu = _bytes_to_unicode()
        var inv = Dict[Int, Int]()
        for b in range(256):
            inv[bu[b]] = b
        var vnode = doc.object_get(model, "vocab")
        var nv = len(doc.nodes[vnode].keys)
        var max_id = 0
        var ids = List[Int](capacity=nv)
        for k in range(nv):
            var id = Int(doc.number(doc.nodes[vnode].children[k]))
            ids.append(id)
            max_id = max(max_id, id)
        var at = doc.object_get(root, "added_tokens")
        for k in range(doc.array_size(at)):
            var nd = doc.at(at, k)
            max_id = max(max_id, Int(doc.number(doc.object_get(nd, "id"))))
        tok.vocab_size = max_id + 1
        if tok.vocab_size >= 1 << _PAIR_SHIFT:
            raise Error("tokenizer: vocabulary too large")
        tok.id_bytes = List[List[UInt8]](length=tok.vocab_size, fill=List[UInt8]())
        for k in range(nv):
            var key = doc.nodes[vnode].keys[k]
            tok.vocab[key] = ids[k]
            var raw = List[UInt8]()
            var kb = List[UInt8](key.as_bytes())
            var i = 0
            while i < len(kb):
                var d = utf8_decode(kb, i)
                var b = inv.get(d[0])
                if b:
                    raw.append(UInt8(b.value()))
                else:
                    utf8_append(raw, d[0])
                i += d[1]
            if len(raw) == 1 and len(kb) <= 2:
                tok.byte_id[Int(raw[0])] = ids[k]
            tok.id_bytes[ids[k]] = raw^
        for b in range(256):
            if tok.byte_id[b] < 0:
                raise Error("tokenizer: no single-byte token for byte " + String(b) + " (not byte-level BPE?)")
        var mnode = doc.object_get(model, "merges")
        for r in range(doc.array_size(mnode)):
            var m = doc.at(mnode, r)
            var a: String
            var b: String
            if doc.nodes[m].kind == 4:
                a = doc.text(doc.at(m, 0))
                b = doc.text(doc.at(m, 1))
            else:
                var s = doc.text(m)
                var sp = s.find(" ")
                if sp < 0:
                    continue
                a = String(s[byte=0:sp])
                b = String(s[byte=sp + 1:])
            var ia = tok.vocab.get(a)
            var ib = tok.vocab.get(b)
            var im = tok.vocab.get(a + b)
            if not ia or not ib or not im:
                continue
            var key = (ia.value() << _PAIR_SHIFT) | ib.value()
            if key not in tok.merges:
                tok.merges[key] = (r << _PAIR_SHIFT) | im.value()
        for k in range(doc.array_size(at)):
            var nd = doc.at(at, k)
            var content = doc.text(doc.object_get(nd, "content"))
            var id = Int(doc.number(doc.object_get(nd, "id")))
            tok.added.append(content)
            tok.added_ids.append(id)
            tok.added_special.append(doc.boolean(doc.object_get(nd, "special")))
            tok.id_bytes[id] = List[UInt8](content.as_bytes())
        return tok^

    def token_id(self, text: String) -> Int:
        """Id of an added token or vocabulary entry with exactly this text, else -1."""
        for k in range(len(self.added)):
            if self.added[k] == text:
                return self.added_ids[k]
        var v = self.vocab.get(text)
        if v:
            return v.value()
        return -1

    def is_added(self, id: Int) -> Bool:
        for a in self.added_ids:
            if a == id:
                return True
        return False

    def _bpe(mut self, data: List[UInt8], start: Int, end: Int, mut out: List[Int]) raises:
        var syms = List[Int](capacity=end - start)
        for i in range(start, end):
            syms.append(self.byte_id[Int(data[i])])
        while len(syms) > 1:
            var best = -1
            var best_rank = Int.MAX
            var best_id = 0
            for i in range(len(syms) - 1):
                var m = self.merges.get((syms[i] << _PAIR_SHIFT) | syms[i + 1])
                if m:
                    var rank = m.value() >> _PAIR_SHIFT
                    if rank < best_rank:
                        best_rank = rank
                        best = i
                        best_id = m.value() & ((1 << _PAIR_SHIFT) - 1)
            if best < 0:
                break
            var a = syms[best]
            var b = syms[best + 1]
            var merged = List[Int](capacity=len(syms))
            var i = 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                    merged.append(best_id)
                    i += 2
                else:
                    merged.append(syms[i])
                    i += 1
            syms = merged^
        for s in syms:
            out.append(s)

    def _encode_plain(mut self, data: List[UInt8], start: Int, end: Int, mut out: List[Int]) raises:
        if end <= start:
            return
        var c = _split_cps(data, start, end)
        var i = 0
        var n = len(c.cp)
        while i < n:
            var j = _match_at(c, i, self.digits)
            var b0 = c.off[i]
            var b1 = c.off[j]
            var piece = List[UInt8](capacity=b1 - b0)
            for b in range(b0, b1):
                piece.append(data[b])
            var key = String(unsafe_from_utf8=piece)
            var hit = self.cache.get(key)
            if hit:
                for t in hit.value():
                    out.append(t)
            else:
                var ids = List[Int]()
                self._bpe(data, b0, b1, ids)
                for t in ids:
                    out.append(t)
                if len(self.cache) < 65536:
                    self.cache[key] = ids^
            i = j

    def encode(mut self, text: String, allow_special: Bool = True) raises -> List[Int]:
        var data = List[UInt8](text.as_bytes())
        var out = List[Int]()
        var seg = 0
        var i = 0
        var n = len(data)
        while i < n:
            var best = -1
            var best_len = 0
            if allow_special:
                for k in range(len(self.added)):
                    var a = self.added[k].as_bytes()
                    var la = len(a)
                    if la > best_len and i + la <= n and data[i] == a[0]:
                        var ok = True
                        for q in range(1, la):
                            if data[i + q] != a[q]:
                                ok = False
                                break
                        if ok:
                            best = k
                            best_len = la
            if best >= 0:
                self._encode_plain(data, seg, i, out)
                out.append(self.added_ids[best])
                i += best_len
                seg = i
            else:
                i += 1
        self._encode_plain(data, seg, n, out)
        return out^

    def decode_bytes(self, ids: List[Int]) -> List[UInt8]:
        var out = List[UInt8]()
        for t in ids:
            if t >= 0 and t < len(self.id_bytes):
                for b in self.id_bytes[t]:
                    out.append(b)
        return out^

    def decode(self, ids: List[Int]) -> String:
        return String(unsafe_from_utf8=self.decode_bytes(ids))

    def piece(self, id: Int) -> List[UInt8]:
        if id < 0 or id >= len(self.id_bytes):
            return List[UInt8]()
        return self.id_bytes[id].copy()


def _find_regex(doc: JsonDocument, node: Int) -> String:
    """The first `Split` regex string under a pre_tokenizer node (Sequence or single)."""
    if node < 0:
        return ""
    var r = doc.object_get(doc.object_get(node, "pattern"), "Regex")
    if r >= 0:
        return doc.text(r)
    var seq = doc.object_get(node, "pretokenizers")
    for i in range(doc.array_size(seq)):
        var s = _find_regex(doc, doc.at(seq, i))
        if s.byte_length() > 0:
            return s
    return ""
