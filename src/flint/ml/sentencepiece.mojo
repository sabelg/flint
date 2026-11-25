"""Pure-Mojo SentencePiece tokenizer (`flint.ml.sentencepiece`): Unigram and BPE models.

`SentencePiece.load(path)` reads either form a model ships in:

  * `spiece.model` / `tokenizer.model`: the SentencePiece `ModelProto` (protobuf), read with
    `flint.ml.protobuf`. Pieces, scores and types; `trainer_spec` (model type, byte fallback,
    unk / bos / eos / pad ids); `normalizer_spec` (precompiled charsmap, dummy prefix, extra
    whitespace removal, whitespace escaping). Encoding follows `sentencepiece.EncodeAsIds`.
  * `tokenizer.json` (Hugging Face `tokenizers`): a `Unigram` model (`vocab` = [piece, score]) or
    a `BPE` model (`vocab` + ranked `merges`, `byte_fallback`), the added tokens, and the
    normalizer / pre-tokenizer chain SentencePiece conversions produce: `Precompiled`,
    `Replace` (literal, or the regex ` {2,}`), `Prepend`, `Strip`, `Lowercase`, `Metaspace`
    (`prepend_scheme` / `add_prefix_space`, `split`), `Split` on a literal and
    `WhitespaceSplit`. Encoding follows `tokenizer.encode(text, add_special_tokens=False)`.

Normalization (both forms): user-defined pieces / added tokens are matched first and kept
verbatim; otherwise the longest match in the precompiled charsmap (a Darts double-array trie over
UTF-8 bytes, e.g. T5's nmt_nfkc) replaces the input, else one UTF-8 character is copied.
Whitespace becomes U+2581 (`▁`).

Models:
  * Unigram: Viterbi over the piece lattice (user-defined pieces get SentencePiece's
    `length * max_score - 0.1` bonus); a character with no piece costs `min_score - 10` and
    becomes `<unk>` or, with byte fallback, its `<0xNN>` bytes.
  * BPE: characters (and user-defined atoms) are merged pairwise, best first: highest piece
    score for `.model` files, lowest merge rank for `tokenizer.json`; ties go to the leftmost
    pair. Symbols left outside the vocabulary become `<0xNN>` bytes or `<unk>`.

`encode` returns ids without specials; `encode_fixed(text, n)` appends `eos` (as SigLIP / T5
templates do), truncates to `n` and right-pads with `pad`. `decode` maps pieces back to text.
The precompiled charsmap lookup is SentencePiece's (longest prefix at each byte); Hugging Face's
`Precompiled` normalizer works per grapheme instead, which differs only for rare combining
sequences.
"""

from std.collections import Dict
from flint.assets.json import JsonDocument, parse_json
from flint.platform.os import read_bytes
from flint.ml.protobuf import PbReader, WIRE_LEN

comptime SP_UNIGRAM: Int = 1
comptime SP_BPE: Int = 2

comptime PIECE_NORMAL: Int = 1
comptime PIECE_UNKNOWN: Int = 2
comptime PIECE_CONTROL: Int = 3
comptime PIECE_USER: Int = 4
comptime PIECE_UNUSED: Int = 5
comptime PIECE_BYTE: Int = 6

# normalizer ops (tokenizer.json chain)
comptime _N_PRECOMPILED: Int = 0
comptime _N_REPLACE: Int = 1
comptime _N_COLLAPSE: Int = 2
comptime _N_PREPEND: Int = 3
comptime _N_STRIP: Int = 4
comptime _N_LOWER: Int = 5

# pre-tokenizer ops
comptime _P_METASPACE: Int = 0
comptime _P_SPLIT: Int = 1
comptime _P_WSPLIT: Int = 2

comptime _UNK_PENALTY: Float32 = 10.0


# ----------------------------------------------------------------------------- UTF-8 helpers

@always_inline
def _clen(lead: UInt8) -> Int:
    """Length of the UTF-8 sequence that starts with `lead` (1 for stray bytes)."""
    if lead < 0x80:
        return 1
    if lead >= 0xF0:
        return 4
    if lead >= 0xE0:
        return 3
    if lead >= 0xC0:
        return 2
    return 1


def _valid_char(b: List[UInt8], i: Int) -> Int:
    """The byte length of a valid UTF-8 character at `i`, or 0 if malformed."""
    var n = _clen(b[i])
    if b[i] >= 0x80 and b[i] < 0xC2:
        return 0
    if b[i] > 0xF4:
        return 0
    if i + n > len(b):
        return 0
    for k in range(1, n):
        if (b[i + k] & 0xC0) != 0x80:
            return 0
    return n


def _bytes_of(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for c in s.as_bytes():
        out.append(c)
    return out^


def _str(b: List[UInt8], s: Int, e: Int) -> String:
    var out = List[UInt8](capacity=e - s)
    for i in range(s, e):
        out.append(b[i])
    return String(unsafe_from_utf8=out)


def _append_str(mut out: List[UInt8], s: String):
    for c in s.as_bytes():
        out.append(c)


def _starts_at(b: List[UInt8], i: Int, pat: List[UInt8]) -> Bool:
    if len(pat) == 0 or i + len(pat) > len(b):
        return False
    for k in range(len(pat)):
        if b[i + k] != pat[k]:
            return False
    return True


def _lower_cp(cp: Int) -> Int:
    """Simple lower-casing: ASCII, Latin-1, Latin Extended-A, Greek, Cyrillic."""
    if cp >= 65 and cp <= 90:
        return cp + 32
    if cp >= 0xC0 and cp <= 0xDE and cp != 0xD7:
        return cp + 32
    if cp >= 0x100 and cp <= 0x17F and cp % 2 == 0 and cp != 0x130 and cp != 0x138:
        return cp + 1
    if cp >= 0x391 and cp <= 0x3A9 and cp != 0x3A2:
        return cp + 32
    if cp >= 0x410 and cp <= 0x42F:
        return cp + 32
    if cp >= 0x400 and cp <= 0x40F:
        return cp + 80
    return cp


def _lowercase(b: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8](capacity=len(b))
    var i = 0
    while i < len(b):
        var n = _valid_char(b, i)
        if n == 0:
            out.append(b[i])
            i += 1
            continue
        var cp = Int(b[i])
        if n == 2:
            cp = ((cp & 0x1F) << 6) | (Int(b[i + 1]) & 0x3F)
        elif n == 3:
            cp = ((cp & 0x0F) << 12) | ((Int(b[i + 1]) & 0x3F) << 6) | (Int(b[i + 2]) & 0x3F)
        elif n == 4:
            cp = ((cp & 0x07) << 18) | ((Int(b[i + 1]) & 0x3F) << 12) | ((Int(b[i + 2]) & 0x3F) << 6) | (Int(b[i + 3]) & 0x3F)
        var lc = _lower_cp(cp)
        if lc == cp:
            for k in range(n):
                out.append(b[i + k])
        elif lc < 0x80:
            out.append(UInt8(lc))
        elif lc < 0x800:
            out.append(UInt8(0xC0 | (lc >> 6)))
            out.append(UInt8(0x80 | (lc & 0x3F)))
        else:
            out.append(UInt8(0xE0 | (lc >> 12)))
            out.append(UInt8(0x80 | ((lc >> 6) & 0x3F)))
            out.append(UInt8(0x80 | (lc & 0x3F)))
        i += n
    return out^


def _base64_decode(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length() * 3 // 4)
    var acc = 0
    var bits = 0
    for c in s.as_bytes():
        var v = -1
        if c >= 65 and c <= 90:
            v = Int(c) - 65
        elif c >= 97 and c <= 122:
            v = Int(c) - 71
        elif c >= 48 and c <= 57:
            v = Int(c) + 4
        elif c == 43 or c == 45:
            v = 62
        elif c == 47 or c == 95:
            v = 63
        if v < 0:
            continue
        acc = (acc << 6) | v
        bits += 6
        if bits >= 8:
            bits -= 8
            out.append(UInt8((acc >> bits) & 0xFF))
    return out^


def _hexval(c: UInt8) -> Int:
    if c >= 48 and c <= 57:
        return Int(c) - 48
    if c >= 65 and c <= 70:
        return Int(c) - 55
    if c >= 97 and c <= 102:
        return Int(c) - 87
    return -1


def _byte_piece_value(p: String) -> Int:
    """0..255 for a `<0xNN>` piece, else -1."""
    if p.byte_length() != 6:
        return -1
    var b = p.as_bytes()
    if b[0] != 60 or b[1] != 48 or b[2] != 120 or b[5] != 62:
        return -1
    var hi = _hexval(b[3])
    var lo = _hexval(b[4])
    if hi < 0 or lo < 0:
        return -1
    return hi * 16 + lo


# ----------------------------------------------------------------------------- charsmap trie

struct CharsMap(Copyable, Movable):
    """SentencePiece's precompiled normalization map: a Darts double-array trie over UTF-8
    bytes whose leaves index a blob of NUL-terminated replacement strings."""
    var units: List[UInt32]
    var normalized: List[UInt8]

    def __init__(out self):
        self.units = List[UInt32]()
        self.normalized = List[UInt8]()

    @staticmethod
    def parse(blob: List[UInt8]) raises -> CharsMap:
        var m = CharsMap()
        if len(blob) == 0:
            return m^
        if len(blob) < 4:
            raise Error("sentencepiece: truncated precompiled charsmap")
        var trie_size = Int(blob[0]) | (Int(blob[1]) << 8) | (Int(blob[2]) << 16) | (Int(blob[3]) << 24)
        if trie_size % 4 != 0 or 4 + trie_size > len(blob):
            raise Error("sentencepiece: bad precompiled charsmap trie size")
        var n = trie_size // 4
        m.units = List[UInt32](capacity=n)
        for i in range(n):
            var o = 4 + i * 4
            m.units.append(UInt32(blob[o]) | (UInt32(blob[o + 1]) << 8) | (UInt32(blob[o + 2]) << 16)
                           | (UInt32(blob[o + 3]) << 24))
        m.normalized = List[UInt8](capacity=len(blob) - 4 - trie_size)
        for i in range(4 + trie_size, len(blob)):
            m.normalized.append(blob[i])
        return m^

    def empty(self) -> Bool:
        return len(self.units) == 0

    @always_inline
    @staticmethod
    def _offset(u: UInt32) -> Int:
        return Int((u >> 10) << ((u & (1 << 9)) >> 6))

    @always_inline
    @staticmethod
    def _label(u: UInt32) -> UInt32:
        return u & ((1 << 31) | 0xFF)

    @always_inline
    @staticmethod
    def _has_leaf(u: UInt32) -> Bool:
        return ((u >> 8) & 1) == 1

    @always_inline
    @staticmethod
    def _value(u: UInt32) -> Int:
        return Int(u & ((1 << 31) - 1))

    def longest(self, b: List[UInt8], start: Int) -> Tuple[Int, Int]:
        """(matched byte length, blob offset) of the longest key that prefixes b[start:], or (0, 0)."""
        if len(self.units) == 0:
            return (0, 0)
        var best_len = 0
        var best_val = 0
        var node = 0
        var unit = self.units[0]
        node ^= CharsMap._offset(unit)
        var i = start
        while i < len(b):
            var c = b[i]
            if c == 0:
                break
            node ^= Int(c)
            if node < 0 or node >= len(self.units):
                break
            unit = self.units[node]
            if CharsMap._label(unit) != UInt32(c):
                break
            node ^= CharsMap._offset(unit)
            if CharsMap._has_leaf(unit):
                if node < 0 or node >= len(self.units):
                    break
                best_len = i + 1 - start
                best_val = CharsMap._value(self.units[node])
            i += 1
        return (best_len, best_val)

    def append_value(self, mut out: List[UInt8], offset: Int):
        var i = offset
        while i < len(self.normalized) and self.normalized[i] != 0:
            out.append(self.normalized[i])
            i += 1


# ----------------------------------------------------------------------------- the tokenizer

struct SentencePiece(Movable):
    var model_type: Int
    var pieces: List[String]
    var scores: List[Float32]
    var types: List[Int]
    var index: Dict[String, Int]        # NORMAL / USER / UNUSED pieces
    var ranks: Dict[String, Int]        # tokenizer.json BPE: "left\x00right" -> rank
    var atoms: Dict[String, Int]        # user-defined pieces / added tokens matched verbatim
    var atom_first: List[Bool]
    var atom_max: Int
    var byte_ids: List[Int]
    var byte_fallback: Bool
    var fuse_unk: Bool
    var unk_id: Int
    var bos_id: Int
    var eos_id: Int
    var pad_id: Int
    var max_piece_bytes: Int
    var min_score: Float32
    var max_score: Float32
    var charsmap: CharsMap
    # SentencePiece normalizer spec
    var add_dummy_prefix: Bool
    var remove_extra_whitespaces: Bool
    var escape_whitespaces: Bool
    # tokenizer.json chain
    var from_json: Bool
    var norm_ops: List[Int]
    var norm_a: List[String]
    var norm_b: List[String]
    var pre_ops: List[Int]
    var pre_a: List[String]
    var pre_scheme: List[Int]           # Metaspace prepend: 0 never, 1 first, 2 always
    var pre_split: List[Bool]
    var strip_leading_space: Bool       # decoder

    def __init__(out self):
        self.model_type = SP_UNIGRAM
        self.pieces = List[String]()
        self.scores = List[Float32]()
        self.types = List[Int]()
        self.index = Dict[String, Int]()
        self.ranks = Dict[String, Int]()
        self.atoms = Dict[String, Int]()
        self.atom_first = List[Bool](length=256, fill=False)
        self.atom_max = 0
        self.byte_ids = List[Int](length=256, fill=-1)
        self.byte_fallback = False
        self.fuse_unk = False
        self.unk_id = 0
        self.bos_id = -1
        self.eos_id = -1
        self.pad_id = -1
        self.max_piece_bytes = 1
        self.min_score = 0.0
        self.max_score = 0.0
        self.charsmap = CharsMap()
        self.add_dummy_prefix = True
        self.remove_extra_whitespaces = True
        self.escape_whitespaces = True
        self.from_json = False
        self.norm_ops = List[Int]()
        self.norm_a = List[String]()
        self.norm_b = List[String]()
        self.pre_ops = List[Int]()
        self.pre_a = List[String]()
        self.pre_scheme = List[Int]()
        self.pre_split = List[Bool]()
        self.strip_leading_space = True

    # ------------------------------------------------------------------ loading

    @staticmethod
    def load(path: String) raises -> SentencePiece:
        """A `.model` (protobuf) or `tokenizer.json` file, chosen by extension."""
        if path.endswith(".json"):
            return SentencePiece.load_json(path)
        return SentencePiece.load_model(path)

    @staticmethod
    def load_dir(dir: String) raises -> SentencePiece:
        """`tokenizer.model`, `spiece.model` or `tokenizer.json` in a Hugging Face folder."""
        for name in ["tokenizer.model", "spiece.model", "sentencepiece.model"]:
            var p = dir + "/" + name
            if len(read_bytes(p)) > 0:
                return SentencePiece.load_model(p)
        return SentencePiece.load_json(dir + "/tokenizer.json")

    def _add_atom(mut self, s: String, id: Int):
        if s.byte_length() == 0:
            return
        self.atoms[s] = id
        self.atom_first[Int(s.as_bytes()[0])] = True
        self.atom_max = max(self.atom_max, s.byte_length())

    def _finish(mut self):
        self.min_score = 3.4e38
        self.max_score = -3.4e38
        for i in range(len(self.pieces)):
            var t = self.types[i]
            if t == PIECE_NORMAL or t == PIECE_USER or t == PIECE_UNUSED:
                self.index[self.pieces[i]] = i
                self.max_piece_bytes = max(self.max_piece_bytes, self.pieces[i].byte_length())
            if t == PIECE_NORMAL:
                self.min_score = min(self.min_score, self.scores[i])
                self.max_score = max(self.max_score, self.scores[i])
            elif t == PIECE_USER:
                var piece = self.pieces[i]
                self._add_atom(piece, i)
            elif t == PIECE_UNKNOWN:
                self.unk_id = i
            elif t == PIECE_BYTE:
                var v = _byte_piece_value(self.pieces[i])
                if v >= 0:
                    self.byte_ids[v] = i
        if self.min_score > self.max_score:
            self.min_score = 0.0
            self.max_score = 0.0

    @staticmethod
    def load_model(path: String) raises -> SentencePiece:
        var data = read_bytes(path)
        if len(data) == 0:
            raise Error("sentencepiece: cannot read " + path)
        var sp = SentencePiece()
        var base = Int(data.unsafe_ptr())
        var r = PbReader(base, base + len(data))
        var charsmap = List[UInt8]()
        while r.more():
            r.key()
            if r.field == 1 and r.wire == WIRE_LEN:          # pieces
                var pr = r.sub()
                var piece = String()
                var score: Float32 = 0.0
                var ty = PIECE_NORMAL
                while pr.more():
                    pr.key()
                    if pr.field == 1 and pr.wire == WIRE_LEN:
                        piece = pr.string()
                    elif pr.field == 2 and pr.wire == 5:
                        var bits = pr.fixed32()
                        score = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(Pointer(to=bits))).unsafe_load()
                    elif pr.field == 3 and pr.wire == 0:
                        ty = pr.varint()
                    else:
                        pr.skip()
                sp.pieces.append(piece)
                sp.scores.append(score)
                sp.types.append(ty)
            elif r.field == 2 and r.wire == WIRE_LEN:        # trainer_spec
                var tr = r.sub()
                while tr.more():
                    tr.key()
                    if tr.wire == 0 and tr.field == 3:
                        sp.model_type = tr.varint()
                    elif tr.wire == 0 and tr.field == 35:
                        sp.byte_fallback = tr.varint() != 0
                    elif tr.wire == 0 and tr.field == 40:
                        sp.unk_id = _i32(tr.varint())
                    elif tr.wire == 0 and tr.field == 41:
                        sp.bos_id = _i32(tr.varint())
                    elif tr.wire == 0 and tr.field == 42:
                        sp.eos_id = _i32(tr.varint())
                    elif tr.wire == 0 and tr.field == 43:
                        sp.pad_id = _i32(tr.varint())
                    else:
                        tr.skip()
            elif r.field == 3 and r.wire == WIRE_LEN:        # normalizer_spec
                var nr = r.sub()
                while nr.more():
                    nr.key()
                    if nr.field == 2 and nr.wire == WIRE_LEN:
                        var span = nr.length_delimited()
                        charsmap = List[UInt8](capacity=span[1] - span[0])
                        for k in range(span[0], span[1]):
                            charsmap.append(data[k - base])
                    elif nr.field == 3 and nr.wire == 0:
                        sp.add_dummy_prefix = nr.varint() != 0
                    elif nr.field == 4 and nr.wire == 0:
                        sp.remove_extra_whitespaces = nr.varint() != 0
                    elif nr.field == 5 and nr.wire == 0:
                        sp.escape_whitespaces = nr.varint() != 0
                    else:
                        nr.skip()
            else:
                r.skip()
        _ = data^
        if len(sp.pieces) == 0:
            raise Error("sentencepiece: no pieces in " + path)
        if sp.model_type != SP_UNIGRAM and sp.model_type != SP_BPE:
            raise Error("sentencepiece: only unigram and BPE models are supported (" + path + ")")
        sp.charsmap = CharsMap.parse(charsmap)
        sp.strip_leading_space = sp.add_dummy_prefix
        sp.fuse_unk = sp.model_type == SP_UNIGRAM
        sp._finish()
        return sp^

    @staticmethod
    def load_json(path: String) raises -> SentencePiece:
        var data = read_bytes(path)
        if len(data) == 0:
            raise Error("sentencepiece: cannot read " + path)
        var doc = parse_json(data)
        if not doc.valid:
            raise Error("sentencepiece: invalid JSON in " + path)
        var sp = SentencePiece()
        sp.from_json = True
        sp.add_dummy_prefix = False
        sp.remove_extra_whitespaces = False
        sp.escape_whitespaces = False
        sp.strip_leading_space = False
        var model = doc.object_get(doc.root, "model")
        var mtype = doc.text(doc.object_get(model, "type"))
        var vocab = doc.object_get(model, "vocab")
        if mtype == "Unigram":
            sp.model_type = SP_UNIGRAM
            var n = doc.array_size(vocab)
            for i in range(n):
                var e = doc.at(vocab, i)
                sp.pieces.append(doc.text(doc.at(e, 0)))
                sp.scores.append(Float32(doc.number(doc.at(e, 1), 0.0)))
                sp.types.append(PIECE_NORMAL)
            sp.unk_id = doc.integer(doc.object_get(model, "unk_id"), 0)
            sp.fuse_unk = True
        elif mtype == "BPE":
            sp.model_type = SP_BPE
            var n = 0
            for c in doc.nodes[vocab].children:
                n = max(n, doc.integer(c, 0) + 1)
            sp.pieces = List[String](length=n, fill=String())
            sp.scores = List[Float32](length=n, fill=0.0)
            sp.types = List[Int](length=n, fill=PIECE_CONTROL)
            for k in range(len(doc.nodes[vocab].keys)):
                var id = doc.integer(doc.nodes[vocab].children[k], -1)
                if id >= 0:
                    sp.pieces[id] = doc.nodes[vocab].keys[k]
                    sp.types[id] = PIECE_NORMAL
            var merges = doc.object_get(model, "merges")
            for i in range(doc.array_size(merges)):
                var m = doc.at(merges, i)
                var key: String
                if doc.nodes[m].kind == 4:
                    key = doc.text(doc.at(m, 0)) + "\x00" + doc.text(doc.at(m, 1))
                else:
                    var s = doc.text(m)
                    var sb = _bytes_of(s)
                    var sp_at = -1
                    for j in range(len(sb)):
                        if sb[j] == 32:
                            sp_at = j
                            break
                    if sp_at < 0:
                        continue
                    key = _str(sb, 0, sp_at) + "\x00" + _str(sb, sp_at + 1, len(sb))
                if key not in sp.ranks:
                    sp.ranks[key] = i
            sp.fuse_unk = doc.boolean(doc.object_get(model, "fuse_unk"), False)
            var unk = doc.text(doc.object_get(model, "unk_token"), "")
            for i in range(len(sp.pieces)):
                if sp.pieces[i] == unk and unk.byte_length() > 0:
                    sp.unk_id = i
        else:
            raise Error("sentencepiece: unsupported tokenizer.json model type '" + mtype + "'")
        sp.byte_fallback = doc.boolean(doc.object_get(model, "byte_fallback"), False)
        if sp.byte_fallback:
            for i in range(len(sp.pieces)):
                if _byte_piece_value(sp.pieces[i]) >= 0:
                    sp.types[i] = PIECE_BYTE
        if sp.unk_id >= 0 and sp.unk_id < len(sp.types):
            sp.types[sp.unk_id] = PIECE_UNKNOWN
        # added tokens: matched verbatim before the normalizer
        var added = doc.object_get(doc.root, "added_tokens")
        for i in range(doc.array_size(added)):
            var a = doc.at(added, i)
            var id = doc.integer(doc.object_get(a, "id"), -1)
            var content = doc.text(doc.object_get(a, "content"))
            if id < 0:
                continue
            while id >= len(sp.pieces):
                sp.pieces.append(String())
                sp.scores.append(0.0)
                sp.types.append(PIECE_CONTROL)
            sp.pieces[id] = content
            var special = doc.boolean(doc.object_get(a, "special"), False)
            if id != sp.unk_id:
                sp.types[id] = PIECE_CONTROL if special else PIECE_USER
            sp._add_atom(content, id)
            if not special:
                continue
            if content == "</s>" or content == "<eos>":
                sp.eos_id = id
            elif content == "<s>" or content == "<bos>":
                sp.bos_id = id
            elif content == "<pad>":
                sp.pad_id = id
        sp._parse_normalizer(doc, doc.object_get(doc.root, "normalizer"))
        sp._parse_pretokenizer(doc, doc.object_get(doc.root, "pre_tokenizer"))
        sp._finish()
        # user-defined pieces of a json model are only the added tokens
        return sp^

    def _parse_normalizer(mut self, doc: JsonDocument, node: Int) raises:
        if node < 0 or doc.nodes[node].kind != 5:
            return
        var t = doc.text(doc.object_get(node, "type"))
        if t == "Sequence":
            var seq = doc.object_get(node, "normalizers")
            for i in range(doc.array_size(seq)):
                self._parse_normalizer(doc, doc.at(seq, i))
            return
        if t == "Precompiled":
            self.charsmap = CharsMap.parse(_base64_decode(doc.text(doc.object_get(node, "precompiled_charsmap"))))
            self.norm_ops.append(_N_PRECOMPILED)
            self.norm_a.append(String())
            self.norm_b.append(String())
        elif t == "Replace":
            var pat = doc.object_get(node, "pattern")
            var content = doc.text(doc.object_get(node, "content"))
            var lit = doc.object_get(pat, "String")
            if lit >= 0:
                self.norm_ops.append(_N_REPLACE)
                self.norm_a.append(doc.text(lit))
                self.norm_b.append(content)
            else:
                var rx = doc.text(doc.object_get(pat, "Regex"))
                if rx == " {2,}" and content == " ":
                    self.norm_ops.append(_N_COLLAPSE)
                    self.norm_a.append(String())
                    self.norm_b.append(String())
                else:
                    raise Error("sentencepiece: unsupported Replace regex '" + rx + "'")
        elif t == "Prepend":
            self.norm_ops.append(_N_PREPEND)
            self.norm_a.append(doc.text(doc.object_get(node, "prepend")))
            self.norm_b.append(String())
        elif t == "Strip":
            self.norm_ops.append(_N_STRIP)
            self.norm_a.append("1" if doc.boolean(doc.object_get(node, "strip_left"), True) else "0")
            self.norm_b.append("1" if doc.boolean(doc.object_get(node, "strip_right"), True) else "0")
        elif t == "Lowercase":
            self.norm_ops.append(_N_LOWER)
            self.norm_a.append(String())
            self.norm_b.append(String())
        else:
            raise Error("sentencepiece: unsupported normalizer '" + t + "'")

    def _parse_pretokenizer(mut self, doc: JsonDocument, node: Int) raises:
        if node < 0 or doc.nodes[node].kind != 5:
            return
        var t = doc.text(doc.object_get(node, "type"))
        if t == "Sequence":
            var seq = doc.object_get(node, "pretokenizers")
            for i in range(doc.array_size(seq)):
                self._parse_pretokenizer(doc, doc.at(seq, i))
            return
        if t == "Metaspace":
            self.pre_ops.append(_P_METASPACE)
            self.pre_a.append(doc.text(doc.object_get(node, "replacement"), "▁"))
            var scheme = doc.text(doc.object_get(node, "prepend_scheme"), "")
            var s = 2
            if scheme == "never":
                s = 0
            elif scheme == "first":
                s = 1
            elif scheme == "" and not doc.boolean(doc.object_get(node, "add_prefix_space"), True):
                s = 0
            self.pre_scheme.append(s)
            self.pre_split.append(doc.boolean(doc.object_get(node, "split"), True))
            if s > 0:
                self.strip_leading_space = True
        elif t == "Split":
            var lit = doc.object_get(doc.object_get(node, "pattern"), "String")
            if lit < 0:
                raise Error("sentencepiece: unsupported Split pattern (only literals)")
            self.pre_ops.append(_P_SPLIT)
            self.pre_a.append(doc.text(lit))
            self.pre_scheme.append(0)
            self.pre_split.append(True)
        elif t == "WhitespaceSplit":
            self.pre_ops.append(_P_WSPLIT)
            self.pre_a.append(String())
            self.pre_scheme.append(0)
            self.pre_split.append(True)
        else:
            raise Error("sentencepiece: unsupported pre_tokenizer '" + t + "'")

    # ------------------------------------------------------------------ queries

    def vocab_size(self) -> Int:
        return len(self.pieces)

    def piece_to_id(self, piece: String) -> Int:
        var f = self.index.get(piece)
        if f:
            return f.value()
        var a = self.atoms.get(piece)
        if a:
            return a.value()
        for i in range(len(self.pieces)):
            if self.pieces[i] == piece:
                return i
        return self.unk_id

    def id_to_piece(self, id: Int) -> String:
        if id < 0 or id >= len(self.pieces):
            return String()
        return self.pieces[id]

    def _atom_at(self, b: List[UInt8], i: Int) -> Int:
        """Byte length of the longest atom (user-defined piece / added token) at b[i:], or 0."""
        if self.atom_max == 0 or not self.atom_first[Int(b[i])]:
            return 0
        var n = min(self.atom_max, len(b) - i)
        while n > 0:
            if self.atoms.get(_str(b, i, i + n)):
                return n
            n -= 1
        return 0

    # ------------------------------------------------------------------ normalization

    def _normalize_prefix(self, b: List[UInt8], i: Int, mut out: List[UInt8], use_atoms: Bool) -> Int:
        """Appends the normalization of the longest rule at b[i:] and returns the bytes consumed."""
        if use_atoms:
            var a = self._atom_at(b, i)
            if a > 0:
                for k in range(a):
                    out.append(b[i + k])
                return a
        var m = self.charsmap.longest(b, i)
        if m[0] > 0:
            self.charsmap.append_value(out, m[1])
            return m[0]
        var n = _valid_char(b, i)
        if n == 0:
            out.append(0xEF)
            out.append(0xBF)
            out.append(0xBD)
            return 1
        for k in range(n):
            out.append(b[i + k])
        return n

    def normalize_spm(self, text: String) -> List[UInt8]:
        """SentencePiece `Normalizer::Normalize` (no alignment)."""
        var b = _bytes_of(text)
        var out = List[UInt8]()
        var i = 0
        if self.remove_extra_whitespaces:
            while i < len(b):
                var tmp = List[UInt8]()
                var used = self._normalize_prefix(b, i, tmp, True)
                if not (len(tmp) == 1 and tmp[0] == 32):
                    break
                i += used
        if i >= len(b):
            return out^
        var ws: List[UInt8] = [0xE2, 0x96, 0x81]
        if not self.escape_whitespaces:
            ws = [32]
        if self.add_dummy_prefix:
            out.extend(ws.copy())
        var prev_space = self.remove_extra_whitespaces
        while i < len(b):
            var sp = List[UInt8]()
            var used = self._normalize_prefix(b, i, sp, True)
            var s0 = 0
            if prev_space and self.remove_extra_whitespaces:
                while s0 < len(sp) and sp[s0] == 32:
                    s0 += 1
            if s0 < len(sp):
                for k in range(s0, len(sp)):
                    if self.escape_whitespaces and sp[k] == 32:
                        out.extend(ws.copy())
                    else:
                        out.append(sp[k])
                prev_space = sp[len(sp) - 1] == 32
            i += used
        if self.remove_extra_whitespaces:
            while len(out) >= len(ws):
                var tail = True
                for k in range(len(ws)):
                    if out[len(out) - len(ws) + k] != ws[k]:
                        tail = False
                if not tail:
                    break
                for _ in range(len(ws)):
                    _ = out.pop()
        return out^

    def _apply_json_normalizers(self, var b: List[UInt8]) -> List[UInt8]:
        for k in range(len(self.norm_ops)):
            var op = self.norm_ops[k]
            if op == _N_PRECOMPILED:
                var out = List[UInt8](capacity=len(b))
                var i = 0
                while i < len(b):
                    i += self._normalize_prefix(b, i, out, False)
                b = out^
            elif op == _N_REPLACE:
                var pat = _bytes_of(self.norm_a[k])
                var rep = _bytes_of(self.norm_b[k])
                if len(pat) == 0:
                    continue
                var out = List[UInt8](capacity=len(b))
                var i = 0
                while i < len(b):
                    if _starts_at(b, i, pat):
                        out.extend(rep.copy())
                        i += len(pat)
                    else:
                        out.append(b[i])
                        i += 1
                b = out^
            elif op == _N_COLLAPSE:
                var out = List[UInt8](capacity=len(b))
                for i in range(len(b)):
                    if b[i] == 32 and len(out) > 0 and out[len(out) - 1] == 32 and i > 0 and b[i - 1] == 32:
                        continue
                    out.append(b[i])
                b = out^
            elif op == _N_PREPEND:
                if len(b) > 0:
                    var out = _bytes_of(self.norm_a[k])
                    out.extend(b.copy())
                    b = out^
            elif op == _N_STRIP:
                var s = 0
                var e = len(b)
                if self.norm_a[k] == "1":
                    while s < e and (b[s] == 32 or b[s] == 9 or b[s] == 10 or b[s] == 13):
                        s += 1
                if self.norm_b[k] == "1":
                    while e > s and (b[e - 1] == 32 or b[e - 1] == 9 or b[e - 1] == 10 or b[e - 1] == 13):
                        e -= 1
                var out = List[UInt8](capacity=e - s)
                for i in range(s, e):
                    out.append(b[i])
                b = out^
            elif op == _N_LOWER:
                b = _lowercase(b)
        return b^

    def _json_pretokenize(self, b: List[UInt8], first_segment: Bool) -> List[List[UInt8]]:
        var words = List[List[UInt8]]()
        words.append(b.copy())
        for k in range(len(self.pre_ops)):
            var op = self.pre_ops[k]
            var next = List[List[UInt8]]()
            if op == _P_METASPACE:
                var rep = _bytes_of(self.pre_a[k])
                for w in range(len(words)):
                    var cur = List[UInt8]()
                    for c in words[w]:
                        if c == 32:
                            cur.extend(rep.copy())
                        else:
                            cur.append(c)
                    var scheme = self.pre_scheme[k]
                    if (scheme == 2 or (scheme == 1 and first_segment and w == 0)) and len(cur) > 0 \
                            and not _starts_at(cur, 0, rep):
                        var pre = rep.copy()
                        pre.extend(cur^)
                        cur = pre^
                    if not self.pre_split[k]:
                        next.append(cur^)
                        continue
                    # split before every replacement char (MergedWithNext)
                    var start = 0
                    var i = 0
                    while i < len(cur):
                        if i > start and _starts_at(cur, i, rep):
                            next.append(_slice(cur, start, i))
                            start = i
                        i += _clen(cur[i]) if not _starts_at(cur, i, rep) else len(rep)
                    if start < len(cur):
                        next.append(_slice(cur, start, len(cur)))
            elif op == _P_SPLIT:
                var pat = _bytes_of(self.pre_a[k])
                for w in range(len(words)):
                    var cur = words[w].copy()
                    var start = 0
                    var i = 0
                    while i < len(cur):
                        if _starts_at(cur, i, pat):
                            next.append(_slice(cur, start, i + len(pat)))   # MergedWithPrevious
                            i += len(pat)
                            start = i
                        else:
                            i += 1
                    if start < len(cur):
                        next.append(_slice(cur, start, len(cur)))
            else:
                for w in range(len(words)):
                    var cur = words[w].copy()
                    var start = -1
                    for i in range(len(cur)):
                        var sp = cur[i] == 32 or cur[i] == 9 or cur[i] == 10 or cur[i] == 13
                        if sp:
                            if start >= 0:
                                next.append(_slice(cur, start, i))
                                start = -1
                        elif start < 0:
                            start = i
                    if start >= 0:
                        next.append(_slice(cur, start, len(cur)))
            words = next^
        return words^

    # ------------------------------------------------------------------ models

    def _emit_unknown(self, b: List[UInt8], s: Int, e: Int, mut ids: List[Int]):
        if self.byte_fallback:
            for k in range(s, e):
                var bid = self.byte_ids[Int(b[k])]
                ids.append(bid if bid >= 0 else self.unk_id)
        else:
            if self.fuse_unk and len(ids) > 0 and ids[len(ids) - 1] == self.unk_id:
                return
            ids.append(self.unk_id)

    def _unigram(self, b: List[UInt8], mut ids: List[Int]):
        var n = len(b)
        if n == 0:
            return
        var best = List[Float32](length=n + 1, fill=-3.4e38)
        var back = List[Int](length=n + 1, fill=-1)
        var piece = List[Int](length=n + 1, fill=-1)
        best[0] = 0.0
        var s = 0
        while s < n:
            var clen = min(_clen(b[s]), n - s)
            if best[s] > -3.0e38:
                var found_char = False
                var e = s
                var chars = 0
                while e < n and e - s < self.max_piece_bytes:
                    e += min(_clen(b[e]), n - e)
                    chars += 1
                    if e - s > self.max_piece_bytes:
                        break
                    var hit = self.index.get(_str(b, s, e))
                    if hit:
                        var id = hit.value()
                        if self.types[id] == PIECE_UNUSED:
                            continue
                        if e - s == clen:
                            found_char = True
                        var sc = self.scores[id]
                        if self.types[id] == PIECE_USER:
                            sc = Float32(chars) * self.max_score - 0.1
                        var total = best[s] + sc
                        if total > best[e]:
                            best[e] = total
                            back[e] = s
                            piece[e] = id
                if not found_char:
                    var total = best[s] + self.min_score - _UNK_PENALTY
                    if total > best[s + clen]:
                        best[s + clen] = total
                        back[s + clen] = s
                        piece[s + clen] = -1
            s += clen
        var starts = List[Int]()
        var ends = List[Int]()
        var e = n
        while e > 0:
            starts.append(back[e])
            ends.append(e)
            e = back[e]
            if e < 0:
                return
        var k = len(starts) - 1
        while k >= 0:
            var p = piece[ends[k]]
            if p >= 0:
                ids.append(p)
            else:
                self._emit_unknown(b, starts[k], ends[k], ids)
            k -= 1

    def _pair_priority(self, b: List[UInt8], s0: Int, e0: Int, e1: Int) -> Float64:
        """Merge priority of b[s0:e0] + b[e0:e1] (higher first), or -inf if they do not merge."""
        if self.from_json:
            var r = self.ranks.get(_str(b, s0, e0) + "\x00" + _str(b, e0, e1))
            if r:
                return -Float64(r.value())
            return -1.0e300
        var hit = self.index.get(_str(b, s0, e1))
        if hit:
            return Float64(self.scores[hit.value()])
        return -1.0e300

    def _bpe(self, b: List[UInt8], mut ids: List[Int]):
        var n = len(b)
        if n == 0:
            return
        var start = List[Int]()
        var end = List[Int]()
        var frozen = List[Bool]()
        var i = 0
        while i < n:
            var a = self._atom_at(b, i) if not self.from_json else 0
            if a > 0:
                start.append(i)
                end.append(i + a)
                frozen.append(True)
                i += a
            else:
                var c = min(_clen(b[i]), n - i)
                start.append(i)
                end.append(i + c)
                frozen.append(False)
                i += c
        var m = len(start)
        var nxt = List[Int](length=m, fill=-1)
        var alive = List[Bool](length=m, fill=True)
        for k in range(m - 1):
            nxt[k] = k + 1
        var prio = List[Float64](length=m, fill=-1.0e300)    # pair (k, nxt[k])
        for k in range(m - 1):
            if not frozen[k] and not frozen[k + 1]:
                prio[k] = self._pair_priority(b, start[k], end[k], end[k + 1])
        var prv = List[Int](length=m, fill=-1)
        for k in range(1, m):
            prv[k] = k - 1
        while True:
            var bk = -1
            var bp: Float64 = -1.0e299
            var k = 0
            while k >= 0 and k < m:
                if alive[k] and nxt[k] >= 0 and prio[k] > bp:
                    bp = prio[k]
                    bk = k
                k = nxt[k] if alive[k] else k + 1
            if bk < 0:
                break
            var r = nxt[bk]
            end[bk] = end[r]
            alive[r] = False
            nxt[bk] = nxt[r]
            if nxt[r] >= 0:
                prv[nxt[r]] = bk
            prio[bk] = -1.0e300
            if nxt[bk] >= 0 and not frozen[nxt[bk]]:
                prio[bk] = self._pair_priority(b, start[bk], end[bk], end[nxt[bk]])
            var l = prv[bk]
            if l >= 0 and not frozen[l]:
                prio[l] = self._pair_priority(b, start[l], end[l], end[bk])
        var k = 0
        while k >= 0 and k < m:
            if alive[k]:
                var s = _str(b, start[k], end[k])
                var hit = self.index.get(s)
                if not hit:
                    hit = self.atoms.get(s)
                if hit:
                    ids.append(hit.value())
                else:
                    self._emit_unknown(b, start[k], end[k], ids)
                k = nxt[k]
            else:
                k += 1

    def _model(self, b: List[UInt8], mut ids: List[Int]):
        if self.model_type == SP_BPE:
            self._bpe(b, ids)
        else:
            self._unigram(b, ids)

    # ------------------------------------------------------------------ public API

    def encode(self, text: String) -> List[Int]:
        """Token ids of `text`, no special tokens added."""
        var ids = List[Int]()
        if not self.from_json:
            var norm = self.normalize_spm(text)
            self._model(norm, ids)
            return ids^
        # tokenizer.json: split out added tokens, then normalize / pre-tokenize each segment
        var b = _bytes_of(text)
        var seg_start = 0
        var i = 0
        var first = True
        while i <= len(b):
            var a = 0
            if i < len(b):
                a = self._atom_at(b, i)
            if a > 0 or i == len(b):
                if i > seg_start:
                    var seg = self._apply_json_normalizers(_slice(b, seg_start, i))
                    var words = self._json_pretokenize(seg, first)
                    for w in range(len(words)):
                        self._model(words[w], ids)
                if i == len(b):
                    break
                var f = self.atoms.get(_str(b, i, i + a))
                if f:
                    ids.append(f.value())
                first = False
                i += a
                seg_start = i
            else:
                i += 1
        return ids^

    def encode_fixed(self, text: String, length: Int, add_eos: Bool = True, pad: Bool = True) -> List[Int]:
        """`encode(text)` + eos, truncated to `length` (the eos is kept) and right-padded."""
        var ids = self.encode(text)
        var room = length - (1 if add_eos and self.eos_id >= 0 else 0)
        var out = List[Int](capacity=length)
        for k in range(min(room, len(ids))):
            out.append(ids[k])
        if add_eos and self.eos_id >= 0:
            out.append(self.eos_id)
        if pad:
            var p = self.pad_id if self.pad_id >= 0 else 0
            while len(out) < length:
                out.append(p)
        return out^

    def decode(self, ids: List[Int]) -> String:
        var bytes = List[UInt8]()
        for id in ids:
            if id < 0 or id >= len(self.pieces):
                continue
            var t = self.types[id]
            if t == PIECE_CONTROL:
                continue
            if t == PIECE_BYTE:
                bytes.append(UInt8(_byte_piece_value(self.pieces[id])))
                continue
            var p = _bytes_of(self.pieces[id])
            var i = 0
            while i < len(p):
                if i + 2 < len(p) and p[i] == 0xE2 and p[i + 1] == 0x96 and p[i + 2] == 0x81:
                    bytes.append(32)
                    i += 3
                else:
                    bytes.append(p[i])
                    i += 1
        var s = 0
        if self.strip_leading_space and len(bytes) > 0 and bytes[0] == 32:
            s = 1
        var out = List[UInt8](capacity=len(bytes))
        var i = s
        while i < len(bytes):
            var n = _valid_char(bytes, i)
            if n == 0:
                out.append(0xEF)
                out.append(0xBF)
                out.append(0xBD)
                i += 1
            else:
                for k in range(n):
                    out.append(bytes[i + k])
                i += n
        return String(unsafe_from_utf8=out)


def _slice(b: List[UInt8], s: Int, e: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=max(0, e - s))
    for i in range(s, e):
        out.append(b[i])
    return out^


@always_inline
def _i32(v: Int) -> Int:
    """Protobuf int32 fields arrive sign-extended to 64 bits (-1 = 2^64 - 1)."""
    if v >= (1 << 31) and v < (1 << 32):
        return v - (1 << 32)
    return v
