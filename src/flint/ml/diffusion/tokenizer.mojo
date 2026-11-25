"""CLIP BPE tokenizer (`flint.ml.diffusion.tokenizer`), as Hugging Face `CLIPTokenizer`.

Reads `vocab.json` + `merges.txt` of any OpenAI-CLIP-family text encoder (SD 1.x / StableMaterials
CLIP-L, sentence-transformers CLIP-B, ...). Steps, matching `transformers.CLIPTokenizer`:

  1. clean + lowercase: control characters dropped, whitespace runs collapsed, lowercase
     (ASCII, Latin-1, Greek, Cyrillic);
  2. split with CLIP's pattern `'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+`
     (digits one by one);
  3. each piece's UTF-8 bytes become byte-level symbols, the last one carrying `</w>`, then merge
     by rank (only the first 49152 - 256 - 2 merges count, as in CLIP);
  4. `<|startoftext|>` + ids + `<|endoftext|>`, truncated to `max_length` (77), optionally padded.
"""

from std.collections import Dict
from flint.assets.json import parse_json
from flint.platform.os import read_bytes
from flint.ml.llm.tokenizer import utf8_decode, utf8_append, char_class, CC_LETTER, CC_NUMBER, CC_SPACE

comptime CLIP_MERGES: Int = 49152 - 256 - 2


def _bytes_to_unicode() -> List[Int]:
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


def _lower(cp: Int) -> Int:
    if cp >= 65 and cp <= 90:
        return cp + 32
    if cp >= 0xC0 and cp <= 0xDE and cp != 0xD7:
        return cp + 32
    if cp >= 0x391 and cp <= 0x3A9 and cp != 0x3A2:
        return cp + 32
    if cp >= 0x410 and cp <= 0x42F:
        return cp + 32
    if cp >= 0x400 and cp <= 0x40F:
        return cp + 80
    if cp >= 0x100 and cp <= 0x17F and cp % 2 == 0 and cp != 0x130 and cp != 0x138:
        return cp + 1
    return cp


struct ClipTokenizer(Movable):
    var vocab: Dict[String, Int]
    var ranks: Dict[String, Int]
    var byte_sym: List[String]
    var bos: Int
    var eos: Int
    var pad: Int
    var max_length: Int
    var cache: Dict[String, List[Int]]

    def __init__(out self):
        self.vocab = Dict[String, Int]()
        self.ranks = Dict[String, Int]()
        self.byte_sym = List[String]()
        self.bos = 49406
        self.eos = 49407
        self.pad = 49407
        self.max_length = 77
        self.cache = Dict[String, List[Int]]()

    @staticmethod
    def load(dir: String) raises -> ClipTokenizer:
        """From a folder with `vocab.json` and `merges.txt` (e.g. `.../tokenizer`)."""
        var tok = ClipTokenizer()
        var vb = read_bytes(dir + "/vocab.json")
        if len(vb) == 0:
            raise Error("clip tokenizer: cannot read " + dir + "/vocab.json")
        var doc = parse_json(vb)
        if not doc.valid:
            raise Error("clip tokenizer: invalid vocab.json")
        var root = doc.root
        for k in range(len(doc.nodes[root].keys)):
            tok.vocab[doc.nodes[root].keys[k]] = Int(doc.number(doc.nodes[root].children[k]))
        var mb = read_bytes(dir + "/merges.txt")
        if len(mb) == 0:
            raise Error("clip tokenizer: cannot read " + dir + "/merges.txt")
        var line_no = 0
        var start = 0
        var rank = 0
        for i in range(len(mb) + 1):
            if i == len(mb) or mb[i] == 10:
                var end = i
                if end > start and mb[end - 1] == 13:
                    end -= 1
                if line_no > 0 and end > start and rank < CLIP_MERGES:
                    var line = String(unsafe_from_utf8=Span(mb)[start:end])
                    tok.ranks[line] = rank
                    rank += 1
                line_no += 1
                start = i + 1
        var bu = _bytes_to_unicode()
        for b in range(256):
            var s = List[UInt8]()
            utf8_append(s, bu[b])
            tok.byte_sym.append(String(unsafe_from_utf8=s))
        var b = tok.vocab.get("<|startoftext|>")
        if b:
            tok.bos = b.value()
        var e = tok.vocab.get("<|endoftext|>")
        if e:
            tok.eos = e.value()
            tok.pad = e.value()
        return tok^

    def _bpe(mut self, piece: List[UInt8]) raises -> List[Int]:
        var key = String(unsafe_from_utf8=piece)
        var hit = self.cache.get(key)
        if hit:
            return hit.value().copy()
        var word = List[String]()
        for k in range(len(piece)):
            word.append(self.byte_sym[Int(piece[k])])
        word[len(word) - 1] = word[len(word) - 1] + "</w>"
        while len(word) > 1:
            var best = -1
            var best_rank = 1 << 60
            for i in range(len(word) - 1):
                var r = self.ranks.get(word[i] + " " + word[i + 1])
                if r and r.value() < best_rank:
                    best_rank = r.value()
                    best = i
            if best < 0:
                break
            var first = word[best]
            var second = word[best + 1]
            var nw = List[String]()
            var i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    nw.append(first + second)
                    i += 2
                else:
                    nw.append(word[i])
                    i += 1
            word = nw^
        var ids = List[Int]()
        for s in word:
            var v = self.vocab.get(s)
            if not v:
                raise Error("clip tokenizer: no vocab entry for '" + s + "'")
            ids.append(v.value())
        self.cache[key] = ids.copy()
        return ids^

    def encode(mut self, text: String, pad_to_max: Bool = False) raises -> List[Int]:
        """Token ids with `<|startoftext|>` / `<|endoftext|>`, truncated to `max_length`."""
        var raw = List[UInt8](text.as_bytes())
        # 1. clean, collapse whitespace, lowercase
        var cps = List[Int]()
        var i = 0
        var last_space = True
        while i < len(raw):
            var d = utf8_decode(raw, i)
            i += d[1]
            var cp = d[0]
            if cp == 0 or cp == 0xFFFD or (cp < 32 and cp != 9 and cp != 10 and cp != 13) or cp == 127:
                continue
            if char_class(cp) == CC_SPACE:
                if not last_space:
                    cps.append(32)
                last_space = True
                continue
            cps.append(_lower(cp))
            last_space = False
        while len(cps) > 0 and cps[len(cps) - 1] == 32:
            _ = cps.pop()
        # 2. pre-tokenize
        var ids = List[Int]()
        ids.append(self.bos)
        var n = len(cps)
        var p = 0
        while p < n:
            var cp = cps[p]
            var cls = char_class(cp)
            if cls == CC_SPACE:
                p += 1
                continue
            var e = p
            if cp == 39 and p + 1 < n:      # contractions
                var c1 = cps[p + 1]
                if c1 == 115 or c1 == 116 or c1 == 109 or c1 == 100:      # s t m d
                    e = p + 2
                elif p + 2 < n and ((c1 == 114 and cps[p + 2] == 101) or (c1 == 118 and cps[p + 2] == 101)
                                    or (c1 == 108 and cps[p + 2] == 108)):  # re ve ll
                    e = p + 3
            if e == p:
                if cls == CC_LETTER:
                    e = p + 1
                    while e < n and char_class(cps[e]) == CC_LETTER:
                        e += 1
                elif cls == CC_NUMBER:
                    e = p + 1
                else:
                    e = p + 1
                    while e < n and char_class(cps[e]) != CC_SPACE and char_class(cps[e]) != CC_LETTER \
                            and char_class(cps[e]) != CC_NUMBER:
                        e += 1
            var piece = List[UInt8]()
            for q in range(p, e):
                utf8_append(piece, cps[q])
            var pid = self._bpe(piece)
            for v in pid:
                ids.append(v)
            p = e
        if len(ids) > self.max_length - 1:
            ids.resize(self.max_length - 1, 0)
        ids.append(self.eos)
        if pad_to_max:
            while len(ids) < self.max_length:
                ids.append(self.pad)
        return ids^


def eos_position(ids: List[Int], eos: Int) -> Int:
    """Index of the first end-of-text token (where CLIP pools the sentence embedding)."""
    for i in range(len(ids)):
        if ids[i] == eos:
            return i
    return len(ids) - 1
