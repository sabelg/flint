"""Moonshine's SentencePiece-style BPE vocabulary, decode side (`flint.ml.moonshine_tokenizer`).

Reads the Hugging Face `tokenizer.json` that ships with the model: `model.vocab` (piece -> id)
and `added_tokens` (the specials `<unk>` `<s>` `</s>` and the `<<ST_n>>` reserved ids).
Decoding follows the file's decoder chain: `▁` becomes a space, `<0xNN>` byte-fallback pieces
become raw bytes, the pieces are fused and one leading space is stripped. Invalid UTF-8 runs
from byte fallback become U+FFFD, as in the reference.
"""

from flint.assets.json import parse_json
from flint.platform.os import read_bytes


def _hex(c: UInt8) -> Int:
    if c >= 48 and c <= 57:
        return Int(c) - 48
    if c >= 65 and c <= 70:
        return Int(c) - 55
    if c >= 97 and c <= 102:
        return Int(c) - 87
    return -1


def _utf8_len(lead: UInt8) -> Int:
    if lead < 0x80:
        return 1
    if lead >= 0xC2 and lead <= 0xDF:
        return 2
    if lead >= 0xE0 and lead <= 0xEF:
        return 3
    if lead >= 0xF0 and lead <= 0xF4:
        return 4
    return 0


def sanitize_utf8(b: List[UInt8]) -> String:
    """`b` as a String, each invalid sequence replaced by U+FFFD."""
    var out = List[UInt8](capacity=len(b))
    var i = 0
    while i < len(b):
        var n = _utf8_len(b[i])
        var ok = n > 0 and i + n <= len(b)
        if ok:
            for j in range(1, n):
                if (b[i + j] & 0xC0) != 0x80:
                    ok = False
        if ok:
            for j in range(n):
                out.append(b[i + j])
            i += n
        else:
            out.append(0xEF)
            out.append(0xBF)
            out.append(0xBD)
            i += 1
    return String(unsafe_from_utf8=out)


struct MoonshineTokenizer(Movable):
    var pieces: List[String]
    var byte_value: List[Int]     # >= 0 for `<0xNN>` pieces
    var special: List[Bool]
    var bos: Int
    var eos: Int

    def __init__(out self):
        self.pieces = List[String]()
        self.byte_value = List[Int]()
        self.special = List[Bool]()
        self.bos = 1
        self.eos = 2

    @staticmethod
    def load(path: String) raises -> MoonshineTokenizer:
        var data = read_bytes(path)
        if len(data) == 0:
            raise Error("moonshine tokenizer: cannot read " + path)
        var doc = parse_json(data)
        if not doc.valid:
            raise Error("moonshine tokenizer: invalid JSON in " + path)
        var tok = MoonshineTokenizer()
        var vocab = doc.object_get(doc.object_get(doc.root, "model"), "vocab")
        if vocab < 0:
            raise Error("moonshine tokenizer: no model.vocab in " + path)
        var n = 0
        for c in doc.nodes[vocab].children:
            n = max(n, doc.integer(c, 0) + 1)
        var added = doc.object_get(doc.root, "added_tokens")
        for i in range(doc.array_size(added)):
            n = max(n, doc.integer(doc.object_get(doc.at(added, i), "id"), 0) + 1)
        tok.pieces = List[String](length=n, fill=String())
        tok.byte_value = List[Int](length=n, fill=-1)
        tok.special = List[Bool](length=n, fill=False)
        for k in range(len(doc.nodes[vocab].keys)):
            var id = doc.integer(doc.nodes[vocab].children[k], -1)
            if id >= 0:
                tok.pieces[id] = doc.nodes[vocab].keys[k]
        for i in range(doc.array_size(added)):
            var a = doc.at(added, i)
            var id = doc.integer(doc.object_get(a, "id"), -1)
            if id >= 0:
                tok.pieces[id] = doc.text(doc.object_get(a, "content"))
                tok.special[id] = doc.boolean(doc.object_get(a, "special"), True)
        for id in range(n):
            var p = tok.pieces[id]
            if p.byte_length() == 6 and p.startswith("<0x") and p.endswith(">"):
                var hi = _hex(p.as_bytes()[3])
                var lo = _hex(p.as_bytes()[4])
                if hi >= 0 and lo >= 0:
                    tok.byte_value[id] = hi * 16 + lo
        return tok^

    def vocab_size(self) -> Int:
        return len(self.pieces)

    def decode(self, ids: List[Int], skip_special: Bool = True) -> String:
        var bytes = List[UInt8]()
        for id in ids:
            if id < 0 or id >= len(self.pieces):
                continue
            if skip_special and self.special[id]:
                continue
            if self.byte_value[id] >= 0:
                bytes.append(UInt8(self.byte_value[id]))
                continue
            var p = self.pieces[id].as_bytes()
            var i = 0
            while i < len(p):
                # U+2581 LOWER ONE EIGHTH BLOCK is E2 96 81
                if i + 2 < len(p) and p[i] == 0xE2 and p[i + 1] == 0x96 and p[i + 2] == 0x81:
                    bytes.append(32)
                    i += 3
                else:
                    bytes.append(p[i])
                    i += 1
        var start = 1 if len(bytes) > 0 and bytes[0] == 32 else 0
        var trimmed = List[UInt8](capacity=len(bytes))
        for i in range(start, len(bytes)):
            trimmed.append(bytes[i])
        return sanitize_utf8(trimmed)
