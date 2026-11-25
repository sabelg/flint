"""Text embeddings with Qwen3-Embedding and a cosine top-k index (`flint.ml.llm.embed`).

`TextEmbedder` runs a Qwen3-Embedding checkpoint (e.g. Qwen3-Embedding-0.6B, Apache-2.0) on the
shared Qwen3 decoder (`LlmModel` without its output head): the text is tokenised, the tokenizer's
`<|endoftext|>` post-processor token is appended, and the final RMS-normed hidden state of that
last token (last-token pooling) is L2-normalised. Matryoshka (MRL) truncation keeps the first
`dim` components (32..1024 for the 0.6B model) and renormalises.

Queries carry a one-sentence task instruction, documents do not (the model card's format):
`instruct_query(task, query)` = `"Instruct: " + task + "\\nQuery:" + query`. Since the decoder is
causal, the shared instruction prefix stays in the KV cache between queries (prefix reuse).

`EmbeddingIndex` keeps unit vectors with a key and a kind (`asset`, `script`, `quest`,
`dialogue`, ...), answers cosine top-k queries (optionally for one kind) and saves to / loads
from a small binary file, so a game can embed its content once at build time.
"""

from std.math import sqrt
from std.memory import bitcast
from flint.platform.os import read_bytes, write_bytes
from flint.ml.llm.tokenizer import BpeTokenizer
from flint.ml.llm.model import LlmModel
from flint.ml.llm.quant import Q_F32

comptime EMBED_MIN_DIM: Int = 32
comptime DEFAULT_TASK: String = "Given a web search query, retrieve relevant passages that answer the query"


def instruct_query(task: String, query: String) -> String:
    """The Qwen3-Embedding query format: `Instruct: {task}\\nQuery:{query}`."""
    return "Instruct: " + task + "\nQuery:" + query


def l2_normalize(mut v: List[Float32]):
    var s: Float64 = 0
    for x in v:
        s += Float64(x) * Float64(x)
    if s <= 0:
        return
    var inv = Float32(1.0 / sqrt(s))
    for i in range(len(v)):
        v[i] *= inv


def mrl_truncate(v: List[Float32], dim: Int) raises -> List[Float32]:
    """The first `dim` components, L2-normalised (Matryoshka truncation)."""
    if dim < 1 or dim > len(v):
        raise Error("embed: cannot truncate a " + String(len(v)) + "-d vector to " + String(dim))
    var out = List[Float32](capacity=dim)
    for i in range(dim):
        out.append(v[i])
    l2_normalize(out)
    return out^


def cosine(a: List[Float32], b: List[Float32]) -> Float32:
    """Cosine similarity of two vectors of the same length."""
    var ab: Float64 = 0
    var aa: Float64 = 0
    var bb: Float64 = 0
    for i in range(min(len(a), len(b))):
        ab += Float64(a[i]) * Float64(b[i])
        aa += Float64(a[i]) * Float64(a[i])
        bb += Float64(b[i]) * Float64(b[i])
    if aa <= 0 or bb <= 0:
        return 0
    return Float32(ab / sqrt(aa * bb))


struct TextEmbedder(Movable):
    var model: LlmModel
    var tok: BpeTokenizer
    var eos: Int
    var dim: Int
    var max_tokens: Int

    def __init__(out self, dir: String, dim: Int = 0, fmt: Int = Q_F32, gpu: Bool = True, max_tokens: Int = 512) raises:
        """`dim` 0 means the full hidden size; otherwise MRL truncation to `dim` (32..hidden)."""
        self.model = LlmModel.load(dir, fmt, gpu, max_tokens, False)
        self.tok = BpeTokenizer.load(dir + "/tokenizer.json")
        self.eos = self.tok.token_id("<|endoftext|>")
        self.max_tokens = max_tokens
        self.dim = self.model.cfg.hidden if dim == 0 else dim
        if self.dim < EMBED_MIN_DIM or self.dim > self.model.cfg.hidden:
            raise Error("embed: dimension must be in " + String(EMBED_MIN_DIM) + ".." + String(self.model.cfg.hidden))

    def full_dim(self) -> Int:
        return self.model.cfg.hidden

    def tokens(mut self, text: String) raises -> List[Int]:
        """Token ids as the Hugging Face tokenizer gives them (post-processor `<|endoftext|>`
        appended), truncated to `max_tokens`."""
        var ids = self.tok.encode(text)
        var room = self.max_tokens - (1 if self.eos >= 0 else 0)
        while len(ids) > room:
            _ = ids.pop()
        if self.eos >= 0:
            ids.append(self.eos)
        if len(ids) == 0:
            raise Error("embed: empty input")
        return ids^

    def embed_tokens_full(mut self, ids: List[Int]) raises -> List[Float32]:
        """Unit-length full-size embedding of a token sequence, reusing the cached prefix."""
        var p = 0
        while p < len(self.model.past) and p < len(ids) and self.model.past[p] == ids[p]:
            p += 1
        if p == len(ids):
            p -= 1
        self.model.rewind(p)
        var rest = List[Int]()
        for i in range(p, len(ids)):
            rest.append(ids[i])
        var h = self.model.eval_hidden(rest)
        l2_normalize(h)
        return h^

    def embed_full(mut self, text: String) raises -> List[Float32]:
        return self.embed_tokens_full(self.tokens(text))

    def embed(mut self, text: String) raises -> List[Float32]:
        """Unit-length embedding of `text` at `self.dim` (documents: no instruction)."""
        var h = self.embed_full(text)
        if self.dim == len(h):
            return h^
        return mrl_truncate(h, self.dim)

    def embed_query(mut self, query: String, task: String = DEFAULT_TASK) raises -> List[Float32]:
        """Unit-length embedding of a search query with its task instruction."""
        return self.embed(instruct_query(task, query))


struct EmbeddingHit(Copyable, Movable):
    var index: Int
    var key: String
    var kind: String
    var score: Float32

    def __init__(out self, index: Int, key: String, kind: String, score: Float32):
        self.index = index
        self.key = key
        self.kind = kind
        self.score = score


def _put_u32(mut b: List[UInt8], v: Int):
    for i in range(4):
        b.append(UInt8((v >> (8 * i)) & 255))


def _get_u32(b: List[UInt8], at: Int) raises -> Int:
    if at + 4 > len(b):
        raise Error("embed: truncated index file")
    var v = 0
    for i in range(4):
        v |= Int(b[at + i]) << (8 * i)
    return v


def _put_str(mut b: List[UInt8], s: String):
    var bs = s.as_bytes()
    _put_u32(b, len(bs))
    for x in bs:
        b.append(x)


struct EmbeddingIndex(Copyable, Movable, Sized):
    """Unit vectors of one dimension with keys and kinds; cosine top-k search."""
    var dim: Int
    var vecs: List[Float32]     # [len, dim], each row L2-normalised
    var keys: List[String]
    var kinds: List[String]

    def __init__(out self, dim: Int):
        self.dim = dim
        self.vecs = List[Float32]()
        self.keys = List[String]()
        self.kinds = List[String]()

    def __len__(self) -> Int:
        return len(self.keys)

    def add(mut self, key: String, kind: String, vec: List[Float32]) raises -> Int:
        """Adds a vector (normalised here; a longer one is MRL-truncated to `dim`)."""
        if len(vec) < self.dim:
            raise Error("embed: index of dim " + String(self.dim) + " given a " + String(len(vec)) + "-d vector")
        var v = mrl_truncate(vec, self.dim)
        for x in v:
            self.vecs.append(x)
        self.keys.append(key)
        self.kinds.append(kind)
        return len(self.keys) - 1

    def vector(self, i: Int) -> List[Float32]:
        var out = List[Float32](capacity=self.dim)
        for j in range(self.dim):
            out.append(self.vecs[i * self.dim + j])
        return out^

    def search(self, query: List[Float32], k: Int, kind: String = "") raises -> List[EmbeddingHit]:
        """The `k` entries most similar to `query` (cosine; ties go to the earlier entry),
        best first; only entries of `kind` unless it is empty."""
        var q = mrl_truncate(query, self.dim)
        var best = List[Int]()
        var score = List[Float32]()
        var D = self.dim
        var qp = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(q.unsafe_ptr()))
        var vp = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(self.vecs.unsafe_ptr()))
        for i in range(len(self.keys)):
            if kind.byte_length() > 0 and self.kinds[i] != kind:
                continue
            var acc = SIMD[DType.float32, 8](0)
            var j = 0
            var row = vp.unsafe_offset(i * D)
            while j + 8 <= D:
                acc += qp.unsafe_offset(j).unsafe_load[width=8, alignment=1]() * row.unsafe_offset(j).unsafe_load[width=8, alignment=1]()
                j += 8
            var s = acc.reduce_add()
            while j < D:
                s += qp.unsafe_offset(j).unsafe_load() * row.unsafe_offset(j).unsafe_load()
                j += 1
            if len(best) == k and (k == 0 or s <= score[len(score) - 1]):
                continue
            var pos = len(best)
            while pos > 0 and score[pos - 1] < s:
                pos -= 1
            best.insert(pos, i)
            score.insert(pos, s)
            if len(best) > k:
                _ = best.pop()
                _ = score.pop()
        _ = len(q)
        var hits = List[EmbeddingHit]()
        for r in range(len(best)):
            hits.append(EmbeddingHit(best[r], self.keys[best[r]], self.kinds[best[r]], score[r]))
        return hits^

    def to_bytes(self) -> List[UInt8]:
        var b = List[UInt8]()
        for c in "FLEMBIX1".as_bytes():
            b.append(c)
        _put_u32(b, self.dim)
        _put_u32(b, len(self.keys))
        for i in range(len(self.keys)):
            _put_str(b, self.keys[i])
            _put_str(b, self.kinds[i])
        for x in self.vecs:
            var bits = Int(bitcast[DType.uint32](x))
            _put_u32(b, bits)
        return b^

    @staticmethod
    def from_bytes(b: List[UInt8]) raises -> EmbeddingIndex:
        var magic = "FLEMBIX1".as_bytes()
        if len(b) < 16:
            raise Error("embed: not an embedding index")
        for i in range(8):
            if b[i] != magic[i]:
                raise Error("embed: not an embedding index")
        var dim = _get_u32(b, 8)
        var n = _get_u32(b, 12)
        var idx = EmbeddingIndex(dim)
        var at = 16
        for _ in range(n):
            for which in range(2):
                var ln = _get_u32(b, at)
                at += 4
                if at + ln > len(b):
                    raise Error("embed: truncated index file")
                var sb = List[UInt8](capacity=ln)
                for c in range(ln):
                    sb.append(b[at + c])
                var s = String(unsafe_from_utf8=sb)
                at += ln
                if which == 0:
                    idx.keys.append(s)
                else:
                    idx.kinds.append(s)
        if at + n * dim * 4 != len(b):
            raise Error("embed: index file size mismatch")
        for _ in range(n * dim):
            idx.vecs.append(bitcast[DType.float32](UInt32(_get_u32(b, at))))
            at += 4
        return idx^

    def save(self, path: String) raises:
        var b = self.to_bytes()
        if not write_bytes(path, b):
            raise Error("embed: cannot write " + path)

    @staticmethod
    def load(path: String) raises -> EmbeddingIndex:
        return EmbeddingIndex.from_bytes(read_bytes(path))
