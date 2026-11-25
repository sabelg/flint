"""MusicCoCa text -> music-style embedding and RVQ style tokens in pure Mojo (`flint.ml.musiccoca`).

MusicCoCa (Google DeepMind, part of Magenta RealTime; weights CC-BY 4.0) embeds a text prompt such
as "dark ambient drone" and 10 s music clips into one 768-d space. Magenta RT steers generation
with the 12-level RVQ quantisation of that embedding. This module runs the text tower:

  - `SpmUnigram`: SentencePiece unigram tokenizer (Viterbi over piece scores, byte fallback),
    nmt_nfkc whitespace handling for ASCII text, as `sentencepiece.EncodeAsIds`.
  - `MusicCoCaText.embed(text)`: lower-case, SOS id 1, at most 128 tokens; 12 pre-LN layers
    (768, 12 x 64 heads, logits soft-capped by 50 tanh(x / 50), exact-erf GELU 3072), LayerNorm,
    12 x 256-head attention pooling with a learned query, projection, LayerNorm. Only real
    tokens are computed: padded rows never reach them (keys are masked and padded FFN rows are
    zeroed in the original), so the result equals the fixed 128-token TFLite graph.
  - `MusicCoCaMapper` (`embed_style(text, use_mapper=True, seed)`): the text -> audio-space
    mapper Magenta RT's own prompting uses, with `numpy.random.RandomState(seed)` noise
    reproduced bit for bit (`NumpyRandomState`).
  - `MusicCoCaText.tokenize(emb)`: greedy 12-level RVQ (1024 x 768 codebooks) -> style codes for
    `flint.ml.magenta_rt.MagentaRT.set_style`. Blend styles by averaging embeddings first
    (`blend_embeddings`), as the reference does.

Weights come from `tools/magenta_rt/convert_musiccoca.py` (TFLite -> `musiccoca_text.safetensors`).
The audio tower (log-mel + music encoder) is not ported; text prompts cover in-game steering.
Non-ASCII text is matched byte for byte without NFKC folding.
"""

from std.math import sqrt, exp, erf, tanh, log, sin, cos
from std.ffi import external_call
from flint.core.jobs import JobPool
from flint.ml.ops import matmul_2d
from flint.ml.llm.safetensors import SafeTensors
from flint.ml.audio_codec import FBuf, fp, copy_floats, load_f32, RvqCodebook

comptime COCA_DIM = 768
comptime COCA_HEADS = 12
comptime COCA_FF = 3072
comptime COCA_LAYERS = 12
comptime COCA_MAX_TOKENS = 128
comptime COCA_POOL_HEAD = 256
comptime COCA_RVQ_LEVELS = 12


def _cpu_count() -> Int:
    return Int(external_call["sysconf", Int64](Int32(84)))


def _layer_norm_rows(x: Int, g: Int, b: Int, o: Int, rows: Int, n: Int):
    for r in range(rows):
        var px = fp(x + r * n * 4)
        var po = fp(o + r * n * 4)
        var mean: Float32 = 0
        for i in range(n):
            mean += px.unsafe_offset(i).unsafe_load()
        mean /= Float32(n)
        var var_: Float32 = 0
        for i in range(n):
            var d = px.unsafe_offset(i).unsafe_load() - mean
            var_ += d * d
        var_ /= Float32(n)
        var rs = 1.0 / sqrt(var_ + 1e-6)
        for i in range(n):
            var v = (px.unsafe_offset(i).unsafe_load() - mean) * rs
            po.unsafe_offset(i).unsafe_store(v * fp(g + i * 4).unsafe_load() + fp(b + i * 4).unsafe_load())


def _softmax_inplace(a: Int, n: Int):
    var p = fp(a)
    var m: Float32 = -3.4e38
    for i in range(n):
        m = max(m, p.unsafe_offset(i).unsafe_load())
    var s: Float32 = 0
    for i in range(n):
        var e = exp(p.unsafe_offset(i).unsafe_load() - m)
        p.unsafe_offset(i).unsafe_store(e)
        s += e
    var inv = 1.0 / s
    for i in range(n):
        p.unsafe_offset(i).unsafe_store(p.unsafe_offset(i).unsafe_load() * inv)


# ------------------------------------------------------------------------- tokenizer

comptime _SPM_NORMAL = 1
comptime _SPM_UNK = 2
comptime _SPM_USER = 4
comptime _SPM_BYTE = 6


struct SpmUnigram(Movable):
    """SentencePiece unigram model (pieces, scores, types) with byte fallback."""
    var ids: Dict[String, Int]
    var scores: List[Float32]
    var byte_ids: List[Int]
    var unk_id: Int
    var unk_score: Float32
    var max_len: Int

    def __init__(out self, pieces: List[String], scores: List[Float32], types: List[Int]):
        self.ids = Dict[String, Int]()
        self.scores = scores.copy()
        self.byte_ids = List[Int](length=256, fill=-1)
        self.unk_id = 0
        self.max_len = 1
        var min_score: Float32 = 0
        for i in range(len(pieces)):
            var t = types[i]
            if t == _SPM_NORMAL or t == _SPM_USER:
                self.ids[pieces[i]] = i
                self.max_len = max(self.max_len, pieces[i].byte_length())
                min_score = min(min_score, scores[i])
            elif t == _SPM_UNK:
                self.unk_id = i
            elif t == _SPM_BYTE:
                # "<0xAB>"
                var hex = String(pieces[i][byte=3:5])
                var v = 0
                for c in range(2):
                    var ch = Int(hex.as_bytes()[c])
                    v = v * 16 + (ch - 48 if ch <= 57 else ch - 55)
                self.byte_ids[v] = i
        self.unk_score = min_score - 10.0

    @staticmethod
    def load(st: SafeTensors) raises -> SpmUnigram:
        var b = load_f32(st, "spm.bytes")
        var offs = load_f32(st, "spm.offsets")
        var sc = load_f32(st, "spm.scores")
        var ty = load_f32(st, "spm.types")
        var n = sc.n
        var pieces = List[String]()
        var scores = List[Float32]()
        var types = List[Int]()
        for i in range(n):
            var a = Int(offs.at(i))
            var e = Int(offs.at(i + 1))
            var bytes = List[UInt8]()
            for j in range(a, e):
                bytes.append(UInt8(Int(b.at(j))))
            pieces.append(String(unsafe_from_utf8=bytes))
            scores.append(sc.at(i))
            types.append(Int(ty.at(i)))
        return SpmUnigram(pieces, scores, types)

    @staticmethod
    def normalize(text: String) -> String:
        """nmt_nfkc whitespace rules: trim, collapse runs, dummy prefix, ' ' -> U+2581."""
        var sep: List[UInt8] = [0xE2, 0x96, 0x81]
        var out = sep.copy()
        var pending = False
        var any = False
        var bs = text.as_bytes()
        for i in range(len(bs)):
            var c = Int(bs[i])
            if c == 32 or c == 9 or c == 10 or c == 13:
                pending = any
                continue
            if pending:
                out.extend(sep.copy())
                pending = False
            out.append(bs[i])
            any = True
        return String(unsafe_from_utf8=out)

    def encode(self, text: String) -> List[Int]:
        """`EncodeAsIds(text)` (text already lower-cased by the caller if wanted)."""
        var norm = SpmUnigram.normalize(text)
        var bs = norm.as_bytes()
        var n = len(bs)
        var best = List[Float32](length=n + 1, fill=-3.4e38)
        var back = List[Int](length=n + 1, fill=-1)     # start position of the last piece
        var piece = List[Int](length=n + 1, fill=-1)    # piece id, -1 = unknown char
        best[0] = 0
        for s in range(n):
            if best[s] <= -3.0e38:
                continue
            if (Int(bs[s]) & 0xC0) == 0x80:
                continue  # not a char boundary
            var clen = 1
            var c0 = Int(bs[s])
            if c0 >= 0xF0:
                clen = 4
            elif c0 >= 0xE0:
                clen = 3
            elif c0 >= 0xC0:
                clen = 2
            var found_char = False
            for l in range(1, min(self.max_len, n - s) + 1):
                var e = s + l
                if e < n and (Int(bs[e]) & 0xC0) == 0x80:
                    continue
                var sub = String(norm[byte=s:e])
                var hit = self.ids.find(sub)
                if hit:
                    var id = hit.value()
                    if l == clen:
                        found_char = True
                    var sc = best[s] + self.scores[id]
                    if sc > best[e]:
                        best[e] = sc
                        back[e] = s
                        piece[e] = id
            if not found_char and s + clen <= n:
                var sc = best[s] + self.unk_score
                if sc > best[s + clen]:
                    best[s + clen] = sc
                    back[s + clen] = s
                    piece[s + clen] = -1
        var rev = List[Int]()
        var e = n
        while e > 0:
            var s = back[e]
            if piece[e] >= 0:
                rev.append(piece[e])
            else:
                # byte fallback, reversed here so the final reverse restores byte order
                var j = e - 1
                while j >= s:
                    var bid = self.byte_ids[Int(bs[j])]
                    rev.append(bid if bid >= 0 else self.unk_id)
                    j -= 1
            e = s
        var ids = List[Int]()
        var k = len(rev) - 1
        while k >= 0:
            ids.append(rev[k])
            k -= 1
        return ids^


# ------------------------------------------------------------- numpy RandomState (mapper noise)

struct NumpyRandomState(Movable):
    """`numpy.random.RandomState(seed)`: MT19937 with the legacy polar Gaussian, bit-exact."""
    var mt: List[UInt32]
    var idx: Int
    var has_gauss: Bool
    var gauss: Float64

    def __init__(out self, seed: Int):
        self.mt = List[UInt32](length=624, fill=0)
        self.mt[0] = UInt32(seed & 0xFFFFFFFF)
        for i in range(1, 624):
            var prev = self.mt[i - 1]
            self.mt[i] = UInt32(1812433253) * (prev ^ (prev >> 30)) + UInt32(i)
        self.idx = 624
        self.has_gauss = False
        self.gauss = 0

    def next_u32(mut self) -> UInt32:
        if self.idx >= 624:
            for i in range(624):
                var y = (self.mt[i] & 0x80000000) | (self.mt[(i + 1) % 624] & 0x7FFFFFFF)
                var v = self.mt[(i + 397) % 624] ^ (y >> 1)
                if (y & 1) != 0:
                    v ^= 0x9908B0DF
                self.mt[i] = v
            self.idx = 0
        var y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    def next_double(mut self) -> Float64:
        var a = Float64(Int(self.next_u32() >> 5))
        var b = Float64(Int(self.next_u32() >> 6))
        return (a * 67108864.0 + b) / 9007199254740992.0

    def randn(mut self) -> Float64:
        if self.has_gauss:
            self.has_gauss = False
            return self.gauss
        while True:
            var x1 = 2.0 * self.next_double() - 1.0
            var x2 = 2.0 * self.next_double() - 1.0
            var r2 = x1 * x1 + x2 * x2
            if r2 < 1.0 and r2 != 0.0:
                var f = sqrt(-2.0 * log(r2) / r2)
                self.gauss = f * x1
                self.has_gauss = True
                return f * x2


# ---------------------------------------------------------------------------- mapper

comptime _MAP_LAYERS = 8
comptime _MAP_TOK = 12
comptime _MAP_D = 256
comptime _MAP_HEADS = 8
comptime _MAP_HD = 64
comptime _MAP_FF = 1024
comptime _MAP_CTX = 1024


struct MusicCoCaMapper(Movable):
    """Text -> audio-space style mapper (`use_mapper=True` in Magenta RT): from a text embedding
    and N(0, 1) noise, 8 AdaLN transformer layers over 12 x 256 tokens (RMSNorm, RoPE, an
    attention sink, logits capped by 30 tanh(x / 30), tanh-GELU 1024), then an L2-normalised
    768-d embedding. Different noise seeds give different renditions of the same prompt."""
    var in_w: FBuf
    var in_b: FBuf
    var out_w: FBuf
    var out_b: FBuf
    var ctx: FBuf
    var freq: FBuf
    var mod1_w: FBuf
    var mod1_b: FBuf
    var norm1: FBuf
    var qkv: FBuf
    var sink_k: FBuf
    var sink_v: FBuf
    var wo: FBuf
    var mod2_w: FBuf
    var mod2_b: FBuf
    var b1: FBuf
    var w1: FBuf
    var b2: FBuf
    var w2: FBuf
    var norm2: FBuf
    var c: FBuf
    var x: FBuf
    var h: FBuf
    var m: FBuf
    var t: FBuf
    var o: FBuf
    var f: FBuf

    def __init__(out self, st: SafeTensors) raises:
        self.in_w = load_f32(st, "mapper.in_w")
        self.in_b = load_f32(st, "mapper.in_b")
        self.out_w = load_f32(st, "mapper.out_w")
        self.out_b = load_f32(st, "mapper.out_b")
        self.ctx = load_f32(st, "mapper.ctx")
        self.freq = load_f32(st, "mapper.rope_freq")
        self.mod1_w = load_f32(st, "mapper.mod1_w")
        self.mod1_b = load_f32(st, "mapper.mod1_b")
        self.norm1 = load_f32(st, "mapper.norm1")
        self.qkv = load_f32(st, "mapper.qkv")
        self.sink_k = load_f32(st, "mapper.sink_k")
        self.sink_v = load_f32(st, "mapper.sink_v")
        self.wo = load_f32(st, "mapper.wo")
        self.mod2_w = load_f32(st, "mapper.mod2_w")
        self.mod2_b = load_f32(st, "mapper.mod2_b")
        self.b1 = load_f32(st, "mapper.b1")
        self.w1 = load_f32(st, "mapper.w1")
        self.b2 = load_f32(st, "mapper.b2")
        self.w2 = load_f32(st, "mapper.w2")
        self.norm2 = load_f32(st, "mapper.norm2")
        self.c = FBuf(_MAP_CTX)
        self.x = FBuf(_MAP_TOK * _MAP_D)
        self.h = FBuf(_MAP_TOK * _MAP_D)
        self.m = FBuf(2 * _MAP_D)
        self.t = FBuf(_MAP_TOK * 3 * _MAP_HEADS * _MAP_HD)
        self.o = FBuf(_MAP_TOK * _MAP_HEADS * _MAP_HD)
        self.f = FBuf(_MAP_TOK * _MAP_FF)

    def _adanorm(mut self, mut pool: JobPool, l: Int, w: Int, b: Int, g: Int):
        """h = rmsnorm(x) * g * (1 + m0) + m1 with [m0; m1] = ctx @ W + b (layer l)."""
        comptime D = _MAP_D
        matmul_2d(pool, self.c.addr, w + l * _MAP_CTX * 2 * D * 4, self.m.addr, b + l * 2 * D * 4, 1, _MAP_CTX, 2 * D, False)
        for i in range(_MAP_TOK):
            var ms: Float32 = 0
            for d in range(D):
                var v = self.x.at(i * D + d)
                ms += v * v
            var r = 1.0 / sqrt(ms / Float32(D) + 1e-6)
            for d in range(D):
                var v = self.x.at(i * D + d) * r * fp(g + (l * D + d) * 4).unsafe_load()
                self.h.set(i * D + d, v * (1.0 + self.m.at(d)) + self.m.at(D + d))

    def _rope(mut self, base: Int):
        """Rotates the [12, 8, 64] q or k block at offset `base` of the qkv rows."""
        comptime ROW = 3 * _MAP_HEADS * _MAP_HD
        for i in range(_MAP_TOK):
            for hh in range(_MAP_HEADS):
                var p = self.t.addr + (i * ROW + base + hh * _MAP_HD) * 4
                for j in range(32):
                    var ang = Float32(i) * self.freq.at(j)
                    var cs = cos(ang)
                    var sn = sin(ang)
                    var a = fp(p + j * 4).unsafe_load()
                    var b = fp(p + (j + 32) * 4).unsafe_load()
                    fp(p + j * 4).unsafe_store(a * cs - b * sn)
                    fp(p + (j + 32) * 4).unsafe_store(b * cs + a * sn)

    def map(mut self, mut pool: JobPool, emb: List[Float32], noise: List[Float32]) -> List[Float32]:
        comptime D = _MAP_D
        comptime HW = _MAP_HEADS * _MAP_HD
        comptime ROW = 3 * HW
        for i in range(128):
            self.c.set(i, self.ctx.at(i))
            self.c.set(128 + i, self.ctx.at(i))
        for i in range(768):
            self.c.set(256 + i, emb[i])
        var nz = FBuf(768)
        for i in range(768):
            nz.set(i, noise[i])
        matmul_2d(pool, nz.addr, self.in_w.addr, self.x.addr, self.in_b.addr, 1, 768, _MAP_TOK * D, True)
        _ = nz.n
        var lg = List[Float32](length=_MAP_TOK + 1, fill=0)
        for l in range(_MAP_LAYERS):
            self._adanorm(pool, l, self.mod1_w.addr, self.mod1_b.addr, self.norm1.addr)
            matmul_2d(pool, self.h.addr, self.qkv.addr + l * D * ROW * 4, self.t.addr, 0, _MAP_TOK, D, ROW, False)
            self._rope(0)
            self._rope(HW)
            for hh in range(_MAP_HEADS):
                var sk = self.sink_k.addr + (l * HW + hh * _MAP_HD) * 4
                var sv = self.sink_v.addr + (l * HW + hh * _MAP_HD) * 4
                for i in range(_MAP_TOK):
                    var q = self.t.addr + (i * ROW + hh * _MAP_HD) * 4
                    var dot: Float32 = 0
                    for d in range(_MAP_HD):
                        dot += fp(sk + d * 4).unsafe_load() * fp(q + d * 4).unsafe_load()
                    lg[0] = tanh(dot / 30.0) * 30.0
                    var mx = lg[0]
                    for j in range(_MAP_TOK):
                        var k = self.t.addr + (j * ROW + HW + hh * _MAP_HD) * 4
                        var a: Float32 = 0
                        for d in range(_MAP_HD):
                            a += fp(q + d * 4).unsafe_load() * 0.125 * fp(k + d * 4).unsafe_load()
                        lg[j + 1] = tanh(a / 30.0) * 30.0
                        mx = max(mx, lg[j + 1])
                    var s: Float32 = 0
                    for j in range(_MAP_TOK + 1):
                        lg[j] = exp(lg[j] - mx)
                        s += lg[j]
                    var o = self.o.addr + (i * HW + hh * _MAP_HD) * 4
                    for d in range(_MAP_HD):
                        var acc = lg[0] * fp(sv + d * 4).unsafe_load()
                        for j in range(_MAP_TOK):
                            acc += lg[j + 1] * self.t.at(j * ROW + 2 * HW + hh * _MAP_HD + d)
                        fp(o + d * 4).unsafe_store(acc / s)
            matmul_2d(pool, self.o.addr, self.wo.addr + l * D * HW * 4, self.h.addr, 0, _MAP_TOK, HW, D, True)
            for i in range(_MAP_TOK * D):
                self.x.set(i, self.x.at(i) + self.h.at(i))
            self._adanorm(pool, l, self.mod2_w.addr, self.mod2_b.addr, self.norm2.addr)
            matmul_2d(pool, self.h.addr, self.w1.addr + l * D * _MAP_FF * 4, self.f.addr, self.b1.addr + l * _MAP_FF * 4, _MAP_TOK, D, _MAP_FF, False)
            for i in range(_MAP_TOK * _MAP_FF):
                var u = self.f.at(i)
                self.f.set(i, 0.5 * u * (1.0 + tanh(0.7978845608028654 * (u + 0.044715 * u * u * u))))
            matmul_2d(pool, self.f.addr, self.w2.addr + l * _MAP_FF * D * 4, self.h.addr, self.b2.addr + l * D * 4, _MAP_TOK, _MAP_FF, D, False)
            for i in range(_MAP_TOK * D):
                self.x.set(i, self.x.at(i) + self.h.at(i))
        var y = FBuf(768)
        matmul_2d(pool, self.x.addr, self.out_w.addr, y.addr, self.out_b.addr, 1, _MAP_TOK * D, 768, True)
        var nrm: Float32 = 0
        for i in range(768):
            nrm += y.at(i) * y.at(i)
        nrm = sqrt(nrm)
        var e = List[Float32](capacity=768)
        for i in range(768):
            e.append(y.at(i) / nrm)
        return e^


# ---------------------------------------------------------------------------- encoder

struct _CocaLayer(Movable):
    var ln1_g: FBuf
    var ln1_b: FBuf
    var wq: FBuf
    var bq: FBuf
    var wk: FBuf
    var bk: FBuf
    var wv: FBuf
    var bv: FBuf
    var wo: FBuf
    var bo: FBuf
    var ln2_g: FBuf
    var ln2_b: FBuf
    var w1: FBuf
    var b1: FBuf
    var w2: FBuf
    var b2: FBuf

    def __init__(out self, st: SafeTensors, p: String) raises:
        self.ln1_g = load_f32(st, p + "ln1_g")
        self.ln1_b = load_f32(st, p + "ln1_b")
        self.wq = load_f32(st, p + "wq")
        self.bq = load_f32(st, p + "bq")
        self.wk = load_f32(st, p + "wk")
        self.bk = load_f32(st, p + "bk")
        self.wv = load_f32(st, p + "wv")
        self.bv = load_f32(st, p + "bv")
        self.wo = load_f32(st, p + "wo")
        self.bo = load_f32(st, p + "bo")
        self.ln2_g = load_f32(st, p + "ln2_g")
        self.ln2_b = load_f32(st, p + "ln2_b")
        self.w1 = load_f32(st, p + "w1")
        self.b1 = load_f32(st, p + "b1")
        self.w2 = load_f32(st, p + "w2")
        self.b2 = load_f32(st, p + "b2")


def blend_embeddings(a: List[Float32], b: List[Float32], t: Float32) -> List[Float32]:
    """(1 - t) a + t b: weighted style mix (`np.mean` of two prompts is t = 0.5)."""
    var o = List[Float32](capacity=len(a))
    for i in range(len(a)):
        o.append((1 - t) * a[i] + t * b[i])
    return o^


struct MusicCoCaText(Movable):
    var spm: SpmUnigram
    var tok_emb: FBuf
    var pos_emb: FBuf
    var layers: List[_CocaLayer]
    var ln_f_g: FBuf
    var ln_f_b: FBuf
    var pool_wk: FBuf
    var pool_bk: FBuf
    var pool_wv: FBuf
    var pool_bv: FBuf
    var pool_wo: FBuf
    var pool_bo: FBuf
    var pool_q: FBuf
    var ln_out_g: FBuf
    var ln_out_b: FBuf
    var rvq: RvqCodebook
    var mapper: List[MusicCoCaMapper]   # empty when the file has no `mapper.*` tensors
    var pool: JobPool
    # scratch
    var x: FBuf
    var h: FBuf
    var q: FBuf
    var k: FBuf
    var v: FBuf
    var o: FBuf
    var f: FBuf
    var att: FBuf
    var pooled: FBuf
    var proj: FBuf

    def __init__(out self, st: SafeTensors) raises:
        self.spm = SpmUnigram.load(st)
        self.tok_emb = load_f32(st, "tok_emb")
        self.pos_emb = load_f32(st, "pos_emb")
        self.layers = List[_CocaLayer]()
        for l in range(COCA_LAYERS):
            self.layers.append(_CocaLayer(st, "layers." + String(l) + "."))
        self.ln_f_g = load_f32(st, "ln_f_g")
        self.ln_f_b = load_f32(st, "ln_f_b")
        self.pool_wk = load_f32(st, "pool.wk")
        self.pool_bk = load_f32(st, "pool.bk")
        self.pool_wv = load_f32(st, "pool.wv")
        self.pool_bv = load_f32(st, "pool.bv")
        self.pool_wo = load_f32(st, "pool.wo")
        self.pool_bo = load_f32(st, "pool.bo")
        self.pool_q = load_f32(st, "pool.query")
        self.ln_out_g = load_f32(st, "ln_out_g")
        self.ln_out_b = load_f32(st, "ln_out_b")
        self.rvq = RvqCodebook.load(st, "rvq.codebooks")
        self.mapper = List[MusicCoCaMapper]()
        if st.has("mapper.in_w"):
            self.mapper.append(MusicCoCaMapper(st))
        self.pool = JobPool()
        _ = self.pool.start(max(1, _cpu_count() - 2))
        var t = COCA_MAX_TOKENS
        self.x = FBuf(t * COCA_DIM)
        self.h = FBuf(t * COCA_DIM)
        self.q = FBuf(t * COCA_DIM)
        self.k = FBuf(t * COCA_FF)
        self.v = FBuf(t * COCA_FF)
        self.o = FBuf(t * COCA_DIM)
        self.f = FBuf(t * COCA_FF)
        self.att = FBuf(t)
        self.pooled = FBuf(COCA_FF)
        self.proj = FBuf(COCA_DIM)

    @staticmethod
    def load(path: String) raises -> MusicCoCaText:
        """`path`: `musiccoca_text.safetensors` from `tools/magenta_rt/convert_musiccoca.py`."""
        var st = SafeTensors()
        st.add_file(path)
        return MusicCoCaText(st)

    def token_ids(self, text: String) -> List[Int]:
        """SOS (1) + `EncodeAsIds(text.lower())`, cut to 128 tokens."""
        var ids = List[Int]()
        ids.append(1)
        var pieces = self.spm.encode(text.lower())
        for i in range(min(len(pieces), COCA_MAX_TOKENS - 1)):
            ids.append(pieces[i])
        return ids^

    def embed(mut self, text: String) -> List[Float32]:
        """768-d style embedding of a text prompt (matches the TFLite text encoder)."""
        return self.embed_ids(self.token_ids(text))

    def embed_ids(mut self, ids: List[Int]) -> List[Float32]:
        var n = len(ids)
        comptime D = COCA_DIM
        comptime HD = D // COCA_HEADS
        for t in range(n):
            var row = self.x.addr + t * D * 4
            copy_floats(row, self.tok_emb.addr + ids[t] * D * 4, D)
            for d in range(D):
                var px = fp(row + d * 4)
                px.unsafe_store(px.unsafe_load() + self.pos_emb.at(t * D + d))
        for li in range(COCA_LAYERS):
            ref L = self.layers[li]
            _layer_norm_rows(self.x.addr, L.ln1_g.addr, L.ln1_b.addr, self.h.addr, n, D)
            matmul_2d(self.pool, self.h.addr, L.wq.addr, self.q.addr, L.bq.addr, n, D, D, True)
            matmul_2d(self.pool, self.h.addr, L.wk.addr, self.k.addr, L.bk.addr, n, D, D, True)
            matmul_2d(self.pool, self.h.addr, L.wv.addr, self.v.addr, L.bv.addr, n, D, D, True)
            for hh in range(COCA_HEADS):
                for i in range(n):
                    var qi = self.q.addr + (i * D + hh * HD) * 4
                    for j in range(n):
                        var kj = self.k.addr + (j * D + hh * HD) * 4
                        var dot: Float32 = 0
                        for d in range(HD):
                            dot += fp(qi + d * 4).unsafe_load() * fp(kj + d * 4).unsafe_load()
                        self.att.set(j, tanh(dot * 0.02) * 50.0)
                    _softmax_inplace(self.att.addr, n)
                    var oi = self.o.addr + (i * D + hh * HD) * 4
                    for d in range(HD):
                        var acc: Float32 = 0
                        for j in range(n):
                            acc += self.att.at(j) * self.v.at(j * D + hh * HD + d)
                        fp(oi + d * 4).unsafe_store(acc)
            matmul_2d(self.pool, self.o.addr, L.wo.addr, self.h.addr, L.bo.addr, n, D, D, True)
            for i in range(n * D):
                self.x.set(i, self.x.at(i) + self.h.at(i))
            _layer_norm_rows(self.x.addr, L.ln2_g.addr, L.ln2_b.addr, self.h.addr, n, D)
            matmul_2d(self.pool, self.h.addr, L.w1.addr, self.f.addr, L.b1.addr, n, D, COCA_FF, True)
            for i in range(n * COCA_FF):
                var u = self.f.at(i)
                self.f.set(i, 0.5 * u * (1.0 + erf(u * 0.7071067811865476)))
            matmul_2d(self.pool, self.f.addr, L.w2.addr, self.h.addr, L.b2.addr, n, COCA_FF, D, True)
            for i in range(n * D):
                self.x.set(i, self.x.at(i) + self.h.at(i))
        # attention pooling
        _layer_norm_rows(self.x.addr, self.ln_f_g.addr, self.ln_f_b.addr, self.h.addr, n, D)
        matmul_2d(self.pool, self.h.addr, self.pool_wk.addr, self.k.addr, self.pool_bk.addr, n, D, COCA_FF, True)
        matmul_2d(self.pool, self.h.addr, self.pool_wv.addr, self.v.addr, self.pool_bv.addr, n, D, COCA_FF, True)
        comptime PH = COCA_POOL_HEAD
        for hh in range(COCA_HEADS):
            for j in range(n):
                var dot: Float32 = 0
                for d in range(PH):
                    dot += self.pool_q.at(hh * PH + d) * self.k.at(j * COCA_FF + hh * PH + d)
                self.att.set(j, dot)
            _softmax_inplace(self.att.addr, n)
            for d in range(PH):
                var acc: Float32 = 0
                for j in range(n):
                    acc += self.att.at(j) * self.v.at(j * COCA_FF + hh * PH + d)
                self.pooled.set(hh * PH + d, acc)
        matmul_2d(self.pool, self.pooled.addr, self.pool_wo.addr, self.proj.addr, self.pool_bo.addr, 1, COCA_FF, D, True)
        _layer_norm_rows(self.proj.addr, self.ln_out_g.addr, self.ln_out_b.addr, self.h.addr, 1, D)
        var e = List[Float32](capacity=D)
        for d in range(D):
            e.append(self.h.at(d))
        return e^

    def tokenize(self, emb: List[Float32], levels: Int = COCA_RVQ_LEVELS) -> List[Int]:
        """Greedy RVQ codes (0..1023 per level) of a style embedding."""
        var b = FBuf(COCA_DIM)
        for d in range(COCA_DIM):
            b.set(d, emb[d])
        var codes = self.rvq.encode(b.addr, levels)
        _ = b.n
        return codes^

    def has_mapper(self) -> Bool:
        return len(self.mapper) > 0

    def map_to_audio(mut self, emb: List[Float32], seed: Int = 0) -> List[Float32]:
        """`use_mapper=True`: text embedding -> unit-norm audio-space embedding (noise from
        `numpy.random.RandomState(seed).randn(768)`). Returns `emb` unchanged without a mapper."""
        if len(self.mapper) == 0:
            return emb.copy()
        var rs = NumpyRandomState(seed)
        var noise = List[Float32](capacity=COCA_DIM)
        for _ in range(COCA_DIM):
            noise.append(Float32(rs.randn()))
        return self.mapper[0].map(self.pool, emb, noise)

    def embed_style(mut self, text: String, use_mapper: Bool = True, seed: Int = 0) -> List[Float32]:
        """Style embedding as Magenta RT's text prompting builds it (mapper on by default)."""
        var e = self.embed(text)
        if use_mapper:
            return self.map_to_audio(e, seed)
        return e^

    def style_tokens(mut self, text: String, use_mapper: Bool = True, seed: Int = 0) -> List[Int]:
        """12 style codes for `MagentaRT.set_style` from a text prompt."""
        return self.tokenize(self.embed_style(text, use_mapper, seed))
