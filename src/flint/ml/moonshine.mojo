"""Moonshine v2 streaming speech-to-text in pure Mojo (`flint.ml.moonshine`).

Runs the released `moonshine-ai/moonshine-streaming-{tiny,small,medium}` checkpoints (MIT) from
their Hugging Face folder as is: `config.json`, `model.safetensors` (mmapped, zero copy) and
`tokenizer.json`. No conversion step.

Architecture (matching `transformers`' `MoonshineStreamingForConditionalGeneration`):

  - Embedder: 16 kHz audio in 5 ms frames (80 samples), per-frame CMVN, asinh(k x), a linear
    80 -> D with SiLU, then two causal stride-2 Conv1d (k = 5; D -> 2D -> D, SiLU between),
    so one encoder frame is 20 ms.
  - Encoder: pre-LN blocks (LayerNorm without affine times (gamma + 1)), multi-head attention
    over a sliding window (16 frames back, and 3 ahead on the first and last two layers; no
    positional encoding), GELU MLP, final norm.
  - Decoder: learned positions added to the encoder states (then a projection when the
    encoder is wider), pre-LN blocks with causal self-attention under interleaved partial RoPE,
    cross-attention, SiLU-gated MLP; untied LM head. Greedy decoding with a KV cache, at most
    6.5 tokens per second of audio, as the model card recommends.

Streaming: `transcribe` works on one utterance (a voice command, cut by `EnergyVad`); the
live demo shows partial transcripts by re-running the growing utterance about every 0.5 s,
which costs a few tens of milliseconds on the small models. Reusing encoder states across
calls, which the model's bounded attention window allows, is not implemented yet.

`Moonshine.load(dir, gpu)` picks the MAX GPU when asked and one is present
(`flint.ml.moonshine_gpu`: weights resident, encoder and decoder steps on the device, only the
logits come back); otherwise linear layers run on the CPU job pool.
"""

from std.math import sqrt, exp, erf, sin, cos
from std.collections import Optional, Dict
from std.time import perf_counter_ns
from flint.core.jobs import JobPool, default_worker_count
from flint.assets.json import parse_json
from flint.platform.os import read_bytes
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.ops import matmul_2d
from flint.ml.safetensors import SafeTensors
from flint.ml.moonshine_tokenizer import MoonshineTokenizer
from flint.ml.moonshine_audio import AudioClip, read_wav, to_model_rate, moonshine_frames, MOONSHINE_FRAME, MOONSHINE_RATE
from flint.ml.gpu import FP
from flint.ml.moonshine_gpu import MoonshineGpu, ACT_GELU, ACT_SILU, ACT_MASK

comptime TOKENS_PER_SECOND: Float64 = 6.5


struct MoonshineConfig(Copyable, Movable):
    var enc_hidden: Int
    var enc_heads: Int
    var enc_head_dim: Int
    var enc_layers: Int
    var enc_ffn: Int
    var win_left: List[Int]
    var win_right: List[Int]
    var dec_hidden: Int
    var dec_heads: Int
    var dec_head_dim: Int
    var dec_layers: Int
    var dec_ffn: Int
    var rot_dim: Int
    var rope_theta: Float64
    var vocab: Int
    var max_pos: Int
    var bos: Int
    var eos: Int

    def __init__(out self):
        self.enc_hidden = 0
        self.enc_heads = 0
        self.enc_head_dim = 0
        self.enc_layers = 0
        self.enc_ffn = 0
        self.win_left = List[Int]()
        self.win_right = List[Int]()
        self.dec_hidden = 0
        self.dec_heads = 0
        self.dec_head_dim = 0
        self.dec_layers = 0
        self.dec_ffn = 0
        self.rot_dim = 0
        self.rope_theta = 10000
        self.vocab = 0
        self.max_pos = 0
        self.bos = 1
        self.eos = 2

    @staticmethod
    def load(path: String) raises -> MoonshineConfig:
        var data = read_bytes(path)
        var doc = parse_json(data)
        if not doc.valid:
            raise Error("moonshine: cannot read " + path)
        var c = MoonshineConfig()
        var r = doc.root
        var e = doc.object_get(r, "encoder_config")
        c.enc_hidden = doc.integer(doc.object_get(e, "hidden_size"), 0)
        c.enc_heads = doc.integer(doc.object_get(e, "num_attention_heads"), 0)
        c.enc_head_dim = doc.integer(doc.object_get(e, "head_dim"), c.enc_hidden // max(c.enc_heads, 1))
        c.enc_layers = doc.integer(doc.object_get(e, "num_hidden_layers"), 0)
        c.enc_ffn = doc.integer(doc.object_get(e, "intermediate_size"), 0)
        var w = doc.object_get(e, "sliding_windows")
        for i in range(doc.array_size(w)):
            c.win_left.append(doc.integer(doc.at(doc.at(w, i), 0), 16))
            c.win_right.append(doc.integer(doc.at(doc.at(w, i), 1), 0))
        c.dec_hidden = doc.integer(doc.object_get(r, "hidden_size"), 0)
        c.dec_heads = doc.integer(doc.object_get(r, "num_attention_heads"), 0)
        c.dec_head_dim = doc.integer(doc.object_get(r, "head_dim"), c.dec_hidden // max(c.dec_heads, 1))
        c.dec_layers = doc.integer(doc.object_get(r, "num_hidden_layers"), 0)
        c.dec_ffn = doc.integer(doc.object_get(r, "intermediate_size"), 0)
        c.vocab = doc.integer(doc.object_get(r, "vocab_size"), 0)
        c.max_pos = doc.integer(doc.object_get(r, "max_position_embeddings"), 4096)
        c.bos = doc.integer(doc.object_get(r, "decoder_start_token_id"), 1)
        c.eos = doc.integer(doc.object_get(r, "eos_token_id"), 2)
        var rp = doc.object_get(r, "rope_parameters")
        var factor = doc.number(doc.object_get(rp, "partial_rotary_factor"), 1.0)
        c.rope_theta = doc.number(doc.object_get(rp, "rope_theta"), 10000.0)
        c.rot_dim = Int(Float64(c.dec_head_dim) * factor)
        if c.enc_layers == 0 or c.dec_layers == 0 or len(c.win_left) != c.enc_layers:
            raise Error("moonshine: unexpected config " + path)
        return c^


struct MoonshineTranscript(Movable):
    var text: String
    var ids: List[Int]
    var audio_seconds: Float64
    var encode_ms: Float64
    var decode_ms: Float64
    var tokens: Int

    def __init__(out self):
        self.text = String()
        self.ids = List[Int]()
        self.audio_seconds = 0
        self.encode_ms = 0
        self.decode_ms = 0
        self.tokens = 0

    def total_ms(self) -> Float64:
        return self.encode_ms + self.decode_ms

    def rtf(self) -> Float64:
        """Real-time factor: compute time / audio time (below 1 is faster than real time)."""
        if self.audio_seconds <= 0:
            return 0
        return self.total_ms() / 1000.0 / self.audio_seconds


@always_inline
def _silu(x: Float32) -> Float32:
    return x / (1.0 + exp(-x))


@always_inline
def _gelu(x: Float32) -> Float32:
    return 0.5 * x * (1.0 + erf(x * 0.7071067811865476))


def _norm_rows(src: Int, dst: Int, rows: Int, n: Int, gamma: Int, offset: Float32):
    """LayerNorm over the last axis (eps 1e-5, no bias) times (gamma + offset)."""
    var ps = fptr_at(src)
    var pd = fptr_at(dst)
    var pg = fptr_at(gamma)
    for r in range(rows):
        var b = r * n
        var s: Float64 = 0
        for i in range(n):
            s += Float64(ps.unsafe_offset(b + i).unsafe_load())
        var mean = Float32(s / Float64(n))
        var q: Float64 = 0
        for i in range(n):
            var d = ps.unsafe_offset(b + i).unsafe_load() - mean
            q += Float64(d * d)
        var inv = Float32(1.0 / sqrt(q / Float64(n) + 1e-5))
        for i in range(n):
            var g = pg.unsafe_offset(i).unsafe_load[alignment=1]() + offset
            pd.unsafe_offset(b + i).unsafe_store((ps.unsafe_offset(b + i).unsafe_load() - mean) * inv * g)


def _add_into(dst: Int, src: Int, n: Int):
    var pd = fptr_at(dst)
    var ps = fptr_at(src)
    for i in range(n):
        pd.unsafe_offset(i).unsafe_store(pd.unsafe_offset(i).unsafe_load() + ps.unsafe_offset(i).unsafe_load())


def _attend(q: Int, k: Int, v: Int, o: Int, qrow: Int, kvrow: Int, heads: Int, hd: Int, keys: List[Int],
            scores: Int, scale: Float32):
    """One query row, all heads, over the listed key rows (row strides `qrow` / `kvrow` floats)."""
    var pq = fptr_at(q)
    var pk = fptr_at(k)
    var pv = fptr_at(v)
    var po = fptr_at(o)
    var ps = fptr_at(scores)
    var nk = len(keys)
    for h in range(heads):
        var qb = h * hd
        for i in range(hd):
            po.unsafe_offset(qb + i).unsafe_store(0)
        if nk == 0:
            continue
        var m: Float32 = -3.0e38
        for j in range(nk):
            var kb = keys[j] * kvrow + h * hd
            var s: Float32 = 0
            for i in range(hd):
                s += pq.unsafe_offset(qb + i).unsafe_load() * pk.unsafe_offset(kb + i).unsafe_load()
            s *= scale
            ps.unsafe_offset(j).unsafe_store(s)
            m = max(m, s)
        var tot: Float32 = 0
        for j in range(nk):
            var e = exp(ps.unsafe_offset(j).unsafe_load() - m)
            ps.unsafe_offset(j).unsafe_store(e)
            tot += e
        var inv = 1.0 / tot
        for j in range(nk):
            var wgt = ps.unsafe_offset(j).unsafe_load() * inv
            var vb = keys[j] * kvrow + h * hd
            for i in range(hd):
                po.unsafe_offset(qb + i).unsafe_store(po.unsafe_offset(qb + i).unsafe_load() + wgt * pv.unsafe_offset(vb + i).unsafe_load())


def _rope(x: Int, heads: Int, hd: Int, rot: Int, pos: Int, inv_freq: List[Float32]):
    """Interleaved (GPT-J) rotation of the first `rot` dims of each head, in place."""
    var p = fptr_at(x)
    for i in range(rot // 2):
        var ang = Float32(pos) * inv_freq[i]
        var c = cos(ang)
        var s = sin(ang)
        for h in range(heads):
            var b = h * hd + 2 * i
            var x0 = p.unsafe_offset(b).unsafe_load()
            var x1 = p.unsafe_offset(b + 1).unsafe_load()
            p.unsafe_offset(b).unsafe_store(x0 * c - x1 * s)
            p.unsafe_offset(b + 1).unsafe_store(x1 * c + x0 * s)


def _grow(mut buf: List[Float32], n: Int):
    if len(buf) < n:
        buf = List[Float32](length=n, fill=0.0)


@always_inline
def _addr(buf: List[Float32]) -> Int:
    return Int(buf.unsafe_ptr())


struct Moonshine(Movable):
    var dir: String
    var cfg: MoonshineConfig
    var weights: SafeTensors
    var tokenizer: MoonshineTokenizer
    var pool: JobPool
    var gpu: Optional[MoonshineGpu]
    var log_k: Float32
    var inv_freq: List[Float32]
    # encoder scratch
    var e_x: List[Float32]
    var e_h: List[Float32]
    var e_q: List[Float32]
    var e_k: List[Float32]
    var e_v: List[Float32]
    var e_a: List[Float32]
    var e_f: List[Float32]
    var e_col: List[Float32]
    var scores: List[Float32]
    # the last encoding
    var enc: List[Float32]       # [frames, enc_hidden] (final-norm output)
    var enc_frames: Int
    var enc_valid: Int           # frames whose key mask is on
    # decoder state
    var cross_k: List[List[Float32]]
    var cross_v: List[List[Float32]]
    var self_k: List[List[Float32]]
    var self_v: List[List[Float32]]
    var steps_cap: Int
    var d_x: List[Float32]
    var d_h: List[Float32]
    var d_q: List[Float32]
    var d_k: List[Float32]
    var d_v: List[Float32]
    var d_a: List[Float32]
    var d_f: List[Float32]
    var d_g: List[Float32]
    var mem: List[Float32]       # decoder-side encoder states [frames, dec_hidden]
    var logits: List[Float32]
    var tracing: Bool
    var trace: List[List[Float32]]
    # device state (GPU present): weight keys and pooled slots
    var gkeys: Dict[String, Int]
    var gslots: Dict[String, Int]

    def __init__(out self, dir: String, var cfg: MoonshineConfig, var weights: SafeTensors,
                 var tokenizer: MoonshineTokenizer) raises:
        self.dir = dir
        self.cfg = cfg^
        self.weights = weights^
        self.tokenizer = tokenizer^
        self.pool = JobPool()
        _ = self.pool.start(default_worker_count())
        self.gpu = None
        self.log_k = self.weights.scalar("model.encoder.embedder.comp.log_k")
        self.inv_freq = List[Float32]()
        for i in range(self.cfg.rot_dim // 2):
            self.inv_freq.append(Float32(1.0 / (self.cfg.rope_theta ** (Float64(2 * i) / Float64(self.cfg.rot_dim)))))
        self.e_x = List[Float32]()
        self.e_h = List[Float32]()
        self.e_q = List[Float32]()
        self.e_k = List[Float32]()
        self.e_v = List[Float32]()
        self.e_a = List[Float32]()
        self.e_f = List[Float32]()
        self.e_col = List[Float32]()
        self.scores = List[Float32]()
        self.enc = List[Float32]()
        self.enc_frames = 0
        self.enc_valid = 0
        self.cross_k = List[List[Float32]]()
        self.cross_v = List[List[Float32]]()
        self.self_k = List[List[Float32]]()
        self.self_v = List[List[Float32]]()
        for _ in range(self.cfg.dec_layers):
            self.cross_k.append(List[Float32]())
            self.cross_v.append(List[Float32]())
            self.self_k.append(List[Float32]())
            self.self_v.append(List[Float32]())
        self.steps_cap = 0
        var hd = self.cfg.dec_hidden
        var att = self.cfg.dec_heads * self.cfg.dec_head_dim
        self.d_x = List[Float32](length=hd, fill=0.0)
        self.d_h = List[Float32](length=hd, fill=0.0)
        self.d_q = List[Float32](length=att, fill=0.0)
        self.d_k = List[Float32](length=att, fill=0.0)
        self.d_v = List[Float32](length=att, fill=0.0)
        self.d_a = List[Float32](length=att, fill=0.0)
        self.d_f = List[Float32](length=2 * self.cfg.dec_ffn, fill=0.0)
        self.d_g = List[Float32](length=self.cfg.dec_ffn, fill=0.0)
        self.mem = List[Float32]()
        self.logits = List[Float32](length=self.cfg.vocab, fill=0.0)
        self.tracing = False
        self.trace = List[List[Float32]]()
        self.gkeys = Dict[String, Int]()
        self.gslots = Dict[String, Int]()

    @staticmethod
    def load(dir: String, gpu: Bool = True) raises -> Moonshine:
        """Opens a Hugging Face `moonshine-streaming-*` folder."""
        var cfg = MoonshineConfig.load(dir + "/config.json")
        var w = SafeTensors.open(dir + "/model.safetensors")
        var tok = MoonshineTokenizer.load(dir + "/tokenizer.json")
        var m = Moonshine(dir, cfg^, w^, tok^)
        if gpu:
            try:
                m.gpu = MoonshineGpu(m.cfg.enc_layers, m.cfg.dec_layers)
            except:
                m.gpu = None
        return m^

    def device(self) raises -> String:
        if self.gpu:
            return self.gpu.value().name()
        return String("CPU")

    def _w(self, name: String) raises -> Int:
        return self.weights.weight(name).fptr()

    def _linear(mut self, name: String, x: Int, m: Int, k: Int, n: Int, dst: Int) raises:
        """dst[m, n] = x[m, k] @ W^T (+ b), W stored [n, k] (torch layout)."""
        var w = self.weights.weight(name + ".weight")
        var b = 0
        if self.weights.has(name + ".bias"):
            b = self.weights.weight(name + ".bias").fptr()
        matmul_2d(self.pool, x, w.fptr(), dst, b, m, k, n, True)

    # ------------------------------------------------------------------ encoder

    def encode(mut self, audio: List[Float32]) raises:
        """Encodes 16 kHz mono audio into `enc` [enc_frames, enc_hidden]."""
        if self.gpu:
            self._encode_gpu(audio)
            return
        var c = self.cfg.copy()
        var D = c.enc_hidden
        var n_valid = len(audio)
        var F = (n_valid + MOONSHINE_FRAME - 1) // MOONSHINE_FRAME
        if F < 1:
            F = 1
        var padded = audio.copy()
        padded.resize(F * MOONSHINE_FRAME, 0.0)
        var feats = moonshine_frames(padded, F, exp(self.log_k))
        # frame masks: valid frames, then each causal stride-2 conv keeps any frame whose
        # window touches a valid one
        var nf0 = n_valid // MOONSHINE_FRAME
        var T1 = (F - 1) // 2 + 1
        var T2 = (T1 - 1) // 2 + 1
        var nf1 = 0
        for t in range(T1):
            if 2 * t - 4 < nf0:
                nf1 = t + 1
        var nf2 = 0
        for t in range(T2):
            if 2 * t - 4 < nf1:
                nf2 = t + 1
        _grow(self.e_x, F * 2 * D)
        _grow(self.e_h, F * 2 * D)
        self._linear("model.encoder.embedder.linear", _addr(feats), F, MOONSHINE_FRAME, D, _addr(self.e_x))
        var px = fptr_at(_addr(self.e_x))
        for t in range(F):
            for i in range(D):
                var at = t * D + i
                px.unsafe_offset(at).unsafe_store(_silu(px.unsafe_offset(at).unsafe_load()) if t < nf0 else 0.0)
        if self.tracing:
            self._snap(_addr(self.e_x), F * D)
        self._conv(_addr(self.e_x), F, D, "model.encoder.embedder.conv1", 2 * D, _addr(self.e_h), nf1, True)
        self._conv(_addr(self.e_h), T1, 2 * D, "model.encoder.embedder.conv2", D, _addr(self.e_x), nf2, False)
        var T = T2
        if self.tracing:
            self._snap(_addr(self.e_x), T * D)
        var att = c.enc_heads * c.enc_head_dim
        _grow(self.e_q, T * att)
        _grow(self.e_k, T * att)
        _grow(self.e_v, T * att)
        _grow(self.e_a, T * max(att, D))
        _grow(self.e_f, T * c.enc_ffn)
        _grow(self.scores, 64 + T)
        var scale = Float32(1.0 / sqrt(Float64(c.enc_head_dim)))
        for l in range(c.enc_layers):
            var pre = "model.encoder.layers." + String(l)
            _norm_rows(_addr(self.e_x), _addr(self.e_h), T, D, self._w(pre + ".input_layernorm.gamma"), 1.0)
            self._linear(pre + ".self_attn.q_proj", _addr(self.e_h), T, D, att, _addr(self.e_q))
            self._linear(pre + ".self_attn.k_proj", _addr(self.e_h), T, D, att, _addr(self.e_k))
            self._linear(pre + ".self_attn.v_proj", _addr(self.e_h), T, D, att, _addr(self.e_v))
            var left = c.win_left[l]
            var right = c.win_right[l]
            for i in range(T):
                var keys = List[Int]()
                for j in range(max(0, i - left + 1), min(T, i + max(right, 1))):
                    if j < nf2:
                        keys.append(j)
                _attend(_addr(self.e_q) + i * att * 4, _addr(self.e_k), _addr(self.e_v), _addr(self.e_a) + i * att * 4,
                        att, att, c.enc_heads, c.enc_head_dim, keys, _addr(self.scores), scale)
            self._linear(pre + ".self_attn.o_proj", _addr(self.e_a), T, att, D, _addr(self.e_h))
            _add_into(_addr(self.e_x), _addr(self.e_h), T * D)
            _norm_rows(_addr(self.e_x), _addr(self.e_h), T, D, self._w(pre + ".post_attention_layernorm.gamma"), 1.0)
            self._linear(pre + ".mlp.fc1", _addr(self.e_h), T, D, c.enc_ffn, _addr(self.e_f))
            var pf = fptr_at(_addr(self.e_f))
            for i in range(T * c.enc_ffn):
                pf.unsafe_offset(i).unsafe_store(_gelu(pf.unsafe_offset(i).unsafe_load()))
            self._linear(pre + ".mlp.fc2", _addr(self.e_f), T, c.enc_ffn, D, _addr(self.e_h))
            _add_into(_addr(self.e_x), _addr(self.e_h), T * D)
            if self.tracing:
                self._snap(_addr(self.e_x), T * D)
        self.enc = List[Float32](length=T * D, fill=0.0)
        _norm_rows(_addr(self.e_x), _addr(self.enc), T, D, self._w("model.encoder.final_norm.gamma"), 1.0)
        self.enc_frames = T
        self.enc_valid = nf2

    def _snap(mut self, buf: Int, n: Int):
        var c = List[Float32](capacity=n)
        var p = fptr_at(buf)
        for i in range(n):
            c.append(p.unsafe_offset(i).unsafe_load())
        self.trace.append(c^)

    def _conv(mut self, src: Int, T: Int, cin: Int, name: String, cout: Int, dst: Int, valid: Int, silu: Bool) raises:
        """Causal Conv1d (k = 5, stride 2, 4 frames of left padding) on time-major [T, cin] ->
        [ceil(T / 2), cout]; frames from `valid` on are zeroed, the rest optionally SiLU'd."""
        var To = (T - 1) // 2 + 1
        var kc = cin * 5
        _grow(self.e_col, To * kc)
        var ps = fptr_at(src)
        var pc = fptr_at(_addr(self.e_col))
        for t in range(To):
            for j in range(5):
                var s = 2 * t + j - 4
                for ch in range(cin):
                    var v: Float32 = 0
                    if s >= 0 and s < T:
                        v = ps.unsafe_offset(s * cin + ch).unsafe_load()
                    pc.unsafe_offset(t * kc + ch * 5 + j).unsafe_store(v)
        self._linear(name, _addr(self.e_col), To, kc, cout, dst)
        var pd = fptr_at(dst)
        for t in range(To):
            for i in range(cout):
                var at = t * cout + i
                if t >= valid:
                    pd.unsafe_offset(at).unsafe_store(0)
                elif silu:
                    pd.unsafe_offset(at).unsafe_store(_silu(pd.unsafe_offset(at).unsafe_load()))

    # ------------------------------------------------------------------ decoder

    def start_decoder(mut self, max_steps: Int) raises:
        """Positions + projection on the last encoding, cross-attention keys / values per layer,
        and a self-attention cache for `max_steps` tokens."""
        if self.gpu:
            self._start_decoder_gpu(max_steps)
            return
        var c = self.cfg.copy()
        var T = self.enc_frames
        var De = c.enc_hidden
        var D = c.dec_hidden
        var att = c.dec_heads * c.dec_head_dim
        if T > c.max_pos:
            raise Error("moonshine: " + String(T) + " encoder frames exceed " + String(c.max_pos) + " positions")
        var e = self.enc.copy()
        var pp = fptr_at(self._w("model.decoder.pos_emb.weight"))
        for i in range(T * De):
            e[i] += pp.unsafe_offset(i).unsafe_load[alignment=1]()
        if self.weights.has("model.decoder.proj.weight"):
            self.mem = List[Float32](length=T * D, fill=0.0)
            self._linear("model.decoder.proj", _addr(e), T, De, D, _addr(self.mem))
        else:
            self.mem = e^
        for l in range(c.dec_layers):
            var pre = "model.decoder.layers." + String(l) + ".encoder_attn"
            self.cross_k[l] = List[Float32](length=T * att, fill=0.0)
            self.cross_v[l] = List[Float32](length=T * att, fill=0.0)
            self._linear(pre + ".k_proj", _addr(self.mem), T, D, att, _addr(self.cross_k[l]))
            self._linear(pre + ".v_proj", _addr(self.mem), T, D, att, _addr(self.cross_v[l]))
        if max_steps > self.steps_cap:
            self.steps_cap = max_steps
            for l in range(c.dec_layers):
                self.self_k[l] = List[Float32](length=max_steps * att, fill=0.0)
                self.self_v[l] = List[Float32](length=max_steps * att, fill=0.0)
        _grow(self.scores, 64 + max(T, max_steps))

    def step(mut self, token: Int, pos: Int) raises:
        """Feeds `token` at position `pos` (0 = the start token); fills `logits`."""
        if self.gpu:
            self._step_gpu(token, pos)
            return
        var c = self.cfg.copy()
        var D = c.dec_hidden
        var H = c.dec_heads
        var hd = c.dec_head_dim
        var att = H * hd
        var I = c.dec_ffn
        if pos >= self.steps_cap:
            raise Error("moonshine: decoder step " + String(pos) + " is past the cache")
        var pe = fptr_at(self._w("model.decoder.embed_tokens.weight"))
        var px = fptr_at(_addr(self.d_x))
        for i in range(D):
            px.unsafe_offset(i).unsafe_store(pe.unsafe_offset(token * D + i).unsafe_load[alignment=1]())
        var scale = Float32(1.0 / sqrt(Float64(hd)))
        var self_keys = List[Int]()
        for j in range(pos + 1):
            self_keys.append(j)
        var cross_keys = List[Int]()
        for j in range(self.enc_valid):
            cross_keys.append(j)
        for l in range(c.dec_layers):
            var pre = "model.decoder.layers." + String(l)
            _norm_rows(_addr(self.d_x), _addr(self.d_h), 1, D, self._w(pre + ".input_layernorm.weight"), 0.0)
            self._linear(pre + ".self_attn.q_proj", _addr(self.d_h), 1, D, att, _addr(self.d_q))
            var kslot = _addr(self.self_k[l]) + pos * att * 4
            var vslot = _addr(self.self_v[l]) + pos * att * 4
            self._linear(pre + ".self_attn.k_proj", _addr(self.d_h), 1, D, att, kslot)
            self._linear(pre + ".self_attn.v_proj", _addr(self.d_h), 1, D, att, vslot)
            _rope(_addr(self.d_q), H, hd, c.rot_dim, pos, self.inv_freq)
            _rope(kslot, H, hd, c.rot_dim, pos, self.inv_freq)
            _attend(_addr(self.d_q), _addr(self.self_k[l]), _addr(self.self_v[l]), _addr(self.d_a), att, att, H, hd,
                    self_keys, _addr(self.scores), scale)
            self._linear(pre + ".self_attn.o_proj", _addr(self.d_a), 1, att, D, _addr(self.d_h))
            _add_into(_addr(self.d_x), _addr(self.d_h), D)
            _norm_rows(_addr(self.d_x), _addr(self.d_h), 1, D, self._w(pre + ".post_attention_layernorm.weight"), 0.0)
            self._linear(pre + ".encoder_attn.q_proj", _addr(self.d_h), 1, D, att, _addr(self.d_q))
            _attend(_addr(self.d_q), _addr(self.cross_k[l]), _addr(self.cross_v[l]), _addr(self.d_a), att, att, H, hd,
                    cross_keys, _addr(self.scores), scale)
            self._linear(pre + ".encoder_attn.o_proj", _addr(self.d_a), 1, att, D, _addr(self.d_h))
            _add_into(_addr(self.d_x), _addr(self.d_h), D)
            _norm_rows(_addr(self.d_x), _addr(self.d_h), 1, D, self._w(pre + ".final_layernorm.weight"), 0.0)
            self._linear(pre + ".mlp.fc1", _addr(self.d_h), 1, D, 2 * I, _addr(self.d_f))
            for i in range(I):
                self.d_g[i] = _silu(self.d_f[I + i]) * self.d_f[i]
            self._linear(pre + ".mlp.fc2", _addr(self.d_g), 1, I, D, _addr(self.d_h))
            _add_into(_addr(self.d_x), _addr(self.d_h), D)
        _norm_rows(_addr(self.d_x), _addr(self.d_h), 1, D, self._w("model.decoder.norm.weight"), 0.0)
        self._linear("proj_out", _addr(self.d_h), 1, D, c.vocab, _addr(self.logits))

    def argmax(self) -> Int:
        var best = 0
        var bv = self.logits[0]
        for i in range(1, len(self.logits)):
            if self.logits[i] > bv:
                bv = self.logits[i]
                best = i
        return best

    # ------------------------------------------------------------------ end to end

    def max_tokens(self, samples: Int) -> Int:
        """The model card's limit: 6.5 tokens per second, counting the start token."""
        return Int(Float64(samples) * TOKENS_PER_SECOND / Float64(MOONSHINE_RATE))

    def transcribe(mut self, audio16k: List[Float32]) raises -> MoonshineTranscript:
        """Greedy transcript of 16 kHz mono audio."""
        var r = MoonshineTranscript()
        r.audio_seconds = Float64(len(audio16k)) / Float64(MOONSHINE_RATE)
        var limit = max(self.max_tokens(len(audio16k)), 2)
        var t0 = perf_counter_ns()
        self.encode(audio16k)
        self.start_decoder(limit)
        var t1 = perf_counter_ns()
        r.ids.append(self.cfg.bos)
        for pos in range(limit - 1):
            self.step(r.ids[len(r.ids) - 1], pos)
            var nxt = self.argmax()
            r.ids.append(nxt)
            if nxt == self.cfg.eos:
                break
        var t2 = perf_counter_ns()
        r.encode_ms = Float64(t1 - t0) / 1e6
        r.decode_ms = Float64(t2 - t1) / 1e6
        r.tokens = len(r.ids) - 1
        r.text = self.tokenizer.decode(r.ids)
        return r^

    def transcribe_clip(mut self, clip: AudioClip) raises -> MoonshineTranscript:
        return self.transcribe(to_model_rate(clip))

    def transcribe_wav(mut self, path: String) raises -> MoonshineTranscript:
        var clip = read_wav(path)
        if not clip.valid:
            raise Error("moonshine: cannot read WAV " + path)
        return self.transcribe_clip(clip)

    # ------------------------------------------------------------------ GPU path

    def _key(mut self, name: String) -> Int:
        var found = self.gkeys.get(name)
        if found:
            return found.value()
        var key = len(self.gkeys)
        self.gkeys[name] = key
        return key

    def _slot(mut self, name: String, n: Int) raises -> FP:
        """A named pooled device buffer of at least `n` floats (grown when too small)."""
        ref g = self.gpu.value().mm
        var found = self.gslots.get(name)
        if found:
            var sl = found.value()
            if g.pool_cap[sl] >= n:
                return g.ptr(sl)
            g.release(sl)
        var sl = g.alloc(n)
        self.gslots[name] = sl
        return g.ptr(sl)

    def _dvec(mut self, name: String, n: Int, offset: Float32 = 0) raises -> FP:
        """A resident [n] vector (plus `offset`: the encoder norms' unit-offset gamma)."""
        var key = self._key(name)
        var addr = self._w(name)
        if offset != 0:
            var found = self.gpu.value().mm.weights.get(key * 2)
            if found:
                return self.gpu.value().mm.wptr(found.value())
            var tmp = List[Float32](length=n, fill=0.0)
            var p = fptr_at(addr)
            for i in range(n):
                tmp[i] = p.unsafe_offset(i).unsafe_load[alignment=1]() + offset
            var w = self.gpu.value().mm.weight(key * 2, Int(tmp.unsafe_ptr()), 1, n, False)
            _ = len(tmp)  # keep the list alive through the upload (ASAP destruction)
            return self.gpu.value().mm.wptr(w)
        return self.gpu.value().mm.wptr(self.gpu.value().mm.weight(key * 2, addr, 1, n, False))

    def _dlinear(mut self, name: String, x: FP, m: Int, k: Int, n: Int, dst: FP) raises:
        var w = self.weights.weight(name + ".weight")
        var b = 0
        if self.weights.has(name + ".bias"):
            b = self.weights.weight(name + ".bias").fptr()
        var key = self._key(name)
        var widx = self.gpu.value().mm.weight(key * 2, w.fptr(), n, k, True)
        self.gpu.value().mm.matmul_dev(x, widx, dst, b, key * 2 + 1, m, k, n)

    def _encode_gpu(mut self, audio: List[Float32]) raises:
        var c = self.cfg.copy()
        var D = c.enc_hidden
        var n_valid = len(audio)
        var F = max((n_valid + MOONSHINE_FRAME - 1) // MOONSHINE_FRAME, 1)
        var padded = audio.copy()
        padded.resize(F * MOONSHINE_FRAME, 0.0)
        var feats = moonshine_frames(padded, F, exp(self.log_k))
        var nf0 = n_valid // MOONSHINE_FRAME
        var T1 = (F - 1) // 2 + 1
        var T2 = (T1 - 1) // 2 + 1
        var nf1 = 0
        for t in range(T1):
            if 2 * t - 4 < nf0:
                nf1 = t + 1
        var nf2 = 0
        for t in range(T2):
            if 2 * t - 4 < nf1:
                nf2 = t + 1
        var T = T2
        var att = c.enc_heads * c.enc_head_dim
        var pfeat = self._slot("feat", F * MOONSHINE_FRAME)
        var px = self._slot("x", F * 2 * D)
        var ph = self._slot("h", F * 2 * D)
        var pcol = self._slot("col", T1 * 2 * D * 5)
        var pq = self._slot("q", T * att)
        var pk = self._slot("k", T * att)
        var pv = self._slot("v", T * att)
        var pa = self._slot("a", T * max(att, D))
        var pf = self._slot("f", T * c.enc_ffn)
        self.gpu.value().mm.upload(self.gslots["feat"], Int(feats.unsafe_ptr()), F * MOONSHINE_FRAME)
        _ = len(feats)
        self._dlinear("model.encoder.embedder.linear", pfeat, F, MOONSHINE_FRAME, D, px)
        self.gpu.value().act(px, F, D, nf0, ACT_SILU)
        self.gpu.value().im2col(px, pcol, F, D, T1)
        self._dlinear("model.encoder.embedder.conv1", pcol, T1, D * 5, 2 * D, ph)
        self.gpu.value().act(ph, T1, 2 * D, nf1, ACT_SILU)
        self.gpu.value().im2col(ph, pcol, T1, 2 * D, T)
        self._dlinear("model.encoder.embedder.conv2", pcol, T, 2 * D * 5, D, px)
        self.gpu.value().act(px, T, D, nf2, ACT_MASK)
        var scale = Float32(1.0 / sqrt(Float64(c.enc_head_dim)))
        for l in range(c.enc_layers):
            var pre = "model.encoder.layers." + String(l)
            var g1 = self._dvec(pre + ".input_layernorm.gamma", D, 1.0)
            self.gpu.value().mm.layer_norm_dev(px, g1, g1, ph, T, D, 1e-5, False)
            self._dlinear(pre + ".self_attn.q_proj", ph, T, D, att, pq)
            self._dlinear(pre + ".self_attn.k_proj", ph, T, D, att, pk)
            self._dlinear(pre + ".self_attn.v_proj", ph, T, D, att, pv)
            self.gpu.value().attention(pq, pk, pv, pa, T, c.enc_heads, c.enc_head_dim, att, att, 0,
                                       c.win_left[l], c.win_right[l], nf2, scale)
            self._dlinear(pre + ".self_attn.o_proj", pa, T, att, D, ph)
            self.gpu.value().add(px, ph, T * D)
            var g2 = self._dvec(pre + ".post_attention_layernorm.gamma", D, 1.0)
            self.gpu.value().mm.layer_norm_dev(px, g2, g2, ph, T, D, 1e-5, False)
            self._dlinear(pre + ".mlp.fc1", ph, T, D, c.enc_ffn, pf)
            self.gpu.value().act(pf, T, c.enc_ffn, T, ACT_GELU)
            self._dlinear(pre + ".mlp.fc2", pf, T, c.enc_ffn, D, ph)
            self.gpu.value().add(px, ph, T * D)
        var gf = self._dvec("model.encoder.final_norm.gamma", D, 1.0)
        var penc = self._slot("enc", T * D)
        self.gpu.value().mm.layer_norm_dev(px, gf, gf, penc, T, D, 1e-5, False)
        self.enc = List[Float32](length=T * D, fill=0.0)
        self.gpu.value().mm.download(self.gslots["enc"], _addr(self.enc), T * D)
        self.enc_frames = T
        self.enc_valid = nf2

    def _start_decoder_gpu(mut self, max_steps: Int) raises:
        """Runs on the device encoding `_encode_gpu` left in the "enc" slot."""
        var c = self.cfg.copy()
        var T = self.enc_frames
        var De = c.enc_hidden
        var D = c.dec_hidden
        var att = c.dec_heads * c.dec_head_dim
        if T > c.max_pos:
            raise Error("moonshine: " + String(T) + " encoder frames exceed " + String(c.max_pos) + " positions")
        var penc = self._slot("enc", T * De)
        var pos_key = self._key("model.decoder.pos_emb.weight")
        var pw = self.gpu.value().mm.weight(pos_key * 2, self._w("model.decoder.pos_emb.weight"), c.max_pos, De, False)
        self.gpu.value().add(penc, self.gpu.value().mm.wptr(pw), T * De)
        var pmem = penc
        if self.weights.has("model.decoder.proj.weight"):
            pmem = self._slot("mem", T * D)
            self._dlinear("model.decoder.proj", penc, T, De, D, pmem)
        for l in range(c.dec_layers):
            var pre = "model.decoder.layers." + String(l) + ".encoder_attn"
            var ck = self._slot("ck" + String(l), T * att)
            var cv = self._slot("cv" + String(l), T * att)
            self._dlinear(pre + ".k_proj", pmem, T, D, att, ck)
            self._dlinear(pre + ".v_proj", pmem, T, D, att, cv)
            _ = self._slot("sk" + String(l), max_steps * att)
            _ = self._slot("sv" + String(l), max_steps * att)
        self.steps_cap = max_steps
        var invf_key = self._key("#rope_inv_freq")
        _ = self.gpu.value().mm.weight(invf_key * 2, Int(self.inv_freq.unsafe_ptr()), 1, len(self.inv_freq), False)
        _ = self._slot("dx", D)
        _ = self._slot("dh", D)
        _ = self._slot("dq", att)
        _ = self._slot("da", att)
        _ = self._slot("df", 2 * c.dec_ffn)
        _ = self._slot("dg", c.dec_ffn)
        _ = self._slot("logits", c.vocab)

    def _step_gpu(mut self, token: Int, pos: Int) raises:
        var c = self.cfg.copy()
        var D = c.dec_hidden
        var H = c.dec_heads
        var hd = c.dec_head_dim
        var att = H * hd
        var I = c.dec_ffn
        if pos >= self.steps_cap:
            raise Error("moonshine: decoder step " + String(pos) + " is past the cache")
        var px = self._slot("dx", D)
        var ph = self._slot("dh", D)
        var pq = self._slot("dq", att)
        var pa = self._slot("da", att)
        var pf = self._slot("df", 2 * I)
        var pg = self._slot("dg", I)
        var plog = self._slot("logits", c.vocab)
        var ekey = self._key("model.decoder.embed_tokens.weight")
        var ew = self.gpu.value().mm.weight(ekey * 2, self._w("model.decoder.embed_tokens.weight"), c.vocab, D, False)
        self.gpu.value().embed(self.gpu.value().mm.wptr(ew), px, token, D)
        var invf = self.gpu.value().mm.wptr(self.gpu.value().mm.weights[self._key("#rope_inv_freq") * 2])
        var scale = Float32(1.0 / sqrt(Float64(hd)))
        for l in range(c.dec_layers):
            var pre = "model.decoder.layers." + String(l)
            var g1 = self._dvec(pre + ".input_layernorm.weight", D)
            self.gpu.value().mm.layer_norm_dev(px, g1, g1, ph, 1, D, 1e-5, False)
            self._dlinear(pre + ".self_attn.q_proj", ph, 1, D, att, pq)
            var sk = self.gpu.value().mm.ptr(self.gslots["sk" + String(l)])
            var sv = self.gpu.value().mm.ptr(self.gslots["sv" + String(l)])
            self._dlinear(pre + ".self_attn.k_proj", ph, 1, D, att, sk.unsafe_offset(pos * att))
            self._dlinear(pre + ".self_attn.v_proj", ph, 1, D, att, sv.unsafe_offset(pos * att))
            self.gpu.value().rope(pq, H, hd, c.rot_dim, pos, invf)
            self.gpu.value().rope(sk.unsafe_offset(pos * att), H, hd, c.rot_dim, pos, invf)
            self.gpu.value().attention(pq, sk, sv, pa, 1, H, hd, att, att, pos, 0, 1, pos + 1, scale)
            self._dlinear(pre + ".self_attn.o_proj", pa, 1, att, D, ph)
            self.gpu.value().add(px, ph, D)
            var g2 = self._dvec(pre + ".post_attention_layernorm.weight", D)
            self.gpu.value().mm.layer_norm_dev(px, g2, g2, ph, 1, D, 1e-5, False)
            self._dlinear(pre + ".encoder_attn.q_proj", ph, 1, D, att, pq)
            var ck = self.gpu.value().mm.ptr(self.gslots["ck" + String(l)])
            var cv = self.gpu.value().mm.ptr(self.gslots["cv" + String(l)])
            self.gpu.value().attention(pq, ck, cv, pa, 1, H, hd, att, att, 0, 0, -1, self.enc_valid, scale)
            self._dlinear(pre + ".encoder_attn.o_proj", pa, 1, att, D, ph)
            self.gpu.value().add(px, ph, D)
            var g3 = self._dvec(pre + ".final_layernorm.weight", D)
            self.gpu.value().mm.layer_norm_dev(px, g3, g3, ph, 1, D, 1e-5, False)
            self._dlinear(pre + ".mlp.fc1", ph, 1, D, 2 * I, pf)
            self.gpu.value().gate(pf, pg, I)
            self._dlinear(pre + ".mlp.fc2", pg, 1, I, D, ph)
            self.gpu.value().add(px, ph, D)
        var gn = self._dvec("model.decoder.norm.weight", D)
        self.gpu.value().mm.layer_norm_dev(px, gn, gn, ph, 1, D, 1e-5, False)
        self._dlinear("proj_out", ph, 1, D, c.vocab, plog)
        self.gpu.value().mm.download(self.gslots["logits"], _addr(self.logits), c.vocab)


# ---------------------------------------------------------------------- voice commands

def _number_word(w: String) -> String:
    var names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                 "eleven", "twelve"]
    for i in range(len(names)):
        if w == String(i):
            return names[i]
    return w


def command_words(text: String) -> List[String]:
    """Lower-case words of `text` without punctuation; digits 0-12 spelled out ("5" -> "five"),
    so "move it 5 meters" and "Move it five meters." compare equal."""
    var words = List[String]()
    var cur = List[UInt8]()
    var b = text.as_bytes()
    for i in range(len(b) + 1):
        var c: UInt8 = 32 if i == len(b) else b[i]
        var keep = (c >= 48 and c <= 57) or (c >= 97 and c <= 122) or c == 39 or c >= 128
        if c >= 65 and c <= 90:
            c += 32
            keep = True
        if keep:
            cur.append(c)
        elif len(cur) > 0:
            words.append(_number_word(String(unsafe_from_utf8=cur)))
            cur = List[UInt8]()
    return words^


struct VoiceCommands(Movable):
    """A fixed phrase list matched against transcripts by word overlap (F1 over word multisets),
    which tolerates the punctuation, casing and small slips of a short command."""
    var phrases: List[String]
    var words: List[List[String]]

    def __init__(out self):
        self.phrases = List[String]()
        self.words = List[List[String]]()

    def add(mut self, phrase: String) -> Int:
        self.phrases.append(phrase)
        self.words.append(command_words(phrase))
        return len(self.phrases) - 1

    def score(self, k: Int, heard: List[String]) -> Float64:
        var want = self.words[k].copy()
        if len(want) == 0 or len(heard) == 0:
            return 0
        var used = List[Bool](length=len(want), fill=False)
        var hits = 0
        for w in heard:
            for j in range(len(want)):
                if not used[j] and want[j] == w:
                    used[j] = True
                    hits += 1
                    break
        if hits == 0:
            return 0
        var p = Float64(hits) / Float64(len(heard))
        var r = Float64(hits) / Float64(len(want))
        return 2 * p * r / (p + r)

    def match(self, text: String, threshold: Float64 = 0.6) -> Int:
        """The best phrase's index, or -1 when none scores `threshold` or more."""
        var heard = command_words(text)
        var best = -1
        var bs = threshold
        for k in range(len(self.phrases)):
            var s = self.score(k, heard)
            if s >= bs:
                bs = s
                best = k
        return best
