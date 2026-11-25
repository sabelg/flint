"""Magenta RealTime 2 streaming music generation in pure Mojo (`flint.ml.magenta_rt`).

Magenta RealTime 2 (Google DeepMind, https://github.com/magenta/magenta-realtime; code
Apache-2.0, weights CC-BY 4.0) generates 48 kHz stereo music frame by frame (25 Hz, 40 ms),
steered by a 12-token MusicCoCa style (text or audio, blendable), an optional 128-pitch MIDI
piano roll, drums and classifier-free-guidance strengths. This module runs the 230M-parameter
`mrt2_small` checkpoint straight from `checkpoints/mrt2_small.safetensors`:

  - Encoder: per frame, the conditioning row (12 style + 128 notes + 1 drum + 3 CFG tokens) is
    embedded (style: summed RVQ dequantiser rows -> 768 -> 256; the rest: mean of per-channel
    embeddings), the two branches averaged and layer-normalised (256).
  - Temporal decoder: 12 pre/post-RMSNorm blocks (1024 wide, 8 x 128 heads, no positional
    encoding) with sliding-window self-attention over the last 41 frames plus a learned
    attention sink, streaming cross-attention over the last 41 encoder frames (also with a sink)
    and a 4096 GELU MLP. Its input is the mean of the previous frame's 12 token embeddings.
  - Depth decoder: 2 blocks (768 wide, 6 heads) that emit the frame's 12 SpectroStream RVQ codes
    one level at a time; each level is sampled with soft-capped logits, temperature, top-k and
    Gumbel noise from a bit-exact `jax.random` threefry stream (`flint.ml.jax_random`), so
    token streams match the JAX reference for the same key.
  - The SpectroStream decoder (`flint.ml.audio_codec`) turns each frame's codes into 1920 stereo
    samples.

    var music = MagentaRtStream.load("/path/to/magenta-realtime-2", Q_INT8)  # tools/magenta_rt/fetch.sh
    _ = music.mrt.enable_gpu_codec()              # optional: SpectroStream on the MAX GPU
    music.set_prompt("upbeat synthwave with punchy drums")
    var ring = PcmStreamRing.create(48000)
    server.add_pcm_stream(ring, BUS_MUSIC)       # flint.audio.server.AudioServer
    ...
    _ = music.pump(ring)                          # every tick / on a generator thread
    music.set_prompt("dark ambient drone", 4.0)   # 4 s style morph on a game event

Lower level: `MagentaRT.set_style(tokens)` (12 MusicCoCa codes from
`flint.ml.musiccoca.MusicCoCaText.style_tokens`), `set_notes`, `set_drums`, `set_guidance`,
`set_sampling`, `next_frame(addr)` / `generate(frames)` (interleaved stereo float32, 48 kHz).

Weights are kept as `flint.ml.llm.quant.QMat` (fp32 by default for parity, int8 for speed), the
same matrices and job-pool kernels as `flint.ml.llm`. Measured on a GB10 (DGX Spark, 20 cores,
shared machine): LM 17 ms (fp32) / 8.6 ms (int8) per 40 ms frame on the CPU; SpectroStream
22 ms on the CPU, 4.4 ms on the GPU; int8 LM + GPU codec runs at RTF 0.33.
"""

from std.math import sqrt, exp, log, tanh, sin, cos
from std.ffi import external_call
from std.time import perf_counter_ns
from flint.core.jobs import JobPool
from flint.ml.llm.safetensors import SafeTensors, SafeTensor, widen_to_f32
from flint.ml.llm.quant import QMat, Q_F32, Q_INT8, Q_INT4, qmatmul
from flint.ml.jax_random import JaxKey, prng_key, fold_in, gumbel
from flint.ml.audio_codec import FBuf, fp, copy_floats, SpectroStreamDecoder, SS_FRAME_SAMPLES
from flint.ml.audio_codec_gpu import SpectroStreamGpu
from flint.ml.musiccoca import MusicCoCaText, blend_embeddings
from flint.audio.pcm_stream import PcmStreamRing

comptime MRT_SAMPLE_RATE: Int = 48000
comptime MRT_FRAME_RATE: Int = 25
comptime MRT_LEVELS: Int = 12
comptime MRT_CODEBOOK: Int = 1024
comptime MRT_RESERVED: Int = 6
comptime MRT_VOCAB: Int = 12294
comptime MRT_COND: Int = 144          # 12 style + 128 notes + 1 drums + 2 CFG + 1 drum CFG
comptime MRT_STYLE_TOKENS: Int = 12
comptime MRT_COND_OFFSET: Int = 7     # 6 reserved tokens + the dropout token
comptime NEG_BIG: Float32 = -2.3819763e38   # -0.7 * float32 max, the reference's mask value

# Indices of the conditioning row
comptime COND_NOTES: Int = 12
comptime COND_DRUMS: Int = 140
comptime COND_CFG: Int = 141


def cpu_count() -> Int:
    return Int(external_call["sysconf", Int64](Int32(84)))


def _alloc(n: Int) -> Int:
    return external_call["calloc", Int](max(n, 1), 4)


def _free(p: Int):
    if p != 0:
        external_call["free", NoneType](p)


@always_inline
def gelu_tanh(x: Float32) -> Float32:
    """jax.nn.gelu (approximate=True)."""
    var c: Float32 = 0.7978845608028654
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)))


@always_inline
def softplus(x: Float32) -> Float32:
    return max(x, 0) + log(1 + exp(-abs(x)))


def rms_norm(x: Int, g: Int, o: Int, n: Int):
    var px = fp(x)
    var ms: Float32 = 0
    for i in range(n):
        var v = px.unsafe_offset(i).unsafe_load()
        ms += v * v
    ms /= Float32(n)
    var r = 1.0 / sqrt(ms + 1e-6)
    for i in range(n):
        fp(o + i * 4).unsafe_store(px.unsafe_offset(i).unsafe_load() * r * fp(g + i * 4).unsafe_load())


def layer_norm(x: Int, g: Int, b: Int, o: Int, n: Int):
    var px = fp(x)
    var mean: Float32 = 0
    for i in range(n):
        mean += px.unsafe_offset(i).unsafe_load()
    mean /= Float32(n)
    var var_: Float32 = 0
    for i in range(n):
        var d = px.unsafe_offset(i).unsafe_load() - mean
        var_ += d * d
    var_ /= Float32(n)
    var r = 1.0 / sqrt(var_ + 1e-6)
    for i in range(n):
        var v = (px.unsafe_offset(i).unsafe_load() - mean) * r
        fp(o + i * 4).unsafe_store(v * fp(g + i * 4).unsafe_load() + fp(b + i * 4).unsafe_load())


# ------------------------------------------------------------------------------- weights

def _transposed(st: SafeTensors, name: String, k: Int, n: Int) raises -> Int:
    """A malloc'd [n, k] float32 copy of a JAX [k, ...] kernel (flattened to [k, n])."""
    var t = st.get(name)
    if t.numel() != k * n:
        raise Error("magenta_rt: unexpected size for " + name)
    var src = _alloc(k * n)
    widen_to_f32(t, src, 0, k * n)
    var dst = _alloc(k * n)
    for i in range(k):
        for j in range(n):
            fp(dst + (j * k + i) * 4).unsafe_store(fp(src + (i * n + j) * 4).unsafe_load())
    _free(src)
    return dst


struct WeightLoader(Movable):
    var st: SafeTensors
    var pool: JobPool
    var fmt: Int
    var tmp: List[Int]

    def __init__(out self, var st: SafeTensors, fmt: Int):
        self.st = st^
        self.pool = JobPool()
        _ = self.pool.start(max(1, cpu_count() - 2))
        self.fmt = fmt
        self.tmp = List[Int]()

    def vec(mut self, name: String) raises -> Int:
        return self.st.to_f32(name)

    def mat(mut self, var names: List[String], k: Int, var ns: List[Int], fmt: Int = -1) raises -> QMat:
        """Stacks JAX kernels (each [k, n_i]) as rows of one [sum n_i, k] QMat."""
        var srcs = List[SafeTensor]()
        var bufs = List[Int]()
        for i in range(len(names)):
            var p = _transposed(self.st, names[i], k, ns[i])
            bufs.append(p)
            srcs.append(SafeTensor(names[i], "F32", [ns[i], k], p))
        var m = QMat.build(self.pool, srcs^, self.fmt if fmt < 0 else fmt)
        for p in bufs:
            _free(p)
        return m^

    def mat_rows(mut self, name: String) raises -> QMat:
        """A kernel already in [out, in...] layout (attention output projections)."""
        var srcs = List[SafeTensor]()
        var t = self.st.get(name)
        var n = t.shape[0]
        var k = t.numel() // n
        srcs.append(SafeTensor(name, t.dtype, [n, k], t.addr))
        return QMat.build(self.pool, srcs^, self.fmt)


def qmat_rows_view(m: QMat, row0: Int, rows: Int) -> QMat:
    """Rows [row0, row0 + rows) of `m` as a non-owning QMat (fp32 / int8 layouts)."""
    var v = QMat()
    v.n = rows
    v.k = m.k
    v.fmt = m.fmt
    if m.fmt == Q_F32:
        v.data = m.data + row0 * m.k * 4
    elif m.fmt == Q_INT8:
        v.data = m.data + row0 * m.k
        v.scales = m.scales + row0 * (m.k // 128) * 4
    else:
        v.data = m.data + row0 * (m.k // 2)
        v.scales = m.scales + row0 * (m.k // 32) * 4
    return v^


struct AttnWeights(Movable):
    var qkv: QMat          # self: q, k, v stacked (3 * H * D rows)
    var q: QMat            # cross: q only
    var kv: QMat           # cross: k, v from the source
    var o: QMat
    var qscale: Int        # [D] per-dim query scale (1.442695 / sqrt(D) * softplus(p))
    var sink_k: Int        # [H * D] or 0
    var sink_v: Int
    var pre: Int
    var post: Int

    def __init__(out self):
        self.qkv = QMat()
        self.q = QMat()
        self.kv = QMat()
        self.o = QMat()
        self.qscale = 0
        self.sink_k = 0
        self.sink_v = 0
        self.pre = 0
        self.post = 0


struct Block(Movable):
    var sa: AttnWeights
    var has_cross: Bool
    var ca: AttnWeights
    var w1: QMat
    var b1: Int
    var w2: QMat
    var b2: Int
    var f_pre: Int
    var f_post: Int

    def __init__(out self):
        self.sa = AttnWeights()
        self.has_cross = False
        self.ca = AttnWeights()
        self.w1 = QMat()
        self.b1 = 0
        self.w2 = QMat()
        self.b2 = 0
        self.f_pre = 0
        self.f_post = 0


def _qscale(mut ld: WeightLoader, name: String, d: Int) raises -> Int:
    var p = ld.vec(name)
    var r_softplus_0: Float32 = 1.442695041
    var base = r_softplus_0 * Float32(1.0 / sqrt(Float64(d)))
    for i in range(d):
        fp(p + i * 4).unsafe_store(base * softplus(fp(p + i * 4).unsafe_load()))
    return p


def load_attention(mut ld: WeightLoader, p: String, d_model: Int, d_src: Int, heads: Int, hd: Int,
                   cross: Bool, sinks: Bool) raises -> AttnWeights:
    var a = AttnWeights()
    var hw = heads * hd
    if cross:
        a.q = ld.mat([p + "attention/query_projection/kernel"], d_model, [hw])
        a.kv = ld.mat([p + "attention/key_projection/kernel", p + "attention/value_projection/kernel"],
                      d_src, [hw, hw])
    else:
        a.qkv = ld.mat([p + "attention/query_projection/kernel", p + "attention/key_projection/kernel",
                        p + "attention/value_projection/kernel"], d_model, [hw, hw, hw])
    a.o = ld.mat_rows(p + "output_projection/kernel")
    a.qscale = _qscale(ld, p + "attention/per_dim_scale", hd)
    if sinks:
        a.sink_k = ld.vec(p + "attention/sink_key_embeddings")
        a.sink_v = ld.vec(p + "attention/sink_value_embeddings")
    a.pre = ld.vec(p + "pre_norm/scale")
    a.post = ld.vec(p + "post_norm/scale")
    return a^


def load_block(mut ld: WeightLoader, p: String, d: Int, d_src: Int, heads: Int, hd: Int, ffn: Int,
               cross: Bool, sinks: Bool) raises -> Block:
    var b = Block()
    b.sa = load_attention(ld, p + "self_attention/", d, d, heads, hd, False, sinks)
    b.has_cross = cross
    if cross:
        b.ca = load_attention(ld, p + "cross_attention/", d, d_src, heads, hd, True, sinks)
    b.w1 = ld.mat([p + "ffn/ffn_layer1/kernel"], d, [ffn])
    b.b1 = ld.vec(p + "ffn/ffn_layer1/bias")
    b.w2 = ld.mat([p + "ffn/ffn_layer2/kernel"], ffn, [d])
    b.b2 = ld.vec(p + "ffn/ffn_layer2/bias")
    b.f_pre = ld.vec(p + "ffn/pre_norm/scale")
    b.f_post = ld.vec(p + "ffn/post_norm/scale")
    return b^


# ---------------------------------------------------------------------------- KV windows

struct KvWindow(Movable):
    """The last `cap` keys / values ([H * D] each) of one attention layer, as a ring."""
    var cap: Int
    var width: Int
    var keys: FBuf
    var vals: FBuf
    var count: Int
    var head: Int    # next write slot

    def __init__(out self, cap: Int, width: Int):
        self.cap = cap
        self.width = width
        self.keys = FBuf(cap * width)
        self.vals = FBuf(cap * width)
        self.count = 0
        self.head = 0

    def reset(mut self):
        self.count = 0
        self.head = 0

    def push(mut self, k: Int, v: Int):
        copy_floats(self.keys.addr + self.head * self.width * 4, k, self.width)
        copy_floats(self.vals.addr + self.head * self.width * 4, v, self.width)
        self.head = (self.head + 1) % self.cap
        self.count = min(self.count + 1, self.cap)


def attend(q: Int, k_cur: Int, v_cur: Int, win: KvWindow, qscale: Int, sink_k: Int, sink_v: Int,
           heads: Int, hd: Int, dst: Int, scores: Int):
    """Single-query attention over the window plus the current key (and an optional sink)."""
    var n = win.count
    for h in range(heads):
        var qh = q + h * hd * 4
        var s = fp(scores)
        # scaled query
        var mx: Float32 = -3.4e38
        var idx = 0
        if sink_k != 0:
            var acc: Float32 = 0
            for d in range(hd):
                acc += fp(qh + d * 4).unsafe_load() * fp(sink_k + (h * hd + d) * 4).unsafe_load()
            s.unsafe_offset(0).unsafe_store(acc)
            mx = acc
            idx = 1
        for j in range(n + 1):
            var kj = (win.keys.addr + (j * win.width + h * hd) * 4) if j < n else (k_cur + h * hd * 4)
            var acc: Float32 = 0
            for d in range(hd):
                acc += fp(qh + d * 4).unsafe_load() * fp(qscale + d * 4).unsafe_load() * fp(kj + d * 4).unsafe_load()
            s.unsafe_offset(idx + j).unsafe_store(acc)
            mx = max(mx, acc)
        var tot = idx + n + 1
        var den: Float32 = 0
        for j in range(tot):
            var e = exp(s.unsafe_offset(j).unsafe_load() - mx)
            s.unsafe_offset(j).unsafe_store(e)
            den += e
        var inv = 1.0 / den
        var oh = dst + h * hd * 4
        for d in range(hd):
            fp(oh + d * 4).unsafe_store(0)
        if sink_k != 0:
            var p0 = s.unsafe_offset(0).unsafe_load() * inv
            for d in range(hd):
                fp(oh + d * 4).unsafe_store(p0 * fp(sink_v + (h * hd + d) * 4).unsafe_load())
        for j in range(n + 1):
            var vj = (win.vals.addr + (j * win.width + h * hd) * 4) if j < n else (v_cur + h * hd * 4)
            var pj = s.unsafe_offset(idx + j).unsafe_load() * inv
            for d in range(hd):
                var od = fp(oh + d * 4)
                od.unsafe_store(od.unsafe_load() + pj * fp(vj + d * 4).unsafe_load())


# ---------------------------------------------------------------------------- the model

struct StackDims(Copyable, Movable, ImplicitlyCopyable):
    var d: Int
    var heads: Int
    var hd: Int
    var ffn: Int
    var window: Int

    def __init__(out self, d: Int, heads: Int, hd: Int, ffn: Int, window: Int):
        self.d = d
        self.heads = heads
        self.hd = hd
        self.ffn = ffn
        self.window = window


struct Scratch(Movable):
    var h: FBuf
    var qkv: FBuf
    var att: FBuf
    var o: FBuf
    var f: FBuf
    var sc: FBuf
    var kv: FBuf

    def __init__(out self, d: Int, hw: Int, ffn: Int, window: Int):
        self.h = FBuf(d)
        self.qkv = FBuf(3 * hw)
        self.att = FBuf(hw)
        self.o = FBuf(d)
        self.f = FBuf(ffn)
        self.sc = FBuf(window + 4)
        self.kv = FBuf(2 * hw)


def block_step(mut pool: JobPool, b: Block, dims: StackDims, x: Int, mut self_win: KvWindow,
               mut cross_win: KvWindow, cross_kv: Int, mut s: Scratch):
    """x[d] += one transformer block for a single new timestep."""
    var d = dims.d
    var hw = dims.heads * dims.hd
    # self-attention
    rms_norm(x, b.sa.pre, s.h.addr, d)
    qmatmul(pool, b.sa.qkv, s.h.addr, s.qkv.addr, 1, False)
    var q = s.qkv.addr
    var k = s.qkv.addr + hw * 4
    var v = s.qkv.addr + 2 * hw * 4
    attend(q, k, v, self_win, b.sa.qscale, b.sa.sink_k, b.sa.sink_v, dims.heads, dims.hd, s.att.addr, s.sc.addr)
    self_win.push(k, v)
    qmatmul(pool, b.sa.o, s.att.addr, s.o.addr, 1, False)
    rms_norm(s.o.addr, b.sa.post, s.o.addr, d)
    for i in range(d):
        fp(x + i * 4).unsafe_store(fp(x + i * 4).unsafe_load() + s.o.at(i))
    # streaming cross-attention (keys / values of the current source frame already projected)
    if b.has_cross:
        rms_norm(x, b.ca.pre, s.h.addr, d)
        qmatmul(pool, b.ca.q, s.h.addr, s.qkv.addr, 1, False)
        var ck = cross_kv
        var cv = cross_kv + hw * 4
        attend(s.qkv.addr, ck, cv, cross_win, b.ca.qscale, b.ca.sink_k, b.ca.sink_v, dims.heads, dims.hd,
               s.att.addr, s.sc.addr)
        cross_win.push(ck, cv)
        qmatmul(pool, b.ca.o, s.att.addr, s.o.addr, 1, False)
        rms_norm(s.o.addr, b.ca.post, s.o.addr, d)
        for i in range(d):
            fp(x + i * 4).unsafe_store(fp(x + i * 4).unsafe_load() + s.o.at(i))
    # MLP
    rms_norm(x, b.f_pre, s.h.addr, d)
    qmatmul(pool, b.w1, s.h.addr, s.f.addr, 1, False)
    for i in range(dims.ffn):
        s.f.set(i, gelu_tanh(s.f.at(i) + fp(b.b1 + i * 4).unsafe_load()))
    qmatmul(pool, b.w2, s.f.addr, s.o.addr, 1, False)
    for i in range(d):
        s.o.set(i, s.o.at(i) + fp(b.b2 + i * 4).unsafe_load())
    rms_norm(s.o.addr, b.f_post, s.o.addr, d)
    for i in range(d):
        fp(x + i * 4).unsafe_store(fp(x + i * 4).unsafe_load() + s.o.at(i))


struct DepthformerLM(Movable):
    """The mrt2 encoder + temporal + depth decoders with their streaming state."""
    var tdims: StackDims
    var ddims: StackDims
    var enc_d: Int
    # encoder
    var e_mulan: Int        # [12 * 1031, 768]
    var e_mulan_proj: QMat  # 768 -> 256
    var e_reg: Int          # [1536, 256]
    var e_reg_off: List[Int]
    var e_ln_g: Int
    var e_ln_b: Int
    # decoder
    var emb: Int            # [12294, 1024]
    var temporal: List[Block]
    var adapter: QMat       # 1024 -> 768
    var depth: List[Block]
    var final_g: Int
    var final_b: Int
    var logits_w: QMat      # [12294, 768]
    var logits_b: Int
    # state
    var t_self: List[KvWindow]
    var t_cross: List[KvWindow]
    var d_self: List[KvWindow]
    var prev: List[Int]     # previous frame's unique tokens
    var frame: Int
    var key: JaxKey
    var temperature: Float32
    var top_k: Int
    # scratch
    var ts: Scratch
    var ds: Scratch
    var x: FBuf
    var xd: FBuf
    var enc: FBuf
    var tmp: FBuf
    var cross_kv: FBuf
    var logits: FBuf
    var sortbuf: List[Float32]
    var no_cross: KvWindow
    var e_sum: FBuf
    var e_reg_sum: FBuf
    var pool: JobPool
    # debug taps
    var last_temporal_in: FBuf
    var last_temporal_out: FBuf

    def __init__(out self, var ld: WeightLoader) raises:
        self.tdims = StackDims(1024, 8, 128, 4096, 41)
        self.ddims = StackDims(768, 6, 128, 3072, 12)
        self.enc_d = 256
        var D = String("params/depthformer/")
        var E = D + "encoder/body/"
        self.e_mulan = ld.vec(E + "layers_1/branched_mulan_embedder/mulan_embedder/mulan_dequantizer/embedding")
        self.e_mulan_proj = ld.mat([E + "layers_1/branched_mulan_embedder/mulan_embedder/depth_input_adapter/kernel"],
                                   768, [256], Q_F32)
        self.e_reg = ld.vec(E + "layers_1/branched_regular_embedder/regular_embedder/embedding")
        self.e_reg_off = List[Int]()
        var acc = 0
        for c in range(132):
            self.e_reg_off.append(acc)
            var vs = 11 if c < 128 else (9 if c == 128 else (47 if c < 131 else 15))   # CFG channels have no dropout token
            acc += vs
        self.e_ln_g = ld.vec(E + "encoder_ln/scale")
        self.e_ln_b = ld.vec(E + "encoder_ln/bias")
        var P = D + "decoder/"
        self.emb = ld.vec(P + "decoder_embedding/embedding/embedding")
        self.temporal = List[Block]()
        for l in range(12):
            self.temporal.append(load_block(ld, P + "temporal_body/transformer/x_layers_" + String(l) + "/",
                                            1024, 256, 8, 128, 4096, True, True))
        self.adapter = ld.mat([P + "depth_body/depth_input_adapter/kernel"], 1024, [768])
        self.depth = List[Block]()
        for l in range(2):
            self.depth.append(load_block(ld, P + "depth_body/transformer/x_layers_" + String(l) + "/",
                                         768, 768, 6, 128, 3072, False, False))
        self.final_g = ld.vec(P + "depth_body/final_ln/scale")
        self.final_b = ld.vec(P + "depth_body/final_ln/bias")
        self.logits_w = ld.mat([P + "depth_body/to_logits/kernel"], 768, [MRT_VOCAB])
        self.logits_b = ld.vec(P + "depth_body/to_logits/bias")
        self.t_self = List[KvWindow]()
        self.t_cross = List[KvWindow]()
        for _ in range(12):
            self.t_self.append(KvWindow(41, 1024))
            self.t_cross.append(KvWindow(41, 1024))
        self.d_self = List[KvWindow]()
        for _ in range(2):
            self.d_self.append(KvWindow(12, 768))
        self.prev = List[Int](length=MRT_LEVELS, fill=0)
        self.frame = 0
        self.key = prng_key(0)
        self.temperature = 1.3
        self.top_k = 40
        self.ts = Scratch(1024, 1024, 4096, 41)
        self.ds = Scratch(768, 768, 3072, 12)
        self.x = FBuf(1024)
        self.xd = FBuf(768)
        self.enc = FBuf(256)
        self.tmp = FBuf(1024)
        self.cross_kv = FBuf(2048)
        self.logits = FBuf(MRT_CODEBOOK)
        self.sortbuf = List[Float32](length=MRT_CODEBOOK, fill=0.0)
        self.no_cross = KvWindow(1, 1)
        self.e_sum = FBuf(768)
        self.e_reg_sum = FBuf(256)
        self.pool = ld.pool.copy()
        self.last_temporal_in = FBuf(1024)
        self.last_temporal_out = FBuf(1024)

    def reset(mut self, key: JaxKey):
        for i in range(len(self.t_self)):
            self.t_self[i].reset()
            self.t_cross[i].reset()
        for i in range(len(self.d_self)):
            self.d_self[i].reset()
        for i in range(MRT_LEVELS):
            self.prev[i] = 0
        self.frame = 0
        self.key = key

    def encode(mut self, cond: List[Int]):
        """Conditioning row (already offset by +7) -> encoder output (256) in self.enc."""
        ref m = self.e_sum
        m.zero()
        for i in range(MRT_STYLE_TOKENS):
            var row = self.e_mulan + ((cond[i] + i * 1031) * 768) * 4
            for d in range(768):
                m.set(d, m.at(d) + fp(row + d * 4).unsafe_load())
        qmatmul(self.pool, self.e_mulan_proj, m.addr, self.tmp.addr, 1, False)
        ref reg = self.e_reg_sum
        reg.zero()
        for c in range(132):
            var row = self.e_reg + ((cond[MRT_STYLE_TOKENS + c] + self.e_reg_off[c]) * 256) * 4
            for d in range(256):
                reg.set(d, reg.at(d) + fp(row + d * 4).unsafe_load())
        for d in range(256):
            reg.set(d, (self.tmp.at(d) + reg.at(d) / 132.0) * 0.5)
        layer_norm(reg.addr, self.e_ln_g, self.e_ln_b, self.enc.addr, 256)

    def _embed_mean(self, tokens: List[Int], dst: Int):
        for d in range(1024):
            fp(dst + d * 4).unsafe_store(0)
        for t in tokens:
            var row = self.emb + (t * 1024) * 4
            for d in range(1024):
                var o = fp(dst + d * 4)
                o.unsafe_store(o.unsafe_load() + fp(row + d * 4).unsafe_load() * 32.0)
        var inv = 1.0 / Float32(len(tokens))
        for d in range(1024):
            fp(dst + d * 4).unsafe_store(fp(dst + d * 4).unsafe_load() * inv)

    def _sample(mut self, level: Int) -> Int:
        """Soft cap, top-k, Gumbel-max over this level's 1024 logits (in self.logits)."""
        var lo = MRT_RESERVED + level * MRT_CODEBOOK
        for j in range(MRT_CODEBOOK):
            var l = self.logits.at(j) + fp(self.logits_b + (lo + j) * 4).unsafe_load()
            l = tanh(l / 30.0) * 30.0
            self.logits.set(j, l)
            self.sortbuf[j] = l
        var kk = max(1, min(self.top_k, MRT_VOCAB))
        var kth: Float32 = NEG_BIG
        if kk <= MRT_CODEBOOK:
            sort(self.sortbuf)
            kth = self.sortbuf[MRT_CODEBOOK - kk]
        var best = -1
        var best_v: Float32 = -3.4e38
        for j in range(MRT_CODEBOOK):
            var l = self.logits.at(j)
            var v = (l if l >= kth else NEG_BIG) + gumbel(self.key, lo + j) * self.temperature
            if v > best_v:
                best_v = v
                best = j
        return lo + best

    def step(mut self, cond: List[Int]) -> List[Int]:
        """One 40 ms frame: conditioning row -> 12 codes (0..1023)."""
        self.encode(cond)
        self._embed_mean(self.prev, self.x.addr)
        copy_floats(self.last_temporal_in.addr, self.x.addr, 1024)
        for l in range(len(self.temporal)):
            qmatmul(self.pool, self.temporal[l].ca.kv, self.enc.addr, self.cross_kv.addr, 1, False)
            block_step(self.pool, self.temporal[l], self.tdims, self.x.addr, self.t_self[l], self.t_cross[l],
                       self.cross_kv.addr, self.ts)
        copy_floats(self.last_temporal_out.addr, self.x.addr, 1024)
        for i in range(len(self.d_self)):
            self.d_self[i].reset()
        var codes = List[Int]()
        var uniq = List[Int]()
        var inp = self.x.addr
        for level in range(MRT_LEVELS):
            qmatmul(self.pool, self.adapter, inp, self.xd.addr, 1, False)
            for l in range(len(self.depth)):
                block_step(self.pool, self.depth[l], self.ddims, self.xd.addr, self.d_self[l], self.no_cross,
                           0, self.ds)
            layer_norm(self.xd.addr, self.final_g, self.final_b, self.ds.h.addr, 768)
            var lo = MRT_RESERVED + level * MRT_CODEBOOK
            var view = qmat_rows_view(self.logits_w, lo, MRT_CODEBOOK)
            qmatmul(self.pool, view, self.ds.h.addr, self.logits.addr, 1, False)
            var tok = self._sample(level)
            self.key = fold_in(self.key, self.frame)
            uniq.append(tok)
            codes.append((tok - MRT_RESERVED) % MRT_CODEBOOK)
            # next depth input: the sampled token's (scaled) embedding
            var row = self.emb + (tok * 1024) * 4
            for d in range(1024):
                self.tmp.set(d, fp(row + d * 4).unsafe_load() * 32.0)
            inp = self.tmp.addr
        self.prev = uniq^
        self.frame += 1
        return codes^


# ------------------------------------------------------------------------- conditioning

def discretize_cfg(value: Float64, step: Float64, max_bin: Int) -> Int:
    var c = max(-1.0, min(7.0, value))
    var b = Int(round((c + 1.0) / step))
    return max(0, min(max_bin, b))


struct MrtConditioning(Copyable, Movable):
    """What steers generation: 12 style tokens (-1 = masked), the 128-pitch MIDI state
    (-1 masked, 0 off, 1 sustain, 2 onset, 3 model's choice), drums (-1 / 0 / 1) and the
    classifier-free-guidance scales for style, notes and drums (each -1..7)."""
    var style: List[Int]
    var notes: List[Int]
    var drums: Int
    var cfg_style: Float64
    var cfg_notes: Float64
    var cfg_drums: Float64

    def __init__(out self):
        self.style = List[Int](length=MRT_STYLE_TOKENS, fill=-1)
        self.notes = List[Int](length=128, fill=-1)
        self.drums = -1
        self.cfg_style = 3.0
        self.cfg_notes = 1.0
        self.cfg_drums = 1.0

    def row(self) -> List[Int]:
        var r = List[Int]()
        for v in self.style:
            r.append(v + MRT_COND_OFFSET)
        for v in self.notes:
            r.append(v + MRT_COND_OFFSET)
        r.append(self.drums + MRT_COND_OFFSET)
        r.append(discretize_cfg(self.cfg_style, 0.2, 40) + MRT_COND_OFFSET)
        r.append(discretize_cfg(self.cfg_notes, 0.2, 40) + MRT_COND_OFFSET)
        r.append(discretize_cfg(self.cfg_drums, 1.0, 8) + MRT_COND_OFFSET)
        return r^


# ----------------------------------------------------------------------------- the system

struct MagentaRT(Movable):
    """Streaming music generator: conditioning in, 40 ms frames of 48 kHz stereo out."""
    var lm: DepthformerLM
    var codec: SpectroStreamDecoder
    var cond: MrtConditioning
    var codec_pool: JobPool
    var gpu: List[SpectroStreamGpu]   # one entry once `enable_gpu_codec` ran
    var lm_ns: Int
    var codec_ns: Int
    var frames: Int
    var emb: FBuf

    def __init__(out self, var lm: DepthformerLM, var codec: SpectroStreamDecoder):
        self.lm = lm^
        self.codec = codec^
        self.cond = MrtConditioning()
        self.codec_pool = self.lm.pool.copy()
        self.gpu = List[SpectroStreamGpu]()
        self.lm_ns = 0
        self.codec_ns = 0
        self.frames = 0
        self.emb = FBuf(256)

    def enable_gpu_codec(mut self) raises -> String:
        """Runs the SpectroStream decoder on the MAX GPU from now on; returns the device name."""
        if len(self.gpu) == 0:
            self.gpu.append(SpectroStreamGpu(self.codec))
        return self.gpu[0].name()

    @staticmethod
    def load(dir: String, fmt: Int = Q_F32) raises -> MagentaRT:
        """`dir`: a google/magenta-realtime-2 download (needs checkpoints/mrt2_small.safetensors)."""
        var path = dir + "/checkpoints/mrt2_small.safetensors"
        var st = SafeTensors()
        st.add_file(path)
        var codec = SpectroStreamDecoder.load(st, "params/soundstream/decoder/", "params/soundstream/quantizer/embedding",
                                              MRT_LEVELS)
        var ld = WeightLoader(st^, fmt)
        var lm = DepthformerLM(ld^)
        return MagentaRT(lm^, codec^)

    def reset(mut self, seed: Int = 0) raises:
        self.reset_key(fold_in(prng_key(seed), 0))

    def reset_key(mut self, key: JaxKey) raises:
        self.lm.reset(key)
        self.codec.reset()
        if len(self.gpu) > 0:
            self.gpu[0].reset()
        self.frames = 0

    def set_style(mut self, tokens: List[Int]):
        for i in range(MRT_STYLE_TOKENS):
            self.cond.style[i] = tokens[i] if i < len(tokens) else -1

    def set_notes(mut self, notes: List[Int]):
        """128 MIDI pitch states (-1 masked, 0 off, 1 sustain, 2 onset, 3 model's choice)."""
        for i in range(128):
            self.cond.notes[i] = notes[i] if i < len(notes) else -1

    def set_drums(mut self, drums: Int):
        """-1 model's choice, 0 no drums, 1 drums."""
        self.cond.drums = drums

    def set_guidance(mut self, style: Float64, notes: Float64, drums: Float64):
        """Classifier-free-guidance strengths (-1..7; defaults 3, 1, 1)."""
        self.cond.cfg_style = style
        self.cond.cfg_notes = notes
        self.cond.cfg_drums = drums

    def set_sampling(mut self, temperature: Float32, top_k: Int):
        self.lm.temperature = temperature
        self.lm.top_k = top_k

    def next_codes(mut self) -> List[Int]:
        var t0 = perf_counter_ns()
        var codes = self.lm.step(self.cond.row())
        self.lm_ns += Int(perf_counter_ns() - t0)
        return codes^

    def decode(mut self, codes: List[Int], out_addr: Int) raises:
        """Codes -> 1920 interleaved stereo samples at out_addr."""
        var t0 = perf_counter_ns()
        var pcm: Int
        if len(self.gpu) > 0:
            self.codec.embed(codes, self.emb.addr)
            var h = self.gpu[0].run_embedding(self.emb.addr)
            pcm = self.codec.finish_frame(self.codec_pool, h, h + SS_FRAME_SAMPLES * 2 * 4)
        else:
            pcm = self.codec.decode_frame(self.codec_pool, codes)
        copy_floats(out_addr, pcm, SS_FRAME_SAMPLES * 2)
        self.codec_ns += Int(perf_counter_ns() - t0)

    def next_frame(mut self, out_addr: Int) raises:
        """One 40 ms frame (1920 interleaved stereo float32 samples) at out_addr."""
        var codes = self.next_codes()
        self.decode(codes, out_addr)
        self.frames += 1

    def generate(mut self, frames: Int) raises -> List[Float32]:
        """`frames` x 40 ms of interleaved stereo float32 at 48 kHz."""
        var out = List[Float32](length=frames * SS_FRAME_SAMPLES * 2, fill=0.0)
        for f in range(frames):
            self.next_frame(Int(out.unsafe_ptr()) + f * SS_FRAME_SAMPLES * 2 * 4)
        return out^


# ------------------------------------------------------------------- adaptive soundtrack

comptime MRT_STYLE_UPDATE_FRAMES: Int = 5   # re-quantise a morphing style every 200 ms


struct MagentaRtStream(Movable):
    """Adaptive game soundtrack: text-prompted Magenta RT music into an `AudioServer`.

    `set_prompt(text, morph_seconds)` changes the style. Over `morph_seconds` the MusicCoCa
    style embedding is interpolated from the current style to the new one and re-quantised
    every 200 ms, so the change is a musical crossfade inside one continuous stream (the model
    keeps its 20 s of context); 0 switches at the next frame. `blend(a, b, t)` mixes two
    prompts. `pump(ring)` generates frames while `ring` has room for them; run it on a
    generator thread or once per game tick, and give the ring to `AudioServer.add_pcm_stream`.
    """
    var mrt: MagentaRT
    var coca: MusicCoCaText
    var cur: List[Float32]
    var src: List[Float32]
    var dst: List[Float32]
    var morph_total: Int
    var morph_done: Int
    var seed: Int
    var frame: List[Float32]

    def __init__(out self, var mrt: MagentaRT, var coca: MusicCoCaText):
        self.mrt = mrt^
        self.coca = coca^
        self.cur = List[Float32]()
        self.src = List[Float32]()
        self.dst = List[Float32]()
        self.morph_total = 0
        self.morph_done = 0
        self.seed = 0
        self.frame = List[Float32](length=SS_FRAME_SAMPLES * 2, fill=0.0)

    @staticmethod
    def load(dir: String, fmt: Int = Q_F32) raises -> MagentaRtStream:
        """`dir`: the magenta-realtime-2 download with
        `resources/musiccoca/musiccoca_text.safetensors` from `tools/magenta_rt/convert_musiccoca.py`."""
        var mrt = MagentaRT.load(dir, fmt)
        var coca = MusicCoCaText.load(dir + "/resources/musiccoca/musiccoca_text.safetensors")
        return MagentaRtStream(mrt^, coca^)

    def style_of(mut self, text: String) -> List[Float32]:
        """Style embedding of a prompt (through the mapper, as Magenta RT prompts are)."""
        return self.coca.embed_style(text, True, self.seed)

    def set_style_embedding(mut self, emb: List[Float32], morph_seconds: Float64 = 0.0):
        var frames = Int(morph_seconds * 25.0 + 0.5)
        if len(self.cur) == 0 or frames <= 0:
            self.cur = emb.copy()
            self.morph_total = 0
            self.mrt.set_style(self.coca.tokenize(self.cur))
            return
        self.src = self.cur.copy()
        self.dst = emb.copy()
        self.morph_total = frames
        self.morph_done = 0

    def set_prompt(mut self, text: String, morph_seconds: Float64 = 0.0):
        self.set_style_embedding(self.style_of(text), morph_seconds)

    def blend(mut self, a: String, b: String, t: Float32, morph_seconds: Float64 = 0.0):
        """Style (1 - t) a + t b."""
        self.set_style_embedding(blend_embeddings(self.style_of(a), self.style_of(b), t), morph_seconds)

    def morphing(self) -> Bool:
        return self.morph_total > 0

    def _advance_morph(mut self):
        if self.morph_total <= 0:
            return
        self.morph_done += 1
        if self.morph_done >= self.morph_total:
            self.cur = self.dst.copy()
            self.morph_total = 0
            self.mrt.set_style(self.coca.tokenize(self.cur))
            return
        if self.morph_done % MRT_STYLE_UPDATE_FRAMES == 1:
            var t = Float32(self.morph_done) / Float32(self.morph_total)
            # equal-power path keeps the blended embedding's norm from dipping mid-way
            var wa = cos(t * 1.5707963267948966)
            var wb = sin(t * 1.5707963267948966)
            var e = List[Float32](capacity=len(self.src))
            for i in range(len(self.src)):
                e.append(wa * self.src[i] + wb * self.dst[i])
            self.cur = e^
            self.mrt.set_style(self.coca.tokenize(self.cur))

    def next_frame(mut self, out_addr: Int) raises:
        self._advance_morph()
        self.mrt.next_frame(out_addr)

    def pump(mut self, ring: PcmStreamRing, max_frames: Int = 4) raises -> Int:
        """Generates up to `max_frames` 40 ms frames into `ring` while it has room; returns
        how many were produced."""
        var made = 0
        while made < max_frames and ring.free_frames() >= SS_FRAME_SAMPLES:
            self.next_frame(Int(self.frame.unsafe_ptr()))
            _ = ring.write(Int(self.frame.unsafe_ptr()), SS_FRAME_SAMPLES)
            made += 1
        return made
