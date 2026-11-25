"""Modern decoder-only transformer with a KV cache (`flint.ml.llm.model`).

Runs Qwen3 (`model_type: qwen3`, e.g. Qwen3-0.6B / 1.7B) and SmolLM3 (`smollm3`, e.g.
SmolLM3-3B) checkpoints straight from their safetensors: pre-RMSNorm blocks, rotary position
embeddings (rotate-half, per-layer opt-out for SmolLM3's NoPE layers), grouped-query attention,
optional per-head QK RMSNorm (Qwen3), SwiGLU MLP and tied input / output embeddings.

Weights are converted once at load into `QMat`s (fp32, int8 or int4; q/k/v and gate/up are
stacked so a block runs four products). With a MAX GPU every block runs on the device
(`flint.ml.llm.kernels`): weights and the KV cache are resident, a step uploads one embedding
row and downloads the logits. Without one, the same block runs on the CPU job pool.

The model remembers the tokens in its cache: `prompt(tokens)` reuses the longest common prefix
with what is cached (an NPC's fixed system prompt is evaluated once), and `push(token)` appends
one token. Prompts run in chunks of `CHUNK` rows.

`rewind(n)` drops every cached token past the first `n` (speculative decoding rolls rejected
draft tokens back this way: K / V rows are addressed by position, so the next evaluation simply
overwrites them, on the GPU as on the CPU). `eval_hidden(tokens)` returns the final-normed
hidden state of the last token instead of logits (embedding models; `head=False` skips the
output projection entirely, and checkpoints saved without the `model.` prefix load as well).
"""

from std.math import sqrt, exp, cos, sin
from std.collections import Optional
from std.ffi import external_call
from flint.core.jobs import JobPool
from flint.core.ecs import run_parallel_ctx
from flint.platform.os import read_bytes
from flint.assets.json import parse_json
from flint.ml.llm.safetensors import SafeTensors, SafeTensor, widen_to_f32
from flint.ml.llm.quant import QMat, Q_F32, Q_INT8, Q_INT4, qmatmul, qmat_bytes, qmat_scales, fmt_name
from flint.ml.llm.kernels import LlmGpu, FP, BP, gpu_layout

comptime CHUNK: Int = 128
comptime LANES: Int = 8
comptime F8 = SIMD[DType.float32, LANES]


@always_inline
def _f(addr: Int) -> Pointer[Float32, MutAnyOrigin]:
    return Pointer[Float32, MutAnyOrigin](unsafe_from_address=addr)


def _alloc(n: Int) -> Int:
    return external_call["calloc", Int](max(n, 1), 4)


def _free(p: Int):
    if p != 0:
        external_call["free", NoneType](p)


def cpu_count() -> Int:
    return Int(external_call["sysconf", Int64](Int32(84)))


struct LlmConfig(Copyable, Movable):
    var family: String
    var hidden: Int
    var layers: Int
    var heads: Int
    var kv_heads: Int
    var head_dim: Int
    var ffn: Int
    var vocab: Int
    var eps: Float32
    var theta: Float64
    var qk_norm: Bool
    var rope_layer: List[Bool]
    var tied: Bool
    var bos: Int
    var eos: List[Int]

    def __init__(out self):
        self.family = ""
        self.hidden = 0
        self.layers = 0
        self.heads = 0
        self.kv_heads = 0
        self.head_dim = 0
        self.ffn = 0
        self.vocab = 0
        self.eps = 1e-6
        self.theta = 10000.0
        self.qk_norm = False
        self.rope_layer = List[Bool]()
        self.tied = True
        self.bos = -1
        self.eos = List[Int]()

    @staticmethod
    def load(path: String) raises -> LlmConfig:
        var data = read_bytes(path)
        var doc = parse_json(data)
        if not doc.valid:
            raise Error("llm: cannot parse " + path)
        var r = doc.root
        var c = LlmConfig()
        c.family = doc.text(doc.object_get(r, "model_type"))
        if c.family != "qwen3" and c.family != "smollm3" and c.family != "llama":
            raise Error("llm: unsupported model_type '" + c.family + "' (qwen3, smollm3, llama)")
        c.hidden = doc.integer(doc.object_get(r, "hidden_size"))
        c.layers = doc.integer(doc.object_get(r, "num_hidden_layers"))
        c.heads = doc.integer(doc.object_get(r, "num_attention_heads"))
        c.kv_heads = doc.integer(doc.object_get(r, "num_key_value_heads"), c.heads)
        c.head_dim = doc.integer(doc.object_get(r, "head_dim"), c.hidden // c.heads)
        c.ffn = doc.integer(doc.object_get(r, "intermediate_size"))
        c.vocab = doc.integer(doc.object_get(r, "vocab_size"))
        c.eps = Float32(doc.number(doc.object_get(r, "rms_norm_eps"), 1e-6))
        c.theta = doc.number(doc.object_get(r, "rope_theta"), 10000.0)
        c.qk_norm = c.family == "qwen3"
        c.tied = doc.boolean(doc.object_get(r, "tie_word_embeddings"), True)
        if doc.boolean(doc.object_get(r, "attention_bias"), False):
            raise Error("llm: attention biases are not supported")
        var rs = doc.object_get(r, "rope_scaling")
        if rs >= 0 and doc.nodes[rs].kind != 0:
            raise Error("llm: rope_scaling is not supported")
        c.bos = doc.integer(doc.object_get(r, "bos_token_id"), -1)
        var e = doc.object_get(r, "eos_token_id")
        if doc.array_size(e) > 0:
            for i in range(doc.array_size(e)):
                c.eos.append(doc.integer(doc.at(e, i)))
        elif e >= 0:
            c.eos.append(doc.integer(e))
        var nr = doc.object_get(r, "no_rope_layers")
        for l in range(c.layers):
            if doc.array_size(nr) == c.layers:
                c.rope_layer.append(doc.integer(doc.at(nr, l)) != 0)
            else:
                c.rope_layer.append(True)
        if c.head_dim % 4 != 0 or c.head_dim > 256:
            raise Error("llm: head_dim must be a multiple of 4 up to 256")
        return c^


struct LlmLayer(Copyable, Movable):
    var qkv: QMat
    var o: QMat
    var gu: QMat
    var down: QMat
    var ln1: Int
    var ln2: Int
    var qn: Int
    var kn: Int
    var d_ln1: Int
    var d_ln2: Int
    var d_qn: Int
    var d_kn: Int

    def __init__(out self):
        self.qkv = QMat()
        self.o = QMat()
        self.gu = QMat()
        self.down = QMat()
        self.ln1 = 0
        self.ln2 = 0
        self.qn = 0
        self.kn = 0
        self.d_ln1 = -1
        self.d_ln2 = -1
        self.d_qn = -1
        self.d_kn = -1


struct _AttnCtx(Movable):
    var q: Int
    var o: Int
    var kc: Int
    var vc: Int
    var cap: Int
    var heads: Int
    var kvh: Int
    var hd: Int
    var pos0: Int
    var scale: Float32
    var scratch: Int

    def __init__(out self, q: Int, o: Int, kc: Int, vc: Int, cap: Int, heads: Int, kvh: Int, hd: Int, pos0: Int,
                 scale: Float32, scratch: Int):
        self.q = q
        self.o = o
        self.kc = kc
        self.vc = vc
        self.cap = cap
        self.heads = heads
        self.kvh = kvh
        self.hd = hd
        self.pos0 = pos0
        self.scale = scale
        self.scratch = scratch


@always_inline
def _dot(a: Int, b: Int, n: Int) -> Float32:
    var pa = _f(a)
    var pb = _f(b)
    var acc = F8(0)
    var i = 0
    while i + LANES <= n:
        acc += pa.unsafe_offset(i).unsafe_load[width=LANES, alignment=1]() * pb.unsafe_offset(i).unsafe_load[width=LANES, alignment=1]()
        i += LANES
    var s = acc.reduce_add()
    while i < n:
        s += pa.unsafe_offset(i).unsafe_load() * pb.unsafe_offset(i).unsafe_load()
        i += 1
    return s


@always_inline
def _axpy(y: Int, a: Float32, x: Int, n: Int):
    var py = _f(y)
    var px = _f(x)
    var av = F8(a)
    var i = 0
    while i + LANES <= n:
        py.unsafe_offset(i).unsafe_store[alignment=1](py.unsafe_offset(i).unsafe_load[width=LANES, alignment=1]()
                                                      + av * px.unsafe_offset(i).unsafe_load[width=LANES, alignment=1]())
        i += LANES
    while i < n:
        py.unsafe_offset(i).unsafe_store(py.unsafe_offset(i).unsafe_load() + a * px.unsafe_offset(i).unsafe_load())
        i += 1


def _attn_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_AttnCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var H = c[].heads
    var D = c[].hd
    var r = job // H
    var h = job % H
    var kh = h // (H // c[].kvh)
    var qa = c[].q + ((r * H + h) * D) * 4
    var n_keys = c[].pos0 + r + 1
    var sc = _f(c[].scratch + job * c[].cap * 4)
    var kb = c[].kc + kh * c[].cap * D * 4
    var vb = c[].vc + kh * c[].cap * D * 4
    var m: Float32 = -3.4e38
    for t in range(n_keys):
        var s = _dot(qa, kb + t * D * 4, D) * c[].scale
        sc.unsafe_offset(t).unsafe_store(s)
        m = max(m, s)
    var tot: Float32 = 0
    for t in range(n_keys):
        var e = exp(sc.unsafe_offset(t).unsafe_load() - m)
        sc.unsafe_offset(t).unsafe_store(e)
        tot += e
    var inv = 1.0 / tot
    var oa = c[].o + ((r * H + h) * D) * 4
    var po = _f(oa)
    for i in range(D):
        po.unsafe_offset(i).unsafe_store(0)
    for t in range(n_keys):
        _axpy(oa, sc.unsafe_offset(t).unsafe_load() * inv, vb + t * D * 4, D)


def rms_norm_rows(src: Int, dst: Int, rows: Int, n: Int, g: Int, eps: Float32):
    for r in range(rows):
        var s = _dot(src + r * n * 4, src + r * n * 4, n)
        var inv = 1.0 / sqrt(s / Float32(n) + eps)
        var ps = _f(src + r * n * 4)
        var pd = _f(dst + r * n * 4)
        var pg = _f(g)
        for i in range(n):
            pd.unsafe_offset(i).unsafe_store(ps.unsafe_offset(i).unsafe_load() * inv * pg.unsafe_offset(i).unsafe_load())


struct LlmModel(Movable):
    var cfg: LlmConfig
    var fmt: Int
    var st: SafeTensors
    var embed: SafeTensor
    var layers: List[LlmLayer]
    var lm_head: QMat
    var norm: Int
    var d_norm: Int
    var pool: JobPool
    var gpu: Optional[LlmGpu]
    var cap: Int
    var past: List[Int]
    var rope_cos: Int     # [cap, head_dim / 2]
    var rope_sin: Int
    # host working memory
    var kc: Int           # [layers, kv_heads, cap, head_dim]
    var vc: Int
    var b_x: Int
    var b_h: Int
    var b_qkv: Int
    var b_q: Int
    var b_att: Int
    var b_gu: Int
    var b_act: Int
    var b_logits: Int
    var b_scratch: Int
    # device working memory (indices into gpu.fbufs)
    var d_kc: Int
    var d_vc: Int
    var d_x: Int
    var d_h: Int
    var d_qkv: Int
    var d_q: Int
    var d_att: Int
    var d_gu: Int
    var d_act: Int
    var d_logits: Int
    var d_cos: Int
    var d_sin: Int
    var weight_bytes: Int
    var tokens_evaluated: Int
    var prefix: String    # "model." for *ForCausalLM checkpoints, "" for bare Qwen3Model ones
    var has_head: Bool

    def __init__(out self, dir: String, fmt: Int, use_gpu: Bool, ctx: Int, head: Bool = True) raises:
        self.cfg = LlmConfig.load(dir + "/config.json")
        self.fmt = fmt
        self.st = SafeTensors.open_dir(dir)
        self.embed = SafeTensor("", "F32", List[Int](), 0)
        self.layers = List[LlmLayer]()
        self.lm_head = QMat()
        self.norm = 0
        self.d_norm = -1
        self.pool = JobPool()
        _ = self.pool.start(max(1, min(31, cpu_count() - 1)))
        self.gpu = None
        self.cap = ctx
        self.past = List[Int]()
        self.rope_cos = 0
        self.rope_sin = 0
        self.kc = 0
        self.vc = 0
        self.b_x = 0
        self.b_h = 0
        self.b_qkv = 0
        self.b_q = 0
        self.b_att = 0
        self.b_gu = 0
        self.b_act = 0
        self.b_logits = 0
        self.b_scratch = 0
        self.d_kc = -1
        self.d_vc = -1
        self.d_x = -1
        self.d_h = -1
        self.d_qkv = -1
        self.d_q = -1
        self.d_att = -1
        self.d_gu = -1
        self.d_act = -1
        self.d_logits = -1
        self.d_cos = -1
        self.d_sin = -1
        self.weight_bytes = 0
        self.tokens_evaluated = 0
        self.prefix = String("model.") if self.st.has("model.embed_tokens.weight") else String("")
        self.has_head = head
        self.embed = self.st.get(self.prefix + "embed_tokens.weight")
        if use_gpu:
            try:
                self.gpu = LlmGpu()
            except:
                self.gpu = None
        self._load_weights()
        self._rope_table()
        self._buffers()

    @staticmethod
    def load(dir: String, fmt: Int = Q_INT8, gpu: Bool = True, ctx: Int = 2048, head: Bool = True) raises -> LlmModel:
        return LlmModel(dir, fmt, gpu, ctx, head)

    def __deinit__(deinit self):
        for ref l in self.layers:
            l.qkv.release_host()
            l.o.release_host()
            l.gu.release_host()
            l.down.release_host()
            _free(l.ln1)
            _free(l.ln2)
            _free(l.qn)
            _free(l.kn)
        self.lm_head.release_host()
        for p in [self.norm, self.rope_cos, self.rope_sin, self.kc, self.vc, self.b_x, self.b_h, self.b_qkv, self.b_q,
                  self.b_att, self.b_gu, self.b_act, self.b_logits, self.b_scratch]:
            _free(p)
        self.pool.stop()

    def device(self) raises -> String:
        if self.gpu:
            return self.gpu.value().name()
        return String("CPU (" + String(cpu_count()) + " threads)")

    def on_gpu(self) -> Bool:
        return Bool(self.gpu)

    def format_name(self) -> String:
        return fmt_name(self.fmt)

    # ------------------------------------------------------------------ loading

    def _vec(mut self, name: String) raises -> Int:
        return self.st.to_f32(name)

    def _mat(mut self, var names: List[String], fmt: Int) raises -> QMat:
        var srcs = List[SafeTensor]()
        for n in names:
            srcs.append(self.st.get(n))
        var m = QMat.build(self.pool, srcs^, fmt)
        self.weight_bytes += m.bytes()
        if self.gpu:
            if m.fmt == Q_F32:
                m.dev = self.gpu.value().upload_bytes(m.data, qmat_bytes(m.fmt, m.n, m.k))
            else:
                var packed = gpu_layout(m.fmt, m.data, m.n, m.k)
                m.dev = self.gpu.value().upload_bytes(packed, qmat_bytes(m.fmt, m.n, m.k))
                _free(packed)
            if m.scales != 0:
                m.dev_scales = self.gpu.value().upload_bytes(m.scales, qmat_scales(m.fmt, m.n, m.k) * 4)
            else:
                m.dev_scales = m.dev
            m.release_host()
        return m^

    def _dvec(mut self, p: Int, n: Int) raises -> Int:
        if self.gpu:
            return self.gpu.value().upload_bytes(p, n * 4)
        return -1

    def _load_weights(mut self) raises:
        var c = self.cfg.copy()
        for l in range(c.layers):
            var p = self.prefix + "layers." + String(l) + "."
            var ly = LlmLayer()
            ly.qkv = self._mat([p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                                p + "self_attn.v_proj.weight"], self.fmt)
            ly.o = self._mat([p + "self_attn.o_proj.weight"], self.fmt)
            ly.gu = self._mat([p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"], self.fmt)
            ly.down = self._mat([p + "mlp.down_proj.weight"], self.fmt)
            ly.ln1 = self._vec(p + "input_layernorm.weight")
            ly.ln2 = self._vec(p + "post_attention_layernorm.weight")
            ly.d_ln1 = self._dvec(ly.ln1, c.hidden)
            ly.d_ln2 = self._dvec(ly.ln2, c.hidden)
            if c.qk_norm:
                ly.qn = self._vec(p + "self_attn.q_norm.weight")
                ly.kn = self._vec(p + "self_attn.k_norm.weight")
                ly.d_qn = self._dvec(ly.qn, c.head_dim)
                ly.d_kn = self._dvec(ly.kn, c.head_dim)
            else:
                ly.d_qn = ly.d_ln1
                ly.d_kn = ly.d_ln1
            self.layers.append(ly^)
        self.norm = self._vec(self.prefix + "norm.weight")
        self.d_norm = self._dvec(self.norm, c.hidden)
        if not self.has_head:
            return
        var head = String("lm_head.weight") if self.st.has("lm_head.weight") else self.prefix + "embed_tokens.weight"
        self.lm_head = self._mat([head], self.fmt)

    def _rope_table(mut self) raises:
        var D = self.cfg.head_dim
        var half = D // 2
        self.rope_cos = _alloc(self.cap * half)
        self.rope_sin = _alloc(self.cap * half)
        var pc = _f(self.rope_cos)
        var ps = _f(self.rope_sin)
        for i in range(half):
            # inv_freq as Hugging Face computes it, in float32
            var expo = Float32(2 * i) / Float32(D)
            var inv = Float32(1.0) / Float32(self.cfg.theta ** Float64(expo))
            for p in range(self.cap):
                var ang = Float32(p) * inv
                pc.unsafe_offset(p * half + i).unsafe_store(Float32(cos(Float64(ang))))
                ps.unsafe_offset(p * half + i).unsafe_store(Float32(sin(Float64(ang))))
        if self.gpu:
            ref g = self.gpu.value()
            self.d_cos = g.alloc(self.cap * half)
            self.d_sin = g.alloc(self.cap * half)
            g.upload(self.d_cos, self.rope_cos, self.cap * half)
            g.upload(self.d_sin, self.rope_sin, self.cap * half)

    def _buffers(mut self) raises:
        var c = self.cfg.copy()
        var qkv_w = (c.heads + 2 * c.kv_heads) * c.head_dim
        var att_w = c.heads * c.head_dim
        var kv = c.layers * c.kv_heads * self.cap * c.head_dim
        if self.gpu:
            ref g = self.gpu.value()
            self.d_kc = g.alloc(kv)
            self.d_vc = g.alloc(kv)
            self.d_x = g.alloc(CHUNK * c.hidden)
            self.d_h = g.alloc(CHUNK * c.hidden)
            self.d_qkv = g.alloc(CHUNK * qkv_w)
            self.d_q = g.alloc(CHUNK * att_w)
            self.d_att = g.alloc(CHUNK * att_w)
            self.d_gu = g.alloc(CHUNK * 2 * c.ffn)
            self.d_act = g.alloc(CHUNK * c.ffn)
            self.d_logits = g.alloc(CHUNK * c.vocab if self.has_head else CHUNK * c.hidden)
            g.sync()
            self.b_x = _alloc(CHUNK * c.hidden)
            return
        self.kc = _alloc(kv)
        self.vc = _alloc(kv)
        self.b_x = _alloc(CHUNK * c.hidden)
        self.b_h = _alloc(CHUNK * c.hidden)
        self.b_qkv = _alloc(CHUNK * qkv_w)
        self.b_q = _alloc(CHUNK * att_w)
        self.b_att = _alloc(CHUNK * att_w)
        self.b_gu = _alloc(CHUNK * 2 * c.ffn)
        self.b_act = _alloc(CHUNK * c.ffn)
        self.b_logits = _alloc(CHUNK * c.vocab if self.has_head else 1)
        self.b_scratch = _alloc(CHUNK * c.heads * self.cap)

    # ------------------------------------------------------------------ evaluation

    def reset(mut self):
        self.past = List[Int]()

    def cached(self) -> Int:
        """How many tokens the KV cache holds."""
        return len(self.past)

    def rewind(mut self, n: Int):
        """Keeps only the first `n` cached tokens (no-op when fewer are cached). The K / V rows
        past `n` stay in memory but are never read again: attention only looks at positions
        below the current length, and the next evaluation overwrites them."""
        while len(self.past) > max(n, 0):
            _ = self.past.pop()

    def eval_hidden(mut self, tokens: List[Int]) raises -> List[Float32]:
        """Appends `tokens` after the cached ones and returns the final RMS-normed hidden state
        of the last one ([hidden]), the input to the output projection (last-token pooling)."""
        return self._eval(tokens, False, True)

    def prompt(mut self, tokens: List[Int]) raises -> List[Float32]:
        """Makes the cache hold exactly `tokens` (reusing the longest cached prefix) and returns
        the next-token logits."""
        if len(tokens) == 0:
            raise Error("llm: empty prompt")
        var p = 0
        while p < len(self.past) and p < len(tokens) and self.past[p] == tokens[p]:
            p += 1
        if p == len(tokens):
            p -= 1
        while len(self.past) > p:
            _ = self.past.pop()
        var rest = List[Int]()
        for i in range(p, len(tokens)):
            rest.append(tokens[i])
        return self.eval(rest, False)

    def push(mut self, token: Int) raises -> List[Float32]:
        var one = List[Int]()
        one.append(token)
        return self.eval(one, False)

    def eval(mut self, tokens: List[Int], all_logits: Bool) raises -> List[Float32]:
        """Appends `tokens` after the cached ones; returns the logits of the last one, or of
        every one ([len, vocab]) when `all_logits`."""
        if not self.has_head:
            raise Error("llm: model loaded without an output head (use eval_hidden)")
        return self._eval(tokens, all_logits, False)

    def _eval(mut self, tokens: List[Int], all_logits: Bool, hidden: Bool) raises -> List[Float32]:
        if len(self.past) + len(tokens) > self.cap:
            raise Error("llm: context of " + String(self.cap) + " tokens exceeded")
        var V = self.cfg.hidden if hidden else self.cfg.vocab
        var out = List[Float32]()
        var i = 0
        while i < len(tokens):
            var m = min(CHUNK, len(tokens) - i)
            var last = i + m == len(tokens)
            var rows_out = m if all_logits else (1 if last else 0)
            var pos0 = len(self.past)
            var hid = self.cfg.hidden
            for r in range(m):
                var t = tokens[i + r]
                if t < 0 or t >= self.embed.rows():
                    raise Error("llm: token id out of range " + String(t))
                widen_to_f32(self.embed, self.b_x + r * hid * 4, t * hid, hid)
            var lg = List[Float32](length=rows_out * V, fill=0.0)
            if self.gpu:
                self._gpu_chunk(m, pos0, rows_out, Int(lg.unsafe_ptr()), hidden)
            else:
                self._cpu_chunk(m, pos0, rows_out, Int(lg.unsafe_ptr()), hidden)
            for r in range(m):
                self.past.append(tokens[i + r])
            if all_logits and i == 0 and last:
                out = lg^
            elif all_logits:
                for v in lg:
                    out.append(v)
            elif last:
                out = lg^
            i += m
        self.tokens_evaluated += len(tokens)
        return out^

    def _cpu_chunk(mut self, m: Int, pos0: Int, rows_out: Int, dst: Int, hidden: Bool) raises:
        var c = self.cfg.copy()
        var H = c.heads
        var KV = c.kv_heads
        var D = c.head_dim
        var hid = c.hidden
        var qkv_w = (H + 2 * KV) * D
        var half = D // 2
        var scale = Float32(1.0 / sqrt(Float64(D)))
        var layer_kv = KV * self.cap * D
        for l in range(c.layers):
            ref ly = self.layers[l]
            rms_norm_rows(self.b_x, self.b_h, m, hid, ly.ln1, c.eps)
            qmatmul(self.pool, ly.qkv, self.b_h, self.b_qkv, m, False)
            var kcl = self.kc + l * layer_kv * 4
            var vcl = self.vc + l * layer_kv * 4
            var use_rope = c.rope_layer[l]
            var tmp = List[Float32](length=D, fill=0.0)
            var pt = Int(tmp.unsafe_ptr())
            for r in range(m):
                var pos = pos0 + r
                for h in range(H + KV):
                    var src = self.b_qkv + (r * qkv_w + h * D) * 4
                    if c.qk_norm:
                        rms_norm_rows(src, pt, 1, D, ly.qn if h < H else ly.kn, c.eps)
                    else:
                        _ = external_call["memcpy", Int](pt, src, D * 4)
                    var dsta: Int
                    if h < H:
                        dsta = self.b_q + (r * H + h) * D * 4
                    else:
                        var j = h - H
                        dsta = kcl + (j * self.cap + pos) * D * 4
                        _ = external_call["memcpy", Int](vcl + (j * self.cap + pos) * D * 4,
                                                         self.b_qkv + (r * qkv_w + (H + KV + j) * D) * 4, D * 4)
                    var pd = _f(dsta)
                    if use_rope:
                        var pc = _f(self.rope_cos + pos * half * 4)
                        var ps = _f(self.rope_sin + pos * half * 4)
                        for f in range(half):
                            var a = tmp[f]
                            var b = tmp[f + half]
                            var co = pc.unsafe_offset(f).unsafe_load()
                            var si = ps.unsafe_offset(f).unsafe_load()
                            pd.unsafe_offset(f).unsafe_store(a * co - b * si)
                            pd.unsafe_offset(f + half).unsafe_store(b * co + a * si)
                    else:
                        _ = external_call["memcpy", Int](dsta, pt, D * 4)
            _ = len(tmp)
            var ctx = _AttnCtx(self.b_q, self.b_att, kcl, vcl, self.cap, H, KV, D, pos0, scale, self.b_scratch)
            run_parallel_ctx(self.pool, m * H, ctx, _attn_worker)
            qmatmul(self.pool, ly.o, self.b_att, self.b_x, m, True)
            rms_norm_rows(self.b_x, self.b_h, m, hid, ly.ln2, c.eps)
            qmatmul(self.pool, ly.gu, self.b_h, self.b_gu, m, False)
            var F = c.ffn
            for r in range(m):
                var pg = _f(self.b_gu + r * 2 * F * 4)
                var pa = _f(self.b_act + r * F * 4)
                for i in range(F):
                    var g = pg.unsafe_offset(i).unsafe_load()
                    pa.unsafe_offset(i).unsafe_store(g / (1.0 + exp(-g)) * pg.unsafe_offset(F + i).unsafe_load())
            qmatmul(self.pool, ly.down, self.b_act, self.b_x, m, True)
        if rows_out == 0:
            return
        var r0 = m - rows_out
        rms_norm_rows(self.b_x + r0 * hid * 4, self.b_h, rows_out, hid, self.norm, c.eps)
        if hidden:
            _ = external_call["memcpy", Int](dst, self.b_h, rows_out * hid * 4)
            return
        qmatmul(self.pool, self.lm_head, self.b_h, dst, rows_out, False)

    def _gpu_chunk(mut self, m: Int, pos0: Int, rows_out: Int, dst: Int, hidden: Bool) raises:
        var c = self.cfg.copy()
        var H = c.heads
        var KV = c.kv_heads
        var D = c.head_dim
        var hid = c.hidden
        var scale = Float32(1.0 / sqrt(Float64(D)))
        var layer_kv = KV * self.cap * D
        ref g = self.gpu.value()
        g.upload(self.d_x, self.b_x, m * hid)
        var px = g.ptr(self.d_x)
        var ph = g.ptr(self.d_h)
        var pqkv = g.ptr(self.d_qkv)
        var pq = g.ptr(self.d_q)
        var patt = g.ptr(self.d_att)
        var pgu = g.ptr(self.d_gu)
        var pact = g.ptr(self.d_act)
        var pcos = g.ptr(self.d_cos)
        var psin = g.ptr(self.d_sin)
        var kc0 = g.ptr(self.d_kc)
        var vc0 = g.ptr(self.d_vc)
        for l in range(c.layers):
            ref ly = self.layers[l]
            g.rmsnorm(px, g.wfptr(ly.d_ln1), ph, m, hid, c.eps)
            g.matvec(ly.qkv.fmt, ph, g.wptr(ly.qkv.dev), g.wfptr(ly.qkv.dev_scales), pqkv, m, hid, ly.qkv.n, False)
            var kcl = kc0.unsafe_offset(l * layer_kv)
            var vcl = vc0.unsafe_offset(l * layer_kv)
            g.qk_rope(pqkv, pq, kcl, vcl, g.wfptr(ly.d_qn), g.wfptr(ly.d_kn), pcos, psin, m, H, KV, D, pos0, self.cap,
                      c.qk_norm, c.rope_layer[l], c.eps)
            g.attention(pq, patt, kcl, vcl, m, H, KV, D, pos0, self.cap, scale)
            g.matvec(ly.o.fmt, patt, g.wptr(ly.o.dev), g.wfptr(ly.o.dev_scales), px, m, ly.o.k, hid, True)
            g.rmsnorm(px, g.wfptr(ly.d_ln2), ph, m, hid, c.eps)
            g.matvec(ly.gu.fmt, ph, g.wptr(ly.gu.dev), g.wfptr(ly.gu.dev_scales), pgu, m, hid, ly.gu.n, False)
            g.silu_mul(pgu, pact, m, c.ffn)
            g.matvec(ly.down.fmt, pact, g.wptr(ly.down.dev), g.wfptr(ly.down.dev_scales), px, m, c.ffn, hid, True)
        if rows_out == 0:
            g.sync()
            return
        var r0 = m - rows_out
        g.rmsnorm(px.unsafe_offset(r0 * hid), g.wfptr(self.d_norm), ph, rows_out, hid, c.eps)
        if hidden:
            g.download(self.d_h, dst, rows_out * hid)
            return
        g.matvec(self.lm_head.fmt, ph, g.wptr(self.lm_head.dev), g.wfptr(self.lm_head.dev_scales), g.ptr(self.d_logits),
                 rows_out, hid, self.lm_head.n, False)
        g.download(self.d_logits, dst, rows_out * c.vocab)
