"""OPT decoder-only transformer with a beam-aware KV cache (`flint.ml.opt`).

The architecture of Meta's OPT (pre-LayerNorm blocks, learned positions offset by 2, ReLU MLP,
final LayerNorm, untied LM head), run from a weights-only ONNX file whose initializers use the
Hugging Face names (`model.decoder.layers.{i}.self_attn.q_proj.weight`, ...).

Generation starts from a prefix of input embeddings (UniRig's mesh latents plus start tokens)
that every beam shares: its keys and values are computed once. Each generated token stores its
keys and values once per (step, beam) slot, and a beam reads its own history through parent
pointers, so reordering beams never copies the cache.

With a MAX GPU the whole decoder runs on the device: weights upload once, the KV cache lives in
device memory (K / V projections write straight into their cache slots), and a step uploads its
embeddings and beam ancestry and downloads only the logits. Without one, linear layers and
attention run on the CPU job pool.
"""

from std.math import sqrt, exp
from std.collections import Optional, Dict
from std.ffi import external_call
from flint.core.jobs import JobPool, default_worker_count
from flint.core.ecs import run_parallel_ctx
from flint.ml.onnx import OnnxModel
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.ops import matmul_2d
from flint.ml.gpu import GpuMatmul, FP, GB_ADD, GU_RELU

comptime LANES: Int = 8
comptime F8 = SIMD[DType.float32, LANES]


def _alloc_f32(n: Int) -> Int:
    var p = external_call["calloc", Int](n, 4)
    return p


def _free(p: Int):
    if p != 0:
        external_call["free", NoneType](p)


@always_inline
def _dot(a: Int, b: Int, n: Int) -> Float32:
    var pa = fptr_at(a)
    var pb = fptr_at(b)
    var acc = F8(0)
    var i = 0
    while i + LANES <= n:
        acc += pa.unsafe_offset(i).unsafe_load[width=LANES, alignment=1]() * pb.unsafe_offset(i).unsafe_load[width=LANES, alignment=1]()
        i += LANES
    var s = acc.reduce_add()
    while i < n:
        s += pa.unsafe_offset(i).unsafe_load[alignment=1]() * pb.unsafe_offset(i).unsafe_load[alignment=1]()
        i += 1
    return s


@always_inline
def _axpy(y: Int, a: Float32, x: Int, n: Int):
    var py = fptr_at(y)
    var px = fptr_at(x)
    var av = F8(a)
    var i = 0
    while i + LANES <= n:
        py.unsafe_offset(i).unsafe_store[alignment=1](py.unsafe_offset(i).unsafe_load[width=LANES, alignment=1]()
                                                      + av * px.unsafe_offset(i).unsafe_load[width=LANES, alignment=1]())
        i += LANES
    while i < n:
        py.unsafe_offset(i).unsafe_store[alignment=1](py.unsafe_offset(i).unsafe_load[alignment=1]() + a * px.unsafe_offset(i).unsafe_load[alignment=1]())
        i += 1


def layer_norm_rows(src: Int, dst: Int, rows: Int, n: Int, gamma: Int, beta: Int, eps: Float32):
    var ps = fptr_at(src)
    var pd = fptr_at(dst)
    var pg = fptr_at(gamma)
    var pb = fptr_at(beta)
    for r in range(rows):
        var base = r * n
        var s: Float32 = 0
        for i in range(n):
            s += ps.unsafe_offset(base + i).unsafe_load[alignment=1]()
        var mean = s / Float32(n)
        var v: Float32 = 0
        for i in range(n):
            var d = ps.unsafe_offset(base + i).unsafe_load[alignment=1]() - mean
            v += d * d
        var inv = 1.0 / sqrt(v / Float32(n) + eps)
        for i in range(n):
            var x = (ps.unsafe_offset(base + i).unsafe_load[alignment=1]() - mean) * inv
            pd.unsafe_offset(base + i).unsafe_store[alignment=1](x * pg.unsafe_offset(i).unsafe_load[alignment=1]()
                                                                  + pb.unsafe_offset(i).unsafe_load[alignment=1]())


struct _AttnCtx(Movable):
    """One attention job per (row, head): a query attends to a shared prefix and its own beam's
    generated history (or, for the prefix pass, causally to the prefix)."""
    var q: Int            # [rows, hidden]
    var out: Int          # [rows, hidden]
    var pk: Int           # prefix keys   [prefix_cap, hidden]
    var pv: Int
    var gk: Int           # generated keys [steps_cap, beams_cap, hidden]
    var gv: Int
    var anc: Int          # Int32 [rows, steps] slot beam for each generated step
    var prefix_len: Int
    var steps: Int        # generated tokens visible to each row (including the current one)
    var beams_cap: Int
    var hidden: Int
    var heads: Int
    var head_dim: Int
    var causal_prefix: Bool
    var scratch: Int      # [rows * heads, prefix_cap + steps_cap] score scratch
    var scratch_stride: Int

    def __init__(out self):
        self.q = 0
        self.out = 0
        self.pk = 0
        self.pv = 0
        self.gk = 0
        self.gv = 0
        self.anc = 0
        self.prefix_len = 0
        self.steps = 0
        self.beams_cap = 1
        self.hidden = 0
        self.heads = 1
        self.head_dim = 0
        self.causal_prefix = False
        self.scratch = 0
        self.scratch_stride = 0


def _attn_worker(ctx_addr: Int, job: Int):
    var c = Pointer[_AttnCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)
    var heads = c[].heads
    var row = job // heads
    var h = job % heads
    var hd = c[].head_dim
    var hid = c[].hidden
    var qa = c[].q + (row * hid + h * hd) * 4
    var sc = fptr_at(c[].scratch + job * c[].scratch_stride * 4)
    var npre = c[].prefix_len
    if c[].causal_prefix:
        npre = row + 1
    var total = npre + c[].steps
    var anc = Pointer[Int32, MutAnyOrigin](unsafe_from_address=c[].anc + row * max(c[].steps, 1) * 4)
    var m: Float32 = -3.4e38
    for t in range(npre):
        var s = _dot(qa, c[].pk + (t * hid + h * hd) * 4, hd)
        sc.unsafe_offset(t).unsafe_store(s)
        m = max(m, s)
    for g in range(c[].steps):
        var b = Int(anc.unsafe_offset(g).unsafe_load())
        var s = _dot(qa, c[].gk + ((g * c[].beams_cap + b) * hid + h * hd) * 4, hd)
        sc.unsafe_offset(npre + g).unsafe_store(s)
        m = max(m, s)
    var tot: Float32 = 0
    for t in range(total):
        var e = exp(sc.unsafe_offset(t).unsafe_load() - m)
        sc.unsafe_offset(t).unsafe_store(e)
        tot += e
    var inv = 1.0 / tot
    var oa = c[].out + (row * hid + h * hd) * 4
    var po = fptr_at(oa)
    for i in range(hd):
        po.unsafe_offset(i).unsafe_store(0)
    for t in range(npre):
        _axpy(oa, sc.unsafe_offset(t).unsafe_load() * inv, c[].pv + (t * hid + h * hd) * 4, hd)
    for g in range(c[].steps):
        var b = Int(anc.unsafe_offset(g).unsafe_load())
        _axpy(oa, sc.unsafe_offset(npre + g).unsafe_load() * inv, c[].gv + ((g * c[].beams_cap + b) * hid + h * hd) * 4, hd)


struct OptDecoder(Movable):
    """Weights, caches and scratch for one generation (a prefix plus up to `steps_cap` tokens
    on up to `beams_cap` beams)."""
    var weights: OnnxModel
    var gpu: Optional[GpuMatmul]
    var pool: JobPool
    var layers: Int
    var hidden: Int
    var ffn: Int
    var heads: Int
    var vocab: Int
    var max_pos: Int
    var prefix_cap: Int
    var steps_cap: Int
    var beams_cap: Int
    var prefix_len: Int
    var pk: List[Int]
    var pv: List[Int]
    var gk: List[Int]
    var gv: List[Int]
    var buf_x: Int
    var buf_h: Int
    var buf_q: Int
    var buf_k: Int
    var buf_v: Int
    var buf_a: Int
    var buf_f: Int
    var buf_s: Int
    var rows_cap: Int
    var keys: Dict[String, Int]
    # device path (GPU present): pooled buffers in `gpu`
    var d_pk: List[Int]
    var d_pv: List[Int]
    var d_gk: List[Int]
    var d_gv: List[Int]
    var d_x: Int
    var d_h: Int
    var d_q: Int
    var d_a: Int
    var d_f: Int
    var d_logit: Int
    var d_anc: Int
    var d_rows: Int
    # post-LayerNorm blocks (OPT-350M's `do_layer_norm_before = False`: no final LayerNorm)
    var post_ln: Bool

    def __init__(out self, var weights: OnnxModel, gpu: Bool) raises:
        self.weights = weights^
        self.gpu = None
        if gpu:
            try:
                self.gpu = GpuMatmul()
            except:
                self.gpu = None
        self.pool = JobPool()
        _ = self.pool.start(default_worker_count())
        var tok = self.weights.weight("model.decoder.embed_tokens.weight")
        var pos = self.weights.weight("model.decoder.embed_positions.weight")
        var fc1 = self.weights.weight("model.decoder.layers.0.fc1.weight")
        self.vocab = tok.shape[0]
        self.hidden = tok.shape[1]
        self.max_pos = pos.shape[0] - 2
        self.ffn = fc1.shape[0]
        self.layers = 0
        while self.weights.has("model.decoder.layers." + String(self.layers) + ".fc1.weight"):
            self.layers += 1
        self.heads = self.hidden // 64
        self.prefix_cap = 0
        self.steps_cap = 0
        self.beams_cap = 0
        self.prefix_len = 0
        self.pk = List[Int]()
        self.pv = List[Int]()
        self.gk = List[Int]()
        self.gv = List[Int]()
        self.buf_x = 0
        self.buf_h = 0
        self.buf_q = 0
        self.buf_k = 0
        self.buf_v = 0
        self.buf_a = 0
        self.buf_f = 0
        self.buf_s = 0
        self.rows_cap = 0
        self.keys = Dict[String, Int]()
        self.d_pk = List[Int]()
        self.d_pv = List[Int]()
        self.d_gk = List[Int]()
        self.d_gv = List[Int]()
        self.d_x = -1
        self.d_h = -1
        self.d_q = -1
        self.d_a = -1
        self.d_f = -1
        self.d_logit = -1
        self.d_anc = -1
        self.d_rows = 0
        self.post_ln = not self.weights.has("model.decoder.final_layer_norm.weight")

    @staticmethod
    def load(path: String, gpu: Bool = True) raises -> OptDecoder:
        return OptDecoder(OnnxModel.load(path, 1), gpu)

    def __deinit__(deinit self):
        for p in self.pk:
            _free(p)
        for p in self.pv:
            _free(p)
        for p in self.gk:
            _free(p)
        for p in self.gv:
            _free(p)
        _free(self.buf_x)
        _free(self.buf_h)
        _free(self.buf_q)
        _free(self.buf_k)
        _free(self.buf_v)
        _free(self.buf_a)
        _free(self.buf_f)
        _free(self.buf_s)
        self.pool.stop()

    def device(self) raises -> String:
        if self.gpu:
            return self.gpu.value().name()
        return String("CPU")

    def _w(self, name: String) raises -> Int:
        return self.weights.weight(name).fptr()

    def _linear(mut self, name: String, x: Int, m: Int, k: Int, n: Int, dst: Int) raises:
        """dst[m, n] = x[m, k] @ W^T + b, with W stored [n, k] (torch layout)."""
        var w = self.weights.weight(name + ".weight")
        var has_b = self.weights.has(name + ".bias")
        var b = 0
        if has_b:
            b = self.weights.weight(name + ".bias").fptr()
        if self.gpu:
            var found = self.keys.get(name)
            var key = len(self.keys)
            if found:
                key = found.value()
            else:
                self.keys[name] = key
            var widx = self.gpu.value().weight(key * 2, w.fptr(), n, k, True)
            self.gpu.value().matmul(x, widx, dst, b, key * 2 + 1, m, k, n)
            return
        matmul_2d(self.pool, x, w.fptr(), dst, b, m, k, n, True)

    def _reserve(mut self, prefix: Int, steps: Int, beams: Int) raises:
        if prefix + steps > self.max_pos:
            raise Error("OPT: " + String(prefix + steps) + " positions exceed " + String(self.max_pos))
        var rows = max(prefix, beams)
        if prefix > self.prefix_cap or steps > self.steps_cap or beams > self.beams_cap:
            for p in self.pk:
                _free(p)
            for p in self.pv:
                _free(p)
            for p in self.gk:
                _free(p)
            for p in self.gv:
                _free(p)
            self.pk = List[Int]()
            self.pv = List[Int]()
            self.gk = List[Int]()
            self.gv = List[Int]()
            self.prefix_cap = max(prefix, self.prefix_cap)
            self.steps_cap = max(steps, self.steps_cap)
            self.beams_cap = max(beams, self.beams_cap)
            if self.gpu:
                self._dev_caches()
            else:
                for _ in range(self.layers):
                    self.pk.append(_alloc_f32(self.prefix_cap * self.hidden))
                    self.pv.append(_alloc_f32(self.prefix_cap * self.hidden))
                    self.gk.append(_alloc_f32(self.steps_cap * self.beams_cap * self.hidden))
                    self.gv.append(_alloc_f32(self.steps_cap * self.beams_cap * self.hidden))
        if rows > self.rows_cap or self.buf_s == 0:
            _free(self.buf_x)
            _free(self.buf_h)
            _free(self.buf_q)
            _free(self.buf_k)
            _free(self.buf_v)
            _free(self.buf_a)
            _free(self.buf_f)
            _free(self.buf_s)
            self.rows_cap = rows
            self.buf_x = _alloc_f32(rows * self.hidden)
            self.buf_h = _alloc_f32(rows * self.hidden)
            self.buf_q = _alloc_f32(rows * self.hidden)
            self.buf_k = _alloc_f32(rows * self.hidden)
            self.buf_v = _alloc_f32(rows * self.hidden)
            self.buf_a = _alloc_f32(rows * self.hidden)
            self.buf_f = _alloc_f32(rows * self.ffn)
        _free(self.buf_s)
        self.buf_s = _alloc_f32(rows * self.heads * (self.prefix_cap + self.steps_cap))
        if self.gpu and rows > self.d_rows:
            self._dev_buffers(rows)

    def _block_stack(mut self, rows: Int, prefix_pass: Bool, step: Int, anc: Int) raises:
        """Runs every layer on buf_x [rows, hidden] in place. The prefix pass writes the prefix
        cache; a generation step writes generated slot `step` for each beam row."""
        var hid = self.hidden
        for l in range(self.layers):
            var pre = "model.decoder.layers." + String(l)
            var src = self.buf_x
            if not self.post_ln:
                layer_norm_rows(self.buf_x, self.buf_h, rows, hid, self._w(pre + ".self_attn_layer_norm.weight"),
                                self._w(pre + ".self_attn_layer_norm.bias"), 1e-5)
                src = self.buf_h
            self._linear(pre + ".self_attn.q_proj", src, rows, hid, hid, self.buf_q)
            var kdst = self.pk[l] if prefix_pass else self.buf_k
            var vdst = self.pv[l] if prefix_pass else self.buf_v
            self._linear(pre + ".self_attn.k_proj", src, rows, hid, hid, kdst)
            self._linear(pre + ".self_attn.v_proj", src, rows, hid, hid, vdst)
            if not prefix_pass:
                for b in range(rows):
                    _ = external_call["memcpy", Int](self.gk[l] + ((step * self.beams_cap + b) * hid) * 4,
                                                     self.buf_k + b * hid * 4, hid * 4)
                    _ = external_call["memcpy", Int](self.gv[l] + ((step * self.beams_cap + b) * hid) * 4,
                                                     self.buf_v + b * hid * 4, hid * 4)
            # q scaling (head_dim ** -0.5) is applied to q_proj's output, bias included
            var pq = fptr_at(self.buf_q)
            var scale = Float32(1.0 / sqrt(Float64(hid // self.heads)))
            for i in range(rows * hid):
                pq.unsafe_offset(i).unsafe_store(pq.unsafe_offset(i).unsafe_load() * scale)
            var ctx = _AttnCtx()
            ctx.q = self.buf_q
            ctx.out = self.buf_a
            ctx.pk = self.pk[l]
            ctx.pv = self.pv[l]
            ctx.gk = self.gk[l]
            ctx.gv = self.gv[l]
            ctx.anc = anc
            ctx.prefix_len = self.prefix_len
            ctx.steps = 0 if prefix_pass else step + 1
            ctx.beams_cap = self.beams_cap
            ctx.hidden = hid
            ctx.heads = self.heads
            ctx.head_dim = hid // self.heads
            ctx.causal_prefix = prefix_pass
            ctx.scratch = self.buf_s
            ctx.scratch_stride = self.prefix_cap + self.steps_cap
            run_parallel_ctx(self.pool, rows * self.heads, ctx, _attn_worker)
            self._linear(pre + ".self_attn.out_proj", self.buf_a, rows, hid, hid, self.buf_h)
            _axpy(self.buf_x, 1.0, self.buf_h, rows * hid)
            if self.post_ln:
                # in place: each row's statistics are taken before it is rewritten
                layer_norm_rows(self.buf_x, self.buf_x, rows, hid, self._w(pre + ".self_attn_layer_norm.weight"),
                                self._w(pre + ".self_attn_layer_norm.bias"), 1e-5)
                self._linear(pre + ".fc1", self.buf_x, rows, hid, self.ffn, self.buf_f)
            else:
                layer_norm_rows(self.buf_x, self.buf_h, rows, hid, self._w(pre + ".final_layer_norm.weight"),
                                self._w(pre + ".final_layer_norm.bias"), 1e-5)
                self._linear(pre + ".fc1", self.buf_h, rows, hid, self.ffn, self.buf_f)
            var pf = fptr_at(self.buf_f)
            for i in range(rows * self.ffn):
                var v = pf.unsafe_offset(i).unsafe_load()
                if v < 0:
                    pf.unsafe_offset(i).unsafe_store(0)
            self._linear(pre + ".fc2", self.buf_f, rows, self.ffn, hid, self.buf_h)
            _axpy(self.buf_x, 1.0, self.buf_h, rows * hid)
            if self.post_ln:
                layer_norm_rows(self.buf_x, self.buf_x, rows, hid, self._w(pre + ".final_layer_norm.weight"),
                                self._w(pre + ".final_layer_norm.bias"), 1e-5)

    def _logits(mut self, rows: Int, row0: Int, dst: Int) raises:
        """Final LayerNorm + LM head for rows [row0, row0 + rows) of buf_x into dst [rows, vocab]."""
        var hid = self.hidden
        if self.post_ln:
            self._linear("lm_head", self.buf_x + row0 * hid * 4, rows, hid, self.vocab, dst)
            return
        layer_norm_rows(self.buf_x + row0 * hid * 4, self.buf_h, rows, hid, self._w("model.decoder.final_layer_norm.weight"),
                        self._w("model.decoder.final_layer_norm.bias"), 1e-5)
        self._linear("lm_head", self.buf_h, rows, hid, self.vocab, dst)

    def _add_positions(self, rows: Int, pos0: Int, same_pos: Bool) raises:
        var pos = self.weights.weight("model.decoder.embed_positions.weight").fptr()
        for r in range(rows):
            var p = pos0 if same_pos else pos0 + r
            _axpy(self.buf_x + r * self.hidden * 4, 1.0, pos + (p + 2) * self.hidden * 4, self.hidden)

    def start(mut self, prefix: Tensor, steps_cap: Int, beams: Int) raises -> List[Float32]:
        """Runs the shared prefix ([L, hidden] input embeddings) and returns the logits of its
        last position (the first generated token's distribution)."""
        var L = prefix.shape[0]
        self._reserve(L, steps_cap, beams)
        self.prefix_len = L
        _ = external_call["memcpy", Int](self.buf_x, prefix.fptr(), L * self.hidden * 4)
        self._add_positions(L, 0, False)
        if self.gpu:
            return self._dev_run(L, True, 0, List[Int32](), L - 1, 1)
        var dummy = List[Int32](length=1, fill=0)
        self._block_stack(L, True, 0, Int(dummy.unsafe_ptr()))
        var logits = List[Float32](length=self.vocab, fill=0.0)
        self._logits(1, L - 1, Int(logits.unsafe_ptr()))
        return logits^

    def step(mut self, tokens: List[Int], step: Int, ancestry: List[Int32]) raises -> List[Float32]:
        """Feeds token `tokens[b]` to every beam b as generated position `step` (0 based) and
        returns logits [beams, vocab]. `ancestry` is [beams, step + 1]: for beam b, the slot beam
        holding each of its generated positions 0..step (its own row b at `step`)."""
        return self.step_extra(tokens, step, ancestry, List[Float32]())

    def step_extra(mut self, tokens: List[Int], step: Int, ancestry: List[Int32], extra: List[Float32]) raises -> List[Float32]:
        """`step` with extra input embeddings [beams, hidden] added to each beam's token embedding
        (models such as MagicArticulate add learned per-slot embeddings); empty `extra` adds none."""
        var B = len(tokens)
        var hid = self.hidden
        var emb = self.weights.weight("model.decoder.embed_tokens.weight").fptr()
        for b in range(B):
            _ = external_call["memcpy", Int](self.buf_x + b * hid * 4, emb + tokens[b] * hid * 4, hid * 4)
        if len(extra) >= B * hid:
            _axpy(self.buf_x, 1.0, Int(extra.unsafe_ptr()), B * hid)
        _ = len(extra)
        self._add_positions(B, self.prefix_len + step, True)
        if self.gpu:
            return self._dev_run(B, False, step, ancestry, 0, B)
        self._block_stack(B, False, step, Int(ancestry.unsafe_ptr()))
        var logits = List[Float32](length=B * self.vocab, fill=0.0)
        self._logits(B, 0, Int(logits.unsafe_ptr()))
        return logits^

    # ------------------------------------------------------------------ device path

    def _dev_caches(mut self) raises:
        ref g = self.gpu.value()
        for sl in self.d_pk:
            g.release(sl)
        for sl in self.d_pv:
            g.release(sl)
        for sl in self.d_gk:
            g.release(sl)
        for sl in self.d_gv:
            g.release(sl)
        self.d_pk = List[Int]()
        self.d_pv = List[Int]()
        self.d_gk = List[Int]()
        self.d_gv = List[Int]()
        for _ in range(self.layers):
            self.d_pk.append(g.alloc(self.prefix_cap * self.hidden))
            self.d_pv.append(g.alloc(self.prefix_cap * self.hidden))
            self.d_gk.append(g.alloc(self.steps_cap * self.beams_cap * self.hidden))
            self.d_gv.append(g.alloc(self.steps_cap * self.beams_cap * self.hidden))
        if self.d_anc >= 0:
            g.release(self.d_anc)
        self.d_anc = g.alloc(self.beams_cap * (self.steps_cap + 1))

    def _dev_buffers(mut self, rows: Int) raises:
        ref g = self.gpu.value()
        for sl in [self.d_x, self.d_h, self.d_q, self.d_a, self.d_f, self.d_logit]:
            g.release(sl)
        self.d_rows = rows
        self.d_x = g.alloc(rows * self.hidden)
        self.d_h = g.alloc(rows * self.hidden)
        self.d_q = g.alloc(rows * self.hidden)
        self.d_a = g.alloc(rows * self.hidden)
        self.d_f = g.alloc(rows * self.ffn)
        self.d_logit = g.alloc(rows * self.vocab)

    def _key(mut self, name: String) -> Int:
        var found = self.keys.get(name)
        if found:
            return found.value()
        var key = len(self.keys)
        self.keys[name] = key
        return key

    def _dvec(mut self, name: String, n: Int) raises -> FP:
        var key = self._key(name)
        var addr = self._w(name)
        return self.gpu.value().wptr(self.gpu.value().weight(key * 2, addr, 1, n, False))

    def _dlinear(mut self, name: String, x: FP, m: Int, k: Int, n: Int, dst: FP) raises:
        var w = self.weights.weight(name + ".weight")
        var b = 0
        if self.weights.has(name + ".bias"):
            b = self.weights.weight(name + ".bias").fptr()
        var key = self._key(name)
        var widx = self.gpu.value().weight(key * 2, w.fptr(), n, k, True)
        self.gpu.value().matmul_dev(x, widx, dst, b, key * 2 + 1, m, k, n)

    def _dev_run(mut self, rows: Int, prefix_pass: Bool, step: Int, ancestry: List[Int32], row0: Int,
                 out_rows: Int) raises -> List[Float32]:
        """Uploads buf_x [rows, hidden] (and the ancestry), runs every layer on the device and
        returns the logits of rows [row0, row0 + out_rows)."""
        var hid = self.hidden
        ref g = self.gpu.value()
        g.upload(self.d_x, self.buf_x, rows * hid)
        var stride = step + 1
        if not prefix_pass:
            var af = List[Float32](length=rows * stride, fill=0.0)
            for i in range(rows * stride):
                af[i] = Float32(ancestry[i])
            g.upload(self.d_anc, Int(af.unsafe_ptr()), rows * stride)
            _ = len(af)  # keep the list alive through the copy (ASAP destruction)
        var px = g.ptr(self.d_x)
        var ph = g.ptr(self.d_h)
        var pq = g.ptr(self.d_q)
        var pa = g.ptr(self.d_a)
        var pf = g.ptr(self.d_f)
        var scale = Float32(1.0 / sqrt(Float64(hid // self.heads)))
        for l in range(self.layers):
            var pre = "model.decoder.layers." + String(l)
            var g1 = self._dvec(pre + ".self_attn_layer_norm.weight", hid)
            var b1 = self._dvec(pre + ".self_attn_layer_norm.bias", hid)
            var src = px    # post-LN: attention reads the residual stream itself
            if not self.post_ln:
                self.gpu.value().layer_norm_dev(px, g1, b1, ph, rows, hid, 1e-5, True)
                src = ph
            self._dlinear(pre + ".self_attn.q_proj", src, rows, hid, hid, pq)
            var pk = self.gpu.value().ptr(self.d_pk[l])
            var pv = self.gpu.value().ptr(self.d_pv[l])
            var gk = self.gpu.value().ptr(self.d_gk[l])
            var gv = self.gpu.value().ptr(self.d_gv[l])
            if prefix_pass:
                self._dlinear(pre + ".self_attn.k_proj", src, rows, hid, hid, pk)
                self._dlinear(pre + ".self_attn.v_proj", src, rows, hid, hid, pv)
            else:
                var off = step * self.beams_cap * hid
                self._dlinear(pre + ".self_attn.k_proj", src, rows, hid, hid, gk.unsafe_offset(off))
                self._dlinear(pre + ".self_attn.v_proj", src, rows, hid, hid, gv.unsafe_offset(off))
            var ok = self.gpu.value().decoder_attention(
                pq, pa, pk, pv, gk, gv, self.gpu.value().ptr(self.d_anc), rows, self.prefix_len, prefix_pass,
                0 if prefix_pass else step + 1, self.beams_cap, hid, self.heads, scale, stride)
            if not ok:
                raise Error("OPT: GPU attention needs head dim 64 and at most 2048 positions")
            self._dlinear(pre + ".self_attn.out_proj", pa, rows, hid, hid, ph)
            self.gpu.value().binary_dev(GB_ADD, px, ph, px, rows * hid, 1, 0, 0, False)
            var g2 = self._dvec(pre + ".final_layer_norm.weight", hid)
            var b2 = self._dvec(pre + ".final_layer_norm.bias", hid)
            if self.post_ln:
                # x = LN(x) through scratch: LN into ph, then copy back into the stream
                self.gpu.value().layer_norm_dev(px, g1, b1, ph, rows, hid, 1e-5, True)
                self.gpu.value().copy_dev(ph, px, 1, rows * hid, 0, 0, 0, 0)
                self._dlinear(pre + ".fc1", px, rows, hid, self.ffn, pf)
            else:
                self.gpu.value().layer_norm_dev(px, g2, b2, ph, rows, hid, 1e-5, True)
                self._dlinear(pre + ".fc1", ph, rows, hid, self.ffn, pf)
            self.gpu.value().unary_dev(GU_RELU, pf, pf, rows * self.ffn)
            self._dlinear(pre + ".fc2", pf, rows, self.ffn, hid, ph)
            self.gpu.value().binary_dev(GB_ADD, px, ph, px, rows * hid, 1, 0, 0, False)
            if self.post_ln:
                self.gpu.value().layer_norm_dev(px, g2, b2, ph, rows, hid, 1e-5, True)
                self.gpu.value().copy_dev(ph, px, 1, rows * hid, 0, 0, 0, 0)
        if self.post_ln:
            self._dlinear("lm_head", px.unsafe_offset(row0 * hid), out_rows, hid, self.vocab, self.gpu.value().ptr(self.d_logit))
            var lg = List[Float32](length=out_rows * self.vocab, fill=0.0)
            self.gpu.value().download(self.d_logit, Int(lg.unsafe_ptr()), out_rows * self.vocab)
            return lg^
        var gf = self._dvec("model.decoder.final_layer_norm.weight", hid)
        var bf = self._dvec("model.decoder.final_layer_norm.bias", hid)
        self.gpu.value().layer_norm_dev(px.unsafe_offset(row0 * hid), gf, bf, ph, out_rows, hid, 1e-5, True)
        self._dlinear("lm_head", ph, out_rows, hid, self.vocab, self.gpu.value().ptr(self.d_logit))
        var logits = List[Float32](length=out_rows * self.vocab, fill=0.0)
        self.gpu.value().download(self.d_logit, Int(logits.unsafe_ptr()), out_rows * self.vocab)
        return logits^

    def embed(self, token: Int) raises -> List[Float32]:
        var emb = fptr_at(self.weights.weight("model.decoder.embed_tokens.weight").fptr())
        var out = List[Float32](length=self.hidden, fill=0.0)
        for i in range(self.hidden):
            out[i] = emb.unsafe_offset(token * self.hidden + i).unsafe_load[alignment=1]()
        return out^
