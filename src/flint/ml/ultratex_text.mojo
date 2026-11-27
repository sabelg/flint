"""Native Qwen3-8B text conditioning for UltraTex's FLUX.2 Klein checkpoint."""
from std.math import cos, exp, sin, sqrt
from flint.assets.json import JsonDocument, parse_json
from flint.core.jobs import JobPool, default_worker_count
from flint.ml.safetensors import SafeTensorWeights
from flint.ml.tensor import Tensor
from flint.ml.ultratex_gpu import UltraTexGpu
from flint.ml.vision_ops import weighted_linear
from flint.platform.os import read_bytes

comptime WIDTH = 4096
comptime HEAD = 128
comptime SEQ = 512


def _bf(v: Float32) -> Float32:
    return Float32(BFloat16(v))


def _norm(x: Tensor, w: Tensor, span: Int) -> Tensor:
    var out = Tensor.zeros(x.shape.copy())
    for p in range(0, len(x.f), span):
        var ss = Float32(0)
        for j in range(span):
            ss += x.f[p + j] * x.f[p + j]
        var inv = 1 / sqrt(ss / Float32(span) + 1e-6)
        for j in range(span):
            out.f[p + j] = _bf(_bf(x.f[p + j] * inv) * w.f[j])
    return out^


def _rotate(mut x: Tensor, cs: Tensor, sn: Tensor):
    var width = x.shape[1]
    for i in range(x.shape[0]):
        for h in range(width // HEAD):
            for j in range(HEAD // 2):
                var a = i * width + h * HEAD + j
                var b = a + HEAD // 2
                var c = cs.f[i * (HEAD // 2) + j]
                var s = sn.f[i * (HEAD // 2) + j]
                var u = x.f[a]
                var v = x.f[b]
                x.f[a] = _bf(_bf(u * c) - _bf(v * s))
                x.f[b] = _bf(_bf(v * c) + _bf(u * s))


def _attend(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    var out = Tensor.zeros(q.shape.copy())
    var length = k.shape[0]
    var scores = List[Float32](length=length, fill=0)
    for r in range(SEQ):
        var count = min(r + 1, length)
        for head in range(32):
            var peak = -Float32.MAX
            var src = r * WIDTH + head * HEAD
            for p in range(count):
                var start = (p * 8 + head // 4) * HEAD
                var dot = Float32(0)
                for j in range(HEAD):
                    dot += q.f[src + j] * k.f[start + j]
                scores[p] = dot / sqrt(Float32(HEAD))
                peak = max(peak, scores[p])
            var total = Float32(0)
            for p in range(count):
                scores[p] = exp(scores[p] - peak)
                total += scores[p]
            for p in range(count):
                var prob = scores[p] / total
                var start = (p * 8 + head // 4) * HEAD
                for j in range(HEAD):
                    out.f[src + j] += prob * v.f[start + j]
            for j in range(HEAD):
                out.f[src + j] = _bf(out.f[src + j])
    return out^


struct _Qwen(Movable):
    var root: String
    var index: JsonDocument
    var map_node: Int
    var open_name: String
    var shard: List[SafeTensorWeights]
    var gpu: List[UltraTexGpu]
    var pool: JobPool

    def __init__(out self, root: String, use_gpu: Bool) raises:
        self.root = root
        self.index = parse_json(read_bytes(root + "/model.safetensors.index.json"))
        self.map_node = self.index.object_get(self.index.root, "weight_map")
        if not self.index.valid or self.map_node < 0:
            raise Error("Missing Qwen3-8B shard index")
        self.open_name = ""
        self.shard = List[SafeTensorWeights]()
        self.gpu = List[UltraTexGpu]()
        if use_gpu:
            self.gpu.append(UltraTexGpu())
        self.pool = JobPool()
        _ = self.pool.start(default_worker_count())

    def select(mut self, key: String) raises:
        var node = self.index.object_get(self.map_node, key)
        var name = self.index.text(node)
        if name.byte_length() == 0 or "/" in name or ".." in name:
            raise Error("Invalid Qwen3 shard entry for " + key)
        if self.open_name != name:
            self.shard.clear()
            self.shard.append(SafeTensorWeights(self.root + "/" + name, 5 << 30))
            self.open_name = name

    def tensor(mut self, key: String, var shape: List[Int]) raises -> Tensor:
        self.select(key)
        return self.shard[0].tensor(key, shape^)

    def linear(mut self, x: Tensor, key: String, co: Int) raises -> Tensor:
        self.select(key + ".weight")
        var out = self.gpu[0].linear(self.shard[0], x, key, co) if len(self.gpu) else weighted_linear(
            self.shard[0], self.pool, x, key, co, False
        )
        for i in range(len(out.f)):
            out.f[i] = _bf(out.f[i])
        return out^


def ultratex_text(root: String, tokens: List[Int], use_gpu: Bool = False) raises -> Tensor:
    """Encode up to 512 tokenizer IDs, including chat framing, using right padding."""
    if len(tokens) < 1 or len(tokens) > SEQ:
        raise Error("Expected 1 to 512 Qwen3 chat-template token IDs")
    for id in tokens:
        if id < 0 or id >= 151936:
            raise Error("Qwen3 vocabulary ID out of range")
    var qwen = _Qwen(root, use_gpu)
    var table = qwen.tensor("model.embed_tokens.weight", [151936, WIDTH])
    var x = Tensor.zeros([SEQ, WIDTH])
    for i in range(SEQ):
        var id = tokens[i] if i < len(tokens) else 151643
        for j in range(WIDTH):
            x.f[i * WIDTH + j] = table.f[id * WIDTH + j]
    var cs = Tensor.zeros([SEQ, HEAD // 2])
    var sn = Tensor.zeros([SEQ, HEAD // 2])
    for r in range(SEQ):
        for j in range(HEAD // 2):
            var phase = Float32(r) * exp(-Float32(13.815510558) * Float32(2 * j) / Float32(HEAD))
            cs.f[r * (HEAD // 2) + j] = _bf(cos(phase))
            sn.f[r * (HEAD // 2) + j] = _bf(sin(phase))
    var ctx = Tensor.zeros([SEQ, 3 * WIDTH])
    for layer in range(27):
        var key = "model.layers." + String(layer) + "."
        var h = _norm(x, qwen.tensor(key + "input_layernorm.weight", [WIDTH]), WIDTH)
        var prefix = Tensor.zeros([len(tokens), WIDTH])
        for i in range(len(prefix.f)):
            prefix.f[i] = h.f[i]
        var q = _norm(
            qwen.linear(h, key + "self_attn.q_proj", WIDTH),
            qwen.tensor(key + "self_attn.q_norm.weight", [HEAD]),
            HEAD,
        )
        var k = _norm(
            qwen.linear(prefix, key + "self_attn.k_proj", 1024),
            qwen.tensor(key + "self_attn.k_norm.weight", [HEAD]),
            HEAD,
        )
        var v = qwen.linear(prefix, key + "self_attn.v_proj", 1024)
        _rotate(q, cs, sn)
        _rotate(k, cs, sn)
        var a = qwen.linear(_attend(q, k, v), key + "self_attn.o_proj", WIDTH)
        for i in range(len(x.f)):
            x.f[i] = _bf(x.f[i] + a.f[i])
        h = _norm(x, qwen.tensor(key + "post_attention_layernorm.weight", [WIDTH]), WIDTH)
        var gate = qwen.linear(h, key + "mlp.gate_proj", 12288)
        var up = qwen.linear(h, key + "mlp.up_proj", 12288)
        for i in range(len(gate.f)):
            var g = gate.f[i]
            gate.f[i] = _bf(_bf(g / (1 + exp(-g))) * up.f[i])
        var ff = qwen.linear(gate, key + "mlp.down_proj", WIDTH)
        for i in range(len(x.f)):
            x.f[i] = _bf(x.f[i] + ff.f[i])
        if (layer + 1) % 9 == 0:
            var offset = ((layer + 1) // 9 - 1) * WIDTH
            for r in range(SEQ):
                for j in range(WIDTH):
                    ctx.f[r * 3 * WIDTH + offset + j] = x.f[r * WIDTH + j]
    return ctx^


def ultratex_empty_tokens() -> List[Int]:
    """Qwen3 tokenizer's empty user message with generation prompt and thinking disabled."""
    return [151644, 872, 198, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
