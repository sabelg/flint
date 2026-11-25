"""SigLIP / SigLIP 2 towers and the sigmoid logit head (`flint.ml.siglip.model`).

Hugging Face `SiglipModel` layout (the fixed-resolution SigLIP 2 checkpoints such as
`google/siglip2-base-patch16-224` and `google/siglip2-so400m-patch14-224` use it too):

  * `SiglipVision`: patch convolution (with bias, stride = patch), learned positions (no class
    token), pre-LN encoder layers with `gelu_pytorch_tanh` MLPs, `post_layernorm` on every
    token, then the multi-head attention pooling head (a learned probe attends over the tokens,
    `out_proj`, LayerNorm + MLP residual) -> the image embedding.
  * `SiglipText`: token + position embeddings over a fixed 64-token sequence (Gemma tokenizer for
    SigLIP 2, `<eos>` then `<pad>` up to 64), the same encoder without a causal mask,
    `final_layer_norm`, the *last* position (padding included, as in the reference) through
    `head` -> the text embedding.
  * `logit_scale`, `logit_bias`: `logits[t, i] = exp(scale) * <t, i> + bias` on L2-normalised
    embeddings; `sigmoid(logits)` are independent match probabilities.

The encoder layers, LayerNorms, attention and GEMMs are `flint.ml.diffusion`'s CLIP layers
(`clip._load_layers` / `clip._layers_forward`) on its `Runtime`, so everything runs on the CPU
(SIMD + JobPool) or a MAX GPU. Images and texts are batched. The 256k x D token table is not
copied: rows are read from the memory-mapped safetensors file.
"""

from std.math import sqrt, exp
from flint.platform.os import read_bytes
from flint.assets.json import parse_json
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.safetensors import SafeTensors
from flint.ml.diffusion.nn import Runtime, Act, Linear, Conv, Norm
from flint.ml.diffusion.kernels import ACT_GELU, ACT_GELU_TANH, ACT_QUICK_GELU
from flint.ml.diffusion.clip import ClipLayer, _load_layers, _layers_forward
from flint.ml.sentencepiece import SentencePiece


def _cfg_section_int(cfg: List[UInt8], section: String, key: String, fallback: Int) -> Int:
    if len(cfg) == 0:
        return fallback
    var doc = parse_json(cfg)
    var s = doc.object_get(doc.root, section)
    if s < 0:
        return fallback
    return doc.integer(doc.object_get(s, key), fallback)


def _cfg_section_float(cfg: List[UInt8], section: String, key: String, fallback: Float64) -> Float64:
    if len(cfg) == 0:
        return fallback
    var doc = parse_json(cfg)
    var s = doc.object_get(doc.root, section)
    if s < 0:
        return fallback
    return doc.number(doc.object_get(s, key), fallback)


def _cfg_act(cfg: List[UInt8], section: String) -> Int:
    if len(cfg) == 0:
        return ACT_GELU_TANH
    var doc = parse_json(cfg)
    var s = doc.object_get(doc.root, section)
    var a = doc.text(doc.object_get(s, "hidden_act"), "gelu_pytorch_tanh")
    if a == "gelu":
        return ACT_GELU
    if a == "quick_gelu":
        return ACT_QUICK_GELU
    return ACT_GELU_TANH


def l2_normalize(mut v: List[Float32], dim: Int):
    """Normalises each `dim`-long row of `v` in place."""
    var rows = len(v) // dim
    for r in range(rows):
        var s: Float64 = 0
        for d in range(dim):
            s += Float64(v[r * dim + d]) * Float64(v[r * dim + d])
        var inv = Float32(1.0 / max(sqrt(s), 1e-12))
        for d in range(dim):
            v[r * dim + d] *= inv


struct SiglipVision(Movable):
    var patch: Conv
    var pos: List[Float32]
    var width: Int
    var patch_size: Int
    var image_size: Int
    var grid: Int
    var layers: List[ClipLayer]
    var post_ln: Norm
    var probe_q: List[Float32]      # q_proj(probe), the same for every image
    var head_k: Linear
    var head_v: Linear
    var head_o: Linear
    var head_ln: Norm
    var head_fc1: Linear
    var head_fc2: Linear
    var heads: Int
    var act: Int

    def __init__(out self):
        self.patch = Conv()
        self.pos = List[Float32]()
        self.width = 0
        self.patch_size = 16
        self.image_size = 224
        self.grid = 14
        self.layers = List[ClipLayer]()
        self.post_ln = Norm()
        self.probe_q = List[Float32]()
        self.head_k = Linear()
        self.head_v = Linear()
        self.head_o = Linear()
        self.head_ln = Norm()
        self.head_fc1 = Linear()
        self.head_fc2 = Linear()
        self.heads = 12
        self.act = ACT_GELU_TANH

    @staticmethod
    def load(mut rt: Runtime, st: SafeTensors, cfg: List[UInt8]) raises -> SiglipVision:
        var v = SiglipVision()
        var eps = Float32(_cfg_section_float(cfg, "vision_config", "layer_norm_eps", 1e-6))
        var pw = st.weight("vision_model.embeddings.patch_embedding.weight")
        v.patch_size = pw.shape[2]
        v.width = pw.shape[0]
        v.patch = rt.conv_from(pw, st.weight("vision_model.embeddings.patch_embedding.bias"), True, v.patch_size,
                               0, 0, 0, 0)
        v.pos = st.weight("vision_model.embeddings.position_embedding.weight").owned_floats()
        var npos = st.shape("vision_model.embeddings.position_embedding.weight")[0]
        v.grid = 1
        while (v.grid + 1) * (v.grid + 1) <= npos:
            v.grid += 1
        if v.grid * v.grid != npos:
            raise Error("siglip: " + String(npos) + " vision positions is not a square grid")
        v.image_size = v.grid * v.patch_size
        v.layers = _load_layers(rt, st, "vision_model", eps)
        v.post_ln = rt.norm(st, "vision_model.post_layernorm", 1, eps)
        v.heads = _cfg_section_int(cfg, "vision_config", "num_attention_heads", max(1, v.width // 64))
        v.act = _cfg_act(cfg, "vision_config")
        # attention-pooling head: nn.MultiheadAttention packs q / k / v in one [3D, D] matrix
        var D = v.width
        var w = st.weight("vision_model.head.attention.in_proj_weight")
        var b = st.weight("vision_model.head.attention.in_proj_bias")
        var probe = st.weight("vision_model.head.probe").owned_floats()
        var bq = b.owned_floats()
        var q = List[Float32](length=D, fill=0.0)
        var pwq = fptr_at(w.fptr())
        for r in range(D):
            var acc: Float64 = Float64(bq[r])
            for c in range(D):
                acc += Float64(pwq.unsafe_offset(r * D + c).unsafe_load[alignment=1]()) * Float64(probe[c])
            q[r] = Float32(acc)
        v.probe_q = q^
        v.head_k = rt.linear_from(Tensor.borrowed([D, D], w.fptr() + D * D * 4),
                                  Tensor.borrowed([D], b.fptr() + D * 4), True)
        v.head_v = rt.linear_from(Tensor.borrowed([D, D], w.fptr() + 2 * D * D * 4),
                                  Tensor.borrowed([D], b.fptr() + 2 * D * 4), True)
        v.head_o = rt.linear(st, "vision_model.head.attention.out_proj")
        v.head_ln = rt.norm(st, "vision_model.head.layernorm", 1, eps)
        v.head_fc1 = rt.linear(st, "vision_model.head.mlp.fc1")
        v.head_fc2 = rt.linear(st, "vision_model.head.mlp.fc2")
        _ = w^
        _ = b^
        return v^

    def encode(mut self, mut rt: Runtime, pixels: List[Float32], batch: Int,
               keep_hidden: Bool = False) raises -> Tuple[List[Float32], List[Float32]]:
        """`pixels`: `batch` normalised channels-last [S, S, 3] images back to back.
        Returns (pooled [batch, D], last hidden state [batch, T, D] if `keep_hidden`)."""
        var S = self.image_size
        var D = self.width
        var T = self.grid * self.grid
        var x = rt.upload(batch, S, S, 3, pixels)
        var p = rt.conv2d(x, self.patch)
        rt.free(x^)
        var h = rt.reshape(p^, batch, T, 1, D)
        var pos = List[Float32](capacity=batch * T * D)
        for _ in range(batch):
            pos.extend(self.pos.copy())
        var pa = rt.upload(batch, T, 1, D, pos)
        rt.add_(h, pa)
        rt.free(pa^)
        h = _layers_forward(rt, self.layers, h^, self.heads, False, self.act)
        var hn = rt.layer_norm(h, self.post_ln)
        rt.free(h^)
        var hidden = List[Float32]()
        if keep_hidden:
            hidden = rt.download(hn)
        # attention pooling
        var qh = List[Float32](capacity=batch * D)
        for _ in range(batch):
            qh.extend(self.probe_q.copy())
        var q = rt.upload(batch, 1, 1, D, qh)
        var k = rt.linear(hn, self.head_k)
        var v = rt.linear(hn, self.head_v)
        rt.free(hn^)
        var a = rt.attention(q, k, v, self.heads, False)
        rt.free(q^)
        rt.free(k^)
        rt.free(v^)
        var o = rt.linear(a, self.head_o)
        rt.free(a^)
        var n = rt.layer_norm(o, self.head_ln)
        var f = rt.linear(n, self.head_fc1)
        rt.free(n^)
        rt.activate(f, self.act)
        var f2 = rt.linear(f, self.head_fc2)
        rt.free(f^)
        rt.add_(o, f2)
        rt.free(f2^)
        var pooled = rt.download(o)
        rt.free(o^)
        return (pooled^, hidden^)


struct SiglipText(Movable):
    var st: SafeTensors           # keeps the mapped token table alive
    var tok_addr: Int
    var vocab: Int
    var pos: List[Float32]
    var width: Int
    var max_len: Int
    var layers: List[ClipLayer]
    var final_ln: Norm
    var head: Linear
    var heads: Int
    var act: Int
    var host_table: List[Float32]  # only used when the table is not float32 on disk

    def __init__(out self, var st: SafeTensors):
        self.st = st^
        self.tok_addr = 0
        self.vocab = 0
        self.pos = List[Float32]()
        self.width = 0
        self.max_len = 64
        self.layers = List[ClipLayer]()
        self.final_ln = Norm()
        self.head = Linear()
        self.heads = 12
        self.act = ACT_GELU_TANH
        self.host_table = List[Float32]()

    @staticmethod
    def load(mut rt: Runtime, path: String, cfg: List[UInt8]) raises -> SiglipText:
        var t = SiglipText(SafeTensors.open(path))
        var eps = Float32(_cfg_section_float(cfg, "text_config", "layer_norm_eps", 1e-6))
        var shp = t.st.shape("text_model.embeddings.token_embedding.weight")
        t.vocab = shp[0]
        t.width = shp[1]
        var tw = t.st.weight("text_model.embeddings.token_embedding.weight")
        if tw.ext == 0:
            t.host_table = tw.owned_floats()
            t.tok_addr = Int(t.host_table.unsafe_ptr())
        else:
            t.tok_addr = tw.fptr()
        t.pos = t.st.weight("text_model.embeddings.position_embedding.weight").owned_floats()
        t.max_len = t.st.shape("text_model.embeddings.position_embedding.weight")[0]
        t.layers = _load_layers(rt, t.st, "text_model", eps)
        t.final_ln = rt.norm(t.st, "text_model.final_layer_norm", 1, eps)
        t.head = rt.linear(t.st, "text_model.head")
        t.heads = _cfg_section_int(cfg, "text_config", "num_attention_heads", max(1, t.width // 64))
        t.act = _cfg_act(cfg, "text_config")
        return t^

    def encode(mut self, mut rt: Runtime, ids: List[Int], batch: Int,
               keep_hidden: Bool = False) raises -> Tuple[List[Float32], List[Float32]]:
        """`ids`: `batch` rows of exactly `max_len` token ids. Returns (pooled [batch, D],
        final hidden states [batch, L, D] if `keep_hidden`)."""
        var L = self.max_len
        var D = self.width
        if len(ids) != batch * L:
            raise Error("siglip: text batch needs " + String(batch * L) + " ids, got " + String(len(ids)))
        var x = List[Float32](length=batch * L * D, fill=0.0)
        var tab = fptr_at(self.tok_addr)
        for r in range(batch * L):
            var id = ids[r]
            if id < 0 or id >= self.vocab:
                raise Error("siglip: token id " + String(id) + " out of range")
            var l = r % L
            for d in range(D):
                x[r * D + d] = tab.unsafe_offset(id * D + d).unsafe_load[alignment=1]() + self.pos[l * D + d]
        var h = rt.upload(batch, L, 1, D, x)
        h = _layers_forward(rt, self.layers, h^, self.heads, False, self.act)
        var hn = rt.layer_norm(h, self.final_ln)
        rt.free(h^)
        var hidden = rt.download(hn)
        rt.free(hn^)
        var last = List[Float32](length=batch * D, fill=0.0)
        for b in range(batch):
            for d in range(D):
                last[b * D + d] = hidden[(b * L + L - 1) * D + d]
        var la = rt.upload(batch, 1, 1, D, last)
        var pa = rt.linear(la, self.head)
        rt.free(la^)
        var pooled = rt.download(pa)
        rt.free(pa^)
        if not keep_hidden:
            hidden = List[Float32]()
        return (pooled^, hidden^)


struct SiglipModel(Movable):
    """A SigLIP / SigLIP 2 checkpoint folder: `model.safetensors`, `config.json`, tokenizer.

    `embed_images` / `embed_texts` return L2-normalised embeddings (rows of `dim`);
    `logits(text_emb, image_emb)` = exp(logit_scale) * cosine + logit_bias, [texts, images]."""
    var rt: Runtime
    var vision: SiglipVision
    var text: SiglipText
    var tokenizer: SentencePiece
    var logit_scale: Float32
    var logit_bias: Float32
    var dim: Int
    var name: String

    def __init__(out self, var rt: Runtime, var vision: SiglipVision, var text: SiglipText,
                 var tokenizer: SentencePiece, logit_scale: Float32, logit_bias: Float32, name: String):
        self.rt = rt^
        self.vision = vision^
        self.text = text^
        self.tokenizer = tokenizer^
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias
        self.dim = self.vision.width
        self.name = name

    @staticmethod
    def load(dir: String, use_gpu: Bool = True, vision: Bool = True, text: Bool = True) raises -> SiglipModel:
        """Loads `dir` (a Hugging Face SigLIP folder). `vision` / `text` = False skip a tower
        (e.g. a text-only query server over cached image embeddings)."""
        var rt = Runtime(use_gpu)
        rt.fused_attention = True
        var path = dir + "/model.safetensors"
        var cfg = read_bytes(dir + "/config.json")
        var st = SafeTensors.open(path)
        var vt = SiglipVision()
        if vision:
            vt = SiglipVision.load(rt, st, cfg)
        var tt = SiglipText(SafeTensors())
        if text:
            tt = SiglipText.load(rt, path, cfg)
        var tok = SentencePiece()
        if text:
            tok = SentencePiece.load_dir(dir)
        var scale = st.scalar("logit_scale")
        var bias = st.scalar("logit_bias")
        var name = dir
        var parts = dir.split("/")
        for i in range(len(parts)):
            if parts[i].byte_length() > 0:
                name = String(parts[i])
        var m = SiglipModel(rt^, vt^, tt^, tok^, scale, bias, name)
        if not vision:
            m.dim = m.text.head.n
        return m^

    def image_size(self) -> Int:
        return self.vision.image_size

    def token_ids(self, text: String) -> List[Int]:
        """The processor's ids: Gemma / SentencePiece tokens + `<eos>`, padded to 64."""
        return self.tokenizer.encode_fixed(text, self.text.max_len)

    def embed_texts(mut self, texts: List[String], normalize: Bool = True) raises -> List[Float32]:
        var ids = List[Int]()
        for i in range(len(texts)):
            ids.extend(self.token_ids(texts[i]))
        var r = self.text.encode(self.rt, ids, len(texts))
        var e = r[0].copy()
        if normalize:
            l2_normalize(e, self.text.head.n)
        return e^

    def embed_pixels(mut self, pixels: List[Float32], batch: Int, normalize: Bool = True) raises -> List[Float32]:
        """`batch` preprocessed images (see `flint.ml.siglip.image.siglip_preprocess`)."""
        var r = self.vision.encode(self.rt, pixels, batch)
        var e = r[0].copy()
        if normalize:
            l2_normalize(e, self.vision.width)
        return e^

    def logits(self, text_emb: List[Float32], image_emb: List[Float32]) -> List[Float32]:
        """[texts, images] sigmoid-head logits from normalised embeddings."""
        var D = self.dim
        var nt = len(text_emb) // D
        var ni = len(image_emb) // D
        var s = exp(self.logit_scale)
        var out = List[Float32](length=nt * ni, fill=0.0)
        for t in range(nt):
            for i in range(ni):
                var acc: Float32 = 0
                for d in range(D):
                    acc += text_emb[t * D + d] * image_emb[i * D + d]
                out[t * ni + i] = acc * s + self.logit_bias
        return out^


def sigmoid(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-x))
