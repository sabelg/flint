"""CLIP text and vision towers (`flint.ml.diffusion.clip`), Hugging Face `transformers` layout.

`ClipText` (`text_model.*` [+ `text_projection`]): token + position embeddings, pre-LN
transformer layers with causal self-attention and quick-GELU / GELU MLPs, final LayerNorm.
`encode` returns the per-token hidden states (the SD cross-attention context) and the pooled
end-of-text state, projected when the checkpoint has `text_projection` (CLIPTextModelWithProjection,
sentence-transformers CLIP).

`ClipVision` (`vision_model.*` [+ `visual_projection`]): patch conv, class token, positions,
`pre_layrnorm`, the same layers without the mask, `post_layernorm` on the class token, projection.
`preprocess` implements `CLIPImageProcessor` (shortest-edge bicubic resize, centre crop, mean / std).

Both run on any `Runtime` (CPU or GPU); a `ClipText` / `ClipVision` can be loaded from a
standalone `CLIPTextModel*` / `CLIPVisionModel*` file or from a full `CLIPModel` file.
"""

from std.math import sqrt
from flint.platform.os import read_bytes
from flint.assets.json import parse_json
from flint.ml.tensor import Tensor, fptr_at
from flint.ml.safetensors import SafeTensors
from flint.ml.diffusion.nn import Runtime, Act, Linear, Conv, Norm
from flint.ml.diffusion.kernels import ACT_GELU, ACT_QUICK_GELU
from flint.ml.diffusion.tokenizer import eos_position


struct ClipLayer(Copyable, Movable):
    var ln1: Norm
    var q: Linear
    var k: Linear
    var v: Linear
    var o: Linear
    var ln2: Norm
    var fc1: Linear
    var fc2: Linear

    def __init__(out self):
        self.ln1 = Norm()
        self.q = Linear()
        self.k = Linear()
        self.v = Linear()
        self.o = Linear()
        self.ln2 = Norm()
        self.fc1 = Linear()
        self.fc2 = Linear()


def _load_layers(mut rt: Runtime, st: SafeTensors, prefix: String, eps: Float32) raises -> List[ClipLayer]:
    var out = List[ClipLayer]()
    var i = 0
    while st.has(prefix + ".encoder.layers." + String(i) + ".layer_norm1.weight"):
        var p = prefix + ".encoder.layers." + String(i)
        var l = ClipLayer()
        l.ln1 = rt.norm(st, p + ".layer_norm1", 1, eps)
        l.q = rt.linear(st, p + ".self_attn.q_proj")
        l.k = rt.linear(st, p + ".self_attn.k_proj")
        l.v = rt.linear(st, p + ".self_attn.v_proj")
        l.o = rt.linear(st, p + ".self_attn.out_proj")
        l.ln2 = rt.norm(st, p + ".layer_norm2", 1, eps)
        l.fc1 = rt.linear(st, p + ".mlp.fc1")
        l.fc2 = rt.linear(st, p + ".mlp.fc2")
        out.append(l^)
        i += 1
    return out^


def _layers_forward(mut rt: Runtime, layers: List[ClipLayer], var h: Act, heads: Int, causal: Bool,
                    act: Int) raises -> Act:
    for i in range(len(layers)):
        var n1 = rt.layer_norm(h, layers[i].ln1)
        var q = rt.linear(n1, layers[i].q)
        var k = rt.linear(n1, layers[i].k)
        var v = rt.linear(n1, layers[i].v)
        rt.free(n1^)
        var a = rt.attention(q, k, v, heads, causal)
        rt.free(q^)
        rt.free(k^)
        rt.free(v^)
        var o = rt.linear(a, layers[i].o)
        rt.free(a^)
        rt.add_(h, o)
        rt.free(o^)
        var n2 = rt.layer_norm(h, layers[i].ln2)
        var f = rt.linear(n2, layers[i].fc1)
        rt.free(n2^)
        rt.activate(f, act)
        var f2 = rt.linear(f, layers[i].fc2)
        rt.free(f^)
        rt.add_(h, f2)
        rt.free(f2^)
    return h^


def _config_act(cfg: List[UInt8], section: String) -> Int:
    if len(cfg) == 0:
        return ACT_QUICK_GELU
    var doc = parse_json(cfg)
    var node = doc.root
    if section != "":
        var s = doc.object_get(doc.root, section)
        if s >= 0:
            node = s
    var a = doc.text(doc.object_get(node, "hidden_act"), "quick_gelu")
    if a == "gelu":
        return ACT_GELU
    return ACT_QUICK_GELU


def _config_int(cfg: List[UInt8], section: String, key: String, fallback: Int) -> Int:
    if len(cfg) == 0:
        return fallback
    var doc = parse_json(cfg)
    var node = doc.root
    if section != "":
        var s = doc.object_get(doc.root, section)
        if s >= 0:
            node = s
    return doc.integer(doc.object_get(node, key), fallback)


struct ClipEmbedding(Copyable, Movable):
    """`hidden` [L, D] (host) plus the pooled (projected) sentence / image embedding."""
    var hidden: List[Float32]
    var tokens: Int
    var width: Int
    var pooled: List[Float32]

    def __init__(out self):
        self.hidden = List[Float32]()
        self.tokens = 0
        self.width = 0
        self.pooled = List[Float32]()


struct ClipText(Movable):
    var tok_emb: List[Float32]
    var pos_emb: List[Float32]
    var width: Int
    var layers: List[ClipLayer]
    var final_ln: Norm
    var has_proj: Bool
    var proj: Linear
    var heads: Int
    var act: Int
    var eos: Int

    def __init__(out self):
        self.tok_emb = List[Float32]()
        self.pos_emb = List[Float32]()
        self.width = 0
        self.layers = List[ClipLayer]()
        self.final_ln = Norm()
        self.has_proj = False
        self.proj = Linear()
        self.heads = 8
        self.act = ACT_QUICK_GELU
        self.eos = 49407

    @staticmethod
    def load(mut rt: Runtime, safetensors_path: String, config_path: String) raises -> ClipText:
        var st = SafeTensors.open(safetensors_path)
        var cfg = read_bytes(config_path)
        var full = len(cfg) > 0 and parse_json(cfg).object_get(parse_json(cfg).root, "text_config") >= 0
        var sec = "text_config" if full else ""
        var t = ClipText()
        t.tok_emb = st.weight("text_model.embeddings.token_embedding.weight").owned_floats()
        t.pos_emb = st.weight("text_model.embeddings.position_embedding.weight").owned_floats()
        t.width = st.shape("text_model.embeddings.token_embedding.weight")[1]
        t.layers = _load_layers(rt, st, "text_model", 1e-5)
        t.final_ln = rt.norm(st, "text_model.final_layer_norm", 1, 1e-5)
        if st.has("text_projection.weight"):
            t.proj = rt.linear(st, "text_projection", False)
            t.has_proj = True
        t.heads = _config_int(cfg, sec, "num_attention_heads", max(1, t.width // 64))
        t.act = _config_act(cfg, sec)
        return t^

    def encode(mut self, mut rt: Runtime, ids: List[Int]) raises -> ClipEmbedding:
        var L = len(ids)
        var D = self.width
        var x = List[Float32](length=L * D, fill=0.0)
        for i in range(L):
            for d in range(D):
                x[i * D + d] = self.tok_emb[ids[i] * D + d] + self.pos_emb[i * D + d]
        var h = rt.upload(1, L, 1, D, x)
        h = _layers_forward(rt, self.layers, h^, self.heads, True, self.act)
        var hn = rt.layer_norm(h, self.final_ln)
        rt.free(h^)
        var out = ClipEmbedding()
        out.hidden = rt.download(hn)
        out.tokens = L
        out.width = D
        rt.free(hn^)
        var e = eos_position(ids, self.eos)
        var pooled = List[Float32](length=D, fill=0.0)
        for d in range(D):
            pooled[d] = out.hidden[e * D + d]
        if self.has_proj:
            var p = rt.upload(1, 1, 1, D, pooled)
            var pp = rt.linear(p, self.proj)
            out.pooled = rt.download(pp)
            rt.free(p^)
            rt.free(pp^)
        else:
            out.pooled = pooled^
        return out^


struct ClipVision(Movable):
    var patch: Conv
    var cls: List[Float32]
    var pos: List[Float32]
    var width: Int
    var patch_size: Int
    var image_size: Int
    var pre_ln: Norm
    var layers: List[ClipLayer]
    var post_ln: Norm
    var has_proj: Bool
    var proj: Linear
    var heads: Int
    var act: Int
    var mean: List[Float32]
    var std: List[Float32]

    def __init__(out self):
        self.patch = Conv()
        self.cls = List[Float32]()
        self.pos = List[Float32]()
        self.width = 0
        self.patch_size = 14
        self.image_size = 224
        self.pre_ln = Norm()
        self.layers = List[ClipLayer]()
        self.post_ln = Norm()
        self.has_proj = False
        self.proj = Linear()
        self.heads = 16
        self.act = ACT_QUICK_GELU
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

    @staticmethod
    def load(mut rt: Runtime, safetensors_path: String, config_path: String) raises -> ClipVision:
        var st = SafeTensors.open(safetensors_path)
        var cfg = read_bytes(config_path)
        var full = len(cfg) > 0 and parse_json(cfg).object_get(parse_json(cfg).root, "vision_config") >= 0
        var sec = "vision_config" if full else ""
        var v = ClipVision()
        var pw = st.weight("vision_model.embeddings.patch_embedding.weight")
        v.patch_size = pw.shape[2]
        v.width = pw.shape[0]
        v.patch = rt.conv_from(pw, Tensor(), False, v.patch_size, 0, 0, 0, 0)
        v.cls = st.weight("vision_model.embeddings.class_embedding").owned_floats()
        v.pos = st.weight("vision_model.embeddings.position_embedding.weight").owned_floats()
        var npos = st.shape("vision_model.embeddings.position_embedding.weight")[0]
        var side = 1
        while (side + 1) * (side + 1) <= npos - 1:
            side += 1
        v.image_size = side * v.patch_size
        v.pre_ln = rt.norm(st, "vision_model.pre_layrnorm", 1, 1e-5)
        v.layers = _load_layers(rt, st, "vision_model", 1e-5)
        v.post_ln = rt.norm(st, "vision_model.post_layernorm", 1, 1e-5)
        if st.has("visual_projection.weight"):
            v.proj = rt.linear(st, "visual_projection", False)
            v.has_proj = True
        v.heads = _config_int(cfg, sec, "num_attention_heads", max(1, v.width // 64))
        v.act = _config_act(cfg, sec)
        return v^

    def encode_pixels(mut self, mut rt: Runtime, pixels: List[Float32]) raises -> ClipEmbedding:
        """`pixels`: normalised channels-last [S, S, 3] (S = image_size), as from `preprocess`."""
        var S = self.image_size
        var D = self.width
        var x = rt.upload(1, S, S, 3, pixels)
        var p = rt.conv2d(x, self.patch)
        rt.free(x^)
        var grid = p.h * p.w
        var patches = rt.download(p)
        rt.free(p^)
        var T = grid + 1
        var e = List[Float32](length=T * D, fill=0.0)
        for d in range(D):
            e[d] = self.cls[d] + self.pos[d]
        for t in range(grid):
            for d in range(D):
                e[(t + 1) * D + d] = patches[t * D + d] + self.pos[(t + 1) * D + d]
        var h0 = rt.upload(1, T, 1, D, e)
        var h = rt.layer_norm(h0, self.pre_ln)
        rt.free(h0^)
        h = _layers_forward(rt, self.layers, h^, self.heads, False, self.act)
        var hidden = rt.download(h)
        rt.free(h^)
        var c = List[Float32](length=D, fill=0.0)
        for d in range(D):
            c[d] = hidden[d]
        var ca = rt.upload(1, 1, 1, D, c)
        var cn = rt.layer_norm(ca, self.post_ln)
        rt.free(ca^)
        var out = ClipEmbedding()
        out.hidden = hidden^
        out.tokens = T
        out.width = D
        if self.has_proj:
            var pp = rt.linear(cn, self.proj)
            out.pooled = rt.download(pp)
            rt.free(pp^)
        else:
            out.pooled = rt.download(cn)
        rt.free(cn^)
        return out^

    def normalize(self, rgb01: List[Float32]) -> List[Float32]:
        """[S, S, 3] in [0, 1] -> (x - mean) / std."""
        var out = List[Float32](length=len(rgb01), fill=0.0)
        for i in range(len(rgb01)):
            var c = i % 3
            out[i] = (rgb01[i] - self.mean[c]) / self.std[c]
        return out^
