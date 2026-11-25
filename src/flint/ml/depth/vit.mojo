"""DINOv2 ViT trunk shared by every `flint.ml.depth` model.

One implementation covers the three checkpoint dialects and the Depth Anything 3 extensions:

  - names: `transformers` Dinov2 (`embeddings.*`, `encoder.layer.N.attention.attention.query`)
    or the original DINOv2 layout (`pos_embed`, `blocks.N.attn.qkv`, `ls1.gamma`) under any prefix;
  - position embeddings resized like the source repo: by output size (`transformers`) or by the
    `interpolate_offset = 0.1` scale factor (DINOv2 / DA3 / MoGe), PyTorch bicubic, a = -0.75;
  - Depth Anything 3: a camera token written into token 0 at block `alt_start`, alternating local
    (per view) / global (all views) attention from there on, LayerNorm QK-norm and 2-D RoPE
    (frequency 100) from `qknorm_start` / `rope_start`, and `cat_token` features
    `[last local output, normed output]`;
  - several views at once (`S` images of the same size) with global attention across them.

LayerScale is folded into `attn.proj` / `mlp.fc2`; q / k / v are separate GEMMs so QK-norm and
RoPE act on contiguous rows. Features are returned per requested block as token values
[S, 1 + N, 1, C'] (token 0 still in front; callers slice it off).
"""

from std.math import floor, sqrt
from std.ffi import external_call
from flint.ml.tensor import fptr_at
from flint.ml.diffusion.nn import Runtime, Act, Linear, Norm, ACT_GELU
from flint.ml.depth.weights import Weights
from flint.ml.depth.layers import (
    load_linear, linear_rows, load_norm, rope_, copy_rows, add_rows, upload_rep, take_tokens,
)

comptime VIT_HF: Int = 0       # transformers Dinov2Model names
comptime VIT_DINO: Int = 1     # original DINOv2 names (blocks.N.attn.qkv ...)

comptime POS_SIZE: Int = 0     # interpolate(size=(gh, gw))
comptime POS_OFFSET: Int = 1   # interpolate(scale_factor=((gh + 0.1) / M, (gw + 0.1) / M))


struct VitConfig(Copyable, Movable):
    var dim: Int
    var depth: Int
    var heads: Int
    var patch: Int
    var style: Int
    var prefix: String
    var pos_mode: Int
    var eps: Float32          # block LayerNorms
    var final_eps: Float32    # the final `norm`
    var qk_eps: Float32
    var alt_start: Int        # -1: no camera token / global attention
    var qknorm_start: Int
    var rope_start: Int
    var rope_freq: Float32
    var cat_token: Bool
    var out_layers: List[Int]

    def __init__(out self):
        self.dim = 384
        self.depth = 12
        self.heads = 6
        self.patch = 14
        self.style = VIT_DINO
        self.prefix = String()
        self.pos_mode = POS_OFFSET
        self.eps = 1e-6
        self.final_eps = 1e-6
        self.qk_eps = 1e-5
        self.alt_start = -1
        self.qknorm_start = -1
        self.rope_start = -1
        self.rope_freq = 100.0
        self.cat_token = False
        self.out_layers = List[Int]()


def vit_size(name: String, mut cfg: VitConfig) raises:
    """vits / vitb / vitl (or dinov2_vits14 ...) -> width, depth, heads."""
    if name.find("vits") >= 0:
        cfg.dim = 384
        cfg.depth = 12
        cfg.heads = 6
    elif name.find("vitb") >= 0:
        cfg.dim = 768
        cfg.depth = 12
        cfg.heads = 12
    elif name.find("vitl") >= 0:
        cfg.dim = 1024
        cfg.depth = 24
        cfg.heads = 16
    else:
        raise Error("depth: unsupported ViT size " + name + " (vits / vitb / vitl)")


struct VitBlock(Movable):
    var n1: Norm
    var q: Linear
    var k: Linear
    var v: Linear
    var qk_norm: Bool
    var qn: Norm
    var kn: Norm
    var proj: Linear
    var n2: Norm
    var fc1: Linear
    var fc2: Linear

    def __init__(out self):
        self.n1 = Norm()
        self.q = Linear()
        self.k = Linear()
        self.v = Linear()
        self.qk_norm = False
        self.qn = Norm()
        self.kn = Norm()
        self.proj = Linear()
        self.n2 = Norm()
        self.fc1 = Linear()
        self.fc2 = Linear()


# ----------------------------------------------------------------------------- bicubic position grid

def _cubic1(x: Float32) -> Float32:
    comptime A: Float32 = -0.75
    return ((A + 2) * x - (A + 3)) * x * x + 1


def _cubic2(x: Float32) -> Float32:
    comptime A: Float32 = -0.75
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A


def _cubic_taps(n_in: Int, n_out: Int, scale: Float32) -> Tuple[List[Int], List[Float32]]:
    var idx = List[Int](length=n_out * 4, fill=0)
    var wts = List[Float32](length=n_out * 4, fill=0.0)
    for o in range(n_out):
        var s = scale * (Float32(o) + 0.5) - 0.5
        var i0 = Int(floor(s))
        var t = s - Float32(i0)
        var ws: List[Float32] = [_cubic2(t + 1), _cubic1(t), _cubic1(1 - t), _cubic2(2 - t)]
        for k in range(4):
            idx[o * 4 + k] = min(max(i0 - 1 + k, 0), n_in - 1)
            wts[o * 4 + k] = ws[k]
    return (idx^, wts^)


def bicubic_grid(src: List[Float32], M: Int, gh: Int, gw: Int, C: Int, sy: Float32, sx: Float32) -> List[Float32]:
    """PyTorch bicubic (align_corners False, no antialias) of a channels-last [M, M, C] grid to
    [gh, gw, C] with source-per-output scales (sy, sx)."""
    var ty = _cubic_taps(M, gh, sy)
    var tx = _cubic_taps(M, gw, sx)
    var out = List[Float32](length=gh * gw * C, fill=0.0)
    for oy in range(gh):
        for ox in range(gw):
            for c in range(C):
                var acc: Float32 = 0
                for ky in range(4):
                    var row: Float32 = 0
                    for kx in range(4):
                        row += src[(ty[0][oy * 4 + ky] * M + tx[0][ox * 4 + kx]) * C + c] * tx[1][ox * 4 + kx]
                    acc += row * ty[1][oy * 4 + ky]
                out[(oy * gw + ox) * C + c] = acc
    return out^


# ----------------------------------------------------------------------------- reference view

comptime REF_SELECT_MIN_VIEWS: Int = 3   # depth_anything_3 THRESH_FOR_REF_SELECTION


def select_reference_view(cls: List[Float32], S: Int, C: Int) -> Int:
    """Depth Anything 3 `select_reference_view(strategy="saddle_balanced")` on the class tokens
    [S, C]: the view closest to the middle of the range of mean similarity to the other views,
    token norm and normalised-token variance."""
    var nrm = List[Float64](length=S, fill=0.0)
    var unit = List[Float64](length=S * C, fill=0.0)
    var var_ = List[Float64](length=S, fill=0.0)
    for s in range(S):
        var acc = 0.0
        for c in range(C):
            var v = Float64(cls[s * C + c])
            acc += v * v
        nrm[s] = sqrt(acc)
        var mean = 0.0
        for c in range(C):
            unit[s * C + c] = Float64(cls[s * C + c]) / nrm[s]
            mean += unit[s * C + c]
        mean /= Float64(C)
        var sq = 0.0
        for c in range(C):
            var d = unit[s * C + c] - mean
            sq += d * d
        var_[s] = sq / Float64(C - 1)
    var sim = List[Float64](length=S, fill=0.0)
    for a in range(S):
        var acc = 0.0
        for b in range(S):
            var d = 0.0
            for c in range(C):
                d += unit[a * C + c] * unit[b * C + c]
            acc += d - (1.0 if a == b else 0.0)
        sim[a] = acc / Float64(S - 1)
    var ms = List[List[Float64]]()
    ms.append(sim^)
    ms.append(nrm^)
    ms.append(var_^)
    var score = List[Float64](length=S, fill=0.0)
    for m in ms:
        var lo = m[0]
        var hi = m[0]
        for s in range(S):
            lo = min(lo, m[s])
            hi = max(hi, m[s])
        for s in range(S):
            score[s] += abs((m[s] - lo) / (hi - lo + 1e-8) - 0.5)
    var best = 0
    for s in range(1, S):
        if score[s] < score[best]:
            best = s
    return best


def reorder_by_reference(S: Int, ref_idx: Int) -> List[Int]:
    """Gather order putting view `ref_idx` first and shifting views before it back by one."""
    var order = List[Int](length=S, fill=0)
    for p in range(S):
        order[p] = p - 1 if (p > 0 and p <= ref_idx) else p
    order[0] = ref_idx
    return order^


def permute_views(mut rt: Runtime, x: Act, order: List[Int]) raises -> Act:
    """out[t] = x[order[t]] over the batch (view) axis."""
    var o = rt.alloc(x.b, x.h, x.w, x.c)
    var per = x.h * x.w * x.c
    for t in range(x.b):
        copy_rows(rt, x, o, 1, per, per, order[t] * per, per, t * per)
    return o^


# ----------------------------------------------------------------------------- model

struct DinoVit(Movable):
    var cfg: VitConfig
    var patch_w: Linear
    var cls: List[Float32]
    var pos: List[Float32]        # [1 + M * M, C]
    var cam: List[Float32]        # [2, C] (DA3 reference / source view camera tokens) or empty
    var norm: Norm
    var blocks: List[VitBlock]
    var rope_freq: List[Float32]
    var pos_cache_key: String
    var pos_cache: List[Float32]
    var raw0: List[Float32]       # [S, C'] token 0 of the last out layer before the final norm
    var ref_view: Int             # the DA3 reference view picked by the last forward

    def __init__(out self, var cfg: VitConfig):
        self.cfg = cfg^
        self.patch_w = Linear()
        self.cls = List[Float32]()
        self.pos = List[Float32]()
        self.cam = List[Float32]()
        self.norm = Norm()
        self.blocks = List[VitBlock]()
        self.rope_freq = List[Float32]()
        self.pos_cache_key = String()
        self.pos_cache = List[Float32]()
        self.raw0 = List[Float32]()
        self.ref_view = 0

    @staticmethod
    def load(mut rt: Runtime, w: Weights, var cfg: VitConfig) raises -> DinoVit:
        var m = DinoVit(cfg^)
        var p = m.cfg.prefix
        var C = m.cfg.dim
        var hf = m.cfg.style == VIT_HF
        if hf:
            m.patch_w = load_linear(rt, w, p + "embeddings.patch_embeddings.projection")
            m.cls = w.floats(p + "embeddings.cls_token")
            m.pos = w.floats(p + "embeddings.position_embeddings")
            m.norm = load_norm(rt, w, p + "layernorm", 1, m.cfg.final_eps)
        else:
            m.patch_w = load_linear(rt, w, p + "patch_embed.proj")
            m.cls = w.floats(p + "cls_token")
            m.pos = w.floats(p + "pos_embed")
            m.norm = load_norm(rt, w, p + "norm", 1, m.cfg.final_eps)
            if w.has(p + "camera_token"):
                m.cam = w.floats(p + "camera_token")
        if w.has(p + "register_tokens") or w.has(p + "embeddings.register_tokens"):
            raise Error("depth: DINOv2 register tokens are not supported")
        for l in range(m.cfg.depth):
            var blk = VitBlock()
            if hf:
                var b = p + "encoder.layer." + String(l) + "."
                blk.n1 = load_norm(rt, w, b + "norm1", 1, m.cfg.eps)
                blk.q = load_linear(rt, w, b + "attention.attention.query")
                blk.k = load_linear(rt, w, b + "attention.attention.key")
                blk.v = load_linear(rt, w, b + "attention.attention.value")
                var g1 = w.floats(b + "layer_scale1.lambda1")
                blk.proj = linear_rows(rt, w, b + "attention.output.dense", 0, C, True, g1)
                blk.n2 = load_norm(rt, w, b + "norm2", 1, m.cfg.eps)
                blk.fc1 = load_linear(rt, w, b + "mlp.fc1")
                var g2 = w.floats(b + "layer_scale2.lambda1")
                blk.fc2 = linear_rows(rt, w, b + "mlp.fc2", 0, C, True, g2)
            else:
                var b = p + "blocks." + String(l) + "."
                blk.n1 = load_norm(rt, w, b + "norm1", 1, m.cfg.eps)
                blk.q = linear_rows(rt, w, b + "attn.qkv", 0, C)
                blk.k = linear_rows(rt, w, b + "attn.qkv", C, 2 * C)
                blk.v = linear_rows(rt, w, b + "attn.qkv", 2 * C, 3 * C)
                if w.has(b + "attn.q_norm.weight"):
                    blk.qk_norm = True
                    blk.qn = load_norm(rt, w, b + "attn.q_norm", 1, m.cfg.qk_eps)
                    blk.kn = load_norm(rt, w, b + "attn.k_norm", 1, m.cfg.qk_eps)
                var g1 = w.floats(b + "ls1.gamma")
                blk.proj = linear_rows(rt, w, b + "attn.proj", 0, C, True, g1)
                blk.n2 = load_norm(rt, w, b + "norm2", 1, m.cfg.eps)
                blk.fc1 = load_linear(rt, w, b + "mlp.fc1")
                var g2 = w.floats(b + "ls2.gamma")
                blk.fc2 = linear_rows(rt, w, b + "mlp.fc2", 0, C, True, g2)
            m.blocks.append(blk^)
        var hd = C // m.cfg.heads
        var F = hd // 2
        for i in range(F // 2):
            var e = Float32(2 * i) / Float32(F)
            m.rope_freq.append(Float32(1.0) / (m.cfg.rope_freq ** e))
        return m^

    def pos_table(mut self, gh: Int, gw: Int) -> List[Float32]:
        """[1 + gh * gw, C] position table for a gh x gw patch grid; row 0 carries cls + pos[0]."""
        var key = String(gh) + "x" + String(gw)
        if key == self.pos_cache_key:
            return self.pos_cache.copy()
        var C = self.cfg.dim
        var N = len(self.pos) // C - 1
        var M = 1
        while M * M < N:
            M += 1
        var out = List[Float32](length=(1 + gh * gw) * C, fill=0.0)
        for c in range(C):
            out[c] = self.pos[c] + self.cls[c]
        if gh * gw == N and gh == gw:
            for i in range(N * C):
                out[C + i] = self.pos[C + i]
        else:
            var grid = List[Float32](length=N * C, fill=0.0)
            for i in range(N * C):
                grid[i] = self.pos[C + i]
            var sy: Float32
            var sx: Float32
            if self.cfg.pos_mode == POS_SIZE:
                sy = Float32(M) / Float32(gh)
                sx = Float32(M) / Float32(gw)
            else:
                sy = Float32(1.0 / ((Float64(gh) + 0.1) / Float64(M)))
                sx = Float32(1.0 / ((Float64(gw) + 0.1) / Float64(M)))
            var g = bicubic_grid(grid, M, gh, gw, C, sy, sx)
            for i in range(gh * gw * C):
                out[C + i] = g[i]
        self.pos_cache_key = key
        self.pos_cache = out.copy()
        return out^

    def embed(mut self, mut rt: Runtime, pixels: List[Float32], S: Int, H: Int, W: Int) raises -> Act:
        """Patch + position embedding of S normalised images [S, 3, H, W] -> tokens [S, 1 + N, 1, C]."""
        var P = self.cfg.patch
        var gh = H // P
        var gw = W // P
        var N = gh * gw
        var C = self.cfg.dim
        var K = 3 * P * P
        var patches = List[Float32](length=S * N * K, fill=0.0)
        for s in range(S):
            for py in range(gh):
                for px in range(gw):
                    var r = (s * N + py * gw + px) * K
                    for c in range(3):
                        for ky in range(P):
                            var src = ((s * 3 + c) * H + py * P + ky) * W + px * P
                            var dst = r + (c * P + ky) * P
                            for kx in range(P):
                                patches[dst + kx] = pixels[src + kx]
        var pa = rt.upload(S, N, 1, K, patches)
        var pe = rt.linear(pa, self.patch_w)
        rt.free(pa^)
        var T = N + 1
        var x = upload_rep(rt, self.pos_table(gh, gw), S, T, 1, C)
        add_rows(rt, x, pe, S, N * C, T * C, C, N * C)
        rt.free(pe^)
        return x^

    def _rope_pos(self, mut rt: Runtime, S: Int, gh: Int, gw: Int, nodiff: Bool) raises -> Act:
        var T = gh * gw + 1
        var p = List[Float32](length=S * T * 2, fill=0.0)
        for s in range(S):
            for i in range(gh * gw):
                var r = s * T + 1 + i
                if nodiff:
                    p[2 * r] = 1
                    p[2 * r + 1] = 1
                else:
                    p[2 * r] = Float32(i // gw + 1)
                    p[2 * r + 1] = Float32(i % gw + 1)
        return rt.upload(S, T, 1, 2, p)

    def _block(mut self, mut rt: Runtime, mut x: Act, l: Int, is_global: Bool, pos: Act, freq: Act,
               use_rope: Bool) raises:
        var S = x.b
        var T = x.h
        var C = self.cfg.dim
        var H = self.cfg.heads
        var hd = C // H
        var n = rt.layer_norm(x, self.blocks[l].n1)
        var q = rt.linear(n, self.blocks[l].q)
        var k = rt.linear(n, self.blocks[l].k)
        var v = rt.linear(n, self.blocks[l].v)
        rt.free(n^)
        if self.blocks[l].qk_norm:
            var qr = rt.reshape(q^, S, T * H, 1, hd)
            var qn = rt.layer_norm(qr, self.blocks[l].qn)
            rt.free(qr^)
            q = rt.reshape(qn^, S, T, 1, C)
            var kr = rt.reshape(k^, S, T * H, 1, hd)
            var kn = rt.layer_norm(kr, self.blocks[l].kn)
            rt.free(kr^)
            k = rt.reshape(kn^, S, T, 1, C)
        if use_rope:
            rope_(rt, q, pos, freq, H)
            rope_(rt, k, pos, freq, H)
        if is_global and S > 1:
            q = rt.reshape(q^, 1, S * T, 1, C)
            k = rt.reshape(k^, 1, S * T, 1, C)
            v = rt.reshape(v^, 1, S * T, 1, C)
        var a = rt.attention(q, k, v, H)
        rt.free(q^)
        rt.free(k^)
        rt.free(v^)
        a = rt.reshape(a^, S, T, 1, C)
        var o = rt.linear(a, self.blocks[l].proj)
        rt.free(a^)
        rt.add_(x, o)
        rt.free(o^)
        var n2 = rt.layer_norm(x, self.blocks[l].n2)
        var h = rt.linear(n2, self.blocks[l].fc1)
        rt.free(n2^)
        rt.activate(h, ACT_GELU)
        var o2 = rt.linear(h, self.blocks[l].fc2)
        rt.free(h^)
        rt.add_(x, o2)
        rt.free(o2^)

    def forward(mut self, mut rt: Runtime, pixels: List[Float32], S: Int, H: Int, W: Int) raises -> List[Act]:
        """Features of `out_layers` for S images [S, 3, H, W] (ImageNet-normalised, H and W
        multiples of the patch): normed tokens, or [local, normed] when `cat_token`."""
        var P = self.cfg.patch
        var gh = H // P
        var gw = W // P
        var C = self.cfg.dim
        var x = self.embed(rt, pixels, S, H, W)
        var T = x.h
        var has_rope = self.cfg.rope_start >= 0
        var pos_l = Act()
        var pos_g = Act()
        var freq = Act()
        if has_rope:
            pos_l = self._rope_pos(rt, S, gh, gw, False)
            pos_g = self._rope_pos(rt, S, gh, gw, True)
            freq = rt.upload(1, len(self.rope_freq), 1, 1, self.rope_freq)
        var feats = List[Act]()
        var local_x = Act()
        var have_local = False
        var alt = self.cfg.alt_start
        self.ref_view = 0
        var order = List[Int]()
        for s in range(S):
            order.append(s)
        for l in range(self.cfg.depth):
            if alt >= 0 and l == alt - 1 and S >= REF_SELECT_MIN_VIEWS:
                # Depth Anything 3 "saddle_balanced": the view whose class token is the most
                # typical becomes the reference (moved to the front, restored on output)
                var t0 = take_tokens(rt, x, 0, 1)
                var cls = rt.download(t0)
                rt.free(t0^)
                self.ref_view = select_reference_view(cls, S, C)
                if self.ref_view != 0:
                    order = reorder_by_reference(S, self.ref_view)
                    var nx = permute_views(rt, x, order)
                    rt.free(x^)
                    x = nx^
                    if have_local:
                        var nl = permute_views(rt, local_x, order)
                        rt.free(local_x^)
                        local_x = nl^
            if alt >= 0 and l == alt:
                # camera tokens: reference view gets camera_token[0], the others camera_token[1]
                var ct = List[Float32](length=S * C, fill=0.0)
                for s in range(S):
                    var src = 0 if s == 0 else C
                    for c in range(C):
                        ct[s * C + c] = self.cam[src + c]
                var cta = rt.upload(S, 1, 1, C, ct)
                copy_rows(rt, cta, x, S, C, C, 0, T * C, 0)
                rt.free(cta^)
            var is_global = alt >= 0 and l >= alt and l % 2 == 1
            var use_rope = has_rope and l >= self.cfg.rope_start
            if is_global:
                self._block(rt, x, l, True, pos_g, freq, use_rope)
            else:
                self._block(rt, x, l, False, pos_l, freq, use_rope)
                if self.cfg.cat_token:
                    if have_local:
                        rt.free(local_x^)
                    local_x = rt.copy(x)
                    have_local = True
            var want = False
            for o in self.cfg.out_layers:
                if o == l:
                    want = True
            if want:
                if l == self.cfg.out_layers[len(self.cfg.out_layers) - 1]:
                    var t0 = take_tokens(rt, x, 0, 1)
                    var hx = rt.download(t0)
                    rt.free(t0^)
                    if self.cfg.cat_token:
                        var t1 = take_tokens(rt, local_x, 0, 1)
                        var hl = rt.download(t1)
                        rt.free(t1^)
                        self.raw0 = List[Float32](length=S * 2 * C, fill=0.0)
                        for s in range(S):
                            for c in range(C):
                                self.raw0[s * 2 * C + c] = hl[s * C + c]
                                self.raw0[s * 2 * C + C + c] = hx[s * C + c]
                    else:
                        self.raw0 = hx^
                var nx = rt.layer_norm(x, self.norm)
                if self.cfg.cat_token:
                    var cat = rt.concat_c(local_x, nx)
                    rt.free(nx^)
                    feats.append(cat^)
                else:
                    feats.append(nx^)
        if self.ref_view != 0:
            var inv = List[Int](length=S, fill=0)
            for s in range(S):
                inv[order[s]] = s
            # restored[t] = reordered[inv[t]]
            var restored = List[Act]()
            for i in range(len(feats)):
                restored.append(permute_views(rt, feats[i], inv))
            while len(feats) > 0:
                var old = feats.pop()
                rt.free(old^)
            feats = restored^
            var Cr = len(self.raw0) // S
            var raw = self.raw0.copy()
            for t in range(S):
                for c in range(Cr):
                    self.raw0[t * Cr + c] = raw[inv[t] * Cr + c]
        if have_local:
            rt.free(local_x^)
        if has_rope:
            rt.free(pos_l^)
            rt.free(pos_g^)
            rt.free(freq^)
        rt.free(x^)
        return feats^

    def raw_token0(mut self, mut rt: Runtime, feat: Act) raises -> List[Float32]:
        """Token 0 of every image of a feature value, as host rows [S, C']."""
        var h = rt.download(feat)
        var S = feat.b
        var T = feat.h * feat.w
        var Cc = feat.c
        var out = List[Float32](length=S * Cc, fill=0.0)
        for s in range(S):
            for c in range(Cc):
                out[s * Cc + c] = h[s * T * Cc + c]
        return out^
