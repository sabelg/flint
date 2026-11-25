"""Convert the released Make-It-Animatable v1 checkpoints into the graphs
`flint.ml.make_it_animatable` runs.

usage: python tools/make_it_animatable/export.py MIA_CHECKOUT WEIGHTS_DIR OUT_DIR [--delete-weights]

  MIA_CHECKOUT  a clone of https://github.com/jasongzy/Make-It-Animatable (main = v1; its Python
                code is imported here, never copied into Flint)
  WEIGHTS_DIR   joints_coarse.pth, joints.pth, pose.pth and bw.pth from output/best/new/ in the
                Hugging Face repo jasongzy/Make-It-Animatable (Apache-2.0)
  OUT_DIR       receives (fp32, opset 17)
     mia_coarse.onnx    (pc [1, 32768, 3], latent_pc [1, 512, 3]) -> joints [1, 52, 6]
     mia_joints.onnx    (pc, latent_pc) -> joints [1, 52, 6]            (kinematic-causal head)
     mia_pose.onnx      (pc, latent_pc, joints [1, 52, 6]) -> pose [1, 52, 6] (global ortho6d)
     mia_bw_trunk.onnx  (pc, latent_pc) -> latents [1, 512, 512]
     mia_bw_head.onnx   (latents, queries [1, Q, 3]) -> bw [1, Q, 52]  (softmax skin weights)

`--delete-weights` removes each .pth right after its graph is written (for small disks).

Each network is upstream's PCAE: a 3DShape2VecSet point encoder (the 512 latent points are chosen
by farthest point sampling, done by the caller and passed as `latent_pc`), 24 self-attention
layers, and a cross-attention decoder whose queries are mesh vertices (skin weights) or 52 learned
joint embeddings. The graphs are the same maths as upstream's modules, rewritten in an
export-friendly form: attention as Einsum -> Mul -> Softmax -> Einsum (the pattern Flint fuses),
nn.TransformerEncoder layers spelled out, and the kinematic-causal heads' level-by-level loop
unrolled (11 levels of the Mixamo tree) with constant masks. `reference.py` checks them against
the untouched upstream modules.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) != 3:
    print(__doc__)
    sys.exit(1)
MIA, WEIGHTS, OUT = (os.path.abspath(a) for a in args)
DELETE = "--delete-weights" in sys.argv
os.makedirs(OUT, exist_ok=True)

import mia_torch  # noqa: E402

mia_model, models_ae, U = mia_torch.import_upstream(MIA)

import onnx  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

N = mia_torch.N_POINTS
J = 52


def attention(attn, x, context=None):
    """models_ae.Attention.forward without a mask, in the [B, L, H, D] einsum layout."""
    h = attn.heads
    ctx = x if context is None else context
    q = attn.to_q(x)
    B, L, inner = q.shape
    k, v = torch.split(attn.to_kv(ctx), inner, dim=-1)
    S = k.shape[1]
    d = inner // h
    q = q.reshape(B, L, h, d)
    k = k.reshape(B, S, h, d)
    v = v.reshape(B, S, h, d)
    sim = torch.einsum("blhd,bshd->bhls", q, k) * attn.scale
    out = torch.einsum("bhls,bshd->blhd", sim.softmax(dim=-1), v)
    return attn.to_out(out.reshape(B, L, inner))


def _geglu(self, x):
    """models_ae.GEGLU with torch.split (exports as one Split, which Flint runs on the GPU,
    instead of two Slices)."""
    a, g = torch.split(x, x.shape[-1] // 2, dim=-1)
    return a * F.gelu(g)


models_ae.GEGLU.forward = _geglu


def prenorm_attn(pn, x, context=None):
    xn = pn.norm(x)
    if pn.norm_context is not None:
        context = pn.norm_context(context)
    return attention(pn.fn, xn, context)


def prenorm_ff(pn, x):
    return pn.fn(pn.norm(x))


def trunk(m, pc, latent_pc):
    """PCAE.encode (with the FPS picks given) + the 24 latent self-attention layers."""
    base = m.base
    s_emb = base.point_embed(latent_pc)
    p_emb = base.point_embed(pc)
    cross_attn, cross_ff = base.cross_attend_blocks
    x = prenorm_attn(cross_attn, s_emb, p_emb) + s_emb
    x = prenorm_ff(cross_ff, x) + x
    for sa, ff in base.layers:
        x = prenorm_attn(sa, x) + x
        x = prenorm_ff(ff, x) + x
    return x


def decode(m, x, q_emb):
    return prenorm_attn(m.base.decoder_cross_attn, q_emb, x)


def encoder_layer(layer, x, add_mask):
    """nn.TransformerEncoderLayer (norm_first, ReLU, eval) with an additive attention mask."""
    sa = layer.self_attn
    h = sa.num_heads
    y = layer.norm1(x)
    E = sa.embed_dim
    q, k, v = torch.split(F.linear(y, sa.in_proj_weight, sa.in_proj_bias), E, dim=-1)
    B, L, _ = q.shape
    d = E // h
    q = q.reshape(B, L, h, d)
    k = k.reshape(B, L, h, d)
    v = v.reshape(B, L, h, d)
    sim = torch.einsum("blhd,bshd->bhls", q, k) * (d ** -0.5) + add_mask
    a = torch.einsum("bhls,bshd->blhd", sim.softmax(dim=-1), v).reshape(B, L, E)
    x = x + sa.out_proj(a)
    y = layer.norm2(x)
    return x + layer.linear2(F.relu(layer.linear1(y)))


def cross3(a, b):
    ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3]
    bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3]
    return torch.cat([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], dim=-1)


def unit(v):
    return v / torch.clamp(torch.sqrt((v * v).sum(-1, keepdim=True)), min=1e-12)


def ortho6d_roundtrip(o):
    """util.utils.matrix_to_ortho6d(ortho6d_to_matrix(o))."""
    x = unit(o[..., 0:3])
    z = unit(cross3(x, o[..., 3:6]))
    y = cross3(z, x)
    return torch.cat([x, y], dim=-1)


def causal_head(head, feat, kind):
    """model.JointsAttentionCausal.forward in eval mode: one pass per kinematic-tree level, each
    level's rows taken from that pass."""
    parent = head.mask_parent.float()                     # [J, J], one 1 per non-root row
    add_mask = torch.zeros(J, J).masked_fill(~head.mask_attn, float("-inf"))
    out = torch.zeros(feat.shape[0], J, head.out_dim)
    for level in head.tree_levels_mask:
        if not bool(level.any()):
            continue
        g = ortho6d_roundtrip(out) if kind == "pose" else out
        in_feat = torch.matmul(parent, head.encoder(g)) + feat
        y = in_feat
        for layer in head.transformer.layers.layers:
            y = encoder_layer(layer, y, add_mask)
        o = head.decoder.rotation_mlp(y) if kind == "pose" else head.decoder(y)
        out = torch.where(level.view(1, J, 1), o, out)
    return out


class Joints(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pc, latent_pc):
        m = self.m
        x = trunk(m, pc, latent_pc)
        lat = decode(m, x, m.joints_embed.expand(pc.shape[0], -1, -1))
        if m.joints_attn_causal:
            return causal_head(m.joints_head, lat, "joints")
        return m.joints_head(lat)


class Pose(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pc, latent_pc, joints):
        m = self.m
        x = trunk(m, pc, latent_pc)
        emb = m.pose_embed.expand(pc.shape[0], -1, -1) + m.joints_embedder(joints)
        return causal_head(m.pose_head, decode(m, x, emb), "pose")


class BwTrunk(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pc, latent_pc):
        return trunk(self.m, pc, latent_pc)


class BwHead(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, latents, queries):
        m = self.m
        return torch.softmax(m.bw_head(decode(m, latents, m.base.point_embed(queries))), dim=-1)


def export(mod, inputs, names, outs, dyn, path):
    torch.onnx.export(mod, inputs, path, input_names=names, output_names=outs, dynamic_axes=dyn,
                      opset_version=17, do_constant_folding=True, dynamo=False)
    g = onnx.load(path, load_external_data=False)
    ops = sorted({n.op_type for n in g.graph.node})
    print(os.path.basename(path), "nodes", len(g.graph.node), "size", os.path.getsize(path) // (1 << 20), "MB")
    print("  ops:", " ".join(ops))


def check(a, b, what):
    d = (a - b).abs().max().item()
    print(f"  {what}: max |export - upstream| = {d:.3g}")


def weights_done(name):
    if DELETE:
        os.remove(os.path.join(WEIGHTS, mia_torch.MODEL_FILES[name] + ".pth"))


torch.manual_seed(0)
pc = torch.rand(1, N, 3) * 2 - 1
pc[..., 1] *= 0.6
latent = pc[:, torch.randperm(N)[:512]]
with torch.no_grad():
    for name in ("coarse", "joints"):
        m = mia_torch.build_model(MIA, WEIGHTS, name)
        mod = Joints(m).eval()
        ref = m(pc).joints if False else None  # upstream FPS differs from `latent`; compared below
        export(mod, (pc, latent), ["pc", "latent_pc"], ["joints"], None, f"{OUT}/mia_{name}.onnx")
        # same maths as upstream on the same latent points
        mia_fps = m.fps
        m.fps = lambda p, _l=latent: _l
        check(mod(pc, latent), m(pc).joints, name + " joints")
        m.fps = mia_fps
        del m, mod
        weights_done(name)

    m = mia_torch.build_model(MIA, WEIGHTS, "pose")
    joints = torch.randn(1, J, 6) * 0.3
    mod = Pose(m).eval()
    export(mod, (pc, latent, joints), ["pc", "latent_pc", "joints"], ["pose"], None, f"{OUT}/mia_pose.onnx")
    m.fps = lambda p, _l=latent: _l
    check(mod(pc, latent, joints), m(pc, joints=joints.clone()).pose_trans, "pose")
    del m, mod
    weights_done("pose")

    m = mia_torch.build_model(MIA, WEIGHTS, "bw")
    q = torch.rand(1, 3000, 3) * 2 - 1
    t = BwTrunk(m).eval()
    h = BwHead(m).eval()
    export(t, (pc, latent), ["pc", "latent_pc"], ["latents"], None, f"{OUT}/mia_bw_trunk.onnx")
    lat = t(pc, latent)
    export(h, (lat, q), ["latents", "queries"], ["bw"], {"queries": {1: "q"}, "bw": {1: "q"}},
           f"{OUT}/mia_bw_head.onnx")
    m.fps = lambda p, _l=latent: _l
    check(h(lat, q), m(pc, q).bw, "skin weights")
    del m
    weights_done("bw")
print("wrote", OUT)
