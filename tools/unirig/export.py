"""Convert the released UniRig checkpoints into the five graphs `flint.ml.unirig_rig` runs.

usage: python tools/unirig/export.py UNIRIG_CHECKOUT WEIGHTS_DIR OUT_DIR

  UNIRIG_CHECKOUT  a clone of https://github.com/VAST-AI-Research/UniRig (its Python code is
                   imported here, never copied into Flint)
  WEIGHTS_DIR      holds skeleton/articulation-xl_quantization_256/model.ckpt and
                   skin/articulation-xl/model.ckpt (from the UniRig Hugging Face release)
  OUT_DIR          receives
     unirig_skel_encoder.onnx  shape encoder + output_proj: (pc, feats, qpc, qfeats) -> cond[1, M, 1024]
     unirig_skel_opt.onnx      OPT-350M decoder weights only (initializers, no graph)
     unirig_skin_global.onnx   skin shape encoder + out_proj -> glat[1, M, 768]
     unirig_skin_body.onnx     PTv3 (neighbour tables / serialisation orders given by the caller),
                               bone encoder, mesh-bone attention -> mfeat[N, 768], bonek[16, J, 768]
     unirig_skin_head.onnx     per-vertex skin head -> pred[n, J]

Needs torch, onnx, transformers, python-box, pyyaml and trimesh. `shims/` replaces the CUDA-only
packages UniRig imports (flash_attn, spconv, torch_scatter, torch_cluster, timm) with small fp32
torch stand-ins; the exported graphs are plain fp32 ONNX (opset 17).

Licensing: UniRig and its weights are MIT. The shape encoders come from Michelangelo (GPL-3.0),
so the exported encoder graphs are GPL-derived files: keep them outside the Flint repository and
do not redistribute them with Flint builds. Flint itself only contains the MIT-licensed runtime.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)
UNIRIG, WEIGHTS, OUT = (os.path.abspath(a) for a in sys.argv[1:4])
sys.path.insert(0, os.path.join(HERE, "shims"))
sys.path.insert(1, UNIRIG)
os.makedirs(OUT, exist_ok=True)

import numpy as np
import onnx
import torch
import yaml
from box import Box
from onnx import helper, numpy_helper


def cfg(rel):
    with open(os.path.join(UNIRIG, "configs", rel)) as fh:
        return Box(yaml.safe_load(fh))


def load_state(path, prefix="model."):
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def export(mod, args, path, names, outs, dyn):
    torch.onnx.export(mod, args, path, input_names=names, output_names=outs, dynamic_axes=dyn,
                      opset_version=17, do_constant_folding=True, dynamo=False)
    g = onnx.load(path, load_external_data=False)
    print(os.path.basename(path), "nodes", len(g.graph.node))


class ShapeEncoder(torch.nn.Module):
    """Michelangelo perceiver encoder with caller-chosen query points, then a projection."""

    def __init__(self, e, proj):
        super().__init__()
        self.e, self.proj = e, proj

    def forward(self, pc, feats, qpc, qfeats):
        e = self.e
        data = e.input_proj(torch.cat([e.fourier_embedder(pc), feats], -1))
        sdata = e.input_proj(torch.cat([e.fourier_embedder(qpc), qfeats], -1))
        lat = e.self_attn(e.cross_attn(sdata, data))
        if e.ln_post is not None:
            lat = e.ln_post(lat)
        return self.proj(lat)


def encoder_args(N=4096, M=1024):
    return (torch.randn(1, N, 3), torch.randn(1, N, 3), torch.randn(1, M, 3), torch.randn(1, M, 3))


ENC_DYN = {"pc": {1: "n"}, "feats": {1: "n"}, "qpc": {1: "m"}, "qfeats": {1: "m"}}

# ---------------------------------------------------------------- skeleton
os.chdir(UNIRIG)
from src.tokenizer.parse import get_tokenizer
from src.tokenizer.spec import TokenizerConfig
from src.model.unirig_ar import UniRigAR

tok = get_tokenizer(config=TokenizerConfig.parse(config=cfg("tokenizer/tokenizer_parts_articulationxl_256.yaml")))
mc = cfg("model/unirig_ar_350m_1024_81920_float32.yaml")
llm = dict(mc.llm); llm["_attn_implementation"] = "eager"
enc = dict(mc.mesh_encoder); enc["flash"] = False
skel = UniRigAR(llm=Box(llm), mesh_encoder=enc, tokenizer=tok).eval()
skel.load_state_dict(load_state(os.path.join(WEIGHTS, "skeleton/articulation-xl_quantization_256/model.ckpt")))
with torch.no_grad():
    export(ShapeEncoder(skel.mesh_encoder.encoder, skel.output_proj).eval(), encoder_args(),
           f"{OUT}/unirig_skel_encoder.onnx", ["pc", "feats", "qpc", "qfeats"], ["cond"],
           dict(ENC_DYN, cond={1: "m"}))
inits = [numpy_helper.from_array(v.detach().float().contiguous().numpy(), name=k)
         for k, v in skel.transformer.state_dict().items()]
g = helper.make_graph([], "unirig_opt_weights", [], [], initializer=inits)
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="flint-unirig-export"),
          f"{OUT}/unirig_skel_opt.onnx")
print("unirig_skel_opt.onnx tensors", len(inits))
del skel

# ---------------------------------------------------------------- skin
from src.model.unirig_skin import UniRigSkin

mc = cfg("model/unirig_skin.yaml")
kw = {k: v for k, v in mc.items() if k not in ("__target__", "mesh_encoder", "global_encoder")}
me = dict(mc.mesh_encoder); ge = dict(mc.global_encoder); ge["flash"] = False
model = UniRigSkin(mesh_encoder=me, global_encoder=ge, **kw).eval()
model.load_state_dict(load_state(os.path.join(WEIGHTS, "skin/articulation-xl/model.ckpt")), strict=False)


def subm(conv, x, nbr):
    """spconv SubMConv3d as gather + matmul: out[p] = sum_k W[:, k, :] . x[nbr[p, k]] (index N = empty)."""
    N, C = x.shape
    K = nbr.shape[1]
    xp = torch.cat([x, torch.zeros(1, C, dtype=x.dtype)], 0)
    g = xp[nbr].reshape(N, K * C)
    y = g @ conv.weight.reshape(conv.weight.shape[0], K * C).t()
    if conv.bias is not None:
        y = y + conv.bias
    return y


def mha(m, x, kv):
    L = x.shape[0]; S = kv.shape[0]; H, D = m.num_heads, m.head_dim
    q = m.Wq(x).view(L, H, D)
    k, v = m.Wkv(kv).view(S, 2, H, D).unbind(1)
    a = (torch.einsum("lhd,shd->hls", q, k) / math.sqrt(D)).softmax(-1)
    return m.out_proj(torch.einsum("hls,shd->lhd", a, v).reshape(L, H * D))


def rca(blk, q, kv):
    x = blk.norm1(q + mha(blk.attention, q, kv))
    return blk.norm2(x + blk.ffn(x))


class Body(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, feat, nbr3, nbr5, order, inverse, bones, glat):
        m = self.m
        pt = m.mesh_encoder
        emb = pt.embedding
        res = emb.res_linear(feat)
        x = emb.stem.act(emb.stem.norm(subm(emb.stem.conv, feat, nbr5))) + res
        for s in range(len(pt.enc._modules)):
            for name, mod in pt.enc._modules[f"enc{s}"]._modules.items():
                if not name.startswith("block"):
                    x = mod(x)
                    continue
                cpe = mod.cpe
                y = cpe._modules["2"](cpe._modules["1"](subm(cpe._modules["0"], x, nbr3)))
                x = x + y
                sc = x
                h = mod.norm1._modules["0"](x)
                at = mod.attn
                H = at.num_heads; C = at.channels; D = C // H; P = at.patch_size
                oi = at.order_index
                q, k, v = at.qkv(h)[order[oi]].view(-1, 3, H, D).unbind(1)
                q = at.qknorm.norm(q).reshape(-1, P, H, D); k = at.qknorm.norm(k).reshape(-1, P, H, D)
                v = v.reshape(-1, P, H, D)
                a = (torch.einsum("blhd,bmhd->bhlm", q, k) * at.scale).softmax(-1)
                o = torch.einsum("bhlm,bmhd->blhd", a, v).reshape(-1, C)
                x = sc + at.proj(o[inverse[oi]])
                x = x + mod.mlp._modules["0"](mod.norm2._modules["0"](x))
        mfeat = m.feat_map(x)
        be = m.bone_encoder
        bf = be.bone_encoder(bones)
        lat = torch.cat([bf, glat], 0)
        for blk in be.attn:
            bf = rca(blk, bf, lat)
        lat = torch.cat([bf, glat], 0)
        for blk in m.mesh_bone_attn:
            mfeat = rca(blk, mfeat, lat)
        J = bones.shape[0]
        return mfeat, m.kmesh(bf).view(J, m.num_heads, m.feat_dim).transpose(0, 1)


class Head(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, mfeat, bonek, vskin):
        m = self.m
        n = mfeat.shape[0]; H = m.num_heads
        q = m.qmesh(mfeat).view(n, H, m.feat_dim).transpose(0, 1)
        a = torch.softmax(torch.matmul(q, bonek.transpose(1, 2)) / math.sqrt(m.feat_dim), -1)
        a = m.attn_skin_norm(a.permute(1, 2, 0))
        e = m.voxel_skin_norm(m.voxel_skin_embed(vskin.unsqueeze(-1)))
        pred = m.skinweight_pred(m.downscale(torch.cat([a, e], -1))).squeeze(-1)
        return torch.softmax(pred, -1)


with torch.no_grad():
    N, M, J = 2048, 256, 5
    export(ShapeEncoder(model.global_encoder.encoder, model.out_proj).eval(), encoder_args(N, M),
           f"{OUT}/unirig_skin_global.onnx", ["pc", "feats", "qpc", "qfeats"], ["glat"],
           dict(ENC_DYN, glat={1: "m"}))
    perm = torch.stack([torch.randperm(N) for _ in range(4)])
    export(Body(model).eval(),
           (torch.randn(N, 9), torch.randint(0, N + 1, (N, 27)), torch.randint(0, N + 1, (N, 125)), perm,
            torch.argsort(perm, 1), torch.rand(J, 3), torch.randn(M, 768)),
           f"{OUT}/unirig_skin_body.onnx", ["feat", "nbr3", "nbr5", "order", "inverse", "bones", "glat"],
           ["mfeat", "bonek"],
           {"feat": {0: "n"}, "nbr3": {0: "n"}, "nbr5": {0: "n"}, "order": {1: "n"}, "inverse": {1: "n"},
            "bones": {0: "j"}, "glat": {0: "m"}, "mfeat": {0: "n"}, "bonek": {1: "j"}})
    export(Head(model).eval(), (torch.randn(64, 768), torch.randn(16, J, 768), torch.rand(64, J)),
           f"{OUT}/unirig_skin_head.onnx", ["mfeat", "bonek", "vskin"], ["pred"],
           {"mfeat": {0: "n"}, "bonek": {1: "j"}, "vskin": {0: "n", 1: "j"}, "pred": {0: "n", 1: "j"}})
print("done:", OUT)
