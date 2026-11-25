"""Convert a MagicArticulate / Puppeteer skeleton checkpoint into the two graphs
`flint.ml.magicarticulate` runs, and optionally dump PyTorch reference results.

usage: python tools/magicarticulate/export.py VARIANT REPO MICHELANGELO CKPT OUT_DIR [--ref MESH ...]

  VARIANT       magicarticulate (bone tokens, MagicArticulate checkpoint_trainonv2_{hier,spatial}.pth)
                or puppeteer (joint tokens + target-aware positions, puppeteer_skeleton_w*_diverse_pose.pth)
  REPO          a clone of https://github.com/Seed3D/MagicArticulate or the `skeleton/` folder of
                https://github.com/Seed3D/Puppeteer (imported, never copied into Flint)
  MICHELANGELO  a clone of https://github.com/ChaoyueSong/Michelangelo (the shape encoder; GPL-3.0)
  CKPT          the released .pth (Apache-2.0, Hugging Face Seed3D/MagicArticulate or Seed3D/Puppeteer)
  OUT_DIR       receives
     <variant>_encoder.onnx  Michelangelo perceiver + KL mode + 16-layer latent transformer + the
                             conditioning projections: (pc[1, N, 3], normals[1, N, 3]) -> cond[1, 257, 1024],
                             the complete decoder prefix (cond_embed[0] and, for Puppeteer, the first
                             target-aware position already added)
     <variant>_opt.onnx      OPT-350M weights only (initializers, no graph), Hugging Face names, plus
                             model.decoder.token_embed_positions.weight, model.decoder.cond_embed.weight
                             and (Puppeteer) target_aware_pos_embed
  --ref MESH... for each mesh (GLB / OBJ, Y-up glTF meshes are turned Z-up like Blender's importer),
                runs the original pipeline on the CPU in fp32 (the repo's MeshProcessor, SkeletonGPT.generate
                and save_utils post-processing; only flash-attention is replaced by eager attention with
                an explicit causal mask) and writes into OUT_DIR/ref:
                  NAME.pc.bin      float32 [8192, 6] encoder input (normalised, fp16-rounded like the demo)
                  NAME.cond.bin    float32 [257, 1024] decoder prefix
                  NAME.tokens.txt  generated token ids (first is the predicted BOS)
                  NAME.rig.txt     post-processed joints (normalised [-0.5, 0.5] frame) and parents

Needs torch, onnx, transformers (4.39.3 for MagicArticulate, 4.46.1 for Puppeteer, the versions the
repos pin), einops, omegaconf, trimesh, scipy; --ref also needs mesh2sdf and scikit-image.

Licensing: MagicArticulate, Puppeteer and their weights are Apache-2.0. The shape encoder is
Michelangelo (code GPL-3.0, weights LGPL-3.0), so the exported encoder graph is a GPL-derived file:
keep it outside the Flint repository and do not redistribute it with Flint builds. Puppeteer's skinning
network depends on NVIDIA PartField (non-commercial licence) and is deliberately not exported.
"""
import math
import os
import sys
import tempfile
import types

if len(sys.argv) < 6:
    print(__doc__)
    sys.exit(1)
VARIANT = sys.argv[1]
REPO, MICH, CKPT, OUT = (os.path.abspath(a) for a in sys.argv[2:6])
REFS = sys.argv[sys.argv.index("--ref") + 1:] if "--ref" in sys.argv else []
REFS = [os.path.abspath(r) for r in REFS]
assert VARIANT in ("magicarticulate", "puppeteer"), VARIANT
PUP = VARIANT == "puppeteer"
os.makedirs(OUT, exist_ok=True)

# `third_partys.Michelangelo...` is how both repos import the encoder: point it at the checkout
PKG = tempfile.mkdtemp(prefix="flint_ma_")
os.makedirs(os.path.join(PKG, "third_partys"))
os.symlink(MICH, os.path.join(PKG, "third_partys", "Michelangelo"))
sys.path.insert(0, REPO)
sys.path.insert(0, PKG)

import numpy as np
import onnx
import torch
from omegaconf import OmegaConf
from onnx import helper, numpy_helper

torch.set_grad_enabled(False)

# Michelangelo wraps every block in a custom autograd checkpoint; run the functions directly
import third_partys.Michelangelo.michelangelo.models.modules.checkpoint as _ckpt  # noqa: E402

_ckpt.checkpoint = lambda func, inputs, params, flag: func(*inputs)
from third_partys.Michelangelo.michelangelo.models.tsal.sal_perceiver import AlignedShapeLatentPerceiver  # noqa: E402

for _m in list(sys.modules.values()):    # modules that bound the name before the patch
    if getattr(_m, "__name__", "").startswith("third_partys.Michelangelo") and hasattr(_m, "checkpoint"):
        if callable(getattr(_m, "checkpoint")) and _m is not _ckpt:
            _m.checkpoint = _ckpt.checkpoint

pkg = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
SD = pkg["model"]
ARGS = pkg.get("args")
print("checkpoint args:", ARGS)
del pkg


def sub(prefix):
    return {k[len(prefix):]: v.float() for k, v in SD.items() if k.startswith(prefix)}


# ------------------------------------------------------------------ shape encoder
mcfg = OmegaConf.load(os.path.join(MICH, "configs", "shapevae-256.yaml")) if os.path.exists(
    os.path.join(MICH, "configs", "shapevae-256.yaml")) else None
params = dict(num_latents=256, embed_dim=64, point_feats=3, num_freqs=8, include_pi=False, heads=12, width=768,
              num_encoder_layers=8, num_decoder_layers=16, use_ln_post=True, init_scale=0.25, qkv_bias=False,
              use_checkpoint=False)
if mcfg is not None:
    p = OmegaConf.to_container(mcfg.model.params.shape_module_cfg.params)
    p["use_checkpoint"] = False
    params.update(p)
shape = AlignedShapeLatentPerceiver(device=None, dtype=None, **params).eval()
missing = shape.load_state_dict(sub("point_encoder.model.shape_model."), strict=True)
cond_head = torch.nn.Linear(768, 1024)
cond_head.load_state_dict(sub("cond_head_proj."))
cond_proj = torch.nn.Linear(768, 1024)
cond_proj.load_state_dict(sub("cond_proj."))
cond_embed = SD["transformer.model.decoder.cond_embed.weight"].float()
tape = SD["target_aware_pos_embed"].float() if PUP and "target_aware_pos_embed" in SD else None
SEQ_SHUFFLE = tape is not None


class Encoder(torch.nn.Module):
    """Everything up to the decoder's first input embeddings (SkeletonGPT.process_point_feature)."""

    def __init__(self):
        super().__init__()
        self.shape, self.head, self.proj = shape, cond_head, cond_proj
        self.register_buffer("ce0", cond_embed[0].clone())
        self.register_buffer("tp0", tape[0, 0].clone() if tape is not None else torch.zeros(1024))

    def forward(self, pc, normals):
        e = self.shape.encoder                                   # CrossAttentionEncoder._forward
        data = e.input_proj(torch.cat([e.fourier_embedder(pc), normals], -1))
        x = e.self_attn(e.cross_attn(e.query.unsqueeze(0).expand(pc.shape[0], -1, -1), data))
        x = e.ln_post(x)                                         # [1, 257, 768]
        moments = self.shape.pre_kl(x[:, 1:])
        lat = self.shape.transformer(self.shape.post_kl(moments[..., :moments.shape[-1] // 2]))
        cond = torch.cat([self.head(x[:, :1]), self.proj(lat)], 1)
        return cond + self.ce0 + self.tp0


enc = Encoder().eval()
if not os.path.exists(f"{OUT}/{VARIANT}_encoder.onnx"):     # (kept when only re-running --ref)
    torch.onnx.export(enc, (torch.rand(1, 4096, 3) - 0.5, torch.randn(1, 4096, 3)), f"{OUT}/{VARIANT}_encoder.onnx",
                      input_names=["pc", "normals"], output_names=["cond"],
                      dynamic_axes={"pc": {1: "n"}, "normals": {1: "n"}}, opset_version=17, do_constant_folding=True,
                      dynamo=False)
    g = onnx.load(f"{OUT}/{VARIANT}_encoder.onnx", load_external_data=False)
    print(f"{VARIANT}_encoder.onnx nodes", len(g.graph.node))

# ------------------------------------------------------------------ decoder weights
inits = []
for k, v in SD.items():
    if k.startswith("transformer."):
        inits.append(numpy_helper.from_array(v.float().contiguous().numpy(), name=k[len("transformer."):]))
if tape is not None:
    inits.append(numpy_helper.from_array(tape[0].contiguous().numpy(), name="target_aware_pos_embed"))
g = helper.make_graph([], f"{VARIANT}_opt_weights", [], [], initializer=inits)
if not os.path.exists(f"{OUT}/{VARIANT}_opt.onnx"):
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="flint-magicarticulate-export"),
              f"{OUT}/{VARIANT}_opt.onnx")
print(f"{VARIANT}_opt.onnx tensors", len(inits))

if not REFS:
    print("done:", OUT)
    sys.exit(0)

# ------------------------------------------------------------------ PyTorch reference
import importlib.machinery  # noqa: E402
import transformers  # noqa: E402,F401  (before the stubs: it probes for cv2)

for name in ("cv2", "pyrender", "data_utils.pyrender_wrapper", "data_utils.data_loader"):
    m = types.ModuleType(name)
    m.PyRenderWrapper = m.DataLoader = object
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = m
import trimesh  # noqa: E402
from transformers.models.opt import modeling_opt  # noqa: E402

if PUP:
    from skeleton_models.skeletongen import SkeletonGPT
    from skeleton_models.skeleton_opt import SkeletonOPTConfig as Cfg, SkeletonOPT as LM
else:
    from skeleton_models.skeletongen import SkeletonGPT
    from skeleton_models.shape_opt import ShapeOPTConfig as Cfg, ShapeOPT as LM
from utils.mesh_to_pc import MeshProcessor  # noqa: E402
from utils.save_utils import pred_joints_and_bones, merge_duplicate_joints_and_fix_bones  # noqa: E402
from data_utils.save_npz import normalize_to_unit_cube  # noqa: E402

# flash-attention 2 does causal masking itself; for eager attention pass an explicit mask
_layer_fwd = modeling_opt.OPTDecoderLayer.forward


def _causal_layer(self, hidden_states, *a, **kw):
    if kw.get("attention_mask") is None:
        B, L, _ = hidden_states.shape
        pkv = kw.get("past_key_value")
        P = 0
        if pkv is not None:
            try:
                P = pkv[0].shape[2] if isinstance(pkv, (tuple, list)) else pkv.get_seq_length(self.self_attn.layer_idx)
            except Exception:
                P = pkv[0].shape[2]
        m = torch.full((L, P + L), torch.finfo(hidden_states.dtype).min)
        m = torch.triu(m, diagonal=P + 1)
        kw["attention_mask"] = m[None, None].expand(B, 1, L, P + L)
    return _layer_fwd(self, hidden_states, *a, **kw)


modeling_opt.OPTDecoderLayer.forward = _causal_layer


class PointEncoderShim:
    """The two methods SkeletonGPT calls on Michelangelo's AlignedShapeAsLatentPLModule."""

    def encode_latents(self, surface):
        x, _ = shape.encoder(surface[..., :3], surface[..., 3:6])
        return x

    def to_shape_latents(self, latents):
        kl, _ = shape.encode_kl_embed(latents, sample_posterior=False)
        return shape.decode(kl)


a = types.SimpleNamespace(n_discrete_size=ARGS.n_discrete_size if ARGS is not None and hasattr(ARGS, "n_discrete_size") else 128,
                          n_max_bones=100, llm="facebook/opt-350m", batchsize_per_gpu=1, num_beams=1,
                          joint_token=PUP, seq_shuffle=SEQ_SHUFFLE, hier_order=not PUP)
model = SkeletonGPT.__new__(SkeletonGPT)
torch.nn.Module.__init__(model)
model.args = a
model.point_encoder = PointEncoderShim()
model.cond_length, model.cond_dim = 257, 768
model.n_discrete_size = a.n_discrete_size
model.joint_token = PUP
if PUP:
    model.bone_per_token = 4
    a.n_max_bones += 1
else:
    model.bone_per_token = 6
model.max_length = int(a.n_max_bones * model.bone_per_token + 2 + model.cond_length)
model.pad_id = -1
model.coor_continuous_range = (-0.5, 0.5)
cfg = Cfg.from_pretrained(a.llm, n_positions=model.max_length, max_position_embeddings=model.max_length,
                          vocab_size=a.n_discrete_size + 3)
model.bos_token_id, model.eos_token_id, model.pad_token_id = 0, 1, 2
cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id = 0, 1, 2
cfg.n_discrete_size = a.n_discrete_size
cfg.bone_per_token = model.bone_per_token
cfg.cond_length = model.cond_length
cfg.joint_token = PUP
cfg.word_embed_proj_dim = cfg.hidden_size
cfg._attn_implementation = "eager"
model.config = cfg
model.transformer = LM(cfg)
model.transformer.model.decoder._use_flash_attention_2 = True   # take the repo's mask path; see _causal_layer
model.transformer.load_state_dict({k[len("transformer."):]: v.float() for k, v in SD.items() if k.startswith("transformer.")})
model.cond_head_proj, model.cond_proj = cond_head, cond_proj
if SEQ_SHUFFLE:
    model.feat_dim = cfg.word_embed_proj_dim
    model.target_aware_pos_embed = torch.nn.Parameter(tape.clone())
model.eval()

os.makedirs(f"{OUT}/ref", exist_ok=True)
for path in REFS:
    nm = os.path.splitext(os.path.basename(path))[0]
    mesh = trimesh.load(path, force="mesh")
    if path.lower().endswith((".glb", ".gltf")):
        v = np.asarray(mesh.vertices)
        mesh = trimesh.Trimesh(np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1), mesh.faces, process=False)
    pc = MeshProcessor.convert_meshes_to_point_clouds([mesh], 8192, apply_marching_cubes=True, octree_depth=7)[0]
    coor, center, scale = normalize_to_unit_cube(pc[:, :3].copy(), scale_factor=0.9995)  # fp16, as in demo.py
    pcn = np.concatenate([coor.astype(np.float32), pc[:, 3:].astype(np.float32)], -1).astype(np.float16).astype(np.float32)
    pcn.tofile(f"{OUT}/ref/{nm}.pc.bin")
    t = torch.from_numpy(pcn)[None]
    cond = enc(t[..., :3], t[..., 3:])
    cond[0].numpy().astype(np.float32).tofile(f"{OUT}/ref/{nm}.cond.bin")
    # SkeletonGPT.generate, capturing the raw ids
    gen = model.transformer.generate
    captured = {}

    def _gen(*ga, **gk):
        r = gen(*ga, **gk)
        captured["ids"] = r[0].tolist()
        return r

    model.transformer.generate = _gen
    bones = model.generate({"pc_normal": t})
    model.transformer.generate = gen
    with open(f"{OUT}/ref/{nm}.tokens.txt", "w") as fh:
        fh.write(" ".join(str(i) for i in captured["ids"]) + "\n")
    lines = []
    if bones is None:
        lines.append("failed")
    else:
        joints, pb = pred_joints_and_bones(bones[0].cpu().numpy().squeeze())
        if not PUP:
            root = pb[0][0]
            joints, pb, root = merge_duplicate_joints_and_fix_bones(joints, pb, root_index=root)
            parent = {}
            for p_, c_ in pb:
                if c_ not in parent:
                    parent[c_] = p_
            parent[root] = None
        else:
            joints, pb = merge_duplicate_joints_and_fix_bones(joints, pb)
            parents_set = set(int(b[0]) for b in pb)
            children_set = set(int(b[1]) for b in pb)
            roots = parents_set - children_set
            root = list(roots)[0] if roots else int(pb[0, 0])
            parent = {}
            for p_, c_ in pb:
                if c_ not in parent:
                    parent[c_] = p_
            parent[root] = None
        lines.append(f"joints {len(joints)} root {int(root)}")
        for j in range(len(joints)):
            p_ = parent.get(j, None)
            lines.append(f"{float(joints[j][0]):.6f} {float(joints[j][1]):.6f} {float(joints[j][2]):.6f} "
                         f"{-1 if p_ is None else int(p_)}")
    with open(f"{OUT}/ref/{nm}.rig.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(nm, "tokens", len(captured["ids"]), lines[0])
print("done:", OUT)
