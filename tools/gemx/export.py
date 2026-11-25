"""Convert the released GEM-X networks into the three graphs `flint.ml.gemx` runs.

usage: python tools/gemx/export.py GEMX_CHECKOUT WEIGHTS_DIR OUT_DIR [vitpose] [pose] [motion]

  GEMX_CHECKOUT  a clone of https://github.com/NVlabs/GEM-X (its Python code is imported here,
                 never copied into Flint)
  WEIGHTS_DIR    holds vitpose.pth, body_s240000.ckpt and motion_native77_e774_s292175.ckpt from
                 the Hugging Face release nvidia/GEM-X
  OUT_DIR        receives (each graph with its weights in a `<name>.onnx.data` side file)
     gemx_vitpose.onnx  native77 ViTPose (DINOv3 ViT-H/16+ and deconvolution head):
                        img[B, 3, 256, 192] -> heatmaps[B, 77, 64, 48]
     gemx_pose.onnx     Body Pose s240000 (DINOv3 ViT-H/16+ and SOMA body decoder):
                        img[B, 3, 256, 256], cliff[B, 3] -> token[B, 1024]
     gemx_motion.onnx   native77 Body Motion (12-layer temporal transformer):
                        obs[1, L, 77, 3], cliff[1, L, 3], cam_angvel[1, L, 6], imgseq[1, L, 1024],
                        positions[1, L], attn_bias[L, L] -> motion[1, L, 655], pred_cam[1, L, 3],
                        static_conf[1, L, 6]; the codec statistics ride along as the initializers
                        `codec_mean` / `codec_std`
  Naming one or more of vitpose / pose / motion exports only those.

Each graph is the released network re-expressed with the operators Flint's interpreter runs well:
attention as Einsum -> Mul -> Softmax -> Einsum (fused on the GPU), the 16x16 patch embedding as a
MatMul, the heatmap head's transposed convolutions as one 9-tap MatMul per layer with BatchNorm
folded in, and the detector's deliberate bf16 rounding of queries and keys (rotary embedding
included) as Cast(BFLOAT16) -> Cast(FLOAT) pairs. Before saving, every module is checked against
the original PyTorch network on random inputs.

Needs torch, onnx, numpy, einops, scipy, opencv-python-headless, omegaconf and transformers 5.3 (for
the DINOv3 model class); run it with a CPU build of torch for bit-stable reference numbers.

Licensing: the GEM-X code is Apache-2.0 and its weights are under the NVIDIA Open Model License
(commercial use allowed); the two image encoders are DINOv3 derivatives, whose weights additionally
carry Meta's DINOv3 License terms. The exported files are model weights: keep them outside the
Flint repository. The YOLOX person detector of the GEM-X demo is not exported (it is trained on the
non-commercial Human-Art data); Flint takes person boxes from the caller or tracks the 2D keypoints.
"""
import math
import os
import sys

if len(sys.argv) < 4:
    print(__doc__)
    sys.exit(1)
GEMX, WEIGHTS, OUT = (os.path.abspath(a) for a in sys.argv[1:4])
WANT = set(sys.argv[4:]) or {"vitpose", "pose", "motion"}
sys.path.insert(0, GEMX)
os.makedirs(OUT, exist_ok=True)

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torch import nn

torch.set_grad_enabled(False)
OPSET = 17


def bf16(x):
    """Round to bfloat16, keep float32 storage (exports as Cast -> Cast)."""
    return x.to(torch.bfloat16).to(torch.float32)


def save(mod, args, path, names, outs, dyn, extra=None):
    """Trace to ONNX, then move every weight to one external data file."""
    import shutil
    import tempfile
    tmpdir = tempfile.mkdtemp(dir=OUT)  # graphs past 2 GB are traced with one file per weight
    tmp = os.path.join(tmpdir, "graph.onnx")
    torch.onnx.export(mod, args, tmp, input_names=names, output_names=outs, dynamic_axes=dyn,
                      opset_version=OPSET, do_constant_folding=True, dynamo=False)
    model = onnx.load(tmp, load_external_data=True)
    shutil.rmtree(tmpdir)
    for name, value in (extra or {}).items():
        model.graph.initializer.append(onnx.numpy_helper.from_array(np.ascontiguousarray(value, np.float32), name))
    data = os.path.basename(path) + ".data"
    if os.path.exists(path + ".data"):
        os.remove(path + ".data")
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location=data,
                    size_threshold=1024)
    ops = sorted({n.op_type for n in model.graph.node})
    print(f"  wrote {path} ({os.path.getsize(path + '.data') / 1e6:.0f} MB weights); ops: {' '.join(ops)}")


def check(name, a, b, tol):
    err = (a - b).abs().max().item()
    scale = b.abs().max().item()
    print(f"  {name}: max |diff| {err:.3g} (max |ref| {scale:.3g})")
    if not err <= tol * max(1.0, scale):
        raise SystemExit(f"{name}: export module differs from the reference network")


# ------------------------------------------------------------------------------ DINOv3 ViT-H/16+

class DinoExport(nn.Module):
    """The HF DINOv3 encoder with fused-attention-friendly operators. Returns patch features
    [B, P, 1280] after the final LayerNorm."""

    def __init__(self, encoder, height, width, detector_bf16):
        super().__init__()
        self.enc = encoder
        self.hp, self.wp = height // 16, width // 16
        self.bf16 = detector_bf16
        cfg = encoder.config
        self.heads = cfg.num_attention_heads
        self.hd = cfg.hidden_size // self.heads
        self.prefix = 1 + cfg.num_register_tokens
        pe = encoder.embeddings.patch_embeddings
        self.register_buffer("w_patch", pe.weight.reshape(pe.weight.shape[0], -1).t().contiguous())
        self.register_buffer("b_patch", pe.bias.detach().clone())
        self.register_buffer("prefix_tokens", torch.cat((encoder.embeddings.cls_token,
                                                         encoder.embeddings.register_tokens), 1))
        # the rotary tables depend only on the image size: take them from the reference module
        cos, sin = encoder.rope_embeddings(torch.zeros(1, 3, height, width))
        self.register_buffer("cos", cos.float()[:, None, :].contiguous())  # [P, 1, D]
        self.register_buffer("sin", sin.float()[:, None, :].contiguous())
        for i, layer in enumerate(encoder.layer):
            at = layer.attention
            w = torch.cat((at.q_proj.weight, at.k_proj.weight, at.v_proj.weight))
            kb = at.k_proj.bias if at.k_proj.bias is not None else torch.zeros_like(at.q_proj.bias)
            b = torch.cat((at.q_proj.bias, kb, at.v_proj.bias))
            self.register_buffer(f"wqkv{i}", w.contiguous())
            self.register_buffer(f"bqkv{i}", b.contiguous())

    def rope(self, x):
        """x [B, N, H, D]: rotate the patch tokens (rotate_half layout)."""
        pre, pat = x[:, :self.prefix], x[:, self.prefix:]
        half = self.hd // 2
        rot = torch.cat((-pat[..., half:], pat[..., :half]), -1)
        if self.bf16:
            pat = bf16(bf16(pat * self.cos) + bf16(rot * self.sin))
        else:
            pat = pat * self.cos + rot * self.sin
        return torch.cat((pre, pat), 1)

    def forward(self, img):
        b = img.shape[0]
        x = img.reshape(b, 3, self.hp, 16, self.wp, 16).permute(0, 2, 4, 1, 3, 5).reshape(b, self.hp * self.wp, 768)
        x = x @ self.w_patch + self.b_patch
        x = torch.cat((self.prefix_tokens.expand(b, -1, -1), x), 1)
        n = x.shape[1]
        for i, layer in enumerate(self.enc.layer):
            h = layer.norm1(x)
            qkv = F.linear(h, getattr(self, f"wqkv{i}"), getattr(self, f"bqkv{i}"))
            q, k, v = qkv.reshape(b, n, 3, self.heads, self.hd).unbind(2)
            if self.bf16:
                q, k = bf16(q), bf16(k)
            q, k = self.rope(q), self.rope(k)
            a = (torch.einsum("blhd,bmhd->bhlm", q, k) * (self.hd ** -0.5)).softmax(-1)
            o = torch.einsum("bhlm,bmhd->blhd", a, v).reshape(b, n, self.heads * self.hd)
            x = x + layer.layer_scale1(layer.attention.o_proj(o))
            h = layer.norm2(x)
            m = layer.mlp
            x = x + layer.layer_scale2(m.down_proj(F.silu(m.gate_proj(h)) * m.up_proj(h)))
        x = self.enc.norm(x)
        return x[:, self.prefix:]


# ------------------------------------------------------------------------------------ ViTPose

def deconv_taps(weight, bn):
    """ConvTranspose2d(k=4, s=2, p=1, no bias) + BatchNorm as one MatMul over the 3x3 neighbourhood
    of the zero-padded input: [9*Cin, 4*Cout] (columns: row phase, column phase, channel) and a
    bias [4*Cout]."""
    cin, cout = weight.shape[:2]
    taps = {0: ((1, 1), (0, 3)), 1: ((2, 0), (1, 2))}  # phase -> ((padded offset, kernel index), ...)
    w9 = torch.zeros(3, 3, cin, 2, 2, cout, dtype=torch.float64)
    for a in (0, 1):
        for oy, ky in taps[a]:
            for c in (0, 1):
                for ox, kx in taps[c]:
                    w9[oy, ox, :, a, c, :] += weight[:, :, ky, kx].double()
    scale = (bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps))
    w9 = w9 * scale
    bias = (bn.bias.double() - bn.running_mean.double() * scale).repeat(4)
    return w9.reshape(9 * cin, 4 * cout).float(), bias.float()


class DeconvExport(nn.Module):
    def __init__(self, deconv, bn):
        super().__init__()
        w, b = deconv_taps(deconv.weight, bn)
        self.register_buffer("w", w)
        self.register_buffer("b", b)
        self.cout = deconv.weight.shape[1]

    def forward(self, x):  # NHWC -> NHWC (2x), ReLU applied
        b, h, w, c = x.shape
        zr = torch.zeros(b, 1, w, c)
        x = torch.cat((zr, x, zr), 1)
        zc = torch.zeros(b, h + 2, 1, c)
        x = torch.cat((zc, x, zc), 2)
        cols = [x[:, dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]
        y = torch.cat(cols, -1) @ self.w + self.b
        y = y.reshape(b, h, w, 2, 2, self.cout).permute(0, 1, 3, 2, 4, 5).reshape(b, 2 * h, 2 * w, self.cout)
        return torch.relu(y)


class VitPoseExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vit = DinoExport(model.backbone.encoder, 256, 192, detector_bf16=True)
        head = model.keypoint_head.deconv_layers
        self.d1 = DeconvExport(head[0], head[1])
        self.d2 = DeconvExport(head[3], head[4])
        fl = model.keypoint_head.final_layer
        self.register_buffer("wf", fl.weight[:, :, 0, 0].t().contiguous())
        self.register_buffer("bf", fl.bias.detach().clone())

    def forward(self, img):
        b = img.shape[0]
        x = self.vit(img).reshape(b, 16, 12, 1280)
        x = self.d2(self.d1(x))
        return (x @ self.wf + self.bf).permute(0, 3, 1, 2)


def export_vitpose():
    from gem.common.pose2d.native77 import load_native77_detector
    print("ViTPose native77")
    det = load_native77_detector(os.path.join(WEIGHTS, "vitpose.pth"), release_id="vitpose77-dinov3-huge-metrosim-256x192-v1")
    model = det.model.float().eval()
    exp = VitPoseExport(model).eval()
    x = torch.randn(2, 3, 256, 192)
    # eager attention: the same matmul / softmax arithmetic as the export, so the deliberate bf16
    # rounding of q / k sees bit-identical inputs (SDPA differs in the last bit and flips roundings)
    model.backbone.encoder.config._attn_implementation = "eager"
    ref = model(x)
    model.backbone.encoder.config._attn_implementation = "sdpa"
    check("heatmaps", exp(x), ref, 2e-4)
    save(exp, (x,), os.path.join(OUT, "gemx_vitpose.onnx"), ["img"], ["heatmaps"],
         {"img": {0: "batch"}, "heatmaps": {0: "batch"}})


# ------------------------------------------------------------------------------------ Body Pose

class PoseExport(nn.Module):
    """Body Pose s240000 up to the final decoder body query (`ImageCondition.pose_tokens`)."""

    def __init__(self, model):
        super().__init__()
        self.vit = DinoExport(model.backbone.encoder, 256, 256, detector_bf16=False)
        head = model.task.head
        if not head.per_layer_predictions or not head.use_init_to_token or head.ray_mlp is not None \
                or head.global_scale_b_query or head.global_scale_sigma_query:
            raise SystemExit("unexpected Body Pose head configuration")
        self.head = head
        dec = head.transformer
        self.heads = dec.transformer.layers[0][0].fn.heads
        self.register_buffer("init", torch.cat((head.init_pose, head.init_shape, head.init_cam), -1).reshape(1, -1))

    def attend(self, q, k, v):
        b, l, _ = q.shape
        s = k.shape[1]
        h = self.heads
        q = q.reshape(b, l, h, -1)
        k = k.reshape(b, s, h, -1)
        v = v.reshape(b, s, h, -1)
        a = (torch.einsum("blhd,bmhd->bhlm", q, k) * (q.shape[-1] ** -0.5)).softmax(-1)
        return torch.einsum("bhlm,bmhd->blhd", a, v).reshape(b, l, -1)

    def forward(self, img, cliff):
        b = img.shape[0]
        feat = self.vit(img)
        tok = torch.cat((self.init.expand(b, -1), cliff), -1)
        dec = self.head.transformer
        x = dec.to_token_embedding(self.head.init_to_token(tok)[:, None, :]) + dec.pos_embedding[:, :1]
        for sa, ca, ff in dec.transformer.layers:
            h = sa.norm(x)
            q, k, v = sa.fn.to_qkv(h).chunk(3, -1)
            x = sa.fn.to_out(self.attend(q, k, v)) + x
            h = ca.norm(x)
            k, v = ca.fn.to_kv(feat).chunk(2, -1)
            x = ca.fn.to_out(self.attend(ca.fn.to_q(h), k, v)) + x
            x = ff.fn(ff.norm(x)) + x
        return x[:, 0]


def export_pose():
    from gem.pose.api import body_config, load_pose_checkpoint
    print("Body Pose s240000")
    model = load_pose_checkpoint(os.path.join(WEIGHTS, "body_s240000.ckpt"), body_config()).float().eval()
    exp = PoseExport(model).eval()
    img = torch.randn(2, 3, 256, 256)
    bbx = torch.tensor([[700.0, 400.0, 500.0], [600.0, 300.0, 350.0]])
    f = math.hypot(1280, 720)
    K = torch.tensor([[f, 0, 640.0], [0, f, 360.0], [0, 0, 1]]).expand(2, 3, 3)
    cliff = model.conditioning({"bbx_xys": bbx, "K_fullimg": K})
    ref = model.encode({"img": img, "bbx_xys": bbx, "K_fullimg": K}).pose_tokens
    check("pose token", exp(img, cliff), ref, 2e-4)
    save(exp, (img, cliff), os.path.join(OUT, "gemx_pose.onnx"), ["img", "cliff"], ["token"],
         {"img": {0: "batch"}, "cliff": {0: "batch"}, "token": {0: "batch"}})


# ----------------------------------------------------------------------------------- Body Motion

class MotionExport(nn.Module):
    """R3MatchedGEMNetwork for one sequence. Rotary positions come from `positions`; frames further
    apart than the trained window are masked through `attn_bias` (0 or -1e30)."""

    def __init__(self, net, cam_mean, cam_std):
        super().__init__()
        self.net = net
        core = net.denoiser
        self.core = core
        self.register_buffer("cam_mean", cam_mean)
        self.register_buffer("cam_std", cam_std)
        rope = core.blocks[0].attn.rope
        self.register_buffer("inv_freq2", rope.inv_freq.float().repeat_interleave(2))
        self.register_buffer("t999", core.embed_timestep.sequence_pos_encoder.pe[999].reshape(1, 1, -1).clone())

    @staticmethod
    def rotate(x, cos, sin):  # x [1, H, L, D], interleaved pairs
        p = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
        r = torch.cat((-p[..., 1:2], p[..., 0:1]), -1).reshape(x.shape)
        return x * cos + r * sin

    def forward(self, obs, cliff, cam_angvel, imgseq, positions, attn_bias):
        net, core = self.net, self.core
        b, l = obs.shape[:2]
        visible = obs[..., 2:3] > 0.5
        coords = torch.where(visible, obs[..., :2], torch.zeros_like(obs[..., :2]))
        enc = torch.where(visible, net.learned_pos_linear(coords), net.learned_pos_params).reshape(b, l, -1)
        cond = net.embed_noisyobs(enc) + core.cliffcam_embedder(cliff)
        cond = cond + core.cam_angvel_embedder((cam_angvel - self.cam_mean) / self.cam_std)
        cond = cond + core.imgseq_embedder(imgseq)
        x = cond + core.embed_timestep.time_embed(self.t999)
        # add_cond_linear over [x, zeros(655)]: the zero half contributes nothing
        acl = core.add_cond_linear
        x = x @ acl.weight[:, :x.shape[-1]].t() + acl.bias
        enc_pos = positions[..., None] * self.inv_freq2  # [1, L, 64]
        cos, sin = enc_pos.cos()[:, None], enc_pos.sin()[:, None]
        for blk in core.blocks:
            h = blk.norm1(x)
            at = blk.attn
            nh = at.num_heads
            q = at.query(h).reshape(b, l, nh, -1).transpose(1, 2)
            k = at.key(h).reshape(b, l, nh, -1).transpose(1, 2)
            v = at.value(h).reshape(b, l, nh, -1).transpose(1, 2)
            q, k = self.rotate(q, cos, sin), self.rotate(k, cos, sin)
            s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(at.head_dim)) + attn_bias
            o = (s.softmax(-1) @ v).transpose(1, 2).reshape(b, l, -1)
            x = x + blk.gate_msa * at.proj(o)
            x = x + blk.gate_mlp * blk.mlp(blk.norm2(x))
        cam = core.pred_cam_head(x) * core.pred_cam_std + core.pred_cam_mean
        cam = torch.cat((cam[..., :1].clamp_min(0.25), cam[..., 1:]), -1)
        return core.final_layer(x), cam, core.static_conf_head(x)


def export_motion():
    import yaml
    from gem.inference.motion import load_predictor
    print("Body Motion native77")
    with open(os.path.join(WEIGHTS, "body_video_native77.yaml")) as f:
        cfg = yaml.safe_load(f)["motion"]
    pred = load_predictor(os.path.join(WEIGHTS, "motion_native77_e774_s292175.ckpt"), cfg,
                          stats_path=os.path.join(WEIGHTS, "motion_stats_raw.npz"))
    pipe = pred.pipeline.pipeline
    net = pipe.denoiser3d.eval()
    exp = MotionExport(net, pipe.cam_angvel_mean.clone(), pipe.cam_angvel_std.clone()).eval()
    for L in (40, 150):
        g = torch.Generator().manual_seed(L)
        obs = torch.rand(1, L, 77, 3, generator=g) * 2 - 1
        obs[..., 2] = torch.rand(1, L, 77, generator=g)
        cliff = torch.randn(1, L, 3, generator=g) * 0.2
        cam = torch.tensor([1.0, 0, 0, 0, 1, 0]).expand(1, L, 6) + torch.randn(1, L, 6, generator=g) * 0.01
        img = torch.randn(1, L, 1024, generator=g)
        pos = torch.arange(L, dtype=torch.float32)[None]
        ref = net(length=torch.tensor([L]), obs=obs, f_cliffcam=cliff,
                  f_cam_angvel=(cam - pipe.cam_angvel_mean) / pipe.cam_angvel_std, f_imgseq=img, positions=pos)
        bias = torch.zeros(L, L)
        if L > core_max(net):
            m = net.denoiser.max_len
            for i in range(L):
                start = min(L - m, max(0, i - m // 2))
                stop = max(m, min(L, i + m // 2))
                bias[i] = -1e30
                bias[i, start:stop] = 0
        out, pcam, conf = exp(obs, cliff, cam, img, pos, bias)
        # the reference averages identity / log-scale over the clip; flint.ml.gemx does that after the graph
        out[..., 456:640] = out[..., 456:640].mean(1, keepdim=True)
        check(f"motion L={L}", out, ref["pred_x"], 1e-4)
        check(f"pred_cam L={L}", pcam, ref["pred_cam"], 1e-4)
        check(f"static_conf L={L}", conf, ref["static_conf_logits"], 1e-4)
    codec = pipe.endecoder
    save(exp, (obs, cliff, cam, img, pos, bias), os.path.join(OUT, "gemx_motion.onnx"),
         ["obs", "cliff", "cam_angvel", "imgseq", "positions", "attn_bias"], ["motion", "pred_cam", "static_conf"],
         {"obs": {1: "len"}, "cliff": {1: "len"}, "cam_angvel": {1: "len"}, "imgseq": {1: "len"},
          "positions": {1: "len"}, "attn_bias": {0: "len", 1: "len"}, "motion": {1: "len"},
          "pred_cam": {1: "len"}, "static_conf": {1: "len"}},
         extra={"codec_mean": codec.mean.numpy(), "codec_std": codec.std.numpy()})


def core_max(net):
    return net.denoiser.max_len


if "motion" in WANT:
    export_motion()
if "vitpose" in WANT:
    export_vitpose()
if "pose" in WANT:
    export_pose()
