"""Writes tiny_fused.onnx + inputs/outputs (onnxruntime reference) for test_flint_v5_ml_onnx.mojo.

The graph has the patterns flint.ml.onnx fuses: a sparse-conv neighbour Gather -> Reshape ->
MatMul(Transpose(Reshape(weight))) chain, BatchNormalization, and Einsum -> Mul -> Softmax -> Einsum
attention.  usage: python make_tiny_fused.py (writes next to this file)
"""
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(3)
N, C, K, CO, H, D, P = 70, 8, 27, 16, 2, 8, 35


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(CO, K, C) * 0.2)
        self.bn = torch.nn.BatchNorm1d(CO).eval()
        self.bn.running_mean.uniform_(-0.5, 0.5); self.bn.running_var.uniform_(0.5, 2.0)
        self.bn.weight.data.uniform_(0.5, 1.5); self.bn.bias.data.uniform_(-0.2, 0.2)
        self.qkv = torch.nn.Linear(CO, 3 * H * D)

    def forward(self, x, nbr):
        xp = torch.cat([x, torch.zeros(1, x.shape[1])], 0)
        g = xp[nbr].reshape(x.shape[0], -1)
        y = g @ self.w.reshape(self.w.shape[0], -1).t()
        y = self.bn(y)
        q, k, v = self.qkv(y).view(-1, 3, H, D).unbind(1)
        q = q.reshape(-1, P, H, D); k = k.reshape(-1, P, H, D); v = v.reshape(-1, P, H, D)
        a = (torch.einsum("blhd,bmhd->bhlm", q, k) * (D ** -0.5)).softmax(-1)
        o = torch.einsum("bhlm,bmhd->blhd", a, v).reshape(-1, H * D)
        return o + y


m = Tiny().eval()
x = torch.randn(N, C)
nbr = torch.randint(0, N + 1, (N, K))
path = os.path.join(HERE, "tiny_fused.onnx")
torch.onnx.export(m, (x, nbr), path, input_names=["x", "nbr"], output_names=["y"], opset_version=17,
                  do_constant_folding=False, dynamo=False)
import onnxruntime as ort
y = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy(), "nbr": nbr.numpy()})[0]
assert np.abs(y - m(x, nbr).detach().numpy()).max() < 1e-4
x.numpy().astype(np.float32).tofile(os.path.join(HERE, "tiny_fused_x.f32"))
nbr.numpy().astype(np.int64).tofile(os.path.join(HERE, "tiny_fused_nbr.i64"))
y.astype(np.float32).tofile(os.path.join(HERE, "tiny_fused_y.f32"))
import onnx
print("ops", sorted({n.op_type for n in onnx.load(path).graph.node}))
