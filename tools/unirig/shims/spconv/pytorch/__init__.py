"""Pure-torch stand-in for spconv.pytorch (SubMConv3d only), used as the UniRig reference.

Semantics: correlation like torch.nn.Conv3d. out[p] = sum_k W[:, k, :] . in[p + (k - c)],
weight layout [out, kD, kH, kW, in] (spconv 2 KRSC), offsets along indices[:, 1:4].
Duplicate voxels: the first point in a voxel answers neighbour lookups.
"""
import torch
from torch import nn

NEIGHBOR_HOOK = {}   # key -> LongTensor [N, K] (optional override, e.g. from Mojo)

def neighbor_table(indices: torch.Tensor, k: int) -> torch.Tensor:
    idx = indices.long()
    b, x, y, z = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    r = k // 2
    S = 1 << 20
    key = ((b * S + (x + r)) * S + (y + r)) * S + (z + r)
    # first point wins per voxel
    order = torch.arange(idx.shape[0])
    uniq, inv = torch.unique(key, return_inverse=True)
    first = torch.full((uniq.shape[0],), idx.shape[0], dtype=torch.long)
    first = first.scatter_reduce(0, inv, order, reduce="amin")
    out = []
    for a in range(k):
        for bb in range(k):
            for c in range(k):
                q = ((b * S + (x + a)) * S + (y + bb)) * S + (z + c)
                pos = torch.searchsorted(uniq, q)
                pos = pos.clamp(max=uniq.shape[0] - 1)
                hit = uniq[pos] == q
                out.append(torch.where(hit, first[pos], torch.full_like(pos, -1)))
    return torch.stack(out, dim=1)


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, indice_dict=None):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.indice_dict = {} if indice_dict is None else indice_dict

    def replace_feature(self, feature):
        return SparseConvTensor(feature, self.indices, self.spatial_shape, self.batch_size, self.indice_dict)


class SubMConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, indice_key=None, **kw):
        super().__init__()
        self.k = kernel_size
        self.indice_key = indice_key
        self.weight = nn.Parameter(torch.zeros(out_channels, kernel_size, kernel_size, kernel_size, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        key = (self.indice_key, self.k)
        nbr = NEIGHBOR_HOOK.get(key)
        if nbr is None:
            nbr = x.indice_dict.get(key)
            if nbr is None:
                nbr = neighbor_table(x.indices, self.k)
                x.indice_dict[key] = nbr
        f = x.features
        N, C = f.shape
        K = nbr.shape[1]
        pad = torch.cat([f, torch.zeros(1, C, dtype=f.dtype)], 0)
        g = pad[torch.where(nbr < 0, torch.full_like(nbr, N), nbr)]          # [N, K, C]
        w = self.weight.reshape(self.weight.shape[0], K, C)                 # [O, K, C]
        out = torch.einsum("nkc,okc->no", g, w)
        if self.bias is not None:
            out = out + self.bias
        return x.replace_feature(out)


class modules:
    @staticmethod
    def is_spconv_module(m):
        return isinstance(m, SubMConv3d)
