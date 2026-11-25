import math, torch
from torch import nn

class MHA(nn.Module):
    """flash_attn.modules.mha.MHA, cross-attention subset (Wq, Wkv, out_proj)."""
    def __init__(self, embed_dim, num_heads, cross_attn=False, bias=True, **kw):
        super().__init__()
        assert cross_attn
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x, x_kv=None, **kw):
        B, L, _ = x.shape
        S = x_kv.shape[1]
        H, D = self.num_heads, self.head_dim
        q = self.Wq(x).view(B, L, H, D)
        kv = self.Wkv(x_kv).view(B, S, 2, H, D)
        k, v = kv[:, :, 0], kv[:, :, 1]
        a = torch.einsum("blhd,bshd->bhls", q, k) / math.sqrt(D)
        a = a.softmax(-1)
        o = torch.einsum("bhls,bshd->blhd", a, v).reshape(B, L, H * D)
        return self.out_proj(o)
