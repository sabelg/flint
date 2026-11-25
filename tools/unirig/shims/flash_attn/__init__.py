"""Pure-torch stand-in for flash_attn (fp32 maths) used as the UniRig reference."""
import torch

def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen=None, dropout_p=0.0, softmax_scale=None, **kw):
    qkv = qkv.float()
    out = torch.empty(qkv.shape[0], qkv.shape[2], qkv.shape[3])
    cu = cu_seqlens.tolist()
    for s, e in zip(cu[:-1], cu[1:]):
        q, k, v = qkv[s:e, 0], qkv[s:e, 1], qkv[s:e, 2]            # [L, H, D]
        a = torch.einsum("lhd,mhd->hlm", q, k) * softmax_scale
        a = a.softmax(-1)
        out[s:e] = torch.einsum("hlm,mhd->lhd", a, v)
    return out
