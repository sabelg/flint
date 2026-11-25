import torch
def segment_csr(src, indptr, reduce="sum"):
    ip = indptr.tolist()
    outs = []
    for s, e in zip(ip[:-1], ip[1:]):
        seg = src[s:e]
        outs.append({"min": lambda t: t.min(0).values, "max": lambda t: t.max(0).values,
                     "sum": lambda t: t.sum(0), "mean": lambda t: t.mean(0)}[reduce](seg))
    return torch.stack(outs)
