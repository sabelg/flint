import torch
def fps(pos, batch=None, ratio=0.5, random_start=False):
    """Farthest point sampling, start at the first point of each batch (deterministic)."""
    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.long)
    out = []
    for b in batch.unique(sorted=True).tolist():
        ids = (batch == b).nonzero().flatten()
        p = pos[ids]
        n = p.shape[0]
        m = int(-(-n * ratio // 1)) if ratio * n != int(ratio * n) else int(ratio * n)
        sel = torch.empty(m, dtype=torch.long)
        dist = torch.full((n,), float("inf"))
        cur = 0
        for i in range(m):
            sel[i] = cur
            d = ((p - p[cur]) ** 2).sum(-1)
            dist = torch.minimum(dist, d)
            cur = int(torch.argmax(dist))
        out.append(ids[sel])
    return torch.cat(out)
