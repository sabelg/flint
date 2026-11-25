import torch


def fps(pos, batch=None, ratio=0.5, random_start=True):
    """Farthest point sampling as torch_cluster.fps, but always starting at each batch's first
    point (torch_cluster picks a random start). `flint.ml.make_it_animatable.farthest_points`
    implements exactly this, so both sides choose the same latent points. Squared distances are
    summed x + y + z in float32 and ties go to the lowest index, like torch.argmax."""
    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.long)
    out = []
    for b in batch.unique(sorted=True).tolist():
        ids = (batch == b).nonzero().flatten()
        p = pos[ids].float()
        n = p.shape[0]
        m = int(-(-n * ratio // 1))
        sel = torch.empty(m, dtype=torch.long)
        dist = torch.full((n,), float("inf"))
        cur = 0
        for i in range(m):
            sel[i] = cur
            d = p - p[cur]
            d = d * d
            dist = torch.minimum(dist, d[:, 0] + d[:, 1] + d[:, 2])
            cur = int(torch.argmax(dist))
        out.append(ids[sel])
    return torch.cat(out)
