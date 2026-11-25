import torch.nn as nn


class DropPath(nn.Identity):
    """Inference stand-in (drop path is the identity in eval mode)."""

    def __init__(self, *a, **k):
        super().__init__()
