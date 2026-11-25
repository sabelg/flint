import torch.nn as nn
class DropPath(nn.Identity):
    def __init__(self, *a, **k):
        super().__init__()
