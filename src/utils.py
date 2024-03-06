from pathlib import Path

import torch.nn.functional as F


SRC_ROOT = Path(__file__).parent
ROOT = (SRC_ROOT / '..').resolve()


def normalize(input, dim=1, eps=1e-7):
    return F.normalize(input, dim=dim, eps=eps)
