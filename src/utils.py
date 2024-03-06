from pathlib import Path

import torch.nn.functional as F


SRC_DIR = Path(__file__).parent
ROOT_DIR = (SRC_DIR/'..').resolve()
DATA_DIR = ROOT_DIR/'data'


def normalize(input, dim=1, eps=1e-7):
    return F.normalize(input, dim=dim, eps=eps)
