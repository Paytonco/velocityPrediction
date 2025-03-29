from pathlib import Path

import torch.nn.functional as F


DIR_SRC = Path(__file__).parent
DIR_ROOT = (DIR_SRC/'..'/'..').resolve()
DIR_DATA = DIR_ROOT/'data'

HYDRA_INIT = dict(version_base=None, config_path='../../conf', config_name='conf')


def normalize(input, dim=1, eps=1e-7):
    return F.normalize(input, dim=dim, eps=eps)
