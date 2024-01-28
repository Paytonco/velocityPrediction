import torch.nn.functional as F


def normalize(input, dim=1, eps=1e-7):
    return F.normalize(input, dim=dim, eps=eps)
