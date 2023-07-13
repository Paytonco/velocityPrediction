import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset


def bifurcation_pnts(num_pnts, epsilon):
    x0 = torch.zeros(num_pnts, 2)
    branch = 2 * torch.bernoulli(.5 * torch.ones(num_pnts)) - 1
    x0[:, 1] = 1 + epsilon * (.5 * torch.rand(num_pnts) + .5) * branch
    x = x0.clone()
    t = 6 * torch.rand(num_pnts)
    x[:, 0] = t
    x[:, 1] = (x0[:, 1] - 1) * torch.exp(.1 * t**2 + x0[:, 0] * t) + 1

    v = torch.stack((
        torch.ones(x.size(0)),
        .2 * x[:, 0] * (x[:, 1] - 1)
    )).T

    return t, x, v


def bifurcation(size, num_neighbors):
    num_pnts = num_neighbors + 1
    t, x, v = bifurcation_pnts(size * num_pnts)
    isolated_nodes_graph = Data(x=x, pos=t, y=v)
    clusters_graph = T.KNNGraph(num_neighbors)(isolated_nodes_graph)


def simple_pnts(num_pnts, epsilon):
    x0 = torch.zeros(num_pnts, 2)
    branch = 2 * torch.bernoulli(.5 * torch.ones(num_pnts)) - 1
    x0[:, 1] = 3 + epsilon * (.5 * torch.rand(num_pnts) + .5)
    x = x0.clone()
    t = 3 * torch.rand(num_pnts)
    x[:, 0] = t
    x[:, 1] = (x0[:, 1] - 1) * torch.exp(.1 * t**2 + x0[:, 0] * t) + 1

    v = torch.stack((
        torch.ones(x.size(0)),
        .2 * x[:, 0] * (x[:, 1] - 1)
    )).T

    return t, x, v

def oscillation_pnts(num_pnts, epsilon):
    x0 = epsilon * (2 * torch.rand(num_pnts, 2) - 1)
    x0[:, 0] += 1
    x0[:, 1] = 0
    x = x0.clone()
    t = 2 * np.pi * torch.rand(num_pnts)
    x[:, 0] = x0[:, 0] * torch.cos(t) + x0[:, 1] * torch.sin(t)
    x[:, 1] = -x0[:, 0] * torch.sin(t) + x0[:, 1] * torch.cos(t)

    v = torch.stack((x[:, 1], -x[:, 0])).T

    return t, x, v
