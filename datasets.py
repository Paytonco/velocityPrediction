import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

torch.set_default_dtype(torch.float64)


class KNNGraph(T.KNNGraph):
    def __init__(self, attr, *args, **kwargs):
        self.attr = attr
        super().__init__(*args, **kwargs)

    def forward(self, data):
        pos = data.pos
        data.pos = data[self.attr]
        data = super()(data)
        data.pos = pos

        return data


class NeighborsDataset(InMemoryDataset):
    seed = 0
    generator = torch.Generator()

    def __init__(self, num_neighbors, neighbor_attr='t', path=None, transform=None):
        super().__init__(None, transform=None)
        self.path = path
        self.neighbor_attr = neighbor_attr
        self.num_neighbors = num_neighbors
        self.set_neighbors = KNNGraph(self.neighbor_attr, self.num_neighbors)
        self.generator.manual_seed(self.seed)
        data = self.load_data()
        data_list = self.split_neighborhoods_into_data(data)
        self.data, self.slices = self.collate(data_list)

    def load_data(self):
        raise NotImplementedError

    def points_to_data(self, t, pos, vel):
        sort = t.argsort()
        t, pos, vel = t[sort], pos[sort], vel[sort]
        data = Data(t=t, pos=pos, vel=vel)
        return data

    def split_neighborhoods_into_data(self, data):
        data = self.set_neighbors(data)
        node_j, node_i = data.edge_index
        data_list = []
        for i in range(data.num_nodes):
            neighborhood = data.subgraph(node_j[node_i ==i])
            neighborhood.poi_t = data.t[[i]]
            neighborhood.poi_pos = data.pos[[i]]
            neighborhood.poi_vel = data.vel[[i]]
            assert neighborhood.num_nodes == self.num_neighbors, f'{neighborhood.num_nodes}'
            data_list.append(neighborhood)
        return data_list


class Bifurcation(NeighborsDataset):
    num_pnts = 4000
    epsilon = .05

    def load_data(self):
        pos0 = torch.zeros(self.num_pnts, 2)
        branch = 2 * torch.bernoulli(.5 * torch.ones(self.num_pnts)) - 1
        pos0[:, 1] = 1 + self.epsilon * (.5 * torch.rand(self.num_pnts, generator=self.generator) + .5) * branch
        pos = pos0.clone()

        t = 6 * torch.rand(self.num_pnts, generator=self.generator)

        pos[:, 0] = t
        pos[:, 1] = (pos0[:, 1] - 1) * torch.exp(.1 * t**2 + pos0[:, 0] * t) + 1

        vel = torch.stack((
            torch.ones(pos.size(0)),
            .2 * pos[:, 0] * (pos[:, 1] - 1)
        )).T

        return self.points_to_data(t, pos, vel)


class Oscillation(NeighborsDataset):
    num_pnts = 4000
    epsilon = .05

    def load_data(self):
        pos0 = torch.zeros(self.num_pnts, 2)
        pos0[:, 0] = 1 + self.epsilon * (2 * torch.rand(self.num_pnts, generator=self.generator) - 1)
        pos = pos0.clone()
        t = 2 * np.pi * torch.rand(self.num_pnts, generator=self.generator)
        pos[:, 0] = pos0[:, 0] * torch.cos(t) + pos0[:, 1] * torch.sin(t)
        pos[:, 1] = -pos0[:, 0] * torch.sin(t) + pos0[:, 1] * torch.cos(t)

        vel = torch.stack((pos[:, 1], -pos[:, 0])).T

        return self.points_to_data(t, pos, vel)


class Simple(NeighborsDataset):
    num_pnts = 4000
    epsilon = .05

    def load_data(self):
        pos0 = torch.zeros(self.num_pnts, 2)
        pos0[:, 1] = self.epsilon * torch.rand(self.num_pnts, generator=self.generator)
        pos = pos0.clone()
        t = 3 * torch.rand(self.num_pnts, generator=self.generator)
        pos[:, 0] = t
        pos[:, 1] = (pos0[:, 1] - 1) * torch.exp(.1 * t**2 + pos0[:, 0] * t) + 1

        vel = torch.stack((
            torch.ones(pos.size(0)),
            .2 * pos[:, 0] * (pos[:, 1] - 1)
        )).T

        return self.points_to_data(t, pos, vel)


class Dentategyrus(NeighborsDataset):
    def load_data(self):
        if not self.path:
            raise ValueError('Provide a path to the dataset')
        df = pd.read_csv(self.path)
        t = torch.tensor(df['t'].to_numpy())
        pos = torch.tensor(df[['x1', 'x2']].to_numpy())
        vel = torch.tensor(df[['v1', 'v2']].to_numpy())

        return self.points_to_data(t, pos, vel)


class Forebrain(NeighborsDataset):
    def load_data(self):
        if not self.path:
            raise ValueError('Provide a path to the dataset')
        df = pd.read_csv(self.path)
        t = torch.tensor(df['t'].to_numpy())
        pos = torch.tensor(df[['x1', 'x2']].to_numpy())
        vel = torch.tensor(df[['v1', 'v2']].to_numpy())

        return self.points_to_data(t, pos, vel)


class Gastrulation(NeighborsDataset):
    def load_data(self):
        if not self.path:
            raise ValueError('Provide a path to the dataset')
        df = pd.read_csv(self.path)
        t = torch.tensor(df['t'].to_numpy())
        pos = torch.tensor(df[['x1', 'x2']].to_numpy())
        vel = torch.tensor(df[['v1', 'v2']].to_numpy())

        return self.points_to_data(t, pos, vel)


class Pancreas(NeighborsDataset):
    def load_data(self):
        if not self.path:
            raise ValueError('Provide a path to the dataset')
        df = pd.read_csv(self.path)
        t = torch.tensor(df['t'].to_numpy())
        pos = torch.tensor(df[['x1', 'x2']].to_numpy())
        vel = torch.tensor(df[['v1', 'v2']].to_numpy())

        return self.points_to_data(t, pos, vel)


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, transform=None, num_workers=8):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.transform = None
        self.generator = torch.Generator()
        self.seed = 0
        self.num_workers = num_workers

    def setup(self, stage):
        self.generator.manual_seed(self.seed)
        splits = torch.utils.data.random_split(self.dataset, [.8, .1, .1], generator=self.generator)
        self.train, self.val, self.test = splits

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.val, batch_size=len(self.val))
