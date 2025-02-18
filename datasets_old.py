import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import scvelo

torch.set_default_dtype(torch.float64)


class KNNGraph(T.KNNGraph):
    def __init__(self, attr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr = attr

    def __call__(self, data):
        pos = data.pos
        data.pos = data[self.attr]
        data = super().__call__(data)
        data.pos = pos

        return data


class RadiusGraph(T.RadiusGraph):
    def __init__(self, attr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr = attr

    def __call__(self, data):
        pos = data.pos
        data.pos = data[self.attr]
        data = super().__call__(data)
        data.pos = pos

        return data


class AnnulusGraph(T.BaseTransform):
    def __init__(self, attr, r1, r2, **kwargs):
        self.r1_graph = T.RadiusGraph(r1, **kwargs)
        self.r2_graph = T.RadiusGraph(r2, **kwargs)


class SlidingWindowGraph(T.BaseTransform):
    def __init__(self, num_prev_neighbors, window_size):
        super().__init__()
        self.num_prev_neighbors = num_prev_neighbors
        self.window_size = window_size

    def __call__(self, data):
        assert (data.t.diff() >= 0).all(), 'Time must be in non-decreasing order.'
        windows = torch.arange(data.t.numel()).unfold(0, self.window_size, 1)
        mask_poi = torch.zeros(self.window_size, dtype=bool)
        mask_poi[self.num_prev_neighbors] = True
        node_i = windows[:, mask_poi].squeeze().repeat_interleave(self.window_size - 1)
        node_j = windows[:, ~mask_poi].ravel()
        data.edge_index = torch.stack((node_j, node_i))
        # points near the time interval boundary have no neighbors
        data = data.subgraph(node_j)

        return data


class PosBoxFilter:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, data):
        x, y = data.poi_pos[0]
        return (self.x1 <= x <= self.x2) and (self.y1 <= y <= self.y2)


class NeighborsDataset(InMemoryDataset):
    seed = 0
    generator = torch.Generator()

    def __init__(self, set_neighbors_transform, sparsity_step, path=None, filter=None):
        super().__init__(None)
        self.set_neighbors_transform = set_neighbors_transform
        self.filter = filter
        self.sparsity_step = sparsity_step
        self.path = path
        self.generator.manual_seed(self.seed)
        data = self.load_data()
        data_list = self.split_neighborhoods_into_data(data)
        if self.filter:
            data_list = [*filter(self.filter, data_list)]
        self.data, self.slices = self.collate(data_list)

    def load_data(self):
        raise NotImplementedError

    def points_to_data(self, t, pos, vel):
        sort = t.argsort()
        t, pos, vel = t[sort], pos[sort], vel[sort]
        step = self.sparsity_step
        t, pos, vel = t[::step], pos[::step], vel[::step]
        data = Data(t=t, pos=pos, vel=vel)
        return data

    def split_neighborhoods_into_data(self, data):
        data = self.set_neighbors_transform(data)
        node_j, node_i = data.edge_index
        data_list = []
        for i in torch.unique(node_i):
            idx = slice(i, i+1)  # -> [i:i+1]
            neighborhood = data.subgraph(node_j[node_i == i])
            neighborhood.poi_t = data.t[idx]
            neighborhood.poi_pos = data.pos[idx]
            neighborhood.poi_vel = data.vel[idx]
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


class SCVeloSimulation(NeighborsDataset):
    def load_data(self):
        if not self.path:
            raise ValueError('Provide a path to the dataset')
        df = pd.read_csv(self.path)
        t = torch.tensor(df['t'].to_numpy())
        pos = torch.tensor(df[['x1', 'x2']].to_numpy())
        vel = torch.tensor(df[['v1', 'v2']].to_numpy())

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
    def __init__(self, dataset, batch_size, pred_split='val', transform=None, num_workers=8):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.pred_split = pred_split
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

    def predict_dataloader(self, split=None, batch_size=None):
        if split is None:
            split = self.pred_split
        ds = getattr(self, split)
        batch_size = batch_size or len(ds)
        return DataLoader(ds, batch_size=batch_size)


def plot_dataset(data):
    fig, ax = plt.subplots()
    pos, vel = data.poi_pos, data.poi_vel
    ax.scatter(pos[:, 0], pos[:, 1], label='State', c='orange')
    ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], color='deepskyblue')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return fig


def get_transform(cfg, rng_seed=0):
    with pl.utilities.seed.isolate_rng():
        pl.seed_everything(rng_seed, workers=True)
        pass


def get_dataset(cfg, rng_seed=0):
    with pl.utilities.seed.isolate_rng():
        pl.seed_everything(rng_seed, workers=True)
        transform = get_transform(cfg.transform)
        if cfg.name == '':
            ds = Simple(transform, cfg.sparsity_step, path=cfg.path)
        else:
            raise ValueError(f'Unknown dataset: {cfg.name}')


if __name__ == '__main__':
    filter = PosBoxFilter(-15, 15, -6, -2.5)
    ds_transform = SlidingWindowGraph(2, 8 + 1)
    ds = Pancreas(ds_transform, 1, path='../../out/paytonco/data/Pancreas/Pancreas.csv', filter=filter)
    dl = DataLoader(ds, batch_size=len(ds))
    data = next(iter(dl))
    fig = plot_dataset(data)
    fig.savefig('Pancreas.pdf', format='pdf')
