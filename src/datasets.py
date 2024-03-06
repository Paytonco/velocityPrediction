from pathlib import Path
import itertools

import hydra
import omegaconf
from omegaconf import OmegaConf
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from torch_geometric.data import Data, InMemoryDataset

import utils


torch.set_default_dtype(torch.float64)


class DatasetMerged(InMemoryDataset):
    def __init__(self, data_lists):
        super().__init__(None)
        data_list = list(itertools.chain.from_iterable(data_lists))
        self.data, self.slices = self.collate(data_list)


def process_measurements2(measurements, sparsifier, num_neighbors, poi_idx):
    measurements = measurements.sort_values('t', ignore_index=True)
    t = torch.tensor(measurements['t'].to_numpy())
    pos = torch.tensor(measurements[['x1', 'x2']].to_numpy())
    vel = torch.tensor(measurements[['v1', 'v2']].to_numpy())
    vel = utils.normalize(vel)
    data = Data(t=t, pos=pos, vel=vel)

    # sparsify
    step = sparsifier.step
    node_i = torch.arange(data.num_nodes).view(-1, step).T
    node_i = [node_i.roll(-i, 1) for i in range(node_i.size(1))]
    node_i = torch.cat(node_i)
    node_j = node_i[:, [0]].broadcast_to(node_i.size())
    # node_i, node_j = node_i[:, 1:], node_j[:, 1:]  # remove self-loops
    # keep self-loops
    data.edge_index = torch.stack((node_i.reshape(-1), node_j.reshape(-1)))

    # split into data list
    data_list = []
    for i in range(data.num_nodes):
        idx = slice(0, 1)  # -> [0:1]
        neighborhood = data.subgraph(node_i[node_j == i])
        assert (data.pos[i] == neighborhood.pos[0]).all()
        assert (data.vel[i] == neighborhood.vel[0]).all()
        assert (data.t[i] == neighborhood.t[0]).all()
        neighborhood.poi_t = neighborhood.t[idx]
        neighborhood.poi_pos = neighborhood.pos[idx]
        neighborhood.poi_vel = neighborhood.vel[idx]
        edge_index_num_neighbors = tg.nn.knn_graph(neighborhood.t, num_neighbors)
        neighbor_i, neighbor_j = edge_index_num_neighbors
        neighborhood = neighborhood.subgraph(neighbor_i[neighbor_j == 0])
        data_list.append(neighborhood)

    return data_list


def process_measurements(measurements, sparsifier, num_neighbors, poi_idx):
    measurements = measurements.sort_values('t', ignore_index=True)
    measurements = sparsifier(measurements)
    t = torch.tensor(measurements['t'].to_numpy())
    pos = torch.tensor(measurements[['x1', 'x2']].to_numpy())
    vel = torch.tensor(measurements[['v1', 'v2']].to_numpy())
    vel = utils.normalize(vel)
    data = Data(t=t, pos=pos, vel=vel)

    windows = torch.arange(data.t.numel()).unfold(0, num_neighbors + 1, 1)
    mask_poi = torch.zeros(num_neighbors + 1, dtype=bool)
    mask_poi[num_neighbors // 2 + poi_idx] = True
    node_i = windows[:, mask_poi].squeeze().repeat_interleave(num_neighbors)
    node_j = windows[:, ~mask_poi].ravel()
    data.edge_index = torch.stack((node_j, node_i))
    # points near the time interval boundary have no neighbors
    data = data.subgraph(node_j)

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


def generate_measurements_simple(num_pnts, epsilon):
    pos0 = torch.zeros(num_pnts, 2)
    pos0[:, 1] = epsilon * torch.rand(num_pnts)
    pos = pos0.clone()
    t = 3 * torch.rand(num_pnts)
    pos[:, 0] = t
    pos[:, 1] = (pos0[:, 1] - 1) * torch.exp(.1 * t**2 + pos0[:, 0] * t) + 1

    vel = torch.stack((
        torch.ones(pos.size(0)),
        .2 * pos[:, 0] * (pos[:, 1] - 1)
    )).T

    return pd.DataFrame(
        torch.cat((t[:, None], pos, vel), axis=1),
        columns=['t', 'x1', 'x2', 'v1', 'v2']
    )


def generate_measurements_oscillation(num_pnts, epsilon):
    pos0 = torch.zeros(num_pnts, 2)
    pos0[:, 0] = 1 + epsilon * (2 * torch.rand(num_pnts) - 1)
    pos = pos0.clone()
    t = 2 * np.pi * torch.rand(num_pnts)
    pos[:, 0] = pos0[:, 0] * torch.cos(t) + pos0[:, 1] * torch.sin(t)
    pos[:, 1] = -pos0[:, 0] * torch.sin(t) + pos0[:, 1] * torch.cos(t)

    vel = torch.stack((pos[:, 1], -pos[:, 0])).T

    return pd.DataFrame(
        torch.cat((t[:, None], pos, vel), axis=1),
        columns=['t', 'x1', 'x2', 'v1', 'v2']
    )


def generate_measurements_bifurcation(num_pnts, epsilon):
    pos0 = torch.zeros(num_pnts, 2)
    branch = 2 * torch.bernoulli(.5 * torch.ones(num_pnts)) - 1
    pos0[:, 1] = 1 + epsilon * (.5 * torch.rand(num_pnts) + .5) * branch
    pos = pos0.clone()

    t = 6 * torch.rand(num_pnts)

    pos[:, 0] = t
    pos[:, 1] = (pos0[:, 1] - 1) * torch.exp(.1 * t**2 + pos0[:, 0] * t) + 1

    vel = torch.stack((
        torch.ones(pos.size(0)),
        .2 * pos[:, 0] * (pos[:, 1] - 1)
    )).T

    return pd.DataFrame(
        torch.cat((t[:, None], pos, vel), axis=1),
        columns=['t', 'x1', 'x2', 'v1', 'v2']
    )


def split_train_val_test(df, train_prec, val_prec, test_prec, rng_seed):
    rng = np.random.default_rng(seed=rng_seed)
    idx = rng.permutation(len(df))
    split_idxs = (len(idx) * np.array([train_prec, 1 - val_prec - test_prec, 1 - test_prec])).astype(int)
    train, _, val, test = np.split(idx, split_idxs)

    return df.iloc[train], df.iloc[val], df.iloc[test]


class Sparsifier:
    def __call__(self, measurements):
        raise NotImplementedError()


class TimeSkipByStep(Sparsifier):
    def __init__(self, step):
        self.step = step

    def __call__(self, measurements):
        return measurements.iloc[::self.step]


def get_sparsifier(cfg):
    if cfg.name == 'TimeSkipByStep':
        return TimeSkipByStep(cfg.step)
    else:
        raise ValueError(f'Unknown sparsifier: {cfg.name}')


def get_dataset(cfg, data_dir, rng_seed=0):
    with pl.utilities.seed.isolate_rng():
        pl.seed_everything(rng_seed, workers=True)
        sparsifier = get_sparsifier(cfg.sparsifier)
        if cfg.name == 'MotifSimple':
            df = generate_measurements_simple(cfg.num_pnts, cfg.epsilon)
        elif cfg.name == 'MotifOscillation':
            df = generate_measurements_oscillation(cfg.num_pnts, cfg.epsilon)
        elif cfg.name == 'MotifBifurcation':
            df = generate_measurements_bifurcation(cfg.num_pnts, cfg.epsilon)
        elif cfg.name == 'Saved':
            df = pd.read_csv(cfg.path)
        else:
            raise ValueError(f'Unknown dataset: {cfg.name}')
        splits = split_train_val_test(df, train_prec=cfg.splits.train, val_prec=cfg.splits.val, test_prec=cfg.splits.test, rng_seed=rng_seed)
        splits = [process_measurements2(s, sparsifier, cfg.num_neighbors, cfg.poi_idx) for s in splits]

        return splits


@hydra.main(version_base=None, config_path='../configs', config_name='main')
def main(cfg):
    with omegaconf.open_dict(cfg):
        cfg.out_dir = str(Path(cfg.out_dir).resolve())
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.get('dataset') is None:
        raise ValueError('No datasets selected. Select a dataset with "+dataset@dataset.<name>=<dataset_cfg>".')

    train, val, test = map(DatasetMerged, zip(*[get_dataset(v, cfg.data_dir, rng_seed=cfg.rng_seed) for v in cfg.dataset.values()]))


if __name__ == "__main__":
    main()
