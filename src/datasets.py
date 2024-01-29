from pathlib import Path
import itertools

import hydra
import omegaconf
from omegaconf import OmegaConf
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

import utils


torch.set_default_dtype(torch.float64)


class DatasetMerged(InMemoryDataset):
    def __init__(self, datasets):
        super().__init__(None)
        data_list = list(itertools.chain.from_iterable(datasets))
        self.data, self.slices = self.collate(data_list)


class Dataset(InMemoryDataset):
    def __init__(self, root, processed_file_name,
                 sparsifier, poi_idx, num_neighbors,
                 transform=None, pre_transform=None, pre_filter=None):
        self.processed_file_name = processed_file_name
        self.sparsifier = sparsifier
        self.poi_idx = poi_idx
        self.num_neighbors = num_neighbors
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.processed_file_name]

    def process(self):
        measurements = self.get_measurements()
        measurements = measurements.sort_values('t', ignore_index=True)
        measurements = self.sparsifier(measurements)
        t = torch.tensor(measurements['t'].to_numpy())
        pos = torch.tensor(measurements[['x1', 'x2']].to_numpy())
        vel = torch.tensor(measurements[['v1', 'v2']].to_numpy())
        vel = utils.normalize(vel)
        data = Data(t=t, pos=pos, vel=vel)

        windows = torch.arange(data.t.numel()).unfold(0, self.num_neighbors + 1, 1)
        mask_poi = torch.zeros(self.num_neighbors + 1, dtype=bool)
        mask_poi[self.num_neighbors // 2 + self.poi_idx] = True
        node_i = windows[:, mask_poi].squeeze().repeat_interleave(self.num_neighbors)
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

        self.save(data_list, self.processed_paths[0])


class Motif(Dataset):
    def __init__(self, root, processed_file_name,
                 num_pnts, epsilon, sparsifier, poi_idx, num_neighbors,
                 transform=None, pre_transform=None, pre_filter=None):
        self.num_pnts = num_pnts
        self.epsilon = epsilon
        super().__init__(root, processed_file_name,
                         sparsifier, poi_idx, num_neighbors,
                         transform, pre_transform, pre_filter)

    def get_measurements(self):
        t, pos, vel = self.generate_measurements()
        return pd.DataFrame(
            torch.cat((t[:, None], pos, vel), axis=1),
            columns=['t', 'x1', 'x2', 'v1', 'v2']
        )

    def generate_measurements(self):
        raise NotImplementedError()


class Simple(Motif):
    def generate_measurements(self):
        pos0 = torch.zeros(self.num_pnts, 2)
        pos0[:, 1] = self.epsilon * torch.rand(self.num_pnts)
        pos = pos0.clone()
        t = 3 * torch.rand(self.num_pnts)
        pos[:, 0] = t
        pos[:, 1] = (pos0[:, 1] - 1) * torch.exp(.1 * t**2 + pos0[:, 0] * t) + 1

        vel = torch.stack((
            torch.ones(pos.size(0)),
            .2 * pos[:, 0] * (pos[:, 1] - 1)
        )).T

        return t, pos, vel


class Oscillation(Motif):
    def generate_measurements(self):
        pos0 = torch.zeros(self.num_pnts, 2)
        pos0[:, 0] = 1 + self.epsilon * (2 * torch.rand(self.num_pnts) - 1)
        pos = pos0.clone()
        t = 2 * np.pi * torch.rand(self.num_pnts)
        pos[:, 0] = pos0[:, 0] * torch.cos(t) + pos0[:, 1] * torch.sin(t)
        pos[:, 1] = -pos0[:, 0] * torch.sin(t) + pos0[:, 1] * torch.cos(t)

        vel = torch.stack((pos[:, 1], -pos[:, 0])).T

        return t, pos, vel


class Bifurcation(Motif):
    def generate_measurements(self):
        pos0 = torch.zeros(self.num_pnts, 2)
        branch = 2 * torch.bernoulli(.5 * torch.ones(self.num_pnts)) - 1
        pos0[:, 1] = 1 + self.epsilon * (.5 * torch.rand(self.num_pnts) + .5) * branch
        pos = pos0.clone()

        t = 6 * torch.rand(self.num_pnts)

        pos[:, 0] = t
        pos[:, 1] = (pos0[:, 1] - 1) * torch.exp(.1 * t**2 + pos0[:, 0] * t) + 1

        vel = torch.stack((
            torch.ones(pos.size(0)),
            .2 * pos[:, 0] * (pos[:, 1] - 1)
        )).T

        return t, pos, vel


class Saved(Dataset):
    def __init__(self, data_path, processed_file_name,
                 sparsifier, poi_idx, num_neighbors,
                 transform=None, pre_transform=None, pre_filter=None):
        self.data_path = Path(data_path)
        super().__init__(self.data_path.parent, processed_file_name,
                         sparsifier, poi_idx, num_neighbors,
                         transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [self.data_path]

    def get_measurements(self):
        return pd.read_csv(self.data_path)


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
            ds = Simple(f'{data_dir}/{cfg.name}', cfg.processed_file_name, cfg.num_pnts, cfg.epsilon, sparsifier, cfg.poi_idx, cfg.num_neighbors)
        elif cfg.name == 'MotifOscillation':
            ds = Oscillation(f'{data_dir}/{cfg.name}', cfg.processed_file_name, cfg.num_pnts, cfg.epsilon, sparsifier, cfg.poi_idx, cfg.num_neighbors)
        elif cfg.name == 'MotifBifurcation':
            ds = Bifurcation(f'{data_dir}/{cfg.name}', cfg.processed_file_name, cfg.num_pnts, cfg.epsilon, sparsifier, cfg.poi_idx, cfg.num_neighbors)
        elif cfg.name == 'Saved':
            ds = Saved(cfg.path, cfg.processed_file_name, sparsifier, cfg.poi_idx, cfg.num_neighbors)
        else:
            raise ValueError(f'Unknown dataset: {cfg.name}')
        ds = ds.shuffle()
        train, val, test = split_train_val_test(list(range(len(ds))), train_prec=cfg.splits.train, val_prec=cfg.splits.val, test_prec=cfg.splits.test)
        return ds[train], ds[val], ds[test]


def split_train_val_test(idx, train_prec=.7, val_prec=.2, test_prec=.1):
    split_idxs = (len(idx) * np.array([train_prec, 1 - val_prec - test_prec, 1 - test_prec])).astype(int)
    train, _, val, test = np.split(idx, split_idxs)
    return train, val, test


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
