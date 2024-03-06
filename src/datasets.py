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


class Dataset(InMemoryDataset):
    def __init__(self, cfg, df, split):
        self.cfg = cfg
        self.df = df
        self.split = split
        super().__init__(cfg.data_dir)
        self.load(self.processed_paths[0])

    def processed_file_names(self):
        return [f'{self.cfg.processed_file_name}__split_{self.split}.pt']

    def process(self):
        data_list = process_measurements2(self.df, self.cfg.sparsify_step_time, self.cfg.num_neighbors, 0)

        self.save(data_list, self.processed_paths[0])


def process_measurements2(measurements, sparsify_step_time, num_neighbors, poi_idx):
    measurements = measurements.sort_values('t', ignore_index=True)
    measurements = measurements.iloc[:len(measurements) // num_neighbors * num_neighbors]
    t = torch.tensor(measurements['t'].to_numpy())
    pos = torch.tensor(measurements[['x1', 'x2']].to_numpy())
    vel = torch.tensor(measurements[['v1', 'v2']].to_numpy())
    vel = utils.normalize(vel)
    data = Data(t=t, pos=pos, vel=vel)

    labels = torch.arange(data.num_nodes, dtype=torch.long)
    edge_index_nodes = []
    for i in range(data.num_nodes):
        node_i = labels.roll(-i)[::sparsify_step_time]
        node_j = torch.full(node_i.size(), i)
        edge_index_nodes.append(torch.stack((node_i, node_j)))
    # keep self-loops
    data.edge_index = torch.cat(edge_index_nodes, dim=1)
    data_keys = ('pos', 'vel', 't')
    data_values = zip(*(
        tg.utils.unbatch(data[k][data.edge_index[0]], data.edge_index[1])
        for k in data_keys
    ))
    data_list = []
    for i, (pos, vel, t) in enumerate(data_values):
        assert (data.pos[i] == pos[0]).all(), i
        assert (data.vel[i] == vel[0]).all(), i
        assert (data.t[i] == t[0]).all(), i
        neighborhood = Data(poi_pos=pos[[0]], poi_vel=vel[[0]], poi_t=t[[0]])
        edge_index_num_neighbors = tg.nn.knn_graph(t, num_neighbors)
        poi_neighbors = edge_index_num_neighbors[0][edge_index_num_neighbors[1] == 0]
        neighborhood.pos = pos[poi_neighbors]
        neighborhood.vel = vel[poi_neighbors]
        neighborhood.t = t[poi_neighbors]
        data_list.append(neighborhood)

    return data_list

    # split into data list
    # data_list = []
    # for i in range(data.num_nodes):
    #     idx = slice(0, 1)  # -> [0:1]
    #     neighborhood = data.subgraph(data.edge_index[0][data.edge_index[1] == i])
    #     neighborhood.edge_index = None
    #     assert (data.pos[i] == neighborhood.pos[0]).all()
    #     assert (data.vel[i] == neighborhood.vel[0]).all()
    #     assert (data.t[i] == neighborhood.t[0]).all()
    #     neighborhood.poi_t = neighborhood.t[idx]
    #     neighborhood.poi_pos = neighborhood.pos[idx]
    #     neighborhood.poi_vel = neighborhood.vel[idx]
    #     edge_index_num_neighbors = tg.nn.knn_graph(neighborhood.t, num_neighbors)
    #     neighbor_i, neighbor_j = edge_index_num_neighbors
    #     neighborhood = neighborhood.subgraph(neighbor_i[neighbor_j == 0])
    #     neighborhood.edge_index = None
    #     data_list.append(neighborhood)
    # breakpoint()

    # return data_list


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


def get_dataset(cfg, rng_seed=0):
    with pl.utilities.seed.isolate_rng():
        pl.seed_everything(rng_seed, workers=True)
        if not (Path(cfg.data_dir)/'processed').exists():
            if cfg.name == 'MotifSimple':
                df = generate_measurements_simple(cfg.num_pnts, cfg.epsilon)
            elif cfg.name == 'MotifOscillation':
                df = generate_measurements_oscillation(cfg.num_pnts, cfg.epsilon)
            elif cfg.name == 'MotifBifurcation':
                df = generate_measurements_bifurcation(cfg.num_pnts, cfg.epsilon)
            elif cfg.name == 'Saved':
                data_dir = Path(cfg.data_dir)
                df = pd.read_csv(data_dir/f'{data_dir.stem}.csv')
            else:
                raise ValueError(f'Unknown dataset: {cfg.name}')
            splits = split_train_val_test(df, train_prec=cfg.splits.train, val_prec=cfg.splits.val, test_prec=cfg.splits.test, rng_seed=rng_seed)
            splits = [process_measurements2(s, cfg.sparsify_step_time, cfg.num_neighbors, 0) for s in splits]
        else:
            splits = [0, 0, 0]
            splits = [Dataset(cfg, df_s, s) for df_s, s in zip(splits, ('train', 'val', 'test'))]

        return splits


@hydra.main(version_base=None, config_path='../configs', config_name='main')
def main(cfg):
    with omegaconf.open_dict(cfg):
        cfg.out_dir = str(Path(cfg.out_dir).resolve())
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.get('dataset') is None:
        raise ValueError('No datasets selected. Select a dataset with "+dataset@dataset.<name>=<dataset_cfg>".')

    train, val, test = map(DatasetMerged, zip(*[get_dataset(v, rng_seed=cfg.rng_seed) for v in cfg.dataset.values()]))
    breakpoint()
    print('end')


if __name__ == "__main__":
    main()
