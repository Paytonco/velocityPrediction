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
import scvelo
import tables

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
        data_list = process_measurements(self.df, self.cfg.sparsify_step_time, self.cfg.num_neighbors, 0)

        self.save(data_list, self.processed_paths[0])


def process_measurements(measurements, sparsify_step_time, num_neighbors, poi_idx):
    measurements = measurements.sort_values('t', ignore_index=True)
    t = torch.tensor(measurements['t'].to_numpy())
    pos = torch.tensor(measurements[['x1', 'x2']].to_numpy())
    vel = torch.tensor(measurements[['v1', 'v2']].to_numpy())
    vel = utils.normalize(vel)
    data = Data(t=t, pos=pos, vel=vel)

    data.labels = torch.arange(data.num_nodes, dtype=torch.long)
    edge_index_nodes = []
    for i in range(data.num_nodes):
        label_idx, roll_by = divmod(i, sparsify_step_time)
        labels = data.labels.roll(-roll_by)[::sparsify_step_time]
        edge_index_num_neighbors = tg.nn.knn_graph(labels, num_neighbors)
        node_i = labels[edge_index_num_neighbors[0][edge_index_num_neighbors[1] == label_idx]]
        node_j = torch.full(node_i.size(), i)
        edge_index_nodes.append(torch.stack((node_i, node_j)))
    # keep self-loops
    data.edge_index = torch.cat(edge_index_nodes, dim=1)
    data_keys = ('pos', 'vel', 't', 'labels')
    data_values = zip(*(
        tg.utils.unbatch(data[k][data.edge_index[0]], data.edge_index[1])
        for k in data_keys
    ))
    data_list = []
    for i, (pos, vel, t, labels) in enumerate(data_values):
        neighborhood = Data(
            poi_pos=data.pos[[i]], poi_vel=data.vel[[i]], poi_t=data.t[[i]],
            pos=pos, vel=vel, t=t,
            # labels=labels
        )
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


def generate_measurements_scvelo_simulation(num_pnts):
    adata = scvelo.datasets.simulation(n_obs=num_pnts)

    scvelo.pp.filter_and_normalize(adata)
    scvelo.pp.moments(adata)
    scvelo.tl.velocity(adata, mode='stochastic')

    # must call velocity_pseudotime before velocity_graph
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_pseudotime(adata)

    scvelo.tl.umap(adata)
    scvelo.tl.velocity_embedding(adata, basis='umap')

    t = adata.obs.velocity_pseudotime.to_numpy()
    pos = adata.obsm['X_umap']
    vel = adata.obsm['velocity_umap']

    return pd.DataFrame(
        data=np.concatenate((t[:, None], pos, vel), axis=1),
        columns=['t', 'x1', 'x2', 'v1', 'v2']
    )


def download_bonemarrow(data_dir):
    path_h5ad = str(data_dir/f'{data_dir.stem}.h5ad')
    adata = scvelo.datasets.bonemarrow(path_h5ad)

    scvelo.pp.filter_and_normalize(adata)
    scvelo.pp.moments(adata)
    scvelo.tl.velocity(adata, mode='stochastic')

    # must call velocity_pseudotime before velocity_graph
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_pseudotime(adata)

    scvelo.tl.umap(adata)  # for forebrain and bonemarrow and pbmc68k
    scvelo.tl.velocity_embedding(adata, basis='umap')

    pseudotime = adata.obs.velocity_pseudotime
    positions = adata.obsm['X_umap']
    velocities = adata.obsm['velocity_umap']
    data = pd.DataFrame(
        data=np.concatenate((positions, velocities), axis=1),
        index=pseudotime.rename('t'),
        columns=['x1', 'x2', 'v1', 'v2']
    )
    data.to_csv(data_dir/f'{data_dir.stem}.csv')


def download_dentategyrus(data_dir):
    path_h5ad = str(data_dir/f'{data_dir.stem}.h5ad')
    adata = scvelo.datasets.dentategyrus(path_h5ad)

    scvelo.pp.filter_and_normalize(adata)
    scvelo.pp.moments(adata)
    scvelo.tl.velocity(adata, mode='stochastic')

    # must call velocity_pseudotime before velocity_graph
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_pseudotime(adata)

    scvelo.tl.velocity_embedding(adata, basis='umap')

    pseudotime = adata.obs.velocity_pseudotime
    positions = adata.obsm['X_umap']
    velocities = adata.obsm['velocity_umap']
    data = pd.DataFrame(
        data=np.concatenate((positions, velocities), axis=1),
        index=pseudotime.rename('t'),
        columns=['x1', 'x2', 'v1', 'v2']
    )
    data.to_csv(data_dir/f'{data_dir.stem}.csv')


def download_forebrain(data_dir):
    path_h5ad = str(data_dir/f'{data_dir.stem}.h5ad')
    try:
        adata = scvelo.datasets.forebrain(path_h5ad)
    except TypeError:
        f = tables.open_file(path_h5ad, mode='r+')
        # these are empty
        f.remove_node('/row_graphs')
        f.remove_node('/col_graphs')
        # rename to match AnnData data structure
        f.rename_node('/row_attrs', 'obs')
        f.rename_node('/col_attrs', 'var')
        f.rename_node('/matrix', 'X')
        f.close()
        adata = scvelo.datasets.forebrain(path_h5ad)

    scvelo.pp.remove_duplicate_cells(adata)  # for forebrain
    scvelo.pp.neighbors(adata)  # for forebrain
    scvelo.pp.filter_and_normalize(adata)
    scvelo.pp.moments(adata)
    scvelo.tl.velocity(adata, mode='stochastic')

    # must call velocity_pseudotime before velocity_graph
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_pseudotime(adata)

    scvelo.tl.umap(adata)  # for forebrain and bonemarrow and pbmc68k
    scvelo.tl.velocity_embedding(adata, basis='umap')

    pseudotime = adata.obs.velocity_pseudotime
    positions = adata.obsm['X_umap']
    velocities = adata.obsm['velocity_umap']
    data = pd.DataFrame(
        data=np.concatenate((positions, velocities), axis=1),
        index=pseudotime.rename('t'),
        columns=['x1', 'x2', 'v1', 'v2']
    )
    data.to_csv(data_dir/f'{data_dir.stem}.csv')


def download_pancreas(data_dir):
    path_h5ad = str(data_dir/f'{data_dir.stem}.h5ad')
    adata = scvelo.datasets.pancreas(path_h5ad)

    scvelo.pp.filter_and_normalize(adata)
    scvelo.pp.moments(adata)
    scvelo.tl.velocity(adata, mode='stochastic')

    # must call velocity_pseudotime before velocity_graph
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_pseudotime(adata)

    scvelo.tl.velocity_embedding(adata, basis='umap')

    pseudotime = adata.obs.velocity_pseudotime
    positions = adata.obsm['X_umap']
    velocities = adata.obsm['velocity_umap']
    data = pd.DataFrame(
        data=np.concatenate((positions, velocities), axis=1),
        index=pseudotime.rename('t'),
        columns=['x1', 'x2', 'v1', 'v2']
    )
    data.to_csv(data_dir/f'{data_dir.stem}.csv')


def download_pbmc68k(data_dir):
    path_h5ad = str(data_dir/f'{data_dir.stem}.h5ad')
    adata = scvelo.datasets.pbmc68k(path_h5ad)

    scvelo.pp.filter_and_normalize(adata)
    scvelo.pp.moments(adata)
    scvelo.tl.velocity(adata, mode='stochastic')

    # must call velocity_pseudotime before velocity_graph
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_pseudotime(adata)

    scvelo.tl.umap(adata)  # for forebrain and bonemarrow and pbmc68k
    scvelo.tl.velocity_embedding(adata, basis='umap')

    pseudotime = adata.obs.velocity_pseudotime
    positions = adata.obsm['X_umap']
    velocities = adata.obsm['velocity_umap']
    data = pd.DataFrame(
        data=np.concatenate((positions, velocities), axis=1),
        index=pseudotime.rename('t'),
        columns=['x1', 'x2', 'v1', 'v2']
    )
    data.to_csv(data_dir/f'{data_dir.stem}.csv')


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
            elif cfg.name == 'SCVeloSimulation':
                df = generate_measurements_scvelo_simulation(cfg.num_pnts)
            elif cfg.name == 'Saved':
                data_dir = Path(cfg.data_dir)
                name = data_dir.stem
                csv_path = data_dir/f'{name}.csv'
                if not csv_path.exists():
                    if name == 'forebrain':
                        download_forebrain(data_dir)
                    elif name == 'pancreas':
                        download_pancreas(data_dir)
                    else:
                        raise ValueError(f'Unknown saved dataset: {name}')
                df = pd.read_csv(data_dir/f'{data_dir.stem}.csv')
            else:
                raise ValueError(f'Unknown dataset: {cfg.name}')
            splits = split_train_val_test(df, train_prec=cfg.splits.train, val_prec=cfg.splits.val, test_prec=cfg.splits.test, rng_seed=rng_seed)
            splits = [process_measurements(s, cfg.sparsify_step_time, cfg.num_neighbors, 0) for s in splits]
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
