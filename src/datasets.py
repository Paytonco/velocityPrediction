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
from torch_geometric.data import Data, Batch, InMemoryDataset
import scvelo
import tables
import anndata

import utils


torch.set_default_dtype(torch.float64)


class DatasetMerged(InMemoryDataset):
    def __init__(self, data_lists):
        super().__init__(None)
        data_list = list(itertools.chain.from_iterable(data_lists))
        self.data, self.slices = self.collate(data_list)


class Dataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(None)
        self.data, self.slices = self.collate(data_list)


def process_measurements(measurements, sparsify_step_time, num_neighbors, poi_idx):
    measurements = measurements.sort_values('t', ignore_index=True)
    measurement_id = torch.tensor(measurements['measurement_id'].to_numpy())
    t = torch.tensor(measurements['t'].to_numpy())
    pos = torch.tensor(measurements[[c for c in measurements.columns if c.startswith('x')]].to_numpy())
    vel = torch.tensor(measurements[[c for c in measurements.columns if c.startswith('v')]].to_numpy())
    vel = utils.normalize(vel)
    data = Data(t=t, pos=pos, vel=vel, measurement_id=measurement_id)

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
    data_keys = ('pos', 'vel', 't', 'labels', 'measurement_id')
    data_values = zip(*(
        tg.utils.unbatch(data[k][data.edge_index[0]], data.edge_index[1])
        for k in data_keys
    ))
    data_list = []
    for i, (pos, vel, t, labels, measurement_id) in enumerate(data_values):
        neighborhood = Data(
            poi_pos=data.pos[[i]], poi_vel=data.vel[[i]], poi_t=data.t[[i]],
            poi_measurement_id=data.measurement_id[[i]],
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


def generate_measurements_scvelo_simulation(cfg):
    adata = scvelo.datasets.simulation(n_obs=cfg.num_pnts)

    scvelo.pp.filter_and_normalize(adata)
    scvelo.pp.moments(adata)
    scvelo.tl.velocity(adata, mode='stochastic')

    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_pseudotime(adata)

    scvelo.tl.umap(adata, n_components=cfg.umap.n_components)
    scvelo.tl.velocity_embedding(adata, basis='umap')

    t = adata.obs.velocity_pseudotime.to_numpy()
    pos = adata.obsm['X_umap']
    vel = adata.obsm['velocity_umap']

    return pd.DataFrame(
        data=np.concatenate((t[:, None], pos, vel), axis=1),
        columns=['t', 'x1', 'x2', 'v1', 'v2']
    )


def download_bonemarrow(cfg, data_dir):
    path_h5ad = str(data_dir/f'{data_dir.stem}.h5ad')
    adata = scvelo.datasets.bonemarrow(path_h5ad)

    scvelo.pp.filter_and_normalize(adata)
    scvelo.pp.moments(adata)
    scvelo.tl.velocity(adata, mode='stochastic')

    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_pseudotime(adata)

    scvelo.tl.umap(adata, n_components=cfg.umap.n_components)
    scvelo.tl.velocity_embedding(adata, basis='umap')

    return dict(
        t=adata.obs.velocity_pseudotime,
        pos=adata.obsm['X_umap'],
        vel=adata.obsm['velocity_umap']
    )


def download_dentategyrus(cfg, data_dir):
    path_h5ad = str(data_dir/f'{data_dir.stem}.h5ad')
    adata = scvelo.datasets.dentategyrus(path_h5ad)

    scvelo.pp.filter_and_normalize(adata)
    scvelo.pp.moments(adata)
    scvelo.tl.velocity(adata, mode='stochastic')

    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_pseudotime(adata)

    scvelo.tl.umap(adata, n_components=cfg.umap.n_components)
    scvelo.tl.velocity_embedding(adata, basis='umap')

    return dict(
        t=adata.obs.velocity_pseudotime,
        pos=adata.obsm['X_umap'],
        vel=adata.obsm['velocity_umap']
    )


def download_forebrain(cfg, data_dir):
    path_h5ad = str(data_dir/f'{data_dir.stem}.h5ad')
    try:
        adata = scvelo.datasets.forebrain(path_h5ad)
    except (TypeError, anndata._io.utils.AnnDataReadError):
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

    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_pseudotime(adata)

    scvelo.tl.umap(adata, n_components=cfg.umap.n_components)
    scvelo.tl.velocity_embedding(adata, basis='umap')

    return dict(
        t=adata.obs.velocity_pseudotime,
        pos=adata.obsm['X_umap'],
        vel=adata.obsm['velocity_umap']
    )


def download_pancreas(cfg, data_dir):
    path_h5ad = str(data_dir/f'{data_dir.stem}.h5ad')
    adata = scvelo.datasets.pancreas(path_h5ad)

    scvelo.pp.filter_and_normalize(adata)
    scvelo.pp.moments(adata)
    scvelo.tl.velocity(adata, mode='stochastic')

    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_pseudotime(adata)

    scvelo.tl.umap(adata, n_components=cfg.umap.n_components)
    scvelo.tl.velocity_embedding(adata, basis='umap')

    return dict(
        t=adata.obs.velocity_pseudotime,
        pos=adata.obsm['X_umap'],
        vel=adata.obsm['velocity_umap']
    )


def download_pbmc68k(cfg, data_dir):
    path_h5ad = str(data_dir/f'{data_dir.stem}.h5ad')
    adata = scvelo.datasets.pbmc68k(path_h5ad)

    scvelo.pp.filter_and_normalize(adata)
    scvelo.pp.moments(adata)
    scvelo.tl.velocity(adata, mode='stochastic')

    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_pseudotime(adata)

    scvelo.tl.umap(adata, n_components=cfg.umap.n_components)
    scvelo.tl.velocity_embedding(adata, basis='umap')

    return dict(
        t=adata.obs.velocity_pseudotime,
        pos=adata.obsm['X_umap'],
        vel=adata.obsm['velocity_umap']
    )


def split_train_val_test(ds, train_prec, val_prec, test_prec, rng_seed):
    rng = np.random.default_rng(seed=rng_seed)
    idx = rng.permutation(len(ds))
    split_idxs = (len(idx) * np.array([train_prec, 1 - val_prec - test_prec, 1 - test_prec])).astype(int)
    train, _, val, test = np.split(idx, split_idxs)

    return ds[train], ds[val], ds[test]


def get_dataset_df(cfg, rng_seed=0):
    with pl.utilities.seed.isolate_rng():
        pl.seed_everything(rng_seed, workers=True)
        if cfg.name == 'MotifSimple':
            df = generate_measurements_simple(cfg.num_pnts, cfg.epsilon)
        elif cfg.name == 'MotifOscillation':
            df = generate_measurements_oscillation(cfg.num_pnts, cfg.epsilon)
        elif cfg.name == 'MotifBifurcation':
            df = generate_measurements_bifurcation(cfg.num_pnts, cfg.epsilon)
        elif cfg.name == 'SCVeloSimulation':
            df = generate_measurements_scvelo_simulation(cfg)
        elif cfg.name == 'SCVeloSaved':
            data_dir = Path(cfg.data_dir)
            name = data_dir.stem
            csv_path = data_dir/f'{name}__{cfg.csv.name_suffix}.csv'
            dims = [*range(1, cfg.umap.n_components + 1)]
            cols_pos = [f'x{i}' for i in dims]
            cols_vel = [f'v{i}' for i in dims]
            if cfg.csv.load_saved and csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                if name == 'bonemarrow':
                    data = download_bonemarrow(cfg, data_dir)
                elif name == 'dentategyrus':
                    data = download_dentategyrus(cfg, data_dir)
                elif name == 'forebrain':
                    data = download_forebrain(cfg, data_dir)
                elif name == 'pancreas':
                    data = download_pancreas(cfg, data_dir)
                elif name == 'pbmc68k':
                    data = download_pbmc68k(cfg, data_dir)
                else:
                    raise ValueError(f'Unknown saved dataset: {name}')
                data = np.concatenate((
                    data['t'].to_numpy()[:, None],
                    data['pos'], data['vel']
                ), axis=1)
                df = pd.DataFrame(
                    data=data,
                    columns=['t', *cols_pos, *cols_vel]
                )
                df.to_csv(csv_path, index=False)
        else:
            raise ValueError(f'Unknown dataset: {cfg.name}')

        if cfg.reverse_velocities:
            df[cols_vel] = -df[cols_vel]

        df['measurement_id'] = range(len(df))

        return df


def get_dataset(cfg, rng_seed=0):
    with pl.utilities.seed.isolate_rng():
        pl.seed_everything(rng_seed, workers=True)
        if not (Path(cfg.data_dir)/'processed').exists():
            df = get_dataset_df(cfg, rng_seed=rng_seed)
            ds = Dataset(process_measurements(df, cfg.sparsify_step_time, cfg.num_neighbors, 0))
            train, val, test = split_train_val_test(ds, train_prec=cfg.splits.train, val_prec=cfg.splits.val, test_prec=cfg.splits.test, rng_seed=rng_seed)
            if cfg.train_max_size is not None:
                train = train[:cfg.train_max_size]
        else:
            splits = [0, 0, 0]
            splits = [Dataset(cfg, df_s, s) for df_s, s in zip(splits, ('train', 'val', 'test'))]

        return train, val, test


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
