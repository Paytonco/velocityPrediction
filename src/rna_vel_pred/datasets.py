from collections import defaultdict
import pprint
import itertools

import hydra
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

from conf import conf, datasets
from rna_vel_pred import utils


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
        labels = labels[labels.diff(prepend=labels[:1]+sparsify_step_time).abs() == sparsify_step_time]
        neighbor_labels, poi_labels = tg.nn.knn_graph(labels, num_neighbors)
        node_j = labels[neighbor_labels[poi_labels == label_idx]]
        node_i = torch.full(node_j.size(), i)
        edge_index_nodes.append(torch.stack((node_j, node_i)))
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


class Motif:
    @staticmethod
    def generate_simple(cfg):
        pos0 = torch.zeros(cfg.measurement_count, 2)
        pos0[:, 1] = cfg.initial_condition_noise_epsilon * torch.rand(cfg.measurement_count)
        pos = pos0.clone()
        t = 3 * torch.rand(cfg.measurement_count)
        pos[:, 0] = t
        pos[:, 1] = (pos0[:, 1] - 1) * torch.exp(.1 * t**2 + pos0[:, 0] * t) + 1

        vel = torch.stack((
            torch.ones(pos.size(0)),
            .2 * pos[:, 0] * (pos[:, 1] - 1)
        )).T

        return Motif._to_dataframe(t, pos, vel)

    @staticmethod
    def generate_oscillation(cfg):
        pos0 = torch.zeros(cfg.measurement_count, 2)
        pos0[:, 0] = 1 + cfg.initial_condition_noise_epsilon * (2 * torch.rand(cfg.measurement_count) - 1)
        pos = pos0.clone()
        t = 2 * np.pi * torch.rand(cfg.measurement_count)
        pos[:, 0] = pos0[:, 0] * torch.cos(t) + pos0[:, 1] * torch.sin(t)
        pos[:, 1] = -pos0[:, 0] * torch.sin(t) + pos0[:, 1] * torch.cos(t)

        vel = torch.stack((pos[:, 1], -pos[:, 0])).T

        return Motif._to_dataframe(t, pos, vel)

    @staticmethod
    def generate_bifurcation(cfg):
        pos0 = torch.zeros(cfg.measurement_count, 2)
        branch = 2 * torch.bernoulli(.5 * torch.ones(cfg.measurement_count)) - 1
        pos0[:, 1] = 1 + cfg.initial_condition_noise_epsilon * (.5 * torch.rand(cfg.measurement_count) + .5) * branch
        pos = pos0.clone()

        t = 6 * torch.rand(cfg.measurement_count)

        pos[:, 0] = t
        pos[:, 1] = (pos0[:, 1] - 1) * torch.exp(.1 * t**2 + pos0[:, 0] * t) + 1

        vel = torch.stack((
            torch.ones(pos.size(0)),
            .2 * pos[:, 0] * (pos[:, 1] - 1)
        )).T

        return Motif._to_dataframe(t, pos, vel)

    @staticmethod
    def _to_dataframe(t, pos, vel):
        return pd.DataFrame(
            torch.cat((t[:, None], pos, vel), axis=1),
            columns=['t', 'x1', 'x2', 'v1', 'v2']
        )


def generate_scvelo_simulation(cfg):
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


class SCVeloDataset:
    @staticmethod
    def download_h5ad(cfg, data_dir):
        if cfg.dataset is datasets.UMapDataset.BONEMARROW:
            adata = scvelo.datasets.bonemarrow(data_dir/cfg.h5ad_path)
        elif cfg.dataset is datasets.UMapDataset.DENTATE_GYRUS:
            adata = scvelo.datasets.dentategyrus(data_dir/cfg.h5ad_path)
        elif cfg.dataset is datasets.UMapDataset.FOREBRAIN:
            try:
                adata = scvelo.datasets.forebrain(data_dir/cfg.h5ad_path)
            except (TypeError, anndata._io.utils.AnnDataReadError):
                f = tables.open_file(data_dir/cfg.h5ad_path, mode='r+')
                # these are empty
                f.remove_node('/row_graphs')
                f.remove_node('/col_graphs')
                # rename to match AnnData data structure
                f.rename_node('/row_attrs', 'obs')
                f.rename_node('/col_attrs', 'var')
                f.rename_node('/matrix', 'X')
                f.close()
                adata = scvelo.datasets.forebrain(data_dir/cfg.h5ad_path)
        elif cfg.dataset is datasets.UMapDataset.PANCREAS:
            adata = scvelo.datasets.pancreas(data_dir/cfg.h5ad_path)
        elif cfg.dataset is datasets.UMapDataset.PBMC68K:
            adata = scvelo.datasets.pbmc68k(data_dir/cfg.h5ad_path)
        else:
            raise ValueError(f'Unknown umap dataset: {cfg.dataset}')
        return adata

    @staticmethod
    def process_umap(cfg, adata):
        if cfg.dataset is datasets.UMapDataset.FOREBRAIN:
            scvelo.pp.remove_duplicate_cells(adata)
            scvelo.pp.neighbors(adata)

        scvelo.pp.filter_and_normalize(adata)
        scvelo.pp.moments(adata)
        scvelo.tl.velocity(adata, mode='stochastic')

        scvelo.tl.velocity_graph(adata)
        scvelo.tl.velocity_pseudotime(adata)

        scvelo.tl.umap(adata, n_components=cfg.umap_dimension)
        scvelo.tl.velocity_embedding(adata, basis='umap')

        return dict(
            t=adata.obs.velocity_pseudotime,
            pos=adata.obsm['X_umap'],
            vel=adata.obsm['velocity_umap']
        )


def split_train_val_test(ds, frac_train, frac_val, frac_test, rng_seed):
    rng = np.random.default_rng(seed=rng_seed)
    idx = rng.permutation(len(ds))
    split_idxs = (len(idx) * np.array([frac_train, 1 - frac_val - frac_test, 1 - frac_test])).astype(int)
    train, _, val, test = np.split(idx, split_idxs)

    return dict(train=ds[train], val=ds[val], test=ds[test])


def get_dataset_df(cfg, data_dir, rng_seed=0):
    with pl.utilities.seed.isolate_rng():
        pl.seed_everything(rng_seed, workers=True)
        if isinstance(cfg, datasets.SimpleMotif):
            df = Motif.generate_simple(cfg)
        elif isinstance(cfg, datasets.OscillationMotif):
            df = Motif.generate_oscillation(cfg)
        elif isinstance(cfg, datasets.BifurcationMotif):
            df = Motif.generate_bifurcation(cfg)
        elif isinstance(cfg, datasets.H5adUMap):
            dims = np.arange(1, cfg.umap_dimension + 1)
            cols_pos = [f'x{i}' for i in dims]
            cols_vel = [f'v{i}' for i in dims]
            if (data_dir/cfg.processed_path).exists():
                df = pd.read_parquet(data_dir/cfg.processed_path)
            else:
                adata = SCVeloDataset.download_h5ad(cfg, data_dir)
                umap_data = SCVeloDataset.process_umap(cfg, adata)
                data = np.concatenate((
                    umap_data['t'].to_numpy()[:, None],
                    umap_data['pos'], umap_data['vel']
                ), axis=1)
                df = pd.DataFrame(
                    data=data,
                    columns=['t', *cols_pos, *cols_vel]
                )
                df.to_parquet(data_dir/cfg.processed_path, index=False)
        else:
            raise ValueError(f'Unknown dataset: {cfg}')

        if cfg.reverse_velocities:
            df[cols_vel] = -df[cols_vel]

        df['measurement_id'] = range(len(df))

        return df


def get_dataset(cfg, data_dir, rng_seed=0):
    with pl.utilities.seed.isolate_rng():
        pl.seed_everything(rng_seed, workers=True)
        df = get_dataset_df(cfg, data_dir, rng_seed=rng_seed)
        ds = Dataset(process_measurements(df, cfg.time_step_count_sparsify, cfg.neighbor_count, 0))
        splits = split_train_val_test(ds, frac_train=cfg.frac_train, frac_val=cfg.frac_val, frac_test=cfg.frac_test, rng_seed=rng_seed)
        if cfg.limit_batch_count_train:
            splits['train'] = splits['train'][:cfg.batch_count_train]

        return splits


def get_merged_dataset(cfg, data_dir, rng_seed=0):
    splits = defaultdict(list)
    for cfg_dataset in cfg.datasets:
        for split, data in get_dataset(cfg_dataset, data_dir, rng_seed=rng_seed).items():
            splits[split].append(data)
    for k, data_lists in splits.items():
        splits[k] = DatasetMerged(data_lists)
    return splits


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = conf.get_engine()
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        pprint.pp(cfg)
        splits = get_merged_dataset(cfg, cfg.data_dir, rng_seed=cfg.rng_seed)
        pprint.pp(splits)
        print('end')


if __name__ == "__main__":
    main()
