from pathlib import Path

import hydra
import omegaconf
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import seaborn as sns
import wandb
import matplotlib
import matplotlib.pyplot as plt
from anndata import AnnData
import scvelo

import datasets
import main as trainer_main
import utils
import wandbruns


def adata_from_pos(pos):
    adata = AnnData(pos)
    scvelo.pp.neighbors(adata)
    scvelo.tl.umap(adata, n_components=2)
    return adata


def compute_adata_velocity(adata, vel):
    adata.layers['velocity'] = vel
    """
    Motivation for the next line, probably wrong.

    Our positions (poi_pos) are the gene expressions, so the change in gene
    expression is the difference of two positions. This approximates the
    velocity vector set above.
    """
    adata.layers['spliced'] = adata.X
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_embedding(adata, basis='umap')

    return adata.obsm['velocity_umap']


def get_runs(cfg):
    rs = wandbruns.query_runs(cfg.wandb.entity, cfg.wandb.project,
                         {'$or': [{'name': i} for i in cfg.run_ids]}, {}, {})
    for r in rs:
        run_cfg = OmegaConf.create(r.config)
        run_dir = Path(cfg.out_dir)/'runs'/r.id

        if r.job_type == 'fit':
            run_datasets = {}
        else:
            run_datasets = {s: InMemoryDataset(run_dir) for s in ('train', 'val', 'test')}
            for k, v in run_datasets.items():
                v.load(run_dir/f'pred_{k}.pt')

            if cfg.use_umap_2d:
                # training_datasets = []
                # for v in OmegaConf.masked_copy(run_cfg, 'dataset').dataset:
                #     OmegaConf.update(v, 'umap.n_components', 2)
                #     OmegaConf.update(v, 'csv.name_suffix', 'umap_n_components_2')
                #     training_datasets.append(datasets.get_dataset(v, rng_seed=run_cfg.rng_seed))
                # training_datasets = map(datasets.DatasetMerged, zip(*training_datasets))
                # training_datasets = {k: s for k, s in zip(('train', 'val', 'test'), training_datasets)}

                for k in run_datasets:
                    data = run_datasets[k]._data
                    # training_data = training_datasets[k]._data

                    dims_most_important = data.poi_pos.var(0, correction=0).topk(2).indices
                    for k in ('poi_pos', 'poi_vel', 'poi_vel_pred'):
                    # for k in ('poi_vel', 'poi_vel_pred'):
                        data[k] = data[k][:, dims_most_important]

                    # data_df = pd.DataFrame(dict(
                    #     idx=range(data.poi_measurement_id.size(0)),
                    #     measurement_id=data.poi_measurement_id.numpy()
                    # ))
                    # training_data_df = pd.DataFrame(dict(
                    #     idx=range(training_data.poi_measurement_id.size(0)),
                    #     measurement_id=training_data.poi_measurement_id.numpy()
                    # ))
                    # mapping = pd.merge(
                    #     data_df, training_data_df,
                    #     on='measurement_id', suffixes=('_data', '_training_data')
                    # )  # preserves order of data_df's rows
                    #
                    # data.poi_pos = training_data.poi_pos[mapping['idx_training_data']]

                    # adata = adata_from_pos(data.poi_pos.numpy())
                    # poi_pos = adata.obsm['X_umap']
                    # poi_vel = compute_adata_velocity(adata, data.poi_vel.numpy())
                    # poi_vel_pred = compute_adata_velocity(adata, data.poi_vel_pred.numpy())
                    #
                    # data.poi_pos = torch.tensor(poi_pos)
                    # data.poi_vel = utils.normalize(torch.tensor(poi_vel))
                    # data.poi_vel_pred = utils.normalize(torch.tensor(poi_vel_pred))

        yield r.id, run_dir, run_cfg, run_datasets


def complete_edges(num_nodes):
    labels = torch.arange(num_nodes, dtype=torch.long)
    edge_index = torch.stack((
        # j: rows of adjacency matrix, and source node in message passing
        labels.repeat_interleave(num_nodes),
        # i: columns of adjacency matrix, and target node in message passing
        labels.repeat(num_nodes)
    ))
    edge_index = tg.utils.remove_self_loops(edge_index)[0]
    return edge_index


def iter_runs(cfg, plotters):
    for run_id, run_dir, run_cfg, run_datasets in tqdm(get_runs(cfg), total=len(cfg.run_ids), desc='Runs'):
        for p in plotters:
            p.iter_run(run_id, run_dir, run_cfg)
        for split, ds in run_datasets.items():
            for p in plotters:
                p.iter_split(split, ds)
        for p in plotters:
            p.end_iter_split()
    for p in plotters:
        p.end_iter_run()


def plot_field(ax, data):
    pos, vel = data.poi_pos, data.poi_vel
    if hasattr(data, 'poi_vel_pred'):
    # if False:
        color_data = (utils.normalize(vel) - utils.normalize(data.poi_vel_pred)).pow(2).sum(1) / 2
        # sc = ax.scatter(pos[:, 0], pos[:, 1], label='State', c=color_data, cmap='viridis', norm=matplotlib.colors.Normalize(vmin=0, vmax=2))
        sc = ax.scatter(pos[:, 0], pos[:, 1], label='State', c=color_data, cmap='viridis', vmin=0, vmax=2)
        cbar = ax.get_figure().colorbar(sc, ax=ax)  # , ticks=[0, 2], format=matplotlib.ticker.FixedFormatter(['0', '2']), label=r'$1 - \cos(\theta)$')
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('vel_pred', ['tab:orange', 'deepskyblue'])
        pos2 = torch.cat((pos, pos))
        indicator = torch.cat((torch.zeros(pos.size(0)), torch.ones(pos.size(0))))
        poi_vel_pred = data.poi_vel_pred
        # vel, poi_vel_pred = utils.normalize(vel), utils.normalize(poi_vel_pred)
        vel_pred = torch.cat((vel, poi_vel_pred))
        ax.quiver(pos2[:, 0], pos2[:, 1], vel_pred[:, 0], vel_pred[:, 1], color=cmap(indicator), label='Vec')
        ax.legend(
            [matplotlib.patches.Patch(color=cmap(i)) for i in [0., 1.]],  # the cmap inputs must be floats!
            ['True', 'Pred']
        )
    else:
        cmap = matplotlib.colormaps['Wistia']
        sc = ax.scatter(pos[:, 0], pos[:, 1], label='State', facecolors='none', c=data.poi_t, cmap='viridis', vmin=0, vmax=1)
        cbar = ax.get_figure().colorbar(sc, ax=ax)
        ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], color='tab:orange')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.margins(.1)


class Plotter:
    def __init__(self, cfg):
        self.cfg = cfg

    def iter_run(self, run_id, run_dir, run_cfg):
        pass

    def end_iter_run(self):
        pass

    def iter_split(self, split, ds):
        pass

    def end_iter_split(self):
        pass


class VelPred(Plotter):
    def iter_run(self, run_id, run_dir, run_cfg):
        self.run_dir = run_dir

    def iter_split(self, split, ds):
        fig, ax = plt.subplots()
        batch_size = len(ds) if split == 'test' else len(ds) // 4
        data = next(iter(DataLoader(ds, batch_size=batch_size)))
        plot_field(ax, data)
        fig.savefig(f'{self.run_dir}/pred_{split}.{self.cfg.fmt}', format=self.cfg.fmt, bbox_inches='tight', pad_inches=.03)
        plt.close(fig)


class MSESparseStepNeighborSetHeatmap(Plotter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mse_dfs = []
        self.dataset_sizes = {
            'pancreas': 3696,
            'bonemarrow': 5780,
            'dentategyrus': 2930,
            'forebrain': 20898
        }

    def iter_run(self, run_id, run_dir, run_cfg):
        self.run_id = run_id
        self.run_cfg = run_cfg

    def iter_split(self, split, ds):
        data = next(iter(DataLoader(ds, batch_size=len(ds))))
        mse = F.mse_loss(data.poi_vel, data.poi_vel_pred)
        data_subdir = self.run_cfg.dataset_summary.data_dir.split('/')[-1]
        df = pd.DataFrame(dict(
            split=split,
            sparsifier_step=self.run_cfg.dataset[0].sparsify_step_time / self.dataset_sizes[data_subdir],
            num_neighbors=self.run_cfg.dataset[0].num_neighbors,
            mse=mse.item(),
            source=self.run_id,
            umap_num_components=self.run_cfg.dataset_summary.umap_num_components,
        ), index=[0])
        self.mse_dfs.append(df)

    def end_iter_run(self):
        df = pd.concat(self.mse_dfs).reset_index(drop=True)
        for s in df['split'].unique():
            pivot = df[df['split'] == s].pivot(index='sparsifier_step', columns='num_neighbors', values='mse')
            print(f'Pivot of {s = }')
            print(pivot)
            fig, ax = plt.subplots(figsize=(7.04, 5.28))
            xx, yy = np.meshgrid(pivot.columns, pivot.index)

            heatmap = ax.pcolor(
                xx, yy, pivot.to_numpy(),
                cmap='viridis',
                vmin=.01, vmax=.04,
                linewidths=.2,
                edgecolors='white'
            )
            cbar = fig.colorbar(heatmap, ax=ax)
            cbar.outline.set_visible(False)

            ax.set_frame_on(False)
            ax.set_xticks(pivot.columns)
            #ax.set_yticks(pivot.index, pivot.index.to_series().round(1))
            ax.set_yticks(pivot.index.to_series().round(4))
            formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, -3))
            formatter.set_scientific(True)
            ax.yaxis.set_major_formatter(formatter)

            ax.set_xlabel('Num. Neighbors')
            ax.set_ylabel('Sparsity')
            ax.tick_params(axis='x', labelrotation=-20)
            ax.tick_params(axis='y', labelrotation=20)
            fig.tight_layout()
            fig.savefig(f'{self.cfg.plot_dir}/mse_sparsifier_step_neighbor_set_heatmap_{s}.{self.cfg.fmt}', format=self.cfg.fmt, bbox_inches='tight', pad_inches=.06)
            plt.close(fig)


class MSEUmapDimension(Plotter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mse_dfs = []
        self.labels_dict = {
            'pancreas': 'Pancreas',
            'dentategyrus': 'Dentate Gyrus',
            'forebrain': 'Forebrain',
            'bonemarrow': 'Bone Marrow'
        }

    def iter_run(self, run_id, run_dir, run_cfg):
        self.run_id = run_id
        self.run_cfg = run_cfg
        # training_run = wandbruns.query_runs(self.cfg.wandb.entity, self.cfg.wandb.project, {'name': run_id}, {}, {})[0]
        # training_history = training_run.history(samples=10000)
        # val_loss = training_history['val_loss'].dropna().to_numpy()[-1]
        # df = pd.DataFrame(dict(
        #     mse=val_loss,
        #     source=self.run_id,
        #     dataset=self.labels_dict[self.run_cfg.dataset_summary.data_dir.split('/')[-1]],
        #     umap_num_components=int(self.run_cfg.dataset_summary.umap_num_components),
        # ), index=[0])
        # self.mse_dfs.append(df)

    def iter_split(self, split, ds):
        data = next(iter(DataLoader(ds, batch_size=len(ds))))
        mse = F.mse_loss(data.poi_vel, data.poi_vel_pred)
        data_subdir = self.run_cfg.dataset_summary.data_dir.split('/')[-1]
        df = pd.DataFrame(dict(
            split=split,
            mse=mse.item(),
            source=self.run_id,
            dataset=self.labels_dict[data_subdir],
            umap_num_components=int(self.run_cfg.dataset_summary.umap_num_components),
        ), index=[0])
        self.mse_dfs.append(df)

    def end_iter_run(self):
        df = pd.concat(self.mse_dfs).reset_index(drop=True)
        for s in df['split'].unique():
            fig, ax = plt.subplots()
            # ax.set_yscale(matplotlib.scale.LogScale(ax, base=4))
            sns.lineplot(
                df[df['split'] == s],
                x='umap_num_components', y='mse',
                hue='dataset',
                err_style='bars',
                ax=ax,
            )
            ax.set_xlabel('Dimension')
            ax.set_ylabel(r'$\mathcal{L}_{\mathrm{MSE}}$')
            ax.get_legend().set_title(None)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(*zip(*sorted(zip(handles, labels), key=lambda t: t[1])), loc='upper right')
            # sns.move_legend(ax, 'upper right')
            fig.savefig(f'{self.cfg.plot_dir}/mse_umap_dimension_{s}.{self.cfg.fmt}', format=self.cfg.fmt, bbox_inches='tight', pad_inches=.03)


@hydra.main(version_base=None, config_path='../configs', config_name='plots')
def main(cfg):
    sns.set_context(cfg.plot.context)
    sns.set_style(cfg.plot.style)
    wrun = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        dir=cfg.wandb.dir,
        job_type='plots'
    )
    with omegaconf.open_dict(cfg):
        cfg.out_dir = str(Path(cfg.out_dir).resolve())
        cfg.plot_dir = str(Path(cfg.plot_dir)/wrun.id)
    Path(cfg.plot_dir).mkdir(parents=True)

    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    print(OmegaConf.to_yaml(cfg, resolve=True))

    plotters = []
    if cfg.plot.vel_pred.do:
        plotters.append(VelPred(cfg))
    if cfg.plot.mse.umap_dimension.do:
        plotters.append(MSEUmapDimension(cfg))
    if cfg.plot.mse.sparsifier_neighbor_set_heatmap.do:
        plotters.append(MSESparseStepNeighborSetHeatmap(cfg))

    iter_runs(cfg, plotters)

    if cfg.plot.dataset.do:
        if cfg.get('dataset') is None:
            raise ValueError('No datasets selected. Select a dataset with "+dataset@dataset.<name>=<dataset_cfg>".')
        plot_dir = Path(cfg.plot_dir)/'dataset'
        plot_dir.mkdir()
        for k, v in cfg.dataset.items():
            for s, ds in zip(('train', 'val', 'test'), map(datasets.DatasetMerged, zip(datasets.get_dataset(v, rng_seed=cfg.rng_seed)))):
                fig, ax = plt.subplots()
                with pl.utilities.seed.isolate_rng():
                    pl.seed_everything(cfg.rng_seed, workers=True)
                    ds = ds.shuffle()
                batch_size = len(ds) if s == 'test' else len(ds)
                data = next(iter(DataLoader(ds, batch_size=batch_size)))
                plot_field(ax, data)
                fig.savefig(plot_dir/f'{k}_{s}.{cfg.fmt}', format=cfg.fmt, bbox_inches='tight', pad_inches=.03)
                plt.close(fig)

    wandb.finish()
    print('WandB Run ID')
    print(wrun.id)


if __name__ == '__main__':
    main()
