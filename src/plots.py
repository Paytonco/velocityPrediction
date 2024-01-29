from pathlib import Path

import hydra
import omegaconf
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import seaborn as sns
import wandb
import matplotlib.pyplot as plt

import datasets
import runs


def get_runs(cfg):
    rs = runs.query_runs(cfg.wandb.entity, cfg.wandb.project,
                         {'$or': [{'name': i} for i in cfg.run_ids]}, {}, {})
    for r in rs:
        run_cfg = OmegaConf.create(r.config)
        run_dir = Path(cfg.out_dir)/'runs'/r.id
        run_datasests = {s: InMemoryDataset(run_dir) for s in ('train', 'val', 'test')}
        for k, v in run_datasests.items():
            v.load(run_dir/f'pred_{k}.pt')

        yield r.id, run_dir, run_cfg, run_datasests


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
    ax.scatter(pos[:, 0], pos[:, 1], label='State', c='orange')
    ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], color='deepskyblue')
    if hasattr(data, 'poi_vel_pred'):
        vel_pred = data.poi_vel_pred
        ax.quiver(pos[:, 0], pos[:, 1], vel_pred[:, 0], vel[:, 1], color='blue')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')


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
        data = next(iter(DataLoader(ds, batch_size=len(ds))))
        plot_field(ax, data)
        fig.savefig(f'{self.run_dir}/pred_{split}.{self.cfg.fmt}', format=self.cfg.fmt, bbox_inches='tight', pad_inches=.03)
        plt.close(fig)


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

    iter_runs(cfg, plotters)

    if cfg.plot.dataset.do:
        if cfg.get('dataset') is None:
            raise ValueError('No datasets selected. Select a dataset with "+dataset@dataset.<name>=<dataset_cfg>".')
        plot_dir = Path(cfg.plot_dir)/'dataset'
        plot_dir.mkdir()
        for k, v in cfg.dataset.items():
            for s, ds in zip(('train', 'val', 'test'), datasets.get_dataset(v, cfg.data_dir, rng_seed=cfg.rng_seed)):
                fig, ax = plt.subplots()
                ds = ds.shuffle()[:40]
                data = next(iter(DataLoader(ds, batch_size=len(ds))))
                plot_field(ax, data)
                fig.savefig(plot_dir/f'{k}_{s}.{cfg.fmt}', format=cfg.fmt, bbox_inches='tight', pad_inches=.03)
                plt.close(fig)

    wandb.finish()
    print('WandB Run ID')
    print(wrun.id)


if __name__ == '__main__':
    main()
