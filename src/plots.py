from pathlib import Path

import hydra
import omegaconf
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import seaborn as sns
import wandb
import matplotlib.pyplot as plt

import datasets


def plot_field(ax, data):
    pos, vel = data.poi_pos, data.poi_vel
    ax.scatter(pos[:, 0], pos[:, 1], label='State', c='orange')
    ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], color='deepskyblue')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')


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

    wandb.finish()
    print('WandB Run ID')
    print(wrun.id)


if __name__ == '__main__':
    main()
