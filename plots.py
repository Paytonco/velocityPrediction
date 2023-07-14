import copy

import hydra
import omegaconf
from omegaconf import OmegaConf
import numpy as np
import torch
from torch_geometric.utils import scatter
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import collections

import toyModel


sns.set_context('paper', font_scale=1.4)


IMG_FMT = 'pdf'
SAVEFIG_SETTINGS = dict(format=IMG_FMT, bbox_inches='tight')


def get_param_from_path(tag, path):
    return int(path.partition(f'{tag}_')[2].partition('_')[0])

def get_num_neighbors(path):
    return get_param_from_path('nn', path)

def get_size(path):
    return get_param_from_path('sz', path)

def scatter_mean_stddev(src, index):
    mean = scatter(src, index, reduce='mean')
    src_sqrd = scatter(src**2, index)
    two_mean_src = 2 * mean * scatter(src, index)
    n = scatter(torch.ones(src.shape), index)

    stddev = ((src_sqrd - two_mean_src + n * mean**2) / (n - 1)).sqrt()

    return mean, stddev, n


def plot_confidence_interval(ax, x, mean, stddev, n, z=1.96, color='#2187bb', horizontal_line_width=0.25):
    confidence_interval = z * stddev / n**(1/2)

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval

    stack_T_ravel = lambda a, b: torch.stack((a, b)).T.ravel()

    vert_pnts = torch.stack((stack_T_ravel(x, x), stack_T_ravel(top, bottom)))
    top_bar = torch.stack((stack_T_ravel(left, right), stack_T_ravel(top, top)))
    bottom_bar = torch.stack((stack_T_ravel(left, right), stack_T_ravel(bottom, bottom)))
    pnts = torch.stack((vert_pnts, top_bar, bottom_bar)).permute(0, 2, 1).reshape(-1, 2, 2)
    verticals = collections.LineCollection(
        pnts,
        linewidths=2
    )

    ax.add_collection(verticals)

    ax.plot(x, mean, 'o', color='#f44336')


def scatter_idx(a):
    res = a.clone()
    for i, v in enumerate(torch.unique(a)):
        res[res == v] = i
    return res


def loss_by_param(ax, cfg, xs, path_models):
    paths, models, datasets = zip(*path_models)
    train, val = zip(*datasets)

    losses = []
    for m, d in zip(models, val):
        output = m(d.positions, d.neighbors)
        loss = toyModel.calc_loss(d.velocities, output)
        losses.append(loss)
    losses = torch.tensor(losses)

    idx = scatter_idx(xs)
    mean, stddev, n = scatter_mean_stddev(losses, idx)

    z_star = 1.96
    x_axis = torch.unique_consecutive(xs)
    ax.errorbar(x_axis, mean, z_star * stddev / n.sqrt(), fmt='o', linewidth=2, capsize=6)
    ax.set_xlabel('$n$')
    ax.set_xticks(x_axis)
    ax.set_ylabel('MSE')


def plot_dataset(name, data):
    # output = model(data.positions, data.neighbors).cpu().detach().numpy()
    data = data.numpy()
    fig, ax = plt.subplots()
    ax.scatter(data.positions[:, 0, 1], data.positions[:, 0, 2],
              label='Neighbors', color='darkorange')
    ax.quiver(data.positions[:, 0, 1], data.positions[:, 0, 2],
              data.velocities[:, 0], data.velocities[:, 1], label='True', color='deepskyblue')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.savefig(f'dataset_{name}.{IMG_FMT}', **SAVEFIG_SETTINGS)
    plt.close(fig)


@hydra.main(version_base=None, config_path='configs', config_name='plots')
def run(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    torch.backends.cudnn.deterministic = True
    # Handle GPU driver madness causing much greater error in f32 calculations.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    models = []
    for path in sorted(cfg.model_paths):
        model_checkpoint = torch.load(path)
        model_cfg = model_checkpoint['cfg']
        toyModel.setup(model_cfg)
        model = toyModel.get_model(model_cfg.dataset.dim).to(model_cfg.setup.device)
        datasets = toyModel.get_dataset(model_cfg)[:2]
        datasets = list(map(lambda a: toyModel.collate_fn(model_cfg, a), datasets))
        model.eval()
        model.load_state_dict(model_checkpoint['model_state_dict'])
        models.append((path, model, datasets))

    plot_dataset(model_cfg.dataset.name, datasets[0][torch.randperm(datasets[0].positions.size(0), generator=toyModel.torch_rng)[:40]])

    if cfg.plot == 'num_neighbors':
        xs = torch.tensor(list(map(get_num_neighbors, (p[0] for p in models)))) // 2
    elif cfg.plot == 'size':
        xs = torch.tensor(list(map(get_size, (p[0] for p in models))))
    else:
        raise ValueError(f'Invalid plot param: {cfg.plot}')

    fig, ax = plt.subplots()
    loss_by_param(ax, cfg, xs, models)
    fig.savefig(f'loss_by_{cfg.plot}.{IMG_FMT}', **SAVEFIG_SETTINGS)
    plt.close(fig)


if __name__ == '__main__':
    run()
