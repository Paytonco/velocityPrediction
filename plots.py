import copy

import hydra
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


def get_num_neighbors(path):
    return int(path.partition('nn_')[2].partition('_')[0])


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


def loss_by_num_neighbors(cfg, path_models, datasets):
    train, val = datasets
    ds = next(iter(train))

    paths, models = zip(*path_models)
    neighbors = torch.tensor(list(map(get_num_neighbors, paths)))
    losses = torch.tensor([toyModel
              .calc_loss(ds.velocities, m(ds.positions, ds.neighbors))
              for m in models])

    # use neighbors / 2 - 1 to set up scatter groups
    scatter_index = neighbors // 2 - 1
    mean, stddev, n = scatter_mean_stddev(losses, scatter_index)


    fig, ax = plt.subplots()
    # plot_confidence_interval(ax, scatter(neighbors // 2, scatter_index, reduce='max'), mean, stddev, n)
    z_star = 1.96
    xs = torch.unique_consecutive(neighbors // 2)
    ax.set_title('Loss per $n$')
    ax.errorbar(xs, mean, z_star * stddev / n.sqrt(), fmt='o', linewidth=2, capsize=6)
    ax.set_xlabel('$n$')
    ax.set_xticks(xs)
    ax.set_ylabel('MSE')
    fig.savefig(f'loss_by_num_neighbors.{IMG_FMT}', **SAVEFIG_SETTINGS)
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
        model, datasets = toyModel.load(model_checkpoint['cfg'])
        datasets = datasets[:2]
        model.eval()
        m = model
        m.load_state_dict(model_checkpoint['model_state_dict'])
        models.append((path, m))

    loss_by_num_neighbors(cfg, models, datasets)


if __name__ == '__main__':
    run()
