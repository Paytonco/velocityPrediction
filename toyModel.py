import itertools
import time
import typing
from pathlib import Path

import hydra
import omegaconf
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_context('paper', font_scale=1.4)


IMG_FMT = 'pdf'
SAVEFIG_SETTINGS = dict(format=IMG_FMT, bbox_inches='tight')


# set in setup
np_rng = None
torch_rng = None


def normalize(vec):
    return nn.functional.normalize(vec, dim=-1)


class Batch(typing.NamedTuple):
    positions: torch.Tensor
    neighbors: torch.Tensor
    velocities: torch.Tensor

    def to(self, device):
        return Batch(
            positions=self.positions.to(device),
            neighbors=self.neighbors.to(device),
            velocities=self.velocities.to(device)
        )

    def numpy(self):
        return self.__class__(
            self.positions.cpu().detach().numpy(),
            self.neighbors.cpu().detach().numpy(),
            self.velocities.cpu().detach().numpy()
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}' +
            '(' +
            f'positions={self.positions.shape}, ' +
            f'neighbors={self.neighbors.shape}, ' +
            f'velocities={self.velocities.shape}' +
            ')'
        )

    def __getitem__(self, idx):
        return self.__class__(
            self.positions[idx],
            self.neighbors[idx],
            self.velocities[idx]
        )


    def __len__(self):
        return self.positions.shape[0]


class DataSimple(Dataset):
    def __init__(self, center_pnt_idx, num_neighbors, size, epsilon):
        super().__init__()
        self.center_pnt_idx = center_pnt_idx
        self.num_neighbors = num_neighbors
        self.size = size
        self.epsilon = epsilon

        self.positions, self.neighbors = self.generate_positions_and_neighbors()
        self.velocities = self.generate_velocities()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        datum = self.positions[idx], self.neighbors[idx]
        target = self.velocities[idx]
        return datum, target

    def _phase_space_states(self, num_pnts):
        x0 = torch.zeros(num_pnts, 2)
        x0[:, 1] = self.epsilon * torch.rand(num_pnts, generator=torch_rng)
        x = x0.clone()
        t = 3 * torch.rand(num_pnts, generator=torch_rng)
        x[:, 0] = t
        x[:, 1] = (x0[:, 1] - 1) * torch.exp(.1 * t**2 + x0[:, 0] * t) + 1

        idx_sort_time = t.argsort()

        return torch.cat((t[:, None] / 3, x), dim=-1)[idx_sort_time]

    def generate_positions_and_neighbors(self):
        num_pnts = self.num_neighbors + 1
        states = self._phase_space_states(self.size * num_pnts).view(self.size, num_pnts, -1)
        idx = torch.zeros(num_pnts, dtype=bool)
        idx[self.center_pnt_idx] = True
        positions = states[:, idx]
        neighbors = states[:, ~idx]

        return positions, neighbors

    def generate_velocities(self):
        pos = self.positions
        return torch.cat((
            torch.ones(pos.size(0), 1),
            .2 * pos[..., 1] * (pos[..., 2] - 1)
        ), dim=-1)


class DataBifurcation(Dataset):
    def __init__(self, center_pnt_idx, num_neighbors, size, epsilon):
        super().__init__()
        self.center_pnt_idx = center_pnt_idx
        self.num_neighbors = num_neighbors
        self.size = size
        self.epsilon = epsilon

        self.positions, self.neighbors = self.generate_positions_and_neighbors()
        self.velocities = self.generate_velocities()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        datum = self.positions[idx], self.neighbors[idx]
        target = self.velocities[idx]
        return datum, target

    def _phase_space_states(self, num_pnts):
        x0 = torch.zeros(num_pnts, 2)
        branch = 2 * torch.bernoulli(.5 * torch.ones(num_pnts)) - 1
        x0[:, 1] = 1 + self.epsilon * (.5 * torch.rand(num_pnts, generator=torch_rng) + .5) * branch
        x = x0.clone()
        t = 6 * torch.rand(num_pnts, generator=torch_rng)
        x[:, 0] = t
        x[:, 1] = (x0[:, 1] - 1) * torch.exp(.1 * t**2 + x0[:, 0] * t) + 1

        idx_sort_time = t.argsort()

        return torch.cat((t[:, None] / 6, x), dim=-1)[idx_sort_time]

    def generate_positions_and_neighbors(self):
        num_pnts = self.num_neighbors + 1
        states = self._phase_space_states(self.size * num_pnts).view(self.size, num_pnts, -1)
        idx = torch.zeros(num_pnts, dtype=bool)
        idx[self.center_pnt_idx] = True
        positions = states[:, idx]
        neighbors = states[:, ~idx]

        return positions, neighbors

    def generate_velocities(self):
        pos = self.positions
        return torch.cat((
            torch.ones(pos.size(0), 1),
            .2 * pos[..., 1] * (pos[..., 2] - 1)
        ), dim=-1)


class DataOscillation(Dataset):
    """
    """
    def __init__(self, center_pnt_idx, num_neighbors, size, epsilon):
        super().__init__()
        self.center_pnt_idx = center_pnt_idx
        self.num_neighbors = num_neighbors
        self.size = size
        self.epsilon = epsilon

        self.positions, self.neighbors = self.generate_positions_and_neighbors()
        self.velocities = self.generate_velocities()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        datum = self.positions[idx], self.neighbors[idx]
        target = self.velocities[idx]
        return datum, target

    def _phase_space_states(self, num_pnts):
        x0 = torch.zeros(num_pnts, 2)
        x0[:, 0] = 1 + self.epsilon * (2 * torch.rand(num_pnts, generator=torch_rng) - 1)
        x = x0.clone()
        t = 2 * np.pi * torch.rand(num_pnts, generator=torch_rng)
        x[:, 0] = x0[:, 0] * torch.cos(t) + x0[:, 1] * torch.sin(t)
        x[:, 1] = -x0[:, 0] * torch.sin(t) + x0[:, 1] * torch.cos(t)

        idx_sort_time = t.argsort()

        return torch.cat((t[:, None] / (2 * np.pi), x), dim=-1)[idx_sort_time]

    def generate_positions_and_neighbors(self):
        num_pnts = self.num_neighbors + 1
        states = self._phase_space_states(self.size * num_pnts).view(self.size, num_pnts, -1)
        idx = torch.zeros(num_pnts, dtype=bool)
        idx[self.center_pnt_idx] = True
        positions = states[:, idx]
        neighbors = states[:, ~idx]

        return positions, neighbors

    def generate_velocities(self):
        pos = self.positions
        return torch.cat((pos[..., 2], -pos[..., 1]), dim=-1)


class Model2(nn.Module):
    """
    Model for predicting the unit velocity vector of a point based on a set
    of neighboring points.
    """
    def __init__(self, vector_dims):
        super().__init__()

        self.vector_dims = vector_dims

        self.model = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, positions, neighbors):
        norm_diffs = neighbors - positions
        dist = (norm_diffs[..., 1:]**2).sum(-1)
        arg = torch.dstack((norm_diffs[..., 0], dist))
        vel = (normalize(norm_diffs[..., 1:]) * self.model(arg)).sum(1)
        return vel


class Model(nn.Module):
    """
    Model for predicting the unit velocity vector of a point based on a set
    of neighboring points.
    """
    def __init__(self, vector_dims, num_neighbors):
        super().__init__()

        self.vector_dims = vector_dims
        self.num_neighbors = num_neighbors

        self.model = nn.Sequential(
            nn.Linear((vector_dims + 1) * num_neighbors + 1, num_neighbors * 2),
            nn.ReLU(),
            nn.Linear(num_neighbors * 2, num_neighbors * 2),
            nn.ReLU(),
            nn.Linear(num_neighbors * 2, num_neighbors),
        )

    def forward(self, positions, neighbors):
        norm_diffs = normalize(neighbors - positions)
        arg = torch.cat((norm_diffs, positions), dim=1).flatten(1)[:, :-self.vector_dims]
        vel = (norm_diffs * self.model(arg)[..., None]).sum(1)[:, 1:]
        return vel


def calc_loss(vel_given, vel_predicted):
    """
    Mean of the square distance between the velocity vectors.
    """
    dist = ((vel_given - vel_predicted)**2).sum(-1).mean()
    dot = -(vel_given * vel_predicted).sum(-1).mean()
    loss = dist

    return loss


def get_model(dim):
    return Model2(dim)


def get_dataset(cfg):
    if cfg.dataset.name == 'simple':
        ds = DataSimple
    elif cfg.dataset.name == 'bifurcation':
        ds = DataBifurcation
    elif cfg.dataset.name == 'oscillation':
        ds = DataOscillation
    else:
        raise ValueError(f'Invalid dataset name in config: {cfg.dataset.name}')

    dataset = ds(cfg.dataset.num_neighbors // 2, cfg.dataset.num_neighbors, cfg.dataset.size, cfg.dataset.epsilon)
    size = len(dataset)
    idx = torch.randperm(size, generator=torch_rng)
    size_train = int(cfg.dataset.size_train * size)
    size_val = int(cfg.dataset.size_val * size)
    size_test = int(cfg.dataset.size_test * size)
    train = dataset[idx[:size_train]]
    val = dataset[idx[-(size_val+size_test):-size_test]]
    test = dataset[idx[-size_test:]]

    return train, val, test


def setup(cfg):
    """
    Set some global settings.
    """
    with omegaconf.open_dict(cfg):
        cfg.load.checkpoint_path = Path(cfg.load.checkpoint_path)

    # Reproduciblity
    global np_rng
    global torch_rng
    np_rng = np.random.default_rng(cfg.setup.rng_seed)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(cfg.setup.rng_seed)
    if cfg.setup.deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        with omegaconf.open_dict(cfg):
            p = cfg.load.checkpoint_path
            cfg.load.checkpoint_path = p.parent / f'{p.stem}_{np.random.randint(10_000)}{p.suffix}'


def collate_fn(cfg, arg):
    return Batch(
        positions=arg[0][0], neighbors=arg[0][1],
        velocities=arg[1]
    ).to(cfg.setup.device)


def load(cfg):
    """
    Create the dataloaders, and construct the model.
    """
    train, val, test = get_dataset(cfg)

    # FIXME: setting shuffle=True flips the order of the dataset's __getitem__ result
    train = DataLoader(train, batch_size=cfg.dataset.batch_size_train, shuffle=False, collate_fn=lambda arg: collate_fn(cfg, arg))
    val = DataLoader(val, batch_size=cfg.dataset.batch_size_val, shuffle=False, collate_fn=lambda arg: collate_fn(cfg, arg))
    test = DataLoader(test, batch_size=cfg.dataset.batch_size_test, shuffle=False, collate_fn=lambda arg: collate_fn(cfg, arg))

    model = get_model(cfg.dataset.dim).to(cfg.setup.device)

    return model, (train, val, test)


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    output = model(data.positions, data.neighbors)

    loss = calc_loss(data.velocities, output)

    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def val(model, data):
    model.eval()

    output = model(data.positions, data.neighbors)

    loss = calc_loss(data.velocities, output)

    return loss


def run_training(cfg, model, dataloader_train, dataloader_val):
    optimizer = torch.optim.AdamW(model.parameters())

    best = 1e8
    for epoch in range(cfg.train.epochs):
        t_start = time.time()

        loss_train = 0
        for data in dataloader_train:
            loss_train += train(model, optimizer, data)

        loss_val = 0
        for data in dataloader_val:
            loss_val += val(model, data)

        loss_train, loss_val = (
            loss_train / len(dataloader_train),
            loss_val / len(dataloader_val)
        )

        t_end = time.time()

        perf_metric = loss_val

        if perf_metric < best:
            torch.save(dict(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                loss_val=loss_val,
                cfg=cfg
            ), cfg.load.checkpoint_path)

        epoch_stats = dict(
            epoch=epoch+1,
            loss_train=loss_train,
            loss_val=loss_val,
            time=t_end-t_start
        )

        if cfg.train.print_stats:
            print(
                'Epoch: {epoch:03d}, '
                'loss_train: {loss_train:.7f}, '
                'loss_val: {loss_val:.7f}, '
                'time: {time:3.3f}, '
                ''.format(**epoch_stats)
            )


def plot(path, model, data):
    perm = torch.randperm(len(data), generator=torch_rng)[:40]
    output = model(data.positions, data.neighbors)[perm].cpu().detach().numpy()
    data = data[perm].numpy()
    fig, ax = plt.subplots()
    ax.scatter(data.positions[:, 0, 1], data.positions[:, 0, 2],
              label='State', color='darkorange')
    ax.quiver(data.positions[:, 0, 1], data.positions[:, 0, 2],
              data.velocities[:, 0], data.velocities[:, 1], label='True', color='deepskyblue')
    ax.quiver(data.positions[:, 0, 1], data.positions[:, 0, 2],
              output[:, 0], output[:, 1], label='Inferred', color='blue')
    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.savefig(f'{path}.{IMG_FMT}', **SAVEFIG_SETTINGS)
    plt.close(fig)


@hydra.main(version_base=None, config_path='configs', config_name='main')
def run(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    params = (cfg.dataset.size, cfg.dataset.num_neighbors, cfg.dataset.epsilon)
    if all(map(np.isscalar, params)):
        param_sets = [params]
    else:
        param_sets = list(itertools.product(*map(np.atleast_1d, params)))

    for rep in range(cfg.train.repeats + 1):
        for i, (size, num_neighbors, epsilon) in enumerate(param_sets):
            print(f'Run {rep + 1} of {cfg.train.repeats + 1}: Param Set {i + 1} of {len(param_sets)}')
            with omegaconf.open_dict(cfg):
                cfg.dataset.size, cfg.dataset.num_neighbors, cfg.dataset.epsilon = int(size), int(num_neighbors), float(epsilon)
                cfg.load.checkpoint_path = Path(f'../../out/paytonco/cp_ds_{cfg.dataset.name}_nn_{cfg.dataset.num_neighbors}_sz_{cfg.dataset.size}_eps_{cfg.dataset.epsilon}.pt')
            setup(cfg)

            model, dataloaders = load(cfg)

            run_training(cfg, model, *dataloaders[:-1])

    model.eval()
    plot(f'output_train_{cfg.dataset.name}', model, next(iter(dataloaders[0]))[:100])
    plot(f'output_val_{cfg.dataset.name}', model, next(iter(dataloaders[1]))[:100])



if __name__ == '__main__':
    run()
