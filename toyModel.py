import time
import typing

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


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


class VectorData(Dataset):
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
        x0 = self.epsilon * (2 * torch.rand(num_pnts, 2) - 1)
        x0[:, 0] += 1
        x0[:, 1] = 0
        x = x0.clone()
        t = 2 * np.pi * torch.rand(num_pnts)
        x[:, 0] = x0[:, 0] * torch.cos(t) + x0[:, 1] * torch.sin(t)
        x[:, 1] = -x0[:, 0] * torch.sin(t) + x0[:, 1] * torch.cos(t)

        idx_sort_time = t.argsort()

        return torch.cat((t[:, None], x), dim=-1)[idx_sort_time]

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
        return normalize(torch.cat((pos[..., 2], -pos[..., 1]), dim=-1))


class Model2(nn.Module):
    """
    Model for predicting the unit velocity vector of a point based on a set
    of neighboring points.
    """
    def __init__(self, vector_dims, num_neighbors):
        super().__init__()

        self.vector_dims = vector_dims
        self.num_neighbors = num_neighbors

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


def get_model(dim, num_neighbors):
    return Model2(dim, num_neighbors)


def setup(cfg):
    """
    Set some global settings.
    """
    # Reproduciblity
    np.random.seed(cfg.setup.rng_seed)
    torch.manual_seed(cfg.setup.rng_seed)
    torch.backends.cudnn.deterministic = True


def collate_fn(cfg, arg):
    return Batch(
        positions=arg[0][0], neighbors=arg[0][1],
        velocities=arg[1]
    ).to(cfg.setup.device)


def load(cfg):
    """
    Create the dataloaders, and construct the model.
    """
    dataset = VectorData(cfg.dataset.center_pnt_idx, cfg.dataset.num_neighbors, cfg.dataset.size, cfg.dataset.epsilon)

    idx = torch.bernoulli(.5 * torch.ones(len(dataset))).to(bool)

    train = dataset[idx]  # [:cfg.dataset.size_train]
    val = dataset[~idx]  # [-(cfg.dataset.size_val+cfg.dataset.size_test):len(dataset)-cfg.dataset.size_test]
    test = None  # dataset[-cfg.dataset.size_test:]

    # FIXME: setting shuffle=True flips the order of the dataset's __getitem__ result
    train = DataLoader(train, batch_size=cfg.dataset.batch_size_train, shuffle=False, collate_fn=lambda arg: collate_fn(cfg, arg))
    val = DataLoader(val, batch_size=cfg.dataset.batch_size_val, shuffle=False, collate_fn=lambda arg: collate_fn(cfg, arg))
    test = None  # DataLoader(test, batch_size=cfg.dataset.batch_size_test, shuffle=False, collate_fn=lambda arg: collate_fn(cfg, arg))

    model = get_model(cfg.dataset.dim, cfg.dataset.num_neighbors).to(cfg.setup.device)

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
                loss_val=loss_val
            ), cfg.load.checkpoint_path)

        epoch_stats = dict(
            epoch=epoch+1,
            loss_train=loss_train,
            loss_val=loss_val,
            time=t_end-t_start
        )

        print(
            'Epoch: {epoch:03d}, '
            'loss_train: {loss_train:.7f}, '
            'loss_val: {loss_val:.7f}, '
            'time: {time:3.3f}, '
            ''.format(**epoch_stats)
        )


def plot(path, model, data):
    output = model(data.positions, data.neighbors).cpu().detach().numpy()
    data = data.numpy()
    fig, ax = plt.subplots()
    ax.scatter(data.neighbors[:, :, 1], data.neighbors[:, :, 2],
              label='Neighbors', color='green')
    ax.quiver(data.positions[:, 0, 1], data.positions[:, 0, 2],
              data.velocities[:, 0], data.velocities[:, 1], label='True', color='red')
    ax.quiver(data.positions[:, 0, 1], data.positions[:, 0, 2],
              output[:, 0], output[:, 1], label='Output', color='blue')
    ax.legend()
    fig.savefig(f'{path}.pdf', format='pdf')
    plt.close(fig)


@hydra.main(version_base=None, config_path='configs', config_name='main')
def run(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    setup(cfg)
    model, dataloaders = load(cfg)

    run_training(cfg, model, *dataloaders[:-1])

    model.eval()
    plot('output_train', model, next(iter(dataloaders[0]))[:100])
    plot('output_val', model, next(iter(dataloaders[1]))[:100])



if __name__ == '__main__':
    run()
