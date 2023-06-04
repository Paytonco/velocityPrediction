import time
import typing

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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

    def __repr__(self):
        return (
            f'{self.__class__.__name__}' +
            '(' +
            f'positions={self.positions.shape}, ' +
            f'neighbors={self.neighbors.shape}, ' +
            f'velocities={self.velocities.shape}' +
            ')'
        )


class VectorData(Dataset):
    """
    Vector field v(t,x) = [1, x] where points (t, x) are sampled from
    [-5, 5] x [-5, 5] uniformly at random.
    """
    def __init__(self, dim, num_neighbors, size):
        """
        Parameters
        ----------
        dim: int
            The spatial dimensions of the position and velocity vectors.

        num_neighbors: int
            The number of points to use for finding the velocity at a given
            point.

        size: int
            The number of vectors to generate.
        """
        super().__init__()
        self.dim = dim
        self.num_neighbors = num_neighbors
        self.size = size

        self.positions = self.generate_positions()
        self.velocities = self.generate_velocities()
        self.neighbors = self.generate_neighbors()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        datum = self.positions[idx], self.neighbors[idx]
        target = self.velocities[idx]
        return datum, target

    def generate_positions(self):
        return 10 * (torch.rand(self.size, self.dim) - 0.5)

    def generate_velocities(self):
        velocities = self.positions.clone()
        velocities[:, 0] = 1.
        return normalize(velocities)

    def generate_neighbors(self):
        """
        Generate a set of neighboring points for each point from
        :func:`generate_positions`.
        """
        return 10 * (torch.rand(self.size, self.num_neighbors, self.dim) - 0.5)


class VelVector(nn.Module):
    """
    Model for predicting the unit velocity vector of a point based on a set
    of neighboring points.
    """
    def __init__(self, vector_dims, num_neighbors):
        super().__init__()

        self.vector_dims = vector_dims
        self.num_neighbors = num_neighbors

        self.model = nn.Sequential(
            nn.Linear(vector_dims, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )

    def norm_diffs(self, positions, neighbors):
        diffs = neighbors - positions[:, None, :]
        return normalize(diffs)

    def forward(self, positions, neighbors):
        norm_diffs = self.norm_diffs(positions, neighbors)
        vel = (norm_diffs * self.model(norm_diffs)).sum(1)
        return normalize(vel)


def calc_loss(vel_given, vel_predicted):
    """
    Mean of the square distance between the velocity vectors.
    """
    dist = ((vel_given - vel_predicted)**2).sum(-1).mean()
    loss = dist  # + angle

    return loss


def get_model(dim, num_neighbors):
    return VelVector(dim, num_neighbors)


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
    dataset = VectorData(cfg.dataset.dim, cfg.dataset.num_neighbors, cfg.dataset.size)

    train = dataset[:cfg.dataset.size_train]
    val = dataset[-(cfg.dataset.size_val+cfg.dataset.size_test):len(dataset)-cfg.dataset.size_test]
    test = dataset[-cfg.dataset.size_test:]

    # FIXME: setting shuffle=True flips the order of the dataset's __getitem__ result
    train = DataLoader(train, batch_size=cfg.dataset.batch_size_train, shuffle=False, collate_fn=lambda arg: collate_fn(cfg, arg))
    val = DataLoader(val, batch_size=cfg.dataset.batch_size_val, shuffle=False, collate_fn=lambda arg: collate_fn(cfg, arg))
    test = DataLoader(test, batch_size=cfg.dataset.batch_size_test, shuffle=False, collate_fn=lambda arg: collate_fn(cfg, arg))

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


@hydra.main(version_base=None, config_path='configs', config_name='main')
def run(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    setup(cfg)
    model, dataloaders = load(cfg)

    run_training(cfg, model, *dataloaders[:-1])


if __name__ == '__main__':
    run()
