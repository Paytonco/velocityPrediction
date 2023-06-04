# come up with a vector field v(t,x) = [1, x]
# instantiate a bunch of points at random (uniform random on [-5,5] x [-5,5])
import time
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class VectorData(Dataset):
    def __init__(self, dim, size):
        """
        Parameters
        ----------
        dim: int
            The spatial dimensions of the position and velocity vectors.

        num_samples: int
            The number of vectors to generate.
        """
        super().__init__()
        self.dim = dim
        self.size = size

        self.positions = self.generate_positions()
        self.velocities = self.generate_velocities()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        datum, target = self.positions[idx], self.velocities[idx]
        return datum, target

    def generate_positions(self):
        return 10 * (torch.rand(self.size, self.dim) - 0.5)

    def generate_velocities(self):
        velocities = self.positions.clone()
        velocities[:, 0] = 1.
        return velocities

    def get_neighbors(self):
        pass


class VelVector(nn.Module):
    def __init__(self, vector_dims, num_neighbors):
        super().__init__()

        self.vector_dims = vector_dims
        self.num_neighbors = num_neighbors

        self.model = nn.Sequential(
            nn.Linear(
                # 1 +  # time
                vector_dims,
                # vector_dims * num_neighbors,  # x_i - x_0 for i \in {-n,...,n} \ {0}
                25,
            ),
            nn.Linear(
                # 1 +  # time
                25,
                # vector_dims * num_neighbors,  # x_i - x_0 for i \in {-n,...,n} \ {0}
                vector_dims,
            )
        )

    def norm_diffs(self, batch):
        x0, neighbors = batch[0], batch[1:]
        diffs = neighbors - x0
        return diffs / (diffs**2).sum(dim=-1, keepdim=True)**(1/2)

    def forward(self, batch):
        norm_diffs = self.norm_diffs(batch)
        vel = (norm_diffs * self.model(norm_diffs)).sum(dim=0)
        return vel / ((vel**2).sum(dim=-1) + 1e-10)**(1/2)


def calc_loss(vel_given, vel_predicted):
    vel_given = vel_given / torch.sqrt((vel_given**2).sum(dim=-1) + 1e-10)
    dist = torch.sum((vel_given - vel_predicted)**2)
    dot_product = torch.sum(vel_given * vel_predicted, dim=-1)
    angle = torch.sum(1 / torch.sqrt(dot_product.abs() + 1e-10))

    # print(f'{dist=}, {angle=}, {dot_product=}')
    loss = dist  # + angle

    return loss


def get_model(dim, num_neighbors):
    model = VelVector(dim, num_neighbors)

    return model


def setup(cfg):
    # Reproduciblity
    np.random.seed(cfg.setup.rng_seed)
    torch.manual_seed(cfg.setup.rng_seed)
    torch.backends.cudnn.deterministic = True


def load(cfg):
    dataset = VectorData(cfg.dataset.dim, cfg.dataset.size)

    train = torch.stack(dataset[:cfg.dataset.size_train])
    val = torch.stack(dataset[-(cfg.dataset.size_val+cfg.dataset.size_test):len(dataset)-cfg.dataset.size_test])
    test = torch.stack(dataset[-cfg.dataset.size_test:])

    train = DataLoader(train, batch_size=cfg.dataset.batch_size_train, shuffle=False)
    val = DataLoader(val, batch_size=cfg.dataset.batch_size_val, shuffle=False)
    test = DataLoader(test, batch_size=cfg.dataset.batch_size_test, shuffle=False)

    model = get_model(cfg.dataset.dim, cfg.model.num_neighbors)

    return model, (train, val, test)


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    positions, vels = data

    output = model(positions)

    loss = calc_loss(vels[0], output)

    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def val(model, data):
    model.eval()

    positions, vels = data

    output = model(positions)

    loss = calc_loss(vels[0], output)

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
        for data in dataloader_train:
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
