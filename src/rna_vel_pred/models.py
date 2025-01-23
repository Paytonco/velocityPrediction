import torch
import torch.nn as nn
import torch_geometric as tg
import lightning.pytorch as pl

from rna_vel_pred import utils


class First(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weighter = nn.Sequential()
        dims = [2, *([cfg.hidden.dim] * cfg.hidden.layers), 1]
        for layer, (d_in, d_out) in enumerate(zip(dims, dims[1:])):
            self.weighter.append(nn.Linear(d_in, d_out, bias=cfg.bias))
            if layer < len(dims) - 2:
                self.weighter.append(getattr(nn, cfg.activation)())

        # self.weighter = nn.Sequential(
        #     nn.Linear(2, 10),
        #     nn.ReLU(),
        #     nn.Linear(10, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 10),
        #     nn.ReLU(),
        #     nn.Linear(10, 5),
        #     nn.ReLU(),
        #     nn.Linear(5, 1),
        # )

    def forward(self, t, pos, poi_t, poi_pos, batch):
        batch = batch.batch
        diff_t = torch.sign(t - poi_t[batch])
        diff_pos = pos - poi_pos[batch]
        r2 = diff_pos.pow(2).sum(1)
        weights = self.weighter(torch.stack((diff_t, r2), dim=1))
        return utils.normalize(
            tg.nn.global_add_pool(weights * utils.normalize(diff_pos), batch)
        )


class FirstDistance(First):
    def forward(self, t, pos, poi_t, poi_pos, batch):
        batch = batch.batch
        diff_t = torch.sign(t - poi_t[batch])
        diff_pos = pos - poi_pos[batch]
        r = diff_pos.pow(2).sum(1).sqrt()
        weights = self.weighter(torch.stack((diff_t, r), dim=1))
        return utils.normalize(
            tg.nn.global_add_pool(weights * utils.normalize(diff_pos), batch)
        )


class Second(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.norm_activation = getattr(nn, cfg.activation)()

        dims = [2, *([cfg.hidden.dim] * cfg.hidden.layers), 2]
        self.conv_layers = nn.ModuleList()
        for idx, (c_in, c_out) in enumerate(zip(dims, dims[1:])):
            step = 10 if idx == 0 else 1
            self.conv_layers.append(nn.Conv2d(
                    in_channels=c_in // 2, out_channels=c_out // 2,
                    kernel_size=(step, 1), stride=(step, 1),
                    bias=False
            ))

    def forward(self, t, pos, poi_t, poi_pos, batch):
        # diff_t = torch.sign(t - poi_t[batch.batch]) * self.weighter_t
        diff_pos = pos - poi_pos[batch.batch]
        diff_pos = diff_pos  # + diff_pos * diff_t[:, None]
        diff_pos = diff_pos.reshape(batch.batch_size, -1, 1, 2)
        x = diff_pos.permute(0, 3, 1, 2)  # now (B, F, K, N)
        x_even, x_odd = x[:, ::2], x[:, 1::2]
        x_double_batch = torch.cat((x_even, x_odd))
        for idx, conv in enumerate(self.conv_layers):
            x_double_batch = conv(x_double_batch)
            if idx < len(self.conv_layers) - 1:
                x_double_batch = self.equivariant_activation(x_double_batch, None, batch.batch_size)

        x = x_double_batch.view(2, batch.batch_size, 1).permute(1, 2, 0).squeeze()

        return utils.normalize(x)

    def equivariant_activation(self, x_double_batch, diff_t, batch_size):
        r = (x_double_batch.view(2, batch_size, -1, 1).pow(2).sum(0, keepdim=True) + 1e-5)
        r = self.norm_activation(r)

        return (x_double_batch.view(2, batch_size, -1, 1) * r).view(2 * batch_size, -1, 1, 1)


def get_model(cfg, rng_seed=0):
    with pl.utilities.seed.isolate_rng():
        pl.seed_everything(rng_seed, workers=True)
        if cfg.name == 'First':
            return First(cfg)
        elif cfg.name == 'FirstDistance':
            return FirstDistance(cfg)
        elif cfg.name == 'Second':
            return Second(cfg)
        else:
            raise ValueError(f'Unknown model: {cfg.name}')
