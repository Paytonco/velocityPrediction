import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import lightning.pytorch as pl


class First(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )

    def forward(self, t, pos, poi_t, poi_pos, batch):
        diff_t = torch.sign(t - poi_t[batch])
        diff_pos = pos - poi_pos[batch]
        r2 = diff_pos.pow(2).sum(1)
        weights = self.model(torch.stack((diff_t, r2), dim=1))
        return tg.nn.global_add_pool(weights * F.normalize(diff_pos, dim=1), batch)


def get_model(cfg, rng_seed=0):
    with pl.utilities.seed.isolate_rng():
        pl.seed_everything(rng_seed, workers=True)
        if cfg.name == 'First':
            return First()
        else:
            raise ValueError(f'Unknown model: {cfg.name}')
