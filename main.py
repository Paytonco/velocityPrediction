import hydra
from omegaconf import OmegaConf
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as tg_nn
import matplotlib.pyplot as plt

import datasets


class Model(pl.LightningModule):
    def __init__(self, dim):
        super().__init__()
        self.save_hyperparameters()
        self.dim = dim
        self.model = nn.Sequential(
            nn.Linear(self.dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, t, pos, poi_t, poi_pos, batch):
        diff_t = t[batch] - poi_t[batch]
        diff_pos = pos[batch] - poi_pos[batch]
        r2 = (diff_pos**2).sum(1)
        weights = self.model(torch.stack((diff_t, r2)).T)
        return tg_nn.global_add_pool(weights * F.normalize(diff_pos, dim=1), batch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())

    def loss(self, input, target):
        return F.mse_loss(input, target)

    def training_step(self, batch, batch_idx):
        out = self(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch.batch)
        return self.loss(out, batch.poi_vel)

    def validation_step(self, batch, batch_idx):
        out = self(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch.batch)
        return self.loss(out, batch.poi_vel)

    def predict_step(self, batch, batch_idx):
        out = self(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch.batch)
        return out


def plot(path, outputs, batch):
    fig, ax = plt.subplots()
    ax.scatter(batch.poi_pos[:, 0], batch.poi_pos[:, 1],
               label='State', color='darkorange')
    ax.quiver(batch.poi_pos[:, 0], batch.poi_pos[:, 1],
              batch.poi_vel[:, 0], batch.poi_vel[:, 1], label='True', color='deepskyblue')
    ax.quiver(batch.poi_pos[:, 0], batch.poi_pos[:, 1],
              outputs[:, 0], outputs[:, 1], label='Inferred', color='blue')
    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.savefig(f'{path}.png')
    plt.close(fig)


class CallbackPlotting(pl.callbacks.Callback):
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        plot('fit', outputs.cpu(), batch.cpu())


@hydra.main(version_base=None, config_path='configs', config_name='main')
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    pl.seed_everything(0, workers=True)
    ds = getattr(datasets, cfg.dataset.name)(cfg.dataset.num_neighbors, path=cfg.dataset.path)
    model = Model(cfg.model.dim)
    dm = datasets.DataModule(ds, cfg.dataset.batch_size)
    trainer = pl.Trainer(
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs,
        logger=cfg.trainer.logger,
        callbacks=CallbackPlotting(),
        precision=64
    )
    if cfg.trainer.fit:
        trainer.fit(model, dm)
    if cfg.trainer.predict:
        trainer.predict(model, dm)


if __name__ == '__main__':
    main()
