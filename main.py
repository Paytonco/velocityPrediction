import pathlib
import hydra
import omegaconf
from omegaconf import OmegaConf
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as tg_nn
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

import datasets


class MLP(nn.Module):
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
        diff_t = t - poi_t[batch]
        diff_pos = pos - poi_pos[batch]
        r2 = (diff_pos**2).sum(1)
        weights = self.model(torch.stack((diff_t, r2), dim=1))
        return tg_nn.global_add_pool(weights * F.normalize(diff_pos, dim=1), batch)


class MLPCentroidDot(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 10),
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
        diff_t = t - poi_t[batch]
        diff_pos = pos - poi_pos[batch]
        r2 = (diff_pos**2).sum(1)
        centroid = tg_nn.global_add_pool(pos, batch)
        diff_centroid = (centroid - poi_pos)[batch]
        cosine = (diff_pos * diff_centroid).sum(1) / (r2 * (diff_centroid**2).sum(1)).sqrt()
        weights = self.model(torch.stack((diff_t, r2, cosine), dim=1))
        return tg_nn.global_add_pool(weights * F.normalize(diff_pos, dim=1), batch)


class Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())

    def loss(self, input, target):
        r = ((target**2).sum(1) + 1e-5).sqrt()
        return F.mse_loss(input / r, target / r)

    def forward(self, t, pos, poi_t, poi_pos, batch):
        return self.model(t, pos, poi_t, poi_pos, batch)

    def training_step(self, batch, batch_idx):
        out = self(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch.batch)
        loss = self.loss(out, batch.poi_vel)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch.batch)
        loss = self.loss(out, batch.poi_vel)
        return dict(loss=loss, out=out)

    def predict_step(self, batch, batch_idx):
        out = self(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch.batch)
        return out


def plot_true_vs_inferred(out, data, neighbor_lines=True):
    fig, ax = plt.subplots()
    if neighbor_lines:
        poi_pos = data.poi_pos[data.batch]
        diff_pos = F.normalize(data.pos - poi_pos, dim=1)
        ax.quiver(poi_pos[:, 0], poi_pos[:, 1], diff_pos[:, 0], diff_pos[:, 1],
                  color='red', headwidth=1)
        # lc = LineCollection(torch.stack((poi_pos, poi_pos + diff_pos), dim=1), linestyles='-', colors='r')
        # ax.add_collection(lc)
    ax.scatter(data.poi_pos[:, 0], data.poi_pos[:, 1], label='State', c='orange')
    cm = LinearSegmentedColormap.from_list('true_inferred', ['deepskyblue', 'blue'])
    pos = data.poi_pos.repeat(2, 1)
    vel = torch.cat((data.poi_vel, out))
    color_vals = torch.tensor([0., 1.]).repeat_interleave(data.poi_pos.size(0))
    ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], color=cm(color_vals))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return fig


class PlotsCB(pl.callbacks.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log('train_loss', outputs['loss'], prog_bar=True, batch_size=batch.t.numel())

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log('val_loss', outputs['loss'], prog_bar=True, batch_size=batch.t.numel())

    def on_validation_end(self, trainer, pl_module):
        splits = ('train', 'val')
        dataloaders = map(lambda s: trainer.datamodule.predict_dataloader(s, 40), splits)
        tensorboard = pl_module.logger.experiment
        for s, dl in zip(splits, dataloaders):
            batch = next(iter(dl)).to(pl_module.device)
            out = pl_module.forward(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch.batch)
            fig = plot_true_vs_inferred(out.cpu(), batch.cpu())
            fig.savefig(f'{trainer.log_dir}/{s}_true_vs_inferred.pdf', format='pdf')
            plt.close(fig)

    def on_predict_end(self, trainer, pl_module):
        self.on_validation_end(trainer, pl_module)


@hydra.main(version_base=None, config_path='configs', config_name='main')
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    pl.seed_everything(0, workers=True)
    if cfg.dataset.neighbors == 'knn':
        ds_transform = datasets.KNNGraph('t', cfg.dataset.num_neighbors)
    elif cfg.dataset.neighbors == 'slidingwindow':
        if cfg.dataset.num_prev_neighbors is None:
            with omegaconf.open_dict(cfg):
                cfg.dataset.num_prev_neighbors = cfg.dataset.num_neighbors // 2
        ds_transform = datasets.SlidingWindowGraph(cfg.dataset.num_prev_neighbors, cfg.dataset.num_neighbors + 1)
    elif cfg.dataset.neighbors == 'radius':
        ds_transform = datasets.RadiusGraph('t', cfg.dataset.neighbor_radius, max_num_neighbors=cfg.dataset.num_neighbors)
    else:
        raise ValueError(f'Invalid method for finding neighbors: {cfg.dataset.neighbors}')
    ds = getattr(datasets, cfg.dataset.name)(ds_transform, cfg.dataset.sparsity_step, path=cfg.dataset.path)
    model = Model(globals()[cfg.model]())
    dm = datasets.DataModule(ds, cfg.dataset.batch_size)
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor='val_loss')
    logger_version = None
    if cfg.trainer.pred_ckpt:
        logger_version = int(pathlib.Path(cfg.trainer.pred_ckpt).parent.parent.stem.split('_')[1])
    logger = pl.loggers.TensorBoardLogger(f'{cfg.trainer.default_root_dir}', name=cfg.dataset.name, version=logger_version)
    trainer = pl.Trainer(
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        default_root_dir=cfg.trainer.default_root_dir,
        max_epochs=cfg.trainer.max_epochs,
        logger=not cfg.trainer.logger or logger,
        callbacks=[ckpt_cb, PlotsCB()],
        precision=64,
    )
    if cfg.trainer.fit:
        trainer.fit(model, dm, ckpt_path=cfg.trainer.fit_ckpt)
    if cfg.trainer.predict:
        trainer.predict(model, dm, ckpt_path=cfg.trainer.pred_ckpt)


if __name__ == '__main__':
    main()
