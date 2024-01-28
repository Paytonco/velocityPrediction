from pathlib import Path

import hydra
import omegaconf
from omegaconf import OmegaConf
import flatten_json
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
import wandb

import callbacks
import datasets
import models


class Runner(pl.LightningModule):
    def __init__(self, cfg, model, dataloaders):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())

    def loss(self, input, target):
        return F.mse_loss(input, target)

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['val']

    def test_dataloader(self):
        return self.dataloaders['test']

    def training_step(self, batch, batch_idx):
        out = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch.batch)
        loss = self.loss(out, batch.poi_vel)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch.batch)
        loss = self.loss(out, batch.poi_vel)
        return loss

    def test_step(self, batch, batch_idx):
        out = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch.batch)
        loss = self.loss(out, batch.poi_vel)
        return loss

    def predict_step(self, batch, batch_idx):
        out = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch.batch)
        loss = self.loss(out, batch.poi_vel)
        return loss


@hydra.main(version_base=None, config_path='../configs', config_name='main')
def main(cfg):
    if cfg.trainer.fit:
        job_type = 'fit'
    elif cfg.trainer.val:
        job_type = 'val'
    elif cfg.trainer.test:
        job_type = 'test'
    else:
        raise ValueError('Trainer will not fit, val or test.')
    if cfg.get('dataset') is None:
        raise ValueError('No datasets selected. Select a dataset with "+dataset@dataset.<name>=<dataset_cfg>".')
    wrun = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        dir=cfg.wandb.dir,
        job_type=job_type
    )
    with omegaconf.open_dict(cfg):
        cfg.out_dir = str(Path(cfg.out_dir).resolve())
        cfg.run_dir = str(Path(cfg.out_dir)/'runs'/wrun.id)
        # normalize dataset list in by sorting by name, then sorting by path
        cfg.dataset = sorted(cfg.dataset.values(), key=lambda c: c.name)
        cfg.dataset = sorted(cfg.dataset, key=lambda c: c.get('path', '\0'))
    Path(cfg.run_dir).mkdir(parents=True)

    logger = pl.loggers.WandbLogger(project=cfg.wandb.project, save_dir=cfg.run_dir)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    print(OmegaConf.to_yaml(cfg, resolve=True))

    cbs = [
        callbacks.PlotCB(),
        callbacks.ModelCheckpoint(
            dirpath=cfg.run_dir,
            filename='{epoch}',
            save_last=True,
            save_top_k=-1,
            save_on_train_epoch_end=False,
        )
    ]

    trainer = pl.Trainer(
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs,
        logger=not cfg.trainer.logger or logger,
        precision=64,
        callbacks=cbs,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        deterministic=True
    )

    splits = map(datasets.DatasetMerged, zip(*[
        datasets.get_dataset(v, cfg.data_dir, rng_seed=cfg.rng_seed)
        for v in cfg.dataset
    ]))

    dataloaders = {k: DataLoader(ds.shuffle(), batch_size=cfg.trainer.batch_size)
                   for k, ds in zip(('train', 'val', 'test'), splits)}

    model = models.get_model(cfg.model, rng_seed=cfg.rng_seed)

    runner = Runner(cfg, model, dataloaders)

    ckpt_path = None
    if cfg.wandb.run and cfg.trainer.ckpt:
        ckpt_path = Path(cfg.out_dir)/'runs'/cfg.wandb.run/f'{cfg.trainer.ckpt}.ckpt'

    if cfg.trainer.fit:
        trainer.fit(runner, ckpt_path=ckpt_path)
    if cfg.trainer.val:
        trainer.validate(runner, ckpt_path=ckpt_path)
    if cfg.trainer.test:
        trainer.test(runner, ckpt_path=ckpt_path)
    if cfg.trainer.pred:
        trainer.predict(runner, ckpt_path=ckpt_path)

    wandb.finish()
    print('WandB Run ID')
    print(wrun.id)


if __name__ == '__main__':
    main()
