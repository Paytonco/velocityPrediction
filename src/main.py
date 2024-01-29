from pathlib import Path

import hydra
import omegaconf
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
import wandb

import callbacks
import datasets
import models


class Runner(pl.LightningModule):
    def __init__(self, cfg, model, splits):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.splits = splits

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())

    def loss(self, input, target):
        return F.mse_loss(input, target)

    def train_dataloader(self):
        ds = self.splits['train']
        return DataLoader(ds, batch_size=self.cfg.trainer.batch_size)

    def val_dataloader(self):
        ds = self.splits['val']
        return DataLoader(ds, batch_size=self.cfg.trainer.batch_size)

    def test_dataloader(self):
        ds = self.splits['test']
        return DataLoader(ds, batch_size=self.cfg.trainer.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader([
            (s, next(iter(DataLoader(ds, batch_size=len(ds))))) for s, ds in self.splits.items()
        ], collate_fn=lambda x: x[0])

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

    def predict_step(self, split_data, batch_idx):
        split, batch = split_data
        out = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch.batch)
        batch.poi_vel_pred = out
        batch._slice_dict['poi_vel_pred'] = batch._slice_dict['poi_vel']
        batch._inc_dict['poi_vel_pred'] = batch._inc_dict['poi_vel']
        ds = InMemoryDataset(None)
        ds.save(batch.cpu().to_data_list(), f'{self.cfg.run_dir}/pred_{split}.pt')
        return True


@hydra.main(version_base=None, config_path='../configs', config_name='main')
def main(cfg):
    if cfg.trainer.fit:
        job_type = 'fit'
    elif cfg.trainer.val:
        job_type = 'val'
    elif cfg.trainer.test:
        job_type = 'test'
    elif cfg.trainer.pred:
        job_type = 'pred'
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
        cfg.run_dir = str(Path(cfg.run_dir)/wrun.id)
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
    splits = {k: s.shuffle() for k, s in zip(('train', 'val', 'test'), splits)}

    model = models.get_model(cfg.model, rng_seed=cfg.rng_seed)

    runner = Runner(cfg, model, splits)

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
