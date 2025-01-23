from collections import defaultdict
from pathlib import Path
import pprint
import sys
import logging

import hydra
import omegaconf
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
import pandas as pd
import wandb

from rna_vel_pred import cs, callbacks, datasets, models, utils


log = logging.getLogger(__file__)


def loss(input, target):
    input_norm2 = input.pow(2).sum(1, keepdim=True)
    # mse_loss \in [0, 4]. If the model outputs at least one zero vector,
    # loss_zero_norm will be at least 4; otherwise, it is zero.
    # Sum reduce is used because a mean reduce could be less than 4.
    loss_zero_norm = 4*(1 - input_norm2).pow(2).sum()
    return input.shape[1] * F.mse_loss(input, target) + loss_zero_norm


class Runner(pl.LightningModule):
    def __init__(self, cfg, model, splits):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.splits = splits

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())

    def loss(self, input, target):
        return loss(input, target)

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
        out = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch)
        loss = self.loss(out, batch.poi_vel)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch)
        loss = self.loss(out, batch.poi_vel)
        return loss

    def test_step(self, batch, batch_idx):
        out = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch)
        loss = self.loss(out, batch.poi_vel)
        return loss

    def predict_step(self, split_data, batch_idx):
        split, batch = split_data
        out = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch)
        batch.poi_vel_pred = out
        batch._slice_dict['poi_vel_pred'] = batch._slice_dict['poi_vel']
        batch._inc_dict['poi_vel_pred'] = batch._inc_dict['poi_vel']
        ds = InMemoryDataset(None)
        ds.save(batch.cpu().to_data_list(), f'{self.cfg.run_dir}/pred_{split}.pt')
        measurement_id, t, pos, vel = [batch[x].cpu().numpy()
                                       for x in ['poi_measurement_id', 'poi_t', 'poi_pos', 'poi_vel_pred']]
        df = pd.DataFrame(dict(
            measurement_id=measurement_id,
            t=t,
            **{f'x{i+1}': x for i, x in enumerate(pos.T)},
            **{f'v{i+1}': v for i, v in enumerate(vel.T)}
        ))
        df = df.sort_values('measurement_id', ignore_index=True)
        df.to_csv(f'{self.cfg.run_dir}/pred_{split}.csv', index=False)
        return True


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = cs.get_engine()
    cs.create_all(engine)
    with cs.orm.Session(engine, expire_on_commit=False) as db:
        cfg = cs.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        pprint.pp(cfg)
        log.info('Command: %s', ' '.join(sys.argv))
        log.info(f'Outputs will be saved to: {cfg.run_dir}')

    torch.set_default_dtype(torch.float64)  # must put here when calling main in a loop

    splits = map(datasets.DatasetMerged, zip(*[
        datasets.get_dataset(v, rng_seed=cfg.rng_seed)
        for v in cfg.dataset
    ]))
    splits = {k: s for k, s in zip(('train', 'val', 'test'), splits)}
    splits['train'] = splits['train'].shuffle()

    logger = pl.loggers.TensorBoardLogger(cfg.run_dir, name='', version='tb_logs')
    trainer = pl.Trainer(
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs,
        logger=not cfg.trainer.logger or logger,
        precision=64,
        callbacks=[
            callbacks.PlotCB(),
            callbacks.ModelCheckpoint(
                dirpath=cfg.run_dir,
                filename='{epoch}',
                save_last='link',
                save_top_k=-1,
                save_on_train_epoch_end=False,
                enable_version_counter=False,
            )
        ],
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        deterministic=True
    )

    if isinstance(cfg.model, cs.ModelTrained):
        model = models.get_model(cfg.model.config.model, rng_seed=cfg.model.config.rng_seed)
        ckpt_path = cfg.model.config.run_dir/cfg.model.ckpt_filename
    else:
        model = models.get_model(cfg.model, rng_seed=cfg.rng_seed)
        ckpt_path = None

    runner = Runner(cfg, model, splits)

    if cfg.trainer.fit:
        trainer.fit(runner, ckpt_path=ckpt_path)
    if cfg.trainer.pred:
        trainer.predict(runner, ckpt_path=ckpt_path)
        # TODO: ??? too complicated
        df = pd.concat([pd.read_csv(f'{cfg.run_dir}/pred_{split}.csv') for split in ('train', 'val', 'test')])
        df = df.set_index('measurement_id').sort_index()
        df.to_csv(f'{cfg.run_dir}/pred.csv', index=False)


def get_run_dir(hydra_init=utils.HYDRA_INIT, commit=True):
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        last_override = None
        overrides = []
        for i, a in enumerate(sys.argv):
            if '=' in a:
                overrides.append(a)
                last_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine = cs.get_engine()
        cs.create_all(engine)
        with cs.orm.Session(engine, expire_on_commit=False) as db:
            cfg = cs.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            if commit and '-c' not in sys.argv:
                db.commit()
                cfg.run_dir.mkdir(exist_ok=True)
            return last_override, str(cfg.run_dir)


if __name__ == '__main__':
    last_override, run_dir = get_run_dir()
    run_dir_override = f'hydra.run.dir={run_dir}'
    if last_override is None:
        sys.argv.append(run_dir_override)
    else:
        sys.argv.insert(last_override + 1, run_dir_override)
    main()
