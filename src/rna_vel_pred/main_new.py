import logging
import pprint
import sys

import hydra
from omegaconf import OmegaConf
import lightning.pytorch as pl
import torch
from torch_geometric.loader import DataLoader
from pytorch_lightning.utilities import CombinedLoader
from einops import reduce

from conf import conf
from rna_vel_pred import callbacks, datasets, models, loggers, utils


log = logging.getLogger(__file__)


class Lightning(pl.LightningModule):
    def __init__(self, cfg, splits, model):
        super().__init__()
        self.cfg = cfg
        self.splits = splits
        self.model = model

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.model.learning_rate)

    def train_dataloader(self):
        return DataLoader(self.splits['train'], batch_size=self.cfg.model.batch_size, shuffle=self.cfg.model.shuffle_training_batches)

    def val_dataloader(self):
        return DataLoader(self.splits['val'], batch_size=self.cfg.model.batch_size)

    def test_dataloader(self):
        return DataLoader(self.splits['test'], batch_size=self.cfg.model.batch_size)

    def predict_dataloader(self):
        return CombinedLoader(self.splits, mode='sequential')

    def loss(self, input, target):
        input_r2 = reduce(input**2, 'vel dim -> vel 1', 'sum')
        loss_zero_norm = 2*(1 - input_r2).pow(2).sum()
        loss_cosine = reduce(
            0.5 * (input - target)**2,
            'vel dim ->',
            'sum',
        )
        return loss_cosine + loss_zero_norm

    def training_step(self, batch, batch_idx):
        pred_vel = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch)
        loss = self.loss(pred_vel, batch.poi_vel)
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        pred_vel = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch)
        loss = self.loss(pred_vel, batch.poi_vel)
        return dict(loss=loss)

    def test_step(self, batch, batch_idx):
        pred_vel = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch)
        loss = self.loss(pred_vel, batch.poi_vel)
        return dict(loss=loss)

    def predict_step(self, batch, batch_idx):
        breakpoint()
        pred_vel = self.model(batch.t, batch.pos, batch.poi_t, batch.poi_pos, batch)
        loss = self.loss(pred_vel, batch.poi_vel)
        return dict(loss=loss)


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = conf.get_engine()
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        log.info('Command: python %s', ' '.join(sys.argv))
        pprint.pp(cfg)
        log.info('Output directory: %s', cfg.run_dir)

    pl.seed_everything(cfg.rng_seed)

    trainer = pl.Trainer(
        logger=loggers.CSVLogger(cfg.run_dir, name=None),
        max_epochs=cfg.model.epoch_count,
        accelerator=cfg.device,
        check_val_every_n_epoch=cfg.model.check_val_every_n_epoch,
        deterministic=True,
        callbacks=[
            callbacks.LogStats(),
            callbacks.ModelCheckpoint(
                dirpath=cfg.run_dir,
                filename='{epoch}',
                save_last='link',
                save_top_k=-1,
                save_on_train_epoch_end=False,
                enable_version_counter=False,
            ),
        ],
    )

    splits = datasets.get_merged_dataset(cfg, cfg.data_dir, rng_seed=cfg.rng_seed)

    model, ckpt_path = models.get_model(cfg.model, rng_seed=cfg.rng_seed)

    lightning = Lightning(cfg, splits, model)

    if cfg.fit:
        trainer.fit(lightning, ckpt_path=ckpt_path)
    if cfg.predict:
        trainer.predict(lightning, ckpt_path=ckpt_path)


def get_run_dir(hydra_init=utils.HYDRA_INIT, commit=True):
    if '-m' in sys.argv or '--multirun' in sys.argv:
        raise ValueError("The flags '-m' and '--multirun' are not supported. Use GNU parallel instead.")
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        last_override = None
        overrides = []
        for i, a in enumerate(sys.argv):
            if '=' in a:
                overrides.append(a)
                last_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine = conf.get_engine()
        conf.orm.create_all(engine)
        with conf.sa.orm.Session(engine, expire_on_commit=False) as db:
            cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
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
