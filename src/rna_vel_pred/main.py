from collections import defaultdict
import pprint
import sys

import hydra
from omegaconf import OmegaConf
import lightning.pytorch as pl
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pytorch_lightning.utilities import CombinedLoader
from einops import reduce

import conf.conf
from rna_vel_pred import callbacks, datasets, models, loggers, utils


log = utils.getLoggerByFilename(__file__)


class Lightning(pl.LightningModule):
    def __init__(self, cfg, splits, model):
        super().__init__()
        self.cfg = cfg
        self.splits = splits
        self.model = model

    def configure_optimizers(self):
        lr = self.cfg.model.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=self.cfg.model.max_steps),
            ),
        )


    def train_dataloader(self):
        return DataLoader(self.splits['train'], batch_size=self.cfg.model.batch_size, shuffle=self.cfg.model.shuffle_training_batches)

    def val_dataloader(self):
        return DataLoader(self.splits['val'], batch_size=self.cfg.model.batch_size)

    def test_dataloader(self):
        return DataLoader(self.splits['test'], batch_size=self.cfg.model.batch_size)

    def predict_dataloader(self):
        return CombinedLoader({
                s: DataLoader(data, batch_size=self.cfg.model.batch_size)
                for s, data in self.splits.items()
            },
            mode='max_size'
        )

    def loss(self, input, target):
        if self.cfg.use_directionless_loss:
            input_r2 = reduce(input.square(), 'vel dim -> vel', 'sum')
            loss_zero_norm = (1 - input_r2).square()
            projection = reduce(
                (input * target),
                'vel dim -> vel',
                'sum',
            )
            return (1 - projection.square()).square().mean()
        else:
            input_r2 = reduce(input.square(), 'vel dim -> vel', 'sum')
            loss_zero_norm = (1 - input_r2).square()
            loss_cosine = reduce(
                0.5 * (input - target).square(),
                'vel dim -> vel',
                'sum',
            )
            return (loss_cosine + loss_zero_norm).mean()

    def training_step(self, batch, batch_idx):
        pred_vel = self.model(batch)
        loss = self.loss(pred_vel, batch.poi_vel)
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        pred_vel = self.model(batch)
        loss = self.loss(pred_vel, batch.poi_vel)
        return dict(loss=loss)

    def test_step(self, batch, batch_idx):
        pred_vel = self.model(batch)
        loss = self.loss(pred_vel, batch.poi_vel)
        return dict(loss=loss)

    def predict_step(self, batches, _):
        pred = {}
        for split, batch in batches[0].items():
            if batch is not None:
                poi_vel_pred = self.model(batch)
                pred[split] = Data(
                    poi_t=batch.poi_t,
                    poi_pos=batch.poi_pos,
                    poi_vel=batch.poi_vel,
                    poi_vel_pred=poi_vel_pred,
                    poi_measurement_id=batch.poi_measurement_id,
                )
        return pred


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    with conf.conf.Session() as db:
        cfg = conf.conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        log.info('Command: python %s', ' '.join(sys.argv))
        log.info(pprint.pformat(cfg))
        log.info('Output directory: %s', cfg.run_dir)

    pl.seed_everything(cfg.rng_seed)

    trainer = pl.Trainer(
        logger=loggers.CSVLogger(cfg.run_dir, name=None),
        max_steps=cfg.model.max_steps,
        accelerator=cfg.device,
        check_val_every_n_epoch=None,
        val_check_interval=cfg.model.val_check_interval,
        deterministic=True,
        callbacks=[
            callbacks.LogStats(),
            callbacks.ModelCheckpoint(
                dirpath=cfg.run_dir,
                filename='{step}',
                save_last='link',
                monitor='val_loss',
                save_top_k=2,
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
        pred_splits_list = trainer.predict(lightning, ckpt_path=ckpt_path)
        pred_splits = defaultdict(list)
        for splits in pred_splits_list:
            for split, data in splits.items():
                pred_splits[split].append(data)
        for split, data_list in pred_splits.items():
            pred_splits[split] = next(iter(DataLoader(data_list, batch_size=len(data_list))))
        dfs = []
        for split, data in pred_splits.items():
            df = pd.DataFrame(dict(
                measurement_id=data.poi_measurement_id,
                split=split,
                t=data.poi_t,
                **{f'x{i+1}': poi_pos_dim for i, poi_pos_dim in enumerate(data.poi_pos.T)},
                **{f'v{i+1}': poi_vel_dim for i, poi_vel_dim in enumerate(data.poi_vel.T)},
                **{f'v{i+1}_pred': poi_vel_pred_dim for i, poi_vel_pred_dim in enumerate(data.poi_vel_pred.T)},
            ))
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(cfg.run_dir/cfg.prediction_filename)


if __name__ == '__main__':
    last_override, run_dir = utils.get_run_dir()
    utils.set_run_dir(last_override, run_dir)
    main()
