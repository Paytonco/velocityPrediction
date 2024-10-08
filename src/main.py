from collections import defaultdict
from pathlib import Path

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

import callbacks
import datasets
import models
import wandbruns


def loss(input, target):
    input_norm2 = input.pow(2).sum(1, keepdim=True)
    # mse_loss \in [0, 4]. If the model outputs at least one zero vector,
    # loss_zero_norm will be at least 4; otherwise, it is zero.
    # Sum reduce is used because a mean reduce could be less than 4.
    loss_zero_norm = 4*(1 - input_norm2).pow(2).sum()
    return F.mse_loss(input, target) + loss_zero_norm


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


def cfg_normalize_dataset(cfg):
    cfg.dataset = sorted(cfg.dataset.values(), key=lambda c: c.name)
    cfg.dataset = sorted(cfg.dataset, key=lambda c: c.get('path', '\0'))


@hydra.main(version_base=None, config_path='../configs', config_name='main')
def main(cfg):
    torch.set_default_dtype(torch.float64)  # must put here when calling main in a loop
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
    wrun = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        dir=cfg.wandb.dir,
        tags=cfg.wandb.tags,
        job_type=job_type,
    )
    with omegaconf.open_dict(cfg):
        cfg.out_dir = str(Path(cfg.out_dir).resolve())
        cfg.run_dir = str(Path(cfg.run_dir)/wrun.id)
        if cfg.wandb.run is not None and len(cfg.trainer.copy_saved_cfg):
            run_saved_cfg = OmegaConf.create(wandbruns.query_runs(cfg.wandb.entity, cfg.wandb.project, {'name': cfg.wandb.run}, {}, {})[0].config)
            for k in cfg.trainer.copy_saved_cfg:
                cfg[k] = run_saved_cfg[k]
            if 'dataset' not in cfg.trainer.copy_saved_cfg:
                if cfg.get('dataset') is None:
                    raise ValueError('No datasets selected. Select a dataset with "+dataset@dataset.<name>=<dataset_cfg>".')
                cfg_normalize_dataset(cfg)
        else:
            if cfg.get('dataset') is None:
                raise ValueError('No datasets selected. Select a dataset with "+dataset@dataset.<name>=<dataset_cfg>".')
            cfg_normalize_dataset(cfg)
        dataset_summary = defaultdict(list)
        for ds in cfg.dataset:
            dataset_summary['name'].append(ds.name)
            dataset_summary['data_dir'].append(ds.get('data_dir', ''))
            dataset_summary['num_neighbors'].append(ds.num_neighbors)
            dataset_summary['sparsify_step_time'].append(ds.sparsify_step_time)
            if ds.name == 'SCVeloSaved':
                dataset_summary['umap_num_components'].append(ds.umap.n_components)
        cfg.dataset_summary = {k: ','.join(map(str, v)) for k, v in dataset_summary.items()}
        # normalize dataset list in by sorting by name, then sorting by path
        # cfg.dataset = sorted(cfg.dataset.values(), key=lambda c: c.name)
        # cfg.dataset = sorted(cfg.dataset, key=lambda c: c.get('path', '\0'))
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
        datasets.get_dataset(v, rng_seed=cfg.rng_seed)
        for v in cfg.dataset
    ]))
    splits = {k: s.shuffle() for k, s in zip(('train', 'val', 'test'), splits)}

    model = models.get_model(cfg.model, rng_seed=cfg.rng_seed)

    runner = Runner(cfg, model, splits)

    ckpt_path = None
    # if cfg.wandb.run and cfg.trainer.ckpt:
    if cfg.trainer.ckpt:
        # ckpt_path = Path(cfg.out_dir)/'runs'/cfg.wandb.run/f'{cfg.trainer.ckpt}.ckpt'
        ckpt_path = cfg.trainer.ckpt

    if cfg.trainer.fit:
        trainer.fit(runner, ckpt_path=ckpt_path)
    if cfg.trainer.val:
        trainer.validate(runner, ckpt_path=ckpt_path)
    if cfg.trainer.test:
        trainer.test(runner, ckpt_path=ckpt_path)
    if cfg.trainer.pred:
        trainer.predict(runner, ckpt_path=ckpt_path)
        df = pd.concat([pd.read_csv(f'{cfg.run_dir}/pred_{split}.csv') for split in ('train', 'val', 'test')])
        df = df.set_index('measurement_id').sort_index()
        df.to_csv(f'{cfg.run_dir}/pred.csv', index=False)

    wandb.finish()
    print('WandB Run ID')
    print(wrun.id)
    print(f'Results saved to {cfg.run_dir}')
    return wrun.id


if __name__ == '__main__':
    main()
