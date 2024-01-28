from pathlib import Path

import hydra
import omegaconf
from omegaconf import OmegaConf
import flatten_json
import lightning.pytorch as pl
import wandb

import callbacks


class Runner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg


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
    Path(cfg.run_dir).mkdir(parents=True)

    logger = pl.loggers.WandbLogger(project=cfg.wandb.project, save_dir=cfg.run_dir)
    logger.log_hyperparams(flatten_json.flatten_json(
        OmegaConf.to_container(cfg, resolve=True),
        separator='|'
    ))

    print(OmegaConf.to_yaml(cfg, resolve=True))

    cbs = [
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

    runner = Runner(cfg)

    ckpt_path = None
    if cfg.wandb.run and cfg.trainer.ckpt:
        ckpt_path = Path(cfg.out_dir)/'runs'/cfg.wandb.run/f'{cfg.trainer.ckpt}.ckpt'

    if cfg.trainer.fit:
        trainer.fit(runner, ckpt_path=ckpt_path)
    if cfg.trainer.val:
        trainer.validate(runner, ckpt_path=ckpt_path)
    if cfg.trainer.pred:
        trainer.predict(runner, ckpt_path=ckpt_path)

    wandb.finish()
    print('WandB Run ID')
    print(wrun.id)


if __name__ == '__main__':
    main()
