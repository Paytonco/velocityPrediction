from functools import partial
import inspect
import logging
import pprint
import sys

import hydra
from omegaconf import OmegaConf
import optuna
import optuna_integration
import lightning.pytorch as pl

from conf import conf
from rna_vel_pred import callbacks, datasets, models, loggers, utils, main as main_module


log = logging.getLogger(__file__)


def apply_trial_params(trial_params, cfg):
    for k, v in trial_params.items():
        if isinstance(v, dict):
            cfg_nested = getattr(cfg, k)
            if isinstance(cfg_nested, list):
                for i, vv in v.items():
                    apply_trial_params(vv, cfg_nested[i])
            else:
                apply_trial_params(v, cfg_nested)
        else:
            setattr(cfg, k, v)


def get_trial_params(cfg, trial):
    suggest_umap_dimension = trial.suggest_int('datasets[*].umap_dimension', 2, 15)
    return dict(
        datasets={
            i: dict(
                umap_dimension=suggest_umap_dimension,
                time_step_count_sparsify=trial.suggest_int(f'datasets[{i}].time_step_count_sparsify', 2, 30),
                neighbor_count=trial.suggest_int(f'datasets[{i}].neighbor_count', 2, 30),
            )
            for i in range(len(cfg.datasets))
        },
    )


def objective_trainer(cfg, trial):
    trial_params = get_trial_params(cfg, trial)

    apply_trial_params(trial_params, cfg)

    pl.seed_everything(cfg.rng_seed)

    trainer = pl.Trainer(
        logger=loggers.CSVLogger(cfg.run_dir, name=None),
        max_epochs=cfg.model.epoch_count,
        accelerator=cfg.device,
        check_val_every_n_epoch=cfg.model.check_val_every_n_epoch,
        deterministic=True,
        callbacks=[
            callbacks.LogStats(),
            optuna_integration.pytorch_lightning.PyTorchLightningPruningCallback(trial, MONITOR),
        ],
    )

    splits = datasets.get_merged_dataset(cfg, cfg.data_dir, rng_seed=cfg.rng_seed)

    model, ckpt_path = models.get_model(cfg.model, rng_seed=cfg.rng_seed)

    lightning = main_module.Lightning(cfg, splits, model)

    trainer.fit(lightning, ckpt_path=ckpt_path)

    return lightning.callback_metrics[MONITOR].item()


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = conf.get_engine()
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        cfg.is_optuna_sweep = True
        db.commit()
        log.info('Command: python %s', ' '.join(sys.argv))
        pprint.pp(cfg)
        log.info('Output directory: %s', cfg.run_dir)

    objective = partial(objective_trainer, cfg)
    study = optuna.create_study(
        storage='sqlite:///optuna.sqlite',
        direction='minimize',
        study_name=f'metric={MONITOR},{hydra.core.hydra_config.HydraConfig.get().job.override_dirname}',
        # sampler=...,
        # pruner=optuna.pruners.SuccessiveHalvingPruner(),
        pruner=optuna.pruners.ThresholdPruner(lower=-1),  # set lower to impossible value to prune only on NaN
    )
    study.set_metric_names([MONITOR])
    study.set_user_attr('trial_params', inspect.getsource(get_trial_params))
    study.set_user_attr('run_dir', str(cfg.run_dir))
    study.optimize(objective, n_trials=50)


if __name__ == '__main__':
    MONITOR = 'val_loss'
    last_override, run_dir = utils.get_run_dir(commit=False)
    utils.set_run_dir(last_override, run_dir)
    main()
