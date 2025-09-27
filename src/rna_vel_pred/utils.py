import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F


DIR_SRC = Path(__file__).parent
DIR_ROOT = (DIR_SRC/'..'/'..').resolve()
DIR_DATA = DIR_ROOT/'data'

HYDRA_INIT = dict(version_base=None, config_path='../../conf', config_name='conf')


def get_run_dir(hydra_init=HYDRA_INIT, commit=True, engine_name='runs'):
    from conf import conf

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
        engine = conf.get_engine(name=engine_name)
        conf.orm.create_all(engine)
        with conf.sa.orm.Session(engine, expire_on_commit=False) as db:
            cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            # if commit and '-c' not in sys.argv:
            if commit:
                db.commit()
                cfg.run_dir.mkdir(exist_ok=True)
            return last_override, str(cfg.run_dir)


def set_run_dir(last_override, run_dir):
    run_dir_override = f'hydra.run.dir={run_dir}'
    if last_override is None:
        sys.argv.append(run_dir_override)
    else:
        sys.argv.insert(last_override + 1, run_dir_override)


def normalize(input, dim=1, eps=1e-7):
    return F.normalize(input, dim=dim, eps=eps)


def mv(mat, vec):
    return (vec[:, None] @ mat.mT).squeeze(1)


def vcos(vec1, vec2):
    return (vec1 * vec2).sum(1)


def vsin(vec1, vec2):
    return vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]


def vangle(vec1, vec2, cos=None, sin=None):
    if cos is None:
        cos = vcos(vec1, vec2)
    if sin is None:
        sin = vsin(vec1, vec2)
    return sin.sign() * cos.acos()
