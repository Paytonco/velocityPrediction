from dataclasses import field
from pathlib import Path
from typing import Any, List

import omegaconf
import hydra_orm
import hydra_orm.utils
from hydra_orm import orm
import sqlalchemy as sa

import conf.datasets
import conf.models
from rna_vel_pred import utils


def get_engine(dir=str(utils.DIR_ROOT), name='runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')


engine = get_engine()
orm.create_all(engine)
Session = sa.orm.sessionmaker(engine)


class Conf(orm.Table):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(model=omegaconf.MISSING),
        '_self_',
    ])
    root_dir: str = field(default=str(utils.DIR_ROOT.resolve()))
    out_dir: str = field(default=str((utils.DIR_ROOT/'..'/'..'/'out'/'rna_vel_pred').resolve()))
    data_subdir: Path = field(default=str((utils.DIR_ROOT/'..'/'..'/'out'/'rna_vel_pred'/'data_redesign').resolve()))
    run_subdir: str = field(default='runs')
    prediction_filename: str = field(default='prediction.parquet')
    device: str = field(default='cuda')

    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2376999025)
    fit: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    predict: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    is_optuna_sweep: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    datasets = orm.ManyToManyField(conf.datasets.Dataset, default_factory=list, enforce_element_type=False)

    model = orm.OneToManyField(conf.models.Model, default=omegaconf.MISSING)

    @property
    def run_dir(self):
        return Path(self.out_dir)/self.run_subdir/self.alt_id

    @property
    def data_dir(self):
        return Path(self.out_dir)/self.data_subdir


class Trained(conf.models.Trainable):
    conf = orm.OneToManyField(Conf, default=omegaconf.MISSING, enforce_element_type=False)
    ckpt_filename: str = orm.make_field(orm.ColumnRequired(sa.String(len('epoch_####.ckpt'))), default='last.ckpt')

    @staticmethod
    def transform_conf(session, conf_alt_id):
        if conf_alt_id == omegaconf.MISSING:
            raise ValueError('Please set a conf alt_id with model.conf=<conf_alt_id>.')
        conf = (
            sa.select(Conf)
            .where(Conf.alt_id == conf_alt_id)
        )
        conf = session.execute(conf)
        conf = list(zip(range(2), conf))
        assert len(conf) == 1
        conf = conf[0][1][0]
        return conf




sa.event.listens_for(Conf, 'before_insert')(
    hydra_orm.utils.set_attr_to_func_value(Conf, Conf.alt_id.key, hydra_orm.utils.generate_random_string)
)


orm.store_config(Conf)
orm.store_config(conf.datasets.SimpleMotif)
orm.store_config(conf.datasets.OscillationMotif)
orm.store_config(conf.datasets.BifurcationMotif)
orm.store_config(conf.datasets.H5adUMap)
orm.store_config(conf.models.First, group=Conf.model.key)
orm.store_config(conf.models.Second, group=Conf.model.key)
orm.store_config(conf.models.GNN, group=Conf.model.key)
orm.store_config(Trained, group=Conf.model.key)
