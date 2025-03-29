import enum

import omegaconf
from hydra_orm import orm
import sqlalchemy as sa


class Dataset(orm.InheritableTable):
    time_step_count_sparsify: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=10)
    neighbor_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=10)

    frac_train: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.7)
    frac_val: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.2)
    frac_test: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.1)
    batch_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=10)

    limit_batch_count_train: bool = orm.make_field(orm.ColumnRequired(sa.Integer), default=False)
    batch_count_train: bool = orm.make_field(orm.ColumnRequired(sa.Integer), default=10)

    reverse_velocities: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)


class SimpleMotif(Dataset):
    data_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4000)
    epsilon: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.05)


class OscillationMotif(Dataset):
    data_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4000)
    epsilon: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.05)


class BifurcationMotif(Dataset):
    data_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4000)
    epsilon: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.05)


class UMapDataset(enum.Enum):
    BONEMARROW = enum.auto()
    DENTATE_GYRUS = enum.auto()
    FOREBRAIN = enum.auto()
    PANCREAS = enum.auto()
    PBMC68K = enum.auto()


class ForUMap(Dataset):
    dataset: UMapDataset = orm.make_field(orm.ColumnRequired(sa.Enum(UMapDataset)), default=omegaconf.MISSING)

    filename_h5ad: str = orm.make_field(orm.ColumnRequired(sa.String(40)), default=omegaconf.MISSING)
    filename_processed: str = orm.make_field(orm.ColumnRequired(sa.String(40)), default=omegaconf.MISSING)

    umap_dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2)
