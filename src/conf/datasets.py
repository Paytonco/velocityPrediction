import enum

import omegaconf
import hydra_orm
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
    measurement_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4000)
    initial_condition_noise_epsilon: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.05)


class OscillationMotif(Dataset):
    measurement_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4000)
    initial_condition_noise_epsilon: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.05)


class BifurcationMotif(Dataset):
    measurement_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4000)
    initial_condition_noise_epsilon: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.05)


class UMapDataset(str, enum.Enum):
    BONEMARROW = 'BONEMARROW'
    DENTATE_GYRUS = 'DENTATE_GYRUS'
    FOREBRAIN = 'FOREBRAIN'
    PANCREAS = 'PANCREAS'
    PBMC68K = 'PBMC68K'


class H5adUMap(Dataset):
    dataset: UMapDataset = orm.make_field(orm.ColumnRequired(sa.Enum(UMapDataset)), default=UMapDataset.PANCREAS)
    processed_filename: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    umap_dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2)

    def __post_init__(self):
        self.dataset = UMapDataset(self.dataset)

    @property
    def h5ad_path(self):
        return f'{self.dataset}/{self.dataset}.h5ad'

    @property
    def processed_path(self):
        return f'{self.dataset}/{self.processed_filename}.parquet'


sa.event.listens_for(H5adUMap, 'before_insert')(
    hydra_orm.utils.set_attr_to_func_value(H5adUMap, H5adUMap.processed_filename.key, hydra_orm.utils.generate_random_string)
)


