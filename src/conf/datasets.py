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

    reverse_velocities: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    scale_t_by_std_ratio: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)


class SimpleMotif(Dataset):
    measurement_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4000)
    initial_condition_noise_epsilon: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.05)


class OscillationMotif(Dataset):
    measurement_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4000)
    initial_condition_noise_epsilon: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.05)


class BifurcationMotif(Dataset):
    measurement_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4000)
    initial_condition_noise_epsilon: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.05)


class DetransitionMotif(Dataset):
    measurement_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4000)
    initial_condition_noise_epsilon: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.05)
    dt: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.001)
    sigma: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.5)
    T: float = orm.make_field(orm.ColumnRequired(sa.Double), default=5.0)


class UMapDataset(str, enum.Enum):
    BONEMARROW = 'BONEMARROW'
    DENTATE_GYRUS = 'DENTATE_GYRUS'
    FOREBRAIN = 'FOREBRAIN'
    PANCREAS = 'PANCREAS'
    PBMC68K = 'PBMC68K'


class H5adUMap(Dataset):
    dataset: UMapDataset = orm.make_field(orm.ColumnRequired(sa.Enum(UMapDataset)), default=UMapDataset.PANCREAS)
    processed_filename: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True), init=False, omegaconf_ignore=True)
    umap_dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2)

    def __post_init__(self):
        self.dataset = UMapDataset(self.dataset)

    @property
    def h5ad_path(self):
        return f'{self.dataset}/{self.dataset}.h5ad'

    @property
    def processed_path(self):
        return f'{self.dataset}/{self.processed_filename}.parquet'


@sa.event.listens_for(H5adUMap, 'before_insert')
def set_processed_filename(mapper, connection, target):
    processed_umap_dataset = connection.execute(
        sa.select(H5adUMap.processed_filename)
        .where(H5adUMap.dataset == target.dataset)
        .where(H5adUMap.umap_dimension == target.umap_dimension)
        .distinct()
    )
    processed_umap_dataset = list(zip(range(2), processed_umap_dataset))
    assert len(processed_umap_dataset) <= 1
    if len(processed_umap_dataset) == 1:
        target.processed_filename = processed_umap_dataset[0][1][0]
    else:
        hydra_orm.utils.set_attr_to_func_value(
            H5adUMap,
            H5adUMap.processed_filename.key,
            hydra_orm.utils.generate_random_string,
        )(mapper, connection, target)
