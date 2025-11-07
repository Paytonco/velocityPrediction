from hydra_orm import orm
import sqlalchemy as sa
import torch


class Model(orm.InheritableTable):
    pass


class Trainable(Model):
    epoch_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=50)
    batch_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=10)
    shuffle_training_batches: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    check_val_every_n_epoch: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=10)

    learning_rate: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-3)


class First(Trainable):
    pass


class Second(Trainable):
    reorient_to_reference_orientation: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    reference_orientation_angle: float = orm.make_field(orm.ColumnRequired(sa.Double), default=torch.pi / 4)

    use_angle_input: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    predict_angle: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    predict_cos_sin: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)


class GNN(Trainable):
    pass
