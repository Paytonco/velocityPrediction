from hydra_orm import orm
import sqlalchemy as sa
import torch


class Model(orm.InheritableTable):
    pass


class Trainable(Model):
    max_steps: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=18_250)
    val_check_interval: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=50)

    batch_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=64)
    shuffle_training_batches: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    learning_rate: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-1)


class First(Trainable):
    pass


class Second(Trainable):
    reorient_to_reference_orientation: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    reference_orientation_angle: float = orm.make_field(orm.ColumnRequired(sa.Double), default=torch.pi / 4)
    direct_vel_toward_forward: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    use_angle_input: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    predict_angle: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    predict_cos_sin: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    dropout_probability: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)


class GNN(Trainable):
    pass
