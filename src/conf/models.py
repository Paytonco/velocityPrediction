from hydra_orm import orm
import sqlalchemy as sa


class Model(orm.InheritableTable):
    pass


class Trainable(Model):
    epoch_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=100)
    batch_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=10)
    shuffle_training_batches: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    check_val_every_n_epoch: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=10)

    learning_rate: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-3)


class First(Trainable):
    pass


class GNN(Trainable):
    pass
