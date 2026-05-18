from django.db import models
from django_experiment_tracker import models as tracker_models


class Experiment(tracker_models.Experiment):
    pass


class ExperimentParameter(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.PROTECT)
    parameter = models.ForeignKey(tracker_models.Parameter, on_delete=models.PROTECT)
    pk = models.CompositePrimaryKey('experiment', 'parameter')
    parameter_value = models.CharField(max_length=100)
