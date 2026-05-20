from django.db import models
from django_experiment_tracker import models as tracker_models


class Experiment(tracker_models.Experiment):
    pass

    def __str__(self):
        parameter_group_names = tracker_models.ParameterGroup.objects.filter(experimentparameter__in=self.experimentparameter_set.all()).values_list("parameter_group_name", flat=True).distinct()
        return f'{self.alt_id} ({", ".join(parameter_group_names)})'


class ExperimentParameter(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    parameter_group = models.ForeignKey(tracker_models.ParameterGroup, on_delete=models.CASCADE)
    parameter = models.ForeignKey(tracker_models.Parameter, on_delete=models.CASCADE)
    parameter_value = models.CharField(max_length=100)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['experiment', 'parameter', 'parameter_group'], name='experiment_parameter_and_group_alt_key'),
        ]

    def __str__(self):
        return f'{self.parameter_group}, {self.parameter}, {self.parameter_value}'
