from django.db import models
from django_experiment_tracker import models as tracker_models


class SavedDatasetFile(tracker_models.SharedFile):
    def __str__(self):
        parameter_group_names = tracker_models.ParameterGroup.objects.filter(saveddatasetfileparameter__in=self.saveddatasetfileparameter_set.all()).values_list("parameter_group_name", flat=True).distinct()
        return f'{self.alt_id} ({", ".join(parameter_group_names)})'


class SavedDatasetFileParameter(tracker_models.ParameterValue):
    saved_dataset_file = models.ForeignKey(SavedDatasetFile, on_delete=models.CASCADE)

    class Meta:
        constraints = tracker_models.ParameterValue.get_constraints('saved_dataset_file')


class Experiment(tracker_models.Experiment):
    def __str__(self):
        parameter_group_names = tracker_models.ParameterGroup.objects.filter(experimentparameter__in=self.experimentparameter_set.all()).values_list("parameter_group_name", flat=True).distinct()
        return f'{self.alt_id} ({", ".join(parameter_group_names)})'


class ExperimentParameter(tracker_models.ParameterValue):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)

    class Meta:
        constraints = tracker_models.ParameterValue.get_constraints('experiment')
