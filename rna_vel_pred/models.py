from django.contrib import admin
from django.db import models
from django_experiment_tracker import models as tracker_models


@admin.register(tracker_models.GitCommit)
class GitCommitAdmin(admin.ModelAdmin):
    pass


# @admin.register(tracker_models.ParameterEnumValue)
# class ParameterEnumValueAdmin(admin.ModelAdmin):
#     pass

class ParameterEnumValueInline(admin.TabularInline):
    model = tracker_models.ParameterEnumValue
    min_num = 1
    extra = 0


@admin.register(tracker_models.ParameterEnum)
class ParameterEnumAdmin(admin.ModelAdmin):
    inlines = [
        ParameterEnumValueInline,
    ]


class ParameterGroupInline(admin.TabularInline):
    model = tracker_models.ParameterGroup.parameters.through
    min_num = 1
    extra = 0


@admin.register(tracker_models.Parameter)
class ParameterAdmin(admin.ModelAdmin):
    inlines = [
        ParameterGroupInline,
    ]


@admin.register(tracker_models.ParameterGroup)
class ParameterGroupAdmin(admin.ModelAdmin):
    pass


class Experiment(tracker_models.Experiment):
    pass

    def __str__(self):
        parameter_group_names = tracker_models.ParameterGroup.objects.filter(experimentparameter__in=self.experimentparameter_set.all()).values_list("parameter_group_name", flat=True).distinct()
        return f'{self.alt_id} ({", ".join(parameter_group_names)})'


class ExperimentParameter(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.PROTECT)
    parameter = models.ForeignKey(tracker_models.Parameter, on_delete=models.PROTECT)
    parameter_group = models.ForeignKey(tracker_models.ParameterGroup, on_delete=models.PROTECT)
    parameter_value = models.CharField(max_length=100)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['experiment', 'parameter', 'parameter_group'], name='experiment_parameter_and_group_alt_key'),
        ]


class ExperimentParameterAdmin(admin.TabularInline):
    model = ExperimentParameter
    min_num = 1
    extra = 0


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    inlines = [
        ExperimentParameterAdmin,
    ]
