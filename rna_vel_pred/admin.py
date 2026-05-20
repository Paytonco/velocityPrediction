from django.contrib import admin

from django_experiment_tracker import models as tracker_models
from rna_vel_pred import models


@admin.register(tracker_models.GitCommit)
class GitCommitAdmin(admin.ModelAdmin):
    pass


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
    min_num = 0
    extra = 0


@admin.register(tracker_models.Parameter)
class ParameterAdmin(admin.ModelAdmin):
    inlines = [
        ParameterGroupInline,
    ]


@admin.register(tracker_models.ParameterGroup)
class ParameterGroupAdmin(admin.ModelAdmin):
    pass


class ExperimentParameterAdmin(admin.TabularInline):
    model = models.ExperimentParameter
    min_num = 1
    extra = 0


@admin.register(models.Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    inlines = [
        ExperimentParameterAdmin,
    ]


@admin.register(tracker_models.Tag)
class TagAdmin(admin.ModelAdmin):
    pass
