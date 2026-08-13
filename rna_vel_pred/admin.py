from django.contrib import admin

from rna_vel_pred import models


class SavedDatasetFileTagAdmin(admin.TabularInline):
    model = models.SavedDatasetFile.tags.through
    min_num = 0
    extra = 0


class SavedDatasetFileParameterAdmin(admin.TabularInline):
    model = models.SavedDatasetFileParameter
    min_num = 1
    extra = 0


@admin.register(models.SavedDatasetFile)
class SavedDatasetFileAdmin(admin.ModelAdmin):
    inlines = [
        SavedDatasetFileParameterAdmin,
        # SavedDatasetFileTagAdmin,
    ]


class ExperimentTagAdmin(admin.TabularInline):
    model = models.Experiment.tags.through
    min_num = 0
    extra = 0


class ExperimentParameterAdmin(admin.TabularInline):
    model = models.ExperimentParameter
    min_num = 1
    extra = 0


@admin.register(models.Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    inlines = [
        ExperimentParameterAdmin,
        # ExperimentTagAdmin,
    ]
