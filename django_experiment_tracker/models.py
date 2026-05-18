import random
import string

from django.db import models
from django.utils.translation import gettext_lazy as _


def generate_random_string(k=8, chars=string.ascii_lowercase+string.digits):
    return ''.join(random.SystemRandom().choices(chars, k=k))


def generate_unique(model_class, field_name, sampler=generate_random_string):
    while model_class.objects.filter(**{field_name: (s := sampler)}).exists():
        pass
    return s


def db_default_random_string(half_length, prefix=''):
    return models.functions.Concat(
        models.Value(prefix),
        models.functions.Lower(
            models.Func(
                models.Func(models.Value(half_length), function='randomblob'),
                function='hex'
            )
        )
    )


class GitCommit(models.Model):
    commit_time = models.DateTimeField()
    branch = models.CharField(max_length=50)
    commit_sha = models.CharField(max_length=40)


class ParameterEnum(models.Model):
    """
    Enum values assignable to a parameter.
    """
    parameter_enum_description = models.CharField(max_length=100)


class ParameterEnumValue(models.Model):
    """
    One value in the enumeration of a parameter enum.
    """
    parameter_enum = models.ForeignKey(ParameterEnum, on_delete=models.CASCADE)


class Parameter(models.Model):
    """
    A parameter assingable to an experiment.
    """
    parameter_name = models.CharField(max_length=100)
    parameter_enum = models.ForeignKey(ParameterEnum, on_delete=models.PROTECT)


class Experiment(models.Model):
    PREFIX_ALT_ID = 'exp_'

    alt_id = models.CharField(max_length=8+len(PREFIX_ALT_ID), unique=True, editable=False, db_default=db_default_random_string(4, PREFIX_ALT_ID))
    git_commit = models.ForeignKey(GitCommit, on_delete=models.CASCADE, related_name='%(app_label)s_%(class)s_related', related_query_name='%(app_label)s_%(class)ss')
    time_created = models.DateTimeField(db_default=models.functions.Now())
    time_completed = models.DateTimeField(blank=True, null=True)
    exit_code = models.IntegerField(blank=True, null=True)

    class Meta:
        abstract = True

    # def save(self, *args, **kwargs):
    #     if not self.alt_id:
    #         self.alt_id = f"{self.PREFIX_ALT_ID}_{generate_unique(type(self), 'alt_id')}"
    #     return super().save(*args, **kwargs)


# class ExperimentParameter(models.Model):
#     experiment = models.ForeignKey(Experiment, on_delete=models.PROTECT)
#     parameter = models.ForeignKey(Parameter, on_delete=models.PROTECT)
#     pk = models.CompositePrimaryKey('experiment', 'parameter')
#     parameter_value = models.CharField(max_length=100)
#
#     class Meta:
#         abstract = True


# class OptunaOptimizationDirection(models.TextChoices):
#     MINIMIZE = ('min', _('Minimize'))
#     MAXIMIZE = ('max', _('Maximize'))
#
#
# class OptunaStudy(models.Model):
#     optimization_direction = models.CharField(max_length=max(map(len, OptunaOptimizationDirection)), choices=OptunaOptimizationDirection)
#     trial_count = models.PositiveIntegerField()
#
#     class Meta:
#         abstract = True
#
#
# class OptunaTrialParamType(models.TextChoices):
#     INTEGER = ('int', _('Integer'))
#     FLOAT = ('float', _('Float'))
#
#
# class OptunaTrialParam(models.Model):
#     param_name = models.CharField(max_length=50)
#     param_type = models.CharField(max_length=max(map(len, OptunaTrialParamType)), choices=OptunaTrialParamType)
#     lower_bound = models.FloatField()
#     upper_bound = models.FloatField()
#     log_sample = models.BooleanField(default=False)
#
#     class Meta:
#         abstract = True
#
#
# class SharedFile(models.Model):
#     PREFIX_ALT_ID = 'sf'
#
#     shared_file_alt_id = models.CharField(max_length=8+len(PREFIX_ALT_ID), unique=True, editable=False)
#
#     class Meta:
#         abstract = True
#
#     def save(self, *args, **kwargs):
#         if not self.shared_file_alt_id:
#             self.shared_file_alt_id = f"{self.PREFIX_ALT_ID}_{generate_unique(type(self), 'shared_file_alt_id')}"
#         return super().save(*args, **kwargs)
