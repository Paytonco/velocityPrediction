from django.db.models import Count, Q

from rna_vel_pred.models import Experiment, ExperimentParameter


def get_or_create_experiment(
    experiment_parameters,
    experiment_model_kwargs,
):
    q = Q()
    for (parameter_group, parameter), parameter_value in experiment_parameters:
        q |= Q(
            experimentparameter__parameter_group=parameter_group,
            experimentparameter__parameter=parameter,
            experimentparameter__parameter_value=parameter_value,
        )
    candidate_experiments = (
        Experiment.objects
        .annotate(
            total_params=Count("experimentparameter", distinct=True),
            matched_params=Count(
                "experimentparameter",
                filter=q,
                distinct=True,
            ),
        )
        .filter(
            total_params=len(experiment_parameters),
            matched_params=len(experiment_parameters),
        )
    )
    candidate_count = candidate_experiments.count()
    if candidate_count == 0:
        experiment_row = Experiment(**experiment_model_kwargs)
        experiment_parameter_rows = []
        for (parameter_group, parameter), parameter_value in experiment_parameters:
            experiment_parameter_row = ExperimentParameter(
                experiment=experiment_row,
                parameter_group=parameter_group,
                parameter=parameter,
                parameter_value=parameter_value,
            )
            experiment_parameter_rows.append(experiment_parameter_row)
    else:
        assert candidate_count == 1, candidate_experiments
        experiment_row = candidate_experiments.prefetch_related('experimentparameter_set').first()
        experiment_parameter_rows = experiment_row.experimentparameter_set.all()
    return candidate_count != 0, (experiment_row, experiment_parameter_rows)


def create_experiments_and_parameters(experiment_rows_to_create, experiment_parameter_rows_to_create, tags):
    _created_experiment_rows = Experiment.objects.bulk_create(experiment_rows_to_create)
    _assigned_tags = [Experiment.tags.through(experiment_id=e.id, tag_id=t.id) for e in _created_experiment_rows for t in tags]
    Experiment.tags.through.objects.bulk_create(_assigned_tags)
    ExperimentParameter.objects.bulk_create(experiment_parameter_rows_to_create)
    experiment_rows_to_create = []
    experiment_parameter_rows_to_create = []


def create_experiments_from_parameters(experiment_parameters, experiment_model_kwargs, tags, insert_batch_size=100):
    experiment_rows_to_create = []
    experiment_parameter_rows_to_create = []
    for _eps in experiment_parameters:
        existed, (experiment_row, experiment_parameter_rows) = get_or_create_experiment(
            _eps,
            experiment_model_kwargs=experiment_model_kwargs,
        )
        if not existed:
            experiment_rows_to_create.append(experiment_row)
            experiment_parameter_rows_to_create.extend(experiment_parameter_rows)
        if len(experiment_rows_to_create) >= insert_batch_size:
            create_experiments_and_parameters(experiment_rows_to_create, experiment_parameter_rows_to_create, tags)
            experiment_rows_to_create = []
            experiment_parameter_rows_to_create = []
    if len(experiment_rows_to_create) > 0:
        create_experiments_and_parameters(experiment_rows_to_create, experiment_parameter_rows_to_create, tags)
