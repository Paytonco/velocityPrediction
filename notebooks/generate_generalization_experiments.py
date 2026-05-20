import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    import os

    import marimo as mo
    import polars as pl

    if not os.getenv('DJANGO_SETTINGS_MODULE'):
        from utils import initialize_django_in_notebook
        initialize_django_in_notebook()
    from django.db.models import Count, Q

    from django_experiment_tracker.models import GitCommit, Parameter, ParameterEnum, ParameterGroup, Tag
    from rna_vel_pred.models import Experiment, ExperimentParameter
    from rna_vel_pred.utils import get_or_create_experiment


@app.function
def get_latest_commit():
    return GitCommit.objects.order_by('-commit_time').first()


@app.cell
def _():
    get_latest_commit()
    return


@app.cell
def _():
    mo.ui.table(ParameterGroup.objects.all().values())
    return


@app.function
def get_tags():
    Tag.objects.get_or_create(tag_value='Generalization')
    return Tag.objects.filter(tag_value='Generalization')


@app.function
def build_experiment_parameters():
    experiment_parameter_inserts = {}
    parameter_groups = ParameterGroup.objects.prefetch_related('parameters')
    total = 1
    for pg in parameter_groups:
        for p in pg.parameters.all():
            if p.parameter_enum:
                p_values = p.parameter_enum.parameterenumvalue_set.values_list('parameter_enum_value', flat=True)
            else:
                p_values = [p.parameter_default_value]
            experiment_parameter_inserts[(pg, p)] = p_values
            total *= len(p_values)
    experiment_parameters = (list(zip(experiment_parameter_inserts.keys(), c)) for c in itertools.product(*experiment_parameter_inserts.values()))
    return experiment_parameters


@app.cell
def _():
    df_rows = []
    for experiment_index, _eps in enumerate(build_experiment_parameters()):
        for (_parameter_group, _parameter), _parameter_value in _eps:
            df_rows.append(dict(
                experiment_index=experiment_index,
                parameter_group=str(_parameter_group),
                parameter=str(_parameter),
                parameter_value=str(_parameter_value),
            ))
    pl.DataFrame(df_rows)
    return


@app.function
def create_experiments_and_parameters(experiment_rows_to_create, experiment_parameter_rows_to_create, tags):
    _created_experiment_rows = Experiment.objects.bulk_create(experiment_rows_to_create)
    _assigned_tags = [Experiment.tags.through(experiment_id=e.id, tag_id=t.id) for e in _created_experiment_rows for t in tags]
    Experiment.tags.through.objects.bulk_create(_assigned_tags)
    ExperimentParameter.objects.bulk_create(experiment_parameter_rows_to_create)
    experiment_rows_to_create = []
    experiment_parameter_rows_to_create = []


@app.cell(disabled=True)
def _(experiment_parameters, total):
    tags = Tag.objects.filter(tag_value='Generalization')
    experiment_rows_to_create = []
    experiment_parameter_rows_to_create = []
    for _eps in mo.status.progress_bar(experiment_parameters, total=total):
        existed, (experiment_row, experiment_parameter_rows) = get_or_create_experiment(
            _eps,
            experiment_model_kwargs=dict(git_commit=get_latest_commit()),
        )
        if not existed:
            experiment_rows_to_create.append(experiment_row)
            experiment_parameter_rows_to_create.extend(experiment_parameter_rows)
        if len(experiment_rows_to_create) >= 100:
            create_experiments_and_parameters(experiment_rows_to_create, experiment_parameter_rows_to_create, tags)
            # _created_experiment_rows = Experiment.objects.bulk_create(experiment_rows_to_create)
            # _assigned_tags = [Experiment.tags.through(experiment_id=e.id, tag_id=t.id) for e in _created_experiment_rows for t in tags]
            # Experiment.tags.through.objects.bulk_create(_assigned_tags)
            # ExperimentParameter.objects.bulk_create(experiment_parameter_rows_to_create)
            experiment_rows_to_create = []
            experiment_parameter_rows_to_create = []
    if len(experiment_rows_to_create) > 0:
        create_experiments_and_parameters(experiment_rows_to_create, experiment_parameter_rows_to_create, tags)
    return


if __name__ == "__main__":
    app.run()
