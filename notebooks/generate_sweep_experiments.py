import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    import itertools
    from functools import partial
    import os

    import marimo as mo
    import polars as pl

    import utils
    from generate_sweep_datasets import get_git_commit, get_tags

    if not os.getenv('DJANGO_SETTINGS_MODULE'):
        utils.initialize_django_in_notebook()
        from django.db.models import Count, Q

    from django_experiment_tracker.models import GitCommit, Parameter, ParameterEnum, ParameterGroup, Tag
    from django_experiment_tracker.experiment_generation import build_parameters, get_or_create_parameterized_model, create_parameterized_model_from_parameters
    from rna_vel_pred import models, datasets


@app.cell
def _():
    get_git_commit()
    return


@app.cell
def _():
    ds_pgs = ParameterGroup.objects.filter(parameter_group_name__startswith='dataset')
    other_pgs = ParameterGroup.objects.exclude(parameter_group_name__startswith='dataset').exclude(parameter_group_name__startswith='file_dataset')
    mo.ui.table((other_pgs | ds_pgs).values())
    return ds_pgs, other_pgs


@app.cell
def _(ds_pgs, other_pgs):
    def loop_pgs():
        _experiment_index = 0
        for ds_pg in ds_pgs:
            pgs = other_pgs | ds_pgs.filter(pk=ds_pg.pk)
            try:
                dsf = models.SavedDatasetFile.objects.get(
                    saveddatasetfileparameter__parameter_group__parameter_group_name=f'file_{ds_pg.parameter_group_name}',
                    tags__in=get_tags(),
                    git_commit_valid_for=get_git_commit(),
                )
                file_dataset_alt_id = dsf.alt_id
            except models.SavedDatasetFile.DoesNotExist:
                file_dataset_alt_id = '???'
            yield partial(
                build_parameters,
                pgs,
                substitutes={
                    (ds_pg.parameter_group_name, 'file_dataset_alt_id'): [file_dataset_alt_id],
                }
            )

    return (loop_pgs,)


@app.cell
def _(loop_pgs):
    _df_rows = []
    for _pg_index, _pbuilder in enumerate(loop_pgs()):
        for _experiment_index, _eps in enumerate(_pbuilder()):
            for _parameter_group, _parameter, _parameter_value in _eps:
                _df_rows.append(dict(
                    experiment_index=_pg_index + _experiment_index,
                    parameter_group=str(_parameter_group),
                    parameter=str(_parameter),
                    parameter_value=str(_parameter_value),
                ))
    pl.DataFrame(_df_rows)
    return


@app.cell
def _(loop_pgs):
    for _pbuilder in loop_pgs():
        create_parameterized_model_from_parameters(
            model=models.Experiment,
            parameter_model=models.ExperimentParameter,
            parameters=_pbuilder(),
            model_kwargs=dict(
                git_commit_created=get_git_commit(),
                git_commit_valid_for=get_git_commit(),
            ),
            tags=get_tags(),
        )
    return


if __name__ == "__main__":
    app.run()
