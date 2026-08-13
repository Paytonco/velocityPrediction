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

    if not os.getenv('DJANGO_SETTINGS_MODULE'):
        utils.initialize_django_in_notebook()
        from django.db.models import Count, Q

    from django_experiment_tracker.models import GitCommit, Parameter, ParameterEnum, ParameterGroup, Tag
    from django_experiment_tracker.experiment_generation import build_parameters_by_group, get_or_create_parameterized_model, create_parameterized_model_from_parameters
    from rna_vel_pred import models, datasets


@app.function
def get_latest_commit():
    return GitCommit.objects.order_by('-commit_time').first()


@app.cell
def _():
    get_latest_commit()
    return


@app.cell
def _():
    pgs = ParameterGroup.objects.filter(parameter_group_name__startswith='dataset_file')
    mo.ui.table(pgs.values())
    return (pgs,)


@app.function
def get_tags():
    return Tag.objects.none()


@app.cell
def _(pgs):
    _df_rows = []
    for _experiment_index, _eps in enumerate(build_parameters_by_group(pgs)):
        for _parameter_group, _parameter, _parameter_value in _eps:
            _df_rows.append(dict(
                experiment_index=_experiment_index,
                parameter_group=str(_parameter_group),
                parameter=str(_parameter),
                parameter_value=str(_parameter_value),
            ))
    pl.DataFrame(_df_rows)
    return


@app.cell
def _(pgs):
    create_parameterized_model_from_parameters(
        model=models.SavedDatasetFile,
        parameter_model=models.SavedDatasetFileParameter,
        parameters=build_parameters_by_group(pgs),
        model_kwargs=dict(git_commit=get_latest_commit()),
        tags=get_tags(),
    )
    return


@app.cell
def _():
    {p.parameter.parameter_name: p.parse() for p in models.SavedDatasetFileParameter.objects.all()}
    return


@app.cell
def _():
    sf = models.SavedDatasetFile.objects.first()
    models.SavedDatasetFileParameter.objects.filter(saved_dataset_file=sf)
    return (sf,)


@app.cell
def _():
    raw_data_dir = utils.DIR_ROOT/'data'
    processed_data_dir = raw_data_dir/'processed'
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    return processed_data_dir, raw_data_dir


@app.cell
def _(sf):
    sf.alt_id
    return


@app.cell
def _(processed_data_dir, raw_data_dir, sf):
    _params = {p.parameter.parameter_name: p.parse() for p in sf.saveddatasetfileparameter_set.all()}
    print(_params)
    ds = datasets.get_dataset(processed_data_dir/f'{sf.alt_id}.nc', partial(datasets.process_pancreas, raw_data_dir/f'{sf.alt_id}.h5ad', **_params))
    ds
    return


if __name__ == "__main__":
    app.run()
