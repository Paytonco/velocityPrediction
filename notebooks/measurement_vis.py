import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import os
    from pathlib import Path
    from functools import partial

    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt
    import xarray as xr
    pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.

    jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt},
                  "CQG": {"onecol": 374.*pt}, # CQG is only one column
                  "IEEEtran": {"twocol": 516.*pt}, # CQG is only one column
                  # Add more journals below. Can add more properties to each journal
                 }

    my_width = jour_sizes["PRD"]["onecol"]
    # Our figure's aspect ratio
    golden = (1 + 5 ** 0.5) / 2
    # golden = 1
    try:
        plt.style.use('style.mplstyle')
    except RuntimeError:
        plt.rcParams.update({
            'mathtext.fontset': 'cm',
            'font.family': 'serif',
        })
    save_dir = Path('/home/ttransue/GitHub/Overleaf/velocityPrediction/figs_new/DatasetVis')
    import numpy as np
    import polars as pl

    if not os.getenv('DJANGO_SETTINGS_MODULE'):
        from utils import initialize_django_in_notebook
        initialize_django_in_notebook()

    from rna_vel_pred import datasets


@app.cell
def _():
    seed = 42
    return (seed,)


@app.function
def make_plot(dataset):
    fig_width = jour_sizes['IEEEtran']['twocol'] / 4
    fig, ax = plt.subplots(figsize=(fig_width, fig_width / golden))
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.30, top=0.95)

    position = dataset['position']
    velocity = dataset['velocity']

    x = position.sel(component='x_1')
    y = position.sel(component='x_2')
    u = velocity.sel(component='x_1')
    v = velocity.sel(component='x_2')

    ax.scatter(
        x, y,
        facecolors='none',
        edgecolors='tab:blue',
        s=15,
    )
    ax.quiver(x, y, u, v)
    ax.set(xlabel='$x_1$', ylabel='$x_2$')

    return fig, ax


@app.function
def make_plot_pseudotime(dataset):
    fig_width = jour_sizes['IEEEtran']['twocol'] / 2
    fig, ax = plt.subplots(figsize=(fig_width, fig_width / golden))
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.15, top=0.99)
    
    position = dataset['position']
    x = position.sel(component='x_1')
    y = position.sel(component='x_2')
    t = dataset.coords['pseudotime']
    
    ax.scatter(
        x, y,
        facecolors='none',
        c=t,
        cmap='viridis',
        vmin=0,
        vmax=1,
        s=15,
    )
    ax.set(xlabel='$x_1$', ylabel='$x_2$')
    return fig, ax


@app.function
def make_plot_pseudotime_colorbar():
    fig_width = jour_sizes['IEEEtran']['twocol']
    fig, ax = plt.subplots(figsize=(my_width, 0.25))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.55, top=0.85)
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
    return fig, ax


@app.cell
def _():
    _fig, _ = make_plot_pseudotime_colorbar()
    _fig.savefig(save_dir/'DatasetPseudotimeColorbar.pdf')
    _fig.savefig('DatasetPseudotimeColorbar.pdf')
    _fig
    return


@app.function
def subsample_dataset(rng, dataset, subsample_frac):
    return dataset.isel(measurement=rng.choice(dataset.sizes['measurement'], size=int(subsample_frac * dataset.sizes['measurement']), replace=False))


@app.cell
def _(seed):
    _measurement_count = 4000
    _rng = np.random.default_rng(seed=seed)
    _fig, _ax = make_plot(subsample_dataset(_rng, datasets.make_measurement_dataset(
        *datasets.get_simple_motif(_rng, _measurement_count, 0.05)
    ), 0.005))
    _fig.savefig(save_dir/'DatasetMotifSimple.pdf')
    _fig.savefig('DatasetMotifSimple.pdf')
    _fig
    return


@app.cell
def _(seed):
    _measurement_count = 4000
    _rng = np.random.default_rng(seed=seed)
    _fig, _ax = make_plot(subsample_dataset(_rng, datasets.make_measurement_dataset(
        *datasets.get_oscillation_motif(_rng, _measurement_count, 0.05)
    ), 0.01))
    _fig.savefig(save_dir/'DatasetMotifOscillation.pdf')
    _fig
    return


@app.cell
def _(seed):
    _measurement_count = 4000
    _rng = np.random.default_rng(seed=seed)
    _dataset = datasets.make_measurement_dataset(
        *datasets.get_bifurcation_motif(_rng, _measurement_count, 0.05)
    )
    _fig, _ax = make_plot(subsample_dataset(_rng, _dataset, 0.008))
    _max = np.abs(_dataset['position'].sel(component='x_2')).max()
    _ax.set_ylim((-_max + 1, _max + 1))
    _fig.savefig(save_dir/'DatasetMotifBifurcation.pdf')
    _fig
    return


@app.cell
def _(seed):
    _measurement_count = 4000
    _rng = np.random.default_rng(seed=seed)
    _fig, _ax = make_plot(subsample_dataset(_rng, datasets.make_measurement_dataset(
        *datasets.get_detransition_motif(_rng, _measurement_count, 0.05)
    ), 0.08))
    _fig.savefig(save_dir/'DatasetMotifDetransition.pdf')
    _fig
    return


@app.cell
def _():
    raw_data_dir = Path('data')
    processed_data_dir = Path('data/processed')
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    return processed_data_dir, raw_data_dir


@app.cell
def _(processed_data_dir, raw_data_dir):
    _fig, _ax = make_plot_pseudotime(datasets.get_dataset(
        processed_file_path=processed_data_dir/'pancreas'/'pancreas.nc',
        process_dataset_func=partial(datasets.process_pancreas, raw_data_dir/'pancreas'/'pancreas.h5ad'),
    ))
    _fig.savefig(save_dir/'DatasetPancreas.pdf')
    _fig.savefig('DatasetPancreas.pdf')
    _fig
    return


@app.cell
def _(processed_data_dir, raw_data_dir):
    datasets.get_dataset(
        processed_file_path=processed_data_dir/'pancreas'/'pancreas.nc',
        process_dataset_func=partial(datasets.process_pancreas, raw_data_dir/'pancreas'/'pancreas.h5ad'),
    )
    return


@app.cell
def _(processed_data_dir, raw_data_dir):
    _fig, _ax = make_plot_pseudotime(datasets.get_dataset(
        processed_file_path=processed_data_dir/'dentate_gyrus'/'dentate_gyrus.nc',
        process_dataset_func=partial(datasets.process_dentate_gyrus, raw_data_dir/'dentate_gyrus'/'dentate_gyrus.h5ad'),
    ))
    _fig.savefig(save_dir/'DatasetDentateGyrus.pdf')
    _fig.savefig('DatasetDentateGyrus.pdf')
    _fig
    return


@app.cell
def _(processed_data_dir, raw_data_dir):
    _fig, _ax = make_plot_pseudotime(datasets.get_dataset(
        processed_file_path=processed_data_dir/'bonemarrow'/'bonemarrow.nc',
        process_dataset_func=partial(datasets.process_bonemarrow, raw_data_dir/'bonemarrow'/'bonemarrow.h5ad'),
    ))
    _fig.savefig(save_dir/'DatasetBonemarrow.pdf')
    _fig.savefig('DatasetBonemarrow.pdf')
    _fig
    return


@app.cell
def _():
    # _fig, _ax = make_plot_pseudotime(datasets.get_dataset(
    #     processed_file_path=processed_data_dir/'bonemarrow'/'bonemarrow.nc',
    #     process_dataset_func=partial(datasets.process_dentate_gyrus, raw_data_dir/'bonemarrow'/'bonemarrow.h5ad'),
    # ))
    # _fig.savefig(save_dir/'DatasetForebrain.pdf')
    # _fig.savefig('DatasetForebrain.pdf')
    # _fig
    return


if __name__ == "__main__":
    app.run()
