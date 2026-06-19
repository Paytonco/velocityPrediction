import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import os
    from pathlib import Path

    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt
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
def make_plot(df):
    fig_width = jour_sizes['IEEEtran']['twocol'] / 4
    fig, ax = plt.subplots(figsize=(fig_width, fig_width / golden))
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.30, top=0.95)
    ax.scatter(
        'pos_1',
        'pos_2',
        data=df,
        facecolors='none',
        edgecolors='tab:blue',
        s=15,
    )
    ax.quiver(df['pos_1'], df['pos_2'], df['vel_1'], df['vel_2'])
    ax.set(xlabel='$x_1$', ylabel='$x_2$')
    return fig, ax


@app.function
def make_plot_pseudotime(df):
    fig_width = jour_sizes['IEEEtran']['twocol'] / 2
    fig, ax = plt.subplots(figsize=(fig_width, fig_width / golden))
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.15, top=0.99)
    ax.scatter(
        'pos_1',
        'pos_2',
        data=df,
        facecolors='none',
        c='t',
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


@app.cell
def _(seed):
    _df = datasets.make_measurement_dataframe(
        *datasets.generate_simple_motif(np.random.default_rng(seed=seed), 4000, 0.05)
    ).sample(fraction=.005, seed=seed)
    _fig, _ax = make_plot(_df)
    _fig.savefig(save_dir/'DatasetMotifSimple.pdf')
    _fig.savefig('DatasetMotifSimple.pdf')
    _fig
    return


@app.cell
def _(seed):
    _df = datasets.make_measurement_dataframe(
        *datasets.generate_oscillation_motif(np.random.default_rng(seed=seed), 4000, 0.05)
    ).sample(fraction=.01, seed=seed)
    _fig, _ax = make_plot(_df)
    _fig.savefig(save_dir/'DatasetMotifOscillation.pdf')
    _fig
    return


@app.cell
def _(seed):
    _df = datasets.make_measurement_dataframe(
        *datasets.generate_bifurcation_motif(np.random.default_rng(seed=seed), 4000, 0.05)
    ).sample(fraction=.008, seed=seed)
    _fig, _ax = make_plot(_df)
    _max = (_df['pos_2'] - 1).abs().max()
    _ax.set_ylim((-_max + 1, _max + 1))
    _fig.savefig(save_dir/'DatasetMotifBifurcation.pdf')
    _fig
    return


@app.cell
def _(seed):
    _df = datasets.make_measurement_dataframe(
        *datasets.generate_detransition_motif(np.random.default_rng(seed=seed), 4000, 0.05)
    ).sample(fraction=.08, seed=seed)
    _fig, _ax = make_plot(_df)
    # _max = (_df['pos_2'] - 1).abs().max()
    # _ax.set_ylim((-_max + 1, _max + 1))
    _fig.savefig(save_dir/'DatasetMotifDetransition.pdf')
    _fig
    return


@app.cell
def _():
    _df = datasets.make_measurement_dataframe(
        *datasets.get_pancreas()
    )
    _fig, _ax = make_plot_pseudotime(_df)
    _fig.savefig(save_dir/'DatasetPancreas.pdf')
    _fig.savefig('DatasetPancreas.pdf')
    _fig
    return


@app.cell
def _():
    _df = datasets.make_measurement_dataframe(
        *datasets.get_dentate_gyrus()
    )
    _fig, _ax = make_plot_pseudotime(_df)
    _fig.savefig(save_dir/'DatasetDentateGyrus.pdf')
    _fig.savefig('DatasetDentateGyrus.pdf')
    _fig
    return


@app.cell
def _():
    _df = datasets.make_measurement_dataframe(
        *datasets.get_bonemarrow()
    )
    _fig, _ax = make_plot_pseudotime(_df)
    _fig.savefig(save_dir/'DatasetBonemarrow.pdf')
    _fig.savefig('DatasetBonemarrow.pdf')
    _fig
    return


@app.cell
def _():
    # _df = datasets.make_measurement_dataframe(
    #     *datasets.get_forebrain()
    # )
    # _fig, _ax = make_plot_pseudotime(_df)
    # _fig.savefig(save_dir/'DatasetForebrain.pdf')
    # _fig.savefig('DatasetForebrain.pdf')
    # _fig
    return


if __name__ == "__main__":
    app.run()
