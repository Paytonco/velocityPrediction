from functools import cache

import numpy as np
import polars as pl
import scvelo
import scanpy


def make_measurement_dataframe(t, pos, vel):
    return pl.DataFrame({
        't': t,
        **{f'pos_{i+1}': v for i, v in enumerate(pos.T)},
        **{f'vel_{i+1}': v for i, v in enumerate(vel.T)},
    })


def generate_simple_motif(rng, measurement_count, initial_noise_scale):
    t = 3 * rng.random(measurement_count)
    y0 = initial_noise_scale * rng.random(measurement_count)
    x = t
    y = (y0 - 1) * np.exp(np.square(x) / 10) + 1
    pos = np.stack((x, y)).T
    vel = np.stack((np.ones(measurement_count), x / 5 * (y - 1))).T
    return t, pos, vel


def generate_oscillation_motif(rng, measurement_count, initial_noise_scale):
    t = 2 * np.pi * rng.random(measurement_count)
    x0 = 1 + initial_noise_scale * (2 * rng.random(measurement_count) - 1)
    x = x0 * np.cos(t)
    y = -x0 * np.sin(t)
    pos = np.stack((x, y)).T
    vel = np.stack((y, -x)).T
    return t, pos, vel


def generate_bifurcation_motif(rng, measurement_count, initial_noise_scale):
    t = 6 * rng.random(measurement_count)
    y0 = 1 + initial_noise_scale * (rng.random(measurement_count) / 2 + 0.5) * (2 * rng.binomial(1, 0.5, size=measurement_count) - 1)
    x = t
    y = (y0 - 1) * np.exp(np.square(x) / 10) + 1
    pos = np.stack((x, y)).T
    vel = np.stack((np.ones(measurement_count), x / 5 * (y - 1))).T
    return t, pos, vel


def generate_detransition_motif(rng, measurement_count, initial_noise_scale):
    T = 5
    n_steps = 5_000
    dt = T / n_steps
    sigma = 0.5
    sqrt_dt = np.sqrt(dt)

    x = 1.0 + initial_noise_scale * rng.random(measurement_count)
    obs_steps = rng.integers(n_steps, size=measurement_count)
    t = obs_steps * dt

    def dot_y(x):
        a, b, c = x - 1, x - 2, x - 3
        f = -10 * a * b * c * (b * c + a * c + a * b) + 1
        return f

    x_obs = x.copy()
    for step in range(n_steps):
        mask = obs_steps == step
        if mask.any():
            x_obs[mask] = x[mask]
        f = dot_y(x)
        x = x + f * dt + sigma * sqrt_dt * rng.standard_normal(measurement_count)

    pos = np.stack((t, x_obs)).T
    vel = np.stack((np.ones(measurement_count), dot_y(x_obs))).T
    return t, pos, vel


def process_scvelo(dataset_func):
    data = dataset_func()
    scvelo.preprocessing.filter_genes(data, min_shared_cells=20)  # scanpy.preprocessing.filter_genes(adata)
    scvelo.preprocessing.normalize_per_cell(data)
    scanpy.preprocessing.log1p(data)
    scanpy.preprocessing.highly_variable_genes(data, n_top_genes=2000, subset=True)  # scanpy.preprocessing.filter_genes_dispersion(data, n_top_genes=2000)

    scanpy.preprocessing.neighbors(data, n_neighbors=30)
    scvelo.preprocessing.moments(data, n_pcs=30)

    scanpy.tools.umap(data, n_components=2)

    scvelo.tools.velocity(data)
    scvelo.tools.velocity_graph(data)
    scvelo.tools.velocity_embedding(data, basis='umap')
    scvelo.tools.velocity_pseudotime(data)

    t = data.obs.velocity_pseudotime
    pos = data.obsm['X_umap']
    vel = data.obsm['velocity_umap']

    return t, pos, vel


@cache
def get_dentate_gyrus():
    return process_scvelo(scvelo.datasets.dentategyrus)


@cache
def get_bonemarrow():
    return process_scvelo(scvelo.datasets.bonemarrow)


# def download_forebrain():
#     try:
#         adata = scvelo.datasets.forebrain()
#     except (TypeError, anndata._io.utils.AnnDataReadError):
#         f = tables.open_file(data_dir/cfg.h5ad_path, mode='r+')
#         # these are empty
#         f.remove_node('/row_graphs')
#         f.remove_node('/col_graphs')
#         # rename to match AnnData data structure
#         f.rename_node('/row_attrs', 'obs')
#         f.rename_node('/col_attrs', 'var')
#         f.rename_node('/matrix', 'X')
#         f.close()


@cache
def get_forebrain():
    return process_scvelo(scvelo.datasets.forebrain)


@cache
def get_pancreas():
    return process_scvelo(scvelo.datasets.pancreas)
    data = scvelo.datasets.pancreas()
    # scvelo.preprocessing.filter_and_normalize(adata, min_shared_cells=20)
    scvelo.preprocessing.filter_genes(data, min_shared_cells=20)  # scanpy.preprocessing.filter_genes(adata)
    scvelo.preprocessing.normalize_per_cell(data)
    scanpy.preprocessing.log1p(data)
    scanpy.preprocessing.highly_variable_genes(data, n_top_genes=2000)  # scanpy.preprocessing.filter_genes_dispersion(data, n_top_genes=2000)

    scanpy.preprocessing.neighbors(data, n_neighbors=30)
    scvelo.preprocessing.moments(data, n_pcs=30)

    scanpy.tools.umap(data, n_components=2)

    scvelo.tools.velocity(data)
    scvelo.tools.velocity_graph(data)
    scvelo.tools.velocity_embedding(data, basis='umap')
    scvelo.tools.velocity_pseudotime(data)

    t = data.obs.velocity_pseudotime
    pos = data.obsm['X_umap']
    vel = data.obsm['velocity_umap']

    return t, pos, vel


if __name__ == '__main__':
    rng = np.random.default_rng(seed=42)
    t, pos, vel = get_pancreas()
    df = make_measurement_dataframe(t, pos, vel)
    print(df)
