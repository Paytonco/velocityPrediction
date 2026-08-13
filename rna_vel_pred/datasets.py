from functools import partial

import numpy as np
import polars as pl
import scvelo
import scanpy
import xarray as xr


def make_measurement_dataframe(t, pos, vel):
    return pl.DataFrame({
        't': t,
        **{f'pos_{i+1}': v for i, v in enumerate(pos.T)},
        **{f'vel_{i+1}': v for i, v in enumerate(vel.T)},
    })


def make_measurement_dataset(t, pos, vel):
    return xr.Dataset(
        data_vars={
            'velocity': (('measurement', 'component'), np.asarray(vel, dtype=np.float32)),
            'position': (('measurement', 'component'), np.asarray(pos, dtype=np.float32)),
        },
        coords={
            'measurement': np.arange(len(pos)),
            'component': [f'x_{i+1}' for i in range(pos.shape[1])],
            'pseudotime': ('measurement', np.asarray(t, dtype=np.float32)),
        },
    )


def get_simple_motif(rng, measurement_count, initial_noise_scale):
    t = 3 * rng.random(measurement_count)
    y0 = initial_noise_scale * rng.random(measurement_count)
    x = t
    y = (y0 - 1) * np.exp(np.square(x) / 10) + 1
    pos = np.stack((x, y)).T
    vel = np.stack((np.ones(measurement_count), x / 5 * (y - 1))).T
    return t, pos, vel


def get_oscillation_motif(rng, measurement_count, initial_noise_scale):
    t = 2 * np.pi * rng.random(measurement_count)
    x0 = 1 + initial_noise_scale * (2 * rng.random(measurement_count) - 1)
    x = x0 * np.cos(t)
    y = -x0 * np.sin(t)
    pos = np.stack((x, y)).T
    vel = np.stack((y, -x)).T
    return t, pos, vel


def get_bifurcation_motif(rng, measurement_count, initial_noise_scale):
    t = 6 * rng.random(measurement_count)
    y0 = 1 + initial_noise_scale * (rng.random(measurement_count) / 2 + 0.5) * (2 * rng.binomial(1, 0.5, size=measurement_count) - 1)
    x = t
    y = (y0 - 1) * np.exp(np.square(x) / 10) + 1
    pos = np.stack((x, y)).T
    vel = np.stack((np.ones(measurement_count), x / 5 * (y - 1))).T
    return t, pos, vel


def get_detransition_motif(rng, measurement_count, initial_noise_scale):
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


def process_scvelo(raw_data, umap_dimension):
    scvelo.preprocessing.filter_genes(raw_data, min_shared_cells=20)  # scanpy.preprocessing.filter_genes(adata)
    scvelo.preprocessing.normalize_per_cell(raw_data)
    scanpy.preprocessing.log1p(raw_data)
    scanpy.preprocessing.highly_variable_genes(raw_data, n_top_genes=2000, subset=True)  # scanpy.preprocessing.filter_genes_dispersion(data, n_top_genes=2000)

    scanpy.preprocessing.neighbors(raw_data, n_neighbors=30)
    scvelo.preprocessing.moments(raw_data, n_pcs=30)

    scanpy.tools.umap(raw_data, n_components=umap_dimension)

    scvelo.tools.velocity(raw_data)
    scvelo.tools.velocity_graph(raw_data, show_progress_bar=False)
    scvelo.tools.velocity_embedding(raw_data, basis='umap')
    scvelo.tools.velocity_pseudotime(raw_data)

    t = raw_data.obs.velocity_pseudotime
    pos = raw_data.obsm['X_umap']
    vel = raw_data.obsm['velocity_umap']

    return t, pos, vel


def get_dataset(processed_file_path, process_dataset_func=None):
    try:
        with xr.open_dataset(processed_file_path) as ds:
            return ds.load()
    except FileNotFoundError as e:
        if process_dataset_func is None:
            raise e

    ds = make_measurement_dataset(*process_dataset_func())
    processed_file_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(processed_file_path)
    return ds


def process_pancreas(raw_file_path, umap_dimension=2):
    raw_data = scvelo.datasets.pancreas(file_path=raw_file_path)
    return process_scvelo(raw_data, umap_dimension)


def process_dentate_gyrus(raw_file_path):
    raw_data = scvelo.datasets.dentategyrus(file_path=raw_file_path)
    return process_scvelo(raw_data)


def process_bonemarrow(raw_file_path):
    raw_data = scvelo.datasets.bonemarrow(file_path=raw_file_path)
    return process_scvelo(raw_data)


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


def process_forebrain():
    return process_scvelo(scvelo.datasets.forebrain)


if __name__ == '__main__':
    pass
    # _measurement_count = 4000
    # _rng = np.random.default_rng(seed=42)
    # dataset = make_measurement_dataset(*get_simple_motif(_rng, _measurement_count, 0.05))
    # print(dataset)
    import os
    import utils
    if not os.getenv('DJANGO_SETTINGS_MODULE'):
        utils.initialize_django_in_notebook()

    from rna_vel_pred import models
    sf = models.SavedDatasetFile.objects.first()
    print(sf.alt_id)
    raw_data_dir = utils.DIR_ROOT/'data'
    processed_data_dir = raw_data_dir/'processed'
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    _params = {p.parameter.parameter_name: p.parse() for p in sf.saveddatasetfileparameter_set.all()}
    print(_params)
    ds = get_dataset(
        processed_data_dir/f'{sf.alt_id}.nc',
        partial(process_pancreas, raw_data_dir/f'{sf.alt_id}.h5ad', **_params)
    )
    print(ds)
