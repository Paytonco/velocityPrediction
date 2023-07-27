from pathlib import Path
import scvelo as scv
import numpy as np
import pandas as pd

"""
scv.datasets.pancreas()
scv.datasets.dentategyrus()
scv.datasets.forebrain()
#scv.datasets.gastrulation()
scv.datasets.bonemarrow()
scv.datasets.pbmc68k()
"""

DATASETS = [
    'pancreas'
    'dentategyrus',
    'forebrain',
    'gastrulation',
    'bonemarrow',
    'pbmc68k',
]


def get_csvs(name, basis_dim=2):
    folder = Path(f'data/{name}')
    adata = getattr(scv.datasets, name)(folder / f'{name}.h5ad')
    scv.pp.filter_and_normalize(adata)
    scv.pp.moments(adata)
    scv.tl.velocity(adata, mode='stochastic')

# must call velocity_pseudotime before velocity_graph
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_pseudotime(adata)

    scv.tl.velocity_embedding(adata, basis='umap')

    pseudotime = adata.obs.velocity_pseudotime
    positions = adata.obsm['X_umap']
    velocities = adata.obsm['velocity_umap']
    data = pd.DataFrame(
        data=np.concatenate((positions, velocities), axis=1),
        index=pseudotime.rename('t'),
        columns=['x1', 'x2', 'v1', 'v2']
    )
    data.to_csv(folder / f'{name}.csv')


for ds in DATASETS:
    get_csvs(ds)
