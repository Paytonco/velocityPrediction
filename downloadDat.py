from pathlib import Path
import scvelo as scv
import numpy as np
import pandas as pd

"""
scv.datasets.pancreas()
scv.datasets.dentategyrus()
scv.datasets.dentategyrus_lamanno()
scv.datasets.forebrain()
scv.datasets.gastrulation()
scv.datasets.bonemarrow()
scv.datasets.pbmc68k()
"""

DATASETS = [
    # scv.datasets.pancreas,
    # scv.datasets.dentategyrus,
    # scv.datasets.dentategyrus_lamanno,
    # scv.datasets.forebrain,
    # scv.datasets.gastrulation,  # too much memory
    # scv.datasets.bonemarrow,
    scv.datasets.pbmc68k,
]

"""
scv.datasets.forebrain

1. Modify .venv/lib/python3.10/site-packages/anndata/_io/h5ad.py in two places
   to create your own AnnData that renames the keys of the HDF5 file, and remove
   some backward compatibility:

   Put this before "return AnnData"
      key_map = dict(var='col_attrs', layers='layers', X='matrix', obs='row_attrs')
      return AnnData(**{k: read_dispatched(elem[v], callback) for k, v in key_map.items()})

   Remove this if block
      Backwards compat to <0.7
      if isinstance(f["obs"], h5py.Dataset):
          _clean_uns(adata)

2. run scv.pp.remove_duplicate_cells and scv.pp.neighbors before filter_and_normalize
3. run scv.tl.umap before scv.tl.velocity_embedding

or replace step 1. with pytables manipulation:
    import tables
    f = tables.open_file('data/forebrain/forebrain.h5ad', mode='r+')
    f.rename_node('/row_attrs', '/obs')
    f.rename_node('/col_attrs', '/var')
    f.rename_node('/matrix', '/X')
"""

"""
scv.datasets.bonemarrow

3. run scv.tl.umap before scv.tl.velocity_embedding
"""

"""
scv.datasets.pbmc68k

3. run scv.tl.umap before scv.tl.velocity_embedding
"""


def get_csvs(ds_func, basis_dim=2):
    name = ds_func.__name__
    folder = Path(f'data/{name}')
    try:
        breakpoint()
        adata = ds_func(str(folder / f'{name}.h5ad'))
    except Exception as e:
        import traceback
        traceback.print_exc()
        breakpoint()
        adata = ds_func(str(folder / f'{name}.h5ad'))
        print('oops')
    # scv.pp.remove_duplicate_cells(adata)  # for forebrain
    # scv.pp.neighbors(adata)  # for forebrain
    scv.pp.filter_and_normalize(adata)
    scv.pp.moments(adata)
    scv.tl.velocity(adata, mode='stochastic')

    scv.tl.velocity_graph(adata)
    scv.tl.velocity_pseudotime(adata)

    scv.tl.umap(adata)  # for forebrain and bonemarrow and pbmc68k
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
