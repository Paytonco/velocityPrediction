import scvelo as scv

"""
scv.datasets.pancreas()
scv.datasets.dentategyrus()
scv.datasets.forebrain()
#scv.datasets.gastrulation()
scv.datasets.bonemarrow()
scv.datasets.pbmc68k()
"""

adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata)
scv.pp.moments(adata)
scv.tl.velocity(adata, mode='stochastic')
scv.pl.velocity_graph(adata)
scv.pl.velocity_embedding(adata, basis='umap')
scv.tl.velocity_pseudotime(adata)
mat = adata.X
breakpoint()