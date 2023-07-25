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

# must call velocity_pseudotime before velocity_graph
scv.tl.velocity_graph(adata)
scv.tl.velocity_pseudotime(adata)

scv.tl.velocity_embedding(adata, basis='umap')
breakpoint()
