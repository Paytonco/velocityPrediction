import pytest
import hydra
from omegaconf import OmegaConf


import datasets
import utils


@pytest.mark.parametrize('overrides_dataset', [
    ['+dataset@dataset.1=MotifSimple'],
    # dict(name='MotifSimple', num_pnts=4000, epsilon=.05),
    # dict(name='MotifOscillation', num_pnts=4000, epsilon=.05),
    # dict(name='MotifBifurcation', num_pnts=4000, epsilon=.05),
    # dict(name='Saved', path=utils.ROOT/'data'/'bonemarrow'/'bonemarrow.csv'),
    # dict(name='Saved', path=utils.ROOT/'data'/'dentategyrus'/'dentategyrus.csv'),
    # dict(name='Saved', path=utils.ROOT/'data'/'forebrain'/'forebrain.csv'),
    # dict(name='Saved', path=utils.ROOT/'data'/'pancreas'/'pancreas.csv'),
    # dict(name='Saved', path=utils.ROOT/'data'/'pbmc68k'/'pbmc68k.csv'),
])
def test_datasets_reproducibilty(overrides_dataset):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(utils.ROOT/'configs')):
        cfg = hydra.compose(config_name='main', overrides=overrides_dataset).dataset['1']

    splits1 = map(datasets.DatasetMerged, zip(datasets.get_dataset(cfg)))
    splits2 = map(datasets.DatasetMerged, zip(datasets.get_dataset(cfg)))

    for s1, s2 in zip(splits1, splits2):
        data1, data2 = s1._data, s2._data
        for (k, v1), (_, v2) in zip(data1, data2):
            assert (v1 == v2).all(), k
