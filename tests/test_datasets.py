import pytest
import torch
import hydra
import omegaconf
import lightning.pytorch as pl


import datasets
import utils


@pytest.mark.parametrize('overrides_dataset', [
    ['+dataset@dataset.A=MotifSimple'],
    ['+dataset@dataset.A=MotifOscillation'],
    ['+dataset@dataset.A=MotifBifurcation'],
    *[
        ['+dataset@dataset.A=Saved', f'dataset.A.data_dir={utils.DATA_DIR/ds}']
        for ds in ('bonemarrow', 'dentategyrus', 'pancreas',
                   # 'forebrain'  # large memory requirement
                   # 'pbmc68k'  # large memory requirement
                   )
    ]
])
def test_datasets_constant_poi_varying_num_neighbors(overrides_dataset):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(utils.ROOT_DIR/'configs')):
        cfg = hydra.compose(config_name='main', overrides=overrides_dataset).dataset.A

    with omegaconf.open_dict(cfg):
        cfg.num_neighbors = 11
    splits1 = map(datasets.DatasetMerged, zip(datasets.get_dataset(cfg)))
    with omegaconf.open_dict(cfg):
        cfg.num_neighbors = 13
    splits2 = map(datasets.DatasetMerged, zip(datasets.get_dataset(cfg)))

    for s1, s2 in zip(splits1, splits2):
        data1, data2 = s1._data, s2._data
        for (k, v1), (_, v2) in zip(data1, data2):
            if k.startswith('poi'):
                assert (v1 == v2).all(), k


@pytest.mark.parametrize('overrides_dataset', [
    ['+dataset@dataset.A=MotifSimple'],
    ['+dataset@dataset.A=MotifOscillation'],
    ['+dataset@dataset.A=MotifBifurcation'],
    *[
        ['+dataset@dataset.A=Saved', f'dataset.A.data_dir={utils.DATA_DIR/ds}']
        for ds in ('bonemarrow', 'dentategyrus', 'pancreas',
                   # 'forebrain'  # large memory requirement
                   # 'pbmc68k'  # large memory requirement
                   )
    ]
])
def test_datasets_constant_poi_varying_sparsify_step_time(overrides_dataset):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(utils.ROOT_DIR/'configs')):
        cfg = hydra.compose(config_name='main', overrides=overrides_dataset).dataset.A

    with omegaconf.open_dict(cfg):
        cfg.sparsify_step_time = 11
    splits1 = map(datasets.DatasetMerged, zip(datasets.get_dataset(cfg)))
    with omegaconf.open_dict(cfg):
        cfg.sparsify_step_time = 13
    splits2 = map(datasets.DatasetMerged, zip(datasets.get_dataset(cfg)))

    for s1, s2 in zip(splits1, splits2):
        data1, data2 = s1._data, s2._data
        for (k, v1), (_, v2) in zip(data1, data2):
            if k.startswith('poi'):
                assert (v1 == v2).all(), k


@pytest.mark.parametrize('num_neighbors', [7, 11, 13])
def test_datasets_num_neighbors_correct_size(num_neighbors):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(utils.ROOT_DIR/'configs')):
        cfg = hydra.compose(config_name='main', overrides=[
            '+dataset@dataset.A=MotifSimple',
            f'dataset.A.num_neighbors={num_neighbors}'
        ]).dataset.A

    splits = map(datasets.DatasetMerged, zip(datasets.get_dataset(cfg)))

    for s in splits:
        # num_neighbors does not count the poi
        assert all(d.num_nodes == num_neighbors for d in s)


@pytest.mark.parametrize('sparsify_step_time', [7, 11, 13])
def test_datasets_sparsify_step_time_sparsifies(sparsify_step_time):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(utils.ROOT_DIR/'configs')):
        cfg = hydra.compose(config_name='main', overrides=[
            '+dataset@dataset.A=MotifSimple',
            'dataset.A.num_neighbors=2',
            f'dataset.A.sparsify_step_time={sparsify_step_time}'
        ])

    splits = map(datasets.DatasetMerged, zip(datasets.get_dataset(cfg.dataset.A, rng_seed=cfg.rng_seed)))
    with pl.utilities.seed.isolate_rng():
        pl.seed_everything(cfg.rng_seed, workers=True)
        df = datasets.generate_measurements_simple(cfg.dataset.A.num_pnts, cfg.dataset.A.epsilon)
        splits_df = datasets.split_train_val_test(df, train_prec=cfg.dataset.A.splits.train, val_prec=cfg.dataset.A.splits.val, test_prec=cfg.dataset.A.splits.test, rng_seed=cfg.rng_seed)

    for s, s_df in zip(splits, splits_df):
        s_df = s_df.sort_values('t', ignore_index=True)
        for i, d in enumerate(s):
            if i < sparsify_step_time*2 + 1 or i >= len(s) - sparsify_step_time + 1:  # skip the poi on the boundary
                continue
            # i+1 is the index of the poi in s_df, assert to be sure
            assert torch.tensor(s_df.loc[i, 't']) == d.poi_t
            # the poi's neighbors should be sparsify_step_time away from i
            idx_ahead = i + sparsify_step_time
            idx_behind = i - sparsify_step_time
            t_df = torch.tensor(s_df.iloc[[idx_behind, idx_ahead]]['t'].to_numpy())
            assert (t_df == d.t.sort()[0]).all()
