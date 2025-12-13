from omegaconf import OmegaConf
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from fixtures import init_hydra_cfg, engine

from conf import conf
from rna_vel_pred import datasets, models, utils


def test_equivariance(engine):
    cfg = init_hydra_cfg('conf', [
        'model=Second',
        'datasets=[{_target_:conf.datasets.SimpleMotif}]',
    ])
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))

    splits = datasets.get_merged_dataset(cfg, cfg.data_dir, rng_seed=cfg.rng_seed)
    datum = next(iter(DataLoader(splits['train'])))

    angle = 2 * torch.pi * torch.rand(1)
    rot = torch.tensor([
        [angle.cos(), -angle.sin()],
        [angle.sin(), angle.cos()],
    ])
    trans = torch.rand((1, 2))

    datum_transformed = Data(
        pos=datum.pos @ rot.T + trans,
        poi_pos=datum.poi_pos @ rot.T + trans,
        vel=datum.vel @ rot.T + trans,
        poi_vel=datum.poi_vel @ rot.T + trans,
        t=datum.t + trans[0, 0],
        poi_t=datum.poi_t + trans[0, 0],
        edge_index=datum.edge_index,
        batch=datum.batch,
        ptr=datum.ptr,
        poi_measurement_id=datum.poi_measurement_id,
    )

    model, ckpt_path = models.get_model(cfg.model, rng_seed=cfg.rng_seed)
    with torch.no_grad():
        vel = model(datum)
        vel_transformed = model(datum_transformed)

        assert (vel - vel_transformed @ rot).abs().sum() < 1e-6


def test_no_data_sample_mixing(engine):
    cfg = init_hydra_cfg('conf', [
        'model=Second',
        'datasets=[{_target_:conf.datasets.SimpleMotif,neighbor_count:2}]',
    ])
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))

    splits = datasets.get_merged_dataset(cfg, cfg.data_dir, rng_seed=cfg.rng_seed)
    batch = next(iter(DataLoader(splits['train'], batch_size=2)))

    model, ckpt_path = models.get_model(cfg.model, rng_seed=cfg.rng_seed)
    batch.pos.requires_grad_()
    vel = model(batch)
    loss = (vel[0] - 1).square().mean()
    loss.backward()
    assert batch.pos.grad[batch.batch == 0].square().mean() > 0
    assert batch.pos.grad[batch.batch == 1].square().mean() == 0
