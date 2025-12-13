import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.data import Data
import lightning.pytorch as pl
from einops import repeat

import conf.models
from rna_vel_pred import utils


class First(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.weighter = nn.Sequential()
        # dims = [2, *([cfg.hidden.dim] * cfg.hidden.layers), 1]
        # for layer, (d_in, d_out) in enumerate(zip(dims, dims[1:])):
        #     self.weighter.append(nn.Linear(d_in, d_out, bias=cfg.bias))
        #     if layer < len(dims) - 2:
        #         self.weighter.append(getattr(nn, cfg.activation)())

        # works for simple/oscillation
        # self.weighter = nn.Sequential(
        #     nn.Linear(2, 2),
        #     nn.ReLU(),
        #     nn.Linear(2, 1),
        # )
        # other?
        self.weighter = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )

    def forward(self, t, pos, poi_t, poi_pos, batch):
        batch = batch.batch
        diff_t = torch.sign(t - poi_t[batch])
        diff_pos = pos - poi_pos[batch]
        r2 = diff_pos.pow(2).sum(1)
        weights = self.weighter(torch.stack((diff_t, r2), dim=1))
        return utils.normalize(
            tg.nn.global_add_pool(weights * diff_pos, batch)
        )


class GeneralReLU(nn.Module):
    def __init__(self, leak=0., sub=0., maxv=None):
        super().__init__()
        self.leak = leak
        self.sub = sub
        self.maxv = maxv

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(x, self.leak)
        x -= self.sub
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


class Linear(nn.Module):
    def __init__(self, *args, act=None, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)
        self.act = act

    def forward(self, input):
        output = self.linear(input)
        if self.act is not None:
            output = self.act(output)
        return output


class GraphConv(nn.Module):
    def __init__(self, *args, act=None, **kwargs):
        super().__init__()
        # self.conv = tg.nn.GraphConv(*args, **kwargs)
        self.conv = tg.nn.GATConv(*args, **kwargs)
        self.act = act

    def forward(self, x=None, edge_index=None):
        output = self.conv(x=x, edge_index=edge_index)
        if self.act is not None:
            output = self.act(x)
        return output


class Second(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        act = GeneralReLU(leak=0.2, sub=0.1)
        in_dim = 3 if cfg.use_angle_input else 4
        out_dim = 2 if cfg.predict_cos_sin else 1

        mult = 8
        hidden_dim = mult * in_dim
        self.weighter = nn.Sequential(
            Linear(hidden_dim, hidden_dim, act=act),
            Linear(hidden_dim, hidden_dim, act=act),
            Linear(hidden_dim, hidden_dim, act=act),
            Linear(hidden_dim, out_dim),
        )

        self.gnn = tg.nn.Sequential('x, edge_index, batch', [
            (nn.Identity(), 'x -> x'),
            (Linear(in_dim, hidden_dim, act=act), 'x -> x'),
            (Linear(hidden_dim, hidden_dim, act=act), 'x -> x'),
            (Linear(hidden_dim, hidden_dim, act=act), 'x -> x'),
            (GraphConv(hidden_dim, hidden_dim, aggr='mean', negative_slope=act.leak, heads=4, act=act), 'x, edge_index -> x'),
            (tg.nn.LayerNorm(hidden_dim), 'x, batch -> x'),
        ])

        for m in self.modules():
            if isinstance(m, (nn.Linear, tg.nn.Linear)):
                print(type(m))
                nn.init.kaiming_normal_(m.weight, a=act.leak)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.cfg.reorient_to_reference_orientation:
            angle = torch.tensor(self.cfg.reference_orientation_angle)
            self.register_buffer('reference_orientation', torch.tensor([[angle.cos(), angle.sin()]]), persistent=False)

    def forward(self, batch):
        # normalize time and distance
        mbe_t = (batch.t - batch.poi_t[batch.batch])
        diff_pos = batch.pos - batch.poi_pos[batch.batch]
        r = diff_pos.pow(2).sum(1).sqrt()

        # deduce orientation of graph
        diff_pos_unit = utils.normalize(diff_pos)
        orientation = utils.normalize(tg.nn.global_mean_pool(diff_pos_unit, batch.batch))

        if self.cfg.reorient_to_reference_orientation:
            # compute rotation that rotates graph to the reference orientation
            # the graph orientation is considered to be a counter-clockwise rotation of the reference orientation
            # therefore, we need to compute a clockwise rotation of the graph
            angle_from_reference = utils.vangle(self.reference_orientation, orientation)
            angle_to_reference = -angle_from_reference
            rotation_to_reference = torch.stack([
                torch.stack([angle_to_reference.cos(), -angle_to_reference.sin()]),
                torch.stack([angle_to_reference.sin(), angle_to_reference.cos()]),
            ]).permute(2, 0, 1)

            # rotate the graph clockwise
            diff_pos_unit = utils.mv(rotation_to_reference[batch.batch], diff_pos_unit)
            orientation = self.reference_orientation.expand(orientation.shape)

        # apply gnn
        orientation_batched = orientation[batch.batch]
        angular_features = self.get_angular_features(orientation_batched, diff_pos_unit)
        features = torch.stack((mbe_t, r, *angular_features), dim=1)
        fmean = tg.nn.global_mean_pool(features, batch.batch)[batch.batch]
        fdiff = features - fmean
        node_count_per_graph = tg.utils.degree(batch.batch)[:, None]
        fstd = (
            tg.nn.global_mean_pool(fdiff.square(), batch.batch)
            * (node_count_per_graph / (node_count_per_graph - 1))  # Bessel's correction
        ).sqrt()[batch.batch]
        features = self.gnn(x=fdiff / fstd, edge_index=batch.edge_index, batch=batch.batch)
        # features = self.gnn(x=features, edge_index=batch.edge_index)

        # construct predicted velocity
        if self.cfg.predict_angle:
            features = tg.nn.global_mean_pool(features, batch.batch)
            angle_from_orientation = self.weighter(features).squeeze(1) * torch.pi
            rotation_from_orientation = torch.stack([
                torch.stack([angle_from_orientation.cos(), -angle_from_orientation.sin()]),
                torch.stack([angle_from_orientation.sin(), angle_from_orientation.cos()]),
            ]).permute(2, 0, 1)
            v = utils.mv(rotation_from_orientation, orientation)
        elif self.cfg.predict_cos_sin:
            """
            WARNING: this is unlikely to work well because the predicted cos and sin
            are not constrained to be consistent; that is, they are the cos and sin of different angles
            """
            features = tg.nn.global_mean_pool(features, batch.batch)
            cos_from_orientation, sin_from_orientation = self.weighter(features).squeeze(1).mT
            rotation_from_orientation = torch.stack([
                torch.stack([cos_from_orientation, -sin_from_orientation]),
                torch.stack([sin_from_orientation, cos_from_orientation]),
            ]).permute(2, 0, 1)
            v = utils.mv(rotation_from_orientation, orientation)
        else:
            weights = self.weighter(features)
            v = utils.normalize(
                tg.nn.global_add_pool(weights * diff_pos_unit, batch.batch)
            )

        if self.cfg.reorient_to_reference_orientation:
            # restore original graph orientation
            v = utils.mv(rotation_to_reference.mT, v)

        return v

    def get_angular_features(self, orientation, other):
        if self.cfg.use_angle_input:
            # normalize angle to [-1, 1]
            return [utils.vangle(orientation, other) / torch.pi]
        else:
            return [
                utils.vcos(orientation, other),
                utils.vsin(orientation, other),
            ]


class SecondRef(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        act = GeneralReLU(leak=0.25, sub=0.3)
        in_dim = 4
        self.weighter = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            act,
            nn.Linear(in_dim, in_dim),
            act,
            nn.Linear(in_dim, in_dim),
            act,
            nn.Linear(in_dim, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=act.leak)
                nn.init.zeros_(m.bias)

        conv = tg.nn.GraphConv(in_dim, in_dim, aggr='mean')
        self.gnn = tg.nn.Sequential('x, edge_index', [
            (conv, 'x, edge_index -> x'),
            act,
            (conv, 'x, edge_index -> x'),
            act,
            (conv, 'x, edge_index -> x'),
            act,
        ])

    def forward_pred_angle(self, t, pos, poi_t, poi_pos, batch, K=1):
        edge_index = batch.edge_index
        batch = batch.batch
        mbe_t = (t - poi_t[batch])
        mbe_t = (mbe_t - mbe_t.mean()) / mbe_t.std()
        diff_pos = pos - poi_pos[batch]
        r = diff_pos.pow(2).sum(1).sqrt()
        r = (r - r.mean()) / r.std()
        diff_pos_unit = utils.normalize(diff_pos)
        rand_angle = torch.rand(poi_t.shape[0], device=t.device) * 2 * torch.pi
        v = torch.stack([rand_angle.cos(), rand_angle.sin()], dim=1)
        for _ in range(K):
            v_batch = v[batch]
            cosine = (diff_pos_unit * v_batch).sum(1)
            sine = diff_pos_unit[:, 0] * v_batch[:, 1] - diff_pos_unit[:, 1] * v_batch[:, 0]
            angle = torch.sign(sine) * cosine.acos()
            h = torch.stack((mbe_t, r, angle / torch.pi), dim=1)
            h = self.gnn(x=h, edge_index=edge_index)
            h = tg.nn.global_mean_pool(h, batch)
            alpha = self.weighter(h).squeeze() * torch.pi
            rot = torch.stack([
                torch.stack([alpha.cos(), -alpha.sin()]),
                torch.stack([alpha.sin(), alpha.cos()]),
            ])
            return (v[:, None] @ rot.T).squeeze(1)
        return v

    def forward_equivariant_1(self, t, pos, poi_t, poi_pos, batch, K=1):
        edge_index = batch.edge_index
        batch = batch.batch
        mbe_t = (t - poi_t[batch])
        mbe_t = (mbe_t - mbe_t.mean()) / mbe_t.std()
        diff_pos = pos - poi_pos[batch]
        r = diff_pos.pow(2).sum(1).sqrt()
        r = (r - r.mean()) / r.std()
        diff_pos_unit = utils.normalize(diff_pos)
        # rand_angle = torch.rand(poi_t.shape[0], device=t.device) * 2 * torch.pi
        # v = torch.stack([rand_angle.cos(), rand_angle.sin()], dim=1)
        # v = (torch.tensor([[1., 1]], device=t.device) / 2**(1/2)).expand(poi_pos.shape)
        v = utils.normalize(diff_pos_unit.mean(0, keepdim=True)).expand(poi_pos.shape)
        # v = torch.tensor([[rand_angle.cos(), rand_angle.sin()]], device=t.device).expand(poi_t.shape[0], 2)
        for _ in range(K):
            v = v[batch]
            cosine = (diff_pos_unit * v).sum(1)
            sine = diff_pos_unit[:, 0] * v[:, 1] - diff_pos_unit[:, 1] * v[:, 0]
            # angle = torch.sign(sine) * cosine.acos()
            h = torch.stack((mbe_t, r, cosine, sine), dim=1)
            # h = torch.stack((mbe_t, r, angle / torch.pi), dim=1)
            h = self.gnn(x=h, edge_index=edge_index)
            # h = h.mean(dim=0, keepdim=True)
            # alpha = self.weighter(h).squeeze()
            # rot = torch.stack([
            #     torch.stack([alpha.cos(), -alpha.sin()]),
            #     torch.stack([alpha.sin(), alpha.cos()]),
            # ])
            # return rand_direction @ rot.T
            weights = self.weighter(h)
            v = utils.normalize(
                tg.nn.global_add_pool(weights * diff_pos, batch)
            )
        return v

    def forward(self, t, pos, poi_t, poi_pos, batch, K=1):
        edge_index = batch.edge_index
        batch = batch.batch
        mbe_t = (t - poi_t[batch])
        mbe_t = (mbe_t - mbe_t.mean()) / mbe_t.std()
        diff_pos = pos - poi_pos[batch]
        r = diff_pos.pow(2).sum(1).sqrt()
        r = (r - r.mean()) / r.std()
        diff_pos_unit = utils.normalize(diff_pos)
        v_ref = torch.ones((1, 2), device=t.device) / 2**(1/2)
        v = utils.normalize(tg.nn.global_mean_pool(diff_pos_unit, batch))
        cosine_ref = (v * v_ref).sum(1)
        sine_ref = v_ref[:, 0] * v[:, 1] - v_ref[:, 1] * v[:, 0]
        angle_ref = torch.sign(sine_ref) * cosine_ref.acos()
        angle_ref = -angle_ref
        rot_ref = torch.stack([
            torch.stack([angle_ref.cos(), -angle_ref.sin()]),
            torch.stack([angle_ref.sin(), angle_ref.cos()]),
        ]).permute(2, 0, 1)
        diff_pos_unit = (diff_pos_unit[:, None] @ rot_ref[batch].mT).squeeze(1)
        v = (v[batch, None] @ rot_ref[batch].mT).squeeze(1)
        for _ in range(K):
            # v = v[batch]
            cosine = (diff_pos_unit * v).sum(1)
            sine = v[:, 0] * diff_pos_unit[:, 1] - v[:, 1] * diff_pos_unit[:, 0]
            # angle = torch.sign(sine) * cosine.acos()  # check right hand rule!
            h = torch.stack((mbe_t, r, cosine, sine), dim=1)
            # h = torch.stack((mbe_t, r, angle / torch.pi), dim=1)
            h = self.gnn(x=h, edge_index=edge_index)
            # h = h.mean(dim=0, keepdim=True)
            # alpha = self.weighter(h).squeeze()
            # rot = torch.stack([
            #     torch.stack([alpha.cos(), -alpha.sin()]),
            #     torch.stack([alpha.sin(), alpha.cos()]),
            # ])
            # return rand_direction @ rot.T
            weights = self.weighter(h)
            v = utils.normalize(
                tg.nn.global_add_pool(weights * diff_pos_unit, batch)
            )
        return (v[:, None] @ rot_ref).squeeze(1)


class GNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.weighter = nn.Sequential()
        # dims = [2, *([cfg.hidden.dim] * cfg.hidden.layers), 1]
        # dims = [2, *([7] * 8), 1]
        # for layer, (d_in, d_out) in enumerate(zip(dims, dims[1:])):
        #     self.weighter.append(nn.Linear(d_in, d_out, bias=False))
        #     if layer < len(dims) - 2:
        #         self.weighter.append(nn.ReLU())
        # self.embed = nn.Sequential(
        #     nn.Linear(2, 8),
        # )
        self.gnn = tg.nn.PNA(2, 8, 3, aggregators=['mean', 'std', 'min', 'max'], scalers=['linear'],  deg=torch.ones(10, device='cuda')*10)
        # self.gnn = tg.nn.Sequential('x, edge_index', [
        #     (tg.nn.GraphConv(8, 8), 'x, edge_index -> x'),
        #     nn.ReLU(),
        #     (tg.nn.GraphConv(8, 8), 'x, edge_index -> x'),
        #     nn.ReLU(),
        #     (tg.nn.GraphConv(8, 8), 'x, edge_index -> x'),
        #     nn.ReLU(),
        #     (tg.nn.GraphConv(8, 8), 'x, edge_index -> x'),
        #     nn.ReLU(),
        #     (tg.nn.GraphConv(8, 8), 'x, edge_index -> x'),
        # ])
        self.unembed = nn.Sequential(
            nn.Linear(8, 1),
        )

    def forward(self, t, pos, poi_t, poi_pos, batch):
        edge_index = batch.edge_index
        batch = batch.batch
        diff_t = torch.sign(t - poi_t[batch])
        diff_pos = pos - poi_pos[batch]
        # signs = torch.sign(pos - poi_pos[batch])
        # means = tg.nn.global_mean_pool(pos, batch)
        # dot = ((means - poi_pos)[batch] * diff_pos).square().sum(1)
        # r = torch.sigmoid(diff_pos.square().sum(1).sqrt()) - .5
        r2 = diff_pos.square().sum(1)
        h = torch.cat((diff_t[:, None], r2[:, None]), dim=1)
        # h = self.embed(x)
        h = self.gnn(x=h, edge_index=edge_index)
        weights = self.unembed(h)
        return utils.normalize(
            tg.nn.global_add_pool(weights * diff_pos, batch)
        )


class FirstDistance(First):
    def forward(self, t, pos, poi_t, poi_pos, batch):
        batch = batch.batch
        diff_t = torch.sign(t - poi_t[batch])
        diff_pos = pos - poi_pos[batch]
        r = diff_pos.pow(2).sum(1).sqrt()
        weights = self.weighter(torch.stack((diff_t, r), dim=1))
        return utils.normalize(
            tg.nn.global_add_pool(weights * utils.normalize(diff_pos), batch)
        )


# class Second(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.linear = nn.Linear(2, 1)
#         self.norm_activation = getattr(nn, cfg.activation)()
#
#         dims = [2, *([cfg.hidden.dim] * cfg.hidden.layers), 2]
#         self.conv_layers = nn.ModuleList()
#         for idx, (c_in, c_out) in enumerate(zip(dims, dims[1:])):
#             step = 10 if idx == 0 else 1
#             self.conv_layers.append(nn.Conv2d(
#                     in_channels=c_in // 2, out_channels=c_out // 2,
#                     kernel_size=(step, 1), stride=(step, 1),
#                     bias=False
#             ))
#
#     def forward(self, t, pos, poi_t, poi_pos, batch):
#         # diff_t = torch.sign(t - poi_t[batch.batch]) * self.weighter_t
#         diff_pos = pos - poi_pos[batch.batch]
#         diff_pos = diff_pos  # + diff_pos * diff_t[:, None]
#         diff_pos = diff_pos.reshape(batch.batch_size, -1, 1, 2)
#         x = diff_pos.permute(0, 3, 1, 2)  # now (B, F, K, N)
#         x_even, x_odd = x[:, ::2], x[:, 1::2]
#         x_double_batch = torch.cat((x_even, x_odd))
#         for idx, conv in enumerate(self.conv_layers):
#             x_double_batch = conv(x_double_batch)
#             if idx < len(self.conv_layers) - 1:
#                 x_double_batch = self.equivariant_activation(x_double_batch, None, batch.batch_size)
#
#         x = x_double_batch.view(2, batch.batch_size, 1).permute(1, 2, 0).squeeze()
#
#         return utils.normalize(x)
#
#     def equivariant_activation(self, x_double_batch, diff_t, batch_size):
#         r = (x_double_batch.view(2, batch_size, -1, 1).pow(2).sum(0, keepdim=True) + 1e-5)
#         r = self.norm_activation(r)
#
#         return (x_double_batch.view(2, batch_size, -1, 1) * r).view(2 * batch_size, -1, 1, 1)


def get_model(cfg, rng_seed=0):
    with pl.utilities.seed.isolate_rng():
        pl.seed_everything(rng_seed, workers=True)
        if isinstance(cfg, conf.models.First):
            return First(cfg), None
        elif isinstance(cfg, conf.models.Second):
            return Second(cfg), None
        elif isinstance(cfg, conf.models.GNN):
            return GNN(cfg), None
        elif isinstance(cfg, conf.conf.Trained):
            model, _ = get_model(cfg.conf.model, rng_seed=cfg.conf.rng_seed)
            ckpt_path = cfg.conf.run_dir/cfg.ckpt_filename
            return model, ckpt_path
        else:
            raise ValueError(f'Unknown model: {cfg.name}')
