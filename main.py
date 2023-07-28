import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as tg_nn


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.model = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, data):
        node_i, node_j = data.edge_index
        diff_t = data.t[node_i] - data.t[node_j]
        diff_pos = data.pos[node_i] - data.pos[node_j]
        r2 = (diff_pos**2).sum(1)
        weights = self.model(torch.cat((diff_t, r2), dim=1))
        return tg_nn.global_add_pool(weights * F.normalize(diff_pos, dim=1), node_i)


class ModelMessagePassing(tg_nn.MessagePassing):
    def __init__(self, dim):
        super().__init__(aggr='add')
        self.dim = dim
        self.model = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def reset_parameters(self):
        self.model.reset_parameters()

    def forward(self, t, pos, edge_index):
        node_i, node_j = edge_index
        diff_t = t[node_i] - t[node_j]
        diff_pos = pos[node_i] - pos[node_j]
        r2 = (diff_pos**2).sum(1)
        weights = self.model(torch.cat((diff_t, r2), dim=1))

        return self.propagate(edge_index, weights=weights, diff_pos=diff_pos)

    def message(self, weights_i, diff_pos_i):
        return weights_i * F.normalize(diff_pos_i, dim=1)


class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def loss(self, input, target):
        return F.mse_loss(input, target)

    def training_step(self, batch):
        return self.loss(self.model(batch), batch.vel)

    def validation_step(self, batch):
        return self.loss(self.model(batch), batch.vel)

    def predict_step(self, batch):
        return self.model(batch)
