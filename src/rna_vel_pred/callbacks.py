from collections import defaultdict
from functools import partial
import itertools

import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch_geometric as tg
from tqdm import tqdm

import rna_vel_pred.models
import rna_vel_pred.utils


log = rna_vel_pred.utils.getLoggerByFilename(__file__)


class LogStats(pl.callbacks.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_dict(outputs, on_epoch=True, prog_bar=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_dict({f'val_{k}': v for k, v in outputs.items()}, on_epoch=True, prog_bar=True)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    CHECKPOINT_EQUALS_CHAR = '_'


class LSUV(pl.callbacks.Callback):
    def hook(self, module, inp, outp):
        self.mean = outp.mean()
        self.std = outp.std(correction=0)

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx):
        if trainer.current_epoch > 0 or batch_idx > 0:
            return

        for m_w, m_act in zip(pl_module.mlp_end_cap, pl_module.mlp_end_cap[1:]):
            if isinstance(m_w, nn.Linear):
                hook = m_act.register_forward_hook(self.hook)

                with torch.no_grad():
                    # this might only for with a one-step ODE solve
                    while pl_module.compute_loss(batch) is not None and (
                        self.mean.abs() > 1e-3
                        or
                        (self.std - 1).abs() > 1e-3
                    ) and (pbar := tqdm(itertools.count(), desc='LSUV')) is not None:
                        pbar.set_description(f'mean={self.mean}, std={self.std}')
                        m_w.bias -= self.mean
                        m_w.weight /= self.std

                hook.remove()


class SaveLoss(pl.callbacks.Callback):
    def on_train_start(self, trainer, pl_module):
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.losses.append(outputs['loss'])


class LayerStats(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.log_prefix_and_target = {}
        self.registered_hooks = []
        self.reset()

    def reset(self):
        raise NotImplementedError()

    def on_fit_start(self, trainer, pl_module: pl.LightningModule):
        for name, m in pl_module.named_modules():
            if isinstance(m, (nn.Identity, rna_vel_pred.models.Linear, rna_vel_pred.models.GraphConv)):
                self.log_prefix_and_target[f'{name}_{type(m).__name__}_'] = m

    def on_train_epoch_start(self, trainer, pl_module: pl.LightningModule):
        for log_prefix, module in self.log_prefix_and_target.items():
            self.registered_hooks.append(
                module.register_forward_hook(partial(self.log_hook, trainer, log_prefix))
            )

    def on_validation_start(self, trainer, pl_module):
        for h in self.registered_hooks:
            h.remove()
        self.registered_hooks = []

    def on_train_end(self, trainer, pl_module):
        for h in self.registered_hooks:
            h.remove()
        self.registered_hooks = []


class ColorfulDimension(LayerStats):
    def __init__(self, bins=40, min=0, max=10):
        super().__init__()
        self.bins = bins
        self.min = min
        self.max = max

    def reset(self):
        self.histograms = defaultdict(list)

    def log_hook(self, trainer: pl.Trainer, log_prefix, module, inp, outp):
        self.histograms[log_prefix].append(torch.histc(outp.abs().detach().cpu(), bins=self.bins, min=self.min, max=self.max))


class ActivationStats(LayerStats):
    def reset(self):
        self.stats = []

    def log_hook(self, trainer: pl.Trainer, log_prefix, module, inp, outp):
        stats = [
            ('mean', outp.mean().item()),
            ('std', outp.std(correction=0).item()),
        ]
        for stat, value in stats:
            self.stats.append(dict(
                step=trainer.global_step, epoch=trainer.current_epoch,
                layer=log_prefix, stat=stat, value=value,
        ))


class GradientStats(LayerStats):
    def __init__(self, bins=40, min=0, max=10):
        super().__init__()
        self.bins = bins
        self.min = min
        self.max = max

    def reset(self):
        self.histograms = defaultdict(list)
        self.stats = []

    def on_train_epoch_start(self, trainer, pl_module: pl.LightningModule):
        for log_prefix, module in self.log_prefix_and_target.items():
            self.registered_hooks.append(
                module.register_full_backward_hook(partial(self.log_hook, trainer, log_prefix))
            )

    def log_hook(self, trainer: pl.Trainer, log_prefix, module, inp, outp):
        outp = outp[0]
        stats = [
            ('mean', outp.mean().item()),
            ('std', outp.std(correction=0).item()),
        ]
        for stat, value in stats:
            self.stats.append(dict(
                step=trainer.global_step, epoch=trainer.current_epoch,
                layer=log_prefix, stat=stat, value=value,
        ))
        self.histograms[log_prefix].append(torch.histc(outp.abs().detach().cpu(), bins=self.bins, min=self.min, max=self.max))
