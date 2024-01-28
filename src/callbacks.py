import re
from typing import Dict, Optional
import torch
from torch import Tensor
import lightning.pytorch as pl


class PlotCB(pl.callbacks.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log('train_loss', outputs['loss'], prog_bar=True, batch_size=batch.t.numel(), on_epoch=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log('val_loss', outputs, prog_bar=True, batch_size=batch.t.numel(), on_epoch=True)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    @classmethod
    def _format_checkpoint_name(
        cls,
        filename: Optional[str],
        metrics: Dict[str, Tensor],
        prefix: str = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        """
        Copied from here: https://github.com/Lightning-AI/pytorch-lightning/blob/2.0.9.post0/src/lightning/pytorch/callbacks/model_checkpoint.py#L497

        Only changed one line for what the checkpoint filename is.
        """
        if not filename:
            # filename is not set, use default name
            filename = "{epoch}" + cls.CHECKPOINT_JOIN_CHAR + "{step}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)

        # sort keys from longest to shortest to avoid replacing substring
        # eg: if keys are "epoch" and "epoch_test", the latter must be replaced first
        groups = sorted(groups, key=lambda x: len(x), reverse=True)

        for group in groups:
            name = group[1:]

            if auto_insert_metric_name:
                # CHANGED THIS LINE ONLY
                filename = filename.replace(group, name + "_{" + name)

            # support for dots: https://stackoverflow.com/a/7934969
            filename = filename.replace(group, f"{{0[{name}]")

            if name not in metrics:
                metrics[name] = torch.tensor(0)
        filename = filename.format(metrics)

        if prefix:
            filename = cls.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename
