import typing as tp

import numpy as np
import torch

from torch import nn

from nlp.prosody_prediction.data_types import (
    ProsodyPredictionOutput,
    ProsodyPredictionTarget,
)

__all__ = ["ProsodyPredictionLoss"]


class ProsodyPredictionLoss(nn.Module):
    def __init__(self, names: tp.List[tp.Literal["binary", "category"]]):
        super().__init__()
        self.names = names
        self.loss_fn = nn.ModuleDict()
        for name in self.names:
            self.loss_fn[name] = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        output: ProsodyPredictionOutput,
        target: ProsodyPredictionTarget,
        batch_idx: int = 0,
        global_step: int = 0,
    ) -> tp.Dict[str, torch.Tensor]:
        total_loss = {}

        for name in self.names:
            p = getattr(output, name)
            t = getattr(target, name)
            total_loss[name] = self.loss_fn[name](p.reshape(-1, p.shape[-1]), t.flatten())

        return total_loss

    def set_weights(self, dataset, n_classes):
        """Computes weights for loss if target is unbalanced."""
        weights = self.compute_idf_weights(dataset, n_classes)
        self.loss_fn = nn.ModuleDict()
        for name in self.names:
            self.loss_fn[name] = nn.CrossEntropyLoss(weight=weights[name])

    @staticmethod
    def compute_idf_weights(dataset, n_classes):
        """Weights are idfs normalized by softmax."""
        N = len(dataset)
        dfs = {"binary": np.zeros(2), "category": np.zeros(n_classes)}

        for batch_idx in range(N):
            batch = dataset.next_batch()
            tags_category = batch.collated_samples.category.reshape(-1)
            tags_binary = batch.collated_samples.binary.reshape(-1)
            for tag_c, tag_b in zip(tags_category, tags_binary):
                tag_c = tag_c.item()
                tag_b = tag_b.item()
                if tag_b != -100:
                    dfs["binary"][tag_b] += 1
                if tag_c != -100:
                    dfs["category"][tag_c] += 1

        idf = {
            name: [np.log(sum(dfs[name]) / (df + 1)) for df in dfs[name]] for name in dfs
        }
        return {
            name: torch.Tensor(np.exp(idf[name]) / sum(np.exp(idf[name]))) for name in idf
        }
