import typing as tp

import torch

__all__ = ["BaseCriterion"]


class BaseCriterion(torch.nn.Module):
    def forward(
        self,
        output: "TrainData",  # noqa: F821
        target: "TrainData",  # noqa: F821
        batch_idx: int = 0,
        global_step: int = 0,
    ) -> tp.Dict[str, torch.Tensor]:
        """
        output: model predictions
        target: target values
        batch_idx: current batch index within this epoch
        global_step: number of optimizer steps
        """
        raise NotImplementedError
