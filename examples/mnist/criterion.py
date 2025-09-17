import typing as tp

import torch
import torch.nn.functional as F

from examples.mnist.data_types import MNISTForwardOutput, MNISTTarget
from speechflow.training import BaseCriterion

__all__ = ["MNISTLoss"]


class MNISTLoss(BaseCriterion):
    def forward(
        self,
        output: MNISTForwardOutput,
        target: MNISTTarget,
        batch_idx: int = 0,
        global_step: int = 0,
    ) -> tp.Dict[str, torch.Tensor]:
        output = F.log_softmax(output.logits, dim=1)
        loss = F.nll_loss(output, target.label)
        return {"NLLLoss": loss}
