import typing as tp

import torch

from torch.nn import functional as F

from speechflow.training.base_loss import BaseLoss, apply_interval_check
from speechflow.training.losses.helpers import get_loss_from_name

__all__ = ["BaseDuration", "DurationLoss", "FewAttentionsLoss"]


class BaseDuration(BaseLoss):
    def __init__(self, loss_fn: str, **kwargs):
        super().__init__(**kwargs)

        if loss_fn in ["l1", "smooth_l1", "l2", "CE"]:
            self.loss_fn = get_loss_from_name(loss_fn)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] {loss_fn} loss is not supported"
            )

    @staticmethod
    def _set_mask(
        tensor_to_mask: torch.Tensor, mask: tp.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            tensor_to_mask = tensor_to_mask.masked_select(mask)
        return tensor_to_mask


class DurationLoss(BaseDuration):
    @apply_interval_check
    def forward(
        self,
        global_step: int,
        predicted_durations: torch.Tensor,
        gt_durations: torch.Tensor,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gt_durations = self._set_mask(gt_durations, mask)
        predicted_durations = self._set_mask(predicted_durations, mask)
        return self.scale * self.loss_fn(gt_durations, predicted_durations)


class FewAttentionsLoss(BaseDuration):
    @apply_interval_check
    def forward(
        self,
        global_step: int,
        predicted_durations: torch.Tensor,
        gt_durations: torch.Tensor,
        alignment: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = gt_durations.size()[0]
        gt_attention = torch.zeros_like(alignment)
        gt_durations_cs = gt_durations.cumsum(dim=1).int()
        gt_durations = gt_durations.int()

        rand_scale = 0.0
        for i in range(batch_size):
            for j in range(gt_durations[i].size()[0]):
                gt_attention[
                    :,
                    i,
                    gt_durations_cs[i, j] - gt_durations[i, j] : gt_durations_cs[i, j],
                    j,
                ] = 1
        if self.loss_fn == F.cross_entropy:
            return -torch.mul(
                (alignment + 1e-9 + torch.rand_like(alignment) * rand_scale).log(),
                gt_attention,
            ).mean()
        else:
            return self.scale * self.loss_fn(alignment, gt_attention)
