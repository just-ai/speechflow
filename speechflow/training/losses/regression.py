import typing as tp

import torch

from speechflow.training.base_loss import BaseLoss, apply_interval_check, multi_output

try:
    from pytorch_msssim import MS_SSIM
except Exception:
    assert False, "Module pytorch-msssim not found (pip install pytorch-msssim)!"

from speechflow.training.losses.helpers import get_loss_from_name

__all__ = ["BaseRegression", "Regression"]


class BaseRegression(BaseLoss):
    @staticmethod
    def _set_mask(
        tensor_to_mask: torch.Tensor, mask: tp.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            tensor_to_mask = tensor_to_mask.masked_select(mask)
        return tensor_to_mask


class Regression(BaseRegression):
    def __init__(self, loss_fn: str, **kwargs):
        super().__init__(**kwargs)

        if loss_fn in ["l1", "smooth_l1", "l2"]:
            self.loss_fn = get_loss_from_name(loss_fn)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] {loss_fn} loss is not supported"
            )

    @apply_interval_check
    @multi_output
    def forward(
        self,
        global_step: int,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self._set_mask(target, mask)
        output = self._set_mask(output, mask)
        return self.scale * self.loss_fn(output, target)
