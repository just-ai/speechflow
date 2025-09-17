import typing as tp
import logging

import torch

from speechflow.logging import trace
from speechflow.training.base_loss import BaseLoss, apply_interval_check
from speechflow.training.losses.dilate.dilate_loss import dilate_loss
from speechflow.training.losses.helpers import get_loss_from_name

try:
    from pytorch_msssim import MS_SSIM, SSIM, ms_ssim, ssim
except ImportError:
    assert False, "Module pytorch-msssim not found (pip install pytorch-msssim)!"

__all__ = [
    "BaseLoss1D",
    "Gate",
    "InverseSpeakerLoss",
    "MLELoss",
]

LOGGER = logging.getLogger("root")


class BaseLoss1D(BaseLoss):
    def __init__(self, loss_fn: str, **kwargs):
        super().__init__(**kwargs)

        if loss_fn in ["l1", "smooth_l1", "l2", "BCEl"]:
            self.loss_fn = get_loss_from_name(loss_fn)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] {loss_fn} loss is not supported"
            )

        self.log_scale: bool = kwargs.get("log_scale", False)
        self.use_diff: bool = kwargs.get("use_diff", False)
        self.use_dilate: bool = kwargs.get("use_dilate", False)
        self.dilate_start: int = kwargs.get("dilate_start", 0)
        self.dilate_every: int = kwargs.get("dilate_every", 1)

    def _set_mask(
        self, tensor_to_mask: torch.Tensor, mask: tp.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            if tensor_to_mask.ndim == 2:
                tensor_to_mask = tensor_to_mask.masked_select(mask)
            elif tensor_to_mask.ndim == 3:
                try:
                    tensor_to_mask = tensor_to_mask.masked_select(mask.unsqueeze(-1))
                except Exception as e:
                    LOGGER.error(
                        trace(
                            self,
                            e,
                            message=f"shape A: {tensor_to_mask.shape}, shape B: {mask.shape}",
                        )
                    )
            else:
                raise RuntimeError(f"Shape out {tensor_to_mask.ndim} is not supported")

        return tensor_to_mask

    @apply_interval_check
    def forward(
        self,
        global_step: int,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = target.float()

        if self.log_scale:
            target = torch.log1p(target)

        mask_target = self._set_mask(target, mask)
        mask_output = self._set_mask(output, mask)

        loss = self.loss_fn(mask_output, mask_target)
        total_loss = self.scale * loss

        if self.use_diff:
            diff_target = target[:, 1:] - target[:, 0:-1]
            diff_output = output[:, 1:] - output[:, 0:-1]
            loss = self.loss_fn(diff_output, diff_target)
            total_loss += self.scale * loss

        if (
            self.use_dilate
            and self.training
            and global_step + 1 > self.dilate_start
            and global_step % self.dilate_every == 0
        ):
            loss, _, _ = dilate_loss(output.unsqueeze(-1), target.unsqueeze(-1))
            total_loss += self.scale * loss

        return total_loss


class Gate(BaseLoss1D):
    def __init__(self, **kwargs):
        kwargs["log_scale"] = False
        super().__init__(**kwargs)


class InverseSpeakerLoss(BaseLoss):
    pass


class MLELoss(BaseLoss):
    pass
