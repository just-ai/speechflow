import math
import typing as tp

import torch

from speechflow.training.base_loss import BaseLoss, apply_interval_check, multi_output

try:
    from pytorch_msssim import MS_SSIM
except Exception:
    assert False, "Module pytorch-msssim not found (pip install pytorch-msssim)!"

from speechflow.training.losses.helpers import get_loss_from_name
from speechflow.utils.init import init_class_from_config

__all__ = ["BaseSpectral", "Spectral", "DiffSpectral", "SSIM"]


class BaseSpectral(BaseLoss):
    @staticmethod
    def _set_mask(
        tensor_to_mask: torch.Tensor, mask: tp.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            tensor_to_mask = tensor_to_mask.masked_select(mask)
        return tensor_to_mask


class Spectral(BaseSpectral):
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


class DiffSpectral(Spectral):
    @apply_interval_check
    @multi_output
    def forward(
        self,
        global_step: int,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        target = target[:, 1:, :] - target[:, 0:-1, :]
        output = output[:, 1:, :] - output[:, 0:-1, :]

        if mask is not None:
            target = self._set_mask(target, mask[:, 1:, :])
            output = self._set_mask(output, mask[:, 1:, :])

        return self.scale * self.loss_fn(output, target)


class SSIM(BaseSpectral):
    def __init__(self, loss_fn: str, **kwargs):
        super().__init__(loss_fn=loss_fn, **kwargs)
        self.min_value = float(kwargs.get("min_value"))
        self.max_value = float(kwargs.get("max_value"))
        self.norm = self.max_value - self.min_value
        self.weights = torch.FloatTensor([0.1, 0.2, 0.4])
        if loss_fn == "msssim":
            params = {
                "data_range": 1,
                "size_average": True,
                "channel": 1,
                "weights": self.weights,
            }
            self.loss_fn = init_class_from_config(MS_SSIM, params)()
        else:
            raise NotImplementedError(f"type {loss_fn} is not implemented")

    @apply_interval_check
    @multi_output
    def forward(
        self,
        global_step: int,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        t_shape = target.shape
        spec_len = 3 * math.floor(t_shape[-2] // 3)
        t_shape_1 = [t_shape[0], t_shape[1], spec_len // 3, 240]
        t_shape_2 = [t_shape[0] * t_shape[1], 1, spec_len // 3, 240]

        if mask is not None:
            mask = ~mask
            target = torch.masked_fill(target, mask, 0.0)
            output = torch.masked_fill(output, mask, 0.0)

        target = target[:, :, :spec_len, :].view(t_shape_1).view(t_shape_2)
        output = output[:, :, :spec_len, :].view(t_shape_1).view(t_shape_2)

        if target.shape[-2] > 160 and target.shape[-1] > 160:
            output_norm = (output + self.min_value) / self.norm
            target_norm = (target + self.min_value) / self.norm
            return self.scale * (
                1 - self.loss_fn(output_norm.float(), target_norm.float())
            )
        else:
            return torch.tensor(0.0, device=output.device)
