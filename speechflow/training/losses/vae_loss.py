from typing import Dict

import torch

from speechflow.training.base_loss import BaseLoss


class VAELoss(BaseLoss):
    def __init__(self, end_anneal_iter: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.end_anneal_iter = end_anneal_iter

    def _scale_scheduler_step(self, current_iter: int) -> float:
        if current_iter % self.every_iter != 0:
            return 0.0

        if current_iter < self.begin_iter:
            scale = 0.0
        elif self.begin_iter < current_iter < self.end_anneal_iter:
            scale = (
                self.scale
                * (
                    (current_iter - self.begin_iter)
                    / (self.end_anneal_iter - self.begin_iter)
                )
                ** 2
            )
        else:
            scale = self.scale

        return scale

    def forward(
        self,
        current_iter: int,
        kl_loss: torch.Tensor,
        name: str,
    ) -> Dict[str, torch.Tensor]:
        scale = self._scale_scheduler_step(current_iter)
        return {
            "constant_kl_annealing_scale": torch.tensor(scale, device=kl_loss.device),
            f"constant_{name}_unscaled": kl_loss,
            name: kl_loss * scale,
        }
