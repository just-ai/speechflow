import torch

from torch import nn

__all__ = ["BaseLoss", "apply_interval_check", "multi_output"]


def apply_interval_check(function_to_decorate):
    def _check(self, *args):
        current_iter: int = args[0]
        if (
            self.begin_iter <= current_iter < self.end_iter
            and (current_iter + 1) % self.every_iter == 0
        ):
            return function_to_decorate(self, *args)
        else:
            return torch.tensor(0.0, device=args[1].device)

    return _check


class BaseLoss(nn.Module):
    def __init__(
        self,
        begin_iter: int = 0,
        end_iter: int = 0,
        max_iter: int = 1_000_000_000_000,
        every_iter: int = 1,
        scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.begin_iter = begin_iter
        self.end_iter = end_iter
        self.max_iter = max_iter
        self.every_iter = every_iter
        self.scale = scale
        if not self.end_iter:
            self.end_iter = self.begin_iter + self.max_iter

    @apply_interval_check
    def precomputed_forward(self, current_iter, loss_value):
        _ = current_iter
        return self.scale * loss_value


def multi_output(function_to_decorate):
    def _loss_apply(self, *args):
        current_iter: int = args[0]
        output: torch.Tensor = args[1]
        target: torch.Tensor = args[2]
        mask: torch.Tensor = args[3]
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        if output.ndim != target.ndim:
            target = target.unsqueeze(0).expand(output.shape)
            mask = mask.unsqueeze(0).expand(output.shape)
            return function_to_decorate(
                self, current_iter, output, target, mask, *args[4:]
            )
        else:
            return function_to_decorate(self, *args)

    return _loss_apply
