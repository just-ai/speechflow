import math

import torch

from torch import optim

__all__ = ["ConstLR", "WarmupInvRsqrtLR", "WarmupCosine"]


class ConstLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_max: float):
        self._lr_max = lr_max
        super().__init__(optimizer)

    def get_lr(self):
        return [self._lr_max for _ in self.optimizer.param_groups]


class WarmupInvRsqrtLR(optim.lr_scheduler._LRScheduler):
    """Increases learning rate linearly for `warmup` steps, then decays it at inverse sqrt
    rate."""

    def __init__(self, optimizer, lr_max: float, step_factor: int = 1):
        self._lr_max = lr_max
        self._step_factor = step_factor
        super().__init__(optimizer)

    def current_rate(self):
        step = self._step_count * self._step_factor
        return min(1e-6 * step, 1 / math.sqrt(step)) * self._lr_max * 100.0

    def get_lr(self):
        rate = self.current_rate()
        return [rate for _ in self.optimizer.param_groups]


class WarmupCosine(optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(
                0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
            )

        super().__init__(optimizer, lr_lambda, last_epoch)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from nemo.core.optim.lr_scheduler import CosineAnnealing, NoamAnnealing

    T = 10**6
    X = list(range(T))
    Y_torch = []
    Y_nemo = []

    _opt_torch = optim.AdamW([torch.FloatTensor(16)])
    _lr_torch = WarmupInvRsqrtLR(_opt_torch, lr_max=1.0e-3)

    _opt_nemo = optim.AdamW([torch.FloatTensor(16)])
    _lr_nemo = CosineAnnealing(_opt_nemo, max_steps=T // 2, min_lr=1.0e-6)

    for _ in X:
        _lr_torch.step()
        Y_torch.append(_lr_torch.get_lr())
        _lr_nemo.step()
        Y_nemo.append(_lr_nemo.get_lr())

    figure, axis = plt.subplots(2)

    axis[0].plot(X, Y_torch)
    axis[0].set_title(_opt_torch.__class__.__name__)

    axis[1].plot(X, Y_nemo)
    axis[1].set_title(_opt_nemo.__class__.__name__)

    plt.show()
