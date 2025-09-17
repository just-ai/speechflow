import numba
import numpy as np
import torch

from speechflow.training.base_loss import BaseLoss, apply_interval_check

__all__ = ["BaseAttention", "GuidedAttention"]


class BaseAttention(BaseLoss):
    pass


class GuidedAttention(BaseAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.g = kwargs.get("g", 0.2)

    @apply_interval_check
    def forward(
        self,
        current_iter: int,
        attention: torch.Tensor,
        seq_lens: torch.Tensor = None,
        spec_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        tensor_w = []
        batch_size, n_max, m_max = attention.shape
        for idx in range(batch_size):
            m = seq_lens[idx] if seq_lens is not None else m_max
            n = spec_lens[idx] if spec_lens is not None else n_max
            w = GuidedAttention._mask((n_max, m_max, n, m), self.g)
            tensor_w.append(torch.as_tensor(w).unsqueeze(0))

        multiplier = max(
            0.0, 1.0 - (current_iter - self.begin_iter) / max(self.max_iter, 1)
        )
        return (
            multiplier
            * (torch.cat(tensor_w, dim=0).to(attention.device) * attention).mean()
        )

    @staticmethod
    @numba.njit
    def _mask(shape, g):
        """JIT-optimized version, speedup varies from ~50x for 10x10 matrices to ~1000x
        for 1000x1000 matrices."""
        n_max, m_max, n, m = shape
        w = np.zeros((n_max, m_max), dtype=np.float32)
        for t_pos in range(n):
            for n_pos in range(m):
                w[t_pos, n_pos] = 1.0 - np.exp(
                    -((t_pos / float(n) - n_pos / float(m)) ** 2) / (2 * g * g)
                )
        return w
