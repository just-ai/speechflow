import torch

from torch import nn

from speechflow.utils.tensor_utils import apply_mask

__all__ = ["FiLMLayer"]


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer
    Reference: https://arxiv.org/abs/1709.07871
    """

    def __init__(self, content_dim, condition_dim):
        super().__init__()
        self.act = nn.SiLU()
        self.fc = nn.Linear(condition_dim, 2 * content_dim)

    def forward(self, x, x_mask, c):
        x = apply_mask(x, x_mask)

        c = self.fc(self.act(c.squeeze(1)))
        if c.ndim == 2:
            c = c.unsqueeze(1).expand(-1, x.shape[1], -1)

        gamma, beta = torch.chunk(c, chunks=2, dim=-1)

        y = gamma * x + beta

        return apply_mask(y, x_mask)
