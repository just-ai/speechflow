import torch

from torch import nn

from speechflow.utils.tensor_utils import apply_mask

__all__ = ["AdaLayerNorm"]


class AdaLayerNorm(nn.Module):
    def __init__(self, content_dim, condition_dim, eps: float = 1.0e-5):
        super().__init__()
        self.content_dim = content_dim
        self.act = nn.SiLU()
        self.fc = nn.Linear(condition_dim, 2 * content_dim)
        self.norm = nn.LayerNorm(content_dim, eps=eps)

    def forward(self, x, x_mask, c):
        x = apply_mask(x, x_mask)

        c = self.fc(self.act(c.squeeze(1)))
        if c.ndim == 2:
            c = c.unsqueeze(1).expand(-1, x.shape[1], -1)

        gamma, beta = torch.chunk(c, chunks=2, dim=-1)

        y = (1 + gamma) * self.norm(x) + beta

        return apply_mask(y, x_mask)
