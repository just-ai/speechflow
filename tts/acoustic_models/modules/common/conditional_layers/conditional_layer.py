import typing as tp

import torch

from torch import nn

from speechflow.utils.tensor_utils import apply_mask
from tts.acoustic_models.modules.common.conditional_layers.ada_layer_norm import (
    AdaLayerNorm,
)
from tts.acoustic_models.modules.common.conditional_layers.diffusion_transformer import (
    DiTConv,
)
from tts.acoustic_models.modules.common.conditional_layers.film_layer import FiLMLayer

CONDITIONAL_TYPES = tp.Literal["add", "cat", "FiLM", "AdaNorm", "DiT"]

__all__ = ["ConditionalLayer", "CONDITIONAL_TYPES"]


class ConditionalLayer(nn.Module):
    def __init__(
        self, condition_type: tp.Optional[CONDITIONAL_TYPES], content_dim, condition_dim
    ):
        super().__init__()
        self.condition_type = condition_type
        self.output_dim = content_dim

        if condition_type is None:
            return
        elif condition_type == "add":
            self.module = nn.Sequential(nn.Linear(condition_dim, content_dim), nn.SiLU())
        elif condition_type == "cat":
            self.module = None
            self.output_dim += condition_dim
        elif condition_type == "FiLM":
            self.module = FiLMLayer(content_dim, condition_dim)
        elif condition_type == "AdaNorm":
            self.module = AdaLayerNorm(content_dim, condition_dim)
        elif condition_type == "DiT":
            self.module = DiTConv(content_dim, condition_dim)
        else:
            raise NotImplementedError

    def forward(self, x, c, x_mask=None):
        """
        Args:
            x : [batch_size, channel] or [batch_size, time, channel]
            c : [batch_size, channel] or [batch_size, time, channel]
            x_mask : [batch_size, time]
        return the same shape as x
        """

        if self.condition_type is None:
            return x

        if x.ndim == 2:
            x = x.unsqueeze(1)

        if x_mask is None:
            x_mask = torch.ones(x.size(0), x.size(1)).to(device=x.device, dtype=x.dtype)

        if c.ndim == 3:
            c = c.squeeze(1)

        if c.ndim == 2 and self.condition_type == "cat":
            c = apply_mask(c.unsqueeze(1).expand(-1, x.shape[1], -1), x_mask)

        if self.condition_type == "add":
            c = self.module(c)
            if c.ndim == 2:
                c = c.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = x + apply_mask(c, x_mask)
        elif self.condition_type == "cat":
            x = torch.cat([x, c], dim=-1)
        else:
            x = self.module(x, x_mask, c)

        return x.squeeze(1)


if __name__ == "__main__":
    _b = 4
    _t = 100
    _ctx_dim = 256
    _c_dim = 64
    _x1 = torch.zeros((_b, _ctx_dim))
    _x2 = torch.zeros((_b, _t, _ctx_dim))
    _c1 = torch.ones((_b, _c_dim))
    _c2 = torch.ones((_b, _t, _c_dim))
    _x_mask = torch.zeros((_b, _t))

    for _cond_type in tp.get_args(CONDITIONAL_TYPES):
        _cond_layer = ConditionalLayer(_cond_type, _ctx_dim, _c_dim)

        for _x in [_x1, _x2]:
            for _c in [_c1, _c2]:
                if _x.ndim == 2 and _c.ndim == 3:
                    continue

                try:
                    _y = _cond_layer(_x, _c, _x_mask if _x.ndim == 3 else None)
                    print(_cond_type, _y.shape)
                except Exception as e:
                    print(e)
