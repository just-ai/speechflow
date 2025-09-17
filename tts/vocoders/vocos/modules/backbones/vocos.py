from typing import Optional

import torch

from torch import nn

from speechflow.training.base_model import BaseTorchModelParams
from tts.vocoders.vocos.modules.backbones.base import Backbone
from tts.vocoders.vocos.modules.backbones.components.blocks import (
    AdaLayerNorm,
    ConvNeXtBlock,
)

__all__ = ["VocosBackbone", "VocosBackboneParams"]


class VocosBackboneParams(BaseTorchModelParams):
    input_dim: int
    inner_dim: int
    intermediate_dim: int
    num_layers: int
    layer_scale_init_value: Optional[float] = None
    condition_dim: Optional[int] = None


class VocosBackbone(Backbone):
    params: VocosBackboneParams

    """Vocos backbones module built with ConvNeXt blocks. Supports additional conditioning
    with Adaptive Layer Normalization.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        condition_dim (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.

    """

    def __init__(self, params: VocosBackboneParams):
        super().__init__(params)
        self.input_channels = params.input_dim
        self.embed = nn.Conv1d(
            params.input_dim, params.inner_dim, kernel_size=7, padding=3
        )
        self.adanorm = params.condition_dim is not None
        if params.condition_dim:
            self.norm = AdaLayerNorm(params.condition_dim, params.inner_dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(params.inner_dim, eps=1e-6)
        layer_scale_init_value = params.layer_scale_init_value or 1 / params.num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=params.inner_dim,
                    intermediate_dim=params.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    condition_dim=params.condition_dim,
                )
                for _ in range(params.num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(params.inner_dim, eps=1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        condition_emb = kwargs.get("condition_emb")
        x = self.embed(x)

        if self.adanorm:
            assert condition_emb is not None
            x = self.norm(x.transpose(1, -1), cond_emb=condition_emb)
        else:
            x = self.norm(x.transpose(1, -1))

        x = x.transpose(1, -1)
        for conv_block in self.convnext:
            x = conv_block(x, cond_emb=condition_emb)

        x = self.final_layer_norm(x.transpose(1, -1))
        return x.transpose(1, -1)
