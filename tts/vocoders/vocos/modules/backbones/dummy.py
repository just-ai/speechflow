import torch

from torch import nn

from speechflow.training.base_model import BaseTorchModelParams
from tts.vocoders.vocos.modules.backbones.base import Backbone

__all__ = ["DummyBackbone", "DummyBackboneParams"]


class DummyBackboneParams(BaseTorchModelParams):
    input_dim: int = 512
    inner_dim: int = 512


class DummyBackbone(Backbone):
    params: DummyBackboneParams

    def __init__(self, params: DummyBackboneParams):
        super().__init__(params)
        if params.input_dim != params.inner_dim:
            self.proj = nn.Conv1d(params.input_dim, params.inner_dim, 1)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.proj(x)
