from torch import nn

from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.components.style_encoders.style_encoder import (
    StyleEncoderParams,
)
from tts.acoustic_models.modules.data_types import MODEL_INPUT_TYPE

__all__ = [
    "SimpleStyle",
    "SimpleStyleParams",
]


class SimpleStyleParams(StyleEncoderParams):
    pass


class SimpleStyle(Component):
    params: SimpleStyleParams

    def __init__(self, params: SimpleStyleParams, input_dim: int):
        super().__init__(params, input_dim)
        self.proj = nn.Sequential(nn.Linear(input_dim, params.style_emb_dim), nn.Tanh())

    @property
    def output_dim(self):
        return self.params.style_emb_dim

    def encode(self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs):
        return self.proj(x.squeeze(-1))

    def forward_step(self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs):
        style_emb = self.encode(x, x_lengths, model_inputs, **kwargs)
        return style_emb, {}, {}
