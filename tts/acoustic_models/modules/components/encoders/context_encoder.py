import typing as tp

import torch

from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentOutput, EncoderOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = [
    "ContextEncoder",
    "ContextEncoderParams",
]


class ContextEncoderParams(EncoderParams):
    encoder_type: tp.Union[str, tp.List[str]] = "RNNEncoder"
    encoder_params: tp.Union[dict, tp.List[dict]] = None  # type: ignore

    def model_post_init(self, __context: tp.Any):
        if isinstance(self.encoder_type, str):
            self.encoder_type = [self.encoder_type]
        if isinstance(self.encoder_params, dict):
            self.encoder_params = [self.encoder_params]


class ContextEncoder(Component):
    params: ContextEncoderParams

    def __init__(
        self, params: ContextEncoderParams, input_dim: tp.Union[int, tp.List[int]]
    ):
        super().__init__(params, input_dim)
        from tts.acoustic_models.modules import TTS_ENCODERS

        if isinstance(input_dim, int):
            input_dim = [input_dim] * len(params.encoder_type)

        self.encoders = torch.nn.ModuleList()
        for enc_type, enc_params, in_dim in zip(
            params.encoder_type, params.encoder_params, input_dim
        ):
            enc_cls, enc_params_cls = TTS_ENCODERS[enc_type]
            params.encoder_type = enc_type
            params.encoder_params = enc_params
            enc_params = enc_params_cls.init_from_parent_params(params, enc_params)
            self.encoders.append(enc_cls(enc_params, in_dim))

    @property
    def output_dim(self):
        return [enc.output_dim for enc in self.encoders]

    def forward_step(self, x: ComponentOutput) -> EncoderOutput:
        if len(x.content) == 1:
            x_content = x.content * len(self.encoders)
            x_content_lengths = x.content_lengths * len(self.encoders)
        else:
            x_content = x.content
            x_content_lengths = x.content_lengths
            assert len(x_content) == len(self.encoders)

        content = []
        content_lengths = []
        for enc, ctx, ctx_lens in zip(self.encoders, x_content, x_content_lengths):
            x.content = ctx
            x.content_lengths = ctx_lens
            result = enc(x)
            content.append(result.content)
            content_lengths.append(result.content_lengths)

        return EncoderOutput.copy_from(x).set_content(content, content_lengths)
