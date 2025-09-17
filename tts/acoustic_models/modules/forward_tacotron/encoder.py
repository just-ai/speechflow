import typing as tp

import torch

from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.components import encoders
from tts.acoustic_models.modules.data_types import ComponentOutput, EncoderOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = [
    "ForwardEncoder",
    "ForwardEncoderParams",
]


class ForwardEncoderParams(EncoderParams):
    encoder_type: str = "RNNEncoder"
    encoder_params: dict = None  # type: ignore
    adaptor_encoder_type: str = None  # type: ignore
    adaptor_encoder_params: dict = None  # type: ignore
    cat_ling_feat_after_encode: bool = False

    def model_post_init(self, __context: tp.Any):
        if self.encoder_params is None:
            self.encoder_params = {}
        if self.adaptor_encoder_type is None:
            self.adaptor_encoder_type = self.encoder_type
        if self.adaptor_encoder_params is None:
            self.adaptor_encoder_params = self.encoder_params


class ForwardEncoder(Component):
    params: ForwardEncoderParams

    def __init__(self, params: ForwardEncoderParams, input_dim: int):
        super().__init__(params, input_dim)

        enc_cls = getattr(encoders, params.encoder_type)
        enc_params_cls = getattr(encoders, f"{params.encoder_type}Params")
        enc_params = enc_params_cls.init_from_parent_params(params, params.encoder_params)
        self.encoder = enc_cls(enc_params, input_dim)

        enc_cls = getattr(encoders, params.adaptor_encoder_type)
        enc_params_cls = getattr(encoders, f"{params.adaptor_encoder_type}Params")
        enc_params = enc_params_cls.init_from_parent_params(
            params, params.adaptor_encoder_params
        )
        self.adaptor_encoder = enc_cls(enc_params, input_dim)

    @property
    def output_dim(self):
        output_dim = self.encoder.output_dim
        if self.params.cat_ling_feat_after_encode:
            output_dim += self.params.token_emb_dim
        return [output_dim, output_dim]

    def encode(self, x: ComponentOutput):
        return self.encoder(x), self.adaptor_encoder(x)

    def forward_step(self, x: ComponentOutput) -> EncoderOutput:
        encoder_output, adaptor_encoder_output = self.encode(x)

        if self.params.cat_ling_feat_after_encode:
            ling_feat = x.embeddings["ling_feat"]
            encoder_content = torch.cat([encoder_output.get_content(0), ling_feat], dim=2)
            adaptor_encoder_content = torch.cat(
                [adaptor_encoder_output.get_content(0), ling_feat], dim=2
            )
            content = [encoder_content, adaptor_encoder_content]
        else:
            content = [
                encoder_output.get_content(0),
                adaptor_encoder_output.get_content(0),
            ]

        content_lens = [
            encoder_output.get_content_lengths(0),
            adaptor_encoder_output.get_content_lengths(0),
        ]
        return EncoderOutput.copy_from(x).set_content(content, content_lens)
