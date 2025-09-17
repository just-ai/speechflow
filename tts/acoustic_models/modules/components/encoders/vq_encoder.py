import typing as tp

import torch

from pydantic import Field
from torch import nn
from vector_quantize_pytorch import ResidualFSQ, ResidualLFQ

from tts.acoustic_models.modules.common.layers import Conv
from tts.acoustic_models.modules.common.length_regulators import SoftLengthRegulator
from tts.acoustic_models.modules.common.vector_quantizer import (
    VectorQuantizer,
    VectorQuantizerOutput,
)
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import (
    ComponentInput,
    ComponentOutput,
    EncoderOutput,
)
from tts.acoustic_models.modules.params import EncoderParams

__all__ = [
    "VQEncoder",
    "VQEncoderParams",
    "VQEncoderWithClassificationAdaptor",
    "VQEncoderWithClassificationAdaptorParams",
]


class VQEncoderParams(EncoderParams):
    vq_type: tp.Literal["vq", "rvq", "rfsq", "rlfq"] = "vq"
    vq_encoder_type: str = "RNNEncoder"
    vq_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    vq_num_quantizers: int = 1
    vq_codebook_size: int = 256
    vq_by_phonemes: bool = False


class VQEncoder(Component):
    params: VQEncoderParams

    def __init__(self, params: VQEncoderParams, input_dim):
        super().__init__(params, input_dim)

        from tts.acoustic_models.modules import TTS_ENCODERS

        base_enc_cls, base_enc_params_cls = TTS_ENCODERS[params.vq_encoder_type]
        base_enc_params = base_enc_params_cls.init_from_parent_params(
            params, params.vq_encoder_params
        )
        self.encoder = base_enc_cls(base_enc_params, input_dim)

        self.pre_vq_conv = torch.nn.Conv1d(
            in_channels=self.encoder.output_dim,
            out_channels=self.encoder.output_dim,
            kernel_size=3,
            padding=1,
        )

        k = self.params.vq_num_quantizers
        levels = [int(self.params.vq_codebook_size ** (1.0 / k))] * k

        if params.vq_type == "vq":
            self.vq = VectorQuantizer(
                embedding_dim=self.encoder.output_dim,
                codebook_size=self.params.vq_codebook_size,
            )
            self.rvq = [self.vq]
        elif params.vq_type == "rvq":
            self.rvq = nn.ModuleList()
            for _ in range(self.params.vq_num_quantizers):
                self.rvq.append(
                    VectorQuantizer(
                        embedding_dim=self.encoder.output_dim,
                        codebook_size=self.params.vq_codebook_size,
                    )
                )
        elif params.vq_type == "rfsq":
            self.rfsq = ResidualFSQ(
                dim=self.encoder.output_dim,
                levels=levels,
                num_quantizers=self.params.vq_num_quantizers,
            )
        elif params.vq_type == "rlfq":
            self.rlfq = ResidualLFQ(
                dim=self.encoder.output_dim,
                num_quantizers=self.params.vq_num_quantizers,
                codebook_size=self.params.vq_codebook_size,
            )
        else:
            raise NotImplementedError(f"'{params.vq_type}' is not implemented")

        if params.vq_by_phonemes:
            self.length_regulator = SoftLengthRegulator()

    @property
    def output_dim(self):
        return self.encoder.output_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        x, x_lens, x_mask = inputs.get_content_and_mask()

        x = ComponentInput.copy_from(inputs).set_content(x, x_lens)
        output: ComponentOutput = self.encoder(x)
        y = output.content

        if self.params.vq_by_phonemes:
            y, _ = self.length_regulator(
                y,
                inputs.model_inputs.invert_durations,
                inputs.model_inputs.transcription.shape[1],
            )

        z = self.pre_vq_conv(y.transpose(2, 1))
        indices = None

        if self.params.vq_type == "vq" or self.params.vq_type == "rvq":
            residual = z
            quantized_out = 0.0
            for idx, layer in enumerate(self.rvq):
                vq_output: VectorQuantizerOutput = layer(residual)  # type: ignore
                assert not isinstance(vq_output.content, list)

                for k, v in vq_output.additional_content.items():
                    inputs.additional_content[
                        f"{k}_{self.params.vq_type}{idx}_encoder_{self.id}"
                    ] = v

                for k, v in vq_output.additional_losses.items():
                    inputs.additional_losses[
                        f"{k}_{self.params.vq_type}{idx}_encoder_{self.id}"
                    ] = v

                quantized = vq_output.content
                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

            z = quantized_out.transpose(1, 2)
        elif self.params.vq_type == "rfsq":
            z, indices = self.rfsq(z.transpose(1, -1))
        elif self.params.vq_type == "rlfq":
            z, indices, commit_loss = self.rlfq(z.transpose(1, -1))
            inputs.additional_losses[
                f"{self.params.vq_type}_encoder_{self.id}"
            ] = commit_loss.sum()

        inputs.additional_content["vq_latents"] = y
        inputs.additional_content["vq_codes"] = indices
        inputs.additional_content["vq_z"] = z

        return EncoderOutput.copy_from(inputs).set_content(z)


class VQEncoderWithClassificationAdaptorParams(VQEncoderParams):
    n_convolutions: int = 3
    kernel_size: int = 5


class VQEncoderWithClassificationAdaptor(VQEncoder):
    params: VQEncoderWithClassificationAdaptorParams

    def __init__(self, params: VQEncoderWithClassificationAdaptorParams, input_dim: int):
        super().__init__(params, input_dim)

        convolutions = []
        for _ in range(params.n_convolutions):
            conv_layer = nn.Sequential(
                Conv(
                    self.output_dim,
                    self.output_dim,
                    kernel_size=params.kernel_size,
                    stride=1,
                    padding=int((params.kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(self.output_dim),
                nn.SiLU(),
            )
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

        self.components_output_dim["adaptor_context"] = lambda: self.output_dim

    def forward_step(self, x: ComponentInput) -> EncoderOutput:
        result: EncoderOutput = super().forward_step(x)

        content = result.content
        adaptor_context = result.additional_content.setdefault(
            f"adaptor_context_{self.id}", []
        )

        if self.training:
            ctx = content.transpose(2, 1)
            for conv in self.convolutions:
                ctx = conv(ctx)

            adaptor_context.append(ctx.transpose(2, 1))

        return result
