import typing as tp

import torch

from torch import nn

from speechflow.utils.tensor_utils import apply_mask
from tts.acoustic_models.modules.common.blocks import ConvPrenet, Regression
from tts.acoustic_models.modules.common.conditional_layers.diffusion_transformer import (
    DiTConv,
)
from tts.acoustic_models.modules.components.encoders.cnn_encoder import (
    CNNEncoder,
    CNNEncoderParams,
)
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput

__all__ = ["DiTEncoder", "DiTEncoderParams"]


class DiTEncoderParams(CNNEncoderParams):
    filter_channels: int = 1024
    n_heads: int = 4
    kernel_size: int = 3
    p_dropout: float = 0.1

    # add long skip connection, see https://arxiv.org/pdf/2209.12152 for more details
    use_lsc: bool = False

    # condition
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0

    # projection
    use_projection: bool = True
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"

    def model_post_init(self, __context: tp.Any):
        if self.condition_dim == 0:
            raise ValueError("It is necessary to use conditioning for DiTEncoder.")

        if self.use_lsc:
            assert self.encoder_num_layers % 2 == 0


class DiTEncoder(CNNEncoder):
    params: DiTEncoderParams

    def __init__(self, params: DiTEncoderParams, input_dim):
        super().__init__(params, input_dim)

        in_dim = super().output_dim
        inner_dim = params.encoder_inner_dim

        self.prenet = ConvPrenet(
            in_channels=in_dim,
            out_channels=inner_dim,
        )

        self.blocks = nn.ModuleList()
        for i in range(params.encoder_num_layers):
            dit = DiTConv(
                inner_dim,
                params.condition_dim,
                params.filter_channels,
                params.n_heads,
                params.kernel_size,
                params.p_dropout,
            )
            self.blocks.append(dit)

        if params.use_lsc:
            self.lsc_layers = nn.ModuleList()
            for _ in range(params.encoder_num_layers // 2):
                layer = nn.Conv1d(
                    2 * inner_dim,
                    inner_dim,
                    params.kernel_size,
                    padding=params.kernel_size // 2,
                )
                self.lsc_layers.append(layer)
        else:
            self.lsc_layers = []

        if params.use_projection:
            self.proj = Regression(
                params.encoder_inner_dim,
                params.encoder_output_dim,
                p_dropout=params.projection_p_dropout,
                activation_fn=params.projection_activation_fn,
            )
        else:
            self.proj = apply_mask

    @property
    def output_dim(self):
        if self.params.use_projection:
            return self.params.encoder_output_dim
        else:
            return self.params.encoder_inner_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        inputs = super().forward_step(inputs)

        x, x_lens, x_mask = inputs.get_content_and_mask()
        c = self.get_condition(inputs, self.params.condition)

        x = self.prenet(x.transpose(1, -1)).transpose(1, -1)

        lsc_outputs = []
        n_lsc_layers = len(self.lsc_layers)

        for idx, block in enumerate(self.blocks):
            if self.params.use_lsc:
                if idx < n_lsc_layers:
                    lsc_outputs.append(x)
                else:
                    lsc_layer = self.lsc_layers[idx - n_lsc_layers]
                    x = torch.cat((x, lsc_outputs.pop()), dim=-1)
                    x = lsc_layer(x.transpose(1, -1)).transpose(1, -1)

            x = self.hook_update_content(x, x_lens, inputs)
            x = block(x, x_mask, c)

        y = self.proj(x, x_mask)

        return EncoderOutput.copy_from(inputs).set_content(y).set_hidden_state(x)
