import typing as tp

import torch

from torch import nn

from speechflow.utils.tensor_utils import apply_mask, run_rnn_on_padded_sequence
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.common.conditional_layers import (
    CONDITIONAL_TYPES,
    ConditionalLayer,
)
from tts.acoustic_models.modules.common.layers import Conv, LearnableSwish
from tts.acoustic_models.modules.components.encoders.cnn_encoder import (
    CNNEncoder,
    CNNEncoderParams,
)
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput

__all__ = ["VarianceEncoder", "VarianceEncoderParams"]


class VarianceEncoderParams(CNNEncoderParams):
    # convolution block
    conv_kernel_sizes: tp.Tuple[int, ...] = (3, 7, 13, 3)
    conv_p_dropout: float = 0.1

    # condition
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0
    condition_type: tp.Optional[CONDITIONAL_TYPES] = None

    # rnn
    use_rnn: bool = True
    rnn_type: tp.Literal["GRU", "LSTM"] = "LSTM"
    rnn_bidirectional: bool = True
    rnn_p_dropout: float = 0.1

    # projection
    use_projection: bool = True
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"


class VarianceEncoder(CNNEncoder):
    params: VarianceEncoderParams

    def __init__(self, params: VarianceEncoderParams, input_dim):
        super().__init__(params, input_dim)

        in_dim = super().output_dim
        inner_dim = params.encoder_inner_dim
        first_convs_kernel_sizes = params.conv_kernel_sizes[:-1]
        second_convs_kernel_sizes = params.conv_kernel_sizes[-1]

        if params.condition:
            self.cond_layer = ConditionalLayer(
                params.condition_type, in_dim, params.condition_dim
            )

        self.first_convs = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(
                        in_dim,
                        inner_dim,
                        kernel_size=k,
                        padding=(k - 1) // 2,
                        w_init_gain=None,
                        swap_channel_dim=True,
                    ),
                    LearnableSwish(),
                    nn.LayerNorm(inner_dim),
                    nn.Dropout(params.conv_p_dropout),
                )
                for k in first_convs_kernel_sizes
            ]
        )

        self.second_conv = nn.Sequential(
            Conv(
                inner_dim * len(first_convs_kernel_sizes),
                inner_dim,
                kernel_size=second_convs_kernel_sizes,
                padding=(second_convs_kernel_sizes - 1) // 2,
                w_init_gain=None,
                swap_channel_dim=True,
            ),
            LearnableSwish(),
            nn.LayerNorm(inner_dim),
            nn.Dropout(params.conv_p_dropout),
        )

        if params.use_rnn:
            rnn_cls = getattr(nn, params.rnn_type)
            self.rnn = rnn_cls(
                inner_dim,
                inner_dim // (params.rnn_bidirectional + 1),
                num_layers=params.encoder_num_layers,
                bidirectional=params.rnn_bidirectional,
                dropout=params.rnn_p_dropout,
                batch_first=True,
            )
        else:
            self.rnn = None

        if params.use_projection:
            self.proj = Regression(
                inner_dim,
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

        if self.params.condition:
            c = self.get_condition(inputs, self.params.condition)
            x = self.cond_layer(x, c, x_mask)

        after_first_conv = []
        for conv_layer in self.first_convs:
            after_first_conv.append(conv_layer(x))

        concatenated = torch.cat(after_first_conv, dim=2)
        after_second_conv = self.second_conv(concatenated)

        for conv_1 in after_first_conv:
            after_second_conv += conv_1

        if self.params.use_rnn:
            x = run_rnn_on_padded_sequence(self.rnn, after_second_conv, x_lens)
        else:
            x = after_second_conv

        y = self.proj(x, x_mask)

        return EncoderOutput.copy_from(inputs).set_content(y).set_hidden_state(x)
