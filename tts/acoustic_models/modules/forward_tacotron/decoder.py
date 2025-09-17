import typing as tp

from torch import nn

from speechflow.utils.tensor_utils import run_rnn_on_padded_sequence
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.common.conditional_layers import (
    CONDITIONAL_TYPES,
    ConditionalLayer,
)
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import DecoderParams

__all__ = [
    "ForwardDecoder",
    "ForwardDecoderParams",
]


class ForwardDecoderParams(DecoderParams):
    # condition
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0
    condition_type: tp.Optional[CONDITIONAL_TYPES] = None

    # rnn
    rnn_type: tp.Literal["GRU", "LSTM"] = "LSTM"
    rnn_bidirectional: bool = True
    rnn_p_dropout: float = 0.1

    # projection
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"


class ForwardDecoder(Component):
    params: ForwardDecoderParams

    def __init__(self, params: ForwardDecoderParams, input_dim):
        super().__init__(params, input_dim)

        in_dim = input_dim
        inner_dim = params.decoder_inner_dim

        self.blocks = nn.ModuleList()
        for i in range(params.decoder_num_layers):
            if params.condition:
                cond_layer = ConditionalLayer(
                    params.condition_type, in_dim, params.condition_dim
                )
                self.blocks.append(cond_layer)
                in_dim = cond_layer.output_dim

            rnn_cls = getattr(nn, params.rnn_type)
            rnn = rnn_cls(
                in_dim,
                inner_dim // (params.rnn_bidirectional + 1),
                bidirectional=params.rnn_bidirectional,
                dropout=params.rnn_p_dropout,
                batch_first=True,
            )
            self.blocks.append(rnn)
            in_dim = inner_dim

        self.proj = Regression(
            inner_dim,
            params.decoder_output_dim,
            p_dropout=params.projection_p_dropout,
            activation_fn=params.projection_activation_fn,
        )
        self.gate_layer = nn.Linear(inner_dim, 1)

    @property
    def output_dim(self):
        return self.params.decoder_output_dim

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        x, x_lens, x_mask = inputs.get_content_and_mask()
        c = self.get_condition(inputs, self.params.condition)

        for block in self.blocks:
            if isinstance(block, nn.RNNBase):
                x = run_rnn_on_padded_sequence(block, x, x_lens)
            else:
                x = block(x, c, x_mask)

        y = self.proj(x)
        gate = self.gate_layer(x)

        outputs = DecoderOutput.copy_from(inputs).set_content(y, x_lens)
        outputs.gate = gate
        return outputs
