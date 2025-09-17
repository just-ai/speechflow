import typing as tp

from torch import nn

from speechflow.utils.tensor_utils import apply_mask, run_rnn_on_padded_sequence
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.common.conditional_layers import (
    CONDITIONAL_TYPES,
    ConditionalLayer,
)
from tts.acoustic_models.modules.components.encoders.cnn_encoder import (
    CNNEncoder,
    CNNEncoderParams,
)
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput

__all__ = ["RNNEncoder", "RNNEncoderParams"]


class RNNEncoderParams(CNNEncoderParams):
    # condition
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0
    condition_type: tp.Optional[CONDITIONAL_TYPES] = None

    # rnn
    rnn_type: tp.Literal["GRU", "LSTM"] = "LSTM"
    rnn_bidirectional: bool = True
    rnn_p_dropout: float = 0.1

    # projection
    use_projection: bool = True
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"


class RNNEncoder(CNNEncoder):
    params: RNNEncoderParams

    def __init__(self, params: RNNEncoderParams, input_dim):
        super().__init__(params, input_dim)

        in_dim = super().output_dim

        self.blocks = nn.ModuleList()
        for i in range(params.encoder_num_layers):
            if params.condition:
                cond_layer = ConditionalLayer(
                    params.condition_type, in_dim, params.condition_dim
                )
                self.blocks.append(cond_layer)
                in_dim = cond_layer.output_dim

            rnn_cls = getattr(nn, params.rnn_type)
            rnn = rnn_cls(
                in_dim,
                params.encoder_inner_dim // (params.rnn_bidirectional + 1),
                bidirectional=params.rnn_bidirectional,
                dropout=params.rnn_p_dropout,
                batch_first=True,
            )
            self.blocks.append(rnn)
            in_dim = params.encoder_inner_dim

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

        for block in self.blocks:
            if isinstance(block, nn.RNNBase):
                x = self.hook_update_content(x, x_lens, inputs)
                x = run_rnn_on_padded_sequence(block, x, x_lens)
            else:
                x = block(x, c, x_mask)

        y = self.proj(x, x_mask)

        return EncoderOutput.copy_from(inputs).set_content(y).set_hidden_state(x)
