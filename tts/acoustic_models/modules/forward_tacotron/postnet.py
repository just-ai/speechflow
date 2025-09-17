from torch import nn

from tts.acoustic_models.modules.common.blocks import CBHG, Regression
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, PostnetOutput
from tts.acoustic_models.modules.params import PostnetParams

__all__ = ["ForwardPostnet", "ForwardPostnetParams"]


class ForwardPostnetParams(PostnetParams):
    highways: int = 5
    n_convolutions: int = 5
    kernel_size: int = 3

    # projection
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"


class ForwardPostnet(Component):
    params: ForwardPostnetParams

    def __init__(self, params: ForwardPostnetParams, input_dim: int):
        super().__init__(params, input_dim)
        self.cbhg = CBHG(
            in_dim=self.input_dim,
            out_dim=params.postnet_inner_dim,
            conv_banks_num=params.n_convolutions,
            highways_num=params.highways,
            kernel_size=params.kernel_size,
            rnn_bidirectional=True,
            rnn_dim=self.input_dim,
        )
        self.proj = Regression(
            params.postnet_inner_dim,
            params.postnet_output_dim,
            p_dropout=params.projection_p_dropout,
            activation_fn=params.projection_activation_fn,
        )

    @property
    def output_dim(self):
        return self.params.postnet_output_dim

    def forward_step(self, inputs: DecoderOutput) -> PostnetOutput:  # type: ignore
        content = inputs.get_content()
        x = content[-1]
        x = self.cbhg(x)
        x_out = self.proj(x)
        return PostnetOutput.copy_from(inputs).set_content(x_out)
