import typing as tp

from torchaudio.models import Conformer

from speechflow.utils.tensor_utils import apply_mask
from tts.acoustic_models.modules.common.blocks import ConvPrenet, Regression
from tts.acoustic_models.modules.common.conditional_layers import (
    CONDITIONAL_TYPES,
    ConditionalLayer,
)
from tts.acoustic_models.modules.common.pos_encoders import PE_TYPES, PositionalEncodings
from tts.acoustic_models.modules.components.encoders.cnn_encoder import (
    CNNEncoder,
    CNNEncoderParams,
)
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput

__all__ = ["ConformerEncoder", "ConformerEncoderParams"]


class ConformerEncoderParams(CNNEncoderParams):
    n_heads: int = 4
    kernel_size: int = 31
    p_dropout: float = 0.1

    use_pe: bool = True
    pe_type: PE_TYPES = "PE"

    # condition
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0
    condition_type: tp.Optional[CONDITIONAL_TYPES] = None

    # projection
    use_projection: bool = True
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"


class ConformerEncoder(CNNEncoder):
    params: ConformerEncoderParams

    def __init__(self, params: ConformerEncoderParams, input_dim):
        super().__init__(params, input_dim)

        in_dim = super().output_dim
        inner_dim = params.encoder_inner_dim

        self.prenet = ConvPrenet(
            in_channels=in_dim,
            out_channels=inner_dim,
        )

        if params.condition:
            self.cond_layer = ConditionalLayer(
                params.condition_type, inner_dim, params.condition_dim
            )

        if params.use_pe:
            self.pe = PositionalEncodings(params.pe_type, inner_dim, batch_first=True)

        self.encoder = Conformer(
            input_dim=inner_dim,
            num_heads=params.n_heads,
            ffn_dim=inner_dim,
            num_layers=params.encoder_num_layers,
            depthwise_conv_kernel_size=params.kernel_size,
            dropout=params.p_dropout,
        )

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

        x = self.prenet(x.transpose(1, -1)).transpose(1, -1)

        if self.params.condition:
            c = self.get_condition(inputs, self.params.condition)
            x = self.cond_layer(x, c, x_mask)

        if self.params.use_pe:
            x = self.pe(x)

        x = self.hook_update_content(x, x_lens, inputs)

        z, _ = self.encoder(x, x_lens)

        y = self.proj(z, x_mask)

        return EncoderOutput.copy_from(inputs).set_content(y).set_hidden_state(z)
