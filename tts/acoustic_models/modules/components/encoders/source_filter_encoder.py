import typing as tp

from pydantic import Field
from torch import nn
from torch.nn import functional as F

from speechflow.utils.tensor_utils import apply_mask
from tts.acoustic_models.modules.common import CONDITIONAL_TYPES, VarianceEmbedding
from tts.acoustic_models.modules.common.layers import Conv
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import (
    ComponentInput,
    ComponentOutput,
    EncoderOutput,
)
from tts.acoustic_models.modules.params import EncoderParams

__all__ = [
    "SFEncoder",
    "SFEncoderParams",
    "SFEncoderWithClassificationAdaptor",
    "SFEncoderWithClassificationAdaptorParams",
]


class SFEncoderParams(EncoderParams):
    # base encoder params
    base_encoder_type: str = "RNNEncoder"
    base_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})

    # condition
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0
    condition_type: tp.Optional[CONDITIONAL_TYPES] = None

    # embedding bucketize
    var_as_embedding: tp.Tuple[bool, bool] = (False, False)
    var_interval: tp.Tuple[tp.Tuple[float, float], tp.Tuple[float, float]] = (
        (0, 150),
        (0, 880),
    )
    var_n_bins: tp.Tuple[int, int] = (256, 256)
    var_embedding_dim: tp.Tuple[int, int] = (64, 64)
    var_log_scale: tp.Tuple[bool, bool] = (False, True)

    def model_post_init(self, __context: tp.Any):
        if self.base_encoder_params is None:
            self.base_encoder_params = {}

        if self.condition:
            self.base_encoder_params.setdefault("condition", self.condition)
            self.base_encoder_params.setdefault("condition_dim", self.condition_dim)
            self.base_encoder_params.setdefault("condition_type", self.condition_type)


class SFEncoder(Component):
    params: SFEncoderParams

    def __init__(self, params: SFEncoderParams, input_dim):
        super().__init__(params, input_dim)

        from tts.acoustic_models.modules import TTS_ENCODERS

        base_enc_cls, base_enc_params_cls = TTS_ENCODERS[params.base_encoder_type]
        base_enc_params = base_enc_params_cls.init_from_parent_params(
            params,
            params.base_encoder_params,
            strict=False,
        )

        in_dim_e = 1
        if params.var_as_embedding[0]:
            self.energy_embeddings = VarianceEmbedding(
                interval=params.var_interval[0],
                n_bins=params.var_n_bins[0],
                emb_dim=params.var_embedding_dim[0],
                log_scale=params.var_log_scale[0],
            )
            in_dim_e = params.var_embedding_dim[0]
        else:
            self.energy_embeddings = None

        in_dim_p = 1
        if params.var_as_embedding[0]:
            self.pitch_embeddings = VarianceEmbedding(
                interval=params.var_interval[1],
                n_bins=params.var_n_bins[1],
                emb_dim=params.var_embedding_dim[1],
                log_scale=params.var_log_scale[1],
            )
            in_dim_p = params.var_embedding_dim[1]
        else:
            self.pitch_embeddings = None

        self.pre_source = nn.Conv1d(input_dim, params.encoder_inner_dim, 1)
        self.pre_filter_e = nn.Conv1d(in_dim_e, params.encoder_inner_dim, 1)
        self.pre_filter_p = nn.Conv1d(in_dim_p, params.encoder_inner_dim, 1)

        self.source_encoder = base_enc_cls(base_enc_params, params.encoder_inner_dim)
        self.filter_encoder_e = base_enc_cls(base_enc_params, params.encoder_inner_dim)
        self.filter_encoder_p = base_enc_cls(base_enc_params, params.encoder_inner_dim)
        self.encoder = base_enc_cls(base_enc_params, params.encoder_output_dim)

    @property
    def output_dim(self):
        return self.encoder.output_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        x, x_lens, x_mask = inputs.get_content_and_mask()

        if (
            self.training
            or "energy_postprocessed" not in inputs.model_inputs.additional_inputs
        ):
            energy = inputs.model_inputs.energy
            pitch = inputs.model_inputs.pitch
        else:
            energy = inputs.model_inputs.additional_inputs["energy_postprocessed"]
            pitch = inputs.model_inputs.additional_inputs["pitch_postprocessed"]

        if self.energy_embeddings is not None:
            energy_embs = self.energy_embeddings(energy)
        else:
            energy_embs = energy.unsqueeze(-1)

        if self.pitch_embeddings is not None:
            pitch_embs = self.pitch_embeddings(pitch)
        else:
            pitch_embs = pitch.unsqueeze(-1)

        x_src = apply_mask(self.pre_source(x.transpose(1, -1)).transpose(1, -1), x_mask)
        x_ftr_e = apply_mask(
            self.pre_filter_e(energy_embs.transpose(1, -1)).transpose(1, -1), x_mask
        )
        x_ftr_p = apply_mask(
            self.pre_filter_p(pitch_embs.transpose(1, -1)).transpose(1, -1), x_mask
        )

        x_src = ComponentInput.copy_from(inputs).set_content(x_src, x_lens)
        output: ComponentOutput = self.source_encoder(x_src)
        y_src = output.get_content()[0]

        x_ftr_e = ComponentInput.copy_from(inputs).set_content(x_ftr_e, x_lens)
        output: ComponentOutput = self.filter_encoder_e(x_ftr_e)
        y_ftr_e = output.get_content()[0]

        x_ftr_p = ComponentInput.copy_from(inputs).set_content(x_ftr_p, x_lens)
        output: ComponentOutput = self.filter_encoder_p(x_ftr_p)
        y_ftr_p = output.get_content()[0]

        y = ComponentInput.copy_from(inputs).set_content(
            y_src + y_ftr_e + y_ftr_p, x_lens
        )
        output: ComponentOutput = self.encoder(y)
        y = output.get_content()[0]

        outputs = EncoderOutput.copy_from(inputs).set_content(y)
        outputs.additional_content[f"{self.__class__.__name__}_{self.id}_src"] = y_src
        outputs.additional_content[f"{self.__class__.__name__}_{self.id}_ftr_e"] = y_ftr_e
        outputs.additional_content[f"{self.__class__.__name__}_{self.id}_ftr_p"] = y_ftr_p
        return outputs


class SFEncoderWithClassificationAdaptorParams(SFEncoderParams):
    n_convolutions: int = 3
    kernel_size: int = 5


class SFEncoderWithClassificationAdaptor(SFEncoder):
    params: SFEncoderWithClassificationAdaptorParams

    def __init__(self, params: SFEncoderWithClassificationAdaptorParams, input_dim: int):
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

        self.conv_module = nn.ModuleList(convolutions)
        self.components_output_dim["adaptor_context"] = lambda: self.output_dim

    def forward_step(self, x: ComponentInput) -> EncoderOutput:
        result: EncoderOutput = super().forward_step(x)

        adaptor_context = result.additional_content.setdefault(
            f"adaptor_context_{self.id}", []
        )

        ctx = x.get_content(0).transpose(1, 2)
        for conv in self.conv_module:
            ctx = conv(ctx)

        adaptor_context.append(ctx.transpose(2, 1))

        return result
