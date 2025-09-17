import typing as tp

from copy import deepcopy as copy

from pydantic import Field

from speechflow.data_pipeline.collate_functions.tts_collate import LinguisticFeatures
from speechflow.training.base_model import BaseTorchModelParams

__all__ = [
    "EncoderParams",
    "VarianceParams",
    "VariancePredictorParams",
    "VarianceAdaptorParams",
    "DecoderParams",
    "GeneralConditionParams",
    "EmbeddingParams",
    "PostnetParams",
]

tp_TEXT_FEATURES = tp.Literal["transcription", "xpbert_feat", "lm_feat"]
tp_AUDIO_FEATURES = tp.Literal["waveform", "spectrogram", "ssl_feat", "ac_feat"]
tp_BIOMETRIC_MODELS = tp.Literal["resemblyzer", "speechbrain", "wespeaker"]
tp_INPUT_FEATURES = tp.Union[tp_TEXT_FEATURES, tp_AUDIO_FEATURES]


class EmbeddingParams(BaseTorchModelParams):
    """Embedding component parameters."""

    input: tp.Union[tp_INPUT_FEATURES, tp.List[tp_INPUT_FEATURES]] = "transcription"
    target: tp_AUDIO_FEATURES = "spectrogram"

    # Transcription embeddings parameters
    alphabet_size: int = Field(ge=1, default=1)
    n_symbols_per_token: int = Field(ge=1, default=1)
    token_emb_dim: int = 256

    # Speaker embeddings parameters
    n_langs: int = Field(ge=1, default=1)
    n_speakers: int = Field(ge=1, default=1)
    use_onehot_speaker_emb: bool = False
    use_learnable_speaker_emb: bool = False
    use_dnn_speaker_emb: bool = False
    use_mean_dnn_speaker_emb: bool = False
    speaker_emb_dim: int = 256
    speaker_biometric_model: tp_BIOMETRIC_MODELS = "resemblyzer"

    # Linguistic sequences
    num_additional_integer_seqs: int = -1
    num_additional_float_seqs: int = -1

    # XPBert parameters
    xpbert_feat_dim: int = 768
    xpbert_feat_proj_dim: int = 768

    # LM features parameters
    lm_feat_dim: int = 1024
    lm_feat_proj_dim: int = 1024

    # spectrogram parameters
    linear_spectrogram_dim: int = 513
    linear_spectrogram_proj_dim: int = 513
    mel_spectrogram_dim: int = 80
    mel_spectrogram_proj_dim: int = 80

    # SSL features parameters
    ssl_feat_dim: int = 1024
    ssl_feat_proj_dim: int = 1024

    # AC features parameters
    ac_feat_dim: int = 1024
    ac_feat_proj_dim: int = 1024

    # Average embeddings parameters
    use_average_emb: bool = False
    average_emb_dim: int = 0
    averages: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})

    # Speech quality embeddings parameters
    speech_quality_emb_dim: int = 4

    max_input_length: tp.Union[int, tp.List[int]] = 512
    max_output_length: int = 4096

    def model_post_init(self, __context: tp.Any):
        super().model_post_init(__context)

        if not isinstance(self.input, list):
            self.input = [self.input]

        if (
            self.use_onehot_speaker_emb
            + self.use_learnable_speaker_emb
            + self.use_dnn_speaker_emb
            + self.use_mean_dnn_speaker_emb
        ) > 1:
            raise AttributeError(
                "Cannot perform with both types of embeddings, choose bio/learnable"
            )

        if self.num_additional_integer_seqs < 0:
            self.num_additional_integer_seqs = LinguisticFeatures.num_integer_features()
        if self.num_additional_float_seqs < 0:
            self.num_additional_float_seqs = LinguisticFeatures.num_float_features()

        if self.use_average_emb:
            for name in self.averages.keys():
                self.averages[name].setdefault("n_bins", 64)
                self.averages[name].setdefault("emb_dim", 16)
                self.averages[name].setdefault("log_scale", False)

            if not self.average_emb_dim:
                dim = 0
                for params in self.averages.values():
                    dim += params["emb_dim"]
                self.average_emb_dim = dim

    def get_feat_dim(self, feat_name: str) -> int:
        if feat_name == "spectrogram":
            feat_name = "mel_spectrogram"
        if hasattr(self, f"{feat_name}_dim"):
            return getattr(self, f"{feat_name}_dim")
        else:
            raise RuntimeError(f"Dim for {feat_name} not found")

    @staticmethod
    def check_deprecated_params(cfg: dict) -> dict:
        if "n_symbols" in cfg:
            cfg["alphabet_size"] = cfg.pop("n_symbols")
        if "input" in cfg:
            if cfg["input"] == "mel_spectrogram":
                cfg["input"] = "spectrogram"
        if "target" in cfg:
            if cfg["target"] == "mel_spectrogram":
                cfg["target"] = "spectrogram"

        return cfg


class EncoderParams(EmbeddingParams):
    """Encoder component parameters."""

    encoder_type: str = "ForwardEncoder"
    encoder_num_layers: int = 2
    encoder_inner_dim: int = 512
    encoder_output_dim: int = 512
    encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})


class DecoderParams(EmbeddingParams):
    """Decoder component parameters."""

    decoder_type: tp.Optional[str] = "ForwardDecoder"
    decoder_num_layers: int = 2
    decoder_inner_dim: int = 512
    decoder_output_dim: int = 80
    decoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})


class PostnetParams(EmbeddingParams):
    """Postnet component parameters."""

    postnet_type: tp.Optional[str] = "ForwardPostnet"
    postnet_num_layers: int = 1
    postnet_inner_dim: int = 512
    postnet_output_dim: int = 80
    postnet_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})


class GeneralConditionParams(EmbeddingParams):
    """Level condition parameters."""

    general_condition: tp.Dict[str, tp.List[tp.Dict[str, tp.Any]]] = Field(
        default_factory=lambda: {}
    )

    def model_post_init(self, __context: tp.Any):
        super().model_post_init(__context)

        from tts.acoustic_models.modules.general_condition import ModelLevel

        model_level = [key.name for key in ModelLevel]
        for level, cond_params in self.general_condition.items():

            if level not in model_level:
                raise ValueError(
                    f"Invalid stage number in GeneralConditionParams: {level}. "
                    f"Only values from {', '.join(model_level)} are allowed."
                )

            for params in cond_params:
                if isinstance(params["condition"], str):
                    params["condition"] = [params["condition"]]


class VariancePredictorParams(EmbeddingParams):
    vp_num_layers: int = 1
    vp_inner_dim: int = 256
    vp_output_dim: int = 1
    vp_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})


class VarianceParams(BaseTorchModelParams):
    predictor_type: str = "TokenLevelPredictor"
    predictor_params: VariancePredictorParams = None  # type: ignore
    dim: int = 1
    target: str = None
    input_content: tp.Tuple[int, ...] = (0,)
    input_content_dim: tp.Tuple[int, ...] = None
    detach_input: tp.Union[bool, tp.Tuple[bool, ...]] = False
    detach_output: bool = True
    use_target: bool = True
    denormalize: bool = False
    upsample: bool = False
    cat_to_content: tp.Optional[tp.Tuple[int, ...]] = ()
    overwrite_content: tp.Optional[tp.Tuple[int, ...]] = None
    as_encoder: bool = False
    as_embedding: bool = False
    interval: tp.Tuple[float, float] = (0.0, 1.0)
    log_scale: bool = False
    n_bins: int = 256
    emb_dim: int = 128
    begin_iter: int = 0
    end_iter: int = 1_000_000_000
    every_iter: int = 1
    skip: bool = False
    use_loss: bool = False
    loss_type: str = "l1_loss"

    def model_post_init(self, __context: tp.Any):
        super().model_post_init(__context)

        if self.predictor_params is None:
            self.predictor_params = VariancePredictorParams()

        if isinstance(self.detach_input, bool):
            self.detach_input = tuple([self.detach_input] * len(self.input_content))
        else:
            self.detach_input = tuple(self.detach_input)

        assert len(self.input_content) == len(self.detach_input)

        if self.as_encoder:
            self.dim = self.predictor_params.vp_output_dim
            self.use_target = False
            self.detach_output = False
            self.as_embedding = False
            self.use_loss = False
        else:
            if self.dim > self.predictor_params.vp_output_dim:
                self.predictor_params.vp_output_dim = self.dim


class VarianceAdaptorParams(VariancePredictorParams):
    va_type: str = "VarianceAdaptor"
    va_length_regulator_type: str = "SoftLengthRegulator"  # "LengthRegulator"
    va_variances: tp.Dict[int, tp.Tuple[str, ...]] = None  # type: ignore
    va_variance_params: tp.Dict[str, VarianceParams] = None  # type: ignore

    def model_post_init(self, __context: tp.Any):
        super().model_post_init(__context)

        if self.va_variances is None:
            self.va_variances = {}
            self.va_variance_params = {}
            return

        if isinstance(self.va_variances, (tuple, set, list)):
            self.va_variances = {0: tuple(self.va_variances)}

        all_va_variances = [t for v in self.va_variances.values() for t in v]
        if len(all_va_variances) != len(set(all_va_variances)):
            raise ValueError("Variances at each level must be unique.")

        vp_global_params = {
            k: v
            for k, v in self.to_dict().items()
            if k in VariancePredictorParams().to_dict()
        }
        variance_params: tp.Dict[str, VarianceParams] = {
            name: VarianceParams() for name in all_va_variances
        }
        if self.va_variance_params:
            for name, params in self.va_variance_params.items():
                variance_params[name] = (
                    VarianceParams(**params) if isinstance(params, dict) else params
                )
                vp_custom_params = variance_params[name].predictor_params
                if vp_custom_params is None:
                    variance_params[name].predictor_params = VariancePredictorParams(
                        **vp_global_params
                    )
                elif isinstance(vp_custom_params, dict):
                    vp_custom_params = copy(vp_global_params)
                    vp_custom_params.update(variance_params[name].predictor_params)
                    variance_params[name].predictor_params = VariancePredictorParams(
                        **vp_custom_params
                    )

        self.va_variance_params = variance_params
