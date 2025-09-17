import typing as tp

from copy import deepcopy as copy

import torch
import numpy.typing as npt

from speechflow.data_pipeline.datasample_processors.spectrogram_processors import (
    MelProcessor,
)
from speechflow.io import Config
from speechflow.training.base_model import BaseTorchModel
from speechflow.utils.dictutils import find_field
from tts.acoustic_models.data_types import TTSForwardInput, TTSForwardOutput
from tts.acoustic_models.modules import (
    TTS_DECODERS,
    TTS_ENCODERS,
    TTS_POSTNETS,
    TTS_VARIANCE_ADAPTORS,
)
from tts.acoustic_models.modules.additional_modules import (
    AdditionalModules,
    AdditionalModulesParams,
)
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.embedding import EmbeddingComponent
from tts.acoustic_models.modules.general_condition import GeneralCondition, ModelLevel
from tts.acoustic_models.modules.params import *

__all__ = ["ParallelTTSModel", "ParallelTTSParams"]


class ParallelTTSParams(
    GeneralConditionParams,
    EncoderParams,
    VarianceAdaptorParams,
    DecoderParams,
    PostnetParams,
    AdditionalModulesParams,
):
    """Parallel TTS model parameters."""

    def model_post_init(self, __context: tp.Any):
        super().model_post_init(__context)


class ParallelTTSModel(BaseTorchModel):
    params: ParallelTTSParams

    def __init__(
        self,
        cfg: tp.Union[tp.MutableMapping, ParallelTTSParams],
        strict_init: bool = True,
    ):
        super().__init__(ParallelTTSParams.create(cfg, False))
        params = self.params

        self.embedding_component = EmbeddingComponent(params)

        self.cond_module_0 = GeneralCondition(
            params,
            input_dim=self.embedding_component.output_dim,
            level=ModelLevel.level_0,
        )

        cls, params_cls = TTS_ENCODERS[params.encoder_type]
        encoder_params = params_cls.init_from_parent_params(params, params.encoder_params)
        self.encoder = cls(encoder_params, input_dim=self.cond_module_0.output_dim)

        self.cond_module_1 = GeneralCondition(
            params, input_dim=self.encoder.output_dim, level=ModelLevel.level_1  # type: ignore
        )

        self.va = torch.nn.ModuleList()
        va_variances = list(params.va_variances.values())
        input_dim = self.cond_module_1.output_dim
        if va_variances:
            for var_names in va_variances:
                if var_names:
                    va_params = copy(params)
                    va_params.va_variances = var_names  # type: ignore
                    adaptor, _ = TTS_VARIANCE_ADAPTORS[va_params.va_type]
                    self.va.append(adaptor(va_params, input_dim=input_dim))
                    input_dim = self.va[-1].output_dim
        else:
            adaptor, _ = TTS_VARIANCE_ADAPTORS["DummyVarianceAdaptor"]
            self.va.append(adaptor(params, input_dim=input_dim))

        output_dim = self.va[-1].output_dim[0]

        if params.decoder_type is not None:
            self.cond_module_2 = GeneralCondition(
                params, input_dim=output_dim, level=ModelLevel.level_2
            )

            cls, params_cls = TTS_DECODERS[params.decoder_type]
            decoder_params = params_cls.init_from_parent_params(
                params, params.decoder_params
            )
            self.decoder = cls(decoder_params, input_dim=self.cond_module_2.output_dim)
            output_dim = self.decoder.output_dim
        else:
            self.cond_module_2 = None
            self.decoder = None

        if params.postnet_type is not None:
            self.cond_module_3 = GeneralCondition(
                params, input_dim=output_dim, level=ModelLevel.level_3  # type: ignore
            )

            cls, params_cls = TTS_POSTNETS[params.postnet_type]
            postnet_params = params_cls.init_from_parent_params(
                params, params.postnet_params
            )
            self.postnet = cls(postnet_params, input_dim=self.cond_module_3.output_dim)  # type: ignore
            output_dim = self.postnet.output_dim
        else:
            self.cond_module_3 = None
            self.postnet = None

        self.additional_modules = AdditionalModules(params, input_dim=output_dim)

    @property
    def device(self) -> torch.device:
        return self.embedding_component.emb_calculator.embedding.weight.device

    @property
    def last_module(self) -> Component:
        for module in reversed(
            [
                self.cond_module_0,
                self.encoder,
                self.cond_module_1,
                self.va[-1],
                self.cond_module_2,
                self.decoder,
                self.cond_module_3,
                self.postnet,
            ]
        ):
            if module is not None:
                return module
        else:
            raise RuntimeError(f"Invalid {self.__class__.__name__} configuration!")

    @property
    def output_dim(self) -> int:
        out_dim = self.last_module.output_dim
        if isinstance(out_dim, int):
            return out_dim
        else:
            return out_dim[0]

    def _encode_inputs(self, inputs, inference: bool = False):
        """Utility method that reduces code duplication."""
        if inputs.additional_inputs is None:
            inputs.additional_inputs = {}

        x = self.embedding_component(inputs)

        if inference:
            x = self.cond_module_0.inference(x)
            x = self.encoder.inference(x)  # type: ignore
        else:
            x = self.cond_module_0(x)
            x = self.encoder(x)  # type: ignore

        if inference:
            x = self.cond_module_1.inference(x)
        else:
            x = self.cond_module_1(x)

        return x

    def _predict_variances(self, x, inference: bool = False, **kwargs):
        """Utility method that reduces code duplication."""
        predictions = {}
        for va in self.va:
            x = va(x, **kwargs) if not inference else va.inference(x, **kwargs)
            predictions.update(x.variance_predictions)
            x.model_inputs.additional_inputs.update(x.additional_content)

        return x.select_content(0), predictions

    def forward(self, inputs: TTSForwardInput, **kwargs) -> TTSForwardOutput:
        x = self._encode_inputs(inputs)

        va_output, variance_predictions = self._predict_variances(x, **kwargs)

        if self.decoder is not None:
            x = self.cond_module_2(va_output)
            x = self.decoder(x)  # type: ignore
            gate = getattr(x, "gate", None)
        else:
            x = va_output
            gate = None

        s_all = x.stack_content()

        if self.postnet is not None:
            x = self.cond_module_3(x)
            x = self.postnet(x)
            if x.content is not None:
                s_all = torch.cat([s_all, x.content.unsqueeze(0)], dim=0)

        x = self.additional_modules(x)

        output = TTSForwardOutput(
            spectrogram=s_all,
            spectrogram_lengths=x.content_lengths,
            after_postnet_spectrogram=s_all[-1],
            gate=gate,
            variance_predictions=variance_predictions,
            additional_content=x.additional_content,
            additional_losses=x.additional_losses,
            embeddings=x.embeddings,
        )
        return output

    def inference(self, inputs: TTSForwardInput, **kwargs) -> TTSForwardOutput:
        x = self._encode_inputs(inputs, inference=True)

        va_output, variance_predictions = self._predict_variances(
            x, inference=True, **kwargs
        )

        if self.decoder is not None:
            x = self.cond_module_2.inference(va_output)
            x = self.decoder.inference(x)
            gate = getattr(x, "gate", None)
        else:
            x = va_output
            gate = None

        if self.postnet is not None:
            x = self.cond_module_3(x)
            x = self.postnet.inference(x)

        output = TTSForwardOutput(
            spectrogram=x.content,
            spectrogram_lengths=x.content_lengths,
            after_postnet_spectrogram=x.content,
            gate=gate,
            variance_predictions=variance_predictions,
            additional_content=x.additional_content,
            additional_losses=x.additional_losses,
            embeddings=x.embeddings,
        )
        return output

    def get_variance(
        self, inputs, ignored_variance: tp.Set = None
    ) -> tp.Tuple[
        tp.Dict[str, torch.Tensor], tp.Dict[str, torch.Tensor], tp.Dict[str, torch.Tensor]
    ]:
        x = self._encode_inputs(inputs, inference=True)
        va_output, variance_predictions = self._predict_variances(
            x, inference=True, ignored_variance=ignored_variance
        )
        return va_output, variance_predictions, x.additional_content

    def get_speaker_embedding(
        self, speaker_id: int, bio_emb: npt.NDArray, mean_bio_emb: npt.NDArray
    ):
        if self.params.use_learnable_speaker_emb and speaker_id is not None:
            sp_id = torch.LongTensor([speaker_id]).to(self.device)
            sp_emb = self.embedding_component.emb_calculator.speaker_emb(sp_id)
        elif self.params.use_dnn_speaker_emb and bio_emb is not None:
            sp_emb = torch.from_numpy(bio_emb).to(self.device)
        elif self.params.use_mean_dnn_speaker_emb and mean_bio_emb is not None:
            sp_emb = torch.from_numpy(mean_bio_emb).to(self.device)
        else:
            raise NotImplementedError

        if hasattr(self.embedding_component.emb_calculator, "speaker_emb_proj"):
            sp_emb = self.embedding_component.emb_calculator.speaker_emb_proj(sp_emb)

        return {"speaker": sp_emb}

    def get_style_embedding(
        self,
        bio_embedding: torch.Tensor,
        spectrogram: torch.Tensor,
        ssl_feat: torch.Tensor,
    ):
        def get_sample(_sampler, feat_type, input_feat):
            input_feat = torch.from_numpy(input_feat).to(self.device)
            if input_feat.ndim == 2:
                input_feat = input_feat.unsqueeze(0)
            if "ssl" in feat_type:
                input_feat = self.embedding_component.emb_calculator.ssl_proj(input_feat)

            if input_feat.ndim > 1:
                lens = torch.LongTensor([input_feat.shape[1]]).to(input_feat.device)
            else:
                input_feat = input_feat.unsqueeze(0).unsqueeze(0)
                lens = None

            emb, _, _ = _sampler.encode(input_feat, lens, None)
            return emb.squeeze(1)

        for k, module in self._modules["va"]._modules.items():
            if any("style" in name for name in module.va_variances):
                break
        else:
            module = None

        if module is not None:
            sampler = list(module._modules["predictors"].values())[0]
            if "ssl" in list(module.predictors.values())[0].params.source:
                return {"style_emb": get_sample(sampler, "ssl_feat", ssl_feat)}
            elif "emb" in list(module.predictors.values())[0].params.source:
                return {"style_emb": get_sample(sampler, "bio", bio_embedding)}
            else:
                return {"style_emb": get_sample(sampler, "spectrogram", spectrogram)}

        return {}

    @classmethod
    def update_and_validate_model_params(cls, cfg_model: Config, cfg_data: Config):
        if "speaker_biometric_model" not in cfg_model["model"]["params"]:
            cfg_model["model"]["params"].speaker_biometric_model = find_field(
                cfg_data["preproc"], "voice_bio.model_type", "resemblyzer"
            )

        if (
            cfg_model["model"]["params"].get("decoder_target", "spectrogram")
            == "spectrogram"
        ):
            cfg_model["model"]["params"].decoder_output_dim = find_field(
                cfg_data["preproc"], "linear_to_mel.n_mels"
            )
            cfg_model["model"]["params"].postnet_output_dim = find_field(
                cfg_data["preproc"], "linear_to_mel.n_mels"
            )

        if "SSIM" in cfg_model["loss"]:
            max_abs_value = find_field(cfg_data["preproc"], "normalize.max_abs_value")
            melscale_pipe = find_field(cfg_data["preproc"], "melscale.pipe", [])
            if max_abs_value is None and "normalize" in melscale_pipe:
                max_abs_value = MelProcessor().max_abs_value

            if max_abs_value:
                min_value = cfg_model["loss"]["SSIM"].get("min_value", 0)
                max_value = cfg_model["loss"]["SSIM"].get("max_value", 0)
                if abs(min_value) != max_abs_value or abs(max_value) != max_abs_value:
                    raise ValueError("SSIM configuration is not valid!")

        return cfg_model

    def load_params(self, state_dict: tp.Dict[str, torch.Tensor], *args):
        return super().load_params(state_dict, args)
