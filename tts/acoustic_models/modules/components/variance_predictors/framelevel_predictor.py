import typing as tp

import torch

from pydantic import Field
from torch.nn import functional as F

from speechflow.utils.tensor_utils import get_mask_from_lengths
from tts.acoustic_models.modules.common import VarianceEmbedding
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.components.discriminators import SignalDiscriminator
from tts.acoustic_models.modules.data_types import MODEL_INPUT_TYPE
from tts.acoustic_models.modules.params import VarianceParams, VariancePredictorParams

__all__ = [
    "FrameLevelPredictor",
    "FrameLevelPredictorParams",
    "FrameLevelPredictorWithDiscriminator",
    "FrameLevelPredictorWithDiscriminatorParams",
]


class FrameLevelPredictorParams(VariancePredictorParams):
    frame_encoder_type: str = "VarianceEncoder"
    frame_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    activation_fn: str = "Identity"
    use_ssl_adjustment: bool = (
        False  # improving the target feature through prediction over SSL model
    )
    use_mtm: bool = False  # masked token modeling
    loss_type: str = "smooth_l1_loss"
    loss_alpha: float = 1.0
    variance_params: VarianceParams = VarianceParams()


class FrameLevelPredictor(Component):
    params: FrameLevelPredictorParams

    def __init__(
        self,
        params: FrameLevelPredictorParams,
        input_dim: tp.Union[int, tp.Tuple[int, ...]],
    ):
        super().__init__(params, input_dim)

        from tts.acoustic_models.modules import TTS_ENCODERS

        def _init_encoder(
            _enc_cls, _enc_params_cls, _encoder_params, _input_dim, _output_dim=None
        ):
            if _output_dim is None:
                _output_dim = params.vp_output_dim

            _enc_params = _enc_params_cls.init_from_parent_params(params, _encoder_params)
            _enc_params.encoder_num_layers = params.vp_num_layers
            _enc_params.encoder_inner_dim = params.vp_inner_dim
            _enc_params.encoder_output_dim = _output_dim
            _enc_params.projection_activation_fn = params.activation_fn
            return _enc_cls(_enc_params, _input_dim)

        enc_cls, enc_params_cls = TTS_ENCODERS[params.frame_encoder_type]

        if params.use_ssl_adjustment:
            self.ssl_encoder = _init_encoder(
                enc_cls, enc_params_cls, params.frame_encoder_params, params.ssl_feat_dim
            )
            self.ssl_proj = Regression(
                params.vp_inner_dim * 2, 1, activation_fn=params.activation_fn
            )
        else:
            self.ssl_encoder = None

        if params.use_mtm:
            assert self.params.variance_params.as_embedding  # type: ignore
            emb_dim = params.variance_params.emb_dim  # type: ignore
            self.mtm_embeddings = VarianceEmbedding(
                interval=params.variance_params.interval,  # type: ignore
                n_bins=params.variance_params.n_bins,  # type: ignore
                log_scale=params.variance_params.log_scale,  # type: ignore
                emb_dim=emb_dim,  # type: ignore
            )
            self.mtm_pre_proj = Regression(input_dim, emb_dim)
            self.mtm_encoder = _init_encoder(
                enc_cls, enc_params_cls, params.frame_encoder_params, 2 * emb_dim, emb_dim
            )
            self.mtm_proj = Regression(params.vp_inner_dim, emb_dim, activation_fn="Tanh")

            input_dim += emb_dim
        else:
            self.mtm_encoder = None

        self.frame_encoder = _init_encoder(
            enc_cls, enc_params_cls, params.frame_encoder_params, input_dim
        )

    @property
    def output_dim(self):
        return self.params.vp_output_dim

    def forward_step(
        self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        name = kwargs.get("name")
        target_by_frames = kwargs.get("target")
        if target_by_frames is not None:
            target_by_frames = target_by_frames.squeeze(-1)

        losses = {}

        if self.mtm_encoder is not None and target_by_frames is not None:
            m = model_inputs.masks["spectrogram"]
            inv_m = ~m.unsqueeze(-1)

            mtm_target_embs = self.mtm_embeddings(target_by_frames)
            mtm_target_embs_mask = mtm_target_embs * m.unsqueeze(-1) - 1 * inv_m

            x_proj = self.mtm_pre_proj(x.detach())
            mtm_x = torch.cat([mtm_target_embs_mask, x_proj], dim=-1)
            _, mtm_enc_ctx = self.mtm_encoder.process_content(
                mtm_x, x_lengths, model_inputs
            )
            mtm_predict_embs = self.mtm_proj(mtm_enc_ctx)

            if self.training:
                losses[f"{name}_mtm_loss_by_frames"] = F.mse_loss(
                    mtm_predict_embs * inv_m, (mtm_target_embs * inv_m).detach()
                )

            merger_embs = mtm_target_embs * m.unsqueeze(-1) + mtm_predict_embs * inv_m
            x = torch.cat([x, merger_embs.detach()], dim=-1)

        predict, enc_ctx = self.frame_encoder.process_content(x, x_lengths, model_inputs)
        predict = predict.squeeze(-1)

        if self.training:
            loss_fn = getattr(F, self.params.loss_type)

            if self.ssl_encoder is not None:
                _, ssl_ctx = self.ssl_encoder.process_content(
                    model_inputs.ssl_feat, x_lengths, model_inputs
                )
                var_from_ssl = self.ssl_proj(
                    torch.cat([enc_ctx.detach(), ssl_ctx], dim=2)
                ).squeeze(-1)

                if self.params.variance_params.log_scale:  # type: ignore
                    target_by_frames = torch.log1p(target_by_frames)

                loss = self.params.loss_alpha * loss_fn(var_from_ssl, target_by_frames)
                losses[f"{name}_ssl_adjustment_loss_by_frames"] = loss

                if self.params.variance_params.log_scale:  # type: ignore
                    var_from_ssl = torch.expm1(var_from_ssl)

                var_by_frames = var_from_ssl.detach()
                target_by_frames = var_by_frames
            else:
                var_by_frames = target_by_frames

            if predict is not None and var_by_frames is not None:
                if self.params.variance_params.log_scale:  # type: ignore
                    var_by_frames = torch.log1p(var_by_frames)

                loss = self.params.loss_alpha * loss_fn(predict, var_by_frames)
                losses[f"{name}_{self.params.loss_type}_by_frames"] = loss

        if self.params.variance_params.log_scale:  # type: ignore
            predict = torch.expm1(predict)

        return (
            predict,
            {
                f"{name}_vp_context": enc_ctx,
                f"{name}_vp_predict": predict,
                f"{name}_vp_target": target_by_frames,
            },
            losses,
        )


class FrameLevelPredictorWithDiscriminatorParams(FrameLevelPredictorParams):
    pass


class FrameLevelPredictorWithDiscriminator(FrameLevelPredictor, Component):
    params: FrameLevelPredictorWithDiscriminatorParams

    def __init__(
        self,
        params: FrameLevelPredictorWithDiscriminatorParams,
        input_dim: tp.Union[int, tp.Tuple[int, ...]],
    ):
        super().__init__(params, input_dim)
        self.disc = SignalDiscriminator(in_channels=params.vp_inner_dim)

    def forward_step(
        self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        var_predict, var_content, var_losses = super().forward_step(
            x, x_lengths, model_inputs, **kwargs
        )

        var = kwargs.get("target")
        name = kwargs.get("name")

        if self.training:
            var_real = var
            var_fake = var_content[f"{name}_vp_predict"]
            context = var_content[f"{name}_vp_context"]
            mask = get_mask_from_lengths(x_lengths, max_length=x.shape[1])

            if var_real.ndim == 2:
                var_real = var_real.unsqueeze(-1)
            if var_fake.ndim == 2:
                var_fake = var_fake.unsqueeze(-1)

            disc_losses = self.disc.calculate_loss(
                context.transpose(1, -1),
                mask.transpose(1, -1),
                var_real.transpose(1, -1),
                var_fake.transpose(1, -1),
                model_inputs.global_step,
            )
            var_losses.update(
                {f"{name}_{k}_by_frames": v for k, v in disc_losses.items()}
            )

        return var_predict, var_content, var_losses
