import typing as tp

from functools import partial

import torch

from pydantic import Field
from torch import nn
from torch.nn import functional as F

from speechflow.utils.tensor_utils import apply_mask, get_mask_from_lengths
from tts.acoustic_models.modules.common import SoftLengthRegulator, VarianceEmbedding
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.components.discriminators import SignalDiscriminator
from tts.acoustic_models.modules.data_types import MODEL_INPUT_TYPE
from tts.acoustic_models.modules.params import VarianceParams, VariancePredictorParams

__all__ = [
    "TokenLevelPredictor",
    "TokenLevelPredictorParams",
    "TokenLevelPredictorWithDiscriminator",
    "TokenLevelPredictorWithDiscriminatorParams",
]


class TokenLevelPredictorParams(VariancePredictorParams):
    word_encoder_type: str = "VarianceEncoder"
    word_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    token_encoder_type: str = "VarianceEncoder"
    token_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    activation_fn: str = "Identity"
    add_lm_feat: bool = False
    use_mtm: bool = False  # masked token modeling
    loss_type: str = "l1_loss"
    loss_alpha: float = 1.0
    variance_params: VarianceParams = VarianceParams()


class TokenLevelPredictor(Component):
    params: TokenLevelPredictorParams

    def __init__(
        self,
        params: TokenLevelPredictorParams,
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

        enc_cls, enc_params_cls = TTS_ENCODERS[params.token_encoder_type]

        if params.add_lm_feat:
            self.lm_proj = nn.Linear(params.lm_feat_dim, input_dim, bias=False)
            self.prenet = nn.Linear(
                input_dim + params.vp_inner_dim, input_dim, bias=False
            )

            enc_cls, enc_params_cls = TTS_ENCODERS[params.word_encoder_type]
            self.word_encoder = _init_encoder(
                enc_cls, enc_params_cls, params.word_encoder_params, input_dim
            )

            self.hard_lr = SoftLengthRegulator(hard=True)
        else:
            self.prenet = nn.Identity()

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
                enc_cls, enc_params_cls, params.token_encoder_params, 2 * emb_dim, emb_dim
            )
            self.mtm_proj = Regression(params.vp_inner_dim, emb_dim, activation_fn="Tanh")

            input_dim += emb_dim
        else:
            self.mtm_encoder = None

        self.token_encoder = _init_encoder(
            enc_cls, enc_params_cls, params.token_encoder_params, input_dim
        )
        self.length_regulator = SoftLengthRegulator()

    @property
    def output_dim(self):
        return self.params.vp_output_dim

    def postprocessing(self, predict):
        if self.params.variance_params.log_scale:  # type: ignore
            predict = torch.expm1(predict)

        return predict

    def compute_loss(
        self, name, predict, target, lengths, global_step: int = 0
    ) -> tp.Dict[str, torch.Tensor]:
        if self.params.variance_params.log_scale:  # type: ignore
            target = torch.log1p(target)

        target = apply_mask(target, get_mask_from_lengths(lengths))
        predict = apply_mask(predict, get_mask_from_lengths(lengths))

        loss_fn = getattr(F, self.params.loss_type)
        if self.params.loss_type == "cross_entropy":
            loss_fn = partial(loss_fn, ignore_index=-1)
            predict = predict.transpose(1, -1)

        loss = self.params.loss_alpha * loss_fn(predict, target)
        return {f"{name}_{self.params.loss_type}_by_tokens": loss}

    def forward_step(
        self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        name = kwargs.get("name")
        target_by_tokens = kwargs.get("target")
        if target_by_tokens is not None:
            target_by_tokens = target_by_tokens.squeeze(-1)

        losses = {}

        word_length = model_inputs.additional_inputs.get("word_lengths")
        word_inv_lengths = model_inputs.additional_inputs.get("word_invert_lengths")

        max_num_tokens = model_inputs.transcription_lengths.max().item()
        max_num_words = model_inputs.num_words.max().item()

        if self.params.add_lm_feat:
            x_by_words, _ = self.length_regulator(
                x.detach(), word_inv_lengths, max_num_words
            )
            x_by_words = x_by_words + self.lm_proj(model_inputs.lm_feat)
            wd_predict, wd_ctx = self.word_encoder.process_content(
                x_by_words, model_inputs.num_words, model_inputs
            )
            wd_ctx, _ = self.hard_lr(wd_ctx, word_length, max_num_tokens)
            x_by_tokens = torch.cat([x, wd_ctx], dim=2)
        else:
            x_by_tokens = x

        if self.mtm_encoder is not None and target_by_tokens is not None:
            m = model_inputs.masks["text"]
            inv_m = ~m.unsqueeze(-1)

            mtm_target_embs = self.mtm_embeddings(target_by_tokens)
            mtm_target_embs_mask = mtm_target_embs * m.unsqueeze(-1) - 1 * inv_m

            x_proj = self.mtm_pre_proj(x_by_tokens.detach())
            mtm_x = torch.cat([mtm_target_embs_mask, x_proj], dim=-1)
            _, mtm_enc_ctx = self.mtm_encoder.process_content(
                mtm_x, x_lengths, model_inputs
            )
            mtm_predict_embs = self.mtm_proj(mtm_enc_ctx)

            if self.training:
                losses[f"{name}_mtm_loss_by_tokens"] = F.mse_loss(
                    mtm_predict_embs * inv_m, (mtm_target_embs * inv_m).detach()
                )

            merger_embs = mtm_target_embs * m.unsqueeze(-1) + mtm_predict_embs * inv_m
            x_by_tokens = torch.cat([x_by_tokens, merger_embs.detach()], dim=-1)

        tk_proj = self.prenet(x_by_tokens)
        predict, enc_ctx = self.token_encoder.process_content(
            tk_proj, x_lengths, model_inputs
        )
        predict = predict.squeeze(-1)

        if self.training:
            losses.update(
                self.compute_loss(
                    name, predict, target_by_tokens, x_lengths, model_inputs.global_step
                )
            )

        predict = self.postprocessing(predict)

        return (
            predict,
            {
                f"{name}_vp_context": enc_ctx,
                f"{name}_vp_predict": predict,
                f"{name}_vp_target": kwargs.get("target"),
            },
            losses,
        )


class TokenLevelPredictorWithDiscriminatorParams(TokenLevelPredictorParams):
    pass


class TokenLevelPredictorWithDiscriminator(TokenLevelPredictor, Component):
    params: TokenLevelPredictorWithDiscriminatorParams

    def __init__(
        self,
        params: TokenLevelPredictorWithDiscriminatorParams,
        input_dim: tp.Union[int, tp.Tuple[int, ...]],
    ):
        if params.vp_output_dim != 1:
            raise ValueError("Feature dimension must be equal to 1")

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
            mask = get_mask_from_lengths(x_lengths).unsqueeze(-1)

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
                {f"{name}_{k}_by_tokens": v for k, v in disc_losses.items()}
            )

        return var_predict, var_content, var_losses
