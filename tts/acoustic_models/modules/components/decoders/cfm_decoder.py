import math
import typing as tp

import torch

from pydantic import Field
from torch import nn

from speechflow.utils.tensor_utils import get_lengths_from_mask
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.common.conditional_layers import ConditionalLayer
from tts.acoustic_models.modules.common.stable_tts.flow_matching import (
    BaseCFM,
    BaseEstimator,
)
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.components.decoders.wrapper_decoder import (
    WrapperDecoder,
    WrapperDecoderParams,
)
from tts.acoustic_models.modules.data_types import (
    ComponentInput,
    DecoderOutput,
    VarianceAdaptorOutput,
)
from tts.acoustic_models.modules.params import DecoderParams

__all__ = [
    "CFMDecoder",
    "CFMDecoderParams",
]


class CFMEstimatorParams(WrapperDecoderParams):
    filter_channels: int = 1024


class CFMEstimator(WrapperDecoder, BaseEstimator):
    def __init__(self, params: CFMEstimatorParams, input_dim):
        super().__init__(params, input_dim)
        inner_dim = params.decoder_inner_dim
        self.time_embeddings = SinusoidalPosEmb(inner_dim)
        self.time_proj = TimestepEmbedding(inner_dim, inner_dim, params.filter_channels)
        self.time_cond_layer = ConditionalLayer("FiLM", inner_dim, inner_dim)

    def hook_update_content(
        self, x: torch.Tensor, x_lengths: torch.Tensor, inputs: ComponentInput
    ) -> torch.Tensor:
        return self.time_cond_layer(x, inputs.additional_content["time_emb"])

    def hook_update_condition(
        self, c: torch.Tensor, inputs: ComponentInput
    ) -> torch.Tensor:
        return inputs.additional_content.get("cfg_condition_masked", c)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        mu_mask: torch.Tensor,
        t: torch.Tensor,
        inputs: ComponentInput,
    ):
        mu_lens = get_lengths_from_mask(mu_mask)
        inputs.additional_content["time_emb"] = self.time_proj(self.time_embeddings(t))
        inputs.set_content(torch.cat([x, mu], dim=-1), mu_lens)
        outputs = super().forward_step(inputs)
        return outputs.get_content()[0]


class CFMDecoderParams(DecoderParams):
    # estimator params
    estimator_type: str = "DiTEncoder"
    estimator_params: dict = Field(default_factory=lambda: {})

    # prior params
    use_prior_decoder: bool = False
    prior_decoder_type: str = "RNNEncoder"
    prior_decoder_params: dict = Field(default_factory=lambda: {})

    use_cfg: bool = False
    cfg_p_dropout: float = 0.1
    cfg_guidance_scale: float = 0.5

    # condition
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0

    # cfm params
    cfm_n_timesteps: int = 30
    cfm_temperature: float = 0.67
    cfm_sigma_min: float = 1e-4


class CFMDecoder(Component):
    """Decoder from Matcha-TTS paper https://browse.arxiv.org/pdf/2309.03199.pdf."""

    params: CFMDecoderParams

    def __init__(self, params: CFMDecoderParams, input_dim):
        super().__init__(params, input_dim)

        est_params = CFMEstimatorParams.init_from_parent_params(params)
        est_params.base_decoder_type = params.estimator_type
        est_params.base_decoder_params = params.estimator_params
        est_params["condition"] = params.condition
        est_params["condition_dim"] = params.condition_dim

        estimator = CFMEstimator(est_params, 2 * params.decoder_output_dim)
        self.decoder = BaseCFM(estimator, params.cfm_sigma_min)

        if params.use_prior_decoder:
            prior_params = WrapperDecoderParams.init_from_parent_params(params)
            prior_params.base_decoder_type = params.prior_decoder_type
            prior_params.base_decoder_params = params.prior_decoder_params

            if "decoder_num_layers" in params.prior_decoder_params:
                prior_params.decoder_num_layers = params.prior_decoder_params.pop(
                    "decoder_num_layers"
                )
            if "decoder_inner_dim" in params.prior_decoder_params:
                prior_params.decoder_inner_dim = params.prior_decoder_params.pop(
                    "decoder_inner_dim"
                )

            prior_params.base_decoder_params["condition"] = params.condition
            prior_params.base_decoder_params["condition_dim"] = params.condition_dim
            self.prior_decoder = WrapperDecoder(prior_params, input_dim)
        else:
            self.prior_decoder = Regression(input_dim, params.decoder_output_dim)

        if self.params.use_cfg:
            self.fake_content = torch.nn.Parameter(
                torch.zeros(1, 1, params.decoder_output_dim)
            )
            self.fake_condition = torch.nn.Parameter(torch.zeros(1, params.condition_dim))
        else:
            self.fake_content = self.fake_condition = None

    @property
    def output_dim(self):
        return self.params.decoder_output_dim

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        x, x_lens, x_mask = inputs.get_content_and_mask()
        y = getattr(inputs.model_inputs, self.params.target)

        if self.params.use_prior_decoder:
            mu, _ = self.prior_decoder.process_content(x, x_lens, inputs.model_inputs)
        else:
            mu = self.prior_decoder(x)

        if self.params.use_cfg:
            # CFG
            # mask content information for better diversity for flow-matching, separate masking for speaker and content

            cfg_rand = torch.rand(y.shape[0], 1, device=y.device)
            cfg_mask_mu = (cfg_rand > self.params.cfg_p_dropout * 2) | (
                cfg_rand < self.params.cfg_p_dropout
            )
            cfg_mask_cond = cfg_rand > self.params.cfg_p_dropout

            cfg_mask_mu = cfg_mask_mu.unsqueeze(-1)
            mu_masked = mu * cfg_mask_mu + ~cfg_mask_mu * self.fake_content.repeat(
                y.shape[0], y.shape[1], 1
            )

            c = self.get_condition(inputs, self.params.condition)
            cfg_mask_cond = cfg_mask_cond.unsqueeze(-1)
            c_masked = c * cfg_mask_cond + ~cfg_mask_cond * self.fake_condition.repeat(
                y.shape[0], 1, 1
            )
            inputs.additional_content["cfg_condition_masked"] = c_masked
        else:
            mu_masked = mu

        # pad = 16 - mu.shape[1] % 16
        # if pad != 16:
        #     mu = F.pad(mu.transpose(2, 1), (0, pad), value=-4).transpose(1, 2)
        #     x_mask = F.pad(x_mask | True, (0, pad), value=True)
        #     y = F.pad(y.transpose(2, 1), (0, pad), value=-4).transpose(1, 2)

        cfm_loss, _ = self.decoder.compute_loss(
            mu=mu_masked,
            mu_mask=x_mask,
            target=y,
            inputs=inputs,
        )
        inputs.additional_losses["cfm_loss"] = cfm_loss

        # if pad != 16:
        #     mu = mu[:, :-pad, :]

        return DecoderOutput.copy_from(inputs).set_content(mu, x_lens)

    def inference_step(self, inputs: VarianceAdaptorOutput, **kwargs) -> DecoderOutput:  # type: ignore
        x, x_lens, x_mask = inputs.get_content_and_mask()

        if self.params.use_prior_decoder:
            mu, _ = self.prior_decoder.process_content(x, x_lens, inputs.model_inputs)
        else:
            mu = self.prior_decoder(x)

        # pad = 16 - mu.shape[1] % 16
        # if pad != 16:
        #     mu = F.pad(mu.transpose(2, 1), (0, pad), value=-4).transpose(1, 2)
        #     x_mask = F.pad(x_mask | True, (0, pad), value=True)
        #     y = F.pad(y.transpose(2, 1), (0, pad), value=-4).transpose(1, 2)

        if self.params.use_cfg:
            decoder_outputs = self.decoder(
                mu=mu,
                mu_mask=x_mask,
                inputs=inputs,
                n_timesteps=self.params.cfm_n_timesteps,
                temperature=self.params.cfm_temperature,
                guidance_scale=self.params.cfg_guidance_scale,
                fake_content=self.fake_content,
                fake_condition=self.fake_condition,
            )
        else:
            decoder_outputs = self.decoder(
                mu=mu,
                mu_mask=x_mask,
                inputs=inputs,
                n_timesteps=self.params.cfm_n_timesteps,
                temperature=self.params.cfm_temperature,
            )

        # if pad != 16:
        #     decoder_outputs = decoder_outputs[:, :-pad, :]

        outputs = DecoderOutput.copy_from(inputs).set_content(decoder_outputs, x_lens)
        return outputs


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_channels, filter_channels),
            nn.SiLU(),
            nn.Linear(filter_channels, out_channels),
        )

    def forward(self, x):
        return self.layer(x)
