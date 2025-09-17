import typing as tp

import torch

from pydantic import Field
from torch import nn
from torch.nn import functional as F
from vector_quantize_pytorch import ResidualFSQ

from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import MODEL_INPUT_TYPE
from tts.acoustic_models.modules.params import EmbeddingParams

__all__ = ["StyleEncoder", "StyleEncoderParams"]


class StyleEncoderParams(EmbeddingParams):
    base_encoder_type: tp.Literal["SimpleStyle", "StyleSpeech"] = "SimpleStyle"
    base_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    source: tp.Optional[str] = "spectrogram"
    source_dim: int = 80
    style_emb_dim: int = 128
    random_chunk: bool = False
    min_spec_len: int = 256
    max_spec_len: int = 512
    use_gmvae: bool = False
    gmvae_n_components: int = 16
    use_fsq: bool = False
    fsq_levels: tp.Tuple[int, ...] = (
        8,
        5,
        5,
    )  # 2^12 [https://arxiv.org/pdf/2309.15505.pdf]
    fsq_num_quantizers: int = 1


class StyleEncoder(Component):
    params: StyleEncoderParams

    def __init__(self, params: StyleEncoderParams, input_dim: int):
        super().__init__(params, input_dim)
        from tts.acoustic_models.modules.components import style_encoders

        enc_cls = getattr(style_encoders, params.base_encoder_type)
        enc_params_cls = getattr(style_encoders, f"{params.base_encoder_type}Params")
        enc_params = enc_params_cls.init_from_parent_params(
            params, params.base_encoder_params
        )

        self.encoder = enc_cls(enc_params, params.source_dim or input_dim)

        if params.use_gmvae:
            self.gmvae = GMVAE(
                self.encoder.output_dim,
                self.params.style_emb_dim,
                self.params.gmvae_n_components,
            )
        elif params.use_fsq:
            self.residual_fsq = ResidualFSQ(
                dim=self.encoder.output_dim,
                levels=list(params.fsq_levels),
                num_quantizers=params.fsq_num_quantizers,
            )

    @property
    def output_dim(self):
        if self.params.use_gmvae:
            return self.params.style_emb_dim
        else:
            return self.encoder.output_dim

    def encode(self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs):
        if x.shape[1] > 1:
            x, x_lengths = self.get_chunk(
                x, x_lengths, self.params.min_spec_len, self.params.max_spec_len
            )

        style_emb, _, _ = self.encoder(x, x_lengths, model_inputs, **kwargs)

        if self.params.use_gmvae:
            gmvae_emb, content, losses = self.gmvae(
                style_emb, sigma_multiplier=kwargs.get("sigma_multiplier", 0.0)
            )
            if losses:
                losses = {
                    f"kl_loss_{kwargs.get('name', f'gmvae{self.id}')}_{k}": v
                    for k, v in losses.items()
                }
            return gmvae_emb, content, losses

        elif self.params.use_fsq:
            quantized_style, indices = self.residual_fsq(style_emb)

            # set_random_seed(get_seed())
            # indices = indices * 0 + random.randint(0, self.residual_fsq.codebook_size)
            # quantized_style = self.residual_fsq.get_output_from_indices(indices)
            # print(indices)

            return quantized_style, {}, {}
        else:
            return style_emb, {}, {}

    def forward_step(self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs):
        if model_inputs.prosody_reference is not None:
            if "style_emb" in model_inputs.prosody_reference.default.model_feats:
                style_emb = model_inputs.prosody_reference.default.get_model_feat(
                    "style_emb", x.shape, x.device
                )
                return style_emb, {}, {}

        if self.params.source is not None:
            x = self.get_condition(
                model_inputs, self.params.source, average_by_time=False
            )

        if self.params.base_encoder_type == "SimpleStyle":
            assert x.shape[1] == 1, ValueError(
                "This style coder requires a biometric embedding."
            )
            x_lengths = None
        else:
            assert x.shape[1] > 1, ValueError(
                "This style coder requires a mel spectrogram."
            )

            x_lengths = model_inputs.get_feat_lengths(self.params.source)
            if x_lengths is None:
                x_lengths = torch.LongTensor([x.shape[1]] * x.shape[0]).to(x.device)

        if self.training and self.params.random_chunk and x.shape[1] > 1:
            chunk, chunk_lengths = self.get_random_chunk(
                x, x_lengths, self.params.min_spec_len, self.params.max_spec_len
            )
        else:
            chunk = x
            chunk_lengths = x_lengths

        return self.encode(chunk, chunk_lengths, model_inputs, **kwargs)


class GMVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        gmvae_n_components: int,
    ):
        super().__init__()

        self.mean_posteriors = nn.Linear(input_dim, latent_dim)
        self.logvar_posteriors = nn.Linear(input_dim, latent_dim)

        self.mean_priors = nn.Embedding(gmvae_n_components, latent_dim)
        self.logvar_priors = nn.Embedding(gmvae_n_components, latent_dim)
        nn.init.uniform_(self.mean_priors.weight, -2, 2)
        nn.init.constant_(self.logvar_priors.weight, -1.0)

    @staticmethod
    def _gm_loss(
        mean_priors,
        logvar_priors,
        mean_posteriors,
        logvar_posteriors,
        posterior_sample,
    ):
        n_components, dim = mean_priors.shape

        gm_prior = torch.distributions.Normal(
            loc=mean_priors.unsqueeze(0),
            scale=(logvar_priors / 2.0).exp().unsqueeze(0),
        )
        gm_posterior = torch.distributions.Normal(
            loc=mean_posteriors.unsqueeze(1),
            scale=(logvar_posteriors / 2.0).exp().unsqueeze(1),
        )

        sample = posterior_sample

        cat_prior = torch.distributions.Categorical(
            probs=torch.ones(1, n_components, device=sample.device) / n_components
        )
        cat_posterior = torch.distributions.Categorical(
            logits=F.log_softmax(gm_prior.log_prob(sample).sum(dim=-1), dim=-1)
        )

        gm_kldiv = torch.distributions.kl_divergence(gm_posterior, gm_prior)
        gm_kldiv = (gm_kldiv * cat_posterior.logits.exp().unsqueeze(-1)).mean(dim=0).sum()
        cat_kldiv = torch.distributions.kl_divergence(cat_posterior, cat_prior).mean(
            dim=0
        )
        return gm_kldiv, cat_kldiv

    def forward(
        self,
        x: torch.FloatTensor,
        sigma_multiplier: float = 0.0,
    ):
        losses = {}

        if x is not None and sigma_multiplier == 0.0:
            mean_posteriors = self.mean_posteriors(x)
            logvar_posteriors = self.logvar_posteriors(x)

            if self.training:
                std = logvar_posteriors.mul(0.5).exp_()
                eps = torch.empty_like(std).normal_()
                z = eps.mul(std).add_(mean_posteriors)
            else:
                z = mean_posteriors

            if self.training:
                kl_loss_gm, kl_loss_cat = self._gm_loss(
                    mean_priors=self.mean_priors.weight,
                    logvar_priors=self.logvar_priors.weight,
                    mean_posteriors=mean_posteriors,
                    logvar_posteriors=logvar_posteriors,
                    posterior_sample=z,
                )
                losses = {"gm": kl_loss_gm, "cat": kl_loss_cat}
        else:
            component_idx = torch.randint(
                0, self.mean_priors.num_embeddings, (1,), dtype=torch.long
            ).to(x.device)
            p_dist = torch.distributions.Normal(
                self.mean_priors(component_idx).cpu(),
                F.relu(
                    self.logvar_priors(component_idx).mul(0.5).exp_() * sigma_multiplier
                ).cpu(),
            )
            z = p_dist.rsample().to(x.device)

        return z, {}, losses
