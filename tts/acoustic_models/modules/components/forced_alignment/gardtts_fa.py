import math
import typing as tp

import torch

from torch.nn import functional as F

from speechflow.utils.tensor_utils import apply_mask, get_mask_from_lengths
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import MODEL_INPUT_TYPE
from tts.acoustic_models.modules.params import VariancePredictorParams
from tts.forced_alignment.model.utils import maximum_path

__all__ = ["GradTTSFA", "GradTTSFAParams"]


class GradTTSFAParams(VariancePredictorParams):
    text_encoder_type: str = "RNNEncoder"
    text_encoder_params: dict = None  # type: ignore
    audio_feat: str = "spectrogram"
    audio_feat_dim: int = 80
    dp_filter_channels_dp: int = 256
    dp_kernel_size: int = 3
    dp_p_dropout: float = 0.1

    def model_post_init(self, __context: tp.Any):
        if self.text_encoder_params is None:
            self.text_encoder_params = {}

        self.text_encoder_params.setdefault("encoder_num_layers", 2)
        self.text_encoder_params.setdefault("encoder_hidden_dim", 256)


class LayerNorm(torch.nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class DurationPredictor(torch.nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = apply_mask(self.conv_1(x), x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = apply_mask(self.conv_2(x), x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x)
        return x


class GradTTSFA(Component):
    params: GradTTSFAParams

    def __init__(
        self, params: GradTTSFAParams, input_dim: tp.Union[int, tp.Tuple[int, ...]]
    ):
        super().__init__(params, input_dim)

        from tts.acoustic_models.modules import TTS_ENCODERS

        enc_cls, params_cls = TTS_ENCODERS[params.text_encoder_type]
        enc_params = params_cls.init_from_parent_params(
            params, params.text_encoder_params
        )

        enc_params.encoder_output_dim = params.audio_feat_dim
        self.encoder = enc_cls(enc_params, input_dim)

        self.proj_w = DurationPredictor(
            enc_params.encoder_inner_dim + enc_params.condition_dim,
            params.dp_filter_channels_dp,
            params.dp_kernel_size,
            params.dp_p_dropout,
        )

    @property
    def output_dim(self):
        return 1

    def _get_target_feat(self, inputs):
        if hasattr(inputs, self.params.audio_feat):
            return getattr(inputs, self.params.audio_feat)
        else:
            return inputs.additional_inputs.get(self.params.audio_feat)

    def mas(self, mu_x, y, attn_mask):
        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.params.audio_feat_dim
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y**2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = maximum_path(log_prior, attn_mask.squeeze(1))

        return attn.detach()

    def forward_step(self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs):
        y = self._get_target_feat(model_inputs)

        x_enc = self.encoder.encode(x, x_lengths, model_inputs)
        if hasattr(self.encoder, "proj"):
            mu_x = self.encoder.proj(x_enc)
        else:
            mu_x = x_enc

        g = self.get_condition(model_inputs, self.encoder.params.condition)
        g = g.unsqueeze(1).expand(-1, x_enc.size(1), -1)
        logw = self.proj_w(torch.cat([x_enc, g], dim=2).transpose(2, 1), x_lengths)

        if y is not None:
            x_mask = get_mask_from_lengths(x_lengths)
            y_lengths = model_inputs.output_lengths
            y_mask = get_mask_from_lengths(y_lengths)

            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(1)
            attn = self.mas(mu_x.transpose(2, 1), y.transpose(2, 1), attn_mask)
            dura = torch.sum(attn.unsqueeze(1), -1).squeeze(1) * x_mask

            logw_ = torch.log(dura + 1e-8) * x_mask
            dura_loss = F.l1_loss(logw.squeeze(1), logw_)

            # Align encoded text with mel-spectrogram and get mu_y segment
            mu_y = torch.matmul(attn.transpose(1, 2), mu_x)

            prior_loss = torch.sum(
                0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask.unsqueeze(2)
            )
            prior_loss = prior_loss / (torch.sum(y_mask) * self.params.audio_feat_dim)

            return (
                dura.detach() if self.training else torch.exp(logw),
                {
                    "fa_prior": mu_y,
                    "fa_attn": attn,
                    "fa_durations": dura.detach(),
                    "fa_prediction": dura,
                },
                {"fa_dura_loss": dura_loss, "fa_prior_loss": prior_loss},
            )
        else:
            dura = torch.exp(logw)
            return torch.exp(logw), {}, {"fa_prediction": dura}
