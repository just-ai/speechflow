import math
import typing as tp

import torch

from torch.nn import functional as F

from speechflow.utils.tensor_utils import (
    apply_mask,
    get_attention_mask,
    get_mask_from_lengths,
)
from tts.acoustic_models.modules.embedding_calculator import EmbeddingCalculator
from tts.acoustic_models.modules.params import EmbeddingParams
from tts.forced_alignment.data_types import AlignerForwardInput, AlignerForwardOutput
from tts.forced_alignment.model.blocks import (
    AlignmentEncoder,
    Encoder,
    FlowSpecDecoder,
    TextEncoder,
)
from tts.forced_alignment.model.utils import (
    binarize_attention,
    generate_path,
    maximum_path,
)

__all__ = ["GlowTTS", "GlowTTSParams"]


class GlowTTSParams(EmbeddingParams):
    """GlowTTS model parameters."""

    flow_type: str = "GlowTTS"  # GlowTTS

    inner_channels_enc: int = 192
    inner_channels_dec: int = 192

    filter_channels: int = 768
    filter_channels_dp: int = 256

    kernel_size_enc: int = 3
    kernel_size_dec: int = 5

    n_layers_enc: int = 6
    n_heads_enc: int = 2
    n_blocks_dec: int = 12
    n_layers_dec: int = 4

    window_size: int = 4
    n_split: int = 4
    n_sqz: int = 2
    dilation_rate: int = 1
    p_dropout: float = 0.05

    use_ling_feat_emb: bool = False
    use_lang_emb: bool = False
    use_speaker_emb: bool = False
    use_speech_quality_emb: bool = False
    use_xpbert: bool = False

    # Nemo TTS Alignment (debugging required)
    use_alignment_encoder: bool = False
    alignment_encoder_n_att_channels: int = 128
    alignment_encoder_temperature: float = 0.0005
    alignment_encoder_dist_type: str = "l2"

    # For adjust attention
    frames_per_sec: float = 125  # 16000 / 128
    max_phoneme_duration: float = 0.15

    def model_post_init(self, __context: tp.Any):
        super().model_post_init(__context)


class GlowTTS(EmbeddingCalculator):
    params: GlowTTSParams

    def __init__(self, cfg: tp.Union[GlowTTSParams, dict], strict_init: bool = True):
        super().__init__(GlowTTSParams.create(cfg, strict_init))
        params = self.params

        self.n_sqz = params.n_sqz

        self.encoder = TextEncoder(
            params.token_emb_dim,
            params.mel_spectrogram_dim,
            params.inner_channels_enc,
            params.filter_channels,
            params.filter_channels_dp,
            params.n_heads_enc,
            params.n_layers_enc,
            params.kernel_size_enc,
            params.p_dropout,
            ling_feat_dim=params.token_emb_dim if params.use_ling_feat_emb else None,
            lang_emb_dim=params.token_emb_dim if params.use_lang_emb else None,
            speaker_emb_dim=params.speaker_emb_dim if params.use_speaker_emb else None,
            window_size=params.window_size,
            use_prenet=True,
            use_xpbert=params.use_xpbert,
        )

        if params.flow_type == "GlowTTS":
            self.decoder = FlowSpecDecoder(
                params.mel_spectrogram_dim,
                params.inner_channels_dec,
                params.kernel_size_dec,
                params.dilation_rate,
                params.n_blocks_dec,
                params.n_layers_dec,
                p_dropout=params.p_dropout,
                n_split=params.n_split,
                n_sqz=params.n_sqz,
                lang_emb_dim=params.token_emb_dim if params.use_lang_emb else None,
                speaker_emb_dim=params.speaker_emb_dim
                if params.use_speaker_emb
                else None,
                speech_quality_emb_dim=params.speech_quality_emb_dim
                if params.use_speech_quality_emb
                else None,
            )
        else:
            raise NotImplementedError(f"'{params.flow_type}' not implemented.")

        if params.use_alignment_encoder:
            self.alignment_encoder = AlignmentEncoder(
                n_mel_channels=params.mel_spectrogram_dim,
                n_text_channels=params.inner_channels_enc,
                n_att_channels=params.alignment_encoder_n_att_channels,
                temperature=params.alignment_encoder_temperature,
                dist_type=params.alignment_encoder_dist_type,
            )
        else:
            self.alignment_encoder = None

    def preprocess(self, y, y_lengths, y_max_length=None):
        if y_max_length is not None:
            y_max_length = (
                torch.div(y_max_length, self.n_sqz, rounding_mode="trunc") * self.n_sqz
            )
            y = y[:, :, :y_max_length]

        y_lengths = torch.div(y_lengths, self.n_sqz, rounding_mode="trunc") * self.n_sqz
        return y, y_lengths

    def store_inverse(self):
        self.decoder.store_inverse()

    def mas(self, x_m, x_logs, z, attn_mask, inputs, adjust_attention: bool = False):
        with torch.no_grad():
            x_s_sq_r = torch.exp(-2 * x_logs)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(
                -1
            )  # [b, t, 1]
            logp2 = torch.matmul(
                x_s_sq_r.transpose(1, 2), -0.5 * (z**2)
            )  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul(
                (x_m * x_s_sq_r).transpose(1, 2), z
            )  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (x_m**2) * x_s_sq_r, [1]).unsqueeze(
                -1
            )  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']

            if adjust_attention:
                sil_mask = inputs.ling_feat.sil_mask.cpu().numpy()
                spectral_flatness = inputs.spectral_flatness.cpu().numpy()
                max_frames_per_phoneme = int(
                    self.params.frames_per_sec * self.params.max_phoneme_duration
                )
            else:
                sil_mask = spectral_flatness = max_frames_per_phoneme = None

            attn = maximum_path(
                logp,
                attn_mask.squeeze(1),
                sil_mask=sil_mask,
                spectral_flatness=spectral_flatness,
                max_frames_per_phoneme=max_frames_per_phoneme,
            )

        return attn.unsqueeze(1).detach()

    def calculate_losses(
        self, z, attn, x_m, x_logs, logw, logdet, x_mask, x_lens, y_lens
    ):
        # [b, t', t], [b, t, d] -> [b, d, t']
        z_m = torch.matmul(
            attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
        ).transpose(1, 2)
        # [b, t', t], [b, t, d] -> [b, d, t']
        z_logs = torch.matmul(
            attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
        ).transpose(1, 2)
        logw_ = apply_mask(torch.log(1e-8 + torch.sum(attn, -1)), x_mask)

        mle_loss = 0.5 * math.log(2 * math.pi) + (
            torch.sum(z_logs)
            + 0.5 * torch.sum(torch.exp(-2 * z_logs) * (z - z_m) ** 2)
            - torch.sum(logdet)
        ) / (
            torch.sum(torch.div(y_lens, self.n_sqz, rounding_mode="trunc"))
            * self.n_sqz
            * self.params.mel_spectrogram_dim
        )
        duration_loss = torch.sum((logw - logw_) ** 2) / torch.sum(x_lens)
        return mle_loss, duration_loss

    def forward(self, inputs: AlignerForwardInput, adjust_attention: bool = False) -> AlignerForwardOutput:  # type: ignore
        x = self.get_transcription_embeddings(inputs)  # type: ignore
        x_lens = inputs.input_lengths
        x_mask = get_mask_from_lengths(x_lens)

        y = inputs.spectrogram.transpose(1, -1)
        y_lens = inputs.spectrogram_lengths
        y, y_lens = self.preprocess(y, y_lens, y.shape[2])
        y_mask = get_mask_from_lengths(y_lens)

        ling_feat_emb = self.get_ling_feat(inputs)  # type: ignore
        lang_emb = self.get_lang_embedding(inputs)  # type: ignore
        speaker_emb = self.get_speaker_embedding(inputs)  # type: ignore
        speech_quality_emb = self.get_speech_quality_embedding(inputs)  # type: ignore

        x, x_m, x_logs, logw = self.encoder(
            x,
            x_mask,
            ling_feat_emb=ling_feat_emb,
            lang_emb=lang_emb,
            speaker_emb=speaker_emb,
        )

        z, logdet = self.decoder(
            y,
            y_mask,
            lang_emb=lang_emb,
            speaker_emb=speaker_emb,
            speech_quality_emb=speech_quality_emb,
        )

        additional_content = {}

        attn_mask = get_attention_mask(x_mask, y_mask)
        attn = self.mas(x_m, x_logs, z, attn_mask, inputs, adjust_attention)

        mle_loss, duration_loss = self.calculate_losses(
            z, attn, x_m, x_logs, logw, logdet, x_mask, x_lens, y_lens
        )

        if self.alignment_encoder is not None:
            attn_soft, attn_logprob = self.alignment_encoder(
                queries=z,
                keys=x,
                mask=(~x_mask).unsqueeze(1).transpose(2, 1),
                attn_prior=attn.squeeze(1).transpose(1, 2),
            )

            try:
                attn_hard = binarize_attention(attn_soft, inputs.input_lengths, y_lens)
                aligning_path = attn_hard.squeeze(1)
            except Exception:  # type: ignore
                aligning_path = attn.squeeze(1).transpose(1, 2)

            additional_content["attn_soft"] = attn_soft
            additional_content["attn_logprob"] = attn_logprob
        else:
            aligning_path = attn.squeeze(1).transpose(1, 2)

        output = AlignerForwardOutput(
            aligning_path=aligning_path,
            mle_loss=mle_loss,
            duration_loss=duration_loss,
            additional_content=additional_content,
            output_lengths=y_lens,
        )
        return output

    @torch.no_grad()
    def generate(self, inputs: AlignerForwardInput):  # type: ignore
        assert inputs.ling_feat
        x = inputs.transcription
        x_lens = inputs.input_lengths
        y = inputs.spectrogram.transpose(1, -1)

        speaker_emb = self.get_speaker_embedding(inputs)  # type: ignore
        ling_feat_emb = self.get_ling_feat(inputs)  # type: ignore

        x_m, x_logs, logw, x_mask = self.encoder(x, ling_feat_emb, x_lens, g=speaker_emb)

        w = apply_mask(torch.exp(logw), x_mask)
        w_ceil = torch.ceil(w)
        y_lens = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()

        y, y_lens = self.preprocess(y, y_lens)
        z_mask = get_mask_from_lengths(y_lens).unsqueeze(1)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        z_m = torch.matmul(
            attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
        ).transpose(1, 2)
        z_logs = torch.matmul(
            attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
        ).transpose(1, 2)

        z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m)) * z_mask
        y, logdet = self.decoder(z, z_mask, reverse=True, g=speaker_emb)

        output = AlignerForwardOutput(
            spectrogram=y.transpose(1, 2),
            aligning_path=attn.squeeze(1).transpose(1, 2),
            output_mask=torch.LongTensor([[y.shape[2]]]),
        )
        return output
