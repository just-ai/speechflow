import math
import typing as tp

import torch

from einops import rearrange
from torch import nn
from transformers import AutoModel

from speechflow.utils.tensor_utils import apply_mask, get_attention_mask
from tts.forced_alignment.model.layers import (
    FFN,
    WN,
    ActNorm,
    ConditionalInput,
    ConvNorm,
    ConvReluNorm,
    InvConvNear,
    LayerNorm,
    MultiHeadAttention,
)
from tts.forced_alignment.model.utils import (
    binarize_attention_parallel,
    squeeze,
    unsqueeze,
)

__all__ = [
    "Encoder",
    "TextEncoder",
    "FlowSpecDecoder",
    "AlignmentEncoder",
]


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0,
        window_size: tp.Optional[int] = None,
        block_length: tp.Optional[int] = None,
    ):
        super().__init__()
        self.n_layers = n_layers

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_dim,
                    hidden_dim,
                    n_heads,
                    window_size=window_size,
                    p_dropout=p_dropout,
                    block_length=block_length,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_dim))
            self.ffn_layers.append(
                FFN(
                    hidden_dim,
                    hidden_dim,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_dim))

    def forward(self, x, x_mask, attention_mask=None):
        if attention_mask is None:
            attention_mask = get_attention_mask(x_mask, x_mask)
        elif attention_mask.ndim == 3:
            attention_mask = attention_mask.unsqueeze(1)

        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attention_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)

        return apply_mask(x, x_mask)


class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels: int, filter_channels: int, kernel_size: int, p_dropout: float
    ):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x)
        return apply_mask(x, x_mask)


class TextEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        outputs_dim: int,
        hidden_dim: int,
        filter_channels: int,
        filter_channels_dp: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        ling_feat_dim: tp.Optional[int] = None,
        lang_emb_dim: tp.Optional[int] = None,
        speaker_emb_dim: tp.Optional[int] = None,
        window_size: tp.Optional[int] = None,
        block_length: tp.Optional[int] = None,
        mean_only: bool = False,
        use_prenet: bool = False,
        use_xpbert: bool = False,
        xpbert_model: str = "vinai/xphonebert-base",
    ):
        super().__init__()
        self.mean_only = mean_only
        self.use_xpbert = use_xpbert

        if use_xpbert:
            self.encoder = AutoModel.from_pretrained(xpbert_model)
            self.pe = self.encoder.embeddings
            self.hidden_dim = self.encoder.config.hidden_size
        else:
            self.encoder = Encoder(
                hidden_dim,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                window_size=window_size,
                block_length=block_length,
            )
            self.hidden_dim = hidden_dim

        self.embedding_proj = nn.Linear(embedding_dim, self.hidden_dim)

        cond_dim = 0
        if ling_feat_dim is not None:
            self.ling_feat_proj = nn.Sequential(
                nn.Linear(ling_feat_dim, embedding_dim), nn.Tanh()
            )
            cond_dim += embedding_dim
        else:
            self.ling_feat_proj = None

        if lang_emb_dim is not None:
            self.lang_emb_proj = nn.Sequential(
                nn.Linear(lang_emb_dim, embedding_dim), nn.Tanh()
            )
            cond_dim += embedding_dim
        else:
            self.lang_emb_proj = None

        if speaker_emb_dim is not None:
            self.speaker_emb_proj = nn.Sequential(
                nn.Linear(speaker_emb_dim, embedding_dim), nn.Tanh()
            )
            cond_dim += embedding_dim
        else:
            self.speaker_emb_proj = None

        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, self.hidden_dim)
        else:
            self.cond_proj = None

        if use_prenet:
            self.prenet = ConvReluNorm(
                self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.1,
            )
        else:
            self.prenet = None

        self.proj_m = nn.Conv1d(self.hidden_dim, outputs_dim, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(self.hidden_dim, outputs_dim, 1)

        self.proj_w = DurationPredictor(
            self.hidden_dim,
            filter_channels_dp,
            kernel_size,
            p_dropout,
        )

    def forward(self, x, x_mask, ling_feat_emb, lang_emb, speaker_emb):
        x = self.embedding_proj(x)

        if self.prenet is not None:
            x = self.prenet(x.transpose(1, -1), x_mask).transpose(1, -1)

        cond_embs = []

        if self.ling_feat_proj is not None:
            cond_embs.append(self.ling_feat_proj(ling_feat_emb))

        if self.lang_emb_proj is not None:
            cond_embs.append(self.lang_emb_proj(lang_emb))
            cond_embs[-1] = cond_embs[-1].unsqueeze(1).expand(-1, x.shape[1], -1)

        if self.speaker_emb_proj is not None:
            cond_embs.append(self.speaker_emb_proj(speaker_emb))
            cond_embs[-1] = cond_embs[-1].unsqueeze(1).expand(-1, x.shape[1], -1)

        if self.cond_proj is not None:
            x = x + self.cond_proj(torch.cat(cond_embs, dim=2))

        x = x * math.sqrt(self.hidden_dim)

        if self.use_xpbert:
            attn_mask = get_attention_mask(x_mask, x_mask)
            position_ids = self.pe.create_position_ids_from_inputs_embeds(x)
            x += self.pe.position_embeddings(position_ids)

            x = self.encoder.encoder(x, attention_mask=attn_mask)[0]
            x = x.transpose(1, -1)
        else:
            x = self.encoder(x.transpose(1, -1), x_mask)

        x_m = self.proj_m(x)

        if not self.mean_only:
            x_logs = self.proj_s(x)
        else:
            x_logs = torch.zeros_like(x_m)

        logw = self.proj_w(x.detach(), x_mask)
        return x, x_m, x_logs, logw


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float,
        lang_emb_dim: tp.Optional[int] = None,
        speaker_emb_dim: tp.Optional[int] = None,
        speech_quality_emb_dim: tp.Optional[int] = None,
    ):
        channels = in_channels
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = False

        self.pre = nn.Conv1d(self.half_channels, hidden_dim, 1)
        self.enc = WN(
            hidden_dim,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout,
            lang_emb_dim,
            speaker_emb_dim,
            speech_quality_emb_dim,
        )
        self.post = nn.Conv1d(hidden_dim, self.half_channels * (2 - self.mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x,
        x_mask=None,
        reverse=False,
        lang_emb=None,
        speaker_emb=None,
        speech_quality_emb=None,
    ):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = apply_mask(self.pre(x0), x_mask)
        h = self.enc(h, x_mask, lang_emb, speaker_emb, speech_quality_emb)
        stats = apply_mask(self.post(h), x_mask)
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + apply_mask(x1 * torch.exp(logs), x_mask)
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = apply_mask((x1 - m) * torch.exp(-logs), x_mask)
            x = torch.cat([x0, x1], 1)
            return x, None

    def store_inverse(self):
        self.enc.remove_weight_norm()


class FlowSpecDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int,
        dilation_rate: int,
        n_blocks: int,
        n_layers: int,
        p_dropout: float,
        n_split: int = 4,
        n_sqz: int = 2,
        lang_emb_dim: tp.Optional[int] = None,
        speaker_emb_dim: tp.Optional[int] = None,
        speech_quality_emb_dim: tp.Optional[int] = None,
    ):
        super().__init__()
        self.n_sqz = n_sqz
        self.is_inverse = False

        self.flows = nn.ModuleList()
        for _ in range(n_blocks):
            self.flows.append(ActNorm(channels=in_channels * self.n_sqz))
            self.flows.append(
                InvConvNear(channels=in_channels * self.n_sqz, n_split=n_split)
            )
            self.flows.append(
                ResidualCouplingLayer(
                    in_channels * self.n_sqz,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    p_dropout=p_dropout,
                    lang_emb_dim=lang_emb_dim,
                    speaker_emb_dim=speaker_emb_dim,
                    speech_quality_emb_dim=speech_quality_emb_dim,
                )
            )

    def forward(
        self,
        x,
        x_mask,
        reverse=False,
        lang_emb=None,
        speaker_emb=None,
        speech_quality_emb=None,
    ):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        x_mask = x_mask.unsqueeze(1)
        if self.n_sqz > 1:
            x, x_mask = squeeze(x, x_mask, self.n_sqz)

        for f in flows:
            if not reverse:
                x, logdet = f(
                    x,
                    x_mask,
                    reverse,
                    lang_emb=lang_emb,
                    speaker_emb=speaker_emb,
                    speech_quality_emb=speech_quality_emb,
                )
                logdet_tot += logdet
            else:
                x, logdet = f(
                    x,
                    x_mask,
                    reverse,
                    lang_emb=lang_emb,
                    speaker_emb=speaker_emb,
                )

        if self.n_sqz > 1:
            x, x_mask = unsqueeze(x, x_mask, self.n_sqz)

        return x, logdet_tot

    def store_inverse(self):
        self.is_inverse = True
        for f in self.flows:
            f.store_inverse()


class AlignmentEncoder(torch.nn.Module):
    """Module for alignment text and mel spectrogram.

    Args:
        n_mel_channels: Dimension of mel spectrogram.
        n_text_channels: Dimension of text embeddings.
        n_att_channels: Dimension of model
        temperature: Temperature to scale distance by.
            Suggested to be 0.0005 when using dist_type "l2" and 15.0 when using "cosine".
        condition_types: List of types for nemo.collections.tts.modules.submodules.ConditionalInput.
        dist_type: Distance type to use for similarity measurement. Supports "l2" and "cosine" distance.

    """

    def __init__(
        self,
        n_mel_channels=80,
        n_text_channels=512,
        n_att_channels=80,
        temperature=0.0005,
        condition_types=[],
        dist_type="l2",
    ):
        super().__init__()
        self.temperature = temperature
        if dist_type == "cosine":
            assert self.temperature > 1

        self.condition_types = condition_types
        if condition_types:
            self.cond_input = ConditionalInput(
                n_text_channels, n_text_channels, condition_types
            )
        else:
            self.cond_input = torch.nn.Identity()
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(
                n_text_channels,
                n_text_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain="relu",
            ),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels * 2, n_att_channels, kernel_size=1, bias=True),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(
                n_mel_channels,
                n_mel_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain="relu",
            ),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels * 2, n_mel_channels, kernel_size=1, bias=True),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels, n_att_channels, kernel_size=1, bias=True),
        )
        if dist_type == "l2":
            self.dist_fn = self.get_euclidean_dist
        elif dist_type == "cosine":
            self.dist_fn = self.get_cosine_dist
        else:
            raise ValueError(f"Unknown distance type '{dist_type}'")

    @staticmethod
    def _apply_mask(inputs, mask, mask_value):
        if mask is None:
            return

        mask = rearrange(mask, "B T2 1 -> B 1 1 T2")
        inputs.data.masked_fill_(mask, mask_value)

    def get_dist(self, keys, queries, mask=None):
        """Calculation of distance matrix.

        Args:
            queries (torch.tensor): B x C1 x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): B x T2 x 1 tensor, binary mask for variable length entries and also can be used
                for ignoring unnecessary elements from keys in the resulting distance matrix (True = mask element, False = leave unchanged).
        Output:
            dist (torch.tensor): B x T1 x T2 tensor.

        """
        # B x C x T1
        queries_enc = self.query_proj(queries)
        # B x C x T2
        keys_enc = self.key_proj(keys)
        # B x 1 x T1 x T2
        dist = self.dist_fn(queries_enc=queries_enc, keys_enc=keys_enc)

        self._apply_mask(dist, mask, float("inf"))

        return dist

    @staticmethod
    def get_euclidean_dist(queries_enc, keys_enc):
        queries_enc = rearrange(queries_enc, "B C T1 -> B C T1 1")
        keys_enc = rearrange(keys_enc, "B C T2 -> B C 1 T2")
        # B x C x T1 x T2
        distance = (queries_enc - keys_enc) ** 2
        # B x 1 x T1 x T2
        l2_dist = distance.sum(axis=1, keepdim=True)
        return l2_dist

    @staticmethod
    def get_cosine_dist(queries_enc, keys_enc):
        queries_enc = rearrange(queries_enc, "B C T1 -> B C T1 1")
        keys_enc = rearrange(keys_enc, "B C T2 -> B C 1 T2")
        cosine_dist = -torch.nn.functional.cosine_similarity(queries_enc, keys_enc, dim=1)
        cosine_dist = rearrange(cosine_dist, "B T1 T2 -> B 1 T1 T2")
        return cosine_dist

    @staticmethod
    def get_durations(attn_soft, text_len, spect_len):
        """Calculation of durations.

        Args:
            attn_soft (torch.tensor): B x 1 x T1 x T2 tensor.
            text_len (torch.tensor): B tensor, lengths of text.
            spect_len (torch.tensor): B tensor, lengths of mel spectrogram.

        """
        attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
        durations = attn_hard.sum(2)[:, 0, :]
        assert torch.all(torch.eq(durations.sum(dim=1), spect_len))
        return durations

    @staticmethod
    def get_mean_dist_by_durations(dist, durations, mask=None):
        """Select elements from the distance matrix for the given durations and mask and
        return mean distance.

        Args:
            dist (torch.tensor): B x T1 x T2 tensor.
            durations (torch.tensor): B x T2 tensor. Dim T2 should sum to T1.
            mask (torch.tensor): B x T2 x 1 binary mask for variable length entries and also can be used
                for ignoring unnecessary elements in dist by T2 dim (True = mask element, False = leave unchanged).
        Output:
            mean_dist (torch.tensor): B x 1 tensor.

        """
        batch_size, t1_size, t2_size = dist.size()
        assert torch.all(torch.eq(durations.sum(dim=1), t1_size))

        AlignmentEncoder._apply_mask(dist, mask, 0)

        # TODO(oktai15): make it more efficient
        mean_dist_by_durations = []
        for dist_idx in range(batch_size):
            mean_dist_by_durations.append(
                torch.mean(
                    dist[
                        dist_idx,
                        torch.arange(t1_size),
                        torch.repeat_interleave(
                            torch.arange(t2_size), repeats=durations[dist_idx]
                        ),
                    ]
                )
            )

        return torch.tensor(mean_dist_by_durations, dtype=dist.dtype, device=dist.device)

    @staticmethod
    def get_mean_distance_for_word(l2_dists, durs, start_token, num_tokens):
        """Calculates the mean distance between text and audio embeddings given a range of
        text tokens.

        Args:
            l2_dists (torch.tensor): L2 distance matrix from Aligner inference. T1 x T2 tensor.
            durs (torch.tensor): List of durations corresponding to each text token. T2 tensor. Should sum to T1.
            start_token (int): Index of the starting token for the word of interest.
            num_tokens (int): Length (in tokens) of the word of interest.
        Output:
            mean_dist_for_word (float): Mean embedding distance between the word indicated and its predicted audio frames.

        """
        # Need to calculate which audio frame we start on by summing all durations up to the start token's duration
        start_frame = torch.sum(durs[:start_token]).data

        total_frames = 0
        dist_sum = 0

        # Loop through each text token
        for token_ind in range(start_token, start_token + num_tokens):
            # Loop through each frame for the given text token
            for frame_ind in range(start_frame, start_frame + durs[token_ind]):
                # Recall that the L2 distance matrix is shape [spec_len, text_len]
                dist_sum += l2_dists[frame_ind, token_ind]

            # Update total frames so far & the starting frame for the next token
            total_frames += durs[token_ind]
            start_frame += durs[token_ind]

        return dist_sum / total_frames

    def forward(self, queries, keys, conditioning=None, mask=None, attn_prior=None):
        """Forward pass of the aligner encoder.

        Args:
            queries (torch.tensor): B x C1 x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): B x T2 x 1 tensor, binary mask for variable length entries (True = mask element, False = leave unchanged).
            attn_prior (torch.tensor): prior for attention matrix.
            conditioning (torch.tensor): B x 1 x C2 conditioning embedding
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.

        """
        if self.condition_types:
            keys = self.cond_input(keys.transpose(1, 2), conditioning).transpose(1, 2)

        # B x C x T1
        queries_enc = self.query_proj(queries)
        # B x C x T2
        keys_enc = self.key_proj(keys)
        # B x 1 x T1 x T2
        distance = self.dist_fn(queries_enc=queries_enc, keys_enc=keys_enc)
        attn = -self.temperature * distance

        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)

        attn_logprob = attn.clone()

        self._apply_mask(attn, mask, -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob
