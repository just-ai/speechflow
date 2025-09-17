import math
import random

import torch
import monotonic_align

from torch import Tensor, nn
from torch.nn import functional as F

from .misc import AttentionBlock, LayerNormDP, sequence_mask


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02, relative=False):
        super().__init__()
        # nn.Embedding
        self.emb = torch.nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def get_emb_pos(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            emb = self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            emb = self.emb(torch.arange(0, sl, device=x.device))

        return emb.unsqueeze(0)

    def forward(self, x):
        emb_pos = self.get_emb_pos(x)
        return x + emb_pos

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class MultiEmbedding(nn.Module):
    """This embedding sums embeddings on different levels."""

    def __init__(self, max_n_levels, n_tokens, token_dim):
        super().__init__()
        self.max_n_levels = max_n_levels
        self.n_tokens = n_tokens
        self.weight = nn.Parameter(torch.randn(max_n_levels, n_tokens, token_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(x, num_classes=self.n_tokens).to(self.weight)
        x = torch.einsum("l k d, n l s k -> n s d", self.weight, x)

        return x


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)


class PrenetText(nn.Module):
    def __init__(self, dim_model: int, is_enable: bool = True, dropout: float = 0.25):
        super().__init__()

        if is_enable:
            self.prenet = nn.Sequential(
                Transpose(),
                nn.Conv1d(dim_model, dim_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(dim_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(dim_model, dim_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(dim_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(dim_model, dim_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(dim_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                Transpose(),
                nn.Linear(dim_model, dim_model),
            )
        else:
            self.prenet = nn.Identity()

    def forward(self, x):
        return self.prenet(x)


class PrenetAudio(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_internal: int,
        is_channel_first: bool = False,
        is_enable: bool = True,
        dropout: float = 0.25,
    ):
        super().__init__()

        self._is_channel_first = is_channel_first
        if is_enable:
            self.prenet = nn.Sequential(
                nn.Linear(dim_model, dim_internal),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_internal, dim_internal),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_internal, dim_model),
            )
        else:
            self.prenet = nn.Identity()

    def forward(self, x):
        if self._is_channel_first:
            x = x.transpose(1, 2)

        x = self.prenet(x)

        if self._is_channel_first:
            x = x.transpose(1, 2)

        return x


class DurationPredictor(nn.Module):
    """https://github.com/WelkinYang/GradTTS/blob/main/models.py."""

    def __init__(
        self, in_channels: int, filter_channels: int, kernel_size: int, p_dropout: float
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNormDP(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNormDP(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x):

        x = self.conv_1(x)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x)

        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x)
        return x


class ConditioningEncoder(nn.Module):
    def __init__(
        self,
        spec_dim,
        embedding_dim,
        attn_blocks=6,
        num_attn_heads=4,
    ):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        """
        x: (b, 80, s)
        """
        h = self.init(x)
        h = self.attn(h)
        return h


class MonotonicAlign(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.ndim = ndim

    @torch.no_grad()
    def forward(self, x, x_lens, y, y_lens):
        x_mask = sequence_mask(x_lens).unsqueeze(1)
        y_mask = sequence_mask(y_lens).unsqueeze(1)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        const = -0.5 * math.log(2 * math.pi) * self.ndim
        factor = -0.5 * torch.ones(x.shape, dtype=x.dtype, device=x.device)
        y_square = torch.matmul(factor.transpose(1, 2), y**2)
        y_mu_double = torch.matmul(2.0 * (factor * x).transpose(1, 2), y)
        mu_square = torch.sum(factor * (x**2), 1).unsqueeze(-1)
        log_prior = y_square - y_mu_double + mu_square + const

        attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
        return attn.detach()
