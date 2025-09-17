# https://github.com/KevinMIN95/StyleSpeech

import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F

from speechflow.utils.tensor_utils import get_mask_from_lengths, masked_fill
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.components.style_encoders.style_encoder import (
    StyleEncoderParams,
)
from tts.acoustic_models.modules.data_types import MODEL_INPUT_TYPE

__all__ = ["StyleSpeech", "StyleSpeechParams"]


class StyleSpeechParams(StyleEncoderParams):
    conv_kernel_size: int = 5
    n_heads: int = 2
    p_dropout: float = 0.1


class StyleSpeech(Component):
    params: StyleSpeechParams

    def __init__(self, params: StyleSpeechParams, input_dim: int):
        super().__init__(params, input_dim)

        in_dim = input_dim
        inner_dim = params.style_emb_dim
        kernel_size = params.conv_kernel_size

        self.spectral = nn.Sequential(
            LinearNorm(in_dim, inner_dim),
            Mish(),
            nn.Dropout(params.p_dropout),
            LinearNorm(inner_dim, inner_dim),
            Mish(),
            nn.Dropout(params.p_dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(inner_dim, inner_dim, kernel_size, params.p_dropout),
            Conv1dGLU(inner_dim, inner_dim, kernel_size, params.p_dropout),
        )

        self.slf_attn = MultiHeadAttention(
            params.n_heads,
            inner_dim,
            inner_dim // params.n_heads,
            inner_dim // params.n_heads,
            params.p_dropout,
        )

        self.fc = LinearNorm(inner_dim, inner_dim)

    @property
    def output_dim(self):
        return self.params.style_emb_dim

    @staticmethod
    def temporal_avg_pool(x, x_lengths, x_mask):
        x = masked_fill(x, x_mask, 0)
        x = x.sum(dim=1)
        out = torch.div(x, x_lengths.unsqueeze(1))
        return out

    def encode(self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs):
        max_len = x.shape[1]
        x_mask = get_mask_from_lengths(x_lengths, max_len)

        slf_attn_mask = x_mask.unsqueeze(1).expand(-1, max_len, -1)

        # spectral
        x = self.spectral(x)

        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)

        # self-attention
        x = masked_fill(x, x_mask, 0)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)

        # fc
        x = self.fc(x)

        # temoral average pooling
        w = self.temporal_avg_pool(x, x_lengths, x_mask).unsqueeze(1)

        return w

    def forward_step(self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs):
        style_emb = self.encode(x, x_lengths, model_inputs, **kwargs)
        return style_emb, {}, {}


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class LinearNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        spectral_norm=False,
    ):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias)

        if spectral_norm:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, input):
        out = self.fc(input)
        return out


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        spectral_norm=False,
    ):
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.0, spectral_norm=False):
        super().__init__()

        self.n_heads = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attn = ScaledDotProductAttention(
            temperature=np.power(d_model, 0.5), dropout=dropout
        )

        self.fc = nn.Linear(n_head * d_v, d_model)
        self.p_dropout = nn.Dropout(dropout)

        if spectral_norm:
            self.w_qs = nn.utils.spectral_norm(self.w_qs)
            self.w_ks = nn.utils.spectral_norm(self.w_ks)
            self.w_vs = nn.utils.spectral_norm(self.w_vs)
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_heads
        sz_b, len_x, _ = x.size()

        residual = x

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_v)  # (n*b) x lv x dv

        if mask is not None:
            slf_mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        else:
            slf_mask = None
        output, attn_mask = self.attn(q, k, v, mask=slf_mask)

        output = output.view(n_head, sz_b, len_x, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_x, -1)
        )  # b x lq x (n*dv)

        output = self.p_dropout(self.fc(output))
        output = output + residual
        return output, attn_mask


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.p_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = masked_fill(attn, mask, -torch.inf)

        attn = self.softmax(attn)
        p_attn = self.p_dropout(attn)

        output = torch.bmm(p_attn, v)
        return output, attn


class Conv1dGLU(nn.Module):
    """Conv1d + GLU(Gated Linear Unit) with residual connection.

    For GLU refer to https://arxiv.org/abs/1612.08083 paper.

    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = ConvNorm(in_channels, 2 * out_channels, kernel_size=kernel_size)
        self.p_dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.p_dropout(x)
        return x
