import math
import typing as tp

import torch
import torch.nn as nn

PE_TYPES = tp.Literal["PE", "RoPE"]

__all__ = [
    "PositionalEncodings",
    "PE_TYPES",
    "BasePositionalEncodings",
    "RotaryPositionalEmbeddings",
]


class PositionalEncodings(nn.Module):
    def __init__(
        self, pe_type: PE_TYPES, d: int, base: int = 10_000, batch_first: bool = False
    ):
        super().__init__()

        if pe_type == "PE":
            self.module = BasePositionalEncodings(d, base, batch_first)
        elif pe_type == "RoPE":
            self.module = RotaryPositionalEmbeddings(d, base, batch_first)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        return self.module(x)


class BasePositionalEncodings(nn.Module):
    """Positional Encoding proposed in "Attention Is All You Need". Since transformer
    contains no recurrence and no convolution, in order for the model to make use of the
    order of the sequence, we must add some positional information.

    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))

    """

    def __init__(self, d, base: int = 10_000, batch_first: bool = False):
        super().__init__()
        self.d = int(d)
        self.base = base
        self.batch_first = batch_first
        self.pe_cached = None

    def _build_cache(self, x: torch.Tensor):
        r"""
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built
        if self.pe_cached is not None and x.shape[1] <= self.pe_cached.shape[1]:
            return

        # Get sequence length
        seq_len = 2 * x.shape[1]

        pe = torch.zeros(seq_len, self.d)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d, 2).float() * (-math.log(self.base) / self.d)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1).to(x.device)

    def forward(self, x):
        if not self.batch_first:
            x = x.transpose(0, 1)

        self._build_cache(x)
        x = x + self.pe[: x.shape[1], :].transpose(0, 1)

        if not self.batch_first:
            x = x.transpose(0, 1)

        return x


class RotaryPositionalEmbeddings(nn.Module):
    """## RoPE module.

    Rotary encoding transforms pairs of features by rotating in the 2D plane. That
    is, it organizes the $d$ features as $\frac{d}{2}$ pairs. Each pair can be
    considered a coordinate in a 2D plane, and the encoding will rotate it by an
    angle depending on the position of the token.

    """

    def __init__(self, d: int, base: int = 10_000, batch_first: bool = False):
        r"""
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()
        self.d = d
        self.base = base
        self.batch_first = batch_first
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        r"""
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        # Get sequence length
        seq_len = 2 * x.shape[0]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(
            x.device
        )

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        ndim = x.ndim

        if self.batch_first:
            x = x.transpose(0, 1)

        if ndim == 3:
            x = x.unsqueeze(1)

        # Cache $\cos$ and $\sin$ values
        x = x.permute(2, 0, 1, 3)  # b h t d -> t b h d

        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (
            neg_half_x * self.sin_cached[: x.shape[0]]
        )

        x = torch.cat((x_rope, x_pass), dim=-1).permute(1, 2, 0, 3)  # t b h d -> b h t d

        if ndim == 3:
            x = x.squeeze(1)

        if self.batch_first:
            x = x.transpose(0, 1)

        return x
