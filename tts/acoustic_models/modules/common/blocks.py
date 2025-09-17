import math
import typing as tp

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from speechflow.utils.tensor_utils import apply_mask, run_rnn_on_padded_sequence
from tts.acoustic_models.modules.common.layers import Conv, HighwayNetwork
from tts.acoustic_models.modules.common.pos_encoders import RotaryPositionalEmbeddings

__all__ = [
    "VarianceEmbedding",
    "Regression",
    "ConvPrenet",
    "CBHG",
    "FFNConv",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
]


class VarianceEmbedding(nn.Module):
    def __init__(
        self,
        interval: tp.Tuple[float, ...],
        n_bins: int = 256,
        emb_dim: int = 32,
        log_scale: bool = False,
        activation_fn: str = "Tanh",
    ):
        super().__init__()
        v_min, v_max = interval
        self.log_scale = log_scale

        if self.log_scale:
            v_min, v_max = np.log1p(v_min), np.log1p(v_max)

        self.bins = torch.linspace(v_min, v_max, n_bins - 1)

        self.embedding = nn.Embedding(n_bins, emb_dim)
        self.af = getattr(nn, activation_fn)()

    def forward(self, x: torch.Tensor):
        embedding = None

        if self.bins.device != x.device:
            self.bins = self.bins.to(x.device)

        if x.ndim == 2:
            y = torch.log1p(x) if self.log_scale else x
            embedding = self.af(self.embedding(torch.bucketize(y, self.bins)))

        elif x.ndim == 3:
            temp = []
            for i in range(x.shape[2]):
                y = torch.log1p(x[:, :, i]) if self.log_scale else x[:, :, i]
                temp.append(self.af(self.embedding(torch.bucketize(y, self.bins))))

            embedding = torch.cat(temp, dim=2)

        return embedding.squeeze(1)


class Regression(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        p_dropout: float = 0.1,
        activation_fn: str = "Identity",
    ):
        super().__init__()

        if in_dim != out_dim:
            self.fc1 = nn.Linear(in_features=in_dim, out_features=in_dim)
            self.fc2 = nn.Linear(in_features=in_dim, out_features=out_dim)
        else:
            self.fc1 = self.fc2 = None

        self.af1 = nn.SiLU()
        self.af2 = getattr(nn, activation_fn)()

        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask=None):
        if x.ndim == 2:
            x = x.unsqueeze(1)

        if self.fc1 is None:
            y = self.af2(x)
        else:
            y = self.af1(self.fc1(x))
            y = self.af2(self.fc2(self.drop(y)))

        if x_mask is not None:
            y = apply_mask(y, x_mask)

        return y


class ConvPrenet(nn.Module):
    """Default Prenet Block.

    Consist of Reflection padding and Convolutional layer.
    x -> ReflectionPadding -> Conv1D -> Activation -> y

    Args:
        in_channels (int): Input channels to Encoder Block.
        out_channels (int): Output channels from Encoder Block.
        kernel_size (int): Kernel size for Conv1D. Defaults to 3.

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd integer, got kernel_size={kernel_size}"
            )

        _padding_size = math.ceil((kernel_size - 1) / 2)

        self.padding = nn.ReflectionPad1d(_padding_size)
        self.encoder = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self.af = nn.SELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, Length)

        Returns:
            y (torch.Tensor): Output tensor of shape (B, out_channels, Length)
        """
        x = self.padding(x)
        x = self.encoder(x)
        x = self.af(x)
        return x


class CBHG(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        conv_banks_num: int,
        highways_num: int,
        kernel_size: int = 3,
        rnn_dim: tp.Optional[int] = None,
        rnn_bidirectional: bool = True,
    ):
        super().__init__()
        if rnn_dim is None:
            rnn_dim = in_dim

        # List of all rnns to call `flatten_parameters()` on
        self._to_flatten = []

        self.bank_kernels = [kernel_size * (i + 1) for i in range(conv_banks_num)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            padding = math.ceil((k - 1) / 2)
            conv = Conv(
                in_dim,
                out_dim,
                kernel_size=k,
                padding=padding,
                bias=False,
                batch_norm=True,
                use_activation=True,
            )
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

        self.conv_project1 = Conv(
            conv_banks_num * out_dim,
            out_dim,
            kernel_size=kernel_size,
            padding=math.ceil((kernel_size - 1) / 2),
            bias=False,
            batch_norm=True,
            use_activation=True,
        )
        self.conv_project2 = Conv(
            out_dim,
            in_dim,
            kernel_size=kernel_size,
            padding=math.ceil((kernel_size - 1) / 2),
            bias=False,
            batch_norm=True,
            use_activation=True,
        )

        self.highways = nn.ModuleList()
        for i in range(highways_num):
            hn = HighwayNetwork(in_dim)
            self.highways.append(hn)

        self.rnn = nn.GRU(
            rnn_dim,
            out_dim // (2 if rnn_bidirectional else 1),
            batch_first=True,
            bidirectional=rnn_bidirectional,
        )
        self._to_flatten.append(self.rnn)

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

        self.receptive_field = (torch.tensor(self.bank_kernels) - 1).sum() + 4 + 1 + 1

    def calc_rnn(self, x, seq_lengths=None):
        return run_rnn_on_padded_sequence(self.rnn, x, seq_lengths)

    def forward(self, x, seq_lengths=None):
        x = x.transpose(2, 1)

        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        # Save these for later
        residual = x
        conv_bank = []

        # Convolution Bank
        for idx, conv in enumerate(self.conv1d_bank):
            c = conv(x)
            if idx % 2 == 0:
                c = F.pad(c, [0, 1])
            conv_bank.append(c)

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        # dump the last padding to fit residual
        x = self.maxpool(conv_bank)

        # Conv1d projections
        x = self.conv_project1(x)
        x = self.conv_project2(x)

        # Residual Connect
        x = x + residual

        # Through the highways
        x = x.transpose(1, 2)
        for h in self.highways:
            x = h(x)

        # And then the RNN
        x = self.calc_rnn(x, seq_lengths)

        return x

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns used by the WaveRNN.

        Used to improve efficiency and avoid PyTorch yelling at us.

        """
        [m.flatten_parameters() for m in self._to_flatten]


class FFNConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.conv_2 = nn.Conv1d(
            filter_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.dropout = nn.Dropout(p_dropout)
        self.act = nn.SiLU()

    def forward(self, x, x_mask):
        x = self.conv_1(apply_mask(x, x_mask))
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv_2(apply_mask(x, x_mask))
        return apply_mask(x, x_mask)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature: float, p_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -torch.inf)

        attn = self.dropout(F.softmax(attn, dim=-1))
        attn = attn.masked_fill(attn.isnan(), 0)

        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v: int,
        p_dropout: float = 0.1,
        use_residual: bool = True,
        use_norm: bool = True,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.use_residual = use_residual

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5, p_dropout=p_dropout
        )

        # from https://nn.labml.ai/transformers/rope/index.html
        self.query_rotary_pe = RotaryPositionalEmbeddings(int(d_model * 0.5))
        self.key_rotary_pe = RotaryPositionalEmbeddings(int(d_model * 0.5))

        if self.use_residual:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = nn.Identity()

        if use_norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = self.query_rotary_pe(q)
        k = self.key_rotary_pe(k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None and mask.ndim == 3:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        if self.use_residual:
            q += residual

        q = self.layer_norm(q)
        return q, attn
