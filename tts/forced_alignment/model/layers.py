import math

import torch

from torch import nn
from torch.nn import functional as F

from speechflow.utils.tensor_utils import apply_mask
from tts.forced_alignment.model.utils import (
    convert_pad_shape,
    fused_add_tanh_sigmoid_multiply,
)

__all__ = [
    "MultiHeadAttention",
    "LayerNorm",
    "FFN",
    "ConvReluNorm",
    "WN",
    "ActNorm",
    "InvConvNear",
    "ConvNorm",
    "ConditionalInput",
]


# noinspection PyTupleAssignmentBalance
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        window_size=None,
        heads_share=True,
        p_dropout=0.0,
        block_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))  # type: ignore
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores * block_mask + -1e4 * (1 - block_mask)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output += self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )

        output = (
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    @staticmethod
    def _matmul_with_relative_values(x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    @staticmethod
    def _matmul_with_relative_keys(x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    @staticmethod
    def _relative_position_to_absolute_position(x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final

    @staticmethod
    def _absolute_position_to_relative_position(x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    @staticmethod
    def _attention_bias_proximal(length):
        """Bias for self-attention to encourage attention to close positions.

        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]

        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = x.ndim
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation

        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.conv_2 = nn.Conv1d(
            filter_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(apply_mask(x, x_mask))
        if self.activation == "gelu":
            x *= torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(apply_mask(x, x_mask))
        return x


class ConvReluNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        n_layers,
        p_dropout,
    ):
        super().__init__()
        self.n_layers = n_layers
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        )
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](apply_mask(x, x_mask))
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)

        x = x_org + self.proj(x)
        return apply_mask(x, x_mask)


class WN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        lang_emb_dim=None,
        speaker_emb_dim=None,
        speech_quality_emb_dim=None,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        assert hidden_channels % 2 == 0
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.lang_emb_dim = lang_emb_dim
        self.speaker_emb_dim = speaker_emb_dim
        self.speech_quality_emb_dim = speech_quality_emb_dim

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        self.cond_dim = 0
        for dim in [
            self.lang_emb_dim,
            self.speaker_emb_dim,
            self.speech_quality_emb_dim,
        ]:
            if dim is not None:
                self.cond_dim += dim

        if self.cond_dim > 0:
            cond_proj = torch.nn.Conv1d(self.cond_dim, 2 * hidden_channels * n_layers, 1)
            self.cond_proj = torch.nn.utils.weight_norm(cond_proj)
        else:
            self.cond_proj = None

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer)
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self,
        x,
        x_mask=None,
        lang_emb=None,
        speaker_emb=None,
        speech_quality_emb=None,
    ):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        cond_embs = []

        if self.lang_emb_dim is not None:
            lang_emb = lang_emb.unsqueeze(1).expand(-1, x.shape[2], -1)
            cond_embs.append(lang_emb)

        if self.speaker_emb_dim is not None:
            speaker_emb = speaker_emb.unsqueeze(1).expand(-1, x.shape[2], -1)
            cond_embs.append(speaker_emb)

        if self.speech_quality_emb_dim is not None:
            speech_quality_emb = speech_quality_emb.unsqueeze(1).expand(
                -1, x.shape[2], -1
            )
            cond_embs.append(speech_quality_emb)

        if self.cond_proj is not None:
            g = self.cond_proj(torch.cat(cond_embs, dim=2).transpose(1, -1))
        else:
            g = torch.zeros(
                (x.shape[0], 2 * self.hidden_channels * self.n_layers, x.shape[2]),
                device=x.device,
            )

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)

            cond_offset = i * 2 * self.hidden_channels
            g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = apply_mask((x + res_skip_acts[:, : self.hidden_channels, :]), x_mask)
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

        return apply_mask(output, x_mask)

    def remove_weight_norm(self):
        if self.speaker_emb_dim is not None:
            torch.nn.utils.remove_weight_norm(self.cond_proj)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)


class ActNorm(nn.Module):
    def __init__(self, channels, ddi=False):
        super().__init__()
        self.channels = channels
        self.initialized = not ddi

        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2)).to(
                device=x.device, dtype=x.dtype
            )

        x_lens = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True

        if reverse:
            z = apply_mask((x - self.bias) * torch.exp(-self.logs), x_mask)
            logdet = None
        else:
            z = apply_mask((self.bias + torch.exp(self.logs) * x), x_mask)
            logdet = torch.sum(self.logs) * x_lens  # [b]

        return z, logdet

    def store_inverse(self):
        pass

    def set_ddi(self, ddi):
        self.initialized = not ddi

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(apply_mask(x, x_mask), [0, 2]) / denom
            m_sq = torch.sum(apply_mask(x * x, x_mask), [0, 2]) / denom
            v = m_sq - (m**2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (
                (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
            )
            logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)


class InvConvNear(nn.Module):
    def __init__(self, channels, n_split=4, no_jacobian=False):
        super().__init__()
        assert n_split % 2 == 0
        self.channels = channels
        self.n_split = n_split
        self.no_jacobian = no_jacobian

        w_init = torch.linalg.qr(torch.FloatTensor(self.n_split, self.n_split).normal_())[
            0
        ]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = nn.Parameter(w_init)
        self.weight_inv = None

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        b, c, t = x.size()
        assert c % self.n_split == 0
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])

        x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
        x = (
            x.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(b, self.n_split, c // self.n_split, t)
        )

        if reverse:
            if hasattr(self, "weight_inv"):
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float()).to(
                    self.weight.device, dtype=self.weight.dtype
                )
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
            else:
                logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len  # [b]

        weight = weight.view(self.n_split, self.n_split, 1, 1).to(self.weight.device)
        z = F.conv2d(x, weight)

        z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
        z = apply_mask(z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t), x_mask)
        return z, logdet

    def store_inverse(self):
        self.weight_inv = torch.inverse(self.weight.float()).to(
            self.weight.device, dtype=self.weight.dtype
        )


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        use_partial_padding=False,
        use_weight_norm=False,
        norm_fn=None,
    ):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.use_partial_padding = use_partial_padding
        conv_fn = torch.nn.Conv1d
        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        if use_weight_norm:
            self.conv = torch.nn.utils.weight_norm(self.conv)
        if norm_fn is not None:
            self.norm = norm_fn(out_channels, affine=True)
        else:
            self.norm = None

    def forward(self, signal, mask=None):
        if self.use_partial_padding:
            ret = self.conv(signal, mask)
            if self.norm is not None:
                ret = self.norm(ret, mask)
        else:
            if mask is not None:
                signal = signal.mul(mask)
            ret = self.conv(signal)
            if self.norm is not None:
                ret = self.norm(ret)

        return ret


class ConditionalInput(torch.nn.Module):
    """This module is used to condition any model inputs.

    If we don't have any conditions, this will be a normal pass.

    """

    def __init__(self, hidden_dim, condition_dim, condition_types=[]):
        super().__init__()
        self.support_types = ["add", "concat"]
        self.condition_types = [tp for tp in condition_types if tp in self.support_types]
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        if "add" in self.condition_types and condition_dim != hidden_dim:
            self.add_proj = torch.nn.Linear(condition_dim, hidden_dim)

        if "concat" in self.condition_types:
            self.concat_proj = torch.nn.Linear(hidden_dim + condition_dim, hidden_dim)

    def forward(self, inputs, conditioning=None):
        """
        Args:
            inputs (torch.tensor): B x T x C tensor.
            conditioning (torch.tensor): B x 1 x C conditioning embedding.
        """
        if len(self.condition_types) > 0:
            if conditioning is None:
                raise ValueError(
                    """You should add additional data types as conditions (e.g. speaker id or reference audio)
                                 and define speaker_encoder in your config."""
                )

            if "add" in self.condition_types:
                if self.condition_dim != self.hidden_dim:
                    conditioning = self.add_proj(conditioning)
                inputs = inputs + conditioning

            if "concat" in self.condition_types:
                conditioning = conditioning.repeat(1, inputs.shape[1], 1)
                inputs = torch.cat([inputs, conditioning], dim=-1)
                inputs = self.concat_proj(inputs)

        return inputs
