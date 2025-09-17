# References:
# https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/transformer.py
# https://github.com/jaywalnut310/vits/blob/main/attentions.py
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py

import torch.nn as nn

from speechflow.utils.tensor_utils import apply_mask, get_attention_mask
from tts.acoustic_models.modules.common.blocks import FFNConv, MultiHeadAttention

__all__ = ["DiTConv"]


# modified from https://github.com/sh-lee-prml/HierSpeechpp/blob/main/modules.py#L390
class DiTConv(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self,
        content_dim,
        conditional_dim,
        filter_channels: int = 1024,
        num_heads: int = 4,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        eps: float = 1.0e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(content_dim, eps=eps, elementwise_affine=False)
        self.attn = MultiHeadAttention(
            num_heads,
            content_dim,
            content_dim,
            content_dim,
            p_dropout=p_dropout,
            use_residual=False,
            use_norm=False,
        )
        self.norm2 = nn.LayerNorm(content_dim, eps=eps, elementwise_affine=False)
        self.mlp = FFNConv(
            content_dim, content_dim, filter_channels, kernel_size, p_dropout=p_dropout
        )
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(conditional_dim, content_dim)
            if conditional_dim != content_dim
            else nn.Identity(),
            nn.SiLU(),
            nn.Linear(content_dim, 6 * content_dim, bias=True),
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, x_mask, c):
        """
        Args:
            x : [batch_size, time, channel]
            x_mask : [batch_size, time]
            c : [batch_size, channel]
        return the same shape as x
        """
        x = apply_mask(x, x_mask).transpose(1, -1)

        c = self.adaLN_modulation(c)

        # shape: [batch_size, channel, 1]
        if c.ndim == 2:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.unsqueeze(
                -1
            ).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.transpose(
                1, 2
            ).chunk(6, dim=1)

        z1 = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        z2 = self.norm2(x.transpose(1, 2)).transpose(1, 2)

        s = self.modulate(z1, shift_msa, scale_msa).transpose(1, -1)
        s_attn, _ = self.attn(s, s, s, get_attention_mask(x_mask, x_mask))
        x = x + apply_mask(gate_msa * s_attn.transpose(1, -1), x_mask)
        x = x + gate_mlp * self.mlp(self.modulate(z2, shift_mlp, scale_mlp), x_mask)

        # no condition version
        # x = x + self.attn(self.norm1(x.transpose(1,2)).transpose(1,2),  attn_mask)
        # x = x + self.mlp(self.norm2(x.transpose(1,2)).transpose(1,2), x_mask)
        return x.transpose(1, -1)

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift
