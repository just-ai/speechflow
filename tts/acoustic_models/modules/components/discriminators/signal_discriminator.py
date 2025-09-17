import typing as tp

import torch

from torch import nn
from torch.nn import functional as F

from speechflow.utils.tensor_utils import apply_mask

__all__ = ["SignalDiscriminator"]


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class SignalDiscriminator(nn.Module):
    #  https://github.com/p0p4k/vits2_pytorch/
    def __init__(
        self,
        in_channels,
        filter_channels=192,
        kernel_size=3,
        p_dropout=0.1,
        gin_channels=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels)
        self.signal_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = LayerNorm(filter_channels)

        # if gin_channels != 0:
        #   self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, x_mask, signal, g=None):
        signal = self.signal_proj(signal)
        x = torch.cat([x, signal], dim=1)
        x = apply_mask(self.pre_out_conv_1(x), x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)
        x = apply_mask(self.pre_out_conv_2(x), x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, signal_r, signal_hat, g=None):
        # if g is not None:
        #   g = torch.detach(g)
        #   x = x + self.cond(g)
        x = apply_mask(self.conv_1(x), x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = apply_mask(self.conv_2(x), x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)

        output_probs = []
        for signal in [signal_r, signal_hat]:
            output_prob = self.forward_probability(x, x_mask, signal, g)
            output_probs.append([output_prob])

        return output_probs

    @staticmethod
    def discriminator_loss(disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            dr = dr.float()
            dg = dg.float()
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses

    @staticmethod
    def generator_loss(disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            dg = dg.float()
            l_ = torch.mean((1 - dg) ** 2)
            gen_losses.append(l_)
            loss += l_

        return loss, gen_losses

    def calculate_loss(
        self, context, context_mask, real, fake, current_iter, discriminator_freq: int = 3
    ) -> tp.Dict[str, torch.Tensor]:
        losses = {}

        if current_iter % discriminator_freq == 0:
            y_hat_r, y_hat_g = self.forward(
                context.detach(),
                context_mask.detach(),
                real.detach(),
                fake.detach(),
            )
            losses["disc_loss"], _, _ = self.discriminator_loss(y_hat_r, y_hat_g)
        else:
            y_hat_r, y_hat_g = self.forward(
                context,
                context_mask.detach(),
                real.detach(),
                fake,
            )
            losses["gen_loss"], _ = self.generator_loss(y_hat_g)

        return losses
