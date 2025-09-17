import typing as tp

import torch
import torch.nn.functional as F

from torch import nn

__all__ = [
    "Conv",
    "LearnableSwish",
    "HighwayNetwork",
]


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: tp.Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: tp.Optional[str] = "linear",
        groups: int = 1,
        swap_channel_dim: bool = False,
        batch_norm: bool = False,
        use_activation: bool = True,
        activation: tp.Optional[tp.Any] = None,
    ):
        """
        :param use_activation: bool
            if True and activation param is not provided then default F.relu is used
        :param activation: activation function from torch.nn.functional
        """
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.swap_channel_dim = swap_channel_dim
        self.groups = groups

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )

        if w_init_gain is not None:
            nn.init.xavier_uniform_(
                self.conv.weight, gain=nn.init.calculate_gain(w_init_gain)
            )
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None

        if activation is not None:
            self.activation = activation
        elif use_activation:
            self.activation = F.relu
        else:
            self.activation = None

    def forward(self, signal):
        if self.swap_channel_dim:
            signal = signal.transpose(1, 2)
        conv_signal = self.conv(signal)
        if self.batch_norm:
            conv_signal = self.batch_norm(conv_signal)
        if self.activation:
            conv_signal = self.activation(conv_signal)
        if self.swap_channel_dim:
            conv_signal = conv_signal.transpose(1, 2)
        return conv_signal


class LearnableSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(1))
        self.sigmoid_fn = nn.Sigmoid()

    def forward(self, x):
        return self.slope * x * self.sigmoid_fn(x)


class HighwayNetwork(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.lin1 = nn.Linear(size, size)
        self.lin2 = nn.Linear(size, size)
        self.sigmoid = nn.Sigmoid()
        self.lin1.bias.data.fill_(0.0)

    def forward(self, x):
        x1 = self.lin1(x)
        x2 = self.lin2(x)
        g = self.sigmoid(x2)
        y = g * F.relu(x1) + (1.0 - g) * x
        return y
