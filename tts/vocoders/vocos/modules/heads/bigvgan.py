import typing as tp

import torch

from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from speechflow.io import tp_PATH
from speechflow.training.base_model import BaseTorchModelParams
from tts.vocoders.vocos.modules.heads.base import WaveformGenerator

from .components import activations
from .components.alias_free_activation.torch.act import Activation1d as TorchActivation1d
from .components.utils import get_padding, init_weights

__all__ = ["BigVGANHead", "BigVGANHeadParams"]


class BigVGANHeadParams(BaseTorchModelParams):
    input_dim: int = 100

    upsample_rates: tp.Tuple[int, ...] = [4, 4, 2, 2, 2, 2]
    upsample_kernel_sizes: tp.Tuple[int, ...] = [8, 8, 4, 4, 4, 4]
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: tp.Tuple[int, ...] = [3, 7, 11]
    resblock_dilation_sizes: tp.Tuple[tp.List[int], ...] = (
        [1, 3, 5],
        [1, 3, 5],
        [1, 3, 5],
    )

    use_tanh_at_final: bool = False
    use_bias_at_final: bool = False

    resblock: str = "1"
    activation: str = "snakebeta"
    log_scale: bool = True

    use_cuda_kernel: bool = False

    pretrain_path: tp.Optional[tp_PATH] = None


class BigVGANHead(WaveformGenerator):
    params: BigVGANHeadParams

    """
    BigVGAN is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).
    New in BigVGAN-v2: it can optionally use optimized CUDA kernels for AMP (anti-aliased multi-periodicity) blocks.

    Note:
        - The `use_cuda_kernel` parameter should be used for inference only, as training with CUDA kernels is not supported.
        - Ensure that the activation function is correctly specified in the hyperparameters (h.activation).
    """

    def __init__(self, params: BigVGANHeadParams):
        super().__init__(params)

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if params.use_cuda_kernel:
            from .components.alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(params.resblock_kernel_sizes)
        self.num_upsamples = len(params.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(params.input_dim, params.upsample_initial_channel, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if params.resblock == "1":
            resblock_class = AMPBlock1
        elif params.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {params.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(params.upsample_rates, params.upsample_kernel_sizes)
        ):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                params.upsample_initial_channel // (2**i),
                                params.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = params.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(params.resblock_kernel_sizes, params.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(
                        ch,
                        k,
                        d,
                        activation=params.activation,
                        log_scale=params.log_scale,
                        use_cuda_kernel=params.use_cuda_kernel,
                    )
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=params.log_scale)
            if params.activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=params.log_scale)
                if params.activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = params.use_bias_at_final
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)

        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = params.use_tanh_at_final

        if params.pretrain_path is not None:
            state_dict = torch.load(params.pretrain_path, map_location="cpu")
            self.load_state_dict(state_dict["generator"])

    def forward(self, x, **kwargs):
        # Pre-conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)

            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)

            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)

        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x.squeeze(1), None, {}

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l_ in self.ups:
                for l_i in l_:
                    remove_weight_norm(l_i)
            for l_ in self.resblocks:
                l_.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass


class AMPBlock1(nn.Module):
    """AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters
    that control periodicity, defined for each layer. AMPBlock1 has additional self.convs2
    that contains additional Conv1d layers with a fixed dilation=1 followed by each layer
    in self.convs1.

    Args:
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.

    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
        log_scale: bool = False,
        use_cuda_kernel: bool = False,
    ):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if use_cuda_kernel:
            from components.alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(channels, alpha_logscale=log_scale)
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels,
                            alpha_logscale=log_scale,
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l_ in self.convs1:
            remove_weight_norm(l_)
        for l_ in self.convs2:
            remove_weight_norm(l_)


class AMPBlock2(nn.Module):
    """AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters
    that control periodicity, defined for each layer. Unlike AMPBlock1, AMPBlock2 does not
    contain extra Conv1d layers with fixed dilation=1.

    Args:
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.

    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
        log_scale: bool = False,
        use_cuda_kernel: bool = False,
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if use_cuda_kernel:
            from components.alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(channels, alpha_logscale=log_scale)
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=log_scale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l_ in self.convs:
            remove_weight_norm(l_)
