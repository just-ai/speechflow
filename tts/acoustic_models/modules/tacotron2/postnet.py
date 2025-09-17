import torch

from torch import nn
from torch.nn import functional as F

from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, PostnetOutput
from tts.acoustic_models.modules.params import PostnetParams
from tts.acoustic_models.modules.tacotron2.layers import ConvNorm

__all__ = ["Tacotron2Postnet", "Tacotron2PostnetParams"]


class Tacotron2PostnetParams(PostnetParams):
    kernel_size: int = 5


class Tacotron2Postnet(Component):
    params: Tacotron2PostnetParams

    def __init__(self, params: Tacotron2PostnetParams, input_dim: int):
        super().__init__(params, input_dim)

        postnet_embedding_dim = params.postnet_inner_dim
        postnet_kernel_size = params.kernel_size
        postnet_n_convolutions = params.postnet_num_layers

        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    input_dim,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    params.postnet_output_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(params.postnet_output_dim),
            )
        )
        self.n_convs = len(self.convolutions)

    @property
    def output_dim(self):
        return self.params.postnet_output_dim

    def forward_step(self, inputs: DecoderOutput) -> PostnetOutput:  # type: ignore
        content = inputs.get_content()
        x = content[-1].transpose(2, 1)

        i = 0
        for conv in self.convolutions:
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
            i += 1

        x_out = content[-1] + x.transpose(1, 2)

        return PostnetOutput.copy_from(inputs).set_content(x_out)
