from torch import nn
from torch.nn import functional as F

from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput
from tts.acoustic_models.modules.params import EncoderParams
from tts.acoustic_models.modules.tacotron2.layers import ConvNorm

__all__ = ["Tacotron2Encoder", "Tacotron2EncoderParams"]


class Tacotron2EncoderParams(EncoderParams):
    n_convolutions: int = 3
    kernel_size: int = 5


class Tacotron2Encoder(Component):
    """Encoder from Tacotron2 paper https://github.com/NVIDIA/DeepLearningExamples/tree/ma
    ster/PyTorch/SpeechSynthesis/Tacotron2."""

    params: Tacotron2EncoderParams

    def __init__(self, params: Tacotron2EncoderParams, input_dim):
        super().__init__(params, input_dim)

        encoder_n_convolutions = params.n_convolutions
        encoder_embedding_dim = params.encoder_inner_dim
        encoder_kernel_size = params.kernel_size

        self.feat_proj = nn.Linear(input_dim, params.encoder_inner_dim)
        self.output_proj = nn.Linear(params.encoder_inner_dim, params.encoder_output_dim)

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            encoder_embedding_dim,
            int(encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    @property
    def output_dim(self):
        return self.params.encoder_output_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        content = inputs.get_content()[0]
        content_lengths = inputs.get_content_lengths()[0]

        x = self.feat_proj(content).transpose(2, 1)
        input_lengths = content_lengths

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False
        )

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return EncoderOutput(
            content=self.output_proj(outputs),
            content_lengths=inputs.content_lengths,
            embeddings=inputs.embeddings,
            model_inputs=inputs.model_inputs,
            additional_content=inputs.additional_content,
            additional_losses=inputs.additional_losses,
        )
