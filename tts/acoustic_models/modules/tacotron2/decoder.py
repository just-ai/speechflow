import torch

from torch import nn
from torch.nn import functional as F

from speechflow.utils.tensor_utils import get_mask_from_lengths
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import DecoderParams
from tts.acoustic_models.modules.tacotron2.layers import LinearNorm
from tts.acoustic_models.modules.tacotron2.modules import Attention, Prenet

__all__ = ["Tacotron2Decoder", "Tacotron2DecoderParams"]


class Tacotron2DecoderParams(DecoderParams):
    n_frames_per_step: int = 1
    prenet_dim: int = 256
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31
    max_decoder_steps: int = 1500
    gate_threshold: float = 0.5
    p_attention_dropout: float = 0.1
    p_decoder_dropout: float = 0.1
    early_stopping: bool = True


class Tacotron2Decoder(Component):
    """Decoder from Tacotron2 paper https://github.com/NVIDIA/DeepLearningExamples/tree/ma
    ster/PyTorch/SpeechSynthesis/Tacotron2."""

    params: Tacotron2DecoderParams

    def __init__(self, params: Tacotron2DecoderParams, input_dim):
        super().__init__(params, input_dim)

        self.n_mel_channels = params.decoder_output_dim
        self.n_frames_per_step = params.n_frames_per_step
        self.encoder_embedding_dim = input_dim
        self.attention_rnn_dim = params.decoder_inner_dim
        self.decoder_rnn_dim = params.decoder_inner_dim
        self.prenet_dim = params.prenet_dim
        self.attention_dim = params.attention_dim
        self.attention_location_n_filters = params.attention_location_n_filters
        self.attention_location_kernel_size = params.attention_location_kernel_size
        self.max_decoder_steps = params.max_decoder_steps
        self.gate_threshold = params.gate_threshold
        self.p_attention_dropout = params.p_attention_dropout
        self.p_decoder_dropout = params.p_decoder_dropout
        self.early_stopping = params.early_stopping

        self.prenet = Prenet(
            self.n_mel_channels * self.n_frames_per_step,
            [self.prenet_dim, self.prenet_dim],
        )

        self.attention_rnn = nn.LSTMCell(
            self.prenet_dim + self.encoder_embedding_dim, self.attention_rnn_dim
        )

        self.attention_layer = Attention(
            self.attention_rnn_dim,
            self.encoder_embedding_dim,
            self.attention_dim,
            self.attention_location_n_filters,
            self.attention_location_kernel_size,
        )

        self.decoder_rnn = nn.LSTMCell(
            self.attention_rnn_dim + self.encoder_embedding_dim, self.decoder_rnn_dim, 1
        )

        self.linear_projection = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            self.n_mel_channels * self.n_frames_per_step,
        )

        self.gate_layer = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            1,
            bias=True,
            w_init_gain="sigmoid",
        )

    @property
    def output_dim(self):
        return self.params.decoder_output_dim

    def get_go_frame(self, memory):
        """Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(
            B, self.n_mel_channels * self.n_frames_per_step, dtype=dtype, device=device
        )
        return decoder_input

    def initialize_decoder_states(self, memory):
        """Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device
        )
        attention_cell = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device
        )

        decoder_hidden = torch.zeros(B, self.decoder_rnn_dim, dtype=dtype, device=device)
        decoder_cell = torch.zeros(B, self.decoder_rnn_dim, dtype=dtype, device=device)

        attention_weights = torch.zeros(B, MAX_TIME, dtype=dtype, device=device)
        attention_weights_cum = torch.zeros(B, MAX_TIME, dtype=dtype, device=device)
        attention_context = torch.zeros(
            B, self.encoder_embedding_dim, dtype=dtype, device=device
        )

        processed_memory = self.attention_layer.memory_layer(memory)

        return (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        )

    def parse_decoder_inputs(self, decoder_inputs):
        """Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1,
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = alignments.transpose(0, 1).contiguous()
        # (T_out, B) -> (B, T_out)
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        shape = (mel_outputs.shape[0], -1, self.n_mel_channels)
        mel_outputs = mel_outputs.view(*shape)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(
        self,
        decoder_input,
        attention_hidden,
        attention_cell,
        decoder_hidden,
        decoder_cell,
        attention_weights,
        attention_weights_cum,
        attention_context,
        memory,
        processed_memory,
        mask,
    ):
        """Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, attention_context), -1)

        attention_hidden, attention_cell = self.attention_rnn(
            cell_input, (attention_hidden, attention_cell)
        )
        attention_hidden = F.dropout(
            attention_hidden, self.p_attention_dropout, self.training
        )

        attention_weights_cat = torch.cat(
            (attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)), dim=1
        )
        attention_context, attention_weights = self.attention_layer(
            attention_hidden, memory, processed_memory, attention_weights_cat, mask
        )

        attention_weights_cum += attention_weights
        decoder_input = torch.cat((attention_hidden, attention_context), -1)

        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input, (decoder_hidden, decoder_cell)
        )
        decoder_hidden = F.dropout(decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (decoder_hidden, attention_context), dim=1
        )
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (
            decoder_output,
            gate_prediction,
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
        )

    def get_audio_feat(self, inputs):
        if hasattr(inputs.model_inputs, self.params.target):
            return getattr(inputs.model_inputs, self.params.target)
        else:
            return inputs.additional_content.get(self.params.target)

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        content = inputs.get_content()[0]
        content_lengths = inputs.get_content_lengths()[0]

        memory = content
        memory_lengths = content_lengths
        decoder_inputs = self.get_audio_feat(inputs).transpose(2, 1)

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        mask = get_mask_from_lengths(memory_lengths)
        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        ) = self.initialize_decoder_states(memory)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            (
                mel_output,
                gate_output,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
            ) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask,
            )

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            torch.stack(mel_outputs), torch.stack(gate_outputs), torch.stack(alignments)
        )

        inputs.additional_content["alignments"] = alignments

        return DecoderOutput(
            content=mel_outputs.transpose(2, 1),
            content_lengths=inputs.model_inputs.output_lengths,
            gate=gate_outputs,
            embeddings=inputs.embeddings,
            model_inputs=inputs.model_inputs,
            additional_content=inputs.additional_content,
            additional_losses=inputs.additional_losses,
        )

    def inference_step(self, inputs: VarianceAdaptorOutput, **kwargs) -> DecoderOutput:  # type: ignore
        content = inputs.get_content()[0]
        content_lengths = inputs.get_content_lengths()[0]

        memory = content
        memory_lengths = content_lengths
        decoder_input = self.get_go_frame(memory)

        mask = get_mask_from_lengths(memory_lengths)
        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        ) = self.initialize_decoder_states(memory)

        mel_lengths = torch.zeros(
            [memory.size(0)], dtype=torch.int32, device=memory.device
        )
        not_finished = torch.ones(
            [memory.size(0)], dtype=torch.int32, device=memory.device
        )

        mel_outputs, gate_outputs, alignments = (
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )
        first_iter = True
        while True:
            decoder_input = self.prenet(decoder_input)
            (
                mel_output,
                gate_output,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
            ) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask,
            )

            if first_iter:
                mel_outputs = mel_output.unsqueeze(0)
                gate_outputs = gate_output
                alignments = attention_weights
                first_iter = False
            else:
                mel_outputs = torch.cat((mel_outputs, mel_output.unsqueeze(0)), dim=0)
                gate_outputs = torch.cat((gate_outputs, gate_output), dim=0)
                alignments = torch.cat((alignments, attention_weights), dim=0)

            dec = (
                torch.le(torch.sigmoid(gate_output), self.gate_threshold)
                .to(torch.int32)
                .squeeze(1)
            )

            not_finished = not_finished * dec
            mel_lengths += not_finished

            if self.early_stopping and torch.sum(not_finished) == 0:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        inputs.additional_content["alignments"] = alignments

        outputs = DecoderOutput.copy_from(inputs)
        outputs.set_content(mel_outputs.transpose(2, 1), mel_lengths.long())
        outputs.gate = gate_outputs
        return outputs
