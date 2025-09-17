from itertools import groupby

import torch

from torch import nn
from torch.nn import functional as F

from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import DecoderParams

__all__ = ["TacoDecoder", "TacoDecoderParams"]


class TacoDecoderParams(DecoderParams):
    prenet_dim: int = 256
    rnn_dim: int = 512

    # projection
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"


class TacoDecoder(Component):
    params: TacoDecoderParams

    def __init__(self, params: TacoDecoderParams, input_dim):
        super().__init__(params, input_dim)

        self.dec_step = DecoderStep(
            encoder_emb_dim=input_dim,
            frame_dim=params.decoder_output_dim,
            prenet_dim=params.prenet_dim,
            rnn_dim=params.rnn_dim,
            p_dropout=params.projection_p_dropout,
            activation_fn=params.projection_activation_fn,
        )

    @property
    def output_dim(self):
        return self.params.decoder_output_dim

    def forward_step(self, inputs: VarianceAdaptorOutput, mask=None) -> DecoderOutput:  # type: ignore
        x = inputs.get_content()[0]
        target = getattr(inputs.model_inputs, self.params.target)

        memory = x.permute(1, 0, 2)
        if target is not None:
            target = target.permute(1, 0, 2)

        self.dec_step.initialize_states(memory.shape[1], memory.device)
        frame = self.dec_step.get_go_frame(memory.shape[1], memory.device)

        gate_outputs = []
        decoder_outputs = []
        decoder_context_outputs = []

        if not self.training and mask is not None and mask.shape[0] == 1:
            group_mask = groupby(mask.tolist()[0])  # type: ignore
        else:
            group_mask = [(True, [True] * memory.shape[0])]  # type: ignore

        begin = 0
        for flag, seq in group_mask:
            end = begin + len(list(seq))
            if flag and not self.training:
                encoder_context = memory[begin:end]
                frames = torch.cat([frame.unsqueeze(0), target[begin : end - 1]])
                (next_frame, gate, self.dec_step.states, dec_context) = self.dec_step(
                    frames,
                    encoder_context,
                    self.dec_step.states,
                )
                decoder_context_outputs.append(dec_context)
                decoder_outputs.append(next_frame)
                gate_outputs.append(gate)
                frame = next_frame[:, -1, :]
            else:
                local_decoder_context_outputs = []
                local_decoder_outputs = []
                local_gate_outputs = []

                for idx in range(begin, end):
                    (
                        next_frame,
                        gate,
                        self.dec_step.states,
                        dec_context,
                    ) = self.dec_step(
                        frame,
                        memory[idx],
                        self.dec_step.states,
                    )

                    if mask is not None:  # inference as imputer
                        frame = target[idx].clone()
                        frame[~mask[:, idx]] = next_frame[~mask[:, idx]]
                    else:  # train or inference as tts
                        frame = target[idx] if self.training else next_frame

                    local_decoder_context_outputs.append(dec_context)
                    local_decoder_outputs.append(next_frame)
                    local_gate_outputs.append(gate)

                decoder_context_outputs.append(
                    torch.stack(local_decoder_context_outputs).transpose(0, 1)
                )
                decoder_outputs.append(torch.stack(local_decoder_outputs).transpose(0, 1))
                gate_outputs.append(torch.stack(local_gate_outputs).transpose(0, 1))

            begin = end

        decoder_context = torch.cat(decoder_context_outputs, dim=1).contiguous()
        spec = torch.cat(decoder_outputs, dim=1).contiguous()
        gate = torch.cat(gate_outputs, dim=1).contiguous()

        outputs = DecoderOutput.copy_from(inputs).set_content(spec)
        outputs.gate = gate
        outputs.hidden_state = decoder_context
        return outputs


def get_lstm_cell(lstm_layer: torch.nn.LSTM) -> torch.nn.LSTMCell:
    lstm_cell = nn.LSTMCell(lstm_layer.input_size, lstm_layer.hidden_size)
    lstm_cell.weight_hh = lstm_layer.weight_hh_l0
    lstm_cell.bias_hh = lstm_layer.bias_hh_l0
    lstm_cell.weight_ih = lstm_layer.weight_ih_l0
    lstm_cell.bias_ih = lstm_layer.bias_ih_l0
    lstm_cell.to(lstm_layer.weight_hh_l0.device)
    return lstm_cell


class Prenet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layers_num: int = 2):
        super().__init__()
        sizes = [output_dim] * layers_num
        in_sizes = [input_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )

        self.dropout = nn.Dropout()

    def forward(self, x):
        for linear in self.layers:
            x = self.dropout(F.relu(linear(x)))
        return x


class DecoderStep(nn.Module):
    def __init__(
        self,
        encoder_emb_dim: int,
        frame_dim: int,
        prenet_dim: int,
        rnn_dim: int,
        p_dropout: float = 0.0,
        activation_fn: str = "Identity",
    ):
        super().__init__()
        self.mask_value = -4.0

        self.encoder_emb_dim = encoder_emb_dim
        self.frame_dim = frame_dim
        self.prenet_dim = prenet_dim
        self.rnn_dim = rnn_dim

        self.prenet_layer = Prenet(frame_dim, prenet_dim)

        self.lstm = nn.LSTM(prenet_dim + self.encoder_emb_dim, rnn_dim, batch_first=True)
        self.lstm_cell = get_lstm_cell(self.lstm)

        self.proj = Regression(
            rnn_dim, frame_dim, p_dropout=p_dropout, activation_fn=activation_fn
        )
        self.gate_layer = nn.Linear(rnn_dim + self.encoder_emb_dim, 1)

    def initialize_states(self, batch_size, device):
        """Initializes attention rnn states, decoder rnn states, attention weights,
        attention cumulative weights, attention context, stores memory and stores
        processed memory."""

        self.states = {
            "rnn_hidden": torch.zeros((batch_size, self.rnn_dim), device=device),
            "rnn_cell": torch.zeros((batch_size, self.rnn_dim), device=device),
        }

    def get_go_frame(self, batch_size, device):
        """Gets all zeros frames to use as first decoder input."""
        return torch.zeros((batch_size, self.frame_dim)).to(device) + self.mask_value

    def one_step(
        self,
        frame,
        memory_emb,
        input_states,
    ):
        """Decoder step using stored states, attention and memory."""

        rnn_hidden = input_states["rnn_hidden"]
        rnn_cell = input_states["rnn_cell"]

        decoder_input = self.prenet_layer(frame)

        rnn_input = torch.cat((decoder_input, memory_emb), -1)
        rnn_hidden, rnn_cell = self.lstm_cell(rnn_input, (rnn_hidden, rnn_cell))

        dec_context = rnn_hidden

        gate = self.gate_layer(torch.cat((rnn_hidden, memory_emb), -1))
        decoder_output = self.frame_prediction(dec_context)

        vars = locals()
        output_states = {s: vars.get(s, input_states[s]) for s in input_states}
        return decoder_output, gate, output_states, dec_context

    def multi_step(
        self,
        frame,
        memory_emb,
        input_states,
    ):
        rnn_hidden = input_states["rnn_hidden"].unsqueeze(0)
        rnn_cell = input_states["rnn_cell"].unsqueeze(0)

        decoder_input = self.prenet_layer(frame)

        rnn_input = torch.cat((decoder_input, memory_emb), -1)
        rnn_input = rnn_input.transpose(0, 1)
        dec_context, (rnn_hidden, rnn_cell) = self.lstm(rnn_input, (rnn_hidden, rnn_cell))

        decoder_output = self.proj(dec_context)
        gate = self.gate_layer(torch.cat((dec_context, memory_emb.transpose(0, 1)), -1))

        rnn_hidden = rnn_hidden.squeeze(0)
        rnn_cell = rnn_cell.squeeze(0)

        vars = locals()
        output_states = {s: vars.get(s, input_states[s]) for s in input_states}
        return decoder_output, gate, output_states, dec_context

    def forward(
        self,
        frame,
        memory_emb,
        input_states,
    ):
        """Decoder step using stored states, attention and memory."""
        if frame.ndim == 2:
            return self.one_step(frame, memory_emb, input_states)
        else:
            return self.multi_step(frame, memory_emb, input_states)
