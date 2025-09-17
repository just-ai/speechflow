import typing as tp

from dataclasses import dataclass

import torch

from speechflow.data_pipeline.core import TrainData
from speechflow.io import AudioChunk
from tts.acoustic_models.data_types import TTSForwardInput, TTSForwardOutput, TTSTarget

__all__ = [
    "VocoderTarget",
    "VocoderForwardInput",
    "VocoderForwardOutput",
]


@dataclass
class VocoderTarget(TTSTarget):
    pass


@dataclass
class VocoderForwardInput(TTSForwardInput):
    lpc: torch.Tensor = None
    lpc_feat: torch.Tensor = None

    @staticmethod
    def init_from_tts(
        tts_input: TTSForwardInput, tts_output: TTSForwardOutput
    ) -> "VocoderForwardInput":
        voc_in = tts_input
        voc_in.spectrogram = tts_output.after_postnet_spectrogram
        voc_in.spectrogram_lengths = tts_output.spectrogram_lengths
        voc_in.energy = tts_output.variance_predictions.get("energy")
        voc_in.pitch = tts_output.variance_predictions.get("pitch")
        return voc_in  # type: ignore


@dataclass
class VocoderForwardOutput(TrainData):
    waveform: torch.Tensor = None
    waveform_length: torch.Tensor = None
    audio_chunk: AudioChunk = None
    additional_content: tp.Dict[str, torch.Tensor] = None

    def __post_init__(self):
        if self.additional_content is None:
            self.additional_content = {}
