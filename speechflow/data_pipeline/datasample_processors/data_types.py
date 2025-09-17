import typing as tp

from dataclasses import dataclass

import torch

from multilingual_text_parser.data_types import Sentence

from speechflow.data_pipeline.core.datasample import (
    DataSample,
    MovableToDevice,
    ToNumpy,
    ToTensor,
    tp_DATA,
)
from speechflow.io import AudioChunk, Timestamps

__all__ = [
    "ImageDataSample",
    "AudioDataSample",
    "SpectrogramDataSample",
    "TextDataSample",
    "TTSDataSample",
    "SSLFeatures",
    "AudioCodecFeatures",
    "ProsodySSMLDataSample",
    "ProsodyPredictionDataSample",
]


@dataclass(eq=False)
class ImageDataSample(DataSample):
    image: tp_DATA = None


@dataclass
class SSLFeatures(ToTensor, ToNumpy, MovableToDevice):
    encoder_feat: tp_DATA = None
    logits: tp_DATA = None
    centroids: tp_DATA = None
    tokens_text: tp.Tuple[str, ...] = None
    tokens_id: tp_DATA = None
    text: str = None

    def __getitem__(self, item):
        return self.encoder_feat[item]

    def get(self):
        return self.encoder_feat


@dataclass
class AudioCodecFeatures(ToTensor, ToNumpy, MovableToDevice):
    encoder_feat: tp_DATA = None
    waveform: tp_DATA = None

    def __getitem__(self, item):
        return self.encoder_feat[item]

    def get(self):
        return self.encoder_feat


@dataclass(eq=False)
class AudioDataSample(DataSample):
    audio_chunk: AudioChunk = None
    lang: str = None
    lang_id: tp_DATA = None
    speaker_name: str = None
    speaker_id: tp_DATA = 0
    speaker_emb: tp_DATA = None
    speaker_emb_mean: tp_DATA = None
    speech_quality_emb: tp_DATA = None
    lpc_feat: tp_DATA = None
    ssl_feat: tp.Union[SSLFeatures, tp_DATA] = None
    ac_feat: tp.Union[AudioCodecFeatures, tp_DATA] = None
    mu_law_waveform: tp_DATA = None
    lpc_waveform: tp_DATA = None

    def __len__(self):
        if self.audio_chunk and self.audio_chunk.duration:
            return int(self.audio_chunk.duration * 1000)  # in milliseconds
        else:
            return 0

    def __lt__(self, other):
        return len(self) < len(other)


@dataclass(eq=False)
class SpectrogramDataSample(AudioDataSample):
    magnitude: tp_DATA = None
    mel: tp_DATA = None
    energy: tp_DATA = None
    spectral_flatness: tp_DATA = None
    spectral_tilt: tp_DATA = None
    spectral_envelope: tp_DATA = None
    pitch: tp_DATA = None
    averages: tp.Dict[str, tp_DATA] = None
    ranges: tp.Dict[str, tp_DATA] = None
    gate: tp_DATA = None

    def __len__(self):
        if self.magnitude is not None:
            return self.magnitude.shape[0]
        elif self.audio_chunk:
            return super().__len__()
        else:
            return 0


@dataclass(eq=False)
class TextDataSample(DataSample):
    sent: Sentence = None
    transcription_text: tp.Tuple[str, ...] = None
    transcription_id: tp_DATA = None
    ling_feat: tp.Dict[str, tp_DATA] = None
    intonation_type: int = None
    word_lengths: tp_DATA = None
    synt_lengths: tp_DATA = None
    xpbert_feat: tp_DATA = None
    lm_feat: tp_DATA = None
    pad_token_id: int = 0
    sil_token_id: int = 0

    def __str__(self) -> str:
        return self.sent.text_orig if self.sent else super().__str__()


@dataclass(eq=False)
class ProsodySSMLDataSample(DataSample):
    temp_modifier: tp_DATA = None
    pitch_modifier: tp_DATA = None
    volume_modifier: tp_DATA = None


@dataclass(eq=False)
class TTSDataSample(SpectrogramDataSample, TextDataSample, ProsodySSMLDataSample):
    word_timestamps: Timestamps = None
    phoneme_timestamps: tp.List[Timestamps] = None
    durations: tp_DATA = None
    invert_durations: tp_DATA = None
    transcription_id_by_frames: tp_DATA = None
    aggregated: tp.Dict[str, tp_DATA] = None
    pauses_durations: torch.Tensor = None


@dataclass(eq=False)
class ProsodyPredictionDataSample(TTSDataSample):
    input_ids: tp_DATA = None
    binary: tp_DATA = None
    category: tp_DATA = None
    attention_mask: tp_DATA = None
    pad_id: int = None
    lang: str = None
    word_ids: tp_DATA = None
    seed_by_words: tp.List[int] = None

    def __len__(self) -> int:
        return self.input_ids.shape[0]
