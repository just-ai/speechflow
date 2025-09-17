import typing as tp

from dataclasses import dataclass

from torch import Tensor

from speechflow.data_pipeline.collate_functions.tts_collate import LinguisticFeatures
from speechflow.data_pipeline.core import TrainData

__all__ = ["AlignerForwardInput", "AlignerForwardOutput", "AlignerForwardTarget"]


@dataclass
class AlignerForwardInput(TrainData):
    lang_id: Tensor = None
    speaker_id: Tensor = None
    speaker_emb: Tensor = None
    speech_quality_emb: Tensor = None
    transcription: Tensor = None
    transcription_lengths: Tensor = None
    ling_feat: tp.Optional[LinguisticFeatures] = None
    spectrogram: Tensor = None
    spectrogram_lengths: Tensor = None
    spectral_flatness: Tensor = None
    ssl_feat: Tensor = None
    ssl_feat_lengths: Tensor = None
    input_lengths: Tensor = None
    output_lengths: Tensor = None
    additional_inputs: tp.Optional[tp.Dict[str, tp.Any]] = None


@dataclass
class AlignerForwardOutput(TrainData):
    aligning_path: Tensor = None
    mle_loss: Tensor = None
    duration_loss: Tensor = None
    spectrogram: Tensor = None
    output_lengths: Tensor = None
    output_mask: Tensor = None
    additional_content: tp.Dict = None  # type: ignore


@dataclass
class AlignerForwardTarget(TrainData):
    transcription: Tensor = None
    input_lengths: Tensor = None
    output_lengths: Tensor = None
