import typing as tp

from dataclasses import dataclass

from torch import Tensor

from speechflow.data_pipeline.collate_functions.utils import (
    collate_integers,
    collate_vectors,
    collete_2d,
)
from speechflow.data_pipeline.core import BaseCollate, BaseCollateOutput
from speechflow.data_pipeline.datasample_processors.data_types import AudioDataSample

__all__ = [
    "AudioCollate",
    "AudioCollateOutput",
]


@dataclass
class AudioCollateOutput(BaseCollateOutput, AudioDataSample):
    mu_law_waveform_lengths: Tensor = None
    lpc_waveform_lengths: Tensor = None
    lpc_feat_lengths: Tensor = None
    ssl_feat_lengths: Tensor = None
    ac_feat_lengths: Tensor = None


class AudioCollate(BaseCollate):
    def collate(self, batch: tp.List[AudioDataSample]) -> AudioCollateOutput:  # type: ignore
        collated = super().collate(batch)  # type: ignore
        collated = AudioCollateOutput(**collated.to_dict())  # type: ignore

        collated.mu_law_waveform, collated.mu_law_waveform_lengths = collete_2d(
            batch, "mu_law_waveform", self.pad_values, self.multiple_values
        )
        collated.lpc_waveform, collated.lpc_waveform_lengths = collete_2d(
            batch, "lpc_waveform", self.pad_values, self.multiple_values
        )

        collated.lang_id = collate_integers(batch, "lang_id")
        collated.speaker_id = collate_integers(batch, "speaker_id")

        collated.speaker_emb = collate_vectors(batch, "speaker_emb")
        collated.speaker_emb_mean = collate_vectors(batch, "speaker_emb_mean")
        collated.speech_quality_emb = collate_vectors(batch, "speech_quality_emb")

        collated.lpc_feat, lpc_feat_lens = collete_2d(
            batch, "lpc_feat", self.pad_values, self.multiple_values
        )
        collated.ssl_feat, collated.ssl_feat_lengths = collete_2d(
            batch, "ssl_feat", self.pad_values, self.multiple_values
        )
        collated.ac_feat, collated.ac_feat_lengths = collete_2d(
            batch, "ac_feat", self.pad_values, self.multiple_values
        )

        return collated
