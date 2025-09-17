import typing as tp

from dataclasses import dataclass

import numpy as np
import torch

from torch import Tensor

from speechflow.data_pipeline.collate_functions.audio_collate import (
    AudioCollate,
    AudioCollateOutput,
)
from speechflow.data_pipeline.collate_functions.utils import (
    collate_sequence,
    collete_1d,
    collete_2d,
)
from speechflow.data_pipeline.datasample_processors.data_types import (
    SpectrogramDataSample,
)

__all__ = [
    "SpectrogramCollate",
    "SpectrogramCollateOutput",
]


@dataclass
class SpectrogramCollateOutput(AudioCollateOutput, SpectrogramDataSample):
    spectrogram: Tensor = None
    spectrogram_lengths: Tensor = None

    def __post_init__(self):
        super().__post_init__()

        if self.averages is None:
            self.averages = {}


class SpectrogramCollate(AudioCollate):
    def collate(self, batch: tp.List[SpectrogramDataSample]) -> SpectrogramCollateOutput:  # type: ignore
        audio_collated = super().collate(batch)  # type: ignore
        collated = SpectrogramCollateOutput(**audio_collated.to_dict())  # type: ignore

        spec_multiple = self.multiple_values.get("spectrogram")

        pad_val = batch[0].get_param_val("min_level_db", 0.0)
        collated.magnitude, mag_spec_lens = collete_2d(
            batch, "magnitude", pad_val, spec_multiple
        )

        pad_val = batch[0].get_param_val("mel_min_val", 0.0)
        collated.mel, mel_spec_lens = collete_2d(batch, "mel", pad_val, spec_multiple)

        collated.gate, gate_lens = collate_sequence(batch, "gate", 0, spec_multiple)
        collated.energy, en_lens = collete_1d(batch, "energy", 0, spec_multiple)
        collated.spectral_flatness, sf_lens = collete_1d(
            batch, "spectral_flatness", 0, spec_multiple
        )
        collated.spectral_envelope, env_lens = collete_1d(
            batch, "spectral_envelope", 0, spec_multiple
        )
        collated.pitch, pitch_lens = collate_sequence(batch, "pitch", 0, spec_multiple)

        collated.averages = {}
        if batch[0].averages is not None:
            for name in batch[0].averages.keys():
                collated.averages[name] = torch.tensor(
                    [[sample.averages[name]] for sample in batch]
                )

        collated.ranges = {}
        if batch[0].ranges is not None:
            for name in batch[0].ranges.keys():
                collated.ranges[name] = torch.from_numpy(
                    np.stack([sample.ranges[name] for sample in batch])
                )

        for lens in [
            mag_spec_lens,
            mel_spec_lens,
            en_lens,
            sf_lens,
            env_lens,
            pitch_lens,
        ]:
            if mag_spec_lens is None:
                mag_spec_lens = lens
            if mag_spec_lens is not None and lens is not None:
                assert (mag_spec_lens == lens).all()

        if collated.mel is not None:
            collated.spectrogram = collated.mel
            collated.spectrogram_lengths = mel_spec_lens
        else:
            collated.spectrogram = collated.magnitude
            collated.spectrogram_lengths = mag_spec_lens

        return collated
