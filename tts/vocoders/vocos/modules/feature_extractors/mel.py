import typing as tp

import torch
import torchaudio

from speechflow.training.base_model import BaseTorchModelParams
from tts.vocoders.data_types import VocoderForwardInput
from tts.vocoders.vocos.modules.feature_extractors import FeatureExtractor
from tts.vocoders.vocos.utils.tensor_utils import safe_log

__all__ = ["MelFeatures", "MelFeaturesParams"]


class MelFeaturesParams(BaseTorchModelParams):
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 80
    padding: tp.Literal["center", "same"] = "center"


class MelFeatures(FeatureExtractor):
    params: MelFeaturesParams

    def __init__(self, params: MelFeaturesParams):
        super().__init__(params)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=params.sample_rate,
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            n_mels=params.n_mels,
            center=params.padding == "center",
            power=1,
        )

    def forward(self, inputs: VocoderForwardInput, **kwargs):
        if self.params.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(
                inputs.waveform, (pad // 2, pad // 2), mode="reflect"
            )
        else:
            audio = inputs.waveform

        mel = self.mel_spec(audio)

        return safe_log(mel), {}
