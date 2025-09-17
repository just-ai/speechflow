import uuid
import random
import typing as tp
import logging
import tempfile

from pathlib import Path

import cdpam
import numpy as np
import torch

from torch.nn import functional as F

from speechflow.data_pipeline.core import BaseDSProcessor
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.data_types import AudioDataSample
from speechflow.io import AudioChunk
from speechflow.utils.init import lazy_initialization

__all__ = ["SpeechQualityAssessment"]

LOGGER = logging.getLogger("root")


class SpeechQualityAssessment(BaseDSProcessor):
    def __init__(
        self,
        model_type: str = "nisqa",
        max_audio_duration: tp.Optional[float] = 10,
        fast_resample: bool = True,
        random_crop: bool = False,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self._model_type = model_type
        self._max_audio_duration = max_audio_duration
        self._fast_resample = fast_resample
        self._random_crop = random_crop

        if random_crop and max_audio_duration is None:
            raise ValueError("Set the crop size in max_audio_duration argument.")

        if self._model_type == "cdpam":
            self._cdpam = None
            self._sample_rate = 22050
            self._embedding_dim = 512
        elif self._model_type == "nisqa":
            self._nisqa = None
            self._sample_rate = 48000
            self._embedding_dim = 4
        else:
            raise ValueError(
                f"Available model_type's: cdpam, nisqa, "
                f"but got model_type={model_type}!"
            )

        self.logging_params(self.get_config_from_locals())

    def init(self):
        from speechflow.thirdparty.nisqa.NISQA_model import nisqaModel

        super().init()
        if self._model_type.startswith("cdpam"):
            self._cdpam = cdpam.CDPAM(dev=self.device)
        elif self._model_type.startswith("nisqa"):
            self._nisqa = nisqaModel(device=self.device)
        else:
            raise ValueError(
                f"Available model_type's: cdpam, nisqa, "
                f"but got model_type={self._model_type}!"
            )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @PipeRegistry.registry(inputs={"audio_chunk"}, outputs={"speech_quality_emb"})
    @lazy_initialization
    def process(self, ds: AudioDataSample) -> AudioDataSample:
        ds = super().process(ds)

        assert np.issubdtype(
            ds.audio_chunk.dtype, np.floating
        ), "Audio data must be floating-point!"
        audio_chunk = ds.audio_chunk

        if self._random_crop:
            begin = audio_chunk.duration - self._max_audio_duration
            if begin > 0:
                begin *= random.random()
                end = begin + self._max_audio_duration
                audio_chunk = audio_chunk.trim(begin=begin, end=end)
        else:
            if (
                self._max_audio_duration
                and audio_chunk.duration > self._max_audio_duration
            ):
                audio_chunk = audio_chunk.trim(end=self._max_audio_duration)

        audio_chunk = audio_chunk.resample(sr=self._sample_rate, fast=self._fast_resample)

        if self._model_type == "cdpam":
            wave = torch.round(torch.from_numpy(audio_chunk.waveform) * 32768.0)
            wave = wave.float().unsqueeze(0).unsqueeze(1).to(self.device)
            with torch.no_grad():
                _, a1, c1 = self._cdpam.model.base_encoder.forward(wave)

            ds.speech_quality_emb = F.normalize(a1, dim=1).cpu()[0]
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_path = Path(tmp_dir) / f"{uuid.uuid4()}.wav"
                audio_chunk.save(temp_path)
                val = self._nisqa.predict(temp_path)
                ds.speech_quality_emb = torch.FloatTensor([v for v in val.values()])[:-1]

        return ds.to_numpy()


if __name__ == "__main__":
    from speechflow.utils.fs import get_root_dir
    from speechflow.utils.profiler import Profiler

    wav_path = get_root_dir() / "tests/data/test_audio.wav"
    ref_wave = AudioChunk(wav_path).load()

    for model in ["cdpam", "nisqa"]:
        print(model)
        sqa = SpeechQualityAssessment(model_type=model)
        clean_emb = (
            sqa.process(AudioDataSample(audio_chunk=ref_wave))
            .to_tensor()
            .speech_quality_emb
        )
        print(clean_emb.shape)

        for noise_scale in [0.001, 0.01, 0.1]:
            noise_wave = ref_wave.copy()
            noise_wave.data += np.random.random(noise_wave.data.shape) * noise_scale  # type: ignore

            with Profiler():
                noise_emb = (
                    sqa.process(AudioDataSample(audio_chunk=noise_wave))
                    .to_tensor()
                    .speech_quality_emb
                )
                print(noise_emb[:10])

            dist = F.l1_loss(clean_emb, noise_emb)
            print(f"noise_scale: {noise_scale}, dist: {float(dist)}")
