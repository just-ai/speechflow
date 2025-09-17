import random
import typing as tp
import logging

from pathlib import Path

import numpy as np
import torch
import wespeaker
import torch.nn.functional as F

from speechbrain.pretrained import EncoderClassifier

from speechflow.data_pipeline.core import BaseDSProcessor, PipeRegistry
from speechflow.data_pipeline.datasample_processors.data_types import AudioDataSample
from speechflow.data_pipeline.datasample_processors.tts_singletons import (
    MeanBioEmbeddings,
)
from speechflow.io import AudioChunk, tp_PATH
from speechflow.utils.fs import get_root_dir
from speechflow.utils.init import lazy_initialization

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
except ImportError as e:
    print(f"Resemblyzer import failed: {e}")

__all__ = ["VoiceBiometricProcessor", "mean_bio_embedding", "wespeaker"]

LOGGER = logging.getLogger("root")


class VoiceBiometricProcessor(BaseDSProcessor):
    """Voice Biometric Processor for computing speaker embeddings.

    Args:
        model_type (str): Model type. Must be "resemblyzer" or "speechbrain".
        device (str): Torch device.

    """

    def __init__(
        self,
        model_type: tp.Literal["resemblyzer", "speechbrain", "wespeaker"] = "resemblyzer",
        model_name: tp.Optional[tp_PATH] = None,
        max_audio_duration: tp.Optional[float] = None,
        fast_resample: bool = True,
        random_crop: bool = False,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self._model_type = model_type
        self._model_name = model_name
        self._max_audio_duration = max_audio_duration
        self._fast_resample = fast_resample
        self._random_crop = random_crop
        self._encoder = None

        if random_crop and max_audio_duration is None:
            raise ValueError("Set the crop size in max_audio_duration argument.")

        if self._model_type.startswith("resemblyzer"):
            self._sample_rate = 16000
            self._embedding_dim = 256
        elif self._model_type.startswith("speechbrain"):
            self._sample_rate = 16000
            self._embedding_dim = 192
            if self._model_name is None:
                self._model_name = "spkrec-ecapa-voxceleb"
        elif self._model_type.startswith("wespeaker"):
            self._sample_rate = 16000
            self._embedding_dim = 256
            if self._model_name is None:
                self._model_name = "english"
        else:
            raise ValueError(
                f"Available model_type's: resemblyzer, speechbrain, "
                f"but got model_type={self._model_type}!"
            )

    def init(self):
        super().init()
        if self._model_type.startswith("resemblyzer"):
            self._encoder = VoiceEncoder(device=self.device, verbose=False)
        elif self._model_type.startswith("speechbrain"):
            if not Path(self._model_name).exists():
                self._model_name = (
                    get_root_dir()
                    / f"speechflow/data/temp/biometric/speechbrain/{self._model_name}"
                )
                self._model_name.mkdir(parents=True, exist_ok=True)
            self._encoder = EncoderClassifier.from_hparams(
                source=f"speechbrain/{Path(self._model_name).name}",
                # savedir=Path(self._model_name).absolute().as_posix(),
                run_opts={"device": self.device},
            )
        elif self._model_type.startswith("wespeaker"):
            if Path(self._model_name).exists():
                self._encoder = wespeaker.load_model_local(
                    Path(self._model_name).absolute().as_posix()
                )
            else:
                self._encoder = wespeaker.load_model(str(self._model_name))
            self._encoder.model.eval()
            if self.device != "cpu":
                if self.device == "cuda":
                    self._encoder.set_gpu(0)
                else:
                    self._encoder.set_gpu(int(self.device[5:]))
        else:
            raise ValueError(
                f"Available model_type's: resemblyzer, speechbrain, wespeaker, "
                f"but got model_type={self._model_type}!"
            )

    @property
    def model(self) -> torch.nn.Module:
        return self._encoder

    @property
    def target_sample_rate(self) -> int:
        return self._sample_rate

    @property
    def embedding_dim(self) -> int:
        """Size (length) of biometric embedding.

        Returns:
            embedding_dim (int): Size of biometric embedding vector

        """
        return self._embedding_dim

    @torch.inference_mode()
    def _get_embedding(self, waveform):
        assert self._encoder is not None
        if self._model_type.startswith("resemblyzer"):
            waveform = preprocess_wav(waveform)
            embedding = self._encoder.embed_utterance(waveform)
        elif self._model_type.startswith("speechbrain"):
            waveform = torch.from_numpy(waveform).to(self.device)
            embedding = self._encoder.encode_batch(waveform).cpu().numpy().flatten()
        elif self._model_type.startswith("wespeaker"):
            waveform = torch.from_numpy(waveform).to(self.device)
            feats = self._encoder.compute_fbank(
                waveform.unsqueeze(0), sample_rate=self._sample_rate, cmn=True
            )
            feats = feats.unsqueeze(0).to(self.device)
            outputs = self._encoder.model(feats)
            embedding = outputs[-1][0].cpu().numpy()
        else:
            raise NotImplementedError
        return embedding

    @PipeRegistry.registry(inputs={"audio_chunk"}, outputs={"speaker_emb"})
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

        ds.speaker_emb = self._get_embedding(audio_chunk.waveform)
        return ds.to_numpy()

    @lazy_initialization
    def compute_sm_loss(self, audio: torch.Tensor, audio_gt: torch.Tensor):
        """Compute speaker similarity loss.

        Args:
            audio:
            audio_gt:
            sample_rate:

        Returns:

        """

        if self._model_type.startswith("speechbrain"):

            def compute_feat(_waveform):
                if _waveform.ndim == 2:
                    _waveform = _waveform.squeeze(0)
                return _waveform

            def compute_embedding(_feat):
                return self._encoder.encode_batch(_feat).squeeze(1)

        elif self._model_type.startswith("wespeaker"):

            def compute_feat(_waveform):
                return self._encoder.compute_fbank(
                    _waveform, sample_rate=self.target_sample_rate, cmn=True
                )

            def compute_embedding(_feat):
                return self._encoder.model(_feat)[-1]

        else:
            raise NotImplementedError

        if self._encoder.device != audio.device:
            if hasattr(self._encoder, "model"):
                self._encoder.model.to(audio.device)
            else:
                self._encoder.to(audio.device)
                self._encoder.device = audio.device

        def get_feat(_audio):
            feats = []
            for waveform in _audio:
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                feats.append(compute_feat(waveform))
            return torch.stack(feats)

        _feat = get_feat(audio)
        _sp_emb = compute_embedding(_feat)

        with torch.no_grad():
            _feat_gt = get_feat(audio_gt)
            _sp_emb_gt = compute_embedding(_feat_gt)

        cos = F.cosine_similarity(_sp_emb, _sp_emb_gt.detach())
        return 1.0 - (1.0 + cos) / 2.0


@PipeRegistry.registry(inputs={"speaker_name"}, outputs={"speaker_emb_mean"})
def mean_bio_embedding(
    ds: AudioDataSample,
    mean_bio_embeddings: MeanBioEmbeddings,
):
    ds.speaker_emb_mean = mean_bio_embeddings.get_embedding(ds.speaker_name)
    return ds


if __name__ == "__main__":
    from speechflow.utils.profiler import Profiler

    wav_path = get_root_dir() / "tests/data/test_audio.wav"
    ref_waveform = AudioChunk(wav_path).load().trim(end=5)

    for model in ["resemblyzer", "speechbrain", "wespeaker"]:
        print("----", model.upper(), "----")
        bio = VoiceBiometricProcessor(model_type=model)
        clean_emb = bio.process(AudioDataSample(audio_chunk=ref_waveform)).speaker_emb
        clean_emb = torch.from_numpy(clean_emb)

        for noise_scale in [0.001, 0.01, 0.1, 0.5]:
            noise_waveform = ref_waveform.copy()
            noise_waveform.data += np.random.random(noise_waveform.data.shape) * noise_scale  # type: ignore

            with Profiler():
                noise_emb = bio.process(
                    AudioDataSample(audio_chunk=noise_waveform)
                ).speaker_emb
                noise_emb = torch.from_numpy(noise_emb)

            dist = 1.0 - F.cosine_similarity(
                clean_emb.unsqueeze(0), noise_emb.unsqueeze(0)
            )
            print(f"noise_scale: {noise_scale}, dist: {float(dist)}")

        try:
            sr = ref_waveform.sr
            t = torch.FloatTensor(4, sr)
            t.requires_grad = True
            loss = bio.compute_sm_loss(t, t)
            assert loss.grad_fn is not None
            print("speaker similarity loss is supported")
        except Exception as e:
            print(e)
