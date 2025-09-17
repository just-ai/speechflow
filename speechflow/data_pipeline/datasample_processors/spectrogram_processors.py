import pickle
import random
import typing as tp
import logging
import multiprocessing as mp

import numpy as np
import torch
import librosa
import pyworld as pw
import torchcrepe
import numpy.typing as npt

from librosa import filters as librosa_stft
from scipy import ndimage, signal
from scipy.interpolate import interp1d
from scipy.signal import cwt, savgol_filter

from speechflow.data_pipeline.core.base_ds_processor import (
    BaseDSProcessor,
    ComputeBackend,
)
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.algorithms.audio_processing import (
    nvidia_stft,
)
from speechflow.data_pipeline.datasample_processors.algorithms.audio_processing.fft_window import (
    FFTWindow,
)
from speechflow.data_pipeline.datasample_processors.algorithms.audio_processing.lpc_from_spectrogram import (
    LPCCompute,
    LPCDecompose,
)
from speechflow.data_pipeline.datasample_processors.algorithms.audio_processing.yin_image import (
    Yingram,
)
from speechflow.data_pipeline.datasample_processors.data_types import (
    AudioDataSample,
    SpectrogramDataSample,
)
from speechflow.data_pipeline.datasample_processors.tts_singletons import StatisticsRange
from speechflow.io import Config, Timestamps
from speechflow.logging import trace
from speechflow.utils.init import (
    get_default_args,
    init_method_from_config,
    lazy_initialization,
)

__all__ = [
    "SpectralProcessor",
    "MelProcessor",
    "NemoMelProcessor",
    "PitchProcessor",
    "LPCProcessor",
    "pitch_to_wavelet",
    "signal_enhancement",
    "normalize",
    "average_by_time",
    "clip",
]

LOGGER = logging.getLogger("root")

try:
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
    from nemo.collections.audio.modules.transforms import AudioToSpectrogram

    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
except ImportError as e:
    if mp.current_process().name == "MainProcess":
        LOGGER.warning(f"NeMo is not available: {e}")


def _fp_eq(a, b):
    return np.abs(np.float32(a) - np.float32(b)) < 1e-5


class BaseSpectrogramProcessor(BaseDSProcessor):
    def process(self, ds: SpectrogramDataSample) -> SpectrogramDataSample:
        if ds.audio_chunk and not ds.audio_chunk.empty:
            assert np.issubdtype(
                ds.audio_chunk.waveform.dtype, np.floating
            ), "Audio data must be floating-point!"

        assert ds.audio_chunk.waveform.max() > 5.0e-3, "Sound is very quiet!"
        return super().process(ds)


class SpectralProcessor(BaseSpectrogramProcessor):
    def __init__(
        self,
        pipe: tp.Tuple[str, ...] = (),
        pipe_cfg: Config = Config.empty(),
        backend: ComputeBackend = ComputeBackend.librosa,
    ):
        super().__init__(pipe, pipe_cfg, backend)
        self.window = None
        self.stft_module: tp.Optional[nvidia_stft.STFT] = None

    @PipeRegistry.registry(
        inputs={"audio_chunk"},
        outputs={
            "magnitude",
            "energy",
            "spectral_flatness",
            "spectral_tilt",
            "spectral_envelope",
            "hop_len",
        },
    )
    def process(self, ds: SpectrogramDataSample) -> SpectrogramDataSample:
        return super().process(ds)

    def _stft(
        self,
        waveform: npt.NDArray,
        n_fft: int,
        hop_len: int,
        win_len: int,
        win_type: str = "hann",
        center: bool = True,
    ) -> tp.Union[npt.NDArray, torch.Tensor]:

        if self.window is None:
            self.window = FFTWindow(win_type).get_window(win_len)

        if self.backend == ComputeBackend.librosa:
            if not center:
                pad_size = (n_fft - hop_len) // 2
                waveform = np.pad(waveform, pad_size, mode="reflect")

            stft = librosa.stft(
                y=waveform,
                n_fft=n_fft,
                hop_length=hop_len,
                win_length=win_len,
                window=self.window,
                center=center,
                pad_mode="reflect",
            )

        elif self.backend == ComputeBackend.torchaudio:
            waveform = torch.from_numpy(waveform)
            window = torch.from_numpy(self.window)
            stft = torch.stft(
                waveform, n_fft, hop_len, win_len, window=window, return_complex=True
            )

        elif self.backend == ComputeBackend.nvidia:
            if not center:
                raise ValueError("center=False is not support for nvidia backend")
            if self.stft_module is None:
                self.stft_module = nvidia_stft.STFT(
                    filter_length=n_fft,
                    hop_length=hop_len,
                    win_length=win_len,
                )

            waveform = torch.from_numpy(waveform)
            stft = self.stft_module(waveform)

        elif self.backend == ComputeBackend.nemo:
            if not center:
                raise ValueError("center=False is not support for nvidia backend")
            if self.stft_module is None:
                self.stft_module = AudioToSpectrogram(
                    fft_length=n_fft, hop_length=hop_len, magnitude_power=1.0
                )

            waveform = torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0)
            stft, _ = self.stft_module(input=waveform)
            stft = stft.squeeze(0).squeeze(0)

        else:
            raise NotImplementedError(
                f"Computing stft not implemented for {self.backend} ComputeBackend."
            )

        return stft

    def magnitude(
        self,
        ds: SpectrogramDataSample,
        n_fft: int,
        hop_len: int,
        win_len: int,
        win_type: str = "hann",
        center: bool = True,
        remove_last_frame: bool = False,
    ) -> SpectrogramDataSample:
        stft = self._stft(
            ds.audio_chunk.waveform[:-1]
            if remove_last_frame
            else ds.audio_chunk.waveform,
            n_fft,
            hop_len,
            win_len,
            win_type,
            center,
        )

        if self.backend == ComputeBackend.librosa:
            ds.magnitude = np.abs(stft).T

        elif self.backend == ComputeBackend.torchaudio:
            ds.magnitude = torch.abs(stft).T

        elif self.backend == ComputeBackend.nvidia:
            ds.magnitude = torch.sqrt(torch.sum(stft**2, dim=2)).T

        elif self.backend == ComputeBackend.nemo:
            ds.magnitude = stft.T

        else:
            raise NotImplementedError(
                f"Computing magnitude not implemented for {self.backend} ComputeBackend."
            )

        return ds

    def amp_to_db(
        self,
        ds: SpectrogramDataSample,
        multiplier: float = 1.0,
        a_min: float = 1e-5,
        a_max: tp.Optional[float] = None,
    ) -> SpectrogramDataSample:
        if self.backend == ComputeBackend.librosa:
            ds.magnitude = np.log(np.clip(ds.magnitude, a_min=a_min, a_max=a_max))

            if multiplier != 1.0:
                ds.magnitude *= multiplier

        else:
            raise NotImplementedError(
                f"Computing amp_to_db not implemented for {self.backend} ComputeBackend."
            )

        return ds

    def energy(self, ds: SpectrogramDataSample) -> SpectrogramDataSample:
        if self.backend == ComputeBackend.librosa:
            ds.energy = np.linalg.norm(ds.magnitude, axis=-1)  # type: ignore

        elif self.backend in [
            ComputeBackend.torchaudio,
            ComputeBackend.nvidia,
            ComputeBackend.nemo,
        ]:
            ds.energy = torch.norm(ds.magnitude, dim=-1)

        else:
            raise NotImplementedError(
                f"Computing energy not implemented for {self.backend} ComputeBackend."
            )

        return ds

    def spectral_flatness(self, ds: SpectrogramDataSample) -> SpectrogramDataSample:
        if self.backend == ComputeBackend.librosa:
            ds.spectral_flatness = librosa.feature.spectral_flatness(S=ds.magnitude.T, power=2.0)[0]  # type: ignore
            ds.spectral_flatness = 1.0 - (ds.spectral_flatness * 100.0).clip(
                min=0.0, max=0.99
            )
        else:
            raise NotImplementedError(
                f"Computing spectral flatness not implemented for {self.backend} ComputeBackend."
            )

        return ds

    def spectral_tilt(self, ds: SpectrogramDataSample) -> SpectrogramDataSample:
        if self.backend == ComputeBackend.librosa:
            total_bins = ds.magnitude.shape[-1]

            # convert spectral values to dB
            dB_val = 20 * (np.log10(ds.magnitude / 0.0002))

            # find maximum dB value, for rescaling purposes
            maxdB = np.max(dB_val, axis=0)
            mindB = np.min(
                dB_val, axis=0
            )  # this is wrong in Owren's script, where mindB = 0
            rangedB = maxdB - mindB

            # stretch the spectrum to a normalized range that matches the number of frequency values
            scalingConstant = (total_bins - 1) / rangedB
            scaled_dB_val = (dB_val + abs(mindB)) * scalingConstant

            # find slope
            sumXX = np.zeros(dB_val.shape[0], dtype=np.float32)
            sumXY = np.zeros(dB_val.shape[0], dtype=np.float32)
            sumX = sum(range(total_bins)) * np.ones(dB_val.shape[0], dtype=np.float32)
            sumY = np.sum(scaled_dB_val, axis=-1)

            for bin in range(total_bins):
                currentX = bin * np.ones(dB_val.shape[0], dtype=np.float32)
                sumXX += currentX**2
                sumXY += currentX * scaled_dB_val[:, bin]

            sXX = sumXX - ((sumX * sumX) / total_bins)
            sXY = sumXY - ((sumX * sumY) / total_bins)
            ds.spectral_tilt = sXY / sXX
            ds.spectral_tilt = ds.spectral_tilt.max() - ds.spectral_tilt

        else:
            raise NotImplementedError(
                f"Computing spectral flatness not implemented for {self.backend} ComputeBackend."
            )

        return ds

    def spectral_envelope(
        self, ds: SpectrogramDataSample, cutoff: int = 3, n_bins: int = 80
    ) -> SpectrogramDataSample:
        min_level = np.exp(-100 / 20 * np.log(10))

        def zero_one_norm(S):
            S_norm = S - np.min(S)
            S_norm /= np.max(S_norm)
            return S_norm

        if self.backend == ComputeBackend.librosa:
            D = ds.magnitude
            ceps = np.fft.irfft(np.log(D + 1e-6), axis=-1).real  # [T, F]
            F = ceps.shape[1]
            lifter = np.zeros(F)
            lifter[:cutoff] = 1
            lifter[cutoff] = 0.5
            lifter = np.diag(lifter)
            envelope = np.matmul(ceps, lifter)
            envelope = np.abs(np.exp(np.fft.rfft(envelope, axis=-1)))
            envelope = 20 * np.log10(np.maximum(min_level, envelope)) - 16
            envelope = (envelope + 100) / 100
            envelope = zero_one_norm(envelope)

            ds.spectral_envelope = signal.resample(envelope, n_bins, axis=-1).astype(
                np.float32
            )

        else:
            raise NotImplementedError(
                f"Computing spectral envelope not implemented for {self.backend} ComputeBackend."
            )

        return ds


class MelProcessor(BaseSpectrogramProcessor):
    def __init__(
        self,
        pipe: tp.Tuple[str, ...] = (),
        pipe_cfg: Config = Config.empty(),
        backend: ComputeBackend = ComputeBackend.librosa,
    ):
        super().__init__(pipe, pipe_cfg, backend)
        self.mel_basis: tp.Optional[npt.NDArray] = None
        self.mel_scale: tp.Optional[torch.Tensor] = None
        self.mel_module: tp.Optional[nvidia_stft.Linear2Mel] = None
        self.inv_mel_basis: tp.Optional[npt.NDArray] = None

    @PipeRegistry.registry(inputs={"magnitude"}, outputs={"mel"})
    def process(self, ds: SpectrogramDataSample) -> SpectrogramDataSample:
        return super().process(ds)

    @property
    def min_level_db(self) -> float:
        multiplier = get_default_args(self.amp_to_db)["multiplier"]
        a_min = get_default_args(self.amp_to_db)["a_min"]
        return multiplier * np.log(a_min)

    @property
    def max_abs_value(self) -> float:
        return get_default_args(self.normalize)["max_abs_value"]

    def load_precomputed_mel(
        self, ds: SpectrogramDataSample, p: float = 0.5
    ) -> SpectrogramDataSample:
        if (p < 0) or (p > 1):
            raise ValueError(
                f"Probability of loading pre-computed mel must be in range [0, 1]. Got p={p}."
            )

        if random.random() < p:
            synth_mel_path = ds.file_path.with_suffix(".mel")
            if not synth_mel_path.exists():
                LOGGER.warning(
                    trace(
                        self,
                        message=f"File with pre-computed mel for {ds.file_path} not found.",
                        full=False,
                    )
                )
            else:
                load_mel = pickle.loads(synth_mel_path.read_bytes())
                if load_mel.shape != ds.mel.shape:
                    LOGGER.error(
                        trace(
                            self,
                            f"shape: {load_mel.shape} != {ds.mel.shape}",
                            full=False,
                        )
                    )
                    raise ValueError("Dimensions of the spectrum is not equal.")

                ds.mel = load_mel

        return ds

    def linear_to_mel(
        self,
        ds: SpectrogramDataSample,
        sample_rate: int = None,  # type: ignore
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = None,  # type: ignore
        librosa_htk: bool = False,
    ) -> SpectrogramDataSample:
        if ds.audio_chunk is not None:
            sample_rate = ds.audio_chunk.sr
        else:
            sample_rate = ds.get_param_val("sample_rate", sample_rate)

        if self.backend == ComputeBackend.librosa:
            if self.mel_basis is None:
                n_fft = (ds.magnitude.shape[-1] - 1) * 2
                self.mel_basis = librosa_stft.mel(
                    sr=sample_rate,
                    n_fft=n_fft,
                    n_mels=n_mels,
                    fmin=f_min,
                    fmax=f_max,
                    htk=librosa_htk,
                )

            ds.mel = np.dot(self.mel_basis, ds.magnitude.T).T

        elif self.backend == ComputeBackend.torchaudio:
            from torchaudio import functional as F
            from torchaudio import transforms

            if self.mel_scale is None:
                n_stft = ds.magnitude.shape[-1]
                self.mel_scale = transforms.MelScale(
                    n_mels,
                    sample_rate,
                    f_min,
                    f_max,
                    n_stft,
                )
                f_max = float(sample_rate // 2) if f_max is None else f_max
                self.mel_scale.fb = F.melscale_fbanks(
                    n_stft,
                    f_min,
                    f_max,
                    n_mels,
                    sample_rate,
                    norm="slaney",
                )

            ds.mel = self.mel_scale(torch.from_numpy(ds.magnitude.T)).t()

        elif self.backend == ComputeBackend.nvidia:
            if self.mel_module is None:
                n_fft = (ds.magnitude.shape[-1] - 1) * 2
                self.mel_module = nvidia_stft.Linear2Mel(
                    n_fft, n_mels, sample_rate, f_min, f_max
                )

            ds.mel = self.mel_module(torch.from_numpy(ds.magnitude.T)).t()

        else:
            raise NotImplementedError(
                f"Computing linear_to_mel not implemented for {self.backend} ComputeBackend."
            )

        return ds

    def mel_to_linear(
        self,
        ds: SpectrogramDataSample,
        sample_rate: int = None,  # type: ignore
        n_fft: int = None,  # type: ignore
        f_min: float = 0.0,
        f_max: float = None,  # type: ignore
        librosa_htk: bool = False,
    ) -> SpectrogramDataSample:
        n_fft = ds.get_param_val("n_fft", n_fft)
        f_min = ds.get_param_val("f_min", f_min)
        f_max = ds.get_param_val("f_max", f_max)
        librosa_htk = ds.get_param_val("librosa_htk", librosa_htk)

        if ds.audio_chunk is not None:
            sample_rate = ds.audio_chunk.sr
        else:
            sample_rate = ds.get_param_val("sample_rate", sample_rate)

        if self.backend == ComputeBackend.librosa:
            if self.inv_mel_basis is None:
                n_mels = ds.mel.shape[-1]
                mel_basis = librosa_stft.mel(
                    sr=sample_rate,
                    n_fft=n_fft,
                    n_mels=n_mels,
                    fmin=f_min,
                    fmax=f_max,
                    htk=librosa_htk,
                )
                self.inv_mel_basis = np.linalg.pinv(mel_basis, rcond=1e-5)  # type: ignore

            ds.magnitude = np.dot(self.inv_mel_basis, ds.mel.T).T
            ds.magnitude = np.maximum(f_min, ds.magnitude)

        else:
            raise NotImplementedError

        return ds

    def amp_to_db(
        self,
        ds: SpectrogramDataSample,
        multiplier: float = 1.0,
        a_min: float = 1e-5,
        a_max: tp.Optional[float] = None,
    ) -> SpectrogramDataSample:
        if self.backend == ComputeBackend.librosa:
            ds.mel = np.log(np.clip(ds.mel, a_min=a_min, a_max=a_max))

        elif self.backend in [
            ComputeBackend.torchaudio,
            ComputeBackend.nvidia,
        ]:
            ds.mel = torch.log(torch.clamp(ds.mel, min=a_min, max=a_max))

        else:
            raise NotImplementedError(
                f"Computing amp_to_db not implemented for {self.backend} ComputeBackend."
            )

        if multiplier != 1.0:
            ds.mel *= multiplier

        min_level_db = multiplier * np.log(a_min)
        ds.transform_params.setdefault("amp_to_db", dict())
        ds.transform_params["amp_to_db"]["min_level_db"] = min_level_db
        ds.transform_params["mel_min_val"] = min_level_db
        return ds

    def db_to_amp(
        self,
        ds: SpectrogramDataSample,
        multiplier: float = 1.0,
    ) -> SpectrogramDataSample:
        multiplier = ds.get_param_val("multiplier", multiplier)
        if multiplier != 1.0:
            ds.mel *= 1.0 / multiplier

        if self.backend == ComputeBackend.librosa:
            ds.mel = np.exp(ds.mel)

        elif self.backend in [
            ComputeBackend.torchaudio,
            ComputeBackend.nvidia,
        ]:
            ds.mel = torch.exp(ds.mel)

        else:
            raise NotImplementedError

        return ds

    def normalize(
        self,
        ds: SpectrogramDataSample,
        max_abs_value: float = 4.0,
        min_level_db: float = None,  # type: ignore
    ) -> SpectrogramDataSample:
        min_level_db = ds.get_param_val("min_level_db", min_level_db)
        if min_level_db is None:
            min_level_db = self.min_level_db

        if self.backend == ComputeBackend.librosa:
            ds.mel = np.clip(
                (2 * max_abs_value) * ((ds.mel - min_level_db) / (-min_level_db))
                - max_abs_value,
                a_min=-max_abs_value,
                a_max=None,
            )

        elif self.backend in [
            ComputeBackend.torchaudio,
            ComputeBackend.nvidia,
        ]:
            ds.mel = torch.clamp(
                (2 * max_abs_value) * ((ds.mel - min_level_db) / (-min_level_db))
                - max_abs_value,
                min=-max_abs_value,
            )

        else:
            raise NotImplementedError(
                f"Computing normalize not implemented for {self.backend} ComputeBackend."
            )

        ds.transform_params["mel_min_val"] = -max_abs_value
        return ds

    def denormalize(
        self,
        ds: SpectrogramDataSample,
        max_abs_value: float = None,  # type: ignore
        min_level_db: float = None,  # type: ignore
    ) -> SpectrogramDataSample:
        max_abs_value = ds.get_param_val("max_abs_value", max_abs_value)
        if max_abs_value is None:
            max_abs_value = self.max_abs_value
        min_level_db = ds.get_param_val("min_level_db", min_level_db)
        if min_level_db is None:
            min_level_db = self.min_level_db

        if self.backend == ComputeBackend.librosa:
            ds.mel = (
                (np.clip(ds.mel, -max_abs_value, a_max=None) + max_abs_value)
                * (-min_level_db)
                / (2 * max_abs_value)
            ) + min_level_db

        elif self.backend in [
            ComputeBackend.torchaudio,
            ComputeBackend.nvidia,
        ]:
            ds.mel = (
                (torch.clamp(ds.mel, -max_abs_value) + max_abs_value)
                * (-min_level_db)
                / (2 * max_abs_value)
            ) + min_level_db

        else:
            raise NotImplementedError(
                f"Computing denormalize not implemented for {self.backend} ComputeBackend."
            )

        ds.transform_params["mel_min_val"] = min_level_db
        return ds


class NemoMelProcessor(BaseSpectrogramProcessor):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_len: int,
        win_len: int,
        n_mels: int,
        **kwargs,
    ):
        super().__init__()
        self._mel_cfg = self.get_config_from_locals(
            ignore=["win_len", "hop_len", "n_mels"]
        )
        self._mel_cfg["window_size"] = win_len / sample_rate
        self._mel_cfg["window_stride"] = hop_len / sample_rate
        self._mel_cfg["features"] = n_mels

        self._mel_module = None

        self.logging_params(self.get_config_from_locals())

    def init(self):
        super().init()
        self._mel_module = AudioToMelSpectrogramPreprocessor(**self._mel_cfg)
        self._mel_module.eval()

    @PipeRegistry.registry(inputs={"audio_chunk"}, outputs={"mel"})
    @lazy_initialization
    def process(self, ds: SpectrogramDataSample) -> SpectrogramDataSample:
        ds = super().process(ds)

        waveform = ds.audio_chunk.waveform
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform_length = torch.LongTensor([waveform.shape[-1]])
        mel, _ = self._mel_module.get_features(
            input_signal=waveform, length=waveform_length
        )
        ds.mel = mel.squeeze(0).T
        return ds.to_numpy()


class PitchProcessor(BaseSpectrogramProcessor):
    def __init__(
        self,
        method: tp.Literal["pyworld", "torchcrepe", "yingram"] = "pyworld",
        f0_min: float = 80,
        f0_max: float = 880,
        n_bins: int = 80,
        pyworld_frame_period: tp.Literal["default", "adaptive"] = "default",
        torchcrepe_model: tp.Literal["full", "tiny"] = "full",
        torchcrepe_batch_size: int = 128,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.method = method
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.n_bins = n_bins
        self.pyworld_frame_period = pyworld_frame_period
        self.torchcrepe_model = torchcrepe_model
        self.torchcrepe_batch_size = torchcrepe_batch_size
        self.logging_params(self.get_config_from_locals())

    @PipeRegistry.registry(inputs={"audio_chunk"}, outputs={"pitch"})
    @lazy_initialization  # required for automatic device selection
    def process(self, ds: SpectrogramDataSample) -> SpectrogramDataSample:
        ds = super().process(ds)

        if ds.audio_chunk.sr != 16000 and self.method in ["yin", "torchcrepe"]:
            audio_chunk = ds.audio_chunk.resample(16000, fast=True)
        else:
            audio_chunk = ds.audio_chunk

        sample_rate = audio_chunk.sr
        waveform = audio_chunk.waveform
        hop_len = ds.get_param_val("hop_len")

        if self.method == "pyworld":
            assert sample_rate >= 16000, "sample rate must be greater or equal 16KHz!"

            if self.pyworld_frame_period == "default":
                frame_period = pw.default_frame_period
            else:
                frame_period = 1000 * hop_len / sample_rate

            waveform = audio_chunk.as_type(np.float64).waveform
            f0, _ = pw.dio(
                waveform,
                audio_chunk.sr,
                frame_period=frame_period,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
            )

        elif self.method == "yin":
            raise ValueError("YIN method is deprecated!")
            # f0, _, _, _ = compute_yin(
            #     waveform,
            #     sr=sample_rate,
            #     w_len=n_fft,
            #     w_step=hop_len,
            #     f0_min=self.f0_min,
            #     f0_max=self.f0_max,
            #     harmo_thresh=0.25,
            # )
            # f0 = np.asarray(f0, dtype=np.float32)

        elif self.method == "crepe":
            raise ValueError("CREPE method is deprecated!")
            # step_size = hop_len * 1000 / sample_rate
            # _, f0, confidence, _ = crepe.predict(
            #     waveform,
            #     sample_rate,
            #     step_size=step_size,
            #     viterbi=True,
            #     verbose=0,
            #     model_capacity=self.model,
            # )
            # f0[confidence < 0.5] = 0
            # f0 = f0.clip(min=0, max=self.f0_max)

        elif self.method == "torchcrepe":
            with torch.inference_mode():
                waveform = torch.from_numpy(waveform).unsqueeze(0)
                f0, periodicity = torchcrepe.predict(
                    waveform,
                    sample_rate,
                    None,
                    self.f0_min,
                    self.f0_max,
                    batch_size=self.torchcrepe_batch_size,
                    return_periodicity=True,
                    model=self.torchcrepe_model,
                    device=self.device,
                )
            periodicity = torchcrepe.filter.mean(periodicity, win_length=3)
            periodicity = torchcrepe.threshold.Silence(-60.0)(
                periodicity, waveform, sample_rate
            )
            f0 = torchcrepe.threshold.At(0.2)(f0, periodicity)
            f0 = torchcrepe.filter.mean(f0, win_length=3)
            f0 = f0.squeeze(0).cpu().numpy()
            f0[np.isnan(f0)] = 0.0

        elif self.method == "yingram":
            assert (
                22050 <= sample_rate <= 24000
            ), "sample rate must be equal 22050Hz or 24000Hz!"
            if not hasattr(self, "yingram"):
                yingram = Yingram(
                    strides=hop_len,
                    windows=2048,  # for 22050Hz only
                    lmin=22,  # for 22050Hz only
                    lmax=2047,  # for 22050Hz only
                    bins=20,
                    sr=sample_rate,
                ).to(self.device)
                setattr(self, "yingram", yingram)
            else:
                yingram = getattr(self, "yingram")

            with torch.inference_mode():
                f0 = yingram(torch.FloatTensor(waveform).to(self.device)).cpu()
                f0 = torch.cat([f0, torch.zeros((f0.shape[0], 1))], dim=1)
                f0 = np.clip(f0.transpose(1, 0).numpy().T, a_min=0, a_max=4)

        else:
            raise NotImplementedError(
                f"Method {self.method} not implemented in PitchProcessor."
            )

        if f0.ndim == 1:
            if f0.shape[0] != ds.magnitude.shape[0]:
                f0_max = f0.max()
                f0 = ndimage.zoom(
                    f0,
                    ds.magnitude.shape[0] / f0.shape[0],
                    order=1,
                )
                f0 = np.clip(f0, a_min=0.0, a_max=f0_max)

                if (f0.shape[0] - ds.magnitude.shape[0]) == 1:
                    f0 = f0[:-1]

                assert f0.shape[0] == ds.magnitude.shape[0], "dim size mismatch!"
        else:
            if f0.shape[1] != ds.magnitude.shape[0]:
                f0 = ndimage.zoom(
                    f0,
                    (ds.magnitude.shape[0] / f0.shape[0], self.n_bins / f0.shape[1]),
                    order=1,
                )

        ds.pitch = f0.astype(np.float32)
        # self._plot_pitch(ds)
        return ds

    @staticmethod
    def _plot_pitch(ds: SpectrogramDataSample):
        import matplotlib
        import matplotlib.pyplot as plt

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        matplotlib.use("TkAgg")

        fig, ax = plt.subplots(1, 1, dpi=160, facecolor="w")  # type: ignore
        _data = np.flip(ds.mel.T, axis=0)
        im = ax.imshow(_data)
        divider = make_axes_locatable(ax)
        fig.colorbar(im, cax=divider.append_axes("right", size=0.25, pad=0.05))

        if ds.pitch.ndim == 1:
            x = np.arange(len(ds.pitch))
            pitch = ds.mel.shape[-1] - 1 - ds.pitch / ds.pitch.max() * 40
            ax.plot(x, pitch, c="r", lw=1, label="pitch")
            ax.legend(
                loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize="small"
            )
        else:
            fig, ax = plt.subplots(1, 1, dpi=160, facecolor="w")  # type: ignore
            _data = np.flip(ds.pitch, axis=0)
            im = ax.imshow(_data)
            divider = make_axes_locatable(ax)
            fig.colorbar(im, cax=divider.append_axes("right", size=0.25, pad=0.05))

        plt.show()


class LPCProcessor(BaseSpectrogramProcessor):
    def __init__(
        self,
        pipe: tp.Tuple[str, ...] = (),
        pipe_cfg: Config = Config.empty(),
        backend: ComputeBackend = ComputeBackend.numpy,
    ):
        super().__init__(pipe, pipe_cfg, backend)
        self._lpc_compute_linear: tp.Optional[LPCCompute] = None
        self._lpc_compute_mel: tp.Optional[LPCCompute] = None
        self._lpc_decompose: tp.Optional[LPCDecompose] = None
        self._mel_proc = MelProcessor()
        self._spec_proc = SpectralProcessor()

    @PipeRegistry.registry(
        inputs={"magnitude", "mel"},
        outputs={"lpc", "lpc_feat"},
    )
    def process(self, ds: SpectrogramDataSample) -> SpectrogramDataSample:
        return super().process(ds)

    def lpc_from_linear(
        self,
        ds: SpectrogramDataSample,
        order: int = 16,
        ac_adjustment: bool = True,
    ):
        if self._lpc_compute_linear is None:
            self._lpc_compute_linear = LPCCompute(
                order=order, ac_adjustment=ac_adjustment
            )

        ds.lpc = self._lpc_compute_linear.linear_to_lpc(ds.magnitude.T).T
        return ds

    def lpc_from_mel(
        self,
        ds: SpectrogramDataSample,
        order: int = 16,
        ac_adjustment: bool = True,
        power: float = 1.0,
    ):
        if self._lpc_compute_mel is None:
            self._lpc_compute_mel = LPCCompute(order=order, ac_adjustment=ac_adjustment)

        temp_ds = ds.copy()

        if "normalize" in ds.transform_params:
            temp_ds = self._mel_proc.denormalize(temp_ds)
        if "amp_to_db" in ds.transform_params:
            temp_ds = self._mel_proc.db_to_amp(temp_ds)
        if "linear_to_mel" in ds.transform_params:
            temp_ds = self._mel_proc.mel_to_linear(temp_ds)

        ds.lpc_feat = self._lpc_compute_mel.linear_to_lpc(temp_ds.magnitude.T).T

        if power != 1.0:
            temp_ds.magnitude = temp_ds.magnitude**power
            if "linear_to_mel" in ds.transform_params:
                temp_ds = self._mel_proc.linear_to_mel(temp_ds)
            if "amp_to_db" in ds.transform_params:
                temp_ds = self._mel_proc.amp_to_db(temp_ds)
            if "normalize" in ds.transform_params:
                temp_ds = self._mel_proc.normalize(temp_ds)
            ds.mel = temp_ds.mel

        return ds

    def lpc_decompose(
        self,
        ds: AudioDataSample,
        ulaw_bits: int = 10,
        add_noise: bool = False,
        noise_std: float = 2,
        frame_size: tp.Optional[int] = None,
    ):
        if self._lpc_decompose is None:
            self._lpc_decompose = LPCDecompose(
                frame_size=frame_size if frame_size else ds.get_param_val("hop_len"),
                ulaw_bits=ulaw_bits,
                add_noise=add_noise,
                noise_std=noise_std,
            )

        ds.lpc_waveform = self._lpc_decompose(ds.audio_chunk.waveform, ds.lpc_feat.T)
        ds.lpc_waveform = ds.lpc_waveform.reshape((-1, ds.lpc_waveform.shape[2]))

        """
        # Reconstruction waveform from LPC

        from speechflow.io import AudioChunk

        ds.lpc_feat = ds.lpc_feat.T
        ds.lpc_waveform = ds.lpc_waveform.T
        hop_len = ds.get_param_val("hop_len")

        err = ds.lpc_waveform[3:4, :].reshape(-1, hop_len)
        err = LPCDecompose.torch_float_2_uint(
            torch.from_numpy(err), bits=ulaw_bits
        ).numpy()
        err = LPCDecompose.torch_ulaw2lin(
            torch.from_numpy(err), bits=ulaw_bits
        ).numpy()
        accumulator = np.zeros_like(ds.lpc_feat)

        lpc_reconstruction = []
        for i in range(hop_len):
            y_pred = -(ds.lpc_feat * accumulator).sum(axis=0)
            y_rel = (y_pred + err[:, i]).clip(-1, 1)
            accumulator[1:, :] = accumulator[:-1, :]
            accumulator[0, :] = y_rel
            lpc_reconstruction.append(y_rel)

        rec_waveform = np.stack(lpc_reconstruction).T.reshape(ds.lpc_waveform.shape[1])

        AudioChunk(data=ds.audio_chunk.waveform, sr=ds.audio_chunk.sr).save(
            "waveform_orig.wav", overwrite=True
        )
        AudioChunk(data=rec_waveform, sr=ds.audio_chunk.sr).save(
            "lpc_reconstruction.wav", overwrite=True
        )
        """

        return ds


@PipeRegistry.registry(inputs={"pitch"}, outputs={"pitch"})
def pitch_to_wavelet(ds: SpectrogramDataSample, num_bands: int = 100):
    """Compute wavelet transform for pitch."""
    widths = np.arange(1, num_bands + 1)
    cwtmatr = cwt(ds.pitch, signal.ricker, widths)
    ds.pitch = cwtmatr.T
    return ds


@PipeRegistry.registry(
    inputs={"pitch", "energy", "spectral_flatness"},
    outputs={"pitch", "energy", "spectral_flatness"},
)
def signal_enhancement(
    ds: SpectrogramDataSample,
    attributes: tp.Union[str, tp.List[str]],
    smooth: bool = False,
    interpolate_zeros: bool = False,
    set_zero_in_pauses: bool = False,
    max_zero_interval: tp.Optional[int] = None,
    smooth_options: tp.Optional[dict] = None,
    interpolate_options: tp.Optional[dict] = None,
):
    """Perform linear interpolation for zero values."""
    attributes = [attributes] if isinstance(attributes, str) else attributes
    for attr in attributes:
        if not hasattr(ds, attr):
            raise KeyError(f"Attribute '{attr}' not found in SpecDataSample.")

        values = getattr(ds, attr)
        assert values.ndim == 1
        # old_pitch = copy(values)

        if interpolate_zeros:
            nonzero_ids = np.where(values != 0)[0]
            if len(nonzero_ids) == 0 or len(nonzero_ids) == len(values):
                continue

            if max_zero_interval:
                max_zero_interval = max(max_zero_interval, 2)

                t = max_zero_interval
                a = np.arange(len(nonzero_ids))[:-1][
                    (nonzero_ids[1:] - nonzero_ids[:-1]) >= t
                ]
                for i in a:
                    idx = np.arange(8) + i - 3
                    idx = idx[0 < idx]
                    idx = idx[idx < len(nonzero_ids)]
                    nonzero_ids = np.delete(nonzero_ids, idx)

                if nonzero_ids[0] != 0:
                    nonzero_ids = np.insert(nonzero_ids, 0, 0)
                if nonzero_ids[-1] != len(values) - 1:
                    nonzero_ids = np.insert(
                        nonzero_ids, len(nonzero_ids), len(values) - 1
                    )

            if interpolate_options is None:
                interpolate_options = {}
            interpolate_options.setdefault("kind", "linear")
            interpolate_options.setdefault("bounds_error", False)

            interp_fn = init_method_from_config(interp1d, interpolate_options)(
                nonzero_ids,
                values[nonzero_ids],
                fill_value=(values[nonzero_ids[0]], values[nonzero_ids[-1]]),
            )
            values = interp_fn(np.arange(0, len(values)))
        # interpolated = copy(values)

        if smooth:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
            if smooth_options is None:
                smooth_options = {}
            smooth_options.setdefault("window_length", 5)
            smooth_options.setdefault("polyorder", 1)
            smooth_options.setdefault("mode", "wrap")
            values = init_method_from_config(savgol_filter, smooth_options)(values)
            values = np.clip(values, a_min=0.0, a_max=None)
        # smoothed = copy(values)

        if set_zero_in_pauses:
            word_timestamps = getattr(ds, "word_timestamps")
            hop_len = ds.get_param_val("hop_len")

            if word_timestamps is not None and ds.sent is not None:
                tokens = [token.text for token in ds.sent.tokens if token.pos != "PUNCT"]
                prev_ts = 0
                for i, ts in enumerate(word_timestamps):
                    if not _fp_eq(prev_ts, ts[0]):
                        a = prev_ts * ds.audio_chunk.sr / hop_len
                        b = ts[0] * ds.audio_chunk.sr / hop_len
                        values[int(a) : int(b)] = 0.0
                    if i < len(tokens) and tokens[i] in ["<BOS>", "<EOS>", "<SIL>"]:
                        a = ts[0] * ds.audio_chunk.sr / hop_len
                        b = ts[1] * ds.audio_chunk.sr / hop_len
                        values[int(a) : int(b)] = 0.0

                    prev_ts = ts[1]

                if not _fp_eq(prev_ts, ds.audio_chunk.duration):
                    a = prev_ts * ds.audio_chunk.sr / hop_len
                    values[int(a) : ds.mel.shape[0]] = 0.0

            elif ds.phoneme_timestamps is not None and ds.symbols is not None:
                ts_phonemes = Timestamps(np.concatenate(ds.phoneme_timestamps))
                symbols = ds.symbols
                for i, ts in enumerate(ts_phonemes):
                    if i < len(symbols) and symbols[i] == "_":
                        a = ts[0] * ds.audio_chunk.sr / hop_len
                        b = ts[1] * ds.audio_chunk.sr / hop_len
                        values[int(a) : int(b)] = 0.0

        setattr(ds, attr, values.astype(np.float32))

    # _plot_pitch(old_pitch, values, smoothed, interpolated)
    # PitchProcessor._plot_pitch(ds)
    return ds


@PipeRegistry.registry(
    inputs={"pitch", "energy", "spectral_flatness"},
    outputs={"pitch", "energy", "spectral_flatness"},
)
def clip(
    ds: SpectrogramDataSample,
    attributes: tp.Union[str, tp.List[str]],
    min_value: tp.Optional[float] = None,
    max_value: tp.Optional[float] = None,
):
    """Clip acoustic attributes. Better applied before normalization.

    :param ds: SpecDataSample
    :param attributes: str, list
    :param min_value: Minimum value. If None, clipping is not performed on the corresponding edge.
    :param max_value: Maximum value. If None, clipping is not performed on the corresponding edge.

    """

    attributes = [attributes] if isinstance(attributes, str) else attributes
    for attr in attributes:
        if not hasattr(ds, attr):
            raise KeyError(f"Attribute '{attr}' not found in SpecDataSample.")
        value = getattr(ds, attr)
        value = np.clip(value, a_min=min_value, a_max=max_value)
        setattr(ds, attr, value)
    return ds


@PipeRegistry.registry(
    inputs={"pitch", "energy", "spectral_flatness"},
    outputs={"pitch", "energy", "spectral_flatness"},
)
def normalize(
    ds: SpectrogramDataSample,
    attributes: tp.Union[str, tp.List[str]],
    normalize_by: tp.Literal["constant", "sample", "speaker"] = "sample",
    ranges: StatisticsRange = None,  # type: ignore
    normalize_aggregate: bool = False,
    method: tp.Literal["minmax", "z-norm"] = "minmax",
    filter_outliers: bool = False,
    quantile: float = 0.98,
    min_value: tp.Optional[float] = None,
    max_value: tp.Optional[float] = None,
):
    """
    :param ds: SpecDataSample
    :param attributes: str, list
    :param normalize_by: str
    :param ranges: StatisticsRange singleton
    :param normalize_aggregate: bool
    :param method: normalization algorithm
    :param filter_outliers: bool
    :param quantile: float
    :param min_value: float
    :param max_value: float

    """
    assert normalize_by in ["sample", "speaker", "constant"]

    def reject_outliers(x, m: float = 2.0):
        return x[abs(x - np.mean(x)) < m * np.std(x)]

    attributes = [attributes] if isinstance(attributes, str) else attributes
    if ds.ranges is None:
        ds.ranges = {}

    for attr in attributes:
        if not hasattr(ds, attr):
            raise KeyError(f"Attribute '{attr}' not found in SpecDataSample.")

        if normalize_aggregate:
            values = getattr(ds, "aggregate")[attr]
        else:
            values = getattr(ds, attr)

        if values is None:
            continue

        if values.ndim != 1:
            LOGGER.warning(f"Attribute '{attr}' must be single dimension.")
            continue

        values = values.astype(np.float32)

        if normalize_by != "constant":

            if method in ["minmax", "quantile"]:
                if method == "minmax":
                    if normalize_by == "speaker":
                        a_min, a_max = ranges.get_range(attr, ds.speaker_name)  # type: ignore
                    elif normalize_by == "sample":
                        if filter_outliers:
                            clip_values = reject_outliers(values)
                        else:
                            clip_values = values
                        a_min, a_max = clip_values.min(), clip_values.max()
                    else:
                        a_min = a_max = 0
                else:
                    if normalize_by == "speaker":
                        raise NotImplementedError
                    elif normalize_by == "sample":
                        if filter_outliers:
                            clip_values = reject_outliers(values)
                        else:
                            clip_values = values
                        a_min = np.quantile(clip_values, 1 - quantile)
                        a_max = np.quantile(clip_values, quantile)
                    else:
                        a_min = a_max = 0

                if abs(a_max - a_min) < 1.0:
                    raise ValueError(f"variation {attr} is equal to zero.")

                if min_value is not None:
                    a_min = min_value
                if max_value is not None:
                    a_max = max_value

                values -= a_min
                values /= a_max - a_min

                ds.ranges[attr] = np.asarray(
                    [a_min, a_max, a_max - a_min], dtype=np.float32
                )

            elif method == "z-norm":
                if normalize_by == "speaker":
                    a_mean, a_var = ranges.get_stat(attr, ds.speaker_name)  # type: ignore
                elif normalize_by == "sample":
                    if filter_outliers:
                        clip_values = reject_outliers(values)
                    else:
                        clip_values = values
                    if min_value is not None:
                        clip_values = clip_values[clip_values >= min_value]
                    if max_value is not None:
                        clip_values = clip_values[clip_values <= max_value]
                    a_mean, a_var = clip_values.mean(), clip_values.var()
                else:
                    a_mean = a_var = 0

                if a_mean < 1.0 or a_var < 1.0:
                    raise ValueError(f"variation {attr} is equal to zero.")

                a_std = np.sqrt(a_var) * 4

                values -= a_mean
                values /= a_std

                ds.ranges[attr] = np.asarray([a_mean, a_mean, a_std], dtype=np.float32)
            else:
                raise ValueError(f"{method} algorithm not defined.")

        else:
            values -= min_value
            values /= max_value - min_value
            ds.ranges[attr] = np.asarray(
                [min_value, max_value, max_value - min_value], dtype=np.float32
            )

        if normalize_aggregate:
            getattr(ds, "aggregate")[attr] = values
        else:
            setattr(ds, attr, values)

    # PitchProcessor._plot_pitch(ds)
    return ds


@PipeRegistry.registry(
    inputs={"pitch", "energy", "spectral_flatness"}, outputs={"average"}
)
def average_by_time(
    ds: SpectrogramDataSample,
    attributes: tp.Union[str, tp.List[str]],
    use_quantile: bool = False,
    quantile: float = 0.95,
    min_value: tp.Optional[float] = None,
):
    def reject_outliers(x, m: float = 2.0):
        return x[abs(x - np.mean(x)) < m * np.std(x)]

    attributes = [attributes] if isinstance(attributes, str) else attributes

    ds.averages = {}
    for attr in attributes:
        if attr == "rate":
            ds.averages[attr] = len(ds.transcription_id) / ds.audio_chunk.duration
            continue

        if not hasattr(ds, attr):
            raise KeyError(f"Attribute '{attr}' not found in SpecDataSample.")

        values = getattr(ds, attr)
        assert values.ndim == 1

        if min_value is not None:
            values = values[values > min_value]

        if len(values) > 0:
            if use_quantile:
                val_min = np.quantile(values, 1 - quantile)
                val_max = np.quantile(values, quantile)
                val_clip = np.clip(values, val_min, val_max)
            else:
                val_clip = reject_outliers(values)

            ds.averages[attr] = np.mean(val_clip)
        else:
            ds.averages[attr] = 0.0

    return ds


def __get_spec_proc(
    n_fft: int, hop_len: int, win_len: int, n_bins: int, center: bool = True
):
    pipe = (
        "magnitude",
        "energy",
        "spectral_flatness",
        "spectral_tilt",
        "spectral_envelope",
    )
    cfg = {
        "magnitude": {
            "n_fft": n_fft,
            "hop_len": hop_len,
            "win_len": win_len,
            "center": center,
        },
        "spectral_envelope": {"n_bins": n_bins},
    }
    return SpectralProcessor(pipe, Config(cfg))


def __get_mel_proc(n_mels: int):
    pipe = (
        "linear_to_mel",
        "amp_to_db",
        "normalize",
    )
    cfg = {
        "linear_to_mel": {"n_mels": n_mels},
    }
    return MelProcessor(pipe, Config(cfg))


def __get_nemo_mel_proc(
    sample_rate: int, n_fft: int, hop_len: int, win_len: int, n_mels: int
):
    return NemoMelProcessor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_len=hop_len,
        win_len=win_len,
        n_mels=n_mels,
    )


def __get_pitch_proc(method: tp.Literal["pyworld", "torchcrepe", "yingram"]):
    return PitchProcessor(method=method, pyworld_frame_period="default")


def __get_lpc_proc(order: int):
    pipe = ("lpc_from_mel",)
    cfg = {"lpc_from_mel": {"order": order}}
    return LPCProcessor(pipe, Config(cfg))


def __plot2d(data, label, fig, ax, idx):
    im = ax[idx].imshow(np.flip(data, axis=1).T)
    divider = make_axes_locatable(ax[idx])
    fig.colorbar(im, cax=divider.append_axes("right", size=0.25, pad=0.05))
    ax[idx].set_xlabel(label)


def __plot_spectrogram(audio_chunk, spec_proc, mel_proc):
    ds = SpectrogramDataSample(audio_chunk=audio_chunk)
    ds = spec_proc.process(ds)
    ds = mel_proc.process(ds)

    invert_ds = ds.copy()
    invert_ds.magnitude = None
    invert_ds = mel_proc.denormalize(invert_ds)
    invert_ds = mel_proc.db_to_amp(invert_ds)
    invert_ds = mel_proc.mel_to_linear(invert_ds)

    fig1, ax1 = plt.subplots(2, 1, dpi=160, facecolor="w")
    for idx, name in enumerate(["mel", "spectral_envelope"]):
        __plot2d(getattr(ds, name), name, fig1, ax1, idx)

    fig2, ax2 = plt.subplots(2, 1, dpi=160, facecolor="w")
    for idx, item in enumerate([ds, invert_ds]):
        __plot2d(item.magnitude, "magnitude", fig2, ax2, idx)

    fig3, ax3 = plt.subplots(3, 1, dpi=160, facecolor="w")
    __plot2d(ds.magnitude, "magnitude", fig3, ax3, 0)
    __plot2d(invert_ds.mel, "mel", fig3, ax3, 1)
    __plot2d(ds.mel, "normalize_mel", fig3, ax3, 2)


def __plot_1d_features(audio_chunk, spec_proc, mel_proc, pitch_proc):
    if not isinstance(pitch_proc, list):
        pitch_proc = [pitch_proc]

    ds = SpectrogramDataSample(audio_chunk=audio_chunk)
    ds = spec_proc.process(ds)
    ds = mel_proc.process(ds)

    fig, ax = plt.subplots(1 + len(pitch_proc), 1, dpi=160, facecolor="w")
    for i in range(len(ax)):
        __plot2d(ds.mel, "mel", fig, ax, i)

    x = np.arange(ds.mel.shape[0])
    n_mels = ds.get_param_val("n_mels")

    energy = n_mels - 1 - ds.energy
    spectral_flatness = n_mels - 1 - ds.spectral_flatness * 20
    spectral_tilt = n_mels - 1 - ds.spectral_tilt * 40
    ax[0].plot(x, energy, c="r", lw=1, label="energy")
    ax[0].plot(x, spectral_flatness, c="b", lw=1, label="spectral_flatness")
    ax[0].plot(x, spectral_tilt, c="y", lw=1, label="spectral_tilt")

    for i, _proc in enumerate(pitch_proc):
        ds = _proc.process(ds.copy())
        pitch = n_mels - 1 - ds.pitch / ds.pitch.max() * 60
        ax[i + 1].plot(x, pitch, c="indigo", lw=1, label=f"pitch[{_proc.method}]")

    for i in range(len(ax)):
        ax[i].legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize="small"
        )


def __plot_2d_features(audio_chunk, spec_proc, mel_proc, pitch_proc, lpc_proc):
    ds = SpectrogramDataSample(audio_chunk=audio_chunk)
    ds = spec_proc.process(ds)
    ds = mel_proc.process(ds)
    ds = pitch_proc.process(ds)
    ds = lpc_proc.process(ds)

    fig, ax = plt.subplots(3, 1, dpi=160, facecolor="w")
    for idx, name in enumerate(["mel", "pitch", "lpc_feat"]):
        __plot2d(getattr(ds, name), name, fig, ax, idx)


def __plot_nemo_features(audio_chunk, librosa_spec_proc, librosa_mel_proc, nemo_mel_proc):
    librosa_ds = SpectrogramDataSample(audio_chunk=audio_chunk)
    librosa_ds = librosa_spec_proc.process(librosa_ds)
    librosa_ds = librosa_mel_proc.process(librosa_ds)
    nemo_ds = nemo_mel_proc.process(librosa_ds.copy())

    fig, ax = plt.subplots(2, 1, dpi=160, facecolor="w")
    __plot2d(librosa_ds.mel, "librosa", fig, ax, 0)
    __plot2d(nemo_ds.mel, "nemo", fig, ax, 1)


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from speechflow.io import AudioChunk
    from speechflow.utils.fs import get_root_dir

    matplotlib.use("TkAgg")

    _file_path = get_root_dir() / "tests/data/test_audio.wav"
    _audio_chunk = AudioChunk(file_path=_file_path).load(sr=24000).trim(end=4)

    _spec_proc = __get_spec_proc(1024, 256, 1024, 80)
    _mel_proc = __get_mel_proc(80)
    _pitch_proc_1 = __get_pitch_proc("pyworld")
    _pitch_proc_2 = __get_pitch_proc("torchcrepe")
    _pitch_proc_3 = __get_pitch_proc("yingram")
    _lpc_proc = __get_lpc_proc(100)

    __plot_spectrogram(_audio_chunk, _spec_proc, _mel_proc)
    __plot_1d_features(
        _audio_chunk, _spec_proc, _mel_proc, [_pitch_proc_1, _pitch_proc_2]
    )
    __plot_2d_features(_audio_chunk, _spec_proc, _mel_proc, _pitch_proc_3, _lpc_proc)

    _audio_chunk = AudioChunk(file_path=_file_path).load(sr=16000).trim(end=8)
    _librosa_spec_proc = __get_spec_proc(512, 160, 400, 80)
    _librosa_mel_proc = __get_mel_proc(80)

    try:
        _nemo_mel_proc = __get_nemo_mel_proc(_audio_chunk.sr, 512, 160, 400, 80)
        __plot_nemo_features(
            _audio_chunk, _librosa_spec_proc, _librosa_mel_proc, _nemo_mel_proc
        )
    except Exception as e:
        print(f"NeMo is not available: {e}")

    plt.show()
