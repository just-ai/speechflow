from copy import deepcopy as copy
from pathlib import Path

import numpy as np
import torch
import pytest
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from speechflow.data_pipeline.core.base_ds_processor import ComputeBackend
from speechflow.data_pipeline.datasample_processors.audio_processors import (
    SignalProcessor,
)
from speechflow.data_pipeline.datasample_processors.data_types import (
    AudioDataSample,
    SpectrogramDataSample,
)
from speechflow.data_pipeline.datasample_processors.spectrogram_processors import (
    LPCProcessor,
    MelProcessor,
    PitchProcessor,
    SpectralProcessor,
)
from speechflow.io import AudioChunk, Config, Timestamps
from speechflow.utils.profiler import Profiler
from tests.data.test_timestamps import (
    INPUT_TIMESTAMPS,
    NUM_FRAMES,
    TARGET_OUTPUT,
    TEST_HOP_LEN,
)

torch.set_num_threads(1)
THIS_DIR = Path(__file__).parent
FILE_PATH = THIS_DIR / "data" / "test_audio.wav"


def test_to_frames():
    for i, timestamp in enumerate(INPUT_TIMESTAMPS):
        ts = Timestamps(timestamp)
        target_output = Timestamps(TARGET_OUTPUT[i])
        test_result = ts.to_frames(TEST_HOP_LEN, NUM_FRAMES[i])
        assert np.max(np.abs(target_output.intervals - test_result.intervals)) < 2


def test_audio_io():
    audio_chunk = AudioChunk(file_path=FILE_PATH)
    audio_chunk.begin = 1
    audio_chunk.end = 6
    audio_chunk.load(sr=22050, load_entire_file=True)
    wave_trim = audio_chunk.trim(begin=1, end=6).trim(inplace=True)
    audio_chunk.load(sr=16000)
    audio_chunk.resample(sr=22050, inplace=True)
    assert len(audio_chunk.waveform) == len(wave_trim.waveform)


@pytest.mark.parametrize("bits, precision", [(9, 1e-2), (16, 1e-5), (24, 1e-6)])
@pytest.mark.parametrize("quantize, split", [(False, False), (True, False), (True, True)])
def test_mu_law(bits: int, precision: float, quantize: bool, split: bool):
    pipe = ("mu_law_encode", "mu_law_decode")
    pipe_cfg = Config(
        {
            "mu_law_encode": {"bits": bits, "quantize": quantize, "split": split},
        }
    )
    signal_proc = SignalProcessor(pipe, pipe_cfg)

    audio_chunk = AudioChunk(file_path=FILE_PATH)
    audio_chunk.load()

    ds = AudioDataSample(audio_chunk=audio_chunk.copy())
    ds = signal_proc.process(ds)

    assert abs(ds.audio_chunk.waveform.max() - audio_chunk.waveform.max()) < precision


def test_spectrogram(n: int = 1, visualize: bool = False):
    pipe = ("magnitude", "energy")
    pipe_cfg = Config({"magnitude": {"n_fft": 1024, "hop_len": 256, "win_len": 1024}})
    backends = ["librosa", "torchaudio", "nvidia"]

    sp_proc = {}
    for backend in backends:
        sp_proc[backend] = SpectralProcessor(pipe, pipe_cfg, ComputeBackend[backend])

    audio_chunk = AudioChunk(file_path=FILE_PATH)
    audio_chunk.load(sr=22050)
    audio_chunk.trim(end=6, inplace=True)

    ds = SpectrogramDataSample(audio_chunk=audio_chunk)

    spec = {}
    for backend in backends:
        spec[backend] = copy(ds)
        with Profiler(f"stft {backend}"):
            for _ in range(n):
                spec[backend] = sp_proc[backend].process(spec[backend])

    lds, tds = spec["librosa"], spec["torchaudio"]
    assert abs(np.sum(lds.energy) - np.sum(tds.energy)) < 1e-2

    lds, nds = spec["librosa"], spec["nvidia"]
    assert abs(np.sum(lds.energy) - np.sum(nds.energy)) < 1e-2

    pipe = ("linear_to_mel", "amp_to_db")
    pipe_cfg = Config({"linear_to_mel": {"n_mels": 80, "f_max": 8000}})

    mel_proc = {}
    for backend in backends:
        mel_proc[backend] = MelProcessor(pipe, pipe_cfg, ComputeBackend[backend])

    for backend in backends:
        with Profiler(f"mel {backend}"):
            for _ in range(n):
                spec[backend] = mel_proc[backend].process(spec[backend])

    # lds, tds = spec["librosa"], spec["torchaudio"]
    # assert abs(np.sum(lds.mel) - torch.sum(tds.mel)) < 1e-2

    for method in ["pyworld", "torchcrepe"]:
        pitch_proc = PitchProcessor(method=method, torchcrepe_model="tiny")
        lds = pitch_proc.process(lds)
        tds = pitch_proc.process(tds)
        with Profiler(f"pitch {method}"):
            for _ in range(n):
                lds = pitch_proc.process(lds)

    if visualize:
        fig, ax = plt.subplots(len(backends), 1, dpi=160, facecolor="w")  # type: ignore

        for idx, backend in enumerate(backends):
            ds = spec[backend]
            im = ax[idx].imshow(np.flipud(ds.mel.T))
            divider = make_axes_locatable(ax[idx])
            cax = divider.append_axes("right", size=0.25, pad=0.05)
            ax[idx].set_title(backend)
            fig.colorbar(im, cax=cax)

        plt.show()


def test_linear_to_mel(visualize: bool = False):
    pipe_cfg = Config(
        {
            "magnitude": {"n_fft": 1024, "hop_len": 256, "win_len": 1024},
            "linear_to_mel": {"n_mels": 80},
        }
    )
    sp_proc = SpectralProcessor(("magnitude",), pipe_cfg)
    mel_proc = MelProcessor(("linear_to_mel",), pipe_cfg)

    audio_chunk = AudioChunk(file_path=FILE_PATH)
    audio_chunk.load(sr=22050)
    audio_chunk.trim(begin=2, end=3, inplace=True)

    ds = SpectrogramDataSample(audio_chunk=audio_chunk)
    ds = sp_proc.process(ds)
    ds = mel_proc.process(ds)

    transform_ds = copy(ds)
    transform_ds = mel_proc.amp_to_db(transform_ds)
    transform_ds = mel_proc.normalize(transform_ds)

    invert_ds = copy(transform_ds)
    invert_ds = mel_proc.denormalize(invert_ds)
    invert_ds = mel_proc.db_to_amp(invert_ds)
    invert_ds = mel_proc.mel_to_linear(invert_ds)

    assert abs(np.sum(ds.mel) - np.sum(invert_ds.mel)) < 1e-2
    assert abs(np.sum(ds.magnitude) - np.sum(invert_ds.magnitude)) < 20

    if visualize:
        fig, ax = plt.subplots(2, 1, dpi=160, facecolor="w")  # type: ignore
        for idx, ds in enumerate([ds, invert_ds]):
            data = np.flip(ds.mel, axis=0)
            im = ax[idx].imshow(data)
            divider = make_axes_locatable(ax[idx])
            cax = divider.append_axes("right", size=0.25, pad=0.05)
            fig.colorbar(im, cax=cax)
        plt.show()

        fig, ax = plt.subplots(2, 1, dpi=160, facecolor="w")  # type: ignore
        for idx, ds in enumerate([ds, invert_ds]):
            data = np.flip(ds.magnitude, axis=0)
            im = ax[idx].imshow(data)
            divider = make_axes_locatable(ax[idx])
            cax = divider.append_axes("right", size=0.25, pad=0.05)
            fig.colorbar(im, cax=cax)
        plt.show()


def test_lpc():
    pipe_cfg = Config(
        {
            "magnitude": {"n_fft": 1024, "hop_len": 256, "win_len": 1024},
            "linear_to_mel": {"n_mels": 80},
            "lpc_from_mel": {"order": 9},
        }
    )

    sp_proc = SpectralProcessor(("magnitude",), pipe_cfg)
    mel_proc = MelProcessor(("linear_to_mel", "amp_to_db"), pipe_cfg)
    lpc_proc = LPCProcessor(("lpc_from_mel", "lpc_decompose"), pipe_cfg)

    audio_chunk = AudioChunk(file_path=FILE_PATH)
    audio_chunk.load(sr=22050)

    ds = SpectrogramDataSample(audio_chunk=audio_chunk)
    ds = sp_proc.process(ds)
    ds = mel_proc.process(ds)
    ds = lpc_proc.process(ds)

    assert ds.lpc_feat is not None


if __name__ == "__main__":
    test_spectrogram(visualize=False)
