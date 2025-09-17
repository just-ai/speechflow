import io
import typing as tp

from copy import deepcopy as copy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pydub
import librosa
import soundfile as sf
import numpy.typing as npt

from scipy import signal

LIBROSA_VERSION = list(map(int, librosa.version.short_version.split(".")))

__all__ = ["AudioFormat", "AudioChunk"]


class AudioFormat(Enum):
    wav = 0
    flac = 1
    opus = 2
    ogg = 4
    mp3 = 5

    @classmethod
    def formats(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def as_extensions(cls):
        return [f".{name}" for name in cls.formats()]


@dataclass
class AudioChunk:
    file_path: tp.Union[str, Path] = None  # type: ignore
    data: npt.NDArray = None  # type: ignore
    sr: int = None  # type: ignore
    begin: float = 0.0
    end: float = None  # type: ignore
    fade_duration: tp.Optional[tp.Tuple[float, float]] = None
    is_trim: bool = False

    def __post_init__(self):
        if self.file_path is not None:
            self.file_path = Path(self.file_path)
            assert (
                self.file_path.exists() or self.data is not None
            ), "audio file not found!"
        else:
            assert self.waveform is not None, "waveform data not set!"
            assert self.sr is not None, "samplerate data not set!"

        if self.sr is None:
            try:
                self.sr = int(librosa.get_samplerate(path=self.file_path.as_posix()))
            except Exception:
                pass

        self._set_end()

    def _set_end(self):
        if self.end is None:
            if self.waveform is None:
                assert self.file_path, "file path not set!"
                try:
                    if LIBROSA_VERSION[1] <= 9:
                        self.end = librosa.get_duration(
                            filename=self.file_path.as_posix()
                        )
                    else:
                        self.end = librosa.get_duration(path=self.file_path.as_posix())
                except Exception:
                    self.load()
            else:
                self.end = len(self.waveform) / self.sr

    @property
    def waveform(self) -> npt.NDArray:
        return self.data

    @waveform.setter
    def waveform(self, waveform: npt.NDArray):
        assert len(waveform) == len(self.waveform)
        self.data = waveform

    @property
    def dtype(self):
        return self.waveform.dtype

    @property
    def empty(self):
        return self.waveform is None

    @property
    def duration(self) -> float:
        if self.end:
            return self.end - self.begin
        else:
            return 0.0

    @property
    def mean_volume(self) -> float:
        s = librosa.magphase(librosa.stft(self.data, window=np.ones, center=False))[0]
        return float(np.mean(librosa.feature.rms(S=s).T, axis=0))

    def load(
        self,
        sr: tp.Optional[int] = None,
        dtype: npt.DTypeLike = np.float32,
        load_entire_file: bool = False,
    ) -> "AudioChunk":
        assert isinstance(self.file_path, Path), "file path not set!"
        assert (
            self.file_path.exists()
        ), f"audio file {self.file_path.as_posix()} not found!"

        if load_entire_file:
            self.data, self.sr = librosa.load(self.file_path, sr=sr)
            self.is_trim = False
        else:
            self.data, self.sr = librosa.load(
                self.file_path, sr=sr, offset=self.begin, duration=self.duration
            )
            if LIBROSA_VERSION[1] <= 9:
                self.is_trim = (
                    librosa.get_duration(filename=self.file_path.as_posix())
                    != self.duration
                )
            else:
                self.is_trim = (
                    librosa.get_duration(path=self.file_path.as_posix()) != self.duration
                )

        self._set_end()

        if self.fade_duration is not None:
            self._apply_fade(
                self.data, self.sr, self.fade_duration[0], self.fade_duration[1]
            )

        return self.as_type(dtype, inplace=True)

    def save(
        self,
        file_path: tp.Optional[tp.Union[str, Path, io.BytesIO]] = None,
        overwrite: bool = False,
    ):
        file_path = file_path if file_path else self.file_path
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if isinstance(file_path, Path):
            format = file_path.suffix[1:]
            if format not in AudioFormat.formats():
                raise ValueError(f"'{format}' audio format is not supported.")

            if not overwrite:
                if isinstance(file_path, (Path, str)):
                    assert not Path(
                        file_path
                    ).exists(), f"file {str(file_path)} is exists!"
        else:
            format = "wav"

        if self.data.ndim == 2 and self.data.shape[0] == 1:
            raise ValueError(
                "Unacceptable data shape, single-channel data must be flatten."
            )

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(file_path, Path):
            file_path.write_bytes(self.to_bytes(format))
        else:
            file_path.write(self.to_bytes(format))

        return self

    def to_bytes(
        self, format: str | AudioFormat = AudioFormat.wav, bitrate: str = "128k"
    ) -> bytes:
        buff = io.BytesIO()
        format = AudioFormat[format]
        if format in [AudioFormat.wav, AudioFormat.flac, AudioFormat.mp3]:
            data = self.as_type(np.float32).data
            sf.write(buff, data, self.sr, format=format.name)
        elif format in [AudioFormat.ogg, AudioFormat.opus]:
            in_buff = io.BytesIO(self.as_type(np.int16).data.tobytes())
            audio = pydub.AudioSegment.from_raw(
                in_buff, sample_width=2, channels=1, frame_rate=self.sr
            )
            buff = audio.export(
                buff,
                format=format.name,
                codec="opus" if format in [AudioFormat.ogg, AudioFormat.opus] else None,
                bitrate=None if format == AudioFormat.ogg else bitrate,
                parameters=["-strict", "-2"],
            )
        else:
            raise NotImplementedError(f"Audio format {format} is not supported")

        return buff.getvalue()

    def as_type(self, dtype, inplace: bool = False) -> "AudioChunk":
        data = self.data

        if self.dtype != dtype:
            if all(
                np.issubdtype(dt, np.signedinteger) for dt in [self.dtype, dtype]
            ) or all(np.issubdtype(dt, np.floating) for dt in [self.dtype, dtype]):
                data = self.data.astype(dtype)
            else:
                scale = np.float32(np.iinfo(np.int16).max)
                if np.issubdtype(self.dtype, np.signedinteger):
                    data = (self.data / scale).astype(dtype)
                else:
                    data = (self.data * scale).astype(dtype)

        if inplace:
            self.data = data
            return self
        else:
            return AudioChunk(
                file_path=self.file_path,
                begin=self.begin,
                end=self.end,
                data=data,
                sr=self.sr,
            )

    def erase(self):
        del self.data
        return self

    def copy(self):
        return copy(self)

    def trim(
        self,
        begin: tp.Optional[float] = None,
        end: tp.Optional[float] = None,
        inplace: bool = False,
    ) -> "AudioChunk":
        if begin is None and end is None:
            if self.is_trim:
                return AudioChunk(
                    begin=0.0, end=self.duration, sr=self.sr, data=self.data.copy()
                )
            else:
                return self if inplace else copy(self)

        begin = int(begin * self.sr) if begin else 0
        end = int(end * self.sr) if end else len(self.data)
        end = min(end, len(self.data))
        assert begin >= 0 and end <= len(self.data)
        assert begin < end

        if inplace:
            assert not self.is_trim, "waveform is already trimmed!"
            self.begin = 0
            self.end = (end - begin) / self.sr
            self.data = self.data[begin:end]
            self.is_trim = True
            return self
        else:
            return AudioChunk(
                data=self.data[begin:end],
                sr=self.sr,
            )

    def pad(
        self,
        left: float = 0,
        right: float = 0,
        mode: str = "constant",
        inplace: bool = False,
    ):
        left = int(left * self.sr)
        right = int(right * self.sr)

        data = self.data
        data = np.pad(data, (left, right), mode=mode, constant_values=0)  # type: ignore

        if inplace:
            self.data = data
            self.end += (left + right) / self.sr
            return self
        else:
            return AudioChunk(data=data, sr=self.sr)

    def multiple(
        self, value: int, mode: str = "constant", odd: bool = False, inplace: bool = False
    ):
        data = self.data
        pad_size = value - data.shape[0] % value
        if pad_size == value:
            pad_size = 0

        data = np.pad(data, (0, pad_size), mode=mode, constant_values=0)  # type: ignore

        if odd:
            data = data[:-1]
            pad_size -= 1

        if inplace:
            self.data = data
            self.end += pad_size / self.sr
            return self
        else:
            return AudioChunk(data=data, sr=self.sr)

    def volume(self, value: float = 1.0, inplace: bool = False):
        sig = self.data
        assert np.issubdtype(
            self.data.dtype, np.floating
        ), "Audio data must be floating-point!"
        if value != 1.0:
            sig = np.clip(sig * value, a_min=-1.0, a_max=1.0)
        if inplace:
            self.data = sig
            return self
        else:
            return AudioChunk(
                file_path=self.file_path,
                begin=self.begin,
                end=self.end,
                sr=self.sr,
                data=sig,
            )

    def resample(
        self, sr: int, inplace: bool = False, fast: bool = False
    ) -> "AudioChunk":
        if self.sr != sr:
            if fast:
                data = librosa.resample(
                    self.data, orig_sr=self.sr, target_sr=sr, res_type="kaiser_fast"
                )
            else:
                data = librosa.resample(self.data, orig_sr=self.sr, target_sr=sr)
        else:
            data = self.data if inplace else self.data.copy()

        if inplace:
            self.data = data
            self.sr = sr
            return self
        else:
            return AudioChunk(
                file_path=self.file_path,
                begin=self.begin,
                end=self.end,
                data=data,
                sr=sr,
            )

    def gsm_preemphasis(self, beta: float = 0.86, inplace: bool = False) -> "AudioChunk":
        """High-pass filter for telephone channel https://edadocs.software.keys
        ight.com/display/ads2009/GSM+Preemphasis."""
        sig = self.data
        assert np.issubdtype(
            self.data.dtype, np.floating
        ), "Audio data must be floating-point!"
        sig = signal.lfilter([1, -beta], [1], sig)
        if inplace:
            self.data = sig
            return self
        else:
            return AudioChunk(
                file_path=self.file_path,
                begin=self.begin,
                end=self.end,
                sr=self.sr,
                data=sig,
            )

    @staticmethod
    def _apply_fade(
        audio, sr: int, left_duration: float = 0.0, right_duration: float = 0.0
    ):
        # convert to audio indices (samples)
        l_fade_len = int(left_duration * sr)
        r_fade_len = int(right_duration * sr)
        r_end = audio.shape[0]
        r_start = r_end - r_fade_len

        # apply the curve
        if l_fade_len > 0:
            l_fade_curve = np.logspace(-1.0, 1.0, l_fade_len) ** 4.0 / 10000.0
            audio[0:l_fade_len] = audio[0:l_fade_len] * l_fade_curve
        if r_fade_len > 0:
            r_fade_curve = np.flip(np.logspace(-1.0, 1.0, r_fade_len) ** 4.0 / 10000.0)
            audio[r_start:r_end] = audio[r_start:r_end] * r_fade_curve

    @staticmethod
    def silence(duration: float, sr: int):
        return AudioChunk(
            data=np.zeros(
                int(duration * sr),
            ),
            sr=sr,
        )

    @staticmethod
    def find_audio(file_path: Path) -> tp.Optional[Path]:
        for ext in AudioFormat.as_extensions():
            audio_path = file_path.with_suffix(ext)
            if audio_path.exists():
                return audio_path


if __name__ == "__main__":
    from speechflow.utils.fs import get_root_dir

    _wav_path = get_root_dir() / "tests/data/test_audio.wav"
    _flac_path = _wav_path.with_suffix(".flac")

    _audio_chunk = AudioChunk(_wav_path).load()
    _audio_chunk.save(_flac_path, overwrite=True)

    _flac_chunk = AudioChunk(_flac_path).load()

    for _format in AudioFormat.formats():
        try:
            print(_format)
            _file_path = f"{_wav_path.stem}.{_format}"
            _audio_chunk.save(_file_path, overwrite=True)
            print(_file_path, AudioChunk(_file_path).load().duration)
        except Exception as e:
            print(e)
