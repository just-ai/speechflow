import logging
import warnings
import functools

import numpy as np
import librosa
import numpy.typing as npt
import librosa.filters

from speechflow.logging import trace

__all__ = ["magnitude_function", "detect_noise", "detect_breath"]

LOGGER = logging.getLogger("root")

warnings.filterwarnings("ignore", module="librosa")


def detect_breath(
    wave: npt.NDArray,
    db_cap: float = 24.0,
    function="stft_max_magnitude",
    win: int = 1024,
) -> bool:
    assert wave.dtype == np.float32
    func = magnitude_function(type=function, win=win)
    try:
        return func(wave) > db_cap
    except Exception as e:
        LOGGER.error(trace("detect_breath", e))
        return False


def detect_noise(
    wave: npt.NDArray, function="stft_max_magnitude", win: int = 1024, scale: float = 10.0
) -> float:
    assert wave.dtype == np.float32
    func = magnitude_function(type=function, win=win)
    try:
        return func(wave) / scale
    except Exception as e:
        LOGGER.error(trace("detect_noise", e))
        return -100.0 / scale


def magnitude_function(type="log_magnitude", win: int = 1024, minval: float = -100.0):
    if type in ["log_magnitude", "log_energy", "log_peak"]:
        return functools.partial(globals()[type], minval=minval)
    elif type in ["win_max_magnitude", "stft_max_magnitude"]:
        return functools.partial(globals()[type], win=win, minval=minval)
    return None


def log_magnitude(x, minval: float = -100.0):
    if np.max(np.abs(x)) == 0:
        return minval
    return max(float(librosa.power_to_db(np.mean(x**2))), minval)


def log_energy(x, minval: float = -100.0):
    if np.max(np.abs(x)) == 0:
        return minval
    return max(float(librosa.power_to_db(np.sum(x**2))), minval)


def log_peak(x, minval: float = -100.0):
    if np.max(np.abs(x)) == 0:
        return minval
    return max(float(librosa.power_to_db(np.max(x**2))), minval)


def win_max_magnitude(x, win, minval: float = -100.0):
    if np.mean(np.abs(x)) == 0:
        return minval
    if len(x) < win:
        return log_magnitude(x, minval=minval)
    wins = np.array(
        [
            log_magnitude(x[i * win // 4 : i * win // 4 + win])
            for i in range(len(x) // win * 4)
        ]
    )
    return max(wins.max(), minval)


def stft_max_magnitude(
    x, win, minval: float = -100.0, min_freq: int = 1000, sf: int = 22050
):
    if np.max(np.abs(x)) == 0:
        return minval

    bins = win * min_freq // sf
    spec = librosa.stft(x, n_fft=win, hop_length=win // 4)
    spec = abs(spec)[bins:, :].sum(axis=0)
    s = librosa.power_to_db(spec**2)
    return max(s.max(), minval)
