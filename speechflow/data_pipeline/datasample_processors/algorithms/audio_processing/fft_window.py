import numpy as np
import torch

from librosa import filters as librosa_stft

__all__ = ["FFTWindow"]


class FFTWindow:
    def __init__(self, win_type: str):
        self.win_type = win_type

    def get_window(self, win_len: int):
        if self.win_type == "half":
            window = self._get_half_window(win_len)
        elif self.win_type == "hann":
            window = self._get_hann_window(win_len)
        else:
            window = librosa_stft.get_window(self.win_type, win_len)
        return window.astype(np.float32)

    @staticmethod
    def _get_half_window(win_len: int):
        windows = [
            FFTWindow._half_window(win_len // 2),
            FFTWindow._half_window(win_len // 2)[::-1],
        ]
        return np.hstack(windows)

    @staticmethod
    def _get_hann_window(win_len: int):
        return torch.hann_window(win_len).numpy()

    @staticmethod
    def _half_window(size: int):
        def _sin(i: int):
            return np.sin(
                0.5
                * np.pi
                * np.sin(0.5 * np.pi * (i + 0.5) / size)
                * np.sin(0.5 * np.pi * (i + 0.5) / size)
            )

        return np.array([_sin(i) for i in range(size)], dtype=np.float32)
