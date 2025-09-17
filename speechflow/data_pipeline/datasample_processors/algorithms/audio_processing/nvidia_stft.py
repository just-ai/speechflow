import numpy as np
import torch
import torch.nn.functional as F

from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from torch.autograd import Variable

__all__ = ["STFT", "Linear2Mel"]


def window_sumsquare(
    window,
    n_frames,
    hop_length=200,
    win_length=800,
    n_fft=800,
    dtype=np.float32,
    norm=None,
):
    """# from librosa 0.6 Compute the sum-square envelope of a window function at a given
    hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : npt.NDArray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function

    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = normalize(win_sq, norm=norm) ** 2
    win_sq = pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


class _STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch- stft."""

    def __init__(self, filter_length=800, hop_length=200, win_length=800, window="hann"):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int(self.filter_length / 2 + 1)
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

        self.num_samples = None
        self.magnitude = None
        self.phase = None

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
        )

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        # magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        # phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        real_part = real_part.permute([1, 2, 0])
        imag_part = imag_part.permute([1, 2, 0])
        return torch.cat([real_part, imag_part], dim=2)

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
        )
        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum.to(magnitude.device)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


def _dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


class STFT(torch.nn.Module):
    """Computes mel-spectrogram from a batch of waves."""

    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.stft_fn = _STFT(filter_length, hop_length, win_length)

    def __call__(self, y):
        """Computes mel-spectrogram from a batch of waves.

        params:
        y: Variable(torchaudio.FloatTensor) with shape (B, T) in range [-1, 1]
        returns:
        mel_output: torchaudio.FloatTensor of shape (B, n_mel_channels, T)

        """
        y = y.unsqueeze(0)
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1
        return self.stft_fn.transform(y)


class Linear2Mel(torch.nn.Module):
    """Computes mel-spectrogram from a batch of waves."""

    def __init__(
        self,
        filter_length: int,
        n_mel_channels: int,
        sampling_rate: int,
        mel_fmin: float,
        mel_fmax: float,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate

        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    @staticmethod
    def spectral_normalize(magnitudes):
        output = _dynamic_range_compression(magnitudes)
        return output

    def __call__(self, magnitudes):
        """Computes mel-spectrogram from a batch of waves.

        params:
        y: Variable(torchaudio.FloatTensor) with shape (B, T) in range [-1, 1]
        returns:
        mel_output: torchaudio.FloatTensor of shape (B, n_mel_channels, T)

        """
        mel_output = torch.matmul(self.mel_basis, magnitudes.unsqueeze(0))
        return mel_output.squeeze(0)
