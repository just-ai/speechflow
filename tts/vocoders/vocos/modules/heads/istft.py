import typing as tp

import torch

from speechflow.training.base_model import BaseTorchModelParams
from tts.vocoders.vocos.modules.heads.base import WaveformGenerator
from tts.vocoders.vocos.utils.spectral_ops import ISTFT

__all__ = ["ISTFTHead", "ISTFTHeadParams"]


class ISTFTHeadParams(BaseTorchModelParams):
    input_dim: int
    n_fft: int
    hop_length: int
    padding: tp.Literal["center", "same"] = "same"


class ISTFTHead(WaveformGenerator):
    params: ISTFTHeadParams

    """ISTFT Head module for predicting STFT complex coefficients.

    Args:
        input_dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".

    """

    def __init__(self, params: ISTFTHeadParams):
        super().__init__(params)
        out_dim = params.n_fft + 2
        self.proj = torch.nn.Linear(params.input_dim, out_dim)
        self.istft = ISTFT(
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            win_length=params.n_fft,
            padding=params.padding,
        )

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.

        """
        x = self.proj(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        s = torch.polar(mag, p)
        audio = self.istft(s)
        return audio, None, {}
