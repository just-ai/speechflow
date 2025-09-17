import typing as tp

import torch

from torch import nn
from torchaudio.functional.functional import _hz_to_mel, _mel_to_hz

from speechflow.training.base_model import BaseTorchModelParams
from tts.vocoders.vocos.modules.heads.base import WaveformGenerator
from tts.vocoders.vocos.utils.spectral_ops import IMDCT
from tts.vocoders.vocos.utils.tensor_utils import symexp

__all__ = [
    "IMDCTSymExpHead",
    "IMDCTSymExpHeadParams",
    "IMDCTCosHead",
    "IMDCTCosHeadParams",
]


class IMDCTHeadParams(BaseTorchModelParams):
    input_dim: int
    mdct_frame_len: int
    padding: tp.Literal["center", "same"] = "same"
    clip_audio: bool = False
    sample_rate: tp.Optional[int] = None


class IMDCTSymExpHeadParams(IMDCTHeadParams):
    pass


class IMDCTSymExpHead(WaveformGenerator):
    params: IMDCTSymExpHeadParams
    """IMDCT Head module for predicting MDCT coefficients with symmetric exponential
    function.

    Args:
        dim (int): Hidden dimension of the model.
        mdct_frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
        sample_rate (int, optional): The sample rate of the audio. If provided, the last layer will be initialized
                                     based on perceptual scaling. Defaults to None.
        clip_audio (bool, optional): Whether to clip the audio output within the range of [-1.0, 1.0]. Defaults to False.

    """

    def __init__(self, params: IMDCTSymExpHeadParams):
        super().__init__(params)
        out_dim = params.mdct_frame_len // 2
        self.out = nn.Linear(params.input_dim, out_dim)
        self.imdct = IMDCT(frame_len=params.mdct_frame_len, padding=params.padding)
        self.clip_audio = params.clip_audio

        if params.sample_rate is not None:
            # optionally init the last layer following mel-scale
            m_max = _hz_to_mel(params.sample_rate // 2)
            m_pts = torch.linspace(0, m_max, out_dim)
            f_pts = _mel_to_hz(m_pts)
            scale = 1 - (f_pts / f_pts.max())

            with torch.no_grad():
                self.out.weight.mul_(scale.view(-1, 1))

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass of the IMDCTSymExpHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.

        """
        x = self.out(x)
        x = symexp(x)
        x = torch.clip(
            x, min=-1e2, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        audio = self.imdct(x)
        if self.clip_audio:
            audio = torch.clip(x, min=-1.0, max=1.0)

        return audio, None, {}


class IMDCTCosHeadParams(IMDCTHeadParams):
    pass


class IMDCTCosHead(WaveformGenerator):
    params: IMDCTCosHeadParams
    """
    IMDCT Head module for predicting MDCT coefficients with parametrizing MDCT = exp(m) · cos(p)

    Args:
        dim (int): Hidden dimension of the model.
        mdct_frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
        clip_audio (bool, optional): Whether to clip the audio output within the range of [-1.0, 1.0]. Defaults to False.
    """

    def __init__(self, params: IMDCTCosHeadParams):
        super().__init__(params)
        self.proj = nn.Linear(params.input_dim, params.mdct_frame_len)
        self.imdct = IMDCT(frame_len=params.mdct_frame_len, padding=params.padding)

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass of the IMDCTCosHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.

        """
        x = self.proj(x)
        m, p = x.chunk(2, dim=2)
        m = torch.exp(m).clip(
            max=1e2
        )  # safeguard to prevent excessively large magnitudes
        audio = self.imdct(m * torch.cos(p))
        if self.params.clip_audio:
            audio = torch.clip(x, min=-1.0, max=1.0)

        return audio, None, {}
