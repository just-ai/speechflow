from typing import Tuple

import torch
import torch.nn as nn


class Denoiser(nn.Module):
    """Removes model bias from audio produced with vocoder."""

    def __init__(
        self,
        bias_audio,
        fft_size: int,
        win_size: int,
        hop_size: int,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.win_size = win_size
        self.hop_size = hop_size
        self.window = torch.hann_window(win_size)

        bias_spec, _ = self.stft_transform(bias_audio)
        self.bias_spec = bias_spec[:, :, 0].unsqueeze(-1)

    @torch.no_grad()
    def stft_transform(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_stft = torch.stft(
            waveform,
            self.fft_size,
            self.hop_size,
            self.win_size,
            self.window.data,
            return_complex=False,
        )
        real_part = x_stft[..., 0]
        imag_part = x_stft[..., 1]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part, real_part)
        return magnitude, phase

    @torch.no_grad()
    def istft_transform(
        self, magnitude: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        return torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.fft_size,
            self.hop_size,
            self.win_size,
            window=self.window.data,
            return_complex=False,
        )

    def forward(
        self, waveform: torch.Tensor, strength: float = 0.1, use_energies: bool = False
    ) -> torch.Tensor:
        # TODO: Denoiser не обрабатывает крайние N семплов справа (строка 67).
        # TODO: Во время батч-инференса нужно делать денойзер над выходными аудио после конкатенации.
        waveform_spec, waveform_angles = self.stft_transform(waveform)

        if use_energies:
            energies = waveform_spec.sum(dim=1)
            energies = torch.log1p(energies)
            weights = 1 - (energies - energies.min()) / (energies.max() - energies.min())
            waveform_spec_denoised = waveform_spec - self.bias_spec * strength * weights
        else:
            waveform_spec_denoised = waveform_spec - self.bias_spec * strength

        waveform_spec_denoised = torch.clamp(waveform_spec_denoised, 0.0)
        waveform_denoised = self.istft_transform(waveform_spec_denoised, waveform_angles)
        waveform[:, : waveform_denoised.shape[1]] = waveform_denoised
        return waveform


if __name__ == "__main__":
    sig = torch.randn((1, 16000))
    d = Denoiser(sig, 800, 800, 200)
    d(sig)
