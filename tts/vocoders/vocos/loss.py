from typing import List, Tuple, Union

import torch
import torchaudio

from torch import nn
from torch.nn import functional as F

from tts.vocoders.vocos.utils.tensor_utils import safe_log


class SpectrogramTransform(nn.Module):
    """Simple transform for computing STFT from given waveform.

    Args:
        fft_size (int): fft_size for stft
        hop_size (int): hop_size for stft
        win_size (int): win_size for stft

    """

    def __init__(self, fft_size: int = 1024, hop_size: int = 256, win_size: int = 800):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.window = nn.Parameter(torch.hann_window(win_size), requires_grad=False)

    def transform(self, waveform: torch.Tensor, global_step: int) -> torch.Tensor:
        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(1)

        # Always compute spectrogram in FP32:
        input_tensor_dtype = waveform.type()
        if input_tensor_dtype != torch.float32:
            waveform = waveform.type(torch.float32)
            self.window.data = self.window.data.type(torch.float32)

        x_stft = torch.stft(
            waveform,
            self.fft_size,
            self.hop_size,
            self.win_size,
            self.window.data,
            return_complex=False,
        )
        real = x_stft[..., 0]
        imag = x_stft[..., 1]
        outputs = torch.clamp(real**2 + imag**2, min=1e-7).transpose(2, 1)
        outputs = torch.sqrt(outputs)

        if input_tensor_dtype != torch.float32:
            outputs = outputs.type(input_tensor_dtype)

        return outputs.unsqueeze(1)

    def forward(self, waveform: torch.Tensor, global_step: int) -> torch.Tensor:
        return self.transform(waveform, global_step)


class MelSpecReconstructionLoss(nn.Module):
    """L1 distance between the mel-scaled magnitude spectrograms of the ground truth
    sample and the generated sample."""

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )

    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        mel_hat = safe_log(self.mel_spec(y_hat))
        mel = safe_log(self.mel_spec(y))

        loss = F.l1_loss(mel, mel_hat)

        return loss


class GeneratorLoss(nn.Module):
    """Generator Loss module.

    Calculates the loss for the generator based on discriminator outputs.

    """

    def forward(
        self, disc_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            disc_outputs (List[Tensor]): List of discriminator outputs.

        Returns:
            Tuple[Tensor, List[Tensor]]: Tuple containing the total loss and a list of loss values from
                                         the sub-discriminators
        """
        loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l

        return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    """Discriminator Loss module.

    Calculates the loss for the discriminator based on real and generated outputs.

    """

    def forward(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_generated_outputs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            disc_real_outputs (List[Tensor]): List of discriminator outputs for real samples.
            disc_generated_outputs (List[Tensor]): List of discriminator outputs for generated samples.

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]: A tuple containing the total loss, a list of loss values from
                                                       the sub-discriminators for real outputs, and a list of
                                                       loss values for generated outputs.
        """
        loss = torch.zeros(
            1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype
        )
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss)
            g_losses.append(g_loss)

        return loss, r_losses, g_losses


class FeatureMatchingLoss(nn.Module):
    """Feature Matching Loss module.

    Calculates the feature matching loss between feature maps of the sub-
    discriminators.

    """

    def forward(
        self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Args:
            fmap_r (List[List[Tensor]]): List of feature maps from real samples.
            fmap_g (List[List[Tensor]]): List of feature maps from generated samples.

        Returns:
            Tensor: The calculated feature matching loss.
        """
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: Union[int, Tuple[int, ...]],
        hop_sizes: Union[int, Tuple[int, ...]],
        win_sizes: Union[int, Tuple[int, ...]],
    ):
        super().__init__()

        self.transforms = nn.ModuleList()
        for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):  # type: ignore
            transform = SpectrogramTransform(
                fft_size=fft_size, hop_size=hop_size, win_size=win_size
            )
            self.transforms.append(transform)

    @staticmethod
    def _log_stft_magnitude(predicts_mag, targets_mag):
        log_predicts_mag = predicts_mag
        log_targets_mag = torch.log(targets_mag)
        outputs = F.l1_loss(log_predicts_mag, log_targets_mag, reduction="none")
        outputs = outputs.mean()
        return outputs

    @staticmethod
    def _spectral_convergence(predicts_mag, targets_mag):
        x = torch.norm((targets_mag - predicts_mag), p="fro")
        y = torch.norm(targets_mag, p="fro")
        return x / y

    def compute_loss_value(self, y_hat, y) -> torch.Tensor:
        spectral_convergence = []
        log_magnitude_loss = []

        for transform in self.transforms:
            fake_spectrogram = transform(y_hat, None)
            real_spectrogram = transform(y, None)
            spectral_convergence.append(
                self._spectral_convergence(fake_spectrogram, real_spectrogram)
            )
            log_magnitude_loss.append(
                self._log_stft_magnitude(fake_spectrogram, real_spectrogram)
            )

        spectral_convergence = sum(spectral_convergence) / len(spectral_convergence)
        log_magnitude_loss = sum(log_magnitude_loss) / len(log_magnitude_loss)

        return spectral_convergence + log_magnitude_loss

    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        return self.compute_loss_value(y_hat, y)
