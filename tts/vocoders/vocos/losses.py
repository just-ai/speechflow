import typing as tp

from typing import Iterator

import cdpam
import torch
import torchaudio

from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from transformers import AutoModel

from speechflow.data_pipeline.datasample_processors.biometric_processors import (
    VoiceBiometricProcessor,
)
from speechflow.io import tp_PATH
from tts.vocoders.vocos.utils.tensor_utils import safe_log

__all__ = [
    "GeneratorLoss",
    "DiscriminatorLoss",
    "MelSpecReconstructionLoss",
    "FeatureMatchingLoss",
    "MultiResolutionSTFTLoss",
    "SpeakerSimilarityLoss",
    "WavLMLoss",
    "CDPAMLoss",
]


class GeneratorLoss(nn.Module):
    """Generator Loss module.

    Calculates the loss for the generator based on discriminator outputs.

    """

    def forward(
        self, disc_outputs: tp.List[torch.Tensor]
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
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
            val = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(val)
            loss += val

        return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    """Discriminator Loss module.

    Calculates the loss for the discriminator based on real and generated outputs.

    """

    def forward(
        self,
        disc_real_outputs: tp.List[torch.Tensor],
        disc_generated_outputs: tp.List[torch.Tensor],
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor], tp.List[torch.Tensor]]:
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


class FeatureMatchingLoss(nn.Module):
    """Feature Matching Loss module.

    Calculates the feature matching loss between feature maps of the sub-
    discriminators.

    """

    def forward(
        self,
        fmap_r: tp.List[tp.List[torch.Tensor]],
        fmap_g: tp.List[tp.List[torch.Tensor]],
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
        fft_sizes: tp.Union[int, tp.Tuple[int, ...]],
        hop_sizes: tp.Union[int, tp.Tuple[int, ...]],
        win_sizes: tp.Union[int, tp.Tuple[int, ...]],
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


class SpeakerSimilarityLoss(nn.Module):
    def __init__(
        self,
        model_type: tp.Literal["speechbrain", "wespeaker"] = "wespeaker",
        model_name: tp.Optional[tp_PATH] = None,
        input_sr: int = 24000,
        device: str = "cpu",
    ):
        super().__init__()
        self.bio_proc = VoiceBiometricProcessor(model_type, model_name, device=device)
        self.bio_proc.init()
        self.resample = torchaudio.transforms.Resample(
            input_sr, self.bio_proc.target_sample_rate
        )
        self.device = device

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return iter(())

    def set_device(self, device: str):
        self.bio_proc.device = device
        self.bio_proc.init()
        self.resample.to(device)
        self.device = device

    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        """
        y_hat = y_hat.to(self.device)
        y = y.to(self.device)

        y_hat_16khz = self.resample(y_hat)
        y_16khz = self.resample(y)

        sm_loss = self.bio_proc.compute_sm_loss(y_hat_16khz, y_16khz)
        return torch.mean(sm_loss, dim=0)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.bio_proc.model.zero_grad()
        self.resample.zero_grad()


class WavLMLoss(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        input_sr: int = 24000,
        slm_sr: int = 16000,
        device: str = "cpu",
    ):
        super().__init__()
        self.wavlm = AutoModel.from_pretrained(model_name).to(device)
        self.wavlm.feature_extractor._requires_grad = False
        self.resample = torchaudio.transforms.Resample(input_sr, slm_sr).to(device)
        self.device = device

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return iter(())

    def set_device(self, device: str):
        self.wavlm.to(device)
        self.resample.to(device)
        self.device = device

    def forward(self, y_hat, y):
        y_hat = y_hat.to(self.wavlm.device)
        y = y.to(self.wavlm.device)

        with torch.no_grad():
            y_16khz = self.resample(y)
            embeddings = self.wavlm(
                input_values=y_16khz, output_hidden_states=True
            ).hidden_states

        y_hat_16khz = self.resample(y_hat)
        embeddings_hat = self.wavlm(
            input_values=y_hat_16khz, output_hidden_states=True
        ).hidden_states

        floss = 0
        for er, eg in zip(embeddings, embeddings_hat):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.wavlm.zero_grad()
        self.resample.zero_grad()


class CDPAMLoss(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.cdpam = cdpam.CDPAM(dev=device)
        self.device = device

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return iter(())

    def set_device(self, device: str):
        self.cdpam.model.to(device)
        self.device = device

    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        """
        y_hat = y_hat.to(self.device)
        y = y.to(self.device)

        y_hat = (y_hat * 32768.0).unsqueeze(1)
        y = (y * 32768.0).unsqueeze(1)

        with torch.no_grad():
            _, enc_gt, _ = self.cdpam.model.base_encoder.forward(y)
            enc_gt = F.normalize(enc_gt.float(), dim=1)

        _, enc_hat, _ = self.cdpam.model.base_encoder.forward(y_hat)
        enc_hat = F.normalize(enc_hat.float(), dim=1)

        cdpam_loss = self.cdpam.model.model_dist.forward(enc_hat, enc_gt.detach())
        return torch.mean(cdpam_loss, dim=0)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.cdpam.model.zero_grad()
