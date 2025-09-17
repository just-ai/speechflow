import torch

from speechflow.training.base_model import BaseTorchModel, BaseTorchModelParams

__all__ = ["WaveformGenerator"]


class WaveformGenerator(BaseTorchModel):
    """Base class for waveform generator modules."""

    def __init__(self, params: BaseTorchModelParams):
        super().__init__(params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")
