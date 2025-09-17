import torch

from speechflow.training.base_model import BaseTorchModel, BaseTorchModelParams
from tts.vocoders.data_types import VocoderForwardInput

__all__ = ["FeatureExtractor"]


class FeatureExtractor(BaseTorchModel):
    """Base class for feature extractors."""

    def __init__(self, params: BaseTorchModelParams):
        super().__init__(params)

    def forward(self, inputs: VocoderForwardInput, **kwargs) -> torch.Tensor:
        """Extract features from the given audio.

        Args:
            inputs (VocoderForwardInput): Input audio features.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.

        """
        raise NotImplementedError("Subclasses must implement the forward method.")
