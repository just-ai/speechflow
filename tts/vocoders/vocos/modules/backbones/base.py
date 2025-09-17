import torch

from speechflow.training.base_model import BaseTorchModel, BaseTorchModelParams

__all__ = ["Backbone"]


class Backbone(BaseTorchModel):
    """Base class for the generator's backbones.

    It preserves the same temporal resolution across all layers.

    """

    def __init__(self, params: BaseTorchModelParams):
        super().__init__(params)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")
