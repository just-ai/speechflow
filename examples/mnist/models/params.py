from speechflow.training.base_model import BaseTorchModelParams

__all__ = ["LeNetParams", "ResNetParams"]


class LeNetParams(BaseTorchModelParams):
    """LeNet parameters."""

    num_classes: int = 10
    input_channels: int = 3


class ResNetParams(BaseTorchModelParams):
    """ResNet parameters."""

    depth: int = 32
    num_classes: int = 10
    input_channels: int = 3
    use_pretrained: bool = False
