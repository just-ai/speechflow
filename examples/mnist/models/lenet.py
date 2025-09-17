import typing as tp

import torch
import torch.nn as nn

from speechflow.io import Config
from speechflow.training.base_model import BaseTorchModel

from ..data_types import MNISTForwardInput, MNISTForwardOutput
from .params import LeNetParams

__all__ = ["LeNet"]


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels, out_channels=32, kernel_size=3, stride=1
            ),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.Tanh(),
        )

        self.fc1 = nn.Linear(in_features=1152, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def calc(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        logits = self.fc2(torch.tanh(x))
        return logits

    def forward(self, *args):
        return self.calc(*args)


class LeNet(BaseTorchModel):
    params: LeNetParams

    def __init__(self, params: tp.Union[Config, LeNetParams], strict_init: bool = True):
        super().__init__(LeNetParams.create(params, strict_init))
        self._model = LeNet5(self.params.num_classes, self.params.input_channels)

    def forward(self, input: MNISTForwardInput) -> MNISTForwardOutput:
        logits = self._model(input.image)
        return MNISTForwardOutput(logits=logits)
