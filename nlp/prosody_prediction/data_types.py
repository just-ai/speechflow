from dataclasses import dataclass

from torch import Tensor

from speechflow.data_pipeline.core.datasample import Detachable, MovableToDevice

__all__ = [
    "ProsodyPredictionTarget",
    "ProsodyPredictionInput",
    "ProsodyPredictionOutput",
]


@dataclass
class ProsodyPredictionTarget(MovableToDevice, Detachable):
    binary: Tensor = None
    category: Tensor = None


@dataclass
class ProsodyPredictionInput(MovableToDevice, Detachable):
    input_ids: Tensor = None
    attention_mask: Tensor = None
    binary: Tensor = None
    category: Tensor = None


@dataclass
class ProsodyPredictionOutput(MovableToDevice, Detachable):
    binary: Tensor = None
    category: Tensor = None
