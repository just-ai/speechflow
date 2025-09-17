import typing as tp

import torch

from speechflow.data_pipeline.core.abstract import AbstractBatchProcessor
from speechflow.data_pipeline.core.batch import Batch
from speechflow.data_pipeline.core.exceptions import InvalidDeviceError

__all__ = ["BaseBatchProcessor"]


class BaseBatchProcessor(AbstractBatchProcessor):
    """Basic batch processors."""

    def __init__(self):
        super().__init__()
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def set_device(self, device: tp.Optional[tp.Union[str, int, torch.device]] = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = -1

        if isinstance(device, str):
            self._device = torch.device(device)
        elif isinstance(device, int):
            if device >= 0:
                self._device = torch.device(f"cuda:{device}")
            else:
                self._device = torch.device("cpu")
        elif isinstance(device, torch.device):
            self._device = device
        else:
            raise InvalidDeviceError(f"Invalid device '{device}'")

    @property
    def on_cpu(self) -> bool:
        return self._device.type == "cpu"

    @property
    def on_gpu(self) -> bool:
        return self._device.type != "cpu"

    def __call__(self, batch: Batch, batch_idx: int = 0, global_step: int = 0):
        """
        batch: current batch
        batch_idx: current batch index within this epoch
        global_step: number of optimizer steps
        """
        return NotImplementedError
