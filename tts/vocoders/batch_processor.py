import typing as tp
import logging

from speechflow.data_pipeline.core import Batch, DataSample
from tts.acoustic_models.batch_processor import TTSBatchProcessor
from tts.vocoders.data_types import VocoderForwardInput, VocoderTarget

__all__ = [
    "VocoderBatchProcessor",
]

LOGGER = logging.getLogger("root")


class VocoderBatchProcessor(TTSBatchProcessor):
    def __init__(self):
        super().__init__()

    def __call__(
        self, batch: Batch, batch_idx: int = 0, global_step: int = 0
    ) -> (VocoderForwardInput, VocoderTarget, tp.List[DataSample]):
        inputs, targets, data_samples = super().__call__(batch, batch_idx, global_step)
        return inputs, targets, data_samples
