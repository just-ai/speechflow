from typing import List, Optional, Tuple

from nlp.prosody_prediction.data_types import (
    ProsodyPredictionInput,
    ProsodyPredictionTarget,
)
from speechflow.data_pipeline.collate_functions.prosody_collate import (
    ProsodyPredictionCollateOutput,
)
from speechflow.data_pipeline.core import BaseBatchProcessor, Batch, DataSample

__all__ = ["ProsodyPredictionProcessor"]


class ProsodyPredictionProcessor(BaseBatchProcessor):
    def __init__(self):
        super().__init__()

    def __call__(
        self, batch: Batch, batch_idx: int = 0, global_step: int = 0
    ) -> Tuple[
        ProsodyPredictionInput, ProsodyPredictionTarget, Optional[List[DataSample]]
    ]:
        collated: ProsodyPredictionCollateOutput = batch.collated_samples  # type: ignore
        _input: ProsodyPredictionInput = ProsodyPredictionInput(
            input_ids=collated.input_ids,
            attention_mask=collated.attention_mask,
            binary=collated.binary,
            category=collated.category,
        )
        _target: ProsodyPredictionTarget = ProsodyPredictionTarget(
            binary=collated.binary,
            category=collated.category,
        )

        return _input.to(self.device), _target.to(self.device), batch.data_samples
