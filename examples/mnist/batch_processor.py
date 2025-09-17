import typing as tp

from examples.mnist.data_types import MNISTForwardInput, MNISTTarget
from speechflow.data_pipeline.collate_functions.image_collate import ImageCollateOutput
from speechflow.data_pipeline.core import BaseBatchProcessor, Batch, DataSample

__all__ = ["MNISTBatchProcessor"]


class MNISTBatchProcessor(BaseBatchProcessor):
    def __call__(
        self, batch: Batch, batch_idx: int = 0, global_step: int = 0
    ) -> tp.Tuple[MNISTForwardInput, MNISTTarget, tp.Optional[tp.List[DataSample]]]:
        collated: ImageCollateOutput = batch.collated_samples  # type: ignore

        _input: MNISTForwardInput = MNISTForwardInput(image=collated.image)
        _target: MNISTTarget = MNISTTarget(label=collated.label)

        return _input.to(self.device), _target.to(self.device), batch.data_samples
