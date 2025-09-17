import typing as tp

from dataclasses import dataclass

from speechflow.data_pipeline.core.base_collate_fn import BaseCollate, BaseCollateOutput
from speechflow.data_pipeline.datasample_processors.data_types import (
    ProsodyPredictionDataSample,
)
from speechflow.utils.pad_utils import pad_1d

__all__ = ["ProsodyPredictionCollate", "ProsodyPredictionCollateOutput"]


@dataclass
class ProsodyPredictionCollateOutput(BaseCollateOutput, ProsodyPredictionDataSample):
    pass


class ProsodyPredictionCollate(BaseCollate):
    def collate(  # type: ignore
        self, batch: tp.List[ProsodyPredictionDataSample]
    ) -> ProsodyPredictionCollateOutput:
        collated = super().collate(batch)  # type: ignore
        collated = ProsodyPredictionCollateOutput(**collated.to_dict())  # type: ignore

        pad_symb_id = batch[0].pad_id

        binary = []
        category = []
        attention_mask = []
        input_ids = []

        for sample in batch:
            binary.append(sample.binary)
            category.append(sample.category)
            attention_mask.append(sample.attention_mask)
            input_ids.append(sample.input_ids)

        if batch[0].binary is not None:
            binary, _ = pad_1d(binary, pad_val=-100)
        else:
            binary = None

        if batch[0].category is not None:
            category, _ = pad_1d(category, pad_val=-100)
        else:
            category = None

        input_ids, _ = pad_1d(input_ids, pad_val=pad_symb_id)
        attention_mask, _ = pad_1d(attention_mask, pad_val=0)

        collated.binary = binary
        collated.category = category
        collated.input_ids = input_ids
        collated.attention_mask = attention_mask
        return collated
