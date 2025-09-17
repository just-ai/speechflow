import typing as tp
import logging
import numbers

from dataclasses import dataclass

import torch

from speechflow.data_pipeline.core.datasample import (
    DataSample,
    Detachable,
    MovableToDevice,
    Pinnable,
)
from speechflow.data_pipeline.core.exceptions import NoDataSamplesError
from speechflow.utils.pad_utils import pad_1d, pad_2d

__all__ = ["BaseCollate", "BaseCollateOutput"]

LOGGER = logging.getLogger("root")


@dataclass
class BaseCollateOutput(DataSample, MovableToDevice, Pinnable, Detachable):
    def __str__(self):
        if self.file_path:
            return "|".join([x for x in self.file_path])
        else:
            return "|".join([x for x in self.label])

    def to_relative_lengths(self) -> "BaseCollateOutput":
        for k in self.keys():
            if k.endswith("_lengths"):
                field = None
                field_lengths = getattr(self, k)
                if field_lengths is None:
                    continue

                t_name = k.replace("_lengths", "", 1)
                for t in self.keys():
                    if t == t_name:
                        field = getattr(self, t)
                        break
                else:
                    for t in self.keys():
                        if t.startswith(t_name):
                            field = getattr(self, t)
                            if isinstance(field, torch.Tensor):
                                break

                if (
                    field is not None
                    and field_lengths is not None
                    and field.ndim >= 2
                    and field_lengths.max() > 1
                ):
                    setattr(self, k, field_lengths / field.shape[1])

        return self


class BaseCollate:
    def __init__(
        self,
        pad: tp.Optional[tp.Dict[str, float]] = None,
        multiple: tp.Optional[tp.Dict[str, float]] = None,
        additional_fields: tp.Optional[tp.Sequence[str]] = None,
        relative_lengths: bool = False,
    ):
        self.pad_values = pad if pad else {}
        self.multiple_values = multiple if multiple else {}
        self.additional_fields = additional_fields if additional_fields else []
        self.relative_lengths = relative_lengths

    def collate(self, batch: tp.List[DataSample]) -> BaseCollateOutput:
        if len(batch) == 0:
            raise NoDataSamplesError("No DataSamples in batch.")

        for ds in batch:
            ds.deserialize(full=True).to_tensor()

        collated = BaseCollateOutput()

        collated.file_path = [sample.file_path.as_posix() for sample in batch]

        if batch[0].label is not None:
            label = [sample.label for sample in batch]
            if isinstance(label[0], numbers.Number):
                label = torch.Tensor(label)
        else:
            label = None

        collated.label = label

        additional_fields = {}
        if batch[0].additional_fields:
            for key in batch[0].additional_fields.keys():
                fields = [sample.additional_fields[key] for sample in batch]
                if fields[0].ndim == 1:
                    fields, _ = pad_1d(fields)
                else:
                    fields, _ = pad_2d(fields, fields[0].shape[1])
                additional_fields[key] = fields

        for field in self.additional_fields:
            additional_fields.update({field: [getattr(b, field) for b in batch]})

        collated.additional_fields = additional_fields
        collated.transform_params = batch[0].transform_params
        return collated

    def __call__(self, batch: tp.List[DataSample]) -> BaseCollateOutput:
        collated = self.collate(batch)
        if self.relative_lengths:
            return collated.to_relative_lengths()
        else:
            return collated
