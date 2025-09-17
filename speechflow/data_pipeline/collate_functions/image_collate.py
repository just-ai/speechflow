import typing as tp

from dataclasses import dataclass

import torch

from speechflow.data_pipeline.core.base_collate_fn import BaseCollate, BaseCollateOutput
from speechflow.data_pipeline.datasample_processors.data_types import ImageDataSample

__all__ = ["ImageCollate", "ImageCollateOutput"]


@dataclass
class ImageCollateOutput(BaseCollateOutput, ImageDataSample):
    pass


class ImageCollate(BaseCollate):
    def collate(self, batch: tp.List[ImageDataSample]) -> ImageCollateOutput:  # type: ignore
        collated = super().collate(batch)  # type: ignore
        collated = ImageCollateOutput(**collated.to_dict())  # type: ignore

        collated.image = torch.cat([ds.image for ds in batch])
        collated.label = torch.cat([torch.LongTensor([ds.label]) for ds in batch])
        return collated
