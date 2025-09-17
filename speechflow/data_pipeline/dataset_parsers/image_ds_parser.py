import typing as tp
import logging

from pathlib import Path

from speechflow.data_pipeline.core import BaseDSParser
from speechflow.data_pipeline.core.base_ds_parser import multi_transform
from speechflow.data_pipeline.core.parser_types import Metadata, MetadataTransform
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.data_types import ImageDataSample

__all__ = ["ImageDSParser"]

LOGGER = logging.getLogger("root")


class ImageDSParser(BaseDSParser):
    """Image database parser."""

    def __init__(self, preproc_fn: tp.Optional[tp.Sequence[MetadataTransform]] = None):
        super().__init__(preproc_fn)

    def reader(
        self, file_path: Path, label: tp.Optional[str] = None
    ) -> tp.List[Metadata]:
        metadata = {"file_path": file_path, "label": label}
        return [metadata]

    def converter(self, metadata: Metadata) -> tp.List[ImageDataSample]:
        datasample = ImageDataSample(**metadata)
        return [datasample]

    @staticmethod
    @PipeRegistry.registry(inputs={"label"}, outputs={"label"})
    def convert_label(metadata: Metadata, label_type: str = "int64") -> tp.List[Metadata]:
        if label_type == "int64":
            metadata["label"] = int(metadata["label"])
        elif label_type == "float":
            metadata["label"] = float(metadata["label"])
        elif label_type == "string":
            pass
        else:
            raise NotImplementedError
        return [metadata]

    @staticmethod
    @multi_transform
    @PipeRegistry.registry(inputs={"label"})
    def class_stat(all_metadata: tp.List[Metadata]) -> tp.List[Metadata]:
        stat: tp.Dict = {}
        for metadata in all_metadata:
            counter = stat.setdefault(metadata["label"], [0])
            counter[0] += 1
        LOGGER.info(f"statistic over classes: {stat}")
        return all_metadata
