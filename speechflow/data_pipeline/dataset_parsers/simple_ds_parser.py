import typing as tp
import logging

from pathlib import Path

from speechflow.data_pipeline.core import BaseDSParser, DataSample
from speechflow.data_pipeline.core.parser_types import Metadata, MetadataTransform

__all__ = ["SimpleDSParser"]

LOGGER = logging.getLogger("root")


class SimpleDSParser(BaseDSParser):
    """Simple database parser."""

    def __init__(
        self,
        preproc_fn: tp.Optional[tp.Sequence[MetadataTransform]] = None,
        progress_bar: bool = True,
    ):
        super().__init__(preproc_fn, progress_bar=progress_bar)

    def reader(
        self, file_path: Path, label: tp.Optional[str] = None
    ) -> tp.List[Metadata]:
        metadata = {"file_path": file_path, "label": label}
        return [metadata]

    def converter(self, metadata: Metadata) -> tp.List[DataSample]:
        datasample = DataSample(**metadata)
        return [datasample]


if __name__ == "__main__":
    from speechflow.io.flist import read_file_list
    from speechflow.utils.fs import get_root_dir

    _root = get_root_dir()
    _fpath = list((_root / "examples/simple_datasets/speech/SEGS").rglob("filelist.txt"))[
        0
    ]
    _flist = read_file_list(_fpath, max_num_samples=100)
    assert isinstance(_flist, tp.List)

    _parser = SimpleDSParser()

    _data = _parser.read_datasamples(file_list=_flist, data_root=_root)
    print(_data.item(0))
