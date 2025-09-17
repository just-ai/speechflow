import typing as tp
import logging

from pathlib import Path

import numpy as np

from speechflow.data_pipeline.core import BaseDSParser
from speechflow.data_pipeline.core.dataset import Dataset
from speechflow.data_pipeline.core.parser_types import Metadata
from speechflow.io import AudioChunk

__all__ = ["EasyDSParser"]

LOGGER = logging.getLogger("root")


class EasyDSParser(BaseDSParser):
    """Easy database parser."""

    def __init__(
        self,
        func: tp.Callable,
        memory_bound: bool = False,
        chunk_size: tp.Optional[int] = None,
        progress_bar: bool = True,
    ):
        super().__init__(
            memory_bound=memory_bound, chunk_size=chunk_size, progress_bar=progress_bar
        )
        self._func = func

    def reader(
        self, file_path: Path, label: tp.Optional[str] = None
    ) -> tp.List[Metadata]:
        metadata = {"file_path": file_path, "label": label}
        return [metadata]

    def converter(self, metadata: Metadata) -> tp.List[tp.Any]:
        return [self._func(metadata["file_path"])]

    def run_from_path_list(
        self, path_list: tp.Union[tp.List[str], tp.List[Path]], n_processes: int = 1
    ) -> Dataset:
        return self.read_datasamples(path_list, n_processes=n_processes)

    def run_from_object_list(
        self, items: tp.List[tp.Any], n_processes: int = 1
    ) -> Dataset:
        if isinstance(items[0], (str, Path)):
            raise ValueError(
                "'items' must not contain file paths. "
                "For paths processing use the 'run_from_path_list' method."
            )

        return self.read_datasamples(items, n_processes=n_processes)

    def run_in_dir(
        self,
        data_root: tp.Union[str, Path],
        file_extension: str = ".*",
        with_subfolders: bool = True,
        file_filter: tp.Optional[tp.Callable] = None,
        n_processes: int = 1,
    ) -> Dataset:
        from speechflow.io import construct_file_list

        file_list = construct_file_list(
            data_root=data_root,
            ext=file_extension,
            with_subfolders=with_subfolders,
            path_filter=file_filter,
        )
        return self.read_datasamples(
            file_list, data_root=data_root, n_processes=n_processes
        )


def _loader(wav_path: Path):
    LOGGER.info(wav_path)
    return AudioChunk(wav_path).duration


class Resynthesize:
    voco: tp.Optional[tp.List] = None

    @classmethod
    def wave_resynt(self, wav_path: Path):
        if self.voco is None:
            self.voco = []
            LOGGER.info("init vocoder interface")

        LOGGER.info(wav_path)
        return AudioChunk(wav_path).duration


if __name__ == "__main__":
    from speechflow.logging.server import LoggingServer
    from speechflow.utils.fs import get_root_dir

    with LoggingServer.ctx("log_1.txt"):
        parser = EasyDSParser(func=_loader)
        data = parser.run_in_dir(
            data_root=get_root_dir() / "speechflow/data",
            file_extension=".wav",
            n_processes=1,
        )
        LOGGER.info(f"Total audio duration in hours: {np.round(sum(data) / 3600, 3)}")

    with LoggingServer.ctx("log_2.txt"):
        parser = EasyDSParser(func=Resynthesize.wave_resynt)
        data = parser.run_in_dir(
            data_root=get_root_dir() / "speechflow/data",
            file_extension=".wav",
            n_processes=1,
        )
        LOGGER.info(f"Total audio duration in hours: {np.round(sum(data) / 3600, 3)}")
