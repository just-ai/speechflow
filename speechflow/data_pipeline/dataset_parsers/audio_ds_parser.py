import io
import typing as tp
import logging

from pathlib import Path

import numpy as np
import pydub

from speechflow.data_pipeline.core import BaseDSParser
from speechflow.data_pipeline.core.parser_types import Metadata, MetadataTransform
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.data_types import (
    SpectrogramDataSample,
)
from speechflow.io import AudioChunk

__all__ = ["AudioDSParser"]

LOGGER = logging.getLogger("root")


class AudioDSParser(BaseDSParser):
    """Simple audio database parser."""

    def __init__(
        self,
        preproc_fn: tp.Optional[tp.Sequence[MetadataTransform]] = None,
        file_ext: tp.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(preproc_fn, {"audio_chunk"})
        self.file_ext = file_ext

    def reader(
        self, file_path: Path, label: tp.Optional[str] = None
    ) -> tp.List[Metadata]:
        if self.file_ext:
            file_path = file_path.with_suffix(".wav")

        audio_chunk = AudioChunk(file_path=file_path)
        metadata = {"file_path": file_path, "label": label, "audio_chunk": audio_chunk}
        return [metadata]

    def converter(self, metadata: Metadata) -> tp.List[SpectrogramDataSample]:
        ds = SpectrogramDataSample(**metadata)
        return [ds]

    @staticmethod
    @PipeRegistry.registry(inputs={"audio_chunk"}, outputs={"audio_chunk"})
    def load(
        metadata: Metadata,
        sample_rate: int = None,  # type: ignore
        dtype=np.float32,
    ) -> tp.List[Metadata]:
        audio_chunk: AudioChunk = metadata["audio_chunk"]
        audio_chunk.load(sr=sample_rate, dtype=dtype, load_entire_file=True)
        return [metadata]

    @staticmethod
    @PipeRegistry.registry(inputs={"waveform"}, outputs={"audio_data"})
    def audio_converter(
        metadata: Metadata, audio_format: str = "ogg"
    ) -> tp.List[Metadata]:
        """Convert audio fragment to compatible format for sending to ASR engines.

        :param audio_format: Output format (default="ogg")
        :return: BytesIO Object with audiofile data

        """
        assert np.issubdtype(
            metadata["waveform"].dtype, np.signedinteger
        ), "Audio data must be integer!"

        in_buff = io.BytesIO(metadata["waveform"])
        audio = pydub.AudioSegment.from_raw(
            in_buff, sample_width=2, channels=1, frame_rate=metadata["sr"]
        )
        codec = "opus" if audio_format == "ogg" else None
        metadata["audio_data"] = audio.export(
            io.BytesIO(), audio_format, codec, parameters=["-strict", "-2"]
        )
        return [metadata]


if __name__ == "__main__":
    from speechflow.data_pipeline.core.components import (
        init_metadata_preprocessing_from_config,
    )
    from speechflow.io import Config
    from speechflow.io.flist import read_file_list
    from speechflow.utils.fs import get_root_dir

    _root = get_root_dir()
    _fpath = list((_root / "examples/simple_datasets/speech/SEGS").rglob("filelist.txt"))[
        0
    ]
    _flist = read_file_list(_fpath, max_num_samples=100)
    assert isinstance(_flist, tp.List)

    _cfg = Config({"pipe": ("load",)})
    _preproc_fn = init_metadata_preprocessing_from_config(AudioDSParser, _cfg)
    _parser = AudioDSParser(_preproc_fn, file_ext=".wav")

    data = _parser.read_datasamples(file_list=_flist, data_root=_root)
    print("sr:", data.item(0).audio_chunk.sr)
