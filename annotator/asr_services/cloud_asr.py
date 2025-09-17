import json
import time
import typing as tp
import logging

from datetime import datetime
from pathlib import Path

import numpy as np

from tqdm import tqdm

from speechflow.data_pipeline.core import BaseDSParser
from speechflow.data_pipeline.core.parser_types import Metadata
from speechflow.io import AudioChunk

__all__ = ["CloudASR", "ASRException", "ASRRequestLimitException"]

LOGGER = logging.getLogger("root")


class ASRException(Exception):
    pass


class ASRRequestLimitException(ASRException):
    pass


class CloudASR(BaseDSParser):
    """Generate transcription for audio files."""

    def __init__(
        self,
        output_file_ext: str = ".json",
        sample_rate: int = 16000,
        raise_on_converter_exc: bool = False,
        raise_on_asr_limit_exc: bool = False,
        release_func: tp.Optional[tp.Callable] = None,
    ):
        super().__init__(
            raise_on_converter_exc=raise_on_converter_exc, release_func=release_func
        )
        self._output_file_ext = output_file_ext
        self._sample_rate = sample_rate
        self._raise_on_asr_limit_exc = raise_on_asr_limit_exc

    def reader(
        self, file_path: Path, label: tp.Optional[str] = None
    ) -> tp.List[Metadata]:
        metadata = {"audio_path": file_path}
        return [metadata]

    def converter(self, metadata: Metadata) -> tp.List[Metadata]:
        audio_path = Path(metadata["audio_path"])
        audio_chunk = AudioChunk(audio_path)
        audio_chunk = audio_chunk.load(sr=self._sample_rate).as_type(np.int16)

        try:
            metadata.update({"waveform": audio_chunk.waveform, "sr": audio_chunk.sr})
            metadata = self._transcription(metadata)
        except ASRException as e:
            LOGGER.error(f"{metadata['audio_path']}: {e}")
            raise e

        transcription: tp.Dict[str, tp.Any] = metadata["transcription"]
        transcription.update(
            {
                "api": self.__class__.__name__,
                "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            }
        )

        output_file_path = audio_path.with_suffix(self._output_file_ext)
        output_file_path.write_text(
            json.dumps(transcription, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

        txt_file_path = audio_path.with_suffix(".txt")
        if not txt_file_path.exists():
            txt_file_path.write_text(metadata["transcription"]["text"], encoding="utf-8")

        return [transcription]

    @classmethod
    def json_to_txt(cls, json_path: tp.Union[str, Path]):
        json_path = Path(json_path)
        transcription = json.loads(json_path.read_text(encoding="utf-8"))
        text = cls._to_text(transcription)
        json_path.with_suffix(".txt").write_text(text, encoding="utf-8")

    def _transcription(self, metadata: Metadata) -> Metadata:
        raise NotImplementedError

    @staticmethod
    def _to_text(metadata: Metadata) -> str:
        raise NotImplementedError

    @staticmethod
    def _sleep(sleep: int = 600, step: int = 1):
        for _ in tqdm(range(sleep // step)):
            time.sleep(step)
