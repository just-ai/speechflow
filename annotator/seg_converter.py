import typing as tp
import argparse

from pathlib import Path

from speechflow.data_pipeline.core import BaseDSParser
from speechflow.io import AudioSeg, construct_file_list


class SegConverter(BaseDSParser):
    def __init__(self, new_sample_rate: int = 24000):
        super().__init__()
        self._new_sample_rate = new_sample_rate

    def reader(self, file_path: Path, label=None) -> tp.List[dict]:
        metadata = {"file_path": file_path}
        return [metadata]

    def converter(self, metadata: dict):
        file_path = metadata["file_path"]
        sega = AudioSeg.load(file_path, load_audio=True)
        if sega.audio_chunk.sr != self._new_sample_rate:
            sega.audio_chunk.resample(self._new_sample_rate, inplace=True)
            sega.audio_chunk.save(overwrite=True)
        return []


if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-d", "--data_root", help="data root directory", type=Path, required=True
    )
    args = arguments_parser.parse_args()

    flist = construct_file_list(
        args.data_root, ext=".TextGridStage3", with_subfolders=True
    )
    SegConverter().read_datasamples(flist, n_processes=0)
