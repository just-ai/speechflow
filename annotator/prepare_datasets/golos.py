import json
import argparse

from pathlib import Path

import numpy as np

from pydub import AudioSegment

from speechflow.data_pipeline.dataset_parsers import EasyDSParser
from speechflow.io.audio_io import AudioChunk
from speechflow.utils.fs import find_files_by_folders


def wave_preprocessing(wav_path: Path, target_dbfs: float = -30.0):
    def match_target_amplitude(sound, dBFS):
        change_in_dBFS = dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    audio_chunk = AudioChunk(wav_path)
    assert audio_chunk.sr >= 16000
    audio_chunk.load(dtype=np.int16)

    audio_segment = AudioSegment(
        audio_chunk.data.tobytes(),
        frame_rate=audio_chunk.sr,
        sample_width=audio_chunk.data.dtype.itemsize,
        channels=1,
    )
    normalized = match_target_amplitude(audio_segment, target_dbfs)

    audio_chunk.data = np.array(normalized.get_array_of_samples())
    return audio_chunk


def text_preprocessing(meta: dict, folder: Path):
    wav_path = folder / meta["audio_filepath"]
    text = meta["text"]

    assert wav_path.exists(), FileNotFoundError(f"File {wav_path.as_posix()} not found!")
    text_path = wav_path.with_suffix(".txt")

    return wav_path, text_path, text


def metadata_preprocessing(item: list):
    line, folder = item
    assert line.strip(), RuntimeError("line is empty!")
    meta = json.loads(line)

    wav_path, text_path, text = text_preprocessing(meta, folder)
    audio_chunk = wave_preprocessing(wav_path)

    audio_chunk.save(overwrite=True)
    text_path.write_text(text, encoding="utf-8")

    return f"{wav_path.as_posix()}|{text}\n"


if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(description="Prepare Golos dataset")
    arguments_parser.add_argument(
        "-d", "--data_root", help="path to dataset", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-nproc",
        "--n_processes",
        help="number of workers for data processing",
        type=int,
        default=1,
    )
    args = arguments_parser.parse_args()

    folder_list = find_files_by_folders(
        args.data_root.as_posix(), extensions="manifest.jsonl"
    )
    print(f"Find {sum([len(item) for item in folder_list])} *.jsonl files")

    db_parser = EasyDSParser(func=metadata_preprocessing, chunk_size=100)

    all_files = []
    total_skip_files = 0
    for file_list in folder_list:
        assert len(file_list) == 1
        meta_path = Path(file_list[0])
        folder_path = Path(file_list[0]).parent
        lines = meta_path.read_text(encoding="utf-8").split("\n")
        items = [[line, folder_path] for line in lines]

        data = db_parser.run_from_object_list(items, n_processes=args.n_processes)
        all_files += data
        total_skip_files += len(lines) - len(data)

    (args.data_root / "all_meta.txt").write_text("".join(all_files), encoding="utf-8")

    print(f"DONE! Prepare {len(all_files)} files, {total_skip_files} files skipped")
