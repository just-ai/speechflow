import json
import argparse

from pathlib import Path

from speechflow.data_pipeline.dataset_parsers import EasyDSParser
from speechflow.io import AudioChunk


def _flac_to_wav(path: Path):
    audio_chunk = AudioChunk(path).load()
    wav_path = path.with_suffix(".wav")
    audio_chunk.save(wav_path)
    path.unlink()
    return True


if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(description="Prepare Hi-Fi TTS dataset")
    arguments_parser.add_argument(
        "-d", "--data_root", help="path to dataset", type=Path, required=True
    )
    args = arguments_parser.parse_args()

    data_root = args.data_root

    for file in data_root.rglob("*.json"):
        for line in file.read_text(encoding="utf-8").split("\n"):
            try:
                meta = json.loads(line)
                text = meta["text_normalized"]
                audio_path = data_root / meta["audio_filepath"]
                if audio_path.exists():
                    audio_path.with_suffix(".txt").write_text(text, encoding="utf-8")
            except Exception as e:
                print(e)

    parser = EasyDSParser(func=_flac_to_wav)
    data = parser.run_in_dir(
        data_root=data_root,
        file_extension=".flac",
        n_processes=0,
    )

    print(f"DONE! Prepare {len(data)} files")
