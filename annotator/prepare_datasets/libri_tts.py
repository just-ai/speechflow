import argparse

from pathlib import Path

from speechflow.io import AudioChunk


def _flac_to_wav(path: Path):
    audio_chunk = AudioChunk(path).load()
    wav_path = path.with_suffix(".wav")
    audio_chunk.save(wav_path)
    path.unlink()
    return True


if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(description="Prepare LibriTTS dataset")
    arguments_parser.add_argument(
        "-d", "--data_root", help="path to dataset", type=Path, required=True
    )
    args = arguments_parser.parse_args()

    data_root = args.data_root

    for file in data_root.rglob("*.normalized.txt"):
        text = file.read_text(encoding="utf-8")
        new_path = file.as_posix().replace(".normalized.txt", ".txt")
        Path(new_path).write_text(text, encoding="utf-8")

    print("DONE!")
