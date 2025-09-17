import argparse

from pathlib import Path

if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(description="Prepare LJSpeech dataset")
    arguments_parser.add_argument(
        "-d", "--data_root", help="path to dataset", type=Path, required=True
    )
    args = arguments_parser.parse_args()

    metadata_path = args.data_root / "metadata.csv"
    metadata = metadata_path.read_text(encoding="utf-8").split("\n")

    output_path = args.data_root / "wavs"
    all_files = []
    for line in metadata:
        if line.strip():
            wav_name, text_orig, text_norm = line.split("|")

            file_path = output_path / f"{wav_name}.wav"
            if file_path.exists():
                file_path.with_suffix(".txt").write_text(text_norm, encoding="utf-8")
                all_files.append(file_path)

    print(f"DONE! Prepare {len(all_files)} files")
