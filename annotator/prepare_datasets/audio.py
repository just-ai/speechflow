import json
import typing as tp
import argparse

from pathlib import Path

import whisper  # https://github.com/openai/whisper

from tqdm import tqdm

from speechflow.io import AudioChunk, AudioFormat

LANGUAGES: tp.Dict = whisper.tokenizer.LANGUAGES  # type: ignore


if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(description="Prepare wav files")
    arguments_parser.add_argument(
        "-d", "--data_root", help="path to dataset", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-l", "--lang", type=str, help="language for whisper decoder", required=True
    )
    arguments_parser.add_argument(
        "-gpu", "--gpu_index", help="cuda device number", type=int, default=-1
    )
    args = arguments_parser.parse_args()

    lang = args.lang.lower()[:2]
    device = "cpu" if args.gpu_index < 0 else f"cuda:{args.gpu_index}"
    model = whisper.load_model("large-v2", device=device)
    print(f"Support language: {list(LANGUAGES.keys())}")

    if lang not in LANGUAGES:
        raise ValueError(f"Language {lang} is not support!")

    for file_ext in AudioFormat.as_extensions():
        file_list: tp.List[Path] = list(args.data_root.rglob(f"*{file_ext}"))
        print(f"Find {len(file_list)} {file_ext} files")

        for path in tqdm(file_list, desc="Audio files transcription"):
            if AudioChunk(path).sr < 24000:
                continue

            assert model.is_multilingual
            result = model.transcribe(
                audio=path.as_posix(), language=lang, condition_on_previous_text=False
            )

            if not path.with_suffix(".txt").exists():
                path.with_suffix(".txt").write_text(result["text"], encoding="utf-8")

            timestamps = []
            for segment in result["segments"]:
                phrase = segment["text"]
                if not phrase.strip():
                    continue

                ts_begin = segment["start"]
                ts_end = segment["end"]

                total_phrase_len = len(phrase)
                a = b = ts_begin  # type: ignore
                for word in phrase.split(" "):
                    if not word.strip():
                        continue

                    curr_len = len(word)
                    b += (ts_end - ts_begin) * curr_len / total_phrase_len
                    timestamps.append((word, a, b))
                    a = b

            transcription = {"text": result["text"], "timestamps": timestamps}
            path.with_suffix(".whisper").write_text(
                json.dumps(transcription, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )
