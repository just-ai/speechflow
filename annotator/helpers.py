import json

from itertools import groupby

import torch

from multilingual_text_parser.data_types import Token

from annotator.asr_services.cloud_asr import CloudASR
from annotator.audiobook_spliter import AudiobookSpliter
from speechflow.data_pipeline.datasample_processors.data_types import TTSDataSample
from speechflow.data_pipeline.datasample_processors.tts_processors import (
    add_pauses_from_text,
)
from speechflow.data_pipeline.datasample_processors.tts_text_processors import (
    TTSTextProcessor,
)
from speechflow.io import check_path, tp_PATH
from speechflow.utils.fs import get_root_dir


@check_path
def split_audio_by_sentences(
    text: str, audio_path: tp_PATH, asr: CloudASR, spliter: AudiobookSpliter
):
    transc_path = audio_path.with_suffix(".whisper")

    if not transc_path.exists():
        data = asr.read_datasamples([audio_path])
        transcription = data.item(0)
    else:
        json_dump = transc_path.read_text(encoding="utf-8")
        transcription = json.loads(json_dump)

    metadata = {
        "audio_path": audio_path,
        "text": text,
        "transcription": transcription,
    }
    metadata = spliter.do_preprocessing([metadata], spliter.preproc_fn)
    metadata = spliter.to_datasample(metadata)

    results = []
    for sega in metadata.item(0)["segmentation"]:
        results.append({"sega": sega, "text": sega.sent.text_orig})

        new_tokens = []
        for token in sega.sent.tokens:
            new_tokens.append(token)
            try:
                if token.asr_pause and float(token.asr_pause) > 0:
                    pause = Token(TTSTextProcessor.sil)
                    pause.meta["duration"] = float(token.asr_pause)
                    new_tokens.append(pause)
            except Exception:
                pass
        results[-1]["text_with_pauses_from_asr"] = new_tokens

        new_tokens = []
        for token in getattr(sega, "transcription", []):
            prev_word_ts = token.meta.get("prev_word_ts")
            next_word_ts = token.meta.get("next_word_ts")

            if prev_word_ts is None and token.meta["ts"][0] > 0:
                pause = Token(TTSTextProcessor.sil)
                pause.meta["duration"] = token.meta["ts"][0]
                new_tokens.append(pause)

            new_tokens.append(token)

            if next_word_ts is not None and next_word_ts[0] > token.meta["ts"][1]:
                pause = Token(TTSTextProcessor.sil)
                pause.meta["duration"] = next_word_ts[0] - token.meta["ts"][1]
                new_tokens.append(pause)

            if next_word_ts is None and token.meta["eos_ts"] > token.meta["ts"][1]:
                pause = Token(TTSTextProcessor.sil)
                pause.meta["duration"] = token.meta["eos_ts"] - token.meta["ts"][1]
                new_tokens.append(pause)

        results[-1]["transcription_with_pauses"] = new_tokens

        tts_ds = TTSDataSample(sent=sega.sent)
        tts_ds = add_pauses_from_text(
            tts_ds,
            num_symbols=2,
            pause_from_punct_map={
                ",": "normal",
                "-": "weak",
                "—": "normal",
                ".": "strong",
            },
        )
        new_tokens = []
        for key, group_items in groupby(tts_ds.sent.tokens, key=lambda x: x.is_pause):
            if key:
                pause = Token(TTSTextProcessor.sil)
                pause.meta["duration"] = 0.05 * sum([x.num_phonemes for x in group_items])
                new_tokens.append(pause)
            else:
                new_tokens += group_items

        results[-1]["text_with_pauses_from_punctuation"] = new_tokens

    return results


if __name__ == "__main__":
    from annotator.asr_services import OpenAIASR

    _lang = "RU"
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    _asr = OpenAIASR(_lang, "medium", _device)
    _spliter = AudiobookSpliter(lang=_lang)

    _audio_path = get_root_dir() / "tests/data/test_audio.wav"
    _text = _asr.read_datasamples([_audio_path]).item(0)["text"]

    _results = split_audio_by_sentences(_text, _audio_path, _asr, _spliter)

    for idx, item in enumerate(_results):
        begin = round(item["sega"].ts_bos, 2)
        end = round(item["sega"].ts_eos, 2)
        print(f"{(idx + 1):03}) [{begin:.2f}, {end:.2f}]: {item['text']}")
