import typing as tp
import logging

from pathlib import Path

import numpy as np
import multilingual_text_parser

from multilingual_text_parser.data_types import Doc, TokenUtils
from multilingual_text_parser.parser import TextParser

from annotator.audiobook_spliter import AudiobookSpliter
from speechflow.data_pipeline.core import BaseDSParser, PipeRegistry
from speechflow.data_pipeline.core.components import (
    init_metadata_preprocessing_from_config,
)
from speechflow.data_pipeline.core.parser_types import Metadata
from speechflow.io import AudioChunk, AudioSeg, Config, Timestamps, construct_file_list
from speechflow.logging import trace

__all__ = ["SimpleSegGenerator"]

LOGGER = logging.getLogger("root")


class SimpleSegGenerator(BaseDSParser):
    def __init__(
        self,
        lang: str,
        device: str = "cpu",
        text_from_label: bool = False,
        raise_on_converter_exc: bool = False,
    ):
        if not TextParser.check_language_support(lang):
            raise ValueError(f"Language {lang} is not support!")

        cfg = Config(
            {
                "pipe": ["text_prepare"],
                "pipe_cfg": {"text_prepare": {"lang": lang, "device": device}},
            }
        )
        preproc_fn = init_metadata_preprocessing_from_config(SimpleSegGenerator, cfg)

        super().__init__(preproc_fn, raise_on_converter_exc=raise_on_converter_exc)

        self._lang = lang
        self._text_from_label = text_from_label

    def reader(self, file_path: Path, label: tp.Optional[str] = None) -> tp.List[dict]:
        audio_path = AudioChunk.find_audio(file_path)
        if audio_path is None:
            return []

        metadata = {
            "file_path": file_path,
            "audio_path": audio_path,
            "text_path": file_path.with_suffix(".txt"),
        }

        if not metadata["audio_path"].exists():
            return []

        if self._text_from_label and label:
            text = label.split("|", 1)[0]
        else:
            text_path = file_path.with_suffix(".txt")
            if text_path.exists():
                text = text_path.read_text(encoding="utf-8")
            else:
                return []

        metadata["text"] = text  # type: ignore
        return [metadata]

    def converter(self, metadata: Metadata) -> tp.List[Metadata]:
        try:
            metadata["segmentation"] = self._get_segmentation(metadata)
        except Exception as e:
            metadata["segmentation"] = []
            LOGGER.error(trace(self, e, message=metadata["file_path"].as_posix()))

        return [metadata]

    @staticmethod
    @PipeRegistry.registry()
    def text_prepare(metadata: Metadata, lang: str, device: str) -> tp.List[Metadata]:
        return AudiobookSpliter.text_prepare(metadata, lang, device)

    def _get_segmentation(self, md: Metadata):
        audio_chunk = AudioChunk(md["audio_path"])
        text: Doc = md["text"]

        assert len(text.sents) == 1
        sent = text.sents[0]

        words = sent.get_words()
        group = TokenUtils.group_tokens_by_word(sent.tokens)
        total_phrase_len = sum([len(t) for t in words])

        ts_begin = 0.0
        ts_end = audio_chunk.duration
        a = b = 0.0
        ts_words = []
        for tokens in group:
            curr_len = sum([len(t) for t in tokens if not t.is_punctuation])
            b += (ts_end - ts_begin) * curr_len / total_phrase_len
            ts_words.append((a, b))
            a = b

        sega = AudioSeg(audio_chunk, sent)
        sega.set_word_timestamps(Timestamps(np.asarray(ts_words)))
        sega.meta.update(
            {
                "lang": self._lang,
                "orig_audio_path": audio_chunk.file_path.as_posix(),
                "orig_audio_chunk": (audio_chunk.begin, audio_chunk.end),
                "orig_audio_samplerate": audio_chunk.sr,
                "text_parser_version": multilingual_text_parser.__version__,
            }
        )
        return [sega]


if __name__ == "__main__":
    import argparse

    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-d", "--data_root", help="data root directory", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-o", "--output_dir", help="output directory", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-l", "--lang", help="speaker language", type=str, required=True
    )
    arguments_parser.add_argument(
        "--device",
        help="device to process on",
        type=str,
        default="cpu",
    )
    args = arguments_parser.parse_args()

    flist = construct_file_list(args.data_root, ext=".txt", with_subfolders=True)

    generator = SimpleSegGenerator(lang=args.lang, device=args.device)
    data = generator.read_datasamples(flist)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for meta in data:
        if "segmentation" in meta:
            for idx, sega in enumerate(meta["segmentation"]):
                fpath = output_path / f"{meta['audio_path'].name}_{idx}.TextGrid"
                sega.save(fpath, with_audio=True)
