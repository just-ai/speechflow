import json
import typing as tp
import logging

from pathlib import Path

import numpy as np
import torch
import multilingual_text_parser

from multilingual_text_parser.data_types import PUNCTUATION_ALL, Doc, Token, TokenUtils
from multilingual_text_parser.parser import TextParser

from annotator.utils import nmalign
from speechflow.data_pipeline.core import BaseDSParser, PipeRegistry
from speechflow.data_pipeline.core.components import (
    init_metadata_preprocessing_from_config,
)
from speechflow.data_pipeline.core.parser_types import Metadata
from speechflow.data_pipeline.datasample_processors.algorithms.text_processing.sequence_algorithms import (
    needleman_wunsch,
)
from speechflow.io import AudioChunk, AudioSeg, Config, Timestamps, construct_file_list
from speechflow.logging import log_to_file, trace
from speechflow.utils.gpu_info import get_freer_gpu

__all__ = ["AudiobookSpliter"]

LOGGER = logging.getLogger("root")


class AudiobookSpliter(BaseDSParser):
    """Split audiobook by sentences."""

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
                "pipe": ["text_prepare", "transcription_prepare"],
                "pipe_cfg": {
                    "text_prepare": {"lang": lang, "device": device},
                    "transcription_prepare": {"lang": lang, "device": device},
                },
            }
        )
        preproc_fn = init_metadata_preprocessing_from_config(AudiobookSpliter, cfg)

        super().__init__(
            preproc_fn,
            chunk_size=1,
            raise_on_converter_exc=raise_on_converter_exc,
            release_func=self.release,
        )

        self._lang = lang
        self._text_from_label = text_from_label

    @staticmethod
    def release(_):
        if hasattr(AudiobookSpliter, "parser"):
            AudiobookSpliter.parser = None

    def reader(
        self, file_path: Path, label: tp.Optional[str] = None
    ) -> tp.List[Metadata]:
        audio_path = AudioChunk.find_audio(file_path)
        if audio_path is None:
            return []

        if self._text_from_label and label:
            text = label.split("|", 1)[0]
        else:
            text_path = file_path.with_suffix(".txt")
            if text_path.exists():
                text = text_path.read_text(encoding="utf-8")
            else:
                return []

        transc_path = file_path.with_suffix(".whisper")
        if not transc_path.exists():
            transc_path = file_path.with_suffix(".json")

        if transc_path.exists():
            json_dump = transc_path.read_text(encoding="utf-8")
            transcription = json.loads(json_dump)
        else:
            return []

        metadata = {
            "file_path": file_path,
            "audio_path": audio_path,
            "text": text,
            "transcription": transcription,
        }
        return [metadata]

    def converter(self, metadata: Metadata) -> tp.List[Metadata]:
        try:
            metadata["segmentation"] = self._text_alignment(metadata)
        except Exception as e:
            metadata["segmentation"] = []
            LOGGER.error(trace(self, e))

        return [metadata]

    @staticmethod
    def _set_pauses_from_asr(orig_tokens: tp.List[Token], transc_tokens: tp.List[Token]):
        orig_words = TokenUtils.get_word_tokens(orig_tokens)
        transc_words = TokenUtils.get_word_tokens(transc_tokens)

        s1 = TokenUtils.get_text_from_tokens(orig_words, as_norm=True)
        s2 = TokenUtils.get_text_from_tokens(transc_words, as_norm=True)

        tokens_alignment = needleman_wunsch(s1.split(" "), s2.split(" "))

        for a, b in zip(tokens_alignment[:-1], tokens_alignment[1:]):
            if None in a or None in b:
                continue
            if b[0] - a[0] == 1 and b[1] - a[1] == 1:
                tm_begin = transc_words[a[1]].meta["ts"][1]
                tm_end = transc_words[b[1]].meta["ts"][0]
                orig_words[a[0]].asr_pause = str(round(tm_end - tm_begin, 3))

    def _text_alignment(
        self, metadata: Metadata, thr1: float = 0.5, thr2: float = 0.7
    ) -> tp.List[AudioSeg]:
        sents = metadata["text"].sents
        transc_sents = metadata["transc_text"].sents

        seq1 = [sent.as_norm() for sent in sents]
        seq2 = [sent.as_norm() for sent in transc_sents]

        res, dst = nmalign.match(seq1, seq2, try_subseg=True, cutoff=thr1)
        res_ind, res_beg, res_end = res

        res_ind_correct = []
        for i in np.arange(0, len(res_ind)):
            is_outlier = False

            if res_ind[i] > 0:
                for j in np.arange(i + 1, len(res_ind) - 1):
                    if res_ind[j] < 0:
                        continue
                    if res_ind[j] < res_ind[i] and res_ind[j + 1] - res_ind[j] == 1:
                        is_outlier = True
                        break

            if is_outlier:
                res_ind_correct.append(-1)
            else:
                res_ind_correct.append(res_ind[i])

        res_ind = res_ind_correct

        result = []
        for real_idx, (transc_idx, transc_begin, transc_end) in enumerate(
            zip(res_ind, res_beg, res_end)
        ):
            try:
                if transc_idx == -1:
                    continue

                sent = sents[real_idx]
                transc_tokens = transc_sents[transc_idx].tokens

                s1 = seq1[real_idx]
                s2 = seq2[transc_idx]
                s2 = s2[
                    (0 if transc_begin == -1 else transc_begin) : (
                        len(s2) if transc_end == -1 else transc_end
                    )
                ]

                if (
                    len(s1) > len(s2)
                    and transc_end == len(s2)
                    and transc_idx + 1 < len(seq2)
                ):
                    s2 += " " + seq2[transc_idx + 1]
                    transc_tokens += transc_sents[transc_idx + 1].tokens
                    transc_end += len(seq2[transc_idx + 1])

                offset = min(15, len(s1))
                if nmalign.match([s1[:offset]], [s2[:offset]], cutoff=thr1)[1][0] < thr2:
                    LOGGER.warning(
                        f"sentences no match [text]/[transcription]: {metadata['audio_path'].as_posix()}|[{s1}]/[{s2}]"
                    )
                    continue
                if (
                    nmalign.match(
                        [s1[len(s1) - offset :]],
                        [s2[len(s2) - offset :]],
                        cutoff=thr1,
                    )[1][0]
                    < thr2
                ):
                    LOGGER.warning(
                        f"sentences no match [text]/[transcription]: {metadata['audio_path'].as_posix()}|[{s1}]/[{s2}]"
                    )
                    continue

                if transc_begin == -1 and transc_end == -1:
                    words = transc_tokens
                else:
                    words = []
                    total_len = 0
                    for token in transc_tokens:
                        if transc_begin <= total_len <= transc_end:
                            if not token.is_punctuation:
                                words.append(token)

                        total_len += len(token.norm) + 1
                        if total_len > transc_end:
                            break

                try:
                    self._set_pauses_from_asr(sent.tokens, words)
                except Exception as e:
                    log_to_file(trace(self, e))

                ts_begin = words[0].meta["ts"][0]
                ts_end = words[-1].meta["ts"][1]

                bos_ts = words[0].meta.get("bos_ts", ts_begin)
                bos_label = words[0].meta.get("bos_label", "")

                eos_ts = words[-1].meta.get("eos_ts", ts_end)
                eos_label = words[-1].meta.get("eos_label", "")

                ts_words: tp.List[tp.Tuple[float, float]] = []
                group = TokenUtils.group_tokens_by_word(sent.tokens)
                total_phrase_len = sum(
                    [len(t) for t in sent.tokens if not t.is_punctuation]
                )
                a = b = ts_begin
                for tokens in group:
                    curr_len = sum([len(t) for t in tokens if not t.is_punctuation])
                    b += (ts_end - ts_begin) * curr_len / total_phrase_len
                    ts_words.append((a, b))
                    a = b

                audio_chunk = AudioChunk(
                    file_path=metadata["audio_path"],
                    begin=bos_ts,
                    end=eos_ts,
                )

                sega = AudioSeg(audio_chunk, sent)
                sega.set_word_timestamps(
                    Timestamps(np.asarray(ts_words)), ts_begin=bos_ts, ts_end=eos_ts
                )
                sega.meta.update(
                    {
                        "lang": TextParser.locale_to_language(self._lang),
                        "bos_label": bos_label,
                        "eos_label": eos_label,
                        "orig_audio_path": Path(audio_chunk.file_path).as_posix(),
                        "orig_audio_chunk": (audio_chunk.begin, audio_chunk.end),
                        "orig_audio_samplerate": audio_chunk.sr,
                        "text_parser_version": multilingual_text_parser.__version__,
                    }
                )
                setattr(sega, "transcription", transc_tokens)
                result.append(sega)
            except Exception as e:
                LOGGER.error(trace(self, e))

        return result

    @staticmethod
    def _init_text_parser(lang: str, device: str) -> "TextParser":
        if getattr(AudiobookSpliter, "parser", None) is None:
            with AudiobookSpliter.lock:
                if device == "cuda":
                    device = f"cuda:{get_freer_gpu(strict=False)}"

                parser = TextParser(lang=lang, device=device)
                setattr(AudiobookSpliter, "parser", parser)
        else:
            parser = getattr(AudiobookSpliter, "parser")

        return parser

    @staticmethod
    @PipeRegistry.registry()
    def text_prepare(metadata: Metadata, lang: str, device: str) -> tp.List[Metadata]:
        parser = AudiobookSpliter._init_text_parser(lang, device)

        if parser.device != "cpu":
            with torch.cuda.device(parser.device):
                metadata["text"] = parser.process(Doc(metadata["text"]))
        else:
            metadata["text"] = parser.process(Doc(metadata["text"]))

        return [metadata]

    @staticmethod
    @PipeRegistry.registry()
    def transcription_prepare(
        metadata: Metadata, lang: str, device: str
    ) -> tp.List[Metadata]:
        parser = AudiobookSpliter._init_text_parser(lang, device)

        if parser.device != "cpu":
            with torch.cuda.device(parser.device):
                return AudiobookSpliter.openai_asr_prepare(metadata, parser=parser)
        else:
            return AudiobookSpliter.openai_asr_prepare(metadata, parser=parser)

    @staticmethod
    def openai_asr_prepare(
        metadata: Metadata,
        parser: TextParser = None,
        word_offset: int = 1,
        wave_offset: float = 1.0,
    ) -> tp.List[Metadata]:
        audio_path: Path = metadata["audio_path"]
        asr_timestamps: dict = metadata["transcription"]["timestamps"]

        timestamps = []
        for item in asr_timestamps:
            word = item[0].translate(str.maketrans("", "", PUNCTUATION_ALL))
            if word.strip():
                timestamps.append(item)

        text_orig = " ".join([item[0] for item in timestamps])
        text = Doc(text_orig, sentenize=True)
        for sent in text.sents:
            sent.tokens = [Token(item) for item in sent.text.split()]

        words_by_ts = text_orig.split(" ")
        words = text.text.split(" ")
        assert len(words_by_ts) == len(words) == len(timestamps)

        ts_index = 0
        for token in text.tokens:
            if token.is_punctuation:
                continue

            assert token.text == words[ts_index]
            token.meta["ts"] = tuple(timestamps[ts_index][1:])

            try:
                if token.is_number:
                    token.text = parser.process(Doc(token.text)).as_norm()
            except Exception as e:
                LOGGER.warning(trace("openai_asr_prepare", e, token.text))

            # fix timestamp
            # dura = token.meta["ts"][1] - token.meta["ts"][0]
            # if len(token.text) <= 3 and dura > 0.2:
            #     token.meta["ts"] = (
            #         token.meta["ts"][0],
            #         token.meta["ts"][0] + 0.5 * dura,
            #     )

            bos_index = max(ts_index - word_offset, 0)
            if bos_index != ts_index:
                bos_ts = timestamps[bos_index][1]
                bos_label = " ".join([item[0] for item in timestamps[bos_index:ts_index]])
                token.meta["bos_ts"] = bos_ts
                token.meta["bos_label"] = bos_label
                token.meta["prev_word_ts"] = tuple(asr_timestamps[bos_index][1:])
            else:
                token.meta["bos_ts"] = max(0.0, token.meta["ts"][0] - wave_offset)
                token.meta["bos_label"] = ""

            eos_index = min(ts_index + word_offset, len(timestamps) - 1)
            if eos_index != ts_index:
                eos_ts = timestamps[eos_index][2]
                eos_label = " ".join(
                    [item[0] for item in timestamps[ts_index + 1 : eos_index + 1]]
                )
                token.meta["eos_ts"] = eos_ts
                token.meta["eos_label"] = eos_label
                token.meta["next_word_ts"] = tuple(asr_timestamps[eos_index][1:])
            else:
                token.meta["eos_ts"] = min(
                    AudioChunk(audio_path).duration,
                    token.meta["ts"][1] + wave_offset,
                )
                token.meta["eos_label"] = ""

            ts_index += 1

        metadata["transc_text"] = text
        return [metadata]


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

    generator = AudiobookSpliter(lang=args.lang)
    data = generator.read_datasamples(flist)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for meta in data:
        if "segmentation" in meta:
            for idx, sega in enumerate(meta["segmentation"]):
                try:
                    _file_name = f"{meta['audio_path'].name.split('.')[0]}_{idx}.TextGrid"
                    _file_path = output_path / _file_name
                    sega.meta["speaker_name"] = "LJSpeech"
                    sega.save(_file_path, with_audio=True)
                    print(sega.sent.text_orig, _file_name)
                except Exception as e:
                    print(e)
