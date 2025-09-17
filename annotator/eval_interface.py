import typing as tp
import logging
import tempfile

from pathlib import Path

import numpy as np
import torch

from multilingual_text_parser.data_types import Doc
from multilingual_text_parser.parser import TextParser

from annotator.align import Aligner, AlignStage
from speechflow.io import AudioChunk, AudioSeg, Timestamps, check_path, tp_PATH
from speechflow.utils.fs import get_root_dir

LOGGER = logging.getLogger("root")


class AnnotatorEvaluationInterface:
    def __init__(
        self,
        ckpt_stage1: tp_PATH,
        ckpt_stage2: tp_PATH,
        device: str = "cpu",
        last_word_correction: bool = False,
        audio_duration_limit: tp.Optional[float] = 15,  # in seconds
    ):
        self.use_reverse_mode = last_word_correction

        self.aligner_stage1 = Aligner(
            ckpt_path=ckpt_stage1,
            stage=AlignStage.stage1,
            device=device,
            max_duration=audio_duration_limit,
        )
        self.aligner_stage2 = Aligner(
            ckpt_path=ckpt_stage2,
            stage=AlignStage.stage2,
            device=device,
            max_duration=audio_duration_limit,
        )

        if self.use_reverse_mode:
            self.aligner_stage1_reverse = Aligner(
                ckpt_path=ckpt_stage1,
                stage=AlignStage.stage1,
                device=device,
                reverse_mode=True,
                max_duration=audio_duration_limit,
                preload=self.aligner_stage1.model,
            )
            self.aligner_stage2_reverse = Aligner(
                ckpt_path=ckpt_stage2,
                stage=AlignStage.stage2,
                device=device,
                reverse_mode=True,
                max_duration=audio_duration_limit,
                preload=self.aligner_stage2.model,
            )

        self.text_parser = {}

    @property
    def lang(self) -> str:
        return self.aligner_stage1.lang

    @staticmethod
    def _cat_sentences(text: str) -> str:
        sents = Doc(text, sentenize=True, tokenize=True).sents
        if len(sents) > 1:
            sents = [sent.tokens[:-1] for sent in sents[:-1]] + [sents[-1].tokens]
            sents = [" ".join([token.text for token in sent]) for sent in sents]
            return ", ".join(sents)
        else:
            return text

    @staticmethod
    def _fix_pauses(file_name: Path, min_pause_len) -> AudioSeg:
        file_name_reverse = file_name.with_suffix(f"{file_name.suffix}_reverse")

        sega = AudioSeg.load(file_name)
        _, ph_ts = sega.get_timestamps()

        sega_reverse = AudioSeg.load(file_name_reverse)
        _, ph_ts_reverse = sega_reverse.get_timestamps()

        for idx, (ts, ts_reverse) in enumerate(zip(ph_ts[:-1], ph_ts_reverse[:-1])):
            a = ts[-1][1]
            ar = ts_reverse[-1][1]
            b = ph_ts[idx + 1][0][0]
            br = ph_ts_reverse[idx + 1][0][0]
            if b - a < min_pause_len and br - ar < min_pause_len:
                ph_ts[idx][-1][1] = b

        sega.set_phoneme_timestamps(ph_ts, ts_begin=sega.ts_bos, ts_end=sega.ts_eos)
        return sega

    @staticmethod
    def _fix_last_word(file_name: Path):
        file_name_reverse = file_name.with_suffix(f"{file_name.suffix}_reverse")

        sega = AudioSeg.load(file_name)
        word_ts, ph_ts = sega.get_timestamps()

        sega_reverse = AudioSeg.load(file_name_reverse)
        word_ts_reverse, ph_ts_reverse = sega_reverse.get_timestamps()

        if (
            word_ts[-1][1] - word_ts[-1][0]
            < word_ts_reverse[-1][1] - word_ts_reverse[-1][0]
        ):
            a, b = ph_ts[-1][0][0], 0
            for idx, ts_reverse in enumerate(ph_ts_reverse[-1]):
                b = a + (ts_reverse[1] - ts_reverse[0])
                b = min(b, sega_reverse.ts_eos)
                ph_ts[-1][idx][0] = a
                ph_ts[-1][idx][1] = b
                a = b

            dura = np.diff(ph_ts[-1])
            if abs(dura[-1]) < 1.0e-4:
                dura -= dura * 0.01
                delta = (ph_ts[-1].duration - dura.sum()) / len(ph_ts[-1])
                dura += delta
                ph_ts[-1] = ph_ts[-1][0][0] + Timestamps.from_durations(dura)

            if sega_reverse.ts_eos - ph_ts[-1][-1][1] < 0.02:
                ph_ts[-1][-1][1] = sega_reverse.ts_eos

            ts_begin = sega.ts_bos
            ts_end = sega_reverse.ts_eos
            sega.set_phoneme_timestamps(ph_ts, ts_begin=ts_begin, ts_end=ts_end)

        return sega

    def prepare_text(
        self,
        text: str,
        lang: str,
    ) -> Doc:
        if (
            self.aligner_stage1.lang_id_map
            and lang not in self.aligner_stage1.lang_id_map
        ):
            raise ValueError(f"Language {lang} is not support in current model!")

        if lang not in self.text_parser:
            LOGGER.info(f"Initial TextParser for {lang} language")
            self.text_parser[lang] = TextParser(
                lang, device=str(self.aligner_stage1.device)
            )

        doc = self.text_parser[lang].process(Doc(text))
        return doc

    def get_sega_from_text(
        self,
        text: str,
        audio_path: Path,
        lang: str,
        speaker_name: str,
    ) -> AudioSeg:
        sents = self.prepare_text(self._cat_sentences(text), lang=lang).sents
        assert len(sents) == 1

        sent = sents[0]
        audio_chunk = AudioChunk(file_path=audio_path).load()

        words = sent.get_words()
        ts_intervals = np.linspace(audio_chunk.begin, audio_chunk.end, len(words) + 1)
        ts = Timestamps(np.asarray(list(zip(ts_intervals[:-1], ts_intervals[1:]))))

        sega = AudioSeg(audio_chunk, sent)
        sega.set_word_timestamps(ts)
        sega.meta["lang"] = lang
        sega.meta["speaker_name"] = speaker_name
        return sega

    @check_path
    def _process(
        self,
        text: tp.Optional[str] = None,
        audio_path: tp.Optional[tp_PATH] = None,
        lang: tp.Optional[str] = None,
        speaker_name: tp.Optional[str] = None,
        sega_path: tp.Optional[tp_PATH] = None,
    ) -> AudioSeg:
        with tempfile.TemporaryDirectory() as tmp_dir:

            if sega_path is not None:
                sega = AudioSeg.load(sega_path)
                audio_path = sega.audio_chunk.file_path
            elif text is not None and audio_path is not None:
                sega = self.get_sega_from_text(text, audio_path, lang, speaker_name)
            else:
                raise NotImplementedError("Set 'text' and 'audio_path' or 'sega_path'")

            file_name = Path(tmp_dir) / f"{sega.audio_chunk.file_path.stem}.TextGrid"
            file_name = file_name.absolute()
            sega.save(file_name, with_audio=True)

            self.aligner_stage1.align_sega(file_name)
            if self.use_reverse_mode:
                self.aligner_stage1_reverse.align_sega(file_name)

            file_name = file_name.with_suffix(".TextGridStage1")
            if self.use_reverse_mode:
                sega = self._fix_pauses(file_name, self.aligner_stage2.min_pause_len)
                sega.save(file_name)

            self.aligner_stage2.align_sega(file_name)
            if self.use_reverse_mode:
                self.aligner_stage2_reverse.align_sega(file_name)

            file_name = file_name.with_suffix(".TextGridStage2")
            if self.use_reverse_mode:
                sega = self._fix_last_word(file_name)
            else:
                sega = AudioSeg.load(file_name)

            sega.audio_chunk.file_path = audio_path

        return sega

    @tp.overload
    def process(
        self,
        text: str,
        audio_path: tp_PATH,
        lang: str,
        speaker_name: tp.Optional[str] = None,
    ) -> AudioSeg:
        ...

    @tp.overload
    def process(self, sega_path: tp_PATH) -> AudioSeg:
        ...

    def process(self, *args, **kwargs) -> AudioSeg:
        if "sega_path" in kwargs:
            return self._process(sega_path=kwargs["sega_path"])
        else:
            return self._process(*args, **kwargs)


if __name__ == "__main__":
    # The minimum requirement is 4GB of VRAM for CUDA devices

    from annotator.audio_transcription import OpenAIASR

    _lang = "RU"
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    _mfa_stage1_path = (
        "multilingual-forced-alignment/mfa_v1.0/mfa_stage1_epoch=29-step=468750.pt"
    )
    _mfa_stage2_path = (
        "multilingual-forced-alignment/mfa_v1.0/mfa_stage2_epoch=59-step=937500.pt"
    )

    _annotator = AnnotatorEvaluationInterface(
        ckpt_stage1=_mfa_stage1_path,
        ckpt_stage2=_mfa_stage2_path,
        device=_device,
        last_word_correction=False,
        audio_duration_limit=None,
    )

    _audio_path = get_root_dir() / "tests/data/test_audio.wav"
    _text = (
        OpenAIASR(_lang, "medium", _device)
        .read_datasamples([_audio_path])
        .item(0)["text"]
    )

    _sega = _annotator.process(_text, _audio_path, _lang)
    _sega.save("sega.tg", with_audio=True)
