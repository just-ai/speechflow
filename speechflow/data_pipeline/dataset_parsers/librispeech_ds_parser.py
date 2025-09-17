import typing as tp
import logging

from pathlib import Path

import numpy as np

from multilingual_text_parser.data_types import Doc, Syntagma
from praatio import tgio

from speechflow.data_pipeline.core import BaseDSParser
from speechflow.data_pipeline.core.parser_types import Metadata, MetadataTransform
from speechflow.data_pipeline.datasample_processors.data_types import TTSDataSample
from speechflow.io import AudioChunk, AudioSeg, Timestamps, tp_PATH

__all__ = ["LibriSpeechDSParser"]

LOGGER = logging.getLogger("root")


class LibriSpeechDSParser(BaseDSParser):
    """Dataset parser for sequence-to-sequence TTS task."""

    def __init__(
        self,
        preproc_fn: tp.Optional[tp.Sequence[MetadataTransform]] = None,
        memory_bound: bool = False,
        chunk_size: tp.Optional[int] = None,
        raise_on_converter_exc: bool = False,
        dump_path: tp.Optional[tp_PATH] = None,
        progress_bar: bool = True,
    ):
        from speechflow.data_pipeline.dataset_parsers import TTSDSParser

        super().__init__(
            preproc_fn,
            input_fields={"file_path", "sega", "label"},
            memory_bound=memory_bound,
            chunk_size=chunk_size,
            raise_on_converter_exc=raise_on_converter_exc,
            dump_path=dump_path,
            progress_bar=progress_bar,
        )
        self.tts_db = TTSDSParser()

    def reader(
        self, file_path: Path, label: tp.Optional[str] = None
    ) -> tp.List[Metadata]:
        if "Speechocean" in file_path.as_posix():
            file_path = file_path.with_suffix(".TextGridStage2")
            return self.tts_db.reader(file_path)

        tg = tgio.openTextgrid(file_path.as_posix())

        tiers = {}
        for name, field in tg.tierDict.items():
            tiers[name] = field.entryList

        words = " ".join([word.label for word in tiers["words"]])
        phones = [phone.label for phone in tiers["phones"]]
        tm_by_word = [(word[0], word[1]) for word in tiers["words"]]
        tm_by_phones_with_sil = [(word[0], word[1]) for word in tiers["phones"]]

        doc = Doc(words, sentenize=True, tokenize=True, add_trailing_punct_token=False)
        assert len(doc.sents) == 1
        sent = doc.sents[0]

        start = 0
        tm_by_phones = []
        for token, tm in zip(sent.tokens, tm_by_word):
            for i in range(start, len(tm_by_phones_with_sil)):
                if tm_by_phones_with_sil[i][0] == tm[0]:
                    break
            else:
                raise ValueError

            for j in range(start, len(tm_by_phones_with_sil)):
                if tm_by_phones_with_sil[j][1] == tm[1]:
                    break
            else:
                raise ValueError

            token.phonemes = tuple(
                (ph if ph != "spn" else "<UNK>") for ph in phones[i : j + 1]
            )
            tm_by_phones += [tm for tm in tm_by_phones_with_sil[i : j + 1]]
            start = j + 1

        sent.syntagmas = [Syntagma(sent.tokens)]
        sent.lang = "EN"

        wave = AudioChunk(file_path.with_suffix(".flac").as_posix().replace("-align", ""))
        sega = AudioSeg(wave, sent)
        sega.set_phoneme_timestamps(Timestamps(np.asarray(tm_by_phones)))

        metadata = {"file_path": file_path, "label": label, "sega": sega}
        return [metadata]

    def converter(self, metadata: Metadata) -> tp.List[TTSDataSample]:
        file_path = metadata.get("file_path", Path())
        if "Speechocean" in file_path.as_posix():
            return self.tts_db.converter(metadata)

        sega: AudioSeg = metadata["sega"]

        word_tm, phoneme_ts = sega.get_timestamps(relative=True)
        datasample = TTSDataSample(
            file_path=metadata.get("file_path", Path()),
            label=metadata.get("label", ""),
            audio_chunk=sega.audio_chunk,
            sent=sega.sent,
            word_timestamps=word_tm,
            phoneme_timestamps=phoneme_ts,
            speaker_name=sega.meta.get("speaker_name"),
            intonation_type=metadata.get("intonation_type"),  # type: ignore
        )
        return [datasample]
