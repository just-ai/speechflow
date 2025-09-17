import typing as tp
import logging

from pathlib import Path

import numpy as np
import multilingual_text_parser

from multilingual_text_parser import INTONATION_TYPES

from speechflow.data_pipeline.core import BaseDSParser
from speechflow.data_pipeline.core.base_ds_parser import multi_transform
from speechflow.data_pipeline.core.dataset import Dataset
from speechflow.data_pipeline.core.parser_types import Metadata, MetadataTransform
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.data_types import TTSDataSample
from speechflow.io import AudioSeg, AudioSegPreview, tp_PATH
from speechflow.logging import trace
from speechflow.utils.versioning import version_check

__all__ = ["TTSDSParser"]

LOGGER = logging.getLogger("root")


class TTSDSParser(BaseDSParser):
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
        super().__init__(
            preproc_fn,
            input_fields={"file_path", "label", "sega"},
            memory_bound=memory_bound,
            chunk_size=chunk_size,
            raise_on_converter_exc=raise_on_converter_exc,
            dump_path=dump_path,
            progress_bar=progress_bar,
        )

    def reader(
        self, file_path: Path, label: tp.Optional[str] = None
    ) -> tp.List[Metadata]:
        sega = AudioSegPreview.load(file_path)

        if "text_parser_version" in sega.meta:
            version_check(multilingual_text_parser, sega.meta["text_parser_version"])

        metadata = {"file_path": file_path, "label": label, "sega": sega}
        return [metadata]

    def converter(self, metadata: Metadata) -> tp.List[TTSDataSample]:
        sega: tp.Union[AudioSeg, AudioSegPreview] = metadata["sega"]

        if isinstance(sega, AudioSeg):
            word_ts, phoneme_ts = sega.get_timestamps(relative=True)
            ds = TTSDataSample(
                file_path=metadata.get("file_path", Path()),
                label=metadata.get("label", ""),
                audio_chunk=sega.audio_chunk,
                sent=sega.sent,
                lang=sega.sent.lang,
                speaker_name=sega.meta.get("speaker_name"),
                intonation_type=metadata.get("intonation_type"),  # type: ignore
                index=(metadata.get("index_text"), metadata.get("index_wave")),
                word_timestamps=word_ts,
                phoneme_timestamps=phoneme_ts,
            )
        elif isinstance(sega, AudioSegPreview):
            ds = TTSDataSample(
                file_path=metadata.get("file_path", Path()),
                label=metadata.get("label", ""),
                audio_chunk=sega.audio_chunk,
                lang=sega.sent.lang,
                speaker_name=sega.meta.get("speaker_name"),
                intonation_type=metadata.get("intonation_type"),  # type: ignore
                index=(metadata.get("index_text"), metadata.get("index_wave")),
            )
        else:
            raise NotImplementedError

        return [ds]

    @staticmethod
    @PipeRegistry.registry(inputs={"sega"}, outputs={"sega"})
    def insert_index(metadata: Metadata):
        sega: AudioSegPreview = metadata["sega"]

        index_text = sega.meta.get("orig_audio_path")
        index_wave = sega.audio_chunk.file_path.parent

        metadata["index_text"] = index_text
        metadata["index_wave"] = index_wave

        return [metadata]

    @staticmethod
    @multi_transform
    @PipeRegistry.registry()
    def filter_neighbors(
        all_metadata: tp.Union[tp.List[Metadata], Dataset], min_neighbors: int = 1
    ) -> tp.List[Metadata]:
        len_before = len(all_metadata)

        list_idx_text = []
        list_idx_wave = []
        for metadata in all_metadata:
            list_idx_text.append(metadata["index_text"])
            list_idx_wave.append(metadata["index_wave"])

        nuniq_text = dict(zip(*np.unique(list_idx_text, return_counts=True)))
        nuniq_wave = dict(zip(*np.unique(list_idx_wave, return_counts=True)))

        def func(md: Metadata):
            ntext = nuniq_text[md["index_text"]] - 1
            nwave = nuniq_wave[md["index_wave"]] - 1
            nboth = ntext + nwave
            return nboth >= min_neighbors

        if isinstance(all_metadata, tp.List):
            all_metadata = list(filter(func, all_metadata))
        else:
            all_metadata.filter(func)

        len_after = len(all_metadata)
        LOGGER.info(
            trace(
                "filter_neighbors",
                f"{len_before} -> {len_after}, filtering neighbors done.",
            )
        )

        return all_metadata

    @staticmethod
    @PipeRegistry.registry(inputs={"sega"}, outputs={"sega"})
    def split_by_phrases(
        metadata: Metadata,
        min_duration: tp.Optional[float] = None,
        max_duration: tp.Optional[float] = None,
        bruteforce: bool = False,
        min_bruteforce_offset: float = 0.1,
    ) -> tp.List[Metadata]:
        """Split segs by phrases.

        :param metadata: dictionary containing data of current sample
        :param min_duration: minimum wave length
        :param max_duration: maximum wave length, if the wave length is greater than this value,
            then it will be cut into pieces
        :param bruteforce: iteration over all splitting combinations
        :param min_bruteforce_offset: minimum pause duration between syntagmas
            for splitting in seconds
        :return: list of Metadata

        """
        min_duration = min_duration if min_duration else 0
        max_duration = max_duration if max_duration else 1e9

        sega: AudioSegPreview = metadata["sega"]

        metadata_to_return = []
        if min_duration <= sega.duration <= max_duration:
            metadata_to_return.append(metadata)
        else:
            LOGGER.warning(
                trace(
                    "TTSDSParser",
                    f"skip {str(metadata['file_path'])} "
                    f"with duration {round(sega.duration, 2)} secs",
                )
            )

        """
        if bruteforce:
            file_path = metadata["file_path"]
            label = metadata.get("label")
            splitted_segs = sega.split_into_syntagmas(min_offset=min_bruteforce_offset)
            for part_sega in splitted_segs:
                new_suffix = ".TextGridStage2_" + part_sega.meta["split_idxs"]
                metadata = {
                    "file_path": file_path.with_suffix(new_suffix),
                    "label": label,
                    "sega": part_sega,
                }
                metadata_to_return.append(metadata)
        """
        return metadata_to_return

    @staticmethod
    @PipeRegistry.registry(inputs={"sega"}, outputs={"sega"})
    def check_phoneme_length(
        metadata: Metadata,
        min_len: tp.Optional[float] = None,
        max_len: tp.Optional[float] = None,
    ) -> tp.List[Metadata]:
        """Skip samples with very long phonemes as outliers.

        :param metadata: dictionary containing data of current sample
        :param min_len: (in seconds) minimum phoneme duration for check the wave duration
        :param max_len: (in seconds) samples with phoneme longer than that will be skipped
        :return: list of Metadata

        """
        sega: AudioSegPreview = metadata["sega"]
        is_valid = True

        if min_len:
            phonemes = sega.sent.phonemes
            is_valid = sega.audio_chunk.duration >= min_len * len(phonemes)

        if max_len:
            assert sega.ts_by_phonemes
            if isinstance(sega.ts_by_phonemes, list):
                ts_phonemes = np.concatenate(sega.ts_by_phonemes)
            else:
                ts_phonemes = np.asarray(sega.ts_by_phonemes, dtype=np.float32)

            longer_phoneme = np.diff(ts_phonemes).max()
            is_valid = longer_phoneme < max_len

        if is_valid:
            return [metadata]
        else:
            LOGGER.warning(trace("TTSDSParser", f"skip {str(metadata['file_path'])}"))
            return []

    @staticmethod
    @PipeRegistry.registry(inputs={"sega"}, outputs={"sega"})
    def audio_strip(
        metadata: Metadata,
        pad: tp.Optional[float] = None,
        add_fade: bool = False,
        fade_threshold: float = 0.05,
    ) -> tp.List[Metadata]:
        sega: AudioSegPreview = metadata["sega"]
        audio_begin = sega.ts_bos
        audio_end = sega.ts_eos

        if pad:
            pad = max(0.0, pad)

            assert sega.ts_by_words
            audio_begin = sega.ts_by_words.begin
            audio_end = sega.ts_by_words.end

            pad_begin = pad
            if len(sega.meta.get("bos_label", "")) > 0:
                pad_begin /= 2
            offset = min(pad_begin, audio_begin - sega.audio_chunk.begin)
            audio_begin -= offset

            pad_end = pad
            if len(sega.meta.get("eos_label", "")) > 0:
                pad_end /= 2
            offset = min(pad_end, sega.audio_chunk.end - audio_end)
            audio_end += offset

        sega.ts_bos = audio_begin
        sega.ts_eos = audio_end
        sega.audio_chunk.begin = audio_begin
        sega.audio_chunk.end = audio_end

        if add_fade:
            assert sega.ts_by_words
            left_dura = sega.ts_by_words.begin - sega.audio_chunk.begin
            right_dura = sega.audio_chunk.end - sega.ts_by_words.end
            left_dura = 0.0 if left_dura < fade_threshold else left_dura / 2
            right_dura = 0.0 if right_dura < fade_threshold else right_dura / 2
            sega.audio_chunk.fade_duration = (left_dura, right_dura)

        return [metadata]

    @staticmethod
    @PipeRegistry.registry(inputs={"sega"}, outputs={"sega"})
    def get_intonation_type(
        metadata: Metadata, intonation_types: tp.Optional[tp.Dict[tp.Any, str]] = None
    ) -> tp.List[Metadata]:
        """Return intonation type tag."""
        if intonation_types is None:
            intonation_types = INTONATION_TYPES

        sega: AudioSegPreview = metadata["sega"]
        src_wav_path = sega.meta.get("orig_audio_path")

        if not sega.meta.get("intonation_type"):
            if "?" in sega.sent.text:
                if src_wav_path and "question" in src_wav_path:
                    pos = src_wav_path.find("question_")
                    pos = pos + 9 if pos != -1 else src_wav_path.find("questions_") + 10
                    q_type = src_wav_path[pos]
                    if q_type.isdigit() and intonation_types.get(int(q_type)):
                        intonation_type = intonation_types[int(q_type)]
                    else:
                        intonation_type = intonation_types["."]
                else:
                    intonation_type = intonation_types[0]
            elif "!" in sega.sent.text:
                intonation_type = intonation_types["!"]
            else:
                intonation_type = intonation_types["."]

            metadata["intonation_type"] = intonation_type

        return [metadata]

    @staticmethod
    @PipeRegistry.registry(inputs={"sega"}, outputs={"sega"})
    def get_simple_intonation_type(
        metadata: Metadata,
        punctuation_marks: tp.Tuple[str, ...] = (".", "!", "?"),
        intonation_types: tp.Optional[tp.Dict[tp.Any, str]] = None,
    ) -> tp.List[Metadata]:
        """Return intonation type tag."""
        if intonation_types is None:
            intonation_types = INTONATION_TYPES

        sega: AudioSegPreview = metadata["sega"]
        for mark in punctuation_marks:
            if mark in sega.sent.text and mark in intonation_types:
                metadata["intonation_type"] = intonation_types[mark]
                break
        else:
            metadata["intonation_type"] = intonation_types["."]

        return [metadata]


if __name__ == "__main__":
    from speechflow.io.flist import read_file_list
    from speechflow.utils.fs import get_root_dir

    _root = get_root_dir()
    _fpath = list((_root / "examples/simple_datasets/speech/SEGS").rglob("filelist.txt"))[
        0
    ]
    _flist = read_file_list(_fpath, max_num_samples=100)
    assert isinstance(_flist, tp.List)

    _parser = TTSDSParser()

    _data = _parser.read_datasamples(file_list=_flist, data_root=_root)
    print(_data.item(0).file_path)
