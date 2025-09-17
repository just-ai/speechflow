import json
import typing as tp

from copy import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from multilingual_text_parser.data_types import Doc, Position, Sentence, Syntagma, Token
from praatio import tgio

from speechflow.io import AudioChunk, AudioFormat, Timestamps
from speechflow.io.utils import check_path, tp_PATH

__all__ = ["AudioSeg", "AudioSegPreview"]


def _fp_eq(a, b, eps: float = 1.0e-6):
    return np.abs(np.float32(a) - np.float32(b)) < eps


def _fix_json_string(string: str) -> str:
    string = string.replace("{'", '{"').replace("'}", '"}')
    string = string.replace("', '", '", "')
    string = string.replace("': '", '": "')
    string = string.replace("n': ", 'n": ')
    string = string.replace("r': ", 'r": ')
    string = string.replace("t': ", 't": ')
    string = string.replace("': [", '": [').replace("], '", '], "')
    string = string.replace("': true, '", '": true, "')
    string = string.replace("': false, '", '": false, "')
    string = string.replace("': null,", '": null,')
    string = string.replace("\\'", "'")
    string = string.replace(", '", ', "')
    string = string.replace("': ", '": ')
    return string


def _extract_meta(string: str):
    string = string[string.rfind('"meta"') + len('"meta"') + 1 :]
    string = string[string.find('"{') + 1 :][::-1]
    string = string[string.find('"}') + 1 :][::-1]
    return json.loads(_fix_json_string(string))


def _remove_service_tokens(
    tiers: tp.Dict,
) -> tp.Tuple[tp.Dict, tp.Optional[float], tp.Optional[float]]:
    ts_bos = ts_eos = None
    if tiers["text"][0][2] == "BOS":
        ts_bos = tiers["text"][0][1]
    if tiers["text"][-1][2] == "EOS":
        ts_eos = tiers["text"][-1][0]

    for name, field in tiers.items():
        tiers[name] = [item for item in field if item[2] not in ["BOS", "EOS"]]

    return tiers, ts_bos, ts_eos


class AudioSeg:
    @check_path(assert_file_exists=True)
    def __init__(
        self,
        audio_chunk: AudioChunk,
        sent: Sentence,
    ):
        assert audio_chunk.duration > 0, "invalid waveform!"
        assert len(sent) > 0, "sentence contains no tokens!"
        self.audio_chunk = audio_chunk
        self.sent: Sentence = sent
        self.ts_by_words: tp.Optional[Timestamps] = None
        self.ts_by_phonemes: tp.Optional[tp.List[Timestamps]] = None
        self.ts_bos: float = self.audio_chunk.begin
        self.ts_eos: float = self.audio_chunk.end
        self.meta: tp.Dict[str, tp.Any] = {}
        self.sega_path: tp.Optional[tp_PATH] = None

    @property
    def duration(self) -> float:
        return self.ts_eos - self.ts_bos

    @staticmethod
    def _get_sent(tiers: tp.Dict[str, tp.Any]) -> Sentence:
        words = " ".join([word[2] for word in tiers["text"]])
        doc = Doc(words, sentenize=True, tokenize=True)

        assert len(doc.sents) == 1, doc.text
        sent = doc.sents[0]

        if "orig" in tiers:
            sent.text_orig = tiers["orig"][0][2]

        for token in sent.tokens:
            if token.is_punctuation:
                token.pos = "PUNCT"

        return sent

    def set_word_timestamps(
        self,
        ts: Timestamps,
        ts_begin: tp.Optional[float] = None,
        ts_end: tp.Optional[float] = None,
        relative: bool = False,
    ):
        ts_begin = self.audio_chunk.begin if ts_begin is None else ts_begin
        ts_end = self.audio_chunk.end if ts_end is None else ts_end
        assert ts_begin is not None and ts_end is not None

        if relative:
            ts += self.audio_chunk.begin
            ts_begin += self.audio_chunk.begin
            ts_end += self.audio_chunk.begin

        self.ts_bos = ts_begin
        self.ts_eos = ts_end

        words = self.sent.get_words()
        self.ts_by_words = ts

        if not len(words) == len(ts):
            raise ValueError(
                f"Number of words: {len(words)} doesnt match number of passed timestamps: {len(ts)}"
            )
        if not np.float32(ts.begin) >= np.float32(self.audio_chunk.begin):
            raise ValueError(
                "Bounds of passed timestamps are not compatible with begin of waveform."
            )
        if not np.float32(ts.end) <= np.float32(self.audio_chunk.end):
            raise ValueError(
                "Bounds of passed timestamps are not compatible with end of waveform."
            )

        if self.ts_by_phonemes is None:
            self.ts_by_phonemes = []
            for (a, b), word in zip(ts, words):  # type: ignore
                if word.phonemes:
                    num_phonemes = len(word.phonemes)
                    step = float((b - a) / num_phonemes)
                    t = [(i * step, (i + 1) * step) for i in range(num_phonemes)]
                else:
                    t = [(0, b - a)]
                self.ts_by_phonemes.append(Timestamps(np.asarray(t)) + a)

    def set_phoneme_timestamps(
        self,
        ts: tp.Union[Timestamps, tp.List[Timestamps]],
        ts_begin: tp.Optional[float] = None,
        ts_end: tp.Optional[float] = None,
        relative: bool = False,
    ):
        if isinstance(ts, tp.List):
            ts = Timestamps(np.concatenate(ts))

        ts_by_phonemes = []
        for ph_word in self.sent.get_phonemes():
            ts_word, ts = ts[: len(ph_word)], ts[len(ph_word) :]
            ts_by_phonemes.append(Timestamps(np.asarray(ts_word)))
        assert (
            len(ts) == 0
        ), "The number of phonemes and the number of timestamps do not match."

        if relative:
            ts_by_phonemes = [ts + self.audio_chunk.begin for ts in ts_by_phonemes]

        self.ts_by_phonemes = ts_by_phonemes

        ts_by_words = [(ts.begin, ts.end) for ts in ts_by_phonemes]
        self.set_word_timestamps(Timestamps(np.asarray(ts_by_words)), ts_begin, ts_end)

    def get_tier(self, name: str, relative: bool = False):
        assert self.ts_by_words, "timestamps not set!"

        seq = []
        if name == "orig":
            seq.append(
                (self.ts_by_words.begin, self.ts_by_words.end, self.sent.text_orig)
            )
        elif name == "syntagmas":
            for synt in self.sent.syntagmas:
                word_begin = self.sent.get_word_index(synt[0])
                word_end = self.sent.get_word_index(synt[-1])
                ts_begin = self.ts_by_words[word_begin][0]
                ts_end = self.ts_by_words[word_end][1]
                seq.append((ts_begin, ts_end, synt.position.name))
        elif name == "phonemes":
            phonemes = self.sent.get_phonemes()
            for ts_list, ph_list in zip(self.ts_by_phonemes, phonemes):  # type: ignore
                if ph_list:
                    assert len(ts_list) == len(ph_list)
                    for ts, ph in zip(ts_list, ph_list):  # type: ignore
                        if not isinstance(ph, str):
                            ph = "|".join(ph)
                        seq.append((ts[0], ts[1], ph))
        elif name == "breath_mask":
            breath_mask = self.sent.get_attr(name, group=True, with_punct=name == "text")
            for ts_list, bm_list in zip(self.ts_by_phonemes, breath_mask):  # type: ignore
                if bm_list:
                    if isinstance(bm_list[0], tp.List):
                        bm_list = bm_list[0]
                    if len(ts_list) != len(bm_list):
                        bm_list = bm_list * len(ts_list)
                    for ts, m in zip(ts_list, bm_list):  # type: ignore
                        seq.append((ts[0], ts[1], "undefined" if m is None else str(m)))
        else:
            words = self.sent.get_attr(name, group=True, with_punct=name == "text")
            for ts, word in zip(self.ts_by_words, words):  # type: ignore
                word = [str(x) if x is not None else "undefined" for x in word]
                label = "".join(word)
                seq.append((ts[0], ts[1], label))

        if not _fp_eq(self.audio_chunk.begin, self.ts_bos):
            seq.insert(0, (self.audio_chunk.begin, self.ts_bos, "BOS"))
        if not _fp_eq(self.ts_eos, self.audio_chunk.end):
            seq.append((self.ts_eos, self.audio_chunk.end, "EOS"))

        if relative:
            offset = self.audio_chunk.begin
            for idx in range(len(seq)):
                begin, end, label = seq[idx]
                seq[idx] = (
                    max(begin - offset, 0),
                    min(end - offset, self.audio_chunk.duration),
                    label,
                )

        return seq

    def get_tier_for_meta(self, meta: dict, relative: bool = False):
        meta.update(self.meta)
        dump = json.dumps(meta, ensure_ascii=False).replace('"', "'")
        begin = self.audio_chunk.begin
        end = self.audio_chunk.end

        if relative:
            offset = self.audio_chunk.begin
            begin = max(begin - offset, 0)
            end = min(end - offset, self.audio_chunk.duration)

        tier = tgio.IntervalTier(
            "meta", [(begin, end, dump)], 0, maxT=self.audio_chunk.duration
        )
        return tier

    @staticmethod
    def timestamps_from_tier(
        tier: tp.List[tp.Tuple[float, float, str]]
    ) -> tp.Tuple["Timestamps", tp.Optional[float], tp.Optional[float]]:
        tiers, ts_begin, ts_end = _remove_service_tokens({"text": tier})
        tm = [t[:2] for t in tiers["text"]]
        return Timestamps(np.asarray(tm)), ts_begin, ts_end

    @staticmethod
    @check_path(assert_file_exists=True)
    def load(
        file_path: tp_PATH,
        load_audio: bool = False,
        audio_path: tp.Optional[tp_PATH] = None,
    ) -> "AudioSeg":
        tg = tgio.openTextgrid(file_path.as_posix())

        tiers = {}
        for name, field in tg.tierDict.items():
            tiers[name] = field.entryList

        tiers, ts_bos, ts_eos = _remove_service_tokens(tiers)

        sent = AudioSeg._get_sent(tiers)
        ts_by_word = [(word[0], word[1]) for word in tiers["text"]]

        if "meta" in tiers:
            s = _fix_json_string(tiers["meta"][0].label)
            try:
                meta = json.loads(s)
            except Exception as e:
                raise RuntimeError(f"{e}: {s}")
            sent.lang = meta.get("lang")
            sent.position = Position[meta.get("sent_position", "first")]
        else:
            meta = {}

        audio_chunk = AudioSeg.load_audio_chunk(file_path, meta, audio_path)

        if load_audio:
            audio_chunk.load()
            assert _fp_eq(audio_chunk.duration, meta["audio_chunk"][1], eps=1.0e-3)

        words = sent.get_words()
        assert len(words) == len(tiers["text"]), f"{file_path}"

        for name in (
            "stress",
            "pos",
            "emphasis",
            "id",
            "head_id",
            "rel",
            "asr_pause",
            "prosody",
        ):
            if name in tiers:
                for (_, _, label), word in zip(tiers[name], words):
                    label = None if label == "undefined" else label
                    setattr(word, name, label)

        ph_word = []
        ts_by_ph = []
        word_idx = 0
        if "phonemes" in tiers:
            for begin, end, label in tiers["phonemes"]:
                if "|" in label:
                    label = tuple(label.split("|"))
                ph_word.append(label)
                ts_by_ph.append((begin, end))
                if _fp_eq(end, tiers["text"][word_idx][1]):
                    words[word_idx].phonemes = tuple(ph_word)
                    ph_word = []
                    word_idx += 1
            assert all([word.phonemes for word in words]), f"{file_path}"

        if "syntagmas" in tiers:
            synt_tokens: tp.List[tp.List[Token]] = []
            words_with_punct = sent.get_words_with_punct()
            for (begin, _, label), tokens in zip(tiers["text"], words_with_punct):
                if any([begin == synt[0] for synt in tiers["syntagmas"]]):
                    synt_tokens.append([])
                synt_tokens[-1] += tokens

            syntagmas = []
            for tokens, synt_position in zip(synt_tokens, tiers["syntagmas"]):
                syntagma = Syntagma(tokens)
                syntagma.position = Position[synt_position.label]
                syntagmas.append(syntagma)
            sent.syntagmas = syntagmas

        sega = AudioSeg(audio_chunk, sent)

        if ts_by_ph:
            sega.set_phoneme_timestamps(
                Timestamps(np.asarray(ts_by_ph)),
                ts_begin=ts_bos,
                ts_end=ts_eos,
            )
        else:
            sega.set_word_timestamps(
                Timestamps(np.asarray(ts_by_word)),
                ts_begin=ts_bos,
                ts_end=ts_eos,
            )

        sega.meta = meta
        sega.sega_path = file_path
        return sega

    @check_path(make_dir=True)
    def save(
        self,
        file_path: tp_PATH,
        overwrite: bool = True,
        with_audio: bool = False,
        audio_format: str = "wav",
        fields: tp.Tuple[str, ...] = (
            "orig",
            "syntagmas",
            "text",
            "stress",
            "phonemes",
            "pos",
            "rel",
            "id",
            "head_id",
            "emphasis",
            "asr_pause",
            "prosody",
        ),
    ):
        if not overwrite and file_path.exists():
            raise RuntimeError(f"Sega {str(file_path)} is exists!")

        seqs = {}
        for name in fields:
            seqs[name] = self.get_tier(name, relative=with_audio)

        meta = self.meta
        meta["lang"] = self.sent.lang
        meta["sent_position"] = self.sent.position.name

        tg = tgio.Textgrid()
        for name, field in seqs.items():
            tier = tgio.IntervalTier(name, field, 0, maxT=self.audio_chunk.duration)
            tg.addTier(tier)

        if with_audio:
            if self.audio_chunk.empty:
                self.audio_chunk.load()

            new_audio_path = file_path.with_suffix(f".{audio_format}")
            self.audio_chunk.save(new_audio_path, overwrite=overwrite)
            self.audio_chunk.file_path = new_audio_path

            meta["audio_path"] = self.audio_chunk.file_path.name
            meta["audio_chunk"] = (0.0, self.audio_chunk.duration)  # type: ignore
        else:
            audio_path = Path(self.audio_chunk.file_path)
            if audio_path.parent == self.sega_path.parent:
                audio_path = audio_path.name
            else:
                audio_path = audio_path.as_posix()

            meta["audio_path"] = audio_path
            meta["audio_chunk"] = (self.audio_chunk.begin, self.audio_chunk.end)  # type: ignore

        tier = self.get_tier_for_meta(meta, relative=with_audio)
        tg.addTier(tier)

        tg.save(file_path.as_posix())

    @staticmethod
    @check_path(assert_file_exists=True)
    def load_audio_chunk(
        file_path: tp_PATH,
        meta: tp.Optional[tp.Dict[str, tp.Any]] = None,
        audio_path: tp.Optional[tp_PATH] = None,
    ):
        if meta:
            # TODO: support legacy models
            if "wav_path" in meta:
                audio_path_from_meta = Path(meta["wav_path"])
            else:
                audio_path_from_meta = Path(meta["audio_path"])
        else:
            meta = {}
            audio_path_from_meta = file_path.with_suffix(".wav")

        if not audio_path_from_meta.exists():
            audio_path_from_meta = file_path.with_name(audio_path_from_meta.name)

        if not audio_path_from_meta.exists():
            for ext in AudioFormat.as_extensions():
                if audio_path_from_meta.with_suffix(ext).exists():
                    audio_path_from_meta = audio_path_from_meta.with_suffix(ext)
                    break

        audio_path = audio_path_from_meta if audio_path is None else audio_path
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file {audio_path.as_posix()} not found!")

        if "audio_chunk" in meta:
            audio_chunk = AudioChunk(
                audio_path,
                begin=meta["audio_chunk"][0],
                end=meta["audio_chunk"][1],
            )
        elif "orig_audio_chunk" in meta:
            audio_chunk = AudioChunk(
                audio_path,
                begin=meta["orig_audio_chunk"][0],
                end=meta["orig_audio_chunk"][1],
            )
        else:
            audio_chunk = AudioChunk(audio_path)

        return audio_chunk

    @staticmethod
    @check_path(assert_file_exists=True)
    def load_meta(file_path: tp_PATH) -> tp.Dict[str, tp.Any]:
        string = file_path.read_text(encoding="utf-8")
        return _extract_meta(string)

    def get_timestamps(
        self, relative: bool = False
    ) -> tp.Tuple[Timestamps, tp.List[Timestamps]]:
        assert self.ts_by_words and self.ts_by_phonemes

        ts_by_words = copy(self.ts_by_words)
        ts_by_phonemes = copy(self.ts_by_phonemes)

        if relative:
            if ts_by_words is not None:
                ts_by_words -= self.audio_chunk.begin
            if ts_by_phonemes is not None:
                ts_by_phonemes = [ts - self.audio_chunk.begin for ts in ts_by_phonemes]

        return ts_by_words, ts_by_phonemes

    """
    def split_into_syntagmas(self, min_offset: float = 0.1) -> tp.List["AudioSeg"]:
        syntagmas_timestamps = self.get_tier("syntagmas")[1:-1]
        part_idxs = {
            tuple(x)
            for x in itertools.chain.from_iterable(
                more_itertools.partitions(range(len(syntagmas_timestamps)))
            )
        }

        splitted_syntagmas = []
        for syntagma_position in part_idxs:

            if len(syntagma_position) == 2:
                idx_start, idx_end = syntagma_position
            elif len(syntagma_position) == 1:
                idx_start, idx_end = syntagma_position[0], syntagma_position[0]
            else:
                raise RuntimeError(
                    f"incorrect syntagmas timestamps in {self.sega_path} in splitting function."
                )

            syntagma_start = (
                syntagmas_timestamps[idx_start - 1][1] if idx_start != 0 else self.ts_bos
            )
            syntagma_end = (
                syntagmas_timestamps[idx_end + 1][0]
                if idx_end + 1 != len(syntagmas_timestamps)
                else self.ts_eos
            )

            if idx_start == 0:
                _bos_ts = self.ts_bos
                is_left_valid = True
            else:
                _bos_ts = syntagmas_timestamps[idx_start - 1][1]
                is_left_valid = syntagmas_timestamps[idx_start][0] - _bos_ts >= min_offset

            if idx_end == len(syntagmas_timestamps) - 1:
                _eos_ts = self.ts_eos
                is_right_valid = True
            else:
                _eos_ts = syntagmas_timestamps[idx_end + 1][0]
                is_right_valid = _eos_ts - syntagmas_timestamps[idx_end][1] >= min_offset

            if not (is_right_valid and is_left_valid):
                continue

            assert self.sega_path
            new_sega = self.load(
                self.sega_path,
                with_audio=True,
                crop_begin=syntagma_start,
                crop_end=syntagma_end,
            )
            new_sega.meta["split_idxs"] = "-".join([str(x) for x in syntagma_position])
            splitted_syntagmas.append(new_sega)

        return splitted_syntagmas
    """


class SentencePreview:
    __slots__ = ("text_orig", "text", "words", "phonemes", "lang", "position")

    def __init__(self):
        self.text_orig = None
        self.text = None
        self.words = None
        self.phonemes = None
        self.lang = None
        self.position = None


@dataclass
class AudioSegPreview:
    audio_chunk: AudioChunk = None
    sent: SentencePreview = None
    ts_by_words: Timestamps = None
    ts_by_phonemes: Timestamps = None
    ts_bos: float = 0.0
    ts_eos: float = 0.0
    meta: tp.Dict[str, tp.Any] = None
    sega_path: tp_PATH = None

    def __post_init__(self):
        self.sent = SentencePreview()
        self.meta = {}

    @property
    def duration(self) -> float:
        return self.ts_eos - self.ts_bos

    @staticmethod
    @check_path(assert_file_exists=True)
    def load(file_path: tp_PATH) -> "AudioSegPreview":
        raw_data = file_path.read_text(encoding="utf-8")

        tiers = {}
        for data in raw_data.split("IntervalTier"):
            for tier_name in ["orig", "text", "phonemes"]:
                if f'\n"{tier_name}"' in data:
                    tokens = tiers.setdefault(tier_name, [])
                    lines = data.split("\n")[5:-1]
                    for i in range(0, len(lines), 3):
                        label = lines[i + 2].replace('"', "")
                        if label:
                            tokens.append((float(lines[i]), float(lines[i + 1]), label))

                    break

        tiers, ts_bos, ts_eos = _remove_service_tokens(tiers)

        sega = AudioSegPreview()

        if "orig" in tiers:
            sega.sent.text_orig = tiers["orig"][0][2]

        if "text" in tiers:
            sega.sent.words = tuple(item[2] for item in tiers["text"])
            sega.ts_by_words = Timestamps.from_list(
                [(item[0], item[1]) for item in tiers["text"]]
            )
            sega.sent.text = " ".join(sega.sent.words)

        if "phonemes" in tiers:
            sega.sent.phonemes = tuple(item[2] for item in tiers["phonemes"])
            sega.ts_by_phonemes = Timestamps.from_list(
                [(item[0], item[1]) for item in tiers["phonemes"]]
            )

        """
        phonemes = []
        ts_by_ph = []
        ph_word = []
        word_idx = 0
        if "phonemes" in tiers:
            for begin, end, label in tiers["phonemes"]:
                if "|" in label:
                    label = tuple(label.split("|"))
                ph_word.append(label)
                ts_by_ph.append((begin, end))
                if _fp_eq(end, tiers["text"][word_idx][1]):
                    phonemes.append(tuple(ph_word))
                    ph_word = []
                    word_idx += 1

        sega.sent["phonemes"] = phonemes
        sega.ts_by_phonemes = Timestamps.from_list(ts_by_ph)
        """

        if '\n"meta"' in raw_data:
            meta = _extract_meta(raw_data)
            sega.sent.lang = meta.get("lang")
            sega.sent.position = Position[meta["sent_position"]]
        else:
            meta = {}

        sega.audio_chunk = AudioSeg.load_audio_chunk(file_path, meta)

        sega.ts_bos = sega.audio_chunk.begin if ts_bos is None else ts_bos
        sega.ts_eos = sega.audio_chunk.end if ts_eos is None else ts_eos

        sega.meta = meta
        sega.sega_path = file_path
        return sega


if __name__ == "__main__":
    from speechflow.utils.fs import get_root_dir
    from speechflow.utils.profiler import Profiler

    _root = get_root_dir()
    _fpath = list(
        (_root / "examples/simple_datasets/speech/SEGS").rglob("*.TextGridStage2")
    )

    with Profiler(name="full sega load", format=Profiler.Format.ms):
        for _ in range(10):
            for _path in _fpath:
                _sega = AudioSeg.load(_path)

    with Profiler(name="preview sega load", format=Profiler.Format.ms):
        for _ in range(10):
            for _path in _fpath:
                _sega = AudioSegPreview.load(_path)
