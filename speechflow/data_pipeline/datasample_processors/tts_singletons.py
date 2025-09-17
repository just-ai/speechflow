import json
import zlib
import pickle
import typing as tp
import hashlib
import logging

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tqdm import tqdm

from speechflow.data_pipeline.core import Dataset, Singleton
from speechflow.data_pipeline.datasample_processors.data_types import TTSDataSample
from speechflow.io import AudioSeg
from speechflow.logging import trace
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.dictutils import find_field

try:
    from annoy import AnnoyIndex
except ImportError as e:
    print(f"Annoy import failed: {e}")

__all__ = [
    "SpeakerIDSetter",
    "StatisticsRange",
    "MeanBioEmbeddings",
    "DatasetStatistics",
    "PhonemeStatistics",
]

tp_PATH = tp.Union[str, Path]

LOGGER = logging.getLogger("root")


class SpeakerIDSetter(metaclass=Singleton):
    def __init__(
        self,
        data_subset_name: str,
        target_speakers: tp.Optional[tp.FrozenSet[str]] = None,
        reserve_speakers: tp.Optional[tp.FrozenSet[str]] = None,
        speakers_filter: tp.Optional[tp.FrozenSet[str]] = None,
        target_langs: tp.Optional[tp.FrozenSet[str]] = None,
        langs_filter: tp.Optional[tp.FrozenSet[str]] = None,
        speaker_id_map: tp.Optional[tp.FrozenSet[str]] = None,
        lang_id_map: tp.Optional[tp.FrozenSet[str]] = None,
        min_samples: tp.Optional[int] = None,
        min_duration: tp.Optional[float] = None,  # in hours
        resume_from_checkpoint: tp.Optional[Path] = None,
        remove_unknown_speakers: bool = False,
        mean_embeddings_file: tp.Optional[tp.Union[str, Path]] = None,
        **kwargs,
    ):
        self.data_subset_name = data_subset_name
        self.target_speakers = (
            [target_speakers] if isinstance(target_speakers, str) else target_speakers
        )
        self.reserve_speakers = (
            [reserve_speakers] if isinstance(reserve_speakers, str) else reserve_speakers
        )
        self.speakers_filter = (
            [speakers_filter] if isinstance(speakers_filter, str) else speakers_filter
        )
        self.target_langs = (
            [target_langs] if isinstance(target_langs, str) else target_langs
        )
        self.langs_filter = (
            [langs_filter] if isinstance(langs_filter, str) else langs_filter
        )

        if resume_from_checkpoint:
            checkpoint = ExperimentSaver.load_checkpoint(resume_from_checkpoint)
            speaker_id_map = [
                f"{key}:{value}"
                for key, value in checkpoint.get("speaker_id_map", {}).items()
            ]
            lang_id_map = [
                f"{key}:{value}"
                for key, value in checkpoint.get("lang_id_map", {}).items()
            ]

        self.speaker2id = {}
        self.id2speaker = {}
        if speaker_id_map:
            self.speaker2id, self.id2speaker = self._map_to_id(speaker_id_map)
            # LOGGER.info(
            #     f"Target speakers ({len(self.speaker2id)}): {', '.join(self.speaker2id.keys())}"
            # )

        self.lang2id = {}
        self.id2lang = {}
        if lang_id_map:
            self.lang2id, self.id2lang = self._map_to_id(lang_id_map)
            # LOGGER.info(
            #     f"Target languages ({len(self.lang2id)}): {', '.join(self.lang2id.keys())}"
            # )

        self.min_samples = min_samples
        self.min_duration = min_duration
        self.remove_unknown_speakers = remove_unknown_speakers
        self.unknown_speakers = set()

        if mean_embeddings_file is not None:
            self.mean_embeddings = json.loads(
                Path(mean_embeddings_file).read_text(encoding="utf-8")
            )
            speaker_id_map_bio = {
                name: idx for idx, name in enumerate(self.mean_embeddings.keys())
            }
            self.speaker2id_bio, self.id2speaker_bio = self._map_to_id(speaker_id_map_bio)
            self.similar_speaker_map = {}
        else:
            self.mean_embeddings = None
            self.speaker2id_bio = None
            self.id2speaker_bio = None
            self.similar_speaker_map = None

    @staticmethod
    def _map_to_id(id_map: tp.Union[tp.List, tp.Dict]):
        name2id = {}
        id2name = {}
        if isinstance(id_map, tp.Mapping):
            for name, idx in id_map.items():
                name2id[name] = int(idx)
                id2name[int(idx)] = name
        else:
            for item in id_map:
                name, idx = item.split(":")
                name2id[name] = int(idx)
                id2name[int(idx)] = name
        return name2id, id2name

    @staticmethod
    def _get_speaker_overall_duration(data: Dataset):
        with data.readonly():
            audio_duration: tp.Dict[str, tp.Any] = {}
            for ds in data:
                duration = ds.audio_chunk.duration
                if duration:
                    seconds = audio_duration.setdefault(ds.speaker_name, [0])
                seconds[0] += duration

        LOGGER.info("Dataset durations by speakers:")
        for key, _ in sorted(audio_duration.items()):
            audio_duration[key] = audio_duration.pop(key)[0] / 3600
            LOGGER.info(f"\t{key}: {round(audio_duration[key], 3)}h")

        return audio_duration

    @staticmethod
    def _build_mean_embed_index(mean_embeddings):
        all_embeddings = np.stack(
            [np.asarray(emb, dtype=np.float32) for emb in mean_embeddings.values()]
        )
        mean_embed_index = AnnoyIndex(all_embeddings.shape[1], "euclidean")
        for i, v in enumerate(all_embeddings):
            mean_embed_index.add_item(i, v)

        mean_embed_index.build(min(500, all_embeddings.shape[0]))
        return mean_embed_index

    def _preprocessing(self, data: Dataset) -> Dataset:
        if self.langs_filter is not None:
            LOGGER.info(trace(self, f"Apply languages filter: {self.langs_filter}"))
            data.filter(lambda ds: ds.lang in self.langs_filter)

        if not self.lang2id and not self.id2lang:
            langs = sorted(list({ds.lang for ds in data if ds.lang}))
            if langs:
                self.lang2id = {name: idx for idx, name in enumerate(langs)}
                self.id2lang = {idx: name for name, idx in self.lang2id.items()}
                LOGGER.info(
                    trace(
                        self,
                        f"All languages ({len(self.lang2id)}): {', '.join(self.lang2id.keys())}",
                    )
                )
        else:
            data.filter(lambda ds: ds.lang in self.lang2id)

        if not self.speaker2id and not self.id2speaker:
            speaker_names = {ds.speaker_name for ds in data if ds.speaker_name}

            if self.speakers_filter is None:
                if self.min_samples is not None:
                    speaker_num_samples = Counter(ds.speaker_name for ds in data)
                    skip_by_samples = {
                        s for s, n in speaker_num_samples.items() if n < self.min_samples
                    }
                else:
                    skip_by_samples = set()

                if self.min_duration is not None:
                    speakers_audio_duration = self._get_speaker_overall_duration(data)
                    skip_by_durations = {
                        s
                        for s in speaker_names
                        if speakers_audio_duration[s] <= self.min_duration
                    }
                else:
                    skip_by_durations = set()

                speakers_to_skip = set()
                speakers_to_skip.update(skip_by_samples)
                speakers_to_skip.update(skip_by_durations)
                self.speakers_to_skip = speakers_to_skip

                speaker_names = [sn for sn in speaker_names if sn not in speakers_to_skip]
            else:
                LOGGER.info(trace(self, f"Apply speakers filter: {self.speakers_filter}"))
                speaker_names = list(self.speakers_filter)

            if self.reserve_speakers is not None:
                speaker_names += list(self.reserve_speakers)

            speaker_names = sorted(list(set(speaker_names)))
            data.filter(lambda ds: ds.speaker_name in speaker_names)

            self.speaker2id = {name: idx for idx, name in enumerate(speaker_names)}
            self.id2speaker = {idx: name for name, idx in self.speaker2id.items()}
            LOGGER.info(
                trace(
                    self,
                    f"All speakers ({len(self.speaker2id)}): {', '.join(self.speaker2id.keys())}",
                )
            )
        elif self.remove_unknown_speakers:
            LOGGER.info(trace(self, "Remove unknown speakers"))
            data.filter(lambda ds: ds.speaker_name in self.speaker2id)

        if self.target_langs is not None:
            data.filter(lambda ds: ds.lang in self.target_langs)
            LOGGER.info(
                trace(
                    self,
                    f"Target langs ({len(self.target_langs)}): {', '.join(self.target_langs)}",
                )
            )

        if self.target_speakers is not None:
            data.filter(lambda ds: ds.speaker_name in self.target_speakers)
            LOGGER.info(
                trace(
                    self,
                    f"Target speakers ({len(self.target_speakers)}): {', '.join(self.target_speakers)}",
                )
            )

        total_duration = sum([ds.audio_chunk.duration for ds in data]) / 3600
        LOGGER.info(
            trace(
                self,
                f"Total dataset duration {round(total_duration, 3)} hours ({len(data)} samples)",
            )
        )
        return data

    def __hash__(self):
        _names = sorted(list(self.speaker2id.keys()))
        _hash = hashlib.md5("_".join(_names).encode("utf-8"))
        return int(_hash.hexdigest(), 16) & 0xFFFFFFFF

    def __call__(self, data: Dataset) -> Dataset:
        with data.readonly():
            data = self._preprocessing(data)

        mean_embed_index = None
        if self.mean_embeddings is not None:
            mean_embed_index = self._build_mean_embed_index(self.mean_embeddings)

        for ds in tqdm(data, "Set language and speaker id"):
            self.set_lang_id(self.set_speaker_id(ds, mean_embed_index))

        data.__class__.__hash__ = self.__hash__
        return data

    @property
    def n_langs(self):
        if self.lang2id:
            return len(self.lang2id)
        else:
            raise AttributeError("Language information has not been calculated yet.")

    @property
    def n_speakers(self):
        if self.speaker2id:
            return len(self.speaker2id)
        else:
            raise AttributeError("Speaker information has not been calculated yet.")

    def set_speaker_id(self, ds: TTSDataSample, mean_embed_index=None):
        if ds.speaker_name is None:
            raise ValueError("'speaker_name' attribute is not set.")

        if ds.speaker_name not in self.speaker2id:
            if ds.speaker_name not in self.unknown_speakers:
                self.unknown_speakers.add(ds.speaker_name)
                LOGGER.warning(
                    trace(self, f"unknown speaker detected: '{ds.speaker_name}'")
                )

            if self.speaker2id_bio is not None and ds.speaker_name in self.speaker2id_bio:
                if ds.speaker_name in self.similar_speaker_map:
                    ds.speaker_id = self.speaker2id[
                        self.similar_speaker_map[ds.speaker_name]
                    ]
                    return ds

                speaker_id = self.speaker2id_bio[ds.speaker_name]
                bio_embedding = mean_embed_index.get_item_vector(speaker_id)
                sorted_idx = mean_embed_index.get_nns_by_vector(
                    bio_embedding, min(len(self.speaker2id_bio), 10)
                )
                # sorted_idx = cdist(self.mean_embeddings, np.atleast_2d(bio_embedding)).reshape(-1).argsort()
                speaker_name = ds.speaker_name
                i = 1
                while (speaker_name not in self.speaker2id) and (i < len(sorted_idx)):
                    speaker_name = self.id2speaker_bio[sorted_idx[i]]
                    i += 1

                    if len(sorted_idx) <= i < len(self.speaker2id_bio):
                        sorted_idx = mean_embed_index.get_nns_by_vector(
                            bio_embedding, min(len(self.speaker2id_bio), i * 10)
                        )

                if speaker_name in self.speaker2id:
                    ds.speaker_id = self.speaker2id[speaker_name]
                    self.similar_speaker_map[ds.speaker_name] = speaker_name
                    LOGGER.warning(
                        trace(
                            self,
                            f"found similar speaker '{speaker_name}' "
                            f"for unknown speaker '{ds.speaker_name}'",
                        )
                    )
                    return ds
        else:
            ds.speaker_id = self.speaker2id.get(ds.speaker_name)
            return ds

        ds.speaker_id = 0
        return ds

    def set_lang_id(self, ds: TTSDataSample):
        if ds.lang is not None:
            ds.lang_id = self.lang2id[ds.lang]
        return ds

    @staticmethod
    def aggregate(a: "SpeakerIDSetter", b: "SpeakerIDSetter") -> "SpeakerIDSetter":
        c = deepcopy(a)
        n_speakers = a.n_speakers
        for name, idx in b.speaker2id.items():
            if name not in c.speaker2id:
                c.speaker2id[name] = n_speakers
                c.id2speaker[n_speakers] = name
                n_speakers += 1

        return c


class StatisticsRange(metaclass=Singleton):
    def __init__(
        self, data_subset_name: str, statistics_file: tp_PATH = "ranges.json", **kwargs
    ):
        if isinstance(statistics_file, tp.Mapping):  # evaluation hack
            self.statistics: tp.Dict[str, tp.Any] = statistics_file  # type: ignore
            return

        self.statistics_file = Path(statistics_file)
        if self.statistics_file.is_file():
            self.statistics: tp.Dict[str, tp.Any] = json.loads(self.statistics_file.read_text(encoding="utf-8"))  # type: ignore
        else:
            raise ValueError(
                f"{self.statistics_file.as_posix()} not found! First do execute dump.py"
            )

    def __call__(self, data: Dataset) -> Dataset:
        return data

    def get_keys(self) -> tp.List[str]:
        return list(self.statistics.keys())

    def get_range(self, attribute: str, speaker_name: str) -> tp.Tuple[float, float]:
        values = self.statistics[attribute][speaker_name]
        a_min, a_max = values["min"], values["max"]
        if attribute in ["energy", "pitch"]:
            a_min = 0.0
        return a_min, a_max

    def get_stat(self, attribute: str, speaker_name: str) -> tp.Tuple[float, float]:
        values = self.statistics[attribute][speaker_name]
        return values["mean"], values["var"]

    @staticmethod
    def aggregate(a: "StatisticsRange", b: "StatisticsRange") -> "StatisticsRange":
        c = deepcopy(a)
        c.statistics.update(b.statistics)
        return c


class MeanBioEmbeddings(metaclass=Singleton):
    def __init__(
        self,
        data_subset_name: str,
        mean_embeddings_file: tp.Union[
            tp_PATH, tp.MutableMapping
        ] = "mean_bio_embeddings.json",
        **kwargs,
    ):
        if isinstance(mean_embeddings_file, tp.MutableMapping):  # evaluation hack
            self.data: tp.Dict[str, tp.Any] = mean_embeddings_file  # type: ignore
        else:
            self.mean_embeddings_file = Path(mean_embeddings_file)
            if self.mean_embeddings_file.is_file():
                self.data: tp.Dict[str, tp.Any] = json.loads(
                    self.mean_embeddings_file.read_text(encoding="utf-8")
                )  # type: ignore
            else:
                raise ValueError(
                    f"{self.mean_embeddings_file.as_posix()} not found! First do execute dump.py"
                )

        self.data = {
            s: np.asarray([emb], dtype=np.float32) for s, emb in self.data.items()
        }

    def __call__(self, data: Dataset) -> Dataset:
        return data

    def get_speakers(self) -> tp.List[str]:
        return list(self.data.keys())

    def get_embedding(self, speaker_name: str) -> npt.NDArray:
        return self.data[speaker_name][0]

    @staticmethod
    def aggregate(a: "MeanBioEmbeddings", b: "MeanBioEmbeddings") -> "MeanBioEmbeddings":
        c = deepcopy(a)
        c.data.update(b.data)
        return c


class DatasetStatistics(metaclass=Singleton):
    def __init__(
        self,
        data_subset_name: str,
        dump_path: tp.Optional[tp_PATH] = None,
        add_dataset_statistics: bool = True,
        add_segmentations: bool = True,
        add_speaker_emb: bool = True,
        **kwargs,
    ):
        self.data_subset_name = data_subset_name
        self.dump_path = Path(dump_path) if dump_path else None
        self.add_dataset_statistics = add_dataset_statistics
        self.add_segmentations = add_segmentations
        self.add_speaker_emb = add_speaker_emb
        self.cfg_data = kwargs.get("cfg_data", {})
        self.pause_step = find_field(self.cfg_data, "step")
        self.hop_len = find_field(self.cfg_data, "hop_len")
        self.hash = None
        self.transcription_length: tp.Dict[str, tp.List[int]] = defaultdict(list)
        self.wave_duration: tp.Dict[str, tp.List[float]] = defaultdict(list)
        self.max_transcription_length: int = 0
        self.max_audio_duration: float = 0.0
        self.segmentations: tp.List[tp.Tuple[str, bytes]] = []
        self.speaker_emb: tp.Dict[str, tp.List[tp.Any]] = defaultdict(list)
        self.cache_folder = self._get_cache_folder(self.dump_path)

    @staticmethod
    def _get_cache_folder(dump: Path) -> tp.Optional[Path]:
        if dump:
            cache_folder = dump / "cache"
            cache_folder.mkdir(parents=True, exist_ok=True)
            return cache_folder
        else:
            return None

    def get_cache_fpath(self) -> tp.Optional[Path]:
        # build hash for speakers and
        # load precomputed attributes if cache exists
        if self.cache_folder is None:
            return

        d = (
            f"{int(self.add_dataset_statistics)}"
            f"{int(self.add_segmentations)}"
            f"{int(self.add_speaker_emb)}"
        )
        return self.cache_folder / f"DatasetStatistics_{self.hash}{d}.pkl"

    def _add_dataset_stat(self, data: Dataset):
        from speechflow.data_pipeline.datasample_processors.tts_processors import (
            add_pauses_from_timestamps,
        )

        for ds in tqdm(data, "Counting statistics over dataset"):
            try:
                ds = add_pauses_from_timestamps(ds.copy(), step=self.pause_step)
                transcription = ds.sent["phonemes"]

                self.transcription_length[ds.speaker_name].append(len(transcription))  # type: ignore
                self.wave_duration[ds.speaker_name].append(ds.audio_chunk.duration)  # type: ignore
            except Exception as e:
                LOGGER.error(trace(self, e, message=ds.file_path.as_posix()))

        for var in [
            self.transcription_length,
            self.wave_duration,
        ]:
            for name, field in var.items():  # type: ignore
                if isinstance(field, tp.Mapping):
                    for ph, ph_lens in field.items():
                        var[name][ph] = np.asarray(ph_lens, dtype=np.float32)  # type: ignore
                else:
                    var[name] = np.asarray(field, dtype=np.float32)  # type: ignore

        if self.transcription_length:
            self.max_transcription_length = max(
                max(v) for v in self.transcription_length.values()
            )
            self.max_audio_duration = max(max(v) for v in self.wave_duration.values())

    def _add_segmentations(self, data: Dataset):
        for ds in tqdm(data, "Loading segmentations"):
            path = Path(ds.file_path)
            assert "TextGrid" in path.suffix
            sega = zlib.compress(path.read_bytes())
            self.segmentations.append((path.as_posix(), sega))

    def _add_speaker_emb(self):
        dump_files = self.dump_path.rglob("*.pkl")

        for file_path in tqdm(dump_files, "Loading speaker embeddings"):
            if "cache" in file_path.as_posix():
                continue

            try:
                item = []
                dump_data: tp.Dict[str, tp.Any] = pickle.loads(file_path.read_bytes())
                file_path = dump_data["fields"].get("file_path")
                if file_path is None or not Path(file_path).exists():
                    continue
                item.append(file_path.as_posix())
                speaker_name = dump_data["fields"].get("speaker_name")
                bio_proc = dump_data["functions"].get("VoiceBiometricProcessor")
                if bio_proc is not None:
                    item.append(bio_proc.get("speaker_emb"))
                else:
                    continue
                seg_meta = AudioSeg.load_meta(file_path)
                item.append(seg_meta.get("orig_audio_path"))
                item.append(np.asarray(seg_meta.get("orig_audio_chunk")))
                self.speaker_emb[speaker_name].append(item)
            except Exception as e:
                LOGGER.error(trace(self, e, message=file_path.as_posix()))

        for name, field in self.speaker_emb.items():
            self.speaker_emb[name] = np.asarray(field, dtype="object")  # type: ignore

    def __call__(self, data: Dataset) -> Dataset:
        if self.hash:
            return data
        else:
            self.hash = hash(data)

        cache_fpath = self.get_cache_fpath()
        if cache_fpath and cache_fpath.exists():
            LOGGER.info(f"Load DatasetStatistics cache from {cache_fpath.name}")
            self.__dict__ = pickle.loads(cache_fpath.read_bytes())
            return data

        with data.readonly():
            if self.add_dataset_statistics:
                self._add_dataset_stat(data)

            if self.add_segmentations:
                self._add_segmentations(data)

            if self.add_speaker_emb and self.dump_path:
                self._add_speaker_emb()

        self.pickle()

        cache_fpath = self.get_cache_fpath()
        if cache_fpath:
            cache_fpath.write_bytes(pickle.dumps(self.__dict__))

        return data

    def pickle(self):
        for key, value in self.__dict__.items():
            if not key.startswith("_") and isinstance(value, tp.Iterable):
                setattr(self, key, pickle.dumps(value))

    def unpickle(self):
        for key, value in self.__dict__.items():
            if isinstance(value, bytes):
                try:
                    setattr(self, key, pickle.loads(value))
                except Exception:
                    pass

    @staticmethod
    def aggregate(a: "DatasetStatistics", b: "DatasetStatistics") -> "DatasetStatistics":
        c = DatasetStatistics(a.data_subset_name)
        c.max_transcription_length = max(
            a.max_transcription_length, b.max_transcription_length
        )
        c.max_audio_duration = max(a.max_audio_duration, b.max_audio_duration)
        return c


class PhonemeStatistics(metaclass=Singleton):
    def __init__(
        self,
        data_subset_name: str,
        dump: tp.Optional[tp_PATH] = None,
        quantile: float = 0.95,
        **kwargs,
    ):
        self._data_subset_name = data_subset_name
        self._dump = dump
        self._quantile = quantile
        self.phonemes_statistics: tp.Dict[
            str, tp.Dict[str, (float, float)]
        ] = defaultdict(dict)

    def __call__(self, data: Dataset) -> Dataset:
        dataset_stat = DatasetStatistics(
            self._data_subset_name, self._dump, True, False, False
        )
        dataset_stat(data)

        for speaker_name, phonemes in dataset_stat.phonemes_statistics.items():  # type: ignore
            for ph, ph_lens in phonemes.items():
                val_min = np.quantile(ph_lens, 1 - self._quantile)
                val_max = np.quantile(ph_lens, self._quantile)
                self.phonemes_statistics[speaker_name].setdefault(ph, [val_min, val_max])

        return data
