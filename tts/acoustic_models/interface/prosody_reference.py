import random
import typing as tp

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy.typing as npt

from speechflow.data_pipeline.datasample_processors.data_types import (
    AudioDataSample,
    SpectrogramDataSample,
)
from speechflow.io import AudioChunk
from speechflow.utils.seed import set_random_seed

__all__ = ["ProsodyReference"]

REFERENECE_TYPE = tp.Union[
    int, str, Path, tp.Tuple[tp.Union[str, int], int], tp.Tuple[npt.NDArray, int]
]


@dataclass
class ProsodyReference:
    lang: tp.Optional[str] = None
    speaker_name: tp.Optional[str] = None
    speaker_id: tp.Optional[int] = None

    speaker_emb: tp.Optional[npt.NDArray] = None
    speaker_emb_index: tp.Optional[tp.Union[int, tp.Tuple[str, int]]] = None
    speaker_emb_mean: tp.Optional[npt.NDArray] = None
    speaker_audio_path: tp.Optional[tp.Union[str, Path]] = None
    speaker_audio_chunk: tp.Optional[AudioChunk] = None
    speaker_spectrogram: tp.Optional[npt.NDArray] = None
    speaker_ssl_feat: tp.Optional[npt.NDArray] = None

    style_emb: tp.Optional[npt.NDArray] = None
    style_emb_index: tp.Optional[tp.Union[int, tp.Tuple[str, int]]] = None
    style_emb_mean: tp.Optional[npt.NDArray] = None
    style_audio_path: tp.Optional[tp.Union[str, Path]] = None
    style_audio_chunk: tp.Optional[AudioChunk] = None
    style_spectrogram: tp.Optional[npt.NDArray] = None
    style_ssl_feat: tp.Optional[npt.NDArray] = None

    model_feats: tp.Optional[tp.Dict[str, torch.Tensor]] = None
    meta: tp.Optional[tp.Dict] = None

    def __post_init__(self):
        if self.model_feats is None:
            self.model_feats = {}
        if self.meta is None:
            self.meta = {}

    def __eq__(self, other: "ProsodyReference"):
        return (
            self.speaker_name == other.speaker_name
            and self.model_feats == other.model_feats
        )

    def speaker_reference_is_empty(self):
        return self.speaker_emb_index is None and self.speaker_audio_chunk is None

    def style_reference_is_empty(self):
        return self.style_emb_index is None and self.style_audio_chunk is None

    def is_empty(self):
        return self.speaker_reference_is_empty() and self.style_reference_is_empty()

    def _set_reference(self, ref_type: str, ref: tp.Optional[REFERENECE_TYPE]):
        if isinstance(ref, int):
            setattr(self, f"{ref_type}_emb_index", ref)
        elif isinstance(ref, str) and "/" not in ref and "\\" not in ref:
            setattr(self, f"{ref_type}_emb_index", (ref, -1))
        elif isinstance(ref, (tuple, list)) and isinstance(ref[0], (int, str)):
            setattr(self, f"{ref_type}_emb_index", tuple(ref))
        elif isinstance(ref, tp.Union[str, Path]):
            setattr(self, f"{ref_type}_audio_path", Path(ref))
        elif isinstance(ref, (tuple, list)):
            waveform, sr = ref
            setattr(self, f"{ref_type}_audio_chunk", AudioChunk(data=waveform, sr=sr))
        else:
            raise AttributeError(
                "Incorrect format of reference audio."
                "Pass path-like or tuple(np.ndarray, sr)"
            )

    def _sampling_reference(
        self,
        ref_type: str,
        all_speakers: tp.List[str],
        bio_embeddings: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        if bio_embeddings is None:
            bio_embeddings = {name: [0] for name in all_speakers}

        if isinstance(getattr(self, f"{ref_type}_emb_index"), int):
            if getattr(self, f"{ref_type}_emb_index") == -1:
                emb_index = random.randint(0, len(bio_embeddings[self.speaker_name]) - 1)
            else:
                emb_index = getattr(self, f"{ref_type}_emb_index")
            setattr(self, f"{ref_type}_emb_index", (self.speaker_name, emb_index))

        elif isinstance(getattr(self, f"{ref_type}_emb_index"), tuple):
            sp_name, emb_index = getattr(self, f"{ref_type}_emb_index")
            if isinstance(sp_name, (str, int)):
                if sp_name == -1:
                    sp_name = random.choice(list(bio_embeddings.keys()))
                if sp_name not in all_speakers:
                    raise ValueError(f"Speaker {sp_name} not found in current TTS model!")
                if emb_index == -1:
                    emb_index = random.randint(0, len(bio_embeddings[sp_name]) - 1)
                setattr(self, f"{ref_type}_emb_index", (sp_name, emb_index))

        if isinstance(getattr(self, f"{ref_type}_audio_path"), str):
            setattr(
                self,
                f"{ref_type}_audio_path",
                Path(getattr(self, f"{ref_type}_audio_path")),
            )

        audio_path = getattr(self, f"{ref_type}_audio_path")
        if audio_path is not None:
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Reference file {audio_path.as_posix()} not found!"
                )
            else:
                setattr(
                    self,
                    f"{ref_type}_audio_chunk",
                    AudioChunk(file_path=audio_path).load(),
                )

    def set_speaker_name(self, speaker_name: str):
        assert isinstance(speaker_name, str)
        self.speaker_name = speaker_name

    def set_speaker_reference(self, ref: REFERENECE_TYPE):
        self._set_reference("speaker", ref)

    def set_style_reference(self, ref: REFERENECE_TYPE):
        self._set_reference("style", ref)

    def set_speaker_id(
        self, speaker_id_map: tp.Dict, mean_bio_embeddings: tp.Optional[tp.Dict] = None
    ):
        if self.speaker_name is not None:
            self.speaker_id = speaker_id_map.get(self.speaker_name, 0)

            if mean_bio_embeddings is not None:
                self.speaker_emb_mean = mean_bio_embeddings[self.speaker_name]

    def set_bio_embedding(
        self, biometric_pipe: callable, bio_embeddings: tp.Optional[tp.Dict] = None
    ):
        for ref_type in ["speaker", "style"]:
            attr_name = f"{ref_type}_emb_index"
            if bio_embeddings is not None and getattr(self, attr_name) is not None:
                sp_name, emb_index = getattr(self, attr_name)
                emb_meta = bio_embeddings[sp_name][emb_index]
                bio_emb = emb_meta[0] if emb_meta.ndim == 2 else emb_meta
                setattr(self, f"{ref_type}_emb", bio_emb)

                try:
                    self.meta[f"{ref_type}_ref_path"] = Path(emb_meta[0]).with_suffix(
                        ".wav"
                    )
                    self.meta[
                        f"{ref_type}_ref_orig"
                    ] = f"{emb_meta[2]}|{emb_meta[3][0]}:{emb_meta[3][1]}"
                except Exception:
                    pass

                continue

            attr_name = f"{ref_type}_audio_chunk"
            if getattr(self, attr_name) is not None:
                audio_chunk = getattr(self, attr_name)
                ds = AudioDataSample(audio_chunk=audio_chunk)
                ds = biometric_pipe.preprocessing_datasample([ds])[0]
                if ds.additional_fields and ref_type == "style":
                    setattr(
                        self,
                        f"{ref_type}_emb",
                        list(ds.additional_fields.values())[0],
                    )
                else:
                    setattr(self, f"{ref_type}_emb", ds.speaker_emb)
            else:
                if ref_type == "speaker" and bio_embeddings is None:
                    raise RuntimeError("Please set speaker_audio_chunk.")

    def set_spectrogram_reference(self, spectrogram_pipe: callable):
        for ref_type in ["speaker", "style"]:
            attr_name = f"{ref_type}_audio_chunk"
            if getattr(self, attr_name) is not None:
                audio_chunk = getattr(self, attr_name)
                ds = AudioDataSample(audio_chunk=audio_chunk)
                ref_ds = spectrogram_pipe.preprocessing_datasample([ds.copy()])[0]
                setattr(self, f"{ref_type}_spectrogram", ref_ds.mel)
                setattr(self, f"{ref_type}_ssl_feat", ref_ds.ssl_feat)
                setattr(self, f"{ref_type}_transform_params", ref_ds.transform_params)
                continue

    def set_feats_from_model(self, model: callable):
        self.model_feats.update(
            model.get_speaker_embedding(
                self.speaker_id, self.speaker_emb, self.speaker_emb_mean
            )
        )

        self.model_feats.update(
            model.get_style_embedding(
                self.style_emb,
                self.style_spectrogram,
                self.style_ssl_feat,
            )
        )

        self.model_feats = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in self.model_feats.items()
        }

    def copy_speaker_name(self, item: "ProsodyReference"):
        self.speaker_name = item.speaker_name
        self.speaker_id = item.speaker_id

    def copy_speaker_reference(self, item: "ProsodyReference"):
        self.speaker_emb_index = item.speaker_emb_index
        self.speaker_emb = item.speaker_emb
        self.speaker_emb_mean = item.speaker_emb_mean
        self.speaker_audio_path = item.speaker_audio_path
        self.speaker_audio_chunk = item.speaker_audio_chunk
        self.speaker_spectrogram = item.speaker_spectrogram
        self.speaker_ssl_feat = item.speaker_ssl_feat

    def copy_style_reference(self, item: "ProsodyReference"):
        self.style_emb_index = item.style_emb_index
        self.style_emb = item.style_emb
        self.style_audio_path = item.style_audio_path
        self.style_audio_chunk = item.style_audio_chunk
        self.style_spectrogram = item.style_spectrogram
        self.style_ssl_feat = item.style_ssl_feat

    def sampling_reference(
        self,
        all_speakers: tp.List[str],
        bio_embeddings: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        self._sampling_reference("speaker", all_speakers, bio_embeddings)
        self._sampling_reference("style", all_speakers, bio_embeddings)

    def swap_reference(self, _from: str, _to: str):
        if getattr(self, f"{_to}_emb") is None:
            setattr(self, f"{_to}_emb", getattr(self, f"{_from}_emb"))
        if getattr(self, f"{_to}_emb_index") is None:
            setattr(self, f"{_to}_emb_index", getattr(self, f"{_from}_emb_index"))
        if getattr(self, f"{_to}_emb_mean") is None:
            setattr(self, f"{_to}_emb_mean", getattr(self, f"{_from}_emb_mean"))
        if getattr(self, f"{_to}_spectrogram") is None:
            setattr(self, f"{_to}_spectrogram", getattr(self, f"{_from}_spectrogram"))
        if getattr(self, f"{_to}_ssl_feat") is None:
            setattr(self, f"{_to}_ssl_feat", getattr(self, f"{_from}_ssl_feat"))

    def filling(self):
        assert self.speaker_name is not None

        if self.style_emb is None and self.speaker_emb is not None:
            self.style_emb = self.speaker_emb
        if self.speaker_emb is None and self.style_emb is not None:
            self.speaker_emb = self.style_emb

        if self.style_spectrogram is None and self.speaker_spectrogram is not None:
            self.style_spectrogram = self.speaker_spectrogram
        if self.speaker_spectrogram is None and self.style_spectrogram is not None:
            self.speaker_spectrogram = self.style_spectrogram

        if self.style_ssl_feat is None and self.speaker_ssl_feat is not None:
            self.style_ssl_feat = self.speaker_ssl_feat
        if self.speaker_ssl_feat is None and self.style_ssl_feat is not None:
            self.speaker_ssl_feat = self.style_ssl_feat

    def copy(self) -> "ProsodyReference":
        return deepcopy(self)

    def get_model_feat(self, name: str, shape=None, device=None) -> torch.Tensor:
        feat = self.model_feats[name]
        if not isinstance(feat, torch.Tensor):
            feat = torch.from_numpy(feat)

        if shape is not None:
            if len(shape) == 3:
                if feat.ndim == 2:
                    feat = feat.unsqueeze(1)
                # feat = feat.expand(shape[0], shape[1], -1)
            else:
                feat = feat.expand(shape[0], -1)

        if device is not None:
            feat = feat.to(device)

        return feat


class RefSpan:
    batch_idx: int
    start: int
    stop: int


@dataclass
class ComplexProsodyReference:
    refs: tp.Dict[tp.Union[str, tp.Tuple[str, RefSpan]], ProsodyReference] = None  # type: ignore
    is_initialize: bool = False

    def __post_init__(self):
        if self.refs is not None:
            raise RuntimeError("Use the 'create' method for make instance this class.")

        self.refs = defaultdict(ProsodyReference)

    def __getitem__(self, item):
        if item in self.refs:
            return self.refs[item]
        else:
            for key, ref in self.refs.items():
                if item in key.split("|"):
                    return ref

        return self.refs["default"]

    def __eq__(self, other: "ComplexProsodyReference"):
        if list(self.refs.keys()) == list(other.refs.keys()):
            return all(
                a.speaker_name == b.speaker_name
                and a.speaker_emb_index == b.speaker_emb_index
                for a, b in zip(self.refs.values(), other.refs.values())
            )
        else:
            return False

    @property
    def default(self):
        return self.refs["default"]

    @staticmethod
    def create(
        lang: str,
        speaker_name: tp.Union[str, tp.Dict[str, str]],
        speaker_reference: tp.Optional[
            tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]
        ] = None,
        style_reference: tp.Optional[
            tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]
        ] = None,
    ) -> "ComplexProsodyReference":
        if speaker_name is None:
            speaker_name = "no_name"

        if isinstance(speaker_name, str):
            speaker_name = {"default": speaker_name}
        else:
            if "default" not in speaker_name:
                raise RuntimeError("You must set the name of the base speaker.")

        prosody_ref = ComplexProsodyReference()
        prosody_ref.refs["default"].lang = lang
        prosody_ref.refs["default"].speaker_name = speaker_name["default"]

        for key, sp in speaker_name.items():
            prosody_ref.refs[key].set_speaker_name(sp)

        if speaker_reference is not None:
            if not isinstance(speaker_reference, dict):
                speaker_reference = {"default": speaker_reference}

            for key, ref in speaker_reference.items():
                prosody_ref.refs[key].set_speaker_reference(ref)

        if style_reference is not None:
            if not isinstance(style_reference, dict):
                style_reference = {"default": style_reference}

            for key, ref in style_reference.items():
                prosody_ref.refs[key].set_style_reference(ref)

        return prosody_ref

    def get(self, tag: str) -> ProsodyReference:
        return self.refs[tag]

    def preprocessing(
        self,
        all_speakers: tp.List[str],
        bio_embeddings: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        for ref in self.refs.values():
            if ref.speaker_name is None:
                ref.speaker_name = self.default.speaker_name

        for ref in self.refs.values():
            if ref.is_empty():
                ref.speaker_emb_index = 0

            ref.sampling_reference(all_speakers, bio_embeddings)

        # print("style reference:", self.default.speaker_emb_index)

    def postprocessing(self):
        if "default" not in self.refs or self.refs["default"].speaker_name is None:
            raise RuntimeError("You must set the name of the base speaker.")

        default_ref = self.refs["default"]

        if default_ref.is_empty():
            raise RuntimeError("You must set the default speaker or style reference.")

        if default_ref.style_reference_is_empty():
            default_ref.swap_reference("speaker", "style")

        for key in self.refs.keys():
            if self.refs[key].speaker_reference_is_empty():
                self.refs[key].copy_speaker_reference(default_ref)

        for key in self.refs.keys():
            if self.refs[key].style_reference_is_empty():
                self.refs[key].copy_style_reference(default_ref)

        for key in self.refs.keys():
            self.refs[key].filling()

    def set_speaker_id(
        self, speaker_id_map: tp.Dict, mean_bio_embeddings: tp.Optional[tp.Dict] = None
    ):
        for key in self.refs.keys():
            self.refs[key].set_speaker_id(speaker_id_map, mean_bio_embeddings)

    def set_bio_embedding(self, biometric_pipe: callable, bio_embeddings: tp.Dict):
        for key in self.refs.keys():
            self.refs[key].set_bio_embedding(biometric_pipe, bio_embeddings)

    def set_spectrogram_reference(self, spectrogram_pipe: callable):
        for key in self.refs.keys():
            self.refs[key].set_spectrogram_reference(spectrogram_pipe)

    @torch.no_grad()
    def set_feats_from_model(self, model: callable):
        for key in self.refs.keys():
            try:
                self.refs[key].set_feats_from_model(model)
            except Exception as e:
                print(e)

    def initialize(
        self,
        speaker_id_map: tp.Dict,
        bio_embeddings: tp.Optional[tp.Dict[str, tp.Any]] = None,
        mean_bio_embeddings: tp.Optional[tp.Dict[str, tp.Any]] = None,
        biometric_pipe: tp.Optional[callable] = None,
        audio_pipe: tp.Optional[callable] = None,
        seed: int = 0,
    ):
        if not self.is_initialize:
            set_random_seed(seed)

            if bio_embeddings is None:
                bio_embeddings = mean_bio_embeddings

            self.preprocessing(list(speaker_id_map.keys()), bio_embeddings)
            self.set_speaker_id(speaker_id_map, mean_bio_embeddings)
            self.set_bio_embedding(biometric_pipe, bio_embeddings)
            self.set_spectrogram_reference(audio_pipe)
            self.postprocessing()

            self.is_initialize = True

    def copy(self) -> "ComplexProsodyReference":
        return deepcopy(self)
