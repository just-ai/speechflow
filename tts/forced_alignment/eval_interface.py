import typing as tp

from pathlib import Path

import numpy as np
import torch

from multilingual_text_parser.data_types import Doc

from speechflow.data_pipeline.core import Batch, PipelineComponents
from speechflow.data_pipeline.datasample_processors.data_types import TTSDataSample
from speechflow.data_pipeline.datasample_processors.tts_text_processors import (
    TTSTextProcessor,
)
from speechflow.io import AudioChunk
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.dictutils import find_field
from speechflow.utils.init import init_class_from_config
from tts import forced_alignment
from tts.forced_alignment.data_types import AlignerForwardOutput

__all__ = ["GlowTTSEvaluationInterface"]

_REFERENECE_AUDIO = tp.Union[Path, tp.Tuple[np.ndarray, int]]


class GlowTTSEvaluationInterface:
    def __init__(
        self,
        ckpt_path: tp.Union[str, Path],
        device: str = "cpu",
    ):
        checkpoint = ExperimentSaver.load_checkpoint(Path(ckpt_path))
        cfg_data, cfg_model = ExperimentSaver.load_configs_from_checkpoint(checkpoint)
        self.device = torch.device(device)

        model_cls = getattr(forced_alignment, cfg_model["model"]["type"])
        self.model = model_cls(checkpoint["params"])
        self.model.eval()
        self.model.load_state_dict(checkpoint["state_dict"])
        if hasattr(self.model, "store_inverse"):
            self.model.store_inverse()
        self.model.to(device)

        def _remove_item(_list: list, _name: str):
            if _name in _list:
                _list.remove(_name)

        _remove_item(cfg_data["preproc"]["pipe"], "add_pauses_from_timestamps")
        cfg_data["preproc"]["pipe"].insert(0, "add_pauses_from_text")
        cfg_data["preproc"]["add_pauses_from_text"] = {
            "level": "syntagmas",
            "num_symbols": 3,
        }

        self.pipeline = PipelineComponents(cfg_data, "valid")

        ignored_fields = {"word_timestamps", "phoneme_timestamps"}
        self.is_gst_used = cfg_model["model"]["params"].get("use_gst", False)
        self.is_bio_embeddings_used = cfg_model["model"]["params"].get(
            "use_biometric_embeddings", False
        )

        self.pipeline = self.pipeline.with_ignored_fields(
            ignored_data_fields=ignored_fields
        )

        self.speaker_id_map = checkpoint.get("speaker_id_map")
        self.lang_id_map = checkpoint.get("lang_id_map")

        batch_processor_cls = getattr(forced_alignment, cfg_model["batch"]["type"])
        self.batch_processor = init_class_from_config(
            batch_processor_cls, cfg_model["batch"]
        )()
        self.batch_processor.device = self.device

        self.lang = find_field(cfg_data["preproc"], "lang")
        self.text_parser = TTSTextProcessor(lang=self.lang)

    @torch.inference_mode()
    def evaluate(self, batch: Batch) -> AlignerForwardOutput:
        inputs, _, _ = self.batch_processor(batch)
        outputs = self.model.inference(inputs)
        return outputs

    def synthesize(
        self,
        text: str,
        reference_audio: tp.Optional[_REFERENECE_AUDIO] = None,
        speaker: tp.Optional[tp.Union[int, str]] = None,
        lang: tp.Optional[tp.Union[int, str]] = None,
    ) -> AlignerForwardOutput:
        text = self.text_parser.process(Doc(text))

        if reference_audio is not None:
            if isinstance(reference_audio, Path):
                audio_chunk = AudioChunk(file_path=reference_audio).load()
            elif isinstance(reference_audio, (tuple, list)):
                wave, sr = reference_audio
                audio_chunk = AudioChunk(data=wave, sr=sr)
            else:
                raise AttributeError(
                    "Incorrect format of reference audio."
                    "Pass path-like or tuple(np.ndarray, sr)"
                )
        else:
            audio_chunk = None  # type: ignore

        if speaker is not None and self.speaker_id_map is not None:
            speaker_id = (
                self.speaker_id_map[speaker] if isinstance(speaker, str) else speaker
            )
        else:
            speaker_id = 0

        if lang is not None and self.lang_id_map is not None:
            lang_id = self.lang_id_map[speaker] if isinstance(lang, str) else lang
        else:
            lang_id = 0

        samples = [
            TTSDataSample(
                sent=sent, audio_chunk=audio_chunk, speaker_id=speaker_id, lang_id=lang_id
            )
            for sent in text.sents
        ]
        batch = self.pipeline.datasample_to_batch(samples)
        return self.evaluate(batch)
