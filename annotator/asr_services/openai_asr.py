import typing as tp
import logging

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import whisper_timestamped as whisper

from whisper.tokenizer import get_tokenizer

from annotator.asr_services.cloud_asr import ASRException, CloudASR
from speechflow.data_pipeline.core.parser_types import Metadata
from speechflow.utils.checks import check_install
from speechflow.utils.gpu_info import get_freer_gpu, get_total_gpu_memory
from speechflow.utils.tqdm_disable import tqdm_disable

__all__ = ["OpenAIASR"]

LOGGER = logging.getLogger("root")
OpenAIASR_LANGUAGES: dict = whisper.tokenizer.LANGUAGES  # type: ignore


class OpenAIASR(CloudASR):
    """Generate transcription for audio files."""

    def __init__(
        self,
        lang: str,
        model_name: tp.Literal[
            "large-v2", "medium", "small", "base", "tiny"
        ] = "large-v2",
        device: str = "cpu",
        raise_on_converter_exc: bool = False,
    ):
        super().__init__(
            sample_rate=16000,
            output_file_ext=".whisper",
            raise_on_converter_exc=raise_on_converter_exc,
            raise_on_asr_limit_exc=False,
            release_func=self.release,
        )

        if not check_install("ffmpeg", "-version"):
            raise RuntimeError("ffmpeg is not installed on your system")

        self._lang = lang.lower()[:2]
        self._model_name = model_name
        self._device = device

        # Hack for kyrgyzstan language
        if self._lang == "ky":
            self._lang = "kk"

        if self._lang not in OpenAIASR_LANGUAGES:
            raise ValueError(f"Language {lang} is not support in Whisper model!")

    @staticmethod
    def release(_):
        if hasattr(OpenAIASR, "parser"):
            OpenAIASR.parser = None

    def _get_device(self) -> str:
        if self._device == "cuda":
            gpu_idx = get_freer_gpu(strict=False)
            total_gpu_memory = get_total_gpu_memory(gpu_idx)
            if "large" in self._model_name and total_gpu_memory < 11:
                LOGGER.info(
                    f"There is not enough GPU memory available for Whisper {self._model_name} model"
                )
                return "cpu"
            elif "medium" in self._model_name and total_gpu_memory < 3.9:
                LOGGER.info(
                    f"There is not enough GPU memory available for Whisper {self._model_name} model"
                )
                return "cpu"
            elif "small" in self._model_name and total_gpu_memory < 1.9:
                LOGGER.info(
                    f"There is not enough GPU memory available for Whisper {self._model_name} model"
                )
                return "cpu"
            else:
                return f"cuda:{gpu_idx}"
        else:
            return self._device

    def _get_suppress_tokens(self, model, lang: str) -> tp.List[int]:
        tokenizer = get_tokenizer(multilingual=model.is_multilingual)
        tokens = [
            i
            for i in range(tokenizer.eot)
            if any(c in "0123456789<>#№@%&*+^°$€¥/\\[]{}" for c in tokenizer.decode([i]))
        ]

        if "ru" in lang:
            chars = [chr(i) for i in range(0x0041, 0x005B)]
            tokens += [
                i
                for i in range(tokenizer.eot)
                if any(c in chars for c in tokenizer.decode([i]).strip())
            ]
            chars = [chr(i).lower() for i in range(0x0041, 0x005B)]
            tokens += [
                i
                for i in range(tokenizer.eot)
                if any(c in chars for c in tokenizer.decode([i]).strip())
            ]

        return list(set(tokens))

    def _transcription(self, metadata: Metadata) -> Metadata:
        if getattr(OpenAIASR, "model", None) is None:
            with OpenAIASR.lock:
                model = whisper.load_model(self._model_name, device=self._get_device())
                suppress_tokens = self._get_suppress_tokens(model, self._lang)
                assert model.is_multilingual
                setattr(OpenAIASR, "model", model)
                setattr(OpenAIASR, "suppress_tokens", suppress_tokens)
        else:
            model = getattr(OpenAIASR, "model")
            suppress_tokens = getattr(OpenAIASR, "suppress_tokens")

        md = {"audio_path": metadata["audio_path"]}

        if "waveform" in metadata:
            audio = metadata["waveform"] / np.float32(np.iinfo(np.int16).max)
        else:
            audio = Path(metadata["audio_path"]).as_posix()

        with tqdm_disable():
            if model.device.type != "cpu":
                with torch.cuda.device(model.device):
                    result = whisper.transcribe(
                        model,
                        audio=audio,
                        language=self._lang,
                        condition_on_previous_text=False,
                        beam_size=5,
                        best_of=5,
                        suppress_tokens=[-1] + suppress_tokens,
                    )
            else:
                result = whisper.transcribe(
                    model,
                    audio=audio,
                    language=self._lang,
                    condition_on_previous_text=False,
                    beam_size=5,
                    best_of=5,
                    suppress_tokens=[-1] + suppress_tokens,
                )

        timestamps = []
        for segment in result["segments"]:
            text = segment.get("text", "")
            words_stat = Counter(text.split())
            if max([v for k, v in words_stat.items() if len(k) > 1]) >= 10:
                if len(result["segments"]) == 1:
                    raise ASRException(f"Whisper hallucination detected: {text}")
                else:
                    continue

            for item in segment.get("words", []):
                word = item["text"]
                start = item["start"]
                end = item["end"]
                if not word.strip():
                    continue

                if not timestamps:
                    word = word.strip().title()

                timestamps.append((word.strip(), start, end))

        text = " ".join([item[0] for item in timestamps]).strip()

        if text == text.upper():
            raise ASRException("There is probably no speech in the audio file.")

        md["transcription"] = {
            "text": text,
            "timestamps": timestamps,
            "lang": self._lang,
            "model_name": self._model_name,
            "version": whisper.__version__,
        }
        return md

    @staticmethod
    def _to_text(transcription) -> str:
        return transcription["text"]
