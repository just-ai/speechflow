import time
import typing as tp
import logging
import argparse

from pathlib import Path

from multilingual_text_parser.parser import TextParser

from annotator.asr_services import GoogleASR, OpenAIASR, YandexASR
from speechflow.data_pipeline.dataset_parsers import EasyDSParser
from speechflow.io import AudioChunk, AudioFormat, construct_file_list
from speechflow.logging.server import LoggingServer
from speechflow.utils.gpu_info import get_total_gpu_memory

__all__ = ["main"]

LOGGER = logging.getLogger("root")
OPENAI_ASR_TOTAL_GPU_MEM: int = 11


def parse_args():
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-d", "--data_root", help="data root directory", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-l", "--lang", help="target language", type=str, required=True
    )
    arguments_parser.add_argument(
        "-asr",
        "--asr_credentials",
        help="path to credentials file for ASR cloud service",
        type=Path,
    )
    arguments_parser.add_argument(
        "-ns",
        "--num_samples",
        help="max samples to load",
        type=int,
        default=0,
    )
    arguments_parser.add_argument(
        "-nproc",
        "--n_processes",
        help="number of workers for data processing",
        type=int,
        default=1,
    )
    arguments_parser.add_argument(
        "-ngpu", "--n_gpus", help="number of GPU device", type=int, default=0
    )
    return arguments_parser.parse_args()


def _convert_to_opus(audio_path: Path):
    wav_path = audio_path.with_suffix(".opus")
    if not wav_path.exists():
        audio_chunk = AudioChunk(audio_path).load()
        if audio_chunk.waveform.ndim == 2:
            audio_chunk.data = audio_chunk.data[0]
        audio_chunk.save(wav_path)


def main(
    data_root: Path,
    lang: str,
    asr_credentials: tp.Optional[Path] = None,
    num_samples: int = 0,
    n_processes: int = 1,
    n_gpus: int = 0,
    raise_on_converter_exc: bool = False,
    raise_on_asr_limit_exc: bool = False,
    disable_logging: bool = False,
):
    use_openai_asr = False

    with LoggingServer.ctx(
        data_root / "AudioTranscription_log.txt", disable=disable_logging
    ):
        try:
            if asr_credentials is None:
                raise ValueError("Credentials for ASR cloud service not set!")
            else:
                if not asr_credentials.exists():
                    raise FileNotFoundError(
                        "Credentials for ASR cloud service not found!"
                    )
        except Exception as e:
            LOGGER.warning(e)
            LOGGER.info("Switch on OpenAI ASR model!")
            use_openai_asr = True

        if not TextParser.check_language_support(lang):
            raise ValueError(f"Language {lang} is not support!")
        else:
            LOGGER.info(f"Run ASR for {lang} language!")

        while True:
            try:
                if "ru" in lang.lower() and not use_openai_asr:
                    LOGGER.info("Usage Yandex ASR")
                    asr = YandexASR(
                        asr_credentials,
                        TextParser.language_to_locale(lang).replace("_", "-"),
                        raise_on_converter_exc=raise_on_converter_exc,
                        raise_on_asr_limit_exc=raise_on_asr_limit_exc,
                    )
                elif not use_openai_asr:
                    LOGGER.info("Usage Google ASR")
                    asr = GoogleASR(
                        asr_credentials,
                        TextParser.language_to_locale(lang).replace("_", "-"),
                        raise_on_converter_exc=raise_on_converter_exc,
                        raise_on_asr_limit_exc=raise_on_asr_limit_exc,
                    )  # type: ignore
                else:
                    LOGGER.info("Usage OpenAI ASR")
                    asr = OpenAIASR(
                        lang,
                        device="cpu" if n_gpus == 0 else "cuda",
                        raise_on_converter_exc=raise_on_converter_exc,
                    )  # type: ignore
                    if n_gpus > 0:
                        n_processes = n_gpus * max(
                            int(get_total_gpu_memory(0) / OPENAI_ASR_TOTAL_GPU_MEM), 1
                        )
                break
            except Exception as e:
                LOGGER.error(e)
                time.sleep(5)

        parser = EasyDSParser(func=_convert_to_opus)
        for file_ext in [
            "*.aac",
            "*.mp4",
            "*.mkv",
            "*.mov",
            "*.m4a",
        ]:
            flist = list(data_root.rglob(file_ext))
            if flist:
                parser.run_from_path_list(flist, n_processes)

        output_file_ext = ".whisper" if use_openai_asr else ".json"
        flist = construct_file_list(
            data_root,
            AudioFormat.as_extensions(),
            with_subfolders=True,
            path_filter=lambda x: not Path(x).with_suffix(output_file_ext).exists(),
        )
        if not flist:
            return

        if num_samples > 0:
            flist = flist[:num_samples]

        asr.read_datasamples(
            file_list=flist, data_root=data_root, n_processes=n_processes
        )
        LOGGER.info("Generation of transcription complete!")


if __name__ == "__main__":
    main(**parse_args().__dict__)
