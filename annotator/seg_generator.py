import io
import typing as tp
import logging
import argparse
import collections

from functools import partial
from pathlib import Path

import numpy as np

from tqdm import tqdm

from annotator.audiobook_spliter import AudiobookSpliter
from annotator.simple_segmentation import SimpleSegGenerator
from speechflow.data_pipeline.core.parser_types import Metadata
from speechflow.data_pipeline.dataset_parsers import EasyDSParser
from speechflow.io import AudioChunk, AudioFormat, generate_file_list, read_file_list
from speechflow.logging.server import LoggingServer

__all__ = ["main"]

LOGGER = logging.getLogger("root")


def parse_args():
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
        "-f", "--file_list", help="path to file list", type=Path
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
        "-sr",
        "--audio_sample_rate",
        help="sample rate for output audio",
        type=int,
        default=24000,
    )
    arguments_parser.add_argument(
        "-af",
        "--audio_format",
        help="format for output audio",
        type=str,
        default="wav",
    )
    arguments_parser.add_argument(
        "-m",
        "--multispeaker_mode",
        help="keep folders structure",
        action="store_true",
    )
    arguments_parser.add_argument(
        "-t", "--use_asr_transcription", help="use asr transcription", action="store_true"
    )
    arguments_parser.add_argument(
        "-s", "--split_audio", help="split audio by chunk", action="store_true"
    )
    arguments_parser.add_argument(
        "-p", "--prefix", help="prefix for segmentation files", type=str, default=""
    )
    return arguments_parser.parse_args()


def _get_speaker_name(md: Metadata, data_root: Path, sep: str = "/"):
    data_root = data_root.absolute().as_posix()
    file_path = md["audio_path"].absolute().as_posix()
    file_path = file_path.replace(data_root, "").lstrip(sep)
    speaker_name = file_path.split(sep)[0]
    return speaker_name


def _audio_converter(audio_path: Path, sample_rate: int, format: str):
    new_audio_path = audio_path.with_suffix(f".{format}")
    AudioChunk(audio_path).load(sr=sample_rate).save(new_audio_path, overwrite=True)
    if new_audio_path != audio_path:
        audio_path.unlink()


def _audio_loudnorm(audio_path: Path):
    from speechflow.data_pipeline.datasample_processors.audio_processors import (
        SignalProcessor,
    )
    from speechflow.data_pipeline.datasample_processors.data_types import AudioDataSample

    ds = AudioDataSample(audio_chunk=AudioChunk(audio_path).load())
    SignalProcessor.ffmpeg_loudnorm(ds)
    ds.audio_chunk.save(overwrite=True)


def main(
    data_root: Path,
    output_dir: Path,
    lang: str,
    file_list: tp.Optional[tp.Union[Path, io.StringIO]] = None,
    num_samples: int = 0,
    n_processes: int = 1,
    n_gpus: int = 0,
    multispeaker_mode: bool = False,
    use_asr_transcription: bool = False,
    split_audio: bool = True,
    resampling_audio: bool = True,
    loudnorm_audio: bool = True,
    audio_sample_rate: int = 24000,
    audio_format: tp.Literal["wav", "flac", "opus"] = "wav",
    prefix: str = "",
    raise_on_converter_exc: bool = False,
    disable_logging: bool = False,
) -> tp.Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    with LoggingServer.ctx(output_dir, disable=disable_logging):

        if file_list is None:
            file_list = generate_file_list(
                data_root, AudioFormat.as_extensions(), with_subfolders=True
            )
            text_from_label = False
        else:
            text_from_label = True

        flist = read_file_list(file_list, max_num_samples=num_samples, use_shuffle=False)
        assert isinstance(flist, list)

        if use_asr_transcription:
            LOGGER.info("Run AudiobookSpliter")
            seg_generator = AudiobookSpliter(
                lang=lang,
                device="cpu" if n_gpus == 0 else "cuda",
                text_from_label=text_from_label,
                raise_on_converter_exc=raise_on_converter_exc,
            )  # type: ignore
            data = seg_generator.read_datasamples(
                file_list=flist,
                data_root=data_root,
                n_processes=n_processes,
            )
            LOGGER.info("Stop AudiobookSpliter")
        else:
            LOGGER.info("Run SimpleSegGenerator")
            seg_generator = SimpleSegGenerator(
                lang=lang,
                device="cpu" if n_gpus == 0 else "cuda",
                text_from_label=text_from_label,
                raise_on_converter_exc=raise_on_converter_exc,
            )  # type: ignore
            data = seg_generator.read_datasamples(
                file_list=flist, data_root=data_root, n_processes=n_processes
            )
            LOGGER.info("Stop SimpleSegGenerator")

        total_sent_count = orig_sent_count = 0
        total_audio_len = orig_audio_len = 0.0
        seg_files = []
        audio_files = []
        seg_count: tp.Dict[str, int] = collections.defaultdict(int)
        folder_id: tp.Dict[str, int] = collections.defaultdict(int)
        speakers = set()
        for md in tqdm(data, desc="Saving segmentations"):
            orig_audio_len += AudioChunk(md["audio_path"]).duration
            orig_sent_count += len(md["text"].sents)

            if not md["segmentation"]:
                LOGGER.error(f"{md['audio_path']}|{md['text'].text}")
                continue

            if multispeaker_mode:
                speaker_name = _get_speaker_name(md, data_root)
                sega_out_dir = output_dir / speaker_name
            else:
                speaker_name = output_dir.name
                sega_out_dir = output_dir

            if speaker_name not in folder_id:
                folder_id[speaker_name] = 0
                while (sega_out_dir / f"{folder_id[speaker_name]:03}").exists():
                    folder_id[speaker_name] += 1

            if seg_count[speaker_name] >= 2000:
                seg_count[speaker_name] = 0
                folder_id[speaker_name] += 1

            catalogs = md["file_path"].relative_to(data_root).parent
            if multispeaker_mode:
                catalogs = catalogs.relative_to(speaker_name)

            subfolder = f"{folder_id[speaker_name]:03}"
            sega_out_dir = sega_out_dir / catalogs / subfolder
            sega_out_dir = Path(sega_out_dir.as_posix().replace(" ", "_"))
            sega_out_dir.mkdir(parents=True, exist_ok=True)

            for sega in md["segmentation"]:
                try:
                    file_name = sega_out_dir / f"{prefix}{total_sent_count}.TextGrid"
                    sega.meta.update({"speaker_name": speaker_name})
                    sega.save(file_name, with_audio=split_audio)
                    seg_files.append(
                        f"{file_name.relative_to(output_dir).as_posix()}|{sega.sent.text_orig}"
                    )
                    audio_files.append(sega.audio_chunk.file_path)
                    del sega.audio_chunk

                    speakers.add(speaker_name)
                    seg_count[speaker_name] += 1
                    total_sent_count += 1
                    total_audio_len += sega.ts_by_words.duration
                except Exception as e:
                    LOGGER.error(e)

        if split_audio and loudnorm_audio:
            LOGGER.info("Volume normalization of audio files")
            EasyDSParser(_audio_loudnorm).run_from_path_list(
                path_list=audio_files, n_processes=n_processes
            )

        if split_audio and resampling_audio:
            LOGGER.info(f"Resampling of audio files to {audio_sample_rate}Hz")
            func = partial(
                _audio_converter, sample_rate=audio_sample_rate, format=audio_format
            )
            EasyDSParser(func).run_from_path_list(
                path_list=audio_files, n_processes=n_processes
            )

        #  write file list
        flist_path = output_dir / "filelist.txt"
        flist_path.write_text("\n".join(seg_files), encoding="utf-8")

        total_audio_duration = np.round(total_audio_len / 3600, 3)
        if orig_sent_count > 0:
            sent_ratio = np.round(total_sent_count / orig_sent_count * 100, 3)
            audio_ratio = np.round(total_audio_len / orig_audio_len * 100, 3)

            LOGGER.info(f"Total aligned sentences: {total_sent_count} | {sent_ratio}%")
            LOGGER.info(
                f"Total audio duration in hours: {total_audio_duration} | {audio_ratio}%"
            )
        else:
            LOGGER.info("Files not processed!")

        LOGGER.info("Generation of segmentation completed!")

        return {
            "audio_len": total_audio_duration,
            "speakers": speakers,
            "flist_path": flist_path.as_posix(),
        }


if __name__ == "__main__":
    main(**parse_args().__dict__)
