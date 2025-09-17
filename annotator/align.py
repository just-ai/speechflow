import os
import enum
import time
import uuid
import typing as tp
import logging
import argparse
import subprocess as sp

from os import environ as env
from pathlib import Path
from random import shuffle

import numpy as np
import torch
import numpy.typing as npt

from tqdm import tqdm

from speechflow.data_pipeline.core import DataPipeline
from speechflow.data_pipeline.datasample_processors.tts_text_processors import (
    TTSTextProcessor,
)
from speechflow.data_pipeline.dataset_parsers import EasyDSParser
from speechflow.data_server.helpers import LoaderParams, init_data_loader
from speechflow.io import (
    AudioSeg,
    AudioSegPreview,
    Timestamps,
    check_path,
    construct_file_list,
    tp_PATH,
)
from speechflow.logging.server import LoggingServer
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.dictutils import find_field
from speechflow.utils.gpu_info import get_freer_gpu
from speechflow.utils.init import init_class_from_config
from tts import forced_alignment

__all__ = ["Aligner", "AlignStage", "main"]

LOGGER = logging.getLogger("root")


class AlignStage(enum.Enum):
    stage1 = 1
    stage2 = 2
    stage3 = 3


def parse_args():
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-d", "--data_root", help="path to dataset", type=str, required=True
    )
    arguments_parser.add_argument(
        "-ckpt",
        "--ckpt_path",
        help="path to model checkpoint",
        type=Path,
        required=True,
    )
    arguments_parser.add_argument(
        "-st", "--stage", help="alignment stage", type=int, required=True
    )
    arguments_parser.add_argument(
        "-bs", "--batch_size", help="num samples in batch", type=int, default=16
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
    arguments_parser.add_argument(
        "--sega_suffix",
        help="suffix for sega file name",
        type=str,
        default="",
    )
    arguments_parser.add_argument(
        "--flist_path",
        help="path to file list for aligner",
        type=str,
        nargs="+",
        default=None,
    )
    args = arguments_parser.parse_args()
    return args


class Aligner:
    @check_path
    def __init__(
        self,
        ckpt_path: tp_PATH,
        stage: AlignStage,
        batch_size: int = 1,
        n_processes: int = 1,
        device: str = "cpu",
        reverse_mode: bool = False,
        min_pause_len: float = 0.08,  # in seconds
        sega_suffix: str = "",
        max_duration: tp.Optional[float] = 15,  # in seconds
        preload: tp.Optional[tp.Union[tp.Dict, tp.Tuple]] = None,
    ):
        self._ckpt_path = ckpt_path
        self._stage = stage
        self._batch_size = batch_size
        self._n_processes = n_processes
        self._device = device
        self._reverse_mode = reverse_mode
        self._min_pause_len = min_pause_len
        self._sega_suffix = sega_suffix

        if isinstance(preload, dict):
            ckpt_preload = preload
        else:
            ckpt_preload = None

        (
            self._cfg_data,
            self._hop_len,
            self._speaker_id_map,
            self._lang_id_map,
            self._lang,
        ) = self._prepare_aligning(
            ckpt_path, reverse_mode, max_duration, ckpt_preload=ckpt_preload
        )

        env["DEVICE"] = device

        self._data_pipeline = DataPipeline(self._cfg_data)
        self._data_pipeline.init_components()

        if isinstance(preload, tuple):
            self._aligner_model, self._batch_processor = preload
        else:
            self._aligner_model, self._batch_processor = self._get_model(
                ckpt_path, device, ckpt_preload
            )

    @property
    def lang(self) -> str:
        return self._lang

    @property
    def lang_id_map(self) -> tp.List[str]:
        return self._lang_id_map

    @property
    def min_pause_len(self) -> float:
        return self._min_pause_len

    @property
    def device(self):
        return self._device

    @property
    def model(self) -> tuple:
        return self._aligner_model, self._batch_processor

    @staticmethod
    def _prepare_aligning(
        ckpt_path: tp_PATH,
        reverse_mode: bool = False,
        max_duration: tp.Optional[float] = None,
        ckpt_preload: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        if ckpt_preload is None:
            checkpoint = ExperimentSaver.load_checkpoint(ckpt_path)
        else:
            checkpoint = ckpt_preload

        cfg_data, _ = ExperimentSaver.load_configs_from_checkpoint(checkpoint)

        cfg_data["dataset"]["subsets"].remove("train")
        cfg_data["parser"]["progress_bar"] = False
        if "split_by_phrases" in cfg_data["parser"]["pipe"]:
            if max_duration:
                cfg_data["parser"]["pipe_cfg"]["split_by_phrases"][
                    "max_duration"
                ] = max_duration
            else:
                cfg_data["parser"]["pipe"].remove("split_by_phrases")

        if "check_phoneme_length" in cfg_data["parser"]["pipe"]:
            cfg_data["parser"]["pipe"].remove("check_phoneme_length")

        cfg_data["processor"]["output_collated_only"] = False
        cfg_data["processor"].pop("dump", None)

        if "augment_wave" in cfg_data["preproc"]["pipe"]:
            cfg_data["preproc"]["pipe"].remove("augment_wave")

        if "ssl" in cfg_data["preproc"]["pipe"]:
            cfg_data["preproc"]["pipe"].remove("ssl")

        # TODO: support legacy models
        if "text" in cfg_data["preproc"]["pipe_cfg"]:
            cfg_data["preproc"]["pipe_cfg"]["text"].type = "TTSTextProcessor"

        if "reverse" in cfg_data["preproc"]["pipe"]:
            if reverse_mode:
                cfg_data["preproc"]["pipe_cfg"].setdefault("reverse", {})
                cfg_data["preproc"]["pipe_cfg"]["reverse"]["p"] = 1.0
            else:
                LOGGER.info("remove reverse function")
                cfg_data["preproc"]["pipe"].remove("reverse")

                # TODO: support legacy models
                if "reverse" in cfg_data["preproc"]["pipe"]:
                    cfg_data["preproc"]["pipe"].remove("reverse")

        hop_len = find_field(cfg_data, "hop_len")

        speaker_id_setter = find_field(cfg_data, "SpeakerIDSetter")
        speaker_id_setter.pop("resume_from_checkpoint", None)

        speaker_id_map = checkpoint.get("speaker_id_map", {})
        if speaker_id_setter is not None and speaker_id_map:
            speaker_id_setter["speaker_id_map"] = [
                f"{key}:{value}" for key, value in speaker_id_map.items()
            ]
            cfg_data["singleton_handlers"]["SpeakerIDSetter"][
                "remove_unknown_speakers"
            ] = False

        lang_id_map = checkpoint.get("lang_id_map")
        if speaker_id_setter is not None and lang_id_map is not None:
            speaker_id_setter["lang_id_map"] = [
                f"{key}:{value}" for key, value in lang_id_map.items()
            ]

        lang = find_field(cfg_data["preproc"], "lang")

        cfg_data["sampler"] = {"type": "SimpleSampler", "comb_by_len": True}

        # TODO: support legacy models
        if "load_audio_segmentation" not in cfg_data["preproc"]["pipe"]:
            cfg_data["preproc"]["pipe"].insert(1, "load_audio_segmentation")
        if "symbols" in cfg_data.collate.additional_fields:
            cfg_data.collate.additional_fields.remove("symbols")

        return (
            cfg_data,
            hop_len,
            speaker_id_map,
            lang_id_map,
            lang,
        )

    @staticmethod
    def _get_model(
        ckpt_path: tp_PATH,
        device: str = "cpu",
        ckpt_preload: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        if ckpt_preload is None:
            checkpoint = ExperimentSaver.load_checkpoint(Path(ckpt_path))
        else:
            checkpoint = ckpt_preload

        _, cfg_model = ExperimentSaver.load_configs_from_checkpoint(checkpoint)

        device = torch.device(device)

        # TODO: support legacy models
        if "plbert_feat_dim" in checkpoint["params"]:
            checkpoint["params"]["xpbert_feat_dim"] = checkpoint["params"].pop(
                "plbert_feat_dim"
            )
            checkpoint["params"]["xpbert_feat_proj_dim"] = checkpoint["params"].pop(
                "plbert_feat_proj_dim"
            )

        model_cls = getattr(forced_alignment, cfg_model["model"]["type"])
        model = model_cls(checkpoint["params"])
        model.eval()

        # TODO: strict=False - support legacy models
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.to(device)

        batch_processor_cls = getattr(forced_alignment, cfg_model["batch"]["type"])
        batch_processor = init_class_from_config(
            batch_processor_cls, cfg_model["batch"]
        )()
        batch_processor.set_device(device)

        return model, batch_processor

    @staticmethod
    def _get_phonemes_lengths(attention_map):
        attention_map[0, 0] = 1
        argmaxed = attention_map.argmax(1)
        num_phonemes = attention_map.shape[1]

        lens = [0] * num_phonemes
        for i in range(attention_map.shape[1]):
            lens[i] = (argmaxed == i).sum()

        iter_counter = 0
        while (not all([x > 0 for x in lens])) and (iter_counter < 5000):
            for i in range(len(lens)):
                if lens[i] <= 0:
                    lens[i] += 1
                    try:
                        lens[i - 1] -= 1
                    except IndexError:
                        lens[i + 1] -= 1
            iter_counter += 1

        if iter_counter == 5000:
            raise RuntimeError("bad attention map!")

        return [x.item() for x in lens]

    @staticmethod
    def _get_intervals(
        cummulative_lengths: npt.NDArray,
        sample_rate: int,
        hop_length: int,
        wave_duration: float,
    ) -> tp.List[tp.Tuple[float, float]]:
        scale = hop_length / sample_rate

        intervals = [(0.0, cummulative_lengths[0] * scale)]
        for i in range(1, len(cummulative_lengths)):
            a = max(0.0, min((cummulative_lengths[i - 1] * scale), wave_duration))
            b = max(0.0, min((cummulative_lengths[i] * scale), wave_duration))
            intervals.append((a, b))

        return intervals  # in seconds

    @staticmethod
    def _remove_small_pauses(timestamps, min_pause_len):
        prev_ts = timestamps[0]
        for ts in timestamps[1:]:
            diff = abs(prev_ts[1] - ts[0])
            if 1.0e-6 < abs(prev_ts[1] - ts[0]) < min_pause_len:
                prev_ts[1] += diff / 2.0
                ts[0] -= diff / 2.0
            prev_ts = ts
        return timestamps

    def _update_sega(self, file_path, audio_chunk, symbols, lens):
        sega = AudioSeg.load(file_path)
        cummulative_lens = np.cumsum(lens)

        raw_intervals = self._get_intervals(
            cummulative_lens,
            audio_chunk.sr,
            self._hop_len,
            audio_chunk.duration,
        )
        intervals = [
            interval
            for i, interval in enumerate(raw_intervals)
            if not TTSTextProcessor.is_service_symbol(symbols[i])
        ]
        aligned_timestamps = Timestamps(np.asarray(intervals))

        if self._stage != AlignStage.stage1:
            aligned_timestamps = self._remove_small_pauses(
                aligned_timestamps, self._min_pause_len
            )

        ts_begin = raw_intervals[0][0] if raw_intervals[0][0] > 0.05 else 0.0
        ts_end = (
            raw_intervals[-1][1]
            if audio_chunk.end - raw_intervals[-1][1] > 0.05
            else audio_chunk.duration
        )

        # inverse wave trim function
        aligned_timestamps += audio_chunk.begin
        if ts_begin is not None:
            ts_begin += audio_chunk.begin
        if ts_end is not None:
            ts_end += audio_chunk.begin

        sega.set_phoneme_timestamps(
            aligned_timestamps,
            relative=True,
            ts_begin=ts_begin,
            ts_end=ts_end,
        )

        sega_name = f".TextGridStage{self._stage.value}{self._sega_suffix}"
        if self._reverse_mode:
            sega_name += "_reverse"

        sega.meta["aligner_model"] = self._ckpt_path.name
        sega.save(file_path.with_suffix(sega_name))

    def _batch_processing(self, batch):
        forward_input, _, samples = self._batch_processor(batch)

        with torch.no_grad():
            output = self._aligner_model(forward_input, adjust_attention=True)

        x_lens = forward_input.input_lengths.cpu().numpy()
        y_lens = forward_input.output_lengths.cpu().numpy()
        alignings = output.aligning_path.cpu().numpy()

        for i, s in enumerate(samples):
            try:
                aligning_path = alignings[i][: y_lens[i], : x_lens[i]]
                if self._reverse_mode:
                    aligning_path = np.fliplr(aligning_path).copy()
                lens = self._get_phonemes_lengths(aligning_path)
                self._update_sega(s.file_path, s.audio_chunk, s.transcription_text, lens)
            except Exception as e:
                LOGGER.error(f"error processing for {s.file_path}: {e}")

    @staticmethod
    def _read_metadata(file_path: str):
        return {"file_path": Path(file_path), "sega": AudioSegPreview.load(file_path)}

    def process(self, file_list: tp.List[str]):
        parser = EasyDSParser(self._read_metadata, progress_bar=len(file_list) > 1)
        dataset = parser.run_from_path_list(
            path_list=file_list, n_processes=self._n_processes
        )
        name = self._data_pipeline.subsets[0]
        dataset = self._data_pipeline[name].metadata_to_datasample(
            dataset, as_dataset=True
        )
        self._data_pipeline[name].set_dataset(dataset)

        with init_data_loader(
            loader_params=LoaderParams(batch_size=self._batch_size, non_stop=False),
            data_pipeline=self._data_pipeline,
            n_processes=self._n_processes,
            n_gpus=int(env.get("N_GPUS", 0)),
        ) as loaders:
            loader = list(loaders.values())[0]
            for _ in tqdm(
                range(len(loader)),
                desc="Generating aligning paths",
                disable=len(loader) == 1,
            ):
                self._batch_processing(next(loader))

    def align_sega(self, file_path: tp.Union[str, Path]):
        pipe = self._data_pipeline[self._data_pipeline.subsets[0]]
        md = {"file_path": Path(file_path), "sega": AudioSegPreview.load(file_path)}
        ds = pipe.metadata_to_datasample([md])[0]
        ds.speaker_id = self._speaker_id_map.get(ds.speaker_name, 0)
        ds.lang_id = self._lang_id_map[ds.lang]
        batch = pipe.datasample_to_batch([ds])
        self._batch_processing(batch)


def _get_file_list(
    data_root: tp.Union[str, Path],
    stage: int,
    sega_suffix: str = "",
    flist_path: tp.Optional[tp.List[tp.Union[str, Path]]] = None,
):
    aligner_stage: AlignStage = AlignStage(stage)
    ext = (
        ".TextGrid"
        if aligner_stage == AlignStage.stage1
        else f".TextGridStage{aligner_stage.value - 1}{sega_suffix}"
    )

    if flist_path is None:
        all_files = construct_file_list(
            data_root=data_root, with_subfolders=True, ext=ext
        )
    else:
        all_files = []
        for path in flist_path:
            lines = Path(path).read_text(encoding="utf-8").split("\n")
            lines = [item.split("|")[0] for item in lines]
            lines = [Path(item).with_suffix(ext) for item in lines]

            for item in lines:
                if not item.is_absolute():
                    item = Path(path).parent / item
                if item.exists():
                    all_files.append(item.as_posix())

    return all_files


def main(
    data_root: tp.Union[str, Path],
    ckpt_path: tp.Union[str, Path],
    stage: int,
    batch_size: int = 1,
    num_samples: int = 0,
    n_processes: int = 1,
    n_gpus: int = 0,
    sega_suffix: str = "",
    flist_path: tp.Optional[tp.List[tp.Union[str, Path]]] = None,
):
    device = "cpu" if n_gpus == 0 else f"cuda:{get_freer_gpu(strict=False)}"
    if "cuda" in device:
        torch.tensor([0.0], device=device)

    with LoggingServer.ctx(data_root):
        file_list = _get_file_list(data_root, stage, sega_suffix, flist_path)

        if num_samples > 0:
            file_list = file_list[:num_samples]

        aligner = Aligner(
            ckpt_path,
            AlignStage(stage),
            batch_size,
            n_processes,
            device=device,
            sega_suffix=sega_suffix,
        )
        aligner.process(file_list)

        LOGGER.info("Segmentation alignment complete!")


if __name__ == "__main__":
    args = parse_args()

    if args.n_gpus <= 1:
        main(**args.__dict__)
    else:
        flist = _get_file_list(
            args.data_root, args.stage, args.sega_suffix, args.flist_path
        )

        if args.n_gpus % 2 == 0:
            env["N_GPUS"] = "1"
            n_proc = args.n_gpus // 2
        else:
            args["n_processes"] = 4
            n_proc = args.n_gpus

        shuffle(flist)
        k = (len(flist) + n_proc) // n_proc
        flist_by_chunk = [flist[i : i + k] for i in range(0, len(flist), k)]

        processes = []
        args = args.__dict__
        args["n_gpus"] = 1
        for i in range(min(n_proc, len(flist_by_chunk))):
            tmp_file = Path(args["data_root"]) / f"{uuid.uuid4()}.txt"
            tmp_file.write_text("\n".join(flist_by_chunk[i]), encoding="utf-8")
            args["flist_path"] = tmp_file.as_posix()

            cmd = ["python", "-m", "annotator.align"]
            for key, value in args.items():
                cmd.append(f"--{key}")
                cmd.append(f"{value}")

            LOGGER.info(f"processes {i}: {cmd}")
            processes.append((sp.Popen(cmd), tmp_file))
            time.sleep(15.0)

        for p, tmp_file in processes:
            p.wait()
            tmp_file.unlink()
            os.system(f"kill -9 {p.pid}")
