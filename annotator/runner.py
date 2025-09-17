import re
import sys
import typing as tp
import logging
import argparse
import subprocess as sp

from collections import Counter
from pathlib import Path

import numpy as np
import distutils

from tqdm import tqdm

from annotator.align import main as run_align
from annotator.audio_transcription import main as run_audio_transcription
from annotator.eval_interface import AnnotatorEvaluationInterface
from annotator.seg_generator import main as run_seg_generator
from speechflow.data_pipeline.dataset_parsers import EasyDSParser
from speechflow.io import AudioSeg, Config
from speechflow.logging.server import LoggingServer
from speechflow.utils.fs import get_root_dir
from speechflow.utils.gpu_info import get_freer_gpu, get_total_gpu_memory
from speechflow.utils.init import init_method_from_config
from tts.forced_alignment.scripts.train import main as train_fa

LOGGER = logging.getLogger("runner")
ANNOTATOR_TOTAL_GPU_MEM: int = 8


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
        "-l", "--lang", help="target language", type=str, default="MULTILANG"
    )
    arguments_parser.add_argument(
        "-lf", "--langs_filter", help="language filter", nargs="+", type=str
    )
    arguments_parser.add_argument(
        "-sf", "--speakers_filter", help="speaker names filter", nargs="+", type=str
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
        "-ngw",
        "--n_workers_per_gpu",
        help="number of workers running on GPU",
        type=int,
        default=1,
    )
    arguments_parser.add_argument(
        "-ns", "--num_samples", help="max samples to load", type=int, default=0
    )
    arguments_parser.add_argument(
        "-bs",
        "--batch_size",
        help="num samples in batch",
        type=int,
        default=16,
    )
    arguments_parser.add_argument(
        "-epochs",
        "--max_epochs",
        help="num epochs for training",
        nargs="+",
        type=int,
        default=[30, 30],
    )
    arguments_parser.add_argument(
        "--use_asr_transcription",
        help="using ASR transcription to split audio into single utterances",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=True,
    )
    arguments_parser.add_argument(
        "--use_resampling_audio",
        help="using resampling audio to sample rate",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=True,
    )
    arguments_parser.add_argument(
        "--use_loudnorm_audio",
        help="using audio volume nomalization",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=True,
    )
    arguments_parser.add_argument(
        "--start_step", help="start from step", type=int, default=0
    )
    arguments_parser.add_argument(
        "--max_step", help="maximum number steps", type=int, default=4
    )
    arguments_parser.add_argument(
        "--start_stage", help="start from stage", type=int, default=1
    )
    arguments_parser.add_argument(
        "--sega_suffix",
        help="suffix for sega file name",
        type=str,
        default="",
    )
    arguments_parser.add_argument(
        "--pretrained_models",
        help="path to pretrained checkpoints",
        nargs="+",
        type=Path,
    )
    arguments_parser.add_argument(
        "--finetune_model",
        help="path to checkpoint for finetune",
        type=Path,
    )
    arguments_parser.add_argument(
        "--resume_from_path", help="path to experiment folder", type=Path
    )
    arguments_parser.add_argument(
        "--asr_credentials",
        help="path to credentials file for ASR cloud service (used OpenAI Whisper by default)",
        type=Path,
    )
    return arguments_parser.parse_args()


def _run_subprocess(
    cmd: tp.List[str], get_last_message: bool = False
) -> tp.Union[str, None]:
    try:
        LOGGER.info(cmd)
        output = sp.run(cmd, check=True, capture_output=True, text=True)
        LOGGER.info(output.stdout)
        LOGGER.info(output.stderr)

        output_messages = output.stdout.split()
        if get_last_message and output_messages:
            LOGGER.info(output_messages[-1])
            return output_messages[-1]

    except sp.CalledProcessError as e:
        LOGGER.info(e.stderr)
        raise e

    return None


def _update_fa_configs(
    cfg_model_name: str,
    cfg_data_name: str,
    lang: str,
    output_dir: Path,
    langs_filter: tp.Optional[tp.List[str]],
    speakers_filter: tp.Optional[tp.List[str]],
    num_samples: int,
    n_processes: int,
    n_gpus: int,
    n_workers_per_gpu: int,
    batch_size: int,
    max_epochs: int,
    experiment_path: tp.Optional[Path],
    finetune_model: tp.Optional[Path] = None,
    sega_suffix: str = "",
) -> tp.Tuple[tp.Optional[Path], tp.Optional[Path]]:
    n_gpus = max(n_gpus - 1, 0)

    root = get_root_dir()
    configs_dir = root / "tts/forced_alignment/configs/2stage"

    lang_dir = lang if langs_filter is None else "_".join(langs_filter)

    cfg_model_path = root / configs_dir / cfg_model_name
    cfg_data_path = root / configs_dir / cfg_data_name
    if not cfg_model_path.exists() or not cfg_data_path.exists():
        raise FileNotFoundError("Configs for forced alignment model not found!")

    # update model config
    cfg_model = Config.create_from_file(cfg_model_path)

    if not Path(cfg_model["dirs"].get("logging", "")).is_absolute():
        cfg_model["dirs"]["logging"] = (
            (output_dir / "forced_alignment" / lang_dir / cfg_model["dirs"]["logging"])
            .absolute()
            .as_posix()
        )
    cfg_model["trainer"]["accelerator"] = "gpu"
    cfg_model["trainer"]["devices"] = [get_freer_gpu()]
    if batch_size:
        cfg_model["data_loaders"]["batch_size"] = batch_size
    if max_epochs:
        cfg_model["trainer"]["max_epochs"] = max_epochs
    if experiment_path:
        cfg_model["trainer"]["resume_from_checkpoint"] = experiment_path.as_posix()
    if finetune_model:
        assert finetune_model.exists()
        cfg_model["model"]["init_from"] = {"ckpt_path": finetune_model.as_posix()}
    if n_gpus == 0:
        cfg_model["model"]["params"]["speaker_biometric_model"] = "resemblyzer"

    cfg_model_path = output_dir / "forced_alignment" / lang_dir / cfg_model_name
    cfg_model.to_file(cfg_model_path)

    # update data config
    config_data = Config.create_from_file(cfg_data_path)

    config_data["dirs"]["data_root"] = output_dir.as_posix()

    if "Stage" in config_data["file_search"]["ext"]:
        config_data["file_search"]["ext"] += sega_suffix

    config_data["data_server"]["n_processes"] = (
        n_workers_per_gpu * n_gpus if n_gpus > 0 else n_processes
    )
    config_data["data_server"]["n_gpus"] = n_gpus

    if num_samples:
        config_data["dataset"]["max_num_samples"] = num_samples
    if langs_filter:
        config_data["dataset"]["directory_filter"] = {"include": langs_filter}

    config_data["singleton_handlers"].setdefault("SpeakerIDSetter", {})
    speaker_ids = config_data["singleton_handlers"]["SpeakerIDSetter"]
    speaker_ids["remove_unknown_speakers"] = False
    if langs_filter:
        speaker_ids["langs_filter"] = langs_filter
    if speakers_filter:
        speaker_ids["speakers_filter"] = speakers_filter
    if finetune_model:
        speaker_ids["resume_from_checkpoint"] = finetune_model.as_posix()

    if n_gpus == 0 and "voice_bio" in config_data["preproc"]["pipe"]:
        config_data["preproc"]["pipe_cfg"]["voice_bio"]["model_type"] = "resemblyzer"

    config_data["preproc"]["pipe_cfg"]["text"]["lang"] = lang

    cfg_data_path = output_dir / "forced_alignment" / lang_dir / cfg_data_name
    config_data.to_file(cfg_data_path)

    return cfg_model_path, cfg_data_path


def _train_fa(**kwargs) -> Path:
    LOGGER.info(f"Run GlowTTS with args {kwargs}")

    if sys.platform == "win32":
        return train_fa(**kwargs)
    else:
        cmd = ["python", "-m", "tts.forced_alignment.scripts.train"]
        for key, value in kwargs.items():
            if value is None:
                continue
            cmd.append(f"--{key}")
            cmd.append(f"{value}")

        result = _run_subprocess(cmd, get_last_message=True)
        assert result

        experiment_path = Path(result)
        assert experiment_path.exists()
        return experiment_path


def _run_align(**kwargs):
    LOGGER.info(f"Run Aligner with args {kwargs}")

    if sys.platform == "win32" or kwargs.get("n_processes") == 1:
        run_align(**kwargs)
    else:
        cmd = ["python", "-m", "annotator.align"]
        for key, value in kwargs.items():
            if value is None or value == []:
                continue
            cmd.append(f"--{key}")
            if isinstance(value, list):
                for v in value:
                    cmd.append(f"{v}")
            else:
                cmd.append(f"{value}")

        _run_subprocess(cmd)


def _seg_processing(
    sega_path: Path,
    pretrained_models: tp.List[Path],
    langs_filter: tp.List,
    n_gpus: int = 0,
    sega_suffix: str = "",
):
    if not hasattr(_seg_processing, "annotator"):
        with EasyDSParser.lock:
            device = "cpu" if n_gpus == 0 else f"cuda:{get_freer_gpu(strict=False)}"
            annotator = AnnotatorEvaluationInterface(
                pretrained_models[0],
                pretrained_models[1],
                device=device,
                last_word_correction=True,
                audio_duration_limit=None,
            )
            setattr(_seg_processing, "annotator", annotator)

    if langs_filter is not None:
        if not any(lang in sega_path.as_posix() for lang in langs_filter):
            return

    new_sega_path = sega_path.with_suffix(".TextGridStage3" + sega_suffix)
    # if new_sega_path.exists():
    #    return

    annotator = getattr(_seg_processing, "annotator")
    sega = annotator.process(sega_path=sega_path)
    sega.save(new_sega_path)
    # LOGGER.info(f"Save sega {new_sega_path.as_posix()}")


def _run_segs_correction(
    pretrained_models: tp.List[Path],
    output_dir: Path,
    langs_filter=None,
    flist_path=None,
    n_processes: int = 1,
    n_gpus: int = 0,
    sega_suffix: str = "",
):
    LOGGER.info("Run segmentation correction")
    func = init_method_from_config(
        _seg_processing,
        {
            "pretrained_models": pretrained_models,
            "langs_filter": langs_filter,
            "n_gpus": n_gpus,
            "sega_suffix": sega_suffix,
        },
    )
    parser = EasyDSParser(func=func)
    if flist_path is None:
        parser.run_in_dir(
            data_root=output_dir,
            file_extension=".TextGrid",
            n_processes=n_processes
            if n_gpus == 0
            else n_gpus * int(get_total_gpu_memory(0) // ANNOTATOR_TOTAL_GPU_MEM),
        )
    else:
        all_files = []
        for path in flist_path:
            lines = Path(path).read_text(encoding="utf-8").split("\n")
            for item in lines:
                item = Path(item.split("|")[0])
                if not item.is_absolute():
                    item = Path(path).parent / item
                if item.exists():
                    all_files.append(item.as_posix())

        parser.run_from_path_list(
            all_files,
            n_processes=n_processes
            if n_gpus == 0
            else n_gpus * int(get_total_gpu_memory(0) // ANNOTATOR_TOTAL_GPU_MEM),
        )


def _get_speakers_profile(
    data_root: Path,
    output_dir: Path,
    speakers_filter: tp.Optional[tp.List[str]] = None,
):
    speakers_profile_path = data_root / "speakers.yml"
    if not speakers_profile_path.exists():
        LOGGER.warning(f"'speakers.yml' not found in path {data_root.as_posix()}")
        speakers_profile_path = None

    if speakers_profile_path:
        speakers_profile = {}
        speakers_cfg = Config.create_from_file(speakers_profile_path)
        for name, meta in speakers_cfg.items():
            if speakers_filter and name not in speakers_filter:
                continue

            speaker_root = data_root / meta["root"]
            if not speaker_root.exists():
                raise FileNotFoundError(speaker_root)
            if "file_list" in meta:
                if isinstance(meta["file_list"], str):
                    file_path = speaker_root / meta["file_list"]
                    if not file_path.exists():
                        raise FileNotFoundError(file_path)
                elif isinstance(meta["file_list"], list):
                    flist: tp.List[str] = []
                    for item in meta["file_list"]:
                        file_path = speaker_root / item
                        lines = Path(file_path).read_text(encoding="utf-8").split("\n")
                        for line in lines:
                            path = Path(line)
                            if not path.is_absolute():
                                path = file_path.parents[0] / path
                            flist.append(path.as_posix())
                    file_path = speaker_root / "all_meta.txt"
                    file_path.write_text("\n".join(flist), encoding="utf-8")
                    meta["file_list"] = file_path.as_posix()
                else:
                    raise ValueError("file_list has an unknown format")
            elif "subfolders" in meta:
                flist: tp.List[str] = []  # type: ignore
                for folder in meta["subfolders"]:
                    folder_path = speaker_root / folder
                    if not folder_path.exists():
                        raise FileNotFoundError(
                            f"Path {folder_path.as_posix()} does not exist!"
                        )
                    flist += list(folder_path.glob("*.wav"))

                flist = [Path(item).as_posix() for item in flist]
                file_path = speaker_root / "all_meta.txt"
                file_path.write_text("\n".join(flist), encoding="utf-8")
                meta["file_list"] = file_path.as_posix()

            speakers_profile[name] = meta
    else:
        speaker_root = data_root
        profile = {"root": ""}
        if (speaker_root / "meta.txt").exists():
            profile["file_list"] = speaker_root / "meta.txt"
        speakers_profile = {f"{output_dir.name}": profile}

    if speakers_filter:
        if any(speaker not in speakers_profile for speaker in speakers_filter):
            raise KeyError(f"Filter is not valid! {speakers_filter}")

        speakers_profile = {
            k: v for k, v in speakers_profile.items() if k in speakers_filter
        }

    return speakers_profile


def _get_multilang_speakers_profile(
    data_root: Path,
    langs_filter: tp.Optional[tp.List[str]] = None,
    speakers_filter: tp.Optional[tp.List[str]] = None,
):
    languages_conf = Config.create_from_file(data_root / "languages.yml")
    total_speakers_profile = {}
    duplicates = []
    for lang, meta in languages_conf.items():
        if langs_filter and lang not in langs_filter:
            LOGGER.info(f"Skip '{lang}' language dataset")
            continue

        speakers_profile_path = data_root / meta["root"] / "speakers.yml"
        if not speakers_profile_path.exists():
            raise FileNotFoundError(speakers_profile_path)

        speakers_profile = _get_speakers_profile(
            data_root / meta["root"],
            speakers_profile_path,
        )
        for sp_name, sp_meta in speakers_profile.items():
            if sp_name in total_speakers_profile:
                duplicates.append(sp_name)
                sp_name = f"{sp_name}_{lang}"

            sp_meta["lang"] = lang
            sp_meta["root"] = Path(meta["root"]) / sp_meta["root"]
            total_speakers_profile[sp_name] = sp_meta

    for sp_name in set(duplicates):
        new_sp_name = f"{sp_name}_{total_speakers_profile[sp_name]['lang']}"
        total_speakers_profile[new_sp_name] = total_speakers_profile.pop(sp_name)

    if speakers_filter:
        if any(speaker not in total_speakers_profile for speaker in speakers_filter):
            raise KeyError(f"Filter is not valid! {speakers_filter}")

        total_speakers_profile = {
            k: v for k, v in total_speakers_profile.items() if k in speakers_filter
        }

    return total_speakers_profile


def _calc_statistics(
    output_dir: Path,
    speakers_profile: dict,
    lang: str,
):
    LOGGER.info("Calculate statistics over speakers")

    def get_duration(file_list: tp.List[Path]) -> float:
        total_dura = 0.0
        for path in file_list:
            sega = AudioSeg.load(path)
            total_dura += sega.duration
        return round(total_dura / 3600, 3)

    num_speakers = num_samples = 0
    total_duration = 0.0
    for name, meta in speakers_profile.items():
        if lang == "MULTILANG":
            segs_root = output_dir / meta.get("lang", lang) / name
        else:
            segs_root = output_dir / name
        if segs_root.exists():
            segs_list = list(segs_root.rglob("*.TextGridStage2"))
            duration = get_duration(segs_list)
            LOGGER.info(f"{name}: {len(segs_list)} samples, {duration} hours")

            if meta.get("multispeaker", False):
                sub_dirs = [x for x in segs_root.iterdir() if x.is_dir()]
                for item in tqdm(
                    sub_dirs, desc=f"Calculating statistics for '{name}' dataset"
                ):
                    segs_list = list(item.rglob("*.TextGridStage2"))
                    duration = get_duration(segs_list)
                    LOGGER.info(
                        f"    {item.name}: {len(segs_list)} samples, {duration} hours"
                    )
                    num_speakers += 1
                    num_samples += len(segs_list)
                    total_duration += duration
            else:
                num_speakers += 1
                num_samples += len(segs_list)
                total_duration += duration

    LOGGER.info(f"Num speakers: {num_speakers}")
    LOGGER.info(f"Num samples: {num_samples}")
    LOGGER.info(f"Total audio duration in hours: {np.round(total_duration, 3)}")


def _get_pretrained_models(
    output_dir: Path, lang: str, langs_filter=None
) -> tp.List[Path]:
    def extract_step(txt):
        return int(re.split(r"stage(\d+)", str(txt))[1])

    lang_dir = lang if langs_filter is None else "_".join(langs_filter)
    ckpt_path = (output_dir / "forced_alignment" / lang_dir).rglob("*.ckpt")
    ckpt_path = [item.parent for item in ckpt_path if "initial" not in item.name]

    pretrained_models = []
    for t in Counter(ckpt_path):
        files = list(t.glob("*.ckpt"))
        pretrained_models.append(max(files, key=lambda f: f.stat().st_mtime))

    pretrained_models.sort(key=extract_step)
    return pretrained_models


def main(
    data_root: Path,
    output_dir: Path,
    lang: str,
    langs_filter: tp.Optional[tp.List[str]] = None,
    speakers_filter: tp.Optional[tp.List[str]] = None,
    audio_sample_rate: tp.Optional[int] = None,
    audio_format: tp.Literal["wav", "flac", "opus"] = "wav",
    n_processes: int = 1,
    n_gpus: int = 0,
    n_workers_per_gpu: int = 1,
    num_samples: int = 0,
    batch_size: int = 16,
    max_epochs: tp.Union[int, tp.List[int]] = 0,
    start_step: int = 0,
    start_stage: int = 1,
    use_asr_transcription: bool = False,
    use_resampling_audio: bool = False,
    use_loudnorm_audio: bool = False,
    max_step: int = 4,
    sega_suffix: str = "",
    pretrained_models: tp.Optional[tp.List[Path]] = None,
    finetune_model: tp.Optional[Path] = None,
    resume_from_path: tp.Optional[Path] = None,
    asr_credentials: tp.Optional[Path] = None,
):
    with LoggingServer.ctx(output_dir, "runner"):

        if pretrained_models is None and (output_dir / "forced_alignment").exists():
            pretrained_models = _get_pretrained_models(output_dir, lang, langs_filter)

        if lang == "MULTILANG":
            speakers_profile = _get_multilang_speakers_profile(
                data_root, langs_filter, speakers_filter
            )
        else:
            speakers_profile = _get_speakers_profile(
                data_root,
                output_dir,
                speakers_filter,
            )

        # step 0
        if start_step <= 0 <= max_step and use_asr_transcription:
            LOGGER.info("Step: 0")

            for name, meta in speakers_profile.items():
                LOGGER.info(f"Run audio transcription of '{name}' dataset")
                run_audio_transcription(
                    data_root=data_root / meta["root"],
                    lang=meta.get("lang", lang),
                    asr_credentials=asr_credentials,
                    num_samples=num_samples,
                    n_processes=n_processes,
                    n_gpus=n_gpus,
                )
                LOGGER.info(f"Stop audio transcription of '{name}' dataset")

        # step 1
        seglist_paths = None
        if start_step <= 1 <= max_step:
            LOGGER.info("Step: 1")

            seglist_paths = []
            for name, meta in speakers_profile.items():
                LOGGER.info(f"Run processing of '{name}' dataset")

                speaker_data_root = data_root / meta["root"]
                if lang == "MULTILANG":
                    seg_dir = output_dir / meta.get("lang", lang) / name
                else:
                    seg_dir = output_dir / name

                file_list = meta.get("file_list")
                if file_list:
                    file_list = Path(file_list)
                    if not file_list.is_absolute():
                        file_list = speaker_data_root / file_list

                ret = run_seg_generator(
                    data_root=speaker_data_root,
                    output_dir=seg_dir,
                    lang=meta.get("lang", lang),
                    file_list=file_list,
                    num_samples=num_samples,
                    n_processes=n_processes,
                    n_gpus=n_gpus,
                    audio_sample_rate=audio_sample_rate,
                    audio_format=audio_format,
                    multispeaker_mode=meta.get("multispeaker", False),
                    use_asr_transcription=meta.get(
                        "use_asr_transcription", use_asr_transcription
                    ),
                    resampling_audio=use_resampling_audio,
                    loudnorm_audio=meta.get("use_loudnorm_audio", use_loudnorm_audio),
                )
                seglist_paths.append(ret["flist_path"])
                if speakers_filter is not None and meta.get("multispeaker", False):
                    speakers_filter += list(ret.get("speakers", []))

                LOGGER.info(f"Stop processing of '{name}' dataset")

        # step 2
        if start_step <= 2 <= max_step:
            if isinstance(max_epochs, int):
                max_epochs = [max_epochs] * 3
            if isinstance(max_epochs, list) and len(max_epochs) == 1:
                max_epochs = [max_epochs[0]] * 3

            experiment_path = pretrained_models[0] if pretrained_models else None
            if resume_from_path is not None:
                experiment_path = resume_from_path

            for stage in range(start_stage, 3):
                LOGGER.info(f"Step: 2 - stage{stage}")

                if pretrained_models and stage <= len(pretrained_models):
                    experiment_path = pretrained_models[stage - 1]
                else:
                    cfg_model_path, cfg_data_path = _update_fa_configs(
                        f"model_stage{stage}.yml",
                        f"data_stage{stage}.yml",
                        lang=lang,
                        output_dir=output_dir,
                        langs_filter=langs_filter,
                        speakers_filter=speakers_filter,
                        num_samples=num_samples,
                        n_processes=n_processes,
                        n_gpus=n_gpus,
                        n_workers_per_gpu=n_workers_per_gpu,
                        batch_size=batch_size,
                        max_epochs=sum(max_epochs[:stage]),
                        experiment_path=experiment_path,
                        sega_suffix=sega_suffix,
                        finetune_model=finetune_model,
                    )
                    experiment_path = _train_fa(
                        model_config_path=cfg_model_path,
                        data_config_path=cfg_data_path,
                        expr_suffix="all_speakers",
                    )

                _run_align(
                    data_root=output_dir,
                    ckpt_path=experiment_path,
                    stage=stage,
                    batch_size=max(batch_size // 2, 1),
                    num_samples=num_samples,
                    n_processes=n_processes if n_gpus <= 1 else n_workers_per_gpu,
                    n_gpus=n_gpus,
                    sega_suffix=sega_suffix,
                    flist_path=seglist_paths,
                )

        # step 3
        if start_step <= 3 <= max_step:
            if not pretrained_models or len(pretrained_models) < 2:
                pretrained_models = _get_pretrained_models(output_dir, lang, langs_filter)

            _run_segs_correction(
                pretrained_models=pretrained_models,
                output_dir=output_dir,
                langs_filter=langs_filter,
                flist_path=seglist_paths,
                n_processes=n_processes,
                n_gpus=n_gpus,
                sega_suffix=sega_suffix,
            )

        # step 4
        if start_step <= 4 <= max_step:
            _calc_statistics(output_dir, speakers_profile, lang)

        LOGGER.info("Data preparation completed!")


if __name__ == "__main__":
    """Launch examples.

    TTS datasets annotation:
        runner.py -d ../examples/simple_datasets/speech/SRC
                  -o ../examples/simple_datasets/speech/SEGS
                  --pretrained_models mfa_stage1_epoch=29-step=468750.pt mfa_stage2_epoch=59-step=937500.pt

    ASR datasets annotation:
        runner.py -d ../examples/simple_datasets/speech/SRC
                  -o ../examples/simple_datasets/speech/SEGS
                  --finetune_model mfa_stage1_epoch=29-step=468750.pt
                  --use_asr_transcription False
                  --use_resampling_audio False
                  --use_loudnorm_audio False

    """

    main(**parse_args().__dict__)
