import typing as tp
import logging

from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from os import environ as env
from pathlib import Path

import pytorch_lightning as pl

from speechflow.io import Config, change_config_file, check_path, tp_PATH, tp_PATH_LIST
from speechflow.logging import trace
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.dictutils import find_field
from speechflow.utils.gpu_info import get_freer_gpu

__all__ = ["train_arguments", "config_prepare"]

LOGGER = logging.getLogger("root")


def _gpu_allocation(devices: str) -> tp.Union[int, tp.MutableSequence[int]]:
    import torch

    if not torch.cuda.is_available():
        LOGGER.info("Torch not compiled with CUDA enabled")
        return 0
    else:
        if devices in ["auto", ["auto"]]:
            devices = [get_freer_gpu()]

        # reserving small memory on GPUs
        if isinstance(devices, tp.MutableSequence):
            for device_index in devices:
                globals()[f"temp_tensor_{device_index}"] = torch.tensor(
                    [0.0], device=f"cuda:{device_index}"
                )

        return devices


class CustomArgumentParser(ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        result = super().parse_args()

        if result.model_config_path is None and result.resume_from is None:
            raise AttributeError("Specify of --model_config_path or --resume_from")

        if result.resume_from:
            result.model_config_path = result.resume_from / "model.yml"
            result.data_config_path = list(result.resume_from.glob("data*.yml"))

            if not result.model_config_path.exists() or len(result.data_config_path) == 0:
                raise ValueError("Invalid path for resume from!")

        return result


class ConfigValidationAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if namespace.resume_from is None:
            if values is not None:
                setattr(namespace, self.dest, values)
            else:
                raise AttributeError(
                    "Specify of --resume_from or both of --model_config_path and --data_config_path"
                )
        else:
            LOGGER.info(f"Training will resume from experiment {namespace.resume_from}")


def train_arguments() -> "CustomArgumentParser":
    arguments_parser = CustomArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arguments_parser.add_argument(
        "-c",
        "--model_config_path",
        help="path to yaml config for model",
        type=Path,
        action=ConfigValidationAction,
    )
    arguments_parser.add_argument(
        "-cd",
        "--data_config_path",
        help="path to yaml configs for data",
        type=Path,
        nargs="+",
    )
    arguments_parser.add_argument(
        "-vs", "--value_select", help="select specific values", type=str, nargs="+"
    )
    arguments_parser.add_argument(
        "-r",
        "--resume_from",
        help="path to experiment to resume from",
        type=Path,
    )
    arguments_parser.add_argument(
        "-addr", "--data_server_address", help="address of data server", type=str
    )
    arguments_parser.add_argument(
        "-s",
        "--expr_suffix",
        help="suffix for experiment folder name",
        type=str,
    )
    arguments_parser.add_argument(
        "-d", "--data_root", help="data root directory", type=Path
    )
    arguments_parser.add_argument(
        "-bs",
        "--batch_size",
        help="batch size",
        type=int,
    )
    arguments_parser.add_argument(
        "-nproc",
        "--n_processes",
        help="number of workers for data processing",
        type=int,
    )
    arguments_parser.add_argument(
        "-ngpu", "--n_gpus", help="number of GPU device", type=int
    )
    return arguments_parser


def _set_device(cfg_model: Config):
    if pl.__version__ == "1.5.9":
        cfg_model["trainer"]["n_gpus"] = _gpu_allocation(cfg_model["trainer"]["n_gpus"])
    else:
        if "gpus" in cfg_model["trainer"]:
            cfg_model["trainer"]["accelerator"] = "gpu"
            cfg_model["trainer"]["devices"] = cfg_model["trainer"].pop("n_gpus")

        cfg_model["trainer"]["devices"] = _gpu_allocation(cfg_model["trainer"]["devices"])

        if (
            cfg_model["trainer"]["accelerator"] == "cpu"
            or cfg_model["trainer"]["devices"] == 0
        ):
            cfg_model["trainer"]["accelerator"] = "cpu"
            cfg_model["trainer"]["devices"] = 1


@check_path(assert_file_exists=True)
def config_prepare(
    model_config_path: tp_PATH,
    data_config_path: tp.Optional[tp.Union[Path, tp_PATH_LIST]] = None,
    value_select: tp.Optional[tp.List[str]] = None,
    resume_from: tp.Optional[Path] = None,
    expr_suffix: tp.Optional[str] = None,
    data_root: tp.Optional[Path] = None,
    batch_size: tp.Optional[int] = None,
    n_processes: tp.Optional[int] = None,
    n_gpus: tp.Optional[int] = None,
) -> tp.Tuple[Config, tp.Optional[tp.Union[Path, tp_PATH_LIST]]]:
    # ------------------------------------------------------------------------------------
    # ------------------ Prepare a model config file for the experiment ------------------
    # ------------------------------------------------------------------------------------

    if batch_size is not None:
        change_config_file(model_config_path, {"batch_size": batch_size})

    cfg_model = Config.create_from_file(model_config_path, value_select=value_select)

    env["MODEL_PROFILING"] = "1" if cfg_model.get("use_profiler", False) else ""

    _set_device(cfg_model)

    date_now = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
    cfg_model["experiment_name"] = f"{date_now}_{cfg_model['experiment_name']}"

    if expr_suffix:
        cfg_model["experiment_name"] += f"_{expr_suffix}"

    experiment_path = Path(cfg_model["dirs"]["logging"]) / cfg_model["experiment_name"]
    cfg_model.setdefault("experiment_path", experiment_path.as_posix())

    if cfg_model["trainer"].get("resume_from_checkpoint"):
        ckpt_path = ExperimentSaver.get_last_checkpoint(
            cfg_model["trainer"]["resume_from_checkpoint"]
        )

        new_ckpt_path = experiment_path / f"initial_checkpoint_{ckpt_path.name}"

        checkpoint = ExperimentSaver.load_checkpoint(ckpt_path)
        if checkpoint.get("callbacks"):
            checkpoint["callbacks"] = {
                k: v
                for k, v in checkpoint["callbacks"].items()
                if "ModelCheckpoint" not in k
            }
        ExperimentSaver.save_checkpoint(checkpoint, new_ckpt_path)

        cfg_model["trainer"]["resume_from_checkpoint"] = (
            new_ckpt_path.as_posix() if isinstance(new_ckpt_path, Path) else new_ckpt_path
        )

    if resume_from:
        assert model_config_path and data_config_path and resume_from.is_dir()
        cfg_model["experiment_path"] = resume_from.as_posix()
        cfg_model["experiment_name"] = resume_from.name
        ckpt_path = ExperimentSaver.get_last_checkpoint(resume_from)
        cfg_model["trainer"]["resume_from_checkpoint"] = (
            ckpt_path.as_posix() if isinstance(ckpt_path, Path) else ckpt_path
        )

    if cfg_model["trainer"].get("finetune_epochs"):
        checkpoint = ExperimentSaver.load_checkpoint(
            cfg_model["trainer"]["resume_from_checkpoint"]
        )
        cfg_model["trainer"]["max_epochs"] = checkpoint["epoch"] + cfg_model[
            "trainer"
        ].pop("finetune_epochs")
    else:
        cfg_model["trainer"].pop("finetune_epochs", None)

    if "finetune" in cfg_model:
        try:
            ckpt_path = cfg_model["finetune"].get("ckpt_path")
            _, cfg_model_temp = ExperimentSaver.load_configs_from_checkpoint(ckpt_path)
            cfg_model["model"]["params"].update(cfg_model_temp["model"]["params"])
        except KeyError as e:
            LOGGER.error(trace("model_config_prepare", e))

    # -----------------------------------------------------------------------------------
    # ------------------ Prepare a data config file for the experiment ------------------
    # -----------------------------------------------------------------------------------

    if isinstance(data_config_path, Path):
        data_config_path = [data_config_path]

    if any(item is not None for item in [data_root, n_processes, n_gpus]):
        assert data_config_path, ValueError("data config path is not set!")
        for idx, path in enumerate(data_config_path):
            change_config_file(
                path,
                {"data_root": data_root, "n_processes": n_processes, "n_gpus": n_gpus},
            )

    if (
        cfg_model["trainer"].get("resume_from_checkpoint") is not None
        or "finetune" in cfg_model
    ):
        assert data_config_path, ValueError("data config path is not set!")
        try:
            assert model_config_path and data_config_path
            ckpt_path = cfg_model["trainer"].get("resume_from_checkpoint")

            if "finetune" in cfg_model:
                ckpt_path = cfg_model["finetune"].get("ckpt_path")

            for idx, path in enumerate(data_config_path):
                cfg_data = Config.create_from_file(path, value_select=value_select)
                speaker_id_setter = find_field(cfg_data, "SpeakerIDSetter")
                if (
                    speaker_id_setter is not None
                    and speaker_id_setter.get("resume_from_checkpoint") is None
                ):
                    speaker_id_setter["resume_from_checkpoint"] = (
                        ckpt_path.as_posix() if isinstance(ckpt_path, Path) else ckpt_path
                    )
                    new_path = path.with_name(f"{path.stem}_sid.yml")
                    cfg_data.to_file(new_path)
                    data_config_path[idx] = new_path

        except Exception as e:
            LOGGER.warning(trace("model_config_prepare", e))

    return cfg_model, data_config_path
