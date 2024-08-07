import shutil
import typing as tp
import logging

from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from os import environ as env
from pathlib import Path

import pytorch_lightning as pl

from speechflow.io import Config, check_path, tp_PATH, tp_PATH_LIST
from speechflow.logging import trace
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.dictutils import find_field
from speechflow.utils.gpu import get_freer_gpu

__all__ = ["train_arguments", "model_config_prepare"]

LOGGER = logging.getLogger()


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
def model_config_prepare(
    model_config_path: tp_PATH,
    data_config_path: tp.Optional[tp.Union[Path, tp_PATH_LIST]] = None,
    value_select: tp.Optional[tp.List[str]] = None,
    resume_from: tp.Optional[Path] = None,
    expr_suffix: tp.Optional[str] = None,
) -> Config:
    model_cfg = Config.create_from_file(model_config_path, value_select=value_select)

    env["MODEL_PROFILING"] = "1" if model_cfg.get("use_profiler", False) else ""

    _set_device(model_cfg)

    date_now = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
    model_cfg["experiment_name"] = f"{date_now}_{model_cfg['experiment_name']}"

    if expr_suffix:
        model_cfg["experiment_name"] += f"_{expr_suffix}"

    experiment_path = Path(model_cfg["dirs"]["logging"]) / model_cfg["experiment_name"]
    model_cfg.setdefault("experiment_path", experiment_path.as_posix())

    if model_cfg["trainer"].get("resume_from_checkpoint"):
        ckpt_path = ExperimentSaver.get_last_checkpoint(
            model_cfg["trainer"]["resume_from_checkpoint"]
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

        model_cfg["trainer"]["resume_from_checkpoint"] = new_ckpt_path.as_posix()

    if resume_from:
        assert model_config_path and data_config_path
        model_cfg["experiment_path"] = resume_from.as_posix()
        ckpt_path = ExperimentSaver.get_last_checkpoint(resume_from)
        model_cfg["trainer"]["resume_from_checkpoint"] = ckpt_path.as_posix()

    if (
        model_cfg["trainer"].get("resume_from_checkpoint") is not None
        or "finetuning" in model_cfg
    ):
        assert data_config_path
        try:
            assert model_config_path and data_config_path
            ckpt_path = model_cfg["trainer"].get("resume_from_checkpoint")

            if "finetuning" in model_cfg:
                ckpt_path = model_cfg["finetuning"].get("checkpoint")

            if isinstance(data_config_path, Path):
                data_config_path = [data_config_path]

            for path in data_config_path:
                data_cfg = Config.create_from_file(path)
                speaker_id_setter = find_field(data_cfg, "SpeakerIDSetter")
                if (
                    speaker_id_setter is not None
                    and speaker_id_setter.get("resume_from_checkpoint") is None
                ):
                    speaker_id_setter["resume_from_checkpoint"] = ckpt_path
                    shutil.copy(path, path.with_name(f"{path.name}_orig"))
                    data_cfg.to_file(path)

        except Exception as e:
            LOGGER.warning(trace("model_config_prepare", e))

    if model_cfg["trainer"].get("finetune_epochs"):
        checkpoint = ExperimentSaver.load_checkpoint(
            model_cfg["trainer"]["resume_from_checkpoint"]
        )
        model_cfg["trainer"]["max_epochs"] = checkpoint["epoch"] + model_cfg[
            "trainer"
        ].pop("finetune_epochs")
    else:
        model_cfg["trainer"].pop("finetune_epochs", None)

    if "finetuning" in model_cfg:
        ckpt_path = model_cfg["finetuning"].get("checkpoint")
        _, cfg_model_temp = ExperimentSaver.load_configs_from_checkpoint(ckpt_path)
        model_cfg["model"]["params"].update(cfg_model_temp["model"]["params"])

    return model_cfg
