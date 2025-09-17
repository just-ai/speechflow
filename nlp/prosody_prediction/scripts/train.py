import sys
import typing as tp
import logging

from pathlib import Path

import pytorch_lightning as pl

from pytorch_lightning import seed_everything

# required for multi-gpu training
THIS_PATH = Path(__file__).absolute()
ROOT = THIS_PATH.parents[3]
sys.path.append(ROOT.as_posix())

from nlp import prosody_prediction
from nlp.prosody_prediction.callbacks import ProsodyCallback
from speechflow.data_server.helpers import init_data_loader_from_config
from speechflow.data_server.loader import DataLoader
from speechflow.io import Config, tp_PATH, tp_PATH_LIST
from speechflow.logging import trace
from speechflow.logging.server import LoggingServer
from speechflow.training.lightning_engine import LightningEngine
from speechflow.training.optimizer import Optimizer
from speechflow.training.saver import ExperimentSaver
from speechflow.training.utils.config_prepare import config_prepare, train_arguments
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler

LOGGER = logging.getLogger("root")


def train(cfg_model: Config, data_loaders: tp.Dict[str, DataLoader]) -> str:
    experiment_path = Path(cfg_model["experiment_path"])

    seed_everything(cfg_model["seed"])

    dl_train, dl_valid = data_loaders.values()

    batch_processor_cls = getattr(prosody_prediction, cfg_model["batch"]["type"])
    batch_processor = init_class_from_config(batch_processor_cls, cfg_model["batch"])()

    model_cls = getattr(prosody_prediction, cfg_model["model"]["type"])
    model = init_class_from_config(model_cls, cfg_model["model"]["params"])()

    criterion_cls = getattr(prosody_prediction, cfg_model["loss"]["type"])
    criterion = init_class_from_config(criterion_cls, cfg_model["loss"])()
    criterion.set_weights(dl_train, cfg_model["model"]["params"]["n_classes"])

    optimizer = init_class_from_config(Optimizer, cfg_model["optimizer"])(model=model)

    saver = ExperimentSaver(
        expr_path=experiment_path,
        additional_files={
            "data.yml": dl_train.client.info["data_config_raw"],
            "model.yml": cfg_model.raw_file,
        },
    )
    saver.to_save.update({"dataset": dl_train.client.info["dataset"]})

    pl_engine: LightningEngine = LightningEngine(
        model=model,
        criterion=criterion,
        batch_processor=batch_processor,
        optimizer=optimizer,
        saver=saver,
    )

    callbacks = [
        ProsodyCallback(
            dl_valid,
            names=cfg_model["loss"]["names"],
            tokenizer_name=cfg_model["callbacks"]["ProsodyCallback"]["tokenizer_name"],
            n_classes=cfg_model["model"]["params"]["n_classes"],
        ),
        saver.get_checkpoint_callback(
            cfg=cfg_model["checkpoint"], prefix=cfg_model["experiment_name"]
        ),
    ]

    ckpt_path = cfg_model["trainer"].pop("resume_from_checkpoint", None)

    trainer = init_class_from_config(pl.Trainer, cfg_model["trainer"])(
        default_root_dir=experiment_path, callbacks=callbacks
    )

    with Profiler("training", format=Profiler.Format.h):
        trainer.fit(pl_engine, dl_train, dl_valid, ckpt_path=ckpt_path)

    LOGGER.info("Model training completed!")
    return experiment_path.as_posix()


def main(
    model_config_path: tp_PATH,
    data_config_path: tp.Optional[tp.Union[tp_PATH, tp_PATH_LIST]] = None,
    value_select: tp.Optional[tp.List[str]] = None,
    resume_from: tp.Optional[Path] = None,
    data_server_address: tp.Optional[str] = None,
    expr_suffix: tp.Optional[str] = None,
    **kwargs,
) -> str:
    cfg_model, data_config_path = config_prepare(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        value_select=value_select,
        resume_from=resume_from,
        expr_suffix=expr_suffix,
        **kwargs,
    )

    with LoggingServer.ctx(cfg_model["experiment_path"]):
        with init_data_loader_from_config(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            value_select=value_select,
            server_addr=data_server_address,
        ) as data_loaders:
            try:
                return train(cfg_model=cfg_model, data_loaders=data_loaders)
            except Exception as e:
                LOGGER.error(trace("main", e))
                raise e


if __name__ == "__main__":
    """
    example:
        train.py -c=../configs/model.yml -cd=../configs/data.yml

    """
    args = train_arguments().parse_args()
    print(main(**args.__dict__))
