import sys
import pickle
import typing as tp
import logging

from pathlib import Path

import pytorch_lightning as pl

# required for multi-gpu training
THIS_PATH = Path(__file__).absolute()
ROOT = THIS_PATH.parents[3]
sys.path.append(ROOT.as_posix())
from speechflow.data_pipeline.datasample_processors import TTSTextProcessor
from speechflow.data_server.helpers import init_data_loader_from_config
from speechflow.data_server.loader import DataLoader
from speechflow.io import Config, tp_PATH, tp_PATH_LIST
from speechflow.logging import trace
from speechflow.logging.server import LoggingServer
from speechflow.training import lightning_callbacks as sf_callbacks
from speechflow.training.lightning_engine import LightningEngine
from speechflow.training.optimizer import Optimizer
from speechflow.training.saver import ExperimentSaver
from speechflow.training.utils.config_prepare import config_prepare, train_arguments
from speechflow.training.utils.finetuning import (
    prepare_model_for_finetune,
    prepare_model_for_warmstart,
)
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler
from tts import acoustic_models

LOGGER = logging.getLogger("root")


def update_model_config(cfg: Config, dl: DataLoader):
    section = dl.client.find_section("TTSTextProcessor")
    text_proc = TTSTextProcessor(lang=section["lang"])
    cfg["model"]["params"].alphabet_size = text_proc.alphabet_size
    cfg["model"]["params"].n_symbols_per_token = text_proc.num_symbols_per_phoneme_token

    ds_stat = dl.client.find_info("DatasetStatistics")
    if ds_stat:
        sr = dl.client.find_info("sample_rate")
        hop_len = dl.client.find_info("hop_len")
        cfg["model"]["params"].setdefault(
            "max_input_length", int(ds_stat.max_transcription_length * 1.1)
        )
        cfg["model"]["params"].setdefault(
            "max_output_length", int(ds_stat.max_audio_duration * sr / hop_len * 1.1)
        )

    speaker_id_handler = dl.client.find_info("SpeakerIDSetter")
    if speaker_id_handler is not None:
        cfg["model"]["params"].n_langs = speaker_id_handler.n_langs
        cfg["model"]["params"].n_speakers = speaker_id_handler.n_speakers
        lang_id_map = speaker_id_handler.lang2id
        speaker_id_map = speaker_id_handler.speaker2id
    else:
        lang_id_map = None
        speaker_id_map = None

    return cfg, lang_id_map, speaker_id_map, text_proc.alphabet


def train(cfg_model: Config, data_loaders: tp.Dict[str, DataLoader]) -> str:
    experiment_path = Path(cfg_model["experiment_path"])

    pl.seed_everything(cfg_model.get("seed"))

    dl_train, dl_valid = data_loaders.values()

    batch_processor_cls = getattr(acoustic_models, cfg_model["batch"]["type"])
    batch_processor = init_class_from_config(batch_processor_cls, cfg_model["batch"])()

    cfg_model, lang_id_map, speaker_id_map, alphabet = update_model_config(
        cfg_model, dl_train
    )

    model_cls = getattr(acoustic_models, cfg_model["model"]["type"])

    if hasattr(model_cls, "update_and_validate_model_params"):
        cfg_model = model_cls.update_and_validate_model_params(
            cfg_model,
            dl_train.client.info["data_config"],
        )

    if cfg_model.get("finetune") is not None:
        model = prepare_model_for_finetune(
            model_cls, cfg_model["finetune"], cfg_model["model"]["params"]
        )
    else:
        model = init_class_from_config(model_cls, cfg_model["model"]["params"])()

    if cfg_model.get("warmstart") is not None:
        model = prepare_model_for_warmstart(model, cfg_model["warmstart"])

    criterion_cls = getattr(acoustic_models, cfg_model["loss"]["type"])
    criterion = init_class_from_config(criterion_cls, cfg_model["loss"])()

    optimizer = init_class_from_config(Optimizer, cfg_model["optimizer"])(
        model=model, criterion=criterion
    )

    saver = ExperimentSaver(
        expr_path=experiment_path,
        additional_files={
            "data.yml": dl_train.client.info["data_config_raw"],
            "model.yml": cfg_model.raw_file,
        },
    )
    saver.to_save.update(
        {
            "lang_id_map": lang_id_map,
            "speaker_id_map": speaker_id_map,
            "alphabet": alphabet,
        }
    )

    if 1:
        saver.to_save["info"] = dl_train.client.info
    else:
        info_file_name = f"{experiment_path.name}_info.pkl"
        info_file_path = experiment_path / info_file_name
        info_file_path.write_bytes(pickle.dumps(dl_train.client.info))
        saver.to_save["files"]["info_file_name"] = info_file_name

    pl_engine: LightningEngine = LightningEngine(
        model=model,
        criterion=criterion,
        batch_processor=batch_processor,
        optimizer=optimizer,
        saver=saver,
        **cfg_model.get("engine", {}),
    )

    callbacks = [
        saver.get_checkpoint_callback(
            cfg=cfg_model["checkpoint"], prefix=cfg_model["experiment_name"]
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    if cfg_model.get("callbacks"):
        for callback_name, callback_cfg in cfg_model["callbacks"].items():
            if hasattr(acoustic_models.callbacks, callback_name):
                cls = getattr(acoustic_models.callbacks, callback_name)
            elif hasattr(sf_callbacks, callback_name):
                cls = getattr(sf_callbacks, callback_name)
            else:
                cls = getattr(pl.callbacks, callback_name)
            callback = init_class_from_config(cls, callback_cfg)()
            callbacks.append(callback)

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
        python -W ignore train.py -c=../configs/tts/forward_bigvgan.yml -cd=../configs/tts/tts_data_24khz.yml
        python -W ignore train.py -c=../configs/tts/forward_bigvgan.yml -cd=../configs/tts/tts_data_24khz.yml -vs debug

        # When training on multiple GPUs you need to set the flag NCCL_P2P_DISABLE:

        NCCL_P2P_DISABLE=1 python -W ignore train.py

    """
    args = train_arguments().parse_args()
    print(main(**args.__dict__))
