import sys
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
from speechflow.io import Config, tp_PATH
from speechflow.logging import trace
from speechflow.logging.server import LoggingServer
from speechflow.training import ExperimentSaver, LightningEngine, Optimizer
from speechflow.training.lightning_callbacks import GradNormCallback
from speechflow.training.utils.config_prepare import config_prepare, train_arguments
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler
from tts import forced_alignment
from tts.forced_alignment.callbacks import AligningVisualisationCallback

LOGGER = logging.getLogger("root")


def _load_pretrain(cfg_model: Config):
    ckpt_path = Path(cfg_model["model"]["init_from"].get("ckpt_path", ""))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path.as_posix()} not found")

    LOGGER.warning(f"Loading {ckpt_path.as_posix()}")
    ckpt = ExperimentSaver.load_checkpoint(ckpt_path)
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}

    if cfg_model["model"]["params"].n_langs < len(ckpt["lang_id_map"]):
        cfg_model["model"]["params"].n_langs = len(ckpt["lang_id_map"])
    if cfg_model["model"]["params"].n_speakers < len(ckpt["speaker_id_map"]):
        cfg_model["model"]["params"].n_speakers = len(ckpt["speaker_id_map"])

    model_cls = getattr(forced_alignment, cfg_model["model"]["type"])
    model = init_class_from_config(model_cls, cfg_model["model"]["params"])()

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        LOGGER.error(trace("train", e))

        remove_modules = ["embedding", "encoder", "_emb", "_proj"]
        remove_modules = cfg_model["model"]["init_from"].get(
            "remove_modules", remove_modules
        )

        state_dict = {
            k: v for k, v in state_dict.items() if all(m not in k for m in remove_modules)
        }
        model.load_state_dict(state_dict, strict=False)
        LOGGER.info(f"List of initialized layers: {list(state_dict.keys())}")

    return model


def train(cfg_model: Config, data_loaders: tp.Dict[str, DataLoader]):
    experiment_path = Path(cfg_model["experiment_path"])

    pl.seed_everything(cfg_model.get("seed"))

    dl_train, dl_valid = data_loaders.values()

    batch_processor_cls = getattr(forced_alignment, cfg_model["batch"]["type"])
    batch_processor = init_class_from_config(batch_processor_cls, cfg_model["batch"])()

    lang = dl_train.client.find_info("lang")
    text_proc = TTSTextProcessor(lang=lang)
    cfg_model["model"]["params"].alphabet_size = text_proc.alphabet_size
    cfg_model["model"][
        "params"
    ].n_symbols_per_token = text_proc.num_symbols_per_phoneme_token

    speaker_id_handler = dl_train.client.find_info("SpeakerIDSetter")
    if speaker_id_handler is not None:
        cfg_model["model"]["params"].n_langs = speaker_id_handler.n_langs
        cfg_model["model"]["params"].n_speakers = speaker_id_handler.n_speakers
        lang_id_map = speaker_id_handler.lang2id
        speaker_id_map = speaker_id_handler.speaker2id
    else:
        lang_id_map = None
        speaker_id_map = None

    if "init_from" in cfg_model["model"]:
        model = _load_pretrain(cfg_model)
    else:
        model_cls = getattr(forced_alignment, cfg_model["model"]["type"])
        model = init_class_from_config(model_cls, cfg_model["model"]["params"])()

    criterion_cls = getattr(forced_alignment, cfg_model["loss"]["type"])
    criterion = init_class_from_config(criterion_cls, cfg_model["loss"])()

    optimizer = init_class_from_config(Optimizer, cfg_model["optimizer"])(model=model)

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
            "alphabet": text_proc.alphabet,
        }
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
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        GradNormCallback(),
        AligningVisualisationCallback(),
        saver.get_checkpoint_callback(
            cfg=cfg_model["checkpoint"], prefix=cfg_model["experiment_name"]
        ),
    ]

    if "early_stopping" in cfg_model:
        early_stop_callback = init_class_from_config(
            pl.callbacks.EarlyStopping, cfg_model["early_stopping"]
        )()
        callbacks.append(early_stop_callback)

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
    data_config_path: tp.Optional[tp_PATH] = None,
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
        train.py -c=../configs/2stage/model_stage1.yml -cd=../configs/2stage/data_stage1.yml -vs debug
        train.py -c=../configs/2stage/model_stage2.yml -cd=../configs/2stage/data_stage2.yml -vs debug

    """
    args = train_arguments().parse_args()
    print(main(**args.__dict__))
