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

from speechflow.data_pipeline.datasample_processors.tts_text_processors import (
    TTSTextProcessor,
)
from speechflow.data_server.helpers import init_data_loader_from_config
from speechflow.data_server.loader import DataLoader
from speechflow.io import Config, tp_PATH, tp_PATH_LIST
from speechflow.logging import trace
from speechflow.logging.server import LoggingServer
from speechflow.training import lightning_callbacks as sf_callbacks
from speechflow.training.saver import ExperimentSaver
from speechflow.training.utils.config_prepare import config_prepare, train_arguments
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler
from tts import vocoders
from tts.vocoders.vocos.modules import VOCOS_BACKBONES, VOCOS_FEATURES, VOCOS_HEADS

LOGGER = logging.getLogger("root")


def train(cfg_model: Config, data_loaders: tp.Dict[str, DataLoader]) -> str:
    experiment_path = Path(cfg_model["experiment_path"])

    pl.seed_everything(cfg_model.get("seed"))

    dl_train, dl_valid = data_loaders.values()

    batch_processor_cls = getattr(vocoders, cfg_model["batch"]["type"])
    batch_processor = init_class_from_config(batch_processor_cls, cfg_model["batch"])()

    speaker_id_handler = dl_train.client.find_info("SpeakerIDSetter")
    if speaker_id_handler is not None:
        lang_id_map = speaker_id_handler.lang2id
        speaker_id_map = speaker_id_handler.speaker2id
    else:
        lang_id_map = None
        speaker_id_map = None

    lang = (
        dl_train.client.info["data_config"].preproc.pipe_cfg.get("text", {}).get("lang")
    )
    if lang:
        text_proc = TTSTextProcessor(lang=lang)
        alphabet = text_proc.alphabet
    else:
        text_proc = None
        alphabet = None

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

    engine_type = cfg_model["engine"].class_name
    if "Vocos" in engine_type:
        pl_engine_cls = getattr(vocoders.vocos, engine_type)

        feat_cfg = cfg_model["model"].feature_extractor

        if feat_cfg.class_name == "TTSFeatures":
            feat_cfg.init_args.alphabet_size = text_proc.alphabet_size
            feat_cfg.init_args.n_symbols_per_token = (
                text_proc.num_symbols_per_phoneme_token
            )
            if speaker_id_handler is not None:
                feat_cfg.init_args.n_langs = speaker_id_handler.n_langs
                feat_cfg.init_args.n_speakers = speaker_id_handler.n_speakers
        else:
            if speaker_id_handler is not None:
                feat_cfg.init_args.n_langs = speaker_id_handler.n_langs
                feat_cfg.init_args.n_speakers = speaker_id_handler.n_speakers

        feat_cls, feat_params_cls = VOCOS_FEATURES[feat_cfg.class_name]
        feat_params = init_class_from_config(feat_params_cls, feat_cfg.init_args)()
        feat = feat_cls(feat_params)

        if feat_cfg.class_name == "TTSFeatures":
            saver.to_save["tts_model_params"] = feat.tts_model.get_params()
            saver.to_save["tts_model_params_after_init"] = feat.tts_model.get_params(
                after_init=True
            )

        backbone_cfg = cfg_model["model"].backbone
        backbone_cls, backbone_params_cls = VOCOS_BACKBONES[backbone_cfg.class_name]
        backbone_params = init_class_from_config(
            backbone_params_cls, backbone_cfg.init_args
        )()
        backbone = backbone_cls(backbone_params)

        head_cfg = cfg_model["model"].head
        head_cls, head_params_cls = VOCOS_HEADS[head_cfg.class_name]
        head_params = init_class_from_config(head_params_cls, head_cfg.init_args)()
        head = head_cls(head_params)

        pl_engine = init_class_from_config(pl_engine_cls, cfg_model["engine"].init_args)(
            feat, backbone, head, batch_processor, saver
        )
    else:
        raise NotImplementedError

    callbacks = [
        saver.get_checkpoint_callback(
            cfg=cfg_model["checkpoint"], prefix=cfg_model["experiment_name"]
        ),
    ]

    if cfg_model.get("callbacks"):
        for callback_name, callback_cfg in cfg_model["callbacks"].items():
            if hasattr(vocoders.callbacks, callback_name):
                cls = getattr(vocoders.callbacks, callback_name)
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
        python -W ignore train.py -c=configs/vocos/mel_bigvgan.yml -cd=configs/vocos/mel_bigvgan_data_24khz.yml
        python -W ignore train.py -c=configs/vocos/mel_bigvgan.yml -cd=configs/vocos/mel_bigvgan_data_24khz.yml -vs debug

        # When training on multiple GPUs you need to set the flag NCCL_P2P_DISABLE:

        NCCL_P2P_DISABLE=1 python -W ignore train.py

    """
    args = train_arguments().parse_args()
    print(main(**args.__dict__))
