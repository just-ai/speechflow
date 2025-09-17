import typing as tp
import logging

import torch

from torch import nn

from speechflow.io import Config
from speechflow.training.base_model import BaseTorchModel
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.init import init_class_from_config

LOGGER = logging.getLogger("root")

__all__ = ["prepare_model_for_warmstart", "prepare_model_for_finetune"]


def prepare_model_for_warmstart(
    model: BaseTorchModel, warmstart_config: Config
) -> BaseTorchModel:
    ckpt_path = warmstart_config.get("checkpoint")
    include_layers = warmstart_config.get("include_layers")
    exclude_layers = warmstart_config.get("exclude_layers")

    if ckpt_path is not None:
        ckpt = ExperimentSaver.load_checkpoint(ckpt_path)
        pretrained_dict = ckpt["state_dict"]
    else:
        pretrained_dict = {}

    model_dict = model.state_dict()
    update_dict = {
        k[6:]: v
        for k, v in pretrained_dict.items()
        if k[6:] in model_dict and v.shape == model_dict[k[6:]].shape
    }
    if include_layers:
        include_layers = (
            [include_layers]
            if not isinstance(include_layers, tp.MutableSequence)
            else include_layers
        )
        update_dict = {
            k: v
            for k, v in update_dict.items()
            if any(in_layer in k for in_layer in include_layers)
        }

    if exclude_layers:
        exclude_layers = (
            [exclude_layers]
            if not isinstance(exclude_layers, tp.MutableSequence)
            else exclude_layers
        )
        update_dict = {
            k: v
            for k, v in update_dict.items()
            if all(ex_layer not in k for ex_layer in exclude_layers)
        }

    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    return model


def prepare_model_for_finetune(
    model_cls: tp.Callable, cfg_finetune: Config, model_params: Config = None
) -> BaseTorchModel:
    # TODO: Move to model utils
    """
    <config>
    finetuning:
        modules:
            postnet:
                reinitialize: True
            decoder:
    <config>
    """
    ckpt_path = cfg_finetune.get("ckpt_path")
    if ckpt_path is not None:
        ckpt = ExperimentSaver.load_checkpoint(ckpt_path)

        if model_params is None:
            model_params = ckpt["params"]
        else:
            model_params["n_langs"] = ckpt["params"]["n_langs"]
            model_params["n_speakers"] = ckpt["params"]["n_speakers"]
            model_params["max_input_length"] = ckpt["params"].get("max_input_length")
            model_params["max_output_length"] = ckpt["params"].get("max_output_length")

        model = init_class_from_config(model_cls, model_params)()
        model.eval()
        model.load_state_dict(ckpt["state_dict"], strict=cfg_finetune.get("strict", True))
    else:
        raise AttributeError(
            "Checkpoint for finetuning is not provided. "
            "Turn off finetuning mode removing `finetuning` key from config or pass ckpt path."
        )

    if cfg_finetune.get("modules") is None:
        return model

    for param in model.parameters():
        param.requires_grad = False

    modules_to_unfreeze: tp.Dict[str, tp.Dict] = cfg_finetune["modules"]
    for module_name, settings in modules_to_unfreeze.items():
        reinitialize = settings.get("reinitialize", False)
        include_submodules = settings.get("include", [])
        exclude_submodules = settings.get("exclude", [])
        module: nn.Module = getattr(model, module_name)
        for name, param in module.named_parameters():
            if include_submodules:
                if not any(item in name for item in include_submodules):
                    continue
            if exclude_submodules:
                if any(item in name for item in exclude_submodules):
                    continue

            param.requires_grad = True
            if reinitialize:
                # TODO: reccurent networks? LSTM, e.g. has weight_h etc, is it ok? Should be orthogonal.
                # Its better not to use yet
                if "weight" in name:
                    # TODO: which strategy to use?
                    if param.data.ndim > 1:
                        torch.nn.init.xavier_uniform_(param.data)
                    else:
                        LOGGER.info(
                            f"trainable params: {name} with shape {list(param.data.shape)} not reinitialize"
                        )
                elif "bias" in name:
                    param.data.fill_(0)

            LOGGER.info(f"trainable params: {name} [reinitialize={reinitialize}]")

    return model
