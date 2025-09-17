import copy
import typing as tp
import logging

from typing import Any

import torch
import pydantic

from speechflow.io import Config
from speechflow.utils.init import init_class_from_config

__all__ = ["BaseTorchModelParams", "BaseTorchModel"]

LOGGER = logging.getLogger("root")


class BaseTorchModelParams(pydantic.BaseModel):
    """Basic class for model parameters."""

    tag: str = "default"

    def __getitem__(self, key: str):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def model_post_init(self, __context: Any):
        pass

    @classmethod
    def create(
        cls,
        cfg: tp.Union[tp.MutableMapping, "BaseTorchModelParams"],
        strict_init: bool = True,
    ):
        if not isinstance(cfg, BaseTorchModelParams):
            return cls.init_from_config(cfg, strict_init)
        else:
            return cfg

    @classmethod
    def init_from_config(
        cls, cfg: tp.Union[tp.MutableMapping, Config], strict_init: bool = True
    ):
        cfg = cls.check_deprecated_params_recursive(cfg)

        if isinstance(cfg, Config):
            cfg = cfg.to_dict()

        params = cls()
        for key in list(cfg.keys()):
            if strict_init:
                assert hasattr(params, key), f"Parameter {key} not found!"
            else:
                if not hasattr(params, key):
                    LOGGER.warning(
                        f"Key '{key}' not found in initial params of {cls.__name__}"
                    )
                    cfg.pop(key)

        return cls(**cfg)

    @classmethod
    def init_from_parent_params(
        cls,
        parent_params: "BaseTorchModelParams",
        update_params: tp.Optional[tp.Dict] = None,
        strict: bool = True,
    ):
        params_as_dict = parent_params.to_dict()

        if update_params:
            update_params = cls.check_deprecated_params_recursive(update_params)

            if strict:  # validation of parameter names
                init_class_from_config(cls, update_params)

            params_as_dict.update(update_params)

        return init_class_from_config(cls, params_as_dict, check_keys=False)()

    def to_dict(self):
        return self.__dict__.copy()

    def copy(self, *args, **kwargs):
        return super().copy(*args, **kwargs, deep=True)

    def pop(self, key):
        value = self[key]
        del self.__dict__[key]
        return value

    @staticmethod
    def check_deprecated_params(cfg: dict) -> dict:
        return cfg

    @classmethod
    def check_deprecated_params_recursive(
        cls, cfg: tp.MutableMapping
    ) -> tp.MutableMapping:
        base = cls.__bases__
        for b in base:
            if hasattr(b, "check_deprecated_params_recursive"):
                cfg = b.check_deprecated_params_recursive(cfg)

        if hasattr(cls, "check_deprecated_params"):
            return cls.check_deprecated_params(cfg)
        else:
            return cfg


class BaseTorchModel(torch.nn.Module):
    """Basic class for torch model."""

    def __init__(self, params: BaseTorchModelParams):
        super().__init__()
        self.params = params
        self.initial_params = copy.deepcopy(params)
        self._register_load_state_dict_pre_hook(self.load_params)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_params(
        self, as_dict: bool = True, after_init: bool = False
    ) -> tp.Union[dict, BaseTorchModelParams]:
        params = self.params if after_init else self.initial_params
        return params.to_dict() if as_dict else params

    def load_params(self, state_dict: tp.Dict[str, torch.Tensor], *args):
        if "params" in state_dict:
            params: tp.Dict = state_dict.pop("params", {})
            for key, value in params.items():
                if self.get_params()[key] != value:
                    LOGGER.warning(f"Mismatch value for key {key}!")

        if "params_after_init" in state_dict:
            params_after_init: tp.Dict = state_dict.pop("params_after_init", {})
            for key, value in params_after_init.items():
                if self.get_params(after_init=True)[key] != value:
                    LOGGER.warning(f"Mismatch value for key {key}!")

        if not self.training:
            for key in list(state_dict.keys()):
                value = state_dict.pop(key)
                if "criterion" in key:
                    continue
                state_dict[key.replace("model.", "", 1)] = value

        return state_dict

    def inference(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == "__main__":

    class ModelParamsA(BaseTorchModelParams):
        a: float = 1.0

        def model_post_init(self, __context: tp.Any):
            if self.a <= 2.0:
                self.a = -1.0

    class ModelParamsB(ModelParamsA):
        b: float = 1.0

        def model_post_init(self, __context: tp.Any):
            super().model_post_init(__context)

            if self.b >= 2.0:
                self.b = 100.0

    print(ModelParamsB(**{"a": 2.0, "b": 2.0}).json())
