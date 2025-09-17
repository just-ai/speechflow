import os
import enum
import typing as tp
import inspect

from copy import deepcopy as copy

from speechflow.data_pipeline.core.datasample import DataSample
from speechflow.io import Config
from speechflow.utils.init import init_method_from_config

__all__ = ["BaseDSProcessor", "ComputeBackend"]


class ComputeBackend(enum.Enum):
    notset = 0
    numpy = 1
    torch = 2
    librosa = 3
    torchaudio = 4
    nvidia = 5
    nemo = 6


class BaseDSProcessor:
    def __init__(
        self,
        pipe: tp.Tuple[str, ...] = (),
        pipe_cfg: Config = Config.empty(),
        backend: ComputeBackend = ComputeBackend.notset,
        device: str = "cpu",
    ):
        self.pipe = pipe
        self.pipe_cfg = pipe_cfg
        self.backend = backend
        self.device = device

        self.components = {}
        self.transform_params = {}
        for step_name in self.pipe:
            method_params = self.pipe_cfg.get(step_name, {})

            if "type" in method_params:
                method_name = method_params.pop("type")
            else:
                method_name = step_name

            method = getattr(self, method_name)

            handler = init_method_from_config(method, method_params)
            self.components[step_name] = handler

            params = copy(handler.keywords)

            params.update(method_params)
            self.transform_params[step_name] = copy(params)  # type: ignore

    @staticmethod
    def get_config_from_locals(ignore: tp.Optional[tp.List[str]] = None) -> Config:
        current_frame = inspect.currentframe()
        if current_frame:
            frame = current_frame.f_back
            local = frame.f_locals
        else:
            local = {}

        ignore = ([] if ignore is None else list(ignore)) + ["self"]
        args = {
            k: v
            for k, v in local.items()
            if k not in ignore and not k.startswith("__") and not isinstance(v, type)
        }

        if "kwargs" in args and isinstance(args["kwargs"], tp.Dict):
            args.update(args.pop("kwargs"))

        return Config(args)

    def logging_params(self, params: tp.Union[Config, tp.Dict[str, tp.Any]]):
        if isinstance(params, Config):
            params = params.to_dict()

        self.transform_params.update({self.__class__.__name__: params})

    def init(self):
        if "DEVICE" in os.environ:
            self.device = os.environ.get("DEVICE")

    def process(self, ds: DataSample):
        ds.transform_params.update(self.transform_params)

        if self.pipe:
            for handler in self.components.values():
                ds = handler(ds)  # type: ignore
                if ds is None:
                    raise RuntimeError(
                        f"Handler {handler} should return DataSample object."
                    )

        return ds.to_numpy()
