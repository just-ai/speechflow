import copy
import typing as tp
import inspect
import logging
import functools

from functools import wraps
from os import environ as env

from speechflow.concurrency import process_worker
from speechflow.io import Config
from speechflow.utils.checks import str_to_bool

__all__ = [
    "init_method_from_config",
    "init_class_from_config",
    "get_default_args",
    "lazy_initialization",
]

LOGGER = logging.getLogger("root")


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def init_method_from_config(
    method, cfg: tp.Union[tp.Dict[str, tp.Any], Config], check_keys: bool = True
) -> tp.Callable:
    # TODO: Avoid copying when have tensors/models in cfg
    try:
        config = copy.deepcopy(cfg)
    except RuntimeError:
        config = cfg

    config_keys = {k for k in cfg.keys() if k not in ["type"]}
    config.update({"config": cfg, "conf": cfg, "cfg": cfg})

    init_params = inspect.signature(method).parameters
    params = get_default_args(method)

    init_keys = set(init_params.keys())
    if (
        check_keys
        and not init_keys >= config_keys
        and not (any(x in init_keys for x in ["args", "kwargs"]))
    ):
        raise ValueError(
            f"Config for {method.__name__} contains invalid or outdated parameters! {config_keys} -> {init_keys}"
        )

    for arg in init_params.keys():
        if arg in config:
            params[arg] = config[arg]

    if "kwargs" in init_params:
        unresolve_keys = init_keys.union(config_keys) - init_keys
        for key in unresolve_keys:
            params[key] = config[key]

    info = f"Set params for {method.__name__}({', '.join(init_params.keys())})"
    for key, field in params.items():
        info = info.replace(key, f"{key}={field}")

    return functools.partial(method, **params)


def init_class_from_config(
    cls,
    cfg: tp.Union[tp.Dict[str, tp.Any], Config, tp.MutableMapping],
    check_keys: bool = True,
) -> tp.Callable:
    config = copy.deepcopy(cfg)
    config_keys = {k for k in cfg.keys() if k not in ["type"]}

    if cls.__class__.__name__ == "ModelMetaclass":
        init_params = cls.model_fields
    else:
        init_params = inspect.signature(cls.__init__).parameters

    init_keys = list(init_params.keys())

    if len(init_keys) > 1 and init_keys[1] in ["cfg", "config", "params"]:
        config[init_keys[1]] = cfg
    else:
        init_keys = set(init_keys)
        if check_keys and "pipe" not in config_keys and not init_keys >= config_keys:
            unresolve_keys = init_keys.union(config_keys) - init_keys
            if "kwargs" in init_keys:
                config["kwargs"] = {arg: config[arg] for arg in unresolve_keys}
            else:
                raise ValueError(
                    f"Config for {cls.__name__} contains invalid or outdated parameters! "
                    f"{config_keys} -> {init_keys} | {unresolve_keys}"
                )

    params = {arg: config[arg] for arg in init_params.keys() if arg in config}

    # info = ", ".join(init_params.keys())
    # for key, field in params.items():
    #     info = info.replace(key, f"{key}={field}")
    # LOGGER.info(f"Set params for {cls.__name__}({info})")

    if "kwargs" in params:
        kwargs = params.pop("kwargs")
        params.update(kwargs)

    return functools.partial(cls, **params)


def lazy_initialization(func):
    @wraps(func)
    def decorated_func(*args, **kwargs):
        if str_to_bool(env.get("MEMORY_SAVE", "False")):
            none_attr_before = [k for k, v in args[0].__dict__.items() if v is None]
            args[0].create()
            none_attr_after = [k for k, v in args[0].__dict__.items() if v is None]

            res = func(*args, **kwargs)

            for attr in none_attr_before:
                if attr not in none_attr_after:
                    attr_value = getattr(args[0], attr)  # noqa: F841
                    del attr_value
                    setattr(args[0], attr, None)

            return res
        else:
            if not getattr(args[0], "__is_init", False):
                with process_worker.LOCK:
                    args[0].init()
                setattr(args[0], "__is_init", True)

            return func(*args, **kwargs)

    return decorated_func
