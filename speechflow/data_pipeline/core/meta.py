import typing as tp
import inspect
import threading


class Singleton(type):
    _instances: tp.Dict = {}
    _init: tp.Dict = {}

    def __init__(cls, name, bases, dct):
        cls._init[cls] = dct.get("__init__", None)

    def __call__(cls, *args, **kwargs):
        for key, field in kwargs.items():
            if isinstance(field, (list, set)):
                kwargs[key] = frozenset(field)

        # init = cls._init[cls]
        # if init is not None:
        #     key = (
        #         cls,
        #         frozenset(inspect.getcallargs(init, None, *args, **kwargs).items()),
        #     )
        # else:
        #     key = cls
        key = f"{threading.get_ident()}_{cls.__name__}"

        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)

        return cls._instances[key]
