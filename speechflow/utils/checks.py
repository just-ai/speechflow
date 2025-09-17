import typing as tp
import inspect
import functools
import subprocess

__all__ = [
    "check_install",
    "check_ismethod",
    "check_isfunction",
    "str_to_bool",
]


def check_install(*args):
    try:
        subprocess.check_output(args, stderr=subprocess.STDOUT)
        return True
    except OSError:
        return False


def check_ismethod(method: tp.Callable) -> bool:
    while True:
        if isinstance(method, functools.partial):
            method = method.func
        else:
            break

    return inspect.ismethod(method)


def check_isfunction(method: tp.Callable) -> bool:
    while True:
        if isinstance(method, functools.partial):
            method = method.func
        else:
            break

    return inspect.isfunction(method)


def str_to_bool(val: str) -> bool:
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n',
    'no', 'f', 'false', 'off', and '0'.  Raises ValueError if 'val' is anything else.

    """
    if val is None:
        return False

    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0", ""):
        return False
    else:
        raise ValueError(f"invalid truth value {val!r}")
