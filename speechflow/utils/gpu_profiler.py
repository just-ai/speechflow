import sys
import logging

from functools import wraps
from os import environ as env

from speechflow.logging import log_profiler
from speechflow.utils.checks import str_to_bool
from speechflow.utils.profiler import Profiler

LOGGER = logging.getLogger("root")


def gpu_profiler(func):
    @wraps(func)
    def decorated_func(*args, **kwargs):
        if str_to_bool(env.get("MODEL_PROFILING", "False")):
            with Profiler(
                format=Profiler.Format.ms, auto_logging=False, gpu=True
            ) as prof:
                res = func(*args, **kwargs)
            func_name = f"{args[0].__class__.__name__}.{func.__name__}"
            log_profiler(
                func_name, prof.get_time(), mem_total=sys.getsizeof(func), group="MODULE"
            )
        else:
            res = func(*args, **kwargs)

        return res

    return decorated_func
