import os
import json
import time
import typing as tp
import inspect
import logging
import traceback
import multiprocessing

from os import environ as env

from speechflow.utils.checks import str_to_bool

__all__ = [
    "trace",
    "log_to_console",
    "log_to_file",
    "log_profiler",
    "track_process",
    "set_verbose_logging",
    "is_verbose_logging",
]


def trace(
    self: tp.Optional = None,
    exception: tp.Optional[tp.Union[Exception, str]] = None,
    message: tp.Optional[str] = None,
    full: bool = True,
) -> str:
    """Trace function for logging erorrs and warnings.

    :param self: reference the class in which the error occurred
    :param exception: exception info (optional)
    :param message: debug message (optional)
    :param full: print full stack trace
    :return: string with information about the location of the error

    """
    try:
        if full:
            exc = traceback.format_exc()
            if "NoneType: None" not in exc:
                exception = exc

        if self is None:
            class_name = ""
        elif isinstance(self, str):
            class_name = self
        else:
            class_name = self.__name__ if type(self) == type else self.__class__.__name__
        tr_msg = f"[{class_name}][{inspect.stack()[1][3]}:{inspect.stack()[1][2]}]"
        if message:
            tr_msg += f": {message}"
        if exception:
            tr_msg += f": {exception}"

    except Exception as e:
        tr_msg = f"[trace] {e}"

    return tr_msg


def _formatter(message: str, time_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    process_name = multiprocessing.current_process().name
    st = time.strftime(time_format, time.gmtime())
    return f"[{process_name}] {st}:{message}"


def log_to_console(message: str):
    print(_formatter(message))


def log_to_file(message: tp.Any):
    from speechflow.logging.logger import ZeroMQFileHandler

    logger = logging.getLogger("root")

    if not isinstance(message, str):
        try:
            message = json.dumps(message, indent=4)
        except TypeError:
            message = str(message)

    for handler in logger.handlers:
        if isinstance(handler, ZeroMQFileHandler):
            handler.log(_formatter(message))


def log_profiler(
    name: str,
    exec_time: float = 0.0,  # seconds
    mem_diff: float = 0.0,  # bytes
    mem_total: float = 0.0,  # bytes
    group: tp.Optional[str] = None,
):
    from speechflow.logging.logger import ZeroMQFileHandler
    from speechflow.logging.server import ProfilerData

    logger = logging.getLogger("root")
    message = ProfilerData(name, exec_time, mem_diff, mem_total, group, pid=os.getpid())

    for handler in logger.handlers:
        if isinstance(handler, ZeroMQFileHandler):
            handler.log(message)


def track_process(name: str, pid: int, auto_terminate: bool = False):
    from speechflow.logging.logger import ZeroMQFileHandler
    from speechflow.logging.server import ProcessData

    logger = logging.getLogger("root")
    message = ProcessData(name=name, pid=pid, auto_terminate=auto_terminate)

    for handler in logger.handlers:
        if isinstance(handler, ZeroMQFileHandler):
            handler.log(message)


def set_verbose_logging():
    env["VERBOSE"] = "True"


def is_verbose_logging():
    return str_to_bool(env.get("VERBOSE", "False"))


if __name__ == "__main__":
    log_to_console(trace("main", message="test message"))
