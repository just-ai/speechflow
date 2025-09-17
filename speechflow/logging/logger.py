import json
import socket
import typing as tp
import logging

from multiprocessing import current_process
from os import environ as env
from pathlib import Path

import numpy as np

from speechflow.data_server.patterns import ZMQPatterns
from speechflow.logging import is_verbose_logging, set_verbose_logging
from speechflow.logging.filters import set_logging_filters
from speechflow.logging.server import ProcessData, ProfilerData

__all__ = ["create_logger"]

LOGGER = logging.getLogger("root")


def _check_if_port_used(addr: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        host, port = addr.split(":")
        return s.connect_ex((host, int(port))) == 0


def _create_log_file(log_file: Path, log_name: str) -> Path:
    if not log_file.exists():
        if log_file.suffix != ".txt":
            log_file /= "log.txt"
        log_dir = log_file.parents[0]
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        if log_file.is_dir():
            log_file /= "log.txt"

    return log_file.with_name(f"{log_name}_{log_file.name}")


def _get_formatter():
    processname = current_process().name
    formatter = logging.Formatter(
        f"[{processname}] %(asctime)s:%(levelname)s:%(message)s"
    )
    return formatter


def _get_file_handler(addr, formatter, level):
    file_handler = ZeroMQFileHandler(addr)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    return file_handler


class ZeroMQFileHandler(logging.StreamHandler):
    def __init__(self, addr: str):
        super().__init__()
        self._zmq_client = ZMQPatterns.async_client(addr)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def emit(self, record: logging.LogRecord):
        if isinstance(record.msg, ProfilerData):
            message = record.msg
        elif isinstance(record.msg, ProcessData):
            message = record.msg
        else:
            message = self.format(record)

        self._zmq_client.send(message)

    def close(self):
        self._zmq_client.close()

    def log(self, message: tp.Any):
        if not isinstance(message, (str, ProfilerData, ProcessData)):
            message = json.dumps(message, indent=4)

        self._zmq_client.send(message)


def create_logger(
    log_name: str = "root",
    log_file: tp.Optional[tp.Union[str, Path]] = None,
    console_level=logging.INFO,
    file_level=logging.INFO,
    use_server_logging: bool = True,
    use_file_logging: bool = True,
    use_console_logging: bool = True,
    use_verbose_logging: bool = False,
):
    if use_verbose_logging:
        set_verbose_logging()

    if is_verbose_logging():
        console_level = logging.DEBUG
        file_level = logging.DEBUG

    root_logger = logging.getLogger(log_name)

    # create logger
    formatter = _get_formatter()
    root_logger.setLevel(file_level)
    root_logger.handlers.clear()
    np.set_printoptions(precision=5)

    # server logging
    if use_server_logging:
        if not any(type(hd) == ZeroMQFileHandler for hd in root_logger.handlers):
            logger_server_addr = env.get("LoggingServerAddress")
            if logger_server_addr and _check_if_port_used(logger_server_addr):
                root_logger.addHandler(
                    _get_file_handler(
                        addr=logger_server_addr, formatter=formatter, level=file_level
                    )
                )

    # file logging
    if use_file_logging and log_file:
        if not any(type(hd) == logging.FileHandler for hd in root_logger.handlers):
            log_file = _create_log_file(Path(log_file), log_name)
            file_handler = logging.FileHandler(str(log_file))
            file_handler.setFormatter(formatter)
            file_handler.setLevel(file_level)
            root_logger.addHandler(file_handler)

    # console logging
    if use_console_logging:
        if not any(type(hd) == logging.StreamHandler for hd in root_logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(console_level)
            root_logger.addHandler(console_handler)

    if not is_verbose_logging():
        set_logging_filters(root_logger)
    else:
        LOGGER.debug("SET VERBOSE LOGGING")

    return root_logger
