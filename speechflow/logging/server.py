import os
import time
import codecs
import typing as tp
import logging
import multiprocessing as mp

from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy as copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

from speechflow.concurrency.process_worker import ProcessWorker
from speechflow.data_server.patterns import ZMQPatterns
from speechflow.logging import trace
from speechflow.utils.serialize import Serialize
from speechflow.utils.sockopt import find_free_port

__all__ = [
    "ProfilerData",
    "ProcessData",
    "LoggingServer",
]

LOGGER = logging.getLogger("root")


@dataclass
class ProfilerData:
    name: str = None
    time: float = None  # seconds
    mem_diff: float = None  # bytes
    mem_total: float = None  # bytes
    group: str = None
    pid: int = None


@dataclass
class ProcessData:
    name: str = None
    pid: int = None
    auto_terminate: bool = False


class LoggingServer(ProcessWorker):
    def __init__(
        self,
        log_file: tp.Optional[tp.Union[str, Path]],
        log_name: str = "root",
        addr: tp.Optional[str] = None,
    ):
        from speechflow.logging.logger import _create_log_file
        from speechflow.utils.profiler import Profiler

        super().__init__(init_logger=False)
        port = find_free_port()
        self._log_file = log_file
        self._log_name = log_name
        self._addr = addr if addr else f"127.0.0.1:{port}"
        self._zmq_server = None
        self._root_logger = None
        self._named_logger = None
        self._msg_queue: mp.Queue = mp.Queue()
        self._msg_storage: tp.List[str] = []
        self._profiler_store: tp.Dict[str, tp.List[ProfilerData]] = defaultdict(list)
        self._track_process: tp.Dict[int, ProcessData] = defaultdict(ProcessData)
        self._timer = Profiler(auto_logging=False)
        self._info_logging_timeout = 3600

        os.environ["LoggingServerAddress"] = self._addr

        if self._log_file is not None:
            self._log_file = _create_log_file(Path(self._log_file), self._log_name)
            suffix = datetime.now().strftime("%d_%b_%Y_%H_%M_%S") + f"_{port}_"
            self._log_file = self._log_file.with_name(suffix + self._log_file.name)

    @property
    def name(self) -> str:
        return self._log_name

    @property
    def logger(self):
        return self._named_logger if self._named_logger else self._root_logger

    @property
    def info(self):
        return self.logger.info

    @property
    def warning(self):
        return self.logger.warning

    @property
    def error(self):
        return self.logger.error

    def start(self):
        from speechflow.logging.logger import create_logger

        try:
            if self.is_started():
                LOGGER.warning(trace(self, message="LoggingServer already started"))
                return

            super().start()

            self._named_logger = create_logger(self._log_name)
            self._write_message_to_file(
                trace(self, message=f"LoggingServer has been started at {self._addr}")
            )
        except Exception as e:
            self.finish()
            raise e

    def finish(self, timeout: float = 3.0):
        super().finish(timeout)
        if self._named_logger:
            self._named_logger.handlers = []

    @staticmethod
    @contextmanager
    def ctx(
        log_file: tp.Optional[tp.Union[str, Path]] = None,
        log_name: str = "root",
        disable: bool = False,
    ):
        if disable:
            yield logging
            return

        logging_server = LoggingServer(log_file, log_name)
        logging_server.start()

        try:
            yield logging_server

        except Exception as e:
            logging_server.error(trace("LoggingServer", e))
            raise e

        finally:
            time.sleep(2.0)
            logging_server.finish()

    def on_start(self):
        self._zmq_server = ZMQPatterns.server(addr_for_clients=self._addr)
        self._track_process[os.getpid()] = ProcessData(
            self.__class__.__name__, os.getpid(), True
        )

    def on_finish(self):
        self._zmq_server.close()
        self._write_message_to_file(trace(self, message="Finish LoggingServer"))
        self._kill_process()

    @staticmethod
    def _parse_raw_message(raw_message) -> str:
        return Serialize.load(raw_message[-1])

    def _write_message_to_file(self, message: str) -> None:
        self._msg_queue.put_nowait(message)

        if self._log_file is not None:
            try:
                with codecs.open(Path(self._log_file).as_posix(), "a", "utf-8") as file:
                    file.write(f"{message}\n")
            except Exception as e:
                print(e)

    def _update_profile_store(self, msg: ProfilerData):
        if msg.group is not None:
            key = f"{msg.group}|{msg.name}"
        else:
            key = msg.name

        self._profiler_store[key].append(msg)

    def _profile_info(self):
        info = []
        for func_name in sorted(list(self._profiler_store.keys())):
            data = self._profiler_store[func_name]
            all_pid = Counter([d.pid for d in data])
            info.append(f"\t{func_name}:")
            for pid in all_pid:
                data_by_pid = [d for d in data if d.pid == pid]
                avr_time = np.asarray([d.time for d in data_by_pid]).mean() * 1000
                avr_mem_total = (
                    np.asarray([d.mem_total for d in data_by_pid]).mean() / 1024**2
                )
                avr_mem_diff = (
                    np.asarray([d.mem_diff for d in data_by_pid]).mean() / 1024**2
                )
                info.append(
                    f"\t\tpid={pid} ({len(data_by_pid)}): "
                    f"time={'%.2f' % float(avr_time)}ms; "
                    f"mem_diff={'%.2f' % float(avr_mem_diff)}MB; "
                    f"mem_total={'%.2f' % float(avr_mem_total)}MB;"
                )

        if info:
            info = ["ProfilerInfo:"] + info
            self._write_message_to_file("\n".join(info))
            self._profiler_store = defaultdict(list)

    def _process_info(self):
        info = []
        for p in self._track_process.values():
            try:
                process = psutil.Process(p.pid)
                is_running = process.is_running()
                if is_running:
                    mem_total = process.memory_info().rss // 1024**2
                    cpu_usage = round(
                        process.cpu_percent(interval=0.1) / psutil.cpu_count(), 3
                    )
                else:
                    mem_total = cpu_usage = 0
                info.append(
                    f"\t{p.name}: pid={p.pid}; mem_total={mem_total}MB; cpu_usage={cpu_usage}%"
                )
            except Exception as e:
                self._write_message_to_file(str(e))

        if info:
            info = ["ProcessInfo:"] + info
            self._write_message_to_file("\n".join(info))

    def _kill_process(self):
        for pid in list(self._track_process.keys()):
            p = self._track_process.pop(pid)
            try:
                process = psutil.Process(p.pid)
                if p.auto_terminate and process.is_running():
                    for child_proc in process.children(recursive=True):
                        child_proc.kill()
                    process.kill()
                    self._write_message_to_file(f"kill process with pid {p.pid}")
            except Exception as e:
                self._write_message_to_file(str(e))

    def do_work_once(self):
        try:
            self._zmq_server.pool(timeout=1000)

            if self._zmq_server.is_frontend_ready():
                raw_message = self._zmq_server.frontend.recv_multipart()
                msg = self._parse_raw_message(raw_message)

                if isinstance(msg, ProfilerData):
                    self._update_profile_store(msg)
                elif isinstance(msg, ProcessData):
                    self._track_process[msg.pid] = msg
                else:
                    self._write_message_to_file(msg)

            if self._timer.get_time() > self._info_logging_timeout:
                self._profile_info()
                self._process_info()
                self._timer.reset()

        except KeyboardInterrupt as e:
            self._write_message_to_file(trace(self, "Interrupt received, stopping ..."))
            raise e
        except Exception as e:
            self._write_message_to_file(trace(self, e))

    def get_last_message(self) -> str:
        all_messages = self.get_all_messages()
        return all_messages[-1] if all_messages else ""

    def get_all_messages(self) -> tp.List[str]:
        while not self._msg_queue.empty():
            self._msg_storage.append(self._msg_queue.get(timeout=1))

        return copy(self._msg_storage)
