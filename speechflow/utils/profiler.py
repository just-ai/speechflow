import os
import time
import typing as tp
import logging

from dataclasses import dataclass
from enum import Enum

import numpy as np
import psutil

from speechflow.logging import log_profiler, log_to_console, trace

__all__ = ["Profiler", "MemoryProfiler", "ProfilerManager"]

LOGGER = logging.getLogger("root")


@dataclass
class Profiler:
    class Format(Enum):
        h = 1.0 / 60.0**2
        m = 1.0 / 60.0
        s = 1.0
        ms = 1000.0
        ns = 1000.0**2

    name: str = ""
    auto_logging: bool = True
    format: Format = Format.s
    gpu: bool = False
    enable: bool = True

    def __post_init__(self):
        self.reset()
        if self.auto_logging and not LOGGER.handlers:
            LOGGER.addHandler(logging.StreamHandler())
            LOGGER.setLevel(10)

    def __enter__(self):
        if self.enable:
            self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            self.total_time()
            self.logging()

    def reset(self):
        self._begin_time = time.perf_counter()
        self._tick_time = self._begin_time
        self._start_time = {}
        self._timings = {}

    def _add(self, name, begin_time):
        if self.gpu:
            import torch

            torch.cuda.synchronize()

        value = self._timings.setdefault(name, [0, 0])
        value[0] += time.perf_counter() - begin_time
        value[1] += 1

    def start(self, name: tp.Optional[str] = None):
        if self.enable:
            name = name if name else self.name
            self._start_time[name] = time.perf_counter()

    def stop(self, name: tp.Optional[str] = None):
        if self.enable:
            name = name if name else self.name
            self._add(name, self._start_time[name])

    def tick(self, name: tp.Optional[str] = None):
        if self.enable:
            name = name if name else self.name
            self._add(name, self._tick_time)
            self._tick_time = time.perf_counter()

    def total_time(self, name: tp.Optional[str] = None):
        if self.enable:
            name = name if name else self.name
            self._timings[name] = [0, 0]
            self._add(name, self._begin_time)

    def _get(self, name: tp.Optional[str] = None) -> tp.Tuple[float, int]:
        name = name if name else self.name
        value = self._timings.get(name)
        if value:
            return round((value[0] / value[1]) * self.format.value, 3), value[1]
        else:
            return (
                round((time.perf_counter() - self._begin_time) * self.format.value, 3),
                1,
            )

    def get_time(self, name: tp.Optional[str] = None) -> float:
        return self._get(name)[0] if self.enable else 0

    def get_counter(self, name: tp.Optional[str] = None) -> int:
        return self._get(name)[1] if self.enable else 0

    def logging(
        self,
        summary_writer: tp.Optional[tp.Any] = None,
        current_iter: tp.Optional[int] = 0,
    ):
        if not self.enable:
            return

        tm = []
        for key, value in self._timings.items():
            avg_time = round((value[0] / value[1]) * self.format.value, 2)
            tm.append(f"{key} time: {avg_time} {self.format.name}")
            if summary_writer:
                summary_writer.add_scalar(
                    f"{key} time ({self.format.name})", avg_time, global_step=current_iter
                )

        if self.auto_logging:
            LOGGER.info(trace(self, message="; ".join(tm)))

    @staticmethod
    def counter(multiple: Format = Format.s) -> float:
        return time.perf_counter() * multiple.value

    @staticmethod
    def sleep(seconds: float = 0):
        if seconds > 0:
            return time.sleep(seconds)


@dataclass
class MemoryProfiler:
    class Format(Enum):
        BYTE = 1.0
        KB = 1.0 / 1024**1
        MB = 1.0 / 1024**2
        GB = 1.0 / 1024**3
        TB = 1.0 / 1024**4

    name: str = ""
    auto_logging: bool = True
    format: Format = Format.BYTE
    enable: bool = True

    def __post_init__(self):
        self.reset()

    def __enter__(self):
        if self.enable:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            self.stop()
            self.logging()

    def _proc_memory_usage(self) -> float:
        return psutil.Process(os.getpid()).memory_info().rss

    def reset(self):
        self._before_memory_usage = self._proc_memory_usage() if self.enable else 0.0
        self._after_memory_usage = self._before_memory_usage

    def start(self):
        if self.enable:
            self.reset()

    def stop(self):
        if self.enable:
            self._after_memory_usage = self._proc_memory_usage()

    def diff(self) -> float:
        return round(
            (self._after_memory_usage - self._before_memory_usage) * self.format.value, 3
        )

    def total(self) -> float:
        return round(self._after_memory_usage * self.format.value, 3)

    def logging(
        self,
        summary_writer: tp.Optional[tp.Any] = None,
        current_iter: tp.Optional[int] = 0,
    ):
        if not self.enable:
            return

        if summary_writer:
            summary_writer.add_scalar(
                f"total memory ({self.format.name})",
                self.total(),
                global_step=current_iter,
            )
            summary_writer.add_scalar(
                f"diff memory ({self.format.name})", self.diff(), global_step=current_iter
            )

        if self.auto_logging:
            LOGGER.info(
                trace(
                    self,
                    message=f"total memory: {self.total()} {self.format.name}; "
                    f"diff memory: {self.diff()} {self.format.name}",
                )
            )


@dataclass
class ProfilerManager:
    name: str
    exec_time: bool = True
    memory: bool = True
    to_file: bool = True
    enable: bool = True
    group: tp.Optional[str] = None

    def __post_init__(self):
        self._profiler_time = Profiler(enable=self.exec_time, auto_logging=False)
        self._profiler_memory = MemoryProfiler(enable=self.memory, auto_logging=False)

    def __enter__(self):
        if self.enable:
            self._profiler_time.start()
            self._profiler_memory.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            self._profiler_time.stop()
            self._profiler_memory.stop()

            time = self._profiler_time.get_time()
            mem_diff = self._profiler_memory.diff()
            mem_total = self._profiler_memory.total()

            if self.to_file:
                log_profiler(
                    name=self.name,
                    group=self.group,
                    exec_time=time,
                    mem_diff=mem_diff,
                    mem_total=mem_total,
                )
            else:
                log_to_console(
                    f"{self.name}: time={time}; mem_diff={mem_diff}; {mem_total}={mem_total}"
                )


if __name__ == "__main__":
    profiler = Profiler(format=Profiler.Format.ms)
    profiler.start("test")

    time.sleep(0.05)
    profiler.tick("sleep1")

    for _ in range(10):
        time.sleep(0.1)
        profiler.tick("sleep2")

    time.sleep(0.5)
    profiler.total_time("total")

    profiler.logging()

    with Profiler("context"):
        time.sleep(0.75)

    with Profiler(auto_logging=False) as prof:
        time.sleep(0.5)
    print("time", prof.get_time())

    profiler.stop("test")
    print("time test", profiler.get_time("test"))

    with MemoryProfiler(format=MemoryProfiler.Format.MB) as prof:
        np.zeros(1000)
