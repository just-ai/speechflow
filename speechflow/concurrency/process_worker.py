import time
import logging

from abc import ABC

import torch.multiprocessing as mp

from speechflow.concurrency.abstract_worker import AbstractWorker
from speechflow.logging import trace, track_process

LOGGER = logging.getLogger("root")


class ProcessWorker(AbstractWorker, mp.Process, ABC):
    """Base worker class which implements "multiprocessing" execution model."""

    def __init__(self, init_logger: bool = True):
        mp.set_start_method("spawn", force=True)
        self._active = mp.Value("i", 0)
        self._started = mp.Value("i", 0)
        self._none_stop = mp.Value("i", 0)
        self._init_logger = init_logger
        mp.Process.__init__(self)

    def __del__(self):
        self.finish()

    def activate(self):
        with self._active.get_lock():
            self._active.value = 1

    def deactivate(self):
        with self._active.get_lock():
            self._active.value = 0

    def is_active(self) -> bool:
        return self._active.value

    def started(self):
        with self._started.get_lock():
            self._started.value = 1

    def finished(self):
        with self._started.get_lock():
            self._started.value = 0

    def is_started(self) -> bool:
        return self._started.value

    def set_none_stop_flag(self):
        with self._none_stop.get_lock():
            self._none_stop.value = 1

    def run(self):
        global LOGGER

        if self._init_logger:
            from speechflow.logging.logger import create_logger

            LOGGER = create_logger()
            track_process(self.__class__.__name__, self.pid, auto_terminate=True)
        else:
            LOGGER.addHandler(logging.StreamHandler())
            LOGGER.setLevel(10)

        self.activate()
        while self.is_active():
            try:
                self._run()
            except Exception as e:
                if not self._none_stop.value:
                    self.deactivate()
                    raise e
                else:
                    LOGGER.info(trace(e, message="restart run()"))

    def start(self, timeout: float = 1.0, tick: float = 0.2):
        try:
            super().start()
            self._start_timeout(timeout, tick)
        except KeyboardInterrupt:
            self.terminate()
            raise KeyboardInterrupt
        except Exception as e:
            LOGGER.error(trace(self, e))

    def finish(self, timeout: float = 1.0, tick: float = 0.2):
        try:
            self.deactivate()
            self._finish_timeout(timeout, tick)
            if self.is_alive():
                self.terminate()
        except KeyboardInterrupt:
            self.terminate()
            raise KeyboardInterrupt
        except Exception as e:
            LOGGER.error(trace(self, e))

    def terminate(self):
        try:
            super().terminate()
        except AttributeError:
            pass

    def _start_timeout(self, timeout: float, tick: float):
        time.sleep(timeout)

        counter = 0
        while not (self.is_started() and self.is_active()):
            if 0 < timeout < counter:
                break
            if self.exitcode is not None:
                if self.exitcode > 0:
                    raise RuntimeError(f"{self.__class__.__name__} fails to start!")
                else:
                    break
            time.sleep(tick)
            counter += tick

    def _finish_timeout(self, timeout: float, tick: float):
        time.sleep(timeout)

        counter = 0
        while self.is_started():
            if 0 < timeout < counter:
                break
            time.sleep(tick)
            counter += tick
