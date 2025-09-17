import time
import logging

from abc import ABC

import torch.multiprocessing as mp

from speechflow.concurrency.abstract_worker import AbstractWorker
from speechflow.logging import trace, track_process

__all__ = ["ProcessWorker"]

LOGGER = logging.getLogger("root")


class ProcessWorker(AbstractWorker, mp.Process, ABC):
    """Base worker class which implements "multiprocessing" execution model."""

    def __init__(self, init_logger: bool = True, lock=None, daemon=None):
        mp.set_start_method("spawn", force=True)
        self._active = mp.Value("i", 0)
        self._started = mp.Value("i", 0)
        self._finished = mp.Value("i", 0)
        self._none_stop = mp.Value("i", 0)
        self._init_logger = init_logger
        self._lock = lock
        mp.Process.__init__(self, daemon=daemon)

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
        with self._finished.get_lock():
            self._finished.value = 0

    def finished(self):
        with self._started.get_lock():
            self._started.value = 0
        with self._finished.get_lock():
            self._finished.value = 1

    def is_started(self) -> bool:
        return self._started.value

    def is_finished(self) -> bool:
        return self._finished.value

    def set_none_stop_flag(self):
        with self._none_stop.get_lock():
            self._none_stop.value = 1

    def run(self):
        global LOCK
        global LOGGER

        if self._lock is not None:
            LOCK = self._lock

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
            except KeyboardInterrupt:
                self.deactivate()
            except Exception as e:
                if not self._none_stop.value:
                    self.deactivate()
                    raise e
                else:
                    LOGGER.info(
                        trace(self, e, message=f"restart {self.__class__.__name__}")
                    )

    def start(self, check_start: bool = True):
        try:
            super().start()
        except KeyboardInterrupt:
            self.terminate()
            raise KeyboardInterrupt
        except Exception as e:
            LOGGER.error(trace(self, e))
            raise e

        while check_start:
            if self.exitcode is not None and self.exitcode > 0:
                raise RuntimeError(f"{self.__class__.__name__} fails to start!")
            if self.is_started() or self.is_finished():
                break
            time.sleep(0.1)

    def finish(self, timeout: float = 1.0):
        if self.exitcode is not None:
            if self.exitcode > 0:
                LOGGER.error(
                    trace(
                        self,
                        message=f"Process {self.__class__.__name__} fails [exitcode={self.exitcode}]",
                    )
                )
            return

        try:
            self.deactivate()
            self._finish_timeout(timeout)
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

    def _finish_timeout(self, timeout: float, tick: float = 0.1):
        counter = 0
        while self.is_started():
            if 0 <= timeout <= counter:
                break
            time.sleep(tick)
            counter += tick


class DummyContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


LOCK = DummyContextManager()
