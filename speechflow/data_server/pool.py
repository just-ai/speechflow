import time
import typing as tp
import logging
import argparse
import multiprocessing as mp

from speechflow.data_server.worker import BatchWorker
from speechflow.io import Config, check_path, tp_PATH
from speechflow.utils.init import init_class_from_config

__all__ = ["WorkerPool"]

LOGGER = logging.getLogger("root")


class WorkerPool:
    def __init__(
        self, server_addr: str, n_processes: int = 0, worker_type: tp.Any = BatchWorker
    ):
        n_processes = n_processes if n_processes else mp.cpu_count()
        self._workers = [worker_type(server_addr) for _ in range(n_processes)]

    @staticmethod
    @check_path(assert_file_exists=True)
    def init_from_config(
        file_path: tp_PATH,
        value_select: tp.Optional[tp.List[str]] = None,
        config_section: str = "data_server",
    ) -> "WorkerPool":
        cfg = Config.create_from_file(
            file_path, section=config_section, value_select=value_select
        )
        return init_class_from_config(WorkerPool, cfg)()

    def start(self, timeout: float = 0.1, tick: float = 0.05):
        for w in self._workers:
            w.start(timeout, tick)

        while True:
            if any([w.exitcode is not None and w.exitcode > 0 for w in self._workers]):
                raise RuntimeError(f"{self.__class__.__name__} fails to start!")
            if all([w.is_started() for w in self._workers]):
                break
            time.sleep(1)

    def join(self):
        for w in self._workers:
            w.join()

    def finish(self):
        for w in self._workers:
            w.finish()


if __name__ == "__main__":
    """
    example:
        worker.py --host=localhost --port=5000
    """

    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "--addr", help="host with data server", type=str, required=True
    )
    arguments_parser.add_argument(
        "--nproc",
        "--n_processes",
        help="number of workers for data processing",
        type=int,
        default=1,
    )
    args = arguments_parser.parse_args()

    worker_pool = WorkerPool(args.addr, args.n_processes)

    worker_pool.start()
    worker_pool.join()
    worker_pool.finish()
