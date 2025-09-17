import uuid
import typing as tp
import logging
import argparse
import multiprocessing as mp

from collections import defaultdict
from dataclasses import dataclass

import psutil

from strenum import StrEnum

from speechflow.concurrency import ProcessWorker
from speechflow.data_pipeline.core import DataPipeline
from speechflow.data_server.patterns import ZMQPatterns, ZMQServer
from speechflow.data_server.pool import WorkerPool
from speechflow.data_server.system_messages import DataClientMessages as DCM
from speechflow.data_server.system_messages import DataServerMessages as DSM
from speechflow.io import Config, check_path, tp_PATH
from speechflow.logging import is_verbose_logging, log_to_file, trace
from speechflow.utils.gpu_info import get_freer_gpu
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler
from speechflow.utils.serialize import Serialize
from speechflow.utils.sockopt import find_free_port

__all__ = ["DataServer", "SubscriberTypes"]

LOGGER = logging.getLogger("root")


class SubscriberTypes(StrEnum):
    CLIENT = "client"
    WORKER = "worker"
    LOADER = "loader"


@dataclass
class SamplingStatus:
    num_batch_in_processing: int = 0
    num_batch_send: int = 0
    is_last_batch: bool = False
    subset: str = None  # type: ignore


class DataServer(ProcessWorker):
    def __init__(
        self,
        data_pipeline: DataPipeline,
        n_processes: int = 1,
        n_gpus: tp.Union[int, tp.List[int]] = 0,
        server_addr: tp.Optional[str] = None,
        synchronize_loaders: bool = False,
    ):
        ProcessWorker.__init__(self)
        self._addr_for_clients = (
            server_addr if server_addr else f"127.0.0.1:{find_free_port()}"
        )
        self._addr_for_workers = f"127.0.0.1:{find_free_port()}"
        self._pipe = data_pipeline
        self._pipe_serialize = Serialize.dump(data_pipeline)
        self._n_processes = n_processes if n_processes else mp.cpu_count()
        self._zmq_server: ZMQServer = None  # type: ignore
        self._synchronize_loaders = synchronize_loaders
        self._work_queues: tp.Dict[str, SamplingStatus] = defaultdict(SamplingStatus)
        self._uid_map: tp.Dict[bytes, str] = {}
        self._sync_samplers = {}

        self._subscribers: tp.Dict[str, int] = {}
        self._info_for_worker = None
        self._info_for_loader = None

        self._batch_counter = 0
        self._total_batch_in_processing = 0
        self._timer = Profiler(auto_logging=False)

        self._gpus = self.init_gpus(n_gpus) if isinstance(n_gpus, int) else n_gpus

    @property
    def address(self) -> str:
        return self._addr_for_clients

    @property
    def num_processes(self) -> int:
        return self._n_processes

    @property
    def num_workers(self) -> int:
        return self._subscribers.get(SubscriberTypes.WORKER, 0)

    @staticmethod
    def init_gpus(num_gpu: int) -> tp.List[int]:
        import torch

        gpus = []
        for _ in range(num_gpu):
            gpus.append(get_freer_gpu())
            torch.tensor([0.0], device=f"cuda:{gpus[-1]}")

        return gpus

    @staticmethod
    @check_path(assert_file_exists=True)
    def init_from_config(
        file_path: tp_PATH,
        value_select: tp.Optional[tp.List[str]] = None,
        server_addr: tp.Optional[str] = None,
        config_section: str = "data_server",
    ) -> "DataServer":
        cfg = Config.create_from_file(
            file_path, section=config_section, value_select=value_select
        )

        if server_addr:
            cfg["server_addr"] = server_addr

        data_pipeline: DataPipeline = DataPipeline.init_from_config(
            file_path=file_path,
            value_select=value_select,
        )

        return init_class_from_config(DataServer, cfg)(data_pipeline=data_pipeline)

    def on_start(self):
        self._zmq_server = ZMQPatterns.server(
            self._addr_for_clients, self._addr_for_workers
        )

        self._pipe = Serialize.load(self._pipe_serialize)
        self._pipe.init_components()
        self._pipe.load_data(n_processes=self._n_processes)

        self._info_for_worker = self._pipe.get_info()
        self._info_for_loader = self._pipe.get_info(object_size_limit=0)
        LOGGER.info(trace(self, message=f"Start DataServer {self._addr_for_clients}"))

    def on_finish(self):
        self._zmq_server.close()
        LOGGER.info(trace(self, message=f"Finish DataServer {self._addr_for_clients}"))

    def status_info(self, timeout: int = 3600):
        if self._timer.get_time() > timeout:
            mem = psutil.virtual_memory()
            info = (
                "\n"
                f"\tsubscribers: {list(self._subscribers.items())}\n"
                f"\tbatches_prepared: {self._batch_counter}\n"
                f"\tcpu_utilization: {psutil.cpu_percent()}\n"
                f"\tavailable_memory: {round(mem.available * 100 / mem.total, 1)}%\n"
                f"\tused_virtual_memory: {round(mem.used / 1024 ** 3, 1)}GB\n"
                f"\tused_swap_memory: {round(psutil.swap_memory().used / 1024 ** 3, 1)}GB"
            )
            log_to_file(trace(self, info))
            self._timer.reset()

    def send_info_message(self, message, text: str, subset: tp.Optional[str] = None):
        subscriber_uid = self._get_subscriber_uid(message)
        client_uid = self._uid_map.get(subscriber_uid, uuid.uuid4().hex)[:6]
        info = f"[{client_uid}] info: {text}"

        response = []
        for m in message:
            if m and m[0] != 0:
                response.append(info.encode())
                break
            else:
                response.append(m)

        self._zmq_server.frontend_send_multipart(response)

        if is_verbose_logging():
            log_to_file(trace(self, f"{subset}: {info}" if subset else info))

    def is_reject_request(self, message, queue_info: SamplingStatus) -> bool:
        if self.num_workers == 0:
            self.send_info_message(message, DSM.NO_WORKERS, queue_info.subset)
            return True

        if self._total_batch_in_processing >= 4 * self.num_workers:
            self.send_info_message(message, DSM.OVERLOAD, queue_info.subset)
            return True

        if (
            not queue_info.is_last_batch
            and queue_info.num_batch_in_processing > self.num_workers
        ):
            self.send_info_message(message, DSM.QUEUE_EXCEEDED, queue_info.subset)
            return True

        if (
            queue_info.is_last_batch
            and queue_info.num_batch_in_processing > self.num_workers
        ):
            self.send_info_message(
                message,
                f"{DSM.EPOCH_ENDING}"
                f" [num_batch_in_processing={queue_info.num_batch_in_processing}]",
                queue_info.subset,
            )
            return True

        if queue_info.is_last_batch and queue_info.num_batch_in_processing == 0:
            self.send_info_message(message, DSM.EPOCH_COMPLETE, queue_info.subset)
            return True

        return False

    def gen_response(self, message):
        request = Serialize.load(message[-1])
        subscriber_uid = self._get_subscriber_uid(message)
        client_uid = request.get("client_uid")

        if client_uid is not None:
            self._uid_map.setdefault(subscriber_uid, client_uid)
        else:
            client_uid = uuid.uuid4().hex

        if request["message"] == DCM.INFO:
            response = {
                "subscriber_id": self._subscribers.setdefault(request["sub_type"], 0),
                "addr_for_workers": self._addr_for_workers,
                "subsets": self._info_for_loader.get("subsets", []),
            }
            if request["sub_type"] == SubscriberTypes.CLIENT:
                pass
            elif request["sub_type"] == SubscriberTypes.LOADER:
                response.update(self._info_for_loader)
                if self._synchronize_loaders:
                    samplers = self._sync_samplers.setdefault(client_uid, {})
                    for subset in self._pipe.subsets:
                        samplers[subset] = self._pipe[subset].sampler.copy()
            elif request["sub_type"] == SubscriberTypes.WORKER:
                response.update(self._info_for_worker)
                if self._gpus:
                    idx = response["subscriber_id"] % len(self._gpus)
                    response.update({"device": f"cuda:{self._gpus[idx]}"})
            else:
                raise RuntimeError(
                    f"Subscriber type {request['sub_type']} is not supported!"
                )

            message[-1] = Serialize.dump(response)
            self._zmq_server.frontend_send_multipart(message)
            self._subscribers[request["sub_type"]] += 1

        elif request["message"] == DCM.IS_READY:
            queue_info = self._work_queues[client_uid]
            if not self.is_reject_request(message, queue_info):
                self.send_info_message(
                    message,
                    f"{DSM.READY}: {queue_info.num_batch_send}",
                    queue_info.subset,
                )

        elif request["message"] == DCM.GET_BATCH:
            subset = request["subset_name"]
            batch_size = request["batch_size"]
            batch_num = request.get("batch_num", 1)

            queue_info = self._work_queues[client_uid]
            queue_info.subset = subset
            if self.is_reject_request(message, queue_info):
                return

            if self._synchronize_loaders:
                sampler = self._sync_samplers[client_uid][subset]
            else:
                sampler = self._pipe[subset].sampler

            batch_list = []
            for _ in range(batch_num):
                if self._total_batch_in_processing >= 4 * self.num_workers:
                    break

                batch_list.append(sampler.sampling(batch_size))

                if sampler.is_last_batch:
                    queue_info.is_last_batch = True
                    break

            message.insert(1, b"")

            for samples in batch_list:
                is_ok = self._zmq_server.backend_send_multipart(
                    message + Serialize.dumps(samples)
                )
                if is_ok:
                    queue_info.num_batch_in_processing += 1
                    self._total_batch_in_processing += 1

        elif request["message"] == DCM.EPOCH_COMPLETE:
            status = self._work_queues[client_uid]
            status.is_last_batch = False
            status.num_batch_in_processing = 0

        elif request["message"] in [DCM.ABORT, DCM.RESET]:
            status = self._work_queues[client_uid]
            status.num_batch_in_processing = 0
            status.num_batch_send = 0

            if request["message"] == DCM.ABORT:
                self.send_info_message(message, DSM.ABORT, status.subset)

            if request["message"] == DCM.RESET:
                status.is_last_batch = False
                self._total_batch_in_processing = 0
                self._pipe[request["subset_name"]].sampler.reset()
                self.send_info_message(message, DSM.RESET, status.subset)
                for item in self._sync_samplers.values():
                    item[request["subset_name"]].reset()

    def do_work_once(self):
        try:
            self._zmq_server.pool(timeout=10)

            if self._zmq_server.is_frontend_ready():
                message = self._zmq_server.frontend_recv_multipart()
                self.gen_response(message)

            if self._zmq_server.is_backend_ready():
                message = self._zmq_server.backend_recv_multipart()
                subscriber_uid = self._get_subscriber_uid(message)
                client_uid = self._uid_map.get(subscriber_uid, uuid.uuid4().hex)

                queue_info = self._work_queues.get(client_uid)
                if queue_info is None:
                    self.send_info_message(message, DSM.SKIP_BATCH)
                    return

                queue_info.num_batch_in_processing = max(
                    0, queue_info.num_batch_in_processing - 1
                )
                self._total_batch_in_processing = max(
                    0, self._total_batch_in_processing - 1
                )

                self._zmq_server.frontend_send_multipart(message)
                queue_info.num_batch_send += 1

                if queue_info.num_batch_in_processing == 0 and queue_info.is_last_batch:
                    self.send_info_message(message, DSM.EPOCH_COMPLETE, queue_info.subset)

                self._batch_counter += 1

            self.status_info()

        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            LOGGER.error(trace(self, e))

    @staticmethod
    def _get_subscriber_uid(message: tp.List[bytes]):
        client_uid = []
        for m in message:
            if m and m[0] != 0:
                break
            if m:
                client_uid.append(m)

        return client_uid[-1]


if __name__ == "__main__":
    """
    example:
        server.py -c=../../../tts_data/config_example.yml
    """

    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-c", "--config_path", help="path to yaml config", type=str, required=True
    )
    arguments_parser.add_argument(
        "-vs", "--value_select", help="select specific values", nargs="+", type=str
    )
    args = arguments_parser.parse_args()

    server = DataServer.init_from_config(**args.__dict__)
    worker_pool = WorkerPool(server_addr=server.address, n_processes=server.num_processes)

    server.start()
    worker_pool.start()

    server.join()

    worker_pool.finish()
    server.finish()
