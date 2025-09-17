import math
import typing as tp
import inspect
import logging
import argparse

from collections import deque
from threading import Event, RLock, Thread

from speechflow.data_pipeline.core import Batch
from speechflow.data_server.client import DataClient
from speechflow.data_server.server import SubscriberTypes
from speechflow.data_server.system_messages import DataClientMessages as DCM
from speechflow.data_server.system_messages import DataServerMessages as DSM
from speechflow.io import Config, check_path, tp_PATH
from speechflow.logging import is_verbose_logging, log_to_file, trace
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler
from speechflow.utils.serialize import Serialize

__all__ = ["DataLoader"]

LOGGER = logging.getLogger("root")


class DataLoader:
    def __init__(
        self,
        server_addr: str,
        subset_name: str,
        epoch_len: int = 0,
        batch_size: int = 1,
        min_batch_size: int = 0,
        drop_non_full: bool = False,
        non_stop: bool = True,
        pin_memory: bool = False,
        prefetch_on_gpu: bool = False,
        min_prefetch_factor: int = 50,
        max_prefetch_factor: int = 150,
    ):
        self._batch_client = DataClient(server_addr, sub_type=SubscriberTypes.LOADER)
        self._msg_client = DataClient(server_addr, uid=self._batch_client.uid)
        self._uid = self._batch_client.uid[:6]
        self._queue_monitoring_task = Thread(target=self._queue_monitoring, daemon=True)
        self._loading_batches_task = Thread(target=self._loading_batches, daemon=True)
        self._lock = RLock()

        if subset_name not in self._batch_client.info["subsets"]:
            raise KeyError(f"subset {subset_name} not provided by data server!")

        self.subset_name = subset_name
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.drop_non_full = drop_non_full
        self.non_stop = non_stop
        self.pin_memory = pin_memory
        self.prefetch_on_gpu = prefetch_on_gpu
        assert self.min_batch_size <= self.batch_size

        assert max_prefetch_factor >= min_prefetch_factor
        self.min_prefetch_factor = min_prefetch_factor
        self.max_prefetch_factor = max_prefetch_factor
        self.prefetch_factor = min_prefetch_factor

        if epoch_len:
            self.epoch_len = epoch_len
        else:
            self.epoch_len = math.ceil(self.epoch_size / batch_size)

        self._stop_event = Event()
        self._batch_queue: tp.Deque = deque()
        self._num_batch_recv = 0

        self._is_verbose_logging = is_verbose_logging()

    def __len__(self) -> int:
        return int(self.epoch_len)

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        return self.next_batch()

    def __del__(self):
        self.finish()

    @property
    def client(self) -> DataClient:
        return self._batch_client

    @property
    def epoch_size(self) -> int:
        return self._batch_client.info["epoch_size"][self.subset_name]

    @property
    def dataset_size(self) -> int:
        return len(self._batch_client.info["dataset"][self.subset_name])

    @staticmethod
    @check_path(assert_file_exists=True)
    def init_from_config(
        file_path: tp_PATH,
        value_select: tp.Optional[tp.List[str]] = None,
        server_addr: tp.Optional[str] = None,
        subset_name: tp.Optional[str] = None,
        config_section: str = "data_loaders",
    ) -> "DataLoader":
        cfg = Config.create_from_file(
            file_path, section=config_section, value_select=value_select
        )

        if server_addr:
            cfg["server_addr"] = server_addr

        if subset_name:
            cfg["subset_name"] = subset_name

        return init_class_from_config(DataLoader, cfg)()

    def _log_to_file(self, text: tp.Union[str, bytes]):
        if self._is_verbose_logging:
            try:
                if isinstance(text, bytes):
                    text = text[:100]
                fn_name = f"{self.__class__.__name__}.{inspect.stack()[1][3]}"
                message = f"[{self._uid}][{self.subset_name}]: {text}"
                log_to_file(trace(fn_name, message=message))
            except Exception as e:
                LOGGER.error(trace(self, e))

    def _send_info_message(self, text: str):
        self._msg_client.send({"message": text, "subset_name": self.subset_name})
        self._log_to_file(text)

    def _recv_message(
        self,
        client: DataClient,
        deserialize: bool = False,
        timeout: int = 1,
        max_num_message: int = 25,
    ) -> tp.List[tp.Any]:
        with self._lock:
            try:
                return client.recv_multipart(deserialize, timeout, max_num_message)
            except Exception as e:
                LOGGER.error(trace(self, e))
                return []

    def _is_stop_iteration(self):
        if self._stop_event.is_set():
            self._log_to_file("stop iteration")
            raise StopIteration

    def _queue_monitoring(self):
        while not self._stop_event.is_set():
            try:
                free_slots = max(self.prefetch_factor - len(self._batch_queue), 0)
                if free_slots == 0:
                    continue

                self._send_info_message(DCM.IS_READY)
                response = self._recv_message(self._msg_client)

                is_ready = False
                is_epoch_complete = False
                for _bytes in response:
                    self._log_to_file(_bytes)
                    if DSM.READY.encode() in _bytes[:100]:
                        num_batch_send = int(_bytes.decode().split(":")[-1])
                        num_batch_delivery = max(num_batch_send - self._num_batch_recv, 0)
                        is_ready = free_slots - num_batch_delivery > 0
                        self._log_to_file(
                            f"num of batch send: {num_batch_send}; "
                            f"num of batch delivery: {num_batch_delivery}; "
                            f"num of batch receive: {self._num_batch_recv}; "
                            f"free slots: {free_slots}"
                        )
                    elif DSM.EPOCH_COMPLETE.encode() in _bytes[:100]:
                        is_epoch_complete = True
                        break

                if is_ready:
                    self._batch_request(free_slots)

                if not self.non_stop:
                    if is_epoch_complete and len(self._batch_queue) == 0:
                        self._stop_event.set()
                        self._send_info_message(DCM.EPOCH_COMPLETE)
                else:
                    if is_epoch_complete:
                        self._send_info_message(DCM.EPOCH_COMPLETE)

            except KeyboardInterrupt:
                LOGGER.error(trace(self, "Interrupt received, stopping ..."))
                break
            except Exception as e:
                LOGGER.error(trace(self, e))
            finally:
                self._log_to_file("thread went to sleep")
                Profiler.sleep(2.0)

    def _loading_batches(self):
        while not self._stop_event.is_set():
            try:
                self._batch_receive()
            except KeyboardInterrupt:
                LOGGER.error(trace(self, "Interrupt received, stopping ..."))
                break
            except Exception as e:
                LOGGER.error(trace(self, e))
            finally:
                Profiler.sleep(0.1)

    def _batch_request(self, batch_num: int):
        message = {
            "message": DCM.GET_BATCH,
            "subset_name": self.subset_name,
            "batch_size": self.batch_size,
            "batch_num": batch_num,
        }

        with self._lock:
            try:
                self._batch_client.send(message)
            except Exception as e:
                LOGGER.error(trace(self, e))

        self._log_to_file(str(message))

    def _batch_receive(self):
        response = self._recv_message(self._batch_client)

        batch_list = []
        for _bytes in response:
            self._log_to_file(_bytes)
            if _bytes == b"" or _bytes[0] == 0 or b"info:" in _bytes[:100]:
                continue
            else:
                batch_list.append(_bytes)

        if batch_list:
            batch_list = Serialize.loads(batch_list, inplace=True)
            self._num_batch_recv += len(batch_list)

        for batch in batch_list:
            if not isinstance(batch, Batch):
                continue

            if (
                self.drop_non_full and batch.size != self.batch_size
            ) or batch.size < self.min_batch_size:
                message = (
                    f"[{self._uid}][{self.subset_name}]: "
                    f"batch size mismatch "
                    f"(expected size {self.batch_size} but received {batch.size})"
                )
                LOGGER.warning(message)
            else:
                if self.pin_memory and batch.collated_samples is not None:
                    batch.collated_samples.pin_memory()

                self._batch_queue.append(batch)

    def start(self):
        self._stop_event.clear()
        self._queue_monitoring_task.start()
        self._loading_batches_task.start()
        self.reset()

    def finish(self):
        self._stop_event.set()
        try:
            if self._queue_monitoring_task.is_alive():
                self._queue_monitoring_task.join(timeout=1)
            if self._loading_batches_task.is_alive():
                self._loading_batches_task.join(timeout=1)
        except RuntimeError:
            pass
        except Exception as e:
            LOGGER.error(trace(self, e))

    def next_batch(self, sleep: float = 0, step: int = 3) -> Batch:
        if sleep > 100:
            raise RuntimeError(
                f"DataServer stopped responding for {self.subset_name} DataLoader!"
            )

        if not self._batch_queue:
            self._is_stop_iteration()

            assert (
                self._queue_monitoring_task.is_alive()
            ), "DataLoader has not been started!"

            if sleep >= step:
                LOGGER.warning(
                    f"[{self._uid}][{self.subset_name}]: Batches receive too slowly!"
                )
            elif sleep == 0:
                self.prefetch_factor = int(
                    min(self.prefetch_factor * 1.2, self.max_prefetch_factor)
                )
                self._log_to_file(f"increase prefetch factor ({self.prefetch_factor})")

            if sleep > 0 and sleep % 12 == 0 and self.non_stop:
                self.abort_processing()

            sleep += 0.5 if sleep < step else step

            Profiler.sleep(sleep)
            return self.next_batch(sleep)

        try:
            batch = self._batch_queue.popleft()
        except IndexError:
            batch = self.next_batch(sleep + 0.5)

        if self.prefetch_on_gpu and len(self._batch_queue) > 0:
            try:
                next_batch = self._batch_queue[0]
                if next_batch.collated_samples is not None:
                    next_batch.collated_samples.cuda()
            except IndexError:
                pass

        return batch

    def abort_processing(self):
        self._num_batch_recv = 0
        self._send_info_message(DCM.ABORT)

    def reset(self):
        self._num_batch_recv = 0
        self._batch_queue.clear()
        self._send_info_message(DCM.RESET)
        self.prefetch_factor = self.min_prefetch_factor

    def get_epoch_iterator(self) -> tp.Union[tp.Iterator[Batch], tp.Sized]:
        self.reset()

        class EpochIterator:
            def __init__(self, dl: "DataLoader"):
                self._dl = dl
                self._is_non_stop = self._dl.non_stop
                self._dl.non_stop = False
                self._batch_idx = 0

            def __iter__(self):
                return self

            def __len__(self) -> int:
                return int(self._dl.epoch_len)

            def __next__(self):
                try:
                    self._batch_idx += 1
                    if self._batch_idx > len(self):
                        raise StopIteration

                    return next(self._dl)
                finally:
                    self._dl.non_stop = self._is_non_stop

        return EpochIterator(self)


def test_connection(data_loader: DataLoader, max_time: int = 10):
    number = batch_size = time = size = 0
    test_time = Profiler()

    while test_time.get_time() < max_time:
        with Profiler(auto_logging=False) as timer:
            batch = next(data_loader)

        if isinstance(batch, Batch):
            number += 1
            batch_size = batch.size
            time += timer.get_time()
            size += Serialize.get_obj_size(batch, format=Serialize.Format.MB)

    assert size > 0, "Not batch received!"
    speed = round(number / time, 3)
    size = round(size / number, 3)

    LOGGER.info(
        trace(
            "test",
            message=f"GET {data_loader.subset_name} {speed} batches/s, "
            f"batch size {batch_size}, "
            f"packet size {size} MB",
        )
    )


if __name__ == "__main__":
    """
    example:
        client.py -a=localhost:8000
    """

    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-a", "--addr", help="address of data server", type=str, required=True
    )
    arguments_parser.add_argument(
        "-ss", "--subset_name", help="subset name", type=str, default="train"
    )
    arguments_parser.add_argument(
        "-bs", "--batch_size", help="batch size", type=int, default=48
    )
    arguments_parser.add_argument(
        "-s", "--sleep", help="batch request interval", type=float, default=0.5
    )
    arguments_parser.add_argument(
        "-n", "--number", help="number of clients", type=int, default=1
    )
    arguments_parser.add_argument(
        "-v",
        "--visualize",
        help="draw wave and spectrogram segmentations",
        action="store_true",
    )
    args = arguments_parser.parse_args()

    def run_client(client_index):
        loader = DataLoader(
            args.addr,
            "train",
            epoch_len=100,
            batch_size=args.batch_size,
            drop_non_full=True,
        )
        loader.start()
        test_connection(loader)

        try:
            for batch_index in range(len(loader)):
                Profiler.sleep(args.sleep)
                with Profiler(auto_logging=False) as timer:
                    batch = next(loader)

                if batch:
                    packet_size = Serialize.get_obj_size(batch, Serialize.Format.MB)
                    print(
                        f"client {client_index}, "
                        f"batch {batch_index}, "
                        f"batch size {batch.size}, "
                        f"packet size {round(packet_size, 3)} MB, "
                        f"time {timer.get_time()} ms"
                    )
                else:
                    print("load dataset completed")

                # if args.visualize:
                #    visialize_batch(batch, batch_index)
        finally:
            loader.finish()

    tasks = [Thread(target=run_client, args=(index,)) for index in range(args.number)]
    for task in tasks:
        task.start()
    for task in tasks:
        task.join()

    print("TEST COMPLETED")
