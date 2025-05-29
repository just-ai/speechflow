import os
import typing as tp
import logging

from contextlib import contextmanager
from dataclasses import dataclass
from os import environ as env

import torch
import torch.distributed as dist

from speechflow.data_pipeline.core import Batch, DataPipeline, DataSample
from speechflow.data_server.client import DataClient
from speechflow.data_server.loader import DataLoader
from speechflow.data_server.pool import WorkerPool
from speechflow.data_server.proxy import Proxy
from speechflow.data_server.server import DataServer
from speechflow.io import Config, check_path, tp_PATH, tp_PATH_LIST
from speechflow.logging import trace, track_process
from speechflow.logging.logger import create_logger
from speechflow.utils.gpu_info import get_freer_gpu
from speechflow.utils.init import init_class_from_config
from speechflow.utils.tensor_utils import string_to_tensor, tensor_to_string

__all__ = [
    "LoaderParams",
    "DatasetIterator",
    "run_server",
    "init_data_loader",
    "init_data_loader_from_config",
    "get_dataset_iterator",
]

LOGGER = logging.getLogger("root")


@dataclass
class LoaderParams:
    epoch_len: int = 0
    batch_size: int = 1
    min_batch_size: int = 0
    drop_non_full: bool = False
    non_stop: bool = True
    pin_memory: bool = False
    prefetch_on_gpu: bool = False
    min_prefetch_factor: int = 50
    max_prefetch_factor: int = 150

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        return self.__dict__

    def to_config(self) -> Config:
        return Config(self.to_dict())


class DatasetIterator:
    def __init__(self, data_pipeline: DataPipeline, subset_name: str, batch_size: int):
        self.data_pipeline = data_pipeline
        self.data_pipeline.init_components()

        self.data_processor = data_pipeline[subset_name].data_processor
        self.sampler = data_pipeline[subset_name].sampler

        self.subset_name = subset_name
        self.batch_size = batch_size

    def reset(self):
        self.sampler.reset()

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.sampler) // self.batch_size

    def __next__(self) -> Batch:
        if self.sampler.is_empty():
            self.data_pipeline.load_data()

        if self.sampler.is_last_batch:
            raise StopIteration

        samples: tp.List[DataSample] = self.sampler.sampling(self.batch_size)
        samples = [s.copy() for s in samples if s is not None]

        if not samples:
            raise StopIteration

        batch = self.data_processor.process(samples)
        return batch if batch is not None else next(self)


def _finish_component(
    servers: tp.Optional[tp.List[DataServer]],
    workers: tp.Optional[tp.List[WorkerPool]],
    proxy: tp.Optional[Proxy],
    data_loaders: tp.Optional[tp.Dict[str, DataLoader]],
):
    if servers:
        [s.finish() for s in servers]

    if workers:
        [w.finish() for w in workers]

    if proxy:
        proxy.finish()

    if data_loaders:
        [dl.finish() for dl in data_loaders.values()]


@contextmanager
def run_server(server, worker_type):
    server.start()

    worker_pool = WorkerPool(
        server_addr=server.address,
        n_processes=server.num_processes,
        worker_type=worker_type,
    )

    try:
        worker_pool.start()
    except Exception as e:
        server.finish()
        LOGGER.error(e)
        raise e

    try:
        yield

    except Exception as e:
        LOGGER.error(e)
        raise e

    finally:
        worker_pool.finish()
        server.finish()


@contextmanager
def init_data_loader(
    loader_params: LoaderParams,
    data_pipeline: DataPipeline,
    n_processes: int = 1,
    n_gpus: tp.Union[int, tp.List[int]] = 0,
    flist_by_subsets: tp.Optional[tp.Dict[str, tp.List[str]]] = None,
) -> tp.Generator[tp.Dict[str, DataLoader], None, None]:
    if n_gpus > 0 and torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = "cpu"

    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()

    servers = []
    workers = []
    if not dist.is_initialized() or dist.get_rank() == 0:
        if flist_by_subsets is not None:
            data_pipeline.set_file_list(flist_by_subsets)

        servers.append(
            DataServer(
                data_pipeline=data_pipeline,
                n_processes=n_processes,
                n_gpus=n_gpus,
                synchronize_loaders=dist.is_initialized(),
            )
        )
        servers[0].start()

        workers.append(
            WorkerPool(server_addr=servers[0].address, n_processes=n_processes)
        )
        workers[0].start()

    if dist.is_initialized():
        if dist.get_rank() == 0:
            t = string_to_tensor(servers[0].address, device, 16)
            dist.barrier()
        else:
            t = torch.zeros(16, dtype=torch.int8).to(device)

        dist.broadcast(t, src=0)
        server_addr = tensor_to_string(t)
    else:
        server_addr = servers[0].address

    if not server_addr:
        raise ValueError("Address of DataServer is not set!")

    data_loaders = {}
    try:
        for name in data_pipeline.subsets:
            loader = init_class_from_config(DataLoader, loader_params.to_config())(
                server_addr=server_addr,
                subset_name=name,
            )
            data_loaders[name] = loader

        for loader in reversed(data_loaders.values()):
            loader.start()

        yield data_loaders

    except Exception as e:
        LOGGER.error(e)
        raise e

    finally:
        if dist.is_initialized():
            dist.barrier()

        _finish_component(servers, workers, None, data_loaders)


@contextmanager
@check_path(assert_file_exists=True)
def init_data_loader_from_config(
    loader_params: tp.Optional[LoaderParams] = None,
    model_config_path: tp.Optional[tp_PATH] = None,
    data_config_path: tp.Optional[tp.Union[tp_PATH, tp_PATH_LIST]] = None,
    value_select: tp.Optional[tp.List[str]] = None,
    proxy_class: tp.Optional[Proxy] = None,
    server_addr: tp.Optional[str] = None,
) -> tp.Generator[tp.Dict[str, DataLoader], None, None]:
    if dist.is_initialized():
        raise RuntimeError(
            "This method not supported inside the DDP process, use the 'init_data_loader' method."
        )

    if loader_params is None and model_config_path is None:
        raise ValueError("DataLoader params not set")

    if server_addr is None and data_config_path is None:
        raise ValueError("DataServer address not set")

    if not isinstance(data_config_path, tp.List):
        data_config_path = [data_config_path]

    servers = workers = proxy = data_loaders = None
    is_main_proc = int(os.environ.get("LOCAL_RANK", 0)) == 0

    if data_config_path and is_main_proc:
        try:
            if len(data_config_path) > 1 and proxy_class:
                raise RuntimeError("Unsupported configuration!")

            servers = []
            workers = []
            for path in data_config_path:
                servers.append(
                    DataServer.init_from_config(
                        file_path=path,
                        value_select=value_select,
                        server_addr=server_addr,
                    )
                )
                servers[-1].start()

                workers.append(
                    WorkerPool(
                        server_addr=servers[-1].address,
                        n_processes=servers[-1].num_processes,
                    )
                )
                workers[-1].start()

            if len(data_config_path) > 1:
                proxy = Proxy(server_addrs=[s.address for s in servers])
                proxy.start()
                server_addr = proxy.address
            elif proxy_class:
                try:
                    proxy = proxy_class.init_from_config(
                        file_path=data_config_path[0],
                        server_addr=servers[0].address,
                        value_select=value_select,
                    )
                    proxy.start()
                    server_addr = proxy.address
                except Exception as e:
                    LOGGER.error(
                        f"{proxy_class.__name__} is not initialised! {trace(exception=e)}"
                    )
                    server_addr = servers[0].address
            else:
                server_addr = servers[0].address

            os.environ["DATASERVER_ADDR"] = server_addr  # type: ignore

        except Exception as e:
            _finish_component(servers, workers, proxy, data_loaders)
            LOGGER.error(e)
            raise e

    server_addr = os.environ.get("DATASERVER_ADDR", server_addr)
    if not server_addr:
        raise ValueError("Address of DataServer is not set!")

    data_client = DataClient(server_addr=server_addr)

    track_process("MAIN", os.getpid())

    try:
        subsets = data_client.find_info("subsets")
        if not subsets:
            raise RuntimeError("subsets not found!")

        data_loaders = {}
        for name in subsets:
            if model_config_path is not None:
                loader = DataLoader.init_from_config(
                    file_path=model_config_path,
                    value_select=(value_select or []) + [name],
                    server_addr=server_addr,
                    subset_name=name,
                )
            else:
                assert loader_params
                loader = init_class_from_config(DataLoader, loader_params.to_config())(
                    server_addr=server_addr, subset_name=name
                )

            data_loaders[name] = loader

        for loader in reversed(data_loaders.values()):
            loader.start()

        yield data_loaders

    except Exception as e:
        LOGGER.error(e)
        raise e

    finally:
        _finish_component(servers, workers, proxy, data_loaders)


@check_path(assert_file_exists=True)
def get_dataset_iterator(
    config: tp.Union[tp_PATH, tp.MutableMapping],
    value_select: tp.Optional[tp.List[str]] = None,
    batch_size: int = 1,
    device: tp.Union[str, torch.device] = "cpu",
    with_dump: bool = True,
    subset_name: tp.Optional[str] = None,
    use_verbose_logging: bool = False,
) -> DatasetIterator:
    create_logger(use_server_logging=False, use_verbose_logging=use_verbose_logging)

    if not isinstance(config, tp.MutableMapping):
        cfg_data = Config.create_from_file(config, value_select=value_select)
    else:
        cfg_data = config

    if subset_name is None:
        subset_name = cfg_data["dataset"]["subsets"][0]
        cfg_data["dataset"]["subsets"] = [subset_name]
        cfg_data["dataset"]["split_ratio"] = {subset_name: [0, 1]}

    cfg_data["dataset"]["use_shuffle"] = False

    cfg_data["processor"]["output_collated_only"] = False
    cfg_data["sampler"] = {"type": "SimpleSampler"}

    if not with_dump:
        cfg_data["parser"]["dump"] = None
        cfg_data["processor"]["dump"] = None

    if "DatasetStatistics" in cfg_data["singleton_handlers"]["handlers"]:
        cfg_data["singleton_handlers"]["handlers"].remove("DatasetStatistics")

    env["DEVICE"] = f"cuda:{get_freer_gpu()}" if device == "cuda" else str(device)

    return DatasetIterator(DataPipeline(cfg_data), subset_name, batch_size)


if __name__ == "__main__":
    from tqdm import tqdm

    _config_path = "../../tts/acoustic_models/configs/tts/tts_data_24khz.yml"

    iterator = get_dataset_iterator(
        _config_path, device="cpu", with_dump=True, value_select=["ru"]
    )

    count = 0
    for b in tqdm(iterator):
        for sample in b.data_samples:
            print(sample.file_path)
            count += 1

    print("total samples:", count)
