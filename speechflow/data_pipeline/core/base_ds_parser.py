import math
import pickle
import typing as tp
import hashlib
import logging
import functools
import itertools
import multiprocessing as mp

from collections import ChainMap
from pathlib import Path

from tqdm import tqdm

from speechflow.data_pipeline.core.dataset import Dataset, DatasetItem
from speechflow.data_pipeline.core.parser_types import (
    Metadata,
    MetadataTransform,
    MultiMetadataTransform,
)
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.io import tp_PATH
from speechflow.logging import is_verbose_logging, trace
from speechflow.logging.logger import create_logger

__all__ = ["BaseDSParser", "multi_transform"]

LOGGER = logging.getLogger("root")


def multi_transform(func: MultiMetadataTransform):
    setattr(func, "multi_transform", True)
    return func


def init_logger(lock):
    global LOGGER
    LOGGER = create_logger()

    BaseDSParser.lock = lock


class DummyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class BaseDSParser:
    """Metadata parser for database."""

    lock = DummyContextManager()

    def __init__(
        self,
        preproc_fn: tp.Optional[tp.Sequence[tp.Callable]] = None,
        input_fields: tp.Optional[tp.Set[str]] = None,
        memory_bound: bool = False,
        chunk_size: tp.Optional[int] = None,
        raise_on_converter_exc: bool = False,
        dump_path: tp.Optional[tp_PATH] = None,
        release_func: tp.Optional[tp.Callable] = None,
        progress_bar: bool = True,
    ):
        """
        :param preproc_fn: sequence of transforms for metadata
        :param input_fields: expected initial metadata fields to validate the sequence of handlers
        :param memory_bound: memory used will be limited by the dataset size
        :param chunk_size: chunk size for single worker
        :param raise_on_converter_exc: Raise exception if convert sample with error
        :param dump_path: folder for storage cache file
        :param progress_bar: whether to display a progress bar showing the progress
        """
        if preproc_fn and input_fields:
            PipeRegistry.check(preproc_fn, input_fields=input_fields)

        self.preproc_fn = preproc_fn
        self.memory_bound = memory_bound
        self.chunk_size = chunk_size

        self.raise_on_converter_exc = raise_on_converter_exc
        self.release_func = release_func
        self.progress_bar = progress_bar

        if dump_path:
            self.cache_folder = Path(dump_path) / "cache"
            self.cache_folder.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_folder = None  # type: ignore

    @staticmethod
    def _hash_preproc_params(functions, hash_len: int = 6):
        all_params = sorted(
            ChainMap(*[func.keywords for func in functions]).items(),
            key=lambda item: item[0],
        )
        params_string = "|".join(str(item.func._name) for item in functions)
        params_string += "_".join([f"{param}" for param in all_params])
        return hashlib.md5(params_string.encode("utf-8")).hexdigest()[:hash_len]

    def _get_cache_fpath(self, num_files: int) -> tp.Optional[Path]:
        if self.cache_folder is None:
            return

        hash_name = self._hash_preproc_params(self.preproc_fn)
        return (
            self.cache_folder / f"{self.__class__.__name__}_{num_files}s_{hash_name}.pkl"
        )

    def reader(
        self, file_path: Path, label: tp.Optional[str] = None
    ) -> tp.List[Metadata]:
        """Read data and annotation from disk.

        :param file_path: absolute file path
        :param label: additional labels for file
        :return: metadata

        """
        raise NotImplementedError

    def converter(self, metadata: Metadata) -> tp.List[tp.Any]:
        """Convert Metadata to DataSample.

        :param metadata
        :return: list of datasample

        """
        raise NotImplementedError

    @staticmethod
    def compose_transforms(metadata, transforms) -> Dataset:
        if isinstance(metadata, DatasetItem):
            metadata = metadata.get()

        metadata = [metadata]
        for transform in transforms:
            next_metadata = []
            for data in metadata:
                try:
                    next_metadata += transform(data)
                except Exception as e:
                    message = f"{transform}"
                    if isinstance(metadata, tp.Dict):
                        message += f"; sample {metadata.get('file_path')}"
                    LOGGER.error(trace("Error transform", e, message))
            metadata = next_metadata
        return Dataset(metadata)

    @staticmethod
    def do_preprocessing(
        all_metadata: tp.Union[tp.List[Metadata], Dataset],
        preproc_fn: tp.Sequence[MetadataTransform],
        n_processes: int = 1,
        memory_bound: bool = False,
        chunk_size: tp.Optional[int] = None,
        release_func: tp.Optional[tp.Callable] = None,
        progress_bar: bool = True,
    ) -> Dataset:
        """Apply preprocessing functions.

        :param all_metadata: list dictionary containing data of current sample
        :param preproc_fn: list of preprocessing functions
        :param n_processes: size of workers' pool
        :param memory_bound: reduce memory usage
        :param chunk_size: chunk size for single worker
        :param release_func: function for delete temporary object
        :param progress_bar: whether to display a progress bar showing the progress
        :return: list of metadata

        """
        if isinstance(all_metadata, tp.List):
            all_metadata = Dataset(all_metadata)

        for is_multi, transforms in itertools.groupby(
            preproc_fn, key=lambda x: getattr(x.func, "multi_transform", False)  # type: ignore
        ):
            if is_multi:
                fns = list(transforms)
                for transform in tqdm(
                    fns,
                    total=len(fns),
                    desc="Multitransform preprocessing",
                    disable=len(fns) == 1 or not progress_bar,
                ):
                    try:
                        all_metadata = transform(all_metadata)
                    except Exception as e:
                        LOGGER.error(
                            trace(
                                "Error multitransform",
                                exception=e,
                                message=f"{transform}",
                            )
                        )
            else:
                all_metadata = BaseDSParser.pool_processing(
                    functools.partial(
                        BaseDSParser.compose_transforms, transforms=list(transforms)
                    ),
                    all_metadata,
                    desc="Preprocessing metadata",
                    n_processes=n_processes,
                    memory_bound=memory_bound,
                    chunk_size=chunk_size,
                    release_func=release_func,
                    progress_bar=progress_bar,
                )

        return all_metadata

    @staticmethod
    def _inplace(idx, func: tp.Callable, inputs) -> list:
        inputs[idx] = func(inputs[idx])
        return []

    @staticmethod
    def _list_to_proxy(inputs: tp.Union[tp.List, Dataset]):
        proxy = mp.Manager().list()
        if isinstance(inputs, tp.List):
            for _ in range(len(inputs)):
                proxy.append(inputs[0])
                del inputs[0]
        elif isinstance(inputs, Dataset):
            proxy += inputs.samples
            inputs.clear()
        else:
            raise NotImplementedError
        return proxy

    @staticmethod
    def _proxy_to_list(proxy) -> Dataset:
        inputs = Dataset()
        for _ in range(len(proxy)):
            item = proxy[0]
            if item is not None:
                if isinstance(item, tp.List):
                    inputs.extend(item)
                else:
                    inputs.append(item)
            del proxy[0]
        del proxy
        return inputs

    @staticmethod
    def pool_processing(
        func: tp.Callable,
        inputs: tp.Union[tp.List[Metadata], Dataset],
        desc: str = "",
        n_processes: int = 1,
        memory_bound: bool = False,
        chunk_size: tp.Optional[int] = None,
        release_func: tp.Optional[tp.Callable] = None,
        progress_bar: bool = True,
    ) -> Dataset:
        memory_bound = memory_bound and n_processes > 1
        tqdm_disable = (
            len(inputs) == 1
            or (n_processes == 1 and not is_verbose_logging())
            or not progress_bar
        )
        if chunk_size is None:
            chunk_size = 1 if len(inputs) < 100 else int(math.sqrt(len(inputs)))

        if memory_bound:
            proxy = BaseDSParser._list_to_proxy(inputs)
            func = functools.partial(BaseDSParser._inplace, func=func, inputs=proxy)  # type: ignore
            inputs = list(range(len(proxy)))

        if n_processes > 1:
            with mp.get_context("spawn").Pool(
                n_processes,
                initializer=init_logger,
                initargs=(mp.Lock(),),
            ) as pool:
                output = pool.imap_unordered(func, inputs, chunksize=chunk_size)
                output = tqdm(output, total=len(inputs), desc=desc, disable=tqdm_disable)
                # output = list(itertools.chain(*output))
                output = Dataset(output)

                if release_func is not None:
                    list(
                        pool.imap_unordered(release_func, range(n_processes), chunksize=1)
                    )
        else:
            output = Dataset()
            for item in tqdm(inputs, total=len(inputs), desc=desc, disable=tqdm_disable):
                output.extend(func(item))

            if release_func is not None:
                release_func(0)

        if memory_bound:
            output = BaseDSParser._proxy_to_list(proxy)

        return output

    def read_sample(
        self,
        file_path: tp.Union[str, Path, tp.Any],
        data_root: tp.Optional[tp.Union[str, Path]] = None,
    ) -> Dataset:
        """Reads a single sample from disk and preprocesses it.

        :param file_path: path to file and label (optional) separated by "|"
        :param data_root: path to data folder
        :return: list of correctly processed files

        """
        if not isinstance(file_path, (str, Path)):
            try:
                return Dataset(self.reader(file_path))
            except Exception as e:
                LOGGER.error(trace(self, e))

        label = None
        if isinstance(file_path, str) and "|" in file_path:
            file_path, label = file_path.split("|", 1)

        file_path = Path(file_path)
        if not file_path.exists() and not file_path.is_absolute():
            if data_root is None:
                raise ValueError(f"Cannot determine absolute path for {file_path}")
            file_path = Path(data_root) / file_path

        if file_path.is_file():
            try:
                return Dataset(self.reader(file_path, label))
            except Exception as e:
                LOGGER.error(trace(self, e, message=file_path.as_posix()))
        else:
            LOGGER.warning(
                trace(self, f"Path {file_path} does not exist or is not a file!")
            )

        return Dataset([])

    def convert_sample(self, metadata: Metadata) -> Dataset:
        """Convert a single metadata to DataSample."""
        if isinstance(metadata, DatasetItem):
            metadata = metadata.get()

        try:
            return Dataset(self.converter(metadata))
        except Exception as e:
            if self.raise_on_converter_exc:
                raise e
            message = None
            if isinstance(metadata, tp.Dict):
                message = f"sample {metadata.get('file_path')}"
            LOGGER.error(trace(self, e, message))

        return Dataset([])

    def to_datasample(
        self, all_metadata: tp.Union[tp.List[Metadata], Dataset], n_processes: int = 1
    ) -> Dataset:
        all_datasample = self.pool_processing(
            self.convert_sample,
            all_metadata,
            desc="Converting metadata to datasample",
            n_processes=n_processes,
            memory_bound=self.memory_bound,
            chunk_size=self.chunk_size,
            release_func=self.release_func,
            progress_bar=self.progress_bar,
        )
        if self.progress_bar:
            LOGGER.info(
                trace(self, message=f"Prepared {len(all_datasample)} datasamples")
            )
        return all_datasample

    def read_datasamples(
        self,
        file_list: tp.Union[tp.List[str], tp.List[Path], tp.List[tp.Any]],
        data_root: tp.Optional[tp.Union[str, Path]] = None,
        n_processes: tp.Optional[int] = 1,
        post_read_hooks: tp.Optional[tp.Sequence[tp.Callable]] = None,
    ) -> Dataset:
        """Reads list of samples from disk and preprocesses its.

        :param data_root:
        :param file_list:
        :param n_processes: size of workers' pool for reading and preprocessing .segs
        :param post_read_hooks:

        """
        assert file_list, "file list is empty!"
        n_processes = n_processes if n_processes else mp.cpu_count()
        cache_fpath = self._get_cache_fpath(len(file_list))

        if cache_fpath and cache_fpath.exists():
            LOGGER.info(f"Load DSParser cache from {cache_fpath.as_posix()}")
            dataset = pickle.loads(cache_fpath.read_bytes())
        else:
            # read
            all_metadata = self.pool_processing(
                functools.partial(self.read_sample, data_root=data_root),
                file_list,
                desc="Reading metadata",
                n_processes=n_processes,
                memory_bound=self.memory_bound,
                chunk_size=self.chunk_size,
                release_func=self.release_func,
                progress_bar=self.progress_bar,
            )
            assert all_metadata, "metadata not read!"

            # preprocess
            if self.preproc_fn:
                all_metadata = self.do_preprocessing(
                    all_metadata,
                    self.preproc_fn,
                    n_processes,
                    self.memory_bound,
                    self.chunk_size,
                    self.release_func,
                )

            # convert to datasamples
            dataset = self.to_datasample(all_metadata, n_processes)

            if cache_fpath:
                cache_fpath.write_bytes(pickle.dumps(dataset))

            # print statistics, etc.
            if post_read_hooks:
                for hook in post_read_hooks:
                    hook(self, all_metadata)

        return dataset
