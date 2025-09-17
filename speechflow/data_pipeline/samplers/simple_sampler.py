import random
import typing as tp
import logging

from copy import deepcopy

import torch

from tqdm import tqdm

from speechflow.data_pipeline.core import DataSample
from speechflow.data_pipeline.core.abstract import AbstractDataSampler
from speechflow.data_pipeline.core.dataset import Dataset
from speechflow.logging import trace

__all__ = ["SimpleSampler"]

LOGGER = logging.getLogger("root")


class SimpleSampler(AbstractDataSampler):
    """Simple sampler for tacotron.

    Forms batches by order. Optional sorting of data by length for efficient memory
    usage.

    """

    def __init__(
        self,
        comb_by_len: bool = False,
        use_neighbors: bool = False,
        use_dynamic_batch: bool = False,
        max_batch_length: int = 1000,  # in milliseconds for AudioDataSample
    ):
        super().__init__()
        self._comb_by_len = comb_by_len
        self._use_neighbors = use_neighbors
        self._use_dynamic_batch = use_dynamic_batch
        self._max_batch_length = max_batch_length

        self._data: Dataset = None  # type: ignore
        self._dataset_size = None
        self._current_idx = 0
        self._is_last_batch = False
        self._neighbor_map = {}

        self._current_data = None  # type: ignore
        self._epoch_size = None

    @property
    def is_empty(self) -> bool:
        return self._data is None

    def set_dataset(self, data: Dataset):
        self._dataset_size = len(data)
        self._epoch_size = len(data)
        self._current_idx = 0
        self._is_last_batch = False

        if self._comb_by_len:
            data.sort(key=len, reverse=True)

        self._data = data
        self._current_data = data

        if self._use_neighbors:
            self.parse_neghbors()

        LOGGER.info(trace(self, message=f"Dataset size: {self._dataset_size}"))

    def parse_neighbors_by_index(self, index_id: int = 0):
        for item in tqdm(
            self._data.samples,
            desc=f"Parse neighbors by index_id {index_id}",
            leave=False,
        ):
            _idx = item.index[index_id]
            if _idx is None:
                continue

            self._neighbor_map.setdefault(_idx, set())
            self._neighbor_map[_idx].add(item)

        return None

    def parse_neghbors(self):
        with self._data.readonly():
            for i in range(len(self._data.item(0).index)):
                self.parse_neighbors_by_index(i)

    def mayby_get_neighbor(
        self, ds: DataSample, index: tp.Any
    ) -> tp.Optional[DataSample]:
        neighbors = self._neighbor_map[index]
        if len(neighbors) == 1:
            return None

        neighbor: DataSample = random.sample(neighbors.difference({ds}), 1)[0]
        return neighbor

    def sample_neighbor(self, ds: DataSample) -> DataSample:
        neighbor_ds = None

        for i in range(len(self._data.item(0).index)):
            neighbor_ds = self.mayby_get_neighbor(ds=ds, index=ds.index[i])
            if neighbor_ds is not None:
                break

        if neighbor_ds is None:
            LOGGER.warning(trace(self, message="neighbors weren't been found"))
            neighbor_ds = ds.copy()

        return neighbor_ds

    @staticmethod
    def add_neighbor_batch_idx(ds: DataSample, n: int) -> DataSample:
        ds.additional_fields["neighbor_idx"] = torch.Tensor([n])
        return ds

    def add_neighbors(self, chunk: tp.List[DataSample]) -> tp.List[DataSample]:
        if not self._use_neighbors:
            return chunk

        chunk_new = []
        for n, _sample in enumerate(chunk):
            if _sample is None:
                continue

            neighbor = self.sample_neighbor(_sample)
            if neighbor is None:
                continue

            chunk_new.append(neighbor)
            self.add_neighbor_batch_idx(neighbor, n)

            chunk_new.append(_sample)
            self.add_neighbor_batch_idx(_sample, n)

        return chunk_new

    def __len__(self):
        return self._dataset_size

    @property
    def dataset(self) -> Dataset:
        return self._data

    @property
    def epoch_size(self) -> int:
        return self._epoch_size

    @property
    def is_last_batch(self) -> bool:
        return self._is_last_batch

    def reset(self):
        self._current_idx = 0
        self._is_last_batch = False

    def fill_epoch(self):
        pass

    def _get_samples(self, batch_size: int) -> tp.List[DataSample]:
        if self._current_idx + batch_size >= self._epoch_size:
            # "None" is signals about the last batch
            chunk = self._current_data[self._current_idx :] + [None]  # type: ignore
            self._is_last_batch = True
        else:
            chunk = self._current_data[self._current_idx : self._current_idx + batch_size]
            self._current_idx += batch_size
        return chunk

    def sampling(self, batch_size: int) -> tp.List[DataSample]:
        self._is_last_batch = False

        if self._use_dynamic_batch:
            chunk = []
            chunk_len = 0
            for _ in range(batch_size):
                for item in self._get_samples(1):
                    chunk.append(item)
                    if item is not None:
                        chunk_len += len(item)
                if self._is_last_batch or chunk_len > self._max_batch_length:
                    break
        else:
            chunk = self._get_samples(batch_size)

        if self._is_last_batch:
            self._current_idx = 0
            self.fill_epoch()

        chunk = self.add_neighbors(chunk)
        return chunk

    def copy(self):
        data = self._data
        current_data = self._current_data
        self._data = None
        self._current_data = None

        sampler = deepcopy(self)

        self._data = data
        self._current_data = current_data
        sampler._data = data
        sampler._current_data = current_data

        return sampler
