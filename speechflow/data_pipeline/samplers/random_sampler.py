import logging

from random import randint, shuffle
from typing import List

from speechflow.data_pipeline.core import DataSample
from speechflow.data_pipeline.core.dataset import Dataset
from speechflow.data_pipeline.samplers.simple_sampler import SimpleSampler

__all__ = ["RandomSampler"]

LOGGER = logging.getLogger("root")


class RandomSampler(SimpleSampler):
    """Returns random sequence of samples from data without actual shuffling Data can be
    optionally ordered by legth."""

    def __init__(
        self,
        comb_by_len: bool = False,
        use_neighbors: bool = False,
        use_dynamic_batch: bool = False,
        max_batch_length: int = 1000,  # in milliseconds for AudioDataSample
    ):
        super().__init__(comb_by_len, use_neighbors, use_dynamic_batch, max_batch_length)
        self._total_samples = 0

    def set_dataset(self, data: Dataset):
        super().set_dataset(data)
        self._total_samples = 0
        self.fill_epoch()

    def fill_epoch(self):
        if not self._comb_by_len:
            shuffle(self._current_data)

    def sampling(self, batch_size: int) -> List[DataSample]:
        self._is_last_batch = False

        if self._comb_by_len:
            idx = randint(0, max(len(self._current_data) - batch_size, 0))
            chunk = self._current_data[idx : idx + batch_size]
            self._total_samples += batch_size
            if self._total_samples >= self._epoch_size:
                self._is_last_batch = True
                self._total_samples = 0
                self.fill_epoch()
            chunk = self.add_neighbors(chunk=chunk)
        else:
            chunk = super().sampling(batch_size)

        if self._use_dynamic_batch:
            new_chunk = []
            chunk_len = 0
            for item in reversed(chunk):
                if (
                    item is None
                    or len(new_chunk) == 0
                    or chunk_len + len(item) < self._max_batch_length
                ):
                    new_chunk.append(item)
                    if item is not None:
                        chunk_len += len(item)

            chunk = new_chunk[::-1]

        return chunk
