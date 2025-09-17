import random
import typing as tp
import logging

import numpy as np
import torch
import numpy.typing as npt

from speechflow.data_pipeline.core.dataset import Dataset
from speechflow.data_pipeline.samplers.random_sampler import RandomSampler

__all__ = ["FillingSampler"]

LOGGER = logging.getLogger("root")


class FillingSampler(RandomSampler):
    """Sampler for unbalanced data, which form an epoch by filling it with samples that
    contain lacking classes.

    It is useful for data where sequences of classes of different length are
    attributed to each sample.

    """

    def __init__(
        self,
        fields_to_compute_weight: tp.List[str],
        epoch_size: int = 1000,
        comb_by_len: bool = False,
        use_neighbors: bool = False,
        use_dynamic_batch: bool = False,
        max_batch_length: int = 1000,  # in milliseconds for AudioDataSample
        chunks_ratio: tp.Optional[tp.List] = None,
        filter_tags: tp.Optional[tp.List] = None,
    ):
        super().__init__(comb_by_len, use_neighbors, use_dynamic_batch, max_batch_length)
        self._fields_to_compute_weight = fields_to_compute_weight
        self._chunks_ratio = chunks_ratio
        self._epoch_size_w = epoch_size
        self._num_labels = 0

        self._chunks_size = None
        self._probs = None
        self._matrix = None  # matrix with frequencies of size (n_samples, n_classes)
        self._filter_tag = [] if filter_tags is None else filter_tags

    def set_dataset(self, data: Dataset):
        super().set_dataset(data)

        if self._epoch_size_w:
            self._epoch_size = self._epoch_size_w

        self.compute_freq_matrix(self._fields_to_compute_weight)
        self._chunks_size = min(self._epoch_size, self._dataset_size)
        self.fill_epoch()

    def compute_freq_matrix(self, fields: tp.List[str]) -> None:
        samples = [d.__getattribute__(fields[0]) for d in self._data]
        all_tags = [
            tag.item()
            for sample in samples
            for tag in sample
            if tag not in self._filter_tag
        ]
        self.unique_labels = np.unique(np.array(all_tags))

        N = len(samples)  # number of documents
        self._num_labels = self.unique_labels.shape[0]  # number of terms
        weights = np.zeros((N, self._num_labels))  # matrix frequencies
        for idx, sample in enumerate(samples):
            weights = self.compute_freqs(idx, sample, weights)
        self._matrix = weights

    def compute_freqs(
        self, idx: int, sample: npt.NDArray, weights: npt.NDArray
    ) -> npt.NDArray:
        tags, tag_weight = np.unique(sample, return_counts=True)
        for i, tag in enumerate(tags):
            if tag not in self._filter_tag:
                if len(weights.shape) == 1:
                    weights[np.where(self.unique_labels == tag)[0][0]] += tag_weight[i]
                else:
                    weights[idx][np.where(self.unique_labels == tag)[0][0]] += tag_weight[
                        i
                    ]
        return weights

    def fill_epoch(self):
        if self._matrix is None:
            return

        indices_epoch = []

        # first, randomly init 0.25 of epoch
        field = self._fields_to_compute_weight[0]
        label_counts = np.zeros(self._num_labels)
        self._current_data = list(
            np.random.choice(self._data, self._chunks_size // 4, False)
        )

        labels = [d.__getattribute__(field) for d in self._current_data]
        assert isinstance(labels[0], (np.ndarray, np.generic, torch.Tensor))
        for idx, sample in enumerate(labels):
            label_counts = self.compute_freqs(idx, sample, label_counts)

        # then, calculate the difference between the most and the least classes
        # and balance them till the difference is lower than the given threshold
        idx_max = np.argmax(label_counts)
        idx_min = np.argmin(label_counts)
        s = sum(label_counts)
        diff = label_counts[idx_max] / s - label_counts[idx_min] / s
        while diff > 0.1 and len(self._current_data) <= self._chunks_size:
            idx_cur = self.unique_labels[idx_min]
            subset = np.random.choice(
                np.arange(self._matrix.shape[0]), self._chunks_size, False
            )
            indices = self._matrix[:, idx_cur].argsort()[::-1]
            for idx in indices:
                if idx not in subset:
                    continue
                sample = self._data[idx].__getattribute__(field)
                self._current_data.append(self._data[idx])
                indices_epoch.append(idx)
                label_counts = self.compute_freqs(idx, sample, label_counts)
                s = sum(label_counts)
                if label_counts[idx_max] / s - label_counts[idx_min] / s <= 0.1:
                    idx_max = np.argmax(label_counts)
                    idx_min = np.argmin(label_counts)
                    diff = label_counts[idx_max] / s - label_counts[idx_min] / s
                    break

        if self._comb_by_len:
            self._current_data.sort(key=len, reverse=True)
        else:
            random.shuffle(self._current_data)
