import random
import typing as tp
import logging

import numpy as np
import torch
import numpy.typing as npt

from scipy.special import softmax
from sklearn.preprocessing import normalize
from tqdm import tqdm

from speechflow.data_pipeline.core import DataSample
from speechflow.data_pipeline.core.dataset import Dataset
from speechflow.data_pipeline.samplers.random_sampler import RandomSampler
from speechflow.logging import trace

__all__ = ["WeightedSampler"]

LOGGER = logging.getLogger("root")


class WeightedSampler(RandomSampler):
    """Samples set of samples in correspondence with their weight to an "epoch" of size
    'epoch_size'.

    'epoch' can be sorted by length. 'epoch size' should be lesser, than dataset size
    weight is a tuple of values -- different weighting results. Probabilities for
    each weight criterion_modules is calculated via softmax, and than probs for all
    criteria are averaged. weight should be in log scale -- softmax is used to get
    probabilties.

    """

    def __init__(
        self,
        fields_to_compute_weight: tp.List[str],
        comb_by_len: bool = False,
        use_neighbors: bool = False,
        use_dynamic_batch: bool = False,
        max_batch_length: int = 1000,  # in milliseconds for AudioDataSample
        epoch_size: tp.Optional[int] = None,
        chunks_ratio: tp.Optional[tp.List] = None,
        is_sequence: tp.Optional[tp.List] = None,
        filter_tags: tp.Optional[tp.List] = None,
        print_stat: bool = False,
    ):
        super().__init__(comb_by_len, use_neighbors, use_dynamic_batch, max_batch_length)
        self._fields_to_compute_weight = fields_to_compute_weight
        self._chunks_ratio = chunks_ratio
        self._epoch_size_w = epoch_size
        self._print_stat = print_stat

        self._chunks_size = None
        self._probs = None
        self._is_sequence = is_sequence if is_sequence is not None else []
        self._filter_tags = filter_tags if filter_tags is not None else []

    def set_dataset(self, data: Dataset):
        super().set_dataset(data)

        if self._epoch_size_w:
            self._epoch_size = self._epoch_size_w
        else:
            self._epoch_size = len(data)

        try:
            probs = self.compute_probs(self._fields_to_compute_weight)
        except Exception as e:
            LOGGER.warning(trace(self, e, "error initializing weighted sampler!"))
            probs = np.ones((1, self._dataset_size)) / self._dataset_size

        probs_num = probs.shape[0]
        epoch_size = min(self._epoch_size, self._dataset_size)

        if self._chunks_ratio:
            if len(self._chunks_ratio) != probs_num:
                raise ValueError("Number of chunk ratio and probs num is not match!")
            if round(sum(self._chunks_ratio), 3) != 1.0:
                raise ValueError("Sum of ratio is not equal to one!")
            self._chunks_size = [int(epoch_size * ratio) for ratio in self._chunks_ratio]
        else:
            self._chunks_size = [epoch_size // probs_num] * probs_num

        self._probs = probs
        self._epoch_size = min(sum(self._chunks_size), self._dataset_size)
        self.fill_epoch()

    @staticmethod
    def compute_idf_weights(labels: npt.NDArray) -> npt.NDArray:
        tags, tags_index, tag_weight = np.unique(
            labels, return_inverse=True, return_counts=True
        )
        weights = tags_index.astype(np.float64)
        for i in range(tags.shape[0]):
            weights[i == tags_index] = labels.shape[0] / tag_weight[i]
        weights = weights / weights.sum()
        return weights

    # @staticmethod
    def compute_tfidf_weights(self, samples: tp.List[npt.NDArray]) -> npt.NDArray:
        """Computes Tf-Idf weights if field consists of a sequence of labels for each
        sample.

        Weights are a sum of Tf-Idf scores from a Tf-Idf matrix normalized by
        softmax.

        """

        all_tags = [
            tag.item()
            for sample in samples
            for tag in sample
            if tag not in self._filter_tags
        ]
        unique_labels = np.unique(np.array(all_tags))

        N = len(samples)  # number of documents
        num_labels = unique_labels.shape[0]  # number of terms

        weights = np.zeros((N, num_labels))  # matrix of tf-idf
        dfs = np.zeros(
            num_labels
        )  # vector of dfs (number of documents that contain the term)

        for idx, sample in enumerate(samples):
            tags, tag_weight = np.unique(sample, return_counts=True)
            for i, tag in enumerate(tags):
                if tag not in self._filter_tags:
                    weights[idx][np.where(unique_labels == tag)[0][0]] = tag_weight[i]
                    dfs[np.where(unique_labels == tag)[0][0]] += 1
            weights[idx] /= weights[idx].sum()  # fill with term frequancies

        idf = [np.log((N + 1) / (df + 1)) for df in dfs]  # idf vector
        for i in range(num_labels):
            weights[:, i] *= idf[i]  # multiply by idf

        return softmax(np.sum(weights, axis=1), axis=-1)

    def compute_probs(self, fields: tp.List[str]) -> npt.NDArray:
        data_slice = self.dataset.slice(fields)
        weights = np.zeros((len(fields), len(self)))

        for i, field in enumerate(tqdm(fields, desc="Compute probs", leave=False)):
            labels = data_slice.get(field)
            if field in self._is_sequence:
                assert isinstance(labels[0], (np.ndarray, np.generic, torch.Tensor))
                weights[i, :] = self.compute_tfidf_weights(labels)
            else:
                if field == "uid":
                    weights[i, :] = 1 / len(self)
                else:
                    weights[i, :] = self.compute_idf_weights(np.array(labels))

        return weights

    def fill_epoch(self):
        if self._probs is None:
            return

        self._current_data = []
        for idx in range(self._probs.shape[0]):
            self._current_data += list(
                np.random.choice(
                    self._data.samples, self._chunks_size[idx], True, self._probs[idx, :]
                )
            )

        if self._comb_by_len:
            self._current_data.sort(key=len, reverse=True)
        else:
            random.shuffle(self._current_data)

        if self._print_stat:
            self._stat()

    def _stat(self, bins: int = 10):
        message = f"random choice {len(self._current_data)} samples"
        LOGGER.info(trace(self, message=message))

        sample_len = [len(sample) for sample in self._current_data]
        mean_len = np.median(sample_len)

        sample_len = bins * normalize([sample_len], norm="max", axis=1)
        hist = np.histogram(sample_len, bins=np.arange(bins))[0]

        print(f"median len: {mean_len}, hist: {hist}")


if __name__ == "__main__":
    from collections import Counter

    test_data = [DataSample(label=random.choice([1, 1, 1, 1, 1, 2])) for i in range(10000)]  # type: ignore
    s = WeightedSampler(fields_to_compute_weight=["label"])
    s.set_dataset(Dataset(test_data))
    print(Counter([i.label for i in s.sampling(batch_size=128) if i]))
