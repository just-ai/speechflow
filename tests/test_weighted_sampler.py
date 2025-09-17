from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from speechflow.data_pipeline.core import DataSample
from speechflow.data_pipeline.core.dataset import Dataset
from speechflow.data_pipeline.samplers.weighted_sampler import WeightedSampler


def test_computed_probabilities():
    labels = ["class_1", "class_1", "class_2"]
    datasamples = [DataSample(label=label) for label in labels]
    sampler = WeightedSampler(fields_to_compute_weight=["label"])
    sampler.set_dataset(Dataset(datasamples))
    assert (sampler._probs == np.array([0.25, 0.25, 0.5])).all()


def test_multifield_distribution():
    np.random.seed(128)
    labels = ["class_1", "class_1", "class_2"] * 1000
    file_path = "sample_1"
    datasamples = [DataSample(label=label, file_path=Path(file_path)) for label in labels]
    sampler = WeightedSampler(
        fields_to_compute_weight=["label", "file_path"],
        epoch_size=102,
    )
    sampler.set_dataset(Dataset(datasamples))
    sampled = [sample.label for sample in sampler.sampling(batch_size=100)]
    output_dist = {k: v / 100 for k, v in Counter(sampled).items()}
    # E(class_1) =
    # p(class_1 | sample_field=label) * 0.5 + p(class_1 | sample_field=file_path) * 0.5
    # = 0.5 * (0.5 + 0.25) = 0.58, probably lies in [0.5; 0.7]
    assert (output_dist["class_1"] >= 0.5) and (output_dist["class_1"] <= 0.7)


@pytest.mark.parametrize("chunks_ratio", ([0.5, 0.5], [0.1, 0.9]))
def test_chunk_ratio_distribution(chunks_ratio):
    np.random.seed(128)
    labels = ["class_1", "class_1", "class_2"] * 1000
    file_path = "sample_1"
    datasamples = [DataSample(label=label, file_path=Path(file_path)) for label in labels]
    sampler = WeightedSampler(
        fields_to_compute_weight=["label", "file_path"],
        epoch_size=102,
        chunks_ratio=chunks_ratio,
    )
    sampler.set_dataset(Dataset(datasamples))
    sampled = [sample.label for sample in sampler.sampling(batch_size=100)]
    output_dist = {k: v / 100 for k, v in Counter(sampled).items()}
    exp = (chunks_ratio[0] * 0.5) + (chunks_ratio[1] * 0.67)
    assert (output_dist["class_1"] >= exp - 0.1) and (output_dist["class_1"] <= exp + 0.1)


def test_output_distribution():
    np.random.seed(128)
    labels = ["class_1", "class_1", "class_2"] * 1000
    datasamples = [DataSample(label=label) for label in labels]
    sampler = WeightedSampler(fields_to_compute_weight=["label"], epoch_size=1000)
    sampler.set_dataset(Dataset(datasamples))
    sampled = [sample.label for sample in sampler.sampling(batch_size=100)]
    output_dist = {k: v / 100 for k, v in Counter(sampled).items()}
    # E(class_1) = 0.5, probably lies in [0.4; 0.6]
    assert (output_dist["class_1"] >= 0.4) and (output_dist["class_1"] <= 0.6)
