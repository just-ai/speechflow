from itertools import product

from speechflow.data_pipeline.core import DataSample
from speechflow.data_pipeline.core.dataset import Dataset
from speechflow.data_pipeline.samplers.simple_sampler import SimpleSampler


def generate_dataset() -> Dataset:
    idx_text = ["1", "2", "3"]
    idx_wave = ["a", "b", "c"]
    idx_prod = list(product(idx_text, idx_wave))[2:]

    dataset = Dataset()
    for filepath, (idxt_item, idxw_item) in enumerate(idx_prod):
        dataset.append(DataSample(file_path=str(filepath), index=(idxt_item, idxw_item)))

    return dataset


def test_wave_select():
    data = generate_dataset()
    sampler = SimpleSampler(use_neighbors=True)
    sampler.set_dataset(data)
    sampled_datasamples = sampler.sampling(batch_size=6)

    assert sampled_datasamples[0].index[0] != sampled_datasamples[1].index[0]
    assert sampled_datasamples[0].index[1] == sampled_datasamples[1].index[1]


def test_doubling_batch():
    data = generate_dataset()
    sampler = SimpleSampler(use_neighbors=True)
    sampler.set_dataset(data)
    sampled_datasamples = sampler.sampling(batch_size=6)

    assert len(sampled_datasamples) == 12


def test_unique_filepath():
    data = generate_dataset()
    sampler = SimpleSampler(use_neighbors=True)
    sampler.set_dataset(data)
    sampled_datasamples = sampler.sampling(batch_size=6)

    for n in range(0, len(sampled_datasamples) - 1, 2):
        m = n + 1
        is_eq_text = sampled_datasamples[n].index[0] != sampled_datasamples[m].index[0]
        is_eq_wave = sampled_datasamples[n].index[1] != sampled_datasamples[m].index[1]

        assert is_eq_text or is_eq_wave
        assert sampled_datasamples[n].file_path != sampled_datasamples[m].file_path
