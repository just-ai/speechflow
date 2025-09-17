import abc
import typing as tp

from speechflow.data_pipeline.core.dataset import Dataset


class AbstractDataSampler:
    """Base class for data loaders for pyTorch nets."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """Everything that we need to ini."""
        pass

    @abc.abstractmethod
    def is_empty(self) -> bool:
        """Check that list data samples is not set."""
        pass

    @abc.abstractmethod
    def set_dataset(self, data: Dataset):
        """Set list of data samples."""
        pass

    @property
    @abc.abstractmethod
    def dataset(self) -> Dataset:
        """Returns list of data samples."""
        pass

    @property
    @abc.abstractmethod
    def epoch_size(self) -> int:
        """Returns number of data samples per epoch."""
        pass

    @property
    @abc.abstractmethod
    def is_last_batch(self) -> bool:
        """Returns true if end of data list is reached."""
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset current batch index in dataset."""
        pass

    @abc.abstractmethod
    def sampling(self, batch_size: int) -> tp.List:
        """Returns next batch from data."""
        pass
