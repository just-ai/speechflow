import abc
import typing as tp


class AbstractDataProcessor:
    """Base class for data processors."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """Everything that we need to init."""
        pass

    @abc.abstractmethod
    def do_preprocessing(
        self,
        samples: tp.List[tp.Any],
        preproc_fn: tp.Sequence[tp.Callable],
        skip_corrupted_samples: bool = True,
    ) -> tp.List[tp.Any]:
        """Apply processing functions."""
        pass

    @abc.abstractmethod
    def process(self, samples: tp.List[tp.Any]) -> tp.Any:
        """Returns processed data."""
        pass
