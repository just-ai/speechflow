import abc
import typing as tp


class AbstractBatchProcessor:
    """Base class for data processors."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """Everything that we need to init."""
        pass

    @abc.abstractmethod
    def __call__(
        self,
        batch: tp.Any,
        batch_idx: int,
        global_step: int,
    ) -> tp.Tuple[tp.Any, tp.Any, tp.Optional[tp.List]]:
        """Returns processed batch."""
        pass
