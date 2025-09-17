import typing as tp

from dataclasses import dataclass

__all__ = ["Batch"]


@dataclass
class Batch:
    size: int
    is_last: bool = False
    data_samples: tp.Optional[tp.List[tp.Any]] = None
    collated_samples: tp.Optional[tp.Any] = None
    tag: tp.Optional[str] = None

    def __len__(self):
        return self.size
