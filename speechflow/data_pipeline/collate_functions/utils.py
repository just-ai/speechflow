import typing as tp

import torch

from speechflow.utils.pad_utils import pad_1d, pad_2d

__all__ = [
    "collate_integers",
    "collate_vectors",
    "collete_1d",
    "collete_2d",
    "collate_sequence",
]


def _get_pad_val(
    attr_name, pad_values: tp.Optional[tp.Union[float, tp.MutableMapping]] = None
) -> float:
    pad_val = (
        pad_values.get(attr_name, 0.0)
        if isinstance(pad_values, tp.MutableMapping)
        else pad_values
    )
    return 0 if pad_values is None else pad_val


def _get_multiple_val(
    attr_name, multiple_values: tp.Optional[tp.Union[int, tp.MutableMapping]] = None
) -> int:
    multiple_val = (
        multiple_values.get(attr_name)
        if isinstance(multiple_values, tp.MutableMapping)
        else multiple_values
    )
    return multiple_val


def collate_integers(
    batch: tp.List[tp.Any], attr_name: str
) -> tp.Optional[torch.LongTensor]:
    if getattr(batch[0], attr_name) is not None:
        fields: tp.List[torch.Tensor] = [getattr(x, attr_name) for x in batch]
        return torch.LongTensor(fields)
    else:
        return None


def collate_vectors(batch: tp.List[tp.Any], attr_name: str) -> tp.Optional[torch.Tensor]:
    if getattr(batch[0], attr_name) is not None:
        fields: tp.List[torch.Tensor] = [getattr(x, attr_name) for x in batch]
        return torch.vstack(fields)
    else:
        return None


def collete_1d(
    batch: tp.List[tp.Any],
    attr_name: str,
    pad_values: tp.Optional[tp.Union[float, tp.MutableMapping]] = None,
    multiple_values: tp.Optional[tp.Union[int, tp.MutableMapping]] = None,
) -> tp.Tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]:
    pad_val = _get_pad_val(attr_name, pad_values)
    multiple_val = _get_multiple_val(attr_name, multiple_values)

    if getattr(batch[0], attr_name) is not None:
        fields: tp.List[torch.Tensor] = [getattr(x, attr_name) for x in batch]
        x, x_lens = pad_1d(fields, pad_val, multiple_val)
        return x, x_lens
    else:
        return None, None


def collete_2d(
    batch: tp.List[tp.Any],
    attr_name: str,
    pad_values: tp.Optional[tp.Union[float, tp.MutableMapping]] = None,
    multiple_values: tp.Optional[tp.Union[int, tp.MutableMapping]] = None,
) -> tp.Tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]:
    pad_val = _get_pad_val(attr_name, pad_values)
    multiple_val = _get_multiple_val(attr_name, multiple_values)

    if getattr(batch[0], attr_name) is not None:
        fields: tp.List[torch.Tensor] = [getattr(x, attr_name) for x in batch]
        if hasattr(fields[0], "get"):
            fields = [x.get() for x in fields]  # type: ignore
        if fields[0].ndim == 1:
            fields = [x.unsqueeze(-1) for x in fields]

        x, x_lens = pad_2d(fields, fields[0].shape[1], pad_val, multiple_val)
        return x, x_lens
    else:
        return None, None


def collate_sequence(
    batch: tp.List[tp.Any],
    attr_name: str,
    pad_values: tp.Optional[tp.Union[str, float, tp.MutableMapping]] = None,
    multiple_values: tp.Optional[tp.Union[int, tp.MutableMapping]] = None,
) -> tp.Tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]:
    pad_val = _get_pad_val(attr_name, pad_values)
    multiple_val = _get_multiple_val(attr_name, multiple_values)

    if getattr(batch[0], attr_name) is not None:
        seq = [getattr(sample, attr_name) for sample in batch]
        if hasattr(seq[0], "get"):
            seq = [x.get() for x in seq]  # type: ignore
        if getattr(batch[0], attr_name).ndim == 1:
            seq, seq_lens = pad_1d(seq, pad_val, multiple_val)
        else:
            seq, seq_lens = pad_2d(seq, seq[0].shape[1], pad_val, multiple_val)

        seq_lens = torch.LongTensor(seq_lens)
        return seq, seq_lens
    else:
        return None, None
