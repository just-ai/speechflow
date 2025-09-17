import typing as tp

from operator import itemgetter

import torch

from torch import Tensor

__all__ = ["pad_1d", "pad_2d"]


def pad_1d(
    sequences: tp.Sequence[Tensor],
    pad_val: tp.Union[int, float] = 0,
    multiple: tp.Optional[int] = None,
):
    lens = [len(x) for x in sequences]
    index, max_len = max(enumerate(lens), key=itemgetter(1))
    dtype = sequences[0].dtype

    if multiple is not None:
        pad_len = multiple - max_len % multiple
        if pad_len == multiple:
            pad_len = 0
    else:
        pad_len = 0

    if not sequences[0].is_floating_point():
        pad_val = int(pad_val)

    tensor_to_fill = (
        torch.zeros((len(sequences), max_len + pad_len), dtype=dtype) + pad_val
    )
    for i, s in enumerate(sequences):
        tensor_to_fill[i, : lens[i]] = s

    # lens[index] += pad_len
    return tensor_to_fill, torch.LongTensor(lens)


def pad_2d(
    sequences: tp.Sequence[Tensor],
    n_channel: int,
    pad_val: tp.Union[int, float] = 0,
    multiple: tp.Optional[int] = None,
):
    lens = [len(x) for x in sequences]
    index, max_len = max(enumerate(lens), key=itemgetter(1))
    dtype = sequences[0].dtype

    if multiple is not None:
        pad_len = multiple - max_len % multiple
        if pad_len == multiple:
            pad_len = 0
    else:
        pad_len = 0

    if not sequences[0].is_floating_point():
        pad_val = int(pad_val)

    tensor_to_fill = (
        torch.zeros((len(lens), max_len + pad_len, n_channel), dtype=dtype) + pad_val
    )
    for i, s in enumerate(sequences):
        tensor_to_fill[i, : lens[i], :] = s

    # lens[index] += pad_len
    return tensor_to_fill, torch.LongTensor(lens)


def pad_3d(
    sequences: tp.Sequence[Tensor],
    pad_id: tp.Union[int, float] = 0,
    width: int = 8,
    height: int = 80,
):
    lens = [len(x) for x in sequences]
    max_len = max(lens)
    dtype = sequences[0].dtype

    if not sequences[0].is_floating_point():
        pad_id = int(pad_id)

    tensor_to_fill = (
        torch.zeros((len(sequences), max_len, width, height), dtype=dtype) + pad_id
    )
    for i, s in enumerate(sequences):
        tensor_to_fill[i, : lens[i], :, :] = s

    return tensor_to_fill, lens
