import random
import typing as tp

from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

ENABLE_ASSERTS: bool = False


def stack(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = []
    for i, batch in enumerate(input_ele):
        if batch.ndim == 1:
            one_batch_padded = F.pad(batch, (0, max_len - batch.size(0)), "constant", 0.0)
        elif batch.ndim == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        else:
            raise AttributeError
        out_list.append(one_batch_padded)

    out_padded = torch.stack(out_list)
    return out_padded


@lru_cache(maxsize=16)
@torch.no_grad()
def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths)

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)
    # assert (get_lengths_from_mask(mask) == lengths).all()
    return mask.detach()  # type: ignore


@lru_cache(maxsize=16)
@torch.no_grad()
def get_lengths_from_mask(mask):
    if ENABLE_ASSERTS:
        assert mask.dtype == torch.bool
        assert mask[:, 0].sum() > 0

    return mask.sum(-1).long().detach()


@lru_cache(maxsize=16)
@torch.no_grad()
def get_lengths_from_durations(durations):
    return durations.sum(1).floor().long().detach()


def apply_mask(t, mask):
    if t.ndim > 4:
        raise RuntimeError(f"Shape out {t.ndim} is not supported")

    if t.ndim == 4:
        return t * mask

    if ENABLE_ASSERTS:
        assert mask.dtype == torch.bool
        if mask.ndim == 2:
            assert mask[:, 0].sum() > 0
        else:
            assert mask[:, 0, 0].sum() > 0

    if t.ndim != mask.ndim:
        if t.shape[1] == mask.shape[1]:
            mask = mask.unsqueeze(-1)
        else:
            mask = mask.unsqueeze(1)

    return t * mask


def masked_fill(t, mask, value):
    if t.ndim > 3:
        raise RuntimeError(f"Shape out {t.ndim} is not supported")

    if ENABLE_ASSERTS:
        assert mask.dtype == torch.bool
        if mask.ndim == 2:
            assert mask[:, 0].sum() > 0
        else:
            assert mask[:, 0, 0].sum() > 0

    if t.ndim != mask.ndim:
        if t.shape[1] == mask.shape[1]:
            mask = mask.unsqueeze(-1)
        else:
            mask = mask.unsqueeze(1)

    return t.masked_fill(~mask, value)


def split_batch_by_lengths(
    chunks_lens: torch.Tensor,
    ilens: torch.Tensor,
    x: torch.Tensor,
    target: tp.Optional[torch.Tensor],
):
    """Split every batch item by its syntagmas.

    :param chunks_lens:
        lengths of chunks with shape [batch_size, max_chunk_count] where
        max_synt_count is a maximum number of chunks across batch elements.
    :param ilens:
        actual lengths of sequence without padding with shape [batch_size, 1].
    :param x:
        input tensor of shape [batch_size, seq_len, hidden_dim]
    :param target:
        just another tensor to split in the same manner. Can be two dimensional with shape [batch_size, seq_len]
    :return:
        new batch with every batch item corresponding to one chunk.

    """
    chunk_batch = defaultdict(list)
    for i in range(x.shape[0]):
        current_chunk = tuple(chunks_lens[i, :][chunks_lens[i, :] > 0])  # simplify

        chunk_batch["x"].extend(torch.split(x[i][: ilens[i]], current_chunk, dim=0))

        if target is not None:
            chunk_batch["target"].extend(
                torch.split(target[i][: ilens[i]], current_chunk, dim=0)
            )

    max_chunk_len = torch.max(chunks_lens)
    chunk_x = stack(chunk_batch["x"], max_chunk_len)
    chunk_target = (
        stack(chunk_batch.get("target"), max_chunk_len) if target is not None else None
    )
    chunk_src_mask = get_mask_from_lengths(chunks_lens[chunks_lens > 0])
    return chunk_x, chunk_target, chunk_src_mask


def assemble_batch_by_lengths(chunks_lens, x, max_len):
    """Assemble batch by chunks according to syntagmas_lens.

    Reverse original batch_size after modyfyng batch with `split_batch_by_lengths`.

    """
    select_without_padding = x.shape[1] != 1
    new_batch = []
    batch_shape = chunks_lens.shape[0]
    start = 0
    for i in range(batch_shape):
        current_batch_len = len(
            [length for length in tuple(chunks_lens[i, :]) if int(length) != 0]
        )
        index = torch.arange(start, start + current_batch_len).to(x.device)
        current_batch = torch.index_select(x, dim=0, index=index)
        current_batch = current_batch.reshape(-1, x.shape[-1])
        if select_without_padding:
            # select with respect to padding
            non_zero_index = []
            current_chunk_start = 0
            for j, chunk_len in enumerate(chunks_lens[i, :]):
                if chunk_len == 0:
                    continue
                synt_non_zero_idx = torch.arange(
                    current_chunk_start, current_chunk_start + chunk_len
                ).to(x.device)
                non_zero_index.extend(synt_non_zero_idx)
                current_chunk_start = x.shape[1] * (j + 1)
            current_batch = torch.index_select(
                current_batch, dim=0, index=torch.tensor(non_zero_index).to(x.device)
            )
            current_batch = current_batch.reshape(-1, x.shape[-1])
        new_batch.append(current_batch)
        start = start + current_batch_len
    new_batch = stack(new_batch, max_len)
    return new_batch


def merge_additional_outputs(input_dict, sequence_of_dicts):
    """Expand additional output dict with a sequence of dicts.

    :param input_dict: dict
    :param sequence_of_dicts: tuple of dicts

    """
    input_dict = {} if input_dict is None else input_dict
    if isinstance(sequence_of_dicts, dict):
        sequence_of_dicts = [sequence_of_dicts]

    for dictionary in sequence_of_dicts:
        if dictionary:
            input_dict.update(dictionary)

    return input_dict


def run_rnn_on_padded_sequence(
    rnn,
    seq: torch.Tensor,
    seq_lengths: torch.Tensor,
    batch_first: bool = True,
    enforce_sorted: bool = False,
):
    if seq_lengths is None:
        return rnn(seq)[0]
    else:
        # pytorch tensor are not reversible, hence the conversion
        seq_lengths = seq_lengths.cpu()
        seq = torch.nn.utils.rnn.pack_padded_sequence(
            seq, seq_lengths, batch_first=batch_first, enforce_sorted=enforce_sorted
        )

        rnn.flatten_parameters()
        seq, _ = rnn(seq.float())

        seq, _ = torch.nn.utils.rnn.pad_packed_sequence(seq, batch_first=batch_first)
        return seq


def tensor_masking(
    tensor: torch.tensor, mask_value: float, step: int = 4
) -> torch.tensor:
    tensor = tensor.clone()
    prev_k = 0
    for k in range(step, tensor.shape[0] + 1, step):
        if random.random() < 0.2:
            tensor[prev_k:k, :, :] = mask_value
            continue

        mask_type = random.randint(0, 5)
        if mask_type == 0:
            a = 0
            b = a + random.randint(0, tensor.shape[1] - a)
            tensor[prev_k:k, a:b, :] = mask_value
        elif mask_type == 5:
            a = random.randint(0, tensor.shape[1] - 1)
            b = tensor.shape[1]
            tensor[prev_k:k, a:b, :] = mask_value
        else:
            for p in range(1, random.randint(1, 4)):
                m_len = random.randint(tensor.shape[1] // 10, tensor.shape[1] // 5)
                a = random.randint(0, tensor.shape[1] - m_len - 1)
                b = a + m_len
                tensor[prev_k:k, a:b, :] = mask_value

        prev_k = k

    return tensor


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Sinusoid position encoding table."""

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)
