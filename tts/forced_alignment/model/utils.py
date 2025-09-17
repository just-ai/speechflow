import numpy as np
import torch

from numba import jit, prange
from torch.nn import functional as torch_func


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def squeeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()

    t = (t // n_sqz) * n_sqz
    x = x[:, :, :t]
    x_sqz = x.view(b, c, t // n_sqz, n_sqz)
    x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

    if x_mask is not None:
        x_mask = x_mask[:, :, n_sqz - 1 :: n_sqz]
    else:
        x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
    return x_sqz * x_mask, x_mask


def unsqueeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()

    x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
    x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

    if x_mask is not None:
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
    else:
        x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
    return x_unsqz * x_mask, x_mask


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def maximum_path(
    value,
    mask,
    max_neg_val=-np.inf,
    sil_mask=None,
    spectral_flatness=None,
    max_frames_per_phoneme=1,
):
    """Numpy-friendly version.

    It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]

    """
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)

    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)

    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    max_index = index
    index_range = np.arange(b)

    if sil_mask is not None:
        ph_len = np.zeros(b, dtype=np.int64)
        thr = np.zeros(b, dtype=np.int64) + max_frames_per_phoneme
        sf = np.zeros(b, dtype=np.float32)

    try:
        for j in reversed(range(t_y)):
            path[index_range, index, j] = 1
            d = direction[index_range, index, j]

            if sil_mask is not None:
                ph_len += d
                d[np.logical_and(ph_len >= thr, ~sil_mask[index_range, index])] = 0

                if spectral_flatness is not None:
                    sf += spectral_flatness[index_range, j]
                    sil_index = np.logical_and(d == 0, sil_mask[index_range, index])
                    if True in sil_index:
                        mean_sf = sf[sil_index] / ph_len[sil_index].clip(min=1)
                        prev_index = (index + 1).clip(max=max_index)
                        k = 0
                        for b, p in enumerate(sil_index):
                            if p and mean_sf[k] > 0.9:
                                path[b, index[b], j + 1 : j + ph_len[b] + 1] = 0
                                path[b, prev_index[b], j + 1 : j + ph_len[b] + 1] = 1
                                k += 1

                ph_len[d == 0] = 0
                sf[d == 0] = 0

            index = index + d - 1

            if sil_mask is not None and 0 in d:
                thr_idx = np.logical_or(
                    sil_mask[index_range, (index - 1).clip(min=0)],
                    sil_mask[index_range, (index + 1).clip(max=max_index)],
                )
                thr[thr_idx] = max_frames_per_phoneme
                thr[~thr_idx] = 4 * max_frames_per_phoneme

    except Exception as e:
        print(e)

    path = path * mask
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path


def convert_pad_shape(pad_shape):
    dim = pad_shape[::-1]
    pad_shape = [item for sublist in dim for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    """
    duration: [b, t_x]
    mask: [b, t_x, t_y]
    """
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = (
        path - torch_func.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    )
    path *= mask
    return path


@jit(nopython=True)
def mas(attn_map, width=1):
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_j = np.arange(max(0, j - width), j + 1)
            prev_log = np.array([log_p[i - 1, prev_idx] for prev_idx in prev_j])

            ind = np.argmax(prev_log)
            log_p[i, j] = attn_map[i, j] + prev_log[ind]
            prev_ind[i, j] = prev_j[ind]

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1

    assert opt.sum(0).all()
    assert opt.sum(1).all()
    return opt


@jit(nopython=True)
def mas_width1(log_attn_map):
    """mas with hardcoded width=1."""
    # assumes mel x text
    neg_inf = log_attn_map.dtype.type(-np.inf)
    log_p = log_attn_map.copy()
    log_p[0, 1:] = neg_inf
    for i in range(1, log_p.shape[0]):
        prev_log1 = neg_inf
        for j in range(log_p.shape[1]):
            prev_log2 = log_p[i - 1, j]
            log_p[i, j] += max(prev_log1, prev_log2)
            prev_log1 = prev_log2

    # now backtrack
    opt = np.zeros_like(log_p)
    one = opt.dtype.type(1)
    j = log_p.shape[1] - 1
    for i in range(log_p.shape[0] - 1, 0, -1):
        opt[i, j] = one
        if log_p[i - 1, j - 1] >= log_p[i - 1, j]:
            j -= 1
            if j == 0:
                opt[1:i, j] = one
                break

    opt[0, j] = one
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_log_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_log_attn_map)

    for b in prange(b_log_attn_map.shape[0]):
        out = mas_width1(b_log_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out

    return attn_out


def binarize_attention(attn, in_len, out_len):
    """Convert soft attention matrix to hard attention matrix.

    Args:
        attn (torch.Tensor): B x 1 x max_mel_len x max_text_len. Soft attention matrix.
        in_len (torch.Tensor): B. Lengths of texts.
        out_len (torch.Tensor): B. Lengths of spectrograms.

    Output:
        attn_out (torch.Tensor): B x 1 x max_mel_len x max_text_len. Hard attention matrix, final dim max_text_len should sum to 1.

    """
    b_size = attn.shape[0]
    with torch.no_grad():
        attn_cpu = attn.data.cpu().numpy()
        attn_out = torch.zeros_like(attn)
        for ind in range(b_size):
            hard_attn = mas(attn_cpu[ind, 0, : out_len[ind], : in_len[ind]])
            attn_out[ind, 0, : out_len[ind], : in_len[ind]] = torch.tensor(
                hard_attn, device=attn.device
            )

    return attn_out


def binarize_attention_parallel(attn, in_lens, out_lens):
    """For training purposes only. Binarizes attention with MAS. These will no longer
    receive a gradient.

    Args:
        attn: B x 1 x max_mel_len x max_text_len

    """
    with torch.no_grad():
        log_attn_cpu = torch.log(attn.data).cpu().numpy()
        attn_out = b_mas(
            log_attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1
        )

    return torch.from_numpy(attn_out).to(attn.device)
