import typing as tp

from collections import deque
from itertools import product


def needleman_wunsch(x: tp.Sequence, y: tp.Sequence):
    """Run the Needleman-Wunsch algorithm on two sequences.

    Code based on pseudocode in Section 3 of:
    Naveed, Tahir; Siddiqui, Imitaz Saeed; Ahmed, Shaftab.
    "Parallel Needleman-Wunsch Algorithm for Grid." n.d.
    https://upload.wikimedia.org/wikipedia/en/c/c4/ParallelNeedlemanAlgorithm.pdf

    Source: https://johnlekberg.com/blog/2020-10-25-seq-align.html

    """
    N, M = len(x), len(y)

    def s(a, b):
        return int(a == b)

    DIAG = -1, -1
    LEFT = -1, 0
    UP = 0, -1

    # Create tables F and Ptr
    F = {}
    Ptr: tp.Dict[tp.Tuple[tp.Any, tp.Any], tp.Any] = {}

    F[-1, -1] = 0
    for i in range(N):
        F[i, -1] = -i
    for j in range(M):
        F[-1, j] = -j

    option_Ptr = DIAG, LEFT, UP
    for i, j in product(range(N), range(M)):
        option_F = (
            F[i - 1, j - 1] + s(x[i], y[j]),
            F[i - 1, j] - 1,
            F[i, j - 1] - 1,
        )
        F[i, j], Ptr[i, j] = max(zip(option_F, option_Ptr))

    # Work backwards from (N - 1, M - 1) to (0, 0)
    # to find the best alignment.
    alignment: tp.Deque = deque()
    element: tp.Tuple[tp.Optional[int], tp.Optional[int]] = (None, None)
    i, j = N - 1, M - 1
    while i >= 0 and j >= 0:
        direction = Ptr[i, j]
        if direction == DIAG:
            element = i, j
        elif direction == LEFT:
            element = i, None
        elif direction == UP:
            element = None, j
        else:
            raise AttributeError
        alignment.appendleft(element)
        di, dj = direction
        i, j = i + di, j + dj
    while i >= 0:
        alignment.appendleft((i, None))
        i -= 1
    while j >= 0:
        alignment.appendleft((None, j))
        j -= 1

    return list(alignment)


def locate_replacements_from_alignment(
    alignment: tp.Deque, seq_a: tp.Sequence, seq_b: tp.Sequence
):
    matches = [
        seq_a[i_a] == seq_b[i_b] if (i_a is not None) and (i_b is not None) else False
        for i_a, i_b in alignment
    ]
    start_idxs_orig = []
    start_idxs_replaced = []
    lens_orig = []
    lens_replaced = []

    is_replacement = False
    len_orig = 0
    len_repl = 0
    for i, m in enumerate(matches):
        if not m:
            if not is_replacement:
                orig, rep = alignment[i]
                start_idxs_orig.append(orig if orig is not None else 0)
                start_idxs_replaced.append(rep if rep is not None else 0)
                is_replacement = True
                len_orig = 1 if orig is not None else 0
                len_repl = 1 if rep is not None else 0
            else:
                orig, rep = alignment[i]
                len_orig += int(orig is not None)
                len_repl += int(rep is not None)
        else:
            if is_replacement:
                lens_orig.append(len_orig)
                lens_replaced.append(len_repl)
                is_replacement = False
                len_orig = 0
                len_repl = 0

    if is_replacement:
        lens_orig.append(len_orig)
        lens_replaced.append(len_repl)

    end_idx_orig = tuple([x + y for x, y in zip(start_idxs_orig, lens_orig)])
    end_idx_rep = tuple([x + y for x, y in zip(start_idxs_replaced, lens_replaced)])

    replacement_map = {
        x: y
        for x, y in zip(
            zip(start_idxs_orig, end_idx_orig), zip(start_idxs_replaced, end_idx_rep)
        )
    }
    return replacement_map


def get_replacement_map(alignment, seq_after):
    _inversed_map = dict([x for x in alignment if None not in x])
    _map = {v: k for k, v in _inversed_map.items()}
    previous_target_idx = 0

    for token_idx in range(len(seq_after)):
        if token_idx not in _map:
            _map[token_idx] = previous_target_idx
        target_idx = _map[token_idx]

        if target_idx > previous_target_idx + 1:
            _map[token_idx - 1] = tuple(range(previous_target_idx, target_idx))
        previous_target_idx = target_idx

    return _map


def expand_iterable(
    to_expand: tp.Sequence[tp.Any], expand_coefficients: tp.Sequence[int]
):
    if len(to_expand) != len(expand_coefficients):
        raise ValueError("Lengths don't match.")
    result = []
    for m, d in zip(to_expand, expand_coefficients):
        result += [m] * d
    return result


def expand_iterable_with_float_weights(
    to_expand: tp.Sequence[tp.Any], expand_coefficients: tp.Sequence[float]
):
    if len(to_expand) != len(expand_coefficients):
        raise ValueError("Lengths don't match.")
    max_len = int(sum(expand_coefficients) + 0.5)  # round to closest
    csum = 0.0
    result: tp.List[tp.Any] = [False] * max_len
    for m, d in zip(to_expand, expand_coefficients):
        start_idx = int(csum)
        end_idx = int(csum + d)
        for idx in range(start_idx, end_idx):
            result[idx] = m
        csum += d
    return result
