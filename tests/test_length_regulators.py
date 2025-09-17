import numpy as np
import torch
import pytest

from speechflow.utils.profiler import Profiler
from tts.acoustic_models.modules.common import LengthRegulator, SoftLengthRegulator

DEVICE = "cpu"
LR = LengthRegulator().to(DEVICE)
SA = SoftLengthRegulator().to(DEVICE)
BS = 64


def prepare_inputs(n=10):
    embs = []
    durs = []
    for _ in range(n):
        batch_size = BS
        seq_len = np.random.randint(1, 164)
        hidden_dim = np.random.randint(1, 256)
        durs.append(torch.randint(1, 10, size=(batch_size, seq_len)).float().to(DEVICE))
        embs.append(torch.randn(size=(batch_size, seq_len, hidden_dim)).to(DEVICE))

    use_max_len = [True if np.random.randint(2) == 0 else False for _ in range(n)]

    return embs, durs, use_max_len


@torch.inference_mode()
@pytest.mark.parametrize("embeddings, lengths, use_max_len", list(zip(*prepare_inputs())))
def test_len_regulators(embeddings, lengths, use_max_len):
    max_len = lengths.sum(dim=1).max().int().item() if use_max_len else None
    lr_result, _ = LR(embeddings, lengths, max_len)
    sa_result, _ = SA(embeddings, lengths, max_len)
    assert lr_result.size() == sa_result.size()


@pytest.mark.parametrize("embeddings, lengths, use_max_len", [prepare_inputs(100)])
def test_softaligner_speedup(embeddings, lengths, use_max_len):

    with Profiler("Classic LengthRegulator"):
        for e, l, m in zip(embeddings, lengths, use_max_len):
            max_len = l.sum(dim=1).max().int().item() if use_max_len else None
            _ = LR(e, l, max_len)

    with Profiler("SoftAligner LengthRegulator"):
        for e, l, m in zip(embeddings, lengths, use_max_len):
            max_len = l.sum(dim=1).max().int().item() if use_max_len else None
            _ = SA(e, l, max_len)
