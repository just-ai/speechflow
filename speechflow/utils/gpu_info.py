import sys
import subprocess

from os import environ as env

import numpy as np

__all__ = ["get_gpu_count", "get_freer_gpu", "get_total_gpu_memory"]


def get_gpu_count() -> int:
    import torch

    return torch.cuda.device_count()


def get_freer_gpu(strict: bool = True) -> int:
    if sys.platform == "win32":
        return 0

    gpu_count = get_gpu_count()
    if gpu_count == 0:
        raise RuntimeError("GPU device not found!")

    arch = subprocess.check_output(
        "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used", shell=True
    )

    memory_used = [int(x.split()[2]) for x in arch.decode("utf-8").split("\n") if x]

    if "CUDA_VISIBLE_DEVICES" in env:
        v = [int(item) for item in env["CUDA_VISIBLE_DEVICES"].split(",")]
        memory_used = [item for idx, item in enumerate(memory_used) if idx in v]

    free_gpu = int(np.argmin(memory_used))

    if strict and memory_used[free_gpu] > 100:
        raise RuntimeError("All GPUs are busy!")

    return free_gpu


def get_total_gpu_memory(gpu_index: int) -> float:  # in GB
    import torch

    return torch.cuda.get_device_properties(gpu_index).total_memory / 1024**3


if __name__ == "__main__":
    import torch

    _data = []
    for _strict in [True, False]:
        for i in range(get_gpu_count()):
            _gpu_idx = get_freer_gpu(strict=_strict)
            _device = f"cuda:{_gpu_idx}"
            _data.append(torch.FloatTensor([1]).to(_device))
            print(f"[strict={_strict}] change device {_device}")
