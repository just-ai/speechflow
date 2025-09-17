import time
import random
import hashlib

import numpy as np


def get_seed() -> int:
    return time.time_ns() % (10**8)


def get_seed_from_string(s: str) -> int:
    if s.isdigit():
        return int(s) % (10**8)
    elif s == "-1":
        return get_seed()
    else:
        return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10**8)


def check_seed(seed: int):
    return get_seed_from_string(str(seed))


def set_random_seed(seed: int):
    random.seed(check_seed(seed))


def set_numpy_seed(seed: int):
    np.random.seed(check_seed(seed))


def set_torch_seed(seed: int):
    import torch

    torch.manual_seed(check_seed(seed))
    torch.cuda.manual_seed(check_seed(seed))


def set_all_seed(seed: int):
    set_random_seed(seed)
    set_numpy_seed(seed)
    set_torch_seed(seed)


if __name__ == "__main__":
    print(get_seed())
    print(get_seed_from_string("34"))
    print(get_seed_from_string("-1"))
    print(get_seed_from_string("seed"))

    set_random_seed(get_seed())
    set_numpy_seed(get_seed())
    set_torch_seed(get_seed())
    set_all_seed(get_seed())
