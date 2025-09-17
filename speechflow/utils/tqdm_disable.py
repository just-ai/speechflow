from contextlib import contextmanager

import tqdm

__all__ = ["tqdm_disable"]


@contextmanager
def tqdm_disable():
    tqdm_backup = tqdm.tqdm

    def tqdm_replacement(iterable_object=None, *args, **kwargs):
        if iterable_object is None:
            return tqdm_backup(disable=True)
        else:
            return iterable_object

    tqdm.tqdm = tqdm_replacement

    try:
        yield
    except Exception as e:
        raise e
    finally:
        tqdm.tqdm = tqdm_backup
