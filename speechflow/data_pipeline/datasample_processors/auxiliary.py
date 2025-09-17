from copy import deepcopy as copy

from speechflow.data_pipeline.core import DataSample
from speechflow.data_pipeline.core.registry import PipeRegistry

__all__ = ["move_field", "store_field", "restore_field"]


@PipeRegistry.registry()
def move_field(ds: DataSample, key: str, as_key: str):
    attr = getattr(ds, key, None)
    if attr is not None:
        setattr(ds, as_key, copy(attr))

    return ds


@PipeRegistry.registry()
def store_field(ds: DataSample, key: str, as_key: str):
    attr = getattr(ds, key, None)
    if attr is not None:
        if hasattr(attr, "get"):
            attr = attr.get()

        ds.additional_fields[as_key] = copy(attr)

    return ds


@PipeRegistry.registry()
def restore_field(ds: DataSample, key: str, as_key: str):
    attr = ds.additional_fields.get(key)
    if attr is not None:
        setattr(ds, as_key, copy(attr))

    return ds
