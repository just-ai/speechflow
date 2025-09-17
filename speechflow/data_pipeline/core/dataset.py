import pickle
import typing as tp

from collections import namedtuple
from contextlib import contextmanager
from os import environ as env
from pathlib import Path

from tqdm import tqdm

from speechflow.utils.checks import str_to_bool

__all__ = ["Dataset", "DatasetItem"]


class DatasetItem:
    __slots__ = ["data", "meta", "cache"]

    def __init__(self, data: tp.Any, meta: tp.Dict = None):
        self.data = data
        self.meta = meta
        self.cache = None

    def __len__(self):
        return self.meta.get("len") if self.meta else 0

    def __str__(self):
        return self.meta.get("str") if self.meta else ""

    def __getattr__(self, attr):
        _unpickle = object.__getattribute__(self, "_unpickle")
        _data = object.__getattribute__(self, "data")
        if not attr.startswith("_"):
            return getattr(_unpickle(_data), attr)
        else:
            return object.__getattribute__(self, attr)

    def get(self):
        return self._unpickle(self.data, full=True)

    @property
    def filepath(self):
        return self.meta.get("filepath") if self.meta else None

    @staticmethod
    def _unpickle(data, full: bool = False):
        try:
            if hasattr(data, "deserialize"):
                data = data.deserialize(full=full)
            elif data.__class__.__name__ == "bytes":
                data = pickle.loads(data)
        finally:
            return data


class Dataset:
    def __init__(self, data: tp.Optional[tp.Iterable] = None, use_serialize: bool = True):
        self._samples: tp.List[tp.Any] = []
        self._use_serialize = use_serialize
        self._readonly_mode: bool = False
        self._mem_save = str_to_bool(env.get("MEMORY_SAVE", "False"))

        if not data:
            return

        assert isinstance(data, tp.Iterable)

        for item in data:
            if isinstance(item, Dataset):
                self._samples += item._samples
            else:
                self._samples.append(self._serialize(item))

    def __del__(self):
        self.clear()

    @property
    def samples(self):
        return self._samples

    @staticmethod
    def _find_filepath(obj: tp.Any) -> tp.Optional[str]:
        for attr in ["file_path", "filepath"]:
            if isinstance(obj, tp.MutableMapping):
                path = obj.get(attr, None)
            else:
                path = getattr(obj, attr, None)
            if path is not None:
                break
        else:
            for item in getattr(obj, "__dict__", {}).values():
                if isinstance(item, Path):
                    path = item
                    break
            else:
                path = None

        if isinstance(path, Path):
            return path.as_posix()
        else:
            return path

    def _serialize(self, obj: tp.Any, **kwargs) -> DatasetItem:
        if obj.__class__.__name__ == "DatasetItem":
            return obj

        meta = kwargs.get("meta", {})

        if obj is not None and not meta:
            if hasattr(obj, "__len__"):
                meta["len"] = len(obj)
            else:
                meta["len"] = 0
            meta["str"] = str(obj)
            meta["filepath"] = self._find_filepath(obj)

        try:
            if self._use_serialize:
                if not self._mem_save and hasattr(obj, "serialize"):
                    obj = obj.serialize(**kwargs)
                elif obj.__class__.__name__ != "bytes":
                    obj = pickle.dumps(obj)
        finally:
            return DatasetItem(obj, meta)

    def _deserialize(self, item: DatasetItem, **kwargs):
        if item.cache:
            return item.cache, item.meta

        obj = item.data
        if not self._use_serialize:
            return obj, item.meta

        try:
            if not self._mem_save and hasattr(obj, "deserialize"):
                obj = obj.deserialize(**kwargs)
            elif obj.__class__.__name__ == "bytes":
                obj = pickle.loads(obj)
        finally:
            return obj, item.meta

    def __getitem__(self, index: int):
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self._samples[i] for i in indices]

        return self._samples[index]

    def __setitem__(self, index: int, value: tp.Any):
        self._samples[index] = self._serialize(value)

    def __add__(self, other: tp.List):
        if isinstance(other, tp.List):
            for item in other:
                self.append(item)
        elif isinstance(other, Dataset):
            self._samples += other._samples
        else:
            raise NotImplementedError

        return self

    def __len__(self):
        return len(self._samples)

    def __bool__(self):
        return len(self) > 0

    def __iter__(self):
        for index in range(len(self)):
            obj, meta = self._deserialize(self._samples[index])
            yield obj
            if not self._readonly_mode:
                self._samples[index] = self._serialize(obj, meta=meta)

    def __delitem__(self, index: int):
        del self._samples[index]

    def __hash__(self):
        return len(self._samples)

    def item(self, index: int):
        return self._samples[index].get()

    def append(self, obj: tp.Any):
        self._samples.append(self._serialize(obj))

    def pop(self, index: int):
        return self._samples.pop(index).get()

    def extend(self, obj: tp.Iterable):
        if isinstance(obj, tp.Iterable):
            if isinstance(obj, Dataset):
                self._samples += obj._samples
            else:
                for item in obj:
                    self._samples.append(self._serialize(item))
        else:
            raise NotImplementedError

    def modify(self, index: int, func: tp.Callable):
        obj, meta = self._deserialize(self._samples[index])
        func(obj)
        self._samples[index] = self._serialize(obj, meta=meta)

    def filter(self, func: tp.Callable):
        self._samples = [
            item
            for item in tqdm(self._samples, desc="Dataset filtering", leave=False)
            if func(self._deserialize(item)[0])
        ]

    def sort(self, *args, **kwargs):
        self._samples.sort(*args, **kwargs)

    def clear(self):
        self._samples = []

    def to_list(self):
        return [self._deserialize(item, full=True)[0] for item in self._samples]

    def slice(self, fields: tp.Iterable[str]):
        result = {field: [] for field in fields}

        with self.readonly():
            for item in self:
                for field in fields:
                    result[field].append(getattr(item, field))

        return result

    @contextmanager
    def readonly(self, cache_fields: tp.Optional[tp.Iterable[str]] = None):
        self._readonly_mode = True

        if cache_fields:
            for index in tqdm(
                range(len(self)),
                desc=f"Caching fields: [{', '.join(cache_fields)}]",
                leave=False,
            ):
                item = self._samples[index]
                item.cache = None

                obj, meta = self._deserialize(item)
                cache = {field: getattr(obj, field) for field in cache_fields}
                item.cache = namedtuple("DatasetItemCache", cache.keys())(*cache.values())

        yield

        if cache_fields:
            for index in range(len(self)):
                self._samples[index].cache = None

        self._readonly_mode = False

    def get_file_list(self) -> tp.Tuple[str, ...]:
        return tuple(item.filepath for item in self._samples if item.filepath)


if __name__ == "__main__":

    from multilingual_text_parser.data_types import Doc

    from speechflow.data_pipeline.datasample_processors.data_types import TTSDataSample
    from speechflow.logging.logger import create_logger
    from speechflow.utils.profiler import MemoryProfiler

    LOGGER = create_logger("root")

    ds = TTSDataSample(
        sent=Doc("Съешь ещё этих мягких французских булочек и выпей чаю.", True, True)
    )

    pl = Dataset()

    pl.append(None)
    pl.append(ds.copy())
    pl.pop(0)

    pl += [ds]
    pl += pl
    pl.extend([ds, ds])

    del pl[3]
    print("list size:", len(pl))

    pl2 = Dataset([pl, pl])

    for _idx in range(len(pl)):

        def set_speaker_id(_ds: TTSDataSample):
            _ds.speaker_id = _idx

        pl.modify(_idx, set_speaker_id)

    for _item in pl.to_list():
        print("speaker_id:", _item.speaker_id)

    for ds in pl:
        ds.speaker_id += 10

    for _item in pl.to_list():
        print("to_list[speaker_id]:", _item.speaker_id)

    for _item in pl.samples:
        print("samples[speaker_id]:", _item.speaker_id)

    try:
        pl[0].speaker_id = 100
    except Exception as e:
        print(e)

    pl_slice = pl.slice({"speaker_id"})

    ds = pl.item(0)

    l_10K = list()
    with MemoryProfiler("Built-in list", format=MemoryProfiler.Format.GB) as prof:
        for _ in tqdm(range(10_000), desc="Create built-in list"):
            l_10K.append(ds.copy())

    pl_10K = Dataset()
    with MemoryProfiler("Dataset", format=MemoryProfiler.Format.GB) as prof:
        for _ in tqdm(range(10_000), desc="Create Dataset"):
            pl_10K.append(ds.copy())
