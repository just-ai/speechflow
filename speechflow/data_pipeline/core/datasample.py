import sys
import uuid
import pickle
import typing as tp
import dataclasses

from copy import deepcopy as copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import numpy.typing as npt

from torch import Tensor

from speechflow.io.utils import tp_PATH
from speechflow.utils.dictutils import flatten_dict, struct_dict

__all__ = [
    "ToDict",
    "ToTensor",
    "ToNumpy",
    "MovableToDevice",
    "Pinnable",
    "Detachable",
    "DataSample",
    "TrainData",
    "tp_DATA",
]

tp_DATA = tp.Union[int, float, str, npt.NDArray, Tensor]


@dataclass
class ToDict:
    def keys(self) -> tp.List[str]:
        return [k for k in self.to_dict().keys() if not k.startswith("_")]

    def to_dict(self) -> tp.Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class ToTensor:
    @staticmethod
    def _from_numpy(data):
        tensor = torch.as_tensor(data)
        if isinstance(tensor, torch.DoubleTensor):
            tensor = tensor.float()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return tensor

    def to_tensor(self):
        temp = flatten_dict(self.__dict__)
        for name, field in temp.items():
            if isinstance(field, np.ndarray):
                temp[name] = self._from_numpy(field)
            elif isinstance(field, ToTensor):
                temp[name] = field.to_tensor()
            elif isinstance(field, (float, np.double)):
                temp[name] = np.float32(field)

        self.__dict__ = struct_dict(temp)
        return self


@dataclass
class ToNumpy:
    @staticmethod
    def _from_tensor(tensor):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return tensor.cpu().numpy()

    def to_numpy(self):
        temp = flatten_dict(self.__dict__)
        for name, field in temp.items():
            if isinstance(field, torch.Tensor):
                temp[name] = self._from_tensor(field)
            elif isinstance(field, ToNumpy):
                temp[name] = field.to_numpy()

        self.__dict__ = struct_dict(temp)
        return self


@dataclass
class MovableToDevice:
    device: torch.device = torch.device("cpu")

    @staticmethod
    def tensors_to_device(data, device, non_blocking: tp.Optional[bool] = None):
        temp = flatten_dict(data.__dict__)

        if non_blocking is None:
            non_blocking = True if "cuda" in device.type else False

        for name, field in temp.items():
            if isinstance(field, (Tensor, MovableToDevice)):
                temp[name] = field.to(device, non_blocking=non_blocking)
            elif dataclasses.is_dataclass(field):
                MovableToDevice.tensors_to_device(field, device)

        data.__dict__ = struct_dict(temp)

    def to(self, device: torch.device, non_blocking: tp.Optional[bool] = None):
        self.tensors_to_device(self, device, non_blocking=non_blocking)
        self.device = device
        return self

    def cpu(self):
        return self.to(torch.device("cpu"))

    def cuda(self):
        if torch.cuda.is_available():
            return self.to(torch.device(f"cuda:{torch.cuda.current_device()}"))
        else:
            return self.cpu()


@dataclass
class Pinnable:
    def pin_memory(self):
        temp = flatten_dict(self.__dict__)
        for name, field in temp.items():
            if isinstance(field, (Tensor, Pinnable)):
                if field.device.type == "cpu":
                    temp[name] = field.pin_memory()

        self.__dict__ = struct_dict(temp)
        return self


@dataclass
class Detachable:
    def detach(self):
        temp = flatten_dict(self.__dict__)
        for name, field in temp.items():
            if isinstance(field, Tensor):
                temp[name] = field.detach()

        self.__dict__ = struct_dict(temp)
        return self


@dataclass
class Serialize:
    _updated_fields = None

    @staticmethod
    def _pickle(obj: tp.Any):
        try:
            if obj.__class__.__name__ != "bytes":
                obj = pickle.dumps(obj)
        finally:
            return obj

    @staticmethod
    def _unpickle(obj: tp.Any):
        try:
            if obj.__class__.__name__ == "bytes":
                obj = pickle.loads(obj)
        finally:
            return obj

    @staticmethod
    def _is_builtin_class_instance(obj):
        # return obj.__class__.__module__ == "builtins"
        return obj is None or obj.__class__.__name__ in (
            "bool",
            "int",
            "float",
            "tuple",
            "str",
            "bytes",
        )

    def __getattribute__(self, attr):
        value = object.__getattribute__(self, attr)
        is_bytes = value.__class__.__name__ == "bytes"
        _updated_fields = object.__getattribute__(self, "_updated_fields")

        if is_bytes and _updated_fields is not None:
            _unpickle = object.__getattribute__(self, "_unpickle")
            value = _unpickle(value)
            object.__setattr__(self, attr, value)
            _updated_fields.add(attr)

        return value

    def __setattr__(self, attr, value):
        _updated_fields = object.__getattribute__(self, "_updated_fields")
        _is_builtin_class_instance = object.__getattribute__(
            self, "_is_builtin_class_instance"
        )

        if (
            _updated_fields is not None
            and not attr.startswith("_")
            and not _is_builtin_class_instance(value)
        ):
            _pickle = object.__getattribute__(self, "_pickle")
            value = _pickle(value)

        object.__setattr__(self, attr, value)

    def serialize(self, full: bool = False, **kwargs):
        _updated_fields = object.__getattribute__(self, "_updated_fields")
        _is_builtin_class_instance = object.__getattribute__(
            self, "_is_builtin_class_instance"
        )
        _pickle = object.__getattribute__(self, "_pickle")

        if not full and _updated_fields is not None:
            for field in _updated_fields:
                value = object.__getattribute__(self, field)
                object.__setattr__(self, field, _pickle(value))
        else:
            for field, value in self.__dict__.items():
                if full or not _is_builtin_class_instance(value):
                    object.__setattr__(self, field, _pickle(value))

        object.__setattr__(self, "_updated_fields", set())
        return self

    def deserialize(self, full: bool = False, **kwargs):
        if full:
            _unpickle = object.__getattribute__(self, "_unpickle")
            for key, value in self.__dict__.items():
                object.__setattr__(self, key, _unpickle(value))

            object.__setattr__(self, "_updated_fields", None)

        return self


@dataclass
class DataSample(ToDict, ToTensor, ToNumpy, Serialize):
    file_path: tp_PATH = None
    label: tp.Union[str, int] = ""
    tag: tp.Optional[str] = None
    index: tp.Optional[tp.Tuple[tp.Any, ...]] = None
    transform_params: tp.Optional[tp.Dict[str, tp.Any]] = None
    additional_fields: tp.Optional[tp.Dict[str, tp.Any]] = None

    __uid = str = uuid.uuid4().hex
    __all_keys = set()  # type: ignore

    def __post_init__(self):
        if self.file_path is None:
            self.file_path = Path()
        elif isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        if self.transform_params is None:
            self.transform_params = {}
        if self.additional_fields is None:
            self.additional_fields = {}

    def __init_subclass__(cls):
        if cls.__all_keys == set():
            cls.__all_keys.update({"file_path", "label", "index"})
        cls.__all_keys.update(set(cls.__dict__.keys()))

    def __len__(self) -> int:
        return sys.getsizeof(self)

    def __str__(self) -> str:
        return self.file_path.as_posix() if self.file_path else self.label

    def __hash__(self) -> int:
        return hash(self.__uid)

    def __eq__(self, other):
        return self.__uid == other.__uid

    @property
    def uid(self):
        return self.__uid

    @staticmethod
    def all_keys() -> tp.Set:
        return DataSample.__all_keys

    def setdefault(self, name: str, value: tp.Optional[tp.Any] = None):
        if name not in self.all_keys() or getattr(self, name) is None:
            setattr(self, name, value)
        else:
            value = getattr(self, name)
        return value

    def update(self, data: tp.Union[tp.Dict, "DataSample"]):
        if isinstance(data, DataSample):
            data = data.to_dict()

        for key, field in data.items():
            if field is not None:
                if isinstance(getattr(self, key, None), dict) and isinstance(field, dict):
                    getattr(self, key, {}).update(field)
                else:
                    setattr(self, key, field)

    def get_param_val(self, name: str, def_val=None) -> tp.Any:
        flatten_params = flatten_dict(self.transform_params)
        found = []
        for key, field in flatten_params.items():
            if key.split(".", 1)[-1].startswith(name):
                found.append(field)
        if not found:
            for key, field in flatten_params.items():
                if key.endswith(name):
                    found.append(field)
        if found:
            return found[-1]
        else:
            return def_val

    def copy(self):
        new = copy(self)
        new.__uid = uuid.uuid4().hex
        return new


@dataclass
class TrainData(ToDict, MovableToDevice, Detachable):
    batch_tag: tp.Optional[str] = None
    batch_idx: tp.Union[int] = 0
    global_step: tp.Union[int] = 0


if __name__ == "__main__":

    def check(flag: bool):
        if flag:
            print("OK!")
        else:
            print("FAIL!")

    @dataclass(eq=False)
    class ChildDataSample(DataSample):
        pass

    ds1 = ChildDataSample()
    ds2 = ChildDataSample()

    check(ds1 == ds1)
    check(ds1 == ds2)
    check(hash(ds1) == hash(ds1))
    check(hash(ds1) == hash(ds2))
