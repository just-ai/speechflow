import sys
import json
import codecs
import pickle
import hashlib
import logging

from enum import Enum
from pathlib import Path
from typing import Any, List

import numpy as np

from speechflow.logging import trace

__all__ = ["Serialize", "JsonEncoder"]

LOGGER = logging.getLogger("root")


class Serialize:
    class Format(Enum):
        B = 1
        KB = 1024
        MB = 1024**2
        GB = 1024**3
        TB = 1024**4

    @staticmethod
    def dump(obj: Any, use_hash: bool = False) -> bytes:
        dump = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if use_hash:
            dump = hashlib.md5(dump).hexdigest().encode() + dump
        return dump

    @staticmethod
    def load(dump: bytes, use_hash: bool = False) -> Any:
        try:
            assert type(dump) == bytes, "Invalid object dump!"
            if use_hash:
                dump = memoryview(dump)
                hash, dump = dump[:32], dump[32:]
                curr_hash = hashlib.md5(dump).hexdigest().encode()
                assert hash == curr_hash, "Invalid object hash!"
            return pickle.loads(dump)
        except Exception as e:
            LOGGER.error(trace("SerializeUtils", e))

    @staticmethod
    def dumps(list_obj: List[Any], inplace: bool = False):
        assert isinstance(list_obj, List), "list_obj must be of type list!"
        # assert len(list_obj) > 0, "list_obj is empty!"

        if any([type(obj) == bytes for obj in list_obj]):
            return list_obj
        else:
            if inplace:
                for index in range(len(list_obj)):
                    list_obj[index] = Serialize.dump(list_obj[index])
                return list_obj
            else:
                return [Serialize.dump(obj) for obj in list_obj]

    @staticmethod
    def loads(list_dump: List[bytes], inplace: bool = False):
        assert isinstance(list_dump, List), "list_dump must be of type list!"
        assert len(list_dump) > 0, "list_dump is empty!"

        if inplace:
            for index in range(len(list_dump)):
                list_dump[index] = Serialize.load(list_dump[index])
            return list_dump
        else:
            return [Serialize.load(dump) for dump in list_dump]

    @staticmethod
    def get_obj_size(obj: Any, format: Format = Format.B):
        if type(obj) != bytes:
            obj = Serialize.dump(obj)
        return sys.getsizeof(obj) / format.value


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        from multilingual_text_parser.data_types import (
            Doc,
            Position,
            Sentence,
            Syntagma,
            Token,
        )

        if isinstance(obj, np.ndarray):
            return codecs.encode(pickle.dumps(obj), "base64").decode()
        if isinstance(obj, Path):
            return obj.as_posix()
        if isinstance(obj, Doc):
            return obj.sents
        if isinstance(obj, Sentence):
            return {
                "syntagmas": obj.syntagmas,
                "ssml_modidfiers": obj.ssml_modidfiers,
                "ssml_insertions": obj.ssml_insertions,
            }
        if isinstance(obj, (Syntagma, Token)):
            return obj.__dict__
        if isinstance(obj, Position):
            return obj.value
        return json.JSONEncoder.default(self, obj)
