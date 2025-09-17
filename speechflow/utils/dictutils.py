import typing as tp

from copy import deepcopy

__all__ = [
    "flatten_dict",
    "struct_dict",
    "get_field",
    "find_field",
    "replace_key",
    "trim_dict",
    "multi_trim_dict",
    "remove_key_or_value",
]


def flatten_dict(
    d: tp.MutableMapping,
    name: str = "dict",
    choice_key: tp.Optional[str] = None,
    sep: str = ".",
    ignore_def_key: bool = True,
    recursion_level: int = 0,
):
    output = {}
    if isinstance(d, tp.MutableMapping):
        if choice_key:
            if choice_key in d.keys():
                output.update(
                    flatten_dict(
                        d[choice_key],
                        name=f"{name}",
                        choice_key=None,
                        sep=sep,
                        recursion_level=recursion_level + 1,
                    )
                )
                return output
            elif not ignore_def_key and "default" in d.keys():
                output.update(
                    flatten_dict(
                        d["default"],
                        name=f"{name}",
                        choice_key=None,
                        sep=sep,
                        recursion_level=recursion_level + 1,
                    )
                )
                return output

        if d or recursion_level == 0:
            for key, field in d.items():
                output.update(
                    flatten_dict(
                        field,
                        name=f"{name}{sep}{key}",
                        choice_key=choice_key,
                        sep=sep,
                        recursion_level=recursion_level + 1,
                    )
                )
        else:
            output.update({name: d})

    elif d is not None:
        output.update({name: d})

    return output


def struct_dict(d: tp.MutableMapping, sep: str = ".") -> tp.MutableMapping:
    if isinstance(d, tp.MutableMapping):
        output: tp.MutableMapping = {}
        for key, field in d.items():
            names = key.split(sep)
            item = output
            for name in names[1:-1]:
                item = item.setdefault(name, {})
            item[names[-1]] = field

        return output


def get_field(d: tp.MutableMapping, path: tp.Union[str, tp.List[str]], default=None):
    if isinstance(path, str):
        path = [path]

    for name in path:
        if isinstance(d, tp.MutableMapping):
            if name in d:
                d = d[name]
            else:
                return default
        elif isinstance(d, object):
            if hasattr(d, name):
                d = getattr(d, name)
            else:
                return default
        else:
            return default

    return d


def find_field(
    d: tp.MutableMapping, key: str, def_value: tp.Any = None, all_result: bool = False
):
    if "." in key:
        sub_keys = key.split(".")
        for key in sub_keys[:-1]:
            d = find_field(d, key, all_result=True)
        if not d:
            return def_value
        else:
            list_d = [item for item in d if isinstance(item, tp.MutableMapping)]
            if not list_d:
                return def_value
            else:
                d = list_d[0]
                key = sub_keys[-1]

    out = []
    if key in d:
        out.append(d[key])
    else:
        for k, v in d.items():
            if isinstance(v, tp.MutableMapping):
                val = find_field(v, key, all_result=True)
                if val is not None:
                    out += val

    if out:
        return out if all_result else out[0]
    else:
        return def_value


def replace_key(d: tp.MutableMapping, key: str, new_key: str):
    if key in d:
        d[new_key] = d.pop(key)
    for k, v in d.items():
        if isinstance(v, tp.MutableMapping):
            replace_key(v, key, new_key)


def trim_dict(d: tp.MutableMapping, key: str) -> tp.MutableMapping:
    """Trims a dict-tree by choosing an element from branch-dict, if possible.

    Example:
    let dict_ be
        {"func": {"train": [foo, bar],
                  "valid": [bar, foo]},
         "foo": {"a": 10,
                 "b": {"train": 123, "valid": 456}}}
    then `trim_dict(dict_, "valid")` is
        {"func": [bar, foo],
         "foo": {"a": 10, "b": 456}}

    """

    def inner(elem):
        if isinstance(elem, tp.MutableMapping):
            if key in elem:
                return inner(elem[key])
            else:
                return {k: inner(v) for k, v in elem.items()}
        elif isinstance(elem, tp.MutableSequence):
            return [inner(e) for e in elem]

        return elem

    return inner(deepcopy(d))


def multi_trim_dict(d: tp.MutableMapping, keys: tp.List[str]) -> tp.MutableMapping:
    ret = d
    for key in keys:
        ret = trim_dict(ret, key)
    return ret


def remove_key_or_value(
    d: tp.MutableMapping, key: str, remove_value: bool = True
) -> tp.MutableMapping:
    d = flatten_dict(d)
    new_dict = {}
    for k, v in d.items():
        if remove_value and isinstance(v, tp.MutableSequence):
            if key in v:
                v.remove(key)

        if not k.endswith(key):
            new_dict[k] = v

    return struct_dict(new_dict)
