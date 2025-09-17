import typing as tp

from dataclasses import is_dataclass
from functools import partial, wraps

from speechflow.data_pipeline.core.dataset import Dataset

__all__ = ["PipeRegistry"]


class PipeRegistry:
    @staticmethod
    def check(
        pipe: tp.Sequence[tp.Callable], input_fields: tp.Optional[tp.Set] = None
    ) -> bool:
        """Checking the sequence functions of pipe.

        :param pipe: list of registered functions
        :param input_fields:
        :return:

        """
        assert pipe, "pipe is empty!"

        if input_fields:
            input_fields = PipeRegistry._parse(input_fields)
        else:
            input_fields = set()

        for fn in pipe:
            while isinstance(fn, partial):
                fn = fn.func
            assert hasattr(fn, "_io"), f"{fn} not registered!"

        for fn in pipe:
            while isinstance(fn, partial):
                fn = fn.func
            _io: tp.Dict[str, tp.Set] = getattr(fn, "_io", dict())
            _name: str = getattr(fn, "_name")

            if not input_fields:
                input_fields = _io["inputs"].copy()
            assert _io["inputs"].issubset(
                input_fields
            ), f"[{_name}<-{_io['inputs']}]: missing required fields"
            input_fields.update(_io["outputs"])

        return True

    @staticmethod
    def filter(
        pipe: tp.Sequence[tp.Callable],
        selection_fn: tp.Callable,
        by_field: bool = True,
        by_handler_name: bool = False,
    ) -> tp.List[tp.Callable]:
        """Filtering of pipe functions in accordance with the condition.

        :param pipe: preprocessing functions
        :param selection_fn: defining a condition based on input fields
            for selection of preprocessing functions
        :param by_field: filtered pipe by io fields
        :param by_handler_name: filtered pipe by handler names
        :return: corresponding preprocessing functions

        """
        assert pipe, "pipe is empty!"

        new_pipe = []
        secondary_ignored_fields: tp.Set[str] = set()
        for fn in pipe:
            _fn = fn
            while isinstance(_fn, partial):
                _fn = _fn.func

            _io: tp.Dict[str, tp.Set] = getattr(_fn, "_io", dict())
            _name: str = getattr(_fn, "_name")
            _classname: str = getattr(_fn, "_classname")

            if by_handler_name:
                if selection_fn({_name, _classname}):
                    new_pipe.append(fn)
                    continue

            if by_field:
                if (
                    selection_fn(_io["inputs"])
                    and selection_fn(_io["outputs"])
                    and not (secondary_ignored_fields & _io["inputs"])
                ):
                    new_pipe.append(fn)
                    continue

            secondary_ignored_fields.update(_io["outputs"] - _io["inputs"])

        return new_pipe

    @staticmethod
    def _parse(fields: tp.Union[tp.Set[str], tp.FrozenSet[str]]) -> tp.Set[str]:
        """Parse field name format.

        Example:
            {"segmentations|Y1,R2"} -> {"segmentations|Y1", "segmentations|R2"}

        """
        prepare_fields = set()
        for name in fields:
            subnames = name.split(",")
            if len(subnames) > 1:
                for i in range(1, len(subnames)):
                    full_path_to_top_level = subnames[0].rsplit("|", 1)[0]
                    subnames[i] = full_path_to_top_level + "|" + subnames[i]
            prepare_fields.update(subnames)

            root = subnames[0].split("|")[:-1]
            prepare_fields.update(root)

        return prepare_fields

    @staticmethod
    def registry(
        func: tp.Optional[tp.Callable] = None,
        inputs: tp.Union[tp.Set[str], tp.FrozenSet[str]] = frozenset(),
        outputs: tp.Union[tp.Set[str], tp.FrozenSet[str]] = frozenset(),
        optional: tp.Union[tp.Set[str], tp.FrozenSet[str]] = frozenset(),
    ) -> tp.Callable:
        """Registry pipe functions.

        :param func: function that takes dictionary or class as input data
            and returns the same
        :param inputs: required fields in input data struct
        :param outputs: produced fields in input data struct
        :param optional: optional fields in input data struct

        """

        if not func:
            return partial(
                PipeRegistry.registry, inputs=inputs, outputs=outputs, optional=optional
            )

        assert (
            isinstance(inputs, (set, frozenset))
            and isinstance(outputs, (set, frozenset))
            and isinstance(optional, (set, frozenset))
        ), f"[{func.__name__}]: argument must be of type of set"

        io_fields = {
            "inputs": PipeRegistry._parse(inputs),
            "outputs": PipeRegistry._parse(outputs),
            "optional": PipeRegistry._parse(optional),
        }

        new_doc = "\n".join(
            [
                func.__doc__ if func.__doc__ else "",
                f"\trequired fields: {', '.join(io_fields['inputs'])}",
                f"\tproduced fields: {', '.join(io_fields['outputs'])}",
                f"\toptional fields: {', '.join(io_fields['optional'])}",
            ]
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Checking for input and output fields."""
            for var in args + tuple(kwargs.values()):
                if isinstance(var, (list, dict, Dataset)) or is_dataclass(var):
                    break
            else:
                raise ValueError(f"no matching argument for {str(func)}!")

            # for fields in io_fields["inputs"]:
            #    assert get_field(var, field.split("|")), \
            #        f"[{func.__name__}]:{field}:{var}: "\
            #        "missing required fields"

            ret = func(*args, **kwargs)

            # var = ret[0] if isinstance(ret, list) and ret else ret
            # if isinstance(var, dict) or is_dataclass(var):
            #    for field in io_fields["outputs"]:
            #        assert get_field(var, field.split("|")) is not None, \
            #            f"[{func.__name__}:{field}:{var}]: "\
            #            "some output fields are missing"

            return ret

        setattr(wrapper, "_name", func.__name__)
        setattr(wrapper, "_classname", func.__qualname__.split(".")[0])
        setattr(wrapper, "_io", io_fields)
        wrapper.__doc__ = new_doc
        return wrapper
