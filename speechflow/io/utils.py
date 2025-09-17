import typing as tp
import inspect

from functools import wraps
from pathlib import Path

tp_PATH = tp.Union[str, Path]
tp_PATH_LIST = tp.MutableSequence[tp_PATH]


def check_path(f_py=None, assert_file_exists: bool = False, make_dir: bool = False):
    assert callable(f_py) or f_py is None

    def decorated_check_path(func):
        @wraps(func)
        def decorated_func(*args, **kwargs):
            def check(path: tp_PATH) -> Path:
                if isinstance(path, str):
                    path = Path(path)
                if isinstance(path, Path):
                    if assert_file_exists and not path.exists():
                        raise FileNotFoundError(
                            f"File or directory '{path.as_posix()}' does not exist!"
                        )
                return path

            func_sig = inspect.signature(func)
            func_parameters = dict(func_sig.parameters)

            params = {k: v for k, v in zip(func_parameters, args)}
            params.update(kwargs)

            for k, v in func_parameters.items():
                if k not in params:
                    continue

                if v.annotation in [tp_PATH, tp.Optional[tp_PATH]]:
                    params[k] = check(params[k])
                    if make_dir:
                        params[k].parent.mkdir(parents=True, exist_ok=True)

                if v.annotation in [tp_PATH_LIST, tp.Optional[tp_PATH_LIST]]:
                    val = params[k]
                    if isinstance(val, tp.MutableSequence):
                        val = [check(item) for item in val]
                        params[k] = val

                if v.annotation in [
                    tp.Union[tp_PATH, tp_PATH_LIST],
                    tp.Optional[tp.Union[tp_PATH, tp_PATH_LIST]],
                ]:
                    val = params[k]
                    if val is not None and not isinstance(val, tp.MutableSequence):
                        params[k] = check(val)
                    if isinstance(val, tp.MutableSequence):
                        val = [check(item) for item in val]
                        params[k] = val

            return func(**params)

        return decorated_func

    return decorated_check_path(f_py) if callable(f_py) else decorated_check_path


if __name__ == "__main__":

    @check_path(assert_file_exists=True)
    def load_file(file_path: tp.Optional[tp_PATH] = None):
        return file_path.read_bytes()

    load_file(file_path="src/file.yml")
