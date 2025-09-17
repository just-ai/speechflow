import os
import typing as tp
import logging
import importlib

from pathlib import Path

__all__ = [
    "get_root_dir",
    "get_module_dir",
    "find_files",
    "find_files_by_folders",
]


common_file_extensions = ".jpg", ".png", ".jpeg", ".bmp"

LOGGER = logging.getLogger("root")


def get_root_dir() -> Path:
    abs_path = Path(__file__).absolute()
    abs_path_parts = abs_path.parent.parts
    root_folder_idx = abs_path_parts[::-1].index("speechflow") + 1
    root_dir = abs_path.parents[root_folder_idx]
    return root_dir


def get_module_dir(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    abs_path = Path(module.__file__).absolute()
    return abs_path.parent


def find_files(
    dir_path: str,
    extensions=common_file_extensions,
    ext_lower=False,
    path_filter: tp.Optional[tp.Callable] = None,
) -> tp.List[str]:

    if ext_lower:
        file_list = [
            os.path.join(r, fn)  # type: ignore
            for r, ds, fs in os.walk(dir_path)
            for fn in fs
            if any(fn.lower().endswith(ext) for ext in extensions)
        ]
    else:
        file_list = [
            os.path.join(r, fn)  # type: ignore
            for r, ds, fs in os.walk(dir_path)
            for fn in fs
            if any(fn.endswith(ext) for ext in extensions)
        ]

    if path_filter is not None:
        file_list = [path for path in file_list if path_filter(path)]

    return file_list


def find_files_by_folders(
    dir_path: str,
    extensions=common_file_extensions,
    ext_lower=False,
    path_filter: tp.Optional[tp.Callable] = None,
) -> tp.List[tp.List[str]]:

    file_list = []
    for r, ds, fs in os.walk(dir_path):
        if ext_lower:
            files = [os.path.join(r, fn) for fn in fs if fn.lower().endswith(extensions)]
        else:
            files = [os.path.join(r, fn) for fn in fs if fn.endswith(extensions)]

        if len(files) > 0:
            if path_filter is not None:
                files = [path for path in files if path_filter(path)]

            file_list.append(files)

    return file_list
