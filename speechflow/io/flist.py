import io
import random
import typing as tp
import logging

from pathlib import Path

from speechflow.io import check_path, tp_PATH
from speechflow.logging import trace
from speechflow.utils.fs import find_files

__all__ = [
    "split_file_list",
    "construct_file_list",
    "generate_file_list",
    "read_file_list",
]

LOGGER = logging.getLogger("root")


def split_file_list(
    flist: tp.List[str], ratio: float = 0.8
) -> tp.Tuple[tp.List[str], tp.List[str]]:
    n = int(len(flist) * ratio)
    return flist[:n], flist[n:]


@check_path(assert_file_exists=True)
def construct_file_list(
    data_root: tp_PATH,
    ext: tp.Union[str, tp.Tuple[str, ...]] = ".*",
    with_subfolders: bool = False,
    path_filter: tp.Optional[tp.Callable] = None,
) -> tp.List[str]:
    flist = find_files(
        data_root.as_posix(), extensions=(ext,) if isinstance(ext, str) else ext
    )
    LOGGER.info(
        trace("construct_file_list", f"find {len(flist)} {ext} files in {data_root}")
    )

    if not with_subfolders:
        flist = [p for p in flist if Path(p).parents[0] == data_root]

    if path_filter:
        flist = [file for file in flist if path_filter(file)]

    return flist


@check_path(assert_file_exists=True)
def generate_file_list(
    data_root: tp_PATH,
    ext: tp.Union[str, tp.Tuple[str, ...]] = ".*",
    with_subfolders: bool = False,
    path_filter: tp.Optional[tp.Callable] = None,
    separator: str = "|",
) -> io.StringIO:
    output_file = io.StringIO(newline="\n")
    file_list = construct_file_list(data_root, ext, with_subfolders, path_filter)
    lines = [f"{line}{separator}0" for line in file_list]
    output_file.write("\n".join(lines))
    return output_file


def read_file_list(
    flist_path: tp.Union[tp_PATH, io.StringIO],
    subsets: tp.Optional[tp.List[str]] = None,
    directory_filter: tp.Optional[
        tp.Union[tp.List[str], tp.Dict[str, tp.List[str]]]
    ] = None,
    max_num_samples: tp.Optional[int] = None,
    split_type: str = "auto",  # auto, manual
    split_ratio: tp.Optional[
        tp.Union[float, tp.List[float], tp.Dict[str, tp.List[float]]]
    ] = None,
    use_shuffle: bool = True,
    use_labels: bool = True,
    separator: str = "|",
) -> tp.Union[tp.List[str], tp.Dict[str, tp.List[str]]]:
    if subsets is None:
        LOGGER.warning(trace("read_file_list", message="List of subsets is not set!"))

    def read_file(file_path: tp.Union[tp_PATH, io.StringIO]):
        if isinstance(file_path, io.StringIO):
            samples = file_path.getvalue().splitlines()
        else:
            samples = Path(file_path).read_text(encoding="utf-8").splitlines()
        samples = [item.strip() for item in samples]
        if not use_labels:
            samples = [item.split(separator, 1)[0] for item in samples]
        return samples

    def shuffle_and_trim(samples: tp.List[str]):
        if max_num_samples:
            samples = samples[:max_num_samples]
        if use_shuffle:
            random.shuffle(samples)
        return samples

    flist = read_file(flist_path)

    if directory_filter is not None:
        if isinstance(directory_filter, tp.MutableSequence):
            flist = [
                item for item in flist if any(path in item for path in directory_filter)
            ]
        elif isinstance(directory_filter, tp.MutableMapping):
            if directory_filter.get("include"):
                include_paths = directory_filter.get("include", [])
                if isinstance(include_paths, str):
                    include_paths = [include_paths]
                flist = [
                    item for item in flist if any(path in item for path in include_paths)
                ]
            if directory_filter.get("exclude"):
                exclude_paths = directory_filter.get("exclude", [])
                if isinstance(exclude_paths, str):
                    exclude_paths = [exclude_paths]
                flist = [
                    item
                    for item in flist
                    if all(path not in item for path in exclude_paths)
                ]
        else:
            raise ValueError("not supported format for directory filter")

    flist_subset: tp.Dict[str, tp.List[str]] = {}
    if split_type == "manual":
        assert subsets is not None

        sublist = []
        for file in flist:
            if file.startswith("["):
                sublist = flist_subset.setdefault(file, [])
            else:
                sublist.append(file)

        for name in subsets:
            tag = f"[{name.upper()}]"
            if tag not in flist_subset:
                raise ValueError(f"Tag {tag} not found in subsets!")
            flist_subset[name] = shuffle_and_trim(flist_subset.pop(tag))

    elif split_type == "auto" and subsets:
        if split_ratio is None:
            assert len(subsets) == 2
            split_ratio = {subsets[0]: [0, 0.8], subsets[1]: [0.8, 1]}

        elif isinstance(split_ratio, (float, tp.MutableSequence)):
            if isinstance(split_ratio, float):
                split_ratio = [split_ratio]

            assert sum(split_ratio) < 1, "Invalid split ratios!"
            rations = [0.0] + split_ratio + [1.0]
            split_ratio = {}
            for name, ratio in zip(subsets, zip(rations[:-1], rations[1:])):
                split_ratio[name] = list(ratio)

        else:
            if set(subsets) != set(split_ratio.keys()):
                raise ValueError("Split coefficients are not specified for all subsets!")

        flist = shuffle_and_trim(flist)
        size = len(flist)
        for name in subsets:
            l, r = split_ratio[name]  # split boundaries
            a, b = int(l * size), int(r * size)
            a, b = min(a, size - 1), min(b, size)
            flist_subset[name] = flist[a:b]

    else:
        return shuffle_and_trim(flist)

    if set(subsets) != set(flist_subset.keys()):
        raise ValueError(f"Invalid split dataset over {subsets}!")

    return flist_subset
