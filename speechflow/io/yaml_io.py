import typing as tp

from pathlib import Path

import yaml

from .utils import check_path

__all__ = ["yaml_load", "yaml_dump", "yaml_load_from_file", "yaml_dump_to_file"]


# define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return "/".join([str(i) for i in seq])


# register the tag handler
def get_yaml_loader():
    loader = yaml.SafeLoader
    yaml.add_constructor("!join", join, Loader=loader)
    return loader


def yaml_load(cfg: str) -> tp.Dict:
    return yaml.load(cfg, Loader=get_yaml_loader())


def yaml_dump(data: tp.Dict) -> str:
    return yaml.safe_dump(data)


@check_path(assert_file_exists=True)
def yaml_load_from_file(file_path: Path) -> tp.Dict:
    return yaml.load(file_path.read_text(encoding="utf-8"), Loader=get_yaml_loader())


@check_path(assert_file_exists=True)
def yaml_dump_to_file(file_path: Path, data: tp.Dict):
    file_path.write_text(yaml.safe_dump(data), encoding="utf-8")


if __name__ == "__main__":

    _cfg = """
        user_dir: &DIR /home/user
        user_pics: !join [*DIR, pics]
    """
    print(yaml_load(_cfg))
