import sys
import typing as tp
import hashlib

from copy import deepcopy as copy
from os import environ as env
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from speechflow.io.utils import check_path, tp_PATH
from speechflow.io.yaml_io import yaml_dump, yaml_load
from speechflow.utils.dictutils import (
    find_field,
    flatten_dict,
    multi_trim_dict,
    trim_dict,
)

__all__ = ["Config", "change_config_file"]


class Config(DictConfig):
    def __init__(self, content: tp.Union[tp.Dict[str, tp.Any], "Config", tp.Any]):
        super().__init__(content)

    @staticmethod
    def empty(sections: tp.Optional[tp.Set[str]] = None) -> "Config":
        cfg = Config({})
        if sections:
            cfg.create_section(sections)
        return cfg

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def hash(self) -> str:
        flat = self.flatten().to_dict()
        flat = {k: v for k, v in flat.items() if "device" not in k}
        return hashlib.md5(yaml_dump(flat).encode("utf-8")).hexdigest()[:8]

    @property
    def raw_file(self) -> str:
        return self._metadata.resolver_cache.get("raw_file", "")

    @property
    def raw_file_path(self) -> Path:
        return self._metadata.resolver_cache.get("raw_file_path", Path())

    def get(self, key, default_value: tp.Any = None, mutable: bool = False) -> tp.Any:
        if mutable:
            return super().get(key, default_value)
        else:
            value = super().get(key, default_value)
            return Config(value) if isinstance(value, tp.MutableMapping) else value

    def section(self, key: str, mutable: bool = False) -> "Config":
        section = self.get(key, {}, mutable=mutable)
        if not isinstance(section, tp.MutableMapping):
            raise ValueError(f"Section {section} is not dictionary!")
        return section if mutable else Config(section)

    def create_section(self, keys: tp.Set[str]):
        for key in keys:
            self.setdefault(key, {})

    def trim(self, key: str) -> "Config":
        as_dict = self.to_dict()
        as_dict = trim_dict(as_dict, key=key)
        return Config(as_dict)

    def multi_trim(self, keys: tp.List[str]) -> "Config":
        as_dict = self.to_dict()
        as_dict = multi_trim_dict(as_dict, keys=keys)
        return Config(as_dict)

    def flatten(self, sep: str = ".") -> "Config":
        as_dict = self.to_dict()
        as_dict = flatten_dict(as_dict, name="cfg", sep=sep)
        return Config(as_dict)

    def find_field(
        self, key: str, default_value: tp.Any = None, all_result: bool = False
    ) -> tp.Optional[tp.Any]:
        return find_field(self, key, default_value, all_result)

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        return OmegaConf.to_container(self)

    def copy(self) -> "Config":
        return Config(copy(self))

    @staticmethod
    @check_path(assert_file_exists=True)
    def create_from_yaml(
        yaml_cfg: str,
        section: tp.Optional[str] = None,
        value_select: tp.Optional[tp.Set] = None,
    ) -> "Config":
        if sys.platform == "win32":
            yaml_cfg = yaml_cfg.replace("PosixPath", "WindowsPath")

        cfg = Config(yaml_load(yaml_cfg))

        if section:
            cfg = cfg.section(section)

        cfg = cfg.multi_trim(
            (value_select or cfg.get("value_select") or []) + ["default"]
        )

        cfg._metadata.resolver_cache["raw_file"] = yaml_cfg
        if value_select:
            cfg._metadata.resolver_cache["raw_file"] += (
                f"\n\nvalue_select: {value_select}" if value_select else ""
            )

        return cfg

    @staticmethod
    @check_path(assert_file_exists=True)
    def create_from_file(
        file_path: tp_PATH,
        section: tp.Optional[str] = None,
        value_select: tp.Optional[tp.Set] = None,
    ) -> "Config":
        if value_select and "debug" in value_select:
            env["VERBOSE"] = "True"

        if file_path.suffix in [".yaml", ".yml"]:
            yaml_cfg = file_path.read_text(encoding="utf-8")
            cfg = Config.create_from_yaml(yaml_cfg, section, value_select)
            cfg._metadata.resolver_cache["raw_file_path"] = file_path
            return cfg
        else:
            raise ValueError(f"Config format {file_path.suffix} is not support!")

    def to_yaml(self) -> str:
        return yaml_dump(self.to_dict())

    @check_path(make_dir=True)
    def to_file(self, file_path: tp_PATH):
        file_path.write_text(self.to_yaml(), encoding="utf-8")


@check_path(assert_file_exists=True)
def change_config_file(
    config_path: tp_PATH,
    replace_map: tp.Dict[str, tp.Optional[bool | int | float | str | Path]],
):
    config = config_path.read_text(encoding="utf-8")

    lines = config.split("\n")
    for key, value in replace_map.items():
        if value is None:
            continue
        if isinstance(value, Path):
            value = value.as_posix()
        else:
            value = str(value)

        for i, line in enumerate(lines):
            if f"{key}:" in line:
                parts = line.rstrip(" ").split(" ")
                parts = [item for item in parts if key in item or "&" in item or not item]
                parts += [value]
                lines[i] = " ".join(parts)
                break

    config_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    _cfg = Config({"dirs": {"file_path": "/path/to/file"}})
    print(_cfg.section("dirs"))
    print(_cfg.section("dirs")["file_path"])
    print(_cfg.section("dirs").file_path)
    print(_cfg.to_dict())
    print(_cfg.flatten())
    print(Config.create_from_file("/path/to/file"))
