import json
import typing as tp

from .utils import check_path, tp_PATH

__all__ = ["json_load_from_file", "json_dump_to_file"]


@check_path(assert_file_exists=True)
def json_load_from_file(file_path: tp_PATH) -> tp.Dict:
    return json.loads(file_path.read_text(encoding="utf-8"))


@check_path
def json_dump_to_file(file_path: tp_PATH, data: tp.Dict):
    file_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
