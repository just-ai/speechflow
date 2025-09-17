import re
import sys
import pickle
import typing as tp
import logging
import pathlib
import subprocess

from contextlib import contextmanager
from pathlib import Path

import git
import torch

from pytorch_lightning.callbacks import ModelCheckpoint

import speechflow

from speechflow.io import Config, check_path, tp_PATH
from speechflow.logging import trace
from speechflow.utils.fs import find_files, get_root_dir
from speechflow.utils.init import init_class_from_config

__all__ = ["ExperimentSaver"]

LOGGER = logging.getLogger("root")


class ExperimentSaver:
    _folder = "_checkpoints"

    @check_path(make_dir=True)
    def __init__(
        self,
        expr_path: tp_PATH,
        additional_files: tp.Optional[tp.Dict[str, str]] = None,
    ):
        self.expr_path = expr_path
        self.expr_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            trace(self, message=f"Experiment folder: {self.expr_path.as_posix()}")
        )

        self.script_files = self._code_dump()
        (self.expr_path / "scripts.pkl").write_bytes(pickle.dumps(self.script_files))

        if additional_files:
            for cfg_name, text in additional_files.items():
                file_path = self.expr_path / cfg_name
                if isinstance(text, tp.MutableSequence):
                    for idx, txt in enumerate(text):
                        new_path = file_path.with_name(
                            f"{file_path.stem}_{idx}{file_path.suffix}"
                        )
                        new_path.write_text(txt, encoding="utf-8")
                else:
                    file_path.write_text(text, encoding="utf-8")

        try:
            repo = git.Repo(search_parent_directories=True)
            commit_hash = repo.head.object.hexsha
        except Exception as e:
            LOGGER.warning(trace(self, e, message="git repo not found!", full=False))
            commit_hash = None

        self.to_save = {
            "files": additional_files,
            "scripts": self.script_files,
            "commit_hash": commit_hash,
            "versions": {"speechflow": speechflow.__version__, "libs": {}},
        }

        try:
            import multilingual_text_parser

            self.to_save["versions"]["libs"][
                "text_parser"
            ] = multilingual_text_parser.__version__
        except ImportError:
            pass

    @property
    def checkpoints_dir(self) -> Path:
        return self.expr_path / ExperimentSaver._folder

    @staticmethod
    def _code_dump(
        extensions=(
            ".py",
            ".yml",
            ".md",
        )
    ):
        root_dir = get_root_dir()
        files = find_files(
            root_dir.as_posix(),
            extensions=extensions,
            path_filter=lambda x: all(s not in x for s in ["conda", "python", "env"]),
        )
        script_files = {}
        for file in files:
            script_files[file.replace(str(root_dir), "")] = Path(file).read_bytes()

        return script_files

    @staticmethod
    @contextmanager
    def portable_pathlib():
        posix_backup = pathlib.PosixPath
        windows_backup = pathlib.WindowsPath
        try:
            if sys.platform == "win32":
                pathlib.PosixPath = pathlib.WindowsPath
            else:
                pathlib.WindowsPath = pathlib.PosixPath
            yield
        finally:
            pathlib.PosixPath = posix_backup
            pathlib.WindowsPath = windows_backup

    @staticmethod
    @check_path(assert_file_exists=True)
    def get_last_checkpoint(
        expr_path: tp_PATH, sort_by_epoch: bool = False
    ) -> tp.Optional[Path]:
        def extract_step(txt):
            """Regular expression extracting number of steps from checkpoint filename."""
            return int(re.split(r"(epoch=)(\d+)(-step)", str(txt))[2])

        if expr_path.is_file():
            return expr_path

        files = list((expr_path / ExperimentSaver._folder).glob("*.ckpt"))
        if files:
            if sort_by_epoch:
                files.sort(key=extract_step)
                ckpt_path = files[-1]
            else:
                ckpt_path = max(files, key=lambda f: f.stat().st_mtime)

            return ckpt_path

        return None

    def get_checkpoint_callback(self, cfg: Config, prefix: tp.Optional[str] = None):
        filename = cfg.get("filename", "{epoch}-{step}")

        if prefix is not None:
            cfg.filename = f"{prefix}_{filename}"

        checkpoint_callback = init_class_from_config(ModelCheckpoint, cfg)(
            dirpath=self.expr_path / ExperimentSaver._folder
        )
        return checkpoint_callback

    @staticmethod
    @check_path(assert_file_exists=True)
    def load_checkpoint(file_path: tp_PATH, map_location: str = "cpu") -> tp.Dict:
        if file_path.is_dir():
            ckpt_path = ExperimentSaver.get_last_checkpoint(file_path)
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"Not found checkpoint in directory {file_path.as_posix()}"
                )
            else:
                file_path = ckpt_path

        LOGGER.info(f"Load checkpoint from {file_path.as_posix()}")
        if file_path.suffix in [".ckpt", ".pt"]:
            with ExperimentSaver.portable_pathlib():
                return torch.load(file_path, map_location, weights_only=False)
        elif file_path.suffix in [".pkl"]:
            with ExperimentSaver.portable_pathlib():
                return pickle.loads(file_path.read_bytes())
        else:
            raise NotImplementedError(f"Unknown checkpoint extension: {file_path.name}")

    @staticmethod
    @check_path(make_dir=True)
    def save_checkpoint(checkpoint, file_path: tp_PATH):
        return torch.save(checkpoint, file_path)

    @staticmethod
    def load_configs_from_checkpoint(
        checkpoint: tp.Union[tp_PATH, tp.Mapping]
    ) -> tp.Tuple[Config, Config]:
        if not isinstance(checkpoint, tp.Mapping):
            checkpoint = ExperimentSaver.load_checkpoint(checkpoint)

        cfg_data_yml = checkpoint["files"]["data.yml"]
        cfg_model_yml = checkpoint["files"]["model.yml"]

        if isinstance(cfg_data_yml, tp.MutableSequence):
            cfg_data_yml = cfg_data_yml[0]

        cfg_data = Config.create_from_yaml(cfg_data_yml)
        cfg_model = Config.create_from_yaml(cfg_model_yml)
        return cfg_data, cfg_model

    def chmod(self):
        try:
            subprocess.call(["chmod", "777", "-R", self.expr_path.as_posix()])
        except Exception as e:
            LOGGER.error(e)
