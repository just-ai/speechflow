import typing as tp

from pathlib import Path

from setuptools import find_packages, setup

THIS_DIR = Path(__file__).parent


def _get_requirements():
    with (THIS_DIR / "requirements.txt").open() as fp:
        return fp.read().splitlines()


about: tp.Dict[tp.Any, tp.Any] = {}
with open("speechflow/_version.py") as f:
    exec(f.read(), about)

packages = find_packages(
    include=[
        "annotator",
        "annotator.*",
        "speechflow",
        "speechflow.*",
        "nlp",
        "nlp.*",
        "tts",
        "tts.*",
    ],
)

flist = Path("speechflow/data").rglob("*")
sdk_data = [path.relative_to("speechflow").as_posix() for path in flist]

setup(
    name="speechflow",
    version=f"{about['__version__']}",
    description="Library for experiments with tts-related pipelines.",
    packages=packages,
    python_requires=">=3.10",
    install_requires=_get_requirements(),
    package_data={"speechflow": sdk_data},
    # https://nuitka.net/doc/user-manual.html#use-case-5-setuptools-wheels
    command_options={
        "nuitka": {
            "--python-flag": "no_docstrings",
            "--module-parameter": "torch-disable-jit=no",
            "--nofollow-import-to": [
                "tests.*",
            ],
        }
    },
)
