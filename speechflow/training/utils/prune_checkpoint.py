import typing as tp
import argparse

from pathlib import Path

import torch

from tqdm import tqdm

from speechflow.training.saver import ExperimentSaver
from speechflow.utils.fs import find_files


def prune(
    checkpoint: tp.Dict[str, tp.Any], target_speakers: tp.Optional[tp.List[str]] = None
) -> tp.Dict[str, tp.Any]:
    remove_keys = [
        "callbacks",
        "optimizer_states",
        "lr_schedulers",
        "scripts",
        "dataset",
    ]
    checkpoint = {k: v for k, v in checkpoint.items() if k not in remove_keys}

    remove_names = ["criterion", "discriminator_model"]
    state_dict = checkpoint["state_dict"]
    state_dict = {
        k: v for k, v in state_dict.items() if all(name not in k for name in remove_names)
    }
    checkpoint["state_dict"] = state_dict

    if "lang_id_map" in checkpoint:
        print("langs:", list(checkpoint["lang_id_map"].keys()))

    if "speaker_id_map" in checkpoint:
        print("speakers:", list(checkpoint["speaker_id_map"].keys()))

    info = checkpoint.get("info")
    handlers = None
    if info:
        info.pop("data_pipeline")
        info.pop("dataset")
        handlers = info.get("singleton_handlers", {})

    if target_speakers:
        for name in target_speakers:
            if name not in checkpoint["speaker_id_map"]:
                raise ValueError(f"Speaker {name} not found!")

        checkpoint["n_langs"] = len(checkpoint["lang_id_map"])
        checkpoint["n_speakers"] = len(checkpoint["speaker_id_map"])
        checkpoint["speaker_id_map"] = {
            k: v for k, v in checkpoint["speaker_id_map"].items() if k in target_speakers
        }

    if target_speakers and handlers:
        if "SpeakerIDSetter" in handlers:
            handlers["SpeakerIDSetter"].id2speaker = {
                k: v
                for k, v in handlers["SpeakerIDSetter"].id2speaker.items()
                if v in target_speakers
            }
            handlers["SpeakerIDSetter"].speaker2id = {
                k: v
                for k, v in handlers["SpeakerIDSetter"].speaker2id.items()
                if k in target_speakers
            }
            handlers["SpeakerIDSetter"].unknown_speakers = None
        if "StatisticsRange" in handlers:
            for key in handlers["StatisticsRange"].statistics.keys():
                handlers["StatisticsRange"].statistics[key] = {
                    k: v
                    for k, v in handlers["StatisticsRange"].statistics[key].items()
                    if k in target_speakers
                }
        if "MeanBioEmbeddings" in handlers:
            handlers["MeanBioEmbeddings"].data = {
                k: v
                for k, v in handlers["MeanBioEmbeddings"].data.items()
                if k in target_speakers
            }

    return checkpoint


def prune_checkpoint(
    ckpt_dir: Path,
    state_dict_only: bool = False,
    target_speakers: tp.Optional[tp.List[str]] = None,
    overwrite: bool = False,
):
    if ckpt_dir.is_file():
        pathes = [ckpt_dir]
    else:
        pathes = find_files(ckpt_dir.as_posix(), extensions=(".ckpt",))

    for path in tqdm(pathes):
        ckpt_path = Path(path)
        print(ckpt_path.as_posix())

        if overwrite:
            out_ckpt_path = ckpt_path
        else:
            out_ckpt_path = ckpt_path.with_suffix(".pt")

        checkpoint = ExperimentSaver.load_checkpoint(ckpt_path)
        checkpoint = prune(checkpoint, target_speakers)

        if state_dict_only:
            checkpoint = {"state_dict": checkpoint["state_dict"]}

        print("checkpoint keys:", list(checkpoint.keys()))

        torch.save(checkpoint, out_ckpt_path)
        print(out_ckpt_path.as_posix())


if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-d", "--ckpt_dir", help="checkpoints directory", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-st",
        "--state_dict_only",
        help="remove all without state_dict",
        type=bool,
        default=False,
    )
    arguments_parser.add_argument(
        "--target_speakers",
        help="available voices for synthesis",
        type=str,
        nargs="+",
        default=None,
    )
    arguments_parser.add_argument(
        "--overwrite", help="overwrite checkpoints", type=bool, default=False
    )
    args = arguments_parser.parse_args()

    prune_checkpoint(
        args.ckpt_dir, args.state_dict_only, args.target_speakers, args.overwrite
    )
