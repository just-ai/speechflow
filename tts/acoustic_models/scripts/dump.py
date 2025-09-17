import random
import shutil
import typing as tp
import logging
import argparse
import itertools

from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import torch
import numpy.typing as npt

from numpy.random import default_rng
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from speechflow.data_pipeline.collate_functions.tts_collate import TTSCollateOutput
from speechflow.data_pipeline.core import Batch
from speechflow.data_pipeline.datasample_processors.tts_processors import (
    ContoursExtractor,
)
from speechflow.data_server.helpers import LoaderParams, init_data_loader_from_config
from speechflow.io import Config, change_config_file, json_dump_to_file
from speechflow.logging import set_verbose_logging
from speechflow.logging.server import LoggingServer

try:
    from annoy import AnnoyIndex
except ImportError as e:
    print(f"Annoy import failed: {e}")

LOGGER = logging.getLogger("root")


def parse_args():
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-cd",
        "--data_config_path",
        help="path to yaml config for data server",
        type=Path,
        required=True,
    )
    arguments_parser.add_argument(
        "-bs",
        "--batch_size",
        help="batch size",
        type=int,
        default=16,
    )
    arguments_parser.add_argument(
        "-nproc",
        "--n_processes",
        help="number of CPU processes for data_server",
        type=int,
        default=1,
    )
    arguments_parser.add_argument(
        "-ngpu",
        "--n_gpus",
        help="number of GPU device for data_server",
        type=int,
        default=0,
    )
    arguments_parser.add_argument(
        "-vs", "--value_select", help="select specific values", nargs="+", type=str
    )
    arguments_parser.add_argument(
        "-d", "--data_root", help="data root directory", type=Path
    )
    arguments_parser.add_argument(
        "-cont_len",
        "--contour_length",
        help="length of the contour",
        type=int,
        default=80,
    )
    arguments_parser.add_argument(
        "-ssize", "--subset_size", help="size of subset", type=int, default=10_000
    )
    arguments_parser.add_argument(
        "-n_cl", "--n_clusters", help="number of clusters", type=int, default=500
    )
    arguments_parser.add_argument(
        "-max_cnt",
        "--max_contours_per_speaker",
        help="number of contours per speaker",
        type=int,
        default=500,
    )
    arguments_parser.add_argument(
        "-n_sp_avg",
        "--num_speaker_emb_to_average",
        help="number of speaker embeddings to average",
        type=int,
        default=20,
    )
    args = arguments_parser.parse_args()
    return args


def update_config(
    cfg: Config,
    n_processes: int,
    n_gpus: int,
    remove_normalize: bool = True,
    remove_augmentation: bool = True,
    contours_clustering: bool = False,
) -> Config:
    cfg["dataset"]["subsets"] = [cfg["dataset"]["subsets"][0]]
    cfg["dataset"]["split_ratio"] = {cfg["dataset"]["subsets"][0]: [0, 1]}

    if not cfg["processor"]["dump"].get("full_dump", False):
        if "trim" in cfg["preproc"]["pipe"]:
            cfg["preproc"]["pipe"].remove("trim")

        if remove_normalize and not contours_clustering:
            cfg["preproc"]["pipe"] = [
                item for item in cfg["preproc"]["pipe"] if "norm" not in item
            ]

        if remove_augmentation:
            cfg["preproc"]["pipe"] = [
                item for item in cfg["preproc"]["pipe"] if "aug" not in item
            ]

    cfg["singleton_handlers"]["handlers"] = [
        item
        for item in cfg["singleton_handlers"]["handlers"]
        if item not in ["StatisticsRange", "MeanBioEmbeddings", "DatasetStatistics"]
    ]

    cfg["sampler"] = {"type": "SimpleSampler"}

    if contours_clustering:
        cfg["sampler"]["comb_by_len"] = True

    cfg["collate"]["type"] = cfg["collate"]["type"].replace("WithPrompt", "")
    cfg["data_server"]["n_processes"] = n_processes
    cfg["data_server"]["n_gpus"] = n_gpus
    cfg["processor"]["output_collated_only"] = False
    cfg["processor"]["dump"]["skip_samples_without_dump"] = False

    for item in cfg["preproc"]["pipe_cfg"].values():
        if item.get("type") == "VoiceBiometricProcessor":
            item.pop("mean_embeddings_file", None)

    cfg["singleton_handlers"]["SpeakerIDSetter"].pop("mean_embeddings_file", None)
    if "mean_bio_embedding" in cfg["preproc"]["pipe"]:
        cfg["preproc"]["pipe"].remove("mean_bio_embedding")

    return cfg


def validate_dump(
    data_config_path: Path,
    dump_folder: Path,
    batch_size: int,
    n_processes: int,
    value_select: tp.Optional[tp.List[str]] = None,
):
    print("Validation of dump files ...")

    cfg = Config.create_from_file(data_config_path, value_select=value_select)
    cfg["data_server"]["n_processes"] = n_processes
    cfg["data_server"]["n_gpus"] = 0

    val_config_path = dump_folder / "cfg_validate.yml"
    cfg.to_file(dump_folder / "cfg_validate.yml")

    with LoggingServer.ctx(dump_folder):
        with init_data_loader_from_config(
            loader_params=LoaderParams(batch_size=batch_size, non_stop=False),
            data_config_path=val_config_path,
            value_select=value_select,
        ) as data_loaders:
            count = 0
            for data_loader in data_loaders.values():
                for batch in tqdm(
                    data_loader.get_epoch_iterator(), desc=f"{data_loader.subset_name}"
                ):
                    count += batch.size
                    if count > 10000:
                        break

            print("total samples:", count)


def clustering(
    contours: np.array,
    subset_size: int = 10_000,
    n_clusters: int = 500,
    n_trees: int = 500,
):
    """Function for contours clustering.

    1. First, a random subset of the given length is obtained.
    2. Then the matrix of euclidian distances is computed for the given subset.
    3. Contours in this subset are clustered by Agglomerative algorithm with the given matrix of distances.
    4. An index is built for labels searching

    """

    rng = default_rng()
    idx = rng.choice(contours.shape[0], size=subset_size, replace=False)
    subset = contours[idx, :]

    D = pairwise_distances(subset)
    clustering = FeatureAgglomeration(
        n_clusters, metric="precomputed", linkage="complete", compute_full_tree=False
    ).fit(D)
    labels = clustering.labels_

    t = AnnoyIndex(contours.shape[1], "euclidean")
    for i, v in enumerate(subset):
        t.add_item(i, v)

    t.build(n_trees)
    return t, labels


def contours_gathering(
    batch: Batch,
    contours: tp.Dict[str, tp.Dict[str, list]],
    contour_length: int,
    max_contours_per_speaker: int = 500,
) -> np.array:
    for ds in batch.data_samples:
        c = contours[ds.speaker_name].setdefault(ds.intonation_type, [])
        if ds is not None and len(c) <= max_contours_per_speaker:
            for contour, words_length in ContoursExtractor.extract(ds, contour_length):
                if contour is not None:
                    c.append(contour)

    return contours


def get_stat(values, quantile: float = 0.05, min_val: float = 1e-2):
    val_min = []
    val_max = []
    val_mean = []
    val_var = []
    values = values.float()
    for i in range(values.shape[0]):
        val = values[i]
        if val.shape[0] != 1:
            _clamp = val[val > min_val]
            if len(_clamp) == 0:
                val_max.append(0.0)
                val_min.append(0.0)
                val_mean.append(0.0)
                val_var.append(0.0)
            else:
                _min = _clamp.quantile(quantile)
                _max = _clamp.quantile(1 - quantile)
                _clamp = torch.clamp(_clamp, _min, _max)
                val_max.append(_max.cpu().numpy())
                val_min.append(_min.cpu().numpy())
                val_mean.append(_clamp.mean().cpu().numpy())
                val_var.append(_clamp.var().cpu().numpy())
        else:
            val_max.append(val.cpu().numpy())
            val_min.append(val.cpu().numpy())
            val_mean.append(val.cpu().numpy())
            val_var.append(0.0)

    return val_min, val_max, val_mean, val_var


def main(
    data_config_path: Path,
    batch_size: int = 16,
    n_processes: int = 1,
    n_gpus: int = 0,
    value_select: tp.Optional[tp.List[str]] = None,
    data_root: tp.Optional[Path] = None,
    attributes: tp.Optional[tp.Tuple[str]] = (
        "durations",
        "energy",
        "pitch",
        "rate",
    ),  # attributes for calculate ranges
    quantile: float = 0.05,  # quantile value for calculate ranges
    contour_length: int = 80,
    subset_size: int = 10_000,
    n_clusters: int = 500,
    max_contours_per_speaker: int = 500,
    num_speaker_emb_to_average: int = 20,
):
    if data_root is not None:
        change_config_file(data_config_path, {"data_root": data_root})

    cfg = Config.create_from_file(data_config_path, value_select=value_select)

    if "contours" in cfg["preproc"]["pipe"]:
        contours_clustering = True
        cfg["preproc"]["pipe"].remove("contours")
    else:
        contours_clustering = False

    if "dump" not in cfg["processor"] or cfg["processor"]["dump"] is None:
        raise ValueError("section 'processor.dump' not configured")

    dump_folder = Path(cfg["processor"]["dump"].get("dump_path"))
    if dump_folder.exists():
        try:
            prev_cfg = Config.create_from_file(
                dump_folder / "cfg_orig.yml", value_select=value_select
            )
            prev_dump_cfg = prev_cfg["processor"]["dump"]
            curr_dump_cfg = cfg["processor"]["dump"]
            if (
                prev_cfg["preproc"]["pipe"] != cfg["preproc"]["pipe"]
                or prev_dump_cfg.get("fields") != curr_dump_cfg.get("fields")
                or prev_dump_cfg.get("handlers") != curr_dump_cfg.get("handlers")
            ):
                if click.confirm(
                    f"The pipe configuration has been changed, do you want to remove the old dump? ({dump_folder.as_posix()})"
                ):
                    LOGGER.warning(f"Remove dump folder {dump_folder.as_posix()}")
                    shutil.rmtree(dump_folder, ignore_errors=False)
        except Exception as e:
            print(e)

    dump_folder.mkdir(parents=True, exist_ok=True)
    if dump_folder.exists():
        shutil.copy(data_config_path, dump_folder / "cfg_orig.yml")

    for key, value in cfg["preproc"].items():
        if isinstance(value, dict):
            if value.get("type") == "PitchProcessor" and value.get("method") == "yingram":
                value["method"] = "pyworld"

    cfg = update_config(cfg, n_processes, n_gpus, contours_clustering=contours_clustering)
    dump_config_path = dump_folder / "cfg_for_dump.yml"
    cfg.to_file(dump_config_path)

    attributes = [] if attributes is None else list(attributes)
    attr_minmax: tp.Dict[str, tp.Dict] = {}
    for attr in attributes:
        attr_minmax[attr] = {
            "min": defaultdict(list),
            "max": defaultdict(list),
            "mean": defaultdict(list),
            "var": defaultdict(list),
        }

    speaker_bio_embeddings: tp.Dict[str, tp.List[npt.NDArray]] = defaultdict(list)
    contours: tp.Dict[str, tp.Dict[str, list]] = defaultdict(dict)

    with LoggingServer.ctx(dump_folder):
        with init_data_loader_from_config(
            loader_params=LoaderParams(batch_size=batch_size, non_stop=False),
            data_config_path=dump_config_path,
            value_select=value_select,
        ) as data_loaders:
            for data_loader in data_loaders.values():

                speaker_id_handler = data_loader.client.find_info("SpeakerIDSetter")
                if speaker_id_handler is None:
                    raise RuntimeError(
                        "Compute ranges supported for multispeaker configuration only"
                    )
                lang_id_map = speaker_id_handler.id2lang
                speaker_id_map = speaker_id_handler.id2speaker

                if len(data_loader) < subset_size:
                    subset_size = len(data_loader)
                if n_clusters > subset_size:
                    n_clusters = max(subset_size // 10, 2)

                LOGGER.info(f"dump process: {data_loader.subset_name}")
                sample_counter = 0
                for _ in tqdm(range(len(data_loader))):
                    batch = next(data_loader)
                    sample_counter += batch.size

                    collated: TTSCollateOutput = batch.collated_samples
                    assert collated is not None

                    speaker_ids = collated.speaker_id.cpu().numpy()
                    speaker_names = [speaker_id_map[s_id] for s_id in speaker_ids]

                    for attr in attributes.copy():
                        values = getattr(collated, attr, None)
                        if values is None and collated.averages is not None:
                            values = collated.averages.get(attr)

                        if values is None:
                            LOGGER.warning(f"Attribute '{attr}' not found in collated.")
                            attributes = list(a for a in attributes if a != attr)

                        if values is not None and values.ndim != 2:
                            LOGGER.warning(
                                f"Attribute '{attr}' must have two dimensions."
                            )
                            attributes = list(a for a in attributes if a != attr)

                    for attr in attributes:
                        values = getattr(collated, attr, None)
                        if values is None:
                            values = collated.averages.get(attr)

                        val_min, val_max, val_mean, val_var = get_stat(values, quantile)
                        for idx, sp_name in enumerate(speaker_names):
                            attr_minmax[attr]["min"][sp_name].append(val_min[idx])
                            attr_minmax[attr]["max"][sp_name].append(val_max[idx])
                            attr_minmax[attr]["mean"][sp_name].append(val_mean[idx])
                            attr_minmax[attr]["var"][sp_name].append(val_var[idx])

                    if num_speaker_emb_to_average:
                        sp_embs = getattr(collated, "speaker_emb", None)
                        if sp_embs is not None:
                            sp_embs = sp_embs.cpu().numpy()
                            for idx, sp_name in enumerate(speaker_names):
                                speaker_bio_embeddings[sp_name].append(sp_embs[idx])

                    if contours_clustering:
                        contours = contours_gathering(
                            batch, contours, contour_length, max_contours_per_speaker
                        )

                if data_loader.subset_name == "train":
                    lang_id_map_path = dump_folder / "lang_id_map.json"
                    json_dump_to_file(lang_id_map_path, lang_id_map)

                    speaker_id_map_path = dump_folder / "speaker_id_map.json"
                    json_dump_to_file(speaker_id_map_path, speaker_id_map)

                    all_speakers_ranges: tp.Dict[str, tp.Dict] = defaultdict(dict)
                    for attr in attributes:
                        for sp_name in attr_minmax[attr]["min"].keys():

                            def to_array(key: str):
                                x = np.array(attr_minmax[attr][key][sp_name])
                                return x[~np.isnan(x)]

                            all_speakers_ranges[attr][sp_name] = {
                                "min": float(to_array("min").min()),
                                "max": float(to_array("max").max()),
                                "mean": float(np.median(to_array("mean"))),
                                "var": float(np.median(to_array("var"))),
                            }

                    ranges_path = dump_folder / "ranges.json"
                    json_dump_to_file(ranges_path, all_speakers_ranges)

                    if num_speaker_emb_to_average:
                        mean_embeddings = {}
                        for idx, sp_name in enumerate(speaker_bio_embeddings):
                            mean_emb = random.choices(
                                speaker_bio_embeddings[sp_name],
                                k=num_speaker_emb_to_average,
                            )
                            mean_emb = np.mean(np.stack(mean_emb), axis=0)
                            mean_embeddings[sp_name] = mean_emb.tolist()

                        mean_embeddings_path = dump_folder / "mean_bio_embeddings.json"
                        json_dump_to_file(mean_embeddings_path, mean_embeddings)

                    if contours_clustering:
                        all_contours = []
                        for c in contours.values():
                            all_contours += list(
                                itertools.chain.from_iterable(c.values())
                            )

                        a_index, labels = clustering(
                            np.stack(all_contours),
                            subset_size=subset_size,
                            n_clusters=n_clusters,
                        )

                        index_filename = dump_folder / "index.ann"
                        labels_filename = dump_folder / "labels.npy"

                        a_index.save(str(index_filename))
                        np.save(labels_filename.as_posix(), labels)

                if data_loader.dataset_size != sample_counter:
                    LOGGER.info(
                        f"Not all data was dumped! "
                        f"(expected quantity: {data_loader.dataset_size}, actual quantity: {sample_counter})"
                    )

    validate_dump(data_config_path, dump_folder, batch_size, n_processes, value_select)


if __name__ == "__main__":
    """
    example:
        dump.py -cd=../configs/tts/tts_data_24khz.yml -nproc=10 -ngpu=1

    """
    set_verbose_logging()
    main(**parse_args().__dict__)
