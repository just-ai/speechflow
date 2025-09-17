import typing as tp
import logging

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import pydantic
import pytorch_lightning as pl

from torch import nn
from tqdm import tqdm

from speechflow.data_pipeline.collate_functions.utils import collate_sequence
from speechflow.data_pipeline.core import (
    BaseBatchProcessor,
    BaseCollate,
    BaseCollateOutput,
    BaseDSParser,
    BaseDSProcessor,
    Batch,
    DataPipeline,
    DataSample,
    Dataset,
    PipeRegistry,
    Singleton,
    TrainData,
    tp_DATA,
)
from speechflow.data_pipeline.datasample_processors.algorithms.audio_processing.ssl_models import (
    Whisper,
)
from speechflow.data_pipeline.samplers import RandomSampler
from speechflow.data_server.helpers import LoaderParams, init_data_loader
from speechflow.data_server.loader import DataLoader
from speechflow.io import (
    AudioChunk,
    AudioSeg,
    Config,
    construct_file_list,
    split_file_list,
)
from speechflow.logging import set_verbose_logging
from speechflow.logging.server import LoggingServer
from speechflow.training import (
    BaseCriterion,
    BaseTorchModel,
    BaseTorchModelParams,
    ExperimentSaver,
    LightningEngine,
    Optimizer,
)
from speechflow.utils.init import lazy_initialization
from speechflow.utils.profiler import Profiler
from speechflow.utils.tensor_utils import run_rnn_on_padded_sequence

LOGGER = logging.getLogger("root")


#
# --------------  DATA DESCRIPTION --------------
#


@dataclass
class SSLDataSample(DataSample):
    audio_chunk: AudioChunk = None
    speaker_name: tp_DATA = None
    speaker_id: tp_DATA = None
    ssl_feat: tp_DATA = None


@dataclass
class SSLCollateOutput(SSLDataSample, BaseCollateOutput):
    ssl_feat_lengths: torch.Tensor = None


@dataclass
class SSLTarget(TrainData):
    speaker_id: torch.Tensor = None


@dataclass
class SSLForwardInput(TrainData):
    ssl_feat: torch.Tensor = None
    ssl_feat_lengths: torch.Tensor = None


@dataclass
class SSLForwardOutput(TrainData):
    logits: torch.Tensor = None


#
# -------------- DATA PROCESSORS --------------
#


class DSParser(BaseDSParser):
    def reader(self, file_path: Path, label=None) -> tp.List[tp.Dict[str, tp.Any]]:
        sega = AudioSeg.load(file_path)
        metadata = {
            "file_path": file_path,
            "audio_chunk": sega.audio_chunk,
            "speaker_name": sega.meta["speaker_name"],
        }
        return [metadata]

    def converter(self, metadata: tp.Dict[str, tp.Any]) -> tp.List[SSLDataSample]:
        ds = SSLDataSample(
            file_path=metadata.get("file_path"),
            audio_chunk=metadata.get("audio_chunk"),
            speaker_name=metadata.get("speaker_name"),
        )
        return [ds]


class SignalProcessor(BaseDSProcessor):
    def __init__(self, sample_rate: int = 24000):
        super().__init__()
        self._sample_rate = sample_rate

    @PipeRegistry.registry(inputs={"audio_chunk"}, outputs={"audio_chunk"})
    def process(self, ds: SSLDataSample) -> SSLDataSample:
        ds.audio_chunk.load(sr=self._sample_rate)
        return ds


class SSLProcessor(BaseDSProcessor):
    def __init__(
        self,
        model_name: tp.Literal["tiny", "base", "small", "medium", "large-v2"] = "tiny",
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.ssl_model: tp.Optional[Whisper] = None

    def init(self):
        super().init()
        self.ssl_model = Whisper(model_name=self.model_name, device=self.device)

    @PipeRegistry.registry(inputs={"audio_chunk"}, outputs={"ssl_feat"})
    @lazy_initialization
    def process(self, ds: SSLDataSample) -> SSLDataSample:
        ds.ssl_feat = self.ssl_model(ds.audio_chunk).encoder_feat
        return ds.to_numpy()


class SSLCollate(BaseCollate):
    def collate(self, batch: tp.List[SSLDataSample]) -> SSLCollateOutput:  # type: ignore
        collated = super().collate(batch)
        collated = SSLCollateOutput(**collated.to_dict())  # type: ignore

        ssl_feat, length = collate_sequence(batch, "ssl_feat", pad_values=0)

        if batch[0].speaker_id is not None:
            speaker_id = [x.speaker_id for x in batch]
            speaker_id = torch.LongTensor(speaker_id)
        else:
            speaker_id = None

        collated.speaker_id = speaker_id
        collated.ssl_feat = ssl_feat
        collated.ssl_feat_lengths = length
        return collated


class SpeakerID(metaclass=Singleton):
    def __init__(self):
        self.speaker2id = {}
        self.id2speaker = {}

    @property
    def n_speakers(self):
        return len(self.speaker2id)

    def __call__(self, data: Dataset) -> Dataset:
        speaker_names = [ds.speaker_name for ds in data if ds.speaker_name]
        speaker_names = sorted(list(set(speaker_names)))

        if not self.speaker2id:
            self.speaker2id = {name: idx for idx, name in enumerate(speaker_names)}
            self.id2speaker = {idx: name for name, idx in self.speaker2id.items()}
        else:
            data.filter(lambda ds: ds.speaker_name in self.speaker2id)

        LOGGER.info(
            f"Target speakers ({len(self.speaker2id)}): {', '.join(self.speaker2id.keys())}"
        )

        for ds in tqdm(data, "Set speaker id"):
            ds.speaker_id = self.speaker2id[ds.speaker_name]

        total_duration = sum([ds.audio_chunk.duration for ds in data]) / 3600
        LOGGER.info(
            f"Total dataset duration {np.round(total_duration, 3)} hours ({len(data)} samples)"
        )
        return data


#
# -------------- MODEL --------------
#


class SSLBiometricModelParams(BaseTorchModelParams):
    n_speakers: int = pydantic.Field(0, ge=0)
    ssl_dim: tp.Literal[384, 1280] = 384
    proj_dim: int = pydantic.Field(256, ge=128, le=1024)
    rnn_dim: int = pydantic.Field(128, ge=64, le=256)


class SSLBiometricModel(BaseTorchModel):
    params: SSLBiometricModelParams

    def __init__(
        self,
        cfg: tp.Union[Config, SSLBiometricModelParams],
        strict_init: bool = True,
    ):
        super().__init__(SSLBiometricModelParams.create(cfg, strict_init))

        self.ssl_proj = nn.Linear(self.params.ssl_dim, self.params.proj_dim)
        self.rnn = nn.GRU(
            self.params.proj_dim,
            self.params.rnn_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.bio_cls = nn.Linear(2 * self.params.rnn_dim, self.params.n_speakers)

    def forward(self, inputs: SSLForwardInput) -> SSLForwardOutput:
        x = self.ssl_proj(inputs.ssl_feat)
        after_rnn = run_rnn_on_padded_sequence(self.rnn, x, inputs.ssl_feat_lengths)
        y = self.bio_cls(torch.mean(after_rnn, dim=1))
        return SSLForwardOutput(logits=y)


class BatchProcessor(BaseBatchProcessor):
    def __call__(
        self, batch: Batch, batch_idx: int = 0, global_step: int = 0
    ) -> (SSLForwardInput, SSLTarget, tp.List[SSLDataSample]):
        _collated: SSLCollateOutput = batch.collated_samples  # type: ignore
        _input = SSLForwardInput(
            ssl_feat=_collated.ssl_feat, ssl_feat_lengths=_collated.ssl_feat_lengths
        )
        _target = SSLTarget(speaker_id=_collated.speaker_id)
        return _input.to(self.device), _target.to(self.device), batch.data_samples


#
# -------------- CRITERION --------------
#


class Criterion(BaseCriterion):
    def forward(
        self,
        output: SSLForwardOutput,
        target: SSLTarget,
        batch_idx: int = 0,
        global_step: int = 0,
    ) -> tp.Dict[str, torch.Tensor]:
        loss = nn.functional.nll_loss(
            nn.functional.log_softmax(output.logits, dim=1), target.speaker_id
        )
        return {"nll_loss": loss}


#
# -------------- TRAIN LOOP --------------
#


def create_data_pipline(subsets):
    set_verbose_logging()

    parser = DSParser()
    preprocessing = [SignalProcessor().process, SSLProcessor().process]
    collate = SSLCollate()
    sampler = RandomSampler()
    singleton_handlers = {"SpeakerID": SpeakerID()}
    return DataPipeline.init_from_components(
        subsets, parser, preprocessing, collate, sampler, singleton_handlers
    )


def train(experiment_path: str, loaders: tp.Dict[str, DataLoader]) -> str:
    # set seed
    pl.seed_everything(1234)

    dl_train, dl_valid = loaders.values()

    # initialize batch processor
    batch_processor = BatchProcessor()

    # create dnn model
    speaker_id_handler: SpeakerID = dl_train.client.find_info(
        "SpeakerID", section="singleton_handlers"
    )
    model_params = SSLBiometricModelParams(n_speakers=speaker_id_handler.n_speakers)
    model = SSLBiometricModel(model_params)

    # create criterion
    criterion = Criterion()

    # create optimizer
    optimizer = Optimizer(
        model,
        method={"type": "Adam", "weight_decay": 1.0e-6},
        lr_scheduler={"type": "ConstLR", "lr_max": 0.001},
    )

    # create experiment saver
    saver = ExperimentSaver(
        expr_path=experiment_path,
    )
    saver.to_save.update({"speaker_id_map": speaker_id_handler.speaker2id})

    # create engine
    pl_engine: LightningEngine = LightningEngine(
        model=model,
        criterion=criterion,
        batch_processor=batch_processor,
        optimizer=optimizer,
        saver=saver,
    )

    # create trainer callbacks
    checkpoint_callback_cfg = Config(
        {
            "monitor": "Epoch",
            "mode": "max",
            "save_top_k": 1,
            "every_n_epochs": 1,
        }
    )
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        saver.get_checkpoint_callback(cfg=checkpoint_callback_cfg),
    ]

    # create trainer
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=10,
        default_root_dir=experiment_path,
        callbacks=callbacks,
    )

    # lets try to train
    with Profiler("training", format=Profiler.Format.h):
        trainer.fit(pl_engine, dl_train, dl_valid)

    LOGGER.info("Model training completed!")
    return experiment_path


def test(expr_path: str | Path, file_path: str | Path):
    ckpt_path = ExperimentSaver.get_last_checkpoint(expr_path)
    if ckpt_path is None:
        raise FileNotFoundError("checkpoint not found")

    checkpoint = ExperimentSaver.load_checkpoint(ckpt_path)

    model = SSLBiometricModel(checkpoint["params"])
    model.eval()
    model.load_state_dict(checkpoint["state_dict"])

    metadata = _data_pipeline["valid"].dataset_parser.reader(file_path)
    batch = _data_pipeline["valid"].metadata_to_batch(metadata)

    with torch.inference_mode():
        model_input, _, _ = BatchProcessor()(batch)
        predict: SSLForwardOutput = model(model_input)

    speaker_id_map = checkpoint["speaker_id_map"]
    speaker_id = speaker_id_map[batch.data_samples[0].speaker_name]

    print("speaker_id target:", speaker_id)
    print("speaker_id predict:", torch.argmax(predict.logits).item())


if __name__ == "__main__":
    from speechflow.utils.fs import get_root_dir

    _dataset_path = get_root_dir() / "examples/simple_datasets/speech/SEGS"
    _expr_path = "_logs/test_biometric"
    _file_ext = ".TextGridStage3"

    _data_pipeline = create_data_pipline(subsets=["train", "valid"])

    _flist = construct_file_list(_dataset_path, _file_ext, with_subfolders=True)
    _flist_train, _flist_valid = split_file_list(_flist, ratio=0.8)

    if 1:  # TRAINING
        with LoggingServer.ctx(_expr_path):
            with init_data_loader(
                loader_params=LoaderParams(batch_size=8, non_stop=True),
                data_pipeline=_data_pipeline,
                flist_by_subsets={"train": _flist_train, "valid": _flist_valid},
                n_processes=1,
                n_gpus=0,
            ) as _loaders:
                train(_expr_path, _loaders)
    else:  # EVALUATION
        test(_expr_path, _flist[-1])
