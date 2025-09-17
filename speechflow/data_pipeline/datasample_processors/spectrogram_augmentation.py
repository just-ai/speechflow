import random
import typing as tp
import logging
import multiprocessing as mp

import numpy as np
import torch

from speechflow.data_pipeline.core.base_ds_processor import BaseDSProcessor
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.audio_augmentation import (
    WaveAugProcessor,
)
from speechflow.data_pipeline.datasample_processors.data_types import (
    SpectrogramDataSample,
)
from speechflow.data_pipeline.datasample_processors.utils import check_probability
from speechflow.utils.init import lazy_initialization

LOGGER = logging.getLogger("root")

try:
    from nemo.collections.asr.modules import SpectrogramAugmentation

    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
except ImportError as e:
    if mp.current_process().name == "MainProcess":
        LOGGER.warning(f"NeMo is not available: {e}")


__all__ = [
    "SpecAugProcessor",
    "NemoSpecAugProcessor",
]

LOGGER = logging.getLogger("root")


class SpecAugProcessor(WaveAugProcessor):
    @PipeRegistry.registry(inputs={"magnitude", "mel"}, outputs={"magnitude", "mel"})
    def process(self, ds: SpectrogramDataSample) -> SpectrogramDataSample:
        ds = super().process(ds)

        if random.random() > self._p:
            return ds

        handlers = list(self.components.values())
        if self._shuffle:
            random.shuffle(handlers)

        for handler in handlers:
            if hasattr(handler, "keywords"):
                p = handler.keywords.get("p", 1.0)  # type: ignore
                ds = handler(ds=ds, p=p)
            else:
                raise NotImplementedError()

        return ds

    @check_probability
    def blur(
        self, ds: SpectrogramDataSample, radius: int = (1, 3, 5), sigma=None, *kwargs
    ):
        from scipy.ndimage.filters import gaussian_filter

        if sigma is None:
            sigma = 0.75 * random.random()

        if isinstance(radius, tp.Iterable):
            r = np.random.choice(radius)
        else:
            r = radius

        ds.mel = gaussian_filter(ds.mel, sigma=sigma, radius=r)
        return ds

    @check_probability
    def noise(self, ds: SpectrogramDataSample, var: float = 1.0, scale=None, *kwargs):
        if scale is None:
            scale = 0.2 * random.random()

        mel_noise = ds.mel + scale * np.random.normal(0, var, ds.mel.shape)
        ds.mel = mel_noise.astype(np.float32)
        return ds


class NemoSpecAugProcessor(BaseDSProcessor):
    def __init__(self, attributes: tp.Union[str, tp.List[str]], p: float = 1.0, **kwargs):
        super().__init__()
        self._spec_aug_cfg = self.get_config_from_locals()
        self._atts = [attributes] if isinstance(attributes, str) else attributes
        self._p = p
        self.logging_params(self.get_config_from_locals())
        self._nemo_spec_aug = None

    def init(self):
        super().init()
        self._nemo_spec_aug = SpectrogramAugmentation(**self._spec_aug_cfg["kwargs"])
        self._nemo_spec_aug.eval()

    @PipeRegistry.registry(inputs={"magnitude", "mel"}, outputs={"magnitude", "mel"})
    @lazy_initialization
    def process(self, ds: SpectrogramDataSample) -> SpectrogramDataSample:
        ds = super().process(ds)
        if random.random() > self._p:
            return ds

        for attr in self._atts:
            field = getattr(ds, attr)
            field = torch.from_numpy(field).T.unsqueeze(0)
            field_length = torch.LongTensor([field.shape[-1]])
            field = self._nemo_spec_aug(input_spec=field, length=field_length)
            setattr(ds, attr, field.squeeze(0).T)

        return ds.to_numpy()
