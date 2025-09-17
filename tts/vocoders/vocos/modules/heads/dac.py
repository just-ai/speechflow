import typing as tp

from torch import nn

from speechflow.data_pipeline.datasample_processors.algorithms.audio_processing.audio_codecs import (
    DescriptAC,
)
from speechflow.io import tp_PATH
from speechflow.training.base_model import BaseTorchModelParams
from tts.vocoders.vocos.modules.heads.base import WaveformGenerator

__all__ = ["DACHead", "DACHeadParams"]


class DACHeadParams(BaseTorchModelParams):
    input_dim: int = 1024
    pretrain_path: tp.Optional[tp_PATH] = None


class DACHead(WaveformGenerator):
    params: DACHeadParams

    def __init__(self, params: DACHeadParams):
        super().__init__(params)
        self.dac_model = DescriptAC(pretrain_path=params.pretrain_path)
        self.proj = nn.Linear(params.input_dim, self.dac_model.embedding_dim)

    def forward(self, x, **kwargs):
        z_hat = self.proj(x.transpose(1, -1)).transpose(1, -1)
        y_g_hat = self.dac_model.model.decoder(10.0 * z_hat)
        return y_g_hat.squeeze(1), None, {}
