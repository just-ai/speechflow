import typing as tp

from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import EncoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import VarianceAdaptorParams

__all__ = ["DummyVarianceAdaptor", "DummyVarianceAdaptorParams"]


class DummyVarianceAdaptorParams(VarianceAdaptorParams):
    pass


class DummyVarianceAdaptor(Component):
    params: DummyVarianceAdaptorParams

    def __init__(
        self,
        params: DummyVarianceAdaptorParams,
        input_dim: tp.Union[int, tp.Tuple[int, ...]],
    ):
        super().__init__(params, input_dim)
        self.input_dim = (
            (input_dim, input_dim) if isinstance(input_dim, int) else input_dim
        )

    @property
    def output_dim(self):
        return self.input_dim

    def forward_step(self, inputs: EncoderOutput, **kwargs) -> VarianceAdaptorOutput:  # type: ignore
        outputs = VarianceAdaptorOutput.copy_from(inputs)
        return outputs
