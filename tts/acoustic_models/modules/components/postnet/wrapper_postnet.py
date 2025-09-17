import typing as tp
import inspect

from pydantic import Field

from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, PostnetOutput
from tts.acoustic_models.modules.params import PostnetParams

__all__ = [
    "WrapperPostnet",
    "WrapperPostnetParams",
]


class WrapperPostnetParams(PostnetParams):
    base_postnet_type: str = "RNNEncoder"
    base_postnet_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})


class WrapperPostnet(Component):
    """WrapperPostnet."""

    params: WrapperPostnetParams

    def __init__(self, params: WrapperPostnetParams, input_dim):
        super().__init__(params, input_dim)

        from tts.acoustic_models.modules import TTS_DECODERS, TTS_ENCODERS

        if params.base_postnet_type in TTS_ENCODERS:
            components = TTS_ENCODERS
        elif params.base_postnet_type in TTS_DECODERS:
            components = TTS_DECODERS
        else:
            raise RuntimeError(f"Component '{params.base_postnet_type}' not found")

        post_cls, post_params_cls = components[params.base_postnet_type]
        post_params = post_params_cls.init_from_parent_params(
            params, params.base_postnet_params
        )

        if components == TTS_ENCODERS:
            post_params.encoder_num_layers = params.postnet_num_layers
            post_params.encoder_inner_dim = params.postnet_inner_dim
            post_params.encoder_output_dim = params.postnet_output_dim
            post_params.max_input_length = params.max_output_length

        self.postnet = post_cls(post_params, input_dim)

        for method in inspect.getmembers(self, predicate=inspect.ismethod):
            if method[0].startswith("hook_"):
                setattr(self.postnet, method[0], getattr(self, method[0]))

    @property
    def output_dim(self):
        return self.postnet.output_dim

    def forward_step(self, inputs: DecoderOutput) -> PostnetOutput:  # type: ignore
        outputs = self.postnet(inputs)

        if not isinstance(outputs, PostnetOutput):
            outputs = PostnetOutput.copy_from(outputs)

        if isinstance(outputs.content_lengths, list):
            outputs.content_lengths = outputs.content_lengths[0]

        return outputs
