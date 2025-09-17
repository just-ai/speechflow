import typing as tp

from copy import deepcopy
from dataclasses import dataclass

import torch

from torch import Tensor

from speechflow.utils.tensor_utils import apply_mask, get_mask_from_lengths
from tts.acoustic_models.data_types import (
    TTSForwardInput,
    TTSForwardInputWithPrompt,
    TTSForwardInputWithSSML,
)

__all__ = [
    "ComponentInput",
    "ComponentOutput",
    "EncoderOutput",
    "VarianceAdaptorOutput",
    "DecoderOutput",
    "PostnetOutput",
    "MODEL_INPUT_TYPE",
]

MODEL_INPUT_TYPE = tp.Union[
    TTSForwardInput, TTSForwardInputWithPrompt, TTSForwardInputWithSSML
]


@dataclass
class ComponentInput:
    content: tp.Union[Tensor, tp.List[Tensor]]
    content_lengths: tp.Union[Tensor, tp.List[Tensor]]
    embeddings: tp.Dict[str, Tensor] = None  # type: ignore
    additional_content: tp.Dict[str, Tensor] = None  # type: ignore
    additional_losses: tp.Dict[str, Tensor] = None  # type: ignore
    model_inputs: MODEL_INPUT_TYPE = None  # type: ignore

    def __post_init__(self):
        if self.embeddings is None:
            self.embeddings = {}
        if self.additional_content is None:
            self.additional_content = {}
        if self.additional_losses is None:
            self.additional_losses = {}

    @property
    def device(self):
        if self.model_inputs is not None:
            return self.model_inputs.device
        elif self.content is not None:
            if isinstance(self.content, tp.Sequence):
                return self.content[0].device
            else:
                return self.content.device
        elif self.embeddings:
            return list(self.embeddings.values())[0].data.device
        else:
            raise RuntimeError("device not found")

    @property
    def is_empty(self) -> bool:
        return self.content is None and self.content_lengths is None

    @staticmethod
    def empty():
        return ComponentInput(
            content=None,  # type: ignore
            content_lengths=None,  # type: ignore
        )

    @classmethod
    def copy_from(cls, x: "ComponentInput", deep: bool = False):
        new = cls(
            content=list(x.content) if isinstance(x.content, list) else x.content,
            content_lengths=list(x.content_lengths)
            if isinstance(x.content_lengths, list)
            else x.content_lengths,
            embeddings=x.embeddings,
            model_inputs=x.model_inputs,
            additional_content=x.additional_content,
            additional_losses=x.additional_losses,
        )
        return deepcopy(new) if deep else new

    def get_content(self, idx: tp.Optional[int] = None) -> tp.List[torch.Tensor]:
        content = (
            [self.content] if not isinstance(self.content, tp.Sequence) else self.content
        )
        if idx is not None:
            return content[idx]
        else:
            return content

    def get_content_lengths(self, idx: tp.Optional[int] = None) -> tp.List[torch.Tensor]:
        lens = (
            self.content_lengths
            if isinstance(self.content_lengths, tp.Sequence)
            else [self.content_lengths]
        )
        if idx is not None:
            return lens[idx]
        else:
            return lens

    def get_content_and_mask(
        self, idx: int = 0
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.get_content(idx)
        x_lens = self.get_content_lengths(idx)
        x_mask = get_mask_from_lengths(x_lens, max_length=x.shape[1])
        return x, x_lens, x_mask

    def set_content(
        self, x: Tensor, x_lens: tp.Optional[Tensor] = None, idx: tp.Optional[int] = None
    ):
        if (
            not isinstance(self.content, tp.Sequence)
            or isinstance(x, tp.Sequence)
            or idx is None
        ):
            self.content = x
            if x_lens is not None:
                self.content_lengths = x_lens
        elif isinstance(self.content, (list, tuple)) and isinstance(idx, int):
            self.content[idx] = x
            if x_lens is not None:
                self.content_lengths[idx] = x_lens
        else:
            raise NotImplementedError

        return self

    def select_content(self, idx: int = 0):
        if isinstance(self.content, tp.Sequence):
            self.content = self.content[idx]
            self.content_lengths = self.content_lengths[idx]

        return self

    def copy_content(self, detach: bool = False) -> tp.Union[Tensor, tp.List[Tensor]]:
        if isinstance(self.content, tp.Sequence):
            if detach:
                return [t.clone().detach() for t in self.content]
            else:
                return [t.clone() for t in self.content]
        else:
            if detach:
                return self.content.clone().detach()
            else:
                return self.content.clone()

    def cat_content(self, dim: int = 0):
        if isinstance(self.content, tp.Sequence):
            return torch.cat(list(self.content), dim=dim)
        else:
            return self.content

    def stack_content(self):
        if isinstance(self.content, tp.Sequence):
            try:
                return torch.stack(list(self.content))
            except RuntimeError:
                return self.content
        else:
            return self.content.unsqueeze(0)

    def apply_mask(self, mask: Tensor):
        self.content = apply_mask(self.content, mask)
        return self


ComponentOutput = ComponentInput


@dataclass
class EncoderOutput(ComponentOutput):
    hidden_state: Tensor = None

    def set_hidden_state(self, x: Tensor):
        self.hidden_state = x
        return self


@dataclass
class VarianceAdaptorOutput(ComponentOutput):
    attention_weights: Tensor = None
    variance_predictions: tp.Dict[str, Tensor] = None  # type: ignore

    def __post_init__(self):
        if self.variance_predictions is None:
            self.variance_predictions = {}


@dataclass
class DecoderOutput(ComponentOutput):
    hidden_state: Tensor = None
    gate: Tensor = None

    def set_hidden_state(self, x: Tensor):
        self.hidden_state = x
        return self


@dataclass
class PostnetOutput(ComponentOutput):
    pass
