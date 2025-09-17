import abc
import random
import typing as tp
import itertools

from copy import deepcopy
from functools import partial

import torch

from torch import nn
from torch.nn import functional as F

from speechflow.training.base_model import BaseTorchModelParams
from speechflow.utils.gpu_profiler import gpu_profiler
from speechflow.utils.tensor_utils import get_lengths_from_mask, get_mask_from_lengths
from tts.acoustic_models.modules.data_types import (
    MODEL_INPUT_TYPE,
    ComponentInput,
    ComponentOutput,
)

__all__ = ["Component"]

_INPUT_DIM = tp.Optional[tp.Union[int, tp.Tuple[int, ...]]]


class InstanceCounterMeta(type):
    """Metaclass to make instance counter not share count with descendants."""

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._ids = itertools.count(0)


class Component(nn.Module, metaclass=InstanceCounterMeta):
    """Base class for Model Components."""

    components_input_dim = {}
    components_output_dim = {}

    def __init__(
        self,
        params: BaseTorchModelParams,
        input_dim: _INPUT_DIM = None,
    ):
        self.params = deepcopy(params)
        self.input_dim = input_dim
        self.id = next(self.__class__._ids)
        self.components_input_dim[self.name] = self.input_dim
        self.components_output_dim[self.name] = partial(
            self.__class__.output_dim.fget, self
        )
        if self.tag != "default":
            self.components_input_dim[self.tag] = self.input_dim
            self.components_output_dim[self.tag] = self.components_output_dim[self.name]

        super().__init__()

    @property
    @abc.abstractmethod
    def output_dim(self) -> _INPUT_DIM:
        raise NotImplementedError

    @abc.abstractmethod
    @tp.overload
    def forward_step(self, inputs: ComponentInput) -> ComponentOutput:
        ...

    @abc.abstractmethod
    @tp.overload
    def forward_step(
        self,
        x: torch.Tensor,
        x_length: torch.BoolTensor,
        model_inputs: MODEL_INPUT_TYPE,
        **kwargs,
    ) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        ...

    @abc.abstractmethod
    def forward_step(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}_{self.id}"

    @property
    def tag(self):
        return self.params.tag

    @tag.setter
    def tag(self, tag: str):
        self.params.tag = tag
        for _, m in self.named_modules():
            if isinstance(m, Component):
                m.params.tag = tag

    @property
    def device(self) -> torch.device:
        if list(self.parameters()):
            return next(self.parameters()).device
        else:
            raise NotImplementedError("impossible to get device type for current module")

    @gpu_profiler
    def forward(self, *args, **kwargs) -> ComponentOutput:
        outputs = self.forward_step(*args, **kwargs)

        if isinstance(outputs, ComponentOutput):
            outputs.additional_content[self.name] = outputs.content
            if self.tag != "default":
                outputs.additional_content[self.tag] = outputs.content

        elif isinstance(outputs, tuple):
            outputs[1][self.name] = outputs[0]
            if self.tag != "default":
                outputs[1][self.tag] = outputs[0]

        return outputs

    def process_content(
        self, x: torch.Tensor, x_lengths: torch.Tensor, model_inputs: MODEL_INPUT_TYPE
    ):
        inputs = ComponentInput(x, x_lengths, model_inputs=model_inputs)
        outputs = self.forward_step(inputs)
        return outputs.content, getattr(outputs, "hidden_state")

    def hook_update_content(
        self, x: torch.Tensor, x_lengths: torch.Tensor, inputs: ComponentInput
    ) -> torch.Tensor:
        return x

    def hook_update_condition(
        self, c: torch.Tensor, inputs: ComponentInput
    ) -> torch.Tensor:
        return c

    def inference_step(self, inputs: ComponentInput, **kwargs) -> ComponentOutput:
        return self.forward_step(inputs, **kwargs)

    @gpu_profiler
    def inference(self, *args, **kwargs) -> ComponentOutput:
        outputs = self.inference_step(*args, **kwargs)
        outputs.additional_content[self.name] = outputs.content
        return outputs

    def get_condition(
        self,
        inputs: tp.Union[ComponentInput, MODEL_INPUT_TYPE],
        feat_name: tp.Optional[tp.Union[str, tp.Tuple[str, ...]]] = None,
        average_by_time: bool = False,
    ):
        if not feat_name:
            return

        if isinstance(feat_name, str):
            feat_name = (feat_name,)

        if hasattr(inputs, "model_inputs"):
            model_inputs = inputs.model_inputs
        else:
            model_inputs = inputs

        if model_inputs.prosody_reference is not None:
            ref = model_inputs.prosody_reference[self.tag]
        else:
            ref = None

        g = []
        for name in feat_name:
            if name.startswith("prompt."):
                inputs = inputs.prompt
                name = name.replace("prompt.", "")

            detach = False
            name, *modifiers = name.split("<", 1)
            if modifiers and "detach" in modifiers[0]:
                detach = True

            if ref is not None and name in ref.model_feats:
                feat = ref.get_model_feat(name, device=inputs.device)
            elif hasattr(inputs, "embeddings") and name in inputs.embeddings:
                feat = inputs.embeddings[name]
            elif (
                hasattr(inputs, "additional_content")
                and name in inputs.additional_content
            ):
                feat = inputs.additional_content[name]
            elif hasattr(inputs, "additional_inputs") and name in getattr(
                inputs, "additional_inputs"
            ):
                feat = inputs.additional_inputs[name]
            elif (
                hasattr(inputs, "model_inputs")
                and name in inputs.model_inputs.additional_inputs
            ):
                feat = inputs.model_inputs.additional_inputs[name]
            elif hasattr(model_inputs, name):
                feat = getattr(model_inputs, name)
            else:
                raise KeyError(f"Condition '{name}' not found")

            if feat.ndim == 2:
                feat = feat.unsqueeze(1)

            if feat.shape[1] > 1 and average_by_time:
                feat = torch.mean(feat, dim=1, keepdim=True)

            if feat.shape[1] == 1:
                average_by_time = True

            g.append(feat.detach() if detach else feat)

        if isinstance(inputs, ComponentOutput) and inputs.content is not None:
            if isinstance(inputs.content, list):
                b = inputs.content[0].shape[0]
            else:
                b = inputs.content.shape[0]
            g = [t.expand([b] + list(t.shape)[1:]) for t in g]

        g = torch.cat(g, dim=-1).squeeze(2)

        return self.hook_update_condition(g, inputs)

    @staticmethod
    def get_chunk(
        x,
        x_lengths,
        min_len: int = None,
        max_len: int = None,
        pad_val: float = -4.0,
    ):
        x_mask = get_mask_from_lengths(x_lengths, max_length=x.shape[1])

        if min_len is None:
            min_len = x_lengths.min()

        if max_len is None:
            max_len = x_lengths.max()

        if x.shape[1] < min_len:
            gt = F.pad(
                x.transpose(2, 1), (0, min_len - x.shape[1]), value=pad_val
            ).transpose(2, 1)
            gt_mask = F.pad(x_mask, (0, min_len - x.shape[1]))
        else:
            chunk_len = min(x_lengths.min(), max_len)
            chunk_len = max(min_len, chunk_len - chunk_len % 16)
            if chunk_len != x.shape[1]:
                gt = x[:, :chunk_len, :]
                gt_mask = x_mask[:, :chunk_len]
            else:
                gt, gt_mask = x, x_mask

        return gt, get_lengths_from_mask(gt_mask)

    def get_random_chunk(
        self,
        x,
        x_lengths,
        min_len: int = None,
        max_len: int = None,
        pad_val: float = -4.0,
    ):
        x_mask = get_mask_from_lengths(x_lengths, max_length=x.shape[1])

        if min_len is None:
            min_len = x_lengths.min() // 2

        if max_len is None:
            max_len = x_lengths.max()

        if x.shape[1] < min_len:
            gt = F.pad(
                x.transpose(1, -1), (0, min_len - x.shape[1]), value=pad_val
            ).transpose(1, -1)
            gt_mask = F.pad(x_mask, (0, min_len - x.shape[1]))
        else:
            if self.training:
                delta = max(0, x_lengths.min() - min_len)
                chunk_len = min(min_len + int(delta * random.random()), max_len)
                chunk_len = max(min_len, chunk_len - chunk_len % 16)

                gt = []
                gt_mask = []
                for idx in range(x.shape[0]):
                    a = max(0, int((x_lengths[idx] - chunk_len) * random.random()))
                    b = a + chunk_len
                    gt.append(x[idx, a:b, :])
                    gt_mask.append(x_mask[idx, a:b])

                gt = torch.stack(gt).detach()
                gt_mask = torch.stack(gt_mask).detach()
            else:
                gt = x[:, :min_len, :]
                gt_mask = x_mask[:, :min_len]

        return gt, get_lengths_from_mask(gt_mask)

    @staticmethod
    def copy_content(
        content: tp.Union[tp.List[torch.Tensor], torch.Tensor, ComponentInput],
        detach: bool = False,
    ):
        if isinstance(content, ComponentInput):
            return content.copy_content(detach)
        else:
            return ComponentInput(content, None).copy_content(detach)


if __name__ == "__main__":

    class A(Component):
        def __init__(self, params, input_dim):
            super().__init__(params, input_dim)

        @property
        def output_dim(self):
            return self.input_dim

        def forward_step(self, inputs: ComponentInput) -> ComponentOutput:
            return inputs

    class B(A):
        pass

    class C(B):
        pass

    a1 = A(BaseTorchModelParams(), 0)
    a2 = A(BaseTorchModelParams(), 0)
    b1 = B(BaseTorchModelParams(), 0)
    a3 = A(BaseTorchModelParams(), 0)

    print(a1.id, a2.id, b1.id, a3.id)
    print(a1.tag)
