import os
import copy
import random
import typing as tp
import numbers

from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .mha import MultiheadAttention
from .modules import ActivationBalancer, BalancedDoubleSwish, ScaledLinear

_shape_t = Union[int, List[int], torch.Size]


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)


class MultiEmbedding(nn.Module):
    """This embedding sums embeddings on different levels."""

    def __init__(self, max_n_levels, n_tokens, token_dim):
        super().__init__()
        self.max_n_levels = max_n_levels
        self.n_tokens = n_tokens
        self.weight = nn.Parameter(torch.randn(max_n_levels, n_tokens, token_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = F.one_hot(x.transpose(1, -1), num_classes=self.n_tokens).to(self.weight)
        x = torch.einsum("l k d, n l s k -> n s d", self.weight, x)
        return x


class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return (
                F.layer_norm(
                    input,
                    self.normalized_shape,
                    self.weight,
                    self.bias,
                    self.eps,
                ),
                embedding,
            )

        assert embedding is None
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super().__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            weight, bias = torch.split(
                self.project_layer(embedding),
                split_size_or_sections=self.d_model,
                dim=-1,
            )
            return (weight * self.norm(input) + bias, embedding)

        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(input) + bias


class _BasicNorm(torch.nn.Module):
    """This is intended to be a simpler, and hopefully cheaper, replacement for LayerNorm.
    The observation this is based on, is that Transformer-type networks, especially with
    pre-norm, sometimes seem to set one of the feature dimensions to a large constant
    value (e.g. 50), which "defeats" the LayerNorm because the output magnitude is then
    not strongly dependent on the other (useful) features.  Presumably the weight and bias
    of the LayerNorm are required to allow it to do this.

    So the idea is to introduce this large constant value as an explicit
    parameter, that takes the role of the "eps" in LayerNorm, so the network
    doesn't have to do this trick.  We make the "eps" learnable.

    Args:
       num_channels: the number of channels, e.g. 512.
      channel_dim: the axis/dimension corresponding to the channel,
        interprted as an offset from the input's ndim if negative.
        shis is NOT the num_channels; it should typically be one of
        {-2, -1, 0, 1, 2, 3}.
       eps: the initial "epsilon" that we add as ballast in:
             scale = ((input_vec**2).mean() + epsilon)**-0.5
          Note: our epsilon is actually large, but we keep the name
          to indicate the connection with conventional LayerNorm.
       learn_eps: if true, we learn epsilon; if false, we keep it
         at the initial value.
    eps_min: float
    eps_max: float

    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        eps: float = 0.25,
        learn_eps: bool = True,
        eps_min: float = -3.0,
        eps_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        if learn_eps:
            self.eps = nn.Parameter(torch.tensor(eps).log().detach())
        else:
            self.register_buffer("eps", torch.tensor(eps).log().detach())
        self.eps_min = eps_min
        self.eps_max = eps_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[self.channel_dim] == self.num_channels
        eps = self.eps
        if self.training and random.random() < 0.25:
            # with probability 0.25, in training mode, clamp eps between the min
            # and max; this will encourage it to learn parameters within the
            # allowed range by making parameters that are outside the allowed
            # range noisy.
            # gradients to allow the parameter to get back into the allowed region
            # if it happens to exit it.
            eps = eps.clamp(min=self.eps_min, max=self.eps_max)
        scales = (
            torch.mean(x**2, dim=self.channel_dim, keepdim=True) + eps.exp()
        ) ** -0.5
        return x * scales


class BasicNorm(_BasicNorm):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__(d_model, eps=eps)

    def forward(self, input: torch.Tensor, embedding: Any = None) -> torch.Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return (
                super().forward(input),
                embedding,
            )

        assert embedding is None
        return super().forward(input)


class BalancedBasicNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.balancer = ActivationBalancer(
            d_model,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            max_abs=6.0,
        )
        self.norm = BasicNorm(d_model, eps, device=device, dtype=dtype)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return self.norm((self.balancer(input), embedding))

        assert embedding is None
        return self.norm(self.balancer(input))


class IdentityNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            return input

        assert embedding is None
        return input


class TransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        linear1_self_attention_cls: nn.Module = nn.Linear,
        linear2_self_attention_cls: nn.Module = nn.Linear,
        linear1_feedforward_cls: nn.Module = nn.Linear,
        linear2_feedforward_cls: nn.Module = nn.Linear,
        layer_norm_cls: nn.Module = LayerNorm,
        layer_norm_eps: float = 1e-5,
        adaptive_layer_norm=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # self.self_attn = MultiheadAttention(
        #     d_model,
        #     nhead,
        #     dropout=dropout,
        #     batch_first=batch_first,
        #     linear1_cls=linear1_self_attention_cls,
        #     linear2_cls=linear2_self_attention_cls,
        #     **factory_kwargs,
        # )
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )

        # Implementation of Feedforward model
        self.linear1 = linear1_feedforward_cls(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        elif isinstance(activation, partial):
            activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            activation = BalancedDoubleSwish(d_model)

        # # We can't test self.activation in forward() in TorchScript,
        # # so stash some information about it instead.
        # if activation is F.relu or isinstance(activation, torch.nn.ReLU):
        #     self.activation_relu_or_gelu = 1
        # elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
        #     self.activation_relu_or_gelu = 2
        # else:
        #     self.activation_relu_or_gelu = 0
        self.activation = activation

        norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
        if layer_norm_cls == IdentityNorm:
            norm2 = BalancedBasicNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            norm2 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)

        if adaptive_layer_norm:
            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
        else:
            self.norm1 = norm1
            self.norm2 = norm2

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x, stage_embedding = src, None
        is_src_tuple = False
        if isinstance(src, tuple):
            x, stage_embedding = src
            is_src_tuple = True

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, stage_embedding),
                src_mask,
                src_key_padding_mask,
            )
            x = x + self._ff_block(self.norm2(x, stage_embedding))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask),
                stage_embedding,
            )
            x = self.norm2(x + self._ff_block(x), stage_embedding)

        if is_src_tuple:
            return (x, stage_embedding)
        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]

        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_layer_states: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            return_layer_states: return layers' state (optional).

        Shape:
            see the docs in Transformer class.
        """
        if return_layer_states:
            layer_states = []  # layers' output
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                )
                layer_states.append(output[0])

            if self.norm is not None:
                output = self.norm(output)

            return layer_states, output

        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        linear1_feedforward_cls: nn.Module = nn.Linear,
        linear2_feedforward_cls: nn.Module = nn.Linear,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        layer_norm_cls: nn.Module = LayerNorm,
        layer_norm_eps: float = 1e-5,
        adaptive_layer_norm=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = linear1_feedforward_cls(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        elif isinstance(activation, partial):
            self.activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            self.activation = BalancedDoubleSwish(d_model)
        else:
            self.activation = activation

        if adaptive_layer_norm:
            norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
            norm2 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
            norm3 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)

            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
            self.norm3 = AdaptiveLayerNorm(d_model, norm3)
        else:
            self.norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
            if layer_norm_cls == IdentityNorm:
                self.norm3 = BalancedBasicNorm(
                    d_model, eps=layer_norm_eps, **factory_kwargs
                )
            else:
                self.norm3 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt_is_tuple = False
        if isinstance(tgt, tuple):
            x, stage_embedding = tgt
            tgt_is_tuple = True
        else:
            x, stage_embedding = tgt, None

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, stage_embedding), tgt_mask, tgt_key_padding_mask
            )
            x = x + self._mha_block(
                self.norm2(x, stage_embedding),
                memory,
                memory_mask,
                memory_key_padding_mask,
            )
            x = x + self._ff_block(self.norm3(x, stage_embedding))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask),
                stage_embedding,
            )
            x = self.norm2(
                x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                stage_embedding,
            )
            x = self.norm3(x + self._ff_block(x), stage_embedding)

        if tgt_is_tuple:
            return (x, stage_embedding)
        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")
