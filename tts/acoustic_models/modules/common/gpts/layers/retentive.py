from copy import deepcopy
from math import log
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from einops import einsum, rearrange, repeat
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

DEFAULT_DEVICE = torch.device("cpu")
ActivationString = Literal["swish", "gelu", "relu"]


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """Return an activation function given a string."""
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    else:
        raise RuntimeError(
            f"Unsupported activation string '{activation}'. "
            "Supported: 'swish', 'gelu', 'relu'"
        )


def _build_decay_gammas(
    num_heads: int,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Decay values are different for each retention head, following the prescribed method
    in the paper.  Conceptually, I think of each head having a different "retention
    window", which is the effective number of steps back in time that the head can attend
    to.  Retention windows are effectively determined by these decay coefficients.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Section 3.1 (Setup)

    """
    xmin, xmax = log(1 / 32), log(1 / 512)
    x = torch.linspace(xmin, xmax, steps=num_heads, device=device, dtype=dtype)
    return 1 - torch.exp(x)


def _build_decay_mask(
    query_length: int,
    key_length: int,
    decay_gammas: Tensor,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """The decay mask is one of the key components that makes *parallel* retention
    equivalent to *recurrent* retention.  The decay coefficients are pre-computed and
    applied to the similarity matrix at once, rather than being applied to each element in
    the recurrent formulation.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Equation 5

    """
    query_pos = torch.arange(query_length, device=device, dtype=dtype)
    key_pos = torch.arange(key_length, device=device, dtype=dtype)

    distance = torch.abs(query_pos.unsqueeze(-1) - key_pos.unsqueeze(0))
    # Set the upper-triangular distances to infinity, so that only *past* keys
    # can affect the current query.  (Setting distance to infinity ensures that
    # the decay matrix is 0 for those positions, since x^(inf) = 0 when -1 < x < 1.
    distance_mask = torch.ones_like(distance, dtype=torch.bool).triu_(diagonal=1)
    distance = distance.masked_fill(distance_mask, float("inf"))

    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(decay_gammas, "h -> h () ()")
    return decay_gammas**distance


# def _build_chunkwise_decay_mask(
#     chunk_size: int,
#     decay_gammas: Tensor,
#     device: Optional[Union[torch.device, str]] = None,
#     dtype: Optional[torch.dtype] = None,
# ) -> Tensor:
#     """The chunkwise decay mask is used in the chunkwise formulation of retention.
#     It is derived from the recurrent formulation -- we apply increasing amounts
#     of decay to the previous state, based on the position of each token in the
#     input chunk.  So, we add a 'sequence' dimension to the decay gammas, and
#     raise them to the power of the position of each token in the chunk.

#     See: https://arxiv.org/pdf/2307.08621v3.pdf, Equation 7
#     """
#     chunk_pos = torch.arange(chunk_size, device=device, dtype=dtype)
#     chunk_pos = rearrange(chunk_pos, "n -> () n")
#     decay_gammas = rearrange(decay_gammas, "h -> h ()")
#     return decay_gammas**chunk_pos


def _build_position_thetas(
    head_dim: int,
    scale: float = 10000,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Positional thetas are different for each value along head_dim, following the
    prescribed method in the paper.  These are used to update the positional
    embeddings in both the parallel and recurrent formulations of retention.
    See: https://arxiv.org/pdf/2307.08621v3.pdf, Section 2.1 (Retention)

    NOTE: The actual values for thetas are not specified in the paper, so I
    copied these values from the official implementation.
    See: https://github.com/microsoft/torchscale/blob/7d231743f4f96c460b7cf0aa0cf242bb192b34f8/torchscale/architecture/retnet.py#L27C1-L28C59
    """
    x = torch.linspace(0, 1, steps=head_dim // 2, device=device, dtype=dtype)
    thetas = 1 / (scale**x)
    return repeat(thetas, "d -> (d n)", n=2)


# NOTE: For the purposes of positional embeddings, we view query/key Tensors as
# complex-valued, where the even-numbered indices are the real part, and the
# odd-numbered indices are the imaginary part.  This makes it easy to compute
# complex values without *actually* using complex dtypes in PyTorch.
# (Complex dtypes have limited support compared to real dtypes.)
#
# I don't re-explain this in the functions below, but it's important to keep in
# mind when reading the code.


def _multiply_by_i(x: Tensor) -> Tensor:
    """Multiply a complex-valued tensor by the imaginary unit 'i'."""
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(start_dim=-2)


def _theta_shift(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # TODO: Add docstring
    return (x * cos) + (_multiply_by_i(x) * sin)


def retention_parallel(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    decay_gammas: Optional[Tensor] = None,
    need_weights: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    if decay_gammas is None:
        decay_gammas = _build_decay_gammas(
            num_heads=query.shape[1], device=query.device, dtype=query.dtype
        )
    decay_mask = _build_decay_mask(
        query_length=query.shape[2],
        key_length=key.shape[2],
        decay_gammas=decay_gammas,
        device=query.device,
        dtype=query.dtype,
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: hidden_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    if need_weights:
        return retention, similarity
    else:
        return retention, None


def retention_recurrent(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    prev_state: Optional[Tensor],
    scale: Optional[float] = None,
    decay_gammas: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    if decay_gammas is None:
        decay_gammas = _build_decay_gammas(
            num_heads=query.shape[1], device=query.device, dtype=query.dtype
        )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - d: hidden_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    state = einsum(key, value, "b h d, b h m -> b h d m")
    if prev_state is not None:
        state = state + prev_state * rearrange(decay_gammas, "h -> () h () ()")
    retention = einsum(query, state, "b h d, b h d m -> b h m")

    return retention, state


def retention_chunkwise(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    prev_state: Optional[Tensor],
    scale: Optional[float] = None,
    decay_gammas: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    if decay_gammas is None:
        decay_gammas = _build_decay_gammas(
            num_heads=query.shape[1], device=query.device, dtype=query.dtype
        )
    decay_mask = _build_decay_mask(
        query_length=query.shape[2],
        key_length=key.shape[2],
        decay_gammas=decay_gammas,
        device=query.device,
        dtype=query.dtype,
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: head_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    # intra-chunk (same as parallel retention)
    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    # cross-chunk (derived from recurrent retention)
    decay_gammas = rearrange(decay_gammas, "h -> () h () ()")
    inner_pos = rearrange(
        torch.arange(key.size(2), device=key.device, dtype=key.dtype) + 1,
        "n -> () () n ()",
    )
    states = einsum(key, value, "b h n d1, b h n d2 -> b h n d1 d2")
    state_decays = decay_gammas ** (key.size(2) - inner_pos)
    state = einsum(states, state_decays, "b h n d1 d2, _ h n _ -> b h d1 d2")
    if prev_state is not None:
        # Update internal state to return to the user
        chunk_decay = decay_gammas ** key.size(2)
        state = state + prev_state * chunk_decay
        # Update the retention Tensor, based on cross-chunk information
        inner_decay = decay_gammas**inner_pos
        retention = retention + (
            einsum(query, prev_state, "b h n d1, b h d1 d2 -> b h n d2") * inner_decay
        )

    return retention, state


class MultiScaleRetention(nn.Module):
    """Multi-scale retention (MSR) layer.  Intended to be (mostly) a drop-in replacement
    for nn.MultiheadAttention, but with the option to use either the parallel or recurrent
    formulation of retention. (Attention only has the parallel formulation.)

    NOTE: As presented in the paper, Multi-Scale Retention includes an explicit
    position embedding, which is based on xPos.  IMO, this is unnecessary and overly
    specific to language modeling, since other domains (e.g. computer vision,
    heterogeneous graphs) will have different positional semantics.

    I have made the relational position embedding optional, so that this module
    can (in theory) support more modalities. Setting 'relative_position=False' will
    remove the positional embedding, and instead rely on the query and key
    embeddings to encode positional information ahead of time (if needed at all).
    See: https://github.com/microsoft/torchscale/issues/48

    Reference:
        "Retentive Network: A Successor to Transformer for Large Language Models"
        https://arxiv.org/pdf/2307.08621v3.pdf

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        relative_position: bool = True,
        bias: bool = True,
        batch_first: bool = True,
        activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
        group_norm_eps: float = 1e-6,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        # TODO???
        # add_bias_kv=False,
        # add_zero_attn=False,
        # kdim=None,
        # vdim=None,
    ):
        if not batch_first:
            raise NotImplementedError("batch_first=False is not yet supported")
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.relative_position = relative_position
        self.bias = bias
        self.activation = activation

        if embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        head_dim = embed_dim // num_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )

        # The q/k/v projection layers are the same as in vanilla MHA.
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.group_norm = nn.GroupNorm(
            num_groups=num_heads,
            num_channels=embed_dim,
            affine=False,
            eps=group_norm_eps,
            device=device,
            dtype=dtype,
        )
        # The output project is slightly different, due to the gated "swish" layer.
        self.g_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        # 'thetas' parameter for updating the relative position embeddings.
        thetas: Optional[Tensor] = None
        if relative_position:
            thetas = _build_position_thetas(head_dim=head_dim, device=device, dtype=dtype)
        self.thetas: Optional[Tensor]
        self.register_buffer("thetas", thetas)

        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: Double-check that we're following the same initialization as in
        # the paper.  This is a generic initialization for MHA linear layers.
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_normal_(self.v_proj.weight)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)
        nn.init.xavier_normal_(self.g_proj.weight)
        if self.g_proj.bias is not None:
            nn.init.constant_(self.g_proj.bias, 0)

    def forward_parallel(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate retention heads.  Move the head
        # dimension to position 1 (makes matrix ops *much* faster).
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        if self.relative_position:
            assert self.thetas is not None
            indices = torch.arange(q.size(2), device=q.device, dtype=q.dtype)
            indices = rearrange(indices, "n -> () () n ()")
            thetas = rearrange(self.thetas, "d -> () () () d")
            angles = indices * thetas
            sin = torch.sin(angles)
            cos = torch.cos(angles)

            q = _theta_shift(q, sin, cos)
            k = _theta_shift(k, sin, cos)

        # Apply retention then group norm.
        retention, weights = retention_parallel(q, k, v, need_weights=need_weights)
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) (h d)")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) e -> b n e", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'query' can equivalently be used as the input.
        gate = self.activation(self.g_proj(query))
        retention = self.out_proj(retention * gate)

        return retention, weights

    def forward_recurrent(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        seq_idx: int,
        prev_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # einstein notation:
        # b - batch size
        # h - number of heads
        # d - embedding dimension
        #
        # input shape: (b, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate retention heads.
        q = rearrange(q, "b (h d) -> b h d", h=self.num_heads)
        k = rearrange(k, "b (h d) -> b h d", h=self.num_heads)
        v = rearrange(v, "b (h d) -> b h d", h=self.num_heads)

        if self.relative_position:
            assert self.thetas is not None
            thetas = rearrange(self.thetas, "d -> () () d")
            angles = seq_idx * thetas
            sin = torch.sin(angles)
            cos = torch.cos(angles)

            q = _theta_shift(q, sin, cos)
            k = _theta_shift(k, sin, cos)

        # Apply retention then group norm.
        retention, state = retention_recurrent(q, k, v, prev_state=prev_state)
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        # Fold heads back into the embedding dimension.
        retention = rearrange(retention, "b h d -> b (h d)")
        retention = self.group_norm(retention)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'query' can equivalently be used as the input.
        gate = self.activation(self.g_proj(query))
        retention = self.out_proj(retention * gate)

        return retention, state

    def forward_chunkwise(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        start_idx: int,
        prev_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate retention heads.  Move the head
        # dimension to position 1 (makes matrix ops *much* faster).
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        if self.relative_position:
            # global (cross-chunk) relative position embedding
            assert self.thetas is not None
            thetas = rearrange(self.thetas, "d -> () () () d")
            angles = start_idx * thetas
            sin = torch.sin(angles)
            cos = torch.cos(angles)
            q = _theta_shift(q, sin, cos)
            k = _theta_shift(k, sin, cos)

            # intra-chunk relative position encoding
            indices = torch.arange(q.size(2), device=q.device, dtype=q.dtype)
            indices = rearrange(indices, "n -> () () n ()")
            thetas = rearrange(self.thetas, "d -> () () () d")
            angles = indices * thetas
            sin = torch.sin(angles)
            cos = torch.cos(angles)
            q = _theta_shift(q, sin, cos)
            k = _theta_shift(k, sin, cos)

        # Apply retention then group norm.
        retention, state = retention_chunkwise(q, k, v, prev_state=prev_state)
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) (h d)")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) e -> b n e", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'query' can equivalently be used as the input.
        gate = self.activation(self.g_proj(query))
        retention = self.out_proj(retention * gate)

        return retention, state

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if query.requires_grad:
            y = checkpoint(
                self.forward_parallel, query, key, value, need_weights=need_weights
            )
        else:
            y = self.forward_parallel(query, key, value, need_weights=need_weights)
        return y


class RetNetDecoderLayer(nn.Module):
    # NOTE: Mostly pulled from 'nn.TransformerDecoderLayer', but with changes:
    #   - use MultiScaleRetention instead of MultiheadAttention
    #   - no cross-attention layer, since retention doesn't play well with that

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
        norm_first: bool = True,
        layer_norm_eps: float = 1e-6,
        relative_position: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> None:
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        print(f"during init RetNetDecoderLayer next parameters unused: {kwargs}")

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first
        self.relative_position = relative_position
        # retention block
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.retention = MultiScaleRetention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            activation=activation,
            relative_position=relative_position,
            device=device,
            dtype=dtype,
        )
        # feedforward block
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: Check that we're following the same initialization as the paper
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def _feedforward_block(self, x: Tensor) -> Tensor:
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

    def forward_parallel(self, x: Tensor) -> Tensor:
        def _retention_block(x: Tensor) -> Tensor:
            x, _ = self.retention.forward_parallel(x, x, x)
            return self.dropout(x)

        if self.norm_first:
            x = x + _retention_block(self.norm1(x))
            x = x + self._feedforward_block(self.norm2(x))
        else:
            x = x + self.norm1(_retention_block(x))
            x = x + self.norm2(self._feedforward_block(x))

        # poor_in_vram = True
        # if self.norm_first:
        #     if x.requires_grad and poor_in_vram:
        #         y = checkpoint(_retention_block, self.norm1(x))
        #     else:
        #         y = _retention_block(self.norm1(x))
        #     x = x + y
        #     x = x + self._feedforward_block(self.norm2(x))
        # else:
        #     if x.requires_grad and poor_in_vram:
        #         y = checkpoint(_retention_block, x)
        #     else:
        #         y = _retention_block(x)
        #     x = x + self.norm1(y)
        #     x = x + self.norm2(self._feedforward_block(x))

        return x

    def forward_recurrent(
        self, x: Tensor, seq_idx: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        def _retention_block(x: Tensor) -> Tuple[Tensor, Tensor]:
            x, state = self.retention.forward_recurrent(
                x, x, x, seq_idx=seq_idx, prev_state=prev_state
            )
            return self.dropout(x), state

        # retention block
        if self.norm_first:
            y, state = _retention_block(self.norm1(x))
            x = x + y
            x = x + self._feedforward_block(self.norm2(x))
        else:
            y, state = _retention_block(x)
            x = x + self.norm1(y)
            x = x + self.norm2(self._feedforward_block(x))

        return x, state

    def forward_chunkwise(
        self, x: Tensor, start_idx: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        def _retention_block(x: Tensor) -> Tuple[Tensor, Tensor]:
            x, state = self.retention.forward_chunkwise(
                x, x, x, start_idx=start_idx, prev_state=prev_state
            )
            return self.dropout(x), state

        # retention block
        if self.norm_first:
            y, state = _retention_block(self.norm1(x))
            x = x + y
            x = x + self._feedforward_block(self.norm2(x))
        else:
            y, state = _retention_block(x)
            x = x + self.norm1(y)
            x = x + self.norm2(self._feedforward_block(x))

        return x, state

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_parallel(x)


class RetNetDecoder(nn.Module):
    def __init__(self, decoder_layer: RetNetDecoderLayer, num_layers: int, **kwargs):
        super().__init__()
        print(f"during init RetNetDecoderLayer next parameters unused: {kwargs}")

        self.num_layers = num_layers
        self.layers = nn.ModuleList([deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward_parallel(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            assert isinstance(layer, RetNetDecoderLayer)
            x = layer.forward_parallel(x)
        return x

    def forward_recurrent(
        self, x: Tensor, seq_idx: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        if not prev_states:
            prev_states = [None] * self.num_layers
        elif len(prev_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(prev_states)}"
            )

        states: List[Tensor] = []
        for layer, prev_state in zip(self.layers, prev_states):
            assert isinstance(layer, RetNetDecoderLayer)
            x, state = layer.forward_recurrent(x, seq_idx, prev_state)
            states.append(state)
        return x, states

    def forward_chunkwise(
        self, x: Tensor, start_idx: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        if not prev_states:
            prev_states = [None] * self.num_layers
        elif len(prev_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(prev_states)}"
            )

        states: List[Tensor] = []
        for layer, prev_state in zip(self.layers, prev_states):
            assert isinstance(layer, RetNetDecoderLayer)
            x, state = layer.forward_chunkwise(x, start_idx, prev_state)
            states.append(state)
        return x, states

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_parallel(x)
