from torch import nn

from tts.acoustic_models.modules.common.gpts.layers.layers import (
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class GPTDecoder(nn.Module):
    def __init__(
        self, dim_hidden: int, n_heads: int, n_layers: int, is_norm_first: bool, **kwargs
    ):
        super().__init__()
        self.model = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=dim_hidden,
                nhead=n_heads,
                dim_feedforward=dim_hidden * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=is_norm_first,
            ),
            num_layers=n_layers,
            norm=LayerNorm(dim_hidden) if is_norm_first else None,
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
