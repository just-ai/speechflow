import typing as tp

import torch

from tts.acoustic_models.modules.common import (
    CONDITIONAL_TYPES,
    ConditionalLayer,
    SoftLengthRegulator,
)
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, ComponentOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = ["LinguisticCondition", "LinguisticConditionParams"]


class LinguisticConditionParams(EncoderParams):
    # linguistic condition
    ling_condition_type: tp.Optional[CONDITIONAL_TYPES] = None

    # language model condition
    lm_condition_type: tp.Optional[CONDITIONAL_TYPES] = None

    # prosodic model condition
    xpbert_condition_type: tp.Optional[CONDITIONAL_TYPES] = None

    p_dropout: float = 0.1


class LinguisticCondition(Component):
    params: LinguisticConditionParams

    def __init__(self, params: LinguisticConditionParams, input_dim):
        super().__init__(params, input_dim)

        self.ling_cond_layer = ConditionalLayer(
            params.ling_condition_type, input_dim, self.params.token_emb_dim
        )
        self.lm_cond_layer = ConditionalLayer(
            params.lm_condition_type,
            self.ling_cond_layer.output_dim,
            self.params.lm_feat_proj_dim,
        )
        self.xpbert_cond_layer = ConditionalLayer(
            params.xpbert_condition_type,
            self.lm_cond_layer.output_dim,
            self.params.xpbert_feat_proj_dim,
        )

        self.hard_lr = SoftLengthRegulator(hard=True)
        self.seq_dropout = torch.nn.Dropout1d(params.p_dropout)

    @property
    def output_dim(self):
        return self.lm_cond_layer.output_dim

    def forward_step(self, inputs: ComponentInput) -> ComponentOutput:
        x, x_lens, x_mask = inputs.get_content_and_mask()

        if self.params.ling_condition_type is not None:
            ling_feat = inputs.embeddings["ling_feat"]
            ling_feat = self.seq_dropout(ling_feat)
            x = self.ling_cond_layer(x, ling_feat, x_mask)

        if self.params.lm_condition_type is not None:
            lm_feat = inputs.embeddings["lm_feat"]
            if lm_feat.shape[1] != x.shape[1]:
                token_length = inputs.model_inputs.additional_inputs["word_lengths"]
                lm_feat, _ = self.hard_lr(lm_feat, token_length, x.shape[1])

            lm_feat = self.seq_dropout(lm_feat)
            x = self.lm_cond_layer(x, lm_feat, x_mask)

        if self.params.xpbert_condition_type is not None:
            xpbert_feat = inputs.embeddings["xpbert_feat"]
            xpbert_feat = self.seq_dropout(xpbert_feat)
            x = self.xpbert_cond_layer(x, xpbert_feat, x_mask)

        return ComponentOutput.copy_from(inputs).set_content(x)
