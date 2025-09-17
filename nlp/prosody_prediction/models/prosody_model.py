import re
import typing as tp

import torch

from transformers import AutoModel

from nlp.prosody_prediction.data_types import (
    ProsodyPredictionInput,
    ProsodyPredictionOutput,
)
from tts.acoustic_models.modules.embedding_calculator import EmbeddingCalculator
from tts.acoustic_models.modules.params import EmbeddingParams


class ProsodyPredictionParams(EmbeddingParams):
    lm_model_name: str = "google-bert/bert-base-multilingual-cased"
    dropout: float = 0.5
    n_classes: int = 10
    n_layers_tune: tp.Optional[int] = None
    classification_task: tp.Literal["both", "binary"] = "both"


class ProsodyModel(EmbeddingCalculator):
    params: ProsodyPredictionParams

    def __init__(
        self,
        cfg: tp.Union[tp.MutableMapping, ProsodyPredictionParams],
        strict_init: bool = True,
    ):
        super().__init__(ProsodyPredictionParams.create(cfg, strict_init))
        params = self.params

        self.bert = AutoModel.from_pretrained(
            params.lm_model_name, add_pooling_layer=False
        )
        self.predictors = torch.nn.ModuleDict()
        self.latent_dim = self.bert.config.hidden_size

        if params.classification_task in ["both", "binary"]:
            self.predictors["binary"] = torch.nn.Sequential(
                torch.nn.Dropout(p=params.dropout),
                torch.nn.Linear(self.latent_dim, self.latent_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=params.dropout),
                torch.nn.Linear(self.latent_dim, 2),
            )
        if params.classification_task in ["both", "category"]:
            self.predictors["category"] = torch.nn.Sequential(
                torch.nn.Dropout(p=params.dropout),
                torch.nn.Linear(self.latent_dim, self.latent_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=params.dropout),
                torch.nn.Linear(self.latent_dim, params.n_classes),
            )

        if (
            params.n_layers_tune is not None
            and self.bert.config.num_hidden_layers > params.n_layers_tune
        ):
            layers_tune = "|".join(
                [
                    str(self.bert.config.num_hidden_layers - i)
                    for i in range(1, params.n_layers_tune)
                ]
            )
            for name, param in self.bert.named_parameters():
                if not re.search(f"pooler|drop|{layers_tune}", name):
                    param.requires_grad = False

    def forward(self, inputs: ProsodyPredictionInput) -> ProsodyPredictionOutput:
        hidden = self.bert(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
        )[0]

        outputs = {"binary": None, "category": None}
        for name in self.predictors:
            outputs[name] = self.predictors[name](hidden)

        output = ProsodyPredictionOutput(
            binary=outputs["binary"],
            category=outputs["category"],
        )
        return output
