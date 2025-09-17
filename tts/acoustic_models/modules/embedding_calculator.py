import typing as tp

import torch
import torch.nn.functional as F

from torch import nn

from speechflow.data_pipeline.datasample_processors.biometric_processors import (
    VoiceBiometricProcessor,
)
from speechflow.training.base_model import BaseTorchModel
from tts.acoustic_models.data_types import TTSForwardInput
from tts.acoustic_models.modules.common import VarianceEmbedding
from tts.acoustic_models.modules.params import EmbeddingParams

__all__ = ["EmbeddingCalculator"]


class EmbeddingCalculator(BaseTorchModel):
    params: EmbeddingParams

    def __init__(self, params: EmbeddingParams):
        super().__init__(params)

        def _get_projection_layer(
            input_dim: int,
            output_dim: int,
            bias: bool = True,
            activation_fn: tp.Optional[str] = "Tanh",
        ):
            af = nn.Identity() if activation_fn is None else getattr(nn, activation_fn)()
            return nn.Sequential(nn.Linear(input_dim, output_dim, bias=bias), af)

        self.embedding = nn.Embedding(params.alphabet_size, params.token_emb_dim)
        # nn.init.normal_(self.embedding.weight, 0.0, params.token_emb_dim**-0.5)
        nn.init.orthogonal_(self.embedding.weight)

        if params.n_symbols_per_token > 1:
            self.token_proj = _get_projection_layer(
                params.token_emb_dim * params.n_symbols_per_token, params.token_emb_dim
            )

        if params.n_langs > 1:
            self.lang_embedding = nn.Embedding(params.n_langs, params.token_emb_dim)
            # nn.init.normal_(self.lang_embedding.weight, 0.0, params.token_emb_dim**-0.5)
            nn.init.orthogonal_(self.lang_embedding.weight)
        else:
            self.lang_embedding = None

        self.use_speaker_emb = True
        if params.use_onehot_speaker_emb:
            self.n_speakers = params.n_speakers
            one_hot = F.one_hot(torch.arange(0, self.n_speakers), self.n_speakers)
            self.register_buffer("speaker_emb", one_hot.float())
            params.speaker_emb_dim = self.n_speakers
            self.speaker_emb_dim = params.speaker_emb_dim
        elif params.use_learnable_speaker_emb:
            self.n_speakers = params.n_speakers
            self.speaker_emb = nn.Embedding(
                num_embeddings=self.n_speakers,
                embedding_dim=params.speaker_emb_dim,
            )
            # nn.init.normal_(self.speaker_emb.weight, 0.0, params.speaker_emb_dim ** -0.5)
            nn.init.orthogonal_(self.speaker_emb.weight)
            self.speaker_emb_dim = params.speaker_emb_dim
        elif params.use_dnn_speaker_emb or params.use_mean_dnn_speaker_emb:
            self.n_speakers = None  # type: ignore
            self.speaker_emb = None
            self.speaker_emb_dim = params.speaker_emb_dim
        else:
            self.n_speakers = 1
            self.speaker_emb_dim = 0
            self.speaker_emb = None
            self.use_speaker_emb = False

        if self.params.use_dnn_speaker_emb:
            bio_processor = VoiceBiometricProcessor(
                model_type=params.speaker_biometric_model
            )
            if self.speaker_emb_dim != bio_processor.embedding_dim:
                self.speaker_emb_proj = _get_projection_layer(
                    bio_processor.embedding_dim, self.speaker_emb_dim
                )

        if self.params.use_mean_dnn_speaker_emb:
            bio_processor = VoiceBiometricProcessor(
                model_type=params.speaker_biometric_model
            )
            if self.speaker_emb_dim != bio_processor.embedding_dim:
                self.speaker_emb_proj = _get_projection_layer(
                    bio_processor.embedding_dim, self.speaker_emb_dim
                )

        if params.num_additional_integer_seqs > 0:
            num_proj = params.num_additional_integer_seqs
            proj_size = (
                params.token_emb_dim - params.num_additional_float_seqs
            ) // params.num_additional_integer_seqs

            self.ling_feat_proj = nn.ModuleList()
            for _ in range(num_proj):
                proj = _get_projection_layer(params.token_emb_dim, proj_size)
                self.ling_feat_proj.append(proj)
        else:
            self.ling_feat_proj = None

        self.proj_layers = nn.ModuleDict()
        for feat_name in [
            "xpbert_feat",
            "lm_feat",
            "linear_spectrogram",
            "mel_spectrogram",
            "ssl_feat",
            "ac_feat",
        ]:
            feat_dim = getattr(params, f"{feat_name}_dim")
            proj_dim = getattr(params, f"{feat_name}_proj_dim")
            if feat_dim != proj_dim:
                self.proj_layers[feat_name] = _get_projection_layer(
                    feat_dim, proj_dim, False, None
                )

        if params.use_average_emb:
            self.averages = nn.ModuleDict()
            for name, vales in params.averages.items():
                self.averages[name] = VarianceEmbedding(**vales)

    def get_transcription_embeddings(
        self, inputs: TTSForwardInput
    ) -> tp.Optional[torch.Tensor]:
        x = inputs.transcription
        if x is not None:
            if self.params.n_symbols_per_token == 1:
                x = self.embedding(x)
            else:
                temp = []
                for i in range(self.params.n_symbols_per_token):
                    temp.append(self.embedding(x[:, :, i]))
                x = self.token_proj(torch.cat(temp, dim=-1))

        return x

    def get_lang_embedding(self, inputs: TTSForwardInput) -> tp.Optional[torch.Tensor]:
        if not hasattr(inputs, "lang_id") or inputs.lang_id is None:
            return None

        if self.lang_embedding is not None:
            return self.lang_embedding(inputs.lang_id)

    def get_speaker_embedding(self, inputs: TTSForwardInput) -> tp.Optional[torch.Tensor]:
        if getattr(inputs, "additional_inputs", None) is not None:
            if inputs.additional_inputs.get("speaker") is not None:  # type: ignore
                return inputs.additional_inputs.get("speaker")  # type: ignore

        speaker_ids = inputs.speaker_id

        if self.params.use_onehot_speaker_emb:
            speaker_emb = self.speaker_emb[speaker_ids]
        elif self.params.use_learnable_speaker_emb:
            speaker_emb = self.speaker_emb(speaker_ids)
        elif self.params.use_dnn_speaker_emb:
            speaker_emb = inputs.speaker_emb
        elif self.params.use_mean_dnn_speaker_emb:
            speaker_emb = inputs.speaker_emb_mean
        else:
            speaker_emb = None

        if hasattr(self, "speaker_emb_proj"):
            speaker_emb = self.speaker_emb_proj(speaker_emb).squeeze(1)

        return speaker_emb

    def get_speaker_biometric_embedding(
        self, inputs: TTSForwardInput
    ) -> tp.Optional[torch.Tensor]:
        if getattr(inputs, "additional_inputs", None) is not None:
            if inputs.additional_inputs.get("speaker_bio_emb") is not None:  # type: ignore
                return inputs.additional_inputs.get("speaker_bio_emb")  # type: ignore

        biometric_embedding = inputs.speaker_emb

        if biometric_embedding is not None and hasattr(self, "speaker_emb_proj"):
            biometric_embedding = self.speaker_emb_proj(biometric_embedding).squeeze(1)

        return biometric_embedding

    def get_ling_feat(self, inputs: TTSForwardInput) -> tp.Optional[torch.Tensor]:
        if inputs.ling_feat is None:
            return None

        proj_idx = 0
        assembled_ling_emb = []
        for feat in inputs.ling_feat.to_dict().values():
            if feat is not None:
                if (
                    feat.dtype == torch.int64
                    and self.params.num_additional_integer_seqs > 0
                ):
                    feat = self.embedding(feat)
                    feat = self.ling_feat_proj[proj_idx](feat)
                    assembled_ling_emb.append(feat)
                    proj_idx += 1
                elif self.params.num_additional_float_seqs > 0:
                    assembled_ling_emb.append(feat.unsqueeze(-1))

        if not assembled_ling_emb:
            return None

        assembled_ling_emb = torch.cat(assembled_ling_emb, dim=2)

        pad = self.params.token_emb_dim - assembled_ling_emb.shape[-1]
        if pad:
            assembled_ling_emb = F.pad(assembled_ling_emb, (0, pad), "constant", 0)

        return assembled_ling_emb

    def _get_features(
        self, inputs: TTSForwardInput, feat_name: str
    ) -> tp.Optional[torch.Tensor]:
        if not hasattr(inputs, feat_name):
            return None

        feat = getattr(inputs, feat_name)

        if feat_name not in self.proj_layers or feat is None:
            return feat
        else:
            return self.proj_layers[feat_name](feat)

    def get_xpbert_feat(self, inputs: TTSForwardInput) -> tp.Optional[torch.Tensor]:
        return self._get_features(inputs, "xpbert_feat")

    def get_lm_feat(self, inputs: TTSForwardInput) -> tp.Optional[torch.Tensor]:
        return self._get_features(inputs, "lm_feat")

    def get_linear_spectrogram(
        self, inputs: TTSForwardInput
    ) -> tp.Optional[torch.Tensor]:
        return self._get_features(inputs, "linear_spectrogram")

    def get_mel_spectrogram(self, inputs: TTSForwardInput) -> tp.Optional[torch.Tensor]:
        return self._get_features(inputs, "mel_spectrogram")

    def get_ssl_feat(self, inputs: TTSForwardInput) -> tp.Optional[torch.Tensor]:
        return self._get_features(inputs, "ssl_feat")

    def get_ac_feat(self, inputs: TTSForwardInput) -> tp.Optional[torch.Tensor]:
        return self._get_features(inputs, "ac_feat")

    def get_averages(
        self, inputs: TTSForwardInput
    ) -> tp.Optional[tp.Dict[str, torch.Tensor]]:
        if not self.params.use_average_emb:
            return None

        assembled_average_emb = {}

        if getattr(inputs, "additional_inputs", None) is not None:
            if inputs.additional_inputs.get("average") is not None:  # type: ignore
                # TODO: костыль для поддержки изменения громкости, переделать
                # if getattr(inputs, "volume_modifier", None) is not None:
                #     volume = getattr(inputs, "volume_modifier") * inputs.averages["energy"]
                #     return self.averages["energy"](volume)
                # else:
                for name, embedding in self.averages.items():
                    assembled_average_emb[name] = inputs.additional_inputs.get(
                        f"average_{name}"
                    )

                return assembled_average_emb

        for name, embedding in self.averages.items():
            emb = embedding(inputs.averages[name])
            assembled_average_emb[name] = emb.squeeze(1)

        return assembled_average_emb

    @staticmethod
    def get_speech_quality_embedding(
        inputs: TTSForwardInput,
    ) -> tp.Optional[torch.Tensor]:
        if not hasattr(inputs, "speech_quality_emb"):
            return None

        if getattr(inputs, "additional_inputs", None) is not None:
            if inputs.additional_inputs.get("sq") is not None:  # type: ignore
                return inputs.additional_inputs.get("sq")  # type: ignore

        return inputs.speech_quality_emb
