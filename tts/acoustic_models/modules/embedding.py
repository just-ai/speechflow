import typing as tp

import torch

from tts.acoustic_models.data_types import TTSForwardInput
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentOutput
from tts.acoustic_models.modules.embedding_calculator import EmbeddingCalculator
from tts.acoustic_models.modules.params import EmbeddingParams

__all__ = ["EmbeddingComponent"]


class EmbeddingComponent(Component):
    params: EmbeddingParams

    def __init__(
        self,
        params: EmbeddingParams,
        input_dim: tp.Optional[int] = None,
    ):
        super().__init__(params, input_dim)

        self.emb_calculator = EmbeddingCalculator(params)  # type: ignore

    def _get_dim(self, feat_name: str) -> int:
        if feat_name == "transcription":
            return self.params.token_emb_dim
        elif feat_name == "xpbert_feat":
            return self.params.xpbert_feat_proj_dim
        elif feat_name == "spectrogram":
            return self.params.spectrogram_proj_dim
        elif feat_name == "ssl_feat":
            return self.params.ssl_feat_proj_dim
        elif feat_name == "ac_feat":
            return self.params.ac_feat_proj_dim
        else:
            raise NotImplementedError(
                f"input_embedding '{self.params.input}' is not support."
            )

    @property
    def output_dim(self):
        if len(self.params.input) == 1:
            return self._get_dim(self.params.input[0])
        else:
            return [self._get_dim(name) for name in self.params.input]

    def forward_step(self, inputs: TTSForwardInput) -> ComponentOutput:  # type: ignore
        transcription = self.emb_calculator.get_transcription_embeddings(inputs)
        xpbert_feat = self.emb_calculator.get_xpbert_feat(inputs)
        lm_feat = self.emb_calculator.get_lm_feat(inputs)
        linear_spectrogram = self.emb_calculator.get_linear_spectrogram(inputs)
        mel_spectrogram = self.emb_calculator.get_mel_spectrogram(inputs)
        ssl_feat = self.emb_calculator.get_ssl_feat(inputs)
        ac_feat = self.emb_calculator.get_ac_feat(inputs)

        ling_feat = self.emb_calculator.get_ling_feat(inputs)
        lang_emb = self.emb_calculator.get_lang_embedding(inputs)
        speaker_emb = self.emb_calculator.get_speaker_embedding(inputs)
        biometric_emb = self.emb_calculator.get_speaker_biometric_embedding(inputs)
        averages = self.emb_calculator.get_averages(inputs)
        sq_emb = self.emb_calculator.get_speech_quality_embedding(inputs)

        content = []
        content_lengths = []
        for name in self.params.input:
            if name == "transcription":
                x = transcription
                x_lengths = inputs.transcription_lengths

                if lang_emb is not None:
                    x = x + lang_emb.unsqueeze(1).expand(-1, x.shape[1], -1)

            elif name == "xpbert_feat":
                x = xpbert_feat
                x_lengths = inputs.xpbert_feat_lengths

            elif name == "lm_feat":
                x = lm_feat
                x_lengths = inputs.transcription_lengths

            elif "spectrogram" in name:
                x = mel_spectrogram if "mel" in name else linear_spectrogram
                x_lengths = inputs.spectrogram_lengths

            elif name == "ssl_feat":
                x = ssl_feat
                x_lengths = inputs.ssl_feat_lengths

            elif name == "ac_feat":
                x = ac_feat
                x_lengths = inputs.ac_feat_lengths

            else:
                x = None
                x_lengths = None

            content.append(x)
            content_lengths.append(x_lengths)

        embeddings = dict(
            transcription=transcription,
            ling_feat=ling_feat,
            xpbert_feat=xpbert_feat,
            lm_feat=lm_feat,
            lang_emb=lang_emb,
            speaker_emb=speaker_emb,
            biometric_emb=biometric_emb,
            speech_quality_emb=sq_emb,
            linear_spectrogram=linear_spectrogram,
            mel_spectrogram=mel_spectrogram,
            ssl_feat=ssl_feat,
            ac_feat=ac_feat,
        )

        if averages is not None:
            embeddings["average_emb"] = torch.cat(list(averages.values()), dim=1)
            embeddings.update({f"average_{k}": v for k, v in averages.items()})

        embeddings = {k: v for k, v in embeddings.items() if v is not None}
        inputs.additional_inputs.update(embeddings)

        return ComponentOutput(
            content=content,
            content_lengths=content_lengths,
            embeddings=embeddings,
            model_inputs=inputs,
        )
