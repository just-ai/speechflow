import typing as tp

import torch

from pydantic import Field
from torch import nn
from torch.nn import functional as F

from tts.acoustic_models.modules.common import (
    InverseGrad1DPredictor,
    InverseGradPhonemePredictor,
    InverseGradSpeakerIDPredictor,
    InverseGradSpeakerPredictor,
    InverseGradStylePredictor,
    Regression,
)
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, ComponentOutput
from tts.acoustic_models.modules.params import EmbeddingParams

__all__ = ["AdditionalModules", "AdditionalModulesParams"]


class AdditionalModulesParams(EmbeddingParams):
    addm_apply_inverse_speaker_classifier: tp.Dict[str, tp.Optional[int]] = Field(
        default_factory=lambda: {}
    )
    addm_apply_inverse_speaker_emb: tp.Dict[str, int] = Field(default_factory=lambda: {})
    addm_apply_inverse_style_emb: tp.Dict[str, int] = Field(default_factory=lambda: {})
    addm_apply_inverse_1d_feature: tp.Dict[str, int] = Field(default_factory=lambda: {})
    addm_apply_inverse_phoneme_classifier: tp.Dict[str, int] = Field(
        default_factory=lambda: {}
    )
    addm_apply_phoneme_classifier: tp.Dict[str, int] = Field(default_factory=lambda: {})

    addm_1d_features: tp.List[str] = ["energy", "pitch"]

    addm_style_emb_name: str = "style_emb"
    addm_style_emb_dim: tp.Optional[int] = None

    addm_disable: bool = False


class AdditionalModules(Component):
    params: AdditionalModulesParams

    def __init__(self, params: AdditionalModulesParams, input_dim=None):
        super().__init__(params, input_dim)

        if params.addm_apply_inverse_speaker_classifier:
            self.inverse_speaker_classifier = nn.ModuleDict()
            for name, emb_size in params.addm_apply_inverse_speaker_classifier.items():
                if emb_size is None:
                    emb_size = self.components_output_dim[name]()
                self.inverse_speaker_classifier[name] = InverseGradSpeakerIDPredictor(
                    input_dim=emb_size, n_speakers=params.n_speakers
                )

        if params.addm_apply_inverse_speaker_emb:
            self.inverse_speaker_emb = nn.ModuleDict()
            for name, emb_size in params.addm_apply_inverse_speaker_emb.items():
                if emb_size is None:
                    emb_size = self.components_output_dim[name]()
                self.inverse_speaker_emb[name] = InverseGradSpeakerPredictor(
                    input_dim=emb_size, target_dim=params.speaker_emb_dim
                )

        if params.addm_apply_inverse_style_emb:
            self.inverse_style_embedding = nn.ModuleDict()
            for name, emb_size in params.addm_apply_inverse_style_emb.items():
                if emb_size is None:
                    emb_size = self.components_output_dim[name]()
                self.inverse_style_embedding[name] = InverseGradStylePredictor(
                    input_dim=emb_size, target_dim=params.addm_style_emb_dim
                )

        if params.addm_apply_inverse_1d_feature:
            self.inverse_1d_feature = nn.ModuleDict()
            for feat in self.params.addm_1d_features:
                self.inverse_1d_feature[feat] = nn.ModuleDict()
                for name, emb_size in params.addm_apply_inverse_1d_feature.items():
                    if emb_size is None:
                        emb_size = self.components_output_dim[name]()
                    self.inverse_1d_feature[feat][name] = InverseGrad1DPredictor(
                        input_dim=emb_size
                    )

        if params.addm_apply_inverse_phoneme_classifier:
            self.inverse_phoneme_proj = nn.ModuleDict()
            for name, emb_size in params.addm_apply_inverse_phoneme_classifier.items():
                if emb_size is None:
                    emb_size = self.components_output_dim[name]()
                self.inverse_phoneme_proj[name] = nn.ModuleList()
                for _ in range(params.n_symbols_per_token):
                    self.inverse_phoneme_proj[name].append(
                        InverseGradPhonemePredictor(emb_size, params.alphabet_size)
                    )

        if params.addm_apply_phoneme_classifier:
            self.phoneme_proj = nn.ModuleDict()
            for name, emb_size in params.addm_apply_phoneme_classifier.items():
                if emb_size is None:
                    emb_size = self.components_output_dim[name]()
                self.phoneme_proj[name] = nn.ModuleList()
                for _ in range(params.n_symbols_per_token):
                    self.phoneme_proj[name].append(
                        Regression(emb_size, params.alphabet_size)
                    )

    @property
    def output_dim(self):
        return None

    def forward_step(self, inputs: ComponentInput) -> ComponentOutput:  # type: ignore
        if not self.training or self.params.addm_disable:
            return inputs

        content = inputs.additional_content
        losses = inputs.additional_losses

        def get_emb(_name: str):
            _emb = content.get(_name)
            # if isinstance(_emb, torch.Tensor):
            #     assert _emb.requires_grad
            return _emb

        def eval_classifier(_module, _emb, _target):
            if isinstance(_emb, torch.Tensor):
                _emb = [_emb]

            _loss = 0
            for _t in _emb:
                if _t.shape[0] > 1:
                    _loss += _module(_t, _target)

            return _loss

        if (
            self.params.addm_apply_inverse_speaker_classifier
            and self.params.n_speakers > 1
        ):
            for name, module in self.inverse_speaker_classifier.items():
                loss = eval_classifier(
                    module, get_emb(name), inputs.model_inputs.speaker_id
                )
                if loss > 0:
                    losses[f"{name}_inverse_speaker_classifier"] = 0.1 * loss

        if self.params.addm_apply_inverse_speaker_emb:
            for name, module in self.inverse_speaker_emb.items():
                loss = eval_classifier(
                    module, get_emb(name), inputs.model_inputs.speaker_emb
                )
                if loss > 0:
                    losses[f"{name}_inverse_speaker_emb"] = 0.1 * loss

        if self.params.addm_apply_inverse_style_emb:
            style = inputs.additional_content[self.params.addm_style_emb_name]
            for name, module in self.inverse_style_embedding.items():
                if name == self.params.addm_style_emb_name:
                    continue
                loss = eval_classifier(module, get_emb(name), style.detach().squeeze(1))
                if loss > 0:
                    losses[f"{name}_inverse_style_embedding"] = 0.1 * loss

        if self.params.addm_apply_inverse_1d_feature:
            for feat_name, module_dict in self.inverse_1d_feature.items():
                feat = getattr(inputs.model_inputs, feat_name)
                for name, module in module_dict.items():
                    if name == self.params.addm_style_emb_name:
                        continue
                    loss = eval_classifier(module, get_emb(name), feat.detach())
                    if loss > 0:
                        losses[f"{feat_name}_{name}_inverse_1d_feature"] = 0.1 * loss

        if self.params.addm_apply_inverse_phoneme_classifier:
            for name, module in self.inverse_phoneme_proj.items():
                loss = 0
                for idx, m in enumerate(module):
                    emb = get_emb(name)
                    if isinstance(emb, torch.Tensor):
                        emb = [emb]
                    for t in emb:
                        target = inputs.model_inputs.transcription
                        if t.shape[1] != target.shape[1]:
                            target = inputs.model_inputs.transcription_by_frames
                        if target.ndim == 2:
                            target = target.unsqueeze(-1)
                        loss += m(t, target[:, :, idx])

                losses[f"{name}_inverse_phoneme_classifier"] = 0.1 * loss

        if self.params.addm_apply_phoneme_classifier:
            for name, module in self.phoneme_proj.items():
                loss = 0
                for idx, m in enumerate(module):
                    emb = get_emb(name)
                    if isinstance(emb, torch.Tensor):
                        emb = [emb]
                    for t in emb:
                        predict = F.log_softmax(m(t), dim=2)
                        target = inputs.model_inputs.transcription
                        if predict.shape[1] != target.shape[1]:
                            target = inputs.model_inputs.transcription_by_frames
                        if target.ndim == 2:
                            target = target.unsqueeze(-1)
                        loss += F.nll_loss(predict.transpose(1, -1), target[:, :, idx])

                losses[f"{name}_phoneme_classifier"] = 0.1 * loss

        return inputs

    def inference_step(self, inputs: ComponentInput, **kwargs) -> ComponentOutput:  # type: ignore
        return inputs
