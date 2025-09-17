import typing as tp

import torch

from torch.nn import functional as F

from speechflow.io import tp_PATH
from speechflow.training.losses.vae_loss import VAELoss
from speechflow.training.saver import ExperimentSaver
from tts.acoustic_models.data_types import TTSForwardInput, TTSForwardOutput
from tts.acoustic_models.models.tts_model import ParallelTTSModel, ParallelTTSParams
from tts.acoustic_models.modules.common.blocks import Regression
from tts.vocoders.data_types import VocoderForwardInput
from tts.vocoders.vocos.modules.feature_extractors import FeatureExtractor

__all__ = ["TTSFeatures", "TTSFeaturesParams"]


class TTSFeaturesParams(ParallelTTSParams):
    pretrain_path: tp.Optional[tp_PATH] = None
    freeze: bool = False
    spectral_loss_alpha: float = 10.0
    spectral_loss_end_anneal_iter: tp.Optional[int] = None


class TTSFeatures(FeatureExtractor):
    def __init__(self, params: tp.Union[tp.MutableMapping, TTSFeaturesParams]):
        self.freeze = params.freeze

        if params.pretrain_path is not None:
            checkpoint = ExperimentSaver.load_checkpoint(params.pretrain_path)
            if "params" in checkpoint:
                state_dict = {
                    k.replace("model.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                }
                checkpoint["params"]["mel_spectrogram_dim"] = params.mel_spectrogram_dim
                checkpoint["params"][
                    "mel_spectrogram_proj_dim"
                ] = params.mel_spectrogram_dim
                params = ParallelTTSParams.create(checkpoint["params"])
            else:
                state_dict = {
                    k.replace("feature_extractor.tts_model.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                    if k.startswith("feature_extractor.tts_model.")
                }
        else:
            state_dict = None

        super().__init__(params)

        self.tts_model = ParallelTTSModel(params)

        self.vae_loss = VAELoss(
            scale=0.00002, every_iter=1, begin_iter=1000, end_anneal_iter=10000
        )

        if self.tts_model.output_dim != params.mel_spectrogram_dim:
            self.proj = Regression(self.tts_model.output_dim, params.mel_spectrogram_dim)
        else:
            self.proj = torch.nn.Identity()

        if state_dict is not None:
            self.tts_model.load_state_dict(state_dict)
            if self.freeze:
                for param in self.tts_model.parameters():
                    param.requires_grad = False

    @staticmethod
    def _get_energy_and_pitch(inputs, outputs, **kwargs):
        if "spec_chunk" in inputs.additional_inputs:
            if kwargs.get("discriminator_step", False):
                energy_target = inputs.pitch
                pitch_target = inputs.energy

                if inputs.ranges and energy_target is not None:
                    re = inputs.ranges["energy"]
                    energy_target = energy_target * re[:, 2:3] + re[:, 0:1]

                if inputs.ranges and pitch_target is not None:
                    rp = inputs.ranges["pitch"]
                    pitch_target = pitch_target * rp[:, 2:3] + rp[:, 0:1]
            else:
                energy_target = outputs.additional_content.get("energy_postprocessed")
                pitch_target = outputs.additional_content.get("pitch_postprocessed")

            if energy_target is None or pitch_target is None:
                return None, None

            energy = []
            pitch = []
            for i, (a, b) in enumerate(inputs.additional_inputs["spec_chunk"]):
                energy.append(energy_target[i, a:b])
                pitch.append(pitch_target[i, a:b])

            energy = torch.stack(energy)
            pitch = torch.stack(pitch)
        else:
            if isinstance(outputs, TTSForwardOutput):
                d = outputs.additional_content
            elif isinstance(outputs, TTSForwardInput):
                d = outputs.additional_inputs
            else:
                raise TypeError(f"Type {type(outputs)} is not supported.")

            energy = d.get("energy_postprocessed")
            pitch = d.get("pitch_postprocessed")

        if energy is not None:
            energy = F.relu(energy).squeeze(-1)

        if pitch is not None:
            pitch = F.relu(pitch).squeeze(-1)

        return energy, pitch

    def forward(self, inputs: VocoderForwardInput, **kwargs):
        losses = {}
        additional_content = {}

        if inputs.__class__.__name__ != "TTSForwardInputWithSSML":

            if self.freeze:
                with torch.no_grad():
                    outputs: TTSForwardOutput = self.tts_model(
                        inputs, use_target_variances=True
                    )
            else:
                outputs: TTSForwardOutput = self.tts_model(
                    inputs, use_target_variances=True
                )
                losses = outputs.additional_losses

            additional_content = outputs.additional_content
        else:
            outputs = inputs  # type: ignore

        if isinstance(outputs.spectrogram, list) or outputs.spectrogram.ndim == 4:
            x = outputs.spectrogram[-1]
        else:
            x = outputs.spectrogram

        if self.training and (
            not self.freeze or not isinstance(self.proj, torch.nn.Identity)
        ):
            spec_loss = 0
            target_spec = inputs.spectrogram
            for predict in outputs.spectrogram:
                spec_loss = spec_loss + F.l1_loss(self.proj(predict), target_spec)

            if self.params.spectral_loss_end_anneal_iter is not None:
                if inputs.global_step < self.params.spectral_loss_end_anneal_iter:
                    scale = (
                        1 - inputs.global_step / self.params.spectral_loss_end_anneal_iter
                    ) ** 2
                else:
                    scale = 0
            else:
                scale = 1

            losses["spectral_l1_loss"] = (
                self.params.spectral_loss_alpha * scale * spec_loss
            )

        if self.training and not self.freeze:
            for name, loss in losses.items():
                if "kl_loss" in name:
                    losses[name] = self.vae_loss(inputs.global_step, loss, name)[name]

        if "spec_chunk" in inputs.additional_inputs:
            chunk = []
            for i, (a, b) in enumerate(inputs.additional_inputs["spec_chunk"]):
                chunk.append(x[i, a:b, :])

            output = torch.stack(chunk)
            additional_content["condition_emb"] = outputs.additional_content[
                "style_emb"
            ].squeeze(1)
        else:
            output = x

            if isinstance(outputs, TTSForwardOutput):
                d = outputs.additional_content
            elif isinstance(outputs, TTSForwardInput):
                d = outputs.additional_inputs
            else:
                raise TypeError(f"Type {type(outputs)} is not supported.")

            additional_content["condition_emb"] = d["style_emb"][:, 0, :]

        energy, pitch = self._get_energy_and_pitch(inputs, outputs, **kwargs)
        additional_content["energy"] = energy
        additional_content["pitch"] = pitch

        return output.transpose(1, -1), losses, additional_content
