import sys
import typing as tp

from torch import nn

from speechflow.io import Config
from speechflow.training.base_criterion import BaseCriterion
from speechflow.training.losses.attention import *
from speechflow.training.losses.loss1d import *
from speechflow.training.losses.spectral import *
from speechflow.training.losses.vae_loss import *
from speechflow.utils.tensor_utils import get_mask_from_lengths
from tts.acoustic_models.data_types import TTSForwardOutput, TTSTarget

__all__ = ["TTSLoss", "MultipleLoss"]


class TTSLoss(BaseCriterion):
    def __init__(self, cfg: Config):
        super().__init__()
        self.spectral_loss = nn.ModuleList()
        self.attention_loss = nn.ModuleList()
        self.durations_loss = None
        self.pitch_loss = None
        self.energy_loss = None
        self.spectral_flatness_loss = None
        self.gate_loss = None
        self.vae_loss = None
        self.inverse_speaker_loss = None
        self.mle_loss = None

        for loss_type in cfg:
            if loss_type[0].islower():
                continue

            if hasattr(sys.modules[__name__], loss_type):
                loss = getattr(sys.modules[__name__], loss_type)(**cfg[loss_type])
                if isinstance(loss, BaseSpectral):
                    self.spectral_loss.append(loss)
                elif isinstance(loss, BaseAttention):
                    self.attention_loss.append(loss)
                elif isinstance(loss, Gate):
                    self.gate_loss = loss
                elif isinstance(loss, VAELoss):
                    self.vae_loss = loss
                elif isinstance(loss, InverseSpeakerLoss):
                    self.inverse_speaker_loss = loss
                elif isinstance(loss, MLELoss):
                    self.mle_loss = loss
                else:
                    raise ValueError(f"{loss_type} not initialized!")
            else:
                raise NotImplementedError(f"{loss_type} not implemented!")

    def forward(
        self,
        output: TTSForwardOutput,
        target: TTSTarget,
        batch_idx: int = 0,
        global_step: int = 0,
    ) -> tp.Dict[str, torch.Tensor]:
        output_spec = output.spectrogram
        target_spec = target.spectrogram

        if target.input_lengths is not None:
            input_mask = get_mask_from_lengths(target.input_lengths)
        else:
            input_mask = None

        if output.spectrogram_lengths is not None:
            output_mask = get_mask_from_lengths(output.spectrogram_lengths)
        else:
            output_mask = None

        total_loss = {}
        for sp_loss in self.spectral_loss:
            total_loss[sp_loss.__class__.__name__] = sp_loss(
                global_step, output_spec, target_spec, output_mask
            )

        for sp_loss in self.attention_loss:
            total_loss[sp_loss.__class__.__name__] = sp_loss(
                global_step,
                output.attention_weights,
                target.input_lengths.cpu().numpy(),
                target.output_lengths.cpu().numpy(),
            )

        additional_losses = (
            output.additional_losses if output.additional_losses is not None else {}
        )

        variance_names = [
            name.replace("_loss", "") for name in self._modules.keys() if "_loss" in name
        ]
        for name in variance_names:
            loss = getattr(self, f"{name}_loss")
            if isinstance(loss, BaseLoss1D):
                if name in additional_losses:
                    precomputed = additional_losses.pop(name)
                    loss_value = loss.precomputed_forward(global_step, precomputed)
                else:
                    var_true = getattr(target, name)
                    if name in output.variance_predictions:
                        var_pred = output.variance_predictions[name]
                    else:
                        var_pred = getattr(output, name)

                    if var_pred is None:
                        raise RuntimeError(
                            f"logits for {loss.__class__.__name__} not found"
                        )

                    if len(var_pred.shape) > 2:
                        var_pred = var_pred.squeeze(-1)

                    if var_true.shape[-1] == target_spec.shape[1]:
                        var_mask = output_mask
                    else:
                        var_mask = input_mask

                    loss_value = loss(
                        global_step,
                        var_pred,
                        var_true,
                        var_mask,
                    )

                total_loss[loss.__class__.__name__] = loss_value

        for name, loss in additional_losses.items():
            if "inverse_speaker" in name:
                if self.inverse_speaker_loss is not None:
                    loss = self.inverse_speaker_loss.precomputed_forward(
                        global_step, loss
                    )

            if "mle" in name:
                if self.mle_loss is not None:
                    loss = self.mle_loss.precomputed_forward(global_step, loss)

            if "kl_loss" in name:
                if self.vae_loss is not None:
                    loss = self.vae_loss(global_step, loss, name)

            if isinstance(loss, dict):
                total_loss.update(loss)
            else:
                total_loss[name] = loss

        return total_loss


class MultipleLoss(BaseCriterion):
    def __init__(self, cfg: Config):
        super().__init__()

        self.losses = nn.ModuleList()
        for name, params in cfg.items():
            if name.islower():
                continue

            if name == "TTSLoss":
                self.losses.append(TTSLoss(params))
            else:
                raise ValueError(f"loss {name} is not implemented")

    def forward(
        self,
        output: TTSForwardOutput,
        target: TTSTarget,
        batch_idx: int = 0,
        global_step: int = 0,
    ) -> tp.Dict[str, torch.Tensor]:

        total_loss = {}
        for loss in self.losses:
            total_loss.update(loss(output, target, batch_idx, global_step))

        return total_loss
