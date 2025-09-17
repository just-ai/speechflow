import typing as tp

import torch

from speechflow.training import BaseCriterion
from tts.forced_alignment.data_types import AlignerForwardOutput, AlignerForwardTarget
from tts.forced_alignment.model.aligner_loss import (
    AttentionCTCLoss,
    BinLoss,
    ForwardSumLoss,
)

__all__ = ["GlowTTSLoss", "NemoAlignerLoss", "LorAlignerLoss"]


class GlowTTSLoss(BaseCriterion):
    def __init__(
        self,
        bin_loss_begin_iter: int = 0,
        bin_loss_end_anneal_iter: int = 100000,
    ):
        super().__init__()
        self.lor_loss = LorAlignerLoss(bin_loss_begin_iter, bin_loss_end_anneal_iter)

    def forward(
        self,
        output: AlignerForwardOutput,
        target: AlignerForwardTarget,
        batch_idx: int = 0,
        global_step: int = 0,
    ) -> tp.Dict[str, torch.Tensor]:
        total_loss = {
            "MLELoss": output.mle_loss,
            "DurationLoss": output.duration_loss,
        }

        if "attn_soft" in output.additional_content:
            total_loss.update(self.lor_loss(output, target, batch_idx, global_step))

        return total_loss


class NemoAlignerLoss(BaseCriterion):
    def __init__(
        self,
        bin_loss_begin_iter: int = 0,
        bin_loss_end_anneal_iter: int = 100000,
    ):
        super().__init__()
        self.forward_sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.begin_iter = bin_loss_begin_iter
        self.end_anneal_iter = bin_loss_end_anneal_iter
        self.every_iter = 1
        self.scale = 1.0

    def _scale_scheduler_step(self, current_iter: int) -> float:
        if current_iter % self.every_iter != 0:
            return 0.0

        if current_iter < self.begin_iter:
            scale = 0.0
        elif self.begin_iter < current_iter < self.end_anneal_iter:
            scale = (
                self.scale
                * (
                    (current_iter - self.begin_iter)
                    / (self.end_anneal_iter - self.begin_iter)
                )
                ** 2
            )
        else:
            scale = self.scale

        return scale

    def forward(
        self,
        output: AlignerForwardOutput,
        target: AlignerForwardTarget,
        batch_idx: int = 0,
        global_step: int = 0,
    ) -> tp.Dict[str, torch.Tensor]:
        attn_soft = output.additional_content.get("attn_soft")
        attn_hard = output.additional_content.get("attn_hard")
        attn_logprob = output.additional_content.get("attn_logprob")

        forward_sum_loss = self.forward_sum_loss(
            attn_logprob=attn_logprob,
            in_lens=target.input_lengths,
            out_lens=output.output_lengths,
        )
        total_loss = {"ForwardSumLoss": forward_sum_loss}

        if attn_hard is not None:
            bin_scale = self._scale_scheduler_step(global_step)
            bin_loss = bin_scale * self.bin_loss(
                hard_attention=attn_hard, soft_attention=attn_soft
            )
            total_loss.update({"BinLoss": bin_loss})

        return total_loss


class LorAlignerLoss(NemoAlignerLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_sum_loss = AttentionCTCLoss()
