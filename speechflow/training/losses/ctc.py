import typing as tp

import torch

from speechflow.training.base_loss import BaseLoss, apply_interval_check

__all__ = ["CTCLoss"]


def ctc_loss(log_probs, targets, input_lens, target_lens, blank_index, reduction="mean"):
    """CTC loss.

    Arguments:
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape [batch, time, chars].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len]
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    blank_index : int
        The location of the blank symbol among the character indexes.
    reduction : str
        What reduction to apply to the output. 'mean', 'sum', 'batch',
        'batch_mean', 'none'.
        See pytorch for 'mean', 'sum', 'none'. The 'batch' option returns
        one loss per item in the batch, 'batch_mean' returns sum / batch size.

    """
    input_lens = (input_lens * log_probs.shape[1]).ceil().long()
    target_lens = (target_lens * targets.shape[1]).ceil().long()
    log_probs = log_probs.transpose(0, 1)

    if reduction == "batch_mean":
        reduction_loss = "sum"
    elif reduction == "batch":
        reduction_loss = "none"
    elif reduction == "weighted":
        reduction_loss = "none"
    else:
        reduction_loss = reduction

    loss = torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lens,
        target_lens,
        blank_index,
        zero_infinity=True,
        reduction=reduction_loss,
    )

    if reduction == "batch_mean":
        return loss / targets.shape[0]
    elif reduction == "batch":
        N = loss.size(0)
        return loss.view(N, -1).sum(1) / target_lens.view(N, -1).sum(1)
    elif reduction == "weighted":
        return loss.mean()
    else:
        return loss


class CTCLoss(BaseLoss):
    def __init__(
        self,
        blank_index: int = 0,
        reduction: tp.Literal["batch_mean", "batch", "weighted"] = "batch_mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blank_index = blank_index
        self.reduction = reduction

    @apply_interval_check
    def forward(
        self,
        global_step: int,
        predict: torch.Tensor,
        predict_lens: torch.Tensor,
        transcription: torch.Tensor,
        transcription_lens: torch.Tensor,
    ) -> torch.Tensor:
        return self.scale * ctc_loss(
            predict,
            transcription,
            predict_lens,
            transcription_lens,
            self.blank_index,
            self.reduction,
        )
