from typing import Optional

import torch

from torch import nn
from torch.nn import functional as F

from speechflow.utils.tensor_utils import get_lengths_from_durations, stack

__all__ = ["LengthRegulator", "SoftLengthRegulator"]


class LengthRegulator(nn.Module):
    """Length Regulator."""

    def regulate_lengthgth(self, x, durations, max_length):
        output = list()
        mel_length = list()
        for batch, expand_target in zip(x, durations):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_length.append(expanded.shape[0])

        if max_length is not None:
            output = stack(output, max_length)
        else:
            output = stack(output)

        return output, torch.LongTensor(mel_length).to(x.device)

    @staticmethod
    def expand(batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)

        return out

    def forward(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
        max_length: Optional[int] = None,
        upsample_x2: bool = False,
    ):
        output, mel_length = self.regulate_lengthgth(x, durations, max_length)
        return output, mel_length


class SoftLengthRegulator(nn.Module):
    def __init__(self, sigma: float = 0.2, hard: bool = False):
        super().__init__()
        self._sigma = sigma
        self._hard = hard

    def _softmax_interpolation(
        self,
        embeddings,
        durations,
        max_length: Optional[int] = None,
        upsample_x2: bool = False,
    ):
        """End-to-End AdversarialText-to-Speech {https://arxiv.org/abs/2006.03575}."""

        with torch.no_grad():
            if max_length is None:
                max_length = get_lengths_from_durations(durations).max()

            if upsample_x2:
                durations = durations * 2
                max_length = max_length * 2

            if self._hard and durations.dtype != torch.long:
                durations = durations.round()

            durations = durations.float()

            durations_cs = torch.cumsum(durations, dim=-1)
            durations_cs_centered = durations_cs - durations

            frames_decoder = torch.matmul(
                torch.ones_like(durations).unsqueeze(2),
                torch.arange(
                    0,
                    max_length,
                    dtype=torch.float,
                    requires_grad=self.training,
                    device=durations.device,
                ).unsqueeze(0),
            )

            frames_decoder_shifted = frames_decoder - torch.matmul(
                durations_cs_centered.unsqueeze(2),
                torch.ones(
                    max_length,
                    dtype=torch.float,
                    requires_grad=self.training,
                    device=durations.device,
                ).unsqueeze(0),
            )

            if self._hard:
                mask_down = frames_decoder_shifted >= 0
                mask_up = torch.roll(mask_down, -1, dims=1)
                mask = torch.bitwise_xor(mask_down, mask_up)
                attention_weights = mask.float()
            else:
                attention_weights = torch.softmax(
                    -(frames_decoder_shifted**2) * self._sigma, dim=1
                )

        return (
            torch.matmul(embeddings, attention_weights),
            attention_weights,
        )

    def forward(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
        max_length: Optional[int] = None,
        upsample_x2: bool = False,
    ):
        aligned_emb, attention_weights = self._softmax_interpolation(
            x.transpose(2, 1),
            durations,
            max_length=max_length,
            upsample_x2=upsample_x2,
        )
        aligned_emb = aligned_emb.transpose(2, 1)

        if upsample_x2:
            aligned_emb = F.avg_pool1d(
                aligned_emb.transpose(2, 1), kernel_size=3, stride=2, ceil_mode=True
            ).transpose(2, 1)

        # mask = get_mask_from_lengthgths2(get_lengthgths_from_durations(durations))
        # if mask.shape[1] == aligned_emb.shape[1]:
        #    aligned_emb = apply_mask(aligned_emb, mask)

        return aligned_emb, attention_weights
