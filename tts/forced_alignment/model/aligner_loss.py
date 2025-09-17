import torch
import torch.nn.functional as F

__all__ = ["ForwardSumLoss", "AttentionCTCLoss", "BinLoss"]


class ForwardSumLoss(torch.nn.Module):
    def __init__(self, blank_logprob: float = -1):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens, **kwargs):
        key_lens = in_lens
        query_lens = out_lens
        max_key_len = attn_logprob.size(-1)

        # Reorder input to [query_len, batch_size, key_len]
        attn_logprob = attn_logprob.squeeze(1)
        attn_logprob = attn_logprob.permute(1, 0, 2)

        # Add blank label
        attn_logprob = F.pad(
            input=attn_logprob, pad=(1, 0, 0, 0, 0, 0), value=self.blank_logprob
        )

        # Convert to log probabilities
        # Note: Mask out probs beyond key_len
        key_inds = torch.arange(
            max_key_len + 1, device=attn_logprob.device, dtype=torch.long
        )

        attn_logprob.masked_fill_(
            key_inds.view(1, 1, -1) > key_lens.view(1, -1, 1),
            -float("inf"),  # key_inds >= key_lens+1
        )
        attn_logprob = self.log_softmax(attn_logprob)

        # Target sequences
        target_seqs = key_inds[1:].unsqueeze(0)
        target_seqs = target_seqs.repeat(key_lens.numel(), 1)

        # Evaluate CTC loss
        cost = self.ctc_loss(
            attn_logprob, target_seqs, input_lengths=query_lens, target_lengths=key_lens
        )
        return cost


class AttentionCTCLoss(ForwardSumLoss):
    def __init__(self, blank_logprob: float = -1):
        super().__init__(blank_logprob)

    def forward(self, attn_logprob, in_lens, out_lens):
        """
        Args:
            attn_logprob: batch x 1 x max(out_lens) x max(in_lens) batched tensor of attention
                          log probabilities, padded to length of longest sequence in each dimension
            in_lens: batch-D vector of length of each text sequence
            out_lens: batch-D vector of length of each mel sequence
        """
        attn_logprob_padded = F.pad(
            input=attn_logprob, pad=(1, 0, 0, 0, 0, 0, 0, 0), value=self.blank_logprob
        )

        cost_total = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, in_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                : out_lens[bid], :, : in_lens[bid] + 1
            ]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            ctc_cost = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=out_lens[bid : bid + 1],
                target_lengths=in_lens[bid : bid + 1],
            )
            cost_total += ctc_cost

        cost = cost_total / attn_logprob.shape[0]
        return cost


class BinLoss(torch.nn.Module):
    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(
            torch.clamp(soft_attention[hard_attention == 1], min=1e-12)
        ).sum()
        return -log_sum / hard_attention.sum()
