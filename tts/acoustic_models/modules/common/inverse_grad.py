import torch
import torch.nn as nn

from torch.autograd import Function
from torch.nn import functional

__all__ = [
    "InverseGradSpeakerIDPredictor",
    "InverseGradSpeakerPredictor",
    "InverseGradStylePredictor",
    "InverseGrad1DPredictor",
    "InverseGradPhonemePredictor",
]


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class RevGradLayer(nn.Module):
    def __init__(self, alpha=1.0, *args, **kwargs):
        """A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient in the
        backward pass.

        """
        super().__init__(*args, **kwargs)
        self._alpha = torch.tensor(alpha)

    def forward(self, input_):
        return RevGrad.apply(input_, self._alpha)


class InverseGradSpeakerIDPredictor(nn.Module):
    def __init__(self, input_dim, n_speakers, hidden_dim=256):
        super().__init__()
        self.predictor = nn.Sequential(
            RevGradLayer(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_speakers),
        )

    def forward(self, z, targets):
        if z.ndim == 3:
            z = z.sum(1)
        return functional.cross_entropy(self.predictor(z.squeeze(1)), targets.detach())


class InverseGradSpeakerPredictor(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_dim=256):
        super().__init__()
        self.predictor = nn.Sequential(
            RevGradLayer(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, z, targets):
        if z.ndim == 3:
            z = z.sum(1)
        return functional.mse_loss(self.predictor(z.squeeze(1)), targets.detach())


class InverseGradStylePredictor(InverseGradSpeakerPredictor):
    pass


class InverseGrad1DPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.predictor = nn.Sequential(
            RevGradLayer(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, targets):
        return functional.mse_loss(self.predictor(z).squeeze(-1), targets.detach())


class InverseGradPhonemePredictor(nn.Module):
    def __init__(self, input_dim, n_phonemes, hidden_dim=256):
        super().__init__()
        self.predictor = nn.Sequential(
            RevGradLayer(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_phonemes),
        )

    def forward(self, z, targets):
        return functional.cross_entropy(
            self.predictor(z).transpose(1, -1), targets.detach()
        )
