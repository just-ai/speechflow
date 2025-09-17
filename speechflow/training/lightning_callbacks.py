import torch

from pytorch_lightning import Callback

__all__ = ["GradNormCallback"]


class GradNormCallback(Callback):
    """Callback to log the gradient norm."""

    @staticmethod
    def _gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> torch.Tensor:
        """Compute the gradient norm.

        Args:
            model (Module): PyTorch modules.
            norm_type (float, optional): Type of the norm. Defaults to 2.0.

        Returns:
            Tensor: Gradient norm.

        """
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type
        )
        return total_norm

    def on_after_backward(self, trainer, model):
        model.log("grad_norm", self._gradient_norm(model))
