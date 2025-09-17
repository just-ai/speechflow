import abc

import torch
import torch.nn as nn

from tts.acoustic_models.modules.data_types import ComponentInput


class BaseEstimator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        mu_mask: torch.Tensor,
        t: torch.Tensor,
        inputs: ComponentInput,
    ) -> torch.Tensor:
        """Forward pass of the CFM decoder.

        Args:
            x (torch.Tensor): noise,
                shape (batch_size, time, input_dim)
            mu (torch.Tensor): output of encoder,
                shape (batch_size, time, input_dim)
            mu_mask (torch.Tensor):
                shape (batch_size, time)
            t (torch.Tensor): timestep,
                shape (batch_size)
            inputs (dict):

        Returns:
            _type_: _description_

        """

        raise NotImplementedError
