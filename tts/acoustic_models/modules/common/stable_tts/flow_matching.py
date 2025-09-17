import typing as tp

from abc import ABC

import torch
import torch.nn.functional as F

from tts.acoustic_models.modules.data_types import ComponentInput

from .estimator import BaseEstimator


class BaseCFM(torch.nn.Module, ABC):
    def __init__(self, estimator: BaseEstimator, sigma_min: float = 1e-4):
        super().__init__()
        self.estimator = estimator
        self.sigma_min = sigma_min

    @torch.inference_mode()
    def forward(
        self,
        mu,
        mu_mask,
        inputs: ComponentInput,
        n_timesteps=10,
        temperature=1.0,
        **kwargs,
    ):
        """Forward diffusion.

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, time, mel_feats)
            mu_mask (torch.Tensor): output_mask
                shape: (batch_size, time)
            inputs (ComponentInput):
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, time, mel_feats)

        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        return self.solve_euler(
            z,
            t_span=t_span,
            mu=mu,
            mu_mask=mu_mask,
            inputs=inputs,
            **kwargs,
        )

    def solve_euler(
        self,
        x,
        t_span,
        mu,
        mu_mask,
        inputs: ComponentInput,
        **kwargs,
    ):
        """Fixed euler solver for ODEs.

        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, time, mel_feats)
            mu_mask (torch.Tensor): output_mask
                shape: (batch_size, time)
            inputs (ComponentInput):

        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        steps = 1
        while steps <= len(t_span) - 1:
            dphi_dt = self.func_dphi_dt(x, mu, mu_mask, t, inputs, **kwargs)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]

    def func_dphi_dt(
        self,
        x,
        mu,
        mu_mask,
        t,
        inputs: ComponentInput,
        guidance_scale: float = 0.0,
        fake_content: tp.Optional[torch.Tensor] = None,
        fake_condition: tp.Optional[torch.Tensor] = None,
    ):
        if "cfg_condition_masked" in inputs.additional_content:
            inputs.additional_content.pop("cfg_condition_masked")

        dphi_dt = self.estimator(x, mu, mu_mask, t, inputs)

        if guidance_scale > 0.0:
            fake_content = fake_content.repeat(x.shape[0], x.shape[1], 1)
            fake_condition = fake_condition.repeat(x.shape[0], 1, 1)
            inputs.additional_content["cfg_condition_masked"] = fake_condition

            dphi_avg = self.estimator(x, fake_content, mu_mask, t, inputs)
            dphi_dt = dphi_dt + guidance_scale * (dphi_dt - dphi_avg)

        return dphi_dt

    def compute_loss(
        self,
        mu,
        mu_mask,
        target,
        inputs: ComponentInput,
    ):
        """Computes diffusion loss.

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, time, mel_feats)
            mu_mask (torch.Tensor): output_mask
                shape: (batch_size, time)
            target (torch.Tensor): output of encoder
                shape: (batch_size, time, mel_feats)
            inputs (dict):

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, time, mel_feats)

        """
        b, t, _ = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        t = 1 - torch.cos(t * 0.5 * torch.pi)

        # sample noise p(x_0)
        z = torch.randn_like(target)

        y = (1 - (1 - self.sigma_min) * t) * z + t * target
        u = target - (1 - self.sigma_min) * z

        est = self.estimator(y, mu, mu_mask, t.squeeze(), inputs)
        loss_mse = F.mse_loss(est, u, reduction="sum") / (torch.sum(mu_mask) * u.shape[1])
        return loss_mse, y
