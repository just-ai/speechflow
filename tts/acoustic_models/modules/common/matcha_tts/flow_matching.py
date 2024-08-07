from abc import ABC

import torch
import torch.nn.functional as F

from speechflow.training.utils.tensor_utils import apply_mask
from tts.acoustic_models.modules.common.matcha_tts.decoder import Decoder


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion.

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, speaker_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)

        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """Fixed euler solver for ODEs.

        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, speaker_emb_dim)
            cond: Not used but kept for future purposes

        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag

        steps = 1
        x = apply_mask(x, mask)
        while steps <= len(t_span) - 1:
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)

            x = apply_mask(x + dt * dphi_dt, mask)
            t = t + dt
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1
            # plot_tensor(x[0])

        return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss.

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, speaker_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)

        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(
            self.estimator(y, mask, mu, t.squeeze(), spks, cond),
            apply_mask(u, mask),
            reduction="sum",
        ) / (torch.sum(mask) * u.shape[1])

        return loss, y


class CFM(BASECFM):
    def __init__(
        self,
        in_channels,
        out_channel,
        cfm_params,
        decoder_params,
        speaker_emb_dim=0,
        cond_dim=0,
    ):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
        )

        # Just change the architecture of the estimator here
        self.estimator = Decoder(
            in_channels=in_channels + speaker_emb_dim,
            out_channels=out_channel,
            cond_dim=cond_dim,
            **decoder_params,
        )
