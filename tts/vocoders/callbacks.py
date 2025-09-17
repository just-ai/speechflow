import typing as tp

import numpy as np
import torch
import numpy.typing as npt
import pytorch_lightning as pl

from pytorch_lightning import Callback

from speechflow.utils.plotting import figure_to_ndarray, plot_1d

__all__ = ["VisualizerCallback"]


class VisualizerCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if batch_idx != 0:
            return

        with torch.no_grad():
            inputs, targets, metadata = pl_module.batch_processor(batch, batch_idx, 0)
            _, _, ft_additional = pl_module.feature_extractor(inputs)

        batch_size = targets.spectrogram.size(0)

        if batch_size <= 1:
            random_idx = 0
        else:
            random_idx = np.random.randint(0, batch_size - 1)

        if inputs.spectrogram is not None:
            T = inputs.spectrogram_lengths[random_idx]
            self._log_2d(
                "target/spectrogram", inputs.spectrogram[random_idx][:T], pl_module
            )

        if inputs.ssl_feat is not None:
            T = inputs.ssl_feat_lengths[random_idx]
            self._log_2d("target/ssl_feat", inputs.ssl_feat[random_idx][:T], pl_module)

        for name in ["energy", "pitch", "durations"]:
            try:
                T = inputs.output_lengths[random_idx]
                target_signal = getattr(targets, name)
                if target_signal is not None:
                    target_signal = target_signal[random_idx][:T]
                    if f"{name}_vp_predict" in ft_additional:
                        predict = ft_additional[f"{name}_vp_predict"]
                    else:
                        predict = ft_additional[name].squeeze(-1)
                    data = torch.stack([target_signal, predict[random_idx][:T]])
                    self._log_1d(f"predict/{name}", data, pl_module)
            except Exception:
                pass

    @staticmethod
    def _log_2d(
        tag: str,
        tensor: tp.Union[npt.NDArray, torch.Tensor],
        pl_module: pl.LightningModule,
    ):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu()
        else:
            tensor = torch.from_numpy(tensor)

        pl_module.log_image(tag, tensor.t())

    @staticmethod
    def _log_1d(
        tag: str,
        signal: tp.Union[npt.NDArray, torch.Tensor],
        pl_module: pl.LightningModule,
    ):
        if isinstance(signal, torch.Tensor):
            signal = signal.cpu().numpy()

        fig_to_plot = plot_1d(signal)
        data_to_log = figure_to_ndarray(fig_to_plot)
        data_to_log = data_to_log.swapaxes(0, 2).swapaxes(0, 1)
        pl_module.log_image(tag, None, fig=data_to_log)
