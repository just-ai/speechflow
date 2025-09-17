import random
import typing as tp

import numpy as np
import torch
import numpy.typing as npt
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback

from speechflow.utils.plotting import (
    figure_to_ndarray,
    phonemes_to_frame_ticks,
    plot_spectrogram,
)


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def _plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(12, 4))  # type: ignore
    im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


class AligningVisualisationCallback(Callback):
    def __init__(self):
        pass

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if batch_idx == 0:
            inputs, outputs, _, _ = pl_module.test_step(batch, batch_idx)
            aligning = outputs.aligning_path

            batch_size = aligning.size(0)
            random_idx = random.randint(0, batch_size - 1)

            spec_lens = inputs.output_lengths.detach().cpu().numpy().tolist()
            phonemes = list(
                batch.collated_samples.additional_fields["transcription_text"][random_idx]
            )
            spectrogram = (
                inputs.spectrogram[random_idx, : spec_lens[random_idx]]
                .T.detach()
                .cpu()
                .numpy()
            )
            alignment = (
                outputs.aligning_path[
                    random_idx, : spec_lens[random_idx], : len(phonemes)
                ]
                .detach()
                .cpu()
                .numpy()
            )

            self._log_target(pl_module, trainer, spectrogram, alignment, phonemes)
            self._log_aligning(pl_module, trainer, np.transpose(alignment))

            if inputs.ssl_feat is not None:
                ssl_feat = inputs.ssl_feat[random_idx].cpu().numpy()
                self._log_spectrogram(pl_module, trainer, ssl_feat, "SSLTarget")

            if "ssl_prediction" in outputs.additional_content:
                ssl_prediction = outputs.additional_content["ssl_prediction"]
                ssl_prediction = ssl_prediction[random_idx].cpu().numpy()
                self._log_spectrogram(pl_module, trainer, ssl_prediction, "SSLPrediction")

    @staticmethod
    def _log_target(
        module: pl.LightningModule,
        trainer: pl.Trainer,
        spectrogram: npt.NDArray,
        alignment: npt.NDArray,
        phonemes: tp.List[str],
        name: str = "TargetSpectrogramWithAlignment",
    ):
        frame_ticks = phonemes_to_frame_ticks(alignment, phonemes)
        frame_ticks = [t for t in frame_ticks]
        fig_to_plot = plot_spectrogram(spectrogram, phonemes, frame_ticks)
        data_to_log = figure_to_ndarray(fig_to_plot)

        module.logger.experiment.add_image(
            name,
            data_to_log,
            trainer.global_step,
            dataformats="CHW",
        )

    @staticmethod
    def _log_aligning(
        module: pl.LightningModule,
        trainer: pl.Trainer,
        aligning: torch.Tensor,
        name: str = "Aligning",
    ):
        module.logger.experiment.add_image(
            name,
            _plot_alignment_to_numpy(aligning),
            trainer.global_step,
            dataformats="HWC",
        )

    @staticmethod
    def _log_spectrogram(
        module: pl.LightningModule,
        trainer: pl.Trainer,
        spectrogram: npt.NDArray,
        tag: str,
        durations=None,
    ):
        fig_to_plot = plot_spectrogram(spectrogram, phonemes_ticks=durations)
        data_to_log = figure_to_ndarray(fig_to_plot)

        module.logger.experiment.add_image(
            tag,
            data_to_log,
            trainer.global_step,
            dataformats="CHW",
        )
