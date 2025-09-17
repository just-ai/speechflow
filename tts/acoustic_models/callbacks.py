import os
import re
import glob
import random
import typing as tp
import logging

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import numpy.typing as npt
import pytorch_lightning as pl

from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from sklearn.manifold import TSNE

from speechflow.data_pipeline.datasample_processors.tts_text_processors import (
    TTSTextProcessor,
)
from speechflow.utils.plotting import figure_to_ndarray, plot_1d, plot_spectrogram

__all__ = ["TTSTrainingVisualizer", "ProsodyTrainingVisualizer"]

LOGGER = logging.getLogger("root")


class TTSTrainingVisualizer(Callback):
    def __init__(
        self, lang: str = "RU", additional_plots: tp.Optional[tp.Tuple[str]] = None
    ):
        self._text_parser = TTSTextProcessor(lang=lang)
        self._additional_plots = additional_plots

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

        _, outputs, targets, _ = pl_module.test_step(batch, batch_idx)
        batch_size = targets.spectrogram.size(0)

        if batch_size <= 1:
            random_idx = 0
        else:
            random_idx = np.random.randint(0, batch_size - 1)

        T = targets.output_lengths[random_idx].item()

        for name in ["spectrogram"]:
            target = getattr(targets, name, None)
            if target is not None:
                t = target[random_idx]
                if t.ndim != 2:
                    continue
                t = t.T.detach().cpu().numpy()
                self._log_spectrogram(pl_module, trainer, t[:, :T], f"Target_{name}")

        predicted = outputs.spectrogram
        if predicted is not None:
            if len(predicted.shape) == 3:
                predicted = predicted.unsqueeze(0)

            for idx in range(predicted.shape[0]):
                spec = predicted[idx][random_idx].T.detach().cpu().numpy()
                self._log_spectrogram(
                    pl_module, trainer, spec[:, :T], f"Predicted_spectrogram_{idx}"
                )

        if "energy" in outputs.variance_predictions:
            energy_target = targets.energy[random_idx]
            energy_predict = outputs.variance_predictions["energy"][random_idx]
            energy = torch.stack([energy_target, energy_predict]).detach().cpu().numpy()
            self._log_1d_signal(pl_module, trainer, energy[:, :T], "Predicted_energy")

        if "pitch" in outputs.variance_predictions:
            pitch_predict = outputs.variance_predictions["pitch"][random_idx]
            if pitch_predict.ndim == 1:
                pitch_target = targets.pitch[random_idx]
                if pitch_predict.max() < 10:
                    pitch_predict = torch.expm1(pitch_predict)
                pitch = torch.stack([pitch_target, pitch_predict]).detach().cpu().numpy()
                self._log_1d_signal(pl_module, trainer, pitch[:, :T], "Predicted_pitch")
            else:
                pitch_predict = pitch_predict.T.detach().cpu().numpy()
                self._log_spectrogram(
                    pl_module, trainer, pitch_predict[:, :T], "Predicted_pitch"
                )

        for name in outputs.additional_content.keys():
            if "vae_latent" in name or "vector_quantizer" in name:
                quant = outputs.additional_content[name]
                if quant.ndim != 2:
                    continue
                quant = quant[random_idx].T.detach().cpu().numpy()
                self._log_spectrogram(pl_module, trainer, quant[:, :T], name)

        if self._additional_plots:
            target_fields: tp.Dict = targets.to_dict()
            target_fields.update(targets.additional_inputs)
            for key, feats in zip(
                ["Target", "Predictions"], [target_fields, outputs.additional_content]
            ):
                for name in self._additional_plots:
                    if name in feats:
                        t = feats[name][random_idx]
                        if t.ndim != 2:
                            continue
                        t = t.T.detach().cpu().numpy()
                        self._log_spectrogram(
                            pl_module, trainer, t[:, :T], f"{key}_{name}"
                        )

        if "predicted_phonemes" in outputs.additional_content:
            ph_target = targets.transcription_id[random_idx]
            if ph_target.ndim == 2:
                ph_target = ph_target[:, 0]

            ph_target = [self._text_parser.id_to_symbol(int(i)) for i in ph_target]
            ph_predicted = outputs.additional_content["predicted_phonemes"][random_idx]
            ph_predicted = [self._text_parser.id_to_symbol(int(i)) for i in ph_predicted]
            pl_module.logger.experiment.add_text(
                "decoder/ph_target", " ".join(ph_target), trainer.global_step
            )
            pl_module.logger.experiment.add_text(
                "decoder/ph_predicted", " ".join(ph_predicted), trainer.global_step
            )

        if "durations_postprocessed" in outputs.additional_content:
            dura = {
                "target_dura": targets.durations,
                "predict_dura": outputs.additional_content["durations_postprocessed"],
            }
            if "fa_durations" in outputs.additional_content:
                dura["fa_dura"] = outputs.additional_content["fa_durations"]

            for k, v in dura.items():
                dura[k] = v[random_idx].cpu().cumsum(0).long().numpy().tolist()
                dura[k] = dura[k][:T]

            spec = targets.spectrogram[random_idx].T.detach().cpu().numpy()
            self._log_spectrogram(pl_module, trainer, spec[:, :T], "Durations", dura)

        if "fa_attn" in outputs.additional_content:
            fa_attn = outputs.additional_content["fa_attn"]
            fa_attn = fa_attn[random_idx].detach().cpu().numpy()
            self._log_spectrogram(pl_module, trainer, fa_attn[:, :T], "fa_attn")

    @staticmethod
    def _log_spectrogram(
        module: pl.LightningModule,
        trainer: pl.Trainer,
        spectrogram: npt.NDArray,
        tag: str,
        durations=None,
    ):
        try:
            fig_to_plot = plot_spectrogram(spectrogram, phonemes_ticks=durations)
            data_to_log = figure_to_ndarray(fig_to_plot)

            module.logger.experiment.add_image(
                tag,
                data_to_log,
                trainer.global_step,
                dataformats="CHW",
            )
        except Exception as e:
            LOGGER.error(e)

    @staticmethod
    def _log_1d_signal(
        module: pl.LightningModule,
        trainer: pl.Trainer,
        signal: npt.NDArray,
        tag: str,
    ):
        fig_to_plot = plot_1d(signal)
        data_to_log = figure_to_ndarray(fig_to_plot)

        module.logger.experiment.add_image(
            tag,
            data_to_log,
            trainer.global_step,
            dataformats="CHW",
        )


class ProsodyTrainingVisualizer(Callback):
    def __init__(
        self,
        n_classes: tp.Optional[int] = 8,
        eps: tp.Optional[int] = None,
    ):
        self.n_classes = n_classes
        self.eps = eps

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

        vq_enc = pl_module.model.encoder.vq_encoder
        if hasattr(vq_enc, "vq"):
            codebook = vq_enc.vq.codebook.weight
        elif hasattr(vq_enc, "rlfq"):
            codebook = vq_enc.rlfq.codebooks[0]
        elif hasattr(vq_enc, "rfsq"):
            codebook = vq_enc.rfsq.codebooks[0]
        else:
            raise NotImplementedError

        embeddings = codebook.cpu().detach().numpy()

        mapping = self.clustering(embeddings, n_classes=self.n_classes)
        if len(mapping) > 0:
            pl_module.logger.experiment.add_image(
                "vq_gaussian_clusters",
                mapping,
                trainer.global_step,
                dataformats="CHW",
            )

        mapping = self.clustering(embeddings, eps=self.eps)
        if len(mapping) > 0:
            pl_module.logger.experiment.add_image(
                "vq_dbscan_clusters",
                mapping,
                trainer.global_step,
                dataformats="CHW",
            )

    @staticmethod
    def clustering(
        embeddings: np.array,
        n_classes: tp.Optional[int] = None,
        eps: tp.Optional[int] = None,
    ):
        from sklearn.cluster import DBSCAN
        from sklearn.mixture import GaussianMixture

        """
        Function for clustering of embeddings from the codebook
        1. Outliers are obtained by DBSCAN
        2. Outliers are clustered by GaussianMixture clustering
        3. Indices from the codebook are mapped with the corresponding classes
        """

        if n_classes:
            classes = GaussianMixture(n_components=n_classes).fit_predict(embeddings)
        elif eps:
            classes = DBSCAN(eps=eps, min_samples=2).fit_predict(embeddings)
        else:
            classes = np.zeros(embeddings.shape[0])

        tsne_ak_2d = TSNE(n_components=2, init="pca", n_iter=3000, random_state=32)
        projections = tsne_ak_2d.fit_transform(embeddings)

        fig = plt.figure(figsize=(16, 9))

        unique = list(set(classes))
        colors = [plt.cm.jet(float(i + 1) / (max(unique) + 1)) for i in unique]
        for i, u in enumerate(unique):
            points = projections[np.where(classes == u)]
            plt.scatter(points[:, 0], points[:, 1], c=colors[i], label=str(u))

        plt.legend()
        fig.canvas.draw()
        array = figure_to_ndarray(fig)
        plt.close()

        return array
