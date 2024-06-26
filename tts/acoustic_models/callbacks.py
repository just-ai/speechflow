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

from pytorch_lightning import Callback

from speechflow.data_pipeline.datasample_processors.text_processors import TextProcessor
from speechflow.utils.plotting import figure_to_ndarray, plot_1d, plot_spectrogram
from tts.acoustic_models.interface.test_utterances_ru import UTTERANCE

LOGGER = logging.getLogger()


class TTSTrainingVisualizer(Callback):
    def __init__(
        self, lang: str = "RU", additional_plots: tp.Optional[tp.Tuple[str]] = None
    ):
        self._text_parser = TextProcessor(lang=lang)
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
            ph_target = targets.transcription[random_idx]
            if ph_target.ndim == 2:
                ph_target = ph_target[:, 0]

            ph_target = [self._text_parser.to_symbol(int(i)) for i in ph_target]
            ph_predicted = outputs.additional_content["predicted_phonemes"][random_idx]
            ph_predicted = [self._text_parser.to_symbol(int(i)) for i in ph_predicted]
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


class TTSAudioSynthesizer(Callback):
    def __init__(
        self,
        vocoder_path: tp.Union[str, Path],
        text_path: Optional[tp.Union[str, Path]] = None,
        num_samples: int = 3,
    ):
        from tts.acoustic_models.interface.eval_interface import TTSEvaluationInterface
        from tts.vocoders.eval_interface import VocoderEvaluationInterface

        vocoder_path = Path(vocoder_path)
        if not vocoder_path.exists():
            raise FileExistsError("Vocoder model not found!")

        self.tts: tp.Optional[TTSEvaluationInterface] = None
        self.vocoder = VocoderEvaluationInterface(vocoder_path, device="cpu")

        if text_path is None:
            self.text_for_synthesis = UTTERANCE[0]
        else:
            self.text_for_synthesis = Path(text_path).read_text(encoding="utf-8")

        self.num_samples = num_samples
        self.sr = self.vocoder.sample_rate

    def on_save_checkpoint(self, pl_module: pl.LightningModule, *args):
        from tts.acoustic_models.interface.eval_interface import TTSEvaluationInterface

        all_checkpoints = glob.glob(
            os.path.join(pl_module.default_root_dir, "_checkpoints/epoch*.ckpt")
        )
        if len(all_checkpoints) == 0:
            return
        else:
            all_checkpoints.sort(key=self._extract_step)

        if self.tts is None:
            self.tts = TTSEvaluationInterface(
                all_checkpoints[-1], device=pl_module.model.device.type, load_model=False
            )

        self.tts.model = pl_module.model.model

        all_outputs = []
        for speaker_name in random.choices(self.tts.get_speakers(), k=self.num_samples):
            try:
                all_outputs.append(
                    self.tts.synthesize(
                        self.text_for_synthesis,
                        self.tts.lang,
                        speaker_name=speaker_name,
                    )
                )
            except Exception as e:
                LOGGER.warning(f"Exit callback with exception: {e}")
                return

        for idx, tts_output in enumerate(all_outputs):
            try:
                voc_output = self.vocoder.synthesize(tts_output)
                output = np.concatenate([s.waveform.numpy() for s in voc_output], axis=1)  # type: ignore
                self.log_waveform(
                    pl_module,
                    output,
                    self.sr,
                    idx,
                    "_test_utterances",
                    pl_module.global_step,
                )
            except Exception as e:
                LOGGER.warning(f"Exit callback with exception: {e}")
                return

    @staticmethod
    def log_waveform(
        pl_module: pl.LightningModule,
        waveform: npt.NDArray,
        sample_rate: int,
        index: int,
        postfix: str,
        step: int,
    ):
        pl_module.logger.experiment.add_audio(
            f"spg_{index}/{postfix}",
            waveform,
            global_step=step,
            sample_rate=sample_rate,
        )

    @staticmethod
    def _extract_step(text):
        """Regular expression extracting number of steps from checkpoint filename."""
        return int(re.split(r"(epoch=)(\d+)(-step)", text)[2])
