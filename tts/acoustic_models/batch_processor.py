import typing as tp
import logging

from speechflow.data_pipeline.collate_functions import (
    TTSCollateOutput,
    TTSCollateOutputWithPrompt,
    TTSCollateOutputWithSSML,
)
from speechflow.data_pipeline.core import BaseBatchProcessor, Batch, DataSample
from speechflow.logging import trace
from speechflow.utils.init import init_class_from_config
from tts.acoustic_models.data_types import (
    TTSForwardInput,
    TTSForwardInputWithPrompt,
    TTSForwardInputWithSSML,
    TTSTarget,
)

__all__ = [
    "TTSBatchProcessor",
    "TTSBatchProcessorWithPrompt",
    "TTSBatchProcessorWithSSML",
]

LOGGER = logging.getLogger("root")


class TTSBatchProcessor(BaseBatchProcessor):
    def __init__(self):
        super().__init__()

    def process_collated(
        self,
        collated: TTSCollateOutput,
        batch_tag: tp.Optional[str] = None,
        batch_idx: tp.Optional[int] = None,
        global_step: tp.Optional[int] = None,
    ) -> (TTSForwardInput, TTSTarget):
        for name in [
            "word_lengths",
            "word_invert_lengths",
            "word_durations",
            "word_invert_durations",
        ]:
            if collated.aggregated and name in collated.aggregated:
                collated.additional_fields[name] = collated.aggregated.get(name)

        _input: TTSForwardInput = init_class_from_config(
            TTSForwardInput, collated.to_dict(), check_keys=False
        )(
            waveform=collated.mu_law_waveform,
            waveform_lengths=collated.mu_law_waveform_lengths,
            symbols=collated.transcription_text,
            transcription=collated.transcription_id,
            transcription_by_frames=collated.transcription_id_by_frames,
            mel_spectrogram=collated.mel,
            linear_spectrogram=collated.magnitude,
            aggregate_energy=collated.aggregated.get("energy"),
            aggregate_pitch=collated.aggregated.get("pitch"),
            aggregate_curv_energy=collated.aggregated.get("curv_energy"),
            aggregate_curv_pitch=collated.aggregated.get("curv_pitch"),
            aggregate_spectral_flatness=collated.aggregated.get("spectral_flatness"),
            aggregate_spectral_envelope=collated.aggregated.get("spectral_envelope"),
            prosody=collated.aggregated.get("pitch_contour"),
            additional_inputs=collated.additional_fields,
            input_lengths=collated.transcription_lengths,
            output_lengths=collated.spectrogram_lengths,
            batch_tag=batch_tag,
            batch_idx=batch_idx,
            global_step=global_step,
        )

        _target: TTSTarget = init_class_from_config(
            TTSTarget, collated.to_dict(), check_keys=False
        )(
            symbols=collated.transcription_text,
            transcription=collated.transcription_id,
            input_lengths=collated.transcription_lengths,
            output_lengths=collated.spectrogram_lengths,
            batch_tag=batch_tag,
            batch_idx=batch_idx,
            global_step=global_step,
        )

        return _input.to(self.device), _target.to(self.device)

    def __call__(
        self, batch: Batch, batch_idx: int = 0, global_step: int = 0
    ) -> (TTSForwardInput, TTSTarget, tp.List[DataSample]):
        collated: TTSCollateOutput = batch.collated_samples  # type: ignore

        inputs, targets = self.process_collated(
            collated=collated,
            batch_tag=batch.tag,
            batch_idx=batch_idx,
            global_step=global_step,
        )

        return inputs, targets, batch.data_samples


class TTSBatchProcessorWithPrompt(TTSBatchProcessor):
    def __call__(
        self, batch: Batch, batch_idx: int = 0, global_step: int = 0
    ) -> (TTSForwardInputWithPrompt, TTSTarget, tp.List[DataSample]):
        collated: TTSCollateOutputWithPrompt = batch.collated_samples  # type: ignore
        if not isinstance(collated, TTSCollateOutputWithPrompt):
            LOGGER.info(trace(self, message="collated is not TTSCollateOutputWithPrompt"))

        _input, _target, data_samples = super().__call__(batch)
        _input = TTSForwardInputWithPrompt(**_input.to_dict())

        if collated.prompt is not None:
            _prompt, *_ = super().process_collated(
                collated.prompt,
                batch_tag=batch.tag,
                batch_idx=batch_idx,
                global_step=global_step,
            )
            _input.prompt = _prompt

        return _input.to(self.device), _target.to(self.device), batch.data_samples


class TTSBatchProcessorWithSSML(TTSBatchProcessor):
    def __call__(
        self,
        batch: Batch,
        batch_idx: int = 0,
        global_step: int = 0,
    ) -> (TTSForwardInputWithSSML, TTSTarget, tp.List[DataSample]):
        collated: TTSCollateOutputWithSSML = batch.collated_samples  # type: ignore
        if not isinstance(collated, TTSCollateOutputWithSSML):
            LOGGER.info(
                trace(self, message="collated is not TTSCollateOutputWithSSML type")
            )

        _input, _target, data_samples = super().__call__(batch)

        _input = TTSForwardInputWithSSML(**_input.to_dict())
        _input.pitch_modifier = getattr(collated, "pitch_modifier", None)
        _input.volume_modifier = getattr(collated, "volume_modifier", None)
        _input.rate_modifier = getattr(collated, "temp_modifier", None)

        return _input.to(self.device), _target, batch.data_samples
