import typing as tp

from speechflow.data_pipeline.collate_functions import TTSCollateOutput
from speechflow.data_pipeline.core import BaseBatchProcessor, Batch, DataSample
from tts.forced_alignment.data_types import AlignerForwardInput, AlignerForwardTarget

__all__ = ["AlignerBatchProcessor"]


class AlignerBatchProcessor(BaseBatchProcessor):
    def __init__(self):
        BaseBatchProcessor.__init__(self)

    def __call__(
        self, batch: Batch, batch_idx: int = 0, global_step: int = 0
    ) -> tp.Tuple[
        AlignerForwardInput, AlignerForwardTarget, tp.Optional[tp.List[DataSample]]
    ]:
        collated: TTSCollateOutput = batch.collated_samples  # type: ignore

        _input: AlignerForwardInput = AlignerForwardInput(
            lang_id=collated.lang_id,
            speaker_id=collated.speaker_id,
            speaker_emb=collated.speaker_emb,
            speech_quality_emb=collated.speech_quality_emb,
            transcription=collated.transcription_id,
            transcription_lengths=collated.transcription_lengths,
            ling_feat=collated.ling_feat,
            spectrogram=collated.spectrogram,
            spectrogram_lengths=collated.spectrogram_lengths,
            spectral_flatness=collated.spectral_flatness,
            ssl_feat=collated.ssl_feat,  # type: ignore
            ssl_feat_lengths=collated.ssl_feat_lengths,
            input_lengths=collated.transcription_lengths,
            output_lengths=collated.spectrogram_lengths,
        )

        _target: AlignerForwardTarget = AlignerForwardTarget(
            transcription=collated.transcription_id,
            input_lengths=collated.transcription_lengths,
            output_lengths=collated.spectrogram_lengths,
        )

        return _input.to(self.device), _target.to(self.device), batch.data_samples
