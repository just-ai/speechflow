import typing as tp

from dataclasses import dataclass

import torch

from torch import Tensor

from speechflow.data_pipeline.collate_functions.spectrogram_collate import (
    SpectrogramCollate,
    SpectrogramCollateOutput,
)
from speechflow.data_pipeline.collate_functions.utils import collate_sequence, collete_2d
from speechflow.data_pipeline.core.datasample import ToNumpy, ToTensor
from speechflow.data_pipeline.datasample_processors.data_types import (
    ProsodySSMLDataSample,
    TTSDataSample,
)
from speechflow.utils.pad_utils import pad_1d, pad_2d

__all__ = [
    "TTSCollate",
    "TTSCollateOutput",
    "TTSCollateWithPrompt",
    "TTSCollateOutputWithPrompt",
    "TTSCollateWithSSML",
    "TTSCollateOutputWithSSML",
    "LinguisticFeatures",
]


@dataclass
class LinguisticFeatures(ToTensor, ToNumpy):
    pos_tags: torch.LongTensor = None
    punctuation: torch.LongTensor = None
    token_ends: torch.LongTensor = None
    syntagma_ends: torch.LongTensor = None
    syntax: torch.LongTensor = None
    syntax_importance: torch.FloatTensor = None
    emphasis: torch.LongTensor = None
    intonation: torch.LongTensor = None
    breath_mask: torch.FloatTensor = None
    prosody: torch.LongTensor = None
    sil_mask: torch.BoolTensor = None

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def num_integer_features() -> int:
        return str(LinguisticFeatures.__annotations__).count("torch.LongTensor")

    @staticmethod
    def num_float_features() -> int:
        return str(LinguisticFeatures.__annotations__).count("torch.FloatTensor")

    @staticmethod
    def collate(batch: tp.List[TTSDataSample]) -> "LinguisticFeatures":
        pad_token_id = batch[0].pad_token_id
        batch_feat = {}
        if batch[0].ling_feat is not None:
            for seq_name in batch[0].ling_feat.keys():
                sequence = [sample.ling_feat.get(seq_name) for sample in batch]
                sequence, input_lens = pad_1d(sequence, pad_token_id)

                if seq_name == "sil_mask":
                    sequence = sequence.bool()

                batch_feat[seq_name] = sequence

        return LinguisticFeatures(**batch_feat)


@dataclass
class TTSCollateOutput(SpectrogramCollateOutput, TTSDataSample):
    transcription_lengths: Tensor = None
    ling_feat: LinguisticFeatures = None  # type: ignore
    xpbert_feat_lengths: Tensor = None
    lm_feat_lengths: Tensor = None
    num_words: Tensor = None
    num_synt: Tensor = None
    token_lengths: Tensor = None

    def __post_init__(self):
        super().__post_init__()


@dataclass
class TTSCollateOutputWithPrompt(TTSCollateOutput):
    prompt: tp.Optional[TTSCollateOutput] = None


@dataclass
class TTSCollateOutputWithSSML(TTSCollateOutput):
    pitch_modifier: Tensor = None
    volume_modifier: Tensor = None
    temp_modifier: Tensor = None


class TTSCollate(SpectrogramCollate):
    def collate(self, batch: tp.List[TTSDataSample]) -> TTSCollateOutput:  # type: ignore
        spec_collated = super().collate(batch)  # type: ignore
        collated = TTSCollateOutput(**spec_collated.to_dict())  # type: ignore

        pad_token_id = batch[0].pad_token_id
        spec_multiple = self.multiple_values.get("spectrogram")

        collated.transcription_text = [
            tuple(item.transcription_text)
            for item in batch
            if item.transcription_text is not None
        ]
        collated.transcription_id, collated.transcription_lengths = collate_sequence(
            batch, "transcription_id", pad_token_id
        )
        collated.transcription_id_by_frames, _ = collate_sequence(
            batch, "transcription_id_by_frames", pad_token_id, spec_multiple
        )
        collated.durations, _ = collate_sequence(batch, "durations")
        collated.invert_durations, _ = collate_sequence(
            batch, "invert_durations", multiple_values=spec_multiple
        )
        collated.word_lengths, collated.num_words = collate_sequence(
            batch, "word_lengths", pad_token_id
        )
        collated.synt_lengths, collated.num_synt = collate_sequence(
            batch, "synt_lengths", pad_token_id
        )

        collated.xpbert_feat, collated.xpbert_feat_lengths = collete_2d(
            batch, "xpbert_feat", multiple_values=self.multiple_values
        )

        collated.lm_feat, collated.lm_feat_lengths = collete_2d(
            batch, "lm_feat", multiple_values=self.multiple_values
        )

        collated.aggregated = {}
        if batch[0].aggregated is not None:
            for name in batch[0].aggregated.keys():
                data = [sample.aggregated[name] for sample in batch]
                pad_val = (
                    batch[0].get_param_val("mel_min_val")
                    if "mel" in name
                    else batch[0].get_param_val(f"{name}_pad", 0)
                )
                collated.aggregated[name], _ = (
                    pad_1d(data, pad_val=pad_val)
                    if data[0].ndim == 1
                    else pad_2d(data, data[0].shape[1], pad_val)
                )

        collated.ling_feat = LinguisticFeatures.collate(batch)
        return collated


class TTSCollateWithPrompt(TTSCollate):
    def collate(self, batch: tp.List[TTSDataSample]) -> TTSCollateOutputWithPrompt:  # type: ignore
        idx_neighbor = [x.additional_fields["neighbor_idx"] for x in batch]
        idx_neighbor = torch.Tensor(idx_neighbor).long()
        idx_right: tp.Iterable = torch.where(idx_neighbor == torch.roll(idx_neighbor, 1))[0]  # type: ignore
        idx_left: tp.Iterable = idx_right - 1  # type: ignore

        batch_prompt = [batch[n] for n in idx_left]
        batch_target = [batch[n] for n in idx_right]
        batch_tts_collated_prompt = super().collate(batch_prompt)
        batch_tts_collated_target = super().collate(batch_target)

        collated = TTSCollateOutputWithPrompt(
            **vars(batch_tts_collated_target), prompt=batch_tts_collated_prompt
        )
        return collated


class TTSCollateWithSSML(TTSCollate):
    def collate(self, batch: tp.List[ProsodySSMLDataSample]) -> TTSCollateOutputWithSSML:  # type: ignore
        tts_collated = super().collate(batch)  # type: ignore
        collated = TTSCollateOutputWithSSML(**tts_collated.to_dict())  # type: ignore

        collated.temp_modifier, _ = collate_sequence(batch, "temp_modifier", 1.0)
        collated.volume_modifier, _ = collate_sequence(batch, "volume_modifier", 1.0)
        collated.pitch_modifier, _ = collate_sequence(batch, "pitch_modifier", 1.0)
        return collated


if __name__ == "__main__":
    print("num_integer_ling_features:", LinguisticFeatures.num_integer_features())
    print("num_float_ling_features:", LinguisticFeatures.num_float_features())
