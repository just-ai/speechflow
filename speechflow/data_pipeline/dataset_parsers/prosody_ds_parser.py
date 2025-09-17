import typing as tp
import logging

from collections import defaultdict
from pathlib import Path

import torch
import multilingual_text_parser

from tqdm import tqdm
from transformers import AutoTokenizer

from speechflow.data_pipeline.core import BaseDSParser
from speechflow.data_pipeline.core.base_ds_parser import multi_transform
from speechflow.data_pipeline.core.parser_types import Metadata, MetadataTransform
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.data_types import (
    ProsodyPredictionDataSample,
)
from speechflow.io import AudioSeg, tp_PATH
from speechflow.logging import trace
from speechflow.utils.versioning import version_check

__all__ = ["ProsodyParser"]

LOGGER = logging.getLogger("root")


class ProsodyParser(BaseDSParser):
    """Dataset parser for sequence-to-sequence Prosody prediction task."""

    def __init__(
        self,
        preproc_fn: tp.Optional[tp.Sequence[MetadataTransform]] = None,
        memory_bound: bool = False,
        raise_on_converter_exc: bool = False,
        dump_path: tp.Optional[tp_PATH] = None,
        progress_bar: bool = True,
        tokenizer_name: str = None,
    ):
        super().__init__(
            preproc_fn,
            input_fields={"file_path", "sega", "label"},
            memory_bound=memory_bound,
            raise_on_converter_exc=raise_on_converter_exc,
            dump_path=dump_path,
            progress_bar=progress_bar,
        )

        if not tokenizer_name:
            raise RuntimeError("You have to specify tokenizer")
        else:
            self._tokenizer_name = tokenizer_name
            self._tokenizer = None
            self._pad_id = None

    def reader(
        self, file_path: Path, label: tp.Optional[str] = None
    ) -> tp.List[Metadata]:
        sega = AudioSeg.load(file_path)

        if "text_parser_version" in sega.meta:
            version_check(multilingual_text_parser, sega.meta["text_parser_version"])

        metadata = {"file_path": file_path, "sega": sega, "label": label}
        return [metadata]

    def converter(self, metadata: Metadata) -> tp.List[ProsodyPredictionDataSample]:
        if getattr(ProsodyParser, "_tokenizer", None) is None:
            with ProsodyParser.lock:
                tokenizer = AutoTokenizer.from_pretrained(
                    self._tokenizer_name, add_prefix_space=True, use_fast=True
                )
                pad_id = tokenizer.pad_token_id
                setattr(ProsodyParser, "_tokenizer", tokenizer)
                setattr(ProsodyParser, "_pad_id", pad_id)
        else:
            tokenizer = getattr(ProsodyParser, "_tokenizer")
            pad_id = getattr(ProsodyParser, "_pad_id")

        sents = metadata["sents"]
        prosody_labels, tokens = [], []
        for sent in sents:
            for idx, token in enumerate(sent.tokens):
                tokens.append(token.text)
                if idx == 0 or token.is_capitalize:
                    tokens[-1] = f"{tokens[-1][0].upper()}{tokens[-1][1:]}"

                prosody_labels.append(int(token.prosody) if token.prosody else -1)

        tokenized_inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )

        word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
        previous_word_idx = None
        binary_label_ids = []
        cat_label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                binary_label_ids.append(-100)
                cat_label_ids.append(-100)
            else:
                cat_label_ids.append(
                    prosody_labels[word_idx] if prosody_labels[word_idx] != -1 else -100
                )
                binary_label_ids.append(1 if prosody_labels[word_idx] != -1 else 0)
            previous_word_idx = word_idx

        datasample = ProsodyPredictionDataSample(
            file_path=metadata["file_path"],
            lang=sents[0].lang,
            attention_mask=tokenized_inputs["attention_mask"].flatten(),
            input_ids=tokenized_inputs["input_ids"].flatten(),
            binary=torch.Tensor(binary_label_ids).long(),
            category=torch.Tensor(cat_label_ids).long(),
            pad_id=pad_id,
        )
        return [datasample]

    @staticmethod
    @PipeRegistry.registry()
    def check_prosody_tags(metadata: Metadata) -> tp.List[Metadata]:
        """Skip samples that do not have any prosody tags.

        :param metadata: dictionary containing data of current sample
        :return: list of Metadata

        """
        sents = metadata["sents"]
        prosody_labels = []

        for sent in sents:
            for token in sent.tokens:
                if token.prosody and int(token.prosody) != -1:
                    prosody_labels.append(int(token.prosody))

        if len(prosody_labels) != 0:
            return [metadata]
        else:
            LOGGER.warning(trace("ProsodyParser", f"skip {str(metadata['file_path'])}"))
            return []

    @staticmethod
    @multi_transform
    @PipeRegistry.registry()
    def combine_texts(all_metadata: tp.List[Metadata]) -> tp.List[Metadata]:
        """Combines original texts and cuts them by 100 words to fit in a model."""

        metadata_by_audio = defaultdict(list)
        for metadata in tqdm(all_metadata, desc="Getting original audio"):
            try:
                meta = metadata["sega"].meta

                # TODO: support legacy models
                if "orig_wav_path" in meta:
                    orig_audio = meta["orig_wav_path"]
                else:
                    orig_audio = meta["orig_audio_path"]

                audio_chunk = meta["orig_audio_chunk"]
                if orig_audio in metadata_by_audio and any(
                    m["audio_chunk"] == audio_chunk for m in metadata_by_audio[orig_audio]
                ):
                    continue

                metadata_by_audio[orig_audio].append(
                    {
                        "metadata": metadata,
                        "audio_chunk": audio_chunk,
                    }
                )
            except Exception as e:
                LOGGER.error(trace("combine_texts", e))

        metadata_processed = []
        for orig_audio in tqdm(metadata_by_audio, desc="Combining texts"):
            try:
                sorted_samples = sorted(
                    metadata_by_audio[orig_audio], key=lambda d: d["audio_chunk"][0]
                )
                combined_metadata = {
                    "file_path": Path(orig_audio),
                    "sents": [],
                    "label": sorted_samples[0]["metadata"]["label"],
                }
                tokens_num = 0
                for sample in sorted_samples:
                    combined_metadata["sents"].append(sample["metadata"]["sega"].sent)
                    tokens_num += len(sample["metadata"]["sega"].sent.tokens)
                    if tokens_num > 100:
                        metadata_processed.append(combined_metadata)
                        tokens_num = 0
                        combined_metadata = {
                            "file_path": Path(orig_audio),
                            "sents": [],
                            "label": sorted_samples[0]["metadata"]["label"],
                        }
                metadata_processed.append(combined_metadata)
            except Exception as e:
                LOGGER.error(trace("combine_texts", e))

        return metadata_processed
