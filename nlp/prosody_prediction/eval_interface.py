import typing as tp

from pathlib import Path

import numpy as np
import torch

from multilingual_text_parser import Doc, TextParser
from transformers import AutoTokenizer

from nlp import prosody_prediction
from nlp.prosody_prediction.data_types import ProsodyPredictionOutput
from speechflow.data_pipeline.core import Batch, PipelineComponents
from speechflow.data_pipeline.datasample_processors.data_types import (
    ProsodyPredictionDataSample,
)
from speechflow.data_pipeline.datasample_processors.text_processors import TextProcessor
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.fs import get_module_dir
from speechflow.utils.init import init_class_from_config
from speechflow.utils.seed import set_numpy_seed

__all__ = ["ProsodyPredictionInterface"]

PITCH_DOWN_CONTURS = ("1", "2", "4", "7", "8")
PITCH_DOWN_PUNCTUATIONS = (",", ".", ";", ":", "-")


class ProsodyPredictionInterface:
    def __init__(
        self,
        ckpt_path: tp.Union[str, Path],
        device: str = "cpu",
        lang: str = "EN",
        text_parser: tp.Optional[tp.Dict[str, TextParser]] = None,
        ckpt_preload: tp.Optional[dict] = None,
    ):
        if ckpt_preload is None:
            checkpoint = ExperimentSaver.load_checkpoint(ckpt_path)
        else:
            checkpoint = ckpt_preload

        cfg_data, cfg_model = ExperimentSaver.load_configs_from_checkpoint(checkpoint)
        self.device = torch.device(device)

        bert_path = checkpoint["params"]["model_name"]
        bert_path = bert_path.replace("/src/libs/text_parser/text_parser/", "")
        bert_path = get_module_dir("text_parser") / bert_path
        checkpoint["params"]["model_name"] = bert_path.as_posix()

        tokenizer_path = cfg_data["parser"]["tokenizer_name"]
        tokenizer_path = tokenizer_path.replace("/src/libs/text_parser/text_parser/", "")
        tokenizer_path = get_module_dir("text_parser") / tokenizer_path
        cfg_data["parser"]["tokenizer_name"] = tokenizer_path.as_posix()

        model_cls = getattr(prosody_prediction, cfg_model["model"]["type"])
        self.model = model_cls(checkpoint["params"])
        self.model.eval()

        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model.to(device)

        self.pipeline = PipelineComponents(cfg_data, "test")

        BatchProcessor = getattr(prosody_prediction, cfg_model["batch"]["type"])
        self.batch_processor = init_class_from_config(
            BatchProcessor, cfg_model["batch"]
        )()
        self.batch_processor.device = self.device

        if text_parser is None:
            self.text_parser = {lang: TextParser(lang=lang)}
        else:
            self.text_parser = text_parser

        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg_data["parser"]["tokenizer_name"], add_prefix_space=True, use_fast=True
        )
        self._pad_id = self._tokenizer.pad_token_id
        self._softmax = torch.nn.Softmax(dim=2)
        self._lang = lang

        self.service_tokens = (
            TextProcessor.pad,
            TextProcessor.bos,
            TextProcessor.eos,
            TextProcessor.sil,
            TextProcessor.unk,
        )

    @torch.inference_mode()
    def evaluate(self, batch: Batch) -> ProsodyPredictionOutput:
        inputs, _, _ = self.batch_processor(batch)
        outputs = self.model(inputs)
        return outputs

    def _prepare_text(
        self,
        doc: Doc,
        seed: int = 0,
    ) -> tp.List[ProsodyPredictionDataSample]:
        datasamples = []
        sents = doc.sents
        tokens = []
        seed_by_words = []

        for idx, sent in enumerate(sents):
            for token in sent.tokens:
                tokens.append(token.text)

                user_seed = seed
                if token.modifiers and "intonation" in token.modifiers:
                    user_seed = token.modifiers["intonation"].get("seed", "")
                    if user_seed.isdigit():
                        user_seed = int(user_seed)

                seed_by_words.append(user_seed)

            if len(tokens) > 100 or idx == len(sents) - 1:
                tokenized_inputs = self._tokenizer(
                    tokens,
                    is_split_into_words=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=512,
                )
                word_ids = (
                    tokenized_inputs.word_ids()
                )  # Map tokens to their respective word.
                datasample = ProsodyPredictionDataSample(
                    lang=sents[0].lang,
                    attention_mask=tokenized_inputs["attention_mask"].flatten(),
                    input_ids=tokenized_inputs["input_ids"].flatten(),
                    pad_id=self._pad_id,
                    word_ids=word_ids,
                    seed_by_words=seed_by_words,
                )
                datasamples.append(datasample)
                tokens = []
                seed_by_words = []

        return datasamples

    def _assign_prosody_tags(
        self,
        doc: Doc,
        output: ProsodyPredictionOutput,
        datasamples: tp.List[ProsodyPredictionDataSample],
        tres_bin: tp.Optional[float] = None,
        predict_proba: bool = True,
    ) -> Doc:
        predicted_tags = []
        if tres_bin is None:
            tres_bin = 0.5

        for datasample in datasamples:
            word_ids = datasample.word_ids
            pred_binary = self._softmax(output.binary)[0, :, 1]
            pred_category = self._softmax(output.category)[0]
            n_classes = pred_category.shape[-1]

            idx = 0
            previous = None
            for i, word_id in enumerate(word_ids):
                if word_id is not None and previous != word_id:
                    if pred_binary[i].item() >= tres_bin:
                        if predict_proba:
                            set_numpy_seed(datasample.seed_by_words[idx])
                            p_index = np.random.choice(
                                np.arange(n_classes),
                                p=pred_category[i].cpu().numpy().ravel(),
                            )
                        else:
                            p_index = pred_category[i].argmax(-1).item()
                        predicted_tags.append(str(p_index))
                    else:
                        predicted_tags.append("-1")
                    idx += 1
                previous = word_id

        idx = 0
        for sent_id, sent in enumerate(doc.sents):
            for token_id, token in enumerate(sent.tokens):
                if (
                    token.prosody is None
                    and token.text not in self.service_tokens
                    and token.emphasis != "accent"
                ):
                    prosody_tag = predicted_tags[idx]
                    if (
                        prosody_tag != -1
                        and doc.sents[sent_id].tokens[token_id]
                        != doc.sents[sent_id].tokens[-1]
                    ):
                        if (
                            doc.sents[sent_id].tokens[token_id + 1].text
                            in PITCH_DOWN_PUNCTUATIONS
                        ):
                            if prosody_tag not in PITCH_DOWN_CONTURS:
                                prosody_tag = str(np.random.choice(PITCH_DOWN_CONTURS))

                    doc.sents[sent_id].tokens[token_id].prosody = prosody_tag

                idx += 1

        return doc

    def predict(
        self,
        text: tp.Union[Doc, str],
        tres_bin: tp.Optional[float] = None,
        predict_proba: bool = True,
        seed: int = 0,
    ) -> Doc:
        if isinstance(text, str):
            doc = self.text_parser[self._lang].process(Doc(text))
        else:
            doc = text

        samples = self._prepare_text(doc, seed)
        batch = self.pipeline.datasample_to_batch(samples)
        outputs = self.evaluate(batch)

        doc = self._assign_prosody_tags(doc, outputs, samples, tres_bin, predict_proba)
        return doc
