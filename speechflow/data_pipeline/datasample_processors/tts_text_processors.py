import typing as tp
import logging
import itertools

from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import torch

from multilingual_text_parser.data_types import Doc, Sentence, Token, TokenUtils
from transformers import AutoModel, AutoTokenizer

from speechflow.data_pipeline.core.base_ds_processor import BaseDSProcessor
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.data_types import TextDataSample
from speechflow.io import AudioSeg
from speechflow.logging import trace
from speechflow.utils.fs import get_module_dir
from speechflow.utils.init import lazy_initialization
from speechflow.utils.profiler import Profiler

__all__ = [
    "load_audio_segmentation",
    "TTSTextProcessor",
    "XPBertProcessor",
    "LMProcessor",
]

LOGGER = logging.getLogger("root")


class ZeroSilTokensError(Exception):
    """Exception raised in TextProcessor if allow_zero_sil parameter is False."""

    pass


@PipeRegistry.registry(
    inputs={"file_path"}, outputs={"word_timestamps", "phoneme_timestamps"}
)
def load_audio_segmentation(ds: TextDataSample):
    sega = AudioSeg.load(ds.file_path)

    sega.ts_bos = ds.audio_chunk.begin
    sega.ts_eos = ds.audio_chunk.end
    sega.audio_chunk.begin = ds.audio_chunk.begin
    sega.audio_chunk.end = ds.audio_chunk.end
    word_ts, phoneme_ts = sega.get_timestamps(relative=True)

    ds.sent = sega.sent
    ds.word_timestamps = word_ts
    ds.phoneme_timestamps = phoneme_ts
    return ds


class TTSTextProcessor(BaseDSProcessor):
    # service tokens
    pad = "<PAD>"
    bos = "<BOS>"
    eos = "<EOS>"
    sil = "<SIL>"
    unk = "<UNK>"
    sntgm = "<SNTGM>"
    eosntgm = "<EOSNTGM>"
    tkn = "<TKN>"
    eotkn = "<EOTKN>"
    unkpos = "<UNK_POS>"
    unkpunct = "<UNK_PUNCT>"
    emphasis = "<EMPHSIS>"
    no_emphasis = "<NOEMPHSIS>"
    breath = "<BREATH>"
    no_breath = "<NOBREATH>"

    def __init__(
        self,
        lang: str,
        ipa_mode: tp.Literal["full", "truncated", "multiline"] = "multiline",
        words_level: bool = False,
        add_service_tokens: bool = False,
        allow_zero_sil: bool = True,
        num_prosodic_classes: tp.Optional[int] = None,
        ignore_ling_feat: tp.Optional[tp.List[str]] = None,
    ):
        from multilingual_text_parser.parser import TextParser

        super().__init__()
        self.logging_params(self.get_config_from_locals())

        text_parser = TextParser(lang=lang, cfg={"pipe": []})

        self.lang = lang
        self.ipa_mode = ipa_mode
        self.is_ipa_phonemes = text_parser.is_ipa_phonemes

        if self.is_ipa_phonemes and ipa_mode == "multiline":
            self.num_symbols_per_phoneme_token = text_parser.num_symbols_per_phoneme
        else:
            self.num_symbols_per_phoneme_token = 1

        if num_prosodic_classes is None:
            num_prosodic_classes = 0

        self.service_tokens = (self.pad, self.bos, self.eos, self.sil, self.unk)
        self.phoneme_tokens = text_parser.phonemes
        self.punctuation_tokens = text_parser.punctuation
        self.pos_tokens = text_parser.pos
        self.rel_tokens = text_parser.rel
        self.intonation_tokens = text_parser.intonation
        self.intonation_contour_tokens = tuple(
            [i + 1 for i in range(num_prosodic_classes)] + [-1]
        )
        self.additional_tokens = (
            self.sntgm,
            self.eosntgm,
            self.tkn,
            self.eotkn,
            self.unkpos,
            self.unkpunct,
            self.emphasis,
            self.no_emphasis,
            self.breath,
            self.no_breath,
        )
        self.additional_sil_tokens = tuple(
            [f"<{p}>{self.sil}" for p in self.punctuation_tokens]
        )

        if self.is_ipa_phonemes and self.ipa_mode == "truncated":
            tokens = ["ˈ" + ph for ph in text_parser.ipa_phonemes]
            tokens += ["ˌ" + ph for ph in text_parser.ipa_phonemes]
            tokens += [ph + "ʲ" for ph in text_parser.ipa_phonemes]
            tokens += [ph + "ː" for ph in text_parser.ipa_phonemes]
            tokens += [ph + "ɪ" for ph in text_parser.ipa_phonemes]
            tokens += [ph + "ʊ" for ph in text_parser.ipa_phonemes]
            # tokens += [ph + "ʃ" for ph in text_parser.ipa_phonemes]
            # tokens += [ph + "ɕ" for ph in text_parser.ipa_phonemes]
            # tokens += [ph + "ʒ" for ph in text_parser.ipa_phonemes]
            self.additional_ipa_tokens = tuple(tokens)
        else:
            self.additional_ipa_tokens = ()

        self.ru2ipa = text_parser.ru2ipa
        self.en2ipa = text_parser.en2ipa

        self.alphabet = self.service_tokens + self.phoneme_tokens
        self._expand_alphabet(self.punctuation_tokens)
        self._expand_alphabet(self.pos_tokens)
        self._expand_alphabet(self.rel_tokens)
        self._expand_alphabet(self.intonation_tokens)
        self._expand_alphabet(self.intonation_contour_tokens)
        self._expand_alphabet(self.additional_tokens)
        self._expand_alphabet(self.additional_sil_tokens)
        self._expand_alphabet(self.additional_ipa_tokens)

        # Mappings from symbol to numeric ID and vice versa:
        self._symbol_to_id = {s: i for i, s in enumerate(self.alphabet)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.alphabet)}

        self._words_level = words_level
        self._add_service_tokens = add_service_tokens
        self._allow_zero_sil = allow_zero_sil
        self._ignore_ling_feat = ignore_ling_feat

        self._float_features = ["syntax_importance", "breath_mask"]

    def _expand_alphabet(self, new_symbols: tp.Tuple[tp.Union[str, int], ...]):
        self.alphabet += new_symbols

    def symbol_to_id(self, symbol: str) -> int:
        return self._symbol_to_id[symbol]

    def id_to_symbol(self, id: int) -> str:
        return self._id_to_symbol[id]

    @property
    def alphabet_size(self) -> int:
        return len(self.alphabet)

    @staticmethod
    def is_service_symbol(symbol: str) -> bool:
        service_tokens = (
            TTSTextProcessor.pad,
            TTSTextProcessor.bos,
            TTSTextProcessor.eos,
            TTSTextProcessor.sil,
            TTSTextProcessor.unk,
        )
        return any(t in symbol for t in service_tokens)

    @PipeRegistry.registry(
        inputs={"sent"}, outputs={"transcription_text", "transcription_id", "ling_feat"}
    )
    def process(self, ds: TextDataSample) -> TextDataSample:
        ds = super().process(ds)

        if self.lang != "MULTILANG" and ds.sent.lang != self.lang:
            raise RuntimeError(
                f"The TextParser does not match the sentence {ds.sent.lang} language."
            )

        symbols_by_word = ds.sent.get_phonemes()
        symbols = list(itertools.chain.from_iterable(symbols_by_word))

        if self.is_ipa_phonemes:
            symbols = self.phons2ipa(ds.sent.lang, symbols)

        if self._words_level:
            ling_feat, word_lens, synt_lens = self._process_words_level(ds)
        else:
            ling_feat, word_lens, synt_lens = self._process_phoneme_level(ds, symbols)

        ling_feat_id = self._apply_ling_feat_tokenizer(ling_feat)

        (
            tokens_id,
            symbols,
            ling_feat_id,
            word_lens,
            synt_lens,
        ) = self._apply_phoneme_tokenizer(symbols, ling_feat_id, word_lens, synt_lens)

        (
            symbols,
            tokens_id,
            ling_feat_id,
            word_lens,
            synt_lens,
            ds.sent,
        ) = self._assign_service_tokens(
            symbols,
            tokens_id,
            ling_feat_id,
            word_lens,
            synt_lens,
            ds.sent,
        )

        tokens_id, ling_feat_id = self._to_numpy(tokens_id, ling_feat_id)

        self._set_word_lengths(ds, word_lens, synt_lens)

        if not self._words_level:
            ds.transcription_text = symbols
            ds.transcription_id = tokens_id
            assert ds.word_lengths.sum() == len(ds.transcription_text)
            assert ds.word_lengths.sum() == ds.synt_lengths.sum()

        ds.ling_feat = ling_feat_id
        ds.pad_token_id = self._symbol_to_id[self.pad]
        ds.sil_token_id = self._symbol_to_id[self.sil]

        ds.transform_params["TTSTextProcessor"] = {
            "ipa_phonemes": self.is_ipa_phonemes,
            "ipa_mode": self.ipa_mode,
        }
        return ds

    def get_prosody(self, ds):
        """Extract prosody features."""
        prosody = []
        for token in ds.sent.tokens:
            if not token.is_punctuation:
                prosody.append(
                    int(token.prosody) + 1
                    if hasattr(token, "prosody")
                    and token.prosody
                    and token.prosody not in ["undefined", "-1"]
                    and token.emphasis != "accent"
                    else -1
                )
        return prosody

    def get_syntax(self, ds):
        """Extract features from SLOVNET."""
        full_rels, head_ids = [], []
        for token in ds.sent.tokens:
            if not token.is_punctuation:
                if token.rel:
                    full_rels.append(token.rel)
                    head_ids.append(token.head_id)
                else:
                    full_rels.append(self.unk)
                    head_ids.append("-1")

        head_counts = Counter()
        for token in ds.sent.tokens:
            if not token.is_punctuation and token.head_id:
                head_counts[token.head_id] += 1

        full_head_counts = []
        for token in ds.sent.tokens:
            if not token.is_punctuation:
                if token.id:
                    full_head_counts.append(head_counts[token.id])
                else:
                    full_head_counts.append(0)

        return full_rels, full_head_counts

    def phons2ipa(self, lang: str, symbols: tp.List[str]) -> tp.List[str]:
        ipa_map = None
        if lang == "RU":
            ipa_map = self.ru2ipa
        elif lang == "EN":
            ipa_map = self.en2ipa

        if ipa_map is not None:
            return [
                ipa_map[phoneme] if phoneme in ipa_map else phoneme for phoneme in symbols
            ]
        else:
            return symbols

    @staticmethod
    def _intonation_model(sentence: Sentence) -> str:
        if "?" in sentence.text:
            intonation_type = "quest_type0"
        elif "!" in sentence.text:
            intonation_type = "excl_type"
        else:
            intonation_type = "dot_type"
        return intonation_type

    def _process_emphasis(self, sentence: Sentence, token_lens):
        emph_labels = []
        for synt in sentence.syntagmas:
            for token in TokenUtils.get_word_tokens(synt.tokens):
                if token.emphasis == "accent":
                    emph_labels.append(self.emphasis)
                else:
                    emph_labels.append(self.no_emphasis)

        return emph_labels

    def _process_breath(self, sentence: Sentence):
        phonemes = sentence.get_phonemes()
        meta = TokenUtils.get_attr(sentence.tokens, attr_names=["meta"])["meta"]
        breath_mask = []
        for idx, (m, ph) in enumerate(zip(meta, phonemes)):
            if (
                self.sil not in ph[0]
                or idx + 1 == len(phonemes)
                or phonemes[idx + 1][0] == self.eos
            ):
                breath_mask.append([-10.0] * len(ph))
            else:
                if "noise_level" in m:
                    breath_mask.append(m["noise_level"])
                else:
                    breath_mask.append([-3.0] * len(ph))

        return tuple(itertools.chain(*breath_mask))

    def _process_phoneme_level(self, ds, symbols):
        word_lens, synt_lens, token_lens, lens_per_postag = self._count_phoneme_lens(
            ds.sent
        )
        emph_labels = self._process_emphasis(ds.sent, token_lens)
        breath_mask = self._process_breath(ds.sent)

        rels, head_counts = self.get_syntax(ds)
        expanded_rels = self._assign_tags_to_phoneme(list(zip(rels, token_lens)))
        expanded_head_count = self._assign_tags_to_phoneme(
            list(zip(head_counts, token_lens))
        )

        prosody = self.get_prosody(ds)
        prosody = self._assign_tags_to_phoneme(list(zip(prosody, token_lens)))

        syntamas_ends = self._assign_ends_of_items(synt_lens, self.sntgm, self.eosntgm)
        token_ends = self._assign_ends_of_items(token_lens, self.tkn, self.eotkn)
        pos_tags = self._assign_tags_to_phoneme(lens_per_postag)
        punctuation = self._assign_punctuation_to_phoneme(ds.sent)
        sil_mask = np.array([self.sil if self.sil in s else self.pad for s in symbols])
        emphasis = self._assign_tags_to_phoneme(list(zip(emph_labels, token_lens)))
        expanded_intonation = [self._intonation_model(ds.sent)] * len(emphasis)

        if not self._allow_zero_sil and len(sil_mask[sil_mask == self.sil]) == 0:
            raise ZeroSilTokensError("No sil tokens in the sentence")

        ling_feat = {
            "sil_mask": sil_mask,
            "token_ends": token_ends,
            "syntagma_ends": syntamas_ends,
            "pos_tags": pos_tags,
            "punctuation": punctuation,
            "emphasis": emphasis,
            "intonation": expanded_intonation,
            "syntax": expanded_rels,
            "syntax_importance": expanded_head_count,
            "breath_mask": breath_mask,
            "prosody": prosody,
        }

        if (
            hasattr(ds, "aggregated")
            and isinstance(ds.aggregated, tp.MutableMapping)
            and ds.aggregated.get("word_durations") is not None
        ):
            word_durations = ds.aggregated.get("word_durations")
            word_durations = self._assign_tags_to_phoneme(
                list(zip(word_durations, token_lens))
            )
            ds.aggregated["word_durations"] = np.asarray(word_durations)

        return ling_feat, word_lens, synt_lens

    def _process_words_level(self, ds):
        word_lens, synt_lens, lens_per_postag = self._count_token_lens(ds.sent)

        rels, head_counts = self.get_syntax(ds)

        prosody = self.get_prosody(ds)

        syntamas_ends = self._assign_ends_of_items(synt_lens, self.sntgm, self.eosntgm)
        pos_tags = self._assign_tags_to_phoneme(lens_per_postag)
        punctuation = self._assign_punctuation_to_token(ds.sent)
        sil_mask = []
        for token in ds.sent.tokens:
            if not token.is_punctuation:
                if self.sil in token.text:
                    sil_mask.append(self.sil)
                else:
                    sil_mask.append(self.pad)
        sil_mask = np.array(sil_mask)
        if not self._allow_zero_sil and self.sil not in sil_mask:
            raise ZeroSilTokensError("No sil tokens in the sentence")

        ling_feat = {
            "sil_mask": sil_mask,
            "syntagma_ends": syntamas_ends,
            "pos_tags": pos_tags,
            "punctuation": punctuation,
            "syntax": rels,
            "syntax_importance": head_counts,
            "prosody": prosody,
        }
        return ling_feat, word_lens, synt_lens

    def _apply_ling_feat_tokenizer(
        self, ling_feat: tp.Dict[str, tp.Sequence]
    ) -> tp.Dict[str, tp.Sequence]:
        ling_feat_seq = {}

        if not self._words_level and self._ignore_ling_feat is not None:
            for name in self._ignore_ling_feat:
                if name not in ling_feat:
                    raise KeyError(f"Linguistic feature '{name}' not found!")

        # encode features
        for key, field in ling_feat.items():
            if self._ignore_ling_feat is not None and key in self._ignore_ling_feat:
                continue

            if key not in self._float_features:  # numerical, doesn't need to be encoded
                ling_feat_seq[key] = self._symbols_to_sequence(field)
            else:
                ling_feat_seq[key] = field

        return ling_feat_seq

    def _apply_phoneme_tokenizer(self, symbols, ling_feat, word_lens, synt_lens):
        tokens_id = []

        if self.num_symbols_per_phoneme_token == 1:
            if self.is_ipa_phonemes:
                if self.ipa_mode == "full":
                    symbols_expand = []
                    ling_feat_expand = defaultdict(list)
                    for i, ph in enumerate(symbols):
                        if isinstance(ph, tuple):
                            symbols_expand += list(ph)
                            for k in ling_feat.keys():
                                ling_feat_expand[k] += [ling_feat[k][i]] * len(ph)
                        else:
                            symbols_expand.append(ph)
                            for k in ling_feat.keys():
                                ling_feat_expand[k].append(ling_feat[k][i])

                    word_lens_expand = deepcopy(word_lens)
                    synt_lens_expand = deepcopy(word_lens)
                    for lens in [word_lens_expand, synt_lens_expand]:
                        cum_lens = np.cumsum(np.asarray([0] + lens))
                        for i, (a, b) in enumerate(zip(cum_lens[:-1], cum_lens[1:])):
                            num_ph = sum(
                                len(ph) if isinstance(ph, tuple) else 1
                                for ph in symbols[a:b]
                            )
                            lens[i] = num_ph

                    symbols = symbols_expand
                    ling_feat = ling_feat_expand
                    word_lens = word_lens_expand
                    synt_lens = synt_lens_expand
                elif self.ipa_mode == "truncated":
                    symbols = tuple(
                        "".join(ph[:2]) if isinstance(ph, tuple) else ph for ph in symbols
                    )
                else:
                    raise NotImplementedError(f"ipa_mode='{self.ipa_mode}'")

            tokens_id = self._symbols_to_sequence(symbols)
        else:
            for i in range(self.num_symbols_per_phoneme_token):
                seq = []
                for s in symbols:
                    if isinstance(s, tuple):
                        seq.append(s[i] if len(s) > i else TTSTextProcessor.unk)
                    else:
                        seq.append(s if i == 0 else TTSTextProcessor.unk)

                tokens_id.append(self._symbols_to_sequence(seq))

        return tokens_id, symbols, ling_feat, word_lens, synt_lens

    def _symbols_to_sequence(self, symbols: tp.Sequence[str]) -> tp.List[int]:
        return [self._symbol_to_id[self._is_symbol_in_alphabet(s)] for s in symbols]

    def _is_symbol_in_alphabet(self, s: str) -> str:
        if not s:
            return self.unk

        if s not in self._symbol_to_id:
            LOGGER.warning(trace(self, message=f"symbol [{s}] not in alphabet!"))
            return self.unk
        else:
            return s

    @staticmethod
    def _assign_tags_to_phoneme(lens_per_tag: tp.List[tuple]) -> tp.List:
        """For every phoneme assign a tag.

        Parameters:
        -----------
        lens_per_tag: list of tuples.
            Every tuple must contain two elements (tag, length) with integer
            length for every tag value.

        """

        res = [[tag] * length for tag, length in lens_per_tag]
        return list(itertools.chain.from_iterable(res))

    def _assign_punctuation_to_token(self, sentence: Sentence) -> tp.Tuple:
        """For every token assign corresponding punctuation symbol."""
        punc_level = []
        in_quote = False
        prev_punct = current_punct = self.unkpunct
        all_tokens = tuple(
            ["T" if not token.is_punctuation else token.text for token in sentence.tokens]
        )
        all_tokens = tuple(itertools.chain.from_iterable(all_tokens))
        for i, symbol in enumerate(all_tokens[::-1]):
            if symbol not in self.punctuation_tokens:
                punc_level.append(current_punct)
                if current_punct == "(":
                    # brackets do not spread intonation further after closure.
                    current_punct = self.unkpunct
            else:
                if symbol == '"':
                    if in_quote:
                        current_punct = prev_punct
                    else:
                        in_quote = True
                        prev_punct, current_punct = current_punct, symbol
                elif symbol == "(":
                    current_punct = symbol
                else:
                    prev_punct, current_punct = current_punct, symbol

        return tuple(punc_level[::-1])

    def _assign_punctuation_to_phoneme(self, sentence: Sentence) -> tp.Tuple:
        """For every phoneme assign corresponding punctuation symbol."""
        punc_level = []
        in_quote = False
        prev_punct = current_punct = self.unkpunct
        all_phonemes = tuple(
            [
                token.phonemes if not token.is_punctuation else token.text
                for token in sentence.tokens
            ]
        )
        all_phonemes = tuple(itertools.chain.from_iterable(all_phonemes))
        for i, symbol in enumerate(all_phonemes[::-1]):
            if symbol not in self.punctuation_tokens:
                punc_level.append(current_punct)
                if current_punct == "(":
                    # brackets do not spread intonation further after closure.
                    current_punct = self.unkpunct
            else:
                if symbol == '"':
                    if in_quote:
                        current_punct = prev_punct
                    else:
                        in_quote = True
                        prev_punct, current_punct = current_punct, symbol
                elif symbol == "(":
                    current_punct = symbol
                else:
                    prev_punct, current_punct = current_punct, symbol

        return tuple(punc_level[::-1])

    @staticmethod
    def _assign_ends_of_items(
        lens: tp.List[int], in_symbol: str, end_symbol: str
    ) -> tp.List[tp.Any]:
        """For every phoneme assign `end_symbol` if phoneme is in the end of an item, else
        `in_symbol`.

        Parameters
        ----------
        lens : array-like
            contains lengths of every item in a sequence.

        """
        res = [
            [in_symbol] * max((length - 1), 0) + ([end_symbol] if length > 0 else [])
            for length in lens
        ]
        return list(itertools.chain.from_iterable(res))

    def _assign_service_tokens(
        self,
        symbols,
        tokens_id,
        ling_feat,
        word_lens,
        synt_lens,
        sentence,
    ) -> tp.Tuple[
        tp.Tuple[str, ...], tp.List[int], tp.Dict, tp.List[int], tp.List[int], Sentence
    ]:
        bos_id = self._symbol_to_id[self.bos]
        eos_id = self._symbol_to_id[self.eos]

        if self._add_service_tokens:
            symbols = [self.bos] + symbols + [self.eos]

            if isinstance(tokens_id[0], list):
                for i in range(len(tokens_id)):
                    tokens_id[i] = [bos_id] + tokens_id[i] + [eos_id]
            else:
                tokens_id = [bos_id] + tokens_id + [eos_id]

            for key, field in ling_feat.items():
                if isinstance(field[0], tp.List):
                    for i in range(len(field)):
                        ling_feat[key][i] = [bos_id] + field[i] + [eos_id]
                elif isinstance(field[0], int):
                    ling_feat[key] = [bos_id] + field + [eos_id]
                else:
                    ling_feat[key] = (-1,) + field + (-1,)

            word_lens = [1] + word_lens + [1]
            synt_lens[0] += 1
            synt_lens[-1] += 1

            bos_token = Token(TTSTextProcessor.bos)
            bos_token.phonemes = (TTSTextProcessor.bos,)
            eos_token = Token(TTSTextProcessor.eos)
            eos_token.phonemes = (TTSTextProcessor.eos,)
            sentence.tokens = [bos_token] + sentence.tokens + [eos_token]

            syntagmas = sentence.syntagmas
            syntagmas[0].tokens = [bos_token] + syntagmas[0].tokens
            syntagmas[-1].tokens = syntagmas[-1].tokens + [eos_token]
            sentence.syntagmas = syntagmas
        else:
            for key, field in ling_feat.items():
                if key == "prosody":
                    continue
                if symbols[0] == self.bos:
                    if isinstance(field[0], tp.List):
                        for i in range(len(field)):
                            ling_feat[key][i][0] = bos_id
                    elif isinstance(field[0], int):
                        ling_feat[key][0] = bos_id
                    else:
                        ling_feat[key] = (-1,) + ling_feat[key][1:]
                if symbols[-1] == self.eos:
                    if isinstance(field[0], tp.List):
                        for i in range(len(field)):
                            ling_feat[key][i][-1] = eos_id
                    elif isinstance(field[0], int):
                        ling_feat[key][-1] = eos_id
                    else:
                        ling_feat[key] = ling_feat[key][:-1] + (-1,)

        return symbols, tokens_id, ling_feat, word_lens, synt_lens, sentence

    def _to_numpy(self, tokens_id, ling_feat_id=None):
        tokens_np = np.asarray(tokens_id, dtype=np.int64)
        if tokens_np.ndim == 2:
            tokens_np = tokens_np.T

        if ling_feat_id is not None:
            ling_feat_np = {}
            for key, field in ling_feat_id.items():
                dtype = (
                    np.float32
                    if (key in self._float_features and key != "prosody")
                    else np.int64
                )
                ling_feat_np[key] = np.asarray(field, dtype=dtype)
                if ling_feat_np[key].ndim == 2:
                    ling_feat_np[key] = ling_feat_np[key].T

                # check all features have equal lengths
                if self._words_level:
                    target_len = ling_feat_np["sil_mask"].shape[0]
                else:
                    target_len = tokens_np.shape[0]

                assert (
                    target_len == ling_feat_np[key].shape[0]
                ), "length sequence is mismatch!"
        else:
            ling_feat_np = None

        return tokens_np, ling_feat_np

    def _count_token_lens(
        self, sentence: Sentence
    ) -> tp.Tuple[tp.List, tp.List, tp.List]:
        """Count token lengths per syntagma."""
        word_lens, synt_lens, lens_per_postag = [], [], []
        for synt in sentence.syntagmas:
            tkn_in_syntagm = 0
            for token in synt.tokens:
                if not token.is_punctuation:
                    tkn_in_syntagm += 1
                    if self.sil not in token.text:
                        pos = token.pos if token.pos in self.pos_tokens else self.unkpos
                        lens_per_postag.append((pos, 1))
                    else:
                        lens_per_postag.append((self.sil, 1))
                    word_lens.append(1)
            synt_lens.append(tkn_in_syntagm)
        return word_lens, synt_lens, lens_per_postag

    def _count_phoneme_lens(
        self, sentence: Sentence
    ) -> tp.Tuple[tp.List, tp.List, tp.List, tp.List]:
        """Count phonemes lengths per token, syntagma and pos-tag."""
        word_lens, synt_lens, token_lens, lens_per_postag = [], [], [], []
        for synt in sentence.syntagmas:
            ph_in_syntagm = 0
            for token in synt.tokens:
                if not token.is_punctuation:
                    ph_in_token = token.num_phonemes
                    token_lens.append(ph_in_token)
                    ph_in_syntagm += ph_in_token
                    if self.sil not in token.text:
                        pos = token.pos if token.pos in self.pos_tokens else self.unkpos
                        lens_per_postag.append((pos, ph_in_token))
                    else:
                        lens_per_postag.append((self.sil, ph_in_token))
                    word_lens.append(ph_in_token)
            synt_lens.append(ph_in_syntagm)
        return word_lens, synt_lens, token_lens, lens_per_postag

    @staticmethod
    def _set_word_lengths(ds: TextDataSample, word_lens, synt_lens):
        ds.word_lengths = np.asarray(word_lens, dtype=np.int64)
        ds.synt_lengths = np.asarray(synt_lens, dtype=np.int64)

        if not hasattr(ds, "aggregated"):
            return

        invert = []
        for p in ds.word_lengths:
            invert += [1 / p] * p

        ds.aggregated = {} if ds.aggregated is None else ds.aggregated
        ds.aggregated["word_lengths"] = ds.word_lengths
        ds.aggregated["word_invert_lengths"] = np.array(invert, dtype=np.float32)


class XPBertProcessor(BaseDSProcessor):  # https://github.com/VinAIResearch/XPhoneBERT
    def __init__(self, device: str = "cpu", model_name: str = "vinai/xphonebert-base"):
        super().__init__(device=device)

        self.lang = "MULTILANG"
        self._model_name = model_name
        self._tokenizer = None
        self._model = None

    def init(self):
        super().init()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, add_prefix_space=True
        )
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.to(self.device).eval()

    @PipeRegistry.registry(inputs={"sent", "transcription_text"}, outputs={"xpbert_feat"})
    @lazy_initialization
    def process(self, ds: TextDataSample) -> TextDataSample:
        ds = super().process(ds)

        if not ds.transform_params["TTSTextProcessor"]["ipa_phonemes"]:
            raise RuntimeError(
                "Please use multilingual mode in TTS Text Processor for get phonemes in IPA format."
            )

        text_proc = TTSTextProcessor(lang=self.lang)
        eotkn_id = text_proc.symbol_to_id(text_proc.eotkn)

        tokens_id, phoneme_emb_pos = self._apply_tokenizer(
            ds.transcription_text, ds.ling_feat["token_ends"], eotkn_id
        )

        with torch.inference_mode():
            embeddings = (
                self._model(torch.LongTensor(tokens_id).unsqueeze(0).to(self.device))
                .last_hidden_state[0]
                .cpu()
            )

        bos_emb = 0.01 * torch.ones(embeddings.shape[-1])
        eos_emb = -0.01 * torch.ones(embeddings.shape[-1])
        sil_emb = 0.1 * torch.ones(embeddings.shape[-1])

        feat = []
        for s in ds.transcription_text:
            if s == TTSTextProcessor.bos:
                feat.append(bos_emb)
            elif s == TTSTextProcessor.eos:
                feat.append(eos_emb)
            elif s == TTSTextProcessor.sil:
                feat.append(sil_emb)
            else:
                feat.append(embeddings[phoneme_emb_pos[0]])
                phoneme_emb_pos = phoneme_emb_pos[1:]

        assert not phoneme_emb_pos

        ds.xpbert_feat = torch.stack(feat)
        return ds.to_numpy()

    def _apply_tokenizer(
        self,
        symbols: tp.Sequence[tp.Union[str, tp.Tuple[str, ...]]],
        token_ends: tp.Tuple[str, ...],
        eotkn_id: int,
    ):
        seq = []
        ph_emb_pos = []
        for s, end in zip(symbols, token_ends):
            if TTSTextProcessor.sil in s:
                if not seq or seq[-1] != "▁":
                    seq.append("▁")
                else:
                    continue
            elif TTSTextProcessor.bos in s:
                continue
            elif TTSTextProcessor.eos in s:
                continue
            else:
                if isinstance(s, tuple):
                    for i in reversed(range(1, len(s) + 1)):
                        ph = "".join(s[:i])
                        if ph in self._tokenizer.vocab:
                            ph_emb_pos.append(len(seq))
                            seq.append(ph)
                            break
                    else:
                        ph_emb_pos.append(len(seq))
                        seq.append("".join(s))
                else:
                    ph_emb_pos.append(len(seq))
                    seq.append(s)

                if end == eotkn_id:
                    seq.append("▁")

        if seq[0] == "▁":
            seq = seq[1:]
            ph_emb_pos = [i - 1 for i in ph_emb_pos]
            if ph_emb_pos[0] == -1:
                ph_emb_pos = ph_emb_pos[1:]

        if seq[-1] == "▁":
            seq = seq[:-1]
            if ph_emb_pos[-1] == len(seq):
                ph_emb_pos = ph_emb_pos[:-1]

        result = self._tokenizer.encode_plus(seq, is_split_into_words=True)
        return result.encodings[0].ids, ph_emb_pos


class LMProcessor(BaseDSProcessor):
    def __init__(
        self,
        lang: str,
        device: str = "cpu",
        model_name: str = "google-bert/bert-base-multilingual-cased",
        by_transcription: bool = True,
    ):
        super().__init__(device=device)

        self.lang = lang
        self.service_tokens = (
            TTSTextProcessor.pad,
            TTSTextProcessor.bos,
            TTSTextProcessor.eos,
            TTSTextProcessor.sil,
            TTSTextProcessor.unk,
        )

        self._model_name = model_name
        self._by_transcription = by_transcription
        self._tokenizer = None
        self._model = None

        self.logging_params(self.get_config_from_locals())

    def init(self):
        super().init()
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.to(self.device).eval()

    @PipeRegistry.registry(inputs={"sent", "transcription_text"}, outputs={"lm_feat"})
    @lazy_initialization
    def process(self, ds: TextDataSample) -> TextDataSample:
        ds = super().process(ds)

        if self.lang != "MULTILANG" and ds.sent.lang != self.lang:
            raise RuntimeError(
                f"The LMProcessor does not match the sentence {ds.sent.lang} language."
            )

        word_lens = self._count_word_lens(ds.sent)
        embeddings = self._process_lm(ds.sent)
        assert len(embeddings) == len(word_lens)

        if self._by_transcription:
            ds.lm_feat = TTSTextProcessor._assign_tags_to_phoneme(
                list(zip(embeddings, word_lens))
            )
        else:
            ds.lm_feat = embeddings

        ds.lm_feat = torch.stack(ds.lm_feat)
        return ds.to_numpy()

    @staticmethod
    def _count_word_lens(sentence: Sentence) -> tp.List:
        word_lens = []
        for synt in sentence.syntagmas:
            for token in synt.tokens:
                if not token.is_punctuation:
                    word_lens.append(token.num_phonemes)

        return word_lens

    @torch.inference_mode()
    def _process_lm(self, sentence: Sentence):
        tokens = [tk for tk in sentence.tokens if tk.text not in self.service_tokens]

        text = []
        for idx, tk in enumerate(tokens):
            t = tk.text
            if idx == 0 or tk.is_capitalize:
                t = f"{t[:1].upper()}{t[1:]}"

            text.append(t)

        inp = self._tokenizer(
            [text],
            return_tensors="pt",
            max_length=512,
            is_split_into_words=True,
            truncation=True,
            padding=True,
        )
        pred = (
            self._model(
                input_ids=inp["input_ids"].to(self.device),
                attention_mask=inp["attention_mask"].to(self.device),
            )
            .last_hidden_state[0]
            .cpu()
        )

        word_ids = inp.word_ids()
        for i, j in enumerate(word_ids):
            if j is not None:
                tokens[j].meta["embeddings"] = pred[i]

        embeddings = []
        zeros = torch.zeros(pred.shape[1])
        for tk in sentence.tokens:
            if tk.is_punctuation:
                continue
            else:
                if "embeddings" in tk.meta:
                    embeddings.append(tk.meta["embeddings"])
                else:
                    embeddings.append(zeros)

        return embeddings


if __name__ == "__main__":
    from multilingual_text_parser.parser import TextParser

    utterance = """

    Летом Минздрав #России сообщал, что срок действия QR-кода может быть сокращен.

    """

    parser = TextParser(lang="RU")
    with Profiler(format=Profiler.Format.ms):
        doc = parser.process(Doc(utterance))

    PAUSE_SYMB = "<SIL>"
    ph_seq = []
    for sent in doc.sents:
        for synt in sent.syntagmas:
            attr = TokenUtils.get_attr(synt.tokens, ["text", "phonemes"], with_punct=True)
            for idx, wd in enumerate(attr["phonemes"]):
                if wd is not None:
                    ph_seq += list(wd)
                else:
                    ph_seq += [attr["text"][idx]]
            ph_seq += [PAUSE_SYMB]
        print("----")
        print(sent.text)
        print("----")
        print(sent.stress)
        print("----")
        for tk in sent.tokens:
            if tk.modifiers:
                print(tk.text, tk.modifiers)

    print(ph_seq)

    _text_processor = TTSTextProcessor(lang="MULTILANG")
    _ds = TextDataSample(sent=doc.sents[0])
    _ds = _text_processor.process(_ds)
    print(_ds.transcription_id)

    _lm = LMProcessor(lang="RU", device="cpu")
    _ds = _lm.process(_ds)
    print(_ds.lm_feat.shape)
